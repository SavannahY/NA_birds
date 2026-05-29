#!/usr/bin/env python3
"""Fetch copyright-safe compact visual bird traits from public APIs.

This crawler is intentionally conservative: it stores compact extracted trait
phrases plus source/license metadata, not raw field-guide paragraphs. Raw API
responses are cached under reports/.cache by default and should not be
committed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import re
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_CLASS_PROMPTS = "reports/milestone2/nabirds_class_prompts.csv"
DEFAULT_OUTPUT = "reports/milestone3/external_traits/bird_traits.jsonl"
DEFAULT_SUMMARY = "reports/milestone3/external_traits/external_trait_crawl_summary.json"
DEFAULT_CACHE_DIR = "reports/.cache/external_traits"
DEFAULT_GBIF_MIN_CONFIDENCE = 90

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GBIF_SEARCH_URL = "https://api.gbif.org/v1/species/search"
GBIF_DESCRIPTIONS_URL = "https://api.gbif.org/v1/species/{usage_key}/descriptions"
GBIF_SPECIES_PAGE = "https://www.gbif.org/species/{usage_key}"

EOL_SEARCH_URL = "https://eol.org/api/search/1.0.json"
EOL_PAGE_URL = "https://eol.org/api/pages/1.0/{page_id}.json"
EOL_SPECIES_PAGE = "https://eol.org/pages/{page_id}"

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_URL = "https://en.wikipedia.org/wiki/{title}"

USER_AGENT = "NA-birds-class-project/0.1 (+https://github.com/SavannahY/NA_birds)"

COLOR_WORDS = {
    "black",
    "blackish",
    "blue",
    "bluish",
    "brown",
    "brownish",
    "buff",
    "chestnut",
    "cinnamon",
    "dark",
    "gray",
    "grayish",
    "green",
    "greenish",
    "orange",
    "pale",
    "pink",
    "purple",
    "red",
    "reddish",
    "rufous",
    "rusty",
    "tan",
    "white",
    "whitish",
    "yellow",
    "yellowish",
}

MARK_WORDS = {
    "back",
    "belly",
    "bill",
    "breast",
    "cap",
    "cheek",
    "crown",
    "eye",
    "face",
    "flank",
    "flanks",
    "head",
    "hood",
    "neck",
    "nape",
    "plumage",
    "rump",
    "tail",
    "throat",
    "underparts",
    "upperparts",
    "wing",
    "wingbar",
    "wingbars",
    "wings",
}

SIZE_WORDS = {"large", "medium-sized", "small", "stocky"}
TYPE_WORDS = {
    "bird",
    "duck",
    "finch",
    "gull",
    "hawk",
    "hummingbird",
    "raptor",
    "sandpiper",
    "sparrow",
    "warbler",
    "woodpecker",
    "wren",
}

PATTERN_TRAITS = (
    re.compile(
        r"\b("
        r"black|blackish|blue|bluish|brown|brownish|buff|chestnut|cinnamon|dark|"
        r"gray|grayish|green|greenish|orange|pale|pink|purple|red|reddish|rufous|"
        r"rusty|tan|white|whitish|yellow|yellowish"
        r")(?:[- ](?:and|or)[- ](?:black|blue|brown|gray|green|orange|red|white|yellow))?"
        r"[- ]+("
        r"back|belly|bill|breast|cap|cheek|crown|eye|face|flank|flanks|head|hood|"
        r"neck|nape|plumage|rump|tail|throat|underparts|upperparts|wingbars?|wings?"
        r")\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b("
        r"barred|bold|contrasting|crested|forked|large|long|long-tailed|long-winged|"
        r"medium-sized|rounded|short|slender|small|spotted|stocky|streaked|striped|thick|thin"
        r")[- ]+("
        r"bill|bird|crest|duck|finch|gull|hawk|hummingbird|neck|raptor|sandpiper|sparrow|tail|"
        r"warbler|wings?|wingbars?|woodpecker|wren|body|head|legs|plumage|underparts|upperparts"
        r")\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b("
        r"wingbars?|eye[- ]rings?|eye[- ]lines?|supercilium|mask|bib|collar|"
        r"streaking|spots?|patch(?:es)?|stripe(?:s)?"
        r")\b",
        re.IGNORECASE,
    ),
)


def _read_csv(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} is empty or missing a header row.")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} missing required column(s): {', '.join(missing)}")
        return list(reader)


def _unique_species(class_prompts: Path) -> List[Dict[str, str]]:
    rows = _read_csv(class_prompts, required_columns=("raw_class_id", "target", "class_name", "class_family"))
    by_family: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        family = row["class_family"].strip()
        if not family:
            continue
        entry = by_family.setdefault(
            family,
            {
                "base_species": family,
                "targets": set(),
                "raw_class_ids": set(),
                "class_names": set(),
            },
        )
        entry["targets"].add(int(row["target"]))
        entry["raw_class_ids"].add(int(row["raw_class_id"]))
        entry["class_names"].add(row["class_name"])
    species = []
    for family in sorted(by_family):
        entry = by_family[family]
        species.append(
            {
                "base_species": family,
                "targets": sorted(entry["targets"]),
                "raw_class_ids": sorted(entry["raw_class_ids"]),
                "class_names": sorted(entry["class_names"]),
            }
        )
    return species


def _cache_key(url: str, params: Mapping[str, Any]) -> str:
    encoded = urlencode(sorted((key, str(value)) for key, value in params.items()))
    return hashlib.sha256(f"{url}?{encoded}".encode("utf-8")).hexdigest()


def _request_json(
    url: str,
    params: Mapping[str, Any],
    *,
    cache_dir: Path,
    force: bool,
    offline: bool,
    timeout: float,
    verify_tls: bool,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(url, params)
    cache_path = cache_dir / f"{key}.json"
    full_url = f"{url}?{urlencode(params)}" if params else url

    if cache_path.is_file() and not force:
        payload = cache_path.read_text(encoding="utf-8")
        return json.loads(payload), {
            "url": full_url,
            "cache_path": str(cache_path),
            "from_cache": True,
            "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
    if offline:
        return None, {"url": full_url, "cache_path": str(cache_path), "error": "cache_miss_offline"}

    request = Request(full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    context = None if verify_tls else ssl._create_unverified_context()
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return None, {"url": full_url, "cache_path": str(cache_path), "error": str(exc)}
    cache_path.write_text(payload, encoding="utf-8")
    return json.loads(payload), {
        "url": full_url,
        "cache_path": str(cache_path),
        "from_cache": False,
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _strip_markup(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _compact_trait(value: str) -> str:
    value = value.lower().replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,;:-")


def _trait_score(trait: str) -> int:
    tokens = {token for token in re.split(r"[^a-z-]+", trait.lower()) if token}
    score = 0
    if tokens & COLOR_WORDS:
        score += 3
    if tokens & MARK_WORDS:
        score += 2
    if any(token in tokens for token in ("barred", "spotted", "streaked", "striped", "wingbar", "wingbars")):
        score += 2
    if "plumage" in tokens:
        score += 1
    if tokens & SIZE_WORDS:
        score += 2
    if tokens & TYPE_WORDS:
        score += 1
    return score


def extract_visual_traits(texts: Iterable[str], max_traits: int) -> List[Dict[str, Any]]:
    seen: Dict[str, int] = {}
    for text in texts:
        clean_text = _strip_markup(text)
        if not clean_text:
            continue
        chunks = re.split(r"(?<=[.!?;])\s+|\n+", clean_text)
        for chunk in chunks:
            if len(chunk) > 600:
                continue
            for pattern in PATTERN_TRAITS:
                for match in pattern.finditer(chunk):
                    trait = _compact_trait(match.group(0))
                    if len(trait.split()) > 5:
                        continue
                    score = _trait_score(trait)
                    if score <= 0:
                        continue
                    seen[trait] = max(score, seen.get(trait, 0))

    ranked = sorted(seen.items(), key=lambda item: (-item[1], item[0]))
    return [{"trait": trait, "score": score} for trait, score in ranked[:max_traits]]


def _description_entries(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, Mapping)]
    if isinstance(payload, Mapping):
        results = payload.get("results")
        if isinstance(results, list):
            return [entry for entry in results if isinstance(entry, Mapping)]
        descriptions = payload.get("descriptions")
        if isinstance(descriptions, list):
            return [entry for entry in descriptions if isinstance(entry, Mapping)]
    return []


def _gbif_candidate_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "key": candidate.get("key"),
        "nub_key": candidate.get("nubKey"),
        "species_key": candidate.get("speciesKey"),
        "scientific_name": candidate.get("scientificName"),
        "canonical_name": candidate.get("canonicalName"),
        "rank": candidate.get("rank"),
        "taxonomic_status": candidate.get("taxonomicStatus") or candidate.get("status"),
        "class": candidate.get("class"),
        "vernacular_names": [
            item.get("vernacularName")
            for item in candidate.get("vernacularNames", [])
            if isinstance(item, Mapping) and item.get("vernacularName")
        ][:5],
    }


def _best_gbif_search_candidate(search_payload: Any, species_name: str) -> Tuple[Optional[Mapping[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(search_payload, Mapping):
        return None, []
    results = search_payload.get("results")
    if not isinstance(results, list):
        return None, []

    lowered = species_name.casefold()
    summaries = []
    fallback = None
    for result in results[:10]:
        if not isinstance(result, Mapping):
            continue
        summaries.append(_gbif_candidate_summary(result))
        rank = str(result.get("rank") or "").upper()
        taxonomic_status = str(result.get("taxonomicStatus") or result.get("status") or "").upper()
        bird_class = str(result.get("class") or "").upper()
        if rank != "SPECIES" or taxonomic_status != "ACCEPTED" or bird_class != "AVES":
            continue
        vernacular_names = result.get("vernacularNames", [])
        exact_common_name = any(
            isinstance(item, Mapping) and str(item.get("vernacularName") or "").casefold() == lowered
            for item in vernacular_names
        )
        if exact_common_name:
            return result, summaries
        if fallback is None:
            fallback = result
    return fallback, summaries


def fetch_gbif_traits(
    species_name: str,
    *,
    cache_dir: Path,
    force: bool,
    offline: bool,
    timeout: float,
    max_traits: int,
    min_confidence: int,
    allow_fuzzy: bool,
    verify_tls: bool,
) -> Dict[str, Any]:
    match_payload, match_meta = _request_json(
        GBIF_MATCH_URL,
        {"name": species_name, "rank": "SPECIES", "verbose": "false"},
        cache_dir=cache_dir / "gbif",
        force=force,
        offline=offline,
        timeout=timeout,
        verify_tls=verify_tls,
    )
    if match_payload is None and match_meta.get("error"):
        return {"source": "gbif", "status": "request_error", "request": match_meta, "traits": []}
    search_meta: Dict[str, Any] = {}
    search_candidates: List[Dict[str, Any]] = []
    match_source = "species_match"
    if not isinstance(match_payload, Mapping) or not match_payload.get("usageKey"):
        search_payload, search_meta = _request_json(
            GBIF_SEARCH_URL,
            {"q": species_name, "rank": "SPECIES", "limit": "10"},
            cache_dir=cache_dir / "gbif",
            force=force,
            offline=offline,
            timeout=timeout,
            verify_tls=verify_tls,
        )
        if search_payload is None and search_meta.get("error"):
            return {
                "source": "gbif",
                "status": "request_error",
                "request": {"match": match_meta, "search": search_meta},
                "traits": [],
            }
        search_candidate, search_candidates = _best_gbif_search_candidate(search_payload, species_name)
        if not search_candidate:
            return {
                "source": "gbif",
                "status": "no_match",
                "request": {"match": match_meta, "search": search_meta},
                "candidates": search_candidates,
                "traits": [],
            }
        match_payload = search_candidate
        match_source = "species_search_vernacular"

    usage_key_value = match_payload.get("usageKey") or match_payload.get("nubKey") or match_payload.get("speciesKey")
    usage_key_value = usage_key_value or match_payload.get("key")
    usage_key = str(usage_key_value)
    confidence = int(match_payload.get("confidence") or 0)
    match_type = str(match_payload.get("matchType") or ("VERNACULAR_EXACT" if match_source != "species_match" else "")).upper()
    rank = str(match_payload.get("rank") or "").upper()
    taxonomic_status = str(match_payload.get("status") or match_payload.get("taxonomicStatus") or "").upper()
    allowed_match_types = {"EXACT", "VERNACULAR_EXACT"}
    if allow_fuzzy:
        allowed_match_types.add("FUZZY")
    rejection_reasons = []
    if match_type and match_type not in allowed_match_types:
        rejection_reasons.append(f"match_type={match_type}")
    if match_source == "species_match" and confidence < min_confidence:
        rejection_reasons.append(f"confidence={confidence}")
    if rank and rank != "SPECIES":
        rejection_reasons.append(f"rank={rank}")
    if taxonomic_status and taxonomic_status != "ACCEPTED":
        rejection_reasons.append(f"status={taxonomic_status}")
    if rejection_reasons:
        return {
            "source": "gbif",
            "status": "rejected_match",
            "rejection_reasons": rejection_reasons,
            "matched_name": match_payload.get("scientificName") or match_payload.get("canonicalName"),
            "usage_key": usage_key,
            "confidence": confidence,
            "match_type": match_type,
            "match_source": match_source,
            "rank": rank,
            "taxonomic_status": taxonomic_status,
            "source_url": GBIF_SPECIES_PAGE.format(usage_key=usage_key),
            "request": {"match": match_meta, "search": search_meta} if search_meta else match_meta,
            "candidates": search_candidates,
            "traits": [],
        }

    descriptions_payload, desc_meta = _request_json(
        GBIF_DESCRIPTIONS_URL.format(usage_key=usage_key),
        {},
        cache_dir=cache_dir / "gbif",
        force=force,
        offline=offline,
        timeout=timeout,
        verify_tls=verify_tls,
    )
    if descriptions_payload is None and desc_meta.get("error"):
        return {
            "source": "gbif",
            "status": "request_error",
            "matched_name": match_payload.get("scientificName") or match_payload.get("canonicalName"),
            "usage_key": usage_key,
            "confidence": confidence,
            "match_type": match_type,
            "match_source": match_source,
            "rank": rank,
            "taxonomic_status": taxonomic_status,
            "source_url": GBIF_SPECIES_PAGE.format(usage_key=usage_key),
            "request": {"match": match_meta, "search": search_meta, "descriptions": desc_meta},
            "candidates": search_candidates,
            "traits": [],
        }
    entries = _description_entries(descriptions_payload)
    search_descriptions = _description_entries(match_payload.get("descriptions") if isinstance(match_payload, Mapping) else None)
    texts = [str(entry.get("description") or entry.get("value") or entry.get("text") or "") for entry in entries]
    texts.extend(
        str(entry.get("description") or entry.get("value") or entry.get("text") or "")
        for entry in search_descriptions
    )
    traits = extract_visual_traits(texts, max_traits=max_traits)
    licenses = sorted(
        {
            str(entry.get("license") or entry.get("rights") or "").strip()
            for entry in entries
            if str(entry.get("license") or entry.get("rights") or "").strip()
        }
    )
    return {
        "source": "gbif",
        "status": "ok" if traits else "no_visual_traits",
        "matched_name": match_payload.get("scientificName") or match_payload.get("canonicalName"),
        "usage_key": usage_key,
        "confidence": confidence,
        "match_type": match_type,
        "match_source": match_source,
        "rank": rank,
        "taxonomic_status": taxonomic_status,
        "source_url": GBIF_SPECIES_PAGE.format(usage_key=usage_key),
        "request": {"match": match_meta, "search": search_meta, "descriptions": desc_meta},
        "candidates": search_candidates,
        "description_count": len(entries) + len(search_descriptions),
        "licenses": licenses,
        "traits": traits,
    }


def _eol_candidates(search_payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(search_payload, Mapping):
        return []
    results = search_payload.get("results")
    if not isinstance(results, list) or not results:
        return []
    candidates = []
    for result in results[:5]:
        if isinstance(result, Mapping):
            candidates.append(
                {
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "link": result.get("link"),
                }
            )
    return candidates


def _best_eol_page_match(search_payload: Any, species_name: str) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
    candidates = _eol_candidates(search_payload)
    if not candidates:
        return None, "no_match", []
    lowered = species_name.casefold()
    for result in candidates:
        if str(result.get("title", "")).casefold() == lowered and result.get("id"):
            return str(result["id"]), "exact_title", candidates
    for result in candidates:
        if result.get("id"):
            return str(result["id"]), "ambiguous_match", candidates
    return None, "no_match", candidates


def fetch_eol_traits(
    species_name: str,
    *,
    cache_dir: Path,
    force: bool,
    offline: bool,
    timeout: float,
    max_traits: int,
    max_text_objects: int,
    allow_ambiguous: bool,
    verify_tls: bool,
) -> Dict[str, Any]:
    search_payload, search_meta = _request_json(
        EOL_SEARCH_URL,
        {"q": species_name, "exact": "true", "page": "1"},
        cache_dir=cache_dir / "eol",
        force=force,
        offline=offline,
        timeout=timeout,
        verify_tls=verify_tls,
    )
    if search_payload is None and search_meta.get("error"):
        return {"source": "eol", "status": "request_error", "request": {"search": search_meta}, "traits": []}
    page_id, match_status, candidates = _best_eol_page_match(search_payload, species_name)
    if not page_id:
        return {
            "source": "eol",
            "status": "no_match",
            "request": {"search": search_meta},
            "candidates": candidates,
            "traits": [],
        }
    if match_status == "ambiguous_match" and not allow_ambiguous:
        return {
            "source": "eol",
            "status": "ambiguous_match",
            "request": {"search": search_meta},
            "candidates": candidates,
            "source_url": EOL_SPECIES_PAGE.format(page_id=page_id),
            "traits": [],
        }

    page_payload, page_meta = _request_json(
        EOL_PAGE_URL.format(page_id=page_id),
        {"details": "true", "images": "0", "videos": "0", "sounds": "0", "maps": "0", "texts": str(max_text_objects)},
        cache_dir=cache_dir / "eol",
        force=force,
        offline=offline,
        timeout=timeout,
        verify_tls=verify_tls,
    )
    if page_payload is None and page_meta.get("error"):
        return {
            "source": "eol",
            "status": "request_error",
            "request": {"search": search_meta, "page": page_meta},
            "candidates": candidates,
            "source_url": EOL_SPECIES_PAGE.format(page_id=page_id),
            "traits": [],
        }
    data_objects = []
    if isinstance(page_payload, Mapping) and isinstance(page_payload.get("dataObjects"), list):
        data_objects = [entry for entry in page_payload["dataObjects"] if isinstance(entry, Mapping)]
    texts = [str(entry.get("description") or "") for entry in data_objects]
    traits = extract_visual_traits(texts, max_traits=max_traits)
    licenses = sorted(
        {
            str(entry.get("license") or "").strip()
            for entry in data_objects
            if str(entry.get("license") or "").strip()
        }
    )
    return {
        "source": "eol",
        "status": "ok" if traits else "no_visual_traits",
        "matched_name": page_payload.get("scientificName") if isinstance(page_payload, Mapping) else None,
        "page_id": page_id,
        "match_status": match_status,
        "source_url": EOL_SPECIES_PAGE.format(page_id=page_id),
        "request": {"search": search_meta, "page": page_meta},
        "description_count": len(data_objects),
        "licenses": licenses,
        "traits": traits,
    }


def _wikipedia_page(payload: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return None
    query = payload.get("query")
    if not isinstance(query, Mapping):
        return None
    pages = query.get("pages")
    if not isinstance(pages, Mapping):
        return None
    for page in pages.values():
        if isinstance(page, Mapping) and "missing" not in page:
            return page
    return None


def fetch_wikipedia_traits(
    species_name: str,
    *,
    cache_dir: Path,
    force: bool,
    offline: bool,
    timeout: float,
    max_traits: int,
    verify_tls: bool,
) -> Dict[str, Any]:
    payload, meta = _request_json(
        WIKIPEDIA_API_URL,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "titles": species_name,
        },
        cache_dir=cache_dir / "wikipedia",
        force=force,
        offline=offline,
        timeout=timeout,
        verify_tls=verify_tls,
    )
    if payload is None and meta.get("error"):
        return {"source": "wikipedia", "status": "request_error", "request": meta, "traits": []}
    page = _wikipedia_page(payload)
    if page is None:
        return {"source": "wikipedia", "status": "no_match", "request": meta, "traits": []}

    title = str(page.get("title") or species_name)
    extract = str(page.get("extract") or "")
    traits = extract_visual_traits([extract], max_traits=max_traits)
    return {
        "source": "wikipedia",
        "status": "ok" if traits else "no_visual_traits",
        "matched_name": title,
        "page_id": page.get("pageid"),
        "source_url": WIKIPEDIA_PAGE_URL.format(title=title.replace(" ", "_")),
        "request": meta,
        "description_count": 1 if extract else 0,
        "licenses": ["CC BY-SA (Wikipedia text; verify page footer)"],
        "traits": traits,
    }


def _merge_traits(source_results: Sequence[Mapping[str, Any]], max_traits: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for result in source_results:
        source = str(result.get("source", "external"))
        source_url = str(result.get("source_url", ""))
        for item in result.get("traits", []):
            if not isinstance(item, Mapping):
                continue
            trait = str(item.get("trait", "")).strip()
            if not trait:
                continue
            entry = merged.setdefault(
                trait,
                {
                    "trait": trait,
                    "score": 0,
                    "sources": [],
                },
            )
            entry["score"] = max(int(entry["score"]), int(item.get("score", 0)))
            source_ref = {"source": source, "url": source_url}
            if source_ref not in entry["sources"]:
                entry["sources"].append(source_ref)
    ranked = sorted(merged.values(), key=lambda item: (-int(item["score"]), str(item["trait"])))
    return ranked[:max_traits]


def crawl(args: argparse.Namespace) -> Dict[str, Any]:
    species_rows = _unique_species(Path(args.class_prompts))
    if args.limit:
        species_rows = species_rows[: args.limit]
    if args.dry_run:
        return {
            "class_prompts": args.class_prompts,
            "species_count": len(species_rows),
            "preview": [row["base_species"] for row in species_rows[:10]],
            "dry_run": True,
        }

    sources = {source.strip().lower() for source in args.sources.split(",") if source.strip()}
    if "wiki" in sources:
        sources.remove("wiki")
        sources.add("wikipedia")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    written = 0
    source_status_counts: Dict[str, int] = {}
    crawl_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    command = [Path(sys.argv[0]).name, *sys.argv[1:]]
    source_config = {
        "sources": sorted(sources),
        "gbif_min_confidence": args.gbif_min_confidence,
        "gbif_allow_fuzzy": args.allow_gbif_fuzzy,
        "eol_allow_ambiguous": args.allow_eol_ambiguous,
        "max_traits": args.max_traits,
        "max_traits_per_source": args.max_traits_per_source,
        "eol_texts": args.eol_texts,
        "verify_tls": not args.insecure_no_verify,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(species_rows, start=1):
            source_results: List[Dict[str, Any]] = []
            if "gbif" in sources:
                source_results.append(
                    fetch_gbif_traits(
                        row["base_species"],
                        cache_dir=cache_dir,
                        force=args.force,
                        offline=args.offline,
                        timeout=args.timeout,
                        max_traits=args.max_traits_per_source,
                        min_confidence=args.gbif_min_confidence,
                        allow_fuzzy=args.allow_gbif_fuzzy,
                        verify_tls=not args.insecure_no_verify,
                    )
                )
                time.sleep(args.sleep)
            if "wikipedia" in sources:
                source_results.append(
                    fetch_wikipedia_traits(
                        row["base_species"],
                        cache_dir=cache_dir,
                        force=args.force,
                        offline=args.offline,
                        timeout=args.timeout,
                        max_traits=args.max_traits_per_source,
                        verify_tls=not args.insecure_no_verify,
                    )
                )
                time.sleep(args.sleep)
            if "eol" in sources:
                source_results.append(
                    fetch_eol_traits(
                        row["base_species"],
                        cache_dir=cache_dir,
                        force=args.force,
                        offline=args.offline,
                        timeout=args.timeout,
                        max_traits=args.max_traits_per_source,
                        max_text_objects=args.eol_texts,
                        allow_ambiguous=args.allow_eol_ambiguous,
                        verify_tls=not args.insecure_no_verify,
                    )
                )
                time.sleep(args.sleep)

            for result in source_results:
                key = f"{result.get('source')}:{result.get('status')}"
                source_status_counts[key] = source_status_counts.get(key, 0) + 1

            payload = {
                "base_species": row["base_species"],
                "targets": row["targets"],
                "raw_class_ids": row["raw_class_ids"],
                "class_names": row["class_names"],
                "traits": _merge_traits(source_results, args.max_traits),
                "sources": source_results,
                "provenance": {
                    "crawl_timestamp_utc": crawl_timestamp,
                    "command": command,
                    "source_config": source_config,
                    "raw_cache_note": "Raw API JSON is cached locally for reproducibility; do not commit cache files.",
                },
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            written += 1
            if args.verbose:
                trait_count = len(payload["traits"])
                print(f"[{index}/{len(species_rows)}] {row['base_species']}: {trait_count} trait(s)", file=sys.stderr)

    return {
        "class_prompts": args.class_prompts,
        "output": str(output_path),
        "cache_dir": str(cache_dir),
        "species_count": len(species_rows),
        "rows_written": written,
        "source_config": source_config,
        "source_status_counts": source_status_counts,
        "raw_cache_note": "Raw API JSON is cached locally for reproducibility; do not commit cache files.",
        "crawl_timestamp_utc": crawl_timestamp,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch external compact bird visual traits for NABirds prompt experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--class-prompts", default=DEFAULT_CLASS_PROMPTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sources", default="gbif,wikipedia", help="Comma-separated subset: gbif,wikipedia,eol")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of base species for smoke tests.")
    parser.add_argument("--max-traits", type=int, default=12)
    parser.add_argument("--max-traits-per-source", type=int, default=12)
    parser.add_argument("--eol-texts", type=int, default=25)
    parser.add_argument("--gbif-min-confidence", type=int, default=DEFAULT_GBIF_MIN_CONFIDENCE)
    parser.add_argument("--allow-gbif-fuzzy", action="store_true")
    parser.add_argument("--allow-eol-ambiguous", action="store_true")
    parser.add_argument(
        "--insecure-no-verify",
        action="store_true",
        help="Disable TLS certificate verification for local smoke tests with broken cert stores.",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep after each API request group.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--force", action="store_true", help="Refetch even when cached API JSON exists.")
    parser.add_argument("--offline", action="store_true", help="Use cache only; missing cache entries become source errors.")
    parser.add_argument("--dry-run", action="store_true", help="List species that would be crawled without network access.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = crawl(args)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
