#!/usr/bin/env python3
"""Match external compact trait JSONL rows back to the 555 NABirds targets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_CLASS_PROMPTS = "reports/milestone2/nabirds_class_prompts.csv"
DEFAULT_EXTERNAL_TRAITS = "reports/milestone3/external_traits/bird_traits.jsonl"
DEFAULT_OUTPUT = "reports/milestone3/external_traits/nabirds_external_trait_matches.csv"
DEFAULT_SUMMARY = "reports/milestone3/external_traits/external_trait_match_summary.json"


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _species_key(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value)
    return " ".join(value.lower().split())


def _variant_text(class_name: str, class_family: str) -> str:
    if class_name == class_family:
        return ""
    variants = re.findall(r"\(([^)]*)\)", class_name)
    if variants:
        return " / ".join(value.strip() for value in variants if value.strip())
    remainder = class_name.replace(class_family, "", 1).strip(" -")
    return remainder


def _load_trait_rows(path: Path) -> Dict[str, Mapping[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing external trait JSONL: {path}")
    by_key: Dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL row") from exc
            base_species = str(payload.get("base_species", "")).strip()
            if not base_species:
                continue
            by_key[_species_key(base_species)] = payload
    return by_key


def _unique_class_rows(class_prompts: Path) -> List[Dict[str, str]]:
    rows = _read_csv(class_prompts, required_columns=("raw_class_id", "target", "class_name", "class_family"))
    by_target: Dict[int, Dict[str, str]] = {}
    for row in rows:
        target = int(row["target"])
        by_target.setdefault(
            target,
            {
                "raw_class_id": row["raw_class_id"],
                "target": row["target"],
                "class_name": row["class_name"],
                "class_family": row["class_family"],
            },
        )
    return [by_target[target] for target in sorted(by_target)]


def _source_metadata(payload: Mapping[str, Any]) -> Dict[str, str]:
    names = []
    urls = []
    licenses = []
    statuses = []
    for source in payload.get("sources", []):
        if not isinstance(source, Mapping):
            continue
        names.append(str(source.get("source", "")))
        statuses.append(f"{source.get('source', '')}:{source.get('status', '')}")
        if source.get("source_url"):
            urls.append(str(source["source_url"]))
        for license_value in source.get("licenses", []):
            value = str(license_value).strip()
            if value:
                licenses.append(value)
    return {
        "source_names": "|".join(sorted(set(names))),
        "source_statuses": "|".join(sorted(set(statuses))),
        "source_urls": "|".join(sorted(set(urls))),
        "licenses": "|".join(sorted(set(licenses))),
    }


def match_traits(args: argparse.Namespace) -> Dict[str, Any]:
    trait_rows = _load_trait_rows(Path(args.external_traits))
    class_rows = _unique_class_rows(Path(args.class_prompts))
    rows: List[Dict[str, Any]] = []
    match_counts = {"matched_with_traits": 0, "matched_without_traits": 0, "missing_base_species": 0}

    for class_row in class_rows:
        key = _species_key(class_row["class_family"])
        payload = trait_rows.get(key)
        if payload is None:
            match_status = "missing_base_species"
            traits = []
            metadata = {"source_names": "", "source_statuses": "", "source_urls": "", "licenses": ""}
        else:
            traits = [str(item.get("trait", "")) for item in payload.get("traits", []) if isinstance(item, Mapping)]
            match_status = "matched_with_traits" if traits else "matched_without_traits"
            metadata = _source_metadata(payload)
        match_counts[match_status] += 1
        rows.append(
            {
                "raw_class_id": class_row["raw_class_id"],
                "target": class_row["target"],
                "class_name": class_row["class_name"],
                "class_family": class_row["class_family"],
                "variant": _variant_text(class_row["class_name"], class_row["class_family"]),
                "match_status": match_status,
                "external_trait_count": len(traits),
                "external_traits": json.dumps(traits, sort_keys=True),
                **metadata,
            }
        )

    fieldnames = [
        "raw_class_id",
        "target",
        "class_name",
        "class_family",
        "variant",
        "match_status",
        "external_trait_count",
        "external_traits",
        "source_names",
        "source_statuses",
        "source_urls",
        "licenses",
    ]
    output_path = Path(args.output)
    _write_csv(output_path, rows, fieldnames)
    summary = {
        "class_prompts": args.class_prompts,
        "external_traits": args.external_traits,
        "output": str(output_path),
        "num_classes": len(rows),
        "match_counts": match_counts,
        "unique_base_species_with_external_rows": len(trait_rows),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match external compact visual traits to NABirds target classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--class-prompts", default=DEFAULT_CLASS_PROMPTS)
    parser.add_argument("--external-traits", default=DEFAULT_EXTERNAL_TRAITS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    print(json.dumps(match_traits(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
