#!/usr/bin/env python3
"""Build NABirds prompt CSVs from matched external compact trait rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_MATCHES = "reports/milestone3/external_traits/nabirds_external_trait_matches.csv"
DEFAULT_OUTPUT = "reports/milestone3/external_traits/nabirds_external_trait_prompts.csv"
EXPECTED_NABIRDS_CLASSES = 555


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


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _squash_spaces(value: str) -> str:
    return " ".join(value.strip().split())


def _article_for(label: str) -> str:
    return "an" if label and label[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _clean_variant_text(value: str) -> str:
    value = value.replace("/", " or ").replace("-", " ")
    return _squash_spaces(value).strip(" .")


def _split_class_name(class_name: str) -> Tuple[str, str, str, str]:
    variants = [_clean_variant_text(match) for match in re.findall(r"\(([^)]*)\)", class_name)]
    variants = [value for value in variants if value]
    base = _squash_spaces(re.sub(r"\s*\([^)]*\)", "", class_name))
    variant = _squash_spaces(" ".join(variants))
    label = _squash_spaces(f"{base} {variant}") if variant else base
    variant_first_label = _squash_spaces(f"{variant.lower()} {base}") if variant else base
    return base, variant, label, variant_first_label


def _variant_trait(variant: str) -> str:
    if not variant:
        return ""
    value = variant.lower()
    if "morph" in value or "form" in value:
        return value
    if any(token in value for token in ("male", "female", "juvenile", "immature", "adult", "winter", "breeding")):
        return f"{value} plumage"
    return value


def _trait_phrase(items: Sequence[str], max_items: int) -> str:
    kept = [item for item in items[:max_items] if item]
    if not kept:
        return "distinct field marks"
    if len(kept) == 1:
        return kept[0]
    return ", ".join(kept[:-1]) + ", and " + kept[-1]


def _parse_traits(raw_value: str) -> List[str]:
    try:
        parsed = json.loads(raw_value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    traits = []
    seen = set()
    for value in parsed:
        trait = _squash_spaces(str(value).strip(" .;:"))
        key = trait.casefold()
        if trait and key not in seen:
            traits.append(trait)
            seen.add(key)
    return traits


def _prompt_rows_for_match(row: Mapping[str, str], max_traits: int, fallback_short_variant: bool) -> List[Dict[str, Any]]:
    class_name = row["class_name"]
    base, variant, label, variant_first_label = _split_class_name(class_name)
    external_traits = _parse_traits(row.get("external_traits", "[]"))
    traits = list(external_traits)
    variant_field_mark = _variant_trait(variant)
    if variant_field_mark and variant_field_mark.casefold() not in {trait.casefold() for trait in traits}:
        traits = [variant_field_mark] + traits
    traits = traits[:max_traits]
    trait_text = _trait_phrase(traits, max_traits)
    article = _article_for(variant_first_label)
    prompt_set = "short_variant_plus_external_traits" if external_traits else "external_traits_fallback"

    if not external_traits and fallback_short_variant:
        templates = (
            "{label}",
            "a photo of {article} {variant_first_label}.",
            "a field guide photo of {article} {variant_first_label}.",
        )
    elif not external_traits:
        return []
    else:
        templates = (
            "{label}",
            "a field guide photo of {article} {variant_first_label}.",
            "a field guide photo of {article} {variant_first_label}, showing {traits}.",
        )

    context = {
        "label": label,
        "article": article,
        "variant_first_label": variant_first_label,
        "traits": trait_text,
    }
    output = []
    for prompt_id, template in enumerate(templates):
        output.append(
            {
                "raw_class_id": row["raw_class_id"],
                "target": row["target"],
                "class_name": class_name,
                "class_family": row["class_family"],
                "prompt_id": prompt_id,
                "prompt": template.format(**context),
                "prompt_set": prompt_set,
                "variant": variant,
                "external_trait_count": len(external_traits),
                "match_status": row.get("match_status", ""),
                "source_names": row.get("source_names", ""),
                "source_urls": row.get("source_urls", ""),
                "licenses": row.get("licenses", ""),
            }
        )
    return output


def build_prompts(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _read_csv(
        Path(args.matches),
        required_columns=("raw_class_id", "target", "class_name", "class_family", "external_traits"),
    )
    prompt_rows: List[Dict[str, Any]] = []
    for row in rows:
        prompt_rows.extend(_prompt_rows_for_match(row, args.max_traits, args.fallback_short_variant))

    targets = sorted({int(row["target"]) for row in prompt_rows})
    if args.expected_num_classes and len(targets) != args.expected_num_classes:
        raise SystemExit(
            f"Expected {args.expected_num_classes} unique targets, found {len(targets)}. "
            "Pass --expected-num-classes 0 to disable."
        )
    fieldnames = [
        "raw_class_id",
        "target",
        "class_name",
        "class_family",
        "prompt_id",
        "prompt",
        "prompt_set",
        "variant",
        "external_trait_count",
        "match_status",
        "source_names",
        "source_urls",
        "licenses",
    ]
    output_path = Path(args.output)
    row_count = _write_csv(output_path, prompt_rows, fieldnames)
    summary = {
        "matches": args.matches,
        "output": str(output_path),
        "prompt_rows": row_count,
        "unique_targets": len(targets),
        "classes_with_external_traits": sum(1 for row in rows if int(row.get("external_trait_count", "0") or 0) > 0),
        "max_traits": args.max_traits,
    }
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VLM prompt CSV from matched external bird visual traits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--matches", default=DEFAULT_MATCHES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default="reports/milestone3/external_traits/external_trait_prompt_summary.json")
    parser.add_argument("--max-traits", type=int, default=5)
    parser.add_argument("--expected-num-classes", type=int, default=EXPECTED_NABIRDS_CLASSES)
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument("--fallback-short-variant", dest="fallback_short_variant", action="store_true", default=True)
    fallback_group.add_argument("--no-fallback-short-variant", dest="fallback_short_variant", action="store_false")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    print(json.dumps(build_prompts(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
