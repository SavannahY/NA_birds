#!/usr/bin/env python3
"""Build descriptor-augmented NABirds prompts for Milestone 3 VLM eval.

The output is intentionally compatible with scripts/eval_vlm_zero_shot.py:
one or more prompt rows per class, keyed by the existing compact 0..554
``target`` IDs from the NABirds manifests.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_CLASS_PROMPTS = "reports/milestone2/nabirds_class_prompts.csv"
DEFAULT_DESCRIPTORS = "bird_class_expert_description_visual_attribute_descriptors.csv"
DEFAULT_OUT_DIR = "reports/milestone3"

PROMPT_TEMPLATES = (
    "a field guide photograph of a {class_name}; visual field marks: {traits}.",
    "a close-up bird photo of a {class_name} showing {traits}.",
    "a fine-grained ID photo of a {class_name}, with {traits}.",
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


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _parse_descriptor_items(raw_value: str) -> List[str]:
    raw_value = (raw_value or "").strip()
    if not raw_value or raw_value == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []

    items: List[str] = []
    seen = set()
    for value in parsed:
        item = " ".join(str(value).strip().split())
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def _trait_phrase(items: Sequence[str], max_items: int) -> str:
    kept = list(items[:max_items])
    if not kept:
        return "the visible plumage, shape, bill, wing, tail, and head pattern"
    if len(kept) == 1:
        return kept[0]
    return ", ".join(kept[:-1]) + ", and " + kept[-1]


def _load_descriptor_rows(path: Path) -> Dict[int, Dict[str, object]]:
    rows = _read_csv(
        path,
        required_columns=(
            "class_id",
            "class_label",
            "label_type",
            "coverage_status",
            "expert_description_status",
            "expert_visual_description",
        ),
    )
    descriptors: Dict[int, Dict[str, object]] = {}
    for row in rows:
        class_id = int(row["class_id"])
        descriptors[class_id] = {
            "class_label": row["class_label"],
            "label_type": row["label_type"],
            "coverage_status": row["coverage_status"],
            "expert_description_status": row["expert_description_status"],
            "items": _parse_descriptor_items(row["expert_visual_description"]),
        }
    return descriptors


def _unique_class_rows(prompt_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    by_target: Dict[int, Dict[str, str]] = {}
    for row in prompt_rows:
        target = int(row["target"])
        by_target.setdefault(
            target,
            {
                "raw_class_id": row["raw_class_id"],
                "target": row["target"],
                "class_name": row["class_name"],
                "class_family": row.get("class_family", row["class_name"]),
            },
        )
    return [by_target[target] for target in sorted(by_target)]


def _quality_label(
    class_row: Mapping[str, str],
    descriptor: Mapping[str, object] | None,
    descriptor_signature_counts: Mapping[str, Counter[str]],
) -> str:
    if descriptor is None:
        return "missing_descriptor_row"
    items = descriptor.get("items", [])
    if not items:
        return "missing_descriptor"

    flags: List[str] = []
    status = str(descriptor.get("expert_description_status", ""))
    if "needs_manual_mapping" in status:
        flags.append("needs_manual_mapping")
    elif "needs_normalization" in status:
        flags.append("needs_normalization")
    else:
        flags.append("compact_verified")

    family = class_row["class_family"]
    signature = "|".join(str(item).casefold() for item in items)
    if descriptor_signature_counts[family][signature] > 1:
        flags.append("shared_variant_descriptor")
    return "|".join(flags)


def build_descriptor_prompts(
    *,
    class_prompts: Path,
    descriptors_csv: Path,
    out_dir: Path,
    max_descriptors: int,
) -> Dict[str, object]:
    prompt_rows = _read_csv(
        class_prompts,
        required_columns=("raw_class_id", "target", "class_name", "class_family", "prompt"),
    )
    class_rows = _unique_class_rows(prompt_rows)
    descriptors = _load_descriptor_rows(descriptors_csv)

    descriptor_signature_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in class_rows:
        descriptor = descriptors.get(int(row["raw_class_id"]))
        if descriptor is None:
            continue
        items = descriptor.get("items", [])
        signature = "|".join(str(item).casefold() for item in items)
        descriptor_signature_counts[row["class_family"]][signature] += 1

    output_rows: List[Dict[str, object]] = []
    missing_raw_ids: List[int] = []
    descriptor_counts: List[int] = []
    quality_counts: Counter[str] = Counter()
    shared_variant_classes = 0

    for row in class_rows:
        raw_class_id = int(row["raw_class_id"])
        descriptor = descriptors.get(raw_class_id)
        if descriptor is None:
            missing_raw_ids.append(raw_class_id)
            items: List[str] = []
        else:
            items = list(descriptor.get("items", []))

        descriptor_count = len(items)
        descriptor_counts.append(descriptor_count)
        quality = _quality_label(row, descriptor, descriptor_signature_counts)
        quality_counts[quality] += 1
        if "shared_variant_descriptor" in quality:
            shared_variant_classes += 1
        traits = _trait_phrase(items, max_descriptors)

        for prompt_id, template in enumerate(PROMPT_TEMPLATES):
            output_rows.append(
                {
                    "raw_class_id": raw_class_id,
                    "target": int(row["target"]),
                    "class_name": row["class_name"],
                    "class_family": row["class_family"],
                    "prompt_id": prompt_id,
                    "prompt": template.format(class_name=row["class_name"], traits=traits),
                    "prompt_set": "descriptor_variant",
                    "descriptor_count": descriptor_count,
                    "descriptor_quality": quality,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "nabirds_descriptor_prompts.csv"
    written = _write_csv(
        output_csv,
        output_rows,
        fieldnames=(
            "raw_class_id",
            "target",
            "class_name",
            "class_family",
            "prompt_id",
            "prompt",
            "prompt_set",
            "descriptor_count",
            "descriptor_quality",
        ),
    )

    targets = {int(row["target"]) for row in output_rows}
    raw_ids = {int(row["raw_class_id"]) for row in output_rows}
    summary = {
        "class_prompts": str(class_prompts),
        "descriptors_csv": str(descriptors_csv),
        "output_csv": str(output_csv),
        "prompt_rows": written,
        "unique_targets": len(targets),
        "unique_raw_class_ids": len(raw_ids),
        "expected_targets": 555,
        "missing_raw_class_ids": missing_raw_ids,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "max_descriptors_per_prompt": max_descriptors,
        "descriptor_count_min": min(descriptor_counts) if descriptor_counts else None,
        "descriptor_count_max": max(descriptor_counts) if descriptor_counts else None,
        "descriptor_count_zero_classes": sum(count == 0 for count in descriptor_counts),
        "shared_variant_descriptor_classes": shared_variant_classes,
        "descriptor_quality_counts": dict(sorted(quality_counts.items())),
    }
    summary_path = out_dir / "descriptor_prompt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if len(targets) != 555:
        raise SystemExit(f"Expected 555 unique targets; wrote {len(targets)}.")
    if missing_raw_ids:
        preview = ", ".join(str(value) for value in missing_raw_ids[:10])
        raise SystemExit(f"Missing descriptor rows for raw_class_id(s): {preview}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build descriptor-augmented NABirds prompts for Milestone 3.")
    parser.add_argument("--class-prompts", default=DEFAULT_CLASS_PROMPTS)
    parser.add_argument("--descriptors-csv", default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-descriptors", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_descriptor_prompts(
        class_prompts=Path(args.class_prompts),
        descriptors_csv=Path(args.descriptors_csv),
        out_dir=Path(args.out_dir),
        max_descriptors=args.max_descriptors,
    )
    print("Milestone 3 descriptor prompts written")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
