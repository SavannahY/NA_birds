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
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_CLASS_PROMPTS = "reports/milestone2/nabirds_class_prompts.csv"
DEFAULT_DESCRIPTORS = "bird_class_expert_description_visual_attribute_descriptors.csv"
DEFAULT_OUT_DIR = "reports/milestone3"

CLASS_NAME_ONLY_TEMPLATES = ("{label}",)

SHORT_VARIANT_TEMPLATES = (
    "{label}",
    "a photo of {article} {variant_first_label}.",
    "a field guide photo of {article} {variant_first_label}.",
)

CLEAN_DESCRIPTOR_TEMPLATES = (
    "{label}; field marks: {traits}.",
    "a field guide photo of {article} {variant_first_label} with {traits}.",
)

DESCRIPTOR_ONLY_TEMPLATES = (
    "a bird with {traits}.",
    "a field guide photo of a bird showing {traits}.",
)

OUTPUT_SPECS = (
    ("class_name_only", "nabirds_class_name_only_prompts.csv"),
    ("short_variant", "nabirds_short_variant_prompts.csv"),
    ("descriptor_only", "nabirds_descriptor_only_prompts.csv"),
    ("clean_descriptor", "nabirds_clean_descriptor_prompts.csv"),
)

LEGACY_DESCRIPTOR_FILENAME = "nabirds_descriptor_prompts.csv"

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
    "bar",
    "barred",
    "bars",
    "belly",
    "bib",
    "border",
    "breast",
    "cap",
    "cheek",
    "chest",
    "collar",
    "crown",
    "eyebrow",
    "eyeline",
    "eye-ring",
    "face",
    "flank",
    "flanks",
    "hood",
    "line",
    "mantle",
    "mask",
    "nape",
    "patch",
    "patches",
    "ring",
    "rump",
    "shoulder",
    "spot",
    "spotted",
    "spots",
    "streak",
    "streaked",
    "streaking",
    "stripe",
    "striped",
    "supercilium",
    "throat",
    "underparts",
    "upperparts",
    "wing",
    "wingbar",
    "wingbars",
}
GENERIC_TRAIT_PATTERNS = (
    re.compile(r"^(small|medium-sized|medium|large|slender|stocky|compact|chunky|plump|heavyset) body$"),
    re.compile(r"^(small|large|rounded|round) head$"),
    re.compile(r"^(short|long|medium-length) tail$"),
    re.compile(r"^(short|long) legs$"),
    re.compile(r"^(short|long|thick) neck$"),
    re.compile(r"^(small|long|short|straight|thin|thick) bill$"),
    re.compile(r"^dark eye$"),
)
BAD_TRAIT_SUBSTRINGS = (
    "also occur",
    " vary ",
    "varies ",
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


def _squash_spaces(value: str) -> str:
    return " ".join(value.strip().split())


def _article_for(label: str) -> str:
    if not label:
        return "a"
    return "an" if label[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _clean_variant_text(value: str) -> str:
    value = value.replace("/", " or ")
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")


def _split_class_name(class_name: str) -> Tuple[str, str, str, str]:
    """Return base name, variant text, class-first label, and variant-first label."""
    normalized = _squash_spaces(class_name.replace(" (", " (").replace("( ", "(").replace(" )", ")"))
    variants = [_clean_variant_text(match) for match in re.findall(r"\(([^)]*)\)", normalized)]
    variants = [value for value in variants if value]
    base = _squash_spaces(re.sub(r"\s*\([^)]*\)", "", normalized))
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


def _normalize_trait(item: str) -> str:
    item = item.strip().strip(" .;:")
    item = re.sub(r"^red-tailed rich\b", "rich", item, flags=re.IGNORECASE)
    item = item.replace("/", " or ")
    item = re.sub(r"\s+", " ", item)
    return item


def _trait_tokens(item: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z-]+", item.lower()) if token}


def _is_generic_trait(item: str) -> bool:
    lowered = item.lower()
    return any(pattern.match(lowered) for pattern in GENERIC_TRAIT_PATTERNS)


def _trait_score(item: str, variant: str) -> int:
    tokens = _trait_tokens(item)
    score = 0
    if tokens & COLOR_WORDS:
        score += 3
    if tokens & MARK_WORDS:
        score += 2
    if "bill" in tokens or "tail" in tokens:
        score += 1
    if _is_generic_trait(item):
        score -= 3
    variant_tokens = _trait_tokens(variant)
    if variant_tokens and tokens & variant_tokens:
        score += 2
    return score


def _clean_descriptor_items(
    items: Sequence[str],
    *,
    variant: str,
    shared_variant_descriptor: bool,
    max_items: int,
) -> List[str]:
    cleaned: List[Tuple[int, int, str]] = []
    seen = set()
    for index, item in enumerate(items):
        trait = _normalize_trait(item)
        if not trait:
            continue
        lowered = f" {trait.lower()} "
        if any(value in lowered for value in BAD_TRAIT_SUBSTRINGS):
            continue
        key = trait.casefold()
        if key in seen:
            continue
        seen.add(key)
        if len(trait.split()) > 7:
            continue
        score = _trait_score(trait, variant)
        if score < 1:
            continue
        cleaned.append((score, index, trait))

    if shared_variant_descriptor and variant:
        # Shared descriptors often describe the base species rather than the
        # labeled sex/age/morph. Keep only the most specific marks.
        cleaned = [entry for entry in cleaned if entry[0] >= 4]

    cleaned.sort(key=lambda entry: (-entry[0], entry[1], entry[2].casefold()))
    selected = [entry[2] for entry in cleaned[:max_items]]
    variant_trait = _variant_trait(variant)
    if variant_trait and variant_trait.casefold() not in {item.casefold() for item in selected}:
        selected = [variant_trait] + selected
    return selected[:max_items]


def _trait_phrase(items: Sequence[str], max_items: int) -> str:
    kept = list(items[:max_items])
    if not kept:
        return "distinct field marks"
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


def _is_shared_variant_descriptor(
    class_row: Mapping[str, str],
    descriptor: Mapping[str, object] | None,
    descriptor_signature_counts: Mapping[str, Counter[str]],
) -> bool:
    if descriptor is None:
        return False
    items = descriptor.get("items", [])
    signature = "|".join(str(item).casefold() for item in items)
    return descriptor_signature_counts[class_row["class_family"]][signature] > 1


def _prompt_context(row: Mapping[str, str]) -> Dict[str, str]:
    base_name, variant, label, variant_first_label = _split_class_name(row["class_name"])
    return {
        "base_name": base_name,
        "variant": variant,
        "label": label,
        "variant_first_label": variant_first_label,
        "article": _article_for(variant_first_label),
    }


def _make_prompt_row(
    row: Mapping[str, str],
    *,
    prompt_id: int,
    prompt: str,
    prompt_set: str,
    descriptor_count: int,
    clean_descriptor_count: int,
    descriptor_quality: str,
    variant: str,
) -> Dict[str, object]:
    return {
        "raw_class_id": int(row["raw_class_id"]),
        "target": int(row["target"]),
        "class_name": row["class_name"],
        "class_family": row["class_family"],
        "prompt_id": prompt_id,
        "prompt": prompt,
        "prompt_set": prompt_set,
        "variant_label": variant,
        "descriptor_count": descriptor_count,
        "clean_descriptor_count": clean_descriptor_count,
        "descriptor_quality": descriptor_quality,
    }


def _validate_prompt_rows(rows: Sequence[Mapping[str, object]], *, expected_targets: int, label: str) -> Dict[str, object]:
    targets = {int(row["target"]) for row in rows}
    raw_ids = {int(row["raw_class_id"]) for row in rows}
    if len(targets) != expected_targets:
        raise SystemExit(f"{label}: expected {expected_targets} unique targets; wrote {len(targets)}.")
    empty_prompts = [row for row in rows if not str(row["prompt"]).strip()]
    if empty_prompts:
        raise SystemExit(f"{label}: wrote {len(empty_prompts)} empty prompt(s).")
    return {
        "rows": len(rows),
        "unique_targets": len(targets),
        "unique_raw_class_ids": len(raw_ids),
    }


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

    prompt_sets: Dict[str, List[Dict[str, object]]] = {name: [] for name, _filename in OUTPUT_SPECS}
    missing_raw_ids: List[int] = []
    descriptor_counts: List[int] = []
    clean_descriptor_counts: List[int] = []
    quality_counts: Counter[str] = Counter()
    shared_variant_classes = 0
    variant_class_count = 0

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
        shared_variant_descriptor = _is_shared_variant_descriptor(row, descriptor, descriptor_signature_counts)
        if shared_variant_descriptor:
            shared_variant_classes += 1
        context = _prompt_context(row)
        variant_class_count += int(bool(context["variant"]))
        clean_items = _clean_descriptor_items(
            items,
            variant=context["variant"],
            shared_variant_descriptor=shared_variant_descriptor,
            max_items=max_descriptors,
        )
        clean_descriptor_counts.append(len(clean_items))
        traits = _trait_phrase(clean_items, max_descriptors)

        for prompt_id, template in enumerate(CLASS_NAME_ONLY_TEMPLATES):
            prompt_sets["class_name_only"].append(
                _make_prompt_row(
                    row,
                    prompt_id=prompt_id,
                    prompt=template.format(**context),
                    prompt_set="class_name_only",
                    descriptor_count=descriptor_count,
                    clean_descriptor_count=len(clean_items),
                    descriptor_quality=quality,
                    variant=context["variant"],
                )
            )

        for prompt_id, template in enumerate(SHORT_VARIANT_TEMPLATES):
            prompt_sets["short_variant"].append(
                _make_prompt_row(
                    row,
                    prompt_id=prompt_id,
                    prompt=template.format(**context),
                    prompt_set="short_variant",
                    descriptor_count=descriptor_count,
                    clean_descriptor_count=len(clean_items),
                    descriptor_quality=quality,
                    variant=context["variant"],
                )
            )

        descriptor_context = {**context, "traits": traits}
        for prompt_id, template in enumerate(DESCRIPTOR_ONLY_TEMPLATES):
            prompt_sets["descriptor_only"].append(
                _make_prompt_row(
                    row,
                    prompt_id=prompt_id,
                    prompt=template.format(**descriptor_context),
                    prompt_set="descriptor_only",
                    descriptor_count=descriptor_count,
                    clean_descriptor_count=len(clean_items),
                    descriptor_quality=quality,
                    variant=context["variant"],
                )
            )

        for prompt_id, template in enumerate(CLEAN_DESCRIPTOR_TEMPLATES):
            prompt_sets["clean_descriptor"].append(
                _make_prompt_row(
                    row,
                    prompt_id=prompt_id,
                    prompt=template.format(**descriptor_context),
                    prompt_set="clean_descriptor",
                    descriptor_count=descriptor_count,
                    clean_descriptor_count=len(clean_items),
                    descriptor_quality=quality,
                    variant=context["variant"],
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "raw_class_id",
        "target",
        "class_name",
        "class_family",
        "prompt_id",
        "prompt",
        "prompt_set",
        "variant_label",
        "descriptor_count",
        "clean_descriptor_count",
        "descriptor_quality",
    )

    output_files: Dict[str, str] = {}
    output_stats: Dict[str, Dict[str, object]] = {}
    for prompt_set, filename in OUTPUT_SPECS:
        output_csv = out_dir / filename
        rows = prompt_sets[prompt_set]
        _write_csv(output_csv, rows, fieldnames=fieldnames)
        output_files[prompt_set] = str(output_csv)
        output_stats[prompt_set] = _validate_prompt_rows(rows, expected_targets=555, label=filename)

    legacy_output = out_dir / LEGACY_DESCRIPTOR_FILENAME
    _write_csv(legacy_output, prompt_sets["clean_descriptor"], fieldnames=fieldnames)
    output_files["legacy_clean_descriptor"] = str(legacy_output)
    output_stats["legacy_clean_descriptor"] = _validate_prompt_rows(
        prompt_sets["clean_descriptor"],
        expected_targets=555,
        label=LEGACY_DESCRIPTOR_FILENAME,
    )

    summary = {
        "class_prompts": str(class_prompts),
        "descriptors_csv": str(descriptors_csv),
        "output_files": output_files,
        "output_stats": output_stats,
        "expected_targets": 555,
        "missing_raw_class_ids": missing_raw_ids,
        "prompt_templates": {
            "class_name_only": list(CLASS_NAME_ONLY_TEMPLATES),
            "short_variant": list(SHORT_VARIANT_TEMPLATES),
            "descriptor_only": list(DESCRIPTOR_ONLY_TEMPLATES),
            "clean_descriptor": list(CLEAN_DESCRIPTOR_TEMPLATES),
        },
        "max_descriptors_per_prompt": max_descriptors,
        "variant_classes": variant_class_count,
        "descriptor_count_min": min(descriptor_counts) if descriptor_counts else None,
        "descriptor_count_max": max(descriptor_counts) if descriptor_counts else None,
        "descriptor_count_zero_classes": sum(count == 0 for count in descriptor_counts),
        "clean_descriptor_count_min": min(clean_descriptor_counts) if clean_descriptor_counts else None,
        "clean_descriptor_count_max": max(clean_descriptor_counts) if clean_descriptor_counts else None,
        "clean_descriptor_count_zero_classes": sum(count == 0 for count in clean_descriptor_counts),
        "shared_variant_descriptor_classes": shared_variant_classes,
        "descriptor_quality_counts": dict(sorted(quality_counts.items())),
    }
    summary_path = out_dir / "descriptor_prompt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if missing_raw_ids:
        preview = ", ".join(str(value) for value in missing_raw_ids[:10])
        raise SystemExit(f"Missing descriptor rows for raw_class_id(s): {preview}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build descriptor-augmented NABirds prompts for Milestone 3.")
    parser.add_argument("--class-prompts", default=DEFAULT_CLASS_PROMPTS)
    parser.add_argument("--descriptors-csv", default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-descriptors", type=int, default=5)
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
