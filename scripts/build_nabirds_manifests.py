#!/usr/bin/env python3
"""Build Milestone 2 manifests for NABirds experiments.

Outputs:
  - nabirds_train.csv / nabirds_val.csv / nabirds_test.csv
  - nabirds_all_manifest.csv
  - nabirds_class_prompts.csv
  - nabirds_hard_negative_groups.csv

The train/val split is created only from the official NABirds training split.
The official test split is kept untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.nabirds_audit import NABirdsMetadata, TRAIN, TEST  # noqa: E402


PROMPT_TEMPLATES = (
    "a photo of a {class_name}, a North American bird.",
    "a field guide photograph of a {class_name}.",
    "a close-up bird photo showing the fine-grained field marks of a {class_name}.",
)


def _class_family(class_name: str) -> str:
    """Collapse class variants such as parenthetical plumage labels."""
    family = re.sub(r"\s*\([^)]*\)", "", class_name).strip()
    return family or class_name


def _stratified_split(
    image_ids: Sequence[str],
    labels: Sequence[int],
    val_fraction: float,
    seed: int,
) -> Tuple[set[str], set[str]]:
    """Return train_ids, val_ids with sklearn when available."""
    try:
        from sklearn.model_selection import train_test_split

        train_ids, val_ids = train_test_split(
            list(image_ids),
            test_size=val_fraction,
            random_state=seed,
            stratify=list(labels),
        )
        return set(train_ids), set(val_ids)
    except Exception:
        # Deterministic fallback: choose at least one validation example per class
        # where possible, roughly matching val_fraction.
        import random

        rng = random.Random(seed)
        by_label: Dict[int, List[str]] = defaultdict(list)
        for image_id, label in zip(image_ids, labels):
            by_label[label].append(image_id)

        train_ids = set()
        val_ids = set()
        for grouped_ids in by_label.values():
            grouped_ids = list(grouped_ids)
            rng.shuffle(grouped_ids)
            n_val = max(1, round(len(grouped_ids) * val_fraction))
            n_val = min(n_val, len(grouped_ids) - 1) if len(grouped_ids) > 1 else 0
            val_ids.update(grouped_ids[:n_val])
            train_ids.update(grouped_ids[n_val:])
        return train_ids, val_ids


def _record_row(record, split: str) -> Dict[str, object]:
    clipped = record.bbox.clipped(record.width, record.height)
    return {
        "split": split,
        "official_split": record.split,
        "image_id": record.image_id,
        "rel_path": record.rel_path,
        "image_path": str(record.image_path),
        "raw_class_id": record.class_id,
        "target": record.class_index,
        "class_name": record.class_name,
        "class_family": _class_family(record.class_name),
        "width": record.width,
        "height": record.height,
        "bbox_x": record.bbox.x,
        "bbox_y": record.bbox.y,
        "bbox_w": record.bbox.width,
        "bbox_h": record.bbox.height,
        "bbox_in_bounds": int(record.bbox.in_bounds(record.width, record.height)),
        "clip_x": clipped.x,
        "clip_y": clipped.y,
        "clip_w": clipped.width,
        "clip_h": clipped.height,
        "bbox_area_ratio": round(record.bbox.area_ratio(record.width, record.height), 6),
        "visible_parts": sum(part.visible for part in record.parts),
        "photographer": record.photographer,
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_manifests(dataset_root: Path, out_dir: Path, val_fraction: float, seed: int) -> Dict[str, object]:
    metadata = NABirdsMetadata(dataset_root)
    official_train = metadata.records(TRAIN)
    official_test = metadata.records(TEST)

    train_ids, val_ids = _stratified_split(
        [record.image_id for record in official_train],
        [record.class_index for record in official_train],
        val_fraction=val_fraction,
        seed=seed,
    )

    rows = []
    for record in metadata.records():
        if record.split == TEST:
            split = TEST
        elif record.image_id in val_ids:
            split = "val"
        else:
            split = TRAIN
        rows.append(_record_row(record, split))

    fieldnames = list(rows[0])
    counts = {
        "all": _write_csv(out_dir / "nabirds_all_manifest.csv", rows, fieldnames),
        "train": _write_csv(out_dir / "nabirds_train.csv", [row for row in rows if row["split"] == TRAIN], fieldnames),
        "val": _write_csv(out_dir / "nabirds_val.csv", [row for row in rows if row["split"] == "val"], fieldnames),
        "test": _write_csv(out_dir / "nabirds_test.csv", [row for row in rows if row["split"] == TEST], fieldnames),
    }

    class_rows = []
    for class_row in metadata.class_counts():
        class_name = str(class_row["class_name"])
        family = _class_family(class_name)
        for prompt_id, template in enumerate(PROMPT_TEMPLATES):
            class_rows.append(
                {
                    "raw_class_id": class_row["class_id"],
                    "target": class_row["class_index"],
                    "class_name": class_name,
                    "class_family": family,
                    "prompt_id": prompt_id,
                    "prompt": template.format(class_name=class_name),
                    "total_images": class_row["total"],
                    "official_train_images": class_row["train"],
                    "official_test_images": class_row["test"],
                }
            )
    prompt_fields = list(class_rows[0])
    counts["class_prompts"] = _write_csv(out_dir / "nabirds_class_prompts.csv", class_rows, prompt_fields)

    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for class_row in metadata.class_counts():
        groups[_class_family(str(class_row["class_name"]))].append(class_row)

    hard_negative_rows = []
    for family, family_rows in sorted(groups.items()):
        if len(family_rows) < 2:
            continue
        targets = [str(row["class_index"]) for row in sorted(family_rows, key=lambda row: int(row["class_index"]))]
        names = [str(row["class_name"]) for row in sorted(family_rows, key=lambda row: int(row["class_index"]))]
        hard_negative_rows.append(
            {
                "class_family": family,
                "num_variants": len(family_rows),
                "targets": "|".join(targets),
                "class_names": " | ".join(names),
            }
        )
    hard_fields = ["class_family", "num_variants", "targets", "class_names"]
    counts["hard_negative_groups"] = _write_csv(
        out_dir / "nabirds_hard_negative_groups.csv",
        hard_negative_rows,
        hard_fields,
    )

    summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(out_dir.resolve()),
        "seed": seed,
        "val_fraction": val_fraction,
        "counts": counts,
        "num_classes": len(metadata.class_counts()),
        "official_train": len(official_train),
        "official_test": len(official_test),
        "bbox_out_of_bounds": sum(
            not record.bbox.in_bounds(record.width, record.height)
            for record in metadata.records()
        ),
        "prompt_templates": list(PROMPT_TEMPLATES),
    }
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NABirds manifests for training and VLM evaluation.")
    parser.add_argument("--dataset-root", default="nabirds")
    parser.add_argument("--out-dir", default="reports/milestone2")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=231)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifests(
        dataset_root=Path(args.dataset_root),
        out_dir=Path(args.out_dir),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print("NABirds Milestone 2 manifests written")
    for key, value in summary["counts"].items():
        print(f"  {key}: {value}")
    print(f"  bbox_out_of_bounds: {summary['bbox_out_of_bounds']}")
    print(f"  output_dir: {summary['output_dir']}")


if __name__ == "__main__":
    main()
