#!/usr/bin/env python3
"""Analyze NABirds prediction CSVs for Step 5/6 error analysis.

Expected prediction CSV columns:
  - target
  - pred_target

Recommended optional columns:
  - image_id, rel_path, image_path, class_name, pred_class_name, top5_targets

The VLM evaluator can produce this format with:
  python3 scripts/eval_vlm_zero_shot.py --output-predictions reports/milestone2/vlm_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} is empty or missing a header row.")
        missing = [column for column in ("target", "pred_target") if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} is missing required column(s): {', '.join(missing)}")
        return list(reader)


def _load_class_names(class_prompts: Path) -> Dict[int, str]:
    rows = _read_csv_with_required(class_prompts, ("target", "class_name"))
    names = {}
    for row in rows:
        target = int(row["target"])
        names.setdefault(target, row["class_name"])
    return names


def _read_csv_with_required(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        if any(column not in reader.fieldnames for column in required_columns):
            return []
        return list(reader)


def _macro_f1(targets: Sequence[int], preds: Sequence[int], labels: Sequence[int]) -> float:
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}
    for target, pred in zip(targets, preds):
        if target == pred:
            tp[target] += 1
        else:
            fn[target] += 1
            fp[pred] = fp.get(pred, 0) + 1

    values = []
    for label in labels:
        denom = (2 * tp.get(label, 0)) + fp.get(label, 0) + fn.get(label, 0)
        values.append(0.0 if denom == 0 else (2 * tp.get(label, 0)) / denom)
    return sum(values) / len(values)


def _balanced_accuracy(targets: Sequence[int], preds: Sequence[int], labels: Sequence[int]) -> float:
    recalls = []
    for label in labels:
        total = sum(target == label for target in targets)
        if total == 0:
            continue
        correct = sum(target == label and pred == label for target, pred in zip(targets, preds))
        recalls.append(correct / total)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def analyze(
    predictions_csv: Path,
    out_dir: Path,
    class_prompts: Path,
    top_n: int,
) -> Dict[str, object]:
    rows = _read_csv(predictions_csv)
    class_names = _load_class_names(class_prompts)

    targets = [int(row["target"]) for row in rows]
    preds = [int(row["pred_target"]) for row in rows]
    labels = sorted(set(targets) | set(preds))
    total = len(rows)
    correct = sum(target == pred for target, pred in zip(targets, preds))
    top5_available = "correct_top5" in rows[0]
    top5_correct = sum(int(row.get("correct_top5", "0")) for row in rows) if top5_available else None

    pair_counts = Counter((target, pred) for target, pred in zip(targets, preds) if target != pred)
    confusion_rows = []
    for (target, pred), count in pair_counts.most_common(top_n):
        confusion_rows.append(
            {
                "target": target,
                "target_name": class_names.get(target, ""),
                "pred_target": pred,
                "pred_name": class_names.get(pred, ""),
                "count": count,
            }
        )

    per_class = []
    grouped: Dict[int, List[int]] = defaultdict(list)
    for target, pred in zip(targets, preds):
        grouped[target].append(pred)
    for target in sorted(grouped):
        class_preds = grouped[target]
        class_total = len(class_preds)
        class_correct = sum(pred == target for pred in class_preds)
        per_class.append(
            {
                "target": target,
                "class_name": class_names.get(target, ""),
                "num_examples": class_total,
                "recall": class_correct / class_total,
                "num_correct": class_correct,
            }
        )
    per_class.sort(key=lambda row: (row["recall"], -row["num_examples"]))

    error_examples = []
    for row in rows:
        if int(row["target"]) == int(row["pred_target"]):
            continue
        error_examples.append(
            {
                "image_id": row.get("image_id", ""),
                "rel_path": row.get("rel_path", ""),
                "target": row["target"],
                "class_name": row.get("class_name") or class_names.get(int(row["target"]), ""),
                "pred_target": row["pred_target"],
                "pred_class_name": row.get("pred_class_name") or class_names.get(int(row["pred_target"]), ""),
                "top5_targets": row.get("top5_targets", ""),
            }
        )
        if len(error_examples) >= top_n:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "top_confusions.csv", confusion_rows, ["target", "target_name", "pred_target", "pred_name", "count"])
    _write_csv(out_dir / "per_class_recall.csv", per_class, ["target", "class_name", "num_examples", "recall", "num_correct"])
    _write_csv(out_dir / "error_examples.csv", error_examples, ["image_id", "rel_path", "target", "class_name", "pred_target", "pred_class_name", "top5_targets"])

    summary = {
        "predictions_csv": str(predictions_csv),
        "num_examples": total,
        "top1_accuracy": correct / total if total else 0.0,
        "top5_accuracy": (top5_correct / total if top5_correct is not None and total else None),
        "macro_f1": _macro_f1(targets, preds, labels),
        "balanced_accuracy": _balanced_accuracy(targets, preds, labels),
        "num_labels_in_predictions": len(labels),
        "top_confusions_csv": str(out_dir / "top_confusions.csv"),
        "per_class_recall_csv": str(out_dir / "per_class_recall.csv"),
        "error_examples_csv": str(out_dir / "error_examples.csv"),
    }
    (out_dir / "prediction_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NABirds prediction errors.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV with target and pred_target columns.")
    parser.add_argument("--out-dir", default="reports/milestone2/error_analysis")
    parser.add_argument("--class-prompts", default="reports/milestone2/nabirds_class_prompts.csv")
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        predictions_csv=Path(args.predictions),
        out_dir=Path(args.out_dir),
        class_prompts=Path(args.class_prompts),
        top_n=args.top_n,
    )
    print("NABirds prediction analysis complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
