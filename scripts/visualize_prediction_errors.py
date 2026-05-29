#!/usr/bin/env python3
"""Visualize NABirds misclassifications and confusion structure from prediction CSVs.

Produces:
  - confusion_matrix_family.png   (aggregated by class_family)
  - confusion_matrix_top_classes.png (heatmap for busiest classes)
  - misclassification_grid.png  (sample wrong predictions with labels)
  - summary.json

Typical workflow:
  python3 scripts/eval_visual_checkpoint.py \\
    --checkpoint path/to/best.pt \\
    --manifest reports/milestone2/nabirds_test.csv \\
    --output-predictions reports/milestone2/resnet50_test_predictions.csv

  uv run python scripts/visualize_prediction_errors.py \\
    --predictions reports/milestone2/resnet50_test_predictions.csv \\
    --out-dir reports/milestone2/error_analysis/resnet50_full
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix


def _read_predictions(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing predictions CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row.")
        required = {"target", "pred_target"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise SystemExit(f"{path} missing columns: {', '.join(missing)}")
        return list(reader)


def _load_class_meta(manifest_path: Path, class_prompts_path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    names: Dict[int, str] = {}
    families: Dict[int, str] = {}

    if class_prompts_path.is_file():
        with class_prompts_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                target = int(row["target"])
                names.setdefault(target, row.get("class_name", ""))
                if row.get("class_family"):
                    families.setdefault(target, row["class_family"])

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                target = int(row["target"])
                names.setdefault(target, row.get("class_name", ""))
                families.setdefault(target, row.get("class_family") or names.get(target, str(target)))

    return names, families


def _resolve_image_path(row: Mapping[str, str], path_root: Path) -> Path:
    raw = row.get("image_path") or row.get("rel_path") or ""
    path = Path(raw)
    if path.is_file():
        return path
    candidate = path_root / raw
    if candidate.is_file():
        return candidate
    if row.get("rel_path"):
        rel = path_root / "nabirds" / "images" / row["rel_path"]
        if rel.is_file():
            return rel
    raise FileNotFoundError(f"Could not resolve image for row: {row.get('image_id', raw)}")


def _plot_confusion(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    out_path: Path,
    *,
    figsize: Tuple[float, float] = (12, 10),
    max_label_len: int = 40,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    short = [label if len(label) <= max_label_len else label[: max_label_len - 3] + "..." for label in labels]
    tick_step = 1 if len(labels) <= 40 else max(1, len(labels) // 40)
    ticks = list(range(0, len(labels), tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([short[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([short[i] for i in ticks], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _family_confusion(
    targets: Sequence[int],
    preds: Sequence[int],
    families: Dict[int, str],
    class_names: Dict[int, str],
) -> Tuple[np.ndarray, List[str]]:
    family_for_target = {t: families.get(t, class_names.get(t, str(t))) for t in set(targets) | set(preds)}
    labels = sorted(set(family_for_target.values()))
    index = {name: i for i, name in enumerate(labels)}
    mapped_true = [index[family_for_target[t]] for t in targets]
    mapped_pred = [index[family_for_target[p]] for p in preds]
    matrix = confusion_matrix(mapped_true, mapped_pred, labels=list(range(len(labels))))
    return matrix, labels


def _top_class_confusion(
    targets: Sequence[int],
    preds: Sequence[int],
    class_names: Dict[int, str],
    top_k: int,
) -> Tuple[np.ndarray, List[str]]:
    counts = Counter(targets)
    top_ids = [label for label, _ in counts.most_common(top_k)]
    index = {label: i for i, label in enumerate(top_ids)}
    mapped_true = [index[t] if t in index else -1 for t in targets]
    mapped_pred = [index[p] if p in index else -1 for p in preds]
    kept_true, kept_pred = [], []
    for t, p in zip(mapped_true, mapped_pred):
        if t >= 0 and p >= 0:
            kept_true.append(t)
            kept_pred.append(p)
    matrix = confusion_matrix(kept_true, kept_pred, labels=list(range(len(top_ids))))
    labels = [class_names.get(cid, str(cid)) for cid in top_ids]
    return matrix, labels


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    *,
    fill: str = "white",
    max_width: int = 280,
    font: Optional[ImageFont.ImageFont] = None,
) -> None:
    font = font or ImageFont.load_default()
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines[:4]:
        draw.text((x, y), line, fill=fill, font=font)
        y += (draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]) + 2


def _plot_misclassification_grid(
    errors: Sequence[Mapping[str, str]],
    class_names: Dict[int, str],
    path_root: Path,
    out_path: Path,
    *,
    cols: int = 4,
    thumb_size: int = 224,
) -> int:
    if not errors:
        return 0
    rows = (len(errors) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.6))
    gs = GridSpec(rows, cols, figure=fig, wspace=0.15, hspace=0.35)
    shown = 0
    for idx, row in enumerate(errors):
        try:
            image_path = _resolve_image_path(row, path_root)
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((thumb_size, thumb_size))
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.imshow(image)
        ax.axis("off")
        true_name = row.get("class_name") or class_names.get(int(row["target"]), row["target"])
        pred_name = row.get("pred_class_name") or class_names.get(int(row["pred_target"]), row["pred_target"])
        ax.set_title(f"TRUE: {true_name}\nPRED: {pred_name}", fontsize=7, color="#b00020")
        shown += 1
    if shown == 0:
        plt.close(fig)
        return 0
    fig.suptitle("Sample misclassifications (test set)", fontsize=12, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return shown


def visualize(
    predictions_csv: Path,
    out_dir: Path,
    manifest_csv: Path,
    class_prompts_csv: Path,
    path_root: Path,
    *,
    top_classes: int,
    num_error_images: int,
    seed: int,
) -> Dict[str, object]:
    rows = _read_predictions(predictions_csv)
    class_names, families = _load_class_meta(manifest_csv, class_prompts_csv)
    targets = [int(row["target"]) for row in rows]
    preds = [int(row["pred_target"]) for row in rows]
    total = len(rows)
    correct = sum(t == p for t, p in zip(targets, preds))

    out_dir.mkdir(parents=True, exist_ok=True)

    family_matrix, family_labels = _family_confusion(targets, preds, families, class_names)
    _plot_confusion(
        family_matrix,
        family_labels,
        f"Confusion matrix by class family (n={total}, acc={correct/total:.3f})",
        out_dir / "confusion_matrix_family.png",
        figsize=(14, 12),
        max_label_len=32,
    )

    top_matrix, top_labels = _top_class_confusion(targets, preds, class_names, top_classes)
    _plot_confusion(
        top_matrix,
        top_labels,
        f"Confusion matrix: top {top_classes} classes by frequency",
        out_dir / "confusion_matrix_top_classes.png",
        figsize=(16, 14),
        max_label_len=36,
    )

    pair_counts = Counter((t, p) for t, p in zip(targets, preds) if t != p)
    top_pairs = []
    for (t, p), count in pair_counts.most_common(30):
        top_pairs.append(
            {
                "target": t,
                "true_name": class_names.get(t, ""),
                "pred_target": p,
                "pred_name": class_names.get(p, ""),
                "count": count,
                "same_family": families.get(t, "") == families.get(p, ""),
            }
        )
    with (out_dir / "top_confusion_pairs.json").open("w", encoding="utf-8") as handle:
        json.dump(top_pairs, handle, indent=2)

    errors = [row for row in rows if int(row["target"]) != int(row["pred_target"])]
    rng = np.random.default_rng(seed)
    if len(errors) > num_error_images:
        indices = rng.choice(len(errors), size=num_error_images, replace=False)
        sampled_errors = [errors[int(i)] for i in sorted(indices)]
    else:
        sampled_errors = errors

    shown = _plot_misclassification_grid(
        sampled_errors,
        class_names,
        path_root,
        out_dir / "misclassification_grid.png",
    )

    summary = {
        "predictions_csv": str(predictions_csv),
        "num_examples": total,
        "top1_accuracy": correct / total if total else 0.0,
        "num_errors": total - correct,
        "num_unique_families": len(family_labels),
        "top_classes_in_heatmap": top_classes,
        "num_error_images_requested": num_error_images,
        "num_error_images_shown": shown,
        "outputs": {
            "confusion_matrix_family": str(out_dir / "confusion_matrix_family.png"),
            "confusion_matrix_top_classes": str(out_dir / "confusion_matrix_top_classes.png"),
            "misclassification_grid": str(out_dir / "misclassification_grid.png"),
            "top_confusion_pairs": str(out_dir / "top_confusion_pairs.json"),
        },
    }
    (out_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion matrices and misclassification grids.")
    parser.add_argument("--predictions", required=True, help="CSV with target and pred_target columns.")
    parser.add_argument("--out-dir", default="reports/milestone2/error_analysis")
    parser.add_argument("--manifest", default="reports/milestone2/nabirds_test.csv")
    parser.add_argument("--class-prompts", default="reports/milestone2/nabirds_class_prompts.csv")
    parser.add_argument("--path-root", default=".", help="Base path for resolving image_path / rel_path.")
    parser.add_argument("--top-classes", type=int, default=30, help="Number of busiest classes in fine-grained heatmap.")
    parser.add_argument("--num-error-images", type=int, default=12, help="Misclassified examples in the image grid.")
    parser.add_argument("--seed", type=int, default=231)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = visualize(
        predictions_csv=Path(args.predictions),
        out_dir=Path(args.out_dir),
        manifest_csv=Path(args.manifest),
        class_prompts_csv=Path(args.class_prompts),
        path_root=Path(args.path_root).resolve(),
        top_classes=args.top_classes,
        num_error_images=args.num_error_images,
        seed=args.seed,
    )
    print("Error visualization complete")
    for key, value in summary.items():
        if key != "outputs":
            print(f"  {key}: {value}")
    print("  outputs:")
    for key, value in summary["outputs"].items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
