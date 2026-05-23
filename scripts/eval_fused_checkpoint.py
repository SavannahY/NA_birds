#!/usr/bin/env python3
"""Evaluate a trained fused full-image + bbox-crop NABirds checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_fused_full_crop import (  # noqa: E402
    _dataloader_kwargs,
    _limit_rows,
    _read_manifest,
    _require_training_deps,
    accuracy_at_k,
    build_fused_model,
    build_transforms,
    make_dataset_class,
)


def _torch_load(torch, path: Path, map_location: str) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_class_names(path: Optional[Path]) -> Dict[int, str]:
    if path is None or not path.is_file():
        return {}
    names: Dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "target" in row and "class_name" in row:
                names.setdefault(int(row["target"]), row["class_name"])
    return names


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _resolve_from_checkpoint(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    ckpt_args = dict(checkpoint.get("args") or {})
    return {
        "model": args.model or ckpt_args.get("model", "resnet50"),
        "branch_mode": args.branch_mode or ckpt_args.get("branch_mode", "shared"),
        "num_classes": args.num_classes or int(ckpt_args.get("num_classes", 555)),
        "image_size": args.image_size or int(ckpt_args.get("image_size", 224)),
        "full_min_scale": args.full_min_scale or float(ckpt_args.get("full_min_scale", 0.55)),
        "crop_min_scale": args.crop_min_scale or float(ckpt_args.get("crop_min_scale", 0.75)),
        "crop_padding": args.crop_padding if args.crop_padding is not None else float(ckpt_args.get("crop_padding", 0.0)),
        "dropout": args.dropout if args.dropout is not None else float(ckpt_args.get("dropout", 0.2)),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    torch, nn, Image, models, transforms = _require_training_deps()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = _torch_load(torch, checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise SystemExit(f"Checkpoint does not contain a model state dict: {checkpoint_path}")

    resolved = _resolve_from_checkpoint(args, checkpoint)
    rows = _limit_rows(_read_manifest(Path(args.manifest)), args.limit)
    Dataset = make_dataset_class(torch, Image)
    dataset = Dataset(
        rows,
        path_root=Path(args.path_root),
        full_transform=build_transforms(transforms, resolved["image_size"], train=False, min_scale=resolved["full_min_scale"]),
        crop_transform=build_transforms(transforms, resolved["image_size"], train=False, min_scale=resolved["crop_min_scale"]),
        crop_padding=resolved["crop_padding"],
    )

    loader_args = argparse.Namespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    loader = torch.utils.data.DataLoader(dataset, **_dataloader_kwargs(device, loader_args, shuffle=False))

    model = build_fused_model(
        torch,
        nn,
        models,
        model_name=resolved["model"],
        num_classes=resolved["num_classes"],
        pretrained=False,
        branch_mode=resolved["branch_mode"],
        dropout=resolved["dropout"],
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    class_names = _load_class_names(Path(args.class_prompts) if args.class_prompts else None)

    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    total_top5 = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    prediction_rows: List[Dict[str, object]] = []
    row_offset = 0

    with torch.no_grad():
        for full_images, crop_images, targets in loader:
            full_images = full_images.to(device, non_blocking=True)
            crop_images = crop_images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(full_images, crop_images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)
            topk = min(5, logits.size(1))
            top_values, top_indices = probs.topk(topk, dim=1)
            top1, top5 = accuracy_at_k(torch, logits, targets, topk=(1, 5))

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_seen += batch_size

            batch_targets = targets.cpu().tolist()
            batch_preds = top_indices[:, 0].cpu().tolist()
            all_targets.extend(int(value) for value in batch_targets)
            all_preds.extend(int(value) for value in batch_preds)

            top_indices_cpu = top_indices.cpu().tolist()
            top_values_cpu = top_values.cpu().tolist()
            for batch_index, (target, pred) in enumerate(zip(batch_targets, batch_preds)):
                source_row = rows[row_offset + batch_index]
                top_targets = [int(value) for value in top_indices_cpu[batch_index]]
                prediction_rows.append(
                    {
                        "image_id": source_row.get("image_id", ""),
                        "rel_path": source_row.get("rel_path", ""),
                        "image_path": source_row.get("image_path", ""),
                        "target": int(target),
                        "class_name": source_row.get("class_name") or class_names.get(int(target), ""),
                        "pred_target": int(pred),
                        "pred_class_name": class_names.get(int(pred), ""),
                        "correct_top1": int(int(target) == int(pred)),
                        "correct_top5": int(int(target) in top_targets),
                        "top5_targets": "|".join(str(value) for value in top_targets),
                        "top5_scores": "|".join(f"{float(value):.6f}" for value in top_values_cpu[batch_index]),
                    }
                )
            row_offset += batch_size

    metrics: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(Path(args.manifest)),
        "num_examples": total_seen,
        "model": resolved["model"],
        "branch_mode": resolved["branch_mode"],
        "image_size": resolved["image_size"],
        "loss": total_loss / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics"),
    }
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score

        metrics["macro_f1"] = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(all_targets, all_preds))
    except Exception as exc:
        metrics["metric_warning"] = str(exc)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.output_predictions:
        _write_csv(
            Path(args.output_predictions),
            prediction_rows,
            [
                "image_id",
                "rel_path",
                "image_path",
                "target",
                "class_name",
                "pred_target",
                "pred_class_name",
                "correct_top1",
                "correct_top5",
                "top5_targets",
                "top5_scores",
            ],
        )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fused NABirds checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="reports/milestone2/nabirds_test.csv")
    parser.add_argument("--class-prompts", default="reports/milestone2/nabirds_class_prompts.csv")
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-predictions", default="")
    parser.add_argument("--model", default="", help="Override model architecture; defaults to checkpoint args.")
    parser.add_argument("--branch-mode", choices=["", "shared", "two_branch"], default="", help="Override branch mode.")
    parser.add_argument("--num-classes", type=int, default=0, help="Override class count.")
    parser.add_argument("--image-size", type=int, default=0, help="Override eval image size.")
    parser.add_argument("--full-min-scale", type=float, default=0.0)
    parser.add_argument("--crop-min-scale", type=float, default=0.0)
    parser.add_argument("--crop-padding", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    metrics = evaluate(parse_args())
    print("NABirds fused checkpoint evaluation")
    for key, value in metrics.items():
        if key == "checkpoint_val_metrics":
            continue
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
