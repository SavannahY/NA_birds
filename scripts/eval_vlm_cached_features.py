#!/usr/bin/env python3
"""Evaluate NABirds VLM zero-shot predictions from cached image/text features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Cached feature evaluation requires torch.") from exc
    return torch


def _torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_image_payload(payload: Mapping[str, Any], path: Path) -> None:
    for key in ("image_features", "targets", "metadata"):
        if key not in payload:
            raise SystemExit(f"{path} missing required key {key!r}.")
    if getattr(payload["image_features"], "ndim", None) != 2:
        raise SystemExit(f"{path}: image_features must be 2D.")
    if getattr(payload["targets"], "ndim", None) != 1:
        raise SystemExit(f"{path}: targets must be 1D.")
    if int(payload["image_features"].shape[0]) != int(payload["targets"].shape[0]):
        raise SystemExit(f"{path}: feature/target row count mismatch.")


def _validate_text_payload(payload: Mapping[str, Any], path: Path) -> None:
    for key in ("class_text_features", "class_targets", "class_names"):
        if key not in payload:
            raise SystemExit(f"{path} missing required key {key!r}.")
    if getattr(payload["class_text_features"], "ndim", None) != 2:
        raise SystemExit(f"{path}: class_text_features must be 2D.")


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> float:
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}
    for target, pred in zip(y_true, y_pred):
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


def _balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> float:
    recalls = []
    for label in labels:
        total = sum(target == label for target in y_true)
        if total == 0:
            continue
        correct = sum(target == label and pred == label for target, pred in zip(y_true, y_pred))
        recalls.append(correct / total)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _metadata_value(metadata: Mapping[str, Sequence[Any]], key: str, index: int) -> str:
    values = metadata.get(key, [])
    if index >= len(values):
        return ""
    return str(values[index])


def _load_image_payloads(torch: Any, paths: Sequence[Path]) -> Dict[str, Any]:
    features = []
    targets: List[int] = []
    metadata: Dict[str, List[Any]] = {}
    for path in paths:
        payload = _torch_load(torch, path)
        if not isinstance(payload, Mapping):
            raise SystemExit(f"{path} did not contain a checkpoint dictionary.")
        _validate_image_payload(payload, path)
        features.append(payload["image_features"].float())
        targets.extend(int(value) for value in payload["targets"].cpu().tolist())
        for key, values in payload["metadata"].items():
            metadata.setdefault(key, []).extend(list(values))
    return {"image_features": torch.cat(features, dim=0), "targets": targets, "metadata": metadata}


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    torch = _require_torch()
    started = time.time()
    image_paths = [Path(value) for value in args.image_features]
    text_path = Path(args.text_features)
    text_payload = _torch_load(torch, text_path)
    if not isinstance(text_payload, Mapping):
        raise SystemExit(f"{text_path} did not contain a checkpoint dictionary.")
    image_payload = _load_image_payloads(torch, image_paths)
    _validate_text_payload(text_payload, text_path)

    image_features = image_payload["image_features"].float()
    image_targets = [int(value) for value in image_payload["targets"]]
    metadata = image_payload["metadata"]
    class_text_features = text_payload["class_text_features"].float()
    class_targets = [int(value) for value in text_payload["class_targets"]]
    class_names = [str(value) for value in text_payload["class_names"]]

    if int(image_features.shape[1]) != int(class_text_features.shape[1]):
        raise SystemExit(
            f"Feature dimension mismatch: image dim {int(image_features.shape[1])}, "
            f"text dim {int(class_text_features.shape[1])}."
        )
    target_to_col = {target: index for index, target in enumerate(class_targets)}
    missing_targets = sorted(set(image_targets) - set(target_to_col))
    if missing_targets:
        preview = ", ".join(str(value) for value in missing_targets[:10])
        raise SystemExit(f"Image targets missing from text features: {preview}")

    device = _resolve_device(torch, args.device)
    text = class_text_features.to(device=device)
    top_k = min(5, len(class_targets))
    y_true: List[int] = []
    y_pred: List[int] = []
    prediction_rows: List[Dict[str, Any]] = []
    top1_correct = 0
    top5_correct = 0

    with torch.inference_mode():
        for start in range(0, len(image_targets), args.batch_size):
            end = min(start + args.batch_size, len(image_targets))
            features = image_features[start:end].to(device=device)
            scores = features @ text.T
            _values, top_indices = scores.topk(top_k, dim=1)
            for offset, predicted_cols in enumerate(top_indices.cpu().tolist()):
                index = start + offset
                target = image_targets[index]
                target_col = target_to_col[target]
                pred_col = int(predicted_cols[0])
                pred_target = class_targets[pred_col]
                pred_top5_targets = [class_targets[int(col)] for col in predicted_cols]
                top1 = int(pred_col == target_col)
                top5 = int(target_col in predicted_cols)

                top1_correct += top1
                top5_correct += top5
                y_true.append(target)
                y_pred.append(pred_target)
                prediction_rows.append(
                    {
                        "image_id": _metadata_value(metadata, "image_id", index),
                        "rel_path": _metadata_value(metadata, "rel_path", index),
                        "image_path": _metadata_value(metadata, "image_path", index),
                        "target": target,
                        "class_name": _metadata_value(metadata, "class_name", index),
                        "pred_target": pred_target,
                        "pred_class_name": class_names[pred_col],
                        "top5_targets": "|".join(str(value) for value in pred_top5_targets),
                        "correct_top1": top1,
                        "correct_top5": top5,
                    }
                )

    num_samples = len(y_true)
    labels = class_targets
    metrics: Dict[str, Any] = {
        "image_features": [str(path) for path in image_paths],
        "text_features": str(text_path),
        "num_samples": num_samples,
        "num_classes": len(class_targets),
        "top1_accuracy": top1_correct / num_samples if num_samples else 0.0,
        "top5_accuracy": top5_correct / num_samples if num_samples else 0.0,
        "macro_f1": _macro_f1(y_true, y_pred, labels),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred, labels),
        "elapsed_seconds": round(time.time() - started, 3),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics["output_json"] = str(output_path)

    if args.output_predictions:
        output_path = Path(args.output_predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(prediction_rows[0]) if prediction_rows else []
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)
        metrics["output_predictions"] = str(output_path)

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score NABirds cached image features against cached text features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-features", nargs="+", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    metrics = evaluate(parse_args(argv))
    print("NABirds cached VLM feature evaluation")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
