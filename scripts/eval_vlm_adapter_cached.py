#!/usr/bin/env python3
"""Evaluate a cached-feature VLM adapter checkpoint on saved image features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_vlm_cached_features import (  # noqa: E402
    _balanced_accuracy,
    _load_image_payloads,
    _macro_f1,
    _metadata_value,
    _resolve_device,
    _torch_load,
    _validate_text_payload,
)
from scripts.train_vlm_adapter import build_alignment_model  # noqa: E402


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit("Cached adapter evaluation requires torch.") from exc
    return torch, nn


def _arg_value(args: Mapping[str, Any], key: str, default: Any) -> Any:
    value = args.get(key, default)
    return default if value is None else value


def _has_class_bias(state_dict: Mapping[str, Any]) -> bool:
    return "class_bias" in state_dict


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    torch, nn = _require_torch()
    started = time.time()
    checkpoint_path = Path(args.checkpoint)
    text_path = Path(args.text_features)
    image_paths = [Path(value) for value in args.image_features]

    checkpoint = _torch_load(torch, checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise SystemExit(f"{checkpoint_path} did not contain a checkpoint dictionary.")
    state_dict = checkpoint.get("adapter")
    if not isinstance(state_dict, Mapping):
        raise SystemExit(f"{checkpoint_path} is missing adapter state_dict.")

    text_payload = _torch_load(torch, text_path)
    if not isinstance(text_payload, Mapping):
        raise SystemExit(f"{text_path} did not contain a checkpoint dictionary.")
    _validate_text_payload(text_payload, text_path)
    image_payload = _load_image_payloads(torch, image_paths)

    class_targets = [int(value) for value in checkpoint.get("class_targets", text_payload["class_targets"])]
    class_names = [str(value) for value in checkpoint.get("class_names", text_payload["class_names"])]
    text_targets = [int(value) for value in text_payload["class_targets"]]
    if class_targets != text_targets:
        raise SystemExit("Checkpoint class_targets do not match text feature class_targets.")

    image_features = image_payload["image_features"].float()
    image_targets = [int(value) for value in image_payload["targets"]]
    class_text_features = text_payload["class_text_features"].float()
    metadata = image_payload["metadata"]

    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, Mapping):
        checkpoint_args = {}
    image_dim = int(checkpoint.get("image_feature_dim", image_features.shape[1]))
    text_dim = int(checkpoint.get("text_feature_dim", class_text_features.shape[1]))

    device = _resolve_device(torch, args.device)
    adapter = build_alignment_model(
        torch=torch,
        nn=nn,
        image_dim=image_dim,
        text_dim=text_dim,
        projection_dim=int(_arg_value(checkpoint_args, "projection_dim", 0)),
        adapter_mode=str(_arg_value(checkpoint_args, "adapter_mode", "residual")),
        adapter_hidden_dim=int(_arg_value(checkpoint_args, "adapter_hidden_dim", 0)),
        dropout=float(_arg_value(checkpoint_args, "dropout", 0.0)),
        shared_adapter=bool(_arg_value(checkpoint_args, "shared_adapter", False)),
        init_temperature=float(_arg_value(checkpoint_args, "init_temperature", 0.07)),
        learnable_temperature=not bool(_arg_value(checkpoint_args, "fixed_temperature", False)),
        class_bias=bool(_arg_value(checkpoint_args, "class_bias", False)) or _has_class_bias(state_dict),
        num_classes=len(class_targets),
    ).to(device=device)
    adapter.load_state_dict(state_dict)
    adapter.eval()

    target_to_col = {target: index for index, target in enumerate(class_targets)}
    missing_targets = sorted(set(image_targets) - set(target_to_col))
    if missing_targets:
        preview = ", ".join(str(value) for value in missing_targets[:10])
        raise SystemExit(f"Image targets missing from checkpoint/text classes: {preview}")

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
            logits, _image_z, _text_z = adapter(features, text)
            _values, top_indices = logits.topk(top_k, dim=1)
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
    metrics: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "image_features": [str(path) for path in image_paths],
        "text_features": str(text_path),
        "num_samples": num_samples,
        "num_classes": len(class_targets),
        "top1_accuracy": top1_correct / num_samples if num_samples else 0.0,
        "top5_accuracy": top5_correct / num_samples if num_samples else 0.0,
        "macro_f1": _macro_f1(y_true, y_pred, class_targets),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred, class_targets),
        "elapsed_seconds": round(time.time() - started, 3),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    output_predictions = Path(args.output_predictions)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(prediction_rows[0]) if prediction_rows else []
    with output_predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    metrics["output_json"] = str(output_json)
    metrics["output_predictions"] = str(output_predictions)
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a cached VLM adapter checkpoint against cached image/text features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-features", nargs="+", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    metrics = evaluate(parse_args(argv))
    print("NABirds cached VLM adapter evaluation")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
