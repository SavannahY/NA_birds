#!/usr/bin/env python3
"""Evaluate open-source VLM zero-shot classification on NABirds.

The script consumes the Milestone 2 test manifest and class prompt CSV:

  - reports/milestone2/nabirds_test.csv
  - reports/milestone2/nabirds_class_prompts.csv

It intentionally keeps torch/transformers/Pillow imports inside the runtime
path so the file remains importable and syntax-checkable without VLM
dependencies installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


DEFAULT_MANIFEST = "reports/milestone2/nabirds_test.csv"
DEFAULT_PROMPTS = "reports/milestone2/nabirds_class_prompts.csv"
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_FALLBACK_MODEL = "google/siglip-base-patch16-224"
EXPECTED_NABIRDS_CLASSES = 555


def _read_csv(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(
            f"Missing required CSV: {path}\n"
            "Build Milestone 2 manifests first, for example:\n"
            "  python3 scripts/build_nabirds_manifests.py --dataset-root nabirds --out-dir reports/milestone2"
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} is empty or missing a header row.")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} is missing required column(s): {', '.join(missing)}")
        return list(reader)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _require_runtime_deps():
    missing: List[str] = []

    try:
        import torch
    except ImportError:
        torch = None
        missing.append("torch")

    try:
        from PIL import Image
    except ImportError:
        Image = None
        missing.append("pillow")

    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        AutoModel = None
        AutoProcessor = None
        missing.append("transformers")

    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise SystemExit(
            "VLM zero-shot evaluation dependencies are missing: "
            f"{unique_missing}.\n"
            "Install a compatible PyTorch build plus transformers and Pillow before running evaluation.\n"
            "Example CPU install:\n"
            "  python3 -m pip install torch transformers pillow\n"
            "For CUDA or Apple Silicon acceleration, use the PyTorch install selector for the correct torch wheel."
        )

    return torch, Image, AutoModel, AutoProcessor


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but torch.cuda.is_available() is false.")
    if device.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise SystemExit("Requested --device mps, but torch.backends.mps.is_available() is false.")
    return device


def _resolve_dtype(torch: Any, requested: str, device: Any) -> Optional[Any]:
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[requested]


def _move_inputs(batch: Mapping[str, Any], device: Any, dtype: Optional[Any]) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            if dtype is not None and hasattr(value, "is_floating_point") and value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _normalize(torch: Any, tensor: Any) -> Any:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _pooled_feature_tensor(output: Any, feature_name: str) -> Any:
    """Return a 2D embedding tensor from HF tensor or ModelOutput variants."""
    if hasattr(output, "norm"):
        return output

    for attribute in ("pooler_output", "text_embeds", "image_embeds"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value

    value = getattr(output, "last_hidden_state", None)
    if value is not None:
        ndim = getattr(value, "ndim", None)
        if ndim == 2:
            return value
        if ndim == 3:
            return value[:, 0]

    if isinstance(output, (tuple, list)):
        for value in output:
            ndim = getattr(value, "ndim", None)
            if ndim == 2:
                return value
        if output:
            return output[0]

    raise TypeError(f"Could not extract {feature_name} features from output type {type(output).__name__}.")


def _batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _progress(items: Iterable[Any], total: int, label: str, enabled: bool) -> Iterable[Any]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm

        return tqdm(items, total=total, desc=label)
    except Exception:
        return items


def _load_prompt_rows(path: Path, expected_num_classes: int) -> Tuple[List[int], List[str], Dict[int, List[str]], int]:
    rows = _read_csv(path, required_columns=("target", "prompt"))
    if not rows:
        raise SystemExit(f"{path}: no prompt rows to evaluate.")
    prompts_by_target: Dict[int, List[str]] = defaultdict(list)
    class_name_by_target: Dict[int, str] = {}

    for row in rows:
        try:
            target = int(row["target"])
        except ValueError as exc:
            raise SystemExit(f"{path}: invalid integer target {row['target']!r}") from exc

        prompt = row["prompt"].strip()
        if not prompt:
            raise SystemExit(f"{path}: empty prompt for target {target}")
        prompts_by_target[target].append(prompt)
        if "class_name" in row and row["class_name"].strip():
            class_name_by_target[target] = row["class_name"].strip()

    class_targets = sorted(prompts_by_target)
    if expected_num_classes and len(class_targets) != expected_num_classes:
        raise SystemExit(
            f"{path}: expected {expected_num_classes} unique targets, found {len(class_targets)}. "
            "Pass --expected-num-classes 0 to disable this check."
        )

    class_names = [class_name_by_target.get(target, str(target)) for target in class_targets]
    return class_targets, class_names, dict(prompts_by_target), len(rows)


def _load_manifest_rows(path: Path, limit: Optional[int], class_targets: Sequence[int]) -> List[Dict[str, str]]:
    rows = _read_csv(path, required_columns=("image_path", "target"))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"{path}: no rows to evaluate.")

    valid_targets = set(class_targets)
    manifest_targets = set()
    for row in rows:
        try:
            manifest_targets.add(int(row["target"]))
        except ValueError as exc:
            raise SystemExit(f"{path}: invalid integer target {row['target']!r}") from exc

    unknown_targets = sorted(target for target in manifest_targets if target not in valid_targets)
    if unknown_targets:
        preview = ", ".join(str(value) for value in unknown_targets[:10])
        raise SystemExit(f"{path}: manifest contains target(s) not present in prompts: {preview}")
    return rows


def _resolve_image_path(row: Mapping[str, str], dataset_root: Path) -> Path:
    image_path = Path(row["image_path"]).expanduser()
    if image_path.is_absolute() and image_path.is_file():
        return image_path
    if not image_path.is_absolute() and image_path.is_file():
        return image_path

    rel_path = row.get("rel_path", "").strip()
    if rel_path:
        candidate = dataset_root / "images" / rel_path
        if candidate.is_file():
            return candidate

    return image_path


def _open_image(Image: Any, row: Mapping[str, str], dataset_root: Path, input_mode: str) -> Any:
    path = _resolve_image_path(row, dataset_root)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if input_mode == "bbox":
        required = ("clip_x", "clip_y", "clip_w", "clip_h")
        missing = [column for column in required if column not in row or row[column] == ""]
        if missing:
            raise ValueError(f"{path}: bbox mode requires manifest column(s): {', '.join(missing)}")
        x = int(row["clip_x"])
        y = int(row["clip_y"])
        w = int(row["clip_w"])
        h = int(row["clip_h"])
        if w <= 0 or h <= 0:
            raise ValueError(f"{path}: invalid clipped bbox width/height ({w}, {h})")
        image = image.crop((x, y, x + w, y + h))
    return image


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> float:
    true_positive = {label: 0 for label in labels}
    false_positive = {label: 0 for label in labels}
    false_negative = {label: 0 for label in labels}

    for target, pred in zip(y_true, y_pred):
        if target == pred:
            true_positive[target] += 1
        else:
            false_negative[target] += 1
            false_positive[pred] += 1

    f1_values: List[float] = []
    for label in labels:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]
        denominator = (2 * tp) + fp + fn
        f1_values.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return sum(f1_values) / len(f1_values)


def _encode_text_features(
    torch: Any,
    model: Any,
    processor: Any,
    prompts_by_target: Mapping[int, Sequence[str]],
    class_targets: Sequence[int],
    device: Any,
    dtype: Optional[Any],
    text_batch_size: int,
    aggregation: str,
    show_progress: bool,
) -> Any:
    if not hasattr(model, "get_text_features"):
        raise SystemExit(
            "The loaded model does not expose get_text_features(). "
            "Use a SigLIP/SigLIP2 checkpoint supported by transformers AutoModel."
        )

    flat_prompts: List[str] = []
    target_spans: Dict[int, Tuple[int, int]] = {}
    for target in class_targets:
        start = len(flat_prompts)
        flat_prompts.extend(prompts_by_target[target])
        target_spans[target] = (start, len(flat_prompts))

    features: List[Any] = []
    batches = list(_batched(flat_prompts, text_batch_size))
    iterator = _progress(batches, total=len(batches), label="encoding text", enabled=show_progress)
    with torch.inference_mode():
        for prompt_batch in iterator:
            inputs = processor(
                text=list(prompt_batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = _move_inputs(inputs, device=device, dtype=dtype)
            text_features = _pooled_feature_tensor(model.get_text_features(**inputs), "text")
            features.append(_normalize(torch, text_features).detach().cpu())

    all_prompt_features = torch.cat(features, dim=0)
    class_features: List[Any] = []
    for target in class_targets:
        start, end = target_spans[target]
        prompt_features = all_prompt_features[start:end]
        if aggregation == "first":
            class_feature = prompt_features[0]
        else:
            class_feature = prompt_features.mean(dim=0)
        class_features.append(_normalize(torch, class_feature.unsqueeze(0)).squeeze(0))

    return torch.stack(class_features, dim=0).to(device=device, dtype=dtype)


def _encode_image_features(
    torch: Any,
    model: Any,
    processor: Any,
    Image: Any,
    rows: Sequence[Mapping[str, str]],
    dataset_root: Path,
    input_mode: str,
    device: Any,
    dtype: Optional[Any],
) -> Any:
    if not hasattr(model, "get_image_features"):
        raise SystemExit(
            "The loaded model does not expose get_image_features(). "
            "Use a SigLIP/SigLIP2 checkpoint supported by transformers AutoModel."
        )

    images = [_open_image(Image, row, dataset_root=dataset_root, input_mode=input_mode) for row in rows]
    inputs = processor(images=images, return_tensors="pt")
    inputs = _move_inputs(inputs, device=device, dtype=dtype)
    with torch.inference_mode():
        image_features = _pooled_feature_tensor(model.get_image_features(**inputs), "image")
    return _normalize(torch, image_features)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    torch, Image, AutoModel, AutoProcessor = _require_runtime_deps()

    prompt_path = Path(args.prompts)
    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)

    class_targets, class_names, prompts_by_target, prompt_rows = _load_prompt_rows(
        prompt_path,
        expected_num_classes=args.expected_num_classes,
    )
    rows = _load_manifest_rows(manifest_path, limit=args.limit, class_targets=class_targets)

    device = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype, device)
    model_name = args.model

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError as exc:
        raise SystemExit(
            f"Could not load model or processor for {model_name!r}.\n"
            "If this is the first run, the checkpoint may need to be downloaded from Hugging Face.\n"
            f"Fallback SigLIP checkpoint configured for this project: {DEFAULT_FALLBACK_MODEL}"
        ) from exc

    model.eval()
    model.to(device=device)
    if dtype is not None:
        model.to(dtype=dtype)

    started = time.time()
    text_features = _encode_text_features(
        torch=torch,
        model=model,
        processor=processor,
        prompts_by_target=prompts_by_target,
        class_targets=class_targets,
        device=device,
        dtype=dtype,
        text_batch_size=args.text_batch_size,
        aggregation=args.prompt_aggregation,
        show_progress=not args.no_progress,
    )

    target_to_col = {target: index for index, target in enumerate(class_targets)}
    y_true: List[int] = []
    y_pred: List[int] = []
    prediction_rows: List[Dict[str, Any]] = []
    top1_correct = 0
    top5_correct = 0
    top_k = min(5, len(class_targets))

    batches = list(_batched(rows, args.batch_size))
    iterator = _progress(batches, total=len(batches), label="scoring images", enabled=not args.no_progress)
    with torch.inference_mode():
        for batch_rows in iterator:
            image_features = _encode_image_features(
                torch=torch,
                model=model,
                processor=processor,
                Image=Image,
                rows=batch_rows,
                dataset_root=dataset_root,
                input_mode=args.input_mode,
                device=device,
                dtype=dtype,
            )
            scores = image_features @ text_features.T
            _, top_indices = scores.topk(top_k, dim=1)
            top_indices_cpu = top_indices.detach().cpu().tolist()

            for row, predicted_cols in zip(batch_rows, top_indices_cpu):
                target = int(row["target"])
                target_col = target_to_col[target]
                pred_col = int(predicted_cols[0])
                pred_target = class_targets[pred_col]
                pred_top5_targets = [class_targets[int(col)] for col in predicted_cols]

                top1_correct += int(pred_col == target_col)
                top5_correct += int(target_col in predicted_cols)
                y_true.append(target)
                y_pred.append(pred_target)
                prediction_rows.append(
                    {
                        "image_id": row.get("image_id", ""),
                        "rel_path": row.get("rel_path", ""),
                        "image_path": row.get("image_path", ""),
                        "target": target,
                        "class_name": row.get("class_name", ""),
                        "pred_target": pred_target,
                        "pred_class_name": class_names[pred_col],
                        "top5_targets": "|".join(str(value) for value in pred_top5_targets),
                        "correct_top1": int(pred_col == target_col),
                        "correct_top5": int(target_col in predicted_cols),
                    }
                )

    elapsed = time.time() - started
    num_samples = len(y_true)
    metrics: Dict[str, Any] = {
        "model": model_name,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", "") if dtype is not None else None,
        "manifest": str(manifest_path),
        "prompts": str(prompt_path),
        "input_mode": args.input_mode,
        "prompt_aggregation": args.prompt_aggregation,
        "num_samples": num_samples,
        "num_classes": len(class_targets),
        "num_prompt_rows": prompt_rows,
        "top1_accuracy": top1_correct / num_samples,
        "top5_accuracy": top5_correct / num_samples,
        "macro_f1": _macro_f1(y_true, y_pred, labels=class_targets),
        "elapsed_seconds": round(elapsed, 3),
    }

    if args.limit is not None:
        metrics["limit"] = args.limit
        metrics["note"] = "Metrics are smoke-test metrics because --limit was used."

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(metrics)
        payload["class_targets"] = class_targets
        payload["class_names"] = class_names
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
        description="Evaluate SigLIP/SigLIP2 zero-shot image-text similarity on NABirds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="NABirds test manifest CSV.")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="NABirds class prompt CSV.")
    parser.add_argument("--dataset-root", default="nabirds", help="Dataset root used to resolve rel_path fallback.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face SigLIP/SigLIP2 model id or local path.")
    parser.add_argument("--input-mode", choices=("full", "bbox"), default="full", help="Use full image or clipped bbox crop.")
    parser.add_argument("--prompt-aggregation", choices=("mean", "first"), default="mean", help="How to combine multiple prompts per class.")
    parser.add_argument("--expected-num-classes", type=int, default=EXPECTED_NABIRDS_CLASSES, help="Expected unique targets in prompt CSV; use 0 to disable.")
    parser.add_argument("--batch-size", type=_positive_int, default=32, help="Image batch size.")
    parser.add_argument("--text-batch-size", type=_positive_int, default=256, help="Prompt text batch size.")
    parser.add_argument("--limit", type=_positive_int, default=None, help="Evaluate only the first N manifest rows for smoke tests.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto", help="Model/input floating dtype.")
    parser.add_argument("--output-json", default=None, help="Optional path to write metrics JSON.")
    parser.add_argument("--output-predictions", default=None, help="Optional path to write per-image predictions CSV.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars when tqdm is installed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        metrics = evaluate(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"VLM zero-shot evaluation failed: {exc}") from exc

    print("NABirds VLM zero-shot evaluation")
    for key in (
        "model",
        "device",
        "dtype",
        "input_mode",
        "prompt_aggregation",
        "num_samples",
        "num_classes",
        "num_prompt_rows",
        "top1_accuracy",
        "top5_accuracy",
        "macro_f1",
        "elapsed_seconds",
        "output_json",
        "output_predictions",
        "note",
    ):
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float) and key.endswith(("accuracy", "f1")):
                value = f"{value:.6f}"
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
