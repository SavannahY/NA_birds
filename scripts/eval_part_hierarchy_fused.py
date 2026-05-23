#!/usr/bin/env python3
"""Evaluate a part-guided hierarchy-fused NABirds checkpoint.

This evaluator intentionally imports the matching training script lazily so it
can be syntax-checked before torch/torchvision and Worker A's trainer exist.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TRAIN_MODULE = "scripts.train_part_hierarchy_fused"
PREDICTION_FIELDNAMES = [
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
]


def _load_train_module():
    try:
        return importlib.import_module(TRAIN_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == TRAIN_MODULE:
            raise SystemExit(
                f"Missing {TRAIN_MODULE}. Worker A's scripts/train_part_hierarchy_fused.py "
                "must be present before running this evaluator."
            ) from exc
        raise SystemExit(f"Could not import {TRAIN_MODULE}: missing dependency {exc.name!r}.") from exc
    except ImportError as exc:
        raise SystemExit(f"Could not import {TRAIN_MODULE}: {exc}") from exc


def _symbol(module: Any, names: Sequence[str], *, required: bool = True) -> Optional[Any]:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    if required:
        raise SystemExit(
            f"{TRAIN_MODULE} does not expose any of: {', '.join(names)}. "
            "Update Worker A's trainer or add an alias so eval can reuse the training implementation."
        )
    return None


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_var_kwargs:
        return func(*args, **kwargs)
    positional_names = [
        name
        for name, param in parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ][: len(args)]
    supported = {key: value for key, value in kwargs.items() if key in parameters and key not in positional_names}
    return func(*args, **supported)


def _torch_load(torch, path: Path, map_location: str) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _read_manifest_fallback(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Manifest is empty: {path}")
    required = {"target"}
    if "image_path" not in rows[0] and "rel_path" not in rows[0]:
        required.add("image_path")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise SystemExit(f"Manifest {path} is missing required columns: {', '.join(missing)}")
    return rows


def _limit_rows_fallback(rows: List[Dict[str, str]], limit: Optional[int]) -> List[Dict[str, str]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


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


def _require_training_deps(module: Any):
    require_deps = getattr(module, "_require_training_deps", None)
    if require_deps is not None:
        return require_deps()
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. Install torch, torchvision, and pillow first.\n"
            "Example: pip install torch torchvision pillow scikit-learn"
        ) from exc
    return torch, nn, Image, models, transforms


def _resolve_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not mps_available:
            print("MPS requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
    return torch.device(requested)


def _model_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if value is not None:
            return value
    raise SystemExit("Checkpoint does not contain a model state dict under model, model_state_dict, or state_dict.")


def _checkpoint_args(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    args = checkpoint.get("args") or checkpoint.get("config") or {}
    if isinstance(args, argparse.Namespace):
        return vars(args)
    if isinstance(args, Mapping):
        return dict(args)
    return {}


def _default_resolved(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    ckpt_args = _checkpoint_args(checkpoint)
    config = checkpoint.get("config") or {}
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    if not isinstance(config, Mapping):
        config = {}

    def pick(name: str, default: Any) -> Any:
        return ckpt_args.get(name, config.get(name, default))

    return {
        "model": args.model or pick("model", "resnet50"),
        "num_classes": int(pick("num_classes", 555)),
        "num_views": args.num_views or int(pick("num_views", 5)),
        "image_size": args.image_size or int(pick("image_size", 224)),
        "fusion": pick("fusion", "concat"),
        "dropout": float(pick("dropout", 0.2)),
        "crop_padding": float(pick("crop_padding", 0.0)),
        "full_min_scale": float(pick("full_min_scale", 0.55)),
        "crop_min_scale": float(pick("crop_min_scale", 0.75)),
        "hierarchy_loss_weight": float(pick("hierarchy_loss_weight", 0.0)),
        "hierarchy_loss_enabled": bool(config.get("hierarchy_loss_enabled", False)),
        "coarse_class_count": int(config.get("coarse_class_count", 0) or 0),
        "dataset_root": str(Path(args.dataset_root)),
        "path_root": str(Path(args.path_root)),
    }


def _resolve_from_checkpoint(module: Any, args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    resolver = _symbol(
        module,
        ("resolve_eval_args", "_resolve_eval_args", "resolve_args", "_resolve_args", "_resolve_from_checkpoint"),
        required=False,
    )
    if resolver is None:
        return _default_resolved(args, checkpoint)
    resolved = _call_with_supported_kwargs(
        resolver,
        args,
        checkpoint,
        checkpoint_args=_checkpoint_args(checkpoint),
        eval_args=args,
    )
    if isinstance(resolved, argparse.Namespace):
        resolved = vars(resolved)
    if not isinstance(resolved, Mapping):
        raise SystemExit(f"{resolver.__name__} returned {type(resolved).__name__}, expected mapping or Namespace.")
    merged = _default_resolved(args, checkpoint)
    merged.update(dict(resolved))
    if args.model:
        merged["model"] = args.model
    if args.num_views:
        merged["num_views"] = args.num_views
    if args.image_size:
        merged["image_size"] = args.image_size
    merged["dataset_root"] = str(Path(args.dataset_root))
    merged["path_root"] = str(Path(args.path_root))
    return merged


def _build_transforms(module: Any, transforms: Any, resolved: Mapping[str, Any]):
    builder = _symbol(module, ("build_eval_transforms", "build_transforms", "make_transforms"))
    try:
        return _call_with_supported_kwargs(
            builder,
            transforms,
            int(resolved["image_size"]),
            train=False,
            image_size=int(resolved["image_size"]),
            num_views=int(resolved["num_views"]),
        )
    except TypeError as exc:
        raise SystemExit(f"Could not build eval transforms from {builder.__name__}: {exc}") from exc


def _make_dataset(module: Any, torch: Any, Image: Any, rows: List[Dict[str, str]], transform: Any, resolved: Mapping[str, Any]):
    dataset_factory = _symbol(
        module,
        (
            "make_dataset_class",
            "make_part_hierarchy_dataset_class",
            "make_part_guided_dataset_class",
            "PartHierarchyFusedDataset",
            "PartGuidedHierarchyFusedDataset",
            "NABirdsPartHierarchyDataset",
        ),
    )
    Dataset = dataset_factory
    if inspect.isfunction(dataset_factory):
        try:
            Dataset = _call_with_supported_kwargs(dataset_factory, torch, Image, torch=torch, Image=Image)
        except TypeError:
            pass

    kwargs = {
        "rows": rows,
        "transform": transform,
        "transforms": transform,
        "eval_transform": transform,
        "image_transform": transform,
        "path_root": Path(str(resolved["path_root"])),
        "dataset_root": Path(str(resolved["dataset_root"])),
        "num_views": int(resolved["num_views"]),
    }
    try:
        return _call_with_supported_kwargs(Dataset, **kwargs)
    except TypeError as exc:
        raise SystemExit(
            f"Could not instantiate dataset {getattr(Dataset, '__name__', Dataset)}. "
            "Expected Worker A's dataset to accept rows plus transform/path_root/dataset_root/num_views-compatible args. "
            f"Original error: {exc}"
        ) from exc


def _build_model(module: Any, torch: Any, nn: Any, models: Any, resolved: Mapping[str, Any]):
    builder = _symbol(
        module,
        (
            "build_part_hierarchy_model",
            "build_part_hierarchy_fused_model",
            "build_hierarchy_fused_model",
            "build_part_guided_model",
            "build_model",
            "build_fused_model",
        ),
    )
    ckpt_config = dict(resolved)
    try:
        return _call_with_supported_kwargs(
            builder,
            torch,
            nn,
            models,
            torch=torch,
            nn=nn,
            models=models,
            model_name=str(resolved["model"]),
            model=str(resolved["model"]),
            num_classes=int(resolved["num_classes"]),
            num_views=int(resolved["num_views"]),
            image_size=int(resolved["image_size"]),
            pretrained=False,
            freeze_backbone=False,
            config=ckpt_config,
            args=argparse.Namespace(**ckpt_config),
        )
    except TypeError as exc:
        raise SystemExit(f"Could not build model with {builder.__name__}: {exc}") from exc


def _accuracy_at_k(torch: Any, logits: Any, targets: Any, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    values = []
    for requested_k in topk:
        k = min(requested_k, logits.size(1))
        values.append(correct[:k].reshape(-1).float().sum(0).item() / targets.size(0))
    return values


def _forward_logits(model: Any, views: Any) -> Any:
    outputs = model(views)
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def _forward_logits_with_optional_tta(model: Any, views: Any, *, tta_hflip: bool) -> Any:
    logits = _forward_logits(model, views)
    if not tta_hflip:
        return logits
    if len(views.shape) != 5:
        raise SystemExit(f"--tta-hflip expects views shaped [B,V,C,H,W], got {list(views.shape)}.")
    flipped_logits = _forward_logits(model, views.flip(dims=(-1,)))
    return (logits + flipped_logits) * 0.5


def _metrics_from_predictions(targets: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
    labels = sorted(set(targets) | set(preds))
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score

        return {
            "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        }
    except Exception:
        pass

    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}
    for target, pred in zip(targets, preds):
        if target == pred:
            tp[target] = tp.get(target, 0) + 1
        else:
            fn[target] = fn.get(target, 0) + 1
            fp[pred] = fp.get(pred, 0) + 1
    f1_values = []
    recalls = []
    for label in labels:
        denom = (2 * tp.get(label, 0)) + fp.get(label, 0) + fn.get(label, 0)
        f1_values.append(0.0 if denom == 0 else (2 * tp.get(label, 0)) / denom)
        total = sum(target == label for target in targets)
        if total:
            correct = sum(target == label and pred == label for target, pred in zip(targets, preds))
            recalls.append(correct / total)
    return {
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
    }


def _batch_to_device(batch: Any, device: Any) -> Tuple[Tuple[Any, ...], Any]:
    if isinstance(batch, Mapping):
        targets = batch.get("target", batch.get("targets", batch.get("label", batch.get("labels"))))
        if targets is None:
            raise SystemExit("Dataset batch mapping must contain target/targets/label/labels.")
        inputs = []
        for key, value in batch.items():
            if key in {"target", "targets", "label", "labels"}:
                continue
            inputs.append(value.to(device, non_blocking=True) if hasattr(value, "to") else value)
        return tuple(inputs), targets.to(device, non_blocking=True)

    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise SystemExit("Dataset batches must be tuple/list inputs plus targets, or a mapping with targets.")
    *inputs, targets = batch
    device_inputs = tuple(value.to(device, non_blocking=True) if hasattr(value, "to") else value for value in inputs)
    return device_inputs, targets.to(device, non_blocking=True)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    module = _load_train_module()
    torch, nn, Image, models, transforms = _require_training_deps(module)
    device = _resolve_device(torch, args.device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = _torch_load(torch, checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise SystemExit(f"Checkpoint is {type(checkpoint).__name__}, expected a mapping: {checkpoint_path}")
    model_state = _model_state_dict(checkpoint)
    resolved = _resolve_from_checkpoint(module, args, checkpoint)

    read_manifest = getattr(module, "_read_manifest", None) or getattr(module, "read_manifest", None) or _read_manifest_fallback
    limit_rows = getattr(module, "_limit_rows", None) or getattr(module, "limit_rows", None) or _limit_rows_fallback
    rows = limit_rows(read_manifest(Path(args.manifest)), args.limit)
    if not rows:
        raise SystemExit("No evaluation rows selected.")

    max_target = max(int(row["target"]) for row in rows)
    if max_target >= int(resolved["num_classes"]):
        raise SystemExit(
            f"Found target id {max_target} but checkpoint num_classes is {resolved['num_classes']}."
        )

    load_part_locations = _symbol(module, ("_load_part_locations", "load_part_locations"))
    part_locations, part_status = load_part_locations(Path(str(resolved["dataset_root"])) / "parts" / "part_locs.txt")
    target_to_coarse: Dict[int, int] = {}
    coarse_class_count = int(resolved.get("coarse_class_count", 0) or 0)
    state_has_coarse_head = any(str(key).startswith("coarse_classifier.") for key in model_state)
    if bool(resolved.get("hierarchy_loss_enabled")) or state_has_coarse_head:
        hierarchy_builder = _symbol(module, ("build_hierarchy_mapping", "build_coarse_mapping"), required=False)
        if hierarchy_builder is not None:
            target_to_coarse, hierarchy_info = hierarchy_builder(Path(str(resolved["dataset_root"])), rows)
            if coarse_class_count <= 0:
                coarse_class_count = int(hierarchy_info.get("coarse_class_count", 0) or 0)

    transform_builder = _symbol(module, ("build_transforms", "build_eval_transforms", "make_transforms"))
    full_transform = transform_builder(
        transforms,
        int(resolved["image_size"]),
        train=False,
        min_scale=float(resolved.get("full_min_scale", 0.55)),
    )
    crop_transform = transform_builder(
        transforms,
        int(resolved["image_size"]),
        train=False,
        min_scale=float(resolved.get("crop_min_scale", 0.75)),
    )
    dataset_factory = _symbol(module, ("make_dataset_class", "make_part_hierarchy_dataset_class"))
    Dataset = dataset_factory(torch, Image)
    dataset = Dataset(
        rows,
        path_root=Path(str(resolved["path_root"])),
        part_locations=part_locations,
        target_to_coarse=target_to_coarse,
        num_views=int(resolved["num_views"]),
        full_transform=full_transform,
        crop_transform=crop_transform,
        crop_padding=float(resolved.get("crop_padding", 0.0)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_builder = _symbol(module, ("build_part_hierarchy_model", "build_part_hierarchy_fused_model"))
    model = model_builder(
        torch,
        nn,
        models,
        model_name=str(resolved["model"]),
        num_classes=int(resolved["num_classes"]),
        num_views=int(resolved["num_views"]),
        pretrained=False,
        dropout=float(resolved.get("dropout", 0.2)),
        fusion=str(resolved.get("fusion", "concat")),
        coarse_num_classes=coarse_class_count if state_has_coarse_head else 0,
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    accuracy_at_k = getattr(module, "accuracy_at_k", None) or _accuracy_at_k
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
        for batch in loader:
            if isinstance(batch, Mapping):
                views = batch.get("views", batch.get("inputs", batch.get("image", batch.get("images"))))
                targets = batch.get("target", batch.get("targets", batch.get("label", batch.get("labels"))))
                if views is None or targets is None:
                    raise SystemExit("Dataset batch mapping must contain views/images and target/targets.")
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                views, targets = batch[0], batch[1]
            else:
                raise SystemExit("Dataset batches must be mappings or tuple/list values with views and targets.")
            views = views.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = _forward_logits_with_optional_tta(model, views, tta_hflip=args.tta_hflip)
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

    model_config = {key: value for key, value in resolved.items() if isinstance(value, (str, int, float, bool, type(None)))}
    metrics: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(Path(args.manifest)),
        "num_examples": total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
        "macro_f1": None,
        "balanced_accuracy": None,
        "loss": total_loss / total_seen,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics"),
        "model_config": model_config,
        "part_locs_status": part_status,
        "tta_hflip": bool(args.tta_hflip),
    }
    metrics.update(_metrics_from_predictions(all_targets, all_preds))
    for key in ("model", "num_views", "image_size", "num_classes"):
        if key in model_config:
            metrics[key] = model_config[key]

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.output_predictions:
        _write_csv(Path(args.output_predictions), prediction_rows, PREDICTION_FIELDNAMES)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a part-guided hierarchy-fused NABirds checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="reports/milestone2/nabirds_test.csv")
    parser.add_argument("--class-prompts", default="reports/milestone2/nabirds_class_prompts.csv")
    parser.add_argument("--dataset-root", default="nabirds")
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-predictions", default="")
    parser.add_argument("--model", default="", help="Override model architecture; defaults to checkpoint args.")
    parser.add_argument("--num-views", type=int, default=0, help="Override eval view count; defaults to checkpoint args.")
    parser.add_argument("--image-size", type=int, default=0, help="Override eval image size; defaults to checkpoint args.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--tta-hflip",
        action="store_true",
        help="Average logits from original and horizontally flipped [B,V,C,H,W] eval views.",
    )
    return parser.parse_args()


def main() -> None:
    metrics = evaluate(parse_args())
    print("NABirds part hierarchy-fused checkpoint evaluation")
    for key, value in metrics.items():
        if key == "checkpoint_val_metrics":
            continue
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
