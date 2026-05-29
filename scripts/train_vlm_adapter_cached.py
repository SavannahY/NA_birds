#!/usr/bin/env python3
"""Train lightweight VLM adapters from precomputed frozen image features.

This is the cached-feature counterpart to scripts/train_vlm_adapter.py. It
loads image_features/targets from scripts/precompute_vlm_image_features.py
outputs, encodes class prompts once with the SigLIP/SigLIP2 text encoder, and
trains only the small image/text alignment adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vlm_adapter import (  # noqa: E402
    DEFAULT_HARD_NEGATIVES,
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    EXPECTED_NABIRDS_CLASSES,
    _encode_text_features,
    _hard_negative_loss,
    _load_hard_negative_groups,
    _load_prompt_rows,
    _macro_f1,
    _normalize,
    _positive_int,
    _nonnegative_float,
    _progress,
    _resolve_device,
    _resolve_dtype,
    _seed_everything,
    _text_processor_kwargs,
    _write_json,
    accuracy_at_k,
    build_alignment_model,
)


def _require_runtime_deps() -> Tuple[Any, Any, Any, Any]:
    missing: List[str] = []
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        torch = None
        nn = None
        missing.append("torch")

    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        AutoModel = None
        AutoProcessor = None
        missing.append("transformers")

    if missing:
        raise SystemExit(
            "Cached VLM adapter training dependencies are missing: "
            f"{', '.join(sorted(set(missing)))}.\n"
            "Install a compatible PyTorch build plus transformers before training."
        )
    return torch, nn, AutoModel, AutoProcessor


def _torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else path


def _load_feature_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Feature manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse feature manifest JSON: {path}") from exc
    splits = payload.get("splits")
    if not isinstance(splits, list):
        raise SystemExit(f"{path} is missing a list-valued 'splits' field.")
    return payload


def _feature_paths_from_args(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Dict[str, Any]]]:
    manifest_payload: Optional[Dict[str, Any]] = None
    train_path = Path(args.train_features).expanduser() if args.train_features else None
    val_path = Path(args.val_features).expanduser() if args.val_features else None

    if args.feature_manifest:
        manifest_path = Path(args.feature_manifest).expanduser()
        manifest_payload = _load_feature_manifest(manifest_path)
        by_split: Dict[str, Path] = {}
        for split in manifest_payload["splits"]:
            if not isinstance(split, Mapping):
                continue
            split_name = str(split.get("split", ""))
            output = split.get("output")
            if split_name and output:
                by_split[split_name] = _resolve_manifest_path(str(output), manifest_path)
        train_path = train_path or by_split.get("train")
        val_path = val_path or by_split.get("val")

    if train_path is None or val_path is None:
        raise SystemExit("Pass --feature-manifest or both --train-features and --val-features.")
    return train_path, val_path, manifest_payload


def _validate_feature_payload(torch: Any, path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Feature checkpoint not found: {path}")
    payload = _torch_load(torch, path)
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{path} did not contain a checkpoint dictionary.")
    for key in ("image_features", "targets"):
        if key not in payload:
            raise SystemExit(f"{path} is missing required key {key!r}.")
    features = payload["image_features"]
    targets = payload["targets"]
    if getattr(features, "ndim", None) != 2:
        raise SystemExit(f"{path}: image_features must be a 2D tensor.")
    if getattr(targets, "ndim", None) != 1:
        raise SystemExit(f"{path}: targets must be a 1D tensor.")
    if int(features.shape[0]) != int(targets.shape[0]):
        raise SystemExit(f"{path}: feature/target row count mismatch.")
    return dict(payload)


def _validate_text_feature_payload(torch: Any, path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Text feature checkpoint not found: {path}")
    payload = _torch_load(torch, path)
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{path} did not contain a checkpoint dictionary.")
    for key in ("class_text_features", "class_targets", "class_names"):
        if key not in payload:
            raise SystemExit(f"{path} is missing required key {key!r}.")
    features = payload["class_text_features"]
    if getattr(features, "ndim", None) != 2:
        raise SystemExit(f"{path}: class_text_features must be a 2D tensor.")
    if int(features.shape[0]) != len(payload["class_targets"]):
        raise SystemExit(f"{path}: feature/class target row count mismatch.")
    return dict(payload)


def _limit_tensor_rows(torch: Any, features: Any, targets: Any, limit: Optional[int]) -> Tuple[Any, Any]:
    if limit is None or limit <= 0:
        return features, targets
    return features[:limit].contiguous(), targets[:limit].contiguous()


def _map_targets(torch: Any, raw_targets: Any, target_to_index: Mapping[int, int], path: Path) -> Any:
    values = [int(value) for value in raw_targets.cpu().tolist()]
    mapped: List[int] = []
    missing: List[int] = []
    for value in values:
        if value in target_to_index:
            mapped.append(target_to_index[value])
        elif 0 <= value < len(target_to_index):
            mapped.append(value)
        else:
            missing.append(value)
    if missing:
        preview = ", ".join(str(value) for value in sorted(set(missing))[:10])
        raise SystemExit(f"{path}: feature target(s) missing from prompt CSV: {preview}")
    return torch.tensor(mapped, dtype=torch.long)


class CachedFeatureDataset:
    def __init__(self, features: Any, targets: Any):
        self.features = features.float().contiguous()
        self.targets = targets.long().contiguous()

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.features[index], self.targets[index]


def _dataloader(torch: Any, dataset: CachedFeatureDataset, batch_size: int, shuffle: bool) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _target_summary(targets: Any) -> Tuple[int, int, int]:
    values = [int(value) for value in targets.cpu().tolist()]
    return min(values), max(values), len(set(values))


def _loss_terms(
    torch: Any,
    nn: Any,
    logits: Any,
    targets: Any,
    criterion: Any,
    hard_negative_weight: float,
    hard_negatives: Mapping[int, Sequence[int]],
) -> Tuple[Any, Dict[str, float]]:
    ce_loss = criterion(logits, targets)
    total_loss = ce_loss
    terms = {"ce_loss": float(ce_loss.detach().item())}
    if hard_negative_weight > 0:
        hard_loss = _hard_negative_loss(torch, logits, targets, hard_negatives)
        total_loss = total_loss + (hard_negative_weight * hard_loss)
        terms["hard_negative_loss"] = float(hard_loss.detach().item())
    terms["loss"] = float(total_loss.detach().item())
    return total_loss, terms


def train_one_epoch(
    torch: Any,
    nn: Any,
    adapter: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    class_text_features: Any,
    hard_negatives: Mapping[int, Sequence[int]],
    device: Any,
    hard_negative_weight: float,
    grad_clip_norm: float,
    show_progress: bool,
) -> Dict[str, float]:
    adapter.train()
    total_seen = 0
    totals = {"loss": 0.0, "ce_loss": 0.0, "hard_negative_loss": 0.0, "top1": 0.0, "top5": 0.0}
    batches = _progress(loader, total=len(loader), label="train", enabled=show_progress)

    for features, targets in batches:
        features = features.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _image_z, _text_z = adapter(features, class_text_features)
        loss, terms = _loss_terms(
            torch=torch,
            nn=nn,
            logits=logits,
            targets=targets,
            criterion=criterion,
            hard_negative_weight=hard_negative_weight,
            hard_negatives=hard_negatives,
        )
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = targets.size(0)
        top1, top5 = accuracy_at_k(torch, logits.detach(), targets, topk=(1, 5))
        total_seen += batch_size
        totals["loss"] += terms["loss"] * batch_size
        totals["ce_loss"] += terms["ce_loss"] * batch_size
        totals["hard_negative_loss"] += terms.get("hard_negative_loss", 0.0) * batch_size
        totals["top1"] += top1 * batch_size
        totals["top5"] += top5 * batch_size

    metrics = {key: value / total_seen for key, value in totals.items() if key != "hard_negative_loss"}
    if hard_negative_weight > 0:
        metrics["hard_negative_loss"] = totals["hard_negative_loss"] / total_seen
    return metrics


def evaluate(
    torch: Any,
    nn: Any,
    adapter: Any,
    loader: Any,
    criterion: Any,
    class_text_features: Any,
    hard_negatives: Mapping[int, Sequence[int]],
    device: Any,
    hard_negative_weight: float,
    compute_macro_f1: bool,
    show_progress: bool,
) -> Dict[str, float]:
    adapter.eval()
    total_seen = 0
    totals = {"loss": 0.0, "ce_loss": 0.0, "hard_negative_loss": 0.0, "top1": 0.0, "top5": 0.0}
    y_true: List[int] = []
    y_pred: List[int] = []
    batches = _progress(loader, total=len(loader), label="val", enabled=show_progress)

    with torch.inference_mode():
        for features, targets in batches:
            features = features.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            logits, _image_z, _text_z = adapter(features, class_text_features)
            loss, terms = _loss_terms(
                torch=torch,
                nn=nn,
                logits=logits,
                targets=targets,
                criterion=criterion,
                hard_negative_weight=hard_negative_weight,
                hard_negatives=hard_negatives,
            )

            batch_size = targets.size(0)
            top1, top5 = accuracy_at_k(torch, logits, targets, topk=(1, 5))
            preds = logits.argmax(dim=1)
            total_seen += batch_size
            totals["loss"] += terms["loss"] * batch_size
            totals["ce_loss"] += terms["ce_loss"] * batch_size
            totals["hard_negative_loss"] += terms.get("hard_negative_loss", 0.0) * batch_size
            totals["top1"] += top1 * batch_size
            totals["top5"] += top5 * batch_size
            if compute_macro_f1:
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

    metrics = {key: value / total_seen for key, value in totals.items() if key != "hard_negative_loss"}
    if hard_negative_weight > 0:
        metrics["hard_negative_loss"] = totals["hard_negative_loss"] / total_seen
    if compute_macro_f1:
        metrics["macro_f1"] = _macro_f1(y_true, y_pred, num_classes=class_text_features.size(0))
    return metrics


def _load_text_model(AutoModel: Any, AutoProcessor: Any, model_name: str) -> Tuple[Any, Any]:
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError as exc:
        raise SystemExit(
            f"Could not load model or processor for {model_name!r}.\n"
            "If this is the first run, the checkpoint may need to be downloaded from Hugging Face."
        ) from exc
    return model, processor


def _model_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight image-text adapter from cached frozen VLM image features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature-manifest", default=None, help="JSON manifest from precompute_vlm_image_features.py.")
    parser.add_argument("--train-features", default=None, help="Explicit train split .pt feature checkpoint.")
    parser.add_argument("--val-features", default=None, help="Explicit val split .pt feature checkpoint.")
    parser.add_argument("--text-features", default=None, help="Optional cached text feature checkpoint from precompute_vlm_text_features.py.")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="NABirds class prompt CSV.")
    parser.add_argument("--hard-negative-csv", default=DEFAULT_HARD_NEGATIVES, help="Family-variant hard-negative CSV.")
    parser.add_argument("--out-dir", default="reports/milestone2/runs", help="Output directory for adapter checkpoints.")
    parser.add_argument("--model", default=None, help="Text encoder model id/path; defaults to feature manifest model or project default.")
    parser.add_argument("--prompt-aggregation", choices=("mean", "first"), default="mean")
    parser.add_argument("--expected-num-classes", type=int, default=EXPECTED_NABIRDS_CLASSES, help="Use 0 to disable.")
    parser.add_argument("--adapter-mode", choices=("linear", "mlp", "residual", "none"), default="residual")
    parser.add_argument("--projection-dim", type=int, default=0, help="0 keeps the VLM feature dimension.")
    parser.add_argument("--adapter-hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=_nonnegative_float, default=0.1)
    parser.add_argument("--shared-adapter", action="store_true")
    parser.add_argument("--class-bias", action="store_true")
    parser.add_argument("--init-temperature", type=float, default=0.07)
    parser.add_argument("--fixed-temperature", action="store_true")
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--text-batch-size", type=_positive_int, default=256)
    parser.add_argument("--text-padding", choices=("auto", "longest", "max_length"), default="auto")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=1e-4)
    parser.add_argument("--label-smoothing", type=_nonnegative_float, default=0.05)
    parser.add_argument("--hard-negative-weight", type=_nonnegative_float, default=0.0)
    parser.add_argument("--max-hard-negatives", type=int, default=8, help="<=0 uses all group variants.")
    parser.add_argument("--grad-clip-norm", type=_nonnegative_float, default=1.0)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto", help="Text encoder dtype.")
    parser.add_argument("--macro-f1", action="store_true")
    parser.add_argument("--limit-train", type=_positive_int, default=None)
    parser.add_argument("--limit-val", type=_positive_int, default=None)
    parser.add_argument("--renormalize-image-features", action="store_true", help="L2-normalize cached image features before training.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _seed_everything(args.seed)
    torch, nn, AutoModel, AutoProcessor = _require_runtime_deps()

    train_path, val_path, feature_manifest = _feature_paths_from_args(args)
    train_payload = _validate_feature_payload(torch, train_path)
    val_payload = _validate_feature_payload(torch, val_path)
    train_features, train_raw_targets = _limit_tensor_rows(
        torch, train_payload["image_features"], train_payload["targets"], args.limit_train
    )
    val_features, val_raw_targets = _limit_tensor_rows(
        torch, val_payload["image_features"], val_payload["targets"], args.limit_val
    )
    if int(train_features.shape[1]) != int(val_features.shape[1]):
        raise SystemExit("Train/val cached feature dimensions do not match.")

    model_name = args.model or (feature_manifest or {}).get("model")
    if not model_name:
        model_name = train_payload.get("config", {}).get("model") or DEFAULT_MODEL

    prompt_path = Path(args.prompts)
    if args.prompts == DEFAULT_PROMPTS and not prompt_path.is_file() and feature_manifest:
        manifest_dir = feature_manifest.get("manifest_dir")
        if manifest_dir:
            prompt_path = Path(manifest_dir) / "nabirds_class_prompts.csv"

    text_payload: Optional[Dict[str, Any]] = None
    if args.text_features:
        text_payload = _validate_text_feature_payload(torch, Path(args.text_features))
        class_targets = [int(value) for value in text_payload["class_targets"]]
        class_names = [str(value) for value in text_payload["class_names"]]
        prompt_rows = int(text_payload.get("config", {}).get("num_prompt_rows", len(class_targets)))
        if args.expected_num_classes and len(class_targets) != args.expected_num_classes:
            raise SystemExit(
                f"{args.text_features}: expected {args.expected_num_classes} classes, found {len(class_targets)}."
            )
    else:
        class_targets, class_names, prompts_by_target, prompt_rows = _load_prompt_rows(
            prompt_path,
            expected_num_classes=args.expected_num_classes,
        )
    target_to_index = {target: index for index, target in enumerate(class_targets)}
    train_targets = _map_targets(torch, train_raw_targets, target_to_index, train_path)
    val_targets = _map_targets(torch, val_raw_targets, target_to_index, val_path)
    train_min, train_max, train_unique = _target_summary(train_targets)
    val_min, val_max, val_unique = _target_summary(val_targets)

    if args.renormalize_image_features:
        train_features = _normalize(torch, train_features.float())
        val_features = _normalize(torch, val_features.float())

    hard_negative_path = Path(args.hard_negative_csv) if args.hard_negative_csv else None
    hard_negatives = (
        _load_hard_negative_groups(hard_negative_path, target_to_index, args.max_hard_negatives)
        if args.hard_negative_weight > 0
        else {}
    )

    device = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype, device)
    if text_payload is not None:
        class_text_features = _normalize(torch, text_payload["class_text_features"].float()).to(device=device)
        cached_model = text_payload.get("config", {}).get("model")
        if cached_model and str(cached_model) != str(model_name):
            print(f"Using cached text features from {cached_model}; image feature model is {model_name}.")
    else:
        print(f"Loading text encoder: {model_name}")
        text_model, processor = _load_text_model(AutoModel, AutoProcessor, model_name)
        text_model.to(device=device)
        if dtype is not None:
            text_model.to(dtype=dtype)
        text_model.eval()
        for parameter in text_model.parameters():
            parameter.requires_grad_(False)

        class_text_features = _encode_text_features(
            torch=torch,
            model=text_model,
            processor=processor,
            prompts_by_target=prompts_by_target,
            class_targets=class_targets,
            device=device,
            dtype=dtype,
            text_batch_size=args.text_batch_size,
            aggregation=args.prompt_aggregation,
            show_progress=not args.no_progress,
            text_processor_kwargs=_text_processor_kwargs(
                str(model_name),
                args.text_padding,
                args.text_max_length,
                model=text_model,
            ),
        ).to(device=device)
        del text_model

    image_dim = int(train_features.shape[1])
    text_dim = int(class_text_features.shape[1])
    adapter = build_alignment_model(
        torch=torch,
        nn=nn,
        image_dim=image_dim,
        text_dim=text_dim,
        projection_dim=args.projection_dim,
        adapter_mode=args.adapter_mode,
        adapter_hidden_dim=args.adapter_hidden_dim,
        dropout=args.dropout,
        shared_adapter=args.shared_adapter,
        init_temperature=args.init_temperature,
        learnable_temperature=not args.fixed_temperature,
        class_bias=args.class_bias,
        num_classes=len(class_targets),
    ).to(device=device)

    train_ds = CachedFeatureDataset(train_features, train_targets)
    val_ds = CachedFeatureDataset(val_features, val_targets)
    train_loader = _dataloader(torch, train_ds, args.batch_size, shuffle=True)
    val_loader = _dataloader(torch, val_ds, args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"vlm_adapter_cached_{_model_slug(str(model_name))}_{args.adapter_mode}_{int(time.time())}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "train_features": str(train_path),
            "val_features": str(val_path),
            "feature_manifest": str(args.feature_manifest) if args.feature_manifest else None,
            "prompts": str(prompt_path),
            "model_resolved": str(model_name),
            "device_resolved": str(device),
            "dtype_resolved": str(dtype).replace("torch.", "") if dtype is not None else None,
            "image_feature_dim": image_dim,
            "text_feature_dim": text_dim,
            "num_classes": len(class_targets),
            "num_prompt_rows": prompt_rows,
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "train_target_min": train_min,
            "train_target_max": train_max,
            "train_unique_targets": train_unique,
            "val_target_min": val_min,
            "val_target_max": val_max,
            "val_unique_targets": val_unique,
            "hard_negative_classes": len(hard_negatives),
            "source_feature_config": {
                "train": train_payload.get("config", {}),
                "val": val_payload.get("config", {}),
            },
        }
    )
    _write_json(out_dir / "config.json", config)
    _write_json(out_dir / "classes.json", {"class_targets": class_targets, "class_names": class_names})

    print(
        "Loaded cached features "
        f"{len(train_ds)} train / {len(val_ds)} val rows; {len(class_targets)} classes; "
        f"feature dims image/text {image_dim}/{text_dim}; output: {out_dir}"
    )
    if args.hard_negative_weight > 0:
        print(f"Hard-negative loss enabled for {len(hard_negatives)} classes from {hard_negative_path}.")

    history: List[Dict[str, Any]] = []
    best_top1 = -math.inf
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = train_one_epoch(
            torch=torch,
            nn=nn,
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            class_text_features=class_text_features,
            hard_negatives=hard_negatives,
            device=device,
            hard_negative_weight=args.hard_negative_weight,
            grad_clip_norm=args.grad_clip_norm,
            show_progress=not args.no_progress,
        )
        val_metrics = evaluate(
            torch=torch,
            nn=nn,
            adapter=adapter,
            loader=val_loader,
            criterion=criterion,
            class_text_features=class_text_features,
            hard_negatives=hard_negatives,
            device=device,
            hard_negative_weight=args.hard_negative_weight,
            compute_macro_f1=args.macro_f1,
            show_progress=not args.no_progress,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr(),
            "seconds": round(time.time() - started, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        checkpoint = {
            "epoch": epoch,
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
            "class_targets": class_targets,
            "class_names": class_names,
            "image_feature_dim": image_dim,
            "text_feature_dim": text_dim,
            "model": str(model_name),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(checkpoint, out_dir / "best.pt")
        _write_json(out_dir / "history.json", {"history": history, "best_top1": best_top1})

    print(f"Best validation top-1: {best_top1:.4f}")
    print(f"Run directory: {out_dir}")


if __name__ == "__main__":
    main()
