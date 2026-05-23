#!/usr/bin/env python3
"""Train a fused full-image + bounding-box-crop NABirds classifier.

The script consumes the manifests produced by scripts/build_nabirds_manifests.py:

  reports/milestone2/nabirds_train.csv
  reports/milestone2/nabirds_val.csv

Torch and torchvision imports are intentionally kept inside runtime helpers so
the file can be syntax-checked on machines without the training stack installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SUPPORTED_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "convnext_tiny",
    "efficientnet_b0",
    "vit_b_16",
)


def _require_training_deps():
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install torch, torchvision, and pillow first.\n"
            "Example: pip install torch torchvision pillow scikit-learn"
        ) from exc
    return torch, nn, Image, models, transforms


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Manifest is empty: {path}")
    required = {"image_path", "target", "clip_x", "clip_y", "clip_w", "clip_h"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise SystemExit(f"Manifest {path} is missing required columns: {', '.join(missing)}")
    return rows


def _limit_rows(rows: List[Dict[str, str]], limit: Optional[int]) -> List[Dict[str, str]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _as_int(value: str) -> int:
    return int(round(float(value)))


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


def build_transforms(transforms, image_size: int, train: bool, min_scale: float):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def make_dataset_class(torch, Image):
    class FusedNABirdsManifestDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            rows: Sequence[Dict[str, str]],
            path_root: Path,
            full_transform=None,
            crop_transform=None,
            crop_padding: float = 0.0,
        ):
            self.rows = list(rows)
            self.path_root = Path(path_root)
            self.full_transform = full_transform
            self.crop_transform = crop_transform
            self.crop_padding = max(0.0, crop_padding)

        def __len__(self):
            return len(self.rows)

        def _image_path(self, row: Dict[str, str]) -> Path:
            image_path = Path(row["image_path"])
            if image_path.is_absolute():
                return image_path
            return self.path_root / image_path

        def _crop_box(self, row: Dict[str, str], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
            image_w, image_h = image_size
            x = _as_int(row["clip_x"])
            y = _as_int(row["clip_y"])
            w = _as_int(row["clip_w"])
            h = _as_int(row["clip_h"])

            if self.crop_padding > 0.0:
                pad_x = int(round(w * self.crop_padding))
                pad_y = int(round(h * self.crop_padding))
            else:
                pad_x = 0
                pad_y = 0

            left = max(0, x - pad_x)
            top = max(0, y - pad_y)
            right = min(image_w, x + w + pad_x)
            bottom = min(image_h, y + h + pad_y)
            if right <= left or bottom <= top:
                return 0, 0, image_w, image_h
            return left, top, right, bottom

        def __getitem__(self, index: int):
            row = self.rows[index]
            with Image.open(self._image_path(row)) as image:
                image = image.convert("RGB")
                crop = image.crop(self._crop_box(row, image.size))
                full = image.copy()

            if self.full_transform is not None:
                full = self.full_transform(full)
            if self.crop_transform is not None:
                crop = self.crop_transform(crop)
            target = int(row["target"])
            return full, crop, target

    return FusedNABirdsManifestDataset


def _make_backbone(nn, models, model_name: str, pretrained: bool):
    weights = "DEFAULT" if pretrained else None
    if model_name in {"resnet18", "resnet34", "resnet50"}:
        model_fn = getattr(models, model_name)
        backbone = model_fn(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    if model_name == "convnext_tiny":
        backbone = models.convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim

    if model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim

    if model_name == "vit_b_16":
        backbone = models.vit_b_16(weights=weights)
        feature_dim = backbone.heads.head.in_features
        backbone.heads.head = nn.Identity()
        return backbone, feature_dim

    raise ValueError(f"Unsupported model: {model_name}")


def build_fused_model(
    torch,
    nn,
    models,
    model_name: str,
    num_classes: int,
    pretrained: bool,
    branch_mode: str,
    dropout: float,
    freeze_backbone: bool,
):
    class FusedFullCropClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            if branch_mode == "shared":
                self.full_backbone, full_dim = _make_backbone(nn, models, model_name, pretrained)
                self.crop_backbone = self.full_backbone
                crop_dim = full_dim
            elif branch_mode == "two_branch":
                self.full_backbone, full_dim = _make_backbone(nn, models, model_name, pretrained)
                self.crop_backbone, crop_dim = _make_backbone(nn, models, model_name, pretrained)
            else:
                raise ValueError(f"branch_mode must be shared or two_branch, got {branch_mode!r}")

            if freeze_backbone:
                for parameter in self.full_backbone.parameters():
                    parameter.requires_grad = False
                if branch_mode == "two_branch":
                    for parameter in self.crop_backbone.parameters():
                        parameter.requires_grad = False

            fused_dim = full_dim + crop_dim
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fused_dim, num_classes),
            )

        @staticmethod
        def _flatten_features(features):
            if features.ndim > 2:
                return torch.flatten(features, start_dim=1)
            return features

        def forward(self, full_images, crop_images):
            full_features = self._flatten_features(self.full_backbone(full_images))
            crop_features = self._flatten_features(self.crop_backbone(crop_images))
            fused_features = torch.cat([full_features, crop_features], dim=1)
            return self.classifier(fused_features)

    return FusedFullCropClassifier()


def accuracy_at_k(torch, logits, targets, topk=(1, 5)):
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    values = []
    for requested_k in topk:
        k = min(requested_k, logits.size(1))
        values.append(correct[:k].reshape(-1).float().sum(0).item() / targets.size(0))
    return values


def train_one_epoch(
    torch,
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp: bool,
    grad_clip_norm: float,
):
    model.train()
    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    total_top5 = 0.0

    for full_images, crop_images, targets in loader:
        full_images = full_images.to(device, non_blocking=True)
        crop_images = crop_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(full_images, crop_images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        top1, top5 = accuracy_at_k(torch, logits.detach(), targets, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_seen += batch_size

    return {
        "loss": total_loss / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
    }


def evaluate(torch, model, loader, criterion, device, compute_macro_f1: bool):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    total_top5 = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for full_images, crop_images, targets in loader:
            full_images = full_images.to(device, non_blocking=True)
            crop_images = crop_images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(full_images, crop_images)
            loss = criterion(logits, targets)
            top1, top5 = accuracy_at_k(torch, logits, targets, topk=(1, 5))

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_seen += batch_size

            if compute_macro_f1:
                all_targets.extend(targets.cpu().tolist())
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "loss": total_loss / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
    }
    if compute_macro_f1:
        try:
            from sklearn.metrics import f1_score

            metrics["macro_f1"] = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        except Exception as exc:
            metrics["macro_f1"] = None
            metrics["macro_f1_error"] = str(exc)
    return metrics


def _target_summary(rows: Sequence[Dict[str, str]]) -> Tuple[int, int, int]:
    targets = [int(row["target"]) for row in rows]
    return min(targets), max(targets), len(set(targets))


def _dataloader_kwargs(device, args, shuffle: bool):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fused full-image + bbox-crop NABirds classifier.")
    parser.add_argument("--manifest-dir", default="reports/milestone2")
    parser.add_argument("--train-csv", default=None, help="Defaults to <manifest-dir>/nabirds_train.csv.")
    parser.add_argument("--val-csv", default=None, help="Defaults to <manifest-dir>/nabirds_val.csv.")
    parser.add_argument("--path-root", default=".", help="Base directory for relative image_path entries.")
    parser.add_argument("--out-dir", default="reports/milestone2/runs")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet50")
    parser.add_argument("--branch-mode", choices=["shared", "two_branch"], default="shared")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--num-classes", type=int, default=555)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--full-min-scale", type=float, default=0.55)
    parser.add_argument("--crop-min-scale", type=float, default=0.75)
    parser.add_argument("--crop-padding", type=float, default=0.0, help="Fractional padding around clipped bbox crop.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--macro-f1", action="store_true", help="Compute validation macro-F1 when scikit-learn is available.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional smoke-test cap for train rows.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional smoke-test cap for val rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    torch, nn, Image, models, transforms = _require_training_deps()

    device = _resolve_device(torch, args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("AMP requested but only CUDA AMP is enabled by this script; continuing without AMP.")

    manifest_dir = Path(args.manifest_dir)
    train_csv = Path(args.train_csv) if args.train_csv else manifest_dir / "nabirds_train.csv"
    val_csv = Path(args.val_csv) if args.val_csv else manifest_dir / "nabirds_val.csv"
    train_rows = _limit_rows(_read_manifest(train_csv), args.limit_train)
    val_rows = _limit_rows(_read_manifest(val_csv), args.limit_val)

    train_min, train_max, train_unique = _target_summary(train_rows)
    val_min, val_max, val_unique = _target_summary(val_rows)
    if max(train_max, val_max) >= args.num_classes:
        raise SystemExit(
            f"Found target id {max(train_max, val_max)} but --num-classes is {args.num_classes}. "
            "Increase --num-classes or rebuild the manifests."
        )

    Dataset = make_dataset_class(torch, Image)
    train_ds = Dataset(
        train_rows,
        path_root=Path(args.path_root),
        full_transform=build_transforms(transforms, args.image_size, train=True, min_scale=args.full_min_scale),
        crop_transform=build_transforms(transforms, args.image_size, train=True, min_scale=args.crop_min_scale),
        crop_padding=args.crop_padding,
    )
    val_ds = Dataset(
        val_rows,
        path_root=Path(args.path_root),
        full_transform=build_transforms(transforms, args.image_size, train=False, min_scale=args.full_min_scale),
        crop_transform=build_transforms(transforms, args.image_size, train=False, min_scale=args.crop_min_scale),
        crop_padding=args.crop_padding,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, **_dataloader_kwargs(device, args, shuffle=True))
    val_loader = torch.utils.data.DataLoader(val_ds, **_dataloader_kwargs(device, args, shuffle=False))

    model = build_fused_model(
        torch,
        nn,
        models,
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        branch_mode=args.branch_mode,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise SystemExit("No trainable parameters found. Check --freeze-backbone and classifier setup.")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = (
        f"fused_{args.model}_{args.branch_mode}_"
        f"{'pretrained' if not args.no_pretrained else 'scratch'}_{int(time.time())}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "device_resolved": str(device),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_target_min": train_min,
            "train_target_max": train_max,
            "train_unique_targets": train_unique,
            "val_target_min": val_min,
            "val_target_max": val_max,
            "val_unique_targets": val_unique,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        "Loaded "
        f"{len(train_rows)} train / {len(val_rows)} val rows; "
        f"train targets {train_min}-{train_max} ({train_unique} unique), "
        f"val targets {val_min}-{val_max} ({val_unique} unique)."
    )
    print(f"Model: {args.model}; branch_mode: {args.branch_mode}; device: {device}; output: {out_dir}")

    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = train_one_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = evaluate(torch, model, val_loader, criterion, device, compute_macro_f1=args.macro_f1)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - started, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Best validation top-1: {best_top1:.4f}")
    print(f"Run directory: {out_dir}")


if __name__ == "__main__":
    main()
