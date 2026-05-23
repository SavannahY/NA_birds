#!/usr/bin/env python3
"""Train full-image or bounding-box-crop visual baselines on NABirds.

This script covers Milestone 2 steps:
  - Step 2: full-image baseline
  - Step 3: bbox-crop baseline

It intentionally keeps imports for torch/torchvision inside main paths so the
file can be syntax-checked on machines without the training stack installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence


def _require_training_deps():
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install torch, torchvision, and pillow first.\n"
            "Example: pip install torch torchvision pillow scikit-learn tqdm"
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


def _limit_rows(rows: List[Dict[str, str]], limit: int | None) -> List[Dict[str, str]]:
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


def build_transforms(transforms, image_size: int, train: bool):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0)),
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
    class NABirdsManifestDataset(torch.utils.data.Dataset):
        def __init__(self, rows: Sequence[Dict[str, str]], transform=None, input_mode: str = "full"):
            if input_mode not in {"full", "bbox"}:
                raise ValueError(f"input_mode must be 'full' or 'bbox', got {input_mode!r}")
            self.rows = list(rows)
            self.transform = transform
            self.input_mode = input_mode

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
                if self.input_mode == "bbox":
                    x = int(row["clip_x"])
                    y = int(row["clip_y"])
                    w = int(row["clip_w"])
                    h = int(row["clip_h"])
                    image = image.crop((x, y, x + w, y + h))
                if self.transform:
                    image = self.transform(image)
            target = int(row["target"])
            return image, target

    return NABirdsManifestDataset


def build_model(torch, nn, models, model_name: str, num_classes: int, pretrained: bool):
    weights = "DEFAULT" if pretrained else None
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def accuracy_at_k(torch, logits, targets, topk=(1, 5)):
    max_k = max(topk)
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    values = []
    for k in topk:
        values.append(correct[:k].reshape(-1).float().sum(0).item() / targets.size(0))
    return values


def train_one_epoch(torch, model, loader, optimizer, criterion, device, use_amp: bool):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        top1, _ = accuracy_at_k(torch, logits.detach(), targets, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_seen += batch_size
    return {"loss": total_loss / total_seen, "top1": total_top1 / total_seen}


def evaluate(torch, model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    total_top5 = 0.0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            top1, top5 = accuracy_at_k(torch, logits, targets, topk=(1, 5))
            preds = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_seen += batch_size
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = {
        "loss": total_loss / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
    }
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score

        metrics["macro_f1"] = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(all_targets, all_preds))
    except Exception:
        pass
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NABirds visual baseline.")
    parser.add_argument("--manifest-dir", default="reports/milestone2")
    parser.add_argument("--out-dir", default="reports/milestone2/runs")
    parser.add_argument("--model", choices=["resnet18", "resnet50", "convnext_tiny", "vit_b_16"], default="resnet50")
    parser.add_argument("--input-mode", choices=["full", "bbox"], default="full")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-classes", type=int, default=555)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional smoke-test cap for train rows.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional smoke-test cap for val rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    torch, nn, Image, models, transforms = _require_training_deps()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    manifest_dir = Path(args.manifest_dir)
    train_rows = _limit_rows(_read_manifest(manifest_dir / "nabirds_train.csv"), args.limit_train)
    val_rows = _limit_rows(_read_manifest(manifest_dir / "nabirds_val.csv"), args.limit_val)
    Dataset = make_dataset_class(torch, Image)

    train_ds = Dataset(train_rows, build_transforms(transforms, args.image_size, train=True), args.input_mode)
    val_ds = Dataset(val_rows, build_transforms(transforms, args.image_size, train=False), args.input_mode)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        torch,
        nn,
        models,
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"{args.model}_{args.input_mode}_{'pretrained' if not args.no_pretrained else 'scratch'}_{int(time.time())}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(torch, model, train_loader, optimizer, criterion, device, args.amp)
        val_metrics = evaluate(torch, model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
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
