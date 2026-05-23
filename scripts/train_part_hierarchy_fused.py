#!/usr/bin/env python3
"""Train a part-hierarchy fused NABirds classifier.

This extends the fused full-image + bounding-box-crop baseline with stable
part-guided crops from NABirds part annotations and an optional hierarchy-aware
auxiliary loss. Torch imports are intentionally lazy so this file can be
syntax-checked without the training stack installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_MODELS = (
    "resnet50",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "swin_t",
    "swin_s",
    "swin_b",
)

PART_GROUPS: Tuple[Tuple[str, Tuple[int, ...], float], ...] = (
    ("head", (0, 1, 2, 3, 4), 0.48),
    ("wing_back", (9, 10, 7), 0.68),
    ("body_tail", (5, 6, 8), 0.72),
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
    required = {"image_path", "image_id", "raw_class_id", "target", "clip_x", "clip_y", "clip_w", "clip_h"}
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


def _sort_key(value: str) -> Tuple[int, object]:
    if str(value).isdigit():
        return 0, int(value)
    return 1, str(value)


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


def _load_part_locations(path: Path) -> Tuple[Dict[str, Dict[int, Tuple[int, int, bool]]], str]:
    if not path.exists():
        return {}, f"part_locs not found: {path}"
    parts: Dict[str, Dict[int, Tuple[int, int, bool]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            pieces = stripped.split()
            if len(pieces) != 5:
                raise SystemExit(f"{path}:{line_number}: expected 5 columns, got {len(pieces)}")
            image_id, raw_part_id, raw_x, raw_y, raw_visible = pieces
            visible = int(raw_visible)
            if visible not in {0, 1}:
                raise SystemExit(f"{path}:{line_number}: visible must be 0 or 1, got {raw_visible!r}")
            parts.setdefault(image_id, {})[int(raw_part_id)] = (int(raw_x), int(raw_y), bool(visible))
    return parts, "loaded"


def _read_classes(path: Path) -> Dict[str, str]:
    classes: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            pieces = stripped.split(maxsplit=1)
            if len(pieces) != 2:
                raise SystemExit(f"{path}:{line_number}: expected '<class_id> <class_name>'")
            classes[pieces[0]] = pieces[1]
    return classes


def _read_hierarchy(path: Path) -> Dict[str, str]:
    hierarchy: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            pieces = stripped.split()
            if len(pieces) != 2:
                raise SystemExit(f"{path}:{line_number}: expected '<child_class_id> <parent_class_id>'")
            hierarchy[pieces[0]] = pieces[1]
    return hierarchy


def _top_level_parent(raw_class_id: str, hierarchy: Dict[str, str]) -> Optional[str]:
    if raw_class_id not in hierarchy:
        return None
    current = raw_class_id
    seen = set()
    while current in hierarchy:
        parent = hierarchy[current]
        if parent == "0":
            return current
        if parent in seen:
            return None
        seen.add(current)
        current = parent
    return None


def build_hierarchy_mapping(
    dataset_root: Path,
    rows: Sequence[Dict[str, str]],
) -> Tuple[Dict[int, int], Dict[str, object]]:
    classes_path = dataset_root / "classes.txt"
    hierarchy_path = dataset_root / "hierarchy.txt"
    info: Dict[str, object] = {
        "classes_path": str(classes_path),
        "hierarchy_path": str(hierarchy_path),
        "strategy": "top_level_parent_below_birds",
        "available": False,
        "reason": "",
        "coarse_class_count": 0,
    }
    if not classes_path.exists() or not hierarchy_path.exists():
        info["reason"] = "classes.txt or hierarchy.txt unavailable"
        return {}, info

    classes = _read_classes(classes_path)
    hierarchy = _read_hierarchy(hierarchy_path)
    target_to_raw: Dict[int, str] = {}
    conflicts: List[int] = []
    for row in rows:
        target = int(row["target"])
        raw_class_id = row["raw_class_id"].strip()
        previous = target_to_raw.setdefault(target, raw_class_id)
        if previous != raw_class_id:
            conflicts.append(target)

    raw_to_coarse_raw: Dict[str, str] = {}
    missing_raw = []
    for raw_class_id in sorted(set(target_to_raw.values()), key=_sort_key):
        if raw_class_id not in classes:
            missing_raw.append(raw_class_id)
            continue
        coarse_raw = _top_level_parent(raw_class_id, hierarchy)
        if coarse_raw is None or coarse_raw not in classes:
            missing_raw.append(raw_class_id)
            continue
        raw_to_coarse_raw[raw_class_id] = coarse_raw

    coarse_raw_ids = sorted(set(raw_to_coarse_raw.values()), key=_sort_key)
    coarse_raw_to_id = {raw_class_id: idx for idx, raw_class_id in enumerate(coarse_raw_ids)}
    target_to_coarse = {
        target: coarse_raw_to_id[raw_to_coarse_raw[raw_class_id]]
        for target, raw_class_id in target_to_raw.items()
        if raw_class_id in raw_to_coarse_raw
    }
    info.update(
        {
            "available": bool(coarse_raw_to_id) and not conflicts,
            "reason": "loaded" if coarse_raw_to_id and not conflicts else "mapping conflicts or no coarse classes",
            "coarse_class_count": len(coarse_raw_to_id),
            "coarse_raw_ids": coarse_raw_ids,
            "coarse_class_names": {coarse_raw_to_id[key]: classes[key] for key in coarse_raw_ids},
            "missing_raw_class_count": len(missing_raw),
            "missing_raw_class_examples": missing_raw[:10],
            "target_raw_conflicts": sorted(set(conflicts))[:10],
            "mapped_fine_target_count": len(target_to_coarse),
        }
    )
    return target_to_coarse, info


def _view_names(num_views: int) -> List[str]:
    names = ["full", "bbox"]
    names.extend(group[0] for group in PART_GROUPS[: max(0, num_views - 2)])
    return names


def make_dataset_class(torch, Image):
    class PartHierarchyNABirdsDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            rows: Sequence[Dict[str, str]],
            path_root: Path,
            part_locations: Dict[str, Dict[int, Tuple[int, int, bool]]],
            target_to_coarse: Dict[int, int],
            num_views: int,
            full_transform=None,
            crop_transform=None,
            crop_padding: float = 0.0,
        ):
            self.rows = list(rows)
            self.path_root = Path(path_root)
            self.part_locations = part_locations
            self.target_to_coarse = target_to_coarse
            self.num_views = num_views
            self.part_groups = PART_GROUPS[: max(0, num_views - 2)]
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

        def _bbox_box(self, row: Dict[str, str], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
            image_w, image_h = image_size
            x = _as_int(row["clip_x"])
            y = _as_int(row["clip_y"])
            w = _as_int(row["clip_w"])
            h = _as_int(row["clip_h"])
            pad_x = int(round(w * self.crop_padding))
            pad_y = int(round(h * self.crop_padding))
            left = max(0, x - pad_x)
            top = max(0, y - pad_y)
            right = min(image_w, x + w + pad_x)
            bottom = min(image_h, y + h + pad_y)
            if right <= left or bottom <= top:
                return 0, 0, image_w, image_h
            return left, top, right, bottom

        @staticmethod
        def _part_box(
            points: Sequence[Tuple[int, int]],
            bbox_box: Tuple[int, int, int, int],
            image_size: Tuple[int, int],
            scale: float,
        ) -> Tuple[int, int, int, int]:
            image_w, image_h = image_size
            bbox_left, bbox_top, bbox_right, bbox_bottom = bbox_box
            bbox_w = max(1, bbox_right - bbox_left)
            bbox_h = max(1, bbox_bottom - bbox_top)
            xs = [min(max(0, point[0]), image_w - 1) for point in points]
            ys = [min(max(0, point[1]), image_h - 1) for point in points]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            point_span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
            side = max(scale * max(bbox_w, bbox_h), point_span * 2.2, 32.0)
            half = side / 2.0
            left = int(round(center_x - half))
            top = int(round(center_y - half))
            right = int(round(center_x + half))
            bottom = int(round(center_y + half))

            if right - left > image_w:
                left, right = 0, image_w
            else:
                if left < 0:
                    right -= left
                    left = 0
                if right > image_w:
                    left -= right - image_w
                    right = image_w
            if bottom - top > image_h:
                top, bottom = 0, image_h
            else:
                if top < 0:
                    bottom -= top
                    top = 0
                if bottom > image_h:
                    top -= bottom - image_h
                    bottom = image_h

            left = max(0, left)
            top = max(0, top)
            right = min(image_w, right)
            bottom = min(image_h, bottom)
            if right <= left or bottom <= top:
                return bbox_box
            return left, top, right, bottom

        def _group_crop(
            self,
            image,
            image_id: str,
            part_ids: Iterable[int],
            bbox_box: Tuple[int, int, int, int],
            scale: float,
        ):
            image_parts = self.part_locations.get(image_id, {})
            points = [
                (x, y)
                for part_id in part_ids
                for x, y, visible in [image_parts.get(part_id, (0, 0, False))]
                if visible
            ]
            if not points:
                return image.crop(bbox_box)
            return image.crop(self._part_box(points, bbox_box, image.size, scale))

        def __getitem__(self, index: int):
            row = self.rows[index]
            with Image.open(self._image_path(row)) as image:
                image = image.convert("RGB")
                bbox_box = self._bbox_box(row, image.size)
                views = [image.copy(), image.crop(bbox_box)]
                for _name, part_ids, scale in self.part_groups:
                    views.append(self._group_crop(image, row["image_id"], part_ids, bbox_box, scale))

            transformed = []
            for view_index, view in enumerate(views):
                transform = self.full_transform if view_index == 0 else self.crop_transform
                transformed.append(transform(view) if transform is not None else view)
            target = int(row["target"])
            coarse_target = self.target_to_coarse.get(target, -1)
            return torch.stack(transformed, dim=0), target, coarse_target

    return PartHierarchyNABirdsDataset


def _find_feature_head(module):
    children = list(module.named_children())
    for child_name, child in reversed(children):
        if hasattr(child, "in_features"):
            return module, child_name, int(child.in_features)
        parent, leaf_name, feature_dim = _find_feature_head(child)
        if parent is not None and leaf_name is not None:
            return parent, leaf_name, feature_dim
    return None, None, None


def _replace_child(parent, child_name: str, replacement) -> None:
    modules = getattr(parent, "_modules", None)
    if isinstance(modules, dict) and child_name in modules:
        modules[child_name] = replacement
        return
    setattr(parent, child_name, replacement)


def _strip_sequence_classifier(nn, backbone, attr_name: str) -> Optional[int]:
    if not hasattr(backbone, attr_name):
        return None
    classifier = getattr(backbone, attr_name)
    if hasattr(classifier, "in_features"):
        feature_dim = int(classifier.in_features)
        setattr(backbone, attr_name, nn.Identity())
        return feature_dim
    if hasattr(classifier, "__len__") and hasattr(classifier, "__getitem__"):
        for index in range(len(classifier) - 1, -1, -1):
            child = classifier[index]
            if hasattr(child, "in_features"):
                feature_dim = int(child.in_features)
                classifier[index] = nn.Identity()
                return feature_dim
    return None


def _strip_classifier_head(nn, backbone, candidate_names: Sequence[str]) -> int:
    for candidate_name in candidate_names:
        if not hasattr(backbone, candidate_name):
            continue
        candidate = getattr(backbone, candidate_name)
        if hasattr(candidate, "in_features"):
            feature_dim = int(candidate.in_features)
            setattr(backbone, candidate_name, nn.Identity())
            return feature_dim

        parent, child_name, feature_dim = _find_feature_head(candidate)
        if parent is not None and child_name is not None and feature_dim is not None:
            _replace_child(parent, child_name, nn.Identity())
            return int(feature_dim)

    raise ValueError(
        "Could not infer classifier feature dimension from "
        f"{backbone.__class__.__name__}. Expected one of: {', '.join(candidate_names)}"
    )


def _make_backbone(nn, models, model_name: str, pretrained: bool):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    if not hasattr(models, model_name):
        raise ValueError(f"torchvision.models does not expose {model_name}")

    weights = "DEFAULT" if pretrained else None
    backbone = getattr(models, model_name)(weights=weights)

    if model_name == "resnet50":
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    if model_name.startswith(("convnext_", "efficientnet_")):
        feature_dim = _strip_sequence_classifier(nn, backbone, "classifier")
        if feature_dim is not None:
            return backbone, feature_dim
    if model_name.startswith("swin_") and hasattr(backbone, "head"):
        feature_dim = int(backbone.head.in_features)
        backbone.head = nn.Identity()
        return backbone, feature_dim

    feature_dim = _strip_classifier_head(nn, backbone, ("fc", "head", "classifier", "heads"))
    return backbone, feature_dim


def build_part_hierarchy_model(
    torch,
    nn,
    models,
    model_name: str,
    num_classes: int,
    num_views: int,
    pretrained: bool,
    dropout: float,
    fusion: str,
    coarse_num_classes: int,
):
    class PartHierarchyFusedClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, feature_dim = _make_backbone(nn, models, model_name, pretrained)
            self.num_views = num_views
            self.fusion = fusion
            self.feature_dim = feature_dim
            if fusion == "concat":
                fused_dim = feature_dim * num_views
                self.gate = None
            elif fusion == "gated":
                hidden_dim = max(64, feature_dim // 4)
                fused_dim = feature_dim
                self.gate = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
            else:
                raise ValueError(f"fusion must be concat or gated, got {fusion!r}")
            self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(fused_dim, num_classes))
            self.coarse_classifier = None
            if coarse_num_classes > 0:
                self.coarse_classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(fused_dim, coarse_num_classes),
                )

        @staticmethod
        def _flatten_features(features):
            if features.ndim > 2:
                return torch.flatten(features, start_dim=1)
            return features

        def _fuse(self, features):
            if self.fusion == "concat":
                return features.reshape(features.size(0), features.size(1) * features.size(2))
            weights = torch.softmax(self.gate(features), dim=1)
            return (features * weights).sum(dim=1)

        def forward(self, views):
            batch_size, num_views, channels, height, width = views.shape
            if num_views != self.num_views:
                raise ValueError(f"Expected {self.num_views} views, got {num_views}")
            flat_views = views.reshape(batch_size * num_views, channels, height, width)
            features = self._flatten_features(self.backbone(flat_views))
            features = features.reshape(batch_size, num_views, self.feature_dim)
            fused_features = self._fuse(features)
            fine_logits = self.classifier(fused_features)
            coarse_logits = None
            if self.coarse_classifier is not None:
                coarse_logits = self.coarse_classifier(fused_features)
            return fine_logits, coarse_logits

    return PartHierarchyFusedClassifier()


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


def _hierarchy_loss(torch, coarse_criterion, coarse_logits, coarse_targets, weight: float):
    if weight <= 0.0 or coarse_logits is None:
        return None, 0
    valid = coarse_targets >= 0
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return None, 0
    return coarse_criterion(coarse_logits[valid], coarse_targets[valid]), valid_count


def train_one_epoch(
    torch,
    model,
    loader,
    optimizer,
    fine_criterion,
    coarse_criterion,
    device,
    scaler,
    use_amp: bool,
    hierarchy_loss_weight: float,
    grad_clip_norm: float,
):
    model.train()
    totals = {
        "loss": 0.0,
        "fine_loss": 0.0,
        "coarse_loss": 0.0,
        "top1": 0.0,
        "top5": 0.0,
        "coarse_top1": 0.0,
    }
    total_seen = 0
    total_coarse_seen = 0

    for views, targets, coarse_targets in loader:
        views = views.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        coarse_targets = coarse_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, coarse_logits = model(views)
            fine_loss = fine_criterion(logits, targets)
            coarse_loss, valid_count = _hierarchy_loss(
                torch, coarse_criterion, coarse_logits, coarse_targets, hierarchy_loss_weight
            )
            loss = fine_loss if coarse_loss is None else fine_loss + hierarchy_loss_weight * coarse_loss

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        top1, top5 = accuracy_at_k(torch, logits.detach(), targets, topk=(1, 5))
        totals["loss"] += loss.item() * batch_size
        totals["fine_loss"] += fine_loss.item() * batch_size
        totals["top1"] += top1 * batch_size
        totals["top5"] += top5 * batch_size
        total_seen += batch_size

        if coarse_loss is not None and coarse_logits is not None:
            valid = coarse_targets >= 0
            coarse_top1 = accuracy_at_k(torch, coarse_logits.detach()[valid], coarse_targets[valid], topk=(1,))[0]
            totals["coarse_loss"] += coarse_loss.item() * valid_count
            totals["coarse_top1"] += coarse_top1 * valid_count
            total_coarse_seen += valid_count

    metrics = {
        "loss": totals["loss"] / total_seen,
        "fine_loss": totals["fine_loss"] / total_seen,
        "top1": totals["top1"] / total_seen,
        "top5": totals["top5"] / total_seen,
    }
    if total_coarse_seen > 0:
        metrics["coarse_loss"] = totals["coarse_loss"] / total_coarse_seen
        metrics["coarse_top1"] = totals["coarse_top1"] / total_coarse_seen
    return metrics


def evaluate(
    torch,
    model,
    loader,
    fine_criterion,
    coarse_criterion,
    device,
    hierarchy_loss_weight: float,
    compute_macro_f1: bool,
):
    model.eval()
    totals = {
        "loss": 0.0,
        "fine_loss": 0.0,
        "coarse_loss": 0.0,
        "top1": 0.0,
        "top5": 0.0,
        "coarse_top1": 0.0,
    }
    total_seen = 0
    total_coarse_seen = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for views, targets, coarse_targets in loader:
            views = views.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            coarse_targets = coarse_targets.to(device, non_blocking=True)

            logits, coarse_logits = model(views)
            fine_loss = fine_criterion(logits, targets)
            coarse_loss, valid_count = _hierarchy_loss(
                torch, coarse_criterion, coarse_logits, coarse_targets, hierarchy_loss_weight
            )
            loss = fine_loss if coarse_loss is None else fine_loss + hierarchy_loss_weight * coarse_loss
            top1, top5 = accuracy_at_k(torch, logits, targets, topk=(1, 5))

            batch_size = targets.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["fine_loss"] += fine_loss.item() * batch_size
            totals["top1"] += top1 * batch_size
            totals["top5"] += top5 * batch_size
            total_seen += batch_size

            if coarse_loss is not None and coarse_logits is not None:
                valid = coarse_targets >= 0
                coarse_top1 = accuracy_at_k(torch, coarse_logits[valid], coarse_targets[valid], topk=(1,))[0]
                totals["coarse_loss"] += coarse_loss.item() * valid_count
                totals["coarse_top1"] += coarse_top1 * valid_count
                total_coarse_seen += valid_count

            if compute_macro_f1:
                all_targets.extend(targets.cpu().tolist())
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "loss": totals["loss"] / total_seen,
        "fine_loss": totals["fine_loss"] / total_seen,
        "top1": totals["top1"] / total_seen,
        "top5": totals["top5"] / total_seen,
    }
    if total_coarse_seen > 0:
        metrics["coarse_loss"] = totals["coarse_loss"] / total_coarse_seen
        metrics["coarse_top1"] = totals["coarse_top1"] / total_coarse_seen
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
    parser = argparse.ArgumentParser(description="Train a part-guided hierarchy-fused NABirds classifier.")
    parser.add_argument("--manifest-dir", default="reports/milestone2")
    parser.add_argument("--train-csv", default=None, help="Defaults to <manifest-dir>/nabirds_train.csv.")
    parser.add_argument("--val-csv", default=None, help="Defaults to <manifest-dir>/nabirds_val.csv.")
    parser.add_argument("--dataset-root", default="nabirds", help="Directory containing classes.txt, hierarchy.txt, parts/.")
    parser.add_argument("--path-root", default=".", help="Base directory for relative image_path entries.")
    parser.add_argument("--out-dir", default="reports/milestone2/runs")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet50")
    parser.add_argument("--num-views", type=int, choices=[4, 5], default=5, help="Total views: 4 or 5.")
    parser.add_argument("--num-classes", type=int, default=555)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--full-min-scale", type=float, default=0.55)
    parser.add_argument("--crop-min-scale", type=float, default=0.75)
    parser.add_argument("--crop-padding", type=float, default=0.0, help="Fractional padding around clipped bbox crop.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hierarchy-loss-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument("--macro-f1", action="store_true", help="Compute validation macro-F1 when scikit-learn is available.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional smoke-test cap for train rows.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional smoke-test cap for val rows.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--resume", default="", help="Optional last.pt/best.pt checkpoint to continue training from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hierarchy_loss_weight < 0:
        raise SystemExit("--hierarchy-loss-weight must be nonnegative.")
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
    all_rows = train_rows + val_rows

    train_min, train_max, train_unique = _target_summary(train_rows)
    val_min, val_max, val_unique = _target_summary(val_rows)
    if max(train_max, val_max) >= args.num_classes:
        raise SystemExit(
            f"Found target id {max(train_max, val_max)} but --num-classes is {args.num_classes}. "
            "Increase --num-classes or rebuild the manifests."
        )

    dataset_root = Path(args.dataset_root)
    part_locations, part_status = _load_part_locations(dataset_root / "parts" / "part_locs.txt")
    target_to_coarse, hierarchy_info = build_hierarchy_mapping(dataset_root, all_rows)
    hierarchy_enabled = bool(
        args.hierarchy_loss_weight > 0.0
        and hierarchy_info.get("available")
        and int(hierarchy_info.get("coarse_class_count", 0)) > 1
    )
    if args.hierarchy_loss_weight > 0.0 and not hierarchy_enabled:
        print(f"Hierarchy loss requested but disabled: {hierarchy_info.get('reason')}")

    view_names = _view_names(args.num_views)
    Dataset = make_dataset_class(torch, Image)
    train_ds = Dataset(
        train_rows,
        path_root=Path(args.path_root),
        part_locations=part_locations,
        target_to_coarse=target_to_coarse if hierarchy_enabled else {},
        num_views=args.num_views,
        full_transform=build_transforms(transforms, args.image_size, train=True, min_scale=args.full_min_scale),
        crop_transform=build_transforms(transforms, args.image_size, train=True, min_scale=args.crop_min_scale),
        crop_padding=args.crop_padding,
    )
    val_ds = Dataset(
        val_rows,
        path_root=Path(args.path_root),
        part_locations=part_locations,
        target_to_coarse=target_to_coarse if hierarchy_enabled else {},
        num_views=args.num_views,
        full_transform=build_transforms(transforms, args.image_size, train=False, min_scale=args.full_min_scale),
        crop_transform=build_transforms(transforms, args.image_size, train=False, min_scale=args.crop_min_scale),
        crop_padding=args.crop_padding,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, **_dataloader_kwargs(device, args, shuffle=True))
    val_loader = torch.utils.data.DataLoader(val_ds, **_dataloader_kwargs(device, args, shuffle=False))

    coarse_num_classes = int(hierarchy_info.get("coarse_class_count", 0)) if hierarchy_enabled else 0
    model = build_part_hierarchy_model(
        torch,
        nn,
        models,
        model_name=args.model,
        num_classes=args.num_classes,
        num_views=args.num_views,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        fusion=args.fusion,
        coarse_num_classes=coarse_num_classes,
    ).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise SystemExit("No trainable parameters found.")

    fine_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    coarse_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    resume_path = Path(args.resume) if args.resume else None
    run_name = resume_path.parent.name if resume_path else (
        f"part_hierarchy_{args.model}_{args.fusion}_{args.num_views}views_"
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
            "view_names": view_names,
            "part_locs_status": part_status,
            "part_image_count": len(part_locations),
            "coarse_class_count": coarse_num_classes,
            "hierarchy_loss_enabled": hierarchy_enabled,
            "hierarchy_info": hierarchy_info,
            "resume": str(resume_path) if resume_path else "",
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        "Loaded "
        f"{len(train_rows)} train / {len(val_rows)} val rows; "
        f"train targets {train_min}-{train_max} ({train_unique} unique), "
        f"val targets {val_min}-{val_max} ({val_unique} unique)."
    )
    print(
        f"Model: {args.model}; fusion: {args.fusion}; views: {view_names}; "
        f"hierarchy_loss: {hierarchy_enabled}; device: {device}; output: {out_dir}"
    )

    history = []
    best_top1 = -1.0
    start_epoch = 1
    if resume_path:
        if not resume_path.exists():
            raise SystemExit(f"--resume checkpoint not found: {resume_path}")
        try:
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model"])
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        history_path = out_dir / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = []
        if history:
            best_top1 = max(float(row.get("val", {}).get("top1", -1.0)) for row in history)
        else:
            best_top1 = float(resume_checkpoint.get("val_metrics", {}).get("top1", -1.0))
        print(f"Resumed from {resume_path} at epoch {start_epoch}; previous best validation top-1: {best_top1:.4f}")
    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}; requested --epochs {args.epochs}. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_metrics = train_one_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            fine_criterion,
            coarse_criterion,
            device,
            scaler,
            use_amp=use_amp,
            hierarchy_loss_weight=args.hierarchy_loss_weight if hierarchy_enabled else 0.0,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = evaluate(
            torch,
            model,
            val_loader,
            fine_criterion,
            coarse_criterion,
            device,
            hierarchy_loss_weight=args.hierarchy_loss_weight if hierarchy_enabled else 0.0,
            compute_macro_f1=args.macro_f1,
        )
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
            "config": config,
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
