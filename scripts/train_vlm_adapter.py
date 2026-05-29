#!/usr/bin/env python3
"""Train a lightweight SigLIP/SigLIP2 image-text adapter on NABirds.

The script consumes the Milestone 2 manifests and class prompt CSV:

  reports/milestone2/nabirds_train.csv
  reports/milestone2/nabirds_val.csv
  reports/milestone2/nabirds_class_prompts.csv

Torch, Pillow, and transformers imports are intentionally kept inside runtime
helpers so this file remains importable and syntax-checkable without the VLM
training stack installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


DEFAULT_MANIFEST_DIR = "reports/milestone2"
DEFAULT_PROMPTS = "reports/milestone2/nabirds_class_prompts.csv"
DEFAULT_HARD_NEGATIVES = "reports/milestone2/nabirds_hard_negative_groups.csv"
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_FALLBACK_MODEL = "google/siglip-base-patch16-224"
EXPECTED_NABIRDS_CLASSES = 555


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


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
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} contains no data rows.")
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


def _require_runtime_deps():
    missing: List[str] = []

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        torch = None
        nn = None
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
            "VLM adapter training dependencies are missing: "
            f"{unique_missing}.\n"
            "Install a compatible PyTorch build plus transformers and Pillow before training.\n"
            "Example CPU install:\n"
            "  python3 -m pip install torch transformers pillow tqdm scikit-learn\n"
            "For CUDA or Apple Silicon acceleration, use the PyTorch install selector for the correct torch wheel."
        )

    return torch, nn, Image, AutoModel, AutoProcessor


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested a CUDA device, but torch.cuda.is_available() is false.")
    if device.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise SystemExit("Requested an MPS device, but torch.backends.mps.is_available() is false.")
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


def _as_int(value: str) -> int:
    return int(round(float(value)))


def _load_prompt_rows(
    path: Path,
    expected_num_classes: int,
) -> Tuple[List[int], List[str], Dict[int, List[str]], int]:
    rows = _read_csv(path, required_columns=("target", "prompt"))
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
        if row.get("class_name", "").strip():
            class_name_by_target[target] = row["class_name"].strip()

    class_targets = sorted(prompts_by_target)
    if expected_num_classes and len(class_targets) != expected_num_classes:
        raise SystemExit(
            f"{path}: expected {expected_num_classes} unique targets, found {len(class_targets)}. "
            "Pass --expected-num-classes 0 to disable this check."
        )
    class_names = [class_name_by_target.get(target, str(target)) for target in class_targets]
    return class_targets, class_names, dict(prompts_by_target), len(rows)


def _load_manifest_rows(
    path: Path,
    limit: Optional[int],
    valid_targets: Sequence[int],
) -> List[Dict[str, str]]:
    rows = _limit_rows(_read_csv(path, required_columns=("image_path", "target")), limit)
    valid_target_set = set(valid_targets)
    manifest_targets = set()
    for row in rows:
        try:
            target = int(row["target"])
        except ValueError as exc:
            raise SystemExit(f"{path}: invalid integer target {row['target']!r}") from exc
        manifest_targets.add(target)

    unknown_targets = sorted(target for target in manifest_targets if target not in valid_target_set)
    if unknown_targets:
        preview = ", ".join(str(value) for value in unknown_targets[:10])
        raise SystemExit(f"{path}: manifest target(s) missing from prompt CSV: {preview}")
    return rows


def _target_summary(rows: Sequence[Mapping[str, str]]) -> Tuple[int, int, int]:
    targets = [int(row["target"]) for row in rows]
    return min(targets), max(targets), len(set(targets))


def _load_hard_negative_groups(
    path: Optional[Path],
    target_to_index: Mapping[int, int],
    max_negatives: int,
) -> Dict[int, List[int]]:
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"Hard-negative CSV not found: {path}")

    rows = _read_csv(path, required_columns=("targets",))
    groups: Dict[int, List[int]] = defaultdict(list)
    for row in rows:
        raw_values = [value.strip() for value in row["targets"].split("|") if value.strip()]
        targets: List[int] = []
        for value in raw_values:
            try:
                target = int(value)
            except ValueError as exc:
                raise SystemExit(f"{path}: invalid target in hard-negative group: {value!r}") from exc
            if target in target_to_index:
                targets.append(target_to_index[target])

        unique_targets = sorted(set(targets))
        for target_index in unique_targets:
            negatives = [candidate for candidate in unique_targets if candidate != target_index]
            if max_negatives > 0:
                negatives = negatives[:max_negatives]
            groups[target_index].extend(negatives)

    deduped = {target: sorted(set(negatives)) for target, negatives in groups.items() if negatives}
    return deduped


class NABirdsVLMManifestDataset:
    """Small picklable dataset that opens PIL images only at runtime."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        target_to_index: Mapping[int, int],
        dataset_root: Path,
        path_root: Path,
        input_mode: str,
        crop_padding: float,
    ):
        if input_mode not in {"full", "bbox"}:
            raise ValueError(f"input_mode must be 'full' or 'bbox', got {input_mode!r}")
        self.rows = [dict(row) for row in rows]
        self.target_to_index = dict(target_to_index)
        self.dataset_root = Path(dataset_root)
        self.path_root = Path(path_root)
        self.input_mode = input_mode
        self.crop_padding = max(0.0, crop_padding)

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, row: Mapping[str, str]) -> Path:
        image_path = Path(row["image_path"]).expanduser()
        candidates: List[Path] = []
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.extend([image_path, self.path_root / image_path])

        rel_path = row.get("rel_path", "").strip()
        if rel_path:
            candidates.append(self.dataset_root / "images" / rel_path)
            candidates.append(self.path_root / self.dataset_root / "images" / rel_path)

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def _crop_box(self, row: Mapping[str, str], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        missing = [column for column in ("clip_x", "clip_y", "clip_w", "clip_h") if not row.get(column, "")]
        if missing:
            raise ValueError(f"bbox input mode requires manifest columns: {', '.join(missing)}")

        image_w, image_h = image_size
        x = _as_int(row["clip_x"])
        y = _as_int(row["clip_y"])
        w = _as_int(row["clip_w"])
        h = _as_int(row["clip_h"])
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid clipped bbox width/height ({w}, {h})")

        pad_x = int(round(w * self.crop_padding))
        pad_y = int(round(h * self.crop_padding))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image_w, x + w + pad_x)
        bottom = min(image_h, y + h + pad_y)
        if right <= left or bottom <= top:
            return 0, 0, image_w, image_h
        return left, top, right, bottom

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        from PIL import Image

        row = self.rows[index]
        path = self._image_path(row)
        if not path.is_file():
            raise FileNotFoundError(str(path))

        with Image.open(path) as handle:
            image = handle.convert("RGB")
            if self.input_mode == "bbox":
                image = image.crop(self._crop_box(row, image.size))

        raw_target = int(row["target"])
        return image, self.target_to_index[raw_target]


def vlm_collate(batch: Sequence[Tuple[Any, int]]) -> Tuple[List[Any], Any]:
    import torch

    images = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, targets


def _dataloader_kwargs(device: Any, args: argparse.Namespace, shuffle: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": vlm_collate,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


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
    text_processor_kwargs: Optional[Mapping[str, Any]] = None,
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
    model_was_training = model.training
    model.eval()
    processor_kwargs = dict(text_processor_kwargs or {"padding": True, "truncation": True, "return_tensors": "pt"})
    with torch.inference_mode():
        for prompt_batch in iterator:
            inputs = processor(text=list(prompt_batch), **processor_kwargs)
            inputs = _move_inputs(inputs, device=device, dtype=dtype)
            text_features = _pooled_feature_tensor(model.get_text_features(**inputs), "text")
            features.append(_normalize(torch, text_features).detach().float().cpu())
    if model_was_training:
        model.train()

    all_prompt_features = torch.cat([feature.clone() for feature in features], dim=0)
    class_features: List[Any] = []
    for target in class_targets:
        start, end = target_spans[target]
        prompt_features = all_prompt_features[start:end]
        if aggregation == "first":
            class_feature = prompt_features[0]
        else:
            class_feature = prompt_features.mean(dim=0)
        class_features.append(_normalize(torch, class_feature.unsqueeze(0)).squeeze(0))

    return torch.stack(class_features, dim=0)


def _is_siglip_model(model_name: str, model: Optional[Any] = None) -> bool:
    values: List[str] = [model_name]
    if model is not None:
        values.append(type(model).__name__)
        config = getattr(model, "config", None)
        if config is not None:
            values.append(str(getattr(config, "model_type", "")))
            architectures = getattr(config, "architectures", None) or []
            values.extend(str(value) for value in architectures)
    return "siglip" in " ".join(values).casefold()


def _text_processor_kwargs(
    model_name: str,
    text_padding: str,
    text_max_length: int,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    padding: Any
    max_length: Optional[int]
    if text_padding == "auto":
        if _is_siglip_model(model_name, model):
            padding = "max_length"
            max_length = text_max_length or 64
        else:
            padding = True
            max_length = None
    elif text_padding == "longest":
        padding = True
        max_length = None
    else:
        padding = "max_length"
        max_length = text_max_length or 64

    kwargs: Dict[str, Any] = {
        "padding": padding,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length:
        kwargs["max_length"] = max_length
    return kwargs


def _encode_image_features(
    torch: Any,
    model: Any,
    processor: Any,
    images: Sequence[Any],
    device: Any,
    dtype: Optional[Any],
    train_image_encoder: bool,
) -> Any:
    if not hasattr(model, "get_image_features"):
        raise SystemExit(
            "The loaded model does not expose get_image_features(). "
            "Use a SigLIP/SigLIP2 checkpoint supported by transformers AutoModel."
        )

    inputs = processor(images=list(images), return_tensors="pt")
    inputs = _move_inputs(inputs, device=device, dtype=dtype)
    if train_image_encoder:
        image_features = _pooled_feature_tensor(model.get_image_features(**inputs), "image")
    else:
        with torch.inference_mode():
            image_features = _pooled_feature_tensor(model.get_image_features(**inputs), "image")
        image_features = image_features.detach().clone()
    return _normalize(torch, image_features).float()


def build_alignment_model(
    torch: Any,
    nn: Any,
    image_dim: int,
    text_dim: int,
    projection_dim: int,
    adapter_mode: str,
    adapter_hidden_dim: int,
    dropout: float,
    shared_adapter: bool,
    init_temperature: float,
    learnable_temperature: bool,
    class_bias: bool,
    num_classes: int,
) -> Any:
    class ProjectionHead(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            output_dim = projection_dim or input_dim
            self.output_dim = output_dim
            self.adapter_mode = adapter_mode

            if adapter_mode == "none":
                if input_dim != output_dim:
                    raise ValueError("--adapter-mode none requires --projection-dim 0 or an input-sized projection.")
                self.net = nn.Identity()
            elif adapter_mode == "linear":
                self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))
            elif adapter_mode == "mlp":
                hidden_dim = adapter_hidden_dim or max(128, min(1024, input_dim))
                self.net = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
            elif adapter_mode == "residual":
                if input_dim != output_dim:
                    raise ValueError("--adapter-mode residual requires --projection-dim 0 or an input-sized projection.")
                hidden_dim = adapter_hidden_dim or max(64, input_dim // 4)
                self.norm = nn.LayerNorm(input_dim)
                self.down = nn.Linear(input_dim, hidden_dim)
                self.up = nn.Linear(hidden_dim, input_dim)
                self.dropout = nn.Dropout(dropout)
                self.activation = nn.GELU()
                self.net = None
            else:
                raise ValueError(f"Unsupported adapter mode: {adapter_mode}")

        def forward(self, features):
            if self.adapter_mode == "residual":
                residual = self.up(self.dropout(self.activation(self.down(self.norm(features)))))
                return features + residual
            return self.net(features)

    class VLMAlignmentAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            if init_temperature <= 0:
                raise ValueError("--init-temperature must be positive.")
            if shared_adapter and image_dim != text_dim:
                raise ValueError("--shared-adapter requires image/text feature dimensions to match.")

            self.image_head = ProjectionHead(image_dim)
            if shared_adapter:
                self.text_head = self.image_head
            else:
                self.text_head = ProjectionHead(text_dim)

            logit_scale_value = math.log(1.0 / init_temperature)
            self.logit_scale = nn.Parameter(torch.tensor(logit_scale_value, dtype=torch.float32))
            self.logit_scale.requires_grad_(learnable_temperature)
            self.class_bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32)) if class_bias else None

        def encode_image(self, image_features):
            return _normalize(torch, self.image_head(image_features.float()))

        def encode_text(self, text_features):
            return _normalize(torch, self.text_head(text_features.float()))

        def forward(self, image_features, class_text_features):
            image_z = self.encode_image(image_features)
            text_z = self.encode_text(class_text_features)
            scale = self.logit_scale.exp().clamp(max=100.0)
            logits = scale * (image_z @ text_z.t())
            if self.class_bias is not None:
                logits = logits + self.class_bias
            return logits, image_z, text_z

    return VLMAlignmentAdapter()


def accuracy_at_k(torch: Any, logits: Any, targets: Any, topk: Sequence[int] = (1, 5)) -> List[float]:
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    values = []
    for requested_k in topk:
        k = min(requested_k, logits.size(1))
        values.append(correct[:k].reshape(-1).float().sum(0).item() / targets.size(0))
    return values


def _multi_positive_contrastive_loss(torch: Any, logits: Any, targets: Any) -> Any:
    if targets.numel() < 2:
        return logits.new_zeros(())
    positives = targets.unsqueeze(0).eq(targets.unsqueeze(1))
    neg_inf = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positives, neg_inf)
    image_to_text = -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()
    text_to_image = -(torch.logsumexp(positive_logits.t(), dim=1) - torch.logsumexp(logits.t(), dim=1)).mean()
    return 0.5 * (image_to_text + text_to_image)


def _hard_negative_loss(torch: Any, logits: Any, targets: Any, hard_negatives: Mapping[int, Sequence[int]]) -> Any:
    losses = []
    for row_index, target_tensor in enumerate(targets):
        target = int(target_tensor.item())
        negatives = hard_negatives.get(target, ())
        if not negatives:
            continue
        candidates = [target] + [candidate for candidate in negatives if candidate != target]
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
        local_logits = logits[row_index, candidate_tensor].unsqueeze(0)
        local_target = torch.zeros(1, dtype=torch.long, device=logits.device)
        losses.append(torch.nn.functional.cross_entropy(local_logits, local_target))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    true_positive = [0 for _ in range(num_classes)]
    false_positive = [0 for _ in range(num_classes)]
    false_negative = [0 for _ in range(num_classes)]

    for target, pred in zip(y_true, y_pred):
        if target == pred:
            true_positive[target] += 1
        else:
            false_negative[target] += 1
            false_positive[pred] += 1

    f1_values: List[float] = []
    for label in range(num_classes):
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]
        denominator = (2 * tp) + fp + fn
        f1_values.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return sum(f1_values) / len(f1_values)


def _loss_terms(
    torch: Any,
    nn: Any,
    logits: Any,
    targets: Any,
    image_z: Any,
    text_z: Any,
    criterion: Any,
    contrastive_weight: float,
    hard_negative_weight: float,
    hard_negatives: Mapping[int, Sequence[int]],
) -> Tuple[Any, Dict[str, float]]:
    ce_loss = criterion(logits, targets)
    total_loss = ce_loss
    terms = {"ce_loss": float(ce_loss.detach().item())}

    if contrastive_weight > 0:
        target_text_z = text_z.index_select(0, targets)
        contrastive_logits = adapter_scale_logits(torch, image_z, target_text_z)
        contrastive_loss = _multi_positive_contrastive_loss(torch, contrastive_logits, targets)
        total_loss = total_loss + (contrastive_weight * contrastive_loss)
        terms["contrastive_loss"] = float(contrastive_loss.detach().item())

    if hard_negative_weight > 0:
        hard_loss = _hard_negative_loss(torch, logits, targets, hard_negatives)
        total_loss = total_loss + (hard_negative_weight * hard_loss)
        terms["hard_negative_loss"] = float(hard_loss.detach().item())

    terms["loss"] = float(total_loss.detach().item())
    return total_loss, terms


def adapter_scale_logits(torch: Any, image_z: Any, text_z: Any, scale: float = 10.0) -> Any:
    return torch.tensor(scale, dtype=image_z.dtype, device=image_z.device) * (image_z @ text_z.t())


def train_one_epoch(
    torch: Any,
    nn: Any,
    vlm_model: Any,
    processor: Any,
    adapter: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    criterion: Any,
    class_text_features: Any,
    hard_negatives: Mapping[int, Sequence[int]],
    device: Any,
    dtype: Optional[Any],
    use_amp: bool,
    train_image_encoder: bool,
    contrastive_weight: float,
    hard_negative_weight: float,
    grad_clip_norm: float,
    show_progress: bool,
) -> Dict[str, float]:
    adapter.train()
    vlm_model.train(mode=train_image_encoder)

    total_seen = 0
    total_loss = 0.0
    total_ce = 0.0
    total_contrastive = 0.0
    total_hard = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    batches = _progress(loader, total=len(loader), label="train", enabled=show_progress)

    for images, targets in batches:
        targets = targets.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            image_features = _encode_image_features(
                torch=torch,
                model=vlm_model,
                processor=processor,
                images=images,
                device=device,
                dtype=dtype,
                train_image_encoder=train_image_encoder,
            )
            logits, image_z, text_z = adapter(image_features, class_text_features)
            loss, terms = _loss_terms(
                torch=torch,
                nn=nn,
                logits=logits,
                targets=targets,
                image_z=image_z,
                text_z=text_z,
                criterion=criterion,
                contrastive_weight=contrastive_weight,
                hard_negative_weight=hard_negative_weight,
                hard_negatives=hard_negatives,
            )

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        top1, top5 = accuracy_at_k(torch, logits.detach(), targets, topk=(1, 5))
        total_seen += batch_size
        total_loss += terms["loss"] * batch_size
        total_ce += terms["ce_loss"] * batch_size
        total_contrastive += terms.get("contrastive_loss", 0.0) * batch_size
        total_hard += terms.get("hard_negative_loss", 0.0) * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size

    metrics = {
        "loss": total_loss / total_seen,
        "ce_loss": total_ce / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
    }
    if contrastive_weight > 0:
        metrics["contrastive_loss"] = total_contrastive / total_seen
    if hard_negative_weight > 0:
        metrics["hard_negative_loss"] = total_hard / total_seen
    return metrics


def evaluate(
    torch: Any,
    nn: Any,
    vlm_model: Any,
    processor: Any,
    adapter: Any,
    loader: Any,
    criterion: Any,
    class_text_features: Any,
    hard_negatives: Mapping[int, Sequence[int]],
    device: Any,
    dtype: Optional[Any],
    contrastive_weight: float,
    hard_negative_weight: float,
    compute_macro_f1: bool,
    show_progress: bool,
) -> Dict[str, float]:
    adapter.eval()
    vlm_model.eval()

    total_seen = 0
    total_loss = 0.0
    total_ce = 0.0
    total_contrastive = 0.0
    total_hard = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    batches = _progress(loader, total=len(loader), label="val", enabled=show_progress)

    with torch.inference_mode():
        for images, targets in batches:
            targets = targets.to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=False):
                image_features = _encode_image_features(
                    torch=torch,
                    model=vlm_model,
                    processor=processor,
                    images=images,
                    device=device,
                    dtype=dtype,
                    train_image_encoder=False,
                )
                logits, image_z, text_z = adapter(image_features, class_text_features)
                loss, terms = _loss_terms(
                    torch=torch,
                    nn=nn,
                    logits=logits,
                    targets=targets,
                    image_z=image_z,
                    text_z=text_z,
                    criterion=criterion,
                    contrastive_weight=contrastive_weight,
                    hard_negative_weight=hard_negative_weight,
                    hard_negatives=hard_negatives,
                )

            batch_size = targets.size(0)
            top1, top5 = accuracy_at_k(torch, logits, targets, topk=(1, 5))
            preds = logits.argmax(dim=1)
            total_seen += batch_size
            total_loss += terms["loss"] * batch_size
            total_ce += terms["ce_loss"] * batch_size
            total_contrastive += terms.get("contrastive_loss", 0.0) * batch_size
            total_hard += terms.get("hard_negative_loss", 0.0) * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            if compute_macro_f1:
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

    metrics = {
        "loss": total_loss / total_seen,
        "ce_loss": total_ce / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
    }
    if contrastive_weight > 0:
        metrics["contrastive_loss"] = total_contrastive / total_seen
    if hard_negative_weight > 0:
        metrics["hard_negative_loss"] = total_hard / total_seen
    if compute_macro_f1:
        metrics["macro_f1"] = _macro_f1(y_true, y_pred, num_classes=class_text_features.size(0))
    return metrics


def _infer_image_dim(
    torch: Any,
    vlm_model: Any,
    processor: Any,
    dataset: NABirdsVLMManifestDataset,
    device: Any,
    dtype: Optional[Any],
) -> int:
    image, _target = dataset[0]
    features = _encode_image_features(
        torch=torch,
        model=vlm_model,
        processor=processor,
        images=[image],
        device=device,
        dtype=dtype,
        train_image_encoder=False,
    )
    return int(features.shape[-1])


def _load_vlm_model(
    AutoModel: Any,
    AutoProcessor: Any,
    model_name: str,
) -> Tuple[Any, Any]:
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError as exc:
        raise SystemExit(
            f"Could not load model or processor for {model_name!r}.\n"
            "If this is the first run, the checkpoint may need to be downloaded from Hugging Face.\n"
            f"Fallback SigLIP checkpoint configured for this project: {DEFAULT_FALLBACK_MODEL}"
        ) from exc
    return model, processor


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train trainable image/text projection adapters on frozen SigLIP/SigLIP2 NABirds features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR, help="Directory containing NABirds train/val manifests.")
    parser.add_argument("--train-csv", default=None, help="Defaults to <manifest-dir>/nabirds_train.csv.")
    parser.add_argument("--val-csv", default=None, help="Defaults to <manifest-dir>/nabirds_val.csv.")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="NABirds class prompt CSV.")
    parser.add_argument("--hard-negative-csv", default=DEFAULT_HARD_NEGATIVES, help="Family-variant hard-negative CSV.")
    parser.add_argument("--dataset-root", default="nabirds", help="Dataset root used to resolve rel_path fallback.")
    parser.add_argument("--path-root", default=".", help="Base directory for relative image_path entries.")
    parser.add_argument("--out-dir", default="reports/milestone2/runs", help="Output directory for adapter checkpoints.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face SigLIP/SigLIP2 model id or local path.")
    parser.add_argument("--input-mode", choices=("full", "bbox"), default="full", help="Train on full images or clipped bbox crops.")
    parser.add_argument("--crop-padding", type=_nonnegative_float, default=0.0, help="Fractional padding around clipped bbox crop.")
    parser.add_argument("--prompt-aggregation", choices=("mean", "first"), default="mean", help="How to aggregate multiple prompts per class.")
    parser.add_argument("--expected-num-classes", type=int, default=EXPECTED_NABIRDS_CLASSES, help="Expected unique class targets; use 0 to disable.")
    parser.add_argument("--adapter-mode", choices=("linear", "mlp", "residual", "none"), default="residual", help="Trainable projection/adaptation head.")
    parser.add_argument("--projection-dim", type=int, default=0, help="Adapter output dimension; 0 keeps the VLM feature dimension.")
    parser.add_argument("--adapter-hidden-dim", type=int, default=0, help="Hidden dimension for mlp/residual adapters; 0 chooses a conservative default.")
    parser.add_argument("--dropout", type=_nonnegative_float, default=0.1, help="Dropout inside MLP/residual adapters.")
    parser.add_argument("--shared-adapter", action="store_true", help="Use one adapter for image and text features when dimensions match.")
    parser.add_argument("--class-bias", action="store_true", help="Learn one scalar bias per class.")
    parser.add_argument("--init-temperature", type=float, default=0.07, help="Initial similarity softmax temperature.")
    parser.add_argument("--fixed-temperature", action="store_true", help="Do not learn the logit temperature.")
    parser.add_argument("--train-image-encoder", action="store_true", help="Experimental: backprop through the image encoder in addition to adapters.")
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--text-batch-size", type=_positive_int, default=256)
    parser.add_argument("--text-padding", choices=("auto", "longest", "max_length"), default="auto")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--vlm-lr", type=float, default=1e-6, help="Used only with --train-image-encoder.")
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=1e-4)
    parser.add_argument("--label-smoothing", type=_nonnegative_float, default=0.05)
    parser.add_argument("--contrastive-weight", type=_nonnegative_float, default=0.0, help="Weight for optional in-batch multi-positive contrastive loss.")
    parser.add_argument("--hard-negative-weight", type=_nonnegative_float, default=0.0, help="Weight for optional family-group hard-negative CE loss.")
    parser.add_argument("--max-hard-negatives", type=int, default=8, help="Max hard negatives per class; <=0 uses all group variants.")
    parser.add_argument("--grad-clip-norm", type=_nonnegative_float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision for the VLM image encoder.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto", help="VLM floating dtype.")
    parser.add_argument("--macro-f1", action="store_true", help="Compute validation macro-F1.")
    parser.add_argument("--limit-train", type=_positive_int, default=None, help="Optional smoke-test cap for train rows.")
    parser.add_argument("--limit-val", type=_positive_int, default=None, help="Optional smoke-test cap for val rows.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars when tqdm is installed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _seed_everything(args.seed)
    torch, nn, _Image, AutoModel, AutoProcessor = _require_runtime_deps()

    device = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype, device)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("AMP requested but only CUDA AMP is enabled by this script; continuing without AMP.")

    manifest_dir = Path(args.manifest_dir)
    train_csv = Path(args.train_csv) if args.train_csv else manifest_dir / "nabirds_train.csv"
    val_csv = Path(args.val_csv) if args.val_csv else manifest_dir / "nabirds_val.csv"
    prompt_path = Path(args.prompts)

    class_targets, class_names, prompts_by_target, prompt_rows = _load_prompt_rows(
        prompt_path,
        expected_num_classes=args.expected_num_classes,
    )
    target_to_index = {target: index for index, target in enumerate(class_targets)}
    train_rows = _load_manifest_rows(train_csv, limit=args.limit_train, valid_targets=class_targets)
    val_rows = _load_manifest_rows(val_csv, limit=args.limit_val, valid_targets=class_targets)
    train_min, train_max, train_unique = _target_summary(train_rows)
    val_min, val_max, val_unique = _target_summary(val_rows)

    hard_negative_path = Path(args.hard_negative_csv) if args.hard_negative_csv else None
    hard_negatives = (
        _load_hard_negative_groups(hard_negative_path, target_to_index, args.max_hard_negatives)
        if args.hard_negative_weight > 0
        else {}
    )

    print(f"Loading VLM: {args.model}")
    vlm_model, processor = _load_vlm_model(AutoModel, AutoProcessor, args.model)
    vlm_model.to(device=device)
    if dtype is not None:
        vlm_model.to(dtype=dtype)
    for parameter in vlm_model.parameters():
        parameter.requires_grad_(args.train_image_encoder)
    vlm_model.train(mode=args.train_image_encoder)

    class_text_features = _encode_text_features(
        torch=torch,
        model=vlm_model,
        processor=processor,
        prompts_by_target=prompts_by_target,
        class_targets=class_targets,
        device=device,
        dtype=dtype,
        text_batch_size=args.text_batch_size,
        aggregation=args.prompt_aggregation,
        show_progress=not args.no_progress,
        text_processor_kwargs=_text_processor_kwargs(args.model, args.text_padding, args.text_max_length, model=vlm_model),
    ).to(device=device)

    train_ds = NABirdsVLMManifestDataset(
        train_rows,
        target_to_index=target_to_index,
        dataset_root=Path(args.dataset_root),
        path_root=Path(args.path_root),
        input_mode=args.input_mode,
        crop_padding=args.crop_padding,
    )
    val_ds = NABirdsVLMManifestDataset(
        val_rows,
        target_to_index=target_to_index,
        dataset_root=Path(args.dataset_root),
        path_root=Path(args.path_root),
        input_mode=args.input_mode,
        crop_padding=args.crop_padding,
    )
    image_dim = _infer_image_dim(
        torch=torch,
        vlm_model=vlm_model,
        processor=processor,
        dataset=train_ds,
        device=device,
        dtype=dtype,
    )
    text_dim = int(class_text_features.shape[-1])

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

    train_loader = torch.utils.data.DataLoader(train_ds, **_dataloader_kwargs(device, args, shuffle=True))
    val_loader = torch.utils.data.DataLoader(val_ds, **_dataloader_kwargs(device, args, shuffle=False))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    param_groups = [{"params": adapter.parameters(), "lr": args.adapter_lr, "weight_decay": args.weight_decay}]
    if args.train_image_encoder:
        vlm_params = [parameter for parameter in vlm_model.parameters() if parameter.requires_grad]
        if not vlm_params:
            raise SystemExit("--train-image-encoder was set, but no VLM parameters are trainable.")
        param_groups.append({"params": vlm_params, "lr": args.vlm_lr, "weight_decay": args.weight_decay})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = (
        f"vlm_adapter_{Path(args.model).name}_{args.input_mode}_{args.adapter_mode}_"
        f"{'image_ft' if args.train_image_encoder else 'frozen'}_{int(time.time())}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "prompts": str(prompt_path),
            "device_resolved": str(device),
            "dtype_resolved": str(dtype).replace("torch.", "") if dtype is not None else None,
            "image_feature_dim": image_dim,
            "text_feature_dim": text_dim,
            "num_classes": len(class_targets),
            "num_prompt_rows": prompt_rows,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_target_min": train_min,
            "train_target_max": train_max,
            "train_unique_targets": train_unique,
            "val_target_min": val_min,
            "val_target_max": val_max,
            "val_unique_targets": val_unique,
            "hard_negative_classes": len(hard_negatives),
        }
    )
    _write_json(out_dir / "config.json", config)
    _write_json(
        out_dir / "classes.json",
        {
            "class_targets": class_targets,
            "class_names": class_names,
        },
    )

    print(
        "Loaded "
        f"{len(train_rows)} train / {len(val_rows)} val rows; "
        f"{len(class_targets)} classes; "
        f"train targets {train_min}-{train_max} ({train_unique} unique), "
        f"val targets {val_min}-{val_max} ({val_unique} unique)."
    )
    print(
        f"Feature dims image/text: {image_dim}/{text_dim}; "
        f"adapter: {args.adapter_mode}; input: {args.input_mode}; device: {device}; output: {out_dir}"
    )
    if args.hard_negative_weight > 0:
        print(f"Hard-negative loss enabled for {len(hard_negatives)} classes from {hard_negative_path}.")

    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = train_one_epoch(
            torch=torch,
            nn=nn,
            vlm_model=vlm_model,
            processor=processor,
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            class_text_features=class_text_features,
            hard_negatives=hard_negatives,
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            train_image_encoder=args.train_image_encoder,
            contrastive_weight=args.contrastive_weight,
            hard_negative_weight=args.hard_negative_weight,
            grad_clip_norm=args.grad_clip_norm,
            show_progress=not args.no_progress,
        )
        val_metrics = evaluate(
            torch=torch,
            nn=nn,
            vlm_model=vlm_model,
            processor=processor,
            adapter=adapter,
            loader=val_loader,
            criterion=criterion,
            class_text_features=class_text_features,
            hard_negatives=hard_negatives,
            device=device,
            dtype=dtype,
            contrastive_weight=args.contrastive_weight,
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
        }
        if args.train_image_encoder:
            checkpoint["vlm_model"] = vlm_model.state_dict()
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(checkpoint, out_dir / "best.pt")
        _write_json(out_dir / "history.json", {"history": history, "best_top1": best_top1})

    print(f"Best validation top-1: {best_top1:.4f}")
    print(f"Run directory: {out_dir}")


if __name__ == "__main__":
    main()
