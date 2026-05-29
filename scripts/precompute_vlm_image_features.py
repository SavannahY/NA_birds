#!/usr/bin/env python3
"""Precompute frozen SigLIP/SigLIP2 image features for NABirds manifests.

The output is one torch checkpoint per split plus a JSON run manifest. Runtime
dependencies are imported lazily so this file remains syntax-checkable without
the VLM stack installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


DEFAULT_MANIFEST_DIR = "reports/milestone2"
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_FALLBACK_MODEL = "google/siglip-base-patch16-224"
DEFAULT_SPLITS = ("train", "val", "test")


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
        raise SystemExit(f"Missing required CSV: {path}")

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


def _select_rows(rows: List[Dict[str, str]], row_offset: int, limit: Optional[int]) -> List[Dict[str, str]]:
    if row_offset < 0:
        raise SystemExit("--row-offset must be non-negative.")
    rows = rows[row_offset:]
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def _require_runtime_deps() -> Tuple[Any, Any, Any, Any]:
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
            "VLM feature precompute dependencies are missing: "
            f"{unique_missing}.\n"
            "Install a compatible PyTorch build plus transformers and Pillow before running.\n"
            "Example CPU install:\n"
            "  python3 -m pip install torch transformers pillow tqdm"
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


def _resolve_save_dtype(torch: Any, requested: str) -> Any:
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


def _feature_tensor(output: Any, feature_name: str) -> Any:
    """Return a 2D embedding tensor from tensor or HF ModelOutput variants."""
    if hasattr(output, "norm"):
        return output

    preferred_attributes = {
        "image": ("image_embeds", "pooler_output", "text_embeds"),
        "text": ("text_embeds", "pooler_output", "image_embeds"),
    }.get(feature_name, ("pooler_output", "image_embeds", "text_embeds"))
    for attribute in preferred_attributes:
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


def _manifest_for_split(manifest_dir: Path, split: str) -> Path:
    return manifest_dir / f"nabirds_{split}.csv"


def _resolve_image_path(row: Mapping[str, str], dataset_root: Path, path_root: Path) -> Path:
    image_path = Path(row["image_path"]).expanduser()
    candidates: List[Path] = []
    if image_path.is_absolute():
        candidates.append(image_path)
    else:
        candidates.extend([image_path, path_root / image_path])

    rel_path = row.get("rel_path", "").strip()
    if rel_path:
        candidates.append(dataset_root / "images" / rel_path)
        candidates.append(path_root / dataset_root / "images" / rel_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _crop_box(row: Mapping[str, str], image_size: Tuple[int, int], crop_padding: float) -> Tuple[int, int, int, int]:
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

    pad_x = int(round(w * crop_padding))
    pad_y = int(round(h * crop_padding))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_w, x + w + pad_x)
    bottom = min(image_h, y + h + pad_y)
    if right <= left or bottom <= top:
        return 0, 0, image_w, image_h
    return left, top, right, bottom


def _open_image(
    Image: Any,
    row: Mapping[str, str],
    dataset_root: Path,
    path_root: Path,
    input_mode: str,
    crop_padding: float,
) -> Any:
    path = _resolve_image_path(row, dataset_root=dataset_root, path_root=path_root)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if input_mode == "bbox":
            image = image.crop(_crop_box(row, image.size, crop_padding))
        return image.copy()


def _encode_image_batch(
    torch: Any,
    model: Any,
    processor: Any,
    images: Sequence[Any],
    device: Any,
    dtype: Optional[Any],
    normalize: bool,
) -> Any:
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = _move_inputs(inputs, device=device, dtype=dtype)
    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            output = model.get_image_features(**inputs)
        else:
            output = model(**inputs)
        features = _feature_tensor(output, "image")
        if normalize:
            features = _normalize(torch, features)
        return features.detach().float().cpu()


def _rows_to_metadata(rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Any]]:
    return {
        "image_id": [row.get("image_id", "") for row in rows],
        "rel_path": [row.get("rel_path", "") for row in rows],
        "image_path": [row.get("image_path", "") for row in rows],
        "target": [int(row["target"]) for row in rows],
        "raw_class_id": [row.get("raw_class_id", "") for row in rows],
        "class_name": [row.get("class_name", "") for row in rows],
        "split": [row.get("split", "") for row in rows],
    }


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _model_slug(model: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in model).strip("_")


def precompute_split(
    *,
    torch: Any,
    Image: Any,
    model: Any,
    processor: Any,
    manifest_path: Path,
    output_path: Path,
    dataset_root: Path,
    path_root: Path,
    split: str,
    model_name: str,
    input_mode: str,
    crop_padding: float,
    batch_size: int,
    device: Any,
    dtype: Optional[Any],
    save_dtype: Any,
    normalize: bool,
    limit: Optional[int],
    row_offset: int,
    show_progress: bool,
) -> Dict[str, Any]:
    rows = _select_rows(_read_csv(manifest_path, required_columns=("image_path", "target")), row_offset, limit)
    if not rows:
        raise SystemExit(f"{manifest_path}: no rows to encode.")

    features: List[Any] = []
    batches = list(_batched(rows, batch_size))
    iterator = _progress(batches, total=len(batches), label=f"encoding {split}", enabled=show_progress)
    started = time.time()
    for batch_rows in iterator:
        images = [
            _open_image(
                Image,
                row,
                dataset_root=dataset_root,
                path_root=path_root,
                input_mode=input_mode,
                crop_padding=crop_padding,
            )
            for row in batch_rows
        ]
        batch_features = _encode_image_batch(
            torch=torch,
            model=model,
            processor=processor,
            images=images,
            device=device,
            dtype=dtype,
            normalize=normalize,
        )
        features.append(batch_features.to(dtype=save_dtype))

    image_features = torch.cat(features, dim=0).contiguous()
    row_metadata = _rows_to_metadata(rows)
    targets = torch.tensor(row_metadata["target"], dtype=torch.long)
    payload = {
        "image_features": image_features,
        "targets": targets,
        "metadata": row_metadata,
        "config": {
            "model": model_name,
            "manifest": str(manifest_path),
            "split": split,
            "input_mode": input_mode,
            "crop_padding": crop_padding,
            "normalized": normalize,
            "feature_dim": int(image_features.shape[1]),
            "num_samples": int(image_features.shape[0]),
            "dtype": _dtype_name(image_features.dtype),
            "row_offset": row_offset,
            "limit": limit,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    return {
        "split": split,
        "manifest": str(manifest_path),
        "output": str(output_path),
        "num_samples": int(image_features.shape[0]),
        "feature_dim": int(image_features.shape[1]),
        "dtype": _dtype_name(image_features.dtype),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def precompute(args: argparse.Namespace) -> Dict[str, Any]:
    torch, Image, AutoModel, AutoProcessor = _require_runtime_deps()

    manifest_dir = Path(args.manifest_dir)
    dataset_root = Path(args.dataset_root)
    path_root = Path(args.path_root)
    output_dir = Path(args.out_dir)
    device = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype, device)
    save_dtype = _resolve_save_dtype(torch, args.save_dtype)

    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
    except OSError as exc:
        raise SystemExit(
            f"Could not load model or processor for {args.model!r}.\n"
            "If this is the first run, the checkpoint may need to be downloaded from Hugging Face.\n"
            f"Fallback SigLIP checkpoint configured for this project: {DEFAULT_FALLBACK_MODEL}"
        ) from exc

    model.eval()
    model.to(device=device)
    if dtype is not None:
        model.to(dtype=dtype)

    split_summaries: List[Dict[str, Any]] = []
    slug = _model_slug(args.model)
    started = time.time()
    for split in args.splits:
        manifest_path = _manifest_for_split(manifest_dir, split)
        suffix_parts = []
        if args.output_suffix:
            suffix_parts.append(_model_slug(args.output_suffix))
        if args.row_offset:
            suffix_parts.append(f"offset_{args.row_offset}")
        if args.limit is not None:
            suffix_parts.append(f"limit_{args.limit}")
        suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
        output_path = output_dir / f"{split}_{slug}_{args.input_mode}{suffix}_image_features.pt"
        summary = precompute_split(
            torch=torch,
            Image=Image,
            model=model,
            processor=processor,
            manifest_path=manifest_path,
            output_path=output_path,
            dataset_root=dataset_root,
            path_root=path_root,
            split=split,
            model_name=args.model,
            input_mode=args.input_mode,
            crop_padding=args.crop_padding,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
            save_dtype=save_dtype,
            normalize=not args.no_normalize,
            limit=args.limit,
            row_offset=args.row_offset,
            show_progress=not args.no_progress,
        )
        split_summaries.append(summary)

    manifest = {
        "model": args.model,
        "device": str(device),
        "dtype": _dtype_name(dtype) if dtype is not None else None,
        "save_dtype": args.save_dtype,
        "manifest_dir": str(manifest_dir),
        "dataset_root": str(dataset_root),
        "path_root": str(path_root),
        "out_dir": str(output_dir),
        "input_mode": args.input_mode,
        "crop_padding": args.crop_padding,
        "normalized": not args.no_normalize,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "row_offset": args.row_offset,
        "output_suffix": args.output_suffix,
        "splits": split_summaries,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_suffix = f"_{_model_slug(args.output_suffix)}" if args.output_suffix else ""
    if args.row_offset:
        manifest_suffix += f"_offset_{args.row_offset}"
    if args.limit is not None:
        manifest_suffix += f"_limit_{args.limit}"
    manifest_path = output_dir / f"{slug}_{args.input_mode}{manifest_suffix}_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["feature_manifest"] = str(manifest_path)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen SigLIP/SigLIP2 image features for NABirds train/val/test manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR, help="Directory containing nabirds_{split}.csv.")
    parser.add_argument("--dataset-root", default="nabirds", help="Dataset root used to resolve rel_path fallback.")
    parser.add_argument("--path-root", default=".", help="Root for resolving relative image_path values.")
    parser.add_argument("--out-dir", required=True, help="Directory for feature .pt files and run manifest JSON.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face SigLIP/SigLIP2 model id or local path.")
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS), help="Splits to encode.")
    parser.add_argument("--input-mode", choices=("full", "bbox"), default="full", help="Use full image or clipped bbox crop.")
    parser.add_argument("--crop-padding", type=_nonnegative_float, default=0.0, help="Fractional padding around bbox crops.")
    parser.add_argument("--batch-size", type=_positive_int, default=64, help="Image batch size.")
    parser.add_argument("--limit", type=_positive_int, default=None, help="Encode only the first N rows per split for smoke tests.")
    parser.add_argument("--row-offset", type=int, default=0, help="Skip this many rows before applying --limit.")
    parser.add_argument("--output-suffix", default="", help="Optional suffix for shard output filenames.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto", help="Model/input floating dtype.")
    parser.add_argument("--save-dtype", choices=("float32", "float16", "bfloat16"), default="float32", help="Feature tensor dtype on disk.")
    parser.add_argument("--no-normalize", action="store_true", help="Save raw image features instead of L2-normalized features.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars when tqdm is installed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        manifest = precompute(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"VLM image feature precompute failed: {exc}") from exc

    print("NABirds frozen VLM image feature precompute")
    for key in (
        "model",
        "device",
        "dtype",
        "save_dtype",
        "input_mode",
        "normalized",
        "batch_size",
        "limit",
        "elapsed_seconds",
        "feature_manifest",
    ):
        print(f"  {key}: {manifest.get(key)}")
    for split in manifest["splits"]:
        print(
            "  split {split}: {num_samples} x {feature_dim} -> {output}".format(
                split=split["split"],
                num_samples=split["num_samples"],
                feature_dim=split["feature_dim"],
                output=split["output"],
            )
        )


if __name__ == "__main__":
    main(sys.argv[1:])
