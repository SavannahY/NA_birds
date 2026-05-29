#!/usr/bin/env python3
"""Precompute frozen SigLIP/CLIP text features for NABirds prompt CSVs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_vlm_zero_shot import (  # noqa: E402
    DEFAULT_MODEL,
    EXPECTED_NABIRDS_CLASSES,
    _encode_text_features,
    _load_prompt_rows,
    _require_runtime_deps,
    _resolve_device,
    _resolve_dtype,
)


def _dtype_name(dtype: Any) -> Optional[str]:
    return str(dtype).replace("torch.", "") if dtype is not None else None


def _model_slug(model: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in model).strip("_")


def precompute(args: argparse.Namespace) -> dict[str, Any]:
    torch, _Image, AutoModel, AutoProcessor = _require_runtime_deps()

    prompt_path = Path(args.prompts)
    out_dir = Path(args.out_dir)
    device = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype, device)

    class_targets, class_names, prompts_by_target, prompt_rows = _load_prompt_rows(
        prompt_path,
        expected_num_classes=args.expected_num_classes,
    )

    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
    except OSError as exc:
        raise SystemExit(
            f"Could not load model or processor for {args.model!r}. "
            "If this is the first run, the checkpoint may need to be downloaded from Hugging Face."
        ) from exc

    model.eval()
    model.to(device=device)
    if dtype is not None:
        model.to(dtype=dtype)

    started = time.time()
    class_text_features = _encode_text_features(
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
    ).detach().float().cpu()

    slug = _model_slug(args.model)
    prompt_slug = _model_slug(prompt_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{slug}_{prompt_slug}_{args.prompt_aggregation}_text_features.pt"
    payload = {
        "class_text_features": class_text_features,
        "class_targets": class_targets,
        "class_names": class_names,
        "config": {
            "model": args.model,
            "prompts": str(prompt_path),
            "prompt_aggregation": args.prompt_aggregation,
            "num_classes": len(class_targets),
            "num_prompt_rows": prompt_rows,
            "feature_dim": int(class_text_features.shape[1]),
            "dtype": str(class_text_features.dtype).replace("torch.", ""),
        },
    }
    torch.save(payload, output_path)

    manifest = {
        "model": args.model,
        "device": str(device),
        "dtype": _dtype_name(dtype),
        "prompts": str(prompt_path),
        "prompt_aggregation": args.prompt_aggregation,
        "num_classes": len(class_targets),
        "num_prompt_rows": prompt_rows,
        "feature_dim": int(class_text_features.shape[1]),
        "output": str(output_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    manifest_path = out_dir / f"{slug}_{prompt_slug}_{args.prompt_aggregation}_text_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["feature_manifest"] = str(manifest_path)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen VLM text features for a NABirds prompt CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts", required=True, help="Prompt CSV accepted by eval_vlm_zero_shot.py.")
    parser.add_argument("--out-dir", required=True, help="Output directory for text feature checkpoint and manifest.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face CLIP/SigLIP/SigLIP2 model id.")
    parser.add_argument("--prompt-aggregation", choices=("mean", "first"), default="mean")
    parser.add_argument("--expected-num-classes", type=int, default=EXPECTED_NABIRDS_CLASSES, help="Use 0 to disable.")
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    manifest = precompute(parse_args(argv))
    print("NABirds frozen VLM text feature precompute")
    for key, value in manifest.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
