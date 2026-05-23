#!/usr/bin/env python3
"""Prompt a Qwen/InternVL-style VLM to rerank SigLIP top-k candidates.

Typical dry run from SigLIP predictions:

  python3 scripts/rerank_with_prompted_vlm.py \
    --siglip-predictions reports/milestone2/vlm_predictions.csv \
    --dry-run --limit 5 \
    --output-jsonl reports/milestone2/prompted_rerank_dryrun.jsonl

The script intentionally imports torch/transformers/Pillow only when model
inference is requested, so it remains syntax-checkable in a lightweight local
environment. Use --dry-run to validate candidate parsing and prompt shape before
running a 7B checkpoint on a GPU machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


DEFAULT_CLASS_PROMPTS = "reports/milestone2/nabirds_class_prompts.csv"
DEFAULT_MANIFEST = "reports/milestone2/nabirds_test.csv"
DEFAULT_OUTPUT_JSONL = "reports/milestone2/prompted_rerank.jsonl"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_INTERNVL_MODEL = "OpenGVLab/InternVL3-8B-Instruct"

AUTO_CANDIDATE_FIELDS = (
    "top5_targets",
    "topk_targets",
    "top_k_targets",
    "candidate_targets",
    "candidates",
    "topk",
    "top_k",
)
AUTO_SCORE_FIELDS = ("top5_scores", "topk_scores", "top_k_scores", "candidate_scores", "scores")
JOIN_KEY_CANDIDATES = ("image_id", "image_path", "rel_path")


@dataclass
class Candidate:
    rank: int
    target: Optional[int]
    class_name: str
    score: Optional[float] = None
    raw_value: Any = None

    def as_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "rank": self.rank,
            "target": self.target,
            "class_name": self.class_name,
        }
        if self.score is not None:
            payload["score"] = self.score
        return payload


@dataclass
class RerankExample:
    row_index: int
    row: Mapping[str, Any]
    image_path: str
    image_id: str
    rel_path: str
    target: Optional[int]
    class_name: str
    candidates: List[Candidate]


@dataclass
class ParsedChoice:
    selected_rank: Optional[int]
    selected_target: Optional[int]
    selected_class_name: str
    parse_status: str


@dataclass
class ModelBundle:
    torch: Any
    Image: Any
    transformers: Any
    model: Any
    processor: Any
    tokenizer: Any
    device: Any
    dtype: Any
    backend: str


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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} is empty or missing a header row.")
        return list(reader)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"{path}:{line_number}: expected a JSON object per line.")
            rows.append(payload)
    return rows


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("rows", "records", "predictions", "examples"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        else:
            raise SystemExit(f"{path}: JSON object must contain one of rows/records/predictions/examples.")
    else:
        raise SystemExit(f"{path}: expected a JSON list or object.")

    bad = [index for index, row in enumerate(rows) if not isinstance(row, dict)]
    if bad:
        raise SystemExit(f"{path}: expected all JSON rows to be objects; first bad row index is {bad[0]}.")
    return rows


def _read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [dict(row) for row in _read_csv(path)]
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    raise SystemExit(f"Unsupported input file extension for {path}; expected .csv, .jsonl, or .json.")


def _load_class_metadata(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    metadata: Dict[int, Dict[str, str]] = {}
    for row in rows:
        raw_target = row.get("target", "")
        try:
            target = int(str(raw_target).strip())
        except ValueError:
            continue
        metadata.setdefault(
            target,
            {
                "class_name": row.get("class_name", "").strip(),
                "class_family": row.get("class_family", "").strip(),
                "raw_class_id": row.get("raw_class_id", "").strip(),
            },
        )
    return metadata


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    return None


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Optional[Any]:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def _candidate_field_name(row: Mapping[str, Any], requested: str) -> Optional[str]:
    if requested != "auto":
        return requested if requested in row and row[requested] not in ("", None) else None
    for name in AUTO_CANDIDATE_FIELDS:
        if name in row and row[name] not in ("", None):
            return name
    return None


def _score_field_name(row: Mapping[str, Any], requested: str) -> Optional[str]:
    if requested == "none":
        return None
    if requested != "auto":
        return requested if requested in row and row[requested] not in ("", None) else None
    for name in AUTO_SCORE_FIELDS:
        if name in row and row[name] not in ("", None):
            return name
    return None


def _parse_list_value(value: Any, separator: str) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        for key in ("candidates", "targets", "topk", "top_k", "values"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]

    text = str(value).strip()
    if not text:
        return []

    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return _parse_list_value(parsed, separator=separator)

    if separator and separator in text:
        return [part.strip() for part in text.split(separator) if part.strip()]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [part.strip() for part in text.split() if part.strip()]


def _candidate_from_value(
    value: Any,
    rank: int,
    class_metadata: Mapping[int, Mapping[str, str]],
    score: Optional[float],
) -> Candidate:
    raw_target: Any = None
    raw_name = ""
    raw_score = score

    if isinstance(value, Mapping):
        raw_target = _first_present(value, ("target", "class_id", "label", "id", "pred_target"))
        raw_name_value = _first_present(value, ("class_name", "name", "label_name", "candidate_name"))
        raw_name = "" if raw_name_value is None else str(raw_name_value).strip()
        parsed_score = _parse_optional_float(_first_present(value, ("score", "prob", "similarity")))
        raw_score = parsed_score if parsed_score is not None else score
    else:
        raw_target = value

    target = _parse_optional_int(raw_target)
    if target is not None and target in class_metadata:
        class_name = class_metadata[target].get("class_name", "") or str(target)
    elif raw_name:
        class_name = raw_name
    else:
        class_name = "" if raw_target is None else str(raw_target).strip()

    if not class_name:
        class_name = f"candidate_{rank}"

    return Candidate(rank=rank, target=target, class_name=class_name, score=raw_score, raw_value=value)


def _parse_candidates_from_row(
    row: Mapping[str, Any],
    class_metadata: Mapping[int, Mapping[str, str]],
    candidate_field: str,
    score_field: str,
    separator: str,
) -> List[Candidate]:
    field_name = _candidate_field_name(row, candidate_field)
    if field_name is None:
        return []

    raw_values = _parse_list_value(row[field_name], separator=separator)
    score_values: List[Any] = []
    score_name = _score_field_name(row, score_field)
    if score_name:
        score_values = _parse_list_value(row[score_name], separator=separator)

    candidates: List[Candidate] = []
    for index, raw_value in enumerate(raw_values, start=1):
        score = None
        if index - 1 < len(score_values):
            score = _parse_optional_float(score_values[index - 1])
        candidates.append(
            _candidate_from_value(
                raw_value,
                rank=index,
                class_metadata=class_metadata,
                score=score,
            )
        )
    return candidates


def _candidate_from_row(
    row: Mapping[str, Any],
    rank: int,
    class_metadata: Mapping[int, Mapping[str, str]],
) -> Candidate:
    raw_target = _first_present(row, ("candidate_target", "candidate_class_id", "pred_target", "target"))
    raw_name = _first_present(row, ("candidate_name", "pred_class_name", "class_name", "name"))
    score = _parse_optional_float(_first_present(row, ("candidate_score", "score", "similarity", "prob")))
    value = {"target": raw_target, "class_name": raw_name, "score": score}
    return _candidate_from_value(value, rank=rank, class_metadata=class_metadata, score=score)


def _resolve_join_key(manifest_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]], requested: str) -> str:
    if requested != "auto":
        return requested
    if not manifest_rows:
        raise SystemExit("Manifest is empty.")
    if not candidate_rows:
        raise SystemExit("Candidate file is empty.")
    manifest_fields = set(manifest_rows[0])
    candidate_fields = set(candidate_rows[0])
    for key in JOIN_KEY_CANDIDATES:
        if key in manifest_fields and key in candidate_fields:
            return key
    raise SystemExit(
        "Could not auto-detect a join key shared by manifest and candidates. "
        "Pass --join-key explicitly; common choices are image_id, image_path, or rel_path."
    )


def _candidate_map_from_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    join_key: str,
    class_metadata: Mapping[int, Mapping[str, str]],
    candidate_field: str,
    score_field: str,
    separator: str,
) -> Dict[str, List[Candidate]]:
    by_key: Dict[str, List[Candidate]] = {}
    for row in candidate_rows:
        if join_key not in row or row[join_key] in ("", None):
            continue
        key = str(row[join_key])
        parsed = _parse_candidates_from_row(
            row,
            class_metadata=class_metadata,
            candidate_field=candidate_field,
            score_field=score_field,
            separator=separator,
        )
        if parsed:
            by_key[key] = parsed
            continue

        current = by_key.setdefault(key, [])
        current.append(_candidate_from_row(row, rank=len(current) + 1, class_metadata=class_metadata))
    return by_key


def _resolve_image_path(row: Mapping[str, Any], dataset_root: Path) -> str:
    image_path = str(row.get("image_path", "") or "").strip()
    if image_path:
        return image_path
    rel_path = str(row.get("rel_path", "") or "").strip()
    if rel_path:
        return str(dataset_root / "images" / rel_path)
    return ""


def _make_example(
    row: Mapping[str, Any],
    row_index: int,
    candidates: List[Candidate],
    class_metadata: Mapping[int, Mapping[str, str]],
    dataset_root: Path,
    top_k: Optional[int],
) -> RerankExample:
    if top_k is not None:
        candidates = candidates[:top_k]
    if not candidates:
        raise ValueError("no candidates found")

    target = _parse_optional_int(row.get("target"))
    class_name = str(row.get("class_name", "") or "").strip()
    if not class_name and target is not None and target in class_metadata:
        class_name = class_metadata[target].get("class_name", "")

    return RerankExample(
        row_index=row_index,
        row=row,
        image_path=_resolve_image_path(row, dataset_root=dataset_root),
        image_id=str(row.get("image_id", "") or ""),
        rel_path=str(row.get("rel_path", "") or ""),
        target=target,
        class_name=class_name,
        candidates=candidates,
    )


def _examples_from_siglip_predictions(args: argparse.Namespace, class_metadata: Mapping[int, Mapping[str, str]]) -> List[RerankExample]:
    rows = _read_records(Path(args.siglip_predictions))
    if args.limit is not None:
        rows = rows[: args.limit]

    examples: List[RerankExample] = []
    skipped = 0
    for index, row in enumerate(rows):
        candidates = _parse_candidates_from_row(
            row,
            class_metadata=class_metadata,
            candidate_field=args.candidate_field,
            score_field=args.score_field,
            separator=args.candidate_sep,
        )
        try:
            examples.append(
                _make_example(
                    row,
                    row_index=index,
                    candidates=candidates,
                    class_metadata=class_metadata,
                    dataset_root=Path(args.dataset_root),
                    top_k=args.top_k,
                )
            )
        except ValueError:
            skipped += 1

    if not examples:
        raise SystemExit(
            f"No rerank examples could be built from {args.siglip_predictions}. "
            "Check --candidate-field and candidate serialization."
        )
    if skipped:
        print(f"warning: skipped {skipped} prediction row(s) without candidates", file=sys.stderr)
    return examples


def _examples_from_manifest_and_candidates(
    args: argparse.Namespace,
    class_metadata: Mapping[int, Mapping[str, str]],
) -> List[RerankExample]:
    manifest_rows = _read_records(Path(args.manifest))
    candidate_rows = _read_records(Path(args.candidates))
    join_key = _resolve_join_key(manifest_rows, candidate_rows, requested=args.join_key)
    candidate_map = _candidate_map_from_rows(
        candidate_rows,
        join_key=join_key,
        class_metadata=class_metadata,
        candidate_field=args.candidate_field,
        score_field=args.score_field,
        separator=args.candidate_sep,
    )

    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    examples: List[RerankExample] = []
    missing = 0
    for index, row in enumerate(manifest_rows):
        key = str(row.get(join_key, "") or "")
        candidates = candidate_map.get(key, [])
        if not candidates:
            missing += 1
            continue
        examples.append(
            _make_example(
                row,
                row_index=index,
                candidates=candidates,
                class_metadata=class_metadata,
                dataset_root=Path(args.dataset_root),
                top_k=args.top_k,
            )
        )

    if not examples:
        raise SystemExit(
            f"No rerank examples could be built from manifest {args.manifest} and candidates {args.candidates}. "
            f"Resolved join key was {join_key!r}."
        )
    if missing:
        print(f"warning: {missing} manifest row(s) had no candidates for join key {join_key!r}", file=sys.stderr)
    return examples


def load_examples(args: argparse.Namespace) -> List[RerankExample]:
    class_metadata = _load_class_metadata(Path(args.class_prompts))
    if args.siglip_predictions:
        return _examples_from_siglip_predictions(args, class_metadata=class_metadata)
    if args.manifest and args.candidates:
        return _examples_from_manifest_and_candidates(args, class_metadata=class_metadata)
    raise SystemExit(
        "Provide either --siglip-predictions, or both --manifest and --candidates. "
        "Use --dry-run to prepare prompts without model inference."
    )


def build_prompt(example: RerankExample, prompt_version: str) -> str:
    candidate_lines = []
    for candidate in example.candidates:
        target_suffix = f" [target={candidate.target}]" if candidate.target is not None else ""
        score_suffix = f" (SigLIP score={candidate.score:.6g})" if candidate.score is not None else ""
        candidate_lines.append(f"{candidate.rank}. {candidate.class_name}{target_suffix}{score_suffix}")

    image_hint_parts = []
    if example.image_id:
        image_hint_parts.append(f"image_id={example.image_id}")
    if example.rel_path:
        image_hint_parts.append(f"rel_path={example.rel_path}")
    image_hint = ", ".join(image_hint_parts) if image_hint_parts else "unlabeled image"

    if prompt_version == "compact":
        return (
            "Choose the bird label that best matches the image from the numbered candidates. "
            "Return JSON only: {\"choice\": <number>, \"reason\": \"short visual reason\"}.\n"
            f"Image: {image_hint}\n"
            "Candidates:\n"
            + "\n".join(candidate_lines)
        )

    return (
        "You are an expert North American bird identifier. Inspect the image and choose the single "
        "best match from the candidate labels produced by SigLIP retrieval. Use visible field marks "
        "such as bill shape, head pattern, wing bars, body shape, plumage, and relative size. "
        "Do not introduce labels outside the candidate list.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\"choice\": <candidate_number>, \"class_name\": \"<chosen label>\", \"reason\": \"<brief visual evidence>\"}\n\n"
        f"Image metadata: {image_hint}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
    )


def _require_runtime_deps() -> Tuple[Any, Any, Any]:
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
        import transformers
    except ImportError:
        transformers = None
        missing.append("transformers")

    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise SystemExit(
            f"Prompted VLM reranking dependencies are missing: {unique_missing}.\n"
            "Install a compatible PyTorch build plus transformers and Pillow before running inference.\n"
            "Example CPU install for smoke tests:\n"
            "  python3 -m pip install torch transformers pillow\n"
            "For Qwen/InternVL 7B inference, install the GPU-specific torch wheel and any checkpoint "
            "extras required by the selected model. Use --dry-run to write prompts without these packages."
        )

    return torch, Image, transformers


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


def _resolve_dtype(torch: Any, requested: str, device: Any) -> Any:
    if requested == "auto":
        if device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[requested]


def _move_to_device(value: Any, device: Any, dtype: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device=device, dtype=dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_to_device(item, device=device, dtype=dtype) for item in value]
        return type(value)(moved)
    if hasattr(value, "to"):
        try:
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        except TypeError:
            return value.to(device)
    return value


def _infer_backend(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = model_name.lower()
    if "internvl" in lowered:
        return "internvl"
    if "qwen" in lowered:
        return "qwen"
    return "generic"


def _from_pretrained_kwargs(args: argparse.Namespace, dtype: Any, trust_remote_code: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def _load_generic_model(transformers: Any, model_name: str, kwargs: Mapping[str, Any]) -> Any:
    class_names = (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    )
    errors: List[str] = []
    for class_name in class_names:
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append(f"{class_name}: {exc.__class__.__name__}: {exc}")
    joined = "\n".join(f"  - {error}" for error in errors[-4:])
    raise SystemExit(f"Could not load model {model_name!r} with transformers auto classes.\n{joined}")


def load_model_bundle(args: argparse.Namespace) -> ModelBundle:
    torch, Image, transformers = _require_runtime_deps()
    device = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype, device)
    backend = _infer_backend(args.model, args.backend)
    trust_remote_code = args.trust_remote_code or backend == "internvl"
    kwargs = _from_pretrained_kwargs(args, dtype=dtype, trust_remote_code=trust_remote_code)

    processor = None
    tokenizer = None
    if backend == "internvl":
        AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
        AutoModel = getattr(transformers, "AutoModel", None)
        if AutoTokenizer is None or AutoModel is None:
            raise SystemExit("The installed transformers package is missing AutoTokenizer or AutoModel.")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code, use_fast=False)
        model = AutoModel.from_pretrained(args.model, **kwargs)
    else:
        AutoProcessor = getattr(transformers, "AutoProcessor", None)
        if AutoProcessor is None:
            raise SystemExit("The installed transformers package is missing AutoProcessor.")
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=trust_remote_code)
        model = _load_generic_model(transformers, args.model, kwargs=kwargs)

    model.eval()
    if not args.device_map:
        model.to(device=device)
        if dtype is not None:
            model.to(dtype=dtype)

    return ModelBundle(
        torch=torch,
        Image=Image,
        transformers=transformers,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        backend=backend,
    )


def _open_image(bundle: ModelBundle, image_path: str) -> Any:
    if not image_path:
        raise FileNotFoundError("input row does not contain image_path or rel_path")
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with bundle.Image.open(path) as handle:
        return handle.convert("RGB")


def _generate_with_processor(bundle: ModelBundle, image: Any, prompt: str, args: argparse.Namespace) -> str:
    processor = bundle.processor
    if processor is None:
        raise RuntimeError("Processor-backed generation requested, but no processor was loaded.")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    try:
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    except TypeError:
        inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = _move_to_device(inputs, device=bundle.device, dtype=bundle.dtype)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with bundle.torch.inference_mode():
        output_ids = bundle.model.generate(**inputs, **generation_kwargs)

    input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
    if input_ids is not None and output_ids.shape[-1] > input_ids.shape[-1]:
        response_ids = output_ids[:, input_ids.shape[-1] :]
    else:
        response_ids = output_ids

    if hasattr(processor, "batch_decode"):
        text_outputs = processor.batch_decode(response_ids, skip_special_tokens=True)
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
        text_outputs = processor.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    else:
        raise RuntimeError("Processor cannot decode generated token ids.")
    response = text_outputs[0].strip()
    if response:
        return response

    full_outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
    return full_outputs[0].strip()


def _internvl_pixel_values(bundle: ModelBundle, image: Any, image_size: int) -> Any:
    image = image.resize((image_size, image_size), resample=bundle.Image.BICUBIC)
    tensor = bundle.torch.tensor(list(image.getdata()), dtype=bundle.torch.float32)
    tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1) / 255.0
    mean = bundle.torch.tensor([0.485, 0.456, 0.406], dtype=bundle.torch.float32).view(3, 1, 1)
    std = bundle.torch.tensor([0.229, 0.224, 0.225], dtype=bundle.torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device=bundle.device, dtype=bundle.dtype)


def _generate_with_internvl(bundle: ModelBundle, image: Any, prompt: str, args: argparse.Namespace) -> str:
    if bundle.tokenizer is None:
        raise RuntimeError("InternVL generation requested, but no tokenizer was loaded.")
    if not hasattr(bundle.model, "chat"):
        raise RuntimeError(
            "Loaded InternVL backend model does not expose .chat(). "
            "Try --backend generic with a processor-compatible checkpoint."
        )

    pixel_values = _internvl_pixel_values(bundle, image=image, image_size=args.internvl_image_size)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_config["temperature"] = args.temperature
        generation_config["top_p"] = args.top_p

    question = "<image>\n" + prompt
    with bundle.torch.inference_mode():
        try:
            response = bundle.model.chat(bundle.tokenizer, pixel_values, question, generation_config)
        except TypeError:
            response = bundle.model.chat(
                bundle.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=False,
            )
    if isinstance(response, tuple):
        response = response[0]
    return str(response).strip()


def generate_response(bundle: ModelBundle, example: RerankExample, prompt: str, args: argparse.Namespace) -> str:
    image = _open_image(bundle, example.image_path)
    if bundle.backend == "internvl":
        return _generate_with_internvl(bundle, image=image, prompt=prompt, args=args)
    return _generate_with_processor(bundle, image=image, prompt=prompt, args=args)


def _extract_json_object(text: str) -> Optional[Mapping[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def parse_choice(response: str, candidates: Sequence[Candidate]) -> ParsedChoice:
    payload = _extract_json_object(response)
    rank: Optional[int] = None
    class_name = ""
    status = "unparsed"

    if payload:
        rank = _parse_optional_int(
            _first_present(payload, ("choice", "candidate", "candidate_number", "rank", "index"))
        )
        raw_name = _first_present(payload, ("class_name", "label", "answer", "name"))
        class_name = "" if raw_name is None else str(raw_name).strip()
        if rank is not None:
            status = "json_rank"
        elif class_name:
            status = "json_name"

    if rank is None:
        match = re.search(r"(?:choice|candidate|answer|option|rank)\D{0,20}([1-9]\d*)", response, flags=re.I)
        if match:
            rank = int(match.group(1))
            status = "regex_rank"

    if rank is None:
        match = re.search(r"\b([1-9]\d*)\b", response)
        if match:
            rank = int(match.group(1))
            status = "first_number"

    if rank is not None and 1 <= rank <= len(candidates):
        candidate = candidates[rank - 1]
        return ParsedChoice(
            selected_rank=rank,
            selected_target=candidate.target,
            selected_class_name=candidate.class_name,
            parse_status=status,
        )

    lowered = response.lower()
    for candidate in candidates:
        if candidate.class_name and candidate.class_name.lower() in lowered:
            return ParsedChoice(
                selected_rank=candidate.rank,
                selected_target=candidate.target,
                selected_class_name=candidate.class_name,
                parse_status="name_match",
            )

    if class_name:
        for candidate in candidates:
            if candidate.class_name.lower() == class_name.lower():
                return ParsedChoice(
                    selected_rank=candidate.rank,
                    selected_target=candidate.target,
                    selected_class_name=candidate.class_name,
                    parse_status="json_name_exact",
                )

    return ParsedChoice(
        selected_rank=None,
        selected_target=None,
        selected_class_name=class_name,
        parse_status="unparsed",
    )


def _base_output_record(example: RerankExample, prompt: str, args: argparse.Namespace) -> Dict[str, Any]:
    top_candidate = example.candidates[0] if example.candidates else None
    top1_before = None
    if top_candidate is not None and example.target is not None and top_candidate.target is not None:
        top1_before = int(top_candidate.target == example.target)

    record: Dict[str, Any] = {
        "row_index": example.row_index,
        "image_id": example.image_id,
        "rel_path": example.rel_path,
        "image_path": example.image_path,
        "target": example.target,
        "class_name": example.class_name,
        "candidates": [candidate.as_json() for candidate in example.candidates],
        "siglip_top1_target": top_candidate.target if top_candidate else None,
        "siglip_top1_class_name": top_candidate.class_name if top_candidate else "",
        "siglip_top1_correct": top1_before,
        "prompt_version": args.prompt_version,
        "model": args.model,
        "backend": _infer_backend(args.model, args.backend),
    }
    if not args.no_prompt_in_output:
        record["prompt"] = prompt
    return record


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_output_records(args: argparse.Namespace, examples: Sequence[RerankExample]) -> Iterator[Dict[str, Any]]:
    bundle = None if args.dry_run else load_model_bundle(args)
    for example in examples:
        prompt = build_prompt(example, prompt_version=args.prompt_version)
        record = _base_output_record(example, prompt=prompt, args=args)

        if args.dry_run:
            record["status"] = "prompt_prepared"
            yield record
            continue

        started = time.time()
        try:
            assert bundle is not None
            response = generate_response(bundle, example=example, prompt=prompt, args=args)
            choice = parse_choice(response, example.candidates)
            record.update(
                {
                    "status": "ok",
                    "raw_response": response,
                    "selected_rank": choice.selected_rank,
                    "selected_target": choice.selected_target,
                    "selected_class_name": choice.selected_class_name,
                    "parse_status": choice.parse_status,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            if example.target is not None and choice.selected_target is not None:
                record["rerank_correct"] = int(choice.selected_target == example.target)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            record.update(
                {
                    "status": "error",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
        yield record


def summarize_output(path: Path) -> Dict[str, Any]:
    total = 0
    prepared = 0
    ok = 0
    errors = 0
    parsed = 0
    before_correct = 0
    before_total = 0
    rerank_correct = 0
    rerank_total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            status = record.get("status")
            prepared += int(status == "prompt_prepared")
            ok += int(status == "ok")
            errors += int(status == "error")
            parsed += int(record.get("selected_rank") is not None)
            if record.get("siglip_top1_correct") is not None:
                before_total += 1
                before_correct += int(record["siglip_top1_correct"])
            if record.get("rerank_correct") is not None:
                rerank_total += 1
                rerank_correct += int(record["rerank_correct"])
    summary: Dict[str, Any] = {
        "output_jsonl": str(path),
        "records": total,
        "prompt_prepared": prepared,
        "ok": ok,
        "errors": errors,
        "parsed_choices": parsed,
    }
    if before_total:
        summary["siglip_top1_accuracy_on_scored_rows"] = before_correct / before_total
    if rerank_total:
        summary["rerank_accuracy_on_scored_rows"] = rerank_correct / rerank_total
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank SigLIP top-k NABirds candidates with a prompted Qwen/InternVL VLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_argument_group("inputs")
    input_group.add_argument("--siglip-predictions", default=None, help="SigLIP prediction/top-k file (.csv, .jsonl, .json).")
    input_group.add_argument("--manifest", default=None, help="Manifest CSV/JSONL/JSON used with --candidates.")
    input_group.add_argument("--candidates", default=None, help="Candidate CSV/JSONL/JSON used with --manifest.")
    input_group.add_argument("--class-prompts", default=DEFAULT_CLASS_PROMPTS, help="Class prompt CSV used to map target ids to names.")
    input_group.add_argument("--dataset-root", default="nabirds", help="Dataset root for resolving rel_path when image_path is absent.")
    input_group.add_argument("--candidate-field", default="auto", help="Field containing a candidate list, or 'auto'.")
    input_group.add_argument("--score-field", default="auto", help="Field containing candidate scores, 'auto', or 'none'.")
    input_group.add_argument("--candidate-sep", default="|", help="Delimiter for serialized candidate lists.")
    input_group.add_argument("--join-key", default="auto", help="Manifest/candidate join key, or 'auto'.")
    input_group.add_argument("--top-k", type=_positive_int, default=None, help="Optional cap on candidates sent to the VLM.")
    input_group.add_argument("--limit", type=_positive_int, default=None, help="Only process the first N rows for smoke tests.")

    prompt_group = parser.add_argument_group("prompting")
    prompt_group.add_argument("--prompt-version", choices=("field_marks_json", "compact"), default="field_marks_json")
    prompt_group.add_argument("--dry-run", action="store_true", help="Write prepared prompts and candidates without loading a VLM.")
    prompt_group.add_argument("--no-prompt-in-output", action="store_true", help="Omit prompt text from JSONL records.")
    prompt_group.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Path for per-image JSONL output.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default=DEFAULT_QWEN_MODEL, help="Hugging Face model id or local path.")
    model_group.add_argument("--backend", choices=("auto", "qwen", "internvl", "generic"), default="auto")
    model_group.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    model_group.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    model_group.add_argument("--device-map", default=None, help="Optional transformers device_map, e.g. auto.")
    model_group.add_argument("--low-cpu-mem-usage", action="store_true")
    model_group.add_argument("--attn-implementation", default=None, help="Optional attention implementation, e.g. flash_attention_2.")
    model_group.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers loaders.")
    model_group.add_argument("--max-new-tokens", type=_positive_int, default=128)
    model_group.add_argument("--temperature", type=_nonnegative_float, default=0.0)
    model_group.add_argument("--top-p", type=_nonnegative_float, default=0.9)
    model_group.add_argument("--internvl-image-size", type=_positive_int, default=448)
    model_group.add_argument("--continue-on-error", action="store_true", help="Write error records instead of aborting on a row failure.")

    args = parser.parse_args(argv)
    if args.siglip_predictions and (args.manifest or args.candidates):
        parser.error("--siglip-predictions cannot be combined with --manifest/--candidates.")
    if not args.siglip_predictions and bool(args.manifest) != bool(args.candidates):
        parser.error("Pass both --manifest and --candidates, or neither.")
    if not args.siglip_predictions and not (args.manifest and args.candidates):
        parser.error("Provide --siglip-predictions, or provide both --manifest and --candidates.")
    if args.backend == "internvl" and not args.trust_remote_code:
        print("warning: InternVL backend will enable trust_remote_code for model loading", file=sys.stderr)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    examples = load_examples(args)
    output_path = Path(args.output_jsonl)
    count = _write_jsonl(output_path, iter_output_records(args, examples))
    summary = summarize_output(output_path)
    print("NABirds prompted VLM rerank")
    print(f"  wrote_records: {count}")
    for key, value in summary.items():
        if isinstance(value, float):
            value = f"{value:.6f}"
        print(f"  {key}: {value}")
    if args.dry_run:
        print("  note: dry run only; prompts were prepared but no model was loaded")


if __name__ == "__main__":
    main(sys.argv[1:])
