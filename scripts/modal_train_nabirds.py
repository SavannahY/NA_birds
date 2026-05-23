#!/usr/bin/env python3
"""Modal scaffold for running NABirds training and evaluation jobs.

Dataset assumptions:
  - A Modal Volume is mounted at /data.
  - Either /data/nabirds already contains the unpacked NABirds dataset, or
    /data/nabirds.tar.gz exists and can be extracted to /data/nabirds.

This file intentionally remains importable and syntax-checkable without Modal
installed. The cloud functions are active when executed by `modal run`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional, Sequence


try:
    import modal
except ImportError:  # Keep local py_compile/import checks working.
    modal = None  # type: ignore[assignment]


APP_NAME = "nabirds-training"
VOLUME_NAME = os.environ.get("NABIRDS_MODAL_VOLUME", "nabirds-data")
DEFAULT_GPU = os.environ.get("NABIRDS_MODAL_GPU", "A10G")

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = Path("/workspace/NA_birds")
DATA_MOUNT = Path("/data")
DEFAULT_DATASET_ROOT = DATA_MOUNT / "nabirds"
DEFAULT_ARCHIVE = DATA_MOUNT / "nabirds.tar.gz"
DEFAULT_WORK_ROOT = DATA_MOUNT / "nabirds_runs"
DEFAULT_MANIFEST_DIR = DEFAULT_WORK_ROOT / "reports" / "milestone2"
DEFAULT_RUN_ROOT = DEFAULT_WORK_ROOT / "runs"
DEFAULT_FEATURE_ROOT = DEFAULT_WORK_ROOT / "features"
HF_CACHE_DIR = DATA_MOUNT / "hf_cache"
TORCH_HOME = DATA_MOUNT / "torch_cache"

IMAGE_PACKAGES = (
    "torch>=2.3,<3",
    "torchvision>=0.18,<1",
    "transformers>=4.42",
    "pillow>=10",
    "scikit-learn>=1.4",
    "tqdm>=4.66",
    "accelerate>=0.30",
    "safetensors>=0.4",
)

NABIRDS_REQUIRED_FILES = (
    "images.txt",
    "classes.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
    "bounding_boxes.txt",
    "sizes.txt",
    "images",
)

MANIFEST_FILES = (
    "nabirds_train.csv",
    "nabirds_val.csv",
    "nabirds_test.csv",
    "nabirds_class_prompts.csv",
    "nabirds_hard_negative_groups.csv",
)


class _LocalApp:
    """No-op subset of modal.App used when Modal is not installed locally."""

    @staticmethod
    def _decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        setattr(function, "remote", function)
        return function

    def function(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        del args, kwargs
        return self._decorate

    def local_entrypoint(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        del args, kwargs
        return self._decorate


if modal is None:
    _DATA_VOLUME = None
    _IMAGE = None
    app = _LocalApp()
else:
    _DATA_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    _IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install(*IMAGE_PACKAGES)
        .add_local_dir(str(LOCAL_PROJECT_ROOT / "scripts"), remote_path=str(REMOTE_PROJECT_ROOT / "scripts"), copy=True)
        .add_local_dir(str(LOCAL_PROJECT_ROOT / "tools"), remote_path=str(REMOTE_PROJECT_ROOT / "tools"), copy=True)
        .add_local_dir(str(LOCAL_PROJECT_ROOT / "configs"), remote_path=str(REMOTE_PROJECT_ROOT / "configs"), copy=True)
        .env(
            {
                "HF_HOME": str(HF_CACHE_DIR),
                "TORCH_HOME": str(TORCH_HOME),
                "PYTHONUNBUFFERED": "1",
            }
        )
    )
    app = modal.App(APP_NAME, image=_IMAGE)


def _modal_options(*, gpu: Optional[str], timeout: int, cpu: float = 4.0, memory: int = 32768) -> dict[str, Any]:
    if modal is None:
        return {}
    options: dict[str, Any] = {
        "volumes": {str(DATA_MOUNT): _DATA_VOLUME},
        "timeout": timeout,
        "cpu": cpu,
        "memory": memory,
    }
    if gpu:
        options["gpu"] = gpu
    return options


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(HF_CACHE_DIR))
    env.setdefault("TORCH_HOME", str(TORCH_HOME))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_python(args: Sequence[str], *, cwd: Path = REMOTE_PROJECT_ROOT) -> None:
    command = [sys.executable, *args]
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), env=_runtime_env(), check=True)


def _has_nabirds_dataset(dataset_root: Path) -> bool:
    return all((dataset_root / name).exists() for name in NABIRDS_REQUIRED_FILES)


def _validate_tar_member(member: tarfile.TarInfo, destination: Path, expected_top_level: str) -> None:
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts:
        raise SystemExit(f"Refusing unsafe archive member path: {member.name}")
    if not name.parts or name.parts[0] != expected_top_level:
        raise SystemExit(
            f"Refusing archive member outside expected top-level {expected_top_level}/ directory: {member.name}"
        )
    if member.issym() or member.islnk():
        raise SystemExit(f"Refusing link member in archive: {member.name}")

    target = (destination / Path(*name.parts)).resolve()
    destination_resolved = destination.resolve()
    if target != destination_resolved and destination_resolved not in target.parents:
        raise SystemExit(f"Refusing archive member outside destination: {member.name}")


def _extract_archive(archive_path: Path, destination: Path, expected_top_level: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} into {destination}", flush=True)
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_tar_member(member, destination, expected_top_level)
        archive.extractall(destination, members=members)


def _ensure_dataset(dataset_root: str, archive_path: str) -> Path:
    dataset = Path(dataset_root)
    archive = Path(archive_path)

    if _has_nabirds_dataset(dataset):
        return dataset

    if dataset.exists():
        missing = [name for name in NABIRDS_REQUIRED_FILES if not (dataset / name).exists()]
        raise SystemExit(
            f"{dataset} exists but does not look like NABirds; missing: {', '.join(missing)}. "
            "Fix the mounted volume or pass --dataset-root to a complete dataset."
        )

    if archive.is_file():
        _extract_archive(archive, dataset.parent, dataset.name)
        if _has_nabirds_dataset(dataset):
            _commit_volume()
            return dataset
        raise SystemExit(
            f"Extracted {archive}, but {dataset} still does not contain the expected NABirds files. "
            "Confirm the archive expands to a top-level nabirds/ directory."
        )

    raise SystemExit(
        "NABirds data is not available in Modal. Expected either "
        f"{dataset} or archive {archive}. Upload an unpacked dataset directory or nabirds.tar.gz "
        f"to the Modal Volume named {VOLUME_NAME!r}."
    )


def _ensure_output_dirs() -> None:
    for path in (DEFAULT_WORK_ROOT, DEFAULT_RUN_ROOT, HF_CACHE_DIR, TORCH_HOME):
        path.mkdir(parents=True, exist_ok=True)


def _manifest_files_present(manifest_dir: Path) -> bool:
    return all((manifest_dir / name).is_file() for name in MANIFEST_FILES)


def _build_manifests_in_place(
    *,
    dataset_root: Path,
    manifest_dir: Path,
    val_fraction: float,
    seed: int,
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _run_python(
        [
            "scripts/build_nabirds_manifests.py",
            "--dataset-root",
            str(dataset_root),
            "--out-dir",
            str(manifest_dir),
            "--val-fraction",
            str(val_fraction),
            "--seed",
            str(seed),
        ]
    )


def _ensure_manifests(
    *,
    dataset_root: Path,
    manifest_dir: Path,
    val_fraction: float,
    seed: int,
    rebuild: bool,
) -> None:
    if rebuild or not _manifest_files_present(manifest_dir):
        _build_manifests_in_place(
            dataset_root=dataset_root,
            manifest_dir=manifest_dir,
            val_fraction=val_fraction,
            seed=seed,
        )


def _commit_volume() -> None:
    if _DATA_VOLUME is not None:
        _DATA_VOLUME.commit()


def _reload_volume() -> None:
    if _DATA_VOLUME is not None and hasattr(_DATA_VOLUME, "reload"):
        _DATA_VOLUME.reload()


def _none_if_zero(value: int) -> Optional[int]:
    return None if value <= 0 else value


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _append_optional_int(args: list[str], flag: str, value: Optional[int]) -> None:
    if value is not None and value > 0:
        args.extend([flag, str(value)])


def _split_extra_args(extra_args: Optional[Sequence[str]] | str) -> list[str]:
    if isinstance(extra_args, str):
        return shlex.split(extra_args) if extra_args else []
    return list(extra_args or [])


@app.function(**_modal_options(gpu=None, timeout=2 * 60 * 60, cpu=2.0, memory=8192))
def build_manifests(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    val_fraction: float = 0.15,
    seed: int = 231,
) -> dict[str, str]:
    """Build NABirds train/val/test manifests on the mounted Modal Volume."""

    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    output_dir = Path(manifest_dir)
    _build_manifests_in_place(dataset_root=dataset, manifest_dir=output_dir, val_fraction=val_fraction, seed=seed)
    _commit_volume()
    return {"dataset_root": str(dataset), "manifest_dir": str(output_dir)}


def _eval_vlm(
    *,
    mode: str,
    dataset_root: str,
    archive_path: str,
    manifest_dir: str,
    out_dir: str,
    model: str,
    input_mode: str,
    batch_size: int,
    text_batch_size: int,
    limit: Optional[int],
    dtype: str,
    prompt_aggregation: str,
    rebuild_manifests: bool,
    no_progress: bool,
    extra_args: Optional[Sequence[str]],
) -> dict[str, str]:
    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=231,
        rebuild=rebuild_manifests,
    )

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"vlm_{_slug(model)}_{input_mode}_{mode}"
    metrics_path = output_dir / f"{output_stem}.json"
    predictions_path = output_dir / f"{output_stem}_predictions.csv"

    args = [
        "scripts/eval_vlm_zero_shot.py",
        "--manifest",
        str(manifests / "nabirds_test.csv"),
        "--prompts",
        str(manifests / "nabirds_class_prompts.csv"),
        "--dataset-root",
        str(dataset),
        "--model",
        model,
        "--input-mode",
        input_mode,
        "--prompt-aggregation",
        prompt_aggregation,
        "--batch-size",
        str(batch_size),
        "--text-batch-size",
        str(text_batch_size),
        "--device",
        "cuda",
        "--dtype",
        dtype,
        "--output-json",
        str(metrics_path),
        "--output-predictions",
        str(predictions_path),
    ]
    _append_optional_int(args, "--limit", limit)
    if no_progress:
        args.append("--no-progress")
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"metrics": str(metrics_path), "predictions": str(predictions_path), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=12 * 60 * 60, cpu=4.0, memory=32768))
def precompute_vlm_image_features(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_FEATURE_ROOT / "vlm_image_features"),
    model: str = "google/siglip2-base-patch16-224",
    input_mode: str = "full",
    batch_size: int = 64,
    limit: Optional[int] = None,
    dtype: str = "auto",
    save_dtype: str = "float32",
    crop_padding: float = 0.0,
    rebuild_manifests: bool = False,
    no_progress: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Precompute frozen SigLIP/SigLIP2 image features on a Modal GPU."""

    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=231,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "scripts/precompute_vlm_image_features.py",
        "--manifest-dir",
        str(manifests),
        "--dataset-root",
        str(dataset),
        "--path-root",
        str(dataset.parent),
        "--out-dir",
        str(output_dir),
        "--model",
        model,
        "--input-mode",
        input_mode,
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        "--dtype",
        dtype,
        "--save-dtype",
        save_dtype,
        "--crop-padding",
        str(crop_padding),
    ]
    _append_optional_int(args, "--limit", limit)
    if no_progress:
        args.append("--no-progress")
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"feature_root": str(output_dir), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=4 * 60 * 60, cpu=4.0, memory=32768))
def eval_vlm_smoke(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "vlm_zero_shot"),
    model: str = "google/siglip2-base-patch16-224",
    input_mode: str = "full",
    batch_size: int = 32,
    text_batch_size: int = 256,
    limit: int = 64,
    dtype: str = "auto",
    prompt_aggregation: str = "mean",
    rebuild_manifests: bool = False,
    no_progress: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Run a limited GPU zero-shot VLM evaluation for Modal smoke testing."""

    return _eval_vlm(
        mode="smoke",
        dataset_root=dataset_root,
        archive_path=archive_path,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        model=model,
        input_mode=input_mode,
        batch_size=batch_size,
        text_batch_size=text_batch_size,
        limit=_none_if_zero(limit),
        dtype=dtype,
        prompt_aggregation=prompt_aggregation,
        rebuild_manifests=rebuild_manifests,
        no_progress=no_progress,
        extra_args=extra_args,
    )


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=10 * 60 * 60, cpu=4.0, memory=32768))
def eval_vlm_full(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "vlm_zero_shot"),
    model: str = "google/siglip2-base-patch16-224",
    input_mode: str = "full",
    batch_size: int = 32,
    text_batch_size: int = 256,
    dtype: str = "auto",
    prompt_aggregation: str = "mean",
    rebuild_manifests: bool = False,
    no_progress: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Run full NABirds GPU zero-shot VLM evaluation."""

    return _eval_vlm(
        mode="full",
        dataset_root=dataset_root,
        archive_path=archive_path,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        model=model,
        input_mode=input_mode,
        batch_size=batch_size,
        text_batch_size=text_batch_size,
        limit=None,
        dtype=dtype,
        prompt_aggregation=prompt_aggregation,
        rebuild_manifests=rebuild_manifests,
        no_progress=no_progress,
        extra_args=extra_args,
    )


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=4 * 60 * 60, cpu=4.0, memory=32768))
def eval_visual_checkpoint(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "visual_eval"),
    checkpoint: str = "",
    batch_size: int = 128,
    num_workers: int = 4,
    limit: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Evaluate a trained visual checkpoint on the official NABirds test split."""

    if not checkpoint:
        raise SystemExit("Pass --checkpoint /path/to/best.pt for visual checkpoint evaluation.")
    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=231,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint)
    output_stem = f"visual_{_slug(checkpoint_path.parent.name)}_test"
    metrics_path = output_dir / f"{output_stem}.json"
    predictions_path = output_dir / f"{output_stem}_predictions.csv"

    args = [
        "scripts/eval_visual_checkpoint.py",
        "--checkpoint",
        str(checkpoint_path),
        "--manifest",
        str(manifests / "nabirds_test.csv"),
        "--class-prompts",
        str(manifests / "nabirds_class_prompts.csv"),
        "--output-json",
        str(metrics_path),
        "--output-predictions",
        str(predictions_path),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        "cuda",
    ]
    _append_optional_int(args, "--limit", limit)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"metrics": str(metrics_path), "predictions": str(predictions_path), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=4 * 60 * 60, cpu=4.0, memory=32768))
def eval_fused_checkpoint(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "fused_eval"),
    checkpoint: str = "",
    batch_size: int = 128,
    num_workers: int = 4,
    limit: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Evaluate a fused full-image + bbox-crop checkpoint on the official test split."""

    if not checkpoint:
        raise SystemExit("Pass --checkpoint /path/to/best.pt for fused checkpoint evaluation.")
    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=231,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint)
    output_stem = f"fused_{_slug(checkpoint_path.parent.name)}_test"
    metrics_path = output_dir / f"{output_stem}.json"
    predictions_path = output_dir / f"{output_stem}_predictions.csv"

    args = [
        "scripts/eval_fused_checkpoint.py",
        "--checkpoint",
        str(checkpoint_path),
        "--manifest",
        str(manifests / "nabirds_test.csv"),
        "--class-prompts",
        str(manifests / "nabirds_class_prompts.csv"),
        "--path-root",
        str(dataset.parent),
        "--output-json",
        str(metrics_path),
        "--output-predictions",
        str(predictions_path),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        "cuda",
    ]
    _append_optional_int(args, "--limit", limit)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"metrics": str(metrics_path), "predictions": str(predictions_path), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=4 * 60 * 60, cpu=4.0, memory=32768))
def eval_part_hierarchy_checkpoint(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "part_hierarchy_eval"),
    checkpoint: str = "",
    batch_size: int = 64,
    num_workers: int = 4,
    limit: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: str = "",
) -> dict[str, str]:
    """Evaluate a part-guided hierarchy-fused checkpoint on the official test split."""

    if not checkpoint:
        raise SystemExit("Pass --checkpoint /path/to/best.pt for part-hierarchy checkpoint evaluation.")
    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=231,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint)
    output_stem = f"part_hierarchy_{_slug(checkpoint_path.parent.name)}_test"
    metrics_path = output_dir / f"{output_stem}.json"
    predictions_path = output_dir / f"{output_stem}_predictions.csv"

    args = [
        "scripts/eval_part_hierarchy_fused.py",
        "--checkpoint",
        str(checkpoint_path),
        "--manifest",
        str(manifests / "nabirds_test.csv"),
        "--class-prompts",
        str(manifests / "nabirds_class_prompts.csv"),
        "--dataset-root",
        str(dataset),
        "--path-root",
        str(dataset.parent),
        "--output-json",
        str(metrics_path),
        "--output-predictions",
        str(predictions_path),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        "cuda",
    ]
    _append_optional_int(args, "--limit", limit)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"metrics": str(metrics_path), "predictions": str(predictions_path), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=24 * 60 * 60, cpu=8.0, memory=32768))
def train_visual_baseline(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "visual_baseline"),
    model: str = "resnet50",
    input_mode: str = "full",
    epochs: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    seed: int = 231,
    amp: bool = True,
    no_pretrained: bool = False,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Run scripts/train_visual_baseline.py on a Modal GPU."""

    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=seed,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "scripts/train_visual_baseline.py",
        "--manifest-dir",
        str(manifests),
        "--out-dir",
        str(output_dir),
        "--model",
        model,
        "--input-mode",
        input_mode,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--image-size",
        str(image_size),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if amp:
        args.append("--amp")
    if no_pretrained:
        args.append("--no-pretrained")
    _append_optional_int(args, "--limit-train", limit_train)
    _append_optional_int(args, "--limit-val", limit_val)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"run_root": str(output_dir), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=24 * 60 * 60, cpu=8.0, memory=32768))
def train_fused_full_crop(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "fused_full_crop"),
    model: str = "resnet50",
    branch_mode: str = "shared",
    epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    seed: int = 231,
    amp: bool = True,
    macro_f1: bool = True,
    no_pretrained: bool = False,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Run scripts/train_fused_full_crop.py on a Modal GPU."""

    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=seed,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "scripts/train_fused_full_crop.py",
        "--manifest-dir",
        str(manifests),
        "--path-root",
        str(dataset.parent),
        "--out-dir",
        str(output_dir),
        "--model",
        model,
        "--branch-mode",
        branch_mode,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--image-size",
        str(image_size),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if amp:
        args.append("--amp")
    if macro_f1:
        args.append("--macro-f1")
    if no_pretrained:
        args.append("--no-pretrained")
    _append_optional_int(args, "--limit-train", limit_train)
    _append_optional_int(args, "--limit-val", limit_val)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"run_root": str(output_dir), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=24 * 60 * 60, cpu=8.0, memory=32768))
def train_part_hierarchy_fused(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "part_hierarchy_fused"),
    model: str = "resnet50",
    epochs: int = 10,
    batch_size: int = 16,
    num_workers: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    seed: int = 231,
    amp: bool = True,
    macro_f1: bool = True,
    no_pretrained: bool = False,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: str = "",
) -> dict[str, str]:
    """Run scripts/train_part_hierarchy_fused.py on a Modal GPU."""

    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=seed,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "scripts/train_part_hierarchy_fused.py",
        "--manifest-dir",
        str(manifests),
        "--dataset-root",
        str(dataset),
        "--path-root",
        str(dataset.parent),
        "--out-dir",
        str(output_dir),
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--image-size",
        str(image_size),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if amp:
        args.append("--amp")
    if macro_f1:
        args.append("--macro-f1")
    if no_pretrained:
        args.append("--no-pretrained")
    _append_optional_int(args, "--limit-train", limit_train)
    _append_optional_int(args, "--limit-val", limit_val)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"run_root": str(output_dir), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=24 * 60 * 60, cpu=8.0, memory=32768))
def train_vlm_adapter(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_dir: str = str(DEFAULT_RUN_ROOT / "vlm_adapter"),
    model: str = "google/siglip2-base-patch16-224",
    input_mode: str = "full",
    epochs: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    adapter_lr: float = 1e-4,
    contrastive_weight: float = 0.05,
    hard_negative_weight: float = 0.05,
    seed: int = 231,
    amp: bool = True,
    macro_f1: bool = True,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Train lightweight SigLIP/SigLIP2 projection adapters on a Modal GPU."""

    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=seed,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "scripts/train_vlm_adapter.py",
        "--manifest-dir",
        str(manifests),
        "--prompts",
        str(manifests / "nabirds_class_prompts.csv"),
        "--hard-negative-csv",
        str(manifests / "nabirds_hard_negative_groups.csv"),
        "--dataset-root",
        str(dataset),
        "--out-dir",
        str(output_dir),
        "--model",
        model,
        "--input-mode",
        input_mode,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--adapter-lr",
        str(adapter_lr),
        "--contrastive-weight",
        str(contrastive_weight),
        "--hard-negative-weight",
        str(hard_negative_weight),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if amp:
        args.append("--amp")
    if macro_f1:
        args.append("--macro-f1")
    _append_optional_int(args, "--limit-train", limit_train)
    _append_optional_int(args, "--limit-val", limit_val)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"run_root": str(output_dir), "manifest_dir": str(manifests)}


@app.function(**_modal_options(gpu=DEFAULT_GPU, timeout=8 * 60 * 60, cpu=4.0, memory=32768))
def train_vlm_adapter_cached(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    feature_manifest: str = "",
    out_dir: str = str(DEFAULT_RUN_ROOT / "vlm_adapter_cached"),
    model: str = "google/siglip2-base-patch16-224",
    epochs: int = 5,
    batch_size: int = 256,
    adapter_lr: float = 3e-4,
    hard_negative_weight: float = 0.05,
    seed: int = 231,
    macro_f1: bool = True,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    rebuild_manifests: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Train lightweight VLM adapters from precomputed image features."""

    _reload_volume()
    _ensure_output_dirs()
    dataset = _ensure_dataset(dataset_root, archive_path)
    manifests = Path(manifest_dir)
    _ensure_manifests(
        dataset_root=dataset,
        manifest_dir=manifests,
        val_fraction=0.15,
        seed=seed,
        rebuild=rebuild_manifests,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "scripts/train_vlm_adapter_cached.py",
        "--prompts",
        str(manifests / "nabirds_class_prompts.csv"),
        "--hard-negative-csv",
        str(manifests / "nabirds_hard_negative_groups.csv"),
        "--out-dir",
        str(output_dir),
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--adapter-lr",
        str(adapter_lr),
        "--hard-negative-weight",
        str(hard_negative_weight),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if feature_manifest:
        args.extend(["--feature-manifest", feature_manifest])
    if macro_f1:
        args.append("--macro-f1")
    _append_optional_int(args, "--limit-train", limit_train)
    _append_optional_int(args, "--limit-val", limit_val)
    args.extend(_split_extra_args(extra_args))
    _run_python(args)
    _commit_volume()
    return {"run_root": str(output_dir), "manifest_dir": str(manifests), "feature_manifest": feature_manifest}


def _help_text() -> str:
    return f"""Usage:
  modal run scripts/modal_train_nabirds.py --task build-manifests
  modal run scripts/modal_train_nabirds.py --task vlm-smoke
  modal run scripts/modal_train_nabirds.py --task vlm-full
  modal run scripts/modal_train_nabirds.py --task eval-visual --checkpoint /data/nabirds_runs/runs/visual_baseline/<run>/best.pt
  modal run scripts/modal_train_nabirds.py --task eval-fused --checkpoint /data/nabirds_runs/runs/fused_full_crop/<run>/best.pt
  modal run scripts/modal_train_nabirds.py --task eval-part-hierarchy --checkpoint /data/nabirds_runs/runs/part_hierarchy_fused/<run>/best.pt
  modal run scripts/modal_train_nabirds.py --task precompute-vlm-features
  modal run scripts/modal_train_nabirds.py --task train-visual --input-mode full
  modal run scripts/modal_train_nabirds.py --task train-fused --branch-mode shared
  modal run scripts/modal_train_nabirds.py --task train-part-hierarchy --extra-args "--num-views 5 --fusion concat --hierarchy-loss-weight 0.1"
  modal run scripts/modal_train_nabirds.py --task train-vlm-adapter
  modal run scripts/modal_train_nabirds.py --task train-vlm-adapter-cached --extra-args "--feature-manifest /data/nabirds_runs/runs/vlm_image_features/google_siglip2_base_patch16_224_full_feature_manifest.json"

Defaults:
  Modal Volume: {VOLUME_NAME}
  Mounted data path: {DATA_MOUNT}
  Dataset root: {DEFAULT_DATASET_ROOT}
  Archive fallback: {DEFAULT_ARCHIVE}
  GPU functions: {DEFAULT_GPU}
"""


def _invoke(function, background: bool, **kwargs):
    if background:
        call = function.spawn(**kwargs)
        return {"status": "spawned", "function_call_id": call.object_id}
    return function.remote(**kwargs)


@app.local_entrypoint()
def main(
    task: str = "help",
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    archive_path: str = str(DEFAULT_ARCHIVE),
    manifest_dir: str = str(DEFAULT_MANIFEST_DIR),
    out_root: str = str(DEFAULT_RUN_ROOT),
    model: str = "resnet50",
    vlm_model: str = "google/siglip2-base-patch16-224",
    input_mode: str = "full",
    branch_mode: str = "shared",
    epochs: int = 10,
    batch_size: int = 0,
    num_workers: int = 4,
    limit: int = 0,
    limit_train: int = 0,
    limit_val: int = 0,
    amp: bool = True,
    no_pretrained: bool = False,
    rebuild_manifests: bool = False,
    background: bool = False,
    checkpoint: str = "",
    extra_args: str = "",
) -> None:
    """Small dispatcher for common Modal runs."""

    extra = shlex.split(extra_args) if extra_args else []
    task_normalized = task.strip().lower().replace("_", "-")

    if task_normalized in {"help", "commands"}:
        print(_help_text())
        return

    if task_normalized == "build-manifests":
        print(
            _invoke(
                build_manifests,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
            )
        )
        return

    if task_normalized == "vlm-smoke":
        print(
            _invoke(
                eval_vlm_smoke,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "vlm_zero_shot"),
                model=vlm_model,
                input_mode=input_mode,
                batch_size=batch_size or 32,
                limit=limit or 64,
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "vlm-full":
        print(
            _invoke(
                eval_vlm_full,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "vlm_zero_shot"),
                model=vlm_model,
                input_mode=input_mode,
                batch_size=batch_size or 32,
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "precompute-vlm-features":
        print(
            _invoke(
                precompute_vlm_image_features,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "vlm_image_features"),
                model=vlm_model,
                input_mode=input_mode,
                batch_size=batch_size or 64,
                limit=_none_if_zero(limit),
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "eval-visual":
        print(
            _invoke(
                eval_visual_checkpoint,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "visual_eval"),
                checkpoint=checkpoint,
                batch_size=batch_size or 128,
                num_workers=num_workers,
                limit=_none_if_zero(limit),
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "eval-fused":
        print(
            _invoke(
                eval_fused_checkpoint,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "fused_eval"),
                checkpoint=checkpoint,
                batch_size=batch_size or 128,
                num_workers=num_workers,
                limit=_none_if_zero(limit),
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized in {"eval-part-hierarchy", "eval-part", "eval-sota"}:
        print(
            _invoke(
                eval_part_hierarchy_checkpoint,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "part_hierarchy_eval"),
                checkpoint=checkpoint,
                batch_size=batch_size or 64,
                num_workers=num_workers,
                limit=_none_if_zero(limit),
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "train-visual":
        shared_limit = _none_if_zero(limit)
        print(
            _invoke(
                train_visual_baseline,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "visual_baseline"),
                model=model,
                input_mode=input_mode,
                epochs=epochs,
                batch_size=batch_size or 64,
                num_workers=num_workers,
                amp=amp,
                no_pretrained=no_pretrained,
                limit_train=_none_if_zero(limit_train) or shared_limit,
                limit_val=_none_if_zero(limit_val) or shared_limit,
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "train-fused":
        shared_limit = _none_if_zero(limit)
        print(
            _invoke(
                train_fused_full_crop,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "fused_full_crop"),
                model=model,
                branch_mode=branch_mode,
                epochs=epochs,
                batch_size=batch_size or 32,
                num_workers=num_workers,
                amp=amp,
                no_pretrained=no_pretrained,
                limit_train=_none_if_zero(limit_train) or shared_limit,
                limit_val=_none_if_zero(limit_val) or shared_limit,
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized in {"train-part-hierarchy", "train-part", "train-sota"}:
        shared_limit = _none_if_zero(limit)
        print(
            _invoke(
                train_part_hierarchy_fused,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "part_hierarchy_fused"),
                model=model,
                epochs=epochs,
                batch_size=batch_size or 16,
                num_workers=num_workers,
                amp=amp,
                no_pretrained=no_pretrained,
                limit_train=_none_if_zero(limit_train) or shared_limit,
                limit_val=_none_if_zero(limit_val) or shared_limit,
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "train-vlm-adapter":
        shared_limit = _none_if_zero(limit)
        print(
            _invoke(
                train_vlm_adapter,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "vlm_adapter"),
                model=vlm_model,
                input_mode=input_mode,
                epochs=epochs,
                batch_size=batch_size or 32,
                num_workers=num_workers,
                amp=amp,
                limit_train=_none_if_zero(limit_train) or shared_limit,
                limit_val=_none_if_zero(limit_val) or shared_limit,
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    if task_normalized == "train-vlm-adapter-cached":
        shared_limit = _none_if_zero(limit)
        print(
            _invoke(
                train_vlm_adapter_cached,
                background,
                dataset_root=dataset_root,
                archive_path=archive_path,
                manifest_dir=manifest_dir,
                out_dir=str(Path(out_root) / "vlm_adapter_cached"),
                model=vlm_model,
                epochs=epochs,
                batch_size=batch_size or 256,
                limit_train=_none_if_zero(limit_train) or shared_limit,
                limit_val=_none_if_zero(limit_val) or shared_limit,
                rebuild_manifests=rebuild_manifests,
                extra_args=extra,
            )
        )
        return

    raise SystemExit(f"Unknown task {task!r}.\n\n{_help_text()}")
