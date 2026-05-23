#!/usr/bin/env python3
"""Python 3 tooling for auditing and loading the NABirds dataset.

This file is intended for CS231n-style project work:

* parse the official NABirds metadata files with explicit types
* validate cross-file consistency and annotation ranges
* write machine-readable and slide/report-friendly dataset summaries
* optionally render diagnostic plots and annotated image samples
* provide an optional PyTorch-compatible Dataset wrapper

The module uses only the Python standard library for core parsing and
validation. Pillow/matplotlib/PyTorch are optional and used only when the
corresponding feature is requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TRAIN = "train"
TEST = "test"
ALL_SPLITS = {TRAIN, TEST}


@dataclass(frozen=True)
class BoundingBox:
    """A single NABirds bounding box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def area_ratio(self, image_width: int, image_height: int) -> float:
        image_area = image_width * image_height
        if image_area <= 0:
            return 0.0
        return self.area / image_area

    def in_bounds(self, image_width: int, image_height: int) -> bool:
        return (
            self.x >= 0
            and self.y >= 0
            and self.width > 0
            and self.height > 0
            and self.x + self.width <= image_width
            and self.y + self.height <= image_height
        )

    def clipped(self, image_width: int, image_height: int) -> "BoundingBox":
        """Return this box clipped to image bounds.

        NABirds has a small number of boxes that exceed the recorded image
        dimensions. Use this before crop-based training/evaluation.
        """
        x0 = min(max(self.x, 0), image_width)
        y0 = min(max(self.y, 0), image_height)
        x1 = min(max(self.x + self.width, 0), image_width)
        y1 = min(max(self.y + self.height, 0), image_height)
        return BoundingBox(x=x0, y=y0, width=max(0, x1 - x0), height=max(0, y1 - y0))

    def pil_crop_box(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """Return a Pillow-compatible clipped crop tuple: left, upper, right, lower."""
        clipped = self.clipped(image_width, image_height)
        return (
            clipped.x,
            clipped.y,
            clipped.x + clipped.width,
            clipped.y + clipped.height,
        )


@dataclass(frozen=True)
class PartLocation:
    """One bird part annotation."""

    part_id: int
    x: int
    y: int
    visible: bool

    def in_bounds(self, image_width: int, image_height: int) -> bool:
        if not self.visible:
            return True
        return 0 <= self.x <= image_width and 0 <= self.y <= image_height


@dataclass(frozen=True)
class ImageRecord:
    """Merged metadata for a single NABirds image."""

    image_id: str
    rel_path: str
    image_path: Path
    class_id: str
    class_index: int
    class_name: str
    split: str
    width: int
    height: int
    bbox: BoundingBox
    photographer: str
    parts: Tuple[PartLocation, ...]

    def to_metadata(self) -> Dict[str, object]:
        """Return a JSON-serializable metadata dict for training/debugging."""
        return {
            "image_id": self.image_id,
            "rel_path": self.rel_path,
            "class_id": self.class_id,
            "class_index": self.class_index,
            "class_name": self.class_name,
            "split": self.split,
            "width": self.width,
            "height": self.height,
            "bbox": asdict(self.bbox),
            "photographer": self.photographer,
            "parts": [asdict(part) for part in self.parts],
        }


@dataclass(frozen=True)
class AuditIssue:
    """A validation finding."""

    severity: str
    check: str
    message: str
    count: int = 0
    examples: Tuple[str, ...] = ()


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _parse_name_mapping(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in _read_nonempty_lines(path):
        key, value = line.split(maxsplit=1)
        mapping[key] = value
    return mapping


def _parse_two_column_mapping(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in _read_nonempty_lines(path):
        key, value = line.split(maxsplit=1)
        mapping[key] = value
    return mapping


def _parse_int_tuple_mapping(path: Path, expected_values: int) -> Dict[str, Tuple[int, ...]]:
    mapping: Dict[str, Tuple[int, ...]] = {}
    for line in _read_nonempty_lines(path):
        pieces = line.split()
        if len(pieces) != expected_values + 1:
            raise ValueError(f"{path}: expected {expected_values + 1} columns, got {len(pieces)}: {line}")
        mapping[pieces[0]] = tuple(int(value) for value in pieces[1:])
    return mapping


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[int(index)])
    weight = index - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _describe(values: Sequence[float], digits: int = 4) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "min": round(float(min(values)), digits),
        "p25": round(_percentile(values, 0.25), digits),
        "median": round(statistics.median(values), digits),
        "mean": round(statistics.mean(values), digits),
        "p75": round(_percentile(values, 0.75), digits),
        "max": round(float(max(values)), digits),
    }


def _take_examples(values: Iterable[str], limit: int = 10) -> Tuple[str, ...]:
    return tuple(list(values)[:limit])


def _prepare_plot_backend(cache_base: Path) -> None:
    """Use a writable cache and non-GUI backend for reproducible report generation."""
    cache_root = cache_base / ".cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


class NABirdsMetadata:
    """Typed loader and auditor for an extracted NABirds directory."""

    REQUIRED_FILES = (
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "bounding_boxes.txt",
        "sizes.txt",
        "classes.txt",
        "hierarchy.txt",
        "photographers.txt",
        "parts/parts.txt",
        "parts/part_locs.txt",
    )

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.images_dir = self.root / "images"
        self._ensure_required_files()

        self.image_paths = _parse_two_column_mapping(self.root / "images.txt")
        self.labels = _parse_two_column_mapping(self.root / "image_class_labels.txt")
        self.classes = _parse_name_mapping(self.root / "classes.txt")
        self.photographers = _parse_name_mapping(self.root / "photographers.txt")
        self.splits = self._load_splits(self.root / "train_test_split.txt")
        self.sizes = _parse_int_tuple_mapping(self.root / "sizes.txt", expected_values=2)
        self.bboxes = self._load_bboxes(self.root / "bounding_boxes.txt")
        self.part_names = _parse_name_mapping(self.root / "parts" / "parts.txt")
        self.parts = self._load_parts(self.root / "parts" / "part_locs.txt")
        self.hierarchy = self._load_hierarchy(self.root / "hierarchy.txt")

        used_class_ids = sorted(set(self.labels.values()), key=lambda value: int(value))
        self.class_to_index = {class_id: index for index, class_id in enumerate(used_class_ids)}
        self.index_to_class = {index: class_id for class_id, index in self.class_to_index.items()}
        self.image_ids = list(self.image_paths.keys())
        self.records_by_id = self._build_records()

    def _ensure_required_files(self) -> None:
        missing = [name for name in self.REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(f"{self.root} is missing required NABirds files: {missing_str}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"{self.root} is missing required image directory: images/")

    @staticmethod
    def _load_splits(path: Path) -> Dict[str, str]:
        splits: Dict[str, str] = {}
        for line in _read_nonempty_lines(path):
            image_id, raw_value = line.split()
            if raw_value not in {"0", "1"}:
                raise ValueError(f"{path}: split flag must be 0 or 1, got {raw_value!r}")
            splits[image_id] = TRAIN if raw_value == "1" else TEST
        return splits

    @staticmethod
    def _load_bboxes(path: Path) -> Dict[str, BoundingBox]:
        bboxes: Dict[str, BoundingBox] = {}
        for image_id, values in _parse_int_tuple_mapping(path, expected_values=4).items():
            bboxes[image_id] = BoundingBox(*values)
        return bboxes

    @staticmethod
    def _load_hierarchy(path: Path) -> Dict[str, str]:
        hierarchy: Dict[str, str] = {}
        for line in _read_nonempty_lines(path):
            child_id, parent_id = line.split()
            hierarchy[child_id] = parent_id
        return hierarchy

    @staticmethod
    def _load_parts(path: Path) -> Dict[str, Tuple[PartLocation, ...]]:
        grouped: Dict[str, Dict[int, PartLocation]] = defaultdict(dict)
        for line in _read_nonempty_lines(path):
            image_id, raw_part_id, raw_x, raw_y, raw_visible = line.split()
            part_id = int(raw_part_id)
            visible_value = int(raw_visible)
            if visible_value not in {0, 1}:
                raise ValueError(f"{path}: visible must be 0 or 1, got {raw_visible!r}")
            grouped[image_id][part_id] = PartLocation(
                part_id=part_id,
                x=int(raw_x),
                y=int(raw_y),
                visible=bool(visible_value),
            )
        return {
            image_id: tuple(parts[part_id] for part_id in sorted(parts))
            for image_id, parts in grouped.items()
        }

    def _build_records(self) -> Dict[str, ImageRecord]:
        records: Dict[str, ImageRecord] = {}
        for image_id, rel_path in self.image_paths.items():
            class_id = self.labels[image_id]
            width, height = self.sizes[image_id]
            records[image_id] = ImageRecord(
                image_id=image_id,
                rel_path=rel_path,
                image_path=self.images_dir / rel_path,
                class_id=class_id,
                class_index=self.class_to_index[class_id],
                class_name=self.classes[class_id],
                split=self.splits[image_id],
                width=width,
                height=height,
                bbox=self.bboxes[image_id],
                photographer=self.photographers[image_id],
                parts=self.parts[image_id],
            )
        return records

    def records(self, split: Optional[str] = None) -> List[ImageRecord]:
        """Return records, optionally filtered by split."""
        if split is None or split == "all":
            return [self.records_by_id[image_id] for image_id in self.image_ids]
        if split not in ALL_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALL_SPLITS)} or None, got {split!r}")
        return [record for record in self.records_by_id.values() if record.split == split]

    def class_counts(self) -> List[Dict[str, object]]:
        """Return per-class train/test/total counts for used visual categories."""
        train_counts = Counter(record.class_id for record in self.records(TRAIN))
        test_counts = Counter(record.class_id for record in self.records(TEST))
        total_counts = Counter(record.class_id for record in self.records())

        rows: List[Dict[str, object]] = []
        for class_id in sorted(total_counts, key=lambda value: int(value)):
            rows.append(
                {
                    "class_id": class_id,
                    "class_index": self.class_to_index[class_id],
                    "class_name": self.classes[class_id],
                    "total": total_counts[class_id],
                    "train": train_counts[class_id],
                    "test": test_counts[class_id],
                }
            )
        return rows

    def summary(self) -> Dict[str, object]:
        """Return a compact dataset summary suitable for JSON or slides."""
        records = self.records()
        split_counts = Counter(record.split for record in records)
        class_count_rows = self.class_counts()
        class_total_counts = [int(row["total"]) for row in class_count_rows]
        class_train_counts = [int(row["train"]) for row in class_count_rows]
        class_test_counts = [int(row["test"]) for row in class_count_rows]
        widths = [record.width for record in records]
        heights = [record.height for record in records]
        aspect_ratios = [record.width / record.height for record in records if record.height > 0]
        bbox_area_ratios = [
            record.bbox.area_ratio(record.width, record.height)
            for record in records
            if record.width > 0 and record.height > 0
        ]
        visible_parts_per_image = [sum(part.visible for part in record.parts) for record in records]

        part_visibility = {}
        for raw_part_id, part_name in sorted(self.part_names.items(), key=lambda item: int(item[0])):
            part_id = int(raw_part_id)
            visible_count = sum(record.parts[part_id].visible for record in records)
            part_visibility[part_name] = {
                "visible": visible_count,
                "total": len(records),
                "visible_fraction": round(visible_count / len(records), 4),
            }

        top_classes = sorted(class_count_rows, key=lambda row: int(row["total"]), reverse=True)[:10]
        bottom_classes = sorted(class_count_rows, key=lambda row: int(row["total"]))[:10]
        min_class_count = min(class_total_counts) if class_total_counts else 0
        max_class_count = max(class_total_counts) if class_total_counts else 0

        return {
            "dataset_root": str(self.root),
            "num_images": len(records),
            "num_image_files": sum(1 for path in self.images_dir.rglob("*") if path.is_file()),
            "num_image_directories": sum(1 for path in self.images_dir.iterdir() if path.is_dir()),
            "num_classes_in_taxonomy": len(self.classes),
            "num_used_visual_categories": len(class_count_rows),
            "num_part_types": len(self.part_names),
            "split_counts": dict(sorted(split_counts.items())),
            "class_count_stats": {
                "total": _describe(class_total_counts, digits=2),
                "train": _describe(class_train_counts, digits=2),
                "test": _describe(class_test_counts, digits=2),
                "max_to_min_ratio": round(max_class_count / min_class_count, 4) if min_class_count else None,
            },
            "image_width_stats": _describe(widths, digits=2),
            "image_height_stats": _describe(heights, digits=2),
            "aspect_ratio_stats": _describe(aspect_ratios, digits=4),
            "bbox_area_ratio_stats": _describe(bbox_area_ratios, digits=4),
            "visible_parts_per_image_stats": _describe(visible_parts_per_image, digits=2),
            "part_visibility": part_visibility,
            "largest_classes": top_classes,
            "smallest_classes": bottom_classes,
        }

    def audit(
        self,
        check_image_dimensions: bool = False,
        dimension_sample: Optional[int] = 512,
        seed: int = 0,
    ) -> List[AuditIssue]:
        """Validate consistency across metadata files and annotations."""
        issues: List[AuditIssue] = []
        image_id_set = set(self.image_paths)

        def compare_id_set(name: str, mapping: Mapping[str, object]) -> None:
            current_ids = set(mapping)
            missing = sorted(image_id_set - current_ids)
            extra = sorted(current_ids - image_id_set)
            if missing:
                issues.append(
                    AuditIssue(
                        severity="error",
                        check=f"{name}_ids",
                        message=f"{name} is missing image IDs listed in images.txt",
                        count=len(missing),
                        examples=_take_examples(missing),
                    )
                )
            if extra:
                issues.append(
                    AuditIssue(
                        severity="error",
                        check=f"{name}_ids",
                        message=f"{name} contains image IDs not listed in images.txt",
                        count=len(extra),
                        examples=_take_examples(extra),
                    )
                )

        compare_id_set("image_class_labels", self.labels)
        compare_id_set("train_test_split", self.splits)
        compare_id_set("sizes", self.sizes)
        compare_id_set("bounding_boxes", self.bboxes)
        compare_id_set("photographers", self.photographers)
        compare_id_set("part_locs", self.parts)

        missing_files = sorted(
            record.rel_path for record in self.records() if not record.image_path.is_file()
        )
        if missing_files:
            issues.append(
                AuditIssue(
                    severity="error",
                    check="image_files_exist",
                    message="Some image paths listed in images.txt do not exist on disk",
                    count=len(missing_files),
                    examples=_take_examples(missing_files),
                )
            )

        actual_files = {
            str(path.relative_to(self.images_dir))
            for path in self.images_dir.rglob("*")
            if path.is_file()
        }
        listed_files = set(self.image_paths.values())
        extra_files = sorted(actual_files - listed_files)
        if extra_files:
            issues.append(
                AuditIssue(
                    severity="warning",
                    check="extra_image_files",
                    message="Some files under images/ are not listed in images.txt",
                    count=len(extra_files),
                    examples=_take_examples(extra_files),
                )
            )

        label_ids = set(self.labels.values())
        unknown_label_ids = sorted(label_ids - set(self.classes), key=lambda value: int(value))
        if unknown_label_ids:
            issues.append(
                AuditIssue(
                    severity="error",
                    check="class_labels_defined",
                    message="Some image labels are not defined in classes.txt",
                    count=len(unknown_label_ids),
                    examples=_take_examples(unknown_label_ids),
                )
            )

        unknown_hierarchy_ids = sorted(
            (set(self.hierarchy) | set(self.hierarchy.values())) - set(self.classes),
            key=lambda value: int(value),
        )
        if unknown_hierarchy_ids:
            issues.append(
                AuditIssue(
                    severity="warning",
                    check="hierarchy_ids_defined",
                    message="Some hierarchy IDs are not defined in classes.txt",
                    count=len(unknown_hierarchy_ids),
                    examples=_take_examples(unknown_hierarchy_ids),
                )
            )

        bad_bbox = sorted(
            record.image_id
            for record in self.records()
            if not record.bbox.in_bounds(record.width, record.height)
        )
        if bad_bbox:
            issues.append(
                AuditIssue(
                    severity="warning",
                    check="bbox_bounds",
                    message="Some bounding boxes are empty or extend outside the recorded image size",
                    count=len(bad_bbox),
                    examples=_take_examples(bad_bbox),
                )
            )

        expected_part_ids = tuple(sorted(int(part_id) for part_id in self.part_names))
        bad_parts = []
        bad_part_bounds = []
        for record in self.records():
            part_ids = tuple(part.part_id for part in record.parts)
            if part_ids != expected_part_ids:
                bad_parts.append(record.image_id)
            for part in record.parts:
                if not part.in_bounds(record.width, record.height):
                    bad_part_bounds.append(record.image_id)
                    break
        if bad_parts:
            issues.append(
                AuditIssue(
                    severity="error",
                    check="part_count",
                    message="Some images do not have exactly the expected part IDs",
                    count=len(bad_parts),
                    examples=_take_examples(sorted(bad_parts)),
                )
            )
        if bad_part_bounds:
            issues.append(
                AuditIssue(
                    severity="warning",
                    check="part_bounds",
                    message="Some visible part locations fall outside the recorded image size",
                    count=len(bad_part_bounds),
                    examples=_take_examples(sorted(set(bad_part_bounds))),
                )
            )

        missing_split_classes = []
        for row in self.class_counts():
            if int(row["train"]) == 0 or int(row["test"]) == 0:
                missing_split_classes.append(f"{row['class_id']}:{row['class_name']}")
        if missing_split_classes:
            issues.append(
                AuditIssue(
                    severity="warning",
                    check="class_split_coverage",
                    message="Some used classes are missing from train or test split",
                    count=len(missing_split_classes),
                    examples=_take_examples(missing_split_classes),
                )
            )

        if check_image_dimensions:
            dimension_issues = self._audit_image_dimensions(dimension_sample, seed)
            issues.extend(dimension_issues)

        if not issues:
            issues.append(
                AuditIssue(
                    severity="info",
                    check="audit_status",
                    message="No blocking dataset consistency issues found",
                    count=0,
                    examples=(),
                )
            )
        return issues

    def _audit_image_dimensions(self, sample_size: Optional[int], seed: int) -> List[AuditIssue]:
        try:
            from PIL import Image
        except ImportError:
            return [
                AuditIssue(
                    severity="warning",
                    check="image_dimension_check",
                    message="Pillow is not installed, so actual image dimensions were not checked",
                )
            ]

        records = self.records()
        if sample_size is not None and sample_size > 0 and sample_size < len(records):
            rng = random.Random(seed)
            records_to_check = rng.sample(records, sample_size)
        else:
            records_to_check = records

        mismatches = []
        unreadable = []
        for record in records_to_check:
            try:
                with Image.open(record.image_path) as image:
                    actual_width, actual_height = image.size
            except Exception as exc:  # pragma: no cover - depends on image files
                unreadable.append(f"{record.rel_path}: {exc}")
                continue
            if (actual_width, actual_height) != (record.width, record.height):
                mismatches.append(
                    f"{record.rel_path}: metadata=({record.width},{record.height}) "
                    f"actual=({actual_width},{actual_height})"
                )

        issues: List[AuditIssue] = []
        if unreadable:
            issues.append(
                AuditIssue(
                    severity="error",
                    check="image_readable",
                    message="Some sampled image files could not be opened by Pillow",
                    count=len(unreadable),
                    examples=_take_examples(unreadable),
                )
            )
        if mismatches:
            issues.append(
                AuditIssue(
                    severity="warning",
                    check="image_dimensions_match",
                    message="Some sampled image files have dimensions that differ from sizes.txt",
                    count=len(mismatches),
                    examples=_take_examples(mismatches),
                )
            )
        issues.append(
            AuditIssue(
                severity="info",
                check="image_dimension_check",
                message=f"Checked actual dimensions for {len(records_to_check)} image files",
                count=len(records_to_check),
            )
        )
        return issues

    def write_reports(
        self,
        out_dir: Path | str,
        check_image_dimensions: bool = False,
        dimension_sample: Optional[int] = 512,
        seed: int = 0,
        plots: bool = False,
        sample_grid: bool = False,
        sample_count: int = 12,
        sample_split: str = "all",
    ) -> Dict[str, Path]:
        """Write JSON, CSV, Markdown, and optional visual diagnostics."""
        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        summary = self.summary()
        issues = self.audit(
            check_image_dimensions=check_image_dimensions,
            dimension_sample=dimension_sample,
            seed=seed,
        )
        report = {
            "summary": summary,
            "issues": [asdict(issue) for issue in issues],
        }

        json_path = out_path / "nabirds_audit.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        class_counts_path = out_path / "nabirds_class_counts.csv"
        with class_counts_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["class_id", "class_index", "class_name", "total", "train", "test"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.class_counts())

        markdown_path = out_path / "nabirds_summary.md"
        markdown_path.write_text(self._format_markdown_summary(summary, issues), encoding="utf-8")

        written = {
            "json": json_path,
            "class_counts_csv": class_counts_path,
            "markdown": markdown_path,
        }
        if plots:
            written.update(self.write_plots(out_path))
        if sample_grid:
            grid_path = out_path / "nabirds_sample_grid.png"
            self.save_sample_grid(
                out_path=grid_path,
                split=None if sample_split == "all" else sample_split,
                count=sample_count,
                seed=seed,
            )
            written["sample_grid"] = grid_path
        return written

    @staticmethod
    def _format_markdown_summary(summary: Mapping[str, object], issues: Sequence[AuditIssue]) -> str:
        split_counts = summary["split_counts"]
        class_stats = summary["class_count_stats"]
        lines = [
            "# NABirds Dataset Audit",
            "",
            "## Core Counts",
            "",
            f"- Dataset root: `{summary['dataset_root']}`",
            f"- Images listed: **{summary['num_images']:,}**",
            f"- Image files on disk: **{summary['num_image_files']:,}**",
            f"- Image directories: **{summary['num_image_directories']:,}**",
            f"- Used visual categories: **{summary['num_used_visual_categories']:,}**",
            f"- Taxonomy entries in classes.txt: **{summary['num_classes_in_taxonomy']:,}**",
            f"- Part types: **{summary['num_part_types']:,}**",
            "",
            "## Train/Test Split",
            "",
            f"- Train: **{split_counts.get(TRAIN, 0):,}**",
            f"- Test: **{split_counts.get(TEST, 0):,}**",
            "",
            "## Class Balance",
            "",
            f"- Images per used class, median: **{class_stats['total']['median']}**",
            f"- Images per used class, min/max: **{class_stats['total']['min']} / {class_stats['total']['max']}**",
            f"- Max/min class imbalance ratio: **{class_stats['max_to_min_ratio']}**",
            "",
            "## Annotation Notes",
            "",
            f"- Bounding-box area ratio, median: **{summary['bbox_area_ratio_stats']['median']}**",
            f"- Visible parts per image, median: **{summary['visible_parts_per_image_stats']['median']}**",
            "",
            "## Audit Findings",
            "",
        ]
        for issue in issues:
            example_text = ""
            if issue.examples:
                example_text = f" Examples: `{'; '.join(issue.examples[:3])}`"
            lines.append(
                f"- **{issue.severity.upper()}** `{issue.check}`: "
                f"{issue.message} (count={issue.count}).{example_text}"
            )
        lines.append("")
        return "\n".join(lines)

    def write_plots(self, out_dir: Path | str) -> Dict[str, Path]:
        """Write lightweight diagnostic plots if matplotlib is installed."""
        out_path = Path(out_dir)
        _prepare_plot_backend(out_path)
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for --plots") from exc

        written: Dict[str, Path] = {}
        rows = self.class_counts()

        totals = [int(row["total"]) for row in rows]
        plt.figure(figsize=(8, 4.5))
        plt.hist(totals, bins=30, color="#4c78a8", edgecolor="white")
        plt.xlabel("Images per class")
        plt.ylabel("Number of classes")
        plt.title("NABirds Class Count Distribution")
        plt.tight_layout()
        path = out_path / "class_count_distribution.png"
        plt.savefig(path, dpi=180)
        plt.close()
        written["class_count_distribution"] = path

        records = self.records()
        widths = [record.width for record in records]
        heights = [record.height for record in records]
        plt.figure(figsize=(6, 5))
        plt.scatter(widths, heights, s=4, alpha=0.25, color="#59a14f")
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.title("NABirds Image Dimensions")
        plt.tight_layout()
        path = out_path / "image_dimensions.png"
        plt.savefig(path, dpi=180)
        plt.close()
        written["image_dimensions"] = path

        bbox_ratios = [
            record.bbox.area_ratio(record.width, record.height)
            for record in records
            if record.width > 0 and record.height > 0
        ]
        plt.figure(figsize=(8, 4.5))
        plt.hist(bbox_ratios, bins=40, color="#f28e2b", edgecolor="white")
        plt.xlabel("Bounding-box area / image area")
        plt.ylabel("Number of images")
        plt.title("NABirds Bounding-Box Scale")
        plt.tight_layout()
        path = out_path / "bbox_area_ratio.png"
        plt.savefig(path, dpi=180)
        plt.close()
        written["bbox_area_ratio"] = path

        summary = self.summary()
        part_names = [name for _, name in sorted(self.part_names.items(), key=lambda item: int(item[0]))]
        visible_fraction = [
            summary["part_visibility"][name]["visible_fraction"] for name in part_names
        ]
        plt.figure(figsize=(8, 4.5))
        plt.bar(part_names, visible_fraction, color="#e15759")
        plt.ylim(0, 1)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Visible fraction")
        plt.title("NABirds Part Visibility")
        plt.tight_layout()
        path = out_path / "part_visibility.png"
        plt.savefig(path, dpi=180)
        plt.close()
        written["part_visibility"] = path

        return written

    def save_sample_grid(
        self,
        out_path: Path | str,
        split: Optional[str] = None,
        count: int = 12,
        seed: int = 0,
    ) -> Path:
        """Save a small annotated sample grid with bounding boxes and parts."""
        out_path = Path(out_path)
        _prepare_plot_backend(out_path.parent)
        try:
            from PIL import Image
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError as exc:
            raise RuntimeError("Pillow and matplotlib are required for sample grids") from exc

        records = self.records(split)
        rng = random.Random(seed)
        chosen = rng.sample(records, min(count, len(records)))
        cols = min(4, len(chosen))
        rows = math.ceil(len(chosen) / cols) if cols else 1

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        if hasattr(axes, "ravel"):
            flat_axes = list(axes.ravel())
        else:
            flat_axes = [axes]

        for axis in flat_axes:
            axis.axis("off")

        for axis, record in zip(flat_axes, chosen):
            with Image.open(record.image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{record.class_name}\n{record.split} | idx {record.class_index}", fontsize=8)
            axis.axis("off")
            bbox = record.bbox
            axis.add_patch(
                patches.Rectangle(
                    (bbox.x, bbox.y),
                    bbox.width,
                    bbox.height,
                    linewidth=1.5,
                    edgecolor="lime",
                    facecolor="none",
                )
            )
            xs = [part.x for part in record.parts if part.visible]
            ys = [part.y for part in record.parts if part.visible]
            axis.scatter(xs, ys, s=10, c="yellow", edgecolors="black", linewidths=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path


try:
    from torch.utils.data import Dataset as _TorchDatasetBase
except Exception:  # pragma: no cover - torch is intentionally optional
    _TorchDatasetBase = object


class NABirdsTorchDataset(_TorchDatasetBase):
    """Optional PyTorch-compatible dataset wrapper.

    The class can be imported without torch installed. In a PyTorch environment,
    it behaves like a standard ``torch.utils.data.Dataset``.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = TRAIN,
        transform: Optional[Callable[[object], object]] = None,
        target_transform: Optional[Callable[[int], object]] = None,
        return_metadata: bool = False,
    ) -> None:
        if split not in ALL_SPLITS and split != "all":
            raise ValueError(f"split must be one of 'train', 'test', or 'all', got {split!r}")
        self.metadata = NABirdsMetadata(root)
        self.samples = self.metadata.records(None if split == "all" else split)
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to load NABirds images") from exc

        record = self.samples[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")

        target = record.class_index
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_metadata:
            return image, target, record.to_metadata()
        return image, target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and summarize an extracted NABirds dataset.")
    parser.add_argument(
        "root",
        nargs="?",
        default="nabirds",
        help="Path to extracted NABirds directory containing images.txt, classes.txt, images/, etc.",
    )
    parser.add_argument("--out-dir", default="reports", help="Directory for generated reports.")
    parser.add_argument(
        "--check-image-dimensions",
        action="store_true",
        help="Open images with Pillow and compare actual dimensions to sizes.txt.",
    )
    parser.add_argument(
        "--dimension-sample",
        type=int,
        default=512,
        help="Number of images to sample for dimension checks. Use 0 or a negative value for all images.",
    )
    parser.add_argument("--plots", action="store_true", help="Write matplotlib diagnostic plots.")
    parser.add_argument("--sample-grid", action="store_true", help="Write an annotated sample image grid.")
    parser.add_argument("--sample-count", type=int, default=12, help="Number of images in the sample grid.")
    parser.add_argument("--sample-split", choices=["all", TRAIN, TEST], default="all")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dimension_sample = args.dimension_sample
    if dimension_sample <= 0:
        dimension_sample = None

    dataset = NABirdsMetadata(args.root)
    written = dataset.write_reports(
        out_dir=args.out_dir,
        check_image_dimensions=args.check_image_dimensions,
        dimension_sample=dimension_sample,
        seed=args.seed,
        plots=args.plots,
        sample_grid=args.sample_grid,
        sample_count=args.sample_count,
        sample_split=args.sample_split,
    )
    report = json.loads(written["json"].read_text(encoding="utf-8"))
    summary = report["summary"]
    issue_counts = Counter(issue["severity"] for issue in report["issues"])

    print("NABirds audit complete")
    print(f"  root: {summary['dataset_root']}")
    print(f"  images: {summary['num_images']:,}")
    print(f"  used visual categories: {summary['num_used_visual_categories']:,}")
    print(f"  split counts: {summary['split_counts']}")
    print(f"  audit severities: {dict(issue_counts)}")
    for name, path in written.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
