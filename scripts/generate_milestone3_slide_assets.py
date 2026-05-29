#!/usr/bin/env python3
"""Generate slide-ready Milestone 3 charts from current metrics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from textwrap import shorten

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "milestone3" / "slide_assets"
IMPROVE_DIR = ROOT / "reports" / "milestone3" / "modal_outputs_improve"

COLORS = {
    "ink": "#16211F",
    "muted": "#5D6661",
    "line": "#D8DDD3",
    "teal": "#167C80",
    "orange": "#C66A2B",
    "indigo": "#596AB0",
    "rose": "#B44E62",
    "olive": "#7B8F45",
    "amber": "#D6A13B",
    "pale": "#F7F8F3",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _percent(value: float) -> float:
    return 100.0 * float(value)


def load_metrics() -> list[dict]:
    resnet = _read_json(
        ROOT
        / "reports"
        / "milestone2"
        / "error_analysis"
        / "official_visual_models"
        / "resnet50_full"
        / "prediction_analysis_summary.json"
    )
    custom = _read_json(
        ROOT
        / "reports"
        / "milestone2"
        / "error_analysis"
        / "official_visual_models"
        / "custom_visual"
        / "prediction_analysis_summary.json"
    )
    clip_generic = _read_json(IMPROVE_DIR / "clip_generic_cached_improved_test.json")
    clip_short = _read_json(IMPROVE_DIR / "clip_short_variant_cached_test.json")
    siglip2_fixed = _read_json(IMPROVE_DIR / "siglip2_generic_fixed_cached_test.json")
    siglip_base_short = _read_json(IMPROVE_DIR / "siglip_base_short_variant_cached_test.json")
    clip_adapter = _read_json(IMPROVE_DIR / "clip_short_variant_residual_adapter_test_test.json")
    rows = [
        {
            "model": "ResNet-50\nvisual",
            "type": "visual",
            "top1": _percent(resnet["top1_accuracy"]),
            "top5": _percent(resnet["top5_accuracy"]),
            "macro_f1": _percent(resnet["macro_f1"]),
            "color": COLORS["indigo"],
        },
        {
            "model": "ConvNeXt-B\n5-view",
            "type": "visual",
            "top1": _percent(custom["top1_accuracy"]),
            "top5": _percent(custom["top5_accuracy"]),
            "macro_f1": _percent(custom["macro_f1"]),
            "color": COLORS["orange"],
        },
        {
            "model": "CLIP ViT-B/32\ngeneric text",
            "type": "vlm",
            "top1": _percent(clip_generic["top1_accuracy"]),
            "top5": _percent(clip_generic["top5_accuracy"]),
            "macro_f1": _percent(clip_generic["macro_f1"]),
            "color": COLORS["teal"],
        },
        {
            "model": "CLIP ViT-B/32\nshort variant text",
            "type": "vlm",
            "top1": _percent(clip_short["top1_accuracy"]),
            "top5": _percent(clip_short["top5_accuracy"]),
            "macro_f1": _percent(clip_short["macro_f1"]),
            "color": COLORS["rose"],
        },
        {
            "model": "SigLIP2 base\nfixed generic",
            "type": "vlm",
            "top1": _percent(siglip2_fixed["top1_accuracy"]),
            "top5": _percent(siglip2_fixed["top5_accuracy"]),
            "macro_f1": _percent(siglip2_fixed["macro_f1"]),
            "color": COLORS["olive"],
        },
        {
            "model": "SigLIP base\nshort variant text",
            "type": "vlm",
            "top1": _percent(siglip_base_short["top1_accuracy"]),
            "top5": _percent(siglip_base_short["top5_accuracy"]),
            "macro_f1": _percent(siglip_base_short["macro_f1"]),
            "color": COLORS["amber"],
        },
        {
            "model": "CLIP ViT-B/32\ncached adapter",
            "type": "vlm",
            "top1": _percent(clip_adapter["top1_accuracy"]),
            "top5": _percent(clip_adapter["top5_accuracy"]),
            "macro_f1": _percent(clip_adapter["macro_f1"]),
            "color": COLORS["ink"],
        },
    ]
    return rows


def results_comparison() -> Path:
    metrics = load_metrics()
    labels = [row["model"] for row in metrics]
    y = np.arange(len(labels))
    h = 0.24

    fig, ax = plt.subplots(figsize=(12.4, 5.9))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(y - h, [row["top1"] for row in metrics], height=h, color=[row["color"] for row in metrics], label="Top-1")
    ax.barh(y, [row["top5"] for row in metrics], height=h, color="#A8B3AA", label="Top-5")
    ax.barh(y + h, [row["macro_f1"] for row in metrics], height=h, color="#D8BFA5", label="Macro-F1")
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=COLORS["ink"])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E9E1", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["line"])
    ax.tick_params(axis="x", colors=COLORS["muted"], labelsize=10)
    ax.set_xlabel("Official test score (%)", fontsize=11.5, color=COLORS["muted"])
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, -0.16), fontsize=10.5)

    for i, row in enumerate(metrics):
        for offset, key in [(-h, "top1"), (0, "top5"), (h, "macro_f1")]:
            value = row[key]
            ax.text(min(value + 1.0, 96), i + offset, f"{value:.1f}", va="center", fontsize=9.2, color=COLORS["ink"])

    ax.set_title("Milestone 3 official test comparison", fontsize=17, fontweight="bold", color=COLORS["ink"], pad=12)
    return _save(fig, "results_comparison.png")


def descriptor_ablation() -> Path:
    rows = [
        ("Generic", _read_json(IMPROVE_DIR / "clip_generic_cached_improved_test.json"), _read_json(IMPROVE_DIR / "siglip_base_generic_cached_test.json")),
        ("Class only", _read_json(IMPROVE_DIR / "clip_class_name_only_cached_test.json"), _read_json(IMPROVE_DIR / "siglip_base_class_name_only_cached_test.json")),
        ("Short variant", _read_json(IMPROVE_DIR / "clip_short_variant_cached_test.json"), _read_json(IMPROVE_DIR / "siglip_base_short_variant_cached_test.json")),
        ("Descriptor only", _read_json(IMPROVE_DIR / "clip_descriptor_only_cached_test.json"), _read_json(IMPROVE_DIR / "siglip_base_descriptor_only_cached_test.json")),
        ("Descriptor + name", _read_json(IMPROVE_DIR / "clip_clean_descriptor_cached_test.json"), _read_json(IMPROVE_DIR / "siglip_base_clean_descriptor_cached_test.json")),
    ]
    labels = [row[0].replace(" ", "\n") for row in rows]
    clip_top1 = [_percent(row[1]["top1_accuracy"]) for row in rows]
    siglip_top1 = [_percent(row[2]["top1_accuracy"]) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.bar(x - 0.18, clip_top1, width=0.36, color=COLORS["teal"], label="CLIP ViT-B/32")
    ax.bar(x + 0.18, siglip_top1, width=0.36, color=COLORS["amber"], label="SigLIP base")
    ax.set_ylim(0, 58)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, color=COLORS["ink"])
    ax.grid(axis="y", color="#E5E9E1")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["line"])
    ax.tick_params(axis="y", labelsize=10, colors=COLORS["muted"])
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    ax.set_ylabel("Official test score (%)", fontsize=11, color=COLORS["muted"])
    ax.set_title("Species names matter; short variant prompts are strongest", fontsize=15.5, fontweight="bold", color=COLORS["ink"])
    for xpos, value in zip(x - 0.18, clip_top1):
        ax.text(xpos, value + 1.2, f"{value:.1f}", ha="center", fontsize=10.5, color=COLORS["ink"], fontweight="bold")
    for xpos, value in zip(x + 0.18, siglip_top1):
        ax.text(xpos, value + 1.2, f"{value:.1f}", ha="center", fontsize=10.5, color=COLORS["ink"], fontweight="bold")
    return _save(fig, "descriptor_ablation.png")


def top_confusions() -> Path:
    rows = []
    path = ROOT / "reports" / "milestone3" / "error_analysis" / "clip_vit_b32_generic" / "top_confusions.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if len(rows) == 10:
                break

    labels = [
        shorten(f"{row['target_name']} -> {row['pred_name']}", width=64, placeholder="...")
        for row in rows
    ]
    values = [int(row["count"]) for row in rows]
    colors = [COLORS["rose"] if any(token in labels[i] for token in ["male", "Female", "juvenile", "Breeding", "Nonbreeding"]) else COLORS["teal"] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    fig.patch.set_facecolor("#FFFFFF")
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, height=0.56)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.3, color=COLORS["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 8)
    ax.grid(axis="x", color="#E5E9E1")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["line"])
    ax.tick_params(axis="x", colors=COLORS["muted"], labelsize=9.5)
    ax.set_xlabel("Incorrect test examples", fontsize=10.5, color=COLORS["muted"])
    ax.set_title("Top CLIP confusion pairs are variants and near-neighbor species", fontsize=15.5, fontweight="bold", color=COLORS["ink"])
    for ypos, value in zip(y, values):
        ax.text(value + 0.8, ypos, str(value), va="center", fontsize=10, color=COLORS["ink"], fontweight="bold")
    return _save(fig, "top_confusions.png")


def write_metrics_table() -> Path:
    metrics = load_metrics()
    path = OUT_DIR / "results_table.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "top1", "top5", "macro_f1"])
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row[key] for key in ("model", "top1", "top5", "macro_f1")})
    return path


def main() -> None:
    outputs = [
        results_comparison(),
        descriptor_ablation(),
        top_confusions(),
        write_metrics_table(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
