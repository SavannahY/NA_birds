# NABirds Dataset Audit

## Core Counts

- Dataset root: `/Users/zhengjieyang/Downloads/NA_birds/nabirds`
- Images listed: **48,562**
- Image files on disk: **48,562**
- Image directories: **555**
- Used visual categories: **555**
- Taxonomy entries in classes.txt: **1,011**
- Part types: **11**

## Train/Test Split

- Train: **23,929**
- Test: **24,633**

## Class Balance

- Images per used class, median: **91**
- Images per used class, min/max: **13.0 / 120.0**
- Max/min class imbalance ratio: **9.2308**

## Annotation Notes

- Bounding-box area ratio, median: **0.2826**
- Visible parts per image, median: **9.0**

## Audit Findings

- **WARNING** `bbox_bounds`: Some bounding boxes are empty or extend outside the recorded image size (count=201). Examples: `004f5dee-8f08-49eb-8ed9-285bd8a5da5b; 017a78eb-20d3-41f5-a1d5-b89bd043590c; 03fd102c-b647-4627-9f59-b32adb356846`
- **INFO** `image_dimension_check`: Checked actual dimensions for 48562 image files (count=48562).
