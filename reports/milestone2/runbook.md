# Milestone 2 Execution Runbook

This runbook turns the Step 1-6 plan into commands the team can run on a GPU machine.

## Environment

Recommended:

- Python 3.10+
- 1 GPU with 16-24 GB VRAM for baselines
- 24 GB+ VRAM for Qwen/InternVL prompted reranking
- 50 GB free disk for dataset, caches, checkpoints, and logs

Install core training dependencies:

```bash
python3 -m pip install torch torchvision transformers pillow scikit-learn tqdm
```

Use the PyTorch install selector for the right CUDA wheel if running on NVIDIA GPUs.

## Step 1: Finalize Dataloaders And Manifests

```bash
python3 scripts/build_nabirds_manifests.py \
  --dataset-root nabirds \
  --out-dir reports/milestone2 \
  --val-fraction 0.15 \
  --seed 231
```

Expected outputs:

- `nabirds_train.csv`: 20,339 images
- `nabirds_val.csv`: 3,590 images
- `nabirds_test.csv`: 24,633 images
- `nabirds_class_prompts.csv`: 1,665 prompt rows
- `nabirds_hard_negative_groups.csv`: 137 fine-grained variant groups

## Step 2: Train Full-Image Baseline

Quick smoke test:

```bash
python3 scripts/train_visual_baseline.py \
  --manifest-dir reports/milestone2 \
  --model resnet18 \
  --input-mode full \
  --no-pretrained \
  --device cpu \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 0 \
  --limit-train 16 \
  --limit-val 16
```

GPU baseline:

```bash
python3 scripts/train_visual_baseline.py \
  --manifest-dir reports/milestone2 \
  --model resnet50 \
  --input-mode full \
  --epochs 10 \
  --batch-size 64 \
  --amp
```

## Step 3: Train BBox-Crop Baseline

```bash
python3 scripts/train_visual_baseline.py \
  --manifest-dir reports/milestone2 \
  --model resnet50 \
  --input-mode bbox \
  --epochs 10 \
  --batch-size 64 \
  --amp
```

Use this to test whether localization reduces background noise.

## Step 4: Evaluate VLM Baselines

SigLIP2 zero-shot over all 555 classes:

```bash
python3 scripts/eval_vlm_zero_shot.py \
  --model google/siglip2-base-patch16-224 \
  --manifest reports/milestone2/nabirds_test.csv \
  --prompts reports/milestone2/nabirds_class_prompts.csv \
  --output-json reports/milestone2/vlm_siglip2_results.json \
  --output-predictions reports/milestone2/vlm_siglip2_predictions.csv
```

BBox-crop VLM variant:

```bash
python3 scripts/eval_vlm_zero_shot.py \
  --model google/siglip2-base-patch16-224 \
  --input-mode bbox \
  --output-json reports/milestone2/vlm_siglip2_bbox_results.json \
  --output-predictions reports/milestone2/vlm_siglip2_bbox_predictions.csv
```

Prompted VLM reranking plan:

1. Use SigLIP2 to retrieve top-k candidates.
2. Ask Qwen2.5-VL or InternVL to choose among those candidates.
3. Report whether prompted reranking fixes fine-grained errors.

## Step 5: Train Proposed Full-Image + Crop Model

Shared-backbone fused model:

```bash
python3 scripts/train_fused_full_crop.py \
  --manifest-dir reports/milestone2 \
  --model resnet50 \
  --branch-mode shared \
  --epochs 10 \
  --batch-size 32 \
  --amp \
  --macro-f1
```

Two-branch variant if GPU memory allows:

```bash
python3 scripts/train_fused_full_crop.py \
  --manifest-dir reports/milestone2 \
  --model resnet50 \
  --branch-mode two_branch \
  --epochs 10 \
  --batch-size 16 \
  --amp \
  --macro-f1
```

Train the VLM adapter, which is the current "our own model" path:

```bash
modal run scripts/modal_train_nabirds.py \
  --task train-vlm-adapter \
  --epochs 5 \
  --batch-size 32 \
  --extra-args "--contrastive-weight 0.05 --hard-negative-weight 0.05"
```

## Step 6: Analyze Errors

```bash
python3 scripts/analyze_predictions.py \
  --predictions reports/milestone2/vlm_siglip2_predictions.csv \
  --out-dir reports/milestone2/error_analysis
```

Outputs:

- `top_confusions.csv`
- `per_class_recall.csv`
- `error_examples.csv`
- `prediction_analysis_summary.json`

## Check-In Story

For the TA check-in, show:

1. VLM-first baseline ladder, not only ResNet/ViT.
2. Step 1 manifest/progress table.
3. Preliminary SigLIP2 zero-shot metrics if available.
4. Full-image vs bbox-crop baseline comparison if training finishes.
5. A few confusion examples from `error_examples.csv`.

Main claim:

> We evaluate whether modern open-source VLMs already solve NABirds, then train a lightweight localization-aware model if they fail on fine-grained variants.
