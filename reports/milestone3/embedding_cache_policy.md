# Embedding Cache Policy

Always save computed text and image embeddings to disk before running scoring,
analysis, adapters, or deck generation. Do not rely on in-memory-only embeddings
for Milestone 3 or later experiments.

The goal is to avoid recomputing expensive VLM features when rerunning prompt
ablations, cached evaluation, adapter training, error analysis, or slide assets.

## Current Modal Cache Locations

- CLIP image embeddings:
  `/data/nabirds_runs/runs/milestone3_clip_shards/vlm_image_features/`
- CLIP train/val image embeddings for cached adapters:
  `/data/nabirds_runs/runs/milestone3_clip_trainval/vlm_image_features/`
- SigLIP2 image embeddings:
  `/data/nabirds_runs/runs/milestone3_siglip2_shards/vlm_image_features/`
- SigLIP base image embeddings:
  `/data/nabirds_runs/runs/milestone3_siglip_base/vlm_image_features/`
- CLIP and SigLIP2 improved text embeddings:
  `/data/nabirds_runs/runs/milestone3_improve/vlm_text_features/`
- SigLIP base text embeddings:
  `/data/nabirds_runs/runs/milestone3_siglip_base/vlm_text_features/`
- Cached CLIP adapter checkpoint:
  `/data/nabirds_runs/runs/milestone3_clip_adapter/vlm_adapter_cached/`

## Required Workflow

1. Build or update prompt CSVs.
2. Run text-feature precompute and save the `.pt` payload plus manifest.
3. Run image-feature precompute and save sharded `.pt` payloads plus manifests.
4. Run cached evaluation from saved text and image features.
5. Generate prediction CSVs, analysis artifacts, and deck assets from cached
   evaluation outputs.

Any new CLIP, SigLIP, adapter, or fusion experiment should follow this policy
unless it is a tiny smoke test explicitly labeled as non-persistent.
