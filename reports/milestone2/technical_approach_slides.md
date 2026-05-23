# Milestone 2 Technical Approach: VLM-Aware NABirds Classification

## Slide 1: Direction and TA Feedback

- Task: fine-grained classification on NABirds: 48,562 images, 555 used categories.
- Shift from "CNN-only" baselines to a VLM-first evaluation ladder.
- TA feedback addressed:
  - Evaluate strong open-source VLMs, not only common CNNs.
  - Use SigLIP/SigLIP2 zero-shot as the main VLM baseline.
  - Add Qwen2.5-VL and InternVL as prompted reranking baselines.
  - Keep ResNet and ViT as visual-only baselines.
- Report top-1, top-5, macro-F1, balanced accuracy, per-class recall, and confusion patterns.

Speaker notes: The goal for Milestone 2 is to show a credible baseline stack before investing in a custom method. The key change is treating modern open-source VLMs as first-class baselines, not only comparing against image classifiers.

---

## Slide 2: Baseline Ladder

- Sanity checks: random and majority-class predictors.
- Visual baselines:
  - ResNet18 from scratch as a weak trainable check.
  - ImageNet ResNet50 on full images and clipped bbox crops.
  - ImageNet ViT-B/16 as a stronger visual transformer baseline.
- Main zero-shot VLM baseline:
  - SigLIP2: `google/siglip2-base-patch16-224`.
  - SigLIP fallback: `google/siglip-base-patch16-224`.
  - Use species prompts such as photo, field-guide, and fine-grained field-mark templates.
- Prompted VLM reranking:
  - Start from SigLIP2 top-k candidates.
  - Rerank with `Qwen/Qwen2.5-VL-7B-Instruct` and `OpenGVLab/InternVL3-8B-Instruct`.

Speaker notes: The baseline ladder separates pure vision capacity, zero-shot image-text alignment, and prompted reasoning. This makes it easier to explain where gains come from and where VLMs still confuse visually similar birds.

---

## Slide 3: Proposed Lightweight Alignment Method

- Use SigLIP2 as the image-text backbone rather than training a large VLM end to end.
- Add a lightweight domain adapter:
  - Train small projection/adaptation layers on image and text embeddings.
  - Encode both full image and clipped bbox crop; fuse scores late.
  - Keep class-name and field-guide prompt text as the text side.
- Training objective:
  - Cross-entropy loss over 555 bird classes.
  - Image-text contrastive loss to keep paired image and class prompt embeddings aligned.
  - Hard-negative loss focused on near-neighbor species and same-family variants.
- Hard negatives come from class-family groupings and high-confusion SigLIP/ResNet predictions.

Speaker notes: This is intentionally lightweight. We use VLM alignment already learned by SigLIP2, then adapt it to NABirds with class labels, prompt text, bbox crops, and hard negatives from similar species.

---

## Slide 4: Step 1-6 Implementation Plan

1. Build manifests and prompts: create train/val/test CSVs, clipped bbox metadata, class prompts, and hard-negative groups.
2. Train full-image visual baseline: ImageNet ResNet50, with ResNet18 from scratch as a sanity check.
3. Train bbox-crop visual baseline: reuse the same model setup on clipped bird crops to test localization value.
4. Run VLM baselines: SigLIP2/SigLIP zero-shot over all class prompts, then Qwen2.5-VL and InternVL prompted reranking on SigLIP2 top-k.
5. Train domain-adapted alignment model: full image plus crop fusion with CE + contrastive + hard-negative losses.
6. Evaluate and analyze: compare metrics, inspect confusion by family/species variants, and choose the strongest M2 result for the check-in.

Speaker notes: Steps 1-3 are already reflected in the current scripts/config. Steps 4-6 add the TA-requested VLM comparisons and the lightweight method that should be feasible within the milestone.

---

## Slide 5: Evaluation and Check-In Deliverables

- Primary comparison table:
  - ResNet50 full image, ResNet50 bbox crop, ViT-B/16.
  - SigLIP2 zero-shot and SigLIP fallback.
  - Qwen2.5-VL and InternVL reranking.
  - Domain-adapted alignment model.
- Error analysis:
  - Top confusions by family and class-family variants.
  - Cases where crop helps vs. hurts.
  - Cases where prompted reranking fixes or worsens SigLIP2 top-k.
- Decision criteria:
  - Best top-1/top-5 plus macro-F1.
  - Evidence that VLM baselines were fairly evaluated.
  - Clear next step for improving fine-grained bird discrimination.

Speaker notes: The check-in should focus on the baseline table and a few concrete errors. The important story is whether the proposed alignment model improves over strong zero-shot and prompted VLM baselines, not just over CNNs.

---

## References / Model Cards

- SigLIP2 baseline: `google/siglip2-base-patch16-224`
- SigLIP fallback: `google/siglip-base-patch16-224`
- Prompted VLM reranker: `Qwen/Qwen2.5-VL-7B-Instruct`
- Prompted VLM reranker: `OpenGVLab/InternVL3-8B-Instruct`

Speaker notes: These are current open-source model-card targets for the Milestone 2 implementation plan. The project should report exact checkpoint names in the final experimental table.
