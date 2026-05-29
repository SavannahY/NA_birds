import { C, addHeader, callout, footer, iconBullet, panel } from "./theme.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = C.pale;

  addHeader(
    ctx,
    slide,
    "Limitations and next steps",
    "Move from zero-shot baselines to cleaned descriptors, cached adapters, and fusion",
    "The milestone result is a working evaluation path plus evidence that prompt quality and variant specificity matter."
  );

  panel(ctx, slide, 54, 172, 560, 346, "#FFFFFF", "#E0E4DC");
  ctx.addText(slide, {
    text: "Current limitations",
    x: 82,
    y: 196,
    w: 300,
    h: 24,
    fontSize: 17,
    bold: true,
    color: C.rose,
  });
  await iconBullet(
    ctx,
    slide,
    "FileWarning",
    "Descriptor normalization is still imperfect",
    "Many rows include long or redundant field marks, and some descriptors mix nonvisual or broad natural-language phrasing.",
    82,
    238,
    480,
    C.rose
  );
  await iconBullet(
    ctx,
    slide,
    "Split",
    "Variant labels often share base descriptors",
    "279 official classes inherit shared descriptors across sex, age, morph, breeding, or nonbreeding variants.",
    82,
    318,
    480,
    C.orange
  );
  await iconBullet(
    ctx,
    slide,
    "Gauge",
    "No official adapter result yet",
    "Zero-shot CLIP is complete; SigLIP2 HF scoring appears non-informative and needs debugging before adapter or LoRA work.",
    82,
    398,
    480,
    C.indigo
  );

  panel(ctx, slide, 666, 172, 556, 346, "#FFFFFF", "#E0E4DC");
  ctx.addText(slide, {
    text: "Next engineering sequence",
    x: 694,
    y: 196,
    w: 330,
    h: 24,
    fontSize: 17,
    bold: true,
    color: C.tealDark,
  });
  await iconBullet(
    ctx,
    slide,
    "Scissors",
    "Clean descriptors",
    "Shorten to visible field marks; add targeted variant terms for adult male, female/juvenile, immature, breeding, and nonbreeding classes.",
    694,
    238,
    470,
    C.teal
  );
  await iconBullet(
    ctx,
    slide,
    "FlaskConical",
    "Run prompt ablations",
    "Compare class-only, descriptor-only, label+descriptor, variant-only, and manually cleaned descriptor prompt sets using cached image features.",
    694,
    318,
    470,
    C.olive
  );
  await iconBullet(
    ctx,
    slide,
    "Workflow",
    "Train cached adapter and fuse logits",
    "Freeze image encoder, train a residual adapter over saved features, then combine VLM logits with the custom visual branch.",
    694,
    398,
    470,
    C.orange
  );

  panel(ctx, slide, 54, 548, 1168, 92, "#FFFFFF", "#E0E4DC");
  ctx.addText(slide, {
    text: "Embedding persistence",
    x: 82,
    y: 570,
    w: 240,
    h: 20,
    fontSize: 13,
    bold: true,
    color: C.tealDark,
  });
  ctx.addText(slide, {
    text:
      "Saved artifacts: CLIP and SigLIP2 image shards plus CLIP/SigLIP2 text features on Modal volume. Full CLIP cached scoring now runs in seconds after feature generation.",
    x: 82,
    y: 592,
    w: 760,
    h: 30,
    fontSize: 12.8,
    bold: true,
    color: C.ink,
  });
  callout(
    ctx,
    slide,
    "M4 target: cleaned prompts + adapter/fusion on the official test split.",
    898,
    566,
    292,
    56,
    C.greenPale,
    "#C7DCD2",
    C.ink
  );

  footer(
    ctx,
    slide,
    "Next work: descriptor cleanup, cached adapter training on Modal, VLM/visual logit fusion, and expanded prompt error analysis."
  );
  slide.speakerNotes.text =
    "The recommendation is to stay lightweight: exploit cached embeddings first, use a residual adapter, and defer LoRA or full model fine-tuning until the descriptor set has been cleaned.";
  return slide;
}
