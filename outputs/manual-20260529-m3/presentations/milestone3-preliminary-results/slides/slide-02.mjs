import { C, ROOT, addHeader, callout, connector, footer, iconBullet, metric, panel, processBox } from "./theme.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = C.pale;

  addHeader(
    ctx,
    slide,
    "CLIP/SigLIP evaluation setup",
    "Expert descriptors are aligned to NABirds IDs and cached for reuse",
    "We keep the official leaf classes intact and preserve variant labels inside the prompt text."
  );

  processBox(ctx, slide, "Descriptor CSV", "1,011 class-description rows from the uploaded file.", 58, 180, 170, 98, C.greenPale, "#C7DCD2");
  connector(ctx, slide, 232, 228, 274, C.line);
  processBox(ctx, slide, "Join on raw_class_id", "Restrict to the 555 image-bearing official classes.", 276, 180, 172, 98, C.bluePale, "#D1D8EE");
  connector(ctx, slide, 452, 228, 494, C.line);
  processBox(ctx, slide, "Variant prompts", "Class label plus visual field marks; no class splitting.", 496, 180, 176, 98, C.amberPale, "#E5D2A4");

  processBox(ctx, slide, "Text features", "Mean prompt embedding per class for generic and descriptor sets.", 164, 336, 180, 100, "#FFFFFF", "#D8DDD3");
  connector(ctx, slide, 348, 386, 390, C.line);
  processBox(ctx, slide, "Image features", "Full-image test embeddings saved in reusable shards.", 392, 336, 180, 100, "#FFFFFF", "#D8DDD3");
  connector(ctx, slide, 576, 386, 616, C.line);
  processBox(ctx, slide, "Cosine retrieval", "Top-1, top-5, macro-F1, prediction CSV, error assets.", 618, 336, 180, 100, "#FFFFFF", "#D8DDD3");

  connector(ctx, slide, 580, 278, 580, C.line);
  ctx.addShape(slide, { x: 580, y: 280, w: 2.5, h: 55, fill: C.line, line: { style: "solid", fill: C.line, width: 0 } });

  panel(ctx, slide, 838, 170, 384, 170, "#FFFFFF", "#E0E4DC");
  metric(ctx, slide, "555", "leaf classes", 862, 195, 104, C.teal);
  metric(ctx, slide, "958", "filled rows", 978, 195, 104, C.indigo);
  metric(ctx, slide, "1,665", "prompt rows", 1094, 195, 104, C.orange);

  panel(ctx, slide, 838, 366, 384, 196, "#FFFFFF", "#E0E4DC");
  ctx.addText(slide, {
    text: "Prompt example",
    x: 862,
    y: 386,
    w: 240,
    h: 22,
    fontSize: 13,
    bold: true,
    color: C.tealDark,
  });
  ctx.addText(slide, {
    text:
      "Broad-billed Hummingbird (Adult Male): small body, long straight bill, long straight tail, rich green plumage, blue throat, red bill, golden-green upperparts.",
    x: 862,
    y: 420,
    w: 330,
    h: 94,
    fontSize: 14,
    bold: true,
    color: C.ink,
  });
  ctx.addText(slide, {
    text: "Policy: preserve labels like Adult male, Female/juvenile, Immature, Breeding, and Nonbreeding.",
    x: 862,
    y: 518,
    w: 328,
    h: 28,
    fontSize: 11.8,
    color: C.muted,
  });

  panel(ctx, slide, 58, 494, 740, 86, "#FFFFFF", "#E0E4DC");
  await iconBullet(
    ctx,
    slide,
    "Database",
    "Embedding cache is now part of the experiment contract",
    "Text and image embeddings are written to Modal volume paths before scoring, so reruns can evaluate new prompt sets without recomputing image features.",
    84,
    518,
    670,
    C.teal
  );

  callout(
    ctx,
    slide,
    "Known issue: 279 classes are variant labels sharing base-species descriptors, so descriptor prompts can blur the exact sex/age/morph target.",
    838,
    580,
    384,
    68,
    C.rosePale,
    "#E5C8D1",
    C.ink
  );

  footer(
    ctx,
    slide,
    "Sources: reports/milestone3/nabirds_descriptor_prompts.csv; descriptor_prompt_summary.json; Modal cached feature manifests."
  );
  slide.speakerNotes.text =
    "The key correction from the initial plan is that we do not split male, female, juvenile, breeding, or nonbreeding labels into new classes. The prompt carries the variant distinction while the target space remains official NABirds.";
  return slide;
}
