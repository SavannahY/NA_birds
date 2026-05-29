import { C, ROOT, addHeader, callout, footer, panel } from "./theme.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = C.pale;

  addHeader(
    ctx,
    slide,
    "Error analysis",
    "Remaining errors concentrate in variants and near-neighbor species",
    "The CLIP baseline fails in places a human field guide would also flag: sex/age labels, seasonal plumage, morphs, and look-alike species."
  );

  panel(ctx, slide, 54, 164, 542, 430, "#FFFFFF", "#E0E4DC");
  await ctx.addImage(slide, {
    path: `${ROOT}/reports/milestone3/error_analysis/clip_vit_b32_generic_viz/misclassification_grid.png`,
    x: 70,
    y: 182,
    w: 510,
    h: 382,
    fit: "contain",
    alt: "Grid of twelve CLIP misclassified NABirds test images with true and predicted labels.",
  });
  ctx.addText(slide, {
    text: "Sample CLIP mistakes from the official test split",
    x: 78,
    y: 566,
    w: 460,
    h: 18,
    fontSize: 11.8,
    bold: true,
    color: C.muted,
  });

  panel(ctx, slide, 626, 164, 596, 316, "#FFFFFF", "#E0E4DC");
  await ctx.addImage(slide, {
    path: `${ROOT}/reports/milestone3/slide_assets/top_confusions.png`,
    x: 648,
    y: 182,
    w: 552,
    h: 270,
    fit: "contain",
    alt: "Horizontal bar chart of top CLIP confusion pairs.",
  });

  panel(ctx, slide, 626, 506, 596, 100, "#FFFFFF", "#E0E4DC");
  ctx.addText(slide, {
    text: "Highest-count confusion examples",
    x: 650,
    y: 524,
    w: 300,
    h: 20,
    fontSize: 13,
    bold: true,
    color: C.tealDark,
  });
  ctx.addText(slide, {
    text:
      "Allen's vs Rufous hummingbird; Wood Duck and Mallard male vs female/eclipse; Yellow- vs Red-shafted Flicker; Spotted Sandpiper breeding vs nonbreeding/juvenile; Western vs Eastern Meadowlark.",
    x: 650,
    y: 552,
    w: 524,
    h: 44,
    fontSize: 12.6,
    color: C.ink,
  });

  callout(
    ctx,
    slide,
    "Interpretation: descriptor prompts need variant-specific field marks. Shared base descriptors are not enough for age, sex, morph, and seasonal labels.",
    54,
    616,
    1168,
    46,
    C.amberPale,
    "#E5D2A4",
    C.ink
  );

  footer(
    ctx,
    slide,
    "Sources: clip_vit_b32_generic_cached_test_predictions.csv; analyze_predictions.py; visualize_prediction_errors.py."
  );
  slide.speakerNotes.text =
    "This slide should be read as evidence for the next data cleanup step. The model is not random; its top failures are exactly the label distinctions that require fine-grained field marks.";
  return slide;
}
