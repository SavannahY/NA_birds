import { C, ROOT, addHeader, callout, footer, metric, panel } from "./theme.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = C.pale;

  addHeader(
    ctx,
    slide,
    "Milestone 3 preliminary results",
    "Visual branch is strong; VLM baselines expose the text-grounding gap",
    "Official 555-class NABirds test results now include cached zero-shot CLIP and SigLIP2 baselines alongside the visual models."
  );

  metric(ctx, slide, "91.0%", "custom ConvNeXt-B 5-view top-1", 54, 170, 250, C.orange);
  metric(ctx, slide, "40.3%", "CLIP ViT-B/32 generic prompt top-1", 54, 262, 250, C.teal);
  metric(ctx, slide, "37.7%", "CLIP descriptor prompt top-1", 54, 354, 250, C.rose);
  metric(ctx, slide, "-2.6 pt", "descriptor prompt delta vs generic CLIP", 54, 446, 250, C.rose);

  callout(
    ctx,
    slide,
    "Takeaway: zero-shot text grounding is measurable, but not yet competitive with the visual branch. Descriptors need cleanup before they help CLIP.",
    54,
    542,
    250,
    96,
    "#FFFFFF",
    "#E0E4DC",
    C.ink
  );

  panel(ctx, slide, 338, 168, 884, 470, "#FFFFFF", "#E0E4DC");
  await ctx.addImage(slide, {
    path: `${ROOT}/reports/milestone3/slide_assets/results_comparison.png`,
    x: 360,
    y: 186,
    w: 840,
    h: 410,
    fit: "contain",
    alt: "Bar chart comparing official test top-1, top-5, and macro-F1 across visual and CLIP models.",
  });

  ctx.addText(slide, {
    text: "Baseline comparison uses the same official 24,633-image test split.",
    x: 370,
    y: 604,
    w: 780,
    h: 24,
    fontSize: 13,
    bold: true,
    color: C.muted,
  });

  footer(
    ctx,
    slide,
    "Sources: milestone2 official visual analyses; milestone3 cached CLIP/SigLIP2 eval JSONs; image/text features saved on Modal volume."
  );
  slide.speakerNotes.text =
    "Present this as evidence that the evaluation path is now real. The visual model remains the performance anchor. CLIP is the useful text-side baseline; the SigLIP2 HF run is included transparently but appears non-informative and needs follow-up debugging.";
  return slide;
}
