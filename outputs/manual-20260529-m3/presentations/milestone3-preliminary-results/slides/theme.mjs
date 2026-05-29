export const ROOT = "/Users/zhengjieyang/Downloads/NA_birds";

export const C = {
  ink: "#16211F",
  muted: "#5D6661",
  pale: "#F7F8F3",
  panel: "#FFFFFF",
  line: "#D8DDD3",
  teal: "#167C80",
  tealDark: "#0E5558",
  orange: "#C66A2B",
  indigo: "#596AB0",
  olive: "#7B8F45",
  rose: "#B44E62",
  amber: "#D6A13B",
  greenPale: "#E9F2EE",
  bluePale: "#EEF1FA",
  rosePale: "#F7EAEE",
  amberPale: "#F8F0DE",
};

export function addHeader(ctx, slide, eyebrow, title, subtitle = "") {
  ctx.addText(slide, {
    text: eyebrow.toUpperCase(),
    x: 54,
    y: 32,
    w: 560,
    h: 20,
    fontSize: 12,
    bold: true,
    color: C.tealDark,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  ctx.addText(slide, {
    text: title,
    x: 54,
    y: 58,
    w: 990,
    h: 68,
    fontSize: 29,
    bold: true,
    color: C.ink,
    typeface: ctx.fonts.title,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  if (subtitle) {
    ctx.addText(slide, {
      text: subtitle,
      x: 56,
      y: 132,
      w: 950,
      h: 42,
      fontSize: 15.5,
      color: C.muted,
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
  }
}

export function panel(ctx, slide, x, y, w, h, fill = C.panel, line = C.line) {
  ctx.addShape(slide, {
    x,
    y,
    w,
    h,
    fill,
    line: { style: "solid", fill: line, width: 1 },
  });
}

export function label(ctx, slide, text, x, y, w, color = C.tealDark) {
  ctx.addText(slide, {
    text,
    x,
    y,
    w,
    h: 18,
    fontSize: 11.5,
    bold: true,
    color,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function metric(ctx, slide, value, labelText, x, y, w, accent = C.teal) {
  panel(ctx, slide, x, y, w, 78, "#FFFFFF", "#E0E4DC");
  ctx.addText(slide, {
    text: value,
    x: x + 14,
    y: y + 10,
    w: w - 28,
    h: 32,
    fontSize: 26,
    bold: true,
    color: accent,
    typeface: ctx.fonts.title,
  });
  ctx.addText(slide, {
    text: labelText,
    x: x + 14,
    y: y + 43,
    w: w - 28,
    h: 24,
    fontSize: 11.5,
    color: C.muted,
  });
}

export async function iconBullet(ctx, slide, icon, title, detail, x, y, w, accent = C.teal) {
  ctx.addShape(slide, { x, y, w: 38, h: 38, fill: "#F2F6F3", line: { style: "solid", fill: "#D4DED7", width: 1 } });
  await ctx.addLucideIcon(slide, { icon, x: x + 9, y: y + 9, w: 20, h: 20, color: accent, strokeWidth: 2.1 });
  ctx.addText(slide, {
    text: title,
    x: x + 50,
    y: y - 1,
    w: w - 50,
    h: 20,
    fontSize: 14.8,
    bold: true,
    color: C.ink,
  });
  ctx.addText(slide, {
    text: detail,
    x: x + 50,
    y: y + 22,
    w: w - 50,
    h: 34,
    fontSize: 11.8,
    color: C.muted,
  });
}

export function processBox(ctx, slide, title, detail, x, y, w, h, fill, stroke) {
  panel(ctx, slide, x, y, w, h, fill, stroke);
  ctx.addText(slide, {
    text: title,
    x: x + 14,
    y: y + 12,
    w: w - 28,
    h: 24,
    fontSize: 14.5,
    bold: true,
    color: C.ink,
  });
  ctx.addText(slide, {
    text: detail,
    x: x + 14,
    y: y + 39,
    w: w - 28,
    h: h - 48,
    fontSize: 11.1,
    color: C.muted,
  });
}

export function connector(ctx, slide, x1, y, x2, color = C.line) {
  const w = x2 - x1;
  ctx.addShape(slide, { x: x1, y, w, h: 2.5, fill: color, line: { style: "solid", fill: color, width: 0 } });
  ctx.addShape(slide, { x: x2 - 8, y: y - 5, w: 10, h: 12, fill: color, line: { style: "solid", fill: color, width: 0 } });
}

export function callout(ctx, slide, text, x, y, w, h, fill, stroke, color = C.ink) {
  panel(ctx, slide, x, y, w, h, fill, stroke);
  ctx.addText(slide, {
    text,
    x: x + 14,
    y: y + 12,
    w: w - 28,
    h: h - 20,
    fontSize: 13.2,
    bold: true,
    color,
  });
}

export function footer(ctx, slide, text) {
  ctx.addText(slide, {
    text,
    x: 54,
    y: 686,
    w: 1100,
    h: 16,
    fontSize: 9.5,
    color: "#78817B",
  });
}
