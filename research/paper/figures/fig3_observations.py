"""Fig. 3 (Section 3.2): how the predicted future is damaged (a) and how it is read (b).

python research/paper/figures/fig3_observations.py [--overwrite]

Since 2026-11-13 the slides in CVPR2027/figures are edited by hand and the PDFs of the paper are exported from them
(PowerPoint, save as PDF), so this script writes to figure3_generated/ next to it and replaces the files of the paper
only with --overwrite, which discards the hand edits. fig3_gap_arrows.py is a later edit made directly on the slide.
  -> CVPR2027/figures/figure3_a.pdf, figure3_b.pdf      the figures of the paper
  -> CVPR2027/figures/figure3_a.pptx, figure3_b.pptx    the same panels as editable PowerPoint slides
  -> figure3_assets/*.png, figure3_*_preview.png            the pictures placed on the slides, and previews

Each panel is one list of primitives (text, rectangle, line, arrow, picture) in inches at print size: the subfigures
take 0.36 and 0.62 of \\textwidth (6.875 in) and share one height. The list is drawn twice, with matplotlib for the
PDF and with python-pptx for the slide, so the two stay identical and every label, bar and arrow of the slide is a
native shape. A slide exported from PowerPoint as PDF replaces the PDF of the same name.

(a) Read counts measured by the LAQ calibration (allocation.json): BF16 weights, the 4-bit rounding error of one source
    with random signs, the sign pattern kept over the denoising steps (frozen) or redrawn at every read. Bars show the
    future error in the reader metric, Z_M, relative to the redrawn error of the same source.
(b) Diagonal of the reader metric M (job 26073871, 80 contexts) summed over the channels of each token of the first
    predicted frame, over that frame as the BF16 model predicts it for one calibration context (decoded by
    experiment/idea-validation/paper-fig3-export-20261112). The two small maps are the same quantity estimated on the
    contexts of even and of odd case id, the split behind the two agreement numbers.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "CVPR2027" / "figures"
ASSETS = HERE / "figure3_assets"

# ---- data -------------------------------------------------------------------
CALIB = ROOT / "experiment/idea-validation/laq-fastwam-20261111/pulled/results/fastwam/calib/allocation.json"
METRIC = ROOT / "experiment/idea-validation/fastwam-guided-hessian-20261028/pulled/26073871.pbs101/ravqc/metric/summary.json"
EXPORT = ROOT / "experiment/idea-validation/paper-fig3-export-20261112/pulled/results"
CASE, FRAME = 2042, 4  # context shown in (b) and its decoded frame; frames 1-4 decode the first predicted latent frame
VIEWS = ["third-person view", "wrist view"]
# sources of panel (a): (key in allocation.json, label)
SOURCES = [("obsA", "observation\ntokens"), ("obsKV", "observation\ncache"), ("text.video", "text\ncache")]
STEPS = 10

# ---- style ------------------------------------------------------------------
RED, GREY = "#B64342", "#9A9AA0"            # (a): frozen error, redrawn error (the reference, kept neutral)
BLUE = "#0F4D92"                            # (b): the number that carries the claim
HEAT = ["#FFF6D6", "#FDCB6E", "#F28C38", "#D8432B", "#8C1020"]  # (b): share of tr(M), light to dark
INK, MUTED, RULE = "#222222", "#6E6E73", "#C9C9CE"
TEXTWIDTH, HEIGHT = 6.875, 1.55
FS = 7.4
FONT = "Cambria"

plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42})


class Panel:
    """Primitives in inches, origin at the bottom left, drawn in the order they are added."""

    def __init__(self, name, width):
        self.name, self.width, self.items = name, width, []
        self.fig = plt.figure(figsize=(width, HEIGHT))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")

    def _z(self):
        return 2 + len(self.items)

    def text(self, x, y, s, size=FS, color=INK, bold=False, ha="center", rot=0):
        """Text centred vertically on y; `ha` places it horizontally about x; rot=90 reads upwards."""
        artist = self.ax.text(x, y, s, fontsize=size, color=color, fontweight="bold" if bold else "normal", ha=ha,
                              va="center", rotation=rot, linespacing=1.12, zorder=self._z())
        self.items.append(("text", artist, s, size, color, bold, ha, rot))

    def rect(self, x, y, w, h, fill=None, edge=None, lw=0.6, round_top=0.0):
        """A rectangle; round_top is the radius of its two upper corners in inches."""
        if round_top:
            r = min(round_top, w / 2, h)
            vertices = [(x, y), (x, y + h - r), (x, y + h), (x + r, y + h), (x + w - r, y + h), (x + w, y + h),
                        (x + w, y + h - r), (x + w, y), (x, y)]
            codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3,
                     MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
            patch = PathPatch(MplPath(vertices, codes), fc=fill or "none", ec=edge or "none", lw=lw, zorder=self._z())
        else:
            patch = Rectangle((x, y), w, h, fc=fill or "none", ec=edge or "none", lw=lw, zorder=self._z())
        self.ax.add_patch(patch)
        self.items.append(("rect", x, y, w, h, fill, edge, lw, round_top))

    def line(self, x1, y1, x2, y2, color=INK, lw=0.8, arrow=False):
        if arrow:
            self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), shrinkA=0, shrinkB=0, lw=lw, color=color,
                                              arrowstyle=f"-|>,head_length={2.6 * lw},head_width={1.5 * lw}",
                                              mutation_scale=1, capstyle="round", zorder=self._z()))
        else:
            self.ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt", zorder=self._z())
        self.items.append(("line", x1, y1, x2, y2, color, lw, arrow))

    def image(self, x, y, w, h, pixels, asset):
        """A picture (float RGB or RGBA in 0..1, drawn as it is); it is also written to figure3_assets/."""
        ASSETS.mkdir(exist_ok=True)
        path = ASSETS / f"{asset}.png"
        Image.fromarray((np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)).save(path)
        self.ax.imshow(pixels, extent=(x, x + w, y, y + h), aspect="auto", interpolation="nearest", zorder=self._z())
        self.items.append(("image", x, y, w, h, path))

    def save(self):
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, HEIGHT)
        out = OUT if "--overwrite" in sys.argv else HERE / "figure3_generated"
        out.mkdir(exist_ok=True)
        self.fig.savefig(out / f"{self.name}.pdf")
        self.fig.savefig(HERE / f"{self.name}_preview.png", dpi=400)
        self._slide().save(out / f"{self.name}.pptx")
        plt.close(self.fig)
        print("wrote", out / f"{self.name}.pdf", "and", out / f"{self.name}.pptx")

    # ---- the same primitives as PowerPoint shapes --------------------------------
    def _slide(self):
        deck = Presentation()
        deck.slide_width, deck.slide_height = Inches(self.width), Inches(HEIGHT)
        shapes = deck.slides.add_slide(deck.slide_layouts[6]).shapes
        renderer, dpi = self.fig.canvas.get_renderer(), self.fig.dpi
        top = lambda y: Inches(HEIGHT - y)  # noqa: E731  (the slide measures from its top edge)
        rgb = lambda c: RGBColor.from_string(c[1:].upper())  # noqa: E731
        for item in self.items:
            kind = item[0]
            if kind == "text":
                _, artist, s, size, color, bold, ha, rot = item
                box = artist.get_window_extent(renderer)
                w, h = (box.width / dpi + 0.04, box.height / dpi + 0.02)
                cx, cy = (box.x0 + box.x1) / 2 / dpi, (box.y0 + box.y1) / 2 / dpi
                if rot:
                    w, h = h, w
                if not rot:  # the pad goes to the side away from the anchor, so that the text stays where it is
                    cx += {"left": 0.02, "center": 0.0, "right": -0.02}[ha]
                shape = shapes.add_textbox(Inches(cx - w / 2), top(cy + h / 2), Inches(w), Inches(h))
                shape.rotation = -rot
                frame = shape.text_frame
                frame.word_wrap, frame.auto_size, frame.vertical_anchor = False, MSO_AUTO_SIZE.NONE, MSO_ANCHOR.MIDDLE
                frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
                for k, row in enumerate(s.split("\n")):
                    paragraph = frame.paragraphs[0] if k == 0 else frame.add_paragraph()
                    paragraph.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[ha]
                    paragraph.line_spacing = 0.93
                    run = paragraph.add_run()
                    run.text = row
                    run.font.name, run.font.size, run.font.bold = FONT, Pt(size), bold
                    run.font.color.rgb = rgb(color)
            elif kind == "rect":
                _, x, y, w, h, fill, edge, lw, round_top = item
                kind_of = MSO_SHAPE.ROUND_2_SAME_RECTANGLE if round_top else MSO_SHAPE.RECTANGLE
                shape = shapes.add_shape(kind_of, Inches(x), top(y + h), Inches(w), Inches(h))
                if round_top:
                    shape.adjustments[0], shape.adjustments[1] = min(round_top / min(w, h), 0.5), 0
                shape.shadow.inherit = False
                if fill:
                    shape.fill.solid()
                    shape.fill.fore_color.rgb = rgb(fill)
                else:
                    shape.fill.background()
                if edge:
                    shape.line.color.rgb, shape.line.width = rgb(edge), Pt(lw)
                else:
                    shape.line.fill.background()
            elif kind == "line":
                _, x1, y1, x2, y2, color, lw, arrow = item
                shape = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), top(y1), Inches(x2), top(y2))
                shape.line.color.rgb, shape.line.width = rgb(color), Pt(lw)
                outline = shape.line._get_or_add_ln()
                if arrow:
                    size = "sm" if lw < 1.2 else "med"
                    outline.append(outline.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": size, "len": size}))
                outline.addnext(outline.makeelement(qn("a:effectLst"), {}))  # no theme shadow under the line
            else:
                _, x, y, w, h, path = item
                shapes.add_picture(str(path), Inches(x), top(y + h), Inches(w), Inches(h))
        return deck


# ---- (a) frozen against redrawn ---------------------------------------------
def panel_reads():
    counts = json.loads(CALIB.read_text(encoding="utf-8"))["read_count"]
    ratio = [counts[key]["frozen"]["ZM"] / counts[key]["redrawn"]["ZM"] for key, _ in SOURCES]
    p = Panel("figure3_a", 0.36 * TEXTWIDTH)

    # key: the error that enters at each denoising step as an arrow, and the sum of the ten arrows
    angles = {"frozen": np.zeros(STEPS), "redrawn": np.deg2rad([206, 39, 118, 343, 248, 335, 26, 308, 359, 102])}
    colors = {"frozen": RED, "redrawn": GREY}
    x0, pitch, reach, sum_x, sum_unit = 0.56, 0.108, 0.040, 1.80, 0.052
    p.text(x0 + STEPS * pitch / 2, 1.475, f"error read at each of the {STEPS} steps", size=FS - 0.8, color=MUTED)
    p.text(sum_x + 5 * sum_unit, 1.475, "their sum", size=FS - 0.8, color=MUTED)
    for name, y in (("frozen", 1.325), ("redrawn", 1.165)):
        p.text(0.02, y, name, bold=True, ha="left")
        for s, t in enumerate(angles[name]):
            cx, dx, dy = x0 + (s + 0.5) * pitch, reach * np.cos(t), reach * np.sin(t)
            p.line(cx - dx, y - dy, cx + dx, y + dy, color=colors[name], lw=0.9, arrow=True)
        total = sum_unit * np.array([np.cos(angles[name]).sum(), np.sin(angles[name]).sum()])
        p.line(sum_x, y, sum_x + total[0], y + total[1], color=colors[name], lw=1.3, arrow=True)

    # bars: future error of a frozen error, relative to the redrawn error of the same source
    left, base, right, unit = 0.36, 0.30, p.width - 0.04, 0.70 / 3.75
    p.text(0.075, base + 1.6 * unit, "future error", rot=90)
    for tick in range(4):
        p.line(left - 0.033, base + tick * unit, left, base + tick * unit)
        p.text(left - 0.055, base + tick * unit, str(tick), ha="right")
    p.line(left, base + unit, right, base + unit, color=GREY, lw=0.8)
    p.text(right, base + unit + 0.07, "redrawn", size=FS - 0.8, color=MUTED, ha="right")
    span, bar = (right - left - 0.36) / len(SOURCES), 0.34
    for k, ((_, label), r) in enumerate(zip(SOURCES, ratio)):
        cx = left + (k + 0.5) * span
        p.rect(cx - bar / 2, base, bar, r * unit, fill=RED, round_top=0.028)
        p.text(cx, base + r * unit + 0.085, f"{r:.1f}×", size=FS + 0.8, bold=True)
        p.text(cx, base - 0.145, label)
    p.line(left, base, left, base + 3.6 * unit)
    p.line(left - 0.006, base, right, base)
    p.save()


# ---- (b) where the reader depends on the predicted future ---------------------
def grey(image, lift=0.45):
    """The frame in grey, so that the map on top of it is the only colour."""
    luma = image[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    return np.repeat((lift + (1 - lift) * luma)[..., None], 3, axis=2)


def cells(values, size, gap=0):
    """Token values (rows x columns x channels) as square cells of `size` pixels, `gap` white pixels apart."""
    pixels = np.kron(values, np.ones((size, size, 1)))
    if gap:
        for axis, count in enumerate(values.shape[:2]):
            for k in range(1, count):
                index = [slice(None)] * 3
                index[axis] = slice(k * size - gap // 2, k * size - gap // 2 + gap)
                pixels[tuple(index)] = 1.0
    return pixels


def panel_gaze():
    stability = json.loads(METRIC.read_text(encoding="utf-8"))["stability"]
    maps = json.loads((EXPORT / "maps.json").read_text(encoding="utf-8"))
    # share of tr(M) per token on the first predicted frame, in percent; columns 0..13 third-person, 14..27 wrist
    space = 100 * np.array(maps["metric_frames"][0])
    halves = [100 * np.array(maps[key][0]) for key in ("even_frames", "odd_frames")]
    frame = plt.imread(EXPORT / "frames" / str(CASE) / f"frame{FRAME}.png")
    norm = PowerNorm(0.6, vmin=0, vmax=space.max())  # the colour scale is drawn with the same norm
    cmap = LinearSegmentedColormap.from_list("share", HEAT)
    p = Panel("figure3_b", 0.62 * TEXTWIDTH)

    # the two camera views of the predicted frame; each token is one cell, the more it carries the more opaque
    side, y0, pad, left, label_y = 1.27, 0.05, 0.045, 0.01, 1.435
    half = frame.shape[1] // 2
    for k, name in enumerate(VIEWS):
        x = left + k * (side + pad)
        level = np.asarray(norm(space[:, 14 * k:14 * k + 14]))
        rgba = cmap(level)
        rgba[..., 3] = np.clip(1.9 * level - 0.22, 0, 0.86)
        p.image(x, y0, side, side, grey(frame[:, half * k:half * (k + 1)]), f"b_frame_{k + 1}")
        p.image(x, y0, side, side, cells(rgba, 32), f"b_overlay_{k + 1}")
        p.rect(x, y0, side, side, edge=INK, lw=0.6)
        p.text(x + side / 2, label_y, name)

    # colour scale
    cx, cw = left + 2 * side + pad + 0.075, 0.055
    ramp = cmap(np.linspace(1, 0, 256))[:, None, :3].repeat(8, axis=1)
    p.image(cx, y0, cw, side, ramp, "b_scale")
    p.rect(cx, y0, cw, side, edge=INK, lw=0.6)
    for value, label in ((0, "0"), (0.5, "0.5"), (1, "1"), (2, "2%")):
        y = y0 + side * float(norm(value))
        p.line(cx + cw, y, cx + cw + 0.025, y, lw=0.6)
        p.text(cx + cw + 0.045, y, label, size=FS - 0.8, ha="left")
    p.text(cx + 0.06, label_y, "share", size=FS - 0.8, color=MUTED)

    # the same map estimated on two disjoint halves of the contexts, and how well the two estimates agree
    rx = cx + 0.37
    rw = p.width - rx - 0.01
    mid = rx + rw / 2
    p.text(mid, label_y, "two halves of the contexts", size=FS - 0.8)
    mw, gap, top = 0.84, 0.045, y0 + side
    for h, view in enumerate(halves):
        y = top - (h + 1) * mw / 2 - h * gap
        p.image(mid - mw / 2, y, mw, mw / 2, cells(cmap(norm(view))[..., :3], 24, gap=2), f"b_half_{h + 1}")
        p.line(mid, y, mid, y + mw / 2, color="#FFFFFF", lw=1.2)
        p.rect(mid - mw / 2, y, mw, mw / 2, edge=INK, lw=0.6)
    base = top - mw - gap
    stats = ((stability["diag_corr"], "stable in\nlocation", BLUE), (stability["subspace_overlap"], "not in\ndirection", MUTED))
    for k, (value, label, color) in enumerate(stats):
        x = mid + (k - 0.5) * 0.56
        p.text(x, base - 0.125, f"{value:.2f}", size=FS + 2.6, bold=True, color=color)
        p.text(x, base - 0.30, label, size=FS - 0.8, color=MUTED)
    p.save()


if __name__ == "__main__":
    panel_reads()
    panel_gaze()
