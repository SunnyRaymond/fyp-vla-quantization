"""Fig. 3 (Section 3.2): how the predicted future is damaged (a) and how it is read (b).

python research/paper/figures/fig3_observations.py
  -> CVPR2027/figures/figure3_a.pdf, figure3_b.pdf (+ PNG previews next to this script)

Both panels are drawn at print size: the subfigures take 0.36 and 0.62 of \\textwidth (6.875 in) and share one height.
The look follows the conventions of github.com/ChenLiu-1996/figures4papers (left and bottom spines only, solid fills,
values printed on the bars, frameless legends, one accent per panel); no code of that repository is used.

(a) Read counts measured by the LAQ calibration (allocation.json): BF16 weights, the 4-bit rounding error of one source
    with random signs, the sign pattern kept over the denoising steps (frozen) or redrawn at every read. Bars show the
    future error in the reader metric, Z_M, relative to the redrawn error of the same source.
(b) Diagonal of the reader metric M (job 26073871, 80 contexts) summed over the channels of each token of the first
    predicted frame, over that frame as the BF16 model predicts it for one calibration context (decoded by
    experiment/idea-validation/paper-fig3-export-20261112). The two small maps are the same quantity estimated on the
    contexts of even and of odd case id, the split behind the two agreement numbers.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "CVPR2027" / "figures"

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
RED, GREY = "#B64342", "#BBBBC0"            # (a): frozen error, redrawn error (the reference, kept neutral)
BLUE, BLUE_LIGHT = "#0F4D92", "#C5D6EA"     # (b): accent of the reader metric
HEAT = ["#FFF6D6", "#FDCB6E", "#F28C38", "#D8432B", "#8C1020"]  # (b): share of tr(M), light to dark
INK, MUTED = "#222222", "#6E6E73"
TEXTWIDTH, HEIGHT = 6.875, 1.55
FS = 7.2

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def canvas(width):
    """A figure of `width` x HEIGHT inches with one overlay axes in inch coordinates."""
    fig = plt.figure(figsize=(width, HEIGHT))
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, width)
    ov.set_ylim(0, HEIGHT)
    ov.axis("off")
    return fig, ov


def axes_in(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(HERE / f"{name}_preview.png", dpi=400)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def rounded_bar(ax, x, height, width, color, radius=1.6):
    """A bar from the baseline whose two top corners are rounded by `radius` points. Call after the limits are set."""
    box, (fw, fh) = ax.get_position(), ax.figure.get_size_inches()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rx = radius / 72 / (box.width * fw) * (x1 - x0)
    ry = min(radius / 72 / (box.height * fh) * (y1 - y0), height)
    a, b = x - width / 2, x + width / 2
    vertices = [(a, 0), (a, height - ry), (a, height), (a + rx, height), (b - rx, height), (b, height),
                (b, height - ry), (b, 0), (a, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3,
             MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), fc=color, ec="none", zorder=2))


# ---- (a) frozen against redrawn ---------------------------------------------
def panel_reads():
    counts = json.loads(CALIB.read_text(encoding="utf-8"))["read_count"]
    ratio = [counts[key]["frozen"]["ZM"] / counts[key]["redrawn"]["ZM"] for key, _ in SOURCES]
    width = 0.36 * TEXTWIDTH
    fig, ov = canvas(width)

    # key: one arrow per denoising step, the error that enters at that step
    rng = np.random.default_rng(11)
    angles = {"frozen": np.full(STEPS, 38.0), "redrawn": rng.uniform(0, 360, STEPS)}
    notes = {"frozen": "the same error", "redrawn": "a new error"}
    colors = {"frozen": RED, "redrawn": "#8E8E93"}
    x0, pitch, reach = 0.60, 0.118, 0.043
    for name, y in (("frozen", 1.315), ("redrawn", 1.165)):
        ov.text(0.02, y, name, va="center", fontsize=FS, fontweight="bold")
        for s, angle in enumerate(angles[name]):
            cx, t = x0 + (s + 0.5) * pitch, np.deg2rad(angle)
            dx, dy = reach * np.cos(t), reach * np.sin(t)
            ov.add_patch(FancyArrowPatch((cx - dx, y - dy), (cx + dx, y + dy), shrinkA=0, shrinkB=0, lw=0.9,
                                         arrowstyle="-|>,head_length=2.6,head_width=1.5", mutation_scale=1,
                                         color=colors[name], capstyle="round"))
        ov.text(x0 + STEPS * pitch + 0.05, y, notes[name], va="center", fontsize=FS - 0.6, color=MUTED)
    ov.text(x0 + STEPS * pitch / 2, 1.47, f"error at each of the {STEPS} denoising steps", ha="center", va="center",
            fontsize=FS - 0.6, color=MUTED)

    # bars: future error relative to the redrawn error of the same source
    ax = axes_in(fig, 0.36, 0.30, width - 0.40, 0.70)
    centers, half = np.arange(len(SOURCES)), 0.19
    ax.set_xlim(-0.55, len(SOURCES) - 0.45)
    ax.set_ylim(0, 3.75)
    for c, r in zip(centers, ratio):
        rounded_bar(ax, c - half, 1.0, 2 * half - 0.05, GREY)
        rounded_bar(ax, c + half, r, 2 * half - 0.05, RED)
        ax.text(c + half, r + 0.09, f"{r:.1f}$\\times$", ha="center", va="bottom", fontsize=FS + 0.8,
                fontweight="bold")
    ax.set_xticks(centers, [label for _, label in SOURCES])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis="y", pad=1.5)
    ax.set_ylabel("future error\n(redrawn = 1)", labelpad=2.5, linespacing=1.15)
    save(fig, "figure3_a")


# ---- (b) where the reader depends on the predicted future ---------------------
def backdrop(image, lift=0.18):
    """The frame in grey, slightly lifted, so that the map on top of it is the only colour."""
    luma = image[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    return np.repeat((lift + (1 - lift) * luma)[..., None], 3, axis=2)


def upsample(view, size):
    """14 x 14 token values on a size x size grid (bicubic)."""
    scale = float(view.max())
    image = Image.fromarray((view / scale).astype(np.float32), mode="F").resize((size, size), Image.BICUBIC)
    return np.clip(np.asarray(image), 0, 1) * scale


def frame_axes(ax, color=INK, lw=0.6):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set(visible=True, lw=lw, color=color)


def panel_gaze():
    stability = json.loads(METRIC.read_text(encoding="utf-8"))["stability"]
    maps = json.loads((EXPORT / "maps.json").read_text(encoding="utf-8"))
    # share of tr(M) per token on the first predicted frame, in percent; columns 0..13 third-person, 14..27 wrist
    space = 100 * np.array(maps["metric_frames"][0])
    halves = [100 * np.array(maps[key][0]) for key in ("even_frames", "odd_frames")]
    frame = plt.imread(EXPORT / "frames" / str(CASE) / f"frame{FRAME}.png")
    norm = PowerNorm(0.6, vmin=0, vmax=space.max())  # the colour scale is drawn with the same norm
    cmap = LinearSegmentedColormap.from_list("share", HEAT)
    width = 0.62 * TEXTWIDTH
    fig, ov = canvas(width)

    # the two camera views of the predicted frame, with the map on top
    side, y0, pad, left, label_y = 1.27, 0.05, 0.045, 0.01, 1.43
    half = frame.shape[1] // 2
    for k, name in enumerate(VIEWS):
        ax = axes_in(fig, left + k * (side + pad), y0, side, side)
        ax.imshow(backdrop(frame[:, half * k:half * (k + 1)]), interpolation="lanczos")
        level = np.asarray(norm(upsample(space[:, 14 * k:14 * k + 14], half)))
        rgba = cmap(level)
        rgba[..., 3] = np.clip(1.7 * level - 0.24, 0, 0.88)
        ax.imshow(rgba, interpolation="bilinear")
        frame_axes(ax)
        ov.text(left + k * (side + pad) + side / 2, label_y, name, ha="center", va="center", fontsize=FS)

    # colour scale
    cx = left + 2 * side + pad + 0.07
    cax = axes_in(fig, cx, y0, 0.055, side)
    bar = fig.colorbar(ScalarMappable(norm, cmap), cax=cax, ticks=[0, 0.5, 1, 2])
    bar.ax.set_yticklabels(["0", "0.5", "1", "2%"])
    bar.ax.tick_params(pad=1.5, length=1.8, labelsize=FS - 0.8, width=0.6)
    bar.outline.set(lw=0.6, edgecolor=INK)
    ov.text(cx + 0.03, label_y, "share", ha="center", va="center", fontsize=FS - 0.6, color=MUTED)

    # the same map estimated on two disjoint halves of the contexts, and how well the two estimates agree
    rx = cx + 0.36
    rw = width - rx - 0.02
    ov.text(rx + rw / 2, label_y, "two halves of the contexts", ha="center", va="center", fontsize=FS - 0.6)
    mw, gap, top = 0.90, 0.04, y0 + side
    for h, view in enumerate(halves):
        ax = axes_in(fig, rx + (rw - mw) / 2, top - (h + 1) * mw / 2 - h * gap, mw, mw / 2)
        ax.imshow(cmap(norm(view)), interpolation="nearest", aspect="auto")
        ax.plot([13.5, 13.5], [-0.5, 13.5], color="white", lw=0.8)
        frame_axes(ax)
    base = top - mw - gap
    stats = ((stability["diag_corr"], "agreement\nin location"), (stability["subspace_overlap"], "agreement\nin direction"))
    for k, (value, label) in enumerate(stats):
        x = rx + rw / 2 + (k - 0.5) * 0.56
        ov.text(x, base - 0.115, f"{value:.2f}", ha="center", va="center", fontsize=FS + 2.4, fontweight="bold",
                color=BLUE if k == 0 else MUTED)
        ov.text(x, base - 0.285, label, ha="center", va="center", fontsize=FS - 0.9, color=MUTED, linespacing=1.1)
    save(fig, "figure3_b")


if __name__ == "__main__":
    panel_reads()
    panel_gaze()
