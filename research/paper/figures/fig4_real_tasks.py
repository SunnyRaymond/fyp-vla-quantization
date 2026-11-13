"""Fig. 4: the three AgileX PiPER tasks, key frames of one full demonstration per task.

python research/paper/figures/fig4_real_tasks.py  ->  CVPR2027/figures/figure4.pdf (+ research/paper/figures/figure4_preview.png)
Needs h5py and PyMuPDF; the raw episodes live in `agilex pepper training data/<task>/data/chunk-000/episode_XXXXXX.hdf5` (30 Hz).
Key frames are placed at the gripper close/open events of the episode, so every frame marks a stage of the task.
Each frame is the third-person view; the inset is the wrist view at the same time step (the two views Fast-WAM reads).
Under every frame a progress bar spans the whole episode: orange where the gripper is closed, the dot at the frame's time,
the part before the dot drawn darker.
Canvas is in inches, 6.875 in wide (= \\textwidth), used as figure* with width=\\textwidth.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "agilex pepper training data"
OUT_PDF = ROOT / "CVPR2027" / "figures" / "figure4.pdf"
OUT_PNG = ROOT / "research" / "paper" / "figures" / "figure4_preview.png"
FPS = 30.0

# ---- tasks ------------------------------------------------------------------
# keys(segs, T): frame indices from the grasp segments [(close, open), ...] of the episode.
TASKS = [
    dict(
        key="cluster_grasp", ep=73, name="Pick-and-place",
        instr="place the red apple and green cube to the box",
        crop_y0=12,
        labels=["Start", "Grasp apple", "Apple in box", "Grasp cube", "Cube in box", "Done"],
        keys=lambda s, T: [0, s[0][0] + 2, s[0][1], s[1][0] + 2, s[1][1], T - 1],
    ),
    dict(
        key="cabinet_bowl", ep=19, name="Shelf placement",
        instr="place the bowl to the second floor",
        crop_y0=0,
        labels=["Start", "Grasp bowl", "Lift", "Into shelf", "Release", "Done"],
        keys=lambda s, T: [0, s[0][0] + 4, s[0][0] + (s[0][1] - s[0][0]) * 27 // 100, (s[0][0] + s[0][1]) // 2,
                            s[0][1] + 6, T - 1],
    ),
    dict(
        key="stackboxes", ep=14, name="Box stacking",
        instr="pile the boxes",
        crop_y0=14,
        labels=["Start", "Grasp white box", "On base", "Grasp brown box", "On top", "Done"],
        keys=lambda s, T: [0, s[0][0] + 2, s[0][1], s[1][0] + 2, s[1][1], T - 1],
    ),
]
N_PICKS = {"cluster_grasp": 2, "cabinet_bowl": 1, "stackboxes": 2}

# ---- geometry (inches) and style ----------------------------------------------
W = 6.875
N = 6
GAP_F = 0.028                      # between frames
GAP_ROW = 0.1                      # between task rows
CROP_H = 213                       # rows kept of the 240-row third-person image (3:2 frames)
HEAD_H, HEAD_GAP = 0.15, 0.04      # task title line, gap to the frames
BAR_GAP, BAR_H = 0.034, 0.036      # progress bar under each frame
BAR_IN, PAD_B = 0.04, 0.03         # bar inset from the frame edges (keeps the end dots whole), bottom margin
R_IMG = 0.03                       # corner radius of the photos
INSET_FRAC = 0.31                  # wrist inset width / frame width; bottom-left corner is empty table in all tasks
TONE_FIXED, TONE_WRIST = (210.0, 0.85), (255.0, 0.88)   # (white point, gamma) per camera, see photo()

INK, MUTED = "#1F2329", "#5F6670"
BADGE = "#2E3A4B"
TRACK, TRACK_DONE = "#E2E5E9", "#A9B1BB"
GRIP, GRIP_FUTURE = "#EE7D1F", "#F7C392"
FS_TITLE, FS_TEXT, FS_CHIP = 8.5, 7.5, 6.4

plt.rcParams.update({"font.family": "Cambria", "pdf.fonttype": 42, "ps.fonttype": 42})


def grasp_segments(g):
    """(close, open) index pairs of gripper-closed spans longer than 0.5 s; end-of-episode flicker is ignored."""
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    closed = g < lo + 0.5 * (hi - lo)
    segs, i, T = [], 0, len(g)
    while i < T:
        if not closed[i]:
            i += 1
            continue
        j = i
        while j < T and closed[j]:
            j += 1
        if j < T and j - i > 15:
            segs.append((i, j))
        i = j
    return segs


def load(task, ep):
    path = DATA / task["key"] / "data" / "chunk-000" / f"episode_{ep:06d}.hdf5"
    with h5py.File(path, "r") as f:
        g = f["observation/state"][:, 6]
        T = len(g)
        segs = grasp_segments(g)
        assert len(segs) == N_PICKS[task["key"]], (task["key"], ep, segs)
        idx = task["keys"](segs, T)
        assert idx == sorted(idx) and idx[-1] < T, idx
        fixed = [f["observation/images/realsense_fixed"][i] for i in idx]
        wrist = [f["observation/images/realsense_wrist"][i] for i in idx]
        instr = f.attrs["instruction"]
    return dict(T=T, segs=segs, idx=idx, fixed=fixed, wrist=wrist, instr=instr)


def photo(a, white=255.0, gamma=1.0, k=2):
    """Same global tone curve for every frame of a camera, then 2x Lanczos.

    The third-person camera tops out at ~200 (the white table reads as grey, median ~140), so its frames get a levels
    stretch (white point 210) and a gamma of 0.85; the wrist camera already spans 0-255 and only gets gamma 0.88.
    """
    x = np.clip(a.astype(np.float32) / white, 0, 1) ** gamma
    im = Image.fromarray((x * 255 + 0.5).astype(np.uint8))
    im = ImageEnhance.Contrast(im).enhance(1.05)
    im = ImageEnhance.Color(im).enhance(1.12)
    im = im.resize((im.width * k, im.height * k), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.4, percent=45, threshold=2))
    return np.asarray(im)


def rounded(ax, x, y, w, h, r, **kw):
    p = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}", **kw)
    ax.add_patch(p)
    return p


def image(ax, a, x, y, w, h, r, z=3):
    im = ax.imshow(a, extent=(x, x + w, y + h, y), interpolation="none", zorder=z)
    im.set_clip_path(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
                                    transform=ax.transData, facecolor="none", edgecolor="none"))
    return im


def text_right(ax, t):
    ax.figure.canvas.draw()
    return t.get_window_extent().transformed(ax.transData.inverted()).x1


def progress(ax, x0, x1, y, T, segs, i, h=BAR_H):
    """Episode progress bar: track, done part, gripper-closed spans (darker once passed), playhead at step i."""
    tx = lambda s: x0 + (x1 - x0) * s / (T - 1)
    r = h / 2 - 1e-4
    rounded(ax, x0, y, x1 - x0, h, r, facecolor=TRACK, edgecolor="none", zorder=2)
    if i > 0:
        rounded(ax, x0, y, tx(i) - x0, h, min(r, (tx(i) - x0) / 2 - 1e-4), facecolor=TRACK_DONE, edgecolor="none",
                zorder=3)
    for c, o in segs:
        if c < i:
            ax.add_patch(Rectangle((tx(c), y), tx(min(o, i)) - tx(c), h, facecolor=GRIP, edgecolor="none", zorder=4))
        if o > i:
            ax.add_patch(Rectangle((tx(max(c, i)), y), tx(o) - tx(max(c, i)), h, facecolor=GRIP_FUTURE,
                                   edgecolor="none", zorder=4))
    ax.plot([tx(i)], [y + h / 2], "o", ms=4.6, mfc="white", mec=BADGE, mew=1.0, zorder=6)


def legend(ax, x1, yc):
    """Right-aligned key on the first title line."""
    items = []
    t = ax.text(x1, yc, "wrist view", color=MUTED, fontsize=FS_TEXT, ha="right", va="center")
    ax.figure.canvas.draw()
    x = t.get_window_extent().transformed(ax.transData.inverted()).x0 - 0.05
    ax.add_patch(Rectangle((x - 0.15, yc - 0.05), 0.15, 0.1, facecolor="#8E98A4", edgecolor="white", lw=1.0, zorder=3))
    ax.add_patch(Rectangle((x - 0.15, yc - 0.05), 0.15, 0.1, facecolor="none", edgecolor="#8E98A4", lw=0.4, zorder=4))
    x -= 0.15 + 0.16
    t = ax.text(x, yc, "gripper closed", color=MUTED, fontsize=FS_TEXT, ha="right", va="center")
    ax.figure.canvas.draw()
    x = t.get_window_extent().transformed(ax.transData.inverted()).x0 - 0.05
    progress(ax, x - 0.4, x - 0.04, yc - BAR_H / 2, 100, [(30, 75)], 55)
    return items


def draw(episodes, wrist=True):
    fw = (W - (N - 1) * GAP_F) / N
    fh = fw * CROP_H / 320
    rh = HEAD_H + HEAD_GAP + fh + BAR_GAP + BAR_H
    H = len(TASKS) * rh + (len(TASKS) - 1) * GAP_ROW + PAD_B

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for k, (task, d) in enumerate(zip(TASKS, episodes)):
        y0 = k * (rh + GAP_ROW)
        yc = y0 + HEAD_H / 2
        # title line: badge, task name, instruction
        b = HEAD_H * 0.92
        rounded(ax, 0, yc - b / 2, b, b, 0.028, facecolor=BADGE, edgecolor="none", zorder=2)
        ax.text(b / 2, yc + 0.004, "abc"[k], color="white", fontsize=FS_TEXT + 0.5, fontweight="bold", ha="center",
                va="center", zorder=3)
        t = ax.text(b + 0.06, yc, task["name"], color=INK, fontsize=FS_TITLE, fontweight="bold", ha="left",
                    va="center")
        ax.text(text_right(ax, t) + 0.08, yc, f"“{task['instr']}”", color=MUTED, fontsize=FS_TEXT,
                style="italic", ha="left", va="center")
        if k == 0:
            legend(ax, W, yc)

        fy = y0 + HEAD_H + HEAD_GAP
        for j in range(N):
            fx = j * (fw + GAP_F)
            a = d["fixed"][j][task["crop_y0"]:task["crop_y0"] + CROP_H]
            image(ax, photo(a, *TONE_FIXED), fx, fy, fw, fh, R_IMG)
            if wrist:
                iw, m = fw * INSET_FRAC, 0.035
                ih = iw * 0.75
                ix, iy = fx + m, fy + fh - ih - m
                rounded(ax, ix - 0.012, iy - 0.012, iw + 0.024, ih + 0.024, 0.022, facecolor="white",
                        edgecolor="none", zorder=4)
                image(ax, photo(d["wrist"][j], *TONE_WRIST), ix, iy, iw, ih, 0.012, z=5)
            ax.text(fx + 0.05, fy + 0.048, task["labels"][j], color="white", fontsize=FS_CHIP, fontweight="bold",
                    ha="left", va="top", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.45", facecolor="#14181D", edgecolor="none",
                              alpha=0.62))
            progress(ax, fx + BAR_IN, fx + fw - BAR_IN, fy + fh + BAR_GAP, d["T"], d["segs"], d["idx"][j])
    return fig


def jpeg_images(src, dst, quality=92):
    """matplotlib embeds images losslessly (~8 MB here); re-encode them as JPEG, text and shapes stay vector."""
    import fitz

    doc = fitz.open(src)
    page = doc[0]
    for xref, *_ in page.get_images(full=True):
        page.replace_image(xref, stream=fitz.Pixmap(doc, xref).tobytes("jpeg", jpg_quality=quality))
    doc.save(dst, garbage=4, deflate=True)
    print(f"images re-encoded as JPEG q{quality}: {src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs=3, help="episode ids in TASKS order")
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--pdf", default=str(OUT_PDF))
    ap.add_argument("--png", default=str(OUT_PNG))
    args = ap.parse_args()
    eps = args.eps or [t["ep"] for t in TASKS]
    data = [load(t, e) for t, e in zip(TASKS, eps)]
    for t, e, d in zip(TASKS, eps, data):
        assert d["instr"] == t["instr"], (d["instr"], t["instr"])
        print(json.dumps(dict(task=t["key"], ep=e, T=d["T"], segs=d["segs"], frames=d["idx"])))
    fig = draw(data, wrist=not args.no_wrist)
    fig.savefig(args.png, dpi=200)
    tmp = Path(args.pdf).with_suffix(".flate.pdf")
    fig.savefig(tmp)
    jpeg_images(tmp, Path(args.pdf))
    tmp.unlink()
    print("wrote", args.pdf, args.png)


if __name__ == "__main__":
    main()
