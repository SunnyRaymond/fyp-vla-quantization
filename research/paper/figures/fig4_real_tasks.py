"""Fig. 4: the three AgileX PiPER tasks, key frames of one full demonstration per task.

python research/paper/figures/fig4_real_tasks.py  ->  CVPR2027/figures/figure4.pdf (+ research/paper/figures/figure4_preview.png)
Needs h5py; the raw episodes live in `agilex pepper training data/<task>/data/chunk-000/episode_XXXXXX.hdf5` (30 Hz).
Key frames are placed at the gripper close/open events of the episode, so every frame marks a stage of the task.
Each frame is the third-person view; the inset is the wrist view at the same time step (the two views Fast-WAM reads).
The bar under each row is the whole episode; shaded spans are the steps with the gripper closed.
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
from PIL import Image

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
        labels=["Start", "Grasp apple", "Release in box", "Grasp cube", "Release in box", "Done"],
        keys=lambda s, T: [0, s[0][0] + 2, s[0][1], s[1][0] + 2, s[1][1], T - 1],
        theme=("#F4F0FA", "#CDBDE6", "#6A4C9C"),
    ),
    dict(
        key="cabinet_bowl", ep=19, name="Shelf placement",
        instr="place the bowl to the second floor",
        crop_y0=0,
        labels=["Start", "Grasp bowl", "Lift", "Into the shelf", "Release", "Done"],
        keys=lambda s, T: [0, s[0][0] + 4, s[0][0] + (s[0][1] - s[0][0]) * 27 // 100, (s[0][0] + s[0][1]) // 2, s[0][1] + 6,
                            T - 1],
        theme=("#FDF3E7", "#F2C590", "#C2641A"),
    ),
    dict(
        key="stackboxes", ep=14, name="Box stacking",
        instr="pile the boxes",
        crop_y0=14,
        labels=["Start", "Grasp white box", "Place on base", "Grasp brown box", "Stack on top", "Done"],
        keys=lambda s, T: [0, s[0][0] + 2, s[0][1], s[1][0] + 2, s[1][1], T - 1],
        theme=("#EEF4FB", "#B6CBE7", "#3B6198"),
    ),
]
N_PICKS = {"cluster_grasp": 2, "cabinet_bowl": 1, "stackboxes": 2}

# ---- geometry (inches) ------------------------------------------------------
W = 6.875
OUTER, PAD, GAP_F, GAP_P = 0.0, 0.065, 0.035, 0.055
CROP_H = 213                       # rows kept of the 240-row third-person image (3:2 frames)
HEAD_H, HEAD_GAP, FAN_H, BAR_H = 0.16, 0.032, 0.065, 0.03
INSET_FRAC = 0.32                  # inset width / frame width; bottom-left corner is empty table in all tasks
INK, MUTED, BAR_BG = "#24272C", "#6B7280", "#DADDE2"

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})


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


def upscale(a, k):
    im = Image.fromarray(a)
    return np.asarray(im.resize((im.width * k, im.height * k), Image.LANCZOS))


def rounded(ax, x, y, w, h, r, **kw):
    p = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}", **kw)
    ax.add_patch(p)
    return p


def image(ax, a, x, y, w, h, r):
    im = ax.imshow(a, extent=(x, x + w, y + h, y), interpolation="none", zorder=3)
    clip = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
                          transform=ax.transData, facecolor="none", edgecolor="none")
    im.set_clip_path(clip)
    return im


def legend(ax, x1, yc, acc):
    """Right-aligned key: shaded bar = gripper closed, framed tile = wrist view."""
    fs = 6.5
    t = ax.text(x1, yc, "wrist view", color=MUTED, fontsize=fs, ha="right", va="center")
    ax.figure.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    x = bb.x0 - 0.05 - 0.13
    ax.add_patch(Rectangle((x, yc - 0.045), 0.13, 0.09, facecolor="#9AA3AE", edgecolor="white", lw=0.9, zorder=3))
    t = ax.text(x - 0.14, yc, "gripper closed", color=MUTED, fontsize=fs, ha="right", va="center")
    ax.figure.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    x = bb.x0 - 0.05 - 0.22
    rounded(ax, x, yc - BAR_H / 2, 0.22, BAR_H, BAR_H / 2 - 1e-4, facecolor=BAR_BG, edgecolor="none", zorder=2)
    ax.add_patch(Rectangle((x + 0.06, yc - BAR_H / 2), 0.1, BAR_H, facecolor=acc, alpha=0.55, edgecolor="none",
                           zorder=3))


def draw(episodes, wrist=True):
    n = 6
    inner = W - 2 * OUTER - 2 * PAD
    fw = (inner - (n - 1) * GAP_F) / n
    fh = fw * CROP_H / 320
    ph = PAD + HEAD_H + HEAD_GAP + fh + FAN_H + BAR_H + PAD * 0.75
    H = 2 * OUTER + len(TASKS) * ph + (len(TASKS) - 1) * GAP_P

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for k, (task, d) in enumerate(zip(TASKS, episodes)):
        bg, edge, acc = task["theme"]
        y0 = OUTER + k * (ph + GAP_P)
        rounded(ax, OUTER, y0, W - 2 * OUTER, ph, 0.07, facecolor=bg, edgecolor=edge, linewidth=0.8, zorder=1)

        # header: badge, task name, instruction
        hx, hy = OUTER + PAD, y0 + PAD
        rounded(ax, hx, hy, HEAD_H, HEAD_H, 0.03, facecolor=acc, edgecolor="none", zorder=2)
        ax.text(hx + HEAD_H / 2, hy + HEAD_H / 2 + 0.004, "abc"[k], color="white", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=3)
        t1 = ax.text(hx + HEAD_H + 0.06, hy + HEAD_H / 2, task["name"], color=INK, fontsize=8.5, fontweight="bold",
                     ha="left", va="center", zorder=3)
        fig.canvas.draw()
        bb = t1.get_window_extent().transformed(ax.transData.inverted())
        ax.text(bb.x1 + 0.09, hy + HEAD_H / 2, f"“{task['instr']}”", color=MUTED, fontsize=7.5,
                style="italic", ha="left", va="center", zorder=3)
        if k == 0:
            legend(ax, W - OUTER - PAD, hy + HEAD_H / 2, acc)

        # frames
        fy = hy + HEAD_H + HEAD_GAP
        cx = []
        for j in range(n):
            fx = OUTER + PAD + j * (fw + GAP_F)
            cx.append(fx + fw / 2)
            a = d["fixed"][j][task["crop_y0"]:task["crop_y0"] + CROP_H]
            image(ax, upscale(a, 2), fx, fy, fw, fh, 0.025)
            if wrist:
                iw = fw * INSET_FRAC
                ih = iw * 0.75
                m = 0.022
                ix, iy = fx + m, fy + fh - ih - m
                rounded(ax, ix - 0.008, iy - 0.008, iw + 0.016, ih + 0.016, 0.018, facecolor="white",
                        edgecolor="none", zorder=4)
                im = image(ax, upscale(d["wrist"][j], 2), ix, iy, iw, ih, 0.012)
                im.set_zorder(5)
            # stage chip
            t = ax.text(fx + 0.045, fy + 0.04, task["labels"][j], color="white", fontsize=6, fontweight="bold",
                        ha="left", va="top", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", facecolor=acc, edgecolor="none",
                                  alpha=0.92))

        # timeline: whole episode between the first and last frame centres
        by = fy + fh + FAN_H
        bx0, bx1 = cx[0], cx[-1]
        tx = lambda i: bx0 + (bx1 - bx0) * i / (d["T"] - 1)
        rounded(ax, bx0, by, bx1 - bx0, BAR_H, BAR_H / 2 - 1e-4, facecolor=BAR_BG, edgecolor="none", zorder=2)
        for c, o in d["segs"]:
            ax.add_patch(Rectangle((tx(c), by), tx(o) - tx(c), BAR_H, facecolor=acc, alpha=0.55, edgecolor="none",
                                   zorder=3))
        for j, i in enumerate(d["idx"]):
            ax.plot([cx[j], tx(i)], [fy + fh + 0.012, by - 0.004], color=acc, lw=0.6, alpha=0.7, zorder=2,
                    solid_capstyle="round")
            ax.plot([tx(i)], [by + BAR_H / 2], "o", ms=3.0, mfc=acc, mec="white", mew=0.6, zorder=4)
        ax.text(bx0 - 0.05, by + BAR_H / 2, "0 s", color=MUTED, fontsize=6, ha="right", va="center")
        ax.text(bx1 + 0.05, by + BAR_H / 2, f"{d['T'] / FPS:.1f} s", color=MUTED, fontsize=6, ha="left", va="center")
    return fig


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


def jpeg_images(src, dst, quality=92):
    """matplotlib embeds images losslessly (~10 MB here); re-encode them as JPEG, text and shapes stay vector."""
    import fitz

    doc = fitz.open(src)
    page = doc[0]
    for xref, *_ in page.get_images(full=True):
        page.replace_image(xref, stream=fitz.Pixmap(doc, xref).tobytes("jpeg", jpg_quality=quality))
    doc.save(dst, garbage=4, deflate=True)
    print(f"images re-encoded as JPEG q{quality}: {src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
