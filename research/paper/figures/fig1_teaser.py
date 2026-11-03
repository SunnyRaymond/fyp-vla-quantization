"""First matplotlib draft of Fig. 1 (teaser): two error paths, LAVQ in one panel, closed-loop success at 2 bits.

The paper now uses CVPR2027/figures/figure1.pdf, exported from figure1_lavq_editable.pptx; this draft is kept for reference.
python research/paper/figures/fig1_teaser.py  ->  research/paper/figures/teaser_draft.pdf (+ .png preview)
Drawn on one canvas of 1000 x 300 units spanning \\textwidth (6.875 in), so 1 unit is about 0.5 pt.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "paper" / "figures" / "teaser_draft"
OBS = ROOT / "experiment/reproduction/fastwam-smoke/artifacts/16180074.pbs101/agentview_image.png"

# ---- data -------------------------------------------------------------------
# §3.2 (a): restoring the predicted future removes 73% (2-bit QTIP) / 85% (3-bit) of the action error, Fast-WAM.
FUTURE_SHARE_W2, FUTURE_SHARE_W3 = 0.73, 0.85
# Appendix table "Bits per weight by module on Fast-WAM", 2-bit rows: (video expert, action expert).
BITS = {"QTIP": (2.01, 2.03), "LAVQ": (2.22, 1.03)}
BPW_TOTAL = 1.99
# Table 1, 2-bit rows (LIBERO-Plus, average success %).
SUCCESS = {
    "Fast-WAM": {"BF16": 77.6, "GPTQ": 5.4, "GPTVQ": 71.7, "QuIP#": 72.5, "QTIP": 73.3, "LAVQ": 74.6},
    "Cosmos-Policy": {"BF16": 83.5, "GPTQ": 0.0, "GPTVQ": 72.6, "QuIP#": 77.3, "QTIP": 79.9, "LAVQ": 80.5},
}
# §1 / §3.2 (c): effective rank of the reader metric.
EFF_RANK, LATENT_DIM = 42, 37632

# ---- style ------------------------------------------------------------------
FP = ("#EEF2F7", "#8A99AD")      # full-precision module (fill, edge)
Q2 = ("#F7DAD6", "#C24A3E")      # 2-bit module
FUT = ("#FBE6CF", "#D9822B")     # predicted future
SWAP = ("#E9E1F3", "#7A5BA6")    # BF16 future swapped in
ERR = "#C24A3E"
INK, MUTED, RULE = "#22262B", "#6B7480", "#D3D7DC"
BASE = ["#D5D9DF", "#B4BBC5", "#8E97A4"]  # GPTVQ, QuIP#, QTIP
LAVQ_C = FUT[1]
HATCH = "//////"

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8, "text.color": INK, "hatch.linewidth": 0.6,
})

W_IN, X0, X1, Y0, Y1 = 6.875, 0, 1000, 36, 330
H_IN = W_IN * (Y1 - Y0) / (X1 - X0)
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X0, X1)
ax.set_ylim(Y0, Y1)
ax.set_aspect("equal")
ax.axis("off")


def frac(x0, y0, x1, y1):
    return [(x0 - X0) / (X1 - X0), (y0 - Y0) / (Y1 - Y0), (x1 - x0) / (X1 - X0), (y1 - y0) / (Y1 - Y0)]


def rbox(x, y, w, h, colors, text="", fs=6, lw=0.7, r=3.5, z=2, hatch=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=colors[0], ec=colors[1], lw=lw, zorder=z, hatch=hatch))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=z + 1, linespacing=1.0)


def arrow(x0, y0, x1, y1, color=MUTED, lw=0.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, color=color, lw=lw,
                                 zorder=1, shrinkA=0, shrinkB=0))


def title(x, text, sub):
    ax.text(x, 321, text, fontsize=7, fontweight="bold", va="center")
    ax.text(x, 308, sub, fontsize=5.6, color=MUTED, va="center")


def vrule(x):
    ax.plot([x, x], [44, 326], color=RULE, lw=0.5, ls=(0, (3, 2)))


# ---- (a) two error paths ------------------------------------------------------
title(2, "(a) 2-bit VQ degrades the predicted future, and the action follows",
      "Fast-WAM quantized with QTIP; action error relative to BF16")
RH = 42
C_OBS, C_GEN, C_FUT, C_READ, C_ERR = 76, 136, 212, 282, 344
SCALE = 150  # units per 1.00 of action error
for x, h, ha in [(C_OBS + RH / 2, "observation", "center"), (C_GEN + 31, "generator", "center"),
                 (C_FUT + 27, "predicted future", "center"), (C_READ + 23, "reader", "center"),
                 (C_ERR, "action error", "left")]:
    ax.text(x, 292, h, fontsize=5.6, color=MUTED, ha=ha, va="center")

obs = plt.imread(OBS)
rows = [("BF16", 260, FP, FUT, "fp"), ("2-bit VQ", 196, Q2, Q2, "q"), ("2-bit VQ,\nBF16 future", 132, Q2, SWAP, "swap")]
for label, yc, mod, fut, kind in rows:
    y = yc - RH / 2
    ax.text(2, yc, label, fontsize=6.2, fontweight="bold", va="center", linespacing=1.05)
    ax.imshow(obs, extent=(C_OBS, C_OBS + RH, y, y + RH), zorder=2, interpolation="lanczos")
    ax.add_patch(Rectangle((C_OBS, y), RH, RH, fill=False, ec=FP[1], lw=0.5, zorder=3))
    arrow(C_OBS + RH + 2, yc, C_GEN - 2, yc)
    rbox(C_GEN, y + 5, 62, RH - 10, mod, "future\ngenerator")
    arrow(C_GEN + 64, yc, C_FUT - 2, yc)
    for k in range(3):
        ax.add_patch(FancyBboxPatch((C_FUT + 9 * k, y + 9 - 4 * k), 32, 28,
                                    boxstyle="round,pad=0,rounding_size=2.5", fc=fut[0], ec=fut[1], lw=0.7,
                                    zorder=3 + k, hatch=HATCH if kind == "q" else None))
    ax.text(C_FUT + 34, y + 15, "ẑ" if kind == "q" else "z", fontsize=6.5, style="italic", color=fut[1],
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc=fut[0], ec="none") if kind == "q" else None)
    arrow(C_FUT + 52, yc, C_READ - 2, yc)
    rbox(C_READ, y + 5, 46, RH - 10, mod, "reader")
    arrow(C_READ + 48, yc, C_ERR - 4, yc)
    if kind == "fp":
        ax.text(C_ERR, yc, "0 (reference)", fontsize=6, color=MUTED, va="center")
    elif kind == "q":
        ax.add_patch(Rectangle((C_ERR, yc - 8), SCALE, 16, fc=ERR, ec="none", alpha=0.85, zorder=3))
        ax.text(C_ERR + SCALE + 4, yc, "1.00", fontsize=6, va="center")
    else:
        rest = 1 - FUTURE_SHARE_W2
        ax.add_patch(Rectangle((C_ERR, yc - 8), SCALE * rest, 16, fc=ERR, ec="none", alpha=0.85, zorder=3))
        ax.add_patch(Rectangle((C_ERR + SCALE * rest, yc - 8), SCALE * FUTURE_SHARE_W2, 16, fc="white",
                               ec=ERR, lw=0.7, ls=(0, (2.5, 1.5)), zorder=3))
        ax.text(C_ERR + SCALE * rest / 2, yc - 15, f"{rest:.2f}", fontsize=6, ha="center", va="center")
        ax.text(C_ERR + SCALE * rest + SCALE * FUTURE_SHARE_W2 / 2, yc, f"−{FUTURE_SHARE_W2:.0%}",
                fontsize=7.5, fontweight="bold", color=FUT[1], ha="center", va="center", zorder=4)

# swap arrow: the BF16 future is copied into the 2-bit run (ROME-style patching)
ax.add_patch(FancyArrowPatch((C_FUT + 6, 246), (C_FUT + 6, 150), connectionstyle="arc3,rad=0.36",
                             arrowstyle="simple,head_length=5,head_width=6,tail_width=2.6",
                             fc=SWAP[1], ec="none", alpha=0.55, zorder=8))
ax.text(C_FUT - 26, 164, "swap in the BF16 future", fontsize=5.8, color=SWAP[1], ha="right", va="center",
        style="italic")

ax.text(2, 90, f"Most of the action error comes through the degraded future "
        f"({FUTURE_SHARE_W2:.0%} at 2 bits, {FUTURE_SHARE_W3:.0%} at 3 bits).",
        fontsize=6.2, color=FUT[1], fontweight="bold", va="center")
lx = 2
for cols, name, h in [(FP, "full-precision module", None), (Q2, "2-bit module", None), (Q2, "degraded future", HATCH),
                      (SWAP, "BF16 future, swapped in", None)]:
    ax.add_patch(FancyBboxPatch((lx, 60), 11, 8, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=cols[0], ec=cols[1], lw=0.6, hatch=h))
    ax.text(lx + 15, 64, name, fontsize=5.6, color=MUTED, va="center")
    lx += 28 + 5.6 * len(name)

vrule(532)

# ---- (b) LAVQ ------------------------------------------------------------------
BX = 542
title(BX, "(b) LAVQ keeps the future", "same bits per weight, no extra storage")
# (1) future error weighted by where the reader looks (schematic tile)
tx, ty, tw, th = BX, 246, 52, 40
gx, gy = np.meshgrid(np.linspace(-1, 1, 120), np.linspace(-1, 1, 96))
blob = np.exp(-(((gx - 0.3) / 0.3) ** 2 + ((gy + 0.15) / 0.3) ** 2))
rgba = np.zeros(blob.shape + (4,))
rgba[..., :3] = matplotlib.colors.to_rgb(ERR)
rgba[..., 3] = 0.7 * blob
ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=2.5", fc=FUT[0], ec=FUT[1],
                            lw=0.7, zorder=2))
ax.imshow(rgba, extent=(tx + 2, tx + tw - 2, ty + 2, ty + th - 2), zorder=3)
ax.text(tx + 5, ty + th - 10, "z", fontsize=6.5, style="italic", color=FUT[1], zorder=4)
ax.text(BX + 60, 276, "① measure future error\n    as the reader sees it", fontsize=6, va="center",
        linespacing=1.15)
ax.text(BX + 60, 252, f"    about {EFF_RANK} of {LATENT_DIM:,} directions", fontsize=5.4, color=MUTED, va="center")

# (2) bits by module, Fast-WAM 2-bit (actual allocation)
ax.text(BX, 224, "② spend bits where they\n    preserve the future", fontsize=6, va="center",
        linespacing=1.15)
bax = fig.add_axes(frac(BX + 46, 138, BX + 168, 198))
ys = np.array([1.0, 0.0])
bh = 0.36
for j, (name, col) in enumerate([("QTIP", BASE[2]), ("LAVQ", LAVQ_C)]):
    vals = BITS[name]
    yy = ys + (0.5 - j) * bh
    bax.barh(yy, vals, bh * 0.9, color=col, label=name)
    for y_, v in zip(yy, vals):
        bax.text(v + 0.05, y_, f"{v:.2f}", va="center", fontsize=5, color=LAVQ_C if name == "LAVQ" else INK,
                 fontweight="bold" if name == "LAVQ" else "normal")
bax.set_yticks(ys, ["generator", "reader"], fontsize=5.6)
bax.set_xlim(0, 2.75)
bax.set_xticks([0, 1, 2])
bax.tick_params(axis="x", labelsize=5, pad=1)
bax.tick_params(axis="y", pad=1.5, length=0)
bax.spines[["top", "right"]].set_visible(False)
bax.legend(fontsize=5, frameon=False, loc="center left", handlelength=0.8, handletextpad=0.3, borderaxespad=0.0,
           bbox_to_anchor=(1.0, 0.5))
ax.text(BX, 114, f"    bits per weight; {BPW_TOTAL} total, Fast-WAM", fontsize=5.4, color=MUTED, va="center")
ax.text(BX, 92, "③ trellis-coded initialization", fontsize=6, va="center")
ax.text(BX, 76, "④ calibrate the shared decode tables", fontsize=6, va="center")
ax.text(BX, 63, "    toward action and future fidelity", fontsize=6, va="center")

vrule(758)

# ---- (c) closed-loop success -----------------------------------------------------
CX = 768
title(CX, "(c) Closed-loop success at 2 bits", "LIBERO-Plus, average success rate (%)")
cax = fig.add_axes(frac(CX + 20, 82, 998, 290))
methods = ["GPTVQ", "QuIP#", "QTIP", "LAVQ"]
colors = BASE + [LAVQ_C]
lo, hi = 65, 87
bw = 0.2
for g, (model, res) in enumerate(SUCCESS.items()):
    for k, (m, c) in enumerate(zip(methods, colors)):
        x = g + (k - 1.5) * bw
        cax.bar(x, res[m] - lo, bw * 0.88, bottom=lo, color=c, label=m if g == 0 else None, zorder=2)
        if m in ("QTIP", "LAVQ"):
            cax.text(x, res[m] + 0.3, f"{res[m]:.1f}", ha="center", va="bottom", fontsize=5,
                     fontweight="bold" if m == "LAVQ" else "normal", color=LAVQ_C if m == "LAVQ" else INK)
    cax.plot([g - 2.2 * bw, g + 2.2 * bw], [res["BF16"]] * 2, color=INK, lw=0.7, ls=(0, (3, 1.6)), zorder=3)
    cax.text(g + 2.2 * bw, res["BF16"] + 0.4, f"BF16 {res['BF16']:.1f}", ha="right", va="bottom", fontsize=5)
cax.set_xticks([0, 1], list(SUCCESS), fontsize=5.8)
cax.tick_params(axis="x", length=0, pad=2)
cax.set_ylim(lo, hi)
cax.set_yticks([65, 70, 75, 80, 85])
cax.tick_params(axis="y", labelsize=5, pad=1)
cax.spines[["top", "right"]].set_visible(False)
cax.grid(axis="y", color="#ECEEF1", lw=0.5, zorder=0)
cax.set_xlim(-0.5, 1.5)
cax.legend(fontsize=5, frameon=False, ncol=2, loc="upper left", handlelength=0.8, handletextpad=0.3,
           columnspacing=0.6, borderaxespad=0.1)
ax.text(CX + 20, 60, f"GPTQ (scalar): {SUCCESS['Fast-WAM']['GPTQ']:.1f} / "
        f"{SUCCESS['Cosmos-Policy']['GPTQ']:.1f}, below the axis", fontsize=5.4, color=MUTED, va="center")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("wrote", OUT.with_suffix(".pdf"), f"{W_IN:.3f} x {H_IN:.3f} in")
