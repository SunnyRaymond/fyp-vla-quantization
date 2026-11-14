"""Supplementary figure: the reader metric of single contexts.

python research/paper/figures/fig_supp_gaze.py
  -> CVPR2027/figures/supp_gaze_maps.pdf, supp_gaze_maps_preview.png (next to this script)

For eight calibration contexts, two per LIBERO suite, the share of the reader metric M carried by each token of the
first predicted frame, estimated from the four probes of that context alone (job 26073871), over the frame that the BF16
model predicts for it (experiment/idea-validation/paper-fig3-export-20261112). Same colours as Fig. 3(b).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXPORT = ROOT / "experiment/idea-validation/paper-fig3-export-20261112/pulled/results"
OUT = ROOT / "CVPR2027" / "figures"

CASES = [2000, 2016, 2026, 2037, 2042, 2053, 2063, 2074]
SUITES = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
FRAME = 4
HEAT = ["#FFF6D6", "#FDCB6E", "#F28C38", "#D8432B", "#8C1020"]
INK = "#222222"
FS = 7.4

plt.rcParams.update({"font.family": ["Cambria", "DejaVu Sans"], "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42})


def grey(image, lift=0.45):
    luma = image[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    return np.repeat((lift + (1 - lift) * luma)[..., None], 3, axis=2)


def main():
    maps = json.loads((EXPORT / "maps.json").read_text(encoding="utf-8"))
    index = {c["case"]: c for c in json.loads((EXPORT / "frames" / "index.json").read_text(encoding="utf-8"))["cases"]}
    cmap = LinearSegmentedColormap.from_list("share", HEAT)
    width = 6.875
    cols, rows = 4, 2
    gap, top = 0.07, 0.16
    cell_w = (width - (cols - 1) * gap) / cols
    cell_h = cell_w / 2
    height = rows * (cell_h + top) + (rows - 1) * 0.03
    fig = plt.figure(figsize=(width, height))
    for k, case in enumerate(CASES):
        r, c = divmod(k, cols)
        x = c * (cell_w + gap)
        y = height - (r + 1) * (cell_h + top) - r * 0.03
        ax = fig.add_axes([x / width, y / height, cell_w / width, cell_h / height])
        frame = plt.imread(EXPORT / "frames" / str(case) / f"frame{FRAME}.png")
        share = 100 * np.array(maps["per_context"][str(case)][0])
        share = share / np.array(maps["per_context"][str(case)]).sum()  # share of the trace of this context
        norm = PowerNorm(0.6, vmin=0, vmax=share.max())
        level = np.asarray(norm(share))
        rgba = cmap(level)
        rgba[..., 3] = np.clip(1.9 * level - 0.22, 0, 0.86)
        ax.imshow(grey(frame), extent=(0, 2, 0, 1), aspect="auto")
        ax.imshow(rgba, extent=(0, 2, 0, 1), aspect="auto", interpolation="nearest")
        ax.plot([1, 1], [0, 1], color="white", lw=1.0)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
            s.set_color(INK)
        task = index[case]["task"]
        fig.text((x + cell_w / 2) / width, (y + cell_h + 0.045) / height,
                 f"{SUITES[index[case]['suite']]}: {task if len(task) < 36 else task[:33] + '...'}",
                 ha="center", va="bottom", fontsize=FS - 2.2, color=INK)
    fig.savefig(OUT / "supp_gaze_maps.pdf")
    fig.savefig(HERE / "supp_gaze_maps_preview.png", dpi=300)


if __name__ == "__main__":
    main()
