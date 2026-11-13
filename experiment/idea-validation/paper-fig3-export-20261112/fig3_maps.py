"""Reader-metric maps for Fig. 3(b), from the probes of job 26073871 (fastwam-guided-hessian-20261028). CPU only.

  fig3_maps.py <out.json>

Every map is the share of tr(M) per token, [frame, 14, 28] over the two predicted frames (columns 0..13 are the
third-person view, 14..27 the wrist view):
  metric_frames   the diagonal stored in metric.pt (what the paper's M uses)
  mean_frames     mean over the 80 contexts of the per-context diagonal, each context at unit trace
  even_frames / odd_frames   the same over the contexts of even / odd case id (the split of the stability numbers)
  per_context     one map per context (4 probes each)
checks: agreement with summary.json, and the split-half correlation of the full diagonal (0.863 in summary.json).
"""
import json
from pathlib import Path
import sys

import torch

SRC = Path("/scratch/users/ntu/yguo017/fastwam-guided-hessian-20261028/results/ravqc/metric")


def pearson(a, b):
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    out = Path(sys.argv[1])
    summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8"))
    metric = torch.load(SRC / "metric.pt", map_location="cpu", weights_only=True)
    shape = tuple(metric["shape"][1:])  # (48, 2, 14, 28)
    diag = metric["diag"].double().reshape(shape)
    metric_frames = diag.sum(0) / diag.sum()

    cases = sorted(int(p.stem) for p in (SRC / "ctx").glob("*.pt"))
    full = {}
    for case in cases:
        row = torch.load(SRC / "ctx" / f"{case}.pt", map_location="cpu", weights_only=True)
        d = row["probes"].double().square().mean(0)
        full[case] = (d / d.sum()).reshape(shape)
    mean = torch.stack([full[c] for c in cases]).mean(0)
    half = [torch.stack([full[c] for c in cases if c % 2 == k]).mean(0) for k in (0, 1)]

    space = torch.tensor(summary["gaze"]["space"], dtype=torch.float64)
    tokens = [h.sum(0) for h in half]  # [frame, 14, 28]
    checks = dict(
        contexts=len(cases),
        space_max_abs=float((metric_frames.sum(0) - space).abs().max()),
        mean_vs_metric_corr=pearson(mean.sum(0), metric_frames),
        halves_diag_corr=pearson(half[0], half[1]),
        halves_token_corr=pearson(tokens[0], tokens[1]),
        halves_token_corr_frame1=pearson(tokens[0][0], tokens[1][0]),
        summary_diag_corr=summary["stability"]["diag_corr"],
        frame_share=[float(x) for x in metric_frames.sum((1, 2))],
    )

    def rounded(t):
        return [[[round(float(v), 8) for v in r] for r in f] for f in t]

    value = dict(source=str(SRC), shape=list(shape), cases=cases, checks=checks,
                 metric_frames=rounded(metric_frames), mean_frames=rounded(mean.sum(0)),
                 even_frames=rounded(tokens[0]), odd_frames=rounded(tokens[1]),
                 per_context={str(c): rounded(full[c].sum(0)) for c in cases})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value) + "\n", encoding="utf-8")
    print("MAPS", json.dumps(checks))


if __name__ == "__main__":
    main()
