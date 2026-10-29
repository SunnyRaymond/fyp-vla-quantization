"""CPU check: does the probe row weight d_c = sum_t g_tc^2 |x_t|^2 vary enough across output rows for a per-row scale
folded into QTIP's SV (Wr = Had^T diag(SV) W diag(SU) Had, SV fp32) to pay?

Model: after the two-sided RHT the rounding error is iid across rows with variance proportional to the RMS of the
scaled weight, so a row scale s gives guided loss ~ (sum_c d_c / s_c^2) (sum_c s_c^2 |W_c|^2). Optimum (Cauchy-Schwarz):
s_c^2 = sqrt(d_c) / |W_c|; loss ratio against s = 1 is (sum sqrt(d_c)|W_c|)^2 / (sum d_c * sum |W_c|^2).
Out of sample: s from contexts 2000-2039, loss on 2040-2079 (and the reverse), so probe noise in d does not count as gain.
Layers are weighted by their loss under isotropic relative noise, sum_c d_c * sum_c |W_c|^2 / (out * in).
argv: guided results root, baseline results root, output json.
"""
import json
from pathlib import Path
import sys
import time

import torch

G, B, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
HALVES = (range(2000, 2040), range(2040, 2080))


def ratio(d, s2, w2):
    return float((d / s2).sum() * (s2 * w2).sum() / (d.sum() * w2.sum()))


def main():
    begin = time.monotonic()
    rows = json.loads((B / 'prep' / 'modules.json').read_text())
    d = [[torch.zeros(r['shape'][0], dtype=torch.float64) for r in rows] for _ in HALVES]
    for h, cases in enumerate(HALVES):
        for case in cases:
            record = torch.load(G / 'stats' / f'{case}.pt', map_location='cpu', mmap=True, weights_only=True)
            for idx, v in record['d'].items():
                d[h][int(idx)] += v.double()
        print(f'half {h} loaded {time.monotonic() - begin:.0f}s', flush=True)
    layers = []
    for r in rows:
        i = r['index']
        w = torch.load(B / 'prep' / 'masters' / f'{i:04d}.pt', map_location='cpu', mmap=True, weights_only=True)
        if w['module'] != r['module']:
            raise RuntimeError(f'master identity differs at {i}')
        w2 = w['weight'].double().square().sum(1)
        d1, d2 = d[0][i], d[1][i]
        both = d1 + d2
        if float(both.sum()) <= 0:
            layers.append(dict(index=i, module=r['module'], loss=0.0, skipped='no probe signal'))
            continue
        floor = 1e-12 * float(both.mean())

        def scale(dv):
            return (dv.clamp_min(floor).sqrt() / w2.clamp_min(1e-30).sqrt()).clamp_min(1e-30)
        layers.append(dict(
            index=i, module=r['module'], rows=int(both.numel()),
            loss=float(both.sum() * w2.sum()) / (r['shape'][0] * r['shape'][1]),
            d_cv=float(both.std() / both.mean()), d_max_over_mean=float(both.max() / both.mean()),
            half_corr=float(torch.corrcoef(torch.stack((d1, d2)))[0, 1]),
            ratio_in_sample=ratio(both, scale(both), w2),
            ratio_split=0.5 * (ratio(d2, scale(d1), w2) + ratio(d1, scale(d2), w2)),
            ratio_split_d_only=0.5 * (ratio(d2, d1.clamp_min(floor).sqrt(), w2) + ratio(d1, d2.clamp_min(floor).sqrt(), w2))))
    used = [l for l in layers if l['loss'] > 0]
    total = sum(l['loss'] for l in used)

    def weighted(key, subset=None):
        s = [l for l in used if subset is None or subset(l['module'])]
        z = sum(l['loss'] for l in s)
        return sum(l[key] * l['loss'] for l in s) / z if z else None
    streams = dict(video=lambda m: m.startswith('video_expert'), action=lambda m: m.startswith('action_expert'),
                   other=lambda m: not m.startswith(('video_expert', 'action_expert')))
    summary = dict(
        layers=len(used), seconds=time.monotonic() - begin,
        loss_ratio=dict(in_sample=weighted('ratio_in_sample'), split=weighted('ratio_split'),
                        split_d_only=weighted('ratio_split_d_only')),
        by_stream={k: dict(loss_share=sum(l['loss'] for l in used if f(l['module'])) / total,
                           split=weighted('ratio_split', f), in_sample=weighted('ratio_in_sample', f)) for k, f in streams.items()},
        half_corr_weighted=weighted('half_corr'), d_cv_weighted=weighted('d_cv'),
        best_layers=sorted(({k: l[k] for k in ('module', 'ratio_split', 'loss')} for l in used),
                           key=lambda l: l['ratio_split'])[:10],
        top_loss_layers=sorted(({k: l[k] for k in ('module', 'ratio_split', 'loss')} for l in used),
                               key=lambda l: -l['loss'])[:10])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(summary=summary, layers=layers), indent=1))
    print('ROWCHECK ' + json.dumps(summary['loss_ratio']), flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
