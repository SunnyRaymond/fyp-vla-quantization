"""Post hoc on lowrank/tfdiag (fwlt1r1, 26065573): the tf step is shared (T 0.52 on other contexts at fresh noise) but
lowers their rollout future loss by only 0.030 of the ro step's in-batch decrease (G; ro itself 0.021). From the saved
Gram matrices, CPU only:
  sets    G with the step restricted to one group of tensors (video body / video time conditioning / other), the
          10-sample batch-mean direction as in tf_analysis; 'all' must reproduce summary.json's G
  pooled  the direction from the samples of k other batches (both seeds, 20 k samples), measured on the left-out batch at
          seed 0 and 1 (leave-one-batch-out over the 4 batches, all k-subsets), same normalizer as G: how the effect on
          the rollout loss of unseen contexts grows with the samples behind the direction, for tf and ro
argv: gram.pt, summary.json, output json. A synthetic self-check against a direct relative step runs first."""
import itertools
import json
import math
import sys
from pathlib import Path

import torch

GROUPS = ('video', 'video_time', 'action', 'action_time', 'other')
SETS = dict(all=GROUPS, video=('video',), video_time=('video_time',), other=('other',))


def tensor_group(name):
    """lowrank_worker.tensor_group: expert body, expert time conditioning, or other."""
    module = name.rsplit('.', 1)[0]
    head = module.split('.')[0]
    if head not in ('video_expert', 'action_expert'):
        return 'other'
    timed = '.time_embedding' in module or '.time_projection' in module
    return head.split('_')[0] + ('_time' if timed else '')


def decrease(gram, norms, member, s, src, pool, targets, eta):
    """First-order decrease of the ro loss of each target sample under D's relative step along the mean of path src's
    gradients over pool (-eta |P_t| d_t / |d_t| in the tensors with member[t]); gram rows/cols [tf_1..tf_S, ro_1..ro_S]."""
    o = 0 if src == 'tf' else s
    cols = torch.tensor([o + i for i in pool])
    rows = torch.tensor([s + j for j in targets])
    dec = torch.zeros(len(targets), dtype=torch.float64)
    for t in range(gram.shape[0]):
        if not member[t]:
            continue
        g = gram[t]
        gg = float(g[cols][:, cols].mean())
        if gg <= 0:
            continue
        dec += eta * float(norms[t]) / math.sqrt(gg) * g[rows][:, cols].mean(1)
    return dec


def analyse(gram, norms, names, samples, batches, eta):
    gram = gram.double()
    s = len(samples)
    index = {(c, sd): i for i, (c, sd, _) in enumerate(samples)}
    groups = [tensor_group(n) for n in names]
    members = {k: [g in v for g in groups] for k, v in SETS.items()}
    every = members['all']
    seed0 = [[index[(c, 0)] for c in b] for b in batches]
    both = [[index[(c, sd)] for sd in (0, 1) for c in b] for b in batches]
    norm = sum(float(decrease(gram, norms, every, s, 'ro', idx, idx, eta).mean()) for idx in seed0)
    sets = {}
    for key, member in members.items():
        for src in ('tf', 'ro'):
            sums = {}
            for bi, b in enumerate(batches):
                cats = {}
                for k, (c, sd, _) in enumerate(samples):
                    cats.setdefault(('in' if sd == 0 else 'seed') if c in b else ('ctx' if sd == 0 else 'ctx_seed'), []).append(k)
                for cat, ks in cats.items():
                    sums[cat] = sums.get(cat, 0.0) + float(decrease(gram, norms, member, s, src, seed0[bi], ks, eta).mean())
            sets.setdefault(key, {})[src] = {cat: v / norm for cat, v in sums.items()}
    pooled = {}
    for key in ('all', 'video', 'video_time'):
        for src in ('tf', 'ro'):
            rows = {}
            for k in (1, 2, 3):
                acc = {0: 0.0, 1: 0.0}
                for bi in range(len(batches)):
                    others = [j for j in range(len(batches)) if j != bi]
                    combos = list(itertools.combinations(others, k))
                    for sd in (0, 1):
                        targets = [index[(c, sd)] for c in batches[bi]]
                        acc[sd] += sum(float(decrease(gram, norms, members[key], s, src,
                                                      [i for j in combo for i in both[j]], targets, eta).mean())
                                       for combo in combos) / len(combos)
                rows[str(20 * k)] = {f'seed{sd}': v / norm for sd, v in acc.items()}
            pooled.setdefault(key, {})[src] = rows
    energy = {key: float(sum(gram[t, :s, :s].diagonal().mean() for t, m in enumerate(member) if m)
                         / sum(gram[t, :s, :s].diagonal().mean() for t in range(len(names))))
              for key, member in members.items()}
    counts = {key: sum(member) for key, member in members.items()}
    return dict(norm=norm, sets=sets, pooled=pooled, tf_energy=energy, tensors=counts)


def relative_step(params, grads, eta):
    for p, g in zip(params, grads):
        gn = float(g.norm())
        if gn > 0:
            p.sub_(eta * float(p.norm()) / gn * g)


def selfcheck():
    """analyse() against a direct relative step on synthetic per-sample gradients."""
    gen = torch.Generator().manual_seed(3)
    names = ['video_expert.blocks.0.ffn.0.U', 'video_expert.time_projection.1.V', 'proprio_encoder.U']
    sizes = [5, 3, 2]
    cases = list(range(2000, 2012))
    samples = [(c, sd, 'a' if c < 2006 else 'b') for sd in (0, 1) for c in cases]
    batches = [cases[0:3], cases[3:6], cases[6:9], cases[9:12]]
    tf = [[torch.randn(n, generator=gen, dtype=torch.float64) for n in sizes] for _ in samples]
    ro = [[0.4 * tf[k][t] + torch.randn(n, generator=gen, dtype=torch.float64) for t, n in enumerate(sizes)]
          for k in range(len(samples))]
    gram = torch.stack([torch.stack([v[t] for v in tf + ro]) @ torch.stack([v[t] for v in tf + ro]).t()
                        for t in range(len(sizes))])
    P = [torch.randn(n, generator=gen, dtype=torch.float64) for n in sizes]
    norms = torch.tensor([float(p.norm()) for p in P], dtype=torch.float64)
    eta = 0.01
    out = analyse(gram, norms, names, samples, batches, eta)
    index = {(c, sd): i for i, (c, sd, _) in enumerate(samples)}
    paths = dict(tf=tf, ro=ro)

    def direct(src, pool, targets, member):
        d = [sum(paths[src][i][t] for i in pool) / len(pool) if member[t] else torch.zeros(sizes[t], dtype=torch.float64)
             for t in range(len(sizes))]
        moved = [p.clone() for p in P]
        relative_step(moved, d, eta)
        delta = [m - p for m, p in zip(moved, P)]
        return sum(-sum(float(ro[j][t] @ delta[t]) for t in range(len(sizes))) for j in targets) / len(targets)
    every = [True] * 3
    norm = sum(direct('ro', [index[(c, 0)] for c in b], [index[(c, 0)] for c in b], every) for b in batches)
    assert abs(out['norm'] - norm) < 1e-12 * max(1.0, abs(norm))
    timed = [False, True, False]
    want = sum(direct('tf', [index[(c, 0)] for c in b], [k for k, (c, sd, _) in enumerate(samples) if c not in b and sd == 1],
                      timed) for b in batches) / norm
    assert abs(out['sets']['video_time']['tf']['ctx_seed'] - want) < 1e-12, (out['sets']['video_time']['tf'], want)
    acc = 0.0
    for bi, b in enumerate(batches):
        others = [j for j in range(4) if j != bi]
        combos = list(itertools.combinations(others, 2))
        acc += sum(direct('ro', [index[(c, sd)] for j in combo for sd in (0, 1) for c in batches[j]],
                          [index[(c, 1)] for c in b], every) for combo in combos) / len(combos)
    assert abs(out['pooled']['all']['ro']['40']['seed1'] - acc / norm) < 1e-12
    return True


def main():
    selfcheck()
    gram_path, summary_path, out_path = (Path(a) for a in sys.argv[1:4])
    g = torch.load(gram_path, map_location='cpu', weights_only=True)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    samples = [tuple(x) for x in g['samples']]
    batches = [b['cases'] for b in summary['analysis']['batches']]
    eta = summary['tfdiag']['eta']
    out = analyse(g['gram'], g['norms'], g['names'], samples, batches, eta)
    want = summary['analysis']['G']
    worst = max(abs(out['sets']['all'][src][k] - v) for src, row in want.items() for k, v in row.items())
    out.update(selfcheck=True, reproduces_summary_G=worst, eta=eta, batches=batches)
    if worst > 1e-9:
        raise RuntimeError(f'all-tensor G differs from summary.json: {worst}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
