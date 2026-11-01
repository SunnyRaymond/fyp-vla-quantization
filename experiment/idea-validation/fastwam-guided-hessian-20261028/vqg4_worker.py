"""G4 (vqg4.json): do gradients of the C1 objectives on shared VQ parameters carry over to other contexts at fresh noise?

GPTVQ W2 (A16 baseline). Codes are recovered from the stored dense BF16 weights (distinct 4-vectors per codebook block).
Every non-time-conditioning target Linear runs through VQGrad, whose backward reduces G = g^T x onto six families of
zero-valued parameters: per-entry codebook offsets (book), row / column log-scales, output offsets; the layer- and
module-tied codebook families are sums of the book gradients over meta-entries (k-means in span-normalized coordinates).
Per (context, seed) sample the gradients of lambda Z, beta K (through the student query) and A_tf (student reading the
teacher future), then per-tensor Gram matrices and fwld1's first-order transfer ratios T -> results/vqg4/GPTVQ/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import lowrank_worker as lw
import lp_worker as lp
import worker as wk

HERE = Path(__file__).resolve().parent
G4 = json.loads((HERE / 'vqg4.json').read_text(encoding='utf-8'))
DEVICE = lw.DEVICE
PATHS = ('Z', 'K', 'A')
FAMILIES = ('book', 'layer', 'module', 'row', 'col', 'bias')
BOOK_FAMILIES = ('book', 'layer', 'module')
ENTRIES = 256
VQ_DIM = 4


def timed(name):
    return '.time_embedding' in name or '.time_projection' in name


def module_of(name):
    head = name.split('.')[0]
    return {'video_expert': 'video', 'action_expert': 'action'}.get(head, 'other')


class Layer:
    """Recovered GPTVQ codes of one Linear: codes [out, in_vec] uint8, entries [books, 256, 4], usage [books, 256],
    span scale per book (max |entry| over the books of its column span)."""

    def __init__(self, weight, bias, rows, spans, in_pad, codes, entries, usage):
        self.weight, self.bias, self.rows, self.spans, self.in_pad = weight, bias, rows, spans, in_pad
        self.codes, self.entries, self.usage = codes, entries, usage
        self.books = entries.shape[0]
        groups = self.books // spans
        span_max = entries.abs().amax(dim=(1, 2)).view(groups, spans).amax(0)
        self.scale = span_max.repeat(groups).clamp_min(1e-30)  # book b = group * spans + span
        self.xsq = torch.zeros((), device=weight.device, dtype=torch.float64)
        self.absgw = torch.zeros((), device=weight.device, dtype=torch.float64)  # sum |G o W| since the last collect
        self.tokens = 0

    def book_index(self):
        out, in_vec = self.codes.shape
        r = torch.arange(out, device=self.codes.device) // self.rows
        s = torch.arange(in_vec, device=self.codes.device) * VQ_DIM // 256
        return r[:, None] * self.spans + s[None, :]

    def entry_index(self):
        return (self.book_index() * ENTRIES + self.codes.long()).reshape(-1)


def recover(weight, config, limit=ENTRIES):
    """Layer codes from a dense GPTVQ weight: per block of rows_per_group rows x 256 columns, the distinct BF16
    4-vectors (columns zero-padded to in_pad) are the entries, their index per vector the code."""
    out, n = weight.shape
    rows, spans, in_pad = config['rows_per_group'], config['column_spans'], -(-n // VQ_DIM) * VQ_DIM  # geometry()'s in_pad
    if config['vq_dim'] != VQ_DIM or config['columns_per_group'] != 256:
        raise ValueError(f'Unexpected GPTVQ geometry: {config}')
    w = F.pad(weight, (0, in_pad - n)) if in_pad > n else weight
    ints = {2: torch.int16, 4: torch.int32}[w.element_size()]
    keys = w.contiguous().view(ints).view(out, in_pad // VQ_DIM, VQ_DIM)  # exact bit patterns
    vectors = w.float().view(out, in_pad // VQ_DIM, VQ_DIM)
    groups = -(-out // rows)
    codes = torch.empty(out, in_pad // VQ_DIM, dtype=torch.uint8, device=weight.device)
    entries = torch.zeros(groups * spans, ENTRIES, VQ_DIM, device=weight.device)
    usage = torch.zeros(groups * spans, ENTRIES, device=weight.device)
    worst, step = 0, 256 // VQ_DIM
    for g in range(groups):
        r0, r1 = g * rows, min((g + 1) * rows, out)
        for s in range(spans):
            c0, c1 = s * step, min((s + 1) * step, in_pad // VQ_DIM)
            k = keys[r0:r1, c0:c1].reshape(-1, VQ_DIM)
            uniq, inv = torch.unique(k, dim=0, return_inverse=True)
            worst = max(worst, uniq.shape[0])
            if uniq.shape[0] > limit:
                return None, None, None, worst
            b = g * spans + s
            codes[r0:r1, c0:c1] = inv.view(r1 - r0, c1 - c0).to(torch.uint8)
            entries[b].index_copy_(0, inv, vectors[r0:r1, c0:c1].reshape(-1, VQ_DIM))
            usage[b] = torch.bincount(inv, minlength=ENTRIES).float()
    return codes, entries, usage, worst


def kmeans(points, weights, k, iters, seed=0):
    """Weighted Lloyd's k-means with weighted k-means++ seeding. Returns the assignment."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    k = min(k, int((weights > 0).sum()))
    w = weights.double()
    first = int(torch.multinomial(w.cpu(), 1, generator=g))
    centers = [points[first]]
    d2 = (points - points[first]).square().sum(1).double()
    for _ in range(k - 1):
        p = (w * d2).cpu()
        if float(p.sum()) <= 0:
            p = w.cpu() * (d2.cpu() >= 0)
        i = int(torch.multinomial(p, 1, generator=g))
        centers.append(points[i])
        d2 = torch.minimum(d2, (points - points[i]).square().sum(1).double())
    centers = torch.stack(centers).clone()
    chunk = 1 << 20
    for _ in range(iters):
        assign = torch.cat([torch.cdist(points[i:i + chunk], centers).argmin(1) for i in range(0, len(points), chunk)])
        mass = torch.zeros(k, device=points.device).index_add_(0, assign, weights)
        total = torch.zeros(k, points.shape[1], device=points.device).index_add_(0, assign, points * weights[:, None])
        live = mass > 0
        centers[live] = total[live] / mass[live, None]
    return torch.cat([torch.cdist(points[i:i + chunk], centers).argmin(1) for i in range(0, len(points), chunk)])


class VQGrad(torch.autograd.Function):
    """y = F.linear(x, W, b) (the host's forward, bitwise); backward reduces G = g^T x (FP32) onto the families:
    row_i = sum_j G_ij W_ij, col_j = sum_i G_ij W_ij, bias = sum_t g_t, book_(b,k) = sum over the vectors coded (b, k)
    of G's 4-vectors. Also accumulates sum |x|^2 and the token count for the bias metric."""

    @staticmethod
    def forward(ctx, x, row, col, bias, book, layer):
        ctx.layer = layer
        ctx.save_for_backward(x)
        return F.linear(x, layer.weight, layer.bias)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        L, w = ctx.layer, ctx.layer.weight
        gx = g.matmul(w).to(x.dtype) if ctx.needs_input_grad[0] else None
        if not any(ctx.needs_input_grad[1:5]):
            return gx, None, None, None, None, None
        g2 = g.reshape(-1, g.shape[-1]).float()
        x2 = x.reshape(-1, x.shape[-1]).float()
        L.xsq += x2.square().sum().double()
        L.tokens += x2.shape[0]
        G = g2.t() @ x2
        gw = G * w.float()
        L.absgw += gw.abs().sum().double()
        out, n = G.shape
        if L.in_pad > n:
            G = F.pad(G, (0, L.in_pad - n))
        book = torch.zeros(L.books * ENTRIES, VQ_DIM, device=G.device).index_add_(
            0, L.entry_index(), G.view(-1, VQ_DIM))
        return gx, gw.sum(1), gw.sum(0), g2.sum(0), book, None


def family_vectors(raw, layers, metas, refs):
    """raw[path][sample] = dict(row, col, bias: lists over attached layers; book: list of [books*256, 4] tensors).
    Returns per family a list of per-tensor stacked [P*S, d] FP64 matrices in the metric coordinates (see vqg4.json)
    built lazily: yields (family, tensor index, matrix)."""
    paths = list(raw)
    order = [(p, s) for p in paths for s in range(len(raw[paths[0]]))]
    module_acc = {}
    for t, L in enumerate(layers):
        for fam in ('row', 'col', 'bias'):
            x = torch.stack([raw[p][s][fam][t].to(DEVICE, torch.float64) for p, s in order])
            if fam == 'row':
                x = x / L.weight.float().norm(dim=1).double().clamp_min(1e-30)
            elif fam == 'col':
                x = x / L.weight.float().norm(dim=0).double().clamp_min(1e-30)
            yield fam, t, x
        b = torch.stack([raw[p][s]['book'][t].to(DEVICE, torch.float64) for p, s in order])  # [N, books*256, 4]
        usage = L.usage.reshape(-1).double()
        yield 'book', t, (b / usage.clamp_min(1).sqrt()[None, :, None] * (usage > 0)[None, :, None]).flatten(1)
        a = L.scale.double().repeat_interleave(ENTRIES)  # per entry: its span scale
        scaled = b * a[None, :, None]
        m = metas['layer'][t]
        tied = torch.zeros(len(order), ENTRIES, VQ_DIM, device=DEVICE, dtype=torch.float64).index_add_(1, m, scaled)
        metric = torch.zeros(ENTRIES, device=DEVICE, dtype=torch.float64).index_add_(0, m, a.square() * usage)
        yield 'layer', t, (tied / metric.clamp_min(1e-300).sqrt()[None, :, None] * (metric > 0)[None, :, None]).flatten(1)
        key = refs['module_of'][t]
        acc = module_acc.setdefault(key, dict(
            g=torch.zeros(len(order), ENTRIES, VQ_DIM, device=DEVICE, dtype=torch.float64),
            metric=torch.zeros(ENTRIES, device=DEVICE, dtype=torch.float64)))
        mm = metas['module'][t]
        acc['g'].index_add_(1, mm, scaled)
        acc['metric'].index_add_(0, mm, a.square() * usage)
        del b, scaled, tied
    for i, key in enumerate(refs['modules']):
        acc = module_acc[key]
        yield 'module', i, (acc['g'] / acc['metric'].clamp_min(1e-300).sqrt()[None, :, None]
                            * (acc['metric'] > 0)[None, :, None]).flatten(1)


def transfer(gram, refs, samples, batches, eta, sources, paths=PATHS):
    """gram[t]: [P*S, P*S] Gram of tensor t's per-sample gradients in path-major order; refs[t] = the tensor's step
    scale. For each source (a set of paths) and batch: step d_t = -eta ref_t gbar_t / |gbar_t| (gbar = mean over the
    batch's seed-0 samples of the source's summed gradients); first-order decrease of every path on every sample.
    T[source][path][category] = sum over batches of the category's mean decrease / the batch's own; 'global' = one
    normalized step over all tensors together."""
    gram = gram.double()
    s, P = len(samples), len(paths)
    pidx = {p: i for i, p in enumerate(paths)}
    index = {(c, sd): i for i, (c, sd, _) in enumerate(samples)}
    out = {}
    for mode in ('tensor', 'global'):
        grams = gram if mode == 'tensor' else gram.sum(0, keepdim=True)
        rf = refs if mode == 'tensor' else torch.ones(1, dtype=torch.float64)
        res = {}
        for src, members in sources.items():
            rows = []
            for b in batches:
                idx = [index[(c, 0)] for c in b]
                cols = [pidx[q] * s + i for q in members for i in idx]
                dec = torch.zeros(P * s, dtype=torch.float64)
                for t in range(grams.shape[0]):
                    g = grams[t]
                    proj = g[:, cols].sum(1) / len(idx)  # <g_(p,k), gbar_t>
                    norm2 = float(g[cols][:, cols].sum()) / len(idx) ** 2
                    if norm2 <= 0:
                        continue
                    dec += eta * float(rf[t]) * proj / norm2 ** 0.5
                cats = {}
                for k, (c, sd, _) in enumerate(samples):
                    cats.setdefault(('in' if sd == 0 else 'seed') if c in b else ('ctx' if sd == 0 else 'ctx_seed'), []).append(k)
                row = {}
                for p in paths:
                    d = dec[pidx[p] * s:(pidx[p] + 1) * s]
                    row[p] = {key: float(d[ks].mean()) for key, ks in cats.items()}
                row['F'] = {key: row['Z'][key] + row['K'][key] for key in cats}
                rows.append(row)
            T = {}
            for p in list(paths) + ['F']:
                base = sum(r[p]['in'] for r in rows)
                T[p] = {key: (sum(r[p][key] for r in rows) / base if base > 0 else None) for key in ('seed', 'ctx', 'ctx_seed')}
                T[p]['in_decrease'] = base
            res[src] = T
        out[mode] = res
    return out


def cosines(gram, samples, groups, paths=PATHS):
    """Per group of tensors and path: mean cosine between samples of the same context at the two seeds / the same suite
    / the rest; energy share of the group per path."""
    s = len(samples)
    pairs = {}
    for i in range(s):
        for j in range(i + 1, s):
            (ci, _, ui), (cj, _, uj) = samples[i], samples[j]
            pairs.setdefault('seed' if ci == cj else 'suite' if ui == uj else 'other', []).append((i, j))
    pairs = {k: torch.tensor(v) for k, v in pairs.items()}
    names = sorted(set(groups)) + ['all']
    sums = {n: torch.zeros(len(paths) * s, len(paths) * s, dtype=torch.float64) for n in names}
    for t, grp in enumerate(groups):
        sums[grp] += gram[t].double()
        sums['all'] += gram[t].double()
    cos, energy = {}, {}
    for n, g in sums.items():
        for i, p in enumerate(paths):
            m = g[i * s:(i + 1) * s, i * s:(i + 1) * s]
            d = m.diagonal()
            whole = sums['all'][i * s:(i + 1) * s, i * s:(i + 1) * s].diagonal().mean()
            energy[f'{n}/{p}'] = float(d.mean() / whole.clamp_min(1e-300))
            if float(d.min()) <= 0:
                continue
            c = m / d.sqrt()[:, None] / d.sqrt()[None, :]
            cos[f'{n}/{p}'] = {k: float(c[v[:, 0], v[:, 1]].mean()) for k, v in pairs.items()}
    return dict(cosines=cos, energy=energy)


def reading(analysis, rule):
    best = {f: analysis[f]['transfer']['tensor']['F']['F']['ctx_seed'] for f in BOOK_FAMILIES}
    top = max(best, key=lambda f: best[f] if best[f] is not None else -1e9)
    v = best[top]
    verdict = 'go' if v is not None and v >= rule['go_min'] else 'weak' if v is not None and v >= rule['weak_min'] else 'no_go'
    return dict(T_F_ctx_seed=best, best=top, verdict=verdict)


class G4Work(lw.LowRankWork):
    def g4_dir(self):
        return self.base / 'vqg4' / 'GPTVQ'

    def attach(self):
        """Recover the host's codes, k-means the meta-entries, and route every non-time Linear through VQGrad while
        self.active (the Bank's plain forward otherwise)."""
        self.active, self.layers, self.names, self.P = False, [], [], {}
        root = self.arm_root(G4['host']['arm'])
        report, skipped = [], []
        for i, name in enumerate(self.bank.names):
            if timed(name):
                continue
            mod = self.bank.modules[name]
            stats = wk.read(root / 'stats' / f'{i:04d}.json')
            if stats['module'] != name:
                raise RuntimeError(f'Layer identity differs at {i}: {name}')
            cfg = stats['config']
            codes, entries, usage, worst = recover(mod.weight.detach(), cfg)
            if codes is None:  # not GPTVQ-coded as configured: left on its plain forward, reported
                skipped.append(dict(index=i, module=name, shape=list(mod.weight.shape), config=cfg, distinct=worst))
                continue
            L = Layer(mod.weight, mod.bias, cfg['rows_per_group'], cfg['column_spans'], codes.shape[1] * VQ_DIM, codes, entries,
                      usage)
            self.layers.append(L)
            self.names.append(name)
            self.P[name] = dict(row=torch.zeros(mod.out_features, device=DEVICE, requires_grad=True),
                                col=torch.zeros(mod.in_features, device=DEVICE, requires_grad=True),
                                bias=torch.zeros(mod.out_features, device=DEVICE, requires_grad=True),
                                book=torch.zeros(L.books * ENTRIES, VQ_DIM, device=DEVICE, requires_grad=True))
            mod.forward = self._wrapped(name, L, mod.forward)
            report.append(dict(index=i, module=name, books=L.books, max_entries=worst,
                               used=int((usage > 0).sum()), slots=L.books * ENTRIES))
            if len(self.layers) % 100 == 0:
                self.progress('codes', layers=len(self.layers))
        self.modules = sorted({module_of(n) for n in self.names})
        self.metas = dict(layer=[], module=[None] * len(self.layers))
        for L in self.layers:
            pts = (L.entries / L.scale[:, None, None]).reshape(-1, VQ_DIM)
            self.metas['layer'].append(kmeans(pts, L.usage.reshape(-1), ENTRIES, 25))
        for m in self.modules:
            ids = [t for t, n in enumerate(self.names) if module_of(n) == m]
            pts = torch.cat([(self.layers[t].entries / self.layers[t].scale[:, None, None]).reshape(-1, VQ_DIM) for t in ids])
            wts = torch.cat([self.layers[t].usage.reshape(-1) for t in ids])
            assign, start = kmeans(pts, wts, ENTRIES, 25), 0
            for t in ids:
                n = self.layers[t].books * ENTRIES
                self.metas['module'][t] = assign[start:start + n]
                start += n
        n_all = sum(m.weight.numel() for m in self.bank.modules.values())
        n_skip = sum(d['shape'][0] * d['shape'][1] for d in skipped)
        wk.save(self.g4_dir() / 'codes.json', dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, layers=report,
                                                   max_entries=max(r['max_entries'] for r in report),
                                                   books=sum(r['books'] for r in report), skipped=skipped,
                                                   skipped_weight_share=n_skip / n_all))
        if n_skip / n_all > G4['checks']['skipped_share_max']:
            raise RuntimeError(f'Codes not recoverable on {len(skipped)} layers ({n_skip / n_all:.3%} of the weights)')

    def _wrapped(self, name, L, plain):
        def run(x):
            if not self.active:
                return plain(x)
            self.bank.seen.add(name)
            p = self.P[name]
            return VQGrad.apply(x, p['row'], p['col'], p['bias'], p['book'], L)
        return run

    def zero(self):
        for p in self.P.values():
            for v in p.values():
                v.grad = None

    def collect(self):
        """This pass's gradients on the CPU (book in BF16) and the worst relative gap among sum_i row, sum_j col and
        sum_(b,k) <book, entry> per layer (each equals sum G o W), relative to sum |G o W|."""
        out, worst = dict(row=[], col=[], bias=[], book=[]), 0.0
        for n, L in zip(self.names, self.layers):
            g = {k: (self.P[n][k].grad if self.P[n][k].grad is not None else torch.zeros_like(self.P[n][k])).detach()
                 for k in out}
            r, c = float(g['row'].double().sum()), float(g['col'].double().sum())
            b = float((g['book'].double() * L.entries.reshape(-1, VQ_DIM).double()).sum())
            scale = max(float(L.absgw), 1e-30)
            worst = max(worst, abs(r - c) / scale, abs(r - b) / scale)
            L.absgw.zero_()
            for k in out:
                out[k].append(g[k].to('cpu', torch.bfloat16 if k == 'book' else torch.float32).clone())
        self.zero()
        return out, worst

    def sample(self, case, seed):
        """Gradients of lambda Z, beta K (student query) and A_tf (student reading the teacher future) on one sample."""
        from reader_loss import reader_terms
        beta, stats = self.cfg['loss']['beta'], {'action': self.p['reader_source']['action_stats']}
        target = self.teacher(case, seed)
        value, description = self.datum(case)
        sampler = self.rows[case]['sampler_seeds'][seed]
        self.enough()
        self.active = False
        with torch.no_grad():
            _, future, _ = self.query(case, seed)
            self.runtime.description = description
            action, _ = self.reader(self.runtime, value, sampler, target['future'].to(DEVICE, torch.float32),
                                    checkpointed=False)
        self.count('reader_queries')
        a_leaf = action.detach().float().clone().requires_grad_(True)
        z_leaf = future.detach().float().clone().requires_grad_(True)
        total, _, _ = self.loss(a_leaf, z_leaf, target)
        ga, gz = torch.autograd.grad(total, (a_leaf, z_leaf))
        with self.bf16_reader() as reader:
            self.runtime.description = description
            leaf = future.detach().float().clone().requires_grad_(True)
            read, _ = reader(self.runtime, value, sampler, leaf, checkpointed=True)
            k = reader_terms(torch, read, target['action'], stats, self.p['reader_source']['config'])['reader_kl']
            gk, = torch.autograd.grad(beta * k, leaf)
        self.count('reader_queries')
        grads, worst, invariant = {}, 0.0, 0.0
        self.active = True
        self.zero()
        for path, cot in (('Z', gz), ('K', gk), ('A', ga)):
            if path == 'A':
                self.runtime.description = description
                given = target['future'].to(DEVICE, torch.float32).requires_grad_(True)
                out, _ = self.reader(self.runtime, value, sampler, given, checkpointed=True)
                self.count('reader_queries')
                reference = action
            else:
                _, out, _ = self.query(case, seed, grad=True)
                reference = future
            if not out.requires_grad:
                raise RuntimeError(f'Replay of path {path} lost its graph')
            worst = max(worst, wk.max_abs(out, reference))
            torch.autograd.backward(out, cot.to(out))
            self.counters['parameter_reverse_passes'] += 1
            grads[path], gap = self.collect()
            invariant = max(invariant, gap)
            del out
        self.active = False
        if worst > G4['checks']['replay_max_abs']:
            raise RuntimeError(f'Grad replay differs from the staged no-grad pass: {worst}')
        if invariant > G4['checks']['invariant_rel_max']:
            raise RuntimeError(f'Gradient reductions disagree: {invariant}')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info = dict(case_id=case, seed_index=seed, az=float(total.detach()), k=float(k.detach()), replay_max_abs=worst,
                    invariant_rel=invariant)
        return grads, info

    def samples(self):
        cases = [c for b in self.schedule[:4] for c in b]
        return [(c, sd, self.rows[c]['suite']) for sd in (0, 1) for c in cases]

    def run(self):
        root = self.g4_dir()
        if (root / 'summary.json').exists():
            return
        self.install(G4['host']['arm'])
        self.attach()
        samples, raw, infos = self.samples(), {p: [] for p in PATHS}, []
        for n, (case, seed, _) in enumerate(samples):
            path = root / 'samples' / f'{case}_{seed}.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
            else:
                grads, info = self.sample(case, seed)
                info['xsq'] = [float(L.xsq) for L in self.layers]
                info['tokens'] = [L.tokens for L in self.layers]
                saved = dict(info=info, grads=grads)
                self.tsave(path, saved)
            for p in PATHS:
                raw[p].append(saved['grads'][p])
            infos.append(saved['info'])
            self.progress('g4_sample', samples=n + 1)
        self.grams(samples, raw, infos)

    def grams(self, samples, raw, infos):
        root = self.g4_dir()
        last = infos[-1]  # cumulative over this process's grad passes (or the saved run's)
        refs = {}
        wn = [float(L.weight.float().norm()) for L in self.layers]
        xmean = [last['xsq'][t] / max(last['tokens'][t], 1) for t in range(len(self.layers))]
        refs['tensor'] = {f: torch.tensor(wn, dtype=torch.float64) for f in ('book', 'layer', 'row', 'col')}
        refs['tensor']['bias'] = torch.tensor([wn[t] * (xmean[t] / self.layers[t].weight.shape[1]) ** 0.5
                                               for t in range(len(self.layers))], dtype=torch.float64)
        mod_of = [module_of(n) for n in self.names]
        refs['tensor']['module'] = torch.tensor([sum(wn[t] ** 2 for t in range(len(wn)) if mod_of[t] == m) ** 0.5
                                                 for m in self.modules], dtype=torch.float64)
        grams = {f: [] for f in FAMILIES}
        for fam, t, x in family_vectors(raw, self.layers, self.metas, dict(module_of=mod_of, modules=self.modules)):
            grams[fam].append((x @ x.t()).cpu())
        gram = {f: torch.stack(v) for f, v in grams.items()}
        groups = {f: (self.modules if f == 'module' else [lw.tensor_group(n + '.W') for n in self.names]) for f in FAMILIES}
        self.tsave(root / 'gram.pt', dict(gram=gram, samples=[list(x) for x in samples], names=self.names,
                                          modules=self.modules, refs={k: v for k, v in refs['tensor'].items()}))
        batches = self.schedule[:4]
        analysis = {f: dict(transfer=transfer(gram[f], refs['tensor'][f], samples, batches, G4['eta'], G4['sources']),
                            **cosines(gram[f], samples, groups[f])) for f in FAMILIES}
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, g4=G4, codes=wk.read(root / 'codes.json'),
                       samples=infos, analysis=analysis, reading=reading(analysis, G4['reading']))
        for s in summary['samples']:
            s.pop('xsq', None)
            s.pop('tokens', None)
        wk.save(root / 'summary.json', summary)
        (root / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')

    def stage(self, name, shard, shards):
        if name != 'vqg4' or shards != 1:
            raise ValueError(f'Unknown step {name} / {shards} shards')
        self.load()
        self.capture_masters()
        self.run()
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


FAMILY_LABELS = dict(book='码本条目偏移（GPTVQ 原生，每本 13 万权重）', layer='每层共享 256 个元条目', module='每模块共享 256 个元条目',
                     row='行 log-scale', col='列 log-scale', bias='输出偏置')


def fmt(v, spec='.3f'):
    return '—' if v is None else format(v, spec)


def report(s):
    a, rd, rule = s['analysis'], s['reading'], s['g4']['reading']
    c = s['codes']
    lines = ['# G4：GPTVQ W2 上共享参数的梯度能否迁移（vqg4.json）', '',
             f"作业 `{s['pbs_jobid']}`。码本从稠密权重恢复：{len(c['layers'])} 层、{c['books']} 本码本，每块最多 {c['max_entries']} 个不同向量。"
             f"样本 {len(s['samples'])} 个（40 context × seed 0/1），回放最大差 {max(x['replay_max_abs'] for x in s['samples']):.2g}，"
             f"梯度归约一致性（最大相对差）{fmt(max(x['invariant_rel'] for x in s['samples']), '.2g')}。", '',
             '## 迁移比 T（逐张量相对步长 η = %g；走一步对“其他 context、新噪声”的一阶下降 / 对这批自身）' % s['g4']['eta'], '',
             '| 参数族 | F→F | Z→Z | K→K | A→A | F→F 同 context 换 seed | F→F 其他 context seed 0 | 全局步 F→F |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        t, g = a[f]['transfer']['tensor'], a[f]['transfer']['global']
        lines.append(f"| {FAMILY_LABELS[f]} | {fmt(t['F']['F']['ctx_seed'])} | {fmt(t['Z']['Z']['ctx_seed'])} | "
                     f"{fmt(t['K']['K']['ctx_seed'])} | {fmt(t['A']['A']['ctx_seed'])} | {fmt(t['F']['F']['seed'])} | "
                     f"{fmt(t['F']['F']['ctx'])} | {fmt(g['F']['F']['ctx_seed'])} |")
    lines += ['', '参照（fwld1，QTIP 上每层 rank-4 支路）：future 一侧 T = 0.021，动作一侧 0.423。', '',
              '## 跨样本余弦（所有张量合起来）', '', '| 参数族 | 路径 | 同 context 两 seed | 同 suite | 其他 |', '|---|---|---:|---:|---:|']
    for f in FAMILIES:
        for p in PATHS:
            v = a[f]['cosines'].get(f'all/{p}')
            if v:
                lines.append(f"| {FAMILY_LABELS[f]} | {p} | {fmt(v.get('seed'))} | {fmt(v.get('suite'))} | {fmt(v.get('other'))} |")
    lines += ['', '## 判读（预先登记）', '',
              f"- 三个码本族的 T(F→F，其他 context，seed 1)：" + '，'.join(f"{f} {fmt(v)}" for f, v in rd['T_F_ctx_seed'].items())
              + f"；最好的是 {rd['best']}。",
              f"- 门槛：≥ {rule['go_min']} 为 go，≥ {rule['weak_min']} 为 weak。结论：**{rd['verdict']}**。",
              '- ' + {'go': rule['go'], 'weak': rule['weak'], 'no_go': rule['no_go']}[rd['verdict']]]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    cfg = wk.read(parser.parse_args().protocol)
    if cfg['protocol'] != wk.PROTOCOL:
        raise ValueError('Protocol identity differs')
    stage, shard, shards = os.environ['FW_STAGE'], int(os.environ['FW_SHARD']), int(os.environ['FW_SHARDS'])
    job = f"{os.environ['FW_JOB']}_{shard}"
    failure = Path(cfg['remote_root']) / 'results' / 'failures' / f'{job}.json'
    failure.unlink(missing_ok=True)
    try:
        G4Work(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
