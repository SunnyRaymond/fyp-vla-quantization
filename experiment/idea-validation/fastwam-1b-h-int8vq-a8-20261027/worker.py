"""Phase 1b: H-only coordinates, integer-codebook row-scale VQ, A8 operands, A+Z+reader-KL.

Reuses the frozen loss-pair runtime (copied from job 25921795 as lp_worker/source_worker and
their dependencies). Only the coordinate change, the activation grid and the weight format are new.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import sampler  # owns the PBS allocation guard; nothing model-scale is imported before it

sampler._require_compute_allocation()
import torch
import torch.nn.functional as F

import batched_transform
import codebook as base_vq
import lp_worker as lp
import micro_codebook as micro
from c8_train_banks import TrainableBanks
from rotation_bank import RotationBanks, stream_for, transformed

PROTOCOL = 'fastwam-1b-h-int8vq-a8-20261027-v1'
HERE = Path(__file__).resolve().parent
EXTRA = ('first10gripper_disagreement',)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def aggregate(records, cases, seeds, suites, metrics):
    if Counter((r['case_id'], r['seed_index']) for r in records) != Counter((c, s) for c in cases for s in seeds):
        raise ValueError('Complete paired endpoint coverage required')
    by_suite = {}
    for suite in suites:
        rows = [r for r in records if r['suite'] == suite]
        ids = sorted({r['case_id'] for r in rows})
        by_suite[suite] = {m: sum(sum(r[m] for r in rows if r['case_id'] == c) / len(seeds) for c in ids) / len(ids)
                           for m in metrics}
        by_suite[suite].update(contexts=len(ids), queries=len(rows))
    return dict(queries=len(records), by_suite=by_suite,
                overall={m: sum(by_suite[s][m] for s in suites) / len(suites) for m in metrics})


class A8(torch.autograd.Function):
    """Per-token symmetric INT8 grid with BF16 output; identity STE like the frozen A4."""
    @staticmethod
    def forward(ctx, z):
        z = z.float()
        scale = z.abs().amax(-1, keepdim=True) / 127
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(z / safe).clamp(-127, 127) * scale).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        return grad.float()


class NoRotation:
    """The U slot of the frozen runtime is the identity and owns no parameters."""
    raw = {}

    def refresh(self):
        pass

    def apply_after_h(self, z, stream):
        return z

    def parameters(self):
        return []

    def snapshot(self):
        return {}

    def orthogonality(self):
        return {}

    def freeze(self):
        pass


NO_ROTATION = NoRotation()


def h_weight(weight, transform, name):
    """W S H in FP32: the weight side of the frozen Smooth/signs/H128 pairing, without U."""
    return transformed(weight.to('cuda'), transform, NO_ROTATION, stream_for(name), weight=True)


# Codebook masters live in grid units / GRID_SCALE: with Adam's ~lr-sized steps (lr 5e-4) each update moves an
# entry by ~0.125 integer, so 8 updates can move it by ~1 grid point, like Q0's ~1 BF16 ulp. In normalized units
# the integer grid (delta ~0.03) is far coarser than lr, and the books would never leave their rounding cells.
GRID_SCALE = 250.0


def integer_books(master, delta, bound):
    return torch.maximum(torch.minimum(torch.round(master / delta), bound), -bound)


def fit_int_vq(weight, second_moment, seed, sample_limit, iterations):
    """Continuous 2x256 additive fit on row-normalized d4 vectors, then one integer grid per layer."""
    rows, width = base_vq._check_weight(weight)
    groups = (width + 3) // 4
    w = weight.detach().float()
    row = w.square().mean(1).sqrt()
    safe = torch.where(row > 0, row, torch.ones_like(row))
    vectors = F.pad(w / safe[:, None], (0, groups * 4 - width)).reshape(-1, 4).contiguous()
    metric = micro.MicroMetric(w, row[:, None].expand(rows, (width + 31) // 32).contiguous(), second_moment)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    count = min(vectors.shape[0], sample_limit)
    ids = (torch.arange(vectors.shape[0], device=w.device) if vectors.shape[0] <= sample_limit
           else torch.randint(vectors.shape[0], (count,), generator=generator).to(w.device))
    sample, sample_metric = vectors[ids], metric.gather(ids)
    book0 = base_vq._kmeans(sample, sample_metric, generator)
    index0 = base_vq._assign(sample, book0, sample_metric)
    book1 = base_vq._kmeans(sample - book0[index0], sample_metric, generator)
    index1 = base_vq._assign(sample - book0[index0], book1, sample_metric)
    for _ in range(iterations):
        target = sample - book1[index1]
        index0 = base_vq._assign(target, book0, sample_metric)
        book0 = base_vq._update_centroids(target, index0, sample_metric, book0)
        residual = sample - book0[index0]
        index1 = base_vq._assign(residual, book1, sample_metric)
        book1 = base_vq._update_centroids(residual, index1, sample_metric, book1)
    # One grid step per layer so that every codeword pair sums inside the signed INT8 range.
    span = float(book0.abs().max() + book1.abs().max())
    delta = span / 127 if span > 0 else 1.0
    while True:
        i0, i1 = torch.round(book0 / delta), torch.round(book1 / delta)
        bounds = [int(i0.abs().max()), int(i1.abs().max())]
        if sum(bounds) <= 127:
            break
        delta *= sum(bounds) / 127
    grid = torch.stack((i0, i1)) * delta
    index0 = base_vq._assign(vectors, grid[0], metric)
    index1 = base_vq._assign(vectors - grid[0][index0], grid[1], metric)
    for _ in range(2):
        index0 = base_vq._assign(vectors - grid[1][index1], grid[0], metric)
        index1 = base_vq._assign(vectors - grid[0][index0], grid[1], metric)
    encoded = dict(
        format=micro.FORMAT, shape=(rows, width), block_size=4, scale_block=micro.SCALE_BLOCK,
        books=torch.stack((i0, i1)).to(torch.bfloat16), row_scales=(row * delta).to(torch.bfloat16),
        scale_exponents=torch.full((rows, (width + 31) // 32), 127, dtype=torch.uint8, device=w.device),
        indices=torch.stack((index0, index1), -1).reshape(rows, groups, 2).to(torch.uint8),
        fit_stats=dict(vectors=vectors.shape[0], sample_vectors=count, iterations=iterations,
                       assignment_metric='A8 input second moment x row scale squared', integer_codebooks=True))
    used = int((i0[index0] + i1[index1]).abs().max())
    return encoded, dict(continuous_books=torch.stack((book0, book1)), delta=delta, bounds=bounds, max_used_sum=used)


class IntDecodedWeight(torch.autograd.Function):
    """Decoded weight = integer books x BF16 row scale; STE through round/clamp to the FP32 masters."""
    @staticmethod
    def forward(ctx, books, log_scales, active, indices, delta, bound, dense):
        ctx.save_for_backward(books, log_scales, active, indices, bound)
        ctx.delta, ctx.shape = delta, tuple(dense.shape)
        return dense.detach().view_as(dense)

    @staticmethod
    def backward(ctx, grad):
        books, log_scales, active, indices, bound = ctx.saved_tensors
        out_width, in_width = ctx.shape
        stored = integer_books(books, ctx.delta, bound)
        raw = log_scales.exp() * active
        gains = raw.to(torch.bfloat16).float()
        book_grad, scale_grad = torch.zeros_like(books), torch.zeros_like(log_scales)
        padded = indices.shape[1] * 4
        chunk = max(1, (1 << 20) // indices.shape[1])
        for start in range(0, out_width, chunk):
            end = min(start + chunk, out_width)
            g = F.pad(grad[start:end].float(), (0, padded - in_width)).reshape(end - start, -1, 4)
            idx = indices[start:end].long()
            values = stored[0][idx[..., 0]] + stored[1][idx[..., 1]]
            scale_grad[start:end] = (g * values).sum((1, 2)) * raw[start:end]
            weighted = (g * gains[start:end, None, None]).reshape(-1, 4)
            for book in range(2):
                book_grad[book].index_add_(0, idx[..., book].reshape(-1), weighted)
        return book_grad / ctx.delta, scale_grad, None, None, None, None, None


class IntBanks(TrainableBanks):
    """Integer-codebook row-scale VQ in H coordinates with A8 operands; reuses snapshot/restore/refresh."""
    def __init__(self, runtime, directory, transforms):
        self.modules, self.transforms, self.rotations = runtime.q.modules, transforms, NO_ROTATION
        self.encodings, self.books, self.logs, self.active, self.delta, self.bound = {}, {}, {}, {}, {}, {}
        self.seen = set()
        for index, (name, mod) in enumerate(self.modules.items()):
            bank = torch.load(Path(directory) / f'{index:04d}.pt', map_location='cuda', weights_only=True)
            if bank.get('protocol') != PROTOCOL or bank.get('module') != name or bank.get('module_index') != index:
                raise RuntimeError('Integer bank identity differs: ' + name)
            encoding = bank['encoded']
            micro.check_encoding(encoding)
            self.encodings[name] = encoding
            self.delta[name] = 1 / GRID_SCALE
            self.bound[name] = torch.tensor(bank['bounds'], dtype=torch.float32, device='cuda').view(2, 1, 1)
            self.books[name] = torch.nn.Parameter(
                bank['continuous_books'].float() / (float(bank['delta']) * GRID_SCALE))
            scales = encoding['row_scales'].float()
            self.active[name] = (scales > 0).float()
            self.logs[name] = torch.nn.Parameter(scales.clamp_min(torch.finfo(torch.float32).tiny).log())
            mod.forward = self._forward(name, mod)
        self.refresh()

    def _forward(self, name, mod):
        def forward(x):
            self.seen.add(name)
            qx = A8.apply(batched_transform.transform_input(x, self.transforms[name]))
            weight = IntDecodedWeight.apply(self.books[name], self.logs[name], self.active[name],
                                            self.encodings[name]['indices'], self.delta[name],
                                            self.bound[name], mod.weight)
            return F.linear(qx, weight, mod.bias)
        return forward

    def stored_encoding(self, name):
        value = dict(self.encodings[name])
        value['books'] = integer_books(self.books[name].detach(), self.delta[name],
                                       self.bound[name]).to(torch.bfloat16)
        value['row_scales'] = (self.logs[name].detach().exp() * self.active[name]).to(torch.bfloat16)
        return value

    def changed_codewords(self):
        changed = sum(int((self.stored_encoding(n)['books'] != self.encodings[n]['books']).sum()) for n in self.modules)
        return dict(changed_integer_entries=changed, total_integer_entries=len(self.modules) * 2 * 256 * 4)


# Learnable H: per layer, 7 Givens stages on the FWHT butterfly pairing (64 angles each, shared by the layer's
# 128-blocks) after the fixed signed H128. Angles start at 0, so G = I exactly and the start is bitwise VQ0'.
# Deployment cost: the same 7-stage butterfly as the FWHT, fusable into the Smooth/H operand kernel.
# Angles live in rad / ANGLE_SCALE: Adam moves each by ~lr per update, i.e. ~0.01 rad, <= ~0.08 rad over 8 updates.
ANGLE_SCALE = 20.0


def butterfly(theta):
    """128x128 orthogonal G with v @ G = the 7 Givens stages applied to v; theta [7, 64] in radians."""
    m = torch.eye(128, dtype=torch.float32, device=theta.device)
    for stage in range(7):
        h = 1 << stage
        v = m.view(128, 64 // h, 2, h)
        a, b = v[:, :, 0], v[:, :, 1]
        t = theta[stage].view(64 // h, h)
        c, s = t.cos(), t.sin()
        m = torch.stack((c * a - s * b, s * a + c * b), 2).reshape(128, 128)
    return m


class RotA8(torch.autograd.Function):
    """z -> per-128-block z @ G -> A8. Saves only the BF16 operand (F.linear keeps it anyway) and G; the
    G gradient uses z ~= q G^T, i.e. the A8 operand in place of the unquantized input (same STE as A8)."""
    @staticmethod
    def forward(ctx, z, g):
        z = z.float()
        full = z.shape[-1] // 128 * 128
        r = z.clone()
        r[..., :full] = (z[..., :full].reshape(-1, 128) @ g).view(*z.shape[:-1], full)
        scale = r.abs().amax(-1, keepdim=True) / 127
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = (torch.round(r / safe).clamp(-127, 127) * scale).to(torch.bfloat16)
        ctx.save_for_backward(q, g)
        ctx.full = full
        return q

    @staticmethod
    def backward(ctx, grad):
        q, g = ctx.saved_tensors
        full = ctx.full
        grad = grad.float()
        out = grad.clone()
        blocks = grad[..., :full].reshape(-1, 128)
        out[..., :full] = (blocks @ g.T).view(*grad.shape[:-1], full)
        z = q[..., :full].float().reshape(-1, 128) @ g.T
        return out, z.T @ blocks


class LearnedHBanks(IntBanks):
    """IntBanks plus one butterfly angle set per Linear, trained jointly with the integer codebooks."""
    def __init__(self, runtime, directory, transforms):
        super().__init__(runtime, directory, transforms)
        self.angles = {name: torch.nn.Parameter(torch.zeros(7, 64, device='cuda')) for name in self.modules}
        for name, mod in self.modules.items():
            mod.forward = self._rotated_forward(name, mod)

    def rotation(self, name):
        return butterfly(self.angles[name] * ANGLE_SCALE)

    def _rotated_forward(self, name, mod):
        def forward(x):
            self.seen.add(name)
            qx = RotA8.apply(batched_transform.transform_input(x, self.transforms[name]), self.rotation(name))
            weight = IntDecodedWeight.apply(self.books[name], self.logs[name], self.active[name],
                                            self.encodings[name]['indices'], self.delta[name],
                                            self.bound[name], mod.weight)
            return F.linear(qx, weight, mod.bias)
        return forward

    def parameters(self):
        return super().parameters() + [self.angles[name] for name in self.modules]

    def snapshot(self):
        state = super().snapshot()
        for name in self.modules:
            state[name]['angles'] = self.angles[name].detach().cpu().clone()
        return state

    @torch.no_grad()
    def restore(self, state):
        for name in self.modules:
            self.angles[name].copy_(state[name]['angles'])
        super().restore(state)

    def angle_stats(self):
        values = torch.cat([self.angles[n].detach().flatten() for n in self.modules]) * ANGLE_SCALE
        moved = [float((self.rotation(n).detach() - torch.eye(128, device='cuda')).norm() / 128 ** .5)
                 for n in self.modules]
        return dict(max_abs_rad=float(values.abs().max()), rms_rad=float(values.square().mean().sqrt()),
                    rotation_rms_distance_from_identity=sum(moved) / len(moved),
                    rotation_max_distance_from_identity=max(moved))


# Two-stage G (C8 decision run): G rotates both sides (x S H G, W S H G), so the function is unchanged and G
# only changes what the VQ has to fit. Stage 1 is ITQ-style on a row sample: assign the rotated row-normalized
# d4 vectors to the layer's continuous H-coordinate books, then Adam on the angles toward those codewords;
# stage 2 refits the integer VQ on W S H G. A layer keeps G only if the refit lowers its weighted error.
C8 = [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070]  # original C8 cohort, inside C80 (teacher cache reused)
G_ROUNDS, G_STEPS, G_LR, G_ROWS, G_SEED = 8, 50, 0.02, 1024, 20261028


def rotate_columns(w, g):
    full = w.shape[1] // 128 * 128
    return torch.cat(((w[:, :full].reshape(-1, 128) @ g).view(w.shape[0], full), w[:, full:]), 1)


def rotate_moment(m, g):
    """Diagonal second moment of x G from that of x (cross-channel terms are not tracked)."""
    full = m.shape[0] // 128 * 128
    return torch.cat(((m[:full].view(-1, 128) @ g.square()).flatten(), m[full:]))


def nearest(vectors, metric, book):
    out = torch.empty(vectors.shape[0], dtype=torch.long, device=vectors.device)
    for s in range(0, vectors.shape[0], 32768):
        v, m = vectors[s:s + 32768, None], metric[s:s + 32768, None]
        out[s:s + 32768] = ((v - book[None]).square() * m).sum(-1).argmin(1)
    return out


def weighted_error(weight, decoded, moment):
    m = moment / moment.mean().clamp_min(1e-12)
    return float(((decoded - weight).square() * m).sum() / (weight.square() * m).sum().clamp_min(1e-12))


def learn_g(weight, moment, books):
    """weight W S H [out, in] FP32, moment [in], books [2, 256, 4] continuous row-normalized books."""
    rows, width = weight.shape
    theta = torch.zeros(7, 64, device=weight.device, requires_grad=True)
    if width < 128:
        return butterfly(theta.detach()), theta.detach(), []
    groups = (width + 3) // 4
    pick = torch.randperm(rows, generator=torch.Generator(device='cpu').manual_seed(G_SEED))[:G_ROWS]
    w = weight[pick.to(weight.device)]
    r = w.square().mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    b0, b1 = books[0].float(), books[1].float()
    optimizer = torch.optim.Adam([theta], lr=G_LR)
    history = []
    for _ in range(G_ROUNDS):
        with torch.no_grad():
            g = butterfly(theta)
            m = rotate_moment(moment, g)
            m = m / m.mean().clamp_min(1e-12)
            wg = rotate_columns(w, g)
            v = F.pad(wg / r, (0, groups * 4 - width)).reshape(-1, 4)
            met = F.pad(m, (0, groups * 4 - width)).view(1, -1).expand(len(pick), -1).reshape(-1, 4)
            i0 = nearest(v, met, b0)
            i1 = nearest(v - b0[i0], met, b1)
            i0 = nearest(v - b1[i1], met, b0)
            target = (b0[i0] + b1[i1]).view(len(pick), -1)[:, :width] * r
            denom = (w.square() * m).sum()
            history.append(float(((wg - target).square() * m).sum() / denom))
        for _ in range(G_STEPS):
            loss = ((rotate_columns(w, butterfly(theta)) - target).square() * m).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return butterfly(theta.detach()), theta.detach(), history


class FixedGBanks(IntBanks):
    """IntBanks fitted in per-layer G coordinates; G is frozen, only integer books and row scales train."""
    def __init__(self, runtime, directory, transforms):
        super().__init__(runtime, directory, transforms)
        saved = torch.load(Path(directory) / 'rotations.pt', map_location='cuda', weights_only=True)
        if saved.get('protocol') != PROTOCOL or set(saved['rotations']) != set(self.modules):
            raise RuntimeError('G rotation set differs')
        self.g = saved['rotations']
        for name, mod in self.modules.items():
            mod.forward = self._g_forward(name, mod)

    def rotation(self, name):
        return self.g[name]

    def _g_forward(self, name, mod):
        def forward(x):
            self.seen.add(name)
            qx = RotA8.apply(batched_transform.transform_input(x, self.transforms[name]), self.g[name])
            weight = IntDecodedWeight.apply(self.books[name], self.logs[name], self.active[name],
                                            self.encodings[name]['indices'], self.delta[name],
                                            self.bound[name], mod.weight)
            return F.linear(qx, weight, mod.bias)
        return forward


class H8Dense:
    """Original BF16 weights in H coordinates with A8 operands; optional per-call moment sums."""
    def __init__(self, runtime, transforms, masters):
        self.modules, self.transforms, self.rotations = runtime.q.modules, transforms, NO_ROTATION
        self.seen, self.collect, self.sums, self.rows = set(), False, {}, {}
        with torch.no_grad():
            for name, mod in self.modules.items():
                mod.weight.data = h_weight(masters[name], transforms[name], name).to(torch.bfloat16)
                mod.forward = self._forward(name, mod)

    def _forward(self, name, mod):
        def forward(x):
            self.seen.add(name)
            qx = A8.apply(batched_transform.transform_input(x, self.transforms[name]))
            if self.collect:
                rows = qx.detach().float().reshape(-1, mod.in_features)
                self.sums[name] = self.sums.get(name, 0) + rows.square().sum(0)
                self.rows[name] = self.rows.get(name, 0) + rows.shape[0]
            return F.linear(qx, mod.weight, mod.bias)
        return forward


class Work(lp.Work):
    """Loss-pair reader_kl worker bound to this root; only the banks and the stage flow differ."""
    def __init__(self, cfg, stage):
        self.cfg, self.stage_name = cfg, stage
        self.results = Path(cfg['remote_root']) / 'results'
        self.stage_dir = self.results / 'stages' / stage
        self.banks_dir = self.results / 'banks'
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        parent = read(HERE / 'lp_protocol.json')
        parent['remote_root'] = cfg['remote_root']
        parent['source_initialization']['bank_directory'] = str(self.banks_dir)
        parent['source_initialization']['common_endpoint'] = str(self.results / 'stages/prep/init_anchor.json')
        super().__init__(parent, 'reader_kl')
        self.deadline = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ['FW_SOFT_DEADLINE_SECONDS'])
        self.export_dir = self.results / 'export'
        self.banks_g_dir = self.results / 'banks_g'
        if stage in ('train_lh', 'c8_h', 'c8_g'):
            # Same root, prep banks and beta preparation as 1b; own training unit, endpoints and bundle.
            self.unit = self.results / 'arms' / ('learned_h' if stage == 'train_lh' else stage)
            self.unit.mkdir(parents=True, exist_ok=True)
            self.export_dir = self.results / 'export_lh'
            counters = self.unit / 'query_counters.json'
            self.counters = (read(counters)['counters'] if counters.exists()
                             else dict.fromkeys(self.counters, 0))

    def progress(self, phase, **extra):
        value = dict(protocol=PROTOCOL, stage=self.stage_name, pbs_jobid=self.job, phase=phase,
                     seconds=time.monotonic() - self.started, **self.counters, **extra)
        save(self.stage_dir / 'progress.json', value)
        print('FW1B_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

    def _runtime(self):
        self.runtime = self.runner.Runtime(self.unit)
        self.runtime.model.eval()
        self.runtime.arm('bf16')
        rb = RotationBanks(self.runtime, self.p['calibration']['bank_directory'])
        # Original BF16 weights: fit target in prep, frozen reader weights in train.
        self.masters = {n: m.weight.detach().cpu().clone() for n, m in self.runtime.q.modules.items()}
        self.transforms = rb.transforms
        rb.close()
        self.initial_rotation = {}

    def load_model(self):
        self._runtime()
        learned = self.stage_name == 'train_lh'
        self.bank = (LearnedHBanks if learned else IntBanks)(self.runtime, self.banks_dir, self.transforms)
        if (len(self.bank.parameters()) != (1842 if learned else 1228)
                or any(v.requires_grad for v in self.runtime.model.parameters())):
            raise ValueError('Only 614 integer codebook/row-scale pairs may train')
        self.helpers.Work.reset_rng(self)

    @torch.no_grad()
    def transform_check(self):
        name = next(n for n, m in self.bank.modules.items() if stream_for(n) == 'video' and m.in_features >= 256)
        mod = self.bank.modules[name]
        generator = torch.Generator(device='cuda').manual_seed(20261027)
        x = torch.randn(8, mod.in_features, device='cuda', generator=generator).to(torch.bfloat16)
        reference = F.linear(x, self.masters[name].to('cuda'))
        actual = F.linear(batched_transform.transform_input(x, self.transforms[name]).to(torch.bfloat16), mod.weight)
        rel = float((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
        if rel > self.cfg['gates']['transform_check_max_relative']:
            raise ValueError(f'H-coordinate weight/activation pairing failed at {name}: {rel}')
        return dict(module=name, relative_error=rel)

    def collect_moments(self):
        path = self.results / 'moments.pt'
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved.get('protocol') == PROTOCOL and saved.get('contexts') == len(lp.CASES):
                return saved['moments']
        bank, totals = self.bank, {}
        bank.collect = True
        for index, case in enumerate(lp.CASES):
            bank.sums.clear()
            bank.rows.clear()
            action, future, _ = self.query(case)
            if len(bank.sums) != 614:
                raise RuntimeError('Missing per-context moments')
            for name, value in bank.sums.items():
                totals[name] = totals.get(name, 0) + value.double() / bank.rows[name]
            del action, future
            self.progress('moments', contexts=index + 1)
        bank.collect = False
        moments = {name: (value / len(lp.CASES)).float().cpu() for name, value in totals.items()}
        self.atomic_torch_save(torch, path, dict(protocol=PROTOCOL, contexts=len(lp.CASES), moments=moments))
        return moments

    def fit_banks(self, moments):
        f = self.cfg['fitting']
        self.banks_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, (name, mod) in enumerate(self.bank.modules.items()):
            path = self.banks_dir / f'{index:04d}.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved.get('protocol') != PROTOCOL or saved.get('module') != name or saved.get('module_index') != index:
                    raise RuntimeError('Saved integer bank identity differs: ' + name)
                rows.append(saved['stats'])
                continue
            self.enough()
            tick = time.monotonic()
            with torch.no_grad():
                weight = h_weight(self.masters[name], self.transforms[name], name)
                encoded, extra = fit_int_vq(weight, moments[name].to('cuda'), f['seed'], f['samples'], f['iterations'])
                micro.check_encoding(encoded)
                decoded = micro.decode_vq(encoded).float()
                nmse = float((decoded - weight).square().sum() / weight.square().sum().clamp_min(1e-12))
            t = self.transforms[name]
            stats = dict(module=name, module_index=index, shape=[mod.out_features, mod.in_features],
                         weight_nmse=nmse, delta=extra['delta'], bounds=extra['bounds'],
                         max_used_sum=extra['max_used_sum'],
                         code_bytes=sum(encoded[k].numel() * encoded[k].element_size()
                                        for k in ('books', 'row_scales', 'indices')),
                         transform_bytes=sum(t[k].numel() * t[k].element_size() for k in ('scale', 'signs')),
                         seconds=time.monotonic() - tick)
            self.atomic_torch_save(torch, path, dict(protocol=PROTOCOL, module=name, module_index=index,
                encoded=self.to_cpu(encoded), continuous_books=extra['continuous_books'].cpu(),
                delta=extra['delta'], bounds=extra['bounds'], stats=stats))
            rows.append(stats)
            del weight, encoded, decoded
            self.progress('fit', modules=index + 1)
        weights = sum(r['shape'][0] * r['shape'][1] for r in rows)
        summary = dict(modules=len(rows), target_weights=weights,
                       weighted_weight_nmse=sum(r['weight_nmse'] * r['shape'][0] * r['shape'][1] for r in rows) / weights,
                       max_weight_nmse=max(r['weight_nmse'] for r in rows),
                       format_bpw=8 * sum(r['code_bytes'] for r in rows) / weights,
                       format_bpw_with_transforms=8 * sum(r['code_bytes'] + r['transform_bytes'] for r in rows) / weights,
                       max_used_sum=max(r['max_used_sum'] for r in rows),
                       note='format_bpw excludes the all-127 exponent arrays kept only for decoder compatibility')
        save(self.results / 'fit_stats.json', dict(protocol=PROTOCOL, summary=summary, per_module=rows))
        return summary

    def endpoint(self, label, case, seed):
        path = self.results / 'endpoints' / label / f'{case}_{seed}.json'
        binding = dict(protocol=PROTOCOL, arm=label, split='heldout', case_id=case, seed_index=seed,
                       sampler_seed=self.rows[case]['sampler_seeds'][seed], suite=self.rows[case]['suite'])
        if path.exists():
            row = read(path)
            if any(row.get(k) != v for k, v in binding.items()):
                raise ValueError('Reusable endpoint identity differs')
            return row
        action, future, _ = self.query(case, seed)
        target = self.teacher(case, seed)
        actual = self.runtime.command(action.detach().clone())
        teacher = self.runtime.command(target['action'].detach().clone())
        row = dict(binding, pbs_jobid=self.job, **self.metrics(action, future, target),
                   first10gripper_disagreement=float((actual[:10, 6] != teacher[:10, 6]).mean()))
        save(path, row)
        return row

    def evaluate(self, label):
        ev = self.cfg['evaluation']
        records = []
        for case in ev['case_ids']:
            for seed in ev['seed_indices']:
                records.append(self.endpoint(label, case, seed))
                self.progress('heldout', arm=label, queries=len(records))
        result = aggregate(records, ev['case_ids'], ev['seed_indices'], self.cfg['suites'],
                           tuple(self.helpers.METRICS) + EXTRA)
        save(self.results / 'heldout' / f'{label}.json', dict(protocol=PROTOCOL, arm=label, pbs_jobid=self.job, **result))
        return result

    def prep(self):
        self._runtime()
        self.bank = H8Dense(self.runtime, self.transforms, self.masters)
        self.helpers.Work.reset_rng(self)
        check = self.transform_check()
        moments = self.collect_moments()
        fit = self.fit_banks(moments)
        self.bank = IntBanks(self.runtime, self.banks_dir, self.transforms)
        for value in self.bank.parameters():
            value.requires_grad_(False)
        case = lp.CASES[0]
        action, future, counts = self.query(case)
        values = self.metrics(action, future, self.teacher(case))
        save(self.stage_dir / 'init_anchor.json', dict(protocol=PROTOCOL, case_id=case, seed_index=0, **values))
        self.atomic_torch_save(torch, self.stage_dir / 'init_anchor.pt',
                               dict(protocol=PROTOCOL, action=action.cpu(), future=future.cpu(), metrics=values))
        heldout = self.evaluate('VQ0_prime')
        save(self.stage_dir / 'summary.json', dict(
            protocol=PROTOCOL, stage='prep', pbs_jobid=self.job, status='complete', complete=True,
            transform_check=check, fit=fit, init_anchor=values, heldout_VQ0_prime=heldout['overall'],
            counters=self.counters, new_teacher_queries=0, training_updates=0,
            seconds=time.monotonic() - self.started))
        self.progress('complete')

    def export(self):
        import export_runtime as ex
        out = self.export_dir
        manifest_path = out / 'manifest.json'
        if manifest_path.exists():
            return read(manifest_path)
        out.mkdir(parents=True, exist_ok=True)
        runtime, modules = self.runtime, self.runtime.q.modules
        excluded = ex._target_state_keys(runtime.model, modules)
        state = runtime.model.state_dict()
        e = self.cfg['export']
        memo = {}
        with torch.no_grad():
            # Same container as the Q0 bundle so the engine evaluator accepts it; U is the identity.
            bundle = dict(format=ex.FORMAT, variant=e['format'], protocol=PROTOCOL, source_job=self.job, arm='reader_kl',
                training_updates=8, use_u=False, integer_codebooks=True,
                activation=dict(bits=8, signed_range=[-127, 127], dynamic_axis=-1, scale='maxabs/127',
                                rounding='half_to_even', output_dtype='bfloat16'),
                execution='BF16 GEMM on decoded weights; codes are INT8-compatible (pair sums within +-127)',
                quantized={}, transforms={}, excluded_model_state_keys=excluded,
                rotations={s: torch.eye(128, dtype=torch.float32) for s in ('action', 'video')},
                layer_rotations=({n: self.bank.rotation(n).detach().cpu() for n in modules}
                                 if isinstance(self.bank, LearnedHBanks) else None),
                model_state=ex._cpu_copy({k: v for k, v in state.items() if k not in excluded}, torch, memo),
                processor_stats=ex._cpu_copy(runtime.processor.normalizer.stats, torch, memo),
                processor_state=(ex._cpu_copy(runtime.processor.state_dict(), torch, memo)
                                 if hasattr(runtime.processor, 'state_dict') else {}),
                assets=ex._deployment_assets(runtime))
            for index, (name, mod) in enumerate(modules.items()):
                encoding = self.bank.stored_encoding(name)
                micro.check_encoding(encoding)
                bundle['quantized'][name] = dict(module_index=index, shape=tuple(encoding['shape']),
                    format=encoding['format'], block_size=4, scale_block=32,
                    bounds=[int(v) for v in self.bank.bound[name].flatten().tolist()],
                    **ex._cpu_copy({k: encoding[k] for k in ex.ENCODING_KEYS}, torch, memo))
                bundle['transforms'][name] = ex._cpu_copy({k: self.transforms[name][k] for k in
                    ('scale', 'signs', 'hadamard', 'block', 'input_dim')}, torch, memo)
        path = out / 'inference_bundle.pt'
        temp = path.with_suffix('.pt.tmp')
        torch.save(bundle, temp)
        temp.replace(path)
        manifest = dict(protocol=PROTOCOL, format=e['format'], bundle=str(path), bytes=path.stat().st_size,
                        anchor=str(self.unit / 'anchor.pt'), anchor_case_id=lp.CASES[0], anchor_seed_index=0,
                        target_modules=len(modules), use_u=False, activation_bits=8,
                        layer_rotations=isinstance(self.bank, LearnedHBanks), engine_notes=e['engine_notes'])
        save(manifest_path, manifest)
        return manifest

    def train_stage(self):
        prep = read(self.results / 'stages/prep/summary.json')
        if prep.get('protocol') != PROTOCOL or not prep.get('complete'):
            raise ValueError('Prep stage is not complete')
        trained = self.train()  # loss-pair loop: init check vs prep anchor, beta, 8 updates, final anchor
        changed = self.bank.changed_codewords()
        heldout = self.evaluate('H_INT8VQ_A8_KL')
        manifest = self.export()
        beta = read(self.out / 'reader_preparation/summary.json')
        summary = dict(protocol=PROTOCOL, stage='train', pbs_jobid=self.job, status='complete', complete=True,
                       beta=trained['beta'], initial_means=beta['initial_means'], codeword_changes=changed,
                       heldout_VQ0_prime=prep['heldout_VQ0_prime'], heldout_trained=heldout['overall'],
                       q0_reference=self.cfg['q0_reference']['overall'], fit=prep['fit'],
                       formal_parity_max_abs=trained['formal_parity_max_abs'], export=manifest,
                       counters=self.counters, new_teacher_queries=0, training_updates=8,
                       seconds=time.monotonic() - self.started)
        save(self.stage_dir / 'summary.json', summary)
        (self.stage_dir / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')
        self.progress('complete')


    def train_lh_stage(self):
        prep = read(self.results / 'stages/prep/summary.json')
        first = read(self.results / 'stages/train/summary.json')
        if prep.get('protocol') != PROTOCOL or not prep.get('complete') or not first.get('complete'):
            raise ValueError('Prep and 1b train must be complete')
        trained = self.train()
        changed = self.bank.changed_codewords()
        angles = self.bank.angle_stats()
        save(self.unit / 'angles.json', dict(protocol=PROTOCOL, **angles))
        heldout = self.evaluate('LH_INT8VQ_A8_KL')
        manifest = self.export()
        beta = read(self.out / 'reader_preparation/summary.json')
        summary = dict(protocol=PROTOCOL, stage='train_lh', pbs_jobid=self.job, status='complete', complete=True,
                       beta=trained['beta'], initial_means=beta['initial_means'], codeword_changes=changed,
                       angles=angles, angle_scale=ANGLE_SCALE,
                       heldout_VQ0_prime=prep['heldout_VQ0_prime'], heldout_1b=first['heldout_trained'],
                       heldout_trained=heldout['overall'], q0_reference=self.cfg['q0_reference']['overall'],
                       fit=prep['fit'], formal_parity_max_abs=trained['formal_parity_max_abs'], export=manifest,
                       counters=self.counters, new_teacher_queries=0, training_updates=8,
                       seconds=time.monotonic() - self.started)
        save(self.stage_dir / 'summary.json', summary)
        (self.stage_dir / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')
        self.progress('complete')


    def fit_g_banks(self, moments):
        f = self.cfg['fitting']
        self.banks_g_dir.mkdir(parents=True, exist_ok=True)
        rows, rotations = [], {}
        for index, (name, mod) in enumerate(self.bank_modules.items()):
            path = self.banks_g_dir / f'{index:04d}.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved.get('protocol') != PROTOCOL or saved.get('module') != name or saved.get('module_index') != index:
                    raise RuntimeError('Saved G bank identity differs: ' + name)
                rows.append(saved['stats'])
                rotations[name] = saved['rotation']
                continue
            self.enough()
            tick = time.monotonic()
            h = torch.load(self.banks_dir / f'{index:04d}.pt', map_location='cuda', weights_only=True)
            moment = moments[name].to('cuda')
            weight = h_weight(self.masters[name], self.transforms[name], name)
            error_h = weighted_error(weight, micro.decode_vq(h['encoded']).float(), moment)
            with torch.enable_grad():
                g, theta, history = learn_g(weight, moment, h['continuous_books'])
            with torch.no_grad():
                wg, mg = rotate_columns(weight, g), rotate_moment(moment, g)
                encoded, extra = fit_int_vq(wg, mg, f['seed'], f['samples'], f['iterations'])
                micro.check_encoding(encoded)
                decoded = micro.decode_vq(encoded).float()
                error_g = weighted_error(wg, decoded, mg)
                nmse_g = float((decoded - wg).square().sum() / wg.square().sum().clamp_min(1e-12))
            accepted = error_g < error_h
            if not accepted:
                g, theta = torch.eye(128, device='cuda'), torch.zeros_like(theta)
                encoded, extra = h['encoded'], dict(continuous_books=h['continuous_books'], delta=h['delta'],
                                                    bounds=h['bounds'], max_used_sum=h['stats']['max_used_sum'])
            stats = dict(h['stats'], weighted_error_h=error_h, weighted_error_g=error_g, weight_nmse_g=nmse_g,
                         accepted=accepted, proxy_history=history,
                         rotation_distance=float((g - torch.eye(128, device='cuda')).norm() / 128 ** .5),
                         weight_nmse=nmse_g if accepted else h['stats']['weight_nmse'],
                         delta=extra['delta'], bounds=extra['bounds'], max_used_sum=extra['max_used_sum'],
                         seconds=time.monotonic() - tick)
            self.atomic_torch_save(torch, path, dict(protocol=PROTOCOL, module=name, module_index=index,
                encoded=self.to_cpu(encoded), continuous_books=extra['continuous_books'].cpu(),
                delta=extra['delta'], bounds=extra['bounds'], rotation=g.cpu(), angles=theta.cpu(), stats=stats))
            rows.append(stats)
            rotations[name] = g.cpu()
            del weight, wg, encoded, decoded, h
            self.progress('fit_g', modules=index + 1, accepted=sum(r['accepted'] for r in rows))
        self.atomic_torch_save(torch, self.banks_g_dir / 'rotations.pt', dict(protocol=PROTOCOL, rotations=rotations))
        weights = sum(r['shape'][0] * r['shape'][1] for r in rows)
        mean = lambda k: sum(r[k] * r['shape'][0] * r['shape'][1] for r in rows) / weights
        summary = dict(modules=len(rows), accepted=sum(r['accepted'] for r in rows),
                       weighted_error_h=mean('weighted_error_h'),
                       weighted_error_kept=sum(min(r['weighted_error_h'], r['weighted_error_g']) * r['shape'][0]
                                               * r['shape'][1] for r in rows) / weights,
                       weight_nmse=mean('weight_nmse'),
                       mean_rotation_distance=sum(r['rotation_distance'] for r in rows) / len(rows),
                       format_bpw=8 * sum(r['code_bytes'] for r in rows) / weights,
                       g_recipe=dict(rounds=G_ROUNDS, steps=G_STEPS, lr=G_LR, rows=G_ROWS, seed=G_SEED))
        save(self.results / 'fit_g_stats.json', dict(protocol=PROTOCOL, summary=summary, per_module=rows))
        return summary

    def c8_stage(self):
        prep = read(self.results / 'stages/prep/summary.json')
        first = read(self.results / 'stages/train/summary.json')
        if prep.get('protocol') != PROTOCOL or not prep.get('complete') or not first.get('complete'):
            raise ValueError('Prep and 1b train must be complete')
        g_arm = self.stage_name == 'c8_g'
        label = 'C8_G' if g_arm else 'C8_H'
        self._runtime()
        self.bank_modules = self.runtime.q.modules
        extra = {}
        if g_arm:
            moments = torch.load(self.results / 'moments.pt', map_location='cpu', weights_only=True)
            if moments.get('protocol') != PROTOCOL or moments.get('contexts') != len(lp.CASES):
                raise ValueError('C80 W16A8 moments differ')
            extra['fit_g'] = self.fit_g_banks(moments['moments'])
            self.bank = FixedGBanks(self.runtime, self.banks_g_dir, self.transforms)
        else:
            self.bank = IntBanks(self.runtime, self.banks_dir, self.transforms)
        if len(self.bank.parameters()) != 1228 or any(v.requires_grad for v in self.runtime.model.parameters()):
            raise ValueError('Only 614 integer codebook/row-scale pairs may train')
        self.helpers.Work.reset_rng(self)
        for value in self.bank.parameters():
            value.requires_grad_(False)
        action, future, _ = self.query(C8[0])
        initial = self.metrics(action, future, self.teacher(C8[0]))
        del action, future
        if not g_arm:
            expected = read(self.results / 'stages/prep/init_anchor.json')
            error = max(abs(initial[k] - expected[k]) for k in self.helpers.METRICS)
            if error > 1e-6:
                raise ValueError(f'H-arm initialization differs from the prep anchor: {error}')
        else:
            extra['heldout_VQ0_G'] = self.evaluate('VQ0_G')['overall']
        for value in self.bank.parameters():
            value.requires_grad_(True)
        beta_prep = self.preparation()  # the existing C80 beta, identical frozen protocol for both arms
        beta = beta_prep['beta']
        optimizer = torch.optim.Adam(self.bank.parameters(), lr=self.p['training']['vq_learning_rate'])
        state = dict(updates=0, history=[], lineage=[])
        checkpoint = self.unit / 'checkpoint.pt'
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if payload['label'] != label or payload['beta'] != beta:
                raise ValueError('Committed C8 recipe differs')
            state = payload['state']
            self.bank.restore(payload['module_masters'])
            optimizer.load_state_dict(payload['optimizer'])
            self.optimizer_to_device(torch, optimizer, 'cuda')
            self.restore_rng(torch, payload['rng'])
        if self.job not in state['lineage']:
            state['lineage'].append(self.job)
        # C8 recipe: eight Adam updates, one context each, first-update optimizer reset, same lr/clip as 1b.
        while state['updates'] < 8:
            self.enough()
            update, started = state['updates'], time.monotonic()
            case = C8[update]
            self.progress('training', updates=update, case_id=case, beta=beta)
            optimizer.zero_grad(set_to_none=True)
            metrics, reader, replay = self.kl_batch([case], beta, beta_prep['stats'], update == 0)
            coverage = self.bank.gradient_report()
            if coverage['connected'] != 614 or coverage['finite'] != 614 or coverage['nonzero'] < 1:
                raise ValueError('VQ gradient coverage failed')
            if any(v.grad is not None for v in self.runtime.model.parameters()):
                raise ValueError('Frozen original model accumulated gradients')
            norm = torch.nn.utils.clip_grad_norm_(self.bank.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if update == 0:
                optimizer.state.clear()
            self.bank.refresh()
            state['updates'] += 1
            # kl_batch averages reader terms over a fixed 10; this batch has one context.
            state['history'].append(dict(update=state['updates'], case_id=case, before=metrics[0], beta=beta,
                                         **{k: v * 10 for k, v in reader.items()}, vq_gradient_norm=float(norm),
                                         staged_replay_max_abs=replay, seconds=time.monotonic() - started))
            self.atomic_torch_save(torch, checkpoint, dict(label=label, beta=beta, state=state,
                module_masters=self.bank.snapshot(), optimizer=self.to_cpu(optimizer.state_dict()),
                rng=self.capture_rng(torch)))
            save(self.unit / 'training.json', dict(protocol=PROTOCOL, arm=label, **state))
        for value in self.bank.parameters():
            value.requires_grad_(False)
            value.grad = None
        optimizer.state.clear()
        changed = self.bank.changed_codewords()
        heldout = self.evaluate(label)
        summary = dict(protocol=PROTOCOL, stage=self.stage_name, pbs_jobid=self.job, status='complete',
                       complete=True, arm=label, calibration_case_ids=C8, beta=beta, initial_c8_case=initial,
                       codeword_changes=changed, heldout_VQ0_prime=prep['heldout_VQ0_prime'],
                       heldout_1b_c80=first['heldout_trained'], heldout_trained=heldout['overall'],
                       q0_reference=self.cfg['q0_reference']['overall'], fit_h=prep['fit'], **extra,
                       counters=self.counters, new_teacher_queries=0, training_updates=8,
                       seconds=time.monotonic() - self.started)
        save(self.stage_dir / 'summary.json', summary)
        (self.stage_dir / 'RESULTS.zh.md').write_text(c8_report(summary), encoding='utf-8')
        self.progress('complete')


def c8_report(s):
    keys = ('first10motor_rmse', 'future_latent_nmse', 'full32motor_rmse', 'first10gripper_rmse',
            'first10gripper_disagreement')
    rows = [('Q0（R0+MXVQ+A4，参照）', s['q0_reference']), ("VQ0′（H 坐标拟合，未校准）", s['heldout_VQ0_prime']),
            ('1b（H，C80 校准）', s['heldout_1b_c80'])]
    if 'heldout_VQ0_G' in s:
        rows.append(('VQ0_G（G 坐标重拟合，未校准）', s['heldout_VQ0_G']))
    rows.append((f'{s["arm"]}（C8 校准 8 步）', s['heldout_trained']))
    lines = [f'# C8 决策实验：`{s["arm"]}`', '',
             f'作业 `{s["pbs_jobid"]}`。held-out 40 contexts × 2 seeds。', '',
             '| 配置 | ' + ' | '.join(keys) + ' |', '|---|' + '---:|' * len(keys)]
    lines += ['| ' + label + ' | ' + ' | '.join(f'{v[k]:.6f}' for k in keys) + ' |' for label, v in rows]
    c = s['codeword_changes']
    lines += ['', f'- β = {s["beta"]:.6g}；整数 codebook 改变条目 {c["changed_integer_entries"]}/{c["total_integer_entries"]}']
    if 'fit_g' in s:
        lines.append(f'- G 拟合：{json.dumps(s["fit_g"], ensure_ascii=False)}')
    lines.append('')
    return '\n'.join(lines)


def report(s):
    keys = ('first10motor_rmse', 'future_latent_nmse', 'full32motor_rmse', 'late22motor_rmse',
            'first10gripper_rmse', 'first10gripper_disagreement')
    rows = (('Q0（R0+MXVQ+A4，参照）', s['q0_reference']), ("VQ0′（整数 VQ 拟合，未校准）", s['heldout_VQ0_prime']),
            ('1b（A+Z+K 8 步）', s.get('heldout_1b', s['heldout_trained'])))
    if 'angles' in s:
        rows += (('1b + 可学 H（butterfly，A+Z+K 8 步）', s['heldout_trained']),)
    title = '1b + 可学 H' if 'angles' in s else '1b：H 坐标 + 整数 codebook VQ + A8，A+Z+K'
    lines = ['# ' + title, '',
             f'作业 `{s["pbs_jobid"]}`。held-out 40 contexts × 2 seeds，聚合 seed→context→suite→四 suite 等权。', '',
             '| 配置 | ' + ' | '.join(keys) + ' |', '|---|' + '---:|' * len(keys)]
    lines += ['| ' + label + ' | ' + ' | '.join(f'{v[k]:.6f}' for k in keys) + ' |' for label, v in rows]
    f, c = s['fit'], s['codeword_changes']
    lines += ['', f'- β = {s["beta"]:.6g}；初始 C80 均值：{json.dumps(s["initial_means"])}',
              f'- 整数 codebook 训练中改变的条目：{c["changed_integer_entries"]}/{c["total_integer_entries"]}',
              f'- 拟合：加权 weight NMSE {f["weighted_weight_nmse"]:.5f}，格式 bpw {f["format_bpw"]:.4f}'
              f'（含 Smooth/signs {f["format_bpw_with_transforms"]:.4f}），最大使用的码字和 {f["max_used_sum"]}',
              f'- 导出：`{s["export"]["bundle"]}`；engine anchor：`{s["export"]["anchor"]}`']
    if 'angles' in s:
        lines.append(f'- 角度：{json.dumps(s["angles"])}（ANGLE_SCALE {s["angle_scale"]}）')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--stage', choices=('prep', 'train', 'train_lh', 'c8_h', 'c8_g'), required=True)
    args = parser.parse_args()
    cfg = read(args.protocol)
    if cfg['protocol'] != PROTOCOL:
        raise ValueError('Protocol identity differs')
    try:
        work = Work(cfg, args.stage)
        {'prep': work.prep, 'train': work.train_stage, 'train_lh': work.train_lh_stage,
         'c8_h': work.c8_stage, 'c8_g': work.c8_stage}[args.stage]()
    except BaseException as error:
        save(Path(cfg['remote_root']) / 'results/stages' / args.stage / 'failure.json',
             dict(protocol=PROTOCOL, stage=args.stage, pbs_jobid=os.environ.get('PBS_JOBID'),
                  error=type(error).__name__, message=str(error), checkpointed=isinstance(error, TimeoutError)))
        raise


if __name__ == '__main__':
    main()
