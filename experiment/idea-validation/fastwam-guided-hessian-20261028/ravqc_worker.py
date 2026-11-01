"""RAVQ calibrated (ravqc.json): the reader metric M on the imagined future (1) and the calibration of QTIP's own shared
parameters on A + lam_M Z_M + beta K (4), on top of the uncalibrated RAVQ arms (results/ravq, QTIP rounding under the C1
module budget).

QTIP decodes a pair of weights of the rotated matrix from a 16-bit trellis state: entry k = hash(state) of a 512 x 2
table, the first coordinate's sign from another hash bit, times the layer's Wscale; W = diag(SV) H_m^T (Wscale D) H_n
diag(SU). The calibrated parameters are that table (one offset table per (role, K) group: a codebook shared by every layer
of a module at one bit width), the row / column scales (SV / SU) and the output biases:
  W' = bf16(e^s * (W_q + diag(SV) H_m^T (Wscale D(d)) H_n diag(SU)) * e^t),  b' = bf16(b + db),
W' = W_q bitwise at zero. QTIPGrad runs the plain forward and reduces the gradient onto the families in its backward.
Stages (FW_STAGE, shard FW_SHARD of FW_SHARDS):
  codes        pair codes (entry | sign << 9, int16 [tiles, 128]) of every QTIP-coded layer of the base arms
  metric       (1) per C80 context: P probes c = J_R(z_T)^T (eps sqrt(h/2)) of the BF16 reader at the teacher future
  metric_final the normalized metric (top-r eigenpairs + residual diagonal) and its split-half stability
  calib        one base arm per shard: lam_M rule (C80 at the start), step check (= update 1), 7 more updates on the
               loss-pair C80 schedule, deploy (dense BF16 weights + biases), C80 fit after, held-out 40x2 + swap
  zm           Z_M on the held-out panel of the reference arms (query only; NMSE checked against their endpoints)
  final        results/ravqc/{summary.json, RESULTS.zh.md, DECISION.json}
  loop_final   closed-loop pairing of the calibrated arms
  gradtx       diagnostic, one arm per job (gradtx_arm): per C80 context the table gradient of the action part and of
               the future part at the calibration start (two grad replays of one staged query)
  gradtx_final their cross-context transfer (transfer_stats) -> results/ravqc/gradtx/<arm>
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import re
import time

import gate_worker as gw  # owns the allocation guard (via worker), the frozen-reader context and the masters
import torch
import torch.nn.functional as F

import lp_worker as lp
import probe as pr
import ravq_qtip as rq
import ravq_worker as rw
import softbin_loss
import worker as wk

HERE = Path(__file__).resolve().parent
RC = json.loads((HERE / 'ravqc.json').read_text(encoding='utf-8'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ARMS = RC['base_arms']
CODED_ROLES = tuple(RC['families']['roles'])
ENTRIES, TILE = 512, 256
FUTURE_FRAMES = slice(1, None)
EXTRA_METRICS = ('future_metric_nmse',)
_HADK = {}


# ---------------- QTIP decode / Hadamard pieces (pure functions) ----------------
def had(Q, X, transpose=False):
    """Q.matmul_hadU(X, transpose) without its per-call gc.collect / empty_cache: X H (transpose: X H^T) over the last
    dim, normalized. Same butterflies; the K x K factor by a broadcast matmul."""
    n = X.shape[-1]
    key = (n, transpose, str(X.device), X.dtype)
    if key not in _HADK:
        hadK, K = Q.get_hadK(n, transpose)
        _HADK[key] = (hadK.to(X.device, X.dtype) if K > 1 else None, K)
    hadK, K = _HADK[key]
    a = X.clone().reshape(-1, n, 1)
    b = a.clone()
    while a.shape[1] > K:
        a = a.view(a.shape[0], a.shape[1] // 2, 2, a.shape[2])
        b = b.view(a.shape)
        b[:, :, 0, :] = a[:, :, 0, :] + a[:, :, 1, :]
        b[:, :, 1, :] = a[:, :, 0, :] - a[:, :, 1, :]
        b = b.view(a.shape[0], a.shape[1], -1)
        a, b = b, a
    if K > 1:
        a = torch.matmul(hadK, a)
    return a.reshape(X.shape) / math.sqrt(n)


def incoherent(Q, R, SU, SV):
    """diag(SV) H_m^T R H_n diag(SU) (the inverse incoherence of QTIP, fast form)."""
    return had(Q, had(Q, R).t().contiguous()).t() * SV[:, None] * SU[None, :]


def coherent(Q, W, SU, SV):
    """H_m diag(SV) W diag(SU) H_n^T (QTIP's forward incoherence)."""
    return had(Q, had(Q, (W.t() * SV[None, :]).contiguous(), True).t().contiguous() * SU[None, :], True)


def states_to_codes(states, tlut_bits=9):
    """quantlut_sym: v = (s + 1) s; sign flip of coordinate 0 from bit 15, entry from bits 6..14 (L = 16, Q = 9)."""
    s = states.long()
    v = (s + 1) * s
    flip = (v >> 15) & 1
    idx = (v >> (16 - tlut_bits - 1)) & ((1 << tlut_bits) - 1)
    return (idx | (flip << 9)).to(torch.int16)


def trellis_codes(Q, codes):
    """Pair codes [tiles, 128] from a stored QTIP trellis (ravq_qtip codes dict), tiles in decode_packed's order."""
    m, n = codes['padded_shape']
    K = codes['K']
    t = codes['trellis'].cpu()
    if codes['kernel_layout']:
        t = Q.unswizzle_kernel_layout(t, m, n, K)
    states = Q.codebook(K, 'cpu').unpack_trellis(t, TILE)
    return states_to_codes(states, Q.TLUT_BITS)


def lut_values(tlut):
    """The values decode_packed uses: the stored BF16 table through float16."""
    return tlut.float().to(torch.float16).float()


def decode(pairs, table, m, n, permute_inv):
    """D [m, n]: pair k of tile t holds (sign * table[e, 0], table[e, 1]), e = code & 511, sign from bit 9; sequence
    order -> row-major 16 x 16 tile (ldlq._INV_PERMUTE) -> matrix (decode_packed's layout). Linear in table."""
    p = pairs.long()
    idx, flip = p & (ENTRIES - 1), (p >> 9) & 1
    v0 = table[idx, 0] * (1 - 2 * flip).to(table.dtype)
    v1 = table[idx, 1]
    seq = torch.stack((v0, v1), -1).reshape(-1, TILE)
    tile = seq[:, permute_inv.to(seq.device)]
    return tile.reshape(m // 16, n // 16, 16, 16).transpose(1, 2).reshape(m, n)


def decode_adjoint(dD, pairs, permute, out):
    """out [512, 2] += the adjoint of decode at dD (the gradient on the table)."""
    m, n = dD.shape
    tiles = dD.reshape(m // 16, 16, n // 16, 16).transpose(1, 2).reshape(-1, TILE)
    seq = tiles[:, permute.to(tiles.device)].reshape(-1, TILE // 2, 2)
    p = pairs.to(torch.int32)
    idx, sign = p & (ENTRIES - 1), (1 - 2 * ((p >> 9) & 1)).to(dD.dtype)
    vals = torch.stack((seq[..., 0] * sign, seq[..., 1]), -1)
    where = (idx * 2)[..., None] + torch.arange(2, device=idx.device, dtype=torch.int32)
    out.view(-1).index_add_(0, where.reshape(-1), vals.reshape(-1).to(out.dtype))
    return out


def nearest_codes(pairs_values, table, chunk=1 << 19):
    """Nearest of the 1024 (sign, entry) candidates per pair value [..., 2]; returns codes and the margin (second
    nearest distance - nearest distance)."""
    cand = torch.cat((torch.stack((table[:, 0], table[:, 1]), -1), torch.stack((-table[:, 0], table[:, 1]), -1)))
    codes = torch.arange(2 * ENTRIES, device=table.device)
    codes = (codes % ENTRIES) | ((codes // ENTRIES) << 9)
    flat = pairs_values.reshape(-1, 2)
    out = torch.empty(flat.shape[0], dtype=torch.int16, device=flat.device)
    margin = torch.empty(flat.shape[0], dtype=flat.dtype, device=flat.device)
    for i in range(0, flat.shape[0], chunk):
        d = torch.cdist(flat[i:i + chunk], cand.to(flat.dtype))
        top = d.topk(2, dim=1, largest=False)
        out[i:i + chunk] = codes[top.indices[:, 0]].to(torch.int16)
        margin[i:i + chunk] = top.values[:, 1] - top.values[:, 0]
    return out.reshape(pairs_values.shape[:-1]), margin


def to_pairs(D, permute):
    """[m, n] -> [tiles, 128, 2] in sequence order (the inverse of decode's layout)."""
    m, n = D.shape
    tiles = D.reshape(m // 16, 16, n // 16, 16).transpose(1, 2).reshape(-1, TILE)
    return tiles[:, permute.to(tiles.device)].reshape(-1, TILE // 2, 2)


# ---------------- the frozen reader under several backward passes ----------------
class Reentrant:
    """A context manager that builds a fresh one from `make` at every entry. Non-reentrant checkpointing enters a
    checkpoint's recompute context once per backward pass; the metric probes run several backward passes over one reader
    graph, and the frozen reader's own recompute context is a one-shot generator."""

    def __init__(self, make):
        self.make, self.open = make, []

    def __enter__(self):
        cm = self.make()
        self.open.append(cm)
        return cm.__enter__()

    def __exit__(self, *exc):
        return self.open.pop().__exit__(*exc)


def reusable_checkpoint_contexts(modules, original_weights, state):
    """The frozen reader's checkpoint contexts (nothing on the forward, the BF16 masters on the recompute), with the
    recompute context re-enterable."""
    return lambda: (nullcontext(), Reentrant(lambda: gw.fr._recompute_bf16_context(modules, original_weights, state)))


# ---------------- the reader metric (1) ----------------
class Metric:
    """M = (1 - rho) (D / tau) (E^T diag(lam) E + diag(d)) + rho I on frames 1: of the future, flattened."""

    def __init__(self, saved, device=DEVICE):
        self.E = saved['E'].to(device, torch.float32)
        self.lam = saved['lam'].to(device, torch.float32)
        self.d = saved['d_res'].to(device, torch.float32)
        self.rho, self.D, self.tau = float(saved['rho']), int(saved['D']), float(saved['tau'])
        self.c = (1 - self.rho) * self.D / self.tau

    def quad(self, v):
        v = v.reshape(-1).to(self.E.dtype)
        p = self.E @ v
        return self.c * ((self.lam * p.square()).sum() + (self.d * v.square()).sum()) + self.rho * v.square().sum()

    def nmse(self, future, target):
        t = target[:, :, FUTURE_FRAMES].to(future.device, torch.float32)
        e = future[:, :, FUTURE_FRAMES].float() - t
        return self.quad(e) / t.square().sum().clamp_min(1e-12)


def build_metric(vectors, contexts, rank, rho):
    """vectors[c]: [P, D] probes of context c. Per-context unit trace, mean over contexts; top-r eigenpairs of the mean
    and the residual diagonal. Returns the saved form and the spectrum statistics."""
    rows = []
    for v in vectors:
        tr = v.square().sum(1).mean().clamp_min(1e-30)
        rows.append(v / (v.shape[0] * tr).sqrt())
    V = torch.cat(rows).double()
    N = len(vectors)
    diag = V.square().sum(0) / N
    _, S, Vh = torch.linalg.svd(V / math.sqrt(N), full_matrices=False)
    lam = S[:rank].square()
    E = Vh[:rank]
    d_res = (diag - (lam[:, None] * E.square()).sum(0)).clamp_min(0)
    tau = float(lam.sum() + d_res.sum())
    gram = V @ V.t() / N
    stats = dict(trace=float(diag.sum()), top_share=float(lam.sum() / diag.sum()),
                 effective_rank=float(diag.sum() ** 2 / gram.square().sum()), spectrum=[float(x) for x in S[:2 * rank].square()],
                 diag_top10_share=float(diag.sort(descending=True).values[:max(1, diag.numel() // 10)].sum() / diag.sum()))
    saved = dict(E=E.float(), lam=lam.float(), d_res=d_res.float(), rho=rho, D=V.shape[1], tau=tau, rank=rank,
                 contexts=list(contexts))
    return saved, stats, diag.float()


def split_stability(vectors, contexts, rank):
    """Even / odd case halves: Pearson correlation of the diagonals and the mean squared cosine of the top-r subspaces."""
    halves = []
    for parity in (0, 1):
        sel = [v for v, c in zip(vectors, contexts) if c % 2 == parity]
        saved, _, diag = build_metric(sel, [c for c in contexts if c % 2 == parity], rank, 0.0)
        halves.append((diag.double(), saved['E'].double()))
    (d0, e0), (d1, e1) = halves
    a, b = d0 - d0.mean(), d1 - d1.mean()
    corr = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
    overlap = float((e0 @ e1.t()).square().sum() / rank)
    return dict(diag_corr=corr, subspace_overlap=overlap)


# ---------------- calibrated layers ----------------
class QLayer:
    """One QTIP-coded Linear of the calibrated arm: the arm's stored BF16 weight W0 and original bias b0, its pair codes
    and incoherence (SU, SV, Wscale), the FP32 parameters s, t, db; the group's table d is shared."""

    def __init__(self, index, name, mod, group, saved, Q, permute, permute_inv):
        self.index, self.name, self.mod, self.group, self.Q = index, name, mod, group, Q
        self.permute, self.permute_inv = permute, permute_inv
        self.m, self.n = mod.weight.shape
        self.W0 = mod.weight.detach().to('cpu').clone()
        self.pairs = saved['pairs'].contiguous()
        if torch.cuda.is_available():
            self.pairs = self.pairs.pin_memory()
        self.Wscale = float(saved['Wscale'])
        self.dtype = torch.float64 if 'float64' in str(saved.get('dtype', '')) else torch.float32  # the codes stage's decode
        self.SU, self.SV = (saved[k].to(DEVICE, torch.float32) for k in ('SU', 'SV'))
        w = self.W0.to(DEVICE, torch.float32)
        self.rn, self.cn = w.norm(dim=1).clamp_min(1e-30), w.norm(dim=0).clamp_min(1e-30)
        self.wnorm = float(w.norm())
        del w
        self.has_bias = mod.bias is not None
        self.b0 = mod.bias.detach().clone() if self.has_bias else None
        self.s = torch.zeros(self.m, device=DEVICE, requires_grad=True)
        self.t = torch.zeros(self.n, device=DEVICE, requires_grad=True)
        self.db = torch.zeros(self.m, device=DEVICE, requires_grad=self.has_bias)
        self.xsq = torch.zeros((), device=DEVICE, dtype=torch.float64)
        self.tokens, self.bias_ref = 0, None

    def pairs_device(self):
        return self.pairs.to(DEVICE, non_blocking=True)

    def decoded(self, table):
        """diag(SV) H_m^T (Wscale D(table)) H_n diag(SU) in the codes stage's precision, as FP32."""
        D = decode(self.pairs_device(), table.to(self.dtype), self.m, self.n, self.permute_inv)
        return incoherent(self.Q, D * self.Wscale, self.SU.to(self.dtype), self.SV.to(self.dtype)).float()

    def rebuild(self, table, base):
        """W' and b' from the current parameters into the module: W' = bf16(e^s (W_hi + dW_lut) e^t), W_hi the FP32
        decode of the layer's codes at the base table (its BF16 rounding is W0, checked bit for bit in the codes stage).
        Folding onto W0 itself rounded twice: a change under half a BF16 spacing fell back onto W0 (fwrc2r2 26074313);
        from W_hi it moves a proportional share of the elements by one spacing. W' = W0, b' = b0 bitwise at zero."""
        with torch.no_grad():
            moved = table is not None and bool(table.ne(0).any())
            scaled = bool(self.s.ne(0).any()) or bool(self.t.ne(0).any())
            if moved or scaled:
                w0 = self.W0.to(DEVICE)
                w = self.decoded(base)
                w = torch.where(w.to(w0.dtype) == w0, w, w0.float())  # an element the fast form rounds elsewhere: W0
                if moved:
                    w = w + self.decoded(table)
                if scaled:
                    w = self.s.exp()[:, None] * w * self.t.exp()[None, :]
            else:
                w = self.W0
            self.mod.weight.data.copy_(w.to(self.mod.weight.dtype))
            if self.has_bias:
                self.mod.bias.data.copy_((self.b0.float() + self.db).to(self.b0.dtype))


class QTIPGrad(torch.autograd.Function):
    """y = F.linear(x, W', b') (the plain forward, bitwise); backward, straight through the BF16 casts:
    ds_r = sum_t g_tr (x_t W'_r), dt_c = sum_t x_tc (g_t W')_c, db = sum_t g_t,
    dD = Wscale hadUt(g o SV e^s)^T hadUt(x o SU e^t) reduced onto the table through the pair codes."""

    @staticmethod
    def forward(ctx, x, s, t, db, table, layer):
        ctx.layer = layer
        ctx.save_for_backward(x, s, t)
        return F.linear(x, layer.mod.weight, layer.mod.bias)

    @staticmethod
    def backward(ctx, g):
        x, s, t = ctx.saved_tensors
        L = ctx.layer
        w = L.mod.weight
        need = ctx.needs_input_grad
        gx = g.matmul(w).to(x.dtype)
        g2 = g.reshape(-1, L.m).float()
        x2 = x.reshape(-1, L.n).float()
        L.xsq += x2.square().sum().double()
        L.tokens += x2.shape[0]
        ds = (g2 * F.linear(x, w).reshape(-1, L.m).float()).sum(0) if need[1] else None
        dt = (x2 * gx.reshape(-1, L.n).float()).sum(0) if need[2] else None
        db = g2.sum(0) if need[3] else None
        dtab = None
        if need[4]:
            gh = had(L.Q, g2 * (L.SV * s.detach().exp())[None, :], True)
            xh = had(L.Q, x2 * (L.SU * t.detach().exp())[None, :], True)
            dD = (gh.t() @ xh).mul_(L.Wscale)
            dtab = decode_adjoint(dD, L.pairs_device(), L.permute, torch.zeros(ENTRIES, 2, device=g.device))
            del gh, xh, dD
        return (gx if need[0] else None), ds, dt, db, dtab, None


def first_step_choice(rows, fraction):
    ok = [r['lr'] for r in rows if r['predicted'] > 0 and r['realized'] >= fraction * r['predicted']]
    return max(ok) if ok else None


class RavqcWork(rw.RavqWork):
    def rc(self, *parts):
        return self.base.joinpath('ravqc', *parts)

    @contextmanager
    def bf16_reader(self):
        """The gate's frozen BF16 reader with a re-enterable recompute context (the metric probes backward through one
        reader graph several times)."""
        with super().bf16_reader() as reader:
            saved, gw.fr._checkpoint_context_fn = gw.fr._checkpoint_context_fn, reusable_checkpoint_contexts
            try:
                yield reader
            finally:
                gw.fr._checkpoint_context_fn = saved

    def arm_root(self, arm):
        if arm in ARMS:
            return self.base / 'quantized' / arm
        return super().arm_root(arm)

    def qtip_module(self):
        Q = self.method('qtip')
        return Q, Q.ldlq._PERMUTE.clone(), Q.ldlq._INV_PERMUTE.clone()

    # ---------------- codes ----------------
    def code_items(self):
        """Sorted (index, K, module, source) over the base arms' sources.json rows with a QTIP K."""
        items = {}
        for c in ARMS.values():
            for r in wk.read(self.arm_root(c['base']) / 'sources.json'):
                if r['role'] in CODED_ROLES:
                    k = int(r['choice'])
                    key = (r['index'], k)
                    if key in items and items[key][1] != r['source']:
                        raise RuntimeError(f'Two sources for layer {r["index"]} at K={k}')
                    items[key] = (r['module'], r['source'])
        return [(i, k, items[(i, k)][0], items[(i, k)][1]) for i, k in sorted(items)]

    def code_path(self, i, k):
        return self.rc('codes', f'K{k}', f'{i:04d}.pt')

    def code_layer(self, i, k, name, source, Q, permute, permute_inv, table):
        stored = torch.load(source, map_location='cpu', weights_only=True)
        if stored['module'] != name:
            raise RuntimeError(f'Stored weight identity differs at {i}')
        Wq = stored['weight']
        m, n = Wq.shape
        if Q.padded_size(m) != m or Q.padded_size(n) != n:
            raise RuntimeError(f'Layer {i} is padded ({m} x {n}); not handled')
        if k in (1, 4):
            cs = torch.load(self.rv(f'qtip_K{k}', 'codes', f'{i:04d}.pt'), map_location='cpu', weights_only=True)
            if not torch.equal(lut_values(cs['tlut']), table):
                raise RuntimeError(f'Layer {i}: stored LUT differs from the global QTIP table')
            config = wk.read(self.rv(f'qtip_K{k}', 'stats', f'{i:04d}.json'))['config']
            SU, SV, Wscale = cs['SU'].float(), cs['SV'].float(), float(cs['Wscale'])
            candidates = [('trellis', lambda: trellis_codes(Q, cs))]
        else:
            config = wk.read(self.baseline / 'quantized' / f'QTIP_W{k}' / 'stats' / f'{i:04d}.json')['config']
            SU, SV = Q.sign_vectors(n, m, config['seed'])
            Wscale = float(config['Wscale'])

            def recovered():
                D = coherent(Q, Wq.to(DEVICE, torch.float64), SU.to(DEVICE, torch.float64), SV.to(DEVICE, torch.float64))
                codes, margin = nearest_codes(to_pairs(D / Wscale, permute).float(), table.to(DEVICE, torch.float32))
                del D
                info['recovery_margin_min'] = float(margin.min())
                info['recovery_small_margin'] = int((margin < 0.01).sum())
                return codes.cpu()

            def requantized():
                prep = self.baseline / 'prep'
                saved = {key: torch.load(p / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                         for key, p in (('master', prep / 'masters'), ('H', prep / 'hessians'))}
                torch.use_deterministic_algorithms(False)
                try:
                    out = rq.quantize_layer(Q, saved['master']['weight'].to(DEVICE, torch.float32),
                                            saved['H']['hessian'].to(DEVICE, torch.float32), k, seed=config['seed'])
                finally:
                    torch.use_deterministic_algorithms(True)
                info['requant_bitwise'] = bool(torch.equal(out['dequant'].to(torch.bfloat16).cpu(), Wq))
                native['trellis'] = {key: v for key, v in out['codes'].items()}  # the packed QTIP code, kept for reuse
                if not info['requant_bitwise']:  # a fresh QTIP rounding (same code, seed, H_U) becomes this layer's start
                    override['weight'] = out['dequant'].to(torch.bfloat16).cpu()
                    info['override_rel'] = float((override['weight'].float() - Wq.float()).norm() / Wq.float().norm())
                return trellis_codes(Q, out['codes'])
            # nearest-candidate recovery from the BF16 weight misreads ~0.2% of the pairs (sign twins of entries near
            # t0 = 0 sit within the BF16 noise; CPU test 26073382), so the trellis is recomputed
            candidates = [('requantized', requantized)]
        dtype = torch.float64 if any(a.get('ok') and 'float64' in a['dtype'] for a in config['attempts']) else torch.float32
        info, override, native = {}, {}, {}
        for method, make in candidates:
            pairs = make()
            if override:
                Wq, method = override['weight'], 'requantized_new'
            D = decode(pairs.to(DEVICE), table.to(DEVICE), m, n, permute_inv)
            hat = D.to(dtype) * torch.tensor(Wscale, dtype=dtype, device=DEVICE)
            dq = Q.inverse_incoherence(hat, SU.to(DEVICE, torch.bfloat16).to(dtype), SV.to(DEVICE, torch.bfloat16).to(dtype))
            ok = bool(torch.equal(dq.float().to(torch.bfloat16).cpu(), Wq))
            info[f'{method}_bitwise'] = ok
            del D, hat, dq
            if ok:
                counts = torch.bincount((pairs.long() & (ENTRIES - 1)).reshape(-1), minlength=ENTRIES)
                return dict(module=name, index=i, K=k, shape=[m, n], pairs=pairs.to(torch.int16), Wscale=Wscale,
                            dtype=str(dtype), SU=SU.to(torch.bfloat16), SV=SV.to(torch.bfloat16), counts=counts,
                            method=method, source=str(source), info=info, **override, **native)
        raise RuntimeError(f'Layer {i} (K={k}): no code reproduces the stored weight: {info}')

    def codes(self, shard, shards):
        Q, permute, permute_inv = self.qtip_module()
        table = lut_values(Q.codebook(2, 'cpu').tlut.to(torch.bfloat16))
        items = self.code_items()
        sizes = {i: math.prod(wk.read(self.baseline / 'quantized' / 'QTIP_W2' / 'stats' / f'{i:04d}.json')['shape'])
                 for i in {it[0] for it in items}}
        mine = wk.owned(shard, shards, lambda j: sizes[items[j][0]] * (4.0 if items[j][1] in (2, 3) else 1.0),
                        range(len(items)))
        report = []
        for done, j in enumerate(mine):
            i, k, name, source = items[j]
            path = self.code_path(i, k)
            if path.exists():
                continue
            self.enough()
            begin = time.monotonic()
            saved = self.code_layer(i, k, name, source, Q, permute, permute_inv, table)
            self.tsave(path, saved)
            report.append(dict(index=i, K=k, module=name, method=saved['method'], seconds=time.monotonic() - begin,
                               **saved['info']))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.progress('codes', items=done + 1, of=len(mine), index=i, K=k, method=saved['method'])
        wk.save(self.rc('codes', 'reports', f'{self.job_name}.json'), report)

    def codes_summary(self):
        items, rows = self.code_items(), []
        for i, k, name, _ in items:
            path = self.code_path(i, k)
            if not path.exists():
                raise RuntimeError(f'Codes missing: layer {i} K={k}')
        for f in sorted(self.rc('codes', 'reports').glob('*.json')):
            rows += wk.read(f)
        methods = {}
        for r in rows:
            methods[r['method']] = methods.get(r['method'], 0) + 1
        return dict(items=len(items), methods=methods,
                    small_margin_pairs=sum(r.get('recovery_small_margin', 0) for r in rows),
                    requant_not_bitwise=sum(1 for r in rows if r.get('requant_bitwise') is False),
                    override_rel_max=max([r['override_rel'] for r in rows if 'override_rel' in r], default=0.0))

    # ---------------- (1) the reader metric ----------------
    def metric_dir(self):
        return self.rc('metric')

    def metric(self, shard, shards):
        cases = wk.owned(shard, shards, lambda c: 1.0, lp.CASES)
        todo = [c for c in cases if not (self.metric_dir() / 'ctx' / f'{c}.pt').exists()]
        if not todo:
            return
        self.load()
        self.capture_masters()
        cfg, stats = RC['metric'], self.p['reader_source']['action_stats']
        tol = self.cfg['gates']['teacher_parity_max_abs']
        with self.bf16_reader() as reader:
            for count, case in enumerate(todo):
                self.enough()
                target = self.teacher(case, 0)
                value, description = self.datum(case)
                self.runtime.description = description
                leaf = target['future'].to(DEVICE, torch.float32).clone().requires_grad_(True)
                read, _ = reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], leaf, checkpointed=True)
                self.count('reader_queries')
                parity = wk.max_abs(read, target['action'])
                if parity > tol:
                    raise RuntimeError(f'BF16 reader at the teacher future differs from the teacher action: {parity}')
                curv = pr.softbin_kl_curvature(softbin_loss, target['action'][:10, :6], stats['mean'], stats['scale'],
                                               self.p['reader_source']['config']).to(DEVICE)
                gen = torch.Generator(device=DEVICE).manual_seed(cfg['seed'] + case)
                probes, frame0 = [], 0.0
                for p in range(cfg['probes_per_context']):
                    gk = torch.zeros_like(read)
                    gk[:10, :6] = pr.rademacher((10, 6), gen, DEVICE) * (curv / 2).sqrt()
                    c, = torch.autograd.grad(read, [leaf], gk, retain_graph=p < cfg['probes_per_context'] - 1)
                    frame0 = max(frame0, float(c[:, :, 0].abs().max()))
                    probes.append(c[:, :, FUTURE_FRAMES].detach().float().reshape(-1).cpu())
                    self.counters['parameter_reverse_passes'] += 1
                if frame0 != 0:
                    raise RuntimeError(f'Reader gradient reaches the observed frame 0: {frame0}')
                zt = target['future'][:, :, FUTURE_FRAMES].float()
                self.tsave(self.metric_dir() / 'ctx' / f'{case}.pt', dict(
                    case=case, probes=torch.stack(probes), parity=parity, curvature_mean=float(curv.mean()),
                    z_energy=float(zt.square().sum()), shape=list(zt.shape)))
                del read, leaf, curv, probes
                self.progress('metric', contexts=count + 1, of=len(todo))

    def metric_final(self):
        out = self.metric_dir() / 'metric.pt'
        if out.exists():
            return
        cfg = RC['metric']
        rows = [torch.load(self.metric_dir() / 'ctx' / f'{c}.pt', map_location='cpu', weights_only=True) for c in lp.CASES]
        vectors = [r['probes'] for r in rows]
        saved, stats, diag = build_metric(vectors, lp.CASES, cfg['rank'], cfg['rho'])
        stab = split_stability(vectors, lp.CASES, cfg['rank'])
        shape = rows[0]['shape']  # [1, 48, 2, 14, 28]
        d = diag.reshape(shape[1:])
        gaze = dict(channels=[float(x) for x in d.sum((1, 2, 3)) / d.sum()],
                    frames=[float(x) for x in d.sum((0, 2, 3)) / d.sum()],
                    space=(d.sum((0, 1)) / d.sum()).tolist())
        saved['shape'] = shape
        self.tsave(out, dict(saved, diag=diag))
        per_ctx = [float(r['probes'].square().sum(1).mean()) for r in rows]
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, metric=cfg, stats=stats, stability=stab, gaze=gaze,
                       context_trace=dict(values=per_ctx, ess=gw.ess(per_ctx)),
                       parity_max=max(r['parity'] for r in rows))
        wk.save(self.metric_dir() / 'summary.json', summary)
        (self.metric_dir() / 'RESULTS.zh.md').write_text(metric_report(summary), encoding='utf-8')

    def load_metric(self):
        self.M = Metric(torch.load(self.metric_dir() / 'metric.pt', map_location='cpu', weights_only=True))

    # ---------------- metrics with Z_M ----------------
    def metrics(self, action, future, target):
        m = super().metrics(action, future, target)
        if getattr(self, 'M', None) is not None:
            m['future_metric_nmse'] = float(self.M.nmse(future.detach(), target['future']))
        return m

    def terms(self, action, future, target):
        """A, Z, Z_M and A + lam_M Z_M (differentiable); A + lambda Z checked against the chain's own loss."""
        a_t = target['action'].to(action.device, torch.float32)
        A = (action[:10, :6].float() - a_t[:10, :6]).square().mean()
        zt = target['future'][:, :, FUTURE_FRAMES].to(future.device, torch.float32)
        Z = (future[:, :, FUTURE_FRAMES].float() - zt).square().sum() / zt.square().sum().clamp_min(1e-12)
        ZM = self.M.nmse(future, target['future'])
        if not getattr(self, 'loss_checked', False):
            ref, _, _ = self.loss(action, future, target)
            want = float(A + RC['loss']['lambda'] * Z)
            if abs(float(ref) - want) > 1e-6 * abs(float(ref)) + 1e-12:
                raise RuntimeError(f'A + lambda Z differs from the chain loss: {want} vs {float(ref)}')
            self.loss_checked = True
        return A, Z, ZM, A + self.lam_M * ZM

    # ---------------- arms ----------------
    def capture_biases(self):
        self.bias0 = {n: (m.bias.detach().cpu().clone() if m.bias is not None else None)
                      for n, m in self.bank.modules.items()}

    def install(self, arm):
        """Every install resets the biases first; a calibrated arm's weight files carry its bias."""
        with torch.no_grad():
            for n, m in self.bank.modules.items():
                if self.bias0[n] is not None:
                    m.bias.data.copy_(self.bias0[n].to(m.bias.device))
        if arm not in ARMS:
            return super().install(arm)
        root = self.arm_root(arm)
        with torch.no_grad():
            for i, name in enumerate(self.bank.names):
                saved = torch.load(root / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                if saved['module'] != name:
                    raise RuntimeError(f'Stored weight identity differs at {i}')
                mod = self.bank.modules[name]
                mod.weight.data.copy_(saved['weight'].to(mod.weight.device))
                if 'bias' in saved:
                    mod.bias.data.copy_(saved['bias'].to(mod.bias.device))

    @contextmanager
    def original_biases(self):
        """The frozen BF16 reader reads with the modules' biases: the originals inside, the calibrated ones after."""
        with torch.no_grad():
            for L in self.layers:
                if L.has_bias:
                    L.mod.bias.data.copy_(L.b0)
        try:
            yield
        finally:
            with torch.no_grad():
                for L in self.layers:
                    if L.has_bias:
                        L.mod.bias.data.copy_((L.b0.float() + L.db).to(L.b0.dtype))

    def attach(self, key):
        """QLayers of the base arm (installed before), the group tables, and wrapped forwards (plain unless active)."""
        Q, permute, permute_inv = self.qtip_module()
        base = ARMS[key]['base']
        rows = wk.read(self.arm_root(base) / 'sources.json')
        self.active, self.layers, self.tables, self.metric_w, self.group_ref = False, [], {}, {}, {}
        self.overridden = []
        for r in rows:
            if r['role'] not in CODED_ROLES:
                continue
            k, name = int(r['choice']), r['module']
            saved = torch.load(self.code_path(r['index'], k), map_location='cpu', weights_only=True)
            if saved['module'] != name or saved['source'] != r['source']:
                raise RuntimeError(f'Codes of layer {r["index"]} do not belong to {base}')
            mod = self.bank.modules[name]
            if not torch.equal(mod.weight.detach().cpu(), torch.load(r['source'], map_location='cpu', weights_only=True)['weight']):
                raise RuntimeError(f'Installed weight of {name} is not the base arm source')
            if 'weight' in saved:  # codes of a fresh QTIP rounding (codes stage): its weight is this layer's start
                with torch.no_grad():
                    mod.weight.data.copy_(saved['weight'].to(mod.weight.device))
                self.overridden.append(name)
            group = f"{r['role']}@{k}"
            L = QLayer(r['index'], name, mod, group, saved, Q, permute, permute_inv)
            self.layers.append(L)
            if group not in self.tables:
                self.tables[group] = torch.zeros(ENTRIES, 2, device=DEVICE, requires_grad=True)
                self.metric_w[group] = torch.zeros(ENTRIES, device=DEVICE, dtype=torch.float64)
                self.group_ref[group] = 0.0
            self.metric_w[group] += L.Wscale ** 2 * saved['counts'].to(DEVICE, torch.float64)
            self.group_ref[group] += L.wnorm ** 2
            mod.forward = self._wrapped(L, mod.forward)
        self.group_ref = {g: v ** 0.5 for g, v in self.group_ref.items()}
        self.table0 = lut_values(Q.codebook(2, 'cpu').tlut.to(torch.bfloat16)).to(DEVICE)
        self.lut_rms = float(self.table0.square().mean().sqrt())
        self.set_families()
        self.progress('attach', layers=len(self.layers), groups=sorted(self.tables), overridden=len(self.overridden),
                      families=sorted(self.families))

    def set_families(self):
        """Only the families in ravqc.json training.families take gradients and steps (v3: the shared tables alone)."""
        self.families = set(RC['training']['families'])
        for L in self.layers:
            L.s.requires_grad_('row' in self.families)
            L.t.requires_grad_('col' in self.families)
            L.db.requires_grad_(L.has_bias and 'bias' in self.families)
        for v in self.tables.values():
            v.requires_grad_('lut' in self.families)

    def _wrapped(self, L, plain):
        def run(x):
            if not self.active:
                return plain(x)
            self.bank.seen.add(L.name)
            return QTIPGrad.apply(x, L.s, L.t, L.db, self.tables[L.group], L)
        return run

    def params(self):
        out = [p for L in self.layers for p in (L.s, L.t) + ((L.db,) if L.has_bias else ())]
        return out + [self.tables[g] for g in sorted(self.tables)]

    def zero(self):
        for p in self.params():
            p.grad = None

    def snapshot(self):
        return dict(layers={L.name: dict(s=L.s.detach().cpu().clone(), t=L.t.detach().cpu().clone(),
                                          db=L.db.detach().cpu().clone()) for L in self.layers},
                    tables={g: v.detach().cpu().clone() for g, v in self.tables.items()})

    def restore(self, saved):
        with torch.no_grad():
            for L in self.layers:
                for k in ('s', 't', 'db'):
                    getattr(L, k).copy_(saved['layers'][L.name][k].to(DEVICE))
            for g, v in self.tables.items():
                v.copy_(saved['tables'][g].to(DEVICE))
        self.rebuild()

    def rebuild(self):
        for L in self.layers:
            L.rebuild(self.tables[L.group], self.table0)

    def grads(self):
        return dict(layers={L.name: dict(s=L.s.grad.detach().clone() if L.s.grad is not None else torch.zeros_like(L.s),
                                         t=L.t.grad.detach().clone() if L.t.grad is not None else torch.zeros_like(L.t),
                                         db=(L.db.grad.detach().clone() if L.db.grad is not None else torch.zeros_like(L.db)))
                            for L in self.layers},
                    tables={g: (v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v))
                            for g, v in self.tables.items()})

    def units(self, g):
        """(family, name, parameter, unit) per tensor and the gradients g_u = unit x g in relative units
        (ravqc.json training.update): log-scales as they are, a bias element relative to its row's output scale
        |W_r| rms(x), a table entry relative to rms(tlut); dead table entries (no pair uses them) carry no gradient.
        Only the families in training.families."""
        out = []
        for L in self.layers:
            gl = g['layers'][L.name]
            if 'row' in self.families:
                out.append(('row', L.name, L.s, 1.0, gl['s']))
            if 'col' in self.families:
                out.append(('col', L.name, L.t, 1.0, gl['t']))
            if L.has_bias and 'bias' in self.families:
                if L.bias_ref is None:  # rms(x) of the layer's input, from the first gradient pass
                    L.bias_ref = (float(L.xsq) / max(L.tokens, 1) / L.n) ** 0.5
                unit = L.rn * L.bias_ref
                out.append(('bias', L.name, L.db, unit, gl['db'] * unit))
        if 'lut' in self.families:
            for grp, v in self.tables.items():
                live = (self.metric_w[grp] > 0)[:, None].float()
                out.append(('lut', grp, v, self.lut_rms, g['tables'][grp] * self.lut_rms * live))
        return out

    def step(self, g, eta):
        """In place: delta u = -eta g_u / max|g_u| over every element (ravqc.json training.update, tune.json's rule).
        Returns the first-order predicted decrease eta sum g_u^2 / max|g_u| and its per-family parts; self.last_max says
        which element set the scale."""
        pred = dict(row=0.0, col=0.0, bias=0.0, lut=0.0)
        with torch.no_grad():
            rows = self.units(g)
            top, at = 0.0, None
            for fam, name, _, _, gu in rows:
                m = float(gu.abs().max()) if gu.numel() else 0.0
                if not math.isfinite(m):
                    raise RuntimeError(f'Non-finite gradient at {fam} {name}')
                if m > top:
                    top, at = m, (fam, name)
            if top == 0:
                raise RuntimeError('Zero gradient on every calibrated parameter')
            for fam, _, param, unit, gu in rows:
                param.sub_(eta * gu / top * unit)
                pred[fam] += eta * float(gu.square().sum()) / top
            self.last_max = dict(max_abs_g_u=top, family=at[0], at=at[1])
        self.rebuild()
        return sum(pred.values()), pred

    # ---------------- objective ----------------
    def objective(self, batch, grad):
        """mean over the batch of A + lam_M Z_M + beta K at the current parameters; with grad, the gradient accumulated
        into the parameters by the staged replay (no-grad queries -> A / Z_M cotangents, frozen BF16 reader -> beta K
        cotangent at the future, grad replay of the same queries)."""
        staged, out = self.staged(batch, grad)
        if grad:
            n = len(staged)
            out['replay_max_abs'] = max(self.replay(row, row['ga'] / n, row['gz'] / n) for row in staged)
            if out['replay_max_abs'] > RC['training']['replay_max_abs']:
                raise RuntimeError(f"Grad replay differs from the staged no-grad query: {out['replay_max_abs']}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def replay(self, row, cot_action, cot_future):
        """One grad replay of a staged query, backward with the given output cotangents (accumulates into .grad);
        returns its max abs difference from the staged no-grad query."""
        self.active = True
        try:
            action, future, _ = self.query(row['case'], 0, grad=True)
            if not (action.requires_grad and future.requires_grad):
                raise RuntimeError('Calibrated query lost its graph')
            worst = max(wk.max_abs(action, row['action']), wk.max_abs(future, row['future']))
            torch.autograd.backward((action, future), (cot_action.to(action), cot_future.to(future)))
            self.counters['parameter_reverse_passes'] += 1
        finally:
            self.active = False
        return worst

    def staged(self, batch, grad):
        """The objective's first two passes: no-grad queries -> A / Z_M cotangents (ga at the action, gz at the future),
        frozen BF16 reader -> beta K cotangent added to gz (its norm kept as gk). Returns the rows and the batch means."""
        from reader_loss import reader_terms
        beta, stats = RC['loss']['beta'], {'action': self.p['reader_source']['action_stats']}
        staged = []
        for case in batch:
            self.enough()
            with torch.no_grad():
                action, future, _ = self.query(case, 0)
            target = self.teacher(case, 0)
            a_leaf = action.detach().float().clone().requires_grad_(True)
            z_leaf = future.detach().float().clone().requires_grad_(True)
            A, Z, ZM, total = self.terms(a_leaf, z_leaf, target)
            ga, gz = torch.autograd.grad(total, (a_leaf, z_leaf))
            staged.append(dict(case=case, action=action.detach().cpu(), future=future.detach().cpu(), ga=ga.detach().cpu(),
                               gz=gz.detach().cpu(), A=float(A), Z=float(Z), ZM=float(ZM), az=float(total.detach())))
            del action, future, a_leaf, z_leaf, total, ga, gz
        with self.original_biases(), self.bf16_reader() as reader:
            for row in staged:
                case = row['case']
                value, description = self.datum(case)
                self.runtime.description = description
                leaf = row['future'].to(DEVICE, torch.float32).requires_grad_(grad)
                with torch.set_grad_enabled(grad):
                    read, _ = reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], leaf, checkpointed=grad)
                    k = reader_terms(torch, read, self.teacher(case, 0)['action'], stats,
                                     self.p['reader_source']['config'])['reader_kl']
                self.count('reader_queries')
                row['k'] = float(k.detach())
                if grad:
                    rg, = torch.autograd.grad(beta * k, leaf)
                    row['gk'] = float(rg.norm())
                    row['gz'] += rg.detach().cpu()
                del leaf, read, k
        out = {key: sum(r[key] for r in staged) / len(staged) for key in ('A', 'Z', 'ZM', 'az', 'k')}
        out['total'] = out['az'] + beta * out['k']
        return staged, out

    # ---------------- cross-context transfer of the table gradient (diagnostic) ----------------
    def table_grads(self):
        return {g: (v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros(ENTRIES, 2))
                for g, v in self.tables.items()}

    def case_grads(self, batch):
        """Per context of the batch at the current parameters: its staged row and the table gradient of its action part
        (cotangent ga at the action: A) and of its future part (gz at the future: lam_M Z_M + beta K), from two grad
        replays of the same query; their sum is the context's share of objective()'s gradient (times the batch size)."""
        staged, _ = self.staged(batch, grad=True)
        for row in staged:
            parts, worst = {}, 0.0
            for part, cots in (('action', (row['ga'], torch.zeros_like(row['gz']))),
                               ('future', (torch.zeros_like(row['ga']), row['gz']))):
                self.zero()
                worst = max(worst, self.replay(row, *cots))
                parts[part] = self.table_grads()
            if worst > RC['training']['replay_max_abs']:
                raise RuntimeError(f'Grad replay differs from the staged no-grad query: {worst}')
            yield row, parts, worst
        self.zero()

    def gradtx_root(self, key):
        return self.rc('gradtx', key)

    def gradtx(self, shard, shards):
        """ravqc.json gradtx: one arm per job (gradtx_arm), its C80 contexts split over the shards, at the base point every
        calibration starts from (zero table offsets, the fwrc2r3 attach); one file per context, so a resubmission
        continues."""
        key = gradtx_arm(os.environ['FW_JOB'])
        self.load()
        self.capture_masters()
        self.capture_biases()
        self.load_metric()
        self.install(ARMS[key]['base'])
        self.attach(key)
        before = self.fit(key, self.rc(key) / 'fit_before.json')
        self.lam_M = RC['loss']['lambda'] * before['future_latent_nmse'] / before['future_metric_nmse']
        self.gradtx_cases(key, lp.CASES[shard::shards], meta=shard == 0)

    def gradtx_cases(self, key, cases, meta=True):
        root, beta = self.gradtx_root(key), RC['loss']['beta']
        if meta:
            self.tsave(root / 'meta.pt', dict(arm=key, groups=sorted(self.tables), lut_rms=self.lut_rms, lam_M=self.lam_M,
                                              live={g: (self.metric_w[g] > 0).cpu() for g in sorted(self.tables)},
                                              pbs_jobid=self.job))
        todo = [c for c in cases if not (root / f'{c}.pt').exists()]
        for i in range(0, len(todo), RC['gradtx']['chunk']):
            for row, parts, worst in self.case_grads(todo[i:i + RC['gradtx']['chunk']]):
                self.tsave(root / f"{row['case']}.pt", dict(
                    case=row['case'], A=row['A'], Z=row['Z'], ZM=row['ZM'], az=row['az'], k=row['k'],
                    total=row['az'] + beta * row['k'], lam_M=self.lam_M, replay_max_abs=worst,
                    cot=dict(action=float(row['ga'].norm()), future=float(row['gz'].norm()), reader=row['gk']),
                    action=parts['action'], future=parts['future']))
                self.progress('gradtx', arm=key, case=row['case'], replay_max_abs=worst)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def gradtx_final(self, key, cases=None):
        """Transfer statistics of the arm's per-context gradients (transfer_stats) in the update's relative units
        (u = g x rms(tlut) on live entries), for the action part, the future part, their sum (what training follows)
        and two reweightings of the sum; per group; and the check that the batch-1 mean reproduces fwrc2r3's step-check
        gradient. results/ravqc/gradtx/<arm>/{summary.json, RESULTS.zh.md}."""
        cases = list(lp.CASES) if cases is None else cases
        root = self.gradtx_root(key)
        meta = torch.load(root / 'meta.pt', map_location='cpu', weights_only=True)
        rows = [torch.load(root / f'{c}.pt', map_location='cpu', weights_only=True) for c in cases]
        training = wk.read(self.rc(key, 'training.json'))
        step = wk.read(self.rc(key, 'step.json'))
        batches = [h['cases'] for h in training['history']]
        summary = gradtx_summary(meta, rows, batches, step, RC['gradtx'])
        summary.update(protocol=wk.PROTOCOL, pbs_jobid=self.job, arm=key, step_job=step.get('pbs_jobid'))
        wk.save(root / 'summary.json', summary)
        (root / 'RESULTS.zh.md').write_text(gradtx_report(summary), encoding='utf-8')
        return summary

    # ---------------- calibration ----------------
    def fit(self, arm, path):
        """C80 at sampler seed index 0 with the current weights: per context the metrics and Z_M."""
        if path.exists():
            return wk.read(path)
        rows = []
        for case in lp.CASES:
            self.enough()
            with torch.no_grad():
                action, future, _ = self.query(case, 0)
            m = self.metrics(action, future, self.teacher(case, 0))
            rows.append(dict(case_id=case, **m))
            self.progress('fit', arm=arm, contexts=len(rows))
        keys = [k for k in rows[0] if k != 'case_id']
        out = dict(protocol=wk.PROTOCOL, arm=arm, rows=rows, **{k: sum(r[k] for r in rows) / len(rows) for k in keys})
        wk.save(path, out)
        return out

    def calib(self, shard, shards):
        keys = list(ARMS)[shard::shards]
        if not keys:
            return
        self.load()
        self.capture_masters()
        self.capture_biases()
        self.load_metric()
        for key in keys:
            self.calibrate(key)

    def archive(self, key):
        """A finished arm from an older update rule: its rule-dependent files get the old job's suffix (fit_before and the
        reference zm files do not depend on the rule and stay)."""
        root = self.rc(key)
        tag = 'r0_' + str(wk.read(root / 'arm.json').get('pbs_jobid', 'old')).split('.')[0]
        moved = []
        for path in [root / n for n in ('arm.json', 'step.json', 'training.json', 'checkpoint.pt', 'fit_after.json',
                                         'params.pt')] + [self.rc('heldout', f'{key}.json'), self.rc('endpoints', key),
                                                          self.arm_root(key)]:
            if path.exists():
                target = path.with_name(f'{path.stem}.{tag}{path.suffix}')
                path.rename(target)
                moved.append(str(target))
        self.progress('archive', arm=key, moved=moved)

    def calibrate(self, key):
        c, root = ARMS[key], self.rc(key)
        if (root / 'arm.json').exists():
            if wk.read(root / 'arm.json').get('update_rule') == RC['training']['update_rule_id']:
                return
            self.archive(key)
        self.install(c['base'])
        self.attach(key)
        before = self.fit(key, root / 'fit_before.json')
        self.lam_M = RC['loss']['lambda'] * before['future_latent_nmse'] / before['future_metric_nmse']
        cfg = RC['training']
        state = dict(updates=0, history=[], eta=None, lam_M=self.lam_M)
        if (root / 'checkpoint.pt').exists():
            saved = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=True)
            self.restore(saved['params'])
            state = saved['state']
            for L in self.layers:
                L.bias_ref = saved['bias_ref'].get(L.name)
        if state['eta'] is None:
            state = self.step_check(key, state)
            if state['eta'] is None:
                wk.save(root / 'arm.json', dict(protocol=wk.PROTOCOL, arm=key, tuned=False, pbs_jobid=self.job,
                                                update_rule=RC['training']['update_rule_id']))
                return
        while state['updates'] < cfg['updates']:
            self.enough()
            u, begin = state['updates'], time.monotonic()
            self.zero()
            out = self.objective(self.schedule[u], grad=True)
            pred, parts = self.step(self.grads(), state['eta'])
            state['updates'] += 1
            state['history'].append(dict(update=u + 1, cases=self.schedule[u], before=out, predicted=pred, parts=parts,
                                         scale=dict(self.last_max), seconds=time.monotonic() - begin))
            self.checkpoint(key, state)
            self.progress('train', arm=key, updates=state['updates'], loss=out['total'])
        self.zero()
        self.deploy(key, state)
        self.fit(key, root / 'fit_after.json')
        self.evaluate_arm(key)
        wk.save(root / 'arm.json', dict(protocol=wk.PROTOCOL, arm=key, tuned=True, eta=state['eta'], lam_M=self.lam_M,
                                        pbs_jobid=self.job, update_rule=RC['training']['update_rule_id']))

    def checkpoint(self, key, state):
        self.tsave(self.rc(key) / 'checkpoint.pt', dict(params=self.snapshot(), state=state,
                                                       bias_ref={L.name: L.bias_ref for L in self.layers}))
        wk.save(self.rc(key) / 'training.json', dict(protocol=wk.PROTOCOL, arm=key, **state))

    def step_check(self, key, state):
        """Batch 1 at the start: one gradient, a relative step per grid eta (realized vs predicted on the same batch);
        the chosen step is kept as update 1."""
        cfg, batch = RC['training'], self.schedule[0]
        self.zero()
        base = self.objective(batch, grad=True)
        g = self.grads()
        start = self.snapshot()
        rows = []
        for eta in cfg['step_grid']:
            pred, parts = self.step(g, eta)
            after = self.objective(batch, grad=False)
            rows.append(dict(lr=eta, predicted=pred, parts=parts, realized=base['total'] - after['total'], loss=after,
                             scale=dict(self.last_max)))
            self.restore(start)
            self.progress('step', arm=key, eta=eta, realized=rows[-1]['realized'], predicted=pred)
        chosen = first_step_choice(rows, cfg['min_realized_fraction'])
        top = sorted(((float(gl['s'].square().sum() + gl['t'].square().sum() + gl['db'].square().sum()), n)
                      for n, gl in g['layers'].items()), reverse=True)[:10]
        wk.save(self.rc(key) / 'step.json', dict(protocol=wk.PROTOCOL, arm=key, pbs_jobid=self.job, batch=batch, base=base,
                                                 lam_M=self.lam_M, rows=rows, chosen=chosen,
                                                 top_layers=[dict(module=n, g2=v) for v, n in top],
                                                 table_grad_norm={k: float(v.norm()) for k, v in g['tables'].items()}))
        state['eta'] = chosen
        if chosen is not None:
            pred, parts = self.step(g, chosen)
            state['updates'] = 1
            state['history'].append(dict(update=1, cases=batch, before=base, predicted=pred, parts=parts, from_step_check=True))
            self.checkpoint(key, state)
        return state

    def deploy(self, key, state):
        """results/quantized/<arm>/weights: the calibrated layers' W' and b' (exactly the training forward's), symlinks
        to the base arm's files for the rest; COMPLETE.json with the bits (the base arm's plus the group tables)."""
        root, base = self.arm_root(key), ARMS[key]['base']
        if (root / 'COMPLETE.json').exists():
            return
        self.tsave(self.rc(key) / 'params.pt', self.snapshot())
        mine = {L.index: L for L in self.layers}
        rows = wk.read(self.arm_root(base) / 'sources.json')
        changed = []
        for r in rows:
            i = r['index']
            link = root / 'weights' / f'{i:04d}.pt'
            link.parent.mkdir(parents=True, exist_ok=True)
            if i in mine:
                L = mine[i]
                value = dict(module=L.name, weight=L.mod.weight.detach().cpu().clone())
                if L.has_bias:
                    value['bias'] = L.mod.bias.detach().cpu().clone()
                self.tsave(link, value)
                changed.append(dict(index=i, module=L.name,
                                    weight_rel=float((value['weight'].float() - L.W0.float()).norm() / max(L.W0.float().norm(), 1e-30)),
                                    bias_rel=(float((value['bias'].float() - L.b0.float().cpu()).norm()
                                                    / max(float(L.b0.float().norm()), 1e-30)) if L.has_bias else None)))
            elif not link.exists():
                os.symlink(r['source'], link)
        wk.save(root / 'sources.json', changed)
        b = wk.read(self.arm_root(base) / 'COMPLETE.json')
        extra = 16 * ENTRIES * 2 * len(self.tables)
        wk.save(root / 'COMPLETE.json', dict(
            protocol=wk.PROTOCOL, complete=True, pbs_jobid=self.job, arm=key, method='qtip', bits=ARMS[key]['family'],
            base=base, bits_total=b['bits_total'] + extra, extra_bits=extra, target_weights=b['target_weights'],
            bpw=(b['bits_total'] + extra) / b['target_weights'], proxy_nmse_weighted=b['proxy_nmse_weighted'],
            eta=state['eta'], lam_M=self.lam_M, updates=state['updates'], groups=sorted(self.tables),
            calibrated_layers=len(self.layers)))

    # ---------------- evaluation ----------------
    def evaluate_arm(self, arm):
        """Held-out 40x2 + C1 swap (worker.Work.endpoint) with Z_M, under results/ravqc."""
        out = self.rc('heldout', f'{arm}.json')
        if out.exists():
            return wk.read(out)
        ev = self.cfg['evaluation']
        self.install(arm)
        records = []
        for case in ev['case_ids']:
            for seed in ev['seed_indices']:
                path = self.rc('endpoints', arm, f'{case}_{seed}.json')
                if path.exists():
                    records.append(wk.read(path))
                    continue
                self.results = self.base / 'ravqc'  # endpoint() writes under self.results / 'endpoints'
                try:
                    row = self.endpoint(arm, case, seed)
                finally:
                    self.results = self.base
                records.append(row)
                self.progress('heldout', arm=arm, queries=len(records))
        result = dict(protocol=wk.PROTOCOL, arm=arm, pbs_jobid=self.job,
                      **wk.aggregate(records, ev['case_ids'], ev['seed_indices'], self.cfg['suites'],
                                     tuple(self.helpers.METRICS) + wk.EXTRA + EXTRA_METRICS))
        wk.save(out, result)
        return result

    def zm(self, shard, shards):
        arms = [a for a in RC['zm_arms'] if not self.rc('zm', f'{a}.json').exists()][shard::shards]
        if not arms:
            return
        self.load()
        self.capture_biases()
        self.load_metric()
        ev = self.cfg['evaluation']
        for arm in arms:
            self.install(arm)
            records, worst = [], 0.0
            for case in ev['case_ids']:
                for seed in ev['seed_indices']:
                    self.enough()
                    with torch.no_grad():
                        _, future, _ = self.query(case, seed)
                    target = self.teacher(case, seed)
                    z = float(self.M.nmse(future, target['future']))
                    zt = target['future'][:, :, FUTURE_FRAMES].float()
                    nmse = float((future[:, :, FUTURE_FRAMES].float().cpu() - zt).square().sum() / zt.square().sum())
                    ref = self.base / 'endpoints' / arm / f'{case}_{seed}.json'
                    if ref.exists():
                        worst = max(worst, abs(nmse - wk.read(ref)['future_latent_nmse']) / max(nmse, 1e-30))
                    records.append(dict(case_id=case, seed_index=seed, suite=self.rows[case]['suite'],
                                        future_metric_nmse=z, future_latent_nmse=nmse))
                    self.progress('zm', arm=arm, queries=len(records))
            if worst > 1e-4:
                raise RuntimeError(f'{arm}: recomputed future NMSE differs from its held-out endpoints: {worst}')
            wk.save(self.rc('zm', f'{arm}.json'), dict(
                protocol=wk.PROTOCOL, arm=arm, pbs_jobid=self.job, endpoint_check_rel=worst,
                **wk.aggregate(records, ev['case_ids'], ev['seed_indices'], self.cfg['suites'],
                               ('future_metric_nmse', 'future_latent_nmse'))))

    # ---------------- final ----------------
    def final(self):
        held, zm = {}, {}
        tuned = [k for k in ARMS if wk.read(self.rc(k) / 'arm.json').get('tuned')]
        for key, c in ARMS.items():
            if key in tuned:
                held[key] = wk.read(self.rc('heldout', f'{key}.json'))['overall']
            for ref in (c['base'], c['fair_reference'], 'QTIP_U_W2' if c['family'] == 2 else c['host_reference']):
                held.setdefault(ref, wk.read(self.base / 'heldout' / f'{ref}.json')['overall'])
        for a in RC['zm_arms']:
            path = self.rc('zm', f'{a}.json')
            if path.exists():
                zm[a] = wk.read(path)['overall']
        arms = {k: dict(arm=wk.read(self.rc(k) / 'arm.json'), training=wk.read(self.rc(k) / 'training.json'),
                        step=wk.read(self.rc(k) / 'step.json'), fit_before=strip(wk.read(self.rc(k) / 'fit_before.json')),
                        fit_after=strip(wk.read(self.rc(k) / 'fit_after.json')),
                        complete=wk.read(self.arm_root(k) / 'COMPLETE.json')) for k in tuned}
        for k in tuned:
            arms[k]['unseen'] = unseen(arms[k]['training']['history'], wk.read(self.rc(k) / 'fit_before.json')['rows'])
        keys = ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse', 'first10gripper_rmse', 'first10motor_rmse_swap')
        ratios = {}
        for k in tuned:
            c = ARMS[k]
            for ref in (c['base'], c['fair_reference'], 'QTIP_U_W2' if c['family'] == 2 else c['host_reference']):
                ratios[f'{k}/{ref}'] = {m: held[k][m] / held[ref][m] for m in keys}
                if ref in zm:
                    ratios[f'{k}/{ref}']['future_metric_nmse'] = held[k]['future_metric_nmse'] / zm[ref]['future_metric_nmse']
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, ravqc=RC, heldout=held, zm=zm, arms=arms, ratios=ratios,
                       metric=wk.read(self.metric_dir() / 'summary.json'))
        wk.save(self.rc('summary.json'), summary)
        loops = [k for k in tuned if held[k]['mixed_loss'] <= held[ARMS[k]['base']]['mixed_loss']]
        wk.save(self.rc('DECISION.json'), dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, loops=loops,
                                               rule='user 2026-10-31: held-out mixed loss not above the uncalibrated base',
                                               mixed_ratio={k: held[k]['mixed_loss'] / held[ARMS[k]['base']]['mixed_loss']
                                                            for k in tuned}))
        self.rc('RESULTS.zh.md').write_text(report(summary), encoding='utf-8')

    def loop_final(self):
        def episodes(root):
            return ({int(f.stem): wk.read(f)['success'] for f in (root / 'episodes').glob('*.json')}
                    if (root / 'DONE').exists() else None)
        rows = {}
        for key, c in ARMS.items():
            t = episodes(self.base / 'loops' / key)
            if t is None:
                continue
            row = dict(successes=sum(t.values()), episodes=len(t), pairs={})
            for ref, where in ((c['host_reference'], self.baseline), (c['base'], self.base), (c['fair_reference'], self.base)):
                u = episodes(where / 'loops' / ref)
                if u is None:
                    continue
                wins = sum(t[e] and not u[e] for e in t if e in u)
                losses = sum(u[e] and not t[e] for e in t if e in u)
                row['pairs'][ref] = dict(reference=sum(u.values()), wins=wins, losses=losses, p=wk.mcnemar(wins, losses))
            rows[key] = row
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, complete=True, arms=rows)
        wk.save(self.rc('loop', 'summary.json'), summary)
        lines = ['# RAVQ 校准版：LIBERO-Plus 280 闭环', '', f'作业 `{self.job}`。', '',
                 '| 臂 | 成功/280 | 对照 | 对照成功 | 胜 / 负 | McNemar p |', '|---|---:|---|---:|---:|---:|']
        for key, r in rows.items():
            for ref, q in r['pairs'].items():
                lines.append(f"| {LABELS.get(key, key)} | {r['successes']} | {LABELS.get(ref, ref)} | {q['reference']} | "
                             f"{q['wins']} / {q['losses']} | {q['p']:.2f} |")
        self.rc('loop', 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return summary

    def stage(self, name, shard, shards):
        if name == 'codes':
            self.codes(shard, shards)
        elif name == 'metric':
            self.metric(shard, shards)
        elif name == 'metric_final':
            if shard == 0:
                wk.save(self.rc('codes', 'summary.json'), self.codes_summary())
                self.metric_final()
        elif name == 'calib':
            self.calib(shard, shards)
        elif name == 'zm':
            self.zm(shard, shards)
        elif name == 'final':
            self.final()
        elif name == 'loop_final':
            self.loop_final()
        elif name == 'gradtx':
            self.gradtx(shard, shards)
        elif name == 'gradtx_final':
            if shard == 0:
                self.gradtx_final(gradtx_arm(os.environ['FW_JOB']))
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


def strip(fit):
    return {k: v for k, v in fit.items() if k != 'rows'}


def unseen(history, before_rows):
    """Prequential check inside C80: batch u >= 2 before its own update has been trained on the other batches only;
    its A, Z, Z_M over the uncalibrated start on the same contexts (fit_before), summed over the batches. Seen contexts
    gaining far more than these is memorization (fwrc2r2 26074313, all families: C80 after Z 0.56x; unseen Z 0.78x,
    one batch 0.35x and the other six 0.95-1.10x; held-out Z 1.15x, mixed 1.22x)."""
    fb = {r['case_id']: r for r in before_rows}
    later = history[1:]
    out = {}
    for short, full in (('A', 'first10motor_mse'), ('Z', 'future_latent_nmse'), ('ZM', 'future_metric_nmse')):
        start = sum(sum(fb[c][full] for c in h['cases']) / len(h['cases']) for h in later)
        out[short] = sum(h['before'][short] for h in later) / start if later else None
    return out


def gradtx_arm(job):
    """The arm of a gradtx job from its PBS job name (ravqc.json gradtx_jobs, attempts rN appended): run_python's
    container starts with --cleanenv and passes FW_JOB but no other new variable (26075618 / 26075620 died on FW_ARM)."""
    found = [a for a, j in RC['gradtx_jobs'].items() if re.fullmatch(re.escape(j['job_name']) + r'(r\d)?', job)]
    if len(found) != 1:
        raise ValueError(f'No gradtx arm for job {job!r}')
    return found[0]


def quantiles(values):
    v = sorted(float(x) for x in values)
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]  # noqa: E731
    return dict(mean=sum(v) / len(v), p05=pick(0.05), p50=pick(0.5), p95=pick(0.95))


def transfer_stats(X, ids, batches=None, splits=1000, seed=0):
    """Cross-context transfer of per-context gradients X [n, D] (float64; a step follows their mean).
    generic: the share of a context's gradient energy common to all contexts, (n |mean|^2 - mean |g_c|^2) /
    ((n - 1) mean |g_c|^2) (unbiased for g_c = mu + independent noise: |mu|^2 / E|g_c|^2);
    split: random halves h1, h2: cos(h1, h2) and T = h1.h2 / |h1|^2, the share of half 1's first-order predicted decrease
    that half 2 receives from a step along h1 (1 = full transfer, 0 = none);
    batch: the same T and cos between the training schedule's batches (one update's transfer to an unseen batch);
    top5: the five contexts with the largest share of |mean|^2 (shares x_c.mean / (n |mean|^2) sum to 1)."""
    n = X.shape[0]
    mean = X.mean(0)
    e = float(X.square().sum(1).mean())
    m2 = float(mean @ mean)
    out = dict(n=n, energy=e, mean_energy=m2, generic=(n * m2 - e) / ((n - 1) * e) if e > 0 else 0.0)
    gen = torch.Generator().manual_seed(seed)
    cos, T = [], []
    for _ in range(splits):
        perm = torch.randperm(n, generator=gen)
        h1, h2 = X[perm[:n // 2]].mean(0), X[perm[n // 2:]].mean(0)
        d = float(h1 @ h2)
        cos.append(d / max(float(h1.norm() * h2.norm()), 1e-300))
        T.append(d / max(float(h1 @ h1), 1e-300))
    out.update(split_cos=quantiles(cos), split_T=quantiles(T))
    if batches:
        pos = {c: i for i, c in enumerate(ids)}
        B = [X[[pos[c] for c in b]].mean(0) for b in batches]
        pairs = [(i, j) for i in range(len(B)) for j in range(len(B)) if i != j]
        out['batch_T'] = quantiles(float(B[j] @ B[i]) / max(float(B[i] @ B[i]), 1e-300) for i, j in pairs)
        out['batch_cos'] = quantiles(float(B[j] @ B[i]) / max(float(B[i].norm() * B[j].norm()), 1e-300) for i, j in pairs)
    share = (X @ mean) / max(n * m2, 1e-300)
    top = sorted(range(n), key=lambda i: -float(share[i]))[:5]
    out['top5_share'] = float(share[top].sum())
    out['top5'] = [dict(case=ids[i], share=float(share[i])) for i in top]
    return out


def gradtx_summary(meta, rows, batches, step, cfg):
    """ravqc.json gradtx on one arm: per-context rows (case_grads files) -> transfer_stats per part and weighting, per
    group of the sum, and the batch-1 check against the step check's table gradient norms."""
    groups, rms = meta['groups'], float(meta['lut_rms'])
    ids = [int(r['case']) for r in rows]
    live = torch.cat([meta['live'][g].double()[:, None].expand(ENTRIES, 2).reshape(-1) for g in groups])

    def mat(part, only=None):
        return torch.stack([torch.cat([r[part][g].double().reshape(-1) for g in (only or groups)]) for r in rows]) * rms

    X = dict(action=mat('action') * live, future=mat('future') * live)
    X['total'] = X['action'] + X['future']
    total = torch.tensor([float(r['total']) for r in rows], dtype=torch.float64)
    Z = torch.tensor([float(r['Z']) for r in rows], dtype=torch.float64)
    X['total_normalized'] = X['total'] / total[:, None] * float(total.mean())  # every context's relative change counts
    keep = (Z <= cfg['trim_factor'] * float(Z.median())).nonzero().flatten().tolist()
    parts = {k: transfer_stats(v, ids, batches, cfg['splits'], cfg['seed']) for k, v in X.items()}
    kept = [ids[i] for i in keep]
    trimmed = [[c for c in b if c in set(kept)] for b in batches]
    parts['total_trimmed'] = transfer_stats(X['total'][keep], kept, [b for b in trimmed if b], cfg['splits'], cfg['seed'])
    per_group, per_group_parts = {}, {}
    for g in groups:
        lg = meta['live'][g].double()[:, None].expand(ENTRIES, 2).reshape(-1)
        per_group[g] = transfer_stats((mat('action', [g]) + mat('future', [g])) * lg, ids, batches, cfg['splits'], cfg['seed'])
        per_group_parts[g] = {p: transfer_stats(mat(p, [g]) * lg, ids, batches, cfg['splits'], cfg['seed'])
                              for p in ('action', 'future')}
    first = [i for i, c in enumerate(ids) if c in set(batches[0])]
    check = {}
    for g in groups:
        raw = sum(rows[i]['action'][g].double() + rows[i]['future'][g].double() for i in first) / len(first)
        want = step['table_grad_norm'].get(g)
        check[g] = dict(norm=float(raw.norm()), step_check=want, ratio=float(raw.norm()) / want if want else None)
    a, f = X['action'].mean(0), X['future'].mean(0)
    return dict(
        groups=groups, lam_M=float(meta['lam_M']), contexts=len(rows), parts=parts, per_group=per_group,
        per_group_parts=per_group_parts,
        batch1_check=check, trimmed_out=[dict(case=ids[i], Z=float(Z[i])) for i in range(len(ids)) if i not in set(keep)],
        mean_split=dict(action_energy=float(a @ a), future_energy=float(f @ f),
                        cos=float(a @ f) / max(float(a.norm() * f.norm()), 1e-300)),
        cotangent=dict(action=quantiles(r['cot']['action'] for r in rows), future=quantiles(r['cot']['future'] for r in rows),
                       reader=quantiles(r['cot']['reader'] for r in rows)),
        losses={k: quantiles(float(r[k]) for r in rows) for k in ('A', 'Z', 'ZM', 'k', 'total')},
        replay_max_abs=max(float(r['replay_max_abs']) for r in rows), go_generic=cfg['go_generic'])


def gradtx_report(s):
    names = dict(action='动作部分（A）', future='future 部分（λ_M Z_M + βK）', total='两者之和（训练实际走的方向）',
                 total_normalized='之和，每个场景按自身损失归一', total_trimmed='之和，去掉 Z 超过中位数 '
                 f"{RC['gradtx']['trim_factor']:g} 倍的场景")
    lines = [f"# 解码表梯度能否跨场景迁移：{LABELS.get(s['arm'], s['arm'])}", '',
             f"作业 `{s['pbs_jobid']}`。起点 = 每次校准的起点（解码表偏移为 0，与 fwrc2r3 `{s['step_job']}` 相同），"
             f"C80 的 {s['contexts']} 个场景各自算一次解码表梯度（动作部分和 future 部分各一次回放），单位与更新规则相同"
             f"（u = g × rms(tlut)，只算被用到的表项）。λ_M = {s['lam_M']:.4f}。", '',
             '"通用份额"是一个场景的梯度里所有场景共有的那部分能量占比；"对半迁移 T"是把 80 个场景随机分两半，'
             '沿前一半的平均梯度走一步，后一半得到的一阶下降占前一半自己一阶下降的比例（1 = 完全迁移，0 = 不迁移）；'
             '"批间迁移 T"是训练用的 10 个场景一批之间的同一个量，也就是一次更新对没见过的一批的作用。'
             f"以前的门槛（fwgr 迁移诊断）是通用份额 ≥ {s['go_generic']:g}。", '',
             '| 梯度 | 通用份额 | 对半迁移 T 均值 [5%, 95%] | 对半余弦 | 批间迁移 T 均值 | 前 5 个场景占平均梯度 |',
             '|---|---:|---|---:|---:|---:|']
    for k, v in s['parts'].items():
        t, b = v['split_T'], v.get('batch_T')
        lines.append(f"| {names[k]} | {v['generic']:.4f} | {t['mean']:.3f} [{t['p05']:.3f}, {t['p95']:.3f}] | "
                     f"{v['split_cos']['mean']:.3f} | {b['mean']:.3f} | {v['top5_share']:.2f} |" if b else
                     f"| {names[k]} | {v['generic']:.4f} | {t['mean']:.3f} | {v['split_cos']['mean']:.3f} | — | "
                     f"{v['top5_share']:.2f} |")
    lines += ['', '按组（"平均梯度能量占比"是这一组在两者之和的平均梯度 |mean|² 里的份额，也就是一步更新主要改哪张表；'
              '后两列是这一组只算动作部分或只算 future 部分时的对半迁移 T）：', '',
              '| 组 | 平均梯度能量占比 | 通用份额 | 对半迁移 T | 批间迁移 T | 动作部分 对半 T | future 部分 对半 T |',
              '|---|---:|---:|---:|---:|---:|---:|']
    whole = s['parts']['total']['mean_energy']
    for g, v in s['per_group'].items():
        pp = s.get('per_group_parts', {}).get(g)
        tail = (f"{pp['action']['split_T']['mean']:.3f} | {pp['future']['split_T']['mean']:.3f}" if pp else '— | —')
        lines.append(f"| {g} | {v['mean_energy'] / whole:.4f} | {v['generic']:.4f} | {v['split_T']['mean']:.3f} | "
                     f"{v['batch_T']['mean']:.3f} | {tail} |")
    ms = s['mean_split']
    lines += ['', f"平均梯度里动作部分的能量 {ms['action_energy']:.3g}、future 部分 {ms['future_energy']:.3g}，"
              f"两者余弦 {ms['cos']:.3f}。去掉的离群场景：" + (', '.join(f"{r['case']}（Z {r['Z']:.3f}）"
                                                           for r in s['trimmed_out']) or '无') + '。',
              '前 5 个场景（两者之和）：' + ', '.join(f"{r['case']} {r['share']:.2f}" for r in s['parts']['total']['top5']) + '。',
              '', '第 1 批平均梯度与 fwrc2r3 步长检查的梯度范数之比（应为 1；BF16 层输出处回传梯度的舍入在 1e-3 量级）：'
              + ', '.join(f"{g} {c['ratio']:.4f}" if c['ratio'] is not None else f"{g} —"
                          for g, c in s['batch1_check'].items()) + f"；回放最大差 {s['replay_max_abs']:.2g}。"]
    return '\n'.join(lines) + '\n'


LABELS = dict(rw.LABELS, RAVQC_W2='RAVQ 校准版，2-bit 预算', RAVQC_W3='RAVQ 校准版，3-bit 预算')


def metric_report(s):
    st, sb, g = s['stats'], s['stability'], s['gaze']
    lines = ['# ① 读出度量 M：动作专家怎样"看"想象的 future', '',
             f"作业 `{s['pbs_jobid']}`。BF16 模型、C80 的 80 个场景、每个场景 {s['metric']['probes_per_context']} 个探针，"
             f"在 teacher future 处取冻结 reader 的 Gauss-Newton 矩阵（K 的线性化），每个场景先归一到单位迹再平均。"
             f"teacher 动作复现最大差 {s['parity_max']:.2g}。", '',
             '| 量 | 值 |', '|---|---:|',
             f"| 前 {s['metric']['rank']} 个特征方向占的迹 | {st['top_share']:.3f} |",
             f"| 有效秩（tr² / tr(M²)） | {st['effective_rank']:.1f} |",
             f"| 对角质量前 10% 元素的占比 | {st['diag_top10_share']:.3f} |",
             f"| 奇偶两半对角的相关 | {sb['diag_corr']:.3f} |",
             f"| 奇偶两半前 {s['metric']['rank']} 维子空间的重合（平均 cos²） | {sb['subspace_overlap']:.3f} |",
             f"| 各场景原始迹的有效样本数 | {s['context_trace']['ess']:.1f} / 80 |", '',
             '帧占比（future 第 1、2 帧）：' + '，'.join(f'{x:.3f}' for x in g['frames']) + '。',
             '通道占比最高的 8 个通道：' + '，'.join(f'{i}:{v:.3f}' for v, i in sorted(((v, i) for i, v in enumerate(g['channels'])), reverse=True)[:8]) + '。']
    return '\n'.join(lines) + '\n'


def report(s):
    held, zm, arms, r = s['heldout'], s['zm'], s['arms'], s['ratios']
    order = ['QTIP_U_W2', 'QTIP_W2I', 'RAVQU_W2', 'RAVQC_W2', 'QTIP_W3', 'QTIP_W3I', 'RAVQU_W3', 'RAVQC_W3']
    m = s['metric']
    names = dict(lut='按（角色，比特）分组、所有层共用的解码表', row='每行尺度（SV）', col='每列尺度（SU）', bias='输出偏置')
    fams = '、'.join(names[f] for f in ('lut', 'row', 'col', 'bias') if f in s['ravqc']['training']['families'])
    lines = ['# RAVQ 校准版：① 读出度量当 future 目标，④ 只调 QTIP 的共享参数', '',
             f"作业 `{s['pbs_jobid']}`，更新规则 `{s['ravqc']['training']['update_rule_id']}`。起点是 RAVQ 未校准版"
             f"（QTIP 舍入 + C1 模块级比特预算），只调：{fams}；目标 A + λ_M Z_M + βK，"
             "Z_M 是读出度量下的 future NMSE。预算与 Q0 相同（C80 一遍，8 次更新）。", '',
             f"读出度量：前 {m['metric']['rank']} 维占迹 {m['stats']['top_share']:.3f}，有效秩 {m['stats']['effective_rank']:.1f}，"
             f"奇偶两半对角相关 {m['stability']['diag_corr']:.3f}、子空间重合 {m['stability']['subspace_overlap']:.3f}。", '',
             '## 训练', '',
             '"未见批次"：第 2–8 批在各自那次更新之前只被别的批次训练过，它们更新前的 A / Z / Z_M 除以同一批场景未校准时的值'
             '（C80 内部的泛化检查；C80 整体变好而这一栏停在 1，就是背下了训练场景）。', '',
             '| 臂 | λ_M | η | C80 mixed 前 → 后 | C80 future NMSE 前 → 后 | C80 Z_M 前 → 后 | 未见批次 A / Z / Z_M | bpw |',
             '|---|---:|---:|---|---|---|---|---:|']
    for k, a in arms.items():
        u = ' / '.join('—' if a['unseen'][x] is None else f"{a['unseen'][x]:.3f}" for x in ('A', 'Z', 'ZM'))
        lines.append(f"| {LABELS[k]} | {a['training']['lam_M']:.4f} | {a['arm'].get('eta')} | "
                     f"{a['fit_before']['mixed_loss']:.5f} → {a['fit_after']['mixed_loss']:.5f} | "
                     f"{a['fit_before']['future_latent_nmse']:.4f} → {a['fit_after']['future_latent_nmse']:.4f} | "
                     f"{a['fit_before']['future_metric_nmse']:.4f} → {a['fit_after']['future_metric_nmse']:.4f} | "
                     f"{u} | {a['complete']['bpw']:.4f} |")
    lines += ['', '## held-out 40×2', '',
              '| 臂 | mixed loss | first10 motor RMSE | future NMSE | Z_M（读出度量） | gripper RMSE | 换 BF16 future 后 motor |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for a in order:
        if a not in held:
            continue
        o = held[a]
        z = o.get('future_metric_nmse', zm.get(a, {}).get('future_metric_nmse'))
        lines.append(f"| {LABELS.get(a, a)} | {o['mixed_loss']:.5f} | {o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | "
                     f"{'—' if z is None else format(z, '.4f')} | {o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} |")
    lines += ['', '## 比值（校准版 / 对照）', '', '| 对比 | mixed | motor | future NMSE | Z_M | gripper |', '|---|---:|---:|---:|---:|---:|']
    for key, v in r.items():
        a, b = key.split('/')
        lines.append(f"| {LABELS.get(a, a)} / {LABELS.get(b, b)} | {v['mixed_loss']:.3f} | {v['first10motor_rmse']:.3f} | "
                     f"{v['future_latent_nmse']:.3f} | {format(v['future_metric_nmse'], '.3f') if 'future_metric_nmse' in v else '—'} | "
                     f"{v['first10gripper_rmse']:.3f} |")
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
        RavqcWork(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
