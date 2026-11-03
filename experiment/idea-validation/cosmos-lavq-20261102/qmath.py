"""Model-independent pieces of LAVQ: QTIP decode / adjoint (ported from fastwam-guided-hessian ravqc_worker.py, tested
there bit for bit), the calibrated table layers (tables only, ravqc v3), and the reader metric M.

QTIP decodes a pair of weights of the rotated matrix from a 16-bit trellis state: entry k = hash(state) of a 512 x 2
table, the first coordinate's sign from another hash bit, times the layer's Wscale; W = diag(SV) H_m^T (Wscale D) H_n
diag(SU). Shared-table calibration learns one offset table per (role, K) group; QTIPGrad runs the plain forward and
reduces the weight gradient onto the table through the pair codes in its backward.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ENTRIES, TILE = 512, 256
_HADK = {}


# ---------------- QTIP decode / Hadamard pieces (pure functions) ----------------
def had(Q, X, transpose=False):
    """Q.matmul_hadU(X, transpose) without its per-call gc.collect / empty_cache: same butterflies, the K x K factor by a
    broadcast matmul."""
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


def states_to_codes(states, tlut_bits=9):
    """quantlut_sym: v = (s + 1) s; sign flip of coordinate 0 from bit 15, entry from bits 6..14 (L = 16, Q = 9)."""
    s = states.long()
    v = (s + 1) * s
    flip = (v >> 15) & 1
    idx = (v >> (16 - tlut_bits - 1)) & ((1 << tlut_bits) - 1)
    return (idx | (flip << 9)).to(torch.int16)


def trellis_codes(Q, codes):
    """Pair codes [tiles, 128] from a stored QTIP trellis (codes dict of the port), tiles in decode_packed's order."""
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


def code_dtype(config):
    """The precision the port's successful attempt decoded in (float64 after a float32 Cholesky failure)."""
    return torch.float64 if any(a.get('ok') and 'float64' in a['dtype'] for a in config['attempts']) else torch.float32


def layer_codes(Q, codes, config, weight, table, permute_inv, device):
    """Pair codes of one natively stored QTIP layer, checked to decode to its stored BF16 weight bit for bit."""
    m, n = weight.shape
    if tuple(codes['padded_shape']) != (m, n):
        raise RuntimeError(f'Padded layer {tuple(codes["padded_shape"])} vs {(m, n)}: not handled')
    pairs = trellis_codes(Q, codes)
    dtype = code_dtype(config)
    Wscale = float(codes['Wscale'])
    SU, SV = codes['SU'].float(), codes['SV'].float()
    D = decode(pairs.to(device), table.to(device), m, n, permute_inv)
    hat = D.to(dtype) * torch.tensor(Wscale, dtype=dtype, device=device)
    dq = Q.inverse_incoherence(hat, SU.to(device, torch.bfloat16).to(dtype), SV.to(device, torch.bfloat16).to(dtype))
    if not torch.equal(dq.float().to(torch.bfloat16).cpu(), weight.cpu()):
        raise RuntimeError('Native QTIP codes do not reproduce the stored weight')
    counts = torch.bincount((pairs.long() & (ENTRIES - 1)).reshape(-1), minlength=ENTRIES)
    return dict(pairs=pairs.to(torch.int16), Wscale=Wscale, dtype=str(dtype), SU=SU.to(torch.bfloat16),
                SV=SV.to(torch.bfloat16), counts=counts)


# ---------------- calibrated layers (tables only) ----------------
class QLayer:
    """One QTIP-coded Linear: its stored BF16 weight W0, pair codes and incoherence (SU, SV, Wscale); the group's table
    offset is shared by every layer of the group."""

    def __init__(self, index, name, mod, group, saved, Q, permute, permute_inv, device):
        self.index, self.name, self.mod, self.group, self.Q = index, name, mod, group, Q
        self.permute, self.permute_inv, self.device = permute, permute_inv, device
        self.m, self.n = mod.weight.shape
        self.W0 = mod.weight.detach().to('cpu').clone()
        self.pairs = saved['pairs'].contiguous()
        if torch.cuda.is_available():
            self.pairs = self.pairs.pin_memory()
        self.Wscale = float(saved['Wscale'])
        self.dtype = torch.float64 if 'float64' in str(saved.get('dtype', '')) else torch.float32
        self.SU, self.SV = (saved[k].to(device, torch.float32) for k in ('SU', 'SV'))

    def pairs_device(self):
        return self.pairs.to(self.device, non_blocking=True)

    def decoded(self, table):
        """diag(SV) H_m^T (Wscale D(table)) H_n diag(SU) in the codes' precision, as FP32."""
        D = decode(self.pairs_device(), table.to(self.dtype), self.m, self.n, self.permute_inv)
        return incoherent(self.Q, D * self.Wscale, self.SU.to(self.dtype), self.SV.to(self.dtype)).float()

    def rebuild(self, table, base):
        """W' = bf16(W_hi + dW(table)) into the module, W_hi the FP32 decode of the codes at the base table (its BF16
        rounding is W0; an element the fast form rounds elsewhere keeps W0's value). W' = W0 bitwise at zero offset;
        folding onto W0 itself would round a sub-spacing change back to W0 (ravqc fwrc2r2)."""
        with torch.no_grad():
            if table is not None and bool(table.ne(0).any()):
                w0 = self.W0.to(self.device)
                w = self.decoded(base)
                w = torch.where(w.to(w0.dtype) == w0, w, w0.float()) + self.decoded(table)
            else:
                w = self.W0
            self.mod.weight.data.copy_(w.to(self.mod.weight.dtype))


class QTIPGrad(torch.autograd.Function):
    """y = F.linear(x, W', b) (the plain forward, bitwise); backward, straight through the BF16 cast:
    dD = Wscale hadUt(g o SV)^T hadUt(x o SU) reduced onto the table through the pair codes."""

    @staticmethod
    def forward(ctx, x, table, layer):
        ctx.layer = layer
        ctx.save_for_backward(x)
        return F.linear(x, layer.mod.weight, layer.mod.bias)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        L = ctx.layer
        gx = g.matmul(L.mod.weight).to(x.dtype) if ctx.needs_input_grad[0] else None
        dtab = None
        if ctx.needs_input_grad[1]:
            g2 = g.reshape(-1, L.m).float()
            x2 = x.reshape(-1, L.n).float()
            gh = had(L.Q, g2 * L.SV[None, :], True)
            xh = had(L.Q, x2 * L.SU[None, :], True)
            dD = (gh.t() @ xh).mul_(L.Wscale)
            dtab = decode_adjoint(dD, L.pairs_device(), L.permute, torch.zeros(ENTRIES, 2, device=g.device))
            del gh, xh, dD
        return gx, dtab, None


def first_step_choice(rows, fraction):
    ok = [r['lr'] for r in rows if r['predicted'] > 0 and r['realized'] >= fraction * r['predicted']]
    return max(ok) if ok else None


# ---------------- the reader metric (1) ----------------
class Metric:
    """M = (1 - rho) (D / tau) (E^T diag(lam) E + diag(d)) + rho I on the flattened future slots of the cut state."""

    def __init__(self, saved, device):
        self.E = saved['E'].to(device, torch.float32)
        self.lam = saved['lam'].to(device, torch.float32)
        self.d = saved['d_res'].to(device, torch.float32)
        self.rho, self.D, self.tau = float(saved['rho']), int(saved['D']), float(saved['tau'])
        self.c = (1 - self.rho) * self.D / self.tau

    def quad(self, v):
        v = v.reshape(-1).to(self.E.dtype)
        p = self.E @ v
        return self.c * ((self.lam * p.square()).sum() + (self.d * v.square()).sum()) + self.rho * v.square().sum()

    def nmse(self, z, target):
        z = z.to(self.E.device).float()
        t = target.to(self.E.device, torch.float32)
        return self.quad(z - t) / t.square().sum().clamp_min(1e-12)


def build_metric(vectors, contexts, rank, rho):
    """vectors[c]: [P, D] probes of context c. Per-context unit trace, mean over contexts; top-r eigenpairs of the mean
    and the residual diagonal. Returns the saved form, the spectrum statistics and the diagonal."""
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
                 effective_rank=float(diag.sum() ** 2 / gram.square().sum()),
                 spectrum=[float(x) for x in S[:2 * rank].square()],
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
