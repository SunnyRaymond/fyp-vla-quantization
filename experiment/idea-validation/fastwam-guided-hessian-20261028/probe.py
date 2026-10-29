"""Gauss-Newton probe pieces for the WAM-guided Hessian. No model imports; CPU-testable.

A probe cotangent v with E[v v^T] = M at an output o gives, at a layer output y, g = J^T v with
E[g g^T] = J^T M J, the Gauss-Newton matrix of a loss whose curvature at o is M. Each Probe node records
per-token |g_t|^2 (token weight u) and per-row sum_t g_tc^2 |x_t|^2 (row weight d), and adds R^T g_t x_t^T to the
layer's sketch: summed over every call of a pass, S = R^T G with G = sum_calls sum_t g_t x_t^T the probe-loss gradient
of the shared weight, so E_R[S^T S] = G^T G keeps the cross-token and cross-step terms the token weights drop.
"""
from __future__ import annotations

import math

import torch


class Probe(torch.autograd.Function):
    """Identity on a Linear output that also depends on a zero anchor, so the BF16 model builds a graph.

    backward hands (tag, g, x) to tag[0].record when tag is set; the anchor receives zeros. x is saved through
    save_for_backward on every call (also checkpoint recomputation, where tag is None) so non-reentrant checkpoint
    regions drop and recompute it like any other saved tensor.
    """

    @staticmethod
    def forward(ctx, y, anchor, tag, x):
        ctx.tag, ctx.anchor_dtype = tag, anchor.dtype
        ctx.save_for_backward(x)
        return y.clone()

    @staticmethod
    def backward(ctx, g):
        if ctx.tag is not None:
            x, = ctx.saved_tensors
            ctx.tag[0].record(ctx.tag, g, x)
        return g, torch.zeros((), dtype=ctx.anchor_dtype, device=g.device), None, None


def rademacher(shape, generator, device):
    return torch.randint(0, 2, tuple(shape), generator=generator, device=device).float().mul_(2).sub_(1)


def sketch_matrix(out_features, rows, generator, device):
    """R [out, rows] of +-1 in BF16 (exact); sketch() divides by sqrt(rows), so the effective R has E[R R^T] = I."""
    return rademacher((out_features, rows), generator, device).to(torch.bfloat16)


def token_and_row(g, xn):
    """g [..., M, out] cotangent at a Linear output, xn [M] squared input norms -> u [M], d [out]."""
    gf = g.detach().float().reshape(-1, g.shape[-1]).square()
    return gf.sum(1), (gf * xn[:, None]).sum(0)


def sketch(g, x, r):
    """R^T sum_t g_t x_t^T / sqrt(rows) for one call: g [..., M, out], x [..., M, in], r [out, rows] +-1 -> [rows, in].

    BF16 operands with FP32 accumulation (no FP32 copy of the M x in input); the result is returned in FP32."""
    gr = g.detach().reshape(-1, g.shape[-1]).to(r.dtype) @ r
    return (gr.t() @ x.detach().reshape(-1, x.shape[-1]).to(r.dtype)).float().div_(math.sqrt(r.shape[1]))


def row_sketch(g, x, q):
    """sum_t g_t (Q^T x_t)^T / sqrt(cols) for one call: q [in, cols] +-1 -> [out, cols] FP32; summed over the calls of a
    pass it is G Q, and E_Q |row c|^2 = |G_c|^2, the coherent weight of output row c."""
    xq = x.detach().reshape(-1, x.shape[-1]).to(q.dtype) @ q
    return (g.detach().reshape(-1, g.shape[-1]).to(q.dtype).t() @ xq).float().div_(math.sqrt(q.shape[1]))


def row_scale(d, w, limit):
    """Per-row scale s for quantizing diag(s) W: s^2 = sqrt(d_c) / |W_c| minimizes (sum d_c / s_c^2)(sum s_c^2 |W_c|^2)
    (row loss weight d, rounding error iid across rows after the RHT with variance set by the scaled weight's RMS);
    normalized to geometric mean 1 and clipped to [1/limit, limit]."""
    w2 = w.double().square().sum(1).clamp_min(1e-30)
    d = d.double().clamp_min(1e-12 * float(d.double().mean().clamp_min(1e-30)))
    s = (d.sqrt() / w2.sqrt()).sqrt()
    return (s / s.log().mean().exp()).clamp(1 / limit, limit).float()


def row_ratio(d, s, w):
    """Predicted guided loss of the row scale s against s = 1 under row weights d (same model as row_scale)."""
    w2, d, s2 = w.double().square().sum(1), d.double(), s.double().square()
    return float((d / s2).sum() * (s2 * w2).sum() / (d.sum() * w2.sum()).clamp_min(1e-300))


def coherent_hessian(stack, h_u, gamma):
    """H = S^T S / mean diag + gamma H_U / mean diag for a stacked sketch S [K rows, in] (K over contexts and probes).

    The coherent Gramian has rank <= K; gamma H_U keeps input directions outside its span."""
    s = stack.float()
    h = s.t() @ s
    dh, du = torch.diagonal(h).mean(), torch.diagonal(h_u).mean()
    if not float(dh) > 0:
        return h_u / du
    return h / dh + gamma * h_u / du


def signature(x):
    """Two FP64 sums that identify a Linear input (equal for bitwise-equal inputs)."""
    s = x.detach().double()
    return torch.stack((s.sum(), s.abs().sum()))


def close(a, b, tol):
    scale = max(abs(a[1]), abs(b[1]), 1e-30)
    return abs(a[0] - b[0]) <= tol * scale and abs(a[1] - b[1]) <= tol * scale


def match_suffix(query, reader, tol):
    """Map each reader call to a query call with an equal signature, keeping order and preferring the latest.

    Returns (mapping, worst relative difference). Raises if a reader call has no partner.
    """
    mapping, worst, i = [None] * len(reader), 0.0, len(query) - 1
    for j in range(len(reader) - 1, -1, -1):
        while i >= 0 and not close(query[i], reader[j], tol):
            i -= 1
        if i < 0:
            raise RuntimeError(f'Reader call {j} has no matching query call')
        scale = max(abs(query[i][1]), 1e-30)
        worst = max(worst, abs(query[i][0] - reader[j][0]) / scale, abs(query[i][1] - reader[j][1]) / scale)
        mapping[j] = i
        i -= 1
    return mapping, worst


def softbin_kl_curvature(softbin, teacher, mean, scale, config):
    """Diagonal of d^2 K / da^2 at a = teacher for K = mean per-scalar soft-bin KL(teacher || a).

    Same standardization, edges and sigma as softbin_loss.pointwise_kl, without its clamp_min(0) (which would cut
    the gradient at the exact zero). teacher [..., C]; mean, scale length C.
    """
    t = teacher.detach().float()
    a = t.clone().requires_grad_(True)
    m = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    s = torch.as_tensor(scale, dtype=torch.float32).reshape(-1).clamp_min(config['std_floor'])
    edges = torch.linspace(config['interior_min'], config['interior_max'], 15, dtype=torch.float32)
    tlp = softbin.softbin_log_probs(torch, (t - m) / s, edges, config['sigma'])
    alp = softbin.softbin_log_probs(torch, (a - m) / s, edges, config['sigma'])
    k = (tlp.exp() * (tlp - alp)).sum(-1).mean()
    grad, = torch.autograd.grad(k, a, create_graph=True)
    curvature, = torch.autograd.grad(grad.sum(), a)
    return curvature.detach().clamp_min(0)


def weighted_rows(x, u, ubar, gamma):
    """Rows of x scaled by sqrt(u/ubar + gamma); u None means the call carries no probe signal."""
    rows = x.detach().float().reshape(-1, x.shape[-1])
    w = torch.full((rows.shape[0],), gamma, device=rows.device) if u is None else u.to(rows.device) / ubar + gamma
    return rows * w.sqrt()[:, None], w.sum()


def head_scales(lam, beta):
    return dict(A=1.0, K=math.sqrt(beta), Z=math.sqrt(lam))
