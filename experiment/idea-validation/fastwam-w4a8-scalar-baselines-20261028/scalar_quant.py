"""Symmetric INT4 g128 weights (RTN / GPTQ), per-token A8 operands, nibble packing. No model imports."""
from __future__ import annotations

import torch

QMAX = 7
GROUP = 128


def a8(z):
    """Per-token symmetric INT8 grid with BF16 output (identical to the 1b A8 forward)."""
    z = z.float()
    scale = z.abs().amax(-1, keepdim=True) / 127
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(z / safe).clamp(-127, 127) * scale).to(torch.bfloat16)


def _codes(w, scale):
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(w / safe).clamp(-QMAX, QMAX)
    return torch.where(scale > 0, q, torch.zeros_like(q))


def absmax_scale(w):
    """[rows, g] -> BF16-representable FP32 scale [rows, 1]."""
    return (w.abs().amax(1, keepdim=True) / QMAX).to(torch.bfloat16).float()


def mse_scale(w, grid=20, maxshrink=0.2, norm=2.4):
    """GPTQ/QuaRot w_clip: shrink the absmax scale to minimize sum |w - q|^norm per row."""
    amax = w.abs().amax(1, keepdim=True)
    best, best_err = None, None
    for i in range(grid + 1):
        s = (amax * (1 - maxshrink * i / grid) / QMAX).to(torch.bfloat16).float()
        err = (_codes(w, s) * s - w).abs().pow(norm).sum(1, keepdim=True)
        if best is None:
            best, best_err = s, err
        else:
            better = err < best_err
            best, best_err = torch.where(better, s, best), torch.where(better, err, best_err)
    return best


def rtn(weight):
    """Per-group absmax RTN. Returns int8 codes [out, K] and BF16 scales [out, ceil(K/128)]."""
    w = weight.float()
    out, k = w.shape
    codes = torch.empty((out, k), dtype=torch.int8, device=w.device)
    scales = []
    for start in range(0, k, GROUP):
        part = w[:, start:start + GROUP]
        s = absmax_scale(part)
        codes[:, start:start + GROUP] = _codes(part, s).to(torch.int8)
        scales.append(s)
    return codes, torch.cat(scales, 1).to(torch.bfloat16)


def gptq(weight, hessian, damp=0.01, block=GROUP):
    """GPTQ (no act-order) with group-128 MSE-clipped scales chosen on the updated weights at each group start."""
    if block != GROUP:
        raise ValueError('block must equal the group size so group starts align with block starts')
    w = weight.float().clone()
    h = hessian.float().clone()
    out, k = w.shape
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damp * torch.diag(h).mean())
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h
    codes = torch.empty((out, k), dtype=torch.int8, device=w.device)
    scales = []
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = hinv[i1:i2, i1:i2]
        s = mse_scale(w1)
        scales.append(s)
        for i in range(i2 - i1):
            col = w1[:, i]
            q = _codes(col[:, None], s)[:, 0]
            codes[:, i1 + i] = q.to(torch.int8)
            e = (col - q * s[:, 0]) / h1[i, i]
            w1[:, i:] -= e[:, None] * h1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes, torch.cat(scales, 1).to(torch.bfloat16)


def dequant(codes, scales):
    k = codes.shape[1]
    return codes.float() * scales.float().repeat_interleave(GROUP, 1)[:, :k]


def proxy_nmse(weight, decoded, hessian):
    """tr(dW H dW^T) / tr(W H W^T): output-energy error on the calibration distribution."""
    w, d = weight.float(), decoded.float() - weight.float()
    return float(((d @ hessian) * d).sum() / ((w @ hessian) * w).sum().clamp_min(1e-30))


def pack(codes):
    """int8 codes in [-7, 7] -> uint8 nibbles (two per byte, low first); odd widths pad one zero code."""
    u = (codes.to(torch.int16) + 8).to(torch.uint8)
    if u.shape[1] % 2:
        u = torch.cat((u, torch.full((u.shape[0], 1), 8, dtype=torch.uint8, device=u.device)), 1)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def unpack(packed, k):
    lo, hi = packed & 15, packed >> 4
    return (torch.stack((lo, hi), 2).reshape(packed.shape[0], -1)[:, :k].to(torch.int16) - 8).to(torch.int8)


def bpw(out, k):
    groups = -(-k // GROUP)
    return (4 * out * k + 16 * out * groups) / (out * k)
