"""W4A4KV4 arithmetic shared by the preparation and the closed-loop workers. Imports torch only.

Grids (all INT4, one scale per `group` consecutive channels; a shorter tail forms its own group):
  q_sym / q_aff   dynamic per-token activation and key/value quantizers, BF16 output
  gptq            second-order weight rounding with per-(row, group) scales
  encode_sym / encode_aff / decode   packed weight codes <-> dequantized weights
  rotate          signs + full Hadamard over the input width (ViDiT-Q); dyadic tail for widths without a real Hadamard
"""
from __future__ import annotations

import math

import torch

QMAX = 7


def _groups(x, group):
    """Yield (block [..., n, g], reshape-back shape) for the head and the tail of the last dimension."""
    d = x.shape[-1]
    head = d // group * group
    if head:
        yield x[..., :head].reshape(*x.shape[:-1], -1, group), (*x.shape[:-1], head)
    if head < d:
        yield x[..., head:], None


def _grid_sym(b):
    s = b.abs().amax(-1, keepdim=True) / QMAX
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.round(b / safe).clamp(-QMAX, QMAX) * s


def _grid_aff(b):
    low = b.amin(-1, keepdim=True).clamp_max(0)
    high = b.amax(-1, keepdim=True).clamp_min(0)
    s = (high - low) / 15
    safe = torch.where(s > 0, s, torch.ones_like(s))
    zero = (-low / safe).round().clamp(0, 15)
    return ((b / safe).round().add(zero).clamp(0, 15) - zero) * s


def _apply(x, group, grid):
    pieces = [grid(b) if back is None else grid(b).reshape(back) for b, back in _groups(x.float(), group)]
    return (pieces[0] if len(pieces) == 1 else torch.cat(pieces, -1)).to(torch.bfloat16)


def q_sym(x, group=32):
    """Dynamic symmetric INT4 per token (per group of channels), BF16 output."""
    return _apply(x, group, _grid_sym)


def q_aff(x, group=32):
    """Dynamic affine zero-including INT4 per token (per group of channels), BF16 output."""
    return _apply(x, group, _grid_aff)


def _codes(w, scale):
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(w / safe).clamp(-QMAX, QMAX)
    return torch.where(scale > 0, q, torch.zeros_like(q))


def _mse_scale(w, grid=20, maxshrink=0.2, norm=2.4):
    """Shrink the absmax scale of each row to minimize sum |w - q|^norm; BF16-representable FP32 scale [rows, 1]."""
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


def gptq(weight, hessian, group=32, damp=0.01):
    """GPTQ (no act-order) with one MSE-clipped scale per (row, group), chosen on the updated weights at the group
    start. Returns int8 codes [out, k] and BF16 scales [out, ceil(k / group)]."""
    w = weight.float().clone()
    h = hessian.float().clone()
    _, k = w.shape
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damp * torch.diag(h).mean())
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h
    codes = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    scales = []
    for i1 in range(0, k, group):
        i2 = min(i1 + group, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = hinv[i1:i2, i1:i2]
        s = _mse_scale(w1)
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


def rtn_sym(weight, group=32):
    """Per-(row, group) absmax symmetric rounding; int8 codes and BF16 scales."""
    w = weight.float()
    codes = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    scales = []
    for start in range(0, w.shape[1], group):
        part = w[:, start:start + group]
        s = (part.abs().amax(1, keepdim=True) / QMAX).to(torch.bfloat16).float()
        codes[:, start:start + group] = _codes(part, s).to(torch.int8)
        scales.append(s)
    return codes, torch.cat(scales, 1).to(torch.bfloat16)


def _pack(u):
    """uint8 values in [0, 15] -> two per byte, low nibble first; an odd width pads one zero code."""
    if u.shape[1] % 2:
        u = torch.cat((u, torch.zeros((u.shape[0], 1), dtype=torch.uint8, device=u.device)), 1)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def _unpack(packed, k):
    return torch.stack((packed & 15, packed >> 4), 2).reshape(packed.shape[0], -1)[:, :k]


def encode_sym(codes, scales, group):
    return dict(packed=_pack((codes.to(torch.int16) + 8).to(torch.uint8)), scales=scales, zero_points=None,
                shape=list(codes.shape), group=group, affine=False)


def encode_aff(weight, group):
    """Per-(row, group) affine zero-including rounding with BF16 scales and uint8 zero points."""
    w = weight.float()
    codes = torch.empty(w.shape, dtype=torch.uint8, device=w.device)
    scales, zeros = [], []
    for start in range(0, w.shape[1], group):
        part = w[:, start:start + group]
        low = part.amin(1, keepdim=True).clamp_max(0)
        high = part.amax(1, keepdim=True).clamp_min(0)
        s = ((high - low) / 15).to(torch.bfloat16).float()
        safe = torch.where(s > 0, s, torch.ones_like(s))
        z = (-low / safe).round().clamp(0, 15)
        codes[:, start:start + group] = ((part / safe).round().add(z).clamp(0, 15)).to(torch.uint8)
        scales.append(s)
        zeros.append(z.to(torch.uint8))
    return dict(packed=_pack(codes), scales=torch.cat(scales, 1).to(torch.bfloat16), zero_points=torch.cat(zeros, 1),
                shape=list(w.shape), group=group, affine=True)


def decode(bank, dtype=torch.bfloat16):
    """Dequantized weight [out, k] from packed codes."""
    _, k = bank["shape"]
    group = bank["group"]
    codes = _unpack(bank["packed"], k).float()
    scales = bank["scales"].float().repeat_interleave(group, 1)[:, :k]
    offset = bank["zero_points"].float().repeat_interleave(group, 1)[:, :k] if bank["affine"] else 8.0
    return ((codes - offset) * scales).to(dtype)


def proxy_nmse(decoded, target, hessian):
    """tr(dW H dW^T) / tr(W H W^T) of the decoded weight against its target, in the coordinates of `hessian`."""
    d = decoded.float() - target.float()
    return float(((d @ hessian) * d).sum() / ((target.float() @ hessian) * target.float()).sum().clamp_min(1e-30))


def fwht(x):
    """Orthonormal Walsh-Hadamard transform over a power-of-two last dimension."""
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"FWHT needs a power-of-two width, got {n}")
    y = x.float()
    h = 1
    while h < n:
        parts = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = parts.unbind(-2)
        y = torch.stack((a + b, a - b), -2).reshape(x.shape)
        h *= 2
    return y * (1.0 / math.sqrt(n))


_H12 = torch.tensor([
    [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
    [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1], [1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1], [1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
    [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
    [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1], [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
    [1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1], [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1]], dtype=torch.float32)


def _power2(n):
    return n > 0 and n & (n - 1) == 0


def rotate(value, signs):
    """x @ diag(signs) @ H: full Hadamard for 2^k and 12 * 2^k widths, otherwise dyadic blocks. FP32."""
    x = value.float() * signs.float()
    n = x.shape[-1]
    if _power2(n):
        return fwht(x)
    if n % 12 == 0 and _power2(n // 12):
        h = _H12.to(x.device) / math.sqrt(12)
        part = fwht(x.reshape(*x.shape[:-1], 12, n // 12))
        return (part.transpose(-1, -2) @ h).transpose(-1, -2).reshape(x.shape)
    pieces, offset = [], 0
    while offset < n:
        span = 1 << ((n - offset).bit_length() - 1)
        pieces.append(fwht(x[..., offset:offset + span]))
        offset += span
    return torch.cat(pieces, -1)


def random_signs(width, seed, device):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.randint(0, 2, (width,), generator=g).to(device=device, dtype=torch.int8) * 2 - 1).to(torch.int8)


def bank_bits(bank):
    """Stored bits of one layer: codes, scales, zero points, channel vectors, signs, low-rank factors."""
    bits = bank["packed"].numel() * 8 + bank["scales"].numel() * 16
    if bank.get("zero_points") is not None:
        bits += bank["zero_points"].numel() * 8
    if bank.get("smoothing") is not None:
        bits += bank["smoothing"].numel() * 32
    if bank.get("signs") is not None:
        bits += bank["signs"].numel() * 8
    for key in ("up", "down"):
        if bank.get(key) is not None:
            bits += bank[key].numel() * 16
    return int(bits)
