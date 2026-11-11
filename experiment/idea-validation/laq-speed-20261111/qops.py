"""Arithmetic of the four published recipes (FP32, torch only).

  fht / had              orthonormal Hadamard transforms of the official QuaRot code (2^k, 12*2^k, 28*2^k widths)
  a_token / a_group      the per-token and per-token-per-group activation/KV quantizers of QuaRot and SpinQuant
  a_int4_group           SVDQuant's INT4 dynamic quantizer (per token, 64 channels per group, signed or unsigned)
  gptq_row               QuaRot/SpinQuant GPTQ: per-row symmetric INT4, MSE clip search, blocked Cholesky updates
  rtn_int4               SVDQuant residual INT4 (per row, group 64)
  vid_quant              ViDiT-Q static affine weights (per row, 4 or 8 bits)
"""
from __future__ import annotations

import math

import torch

_H12 = torch.tensor([
    [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
    [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1], [1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1], [1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
    [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
    [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1], [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
    [1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1], [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1]], dtype=torch.float32)
# get_had28 of spcl/QuaRot fake_quant/hadamard_utils.py; rows are checked orthogonal in the unit tests.
_H28 = torch.tensor([1.0 if c == "+" else -1.0 for c in (
    "++++++++++++++-++++++++++++++++-++----++-++-+-++----++-+++++-++----++-++-+-++----++-+-+++-++----+++-+-+-++----++"
    "++-+++-++----+++-+-+-++----++++-+++-++----+++-+-+-++----+-++-+++-++---+-++-+-+-++---+--++-+++-++--+--++-+-+-++--"
    "+---++-+++-++-+---++-+-+-++-+----++-+++-+++----++-+-+-++++----++-+++-+++----++-+-+-++++----++-+++-+++----++-+-+-"
    "+-++----++-++++-++----++-+-+++-++----++-++++-++----++-+--+++++++++++++--------------+-+-++----++-+---+--++++--+-"
    "++-+-++----++-----+--++++--++-+-+-++----++-+---+--++++--++-+-+-++----+--+---+--++++-+++-+-+-++-------+---+--++++"
    "+-++-+-+-++----+--+---+--++++--++-+-+-++---++--+---+--+++---++-+-+-++--+++--+---+--++----++-+-+-++-++++--+---+--"
    "++----++-+-+-+--++++--+---+-+++----++-+-+----++++--+---++-++----++-+-+-+--++++--+---++-++----++-+---+--++++--+--")],
    dtype=torch.float32).reshape(28, 28)
_BASE = {12: _H12, 28: _H28}


def fht(x):
    """Orthonormal Walsh-Hadamard transform over a power-of-two last dimension."""
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Power-of-two width required, got {n}")
    y = x
    h = 1
    while h < n:
        a, b = y.reshape(*y.shape[:-1], n // (2 * h), 2, h).unbind(-2)
        y = torch.stack((a + b, a - b), -2).reshape(x.shape)
        h *= 2
    return y / math.sqrt(n)


CACHE_BASE = True  # keep the scaled 12 x 12 / 28 x 28 bases on the device: no host-to-device copy (a sync) per call
_BASE_CACHE = {}


def _base(k, device, dtype):
    if not CACHE_BASE:
        return _BASE[k].to(device, dtype) / math.sqrt(k)
    key = (k, device, dtype)
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = _BASE[k].to(device, dtype) / math.sqrt(k)
    return _BASE_CACHE[key]


def had(x, inverse=False):
    """x @ H for widths 2^k, 12 * 2^k and 28 * 2^k; `inverse` applies H^T."""
    n = x.shape[-1]
    k = 28 if n % 28 == 0 else 12 if n % 12 == 0 else 1
    if k == 1:
        return fht(x)
    y = fht(x.reshape(*x.shape[:-1], k, n // k))
    base = _base(k, y.device, y.dtype)
    return ((base.t() if inverse else base) @ y).reshape(x.shape)


def ste(quantized, x):
    return x + (quantized - x).detach()


def a_token(x, sym, clip=1.0, bits=4):
    """Per-token dynamic quantizer of quant_utils.ActQuantizer (group -1): range includes zero, scaled by clip."""
    low = x.amin(-1, keepdim=True).clamp_max(0) * clip
    high = x.amax(-1, keepdim=True).clamp_min(0) * clip
    if sym:
        qmax = (1 << (bits - 1)) - 1
        m = torch.maximum(low.abs(), high)
        scale = torch.where(m == 0, torch.ones_like(m), m / qmax)
        return torch.round(x / scale).clamp(-qmax - 1, qmax) * scale
    both = (low == 0) & (high == 0)
    low = torch.where(both, -torch.ones_like(low), low)
    high = torch.where(both, torch.ones_like(high), high)
    qmax = (1 << bits) - 1
    scale = (high - low) / qmax
    zero = torch.round(-low / scale)
    return (torch.round(x / scale) + zero).clamp(0, qmax).sub(zero) * scale


def a_group(x, group, clip=1.0, bits=4):
    """Per-token group-wise asymmetric quantizer (find_params_per_token_groupwise): raw min/max of each group."""
    d = x.shape[-1]
    if d % group:
        raise ValueError(f"Width {d} is not a multiple of group {group}")
    b = x.reshape(*x.shape[:-1], d // group, group)
    high = b.amax(-1, keepdim=True) * clip
    low = b.amin(-1, keepdim=True) * clip
    both = (low == 0) & (high == 0)
    low = torch.where(both, -torch.ones_like(low), low)
    high = torch.where(both, torch.ones_like(high), high)
    qmax = (1 << bits) - 1
    scale = (high - low) / qmax
    zero = torch.round(-low / scale)
    return ((torch.round(b / scale) + zero).clamp(0, qmax).sub(zero) * scale).reshape(x.shape)


def a_int4_group(x, group=64, unsigned=False):
    """SVDQuant INT4 dynamic activation: per token, per `group` channels; unsigned for shifted inputs."""
    d = x.shape[-1]
    head = d // group * group
    pieces = []
    for part, back in (((x[..., :head].reshape(*x.shape[:-1], -1, group), (*x.shape[:-1], head)) if head else (None, None)),
                       ((x[..., head:], None) if head < d else (None, None))):
        if part is None:
            continue
        if unsigned:
            span = part.amax(-1, keepdim=True).clamp_min(0)
            scale = torch.where(span == 0, torch.ones_like(span), span / 15)
            q = torch.round(part / scale).clamp(0, 15) * scale
        else:
            span = part.abs().amax(-1, keepdim=True)
            scale = torch.where(span == 0, torch.ones_like(span), span / 7)
            q = torch.round(part / scale).clamp(-8, 7) * scale
        pieces.append(q if back is None else q.reshape(back))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, -1)


def _pack4(u):
    if u.shape[1] % 2:
        u = torch.cat((u, torch.zeros((u.shape[0], 1), dtype=torch.uint8, device=u.device)), 1)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def _unpack4(packed, k):
    return torch.stack((packed & 15, packed >> 4), 2).reshape(packed.shape[0], -1)[:, :k]


def clip_scales(w, bits=4, grid=100, maxshrink=0.8, norm=2.4):
    """WeightQuantizer.find_params with mse=True: per-row symmetric scales from an Lp linear search. FP32 [rows]."""
    qmax = (1 << (bits - 1)) - 1
    high = torch.maximum(w.amin(1).clamp_max(0).abs(), w.amax(1).clamp_min(0)).clamp(min=1e-5)
    best = torch.full((w.shape[0],), float("inf"), device=w.device)
    out = high / qmax
    for i in range(int(maxshrink * grid)):
        s = high * (1 - i / grid) / qmax
        err = ((torch.round(w / s[:, None]).clamp(-qmax - 1, qmax) * s[:, None] - w).abs().pow(norm)).sum(1)
        better = err < best
        best = torch.where(better, err, best)
        out = torch.where(better, s, out)
    return out


def gptq_row(weight, hessian, damp=0.01, blocksize=128, bits=4):
    """Blocked GPTQ without act-order, with one MSE-clipped scale per row fixed at the start (group size -1).
    Returns int8 codes in [-8, 7] and FP32 scales [rows]."""
    w = weight.float().clone()
    h = hessian.float().clone()
    qmax = (1 << (bits - 1)) - 1
    scale = clip_scales(w, bits)
    _, k = w.shape
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damp * torch.diag(h).mean())
    inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h
    codes = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = inv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            q = torch.round(col / scale).clamp(-qmax - 1, qmax)
            codes[:, i1 + i] = q.to(torch.int8)
            e = (col - q * scale) / h1[i, i]
            w1[:, i:] -= e[:, None] * h1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ inv[i1:i2, i2:]
    return codes, scale


def rtn_int4(weight, group=64):
    """Symmetric INT4 rounding per row per `group` input channels (SVDQuant); int8 codes and FP32 scales [rows, groups]."""
    w = weight.float()
    codes = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    scales = []
    for start in range(0, w.shape[1], group):
        part = w[:, start:start + group]
        s = part.abs().amax(1, keepdim=True) / 7
        safe = torch.where(s > 0, s, torch.ones_like(s))
        codes[:, start:start + group] = torch.round(part / safe).clamp(-8, 7).to(torch.int8)
        scales.append(s)
    return codes, torch.cat(scales, 1)


def encode_sym(codes, scales, group):
    """Packed INT4 codes (+8 offset) and FP32 scales [rows, groups]."""
    return dict(kind="sym", packed=_pack4((codes.to(torch.int16) + 8).to(torch.uint8)), scales=scales.float(),
                shape=list(codes.shape), group=group, bits=4)


def decode_sym(bank, dtype=torch.bfloat16):
    r, k = bank["shape"]
    codes = _unpack4(bank["packed"], k).float() - 8
    group = bank["group"]
    return (codes * bank["scales"].float().repeat_interleave(group, 1)[:, :k]).to(dtype)


def vid_quant(w, bits):
    """ViDiT-Q StaticQuantizer (asymmetric, one delta and zero point per output row).
    Returns integer codes (zero-offset) [r, k], delta [r, 1], zero point [r, 1]."""
    high = w.amax(1, keepdim=True).clamp_min(0)
    low = w.amin(1, keepdim=True).clamp_max(0)
    levels = 1 << bits
    delta = ((high - low) / (levels - 1)).clamp_min(1e-8)
    zero = torch.round(low / delta) + levels / 2
    codes = (torch.round(w / delta) - zero).clamp(-levels // 2, levels // 2 - 1)
    return codes, delta, zero


def vid_dequant(codes, delta, zero):
    return (codes + zero) * delta


def encode_vid(codes, delta, zero, bits):
    r, k = codes.shape
    u = (codes + (1 << (bits - 1))).to(torch.uint8)
    return dict(kind="vid", packed=_pack4(u) if bits == 4 else u, delta=delta.float(), zero=zero.to(torch.int16),
                shape=[r, k], bits=bits)


def decode_vid(bank, dtype=torch.bfloat16):
    r, k = bank["shape"]
    bits = bank["bits"]
    u = _unpack4(bank["packed"], k) if bits == 4 else bank["packed"]
    codes = u.float() - (1 << (bits - 1))
    return vid_dequant(codes, bank["delta"].float(), bank["zero"].float()).to(dtype)


def decode(bank, dtype=torch.bfloat16):
    return decode_sym(bank, dtype) if bank["kind"] == "sym" else decode_vid(bank, dtype)


def random_signs(width, seed, device):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.randint(0, 2, (width,), generator=g).to(device=device, dtype=torch.float32) * 2 - 1)


def proxy_nmse(decoded, target, hessian):
    """tr(dW H dW^T) / tr(W H W^T) in the coordinates of `hessian`."""
    d = decoded.float() - target.float()
    return float(((d @ hessian) * d).sum() / ((target.float() @ hessian) * target.float()).sum().clamp_min(1e-30))
