"""Triton kernels of the integer LAQ runtime (GPU only; int4.TorchOps is the reference they are checked against).

  quant    rotated FP32 operand -> INT4 codes in an INT8 container + one FP32 scale per row; the refined rows get the
           first pass in place and their second pass as extra rows appended after the T token rows
  unpack   packed 4-bit weights (two codes per byte, offset 8) -> INT8 scratch for the cuBLASLt INT8 GEMM
  epilogue INT32 accumulators -> BF16 output: (acc * token scale [+ acc of the second pass * its scale]) * row scale + bias
  kvq      quantize-dequantize of cached keys / values per (token, head), two passes on the refined tokens
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rne(v):
    lower = tl.floor(v)
    frac = v - lower
    up = (frac > 0.5) | ((frac == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    return lower + up.to(tl.float32)


@triton.jit
def _a_base(X, Q, S, K, CLIP, SEVEN, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < K
    x = tl.load(X + row * K + lane, mask, other=0.0)
    m = tl.max(tl.abs(x), axis=0) * CLIP
    s = tl.where(m == 0.0, 1.0, tl.div_rn(m, SEVEN))
    c = tl.minimum(tl.maximum(_rne(tl.div_rn(x, s)), -8.0), 7.0)
    tl.store(Q + row * K + lane, c.to(tl.int8), mask)
    tl.store(S + row, s)


@triton.jit
def _a_fine(X, IDX, Q, S, T, K, SEVEN, BLOCK: tl.constexpr):
    j = tl.program_id(0)
    row = tl.load(IDX + j)
    lane = tl.arange(0, BLOCK)
    mask = lane < K
    x = tl.load(X + row * K + lane, mask, other=0.0)
    m1 = tl.max(tl.abs(x), axis=0)
    s1 = tl.where(m1 == 0.0, 1.0, tl.div_rn(m1, SEVEN))
    q1 = tl.minimum(tl.maximum(_rne(tl.div_rn(x, s1)), -8.0), 7.0)
    res = x - q1 * s1
    m2 = tl.max(tl.abs(res), axis=0)
    s2 = tl.where(m2 == 0.0, 1.0, tl.div_rn(m2, SEVEN))
    q2 = tl.minimum(tl.maximum(_rne(tl.div_rn(res, s2)), -8.0), 7.0)
    tl.store(Q + row * K + lane, q1.to(tl.int8), mask)
    tl.store(S + row, s1)
    tl.store(Q + (T + j) * K + lane, q2.to(tl.int8), mask)
    tl.store(S + T + j, s2)


def quant(r, idx, clip):
    """r FP32 [T, K] contiguous, idx int64 [R] or None -> (codes INT8 [T + R, K], scales FP32 [T + R])."""
    T, K = r.shape
    R = 0 if idx is None else int(idx.numel())
    codes = torch.empty((T + R, K), dtype=torch.int8, device=r.device)
    scales = torch.empty(T + R, dtype=torch.float32, device=r.device)
    block = triton.next_power_of_2(K)
    warps = 8 if block >= 8192 else 4
    _a_base[(T,)](r, codes, scales, K, clip, 7.0, block, num_warps=warps, enable_fp_fusion=False)
    if R:
        _a_fine[(R,)](r, idx, codes, scales, T, K, 7.0, block, num_warps=warps, enable_fp_fusion=False)
    return codes, scales


@triton.jit
def _unpack(P, W, NBYTES, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NBYTES
    b = tl.load(P + offs, mask, other=0).to(tl.int32)
    tl.store(W + 2 * offs, ((b & 15) - 8).to(tl.int8), mask)
    tl.store(W + 2 * offs + 1, ((b >> 4) - 8).to(tl.int8), mask)


def unpack(packed, out):
    """packed uint8 [N, K / 2] contiguous -> out INT8 [N, K] (K even)."""
    nbytes = packed.numel()
    _unpack[(triton.cdiv(nbytes, 4096),)](packed, out, nbytes, 4096, num_warps=4)
    return out


@triton.jit
def _epilogue(ACC, S, RS, BIAS, PART, Y, T, N, HAS_BIAS: tl.constexpr, HAS_REF: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    mm = rm < T
    mn = rn < N
    mask = mm[:, None] & mn[None, :]
    acc = tl.load(ACC + rm[:, None] * N + rn[None, :], mask, other=0)
    t = acc.to(tl.float32) * tl.load(S + rm, mm, other=0.0)[:, None]
    if HAS_REF:
        p = tl.load(PART + rm, mm, other=-1)
        has = p >= 0
        row2 = T + tl.where(has, p, 0)
        acc2 = tl.load(ACC + row2[:, None] * N + rn[None, :], mask & has[:, None], other=0)
        t = t + acc2.to(tl.float32) * tl.load(S + row2, mm & has, other=0.0)[:, None]
    y = t * tl.load(RS + rn, mn, other=0.0)[None, :]
    if HAS_BIAS:
        y = y + tl.load(BIAS + rn, mn, other=0.0)[None, :]
    tl.store(Y + rm[:, None] * N + rn[None, :], y.to(tl.bfloat16), mask)


def epilogue(acc, scales, row_scales, bias, part, T):
    """acc INT32 [T + R, N] contiguous -> BF16 [T, N]; part INT32 [T] maps a refined row to its second-pass row, else -1."""
    N = acc.shape[1]
    y = torch.empty((T, N), dtype=torch.bfloat16, device=acc.device)
    _epilogue[(triton.cdiv(N, 256), triton.cdiv(T, 16))](
        acc, scales, row_scales, bias if bias is not None else row_scales, part if part is not None else acc, y, T, N,
        bias is not None, part is not None, 16, 256, num_warps=4, enable_fp_fusion=False)
    return y


@triton.jit
def _asym(x, clip, zero_in, fifteen):
    lo = tl.min(x, axis=0)
    hi = tl.max(x, axis=0)
    lo = tl.where(zero_in != 0, tl.minimum(lo, 0.0), lo) * clip
    hi = tl.where(zero_in != 0, tl.maximum(hi, 0.0), hi) * clip
    both = (lo == 0.0) & (hi == 0.0)
    lo = tl.where(both, -1.0, lo)
    hi = tl.where(both, 1.0, hi)
    scale = tl.div_rn(hi - lo, fifteen)
    zero = _rne(tl.div_rn(-lo, scale))
    code = tl.minimum(tl.maximum(_rne(tl.div_rn(x, scale)) + zero, 0.0), 15.0)
    return (code - zero) * scale


@triton.jit
def _kvq(X, FLAG, Y, H, CLIP, ZERO_IN, ONE, FIFTEEN, D: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, D)
    x = tl.load(X + row * D + lane)
    base = _asym(x, CLIP, ZERO_IN, FIFTEEN)
    first = _asym(x, ONE, ZERO_IN, FIFTEEN)
    fine = first + _asym(x - first, ONE, ZERO_IN, FIFTEEN)
    refined = (lane * 0 + tl.load(FLAG + row // H).to(tl.int32)) != 0
    tl.store(Y + row * D + lane, tl.where(refined, fine, base))


def kvq(r, flags, heads, clip, zero_in):
    """r FP32 [T * heads, D] contiguous (D a power of two), flags INT8 [T] -> dequantized FP32 [T * heads, D]."""
    rows, d = r.shape
    if d & (d - 1):
        raise ValueError(f"Head dimension {d} is not a power of two")
    out = torch.empty_like(r)
    _kvq[(rows,)](r, flags, out, heads, clip, int(zero_in), 1.0, 15.0, d, num_warps=4, enable_fp_fusion=False)
    return out
