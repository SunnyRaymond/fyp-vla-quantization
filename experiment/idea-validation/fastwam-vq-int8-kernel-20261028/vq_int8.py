"""Fused VQ -> INT8 GEMM for integer-codebook d4 VQ (1b format) and a scalar W4A8 kernel in the same template.

Y[m, n] = (sum_k q[m, k] * w[n, k])_INT32 * a_scale[m] * row_scale[n] + bias[n]  -> BF16
w[n, 4g + c] = C0[i0[n, g], c] + C1[i1[n, g], c], decoded on chip into the INT8 MMA operand.

INT32 accumulation is exact and order independent, so every tiling and split-K choice is bitwise equal to the
FP64-matmul reference followed by the same FP32 epilogue (multiply by a_scale, then row_scale, then add bias).
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# ---------------------------------------------------------------- activation: FP32 operand -> INT8 + per-token scale

@triton.jit
def _quant_a8_kernel(z_ptr, q_ptr, s_ptr, K, BLOCK: tl.constexpr):
    # Same arithmetic as plus_engine._a8_kernel (reciprocal-multiply scale, half-to-even), INT8 + scale output.
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < K
    x = tl.load(z_ptr + row * K + lane, mask, other=0.0).to(tl.float32)
    row_max = tl.max(tl.abs(x), axis=0)
    scale = row_max * (1.0 / 127.0)
    safe_scale = tl.where(scale > 0.0, scale, 1.0)
    scaled = tl.div_rn(x, safe_scale)
    lower = tl.floor(scaled)
    fraction = scaled - lower
    lower_i = lower.to(tl.int32)
    round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower_i & 1) != 0))
    rounded = lower + round_up.to(tl.float32)
    rounded = libdevice.copysign(rounded, scaled)
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    tl.store(q_ptr + row * K + lane, quantized.to(tl.int8), mask)
    tl.store(s_ptr + row, scale)


def quant_a8(z, q_out=None, s_out=None):
    """z [rows, K] FP32 contiguous -> (q INT8 [rows, K], scale FP32 [rows])."""
    rows, width = z.shape
    q = q_out if q_out is not None else torch.empty((rows, width), dtype=torch.int8, device=z.device)
    s = s_out if s_out is not None else torch.empty(rows, dtype=torch.float32, device=z.device)
    block = triton.next_power_of_2(width)
    _quant_a8_kernel[(rows,)](z, q, s, width, block, num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False)
    return q, s


def reference_quant_a8(z):
    z = z.float()
    scale = z.abs().amax(-1) / 127
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(z / safe[:, None]).clamp(-127, 127).to(torch.int8), scale


# ---------------------------------------------------------------- weights

class PreparedVQ8(NamedTuple):
    rows: int
    width: int
    groups: int
    indices: torch.Tensor   # uint8 [N, G, 2]
    books: torch.Tensor     # int8 [2, 256, 4]; books.view(int32) is the packed codeword table [2, 256]
    row_scales: torch.Tensor  # FP32 [N]


def prepare_vq(encoding):
    """1b encoding (BF16 integer books, all exponents 127) -> on-device INT8 tables; indices are reused, not copied."""
    rows, width = map(int, encoding["shape"])
    groups = -(-width // 4)
    books = encoding["books"].float()
    if tuple(books.shape) != (2, 256, 4) or not torch.equal(books, books.round()):
        raise ValueError("Books must be 2x256x4 integers")
    if int(books[0].abs().max() + books[1].abs().max()) > 127:
        raise ValueError("Codebook pair sums can leave the INT8 range")
    exponents = encoding.get("scale_exponents")
    if exponents is not None and not bool((exponents == 127).all()):
        raise ValueError("Microscale exponents are not supported on the INT8 path")
    indices = encoding["indices"]
    if tuple(indices.shape) != (rows, groups, 2) or indices.dtype != torch.uint8 or not indices.is_contiguous():
        raise ValueError("Indices must be contiguous uint8 [N, G, 2]")
    return PreparedVQ8(rows, width, groups, indices, books.to(torch.int8).contiguous(),
                       encoding["row_scales"].float().contiguous())


class PreparedW4(NamedTuple):
    rows: int
    width: int
    nibbles: torch.Tensor      # uint8 [N, ceil(K/2)]: element 2j in the low nibble, 2j+1 in the high nibble
    group_scales: torch.Tensor  # int8 [N, ceil(K/128)]: w8 = q4 * group_scale (QoQ-style progressive)
    row_scales: torch.Tensor    # FP32 [N]


# ---------------------------------------------------------------- fused GEMM

@triton.jit
def _vq_int8_kernel(A, AS, IDX, BOOK32, BOOK8, RS, BIAS, Y, WS,
                    M, N, K, G, SPLIT, PER, stride_am, stride_in, stride_ym,
                    HAS_BIAS: tl.constexpr, WORD: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    # PER = ceil(k_tiles / SPLIT) from the host; tiles past K are fully masked and add zero.
    for i in range(0, PER):
        t = pid_k * PER + i
        rk = t * BK + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * stride_am + rk[None, :], (rm[:, None] < M) & (rk[None, :] < K), other=0)
        if WORD:
            # One 32-bit gather per codebook per 4 weights, then byte extraction (sign-extending shifts).
            rg = t * (BK // 4) + tl.arange(0, BK // 4)
            ip = IDX + rn[:, None] * stride_in + rg[None, :] * 2
            gm = (rn[:, None] < N) & (rg[None, :] < G)
            w0 = tl.load(BOOK32 + tl.load(ip, gm, other=0).to(tl.int32))
            w1 = tl.load(BOOK32 + 256 + tl.load(ip + 1, gm, other=0).to(tl.int32))
            b0 = ((w0 << 24) >> 24) + ((w1 << 24) >> 24)
            b1 = ((w0 << 16) >> 24) + ((w1 << 16) >> 24)
            b2 = ((w0 << 8) >> 24) + ((w1 << 8) >> 24)
            b3 = (w0 >> 24) + (w1 >> 24)
            # join(join(b0,b2), join(b1,b3)) -> [BN, BK/4, 2(i), 2(j)] holding b[2i + j]; flat 4g + 2i + j = k
            w = tl.reshape(tl.join(tl.join(b0, b2), tl.join(b1, b3)), (BN, BK)).to(tl.int8)
        else:
            # Byte gather per weight: simplest decode, two L1-resident 1 KB tables.
            km = (rn[:, None] < N) & (rk[None, :] < K)
            ip = IDX + rn[:, None] * stride_in + (rk[None, :] // 4) * 2
            c = rk[None, :] % 4
            v0 = tl.load(BOOK8 + tl.load(ip, km, other=0).to(tl.int32) * 4 + c)
            v1 = tl.load(BOOK8 + 1024 + tl.load(ip + 1, km, other=0).to(tl.int32) * 4 + c)
            w = (v0.to(tl.int32) + v1.to(tl.int32)).to(tl.int8)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.int32)
    _epilogue(acc, AS, RS, BIAS, Y, WS, rm, rn, M, N, SPLIT, stride_ym, HAS_BIAS)


@triton.jit
def _w4_int8_kernel(A, AS, NIB, GS, RS, BIAS, Y, WS,
                    M, N, K, SPLIT, PER, stride_am, stride_nb, stride_gs, stride_ym,
                    HAS_BIAS: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Scalar W4A8 (per-row INT4 with integer group-128 scales), the QServe/QoQ main-loop pattern.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    # PER = ceil(k_tiles / SPLIT) from the host; tiles past K are fully masked and add zero.
    for i in range(0, PER):
        t = pid_k * PER + i
        rk = t * BK + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * stride_am + rk[None, :], (rm[:, None] < M) & (rk[None, :] < K), other=0)
        rb = t * (BK // 2) + tl.arange(0, BK // 2)
        byte = tl.load(NIB + rn[:, None] * stride_nb + rb[None, :],
                       (rn[:, None] < N) & (rb[None, :] < (K + 1) // 2), other=0).to(tl.int32)
        lo = (byte << 28) >> 28
        hi = (byte << 24) >> 28
        q4 = tl.reshape(tl.join(lo, hi), (BN, BK))
        # BK divides 128, so the tile lies inside one group.
        gs = tl.load(GS + rn * stride_gs + (t * BK) // 128, (rn < N) & (t * BK < K), other=0).to(tl.int32)
        w = (q4 * gs[:, None]).to(tl.int8)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.int32)
    _epilogue(acc, AS, RS, BIAS, Y, WS, rm, rn, M, N, SPLIT, stride_ym, HAS_BIAS)


@triton.jit
def _epilogue(acc, AS, RS, BIAS, Y, WS, rm, rn, M, N, SPLIT, stride_ym, HAS_BIAS: tl.constexpr):
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT == 1:
        a_scale = tl.load(AS + rm, rm < M, other=0.0)
        r_scale = tl.load(RS + rn, rn < N, other=0.0)
        y = acc.to(tl.float32) * a_scale[:, None]
        y = y * r_scale[None, :]
        if HAS_BIAS:
            y = y + tl.load(BIAS + rn, rn < N, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + rm[:, None] * stride_ym + rn[None, :], y.to(tl.bfloat16), mask)
    else:
        # Integer atomics: split-K stays exact and deterministic.
        tl.atomic_add(WS + rm[:, None] * N + rn[None, :], acc, mask=mask)


@triton.jit
def _splitk_epilogue_kernel(WS, AS, RS, BIAS, Y, M, N, stride_ym, HAS_BIAS: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc = tl.load(WS + rm[:, None] * N + rn[None, :], mask, other=0)
    a_scale = tl.load(AS + rm, rm < M, other=0.0)
    r_scale = tl.load(RS + rn, rn < N, other=0.0)
    y = acc.to(tl.float32) * a_scale[:, None]
    y = y * r_scale[None, :]
    if HAS_BIAS:
        y = y + tl.load(BIAS + rn, rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], y.to(tl.bfloat16), mask)


class Config(NamedTuple):
    BM: int
    BN: int
    BK: int
    split: int
    warps: int
    stages: int
    word: bool = True

    @property
    def name(self):
        return (f"bm{self.BM}_bn{self.BN}_bk{self.BK}_s{self.split}_w{self.warps}_p{self.stages}"
                f"_{'word' if self.word else 'byte'}")


def config_from_name(name):
    parts = name.split("_")
    return Config(int(parts[0][2:]), int(parts[1][2:]), int(parts[2][2:]), int(parts[3][1:]),
                  int(parts[4][1:]), int(parts[5][1:]), parts[6] == "word")


def candidates(M, K, N, scalar=False):
    """Tile candidates for one shape; split-K only for the small-M, latency/bandwidth-bound shapes."""
    top = max(16, triton.next_power_of_2(M))
    rows = []
    for bm in (16, 32, 64, 128):
        if bm > top or (M > 64 and bm < 32):
            continue
        for bn in (32, 64, 128):
            for bk in (64, 128):
                k_tiles = -(-K // bk)
                for split in ((1, 2, 4, 8) if M <= 129 else (1,)):
                    if split > 1 and k_tiles < 2 * split:
                        continue
                    warps = 8 if bm * bn >= 8192 else 4
                    for word in ((False,) if scalar else (True, False)):
                        rows.append(Config(bm, bn, bk, split, warps, 3, word))
    return rows


_DUMMY_WS = {}


def _per(K, cfg):
    return -(-(-(-K // cfg.BK)) // cfg.split)


def _launch_common(M, N, cfg, device, bias):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    if cfg.split > 1:
        ws = torch.zeros((M, N), dtype=torch.int32, device=device)
    else:
        # SPLIT is a runtime argument, so the atomic branch is compiled too and needs an INT32 pointer.
        ws = _DUMMY_WS.get(device)
        if ws is None:
            ws = _DUMMY_WS[device] = torch.zeros(1, dtype=torch.int32, device=device)
    grid = (triton.cdiv(N, cfg.BN), triton.cdiv(M, cfg.BM), cfg.split)
    return y, ws, grid


def _finish(y, ws, a_scale, row_scales, bias, M, N, cfg):
    if cfg.split > 1:
        _splitk_epilogue_kernel[(triton.cdiv(N, 64), triton.cdiv(M, 32))](
            ws, a_scale, row_scales, bias if bias is not None else row_scales, y, M, N, y.stride(0),
            bias is not None, 32, 64, num_warps=4, enable_fp_fusion=False)
    return y


def vq_linear(q, a_scale, prepared, bias, cfg):
    """q INT8 [M, K] contiguous, a_scale FP32 [M] -> BF16 [M, N]."""
    M, K = q.shape
    N = prepared.rows
    y, ws, grid = _launch_common(M, N, cfg, q.device, bias)
    _vq_int8_kernel[grid](
        q, a_scale, prepared.indices, prepared.books.view(torch.int32), prepared.books,
        prepared.row_scales, bias if bias is not None else prepared.row_scales, y, ws,
        M, N, K, prepared.groups, cfg.split, _per(K, cfg), q.stride(0), prepared.indices.stride(0), y.stride(0),
        bias is not None, cfg.word, cfg.BM, cfg.BN, cfg.BK,
        num_warps=cfg.warps, num_stages=cfg.stages, enable_fp_fusion=False)
    return _finish(y, ws, a_scale, prepared.row_scales, bias, M, N, cfg)


def w4_linear(q, a_scale, prepared, bias, cfg):
    M, K = q.shape
    N = prepared.rows
    y, ws, grid = _launch_common(M, N, cfg, q.device, bias)
    _w4_int8_kernel[grid](
        q, a_scale, prepared.nibbles, prepared.group_scales, prepared.row_scales,
        bias if bias is not None else prepared.row_scales, y, ws,
        M, N, K, cfg.split, _per(K, cfg), q.stride(0), prepared.nibbles.stride(0), prepared.group_scales.stride(0), y.stride(0),
        bias is not None, cfg.BM, cfg.BN, cfg.BK,
        num_warps=cfg.warps, num_stages=cfg.stages, enable_fp_fusion=False)
    return _finish(y, ws, a_scale, prepared.row_scales, bias, M, N, cfg)


# ---------------------------------------------------------------- references (exact)

def decode_vq_int(prepared):
    books = prepared.books.long()
    idx = prepared.indices.long()
    w = books[0][idx[..., 0]] + books[1][idx[..., 1]]           # [N, G, 4]
    return w.reshape(prepared.rows, -1)[:, :prepared.width]


def decode_w4_int(prepared):
    b = prepared.nibbles.to(torch.int32)
    lo = (b << 28) >> 28
    hi = (b << 24) >> 28
    q4 = torch.stack((lo, hi), -1).reshape(prepared.rows, -1)[:, :prepared.width]
    gs = prepared.group_scales.to(torch.int32).repeat_interleave(128, 1)[:, :prepared.width]
    return q4 * gs


def reference_linear(q, a_scale, w_int, row_scales, bias):
    """FP64 matmul of integers is exact (|acc| < 2^31 << 2^53); then the kernel's FP32 epilogue order."""
    acc = (q.double() @ w_int.double().t()).to(torch.int32)
    y = acc.float() * a_scale[:, None]
    y = y * row_scales[None, :]
    if bias is not None:
        y = y + bias.float()[None, :]
    return y.to(torch.bfloat16)


# ---------------------------------------------------------------- random weights for the microbenchmark

def random_vq(N, K, device, generator, bound=63):
    groups = -(-K // 4)
    books = torch.randint(-bound, bound + 1, (2, 256, 4), device=device, generator=generator).to(torch.int8)
    indices = torch.randint(0, 256, (N, groups, 2), device=device, generator=generator).to(torch.uint8)
    scales = torch.rand(N, device=device, generator=generator) * 1e-3
    return PreparedVQ8(N, K, groups, indices.contiguous(), books.contiguous(), scales)


def random_w4(N, K, device, generator):
    nibbles = torch.randint(0, 256, (N, -(-K // 2)), device=device, generator=generator).to(torch.uint8)
    gs = torch.randint(1, 16, (N, -(-K // 128)), device=device, generator=generator).to(torch.int8)
    scales = torch.rand(N, device=device, generator=generator) * 1e-3
    return PreparedW4(N, K, nibbles.contiguous(), gs.contiguous(), scales)
