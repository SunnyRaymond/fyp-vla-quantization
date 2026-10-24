"""Independent VQ GEMM layout/tiling prototype for PBS-side evaluation.

Decoded weights are loaded in their stored [N, K] order, then transposed for
the GEMM. CONFIGS are unbenchmarked candidates; no speedup is implied.
"""
from __future__ import annotations

from typing import NamedTuple
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

FORMAT = "fastwam_additive_vq_microscale_v1"
BLOCK_SIZE = 4
SCALE_BLOCK = 32


class KernelConfig(NamedTuple):
    BM: int
    BN: int
    BK: int
    num_warps: int = 4


# Test these by name with prepared_linear(..., config=name).
CONFIGS = {
    "bm32_bn32_bk32": KernelConfig(32, 32, 32),
    "bm32_bn32_bk64": KernelConfig(32, 32, 64),
    "bm32_bn64_bk32": KernelConfig(32, 64, 32),
    "bm32_bn64_bk64": KernelConfig(32, 64, 64),
    "bm64_bn32_bk32": KernelConfig(64, 32, 32),
    "bm64_bn32_bk64": KernelConfig(64, 32, 64),
    "bm64_bn64_bk32": KernelConfig(64, 64, 32),
    "bm64_bn64_bk64": KernelConfig(64, 64, 64),
    "bm128_bn32_bk32": KernelConfig(128, 32, 32),
    "bm128_bn32_bk64": KernelConfig(128, 32, 64),
    "bm128_bn64_bk32": KernelConfig(128, 64, 32),
    "bm128_bn64_bk64": KernelConfig(128, 64, 64),
}


@triton.jit
def _linear_kernel(
    X, BOOKS, INDICES, ROW_SCALES, EXPONENTS, BIAS, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr,
    BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
    IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
    RS0: tl.constexpr, ES0: tl.constexpr, ES1: tl.constexpr,
    BIAS_S0: tl.constexpr, HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k_lanes = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for kt in range(tl.cdiv(K, BK)):
        k = kt * BK + k_lanes
        a = tl.load(
            X + m[:, None] * XS0 + k[None, :] * XS1,
            (m[:, None] < M) & (k[None, :] < K), other=0,
        )

        # Indices and scales are stored [N, K/...], so K-contiguous loads
        # avoid the strided weight-decode traffic of the original kernel.
        valid_nk = (n[:, None] < N) & (k[None, :] < K)
        code_ptr = INDICES + n[:, None] * IS0 + (k[None, :] // 4) * IS1
        i0 = tl.load(code_ptr, valid_nk, other=0).to(tl.int32)
        i1 = tl.load(code_ptr + IS2, valid_nk, other=0).to(tl.int32)
        coord = k[None, :] % 4
        v0 = tl.load(
            BOOKS + i0 * BS1 + coord * BS2, valid_nk, other=0,
        ).to(tl.float32)
        v1 = tl.load(
            BOOKS + BS0 + i1 * BS1 + coord * BS2, valid_nk, other=0,
        ).to(tl.float32)
        exp = tl.load(
            EXPONENTS + n[:, None] * ES0 + (k[None, :] // 32) * ES1,
            valid_nk, other=127,
        ).to(tl.int32) - 127
        row_scale = tl.load(ROW_SCALES + n * RS0, n < N, other=0).to(tl.float32)
        scale = libdevice.ldexp(row_scale[:, None], exp)
        # Preserve FP32 book-sum/scaling followed by the deployed BF16 rounding.
        weight_nk = tl.where(valid_nk, (v0 + v1) * scale, 0.0).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(weight_nk), acc, out_dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(BIAS + n * BIAS_S0, n < N, other=0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        Y + m[:, None] * N + n[None, :], acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _decode_kernel(
    BOOKS, INDICES, ROW_SCALES, EXPONENTS, OUT,
    N: tl.constexpr, K: tl.constexpr,
    BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
    IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
    RS0: tl.constexpr, ES0: tl.constexpr, ES1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = N * K
    row = flat // K
    col = flat % K
    valid = flat < total
    group = col // 4
    coord = col % 4
    code_ptr = INDICES + row * IS0 + group * IS1
    i0 = tl.load(code_ptr, valid, other=0).to(tl.int32)
    i1 = tl.load(code_ptr + IS2, valid, other=0).to(tl.int32)
    v0 = tl.load(BOOKS + i0 * BS1 + coord * BS2, valid, other=0).to(tl.float32)
    v1 = tl.load(BOOKS + BS0 + i1 * BS1 + coord * BS2, valid, other=0).to(tl.float32)
    exponent = tl.load(
        EXPONENTS + row * ES0 + (col // 32) * ES1, valid, other=127,
    ).to(tl.int32) - 127
    row_scale = tl.load(ROW_SCALES + row * RS0, valid, other=0).to(tl.float32)
    scale = libdevice.ldexp(row_scale, exponent)
    value = ((v0 + v1) * scale).to(tl.bfloat16)
    tl.store(OUT + flat, value, valid)


class PreparedVQ(NamedTuple):
    rows: int
    width: int
    books: torch.Tensor
    indices: torch.Tensor
    row_scales: torch.Tensor
    exponents: torch.Tensor


def prepare(encoding):
    """Validate encoding metadata once, then reuse it with ``prepared_linear``."""
    if (encoding.get("format") != FORMAT or encoding.get("block_size") != BLOCK_SIZE
            or encoding.get("scale_block") != SCALE_BLOCK):
        raise ValueError("Expected fastwam_additive_vq_microscale_v1 encoding")
    shape = encoding.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("encoding['shape'] must be (rows, width)")
    rows, width = map(int, shape)
    if rows <= 0 or width <= 0:
        raise ValueError("VQ weight dimensions must be positive")
    groups = triton.cdiv(width, 4)
    blocks = triton.cdiv(width, 32)
    books, indices = encoding["books"], encoding["indices"]
    row_scales, exponents = encoding["row_scales"], encoding["scale_exponents"]
    expected = (
        (books, (2, 256, 4), torch.bfloat16, "books"),
        (indices, (rows, groups, 2), torch.uint8, "indices"),
        (row_scales, (rows,), torch.bfloat16, "row_scales"),
        (exponents, (rows, blocks), torch.uint8, "scale_exponents"),
    )
    device = books.device
    for tensor, dims, dtype, name in expected:
        if tuple(tensor.shape) != dims or tensor.dtype != dtype or tensor.device != device:
            raise ValueError(f"Invalid {name} shape, dtype, or device")
    return PreparedVQ(rows, width, books, indices, row_scales, exponents)


def _check_input(x, rows, width, books, bias):
    if x.ndim < 1 or x.shape[-1] != width or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("x must be a CUDA BF16 tensor with last dimension equal to VQ width")
    if x.device != books.device:
        raise ValueError("x and encoding tensors must share a device")
    if torch.is_grad_enabled() and (x.requires_grad or (bias is not None and bias.requires_grad)):
        raise RuntimeError("fused_vq_tuned is inference-only")
    if bias is not None and (bias.shape != (rows,) or bias.dtype not in (torch.bfloat16, torch.float32)
                             or bias.device != x.device):
        raise ValueError("bias must be BF16/FP32 with shape [out_features] on x.device")


def _default_config(m, n, k):
    # Deterministic unbenchmarked starting rule; select a measured candidate for deployment.
    bm = 32 if m <= 32 else (64 if m < 128 else 128)
    bn, bk = (32, 64) if max(n, k) >= 8192 else (64, 32)
    return CONFIGS[f"bm{bm}_bn{bn}_bk{bk}"]


SELECTED_CONFIGS = {}

def load_dispatch(path):
    record = json.loads(Path(path).read_text())
    if not record.get("complete"):
        raise ValueError("Incomplete measured dispatch")
    global SELECTED_CONFIGS
    SELECTED_CONFIGS = {key: row["config"] for key,row in record["selected"].items()}
    if not all(name in CONFIGS for name in SELECTED_CONFIGS.values()):
        raise ValueError("Invalid measured dispatch config")

def _resolve_config(config, m, n, k):
    if config is None:
        name=SELECTED_CONFIGS.get(f"{m},{n},{k}")
        return CONFIGS[name] if name else _default_config(m, n, k)
    if isinstance(config, str):
        try:
            return CONFIGS[config]
        except KeyError as exc:
            raise ValueError(f"Unknown VQ GEMM config {config!r}; choose from CONFIGS") from exc
    if isinstance(config, KernelConfig) and config in CONFIGS.values():
        return config
    raise ValueError("config must be None, a CONFIGS name, or a CONFIGS value")


def prepared_linear(x, prepared, bias=None, config=None):
    """Run the fused kernel; pass a CONFIGS name/value to test one tiling."""
    rows, width = prepared.rows, prepared.width
    books, indices = prepared.books, prepared.indices
    row_scales, exponents = prepared.row_scales, prepared.exponents
    _check_input(x, rows, width, books, bias)
    leading = tuple(x.shape[:-1])
    x2 = x.contiguous().view(-1, width)
    m = x2.shape[0]
    if m == 0:
        return torch.empty((*leading, rows), dtype=torch.bfloat16, device=x.device)
    cfg = _resolve_config(config, m, rows, width)
    y = torch.empty((m, rows), dtype=torch.bfloat16, device=x.device)
    bias_arg = bias if bias is not None else x2
    _linear_kernel[(triton.cdiv(rows, cfg.BN), triton.cdiv(m, cfg.BM))](
        x2, books, indices, row_scales, exponents, bias_arg, y,
        m, rows, width, x2.stride(0), x2.stride(1),
        books.stride(0), books.stride(1), books.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        row_scales.stride(0), exponents.stride(0), exponents.stride(1),
        bias.stride(0) if bias is not None else 1, bias is not None,
        cfg.BM, cfg.BN, cfg.BK,
        num_warps=cfg.num_warps, enable_fp_fusion=False,
    )
    return y.view(*leading, rows)


def linear(x, encoding, bias=None, config=None):
    """Compatibility wrapper for ``fused_vq.linear`` with optional tiling."""
    return prepared_linear(x, prepare(encoding), bias, config)


def decode_weight(encoding):
    """Materialize decoded BF16 W for validation only; ``linear`` never uses it."""
    prepared = prepare(encoding)
    rows, width = prepared.rows, prepared.width
    books, indices = prepared.books, prepared.indices
    row_scales, exponents = prepared.row_scales, prepared.exponents
    if not books.is_cuda:
        raise ValueError("decode_weight requires CUDA encoding tensors")
    weight = torch.empty((rows, width), dtype=torch.bfloat16, device=books.device)
    block = 256
    _decode_kernel[(triton.cdiv(rows * width, block),)](
        books, indices, row_scales, exponents, weight,
        rows, width, books.stride(0), books.stride(1), books.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        row_scales.stride(0), exponents.stride(0), exponents.stride(1), block,
        num_warps=4, enable_fp_fusion=False,
    )
    return weight


def _reference_decode(encoding):
    rows, width = encoding["shape"]
    books = encoding["books"].float()
    indices = encoding["indices"].long()
    vectors = books[0][indices[..., 0]] + books[1][indices[..., 1]]
    scales = torch.ldexp(encoding["row_scales"].float()[:, None],
                         encoding["scale_exponents"].to(torch.int32) - 127)
    scales = scales.repeat_interleave(8, dim=1)[:, :vectors.shape[1]]
    return (vectors * scales[..., None]).reshape(rows, -1)[:, :width].to(torch.bfloat16)


def _random_encoding(rows, width, device, generator, edge=False):
    groups, blocks = triton.cdiv(width, 4), triton.cdiv(width, 32)
    books = torch.randn((2, 256, 4), generator=generator).to(device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, 256, (rows, groups, 2), generator=generator).to(device=device, dtype=torch.uint8)
    if edge:
        row_scales = torch.full((rows,), 0.5, dtype=torch.bfloat16)
        exponents = torch.full((rows, blocks), 127, dtype=torch.uint8)
        row_scales[0], exponents[0] = 0, 0
        if rows > 1:
            row_scales[1], exponents[1] = 0.125, 254
    else:
        row_scales = (0.5 + torch.rand((rows,), generator=generator)).to(torch.bfloat16)
        row_scales[0] = 0
        exponents = torch.randint(123, 132, (rows, blocks), generator=generator).to(torch.uint8)
    return {
        "format": FORMAT, "shape": (rows, width), "block_size": 4, "scale_block": 32,
        "books": books, "indices": indices,
        "row_scales": row_scales.to(device), "scale_exponents": exponents.to(device),
    }


def self_test(device="cuda"):
    """Optional PBS-side correctness check; reports linear abs-max and RMSE."""
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(20261023)
    cases = [(32, 129, 294), (129, 129, 294), (294, 129, 294),
             (1, 4096, 8), (1, 7, 7)]
    reports = []
    for case_id, (m, n, k) in enumerate(cases):
        enc = _random_encoding(n, k, device, generator)
        decoded = decode_weight(enc)
        if not torch.equal(decoded.view(torch.int16), _reference_decode(enc).view(torch.int16)):
            raise AssertionError(f"decoded BF16 mismatch for N={n}, K={k}")
        x = torch.randn((m, k), generator=generator).to(device=device, dtype=torch.bfloat16)
        bias = torch.randn((n,), generator=generator).to(device=device, dtype=torch.bfloat16)
        actual = prepared_linear(x, prepare(enc), bias)
        reference = torch.nn.functional.linear(x, decoded, bias)
        diff = actual.float() - reference.float()
        report = {"M": m, "N": n, "K": k,
                  "max_abs": diff.abs().max().item(),
                  "rmse": diff.square().mean().sqrt().item(),
                  "decode_bitwise": True}
        reports.append(report)
        print(f"FUSED_VQ_SELF_TEST {case_id}: {report}", flush=True)

    edge = _random_encoding(3, 37, device, generator, edge=True)
    if not torch.equal(decode_weight(edge).view(torch.int16), _reference_decode(edge).view(torch.int16)):
        raise AssertionError("decoded BF16 mismatch at zero/tail/exponent boundaries")
    print("FUSED_VQ_DECODE_EDGE_TEST_PASSED zero exponent_0 exponent_254 tail", flush=True)
    return reports
