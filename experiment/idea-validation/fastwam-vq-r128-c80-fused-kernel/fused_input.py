"""Triton input transform and signed A4 activation quantization for Fast-WAM."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _smooth_h_kernel(
    x_ptr,
    scale_ptr,
    signs_ptr,
    out_ptr,
    K: tl.constexpr,
    SEG_START: tl.constexpr,
    SEG_WIDTH: tl.constexpr,
    SEG_COUNT: tl.constexpr,
    INV_SQRT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    segment = pid % SEG_COUNT
    row = pid // SEG_COUNT
    start = SEG_START + segment * SEG_WIDTH

    lane = tl.arange(0, BLOCK)
    feature = start + lane
    mask = lane < SEG_WIDTH
    x = tl.load(x_ptr + row * K + feature, mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + feature, mask, other=1.0).to(tl.float32)
    signs = tl.load(signs_ptr + feature, mask, other=1.0).to(tl.float32)
    value = tl.div_rn(x, scale) * signs

    for step in tl.static_range(7):
        h = 1 << step
        if h < SEG_WIDTH:
            partner = tl.gather(value, lane ^ h, axis=0)
            value = tl.where((lane & h) == 0, value + partner, partner - value)

    value = value * INV_SQRT
    tl.store(out_ptr + row * K + feature, value, mask)


@triton.jit
def _a4_kernel(x_ptr, out_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < K
    x = tl.load(x_ptr + row * K + lane, mask, other=0.0).to(tl.float32)
    row_max = tl.max(tl.abs(x), axis=0)
    scale = row_max * (1.0 / 7.0)
    safe_scale = tl.where(scale > 0.0, scale, 1.0)
    scaled = tl.div_rn(x, safe_scale)

    # Floor plus an explicit parity check implements IEEE round-to-nearest-even.
    lower = tl.floor(scaled)
    fraction = scaled - lower
    lower_i = lower.to(tl.int32)
    round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower_i & 1) != 0))
    rounded = lower + round_up.to(tl.float32)
    rounded = libdevice.copysign(rounded, scaled)
    quantized = tl.minimum(tl.maximum(rounded, -7.0), 7.0)
    decoded = quantized * scale
    tl.store(out_ptr + row * K + lane, decoded.to(tl.bfloat16), mask)


@lru_cache(maxsize=None)
def _tail_segments(
    width: int, block: int = 128
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    starts: list[int] = []
    widths: list[int] = []
    start = width // block * block
    tail = width - start
    while tail:
        part = 1 << (tail.bit_length() - 1)
        starts.append(start)
        widths.append(part)
        start += part
        tail -= part
    return tuple(starts), tuple(widths), tuple(1.0 / math.sqrt(part) for part in widths)


def smooth_h(x: torch.Tensor, transform: dict) -> torch.Tensor:
    """Apply FP32 Smooth division, signed normalized H128, and power-of-two tails."""
    if x.ndim < 1:
        raise ValueError("x must have a feature dimension")
    if x.device.type != "cuda":
        raise ValueError("smooth_h requires a CUDA tensor")
    width = x.shape[-1]
    if width < 1 or width > 14336:
        raise ValueError("input width must be in [1, 14336]")
    if width != transform["input_dim"]:
        raise ValueError("input width does not match transform")
    if not transform.get("hadamard", True) or transform.get("block", 128) != 128:
        raise ValueError("smooth_h supports signed H128 transforms only")

    values = x.contiguous()
    scales = transform["scale"]
    signs = transform["signs"]
    if scales.device != x.device or signs.device != x.device:
        raise ValueError("transform scale and signs must be on the input device")
    if scales.dtype != torch.float32:
        raise ValueError("transform scale must be FP32")
    if signs.dtype not in (torch.int8, torch.float32):
        raise ValueError("transform signs must be int8 or FP32")
    scales = scales.reshape(-1).contiguous()
    signs = signs.reshape(-1).contiguous()
    if scales.numel() != width or signs.numel() != width:
        raise ValueError("transform scale and signs must match the input width")
    output = torch.empty(values.shape, dtype=torch.float32, device=x.device)
    rows = values.numel() // width
    if rows == 0:
        return output

    full_blocks = width // 128
    tail_starts, tail_widths, tail_invs = _tail_segments(width)
    if full_blocks:
        _smooth_h_kernel[(rows * full_blocks,)](
            values,
            scales,
            signs,
            output,
            width,
            0,
            128,
            full_blocks,
            1.0 / math.sqrt(128),
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    for start, segment_width, inv_sqrt in zip(tail_starts, tail_widths, tail_invs):
        _smooth_h_kernel[(rows,)](
            values,
            scales,
            signs,
            output,
            width,
            start,
            segment_width,
            1,
            inv_sqrt,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def a4(z: torch.Tensor) -> torch.Tensor:
    """Return whole-row signed A4 decoded activations in BF16."""
    if z.ndim < 1:
        raise ValueError("z must have a feature dimension")
    if z.device.type != "cuda":
        raise ValueError("a4 requires a CUDA tensor")
    width = z.shape[-1]
    if width < 1 or width > 14336:
        raise ValueError("input width must be in [1, 14336]")

    values = z.contiguous()
    output = torch.empty(values.shape, dtype=torch.bfloat16, device=z.device)
    rows = values.numel() // width
    if rows == 0:
        return output
    block = triton.next_power_of_2(width)
    _a4_kernel[(rows,)](
        values,
        output,
        width,
        block,
        num_warps=8 if block >= 8192 else 4,
        enable_fp_fusion=False,
    )
    return output


def self_test() -> dict[str, float | bool]:
    """Compute-only CUDA parity checks; intended to run inside an approved GPU job."""
    if not torch.cuda.is_available():
        raise RuntimeError("self_test requires an available CUDA GPU")
    device = torch.device("cuda")

    def fwht(segment: torch.Tensor) -> torch.Tensor:
        n = segment.shape[-1]
        y = segment.float()
        h = 1
        while h < n:
            groups = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
            left, right = groups.unbind(dim=-2)
            y = torch.stack((left + right, left - right), dim=-2).reshape_as(y)
            h *= 2
        return y * (1.0 / math.sqrt(n))

    def smooth_reference(value: torch.Tensor, transform: dict) -> torch.Tensor:
        scaled = value.float() / transform["scale"]
        pieces = []
        starts, widths, _ = _tail_segments(value.shape[-1])
        full = value.shape[-1] // 128 * 128
        for start in range(0, full, 128):
            part = scaled[..., start : start + 128] * transform["signs"][start : start + 128]
            pieces.append(fwht(part))
        for start, size in zip(starts, widths):
            part = scaled[..., start : start + size] * transform["signs"][start : start + size]
            pieces.append(fwht(part))
        return torch.cat(pieces, dim=-1)

    max_h_error = 0.0
    h_exact = True
    for width in (7, 128, 135, 256):
        x = torch.randn((2, 3, width), device=device, dtype=torch.float32)
        if width == 7:
            x.zero_()
        scale = torch.linspace(0.75, 1.25, width, device=device, dtype=torch.float32)
        signs = torch.where(torch.arange(width, device=device) % 2 == 0, 1.0, -1.0)
        transform = {
            "input_dim": width,
            "hadamard": True,
            "block": 128,
            "scale": scale,
            "signs": signs,
        }
        actual = smooth_h(x, transform)
        expected = smooth_reference(x, transform)
        err = float((actual - expected).abs().max().item())
        max_h_error = max(max_h_error, err)
        h_exact = h_exact and torch.equal(actual, expected)
    if not h_exact:
        raise AssertionError(f"Smooth H128 parity failed; max FP32 error={max_h_error}")

    half_codes = torch.arange(0.5, 7.0, 1.0, device=device, dtype=torch.float32)
    boundary_scale = torch.tensor(10.0, device=device, dtype=torch.float32) / 7
    boundaries = half_codes * boundary_scale
    near_half = torch.cat((
        torch.tensor([10.0], device=device, dtype=torch.float32),
        boundaries,
        torch.nextafter(boundaries, torch.full_like(boundaries, -float("inf"))),
        torch.nextafter(boundaries, torch.full_like(boundaries, float("inf"))),
        -boundaries,
        torch.nextafter(-boundaries, torch.full_like(boundaries, -float("inf"))),
        torch.nextafter(-boundaries, torch.full_like(boundaries, float("inf"))),
    )).reshape(1, -1)
    a4_inputs = [
        torch.randn((2, 3, 257), device=device, dtype=torch.float32) * amplitude
        for amplitude in (1e-10, 1.0, 1e10)
    ]
    a4_inputs.extend((
        torch.zeros((2, 257), device=device, dtype=torch.float32),
        near_half,
        torch.tensor(
            [[14.0, 13.0, 11.0, 9.0, -13.0, -11.0, -9.0, 0.0]],
            device=device,
            dtype=torch.float32,
        ),
    ))
    a4_exact = True
    for z in a4_inputs:
        row_max = z.float().abs().amax(dim=-1, keepdim=True)
        scale = row_max / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        expected = (torch.round(z.float() / safe).clamp(-7, 7) * scale).to(torch.bfloat16)
        actual = a4(z)
        a4_exact = a4_exact and torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if not a4_exact:
        raise AssertionError("A4 BF16 bitwise parity failed")
    return {"smooth_max_abs_error": max_h_error, "smooth_bitwise_equal": h_exact, "a4_bf16_bitwise_equal": a4_exact}
