"""Inference-only VQ tile decode into one reusable BF16 Linear scratch."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


_SHARED = Path(__file__).resolve().parent.parent / "fastwam-vq-r128-c80-fused-tuning"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

MODES = ("layer_decode_fp32", "layer_decode_bf16_u", "layer_decode_bf16_all")
_DECODE_BLOCK = 256


@triton.jit
def _smooth_h_bf16_kernel(
    x_ptr, scale_ptr, signs_ptr, out_ptr,
    K: tl.constexpr, SEG_START: tl.constexpr, SEG_WIDTH: tl.constexpr,
    SEG_COUNT: tl.constexpr, INV_SQRT: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    segment = pid % SEG_COUNT
    row = pid // SEG_COUNT
    start = SEG_START + segment * SEG_WIDTH
    lane = tl.arange(0, BLOCK)
    feature = start + lane
    mask = lane < SEG_WIDTH
    x = tl.load(x_ptr + row * K + feature, mask, other=0.0).to(tl.bfloat16)
    scale = tl.load(scale_ptr + feature, mask, other=1.0).to(tl.bfloat16)
    signs = tl.load(signs_ptr + feature, mask, other=1.0).to(tl.bfloat16)
    ratio = tl.div_rn(x.to(tl.float32), scale.to(tl.float32)).to(tl.bfloat16)
    value = (ratio * signs).to(tl.bfloat16)

    for step in tl.static_range(7):
        h = 1 << step
        if h < SEG_WIDTH:
            partner = tl.gather(value, lane ^ h, axis=0)
            plus = (value + partner).to(tl.bfloat16)
            minus = (partner - value).to(tl.bfloat16)
            value = tl.where((lane & h) == 0, plus, minus)

    inv_sqrt = tl.full((), INV_SQRT, tl.float32).to(tl.bfloat16)
    value = (value * inv_sqrt).to(tl.bfloat16)
    tl.store(out_ptr + row * K + feature, value, mask)


@triton.jit
def _a4_bf16_kernel(x_ptr, out_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < K
    x = tl.load(x_ptr + row * K + lane, mask, other=0.0).to(tl.bfloat16)
    row_max = tl.max(tl.abs(x.to(tl.float32)), axis=0).to(tl.bfloat16)
    scale = (row_max.to(tl.float32) / 7.0).to(tl.bfloat16)
    safe_scale = tl.where(scale > 0.0, scale, 1.0).to(tl.bfloat16)
    scaled = (x.to(tl.float32) / safe_scale.to(tl.float32)).to(tl.bfloat16)
    scaled_f32 = scaled.to(tl.float32)

    lower = tl.floor(scaled_f32)
    fraction = scaled_f32 - lower
    lower_i = lower.to(tl.int32)
    round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower_i & 1) != 0))
    rounded = lower + round_up.to(tl.float32)
    rounded = libdevice.copysign(rounded, scaled_f32)
    quantized = tl.minimum(tl.maximum(rounded, -7.0), 7.0).to(tl.bfloat16)
    decoded = (quantized * scale).to(tl.bfloat16)
    tl.store(out_ptr + row * K + lane, decoded, mask)


def decode_prepared(prepared, out):
    """Decode a prepared VQ matrix into caller-owned flat BF16 storage.

    The caller provides a CUDA BF16 output buffer with at least rows*width values.
    The returned [rows, width] view aliases that buffer; no dense weight is kept.
    """
    from fused_vq import _decode_kernel

    rows, width = prepared.rows, prepared.width
    books, indices = prepared.books, prepared.indices
    row_scales, exponents = prepared.row_scales, prepared.exponents
    view = out[: rows * width].view(rows, width)
    _decode_kernel[(triton.cdiv(rows * width, _DECODE_BLOCK),)](
        books, indices, row_scales, exponents, view,
        rows, width, books.stride(0), books.stride(1), books.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        row_scales.stride(0), exponents.stride(0), exponents.stride(1), _DECODE_BLOCK,
        num_warps=4, enable_fp_fusion=False,
    )
    return view


def _smooth_h_bf16(x, transform):
    import torch
    import fused_input

    width = x.shape[-1]
    values = x.contiguous()
    scales, signs = transform["scale"], transform["signs"]
    output = torch.empty(values.shape, dtype=torch.bfloat16, device=x.device)
    rows = values.numel() // width
    if rows == 0:
        return output
    full_blocks = width // 128
    starts, widths, inverses = fused_input._tail_segments(width)
    if full_blocks:
        _smooth_h_bf16_kernel[(rows * full_blocks,)](
            values, scales, signs, output, width, 0, 128, full_blocks,
            1.0 / (128 ** 0.5), 128, num_warps=4, enable_fp_fusion=False,
        )
    for start, segment_width, inv_sqrt in zip(starts, widths, inverses):
        _smooth_h_bf16_kernel[(rows,)](
            values, scales, signs, output, width, start, segment_width, 1,
            inv_sqrt, 128, num_warps=4, enable_fp_fusion=False,
        )
    return output


def _smooth_h_fp32(x, transform):
    """Call the shared FP32 kernel without repeating its install-time checks."""
    import torch
    import fused_input

    width = x.shape[-1]
    values = x.contiguous()
    output = torch.empty(values.shape, dtype=torch.float32, device=x.device)
    rows = values.numel() // width
    if rows == 0:
        return output
    full_blocks = width // 128
    starts, widths, inverses = fused_input._tail_segments(width)
    if full_blocks:
        fused_input._smooth_h_kernel[(rows * full_blocks,)](
            values, transform["scale"], transform["signs"], output, width,
            0, 128, full_blocks, 1.0 / (128 ** 0.5), 128,
            num_warps=4, enable_fp_fusion=False,
        )
    for start, segment_width, inv_sqrt in zip(starts, widths, inverses):
        fused_input._smooth_h_kernel[(rows,)](
            values, transform["scale"], transform["signs"], output, width,
            start, segment_width, 1, inv_sqrt, 128,
            num_warps=4, enable_fp_fusion=False,
        )
    return output


def _a4_fp32(value):
    import torch
    import fused_input

    width = value.shape[-1]
    values = value.contiguous()
    output = torch.empty(values.shape, dtype=torch.bfloat16, device=value.device)
    rows = values.numel() // width
    if rows:
        block = triton.next_power_of_2(width)
        fused_input._a4_kernel[(rows,)](
            values, output, width, block,
            num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False,
        )
    return output


def _a4_bf16(value):
    import torch

    width = value.shape[-1]
    values = value.contiguous()
    output = torch.empty(values.shape, dtype=torch.bfloat16, device=value.device)
    rows = values.numel() // width
    if rows:
        block = triton.next_power_of_2(width)
        _a4_bf16_kernel[(rows,)](
            values, output, width, block,
            num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False,
        )
    return output


def make_runtime(out, p, bundle, direct=True):
    """Build the existing frozen inference runtime, optionally using the meta loader."""
    import kernel_runtime

    return kernel_runtime.make_runtime(out, p, bundle, direct=direct)


class LayerDecodeBanks:
    def __init__(self, runtime, base, mode):
        import torch

        self.runtime, self.mode = runtime, mode
        self.modules = base.modules
        self.packed, self.fused = base.packed, base.fused
        self.seen, self.calls = base.seen, base.calls
        self.encodings, self.prepared = base.encodings, base.prepared
        self.transforms, self.rotations = base.transforms, base.rotations
        self.reference_transforms = {}
        self.diagnostic_dir = base.diagnostic_dir
        self.validate_input = mode == "layer_decode_fp32"
        self.validation = {name: {"status": "pending"} for name in self.modules}
        self.decoded_calls = Counter()
        self.scratch_elements = max(p.rows * p.width for p in self.prepared.values())
        device = next(iter(self.prepared.values())).books.device
        for stream in ("action", "video"):
            matrix = self.rotations.matrix(stream)
            if matrix.device != device or matrix.dtype != torch.float32 or tuple(matrix.shape) != (128, 128):
                raise ValueError(f"Invalid FP32 fixed rotation matrix for {stream}")
        for name, module in self.modules.items():
            width = module.in_features
            transform = self.transforms[name]
            scale, signs = transform["scale"], transform["signs"]
            if (width < 1 or width > 14336 or transform["input_dim"] != width
                    or not transform["hadamard"] or transform["block"] != 128
                    or scale.device != device or scale.dtype != torch.float32
                    or signs.device != device or signs.dtype not in (torch.int8, torch.float32)
                    or scale.numel() != width or signs.numel() != width):
                raise ValueError(f"Invalid Smooth/H transform at {name}")
            transform["scale"] = scale.reshape(-1).contiguous()
            transform["signs"] = signs.reshape(-1).contiguous()
            self.reference_transforms[name] = dict(transform)
            if mode == "layer_decode_bf16_all":
                transform["scale"] = transform["scale"].to(dtype=torch.bfloat16)
            bias = module.bias
            if (bias is not None and (tuple(bias.shape) != (module.out_features,)
                    or bias.device != device or bias.dtype not in (torch.bfloat16, torch.float32))):
                raise ValueError(f"Invalid Linear bias at {name}")
        self.scratch = torch.empty(self.scratch_elements, dtype=torch.bfloat16, device=device)
        self.scratch_bytes = self.scratch_elements * self.scratch.element_size()
        self.bf16_rotations = ({name: matrix.to(dtype=torch.bfloat16)
                                for name, matrix in self.rotations.matrices.items()}
                               if mode != "layer_decode_fp32" else {})

    def parameters(self):
        return []

    def coverage_summary(self):
        return dict(
            module_count=len(self.modules), seen_count=len(self.seen),
            call_count=sum(self.calls.values()), decoded_call_count=sum(self.decoded_calls.values()),
            encoded_module_count=len(self.encodings), scratch_bytes=self.scratch_bytes,
            validation_entries=len(self.validation), mode=self.mode,
        )

    def decode(self, name):
        self.decoded_calls[name] += 1
        return decode_prepared(self.prepared[name], self.scratch)

    def _rotate_bf16(self, value, stream):
        import torch

        full = value.shape[-1] // 128 * 128
        if not full:
            return value
        keep_bf16 = self.mode == "layer_decode_bf16_all"
        head = value[..., :full].to(torch.bfloat16)
        head = head.reshape(*value.shape[:-1], full // 128, 128)
        head = head @ self.bf16_rotations[stream]
        if not keep_bf16:
            head = head.to(torch.float32)
        head = head.reshape(*value.shape[:-1], full)
        if full < value.shape[-1]:
            tail = value[..., full:] if keep_bf16 else value[..., full:].float()
            return torch.cat((head, tail), -1)
        return head

    def _operand(self, x, name):
        import torch
        import batched_transform
        import export_runtime
        from rotation_bank import stream_for

        transform = self.transforms[name]
        if self.mode == "layer_decode_bf16_all":
            h = _smooth_h_bf16(x, transform)
        else:
            h = _smooth_h_fp32(x, transform)
        stream = stream_for(name)
        if self.mode == "layer_decode_fp32":
            z = self.rotations.apply_after_h(h, stream)
            operand = _a4_fp32(z)
        else:
            z = self._rotate_bf16(h, stream)
            operand = _a4_bf16(z) if self.mode == "layer_decode_bf16_all" else _a4_fp32(z)

        if self.validation[name]["status"] == "pending":
            if self.validate_input:
                expected_h = batched_transform.transform_input(x, transform)
                h_error = float((h - expected_h).abs().max().item())
                if h_error != 0:
                    self.validation[name] = {"status": "failed", "smooth_h_max_abs": h_error}
                    raise ValueError(f"Fused Smooth/H differs at {name}: {h_error}")
                expected_operand = export_runtime.a4(z)
                a4_bitwise = torch.equal(operand.view(torch.int16), expected_operand.view(torch.int16))
                if not a4_bitwise:
                    try:
                        from kernel_runtime import _save_a4_mismatch_diagnostic
                        _save_a4_mismatch_diagnostic(self.diagnostic_dir, name, z, operand, expected_operand)
                    except Exception:
                        pass
                    self.validation[name] = {"status": "failed", "smooth_h_max_abs": h_error,
                                             "a4_bitwise": False}
                    raise ValueError("Fused whole-row A4 differs at " + name)
                self.validation[name] = {"status": "passed", "smooth_h_max_abs": h_error,
                                         "a4_bitwise": True}
            else:
                expected_h = batched_transform.transform_input(x, self.reference_transforms[name])
                expected_z = self.rotations.apply_after_h(expected_h, stream)
                expected_operand = export_runtime.a4(expected_z)
                self.validation[name] = {
                    "status": "recorded",
                    "smooth_h_max_abs": float((h - expected_h).abs().max().item()),
                    "rotation_max_abs": float((z - expected_z).abs().max().item()),
                    "a4_max_abs": float((operand.float() - expected_operand.float()).abs().max().item()),
                    "a4_bitwise": torch.equal(operand.view(torch.int16), expected_operand.view(torch.int16)),
                }
        return operand

    def forward(self, name, module):
        import torch
        import torch.nn.functional as F

        @torch.no_grad()
        def run(x):
            self.seen.add(name)
            self.calls[name] += 1
            self.runtime.linear_calls += 1
            operand = self._operand(x, name)
            # shortcut: one scratch is safe on this runtime's single CUDA stream; concurrent streams need one buffer per stream.
            weight = self.decode(name)
            return F.linear(operand, weight, module.bias)
        return run


def install(runtime, bundle, mode="layer_decode_fp32"):
    """Install packed/fused/direct tensors and replace fused GEMM with layer decode + cuBLAS."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    import export_runtime
    import kernel_runtime

    export_runtime.compute_guard()
    base = kernel_runtime.install(runtime, bundle, packed=True, fused=True, direct=True)
    banks = LayerDecodeBanks(runtime, base, mode)
    for name, module in banks.modules.items():
        module.forward = banks.forward(name, module)
    runtime.banks, runtime.seen, runtime.forwards = banks, banks.seen, {}
    runtime.current_arm = runtime.arm_name = "reader_kl"
    runtime.install_metadata.update(
        mode=mode, packed_gpu_resident=True, dense_target_weights=False,
        native_vq_kernel=False, layer_decode=True, decoded_weight_scratch_bytes=banks.scratch_bytes,
        native_bf16_linear_gemm=True,
        u_rotation_uses_native_bf16_gemm=mode != "layer_decode_fp32",
        u_rotation_arithmetic="fp32_gemm" if mode == "layer_decode_fp32" else "bf16_gemm",
        u_rotation_output_dtype="fp32" if mode != "layer_decode_bf16_all" else "bf16",
        smooth_h_scale_dtype="bf16" if mode == "layer_decode_bf16_all" else "fp32",
        smooth_h_arithmetic=("bf16_butterflies_with_fp32_division" if mode == "layer_decode_bf16_all"
                             else "fp32"),
        a4_arithmetic=("bf16_values_with_fp32_reduction_division_rounding"
                       if mode == "layer_decode_bf16_all" else "fp32"),
        all_transform_math_native_bf16=False,
    )
    return banks
