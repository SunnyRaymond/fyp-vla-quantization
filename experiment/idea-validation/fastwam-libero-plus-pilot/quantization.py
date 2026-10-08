"""Packed W4/A{4,8} Fast-WAM inference and optional packed video KV cache.

W4A4 uses native SM80+ S4 x S4 Tensor Core ``mma.sync`` through Triton inline
PTX. W4A8 uses a separate exact integer dot path. Attention remains BF16.
"""
from __future__ import annotations

import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # dependency is checked when packed inference is enabled
    triton = None
    tl = None


_MISSING = object()
_DEN0 = ("video_expert", "action_expert")


def _canonical_name(name: str) -> str:
    parts = name.split(".") if name else []
    for i in range(len(parts) - 2):
        if parts[i : i + 2] == ["mot", "mixtures"]:
            alias = parts[i + 2]
            if alias in {"video", "action"}:
                parts[i : i + 3] = [f"{alias}_expert"]
                break
    return ".".join(parts)


def _in_scope(name: str) -> bool:
    return name.startswith(("video_expert.", "action_expert.")) or name in {
        "video_expert",
        "action_expert",
        "proprio_encoder",
    } or name.startswith("proprio_encoder.")


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    """Pack the last dimension, low nibble first; pad odd rows with zero."""
    if values.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise TypeError("nibble values must be integer tensors")
    code = values.to(torch.int16).bitwise_and(15).to(torch.uint8)
    if code.shape[-1] % 2:
        code = torch.nn.functional.pad(code, (0, 1))
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def _unpack_nibbles(
    packed: torch.Tensor, length: int, *, signed: bool = True
) -> torch.Tensor:
    low = packed & 15
    high = packed >> 4
    code = torch.stack((low, high), dim=-1).flatten(start_dim=-2)[..., :length]
    if signed:
        code = torch.where(code >= 8, code.to(torch.int16) - 16, code.to(torch.int16))
        return code.to(torch.int8)
    return code


def _quantize_activation(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric per-row RTN. Returns integer codes and FP32 scales."""
    if bits not in (4, 8):
        raise ValueError("activation bits must be 4 or 8")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("Linear input must have a non-empty feature dimension")
    qmax = 7 if bits == 4 else 127
    rows = x.reshape(-1, x.shape[-1]).to(torch.float32)
    maximum = rows.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.zeros_like(maximum))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(rows / safe_scale).clamp(-qmax, qmax).to(torch.int8)
    return codes.contiguous(), scale.squeeze(1).contiguous()


def _affine_quantize_last(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit min/max quantization over the last (group) axis."""
    minimum = x.amin(dim=-1, keepdim=True).to(torch.bfloat16)
    maximum = x.amax(dim=-1, keepdim=True).to(torch.bfloat16)
    span = (maximum.float() - minimum.float()).clamp_min(0)
    scale = torch.where(span > 0, span / 15.0, torch.ones_like(span)).to(torch.bfloat16)
    codes = torch.round((x.float() - minimum.float()) / scale.float()).clamp(0, 15)
    return codes.to(torch.uint8), scale, minimum


class _PackedKV:
    """One immutable packed layer cache; no BF16 residual is retained."""

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        minima: torch.Tensor,
        shape: tuple[int, ...],
        kind: str,
        heads: int,
        head_dim: int,
        group_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.packed = packed
        self.scales = scales
        self.minima = minima
        self.shape = shape
        self.kind = kind
        self.heads = heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.dtype = dtype

    @classmethod
    def encode(
        cls,
        tensor: torch.Tensor,
        *,
        kind: str,
        heads: int,
        head_dim: int,
        group_size: int = 64,
    ) -> "_PackedKV":
        if tensor.ndim != 3 or tensor.shape[-1] != heads * head_dim:
            raise ValueError(f"expected [B,S,{heads * head_dim}] cache, got {tuple(tensor.shape)}")
        batch, seq_len, _ = tensor.shape
        view = tensor.reshape(batch, seq_len, heads, head_dim)
        if kind == "k":
            # [B,H,D,S]: each head/channel is grouped along post-RoPE tokens.
            values = view.permute(0, 2, 3, 1).contiguous()
            group_axis_len = seq_len
        elif kind == "v":
            # [B,S,H,D]: each head/token is grouped along channels.
            values = view
            group_axis_len = head_dim
        else:
            raise ValueError("kind must be 'k' or 'v'")

        code_groups, scale_groups, min_groups = [], [], []
        for start in range(0, group_axis_len, group_size):
            piece = values[..., start : start + group_size].float()
            code, scale, minimum = _affine_quantize_last(piece)
            code_groups.append(code)
            scale_groups.append(scale)
            min_groups.append(minimum)
        codes = torch.cat(code_groups, dim=-1)
        # Keep the trailing group axis even when a cache has only one group.
        scales = torch.cat(scale_groups, dim=-1)
        minima = torch.cat(min_groups, dim=-1)
        return cls(
            _pack_nibbles(codes), scales, minima, tuple(tensor.shape), kind,
            heads, head_dim, group_size, tensor.dtype,
        )

    def clone(self) -> "_PackedKV":
        # Packed cache is immutable; callers' compile=false clone path needs no copy.
        return self

    def decode(self) -> torch.Tensor:
        if self.kind == "k":
            batch, seq_len, _ = self.shape
            values_shape = (batch, self.heads, self.head_dim, seq_len)
            group_axis_len = seq_len
        else:
            batch, seq_len, _ = self.shape
            values_shape = (batch, seq_len, self.heads, self.head_dim)
            group_axis_len = self.head_dim
        codes = _unpack_nibbles(self.packed, group_axis_len, signed=False)
        parts = []
        for group_idx, start in enumerate(range(0, group_axis_len, self.group_size)):
            end = min(start + self.group_size, group_axis_len)
            code = codes[..., start:end].float()
            scale = self.scales[..., group_idx : group_idx + 1].float()
            minimum = self.minima[..., group_idx : group_idx + 1].float()
            parts.append(code * scale + minimum)
        values = torch.cat(parts, dim=-1).reshape(values_shape)
        if self.kind == "k":
            values = values.permute(0, 3, 1, 2).contiguous()
        return values.reshape(self.shape).to(self.dtype)


class _PackedKVList:
    def __init__(self, items: list[_PackedKV]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __iter__(self):
        return iter(self._items)


if triton is not None:

    @triton.jit
    def _w4_int8_dot_kernel(
        A,
        A_SCALE,
        W,
        W_SCALE,
        BIAS,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_K: tl.constexpr,
        A4: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        SM: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m = pid_m * BM + tl.arange(0, BM)
        n = pid_n * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        padded_k = tl.cdiv(K, GROUP_K) * GROUP_K
        w_bytes_per_row = padded_k // 2
        w_groups = (K + GROUP_K - 1) // GROUP_K
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for group in range(w_groups):
            kk = group * GROUP_K + k
            if A4:
                a_byte = tl.load(
                    A + m[:, None] * ((K + 1) // 2) + kk[None, :] // 2,
                    mask=(m[:, None] < M) & (kk[None, :] < K), other=0,
                ).to(tl.int32)
                a_nib = tl.where((kk[None, :] & 1) == 0, a_byte & 15, a_byte >> 4)
                a_q = tl.where(a_nib >= 8, a_nib - 16, a_nib).to(tl.int8)
            else:
                a_q = tl.load(
                    A + m[:, None] * K + kk[None, :],
                    mask=(m[:, None] < M) & (kk[None, :] < K), other=0,
                ).to(tl.int8)

            w_byte = tl.load(
                W + n[:, None] * w_bytes_per_row + kk[None, :] // 2,
                mask=(n[:, None] < N) & (kk[None, :] < K), other=0,
            ).to(tl.int32)
            w_nib = tl.where((kk[None, :] & 1) == 0, w_byte & 15, w_byte >> 4)
            w_q = tl.where(w_nib >= 8, w_nib - 16, w_nib).to(tl.int8)

            if SM >= 75:
                # int32 accumulator forces integer dot semantics.
                dot = tl.dot(
                    a_q,
                    tl.trans(w_q),
                    tl.zeros((BM, BN), dtype=tl.int32),
                )
            else:
                product = a_q.to(tl.int32)[:, None, :] * w_q.to(tl.int32)[None, :, :]
                dot = tl.sum(product, axis=2)
            sa = tl.load(A_SCALE + m, mask=m < M, other=0).to(tl.float32)
            sw = tl.load(W_SCALE + n * w_groups + group, mask=n < N, other=0).to(tl.float32)
            acc += dot.to(tl.float32) * sa[:, None] * sw[None, :]

        if HAS_BIAS:
            bias = tl.load(BIAS + n, mask=n < N, other=0).to(tl.float32)
            acc += bias[None, :]
        tl.store(Y + m[:, None] * N + n[None, :], acc, mask=(m[:, None] < M) & (n[None, :] < N))


    @triton.jit
    def _w4a4_native_mma_kernel(
        A32,
        A_SCALE,
        W32,
        W_SCALE,
        BIAS,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        PAD_K: tl.constexpr,
        GROUP_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # One CTA owns M16 x N32. Its four warps each execute an independent
        # M16 x N8 PTX fragment; the lane/register mapping follows PTX ISA 8.0.
        lane = tl.arange(0, 128)
        warp = lane // 32
        lane32 = lane % 32
        group_id = lane32 // 4
        thread_in_group = lane32 % 4
        row0 = tl.program_id(0) * 16 + group_id
        row1 = row0 + 8
        nbase = tl.program_id(1) * 32 + warp * 8
        b_col = nbase + group_id
        col0 = nbase + thread_in_group * 2
        col1 = col0 + 1
        a_row_stride = PAD_K // 8

        out0 = tl.zeros((128,), dtype=tl.float32)
        out1 = tl.zeros((128,), dtype=tl.float32)
        out2 = tl.zeros((128,), dtype=tl.float32)
        out3 = tl.zeros((128,), dtype=tl.float32)
        groups = tl.cdiv(K, GROUP_K)

        for group in range(groups):
            c0 = tl.full((128,), 0, dtype=tl.int32)
            c1 = tl.full((128,), 0, dtype=tl.int32)
            c2 = tl.full((128,), 0, dtype=tl.int32)
            c3 = tl.full((128,), 0, dtype=tl.int32)
            for half in range(2):
                kbase = group * GROUP_K + half * 64 + thread_in_group * 8
                a0 = tl.load(
                    A32 + row0 * a_row_stride + kbase // 8,
                    mask=row0 < M, other=0,
                )
                a1 = tl.load(
                    A32 + row1 * a_row_stride + kbase // 8,
                    mask=row1 < M, other=0,
                )
                a2 = tl.load(
                    A32 + row0 * a_row_stride + (kbase + 32) // 8,
                    mask=row0 < M, other=0,
                )
                a3 = tl.load(
                    A32 + row1 * a_row_stride + (kbase + 32) // 8,
                    mask=row1 < M, other=0,
                )
                w0 = tl.load(
                    W32 + b_col * a_row_stride + kbase // 8,
                    mask=b_col < N, other=0,
                )
                w1 = tl.load(
                    W32 + b_col * a_row_stride + (kbase + 32) // 8,
                    mask=b_col < N, other=0,
                )
                c0, c1, c2, c3 = tl.inline_asm_elementwise(
                    asm=(
                        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
                        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
                        "{$10, $11, $12, $13};"
                    ),
                    constraints="=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
                    args=[a0, a1, a2, a3, w0, w1, c0, c1, c2, c3],
                    dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
                    is_pure=True,
                    pack=1,
                )

            sa0 = tl.load(A_SCALE + row0, mask=row0 < M, other=0).to(tl.float32)
            sa1 = tl.load(A_SCALE + row1, mask=row1 < M, other=0).to(tl.float32)
            sw0 = tl.load(W_SCALE + col0 * groups + group, mask=col0 < N, other=0).to(tl.float32)
            sw1 = tl.load(W_SCALE + col1 * groups + group, mask=col1 < N, other=0).to(tl.float32)
            out0 += c0.to(tl.float32) * sa0 * sw0
            out1 += c1.to(tl.float32) * sa0 * sw1
            out2 += c2.to(tl.float32) * sa1 * sw0
            out3 += c3.to(tl.float32) * sa1 * sw1

        if HAS_BIAS:
            bias0 = tl.load(BIAS + col0, mask=col0 < N, other=0).to(tl.float32)
            bias1 = tl.load(BIAS + col1, mask=col1 < N, other=0).to(tl.float32)
            out0 += bias0
            out1 += bias1
            out2 += bias0
            out3 += bias1

        tl.store(Y + row0 * N + col0, out0, mask=(row0 < M) & (col0 < N))
        tl.store(Y + row0 * N + col1, out1, mask=(row0 < M) & (col1 < N))
        tl.store(Y + row1 * N + col0, out2, mask=(row1 < M) & (col0 < N))
        tl.store(Y + row1 * N + col1, out3, mask=(row1 < M) & (col1 < N))


def _packed_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    owner = module._quantization_owner
    if not owner._active:
        raise RuntimeError("packed Linear called while Quantization is disabled")
    return owner._linear(module, x)


def _packed_video_action_forward(
    mot,
    action_tokens,
    action_freqs,
    action_t_mod,
    action_context,
    action_context_mask,
    video_cache_k,
    video_cache_v,
    action_attention_mask,
):
    expert = mot.mixtures["action"]
    x = action_tokens
    for layer_idx in range(mot.num_layers):
        block = expert.blocks[layer_idx]
        (
            q_action, k_action, v_action, residual_x, gate_msa, shift_mlp,
            scale_mlp, gate_mlp, _use_gradient_checkpointing,
        ) = mot._build_expert_attention_io(
            expert=expert, block=block, x=x, freqs=action_freqs, t_mod=action_t_mod,
        )
        cache_k = video_cache_k[layer_idx]
        cache_v = video_cache_v[layer_idx]
        if isinstance(cache_k, _PackedKV):
            cache_k = cache_k.decode()
        if isinstance(cache_v, _PackedKV):
            cache_v = cache_v.decode()
        mot._quantization_owner._kv_layer_reads += 1
        mixed = mot._quantization_flash_attention(
            q=q_action,
            k=torch.cat([cache_k, k_action], dim=1),
            v=torch.cat([cache_v, v_action], dim=1),
            num_heads=mot.num_heads,
            ctx_mask=action_attention_mask.to(device=q_action.device),
        )
        x = mot._apply_expert_post_block_tensor(
            block=block,
            residual_x=residual_x,
            mixed_attn_out=mixed,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            context=action_context,
            context_mask=action_context_mask,
        )
    return x


class Quantization:
    """Packed Fast-WAM W4 inference; call BF16 first, then convert once."""

    ARMS = {"bf16", "w4a8", "w4a4", "w4a4kv4", "w4"}

    def __init__(
        self,
        model: nn.Module,
        activation_bits: int = 8,
        kv4: bool = False,
        group_size: int = 128,
    ):
        if activation_bits not in (4, 8):
            raise ValueError("activation_bits must be 4 or 8")
        if isinstance(group_size, bool) or int(group_size) != group_size or int(group_size) != 128:
            raise ValueError("the packed Triton kernel currently requires group_size=128")
        self.model = model
        self.activation_bits = int(activation_bits)
        self.kv4 = bool(kv4)
        self.group_size = int(group_size)
        self.modules: dict[str, nn.Linear] = {}
        by_object: dict[int, tuple[str, nn.Linear]] = {}
        for raw_name, module in model.named_modules(remove_duplicate=False):
            if not raw_name or not isinstance(module, nn.Linear):
                continue
            by_object.setdefault(id(module), (_canonical_name(raw_name), module))
        for name, module in by_object.values():
            if _in_scope(name):
                if name in self.modules and self.modules[name] is not module:
                    raise ValueError(f"canonical module name collision: {name}")
                self.modules[name] = module
        if not self.modules:
            raise ValueError("no Linear found under video_expert, action_expert, or proprio_encoder")

        self._target_parameter_count = sum(
            p.numel() for module in self.modules.values() for p in module.parameters(recurse=False)
        )
        self._target_weight_count = sum(module.weight.numel() for module in self.modules.values())
        self._weight_state = "bf16"
        self._active = False
        self._arm: str | None = None
        self._activation_bits = self.activation_bits
        self._kv_active = False
        self._bf16_calls = 0
        self._bf16_hooks = [module.register_forward_hook(self._count_bf16) for module in self.modules.values()]
        self._mot = getattr(model, "mot", None)
        self._prefill_original = None
        self._prefill_wrapper = None
        self._prefill_override = _MISSING
        self._action_override = _MISSING
        self._flash_override = _MISSING
        self._owner_override = _MISSING
        self._closed = False
        self._integer_gemm_calls = 0
        self._native_int4_gemm_calls = 0
        self._native_int4_tensorcore = False
        self._native_int4_ptx_verified = False
        self._native_int4_instruction = (
            "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"
        )
        self._native_int4_ptx_line: str | None = None
        self._kv_packed_prefills = 0
        self._kv_layer_reads = 0

    def _count_bf16(self, module, args, output):
        if self._weight_state == "bf16" and self._arm == "bf16":
            self._bf16_calls += 1
            for handle in self._bf16_hooks:
                handle.remove()
            self._bf16_hooks.clear()

    @property
    def weights_converted(self) -> bool:
        return self._weight_state == "w4"

    def enable(self, arm: str) -> "Quantization":
        if self._closed:
            raise RuntimeError("Quantization is closed")
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {sorted(self.ARMS)}")
        self.disable()
        if arm == "bf16":
            if self.weights_converted:
                raise RuntimeError("BF16 must run before convert_weights(); FP weights are discarded")
            self._arm, self._active, self._activation_bits = "bf16", False, self.activation_bits
            return self
        if not self.weights_converted:
            raise RuntimeError("run the BF16 reference, then call convert_weights() before W4 arms")
        if arm == "w4a4kv4" and not self.kv4:
            raise RuntimeError("construct Quantization(..., kv4=True) to enable the KV4 arm")
        bits = self.activation_bits if arm == "w4" else (8 if arm == "w4a8" else 4)
        if bits == 4:
            if not torch.cuda.is_available():
                raise RuntimeError("native W4A4 requires an SM80+ CUDA allocation")
            device = next(iter(self.modules.values()))._quant_packed_weight.device
            major, _minor = torch.cuda.get_device_capability(device)
            if major < 8:
                raise RuntimeError(
                    f"native W4A4 requires SM80+ S4 Tensor Cores; refusing fallback on SM{major}x"
                )
            self._native_int4_tensorcore = True
        else:
            self._native_int4_tensorcore = False
        self._arm, self._active, self._activation_bits = arm, True, bits
        self._kv_active = arm == "w4a4kv4"
        if self._kv_active:
            self._install_kv_path()
        return self

    def disable(self) -> None:
        self._active = False
        self._kv_active = False
        self._native_int4_tensorcore = False
        self._restore_kv_path()

    def convert_weights(self) -> None:
        """Replace each FP Linear weight with packed W4 and BF16 group scales once."""
        if self._weight_state != "bf16":
            raise RuntimeError("convert_weights() may run once only; reload source after a failed conversion")
        if self._arm != "bf16" or self._bf16_calls == 0:
            raise RuntimeError("complete at least one BF16 Linear inference before convert_weights()")
        if self._active:
            raise RuntimeError("disable BF16 mode before convert_weights()")
        self._weight_state = "converting"
        try:
            with torch.no_grad():
                for module in self.modules.values():
                    module.to(dtype=torch.bfloat16)
                    weight = module.weight.detach()
                    out_features, in_features = weight.shape
                    groups = math.ceil(in_features / self.group_size)
                    padded_k = math.ceil(in_features / self.group_size) * self.group_size
                    codes = torch.zeros(
                        (out_features, padded_k), device=weight.device, dtype=torch.int8
                    )
                    scales = torch.zeros(
                        (out_features, groups), device=weight.device, dtype=torch.bfloat16
                    )
                    for group_idx, start in enumerate(range(0, in_features, self.group_size)):
                        end = min(start + self.group_size, in_features)
                        part = weight[:, start:end].float()
                        scale = (part.abs().amax(dim=1, keepdim=True) / 7.0).to(torch.bfloat16)
                        safe = torch.where(scale > 0, scale, torch.ones_like(scale)).float()
                        q = torch.round(part / safe).clamp(-7, 7).to(torch.int8)
                        q = torch.where(scale > 0, q, torch.zeros_like(q))
                        codes[:, start:end] = q
                        scales[:, group_idx] = scale[:, 0]
                    packed = _pack_nibbles(codes)
                    module.register_buffer("_quant_packed_weight", packed, persistent=True)
                    module.register_buffer("_quant_weight_scales", scales, persistent=True)
                    module.register_buffer("_quant_weight_shape", torch.tensor(weight.shape, device=weight.device), persistent=False)
                    module._quant_padded_k = padded_k
                    module._quantization_owner = self
                    module.register_parameter("weight", None)
                    module.forward = types.MethodType(_packed_linear_forward, module)
            self._weight_state = "w4"
        except Exception:
            self._weight_state = "failed"
            raise
        finally:
            if self._weight_state == "w4":
                for handle in self._bf16_hooks:
                    handle.remove()
                self._bf16_hooks.clear()

    @torch.no_grad()
    def _linear(self, module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("packed inference requires Triton; install/verify it inside PBS allocation")
        if x.device.type != "cuda":
            raise RuntimeError("packed W4 Linear requires a CUDA allocation")
        major, minor = torch.cuda.get_device_capability(x.device)
        sm = major * 10 + minor
        if sm < 70:
            raise RuntimeError(f"packed integer GEMM requires SM70+, got SM{sm}")
        if self._activation_bits == 4 and sm < 80:
            raise RuntimeError(f"native W4A4 S4 Tensor Core MMA requires SM80+, got SM{sm}")
        features = module.in_features
        if x.shape[-1] != features:
            raise ValueError(f"expected Linear input width {features}, got {x.shape[-1]}")
        shape = tuple(x.shape)
        activation, a_scale = _quantize_activation(x, self._activation_bits)
        rows = activation.shape[0]
        output = torch.empty((rows, module.out_features), device=x.device, dtype=torch.float32)
        bias = module.bias
        bias_ptr = bias if bias is not None else output
        if self._activation_bits == 4:
            padded_k = module._quant_padded_k
            a_codes = torch.nn.functional.pad(
                activation, (0, padded_k - features), value=0
            ).contiguous()
            a32 = _pack_nibbles(a_codes).view(torch.int32)
            w32 = module._quant_packed_weight.view(torch.int32)
            grid = (triton.cdiv(rows, 16), triton.cdiv(module.out_features, 32))
            args = (
                a32,
                a_scale,
                w32,
                module._quant_weight_scales,
                bias_ptr,
                output,
                rows,
                module.out_features,
                features,
                padded_k,
                self.group_size,
                bias is not None,
            )
            if not self._native_int4_ptx_verified:
                compiled = _w4a4_native_mma_kernel.warmup(
                    *args, grid=grid, num_warps=4, num_stages=1
                )
                ptx = compiled.asm.get("ptx", "")
                if self._native_int4_instruction not in ptx:
                    raise RuntimeError(
                        "compiled Triton PTX does not contain the required native S4xS4 MMA"
                    )
                self._native_int4_ptx_line = next(
                    line.strip() for line in ptx.splitlines()
                    if self._native_int4_instruction in line
                )
                self._native_int4_ptx_verified = True
            _w4a4_native_mma_kernel[grid](*args, num_warps=4, num_stages=1)
            self._native_int4_gemm_calls += 1
        else:
            block_m = 64 if rows >= 64 else 32 if rows >= 32 else 16
            block_n = 64 if module.out_features >= 64 else 32
            _w4_int8_dot_kernel[(triton.cdiv(rows, block_m), triton.cdiv(module.out_features, block_n))](
                activation,
                a_scale,
                module._quant_packed_weight,
                module._quant_weight_scales,
                bias_ptr,
                output,
                rows,
                module.out_features,
                features,
                self.group_size,
                False,
                bias is not None,
                sm,
                block_m,
                block_n,
                128,
                num_warps=4,
                num_stages=1,
            )
        self._integer_gemm_calls += 1
        return output.reshape(*shape[:-1], module.out_features).to(torch.bfloat16)

    def _install_kv_path(self) -> None:
        mot = self._mot
        if mot is None or not callable(getattr(mot, "prefill_video_cache_tensor", None)):
            raise RuntimeError("KV4 requires model.mot.prefill_video_cache_tensor")
        if not callable(getattr(mot, "forward_action_with_video_cache_tensor", None)):
            raise RuntimeError("KV4 requires model.mot.forward_action_with_video_cache_tensor")
        method_name = "prefill_video_cache_tensor"
        self._prefill_override = mot.__dict__.get(method_name, _MISSING)
        self._prefill_original = getattr(mot, method_name)

        def prefill_wrapper(*args, **kwargs):
            keys, values = self._prefill_original(*args, **kwargs)
            heads = int(mot.num_heads)
            head_dim = int(mot.attn_head_dim)
            if len(keys) != int(mot.num_layers) or len(values) != int(mot.num_layers):
                raise ValueError("prefill cache layer count does not match MoT")
            packed_k = [
                _PackedKV.encode(k, kind="k", heads=heads, head_dim=head_dim)
                for k in keys
            ]
            packed_v = [
                _PackedKV.encode(v, kind="v", heads=heads, head_dim=head_dim)
                for v in values
            ]
            self._kv_packed_prefills += 1
            return _PackedKVList(packed_k), _PackedKVList(packed_v)

        self._prefill_wrapper = prefill_wrapper
        setattr(mot, method_name, prefill_wrapper)
        self._owner_override = mot.__dict__.get("_quantization_owner", _MISSING)
        setattr(mot, "_quantization_owner", self)
        action_name = "forward_action_with_video_cache_tensor"
        self._action_override = mot.__dict__.get(action_name, _MISSING)
        self._flash_override = mot.__dict__.get("_quantization_flash_attention", _MISSING)
        original = getattr(mot, action_name)
        flash_attention = original.__func__.__globals__.get("flash_attention")
        if not callable(flash_attention):
            if hasattr(mot, "_build_expert_attention_io"):
                self._restore_kv_path()
                raise RuntimeError("could not find the pinned MoT flash_attention implementation")
            flash_attention = lambda **kwargs: None  # toy self-check does not run action attention
        setattr(mot, "_quantization_flash_attention", flash_attention)
        setattr(mot, action_name, types.MethodType(_packed_video_action_forward, mot))

    def _restore_kv_path(self) -> None:
        mot = self._mot
        if mot is None:
            return
        if (
            self._prefill_wrapper is None
            and self._action_override is _MISSING
            and self._flash_override is _MISSING
            and self._owner_override is _MISSING
        ):
            return
        if self._prefill_wrapper is not None:
            if self._prefill_override is _MISSING:
                mot.__dict__.pop("prefill_video_cache_tensor", None)
            else:
                setattr(mot, "prefill_video_cache_tensor", self._prefill_override)
        self._prefill_wrapper = None
        self._prefill_original = None
        action_name = "forward_action_with_video_cache_tensor"
        if self._action_override is not _MISSING:
            setattr(mot, action_name, self._action_override)
        elif "_packed_video_action_forward" in getattr(mot, "__dict__", {}):
            mot.__dict__.pop(action_name, None)
        # MethodType stores the underlying function as __func__.
        current = mot.__dict__.get(action_name)
        if isinstance(current, types.MethodType) and current.__func__ is _packed_video_action_forward:
            if self._action_override is _MISSING:
                mot.__dict__.pop(action_name, None)
            else:
                setattr(mot, action_name, self._action_override)
        if self._flash_override is _MISSING:
            mot.__dict__.pop("_quantization_flash_attention", None)
        elif hasattr(mot, "_quantization_flash_attention"):
            setattr(mot, "_quantization_flash_attention", self._flash_override)
        if getattr(self, "_owner_override", _MISSING) is _MISSING:
            mot.__dict__.pop("_quantization_owner", None)
        else:
            setattr(mot, "_quantization_owner", self._owner_override)
        self._owner_override = _MISSING
        self._action_override = _MISSING
        self._flash_override = _MISSING
        self._prefill_override = _MISSING

    def summary(self) -> dict[str, Any]:
        active_a4 = bool(self._active and self._activation_bits == 4)
        active_a8 = bool(self._active and self._activation_bits == 8)
        return {
            "arm": self._arm,
            "weight_state": self._weight_state,
            "weight_bank_shared_across_w4_arms": True,
            "activation_bits": self._activation_bits if self._arm != "bf16" else None,
            "kv4": bool(self._kv_active),
            "real_quant": self.weights_converted,
            "packed_kernel": self.weights_converted,
            "integer_gemm_calls": self._integer_gemm_calls,
            "kv_packed_prefills": self._kv_packed_prefills,
            "kv_layer_reads": self._kv_layer_reads,
            "packed_weight_tensor_bytes": sum(
                module._quant_packed_weight.numel() * module._quant_packed_weight.element_size()
                for module in self.modules.values()
            ) if self.weights_converted else 0,
            "bf16_target_linear_weights_absent": bool(
                self.weights_converted and all(module.weight is None for module in self.modules.values())
            ),
            "bf16_target_weight_absent": bool(
                self.weights_converted and all(module.weight is None for module in self.modules.values())
            ),
            "bf16_weight_absent": bool(
                self.weights_converted and all(module.weight is None for module in self.modules.values())
            ),
            "weight_group_size": self.group_size,
            "kv_group_size": 64,
            "packed_weight_modules": sorted(self.modules),
            "packed_weight_module_count": len(self.modules) if self.weights_converted else 0,
            "target_linear_parameter_count": self._target_parameter_count,
            "target_linear_weight_count": self._target_weight_count,
            "unquantized_scope": [
                "text_encoder and VAE",
                "sampler/scheduler",
                "all norm layers and non-Linear denoiser parameters",
                "all non-Linear modules in video_expert/action_expert/proprio_encoder",
            ],
            "attention": "BF16",
            "weight_storage": "uint8 packed signed W4 + BF16 G128 scales" if self.weights_converted else "BF16 source weights",
            "activation_storage": "packed_int4" if active_a4 else "int8" if active_a8 else None,
            "dot_arithmetic": (
                "native S4 x S4 mma.sync -> S32, group-scaled FP32 accumulation"
                if active_a4 else
                "signed W4 x signed A8 -> S32 integer dot, group-scaled FP32 accumulation"
                if active_a8 else "inactive"
            ),
            "native_int4_tensorcore": bool(active_a4 and self._native_int4_ptx_verified),
            "native_int4_gemm_calls": self._native_int4_gemm_calls,
            "native_int4_ptx_verified": self._native_int4_ptx_verified,
            "native_int4_instruction": self._native_int4_instruction,
            "native_int4_ptx_line": self._native_int4_ptx_line,
        }

    def _parameter_groups(self, *, exclude: set[int] | None = None):
        exclude = exclude or set()
        groups: dict[str, dict[str, torch.Tensor]] = {}
        seen: set[int] = set()
        for raw_name, parameter in self.model.named_parameters(remove_duplicate=False):
            canonical = _canonical_name(raw_name)
            if not _in_scope(canonical) or id(parameter) in exclude or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            owner, _, leaf = raw_name.rpartition(".")
            groups.setdefault(owner or "<root>", {})[leaf or raw_name] = parameter
        return groups

    @staticmethod
    def _finish_export(root: Path, files: list[Path], manifest: dict[str, Any]) -> dict[str, Any]:
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        file_bytes = sum(path.stat().st_size for path in files) + manifest_path.stat().st_size
        return {
            "path": str(root),
            "manifest": str(manifest_path),
            "file_bytes": file_bytes,
            "storage_gib": file_bytes / (1024 ** 3),
            "files": len(files) + 1,
        }

    def export_packed_weights(self, path: str | Path) -> dict[str, Any]:
        """Write real packed INT4 weights, BF16 scales/biases and remaining core params."""
        if not self.weights_converted:
            return self.export_bf16_weights(path)
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        entries = []
        packed_bytes = scale_bytes = bias_bytes = 0
        for index, (name, module) in enumerate(self.modules.items()):
            packed = module._quant_packed_weight.detach().cpu().contiguous()
            scales = module._quant_weight_scales.detach().cpu().to(torch.bfloat16).contiguous()
            bias = None if module.bias is None else module.bias.detach().cpu().to(torch.bfloat16).contiguous()
            target = root / f"linear_{index:04d}.pt"
            torch.save(
                {
                    "module": name,
                    "shape": [module.out_features, module.in_features],
                    "padded_in_features": module._quant_padded_k,
                    "group_size": self.group_size,
                    "code_range": [-7, 7],
                    "nibble_order": "low nibble first; signed two's complement",
                    "packed_weight": packed,
                    "weight_scales_bf16": scales,
                    "bias_bf16": bias,
                },
                target,
            )
            files.append(target)
            packed_bytes += packed.numel() * packed.element_size()
            scale_bytes += scales.numel() * scales.element_size()
            if bias is not None:
                bias_bytes += bias.numel() * bias.element_size()
            entries.append({
                "module": name,
                "file": target.name,
                "shape": [module.out_features, module.in_features],
                "padded_in_features": module._quant_padded_k,
            })

        selected_parameters = {
            id(parameter)
            for module in self.modules.values()
            for parameter in module.parameters(recurse=False)
        }
        unquant_groups = self._parameter_groups(exclude=selected_parameters)
        unquant_bytes = 0
        for index, (owner, params) in enumerate(unquant_groups.items()):
            cpu = {name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous() for name, tensor in params.items()}
            target = root / f"unquantized_{index:04d}.pt"
            torch.save({"module": owner, "parameters_bf16": cpu}, target)
            files.append(target)
            unquant_bytes += sum(t.numel() * t.element_size() for t in cpu.values())
        stats = self._finish_export(
            root,
            files,
            {
                "format": "fastwam-packed-w4-g128-v1",
                "packed_kernel": True,
                "dot_arithmetic": {
                    "w4a4": "native S4 x S4 mma.sync -> S32",
                    "w4a8": "signed W4 x signed A8 integer dot -> S32",
                },
                "native_int4_instruction": self._native_int4_instruction,
                "native_int4_tensorcore_supported": "SM80+",
                "activation_storage": "packed_int4 or int8, selected by arm",
                "activation_bits": self._activation_bits,
                "kv4": self._arm == "w4a4kv4",
                "linear_modules": entries,
                "unquantized_scope": self.summary()["unquantized_scope"],
                "unquantized_parameter_modules": [
                    {"module": name, "file": f"unquantized_{idx:04d}.pt"}
                    for idx, name in enumerate(unquant_groups)
                ],
            },
        )
        stats.update(
            {
                "packed_tensor_bytes": packed_bytes,
                "scale_tensor_bytes": scale_bytes,
                "bias_bf16_bytes": bias_bytes,
                "unquantized_bf16_parameter_bytes": unquant_bytes,
            }
        )
        return stats

    def export_bf16_weights(self, path: str | Path) -> dict[str, Any]:
        """Stream the same denoiser scope to per-module BF16 files before conversion."""
        if self.weights_converted:
            raise RuntimeError("export the BF16 artifact before convert_weights()")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        entries = []
        tensor_bytes = 0
        for index, (owner, params) in enumerate(self._parameter_groups().items()):
            cpu = {name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous() for name, tensor in params.items()}
            target = root / f"module_{index:04d}.pt"
            torch.save({"module": owner, "parameters_bf16": cpu}, target)
            files.append(target)
            tensor_bytes += sum(t.numel() * t.element_size() for t in cpu.values())
            entries.append({"module": owner, "file": target.name})
        stats = self._finish_export(
            root, files, {"format": "fastwam-denoiser-bf16-v1", "modules": entries}
        )
        stats["bf16_parameter_bytes"] = tensor_bytes
        return stats

    def close(self) -> None:
        self.disable()
        for handle in self._bf16_hooks:
            handle.remove()
        self._bf16_hooks.clear()
        self._closed = True


def self_check() -> None:
    """PBS-only numerical check for native S4 MMA, BF16 identity, and packed KV."""
    if not torch.cuda.is_available():
        raise RuntimeError("self_check must run inside the requested PBS CUDA allocation")
    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError(
            f"native S4 mma.sync self_check requires SM80+; refusing SM{major}{minor} fallback"
        )
    for bits, qmax in ((4, 7), (8, 127)):
        x = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]], device=device)
        codes, scales = _quantize_activation(x, bits)
        assert torch.equal(codes[0], torch.zeros_like(codes[0]))
        assert scales[0].item() == 0
        restored = codes[1].float() * scales[1]
        assert restored.shape == x[1].shape and torch.isfinite(restored).all()
        assert int(codes[1].abs().max().item()) <= qmax

    signed = torch.tensor([[-7, -1, 0, 1, 7]], dtype=torch.int8, device=device)
    assert torch.equal(_unpack_nibbles(_pack_nibbles(signed), 5), signed)
    unsigned = torch.tensor([[0, 1, 7, 8, 15]], dtype=torch.uint8, device=device)
    assert torch.equal(_unpack_nibbles(_pack_nibbles(unsigned), 5, signed=False), unsigned)
    print("SELF_CHECK nibble_roundtrip=pass signed_codes=5 unsigned_codes=5 odd_length=pass")

    fixture = torch.zeros((1, 128, 128), device=device, dtype=torch.bfloat16)
    fixture[0, :, 0] = torch.linspace(0, 63, 128, device=device, dtype=torch.float32).to(torch.bfloat16)
    fixture[0, :, 1] = 100
    fixture[0, 0, :] = torch.linspace(-3, 3, 128, device=device).to(torch.bfloat16)
    k_packed = _PackedKV.encode(fixture, kind="k", heads=1, head_dim=128)
    v_packed = _PackedKV.encode(fixture, kind="v", heads=1, head_dim=128)
    k_decoded, v_decoded = k_packed.decode(), v_packed.decode()
    assert k_decoded.shape == fixture.shape and v_decoded.shape == fixture.shape
    assert not torch.equal(k_decoded, v_decoded), "K and V must use different quantization axes"
    assert k_packed.packed.dtype == torch.uint8 and k_packed.scales.dtype == torch.bfloat16
    assert not hasattr(k_packed, "residual")
    print(
        "SELF_CHECK kv_axes=pass "
        f"K_shape={tuple(k_packed.packed.shape)} V_shape={tuple(v_packed.packed.shape)} "
        f"max_axis_difference={(k_decoded.float() - v_decoded.float()).abs().max().item():.6g}"
    )
    short_k = torch.full((1, 32, 128), 3.0, device=device, dtype=torch.bfloat16)
    one_group = _PackedKV.encode(short_k, kind="k", heads=1, head_dim=128)
    assert one_group.scales.shape[-1] == 1 and torch.equal(one_group.decode(), short_k)
    tail_k = torch.randn((1, 70, 128), device=device, dtype=torch.bfloat16)
    tail_packed = _PackedKV.encode(tail_k, kind="k", heads=1, head_dim=128)
    assert tail_packed.scales.shape[-1] == 2 and tail_packed.decode().shape == tail_k.shape
    assert torch.equal(one_group.decode(), short_k)
    print("SELF_CHECK kv_single_constant_group=pass kv_token_tail_70=pass kv_no_fp_residual=pass")

    class ToyMoT(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads, self.attn_head_dim, self.num_layers = 1, 128, 1
            self.key = tail_k.clone()
            self.value = tail_k.clone()

        def prefill_video_cache_tensor(self, **kwargs):
            return [self.key], [self.value]

        def forward_action_with_video_cache_tensor(self, **kwargs):
            raise AssertionError("toy check only validates the packed prefill boundary")

    model = nn.Module()
    model.video_expert = nn.Module()
    model.video_expert.proj = nn.Linear(128, 128, device=device, dtype=torch.bfloat16)
    model.video_expert.tail = nn.Linear(257, 7, device=device, dtype=torch.bfloat16)
    model.action_expert = nn.Module()
    model.action_expert.proj = nn.Linear(16, 12, device=device, dtype=torch.bfloat16)
    model.proprio_encoder = nn.Linear(8, 5, device=device, dtype=torch.bfloat16)
    model.mot = ToyMoT()
    model.mot.mixtures = nn.ModuleDict({"action": model.action_expert, "video": model.video_expert})
    quant = Quantization(model, activation_bits=8, kv4=True, group_size=128)
    try:
        quant.enable("bf16")
        cases = [
            ("video_expert.proj", model.video_expert.proj, torch.randn((1, 2, 128), device=device, dtype=torch.bfloat16)),
            ("video_expert.tail", model.video_expert.tail, torch.randn((1, 32, 257), device=device, dtype=torch.bfloat16)),
            ("action_expert.proj", model.action_expert.proj, torch.randn((1, 32, 16), device=device, dtype=torch.bfloat16)),
            ("proprio_encoder", model.proprio_encoder, torch.randn((1, 1, 8), device=device, dtype=torch.bfloat16)),
        ]
        for _name, module, probe in cases:
            expected = torch.nn.functional.linear(probe, module.weight, module.bias)
            actual = module(probe)
            assert torch.equal(expected, actual), f"BF16 identity path changed {_name} output"
        assert quant._bf16_calls >= 1 and not quant._bf16_hooks
        print("SELF_CHECK bf16_identity=pass video/action/proprio=pass")
        quant.convert_weights()
        assert len(quant.modules) == 4, f"expected four targeted Linear modules, got {sorted(quant.modules)}"

        def reference(module, probe, bits):
            a_code, a_scale = _quantize_activation(probe, bits)
            qweight = _unpack_nibbles(module._quant_packed_weight, module.in_features).float()
            wscale = module._quant_weight_scales.repeat_interleave(128, dim=1)[:, : module.in_features].float()
            weight = qweight * wscale
            result = (a_code.float() * a_scale[:, None]) @ weight.T
            if module.bias is not None:
                result += module.bias.float()
            return result.reshape(*probe.shape[:-1], module.out_features).to(torch.bfloat16)

        errors = {8: 0.0, 4: 0.0}
        for arm, bits in (("w4a8", 8), ("w4a4", 4)):
            quant.enable(arm)
            for name, module, probe in cases:
                actual = module(probe)
                expected = reference(module, probe, bits)
                assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
                assert torch.isfinite(actual.float()).all(), f"non-finite output for {name}/{arm}"
                error = (actual.float() - expected.float()).abs().max().item()
                errors[bits] = max(errors[bits], error)
                assert torch.allclose(actual.float(), expected.float(), rtol=0.02, atol=0.02), (
                    f"integer GEMM mismatch for {name}/{arm}: max_abs_error={error}"
                )
        print(
            "SELF_CHECK integer_gemm=pass "
            f"W4A8_max_abs_error={errors[8]:.6g} W4A4_max_abs_error={errors[4]:.6g} "
            "fixtures=M1xK8,M2xK128,M32xK16,M32xK257_N7_tail"
        )
        proof = quant.summary()
        assert proof["native_int4_tensorcore"] and proof["native_int4_ptx_verified"]
        assert proof["native_int4_gemm_calls"] > 0
        print(
            "SELF_CHECK native_int4_mma=pass "
            f"ptx={proof['native_int4_ptx_line']} "
            f"compiled_ptx_verified={proof['native_int4_ptx_verified']} "
            f"gemm_calls={proof['native_int4_gemm_calls']}"
        )
        quant.enable("w4a4kv4")
        original_key, original_value = model.mot.key.clone(), model.mot.value.clone()
        packed_k, packed_v = model.mot.prefill_video_cache_tensor()
        assert torch.equal(model.mot.key, original_key) and torch.equal(model.mot.value, original_value)
        assert isinstance(packed_k[0], _PackedKV) and isinstance(packed_v[0], _PackedKV)
        assert packed_k[0].decode().shape == original_key.shape
        print("SELF_CHECK prefill_input_unchanged=pass packed_prefill=pass packed_layers=1")
    finally:
        quant.close()
