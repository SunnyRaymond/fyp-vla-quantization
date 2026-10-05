"""Minimal full-path fake W4A{4,8} instrumentation for FastWAM denoiser Linears.

This is a fake-quant mechanism wrapper, not a packed-weight implementation or
a latency/kernel claim. Construct it before making teacher references, leave
its hooks disabled for identity use, make all BF16 references, then call
``weight_quantize()`` once before any W4/W4A{4,8} arm.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


_MODES = {"bf16", "w4", "rtn", "phase", "independent"}
_ACTIVATION_MODES = {"rtn", "phase", "independent"}


def _canonical_name(name: str) -> str:
    """Map FastWAM's MoT expert aliases to their direct expert paths."""
    parts = name.split(".") if name else []
    for i in range(len(parts) - 2):
        if parts[i : i + 2] == ["mot", "mixtures"]:
            alias = parts[i + 2]
            if alias in {"video", "action"}:
                parts[i : i + 3] = [f"{alias}_expert"]
                break
    return ".".join(parts)


def _in_scope(name: str) -> bool:
    if name.startswith(("video_expert.", "action_expert.")):
        return True
    if name == "proprio_encoder" or name.startswith("proprio_encoder."):
        return True
    return False


class Quantizer:
    """Attach disabled identity hooks to canonical denoiser ``nn.Linear``s."""

    def __init__(
        self, model: nn.Module, group_size: int = 128, activation_bits: int = 8
    ):
        if isinstance(group_size, bool) or int(group_size) != group_size or group_size < 1:
            raise ValueError("group_size must be a positive integer")
        if (
            isinstance(activation_bits, bool)
            or not isinstance(activation_bits, int)
            or activation_bits not in (4, 8)
        ):
            raise ValueError("activation_bits must be 4 or 8")
        if not callable(getattr(model, "named_modules", None)):
            raise TypeError("model must provide named_modules()")

        self.group_size = int(group_size)
        self.activation_bits = activation_bits
        self.rtn_full_range = False
        self.model = model
        self.modules: dict[str, nn.Linear] = {}
        by_object: dict[int, tuple[str, nn.Linear]] = {}
        for raw_name, module in model.named_modules(remove_duplicate=False):
            if not raw_name or not isinstance(module, nn.Linear):
                continue
            name = _canonical_name(raw_name)
            by_object.setdefault(id(module), (name, module))

        for name, module in by_object.values():
            if _in_scope(name):
                if name in self.modules and self.modules[name] is not module:
                    raise ValueError(f"canonical module name collision: {name}")
                self.modules[name] = module
        if not self.modules:
            raise ValueError(
                "no denoiser nn.Linear modules found under video_expert, "
                "action_expert, or proprio_encoder"
            )
        self.selected_modules = self.modules

        self._handles = []
        for name, module in self.modules.items():
            self._handles.append(
                module.register_forward_pre_hook(self._make_pre_hook(name))
            )

        self._active = False
        self.mode: str | None = None
        self._weight_state = "float"
        self._phases: dict[str, float] = {}
        self._clips: dict[str, float] = {}
        self._rng: torch.Generator | None = None
        self._base_uniform: float | None = None
        self.executed_sites: list[str] = []
        self.module_call_counts: dict[str, int] = {}
        self.site_stats: dict[str, dict[str, Any]] = {}
        self.quantized_calls = 0
        self._forward_counts: dict[str, dict[str, torch.Tensor]] = {}
        self._weight_counts: dict[str, dict[str, torch.Tensor]] = {}

    @property
    def weights_quantized(self) -> bool:
        return self._weight_state == "w4"

    def _accumulate(
        self,
        bucket: dict[str, dict[str, torch.Tensor]],
        name: str,
        amount: torch.Tensor,
    ) -> None:
        by_device = bucket.setdefault(name, {})
        device = str(amount.device)
        value = amount.detach().to(dtype=torch.int64)
        if device in by_device:
            by_device[device] = by_device[device] + value
        else:
            by_device[device] = value

    @staticmethod
    def _draw_uniform(generator: torch.Generator) -> float:
        # A private CPU generator leaves the model/scheduler RNG untouched.
        return float(torch.rand((), generator=generator, device="cpu").item())

    def _make_pre_hook(self, canonical_name: str):
        def pre_hook(module: nn.Module, args: tuple[Any, ...]):
            if not self._active or not args or not isinstance(args[0], torch.Tensor):
                return None

            x = args[0]
            call_index = self.module_call_counts.get(canonical_name, 0)
            self.module_call_counts[canonical_name] = call_index + 1
            site = f"{canonical_name}#{call_index}"
            self.executed_sites.append(site)
            self.site_stats[site] = {
                "module": canonical_name,
                "call_index": call_index,
                "dtype": str(x.dtype),
                "shape": list(x.shape),
            }

            if self.mode not in _ACTIVATION_MODES:
                return None
            quantized = self._quantize_activation(x, site)
            return (quantized, *args[1:])

        return pre_hook

    def _quantize_activation(self, x: torch.Tensor, site: str) -> torch.Tensor:
        if x.ndim == 0 or x.shape[-1] == 0:
            raise ValueError(f"Linear activation at {site} has no feature dimension")

        x32 = x.to(dtype=torch.float32)
        finite = torch.isfinite(x32)
        clean = torch.where(finite, x32, torch.zeros_like(x32))
        features = x32.shape[-1]
        rows = clean.reshape(-1, features)
        finite_rows = finite.reshape(-1, features)
        row_max = rows.abs().amax(dim=1, keepdim=True)
        nonzero_rows = row_max > 0
        denominator, code_min, code_max = self._activation_grid()
        clip = self._clips.get(site, 1.0)
        # Zero rows use scale=1 and no dither; their true delta is never used.
        delta = torch.where(nonzero_rows, (row_max / denominator) * clip, 1.0)

        phase = self._phases.get(site, 0.0)
        if self.mode == "phase":
            assert self._base_uniform is not None
            dither = (self._base_uniform + phase) % 1.0
        elif self.mode == "independent":
            assert self._rng is not None
            dither = (self._draw_uniform(self._rng) + phase) % 1.0
        else:
            dither = 0.0

        # Suppress dither for zero rows; the final branch returns exact zeros.
        row_dither = torch.where(nonzero_rows, dither, 0.0)
        raw_q = torch.round(rows / delta + row_dither)
        overload = finite_rows & nonzero_rows & ((raw_q < code_min) | (raw_q > code_max))
        self._accumulate(
            self._forward_counts, "activation_overload", overload.sum(dtype=torch.int64)
        )
        self._accumulate(
            self._forward_counts,
            "activation_nonfinite",
            (~finite).sum(dtype=torch.int64),
        )
        self._accumulate(
            self._forward_counts,
            "zero_rows",
            (~nonzero_rows).sum(dtype=torch.int64),
        )
        self.quantized_calls += 1

        q = raw_q.clamp(min=code_min, max=code_max)
        dequantized = (q - row_dither) * torch.where(nonzero_rows, delta, 0.0)
        result = torch.where(nonzero_rows, dequantized, torch.zeros_like(rows))
        result = result.reshape_as(x32)
        result = torch.where(finite, result, x32)
        return result.to(dtype=x.dtype)

    def _activation_grid(self) -> tuple[int, int, int]:
        if self.mode == "rtn" and self.rtn_full_range:
            qmax = 2 ** (self.activation_bits - 1) - 1
            return qmax, -qmax, qmax
        denominator = 2 ** (self.activation_bits - 1) - 2
        return denominator, -denominator, denominator + 1

    def begin(
        self,
        mode: str = "bf16",
        seed: int = 0,
        phases: dict[str, float] | None = None,
        clips: dict[str, float] | None = None,
    ) -> None:
        """Start one infer trace; reset module call indices and forward counts."""
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        if mode == "bf16" and self._weight_state != "float":
            raise RuntimeError("BF16 teacher references must be completed before weight_quantize()")
        if mode != "bf16" and not self.weights_quantized:
            raise RuntimeError("call weight_quantize() before a W4 activation-quantized mode")

        checked_phases: dict[str, float] = {}
        for key, value in (phases or {}).items():
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"phase for {key!r} must be finite and in [0, 1)")
            checked_phases[str(key)] = value
        checked_clips: dict[str, float] = {}
        if clips and mode != "rtn":
            raise ValueError("clips are reserved for the direct-RTN control")
        for key, value in (clips or {}).items():
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"clip multiplier for {key!r} must be finite and > 0")
            checked_clips[str(key)] = value

        self._active = False
        self.mode = mode
        self._phases = checked_phases
        self._clips = checked_clips
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)
        self._base_uniform = (
            self._draw_uniform(self._rng) if mode == "phase" else None
        )
        self.executed_sites = []
        self.module_call_counts = {}
        self.site_stats = {}
        self.quantized_calls = 0
        self._forward_counts = {}
        self._active = True

    def disable(self) -> None:
        """Return all hooks to disabled identity behavior, preserving the trace."""
        self._active = False

    def _materialize_counts(
        self, bucket: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, int]:
        return {
            name: sum(int(value.item()) for value in by_device.values())
            for name, by_device in bucket.items()
        }

    def end(self) -> dict[str, Any]:
        """Stop tracing and return a JSON-safe summary; scalar syncs happen here."""
        self._active = False
        forward = self._materialize_counts(self._forward_counts)
        weight = self._materialize_counts(self._weight_counts)
        activation_nonfinite = forward.get("activation_nonfinite", 0)
        activation_overload = forward.get("activation_overload", 0)
        weight_nonfinite = weight.get("weight_nonfinite", 0)
        weight_overload = weight.get("weight_overload", 0)
        zero_rows = forward.get("zero_rows", 0)
        if self.mode in _ACTIVATION_MODES:
            activation_denominator, activation_code_min, activation_code_max = (
                self._activation_grid()
            )
        else:
            activation_denominator = activation_code_min = activation_code_max = None
        return {
            "mode": self.mode,
            "weights_quantized": self.weights_quantized,
            "group_size": self.group_size,
            "activation_bits": self.activation_bits,
            "activation_denominator": activation_denominator,
            "activation_code_min": activation_code_min,
            "activation_code_max": activation_code_max,
            "rtn_full_range": bool(self.rtn_full_range),
            "executed_sites": list(self.executed_sites),
            "module_call_counts": dict(self.module_call_counts),
            "site_stats": {key: dict(value) for key, value in self.site_stats.items()},
            "nonfinite_values": activation_nonfinite + weight_nonfinite,
            "overload_codes": activation_overload + weight_overload,
            "zero_rows": zero_rows,
            "quantized_calls": self.quantized_calls,
            "site_count": len(self.executed_sites),
            "activation_nonfinite_count": activation_nonfinite,
            "activation_overload_count": activation_overload,
            "weight_nonfinite_count": weight_nonfinite,
            "weight_overload_count": weight_overload,
            "nonfinite_count": activation_nonfinite + weight_nonfinite,
            "overload_count": activation_overload + weight_overload,
        }

    def weight_quantize(self) -> None:
        """In-place symmetric grouped W4 fake quantization; retains no BF16 bank."""
        if self._weight_state != "float":
            raise RuntimeError(
                "weight_quantize() may run once only; reload the original model after a partial failure"
            )
        self._weight_state = "in_progress"
        try:
            with torch.no_grad():
                for module in self.modules.values():
                    weight = module.weight
                    w = weight.detach().to(dtype=torch.float32)
                    finite = torch.isfinite(w)
                    self._accumulate(
                        self._weight_counts,
                        "weight_nonfinite",
                        (~finite).sum(dtype=torch.int64),
                    )
                    for start in range(0, w.shape[1], self.group_size):
                        end = min(start + self.group_size, w.shape[1])
                        group = w[:, start:end]
                        group_finite = finite[:, start:end]
                        clean = torch.where(group_finite, group, torch.zeros_like(group))
                        scale = clean.abs().amax(dim=1, keepdim=True) / 7.0
                        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                        raw_q = torch.round(clean / safe_scale)
                        overload = group_finite & ((raw_q < -7) | (raw_q > 7))
                        self._accumulate(
                            self._weight_counts,
                            "weight_overload",
                            overload.sum(dtype=torch.int64),
                        )
                        dequantized = raw_q.clamp(min=-7, max=7) * scale
                        replacement = torch.where(group_finite, dequantized, group)
                        weight[:, start:end].copy_(replacement.to(dtype=weight.dtype))
        except Exception:
            # A partially quantized bank cannot be safely retried without a source reload.
            raise
        else:
            self._weight_state = "w4"

    def unit_test(self) -> None:
        """Small runnable CPU checks for activation grids and legacy A8 behavior."""
        def make_model() -> nn.Module:
            model = nn.Module()
            model.video_expert = nn.Module()
            model.video_expert.proj = nn.Linear(3, 2, bias=True)
            with torch.no_grad():
                model.video_expert.proj.weight.copy_(
                    torch.tensor([[1.0, -1.0, 0.5], [0.25, 0.5, -1.0]])
                )
                model.video_expert.proj.bias.copy_(torch.tensor([0.25, -0.5]))
            return model

        for bits in (4, 8):
            model = make_model()
            quantizer = Quantizer(model, group_size=2, activation_bits=bits)
            try:
                quantizer.weight_quantize()
                site = "unit"
                headroom = 2 ** (bits - 1) - 2
                values = torch.tensor([[-2.0, 2.0, 0.0], [0.0, 0.0, 0.0]])

                quantizer.begin(mode="rtn")
                output = quantizer._quantize_activation(values, site)
                stats = quantizer.end()
                assert torch.equal(output[1], torch.zeros_like(output[1]))
                assert stats["activation_overload_count"] == 0
                assert stats["zero_rows"] == 1
                assert stats["activation_denominator"] == headroom
                assert stats["activation_code_min"] == -headroom
                assert stats["activation_code_max"] == headroom + 1

                # An RTN-only clip deliberately drives both asymmetric endpoints.
                quantizer.begin(mode="rtn", clips={site: 0.5})
                endpoints = quantizer._quantize_activation(
                    torch.tensor([[-2.0, 2.0]]), site
                )
                stats = quantizer.end()
                scale = (2.0 / headroom) * 0.5
                endpoint_codes = torch.round(endpoints / scale)
                assert torch.equal(
                    endpoint_codes,
                    torch.tensor([[-headroom, headroom + 1]], dtype=torch.float32),
                )
                assert stats["activation_overload_count"] == 2

                # The standard RTN control uses the symmetric signed range.
                quantizer.rtn_full_range = True
                full_qmax = 2 ** (bits - 1) - 1
                quantizer.begin(mode="rtn", clips={site: 0.5})
                endpoints = quantizer._quantize_activation(
                    torch.tensor([[-2.0, 2.0]]), site
                )
                stats = quantizer.end()
                scale = (2.0 / full_qmax) * 0.5
                endpoint_codes = torch.round(endpoints / scale)
                assert torch.equal(
                    endpoint_codes,
                    torch.tensor([[-full_qmax, full_qmax]], dtype=torch.float32),
                )
                assert stats["activation_denominator"] == full_qmax
                assert stats["activation_code_min"] == -full_qmax
                assert stats["activation_code_max"] == full_qmax
                assert stats["rtn_full_range"] is True

                # rtn_full_range never changes the shared phase/independent grid.
                for mode in ("phase", "independent"):
                    quantizer.begin(mode=mode, seed=7, phases={site: 0.25})
                    quantizer._quantize_activation(values, site)
                    stats = quantizer.end()
                    assert stats["activation_denominator"] == headroom
                    assert stats["activation_code_min"] == -headroom
                    assert stats["activation_code_max"] == headroom + 1
                if bits == 8:
                    x = torch.tensor(
                        [[0.2, -0.4, 0.1], [0.0, 0.0, 0.0]], dtype=torch.float32
                    )
                    quantizer.begin(mode="phase", seed=7, phases={site: 0.25})
                    actual = quantizer._quantize_activation(x, site)
                    dither = (quantizer._base_uniform + 0.25) % 1.0
                    rows = x.reshape(-1, x.shape[-1])
                    row_max = rows.abs().amax(dim=1, keepdim=True)
                    nonzero = row_max > 0
                    delta = torch.where(nonzero, row_max / 126.0, 1.0)
                    row_dither = torch.where(nonzero, dither, 0.0)
                    old_q = torch.round(rows / delta + row_dither).clamp(
                        min=-126, max=127
                    )
                    old = (old_q - row_dither) * torch.where(
                        nonzero, delta, 0.0
                    )
                    old = torch.where(nonzero, old, torch.zeros_like(rows)).reshape_as(x)
                    assert torch.equal(actual, old)
                    stats = quantizer.end()
                    assert stats["activation_denominator"] == 126
                    assert stats["activation_code_min"] == -126
                    assert stats["activation_code_max"] == 127

                    # Keep the original no-argument Screen.q.unit_test contract.
                    linear_site = "video_expert.proj#0"
                    quantizer.begin(
                        mode="phase", seed=7, phases={linear_site: 0.25}
                    )
                    output = model.video_expert.proj(
                        torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.4, 0.1]])
                    )
                    stats = quantizer.end()
                    assert torch.equal(output[0], model.video_expert.proj.bias)
                    assert bool((output[1] != model.video_expert.proj.bias).any().item())
                    assert stats["executed_sites"] == [linear_site]
                    assert stats["module_call_counts"] == {"video_expert.proj": 1}
            finally:
                quantizer.close()

        model = make_model()
        for invalid_bits in (2, 6, True, 4.0):
            try:
                Quantizer(model, activation_bits=invalid_bits)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid activation_bits={invalid_bits!r}")

    def close(self) -> None:
        self.disable()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
