"""Scope-specific activation bits and bounded local diagnostics for Fast-WAM."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


_SCOPES = ("video", "action", "proprio")
_METRIC_KEYS = (
    "input_absmax", "input_rms",
    "a4_scale_min", "a4_scale_mean", "a4_scale_max", "a4_zero_code_fraction",
    "a4_input_rmse", "a4_input_relative_rmse",
    "a8_scale_min", "a8_scale_mean", "a8_scale_max", "a8_zero_code_fraction",
    "a8_input_rmse", "a8_input_relative_rmse",
    "native_vs_a4_reference_rmse", "native_vs_a4_reference_max_abs",
    "native_vs_a4_reference_normalized_rmse",
    "a4_vs_a8_reference_rmse", "a4_vs_a8_reference_max_abs",
    "a4_vs_a8_reference_normalized_rmse",
)


def scope_for_name(name: str) -> str:
    """Return the Fast-WAM activation scope for a canonical or aliased module name."""
    parts = name.split(".")
    if "video_expert" in parts or any(
        parts[i : i + 3] == ["mot", "mixtures", "video"] for i in range(len(parts) - 2)
    ):
        return "video"
    if "action_expert" in parts or any(
        parts[i : i + 3] == ["mot", "mixtures", "action"] for i in range(len(parts) - 2)
    ):
        return "action"
    if "proprio_encoder" in parts:
        return "proprio"
    raise ValueError(f"module is outside the video/action/proprio scopes: {name!r}")


class StageQuantization:
    """Wrap an existing packed Quantization object without retaining FP weights."""

    def __init__(self, q: Any):
        if int(q.group_size) != 128:
            raise ValueError("StageQuantization requires the pilot's fixed G128 weights")
        self.q = q
        self.bits: dict[str, int] | None = None
        self._native_linear = q._linear
        self._name_by_id = {id(module): name for name, module in q.modules.items()}
        self._scope_by_id = {
            module_id: scope_for_name(name) for module_id, name in self._name_by_id.items()
        }
        self._mode = "native"
        self._stage: str | None = None
        self._step: int | None = None
        self._trace_records: list[dict[str, Any]] | None = None
        self._trace_values: list[Any] | None = None
        self._trace_path: Path | None = None
        self._trace_context: dict[str, Any] = {}
        self.trace_result: dict[str, Any] | None = None
        self._call_index = 0

    def configure(self, bits: tuple[int, int, int]) -> "StageQuantization":
        if self._mode != "native" or self._trace_records is not None:
            raise RuntimeError("cannot reconfigure while a reference or trace context is active")
        if len(bits) != 3 or any(isinstance(b, bool) or b not in (4, 8) for b in bits):
            raise ValueError("bits must be (video, action, proprio), each 4 or 8")
        self.bits = dict(zip(_SCOPES, (int(b) for b in bits)))
        self.q.enable("w4a4" if 4 in bits else "w4a8")
        # enable() installs KV4 only for the explicit w4a4kv4 arm; keep this hook disabled.
        if hasattr(self.q, "_kv_active"):
            self.q._kv_active = False
        self.q._linear = self._dispatch
        return self

    def set_stage(self, stage: str, step: int) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("step must be an integer scheduler index")
        self._stage, self._step = stage, step

    def _scope(self, module: Any) -> str:
        try:
            return self._scope_by_id[id(module)]
        except KeyError as exc:
            raise ValueError("q._linear received a module outside q.modules") from exc

    def _dispatch(self, module: Any, x: Any) -> Any:
        if self.bits is None:
            raise RuntimeError("call configure() before inference")
        scope = self._scope(module)
        bits = self.bits[scope]
        previous_bits = self.q._activation_bits
        self.q._activation_bits = bits
        try:
            if self._mode == "reference":
                return _reference_linear(self.q, module, x, bits)
            native = self._native_linear(module, x)
            if self._mode == "trace":
                self._record_trace(module, x, native, scope)
            return native
        finally:
            self.q._activation_bits = previous_bits

    @contextmanager
    def reference_mode(self) -> Iterator["StageQuantization"]:
        self._require_configured()
        if self._mode != "native" or self._trace_records is not None:
            raise RuntimeError("reference_mode cannot be nested with another diagnostic context")
        self._mode = "reference"
        try:
            yield self
        finally:
            self._mode = "native"

    @contextmanager
    def trace_mode(
        self, out_dir: str | Path, context_meta: Mapping[str, Any]
    ) -> Iterator["StageQuantization"]:
        self._require_configured()
        if self._mode != "native" or self._trace_records is not None:
            raise RuntimeError("trace_mode cannot be nested with another diagnostic context")
        if tuple(self.bits[s] for s in _SCOPES) != (4, 4, 4):
            raise ValueError("trace_mode is reserved for the native all-A4 query")
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        index = 1
        path = root / "stage_quantization_trace.json"
        while path.exists():
            path = root / f"stage_quantization_trace_{index:03d}.json"
            index += 1
        self._trace_path = path
        self._trace_context = dict(context_meta)
        self._trace_records, self._trace_values = [], []
        self.trace_result = None
        self._mode = "trace"
        try:
            yield self
        except BaseException:
            try:
                self._finalize_trace(completed=False)
            finally:
                self._mode = "native"
            raise
        else:
            try:
                self._finalize_trace(completed=True)
            finally:
                self._mode = "native"

    def finish_trace(self) -> dict[str, Any] | None:
        """Return the finalized trace; trace_mode flushes it automatically on exit."""
        if self._trace_records is not None:
            raise RuntimeError("finish_trace() is called automatically when trace_mode exits")
        return self.trace_result

    def _record_trace(self, module: Any, x: Any, native: Any, scope: str) -> None:
        if self._trace_records is None or self._trace_values is None:
            raise RuntimeError("trace state was not initialized")
        import torch

        shape = tuple(x.shape)
        features = int(module.in_features)
        rows = x.reshape(-1, features).float()
        reference4, scale4, codes4 = _activation_reconstruction(torch, rows, 4)
        reference8, scale8, codes8 = _activation_reconstruction(torch, rows, 8)
        out4 = _reference_from_activation(self.q, module, reference4, shape)
        out8 = _reference_from_activation(self.q, module, reference8, shape)
        native32, out4_32, out8_32 = native.float(), out4.float(), out8.float()
        input_rms = rows.square().mean().sqrt()
        eps = torch.finfo(torch.float32).eps
        input_metrics = {
            "input_absmax": rows.abs().amax(),
            "input_rms": input_rms,
        }
        for bits, scale, codes, reconstructed in (
            (4, scale4, codes4, reference4), (8, scale8, codes8, reference8)
        ):
            delta = reconstructed - rows
            rmse = delta.square().mean().sqrt()
            input_metrics.update({
                f"a{bits}_scale_min": scale.amin(),
                f"a{bits}_scale_mean": scale.mean(),
                f"a{bits}_scale_max": scale.amax(),
                f"a{bits}_zero_code_fraction": (codes == 0).float().mean(),
                f"a{bits}_input_rmse": rmse,
                f"a{bits}_input_relative_rmse": rmse / input_rms.clamp_min(eps),
            })
        for prefix, left, right in (
            ("native_vs_a4_reference", native32, out4_32),
            ("a4_vs_a8_reference", out4_32, out8_32),
        ):
            delta = left - right
            rmse = delta.square().mean().sqrt()
            input_metrics.update({
                f"{prefix}_rmse": rmse,
                f"{prefix}_max_abs": delta.abs().amax(),
                f"{prefix}_normalized_rmse": rmse / right.square().mean().sqrt().clamp_min(eps),
            })
        self._trace_records.append({
            "call_index": self._call_index,
            "module": self._name_by_id[id(module)],
            "scope": scope,
            "stage": self._stage,
            "step": self._step,
            "input_shape": list(shape),
            "output_shape": list(native.shape),
            "input_features": features,
            "output_features": int(module.out_features),
        })
        self._trace_values.append(torch.stack([input_metrics[key] for key in _METRIC_KEYS]).detach())
        self._call_index += 1

    def _finalize_trace(self, completed: bool) -> None:
        if self._trace_records is None or self._trace_values is None or self._trace_path is None:
            return
        if self._trace_values:
            values = __import__("torch").stack(self._trace_values).detach().cpu().tolist()
        else:
            values = []
        records = []
        coverage: dict[tuple[Any, ...], int] = {}
        for meta, row in zip(self._trace_records, values):
            metrics = {
                key: (float(value) if value == value and abs(value) != float("inf") else None)
                for key, value in zip(_METRIC_KEYS, row)
            }
            records.append({**meta, "metrics": metrics})
            key = (
                meta["module"], meta["scope"], meta["stage"], meta["step"],
                tuple(meta["input_shape"]), tuple(meta["output_shape"]),
            )
            coverage[key] = coverage.get(key, 0) + 1
        report = {
            "diagnostic": "fastwam-stage-quantization-trace-v1",
            "completed": completed,
            "context": self._trace_context,
            "calls": len(records),
            "records": records,
            "coverage": [
                {
                    "module": key[0], "scope": key[1], "stage": key[2], "step": key[3],
                    "input_shape": key[4], "output_shape": key[5], "calls": count,
                }
                for key, count in coverage.items()
            ],
            "reference": (
                "independent signed-nibble W4 decode with BF16 G128 scales; FP32 "
                "per-row absmax RTN (A4/A8), TF32 disabled, FP32 matmul and bias, BF16 output"
            ),
            "retention": "scalar summaries per call only; no activation or full-weight tensors retained",
            "timing_claim": "none; trace includes independent FP32 local references",
            "trace_path": str(self._trace_path),
        }
        with self._trace_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        self.trace_result = report
        self._trace_records = self._trace_values = None

    def _require_configured(self) -> None:
        if self.bits is None:
            raise RuntimeError("call configure() before inference")


def _activation_reconstruction(torch: Any, rows: Any, bits: int) -> tuple[Any, Any, Any]:
    qmax = 7 if bits == 4 else 127
    maximum = rows.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.zeros_like(maximum))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(rows / safe_scale).clamp(-qmax, qmax)
    return codes * scale, scale, codes


@contextmanager
def _tf32_disabled(torch: Any) -> Iterator[None]:
    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if cuda_matmul is None or not hasattr(cuda_matmul, "allow_tf32"):
        yield
        return
    previous = cuda_matmul.allow_tf32
    cuda_matmul.allow_tf32 = False
    try:
        yield
    finally:
        cuda_matmul.allow_tf32 = previous


def _reference_from_activation(q: Any, module: Any, activation: Any, shape: tuple[int, ...]) -> Any:
    import torch

    features = int(module.in_features)
    output_features = int(module.out_features)
    packed_weight = module._quant_packed_weight
    scales = module._quant_weight_scales
    # Chunk output rows so a diagnostic never materializes the full model's FP32 weight bank.
    chunk_rows = 128
    output = torch.empty(
        (activation.shape[0], output_features), device=activation.device, dtype=torch.float32
    )
    with _tf32_disabled(torch):
        for start in range(0, output_features, chunk_rows):
            end = min(start + chunk_rows, output_features)
            packed = packed_weight[start:end].to(torch.int32)
            low, high = packed & 15, (packed >> 4) & 15
            nibble = torch.stack((low, high), dim=-1).flatten(start_dim=-2)
            signed = torch.where(nibble >= 8, nibble - 16, nibble).float()[:, :features]
            weight_scales = scales[start:end].float().repeat_interleave(q.group_size, dim=1)[:, :features]
            weight = signed * weight_scales
            output[:, start:end] = activation @ weight.T
    if module.bias is not None:
        output = output + module.bias.float()
    return output.reshape(*shape[:-1], output_features).to(torch.bfloat16)


def _reference_linear(q: Any, module: Any, x: Any, bits: int) -> Any:
    import torch

    shape = tuple(x.shape)
    rows = x.reshape(-1, int(module.in_features)).float()
    activation, _scale, _codes = _activation_reconstruction(torch, rows, bits)
    return _reference_from_activation(q, module, activation, shape)
