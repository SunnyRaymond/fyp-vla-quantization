"""Small, dependency-light numerical primitives for the FRT CCDS runner.

The module deliberately imports NumPy only.  PyTorch is imported lazily by
functions which actually need tensors, so a source/protocol check cannot
accidentally load a model on a login node.  Quantization is fake/emulation
only: hard W4 weights are ordinary floating point tensors after materializing.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


SCHEMA = "frt-ccds-v1"


def _torch():
    import torch

    return torch


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        torch = _torch()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    tmp.replace(path)


def atomic_pickle(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
    tmp.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class W4Spec:
    q_min: int = -7
    q_max: int = 7
    lower_factor: float = 0.25
    upper_factor: float = 4.0
    hard_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (self.q_min, self.q_max) != (-7, 7):
            raise ValueError("FRT pilot is fixed to signed symmetric W4 [-7,7]")
        if not (0 < self.lower_factor < 1 < self.upper_factor):
            raise ValueError("invalid positive scale bounds")


def _channel_view(weight: Any, scale: Any) -> Any:
    torch = _torch()
    if scale.ndim == 1:
        return scale.reshape((scale.shape[0],) + (1,) * (weight.ndim - 1))
    return scale


def per_output_channel_scale(weight: Any, spec: W4Spec = W4Spec()) -> Any:
    """Q0 scale: max(abs(w))/7, with scale=1 and integer zero for zero rows."""
    torch = _torch()
    rows = weight.detach().reshape(weight.shape[0], -1)
    max_abs = rows.abs().amax(dim=1)
    return torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / spec.q_max)


def rtn_integer(weight: Any, scale: Any, spec: W4Spec = W4Spec()) -> Any:
    torch = _torch()
    s = _channel_view(weight, scale)
    # torch.round is ties-to-even, matching the Q0 contract.
    return torch.clamp(torch.round(weight / s), spec.q_min, spec.q_max)


def rtn_weight(weight: Any, spec: W4Spec = W4Spec()) -> tuple[Any, Any, Any]:
    scale = per_output_channel_scale(weight, spec)
    integer = rtn_integer(weight, scale, spec)
    return _channel_view(weight, scale) * integer, scale, integer


def _logit(value: Any, eps: float = 1e-5) -> Any:
    torch = _torch()
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def soft_round_weight(weight: Any, scale: Any, alpha: Any, spec: W4Spec = W4Spec()) -> Any:
    """Differentiable AdaRound-style fake weight used only during CAL fitting."""
    torch = _torch()
    s = _channel_view(weight, scale)
    # The bin index is deliberately detached; scale and alpha remain trainable.
    base = torch.floor((weight.detach() / s).detach())
    h = torch.sigmoid(alpha)
    integer = torch.clamp(base + h, spec.q_min, spec.q_max)
    return s * integer


def hard_round_weight(weight: Any, scale: Any, alpha: Any, spec: W4Spec = W4Spec()) -> Any:
    """Materialize floor+threshold rounding, then clamp and cast the integer grid."""
    torch = _torch()
    s = _channel_view(weight, scale)
    base = torch.floor((weight.detach() / s).detach())
    hard = (torch.sigmoid(alpha).detach() >= spec.hard_threshold).to(weight.dtype)
    integer = torch.clamp(base + hard, spec.q_min, spec.q_max).to(weight.dtype)
    return s * integer


class SoftRoundParametrizationMixin:
    """Marker used for introspection without importing torch at module import."""


def _make_parametrization(weight: Any, spec: W4Spec, lower_factor: float, upper_factor: float) -> Any:
    torch = _torch()
    import torch.nn as nn

    class _SoftRound(nn.Module, SoftRoundParametrizationMixin):
        def __init__(self, original: Any):
            super().__init__()
            q0 = per_output_channel_scale(original, spec).to(original.dtype)
            lower = (q0 * lower_factor).clamp_min(torch.finfo(original.dtype).eps)
            upper = torch.maximum(q0 * upper_factor, lower * 1.0001)
            unit = (q0 - lower) / (upper - lower)
            self.lower = lower.detach()
            self.upper = upper.detach()
            self.scale_raw = nn.Parameter(_logit(unit).detach())
            ratio = original.detach() / _channel_view(original, q0)
            fraction = (ratio - torch.floor(ratio)).clamp(1e-4, 1.0 - 1e-4)
            self.alpha = nn.Parameter(_logit(fraction).detach())
            self.spec = spec
            # Runner anneals this scalar from a smooth start to a sharp final
            # grid; it is intentionally not a trainable parameter.
            self.temperature = 1.0

        def scale_values(self) -> Any:
            return self.lower + (self.upper - self.lower) * torch.sigmoid(self.scale_raw)

        def forward(self, original: Any) -> Any:
            temperature = max(float(self.temperature), 1e-4)
            return soft_round_weight(original, self.scale_values(), self.alpha / temperature, self.spec)

        def rounding_regularizer(self) -> Any:
            torch = _torch()
            temperature = max(float(self.temperature), 1e-4)
            fraction = torch.sigmoid(self.alpha / temperature)
            return (fraction * (1.0 - fraction)).mean()

        def hard_weight(self, original: Any) -> Any:
            return hard_round_weight(original, self.scale_values(), self.alpha, self.spec)

    return _SoftRound(weight)


@dataclass
class SoftHandle:
    module: Any
    path: str
    parametrization: Any


def register_soft_quantizers(
    modules: Sequence[tuple[str, Any]], spec: W4Spec = W4Spec()
) -> list[SoftHandle]:
    """Register one parametrization per target Linear without copying weights."""
    torch = _torch()
    from torch.nn.utils import parametrize

    handles: list[SoftHandle] = []
    for path, module in modules:
        if not hasattr(module, "weight"):
            raise TypeError(f"target {path!r} has no weight")
        parametrization = _make_parametrization(module.weight.detach(), spec, spec.lower_factor, spec.upper_factor)
        parametrize.register_parametrization(module, "weight", parametrization)
        module.parametrizations.weight.original.requires_grad_(False)
        handles.append(SoftHandle(module=module, path=path, parametrization=parametrization))
    return handles


def _remove_one(handle: SoftHandle, hard: bool) -> dict[str, Any]:
    torch = _torch()
    from torch.nn.utils import parametrize

    module = handle.module
    if not hasattr(module, "parametrizations") or not hasattr(module.parametrizations, "weight"):
        raise RuntimeError(f"weight parametrization missing for {handle.path}")
    # Clone before removing the parametrization.  ``original`` aliases the
    # module parameter and would otherwise be overwritten by the copy below.
    original = module.parametrizations.weight.original.detach().clone()
    parametrization = module.parametrizations.weight[0]
    if hard:
        scale_values = parametrization.scale_values().detach()
        learned_integer = torch.clamp(
            torch.floor((original / _channel_view(original, scale_values)).detach())
            + (torch.sigmoid(parametrization.alpha).detach() >= parametrization.spec.hard_threshold).to(original.dtype),
            parametrization.spec.q_min,
            parametrization.spec.q_max,
        ).to(original.dtype)
        materialized = (_channel_view(original, scale_values) * learned_integer).to(original.dtype)
        values = parametrization.scale_values().detach().cpu().reshape(-1).tolist()
    else:
        materialized = original
        values = None
    parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
    with torch.no_grad():
        module.weight.copy_(materialized)
    result = {"path": handle.path, "hard": hard, "scale": values}
    if hard:
        result.update({"q_min": -7, "q_max": 7, "integer_min": int(learned_integer.min().item()), "integer_max": int(learned_integer.max().item())})
    return result


def materialize_hard(handles: Sequence[SoftHandle]) -> list[dict[str, Any]]:
    return [_remove_one(handle, hard=True) for handle in handles]


def discard_soft(handles: Sequence[SoftHandle]) -> None:
    for handle in handles:
        _remove_one(handle, hard=False)


def materialize_rtn(modules: Sequence[tuple[str, Any]], spec: W4Spec = W4Spec()) -> list[dict[str, Any]]:
    """Apply exact Q0 RTN in place and return its scale/grid ledger."""
    torch = _torch()
    rows = []
    with torch.no_grad():
        for path, module in modules:
            value, scale, integer = rtn_weight(module.weight, spec)
            module.weight.copy_(value.to(module.weight.dtype))
            rows.append({
                "path": path,
                "q_min": spec.q_min,
                "q_max": spec.q_max,
                "scale": scale.detach().cpu().reshape(-1).tolist(),
                "integer_min": int(integer.min().item()),
                "integer_max": int(integer.max().item()),
                "zero_channel_count": int((module.weight.detach().reshape(module.weight.shape[0], -1).abs().amax(dim=1) == 0).sum().item()),
            })
    return rows


def snapshot_modules(modules: Sequence[tuple[str, Any]], cpu: bool = True) -> dict[str, Any]:
    return {
        path: module.weight.detach().cpu().clone() if cpu else module.weight.detach().clone()
        for path, module in modules
    }


def restore_modules(modules: Sequence[tuple[str, Any]], snapshot: Mapping[str, Any]) -> None:
    torch = _torch()
    with torch.no_grad():
        for path, module in modules:
            if path not in snapshot:
                raise KeyError(f"snapshot missing {path}")
            module.weight.copy_(snapshot[path].to(device=module.weight.device, dtype=module.weight.dtype))


def action_mask_like(value: Any, *, concat_dim: int | None = None, action_dim: int | None = None) -> Any:
    """Return True at action coordinates in a latent slot."""
    torch = _torch()
    mask = torch.zeros_like(value, dtype=torch.bool)
    if value.ndim < 3:
        raise ValueError(f"latent slot must have at least 3 dimensions, got {tuple(value.shape)}")
    if concat_dim == 0:
        mask[..., -1, :] = True
    elif concat_dim == 1:
        if not action_dim or action_dim <= 0 or action_dim > value.shape[-1]:
            raise ValueError(f"invalid expanded action_dim={action_dim} for latent dim {value.shape[-1]}")
        mask[..., -action_dim:] = True
    else:
        raise ValueError("concat_dim must be explicitly 0 or 1")
    return mask


def mask_action_delta(delta: Any, action_mask: Any | None = None, **layout: Any) -> Any:
    out = delta.clone()
    if action_mask is None:
        action_mask = action_mask_like(out, **layout)
    if tuple(action_mask.shape) != tuple(out.shape):
        action_mask = action_mask.expand_as(out)
    return out.masked_fill(action_mask, 0)


def shift_append(history: Any, new_slot: Any, max_length: int | None = None) -> Any:
    torch = _torch()
    if history.ndim != 4 or new_slot.ndim != 4:
        raise ValueError(f"history/new slot must be [B,T,P,D], got {tuple(history.shape)}/{tuple(new_slot.shape)}")
    if new_slot.shape[0] != history.shape[0] or tuple(new_slot.shape[2:]) != tuple(history.shape[2:]):
        raise ValueError("history and new slot disagree in batch/token/latent dimensions")
    if new_slot.shape[1] != 1:
        raise ValueError("one-step append expects exactly one new slot")
    if max_length is None:
        max_length = int(history.shape[1])
    if max_length < 1:
        raise ValueError("max_length must be positive")
    combined = torch.cat((history, new_slot), dim=1)
    return combined[:, -max_length:, ...]


def insert_slot_delta(history: Any, delta: Any) -> Any:
    if delta.ndim == 3:
        delta = delta.unsqueeze(1)
    if history.ndim != 4 or delta.ndim != 4 or delta.shape[1] != 1:
        raise ValueError("delta must be [B,1,P,D] for a history")
    if tuple(history.shape[0:1] + history.shape[2:]) != tuple(delta.shape[0:1] + delta.shape[2:]):
        raise ValueError("history and delta shape mismatch")
    out = history.clone()
    out[:, -1:, ...] = out[:, -1:, ...] + delta
    return out


class SourceHistoryAdapter:
    """Pure one-step map following VWorldModel.rollout's order exactly."""

    def __init__(self, model: Any, num_hist: int | None = None):
        self.model = model
        self.num_hist = int(num_hist if num_hist is not None else getattr(model, "num_hist"))
        if self.num_hist < 1:
            raise ValueError("model.num_hist must be positive")
        self.concat_dim = int(getattr(model, "concat_dim"))
        self.expanded_action_dim = int(getattr(model, "action_dim", 0))

    def encode(self, obs: Mapping[str, Any], actions: Any) -> Any:
        return self.model.encode(obs, actions)

    def one_step(self, history: Any, action: Any) -> Any:
        torch = _torch()
        if history.ndim == 3:
            history = history.unsqueeze(0)
        if history.ndim != 4:
            raise ValueError(f"history must be [B,T,P,D], got {tuple(history.shape)}")
        if history.shape[1] < 1:
            raise ValueError("history has no temporal slots")
        window = history[:, -self.num_hist:, ...]
        prediction = self.model.predict(window)
        new_slot = prediction[:, -1:, ...].clone()
        if action is not None:
            if action.ndim == 1:
                action = action.reshape(1, 1, -1)
            elif action.ndim == 2:
                action = action.unsqueeze(1)
            if action.ndim != 3 or action.shape[1] != 1:
                raise ValueError(f"one-step action must be [B,1,A], got {tuple(action.shape)}")
            if action.shape[0] != new_slot.shape[0]:
                raise ValueError("action/history batch mismatch")
            new_slot = self.model.replace_actions_from_z(new_slot, action.to(device=new_slot.device, dtype=new_slot.dtype))
        return new_slot

    def append(self, history: Any, new_slot: Any) -> Any:
        return shift_append(history, new_slot, max_length=self.num_hist)

    def with_delta(self, history: Any, delta: Any) -> Any:
        return insert_slot_delta(history, delta)

    def action_mask(self, slot: Any) -> Any:
        return action_mask_like(slot, concat_dim=self.concat_dim, action_dim=self.expanded_action_dim)

    def source_rollout_one_step(self, obs: Mapping[str, Any], actions: Any) -> tuple[Any, Any]:
        """Call the unmodified source rollout and return its first appended slot."""
        initial = self.model.encode(obs, actions[:, : obs["visual"].shape[1], ...])
        source_obs, source_z = self.model.rollout(obs_0=obs, act=actions)
        expected = self.one_step(initial, actions[:, obs["visual"].shape[1] : obs["visual"].shape[1] + 1, ...])
        return expected, source_z[:, obs["visual"].shape[1] : obs["visual"].shape[1] + 1, ...]


def transport(adapter: SourceHistoryAdapter, history: Any, delta: Any, action: Any) -> Any:
    base = adapter.one_step(history, action)
    shifted = adapter.one_step(adapter.with_delta(history, delta), action)
    return shifted - base


def weighted_mse(prediction: Any, target: Any, wz: Any | None = None) -> Any:
    diff = prediction - target
    if wz is not None:
        diff = diff * wz
    return diff.square().mean()


def frt_loss(
    clean_prediction: Any,
    clean_target: Any,
    transport_prediction: Any,
    transport_target: Any,
    wz: Any | None = None,
    lambda_transport: float = 1.0,
) -> dict[str, Any]:
    clean = weighted_mse(clean_prediction, clean_target, wz)
    transport_value = weighted_mse(transport_prediction, transport_target, wz)
    return {"clean": clean, "transport": transport_value, "total": clean + float(lambda_transport) * transport_value}


def estimate_wz(values: Any, floor: float = 1e-3) -> Any:
    torch = _torch()
    if values.ndim < 1:
        raise ValueError("cannot estimate W_z from a scalar")
    flattened = values.detach().reshape(-1, values.shape[-1])
    std = flattened.std(dim=0, unbiased=False)
    return 1.0 / std.clamp_min(float(floor))


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    try:
        torch = _torch()
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def tensor_finite(value: Any) -> bool:
    torch = _torch()
    return bool(torch.isfinite(value).all().item())


def tensor_max_abs(value: Any) -> float:
    return float(value.detach().abs().max().item())


def one_step_hard_reload(
    model: Any,
    modules: Sequence[tuple[str, Any]],
    snapshot: Mapping[str, Any],
    adapter: SourceHistoryAdapter,
    bank: Mapping[str, Any],
    output: Path,
    lr: float = 1e-2,
) -> dict[str, Any]:
    """Exercise learned soft->hard materialization and serialized reload once."""
    torch = _torch()
    device = next(model.parameters()).device
    restore_modules(modules, snapshot)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handles = register_soft_quantizers(modules, W4Spec())
    optimizer = torch.optim.Adam([p for h in handles for p in (h.parametrization.scale_raw, h.parametrization.alpha)], lr=lr)
    data = {key: value.to(device) for key, value in bank.items() if isinstance(value, torch.Tensor)}
    for handle in handles:
        handle.parametrization.temperature = 2.0
    with torch.enable_grad():
        clean = adapter.one_step(data["history"], data["action"])
        tr = transport(adapter, data["x_history"], data["delta"], data["next_action"])
        losses = frt_loss(clean, data["fp_current"], tr, data["fp_transport"], data["wz"], 1.0)
        reg = torch.stack([h.parametrization.rounding_regularizer() for h in handles]).mean()
        total = losses["total"] + 0.01 * reg
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
    ledger = materialize_hard(handles)
    with torch.no_grad():
        hard_output = adapter.one_step(data["history"], data["action"]).detach()
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    path = Path(output) / "hard_fit_reload.pt"
    atomic_torch_save(path, {"schema": "frt-hard-w4-checkpoint-v1", "method": "FRT-one-step", "state_dict": state_dict, "hard_ledger": ledger})
    atomic_json(Path(output) / "hard_fit_ledger.json", {"schema": "frt-hard-w4-ledger-v1", "rows": ledger})
    restore_modules(modules, snapshot)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"], strict=True)
    with torch.no_grad():
        reloaded = adapter.one_step(data["history"], data["action"]).detach()
    error = float((hard_output - reloaded).abs().max().item())
    same = bool(torch.allclose(hard_output, reloaded, rtol=0.0, atol=0.0))
    restore_modules(modules, snapshot)
    return {"passed": same, "max_abs_error": error, "checkpoint": str(path.resolve()), "ledger_rows": len(ledger), "ledger_sha256": stable_hash(ledger), "soft_clean_loss": float(losses["clean"].detach().item()), "soft_transport_loss": float(losses["transport"].detach().item()), "hard_clean_loss": float(weighted_mse(hard_output, data["fp_current"], data["wz"]).detach().item())}
