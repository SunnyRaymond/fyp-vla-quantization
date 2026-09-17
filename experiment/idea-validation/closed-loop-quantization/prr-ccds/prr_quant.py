"""PRR's one quantizer for quantizer-only and LoRA-r4 arms.

The module keeps the deployment representation explicit: every target Linear
is finally represented by an int8 tensor whose values are restricted to the
signed W4 grid [-7, 7], plus one float32 scale per output channel.  During CAL
the same floor STE is used in both families; LoRA only changes ``W_eff``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _torch():
    import torch
    return torch


@dataclass(frozen=True)
class W4Config:
    q_min: int = -7
    q_max: int = 7
    lower_factor: float = 0.25
    upper_factor: float = 4.0
    hard_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (self.q_min, self.q_max) != (-7, 7):
            raise ValueError("PRR is fixed to signed W4 [-7,7]")
        if not 0 < self.lower_factor < 1 < self.upper_factor:
            raise ValueError("invalid scale bounds")


def _view(weight: Any, scale: Any) -> Any:
    return scale.reshape((scale.shape[0],) + (1,) * (weight.ndim - 1)) if scale.ndim == 1 else scale


def q0_scale(weight: Any, cfg: W4Config = W4Config()) -> Any:
    torch = _torch()
    rows = weight.detach().reshape(weight.shape[0], -1)
    maximum = rows.abs().amax(dim=1)
    return torch.where(maximum == 0, torch.ones_like(maximum), maximum / cfg.q_max)


def q0_integer(weight: Any, scale: Any, cfg: W4Config = W4Config()) -> Any:
    torch = _torch()
    return torch.clamp(torch.round(weight / _view(weight, scale)), cfg.q_min, cfg.q_max).to(torch.int8)


def q0_weight(weight: Any, cfg: W4Config = W4Config()) -> tuple[Any, Any, Any]:
    scale = q0_scale(weight, cfg)
    integer = q0_integer(weight, scale, cfg)
    return _view(weight, scale) * integer.to(weight.dtype), scale, integer


def _logit(value: Any, eps: float = 1e-5) -> Any:
    torch = _torch()
    value = value.clamp(eps, 1 - eps)
    return torch.log(value) - torch.log1p(-value)


def _floor_ste(x: Any) -> Any:
    # Forward is exactly floor(x); backward is the identity surrogate.
    floor = _torch().floor(x)
    return floor.detach() + (x - x.detach())


class QuantParam(_torch().nn.Module):
    """Parametrization shared by all four PRR arms."""

    def __init__(self, original: Any, use_lora: bool, rank: int, cfg: W4Config):
        torch = _torch()
        import torch.nn as nn
        super().__init__()
        qscale = q0_scale(original, cfg).to(original.dtype)
        lower = (qscale * cfg.lower_factor).clamp_min(torch.finfo(original.dtype).eps)
        upper = torch.maximum(qscale * cfg.upper_factor, lower * 1.0001)
        self.lower, self.upper, self.cfg = lower.detach(), upper.detach(), cfg
        self.scale_raw = nn.Parameter(_logit(((qscale - lower) / (upper - lower)).detach()))
        ratio = original.detach() / _view(original, qscale)
        fraction = (ratio - torch.floor(ratio)).clamp(1e-4, 1 - 1e-4)
        self.alpha = nn.Parameter(_logit(fraction).detach())
        self.use_lora = bool(use_lora)
        self.rank = int(rank if use_lora else 0)
        if self.use_lora:
            if self.rank < 1:
                raise ValueError("LoRA rank must be positive")
            self.lora_A = nn.Parameter(torch.empty(self.rank, original.shape[1], dtype=original.dtype, device=original.device))
            self.lora_B = nn.Parameter(torch.zeros(original.shape[0], self.rank, dtype=original.dtype, device=original.device))
            nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        self.temperature = 1.0

    def scale_values(self) -> Any:
        torch = _torch()
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.scale_raw)

    def effective(self, original: Any) -> Any:
        if not self.use_lora:
            return original
        return original + self.lora_B @ self.lora_A

    def forward(self, original: Any) -> Any:
        torch = _torch()
        temperature = max(float(self.temperature), 1e-4)
        effective = self.effective(original)
        scale = self.scale_values()
        x = effective / _view(effective, scale)
        integer = torch.clamp(_floor_ste(x) + torch.sigmoid(self.alpha / temperature), self.cfg.q_min, self.cfg.q_max)
        return _view(effective, scale) * integer

    def rounding_regularizer(self) -> Any:
        torch = _torch()
        temperature = max(float(self.temperature), 1e-4)
        h = torch.sigmoid(self.alpha / temperature)
        return (h * (1 - h)).mean()

    def hard_parts(self, original: Any) -> tuple[Any, Any]:
        torch = _torch()
        effective = self.effective(original).detach()
        scale = self.scale_values().detach()
        integer = torch.clamp(torch.floor(effective / _view(effective, scale)) + (torch.sigmoid(self.alpha).detach() >= self.cfg.hard_threshold), self.cfg.q_min, self.cfg.q_max).to(torch.int8)
        return integer, scale.to(torch.float32)


@dataclass
class Handle:
    path: str
    module: Any
    parametrization: QuantParam


def attach(modules: Sequence[tuple[str, Any]], *, use_lora: bool, seed: int, rank: int = 4, cfg: W4Config = W4Config()) -> list[Handle]:
    """Register one identical quantizer per target Linear.

    ``seed`` controls LoRA A initialization.  B starts at zero; quantizer
    scale/alpha initialization is deterministic from the current FP weight.
    """
    torch = _torch()
    from torch.nn.utils import parametrize
    torch.manual_seed(int(seed))
    handles: list[Handle] = []
    for path, module in modules:
        if not hasattr(module, "weight"):
            raise TypeError(f"target {path!r} has no weight")
        p = QuantParam(module.weight.detach(), use_lora, rank, cfg)
        parametrize.register_parametrization(module, "weight", p)
        module.parametrizations.weight.original.requires_grad_(False)
        handles.append(Handle(str(path), module, p))
    return handles


def _materialize_one(handle: Handle) -> dict[str, Any]:
    torch = _torch()
    from torch.nn.utils import parametrize
    module = handle.module
    p = module.parametrizations.weight[0]
    original = module.parametrizations.weight.original.detach().clone()
    integer, scale = p.hard_parts(original)
    value = (_view(original, scale) * integer.to(original.dtype)).to(original.dtype)
    parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
    with torch.no_grad():
        module.weight.copy_(value)
    return {
        "path": handle.path,
        "q_min": -7,
        "q_max": 7,
        "shape": list(integer.shape),
        "integer": integer.detach().cpu(),
        "scale": scale.detach().cpu(),
        "integer_min": int(integer.min().item()),
        "integer_max": int(integer.max().item()),
        "use_lora": bool(p.use_lora),
    }


def materialize_hard(handles: Sequence[Handle]) -> list[dict[str, Any]]:
    return [_materialize_one(handle) for handle in handles]


def discard(handles: Sequence[Handle]) -> None:
    torch = _torch()
    from torch.nn.utils import parametrize
    for handle in handles:
        module = handle.module
        original = module.parametrizations.weight.original.detach().clone()
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
        with torch.no_grad():
            module.weight.copy_(original)


def materialize_rtn(modules: Sequence[tuple[str, Any]], cfg: W4Config = W4Config()) -> list[dict[str, Any]]:
    rows = []
    torch = _torch()
    with torch.no_grad():
        for path, module in modules:
            value, scale, integer = q0_weight(module.weight, cfg)
            module.weight.copy_(value.to(module.weight.dtype))
            rows.append({"path": str(path), "q_min": cfg.q_min, "q_max": cfg.q_max, "shape": list(integer.shape), "integer": integer.detach().cpu(), "scale": scale.detach().cpu()})
    return rows


def hard_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["path"]): {"integer": row["integer"], "scale": row["scale"]} for row in rows}


def restore_hard_map(modules: Sequence[tuple[str, Any]], rows: Mapping[str, Mapping[str, Any]]) -> None:
    torch = _torch()
    with torch.no_grad():
        for path, module in modules:
            row = rows[str(path)]
            integer, scale = row["integer"], row["scale"]
            if integer.dtype != torch.int8:
                raise TypeError(f"hard integer for {path} must be int8")
            if tuple(integer.shape) != tuple(module.weight.shape) or scale.shape[0] != module.weight.shape[0]:
                raise ValueError(f"hard quant shape mismatch for {path}")
            if int(integer.min()) < -7 or int(integer.max()) > 7:
                raise ValueError(f"hard integer outside signed W4 for {path}")
            if not bool(torch.isfinite(scale).all().item()) or bool((scale <= 0).any().item()):
                raise ValueError(f"hard scale for {path} is not finite and positive")
            module.weight.copy_(_view(module.weight, scale.to(module.weight.device)) * integer.to(device=module.weight.device, dtype=module.weight.dtype))


def grad_norms(handles: Sequence[Handle]) -> dict[str, float]:
    result: dict[str, float] = {}
    for handle in handles:
        p = handle.parametrization
        for name in ("scale_raw", "alpha", "lora_A", "lora_B"):
            value = getattr(p, name, None)
            if value is not None and value.grad is not None:
                result[f"{handle.path}:{name}"] = float(value.grad.detach().norm().item())
    return result


def parameter_counts(handles: Sequence[Handle]) -> dict[str, int]:
    quant = sum(int(p.numel()) for h in handles for p in (h.parametrization.scale_raw, h.parametrization.alpha))
    lora = sum(int(p.numel()) for h in handles for p in (getattr(h.parametrization, "lora_A", None), getattr(h.parametrization, "lora_B", None)) if p is not None)
    return {"quantizer": quant, "lora": lora}
