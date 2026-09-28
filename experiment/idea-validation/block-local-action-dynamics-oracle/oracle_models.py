"""Known-coordinate wrappers around the first-round predictors."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


ORACLE_TO_INITIALIZER = {
    "oracle_block": "learned_block",
    "oracle_global4": "learned_global4",
    "oracle_local4": "learned_local4",
    "oracle_global16": "learned_global16",
}


def build_oracle_model(
    arm: str,
    config: dict[str, Any],
    seed: int,
    mean: torch.Tensor,
    generator_m: torch.Tensor,
) -> nn.Module:
    """Freeze Q=M while retaining the corresponding learned arm's f initialization."""
    from models import build_model

    if arm not in ORACLE_TO_INITIALIZER:
        raise ValueError(f"unknown oracle arm: {arm}")
    state_dim = int(config["dimensions"]["state"])
    if tuple(generator_m.shape) != (state_dim, state_dim):
        raise ValueError(f"expected M shape {(state_dim, state_dim)}, got {tuple(generator_m.shape)}")
    base_arm = ORACLE_TO_INITIALIZER[arm]
    model = build_model(base_arm, config, seed, mean)
    with torch.random.fork_rng(devices=[]):
        fixed_transform = nn.Linear(state_dim, state_dim, bias=False)
    with torch.no_grad():
        fixed_transform.weight.copy_(generator_m.detach().to(dtype=fixed_transform.weight.dtype).T)
    fixed_transform.weight.requires_grad_(False)
    model.transform = fixed_transform
    model.arm = base_arm
    model._inference_q = None
    return model


def predictor_parameter_views(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return predictor/message tensors while excluding its coordinate transform."""
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("transform.")
    }
