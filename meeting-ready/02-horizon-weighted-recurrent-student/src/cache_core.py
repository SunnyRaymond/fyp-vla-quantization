"""Minimal cached-prefix helpers used by the bundled DINO-WM teacher path.

This is the local copy of the exact-prefix rollout helper required by the
predictor runners.  It keeps observation encoding outside the action-prefix
student and preserves the official action encoder/predictor ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _require_prefix(encoded_obs: Mapping[str, torch.Tensor]) -> None:
    missing = {"visual", "proprio"}.difference(encoded_obs)
    if missing:
        raise KeyError(f"encoded observation is missing {sorted(missing)}")
    if not isinstance(encoded_obs["visual"], torch.Tensor):
        raise TypeError("encoded_obs['visual'] must be a torch.Tensor")
    if not isinstance(encoded_obs["proprio"], torch.Tensor):
        raise TypeError("encoded_obs['proprio'] must be a torch.Tensor")


def _encode_from_encoded_obs(
    model: Any,
    encoded_obs: Mapping[str, torch.Tensor],
    act_0: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the action-conditioning part of ``VWorldModel.encode``."""

    _require_prefix(encoded_obs)
    act_emb = model.encode_act(act_0)
    visual = encoded_obs["visual"]
    proprio = encoded_obs["proprio"]
    if getattr(model, "concat_dim") == 0:
        return torch.cat([visual, proprio.unsqueeze(2), act_emb.unsqueeze(2)], dim=2)
    if getattr(model, "concat_dim") == 1:
        num_patches = visual.shape[2]
        proprio_tiled = proprio.unsqueeze(2).expand(-1, -1, num_patches, -1)
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, model.num_proprio_repeat)
        act_tiled = act_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
        act_repeated = act_tiled.repeat(1, 1, 1, model.num_action_repeat)
        return torch.cat([visual, proprio_repeated, act_repeated], dim=3)
    raise ValueError(f"unsupported model concat_dim={model.concat_dim!r}")


def rollout_from_encoded_obs(
    model: Any,
    encoded_obs: Mapping[str, torch.Tensor],
    act: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run the official rollout after reusing an encoded observation prefix."""

    _require_prefix(encoded_obs)
    if not isinstance(act, torch.Tensor) or act.ndim != 3:
        raise ValueError("act must be a tensor with shape (batch, time, dim)")
    visual = encoded_obs["visual"]
    if visual.ndim < 2:
        raise ValueError("encoded visual prefix must have batch and time axes")
    if act.shape[0] != visual.shape[0]:
        raise ValueError(
            f"batch mismatch: encoded prefix has {visual.shape[0]}, "
            f"actions have {act.shape[0]}"
        )
    num_obs_init = visual.shape[1]
    if act.shape[1] < num_obs_init:
        raise ValueError("act must include one action for each initial observation frame")

    act_0 = act[:, :num_obs_init]
    action = act[:, num_obs_init:]
    z = _encode_from_encoded_obs(model, encoded_obs, act_0)
    t = 0
    inc = 1
    while t < action.shape[1]:
        z_pred = model.predict(z[:, -model.num_hist :])
        z_new = z_pred[:, -inc:, ...]
        z_new = model.replace_actions_from_z(z_new, action[:, t : t + inc, :])
        z = torch.cat([z, z_new], dim=1)
        t += inc
    z_pred = model.predict(z[:, -model.num_hist :])
    z = torch.cat([z, z_pred[:, -1:, ...]], dim=1)
    z_obses, _ = model.separate_emb(z)
    return z_obses, z


__all__ = ["rollout_from_encoded_obs"]
