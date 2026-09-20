"""Small, exact-prefix helpers for the DINO-WM rollout path.

The helpers mirror ``VWorldModel.encode`` and ``VWorldModel.rollout`` without
changing the model implementation.  The observation prefix is encoded once;
the action encoder, predictor, and action replacement remain on the original
candidate-specific path.

The intended use in the CEM planner is:

* ``baseline``: call the model's original ``rollout``;
* ``iteration_cache``: encode the already candidate-batched observation once,
  retain the returned dictionary between CEM iterations, and pass it back as
  ``cached_prefix``;
* ``factorized_cache``: encode a single observation and expand the encoded
  dictionary to the candidate batch for each CEM iteration.

This module deliberately does not implement cache keys, dependency discovery,
canaries, or planner logic.  Those policies belong to the experiment driver.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch


CacheMode = Literal["baseline", "iteration_cache", "factorized_cache"]
_CACHE_MODES = {"baseline", "iteration_cache", "factorized_cache"}


def _require_mode(mode: str) -> None:
    if mode not in _CACHE_MODES:
        raise ValueError(
            f"unknown cache mode {mode!r}; expected one of "
            "baseline, iteration_cache, factorized_cache"
        )


def _require_prefix(encoded_obs: Mapping[str, torch.Tensor]) -> None:
    missing = {"visual", "proprio"}.difference(encoded_obs)
    if missing:
        raise KeyError(f"encoded observation is missing {sorted(missing)}")
    if not isinstance(encoded_obs["visual"], torch.Tensor):
        raise TypeError("encoded_obs['visual'] must be a torch.Tensor")
    if not isinstance(encoded_obs["proprio"], torch.Tensor):
        raise TypeError("encoded_obs['proprio'] must be a torch.Tensor")


def repeat_encoded_prefix(
    encoded_obs: Mapping[str, torch.Tensor], repeats: int
) -> dict[str, torch.Tensor]:
    """Return a read-only expanded view of a single encoded observation.

    ``factorized_cache`` uses this after encoding a batch of size one.  The
    returned tensors share storage with the cached tensors and are only read by
    the rollout helper; no in-place operation is performed here.
    """

    _require_prefix(encoded_obs)
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    expanded: dict[str, torch.Tensor] = {}
    for key, value in encoded_obs.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"encoded_obs[{key!r}] must be a torch.Tensor")
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError(
                "repeat_encoded_prefix expects every prefix tensor to have "
                "a leading batch dimension of size one"
            )
        expanded[key] = value.expand((repeats,) + tuple(value.shape[1:]))
    return expanded


def encode_and_repeat(
    model: Any,
    obs_0: Mapping[str, torch.Tensor],
    *,
    mode: CacheMode,
    repeats: int | None = None,
    cached_prefix: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Encode an observation prefix according to one of the three cache modes.

    ``baseline`` expects ``obs_0`` to already have the candidate batch and
    simply calls ``model.encode_obs``.  ``iteration_cache`` returns
    ``cached_prefix`` when supplied, otherwise it encodes ``obs_0``; callers
    should retain the first result across CEM iterations.  ``factorized_cache``
    expects a batch of one and expands that prefix to ``repeats`` candidates.
    """

    _require_mode(mode)
    if mode == "baseline":
        return dict(model.encode_obs(obs_0))

    if mode == "iteration_cache":
        if cached_prefix is not None:
            _require_prefix(cached_prefix)
            return dict(cached_prefix)
        return dict(model.encode_obs(obs_0))

    if repeats is None:
        raise ValueError("factorized_cache requires repeats")
    prefix = cached_prefix
    if prefix is None:
        prefix = model.encode_obs(obs_0)
    _require_prefix(prefix)
    return repeat_encoded_prefix(prefix, repeats)


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
        return torch.cat(
            [visual, proprio.unsqueeze(2), act_emb.unsqueeze(2)], dim=2
        )

    if getattr(model, "concat_dim") == 1:
        num_patches = visual.shape[2]
        proprio_tiled = proprio.unsqueeze(2).expand(-1, -1, num_patches, -1)
        proprio_repeated = proprio_tiled.repeat(
            1, 1, 1, model.num_proprio_repeat
        )
        act_tiled = act_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
        act_repeated = act_tiled.repeat(1, 1, 1, model.num_action_repeat)
        return torch.cat([visual, proprio_repeated, act_repeated], dim=3)

    raise ValueError(f"unsupported model concat_dim={model.concat_dim!r}")


def rollout_from_encoded_obs(
    model: Any,
    encoded_obs: Mapping[str, torch.Tensor],
    act: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run the original DINO-WM rollout after reusing ``encoded_obs``.

    ``encoded_obs`` is the output of ``model.encode_obs`` and contains only
    visual/proprio features.  Every action-conditioned operation is kept in
    the same order as ``VWorldModel.rollout``: initial ``encode_act``,
    predictor calls, and ``replace_actions_from_z`` for each suffix step.
    """

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
        raise ValueError(
            "act must include one action for each initial observation frame"
        )

    act_0 = act[:, :num_obs_init]
    action = act[:, num_obs_init:]
    z = _encode_from_encoded_obs(model, encoded_obs, act_0)

    t = 0
    inc = 1
    while t < action.shape[1]:
        z_pred = model.predict(z[:, -model.num_hist :])
        z_new = z_pred[:, -inc:, ...]
        z_new = model.replace_actions_from_z(
            z_new, action[:, t : t + inc, :]
        )
        z = torch.cat([z, z_new], dim=1)
        t += inc

    z_pred = model.predict(z[:, -model.num_hist :])
    z_new = z_pred[:, -1:, ...]
    z = torch.cat([z, z_new], dim=1)
    z_obses, _ = model.separate_emb(z)
    return z_obses, z


def rollout_with_cache(
    model: Any,
    obs_0: Mapping[str, torch.Tensor],
    act: torch.Tensor,
    *,
    mode: CacheMode,
    cached_prefix: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run a baseline or cached rollout through one explicit mode.

    For ``factorized_cache``, ``obs_0`` may be either a single observation or
    the candidate-repeated observation used by CEM.  Only its first batch item
    is encoded, under the caller's contract that all candidates share that
    observation; the encoded prefix is expanded to ``act.shape[0]``.
    """

    _require_mode(mode)
    if mode == "baseline":
        return model.rollout(obs_0, act)

    if mode == "iteration_cache":
        prefix = encode_and_repeat(
            model, obs_0, mode=mode, cached_prefix=cached_prefix
        )
    else:
        single_obs = {
            key: value[:1] if isinstance(value, torch.Tensor) else value
            for key, value in obs_0.items()
        }
        prefix = encode_and_repeat(
            model,
            single_obs,
            mode=mode,
            repeats=act.shape[0],
            cached_prefix=cached_prefix,
        )

    return rollout_from_encoded_obs(model, prefix, act)


__all__ = [
    "CacheMode",
    "encode_and_repeat",
    "repeat_encoded_prefix",
    "rollout_from_encoded_obs",
    "rollout_with_cache",
]
