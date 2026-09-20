"""LeWM CEM model wrapper for planner-call embedding reuse.

The wrapper deliberately leaves candidate-axis expansion and the action-conditioned
suffix untouched.  It only replaces the repeated initial/goal encodes made by the
official ``JEPA.get_cost`` path with two detached tensors scoped to one solve call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EmbeddingCache:
    initial_embedding: torch.Tensor
    goal_embedding: torch.Tensor


def _detach_clone(value: Any) -> Any:
    return value.detach().clone() if torch.is_tensor(value) else value


def _goal_info(info_dict: dict[str, Any]) -> dict[str, Any]:
    """Build the goal-only view consumed by the official JEPA encoder."""
    return {"pixels": info_dict["goal"][:, 0]}


def _initial_info(info_dict: dict[str, Any]) -> dict[str, Any]:
    """Build the state-only view; candidate actions are encoded during rollout."""
    return {"pixels": info_dict["pixels"][:, 0]}


def rollout_with_cached_initial(
    model: torch.nn.Module,
    info_dict: dict[str, Any],
    action_candidates: torch.Tensor,
    initial_embedding: torch.Tensor,
    history_size: int = 3,
) -> dict[str, Any]:
    """Mirror the official rollout while replacing only its initial encode.

    Shapes remain the official ``(B,S,T,...)`` candidate layout.  ``initial_embedding``
    is ``(B,T,D)`` and is expanded across S exactly where the official rollout
    expands the freshly encoded embedding.
    """
    pixels = info_dict["pixels"]
    H = pixels.size(2)
    batch, samples, horizon = action_candidates.shape[:3]
    act_0, act_future = torch.split(action_candidates, [H, horizon - H], dim=2)
    emb = initial_embedding.unsqueeze(1).expand(batch, samples, -1, -1)
    emb = torch.reshape(emb, (batch * samples,) + tuple(emb.shape[2:])).clone()
    act = torch.reshape(act_0, (batch * samples,) + tuple(act_0.shape[2:]))
    act_future = torch.reshape(act_future, (batch * samples,) + tuple(act_future.shape[2:]))

    for step in range(horizon - H):
        act_emb = model.action_encoder(act)
        emb_trunc = emb[:, -history_size:]
        act_trunc = act_emb[:, -history_size:]
        pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)
        next_act = act_future[:, step : step + 1, :]
        act = torch.cat([act, next_act], dim=1)

    act_emb = model.action_encoder(act)
    emb_trunc = emb[:, -history_size:]
    act_trunc = act_emb[:, -history_size:]
    pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]
    emb = torch.cat([emb, pred_emb], dim=1)
    predicted = torch.reshape(emb, (batch, samples) + tuple(emb.shape[1:]))
    result = dict(info_dict)
    result["predicted_emb"] = predicted
    return result


class IterationCacheModel(torch.nn.Module):
    """A read-only-input wrapper compatible with stable-worldmodel CEMSolver."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self._cache_key: int | None = None
        self._cache: EmbeddingCache | None = None
        self._cache_events: list[tuple[Any, Any]] = []

    def clear_cache(self) -> None:
        self._cache_key = None
        self._cache = None
        self._cache_events.clear()

    @property
    def cache_setup_ms(self) -> float:
        if not self._cache_events:
            return 0.0
        torch.cuda.synchronize()
        return float(sum(start.elapsed_time(end) for start, end in self._cache_events))

    def get_cost(self, info_dict: dict[str, Any], action_candidates: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        working = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info_dict.items()
        }
        key = id(info_dict)
        if self._cache_key != key or self._cache is None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                initial = self.model.encode(_initial_info(working))["emb"]
                goal = self.model.encode(_goal_info(working))["emb"]
            end_event.record()
            self._cache_events.append((start_event, end_event))
            self._cache = EmbeddingCache(
                initial_embedding=_detach_clone(initial),
                goal_embedding=_detach_clone(goal),
            )
            self._cache_key = key

        assert self._cache is not None
        rolled = rollout_with_cached_initial(
            self.model,
            working,
            action_candidates,
            self._cache.initial_embedding,
        )
        rolled["goal_emb"] = self._cache.goal_embedding
        return self.model.criterion(rolled)
