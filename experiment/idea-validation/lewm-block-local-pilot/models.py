"""Small recurrent adapters for the LeWM 192-D latent interface."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LATENT_DIM = 192
ACTION_DIM = 10
HISTORY_STEPS = 3
BLOCKS = 6
BLOCK_DIM = 32
MESSAGE_DIM = 16
BLOCK_HIDDEN = 271
FLAT_HIDDEN = 550
ACTION_WINDOW = 3

BLOCK_ARMS = {
    "block_adapter_global16",
    "block_adapter_local16",
    "block_identity_global16",
}
ARMS = BLOCK_ARMS | {"flat_adapter"}


def _latest3(history: Tensor) -> Tensor:
    if history.ndim != 3 or history.shape[-1] != LATENT_DIM:
        raise ValueError(f"history must have shape [B,T,{LATENT_DIM}]")
    if history.shape[1] < 1:
        raise ValueError("history must contain at least one latent")
    recent = history[:, -HISTORY_STEPS:, :]
    if recent.shape[1] == HISTORY_STEPS:
        return recent
    left = recent[:, :1, :].expand(-1, HISTORY_STEPS - recent.shape[1], -1)
    return torch.cat((left, recent), dim=1)


def _previous_actions(history: Tensor, actions: Tensor | None) -> Tensor:
    """Validate an optional initial action history and return its trailing tokens."""
    required = min(max(history.shape[1] - 1, 0), ACTION_WINDOW - 1)
    if actions is None:
        return history.new_zeros((history.shape[0], 0, ACTION_DIM))
    if actions.ndim != 3 or actions.shape[0] != history.shape[0] or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"initial_action_history must have shape [B,P,{ACTION_DIM}]")
    if actions.shape[1] != required:
        raise ValueError(f"expected {required} previous action tokens for this history, got {actions.shape[1]}")
    return actions


def _action_window(action: Tensor, previous: Tensor) -> Tensor:
    if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must have shape [B,{ACTION_DIM}]")
    if previous.ndim != 3 or previous.shape[0] != action.shape[0] or previous.shape[-1] != ACTION_DIM:
        raise ValueError(f"previous actions must have shape [B,P,{ACTION_DIM}]")
    if previous.shape[1] > ACTION_WINDOW - 1:
        previous = previous[:, -(ACTION_WINDOW - 1):, :]
    missing = ACTION_WINDOW - 1 - previous.shape[1]
    left = action.new_zeros((action.shape[0], missing, ACTION_DIM))
    return torch.cat((left, previous, action.unsqueeze(1)), dim=1)


class CayleyRotation(nn.Module):
    """Learnable full-width orthogonal coordinates with an eval-only cache."""

    def __init__(self, frozen: bool = False) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(LATENT_DIM, LATENT_DIM), requires_grad=not frozen)
        self.register_buffer("_cached_q", None, persistent=False)

    def compute_matrix(self) -> Tensor:
        a = self.raw - self.raw.transpose(0, 1)
        eye = torch.eye(LATENT_DIM, device=a.device, dtype=a.dtype)
        return torch.linalg.solve(eye + a, eye - a)

    def matrix_for_forward(self) -> Tensor:
        if self.training:
            return self.compute_matrix()
        if self._cached_q is None:
            raise RuntimeError("Call eval() followed by prepare_for_inference() before evaluation")
        return self._cached_q

    def prepare_for_inference(self) -> None:
        if self.training:
            raise RuntimeError("Call eval() before prepare_for_inference()")
        with torch.no_grad():
            self._cached_q = self.compute_matrix().detach()

    def train(self, mode: bool = True) -> "CayleyRotation":
        super().train(mode)
        self._cached_q = None
        return self

    def _apply(self, fn):
        self._cached_q = None
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        self._cached_q = None
        super()._load_from_state_dict(*args, **kwargs)


class _DynamicsModel(nn.Module):
    arm: str

    def __init__(self, arm: str, frozen_rotation: bool = False) -> None:
        super().__init__()
        self.arm = arm
        self.rotation = CayleyRotation(frozen=frozen_rotation)

    def prepare_for_inference(self) -> "_DynamicsModel":
        if self.training:
            raise RuntimeError("Call model.eval() before prepare_for_inference()")
        self.rotation.prepare_for_inference()
        return self

    def step(
        self, history: Tensor, action: Tensor, action_history: Tensor | None = None
    ) -> Tensor:
        return self._step(history, action, action_history)

    def _step(self, history: Tensor, action: Tensor, action_history: Tensor | None) -> Tensor:
        if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
            raise ValueError(f"action must have shape [B,{ACTION_DIM}]")
        if history.shape[0] != action.shape[0]:
            raise ValueError("history and action batch sizes differ")
        previous = _previous_actions(history, action_history)
        return self._step_with_q(
            _latest3(history), _action_window(action, previous), self.rotation.matrix_for_forward()
        )

    def _step_with_q(self, recent: Tensor, action_window: Tensor, q: Tensor) -> Tensor:
        s_history = recent @ q
        delta_s = self._predict_delta_rotated(s_history, action_window)
        delta_z = delta_s @ q.transpose(0, 1)
        return recent[:, -1, :] + delta_z

    def predict_rollout(
        self, history: Tensor, actions: Tensor, initial_action_history: Tensor | None = None
    ) -> Tensor:
        """Return only autoregressively predicted latents, with no teacher forcing."""
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"actions must have shape [B,H,{ACTION_DIM}]")
        if actions.shape[0] != history.shape[0]:
            raise ValueError("history and actions batch sizes differ")
        if actions.shape[1] < 1:
            raise ValueError("actions must contain at least one step")

        recent = _latest3(history)
        previous = _previous_actions(history, initial_action_history)
        q = self.rotation.matrix_for_forward()  # One solve per training rollout.
        future = []
        for t in range(actions.shape[1]):
            current = actions[:, t, :]
            window = _action_window(current, previous)
            z_next = self._step_with_q(recent, window, q)
            future.append(z_next)
            recent = torch.cat((recent, z_next.unsqueeze(1)), dim=1)[:, -HISTORY_STEPS:, :]
            previous = torch.cat((previous, current.unsqueeze(1)), dim=1)[:, -(ACTION_WINDOW - 1):, :]
        return torch.stack(future, dim=1)

    def forward(
        self, history: Tensor, actions: Tensor, initial_action_history: Tensor | None = None
    ) -> Tensor:
        return self.predict_rollout(history, actions, initial_action_history)

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def _predict_delta_rotated(self, s_history: Tensor, action_window: Tensor) -> Tensor:
        raise NotImplementedError


class _BatchedBlockMLP(nn.Module):
    """Six independent 142-271-271-32 MLPs evaluated with batched einsums."""

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(BLOCKS, BLOCK_HIDDEN, 142))
        self.b1 = nn.Parameter(torch.empty(BLOCKS, BLOCK_HIDDEN))
        self.w2 = nn.Parameter(torch.empty(BLOCKS, BLOCK_HIDDEN, BLOCK_HIDDEN))
        self.b2 = nn.Parameter(torch.empty(BLOCKS, BLOCK_HIDDEN))
        self.w3 = nn.Parameter(torch.empty(BLOCKS, BLOCK_DIM, BLOCK_HIDDEN))
        self.b3 = nn.Parameter(torch.empty(BLOCKS, BLOCK_DIM))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in ((self.w1, self.b1), (self.w2, self.b2), (self.w3, self.b3)):
            for block in range(BLOCKS):
                nn.init.kaiming_uniform_(weight[block], a=math.sqrt(5))
                bound = 1 / math.sqrt(weight.shape[-1])
                nn.init.uniform_(bias[block], -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(torch.einsum("bki,koi->bko", x, self.w1) + self.b1)
        x = F.gelu(torch.einsum("bki,koi->bko", x, self.w2) + self.b2)
        return torch.einsum("bki,koi->bko", x, self.w3) + self.b3


class BlockAdapter(_DynamicsModel):
    def __init__(self, arm: str) -> None:
        super().__init__(arm, frozen_rotation=(arm == "block_identity_global16"))
        if arm not in BLOCK_ARMS:
            raise ValueError(f"unsupported block arm: {arm}")
        # Drawing the shared projection first makes global/local seeded inits pair exactly.
        seed_projection = nn.Linear(HISTORY_STEPS * LATENT_DIM, MESSAGE_DIM, bias=False)
        if arm == "block_adapter_local16":
            local = seed_projection.weight.detach().reshape(
                MESSAGE_DIM, HISTORY_STEPS, BLOCKS, BLOCK_DIM
            ).permute(2, 0, 1, 3).reshape(BLOCKS, MESSAGE_DIM, HISTORY_STEPS * BLOCK_DIM)
            self.message_weight = nn.Parameter(local.clone())
        else:
            self.message_weight = nn.Parameter(seed_projection.weight.detach().clone())
        self.functions = _BatchedBlockMLP()
        self.local_messages = arm == "block_adapter_local16"

    def _predict_delta_rotated(self, s_history: Tensor, action_window: Tensor) -> Tensor:
        batch = s_history.shape[0]
        blocks = s_history.reshape(batch, HISTORY_STEPS, BLOCKS, BLOCK_DIM)
        block_history = blocks.permute(0, 2, 1, 3).reshape(batch, BLOCKS, HISTORY_STEPS * BLOCK_DIM)
        if self.local_messages:
            message = torch.einsum("bki,kmi->bkm", block_history, self.message_weight)
        else:
            message = F.linear(s_history.reshape(batch, -1), self.message_weight)
            message = message.unsqueeze(1).expand(-1, BLOCKS, -1)
        actions = action_window.reshape(batch, -1).unsqueeze(1).expand(-1, BLOCKS, -1)
        x = torch.cat((block_history, actions, message), dim=-1)
        return self.functions(x).reshape(batch, LATENT_DIM)


class FlatAdapter(_DynamicsModel):
    def __init__(self) -> None:
        super().__init__("flat_adapter")
        self.fc1 = nn.Linear(HISTORY_STEPS * LATENT_DIM + ACTION_WINDOW * ACTION_DIM, FLAT_HIDDEN)
        self.fc2 = nn.Linear(FLAT_HIDDEN, FLAT_HIDDEN)
        self.fc3 = nn.Linear(FLAT_HIDDEN, LATENT_DIM)

    def _predict_delta_rotated(self, s_history: Tensor, action_window: Tensor) -> Tensor:
        x = torch.cat((s_history.reshape(s_history.shape[0], -1), action_window.reshape(action_window.shape[0], -1)), dim=-1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


def build_model(arm: str) -> _DynamicsModel:
    if arm == "flat_adapter":
        return FlatAdapter()
    if arm in BLOCK_ARMS:
        return BlockAdapter(arm)
    raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
