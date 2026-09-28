"""Matched predictors for the frozen block-local dynamics experiment."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal


STATE_DIM, BLOCKS, BLOCK_DIM, ACTION_DIM, MESSAGE_DIM, HIDDEN_DIM = 64, 4, 16, 8, 4, 64
BLOCK_ARMS = {"random_block", "learned_block", "learned_global4", "learned_global16", "learned_local4"}


def _dimensions(config: dict[str, Any]) -> tuple[int, int, int, int, int]:
    dims = config.get("dimensions", config)
    return (
        int(dims.get("state", STATE_DIM)),
        int(dims.get("blocks", BLOCKS)),
        int(dims.get("block_state", BLOCK_DIM)),
        int(dims.get("action", ACTION_DIM)),
        int(dims.get("message", MESSAGE_DIM)),
    )


def _block_parameter_count(config: dict[str, Any]) -> int:
    state_dim, blocks, block_dim, action_dim, message_dim = _dimensions(config)
    mlp_input = block_dim + action_dim + message_dim
    mlp = blocks * (
        (mlp_input + 1) * HIDDEN_DIM
        + (HIDDEN_DIM + 1) * HIDDEN_DIM
        + (HIDDEN_DIM + 1) * block_dim
    )
    # Learned full orthogonal coordinate transform plus either message projection.
    return mlp + state_dim * state_dim + message_dim * state_dim


def matched_dense_hidden(config: dict[str, Any], mean: torch.Tensor | None) -> int:
    """Choose the integer hidden width nearest the configured global arm's parameter count."""
    del mean  # The parameter budget does not depend on data values.
    state_dim, _, _, action_dim, _ = _dimensions(config)
    target = _block_parameter_count(config)

    def count(width: int) -> int:
        return (state_dim + action_dim + 1) * width + (width + 1) * width + (width + 1) * state_dim

    linear = 2 * state_dim + action_dim + 2
    root = (-linear + (linear * linear + 4 * (target - state_dim)) ** 0.5) / 2
    estimate = max(1, int(root))
    return min(range(max(1, estimate - 2), estimate + 3), key=lambda width: abs(count(width) - target))


def _seeded_orthogonal(seed: int, dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 0x51C)
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    q = q * torch.where(torch.diag(r) < 0, -1.0, 1.0)[None, :]
    return q.to(torch.float32)


class _BlockLinear(nn.Module):
    """Independent linear map per block, applied together without a block loop."""

    def __init__(self, blocks: int, out_features: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(blocks, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(blocks, out_features))
        bound = in_features**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bki,koi->bko", x, self.weight) + self.bias


class _BlockPredictor(nn.Module):
    def __init__(
        self,
        arm: str,
        config: dict[str, Any],
        seed: int,
        mean: torch.Tensor,
    ):
        super().__init__()
        self.arm = arm
        self.state_dim, self.blocks, self.block_dim, self.action_dim, self.message_dim = _dimensions(config)
        self.register_buffer("mean", mean.detach().to(device="cpu", dtype=torch.float32).reshape(self.state_dim).clone())

        q = _seeded_orthogonal(seed, self.state_dim)
        self.transform = nn.Linear(self.state_dim, self.state_dim, bias=False)
        with torch.no_grad():
            self.transform.weight.copy_(q.T)
        if arm == "learned_block" or arm.startswith("learned_"):
            orthogonal(self.transform, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
        else:
            self.transform.weight.requires_grad_(False)

        if arm in {"learned_global4", "learned_global16"}:
            projection = torch.randn(self.message_dim, self.state_dim) / self.state_dim**0.5
            self.global_message = nn.Parameter(projection)
        elif arm == "learned_local4":
            projection = torch.randn(self.message_dim, self.state_dim) / self.state_dim**0.5
            local = projection.reshape(self.message_dim, self.blocks, self.block_dim).permute(1, 0, 2).contiguous()
            self.local_message = nn.Parameter(local)

        input_dim = self.block_dim + self.action_dim + (self.message_dim if arm in {"learned_global4", "learned_global16", "learned_local4"} else 0)
        self.fc1 = _BlockLinear(self.blocks, HIDDEN_DIM, input_dim)
        self.fc2 = _BlockLinear(self.blocks, HIDDEN_DIM, HIDDEN_DIM)
        self.out = _BlockLinear(self.blocks, self.block_dim, HIDDEN_DIM)
        self._inference_q: torch.Tensor | None = None

    def orthogonal_matrix(self) -> torch.Tensor:
        """Return Q in s=(z-mean)@Q coordinates."""
        return self.transform.weight.T

    def _q_for_call(self) -> torch.Tensor:
        if not self.training and self._inference_q is not None:
            return self._inference_q.to(device=self.mean.device, dtype=self.mean.dtype)
        return self.orthogonal_matrix()

    def prepare_inference(self) -> "_BlockPredictor":
        self._inference_q = self.orthogonal_matrix().detach().clone()
        return self

    def clear_inference_cache(self) -> None:
        self._inference_q = None

    def train(self, mode: bool = True) -> "_BlockPredictor":
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _advance(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        blocks = s.reshape(s.shape[0], self.blocks, self.block_dim)
        action_blocks = action[:, None, :].expand(-1, self.blocks, -1)
        pieces = [blocks, action_blocks]
        if self.arm in {"learned_global4", "learned_global16"}:
            message = F.linear(s, self.global_message)
            pieces.append(message[:, None, :].expand(-1, self.blocks, -1))
        elif self.arm == "learned_local4":
            message = torch.einsum("bkd,kmd->bkm", blocks, self.local_message)
            pieces.append(message)
        hidden = F.silu(self.fc1(torch.cat(pieces, dim=-1)))
        hidden = F.silu(self.fc2(hidden))
        delta = self.out(hidden).reshape_as(s)
        return s + delta

    def step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = self._q_for_call()
        s = (z - self.mean) @ q
        return self.mean + self._advance(s, action) @ q.T

    forward = step

    def _rollout_native(self, z0: torch.Tensor, action_seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        q = self._q_for_call()
        s = (z0 - self.mean) @ q
        outputs: list[torch.Tensor] = []
        for t in range(action_seq.shape[1]):
            s = self._advance(s, action_seq[:, t])
            outputs.append(self.mean + s @ q.T)
        return s, outputs

    def rollout(self, z0: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        _, outputs = self._rollout_native(z0, action_seq)
        if outputs:
            return torch.stack(outputs, dim=1)
        return z0.new_empty((z0.shape[0], 0, self.state_dim))

    def rollout_final(self, z0: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        if action_seq.shape[1] == 0:
            return z0
        q = self._q_for_call()
        s = (z0 - self.mean) @ q
        for t in range(action_seq.shape[1]):
            s = self._advance(s, action_seq[:, t])
        return self.mean + s @ q.T


class _DensePredictor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, mean: torch.Tensor, hidden: int):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer("mean", mean.detach().to(device="cpu", dtype=torch.float32).reshape(state_dim).clone())
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat((z - self.mean, action), dim=-1))

    forward = step

    def rollout(self, z0: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        state = z0
        outputs = []
        for t in range(action_seq.shape[1]):
            state = self.step(state, action_seq[:, t])
            outputs.append(state)
        if outputs:
            return torch.stack(outputs, dim=1)
        return z0.new_empty((z0.shape[0], 0, self.state_dim))

    def rollout_final(self, z0: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        state = z0
        for t in range(action_seq.shape[1]):
            state = self.step(state, action_seq[:, t])
        return state

    def prepare_inference(self) -> "_DensePredictor":
        return self

    def clear_inference_cache(self) -> None:
        return None


def build_model(
    arm: str,
    config: dict[str, Any],
    seed: int,
    mean: torch.Tensor,
    dense_hidden: int | None = None,
) -> nn.Module:
    """Build an original-coordinate one-step predictor with paired initialization."""
    if arm not in {"dense", *BLOCK_ARMS}:
        raise ValueError(f"unknown arm {arm!r}")
    state_dim, _, _, action_dim, message_dim = _dimensions(config)
    if arm == "learned_global16" and message_dim != 16:
        raise ValueError("learned_global16 requires dimensions.message == 16")
    mean_cpu = torch.as_tensor(mean).detach().to(device="cpu", dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if arm == "dense":
            hidden = dense_hidden if dense_hidden is not None else matched_dense_hidden(config, mean_cpu)
            model = _DensePredictor(state_dim, action_dim, mean_cpu, int(hidden))
        else:
            model = _BlockPredictor(arm, config, int(seed), mean_cpu)
    return model
