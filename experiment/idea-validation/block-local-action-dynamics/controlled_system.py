"""Paired 64D controlled systems for the frozen round-one mechanism study."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


CONDITIONS = ("independent", "lowrank_coupled", "dense_coupled")
STATE_DIM, BLOCKS, BLOCK_DIM, ACTION_DIM = 64, 4, 16, 8


def _data_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("data", config)


def _orthogonal(generator: torch.Generator, dim: int) -> torch.Tensor:
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    q = q * torch.where(torch.diag(r) < 0, -1.0, 1.0)[None, :]
    return q.to(torch.float32)


class ControlledSystem:
    """Dynamics live in known x coordinates; observations are z = M x."""

    def __init__(self, condition: str, seed: int, device: torch.device | str):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        self.condition = condition
        self.device = torch.device(device)

        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        self.M = _orthogonal(rng, STATE_DIM).to(self.device)

        local_a = torch.randn(STATE_DIM, STATE_DIM, generator=rng) / BLOCK_DIM**0.5
        mask = torch.zeros(STATE_DIM, STATE_DIM)
        for block in range(BLOCKS):
            sl = slice(block * BLOCK_DIM, (block + 1) * BLOCK_DIM)
            mask[sl, sl] = 1
        local_a *= mask
        local_b = torch.randn(STATE_DIM, ACTION_DIM, generator=rng) / ACTION_DIM**0.5
        self.local_A = local_a.to(self.device)
        self.local_B = local_b.to(self.device)

        # Each rank-one component reads one block and writes only to its neighbor.
        source = torch.zeros(STATE_DIM, BLOCKS)
        target = torch.zeros(STATE_DIM, BLOCKS)
        for rank in range(BLOCKS):
            src = torch.randn(BLOCK_DIM, generator=rng)
            dst = torch.randn(BLOCK_DIM, generator=rng)
            src = src / src.norm()
            dst = dst / dst.norm()
            source[rank * BLOCK_DIM : (rank + 1) * BLOCK_DIM, rank] = src
            dest_block = (rank + 1) % BLOCKS
            target[dest_block * BLOCK_DIM : (dest_block + 1) * BLOCK_DIM, rank] = dst
        self.lowrank_source = source.to(self.device)
        self.lowrank_target = target.to(self.device)

        self.dense_matrix = _orthogonal(rng, STATE_DIM).to(self.device)
        self.dense_action = (torch.randn(STATE_DIM, ACTION_DIM, generator=rng) / ACTION_DIM**0.5).to(self.device)
        self.coupling_scale = 0.08 if condition == "lowrank_coupled" else 0.035

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.M.T

    def recover(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.M

    def local_delta(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        local = F.linear(x, self.local_A) + F.linear(action, self.local_B)
        return -0.2 * x + 0.15 * torch.tanh(local)

    def coupling_delta(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.condition == "independent":
            return torch.zeros_like(x)
        if self.condition == "lowrank_coupled":
            message = x @ self.lowrank_source
            return self.coupling_scale * (message @ self.lowrank_target.T)
        state_features = torch.tanh(F.linear(x, self.dense_matrix))
        action_gate = 1.0 + 0.5 * torch.tanh(F.linear(action, self.dense_action))
        return self.coupling_scale * state_features * action_gate

    def step_x(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return x + self.local_delta(x, action) + self.coupling_delta(x, action)

    def step_z(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.observe(self.step_x(self.recover(z), action))

    def known_coordinate_diagnostics(self, x: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        local = self.local_delta(x, action)
        coupling = self.coupling_delta(x, action)
        return {
            "x": x,
            "local_delta": local,
            "coupling_delta": coupling,
            "next_x": x + local + coupling,
        }


def _make_split(
    system: ControlledSystem,
    episodes: int,
    steps: int,
    seed: int,
    action_range: tuple[float, float],
    initial_state_std: float,
) -> dict[str, torch.Tensor]:
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    low, high = action_range
    actions = low + (high - low) * torch.rand(episodes, steps, ACTION_DIM, generator=rng)
    x = initial_state_std * torch.randn(episodes, STATE_DIM, generator=rng)
    states = [system.observe(x.to(system.device)).cpu()]
    with torch.no_grad():
        for t in range(steps):
            a = actions[:, t].to(system.device)
            x = system.step_x(x.to(system.device), a)
            states.append(system.observe(x).cpu())
    return {"z": torch.stack(states, dim=1), "action": actions}


def make_datasets(
    config: dict[str, Any], condition: str, device: torch.device | str
) -> tuple[ControlledSystem, dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Generate paired train/dev/test episodes and train-only normalization."""
    data = _data_config(config)
    target_device = torch.device(device)
    seed = int(data.get("system_seed", 20260926))
    system = ControlledSystem(condition, seed, target_device)
    steps = int(data.get("trajectory_steps", 40))
    action_range = tuple(data.get("action_range", (-1.0, 1.0)))
    initial_std = float(data.get("initial_state_std", 0.7))

    split_specs = (
        ("train", "train_seed", "train_episodes", 4101, 512),
        ("dev", "dev_seed", "dev_episodes", 4102, 128),
        ("test", "test_seed", "test_episodes", 4103, 128),
    )
    datasets: dict[str, dict[str, torch.Tensor]] = {}
    split_seeds: dict[str, int] = {}
    for name, seed_key, count_key, default_seed, default_count in split_specs:
        split_seed = int(data.get(seed_key, default_seed))
        split_seeds[name] = split_seed
        datasets[name] = _make_split(
            system,
            int(data.get(count_key, default_count)),
            steps,
            split_seed,
            action_range,
            initial_std,
        )

    train = datasets["train"]
    delta = train["z"][:, 1:] - train["z"][:, :-1]
    mean = train["z"].mean(dim=(0, 1))
    delta_mean = delta.mean()
    delta_energy = float((delta - delta_mean).square().mean().clamp_min(1e-6).item())
    datasets = {
        split: {key: tensor.to(target_device) for key, tensor in values.items()}
        for split, values in datasets.items()
    }
    metadata: dict[str, Any] = {
        "mean": mean.to(target_device),
        "delta_energy": delta_energy,
        "delta_mean": float(delta_mean.item()),
        "condition": condition,
        "system_seed": seed,
        "coupling_scale": float(system.coupling_scale),
        "coupling_rank": 0 if condition == "independent" else 4 if condition == "lowrank_coupled" else STATE_DIM,
        "split_seeds": split_seeds,
        "episode_counts": {split: int(values["z"].shape[0]) for split, values in datasets.items()},
        "trajectory_steps": steps,
    }
    return system, datasets, metadata
