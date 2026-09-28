"""iCEM-inspired population decay for the pinned stable-worldmodel CEM."""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import torch

try:
    from stable_worldmodel.solver import CEMSolver
    from stable_worldmodel.solver.utils import prepare_init_action
except ModuleNotFoundError:
    if __name__ != "__main__" or "--self-test" not in sys.argv:
        raise

    # Keep the CPU self-test runnable without installing the pinned package.
    class CEMSolver:
        def __init__(
            self, model: Any, batch_size: int = 1, num_samples: int = 300,
            var_scale: float = 1, n_steps: int = 30, topk: int = 30,
            device: str | torch.device = "cpu", seed: int = 1234,
            callbacks: list[Any] | None = None,
        ) -> None:
            self.model, self.batch_size = model, batch_size
            self.num_samples, self.n_steps, self.topk = num_samples, n_steps, topk
            self.var_scale, self.device = var_scale, device
            self.torch_gen = torch.Generator(device=device).manual_seed(seed)
            self.callbacks = list(callbacks) if callbacks else []
            self._dtype = torch.float32

        def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
            self._action_space, self._n_envs, self._config = action_space, n_envs, config
            self._action_dim = int(np.prod(action_space.shape[1:]))

        @property
        def n_envs(self) -> int:
            return self._n_envs

        @property
        def action_dim(self) -> int:
            return self._action_dim * self._config.action_block

        @property
        def horizon(self) -> int:
            return self._config.horizon

        @property
        def dtype(self) -> torch.dtype:
            return self._dtype

        def init_action_distrib(self, n_envs: int, actions: torch.Tensor | None = None):
            var = self.var_scale * torch.ones(
                n_envs, self.horizon, self.action_dim, dtype=self.dtype
            )
            mean = torch.zeros(n_envs, 0, self.action_dim, dtype=self.dtype) if actions is None else actions
            if mean.shape[1] < self.horizon:
                mean = torch.cat((mean, torch.zeros(
                    n_envs, self.horizon - mean.shape[1], self.action_dim,
                    dtype=self.dtype, device=mean.device,
                )), dim=1)
            return mean, var

    def prepare_init_action(model: Any, info: dict, init_action: Any, horizon: int, **kwargs: Any):
        return init_action


class DecayCEMSolver(CEMSolver):
    """Pinned CEM semantics with an explicitly supplied per-step population."""

    def __init__(self, *args: Any, candidate_schedule: tuple[int, ...], **kwargs: Any) -> None:
        num_samples = int(kwargs.get("num_samples", args[2] if len(args) > 2 else 300))
        n_steps = int(kwargs.get("n_steps", args[4] if len(args) > 4 else 30))
        topk = int(kwargs.get("topk", args[5] if len(args) > 5 else 30))
        schedule = tuple(candidate_schedule)
        if n_steps != 30 or len(schedule) != 30:
            raise ValueError("candidate_schedule and n_steps must both contain 30 iterations")
        if any(isinstance(n, bool) or not isinstance(n, int) for n in schedule):
            raise TypeError("candidate_schedule must contain 30 integers")
        if not 1 <= topk <= num_samples:
            raise ValueError("topk must satisfy 1 <= topk <= num_samples")
        if any(n < topk or n > num_samples for n in schedule):
            raise ValueError("every candidate count must satisfy topk <= N_i <= num_samples")
        super().__init__(*args, **kwargs)
        self.candidate_schedule = schedule
        self.last_candidate_counts: list[int] = []
        self.last_cost_evaluations = 0

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(
            self.model, info_dict, init_action, self.horizon,
            n_envs=total_envs, action_dim=self.action_dim,
        )
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean, var = mean.to(self.device), var.to(self.device)

        for cb in self.callbacks:
            cb.reset()

        self.last_candidate_counts = []
        self.last_cost_evaluations = 0
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean, batch_var = mean[start_idx:end_idx], var[start_idx:end_idx]
            expanded_infos = {}
            for key, value in info_dict.items():
                value = value[start_idx:end_idx]
                if torch.is_tensor(value):
                    dtype = self.dtype if value.is_floating_point() else None
                    value = value.to(device=self.device, dtype=dtype).unsqueeze(1).expand(
                        current_bs, self.num_samples, *value.shape[1:]
                    )
                elif isinstance(value, np.ndarray):
                    value = np.repeat(value[:, None, ...], self.num_samples, axis=1)
                expanded_infos[key] = value

            final_batch_cost = None
            for cb in self.callbacks:
                cb.start_batch()
            for step, num_candidates in enumerate(self.candidate_schedule):
                candidates = torch.randn(
                    current_bs, num_candidates, self.horizon, self.action_dim,
                    generator=self.torch_gen, device=self.device, dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean
                step_infos = {
                    key: value[:, :num_candidates]
                    if torch.is_tensor(value) or isinstance(value, np.ndarray)
                    else value
                    for key, value in expanded_infos.items()
                }
                costs = self.model.get_cost(step_infos, candidates)
                assert isinstance(costs, torch.Tensor), f"Expected tensor costs, got {type(costs)}"
                assert costs.ndim == 2 and costs.shape == (current_bs, num_candidates), (
                    f"Expected cost shape ({current_bs}, {num_candidates}), got {tuple(costs.shape)}"
                )
                if start_idx == 0:
                    self.last_candidate_counts.append(num_candidates)
                self.last_cost_evaluations += num_candidates * current_bs
                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                prev_mean, prev_var = batch_mean, batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)
                for cb in self.callbacks:
                    cb(
                        step=step, candidates=candidates, costs=costs,
                        topk_vals=topk_vals, topk_inds=topk_inds,
                        topk_candidates=topk_candidates, mean=batch_mean,
                        var=batch_var, prev_mean=prev_mean, prev_var=prev_var,
                    )
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx], var[start_idx:end_idx] = batch_mean, batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        if self.callbacks:
            outputs["callbacks"] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs["callbacks"][cb.output_key] = cb.history
        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs


def _self_test() -> None:
    from types import SimpleNamespace

    class FakeModel:
        def __init__(self) -> None:
            self.shapes = []

        def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
            self.shapes.append((tuple(candidates.shape), tuple(info["tensor"].shape), info["array"].shape))
            return candidates.square().sum(dim=(-1, -2))

    class FakeSpace:
        shape = (3, 2)

    def run(schedule: tuple[int, ...]) -> tuple[DecayCEMSolver, FakeModel, dict]:
        model = FakeModel()
        solver = DecayCEMSolver(
            model=model, batch_size=2, num_samples=max(schedule), n_steps=30,
            topk=30, seed=7, candidate_schedule=schedule,
        )
        solver.configure(action_space=FakeSpace(), n_envs=3, config=SimpleNamespace(horizon=2, action_block=1))
        result = solver.solve({
            "tensor": torch.arange(3).reshape(3, 1),
            "array": np.arange(3).reshape(3, 1),
        })
        assert result["actions"].shape == (3, 2, 2)
        assert torch.isfinite(result["actions"]).all()
        return solver, model, result

    decay = tuple(max(60, round(300 / 1.0572**i)) for i in range(30))
    solver, model, _ = run(decay)
    expected = [((2, n, 2, 2), (2, n, 1), (2, n, 1)) for n in decay]
    expected += [((1, n, 2, 2), (1, n, 1), (1, n, 1)) for n in decay]
    assert model.shapes == expected
    assert solver.last_candidate_counts == list(decay)
    assert solver.last_cost_evaluations == sum(decay) * 3

    fixed = (60,) * 30
    solver, model, fixed_result = run(fixed)
    assert solver.last_candidate_counts == list(fixed)
    assert all(shape[0][1] == shape[1][1] == shape[2][1] == 60 for shape in model.shapes)
    if callable(getattr(CEMSolver, "solve", None)):
        reference_model = FakeModel()
        reference = CEMSolver(
            model=reference_model, batch_size=2, num_samples=60, n_steps=30,
            topk=30, seed=7,
        )
        reference.configure(action_space=FakeSpace(), n_envs=3, config=SimpleNamespace(horizon=2, action_block=1))
        reference_result = reference.solve({
            "tensor": torch.arange(3).reshape(3, 1),
            "array": np.arange(3).reshape(3, 1),
        })
        assert reference_model.shapes == model.shapes
        assert torch.equal(reference_result["actions"], fixed_result["actions"])
        assert torch.equal(torch.as_tensor(reference_result["costs"]), torch.as_tensor(fixed_result["costs"]))
    print("CPU fake-model self-test PASS: decay schedule shapes/counts and fixed-N schedule")


if __name__ == "__main__" and "--self-test" in sys.argv:
    _self_test()
