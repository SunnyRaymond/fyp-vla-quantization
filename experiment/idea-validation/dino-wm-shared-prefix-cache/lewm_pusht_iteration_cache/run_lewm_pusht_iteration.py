#!/usr/bin/env python3
"""Bounded paired LeWM PushT CEM iteration-cache benchmark.

This runner is intentionally compute-node-only.  It uses the official LeWM
policy preprocessing and PushT world only to capture two fixed, transformed
observation/goal batches; environment interaction is outside every timed path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from iteration_cache import IterationCacheModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing model work outside PBS")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return value is None or isinstance(value, str)


def copy_info(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: copy_info(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_info(item) for item in value]
    return copy.deepcopy(value)


class CaptureSolver:
    """Minimal solver used only to obtain policy-preprocessed info tensors."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] | None = None
        self._action_dim = 2
        self._n_envs = 1
        self._horizon = 5

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        """Implement the official solver binding used by ``World.set_policy``."""
        self._action_dim = int(math.prod(action_space.shape[1:])) * int(
            config.action_block
        )
        self._n_envs = int(n_envs)
        self._horizon = int(config.horizon)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def horizon(self) -> int:
        return self._horizon

    def __call__(self, info_dict: dict[str, Any], init_action: Any = None) -> dict[str, Any]:
        del init_action
        self.captured = copy_info(info_dict)
        import torch

        batch = len(next(value for value in info_dict.values() if hasattr(value, "shape")))
        return {
            "actions": torch.zeros(
                (batch, self._horizon, self._action_dim), dtype=torch.float32
            )
        }

    solve = __call__


class TraceCallback:
    output_key = "iteration_trace"

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.history.clear()

    def start_batch(self) -> None:
        return None

    def __call__(self, **kwargs: Any) -> None:
        def as_list(value: Any) -> Any:
            return value.detach().cpu().tolist() if hasattr(value, "detach") else value

        self.history.append(
            {
                "step": int(kwargs["step"]),
                "topk_values": as_list(kwargs["topk_vals"]),
                "topk_indices": as_list(kwargs["topk_inds"]),
                "mean": as_list(kwargs["mean"]),
                "variance": as_list(kwargs["var"]),
                "costs": as_list(kwargs["costs"]),
            }
        )

    def end_solve(self) -> None:
        return None


class TimedCostModel:
    """Preserve the model interface while timing every synchronized cost call."""

    def __init__(self, model: Any) -> None:
        import torch

        self.model = model
        self.events: list[tuple[Any, Any]] = []
        self._torch = torch

    def parameters(self):
        return self.model.parameters()

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Any) -> Any:
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        result = self.model.get_cost(info_dict, action_candidates)
        end.record()
        self.events.append((start, end))
        return result

    def section_ms(self) -> float:
        self._torch.cuda.synchronize()
        return float(sum(start.elapsed_time(end) for start, end in self.events))


def prepare_policy_info(
    stable_home: Path, n_observations: int
) -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    """Capture policy-preprocessed PushT info without timing environment work."""
    os.environ["STABLEWM_HOME"] = str(stable_home)
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf
    from torchvision.transforms import v2 as transforms

    from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

    world = swm.World("swm/PushT-v1", num_envs=n_observations, image_shape=(224, 224))
    world.reset(seed=42)
    transform = {
        "pixels": transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.Resize(size=224),
            ]
        ),
        "goal": transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.Resize(size=224),
            ]
        ),
    }
    capture = CaptureSolver()
    policy = WorldModelPolicy(
        solver=capture,
        config=PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process={},
        transform=transform,
    )
    world.set_policy(policy)
    policy.get_action(world.infos)
    if capture.captured is None:
        raise RuntimeError("WorldModelPolicy did not provide transformed info")
    captured = capture.captured
    observations = [
        (
            f"pusht_obs_{index:02d}",
            {key: value[index : index + 1] for key, value in captured.items()},
        )
        for index in range(n_observations)
    ]
    return observations, world.envs.action_space


def make_solver(
    model: Any,
    settings: dict[str, Any],
    seed: int,
    trace: TraceCallback,
    action_space: Any,
    config: Any,
) -> Any:
    try:
        from stable_worldmodel.solver import CEMSolver
    except ImportError:
        from stable_worldmodel.planning import CEMSolver
    solver = CEMSolver(
        model=model,
        callbacks=[trace],
        seed=seed,
        **settings,
    )
    solver.configure(action_space=action_space, n_envs=1, config=config)
    return solver


def one_call(
    model: Any,
    info: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
    cache: bool,
    action_space: Any,
    config: Any,
) -> dict[str, Any]:
    import torch

    trace = TraceCallback()
    model_for_solver = IterationCacheModel(model) if cache else model
    timed_model = TimedCostModel(model_for_solver)
    solver = make_solver(timed_model, settings, seed, trace, action_space, config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    full_start = time.perf_counter()
    working_info = copy_info(info)
    torch.cuda.synchronize()
    plan_start = time.perf_counter()
    output = solver.solve(working_info)
    torch.cuda.synchronize()
    plan_end = time.perf_counter()
    full_end = plan_end
    cost_ms = timed_model.section_ms()
    cache_ms = float(getattr(model_for_solver, "cache_setup_ms", 0.0))
    actions = output["actions"]
    record = {
        "full_planner_latency_ms": (full_end - full_start) * 1000.0,
        "plan_section_latency_ms": (plan_end - plan_start) * 1000.0,
        "preprocessing_latency_ms": (plan_start - full_start) * 1000.0,
        "cem_loop_latency_ms": cost_ms,
        "cache_setup_latency_ms": cache_ms,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "final_actions": actions.tolist(),
        "final_first_actions": actions[:, 0].tolist(),
        "costs": output.get("costs", []),
        "decision_trace": trace.history,
    }
    if not finite_tree(record):
        raise RuntimeError("non-finite result in planner record")
    return record


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_local_checkpoint(stable_home: Path) -> Any:
    """Load the official LeWM object checkpoint staged by the CPU job.

    The preparation step converts the official HF state dict into the object
    checkpoint consumed by the LeWM evaluation path.  Passing ``pusht/lewm``
    to ``stable_worldmodel.wm.utils.load_pretrained`` is incorrect here: that
    API treats the identifier as an HF repo and attempts an unauthorized
    network fetch instead of using the staged object.
    """
    import torch

    checkpoint = stable_home / "pusht" / "lewm_object.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"staged LeWM object checkpoint not found: {checkpoint}")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not hasattr(model, "get_cost"):
        raise TypeError(f"staged checkpoint is not a cost model: {checkpoint}")
    return model


def main() -> int:
    args = parse_args()
    require_compute_node()
    freeze = load_json(args.freeze)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["STABLEWM_HOME"] = str(args.stablewm_home.resolve())
    import torch
    from stable_worldmodel.policy import PlanConfig

    model = load_local_checkpoint(args.stablewm_home.resolve())
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    settings = dict(freeze["official_pusht"]["cem_solver"])
    plan_config = PlanConfig(**freeze["official_pusht"]["plan_config"])
    observations, action_space = prepare_policy_info(args.stablewm_home.resolve(), 2)

    records: list[dict[str, Any]] = []
    timing = freeze["timing"]
    for obs_index, (observation_id, info) in enumerate(observations):
        for warmup in range(int(timing["warmup_repeats"])):
            seed = 920000 + obs_index * 10000 + warmup
            order = ["baseline", "iteration_cache"] if (obs_index + warmup) % 2 == 0 else ["iteration_cache", "baseline"]
            for path in order:
                set_seed(seed)
                result = one_call(
                    model,
                    info,
                    settings,
                    seed,
                    path == "iteration_cache",
                    action_space,
                    plan_config,
                )
                records.append(
                    {
                        "observation_id": observation_id,
                        "planner_call_id": f"{observation_id}/warmup_{warmup:02d}",
                        "path": path,
                        "warmup": True,
                        "technical_repeat": warmup,
                        "seed": seed,
                        "path_order": order,
                        **result,
                    }
                )
        for repeat in range(int(timing["technical_repeats"])):
            seed = 1020000 + obs_index * 10000 + repeat
            order = ["baseline", "iteration_cache"] if (obs_index + repeat) % 2 == 0 else ["iteration_cache", "baseline"]
            for path in order:
                set_seed(seed)
                result = one_call(
                    model,
                    info,
                    settings,
                    seed,
                    path == "iteration_cache",
                    action_space,
                    plan_config,
                )
                records.append(
                    {
                        "observation_id": observation_id,
                        "planner_call_id": f"{observation_id}/technical_{repeat:02d}",
                        "path": path,
                        "warmup": False,
                        "technical_repeat": repeat,
                        "seed": seed,
                        "path_order": order,
                        **result,
                    }
                )

    summary = {
        "schema": "dino-wm-shared-prefix-cache.lewm-pusht-iteration-results",
        "schema_version": 1,
        "experiment_id": freeze["experiment_id"],
        "observations": [item[0] for item in observations],
        "warmup_repeats": timing["warmup_repeats"],
        "technical_repeats": timing["technical_repeats"],
        "cem_solver": freeze["official_pusht"]["cem_solver"],
        "measurement": freeze["timing"],
        "records": records,
        "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    (args.output / "system_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
