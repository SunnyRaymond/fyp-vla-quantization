#!/usr/bin/env python3
"""Run paired official-GD versus cached-prefix GD calls on two Wall observations.

This file is intentionally a compute-node runner.  It loads the already staged
official assets, generates the two observations through the official Wall target
preparation path, and records complete planner-call timings and decision traces.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM reproduction root")
    parser.add_argument("--output", type=Path, required=True, help="artifact directory")
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path(__file__).with_name("GD_PORTABILITY_FREEZE.json"),
    )
    return parser.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing to load the model outside a PBS allocation")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_tree(value: Any) -> bool:
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    raise TypeError(f"cannot serialize {type(value).__name__}")


class TraceRun:
    """Capture the official planner's scalar logging without writing logs.json."""

    def __init__(self) -> None:
        self.losses: list[float] = []

    def clear(self) -> None:
        self.losses.clear()

    def log(self, values: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        for key, value in values.items():
            if key.endswith("/loss"):
                scalar = value.item() if hasattr(value, "item") else value
                self.losses.append(float(scalar))


class CachedGDPlanner:
    """Official GDPlanner logic with one detached start prefix per plan call."""

    def __init__(self, official_cls: type, **kwargs: Any) -> None:
        self._planner = official_cls(**kwargs)
        self._wm = kwargs["wm"]

    @property
    def trace(self) -> TraceRun:
        return self._planner.wandb_run

    @property
    def wandb_run(self) -> TraceRun:
        return self._planner.wandb_run

    def plan(self, obs_0: dict[str, Any], obs_g: dict[str, Any], actions: Any = None):
        import torch
        from utils import move_to_device

        planner = self._planner
        trans_obs_0 = move_to_device(
            planner.preprocessor.transform_obs(obs_0), planner.device
        )
        trans_obs_g = move_to_device(
            planner.preprocessor.transform_obs(obs_g), planner.device
        )
        z_obs_g = self._wm.encode_obs(trans_obs_g)
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        # The detached prefix has no action dependency.  The action-conditioned
        # suffix below remains differentiable with respect to ``actions``.
        with torch.no_grad():
            z_obs_0 = self._wm.encode_obs(trans_obs_0)
        z_obs_0 = {key: value.detach() for key, value in z_obs_0.items()}

        actions = planner.init_actions(obs_0, actions).to(planner.device)
        actions.requires_grad = True
        optimizer = planner.get_action_optimizer(actions)
        n_evals = actions.shape[0]

        for i in range(planner.opt_steps):
            optimizer.zero_grad()
            from cache_core import rollout_with_cache

            i_z_obses, _ = rollout_with_cache(
                self._wm,
                trans_obs_0,
                actions,
                mode="iteration_cache",
                cached_prefix=z_obs_0,
            )
            loss = planner.objective_fn(i_z_obses, z_obs_g_detached)
            total_loss = loss.mean() * n_evals
            total_loss.backward()
            with torch.no_grad():
                actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                actions_new += torch.randn_like(actions_new) * planner.action_noise
                actions.copy_(actions_new)

            planner.wandb_run.log(
                {f"{planner.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )

            # This runner deliberately supplies evaluator=None.  Keeping this
            # guard mirrors the official planner while preventing environment
            # interaction from entering the timing boundary.
            if planner.evaluator is not None and i % planner.eval_every == 0:
                raise RuntimeError("cached GD runner unexpectedly received an evaluator")

        return actions, [float("inf")] * n_evals


def _set_seed(value: int) -> None:
    import numpy as np
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _timed_call(planner: Any, obs_0: dict[str, Any], obs_g: dict[str, Any], seed_value: int) -> dict[str, Any]:
    import torch

    _set_seed(seed_value)
    planner.wandb_run.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    actions, _ = planner.plan(obs_0=obs_0, obs_g=obs_g, actions=None)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "latency_ms": elapsed_ms,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "final_actions": actions.detach().cpu().tolist(),
        "losses": list(planner.wandb_run.losses),
    }
    if not _finite_tree(result["final_actions"]) or not _finite_tree(result["losses"]):
        raise RuntimeError("planner produced a non-finite decision trace")
    return result


def _planner_kwargs(model: Any, preprocessor: Any, objective_fn: Any, trace: TraceRun, settings: dict[str, Any], action_dim: int) -> dict[str, Any]:
    return {
        "horizon": settings["horizon"],
        "action_noise": settings["action_noise"],
        "sample_type": settings["sample_type"],
        "lr": settings["lr"],
        "opt_steps": settings["opt_steps"],
        "eval_every": settings["eval_every"],
        "wm": model,
        "action_dim": action_dim,
        "objective_fn": objective_fn,
        "preprocessor": preprocessor,
        "evaluator": None,
        "wandb_run": trace,
        "logging_prefix": "plan_0",
        "log_filename": None,
    }


def main() -> int:
    args = _args()
    _require_compute_node()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    protocol = freeze["wall_protocol"]
    timing = freeze["timing"]
    settings = freeze["gd_settings"]

    if protocol["n_observations"] != 2:
        raise ValueError("this runner is frozen to exactly two observations")
    if timing["paths"] != ["baseline", "iteration_cache"]:
        raise ValueError("unexpected frozen path set")

    os.environ["DATASET_DIR"] = str(root / "data")
    os.environ["WANDB_MODE"] = "disabled"
    source = root / "source"
    idea_root = root / "idea-validation" / "dino-wm-shared-prefix-cache"
    sys.path.insert(0, str(idea_root))
    sys.path.insert(0, str(source))

    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    import plan
    from env.serial_vector_env import SerialVectorEnv
    from planning.gd import GDPlanner
    from utils import seed

    torch.set_num_threads(8)
    torch.set_grad_enabled(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing accidental CPU execution")

    model_path = root / "checkpoints" / "outputs" / freeze["model"]["model_name"]
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    seed(int(protocol["seed"]))
    _, datasets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dataset = datasets["valid"]
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    if not model_ckpt.is_file():
        raise FileNotFoundError(model_ckpt)
    model = plan.load_model(
        model_ckpt,
        model_cfg,
        model_cfg.num_action_repeat,
        device=torch.device("cuda:0"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    environment = SerialVectorEnv(
        [
            __import__("gym").make(
                model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
            )
            for _ in range(protocol["n_observations"])
        ]
    )

    # PlanWorkspace owns the official Wall random_state target preparation.  We
    # use it only to freeze the two input pairs, then call the planners directly
    # with evaluator=None so no environment interaction is timed.
    cfg = {
        "seed": int(protocol["seed"]),
        "n_evals": int(protocol["n_observations"]),
        "goal_source": protocol["goal_source"],
        "goal_H": int(protocol["goal_H"]),
        "n_plot_samples": 0,
        "debug_dset_init": False,
        "objective": {
            "_target_": "planning.objectives.create_objective_fn",
            "alpha": protocol["objective"]["alpha"],
            "base": protocol["objective"]["base"],
            "mode": protocol["objective"]["mode"],
        },
        "planner": {
            "_target_": "planning.gd.GDPlanner",
            **settings,
            "name": "gd",
        },
        "saved_folder": str(output),
        "wandb_logging": False,
    }
    os.chdir(output)
    workspace = plan.PlanWorkspace(
        cfg_dict=cfg,
        wm=model,
        dset=dataset,
        env=environment,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=None,
    )
    objective_fn = workspace.planner.objective_fn
    action_dim = workspace.action_dim
    preprocessor = workspace.data_preprocessor

    observations: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for index in range(protocol["n_observations"]):
        observations.append(
            (
                f"wall_obs_{index:02d}",
                {key: value[index : index + 1] for key, value in workspace.obs_0.items()},
                {key: value[index : index + 1] for key, value in workspace.obs_g.items()},
            )
        )
    (output / "observation_manifest.json").write_text(
        json.dumps(
            {
                "observation_ids": [item[0] for item in observations],
                "source": "PlanWorkspace.prepare_targets with official Wall random_state protocol",
                "eval_seeds": [int(value) for value in workspace.eval_seed],
                "obs_0_shapes": {
                    key: list(value.shape) for key, value in workspace.obs_0.items()
                },
                "obs_g_shapes": {
                    key: list(value.shape) for key, value in workspace.obs_g.items()
                },
                "frameskip": int(model_cfg.frameskip),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    warmups = int(timing["warmup_repeats"])
    technical = int(timing["technical_repeats"])
    base_seed = 910000
    for obs_index, (observation_id, obs_0, obs_g) in enumerate(observations):
        for warmup in range(warmups):
            pair_seed = base_seed + obs_index * 10000 + warmup
            order = ["baseline", "iteration_cache"] if (obs_index + warmup) % 2 == 0 else ["iteration_cache", "baseline"]
            for path in order:
                trace = TraceRun()
                kwargs = _planner_kwargs(model, preprocessor, objective_fn, trace, settings, action_dim)
                planner = GDPlanner(**kwargs) if path == "baseline" else CachedGDPlanner(GDPlanner, **kwargs)
                result = _timed_call(planner, obs_0, obs_g, pair_seed)
                records.append(
                    {
                        "observation_id": observation_id,
                        "planner_call_id": f"{observation_id}/warmup_{warmup:02d}",
                        "path": path,
                        "warmup": True,
                        "technical_repeat": warmup,
                        "seed": pair_seed,
                        "path_order": order,
                        **result,
                    }
                )
        for repeat in range(technical):
            pair_seed = base_seed + 100000 + obs_index * 10000 + repeat
            order = ["baseline", "iteration_cache"] if (obs_index + repeat) % 2 == 0 else ["iteration_cache", "baseline"]
            for path in order:
                trace = TraceRun()
                kwargs = _planner_kwargs(model, preprocessor, objective_fn, trace, settings, action_dim)
                planner = GDPlanner(**kwargs) if path == "baseline" else CachedGDPlanner(GDPlanner, **kwargs)
                result = _timed_call(planner, obs_0, obs_g, pair_seed)
                records.append(
                    {
                        "observation_id": observation_id,
                        "planner_call_id": f"{observation_id}/technical_{repeat:02d}",
                        "path": path,
                        "warmup": False,
                        "technical_repeat": repeat,
                        "seed": pair_seed,
                        "path_order": order,
                        **result,
                    }
                )

    summary = {
        "schema": "dino-wm-shared-prefix-cache.gd-portability-results",
        "schema_version": 1,
        "experiment_id": freeze["experiment_id"],
        "freeze_file": str(args.freeze.resolve()),
        "model": freeze["model"],
        "source_commit": freeze["source"]["source_commit"],
        "measurement_unit": timing["measurement_unit"],
        "timing_boundary": timing["timing_boundary"],
        "includes_preprocessing": timing["includes_preprocessing"],
        "includes_environment_interaction": timing["includes_environment_interaction"],
        "device_synchronization": timing["device_synchronization"],
        "path_order_policy": timing["path_order_policy"],
        "warmup_repeats": warmups,
        "technical_repeats": technical,
        "observations": [item[0] for item in observations],
        "gd_settings": settings,
        "records": records,
        "runtime": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "numpy": np.__version__,
        },
    }
    (output / "system_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    (output / "GD_PORTABILITY_COMPLETE").touch()
    print(json.dumps({"records": len(records), "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
