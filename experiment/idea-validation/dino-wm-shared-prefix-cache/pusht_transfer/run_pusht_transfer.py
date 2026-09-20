#!/usr/bin/env python3
"""Run the frozen paired official-CEM versus factorized-prefix PushT transfer.

The runner is intentionally compute-node-only.  It requires already staged
official PushT assets and never downloads or prepares them.  The timed unit is
one direct CEMPlanner.plan call; the official MPC outer loop and environment
action execution are deliberately excluded from this bounded transfer.
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
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM root")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="job-specific staged asset root; defaults to --root for pre-staged assets",
    )
    parser.add_argument("--output", type=Path, required=True, help="artifact directory")
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path(__file__).with_name("PUSHT_TRANSFER_FREEZE.json"),
    )
    return parser.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing model loading outside a PBS allocation")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_pusht_assets(asset_root: Path, freeze: dict[str, Any]) -> None:
    assets = freeze["assets"]
    required = [
        asset_root / assets["checkpoint_config"],
        asset_root / assets["checkpoint_file"],
    ]
    for split in assets["dataset_splits"]:
        split_root = asset_root / assets["dataset_root"] / split
        required.extend(split_root / name for name in assets["required_dataset_files"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "PushT assets are not fully staged; failing closed without download/extraction: "
            + ", ".join(missing)
        )


class TraceRun:
    """Small W&B-compatible sink retaining the official per-step loss trace."""

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


class TraceCEMPlanner:
    """Official CEM loop with either original rollout or factorized prefix reuse."""

    def __init__(self, official_cls: type, *, cache_mode: str, trace: TraceRun, **kwargs: Any) -> None:
        self._planner = official_cls(**kwargs)
        self._cache_mode = cache_mode

    @property
    def wandb_run(self) -> TraceRun:
        return self._planner.wandb_run

    @property
    def decision_trace(self) -> list[dict[str, Any]]:
        return self._decision_trace

    def plan(self, obs_0: dict[str, Any], obs_g: dict[str, Any], actions: Any = None):
        import torch
        import numpy as np
        from einops import repeat
        from cache_core import rollout_with_cache
        from utils import move_to_device

        planner = self._planner
        trans_obs_0 = move_to_device(
            planner.preprocessor.transform_obs(obs_0), planner.device
        )
        trans_obs_g = move_to_device(
            planner.preprocessor.transform_obs(obs_g), planner.device
        )
        z_obs_g = planner.wm.encode_obs(trans_obs_g)

        factorized_prefixes: list[dict[str, torch.Tensor]] | None = None
        if self._cache_mode == "factorized_cache":
            factorized_prefixes = []
            with torch.no_grad():
                for traj in range(obs_0["visual"].shape[0]):
                    one_obs = {
                        key: value[traj : traj + 1]
                        for key, value in trans_obs_0.items()
                    }
                    prefix = planner.wm.encode_obs(one_obs)
                    factorized_prefixes.append(
                        {key: value.detach() for key, value in prefix.items()}
                    )
        elif self._cache_mode != "baseline":
            raise ValueError(f"unknown cache mode: {self._cache_mode}")

        mu, sigma = planner.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(planner.device), sigma.to(planner.device)
        n_evals = mu.shape[0]
        self._decision_trace = []

        for iteration in range(planner.opt_steps):
            losses: list[float] = []
            iteration_trace: list[dict[str, Any]] = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=planner.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=planner.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(planner.num_samples, planner.horizon, planner.action_dim)
                    .to(planner.device)
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]
                with torch.no_grad():
                    if self._cache_mode == "baseline":
                        i_z_obses, _ = planner.wm.rollout(
                            obs_0=cur_trans_obs_0,
                            act=action,
                        )
                    else:
                        assert factorized_prefixes is not None
                        i_z_obses, _ = rollout_with_cache(
                            planner.wm,
                            cur_trans_obs_0,
                            action,
                            mode="factorized_cache",
                            cached_prefix=factorized_prefixes[traj],
                        )

                loss = planner.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: planner.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)
                iteration_trace.append(
                    {
                        "elite_indices": topk_idx.detach().cpu().tolist(),
                        "elite_losses": loss[topk_idx].detach().cpu().tolist(),
                        "mu": mu[traj].detach().cpu().tolist(),
                        "sigma": sigma[traj].detach().cpu().tolist(),
                        "first_action": mu[traj, 0].detach().cpu().tolist(),
                    }
                )

            planner.wandb_run.log(
                {f"{planner.logging_prefix}/loss": np.mean(losses), "step": iteration + 1}
            )
            if planner.evaluator is not None and iteration % planner.eval_every == 0:
                raise RuntimeError("timed transfer must use evaluator=None and no closed-loop interaction")
            self._decision_trace.append({"trajectories": iteration_trace})

        return mu, [float("inf")] * n_evals


def _set_seed(value: int) -> None:
    import numpy as np
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _timed_call(
    planner: TraceCEMPlanner,
    obs_0: dict[str, Any],
    obs_g: dict[str, Any],
    seed_value: int,
) -> dict[str, Any]:
    import torch

    _set_seed(seed_value)
    planner.wandb_run.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    actions, _ = planner.plan(obs_0=obs_0, obs_g=obs_g, actions=None)
    torch.cuda.synchronize()
    result = {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "final_actions": actions.detach().cpu().tolist(),
        "final_first_actions": actions.detach()[:, 0].cpu().tolist(),
        "losses": list(planner.wandb_run.losses),
        "decision_trace": planner.decision_trace,
    }
    if not all(
        _finite_tree(result[field])
        for field in ("final_actions", "final_first_actions", "losses", "decision_trace")
    ):
        raise RuntimeError("planner produced a non-finite decision trace")
    return result


def _planner_kwargs(
    model: Any,
    preprocessor: Any,
    objective_fn: Any,
    trace: TraceRun,
    settings: dict[str, Any],
    action_dim: int,
    cache_mode: str,
) -> dict[str, Any]:
    return {
        "cache_mode": cache_mode,
        "trace": trace,
        "horizon": settings["horizon"],
        "topk": settings["topk"],
        "num_samples": settings["num_samples"],
        "var_scale": settings["var_scale"],
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


def _check_official_config(source: Path, freeze: dict[str, Any]) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(source / "conf" / "plan_pusht.yaml")
    official = freeze["official_pushT"]
    if cfg.planner._target_ != "planning.mpc.MPCPlanner":
        raise ValueError("official PushT config is no longer the expected MPC mainline")
    if cfg.planner.sub_planner.target != "planning.cem.CEMPlanner":
        raise ValueError("official PushT config is no longer CEM-based")
    if cfg.planner.max_iter is not None or int(cfg.planner.n_taken_actions) != official["mpc_n_taken_actions"]:
        raise ValueError("frozen MPC outer settings do not match plan_pusht.yaml")
    for key, expected in official["cem_settings"].items():
        actual = cfg.planner.sub_planner[key]
        if float(actual) != float(expected):
            raise ValueError(f"frozen CEM setting {key} does not match plan_pusht.yaml")
    if int(cfg.seed) != official["seed"] or cfg.goal_source != official["goal_source"]:
        raise ValueError("frozen PushT target protocol does not match plan_pusht.yaml")
    if int(cfg.goal_H) != official["goal_H"]:
        raise ValueError("frozen PushT goal_H does not match plan_pusht.yaml")


def main() -> int:
    args = _args()
    _require_compute_node()
    root = args.root.resolve()
    asset_root = (args.asset_root or root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    _require_pusht_assets(asset_root, freeze)

    source = root / "source"
    idea_root = root / "idea-validation" / "dino-wm-shared-prefix-cache"
    sys.path.insert(0, str(idea_root))
    sys.path.insert(0, str(source))
    _check_official_config(source, freeze)

    os.environ["DATASET_DIR"] = str(asset_root / "data")
    os.environ["WANDB_MODE"] = "disabled"

    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    import plan
    from env.pusht.pusht_wrapper import PushTWrapper
    from env.serial_vector_env import SerialVectorEnv
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing accidental CPU execution")

    model_path = asset_root / "checkpoints" / "outputs" / freeze["model"]["model_name"]
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    if model_cfg.env.name != "pusht":
        raise ValueError(f"checkpoint config env must be pusht, got {model_cfg.env.name!r}")
    seed_value = int(freeze["official_pushT"]["seed"])
    seed_value_for_model = seed_value
    _, trajectory_datasets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dataset = trajectory_datasets["valid"]
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    model = plan.load_model(
        model_ckpt,
        model_cfg,
        model_cfg.num_action_repeat,
        device=torch.device("cuda:0"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    env_kwargs = OmegaConf.to_container(model_cfg.env.kwargs, resolve=True)
    environment = SerialVectorEnv(
        [PushTWrapper(**env_kwargs) for _ in range(freeze["protocol"]["n_observations"])]
    )

    settings = freeze["official_pushT"]["cem_settings"]
    cfg = {
        "seed": seed_value_for_model,
        "n_evals": int(freeze["protocol"]["n_observations"]),
        "goal_source": freeze["official_pushT"]["goal_source"],
        "goal_H": int(freeze["official_pushT"]["goal_H"]),
        "n_plot_samples": 0,
        "debug_dset_init": False,
        "objective": {
            "_target_": "planning.objectives.create_objective_fn",
            **freeze["official_pushT"]["objective"],
        },
        "planner": {
            "_target_": "planning.cem.CEMPlanner",
            **settings,
            "name": "cem",
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
    preprocessor = workspace.data_preprocessor
    action_dim = workspace.action_dim

    observations = [
        (
            freeze["protocol"]["observation_ids"][index],
            {key: value[index : index + 1] for key, value in workspace.obs_0.items()},
            {key: value[index : index + 1] for key, value in workspace.obs_g.items()},
        )
        for index in range(int(freeze["protocol"]["n_observations"]))
    ]

    (output / "observation_manifest.json").write_text(
        json.dumps(
            {
                "schema": "dino-wm-shared-prefix-cache.pusht-observation-manifest",
                "source": "PlanWorkspace.prepare_targets with official PushT dset protocol",
                "observation_ids": [item[0] for item in observations],
                "eval_seeds": [int(value) for value in workspace.eval_seed],
                "obs_0_shapes": {key: list(value.shape) for key, value in workspace.obs_0.items()},
                "obs_g_shapes": {key: list(value.shape) for key, value in workspace.obs_g.items()},
                "frameskip": int(model_cfg.frameskip),
                "action_dim": int(action_dim),
                "closed_loop": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    timing = freeze["timing"]
    warmups = int(timing["warmup_repeats"])
    technical = int(timing["technical_repeats"])
    from planning.cem import CEMPlanner as OfficialCEMPlanner

    base_seed = 920000
    for obs_index, (observation_id, obs_0, obs_g) in enumerate(observations):
        for warmup in range(warmups):
            pair_seed = base_seed + obs_index * 10000 + warmup
            order = (
                ["baseline", "factorized_cache"]
                if (obs_index + warmup) % 2 == 0
                else ["factorized_cache", "baseline"]
            )
            for path in order:
                trace = TraceRun()
                kwargs = _planner_kwargs(
                    model, preprocessor, objective_fn, trace, settings, action_dim, path
                )
                planner = TraceCEMPlanner(OfficialCEMPlanner, **kwargs)
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
            order = (
                ["baseline", "factorized_cache"]
                if (obs_index + repeat) % 2 == 0
                else ["factorized_cache", "baseline"]
            )
            for path in order:
                trace = TraceRun()
                kwargs = _planner_kwargs(
                    model, preprocessor, objective_fn, trace, settings, action_dim, path
                )
                planner = TraceCEMPlanner(OfficialCEMPlanner, **kwargs)
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
        "schema": "dino-wm-shared-prefix-cache.pusht-transfer-results",
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
        "cem_settings": settings,
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
    (output / "PUSHT_TRANSFER_COMPLETE").touch()
    print(json.dumps({"records": len(records), "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
