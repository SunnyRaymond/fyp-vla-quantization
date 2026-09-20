"""Bounded chained-CEM screen for an approximate factorized observation cache."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


N_OBSERVATIONS = 10
LATENCY_OBSERVATIONS = 2
CANDIDATES = 300
HORIZON = 5
TOPK = 30
OPT_STEPS = 10
WARMUPS = 5
TECHNICAL_REPEATS = 10
PATHS = ("baseline", "factorized_cache")
PATH_ORDER_SEED = 730000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-observations", type=int, default=N_OBSERVATIONS)
    parser.add_argument("--latency-observations", type=int, default=LATENCY_OBSERVATIONS)
    return parser.parse_args()


def _guard_compute_node() -> None:
    if not os.environ.get("PBS_JOBID", "").strip():
        raise RuntimeError("PBS_JOBID is empty; approximate screen requires PBS")
    host = socket.gethostname().lower()
    if any(token in host for token in ("login", "asp2a-login", "ntu-login")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, default=_json_default) + "\n")


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repeat_batch(value: Any, repeats: int) -> Any:
    return value[0:1].repeat((repeats,) + (1,) * (value.ndim - 1))


def _repeat_dict(values: Mapping[str, Any], repeats: int) -> Dict[str, Any]:
    return {key: _repeat_batch(value, repeats) for key, value in values.items()}


class _NullWandb:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FactorizedRolloutPatch:
    """One-call cache patch; every action-conditioned suffix remains in cache_core."""

    def __init__(self, model: Any, cache_core: Any):
        self.model = model
        self.cache_core = cache_core
        self.original = model.rollout
        self.prefix = None

    def __enter__(self) -> "_FactorizedRolloutPatch":
        self.model.rollout = self._rollout
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.model.rollout = self.original

    def _rollout(self, obs_0: Mapping[str, Any], act: Any) -> Any:
        if self.prefix is None:
            single = {key: value[:1] for key, value in obs_0.items()}
            self.prefix = self.cache_core.encode_and_repeat(
                self.model, single, mode="factorized_cache", repeats=1
            )
        expanded = self.cache_core.repeat_encoded_prefix(self.prefix, int(act.shape[0]))
        return self.cache_core.rollout_from_encoded_obs(self.model, expanded, act)


def _make_instrumented_class(cem_module: Any) -> Any:
    """Copy CEMPlanner.plan semantics while recording each post-update state."""

    import numpy as np
    import torch
    from einops import repeat
    from utils import move_to_device

    class InstrumentedCEMPlanner(cem_module.CEMPlanner):
        def plan(self, obs_0: Mapping[str, Any], obs_g: Mapping[str, Any], actions: Any = None):
            trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
            trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
            z_obs_g = self.wm.encode_obs(trans_obs_g)

            mu, sigma = self.init_mu_sigma(obs_0, actions)
            mu, sigma = mu.to(self.device), sigma.to(self.device)
            n_evals = mu.shape[0]
            trace: List[Dict[str, Any]] = []
            record_trace = bool(getattr(self, "record_trace", True))

            for iteration in range(self.opt_steps):
                losses = []
                topk_by_eval = []
                if record_trace:
                    before_mu = mu.detach().clone()
                    before_sigma = sigma.detach().clone()
                for traj in range(n_evals):
                    cur_trans_obs_0 = {
                        key: repeat(
                            arr[traj].unsqueeze(0),
                            "1 ... -> n ...",
                            n=self.num_samples,
                        )
                        for key, arr in trans_obs_0.items()
                    }
                    cur_z_obs_g = {
                        key: repeat(
                            arr[traj].unsqueeze(0),
                            "1 ... -> n ...",
                            n=self.num_samples,
                        )
                        for key, arr in z_obs_g.items()
                    }
                    action = (
                        torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                        * sigma[traj]
                        + mu[traj]
                    )
                    action[0] = mu[traj]
                    with torch.no_grad():
                        i_z_obses, _ = self.wm.rollout(obs_0=cur_trans_obs_0, act=action)
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                    topk_idx = torch.argsort(loss, stable=True)[: self.topk]
                    topk_action = action[topk_idx]
                    if record_trace:
                        losses.append(loss[topk_idx[0]].item())
                    if record_trace:
                        topk_by_eval.append(topk_idx.detach().cpu().tolist())
                    mu[traj] = topk_action.mean(dim=0)
                    sigma[traj] = topk_action.std(dim=0)
                if record_trace:
                    self.wandb_run.log(
                        {f"{self.logging_prefix}/loss": np.mean(losses), "step": iteration + 1}
                    )
                if record_trace:
                    trace.append(
                        {
                            "cem_iteration": iteration,
                            "topk_indices": topk_by_eval[0],
                            "mu_before": before_mu[0].detach().cpu().tolist(),
                            "sigma_before": before_sigma[0].detach().cpu().tolist(),
                            "mu_after": mu[0].detach().cpu().tolist(),
                            "sigma_after": sigma[0].detach().cpu().tolist(),
                        }
                    )
            self.last_trace = trace
            return mu, np.full(n_evals, np.inf)

    return InstrumentedCEMPlanner


def _runtime(root: Path) -> Dict[str, Any]:
    os.environ["DATASET_DIR"] = str(root / "data")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["PYTHONUNBUFFERED"] = "1"
    source = root / "source"
    sys.path.insert(0, str(source))

    import gym
    import hydra
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; refusing CPU approximate screen")
    checkpoint_dir = root / "checkpoints" / "outputs" / "wall_single"
    config_path = checkpoint_dir / "hydra.yaml"
    checkpoint_path = checkpoint_dir / "checkpoints" / "model_latest.pth"
    for required in (root / "data", config_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(required)
    cache_core = importlib.import_module("cache_core")
    for name in ("encode_and_repeat", "repeat_encoded_prefix", "rollout_from_encoded_obs"):
        if not callable(getattr(cache_core, name, None)):
            raise AttributeError(f"cache_core.{name} is required")
    model_cfg = OmegaConf.load(config_path)
    _, trajectory_datasets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    plan = importlib.import_module("plan")
    model = plan.load_model(
        checkpoint_path,
        model_cfg,
        model_cfg.num_action_repeat,
        device=torch.device("cuda:0"),
    )
    model.eval()
    model.decoder = None
    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    cem_module = importlib.import_module("planning.cem")
    planner_cls = _make_instrumented_class(cem_module)
    return {
        "torch": torch,
        "model": model,
        "model_cfg": model_cfg,
        "dset": trajectory_datasets["valid"],
        "env": env,
        "checkpoint": checkpoint_path,
        "cache_core": cache_core,
        "planner_cls": planner_cls,
    }


def _make_targets(runtime: Mapping[str, Any], count: int) -> List[Dict[str, Any]]:
    import numpy as np

    def to_numpy(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    dset = runtime["dset"]
    env = runtime["env"]
    targets = []
    for case_id in range(count):
        if case_id >= len(dset):
            raise RuntimeError(f"validation dataset has only {len(dset)} trajectories")
        _, _, _, env_info = dset[case_id]
        env.update_env(env_info)
        seed = 100000 + case_id
        init_state, goal_state = env.sample_random_init_goal_states(seed)
        obs_0, state_0 = env.prepare(seed, init_state)
        obs_g, state_g = env.prepare(seed, goal_state)
        obs_0 = {key: to_numpy(value)[None, None, ...] for key, value in obs_0.items()}
        obs_g = {key: to_numpy(value)[None, None, ...] for key, value in obs_g.items()}
        targets.append(
            {
                "observation_id": f"wall_case_{case_id:02d}",
                "env_seed": seed,
                "obs_0": obs_0,
                "obs_g": obs_g,
                "state_0": np.asarray(state_0).tolist(),
                "state_g": np.asarray(state_g).tolist(),
            }
        )
    return targets


def _preprocessor(runtime: Mapping[str, Any]) -> Any:
    from preprocessor import Preprocessor

    dset = runtime["dset"]
    return Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )


def _make_planner(
    runtime: Mapping[str, Any],
    preprocessor: Any,
    objective_fn: Any,
    *,
    record_trace: bool,
) -> Any:
    model = runtime["model"]
    action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
    planner = runtime["planner_cls"](
        horizon=HORIZON,
        topk=TOPK,
        num_samples=CANDIDATES,
        var_scale=1,
        opt_steps=OPT_STEPS,
        eval_every=1,
        wm=model,
        action_dim=action_dim,
        objective_fn=objective_fn,
        preprocessor=preprocessor,
        evaluator=None,
        wandb_run=_NullWandb(),
        logging_prefix="approx_screen",
        log_filename=None,
    )
    planner.record_trace = record_trace
    return planner


def _plan_once(
    runtime: Mapping[str, Any],
    preprocessor: Any,
    objective_fn: Any,
    target: Mapping[str, Any],
    path: str,
    seed: int,
    timed: bool = False,
) -> Dict[str, Any]:
    torch = runtime["torch"]
    model = runtime["model"]
    _set_seed(torch, seed)
    planner = _make_planner(
        runtime, preprocessor, objective_fn, record_trace=not timed
    )
    _sync(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(torch)
    started = time.monotonic()
    try:
        if path == "baseline":
            action, _ = planner.plan(target["obs_0"], target["obs_g"], actions=None)
        else:
            # Constructing this context inside the timed wrapper keeps the
            # factorized cache lifecycle inside one planner call.
            with _FactorizedRolloutPatch(model, runtime["cache_core"]):
                action, _ = planner.plan(target["obs_0"], target["obs_g"], actions=None)
    finally:
        _sync(torch)
    elapsed = time.monotonic() - started
    peak_memory_mib = (
        float(torch.cuda.max_memory_allocated() / 2**20)
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "action": action.detach().cpu() if not timed else None,
        "trace": getattr(planner, "last_trace", []),
        "seconds": elapsed,
        "latency_ms": elapsed * 1000.0,
        "peak_memory_mib": peak_memory_mib,
        "seed": seed,
        "timed": timed,
    }


def _max_abs_diff(left: Any, right: Any) -> float:
    import numpy as np

    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _l2_diff(left: Any, right: Any) -> float:
    import math
    import numpy as np

    left_values = np.asarray(left, dtype=np.float64).reshape(-1).tolist()
    right_values = np.asarray(right, dtype=np.float64).reshape(-1).tolist()
    return math.sqrt(
        sum((left_value - right_value) ** 2 for left_value, right_value in zip(left_values, right_values))
    )


def _decision_screen(
    runtime: Mapping[str, Any],
    preprocessor: Any,
    objective_fn: Any,
    targets: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    records = []
    for observation_index, target in enumerate(targets):
        noise_seed = 610000 + observation_index
        baseline = _plan_once(
            runtime, preprocessor, objective_fn, target, "baseline", noise_seed
        )
        factorized = _plan_once(
            runtime, preprocessor, objective_fn, target, "factorized_cache", noise_seed
        )
        rounds = []
        for iteration in range(OPT_STEPS):
            base_step = baseline["trace"][iteration]
            fac_step = factorized["trace"][iteration]
            base_topk = base_step["topk_indices"]
            fac_topk = fac_step["topk_indices"]
            overlap = len(set(base_topk).intersection(fac_topk))
            base_first = base_step["mu_after"][0]
            fac_first = fac_step["mu_after"][0]
            rounds.append(
                {
                    "round_index": iteration,
                    "noise_schedule_seed": noise_seed,
                    "baseline": {
                        "path": "baseline",
                        "noise_schedule_seed": noise_seed,
                        "input_mu": base_step["mu_before"],
                        "input_sigma": base_step["sigma_before"],
                        "topk_indices": base_topk,
                        "mu": base_step["mu_after"],
                        "sigma": base_step["sigma_after"],
                        "first_action": base_first,
                    },
                    "factorized_cache": {
                        "path": "factorized_cache",
                        "noise_schedule_seed": noise_seed,
                        "input_mu": fac_step["mu_before"],
                        "input_sigma": fac_step["sigma_before"],
                        "topk_indices": fac_topk,
                        "mu": fac_step["mu_after"],
                        "sigma": fac_step["sigma_after"],
                        "first_action": fac_first,
                    },
                    "topk_set_overlap": overlap / TOPK,
                    "topk_set_overlap_count": overlap,
                    "topk_set_diff": TOPK - overlap,
                    "mu_max_abs_diff": _max_abs_diff(base_step["mu_after"], fac_step["mu_after"]),
                    "sigma_max_abs_diff": _max_abs_diff(base_step["sigma_after"], fac_step["sigma_after"]),
                    "first_action_max_abs_diff": _max_abs_diff(base_first, fac_first),
                    "first_action_l2_diff": _l2_diff(base_first, fac_first),
                }
            )
        final_baseline = baseline["action"][0, 0].tolist()
        final_factorized = factorized["action"][0, 0].tolist()
        records.append(
            {
                "schema": "dino-wm-shared-prefix-cache-approx-decision-v2",
                "observation_id": target["observation_id"],
                "observation_seed": target["env_seed"],
                "candidate_count": CANDIDATES,
                "horizon": HORIZON,
                "topk": TOPK,
                "cem_opt_steps": OPT_STEPS,
                "noise_schedule_seed": noise_seed,
                "rounds": rounds,
                "final_first_action_baseline": final_baseline,
                "final_first_action_factorized_cache": final_factorized,
                "final_first_action_max_abs_diff": _max_abs_diff(final_baseline, final_factorized),
                "final_first_action_l2_diff": _l2_diff(final_baseline, final_factorized),
            }
        )
    return records


def _latency_screen(runtime: Mapping[str, Any], preprocessor: Any, objective_fn: Any, targets: Sequence[Mapping[str, Any]], output: Path) -> Dict[str, Any]:
    import statistics

    records = []
    for observation_index, target in enumerate(targets[:LATENCY_OBSERVATIONS]):
        for warmup, repeats in ((True, WARMUPS), (False, TECHNICAL_REPEATS)):
            for repeat_index in range(repeats):
                order = list(PATHS)
                random.Random(
                    PATH_ORDER_SEED
                    + observation_index * 100
                    + repeat_index
                    + int(warmup) * 10000
                ).shuffle(order)
                for order_index, path in enumerate(order):
                    seed = 710000 + observation_index * 1000 + repeat_index
                    result = _plan_once(runtime, preprocessor, objective_fn, target, path, seed, timed=True)
                    records.append(
                        {
                            "observation_id": target["observation_id"],
                            "planner_call_id": f"{target['observation_id']}_plan",
                            "path": path,
                            "warmup": warmup,
                            "technical_repeat": repeat_index,
                            "order_index": order_index,
                            "rng_seed": seed,
                            "noise_schedule_seed": seed,
                            "latency_ms": result["latency_ms"],
                            "peak_memory_mib": result["peak_memory_mib"],
                        }
                    )
    paired_reductions = []
    memory_ratios = []
    for target in targets[:LATENCY_OBSERVATIONS]:
        observation_id = target["observation_id"]
        baseline_rows = [
            row for row in records
            if row["observation_id"] == observation_id
            and not row["warmup"]
            and row["path"] == "baseline"
        ]
        factorized_rows = [
            row for row in records
            if row["observation_id"] == observation_id
            and not row["warmup"]
            and row["path"] == "factorized_cache"
        ]
        baseline_latency = statistics.median(row["latency_ms"] for row in baseline_rows)
        factorized_latency = statistics.median(row["latency_ms"] for row in factorized_rows)
        baseline_memory = max(row["peak_memory_mib"] for row in baseline_rows)
        factorized_memory = max(row["peak_memory_mib"] for row in factorized_rows)
        paired_reductions.append(1.0 - factorized_latency / baseline_latency)
        memory_ratios.append(
            factorized_memory / baseline_memory if baseline_memory else float("inf")
        )

    summary = {
        "schema": "dino-wm-shared-prefix-cache-approx-system-v1",
        "warmup_repeats": WARMUPS,
        "technical_repeats": TECHNICAL_REPEATS,
        "measurement_unit": "observation/planner call",
        "timing_boundary": "Complete planner call from entry through returned first action",
        "includes_preprocessing": True,
        "includes_environment_interaction": False,
        "device_synchronization": "CUDA synchronize immediately before timing and immediately after timing",
        "path_order": "seeded interleaved order per observation/repeat",
        "path_order_policy": "interleaved_seeded",
        "path_order_seed": PATH_ORDER_SEED,
        "paths": list(PATHS),
        "records": records,
        "record_count": len(records),
        "latency_observations": [target["observation_id"] for target in targets[:LATENCY_OBSERVATIONS]],
        "latency_subset_observation_ids": [
            target["observation_id"] for target in targets[:LATENCY_OBSERVATIONS]
        ],
        "latency_statistic": "per-call technical-repeat median; report median paired reduction",
        "memory_statistic": "maximum technical-repeat peak_memory_mib ratio per observation",
        "paired_median_reduction": {"factorized_cache": statistics.median(paired_reductions)},
        "max_peak_memory_ratio": {"factorized_cache": max(memory_ratios)},
    }
    _write_json(output / "system_summary.json", summary)
    return summary


def _run(args: argparse.Namespace) -> int:
    from planning.objectives import create_objective_fn

    _guard_compute_node()
    if args.n_observations != N_OBSERVATIONS or args.latency_observations != LATENCY_OBSERVATIONS:
        raise ValueError("frozen approximate screen requires 10 observations and 2 latency observations")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(args.root.resolve())
    preprocessor = _preprocessor(runtime)
    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    targets = _make_targets(runtime, N_OBSERVATIONS)
    decision_records = _decision_screen(runtime, preprocessor, objective_fn, targets)
    _write_jsonl(output / "decision.jsonl", decision_records)
    _latency_screen(runtime, preprocessor, objective_fn, targets, output)
    _write_json(
        output / "summary.json",
        {
            "schema": "dino-wm-shared-prefix-cache-approx-screen-v1",
            "status": "screen_complete",
            "observations": N_OBSERVATIONS,
            "candidate_count": CANDIDATES,
            "horizon": HORIZON,
            "topk": TOPK,
            "opt_steps": OPT_STEPS,
            "checkpoint": str(runtime["checkpoint"]),
            "gpu": runtime["torch"].cuda.get_device_name(),
            "torch": runtime["torch"].__version__,
            "decision_file": "decision.jsonl",
            "system_file": "system_summary.json",
            "closed_loop": "NOT_RUN",
        },
    )
    runtime["env"].close()
    return 0


def main() -> int:
    try:
        return _run(_parse_args())
    except Exception as exc:
        if "--output" in sys.argv:
            try:
                output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
                output.mkdir(parents=True, exist_ok=True)
                _write_json(output / "summary.json", {"status": "failed", "error": repr(exc)})
            except Exception:
                pass
        print(f"APPROX_SCREEN_FAILED: {exc!r}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
