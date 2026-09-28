"""Instrument the official Fast-LeWM PushT evaluator inside a PBS allocation."""
import argparse
import importlib.util
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_allocation():
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    host = platform.node().split(".")[0].lower()
    if not os.environ.get("PBS_JOBID") or not nodefile.is_file():
        raise RuntimeError("A PBS compute allocation is required")
    nodes = {line.strip().split(".")[0].lower() for line in nodefile.read_text().splitlines()}
    if host not in nodes or any(word in host for word in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing unallocated/control host {host}")
    return host


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "stage", "out", "freeze"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    host = require_allocation()
    out = args.out.resolve(strict=True)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    prepared = json.loads((args.root / "prepared.json").read_text())
    if prepared["status"] != "PASS":
        raise RuntimeError("CPU preparation has not passed")
    source = args.root / "upstream"
    stablewm_root = Path(prepared.get("stablewm_root", args.stage / "stable-worldmodel"))
    sys.path[:0] = [str(source), str(stablewm_root)]
    identity = json.loads((args.root / "source_identity.json").read_text())
    if identity["source_sha"] != freeze["fast_source_commit"] or prepared["source_sha"] != freeze["fast_source_commit"]:
        raise RuntimeError("Fast-LeWM source commit differs from the freeze")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"

    cache = out / "cache"
    (cache / "datasets").mkdir(parents=True)
    dataset = args.stage / "stablewm_home" / "pusht_expert_train.h5"
    checkpoint = args.root / "checkpoints" / "Fast-lewm_pusht_object.ckpt"
    (cache / "datasets" / dataset.name).symlink_to(dataset)
    # Official parser removes _object.ckpt and AutoCostModel adds it back.
    (cache / checkpoint.name).symlink_to(checkpoint)
    os.environ["STABLEWM_HOME"] = str(cache)
    # Imports, model/data access and numerical work are behind the allocation guard.
    import hdf5plugin  # noqa: F401: register compressed HDF5 filters
    import hydra
    from hydra.core.hydra_config import HydraConfig
    import numpy as np
    import stable_worldmodel as swm
    from stable_worldmodel.solver import CEMSolver
    import torch
    from omegaconf import OmegaConf
    from jepa import JEPA
    if not Path(swm.__file__).resolve().is_relative_to(stablewm_root.resolve()):
        raise RuntimeError("The isolated official backend was not imported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reproduction")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(source / "config" / "eval")):
        cfg = hydra.compose(config_name="pusht", return_hydra_config=True)
    cfg.hydra.runtime.output_dir = str(out / "official")
    HydraConfig.instance().set_config(cfg)
    cfg.cache_dir = str(cache)
    cfg.eval.dataset_path = str(cache / "datasets" / dataset.name)
    cfg.eval.ckpt_path = str(cache / checkpoint.name)
    expected = freeze["settings"]
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise RuntimeError(f"Official setting differs from freeze: {key}")
    app_cfg = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])
    write_json(out / "resolved_config.json", OmegaConf.to_container(app_cfg, resolve=True, throw_on_missing=False))

    spec = importlib.util.spec_from_file_location("fastlewm_official_eval", source / "eval.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    worlds = []
    results = {}
    solve_rows = []
    cost_shapes = []
    cost_calls = 0
    cost_candidate_scores = 0
    nonfinite_cost_calls = 0
    original_init = swm.World.__init__
    original_set_policy = swm.World.set_policy
    solver_cls = CEMSolver
    original_solve = solver_cls.solve
    original_cost = JEPA.get_cost
    original_fast_path = official._install_fast_buffered_action_path
    original_task_labels = official._build_task_labels
    selected_rows = []
    def task_labels(*pos, **kwargs):
        rows = kwargs.get("row_indices", pos[1] if len(pos) > 1 else None)
        selected_rows.extend(int(x) for x in rows)
        return original_task_labels(*pos, **kwargs)
    official._build_task_labels = task_labels
    buffered_fast_path = {"enabled": False}
    def install_fast_path(policy):
        result = original_fast_path(policy)
        buffered_fast_path["enabled"] = result is not None
        return result
    official._install_fast_buffered_action_path = install_fast_path

    def world_init(self, *pos, **kwargs):
        original_init(self, *pos, **kwargs)
        worlds.append(self)
        write_json(out / "world_settings.json", kwargs)
        write_json(out / "resolved_config.json", OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"]),
            resolve=True, throw_on_missing=False))

    def set_policy(self, policy):
        solver = policy.solver
        plan = policy.cfg
        plan_values = {key: getattr(plan, key) for key in inspect.signature(swm.PlanConfig).parameters if hasattr(plan, key)}
        write_json(out / "runtime_contract.json", {
            "plan_config": plan_values,
            "predicted_primitive_actions": plan.horizon * plan.action_block,
            "buffered_primitive_actions": plan.receding_horizon * plan.action_block,
            "solver": {key: getattr(solver, key) for key in ("num_samples", "topk", "n_steps", "batch_size", "n_envs") if hasattr(solver, key)},
            "model_class": type(solver.model).__module__ + "." + type(solver.model).__name__,
            "stable_worldmodel_file": swm.__file__,
            "python": sys.executable,
            "torch": torch.__version__,
        })
        if plan.horizon * plan.action_block != 25 or plan.receding_horizon * plan.action_block != 25:
            raise RuntimeError("Official packed-action planning/execution does not cover 25 primitive steps")
        return original_set_policy(self, policy)

    def get_cost(self, info, actions):
        nonlocal cost_calls, cost_candidate_scores, nonfinite_cost_calls
        result = original_cost(self, info, actions)
        cost_calls += 1
        cost_candidate_scores += actions.shape[0] * actions.shape[1]
        # Check every call; this modest synchronization is explicitly part of instrumented timings.
        if not bool(torch.isfinite(result).all()):
            nonfinite_cost_calls += 1
            raise RuntimeError("Non-finite candidate costs")
        if len(cost_shapes) < 3:
            cost_shapes.append({"actions": list(actions.shape), "costs": list(result.shape),
                                "predicted_emb": list(info["predicted_emb"].shape) if "predicted_emb" in info else None})
            write_json(out / "interface_samples.json", cost_shapes)
        return result

    def solve(self, info_dict, *pos, **kwargs):
        before_calls, before_scores = cost_calls, cost_candidate_scores
        active = int(info_dict["pixels"].shape[0])
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_solve(self, info_dict, *pos, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(result["actions"]).all()):
            raise RuntimeError("Non-finite solver action")
        row = {"ordinal": len(solve_rows), "active_environments": active,
               "wall_seconds_synchronized": elapsed, "candidate_cost_calls": cost_calls - before_calls,
               "candidate_scores": cost_candidate_scores - before_scores,
               "action_shape": list(result["actions"].shape)}
        solve_rows.append(row)
        with (out / "solve_times.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return result

    original_evaluate = swm.World.evaluate_from_dataset
    def evaluate(self, data, *, start_steps, goal_offset_steps, eval_budget, episodes_idx, callables=None, video_path=None):
        tasks = [{"task_index": i, "row_index": selected_rows[i], "episode_idx": int(ep), "start_step": int(step)}
                 for i, (ep, step) in enumerate(zip(episodes_idx, start_steps, strict=True))]
        write_json(out / "selected_tasks.json", {"seed": int(cfg.seed), "tasks": tasks,
                   "goal_offset_steps": int(goal_offset_steps), "eval_budget": int(eval_budget),
                   "unique_source_episodes": len(set(episodes_idx))})
        started = time.perf_counter()
        metrics = original_evaluate(self, data, start_steps=start_steps, goal_offset_steps=goal_offset_steps,
                  eval_budget=eval_budget, episodes_idx=episodes_idx, callables=callables, save_video=False, video_path=None)
        torch.cuda.synchronize()
        results.update(metrics=jsonable(metrics, torch, np), evaluation_seconds=time.perf_counter()-started)
        write_json(out / "raw_metrics.json", results)
        return metrics

    def loss_metrics(trace, *pos, **kwargs):
        # Upstream np.stack fails when successful envs leave the active set. Preserve
        # each solve's costs without pretending different active envs share columns.
        values = [np.asarray(row, dtype=float).tolist() for row in trace]
        write_json(out / "cem_optimized_cost_trace.json", values)
        return {"source": "cem_optimized_latent_cost", "replans": len(values),
                "aggregation": "per-solve arrays retained; no cross-replan per-task mean"}

    swm.World.__init__ = world_init
    swm.World.set_policy = set_policy
    swm.World.evaluate_from_dataset = evaluate
    solver_cls.solve = solve
    JEPA.get_cost = get_cost
    official._build_loss_metrics = loss_metrics
    started = time.perf_counter()
    try:
        official.run.__wrapped__(cfg)
    finally:
        for world in worlds:
            world.close()
    metrics = results["metrics"]
    successes = metrics.get("episode_successes")
    if not isinstance(successes, list) or len(successes) != 50 or not solve_rows:
        raise RuntimeError("Missing complete 50-task outcomes or solve records")
    if not all_finite(metrics) or nonfinite_cost_calls:
        raise RuntimeError("Invalid numeric metrics")
    write_json(out / "stage1_gate.json", {"status": "PASS", "complete_task_outcomes": len(successes),
               "cost_calls": cost_calls, "nonfinite_cost_calls": nonfinite_cost_calls,
               "checkpoint_loaded": True, "finite_actions": True})
    write_json(out / "run_summary.json", {
        "status": "COMPLETED", "validity": "PASS", "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host, "source_identity": prepared, "tasks": 50,
        "successes": sum(bool(x) for x in successes), "success_rate_percent": 2*sum(bool(x) for x in successes),
        "episode_successes": successes, "metrics": metrics, "evaluation_seconds": results["evaluation_seconds"],
        "runner_seconds": time.perf_counter()-started, "solve_records": solve_rows,
        "candidate_cost_calls": cost_calls, "candidate_scores": cost_candidate_scores,
        "evaluation_api": "native evaluate_from_dataset",
        "video": None, "official_buffered_action_optimization": buffered_fast_path,
        "object_loader_adapter": False,
        "instrumentation": "Each cost call checks finiteness; solve timing includes instrumentation. No speedup claim.",
        "claim_boundary": "Pretrained base Fast-LeWM PushT, one official seed, 50 dataset-source tasks; no training or four-task paper reproduction.",
    })


def jsonable(value, torch, np):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v, torch, np) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, torch, np) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(all_finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


if __name__ == "__main__":
    main()
