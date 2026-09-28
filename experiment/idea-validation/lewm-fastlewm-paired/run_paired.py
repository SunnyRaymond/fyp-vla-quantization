#!/usr/bin/env python3
"""Run the frozen LeWM/Fast-LeWM × CEM-budget arms in isolated processes."""

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_allocation():
    host = platform.node().split(".")[0].lower()
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    if not os.environ.get("PBS_JOBID") or not nodefile.is_file():
        raise RuntimeError("A real PBS compute allocation is required")
    nodes = {line.strip().split(".")[0].lower() for line in nodefile.read_text().splitlines()}
    if host not in nodes or any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-allocation/control host: {host}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
        raise RuntimeError("A visible allocated GPU is required")
    return host


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def self_check(freeze):
    arms = freeze["arms"]
    ids = [arm["id"] for arm in arms]
    expected = {(model, it, seed) for model in ("lewm", "fastlewm") for it in (10, 30) for seed in (42, 43, 44)}
    actual = {(arm["model"], int(arm["iterations"]), int(arm["seed"])) for arm in arms}
    if len(arms) != 12 or len(set(ids)) != 12 or actual != expected:
        raise RuntimeError("Frozen arm schedule is not the required 12 unique model × budget × seed cells")
    settings = freeze["settings"]
    for key, value in {"num_eval": 50, "num_samples": 300, "topk": 30, "batch_size": 1,
                        "goal_offset_steps": 25, "eval_budget": 50, "steady_repetitions": 3,
                        "warmup_repetitions": 1}.items():
        if settings.get(key) != value:
            raise RuntimeError(f"Frozen setting drift: {key}")
    outcomes = freeze["validity"]["fast_seed42_i30_expected_successes"]
    if len(outcomes) != 50 or [i for i, ok in enumerate(outcomes) if not ok] != [5]:
        raise RuntimeError("Frozen Fast-LeWM control vector is malformed")
    for arm in arms:
        plan = freeze["plans"][arm["model"]]
        if int(plan["horizon"]) * int(plan["action_block"]) != 25:
            raise RuntimeError(f"Plan does not predict 25 primitive actions: {arm['id']}")
        if int(plan["receding_horizon"]) * int(plan["action_block"]) != 25:
            raise RuntimeError(f"Plan does not execute 25 primitive actions: {arm['id']}")
    return {"status": "PASS", "arms": len(arms), "settings": settings}


def run_parent(args):
    host = require_allocation()
    freeze = load_json(args.freeze)
    self_check(freeze)
    prepared = load_json(args.prepared / "prepared.json")
    if prepared.get("status") != "PASS" or not (args.prepared / "process.pkl").is_file():
        raise RuntimeError("Shared CPU preparation is missing or invalid")
    source_tasks = load_json(Path(__file__).with_name("selected_tasks.json"))
    prepared_tasks = load_json(args.prepared / "selected_tasks.json")
    if source_tasks.get("tasks") != prepared_tasks.get("tasks"):
        raise RuntimeError("Prepared task manifest differs from the frozen source manifest")

    outcomes, failed, stop_reason = {}, [], None
    for arm in freeze["arms"]:
        arm_dir = args.out / "arms" / arm["id"]
        if arm_dir.exists():
            raise FileExistsError(f"Refusing to overwrite arm output: {arm_dir}")
        arm_dir.mkdir(parents=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--child",
                   "--stage", str(args.stage), "--fast-root", str(args.fast_root),
                   "--out", str(args.out), "--freeze", str(args.freeze),
                   "--prepared", str(args.prepared), "--model", arm["model"],
                   "--iterations", str(arm["iterations"]), "--seed", str(arm["seed"]),
                   "--arm-id", arm["id"], "--arm-dir", str(arm_dir)]
        print(json.dumps({"event": "arm_start", "arm": arm["id"]}), flush=True)
        proc = subprocess.run(command, check=False)
        result_path = arm_dir / "result.json"
        if result_path.is_file():
            outcomes[arm["id"]] = load_json(result_path)
        else:
            outcomes[arm["id"]] = {"status": "FAILED", "returncode": proc.returncode}
        if proc.returncode != 0 or outcomes[arm["id"]].get("status") != "PASS":
            failed.append(arm["id"])
            stop_reason = "technical_arm_failure"
            break
        if arm["id"] == "fastlewm_i30_s42" and outcomes[arm["id"]].get("episode_successes") != freeze["validity"]["fast_seed42_i30_expected_successes"]:
            failed.append(arm["id"])
            stop_reason = "frozen_fast_control_vector_mismatch"
            break

    validity = {"complete_arms": len(outcomes) == 12 and not failed,
                "fast_seed42_i30_exact_control": None, "lewm_i30_quality_screen": {}}
    control = outcomes.get("fastlewm_i30_s42", {})
    expected = freeze["validity"]["fast_seed42_i30_expected_successes"]
    validity["fast_seed42_i30_exact_control"] = control.get("episode_successes") == expected
    for seed in (42, 43, 44):
        arm_id = f"lewm_i30_s{seed}"
        result = outcomes.get(arm_id, {})
        count = sum(bool(v) for v in result.get("episode_successes", [])) if isinstance(result.get("episode_successes"), list) else None
        validity["lewm_i30_quality_screen"][arm_id] = {
            "successes": count, "pass": count is not None and count >= 45,
        }
    overall = "PASS" if validity["complete_arms"] and validity["fast_seed42_i30_exact_control"] else "INVALID"
    summary = {"status": overall, "pbs_job_id": os.environ["PBS_JOBID"], "compute_host": host,
               "protocol": freeze["protocol"], "arms": outcomes, "failed_arms": failed,
               "not_started_arms": [arm["id"] for arm in freeze["arms"] if arm["id"] not in outcomes],
               "stop_reason": stop_reason,
               "validity": validity,
               "note": "Quality screens are reported without deleting or tuning any completed arm."}
    write_json(args.out / "run_summary.json", summary)
    return 0 if overall == "PASS" else 1


def clone_value(value, torch):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_value(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item, torch) for item in value)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.copy()
    except ImportError:
        pass
    return copy.deepcopy(value)


def rng_state(torch, np):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(state, torch, np):
    py_state, np_state, cpu_state, cuda_states = state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)


def finite(value, torch):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite(v, torch) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v, torch) for v in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def run_child(args):
    host = require_allocation()
    freeze = load_json(args.freeze)
    arm = next((item for item in freeze["arms"] if item["id"] == args.arm_id), None)
    if arm is None or (arm["model"], int(arm["iterations"]), int(arm["seed"])) != (args.model, args.iterations, args.seed):
        raise RuntimeError("Child arm parameters do not match the frozen schedule")
    stage, fast_root = args.stage.resolve(strict=True), args.fast_root.resolve(strict=True)
    fast_source = fast_root / "upstream"
    lewm_source = stage / "le-wm"
    model_source = fast_source if args.model == "fastlewm" else lewm_source
    backend_root = fast_root / "runtime_overlay"
    checkpoint = (fast_root / "checkpoints" / "Fast-lewm_pusht_object.ckpt" if args.model == "fastlewm"
                  else stage / "stablewm_home" / "pusht" / "lewm_object.ckpt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (args.prepared / "process.pkl").is_file():
        raise FileNotFoundError(args.prepared / "process.pkl")
    source_identity_path = fast_root / "source_identity.json"
    source_identity = load_json(source_identity_path)
    if source_identity.get("source_sha") != freeze["fast_source_commit"]:
        raise RuntimeError("Fast source_identity.json differs from the frozen commit")
    revision_file = fast_source / ".fastlewm-revision"
    if revision_file.is_file() and revision_file.read_text(encoding="utf-8").strip() != freeze["fast_source_commit"]:
        raise RuntimeError("Fast upstream .fastlewm-revision differs from the frozen commit")
    sys.path[:0] = [str(model_source), str(fast_source), str(backend_root)]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["STABLEWM_HOME"] = str(args.arm_dir / "cache")
    (args.arm_dir / "cache" / "datasets").mkdir(parents=True, exist_ok=True)
    checkpoint_link = args.arm_dir / "cache" / checkpoint.name
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(checkpoint)

    # Runtime imports and model/data I/O occur only after the allocation guard.
    import hdf5plugin  # noqa: F401
    import hydra
    import numpy as np
    import stable_worldmodel as swm
    import torch
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    if not Path(swm.__file__).resolve().is_relative_to(backend_root.resolve()):
        raise RuntimeError(f"Wrong stable_worldmodel backend: {swm.__file__}")
    if importlib.metadata.version("stable-worldmodel") != "0.0.6":
        raise RuntimeError("The frozen runtime requires stable-worldmodel==0.0.6")
    if Path(sys.prefix).resolve() != (stage / "venv").resolve():
        raise RuntimeError(f"Unexpected Python environment: {sys.prefix}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated child")
    cfg_dir = fast_source / "config" / "eval"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = hydra.compose(config_name="pusht", return_hydra_config=True)
    cfg.hydra.runtime.output_dir = str(args.arm_dir / "native")
    HydraConfig.instance().set_config(cfg)
    settings = freeze["settings"]
    plan = freeze["plans"][args.model]
    cfg.world.num_envs = int(settings["num_eval"])
    cfg.world.max_episode_steps = int(settings["max_episode_steps"])
    cfg.world.history_size = int(plan["history_len"])
    cfg.world.frame_skip = int(settings["frame_skip"])
    cfg.eval.dataset_path = str(stage / "stablewm_home" / "pusht_expert_train.h5")
    cfg.eval.ckpt_path = str(checkpoint)
    cfg.eval.num_eval = int(settings["num_eval"])
    cfg.eval.img_size = int(settings["image_size"])
    cfg.eval.goal_offset_steps = int(settings["goal_offset_steps"])
    cfg.eval.eval_budget = int(settings["eval_budget"])
    cfg.dataset.keys_to_cache = []
    cfg.seed = int(args.seed)
    cfg.solver.num_samples = int(settings["num_samples"])
    cfg.solver.topk = int(settings["topk"])
    cfg.solver.batch_size = int(settings["batch_size"])
    cfg.solver.var_scale = float(settings["var_scale"])
    cfg.solver.n_steps = int(args.iterations)
    cfg.solver.seed = int(args.seed)
    cfg.cache_dir = str(args.arm_dir / "cache")

    eval_path = fast_source / "eval.py"
    spec = importlib.util.spec_from_file_location("paired_official_eval", eval_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    dataset = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=[], cache_dir=stage / "stablewm_home")
    manifest = load_json(args.prepared / "selected_tasks.json")
    tasks = manifest["tasks"]
    row_ids = np.asarray([int(row["row_index"]) for row in tasks], dtype=np.int64)
    episode_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected = dataset.get_row_data(row_ids)
    episodes, starts = [], []
    for i, task in enumerate(tasks):
        episode, start = int(selected[episode_col][i]), int(selected["step_idx"][i])
        if episode != int(task["episode_idx"]) or start != int(task["start_step"]):
            raise RuntimeError(f"Dataset row identity drift at frozen task {i}")
        episodes.append(episode)
        starts.append(start)
    if row_ids.tolist() != sorted(row_ids.tolist()):
        raise RuntimeError("Frozen rows must retain official increasing-index evaluation order")

    with (args.prepared / "process.pkl").open("rb") as handle:
        process = pickle.load(handle)
    expected_keys = {"action", "proprio", "goal_proprio", "state", "goal_state"}
    if set(process) != expected_keys:
        raise RuntimeError(f"Prepared scaler keys differ from contract: {sorted(process)}")
    transform = {"pixels": official.img_transform(cfg), "goal": official.img_transform(cfg)}
    model_name = checkpoint.name[:-len("_object.ckpt")] if checkpoint.name.endswith("_object.ckpt") else checkpoint.stem
    model = swm.policy.AutoCostModel(model_name, cache_dir=args.arm_dir / "cache").to("cuda").eval()
    import jepa
    import module as model_module
    loaded_source_paths = {"jepa": str(Path(jepa.__file__).resolve()),
                           "module": str(Path(model_module.__file__).resolve())}
    for key, value in loaded_source_paths.items():
        if not Path(value).is_relative_to(model_source.resolve()):
            raise RuntimeError(f"{key} imported from unexpected model source: {value}")
    model.requires_grad_(False)
    official._configure_rollout_consistency(model, float(settings["rollout_consistency_weight"]), None)
    for module in model.modules():
        if isinstance(module, torch.nn.GRU):
            module.train()
    model.interpolate_pos_encoding = True
    plan_object = swm.PlanConfig(**{key: value for key, value in plan.items()})
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=plan_object, process=process, transform=transform)
    world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    world.set_policy(policy)
    fast_path_backup = official._install_fast_buffered_action_path(policy)

    cost_state = {"calls": 0, "candidate_scores": 0, "shapes": []}
    original_cost = model.get_cost
    def counted_cost(info, actions):
        cost_state["calls"] += 1
        if hasattr(actions, "shape") and len(actions.shape) >= 2:
            cost_state["candidate_scores"] += int(actions.shape[0]) * int(actions.shape[1])
            if len(cost_state["shapes"]) < 1:
                cost_state["shapes"].append({"actions": list(actions.shape)})
        result = original_cost(info, actions)
        if cost_state["shapes"] and "costs" not in cost_state["shapes"][0]:
            cost_state["shapes"][0]["costs"] = list(result.shape)
        return result
    model.get_cost = counted_cost

    solve_records, benchmark = [], {"steady_times_seconds": [], "steady_active_environments": [],
                                    "steady_cost_calls": [], "steady_candidate_scores": [],
                                    "warmup_cost_calls": None, "warmup_candidate_scores": None,
                                    "context_active_environments": None, "overhead_seconds": 0.0}
    native_solve = solver.solve
    native_solver_method = type(solver).solve
    benchmark_done = False
    def assert_solver_finite(value):
        for key in ("actions", "costs"):
            if key in value and not finite(value[key], torch):
                raise RuntimeError(f"Non-finite native solver {key}")

    def isolated_call(info, init_action, timed):
        # Copy a pristine native solver; keep the heavyweight model shared and its RNG independent.
        fresh = copy.deepcopy(solver, {id(model): model})
        local_info = clone_value(info, torch)
        local_init = clone_value(init_action, torch)
        state = rng_state(torch, np)
        try:
            torch.cuda.synchronize()
            started = time.perf_counter() if timed else None
            output = native_solver_method(fresh, local_info, init_action=local_init)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started if timed else None
            assert_solver_finite(output)
            return output, elapsed
        finally:
            restore_rng(state, torch, np)

    def wrapped_solve(info, *pos, **kwargs):
        nonlocal benchmark_done
        init_action = kwargs.get("init_action", pos[0] if pos else None)
        active = int(info["pixels"].shape[0])
        if not benchmark_done:
            benchmark_done = True
            benchmark["context_active_environments"] = active
            overall_start = time.perf_counter()
            torch.cuda.synchronize()
            before = cost_state["calls"], cost_state["candidate_scores"]
            real_generator = getattr(solver, "torch_gen", None)
            real_generator_state = real_generator.get_state().clone() if real_generator is not None else None
            isolated_call(info, init_action, timed=False)
            benchmark["warmup_cost_calls"] = cost_state["calls"] - before[0]
            benchmark["warmup_candidate_scores"] = cost_state["candidate_scores"] - before[1]
            for _ in range(int(settings["steady_repetitions"])):
                before = cost_state["calls"], cost_state["candidate_scores"]
                _, elapsed = isolated_call(info, init_action, timed=True)
                benchmark["steady_times_seconds"].append(elapsed)
                benchmark["steady_active_environments"].append(active)
                benchmark["steady_cost_calls"].append(cost_state["calls"] - before[0])
                benchmark["steady_candidate_scores"].append(cost_state["candidate_scores"] - before[1])
            torch.cuda.synchronize()
            if real_generator_state is not None and not torch.equal(real_generator.get_state(), real_generator_state):
                raise RuntimeError("Fixed-context benchmark consumed the real evaluation solver torch_gen")
            benchmark["overhead_seconds"] += time.perf_counter() - overall_start
        before = cost_state["calls"], cost_state["candidate_scores"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = native_solve(info, *pos, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert_solver_finite(output)
        solve_records.append({"ordinal": len(solve_records), "active_environments": active,
                              "elapsed_seconds": elapsed, "cost_calls": cost_state["calls"] - before[0],
                              "candidate_scores": cost_state["candidate_scores"] - before[1],
                              "action_shape": list(output["actions"].shape)})
        return output
    solver.solve = wrapped_solve

    evaluation_started = time.perf_counter()
    metrics = None
    try:
        metrics = world.evaluate_from_dataset(
            dataset, episodes_idx=episodes, start_steps=starts,
            goal_offset_steps=int(settings["goal_offset_steps"]),
            eval_budget=int(settings["eval_budget"]),
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
            save_video=False, video_path=None,
        )
        torch.cuda.synchronize()
        evaluation_wall = time.perf_counter() - evaluation_started
    finally:
        if fast_path_backup is not None:
            policy.get_action = fast_path_backup
        world.close()
    successes = [bool(item) for item in np.asarray(metrics["episode_successes"]).tolist()]
    if len(successes) != int(settings["num_eval"]) or not solve_records:
        raise RuntimeError("Native evaluation did not produce 50 outcomes and solve records")
    if len(benchmark["steady_times_seconds"]) != int(settings["steady_repetitions"]):
        raise RuntimeError("First-solve steady benchmark did not complete")
    params = sum(p.numel() for p in model.parameters())
    versions = {name: importlib.metadata.version(name) for name in
                ("stable-worldmodel", "stable-pretraining", "torch", "torchvision", "hydra-core", "numpy")}
    result = {"status": "PASS", "arm_id": args.arm_id, "model": args.model,
              "iterations": args.iterations, "seed": args.seed, "tasks": len(successes),
              "successes": sum(successes), "episode_successes": successes,
              "steady_times_seconds": benchmark["steady_times_seconds"],
              "steady_active_environments": benchmark["context_active_environments"],
              "steady_cost_calls": benchmark["steady_cost_calls"],
              "steady_candidate_scores": benchmark["steady_candidate_scores"],
              "warmup_cost_calls": benchmark["warmup_cost_calls"],
              "warmup_candidate_scores": benchmark["warmup_candidate_scores"],
              "benchmark_context_active_environments": benchmark["context_active_environments"],
              "benchmark_overhead_seconds": benchmark["overhead_seconds"],
              "evaluation_wall_seconds_including_benchmark": evaluation_wall,
              "evaluation_seconds": evaluation_wall - benchmark["overhead_seconds"],
              "solve_records": solve_records, "candidate_cost_calls": cost_state["calls"],
              "candidate_scores": cost_state["candidate_scores"], "cost_shapes": cost_state["shapes"],
              "active_environments_configured": int(settings["num_eval"]),
              "model_parameters": int(params), "model_class": type(model).__module__ + "." + type(model).__name__,
              "versions": versions, "python": sys.version, "compute_host": host,
              "pbs_job_id": os.environ["PBS_JOBID"], "source_paths": {
                  "model_source": str(model_source), "fast_source": str(fast_source),
                  "runtime_overlay": str(backend_root), "checkpoint": str(checkpoint),
                  "dataset": str(stage / "stablewm_home" / "pusht_expert_train.h5")},
              "source_commits": {"fast": freeze["fast_source_commit"], "lewm": freeze["lewm_source_historical_identity"]},
              "historical_provenance": {"lewm_source_identity": freeze["lewm_source_historical_identity"],
                                        "current_git_head_verified": False,
                                        "loaded_model_source_paths": loaded_source_paths,
                                        "fast_source_identity_file": str(source_identity_path)},
              "evaluation_api": "stable_worldmodel.World.evaluate_from_dataset",
              "solver": {"class": type(solver).__module__ + "." + type(solver).__name__,
                         "num_samples": int(settings["num_samples"]), "topk": int(settings["topk"]),
                         "iterations": int(args.iterations), "batch_size": int(settings["batch_size"]),
                         "var_scale": float(settings["var_scale"]), "seed": int(args.seed)},
              "official_buffered_action_path": fast_path_backup is not None,
              "claim_boundary": "Frozen paired model/budget evaluation; native solver timings are per-model and latent costs are not cross-model comparable."}
    write_json(args.arm_dir / "result.json", result)
    print(json.dumps({"event": "arm_complete", "arm": args.arm_id, "successes": result["successes"]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=False)
    parser.add_argument("--fast-root", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=False)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model", choices=("lewm", "fastlewm"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm-id")
    parser.add_argument("--arm-dir", type=Path)
    args = parser.parse_args()
    freeze = load_json(args.freeze)
    if args.self_check:
        print(json.dumps(self_check(freeze), indent=2))
        return 0
    for name in ("stage", "fast_root", "out", "prepared"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.child:
        if None in (args.model, args.iterations, args.seed, args.arm_id, args.arm_dir):
            parser.error("child mode requires --model --iterations --seed --arm-id --arm-dir")
        try:
            return run_child(args)
        except Exception as exc:
            arm_dir = args.arm_dir
            if arm_dir is not None:
                write_json(arm_dir / "result.json", {"status": "FAILED", "arm_id": args.arm_id,
                           "model": args.model, "iterations": args.iterations, "seed": args.seed,
                           "error": repr(exc), "traceback": traceback.format_exc()})
            raise
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
