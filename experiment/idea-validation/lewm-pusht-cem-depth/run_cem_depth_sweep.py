#!/usr/bin/env python3
"""Run the frozen LeWM PushT CEM iteration-depth gate and paired sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


BASELINE_JOB = "25534994.pbs101"
ARM_ORDER = ("cem_300_30_iter_30", "cem_300_30_iter_05", "cem_300_30_iter_10", "cem_300_30_iter_20")
ARM_CONFIGS = {
    "cem_300_30_iter_30": (300, 30, 30),
    "cem_300_30_iter_05": (300, 30, 5),
    "cem_300_30_iter_10": (300, 30, 10),
    "cem_300_30_iter_20": (300, 30, 20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-dir", "baseline-artifacts", "lewm-root", "stablewm-root",
        "control-root", "staged-home", "cache-root", "freeze", "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> str:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("runner requires PBS_JOBID and a real PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    allocated = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in allocated:
        raise RuntimeError(f"host {host} is not present in PBS_NODEFILE")
    return host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def jsonable(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return False


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def validate_baseline(args: argparse.Namespace, freeze: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, int]], list[bool]]:
    baseline_dir = args.baseline_dir.resolve(strict=True)
    artifacts = args.baseline_artifacts.resolve(strict=True)
    summary_path = artifacts / "summary.json"
    tasks_path = artifacts / "selected_tasks.json"
    status_path = artifacts / "job_status"
    final_path = artifacts / "final_exit_status.txt"
    for path in (summary_path, tasks_path, status_path, final_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if "EXIT_STATUS=0" not in status_path.read_text(encoding="utf-8"):
        raise RuntimeError("reference baseline PBS job_status is not successful")
    if "RUNNER_EXIT_STATUS=0" not in final_path.read_text(encoding="utf-8"):
        raise RuntimeError("reference baseline runner exit status is not successful")
    summary = read_json(summary_path)
    selected = read_json(tasks_path)
    if summary.get("pbs_job_id") != BASELINE_JOB or selected.get("pbs_job_id") != BASELINE_JOB:
        raise RuntimeError(f"reference inputs must be bound to {BASELINE_JOB}")
    if summary.get("status") != "COMPLETED" or summary.get("validity") != "PASS":
        raise RuntimeError("reference 300/30 baseline is not complete and valid")
    if int(summary.get("successes", -1)) < 5:
        raise RuntimeError("reference baseline is below the frozen 5/50 engineering floor")
    if summary.get("successes") != freeze["reference_baseline"]["successes"]:
        raise RuntimeError("reference baseline success count differs from freeze")
    source = summary.get("source_metadata", {})
    task_source = selected.get("source_metadata", {})
    for key in ("dataset_path", "dataset_size_bytes", "checkpoint_path", "checkpoint_size_bytes"):
        if source.get(key) != task_source.get(key):
            raise RuntimeError(f"reference task/source metadata mismatch: {key}")
    protocol = freeze["protocol"]
    for key in ("dataset_path", "checkpoint_path", "dataset_size_bytes", "checkpoint_size_bytes"):
        if source.get(key) != protocol[key]:
            raise RuntimeError(f"reference source differs from frozen protocol: {key}")
    tasks = selected.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise RuntimeError("reference selected_tasks.json must contain exactly 50 rows")
    tasks = [{key: int(row[key]) for key in ("row_index", "episode_idx", "start_step")} for row in tasks]
    if len({row["row_index"] for row in tasks}) != 50:
        raise RuntimeError("reference task row indices are not unique")
    if len({row["episode_idx"] for row in tasks}) != 50:
        raise RuntimeError("reference task manifest must contain 50 unique source episodes")
    if int(selected.get("selection_rng_seed", -1)) != 42:
        raise RuntimeError("reference task-selection seed is not the upstream seed 42")
    successes = summary.get("episode_successes")
    if not isinstance(successes, list) or len(successes) != 50:
        raise RuntimeError("reference baseline lacks its 50-item success vector")
    successes = [bool(value) for value in successes]
    if sum(successes) != int(summary["successes"]):
        raise RuntimeError("reference success vector disagrees with its count")
    if baseline_dir.name != "official-lewm-dataset-teacher-baseline":
        raise RuntimeError("unexpected reference baseline directory")
    return summary, tasks, successes


def run_arm(
    name: str,
    num_samples: int,
    topk: int,
    iterations: int,
    baseline: Any,
    swm: Any,
    cfg: Any,
    dataset: Any,
    process: dict[str, Any],
    transforms: Any,
    torch: Any,
    spt: Any,
    teacher: Any,
    tasks: list[dict[str, int]],
    out: Path,
) -> dict[str, Any]:
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.solver.num_samples = num_samples
    cfg.solver.topk = topk
    cfg.solver.n_steps = iterations
    if (int(cfg.solver.n_steps), int(cfg.solver.num_samples), int(cfg.solver.topk)) != (iterations, num_samples, topk):
        raise RuntimeError(f"CEM iteration/config drifted for {name}")
    solver = hydra.utils.instantiate(cfg.solver, model=teacher)
    if not hasattr(solver, "torch_gen") or int(solver.torch_gen.initial_seed()) != 42:
        raise RuntimeError("pinned CEM solver does not expose the frozen seed-42 generator")
    if (int(solver.num_samples), int(solver.topk), int(solver.n_steps)) != (num_samples, topk, iterations):
        raise RuntimeError(f"instantiated CEM config differs from the frozen arm {name}")

    world = baseline.make_pinned_world(swm, cfg)
    image_transform = baseline.make_image_transform(cfg, spt, transforms, torch)
    plan_config = swm.PlanConfig(**cfg.plan_config)
    if not bool(getattr(plan_config, "warm_start", False)):
        raise RuntimeError("official PlanConfig warm_start must remain enabled")
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=dict(process),
        transform={"pixels": image_transform, "goal": baseline.make_image_transform(cfg, spt, transforms, torch)},
    )
    solve_records: list[dict[str, Any]] = []
    original_solve = solver.solve

    def timed_solve(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_solve(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actions = result.get("actions") if isinstance(result, dict) else None
        solve_records.append({
            "ordinal": len(solve_records),
            "wall_seconds": elapsed,
            "candidate_count": num_samples,
            "elite_count": topk,
            "requested_cem_iterations": iterations,
            "cem_iterations": int(solver.n_steps),
            "environment_count": int(solver.n_envs),
            "env_chunks_per_solve": math.ceil(int(solver.n_envs) / int(solver.batch_size)),
            "cost_evaluations_per_solve": int(solver.n_steps) * math.ceil(int(solver.n_envs) / int(solver.batch_size)),
            "actions_finite": bool(torch.is_tensor(actions) and torch.isfinite(actions).all()),
        })
        return result

    solver.solve = timed_solve
    started = time.perf_counter()
    world.set_policy(policy)
    try:
        metrics = baseline.evaluate_selected_tasks(world, dataset, tasks, cfg, OmegaConf)
        torch.cuda.synchronize()
    finally:
        world.close()

    metrics = jsonable(metrics)
    raw_successes = metrics.get("episode_successes") if isinstance(metrics, dict) else None
    vector_complete = isinstance(raw_successes, list) and len(raw_successes) == 50
    successes = [bool(value) for value in raw_successes] if isinstance(raw_successes, list) else []
    invalid_reasons = []
    if not vector_complete:
        invalid_reasons.append("World.evaluate did not return exactly 50 episode outcomes")
    if not all_finite(metrics):
        invalid_reasons.append("World.evaluate returned non-finite metrics")
    if not solve_records:
        invalid_reasons.append("no solver.solve calls were recorded")
    if any(not row["actions_finite"] for row in solve_records):
        invalid_reasons.append("a solver.solve output contained non-finite actions")
    solve_times = [float(row["wall_seconds"]) for row in solve_records]
    steady = solve_times[1:]
    if invalid_reasons:
        arm_dir = out / "arms"
        arm_dir.mkdir(exist_ok=True)
        partial = {
            "arm": name,
            "validity": "FAIL",
            "invalid_reasons": invalid_reasons,
            "num_samples": num_samples,
            "topk": topk,
            "requested_iterations": iterations,
            "iterations": int(solver.n_steps),
            "tasks_requested": len(tasks),
            "episode_outcomes_observed": len(successes),
            "successes_observed": sum(successes),
            "episode_successes": successes if vector_complete else None,
            "planner_solve_calls": len(solve_records),
            "planner_solve_wall_seconds_synchronized_total": sum(solve_times),
            "planner_solve_wall_seconds_each": solve_times,
            "second_solve_seconds": solve_times[1] if len(solve_times) > 1 else None,
            "world_evaluate_metrics": json_safe(metrics),
        }
        write_json(arm_dir / f"{name}.json", partial)
        with (arm_dir / "solve_times.jsonl").open("a", encoding="utf-8") as handle:
            for record in solve_records:
                handle.write(json.dumps({"arm": name, **record}, ensure_ascii=False, allow_nan=False) + "\n")
        if isinstance(raw_successes, list):
            with (arm_dir / "episode_results.jsonl").open("a", encoding="utf-8") as handle:
                for index, (task, success) in enumerate(zip(tasks, successes)):
                    handle.write(json.dumps({"arm": name, "task_index": index, **task, "success": success}, ensure_ascii=False) + "\n")
        raise RuntimeError(f"invalid planner result for {name}: {'; '.join(invalid_reasons)}")
    result = {
        "arm": name,
        "validity": "PASS",
        "num_samples": num_samples,
        "topk": topk,
        "requested_iterations": iterations,
        "iterations": int(solver.n_steps),
        "solver_seed": int(solver.torch_gen.initial_seed()),
        "tasks": len(tasks),
        "successes": sum(successes),
        "success_rate_percent": 2.0 * sum(successes),
        "episode_successes": successes,
        "planner_solve_calls": len(solve_records),
        "candidate_cost_evaluations": sum(row["cost_evaluations_per_solve"] * num_samples for row in solve_records),
        "planner_solve_wall_seconds_synchronized_total": sum(solve_times),
        "planner_solve_wall_seconds_post_first_median": statistics.median(steady) if steady else None,
        "post_first_solve_sample_count": len(steady),
        "second_solve_seconds": solve_times[1] if len(solve_times) > 1 else None,
        "planner_solve_wall_seconds_each": solve_times,
        "evaluation_wall_seconds": time.perf_counter() - started,
        "world_evaluate_metrics": metrics,
    }
    arm_dir = out / "arms"
    arm_dir.mkdir(exist_ok=True)
    write_json(arm_dir / f"{name}.json", result)
    with (arm_dir / "episode_results.jsonl").open("a", encoding="utf-8") as handle:
        for index, (task, success) in enumerate(zip(tasks, successes, strict=True)):
            handle.write(json.dumps({"arm": name, "task_index": index, **task, "success": success}, ensure_ascii=False) + "\n")
    with (arm_dir / "solve_times.jsonl").open("a", encoding="utf-8") as handle:
        for record in solve_records:
            handle.write(json.dumps({"arm": name, **record}, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def pair_counts(control: list[bool], treatment: list[bool]) -> dict[str, int]:
    pairs = list(zip(control, treatment, strict=True))
    return {
        "both_success": sum(a and b for a, b in pairs),
        "control_only_success": sum(a and not b for a, b in pairs),
        "treatment_only_success": sum(not a and b for a, b in pairs),
        "both_fail": sum(not a and not b for a, b in pairs),
    }


def main() -> None:
    args = parse_args()
    host = require_compute_node()
    out = args.out.resolve(strict=True)
    if out == Path("/") or any((out / name).exists() for name in ("run_summary.json", "selected_tasks.json")):
        raise FileExistsError("output directory is invalid or already contains sweep results")
    freeze = read_json(args.freeze.resolve(strict=True))
    if freeze.get("schema") != "lewm-pusht-cem-depth.freeze" or freeze.get("revision") != 1:
        raise RuntimeError("CEM depth freeze identity mismatch")
    if freeze.get("arm_order") != list(ARM_ORDER) or freeze.get("arms") != {key: list(value) for key, value in ARM_CONFIGS.items()}:
        raise RuntimeError("CEM depth arms/order differ from the pre-results freeze")
    reference, tasks, reference_successes = validate_baseline(args, freeze)

    baseline_dir = args.baseline_dir.resolve(strict=True)
    control_root = args.control_root.resolve(strict=True)
    lewm_root = args.lewm_root.resolve(strict=True)
    stablewm_root = args.stablewm_root.resolve(strict=True)
    staged_home = args.staged_home.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    h5 = staged_home / "pusht_expert_train.h5"
    checkpoint = staged_home / "pusht" / "lewm_object.ckpt"
    cache_h5 = cache_root / "datasets" / h5.name
    cache_checkpoint = cache_root / "pusht" / checkpoint.name
    protocol = freeze["protocol"]
    if str(h5.resolve(strict=True)) != protocol["dataset_path"]:
        raise RuntimeError("resolved staged HDF5 path differs from the frozen dataset path")
    if str(checkpoint.resolve(strict=True)) != protocol["checkpoint_path"]:
        raise RuntimeError("resolved staged checkpoint path differs from the frozen checkpoint path")
    if h5.stat().st_size != int(reference["source_metadata"]["dataset_size_bytes"]):
        raise RuntimeError("staged HDF5 size differs from the valid reference baseline")
    if checkpoint.stat().st_size != int(reference["source_metadata"]["checkpoint_size_bytes"]):
        raise RuntimeError("staged teacher size differs from the valid reference baseline")
    if cache_h5.resolve(strict=True) != h5.resolve(strict=True) or cache_checkpoint.resolve(strict=True) != checkpoint.resolve(strict=True):
        raise RuntimeError("job-private cache links do not resolve to frozen inputs")

    sys.path[:0] = [str(lewm_root), str(stablewm_root), str(control_root), str(baseline_dir)]
    os.environ["STABLEWM_HOME"] = str(cache_root)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Imports that can initialize data/model runtimes remain behind the allocation guard.
    import hdf5plugin
    import hydra
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    import run_dataset_teacher_baseline as baseline
    from run_lewm_recurrent_student import load_official_checkpoint

    if not Path(swm.__file__).resolve().is_relative_to(stablewm_root):
        raise RuntimeError("stable_worldmodel imported outside the frozen source root")
    if not Path(spt.__file__).resolve().is_relative_to((stablewm_root.parent / "venv").resolve(strict=True)):
        raise RuntimeError("stable_pretraining imported outside the staged virtual environment")
    if Path(sys.prefix).resolve(strict=True) != (stablewm_root.parent / "venv").resolve(strict=True):
        raise RuntimeError("Python sys.prefix differs from the staged virtual environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    cfg = baseline.compose_pinned_config(lewm_root)
    if int(cfg.world.max_episode_steps) != 100 or int(cfg.eval.eval_budget) != 50:
        raise RuntimeError("pinned upstream evaluator limits drifted")
    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=cache_root
    )
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected_data = dataset.get_row_data(np.asarray([row["row_index"] for row in tasks], dtype=np.int64))
    for index, task in enumerate(tasks):
        if int(selected_data[episode_column][index]) != task["episode_idx"] or int(selected_data["step_idx"][index]) != task["start_step"]:
            raise RuntimeError(f"frozen task row differs from staged HDF5 at task {index}")

    teacher = load_official_checkpoint(cache_root).to("cuda").eval()
    teacher.requires_grad_(False)
    teacher.interpolate_pos_encoding = True
    write_json(out / "selected_tasks.json", {
        "source_job_id": BASELINE_JOB,
        "selection_rng_seed": 42,
        "tasks": tasks,
        "source_metadata": reference["source_metadata"],
    })
    write_json(out / "run_order.json", {
        "arms": list(ARM_ORDER),
        "note": "The 30-iteration 300/30 arm runs first as the stage-1 gate and paired control; 5, 10, and 20 iterations run only if its exact baseline success vector matches.",
    })

    outcomes: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        num_samples, topk, iterations = ARM_CONFIGS[arm]
        print(json.dumps({"event": "arm_start", "arm": arm, "num_samples": num_samples, "topk": topk, "iterations": iterations}), flush=True)
        result = run_arm(
            arm, num_samples, topk, iterations, baseline, swm, cfg, dataset, process,
            transforms, torch, spt, teacher, tasks, out,
        )
        outcomes[arm] = result
        if arm == "cem_300_30_iter_30":
            match = result["episode_successes"] == reference_successes
            gate = result["successes"] >= 5 and match
            stage1 = {
                "status": "PASS" if gate else "STOP",
                "success_vector_matches_25534994": match,
                "successes": result["successes"],
                "engineering_floor_passes": result["successes"] >= 5,
                "result": result,
            }
            write_json(out / "stage1_gate.json", stage1)
            if not gate:
                write_json(out / "run_summary.json", {
                    "status": "STOPPED_AFTER_STAGE1_GATE",
                    "reason": "invalid 30-iteration control, control success vector mismatch, or below 5/50 engineering floor",
                    "pbs_job_id": os.environ["PBS_JOBID"],
                    "compute_host": host,
                    "task_count": len(tasks),
                    "baseline_job_id": BASELINE_JOB,
                    "arms_completed": [arm],
                    "stage1_gate": stage1,
                })
                print("STOP_AFTER_STAGE1_GATE", flush=True)
                return

    control = outcomes["cem_300_30_iter_30"]
    contrasts = {}
    for arm in ARM_ORDER[1:]:
        result = outcomes[arm]
        second_control = control["second_solve_seconds"]
        second_arm = result["second_solve_seconds"]
        median_control = control["planner_solve_wall_seconds_post_first_median"]
        median_arm = result["planner_solve_wall_seconds_post_first_median"]
        median_ratio = median_arm / median_control if median_control else None
        second_ratio = second_arm / second_control if second_control else None
        delta = result["successes"] - control["successes"]
        contrasts[arm] = {
            "success_delta_vs_iter_30": delta,
            "paired_success_counts": pair_counts(control["episode_successes"], result["episode_successes"]),
            "post_first_median_solve_time_ratio_vs_iter_30": median_ratio,
            "post_first_median_solve_sample_counts": {
                "iter_30": control["post_first_solve_sample_count"],
                arm: result["post_first_solve_sample_count"],
            },
            "second_solve_time_ratio_vs_iter_30": second_ratio,
            "planner_solve_call_count_ratio_vs_iter_30": (
                result["planner_solve_calls"] / control["planner_solve_calls"]
                if control["planner_solve_calls"] else None
            ),
            "total_solve_wall_seconds_ratio_vs_iter_30": (
                result["planner_solve_wall_seconds_synchronized_total"]
                / control["planner_solve_wall_seconds_synchronized_total"]
                if control["planner_solve_wall_seconds_synchronized_total"] else None
            ),
            "timing_interpretation": "Descriptive single-run timing; report solve-call counts, all per-solve times, full totals, and post-first median. Arm order and cold-start effects are not counterbalanced.",
        }
    write_json(out / "run_summary.json", {
        "status": "COMPLETED",
        "experiment_id": freeze["experiment_id"],
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "validity": "PASS",
        "task_count": len(tasks),
        "baseline_job_id": BASELINE_JOB,
        "baseline_success_vector_match_required": True,
        "stage1_gate": read_json(out / "stage1_gate.json"),
        "arm_order": list(ARM_ORDER),
        "arms": outcomes,
        "contrasts_vs_iter_30": contrasts,
        "claim_boundary": "Protocol-aligned fixed dataset evaluation over 50 paired source tasks; action trajectories diverge across arms, RNG streams are not common innovations after iteration counts diverge, results are single-seed descriptive evidence, and this is not random-reset simulator success.",
    })
    print(json.dumps({"event": "completed", "successes": {arm: value["successes"] for arm, value in outcomes.items()}}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
