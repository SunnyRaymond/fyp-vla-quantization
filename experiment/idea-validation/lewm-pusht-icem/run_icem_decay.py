#!/usr/bin/env python3
"""Run the frozen LeWM PushT iCEM population-decay ablation."""

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
SEEDS = (42, 43, 44)
ARM_ORDER = ("cem_300_30", "cem_150_30", "decay_equal_4500", "decay_paper_125")
SCHEDULES = {
    "cem_300_30": (300,) * 30,
    "cem_150_30": (150,) * 30,
    "decay_equal_4500": tuple(max(60, round(300 / 1.0572**i)) for i in range(30)),
    "decay_paper_125": tuple(max(60, round(300 / 1.25**i)) for i in range(30)),
}
assert sum(SCHEDULES["decay_equal_4500"]) == 4500


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
    if any(type(value) is not bool for value in successes):
        raise RuntimeError("reference baseline success vector contains a non-boolean value")
    if sum(successes) != int(summary["successes"]):
        raise RuntimeError("reference success vector disagrees with its count")
    if baseline_dir.name != "official-lewm-dataset-teacher-baseline":
        raise RuntimeError("unexpected reference baseline directory")
    return summary, tasks, successes


def run_arm(
    name: str,
    seed: int,
    schedule: tuple[int, ...],
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
    num_samples = max(schedule)
    topk = 30
    cfg.solver.num_samples = num_samples
    cfg.solver.topk = topk
    cfg.solver.seed = seed
    if int(cfg.solver.n_steps) != 30 or len(schedule) != 30 or min(schedule) < 2 * topk:
        raise RuntimeError(f"CEM budget drifted for {name}")
    if "_decay_" in name:
        from decay_cem import DecayCEMSolver
        solver = DecayCEMSolver(
            model=teacher, batch_size=int(cfg.solver.batch_size),
            num_samples=num_samples, var_scale=float(cfg.solver.var_scale),
            n_steps=int(cfg.solver.n_steps), topk=topk,
            device=str(cfg.solver.device), seed=seed, candidate_schedule=schedule,
        )
    else:
        solver = hydra.utils.instantiate(cfg.solver, model=teacher)
    if not hasattr(solver, "torch_gen") or int(solver.torch_gen.initial_seed()) != seed:
        raise RuntimeError("CEM solver does not expose the frozen seed generator")
    if (int(solver.num_samples), int(solver.topk), int(solver.n_steps)) != (num_samples, topk, 30):
        raise RuntimeError(f"instantiated CEM budget differs from the frozen arm {name}")

    world = baseline.make_pinned_world(swm, cfg)
    image_transform = baseline.make_image_transform(cfg, spt, transforms, torch)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg.plan_config),
        process=dict(process),
        transform={"pixels": image_transform, "goal": baseline.make_image_transform(cfg, spt, transforms, torch)},
    )
    solve_records: list[dict[str, Any]] = []
    original_solve = solver.solve

    def timed_solve(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        started = time.perf_counter()
        cost_shapes: list[tuple[int, int]] = []
        original_get_cost = solver.model.get_cost
        had_get_cost_override = "get_cost" in vars(solver.model)

        def counted_get_cost(infos: Any, candidates: Any) -> Any:
            cost_shapes.append((int(candidates.shape[0]), int(candidates.shape[1])))
            return original_get_cost(infos, candidates)

        solver.model.get_cost = counted_get_cost
        try:
            result = original_solve(*args, **kwargs)
        except Exception as exc:
            arm_dir = out / "arms"
            arm_dir.mkdir(exist_ok=True)
            with (arm_dir / "partial_failures.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "arm": name, "seed": seed, "solve_ordinal": len(solve_records),
                    "get_cost_calls_observed": len(cost_shapes),
                    "candidate_evaluations_observed": sum(batch * n for batch, n in cost_shapes),
                    "exception": f"{type(exc).__name__}: {exc}",
                }, ensure_ascii=False) + "\n")
            raise
        finally:
            if had_get_cost_override:
                solver.model.get_cost = original_get_cost
            else:
                del solver.model.get_cost
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        info_dict = args[0] if args else kwargs["info_dict"]
        call_envs = len(next(iter(info_dict.values())))
        chunks = math.ceil(call_envs / int(solver.batch_size))
        expected_counts = list(schedule) * chunks
        if [n for _, n in cost_shapes] != expected_counts:
            raise RuntimeError(
                f"{name}: observed per-call candidate counts differ from frozen schedule "
                f"(call_envs={call_envs}, chunks={chunks}, "
                f"observed={len(cost_shapes)}, expected={len(expected_counts)})"
            )
        actual_evaluations = sum(batch * n for batch, n in cost_shapes)
        if actual_evaluations != call_envs * sum(schedule):
            raise RuntimeError(f"{name}: observed candidate evaluation total differs from frozen schedule")
        actions = result.get("actions") if isinstance(result, dict) else None
        solve_records.append({
            "ordinal": len(solve_records),
            "wall_seconds": elapsed,
            "candidate_schedule": list(schedule),
            "elite_count": topk,
            "cem_iterations": int(solver.n_steps),
            "environment_count": call_envs,
            "env_chunks_per_solve": chunks,
            "get_cost_call_count": len(cost_shapes),
            "cost_evaluations_per_solve": actual_evaluations,
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
    vector_complete = (
        isinstance(raw_successes, list)
        and len(raw_successes) == 50
        and all(type(value) is bool for value in raw_successes)
    )
    successes = raw_successes if vector_complete else []
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
            "seed": seed,
            "validity": "FAIL",
            "invalid_reasons": invalid_reasons,
            "num_samples": num_samples,
            "topk": topk,
            "candidate_schedule": list(schedule),
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
        "iterations": int(solver.n_steps),
        "solver_seed": int(solver.torch_gen.initial_seed()),
        "tasks": len(tasks),
        "successes": sum(successes),
        "success_rate_percent": 2.0 * sum(successes),
        "episode_successes": successes,
        "planner_solve_calls": len(solve_records),
        "candidate_schedule": list(schedule),
        "candidate_cost_evaluations": sum(row["cost_evaluations_per_solve"] for row in solve_records),
        "seed": seed,
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
    if freeze.get("schema") != "lewm-pusht-icem-decay.freeze" or freeze.get("revision") != 1:
        raise RuntimeError("iCEM decay freeze identity mismatch")
    if freeze.get("arm_order") != list(ARM_ORDER) or freeze.get("seeds") != list(SEEDS):
        raise RuntimeError("iCEM decay arms/seeds differ from the pre-results freeze")
    if freeze.get("schedules") != {key: list(value) for key, value in SCHEDULES.items()}:
        raise RuntimeError("iCEM decay schedules differ from the pre-results freeze")
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
        "seeds": list(SEEDS), "arms": list(ARM_ORDER),
        "note": "Seed 42 300/30 is the stage-1 gate; every seed has its own paired control.",
    })

    outcomes: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for arm in ARM_ORDER:
            key = f"seed{seed}_{arm}"
            schedule = SCHEDULES[arm]
            print(json.dumps({"event": "arm_start", "arm": arm, "seed": seed, "schedule_sum": sum(schedule)}), flush=True)
            result = run_arm(
                key, seed, schedule, baseline, swm, cfg, dataset, process,
                transforms, torch, spt, teacher, tasks, out,
            )
            outcomes[key] = result
            if seed == 42 and arm == "cem_300_30":
                match = result["episode_successes"] == reference_successes
                gate = result["successes"] >= 5 and match
                stage1 = {
                    "status": "PASS" if gate else "STOP",
                    "success_vector_matches_25534994": match,
                    "successes": result["successes"],
                    "engineering_floor_passes": result["successes"] >= 5,
                }
                write_json(out / "stage1_gate.json", stage1)
                if not gate:
                    write_json(out / "run_summary.json", {
                        "status": "STOPPED_AFTER_STAGE1_GATE",
                        "reason": "seed-42 control differs from the valid reference",
                        "pbs_job_id": os.environ["PBS_JOBID"],
                        "compute_host": host,
                        "arms_completed": list(outcomes),
                        "stage1_gate": stage1,
                    })
                    raise RuntimeError("seed-42 stage-1 control gate stopped the experiment")

    contrasts = {}
    for seed in SEEDS:
        control = outcomes[f"seed{seed}_cem_300_30"]
        fixed = outcomes[f"seed{seed}_cem_150_30"]
        for arm in ARM_ORDER[1:]:
            result = outcomes[f"seed{seed}_{arm}"]
            comparator = fixed if arm == "decay_equal_4500" else control
            contrasts[f"seed{seed}_{arm}"] = {
                "vs_300_30": pair_counts(control["episode_successes"], result["episode_successes"]),
                "success_delta_vs_300_30": result["successes"] - control["successes"],
                "second_solve_ratio_vs_300_30": result["second_solve_seconds"] / control["second_solve_seconds"],
                "vs_budget_matched_150_30": pair_counts(fixed["episode_successes"], result["episode_successes"]) if arm == "decay_equal_4500" else None,
                "success_delta_vs_comparator": result["successes"] - comparator["successes"],
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
        "seeds": list(SEEDS),
        "arm_order": list(ARM_ORDER),
        "schedules": {key: list(value) for key, value in SCHEDULES.items()},
        "arms": outcomes,
        "contrasts": contrasts,
        "claim_boundary": "iCEM-inspired population decay only, not full iCEM. Paired protocol-aligned fixed dataset evaluation, not formal noninferiority or random-reset simulator success.",
    })
    print(json.dumps({"event": "completed", "successes": {arm: value["successes"] for arm, value in outcomes.items()}}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
