#!/usr/bin/env python3
"""Compare early and late seven-round teacher schedules on frozen PushT tasks."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ARMS = ("student_only", "early_teacher7", "late_teacher7", "teacher_only")
EXPECTED_ORDER = ("early_teacher7", "teacher_only", "late_teacher7", "student_only")
SCHEDULES = {
    "student_only": (),
    "early_teacher7": (1, 2, 3, 4, 5, 6, 7),
    "late_teacher7": (24, 25, 26, 27, 28, 29, 30),
    "teacher_only": tuple(range(1, 31)),
}
TRACE_ROUNDS = (1, 7, 8, 23, 24, 30)
OBS_KEYS = ("pixels", "goal", "state", "goal_state", "proprio", "action")
BASELINE_JOB = "25534994.pbs101"
SCHEMA = "lewm-pusht-phase-timing-diagnostic.result"
CEM_STEPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-dir", "baseline-artifacts", "lewm-root", "stablewm-root",
        "control-root", "staged-home", "cache-root", "checkpoint", "probe",
        "freeze", "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def solve_record_matches_schedule(record: Mapping[str, Any], arm: str) -> bool:
    """Validate the frozen schedule independently in every CEM callback batch."""
    batch_count = record.get("batch_count")
    if not isinstance(batch_count, int) or batch_count <= 0:
        return False
    flags = record.get("round_teacher_flags")
    finite_flags = record.get("round_finite")
    if not isinstance(flags, list) or not isinstance(finite_flags, list):
        return False
    try:
        wall_s = float(record.get("wall_s"))
    except (TypeError, ValueError):
        return False
    expected_flags = [index in SCHEDULES[arm] for index in range(1, CEM_STEPS + 1)]
    expected_rounds = list(SCHEDULES[arm])
    return (
        len(flags) == CEM_STEPS * batch_count
        and len(finite_flags) == CEM_STEPS * batch_count
        and all(
            flags[start : start + CEM_STEPS] == expected_flags
            for start in range(0, len(flags), CEM_STEPS)
        )
        and record.get("teacher_rounds") == expected_rounds * batch_count
        and all(finite_flags)
        and bool(record.get("output_finite"))
        and math.isfinite(wall_s)
        and wall_s > 0
    )


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


def atomic_json(path: Path, value: Any) -> None:
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
    if isinstance(value, Mapping):
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
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return False


def validate_input(args: argparse.Namespace, freeze: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, int]], list[bool]]:
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
        raise RuntimeError("baseline PBS job_status is not successful")
    if "RUNNER_EXIT_STATUS=0" not in final_path.read_text(encoding="utf-8"):
        raise RuntimeError("baseline runner exit status is not successful")
    summary = read_json(summary_path)
    selected = read_json(tasks_path)
    if summary.get("pbs_job_id") != BASELINE_JOB or selected.get("pbs_job_id") != BASELINE_JOB:
        raise RuntimeError("baseline inputs are not bound to the sole valid retry 25534994.pbs101")
    if summary.get("status") != "COMPLETED" or summary.get("validity") != "PASS":
        raise RuntimeError("teacher baseline is not complete and valid")
    if int(summary.get("successes", -1)) < 5 or summary.get("engineering_continuation_gate") != "ELIGIBLE_FOR_SEPARATE_REVIEW":
        raise RuntimeError("teacher baseline did not meet its frozen 5/50 engineering floor")
    source = summary.get("source_metadata", {})
    tasks_meta = selected.get("source_metadata", {})
    for key in ("dataset_path", "dataset_size_bytes", "checkpoint_path", "checkpoint_size_bytes"):
        if source.get(key) != tasks_meta.get(key):
            raise RuntimeError(f"baseline task/source metadata mismatch: {key}")
    required_source = freeze["task_and_protocol"]
    if source.get("checkpoint_path") != required_source["teacher_checkpoint"]:
        raise RuntimeError("teacher checkpoint path differs from conditional freeze")
    tasks = selected.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise RuntimeError("selected_tasks.json must contain the exact 50 baseline rows")
    normalized: list[dict[str, int]] = []
    for row in tasks:
        normalized.append({key: int(row[key]) for key in ("row_index", "episode_idx", "start_step")})
    if len({row["row_index"] for row in normalized}) != 50:
        raise RuntimeError("baseline task row indices are not unique")
    expected_clusters = int(required_source["expected_unique_source_episode_count"])
    if len({row["episode_idx"] for row in normalized}) != expected_clusters:
        raise RuntimeError(
            f"baseline task list must contain {expected_clusters} unique source episodes"
        )
    if int(selected.get("selection_rng_seed", -1)) != 42:
        raise RuntimeError("baseline task selection seed differs from upstream seed 42")
    baseline_successes = summary.get("episode_successes")
    if not isinstance(baseline_successes, list) or len(baseline_successes) != 50:
        baseline_successes = summary.get("metrics", {}).get("episode_successes")
    if not isinstance(baseline_successes, list) or len(baseline_successes) != 50:
        raise RuntimeError("baseline summary lacks its exact 50-item teacher-only success vector")
    baseline_successes = [bool(value) for value in baseline_successes]
    if sum(baseline_successes) != int(summary["successes"]):
        raise RuntimeError("baseline teacher-only success vector disagrees with success count")
    if baseline_dir.name != "official-lewm-dataset-teacher-baseline":
        raise RuntimeError("unexpected baseline experiment directory")
    return summary, normalized, baseline_successes


def semantic_tensor_equal(left: Any, right: Any, key: str) -> bool:
    import torch

    if left is None or right is None:
        return left is None and right is None
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return False
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if key == "action" and left.is_floating_point():
        pairs = (
            (torch.isnan(left), torch.isnan(right)),
            (torch.isposinf(left), torch.isposinf(right)),
            (torch.isneginf(left), torch.isneginf(right)),
            (torch.isfinite(left), torch.isfinite(right)),
        )
        return all(torch.equal(a, b) for a, b in pairs) and torch.equal(
            left[torch.isfinite(left)], right[torch.isfinite(right)]
        )
    return bool(torch.isfinite(left).all() and torch.isfinite(right).all() and torch.equal(left, right))


def capture_cpu(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


class ProgressCapture:
    def __init__(self, world: Any, n_tasks: int) -> None:
        self.world = world
        self.distances: list[list[float]] = [[] for _ in range(n_tasks)]
        self.error: str | None = None
        original = world._run_iter

        def wrapped(*args: Any, **kwargs: Any):
            original_on_step = kwargs.get("on_step")

            def on_step(current_world: Any) -> None:
                if original_on_step is not None:
                    original_on_step(current_world)
                try:
                    import numpy as np

                    infos = current_world.infos
                    states = np.asarray(infos["state"])
                    goals = np.asarray(infos["goal_state"])
                    envs = current_world.envs.envs
                    if len(envs) != len(self.distances) or states.shape[0] != len(self.distances):
                        raise ValueError("vector world state count differs from selected task count")
                    for index, env_wrapper in enumerate(envs):
                        state = np.asarray(states[index, -1], dtype=np.float64)
                        goal = np.asarray(goals[index, -1], dtype=np.float64)
                        _, distance = env_wrapper.unwrapped.eval_state(goal, state)
                        self.distances[index].append(float(np.asarray(distance).reshape(-1)[0]))
                except Exception as exc:  # Progress is secondary; preserve evaluator execution.
                    self.error = f"{type(exc).__name__}: {exc}"

            kwargs["on_step"] = on_step
            yield from original(*args, **kwargs)

        world._run_iter = wrapped

    def rows(self) -> list[dict[str, Any]]:
        output = []
        for values in self.distances:
            finite_values = [value for value in values if math.isfinite(value)]
            output.append({
                "steps_observed": len(values),
                "initial_state_distance": finite_values[0] if finite_values else None,
                "final_state_distance": finite_values[-1] if finite_values else None,
                "distance_reduction": (finite_values[0] - finite_values[-1]) if finite_values else None,
            })
        return output


def exact_cluster_sign_flip(rows: list[dict[str, Any]], left_arm: str, right_arm: str) -> dict[str, Any]:
    by_episode: dict[int, int] = defaultdict(int)
    by_count: Counter[int] = Counter()
    by_arm_and_task = {
        (row["arm"], int(row["task_index"])): row
        for row in rows if row["arm"] in (left_arm, right_arm)
    }
    for row in rows:
        if row["arm"] != left_arm:
            continue
        task_index = int(row["task_index"])
        paired = by_arm_and_task.get((right_arm, task_index))
        if paired is None or int(paired["row_index"]) != int(row["row_index"]):
            raise RuntimeError("cluster test requires exact paired row identities")
        episode = int(row["episode_idx"])
        if int(paired["episode_idx"]) != episode:
            raise RuntimeError("paired task source episode IDs differ")
        by_episode[episode] += int(bool(row["success"])) - int(bool(paired["success"]))
        by_count[episode] += 1
    values = list(by_episode.values())
    distribution = {0: 1}
    for difference in values:
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            updated[total + difference] += count
            updated[total - difference] += count
        distribution = dict(updated)
    observed = sum(values)
    tail_count = sum(count for total, count in distribution.items() if abs(total) >= abs(observed))
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "observed_cluster_sum": observed,
        "n_clusters": len(values),
        "rows_per_cluster": {str(key): int(value) for key, value in sorted(by_count.items())},
        "exact_two_sided_p": tail_count / (2 ** len(values)) if values else 1.0,
        "method": "exact independent source-episode cluster sign-flip via integer dynamic programming",
    }


def run_arm(
    arm: str,
    tasks: list[dict[str, int]],
    baseline_module: Any,
    swm: Any,
    cfg: Any,
    dataset: Any,
    process: Mapping[str, Any],
    transforms: Any,
    torch: Any,
    spt: Any,
    official: Any,
    reference: Any,
    student: Any,
    old_router_module: Any,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    world = baseline_module.make_pinned_world(swm, cfg)
    image_transform = baseline_module.make_image_transform(cfg, spt, transforms, torch)
    transform = {"pixels": image_transform, "goal": baseline_module.make_image_transform(cfg, spt, transforms, torch)}

    class DiagnosticRoutedCostModel(old_router_module.RoutedCostModel):
        """Use this diagnostic's schedules without changing shared router defaults."""

        def get_cost(self, info_dict: dict[str, Any], action_candidates: Any):
            self.round_index += 1
            chosen = self.arm == "teacher_only" or self.round_index in SCHEDULES[self.arm]
            self.solve_round_choices.append(chosen)
            if chosen:
                self.solve_teacher_rounds.append(self.round_index)
                return self.official.get_cost(info_dict, action_candidates)
            return self.student_get_cost(info_dict, action_candidates)

    route = DiagnosticRoutedCostModel(official, reference, student, arm)

    class BatchAwareTraceCallback(old_router_module.TraceCallback):
        """Reset the routed CEM round counter at the pinned solver batch boundary."""

        def __init__(self) -> None:
            super().__init__(retain_tensors=False)
            self.batch_count = 0
            self._current_trace: dict[str, Any] | None = None
            self.distribution_batches: list[dict[str, Any]] = []

        def reset(self) -> None:
            super().reset()
            self.batch_count = 0
            self._current_trace = None
            self.distribution_batches = []

        def start_batch(self) -> None:
            # CEMSolver calls start_batch once per environment chunk, then runs
            # n_steps cost evaluations. Keep solve-wide records intact.
            self._finish_batch()
            route.round_index = 0
            self.batch_count += 1
            self._current_trace = {"batch_index": self.batch_count - 1, "checkpoints": {}}
            super().start_batch()

        def __call__(self, **kwargs: Any) -> None:
            super().__call__(**kwargs)
            round_index = int(kwargs["step"]) + 1
            if round_index in TRACE_ROUNDS and self._current_trace is not None:
                mean = kwargs["mean"].detach().float()
                variance = kwargs["var"].detach().float()
                self._current_trace["checkpoints"][str(round_index)] = {
                    "mean_rms": mean.square().mean().sqrt(),
                    "variance_mean": variance.mean(),
                }

        def _finish_batch(self) -> None:
            if self._current_trace is not None:
                self.distribution_batches.append(self._current_trace)
                self._current_trace = None

        def end_solve(self) -> None:
            self._finish_batch()
            super().end_solve()

        def distribution_scalars(self) -> list[dict[str, Any]]:
            return [
                {
                    "batch_index": batch["batch_index"],
                    "checkpoints": {
                        round_key: {
                            name: float(value.detach().cpu().item())
                            for name, value in metrics.items()
                        }
                        for round_key, metrics in batch["checkpoints"].items()
                    },
                }
                for batch in self.distribution_batches
            ]

    callback = BatchAwareTraceCallback()
    config = swm.PlanConfig(**cfg.plan_config)
    from hydra.utils import instantiate

    solver = instantiate(cfg.solver, model=route, callbacks=[callback])
    if not hasattr(solver, "torch_gen") or int(solver.torch_gen.initial_seed()) != 42:
        raise RuntimeError("pinned CEM solver does not expose the frozen native seed-42 generator")
    if int(solver.n_steps) != CEM_STEPS:
        raise RuntimeError(f"pinned CEM step count drifted: {solver.n_steps} != {CEM_STEPS}")
    snapshots: dict[str, Any] = {"keys": None, "values": None, "equal_to_first": {}}

    class Policy(swm.policy.WorldModelPolicy):
        def __init__(self) -> None:
            super().__init__(solver=solver, config=config, process=dict(process), transform=transform)

        def _prepare_info(self, info_dict: Any) -> Any:
            prepared = super()._prepare_info(info_dict)
            if snapshots["keys"] is None:
                snapshots["keys"] = tuple(key for key in OBS_KEYS if key in prepared)
                snapshots["values"] = {key: capture_cpu(prepared[key]) for key in snapshots["keys"]}
            return prepared

    policy = Policy()
    solve_records: list[dict[str, Any]] = []
    rng_states: list[Any] = []
    original_solve = solver.solve

    def timed_solve(*solve_args: Any, **solve_kwargs: Any) -> Any:
        route.begin_solve()
        callback.reset()
        state_before = solver.torch_gen.get_state().clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        result = original_solve(*solve_args, **solve_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        route.end_solve()
        actions = result.get("actions") if isinstance(result, Mapping) else None
        output_finite = bool(torch.isfinite(actions).all()) if torch.is_tensor(actions) else False
        round_finite = [bool(item["finite_flags"].item()) for item in callback.history]
        distribution_scalars = callback.distribution_scalars()
        solve_records.append({
            "ordinal": len(solve_records),
            "wall_s": float(elapsed),
            "solver_seed": int(solver.torch_gen.initial_seed()),
            "candidate_shape": callback.cem_input_shapes.get("candidates") if callback.cem_input_shapes else None,
            "batch_count": callback.batch_count,
            "distribution_scalars": distribution_scalars,
            "round_finite": round_finite,
            "round_teacher_flags": list(route.solve_round_choices),
            "teacher_rounds": list(route.solve_teacher_rounds),
            "output_finite": output_finite,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        })
        rng_states.append(state_before)
        # The legacy router captures only the first cost input for Stage 1 audits.
        # This experiment needs shapes from callback and generator-state pairing only.
        route.first_cost_info = None
        route.first_cost_input_shapes = None
        route.first_cost_action_shape = None
        route.single_sample_cost_info_shapes = None
        return result

    solver.solve = timed_solve
    progress = ProgressCapture(world, len(tasks))
    world.set_policy(policy)
    try:
        metrics = baseline_module.evaluate_selected_tasks(world, dataset, tasks, cfg, __import__("omegaconf").OmegaConf)
        torch.cuda.synchronize()
    finally:
        world.close()

    json_metrics = jsonable(metrics)
    successes = json_metrics.get("episode_successes") if isinstance(json_metrics, dict) else None
    if not isinstance(successes, list) or len(successes) != len(tasks):
        raise RuntimeError(f"World.evaluate returned no complete success vector for {arm}")
    successes = [bool(value) for value in successes]
    finite = baseline_module.all_finite(json_metrics)
    schedule_exact = bool(solve_records) and all(
        solve_record_matches_schedule(item, arm) for item in solve_records
    )
    trace_valid = bool(solve_records) and all(
        len(item["distribution_scalars"]) == item["batch_count"]
        and all(
            len(batch["checkpoints"]) == len(TRACE_ROUNDS)
            and set(TRACE_ROUNDS) == {int(key) for key in batch["checkpoints"]}
            for batch in item["distribution_scalars"]
        )
        and baseline_module.all_finite(item["distribution_scalars"])
        for item in solve_records
    )
    trace_by_round: dict[str, dict[str, float | int | None]] = {}
    for checkpoint in TRACE_ROUNDS:
        values = [
            batch["checkpoints"][str(checkpoint)]
            for item in solve_records
            for batch in item["distribution_scalars"]
        ]
        trace_by_round[str(checkpoint)] = {
            "batch_count": len(values),
            "mean_rms": sum(value["mean_rms"] for value in values) / len(values) if values else None,
            "variance_mean": sum(value["variance_mean"] for value in values) / len(values) if values else None,
        }
    for key in OBS_KEYS:
        if key not in (snapshots["keys"] or ()):
            snapshots["equal_to_first"][key] = False
    rows = []
    progress_rows = progress.rows()
    for index, (task, success) in enumerate(zip(tasks, successes, strict=True)):
        row = {
            "arm": arm,
            "task_index": index,
            **task,
            "success": success,
            "progress": progress_rows[index],
        }
        rows.append(row)

    arm_result = {
        "arm": arm,
        "successes": sum(successes),
        "success_rate_percent": 100.0 * sum(successes) / len(successes),
        "episode_successes": successes,
        "teacher_cost_calls": len(route.teacher_rounds),
        "total_cem_solve_calls": len(solve_records),
        "schedule_exact": schedule_exact,
        "finite": bool(finite and trace_valid and all(item["output_finite"] and all(item["round_finite"]) for item in solve_records)),
        "distribution_trace_valid": trace_valid,
        "distribution_trace_mean_by_round": trace_by_round,
        "planner_solve_seconds": sum(item["wall_s"] for item in solve_records),
        "evaluation_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": max((item["peak_cuda_allocated_bytes"] for item in solve_records), default=0),
        "peak_cuda_reserved_bytes": max((item["peak_cuda_reserved_bytes"] for item in solve_records), default=0),
        "progress_capture_error": progress.error,
        "world_evaluate_metrics": json_metrics,
        "prepared_observation_keys": list(snapshots["keys"] or ()),
    }
    with (output / "phase_timing_episodes.jsonl").open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    with (output / "phase_timing_solve_records.jsonl").open("a", encoding="utf-8") as handle:
        for item in solve_records:
            handle.write(json.dumps({"arm": arm, **item}, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "result": arm_result,
        "rows": rows,
        "solve_records": solve_records,
        "rng_states": rng_states,
        "prepared": snapshots["values"],
    }


def main() -> None:
    args = parse_args()
    host = require_compute_node()
    out = args.out.resolve(strict=True)
    if out == Path("/") or any((out / name).exists() for name in (
        "phase_timing_summary.json", "phase_timing_episodes.jsonl", "phase_timing_solve_records.jsonl", "rng_pairing.json"
    )):
        raise FileExistsError("output directory is invalid or already contains phase diagnostic results")
    freeze = read_json(args.freeze.resolve(strict=True))
    if freeze.get("schema") != "lewm-pusht-phase-timing-diagnostic.freeze" or freeze.get("revision") != 1:
        raise RuntimeError("phase-timing diagnostic freeze identity mismatch")
    baseline_summary, tasks, baseline_successes = validate_input(args, freeze)

    arm_order = list(ARMS)
    random.Random(int(freeze["run_order"]["seed"])).shuffle(arm_order)
    if tuple(arm_order) != EXPECTED_ORDER or arm_order != freeze["run_order"]["frozen_order"]:
        raise RuntimeError("frozen randomized arm order does not reproduce")
    atomic_json(out / "run_order.json", {"seed": int(freeze["run_order"]["seed"]), "arms": arm_order, "task_count": len(tasks)})
    atomic_json(out / "selected_tasks.json", {
        "source_job_id": BASELINE_JOB,
        "selection_rng_seed": 42,
        "tasks": tasks,
        "source_metadata": baseline_summary["source_metadata"],
    })

    baseline_dir = args.baseline_dir.resolve(strict=True)
    control_root = args.control_root.resolve(strict=True)
    lewm_root = args.lewm_root.resolve(strict=True)
    stablewm_root = args.stablewm_root.resolve(strict=True)
    staged_home = args.staged_home.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    dataset_path = staged_home / "pusht_expert_train.h5"
    teacher_path = staged_home / "pusht" / "lewm_object.ckpt"
    student_path = args.checkpoint.resolve(strict=True)
    if not all(path.is_file() for path in (dataset_path, teacher_path, student_path)):
        raise FileNotFoundError("frozen PushT dataset or checkpoint path is missing")
    baseline_source = baseline_summary["source_metadata"]
    if dataset_path.stat().st_size != int(baseline_source["dataset_size_bytes"]):
        raise RuntimeError("staged HDF5 size differs from the successful teacher baseline")
    if teacher_path.stat().st_size != int(baseline_source["checkpoint_size_bytes"]):
        raise RuntimeError("staged official teacher size differs from the successful baseline")
    if str(args.checkpoint).replace("\\", "/") != freeze["task_and_protocol"]["student_checkpoint"]:
        raise RuntimeError("student checkpoint path differs from the phase diagnostic freeze")
    if not (cache_root / "datasets" / dataset_path.name).is_file() or not (cache_root / "pusht" / teacher_path.name).is_file():
        raise RuntimeError("PBS cache symlinks for the frozen dataset/teacher are missing")
    if (cache_root / "datasets" / dataset_path.name).resolve(strict=True) != dataset_path.resolve(strict=True):
        raise RuntimeError("dataset cache does not resolve to the baseline HDF5")
    if (cache_root / "pusht" / teacher_path.name).resolve(strict=True) != teacher_path.resolve(strict=True):
        raise RuntimeError("teacher cache does not resolve to the baseline checkpoint")

    sys.path[:0] = [
        str(lewm_root), str(stablewm_root), str(control_root / "lewm-transfer"),
        str(baseline_dir), str(control_root / "lewm-transfer" / "adaptive-teacher-schedule"),
        str(control_root / "lewm-transfer" / "cem-distribution-distill"),
        str(control_root / "lewm-transfer" / "official-pusht-cem"),
    ]
    os.environ["STABLEWM_HOME"] = str(cache_root)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Heavy imports and all HDF5/model reads stay behind the real PBS allocation guard.
    import hdf5plugin  # Register compressed HDF5 filters used by the frozen dataset.
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import jepa
    import torch
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    import run_dataset_teacher_baseline as baseline
    import run_adaptive_teacher_schedule as schedule
    import run_official_pusht_cem as old_router
    from run_lewm_recurrent_student import load_official_checkpoint

    if not Path(swm.__file__).resolve().is_relative_to(stablewm_root):
        raise RuntimeError("stable_worldmodel imported outside staged root")
    if not Path(jepa.__file__).resolve().is_relative_to(lewm_root):
        raise RuntimeError("LeWM JEPA imported outside staged root")
    venv_root = stablewm_root.parent / "venv"
    if Path(sys.prefix).resolve(strict=True) != venv_root.resolve(strict=True):
        raise RuntimeError("Python sys.prefix differs from staged venv")
    if not Path(spt.__file__).resolve().is_relative_to(venv_root.resolve(strict=True)):
        raise RuntimeError("stable_pretraining imported outside staged venv")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    cfg = baseline.compose_pinned_config(lewm_root)
    if int(cfg.world.max_episode_steps) != 100 or int(cfg.eval.eval_budget) != 50:
        raise RuntimeError("upstream evaluator budget/cap drifted")
    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=cache_root
    )
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected_data = dataset.get_row_data(np.asarray([row["row_index"] for row in tasks], dtype=np.int64))
    for index, row in enumerate(tasks):
        if int(selected_data[episode_column][index]) != row["episode_idx"] or int(selected_data["step_idx"][index]) != row["start_step"]:
            raise RuntimeError(f"baseline row/task identity differs from the staged HDF5 at task {index}")

    schedule, _, reference, _ = old_router.load_modules(control_root, lewm_root)
    schedule.validate_interface(reference, args.probe.resolve(strict=True))
    official = load_official_checkpoint(cache_root)
    official.interpolate_pos_encoding = True
    official.requires_grad_(False)
    student, student_meta = schedule.load_main_student(reference, student_path)
    student = student.to("cuda").eval()
    student.requires_grad_(False)

    arm_outputs: dict[str, dict[str, Any]] = {}
    episode_rows: list[dict[str, Any]] = []
    reference_prepared = None
    observation_checks: dict[str, dict[str, bool]] = {}
    rng_states_by_arm: dict[str, list[Any]] = {}
    shapes_by_arm: dict[str, list[Any]] = {}
    for arm in arm_order:
        result = run_arm(
            arm, tasks, baseline, swm, cfg, dataset, process, transforms, torch, spt,
            official, reference, student, old_router, out,
        )
        arm_outputs[arm] = result["result"]
        episode_rows.extend(result["rows"])
        rng_states_by_arm[arm] = result["rng_states"]
        shapes_by_arm[arm] = [item["candidate_shape"] for item in result["solve_records"]]
        if reference_prepared is None:
            reference_prepared = result["prepared"]
        details = {
            key: semantic_tensor_equal(reference_prepared.get(key), result["prepared"].get(key), key)
            for key in OBS_KEYS
        }
        observation_checks[arm] = details
        arm_outputs[arm]["prepared_observation_equal_to_first_arm"] = details
        print(json.dumps({
            "arm": arm,
            "successes": arm_outputs[arm]["successes"],
            "teacher_calls": arm_outputs[arm]["teacher_cost_calls"],
            "solve_calls": arm_outputs[arm]["total_cem_solve_calls"],
            "planner_s": round(arm_outputs[arm]["planner_solve_seconds"], 3),
        }, ensure_ascii=False), flush=True)

    common_count = min((len(values) for values in rng_states_by_arm.values()), default=0)
    rng_records = []
    rng_prefix_equal = common_count > 0
    for ordinal in range(common_count):
        states_equal = all(
            torch.equal(rng_states_by_arm[arm_order[0]][ordinal], rng_states_by_arm[arm][ordinal])
            for arm in arm_order[1:]
        )
        shapes = {arm: shapes_by_arm[arm][ordinal] for arm in arm_order}
        shapes_equal = len({json.dumps(shape, sort_keys=True) for shape in shapes.values()}) == 1
        rng_prefix_equal = rng_prefix_equal and states_equal and shapes_equal
        rng_records.append({
            "solve_ordinal": ordinal,
            "generator_state_equal_across_arms": states_equal,
            "candidate_noise_shape_equal_across_arms": shapes_equal,
            "candidate_tensor_shapes": shapes,
        })

    baseline_vector_match = arm_outputs["teacher_only"]["episode_successes"] == baseline_successes
    mismatch_indices = [
        index for index, (left, right) in enumerate(zip(arm_outputs["teacher_only"]["episode_successes"], baseline_successes, strict=True))
        if left != right
    ]
    teacher_mismatches = [
        {
            **tasks[index],
            "task_index": index,
            "baseline_success": baseline_successes[index],
            "diagnostic_teacher_success": arm_outputs["teacher_only"]["episode_successes"][index],
        }
        for index in mismatch_indices
    ]
    task_keys_exact = all(
        all(episode_rows[arm_index * 50 + index][key] == tasks[index][key] for key in ("row_index", "episode_idx", "start_step"))
        for arm_index, arm in enumerate(arm_order)
        for index in range(50)
    )
    observations_equal = all(
        set(OBS_KEYS).issubset(arm_outputs[arm]["prepared_observation_keys"])
        and all(value.values())
        for arm, value in observation_checks.items()
    )
    schedules_exact = all(value["schedule_exact"] for value in arm_outputs.values())
    finite = all(value["finite"] for value in arm_outputs.values()) and all_finite(arm_outputs)
    outcome_keys = [(row["arm"], int(row["task_index"])) for row in episode_rows]
    complete = (
        len(episode_rows) == 200
        and len(set(outcome_keys)) == 200
        and all(sum(1 for row in episode_rows if row["arm"] == arm) == 50 for arm in ARMS)
    )
    trace_valid = all(value["distribution_trace_valid"] for value in arm_outputs.values())
    validity = all((baseline_vector_match, task_keys_exact, observations_equal, schedules_exact, finite, complete, rng_prefix_equal, trace_valid))

    early = arm_outputs["early_teacher7"]
    late = arm_outputs["late_teacher7"]
    student_result = arm_outputs["student_only"]
    teacher = arm_outputs["teacher_only"]
    primary_test = exact_cluster_sign_flip(episode_rows, "early_teacher7", "late_teacher7")
    early_late_delta = early["successes"] - late["successes"]
    primary_p = primary_test["exact_two_sided_p"]
    if early_late_delta >= 5 and primary_p < 0.05:
        phase_verdict = "EARLY_ANCHORING_FAVORED"
        favored_arm = "early_teacher7"
    elif early_late_delta <= -5 and primary_p < 0.05:
        phase_verdict = "LATE_RERANKING_FAVORED"
        favored_arm = "late_teacher7"
    else:
        phase_verdict = "INCONCLUSIVE"
        favored_arm = None

    student_tests = {
        arm: exact_cluster_sign_flip(episode_rows, arm, "student_only")
        for arm in ("early_teacher7", "late_teacher7")
    }
    usefulness_gates = {}
    for arm in ("early_teacher7", "late_teacher7"):
        result = arm_outputs[arm]
        paired_test = student_tests[arm]
        delta = result["successes"] - student_result["successes"]
        time_ratio = result["planner_solve_seconds"] / teacher["planner_solve_seconds"] if teacher["planner_solve_seconds"] > 0 else None
        usefulness_gates[arm] = {
            "success_delta_vs_student": delta,
            "paired_exact_test_vs_student": paired_test,
            "legacy_practical_gain_pass": bool(delta >= 5 and paired_test["exact_two_sided_p"] < 0.05),
            "teacher_gap": teacher["successes"] - result["successes"],
            "legacy_teacher_gap_within_5_pass": bool(result["successes"] >= teacher["successes"] - 5),
            "planner_time_ratio_to_teacher": time_ratio,
            "legacy_latency_ratio_at_most_0_70_pass": bool(time_ratio is not None and math.isfinite(time_ratio) and time_ratio <= 0.70),
        }
    checks = {
        "validity": validity,
        "teacher_only_success_vector_exactly_matches_baseline": baseline_vector_match,
        "teacher_only_vector_mismatch_indices": mismatch_indices,
        "exact_50_tasks_and_200_outcomes": complete,
        "all_arm_task_outcome_keys_unique": len(set(outcome_keys)) == 200,
        "task_keys_and_order_match": task_keys_exact,
        "prepared_observation_keys_pair_exactly": observations_equal,
        "all_schedules_and_finite_outputs": bool(schedules_exact and finite),
        "all_distribution_scalar_traces_complete_and_finite": trace_valid,
        "native_solver_generator_state_and_shape_common_prefix": rng_prefix_equal,
        "primary_early7_vs_late7_phase_verdict": phase_verdict,
        "phase_favored_schedule_legacy_usefulness_gate": None if favored_arm is None else usefulness_gates[favored_arm]["legacy_practical_gain_pass"],
        "phase_favored_schedule_legacy_teacher_gap_gate": None if favored_arm is None else usefulness_gates[favored_arm]["legacy_teacher_gap_within_5_pass"],
        "phase_favored_schedule_legacy_latency_gate": None if favored_arm is None else usefulness_gates[favored_arm]["legacy_latency_ratio_at_most_0_70_pass"],
    }
    if not validity:
        overall = "FAIL_CLOSED"
    else:
        overall = phase_verdict

    trace_summary_by_arm = {
        arm: arm_outputs[arm]["distribution_trace_mean_by_round"]
        for arm in ARMS
    }

    rng_summary = {
        "common_prefix_solve_count": common_count,
        "per_arm_solve_counts": {arm: len(rng_states_by_arm[arm]) for arm in ARMS},
        "common_prefix_equal": rng_prefix_equal,
        "solves": rng_records,
    }
    atomic_json(out / "rng_pairing.json", rng_summary)
    atomic_json(out / "phase_timing_summary.json", {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "overall": overall,
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "baseline_job_id": BASELINE_JOB,
        "baseline_successes": int(baseline_summary["successes"]),
        "baseline_teacher_success_vector": baseline_successes,
        "run_order": arm_order,
        "task_count": len(tasks),
        "outcome_count": len(episode_rows),
        "source_episode_clusters": len({row["episode_idx"] for row in tasks}),
        "successes_by_arm": {arm: arm_outputs[arm]["successes"] for arm in ARMS},
        "teacher_only_success_vector_matches_baseline": baseline_vector_match,
        "teacher_only_mismatch_indices": mismatch_indices,
        "teacher_only_mismatches": teacher_mismatches,
        "primary_comparison": "early_teacher7_vs_late_teacher7",
        "phase_verdict": phase_verdict,
        "phase_favored_arm": favored_arm,
        "early_teacher7_vs_late_teacher7_success_delta": early_late_delta,
        "cluster_sign_flip_early_vs_late": primary_test,
        "secondary_cluster_sign_flip_vs_student": student_tests,
        "legacy_usefulness_gates_by_phase_arm": usefulness_gates,
        "planner_seconds_by_arm": {arm: arm_outputs[arm]["planner_solve_seconds"] for arm in ARMS},
        "distribution_trace_mean_by_round_by_arm": trace_summary_by_arm,
        "teacher_calls_by_arm": {arm: arm_outputs[arm]["teacher_cost_calls"] for arm in ARMS},
        "total_evaluation_seconds_by_arm": {arm: arm_outputs[arm]["evaluation_seconds"] for arm in ARMS},
        "peak_cuda_allocated_bytes_by_arm": {arm: arm_outputs[arm]["peak_cuda_allocated_bytes"] for arm in ARMS},
        "peak_cuda_reserved_bytes_by_arm": {arm: arm_outputs[arm]["peak_cuda_reserved_bytes"] for arm in ARMS},
        "progress_capture_errors": {arm: arm_outputs[arm]["progress_capture_error"] for arm in ARMS},
        "checks": checks,
        "arms": arm_outputs,
        "source": {
            "dataset_path": str(dataset_path),
            "dataset_size_bytes": dataset_path.stat().st_size,
            "teacher_checkpoint_path": str(teacher_path),
            "teacher_checkpoint_size_bytes": teacher_path.stat().st_size,
            "student_checkpoint_path": str(student_path),
            "student_checkpoint_metadata": jsonable(student_meta),
            "baseline_task_artifact": str(args.baseline_artifacts / "selected_tasks.json"),
            "conditional_freeze": str(args.freeze),
        },
        "claim_boundary": "Protocol-aligned upstream dataset evaluation with a frozen early-vs-late teacher timing diagnostic; not a verbatim upstream eval.py run or a random-reset simulator benchmark.",
    })
    print(json.dumps({"overall": overall, "checks": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
