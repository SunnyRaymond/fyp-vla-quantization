#!/usr/bin/env python3
"""Run a paired 50-task dataset evaluation for the terminal-response student."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_SCHEMA = "lewm-pusht-terminal-response-late7.result"
FREEZE_SCHEMA = "lewm-pusht-terminal-response-late7.freeze"
TRAINING_CHECKPOINT_SCHEMA = "lewm-recurrent-student.terminal-response-loss-freeze.checkpoint"
ARMS = ("student_only", "teacher_only", "late_teacher7")
EXPECTED_TERMINAL_PROVENANCE = {
    "training_rows_source_job": "25213164.pbs101",
    "initialization_seed": 20300901,
    "training_seed": 20300902,
    "context_schedule_seed": 20300904,
    "candidate_slate_seed": 20300905,
    "teacher_targets_reused_from_cached_rows": True,
    "training_arm": "terminal_response",
    "response_loss_weight": 0.01,
    "response_denominator_floor": 1e-6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-dir", "baseline-artifacts", "lewm-root", "stablewm-root",
        "control-root", "staged-home", "cache-root", "candidate-checkpoint",
        "probe", "freeze", "out",
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


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def validate_new_freeze(
    freeze: Mapping[str, Any],
    candidate_path: Path,
    tasks: list[dict[str, int]],
    baseline_successes: list[bool],
    baseline_tasks_path: Path,
    teacher_checkpoint_path: str,
) -> list[str]:
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("revision") != 1
        or freeze.get("status") != "frozen_before_evaluation"
    ):
        raise RuntimeError("terminal-response-late7 freeze identity mismatch")
    frozen_candidate = str(freeze.get("candidate_checkpoint_path", "")).replace("\\", "/")
    if frozen_candidate != str(candidate_path).replace("\\", "/"):
        raise RuntimeError("candidate checkpoint path differs from the new freeze")
    checkpoint_record = freeze.get("candidate_checkpoint_provenance", {})
    if checkpoint_record != {
        "job_id": "25549480.pbs101",
        "filename": "terminal_response_step3000.pt",
        "source_arm": "terminal_response",
        "updates": 3000,
        "architecture": "LeWMCompactRecurrentTransitionStudent h256 shared recurrent, no attention/conditioner, autoregressive predicted-latent feedback",
    }:
        raise RuntimeError("candidate checkpoint provenance record drifted")
    if candidate_path.name != "terminal_response_step3000.pt" or "25549480.pbs101" not in candidate_path.parts:
        raise RuntimeError("candidate path is not the frozen 25549480 terminal-response checkpoint")
    if freeze.get("arms") != list(ARMS):
        raise RuntimeError("frozen arms must be student_only, teacher_only, late_teacher7")
    if str(freeze.get("teacher_checkpoint", "")).replace("\\", "/") != teacher_checkpoint_path.replace("\\", "/"):
        raise RuntimeError("teacher checkpoint differs from the original Stage 2 freeze")

    task_set = freeze.get("task_set", {})
    if (
        task_set.get("source_job_id") != "25534994.pbs101"
        or int(task_set.get("selection_rng_seed", -1)) != 42
        or int(task_set.get("task_count", -1)) != 50
        or int(task_set.get("unique_source_episodes", -1)) != 50
        or task_set.get("task_keys") != ["row_index", "episode_idx", "start_step"]
        or str(task_set.get("source_path", "")).replace("\\", "/")
        != str(baseline_tasks_path).replace("\\", "/")
    ):
        raise RuntimeError("new freeze task source or selection seed drifted")
    frozen_tasks = task_set.get("tasks")
    if not isinstance(frozen_tasks, list):
        raise RuntimeError("new freeze lacks its materialized 50-task list")
    normalized_frozen = [
        {key: int(row[key]) for key in ("row_index", "episode_idx", "start_step")}
        for row in frozen_tasks
    ]
    if normalized_frozen != tasks or len(normalized_frozen) != 50:
        raise RuntimeError("new freeze task keys/order differ from the baseline selected_tasks.json")

    expected_teacher = freeze.get("validity", {}).get("teacher_only_expected_successes")
    if not isinstance(expected_teacher, list) or [bool(x) for x in expected_teacher] != baseline_successes:
        raise RuntimeError("new freeze teacher-only vector differs from the successful baseline")
    if len(expected_teacher) != 50 or sum(bool(x) for x in expected_teacher) != 49:
        raise RuntimeError("new freeze expected teacher vector must be the frozen 49/50 result")

    schedules = freeze.get("schedule_rounds", {})
    if (
        schedules.get("student_only") != []
        or schedules.get("teacher_only") != "all rounds 1-30"
        or schedules.get("late_teacher7") != [24, 25, 26, 27, 28, 29, 30]
        or schedules.get("calls_per_batch") != {"student_only": 0, "late_teacher7": 7, "teacher_only": 30}
    ):
        raise RuntimeError("frozen late_teacher7 rounds must be 24-30 on every CEM batch")

    evaluation = freeze.get("dataset_evaluation", {})
    if (
        evaluation.get("dataset") != "pusht_expert_train"
        or int(evaluation.get("goal_offset_steps", -1)) != 25
        or int(evaluation.get("eval_budget", -1)) != 50
        or int(evaluation.get("world_max_episode_steps", -1)) != 100
        or evaluation.get("video", "not-null") is not None
    ):
        raise RuntimeError("dataset-mode evaluator contract drifted")
    cem = freeze.get("cem", {})
    if (
        int(cem.get("batch_size", -1)) != 1
        or int(cem.get("num_samples", -1)) != 300
        or int(cem.get("n_steps", -1)) != 30
        or int(cem.get("topk", -1)) != 30
        or float(cem.get("var_scale", math.nan)) != 1.0
        or int(cem.get("seed_per_arm", -1)) != 42
    ):
        raise RuntimeError("CEM solver contract drifted")
    gates = freeze.get("gates", {})
    if (
        int(gates.get("quality_gain_vs_student_only", {}).get("minimum_success_gain", -1)) != 5
        or float(gates.get("quality_gain_vs_student_only", {}).get("paired_test_alpha_two_sided", math.nan)) != 0.05
        or int(gates.get("practical_teacher_gap", {}).get("maximum_successes_below_teacher_only", -1)) != 5
        or float(gates.get("planner_latency", {}).get("maximum_ratio", math.nan)) != 0.7
    ):
        raise RuntimeError("inherited late7 quality/latency gates drifted")
    run_order = freeze.get("run_order", {})
    order = run_order.get("frozen_order")
    if not isinstance(order, list) or set(order) != set(ARMS) or len(order) != len(ARMS):
        raise RuntimeError("new freeze run_order must contain each required arm exactly once")
    randomized = list(ARMS)
    random.Random(int(run_order.get("seed", -1))).shuffle(randomized)
    if randomized != order:
        raise RuntimeError("new freeze run_order does not match its seed")
    return order


def load_terminal_response_student(
    reference: Any,
    checkpoint_path: Path,
    training_freeze_path: Path,
    training_freeze: Mapping[str, Any],
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError("terminal-response checkpoint is not a mapping")
    if raw.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
        raise RuntimeError("terminal-response checkpoint schema mismatch")
    if raw.get("arm") != "terminal_response" or int(raw.get("step", -1)) != 3000:
        raise RuntimeError("checkpoint must be terminal_response arm at step 3000")
    if raw.get("architecture") != "LeWMCompactRecurrentTransitionStudent h256":
        raise RuntimeError("terminal-response checkpoint architecture mismatch")
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("terminal-response checkpoint provenance is missing")
    if str(provenance.get("freeze", "")).replace("\\", "/") != str(training_freeze_path).replace("\\", "/"):
        raise RuntimeError("checkpoint training-freeze provenance mismatch")
    expected_rows = training_freeze.get("source", {}).get("balanced_rows_path")
    if not expected_rows or str(provenance.get("training_rows", "")).replace("\\", "/") != str(expected_rows).replace("\\", "/"):
        raise RuntimeError("checkpoint cached training-row provenance mismatch")
    for key, expected in EXPECTED_TERMINAL_PROVENANCE.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f"terminal-response checkpoint provenance mismatch: {key}")

    state = raw.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("terminal-response checkpoint has no state_dict")
    if any(not torch.is_tensor(value) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("terminal-response state_dict has non-tensor or non-finite values")
    student = reference.base.make_student("baseline").to("cuda")
    student.load_state_dict(dict(state), strict=True)
    student.eval().requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if parameter_count != 775872:
        raise RuntimeError(f"candidate student parameter count drifted: {parameter_count}")
    return student, {
        "path": str(checkpoint_path),
        "schema": raw["schema"],
        "arm": raw["arm"],
        "step": int(raw["step"]),
        "architecture": raw["architecture"],
        "parameter_count": parameter_count,
        "provenance": dict(provenance),
    }


def main() -> None:
    args = parse_args()
    host = require_compute_node()
    out = args.out.resolve(strict=True)
    output_names = (
        "terminal_response_late7_summary.json",
        "terminal_response_late7_episodes.jsonl",
        "rng_pairing.json",
        "selected_tasks.json",
        "run_order.json",
    )
    if out == Path("/") or any((out / name).exists() for name in output_names):
        raise FileExistsError("output directory is invalid or already contains evaluation results")

    transfer_dir = args.control_root.resolve(strict=True) / "lewm-transfer"
    baseline_dir = args.baseline_dir.resolve(strict=True)
    baseline_artifacts = args.baseline_artifacts.resolve(strict=True)
    freeze_path = args.freeze.resolve(strict=True)
    candidate_path = args.candidate_checkpoint.resolve(strict=True)
    freeze = read_json(freeze_path)
    historical_freeze = read_json(baseline_dir / "CONTINGENT_STAGE2_FREEZE.json")

    # Import only lightweight Stage 2 helpers before loading any model or dataset.
    sys.path[:0] = [str(baseline_dir), str(transfer_dir)]
    import run_dataset_paired_stage2 as stage2

    baseline_summary, tasks, baseline_successes = stage2.validate_input(
        args, historical_freeze
    )
    arm_order = validate_new_freeze(
        freeze,
        candidate_path,
        tasks,
        baseline_successes,
        baseline_artifacts / "selected_tasks.json",
        str(historical_freeze["task_and_protocol"]["teacher_checkpoint"]),
    )
    atomic_json(out / "run_order.json", {
        "seed": int(freeze["run_order"]["seed"]),
        "arms": arm_order,
        "task_count": len(tasks),
    })
    atomic_json(out / "selected_tasks.json", {
        "source_job_id": "25534994.pbs101",
        "selection_rng_seed": 42,
        "tasks": tasks,
        "source_metadata": baseline_summary["source_metadata"],
    })

    lewm_root = args.lewm_root.resolve(strict=True)
    stablewm_root = args.stablewm_root.resolve(strict=True)
    staged_home = args.staged_home.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    dataset_path = staged_home / "pusht_expert_train.h5"
    teacher_path = staged_home / "pusht" / "lewm_object.ckpt"
    if not all(path.is_file() for path in (dataset_path, teacher_path, candidate_path)):
        raise FileNotFoundError("frozen PushT dataset, teacher, or candidate checkpoint is missing")
    source = baseline_summary["source_metadata"]
    if dataset_path.stat().st_size != int(source["dataset_size_bytes"]):
        raise RuntimeError("staged HDF5 size differs from the successful teacher baseline")
    if teacher_path.stat().st_size != int(source["checkpoint_size_bytes"]):
        raise RuntimeError("staged official teacher size differs from the successful baseline")
    if historical_freeze["task_and_protocol"]["teacher_checkpoint"] != source["checkpoint_path"]:
        raise RuntimeError("teacher checkpoint differs from the original Stage 2 freeze")
    for link, target in (
        (cache_root / "datasets" / dataset_path.name, dataset_path),
        (cache_root / "pusht" / teacher_path.name, teacher_path),
    ):
        if not link.is_file() or link.resolve(strict=True) != target.resolve(strict=True):
            raise RuntimeError(f"PBS cache link does not resolve to the frozen source: {link}")

    old_router_dir = transfer_dir / "official-pusht-cem"
    sys.path[:0] = [
        str(lewm_root), str(stablewm_root), str(old_router_dir),
        str(transfer_dir / "adaptive-teacher-schedule"),
        str(transfer_dir / "cem-distribution-distill"),
    ]
    os.environ["STABLEWM_HOME"] = str(cache_root)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Heavy imports, HDF5 access, and checkpoint loading happen only after the PBS guard.
    import hdf5plugin
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import jepa
    import torch
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    import run_dataset_teacher_baseline as baseline
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
    if (
        int(cfg.world.max_episode_steps) != 100
        or int(cfg.eval.eval_budget) != 50
        or int(cfg.eval.goal_offset_steps) != 25
    ):
        raise RuntimeError("pinned World.evaluate task budget or goal offset drifted")
    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=cache_root
    )
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected_data = dataset.get_row_data(np.asarray([row["row_index"] for row in tasks], dtype=np.int64))
    for index, row in enumerate(tasks):
        if int(selected_data[episode_column][index]) != row["episode_idx"] or int(selected_data["step_idx"][index]) != row["start_step"]:
            raise RuntimeError(f"baseline row/task identity differs from staged HDF5 at task {index}")

    schedule, _, reference, _ = old_router.load_modules(args.control_root.resolve(strict=True), lewm_root)
    schedule.validate_interface(reference, args.probe.resolve(strict=True))
    training_freeze_path = transfer_dir / "terminal-response-loss" / "FREEZE.json"
    training_freeze = read_json(training_freeze_path.resolve(strict=True))
    student, candidate_meta = load_terminal_response_student(
        reference, candidate_path, training_freeze_path.resolve(strict=True), training_freeze, torch
    )
    official = load_official_checkpoint(cache_root)
    official.interpolate_pos_encoding = True
    official.requires_grad_(False)

    arm_outputs: dict[str, dict[str, Any]] = {}
    episode_rows: list[dict[str, Any]] = []
    reference_prepared = None
    observation_checks: dict[str, dict[str, bool]] = {}
    rng_states_by_arm: dict[str, list[Any]] = {}
    shapes_by_arm: dict[str, list[Any]] = {}
    for arm in arm_order:
        result = stage2.run_arm(
            arm, tasks, baseline, swm, cfg, dataset, process, transforms, torch, spt,
            official, reference, student, old_router, out,
        )
        arm_outputs[arm] = result["result"]
        episode_rows.extend(result["rows"])
        rng_states_by_arm[arm] = result["rng_states"]
        shapes_by_arm[arm] = [item["candidate_shape"] for item in result["solve_records"]]
        if reference_prepared is None:
            reference_prepared = result["prepared"]
        checks = {
            key: stage2.semantic_tensor_equal(reference_prepared.get(key), result["prepared"].get(key), key)
            for key in stage2.OBS_KEYS
        }
        observation_checks[arm] = checks
        arm_outputs[arm]["prepared_observation_equal_to_first_arm"] = checks
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

    mismatch_indices = [
        index for index, (left, right) in enumerate(zip(arm_outputs["teacher_only"]["episode_successes"], baseline_successes, strict=True))
        if left != right
    ]
    teacher_vector_match = not mismatch_indices
    task_keys_exact = all(
        all(row[key] == tasks[row["task_index"]][key] for key in ("row_index", "episode_idx", "start_step"))
        for row in episode_rows
    )
    observations_equal = all(
        set(stage2.OBS_KEYS).issubset(arm_outputs[arm]["prepared_observation_keys"])
        and all(observation_checks[arm].values())
        for arm in arm_order
    )
    schedules_exact = all(value["schedule_exact"] for value in arm_outputs.values())
    candidate_shapes_exact = all(
        shape == [1, 300, 5, 10]
        for arm in ARMS
        for shape in shapes_by_arm[arm]
    )
    finite = all(value["finite"] for value in arm_outputs.values()) and stage2.all_finite(arm_outputs)
    complete = len(episode_rows) == 150 and all(
        sum(row["arm"] == arm for row in episode_rows) == 50 for arm in ARMS
    )
    validity = all((
        teacher_vector_match,
        task_keys_exact,
        observations_equal,
        schedules_exact,
        candidate_shapes_exact,
        finite,
        complete,
        rng_prefix_equal,
    ))

    late = arm_outputs["late_teacher7"]
    student_result = arm_outputs["student_only"]
    teacher = arm_outputs["teacher_only"]
    primary_test = stage2.exact_cluster_sign_flip(episode_rows, "late_teacher7", "student_only")
    benefit_gate = late["successes"] - student_result["successes"] >= 5 and primary_test["exact_two_sided_p"] < 0.05
    teacher_gap = late["successes"] >= teacher["successes"] - 5
    latency_ratio = late["planner_solve_seconds"] / teacher["planner_solve_seconds"] if teacher["planner_solve_seconds"] > 0 else None
    latency_gate = latency_ratio is not None and math.isfinite(latency_ratio) and latency_ratio <= 0.70
    checks = {
        "validity": validity,
        "teacher_only_success_vector_exactly_matches_baseline": teacher_vector_match,
        "teacher_only_vector_mismatch_indices": mismatch_indices,
        "exact_50_tasks_and_150_outcomes": complete,
        "task_keys_and_order_match": task_keys_exact,
        "prepared_observation_keys_pair_exactly": observations_equal,
        "frozen_candidate_shapes_exact": candidate_shapes_exact,
        "all_schedules_and_finite_outputs": bool(schedules_exact and finite),
        "native_solver_generator_state_and_shape_common_prefix": rng_prefix_equal,
        "late_teacher7_benefit_vs_candidate_student_only": bool(benefit_gate),
        "late_teacher7_teacher_gap_within_5": bool(teacher_gap),
        "late_teacher7_planner_time_ratio_at_most_0_70": bool(latency_gate),
    }
    overall = (
        "FAIL_CLOSED" if not complete or not validity
        else "PASS" if benefit_gate and teacher_gap and latency_gate
        else "FAIL"
    )

    # The shared Stage 2 run_arm helper writes its established filenames; rename
    # them inside this experiment's unique output directory before publishing summary.
    for old_name, new_name in (
        ("stage2_episodes.jsonl", "terminal_response_late7_episodes.jsonl"),
        ("stage2_solve_records.jsonl", "terminal_response_late7_solve_records.jsonl"),
    ):
        source_path = out / old_name
        if source_path.is_file():
            source_path.replace(out / new_name)

    atomic_json(out / "rng_pairing.json", {
        "common_prefix_solve_count": common_count,
        "per_arm_solve_counts": {arm: len(rng_states_by_arm[arm]) for arm in ARMS},
        "common_prefix_equal": rng_prefix_equal,
        "solves": rng_records,
    })
    atomic_json(out / "terminal_response_late7_summary.json", {
        "schema": EXPERIMENT_SCHEMA,
        "status": "COMPLETED",
        "overall": overall,
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "baseline_job_id": "25534994.pbs101",
        "baseline_successes": int(baseline_summary["successes"]),
        "baseline_teacher_success_vector": baseline_successes,
        "run_order": arm_order,
        "task_count": len(tasks),
        "outcome_count": len(episode_rows),
        "source_episode_clusters": len({row["episode_idx"] for row in tasks}),
        "successes_by_arm": {arm: arm_outputs[arm]["successes"] for arm in ARMS},
        "teacher_only_success_vector_matches_baseline": teacher_vector_match,
        "teacher_only_mismatch_indices": mismatch_indices,
        "cluster_sign_flip_late_vs_candidate_student": primary_test,
        "late_teacher7_vs_student_success_delta": late["successes"] - student_result["successes"],
        "late_teacher7_vs_teacher_success_delta": late["successes"] - teacher["successes"],
        "planner_seconds_by_arm": {arm: arm_outputs[arm]["planner_solve_seconds"] for arm in ARMS},
        "late_teacher7_to_teacher_planner_time_ratio": latency_ratio,
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
            "candidate_student_checkpoint": candidate_meta,
            "baseline_task_artifact": str(baseline_artifacts / "selected_tasks.json"),
            "new_freeze": str(freeze_path),
            "historical_stage2_freeze": str(baseline_dir / "CONTINGENT_STAGE2_FREEZE.json"),
        },
        "claim_boundary": "Three-arm paired upstream dataset-mode evaluation on the frozen 50 tasks; not a verbatim upstream eval.py run or an independent fresh cohort.",
    })

    print(json.dumps({"overall": overall, "checks": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
