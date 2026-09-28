#!/usr/bin/env python3
"""Evaluate one late teacher re-score on the frozen LeWM PushT dataset tasks."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ARM = "final_only_round30"
TEACHER_ROUNDS = (30,)
REFERENCE_ARMS = ("student_only", "late_teacher7")
REQUIRED_OBS_KEYS = ("pixels", "goal", "state", "goal_state", "proprio", "action")
BASELINE_JOB = "25534994.pbs101"
CHECKPOINT_ATTRIBUTION_JOB = "25536308.pbs101"
PHASE_TIMING_JOB = "25536049.pbs101"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-dir", "baseline-artifacts", "lewm-root", "stablewm-root",
        "control-root", "staged-home", "cache-root", "checkpoint", "probe",
        "reference-outcomes", "freeze", "out",
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
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes:
        raise RuntimeError(f"host {host} is not present in PBS_NODEFILE")
    return host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def exact_mcnemar(rows: list[dict[str, Any]], left_arm: str, right_arm: str) -> dict[str, Any]:
    by_arm_and_task = {
        (row["arm"], int(row["task_index"])): row
        for row in rows if row["arm"] in (left_arm, right_arm)
    }
    left_only = right_only = paired = 0
    for row in rows:
        if row["arm"] != left_arm:
            continue
        other = by_arm_and_task.get((right_arm, int(row["task_index"])))
        if other is None:
            raise RuntimeError(f"missing paired row for {right_arm}, task {row['task_index']}")
        for key in ("row_index", "episode_idx", "start_step"):
            if int(row[key]) != int(other[key]):
                raise RuntimeError(f"paired task identity differs for task {row['task_index']}: {key}")
        paired += 1
        left_only += int(bool(row["success"]) and not bool(other["success"]))
        right_only += int(bool(other["success"]) and not bool(row["success"]))
    if paired != 50:
        raise RuntimeError(f"expected exactly 50 paired tasks, got {paired}")
    discordant = left_only + right_only
    tail = min(left_only, right_only)
    p_value = min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)) if discordant else 1.0
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "paired_tasks": paired,
        "left_success_only": left_only,
        "right_success_only": right_only,
        "discordant_pairs": discordant,
        "success_delta": left_only - right_only,
        "exact_two_sided_p": p_value,
        "method": "exact two-sided McNemar test, conditional binomial tails",
    }


def finite_metrics(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_metrics(item) for item in value)
    if isinstance(value, dict):
        return all(finite_metrics(item) for item in value.values())
    return False


def task_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(row["row_index"]), int(row["episode_idx"]), int(row["start_step"])


def validate_reference(reference: Mapping[str, Any], tasks: list[dict[str, int]], baseline_teacher: list[bool]) -> dict[str, list[bool]]:
    if reference.get("schema") != "lewm-pusht-final-only-teacher.reference-outcomes":
        raise RuntimeError("reference outcome schema mismatch")
    if reference.get("task_source_job_id") != BASELINE_JOB or reference.get("task_count") != 50:
        raise RuntimeError("reference task source does not match the frozen baseline")
    if [task_key(row) for row in reference.get("tasks", [])] != [task_key(row) for row in tasks]:
        raise RuntimeError("reference outcomes use different task identities or order")
    outcomes = reference.get("outcomes")
    if not isinstance(outcomes, Mapping):
        raise RuntimeError("reference outcomes are missing")
    expected = {
        "student_only": (CHECKPOINT_ATTRIBUTION_JOB, "treatment_step1000", 11),
        "late_teacher7": (PHASE_TIMING_JOB, "late_teacher7", 25),
        "teacher_only_positive_control": (CHECKPOINT_ATTRIBUTION_JOB, "teacher_only", 49),
    }
    vectors: dict[str, list[bool]] = {}
    for name, (job_id, source_arm, expected_count) in expected.items():
        item = outcomes.get(name)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"missing frozen comparison vector {name}")
        if item.get("source_job_id") != job_id or item.get("source_arm") != source_arm:
            raise RuntimeError(f"reference provenance mismatch for {name}")
        vector = item.get("successes")
        if not isinstance(vector, list) or len(vector) != 50 or sum(bool(v) for v in vector) != expected_count:
            raise RuntimeError(f"reference vector count/length mismatch for {name}")
        vectors[name] = [bool(v) for v in vector]
    if vectors["teacher_only_positive_control"] != baseline_teacher:
        raise RuntimeError("frozen teacher positive-control vector differs from baseline")
    if not reference.get("validated_sources", {}).get("phase_timing_validity"):
        raise RuntimeError("phase-timing reference job is not marked valid")
    if not reference.get("validated_sources", {}).get("checkpoint_attribution_validity"):
        raise RuntimeError("checkpoint-attribution reference job is not marked valid")
    return vectors


def main() -> None:
    args = parse_args()
    host = require_compute_node()
    out = args.out.resolve(strict=True)
    output_names = ("final_only_summary.json", "final_only_episodes.jsonl", "final_only_solve_records.jsonl", "comparison_tests.json", "rng_state_records.json")
    if out == Path("/") or any((out / name).exists() for name in output_names):
        raise FileExistsError("output directory is invalid or already contains final-only results")

    freeze = read_json(args.freeze.resolve(strict=True))
    if freeze.get("schema") != "lewm-pusht-final-only-teacher.freeze" or freeze.get("revision") != 1 or freeze.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise RuntimeError("final-only freeze identity mismatch")
    reference_path = args.reference_outcomes.resolve(strict=True)
    if str(reference_path).replace("\\", "/") != freeze["task_and_protocol"]["reference_outcomes_path"]:
        raise RuntimeError("reference outcome path differs from freeze")

    # Reuse the exact baseline validator and route runner from the successful
    # checkpoint-attribution task; no shared router defaults are changed.
    baseline_dir = args.baseline_dir.resolve(strict=True)
    control_root = args.control_root.resolve(strict=True)
    lewm_root = args.lewm_root.resolve(strict=True)
    stablewm_root = args.stablewm_root.resolve(strict=True)
    staged_home = args.staged_home.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    probe = args.probe.resolve(strict=True)
    baseline_artifacts = args.baseline_artifacts.resolve(strict=True)
    task_protocol = freeze["task_and_protocol"]
    teacher_identity = freeze["provenance"]["teacher"]
    if str(checkpoint).replace("\\", "/") != freeze["provenance"]["student_checkpoint"]["path"]:
        raise RuntimeError("student checkpoint path differs from freeze")
    if not all(path.is_file() for path in (checkpoint, staged_home / "pusht_expert_train.h5", staged_home / "pusht" / "lewm_object.ckpt", probe)):
        raise FileNotFoundError("a frozen input file is missing")
    if (staged_home / "pusht_expert_train.h5").stat().st_size != int(task_protocol["dataset_expected_size_bytes"]):
        raise RuntimeError("frozen HDF5 size mismatch")
    if (staged_home / "pusht" / "lewm_object.ckpt").stat().st_size != int(teacher_identity["expected_size_bytes"]):
        raise RuntimeError("frozen teacher checkpoint size mismatch")
    if not (cache_root / "datasets" / "pusht_expert_train.h5").is_file() or not (cache_root / "pusht" / "lewm_object.ckpt").is_file():
        raise RuntimeError("job-private data/teacher cache links are missing")
    if (cache_root / "datasets" / "pusht_expert_train.h5").resolve(strict=True) != (staged_home / "pusht_expert_train.h5").resolve(strict=True):
        raise RuntimeError("HDF5 cache path differs from frozen source")
    if (cache_root / "pusht" / "lewm_object.ckpt").resolve(strict=True) != (staged_home / "pusht" / "lewm_object.ckpt").resolve(strict=True):
        raise RuntimeError("teacher cache path differs from frozen source")

    sys.path[:0] = [
        str(lewm_root), str(stablewm_root), str(control_root / "lewm-transfer"),
        str(baseline_dir), str(control_root / "lewm-transfer" / "adaptive-teacher-schedule"),
        str(control_root / "lewm-transfer" / "cem-distribution-distill"),
        str(control_root / "lewm-transfer" / "official-pusht-cem"),
        str(control_root / "lewm-transfer" / "checkpoint-attribution"),
    ]
    os.environ["STABLEWM_HOME"] = str(cache_root)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Dataset, teacher, and model reads happen only after the PBS allocation guard.
    import hdf5plugin
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import jepa
    import torch
    from omegaconf import OmegaConf
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    import run_dataset_teacher_baseline as baseline
    import run_checkpoint_attribution as shared
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

    shared.SCHEDULES = {ARM: TEACHER_ROUNDS}
    shared.CEM_STEPS = 30
    baseline_summary, tasks, baseline_teacher = shared.validate_input(args, freeze)
    reference = read_json(reference_path)
    reference_vectors = validate_reference(reference, tasks, baseline_teacher)
    reference_rows: list[dict[str, Any]] = []
    for arm in REFERENCE_ARMS:
        vector = reference_vectors[arm]
        for index, (task, success) in enumerate(zip(tasks, vector, strict=True)):
            reference_rows.append({"arm": arm, "task_index": index, **task, "success": success})

    cfg = baseline.compose_pinned_config(lewm_root)
    if int(cfg.world.max_episode_steps) != 100 or int(cfg.eval.eval_budget) != 50 or int(cfg.eval.goal_offset_steps) != 25:
        raise RuntimeError("pinned dataset evaluation settings drifted")
    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=cache_root
    )
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected_data = dataset.get_row_data(np.asarray([row["row_index"] for row in tasks], dtype=np.int64))
    for index, row in enumerate(tasks):
        if int(selected_data[episode_column][index]) != row["episode_idx"] or int(selected_data["step_idx"][index]) != row["start_step"]:
            raise RuntimeError(f"frozen baseline row/task identity differs in HDF5 at task {index}")

    schedule, _, reference_model, _ = importlib.import_module("run_official_pusht_cem").load_modules(control_root, lewm_root)
    schedule.validate_interface(reference_model, probe)
    official = load_official_checkpoint(cache_root)
    official.interpolate_pos_encoding = True
    official.requires_grad_(False)
    student, checkpoint_meta = schedule.load_main_student(reference_model, checkpoint)
    student = student.to("cuda").eval()
    student.requires_grad_(False)
    expected_provenance = freeze["provenance"]["student_checkpoint"]
    for key in ("source", "arm", "extra_updates"):
        if checkpoint_meta["provenance"].get(key) != expected_provenance[key]:
            raise RuntimeError(f"treatment checkpoint provenance mismatch: {key}")
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if parameter_count != int(expected_provenance["parameter_count"]):
        raise RuntimeError("treatment student parameter count differs from freeze")

    arm_output_dir = out / "routed_arm_capture"
    arm_output_dir.mkdir()
    tick = time.perf_counter()
    outcome = shared.run_arm(
        ARM, tasks, baseline, swm, cfg, dataset, process, transforms, torch, spt,
        official, reference_model, student,
        importlib.import_module("run_official_pusht_cem"), arm_output_dir,
    )
    torch.cuda.synchronize()
    total_arm_seconds = time.perf_counter() - tick
    arm = outcome["result"]
    solve_records = outcome["solve_records"]
    rng_states = outcome["rng_states"]
    route_records_ok = bool(solve_records) and all(
        shared.solve_record_matches_schedule(record, ARM) for record in solve_records
    )
    expected_calls = sum(int(record["batch_count"]) for record in solve_records)
    route_count_exact = arm["teacher_cost_calls"] == expected_calls
    rng_records_complete = len(rng_states) == len(solve_records) and all(
        int(record["solver_seed"]) == 42 for record in solve_records
    )
    atomic_json(out / "rng_state_records.json", {
        "solver_seed": 42,
        "solve_count": len(solve_records),
        "records": [
            {
                "solve_ordinal": index,
                "solver_seed": int(record["solver_seed"]),
                "candidate_shape": record["candidate_shape"],
                "generator_state_before_solve": state.tolist(),
            }
            for index, (record, state) in enumerate(zip(solve_records, rng_states, strict=True))
        ],
    })

    prepared = outcome["prepared"] or {}
    observation_shapes: dict[str, dict[str, Any]] = {}
    observations_valid = set(REQUIRED_OBS_KEYS).issubset(prepared)
    for key in REQUIRED_OBS_KEYS:
        value = prepared.get(key)
        if not torch.is_tensor(value):
            observations_valid = False
            continue
        finite = bool(torch.isfinite(value).all())
        no_infinity = not bool(torch.isinf(value).any())
        if key != "action" and not finite:
            observations_valid = False
        if key == "action" and not no_infinity:
            observations_valid = False
        observation_shapes[key] = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "numel": int(value.numel()), "finite": finite,
            "nan_count": int(torch.isnan(value).sum()) if value.is_floating_point() else 0,
            "no_infinity": no_infinity,
        }

    generated_rows = outcome["rows"]
    complete = (
        len(generated_rows) == 50
        and len({(row["row_index"], row["episode_idx"], row["start_step"]) for row in generated_rows}) == 50
        and [task_key(row) for row in generated_rows] == [task_key(row) for row in tasks]
    )
    primary = exact_mcnemar(generated_rows + reference_rows, ARM, "student_only")
    late7_secondary = exact_mcnemar(generated_rows + reference_rows, ARM, "late_teacher7")
    student_gain = arm["successes"] - sum(reference_vectors["student_only"])
    late7_reference_count = sum(reference_vectors["late_teacher7"])
    primary_gain_gate = student_gain >= 5 and primary["exact_two_sided_p"] < 0.05
    if primary_gain_gate and arm["successes"] >= 20:
        verdict = "FINAL_ONLY_ENGINEERING_SUFFICIENCY_GATE"
    elif primary_gain_gate:
        verdict = "FINAL_ONLY_GAIN_BUT_BELOW_LATE7_POINT_MARGIN"
    elif late7_reference_count - sum(reference_vectors["student_only"]) >= 5:
        verdict = "FINAL_ONLY_GAIN_NOT_DEMONSTRATED_MULTIROUND_REMAINS_HYPOTHESIS"
    else:
        verdict = "INCONCLUSIVE"

    all_finite = bool(arm["finite"] and finite_metrics(arm) and all(
        math.isfinite(float(record["wall_s"])) and bool(record["output_finite"])
        and all(record["round_finite"]) for record in solve_records
    ))
    schedule_exact = bool(arm["schedule_exact"] and route_records_ok and route_count_exact)
    validity = all((complete, observations_valid, schedule_exact, rng_records_complete, all_finite, bool(reference_vectors["teacher_only_positive_control"] == baseline_teacher)))

    episode_path = out / "final_only_episodes.jsonl"
    with episode_path.open("w", encoding="utf-8") as handle:
        for row in generated_rows:
            index = int(row["task_index"])
            enriched = {
                **row,
                "student_only_reference_success": reference_vectors["student_only"][index],
                "late_teacher7_reference_success": reference_vectors["late_teacher7"][index],
                "teacher_only_positive_control_success": reference_vectors["teacher_only_positive_control"][index],
            }
            handle.write(json.dumps(enriched, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    record_path = out / "final_only_solve_records.jsonl"
    with record_path.open("w", encoding="utf-8") as handle:
        for record in solve_records:
            handle.write(json.dumps({"arm": ARM, **record}, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_json(out / "comparison_tests.json", {"primary_final_only_vs_student": primary, "secondary_final_only_vs_late7_reference": late7_secondary})

    checks = {
        "validity": validity,
        "exact_50_tasks_in_frozen_order": complete,
        "hdf5_episode_and_step_identity_matches": True,
        "student_and_late7_reference_vectors_match_frozen_sources": True,
        "teacher_positive_control_reference_matches_baseline": reference_vectors["teacher_only_positive_control"] == baseline_teacher,
        "all_six_prepared_observation_keys_present_and_semantically_valid": observations_valid,
        "round30_teacher_only_schedule_each_real_batch": schedule_exact,
        "teacher_calls_equal_actual_batch_count": route_count_exact,
        "native_seed_and_pre_solve_generator_state_records_complete": rng_records_complete,
        "all_outputs_and_timings_finite": all_finite,
        "treatment_checkpoint_strict_provenance_and_parameter_count": parameter_count == int(expected_provenance["parameter_count"]),
        "pbs_compute_node_guard_passed": bool(os.environ.get("PBS_JOBID")) and bool(host),
    }
    overall = verdict if validity else "FAIL_CLOSED"
    result = {
        "schema": "lewm-pusht-final-only-teacher.result",
        "status": "COMPLETED",
        "overall": overall,
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "task_count": len(tasks),
        "outcome_count": len(generated_rows),
        "successes": arm["successes"],
        "success_rate_percent": arm["success_rate_percent"],
        "student_only_reference_successes": sum(reference_vectors["student_only"]),
        "late_teacher7_reference_successes": late7_reference_count,
        "late_teacher7_variation_note": "25536049.pbs101=25/50; another valid run 25535873.pbs101=26/50; descriptive reference only.",
        "teacher_only_positive_control_successes": sum(reference_vectors["teacher_only_positive_control"]),
        "student_gain": student_gain,
        "primary_exact_mcnemar_final_only_vs_student": primary,
        "secondary_exact_mcnemar_final_only_vs_late7_reference": late7_secondary,
        "primary_practical_gain_gate": primary_gain_gate,
        "teacher_cost_calls": arm["teacher_cost_calls"],
        "total_cem_solve_calls": arm["total_cem_solve_calls"],
        "actual_batch_count": expected_calls,
        "schedule_exact": schedule_exact,
        "planner_solve_seconds_this_job_only": arm["planner_solve_seconds"],
        "evaluation_seconds_this_job_only": arm["evaluation_seconds"],
        "arm_wrapper_seconds": total_arm_seconds,
        "latency_boundary": "This single-arm timing is descriptive only and is not directly compared to earlier multi-arm jobs.",
        "native_solver_seed": 42,
        "candidate_shapes_by_solve": [record["candidate_shape"] for record in solve_records],
        "observation_shape_metadata": observation_shapes,
        "checkpoint": checkpoint_meta,
        "dataset_path": str(staged_home / "pusht_expert_train.h5"),
        "dataset_size_bytes": (staged_home / "pusht_expert_train.h5").stat().st_size,
        "teacher_checkpoint_path": str(staged_home / "pusht" / "lewm_object.ckpt"),
        "teacher_checkpoint_size_bytes": (staged_home / "pusht" / "lewm_object.ckpt").stat().st_size,
        "reference_sources": reference["validated_sources"],
        "reference_outcomes_file": str(reference_path),
        "checks": checks,
        "claim_boundary": "One round-30 teacher re-score on the exact frozen upstream-protocol-aligned dataset tasks. Late7 is one variable historical reference; this is not statistical equivalence, random-reset PushT, or a direct cross-job speed comparison.",
    }
    atomic_json(out / "final_only_summary.json", result)
    print(json.dumps({"overall": overall, "successes": arm["successes"], "primary": primary, "checks": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
