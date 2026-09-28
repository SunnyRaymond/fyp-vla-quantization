#!/usr/bin/env python3
"""Validate an existing capture artifact and write a separate recovered summary."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any


SOURCE_JOB_ID = "25543623.pbs101"
SELECTION_JOB_ID = "25543404.pbs101"
EXPECTED_FAILURE = "JSONEncoder.__init__() got an unexpected keyword argument 'flush'"
EXPECTED_SAMPLES = 141
EXPECTED_TRAIN_SAMPLES = 111
EXPECTED_VALIDATION_SAMPLES = 30


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-job-id", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require_compute_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("recovery requires PBS_JOBID and an allocated PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes or any(any(token in node for token in ("login", "head", "submit")) for node in nodes):
        raise RuntimeError("hostname and PBS_NODEFILE do not establish a compute allocation")
    return job_id, host


def main() -> int:
    args = args_parser()
    recovery_job_id, host = require_compute_allocation()
    import numpy as np
    if args.source_job_id != SOURCE_JOB_ID:
        raise RuntimeError(f"recovery is frozen to capture job {SOURCE_JOB_ID}")
    source = args.source_summary.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    manifest_path = args.selection_manifest.resolve(strict=True)
    output = args.output.resolve()
    expected_dir = source.parent
    if expected_dir.name != SOURCE_JOB_ID or expected_dir.parent.name != "capture":
        raise RuntimeError("source artifacts must belong to the frozen capture job directory")
    if archive.parent != expected_dir or archive.name != "real_observation_training_data.npz":
        raise RuntimeError("archive must be the existing NPZ beside the frozen source summary")
    if output != expected_dir / "collection_summary_recovered.json":
        raise RuntimeError("recovery output must be collection_summary_recovered.json beside the source artifacts")
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite recovery output: {output}")

    original = read_json(source)
    manifest = read_json(manifest_path)
    if original.get("schema") != "lewm-pusht-student-driven-real-observation-training-collection-result-v1":
        raise RuntimeError("source collection summary schema mismatch")
    if original.get("pbs_job_id") != SOURCE_JOB_ID or original.get("status") != "FAIL_CLOSED_COLLECTION_ERROR":
        raise RuntimeError("source summary is not the frozen failed capture record")
    failure = original.get("failure", {})
    if failure.get("type") != "TypeError" or failure.get("message") != EXPECTED_FAILURE:
        raise RuntimeError("source failure is not the known final json.dumps flush-argument error")
    if manifest.get("schema") != "lewm-pusht-student-driven-real-observation-training-selection-manifest-v1" or manifest.get("schema_version") != 1:
        raise RuntimeError("selection manifest schema mismatch")
    if manifest.get("status") != "COMPLETE_METADATA_ONLY_SELECTION" or manifest.get("pbs_job_id") != SELECTION_JOB_ID:
        raise RuntimeError("selection manifest is not the frozen completed CPU selection")
    if (
        original.get("selection_job_id") != SELECTION_JOB_ID
        or original.get("selection_manifest_path") != str(manifest_path)
        or manifest.get("freeze_path") != original.get("freeze_path")
    ):
        raise RuntimeError("source summary is not bound to the frozen selection manifest")
    if len(manifest.get("tasks", [])) != 80 or manifest.get("split_counts") != {"collection_train": 64, "collection_validation": 16}:
        raise RuntimeError("frozen selection manifest must contain exactly 64 train and 16 validation tasks")

    expected_tasks = {int(row["episode_idx"]): row for row in manifest["tasks"]}
    if len(expected_tasks) != 80 or [int(row["selection_order"]) for row in manifest["tasks"]] != list(range(80)):
        raise RuntimeError("selection manifest episode identities/order are invalid")
    if sum(row["split"] == "collection_train" for row in manifest["tasks"]) != 64 or sum(row["split"] == "collection_validation" for row in manifest["tasks"]) != 16:
        raise RuntimeError("selection manifest split membership is invalid")

    episodes = original.get("episodes", [])
    if len(episodes) != 80 or int(original.get("tasks_frozen", -1)) != 80:
        raise RuntimeError("source summary must contain all 80 frozen episode records")
    episode_records: dict[int, dict[str, Any]] = {}
    sample_windows: dict[int, tuple[int, dict[str, Any], dict[str, Any]]] = {}
    for episode in episodes:
        task = episode.get("task", {})
        episode_id = int(task["episode_idx"])
        frozen = expected_tasks.get(episode_id)
        if frozen is None or episode_id in episode_records:
            raise RuntimeError("source summary has a duplicate or unfrozen episode identity")
        for key in ("episode_idx", "selection_order", "split", "row_index", "start_step"):
            if task.get(key) != frozen.get(key):
                raise RuntimeError(f"episode {episode_id} metadata differs from the selection manifest ({key})")
        episode_records[episode_id] = episode
        for window in episode.get("solve_windows", []):
            sample_index = window.get("sample_index")
            if window.get("full_window_available"):
                if sample_index is None or int(sample_index) in sample_windows:
                    raise RuntimeError("full solve windows must reference unique archive rows")
                sample_windows[int(sample_index)] = (episode_id, task, window)
            elif sample_index is not None:
                raise RuntimeError("incomplete solve windows must not reference archive rows")
    if len(episode_records) != 80:
        raise RuntimeError("source summary is missing one or more selected episode records")

    required_shapes = {
        "episode_idx": (EXPECTED_SAMPLES,),
        "selection_order": (EXPECTED_SAMPLES,),
        "split": (EXPECTED_SAMPLES,),
        "solve_start_env_step": (EXPECTED_SAMPLES,),
        "reset_seed": (EXPECTED_SAMPLES,),
        "native_cem_seed": (EXPECTED_SAMPLES,),
        "relative_env_step_indices": (EXPECTED_SAMPLES, 5),
        "z_start": (EXPECTED_SAMPLES, 192),
        "packed_actions": (EXPECTED_SAMPLES, 5, 10),
        "poststep_latents": (EXPECTED_SAMPLES, 5, 192),
        "max_plan_action_abs_error": (EXPECTED_SAMPLES,),
    }
    with np.load(archive, allow_pickle=False) as data:
        missing = sorted(set(required_shapes) - set(data.files))
        if missing:
            raise RuntimeError(f"archive is missing required arrays: {missing}")
        arrays = {name: data[name] for name in required_shapes}
    for name, shape in required_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"archive array {name} shape {arrays[name].shape} != {shape}")
    for name in ("z_start", "packed_actions", "poststep_latents", "max_plan_action_abs_error"):
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"archive array {name} contains non-finite values")
    if len(sample_windows) != EXPECTED_SAMPLES or int(original.get("sample_count", -1)) != EXPECTED_SAMPLES:
        raise RuntimeError("archive sample count differs from the 141 full-window records")
    if int(original.get("train_sample_count", -1)) != EXPECTED_TRAIN_SAMPLES or int(original.get("validation_sample_count", -1)) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("source summary sample split counts differ from the frozen output")
    if arrays["split"].tolist().count("collection_train") != EXPECTED_TRAIN_SAMPLES or arrays["split"].tolist().count("collection_validation") != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("archive sample split counts differ from the verified source summary")
    if not np.array_equal(arrays["relative_env_step_indices"], np.tile(np.array([5, 10, 15, 20, 25]), (EXPECTED_SAMPLES, 1))):
        raise RuntimeError("archive token-boundary metadata differs from the frozen action geometry")
    if not np.all(arrays["reset_seed"] == 42) or not np.all(arrays["native_cem_seed"] == 42):
        raise RuntimeError("archive reset/CEM seeds differ from the frozen capture protocol")
    errors = arrays["max_plan_action_abs_error"].astype(np.float64)
    if np.any(errors < 0) or float(errors.max()) > 1e-5:
        raise RuntimeError("archive contains a sample outside the frozen action-plan alignment gate")

    sample_ids = arrays["episode_idx"].astype(np.int64)
    sample_orders = arrays["selection_order"].astype(np.int64)
    sample_splits = arrays["split"].astype(str)
    sample_starts = arrays["solve_start_env_step"].astype(np.int64)
    if set(sample_windows) != set(range(EXPECTED_SAMPLES)):
        raise RuntimeError("source window references do not cover each archive row exactly once")
    for index in range(EXPECTED_SAMPLES):
        episode_id, task, window = sample_windows[index]
        frozen = expected_tasks[episode_id]
        if (sample_ids[index], sample_orders[index], sample_splits[index]) != (episode_id, int(frozen["selection_order"]), str(frozen["split"])):
            raise RuntimeError(f"archive sample {index} identity/split differs from its frozen task")
        if sample_starts[index] != int(window["solve_start_env_step"]) or sample_starts[index] not in (0, 25):
            raise RuntimeError(f"archive sample {index} solve-step metadata differs from its source window")
        summary_error = float(window["max_plan_action_abs_error"])
        if (
            not math.isfinite(summary_error)
            or summary_error > 1e-5
            or float(errors[index]) > 1e-5
            or not np.isclose(float(errors[index]), summary_error, rtol=1e-6, atol=1e-7)
        ):
            raise RuntimeError(f"archive sample {index} fails its recorded action-plan alignment gate")

    if original.get("reserved_holdout_evaluated_or_scored") is not False or original.get("teacher_forecasts_costs_or_shadow_calls") is not False:
        raise RuntimeError("source summary violates the frozen holdout/teacher-use boundary")
    if str(original["archive"].get("filename")) != archive.name or int(original["archive"].get("sample_count", -1)) != EXPECTED_SAMPLES:
        raise RuntimeError("source summary archive identity/count differs from the validated NPZ")

    recovered = dict(original)
    recovered.pop("failure", None)
    recovered["status"] = "COMPLETE_WITH_TERMINAL_OR_BUDGET_LIMITED_WINDOWS"
    recovered["recovery_provenance"] = {
        "schema": "lewm-pusht-student-driven-real-observation-training-summary-recovery-v1",
        "source_job_id": SOURCE_JOB_ID,
        "source_pbs_exit_status": 1,
        "source_summary": str(source),
        "source_archive": str(archive),
        "selection_manifest": str(manifest_path),
        "recovery_pbs_job_id": recovery_job_id,
        "recovery_compute_host": host,
        "failure_confirmed": EXPECTED_FAILURE,
        "checks_passed": [
            "known_final_json_dumps_flush_TypeError",
            "80_episode_ids_and_64_train_16_validation_split_match_selection_manifest",
            "141_archive_rows_and_111_train_30_validation_sample_counts",
            "all_archive_shapes_and_float_values_are_valid_and_finite",
            "all_sample_rows_match_summary_windows_and_frozen_episode_metadata",
            "all_recorded_action_plan_errors_are_at_most_1e-5",
            "reserved_holdout_and_teacher_scoring_boundaries_remain_false",
        ],
        "original_summary_and_archive_modified": False,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(recovered, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": recovered["status"], "episodes": 80, "samples": EXPECTED_SAMPLES, "recovered_summary": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
