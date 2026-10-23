"""Resumable, PBS-only calibration-cohort VQ fitting and paired action queries."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time


PROTOCOL = "fastwam-vq-calibration-suites-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
METRICS = ("motor_rmse_first10_vs_bf16", "gripper_rmse_first10_vs_bf16",
           "motor_rmse_32_vs_bf16")
INPUT_IDENTITY = ("suite", "task_id", "task_name", "domain", "phase", "demo_id",
                  "frame_index", "dimension", "variant", "variant_id", "original_task",
                  "source_index", "environment_seed")


def guard() -> None:
    job, nodefile = os.environ.get("PBS_JOBID"), os.environ.get("PBS_NODEFILE")
    if not job or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("Numerical work requires an active PBS compute allocation")
    host = socket.gethostname().split(".", 1)[0]
    allocated = {line.split(".", 1)[0] for line in Path(nodefile).read_text().split()}
    if "login" in host.lower() or host not in allocated:
        raise RuntimeError(f"PBS compute host required; host={host}")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_tensor(torch, path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def restore_final_moments(torch, path: Path, protocol: dict, cohort: str,
                          calibration: list[dict], modules: dict):
    if not path.is_file():
        return None
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Existing final moments are unreadable; refusing to recalibrate: {exc}") from exc
    expected_ids = [int(row["case_id"]) for row in calibration]
    expected_counts = {suite: protocol["calibration_cohorts"][cohort]["per_suite"]
                       for suite in SUITES}
    if (not isinstance(saved, dict)
            or (saved.get("protocol"), saved.get("cohort"), saved.get("normalization"),
                saved.get("completed_case_ids"), saved.get("suite_case_counts"),
                saved.get("target_linears")) != (
                protocol["protocol"], cohort, protocol["moments"]["normalization"],
                expected_ids, expected_counts, 614)):
        raise RuntimeError("Existing final moment metadata differs; refusing to recalibrate")
    for key in ("source_full_job", "source_micro_full_job"):
        expected = protocol[key]
        if key in saved and saved[key] != expected:
            raise RuntimeError(f"Existing final moment source identity differs: {key}")
    moments = saved.get("moments")
    if not isinstance(moments, dict) or len(modules) != 614 or set(moments) != set(modules):
        raise RuntimeError("Existing final moments do not cover the exact 614 Linear module names")
    for name, value in moments.items():
        if (not torch.is_tensor(value) or value.ndim != 1
                or value.numel() != modules[name].weight.shape[1]
                or not bool(torch.isfinite(value).all()) or bool((value < 0).any())):
            raise RuntimeError(f"Invalid final moment vector: {name}")
    return moments


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_protocol(p: dict) -> None:
    if p.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected calibration protocol")
    if p.get("walltime") != "01:59:00" or p.get("soft_deadline_seconds") != 6300:
        raise ValueError("Frozen PBS walltime/deadline differs")
    if p.get("target_linears") != 614 or tuple(p.get("suites", ())) != SUITES:
        raise ValueError("Frozen target coverage or suite order differs")
    if (p.get("weight_scope") != "all 614 target Linears; video306/action307/root-proprio1; no highprecision restoration branch"
            or p.get("activation") != "dynamic per-row signed symmetric absmax A4 qmax7; decoded BF16"):
        raise ValueError("Only the frozen 614-Linear decoded-BF16/A4 execution is supported")
    if p.get("source_full_job") != "25743693.pbs101" or p.get("source_micro_full_job") != "25755896.pbs101":
        raise ValueError("Frozen source campaign identities differ")
    transform = p["transform"]
    if (transform.get("name"), transform.get("alpha"), transform.get("hadamard_block")) != (
            "smooth05_hadamard", 0.5, 128):
        raise ValueError("Only the frozen Smooth0.5/H128 transform is supported")
    vq = p["vq"]
    if (vq.get("dimension"), vq.get("codebooks"), vq.get("entries"), vq.get("books_dtype"),
            vq.get("row_scale_dtype"),
            vq.get("samples"), vq.get("iterations"), vq.get("seed"), vq.get("action_lambda")) != (
            4, 2, 256, "bfloat16", "bfloat16", 8192, 8, 20261006, 0):
        raise ValueError("Frozen d4/M2/K256 BF16 VQ settings differ")
    if vq.get("local_tuning_steps") != 0 or p["microscaling"].get("scale_block") != 32:
        raise ValueError("Only untuned B32 microscaling is supported")
    evaluation = p["evaluation"]
    if (evaluation.get("case_count"), evaluation.get("cases_per_suite"),
            tuple(evaluation.get("seed_indices", ())), evaluation.get("query_count_per_cohort")) != (
            40, 10, (0, 1), 80):
        raise ValueError("Frozen paired evaluation cohort differs")
    if (evaluation.get("episodes"), evaluation.get("predicted_actions_executed"),
            evaluation.get("method_selection")) != (0, 0, "none; only c8 and c80 predetermined"):
        raise ValueError("Evaluation must remain fixed-input with no selection or executed actions")
    if ((p["calibration_cohorts"].get("c8", {}).get("contexts"),
         p["calibration_cohorts"].get("c8", {}).get("per_suite"),
         p["calibration_cohorts"].get("c80", {}).get("contexts"),
         p["calibration_cohorts"].get("c80", {}).get("per_suite")) != (8, 2, 80, 20)):
        raise ValueError("Frozen C8/C80 cohort sizes differ")
    moments = p["moments"]
    if (moments.get("source_arm") != "bf16w_a4" or moments.get("calibration_seed_index") != 0
            or moments.get("normalization") != "each context normalize token-row sum by row count; equal context and equal suite weights"):
        raise ValueError("Frozen BF16W+A4 moment source/seed differs")


def verify_campaign(root: Path, job_id: str, expected_protocol: str, *, source_protocol: dict | None = None) -> dict:
    summary_path = root / "results" / "full" / "summary.json"
    protocol_path = root / "results" / "full" / "protocol.json"
    if not summary_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(f"Completed source campaign is missing: {root}")
    summary, frozen = load_json(summary_path), load_json(protocol_path)
    actual_job = summary.get("pbs_jobid", summary.get("job_id"))
    if summary.get("status") != "complete" or actual_job != job_id:
        raise RuntimeError(f"Source campaign status/job differs at {root}")
    if (summary.get("protocol") != expected_protocol
            or summary.get("target_module_count", summary.get("target_linears")) != 614
            or summary.get("calibrated_module_count", summary.get("fitted_linears")) != 614):
        raise RuntimeError(f"Source campaign protocol/614-module coverage differs at {root}")
    if frozen.get("protocol") != expected_protocol:
        raise RuntimeError(f"Source protocol identity differs at {root}")
    if frozen.get("remote_root") and Path(frozen["remote_root"]).resolve() != root.resolve():
        raise RuntimeError(f"Source campaign root differs at {root}")
    if source_protocol is not None and frozen.get("model") != source_protocol.get("model"):
        raise RuntimeError("Source model identity differs from the frozen calibration protocol")
    return frozen


def safe_input_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Input path must be relative to data root: {relative}")
    resolved = (root / candidate).resolve()
    resolved.relative_to(root.resolve())
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_observation(np, data_root: Path, row: dict) -> tuple[dict, dict]:
    observation_path = safe_input_path(data_root, row["observation_file"])
    metadata_path = safe_input_path(data_root, row["metadata_file"])
    with np.load(observation_path, allow_pickle=False) as bundle:
        observation = {key: bundle[key] for key in bundle.files}
    required = {"agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                "robot0_eef_quat", "robot0_gripper_qpos"}
    if not observation or not required.issubset(observation):
        raise ValueError(f"Input {row['case_id']} has missing observation fields")
    if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all()
           for value in observation.values()):
        raise ValueError(f"Input {row['case_id']} contains nonfinite/non-numeric observations")
    metadata = load_json(metadata_path)
    for key in ("case_id", "suite", "task_id", "task_name", "domain", "phase", "demo_id",
                "frame_index", "sampler_seeds", "observation_file", "metadata_file", "dimension",
                "variant", "variant_id", "original_task", "source_index", "environment_seed"):
        if metadata.get(key) != row.get(key):
            raise RuntimeError(f"Input {row['case_id']} {key} differs between plan and metadata")
    expected_identity = {key: row[key] for key in INPUT_IDENTITY}
    if metadata.get("input_identity") != expected_identity:
        raise RuntimeError(f"Input {row['case_id']} nested input identity differs")
    if (row["dimension"], row["variant"], row["variant_id"], row["original_task"],
            row["source_index"], int(row["environment_seed"])) != (
            "original", "original", None, row["task_name"], None,
            830_000_000 + int(row["case_id"]) * 17):
        raise RuntimeError(f"Input {row['case_id']} legacy/source identity differs")
    if metadata.get("domain") != "original" or metadata.get("settling_steps") != 0:
        raise RuntimeError(f"Input {row['case_id']} is outside the frozen original-demo boundary")
    if not isinstance(metadata.get("description"), str) or not metadata["description"].strip():
        raise ValueError(f"Input {row['case_id']} has no task description")
    return observation, metadata


def validate_plan(root: Path, p: dict, cohort: str) -> tuple[dict, dict[int, dict], list[dict], list[dict]]:
    plan = load_json(root / p["data"]["plan_file"])
    if plan.get("protocol") != PROTOCOL or plan.get("status") != "complete":
        raise RuntimeError("Frozen calibration input plan is missing or incomplete")
    rows = plan.get("inputs")
    if not isinstance(rows, list) or len(rows) != 120:
        raise RuntimeError("Expected the frozen 80-calibration/40-test input plan")
    by_id = {int(row["case_id"]): row for row in rows}
    if len(by_id) != 120 or set(by_id) != set(range(2000, 2080)) | set(range(3000, 3040)):
        raise RuntimeError("Input plan IDs differ from the frozen calibration/test ranges")
    if any(row.get("domain") != "original" or row.get("suite") not in SUITES
           or not isinstance(row.get("sampler_seeds"), list) or len(row["sampler_seeds"]) != 2
           or any(type(seed) is not int for seed in row["sampler_seeds"])
           or row["sampler_seeds"][0] == row["sampler_seeds"][1] for row in rows):
        raise RuntimeError("Input plan includes an invalid domain, suite, or sampler pair")
    for row in rows:
        if (row.get("dimension"), row.get("variant"), row.get("variant_id"),
                row.get("original_task"), row.get("source_index"), row.get("environment_seed")) != (
                "original", "original", None, row.get("task_name"), None,
                830_000_000 + int(row["case_id"]) * 17):
            raise RuntimeError(f"Input plan source identity differs for case {row['case_id']}")
    cohorts = plan.get("calibration_cohorts", {})
    c8, c80 = cohorts.get("c8"), cohorts.get("c80")
    if not isinstance(c8, list) or not isinstance(c80, list):
        raise RuntimeError("Input plan calibration_cohorts must contain c8/c80 ID lists")
    if (len(c8), len(set(c8)), len(c80), len(set(c80))) != (8, 8, 80, 80) or not set(c8) <= set(c80):
        raise RuntimeError("C8/C80 membership is malformed or not nested")
    calibration_ids = list(cohorts[cohort]) if cohort in ("c8", "c80") else []
    expected_count = p["calibration_cohorts"][cohort]["contexts"] if cohort in ("c8", "c80") else 0
    if len(calibration_ids) != expected_count:
        raise RuntimeError(f"{cohort} calibration context count differs")
    calibration = [by_id[case] for case in calibration_ids]
    test = [by_id[case] for case in range(3000, 3040)]
    for suite in SUITES:
        suite_cal = [row for row in calibration if row["suite"] == suite]
        suite_test = [row for row in test if row["suite"] == suite]
        expected_calibration = 0 if cohort == "teacher" else p["calibration_cohorts"][cohort]["per_suite"]
        if len(suite_test) != 10 or len(suite_cal) != expected_calibration:
            raise RuntimeError(f"Frozen per-suite counts differ for {suite}/{cohort}")
        if {int(row["task_id"]) for row in suite_test} != set(range(10)):
            raise RuntimeError(f"Test task coverage differs for {suite}")
        if cohort == "c8" and len({int(row["task_id"]) for row in suite_cal}) != 2:
            raise RuntimeError(f"C8 requires two distinct tasks in {suite}")
        if cohort == "c80" and (len({int(row["task_id"]) for row in suite_cal}) != 10
                                 or any(sum(int(item["task_id"]) == task for item in suite_cal) != 2
                                        or len({int(item["demo_id"]) for item in suite_cal
                                                if int(item["task_id"]) == task}) != 2
                                        for task in range(10))
                                 or [sum(row["phase"] == phase for row in suite_cal)
                                     for phase in ("early", "middle", "late")] != [7, 7, 6]):
            raise RuntimeError(f"C80 requires two contexts for each task in {suite}")
    for row in test:
        if row["demo_id"] in {item["demo_id"] for item in rows
                              if 2000 <= int(item["case_id"]) < 2080
                              and item["suite"] == row["suite"]
                              and int(item["task_id"]) == int(row["task_id"])}:
            raise RuntimeError("Held-out test demonstration overlaps calibration")
    return plan, by_id, calibration, test


def model_identity(p: dict, runtime) -> dict:
    return {"model": p["model"], "checkpoint": str(runtime.cfg.ckpt),
            "dataset_stats": str(runtime.cfg.EVALUATION.dataset_stats_path),
            "mode": "idm", "inference_steps": 10}


def validate_action(action, label: str) -> None:
    if (not isinstance(action, list) or len(action) != 32
            or any(not isinstance(step, list) or len(step) != 7
                   or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                          for value in step) for step in action)):
        raise RuntimeError(f"Malformed finite [32,7] action: {label}")


def validate_teacher(root: Path, p: dict, test: list[dict], identity: dict) -> dict[tuple[int, int], dict]:
    directory = root / "results" / "teacher"
    summary = load_json(directory / "summary.json")
    expected_cases = [int(row["case_id"]) for row in test]
    if (summary.get("status") != "complete" or summary.get("protocol") != p["protocol"]
            or summary.get("source_full_job") != p["source_full_job"]
            or summary.get("source_micro_full_job") != p["source_micro_full_job"]
            or summary.get("model_identity") != identity
            or summary.get("case_ids") != expected_cases or summary.get("query_count") != 80
            or summary.get("target_linears") != 614
            or summary.get("suites") != {suite: 10 for suite in SUITES}):
        raise RuntimeError("Common BF16 teacher summary identity/coverage differs")
    refs = {}
    for row in test:
        case = int(row["case_id"])
        for seed_index, sampler_seed in enumerate(row["sampler_seeds"]):
            query = load_json(directory / "query_records" / f"test_{case}_{seed_index}.json")
            if (query.get("case_id"), query.get("seed_index"), query.get("sampler_seed"),
                    query.get("suite"), query.get("task_id"), query.get("protocol"),
                    query.get("split"), query.get("model_identity"), query.get("arm"),
                    query.get("observed_module_count")) != (
                    case, seed_index, sampler_seed, row["suite"], int(row["task_id"]),
                    p["protocol"], "test", identity, "bf16", 614):
                raise RuntimeError(f"Teacher query identity differs: {case}/{seed_index}")
            action = query.get("action")
            validate_action(action, f"teacher {case}/{seed_index}")
            refs[case, seed_index] = query
    return refs


def aggregate_test(rows: list[dict]) -> dict:
    by_case: dict[int, list[dict]] = {}
    for row in rows:
        if any(not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric])
               for metric in METRICS):
            raise RuntimeError("Test query contains a malformed RMSE metric")
        by_case.setdefault(int(row["case_id"]), []).append(row)
    if (len(by_case) != 40
            or any(len(values) != 2 or {int(value["seed_index"]) for value in values} != {0, 1}
                   or len({value["suite"] for value in values}) != 1 for values in by_case.values())):
        raise RuntimeError("Paired test denominator must be 40 cases × 2 sampler seeds")
    per_case = {case: {metric: statistics.mean(record[metric] for record in values) for metric in METRICS}
                for case, values in by_case.items()}
    per_suite = {}
    for suite in SUITES:
        cases = [case for case, values in by_case.items() if values[0]["suite"] == suite]
        if len(cases) != 10:
            raise RuntimeError(f"Expected ten test contexts for {suite}")
        per_suite[suite] = {metric: statistics.mean(per_case[case][metric] for case in cases)
                            for metric in METRICS}
    return {"case_count": 40, "query_count": len(rows), "per_case": per_case,
            "per_suite": per_suite,
            "suite_macro": {metric: statistics.mean(per_suite[suite][metric] for suite in SUITES)
                            for metric in METRICS}}


def run(args) -> None:
    guard()
    import faulthandler
    import numpy as np
    import torch
    import runner as pilot
    import run_diagnostic as source
    import run_experiment as original
    import micro_codebook as micro

    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    p = load_json(args.protocol)
    validate_protocol(p)
    root = Path(p["remote_root"])
    plan, by_id, calibration, test = validate_plan(root, p, args.cohort)
    data_root = (root / Path(p["data"]["plan_file"])).parent
    source_root = Path(p["source_root"])
    original_source_protocol = verify_campaign(source_root, p["source_full_job"],
                                               "fastwam-vq-action-aware-v1", source_protocol=p)
    if (original_source_protocol.get("transform") != p["transform"]
            or Path(original_source_protocol["roots"]["old_banks"]).resolve()
            != Path(p["roots"]["old_banks"]).resolve()):
        raise RuntimeError("Original source transform/bank root differs from the frozen source")
    micro_source_protocol = verify_campaign(Path(p["source_micro_root"]), p["source_micro_full_job"],
                                            "fastwam-vq-microscale-v1", source_protocol=p)
    if (micro_source_protocol.get("transform") != p["transform"]
            or micro_source_protocol.get("vq") != p["vq"]
            or micro_source_protocol.get("microscaling") != p["microscaling"]):
        raise RuntimeError("Completed microscale source settings differ from this frozen campaign")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol_copy = out / "protocol.json"
    if protocol_copy.is_file() and load_json(protocol_copy) != p:
        raise RuntimeError("Cannot resume this cohort with a changed protocol")
    save_json(protocol_copy, p)
    artifact = Path(os.environ["ARTIFACTS"])
    artifact.mkdir(parents=True, exist_ok=True)
    job_id = os.environ["PBS_JOBID"]
    try:
        elapsed_at_start = max(0.0, time.time() - float(os.environ["FW_JOB_START_EPOCH"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("FW_JOB_START_EPOCH is required for the 6300-second soft deadline") from exc
    started = time.monotonic()
    deadline = started + max(0.0, p["soft_deadline_seconds"] - elapsed_at_start)
    device = torch.cuda.get_device_name(0)
    if "A100" not in device:
        raise RuntimeError(f"Frozen campaign requires an A100; found {device}")
    identity = {"model": p["model"], "checkpoint": str(Path(p["roots"]["base"]) /
                 "checkpoints/libero_optional_idm_2cam224.clean.pt"),
                "dataset_stats": str(Path(p["roots"]["base"]) /
                                     "checkpoints/libero_optional_idm_2cam224_dataset_stats.json"),
                "mode": "idm", "inference_steps": 10}
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = banks = None
    fit_records, query_records = [], []
    moment_completed: list[int] = []
    recent_units = {"moment": [], "fit": [], "query": []}
    stage = "source_validation"

    def status_payload(status: str, **extra) -> dict:
        return {"status": status, "protocol": p["protocol"], "cohort": args.cohort,
                "pbs_jobid": job_id, "source_full_job": p["source_full_job"],
                "source_micro_full_job": p["source_micro_full_job"],
                "target_linears": 614, "device": device, "model_identity": identity,
                "calibration_contexts": len(calibration), "calibration_completed": len(moment_completed),
                "fitted_linears": len(fit_records), "queries_completed": len(query_records),
                "expected_queries": 80,
                "walltime": p["walltime"], "soft_deadline_seconds": p["soft_deadline_seconds"],
                "elapsed_job_seconds": time.monotonic() - started + elapsed_at_start,
                "real_quant": False, "native_low_bit_kernel": False,
                "execution": ("shared BF16 teacher; original BF16 weights and activations"
                              if args.cohort == "teacher" else
                              "saved microscale additive VQ decoded to BF16 Linear; dynamic A4 BF16 inputs"),
                "episodes": 0, "predicted_actions_executed": 0, **extra}

    def progress(**extra) -> None:
        extra.setdefault("stage", stage)
        row = status_payload("running", **extra)
        save_json(out / "progress.json", row)
        save_json(artifact / "progress.json", row)
        print("CALIBRATION_PROGRESS " + json.dumps(row), flush=True)

    def checkpoint(**extra) -> None:
        row = status_payload("checkpointed", stage=stage, requires_resume=True, **extra)
        save_json(out / "checkpoint.json", row)
        save_json(out / "summary.json", row)
        save_json(artifact / "summary.json", row)
        print("CALIBRATION_CHECKPOINT " + json.dumps(row), flush=True)

    def complete(**extra) -> None:
        row = status_payload("complete", stage="complete", requires_resume=False, **extra)
        save_json(out / "summary.json", row)
        save_json(artifact / "summary.json", row)
        save_json(out / "progress.json", row)
        print("CALIBRATION_COMPLETE " + json.dumps(row), flush=True)

    def enough_time(reserve: float = 300.0) -> bool:
        return deadline - time.monotonic() > reserve

    def load_datum(row: dict):
        observation, metadata = load_observation(np, data_root, row)
        return runtime.datum(observation), metadata["description"]

    def infer(row: dict, seed_index: int, arm: str) -> torch.Tensor:
        datum, description = load_datum(row)
        runtime.description, runtime.current_arm = description, arm
        banks.seen.clear()
        action, _ = runtime.infer(datum, int(row["sampler_seeds"][seed_index]), measure=False)
        if len(banks.seen) != 614:
            raise RuntimeError(f"Query called {len(banks.seen)}/614 target Linears")
        action = action.detach().float().cpu()
        if tuple(action.shape) != (32, 7) or not bool(torch.isfinite(action).all()):
            raise RuntimeError("Expected a finite normalized [32,7] action")
        return action

    try:
        stage = "model_loading"
        progress()
        runtime = pilot.Runtime(artifact)
        runtime.arm("bf16")
        identity = model_identity(p, runtime)
        if (identity["checkpoint"] != str(Path(p["roots"]["base"]) /
                                            "checkpoints/libero_optional_idm_2cam224.clean.pt")
                or identity["dataset_stats"] != str(Path(p["roots"]["base"]) /
                                                     "checkpoints/libero_optional_idm_2cam224_dataset_stats.json")):
            raise RuntimeError("Loaded model checkpoint/statistics differ from the frozen source")
        banks = original.FrozenBanks(runtime, out, p)
        runtime_record = load_json(artifact / "runtime.json")
        runtime_record.update(experiment_protocol=p["protocol"], real_quant=False,
                              native_low_bit_kernel=False,
                              execution="decoded BF16 references; no packed VQ compute kernel")
        save_json(artifact / "runtime.json", runtime_record)

        if args.cohort == "teacher":
            stage = "shared_bf16_teacher_queries"
            banks.activate("bf16")
            query_dir = out / "query_records"
            query_dir.mkdir(parents=True, exist_ok=True)
            done = set()
            for row in test:
                case = int(row["case_id"])
                for seed_index, sampler_seed in enumerate(row["sampler_seeds"]):
                    path = query_dir / f"test_{case}_{seed_index}.json"
                    if path.is_file():
                        saved = load_json(path)
                        if (saved.get("protocol"), saved.get("case_id"), saved.get("seed_index"),
                                saved.get("sampler_seed"), saved.get("arm"),
                                saved.get("model_identity"), saved.get("observed_module_count")) != (
                                p["protocol"], case, seed_index, sampler_seed, "bf16", identity, 614):
                            raise RuntimeError(f"Resumed teacher query identity differs: {case}/{seed_index}")
                        if saved.get("split") != "test" or saved.get("suite") != row["suite"] \
                                or saved.get("task_id") != int(row["task_id"]):
                            raise RuntimeError(f"Resumed teacher query case identity differs: {case}/{seed_index}")
                        validate_action(saved.get("action"), f"resumed teacher {case}/{seed_index}")
                        if saved.get("query_wall_seconds", 0) > 0:
                            recent_units["query"].append(float(saved["query_wall_seconds"]))
                        done.add((case, seed_index))
                        query_records.append(saved)
                        continue
                    reserve = max(300.0, 2 * max(recent_units["query"][-3:], default=0.0))
                    if not enough_time(reserve):
                        checkpoint(case_ids=[int(item["case_id"]) for item in test],
                                   test_case_count=len({case for case, _ in done}))
                        return
                    progress(case_id=case, seed_index=seed_index, arm="bf16", split="test")
                    query_started = time.monotonic()
                    action = infer(row, seed_index, "bf16")
                    record = {"protocol": p["protocol"], "case_id": case,
                              "suite": row["suite"], "task_id": int(row["task_id"]),
                              "seed_index": seed_index, "sampler_seed": int(sampler_seed),
                              "split": "test", "arm": "bf16", "action": action.tolist(),
                              "query_wall_seconds": time.monotonic() - query_started,
                              "observed_module_count": len(banks.seen), "model_identity": identity,
                              "episodes": 0, "predicted_actions_executed": 0}
                    save_json(path, record)
                    query_records.append(record)
                    done.add((case, seed_index))
                    recent_units["query"].append(record["query_wall_seconds"])
                    progress(case_id=case, seed_index=seed_index, arm="bf16", split="test")
            if len(done) != 80:
                raise RuntimeError(f"Teacher query coverage is {len(done)}/80")
            complete(case_ids=[int(row["case_id"]) for row in test], case_count=40,
                     query_count=80, target_linears=614,
                     suites={suite: 10 for suite in SUITES},
                     teacher_path=str(out / "query_records"))
            return

        micro.self_check()

        # Per-context moments are committed independently so an interrupted PBS
        # job can continue without repeating completed calibration forwards.
        stage = "per_context_a4_moments"
        banks.activate("bf16w_a4")
        moment_dir = out / "moments" / "contexts"
        moment_dir.mkdir(parents=True, exist_ok=True)
        final_path = out / "moments" / "final.pt"
        moments = restore_final_moments(torch, final_path, p, args.cohort,
                                         calibration, banks.modules)
        if moments is not None:
            moment_completed.extend(int(row["case_id"]) for row in calibration)
            print(f"MOMENTS_REUSED final contexts={len(moment_completed)} modules={len(moments)}", flush=True)
            progress(moments_reused="final", contexts_completed=len(moment_completed),
                     moment_file=str(final_path))
        else:
            state_path = out / "moments" / "progress.json"
            if state_path.is_file():
                state = load_json(state_path)
                if (state.get("protocol"), state.get("cohort"), state.get("source_full_job"),
                        state.get("source_micro_full_job")) != (
                        p["protocol"], args.cohort, p["source_full_job"], p["source_micro_full_job"]):
                    raise RuntimeError("Moment checkpoint identity differs")
                declared = set(int(value) for value in state.get("completed_case_ids", []))
            else:
                declared = set()
            context_data = {}
            for case in (int(row["case_id"]) for row in calibration):
                path = moment_dir / f"context_{case}.pt"
                if not path.is_file():
                    if case in declared:
                        raise RuntimeError(f"Moment checkpoint lists a missing context: {case}")
                    continue
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (saved.get("protocol"), saved.get("cohort"), saved.get("case_id"),
                        saved.get("source_full_job"), saved.get("source_micro_full_job")) != (
                        p["protocol"], args.cohort, case, p["source_full_job"], p["source_micro_full_job"]):
                    raise RuntimeError(f"Moment context identity differs: {case}")
                if set(saved.get("sums", {})) != set(banks.modules) or set(saved.get("rows", {})) != set(banks.modules):
                    raise RuntimeError(f"Moment context lacks 614 target Linears: {case}")
                for name, value in saved["sums"].items():
                    if value.ndim != 1 or value.numel() != banks.modules[name].weight.shape[1] or not bool(torch.isfinite(value).all()):
                        raise RuntimeError(f"Invalid moment vector: {case}/{name}")
                    if int(saved["rows"][name]) <= 0:
                        raise RuntimeError(f"Invalid moment row count: {case}/{name}")
                context_data[case] = saved
                moment_completed.append(case)
                if saved.get("context_seconds", 0) > 0:
                    recent_units["moment"].append(float(saved["context_seconds"]))
            if not declared <= set(context_data):
                raise RuntimeError("Moment progress names unvalidated context files")

            for row in calibration:
                case = int(row["case_id"])
                if case in context_data:
                    continue
                reserve = max(300.0, 2 * max(recent_units["moment"][-3:], default=0.0))
                if not enough_time(reserve):
                    checkpoint(moment_case_ids=sorted(moment_completed))
                    return
                banks.moments.clear()
                banks.moment_rows.clear()
                banks.collect_moment = True
                progress(case_id=case, suite=row["suite"], sampler_seed=int(row["sampler_seeds"][p["moments"]["calibration_seed_index"]]),
                         stage=stage)
                unit_started = time.monotonic()
                infer(row, p["moments"]["calibration_seed_index"], "bf16w_a4")
                banks.collect_moment = False
                if set(banks.moments) != set(banks.modules) or set(banks.moment_rows) != set(banks.modules):
                    raise RuntimeError(f"A4 moments do not cover all 614 Linears for context {case}")
                saved = {"protocol": p["protocol"], "cohort": args.cohort, "case_id": case,
                         "suite": row["suite"], "task_id": int(row["task_id"]),
                         "demo_id": int(row["demo_id"]), "phase": row["phase"],
                         "sampler_seed": int(row["sampler_seeds"][p["moments"]["calibration_seed_index"]]),
                         "source_full_job": p["source_full_job"],
                         "source_micro_full_job": p["source_micro_full_job"],
                         "context_seconds": time.monotonic() - unit_started,
                         "sums": {name: value.detach().cpu() for name, value in banks.moments.items()},
                         "rows": {name: int(value) for name, value in banks.moment_rows.items()}}
                path = moment_dir / f"context_{case}.pt"
                save_tensor(torch, path, saved)
                context_data[case] = saved
                moment_completed.append(case)
                recent_units["moment"].append(saved["context_seconds"])
                save_json(state_path, {"protocol": p["protocol"], "cohort": args.cohort,
                                       "source_full_job": p["source_full_job"],
                                       "source_micro_full_job": p["source_micro_full_job"],
                                       "completed_case_ids": sorted(moment_completed),
                                       "contexts_completed": len(moment_completed), "target_linears": 614})
                progress(case_id=case, contexts_completed=len(moment_completed),
                         last_context_seconds=recent_units["moment"][-1])

            if len(context_data) != len(calibration):
                raise RuntimeError(f"Calibration moment coverage is {len(context_data)}/{len(calibration)}")
            moments_by_suite = {}
            for suite in SUITES:
                cases = [int(row["case_id"]) for row in calibration if row["suite"] == suite]
                if len(cases) != p["calibration_cohorts"][args.cohort]["per_suite"]:
                    raise RuntimeError(f"Moment suite denominator differs for {suite}")
                moments_by_suite[suite] = {
                    name: torch.stack([context_data[case]["sums"][name].float()
                                       / int(context_data[case]["rows"][name]) for case in cases]).mean(0)
                    for name in banks.modules
                }
            moments = {name: torch.stack([moments_by_suite[suite][name] for suite in SUITES]).mean(0)
                       for name in banks.modules}
            save_tensor(torch, final_path, {
                "protocol": p["protocol"], "cohort": args.cohort,
                "normalization": p["moments"]["normalization"],
                "completed_case_ids": [int(row["case_id"]) for row in calibration],
                "suite_case_counts": {suite: p["calibration_cohorts"][args.cohort]["per_suite"] for suite in SUITES},
                "target_linears": 614, "moments": moments,
            })
        banks.moments.clear()
        banks.moment_rows.clear()

        stage = "fit_all_614_vq_banks"
        bank_dir = out / "banks" / "micro_scale_vq_a4"
        bank_dir.mkdir(parents=True, exist_ok=True)
        settings = p["vq"]
        with torch.no_grad():
            for index, (name, module) in enumerate(banks.modules.items()):
                path = bank_dir / f"{index:04d}.pt"
                if path.is_file():
                    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                    if (saved.get("protocol"), saved.get("cohort"), saved.get("module"),
                            saved.get("module_index"), tuple(saved.get("encoded", {}).get("shape", ()))) != (
                            p["protocol"], args.cohort, name, index, tuple(module.weight.shape)):
                        raise RuntimeError(f"Resumed bank identity/shape differs: {index}/{name}")
                    micro.check_encoding(saved["encoded"])
                    expected_transform = original.cpu_encoding(torch, banks.transforms[name])
                    for key, value in expected_transform.items():
                        actual = saved["transform"].get(key)
                        if torch.is_tensor(value):
                            if not torch.is_tensor(actual) or not torch.equal(value, actual):
                                raise RuntimeError(f"Resumed bank transform differs: {name}/{key}")
                        elif actual != value:
                            raise RuntimeError(f"Resumed bank transform differs: {name}/{key}")
                    fit_record = saved["fit_record"]
                    fit_record.setdefault("scale_exponent_count", int(saved["encoded"]["scale_exponents"].numel()))
                    fit_record.setdefault("transform_tensor_bytes", sum(
                        value.numel() * value.element_size() for value in saved["transform"].values()
                        if torch.is_tensor(value)))
                    fit_records.append(fit_record)
                    if fit_record.get("fit_wall_seconds", 0) > 0:
                        recent_units["fit"].append(float(fit_record["fit_wall_seconds"]))
                    continue
                reserve = max(300.0, 2 * max(recent_units["fit"][-3:], default=0.0))
                if not enough_time(reserve):
                    checkpoint(moment_case_ids=sorted(moment_completed))
                    return
                progress(module=name, module_index=index, fitted_linears=len(fit_records))
                unit_started = time.monotonic()
                weight = banks.dense_teacher_weight(name)
                moment = moments[name].to("cuda")
                encoded = micro.fit_vq(weight, second_moment=moment, seed=settings["seed"],
                                       sample_limit=settings["samples"], iterations=settings["iterations"])
                micro.check_encoding(encoded)
                dense = micro.decode_vq(encoded)
                delta = dense.float() - weight.float()
                record = {"module": name, "module_index": index, "weights": weight.numel(),
                          "encoded_bytes": micro.encoded_nbytes(encoded),
                          "scale_exponent_count": int(encoded["scale_exponents"].numel()),
                          "weight_mse": float(delta.square().mean()),
                          "moment_weighted_weight_mse": float((delta.square() * moment[None]).mean()),
                          "nonzero_exponents": int((encoded["scale_exponents"] != 127).sum()),
                          "exponent_min": int(encoded["scale_exponents"].to(torch.int32).min()) - 127,
                          "exponent_max": int(encoded["scale_exponents"].to(torch.int32).max()) - 127,
                          "clipped_exponents": int(encoded["fit_stats"]["clipped_exponents"]),
                          "fit_wall_seconds": time.monotonic() - unit_started}
                transform = original.cpu_encoding(torch, banks.transforms[name])
                transform_tensor_bytes = sum(value.numel() * value.element_size()
                                             for value in transform.values() if torch.is_tensor(value))
                record["transform_tensor_bytes"] = transform_tensor_bytes
                serialized = {"protocol": p["protocol"], "cohort": args.cohort,
                              "module": name, "module_index": index, "transform": transform,
                              "encoded": original.cpu_encoding(torch, encoded), "fit_record": record}
                save_tensor(torch, path, serialized)
                fit_records.append(record)
                recent_units["fit"].append(record["fit_wall_seconds"])
                del weight, moment, encoded, dense, delta
                progress(module=name, module_index=index + 1, last_fit_seconds=record["fit_wall_seconds"])

        if len(fit_records) != 614:
            raise RuntimeError(f"Fitted bank coverage is {len(fit_records)}/614")
        freeze = {"protocol": p["protocol"], "cohort": args.cohort,
                  "fitted_linears": 614, "method_selection": False,
                  "settings": settings, "microscaling": p["microscaling"]}
        freeze_path = out / "bank_freeze.json"
        if freeze_path.is_file() and load_json(freeze_path) != freeze:
            raise RuntimeError("Frozen bank receipt differs on resume")
        save_json(freeze_path, freeze)

        stage = "paired_shared_test_queries"
        teacher_summary_path = root / "results" / "teacher" / "summary.json"
        if not teacher_summary_path.is_file() or load_json(teacher_summary_path).get("status") != "complete":
            checkpoint(moment_case_ids=sorted(moment_completed), awaiting_teacher=True)
            return
        refs = validate_teacher(root, p, test, identity)
        for index, (name, module) in enumerate(banks.modules.items()):
            bank = torch.load(bank_dir / f"{index:04d}.pt", map_location="cuda", weights_only=True)
            if bank["module"] != name or bank["protocol"] != p["protocol"] or bank["cohort"] != args.cohort:
                raise RuntimeError(f"Evaluation bank identity differs: {name}")
            module.weight.data = micro.decode_vq(bank["encoded"])
            del bank
        banks.transformed = True

        query_dir = out / "query_records"
        query_dir.mkdir(parents=True, exist_ok=True)
        done = set()
        for row in test:
            case = int(row["case_id"])
            for seed_index, sampler_seed in enumerate(row["sampler_seeds"]):
                path = query_dir / f"test_{case}_{seed_index}.json"
                if path.is_file():
                    saved = load_json(path)
                    if (saved.get("protocol"), saved.get("cohort"), saved.get("case_id"),
                            saved.get("suite"), saved.get("task_id"), saved.get("split"),
                            saved.get("seed_index"), saved.get("sampler_seed"), saved.get("arm"),
                            saved.get("observed_module_count"), saved.get("teacher_source_full_job"),
                            saved.get("teacher_model_identity")) != (
                            p["protocol"], args.cohort, case, row["suite"], int(row["task_id"]),
                            "test", seed_index, sampler_seed, "micro_scale_vq_a4", 614,
                            p["source_full_job"], identity):
                        raise RuntimeError(f"Resumed evaluation query identity differs: {case}/{seed_index}")
                    validate_action(saved.get("action"), f"resumed arm {case}/{seed_index}")
                    if any(not isinstance(saved.get(metric), (int, float))
                           or not math.isfinite(saved[metric]) for metric in METRICS):
                        raise RuntimeError(f"Resumed evaluation metrics are malformed: {case}/{seed_index}")
                    if saved.get("query_wall_seconds", 0) > 0:
                        recent_units["query"].append(float(saved["query_wall_seconds"]))
                    query_records.append(saved)
                    done.add((case, seed_index))
                    continue
                reserve = max(300.0, 2 * max(recent_units["query"][-3:], default=0.0))
                if not enough_time(reserve):
                    checkpoint(moment_case_ids=sorted(moment_completed),
                               test_case_count=len({case_id for case_id, _ in done}))
                    return
                progress(case_id=case, seed_index=seed_index, arm="micro_scale_vq_a4", split="test")
                query_started = time.monotonic()
                action = infer(row, seed_index, "micro_scale_vq_a4")
                reference = torch.tensor(refs[case, seed_index]["action"], dtype=torch.float32)
                record = {"protocol": p["protocol"], "cohort": args.cohort,
                          "case_id": case, "suite": row["suite"], "task_id": int(row["task_id"]),
                          "seed_index": seed_index, "sampler_seed": int(sampler_seed),
                          "split": "test", "arm": "micro_scale_vq_a4", "action": action.tolist(),
                          "query_wall_seconds": time.monotonic() - query_started,
                          "observed_module_count": len(banks.seen),
                          "teacher_source_full_job": p["source_full_job"],
                          "teacher_model_identity": identity,
                          **source.action_metrics(torch, action, reference),
                          "episodes": 0, "predicted_actions_executed": 0}
                save_json(path, record)
                query_records.append(record)
                done.add((case, seed_index))
                recent_units["query"].append(record["query_wall_seconds"])
                progress(case_id=case, seed_index=seed_index, arm="micro_scale_vq_a4", split="test")
        result = aggregate_test(query_records)
        if len(done) != 80 or result["query_count"] != 80:
            raise RuntimeError(f"Test query coverage is {len(done)}/80")
        save_json(out / "module_fit.json", {"protocol": p["protocol"], "cohort": args.cohort,
                  "fitted_linears": 614, "modules": fit_records})
        total_weights = sum(int(record["weights"]) for record in fit_records)
        encoded_bytes = sum(int(record["encoded_bytes"]) for record in fit_records)
        transform_tensor_bytes = sum(int(record["transform_tensor_bytes"]) for record in fit_records)
        serialized_bank_bytes = sum((bank_dir / f"{index:04d}.pt").stat().st_size
                                    for index in range(len(fit_records)))
        exponent_count = sum(int(record["scale_exponent_count"]) for record in fit_records)
        payload_with_transforms = encoded_bytes + transform_tensor_bytes
        storage_summary = {"encoded_bytes": encoded_bytes,
                           "transform_tensor_bytes": transform_tensor_bytes,
                           "payload_bytes_with_transforms": payload_with_transforms,
                           "serialized_bank_file_bytes": serialized_bank_bytes,
                           "dense_bf16_bytes": total_weights * 2,
                           "effective_bits_per_weight_with_transforms": payload_with_transforms * 8 / total_weights,
                           "payload_to_dense_bf16_ratio": payload_with_transforms / (total_weights * 2)}
        exponent_summary = {"count": exponent_count,
                            "nondefault_count": sum(int(record["nonzero_exponents"]) for record in fit_records),
                            "clipped_count": sum(int(record["clipped_exponents"]) for record in fit_records),
                            "minimum": min(int(record["exponent_min"]) for record in fit_records),
                            "maximum": max(int(record["exponent_max"]) for record in fit_records)}
        complete(case_ids=[int(row["case_id"]) for row in test], case_count=40,
                 query_count=80, fitted_linears=614, calibrated_contexts=len(calibration),
                 calibration_case_ids=[int(row["case_id"]) for row in calibration],
                 suites=list(SUITES), suite_macro_primary_metric="motor_rmse_first10_vs_bf16",
                 test=result, bank_directory=str(bank_dir), teacher_query_directory=str(root / "results" / "teacher" / "query_records"),
                 module_fit_path=str(out / "module_fit.json"), storage=storage_summary,
                 scale_exponents=exponent_summary,
                 method_selection=False, test_used_for_method_selection=False,
                 transform=p["transform"], vq=settings, microscaling=p["microscaling"])
    except BaseException as exc:
        save_json(artifact / "failure.json", {"protocol": p["protocol"], "cohort": args.cohort,
                  "pbs_jobid": job_id, "stage": stage,
                  "calibration_completed": len(moment_completed), "fitted_linears": len(fit_records),
                  "queries_completed": len(query_records),
                  "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if banks is not None:
            banks.collect_moment = False
            banks.close()
        if runtime is not None:
            runtime.q.close()
        torch.backends.cuda.matmul.allow_tf32 = tf32
        faulthandler.cancel_dump_traceback_later()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Persistent result directory for this cohort")
    parser.add_argument("--cohort", choices=("teacher", "c8", "c80"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
