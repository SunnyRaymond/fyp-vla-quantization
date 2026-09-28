#!/usr/bin/env python3
"""Collect evidence for the one frozen PLDM formal run; never load model/data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile


SOURCE_SHA = "1bd7e564ecd961205bc18b23067b19e9ca24ac90"
DEFAULT_ROOT = Path("/scratch/users/ntu/yguo017/pldm-reproduction")
RUN_NAME = "tworooms-seqlen90-3M-seed101"
UPDATES_PER_PASS = 48_000
TOTAL_UPDATES = 144_000
FINAL_STEP = TOTAL_UPDATES - 1
FINAL_SAMPLE_STEP = TOTAL_UPDATES * 64
METRICS = (
    "wall_medium_planning_error_mean",
    "wall_medium_planning_error_mean_rmse",
    "wall_medium_success_rate",
    "wall_medium_cross_wall_rate",
    "wall_medium_init_plan_cross_wall_rate",
    "wall_medium_avg_termination_step",
)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def bounded_log_text(path: Path, limit: int = 65_536) -> str:
    """Read only the log header and tail; training progress can make it large."""
    try:
        with path.open("rb") as stream:
            head = stream.read(limit)
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            tail = stream.read(limit)
        return (head + b"\n" + tail).decode("utf-8", errors="replace")
    except OSError:
        return ""


def key_values(path: Path) -> dict[str, str]:
    try:
        return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)
    except OSError:
        return {}


def metadata(path: Path, started: datetime | None = None):
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    info = {"path": str(path), "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    if started is not None:
        info["written_during_job"] = stat.st_mtime >= started.timestamp() - 2
    return info


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def json_metric(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def collect(root: Path, job_id: str, pbs_exit_status: int) -> tuple[dict, int]:
    report = root / "reports" / job_id
    run = root / "checkpoints" / RUN_NAME
    report.mkdir(parents=True, exist_ok=True)
    incomplete: list[str] = []
    checks: dict[str, str] = {}

    def check(name: str, passed: bool, why: str):
        checks[name] = "PASS" if passed else "FAIL"
        if not passed:
            incomplete.append(f"{name}: {why}")

    freeze = read_json(report / "FREEZE.json")
    prep = read_json(report / "PREPARATION.json")
    source = read_json(report / "source_identity.json")
    staged = read_json(report / "staged_source_identity.json")
    source_ok = (
        isinstance(freeze, dict) and freeze.get("source", {}).get("commit") == SOURCE_SHA
        and isinstance(prep, dict) and prep.get("status") == "PASS" and prep.get("source_sha") == SOURCE_SHA
        and isinstance(source, dict) and source.get("source_sha") == SOURCE_SHA
        and isinstance(staged, dict) and staged.get("source_sha") == SOURCE_SHA
        and staged.get("algorithm_or_config_changes") == []
    )
    check("pinned_source_and_preparation", source_ok,
          "copied freeze/preparation/source identities do not establish the pinned run")

    training = prep.get("actual_training_loader", {}) if isinstance(prep, dict) else {}
    passes_ok = (
        training.get("status") == "PASS" and training.get("dataset_samples") == 3_072_000
        and training.get("batch_size") == 64 and training.get("drop_last") is True
        and training.get("optimizer_updates_per_epoch") == UPDATES_PER_PASS
        and training.get("actual_loop_epoch_indices") == [0, 1, 2]
        and training.get("actual_passes") == 3
        and training.get("expected_optimizer_updates") == TOTAL_UPDATES
    )
    probe = prep.get("actual_probing_loader", {}) if isinstance(prep, dict) else {}
    check("three_pass_preflight", passes_ok, "CPU preflight counts differ from 3 x 48,000 updates")
    check("evaluation_preflight", probe.get("status") == "PASS" and probe.get("total_training_passes") == 50,
          "official probing data preflight is missing or incomplete")

    cfg = prep.get("stages", {}).get("config_composition", {}).get("config", {}) if isinstance(prep, dict) else {}
    eval_ok = (
        cfg.get("quick_debug") is False and cfg.get("eval_levels") == "medium"
        and cfg.get("eval_envs") == 100 and cfg.get("eval_steps") == 200
        and cfg.get("eval_batch") == 20 and cfg.get("replan_every") == 1
        and cfg.get("mppi_samples") == 2000
    )
    check("native_eval_config", eval_ok, "CPU-composed native evaluation differs from the freeze")

    log = bounded_log_text(report / "job.log")
    job_id_ok = re.search(rf"(?m)^job_id={re.escape(job_id)}\s*$", log) is not None
    source_marker_ok = re.search(rf"(?m)^source_sha={SOURCE_SHA}\s*$", log) is not None
    start_match = re.search(r"(?m)^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s*$", log)
    started = None
    if start_match:
        try:
            started = datetime.fromisoformat(start_match.group(1).replace("Z", "+00:00"))
        except ValueError:
            pass
    check("formal_job_header", job_id_ok and source_marker_ok and started is not None,
          "job log lacks the requested job id, pinned source marker, or UTC start time")

    checkpoint_path = run / f"epoch=2_sample_step={FINAL_SAMPLE_STEP}.ckpt"
    summary_path = run / f"summary_epoch=2_sample_step={FINAL_SAMPLE_STEP}.json"
    checkpoint = metadata(checkpoint_path, started)
    summary_file = metadata(summary_path, started)
    check("fresh_final_checkpoint", checkpoint is not None and checkpoint["bytes"] > 0
          and checkpoint.get("written_during_job") is True,
          "official final checkpoint metadata is missing, empty, or predates this job")

    summary = read_json(summary_path)
    final_step_ok = isinstance(summary, dict) and summary.get("custom_step") == FINAL_STEP
    metric_fields_ok = isinstance(summary, dict) and all(key in summary for key in METRICS)
    check("final_native_summary", summary_file is not None and summary_file["bytes"] > 0
          and summary_file.get("written_during_job") is True and final_step_ok and metric_fields_ok,
          f"fresh official summary must contain custom_step={FINAL_STEP} and all six wall_medium fields")

    status = key_values(report / "runner_status.txt")
    finished = None
    if status.get("finished_utc"):
        try:
            finished = datetime.fromisoformat(status["finished_utc"].replace("Z", "+00:00"))
        except ValueError:
            pass
    log_exit = re.findall(r"(?m)^training_and_native_eval_exit_code=(-?\d+)\s*$", log)
    native_exit_ok = (
        status.get("job_id") == job_id and status.get("process_exit_code") == "0"
        and finished is not None and started is not None and finished >= started
        and bool(log_exit) and log_exit[-1] == "0"
    )
    check("native_evaluation_returned", native_exit_ok,
          "native process trap/zero process exit is absent from this job")
    check("pbs_job_exit", pbs_exit_status == 0,
          f"actual PBS Exit_status must be 0, got {pbs_exit_status}")

    native_values = {key: json_metric(summary[key]) for key in METRICS} if metric_fields_ok else {}
    quality_failures: list[str] = []
    if metric_fields_ok:
        for key in METRICS:
            if not finite_number(summary[key]):
                quality_failures.append(f"{key} is undefined/non-finite in native MPCReport output")
        if all(finite_number(summary[key]) for key in METRICS):
            if summary["wall_medium_planning_error_mean"] < 0:
                quality_failures.append("planning_error_mean is negative")
            for key in ("wall_medium_success_rate", "wall_medium_cross_wall_rate", "wall_medium_init_plan_cross_wall_rate"):
                if not 0 <= summary[key] <= 1:
                    quality_failures.append(f"{key} is outside [0,1]")
            if not 0 <= summary["wall_medium_avg_termination_step"] <= 200:
                quality_failures.append("avg_termination_step is outside [0,200]")
            if not math.isclose(summary["wall_medium_planning_error_mean_rmse"] ** 2,
                                summary["wall_medium_planning_error_mean"], rel_tol=1e-6, abs_tol=1e-8):
                quality_failures.append("RMSE squared does not match planning_error_mean")

    complete = not incomplete
    result = {
        "schema": "pldm-native-result-v1",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "job_id": job_id,
        "pbs_exit_status": pbs_exit_status,
        "source_sha": SOURCE_SHA,
        "run_name": RUN_NAME,
        "job_started_utc": started.isoformat() if started else None,
        "job_finished_utc": finished.isoformat() if finished else None,
        "supporting_log_markers": {
            "native_eval_start_visible_in_bounded_head_tail": "evaluating planning level medium for 100 envs" in log,
            "mpc_timing_visible_in_bounded_head_tail": "mpc planning took" in log,
        },
        "native_training": {"passes": 3, "updates_per_pass": UPDATES_PER_PASS,
                            "total_updates": TOTAL_UPDATES, "expected_final_custom_step": FINAL_STEP,
                            "observed_final_custom_step": summary.get("custom_step") if isinstance(summary, dict) else None},
        "checkpoint_metadata": checkpoint,
        "native_final_evaluation": {
            "level": "medium", "environments": 100, "steps_per_environment": 200,
            "chunk_size": 20, "native_planner_invocations": 1000,
            "mppi_samples": 2000, "planning_horizon": 96, "replan_every": 1,
            "summary_path": str(summary_path), "summary_metadata": summary_file,
            "metrics": native_values,
        },
        "quality_status": "NOT_ASSESSED" if not metric_fields_ok else ("PASS" if not quality_failures else "FAIL"),
        "quality_failures": quality_failures,
        "checks": checks,
        "incomplete_reasons": incomplete,
    }
    (report / "RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result, 0 if complete else 2


def self_check() -> None:
    """Synthetic stdlib-only checks: zero-success remains complete; missing eval fails."""
    with tempfile.TemporaryDirectory(prefix="pldm-result-check-") as tmp:
        root = Path(tmp)
        job_id = "25570000.pbs101"
        report, run = root / "reports" / job_id, root / "checkpoints" / RUN_NAME
        report.mkdir(parents=True)
        run.mkdir(parents=True)
        freeze = {"source": {"commit": SOURCE_SHA}}
        prep = {
            "status": "PASS", "source_sha": SOURCE_SHA,
            "actual_training_loader": {"status": "PASS", "dataset_samples": 3_072_000, "batch_size": 64,
                "drop_last": True, "optimizer_updates_per_epoch": UPDATES_PER_PASS,
                "actual_loop_epoch_indices": [0, 1, 2], "actual_passes": 3,
                "expected_optimizer_updates": TOTAL_UPDATES},
            "actual_probing_loader": {"status": "PASS", "total_training_passes": 50},
            "stages": {"config_composition": {"config": {"quick_debug": False, "eval_levels": "medium",
                "eval_envs": 100, "eval_steps": 200, "eval_batch": 20, "replan_every": 1, "mppi_samples": 2000}}},
        }
        docs = {"FREEZE.json": freeze, "PREPARATION.json": prep,
                "source_identity.json": {"source_sha": SOURCE_SHA},
                "staged_source_identity.json": {"source_sha": SOURCE_SHA, "algorithm_or_config_changes": []}}
        for name, value in docs.items():
            (report / name).write_text(json.dumps(value), encoding="utf-8")
        started = datetime.now(timezone.utc).replace(microsecond=0)
        (report / "job.log").write_text(
            f"job_id={job_id}\nsource_sha={SOURCE_SHA}\n{started.isoformat().replace('+00:00','Z')}\n"
            + "x" * 70_000 + "\nevaluating planning level medium for 100 envs\nmpc planning took 00:01\n"
            + "x" * 70_000 + "\ntraining_and_native_eval_exit_code=0\n", encoding="utf-8")
        (report / "runner_status.txt").write_text(
            f"job_id={job_id}\nprocess_exit_code=0\nfinished_utc={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8")
        (run / f"epoch=2_sample_step={FINAL_SAMPLE_STEP}.ckpt").write_bytes(b"metadata-only")
        metrics = {"custom_step": FINAL_STEP, "wall_medium_planning_error_mean": 0.25,
                   "wall_medium_planning_error_mean_rmse": 0.5, "wall_medium_success_rate": 0.0,
                   "wall_medium_cross_wall_rate": 0.4, "wall_medium_init_plan_cross_wall_rate": 0.5,
                   "wall_medium_avg_termination_step": 64.0}
        summary_path = run / f"summary_epoch=2_sample_step={FINAL_SAMPLE_STEP}.json"
        summary_path.write_text(json.dumps(metrics), encoding="utf-8")
        result, code = collect(root, job_id, 0)
        assert code == 0 and result["status"] == "COMPLETE" and result["quality_status"] == "PASS"
        assert not result["supporting_log_markers"]["native_eval_start_visible_in_bounded_head_tail"]
        metrics["wall_medium_cross_wall_rate"] = float("nan")
        summary_path.write_text(json.dumps(metrics), encoding="utf-8")
        result, code = collect(root, job_id, 0)
        assert code == 0 and result["status"] == "COMPLETE" and result["quality_status"] == "FAIL"
        result, code = collect(root, job_id, 1)
        assert code != 0 and result["status"] == "INCOMPLETE"
        summary_path.unlink()
        result, code = collect(root, job_id, 0)
        assert code != 0 and result["status"] == "INCOMPLETE"
    print("PLDM_NATIVE_RESULT_SELFCHECK=PASS")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--pbs-exit-status", type=int)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if not args.job_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.job_id):
        parser.error("--job-id must be the actual PBS job id")
    if args.pbs_exit_status is None:
        parser.error("--pbs-exit-status is required to preserve formal job provenance")
    result, code = collect(DEFAULT_ROOT, args.job_id, args.pbs_exit_status)
    print(f"PLDM_NATIVE_RESULT_{result['status']} {DEFAULT_ROOT / 'reports' / args.job_id / 'RESULT.json'}")
    for reason in result["incomplete_reasons"]:
        print(f"INCOMPLETE: {reason}")
    for reason in result["quality_failures"]:
        print(f"QUALITY_FAIL: {reason}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
