"""Race-gated two-GPU SVDQuant/ViDiT-Q follow-up bundle."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import bootstrap_bundle as shared


ROOT = Path("/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024")
JOB_NAME = "cpvq24c2"
QUEUE = "g2"
RACE_DIRECTORY = "control/baseline_followup_race"
VIDITQ_TARGET_WEIGHT_NUMEL = 1_761_607_680
STAGES = ("perf_svdquant", "vidcal", "perf_viditq")
ARM = {"perf_svdquant": "svdquant_w4a4", "vidcal": "viditq_w4a4",
       "perf_viditq": "viditq_w4a4"}


def _summary_path(root: Path, stage: str, protocol: dict) -> Path:
    if stage == "vidcal":
        return root / "results" / "viditq" / "summary.json"
    return Path(shared.summary_path(root, stage, protocol))


def _vidcal_complete(value: object, protocol: dict, job_id: str) -> bool:
    if (not isinstance(value, dict) or value.get("status") != "complete"
            or value.get("protocol") != protocol.get("protocol")
            or value.get("pbs_job_id") != job_id or value.get("arm") != ARM["vidcal"]):
        return False
    expected = value.get("expected_modules")
    return (type(expected) is int and expected == 280
            and value.get("completed_modules") == expected
            and type(value.get("target_weight_numel")) is int
            and value["target_weight_numel"] == VIDITQ_TARGET_WEIGHT_NUMEL
            and type(value.get("completed_weight_numel")) is int
            and value.get("completed_weight_numel") == value["target_weight_numel"]
            and value.get("bpw_within_target_tolerance") is True)


def _valid_complete(stage: str, value: object, protocol: dict, job_id: str) -> bool:
    if not isinstance(value, dict) or value.get("pbs_job_id") != job_id:
        return False
    if stage == "vidcal":
        return _vidcal_complete(value, protocol, job_id)
    return shared.completed_summary(stage, value, protocol)


def _record_stage(root: Path, stage: str, exit_code: int, protocol: dict,
                  job_id: str, row: dict) -> None:
    path = _summary_path(root, stage, protocol)
    marker = root / "artifacts" / job_id / "stages" / stage / "PIPELINE_COMPLETE"
    value = shared._read_json(path) if path.is_file() else None
    same_job = isinstance(value, dict) and value.get("pbs_job_id") == job_id
    status = value.get("status") if same_job else None
    row.update(exit_code=exit_code, summary_pbs_job_id=job_id if same_job else None,
               summary_status=status, summary_path=str(path),
               complete_marker_path=str(marker), complete_marker_exists=marker.is_file())
    if (exit_code == 0 and status == "complete" and marker.is_file()
            and _valid_complete(stage, value, protocol, job_id)):
        row["status"] = "complete"
    elif exit_code in (0, 2) and status == "checkpointed" and same_job:
        row["status"] = "checkpointed"
    elif status == "complete":
        row["status"] = "invalid_complete_summary"
        row["error"] = "Summary failed current-job, protocol, arm, or coverage checks"
    else:
        row["status"] = "failed"
        row["error"] = "Stage exit or summary did not validate"


def _launch(root: Path, stage: str, device: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env["COSMOS_PARENT_BUNDLE"] = JOB_NAME
    env["CUDA_VISIBLE_DEVICES"] = device["token"]
    command = ["/bin/bash", str(root / "job.sh"), stage]
    print(f"BUNDLE_STAGE_START stage={stage} arm={ARM[stage]} "
          f"CUDA_VISIBLE_DEVICES={device['token']} GPU_UUID={device['uuid']}", flush=True)
    return subprocess.Popen(command, cwd=root, env=env)


def _finish_status(stages: dict[str, dict]) -> tuple[str, int, bool]:
    complete = all(row.get("status") == "complete" and row.get("exit_code") == 0
                   and row.get("summary_pbs_job_id") == row.get("pbs_job_id")
                   and row.get("complete_marker_exists") is True
                   for row in stages.values())
    if complete:
        return "complete", 0, True
    if any(row.get("status") in ("failed", "invalid_complete_summary") for row in stages.values()):
        return "failed", 1, False
    return "checkpointed", 2, False


def run_bundle(root: Path = ROOT) -> int:
    job_id, host, queue = shared.verify_allocation(root, JOB_NAME, QUEUE)
    protocol_path = shared._protocol_file(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not isinstance(protocol.get("protocol"), str):
        raise shared.BundleError("protocol.json has no protocol identity")
    shared.expected_panel(protocol)
    devices = shared.probe_devices(shared.parse_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), expected_count=2), expected_count=2)
    profile_resources = protocol.get("resources", {}).get("baseline_followup_bundle")
    if profile_resources and profile_resources.get("required_execution_queue") != QUEUE:
        raise shared.BundleError(f"Protocol execution queue must be {QUEUE!r}")

    artifacts = root / "artifacts" / job_id
    summary_path = root / "results" / "baseline_followup_bundle" / "summary.json"
    stages = {stage: {"arm": ARM[stage], "status": "pending", "exit_code": None,
                      "pbs_job_id": job_id, "summary_pbs_job_id": None,
                      "summary_path": str(_summary_path(root, stage, protocol)),
                      "complete_marker_path": str(root / "artifacts" / job_id / "stages" / stage
                                                   / "PIPELINE_COMPLETE"),
                      "complete_marker_exists": False}
              for stage in STAGES}
    summary = {"format": "cosmos_baseline_followup_bundle_v1",
               "protocol": protocol["protocol"], "protocol_file": str(protocol_path),
               "pbs_job_id": job_id, "job_name": JOB_NAME, "host": host,
               "execution_queue": queue, "status": "running", "complete": False,
               "stages": stages, "created_utc": datetime.now(timezone.utc).isoformat()}
    shared._atomic_json(summary_path, summary)

    race_script = artifacts / "control_sources" / "bootstrap_race.py"
    if not race_script.is_file():
        race_script = root / "bootstrap_race.py"
    if not race_script.is_file():
        raise shared.BundleError(f"Missing bootstrap race controller: {race_script}")
    claim = subprocess.run(["python3", str(race_script), "claim", "--candidate", "bundle",
                            "--race-directory", RACE_DIRECTORY], cwd=root, check=False)
    if claim.returncode:
        raise shared.BundleError(f"Baseline follow-up race claim failed ({claim.returncode})")

    processes: dict[str, subprocess.Popen] = {}
    try:
        for stage, device in (("perf_svdquant", devices[0]), ("vidcal", devices[1])):
            try:
                processes[stage] = _launch(root, stage, device)
            except OSError as exc:
                stages[stage].update(status="failed", exit_code=127, error=f"Popen: {exc}")
        vidcal_launched = False
        while processes:
            for stage, process in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                del processes[stage]
                _record_stage(root, stage, code, protocol, job_id, stages[stage])
                print(f"BUNDLE_STAGE_EXIT stage={stage} exit_code={code} "
                      f"summary_status={stages[stage]['summary_status']}", flush=True)
                shared._atomic_json(summary_path, summary)
                if stage == "vidcal":
                    if stages[stage]["status"] == "complete":
                        try:
                            processes["perf_viditq"] = _launch(root, "perf_viditq", devices[1])
                            vidcal_launched = True
                        except OSError as exc:
                            stages["perf_viditq"].update(status="failed", exit_code=127,
                                                         error=f"Popen: {exc}")
                    else:
                        stages["perf_viditq"].update(
                            status="not_started_dependency", exit_code=None,
                            error="VIDIT-Q performance requires a complete current-job calibration")
            if processes:
                time.sleep(1)
        if not vidcal_launched and stages["perf_viditq"].get("status") == "pending":
            stages["perf_viditq"].update(
                status="not_started_dependency", error="VIDIT-Q calibration did not complete")
    except BaseException:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise

    status, code, complete = _finish_status(stages)
    summary.update(status=status, complete=complete, exit_code=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    shared._atomic_json(summary_path, summary)
    if complete:
        (artifacts / "PIPELINE_CHECKPOINTED").unlink(missing_ok=True)
        (artifacts / "PIPELINE_COMPLETE").touch()
    elif status == "checkpointed":
        (artifacts / "PIPELINE_CHECKPOINTED").touch()
    print(f"BUNDLE_STATUS={status} EXIT_CODE={code} SUMMARY={summary_path}", flush=True)
    return code


def main() -> int:
    job_id = os.environ.get("PBS_JOBID", "")
    artifacts = ROOT / "artifacts" / job_id if job_id else None
    try:
        return run_bundle(ROOT)
    except Exception as exc:
        print(f"BUNDLE_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if artifacts is not None:
            path = ROOT / "results" / "baseline_followup_bundle" / "summary.json"
            value = shared._read_json(path) if path.is_file() else None
            if not isinstance(value, dict) or value.get("pbs_job_id") != job_id:
                value = {"format": "cosmos_baseline_followup_bundle_v1",
                         "protocol": None, "pbs_job_id": job_id, "job_name": JOB_NAME,
                         "stages": {}}
            for row in value.get("stages", {}).values():
                if row.get("status") == "pending":
                    row.update(status="not_started", error="Bundle stopped before stage launch")
            value.update(status="failed", complete=False,
                         exit_code=1,
                         error=f"{type(exc).__name__}: {exc}",
                         finished_utc=datetime.now(timezone.utc).isoformat())
            shared._atomic_json(path, value)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
