"""Run the three independent Cosmos bootstrap stages inside one PBS allocation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path("/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024")
JOB_NAME = "cpvq24b2"
EXECUTION_QUEUE = "g2"
STAGES = ("teacher", "perf_bf16", "perf_spinquant")
PERF_ARMS = {"perf_bf16": "bf16", "perf_spinquant": "spinquant_w4a4",
             "perf_svdquant": "svdquant_w4a4", "perf_viditq": "viditq_w4a4"}


class BundleError(RuntimeError):
    pass


def _pbs_attribute(output: str, name: str) -> str | None:
    pattern = re.compile(r"^\s*" + re.escape(name) + r"\s*=\s*(.*?)\s*$")
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def verify_allocation(root: Path, job_name: str = JOB_NAME,
                      execution_queue: str = EXECUTION_QUEUE) -> tuple[str, str, str]:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    host = socket.gethostname().split(".")[0]
    if not job_id or not nodefile:
        raise BundleError("PBS_JOBID and PBS_NODEFILE are required")
    if "login" in host.lower():
        raise BundleError(f"Refusing login node {host}")
    node_path = Path(nodefile)
    if not node_path.is_file():
        raise BundleError(f"PBS_NODEFILE is missing: {node_path}")
    allocated = {line.strip().split(".")[0] for line in node_path.read_text().splitlines() if line.strip()}
    if host not in allocated:
        raise BundleError(f"Current host {host} is not listed in PBS_NODEFILE")
    result = subprocess.run(["qstat", "-f", job_id], check=False, capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        raise BundleError(f"qstat could not verify PBS job {job_id}: {result.stderr.strip()}")
    state = _pbs_attribute(result.stdout, "job_state")
    name = _pbs_attribute(result.stdout, "Job_Name")
    queue = _pbs_attribute(result.stdout, "queue")
    if state != "R" or name != job_name or queue != execution_queue:
        raise BundleError(f"PBS identity mismatch: state={state!r}, Job_Name={name!r}, queue={queue!r}")
    return job_id, host, queue


def parse_visible_devices(value: str, expected_count: int = 3) -> list[str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if len(devices) != expected_count or len(set(devices)) != expected_count:
        raise BundleError(f"PBS CUDA_VISIBLE_DEVICES must contain exactly {expected_count} unique devices")
    return devices


def probe_devices(tokens: list[str], expected_count: int = 3) -> list[dict[str, str]]:
    rows = []
    for token in tokens:
        result = subprocess.run(
            ["nvidia-smi", "-i", token, "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=30)
        uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode or len(uuids) != 1:
            raise BundleError(f"Cannot map allocated CUDA device {token!r} to one GPU UUID")
        rows.append({"token": token, "uuid": uuids[0]})
    if len({row["uuid"] for row in rows}) != expected_count:
        raise BundleError(f"CUDA_VISIBLE_DEVICES aliases fewer than {expected_count} unique physical GPUs")
    return rows


def expected_panel(protocol: dict) -> tuple[int, int, list[int]]:
    performance = protocol.get("performance", {})
    query_count = len(performance.get("case_ids", [])) * len(performance.get("seed_indices", []))
    blocks, repetitions = performance.get("blocks"), performance.get("repetitions")
    if type(blocks) is not int or type(repetitions) is not int:
        raise BundleError("Frozen performance protocol lacks integer block/repetition counts")
    samples = query_count * blocks * repetitions
    if query_count != 80 or samples != 720:
        raise BundleError(f"Expected frozen performance panel 80 queries/720 samples, got {query_count}/{samples}")
    cases = protocol.get("calibration", {}).get("case_ids", [])
    if len(cases) != 80 or len(set(cases)) != 80:
        raise BundleError("Expected exactly 80 unique teacher cases in the frozen protocol")
    return query_count, samples, list(cases)


def summary_path(root: Path, stage: str, protocol: dict) -> Path:
    if stage == "teacher":
        return root / "results" / "teacher" / "summary.json"
    arm = protocol.get("stage_arms", {}).get(stage, PERF_ARMS.get(stage))
    if arm != PERF_ARMS.get(stage):
        raise BundleError(f"Unexpected frozen arm for {stage}: {arm!r}")
    return root / "results" / "performance" / arm / "summary.json"


def completed_summary(stage: str, value: object, protocol: dict) -> bool:
    if not isinstance(value, dict) or value.get("status") != "complete" or value.get("protocol") != protocol.get("protocol"):
        return False
    if not isinstance(value.get("pbs_job_id"), str) or not value["pbs_job_id"]:
        return False
    if stage == "teacher":
        cases = protocol["calibration"]["case_ids"]
        completed = value.get("completed_cases")
        expected = value.get("expected_cases")
        return (value.get("stage") == "teacher" and isinstance(completed, list)
                and isinstance(expected, list) and expected == cases and completed == cases
                and all(type(case) is int for case in completed) and len(set(completed)) == 80
                and isinstance(value.get("target_modules"), list)
                and len(value["target_modules"]) == 280)
    arm = protocol.get("stage_arms", {}).get(stage)
    query_count, sample_count, _ = expected_panel(protocol)
    coverage = value.get("coverage", {})
    if not isinstance(coverage, dict):
        return False
    performance = protocol["performance"]
    by_query = coverage.get("by_query")
    expected_repetitions = performance["blocks"] * performance["repetitions"]
    expected_pairs = {(case, seed) for case in performance["case_ids"]
                      for seed in performance["seed_indices"]}
    if not isinstance(by_query, list) or len(by_query) != query_count:
        return False
    observed_pairs = set()
    for row in by_query:
        if not isinstance(row, dict):
            return False
        case_id, seed_index = row.get("case_id"), row.get("seed_index")
        if type(case_id) is not int or type(seed_index) is not int:
            return False
        pair = (case_id, seed_index)
        if (pair in observed_pairs or pair not in expected_pairs
                or type(row.get("completed_repetitions")) is not int
                or type(row.get("expected_repetitions")) is not int
                or row.get("completed_repetitions") != expected_repetitions
                or row.get("expected_repetitions") != expected_repetitions):
            return False
        observed_pairs.add(pair)
    return (arm == PERF_ARMS.get(stage) and value.get("arm") == arm
            and value.get("complete") is True
            and observed_pairs == expected_pairs
            and coverage.get("queries") == query_count
            and coverage.get("completed_queries") == query_count
            and coverage.get("samples") == sample_count
            and coverage.get("expected_samples") == sample_count
            and coverage.get("missing") == [])


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _protocol_file(root: Path) -> Path:
    snapshot = Path(__file__).resolve().with_name("protocol.json")
    return snapshot if snapshot.is_file() else root / "protocol.json"


def _source_job(value: dict | None) -> str | None:
    job_id = value.get("pbs_job_id") if isinstance(value, dict) else None
    return job_id if isinstance(job_id, str) and job_id else None


def classify_outcomes(stages: dict[str, dict], job_id: str) -> tuple[str, int, bool]:
    teacher = stages["teacher"]
    teacher_incomplete = (teacher.get("status") == "checkpointed"
                          and teacher.get("source_job") == job_id
                          and teacher.get("exit_code") == 2)
    all_complete = all(row.get("status") == "complete" and row.get("exit_code") == 0
                       for row in stages.values())
    teacher_valid = ((teacher.get("status") == "complete" and teacher.get("exit_code") == 0)
                     or teacher_incomplete)
    perf_valid = all(
        (row.get("status") == "complete" and row.get("exit_code") == 0)
        or (row.get("status") == "checkpointed" and row.get("exit_code") == 0)
        for stage, row in stages.items() if stage != "teacher")
    if all_complete:
        return "complete", 0, True
    if not perf_valid or not teacher_valid:
        return "failed", 1, False
    if teacher_incomplete:
        return "checkpointed", 2, False
    return "checkpointed", 0, False


def run_bundle(root: Path = ROOT) -> int:
    job_id, host, queue = verify_allocation(root)
    protocol_path = _protocol_file(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not isinstance(protocol.get("protocol"), str):
        raise BundleError("protocol.json has no protocol identity")
    expected_panel(protocol)
    devices = probe_devices(parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    artifacts = root / "artifacts" / job_id
    result_dir = root / "results" / "bootstrap_bundle"
    parent_summary = result_dir / "summary.json"
    stages: dict[str, dict] = {}
    pending: list[str] = []

    for index, stage in enumerate(STAGES):
        arm = "teacher" if stage == "teacher" else PERF_ARMS[stage]
        path = summary_path(root, stage, protocol)
        value = _read_json(path) if path.is_file() else None
        device = devices[index]
        if completed_summary(stage, value, protocol):
            stages[stage] = {"arm": arm, "status": "complete", "summary_status": "complete",
                             "exit_code": 0, "source_job": _source_job(value),
                             "summary_path": str(path), "skipped": True,
                             "cuda_visible_device": device["token"], "gpu_uuid": device["uuid"]}
            print(f"BUNDLE_STAGE_SKIP stage={stage} arm={arm} source_job={stages[stage]['source_job']}",
                  flush=True)
        else:
            stages[stage] = {"arm": arm, "status": "pending", "summary_status": None,
                             "exit_code": None, "source_job": None,
                             "summary_path": str(path), "skipped": False,
                             "cuda_visible_device": device["token"], "gpu_uuid": device["uuid"]}
            pending.append(stage)

    required_queue = protocol.get("resources", {}).get("bootstrap_bundle", {}).get("required_execution_queue")
    if required_queue != EXECUTION_QUEUE:
        raise BundleError(f"Protocol execution queue must be {EXECUTION_QUEUE!r}, got {required_queue!r}")
    summary = {"protocol": protocol["protocol"], "protocol_file": str(protocol_path),
               "pbs_job_id": job_id, "job_name": JOB_NAME, "host": host,
               "execution_queue": queue,
               "status": "running", "complete": False, "stages": stages,
               "created_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(parent_summary, summary)

    # The atomic claim precedes all model-loading child processes.
    race_script = artifacts / "control_sources" / "bootstrap_race.py"
    if not race_script.is_file():
        race_script = root / "bootstrap_race.py"
    if not race_script.is_file():
        raise BundleError(f"Missing bootstrap race controller: {race_script}")
    print("BUNDLE_RACE_CLAIM candidate=bundle", flush=True)
    claim = subprocess.run(["python3", str(race_script), "claim", "--candidate", "bundle"],
                           cwd=root, check=False)
    if claim.returncode:
        raise BundleError(f"Bundle race claim failed ({claim.returncode}); no model stage launched")

    processes: dict[str, subprocess.Popen] = {}
    for stage in pending:
        device = stages[stage]["cuda_visible_device"]
        child_env = os.environ.copy()
        child_env["COSMOS_PARENT_BUNDLE"] = JOB_NAME
        child_env["CUDA_VISIBLE_DEVICES"] = device
        command = ["/bin/bash", str(root / "job.sh"), stage]
        print(f"BUNDLE_STAGE_START stage={stage} arm={stages[stage]['arm']} "
              f"CUDA_VISIBLE_DEVICES={device} GPU_UUID={stages[stage]['gpu_uuid']}", flush=True)
        try:
            processes[stage] = subprocess.Popen(command, cwd=root, env=child_env)
        except OSError as exc:
            stages[stage].update(status="failed", exit_code=127, error=f"Popen: {exc}")
    _atomic_json(parent_summary, summary)

    try:
        while processes:
            for stage, process in list(processes.items()):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                del processes[stage]
                path = Path(stages[stage]["summary_path"])
                value = _read_json(path) if path.is_file() else None
                same_job = _source_job(value) == job_id
                summary_status = value.get("status") if isinstance(value, dict) and same_job else None
                stages[stage].update(exit_code=exit_code, source_job=_source_job(value) if same_job else None,
                                     summary_status=summary_status,
                                     status=summary_status or "missing",
                                     skipped=False)
                if not same_job:
                    stages[stage]["error"] = "Stage summary is missing or belongs to another PBS job"
                elif summary_status == "complete" and not completed_summary(stage, value, protocol):
                    stages[stage]["status"] = "invalid_complete_summary"
                    stages[stage]["error"] = "Complete stage summary failed protocol/arm/count checks"
                print(f"BUNDLE_STAGE_EXIT stage={stage} exit_code={exit_code} "
                      f"summary_status={stages[stage]['summary_status']} "
                      f"source_job={stages[stage]['source_job']}", flush=True)
                _atomic_json(parent_summary, summary)
            if processes:
                time.sleep(1)
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

    status, code, all_complete = classify_outcomes(stages, job_id)
    summary.update(status=status, complete=all_complete,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    _atomic_json(parent_summary, summary)
    if all_complete:
        (artifacts / "PIPELINE_CHECKPOINTED").unlink(missing_ok=True)
        (artifacts / "PIPELINE_COMPLETE").touch()
    elif status == "checkpointed":
        (artifacts / "PIPELINE_CHECKPOINTED").touch()
    print(f"BUNDLE_STATUS={status} EXIT_CODE={code} SUMMARY={parent_summary}", flush=True)
    return code


def main() -> int:
    job_id = os.environ.get("PBS_JOBID", "")
    artifacts = ROOT / "artifacts" / job_id if job_id else None
    try:
        return run_bundle(ROOT)
    except Exception as exc:
        print(f"BUNDLE_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if artifacts is not None:
            summary_path = ROOT / "results" / "bootstrap_bundle" / "summary.json"
            value = _read_json(summary_path) if summary_path.is_file() else None
            if not isinstance(value, dict) or value.get("pbs_job_id") != job_id:
                value = {"protocol": None, "protocol_file": str(ROOT / "protocol.json"),
                         "pbs_job_id": job_id, "job_name": JOB_NAME, "stages": {}}
            for row in value.get("stages", {}).values():
                if row.get("status") == "pending":
                    row.update(status="not_started", error="Bundle stopped before stage launch")
            value.update(status="failed", complete=False, error=f"{type(exc).__name__}: {exc}",
                         finished_utc=datetime.now(timezone.utc).isoformat())
            try:
                protocol_file = _protocol_file(ROOT)
                protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
                value["protocol"] = protocol.get("protocol")
                value["protocol_file"] = str(protocol_file)
            except (OSError, json.JSONDecodeError):
                pass
            _atomic_json(summary_path, value)
            if value.get("pbs_job_id") == job_id and value.get("status") == "checkpointed":
                (artifacts / "PIPELINE_CHECKPOINTED").parent.mkdir(parents=True, exist_ok=True)
                (artifacts / "PIPELINE_CHECKPOINTED").touch()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
