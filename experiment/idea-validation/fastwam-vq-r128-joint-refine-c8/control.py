"""Small-file, host-key-verified PBS control for the joint-refine R128 C8 experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "fastwam-libero-plus-pilot"
REMOTE_HELPER = PILOT / "remote.py"
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-r128-joint-refine-c8-20261021"
PROTOCOL_ID = "fastwam-vq-r128-joint-refine-c8-v1"
USER = "yguo017"
JOB_NAME = "fwvrjrefc8"
QUEUE = "normal"
PROJECT = "personal-yguo017"
SELECT = "1:ncpus=16:mem=110gb:ngpus=1"
WALLTIME = "04:00:00"
LIMIT = 256 * 1024
MAX_BYTES = LIMIT - 1
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"
STATE_PATH = ROOT + "/control/state.json"
PROGRESS_PATH = RESULTS + "/progress.json"
SUMMARY_PATH = RESULTS + "/summary.json"
CHECKPOINT_PATH = RESULTS + "/state.pt"
RESULT_FILES = ("progress.json", "summary.json", "comparison.json", "RESULTS.zh.md",
                "training.json", "selfcheck.json")
ARMS = ("H_W16A4", "R8_W16A4", "H_VQ0", "H_VQ8", "R8_VQ0", "R8_VQ8",
        "Joint8_R1e4", "H_currentpath", "Joint8")
HELDOUT_ARMS = ("H_W16A4", "R8_W16A4", "H_VQ8", "R8_VQ8", "Joint8_R1e4",
                "H_currentpath", "Joint8")


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("protocol.json does not match the frozen joint-refine R128 C8 protocol")
    if value.get("remote_root", ROOT) != ROOT:
        raise RuntimeError("protocol.json names an unexpected remote root")
    return value


def load_remote_helper():
    sys.path.insert(0, str(PILOT))
    spec = importlib.util.spec_from_file_location("fastwam_vq_r128_joint_c8_remote", REMOTE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the trusted ASPIRE2A helper: {REMOTE_HELPER}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_local(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read_local_state() -> dict:
    path = HERE / "control_state.json"
    if not path.is_file():
        return {}
    if path.stat().st_size >= LIMIT:
        raise RuntimeError("Local control state exceeds the 256 KiB limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Local control state must be a JSON object")
    return value


def write_remote_json(sftp, path: str, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) >= LIMIT:
        raise RuntimeError("Remote control JSON exceeds the 256 KiB limit")
    with sftp.file(path + ".tmp", "wb") as stream:
        stream.write(data)
    sftp.posix_rename(path + ".tmp", path)


def read_small_bytes(sftp, path: str, limit: int = MAX_BYTES) -> bytes | None:
    try:
        attrs = sftp.stat(path)
    except OSError as exc:
        if getattr(exc, "errno", None) == 2:
            return None
        raise
    if attrs.st_size > limit:
        raise RuntimeError(f"Refusing a remote control/result file larger than {limit} bytes: {path}")
    with sftp.file(path, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"Remote file grew beyond its bounded read: {path}")
    return data


def read_remote_json(sftp, path: str) -> dict | None:
    data = read_small_bytes(sftp, path)
    if data is None:
        return None
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def local_files() -> list[Path]:
    paths = sorted(HERE.glob("*.py"), key=lambda path: path.name)
    paths.extend(HERE / name for name in ("protocol.json", "run.pbs", "environment.sh"))
    if not all(path.is_file() and path.parent.resolve() == HERE.resolve() for path in paths):
        missing = [path.name for path in paths if not path.is_file()]
        raise FileNotFoundError("Missing upload files: " + ", ".join(missing))
    for path in paths:
        if path.stat().st_size >= LIMIT:
            raise RuntimeError(f"Only small code/control files below 256 KiB may be uploaded: {path.name}")
        path.read_bytes().decode("utf-8")
    return paths


def normalized_local_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def transfer_files(sftp, *, include_protocol: bool) -> list[dict]:
    sent = []
    for path in local_files():
        if path.name == "protocol.json" and not include_protocol:
            continue
        data = normalized_local_bytes(path)
        target = ROOT + "/" + path.name
        with sftp.file(target + ".tmp", "wb") as stream:
            stream.write(data)
        sftp.posix_rename(target + ".tmp", target)
        sent.append({"name": path.name, "bytes": len(data)})
    return sent


def assert_remote_files(sftp, *, include_protocol: bool = True) -> None:
    paths = local_files()
    local_python = {path.name for path in paths if path.suffix == ".py"}
    remote_python = {name for name in sftp.listdir(ROOT) if name.endswith(".py")}
    if remote_python != local_python:
        raise RuntimeError("Remote Python snapshot contains missing or unexpected files")
    for path in paths:
        if path.name == "protocol.json" and not include_protocol:
            continue
        remote = read_small_bytes(sftp, ROOT + "/" + path.name)
        if remote != normalized_local_bytes(path):
            raise RuntimeError(f"Remote file is missing or differs from the frozen local file: {path.name}")


def read_state(remote, transport) -> tuple[dict, dict]:
    with remote.sftp_client(transport) as sftp:
        remote_state = read_remote_json(sftp, STATE_PATH) or {}
    local_state = read_local_state()
    return remote_state, local_state


def merged_state(remote_state: dict, local_state: dict, handle: dict | None = None) -> dict:
    # Keep cluster receipts authoritative when the local copy is stale.
    state = {**local_state, **remote_state}
    if handle is not None:
        state["handle"] = handle
    for key in ("intent", "last_terminal", "last_verified"):
        if not state.get(key):
            state[key] = remote_state.get(key) or local_state.get(key)
    return state


def save_submission_state(remote, transport, state: dict) -> None:
    with remote.sftp_client(transport) as sftp:
        write_remote_json(sftp, STATE_PATH, state)
    save_local(HERE / "control_state.json", state)


def job_name(attempt: int) -> str:
    if type(attempt) is not int or attempt < 0:
        raise ValueError("Attempt number must be a nonnegative integer")
    return JOB_NAME if attempt == 0 else f"{JOB_NAME}-a{attempt}"


def attempt_number(handle: dict) -> int:
    attempt = handle.get("attempt")
    if type(attempt) is int and attempt >= 0:
        return attempt
    name = str(handle.get("job_name", ""))
    if name == JOB_NAME:
        return 0
    match = re.fullmatch(re.escape(JOB_NAME) + r"-a([1-9][0-9]*)", name)
    if match:
        return int(match.group(1))
    raise RuntimeError("Saved PBS handle has no valid attempt number or job name")


def assert_handle_scope(handle: dict) -> None:
    if handle.get("remote_root", ROOT) != ROOT:
        raise RuntimeError("Saved PBS handle belongs to a different remote root")
    if handle.get("protocol") not in (None, PROTOCOL_ID):
        raise RuntimeError("Saved PBS handle belongs to a different protocol")


def named_jobs(remote, transport, name: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Invalid PBS job name")
    code, output = remote.command(
        transport, f"qselect -x -u {USER} -N {shlex.quote(name)} 2>&1", timeout=35
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError(f"PBS history query failed for {name}; refusing to submit")
    matches = output.split()
    if any(not HANDLE.fullmatch(job_id) for job_id in matches):
        raise RuntimeError(f"PBS returned an invalid job handle for {name}")
    return matches


def recover_intent(remote, transport, state: dict) -> dict | None:
    intent = state.get("intent") or {}
    if intent.get("kind") not in ("submit", "resume") or intent.get("status") not in (
        "intent_before_qsub", "qsub_outcome_unknown",
    ):
        return None
    name = intent.get("job_name")
    if not isinstance(name, str):
        raise RuntimeError("Unresolved qsub intent has no candidate job name")
    matches = named_jobs(remote, transport, name)
    if len(matches) != 1:
        raise RuntimeError("qsub outcome is unresolved; PBS history did not identify exactly one job. Do not resubmit.")
    handle = {
        "job_id": matches[0], "job_name": name, "attempt": intent.get("attempt"),
        "remote_root": ROOT, "protocol": PROTOCOL_ID,
        "status": "recovered_from_pbs_history", "confirmed_utc": utc_now(),
    }
    intent.update(status="confirmed", job_id=matches[0], recovered_utc=utc_now())
    state.update(intent=intent, handle=handle)
    save_submission_state(remote, transport, state)
    return handle


def remote_protocol_matches(sftp, protocol: dict) -> None:
    if read_remote_json(sftp, ROOT + "/protocol.json") != protocol:
        raise RuntimeError("Remote protocol differs from the frozen local protocol")


def qsub_command(name: str) -> str:
    return (
        f"qsub -N {shlex.quote(name)} -q {QUEUE} -P {PROJECT} "
        f"-l select={SELECT} -l walltime={WALLTIME} "
        f"-o {shlex.quote(ARTIFACTS + '/' + name + '.pbs.out')} -j oe "
        f"{shlex.quote(ROOT + '/run.pbs')}"
    )


def submit(remote, transport) -> dict:
    protocol = load_protocol()
    remote_state, local_state = read_state(remote, transport)
    state = merged_state(remote_state, local_state)
    if state.get("handle"):
        handle = state["handle"]
        if not HANDLE.fullmatch(str(handle.get("job_id", ""))):
            raise RuntimeError("Saved PBS handle is invalid")
        assert_handle_scope(handle)
        return {"existing": True, "handle": handle}
    recovered = recover_intent(remote, transport, state)
    if recovered:
        return {"recovered": True, "handle": recovered}
    if state.get("intent"):
        raise RuntimeError("Unresolved submission intent exists; do not submit a second job")

    with remote.sftp_client(transport) as sftp:
        remote_protocol_matches(sftp, protocol)
        assert_remote_files(sftp)
    matches = named_jobs(remote, transport, JOB_NAME)
    if len(matches) > 1:
        raise RuntimeError("Multiple jobs already use this protocol job name; refusing submission")
    if matches:
        handle = {"job_id": matches[0], "job_name": JOB_NAME, "attempt": 0,
                  "remote_root": ROOT, "protocol": PROTOCOL_ID,
                  "status": "recovered_from_pbs_history", "confirmed_utc": utc_now()}
        intent = {"kind": "submit", "job_name": JOB_NAME, "status": "confirmed",
                  "job_id": matches[0], "recovered_utc": utc_now()}
        save_submission_state(remote, transport, {**state, "intent": intent, "handle": handle})
        return {"recovered": True, "handle": handle}

    intent = {
        "kind": "submit", "attempt": 0, "job_name": JOB_NAME,
        "script": ROOT + "/run.pbs", "created_utc": utc_now(),
        "status": "intent_before_qsub", "queue": QUEUE, "project": PROJECT,
        "select": SELECT, "walltime": WALLTIME,
    }
    state.update(intent=intent, handle=None)
    save_submission_state(remote, transport, state)
    try:
        code, output = remote.command(transport, qsub_command(JOB_NAME), timeout=40)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        save_submission_state(remote, transport, state)
        raise RuntimeError("qsub outcome is unknown; run submit again only to recover history, never to retry") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", response=output[:1000], return_code=code, updated_utc=utc_now())
        save_submission_state(remote, transport, state)
        raise RuntimeError("qsub outcome is unknown or rejected; recover the exact job from PBS history before another submission")
    handle = {"job_id": job_id, "job_name": JOB_NAME, "attempt": 0, "remote_root": ROOT,
              "protocol": PROTOCOL_ID, "status": "confirmed", "confirmed_utc": utc_now()}
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    state.update(intent=intent, handle=handle)
    save_submission_state(remote, transport, state)
    return {"handle": handle}


def scheduler(remote, transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, f"qstat -xf {shlex.quote(job_id)}", timeout=35)
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "exec_host", "comment", "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return {"job_id": job_id, "query_rc": code, **fields}


def retain_terminal(current: dict, saved: dict | None, job_id: str) -> tuple[dict, dict | None]:
    if current.get("query_rc") == 0:
        current = {**current, "terminal_from_history": False}
        if current.get("job_state") in ("F", "X") and current.get("Exit_status") is not None:
            saved = {key: current.get(key) for key in (
                "job_id", "query_rc", "job_state", "Exit_status", "Job_Name", "exec_host",
                "resources_used.walltime",
            )}
    elif isinstance(saved, dict) and saved.get("job_id") == job_id:
        current = {**saved, "current_query_rc": current.get("query_rc"), "terminal_from_history": True}
    return current, saved


def read_exit_code(sftp, path: str) -> int | None:
    data = read_small_bytes(sftp, path, 64)
    if data is None:
        return None
    match = re.fullmatch(rb"\s*(-?\d+)\s*", data)
    if not match:
        raise RuntimeError(f"Malformed exit code receipt: {path}")
    return int(match.group(1))


def artifact_observation(sftp, job_id: str) -> dict:
    artifact = ARTIFACTS + "/" + job_id
    progress = read_remote_json(sftp, PROGRESS_PATH)
    summary = read_remote_json(sftp, SUMMARY_PATH)
    worker_summary = read_remote_json(sftp, artifact + "/worker_summary.json")
    result_files = {}
    result_json = {}
    for name in RESULT_FILES:
        path = RESULTS + "/" + name
        data = read_small_bytes(sftp, path)
        result_files[name] = data is not None
        if data is not None and name.endswith(".json"):
            value = json.loads(data.decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected an object in results/{name}")
            result_json[name] = value
    exit_code = read_exit_code(sftp, artifact + "/exit_code.txt")
    try:
        sftp.stat(artifact + "/PHASE_COMPLETE")
        marker = True
    except OSError as exc:
        if getattr(exc, "errno", None) != 2:
            raise
        marker = False
    log_tail = None
    try:
        attrs = sftp.stat(artifact + "/job.log")
        with sftp.file(artifact + "/job.log", "rb") as stream:
            stream.seek(max(0, attrs.st_size - 8192))
            log_tail = stream.read(8192).decode("utf-8", errors="replace")
    except OSError as exc:
        if getattr(exc, "errno", None) != 2:
            raise
    try:
        checkpoint_size = sftp.stat(CHECKPOINT_PATH).st_size
    except OSError as exc:
        if getattr(exc, "errno", None) != 2:
            raise
        checkpoint_size = None
    return {
        "progress": progress, "summary": summary, "worker_summary": worker_summary,
        "result_files": result_files, "result_json": result_json,
        "artifact_exit_code": exit_code, "phase_complete": marker,
        "checkpoint_size": checkpoint_size, "log_tail": log_tail,
    }


def observe(remote, transport, handle: dict, state: dict) -> tuple[dict, dict]:
    job_id = str(handle.get("job_id", ""))
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    current, last_terminal = retain_terminal(scheduler(remote, transport, job_id), state.get("last_terminal"), job_id)
    with remote.sftp_client(transport) as sftp:
        artifacts = artifact_observation(sftp, job_id)
    updated = {**state, "handle": handle, "last_terminal": last_terminal}
    return {"handle": handle, "scheduler": current, **artifacts}, updated


def status(remote, transport) -> dict:
    protocol = load_protocol()
    remote_state, local_state = read_state(remote, transport)
    state = merged_state(remote_state, local_state)
    handle = state.get("handle")
    if not isinstance(handle, dict):
        recovered = recover_intent(remote, transport, state)
        if recovered is None:
            raise RuntimeError("No confirmed PBS handle is available")
        handle = recovered
        state["handle"] = handle
    assert_handle_scope(handle)
    with remote.sftp_client(transport) as sftp:
        remote_protocol_matches(sftp, protocol)
    observation, updated = observe(remote, transport, handle, state)
    checked = utc_now()
    updated["last_status_utc"] = checked
    save_submission_state(remote, transport, updated)
    return {"checked_utc": checked, **observation}


def query_count(value) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, dict):
        for key in ("completed", "count", "completed_queries"):
            if type(value.get(key)) is int:
                return value[key]
    return None


def verification_failures(observation: dict, protocol: dict) -> list[str]:
    scheduler_state = observation.get("scheduler") or {}
    handle = observation.get("handle") or {}
    summary = observation.get("summary") or {}
    progress = observation.get("progress") or {}
    failures = []
    if scheduler_state.get("job_state") not in ("F", "X"):
        failures.append("PBS job is not in terminal F/X state")
    if str(scheduler_state.get("Exit_status")) != "0":
        failures.append("PBS Exit_status is not 0")
    if observation.get("artifact_exit_code") != 0:
        failures.append("artifact exit_code.txt is not 0")
    if observation.get("phase_complete") is not True:
        failures.append("PHASE_COMPLETE marker is missing")
    if summary.get("status") != "complete" or summary.get("complete") is not True:
        failures.append("results/summary.json is not complete")
    if summary.get("protocol") != PROTOCOL_ID or protocol.get("protocol") != PROTOCOL_ID:
        failures.append("summary protocol does not match the frozen protocol")
    if summary.get("pbs_jobid") is not None and summary.get("pbs_jobid") != handle.get("job_id"):
        failures.append("summary pbs_jobid does not match the confirmed PBS handle")
    expected = {
        "target_linears": 614,
        "existing_R_updates": 8,
        "joint_updates": 8,
        "control_updates": 8,
        "diagnostic_heldout_queries_per_arm": 80,
        "joint_heldout_queries": 80,
        "calibration_arms": 9,
        "calibration_queries_per_arm": 8,
        "new_teacher_queries": 0,
    }
    for name, value in expected.items():
        if type(summary.get(name)) is not int or summary.get(name) != value:
            failures.append(f"summary {name} is not {value}")
    if summary.get("all_parameters_frozen") is not True:
        failures.append("summary does not confirm all parameters are frozen")
    comparison = summary.get("comparison_file")
    if not isinstance(comparison, str) or Path(comparison).name != "comparison.json":
        failures.append("summary comparison_file does not name comparison.json")
    if progress.get("complete") is not True or not isinstance(progress.get("stage"), str) or not progress.get("stage"):
        failures.append("results/progress.json does not confirm a named complete stage")
    if progress.get("joint_updates") != 8:
        failures.append("progress joint_updates is not 8")
    if progress.get("control_updates") != 8:
        failures.append("progress control_updates is not 8")
    completed = progress.get("completed_queries")
    if not isinstance(completed, dict):
        failures.append("progress completed_queries is missing")
    else:
        for arm in ARMS:
            if query_count(completed.get("calibration/" + arm)) != 8:
                failures.append(f"progress completed_queries[calibration/{arm}] is not 8")
        for arm in HELDOUT_ARMS:
            if query_count(completed.get("heldout/" + arm)) != 80:
                failures.append(f"progress completed_queries[heldout/{arm}] is not 80")
    worker_summary = observation.get("worker_summary") or {}
    if worker_summary.get("status") != "complete" or worker_summary.get("complete") is not True:
        failures.append("artifacts/worker_summary.json is not complete")
    if worker_summary != summary:
        failures.append("artifacts/worker_summary.json differs from results/summary.json")
    for name in RESULT_FILES:
        if observation.get("result_files", {}).get(name) is not True:
            failures.append(f"results/{name} is missing")
    comparison_report = observation.get("result_json", {}).get("comparison.json") or {}
    if comparison_report.get("protocol") != PROTOCOL_ID or comparison_report.get("complete") is not True:
        failures.append("results/comparison.json does not confirm this complete protocol")
    for name, value in {
        "heldout_queries_per_arm": 80,
        "heldout_arm_count": 7,
        "calibration_queries_per_arm": 8,
        "calibration_arm_count": 9,
    }.items():
        if type(comparison_report.get(name)) is not int or comparison_report.get(name) != value:
            failures.append(f"results/comparison.json {name} is not {value}")
    training = observation.get("result_json", {}).get("training.json") or {}
    history = training.get("history")
    if training.get("protocol") != PROTOCOL_ID or not isinstance(history, list) or len(history) != 8:
        failures.append("results/training.json does not contain eight frozen-protocol Joint8 candidate updates")
    control_history = training.get("control_history")
    if not isinstance(control_history, list) or len(control_history) != 8:
        failures.append("results/training.json does not contain eight H_currentpath control updates")
    return failures


def resume_failures(observation: dict, protocol: dict) -> list[str]:
    scheduler_state = observation.get("scheduler") or {}
    handle = observation.get("handle") or {}
    summary = observation.get("summary") or {}
    progress = observation.get("progress") or {}
    failures = []
    confirmed_terminal = scheduler_state.get("job_state") in ("F", "X") and (
        scheduler_state.get("query_rc") == 0 or (
            scheduler_state.get("terminal_from_history") is True
            and scheduler_state.get("job_id") == handle.get("job_id")
        )
    )
    if not confirmed_terminal:
        failures.append("current PBS handle is not confirmed terminal F/X")
    if scheduler_state.get("Exit_status") is None:
        failures.append("PBS terminal receipt has no Exit_status")
    if str(scheduler_state.get("Exit_status")) == "0":
        failures.append("PBS Exit_status is 0; refusing to rerun a successful job")
    if observation.get("artifact_exit_code") == 0:
        failures.append("artifact exit_code.txt is 0; refusing to rerun a successful worker")
    if observation.get("phase_complete") is True:
        failures.append("PHASE_COMPLETE exists; refusing to rerun a completed worker")
    if summary.get("status") == "complete" or summary.get("complete") is True:
        failures.append("results/summary.json is complete; refusing to resume")
    if progress.get("complete") is True:
        failures.append("results/progress.json claims completion; refusing to resume")
    if protocol.get("protocol") != PROTOCOL_ID:
        failures.append("local protocol does not match the frozen protocol ID")
    return failures


def checkpoint_resume_mode(
    progress: dict | None, checkpoint_size: int | None, training: dict | None = None
) -> str:
    if checkpoint_size is not None:
        if checkpoint_size <= 0:
            raise RuntimeError("results/state.pt exists but is empty; safe resume is unavailable")
        return "checkpoint"
    if isinstance(training, dict) and (training.get("history") or training.get("control_history")):
        raise RuntimeError("No checkpoint exists but training.json records committed optimizer updates")
    if not isinstance(progress, dict):
        return "fresh"
    if progress.get("complete") is True:
        raise RuntimeError("Progress claims completion; refusing to resume without its checkpoint")
    if type(progress.get("joint_updates")) is not int or progress["joint_updates"] != 0:
        raise RuntimeError("No checkpoint exists but progress does not confirm zero Joint8 updates")
    if type(progress.get("control_updates")) is not int or progress["control_updates"] != 0:
        raise RuntimeError("No checkpoint exists but progress does not confirm zero control updates")
    if progress.get("history") or progress.get("control_history"):
        raise RuntimeError("No checkpoint exists but progress records committed optimizer updates")
    return "fresh"


def resume(remote, transport) -> dict:
    protocol = load_protocol()
    remote_state, local_state = read_state(remote, transport)
    state = merged_state(remote_state, local_state)
    recovered = recover_intent(remote, transport, state)
    if recovered is not None:
        return {"recovered": True, "handle": recovered, "intent": state.get("intent")}
    handle = state.get("handle")
    if not isinstance(handle, dict) or not HANDLE.fullmatch(str(handle.get("job_id", ""))):
        raise RuntimeError("Resume requires a confirmed current PBS handle")
    assert_handle_scope(handle)
    intent = state.get("intent") or {}
    if intent.get("kind") == "resume" and intent.get("status") in ("resume_prepared", "resume_code_checked"):
        if intent.get("from_job_id") != handle.get("job_id"):
            raise RuntimeError("Pending resume intent does not belong to the current PBS handle")

    observation, state = observe(remote, transport, handle, state)
    state["last_status_utc"] = utc_now()
    save_submission_state(remote, transport, state)
    failures = resume_failures(observation, protocol)
    if failures:
        raise RuntimeError("Resume refused: " + "; ".join(failures))

    with remote.sftp_client(transport) as sftp:
        remote_protocol_matches(sftp, protocol)
        assert_remote_files(sftp)
    mode = checkpoint_resume_mode(
        observation.get("progress"), observation.get("checkpoint_size"),
        (observation.get("result_json") or {}).get("training.json"),
    )
    previous_attempt = attempt_number(handle)

    pending = state.get("intent") or {}
    if pending.get("kind") == "resume" and pending.get("status") in ("resume_prepared", "resume_code_checked"):
        attempt = pending.get("attempt")
        name = pending.get("job_name")
        if type(attempt) is not int or attempt <= previous_attempt or name != job_name(attempt):
            raise RuntimeError("Pending pre-qsub resume intent is malformed")
    else:
        attempt = previous_attempt + 1
        name = job_name(attempt)
        pending = {
            "kind": "resume", "status": "resume_prepared", "attempt": attempt,
            "job_name": name, "from_job_id": handle["job_id"], "script": ROOT + "/run.pbs",
            "created_utc": utc_now(), "queue": QUEUE, "project": PROJECT,
            "select": SELECT, "walltime": WALLTIME,
        }
        state["intent"] = pending
        save_submission_state(remote, transport, state)

    matches = named_jobs(remote, transport, name)
    if matches:
        raise RuntimeError(f"Candidate job name {name} already exists in PBS history; refusing duplicate submission")
    pending.update(status="resume_code_checked", code_checked_utc=utc_now())
    state.update(intent=pending, handle=handle)
    save_submission_state(remote, transport, state)
    pending.update(status="intent_before_qsub", qsub_started_utc=utc_now())
    save_submission_state(remote, transport, state)
    try:
        code, output = remote.command(transport, qsub_command(name), timeout=40)
    except Exception as exc:
        pending.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        save_submission_state(remote, transport, state)
        raise RuntimeError("Resume qsub outcome is unknown; a later resume may only recover its exact PBS job") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        pending.update(status="qsub_outcome_unknown", response=output[:1000], return_code=code, updated_utc=utc_now())
        save_submission_state(remote, transport, state)
        raise RuntimeError("Resume qsub outcome is unknown or rejected; do not submit another attempt")
    new_handle = {"job_id": job_id, "job_name": name, "attempt": attempt, "remote_root": ROOT,
                  "protocol": PROTOCOL_ID, "status": "confirmed", "confirmed_utc": utc_now()}
    pending.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    state.update(intent=pending, handle=new_handle)
    save_submission_state(remote, transport, state)
    return {"resumed": True, "from_job_id": handle["job_id"], "handle": new_handle,
            "resume_mode": mode, "checkpoint": CHECKPOINT_PATH if mode == "checkpoint" else None}


def verify(remote, transport) -> dict:
    protocol = load_protocol()
    result = status(remote, transport)
    failures = verification_failures(result, protocol)
    checked = utc_now()
    verified = not failures
    remote_state, local_state = read_state(remote, transport)
    state = merged_state(remote_state, local_state, result["handle"])
    if verified:
        state["last_verified"] = {"job_id": result["handle"]["job_id"], "verified_utc": checked,
                                  "scheduler": result["scheduler"], "summary": result["summary"]}
    state["last_status_utc"] = checked
    save_submission_state(remote, transport, state)
    return {"checked_utc": checked, "verified": verified, "failures": failures, **result}


def pull(remote, transport) -> dict:
    result = status(remote, transport)
    job_id = result["handle"]["job_id"]
    destination = HERE / "results" / job_id
    destination.mkdir(parents=True, exist_ok=True)
    with remote.sftp_client(transport) as sftp:
        for name in RESULT_FILES:
            data = read_small_bytes(sftp, RESULTS + "/" + name)
            if data is not None:
                (destination / name).write_bytes(data)
        artifact = ARTIFACTS + "/" + job_id
        artifact_dir = destination / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        for name in ("worker_summary.json", "exit_code.txt"):
            data = read_small_bytes(sftp, artifact + "/" + name)
            if data is not None:
                (artifact_dir / name).write_bytes(data)
        try:
            sftp.stat(artifact + "/PHASE_COMPLETE")
            (artifact_dir / "PHASE_COMPLETE").write_text("\n", encoding="ascii")
        except OSError as exc:
            if getattr(exc, "errno", None) != 2:
                raise
        log_tail = result.get("log_tail")
        if log_tail is not None:
            (artifact_dir / "job.log.tail").write_text(log_tail, encoding="utf-8")
    metadata = {key: result.get(key) for key in (
        "checked_utc", "handle", "scheduler", "artifact_exit_code", "phase_complete", "checkpoint_size",
    )}
    save_local(destination / "pull_metadata.json", metadata)
    return {"job_id": job_id, "local_directory": str(destination), "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "resume", "status", "verify", "pull"))
    args = parser.parse_args()
    helper = load_remote_helper()
    jump, transport = helper.direct_connect()
    try:
        if args.action == "upload":
            result = upload(helper, transport)
        elif args.action == "submit":
            result = submit(helper, transport)
        elif args.action == "resume":
            result = resume(helper, transport)
        elif args.action == "status":
            result = status(helper, transport)
        elif args.action == "verify":
            result = verify(helper, transport)
        else:
            result = pull(helper, transport)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("verified", True) else 1
    finally:
        transport.close()
        if jump:
            jump.close()


def upload(remote, transport) -> dict:
    protocol = load_protocol()
    code, output = remote.command(
        transport, "mkdir -p " + shlex.quote(ROOT + "/control") + " " +
        shlex.quote(ARTIFACTS) + " " + shlex.quote(RESULTS), timeout=25
    )
    if code:
        raise RuntimeError(f"Could not create remote control directories: {output[-500:]}")
    with remote.sftp_client(transport) as sftp:
        remote_state = read_remote_json(sftp, STATE_PATH) or {}
        local_state = read_local_state()
        if any(value.get(key) for value in (remote_state, local_state) for key in ("handle", "intent")):
            raise RuntimeError("A submission handle or intent exists; refusing to change files after possible qsub")
        sent = transfer_files(sftp, include_protocol=True)
        remote_protocol_matches(sftp, protocol)
        assert_remote_files(sftp)
    return {"uploaded": sent, "remote_root": ROOT, "protocol": PROTOCOL_ID}


if __name__ == "__main__":
    raise SystemExit(main())
