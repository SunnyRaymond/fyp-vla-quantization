"""Small-file, host-key-verified PBS control for decoded-weight transport."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "fastwam-libero-plus-pilot"
REMOTE_HELPER = PILOT / "remote.py"
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-transport-20261022"
PROTOCOL_ID = "fastwam-vq-r128-c80-transport-v1"
USER = "yguo017"
JOB_NAME = "fwvr128tr"
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
RESULT_FILES = ("progress.json", "summary.json", "training.json", "RESULTS.zh.md")
ARMS = ("Comp8",)
PROBE_ARMS = ("Seq_A4", "Seq_to_Joint_A4", "Joint_A4", "Joint_to_Seq_A4",
              "Seq_noA4", "Seq_to_Joint_noA4", "Joint_noA4", "Joint_to_Seq_noA4")
STAGES = ("probe_0", "probe_1", "gate_0", "finalize_0")
STAGE_UNITS = {"probe_0": 80, "probe_1": 80, "gate_0": 1, "finalize_0": 1}
RECEIPTS = ("source_receipt.json", "unit_receipt.json", "gate_receipt.json")
MAX_FORMAL_PARITY = 1e-6
MAX_ORTHOGONALITY = 5e-5


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("protocol.json does not match the frozen decoded-weight transport protocol")
    if value.get("remote_root") != ROOT:
        raise RuntimeError("protocol.json names an unexpected remote root")
    expected = {"queue": QUEUE, "walltime": "08:00:00", "gpus": 2, "ncpus": 32,
                "memory_gb": 220, "execution_queue": "g2"}
    allocation = value.get("allocation") or {}
    if any(allocation.get(key) != setting for key, setting in expected.items()):
        raise RuntimeError("protocol.json allocation differs from the frozen dual-GPU request")
    source = value.get("source_dual") or {}
    if (source.get("root") != "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-dual-20261022"
            or source.get("protocol") != "fastwam-vq-r128-c80-dual-v1"
            or source.get("job") != "25912619.pbs101"
            or source.get("arms") != ["Seq8", "Joint8"]):
        raise RuntimeError("protocol.json source dual binding differs from the completed frozen job")
    if value.get("probe_arms") != list(PROBE_ARMS) or value.get("training_arms") != ["Comp8", "Uncomp8"]:
        raise RuntimeError("protocol.json arm lists differ from the frozen transport protocol")
    stages = value.get("stages") or {}
    if {name: (stages.get(name) or {}).get("units") for name in STAGES} != STAGE_UNITS:
        raise RuntimeError("protocol.json stage units differ from the frozen transport protocol")
    return value


def load_execution() -> dict:
    value = json.loads((HERE / "execution.json").read_text(encoding="utf-8"))
    if (value.get("mode") != "comp_only_reuse_joint8" or value.get("resume_arm") != "Comp8"
            or value.get("source_cancelled_job") != "25919309.pbs101"
            or value.get("qdel_return_code") != 0 or value.get("confirmed_terminal_exit") != "141"
            or value.get("reference") != {"arm": "Joint8", "job": "25912619.pbs101", "kind": "historical_reuse"}
            or value.get("allocation") != {"gpus": 1, "ncpus": 16, "memory_gb": 110,
                "walltime": WALLTIME, "execution_queue": "g1"}):
        raise RuntimeError("User-authorized single-GPU execution amendment differs")
    return value


def load_remote_helper():
    sys.path.insert(0, str(PILOT))
    spec = importlib.util.spec_from_file_location("fastwam_vq_r128_c80_transport_remote", REMOTE_HELPER)
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
    paths = sorted((path for path in HERE.glob("*.py") if not path.name.startswith("test_")),
                   key=lambda path: path.name)
    load_execution()
    paths.extend(HERE / name for name in ("protocol.json", "execution.json", "run.pbs", "environment.sh"))
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
    if handle.get("remote_root") != ROOT:
        raise RuntimeError("Saved PBS handle belongs to a different remote root")
    if handle.get("protocol") != PROTOCOL_ID:
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


def parse_qstat_fields(output: str) -> dict:
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "queue", "exec_host", "comment", "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return fields


def scheduler(remote, transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, f"qstat -xf {shlex.quote(job_id)}", timeout=35)
    fields = parse_qstat_fields(output)
    return {"job_id": job_id, "query_rc": code, **fields}


def retain_terminal(current: dict, saved: dict | None, job_id: str) -> tuple[dict, dict | None]:
    if current.get("query_rc") == 0:
        current = {**current, "terminal_from_history": False}
        if current.get("job_state") in ("F", "X") and current.get("Exit_status") is not None:
            saved = {key: current.get(key) for key in (
                "job_id", "query_rc", "job_state", "Exit_status", "Job_Name", "queue", "exec_host",
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


def remote_size(sftp, path: str) -> int | None:
    try:
        return sftp.stat(path).st_size
    except OSError as exc:
        if getattr(exc, "errno", None) == 2:
            return None
        raise


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
    comparison_size = remote_size(sftp, RESULTS + "/comparison.json")
    result_files["comparison.json"] = comparison_size is not None
    receipts = {
        name: read_remote_json(sftp, RESULTS + "/" + name)
        for name in RECEIPTS
    }
    stage_progress = {}
    stage_summaries = {}
    for stage in STAGES:
        base = RESULTS + "/stages/" + stage
        stage_progress[stage] = read_remote_json(sftp, base + "/progress.json")
        stage_summaries[stage] = read_remote_json(sftp, base + "/summary.json")
    arm_progress = {}
    arm_summaries = {}
    checkpoint_sizes = {}
    for arm in ARMS:
        base = RESULTS + "/arms/" + arm
        arm_progress[arm] = read_remote_json(sftp, base + "/progress.json")
        arm_summaries[arm] = read_remote_json(sftp, base + "/summary.json")
        checkpoint_sizes[arm] = {
            name: remote_size(sftp, base + "/" + name)
            for name in ("state.pt", "deployment.pt")
        }
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
    return {
        "progress": progress, "summary": summary, "worker_summary": worker_summary,
        "result_files": result_files, "result_json": result_json,
        "comparison_size": comparison_size, "receipts": receipts,
        "stage_progress": stage_progress, "stage_summaries": stage_summaries,
        "arm_progress": arm_progress, "arm_summaries": arm_summaries,
        "checkpoint_sizes": checkpoint_sizes,
        "artifact_exit_code": exit_code, "phase_complete": marker,
        "log_tail": log_tail,
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


def within_limit(value, limit: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value <= limit


def verification_failures(observation: dict, protocol: dict) -> list[str]:
    scheduler_state = observation.get("scheduler") or {}
    handle = observation.get("handle") or {}
    summary = observation.get("summary") or {}
    progress = observation.get("progress") or {}
    failures = []
    job_id = handle.get("job_id")
    if not HANDLE.fullmatch(str(job_id or "")):
        failures.append("confirmed PBS handle is invalid")
    if handle.get("remote_root") != ROOT or handle.get("protocol") != PROTOCOL_ID:
        failures.append("confirmed PBS handle belongs to a different root or protocol")
    if scheduler_state.get("job_id") != job_id:
        failures.append("scheduler job handle does not match the confirmed PBS handle")
    if scheduler_state.get("query_rc") != 0 and scheduler_state.get("terminal_from_history") is not True:
        failures.append("PBS terminal state is not confirmed by the current query or saved history")
    if scheduler_state.get("job_state") not in ("F", "X"):
        failures.append("PBS job is not in terminal F/X state")
    if str(scheduler_state.get("Exit_status")) != "0":
        failures.append("PBS Exit_status is not 0")
    if scheduler_state.get("queue") != "g1":
        failures.append("PBS queue does not confirm single-GPU execution on g1")
    if observation.get("artifact_exit_code") != 0:
        failures.append("artifact exit_code.txt is not 0")
    if observation.get("phase_complete") is not True:
        failures.append("PHASE_COMPLETE marker is missing")
    if summary.get("status") != "complete" or summary.get("complete") is not True:
        failures.append("results/summary.json is not complete")
    if summary.get("protocol") != PROTOCOL_ID or protocol.get("protocol") != PROTOCOL_ID:
        failures.append("summary protocol does not match the frozen protocol")
    if summary.get("pbs_jobid") != job_id:
        failures.append("summary pbs_jobid does not match the confirmed PBS handle")
    if type(summary.get("target_linears")) is not int or summary["target_linears"] != 614:
        failures.append("summary target_linears is not 614")
    if type(summary.get("new_teacher_queries")) is not int or summary["new_teacher_queries"] != 0:
        failures.append("summary new_teacher_queries is not 0")
    if summary.get("source_dual_binding") is not True:
        failures.append("summary source_dual_binding did not pass")
    if type(summary.get("allocation_gpus")) is not int or summary["allocation_gpus"] != 1:
        failures.append("summary allocation_gpus is not 1")
    if summary.get("reused_reference") != {"Uncomp8": "Joint8", "job": "25912619.pbs101"}:
        failures.append("historical Joint8 reference is not identified")
    if summary.get("execution_amendment") != load_execution():
        failures.append("single-GPU execution amendment is missing or changed")
    if summary.get("encoding_scope") != "decoded_weight_transport_reference":
        failures.append("summary encoding_scope is incorrect")

    gates = summary.get("gates") or {}
    for name in ("source", "fp32_invariance", "gradient_coupling", "initial_parity", "frozen"):
        if gates.get(name) is not True:
            failures.append(f"summary gate {name} did not pass")
    if not within_limit(gates.get("formal_parity_max_error"), MAX_FORMAL_PARITY):
        failures.append(f"summary formal parity error exceeds {MAX_FORMAL_PARITY:g}")
    if not within_limit(gates.get("deployment_reload_max_error"), MAX_FORMAL_PARITY):
        failures.append(f"summary deployment reload error exceeds {MAX_FORMAL_PARITY:g}")
    if not within_limit(gates.get("orthogonality_max_error"), MAX_ORTHOGONALITY):
        failures.append(f"summary orthogonality error exceeds {MAX_ORTHOGONALITY:g}")

    if summary.get("comparison_file") != "results/comparison.json":
        failures.append("summary comparison_file does not name results/comparison.json")
    if summary.get("comparison_complete") is not True:
        failures.append("summary does not confirm a complete comparison")
    if progress.get("protocol") != PROTOCOL_ID or progress.get("complete") is not True:
        failures.append("results/progress.json does not confirm completion for this protocol")
    if progress.get("pbs_jobid") != job_id:
        failures.append("results/progress.json PBS handle does not match")

    probe_counts = summary.get("probe_counts") or {}
    for probe in PROBE_ARMS:
        record = probe_counts.get(probe) or {}
        for split in ("calibration", "heldout"):
            if type(record.get(split)) is not int or record[split] != 80:
                failures.append(f"summary probe_counts[{probe}].{split} is not 80")
    endpoints = summary.get("endpoints") or {}
    for arm in ARMS:
        record = endpoints.get(arm) or {}
        for key, expected in (("calibration", 80), ("heldout", 80), ("updates", 8), ("exposures", 80)):
            if type(record.get(key)) is not int or record[key] != expected:
                failures.append(f"summary endpoints[{arm}].{key} is not {expected}")

    for relative in RECEIPTS:
        receipt = (observation.get("receipts") or {}).get(relative) or {}
        if receipt.get("complete") is not True or receipt.get("passed", True) is not True:
            failures.append(f"results/{relative} did not pass")
    stage_summaries = observation.get("stage_summaries") or {}
    stage_progress = observation.get("stage_progress") or {}
    for stage, units in STAGE_UNITS.items():
        record = stage_summaries.get(stage) or {}
        if (record.get("protocol") != PROTOCOL_ID or record.get("status") != "complete"
                or record.get("complete") is not True or record.get("pbs_jobid") != job_id
                or type(record.get("units")) is not int or record["units"] != units):
            failures.append(f"results/stages/{stage}/summary.json is incomplete or has wrong units")
        stage_receipt = stage_progress.get(stage) or {}
        if (stage_receipt.get("protocol") != PROTOCOL_ID or stage_receipt.get("status") != "complete"
                or stage_receipt.get("pbs_jobid") != job_id):
            failures.append(f"results/stages/{stage}/progress.json is incomplete")

    arm_summaries = observation.get("arm_summaries") or {}
    arm_progress = observation.get("arm_progress") or {}
    checkpoint_sizes = observation.get("checkpoint_sizes") or {}
    for arm in ARMS:
        record = arm_summaries.get(arm) or {}
        expected_fields = {"protocol": PROTOCOL_ID, "arm": arm, "status": "complete",
                           "complete": True, "pbs_jobid": job_id, "updates": 8,
                           "exposures": 80, "calibration": 80, "heldout": 80,
                           "all_parameters_frozen": True}
        for name, expected in expected_fields.items():
            if record.get(name) != expected:
                failures.append(f"results/arms/{arm}/summary.json {name} is not {expected}")
        if not within_limit(record.get("formal_infer_max_abs"), MAX_FORMAL_PARITY):
            failures.append(f"results/arms/{arm}/formal_infer_max_abs exceeds {MAX_FORMAL_PARITY:g}")
        if not within_limit(record.get("deployment_reload_max_abs"), MAX_FORMAL_PARITY):
            failures.append(f"results/arms/{arm}/deployment_reload_max_abs exceeds {MAX_FORMAL_PARITY:g}")
        if not within_limit(record.get("initial_parity_max_abs"), MAX_FORMAL_PARITY):
            failures.append(f"results/arms/{arm}/initial_parity_max_abs exceeds {MAX_FORMAL_PARITY:g}")
        orth = record.get("orthogonality") or {}
        if any(not within_limit(orth.get(name), MAX_ORTHOGONALITY) for name in ("action", "video")):
            failures.append(f"results/arms/{arm}/orthogonality exceeds {MAX_ORTHOGONALITY:g}")
        arm_receipt = arm_progress.get(arm) or {}
        if (arm_receipt.get("status") != "complete" or arm_receipt.get("protocol") != PROTOCOL_ID
                or arm_receipt.get("pbs_jobid") != job_id):
            failures.append(f"results/arms/{arm}/progress.json is incomplete")
        for name in ("state.pt", "deployment.pt"):
            size = (checkpoint_sizes.get(arm) or {}).get(name)
            if type(size) is not int or size <= 0:
                failures.append(f"results/arms/{arm}/{name} is missing or empty")

    worker_summary = observation.get("worker_summary") or {}
    if worker_summary.get("status") != "complete" or worker_summary.get("complete") is not True:
        failures.append("artifacts/worker_summary.json is not complete")
    if worker_summary != summary:
        failures.append("artifacts/worker_summary.json differs from results/summary.json")
    for name in RESULT_FILES:
        if observation.get("result_files", {}).get(name) is not True:
            failures.append(f"results/{name} is missing")
    if observation.get("comparison_size") is None:
        failures.append("results/comparison.json is missing")
    training = observation.get("result_json", {}).get("training.json") or {}
    if training and training.get("protocol") != PROTOCOL_ID:
        failures.append("results/training.json protocol does not match")
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
    authorized_cancel = (handle.get("job_id") == "25919309.pbs101"
        and str(scheduler_state.get("Exit_status")) == "141"
        and load_execution().get("qdel_return_code") == 0
        and (observation.get("checkpoint_sizes", {}).get("Comp8") or {}).get("state.pt", 0) > 0)
    if observation.get("artifact_exit_code") is None and not authorized_cancel:
        failures.append("artifact exit_code.txt is missing; refusing an unconfirmed resume")
    if summary.get("status") == "complete" or summary.get("complete") is True:
        failures.append("results/summary.json is complete; refusing to resume")
    if progress.get("complete") is True:
        failures.append("results/progress.json claims completion; refusing to resume")
    if protocol.get("protocol") != PROTOCOL_ID:
        failures.append("local protocol does not match the frozen protocol ID")
    return failures


def checkpoint_resume_mode(
    progress: dict | None, checkpoint_sizes: dict, arm_progress: dict | None = None,
    training: dict | None = None,
) -> str:
    state_sizes = [(checkpoint_sizes.get(arm) or {}).get("state.pt") for arm in ARMS]
    if any(size == 0 for size in state_sizes):
        raise RuntimeError("An arm state.pt checkpoint exists but is empty; safe resume is unavailable")
    if any(type(size) is int and size > 0 for size in state_sizes):
        return "checkpoint"
    if isinstance(training, dict) and any(training.get(arm) for arm in ARMS) or (
        isinstance(training, dict) and training.get("history")
    ):
        raise RuntimeError("Training history exists but no arm state.pt checkpoint is available")
    for arm in ARMS:
        record = (arm_progress or {}).get(arm) or {}
        updates = record.get("updates", record.get("completed_updates", 0))
        deployment_size = (checkpoint_sizes.get(arm) or {}).get("deployment.pt")
        has_deployment = type(deployment_size) is int and deployment_size > 0
        if type(updates) is int and updates > 0 and not has_deployment:
            raise RuntimeError(f"{arm} has committed updates but no checkpoint or deployment is available")
    if isinstance(progress, dict) and progress.get("complete") is True:
        raise RuntimeError("Progress claims completion; refusing to resume without an optimizer checkpoint")
    return "stage-replay" if isinstance(progress, dict) else "fresh"


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
        # The confirmed terminal handle above permits a code repair; the protocol stays frozen.
        transferred = transfer_files(sftp, include_protocol=False)
        assert_remote_files(sftp)
    state['last_code_transfer'] = {'from_job_id': handle['job_id'], 'checked_utc': utc_now(),
                                   'files': transferred}
    save_submission_state(remote, transport, state)
    mode = checkpoint_resume_mode(
        observation.get("progress"), observation.get("checkpoint_sizes") or {},
        observation.get("arm_progress"), (observation.get("result_json") or {}).get("training.json"),
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
            "resume_mode": mode,
            "checkpoints": {arm: RESULTS + "/arms/" + arm + "/state.pt"
                            for arm, sizes in (observation.get("checkpoint_sizes") or {}).items()
                            if type(sizes.get("state.pt")) is int and sizes["state.pt"] > 0}}


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
        for relative in RECEIPTS:
            data = read_small_bytes(sftp, RESULTS + "/" + relative)
            if data is not None:
                target = destination / "evidence" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for group, names in (("stages", STAGES), ("arms", ARMS)):
            for name in names:
                for leaf in ("progress.json", "summary.json"):
                    relative = f"{group}/{name}/{leaf}"
                    data = read_small_bytes(sftp, RESULTS + "/" + relative)
                    if data is not None:
                        target = destination / "evidence" / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
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
        "checked_utc", "handle", "scheduler", "artifact_exit_code", "phase_complete", "checkpoint_sizes",
    )}
    metadata["comparison_bytes"] = result.get("comparison_size")
    save_local(destination / "pull_metadata.json", metadata)
    return {"job_id": job_id, "local_directory": str(destination), "metadata": metadata}


def brief_result(action: str, result: dict) -> dict:
    if action not in ("status", "verify"):
        return result
    summary = result.get("summary") or {}
    stages = result.get("stage_summaries") or {}
    arms = result.get("arm_summaries") or {}
    return {
        "action": action, "checked_utc": result.get("checked_utc"),
        "handle": result.get("handle"), "scheduler": result.get("scheduler"),
        "verified": result.get("verified"), "failures": result.get("failures"),
        "summary": summary,
        "progress": {key: (result.get("progress") or {}).get(key) for key in ("stage", "status", "complete")},
        "stage_progress": result.get("stage_progress"),
        "arm_progress": result.get("arm_progress"),
        "receipts": {name: {key: (value or {}).get(key) for key in ("complete", "passed", "expected", "available")}
                     for name, value in (result.get("receipts") or {}).items()},
        "stages": {name: {key: (value or {}).get(key) for key in ("status", "complete", "units")}
                   for name, value in stages.items()},
        "arms": {name: {key: (value or {}).get(key) for key in
                        ("status", "complete", "updates", "exposures", "calibration", "heldout")}
                 for name, value in arms.items()},
        "comparison_bytes": result.get("comparison_size"),
        "artifact_exit_code": result.get("artifact_exit_code"),
        "phase_complete": result.get("phase_complete"),
        "log_tail": result.get("log_tail"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "resume", "status", "verify", "pull"))
    args = parser.parse_args()
    load_protocol()
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
        print(json.dumps(brief_result(args.action, result), indent=2, ensure_ascii=False))
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
