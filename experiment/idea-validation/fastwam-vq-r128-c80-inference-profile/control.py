"""Bounded PBS control for frozen C80 R128 inference export and GPU profiling."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex
import tempfile


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "r128_loss_pair_remote",
    HERE.parent / "fastwam-vq-r128-c80-loss-pair" / "control.py",
)
_base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_base)
remote = _base.remote

PROTOCOL_ID = "fastwam-vq-r128-c80-inference-profile-v1"
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-inference-profile-20261023-r3"
USER = "yguo017"
PROJECT = "personal-yguo017"
JOB_NAME = "vpinf0003"
QUEUE = "normal"
EXECUTION_QUEUE = "g1"
SELECT = "1:ncpus=16:mem=120gb:ngpus=1"
WALLTIME = "04:00:00"
LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
LOG_TAIL_BYTES = 8192
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
JOB_NAME_RE = re.compile(r"^vpinf0003$")
FILES = (
    "protocol.json", "run.pbs", "environment.sh", "profile_worker.py",
    "export_runtime.py", "analyze.py",
)
ALLOCATION = {
    "queue": QUEUE, "execution_queue": EXECUTION_QUEUE, "gpus": 1,
    "ncpus": 16, "memory_gb": 120, "walltime": WALLTIME,
}
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"
STATE_PATH = ROOT + "/control/state.json"


def direct_connect():
    """Use the established SSH helper, which retains host-key verification."""
    return remote.direct_connect()


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("protocol.json does not match the frozen inference-profile protocol")
    if value.get("remote_root") != ROOT or any(value.get("allocation", {}).get(k) != v for k,v in ALLOCATION.items()):
        raise RuntimeError("protocol.json root or allocation differs from the frozen request")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absent(exc: OSError) -> bool:
    return isinstance(exc, FileNotFoundError) or getattr(exc, "errno", None) == 2


def _small_bytes(sftp, path: str, limit: int = LIMIT) -> bytes | None:
    try:
        attrs = sftp.stat(path)
    except OSError as exc:
        if _absent(exc):
            return None
        raise
    if attrs.st_size > limit:
        raise RuntimeError(f"Remote control file exceeds 256 KiB ({limit} bytes): {path}")
    with sftp.file(path, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"Remote file grew beyond its {limit}-byte read limit: {path}")
    return data


def _json(sftp, path: str) -> dict | None:
    data = _small_bytes(sftp, path)
    if data is None:
        return None
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid remote JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _text(sftp, path: str, limit: int = LIMIT) -> str | None:
    data = _small_bytes(sftp, path, limit)
    return data.decode("utf-8", errors="replace").strip() if data is not None else None


def _write_json(sftp, path: str, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > LIMIT:
        raise RuntimeError("Remote control JSON exceeds 256 KiB")
    with sftp.file(path + ".tmp", "wb") as stream:
        stream.write(data)
    sftp.posix_rename(path + ".tmp", path)


def _local_bytes(name: str) -> bytes:
    path = HERE / name
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError(f"Required source is missing or exceeds 256 KiB: {name}")
    data = path.read_bytes().replace(b"\r\n", b"\n")
    data.decode("utf-8")
    return data


def _mkdirs(transport) -> None:
    paths = (ROOT + "/control", ARTIFACTS, RESULTS)
    code, output = remote.command(
        transport, "mkdir -p " + " ".join(shlex.quote(path) for path in paths), timeout=25
    )
    if code:
        raise RuntimeError("Cannot prepare small PBS control directories: " + output[:1000])


def _state(sftp) -> dict:
    value = _json(sftp, STATE_PATH) or {"schema_version": 1}
    if value.get("schema_version") != 1:
        raise RuntimeError("Unsupported remote submission state")
    return value


def _frozen(state: dict) -> bool:
    return bool(state.get("intent") or state.get("handle"))


def upload(transport) -> list[str]:
    """Upload only the six named, bounded source/control files."""
    _mkdirs(transport)
    load_protocol()
    uploaded = []
    with remote.sftp_client(transport) as sftp:
        frozen = _frozen(_state(sftp))
        for name in FILES:
            data = _local_bytes(name)
            target = ROOT + "/" + name
            previous = _small_bytes(sftp, target)
            if name == "protocol.json" and previous is not None and previous != data:
                raise RuntimeError("Remote frozen protocol bytes differ; refusing overwrite")
            if previous == data:
                continue
            if frozen:
                raise RuntimeError("Submission is frozen; refusing to change " + name)
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            uploaded.append(name)
    return uploaded


def _require_uploaded(transport) -> None:
    load_protocol()
    with remote.sftp_client(transport) as sftp:
        for name in FILES:
            if _small_bytes(sftp, ROOT + "/" + name) != _local_bytes(name):
                raise RuntimeError("Upload is incomplete or differs from local source: " + name)
        if _json(sftp, ROOT + "/protocol.json") != load_protocol():
            raise RuntimeError("Remote protocol differs from the frozen local protocol")


def _named_jobs(transport) -> list[str]:
    code, output = remote.command(
        transport,
        "qselect -x -u " + USER + " -N " + shlex.quote(JOB_NAME) + " 2>&1",
        timeout=35,
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("Exact-name PBS query failed; submission was not attempted")
    jobs = output.split()
    if any(not HANDLE.fullmatch(job) for job in jobs):
        raise RuntimeError("PBS returned an invalid exact-name job handle")
    return jobs


def _parse_qstat(output: str) -> dict:
    fields = {}
    for key in (
        "job_state", "Exit_status", "Job_Name", "queue", "exec_host",
        "resources_used.walltime", "Resource_List.walltime", "Resource_List.ngpus",
        "Resource_List.ncpus", "Resource_List.mem", "Resource_List.select",
    ):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return fields


def _scheduler(transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=35)
    return {"job_id": job_id, "query_rc": code, **_parse_qstat(output)}


def _save_state(transport, value: dict) -> None:
    with remote.sftp_client(transport) as sftp:
        _write_json(sftp, STATE_PATH, value)


def _recover(transport, state: dict) -> dict:
    intent = state.get("intent") or {}
    if intent.get("job_name") != JOB_NAME or intent.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("Durable PBS intent is malformed or belongs to another protocol")
    jobs = _named_jobs(transport)
    if len(jobs) != 1:
        raise RuntimeError("qsub outcome is unresolved; do not issue a second submission")
    job_id = jobs[0]
    scheduler = _scheduler(transport, job_id)
    if scheduler.get("query_rc") != 0 or scheduler.get("Job_Name") != JOB_NAME:
        raise RuntimeError("Exact-name PBS handle could not be verified")
    handle = {
        "remote_root": ROOT, "job_id": job_id, "job_name": JOB_NAME,
        "protocol": PROTOCOL_ID, "queue": QUEUE, "execution_queue": EXECUTION_QUEUE,
        "gpus": 1, "ncpus": 16, "memory_gb": 120, "walltime": WALLTIME,
        "status": "recovered", "confirmed_utc": utc_now(),
    }
    intent.update(job_id=job_id, status="confirmed", recovered_utc=utc_now())
    state.update(intent=intent, handle=handle)
    _save_state(transport, state)
    return handle


def _resolve(transport) -> tuple[dict | None, dict]:
    with remote.sftp_client(transport) as sftp:
        state = _state(sftp)
    handle = state.get("handle")
    if handle:
        if (not HANDLE.fullmatch(str(handle.get("job_id", "")))
                or handle.get("job_name") != JOB_NAME
                or handle.get("protocol") != PROTOCOL_ID
                or handle.get("remote_root") != ROOT):
            raise RuntimeError("Saved PBS handle is invalid or belongs to another job")
        return handle, state
    intent = state.get("intent")
    if intent:
        if intent.get("status") in ("intent_before_qsub", "qsub_outcome_unknown"):
            return _recover(transport, state), state
        raise RuntimeError("Unknown durable PBS intent state")
    return None, state


def _qsub_command() -> str:
    return (
        "qsub -N " + JOB_NAME + " -q " + QUEUE + " -P " + PROJECT
        + " -l select=" + shlex.quote(SELECT)
        + " -l walltime=" + WALLTIME
        + " -o " + shlex.quote(ARTIFACTS + "/" + JOB_NAME + ".pbs.out")
        + " -j oe " + shlex.quote(ROOT + "/run.pbs")
    )


def submit(transport) -> dict:
    _mkdirs(transport)
    protocol = load_protocol()
    handle, state = _resolve(transport)
    if handle:
        return {"existing": True, "handle": handle}
    if state.get("intent"):
        raise RuntimeError("An unresolved durable submission exists; qsub was not repeated")
    _require_uploaded(transport)
    if _named_jobs(transport):
        raise RuntimeError("Exact job name already exists without a matching durable intent")
    intent = {
        "protocol": protocol["protocol"], "job_name": JOB_NAME,
        "script": ROOT + "/run.pbs", "status": "intent_before_qsub",
        "created_utc": utc_now(), "queue": QUEUE, "select": SELECT,
        "walltime": WALLTIME,
    }
    state.update(intent=intent, handle=None)
    _save_state(transport, state)
    try:
        code, output = remote.command(transport, _qsub_command(), timeout=45)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        _save_state(transport, state)
        raise RuntimeError("qsub outcome is unknown; recover by exact job name before any retry") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", return_code=code,
                       response=output[-1000:], updated_utc=utc_now())
        _save_state(transport, state)
        raise RuntimeError("qsub outcome is unknown or rejected; do not submit again")
    handle = {
        "remote_root": ROOT, "job_id": job_id, "job_name": JOB_NAME,
        "protocol": protocol["protocol"], "queue": QUEUE,
        "execution_queue": EXECUTION_QUEUE, "select": SELECT,
        "gpus": 1, "ncpus": 16, "memory_gb": 120, "walltime": WALLTIME,
        "status": "confirmed", "confirmed_utc": utc_now(),
    }
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    state.update(intent=intent, handle=handle)
    _save_state(transport, state)
    return {"handle": handle}


def _exists(sftp, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except OSError as exc:
        if _absent(exc):
            return False
        raise


def _read_evidence(sftp, handle: dict) -> dict:
    job_id = handle["job_id"]
    artifact = ARTIFACTS + "/" + job_id
    return {
        "progress": _json(sftp, RESULTS + "/progress.json"),
        "summary": _json(sftp, RESULTS + "/summary.json"),
        "artifact": {
            "exit_code": _text(sftp, artifact + "/exit_code.txt", 64),
            "pipeline_complete": _exists(sftp, artifact + "/PIPELINE_COMPLETE"),
        },
    }


def complete_gate(value: dict) -> bool:
    """Require the matching successful PBS job and all inference-only receipts."""
    try:
        handle, scheduler = value["handle"], value["scheduler"]
        summary, artifact = value["summary"], value["artifact"]
        job_id = handle["job_id"]
        return (
            HANDLE.fullmatch(str(job_id)) is not None
            and handle.get("job_name") == JOB_NAME
            and handle.get("protocol") == PROTOCOL_ID
            and handle.get("remote_root") == ROOT
            and scheduler.get("query_rc") == 0
            and scheduler.get("job_state") == "F"
            and scheduler.get("Exit_status") == "0"
            and scheduler.get("Job_Name") == JOB_NAME
            and scheduler.get("queue") == EXECUTION_QUEUE
            and scheduler.get("Resource_List.walltime") == WALLTIME
            and artifact.get("exit_code") == "0"
            and artifact.get("pipeline_complete") is True
            and isinstance(summary, dict)
            and summary.get("protocol") == PROTOCOL_ID
            and summary.get("pbs_jobid") == job_id
            and summary.get("complete") is True
            and summary.get("export_complete") is True
            and summary.get("parity_passed") is True
            and summary.get("profile_complete") is True
            and type(summary.get("new_training_updates")) is int
            and summary.get("new_training_updates") == 0
        )
    except (KeyError, TypeError, AttributeError):
        return False


def status(transport) -> dict:
    handle, _state_value = _resolve(transport)
    result = {
        "checked_utc": utc_now(), "handle": handle, "scheduler": None,
        "progress": None, "summary": None, "artifact": None,
        "log_tail": None,
    }
    if not handle:
        result["status"] = "not_submitted"
        return result
    with remote.sftp_client(transport) as sftp:
        result.update(_read_evidence(sftp, handle))
    scheduler = _scheduler(transport, handle["job_id"])
    result["scheduler"] = scheduler
    code, tail = remote.command(
        transport,
        "tail -c " + str(LOG_TAIL_BYTES) + " "
        + shlex.quote(ARTIFACTS + "/" + handle["job_id"] + "/job.log") + " 2>/dev/null",
        timeout=20,
    )
    result["log_tail"] = tail.encode("utf-8", errors="replace")[-LOG_TAIL_BYTES:].decode(
        "utf-8", errors="replace"
    ) if code == 0 else None
    result["completion_passed"] = complete_gate(result)
    if scheduler.get("query_rc") != 0:
        result["status"] = "query_inconclusive"
    elif scheduler.get("job_state") == "Q":
        result["status"] = "queued"
    elif scheduler.get("job_state") == "R":
        result["status"] = "running"
    elif scheduler.get("job_state") == "F" and result["completion_passed"]:
        result["status"] = "complete"
    elif scheduler.get("job_state") in ("F", "X"):
        result["status"] = "terminal_unverified"
    else:
        result["status"] = "unknown"
    return result


def _compact(value: dict) -> dict:
    handle = value.get("handle") or {}
    scheduler = value.get("scheduler") or {}
    progress = value.get("progress") or {}
    summary = value.get("summary") or {}
    return {
        "status": value.get("status"), "job_id": handle.get("job_id"),
        "job_state": scheduler.get("job_state"), "queue": scheduler.get("queue"),
        "exec_host": scheduler.get("exec_host"), "phase": progress.get("phase"),
        "profile_complete": summary.get("profile_complete"),
        "completion_passed": value.get("completion_passed", False),
    }


def logs(transport) -> dict:
    value = status(transport)
    if not value.get("handle"):
        raise RuntimeError("No PBS job handle is available")
    return {"job_id": value["handle"]["job_id"], "tail": value.get("log_tail")}


def _pull_bytes(sftp, path: str) -> tuple[bytes | None, str | None]:
    try:
        attrs = sftp.stat(path)
    except OSError as exc:
        if _absent(exc):
            return None, None
        raise
    if attrs.st_size > LIMIT:
        return None, "over_256KiB"
    with sftp.file(path, "rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        return None, "grew_over_256KiB"
    return data, None


def _save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(path)


def pull(transport) -> dict:
    value = status(transport)
    handle = value.get("handle")
    if not handle:
        raise RuntimeError("No PBS job handle is available")
    target = HERE / "pulled" / handle["job_id"]
    files: list[str] = []
    skipped: dict[str, str] = {}
    if value.get("progress") is not None:
        data = (json.dumps(value["progress"], indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        _save_bytes(target / "progress.json", data)
        files.append("progress.json")
    if value.get("summary") is not None:
        data = (json.dumps(value["summary"], indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        _save_bytes(target / "summary.json", data)
        files.append("summary.json")
    with remote.sftp_client(transport) as sftp:
        for name in ("storage.json", "parity.json"):
            remote_path = RESULTS + "/" + name
            data, reason = _pull_bytes(sftp, remote_path)
            if reason:
                skipped[name] = reason
            elif data is not None:
                _save_bytes(target / name, data)
                files.append(name)
        per_layer = RESULTS + "/per_layer"
        try:
            names = sftp.listdir(per_layer)
        except OSError as exc:
            if not _absent(exc):
                raise
            names = []
        for name in names:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name):
                continue
            remote_path = per_layer + "/" + name
            data, reason = _pull_bytes(sftp, remote_path)
            if reason:
                skipped["per_layer/" + name] = reason
            elif data is not None:
                _save_bytes(target / "per_layer" / name, data)
                files.append("per_layer/" + name)
        for name in ("RESULTS.zh.md", "report.md"):
            remote_path = RESULTS + "/" + name
            data, reason = _pull_bytes(sftp, remote_path)
            if reason:
                skipped[name] = reason
            elif data is not None:
                _save_bytes(target / name, data)
                files.append(name)
    compact = _compact(value)
    _save_bytes(target / "status.json", (json.dumps(compact, indent=2) + "\n").encode("utf-8"))
    files.append("status.json")
    return {**compact, "pulled_files": files, "skipped_oversized": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "logs", "pull"))
    args = parser.parse_args()
    jump, transport = direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": upload(transport)}
            code = 0
        elif args.action == "submit":
            result, code = submit(transport), 0
        elif args.action == "status":
            value = status(transport)
            result, code = _compact(value), 0
        elif args.action == "verify":
            value = status(transport)
            result = {**_compact(value), "verified_complete": complete_gate(value)}
            code = 0 if result["verified_complete"] else 1
        elif args.action == "logs":
            result, code = logs(transport), 0
        else:
            result, code = pull(transport), 0
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return code
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
