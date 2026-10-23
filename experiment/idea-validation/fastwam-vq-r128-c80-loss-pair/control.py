"""Small-file, host-key-verified PBS control for the frozen C80 R128 loss-pair calibration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import tempfile


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "r128_loss_pair_remote",
    HERE.parent / "fastwam-vq-c80-per-channel-plus" / "control.py",
)
_base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_base)
remote = _base.remote

LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
JOB_PREFIX = "r128lp"
JOB_NAME = JOB_PREFIX + "0001"
JOB_RE = re.compile(r"^r128lp[0-9]{4}$")
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
USER = "yguo017"
PROJECT = "personal-yguo017"
FILES = ("protocol.json", "run.pbs", "environment.sh", "control.py", "test_control.py", "worker.py", "finalize.py")
ARMS = ("az", "reader_kl")


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != "fastwam-vq-r128-c80-loss-pair-v1":
        raise RuntimeError("protocol.json does not match the frozen R128 C80 loss-pair protocol")
    if value.get("remote_root") != "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-loss-pair-20261022":
        raise RuntimeError("protocol.json names an unexpected remote root")
    expected = {
        "queue": "normal", "execution_queue": "g2", "gpus": 2, "ncpus": 32,
        "memory_gb": 220, "walltime": "12:00:00", "soft_deadline_seconds": 42600,
        "gpu_monitor_seconds": 15,
    }
    allocation = value.get("allocation") or {}
    if any(allocation.get(key) != item for key, item in expected.items()):
        raise RuntimeError("protocol.json allocation differs from the frozen calibration request")
    if value.get("arms") != list(ARMS):
        raise RuntimeError("protocol.json does not contain the frozen az/reader_kl pair")
    return value


PROTOCOL = load_protocol()
ROOT = PROTOCOL["remote_root"]
CONTROL = ROOT + "/control"
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"
STATE_PATH = CONTROL + "/submission.json"
RESOURCES = {
    "queue": "normal", "execution_queue": "g2",
    "select": "1:ncpus=32:mem=220gb:ngpus=2", "gpus": 2, "ncpus": 32,
    "memory_gb": 220, "walltime": "12:00:00",
}


def direct_connect():
    return remote.direct_connect()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absent(exc: OSError) -> bool:
    return getattr(exc, "errno", None) == 2


def _small_bytes(sftp, path: str) -> bytes | None:
    try:
        attrs = sftp.stat(path)
    except OSError as exc:
        if _absent(exc):
            return None
        raise
    if attrs.st_size > LIMIT:
        raise RuntimeError(f"Remote control file exceeds 256 KiB: {path}")
    with sftp.file(path, "rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise RuntimeError(f"Remote control file grew beyond 256 KiB: {path}")
    return data


def _json(sftp, path: str):
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
    data = _small_bytes(sftp, path)
    if data is None:
        return None
    if len(data) > limit:
        raise RuntimeError(f"Remote text receipt exceeds {limit} bytes: {path}")
    return data.decode("utf-8", errors="replace").strip()


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
        raise RuntimeError(f"Required small source is missing or too large: {name}")
    data = path.read_bytes().replace(b"\r\n", b"\n")
    data.decode("utf-8")
    return data


def _mkdirs(transport) -> None:
    paths = (CONTROL, ARTIFACTS, RESULTS)
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
    _mkdirs(transport)
    uploaded = []
    with remote.sftp_client(transport) as sftp:
        state = _state(sftp)
        frozen = _frozen(state)
        for name in FILES:
            data = _local_bytes(name)
            target = ROOT + "/" + name
            previous = _small_bytes(sftp, target)
            if name == "protocol.json" and previous is not None:
                if previous != data:
                    raise RuntimeError("Remote frozen protocol bytes differ; refusing overwrite")
                continue
            if previous == data:
                continue
            if frozen:
                raise RuntimeError("Submitted sources are frozen; remote differs from " + name)
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            uploaded.append(name)
    return uploaded


def _require_uploaded(transport) -> None:
    with remote.sftp_client(transport) as sftp:
        for name in FILES:
            if _small_bytes(sftp, ROOT + "/" + name) != _local_bytes(name):
                raise RuntimeError("Upload is incomplete or differs from local frozen source: " + name)
        if _json(sftp, ROOT + "/protocol.json") != load_protocol():
            raise RuntimeError("Remote frozen protocol differs from the local protocol")


def _named_jobs(transport, name: str) -> list[str]:
    if not JOB_RE.fullmatch(name):
        raise ValueError("Invalid exact PBS job name")
    code, output = remote.command(
        transport, "qselect -x -u " + USER + " -N " + shlex.quote(name) + " 2>&1", timeout=35
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS exact-name history query failed; submission was not attempted")
    jobs = output.split()
    if any(not HANDLE.fullmatch(job) for job in jobs):
        raise RuntimeError("PBS returned an invalid exact-name handle")
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
    if intent.get("job_name") != JOB_NAME or intent.get("protocol") != PROTOCOL["protocol"]:
        raise RuntimeError("Durable PBS intent is malformed or belongs to another protocol")
    jobs = _named_jobs(transport, JOB_NAME)
    if len(jobs) != 1:
        raise RuntimeError("qsub outcome is unresolved; do not issue a second submission")
    job_id = jobs[0]
    scheduler = _scheduler(transport, job_id)
    if scheduler.get("query_rc") != 0 or scheduler.get("Job_Name") != JOB_NAME:
        raise RuntimeError("Exact-name PBS handle could not be verified")
    handle = {**RESOURCES, "job_id": job_id, "job_name": JOB_NAME, "protocol": PROTOCOL["protocol"],
              "status": "recovered", "confirmed_utc": utc_now()}
    intent.update(job_id=job_id, status="confirmed", recovered_utc=utc_now())
    state.update(intent=intent, handle=handle)
    _save_state(transport, state)
    return handle


def _resolve(transport) -> tuple[dict | None, dict | None]:
    with remote.sftp_client(transport) as sftp:
        state = _state(sftp)
    handle = state.get("handle")
    if handle:
        if not HANDLE.fullmatch(str(handle.get("job_id", ""))) or handle.get("job_name") != JOB_NAME:
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
        "qsub -N " + JOB_NAME + " -q normal -P " + PROJECT
        + " -l select=" + shlex.quote(RESOURCES["select"])
        + " -l walltime=12:00:00 -o " + shlex.quote(ARTIFACTS + "/" + JOB_NAME + ".pbs.out")
        + " -j oe " + shlex.quote(ROOT + "/run.pbs")
    )


def submit(transport) -> dict:
    _mkdirs(transport)
    handle, state = _resolve(transport)
    if handle:
        return {"existing": True, "handle": handle}
    if state.get("intent"):
        raise RuntimeError("An unresolved durable submission exists; qsub was not repeated")
    _require_uploaded(transport)
    if _named_jobs(transport, JOB_NAME):
        raise RuntimeError("Exact job name already exists without a matching durable intent")
    intent = {
        "protocol": PROTOCOL["protocol"], "job_name": JOB_NAME, "script": ROOT + "/run.pbs",
        "status": "intent_before_qsub", "created_utc": utc_now(), **RESOURCES,
    }
    state.update(intent=intent, handle=None)
    _save_state(transport, state)
    try:
        code, output = remote.command(transport, _qsub_command(), timeout=45)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        _save_state(transport, state)
        raise RuntimeError("qsub outcome is unknown; recover the exact job by name before any retry") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", return_code=code, response=output[-1000:], updated_utc=utc_now())
        _save_state(transport, state)
        raise RuntimeError("qsub outcome is unknown or rejected; do not submit again")
    handle = {**RESOURCES, "job_id": job_id, "job_name": JOB_NAME,
              "protocol": PROTOCOL["protocol"], "status": "confirmed", "confirmed_utc": utc_now()}
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    state.update(intent=intent, handle=handle)
    _save_state(transport, state)
    return {"handle": handle}


def _arm_summary_ok(summary: dict, arm: str, job_id: str) -> bool:
    beta = summary.get("beta")
    common = (
        summary.get("protocol") == PROTOCOL["protocol"] and summary.get("arm") == arm
        and summary.get("pbs_jobid") == job_id and summary.get("complete") is True
        and summary.get("status") == "complete" and summary.get("updates") == 8
        and summary.get("exposures") == 80 and summary.get("target_linears") == 614
        and summary.get("frozen_rotation") is True
        and _finite_leq(summary.get("initial_parity_max_abs"), 1e-6)
        and _finite_leq(summary.get("formal_parity_max_abs"), 1e-6)
        and summary.get("final_anchor") is True and summary.get("new_teacher_queries") == 0
        and isinstance(beta, (int, float)) and not isinstance(beta, bool) and math.isfinite(beta)
    )
    return common and (beta == 0 if arm == "az" else beta > 0)


def _finite_leq(value, maximum: float) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= maximum)


def complete_gate(value: dict) -> bool:
    """Require PBS success, the root summary, and both matching arm receipts."""
    try:
        handle, scheduler = value["handle"], value["scheduler"]
        artifact, results = value["artifact"], value["results"]
        job_id = handle["job_id"]
        if (
            not HANDLE.fullmatch(str(job_id)) or handle.get("job_name") != JOB_NAME
            or value.get("protocol_matches") is not True
            or scheduler.get("query_rc") != 0 or scheduler.get("job_state") not in ("F", "X")
            or scheduler.get("Exit_status") != "0" or scheduler.get("Job_Name") != JOB_NAME
            or scheduler.get("queue") != "g2"
            or scheduler.get("Resource_List.walltime") != "12:00:00"
            or handle.get("gpus") != 2 or handle.get("walltime") != "12:00:00"
            or artifact.get("exit_code") != "0" or artifact.get("PHASE_COMPLETE") is not True
        ):
            return False
        phase = artifact.get("RUN_PHASE")
        root = results.get("summary")
        actual = results.get("arms", {})
        if (
            not isinstance(phase, dict) or phase.get("protocol") != PROTOCOL["protocol"]
            or phase.get("pbs_jobid") != job_id or phase.get("phase") != "complete"
            or not isinstance(root, dict) or root.get("protocol") != PROTOCOL["protocol"]
            or root.get("pbs_jobid") != job_id or root.get("complete") is not True
            or root.get("status") != "complete" or root.get("new_teacher_queries") != 0
            or set(root.get("arms", {})) != set(ARMS) or set(actual) != set(ARMS)
        ):
            return False
        for arm in ARMS:
            row = actual.get(arm)
            summary = row.get("summary") if isinstance(row, dict) else None
            if not isinstance(summary, dict) or not _arm_summary_ok(summary, arm, job_id):
                return False
            if root["arms"].get(arm) != summary:
                return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def _read_evidence(sftp, job_id: str) -> dict:
    artifact = ARTIFACTS + "/" + job_id
    arms = {}
    for arm in ARMS:
        base = RESULTS + "/arms/" + arm
        arms[arm] = {
            "progress": _json(sftp, base + "/progress.json"),
            "training": _json(sftp, base + "/training.json"),
            "summary": _json(sftp, base + "/summary.json"),
            "setup_checks": _json(sftp, base + "/setup_checks.json"),
            "failure": _json(sftp, base + "/failure.json"),
        }
    try:
        sftp.stat(artifact + "/PHASE_COMPLETE")
        phase_complete = True
    except OSError as exc:
        if not _absent(exc):
            raise
        phase_complete = False
    return {
        "artifact": {
            "exit_code": _text(sftp, artifact + "/exit_code.txt", 64),
            "RUN_PHASE": _json(sftp, artifact + "/RUN_PHASE.json"),
            "PHASE_COMPLETE": phase_complete,
        },
        "results": {
            "summary": _json(sftp, RESULTS + "/summary.json"),
            "arms": arms,
            "reader_preparation": _json(sftp, RESULTS + "/reader_preparation/summary.json"),
        },
    }


def _compact(value: dict) -> dict:
    handle = value.get("handle") or {}
    scheduler = value.get("scheduler") or {}
    result = {
        "status": value.get("status"), "job_id": handle.get("job_id"),
        "job_state": scheduler.get("job_state"), "queue": scheduler.get("queue"),
        "exec_host": scheduler.get("exec_host"), "query_rc": scheduler.get("query_rc"),
        "completion_passed": value.get("completion_passed", False),
    }
    arms = (value.get("results") or {}).get("arms") or {}
    if arms:
        result["arms"] = {
            arm: {
                "phase": (row.get("progress") or {}).get("phase"),
                "updates": (row.get("progress") or {}).get("updates"),
                "exposures": (row.get("progress") or {}).get("exposures"),
                "failure": bool(row.get("failure")),
            }
            for arm, row in arms.items()
        }
    reader = arms.get("reader_kl", {})
    progress = reader.get("progress") or {}
    preparation = (value.get("results") or {}).get("reader_preparation") or {}
    if progress.get("phase", "").startswith("initial_") or preparation:
        result["reader_preparation"] = {
            "phase": progress.get("phase") if progress.get("phase", "").startswith("initial_") else "complete",
            "initial_cases": progress.get("initial_cases", 80 if preparation else None),
            "beta": preparation.get("beta"),
        }
    return result


def status(transport) -> dict:
    handle, state = _resolve(transport)
    result = {"checked_utc": utc_now(), "handle": handle, "scheduler": None,
              "artifact": None, "results": None, "protocol_matches": False}
    with remote.sftp_client(transport) as sftp:
        protocol = _json(sftp, ROOT + "/protocol.json")
        result["protocol_matches"] = protocol == load_protocol()
        if handle:
            result.update(_read_evidence(sftp, handle["job_id"]))
    if not handle:
        result["status"] = "not_submitted" if not state.get("intent") else "intent_unresolved"
        return result
    scheduler = _scheduler(transport, handle["job_id"])
    result["scheduler"] = scheduler
    result["completion_passed"] = complete_gate(result)
    if scheduler.get("query_rc") != 0:
        result["status"] = "query_inconclusive"
    elif scheduler.get("job_state") == "Q":
        result["status"] = "queued"
    elif scheduler.get("job_state") == "R":
        result["status"] = "running"
    elif scheduler.get("job_state") in ("F", "X") and result["completion_passed"]:
        result["status"] = "complete"
    elif scheduler.get("job_state") in ("F", "X"):
        result["status"] = "terminal_unverified"
    else:
        result["status"] = "unknown"
    return result


def logs(transport) -> dict:
    current = status(transport)
    handle = current.get("handle")
    if not handle:
        raise RuntimeError("No PBS job handle is available")
    path = ARTIFACTS + "/" + handle["job_id"] + "/job.log"
    code, output = remote.command(transport, "tail -c 8192 " + shlex.quote(path) + " 2>/dev/null", timeout=20)
    return {"job_id": handle["job_id"], "query_rc": code,
            "tail": output.encode("utf-8", errors="replace")[-8192:].decode("utf-8", errors="replace")}


def pull(transport) -> dict:
    value = status(transport)
    target = HERE / "pulled" / (value["handle"]["job_id"] if value.get("handle") else "not_submitted")
    target.mkdir(parents=True, exist_ok=True)
    files = {"status.json": _compact(value)}
    results = value.get("results") or {}
    if results.get("summary") is not None:
        files["summary.json"] = results["summary"]
    if results.get("reader_preparation") is not None:
        files["reader_preparation_summary.json"] = results["reader_preparation"]
    for arm, rows in (results.get("arms") or {}).items():
        for name, data in rows.items():
            if data is not None:
                files[f"arms/{arm}/{name}.json"] = data
    written = []
    for name, value_to_write in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            json.dump(value_to_write, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            temp = Path(stream.name)
        temp.replace(path)
        written.append(str(path))
    return {**_compact(value), "pulled_files": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "logs", "pull"))
    args = parser.parse_args()
    jump, transport = direct_connect()
    try:
        if args.action == "upload":
            result, code = {"uploaded": upload(transport)}, 0
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
