"""Small-file PBS controller for the frozen C80 per-channel evaluation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "c80_clip_joint_control", HERE.parent / "fastwam-vq-c80-clip-joint-plus" / "control.py"
)
_c80 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_c80)
remote = _c80.remote

PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = PROTOCOL["remote_root"]
CONTROL = ROOT + "/control"
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"
ARM = "per_channel_clip"
UNIT = "evaluation"
JOB_PREFIX = "c80pceval"
JOB_NAME = JOB_PREFIX + "0001"
USER = "yguo017"
LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
JOB_RE = re.compile(r"^c80pceval[0-9]{4}$")
FILES = (
    "protocol.json", "source_protocol.json", "run.pbs", "vq_adapter.py", "evaluate.py",
    "aggregate.py", "environment.sh", "batched_transform.py", "transform_checks.py",
)
RESOURCES = {
    "queue": "normal", "select": "1:ncpus=32:mem=220gb:ngpus=4",
    "gpus": 4, "ncpus": 32, "memory_gb": 220, "walltime": "24:00:00",
}


def direct_connect():
    return remote.direct_connect()


def _absent(exc: OSError) -> bool:
    return isinstance(exc, FileNotFoundError) or getattr(exc, "errno", None) == 2


def _bytes(sftp, path: str) -> bytes | None:
    try:
        with sftp.file(path, "rb") as stream:
            data = stream.read(LIMIT + 1)
    except OSError as exc:
        if _absent(exc):
            return None
        raise
    if len(data) > LIMIT:
        raise RuntimeError(f"Remote control file exceeds 256 KiB: {path}")
    return data


def _json(sftp, path: str):
    data = _bytes(sftp, path)
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid remote JSON: {path}") from exc


def _text(sftp, path: str) -> str | None:
    data = _bytes(sftp, path)
    return data.decode("utf-8", errors="replace").strip() if data is not None else None


def _write_json(sftp, path: str, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(data) > LIMIT:
        raise ValueError("Remote control record exceeds 256 KiB")
    temp = path + ".tmp"
    with sftp.file(temp, "wb") as stream:
        stream.write(data)
    sftp.posix_rename(temp, path)


def _exists(sftp, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except OSError as exc:
        if _absent(exc):
            return False
        raise


def _mkdirs(transport) -> None:
    paths = (CONTROL, ARTIFACTS, RESULTS, RESULTS + "/evaluations/" + ARM)
    code, output = remote.command(
        transport, "mkdir -p " + " ".join(shlex.quote(path) for path in paths), timeout=25
    )
    if code:
        raise RuntimeError("Cannot prepare PBS control directories: " + output[:1000])


def _intent_path() -> str:
    return CONTROL + "/evaluation_intent.json"


def _records(sftp) -> dict:
    value = _json(sftp, CONTROL + "/handles.json") or {"schema_version": 1, "attempts": []}
    if not isinstance(value, dict) or not isinstance(value.get("attempts", []), list):
        raise RuntimeError("Invalid remote handles.json")
    return value


def _latest(records: dict) -> dict | None:
    attempts = records.get("attempts", [])
    valid = [row for row in attempts if isinstance(row, dict) and type(row.get("attempt")) is int]
    return max(valid, key=lambda row: row["attempt"], default=None)


def _save_handle(sftp, records: dict, handle: dict) -> None:
    rows = records.setdefault("attempts", [])
    old = [row for row in rows if row.get("attempt") == handle["attempt"]]
    if old and old[0].get("job_id") not in (None, handle.get("job_id")):
        raise RuntimeError("PBS handle conflicts with durable history")
    rows[:] = [row for row in rows if row.get("attempt") != handle["attempt"]]
    rows.append(handle)
    rows.sort(key=lambda row: int(row["attempt"]))
    records["schema_version"] = 1
    _write_json(sftp, CONTROL + "/handles.json", records)


def _local_bytes(name: str) -> bytes:
    path = HERE / name
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if len(data) > LIMIT:
        raise ValueError(f"Source exceeds 256 KiB: {name}")
    data.decode("utf-8")
    return data


def _has_intent(sftp, records: dict) -> bool:
    return bool(records.get("attempts") or _json(sftp, _intent_path()))


def upload(transport) -> list[str]:
    _mkdirs(transport)
    with remote.sftp_client(transport) as sftp:
        old_protocol = _json(sftp, ROOT + "/protocol.json")
        if old_protocol is not None and old_protocol != PROTOCOL:
            raise RuntimeError("Remote frozen protocol differs; refusing overwrite")
        records = _records(sftp)
        frozen = _has_intent(sftp, records)
        uploaded = []
        for name in FILES:
            data = _local_bytes(name)
            target = ROOT + "/" + name
            previous = _bytes(sftp, target)
            if name == "protocol.json" and previous is not None:
                if previous != data:
                    raise RuntimeError("Frozen remote protocol bytes differ; refusing overwrite")
                continue
            if previous == data:
                continue
            if frozen:
                raise RuntimeError("Submitted source is frozen; remote differs from " + name)
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            uploaded.append(name)
    return uploaded


def _require_uploaded(transport) -> None:
    with remote.sftp_client(transport) as sftp:
        for name in FILES:
            expected = _local_bytes(name)
            if _bytes(sftp, ROOT + "/" + name) != expected:
                raise RuntimeError("Upload is incomplete or changed: " + name)
        if _json(sftp, ROOT + "/protocol.json") != PROTOCOL:
            raise RuntimeError("Remote protocol differs from the local frozen protocol")


def _query_name(transport, name: str) -> list[str]:
    if not JOB_RE.fullmatch(name):
        raise ValueError("Invalid exact PBS job name")
    code, output = remote.command(
        transport, "qselect -x -u " + USER + " -N " + shlex.quote(name) + " 2>&1", timeout=35
    )
    if code:
        raise RuntimeError("PBS exact-name history query failed; submission was not attempted")
    jobs = output.split()
    if any(not HANDLE.fullmatch(job) for job in jobs):
        raise RuntimeError("PBS returned an invalid exact-name handle")
    return jobs


def _qstat(transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=35)
    result = {"job_id": job_id, "query_rc": code}
    if code:
        result["query_output"] = output[-1000:]
        return result
    for key in (
        "job_state", "Exit_status", "Job_Name", "queue", "exec_host", "comment",
        "resources_used.walltime", "Resource_List.walltime", "Resource_List.ngpus",
        "Resource_List.ncpus", "Resource_List.mem", "Resource_List.select",
    ):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        result[key] = match.group(1).strip() if match else None
    return result


def _paths(job_id: str) -> tuple[str, str]:
    return ARTIFACTS + "/" + job_id, RESULTS + "/evaluations/" + ARM


def _read_evidence(sftp, job_id: str) -> dict:
    artifact, evaluation = _paths(job_id)
    return {
        "artifact": {
            "exit_code": _text(sftp, artifact + "/exit_code.txt"),
            "RUN_PHASE": _json(sftp, artifact + "/RUN_PHASE.json"),
            "PIPELINE_COMPLETE": _exists(sftp, artifact + "/PIPELINE_COMPLETE"),
            "PIPELINE_CHECKPOINTED": _exists(sftp, artifact + "/PIPELINE_CHECKPOINTED"),
        },
        "evaluation": {
            "summary": _json(sftp, evaluation + "/summary.json"),
            "PIPELINE_COMPLETE": _json(sftp, evaluation + "/PIPELINE_COMPLETE"),
            "PIPELINE_CHECKPOINTED": _json(sftp, evaluation + "/PIPELINE_CHECKPOINTED"),
        },
        "workers": {
            str(worker): {
                "progress": _json(sftp, evaluation + f"/worker_{worker}/progress.json"),
                "summary": _json(sftp, evaluation + f"/worker_{worker}/summary.json"),
            }
            for worker in range(4)
        },
    }


def _recover(transport, intent: dict, records: dict) -> dict | None:
    if (intent.get("unit") != UNIT or type(intent.get("attempt")) is not int
            or not JOB_RE.fullmatch(str(intent.get("job_name", "")))):
        raise RuntimeError("Durable PBS intent is malformed")
    jobs = _query_name(transport, intent["job_name"])
    if len(jobs) > 1:
        raise RuntimeError("Multiple PBS jobs share this exact name; reconcile before proceeding")
    if not jobs:
        return None
    job_id = jobs[0]
    observed = _qstat(transport, job_id)
    if observed.get("query_rc") != 0 or observed.get("Job_Name") != intent["job_name"]:
        raise RuntimeError("Exact-name PBS handle could not be verified")
    handle = dict(intent, job_id=job_id, status="recovered")
    with remote.sftp_client(transport) as sftp:
        _write_json(sftp, _intent_path(), handle)
        _save_handle(sftp, records, handle)
    return handle


def _resolve(transport) -> tuple[dict | None, dict | None]:
    with remote.sftp_client(transport) as sftp:
        records = _records(sftp)
        intent = _json(sftp, _intent_path())
    if intent:
        if intent.get("job_id"):
            handle = intent
            if not HANDLE.fullmatch(str(handle["job_id"])):
                raise RuntimeError("Durable PBS intent has an invalid job handle")
            with remote.sftp_client(transport) as sftp:
                _save_handle(sftp, records, handle)
            return handle, intent
        recovered = _recover(transport, intent, records)
        return recovered, intent
    return _latest(records), None


def _intent(attempt: int, source_job_id: str | None = None) -> dict:
    if not 1 <= attempt <= 9999:
        raise ValueError("Evaluation attempt is outside the PBS job-name range")
    name = f"{JOB_PREFIX}{attempt:04d}"
    value = {
        "schema_version": 1, "unit": UNIT, "arm": ARM, "attempt": attempt,
        "job_name": name, "script": ROOT + "/run.pbs", "remote_root": ROOT,
        "created_utc": datetime.now(timezone.utc).isoformat(), **RESOURCES,
        "status": "intent_before_qsub",
    }
    if source_job_id:
        value.update(resume=True, source_job_id=source_job_id)
    return value


def _submit_attempt(transport, attempt: int, source_job_id: str | None = None) -> dict:
    _require_uploaded(transport)
    intent = _intent(attempt, source_job_id)
    jobs = _query_name(transport, intent["job_name"])
    if len(jobs) > 1:
        raise RuntimeError("Multiple exact-name PBS jobs exist; qsub was not attempted")
    if jobs:
        raise RuntimeError("Exact PBS job name already exists without a matching durable intent")
    with remote.sftp_client(transport) as sftp:
        current = _json(sftp, _intent_path())
        if current and (current.get("status") not in ("confirmed", "recovered")
                        or current.get("attempt", 0) >= attempt):
            raise RuntimeError("An unresolved durable PBS intent exists; qsub was not repeated")
        _write_json(sftp, _intent_path(), intent)

    command = (
        "qsub -N " + shlex.quote(intent["job_name"])
        + " -q normal -l select=" + shlex.quote(RESOURCES["select"])
        + " -l walltime=24:00:00 -o "
        + shlex.quote(ARTIFACTS + "/" + intent["job_name"] + ".pbs.out")
    )
    if source_job_id:
        command += " -v " + shlex.quote("FW_EVAL_RESUME=1,FW_EVAL_SOURCE_JOB_ID=" + source_job_id)
    command += " " + shlex.quote(intent["script"])
    code, output = remote.command(transport, command, timeout=45)
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="unknown_or_rejected", return_code=code, response=output[-1000:])
        with remote.sftp_client(transport) as sftp:
            _write_json(sftp, _intent_path(), intent)
        recovered = _recover(transport, intent, _records_after(transport))
        if recovered:
            return recovered
        raise RuntimeError("qsub outcome is unknown; exact-name recovery found no job, so no retry was made")
    intent.update(job_id=job_id, status="confirmed")
    with remote.sftp_client(transport) as sftp:
        _write_json(sftp, _intent_path(), intent)
        _save_handle(sftp, _records(sftp), intent)
    return intent


def _records_after(transport) -> dict:
    with remote.sftp_client(transport) as sftp:
        return _records(sftp)


def submit(transport) -> dict:
    _mkdirs(transport)
    handle, intent = _resolve(transport)
    if handle:
        return handle
    if intent:
        raise RuntimeError("A durable attempt is unresolved; status may recover it by exact job name")
    return _submit_attempt(transport, 1)


def _worker_receipt_ok(row: dict, worker: int, *, complete: bool) -> bool:
    progress = row.get("progress")
    receipt = row.get("summary")
    indices = list(range(worker, 280, 4))
    if not isinstance(progress, dict) or not isinstance(receipt, dict) or progress != receipt:
        return False
    count = receipt.get("episodes")
    common = (
        receipt.get("protocol") == PROTOCOL["protocol"] and receipt.get("arm") == ARM
        and receipt.get("worker_id") == worker and receipt.get("expected") == 70
        and receipt.get("indices") == indices and type(count) is int
        and receipt.get("completed_indices") == indices[:count]
    )
    if complete:
        return common and count == 70 and receipt.get("status") == "complete" and receipt.get("complete") is True
    return (common and 0 <= count <= 70 and receipt.get("status") in ("complete", "checkpointed")
            and ((receipt.get("status") == "complete" and count == 70 and receipt.get("complete") is True)
                 or (receipt.get("status") == "checkpointed" and receipt.get("complete") is False
                     and receipt.get("soft_deadline_reached") is True)))


def complete_gate(value: dict) -> bool:
    """True only when PBS, artifact, result markers, 280 episodes and 2,800 refs agree."""
    try:
        handle, scheduler = value["handle"], value["scheduler"]
        artifact = value["artifact"]
        evaluation = value["evaluation"]
        summary, marker = evaluation["summary"], evaluation["PIPELINE_COMPLETE"]
        phase = artifact["RUN_PHASE"]
        if (not isinstance(handle, dict) or not HANDLE.fullmatch(str(handle.get("job_id", "")))
                or value.get("protocol_matches") is not True
                or scheduler.get("query_rc") != 0 or scheduler.get("job_state") not in ("F", "X")
                or scheduler.get("Exit_status") != "0" or scheduler.get("Job_Name") != handle.get("job_name")
                or handle.get("arm") != ARM or handle.get("gpus") != 4 or handle.get("ncpus") != 32
                or handle.get("memory_gb") != 220 or handle.get("walltime") != "24:00:00"
                or artifact.get("exit_code") != "0" or artifact.get("PIPELINE_COMPLETE") is not True
                or artifact.get("PIPELINE_CHECKPOINTED") is not False
                or not isinstance(phase, dict) or phase.get("protocol") != PROTOCOL["protocol"]
                or phase.get("arm") != ARM or phase.get("phase") != "complete"):
            return False
        if (not isinstance(summary, dict) or summary.get("protocol") != PROTOCOL["protocol"]
                or summary.get("arm") != ARM or summary.get("status") != "complete"
                or summary.get("complete") is not True or summary.get("episode_slots") != 280
                or summary.get("arm_episodes") != {ARM: 280}
                or summary.get("reference_episode_slots") != 2800):
            return False
        if (not isinstance(marker, dict) or marker.get("protocol") != PROTOCOL["protocol"]
                or marker.get("arm") != ARM or marker.get("complete") is not True
                or marker.get("episode_slots") != 280 or marker.get("reference_episode_slots") != 2800):
            return False
        return all(_worker_receipt_ok(value.get("workers", {}).get(str(i), {}), i, complete=True)
                   for i in range(4))
    except (KeyError, TypeError, AttributeError):
        return False


def _checkpoint_gate(value: dict) -> bool:
    try:
        handle, scheduler = value["handle"], value["scheduler"]
        artifact, evaluation = value["artifact"], value["evaluation"]
        summary, marker = evaluation["summary"], evaluation["PIPELINE_CHECKPOINTED"]
        phase = artifact["RUN_PHASE"]
        if (not isinstance(handle, dict) or scheduler.get("query_rc") != 0
                or value.get("protocol_matches") is not True
                or scheduler.get("job_state") not in ("F", "X") or scheduler.get("Exit_status") != "0"
                or scheduler.get("Job_Name") != handle.get("job_name") or handle.get("arm") != ARM
                or artifact.get("exit_code") != "0" or artifact.get("PIPELINE_CHECKPOINTED") is not True
                or artifact.get("PIPELINE_COMPLETE") is not False
                or not isinstance(phase, dict) or phase.get("protocol") != PROTOCOL["protocol"]
                or phase.get("arm") != ARM or phase.get("phase") != "checkpointed"):
            return False
        if (not isinstance(summary, dict) or summary.get("protocol") != PROTOCOL["protocol"]
                or summary.get("arm") != ARM or summary.get("status") != "checkpointed"
                or summary.get("complete") is not False or type(summary.get("episode_slots")) is not int
                or not 0 <= summary["episode_slots"] < 280 or summary.get("reference_episode_slots") != 2800
                or summary.get("arm_episodes") != {ARM: summary["episode_slots"]}):
            return False
        if (not isinstance(marker, dict) or marker.get("protocol") != PROTOCOL["protocol"]
                or marker.get("arm") != ARM or marker.get("complete") is not False
                or marker.get("episode_slots") != summary["episode_slots"]
                or marker.get("reference_episode_slots") != 2800
                or evaluation.get("PIPELINE_COMPLETE") is not None):
            return False
        workers = value.get("workers", {})
        if not all(_worker_receipt_ok(workers.get(str(i), {}), i, complete=False) for i in range(4)):
            return False
        return sum(workers[str(i)]["summary"]["episodes"] for i in range(4)) == summary["episode_slots"]
    except (KeyError, TypeError, AttributeError):
        return False


def resume(transport) -> dict:
    current = status(transport)
    if not _checkpoint_gate(current):
        raise RuntimeError("Resume requires a verified exit-0 checkpoint with resumable worker prefixes")
    attempt = current["handle"]["attempt"] + 1
    return _submit_attempt(transport, attempt, current["handle"]["job_id"])


def status(transport) -> dict:
    _mkdirs(transport)
    handle, intent = _resolve(transport)
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "handle": handle,
        "scheduler": None, "artifact": None, "evaluation": None, "workers": {},
        "protocol": None, "protocol_matches": False,
    }
    if intent and not handle:
        result["intent"] = {key: intent.get(key) for key in ("attempt", "job_name", "status", "created_utc")}
        return result
    with remote.sftp_client(transport) as sftp:
        result["protocol"] = _json(sftp, ROOT + "/protocol.json")
        result["protocol_matches"] = result["protocol"] == PROTOCOL
        if handle:
            result.update(_read_evidence(sftp, handle["job_id"]))
    if not handle:
        result["status"] = "not_submitted"
        return result
    result["scheduler"] = _qstat(transport, handle["job_id"])
    result["completion_passed"] = complete_gate(result)
    result["checkpoint_passed"] = _checkpoint_gate(result)
    return result


def logs(transport) -> dict:
    current = status(transport)
    handle = current.get("handle")
    if not handle:
        raise RuntimeError("No PBS job handle is available")
    path = ARTIFACTS + "/" + handle["job_id"] + "/job.log"
    code, output = remote.command(transport, "tail -c 8192 " + shlex.quote(path) + " 2>/dev/null", timeout=20)
    return {"job_id": handle["job_id"], "query_rc": code, "tail": output.encode("utf-8", errors="replace")[-8192:].decode("utf-8", errors="replace")}


def pull(transport) -> dict:
    value = status(transport)
    value["scope"] = "bounded PBS and JSON status only; model checkpoints and episode payloads are excluded"
    target = HERE / "pulled"
    target.mkdir(exist_ok=True)
    name = value["handle"]["job_id"] if value.get("handle") else "not_submitted"
    path = target / f"evaluation_{name}.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
    value["pulled_file"] = str(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "resume", "status", "logs", "pull"))
    args = parser.parse_args()
    _jump, transport = direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": upload(transport)}
        elif args.action == "submit":
            result = submit(transport)
        elif args.action == "resume":
            result = resume(transport)
        elif args.action == "status":
            result = status(transport)
        elif args.action == "logs":
            result = logs(transport)
        else:
            result = pull(transport)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if transport:
            transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
