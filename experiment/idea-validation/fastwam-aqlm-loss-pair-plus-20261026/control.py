"""Durable, small-file PBS controller for the bound 4-GPU closed-loop AQLM pair.

The parent fixed-input pair is an immutable source, not an implicit handle.  A
local ``bind-source`` receipt is required before this controller can upload or
submit.  Scheduler writes are guarded by an exact-name query and a durable
intent; an uncertain qsub result is never retried automatically.
"""
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
_remote_spec = importlib.util.spec_from_file_location(
    "aqlm_plus_remote_helper",
    HERE.parent / "fastwam-vq-c80-per-channel-plus" / "control.py",
)
if _remote_spec is None or _remote_spec.loader is None:
    raise RuntimeError("Trusted host-verified remote helper is unavailable")
_remote_module = importlib.util.module_from_spec(_remote_spec)
_remote_spec.loader.exec_module(_remote_module)
remote = _remote_module.remote

LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
USER = "yguo017"
PROJECT = "personal-yguo017"
PROTOCOL_NAME = "fastwam-aqlm-loss-pair-plus-v1"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
ATTEMPT = re.compile(r"^[a-z][a-z0-9]{0,5}$")
JOB_PREFIX = "aqlmplus261026"
JOB_RE = re.compile(r"^aqlmplus261026[a-z][a-z0-9]{0,5}$")
ARMS = ("reconstruction", "ours")
RESOURCES = {
    "queue": "normal", "execution_queue": "g3", "select": "1:ncpus=32:mem=220gb:ngpus=4",
    "gpus": 4, "ncpus": 32, "memory_gb": 220, "walltime": "16:00:00",
}
FILES = ("protocol.json", "control.py", "run.pbs", "environment.sh",
         "runtime.py", "evaluate.py", "aggregate.py", "README.zh.md")
WORKERS = tuple(range(4))


def _load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds 256 KiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("protocol.json does not match the closed-loop AQLM-plus protocol")
    allocation = value.get("allocation") or {}
    expected = {"jobs": 1, "queue": "normal", "execution_queue": "g3", "gpus": 4,
                "ncpus": 32, "memory_gb": 220, "walltime": "16:00:00",
                "episodes_per_job": 560, "gpu_monitor_seconds": 15}
    if any(allocation.get(key) != expected[key] for key in expected):
        raise RuntimeError("protocol allocation is not the frozen 4-GPU 16-hour job")
    if value.get("arms") != list(ARMS) or value.get("boundary", {}).get("closed_loop_authorized") is not True:
        raise RuntimeError("closed-loop authorization or arm order differs")
    assignment = value.get("worker_assignment") or {}
    if assignment.get("workers") != 4 or assignment.get("variants_per_worker") != 140:
        raise RuntimeError("worker assignment is not four 140-episode shards")
    if assignment.get("start_stagger_seconds") != 20:
        raise RuntimeError("worker start staggering is not frozen at 20 seconds")
    return value


PROTOCOL = _load_protocol()
ROOT = PROTOCOL["remote_root"]
CONTROL = ROOT + "/control"
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"


def direct_connect():
    return remote.direct_connect()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absent(exc: OSError) -> bool:
    return isinstance(exc, FileNotFoundError) or getattr(exc, "errno", None) == 2


def _small_bytes(sftp, path: str) -> bytes | None:
    # Shared-filesystem control reads can exceed the SSH helper's 30s default.
    sftp.get_channel().settimeout(90)
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
        raise RuntimeError(f"Remote control file exceeds 256 KiB: {path}")
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


def _text(sftp, path: str, limit: int = 4096) -> str | None:
    data = _small_bytes(sftp, path)
    if data is None:
        return None
    if len(data) > limit:
        raise RuntimeError(f"Remote text receipt is too large: {path}")
    return data.decode("utf-8", errors="replace").strip()


def _write_bytes(sftp, path: str, data: bytes) -> None:
    sftp.get_channel().settimeout(90)
    if len(data) > LIMIT:
        raise RuntimeError(f"Control file exceeds 256 KiB: {path}")
    with sftp.file(path + ".tmp", "wb") as stream:
        stream.write(data)
    sftp.posix_rename(path + ".tmp", path)


def _write_json(sftp, path: str, value: dict) -> None:
    _write_bytes(sftp, path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def _local_bytes(name: str) -> bytes:
    path = HERE / name
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise RuntimeError(f"Required small source is missing or too large: {name}")
    data = path.read_bytes().replace(b"\r\n", b"\n")
    data.decode("utf-8")
    return data


def _mkdirs(transport) -> None:
    command = "mkdir -p " + " ".join(shlex.quote(path) for path in (CONTROL, ARTIFACTS, RESULTS))
    code, output = remote.command(transport, command, timeout=90)
    if code:
        raise RuntimeError("Cannot prepare PBS control directories: " + output[-1000:])


def _stage_paths() -> tuple[str, str]:
    return CONTROL + "/pair_intent.json", CONTROL + "/pair_state.json"


def _state(sftp) -> dict:
    intent_path, state_path = _stage_paths()
    record = _json(sftp, state_path)
    if record is None:
        intent = _json(sftp, intent_path)
        record = {"schema_version": 1, "protocol": PROTOCOL_NAME, "stage": "pair",
                  "attempts": [intent] if intent else []}
    if record.get("schema_version") != 1 or record.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("Unsupported or cross-protocol pair state")
    if not isinstance(record.get("attempts"), list):
        raise RuntimeError("Invalid pair attempt ledger")
    return record


def _latest(record: dict) -> dict | None:
    rows = record.get("attempts") or []
    return rows[-1] if rows else None


def _save_state(sftp, record: dict) -> None:
    intent_path, state_path = _stage_paths()
    _write_json(sftp, state_path, record)
    latest = _latest(record)
    if latest:
        _write_json(sftp, intent_path, latest)
        if latest.get("job_id"):
            _write_json(sftp, CONTROL + "/pair_handle.json", latest)


def _parse_qstat(output: str) -> dict:
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "queue", "exec_host",
                "resources_used.walltime", "Resource_List.walltime", "Resource_List.ngpus",
                "Resource_List.ncpus", "Resource_List.mem", "Resource_List.select", "depend"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return fields


def _scheduler(transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(str(job_id)):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=90)
    return {"job_id": job_id, "query_rc": code, **_parse_qstat(output),
            "query_output": output[-1000:] if code else None}


def _named_jobs(transport, name: str) -> list[str]:
    if not JOB_RE.fullmatch(name):
        raise ValueError("Invalid exact PBS job name")
    code, output = remote.command(
        transport, "qselect -x -u " + USER + " -N " + shlex.quote(name) + " 2>&1", timeout=90)
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS exact-name history query failed; submission was not attempted")
    jobs = output.split()
    if any(not HANDLE.fullmatch(job) for job in jobs) or len(set(jobs)) != len(jobs):
        raise RuntimeError("PBS exact-name history returned an invalid or duplicate handle")
    return jobs


def _artifact(sftp, job_id: str) -> dict:
    base = ARTIFACTS + "/" + job_id
    failure = _json(sftp, base + "/failure.json") or _json(sftp, RESULTS + "/failure.json")
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "progress": _json(sftp, RESULTS + f"/arms/{arm}/progress.json"),
            "summary": _json(sftp, RESULTS + f"/arms/{arm}/summary.json"),
            "failure": _json(sftp, RESULTS + f"/arms/{arm}/failure.json"),
        }
    return {
        "exit_code": _text(sftp, base + "/exit_code.txt", 64),
        "failure_exit_status": _text(sftp, base + "/failure.exit_status", 64),
        "PHASE_COMPLETE": _exists(sftp, base + "/PHASE_COMPLETE"),
        "PIPELINE_COMPLETE": _exists(sftp, base + "/PIPELINE_COMPLETE"),
        "PIPELINE_CHECKPOINTED": _exists(sftp, base + "/PIPELINE_CHECKPOINTED"),
        "RUN_PHASE": _json(sftp, base + "/RUN_PHASE.json"),
        "failure": failure, "arms": arms,
    }


def _exists(sftp, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except OSError as exc:
        if _absent(exc):
            return False
        raise


def _read_results(sftp, job_id: str | None = None) -> dict:
    workers = {}
    for worker in WORKERS:
        workers[str(worker)] = {
            "progress": _json(sftp, RESULTS + f"/workers/worker_{worker}/progress.json"),
            "summary": _json(sftp, RESULTS + f"/workers/worker_{worker}/summary.json"),
            "failure": _json(sftp, RESULTS + f"/workers/worker_{worker}/failure.json"),
            "rc": _text(sftp, ARTIFACTS + f"/{job_id}/worker_{worker}.rc", 64) if job_id else None,
        }
    return {"summary": _json(sftp, RESULTS + "/summary.json"),
            "progress": _json(sftp, RESULTS + "/progress.json"),
            "PIPELINE_COMPLETE": _json(sftp, RESULTS + "/PIPELINE_COMPLETE.json"),
            "workers": workers}


def _is_terminal_success(scheduler: dict) -> bool:
    return (scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("F", "X")
            and str(scheduler.get("Exit_status")) == "0")


def _summary_belongs(summary: dict | None, job_id: str) -> bool:
    return summary is None or summary.get("pbs_jobid") in (None, job_id)


def _is_checkpoint_resume(scheduler: dict, artifact: dict, summary: dict | None) -> bool:
    return (scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("F", "X")
            and str(scheduler.get("Exit_status")) == "75"
            and artifact.get("PIPELINE_CHECKPOINTED") is True
            and (summary is None or (summary.get("status") == "checkpointed"
                                     and summary.get("complete") is not True)))


def _is_explicit_failure_resume(scheduler: dict, artifact: dict, summary: dict | None) -> bool:
    failure = artifact.get("failure")
    return (scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("F", "X")
            and str(scheduler.get("Exit_status")) not in ("0", "75", "None")
            and isinstance(failure, dict) and failure.get("protocol") == PROTOCOL_NAME
            and failure.get("pbs_jobid") == scheduler.get("job_id")
            and any(failure.get(key) for key in ("error", "traceback", "reason")))


def _summary_complete(summary: dict | None) -> bool:
    return (isinstance(summary, dict) and summary.get("protocol") == PROTOCOL_NAME
            and summary.get("complete") is True and summary.get("status", "complete") == "complete")


def _arm_row(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    row = value.get("summary")
    return row if isinstance(row, dict) else value


def _number(row: dict, names: tuple[str, ...]) -> object:
    for name in names:
        if name in row:
            return row[name]
    for nested in ("training", "evaluation", "counts", "progress"):
        child = row.get(nested)
        if isinstance(child, dict):
            for name in names:
                if name in child:
                    return child[name]
    return None


def _worker_complete(worker: int, value: dict | None) -> bool:
    if not isinstance(value, dict):
        return False
    row = value.get("summary") if isinstance(value.get("summary"), dict) else value
    if not isinstance(row, dict):
        return False
    expected_arm = "reconstruction" if worker < 2 else "ours"
    count = _number(row, ("episodes", "count", "completed", "episode_count"))
    return (value.get("rc") == "0" and row.get("protocol") in (None, PROTOCOL_NAME)
            and row.get("worker_id", worker) == worker and row.get("arm", expected_arm) == expected_arm
            and row.get("complete", row.get("status") == "complete") is True
            and count == 140)


def _count(value: dict, names: tuple[str, ...], expected: int) -> bool:
    got = _number(value, names)
    return got == expected


def complete_gate(value: dict) -> bool:
    """Require the exact compact schema emitted by the CL aggregate.py."""
    try:
        handle, scheduler = value["handle"], value["scheduler"]
        artifact, results = value["artifact"], value["results"]
        summary = results.get("summary")
        if (not isinstance(handle, dict) or not HANDLE.fullmatch(str(handle.get("job_id", "")))
                or handle.get("job_name") != _stage_name(handle.get("attempt", "a"))
                or scheduler.get("query_rc") != 0 or scheduler.get("job_state") not in ("F", "X")
                or str(scheduler.get("Exit_status")) != "0"
                or scheduler.get("Job_Name") != handle.get("job_name")
                or scheduler.get("queue") != RESOURCES["execution_queue"]
                or artifact.get("exit_code") != "0"
                or artifact.get("PHASE_COMPLETE") is not True
                or artifact.get("PIPELINE_COMPLETE") is not True
                or artifact.get("PIPELINE_CHECKPOINTED") is True
                or value.get("protocol_matches") is not True
                or not _summary_complete(summary)):
            return False
        if (summary.get("episodes") != 560
                or summary.get("episodes_per_arm") != {"reconstruction": 280, "ours": 280}
                or summary.get("native_low_bit_kernel") is not False
                or summary.get("new_training_updates") != 0
                or summary.get("new_teacher_queries") != 0
                or summary.get("reference_episode_slots") != 0):
            return False
        overall = summary.get("overall") or {}
        arms = overall.get("arms") or {}
        paired = overall.get("ours_vs_reconstruction") or {}
        workers = results.get("workers") or {}
        if (set(arms) != set(ARMS)
                or any((arms[arm] or {}).get("episodes") != 280 for arm in ARMS)
                or paired.get("episodes") != 280):
            return False
        return (set(workers) == {str(worker) for worker in WORKERS}
                and all(_worker_complete(worker, workers.get(str(worker))) for worker in WORKERS))
    except (KeyError, TypeError, AttributeError):
        return False


def _read_protocol_remote(sftp) -> dict | None:
    return _json(sftp, ROOT + "/protocol.json")


def _bound_source(protocol: dict | None = None) -> bool:
    value = protocol or PROTOCOL
    sources = value.get("sources") or {}
    job = sources.get("parent_pair_job")
    artifacts = sources.get("parent_pair_artifacts")
    return (isinstance(job, str) and HANDLE.fullmatch(job) is not None
            and isinstance(artifacts, str) and artifacts.endswith("/artifacts/" + job)
            and all((value.get("arm_sources") or {}).get(arm, {}).get("source_job") == job
                    for arm in ARMS)
            and all((value.get("arm_sources") or {}).get(arm, {}).get("source_protocol_file")
                    == artifacts + "/protocol.json" for arm in ARMS)
            and sources.get("parent_protocol_file") == artifacts + "/protocol.json")


def _parent_status(transport) -> dict:
    path = HERE.parent / "fastwam-aqlm-loss-pair-20261026" / "control.py"
    spec = importlib.util.spec_from_file_location("aqlm_parent_controller_for_binding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Parent AQLM controller is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.status(transport)


def _source_gate(parent: dict) -> tuple[str, str]:
    if parent.get("completion_passed") is not True:
        raise RuntimeError("Parent fixed-input pair has not passed its completion gate")
    pair = (parent.get("stages") or {}).get("pair") or {}
    scheduler = pair.get("scheduler") or {}
    artifact = pair.get("artifact") or {}
    if (scheduler.get("query_rc") != 0 or scheduler.get("job_state") not in ("F", "X")
            or str(scheduler.get("Exit_status")) != "0"
            or artifact.get("exit_code") != "0"
            or artifact.get("phase_complete") is not True):
        raise RuntimeError("Parent pair is not an identity-verified F/Exit0 PHASE_COMPLETE job")
    summary = pair.get("summary")
    if (not isinstance(summary, dict) or summary.get("complete") is not True
            or summary.get("status", "complete") != "complete"
            or summary.get("closed_loop_episodes", 0) != 0
            or summary.get("closed_loop") is True or summary.get("episodes", 0) != 0):
        raise RuntimeError("Parent summary is not complete fixed-input evidence")
    arms = pair.get("arms") or {}
    for arm in ARMS:
        if not _arm_complete_parent(arm, arms.get(arm)):
            raise RuntimeError(f"Parent arm {arm} lacks final8/exposures80/calibration80/heldout80/final_anchor")
    job_id = pair.get("job_id") or scheduler.get("job_id")
    if not HANDLE.fullmatch(str(job_id or "")):
        raise RuntimeError("Parent completion status did not expose an exact pair job handle")
    parent_root = PROTOCOL["sources"]["parent_root"].rstrip("/")
    return job_id, parent_root + "/artifacts/" + job_id


def _arm_complete_parent(arm: str, value: dict | None) -> bool:
    row = _arm_row(value)
    if not isinstance(row, dict):
        return False
    return (row.get("arm", arm) == arm and row.get("complete") is True
            and _number(row, ("updates", "final_update")) == 8
            and _number(row, ("exposures", "training_exposures")) == 80
            and _number(row, ("calibration", "calibration_queries")) == 80
            and _number(row, ("heldout", "heldout_queries")) == 80
            and row.get("final_anchor") is True)


def _bind_local_protocol(job_id: str, artifact_root: str) -> dict:
    global PROTOCOL
    updated = json.loads(json.dumps(PROTOCOL))
    sources = updated["sources"]
    source_protocol = artifact_root + "/protocol.json"
    sources["parent_pair_job"] = job_id
    sources["parent_pair_artifacts"] = artifact_root
    sources["parent_protocol_file"] = source_protocol
    for arm in ARMS:
        row = updated["arm_sources"][arm]
        row["source_job"] = job_id
        row["source_protocol_file"] = source_protocol
    data = (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode()
    if len(data) > LIMIT:
        raise RuntimeError("Bound protocol exceeds 256 KiB")
    target = HERE / "protocol.json"
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(target)
    PROTOCOL = updated
    return updated


def bind_source(transport) -> dict:
    """Bind only a verified, immutable parent pair before this job is attempted."""
    parent = _parent_status(transport)
    job_id, artifacts = _source_gate(parent)
    _mkdirs(transport)
    with remote.sftp_client(transport) as sftp:
        record = _state(sftp)
        if _latest(record):
            raise RuntimeError("Cannot bind source after this controller has attempted submission")
        _verify_quiescent(transport, record)
        remote_protocol = _read_protocol_remote(sftp)
        local_raw = _local_bytes("protocol.json")
        if remote_protocol is not None and _small_bytes(sftp, ROOT + "/protocol.json") != local_raw:
            raise RuntimeError("Remote protocol differs before source binding")
    updated = _bind_local_protocol(job_id, artifacts)
    return {"bound": True, "parent_pair_job": job_id, "parent_pair_artifacts": artifacts,
            "protocol": updated["protocol"]}


def _verify_quiescent(transport, record: dict) -> None:
    for attempt in record.get("attempts") or []:
        job_id = attempt.get("job_id")
        if not HANDLE.fullmatch(str(job_id or "")):
            if attempt.get("job_name") or attempt.get("status") in ("intent_before_qsub", "qsub_outcome_unknown"):
                raise RuntimeError("Unresolved pair intent blocks source transition")
            continue
        scheduler = _scheduler(transport, job_id)
        if scheduler.get("query_rc") != 0:
            raise RuntimeError("Scheduler query is inconclusive")
        if scheduler.get("job_state") in ("Q", "R", "E", "H"):
            raise RuntimeError(f"Active PBS job {job_id} blocks source transition")


def _stage_name(attempt: str) -> str:
    if not ATTEMPT.fullmatch(attempt):
        raise ValueError("Invalid attempt identifier")
    return JOB_PREFIX + attempt


def _qsub_command(attempt: str, dependency: str | None = None) -> str:
    parts = ["qsub", "-N", shlex.quote(_stage_name(attempt)), "-q", RESOURCES["queue"],
             "-P", PROJECT, "-l", "select=" + RESOURCES["select"],
             "-l", "walltime=" + RESOURCES["walltime"], "-v", "FW_AQLM_STAGE=pair",
             "-o", shlex.quote(ARTIFACTS + "/" + _stage_name(attempt) + ".pbs.out"), "-j", "oe"]
    if dependency:
        parts += ["-W", "depend=afterok:" + shlex.quote(dependency)]
    parts.append(shlex.quote(ROOT + "/run.pbs"))
    return " ".join(parts)


def _require_uploaded(transport) -> None:
    if not _bound_source():
        raise RuntimeError("Source binding is required before upload or submit")
    with remote.sftp_client(transport) as sftp:
        for name in FILES:
            if _small_bytes(sftp, ROOT + "/" + name) != _local_bytes(name):
                raise RuntimeError("Upload is incomplete or differs from the local frozen source: " + name)
        if _read_protocol_remote(sftp) != PROTOCOL:
            raise RuntimeError("Remote frozen protocol differs from the local protocol")


def _resume_candidate(transport, sftp, record: dict) -> tuple[dict, dict, dict | None]:
    latest = _latest(record)
    if not latest or not HANDLE.fullmatch(str(latest.get("job_id", ""))):
        raise RuntimeError("Resume requires an exact prior pair handle")
    scheduler = _scheduler(transport, latest["job_id"])
    artifact = _artifact(sftp, latest["job_id"])
    summary = (_read_results(sftp).get("summary"))
    if not _summary_belongs(summary, latest["job_id"]):
        raise RuntimeError("Resume receipt belongs to a different PBS job")
    if not (_is_checkpoint_resume(scheduler, artifact, summary)
            or _is_explicit_failure_resume(scheduler, artifact, summary)):
        raise RuntimeError("Resume requires an explicit checkpoint or failure receipt")
    return latest, scheduler, summary


def upload(transport, *, resume: bool = False) -> list[str]:
    if not _bound_source():
        raise RuntimeError("Source binding is required before upload")
    _mkdirs(transport)
    uploaded = []
    with remote.sftp_client(transport) as sftp:
        record = _state(sftp)
        latest = _latest(record)
        if latest:
            _verify_quiescent(transport, record)
            if not resume:
                raise RuntimeError("Submitted source is frozen; use --resume only for an explicit terminal receipt")
            _resume_candidate(transport, sftp, record)
        for name in FILES:
            data = _local_bytes(name)
            target = ROOT + "/" + name
            previous = _small_bytes(sftp, target)
            if previous == data:
                continue
            if latest and not resume:
                raise RuntimeError("Submitted source is frozen")
            _write_bytes(sftp, target, data)
            uploaded.append(name)
    return uploaded


def _resolve_latest(transport, record: dict) -> dict | None:
    latest = _latest(record)
    if not latest:
        return None
    if latest.get("status") in ("intent_before_qsub", "qsub_outcome_unknown") or not latest.get("job_id"):
        jobs = _named_jobs(transport, latest.get("job_name", ""))
        if len(jobs) != 1:
            raise RuntimeError("qsub outcome is unresolved; no second submission will be issued")
        scheduler = _scheduler(transport, jobs[0])
        if scheduler.get("query_rc") != 0 or scheduler.get("Job_Name") != latest.get("job_name"):
            raise RuntimeError("Exact-name PBS handle could not be verified")
        latest.update(job_id=jobs[0], status="recovered", recovered_utc=utc_now())
        with remote.sftp_client(transport) as sftp:
            _save_state(sftp, record)
    elif not HANDLE.fullmatch(str(latest.get("job_id"))):
        raise RuntimeError("Saved PBS handle is invalid")
    return latest


def submit(transport, attempt: str = "a", *, resume: bool = False) -> dict:
    if not ATTEMPT.fullmatch(attempt):
        raise ValueError("Invalid attempt identifier")
    if not _bound_source():
        raise RuntimeError("Source binding is required before submit")
    _mkdirs(transport)
    with remote.sftp_client(transport) as sftp:
        record = _state(sftp)
    latest = _resolve_latest(transport, record)
    dependency = None
    if latest:
        scheduler = _scheduler(transport, latest["job_id"])
        with remote.sftp_client(transport) as sftp:
            artifact = _artifact(sftp, latest["job_id"])
            summary = _read_results(sftp).get("summary")
        if scheduler.get("job_state") in ("Q", "R", "E", "H"):
            return {"existing": True, "handle": latest, "scheduler": scheduler}
        if (_is_terminal_success(scheduler) and artifact.get("PHASE_COMPLETE")
                and _summary_complete(summary) and summary.get("pbs_jobid") in (None, latest["job_id"])):
            return {"existing": True, "handle": latest, "scheduler": scheduler}
        resumable = (_summary_belongs(summary, latest["job_id"])
                     and (_is_checkpoint_resume(scheduler, artifact, summary)
                          or _is_explicit_failure_resume(scheduler, artifact, summary)))
        if not resume or not resumable:
            raise RuntimeError("Pair is not an exact explicit resume candidate")
        if attempt == latest.get("attempt"):
            raise RuntimeError("Resume requires a new attempt identifier")
    _require_uploaded(transport)
    name = _stage_name(attempt)
    if _named_jobs(transport, name):
        raise RuntimeError("Exact PBS job name already exists without matching durable intent")
    intent = {"schema_version": 1, "protocol": PROTOCOL_NAME, "stage": "pair", "attempt": attempt,
              "job_name": name, "script": ROOT + "/run.pbs", "status": "intent_before_qsub",
              "created_utc": utc_now(), **RESOURCES}
    record.setdefault("attempts", []).append(intent)
    with remote.sftp_client(transport) as sftp:
        _save_state(sftp, record)
    try:
        code, output = remote.command(transport, _qsub_command(attempt, dependency), timeout=90)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        with remote.sftp_client(transport) as sftp:
            _save_state(sftp, record)
        raise RuntimeError("qsub outcome is unknown; recover exact name before retrying") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", return_code=code, response=output[-1000:], updated_utc=utc_now())
        with remote.sftp_client(transport) as sftp:
            _save_state(sftp, record)
        raise RuntimeError("qsub outcome is unknown or rejected; do not submit again")
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    record["attempts"][-1] = intent
    with remote.sftp_client(transport) as sftp:
        _save_state(sftp, record)
    return {"existing": False, "handle": intent}


def _stage_status(transport) -> dict:
    with remote.sftp_client(transport) as sftp:
        record = _state(sftp)
        latest = _latest(record)
        result = {"stage": "pair", "attempt": latest.get("attempt") if latest else None,
                  "job_id": latest.get("job_id") if latest else None, "scheduler": None,
                  "artifact": None, "results": _read_results(sftp, latest.get("job_id") if latest else None),
                  "attempts": len(record.get("attempts") or [])}
        if latest and latest.get("job_id"):
            result["scheduler"] = _scheduler(transport, latest["job_id"])
            result["artifact"] = _artifact(sftp, latest["job_id"])
    return result


def status(transport) -> dict:
    row = _stage_status(transport)
    with remote.sftp_client(transport) as sftp:
        remote_protocol = _read_protocol_remote(sftp)
    result = {"checked_utc": utc_now(), "protocol_matches": remote_protocol == PROTOCOL,
              "source_bound": _bound_source(), "stages": {"pair": row},
              "handle": None, "scheduler": row.get("scheduler"),
              "artifact": row.get("artifact"), "results": row.get("results")}
    if row.get("job_id"):
        latest = {"attempt": row.get("attempt"), "job_id": row.get("job_id"),
                  "job_name": _stage_name(row.get("attempt") or "a")}
        result["handle"] = latest
    result["completion_passed"] = complete_gate(result)
    return result


def _compact(value: dict) -> dict:
    row = (value.get("stages") or {}).get("pair") or {}
    scheduler = row.get("scheduler") or {}
    artifact = row.get("artifact") or {}
    results = row.get("results") or {}
    summary = results.get("summary") or {}
    arm_counts = summary.get("episodes_per_arm") if isinstance(summary, dict) else {}
    arms = {arm: {"episodes": (arm_counts or {}).get(arm),
                  "overall_episodes": (((summary.get("overall") or {}).get("arms") or {}).get(arm) or {}).get("episodes")}
            for arm in ARMS}
    workers = {}
    for worker in WORKERS:
        data = (results.get("workers") or {}).get(str(worker)) or {}
        progress = data.get("progress") or {}
        receipt = data.get("summary") or progress
        workers[str(worker)] = {"phase": progress.get("phase"),
                                "episodes": _number(receipt, ("episodes", "count", "completed")),
                                "rc": data.get("rc"), "failure": bool(data.get("failure"))}
    return {"checked_utc": value.get("checked_utc"), "source_bound": value.get("source_bound"),
            "protocol_matches": value.get("protocol_matches"), "completion_passed": value.get("completion_passed", False),
            "pair": {"attempt": row.get("attempt"), "job_id": row.get("job_id"),
                     "job_state": scheduler.get("job_state"), "exit_status": scheduler.get("Exit_status"),
                     "queue": scheduler.get("queue"), "exec_host": scheduler.get("exec_host"),
                     "query_rc": scheduler.get("query_rc"), "phase_complete": artifact.get("PHASE_COMPLETE", False),
                     "checkpointed": artifact.get("PIPELINE_CHECKPOINTED", False),
                     "summary_complete": _summary_complete(summary), "summary": summary,
                     "arms": arms, "workers": workers}}


def logs(transport) -> dict:
    row = _stage_status(transport)
    job_id = row.get("job_id")
    if not job_id:
        raise RuntimeError("No PBS pair handle is available")
    code, output = remote.command(transport, "tail -c 8192 " + shlex.quote(ARTIFACTS + "/" + job_id + "/job.log") + " 2>/dev/null", timeout=90)
    return {"job_id": job_id, "query_rc": code,
            "tail": output.encode("utf-8", errors="replace")[-8192:].decode("utf-8", errors="replace")}


def pull(transport) -> dict:
    value = status(transport)
    target = HERE / "pulled" / ((value.get("handle") or {}).get("job_id") or "not_submitted")
    target.mkdir(parents=True, exist_ok=True)
    path = target / "status.json"
    data = (json.dumps(_compact(value), indent=2, ensure_ascii=False) + "\n").encode()
    if len(data) > LIMIT:
        raise RuntimeError("Refusing to pull non-small compact status")
    with tempfile.NamedTemporaryFile("wb", dir=target, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(path)
    return {"pulled_file": str(path), **_compact(value)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bind-source", "upload", "submit", "status", "verify", "logs", "pull"))
    parser.add_argument("--attempt", default="a")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not ATTEMPT.fullmatch(args.attempt):
        parser.error("Use a short lowercase attempt")
    jump, transport = direct_connect()
    try:
        if args.action == "bind-source":
            result = bind_source(transport)
        elif args.action == "upload":
            result = {"uploaded": upload(transport, resume=args.resume)}
        elif args.action == "submit":
            result = submit(transport, args.attempt, resume=args.resume)
        elif args.action == "status":
            result = _compact(status(transport))
        elif args.action == "verify":
            value = status(transport)
            result = {**_compact(value), "verified_complete": value["completion_passed"]}
        elif args.action == "logs":
            result = logs(transport)
        else:
            result = pull(transport)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if args.action not in ("verify",) or result.get("verified_complete") else 1
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
