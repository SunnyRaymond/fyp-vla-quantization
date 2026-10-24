"""Small-file, host-key-verified PBS control for the frozen C80 R128 Q-Clip run."""
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
    "qclip_remote", HERE.parent / "fastwam-vq-c80-per-channel-plus" / "control.py"
)
_base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_base)
remote = _base.remote

LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
USER = "yguo017"
PROJECT = "personal-yguo017"
JOB_PREFIX = "r128qc"
JOB_NAMES = {"short": JOB_PREFIX + "0001", "long": JOB_PREFIX + "0002", "resume": JOB_PREFIX + "0003"}
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
JOB_NAME_RE = re.compile(r"^r128qc000[123]$")
FILES = ("protocol.json", "run.pbs", "environment.sh", "control.py", "test_control.py",
         "worker.py", "test_worker.py", "clip_bank.py", "README.zh.md")
METRICS = (
    "first10motor_rmse", "first10motor_mse", "future_latent_nmse", "mixed_az_loss",
    "full32motor_rmse", "late22motor_rmse", "first10gripper_rmse", "first10gripper_disagreement",
)


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != "fastwam-vq-r128-c80-qclip-v1":
        raise RuntimeError("protocol.json does not match the frozen R128 C80 Q-Clip protocol")
    if value.get("remote_root") != "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-qclip-20261023":
        raise RuntimeError("protocol.json names an unexpected remote root")
    if value.get("parent_runtime") != value.get("parent_root", "") + "/artifacts/25921795.pbs101":
        raise RuntimeError("protocol.json names an unexpected parent runtime")
    allocation = value.get("allocation") or {}
    expected_allocation = {
        "queue": "normal", "gpus": 1, "ncpus": 16, "memory_gb": 110,
        "short_walltime": "01:59:00", "long_walltime": "12:00:00",
        "gpu_monitor_seconds": 15,
    }
    if any(allocation.get(key) != item for key, item in expected_allocation.items()):
        raise RuntimeError("protocol.json allocation differs from the frozen Q-Clip request")
    training = value.get("training") or {}
    if any(training.get(k) != v for k, v in {"updates": 80, "contexts_per_update": 1, "exposures": 80}.items()):
        raise RuntimeError("protocol.json training coverage differs from the frozen Q-Clip request")
    expected = value.get("expected") or {}
    if any(expected.get(k) != v for k, v in {
        "heldout_queries_per_arm": 80, "calibration_queries_per_arm": 80,
        "target_linears": 614, "new_teacher_queries": 0, "closedloop_episodes": 0,
    }.items()):
        raise RuntimeError("protocol.json evidence scope differs from the frozen Q-Clip request")
    return value


PROTOCOL = load_protocol()
ROOT = PROTOCOL["remote_root"]
CONTROL = ROOT + "/control"
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"
STATE_PATH = CONTROL + "/submission.json"
SELECT = "1:ncpus=16:mem=110gb:ngpus=1"
SPECS = {
    "short": {"job_name": JOB_NAMES["short"], "walltime": PROTOCOL["allocation"]["short_walltime"]},
    "long": {"job_name": JOB_NAMES["long"], "walltime": PROTOCOL["allocation"]["long_walltime"]},
    "resume": {"job_name": JOB_NAMES["resume"], "walltime": PROTOCOL["allocation"]["long_walltime"]},
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
    value = _json(sftp, STATE_PATH) or {"schema_version": 1, "jobs": {}}
    if value.get("schema_version") != 1 or not isinstance(value.get("jobs", {}), dict):
        raise RuntimeError("Unsupported remote submission state")
    value.setdefault("jobs", {})
    return value


def _save_state(transport, value: dict) -> None:
    with remote.sftp_client(transport) as sftp:
        _write_json(sftp, STATE_PATH, value)


def upload(transport) -> list[str]:
    _mkdirs(transport)
    uploaded = []
    with remote.sftp_client(transport) as sftp:
        state = _state(sftp)
        frozen = bool(state.get("jobs"))
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
        if _json(sftp, ROOT + "/protocol.json") != PROTOCOL:
            raise RuntimeError("Remote frozen protocol differs from the local protocol")


def _named_jobs(transport, name: str) -> list[str]:
    if name not in JOB_NAMES.values() or not JOB_NAME_RE.fullmatch(name):
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
    keys = (
        "job_state", "Exit_status", "Job_Name", "queue", "exec_host",
        "resources_used.walltime", "Resource_List.walltime", "Resource_List.ngpus",
        "Resource_List.ncpus", "Resource_List.mem", "Resource_List.select",
    )
    fields = {}
    for key in keys:
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return fields


def _scheduler(transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=35)
    return {"job_id": job_id, "query_rc": code, **_parse_qstat(output)}


def _job_record(state: dict, label: str) -> dict:
    if label not in SPECS:
        raise ValueError("Unknown Q-Clip walltime label")
    jobs = state.setdefault("jobs", {})
    record = jobs.setdefault(label, {})
    if not isinstance(record, dict):
        raise RuntimeError("Malformed durable PBS job record")
    return record


def _valid_handle(handle: dict | None, label: str) -> bool:
    return bool(
        isinstance(handle, dict)
        and HANDLE.fullmatch(str(handle.get("job_id", "")))
        and handle.get("job_name") == SPECS[label]["job_name"]
        and handle.get("protocol") == PROTOCOL["protocol"]
        and handle.get("walltime") == SPECS[label]["walltime"]
    )


def _recover(transport, state: dict, label: str) -> dict:
    record = _job_record(state, label)
    intent = record.get("intent") or {}
    spec = SPECS[label]
    if (intent.get("job_name") != spec["job_name"] or intent.get("protocol") != PROTOCOL["protocol"]
            or intent.get("walltime") != spec["walltime"]):
        raise RuntimeError("Durable PBS intent is malformed or belongs to another Q-Clip job")
    jobs = _named_jobs(transport, spec["job_name"])
    if len(jobs) != 1:
        raise RuntimeError("qsub outcome is unresolved; do not issue a second submission")
    job_id = jobs[0]
    scheduler = _scheduler(transport, job_id)
    if scheduler.get("query_rc") != 0 or scheduler.get("Job_Name") != spec["job_name"]:
        raise RuntimeError("Exact-name PBS handle could not be verified")
    handle = {
        "job_id": job_id, "job_name": spec["job_name"], "protocol": PROTOCOL["protocol"],
        "walltime": spec["walltime"], "status": "recovered", "confirmed_utc": utc_now(),
    }
    intent.update(job_id=job_id, status="confirmed", recovered_utc=utc_now())
    record.update(intent=intent, handle=handle)
    _save_state(transport, state)
    return handle


def _resolve_jobs(transport, state: dict) -> dict:
    handles = {}
    for label in SPECS:
        record = _job_record(state, label)
        handle = record.get("handle")
        if handle:
            if not _valid_handle(handle, label):
                raise RuntimeError("Saved PBS handle is invalid or belongs to another Q-Clip job")
            handles[label] = handle
        elif record.get("intent"):
            handles[label] = _recover(transport, state, label)
    return handles


def _qsub_command(label: str) -> str:
    spec = SPECS[label]
    name = spec["job_name"]
    return (
        "qsub -N " + name + " -q normal -P " + PROJECT
        + " -l select=" + shlex.quote(SELECT)
        + " -l walltime=" + shlex.quote(spec["walltime"])
        + " -o " + shlex.quote(ARTIFACTS + "/" + name + ".pbs.out")
        + " -j oe " + shlex.quote(ROOT + "/run.pbs")
    )


def _submit_one(transport, state: dict, label: str) -> dict:
    record = _job_record(state, label)
    if record.get("handle"):
        if not _valid_handle(record["handle"], label):
            raise RuntimeError("Saved PBS handle is invalid")
        return record["handle"]
    if record.get("intent"):
        return _recover(transport, state, label)
    spec = SPECS[label]
    if _named_jobs(transport, spec["job_name"]):
        raise RuntimeError("Exact PBS job name already exists without a matching durable intent")
    intent = {
        "protocol": PROTOCOL["protocol"], "job_name": spec["job_name"],
        "script": ROOT + "/run.pbs", "walltime": spec["walltime"],
        "status": "intent_before_qsub", "created_utc": utc_now(),
        "select": SELECT, "queue": "normal", "ncpus": 16, "memory_gb": 110, "gpus": 1,
    }
    record.update(intent=intent, handle=None)
    _save_state(transport, state)
    try:
        code, output = remote.command(transport, _qsub_command(label), timeout=45)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        _save_state(transport, state)
        raise RuntimeError("qsub outcome is unknown; recover the exact name before any retry") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", return_code=code, response=output[-1000:], updated_utc=utc_now())
        _save_state(transport, state)
        raise RuntimeError("qsub outcome is unknown or rejected; do not submit again")
    handle = {
        "job_id": job_id, "job_name": spec["job_name"], "protocol": PROTOCOL["protocol"],
        "walltime": spec["walltime"], "status": "confirmed", "confirmed_utc": utc_now(),
    }
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    record.update(intent=intent, handle=handle)
    _save_state(transport, state)
    return handle


def _has_checkpoint_or_endpoint(sftp) -> bool:
    try:
        if sftp.stat(RESULTS + "/checkpoint.pt").st_size > 0:
            return True
    except OSError as exc:
        if not _absent(exc):
            raise
    # Four bounded result directories; inspect names only, never recurse or read endpoint payloads.
    for arm in ("q0", "qclip"):
        for split in ("calibration", "heldout"):
            path = RESULTS + "/endpoints/" + arm + "/" + split
            try:
                names = sftp.listdir(path)
            except OSError as exc:
                if _absent(exc):
                    continue
                raise
            if any(name.endswith(".json") for name in names):
                return True
    return False


def _resume_ready(transport, value: dict) -> dict:
    winner = value.get("winner") or {}
    previous_name = winner.get("winner_job_name")
    previous_label = next((key for key in ("short", "long") if JOB_NAMES[key] == previous_name), None)
    previous = value.get("jobs", {}).get(previous_label, {}) if previous_label else {}
    scheduler = previous.get("scheduler") or {}
    handle = previous.get("handle") or {}
    if (winner.get("protocol") != PROTOCOL["protocol"] or winner.get("status") != "interrupted" or not previous_label
            or winner.get("winner_job_id") != handle.get("job_id")
            or scheduler.get("query_rc") != 0 or scheduler.get("Job_Name") != previous_name
            or scheduler.get("job_state") not in ("F", "X") or scheduler.get("Exit_status") != "75"):
        raise RuntimeError("--resume requires the recorded writer to be terminal with PBS Exit_status=75")
    failure = value.get("failure")
    if (not isinstance(failure, dict) or failure.get("protocol") != PROTOCOL["protocol"]
            or failure.get("pbs_jobid") != handle.get("job_id")
            or failure.get("error") != "TimeoutError" or failure.get("checkpointed") is not True):
        raise RuntimeError("--resume requires the worker's checkpointed timeout receipt")
    for label in ("short", "long"):
        row = value.get("jobs", {}).get(label) or {}
        state = (row.get("scheduler") or {}).get("job_state")
        query_rc = (row.get("scheduler") or {}).get("query_rc")
        if row and (query_rc != 0 or state in ("Q", "R")):
            raise RuntimeError("--resume requires all original short/long jobs to be terminal")
    with remote.sftp_client(transport) as sftp:
        if not _has_checkpoint_or_endpoint(sftp):
            raise RuntimeError("--resume requires results/checkpoint.pt or at least one committed endpoint JSON")
    return {"previous_label": previous_label, "previous_handle": handle, "failure": failure}


def submit(transport, resume: bool = False) -> dict:
    _mkdirs(transport)
    with remote.sftp_client(transport) as sftp:
        state = _state(sftp)
    _resolve_jobs(transport, state)
    _require_uploaded(transport)
    if resume:
        if _job_record(state, "resume").get("handle"):
            return {"existing": True, "handle": _job_record(state, "resume")["handle"]}
        if _job_record(state, "resume").get("intent"):
            return {"handle": _submit_one(transport, state, "resume")}
        current = status(transport)
        previous = _resume_ready(transport, current)
        record = _job_record(state, "resume")
        if _named_jobs(transport, SPECS["resume"]["job_name"]):
            raise RuntimeError("Exact resume job name already exists without a matching durable intent")
        intent = {
            "protocol": PROTOCOL["protocol"], "job_name": SPECS["resume"]["job_name"],
            "script": ROOT + "/run.pbs", "walltime": SPECS["resume"]["walltime"],
            "status": "intent_before_qsub", "created_utc": utc_now(), "resume_of": previous["previous_handle"]["job_id"],
            "select": SELECT, "queue": "normal", "ncpus": 16, "memory_gb": 110, "gpus": 1,
        }
        record.update(intent=intent, handle=None)
        _save_state(transport, state)
        handle = _submit_one(transport, state, "resume")
        reconciled = reconcile(transport)
        return {"handle": handle, "resume_of": previous["previous_handle"]["job_id"], "reconcile": reconciled}
    submitted = []
    for label in SPECS:
        if label == "resume":
            continue
        existing = _job_record(state, label).get("handle")
        handle = _submit_one(transport, state, label)
        if not existing:
            submitted.append(handle)
    reconciled = reconcile(transport)
    with remote.sftp_client(transport) as sftp:
        state = _state(sftp)
    return {"submitted": submitted, "jobs": {k: v.get("handle") for k, v in state["jobs"].items()},
            "reconcile": reconciled}


def _finite_leq(value, maximum: float) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= maximum)


def _resource_match(scheduler: dict, label: str) -> bool:
    queue = str(scheduler.get("queue") or "")
    if (queue != "normal" and queue != "gdev" and not re.fullmatch(r"g[0-9]+", queue)) or scheduler.get("Resource_List.walltime") != SPECS[label]["walltime"]:
        return False
    select = scheduler.get("Resource_List.select") or ""
    select_fields = dict(part.split("=", 1) for part in select.split(":") if "=" in part)
    ncpus = scheduler.get("Resource_List.ncpus") or select_fields.get("ncpus")
    ngpus = scheduler.get("Resource_List.ngpus") or select_fields.get("ngpus")
    mem = scheduler.get("Resource_List.mem") or select_fields.get("mem", "")
    return str(ncpus) == "16" and str(ngpus) == "1" and str(mem).lower().replace(" ", "") in ("110gb", "110g")


def _summary_gate(summary: dict | None, winner: dict | None) -> bool:
    if not isinstance(summary, dict) or not isinstance(winner, dict):
        return False
    job_id = winner.get("winner_job_id")
    arms = summary.get("arms")
    required = (
        summary.get("protocol") == PROTOCOL["protocol"]
        and summary.get("pbs_jobid") == job_id
        and summary.get("complete") is True and summary.get("status") == "complete"
        and summary.get("updates") == 80 and summary.get("exposures") == 80
        and summary.get("target_linears") == 614
        and summary.get("heldout_queries_per_arm") == 80
        and summary.get("calibration_queries_per_arm") == 80
        and summary.get("new_teacher_queries") == 0 and summary.get("closedloop_episodes") == 0
        and summary.get("weights_frozen") is True and summary.get("rotation_frozen") is True
        and _finite_leq(summary.get("initial_parity_max_abs"), 1e-6)
        and _finite_leq(summary.get("formal_parity_max_abs"), 1e-6)
        and isinstance(arms, dict) and set(arms) == {"q0", "qclip"}
    )
    if not required:
        return False
    suites = PROTOCOL["suites"]
    def valid_metrics(value: dict) -> bool:
        return all(_finite_leq(value.get(metric), float("inf")) for metric in METRICS)
    for arm in ("q0", "qclip"):
        row = arms[arm]
        if not isinstance(row, dict):
            return False
        for split, contexts_per_suite in (("calibration", 20), ("heldout", 10)):
            aggregate = row.get(split)
            if not isinstance(aggregate, dict):
                return False
            expected_contexts = 80 if split == "calibration" else 40
            if aggregate.get("queries") != 80 or aggregate.get("contexts") != expected_contexts:
                return False
            by_suite = aggregate.get("by_suite")
            overall = aggregate.get("overall")
            if (not isinstance(by_suite, dict) or set(by_suite) != set(suites)
                    or not isinstance(overall, dict) or not valid_metrics(overall)):
                return False
            for suite in suites:
                value = by_suite[suite]
                if (not isinstance(value, dict) or value.get("contexts") != contexts_per_suite
                        or value.get("queries") != contexts_per_suite * (1 if split == "calibration" else 2)
                        or not valid_metrics(value)):
                    return False
    return True


def complete_gate(value: dict) -> bool:
    """Require PBS success, winner identity, fixed-input coverage, and worker receipts."""
    try:
        winner = value["winner"]
        winner_name = winner["winner_job_name"]
        label = next(k for k, name in JOB_NAMES.items() if name == winner_name)
        job = value["jobs"][label]
        scheduler, artifact = job["scheduler"], job["artifact"]
        handle = job["handle"]
        return bool(
            winner.get("protocol") == PROTOCOL["protocol"]
            and winner.get("status") == "complete"
            and winner.get("winner_job_id") == handle.get("job_id")
            and _valid_handle(handle, label)
            and scheduler.get("query_rc") == 0
            and scheduler.get("job_state") in ("F", "X")
            and scheduler.get("Exit_status") == "0"
            and scheduler.get("Job_Name") == JOB_NAMES[label]
            and _resource_match(scheduler, label)
            and artifact.get("exit_code") == "0"
            and artifact.get("PHASE_COMPLETE") is True
            and artifact.get("RUN_PHASE", {}).get("phase") == "complete"
            and artifact.get("RUN_PHASE", {}).get("pbs_jobid") == handle.get("job_id")
            and artifact.get("RUN_PHASE", {}).get("protocol") == PROTOCOL["protocol"]
            and _summary_gate(value.get("summary"), winner)
        )
    except (KeyError, TypeError, AttributeError, StopIteration):
        return False


def _read_evidence(sftp) -> dict:
    winner = _json(sftp, CONTROL + "/winner.json")
    jobs = {}
    state = _state(sftp)
    for label, record in state.get("jobs", {}).items():
        if label not in SPECS or not _valid_handle(record.get("handle"), label):
            continue
        job_id = record["handle"]["job_id"]
        artifact = ARTIFACTS + "/" + job_id
        try:
            marker = sftp.stat(artifact + "/PHASE_COMPLETE")
            phase_complete = True
        except OSError as exc:
            if not _absent(exc):
                raise
            phase_complete = False
        jobs[label] = {
            "handle": record["handle"], "artifact": {
                "exit_code": _text(sftp, artifact + "/exit_code.txt", 64),
                "RUN_PHASE": _json(sftp, artifact + "/RUN_PHASE.json"),
                "PHASE_COMPLETE": phase_complete,
            },
        }
    return {
        "winner": winner,
        "jobs": jobs,
        "progress": _json(sftp, RESULTS + "/progress.json"),
        "summary": _json(sftp, RESULTS + "/summary.json"),
        "failure": _json(sftp, RESULTS + "/failure.json"),
    }


def _job_status(job: dict) -> str:
    scheduler = job["scheduler"]
    if scheduler.get("query_rc") != 0:
        return "query_inconclusive"
    if scheduler.get("job_state") == "Q":
        return "queued"
    if scheduler.get("job_state") == "R":
        return "running"
    if scheduler.get("job_state") in ("F", "X"):
        return "terminal"
    return "unknown"


def status(transport) -> dict:
    with remote.sftp_client(transport) as sftp:
        state = _state(sftp)
    handles = _resolve_jobs(transport, state)
    with remote.sftp_client(transport) as sftp:
        result = {"checked_utc": utc_now(), **_read_evidence(sftp)}
    for label, handle in handles.items():
        row = result["jobs"].setdefault(label, {"handle": handle, "artifact": {}})
        row["handle"] = handle
        row["scheduler"] = _scheduler(transport, handle["job_id"])
        row["status"] = _job_status(row)
    result["completion_passed"] = complete_gate(result)
    winner = result.get("winner") or {}
    result["status"] = "complete" if result["completion_passed"] else (
        "writer_running" if winner.get("status") == "running" else
        "submitted" if handles else "intent_unresolved" if state.get("jobs") else "not_submitted"
    )
    return result


def _compact(value: dict) -> dict:
    winner = value.get("winner") or {}
    progress = value.get("progress") or {}
    return {
        "status": value.get("status"), "winner_job_id": winner.get("winner_job_id"),
        "winner_job_name": winner.get("winner_job_name"), "winner_status": winner.get("status"),
        "completion_passed": value.get("completion_passed", False),
        "progress": {k: progress.get(k) for k in ("phase", "updates", "exposures", "heldout_queries", "calibration_queries") if k in progress},
        "jobs": {
            label: {"job_id": row.get("handle", {}).get("job_id"), "status": row.get("status"),
                    "job_state": (row.get("scheduler") or {}).get("job_state"),
                    "queue": (row.get("scheduler") or {}).get("queue"),
                    "exec_host": (row.get("scheduler") or {}).get("exec_host"),
                    "query_rc": (row.get("scheduler") or {}).get("query_rc")}
            for label, row in value.get("jobs", {}).items()
        },
    }


def reconcile(transport) -> dict:
    value = status(transport)
    winner = value.get("winner") or {}
    label = next((key for key, name in JOB_NAMES.items() if name == winner.get("winner_job_name")), None)
    if (not label or winner.get("protocol") != PROTOCOL["protocol"]
            or winner.get("winner_job_id") != (value.get("jobs", {}).get(label, {}).get("handle") or {}).get("job_id")):
        return {**_compact(value), "cancelled": [], "reason": "no verified winner receipt"}
    cancelled, attempts = [], []
    for loser, row in value.get("jobs", {}).items():
        if loser == label:
            continue
        scheduler, handle = row.get("scheduler") or {}, row.get("handle") or {}
        if scheduler.get("query_rc") != 0:
            attempts.append({"job_id": handle.get("job_id"), "reason": "scheduler query inconclusive"})
            continue
        if scheduler.get("Job_Name") != JOB_NAMES[loser] or not _valid_handle(handle, loser):
            attempts.append({"job_id": handle.get("job_id"), "reason": "identity mismatch"})
            continue
        if scheduler.get("job_state") not in ("Q", "R"):
            continue
        code, output = remote.command(transport, "qdel " + shlex.quote(handle["job_id"]), timeout=35)
        if code == 0:
            cancelled.append(handle["job_id"])
        attempts.append({"job_id": handle["job_id"], "qdel_rc": code, "output": output[-1000:]})
    return {**_compact(value), "cancelled": cancelled, "attempts": attempts}


def logs(transport) -> dict:
    value = status(transport)
    tails = {}
    for label, row in value.get("jobs", {}).items():
        handle = row.get("handle") or {}
        if not _valid_handle(handle, label):
            continue
        path = ARTIFACTS + "/" + handle["job_id"] + "/job.log"
        code, output = remote.command(transport, "tail -c 8192 " + shlex.quote(path) + " 2>/dev/null", timeout=20)
        tails[label] = {"job_id": handle["job_id"], "query_rc": code,
                        "tail": output.encode("utf-8", errors="replace")[-8192:].decode("utf-8", errors="replace")}
    return {**_compact(value), "logs": tails}


def pull(transport) -> dict:
    value = status(transport)
    target_name = (value.get("winner") or {}).get("winner_job_id") or "submission"
    target = HERE / "pulled" / target_name
    files = {"status.json": _compact(value)}
    for key in ("winner", "summary", "progress"):
        if value.get(key) is not None:
            files[key + ".json"] = value[key]
    for label, row in value.get("jobs", {}).items():
        files[f"jobs/{label}/scheduler.json"] = row.get("scheduler") or {"query_rc": None}
        artifact = row.get("artifact") or {}
        for key in ("RUN_PHASE",):
            if artifact.get(key) is not None:
                files[f"jobs/{label}/{key}.json"] = artifact[key]
        if artifact.get("exit_code") is not None:
            files[f"jobs/{label}/exit_code.txt"] = artifact["exit_code"] + "\n"
    written = []
    for name, data in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            if name.endswith(".txt"):
                stream.write(data)
            else:
                json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            temp = Path(stream.name)
        temp.replace(path)
        written.append(str(path))
    return {**_compact(value), "pulled_files": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "reconcile", "verify", "logs", "pull"))
    parser.add_argument("--resume", action="store_true", help="submit one 12-hour resume job after a checkpointed exit 75")
    args = parser.parse_args()
    if args.resume and args.action != "submit":
        parser.error("--resume is only valid with submit")
    jump, transport = direct_connect()
    try:
        if args.action == "upload":
            result, code = {"uploaded": upload(transport)}, 0
        elif args.action == "submit":
            result, code = submit(transport, resume=args.resume), 0
        elif args.action == "status":
            value = status(transport)
            result, code = _compact(value), 0
        elif args.action == "reconcile":
            result, code = reconcile(transport), 0
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
