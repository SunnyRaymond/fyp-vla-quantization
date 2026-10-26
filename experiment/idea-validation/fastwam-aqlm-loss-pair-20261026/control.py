"""Small-file, host-key-verified PBS control for the frozen AQLM loss pair.

Only this controller performs scheduler writes.  A qsub whose result is not
verified is recorded as an unresolved intent and is never blindly repeated.
"""
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
_remote_spec = importlib.util.spec_from_file_location(
    "aqlm_existing_remote_controller",
    HERE.parent / "fastwam-vq-c80-per-channel-plus" / "control.py",
)
if _remote_spec is None or _remote_spec.loader is None:
    raise RuntimeError("Trusted remote helper is unavailable")
_remote_module = importlib.util.module_from_spec(_remote_spec)
_remote_spec.loader.exec_module(_remote_module)
remote = _remote_module.remote

LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
USER = "yguo017"
PROJECT = "personal-yguo017"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
ATTEMPT = re.compile(r"^[a-z][a-z0-9]{0,5}$")
PROTOCOL_NAME = "fastwam-aqlm-loss-pair-v1"
STAGES = ("bootstrap", "prepare", "pair")
ARMS = ("reconstruction", "ours")
JOB_PREFIX = {"bootstrap": "aqlmb261026", "prepare": "aqlmp261026", "pair": "aqlmx261026"}
FITTING_BUDGET_FIELDS = {
    "init_max_iter", "init_max_points_per_centroid", "max_epochs", "steps_per_epoch",
    "relative_mse_tolerance", "profile", "max_cluster_vectors", "note",
}
FITTING_BUDGET_VALUES = {
    "init_max_iter": 20, "init_max_points_per_centroid": 64, "max_epochs": 10,
    "steps_per_epoch": 20, "relative_mse_tolerance": 0.005, "beam_size": 1,
    "profile": "light_v2", "max_cluster_vectors": 16384,
}
FITTING_MINIMAL_VALUES = {
    "init_max_iter": 5, "init_max_points_per_centroid": 16, "max_cluster_vectors": 4096,
    "max_epochs": 1, "steps_per_epoch": 5, "relative_mse_tolerance": 0.005,
    "beam_size": 1, "profile": "minimal_v3",
}
FILES = ("protocol.json", "fit_budget_amendment.json", "run.pbs", "environment.sh", "bootstrap.py",
         "worker.py", "aqlm_banks.py", "analysis.py", "test_worker.py", "test_control.py",
         "control.py", "README.zh.md")
REQUIRED_BASE = ("protocol.json", "run.pbs", "environment.sh", "control.py")
RESOURCES = {
    "bootstrap": {"queue": "qdev", "select": "1:ncpus=4:mem=32gb:ngpus=0", "gpus": 0,
                  "ncpus": 4, "memory_gb": 32, "walltime": "01:00:00"},
    "prepare": {"queue": "normal", "select": "1:ncpus=16:mem=180gb:ngpus=1", "gpus": 1,
                "ncpus": 16, "memory_gb": 180, "walltime": "24:00:00"},
    "pair": {"queue": "normal", "select": "1:ncpus=32:mem=220gb:ngpus=2", "gpus": 2,
             "ncpus": 32, "memory_gb": 220, "walltime": "24:00:00"},
}


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("protocol.json does not match the frozen AQLM loss-pair protocol")
    if value.get("remote_root") != "/scratch/users/ntu/yguo017/fastwam-aqlm-loss-pair-20261026":
        raise RuntimeError("protocol.json names an unexpected remote root")
    if value.get("arms") != list(ARMS):
        raise RuntimeError("protocol.json does not contain reconstruction/ours")
    expected = {
        "bootstrap": ("qdev", 4, 32, 0, "01:00:00"),
        "prepare": ("normal", 16, 180, 1, "24:00:00"),
        "pair": ("normal", 32, 220, 2, "24:00:00"),
    }
    execution = value.get("execution") or {}
    for stage, (queue, ncpus, memory, gpus, walltime) in expected.items():
        got = execution.get(stage) or {}
        if (got.get("queue"), got.get("ncpus"), got.get("memory_gb"), got.get("gpus"),
                got.get("walltime")) != (queue, ncpus, memory, gpus, walltime):
            raise RuntimeError(f"protocol.json execution differs for {stage}")
    if value.get("boundary", {}).get("closed_loop_authorized") is not False:
        raise RuntimeError("closed-loop execution is not frozen off")
    return value


PROTOCOL = load_protocol()
ROOT = PROTOCOL["remote_root"]
CONTROL = ROOT + "/control"
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"


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


def _write_bytes(sftp, path: str, data: bytes) -> None:
    if len(data) > LIMIT:
        raise RuntimeError(f"Remote control file exceeds 256 KiB: {path}")
    with sftp.file(path + ".tmp", "wb") as stream:
        stream.write(data)
    sftp.posix_rename(path + ".tmp", path)


def _validate_fitting_amendment(old_protocol: dict, new_protocol: dict, amendment: dict) -> dict:
    """Allow exactly one declared fitting-only protocol transition."""
    if (not isinstance(amendment, dict) or amendment.get("protocol") != PROTOCOL_NAME
            or amendment.get("revision") not in (2, 3)
            or amendment.get("allowed_scope") != "fitting_only"):
        raise RuntimeError("Invalid fitting-only amendment metadata")
    revision = amendment["revision"]
    old_fitting = old_protocol.get("fitting")
    new_fitting = new_protocol.get("fitting")
    if (not isinstance(old_fitting, dict) or not isinstance(new_fitting, dict)
            or amendment.get("old_fitting") != old_fitting
            or amendment.get("new_fitting") != new_fitting
            or old_fitting == new_fitting):
        raise RuntimeError("Fitting amendment old/new values do not match the protocols")
    old_rest = {key: value for key, value in old_protocol.items() if key != "fitting"}
    new_rest = {key: value for key, value in new_protocol.items() if key != "fitting"}
    if old_rest != new_rest:
        raise RuntimeError("Protocol amendment changes fields outside fitting")
    changed = {key for key in set(old_fitting) | set(new_fitting)
               if old_fitting.get(key) != new_fitting.get(key)}
    if not changed.issubset(FITTING_BUDGET_FIELDS):
        raise RuntimeError("Protocol amendment changes fitting cohort or non-budget fields")
    expected = FITTING_BUDGET_VALUES if revision == 2 else FITTING_MINIMAL_VALUES
    if revision == 3 and any(old_fitting.get(key) != value for key, value in FITTING_BUDGET_VALUES.items()):
        raise RuntimeError("Minimal v3 amendment does not start from the light_v2 fitting budget")
    if any(new_fitting.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Protocol amendment does not match the declared light fitting budget")
    return {"revision": revision,
            "archive": ("control/protocol_before_light_v2.json" if revision == 2
                         else "control/protocol_before_minimal_v3.json"),
            "receipt": ("control/fit_budget_amendment_receipt.json" if revision == 2
                        else "control/fit_budget_amendment_receipt_v3.json")}


def _local_bytes(name: str) -> bytes:
    path = HERE / name
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError(f"Required small source is missing or too large: {name}")
    data = path.read_bytes().replace(b"\r\n", b"\n")
    data.decode("utf-8")
    return data


def _mkdirs(transport) -> None:
    command = "mkdir -p " + " ".join(shlex.quote(path) for path in (CONTROL, ARTIFACTS, RESULTS))
    code, output = remote.command(transport, command, timeout=25)
    if code:
        raise RuntimeError("Cannot prepare small PBS control directories: " + output[:1000])


def _stage_name(stage: str, attempt: str) -> str:
    if stage not in STAGES or not ATTEMPT.fullmatch(attempt):
        raise ValueError("Invalid stage or attempt")
    return JOB_PREFIX[stage] + attempt


def _stage_paths(stage: str) -> tuple[str, str]:
    if stage not in STAGES:
        raise ValueError("Unknown AQLM stage")
    return CONTROL + f"/{stage}_intent.json", CONTROL + f"/{stage}_state.json"


def _summary_path(stage: str) -> str:
    return RESULTS + "/summary.json" if stage == "pair" else RESULTS + f"/{stage}/summary.json"


def _artifact(stage: str, job_id: str) -> str:
    return ARTIFACTS + "/" + job_id


def _stage_record(sftp, stage: str) -> dict:
    intent_path, state_path = _stage_paths(stage)
    record = _json(sftp, state_path)
    if record is not None:
        return record
    intent = _json(sftp, intent_path)
    if intent is not None:
        return {"schema_version": 1, "protocol": PROTOCOL_NAME, "stage": stage,
                "attempts": [intent]}
    return {"schema_version": 1, "protocol": PROTOCOL_NAME, "stage": stage, "attempts": []}


def _write_stage_record(sftp, stage: str, record: dict) -> None:
    _, state_path = _stage_paths(stage)
    _write_json(sftp, state_path, record)


def _write_stage_receipts(sftp, stage: str, record: dict) -> None:
    """Keep a human-readable latest intent and exact handle beside the attempt ledger."""
    intent_path, _ = _stage_paths(stage)
    latest = _latest(record)
    if latest:
        _write_json(sftp, intent_path, latest)
        if latest.get("job_id"):
            _write_json(sftp, CONTROL + f"/{stage}_handle.json", latest)


def _latest(record: dict) -> dict | None:
    attempts = record.get("attempts") or []
    return attempts[-1] if attempts else None


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
    code, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=35)
    return {"job_id": job_id, "query_rc": code, **_parse_qstat(output)}


def _named_jobs(transport, name: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,15}", name):
        raise ValueError("Invalid exact PBS job name")
    code, output = remote.command(
        transport, "qselect -x -u " + USER + " -N " + shlex.quote(name) + " 2>&1", timeout=35)
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS exact-name history query failed; submission was not attempted")
    jobs = output.split()
    if any(not HANDLE.fullmatch(job) for job in jobs):
        raise RuntimeError("PBS returned an invalid exact-name handle")
    if len(set(jobs)) != len(jobs):
        raise RuntimeError("PBS exact-name history returned duplicate handles")
    return jobs


def _state(sftp, stage: str) -> dict:
    record = _stage_record(sftp, stage)
    if record.get("schema_version") != 1 or record.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("Unsupported or cross-protocol stage state")
    return record


def _save_stage(transport, stage: str, record: dict) -> None:
    with remote.sftp_client(transport) as sftp:
        _write_stage_record(sftp, stage, record)
        _write_stage_receipts(sftp, stage, record)


def _phase_marker(sftp, stage: str, job_id: str, marker: str = "PHASE_COMPLETE") -> bool:
    try:
        sftp.stat(_artifact(stage, job_id) + "/" + marker)
        return True
    except OSError as exc:
        if _absent(exc):
            return False
        raise


def _read_artifact(sftp, stage: str, job_id: str) -> dict:
    base = _artifact(stage, job_id)
    failure = _json(sftp, base + "/failure.json")
    if failure is None:
        failure_path = RESULTS + ("/failure.json" if stage == "pair" else f"/{stage}/failure.json")
        failure = _json(sftp, failure_path)
    if failure is None and stage == "pair":
        for arm in ARMS:
            failure = _json(sftp, RESULTS + f"/arms/{arm}/failure.json")
            if failure is not None:
                break
    return {
        "exit_code": _text(sftp, base + "/exit_code.txt", 64),
        "failure_exit_status": _text(sftp, base + "/failure.exit_status", 64),
        "phase_complete": _phase_marker(sftp, stage, job_id),
        "checkpointed": _phase_marker(sftp, stage, job_id, "PIPELINE_CHECKPOINTED"),
        "phase": _json(sftp, base + "/RUN_PHASE.json"),
        "failure": failure,
    }


def _is_terminal_success(scheduler: dict) -> bool:
    return (scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("F", "X")
            and str(scheduler.get("Exit_status")) == "0")


def _is_checkpoint_resume(scheduler: dict, artifact: dict, summary: dict | None) -> bool:
    return (scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("F", "X")
            and str(scheduler.get("Exit_status")) in {"75", "124", "143", "271"}
            and artifact.get("checkpointed") is True
            and (summary is None or (summary.get("status") == "checkpointed"
                                     and summary.get("complete") is not True)))


_is_checkpoint75 = _is_checkpoint_resume


def _is_explicit_failure_resume(scheduler: dict, artifact: dict, summary: dict | None) -> bool:
    failure = artifact.get("failure")
    return (scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("F", "X")
            and str(scheduler.get("Exit_status")) not in ("0", "75", "None")
            and isinstance(failure, dict) and failure.get("protocol") == PROTOCOL_NAME
            and failure.get("pbs_jobid") == scheduler.get("job_id")
            and any(failure.get(key) for key in ("error", "traceback", "reason")))


def _summary_belongs(summary: dict | None, job_id: str) -> bool:
    return summary is None or (isinstance(summary, dict) and summary.get("pbs_jobid") in (None, job_id))


def _read_summary(sftp, stage: str) -> dict | None:
    return _json(sftp, _summary_path(stage))


def _summary_complete(stage: str, summary: dict | None) -> bool:
    if not isinstance(summary, dict):
        return False
    return (summary.get("protocol") == PROTOCOL_NAME and summary.get("complete") is True
            and summary.get("status", "complete") == "complete"
            and (summary.get("stage", stage) == stage or stage == "pair"))


def _parent_complete(transport, stage: str) -> dict:
    if stage == "bootstrap":
        return {}
    parent = "bootstrap" if stage == "prepare" else "prepare"
    with remote.sftp_client(transport) as sftp:
        record = _state(sftp, parent)
        previous = _latest(record)
        if not previous or not HANDLE.fullmatch(str(previous.get("job_id", ""))):
            raise RuntimeError(f"{stage} requires a confirmed {parent} handle")
        scheduler = _scheduler(transport, previous["job_id"])
        if scheduler.get("Job_Name") != previous.get("job_name") or not _is_terminal_success(scheduler):
            raise RuntimeError(f"{parent} is not an identity-verified successful terminal job")
        artifact = _read_artifact(sftp, parent, previous["job_id"])
        summary = _read_summary(sftp, parent)
        if (artifact.get("phase_complete") is not True or artifact.get("exit_code") != "0"
                or not _summary_complete(parent, summary)
                or (summary.get("pbs_jobid") not in (None, previous["job_id"]))):
            raise RuntimeError(f"{parent} lacks summary.complete and PHASE_COMPLETE")
    return {"job_id": previous["job_id"], "job_name": previous.get("job_name"),
            "scheduler": scheduler, "summary": summary}


def _qsub_command(stage: str, attempt: str, dependency: str | None = None) -> str:
    resources = RESOURCES[stage]
    name = _stage_name(stage, attempt)
    parts = ["qsub", "-N", shlex.quote(name), "-q", resources["queue"], "-P", PROJECT,
             "-l", "select=" + resources["select"], "-l", "walltime=" + resources["walltime"],
             "-v", "FW_AQLM_STAGE=" + stage, "-o", shlex.quote(ARTIFACTS + "/" + name + ".pbs.out"),
             "-j", "oe"]
    if dependency:
        parts.extend(["-W", "depend=afterok:" + shlex.quote(dependency)])
    parts.append(shlex.quote(ROOT + "/run.pbs"))
    return " ".join(parts)


def _require_uploaded(transport) -> None:
    with remote.sftp_client(transport) as sftp:
        local_names = [name for name in FILES if (HERE / name).is_file()]
        for name in local_names:
            if _small_bytes(sftp, ROOT + "/" + name) != _local_bytes(name):
                raise RuntimeError("Upload is incomplete or differs from the local frozen source: " + name)
        if _json(sftp, ROOT + "/protocol.json") != PROTOCOL:
            raise RuntimeError("Remote frozen protocol differs from the local protocol")


def _active(transport, record: dict) -> bool:
    for attempt in record.get("attempts") or []:
        job_id = attempt.get("job_id")
        if not HANDLE.fullmatch(str(job_id or "")):
            continue
        scheduler = _scheduler(transport, job_id)
        if scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("Q", "R", "E", "H"):
            return True
    return False


def _verify_all_stage_schedulers_quiescent(transport, records: dict[str, dict]) -> None:
    """Require a successful scheduler read for every existing handle before a source transition."""
    for stage in STAGES:
        for attempt in records[stage].get("attempts") or []:
            job_id = attempt.get("job_id")
            if not HANDLE.fullmatch(str(job_id or "")):
                if attempt.get("status") in ("intent_before_qsub", "qsub_outcome_unknown") or attempt.get("job_name"):
                    raise RuntimeError(f"Unresolved {stage} handle blocks source transition")
                continue
            scheduler = _scheduler(transport, job_id)
            if scheduler.get("query_rc") != 0:
                raise RuntimeError(f"Scheduler query for {stage} {job_id} is inconclusive")
            if scheduler.get("job_state") in ("Q", "R", "E", "H"):
                raise RuntimeError(f"Active PBS job {job_id} blocks source transition")


def _apply_fitting_amendment(transport, sftp, records: dict[str, dict], stage: str) -> bool:
    """Atomically archive the old protocol and publish one declared fitting-only amendment."""
    old_raw = _small_bytes(sftp, ROOT + "/protocol.json")
    new_raw = _local_bytes("protocol.json")
    if old_raw is None or old_raw == new_raw:
        return False
    if stage != "prepare":
        raise RuntimeError("Fitting amendment is allowed only on prepare resume")
    if not (HERE / "fit_budget_amendment.json").is_file():
        raise RuntimeError("Changed protocol requires fit_budget_amendment.json")
    if _latest(records["pair"]):
        raise RuntimeError("Fitting amendment is forbidden after pair submission")
    latest = _latest(records["prepare"])
    if not latest or not HANDLE.fullmatch(str(latest.get("job_id", ""))):
        raise RuntimeError("Fitting amendment requires an existing prepare attempt")
    scheduler = _scheduler(transport, latest["job_id"])
    artifact = _read_artifact(sftp, "prepare", latest["job_id"])
    summary = _read_summary(sftp, "prepare")
    if not _summary_belongs(summary, latest["job_id"]):
        raise RuntimeError("Prepare resume receipt belongs to another PBS job")
    if not (_is_checkpoint_resume(scheduler, artifact, summary)
            or _is_explicit_failure_resume(scheduler, artifact, summary)):
        raise RuntimeError("Prepare is not an explicit resumable terminal attempt")
    _parent_complete(transport, "prepare")
    try:
        old_protocol = json.loads(old_raw.decode("utf-8"))
        amendment = json.loads(_local_bytes("fit_budget_amendment.json").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Protocol or fitting amendment JSON is invalid") from exc
    spec = _validate_fitting_amendment(old_protocol, PROTOCOL, amendment)

    archive_path = ROOT + "/" + spec["archive"]
    archived = _small_bytes(sftp, archive_path)
    if archived is None:
        _write_bytes(sftp, archive_path, old_raw)
    elif archived != old_raw:
        raise RuntimeError("Archived pre-amendment protocol differs; refusing overwrite")

    receipt_path = ROOT + "/" + spec["receipt"]
    receipt = {"protocol": PROTOCOL_NAME, "revision": spec["revision"], "stage": "prepare",
               "pbs_parent": latest["job_id"], "archive": spec["archive"],
               "allowed_scope": amendment["allowed_scope"], "user_request": amendment.get("user_request"),
               "old_fitting": amendment["old_fitting"], "new_fitting": amendment["new_fitting"],
               "amendment": amendment, "updated_utc": utc_now()}
    previous_receipt = _json(sftp, receipt_path)
    if previous_receipt is not None:
        if (previous_receipt.get("protocol") != PROTOCOL_NAME
                or previous_receipt.get("revision") != spec["revision"]
                or previous_receipt.get("archive") != spec["archive"]
                or previous_receipt.get("amendment") != amendment
                or previous_receipt.get("pbs_parent") != latest["job_id"]):
            raise RuntimeError("Existing fitting amendment receipt conflicts")
    else:
        _write_json(sftp, receipt_path, receipt)
    _write_bytes(sftp, ROOT + "/protocol.json", new_raw)
    return True


def upload(transport, *, resume_stage: str | None = None) -> list[str]:
    """Upload only small control sources; a frozen protocol is never overwritten."""
    _mkdirs(transport)
    if resume_stage is not None and resume_stage not in STAGES:
        raise ValueError("Unknown resume stage")
    uploaded = []
    # Runtime receipts, pulled folders, and vendor trees are deliberately outside this whitelist.
    local_files = [name for name in FILES if (HERE / name).is_file()]
    if "protocol.json" not in local_files:
        raise RuntimeError("protocol.json is required")
    with remote.sftp_client(transport) as sftp:
        records = {stage: _state(sftp, stage) for stage in STAGES}
        frozen = any(_latest(record) for record in records.values())
        transition = None
        if resume_stage:
            latest = _latest(records[resume_stage])
            _verify_all_stage_schedulers_quiescent(transport, records)
            _apply_fitting_amendment(transport, sftp, records, resume_stage)
            if not latest:
                parent = _parent_complete(transport, resume_stage)
                transition = {"protocol": PROTOCOL_NAME, "stage": resume_stage,
                              "pbs_parent": parent.get("job_id"), "updated_utc": utc_now()}
            else:
                scheduler = _scheduler(transport, latest["job_id"])
                artifact = _read_artifact(sftp, resume_stage, latest["job_id"])
                summary = _read_summary(sftp, resume_stage)
                if not _summary_belongs(summary, latest["job_id"]):
                    raise RuntimeError("Resume receipt belongs to a different PBS job")
                if not (_is_checkpoint_resume(scheduler, artifact, summary)
                        or _is_explicit_failure_resume(scheduler, artifact, summary)):
                    raise RuntimeError("Source updates require an explicit checkpoint or failure receipt")
                latest["status"] = "checkpointed" if _is_checkpoint_resume(scheduler, artifact, summary) else "failed"
                records[resume_stage]["attempts"][-1] = latest
                _write_stage_record(sftp, resume_stage, records[resume_stage])
                _write_stage_receipts(sftp, resume_stage, records[resume_stage])
        for name in local_files:
            data = _local_bytes(name)
            target = ROOT + "/" + name
            previous = _small_bytes(sftp, target)
            if name == "protocol.json" and previous is not None:
                if previous != data:
                    raise RuntimeError("Remote frozen protocol bytes differ; refusing overwrite")
                continue
            if previous == data:
                continue
            if frozen and not resume_stage:
                raise RuntimeError("Submitted sources are frozen; use a checkpointed later attempt")
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            uploaded.append(name)
        if transition:
            previous = _json(sftp, CONTROL + "/source_transition.json")
            if previous is not None and previous.get("protocol") != PROTOCOL_NAME:
                raise RuntimeError("Source-transition metadata belongs to another protocol")
            history = list((previous or {}).get("transitions") or [])
            history.append(transition)
            _write_json(sftp, CONTROL + "/source_transition.json",
                        {"protocol": PROTOCOL_NAME, "transitions": history})
    return uploaded


def _resolve_latest(transport, stage: str, record: dict) -> tuple[dict | None, dict]:
    latest = _latest(record)
    if latest:
        if (latest.get("status") in ("intent_before_qsub", "qsub_outcome_unknown")
                or not latest.get("job_id")):
            jobs = _named_jobs(transport, latest.get("job_name", ""))
            if len(jobs) != 1:
                raise RuntimeError("qsub outcome is unresolved; no second submission will be issued")
            job_id = jobs[0]
            scheduler = _scheduler(transport, job_id)
            if scheduler.get("query_rc") != 0 or scheduler.get("Job_Name") != latest.get("job_name"):
                raise RuntimeError("Exact-name PBS handle could not be identity-verified")
            latest = dict(latest, job_id=job_id, status="confirmed", recovered_utc=utc_now())
            record["attempts"][-1] = latest
            _save_stage(transport, stage, record)
        elif not HANDLE.fullmatch(str(latest.get("job_id"))):
            raise RuntimeError("Saved PBS handle is invalid")
        return latest, record
    return None, record


def _required_for_stage(stage: str) -> tuple[str, ...]:
    required = list(REQUIRED_BASE)
    if stage == "bootstrap":
        required += ["bootstrap.py", "aqlm_banks.py"]
    elif stage == "prepare":
        required += ["worker.py", "aqlm_banks.py"]
    else:
        required += ["worker.py", "aqlm_banks.py", "analysis.py"]
    return tuple(required)


def _require_stage_sources_uploaded(transport, stage: str) -> None:
    _require_uploaded(transport)
    with remote.sftp_client(transport) as sftp:
        for name in _required_for_stage(stage):
            if _small_bytes(sftp, ROOT + "/" + name) is None:
                raise RuntimeError(f"Required {stage} source is not uploaded: {name}")


def submit(transport, stage: str, attempt: str = "a", *, resume: bool = False) -> dict:
    if stage not in STAGES or not ATTEMPT.fullmatch(attempt):
        raise ValueError("Invalid stage or attempt")
    _mkdirs(transport)
    with remote.sftp_client(transport) as sftp:
        record = _state(sftp, stage)
    latest, record = _resolve_latest(transport, stage, record)
    if latest:
        scheduler = _scheduler(transport, latest["job_id"])
        with remote.sftp_client(transport) as sftp:
            artifact = _read_artifact(sftp, stage, latest["job_id"])
            summary = _read_summary(sftp, stage)
        if scheduler.get("job_state") in ("Q", "R", "E", "H"):
            return {"existing": True, "stage": stage, "handle": latest, "scheduler": scheduler}
        if (_is_terminal_success(scheduler) and artifact.get("phase_complete")
                and _summary_complete(stage, summary)
                and summary.get("pbs_jobid") in (None, latest["job_id"])):
            return {"existing": True, "stage": stage, "handle": latest, "scheduler": scheduler}
        resumable = (_summary_belongs(summary, latest["job_id"])
                     and (_is_checkpoint_resume(scheduler, artifact, summary)
                          or _is_explicit_failure_resume(scheduler, artifact, summary)))
        if not resume or not resumable:
            if str(scheduler.get("Exit_status")) not in ("0", "75", "124", "143", "271"):
                raise RuntimeError("A non-checkpoint terminal attempt cannot be resubmitted")
            raise RuntimeError("Stage is not an exact checkpointed resume candidate")
        if attempt == latest.get("attempt"):
            raise RuntimeError("Resume requires a new attempt identifier")
    if stage != "bootstrap":
        parent = _parent_complete(transport, stage)
        dependency = parent["job_id"]
    else:
        dependency = None
    _require_stage_sources_uploaded(transport, stage)
    name = _stage_name(stage, attempt)
    if _named_jobs(transport, name):
        raise RuntimeError("Exact PBS job name already exists without a matching durable intent")
    resources = RESOURCES[stage]
    intent = {"schema_version": 1, "protocol": PROTOCOL_NAME, "stage": stage, "attempt": attempt,
              "job_name": name, "script": ROOT + "/run.pbs", "dependency": dependency,
              "status": "intent_before_qsub", "created_utc": utc_now(), **resources}
    record.setdefault("attempts", []).append(intent)
    _save_stage(transport, stage, record)
    try:
        code, output = remote.command(transport, _qsub_command(stage, attempt, dependency), timeout=45)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        _save_stage(transport, stage, record)
        raise RuntimeError("qsub outcome is unknown; recover exact name before retrying") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", return_code=code, response=output[-1000:], updated_utc=utc_now())
        _save_stage(transport, stage, record)
        raise RuntimeError("qsub outcome is unknown or rejected; do not submit again")
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    record["attempts"][-1] = intent
    _save_stage(transport, stage, record)
    return {"existing": False, "stage": stage, "handle": intent}


def _stage_status(transport, stage: str) -> dict:
    with remote.sftp_client(transport) as sftp:
        record = _state(sftp, stage)
        latest = _latest(record)
        result = {"stage": stage, "attempt": latest.get("attempt") if latest else None,
                  "job_id": latest.get("job_id") if latest else None, "scheduler": None,
                  "summary": _read_summary(sftp, stage), "progress": None,
                  "artifact": None, "attempts": len(record.get("attempts") or [])}
        if latest and latest.get("job_id"):
            result["scheduler"] = _scheduler(transport, latest["job_id"])
            result["artifact"] = _read_artifact(sftp, stage, latest["job_id"])
            summary = result["summary"] or {}
            result["progress"] = (_json(sftp, RESULTS + f"/{stage}/progress.json")
                                  if stage != "pair" else summary.get("progress"))
            if stage == "pair":
                result["arms"] = {
                    arm: {
                        "progress": _json(sftp, RESULTS + f"/arms/{arm}/progress.json"),
                        "summary": _json(sftp, RESULTS + f"/arms/{arm}/summary.json"),
                        "failure": _json(sftp, RESULTS + f"/arms/{arm}/failure.json"),
                    } for arm in ARMS
                }
        return result


def _compact_stage(value: dict) -> dict:
    scheduler = value.get("scheduler") or {}
    artifact = value.get("artifact") or {}
    summary = value.get("summary") or {}
    row = {
        "attempt": value.get("attempt"), "job_id": value.get("job_id"),
        "job_state": scheduler.get("job_state"), "exit_status": scheduler.get("Exit_status"),
        "queue": scheduler.get("queue"), "exec_host": scheduler.get("exec_host"),
        "query_rc": scheduler.get("query_rc"), "summary_complete": _summary_complete(value["stage"], summary),
        "phase_complete": artifact.get("phase_complete", False),
        "checkpointed": artifact.get("checkpointed", False), "failure": bool(artifact.get("failure")),
        "progress": value.get("progress"),
    }
    if value["stage"] == "pair":
        row["arms"] = {}
        for arm, data in (value.get("arms") or {}).items():
            progress = data.get("progress") or {}
            row["arms"][arm] = {"phase": progress.get("phase"), "update": progress.get("update", progress.get("updates")),
                                "exposures": progress.get("exposures"), "calibration": progress.get("calibration"),
                                "heldout": progress.get("heldout"), "failure": bool(data.get("failure"))}
    return row


def _number(row: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in row:
            return row[key]
    for nested_name in ("counts", "evaluation", "metrics", "progress", "training"):
        nested = row.get(nested_name)
        if isinstance(nested, dict):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None


def _arm_complete(arm: str, row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    updates = _number(row, ("final_update", "final_updates", "updates", "update"))
    calibration = _number(row, ("calibration", "calibration_queries", "calibration_evaluation_queries"))
    heldout = _number(row, ("heldout", "heldout_queries", "heldout_evaluation_queries"))
    return (row.get("arm", arm) == arm and row.get("complete") is True
            and row.get("status", "complete") == "complete"
            and updates == 8 and calibration == 80 and heldout == 80)


def complete_gate(value: dict) -> bool:
    """Verify only fixed-input AQLM evidence; closed-loop is intentionally rejected."""
    try:
        handle, scheduler = value["handle"], value["scheduler"]
        artifact, results = value["artifact"], value["results"]
        job_id = handle["job_id"]
        if (not HANDLE.fullmatch(str(job_id)) or handle.get("job_name") != _stage_name("pair", handle.get("attempt", "a"))
                or scheduler.get("query_rc") != 0 or scheduler.get("job_state") not in ("F", "X")
                or str(scheduler.get("Exit_status")) != "0" or scheduler.get("Job_Name") != handle.get("job_name")
                or artifact.get("exit_code") != "0" or artifact.get("PHASE_COMPLETE") is not True
                or value.get("protocol_matches") is not True):
            return False
        summary = results.get("summary")
        if (not isinstance(summary, dict) or summary.get("protocol") != PROTOCOL_NAME
                or summary.get("complete") is not True
                or summary.get("status", "complete") != "complete"
                or summary.get("pbs_jobid") not in (None, job_id)):
            return False
        if summary.get("closed_loop_episodes", 0) != 0 or summary.get("closed_loop") is True \
                or summary.get("episodes", 0) != 0:
            return False
        arms = summary.get("arms")
        actual = results.get("arms") or {}
        if set(arms or {}) != set(ARMS) or set(actual) != set(ARMS):
            return False
        def actual_summary(arm: str) -> dict | None:
            row = actual[arm]
            if not isinstance(row, dict):
                return None
            return row.get("summary", row)
        return all(_arm_complete(arm, actual_summary(arm)) and arms[arm] == actual_summary(arm)
                   for arm in ARMS)
    except (KeyError, TypeError, AttributeError):
        return False


def status(transport) -> dict:
    stages = {stage: _stage_status(transport, stage) for stage in STAGES}
    pair = stages["pair"]
    pair_summary = pair.get("summary")
    value = {"checked_utc": utc_now(), "protocol_matches": False, "stages": stages,
             "handle": None, "scheduler": None, "artifact": None, "results": {"summary": pair_summary}}
    with remote.sftp_client(transport) as sftp:
        value["protocol_matches"] = _json(sftp, ROOT + "/protocol.json") == PROTOCOL
        value["results"]["arms"] = pair.get("arms") or {}
    if pair.get("job_id"):
        value.update(handle=pair.get("scheduler") and {
            **(_latest(_state_from_stage(stages, "pair")) or {}),
        }, scheduler=pair.get("scheduler"), artifact={
            "exit_code": (pair.get("artifact") or {}).get("exit_code"),
            "PHASE_COMPLETE": (pair.get("artifact") or {}).get("phase_complete", False),
        })
    value["completion_passed"] = _completion_from_status(value, pair)
    return value


def _state_from_stage(stages: dict, stage: str) -> dict:
    # Status rows retain the latest receipt only; this helper keeps complete_gate small.
    row = stages[stage]
    return {"attempts": [{"attempt": row.get("attempt"), "job_id": row.get("job_id"),
                           "job_name": _stage_name(stage, row.get("attempt") or "a")}]} 


def _completion_from_status(value: dict, pair: dict) -> bool:
    artifact = pair.get("artifact") or {}
    summary = pair.get("summary")
    arms = pair.get("arms") or {}
    synthetic = {
        "handle": _latest(_state_from_stage(value["stages"], "pair")) or {},
        "scheduler": pair.get("scheduler") or {},
        "artifact": {"exit_code": artifact.get("exit_code"), "PHASE_COMPLETE": artifact.get("phase_complete")},
        "protocol_matches": value.get("protocol_matches"),
        "results": {"summary": summary, "arms": arms},
    }
    return complete_gate(synthetic)


def logs(transport, stage: str) -> dict:
    if stage not in STAGES:
        raise ValueError("Unknown AQLM stage")
    current = _stage_status(transport, stage)
    job_id = current.get("job_id")
    if not job_id:
        raise RuntimeError("No PBS job handle is available for this stage")
    path = _artifact(stage, job_id) + "/job.log"
    code, output = remote.command(transport, "tail -c 8192 " + shlex.quote(path) + " 2>/dev/null", timeout=20)
    return {"stage": stage, "job_id": job_id, "query_rc": code,
            "tail": output.encode("utf-8", errors="replace")[-8192:].decode("utf-8", errors="replace")}


def pull(transport, stage: str | None = None) -> dict:
    value = status(transport)
    target = HERE / "pulled" / (value["stages"].get("pair", {}).get("job_id") or "not_submitted")
    target.mkdir(parents=True, exist_ok=True)
    selected = [stage] if stage else list(STAGES)
    files = {"status.json": _compact(value)}
    for item in selected:
        if item not in STAGES:
            raise ValueError("Unknown AQLM stage")
        row = value["stages"][item]
        for name, data in (("summary.json", row.get("summary")), ("progress.json", row.get("progress"))):
            if data is not None:
                files[f"{item}/{name}"] = data
    written = []
    for name, data in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > LIMIT:
            raise RuntimeError("Refusing to pull a non-small JSON artifact")
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            stream.write(encoded)
            temporary = Path(stream.name)
        temporary.replace(path)
        written.append(str(path))
    return {"pulled_files": written, **_compact(value)}


def _compact(value: dict) -> dict:
    return {"checked_utc": value.get("checked_utc"), "protocol_matches": value.get("protocol_matches"),
            "completion_passed": value.get("completion_passed", False),
            "stages": {stage: _compact_stage(row) for stage, row in (value.get("stages") or {}).items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "logs", "pull"))
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--attempt", default="a")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action in ("submit", "upload") and args.attempt and not ATTEMPT.fullmatch(args.attempt):
        parser.error("Use a short lowercase attempt")
    if args.action == "submit" and not args.stage:
        parser.error("--stage is required for submit")
    if args.action == "logs" and not args.stage:
        parser.error("--stage is required for logs")
    jump, transport = direct_connect()
    try:
        if args.action == "upload":
            result, code = {"uploaded": upload(transport, resume_stage=args.stage if args.resume else None)}, 0
        elif args.action == "submit":
            result, code = submit(transport, args.stage, args.attempt, resume=args.resume), 0
        elif args.action == "status":
            result, code = _compact(status(transport)), 0
        elif args.action == "verify":
            value = status(transport)
            result, code = {**_compact(value), "verified_complete": value["completion_passed"]}, int(not value["completion_passed"])
        elif args.action == "logs":
            result, code = logs(transport, args.stage), 0
        else:
            result, code = pull(transport, args.stage), 0
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return code
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
