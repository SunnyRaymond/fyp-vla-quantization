"""Small-file uploader and durable PBS submit/status controller."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import sys


HERE = Path(__file__).resolve().parent
ROTATION_CONTROL = HERE.parent / "fastwam-rotation-baselines" / "control.py"
_spec = importlib.util.spec_from_file_location("c8_latent_rotation_control", ROTATION_CONTROL)
previous = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(previous)
remote = previous.remote

PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = PROTOCOL["remote_root"]
CONTROL_ROOT = ROOT + "/control"
JOB_NAME = "fwalp1015"
EVALUATION_JOB_NAME = "fwale1015"
TRAINING_JOB_NAMES = {"rho0.1": "fwalt15r01", "rho1.5": "fwalt15r15"}
VALIDATION_JOB_NAMES = {"rho0.1": "fwav15r01", "rho1.5": "fwav15r15"}
TRAINING_SCRIPT = ROOT + "/train_one.pbs"
VALIDATION_SCRIPT = ROOT + "/validate_training.pbs"
EVALUATION_SCRIPT = ROOT + "/run.pbs"
TRAINING_RESOURCES = {
    "queue": "normal", "select": "1:ncpus=16:mem=110gb:ngpus=1",
    "gpus": 1, "ncpus": 16, "memory_gb": 110,
    "walltime": "01:59:00",
}
EVALUATION_RESOURCES = {
    "queue": "normal", "select": "1:ncpus=32:mem=220gb:ngpus=4",
    "gpus": 4, "ncpus": 32, "memory_gb": 220,
    "walltime": "24:00:00",
}
VALIDATION_RESOURCES = {
    "queue": "qdev", "select": "1:ncpus=1:mem=4gb:ngpus=0",
    "gpus": 0, "ncpus": 1, "memory_gb": 4, "walltime": "00:10:00",
}
FILES = (
    "run.pbs", "train_one.pbs", "validate_training.pbs", "protocol.json", "prepare_c8.py", "train_c8.py",
    "train_banks.py", "vq_adapter.py", "evaluate.py", "aggregate.py",
    "batched_transform.py", "transform_checks.py", "environment.sh",
)
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")


def _remote_json(sftp, path: str) -> dict | None:
    try:
        with sftp.file(path, "rb") as stream:
            data = stream.read(remote.LIMIT + 1)
    except OSError as exc:
        if getattr(exc, "errno", None) == 2:
            return None
        raise
    if len(data) > remote.LIMIT:
        raise RuntimeError(f"Remote control file exceeds the small-file limit: {path}")
    return json.loads(data.decode("utf-8"))


def _remote_text(sftp, path: str) -> str | None:
    try:
        with sftp.file(path, "rb") as stream:
            data = stream.read(remote.LIMIT + 1)
    except OSError as exc:
        if getattr(exc, "errno", None) == 2:
            return None
        raise
    if len(data) > remote.LIMIT:
        raise RuntimeError(f"Remote control file exceeds the small-file limit: {path}")
    return data.decode("utf-8", errors="replace").strip()


def _save_remote_json(sftp, path: str, value: dict) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(data) > remote.LIMIT:
        raise ValueError("Submission state exceeds the small-file limit")
    temp = path + ".tmp"
    with sftp.file(temp, "wb") as stream:
        stream.write(data)
    sftp.posix_rename(temp, path)


def _mkdirs(transport) -> None:
    command = "mkdir -p " + " ".join(shlex.quote(path) for path in (
        ROOT + "/results", ROOT + "/artifacts", CONTROL_ROOT))
    code, output = remote.command(transport, command, timeout=30)
    if code:
        raise RuntimeError(output)


def _query_matches(transport, job_name: str) -> list[str]:
    code, output = remote.command(
        transport,
        "qselect -x -u yguo017 -N " + shlex.quote(job_name) + " 2>&1",
        timeout=35,
    )
    if code != 0:
        raise RuntimeError("PBS history query failed; no submission was attempted")
    matches = output.split()
    if any(not HANDLE.fullmatch(job) for job in matches):
        raise RuntimeError("PBS history returned an invalid handle; no submission was attempted")
    return matches


def upload(transport, names: list[str] | None = None) -> list[str]:
    _mkdirs(transport)
    uploaded = []
    with remote.sftp_client(transport) as sftp:
        for name in names or list(FILES):
            if name not in FILES:
                raise ValueError(f"Not an approved small control file: {name}")
            path = HERE / name
            if not path.is_file():
                raise FileNotFoundError(path)
            data = path.read_bytes().replace(b"\r\n", b"\n")
            if len(data) > remote.LIMIT:
                raise ValueError(f"Control file exceeds the small-transfer limit: {name}")
            data.decode("utf-8")
            target, temp = ROOT + "/" + name, ROOT + "/" + name + ".tmp"
            with sftp.file(temp, "wb") as stream:
                stream.write(data)
            sftp.posix_rename(temp, target)
            uploaded.append(name)
    return uploaded


def _require_uploaded(transport) -> None:
    with remote.sftp_client(transport) as sftp:
        for name in FILES:
            try:
                attrs = sftp.stat(ROOT + "/" + name)
            except OSError as exc:
                if getattr(exc, "errno", None) == 2:
                    raise RuntimeError(f"Upload is incomplete; missing {name}") from exc
                raise
            if attrs.st_size > remote.LIMIT:
                raise RuntimeError(f"Remote control file exceeds the small-file limit: {name}")
        if _remote_json(sftp, ROOT + "/protocol.json") != PROTOCOL:
            raise RuntimeError("Remote protocol differs from the local frozen protocol")


def _submit(transport, job_name: str, intent_path: str, handle_path: str,
            previous_handle: dict | None = None, *, retry: bool = False,
            script: str = EVALUATION_SCRIPT, resources: dict | None = None,
            stage: str = "evaluation", arm: str | None = None,
            env: dict | None = None, update_latest: bool = True,
            purpose: str | None = None, source_training_job_id: str | None = None) -> dict:
    allocation = dict(EVALUATION_RESOURCES if resources is None else resources)
    intent = {
        "job_name": job_name,
        "script": script,
        "stage": stage,
        "remote_root": ROOT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **allocation,
        "resume_sequence": (previous_handle or {}).get("resume_sequence", 0),
        "status": "intent_before_qsub",
    }
    if stage == "evaluation":
        intent.update(episodes=560, workers=4, variants_per_worker=140)
    if arm:
        intent["arm"] = arm
    if stage == "evaluation" and (job_name == EVALUATION_JOB_NAME
                                   or job_name.startswith(EVALUATION_JOB_NAME + "r")):
        intent["restart_job_prefix"] = EVALUATION_JOB_NAME
    if env:
        intent["env"] = dict(env)
    if purpose:
        intent["purpose"] = purpose
    if source_training_job_id:
        intent["source_training_job_id"] = source_training_job_id
    if previous_handle:
        intent["resumes_job_id"] = previous_handle["job_id"]
    if retry:
        intent["retry"] = True
    with remote.sftp_client(transport) as sftp:
        _save_remote_json(sftp, intent_path, intent)

    log_name = "pbs.out" if job_name == JOB_NAME else job_name + ".pbs.out"
    command = ("qsub -N " + shlex.quote(job_name)
               + " -o " + shlex.quote(ROOT + "/artifacts/" + log_name))
    if env:
        exported = ",".join(f"{key}={value}" for key, value in env.items())
        command += " -v " + shlex.quote(exported)
    command += " " + shlex.quote(script)
    try:
        code, output = remote.command(transport, command, timeout=40)
    except Exception:
        raise
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="unknown_or_rejected", return_code=code, response=output[:1000])
        with remote.sftp_client(transport) as sftp:
            _save_remote_json(sftp, intent_path, intent)
        raise RuntimeError("qsub outcome is unknown or rejected; do not repeat blindly")
    intent.update(status="confirmed", job_id=job_id)
    with remote.sftp_client(transport) as sftp:
        _save_remote_json(sftp, intent_path, intent)
        _save_remote_json(sftp, handle_path, intent)
        if update_latest:
            _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", intent)
    return intent


def submit(transport) -> dict:
    _mkdirs(transport)
    _require_uploaded(transport)
    intent_path, handle_path = CONTROL_ROOT + "/submit_intent.json", CONTROL_ROOT + "/submit_handle.json"
    with remote.sftp_client(transport) as sftp:
        intent, handle = _remote_json(sftp, intent_path), _remote_json(sftp, handle_path)
        latest = _remote_json(sftp, CONTROL_ROOT + "/latest_handle.json")
    if latest:
        raise RuntimeError("A submitted run already exists; use status or resume")
    matches = _query_matches(transport, JOB_NAME)
    if len(matches) > 1:
        raise RuntimeError("More than one matching PBS job exists; reconcile manually without submitting")
    if matches:
        job_id = matches[0]
        if (intent or {}).get("job_id") not in (None, job_id) or (handle or {}).get("job_id") not in (None, job_id):
            raise RuntimeError("PBS history conflicts with the durable submission handle")
        recovered = {**(intent or {}), "job_name": JOB_NAME, "job_id": job_id,
                     "resume_sequence": 0, "status": "confirmed" if intent else "recovered"}
        with remote.sftp_client(transport) as sftp:
            _save_remote_json(sftp, intent_path, recovered)
            _save_remote_json(sftp, handle_path, recovered)
            _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", recovered)
        return recovered
    if intent or handle:
        raise RuntimeError("A durable or unresolved submission record exists; no second qsub will be issued")
    return _submit(transport, JOB_NAME, intent_path, handle_path)


def _restart(transport, *, retry_failed: bool) -> dict:
    with remote.sftp_client(transport) as sftp:
        previous_handle = _remote_json(sftp, CONTROL_ROOT + "/latest_handle.json")
        if previous_handle is None and not retry_failed:
            previous_handle = _remote_json(sftp, CONTROL_ROOT + "/submit_handle.json")
    if not previous_handle or not previous_handle.get("job_id"):
        action = "retry" if retry_failed else "resume"
        raise RuntimeError(f"No previous confirmed handle is available to {action}")
    if previous_handle.get("status") not in ("confirmed", "recovered"):
        raise RuntimeError("The latest handle is unresolved and must be reconciled before restart")
    sequence = int(previous_handle.get("resume_sequence", 0)) + 1
    restart_prefix = previous_handle.get("restart_job_prefix", JOB_NAME)
    job_name = f"{restart_prefix}r{sequence}"
    if len(job_name) > 15:
        raise RuntimeError("Resume sequence exceeds the supported PBS job-name length")
    intent_path = CONTROL_ROOT + f"/resume_intent_{sequence}.json"
    handle_path = CONTROL_ROOT + f"/resume_handle_{sequence}.json"
    with remote.sftp_client(transport) as sftp:
        intent, handle = _remote_json(sftp, intent_path), _remote_json(sftp, handle_path)
    matches = _query_matches(transport, job_name)
    if len(matches) > 1:
        raise RuntimeError("More than one matching resume job exists; reconcile without submitting")
    if matches:
        job_id = matches[0]
        if (intent or {}).get("job_id") not in (None, job_id) or (handle or {}).get("job_id") not in (None, job_id):
            raise RuntimeError("Restart PBS history conflicts with its durable intent")
        resumes_job_id = previous_handle["job_id"]
        if (intent or {}).get("resumes_job_id") not in (None, resumes_job_id):
            raise RuntimeError("Restart durable intent points to a different source job")
        recovered = {**(intent or {}), "job_name": job_name, "job_id": job_id,
                     "resume_sequence": sequence, "resumes_job_id": resumes_job_id,
                     "status": "confirmed" if intent else "recovered"}
        if retry_failed:
            recovered["retry"] = True
        with remote.sftp_client(transport) as sftp:
            _save_remote_json(sftp, intent_path, recovered)
            _save_remote_json(sftp, handle_path, recovered)
            _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", recovered)
        return recovered
    if intent or handle:
        raise RuntimeError("A restart intent exists without a recoverable PBS job; no second qsub will be issued")

    previous_status = scheduler(transport, previous_handle["job_id"])
    summary = previous_status.get("summary.json") or {}
    if retry_failed:
        try:
            pbs_failed = int(previous_status.get("Exit_status")) != 0
            artifact_failed = int(previous_status.get("exit_code.txt")) != 0
        except (TypeError, ValueError):
            pbs_failed = artifact_failed = False
        if (previous_status.get("query_rc") != 0
                or previous_status.get("job_state") not in ("F", "X")
                or not pbs_failed or not artifact_failed
                or previous_status.get("PIPELINE_COMPLETE") is True):
            raise RuntimeError("Retry requires an authoritative terminal PBS and artifact failure without PIPELINE_COMPLETE")
    elif (previous_status.get("job_state") not in ("F", "X")
            or previous_status.get("Exit_status") != "0"
            or previous_status.get("exit_code.txt") != "0"
            or previous_status.get("PIPELINE_CHECKPOINTED") is not True
            or previous_status.get("PIPELINE_COMPLETE") is not False
            or summary.get("status") != "checkpointed"
            or summary.get("complete") is not False):
        raise RuntimeError("Resume requires a terminal zero-exit checkpointed pipeline")
    _require_uploaded(transport)
    next_handle = {**previous_handle, "resume_sequence": sequence}
    return _submit(
        transport, job_name, intent_path, handle_path, next_handle, retry=retry_failed,
        script=previous_handle.get("script", EVALUATION_SCRIPT),
        resources={key: previous_handle[key] for key in
                   ("queue", "select", "gpus", "ncpus", "memory_gb", "walltime")
                   if key in previous_handle},
        stage=previous_handle.get("stage", "evaluation"),
        update_latest=True,
    )


def resume(transport) -> dict:
    return _restart(transport, retry_failed=False)


def retry(transport) -> dict:
    return _restart(transport, retry_failed=True)


def _lane_record_paths(stage: str, arm: str | None = None) -> tuple[str, str]:
    if stage == "training":
        if arm not in TRAINING_JOB_NAMES:
            raise ValueError("Training arm must be rho0.1 or rho1.5")
        return (CONTROL_ROOT + f"/train_{arm}_intent.json",
                CONTROL_ROOT + f"/train_{arm}_handle.json")
    if stage == "validation":
        if arm not in VALIDATION_JOB_NAMES:
            raise ValueError("Validation arm must be rho0.1 or rho1.5")
        return (CONTROL_ROOT + f"/validate_{arm}_intent.json",
                CONTROL_ROOT + f"/validate_{arm}_handle.json")
    if stage == "evaluation":
        return CONTROL_ROOT + "/eval_intent.json", CONTROL_ROOT + "/eval_handle.json"
    raise ValueError(f"Unsupported submission stage: {stage}")


def _resolve_lane(transport, *, job_name: str, intent_path: str, handle_path: str,
                  stage: str, arm: str | None = None, update_latest: bool = False,
                  source_training_job_id: str | None = None) -> dict | None:
    with remote.sftp_client(transport) as sftp:
        intent, handle = _remote_json(sftp, intent_path), _remote_json(sftp, handle_path)
    if handle and HANDLE.fullmatch(str(handle.get("job_id", ""))):
        if (handle.get("job_name") not in (None, job_name)
                or handle.get("stage") not in (None, stage)
                or (arm and handle.get("arm") not in (None, arm))
                or (intent and (intent.get("job_id") not in (None, handle["job_id"])
                                or intent.get("job_name") not in (None, job_name)
                                or intent.get("stage") not in (None, stage)
                                or (arm and intent.get("arm") not in (None, arm))))):
            raise RuntimeError(f"Durable {stage} handle conflicts with its intent")
        if update_latest:
            with remote.sftp_client(transport) as sftp:
                _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", handle)
        return handle
    matches = _query_matches(transport, job_name)
    if len(matches) > 1:
        raise RuntimeError(f"More than one matching {stage} PBS job exists; reconcile manually")
    if not matches:
        if intent or handle:
            raise RuntimeError(f"A durable {stage} intent exists without a recoverable PBS job; no second qsub will be issued")
        return None

    job_id = matches[0]
    for record in (intent, handle):
        if not record:
            continue
        if record.get("job_id") not in (None, job_id):
            raise RuntimeError(f"PBS history conflicts with the durable {stage} handle")
        if record.get("job_name") not in (None, job_name):
            raise RuntimeError(f"Durable {stage} record has a different PBS job name")
        if record.get("stage") not in (None, stage):
            raise RuntimeError(f"Durable record belongs to a different stage")
        if arm and record.get("arm") not in (None, arm):
            raise RuntimeError("Durable training record belongs to a different arm")
    recovered = {**(intent or handle or {}), "job_name": job_name, "job_id": job_id,
                 "stage": stage, "status": "confirmed" if intent else "recovered"}
    if stage == "training":
        recovered["arm"] = arm
        recovered["script"] = TRAINING_SCRIPT
        recovered.update(TRAINING_RESOURCES)
    elif stage == "validation":
        recovered.update({"arm": arm, "script": VALIDATION_SCRIPT,
                          "purpose": "validation_only", **VALIDATION_RESOURCES})
        if source_training_job_id:
            recovered["source_training_job_id"] = source_training_job_id
    elif stage == "evaluation":
        recovered["script"] = EVALUATION_SCRIPT
        recovered["restart_job_prefix"] = EVALUATION_JOB_NAME
        recovered.update(EVALUATION_RESOURCES)
    with remote.sftp_client(transport) as sftp:
        _save_remote_json(sftp, intent_path, recovered)
        _save_remote_json(sftp, handle_path, recovered)
        if update_latest:
            _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", recovered)
    return recovered


def _valid_validation_handle(handle: dict | None, arm: str) -> bool:
    return bool(handle
                and handle.get("status") in ("confirmed", "recovered")
                and handle.get("stage") == "validation"
                and handle.get("arm") == arm
                and handle.get("purpose") == "validation_only"
                and handle.get("job_name") == VALIDATION_JOB_NAMES[arm]
                and handle.get("script") == VALIDATION_SCRIPT
                and handle.get("queue") == VALIDATION_RESOURCES["queue"]
                and handle.get("select") == VALIDATION_RESOURCES["select"]
                and all(handle.get(key) == VALIDATION_RESOURCES[key]
                        for key in ("gpus", "ncpus", "memory_gb", "walltime"))
                and HANDLE.fullmatch(str(handle.get("source_training_job_id", ""))))


def _valid_gpu_training_handle(handle: dict | None, arm: str) -> bool:
    return bool(handle
                and HANDLE.fullmatch(str(handle.get("job_id", "")))
                and handle.get("status") in ("confirmed", "recovered")
                and handle.get("stage") == "training"
                and handle.get("arm") == arm
                and handle.get("job_name") == TRAINING_JOB_NAMES[arm]
                and handle.get("script") == TRAINING_SCRIPT
                and all(handle.get(key) == TRAINING_RESOURCES[key]
                        for key in ("queue", "select", "gpus", "ncpus", "memory_gb", "walltime")))


def _validate_metadata_failure_source(transport, arm: str, gpu_handle: dict) -> dict:
    if not _valid_gpu_training_handle(gpu_handle, arm):
        raise RuntimeError(f"{arm} CPU metadata validation is not linked to its approved GPU training handle")
    status = scheduler(transport, gpu_handle["job_id"], stage="training", arm=arm)
    try:
        pbs_failed = int(status.get("Exit_status")) != 0
        artifact_failed = int(status.get("exit_code.txt")) != 0
    except (TypeError, ValueError):
        pbs_failed = artifact_failed = False
    summary = status.get("training_summary.json") or {}
    if (status.get("query_rc") != 0 or status.get("job_state") not in ("F", "X")
            or status.get("Job_Name") != TRAINING_JOB_NAMES[arm]
            or not pbs_failed or not artifact_failed
            or status.get("TRAINING_COMPLETE") is not False
            or status.get("TRAINING_CHECKPOINTED") is not False
            or summary.get("schema_version") != 1
            or summary.get("protocol") != PROTOCOL["protocol"]
            or summary.get("pbs_jobid") != gpu_handle["job_id"]
            or summary.get("job_name") != TRAINING_JOB_NAMES[arm]
            or summary.get("arm") != arm
            or summary.get("status") != "failed" or summary.get("complete") is not False
            or summary.get("failed_phase") != "complete_validation"):
        raise RuntimeError(f"{arm} is not an eligible complete_validation failure for CPU metadata validation")
    _validate_complete_training_receipt(arm, status.get("training_receipt.json"))
    _require_training_artifacts(transport, arm)
    return status


def _promote_validation_handle(transport, arm: str, handle: dict) -> None:
    if not _valid_validation_handle(handle, arm):
        raise RuntimeError("Validation handle does not match the approved CPU-only role")
    _, canonical_handle_path = _lane_record_paths("training", arm)
    canonical_intent_path, _ = _lane_record_paths("training", arm)
    archive_handle_path = CONTROL_ROOT + f"/train_{arm}_gpu_handle.json"
    archive_intent_path = CONTROL_ROOT + f"/train_{arm}_gpu_intent.json"
    with remote.sftp_client(transport) as sftp:
        old_handle = _remote_json(sftp, canonical_handle_path)
        old_intent = _remote_json(sftp, canonical_intent_path)
        archived = _remote_json(sftp, archive_handle_path)
        if old_handle and old_handle.get("purpose") != "validation_only":
            if (old_handle.get("stage") != "training" or old_handle.get("arm") != arm
                    or old_handle.get("script") != TRAINING_SCRIPT
                    or old_handle.get("job_name") != TRAINING_JOB_NAMES[arm]
                    or old_handle.get("job_id") != handle["source_training_job_id"]):
                raise RuntimeError("Canonical training handle differs from the validation source job")
            if archived and archived.get("job_id") != old_handle.get("job_id"):
                raise RuntimeError("GPU training archive already points to a different job")
            _save_remote_json(sftp, archive_handle_path, old_handle)
            if old_intent:
                _save_remote_json(sftp, archive_intent_path, old_intent)
        canonical = dict(handle)
        _save_remote_json(sftp, canonical_intent_path, canonical)
        _save_remote_json(sftp, canonical_handle_path, canonical)


def _current_training_handle(transport, arm: str) -> dict | None:
    train_intent_path, train_handle_path = _lane_record_paths("training", arm)
    validate_intent_path, validate_handle_path = _lane_record_paths("validation", arm)
    with remote.sftp_client(transport) as sftp:
        train_handle = _remote_json(sftp, train_handle_path)
        validate_intent = _remote_json(sftp, validate_intent_path)
        validate_handle = _remote_json(sftp, validate_handle_path)
        archived = _remote_json(sftp, CONTROL_ROOT + f"/train_{arm}_gpu_handle.json")
    if validate_intent or validate_handle or (train_handle and train_handle.get("purpose") == "validation_only"):
        source_job_id = ((train_handle or {}).get("source_training_job_id")
                         or (validate_handle or {}).get("source_training_job_id")
                         or (validate_intent or {}).get("source_training_job_id")
                         or (archived or {}).get("job_id"))
        handle = _resolve_lane(
            transport, job_name=VALIDATION_JOB_NAMES[arm],
            intent_path=validate_intent_path, handle_path=validate_handle_path,
            stage="validation", arm=arm, source_training_job_id=source_job_id,
        )
        if not handle and train_handle and train_handle.get("purpose") == "validation_only":
            handle = train_handle
        if handle:
            _promote_validation_handle(transport, arm, handle)
        return handle
    return _resolve_lane(transport, job_name=TRAINING_JOB_NAMES[arm],
                         intent_path=train_intent_path, handle_path=train_handle_path,
                         stage="training", arm=arm)


def train_submit(transport, arm: str) -> dict:
    if arm not in TRAINING_JOB_NAMES:
        raise ValueError("Training arm must be rho0.1 or rho1.5")
    job_name = TRAINING_JOB_NAMES[arm]
    existing = _current_training_handle(transport, arm)
    if existing:
        status = scheduler(transport, existing["job_id"], stage="training", arm=arm)
        if (status.get("query_rc") != 0
                or status.get("Job_Name") not in (None, existing.get("job_name"))):
            raise RuntimeError("Existing training handle could not be queried; no second qsub was attempted")
        return {"handle": existing, "scheduler": status}
    intent_path, handle_path = _lane_record_paths("training", arm)
    _require_uploaded(transport)
    return _submit(transport, job_name, intent_path, handle_path,
                   script=TRAINING_SCRIPT, resources=TRAINING_RESOURCES,
                   stage="training", arm=arm, env={"FW_TRAIN_ARM": arm},
                   update_latest=False)


def train_status(transport, arm: str) -> dict:
    if arm not in TRAINING_JOB_NAMES:
        raise ValueError("Training arm must be rho0.1 or rho1.5")
    saved = _current_training_handle(transport, arm)
    if not saved:
        raise RuntimeError(f"No training handle is available for {arm}")
    return scheduler(transport, saved["job_id"], stage="training", arm=arm)


def _remote_file_exists(sftp, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except OSError as exc:
        if getattr(exc, "errno", None) == 2:
            return False
        raise


def _compact_training_receipt(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    keys = ("schema_version", "protocol", "status", "complete", "arm", "rho", "epochs",
            "updates", "epochs_completed", "target_linears", "case_ids", "source_c8",
            "normalization_c", "optimizer_reset_after_update1", "gradient_coverage", "counts")
    return {key: value[key] for key in keys if key in value}


def _validate_complete_training_receipt(arm: str, receipt: dict | None) -> float:
    config = PROTOCOL["arm_configs"][arm]
    case_ids = PROTOCOL["calibration"]["case_ids"]
    receipt = receipt or {}
    source = receipt.get("source_c8") or {}
    counts = receipt.get("counts") or {}
    if (receipt.get("schema_version") != 1 or receipt.get("protocol") != PROTOCOL["protocol"]
            or receipt.get("status") != "complete" or receipt.get("complete") is not True
            or receipt.get("arm") != arm or receipt.get("epochs") != config["epochs"]
            or receipt.get("rho") != config["rho"]
            or receipt.get("epochs_completed") != config["epochs"]
            or receipt.get("updates") != config["updates"]
            or receipt.get("target_linears") != 614 or receipt.get("case_ids") != case_ids
            or receipt.get("optimizer_reset_after_update1") is not True
            or source.get("protocol") != "fastwam-vq-calibration-suites-v1"
            or source.get("cohort") != "c8"
            or source.get("source_job_id") != PROTOCOL["calibration"]["source_job_id"]
            or source.get("case_ids") != case_ids
            or counts.get("gradient_updates") != config["updates"]
            or counts.get("reference_contexts") != 8):
        raise RuntimeError(f"{arm} has no exact completed training receipt")
    c = receipt.get("normalization_c")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0:
        raise RuntimeError(f"{arm} training receipt has no valid normalization_c")
    return c


def _require_training_artifacts(transport, arm: str) -> None:
    with remote.sftp_client(transport) as sftp:
        for filename in ("deployment.pt", "reference_action.json"):
            path = ROOT + f"/results/training/{arm}/{filename}"
            if not _remote_file_exists(sftp, path):
                raise RuntimeError(f"{arm} is missing required artifact {filename}")


def _validate_training_lane(transport, arm: str) -> dict:
    handle = _current_training_handle(transport, arm)
    is_validation = bool(handle and handle.get("purpose") == "validation_only")
    if is_validation:
        if not _valid_validation_handle(handle, arm):
            raise RuntimeError(f"{arm} validation handle is outside the approved CPU-only role")
        expected_job_name = VALIDATION_JOB_NAMES[arm]
        with remote.sftp_client(transport) as sftp:
            source_handle = _remote_json(sftp, CONTROL_ROOT + f"/train_{arm}_gpu_handle.json")
        if (not source_handle or source_handle.get("job_id") != handle.get("source_training_job_id")
                or not _valid_gpu_training_handle(source_handle, arm)):
            raise RuntimeError(f"{arm} CPU validation is not linked to its archived GPU training job")
        _validate_metadata_failure_source(transport, arm, source_handle)
    else:
        expected_job_name = TRAINING_JOB_NAMES[arm]
        if (not HANDLE.fullmatch(str((handle or {}).get("job_id", "")))
                or not _valid_gpu_training_handle(handle, arm)):
            raise RuntimeError(f"{arm} does not have an approved training handle")
    status = scheduler(transport, handle["job_id"], stage="training", arm=arm)
    if (status.get("query_rc") != 0 or status.get("job_state") not in ("F", "X")
            or status.get("Job_Name") != expected_job_name
            or status.get("Exit_status") != "0" or status.get("exit_code.txt") != "0"
            or status.get("TRAINING_COMPLETE") is not True
            or status.get("TRAINING_CHECKPOINTED") is not False):
        raise RuntimeError(f"{arm} training is not authoritatively complete; evaluation was not submitted")
    summary = status.get("training_summary.json") or {}
    receipt = status.get("training_receipt.json") or {}
    config = PROTOCOL["arm_configs"][arm]
    if (summary.get("schema_version") != 1 or summary.get("protocol") != PROTOCOL["protocol"]
            or summary.get("pbs_jobid") != handle["job_id"]
            or summary.get("job_name") != expected_job_name or summary.get("arm") != arm
            or summary.get("rho") != config["rho"] or summary.get("status") != "complete"
            or summary.get("complete") is not True
            or summary.get("updates") != config["updates"]
            or summary.get("epochs_completed") != config["epochs"]
            or summary.get("source_job_id") != PROTOCOL["calibration"]["source_job_id"]):
        raise RuntimeError(f"{arm} training receipt or artifact summary fails the frozen completion gate")
    if is_validation:
        if (summary.get("purpose") != "validation_only"
                or summary.get("source_training_job_id") != handle.get("source_training_job_id")
                or summary.get("new_model_forwards") != 0
                or handle.get("purpose") != "validation_only"):
            raise RuntimeError(f"{arm} CPU validation summary has the wrong provenance")
    c = _validate_complete_training_receipt(arm, receipt)
    _require_training_artifacts(transport, arm)
    return {"arm": arm, "normalization_c": c, "receipt": receipt}


def training_validate(transport, arm: str) -> dict:
    if arm not in TRAINING_JOB_NAMES:
        raise ValueError("Training arm must be rho0.1 or rho1.5")
    validate_intent_path, validate_handle_path = _lane_record_paths("validation", arm)
    _, train_handle_path = _lane_record_paths("training", arm)
    with remote.sftp_client(transport) as sftp:
        train_handle = _remote_json(sftp, train_handle_path)
        archived = _remote_json(sftp, CONTROL_ROOT + f"/train_{arm}_gpu_handle.json")
    source_training_job_id = ((train_handle or {}).get("source_training_job_id")
                              or (train_handle or {}).get("job_id")
                              or (archived or {}).get("job_id"))
    existing = _resolve_lane(
        transport, job_name=VALIDATION_JOB_NAMES[arm],
        intent_path=validate_intent_path, handle_path=validate_handle_path,
        stage="validation", arm=arm, source_training_job_id=source_training_job_id,
    )
    if existing:
        status = scheduler(transport, existing["job_id"], stage="training", arm=arm)
        if (status.get("query_rc") != 0
                or status.get("Job_Name") not in (None, VALIDATION_JOB_NAMES[arm])):
            raise RuntimeError("Existing CPU validation handle could not be queried; no second qsub was attempted")
        _promote_validation_handle(transport, arm, existing)
        return {"handle": existing, "scheduler": status}

    train_intent_path, train_handle_path = _lane_record_paths("training", arm)
    gpu_handle = _resolve_lane(
        transport, job_name=TRAINING_JOB_NAMES[arm],
        intent_path=train_intent_path, handle_path=train_handle_path,
        stage="training", arm=arm,
    )
    if not _valid_gpu_training_handle(gpu_handle, arm):
        raise RuntimeError("CPU validation requires the confirmed GPU training handle for this arm")
    _validate_metadata_failure_source(transport, arm, gpu_handle)
    _require_uploaded(transport)
    validated = _submit(
        transport, VALIDATION_JOB_NAMES[arm], validate_intent_path, validate_handle_path,
        script=VALIDATION_SCRIPT, resources=VALIDATION_RESOURCES,
        stage="validation", arm=arm,
        env={"FW_VALIDATE_ARM": arm, "FW_SOURCE_TRAIN_JOB_ID": gpu_handle["job_id"]},
        update_latest=False, purpose="validation_only",
        source_training_job_id=gpu_handle["job_id"],
    )
    _promote_validation_handle(transport, arm, validated)
    return validated


def evaluation_submit(transport) -> dict:
    intent_path, handle_path = _lane_record_paths("evaluation")
    existing = _resolve_lane(transport, job_name=EVALUATION_JOB_NAME,
                             intent_path=intent_path, handle_path=handle_path,
                             stage="evaluation", update_latest=True)
    if existing:
        status = scheduler(transport, existing["job_id"])
        if status.get("query_rc") != 0 or status.get("Job_Name") not in (None, EVALUATION_JOB_NAME):
            raise RuntimeError("Existing evaluation handle could not be queried; no second qsub was attempted")
        return {"handle": existing, "scheduler": status}

    _require_uploaded(transport)
    receipts = [_validate_training_lane(transport, arm) for arm in TRAINING_JOB_NAMES]
    if receipts[0]["normalization_c"] != receipts[1]["normalization_c"]:
        raise RuntimeError("Training arms do not share the same normalization_c; evaluation was not submitted")
    return _submit(transport, EVALUATION_JOB_NAME, intent_path, handle_path,
                   script=EVALUATION_SCRIPT, resources=EVALUATION_RESOURCES,
                   stage="evaluation", update_latest=True)


def _compact_summary(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    keys = ("protocol", "status", "complete", "phase", "episode_slots",
            "reference_episode_slots", "arm_episodes", "new_arms",
            "checkpointed_workers", "completed_training_arms",
            "next_training_arm", "training_updates", "normalization_c")
    return {key: value[key] for key in keys if key in value}


def scheduler(transport, job_id: str, *, stage: str | None = None, arm: str | None = None) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS handle")
    code, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=35)
    fields = {"job_id": job_id, "query_rc": code}
    for key in (
        "job_state", "Exit_status", "Job_Name", "queue", "exec_host",
        "resources_used.walltime", "Resource_List.walltime", "Resource_List.ngpus",
        "Resource_List.ncpus", "Resource_List.mem", "Resource_List.select",
    ):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    artifact_root = ROOT + "/artifacts/" + job_id
    with remote.sftp_client(transport) as sftp:
        fields["RUN_PHASE.json"] = _remote_json(sftp, artifact_root + "/RUN_PHASE.json")
        if fields["job_state"] in ("F", "X"):
            fields["exit_code.txt"] = _remote_text(sftp, artifact_root + "/exit_code.txt")
            fields["summary.json"] = _compact_summary(_remote_json(sftp, artifact_root + "/summary.json"))
            for marker in ("PIPELINE_COMPLETE", "PIPELINE_CHECKPOINTED"):
                try:
                    sftp.stat(artifact_root + "/" + marker)
                    fields[marker] = True
                except OSError as exc:
                    if getattr(exc, "errno", None) != 2:
                        raise
                    fields[marker] = False
        if stage == "training":
            fields["training_summary.json"] = _remote_json(sftp, artifact_root + "/training_summary.json")
            for marker in ("TRAINING_COMPLETE", "TRAINING_CHECKPOINTED"):
                fields[marker] = _remote_file_exists(sftp, artifact_root + "/" + marker)
            if arm in TRAINING_JOB_NAMES:
                receipt_path = ROOT + f"/results/training/{arm}/training_receipt.json"
                fields["training_receipt.json"] = _compact_training_receipt(_remote_json(sftp, receipt_path))
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "resume", "retry", "status",
                                             "train-submit", "train-status", "training-validate",
                                             "evaluation-submit"))
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--job")
    parser.add_argument("--arm", choices=tuple(TRAINING_JOB_NAMES))
    args = parser.parse_args()
    if args.action != "upload" and args.files:
        parser.error("--files is only supported for upload")
    if args.action != "status" and args.job:
        parser.error("--job is only supported for status")
    if args.action in ("train-submit", "train-status", "training-validate") and not args.arm:
        parser.error("--arm is required for training actions")
    if args.action not in ("train-submit", "train-status", "training-validate") and args.arm:
        parser.error("--arm is only supported for training actions")
    jump, transport = remote.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": upload(transport, args.files)}
        elif args.action == "submit":
            result = submit(transport)
        elif args.action == "resume":
            result = resume(transport)
        elif args.action == "retry":
            result = retry(transport)
        elif args.action == "train-submit":
            result = train_submit(transport, args.arm)
        elif args.action == "train-status":
            result = train_status(transport, args.arm)
        elif args.action == "training-validate":
            result = training_validate(transport, args.arm)
        elif args.action == "evaluation-submit":
            result = evaluation_submit(transport)
        else:
            job_id = args.job
            if not job_id:
                with remote.sftp_client(transport) as sftp:
                    saved = (_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json")
                             or _remote_json(sftp, CONTROL_ROOT + "/submit_handle.json"))
                if not saved or not saved.get("job_id"):
                    raise RuntimeError("No saved PBS handle is available; provide --job")
                job_id = saved["job_id"]
            result = scheduler(transport, job_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
