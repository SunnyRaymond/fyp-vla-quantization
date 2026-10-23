"""Small-file PBS control for C80 clip_only/joint calibration and evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex


HERE = Path(__file__).resolve().parent
C8_CONTROL = HERE.parent / "fastwam-vq-c8-action-latent-plus" / "control.py"
_spec = importlib.util.spec_from_file_location("c80_clip_joint_c8_control", C8_CONTROL)
_c8 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_c8)
remote = _c8.remote

PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = PROTOCOL["remote_root"]
CONTROL = ROOT + "/control"
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"
USER = "yguo017"
ARMS = ("clip_only", "joint")
EVAL_UNIT = {"clip_only": "clip_only_eval", "joint": "joint_eval"}
UNITS = (*ARMS, *EVAL_UNIT.values())
LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
JOB_NAME = re.compile(r"^c80(?:clip|joint|eclip|ejoint)[0-9]{4}$")
BACKWARD_IMPLEMENTATION = "bounded_vector_chunks_v2"
BACKWARD_MAX_VECTORS = 1048576
FILES = (
    "protocol.json", "calibrate.py", "calibrate.pbs", "backward_check.py", "run.pbs",
    "vq_adapter.py", "evaluate.py", "aggregate.py", "batched_transform.py",
    "transform_checks.py", "environment.sh",
)
CAL_FILES = ("protocol.json", "calibrate.py", "calibrate.pbs", "backward_check.py")
EVAL_FILES = tuple(name for name in FILES if name not in ("calibrate.py", "calibrate.pbs"))
CAL_RESOURCES = {
    "queue": "normal", "select": "1:ncpus=16:mem=110gb:ngpus=1",
    "gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "01:59:00",
}
EVAL_RESOURCES = {
    "queue": "normal", "select": "1:ncpus=32:mem=220gb:ngpus=4",
    "gpus": 4, "ncpus": 32, "memory_gb": 220, "walltime": "24:00:00",
}


def _absent(exc: OSError) -> bool:
    return getattr(exc, "errno", None) == 2


def _bytes(sftp, path: str) -> bytes | None:
    try:
        with sftp.file(path, "rb") as stream:
            value = stream.read(LIMIT + 1)
    except OSError as exc:
        if _absent(exc):
            return None
        raise
    if len(value) > LIMIT:
        raise RuntimeError(f"Remote file exceeds 256 KiB: {path}")
    return value


def _json(sftp, path: str):
    value = _bytes(sftp, path)
    if value is None:
        return None
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid remote JSON: {path}") from exc


def _text(sftp, path: str) -> str | None:
    value = _bytes(sftp, path)
    return value.decode("utf-8", errors="replace").strip() if value is not None else None


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
    paths = (CONTROL, ARTIFACTS, RESULTS, RESULTS + "/training/clip_only",
             RESULTS + "/training/joint", RESULTS + "/evaluations/clip_only",
             RESULTS + "/evaluations/joint")
    code, output = remote.command(
        transport, "mkdir -p " + " ".join(shlex.quote(path) for path in paths), timeout=25)
    if code:
        raise RuntimeError("Cannot prepare remote control directories: " + output[:1000])


def _intent_path(unit: str) -> str:
    if unit not in UNITS:
        raise ValueError("Unknown C80 unit")
    return f"{CONTROL}/{unit}_intent_attempt.json"


def _records(sftp) -> dict:
    value = _json(sftp, CONTROL + "/handles.json") or {"schema_version": 1, "units": {}}
    if not isinstance(value, dict) or not isinstance(value.get("units"), dict):
        raise RuntimeError("Invalid remote handles.json")
    return value


def _state(records: dict, unit: str) -> dict:
    value = records["units"].get(unit) or {"attempts": [], "latest_attempt": None}
    if not isinstance(value, dict) or not isinstance(value.get("attempts", []), list):
        raise RuntimeError(f"Invalid handle history for {unit}")
    return value


def _latest(records: dict, unit: str) -> dict | None:
    state = _state(records, unit)
    attempt = state.get("latest_attempt")
    if attempt is None:
        return None
    found = [row for row in state["attempts"] if row.get("attempt") == attempt]
    if len(found) != 1:
        raise RuntimeError(f"No unique latest handle for {unit}")
    return found[0]


def _arm_for_unit(unit: str) -> str | None:
    if unit in ARMS:
        return unit
    for arm, name in EVAL_UNIT.items():
        if unit == name:
            return arm
    raise ValueError("Unknown C80 unit")


def _job_name(unit: str, attempt: int) -> str:
    tag = {"clip_only": "clip", "joint": "joint",
           "clip_only_eval": "eclip", "joint_eval": "ejoint"}[unit]
    name = f"c80{tag}{attempt:04d}"
    if len(name) > 15 or not JOB_NAME.fullmatch(name):
        raise ValueError("C80 attempt exceeds PBS job-name limit")
    return name


def _query_name(transport, name: str) -> list[str]:
    if len(name) > 15 or not JOB_NAME.fullmatch(name):
        raise ValueError("Invalid C80 PBS job name")
    code, output = remote.command(
        transport, "qselect -x -u " + USER + " -N " + shlex.quote(name) + " 2>&1", timeout=35)
    if code:
        raise RuntimeError("PBS exact-name history query failed; no submission was attempted")
    result = output.split()
    if any(not HANDLE.fullmatch(job) for job in result):
        raise RuntimeError("PBS returned an invalid exact-name handle")
    return result


def _save_handle(sftp, records: dict, unit: str, handle: dict) -> None:
    state = _state(records, unit)
    rows = state["attempts"]
    old = [row for row in rows if row.get("attempt") == handle["attempt"]]
    if old and old[0].get("job_id") not in (None, handle.get("job_id")):
        raise RuntimeError("PBS job conflicts with durable handle history")
    rows[:] = [row for row in rows if row.get("attempt") != handle["attempt"]]
    rows.append(handle)
    rows.sort(key=lambda row: int(row["attempt"]))
    state["latest_attempt"] = handle["attempt"]
    records["units"][unit] = state
    _write_json(sftp, CONTROL + "/handles.json", records)


def _recover(transport, unit: str, intent: dict, records: dict) -> dict | None:
    if intent.get("unit") != unit or type(intent.get("attempt")) is not int:
        raise RuntimeError("Durable attempt intent has invalid unit/attempt")
    matches = _query_name(transport, intent.get("job_name", ""))
    if len(matches) > 1:
        raise RuntimeError("Multiple PBS jobs share this exact attempt name; reconcile manually")
    if not matches:
        return None
    job_id = matches[0]
    observed = scheduler(transport, job_id, unit=unit)
    if observed.get("query_rc") != 0 or observed.get("Job_Name") != intent["job_name"]:
        raise RuntimeError("Exact-name PBS handle is not verified; intent remains unresolved")
    handle = dict(intent, job_id=job_id,
                  status="confirmed" if intent.get("status") == "confirmed" else "recovered")
    with remote.sftp_client(transport) as sftp:
        _write_json(sftp, _intent_path(unit), handle)
        _save_handle(sftp, records, unit, handle)
    return handle


def _resolve(transport, unit: str, records: dict, intent: dict | None) -> dict | None:
    latest = _latest(records, unit)
    if intent:
        if intent.get("unit") != unit or type(intent.get("attempt")) is not int:
            raise RuntimeError("Durable attempt intent has invalid unit/attempt")
        if latest and intent["attempt"] < latest["attempt"]:
            return latest
        if latest and intent["attempt"] == latest["attempt"]:
            if intent.get("job_id") not in (None, latest.get("job_id")):
                raise RuntimeError("Latest intent differs from handles.json")
            return latest
        recovered = _recover(transport, unit, intent, records)
        if recovered:
            return recovered
        raise RuntimeError("Durable intent has no exact-name PBS handle; no duplicate is allowed")
    if latest and not HANDLE.fullmatch(str(latest.get("job_id", ""))):
        raise RuntimeError("Latest handle is invalid")
    return latest


def _protocol_matches(sftp) -> bool:
    return _json(sftp, ROOT + "/protocol.json") == PROTOCOL


def upload(transport, names: list[str] | None = None) -> list[str]:
    selected = list((name for name in FILES if (HERE / name).is_file()) if names is None else names)
    if len(set(selected)) != len(selected) or any(name not in FILES for name in selected):
        raise ValueError("Upload accepts only the approved small C80 source files")
    _mkdirs(transport)
    with remote.sftp_client(transport) as sftp:
        old_protocol = _json(sftp, ROOT + "/protocol.json")
        if old_protocol is not None and old_protocol != PROTOCOL:
            raise RuntimeError("Remote frozen protocol differs and will not be overwritten")
        records = _records(sftp)
        intent_values = {unit: _json(sftp, _intent_path(unit)) for unit in UNITS}
        calibration_frozen = any(_state(records, unit)["attempts"] or intent_values[unit]
                                 for unit in ARMS)
        evaluation_frozen = any(_state(records, unit)["attempts"] or intent_values[unit]
                                for unit in EVAL_UNIT.values())
        uploaded = []
        for name in selected:
            local = HERE / name
            if not local.is_file():
                raise FileNotFoundError(local)
            data = local.read_bytes().replace(b"\r\n", b"\n")
            if len(data) > LIMIT:
                raise ValueError(f"Source exceeds 256 KiB: {name}")
            data.decode("utf-8")
            target = ROOT + "/" + name
            previous = _bytes(sftp, target)
            if name == "protocol.json" and old_protocol is not None:
                continue
            if previous == data:
                continue
            if evaluation_frozen or (calibration_frozen and name in CAL_FILES):
                raise RuntimeError("Submitted C80 source is frozen; remote differs from " + name)
            temp = target + ".tmp"
            with sftp.file(temp, "wb") as stream:
                stream.write(data)
            sftp.posix_rename(temp, target)
            uploaded.append(name)
    return uploaded


def _require_uploaded(transport, names=FILES) -> None:
    with remote.sftp_client(transport) as sftp:
        for name in names:
            if _bytes(sftp, ROOT + "/" + name) is None:
                raise RuntimeError(f"Upload is incomplete; missing {name}")
        if not _protocol_matches(sftp):
            raise RuntimeError("Remote protocol differs from the local frozen protocol")


def scheduler(transport, job_id: str, *, unit: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=35)
    result = {"job_id": job_id, "query_rc": code}
    for key in ("job_state", "Exit_status", "Job_Name", "queue", "exec_host",
                "resources_used.walltime", "Resource_List.walltime", "Resource_List.ngpus",
                "Resource_List.ncpus", "Resource_List.mem", "Resource_List.select"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        result[key] = match.group(1).strip() if match else None
    artifact = ARTIFACTS + "/" + job_id
    arm = _arm_for_unit(unit)
    with remote.sftp_client(transport) as sftp:
        result["RUN_PHASE.json"] = _json(sftp, artifact + "/RUN_PHASE.json")
        if unit in ARMS:
            result["calibration_summary.json"] = _json(sftp, artifact + "/calibration_summary.json")
            result["exit_code.txt"] = _text(sftp, artifact + "/exit_code.txt")
            result["TRAINING_COMPLETE"] = _exists(sftp, artifact + "/TRAINING_COMPLETE")
            result["TRAINING_CHECKPOINTED"] = _exists(sftp, artifact + "/TRAINING_CHECKPOINTED")
            train_root = RESULTS + f"/training/{arm}"
            result["progress.json"] = _json(sftp, train_root + "/progress.json")
            result["training_receipt.json"] = _json(sftp, train_root + "/training_receipt.json")
            result["reference_action.json"] = _json(sftp, train_root + "/reference_action.json")
            result["deployment_exists"] = _exists(sftp, train_root + "/deployment.pt")
            result["backward_check.json"] = _json(sftp, RESULTS + "/backward_check.json")
            progress = result["progress.json"] or {}
            checkpoint = progress.get("checkpoint")
            expected_prefix = train_root + "/"
            safe = (isinstance(checkpoint, str) and checkpoint.startswith(expected_prefix)
                    and ".." not in checkpoint.split("/"))
            result["checkpoint_exists"] = _exists(sftp, checkpoint) if safe else False
        else:
            eval_root = RESULTS + f"/evaluations/{arm}"
            result["exit_code.txt"] = _text(sftp, artifact + "/exit_code.txt")
            result["PIPELINE_COMPLETE"] = _exists(sftp, artifact + "/PIPELINE_COMPLETE")
            result["PIPELINE_CHECKPOINTED"] = _exists(sftp, artifact + "/PIPELINE_CHECKPOINTED")
            result["result_PIPELINE_COMPLETE"] = _json(sftp, eval_root + "/PIPELINE_COMPLETE")
            result["summary.json"] = _json(sftp, eval_root + "/summary.json")
            result["combined_summary.json"] = _json(sftp, RESULTS + "/summary.json")
            result["combined_PIPELINE_COMPLETE"] = _json(sftp, RESULTS + "/PIPELINE_COMPLETE")
            result["workers"] = {
                str(worker): {
                    "progress": _json(sftp, eval_root + f"/worker_{worker}/progress.json"),
                    "summary": _json(sftp, eval_root + f"/worker_{worker}/summary.json"),
                }
                for worker in range(4)
            }
    return result


def _cal_ids(value: dict):
    return value.get("calibration_ids", value.get("case_ids"))


def _verify_resume_source(handle: dict, summary: dict, progress: dict) -> None:
    source = handle.get("source_job_id")
    if source and (summary.get("resume_from") != source or progress.get("resume_from") != source):
        raise RuntimeError("Calibration summary/progress does not preserve its exact resume source")


def _backward_passed(value) -> bool:
    return (isinstance(value, dict) and value.get("passed") is True
            and value.get("implementation") == BACKWARD_IMPLEMENTATION
            and value.get("max_vectors_per_chunk") == BACKWARD_MAX_VECTORS)


def _training_complete(handle: dict, value: dict, arm: str) -> dict:
    if (value.get("query_rc") != 0 or value.get("job_state") not in ("F", "X")
            or value.get("Job_Name") != handle.get("job_name")
            or value.get("Exit_status") != "0" or value.get("exit_code.txt") != "0"
            or value.get("TRAINING_COMPLETE") is not True
            or value.get("TRAINING_CHECKPOINTED") is not False):
        raise RuntimeError(f"{arm} is not terminal PBS/artifact success")
    summary = value.get("calibration_summary.json") or {}
    progress = value.get("progress.json") or {}
    receipt = value.get("training_receipt.json") or {}
    _verify_resume_source(handle, summary, progress)
    if arm == "clip_only" and not _backward_passed(value.get("backward_check.json")):
        raise RuntimeError("clip_only export requires the passed bounded-vector backward check")
    ids = PROTOCOL["calibration"]["case_ids"]
    order = PROTOCOL["calibration"].get("update_order", ids)
    parity = receipt.get("actual_infer_parity") or receipt.get("reference_action_parity") or {}
    max_abs = parity.get("max_abs")
    tolerance = PROTOCOL.get("closed_loop_reference", {}).get("max_abs_tolerance", 1e-6)
    if (summary.get("protocol") != PROTOCOL["protocol"]
            or summary.get("jobid", summary.get("pbs_jobid")) != handle.get("job_id")
            or summary.get("arm") != arm or summary.get("status") != "complete"
            or summary.get("complete") is not True or summary.get("updates") != 80
            or summary.get("target_linears") != 614 or _cal_ids(summary) != ids
            or summary.get("update_order") != order
            or progress.get("arm") != arm or progress.get("status") != "complete"
            or progress.get("complete") is not True or progress.get("updates") != 80
            or receipt.get("protocol") != PROTOCOL["protocol"] or receipt.get("arm") != arm
            or receipt.get("status") != "complete" or receipt.get("complete") is not True
            or receipt.get("updates") != 80 or receipt.get("target_linears") != 614
            or _cal_ids(receipt) != ids or receipt.get("update_order") != order
            or isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
            or not math.isfinite(max_abs) or max_abs > tolerance
            or not value.get("deployment_exists")
            or not isinstance(value.get("reference_action.json"), dict)):
        raise RuntimeError(f"{arm} training artifact fails exact 80/C80/614/parity gate")
    return {"arm": arm, "job_id": handle["job_id"], "updates": 80,
            "target_linears": 614, "case_ids": ids, "parity_max_abs": max_abs}


def _training_checkpointed(handle: dict, value: dict, arm: str) -> dict:
    if (value.get("query_rc") != 0 or value.get("job_state") not in ("F", "X")
            or value.get("Job_Name") != handle.get("job_name")
            or value.get("Exit_status") != "75" or value.get("exit_code.txt") != "75"
            or value.get("TRAINING_CHECKPOINTED") is not True
            or value.get("TRAINING_COMPLETE") is not False):
        raise RuntimeError("Resume requires F/X, PBS rc=75, artifact rc=75, checkpoint marker only")
    summary = value.get("calibration_summary.json") or {}
    progress = value.get("progress.json") or {}
    _verify_resume_source(handle, summary, progress)
    ids = PROTOCOL["calibration"]["case_ids"]
    order = PROTOCOL["calibration"].get("update_order", ids)
    updates = progress.get("updates")
    if (summary.get("protocol") != PROTOCOL["protocol"]
            or summary.get("jobid", summary.get("pbs_jobid")) != handle.get("job_id")
            or summary.get("arm") != arm or summary.get("status") != "checkpointed"
            or summary.get("complete") is not False or summary.get("updates") != updates
            or summary.get("target_linears") != 614 or _cal_ids(summary) != ids
            or summary.get("update_order") != order
            or progress.get("arm") != arm or progress.get("status") != "checkpointed"
            or progress.get("complete") is not False or type(updates) is not int
            or not 0 <= updates <= 80 or not value.get("checkpoint_exists")):
        raise RuntimeError("Checkpoint/progress does not prove an exact resumable C80 joint update")
    return {"updates": updates, "checkpoint": progress.get("checkpoint")}


def _eval_complete(handle: dict, value: dict, arm: str) -> dict:
    if (value.get("query_rc") != 0 or value.get("job_state") not in ("F", "X")
            or value.get("Job_Name") != handle.get("job_name")
            or value.get("Exit_status") != "0" or value.get("exit_code.txt") != "0"
            or value.get("PIPELINE_COMPLETE") is not True
            or value.get("PIPELINE_CHECKPOINTED") is not False):
        raise RuntimeError(f"{arm} evaluation is not terminal PBS/artifact success")
    summary = value.get("summary.json") or {}
    marker = value.get("result_PIPELINE_COMPLETE") or {}
    if (summary.get("protocol") != PROTOCOL["protocol"] or summary.get("arm") != arm
            or summary.get("status") != "complete" or summary.get("complete") is not True
            or summary.get("episode_slots") != 280
            or summary.get("reference_episode_slots") != 2240
            or summary.get("arm_episodes") != {arm: 280}
            or marker.get("protocol") != PROTOCOL["protocol"] or marker.get("arm") != arm
            or marker.get("complete") is not True or marker.get("episode_slots") != 280
            or marker.get("reference_episode_slots") != 2240):
        raise RuntimeError(f"{arm} evaluation summary fails 280/2240 gate")
    combined = value.get("combined_summary.json")
    if combined is not None:
        combined_marker = value.get("combined_PIPELINE_COMPLETE") or {}
        if (combined.get("protocol") != PROTOCOL["protocol"]
                or combined.get("status") != "complete" or combined.get("complete") is not True
                or combined.get("episode_slots") != 560
                or combined.get("arm_episodes") != {"clip_only": 280, "joint": 280}
                or combined.get("reference_episode_slots") != 2240
                or combined_marker.get("protocol") != PROTOCOL["protocol"]
                or combined_marker.get("complete") is not True
                or combined_marker.get("episode_slots") != 560
                or combined_marker.get("arm_episodes") != {"clip_only": 280, "joint": 280}
                or combined_marker.get("reference_episode_slots") != 2240):
            raise RuntimeError("Combined C80 result fails the exact 560/2240 completion gate")
    return summary


def _eval_checkpointed(handle: dict, value: dict, arm: str) -> dict:
    if (value.get("query_rc") != 0 or value.get("job_state") not in ("F", "X")
            or value.get("Job_Name") != handle.get("job_name")
            or value.get("Exit_status") != "0" or value.get("exit_code.txt") != "0"
            or value.get("PIPELINE_CHECKPOINTED") is not True
            or value.get("PIPELINE_COMPLETE") is not False):
        raise RuntimeError("Evaluation resume requires a terminal exit0 checkpointed PBS artifact")
    summary = value.get("summary.json") or {}
    if (summary.get("protocol") != PROTOCOL["protocol"] or summary.get("arm") != arm
            or summary.get("status") != "checkpointed" or summary.get("complete") is not False
            or type(summary.get("episode_slots")) is not int
            or not 0 <= summary["episode_slots"] < 280
            or summary.get("reference_episode_slots") != 2240):
        raise RuntimeError("Evaluation aggregate is not a resumable 280-slot prefix")
    total = 0
    for worker in range(4):
        row = value.get("workers", {}).get(str(worker), {})
        progress, receipt = row.get("progress") or {}, row.get("summary") or {}
        indices = list(range(worker, 280, 4))
        count = receipt.get("episodes")
        if (progress != receipt or receipt.get("protocol") != PROTOCOL["protocol"]
                or receipt.get("arm") != arm or receipt.get("worker_id") != worker
                or receipt.get("expected") != 70 or type(count) is not int
                or not 0 <= count <= 70 or receipt.get("indices") != indices
                or receipt.get("completed_indices") != indices[:count]
                or receipt.get("status") not in ("complete", "checkpointed")
                or (receipt.get("status") == "complete"
                    and (count != 70 or receipt.get("complete") is not True))
                or (receipt.get("status") == "checkpointed"
                    and (receipt.get("complete") is not False
                         or receipt.get("soft_deadline_reached") is not True))):
            raise RuntimeError(f"Worker {worker} does not prove a resumable episode prefix")
        total += count
    if (total != summary["episode_slots"]
            or summary.get("arm_episodes") != {arm: total}):
        raise RuntimeError("Worker receipts do not sum to the checkpointed aggregate")
    return summary


def _intent(unit: str, attempt: int, source_job_id: str | None = None) -> dict:
    arm = _arm_for_unit(unit)
    evaluation = unit in EVAL_UNIT.values()
    resources = EVAL_RESOURCES if evaluation else CAL_RESOURCES
    script = ROOT + ("/run.pbs" if evaluation else "/calibrate.pbs")
    value = {
        "schema_version": 1, "unit": unit, "arm": arm,
        "attempt": attempt, "job_name": _job_name(unit, attempt), "script": script,
        "remote_root": ROOT, "created_utc": datetime.now(timezone.utc).isoformat(),
        **resources, "status": "intent_before_qsub",
    }
    if source_job_id:
        value.update(resume=True, source_job_id=source_job_id, resumes_job_id=source_job_id)
    return value


def _submit(transport, unit: str, intent: dict, records: dict) -> dict:
    _require_uploaded(transport, CAL_FILES if unit in ARMS else EVAL_FILES)
    name = intent["job_name"]
    matches = _query_name(transport, name)
    if len(matches) > 1:
        raise RuntimeError("Multiple exact-name PBS jobs exist; no qsub was attempted")
    if matches:
        recovered = _recover(transport, unit, intent, records)
        if recovered:
            return recovered
        raise RuntimeError("Matching PBS job was not recoverable")
    with remote.sftp_client(transport) as sftp:
        current = _json(sftp, _intent_path(unit))
        if current and current.get("attempt", 0) >= intent["attempt"]:
            raise RuntimeError("A durable attempt intent already exists; no second qsub")
        _write_json(sftp, _intent_path(unit), intent)

    resource = EVAL_RESOURCES if unit in EVAL_UNIT.values() else CAL_RESOURCES
    cmd = ("qsub -N " + shlex.quote(name) + " -q " + resource["queue"]
           + " -l select=" + shlex.quote(resource["select"])
           + " -l walltime=" + resource["walltime"]
           + " -o " + shlex.quote(ARTIFACTS + "/" + name + ".pbs.out"))
    if unit in ARMS:
        env = {"FW_CAL_ARM": unit}
        if intent.get("resume"):
            env.update(FW_CAL_RESUME="1", FW_CAL_SOURCE_JOB_ID=intent["source_job_id"])
    else:
        env = {"FW_EVAL_ARM": _arm_for_unit(unit)}
        if intent.get("resume"):
            env.update(FW_EVAL_RESUME="1", FW_EVAL_SOURCE_JOB_ID=intent["source_job_id"])
    cmd += " -v " + shlex.quote(",".join(f"{key}={value}" for key, value in env.items()))
    cmd += " " + shlex.quote(intent["script"])
    code, output = remote.command(transport, cmd, timeout=45)
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="unknown_or_rejected", return_code=code, response=output[:1000])
        with remote.sftp_client(transport) as sftp:
            _write_json(sftp, _intent_path(unit), intent)
        raise RuntimeError("qsub outcome is unknown/rejected; query its unique exact job name before retrying")
    intent.update(job_id=job_id, status="confirmed")
    with remote.sftp_client(transport) as sftp:
        _write_json(sftp, _intent_path(unit), intent)
        _save_handle(sftp, records, unit, intent)
    return intent


def _backward_check(transport) -> dict:
    with remote.sftp_client(transport) as sftp:
        result = _json(sftp, RESULTS + "/backward_check.json")
    if not (isinstance(result, dict) and result.get("passed") is True
            and result.get("implementation") == BACKWARD_IMPLEMENTATION
            and result.get("max_vectors_per_chunk") == BACKWARD_MAX_VECTORS):
        raise RuntimeError("joint requires backward_check.json passed for bounded_vector_chunks_v2 (1048576)")
    return result


def submit_calibration(transport, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Arm must be clip_only or joint")
    _require_uploaded(transport, CAL_FILES)
    if arm == "joint":
        _backward_check(transport)
    with remote.sftp_client(transport) as sftp:
        records = _records(sftp)
        pending = _json(sftp, _intent_path(arm))
    old = _resolve(transport, arm, records, pending)
    if old:
        observed = scheduler(transport, old["job_id"], unit=arm)
        if observed.get("query_rc") != 0 or observed.get("Job_Name") != old.get("job_name"):
            raise RuntimeError("Existing calibration handle could not be verified")
        if observed.get("job_state") not in ("F", "X"):
            return {"handle": old, "scheduler": observed}
        _training_complete(old, observed, arm)
        return {"handle": old, "scheduler": observed, "complete": True}
    return _submit(transport, arm, _intent(arm, 1), records)


def resume_joint(transport) -> dict:
    _require_uploaded(transport, CAL_FILES)
    _backward_check(transport)
    with remote.sftp_client(transport) as sftp:
        records = _records(sftp)
        pending = _json(sftp, _intent_path("joint"))
    previous = _resolve(transport, "joint", records, pending)
    if not previous:
        raise RuntimeError("Joint has no confirmed attempt to resume")
    observed = scheduler(transport, previous["job_id"], unit="joint")
    _training_checkpointed(previous, observed, "joint")
    attempt = int(previous["attempt"]) + 1
    return _submit(transport, "joint", _intent("joint", attempt, previous["job_id"]), records)


def submit_evaluation(transport, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Evaluation arm must be clip_only or joint")
    _require_uploaded(transport, EVAL_FILES)
    if arm == "joint":
        _backward_check(transport)
    with remote.sftp_client(transport) as sftp:
        records = _records(sftp)
        unit = EVAL_UNIT[arm]
        pending = _json(sftp, _intent_path(unit))
    old_eval = _resolve(transport, unit, records, pending)
    if old_eval:
        observed = scheduler(transport, old_eval["job_id"], unit=unit)
        if observed.get("query_rc") != 0 or observed.get("Job_Name") != old_eval.get("job_name"):
            raise RuntimeError("Existing evaluation handle could not be verified")
        if observed.get("job_state") in ("F", "X"):
            try:
                _eval_complete(old_eval, observed, arm)
            except RuntimeError:
                if observed.get("Exit_status") == "0" and observed.get("PIPELINE_CHECKPOINTED") is True:
                    _eval_checkpointed(old_eval, observed, arm)
                    return {"handle": old_eval, "scheduler": observed, "checkpointed": True}
                raise RuntimeError("Evaluation is terminal but not complete; no second evaluation was submitted")
        return {"handle": old_eval, "scheduler": observed}
    train = _latest(records, arm)
    if not train:
        raise RuntimeError(f"{arm} evaluation requires a confirmed {arm} calibration handle")
    training_status = scheduler(transport, train["job_id"], unit=arm)
    _training_complete(train, training_status, arm)
    return _submit(transport, unit, _intent(unit, 1), records)


def resume_evaluation(transport, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Evaluation arm must be clip_only or joint")
    _require_uploaded(transport, EVAL_FILES)
    unit = EVAL_UNIT[arm]
    with remote.sftp_client(transport) as sftp:
        records = _records(sftp)
        pending = _json(sftp, _intent_path(unit))
    previous = _resolve(transport, unit, records, pending)
    if not previous:
        raise RuntimeError(f"{arm} evaluation has no confirmed attempt to resume")
    observed = scheduler(transport, previous["job_id"], unit=unit)
    _eval_checkpointed(previous, observed, arm)
    attempt = int(previous["attempt"]) + 1
    return _submit(transport, unit, _intent(unit, attempt, previous["job_id"]), records)


def _status_unit(transport, unit: str, records: dict) -> dict:
    with remote.sftp_client(transport) as sftp:
        intent = _json(sftp, _intent_path(unit))
    handle = _resolve(transport, unit, records, intent)
    if not handle:
        return {"unit": unit, "status": "not_submitted", "handle": None}
    return {"unit": unit, "handle": handle,
            "scheduler": scheduler(transport, handle["job_id"], unit=unit)}


def status(transport, arm: str | None = None, unit: str | None = None) -> dict:
    if arm and arm not in ARMS:
        raise ValueError("Arm must be clip_only or joint")
    if unit and unit not in UNITS:
        raise ValueError("Unknown C80 unit")
    if arm and unit:
        raise ValueError("Use either --arm or --unit")
    selected = (unit,) if unit else ((arm, EVAL_UNIT[arm]) if arm else UNITS)
    with remote.sftp_client(transport) as sftp:
        records = _records(sftp)
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "units": {name: _status_unit(transport, name, records) for name in selected}}


def pull(transport, arm: str | None = None, unit: str | None = None) -> dict:
    result = status(transport, arm, unit)
    target = HERE / "pulled"
    target.mkdir(exist_ok=True)
    saved = []
    for name, row in result["units"].items():
        if row.get("handle"):
            path = target / f"{name}_{row['handle']['job_id']}.json"
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(path)
            saved.append(str(path))
    result["pulled_files"] = saved
    result["scope"] = "bounded control JSON only; checkpoints and episode files are excluded"
    return result


def logs(transport, unit: str) -> dict:
    if unit not in UNITS:
        raise ValueError("Unknown C80 unit")
    row = status(transport, unit=unit)["units"][unit]
    handle = row.get("handle")
    if not handle:
        raise RuntimeError(f"{unit} has no PBS handle")
    path = ARTIFACTS + "/" + handle["job_id"] + "/job.log"
    code, output = remote.command(
        transport, "tail -c 8192 " + shlex.quote(path) + " 2>/dev/null", timeout=20)
    raw = output.encode("utf-8", errors="replace")
    return {"unit": unit, "job_id": handle["job_id"], "query_rc": code,
            "tail": raw[-8192:].decode("utf-8", errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "status", "submit", "resume",
                                             "submit-evaluation", "resume-evaluation", "pull", "logs"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--unit", choices=UNITS)
    parser.add_argument("--scope", choices=("all", "calibration", "evaluation"), default="all",
                        help="Source subset for upload only")
    args = parser.parse_args()
    if args.action in ("submit", "resume", "submit-evaluation", "resume-evaluation") and not args.arm:
        parser.error("--arm is required")
    if args.action == "resume" and args.arm != "joint":
        parser.error("Only the joint training arm can resume")
    if args.action == "logs" and not args.unit:
        parser.error("--unit is required for logs")
    if args.action in ("upload", "submit", "resume", "submit-evaluation", "resume-evaluation", "logs") and (args.arm and args.unit):
        parser.error("Choose --arm or --unit, not both")
    if args.arm and args.action not in ("status", "pull", "submit", "resume", "submit-evaluation", "resume-evaluation"):
        parser.error("--arm is not supported for this action")
    if args.unit and args.action not in ("status", "pull", "logs"):
        parser.error("--unit is supported only for status, pull, and logs")
    if args.action != "upload" and args.scope != "all":
        parser.error("--scope is supported only for upload")

    jump, transport = remote.direct_connect()
    try:
        _mkdirs(transport)
        if args.action == "upload":
            names = (CAL_FILES if args.scope == "calibration" else
                     EVAL_FILES if args.scope == "evaluation" else None)
            result = {"uploaded": upload(transport, names)}
        elif args.action == "status":
            result = status(transport, args.arm, args.unit)
        elif args.action == "submit":
            result = submit_calibration(transport, args.arm)
        elif args.action == "resume":
            result = resume_joint(transport)
        elif args.action == "submit-evaluation":
            result = submit_evaluation(transport, args.arm)
        elif args.action == "resume-evaluation":
            result = resume_evaluation(transport, args.arm)
        elif args.action == "pull":
            result = pull(transport, args.arm, args.unit)
        else:
            result = logs(transport, args.unit)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
