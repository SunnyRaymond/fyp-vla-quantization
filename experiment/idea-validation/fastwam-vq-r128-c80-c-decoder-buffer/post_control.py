"""Durable bounded PBS control for the CPU-only final report recomputation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "postprocess_pbs_base_control",
    HERE.parent / "fastwam-vq-r128-c80-fused-tuning" / "control.py",
)
_base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_base)
control = _base
remote = control.remote

PROTOCOL_ID = "fastwam-vq-r128-c80-c-decoder-buffer-post-v1"
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-c-decoder-buffer-20261024-post-r1"
USER = "yguo017"
PROJECT = "personal-yguo017"
JOB_NAME = "vqcdp001"
QUEUE = "normal"
EXECUTION_QUEUE = "qdev"
SELECT = "1:ncpus=2:mem=8gb"
WALLTIME = "00:20:00"
FILES = ("protocol.json", "postprocess.pbs", "postprocess.py", "patchedfinalize.py", "reference_profile.py")
ALLOCATION = {
    "queue": QUEUE, "execution_queue": EXECUTION_QUEUE, "gpus": 0,
    "ncpus": 2, "memory_gb": 8, "walltime": WALLTIME,
}
ARTIFACTS = ROOT + "/artifacts"
RESULTS = ROOT + "/results"
STATE_PATH = ROOT + "/control/state.json"
LIMIT = min(getattr(remote, "LIMIT", 256 * 1024), 256 * 1024)
HANDLE = control.HANDLE


def configure() -> None:
    control.HERE = HERE / "post"
    control.PROTOCOL_ID = PROTOCOL_ID
    control.ROOT = ROOT
    control.USER = USER
    control.PROJECT = PROJECT
    control.JOB_NAME = JOB_NAME
    control.JOB_NAME_RE = re.compile("^" + re.escape(JOB_NAME) + "$" )
    control.QUEUE = QUEUE
    control.EXECUTION_QUEUE = EXECUTION_QUEUE
    control.SELECT = SELECT
    control.WALLTIME = WALLTIME
    control.ALLOCATION = ALLOCATION
    control.FILES = FILES
    control.ARTIFACTS = ARTIFACTS
    control.RESULTS = RESULTS
    control.STATE_PATH = STATE_PATH
    control._local_bytes = local_bytes
    control._resolve = resolve
    control.submit = submit
    control._qsub_command = qsub_command
    control.complete_gate = complete_gate
    control._compact = compact
    control.pull = pull


def local_bytes(name: str) -> bytes:
    paths = {
        "protocol.json": HERE / "post" / "protocol.json",
        "postprocess.pbs": HERE / "postprocess.pbs",
        "postprocess.py": HERE / "postprocess.py",
        "patchedfinalize.py": HERE / "finalize.py",
        "reference_profile.py": HERE / "reference_profile.py",
    }
    path = paths.get(name)
    if path is None or not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("Missing or oversized postprocess source: " + name)
    data = path.read_bytes().replace(b"\r\n", b"\n")
    data.decode("utf-8")
    return data


def qsub_command() -> str:
    return (
        "qsub -N " + JOB_NAME + " -q " + QUEUE + " -P " + PROJECT
        + " -l select=" + shlex.quote(SELECT)
        + " -l walltime=" + WALLTIME
        + " -o " + shlex.quote(ARTIFACTS + "/" + JOB_NAME + ".pbs.out")
        + " -j oe " + shlex.quote(ROOT + "/postprocess.pbs")
    )


def resolve(transport) -> tuple[dict | None, dict]:
    with remote.sftp_client(transport) as sftp:
        state = control._state(sftp)
    handle = state.get("handle")
    if handle:
        if (not HANDLE.fullmatch(str(handle.get("job_id", "")))
                or handle.get("job_name") != JOB_NAME
                or handle.get("protocol") != PROTOCOL_ID
                or handle.get("remote_root") != ROOT):
            raise RuntimeError("Saved CPU postprocess PBS handle belongs to another job")
        return handle, state
    intent = state.get("intent")
    if not intent:
        return None, state
    if (intent.get("job_name") != JOB_NAME or intent.get("protocol") != PROTOCOL_ID
            or intent.get("status") not in ("intent_before_qsub", "qsub_outcome_unknown")):
        raise RuntimeError("Durable postprocess submission intent is unresolved")
    jobs = control._named_jobs(transport)
    if len(jobs) != 1:
        raise RuntimeError("qsub outcome is unresolved; refusing a second CPU job")
    job_id = jobs[0]
    scheduler = control._scheduler(transport, job_id)
    if scheduler.get("query_rc") != 0 or scheduler.get("Job_Name") != JOB_NAME:
        raise RuntimeError("Could not verify the exact postprocess PBS handle")
    handle = {
        "remote_root": ROOT, "job_id": job_id, "job_name": JOB_NAME,
        "protocol": PROTOCOL_ID, "queue": QUEUE,
        "execution_queue": EXECUTION_QUEUE, "select": SELECT,
        "gpus": 0, "ncpus": 2, "memory_gb": 8, "walltime": WALLTIME,
        "status": "recovered", "confirmed_utc": utc_now(),
    }
    intent.update(job_id=job_id, status="confirmed", recovered_utc=utc_now())
    state.update(intent=intent, handle=handle)
    control._save_state(transport, state)
    return handle, state


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit(transport) -> dict:
    control._mkdirs(transport)
    protocol = control.load_protocol()
    handle, state = resolve(transport)
    if handle:
        return {"existing": True, "handle": handle}
    if state.get("intent"):
        raise RuntimeError("Unresolved durable submission exists; qsub was not repeated")
    control._require_uploaded(transport)
    if control._named_jobs(transport):
        raise RuntimeError("Exact postprocess job name exists without a durable intent")
    intent = {
        "protocol": protocol["protocol"], "job_name": JOB_NAME,
        "script": ROOT + "/postprocess.pbs", "status": "intent_before_qsub",
        "created_utc": utc_now(), "queue": QUEUE, "select": SELECT,
        "walltime": WALLTIME,
    }
    state.update(intent=intent, handle=None)
    control._save_state(transport, state)
    try:
        code, output = remote.command(transport, qsub_command(), timeout=45)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        control._save_state(transport, state)
        raise RuntimeError("qsub outcome unknown; recover by exact job name before retry") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", return_code=code,
                       response=output[-1000:], updated_utc=utc_now())
        control._save_state(transport, state)
        raise RuntimeError("qsub outcome is unresolved; do not submit again")
    handle = {
        "remote_root": ROOT, "job_id": job_id, "job_name": JOB_NAME,
        "protocol": PROTOCOL_ID, "queue": QUEUE,
        "execution_queue": EXECUTION_QUEUE, "select": SELECT,
        "gpus": 0, "ncpus": 2, "memory_gb": 8, "walltime": WALLTIME,
        "status": "confirmed", "confirmed_utc": utc_now(),
    }
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    state.update(intent=intent, handle=handle)
    control._save_state(transport, state)
    return {"handle": handle}


def complete_gate(value: dict) -> bool:
    try:
        handle = value["handle"]
        scheduler = value["scheduler"]
        summary = value["summary"]
        artifact = value["artifact"]
        post = json.loads((HERE / "post" / "protocol.json").read_text(encoding="utf-8"))
        modes = summary["modes"]
        allocation_ok = (
            handle.get("remote_root") == ROOT
            and handle.get("protocol") == PROTOCOL_ID
            and handle.get("job_name") == JOB_NAME
            and handle.get("execution_queue") == EXECUTION_QUEUE
            and handle.get("gpus") == 0
            and handle.get("ncpus") == 2
            and handle.get("memory_gb") == 8
            and handle.get("walltime") == WALLTIME
        )
        scheduler_ok = (
            scheduler.get("query_rc") == 0
            and scheduler.get("job_state") == "F"
            and scheduler.get("Exit_status") == "0"
            and scheduler.get("Job_Name") == JOB_NAME
            and scheduler.get("queue") == EXECUTION_QUEUE
            and scheduler.get("Resource_List.walltime") == WALLTIME
        )
        post_ok = (
            summary.get("protocol") == post["source"]["protocol"]
            and summary.get("pbs_jobid") == post["source"]["job_id"]
            and summary.get("complete") is True
            and summary.get("postprocess_pbs_jobid") == handle.get("job_id")
            and summary.get("source_raw_complete") is False
            and summary.get("source_main_state") == "F"
            and summary.get("source_main_exit_status") == 1
            and summary.get("source_main_F1") is True
            and summary.get("postprocess_kind") == "cpu_recompute"
            and summary.get("new_model_forwards") == 0
            and summary.get("postprocess_validation", {}).get("passed") is True
            and summary.get("storage_inventory", {}).get("passed") is True
            and summary.get("candidate_modes") == ["c_decoder", "c_buffer"]
            and summary.get("full_queries_per_mode") == 80
            and summary.get("new_training_updates") == 0
            and summary.get("closed_loop_episodes") == 0
        )
        return (
            HANDLE.fullmatch(str(handle.get("job_id", ""))) is not None
            and allocation_ok and scheduler_ok and post_ok
            and artifact.get("exit_code") == "0"
            and artifact.get("pipeline_complete") is True
            and all(candidate_gate(name, modes.get(name, {}), post) for name in ("c_decoder", "c_buffer"))
        )
    except (KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError):
        return False


def candidate_gate(name: str, row: dict, post: dict) -> bool:
    parity = row.get("parity_vs_primary", {})
    validation = row.get("validation_checks", {})
    checks = validation.get("checks", {})
    numerics = row.get("numerical_flags", {})
    provenance = row.get("provenance", {})
    source = post["source"]
    if name == "c_decoder":
        reuse = json.loads((HERE / "post" / "protocol.json").read_text(encoding="utf-8"))["completed_candidate_reuse"]
        provenance_ok = (
            row.get("status") == "reused_candidate"
            and provenance.get("reused") is True
            and provenance.get("pbs_jobid") == reuse.get("source_job")
            and provenance.get("source_job") == reuse.get("source_job")
            and provenance.get("protocol") == reuse.get("source_protocol")
        )
    else:
        provenance_ok = (
            row.get("status") == "measured"
            and provenance.get("reused") is False
            and provenance.get("pbs_jobid") == source.get("job_id")
        )
    return (
        provenance_ok
        and row.get("queries") == 80
        and parity.get("passed") is True
        and parity.get("exact") is True
        and parity.get("queries") == 80
        and parity.get("max_abs") == 0.0
        and parity.get("rmse") == 0.0
        and validation.get("passed") is True
        and row.get("stage_gate", {}).get("passed") is True
        and all(checks.get(key) is True for key in (
            "validation_full80_exact_80", "decoder_bitwise_all_modules",
            "anchor_action_max_abs_eq_0", "anchor_future_max_abs_eq_0",
            "expected_runtime_kind", "logical_coverage_5258_per_query",
            "only_expected_graph_regions", "video_step_replays_exact",
            "video_prefill_replays_exact", "action_step_replays_exact",
        ))
        and numerics.get("deterministic_algorithms") is True
        and numerics.get("fill_uninitialized_memory") is True
        and numerics.get("allow_tf32") is False
        and numerics.get("allow_bf16_reduced_precision_reduction") is True
        and numerics.get("compiled") is False
        and numerics.get("cuda_graph") is True
        and numerics.get("new_training_updates") == 0
        and numerics.get("closed_loop_episodes") == 0
    )


def compact(value: dict) -> dict:
    handle = value.get("handle") or {}
    scheduler = value.get("scheduler") or {}
    summary = value.get("summary") or {}
    return {
        "status": value.get("status"),
        "job_id": handle.get("job_id"),
        "job_state": scheduler.get("job_state"),
        "queue": scheduler.get("queue"),
        "source_producer_job": summary.get("pbs_jobid"),
        "postprocess_complete": summary.get("complete"),
        "inventory_passed": summary.get("storage_inventory", {}).get("passed"),
        "completion_passed": value.get("completion_passed", False),
    }


def pull(transport) -> dict:
    value = control.status(transport)
    handle = value.get("handle")
    if not handle:
        raise RuntimeError("No submitted CPU postprocess job")
    target = HERE / "pulled" / handle["job_id"]
    files, skipped = [], {}
    with remote.sftp_client(transport) as sftp:
        for name in ("summary.json", "stage_gates.json", "postprocess_receipt.json"):
            data, reason = control._pull_bytes(sftp, RESULTS + "/" + name)
            if reason:
                skipped[name] = reason
            elif data is not None:
                control._save_bytes(target / name, data)
                files.append(name)
    log_tail = value.get("log_tail")
    if isinstance(log_tail, str):
        control._save_bytes(target / "job.log.tail.txt", log_tail.encode("utf-8")[-8192:])
        files.append("job.log.tail.txt")
    control._save_bytes(target / "status.json", (json.dumps(compact(value), indent=2) + "\n").encode("utf-8"))
    files.append("status.json")
    return {**compact(value), "pulled_files": files, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "logs", "pull"))
    args = parser.parse_args()
    configure()
    jump, transport = control.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": control.upload(transport)}
            code = 0
        elif args.action == "submit":
            result, code = control.submit(transport), 0
        elif args.action == "status":
            result, code = compact(control.status(transport)), 0
        elif args.action == "verify":
            value = control.status(transport)
            result = {**compact(value), "verified_complete": complete_gate(value)}
            code = 0 if result["verified_complete"] else 1
        elif args.action == "logs":
            result, code = control.logs(transport), 0
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
