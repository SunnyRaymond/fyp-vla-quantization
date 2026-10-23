"""Bounded, host-key-verified PBS control for the learned R128 C8 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "fastwam-libero-plus-pilot"
REMOTE_HELPER = PILOT / "remote.py"
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021"
PROTOCOL_ID = "fastwam-vq-learned-r128-c8-v1"
JOB_NAME = "fwlr128c8"
LIMIT = 256 * 1024
MAX_BYTES = LIMIT - 1
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
ARTIFACTS = ROOT + "/artifacts"
STATE_PATH = ROOT + "/control/state.json"
PROGRESS_PATH = ROOT + "/results/progress.json"
SUMMARY_PATH = ROOT + "/results/summary.json"
CHECKPOINT_PATH = ROOT + "/results/state.pt"
HISTORY_DIR = ROOT + "/control/history"


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or is not below the 256 KiB limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("protocol.json does not match the frozen learned R128 C8 protocol")
    if value.get("remote_root", ROOT) != ROOT:
        raise RuntimeError("protocol.json names an unexpected remote root")
    return value


def load_remote_helper():
    sys.path.insert(0, str(PILOT))
    spec = importlib.util.spec_from_file_location("fastwam_learned_r128_c8_remote", REMOTE_HELPER)
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
        raise RuntimeError("Local control state is not below the 256 KiB limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Local control state must be a JSON object")
    return value


def write_remote_json(sftp, path: str, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) >= LIMIT:
        raise RuntimeError("Remote control JSON is not below the 256 KiB limit")
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


def scheduler(remote, transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, f"qstat -xf {shlex.quote(job_id)}", timeout=35)
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "exec_host", "comment", "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return {"job_id": job_id, "query_rc": code, **fields}


def local_files() -> list[Path]:
    paths = sorted(HERE.glob("*.py"), key=lambda path: path.name)
    paths.extend(HERE / name for name in ("protocol.json", "run.pbs", "environment.sh"))
    if not all(path.is_file() and path.parent.resolve() == HERE.resolve() for path in paths):
        missing = [path.name for path in paths if not path.is_file()]
        raise FileNotFoundError("Missing upload files: " + ", ".join(missing))
    return paths


def transfer_files(sftp, *, include_protocol: bool) -> list[dict]:
    sent = []
    for path in local_files():
        if path.name == "protocol.json" and not include_protocol:
            continue
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if len(data) >= LIMIT:
            raise RuntimeError(f"Only small code/control files below 256 KiB may be uploaded: {path.name}")
        data.decode("utf-8")
        target = ROOT + "/" + path.name
        with sftp.file(target + ".tmp", "wb") as stream:
            stream.write(data)
        sftp.posix_rename(target + ".tmp", target)
        sent.append({"name": path.name, "bytes": len(data)})
    return sent


def upload(remote, transport) -> dict:
    load_protocol()
    code, output = remote.command(
        transport, "mkdir -p " + shlex.quote(ROOT + "/control") + " " + shlex.quote(ARTIFACTS), timeout=25
    )
    if code:
        raise RuntimeError(f"Unable to create the remote control/artifact directories: {output[-500:]}")
    with remote.sftp_client(transport) as sftp:
        remote_state = read_remote_json(sftp, STATE_PATH) or {}
        if remote_state.get("handle") or remote_state.get("intent"):
            raise RuntimeError("Submission state exists; refusing to change files after a possible qsub")
        sent = transfer_files(sftp, include_protocol=True)
        remote_protocol = read_remote_json(sftp, ROOT + "/protocol.json")
        if remote_protocol != load_protocol():
            raise RuntimeError("Uploaded protocol does not match the local frozen protocol")
    return {"uploaded": sent, "remote_root": ROOT}


def read_state(remote, transport) -> tuple[dict, dict]:
    with remote.sftp_client(transport) as sftp:
        remote_state = read_remote_json(sftp, STATE_PATH) or {}
    local_state = read_local_state()
    return remote_state, local_state


def merged_state(remote_state: dict, local_state: dict, handle: dict) -> dict:
    # Remote state is the durable authority; a stale local copy must not erase
    # terminal evidence or an unresolved intent already recorded on the cluster.
    state = {**local_state, **remote_state, "handle": handle}
    for key in ("intent", "last_terminal", "last_verified"):
        if not state.get(key):
            state[key] = remote_state.get(key) or local_state.get(key)
    return state


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
    raise RuntimeError("Saved handle has no valid attempt number/job name")


def attempt_job_name(attempt: int) -> str:
    if type(attempt) is not int or attempt < 0:
        raise ValueError("Attempt number must be a nonnegative integer")
    return JOB_NAME if attempt == 0 else f"{JOB_NAME}-a{attempt}"


def is_incomplete_failure(observation: dict, protocol: dict) -> list[str]:
    scheduler_state = observation.get("scheduler") or {}
    handle = observation.get("handle") or {}
    summary = observation.get("summary") or {}
    failures = []
    terminal = scheduler_state.get("job_state") in ("F", "X")
    terminal_witness = scheduler_state.get("query_rc") == 0 or (
        scheduler_state.get("terminal_from_history") is True
        and scheduler_state.get("job_id") == handle.get("job_id")
    )
    if not terminal or not terminal_witness:
        failures.append("current PBS handle is not confirmed terminal F/X")
    if observation.get("artifact_exit_code") == 0:
        failures.append("artifact exit_code.txt is 0; refusing to rerun a successful worker")
    if observation.get("phase_complete") is True:
        failures.append("PHASE_COMPLETE exists; refusing to rerun a completed worker")
    if summary.get("status") == "complete" or summary.get("complete") is True:
        failures.append("stable results/summary.json is complete; refusing to resume")
    if protocol.get("protocol") != PROTOCOL_ID:
        failures.append("local protocol does not match the frozen protocol ID")
    return failures


def checkpoint_resume_mode(progress: dict | None, checkpoint_size: int | None) -> str:
    """Choose resume or a safe zero-update restart without altering results."""
    if checkpoint_size is not None:
        if checkpoint_size <= 0:
            raise RuntimeError("results/state.pt exists but is empty; safe resume is unavailable")
        return "checkpoint"
    if not isinstance(progress, dict):
        raise RuntimeError("results/state.pt and worker progress are both missing; refusing a fresh attempt")
    if progress.get("status") == "complete" or progress.get("complete") is True:
        raise RuntimeError("worker progress claims completion; refusing a fresh attempt")
    for name in ("rotation_updates", "vq_updates"):
        value = progress.get(name)
        if type(value) is not int or value != 0:
            raise RuntimeError(
                f"results/state.pt is missing but progress {name} is not confirmed zero; refusing a fresh attempt"
            )
    return "fresh"


def named_jobs(remote, transport, job_name: str, *, include_history: bool) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_name):
        raise ValueError("Invalid PBS job name")
    history = " -x" if include_history else ""
    code, output = remote.command(
        transport,
        f"qselect{history} -u yguo017 -N {shlex.quote(job_name)} 2>&1",
        timeout=35,
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError(f"PBS job-name query failed for {job_name}; refusing to submit")
    matches = output.split()
    if any(not HANDLE.fullmatch(job_id) for job_id in matches):
        raise RuntimeError(f"PBS returned an invalid job handle for {job_name}")
    return matches


def save_history(remote, transport, state: dict, previous: dict, observation: dict) -> dict:
    history = list(state.get("history") or [])
    job_id = previous["job_id"]
    if not any(item.get("job_id") == job_id for item in history if isinstance(item, dict)):
        record = {
            "job_id": job_id,
            "job_name": previous.get("job_name"),
            "attempt": attempt_number(previous),
            "confirmed_utc": previous.get("confirmed_utc"),
            "terminal": observation.get("scheduler"),
            "artifact_exit_code": observation.get("artifact_exit_code"),
            "phase_complete": observation.get("phase_complete"),
            "summary_status": (observation.get("summary") or {}).get("status"),
            "archived_utc": utc_now(),
        }
        history.append(record)
        code, output = remote.command(
            transport, "mkdir -p " + shlex.quote(HISTORY_DIR), timeout=25
        )
        if code:
            raise RuntimeError(f"Unable to preserve previous attempt history: {output[-500:]}")
        with remote.sftp_client(transport) as sftp:
            write_remote_json(sftp, f"{HISTORY_DIR}/{job_id}.json", record)
    state["history"] = history
    return state


def recover_pending_resume(remote, transport, state: dict) -> dict | None:
    intent = state.get("intent") or {}
    if intent.get("kind") != "resume" or intent.get("status") not in (
        "intent_before_qsub", "qsub_outcome_unknown",
    ):
        return None
    candidate_name = intent.get("job_name")
    if not isinstance(candidate_name, str):
        raise RuntimeError("Unresolved resume intent has no candidate job name")
    matches = named_jobs(remote, transport, candidate_name, include_history=True)
    if len(matches) != 1:
        raise RuntimeError(
            "Resume qsub outcome is unresolved; exact PBS history did not identify one job. "
            "Do not submit another attempt."
        )
    handle = {
        "job_id": matches[0], "job_name": candidate_name, "attempt": intent.get("attempt"),
        "remote_root": ROOT, "status": "recovered_from_pbs_history", "confirmed_utc": utc_now(),
    }
    intent.update(status="confirmed", job_id=matches[0], recovered_utc=utc_now())
    state.update(intent=intent, handle=handle)
    save_submission_state(remote, transport, state)
    return {"recovered": True, "handle": handle, "intent": intent}


def save_submission_state(remote, transport, state: dict) -> None:
    with remote.sftp_client(transport) as sftp:
        write_remote_json(sftp, STATE_PATH, state)
    save_local(HERE / "control_state.json", state)


def submit(remote, transport) -> dict:
    protocol = load_protocol()
    remote_state, local_state = read_state(remote, transport)
    existing = remote_state.get("handle") or local_state.get("handle")
    if existing:
        if not HANDLE.fullmatch(str(existing.get("job_id", ""))):
            raise RuntimeError("Saved PBS handle is invalid")
        return {"existing": True, "handle": existing}

    with remote.sftp_client(transport) as sftp:
        if read_remote_json(sftp, ROOT + "/protocol.json") != protocol:
            raise RuntimeError("Upload the frozen protocol and code before submission")
        for name in ("run.pbs", "environment.sh", "worker.py", "protocol.json"):
            try:
                attrs = sftp.stat(ROOT + "/" + name)
            except OSError as exc:
                if getattr(exc, "errno", None) == 2:
                    raise RuntimeError(f"Remote code file is missing: {name}") from exc
                raise
            if attrs.st_size >= LIMIT:
                raise RuntimeError(f"Remote code file exceeds the upload policy: {name}")

    code, output = remote.command(
        transport,
        f"qselect -x -u yguo017 -N {shlex.quote(JOB_NAME)} 2>&1",
        timeout=35,
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS history query failed; submission was not attempted")
    matches = output.split()
    if len(matches) > 1 or any(not HANDLE.fullmatch(job) for job in matches):
        raise RuntimeError("Ambiguous PBS history; submission was not attempted")

    intent = remote_state.get("intent") or local_state.get("intent")
    if matches:
        handle = {
            "job_id": matches[0], "job_name": JOB_NAME, "remote_root": ROOT,
            "status": "recovered_from_pbs_history", "confirmed_utc": utc_now(),
        }
        state = {"intent": intent, "handle": handle,
                 "last_terminal": local_state.get("last_terminal"),
                 "last_verified": local_state.get("last_verified")}
        save_submission_state(remote, transport, state)
        return {"recovered": True, "handle": handle}
    if intent:
        raise RuntimeError("Unresolved durable submission intent exists; do not submit a second job")

    intent = {
        "kind": "submit", "attempt": 0,
        "job_name": JOB_NAME, "script": ROOT + "/run.pbs", "created_utc": utc_now(),
        "status": "intent_before_qsub", "queue": "normal", "project": "personal-yguo017",
        "select": "1:ncpus=16:mem=110gb:ngpus=1", "walltime": "12:00:00",
    }
    state = {"intent": intent, "handle": None,
             "last_terminal": local_state.get("last_terminal"),
             "last_verified": local_state.get("last_verified")}
    save_submission_state(remote, transport, state)
    command = (
        f"qsub -N {shlex.quote(JOB_NAME)} -q normal -P personal-yguo017 "
        f"-l select=1:ncpus=16:mem=110gb:ngpus=1 -l walltime=12:00:00 "
        f"-o {shlex.quote(ARTIFACTS + '/' + JOB_NAME + '.pbs.out')} -j oe {shlex.quote(ROOT + '/run.pbs')}"
    )
    code, output = remote.command(transport, command, timeout=40)
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="unknown_or_rejected", response=output[:1000], return_code=code)
        state.update(intent=intent)
        save_submission_state(remote, transport, state)
        raise RuntimeError("qsub outcome is unknown or rejected; recover this intent before another submission")
    handle = {
        "job_id": job_id, "job_name": JOB_NAME, "attempt": 0, "remote_root": ROOT,
        "status": "confirmed", "confirmed_utc": utc_now(),
    }
    intent.update(status="confirmed", job_id=job_id)
    state.update(intent=intent, handle=handle)
    save_submission_state(remote, transport, state)
    return {"handle": handle}


def artifact_observation(sftp, job_id: str) -> dict:
    artifact = ARTIFACTS + "/" + job_id
    progress = read_remote_json(sftp, PROGRESS_PATH)
    summary = read_remote_json(sftp, SUMMARY_PATH)
    worker_summary = read_remote_json(sftp, artifact + "/worker_summary.json")
    exit_data = read_small_bytes(sftp, artifact + "/exit_code.txt", 64)
    exit_code = None
    if exit_data is not None:
        match = re.fullmatch(rb"\s*(-?\d+)\s*", exit_data)
        if not match:
            raise RuntimeError("Artifact exit_code.txt is malformed")
        exit_code = int(match.group(1))
    try:
        sftp.stat(artifact + "/PHASE_COMPLETE")
        marker = True
    except OSError as exc:
        if getattr(exc, "errno", None) != 2:
            raise
        marker = False
    log_tail = None
    log_path = artifact + "/job.log"
    try:
        attrs = sftp.stat(log_path)
        with sftp.file(log_path, "rb") as stream:
            stream.seek(max(0, attrs.st_size - 8192))
            log_tail = stream.read(8192).decode("utf-8", errors="replace")
    except OSError as exc:
        if getattr(exc, "errno", None) != 2:
            raise
    return {
        "progress": progress, "summary": summary, "worker_summary": worker_summary,
        "artifact_exit_code": exit_code, "phase_complete": marker, "log_tail": log_tail,
    }


def retain_terminal_history(current: dict, last_terminal: dict | None, job_id: str) -> tuple[dict, dict | None]:
    if current.get("query_rc") == 0:
        current = {**current, "terminal_from_history": False}
        if current.get("job_state") in ("F", "X") and current.get("Exit_status") is not None:
            last_terminal = {key: current.get(key) for key in (
                "job_id", "query_rc", "job_state", "Exit_status", "Job_Name",
                "exec_host", "resources_used.walltime",
            )}
    elif isinstance(last_terminal, dict) and last_terminal.get("job_id") == job_id:
        current = {
            **last_terminal,
            "current_query_rc": current.get("query_rc"),
            "terminal_from_history": True,
        }
    return current, last_terminal


def observe(remote, transport, handle: dict, local_state: dict) -> tuple[dict, dict]:
    job_id = str(handle.get("job_id", ""))
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    current, last_terminal = retain_terminal_history(
        scheduler(remote, transport, job_id), local_state.get("last_terminal"), job_id
    )
    with remote.sftp_client(transport) as sftp:
        artifacts = artifact_observation(sftp, job_id)
    updated = {**local_state, "handle": handle, "last_terminal": last_terminal}
    return {"handle": handle, "scheduler": current, **artifacts}, updated


def status(remote, transport) -> dict:
    remote_state, local_state = read_state(remote, transport)
    handle = remote_state.get("handle") or local_state.get("handle")
    if not isinstance(handle, dict):
        raise RuntimeError("No confirmed PBS handle is available")
    state = merged_state(remote_state, local_state, handle)
    observation, updated = observe(remote, transport, handle, state)
    updated["last_status_utc"] = utc_now()
    with remote.sftp_client(transport) as sftp:
        write_remote_json(sftp, STATE_PATH, updated)
    save_local(HERE / "control_state.json", updated)
    return {"checked_utc": utc_now(), **observation}


def verification_failures(observation: dict, protocol: dict) -> list[str]:
    scheduler_state = observation.get("scheduler") or {}
    job_id = (observation.get("handle") or {}).get("job_id")
    summary = observation.get("summary") or {}
    failures = []
    if scheduler_state.get("job_state") not in ("F", "X"):
        failures.append("PBS job is not in terminal F/X state")
    if str(scheduler_state.get("Exit_status")) != "0":
        failures.append("PBS Exit_status is not 0")
    if observation.get("artifact_exit_code") != 0:
        failures.append("artifact exit_code.txt is not 0")
    if observation.get("phase_complete") is not True:
        failures.append("PHASE_COMPLETE marker is missing")
    if summary.get("status") != "complete" or summary.get("complete") is not True:
        failures.append("results/summary.json is not complete")
    if summary.get("protocol") != protocol.get("protocol") or protocol.get("protocol") != PROTOCOL_ID:
        failures.append("summary protocol does not match the frozen protocol")
    expected = {
        "rotation_updates": 8,
        "vq_updates": 8,
        "target_linears": 614,
        "heldout_queries": 80,
        "calibration_queries": 8,
    }
    for name, value in expected.items():
        if type(summary.get(name)) is not int or summary.get(name) != value:
            failures.append(f"summary {name} is not {value}")
    if summary.get("pbs_jobid") != job_id:
        failures.append("summary pbs_jobid does not match the confirmed PBS handle")
    if summary.get("all_parameters_frozen") is not True:
        failures.append("summary does not confirm all parameters are frozen")
    return failures


def resume(remote, transport) -> dict:
    protocol = load_protocol()
    remote_state, local_state = read_state(remote, transport)
    handle = remote_state.get("handle") or local_state.get("handle")
    if not isinstance(handle, dict) or not HANDLE.fullmatch(str(handle.get("job_id", ""))):
        raise RuntimeError("Resume requires a confirmed current PBS handle")
    state = merged_state(remote_state, local_state, handle)

    # Once qsub may have been called, only history recovery is safe. An empty
    # or failed qselect result is inconclusive and must never trigger another qsub.
    recovered = recover_pending_resume(remote, transport, state)
    if recovered is not None:
        return recovered

    intent = state.get("intent") or {}
    if intent.get("kind") == "resume" and intent.get("status") in (
        "resume_prepared", "resume_code_uploaded",
    ) and intent.get("from_job_id") != handle.get("job_id"):
        raise RuntimeError("Pending resume intent does not belong to the current confirmed handle")

    observation, updated = observe(remote, transport, handle, state)
    updated["last_status_utc"] = utc_now()
    save_submission_state(remote, transport, updated)
    failures = is_incomplete_failure(observation, protocol)
    if failures:
        raise RuntimeError("Resume refused: " + "; ".join(failures))

    checkpoint_size = None
    with remote.sftp_client(transport) as sftp:
        remote_protocol = read_remote_json(sftp, ROOT + "/protocol.json")
        if remote_protocol != protocol:
            raise RuntimeError("Remote protocol differs from the frozen local protocol; resume refused")
        try:
            checkpoint = sftp.stat(CHECKPOINT_PATH)
        except OSError as exc:
            if getattr(exc, "errno", None) == 2:
                checkpoint = None
            else:
                raise
        if checkpoint is not None:
            checkpoint_size = checkpoint.st_size
    checkpoint_mode = checkpoint_resume_mode(observation.get("progress"), checkpoint_size)

    prior_attempt = attempt_number(handle)
    pending = state.get("intent") or {}
    if pending.get("kind") == "resume" and pending.get("status") in (
        "resume_prepared", "resume_code_uploaded",
    ):
        attempt = pending.get("attempt")
        candidate_name = pending.get("job_name")
        if type(attempt) is not int or attempt <= prior_attempt or candidate_name != attempt_job_name(attempt):
            raise RuntimeError("Pending pre-qsub resume intent is malformed; refusing to continue")
        if pending.get("status") == "resume_code_uploaded":
            # Re-upload is deterministic and bounded; this also repairs a partial local update.
            pass
    else:
        attempt = max(
            prior_attempt + 1,
            max((item.get("attempt", -1) + 1 for item in (state.get("history") or [])
                 if isinstance(item, dict) and type(item.get("attempt")) is int), default=1),
        )
        candidate_name = attempt_job_name(attempt)
        updated = save_history(remote, transport, updated, handle, observation)
        pending = {
            "kind": "resume", "status": "resume_prepared", "attempt": attempt,
            "job_name": candidate_name, "from_job_id": handle["job_id"],
            "script": ROOT + "/run.pbs", "created_utc": utc_now(),
            "queue": "normal", "project": "personal-yguo017",
            "select": "1:ncpus=16:mem=110gb:ngpus=1", "walltime": "12:00:00",
        }
        updated["intent"] = pending
        save_submission_state(remote, transport, updated)

    code, output = remote.command(
        transport,
        "mkdir -p " + shlex.quote(ROOT + "/control") + " " + shlex.quote(ARTIFACTS),
        timeout=25,
    )
    if code:
        raise RuntimeError(f"Unable to prepare resume directories: {output[-500:]}")
    with remote.sftp_client(transport) as sftp:
        # Protocol remains byte-for-byte untouched on the remote. Compare its
        # parsed value first, then upload only small code/control files.
        if read_remote_json(sftp, ROOT + "/protocol.json") != protocol:
            raise RuntimeError("Remote protocol changed before resume upload")
        sent = transfer_files(sftp, include_protocol=False)
        if read_remote_json(sftp, ROOT + "/protocol.json") != protocol:
            raise RuntimeError("Resume upload unexpectedly changed protocol.json")
    pending.update(status="resume_code_uploaded", uploaded_utc=utc_now())
    updated.update(intent=pending, handle=handle)
    save_submission_state(remote, transport, updated)

    # -x includes completed history as well as live jobs. A unique attempt name
    # must be unused before qsub; a query problem always stops this path.
    matches = named_jobs(remote, transport, candidate_name, include_history=True)
    if matches:
        raise RuntimeError(
            f"Candidate job name {candidate_name} already exists in PBS; refusing duplicate submission"
        )
    pending.update(status="intent_before_qsub", qsub_started_utc=utc_now())
    updated.update(intent=pending)
    save_submission_state(remote, transport, updated)
    command = (
        f"qsub -N {shlex.quote(candidate_name)} -q normal -P personal-yguo017 "
        f"-l select=1:ncpus=16:mem=110gb:ngpus=1 -l walltime=12:00:00 "
        f"-o {shlex.quote(ARTIFACTS + '/' + candidate_name + '.pbs.out')} -j oe {shlex.quote(ROOT + '/run.pbs')}"
    )
    try:
        code, output = remote.command(transport, command, timeout=40)
    except Exception as exc:
        pending.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        updated["intent"] = pending
        save_submission_state(remote, transport, updated)
        raise RuntimeError(
            "Resume qsub outcome is unknown; a later resume may only recover the exact job from PBS history"
        ) from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        pending.update(
            status="qsub_outcome_unknown", response=output[:1000], return_code=code,
            updated_utc=utc_now(),
        )
        updated["intent"] = pending
        save_submission_state(remote, transport, updated)
        raise RuntimeError(
            "Resume qsub outcome is unknown/rejected; a later resume may only recover the exact job from PBS history"
        )
    new_handle = {
        "job_id": job_id, "job_name": candidate_name, "attempt": attempt,
        "remote_root": ROOT, "status": "confirmed", "confirmed_utc": utc_now(),
    }
    pending.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    updated.update(intent=pending, handle=new_handle)
    save_submission_state(remote, transport, updated)
    return {"resumed": True, "from_job_id": handle["job_id"], "handle": new_handle,
            "uploaded": sent, "checkpoint": CHECKPOINT_PATH if checkpoint_mode == "checkpoint" else None,
            "resume_mode": checkpoint_mode, "intent": pending}


def verify(remote, transport) -> dict:
    protocol = load_protocol()
    remote_state, local_state = read_state(remote, transport)
    handle = remote_state.get("handle") or local_state.get("handle")
    if not isinstance(handle, dict):
        raise RuntimeError("No confirmed PBS handle is available")
    state = merged_state(remote_state, local_state, handle)
    observation, updated = observe(remote, transport, handle, state)
    failures = verification_failures(observation, protocol)
    result = {"checked_utc": utc_now(), "verified": not failures,
              "failures": failures, **observation}
    if not failures:
        updated["last_verified"] = {
            "job_id": handle["job_id"], "verified_utc": result["checked_utc"],
            "scheduler": observation["scheduler"], "summary": observation["summary"],
        }
    updated["last_status_utc"] = result["checked_utc"]
    with remote.sftp_client(transport) as sftp:
        write_remote_json(sftp, STATE_PATH, updated)
    save_local(HERE / "control_state.json", updated)
    return result


def pull(remote, transport) -> dict:
    result = status(remote, transport)
    job_id = result["handle"]["job_id"]
    destination = HERE / "results" / job_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, key in (("progress.json", "progress"), ("summary.json", "summary"),
                      ("worker_summary.json", "worker_summary")):
        value = result.get(key)
        if value is not None:
            save_local(destination / name, value)
    with remote.sftp_client(transport) as sftp:
        for name, remote_path in (
            ("comparison.json", ROOT + "/results/comparison.json"),
            ("RESULTS.zh.md", ROOT + "/results/RESULTS.zh.md"),
            ("costs.json", ROOT + "/results/costs.json"),
            ("COSTS.zh.md", ROOT + "/COSTS.zh.md"),
            ("selfcheck.json", ARTIFACTS + "/" + job_id + "/selfcheck.json"),
        ):
            data = read_small_bytes(sftp, remote_path)
            if data is not None:
                (destination / name).write_bytes(data)
    if result.get("artifact_exit_code") is not None:
        (destination / "exit_code.txt").write_text(str(result["artifact_exit_code"]) + "\n", encoding="utf-8")
    if result.get("log_tail") is not None:
        (destination / "job.log.tail").write_text(result["log_tail"], encoding="utf-8")
    metadata = {key: result.get(key) for key in (
        "checked_utc", "handle", "scheduler", "artifact_exit_code", "phase_complete",
    )}
    save_local(destination / "pull_metadata.json", metadata)
    return {"job_id": job_id, "local_directory": str(destination), "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "resume", "status", "verify", "pull"))
    args = parser.parse_args()
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
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("verified", True) else 1
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
