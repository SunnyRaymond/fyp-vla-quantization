"""Small-file ASPIRE2A upload and durable PBS submit/status control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys


HERE = Path(__file__).resolve().parent
ROTATION_CONTROL = HERE.parent / "fastwam-rotation-baselines" / "control.py"
_spec = importlib.util.spec_from_file_location("vq_plus_rotation_control", ROTATION_CONTROL)
previous = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(previous)
remote = previous.remote

ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-plus-20261012"
CONTROL_ROOT = ROOT + "/control"
JOB_NAME = "fwvqplus1012"
FILES = ("control.py", "run.pbs", "protocol.json", "evaluate.py", "aggregate.py", "vq_adapter.py")
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


def _save_remote_json(sftp, path: str, value: dict) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(data) > remote.LIMIT:
        raise ValueError("Submission state exceeds the small-file limit")
    temp = path + ".tmp"
    with sftp.file(temp, "wb") as stream:
        stream.write(data)
    sftp.posix_rename(temp, path)


def _query_matches(transport, job_name: str = JOB_NAME) -> list[str]:
    code, output = remote.command(
        transport,
        f"qselect -x -u yguo017 -N {shlex.quote(job_name)} 2>&1",
        timeout=35,
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS history query failed; no submission was attempted")
    matches = output.split()
    if any(not HANDLE.fullmatch(job) for job in matches):
        raise RuntimeError("PBS history returned an invalid handle; no submission was attempted")
    return matches


def upload(transport, names: list[str] | None = None) -> list[str]:
    code, output = remote.command(
        transport, "mkdir -p " + shlex.quote(ROOT + "/results") + " "
        + shlex.quote(ROOT + "/artifacts") + " " + shlex.quote(CONTROL_ROOT), timeout=30
    )
    if code:
        raise RuntimeError(output)
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
            target = ROOT + "/" + name
            temp = target + ".tmp"
            with sftp.file(temp, "wb") as stream:
                stream.write(data)
            sftp.posix_rename(temp, target)
            uploaded.append(name)
    return uploaded


def submit(transport) -> dict:
    code, output = remote.command(
        transport, "mkdir -p " + shlex.quote(ROOT + "/results") + " "
        + shlex.quote(ROOT + "/artifacts") + " " + shlex.quote(CONTROL_ROOT), timeout=30
    )
    if code:
        raise RuntimeError(output)
    with remote.sftp_client(transport) as sftp:
        for name in FILES:
            try:
                attrs = sftp.stat(ROOT + "/" + name)
            except OSError as exc:
                if getattr(exc, "errno", None) != 2:
                    raise
                raise RuntimeError(f"Upload is incomplete; missing {name}") from exc
            if attrs.st_size > remote.LIMIT:
                raise RuntimeError(f"Remote control file exceeds the small-file limit: {name}")
        intent_path = CONTROL_ROOT + "/submit_intent.json"
        handle_path = CONTROL_ROOT + "/submit_handle.json"
        latest = _remote_json(sftp, CONTROL_ROOT + "/latest_handle.json")
        intent = _remote_json(sftp, intent_path)
        handle = _remote_json(sftp, handle_path)

    if latest and latest.get("resume_sequence", 0) > 0:
        raise RuntimeError("A resumed run already exists; use status or the explicit resume action")

    matches = _query_matches(transport)
    if len(matches) > 1:
        raise RuntimeError("More than one matching PBS job exists; reconcile manually without submitting")
    if matches:
        job_id = matches[0]
        if (intent or {}).get("job_id") not in (None, job_id) or (handle or {}).get("job_id") not in (None, job_id):
            raise RuntimeError("PBS history conflicts with the durable submission handle")
        recovered = {
            **(intent or {}), "job_name": JOB_NAME, "job_id": job_id,
            "resume_sequence": 0, "status": "confirmed" if intent else "recovered",
        }
        with remote.sftp_client(transport) as sftp:
            _save_remote_json(sftp, intent_path, recovered)
            _save_remote_json(sftp, handle_path, recovered)
            _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", recovered)
        return recovered
    if intent or handle:
        raise RuntimeError("A durable or unresolved submission record exists; no second qsub will be issued")

    intent = {
        "job_name": JOB_NAME,
        "script": ROOT + "/run.pbs",
        "remote_root": ROOT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": 4,
        "walltime": "24:00:00",
        "episodes": 560,
        "workers": 4,
        "variants_per_worker": 140,
        "status": "intent_before_qsub",
        "resume_sequence": 0,
    }
    with remote.sftp_client(transport) as sftp:
        _save_remote_json(sftp, intent_path, intent)

    command = (
        "qsub -N " + shlex.quote(JOB_NAME)
        + " -o " + shlex.quote(ROOT + "/artifacts/pbs.out")
        + " " + shlex.quote(ROOT + "/run.pbs")
    )
    try:
        code, output = remote.command(transport, command, timeout=40)
    except Exception:
        # The durable intent deliberately remains unresolved; never issue a blind retry.
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
        _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", intent)
    return intent


def resume(transport) -> dict:
    with remote.sftp_client(transport) as sftp:
        previous_handle = _remote_json(sftp, CONTROL_ROOT + "/latest_handle.json")
        if previous_handle is None:
            previous_handle = _remote_json(sftp, CONTROL_ROOT + "/submit_handle.json")
    if not previous_handle or not previous_handle.get("job_id"):
        raise RuntimeError("No previous confirmed handle is available to resume")

    sequence = int(previous_handle.get("resume_sequence", 0)) + 1
    job_name = f"{JOB_NAME}r{sequence}"
    intent_path = CONTROL_ROOT + f"/resume_intent_{sequence}.json"
    handle_path = CONTROL_ROOT + f"/resume_handle_{sequence}.json"
    with remote.sftp_client(transport) as sftp:
        intent = _remote_json(sftp, intent_path)
        handle = _remote_json(sftp, handle_path)

    matches = _query_matches(transport, job_name)
    if len(matches) > 1:
        raise RuntimeError("More than one matching resume job exists; reconcile without submitting")
    if matches:
        job_id = matches[0]
        if (intent or {}).get("job_id") not in (None, job_id) or (handle or {}).get("job_id") not in (None, job_id):
            raise RuntimeError("Resume PBS history conflicts with its durable intent")
        recovered = {
            **(intent or {}), "job_name": job_name, "job_id": job_id,
            "resume_sequence": sequence, "status": "confirmed" if intent else "recovered",
        }
        with remote.sftp_client(transport) as sftp:
            _save_remote_json(sftp, intent_path, recovered)
            _save_remote_json(sftp, handle_path, recovered)
            _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", recovered)
        return recovered
    if intent or handle:
        raise RuntimeError("A resume intent exists without a recoverable PBS job; no second qsub will be issued")

    previous_status = scheduler(transport, previous_handle["job_id"])
    if (previous_status.get("job_state") not in ("F", "X")
            or previous_status.get("Exit_status") != "0"
            or previous_status.get("exit_code.txt") != "0"
            or previous_status.get("PIPELINE_CHECKPOINTED") is not True
            or previous_status.get("PIPELINE_COMPLETE") is not False
            or not isinstance(previous_status.get("summary.json"), dict)
            or previous_status["summary.json"].get("status") != "checkpointed"
            or previous_status["summary.json"].get("complete") is not False):
        raise RuntimeError("Resume requires a terminal, zero-exit PIPELINE_CHECKPOINTED job")

    intent = {
        "job_name": job_name,
        "script": ROOT + "/run.pbs",
        "remote_root": ROOT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": 4,
        "walltime": "24:00:00",
        "episodes": 560,
        "workers": 4,
        "variants_per_worker": 140,
        "resume_sequence": sequence,
        "resumes_job_id": previous_handle["job_id"],
        "status": "intent_before_qsub",
    }
    with remote.sftp_client(transport) as sftp:
        _save_remote_json(sftp, intent_path, intent)
    command = (
        "qsub -N " + shlex.quote(job_name)
        + " -o " + shlex.quote(ROOT + "/artifacts/pbs.out")
        + " " + shlex.quote(ROOT + "/run.pbs")
    )
    try:
        code, output = remote.command(transport, command, timeout=40)
    except Exception:
        raise
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="unknown_or_rejected", return_code=code, response=output[:1000])
        with remote.sftp_client(transport) as sftp:
            _save_remote_json(sftp, intent_path, intent)
        raise RuntimeError("Resume qsub outcome is unknown or rejected; do not repeat blindly")
    intent.update(status="confirmed", job_id=job_id)
    with remote.sftp_client(transport) as sftp:
        _save_remote_json(sftp, intent_path, intent)
        _save_remote_json(sftp, handle_path, intent)
        _save_remote_json(sftp, CONTROL_ROOT + "/latest_handle.json", intent)
    return intent


def scheduler(transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS handle")
    code, output = remote.command(
        transport, "qstat -xf " + shlex.quote(job_id), timeout=35
    )
    fields = {"job_id": job_id, "query_rc": code}
    for key in (
        "job_state", "Exit_status", "Job_Name", "queue", "exec_host", "comment",
        "resources_used.walltime", "Resource_List.ngpus", "Resource_List.ncpus",
        "Resource_List.mem",
    ):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    if fields["job_state"] in ("F", "X"):
        artifact_root = ROOT + "/artifacts/" + job_id
        with remote.sftp_client(transport) as sftp:
            for name in ("exit_code.txt", "summary.json"):
                try:
                    attrs = sftp.stat(artifact_root + "/" + name)
                except OSError as exc:
                    if getattr(exc, "errno", None) != 2:
                        raise
                    fields[name] = None
                    continue
                if attrs.st_size > remote.LIMIT:
                    fields[name] = "exceeds_small_file_limit"
                    continue
                if name == "summary.json":
                    fields[name] = _remote_json(sftp, artifact_root + "/" + name)
                else:
                    with sftp.file(artifact_root + "/" + name, "rb") as stream:
                        fields[name] = stream.read(remote.LIMIT + 1).decode("utf-8", errors="replace").strip()
            for marker in ("PIPELINE_COMPLETE", "PIPELINE_CHECKPOINTED"):
                try:
                    sftp.stat(artifact_root + "/" + marker)
                    fields[marker] = True
                except OSError as exc:
                    if getattr(exc, "errno", None) != 2:
                        raise
                    fields[marker] = False
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "resume", "status"))
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.action != "upload" and args.files:
        parser.error("--files is only supported for upload")
    if args.action != "status" and args.job:
        parser.error("--job is only supported for status")
    jump, transport = remote.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": upload(transport, args.files)}
        elif args.action == "submit":
            result = submit(transport)
        elif args.action == "resume":
            result = resume(transport)
        else:
            job_id = args.job
            if not job_id:
                with remote.sftp_client(transport) as sftp:
                    saved = _remote_json(sftp, CONTROL_ROOT + "/latest_handle.json")
                    if saved is None:
                        saved = _remote_json(sftp, CONTROL_ROOT + "/submit_handle.json")
                if not saved or not saved.get("job_id"):
                    raise RuntimeError("No confirmed PBS handle is saved; supply --job or reconcile the intent")
                job_id = saved["job_id"]
            result = scheduler(transport, job_id)
        if args.action in ("submit", "resume", "status"):
            destination = HERE / ("full_handle.json" if args.action != "status" else "last_scheduler_snapshot.json")
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(destination)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
