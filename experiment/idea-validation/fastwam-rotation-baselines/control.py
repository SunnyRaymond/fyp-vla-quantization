"""Bounded small-file PBS control with durable, non-retriable submission intent."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "fastwam-libero-plus-pilot"
OLD_CONTROL = HERE.parent / "fastwam-a4-phases12" / "control.py"
sys.path.insert(0, str(PILOT))
import remote

_spec = importlib.util.spec_from_file_location("rotation_previous_control", OLD_CONTROL)
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)

ROOT = "/scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008"
PHASES = {
    "preparation": {"job_name": "fwrot1008p1", "script": "prepare.pbs"},
    "evaluation": {"job_name": "fwrot1008e1", "script": "run.pbs"},
}
FILES = (
    "cohort.py", "test_cohort.py", "control.py", "calibration_inputs.py", "rotation.py", "test_rotation.py",
    "prepare.py", "test_prepare.py", "evaluate.py", "test_evaluation.py", "aggregate.py", "run.pbs",
    "prepare.pbs", "environment.sh", "protocol.json", "PREREG.zh.md",
)
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


def direct_native_command(_transport, command, timeout=60):
    """Reach the same pinned ASPIRE2A host without changing saved SSH config."""
    env = os.environ.copy()
    env.update(NSCC_CREDENTIAL_FILE=str(remote.PROJECT / "credentials.env"),
               SSH_ASKPASS=str(remote.PROJECT / "nscc-access" / "nscc-askpass.cmd"),
               SSH_ASKPASS_REQUIRE="force", DISPLAY="1")
    result = subprocess.run(
        [r"C:\Windows\System32\OpenSSH\ssh.exe", "-T", "-o", "ProxyJump=none",
         "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20",
         "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "ASPIRE2A", command],
        env=env, text=True, encoding="utf-8", capture_output=True, timeout=timeout + 30,
    )
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode, result.stdout


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def scheduler(transport, job: str) -> dict:
    if not HANDLE.fullmatch(job):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, f"qstat -xf {shlex.quote(job)}", timeout=35)
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "exec_host", "comment", "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return {"job_id": job, "query_rc": code, **fields}


def require_completed_preparation(transport) -> None:
    handle = json.loads((HERE / "submit_handle_preparation.json").read_text(encoding="utf-8"))
    job = handle["job_id"]
    observed = scheduler(transport, job)
    code, output = remote.command(
        transport,
        f"head -c {remote.LIMIT + 1} {shlex.quote(ROOT + '/results/PREPARED.json')}",
        timeout=25,
    )
    if code or len(output.encode("utf-8")) > remote.LIMIT:
        raise RuntimeError("Bounded preparation receipt is unavailable")
    prepared = json.loads(output)
    if (prepared.get("pbs_jobid") != job or prepared.get("ready") is not True
            or prepared.get("numeric_check", {}).get("passed") is not True):
        raise RuntimeError("Current preparation receipt has not passed the numerical checks")
    authorization = prepared.get("user_authorized_native_gate_waiver", {})
    if (authorization.get("approved") is True
            and authorization.get("source_job_id") == job
            and authorization.get("scope") == "four_gpu_full_evaluation"
            and observed.get("job_state") in ("F", "X")
            and observed.get("Exit_status") == "1"):
        print("USER_AUTHORIZED_NATIVE_GATE_WAIVER", job)
        return
    if (observed.get("job_state") not in ("F", "X") or observed.get("Exit_status") != "0"
            or prepared.get("native_int4_check", {}).get("passed") is not True):
        raise RuntimeError(f"Preparation job has not finished successfully: {observed}")
    code, _ = remote.command(transport, f"test -f {shlex.quote(ROOT + '/artifacts/' + job + '/PREPARATION_COMPLETE')}", timeout=25)
    if code:
        raise RuntimeError("Preparation completion marker is unavailable")


def submit(transport, phase: str) -> dict:
    settings = PHASES[phase]
    job_name = settings["job_name"]
    intent_path = HERE / f"submit_intent_{phase}.json"
    handle_path = HERE / f"submit_handle_{phase}.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.exists() else {}
    if intent.get("job_id"):
        if not HANDLE.fullmatch(intent["job_id"]):
            raise RuntimeError("Saved submission intent has an invalid job handle")
        save(handle_path, intent)
        save(HERE / "submit_handle.json", intent)
        print("CONFIRMED_HANDLE", intent["job_id"])
        return intent
    if phase == "evaluation":
        require_completed_preparation(transport)

    code, output = remote.command(
        transport,
        f"qselect -x -u yguo017 -N {shlex.quote(job_name)} 2>&1",
        timeout=35,
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS history unavailable; submission was not attempted")
    matches = output.split()
    if matches:
        if len(matches) != 1 or not HANDLE.fullmatch(matches[0]):
            raise RuntimeError("Ambiguous matching PBS history; submission was not attempted")
        intent.update(job_id=matches[0], job_name=job_name, phase=phase, status="recovered")
        save(intent_path, intent)
        save(handle_path, intent)
        save(HERE / "submit_handle.json", intent)
        print("RECOVERED_HANDLE", matches[0])
        return intent
    if intent:
        raise RuntimeError("An unresolved durable intent exists; no second qsub will be issued")

    intent = {
        "job_name": job_name,
        "phase": phase,
        "script": f"{ROOT}/{settings['script']}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "intent_before_qsub",
    }
    save(intent_path, intent)
    code, output = remote.command(
        transport,
        f"qsub -N {shlex.quote(job_name)} {shlex.quote(intent['script'])}",
        timeout=40,
    )
    job = output.strip()
    if code or not HANDLE.fullmatch(job):
        intent.update(status="unknown_or_rejected", response=output[:1000], return_code=code)
        save(intent_path, intent)
        raise RuntimeError("qsub outcome is unknown or rejected; do not repeat blindly")
    intent.update(job_id=job, status="confirmed")
    save(intent_path, intent)
    save(handle_path, intent)
    save(HERE / "submit_handle.json", intent)
    print("CONFIRMED_HANDLE", job)
    return intent


def pull_allowed(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError("Use a relative POSIX artifact path")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Invalid artifact path")
    if not all(SAFE_PART.fullmatch(part) for part in path.parts):
        raise ValueError("Invalid artifact path")
    basename = path.name.lower()
    is_summary = basename in {"summary.json", "per_worker_progress.json"}
    is_preparation = basename in {"preparation_progress.json", "prepared.json"}
    is_worker_progress = re.fullmatch(r"worker[_-]?[0-3][_-]progress\.json", basename) is not None
    is_worker_directory_progress = (
        len(path.parts) >= 2
        and re.fullmatch(r"worker[_-]?[0-3]", path.parts[-2].lower()) is not None
        and basename == "progress.json"
    )
    is_receipt = "receipt" in basename and basename.endswith(".json")
    if not (is_summary or is_preparation or is_worker_progress or is_worker_directory_progress or is_receipt):
        raise ValueError("Pull is limited to summaries, per-worker progress, and JSON receipts")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "pull"))
    parser.add_argument("--job")
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Use the same strictly verified SSH host without the jump route")
    parser.add_argument("--phase", choices=tuple(PHASES), help="Submission phase; default is preparation")
    args = parser.parse_args()
    if args.direct and not args.native:
        parser.error("--direct requires --native")
    if args.native and args.action not in ("submit", "status"):
        parser.error("--native is supported for lightweight submit/status commands only")
    jump = transport = None
    try:
        if args.native:
            remote.command = direct_native_command if args.direct else previous.native_command
        else:
            jump, transport = remote.connect()

        if args.action == "upload":
            code, output = remote.command(transport, f"mkdir -p {shlex.quote(ROOT)}", timeout=25)
            if code:
                raise RuntimeError(f"Unable to create remote control directory: {output[-500:]}")
            names = args.files or list(FILES)
            with remote.sftp_client(transport) as sftp:
                for name in names:
                    if name not in FILES:
                        raise ValueError(f"Unknown control file: {name}")
                    path = HERE / name
                    data = path.read_bytes().replace(b"\r\n", b"\n")
                    if len(data) > remote.LIMIT:
                        raise ValueError(f"Control file exceeds small-transfer limit: {name}")
                    data.decode("utf-8")
                    with sftp.file(f"{ROOT}/{name}", "wb") as stream:
                        stream.write(data)
                    print("UPLOADED", name, len(data))
        elif args.action == "submit":
            submit(transport, args.phase or "preparation")
        elif args.action == "status":
            if args.job:
                job = args.job
            else:
                handle_path = HERE / (f"submit_handle_{args.phase}.json" if args.phase else "submit_handle.json")
                if not handle_path.exists():
                    raise RuntimeError("No saved handle; provide --job or resolve the submission intent")
                job = json.loads(handle_path.read_text(encoding="utf-8"))["job_id"]
            observed = scheduler(transport, job)
            save(HERE / "last_scheduler_snapshot.json", observed)
            print(json.dumps(observed, ensure_ascii=False, separators=(",", ":")))
        else:
            if not args.files:
                parser.error("pull requires --files")
            with remote.sftp_client(transport) as sftp:
                for name in args.files:
                    relative = pull_allowed(name)
                    source = f"{ROOT}/{relative.as_posix()}"
                    if sftp.stat(source).st_size > remote.LIMIT:
                        raise ValueError("Only small summary, progress, and receipt files may be pulled")
                    with sftp.file(source, "rb") as stream:
                        data = stream.read(remote.LIMIT + 1)
                    if len(data) > remote.LIMIT:
                        raise ValueError("Artifact exceeded small-transfer limit while reading")
                    target = HERE.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    print("PULLED", relative.as_posix(), len(data))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
