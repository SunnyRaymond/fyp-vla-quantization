"""Small-file control for the action-aware VQ PBS phases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009"
FILES = (
    "protocol.json", "experiment_protocol.json", "environment.sh", "prepare_inputs.py", "prepare.pbs", "run.pbs", "analysis.pbs",
    "run_experiment.py", "codebook.py", "test_codebook.py", "sensitivity.py",
    "sensitivity_adapter.py", "analysis.py",
)
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
SAFE_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")

sys.path.insert(0, str(HERE.parent / "fastwam-libero-plus-pilot"))
import remote  # noqa: E402

old = HERE.parent / "fastwam-a4-phases12" / "control.py"
spec = importlib.util.spec_from_file_location("previous_phase_control", old)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load trusted control helper: {old}")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def root_path() -> str:
    protocol_path = HERE / "protocol.json"
    if not protocol_path.is_file():
        return DEFAULT_ROOT
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    value = protocol.get("remote_root", DEFAULT_ROOT)
    if value != DEFAULT_ROOT:
        raise ValueError(f"Refusing unexpected remote root from protocol: {value!r}")
    return value


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def scheduler(transport, job: str) -> dict:
    if not HANDLE.fullmatch(job):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, f"qstat -xf {shlex.quote(job)}", timeout=35)
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "exec_host", "comment", "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return {"job_id": job, "query_rc": code, **fields}


def read_small_json(transport, path: str) -> dict:
    if transport is None:
        command = (f"size=$(wc -c < {shlex.quote(path)}) && test \"$size\" -le {remote.LIMIT} "
                   f"&& cat {shlex.quote(path)}")
        code, output = remote.command(None, command, timeout=25)
        if code:
            raise RuntimeError(f"Unable to read bounded remote summary: {path}")
        raw = output.encode("utf-8")
    else:
        with remote.sftp_client(transport) as sftp:
            if sftp.stat(path).st_size > remote.LIMIT:
                raise RuntimeError(f"Refusing summary larger than 256 KiB: {path}")
            with sftp.file(path, "rb") as stream:
                raw = stream.read(remote.LIMIT + 1)
    if len(raw) > remote.LIMIT:
        raise RuntimeError(f"Refusing summary larger than 256 KiB: {path}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def verified_success(transport, phase: str) -> dict:
    handle_path = HERE / f"{phase}_handle.json"
    receipt = json.loads(handle_path.read_text(encoding="utf-8"))
    observed = scheduler(transport, receipt["job_id"])
    if observed["query_rc"] or observed["job_state"] not in ("F", "X") or observed["Exit_status"] != "0":
        raise RuntimeError(f"{phase} is not a verified successful terminal job: {observed}")

    root = root_path()
    artifact = f"{root}/artifacts/{receipt['job_id']}"
    code, _ = remote.command(
        transport,
        f"test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')} && "
        f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0",
        timeout=25,
    )
    if code:
        raise RuntimeError(f"{phase} completion receipt is missing")
    if phase == "prepare":
        summary = read_small_json(transport, f"{root}/prepare_summary.json")
        required = (summary.get("status") == "complete" and summary.get("cases") == 26
                    and summary.get("selection_cases") == 4 and summary.get("test_cases") == 22
                    and summary.get("old_variant_exclusion_passed") is True)
        if not required:
            raise RuntimeError(f"Input preparation summary is incomplete: {summary}")
    elif phase in ("preflight", "full"):
        summary = read_small_json(transport, f"{root}/results/{phase}/summary.json")
        if summary.get("status") != "complete":
            raise RuntimeError(f"{phase} summary does not report complete: {summary}")
    else:
        summary = read_small_json(transport, f"{root}/results/analysis/summary.json")
        if summary.get("status") != "complete":
            raise RuntimeError(f"Analysis summary does not report complete: {summary}")
    return {**observed, "summary": summary}


def submit(transport, phase: str, attempt: str, queue: str, project: str) -> None:
    prerequisites = {"preflight": "prepare", "full": "preflight", "analysis": "full"}
    if phase in prerequisites:
        verified_success(transport, prerequisites[phase])

    tag = {"prepare": "p", "preflight": "r", "full": "f", "analysis": "a"}[phase]
    name = f"fwva1009{tag}{attempt}"
    intent_path = HERE / f"{phase}_intent_{attempt}.json"
    receipt = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.exists() else {}
    if receipt.get("job_id"):
        save(HERE / f"{phase}_handle.json", receipt)
        print("CONFIRMED_HANDLE", receipt["job_id"])
        return

    code, output = remote.command(
        transport, f"qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1", timeout=35
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS history unavailable; submission not attempted")
    matches = output.split()
    if matches:
        if len(matches) != 1 or not HANDLE.fullmatch(matches[0]):
            raise RuntimeError("Ambiguous PBS submission history; submission not attempted")
        receipt.update(job_id=matches[0], status="recovered", phase=phase, job_name=name)
    else:
        if receipt:
            raise RuntimeError("Unconfirmed durable intent exists; recover it before another attempt")
        receipt = {
            "phase": phase, "attempt": attempt, "job_name": name,
            "created_utc": datetime.now(timezone.utc).isoformat(), "status": "intent_before_qsub",
        }
        resources = {
            "prepare": ("select=1:ncpus=4:mem=16gb", "01:00:00", "prepare.pbs"),
            "preflight": ("select=1:ncpus=16:mem=110gb:ngpus=1", "02:00:00", "run.pbs"),
            "full": ("select=1:ncpus=16:mem=110gb:ngpus=1", "08:00:00", "run.pbs"),
            "analysis": ("select=1:ncpus=4:mem=16gb", "00:15:00", "analysis.pbs"),
        }
        select, walltime, script = resources[phase]
        variables = f"FW_PHASE={phase}"
        if phase == "analysis":
            variables += f",FW_FULL_JOB={json.loads((HERE / 'full_handle.json').read_text(encoding='utf-8'))['job_id']}"
        receipt.update(queue=queue, project=project, script=script, select=select,
                       walltime=walltime, variables=variables)
        save(intent_path, receipt)
        command = (
            f"qsub -q {shlex.quote(queue)} -P {shlex.quote(project)} "
            f"-N {shlex.quote(name)} -l {shlex.quote(select)} "
            f"-l walltime={walltime} -v {shlex.quote(variables)} {shlex.quote(root_path() + '/' + script)}"
        )
        code, output = remote.command(transport, command, timeout=40)
        if code or not HANDLE.fullmatch(output.strip()):
            receipt.update(status="unknown_or_rejected", message=output[:2000])
            save(intent_path, receipt)
            raise RuntimeError("qsub outcome unknown/rejected; do not repeat this attempt blindly")
        receipt.update(job_id=output.strip(), status="confirmed")
    save(intent_path, receipt)
    save(HERE / f"{phase}_handle.json", receipt)
    handles_path = HERE / "handles.json"
    handles = json.loads(handles_path.read_text(encoding="utf-8")) if handles_path.exists() else {}
    handles[phase] = receipt
    save(handles_path, handles)
    print("CONFIRMED_HANDLE", receipt["job_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "pull"))
    parser.add_argument("--phase", choices=("prepare", "preflight", "full", "analysis"))
    parser.add_argument("--attempt", default="a")
    parser.add_argument("--job")
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--queue", default=os.environ.get("FW_QUEUE", "normal"))
    parser.add_argument("--project", default=os.environ.get("FW_PROJECT", "personal-yguo017"))
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9]{0,5}", args.attempt):
        parser.error("Use a short lowercase attempt")
    if args.action in ("submit", "verify") and not args.phase:
        parser.error("--phase required")
    if args.action == "submit" and (not SAFE_TOKEN.fullmatch(args.queue) or not SAFE_TOKEN.fullmatch(args.project)):
        parser.error("Queue and project must be simple PBS names")
    if args.action == "pull" and not args.files:
        parser.error("--files required")
    if args.native and args.action not in ("submit", "status", "verify"):
        parser.error("--native supports lightweight command actions only")

    jump = transport = None
    try:
        if args.native:
            remote.command = previous.native_command
        else:
            jump, transport = remote.connect()
        if args.action == "upload":
            names = args.files or [name for name in FILES if (HERE / name).is_file()]
            if any(name not in FILES for name in names):
                raise ValueError("Unknown control file")
            root = root_path()
            code, output = remote.command(transport, f"mkdir -p {shlex.quote(root)}", timeout=25)
            if code:
                raise RuntimeError(f"Unable to create control directory: {output}")
            with remote.sftp_client(transport) as sftp:
                for name in names:
                    path = HERE / name
                    if not path.is_file() or path.parent.resolve() != HERE.resolve():
                        raise FileNotFoundError(f"Task control file is unavailable: {name}")
                    data = path.read_bytes()
                    if len(data) > remote.LIMIT:
                        raise ValueError(f"Control file exceeds 256 KiB: {name}")
                    target = f"{root}/{name}"
                    try:
                        existing_size = sftp.stat(target).st_size
                    except OSError as exc:
                        if getattr(exc, "errno", None) != errno.ENOENT:
                            raise
                        existing_size = None
                    if existing_size is not None:
                        if existing_size > remote.LIMIT:
                            raise RuntimeError(f"Existing remote control path is too large to compare: {name}")
                        with sftp.file(target, "rb") as existing:
                            old_data = existing.read(remote.LIMIT + 1)
                        if len(old_data) > remote.LIMIT:
                            raise RuntimeError(f"Existing remote control path grew beyond the small-file limit: {name}")
                        if old_data != data:
                            raise FileExistsError(f"Refusing to replace different remote control content: {name}")
                        print("ALREADY_PRESENT", name, len(data))
                        continue
                    with sftp.file(target, "wb") as stream:
                        stream.write(data)
                    print("UPLOADED", name, len(data))
        elif args.action == "submit":
            submit(transport, args.phase, args.attempt, args.queue, args.project)
        elif args.action == "verify":
            evidence = verified_success(transport, args.phase)
            save(HERE / f"{args.phase}_terminal_evidence.json", evidence)
            print(json.dumps(evidence))
        elif args.action == "status":
            job = args.job
            if not job:
                if not args.phase:
                    parser.error("--phase or --job required")
                job = json.loads((HERE / f"{args.phase}_handle.json").read_text(encoding="utf-8"))["job_id"]
            observed = scheduler(transport, job)
            save(HERE / "last_scheduler_snapshot.json", observed)
            print(json.dumps(observed))
            if observed["job_state"] in ("R", "F", "X"):
                path = f"{root_path()}/artifacts/{job}/job.log"
                code, output = remote.command(transport, f"tail -c 6000 {shlex.quote(path)}", timeout=25)
                print("LOG_TAIL_RC", code)
                print(output)
        else:
            root = root_path()
            with remote.sftp_client(transport) as sftp:
                for name in args.files:
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Invalid artifact path")
                    source = f"{root}/{relative.as_posix()}"
                    size = sftp.stat(source).st_size
                    if size > remote.LIMIT:
                        raise ValueError("Only small summary/control files may be pulled")
                    with sftp.file(source, "rb") as stream:
                        data = stream.read(remote.LIMIT + 1)
                    if len(data) > remote.LIMIT:
                        raise ValueError("Remote file grew beyond the small summary/control limit")
                    target = HERE / "retrieved" / relative
                    if target.exists():
                        raise FileExistsError(f"Refusing to overwrite retrieved file: {target}")
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
