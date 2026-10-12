"""Bounded small-file control for the proxy diagnostic PBS phases."""
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
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-proxy-diagnostic-20261010"
SOURCE_ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009"
SOURCE_JOB = "25743693.pbs101"
LIMIT = 256 * 1024
PHASES = ("original", "preflight", "full")
FILES = (
    "protocol.json", "control.py", "environment.sh", "original.pbs", "gpu.pbs",
    "test_original_projection.py", "original_projection.py", "test_sampling.py",
    "run_diagnostic.py", "sensitivity.py",
)
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
SAFE_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")

sys.path.insert(0, str(HERE.parent / "fastwam-libero-plus-pilot"))
import remote  # noqa: E402

previous_path = HERE.parent / "fastwam-a4-phases12" / "control.py"
spec = importlib.util.spec_from_file_location("proxy_previous_control", previous_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load trusted native SSH helper: {previous_path}")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("protocol.json must be a JSON object")
    if value.get("source_root") != SOURCE_ROOT or value.get("source_full_job") != SOURCE_JOB:
        raise RuntimeError("protocol.json has an unexpected source campaign or source full job")
    if value.get("remote_root", ROOT) != ROOT:
        raise RuntimeError("Refusing an unexpected remote root in protocol.json")
    return value


def scheduler(transport, job: str) -> dict:
    if not HANDLE.fullmatch(job):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, f"qstat -xf {shlex.quote(job)}", timeout=35)
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "exec_host", "comment", "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return {"job_id": job, "query_rc": code, **fields}


def read_remote_json(transport, path: str) -> dict:
    if transport is None:
        code, output = remote.command(
            None,
            f"size=$(wc -c < {shlex.quote(path)}) && test \"$size\" -le {LIMIT} && cat {shlex.quote(path)}",
            timeout=25,
        )
        if code:
            raise RuntimeError(f"Unable to read bounded remote JSON: {path}")
        raw = output.encode("utf-8")
    else:
        with remote.sftp_client(transport) as sftp:
            size = sftp.stat(path).st_size
            if size > LIMIT:
                raise RuntimeError(f"Refusing remote JSON larger than 256 KiB: {path}")
            with sftp.file(path, "rb") as stream:
                raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise RuntimeError(f"Remote JSON grew beyond 256 KiB: {path}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def require_preflight_coverage(summary: dict) -> None:
    counts = summary.get("probe_counts")
    rows = summary.get("row_sets")
    if (summary.get("target_linears") != 614 or summary.get("case_ids") != [0]
            or summary.get("original_proxy_reproduction_passed") is not True
            or not isinstance(counts, dict)
            or (counts.get("old"), counts.get("new"), counts.get("total")) != (4, 16, 20)
            or not isinstance(rows, dict)
            or (rows.get("old"), rows.get("new")) != (2, 2)):
        raise RuntimeError(
            "Preflight coverage is incomplete; expected 614 targets, case 0, "
            "20 probes (4 old + 16 new), and two row sets (2 old + 2 new)"
        )


def verified_success(transport, phase: str) -> dict:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    handle_path = HERE / f"{phase}_handle.json"
    if not handle_path.is_file() or handle_path.stat().st_size > LIMIT:
        raise RuntimeError(f"No bounded {phase} handle receipt is available")
    receipt = json.loads(handle_path.read_text(encoding="utf-8"))
    job = receipt.get("job_id")
    observed = scheduler(transport, job)
    if (observed.get("query_rc") != 0 or observed.get("job_state") not in ("F", "X")
            or observed.get("Exit_status") != "0"):
        raise RuntimeError(f"{phase} is not a verified successful terminal job: {observed}")

    artifact = f"{ROOT}/artifacts/{job}"
    code, _ = remote.command(
        transport,
        f"test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')} && "
        f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0",
        timeout=25,
    )
    if code:
        raise RuntimeError(f"{phase} success marker or zero exit receipt is missing")
    summary = read_remote_json(transport, artifact + "/summary.json")
    if (summary.get("status") != "complete" or summary.get("phase") != phase
            or summary.get("job_id") != job):
        raise RuntimeError(f"{phase} summary does not match its verified job: {summary}")
    if phase == "preflight":
        require_preflight_coverage(summary)
    evidence = {**observed, "summary": summary, "success_marker": artifact + "/PIPELINE_COMPLETE"}
    save(HERE / f"{phase}_terminal_evidence.json", evidence)
    return evidence


def submit(transport, phase: str, attempt: str, queue: str, project: str) -> None:
    protocol()

    tag = {"original": "o", "preflight": "p", "full": "f"}[phase]
    name = f"fwvp1010{tag}{attempt}"
    intent_path = HERE / f"{phase}_intent_{attempt}.json"
    handle_path = HERE / f"{phase}_handle.json"
    receipt = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.exists() else {}
    if receipt.get("job_id"):
        save(handle_path, receipt)
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
        resource = {
            "original": ("select=1:ncpus=4:mem=32gb", "00:20:00", "original.pbs"),
            "preflight": ("select=1:ncpus=16:mem=110gb:ngpus=1", "02:00:00", "gpu.pbs"),
            "full": ("select=1:ncpus=16:mem=110gb:ngpus=1", "12:00:00", "gpu.pbs"),
        }[phase]
        select, walltime, script = resource
        variables = f"FW_PHASE={phase}" if script == "gpu.pbs" else None
        receipt = {
            "phase": phase, "attempt": attempt, "job_name": name,
            "created_utc": datetime.now(timezone.utc).isoformat(), "status": "intent_before_qsub",
            "queue": queue, "project": project, "select": select,
            "walltime": walltime, "script": script, "variables": variables,
        }
        save(intent_path, receipt)
        variable_arg = f"-v {shlex.quote(variables)} " if variables else ""
        command = (
            f"qsub -q {shlex.quote(queue)} -P {shlex.quote(project)} "
            f"-N {shlex.quote(name)} -l {shlex.quote(select)} -l walltime={walltime} "
            f"{variable_arg}{shlex.quote(ROOT + '/' + script)}"
        )
        code, output = remote.command(transport, command, timeout=40)
        if code or not HANDLE.fullmatch(output.strip()):
            receipt.update(status="unknown_or_rejected", message=output[:2000])
            save(intent_path, receipt)
            raise RuntimeError("qsub outcome unknown/rejected; recover this attempt before considering another")
        receipt.update(job_id=output.strip(), status="confirmed")

    save(intent_path, receipt)
    save(handle_path, receipt)
    print("CONFIRMED_HANDLE", receipt["job_id"])


def upload(transport, names: list[str] | None) -> None:
    selected = names or [name for name in FILES if (HERE / name).is_file()]
    if any(name not in FILES for name in selected):
        raise ValueError("Unknown task control file")
    code, output = remote.command(transport, "mkdir -p " + shlex.quote(ROOT), timeout=25)
    if code:
        raise RuntimeError(f"Unable to create control directory: {output}")
    with remote.sftp_client(transport) as sftp:
        for name in selected:
            path = HERE / name
            if not path.is_file() or path.parent.resolve() != HERE.resolve():
                raise FileNotFoundError(f"Task control file is unavailable: {name}")
            data = path.read_bytes().replace(b"\r\n", b"\n")
            if len(data) > LIMIT:
                raise ValueError(f"Control file exceeds 256 KiB: {name}")
            target = f"{ROOT}/{name}"
            try:
                size = sftp.stat(target).st_size
            except OSError as exc:
                if getattr(exc, "errno", None) != errno.ENOENT:
                    raise
                size = None
            if size is not None:
                if size > LIMIT:
                    raise RuntimeError(f"Existing remote control file is too large to compare: {name}")
                with sftp.file(target, "rb") as stream:
                    existing = stream.read(LIMIT + 1)
                if len(existing) > LIMIT:
                    raise RuntimeError(f"Remote control file grew beyond 256 KiB: {name}")
                if existing != data:
                    raise FileExistsError(f"Refusing to replace different remote content: {name}")
                print("ALREADY_PRESENT", name, len(data))
                continue
            with sftp.file(target, "wb") as stream:
                stream.write(data)
            print("UPLOADED", name, len(data))


def pull(transport, names: list[str]) -> None:
    with remote.sftp_client(transport) as sftp:
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid artifact path")
            source = f"{ROOT}/{relative.as_posix()}"
            size = sftp.stat(source).st_size
            if size > LIMIT:
                raise ValueError("Only control and summary files up to 256 KiB may be pulled")
            with sftp.file(source, "rb") as stream:
                data = stream.read(LIMIT + 1)
            if len(data) > LIMIT:
                raise ValueError("Remote file grew beyond 256 KiB")
            target = HERE / "retrieved" / relative
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite retrieved file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print("PULLED", relative.as_posix(), len(data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "pull"))
    parser.add_argument("--phase", choices=PHASES)
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
    if args.action == "pull" and not args.files:
        parser.error("--files required")
    if args.action == "status" and not args.job and not args.phase:
        parser.error("--job or --phase required")
    if args.action == "submit" and (not SAFE_TOKEN.fullmatch(args.queue) or not SAFE_TOKEN.fullmatch(args.project)):
        parser.error("Queue and project must be simple PBS names")
    if args.native and args.action not in ("submit", "status", "verify"):
        parser.error("--native supports only lightweight command actions")

    jump = transport = None
    try:
        if args.native:
            remote.command = previous.native_command
        else:
            jump, transport = remote.connect()
        if args.action == "upload":
            upload(transport, args.files)
        elif args.action == "submit":
            submit(transport, args.phase, args.attempt, args.queue, args.project)
        elif args.action == "verify":
            print(json.dumps(verified_success(transport, args.phase)))
        elif args.action == "status":
            job = args.job
            if not job:
                job_path = HERE / f"{args.phase}_handle.json"
                if not job_path.is_file():
                    raise FileNotFoundError(f"No confirmed {args.phase} handle is available")
                job = json.loads(job_path.read_text(encoding="utf-8"))["job_id"]
            observed = scheduler(transport, job)
            save(HERE / "last_scheduler_snapshot.json", observed)
            print(json.dumps(observed))
            if observed.get("job_state") in ("R", "F", "X"):
                path = f"{ROOT}/artifacts/{job}/job.log"
                code, output = remote.command(
                    transport,
                    f"test ! -f {shlex.quote(path)} || tail -c 6000 {shlex.quote(path)}",
                    timeout=25,
                )
                print("LOG_TAIL_RC", code)
                print(output)
        else:
            pull(transport, args.files)
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
