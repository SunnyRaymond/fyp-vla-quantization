"""Small-file PBS control for the microscaling VQ campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-microscale-20261011"
FILES = (
    "protocol.json", "control.py", "environment.sh", "gpu.pbs",
    "micro_codebook.py", "run_microscale.py",
)
spec = importlib.util.spec_from_file_location(
    "microscale_trusted_control",
    HERE.parent / "fastwam-vq-proxy-diagnostic" / "control.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the trusted PBS control helper")
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
trusted.HERE, trusted.ROOT, trusted.FILES = HERE, ROOT, FILES
remote, save, scheduler, read_remote_json = (
    trusted.remote, trusted.save, trusted.scheduler, trusted.read_remote_json
)


def protocol() -> dict:
    return trusted.protocol()


def read_local_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > trusted.LIMIT:
        raise RuntimeError(f"Missing or oversized local control file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path.name}")
    return value


def current_job() -> str:
    job = read_local_json(HERE / "full_handle.json").get("job_id")
    if not isinstance(job, str) or not trusted.HANDLE.fullmatch(job):
        raise RuntimeError("No valid confirmed full-job handle is available")
    return job


def query_named_submission(transport, name: str) -> list[str]:
    code, output = remote.command(
        transport,
        f"qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1",
        timeout=35,
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS submission history is unavailable")
    matches = output.split()
    if any(not trusted.HANDLE.fullmatch(job) for job in matches):
        raise RuntimeError("PBS returned an invalid handle for the named submission")
    return matches


def require_previous_terminal(transport) -> None:
    handle_path = HERE / "full_handle.json"
    if not handle_path.exists():
        return
    previous = current_job()
    observed = scheduler(transport, previous)
    if observed.get("query_rc") != 0 or observed.get("job_state") not in ("F", "X"):
        raise RuntimeError(
            "The previous full attempt is not confirmed terminal; refusing a new attempt"
        )


def submit(transport, attempt: str) -> None:
    protocol()
    if not re.fullmatch(r"[a-z][a-z0-9]{0,5}", attempt):
        raise ValueError("Use a short lowercase attempt label")
    name = f"fwvqm1011{attempt}"
    intent = HERE / f"full_intent_{attempt}.json"
    handle = HERE / "full_handle.json"

    if intent.exists():
        receipt = read_local_json(intent)
        if receipt.get("attempt") != attempt or receipt.get("job_name") != name:
            raise RuntimeError("Existing durable intent does not match this attempt")
        if receipt.get("job_id"):
            if (receipt.get("job_name") != name
                    or not trusted.HANDLE.fullmatch(str(receipt["job_id"]))):
                raise RuntimeError("Existing attempt receipt does not match this named attempt")
            save(handle, receipt)
            print("CONFIRMED_HANDLE", receipt["job_id"])
            return
        matches = query_named_submission(transport, name)
        if len(matches) == 1:
            receipt.update(job_id=matches[0], status="recovered")
        elif matches:
            raise RuntimeError("Ambiguous PBS history for this attempt")
        else:
            raise RuntimeError("Unconfirmed intent exists; inspect it before any retry")
    else:
        for pending in HERE.glob("full_intent_*.json"):
            older = read_local_json(pending)
            if not older.get("job_id"):
                raise RuntimeError("An earlier attempt has an unresolved submission intent")
        require_previous_terminal(transport)
        matches = query_named_submission(transport, name)
        if len(matches) == 1:
            receipt = {
                "job_id": matches[0], "status": "recovered", "attempt": attempt,
                "job_name": name,
            }
        elif matches:
            raise RuntimeError("Ambiguous PBS history for this attempt")
        else:
            receipt = {
                "status": "intent_before_qsub", "attempt": attempt,
                "job_name": name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "walltime": "02:00:00", "queue": "normal",
                "project": "personal-yguo017",
                "select": "select=1:ncpus=16:mem=110gb:ngpus=1",
                "remote_root": ROOT,
            }
            save(intent, receipt)
            code, output = remote.command(
                transport,
                f"qsub -N {shlex.quote(name)} {shlex.quote(ROOT + '/gpu.pbs')}",
                timeout=40,
            )
            if code or not trusted.HANDLE.fullmatch(output.strip()):
                receipt.update(status="unknown_or_rejected")
                save(intent, receipt)
                raise RuntimeError("qsub outcome is unknown or rejected; do not retry automatically")
            receipt.update(job_id=output.strip(), status="confirmed")

    save(intent, receipt)
    save(handle, receipt)
    print("CONFIRMED_HANDLE", receipt["job_id"])


def status(transport, job: str) -> None:
    observed = scheduler(transport, job)
    save(HERE / "last_scheduler_snapshot.json", observed)
    artifact = f"{ROOT}/artifacts/{job}"
    summary_path = artifact + "/summary.json"
    code, _ = remote.command(
        transport, f"test -f {shlex.quote(summary_path)}", timeout=25
    )
    if code != 0:
        summary_path = ROOT + '/results/full/progress.json'
        code, _ = remote.command(transport, f'test -f {shlex.quote(summary_path)}', timeout=25)
    progress = read_remote_json(transport, summary_path) if code == 0 else None
    log_path = artifact + "/job.log"
    code, output = remote.command(
        transport,
        f"if test -f {shlex.quote(log_path)}; then tail -c 7000 {shlex.quote(log_path)}; fi",
        timeout=25,
    )
    print(json.dumps({"scheduler": observed, "progress": progress}))
    print("LOG_TAIL_RC", code)
    print(output)


def require_complete_coverage(summary: dict) -> None:
    selection = summary.get("selection")
    test = summary.get("test")
    if (summary.get("target_linears") != 614
            or summary.get("fitted_linears") != 614
            or not isinstance(selection, dict)
            or selection.get("context_count") != 8
            or selection.get("case_count") != 4
            or not isinstance(test, dict)
            or test.get("context_count") != 44
            or test.get("case_count") != 22
            or summary.get("new_queries") != 52):
        raise RuntimeError("Complete microscaling coverage is incomplete")


def verify(transport, job: str) -> dict:
    observed = scheduler(transport, job)
    if (observed.get("query_rc") != 0
            or observed.get("job_state") not in ("F", "X")
            or observed.get("Exit_status") != "0"):
        raise RuntimeError(f"Not a successful terminal job: {observed}")

    artifact = f"{ROOT}/artifacts/{job}"
    marker_command = (
        f"if test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')} "
        f"&& test -f {shlex.quote(artifact + '/PIPELINE_CHECKPOINTED')}; then printf both; "
        f"elif test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')}; then printf complete; "
        f"elif test -f {shlex.quote(artifact + '/PIPELINE_CHECKPOINTED')}; then printf checkpointed; "
        "else printf missing; fi"
    )
    code, marker = remote.command(transport, marker_command, timeout=25)
    if code or marker.strip() not in ("complete", "checkpointed"):
        raise RuntimeError("Missing or conflicting pipeline completion markers")
    code, _ = remote.command(
        transport,
        f"test -f {shlex.quote(artifact + '/exit_code.txt')} && "
        f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0",
        timeout=25,
    )
    if code:
        raise RuntimeError("Missing zero exit receipt")

    summary = read_remote_json(transport, artifact + "/summary.json")
    if summary.get("job_id") != job or summary.get("phase") != "full":
        raise RuntimeError("Summary job identity or phase does not match this full attempt")
    if marker == "complete":
        if summary.get("status") != "complete":
            raise RuntimeError("Complete marker does not match summary status")
        require_complete_coverage(summary)
    elif summary.get("status") != "checkpointed":
        raise RuntimeError("Checkpoint marker does not match summary status")

    evidence = {
        **observed, "summary": summary, "verified_status": marker,
        "success_marker": f"{artifact}/PIPELINE_{'COMPLETE' if marker == 'complete' else 'CHECKPOINTED'}",
    }
    save(HERE / "full_terminal_evidence.json", evidence)
    print(json.dumps(evidence))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "pull"))
    parser.add_argument("--attempt", default="a")
    parser.add_argument("--job")
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    if args.action == "pull" and not args.files:
        parser.error("--files required for pull")
    if args.action == "submit" and not re.fullmatch(r"[a-z][a-z0-9]{0,5}", args.attempt):
        parser.error("Use a short lowercase attempt label")

    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        if args.action == "upload":
            protocol()
            trusted.upload(transport, args.files)
        elif args.action == "submit":
            submit(transport, args.attempt)
        elif args.action in ("status", "verify"):
            job = args.job or current_job()
            if not trusted.HANDLE.fullmatch(job):
                parser.error("Invalid PBS job handle")
            if args.action == "status":
                status(transport, job)
            else:
                protocol()
                verify(transport, job)
        else:
            trusted.pull(transport, args.files)
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
