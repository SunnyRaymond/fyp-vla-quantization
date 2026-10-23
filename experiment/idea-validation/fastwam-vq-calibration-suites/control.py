"""Bounded PBS control for the original-LIBERO calibration-suites campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-calibration-suites-20261013"
SOURCE_ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009"
SOURCE_JOB = "25743693.pbs101"
LIMIT = 256 * 1024
FILES = (
    "protocol.json", "control.py", "environment.sh", "cpu.pbs", "gpu.pbs",
    "prepare_calibration.py", "run_calibration.py", "micro_codebook.py",
)
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
ATTEMPT = re.compile(r"^[a-z][a-z0-9]{0,5}$")
GPU_COHORTS = ("teacher", "c8", "c80")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

spec = importlib.util.spec_from_file_location(
    "calibration_trusted_control",
    HERE.parent / "fastwam-vq-proxy-diagnostic" / "control.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load trusted PBS control helper")
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
trusted.HERE, trusted.ROOT, trusted.FILES = HERE, ROOT, FILES
remote = trusted.remote


def read_local_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise RuntimeError(f"Missing or oversized local control file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path.name}")
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


def protocol() -> dict:
    value = read_local_json(HERE / "protocol.json")
    roots = value.get("roots", {})
    if (value.get("protocol") != "fastwam-vq-calibration-suites-v1"
            or value.get("remote_root") != ROOT
            or value.get("source_root") != SOURCE_ROOT
            or value.get("source_full_job") != SOURCE_JOB
            or roots.get("base") != "/scratch/users/ntu/yguo017/fastwam-smoke"
            or value.get("walltime") != "01:59:00"
            or value.get("soft_deadline_seconds") != 6300):
        raise RuntimeError("protocol.json does not match the frozen calibration-suites roots or limits")
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


def artifact_path(job: str) -> str:
    return f"{ROOT}/artifacts/{job}"


def require_terminal_success(transport, job: str) -> dict:
    observed = scheduler(transport, job)
    if (observed.get("query_rc") != 0 or observed.get("job_state") not in ("F", "X")
            or observed.get("Exit_status") != "0"):
        raise RuntimeError(f"Job is not a verified successful terminal job: {observed}")
    return observed


def artifact_marker(transport, job: str) -> str:
    artifact = artifact_path(job)
    command = (
        f"if test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')} "
        f"&& test -f {shlex.quote(artifact + '/PIPELINE_CHECKPOINTED')}; then printf conflict; "
        f"elif test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')}; then printf complete; "
        f"elif test -f {shlex.quote(artifact + '/PIPELINE_CHECKPOINTED')}; then printf checkpointed; "
        "else printf missing; fi"
    )
    code, marker = remote.command(transport, command, timeout=25)
    if code or marker.strip() not in ("complete", "checkpointed"):
        raise RuntimeError(f"Missing or conflicting artifact marker for {job}")
    code, _ = remote.command(
        transport,
        f"test -f {shlex.quote(artifact + '/exit_code.txt')} && "
        f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0",
        timeout=25,
    )
    if code:
        raise RuntimeError(f"Missing zero exit receipt for {job}")
    return marker.strip()


def artifact_receipt(transport, job: str) -> tuple[dict, str]:
    artifact = artifact_path(job)
    marker = artifact_marker(transport, job)
    summary = read_remote_json(transport, artifact + "/summary.json")
    if summary.get("status") != marker:
        raise RuntimeError(f"Artifact marker and summary status disagree for {job}")
    return summary, marker


def verify_inventory(transport, job: str) -> dict:
    observed = require_terminal_success(transport, job)
    marker = artifact_marker(transport, job)
    if marker != "complete":
        raise RuntimeError(f"Inventory artifact is not complete: {job}")
    inventory = read_remote_json(transport, ROOT + "/source_inventory.json")
    suites = inventory.get("suites")
    libraries = inventory.get("libraries")
    if (inventory.get("status") != "inventory_complete" or inventory.get("job_id") != job
            or not isinstance(suites, dict) or set(suites) != set(SUITES)
            or inventory.get("protocol") != "fastwam-vq-calibration-suites-v1"
            or not isinstance(libraries, dict)
            or any(libraries.get(name) is not True for name in ("h5py", "fsspec", "huggingface_hub", "scipy"))
            or any(len(suites[name]) != 10 for name in SUITES)):
        raise RuntimeError(f"Original-LIBERO source inventory is incomplete: {inventory}")
    evidence = {**observed, "verified_status": "complete", "source_inventory": inventory}
    save(HERE / "inventory_terminal_evidence.json", evidence)
    return evidence


def inventory_job(transport, job: str | None = None) -> str:
    handle = read_local_json(HERE / "inventory_handle.json")
    actual = handle.get("job_id")
    if not isinstance(actual, str) or not HANDLE.fullmatch(actual):
        raise RuntimeError("No valid submitted inventory handle is available")
    if job and job != actual:
        raise RuntimeError("Inventory query must use the already submitted inventory handle")
    return actual


def verify_cpu(transport, job: str) -> dict:
    observed = require_terminal_success(transport, job)
    summary, marker = artifact_receipt(transport, job)
    if summary.get("phase") != "cpu_prepare" or summary.get("pbs_jobid") != job:
        raise RuntimeError(f"CPU preparation summary does not match {job}: {summary}")
    if marker == "complete":
        counts = (summary.get("input_count"), summary.get("calibration_count"),
                  summary.get("test_count"), summary.get("c8_context_count"),
                  summary.get("c80_context_count"))
        if counts != (120, 80, 40, 8, 80):
            raise RuntimeError(f"CPU preparation coverage is incomplete: {counts}")
    elif (not isinstance(summary.get("completed_contexts"), int)
          or not 0 <= summary["completed_contexts"] < 120):
        raise RuntimeError(f"CPU checkpoint summary is malformed: {summary}")
    evidence = {**observed, "summary": summary, "verified_status": marker}
    save(HERE / "cpu_prepare_terminal_evidence.json", evidence)
    return evidence


def verify_gpu(transport, cohort: str, job: str) -> dict:
    if cohort not in GPU_COHORTS:
        raise ValueError(f"Unsupported GPU cohort: {cohort}")
    observed = require_terminal_success(transport, job)
    summary, marker = artifact_receipt(transport, job)
    if (summary.get("cohort") != cohort or summary.get("pbs_jobid") != job
            or summary.get("protocol") != "fastwam-vq-calibration-suites-v1"
            or summary.get("episodes") != 0 or summary.get("predicted_actions_executed") != 0):
        raise RuntimeError(f"GPU summary identity or boundary does not match {cohort}/{job}: {summary}")
    if marker == "complete":
        plan = completed_data_plan(transport)
        rows = plan["inputs"]
        test_ids = list(range(3000, 3040))
        if (summary.get("case_ids") != test_ids or summary.get("case_count") != 40
                or summary.get("query_count") != 80):
            raise RuntimeError(f"GPU held-out case/query coverage is incomplete: {summary}")
        if cohort == "teacher":
            if (summary.get("target_linears") != 614
                    or summary.get("queries_completed") != 80 or summary.get("expected_queries") != 80
                    or summary.get("suites") != {suite: 10 for suite in SUITES}):
                raise RuntimeError(f"Teacher BF16 query coverage is incomplete: {summary}")
        else:
            if summary.get("target_linears") != 614 or summary.get("fitted_linears") != 614:
                raise RuntimeError(f"GPU fit coverage is incomplete: {summary}")
            expected_contexts = 8 if cohort == "c8" else 80
            cohort_ids = plan.get("calibration_cohorts", {}).get(cohort, [])
            if (summary.get("calibration_contexts") != expected_contexts
                    or summary.get("calibration_completed") != expected_contexts
                    or summary.get("calibrated_contexts") != expected_contexts
                    or summary.get("calibration_case_ids") != cohort_ids
                    or summary.get("queries_completed") != 80 or summary.get("expected_queries") != 80
                    or summary.get("suites") != list(SUITES)):
                raise RuntimeError(f"{cohort} calibration/test coverage is incomplete: {summary}")
            test = summary.get("test", {})
            if (test.get("case_count") != 40 or test.get("query_count") != 80
                    or len(test.get("per_case", {})) != 40
                    or set(test.get("per_suite", {})) != set(SUITES)
                    or "motor_rmse_first10_vs_bf16" not in test.get("suite_macro", {})):
                raise RuntimeError(f"{cohort} held-out aggregate is incomplete: {summary}")
    elif summary.get("status") != "checkpointed":
        raise RuntimeError(f"GPU checkpoint summary is malformed: {summary}")
    evidence = {**observed, "summary": summary, "verified_status": marker}
    save(HERE / f"{cohort}_terminal_evidence.json", evidence)
    return evidence


def cohort_files(cohort: str) -> tuple[Path, Path]:
    key = "cpu_prepare" if cohort == "cpu" else cohort
    return HERE / f"{key}_intent_*.json", HERE / f"{key}_handle.json"


def query_named(transport, name: str) -> list[str]:
    code, output = remote.command(
        transport, f"qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1", timeout=35
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("PBS submission history is unavailable")
    matches = output.split()
    if any(not HANDLE.fullmatch(job) for job in matches):
        raise RuntimeError("PBS returned an invalid handle for the named submission")
    return matches


def ensure_no_prior_writer(transport, key: str) -> None:
    for path in HERE.glob(f"{key}_intent_*.json"):
        receipt = read_local_json(path)
        job = receipt.get("job_id")
        if not job:
            raise RuntimeError(f"Unresolved durable intent blocks another {key} submission: {path.name}")
        observed = scheduler(transport, job)
        if (observed.get("query_rc") != 0 or observed.get("job_state") not in ("F", "X")):
            raise RuntimeError(f"A prior {key} job is not confirmed terminal: {observed}")
        if observed.get("Exit_status") == "0":
            try:
                summary = read_remote_json(transport, artifact_path(job) + "/summary.json")
            except Exception:
                summary = {}
            if summary.get("status") == "complete":
                raise RuntimeError(f"{key} already has a complete result from {job}")


def completed_data_plan(transport) -> dict:
    plan = read_remote_json(transport, ROOT + "/data/plan.json")
    rows = plan.get("inputs")
    expected_ids = list(range(2000, 2080)) + list(range(3000, 3040))
    cohorts = plan.get("calibration_cohorts", {})
    c8, c80 = cohorts.get("c8"), cohorts.get("c80")
    ids = [int(row.get("case_id", -1)) for row in rows] if isinstance(rows, list) else []
    if (plan.get("status") != "complete" or ids != expected_ids
            or (plan.get("input_count"), plan.get("calibration_count"), plan.get("test_count")) != (120, 80, 40)
            or not isinstance(c8, list) or not isinstance(c80, list)
            or any(not isinstance(value, int) for value in c8 + c80)
            or len(c8) != 8 or len(set(c8)) != 8 or c8 != sorted(c8)
            or len(c80) != 80 or len(set(c80)) != 80
            or c80 != list(range(2000, 2080)) or not set(c8) <= set(c80)):
        raise RuntimeError(f"Shared original-LIBERO input plan is incomplete or malformed: {plan}")
    return plan


def verify_preparation_ready(transport) -> dict:
    handle = read_local_json(HERE / "cpu_prepare_handle.json")
    job = handle.get("job_id")
    if not isinstance(job, str) or not HANDLE.fullmatch(job):
        raise RuntimeError("No completed CPU preparation handle is available")
    evidence = verify_cpu(transport, job)
    if evidence.get("verified_status") != "complete":
        raise RuntimeError("CPU inputs are checkpointed but incomplete; resume preparation first")
    return completed_data_plan(transport)


def verified_teacher_for_arm(transport, requested: str | None) -> str:
    receipt = read_local_json(HERE / "teacher_handle.json")
    job = receipt.get("job_id")
    if not isinstance(job, str) or not HANDLE.fullmatch(job):
        raise RuntimeError("No validated teacher handle is available")
    if requested and requested != job:
        raise RuntimeError("--depend must equal the current teacher handle")
    result = verify_gpu(transport, "teacher", job)
    if result.get("verified_status") != "complete":
        raise RuntimeError("Only a final complete teacher job can release calibration arms")
    return job


def submit(transport, cohort: str, attempt: str, depend: str | None) -> None:
    value = protocol()
    if not ATTEMPT.fullmatch(attempt):
        raise ValueError("Use a short lowercase attempt label")
    if cohort not in ("cpu", *GPU_COHORTS):
        raise ValueError("Submit cohort must be cpu, teacher, c8, or c80")
    if cohort == "cpu":
        verify_inventory(transport, inventory_job(transport))
        key, tag, queue, select = "cpu_prepare", "p", "normal", "select=1:ncpus=8:mem=32gb"
        walltime, script = "01:59:00", "cpu.pbs"
        teacher_job = None
    else:
        verify_preparation_ready(transport)
        key = cohort
        tag, queue, select = {"teacher": ("t", "normal", "select=1:ncpus=16:mem=110gb:ngpus=1"),
                              "c8": ("8", "normal", "select=1:ncpus=16:mem=110gb:ngpus=1"),
                              "c80": ("0", "normal", "select=1:ncpus=16:mem=110gb:ngpus=1")}[cohort]
        walltime, script = "01:59:00", "gpu.pbs"
        teacher_job = verified_teacher_for_arm(transport, depend) if cohort in ("c8", "c80") else None

    name = f"fwvc1013{tag}{attempt}"
    intent_path = HERE / f"{key}_intent_{attempt}.json"
    handle_path = HERE / f"{key}_handle.json"
    if intent_path.exists():
        receipt = read_local_json(intent_path)
        if receipt.get("attempt") != attempt or receipt.get("job_name") != name:
            raise RuntimeError("Existing durable intent does not match the requested attempt")
        if receipt.get("job_id"):
            save(handle_path, receipt)
            print("CONFIRMED_HANDLE", receipt["job_id"])
            return
        matches = query_named(transport, name)
        if len(matches) == 1:
            receipt.update(job_id=matches[0], status="recovered")
        elif matches:
            raise RuntimeError("Ambiguous PBS history for this attempt")
        else:
            raise RuntimeError("Unconfirmed intent exists; inspect it before any retry")
    else:
        ensure_no_prior_writer(transport, key)
        if cohort == "cpu":
            ensure_no_prior_writer(transport, "cpu_prepare")
        matches = query_named(transport, name)
        if len(matches) == 1:
            receipt = {"job_id": matches[0], "status": "recovered", "cohort": cohort,
                       "attempt": attempt, "job_name": name}
        elif matches:
            raise RuntimeError("Ambiguous PBS history for this attempt")
        else:
            receipt = {
                "status": "intent_before_qsub", "cohort": cohort, "attempt": attempt,
                "job_name": name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "walltime": walltime, "queue": queue, "project": "personal-yguo017",
                "select": select, "remote_root": ROOT,
            }
            if teacher_job:
                receipt["teacher_job_id"] = teacher_job
            save(intent_path, receipt)
            args = ["qsub", "-N", name, "-P", "personal-yguo017", "-q", queue,
                    "-l", select, "-l", f"walltime={walltime}"]
            if cohort != "cpu":
                args += ["-v", f"FW_COHORT={cohort}"]
                if teacher_job:
                    args[-1] += f",FW_TEACHER_JOBID={teacher_job}"
                    args += ["-W", f"depend=afterok:{teacher_job}"]
            args.append(ROOT + "/" + script)
            command = " ".join(shlex.quote(arg) for arg in args)
            code, output = remote.command(transport, command, timeout=40)
            result = output.strip()
            if code or not HANDLE.fullmatch(result):
                receipt.update(status="unknown_or_rejected", qsub_rc=code)
                save(intent_path, receipt)
                raise RuntimeError("qsub outcome is unknown or rejected; do not retry automatically")
            receipt.update(job_id=result, status="confirmed")

    save(intent_path, receipt)
    save(handle_path, receipt)
    print("CONFIRMED_HANDLE", receipt["job_id"])


def status(transport, cohort: str, job: str | None) -> None:
    if cohort == "inventory":
        actual = inventory_job(transport, job)
        observed = scheduler(transport, actual)
        remote_path = ROOT + "/source_inventory.json"
        code, _ = remote.command(transport, f"test -f {shlex.quote(remote_path)}", timeout=20)
        inventory = read_remote_json(transport, remote_path) if code == 0 else None
        print(json.dumps({"scheduler": observed, "source_inventory": inventory}))
        return
    if cohort == "cpu":
        receipt = read_local_json(HERE / "cpu_prepare_handle.json")
    elif cohort in GPU_COHORTS:
        receipt = read_local_json(HERE / f"{cohort}_handle.json")
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")
    actual = job or receipt.get("job_id")
    if not isinstance(actual, str) or not HANDLE.fullmatch(actual):
        raise RuntimeError("No valid cohort handle is available")
    observed = scheduler(transport, actual)
    remote_path = artifact_path(actual) + "/summary.json"
    code, _ = remote.command(transport, f"test -f {shlex.quote(remote_path)}", timeout=20)
    summary = read_remote_json(transport, remote_path) if code == 0 else None
    save(HERE / f"{cohort}_last_scheduler_snapshot.json", observed)
    print(json.dumps({"scheduler": observed, "summary": summary}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "pull"))
    parser.add_argument("--cohort", choices=("inventory", "cpu", *GPU_COHORTS))
    parser.add_argument("--attempt", default="a")
    parser.add_argument("--depend")
    parser.add_argument("--job")
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    if args.action in ("status", "verify", "submit") and not args.cohort:
        parser.error("--cohort required")
    if args.action == "submit" and not ATTEMPT.fullmatch(args.attempt):
        parser.error("Use a short lowercase attempt label")
    if args.action == "pull" and not args.files:
        parser.error("--files required")
    if args.action == "submit" and args.cohort == "inventory":
        parser.error("The already submitted inventory has a fixed handle; it cannot be resubmitted here")
    if args.action == "submit" and args.depend and args.cohort not in ("c8", "c80"):
        parser.error("--depend is supported only for c8 and c80")

    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        if args.action == "upload":
            protocol()
            names = args.files
            if names and any(name not in FILES for name in names):
                parser.error("--files contains an unsupported control file")
            trusted.upload(transport, names)
        elif args.action == "submit":
            submit(transport, args.cohort, args.attempt, args.depend)
        elif args.action == "status":
            status(transport, args.cohort, args.job)
        elif args.action == "verify":
            if args.cohort == "inventory":
                result = verify_inventory(transport, inventory_job(transport, args.job))
            elif args.cohort == "cpu":
                job = args.job or read_local_json(HERE / "cpu_prepare_handle.json").get("job_id")
                result = verify_cpu(transport, job)
            else:
                job = args.job or read_local_json(HERE / f"{args.cohort}_handle.json").get("job_id")
                result = verify_gpu(transport, args.cohort, job)
            print(json.dumps(result))
        else:
            trusted.pull(transport, args.files)
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
