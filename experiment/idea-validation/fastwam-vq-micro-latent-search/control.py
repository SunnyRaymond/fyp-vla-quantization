"""Bounded local receipts and trusted PBS/SSH control for the micro latent search."""
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
import time

HERE = Path(__file__).resolve().parent
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-micro-latent-search-20261014"
PROTOCOL = "fastwam-vq-micro-latent-search-v1"
NORMALIZATION_POLICY = "normalization_c = mean_8cal_micro_initial_action_mse / mean_8cal_micro_initial_future_nmse"
LIMIT = 256 * 1024
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
ATTEMPT = re.compile(r"^[a-z][a-z0-9]{0,5}$")
RUN_ID = re.compile(r"^r[12]-r\d{3}$")
R1_RUNS = {"r1-r000": 0.1, "r1-r001": 0.5, "r1-r002": 1.0,
           "r1-r003": 0.7, "r1-r004": 1.5}

TRUSTED_PATH = HERE.parent / "fastwam-vq-proxy-diagnostic" / "control.py"
spec = importlib.util.spec_from_file_location("micro_search_trusted_control", TRUSTED_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load trusted SSH control: {TRUSTED_PATH}")
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
trusted.HERE, trusted.ROOT = HERE, ROOT
trusted.FILES = (
    "protocol.json", "control.py", "environment.sh", "gpu.pbs", "references.pbs",
    "selfcheck.pbs", "choose.pbs", "analysis.pbs", "analyze_final.py",
    "sampler.py", "train_banks.py", "checkpointing.py",
    "test_checkpointing.py", "test_driver.py", "run_distill.py", "latent_reference.py",
    "prepare_references.py", "test_latent_loss.py", "test_micro_training.py",
    "test_micro_reference.py", "metrics.py", "micro_codebook.py", "choose.py",
)
remote, save, scheduler = trusted.remote, trusted.save, trusted.scheduler


def _local_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise RuntimeError(f"Missing or oversized local control receipt: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path.name}")
    return value


def protocol() -> dict:
    value = trusted.protocol()
    if (value.get("protocol") != PROTOCOL or value.get("target_linears") != 614
            or value.get("remote_root") != ROOT):
        raise RuntimeError("protocol.json does not match the frozen micro latent search")
    return value


def _remote_json(transport, path: str) -> dict:
    return trusted.read_remote_json(transport, path)


def _atomic_remote_json(transport, path: str, value: dict) -> None:
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > LIMIT:
        raise ValueError("Control JSON exceeds the 256 KiB limit")
    with remote.sftp_client(transport) as sftp:
        try:
            size = sftp.stat(path).st_size
        except OSError as exc:
            if getattr(exc, "errno", None) != 2:
                raise
            size = None
        if size is not None:
            if size > LIMIT:
                raise RuntimeError(f"Existing remote control JSON is oversized: {path}")
            with sftp.file(path, "rb") as stream:
                current = stream.read(LIMIT + 1)
            if current != data:
                raise FileExistsError(f"Refusing to replace different remote control JSON: {path}")
            return
        parent = path.rsplit("/", 1)[0]
        code, output = remote.command(transport, "mkdir -p " + shlex.quote(parent), timeout=25)
        if code:
            raise RuntimeError(f"Could not create remote control directory: {output}")
        temporary = path + f".tmp.{time.time_ns()}"
        with sftp.file(temporary, "wb") as stream:
            stream.write(data)
        sftp.rename(temporary, path)


def _job_name(kind: str, attempt: str, run_id: str | None = None) -> str:
    if not ATTEMPT.fullmatch(attempt):
        raise ValueError("Use a short lowercase attempt label")
    if kind == "run":
        if not run_id or not RUN_ID.fullmatch(run_id):
            raise ValueError("run-id must use r1-rNNN or r2-rNNN")
        return "fwm14" + run_id.replace("-", "") + attempt
    return f"fwm14{kind}{attempt}"


def _submit(transport, *, name: str, script: str, intent_path: Path, handle_path: Path,
            variables: dict[str, str], select: str, walltime: str, previous_job: str | None = None) -> dict:
    receipt = _local_json(intent_path) if intent_path.exists() else {}
    current = _local_json(handle_path) if handle_path.exists() else {}
    stem = handle_path.stem.removesuffix("_handle")
    related = [path for path in intent_path.parent.glob(f"{stem}_intent_*.json")
               if path != intent_path]
    prior_intents = [_local_json(path) for path in related]
    if any(not item.get("job_id") for item in prior_intents):
        raise RuntimeError("Another attempt has an unknown qsub outcome; recover that exact attempt first")
    if previous_job is None and prior_intents:
        raise RuntimeError("This stage already has a submission intent; use its current handle")
    if receipt.get("job_id"):
        job = receipt["job_id"]
        if not isinstance(job, str) or not HANDLE.fullmatch(job) or receipt.get("job_name") != name:
            raise RuntimeError("Intent receipt is bound to another or invalid PBS job")
        if current and current.get("job_id") not in (job, previous_job):
            raise RuntimeError("Current handle is already bound to a different job")
        save(handle_path, receipt)
        print("CONFIRMED_HANDLE", job)
        return receipt
    if current and current.get("job_id") != previous_job:
        raise RuntimeError("This run already has a current handle; use its exact resumable state")

    rc, output = remote.command(
        transport, f"qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1", timeout=35
    )
    if rc not in (0, 1) or (rc == 1 and output.strip()):
        raise RuntimeError("PBS submission history is unavailable; no job was submitted")
    matches = output.split()
    if matches:
        if len(matches) != 1 or not HANDLE.fullmatch(matches[0]):
            raise RuntimeError("Ambiguous PBS job-name recovery; no new job was submitted")
        receipt.update(job_id=matches[0], status="recovered", job_name=name)
    else:
        if receipt:
            raise RuntimeError("An unconfirmed intent has no qselect match; refusing duplicate qsub")
        receipt = {
            "status": "intent_before_qsub", "job_name": name,
            "created_utc": datetime.now(timezone.utc).isoformat(), "remote_root": ROOT,
            "script": script, "select": select, "walltime": walltime, "variables": variables,
            "resume_job": previous_job,
        }
        save(intent_path, receipt)
        variable_arg = "-v " + shlex.quote(",".join(f"{key}={value}" for key, value in variables.items())) + " "
        command = (f"qsub -P personal-yguo017 -q normal -N {shlex.quote(name)} "
                   f"-l {shlex.quote(select)} -l walltime={shlex.quote(walltime)} "
                   f"{variable_arg}{shlex.quote(ROOT + '/' + script)}")
        rc, output = remote.command(transport, command, timeout=40)
        if rc or not HANDLE.fullmatch(output.strip()):
            receipt.update(status="unknown_or_rejected", message=output[:1500])
            save(intent_path, receipt)
            raise RuntimeError("qsub outcome is unknown; recover this same attempt with qselect")
        receipt.update(job_id=output.strip(), status="confirmed")
    save(intent_path, receipt)
    save(handle_path, receipt)
    print("CONFIRMED_HANDLE", receipt["job_id"])
    return receipt


def _handle(path: Path) -> dict:
    receipt = _local_json(path)
    if not isinstance(receipt.get("job_id"), str) or not HANDLE.fullmatch(receipt["job_id"]):
        raise RuntimeError(f"No confirmed PBS handle in {path.name}")
    return receipt


def _resume_attempt(prefix: str, previous_job: str, name_for) -> str:
    index = 1
    while True:
        attempt = f"c{index}"
        path = HERE / f"{prefix}_intent_{attempt}.json"
        if not path.exists():
            return attempt
        receipt = _local_json(path)
        if not receipt.get("job_id"):
            if (receipt.get("resume_job") == previous_job
                    and receipt.get("job_name") == name_for(attempt)):
                return attempt
            raise RuntimeError("An unresolved intent belongs to a different continuation; recover it first")
        index += 1


def _terminal(transport, job: str, artifact: str, marker: str) -> tuple[dict, dict]:
    observed = scheduler(transport, job)
    if (observed.get("query_rc") != 0 or observed.get("job_state") not in ("F", "X")
            or observed.get("Exit_status") != "0"):
        raise RuntimeError(f"Job is not a successful terminal allocation: {observed}")
    check = (f"test -f {shlex.quote(artifact + '/' + marker)} && "
             f"test -f {shlex.quote(artifact + '/exit_code.txt')} && "
             f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0")
    rc, _ = remote.command(transport, check, timeout=25)
    if rc:
        raise RuntimeError(f"Missing {marker} or zero exit receipt for {job}")
    return observed, _remote_json(transport, artifact + "/summary.json")


def _exact_resumable(transport, observed: dict, job: str, result: str, artifact: str,
                     summary: dict, binding: dict) -> dict:
    if (summary.get("status") != "resumable" or summary.get("job_id") != job
            or any(summary.get(key) != value for key, value in binding.items())):
        raise RuntimeError("Terminal summary does not bind to this exact resumable run")
    checks = (
        f"test -f {shlex.quote(artifact + '/RESUME_READY')} && "
        f"test -f {shlex.quote(result + '/CHECKPOINT_READY')} && "
        f"test -s {shlex.quote(result + '/checkpoint.pt')} && "
        f"test -s {shlex.quote(result + '/checkpoint_receipt.json')}")
    rc, _ = remote.command(transport, checks, timeout=25)
    if rc:
        raise RuntimeError("Resumable summary lacks its checkpoint and continuation markers")
    checkpoint = _remote_json(transport, result + "/checkpoint_receipt.json")
    if (checkpoint.get("schema_version") != 2 or checkpoint.get("job_id") != job
            or checkpoint.get("mode") != "search"
            or checkpoint.get("run_id") != binding["run_id"]
            or checkpoint.get("candidate_run_id") != binding["run_id"]
            or checkpoint.get("round") != binding["round"]
            or float(checkpoint.get("rho", float("nan"))) != float(summary.get("rho", float("nan")))
            or checkpoint.get("latent_weight") != summary.get("latent_weight")
            or checkpoint.get("normalization_c") != summary.get("normalization_c")
            or checkpoint.get("updates_completed") != summary.get("updates")
            or checkpoint.get("epochs_completed") != summary.get("completed_epochs")
            or checkpoint.get("checkpoint") != result + "/checkpoint.pt"
            or checkpoint.get("target_linears") != 614):
        raise RuntimeError("Checkpoint receipt is not bound to the terminal resumable job")
    return {**observed, "status": "resumable", "summary": summary,
            "checkpoint_receipt": checkpoint, "resume_ready": True}


def selfcheck_submit(transport, attempt: str) -> dict:
    protocol()
    path, intent = HERE / "selfcheck_handle.json", HERE / f"selfcheck_intent_{attempt}.json"
    previous_job = None
    if path.exists() and not intent.exists():
        previous_job = _handle(path)["job_id"]
        failed = scheduler(transport, previous_job)
        if (failed.get("query_rc") or failed.get("job_state") not in ("F", "X")
                or failed.get("Exit_status") in (None, "0")):
            raise RuntimeError("A new self-check attempt requires a confirmed failed terminal job")
        save(HERE / f"selfcheck_failed_{previous_job}.json", failed)
    return _submit(transport, name=_job_name("sc", attempt), script="selfcheck.pbs",
                   intent_path=intent, handle_path=path, variables={},
                   select="select=1:ncpus=4:mem=16gb", walltime="00:10:00", previous_job=previous_job)


def selfcheck_verify(transport) -> dict:
    handle = _handle(HERE / "selfcheck_handle.json")
    job = handle["job_id"]
    observed, summary = _terminal(transport, job, f"{ROOT}/artifacts/{job}", "PIPELINE_COMPLETE")
    if (summary.get("status") != "complete" or summary.get("protocol") != PROTOCOL
            or summary.get("job_id") != job or summary.get("self_check") != "passed"
            or summary.get("training_self_check") != "passed"
            or summary.get("reference_self_check", {}).get("passed") is not True
            or summary.get("chooser_self_check") != "passed"):
        raise RuntimeError("Self-check result is incomplete or bound to another job")
    evidence = {**observed, "status": "complete", "summary": summary}
    save(HERE / "selfcheck_terminal.json", evidence)
    return evidence


def reference_submit(transport, attempt: str) -> dict:
    p = protocol()
    if selfcheck_verify(transport).get("status") != "complete":
        raise RuntimeError("Reference preparation requires the verified CPU self-check first")
    current_path = HERE / "reference_handle.json"
    previous_job = None
    if current_path.exists():
        previous_job = _handle(current_path)["job_id"]
        observed = scheduler(transport, previous_job)
        if (observed.get("query_rc") or observed.get("job_state") not in ("F", "X")
                or observed.get("Exit_status") is None):
            raise RuntimeError("Reference retry requires a confirmed terminal job")
        if observed["Exit_status"] == "0":
            prior = reference_verify(transport)
            if prior.get("status") != "resumable":
                raise RuntimeError("Reference job is already complete or not exactly resumable")
        else:
            save(HERE / f"reference_failed_{previous_job}.json", observed)
    intent = HERE / f"reference_intent_{attempt}.json"
    if previous_job and intent.exists() and _local_json(intent).get("job_id") == previous_job:
        raise RuntimeError("Reference continuation needs a fresh --attempt label; the prior label is already used")
    return _submit(transport, name=_job_name("rf", attempt), script="references.pbs",
                   intent_path=intent, handle_path=current_path, previous_job=previous_job,
                   variables={"FW_SOFT_DEADLINE_SECONDS": "3300"},
                   select="select=1:ncpus=16:mem=110gb:ngpus=1", walltime="01:00:00")


def reference_verify(transport) -> dict:
    p = protocol()
    handle = _handle(HERE / "reference_handle.json")
    job = handle["job_id"]
    observed = scheduler(transport, job)
    if (observed.get("query_rc") != 0 or observed.get("job_state") not in ("F", "X")
            or observed.get("Exit_status") != "0"):
        raise RuntimeError(f"Reference job failed: {observed}")
    artifact = f"{ROOT}/artifacts/{job}"
    artifact_summary = _remote_json(transport, artifact + "/summary.json")
    shared_summary = _remote_json(transport, ROOT + "/reference_summary.json")
    if artifact_summary != shared_summary:
        raise RuntimeError("Reference artifact summary differs from shared summary")
    binding = {"protocol": PROTOCOL, "initial_arm": p["initial_arm"],
               "bank_root": p["bank_root"], "bank_full_job": p["bank_full_job"],
               "source_root": p["source_root"], "source_full_job": p["source_full_job"]}
    for key, value in binding.items():
        if shared_summary.get(key) != value:
            raise RuntimeError(f"Reference summary has unexpected {key}")
    if shared_summary.get("status") == "resumable":
        check = (f"test -f {shlex.quote(artifact + '/RESUME_READY')} && "
                 f"test -f {shlex.quote(artifact + '/exit_code.txt')} && "
                 f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0 && "
                 f"test -s {shlex.quote(ROOT + '/reference_progress.json')}")
        rc, _ = remote.command(transport, check, timeout=25)
        if rc:
            raise RuntimeError("Reference continuation lacks its saved progress marker")
        progress = _remote_json(transport, ROOT + "/reference_progress.json")
        completed = progress.get("completed_parent_contexts")
        pending = progress.get("pending_parent_context_count")
        if (progress.get("job_id") != job or progress.get("protocol") != PROTOCOL
                or progress.get("status") != "resumable"
                or progress.get("expected_parent_contexts") != 16
                or not isinstance(completed, int) or not isinstance(pending, int)
                or completed < 0 or pending < 0 or completed + pending != 16
                or progress.get("model_forward_count") != completed
                or progress.get("teacher_forward_count") != 0
                or shared_summary.get("job_id") != job
                or shared_summary.get("parent_context_count") != completed
                or shared_summary.get("expected_parent_context_count") != 16
                or shared_summary.get("model_forward_count") != completed
                or shared_summary.get("teacher_forward_count") != 0):
            raise RuntimeError("Reference progress does not bind to this resumable job")
        evidence = {**observed, "status": "resumable", "summary": shared_summary}
    elif shared_summary.get("status") == "complete":
        if (shared_summary.get("job_id") != job or shared_summary.get("target_linears") != 614
                or shared_summary.get("teacher_context_count") != 60
                or shared_summary.get("parent_context_count") != 16
                or shared_summary.get("calibration_context_count") != 8
                or shared_summary.get("selection_context_count") != 8
                or shared_summary.get("model_forward_count") != 16
                or shared_summary.get("teacher_forward_count") != 0
                or shared_summary.get("self_check") != "passed"
                or shared_summary.get("self_check_passed") is not True
                or shared_summary.get("selection_cache_parity") is not True):
            raise RuntimeError("Reference summary does not meet frozen reference coverage")
        normalization = shared_summary.get("loss_normalization")
        if (not isinstance(normalization, dict)
                or normalization.get("policy") != NORMALIZATION_POLICY
                or normalization.get("calibration_ids") != p["calibration_cases"]
                or not isinstance(normalization.get("normalization_c"), (int, float))
                or not math.isfinite(normalization["normalization_c"])
                or normalization["normalization_c"] <= 0):
            raise RuntimeError("Reference normalization contract is incomplete")
        check = (f"test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')} && "
                 f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0 && "
                 f"test -s {shlex.quote(p['reference_cache'])}")
        rc, _ = remote.command(transport, check, timeout=25)
        if rc:
            raise RuntimeError("Reference completion marker, receipt, or cache is missing")
        evidence = {**observed, "status": "complete", "summary": shared_summary}
    else:
        raise RuntimeError(f"Unexpected reference status: {shared_summary.get('status')}")
    save(HERE / "reference_terminal.json", evidence)
    return evidence


def _run_handle(run_id: str) -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run-id must use r1-rNNN or r2-rNNN")
    return _handle(HERE / f"run_{run_id}_handle.json")


def search_submit(transport, run_id: str, rho: float, round_number: int, attempt: str) -> dict:
    p = protocol()
    if not RUN_ID.fullmatch(run_id) or round_number not in (1, 2):
        raise ValueError("Invalid run-id or round")
    if (run_id.startswith(f"r{round_number}-") is False
            or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho < 0):
        raise ValueError("run-id, rho, and round do not agree")
    if round_number == 1:
        if R1_RUNS.get(run_id) != float(rho):
            raise ValueError("Round 1 run ID/rho differs from the user-approved frozen candidates")
    else:
        candidates = _round2_candidates(transport)
        item = next((row for row in candidates if row.get("run_id") == run_id), None)
        if item is None or float(item.get("rho", float("nan"))) != float(rho):
            raise ValueError("Round 2 run-id/rho is not one of the verified chooser candidates")
    if reference_verify(transport).get("status") != "complete":
        raise RuntimeError("Search runs require the verified complete reference cache")
    rho_text = str(float(rho))
    variables = {"FW_RUN_ID": run_id, "FW_RHO": rho_text,
                 "FW_ROUND": str(round_number), "FW_MODE": "search"}
    intent = HERE / f"run_{run_id}_intent_{attempt}.json"
    handle = HERE / f"run_{run_id}_handle.json"
    return _submit(transport, name=_job_name("run", attempt, run_id), script="gpu.pbs",
                   intent_path=intent, handle_path=handle, variables=variables,
                   select="select=1:ncpus=16:mem=110gb:ngpus=1", walltime="02:00:00")


def _verify_search(transport, run_id: str, *, require_complete: bool) -> dict:
    p = protocol()
    handle = _run_handle(run_id)
    job = handle["job_id"]
    observed = scheduler(transport, job)
    if observed.get("query_rc") or observed.get("job_state") not in ("F", "X"):
        raise RuntimeError(f"Search job is not terminal: {observed}")
    artifact, result = f"{ROOT}/artifacts/{job}", f"{ROOT}/results/{run_id}-{job}"
    summary = _remote_json(transport, result + "/summary.json")
    binding = {"protocol": PROTOCOL, "run_id": run_id, "job_id": job,
               "round": int(run_id[1]), "target_linears": 614}
    if any(summary.get(key) != value for key, value in binding.items()):
        raise RuntimeError("Search summary does not match its run, round, or PBS handle")
    expected_rho = float(handle.get("variables", {}).get("FW_RHO", "nan"))
    if (not math.isfinite(expected_rho) or summary.get("rho") != expected_rho
            or summary.get("normalization_c") is None):
        raise RuntimeError("Search summary rho or frozen normalization is not bound to its current handle")
    if _remote_json(transport, artifact + "/summary.json") != summary:
        raise RuntimeError("Search artifact summary differs from the result directory")
    if observed.get("Exit_status") != "0":
        raise RuntimeError(f"Search job failed: {observed}")
    if summary.get("status") == "resumable":
        evidence = _exact_resumable(transport, observed, job, result, artifact, summary, binding)
        if require_complete:
            raise RuntimeError("Search run is resumable, not complete")
    elif summary.get("status") == "search_complete":
        check = (f"test -f {shlex.quote(artifact + '/PIPELINE_COMPLETE')} && "
                 f"test -f {shlex.quote(result + '/SEARCH_COMPLETE')} && "
                 f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0")
        rc, _ = remote.command(transport, check, timeout=25)
        if rc:
            raise RuntimeError("Search completion markers or zero exit receipt are missing")
        ref = reference_verify(transport)["summary"]
        history = summary.get("selection_history")
        if (summary.get("updates") != 16 or summary.get("completed_epochs") != 2
                or summary.get("selection_context_count") != 16
                or summary.get("initial_selection_context_count") != 8
                or summary.get("test_forward_count") != 0
                or summary.get("final_test_context_count") != 0
                or summary.get("teacher_forward_count") != 0
                or not isinstance(history, list) or len(history) != 3
                or [row.get("epoch") for row in history] != [0, 1, 2]
                or any(row.get("case_count") != 4 or row.get("context_count") != 8 for row in history)
                or summary.get("normalization_c") != ref["loss_normalization"]["normalization_c"]):
            raise RuntimeError("Search summary does not satisfy the frozen 16-update/3-epoch-history gates")
        for row in history:
            for field in ("mean_motor_rmse_first10", "mean_gripper_rmse_first10",
                          "mean_motor_rmse_late22", "p90_motor_rmse_first10"):
                value = row.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise RuntimeError(f"Search selection history lacks finite {field}")
        evidence = {**observed, "status": "search_complete", "summary": summary,
                    "summary_path": result + "/summary.json", "result_dir": result,
                    "verified_terminal": True}
    else:
        raise RuntimeError(f"Unexpected search summary status: {summary.get('status')}")
    save(HERE / f"run_{run_id}_terminal.json", evidence)
    return evidence


def search_verify(transport, run_id: str) -> dict:
    return _verify_search(transport, run_id, require_complete=False)


def continue_search(transport, run_id: str) -> dict:
    evidence = _verify_search(transport, run_id, require_complete=False)
    if evidence.get("status") != "resumable":
        raise RuntimeError("Continue requires this exact run's verified terminal resumable state")
    if reference_verify(transport).get("status") != "complete":
        raise RuntimeError("Search continuation requires the verified complete reference cache")
    old_job = evidence["job_id"]
    old_handle = _run_handle(run_id)
    attempt = _resume_attempt(f"run_{run_id}", old_job,
                              lambda label: _job_name("run", label, run_id))
    if not ATTEMPT.fullmatch(attempt):
        raise RuntimeError("Too many continuation attempts for this run")
    variables = {"FW_RUN_ID": run_id, "FW_RHO": str(float(old_handle["variables"]["FW_RHO"])),
                 "FW_ROUND": str(old_handle["variables"]["FW_ROUND"]), "FW_MODE": "search",
                 "FW_RESUME_JOB": old_job}
    intent = HERE / f"run_{run_id}_intent_{attempt}.json"
    handle = HERE / f"run_{run_id}_handle.json"
    return _submit(transport, name=_job_name("run", attempt, run_id), script="gpu.pbs",
                   intent_path=intent, handle_path=handle, variables=variables,
                   select="select=1:ncpus=16:mem=110gb:ngpus=1", walltime="02:00:00",
                   previous_job=old_job)


def _round2_candidates(transport) -> list[dict]:
    evidence_path = HERE / "choice_round1_terminal.json"
    if not evidence_path.is_file():
        choice_verify(transport, 1)
    evidence = _local_json(evidence_path)
    if evidence.get("status") != "complete" or evidence.get("round2_candidates") is None:
        choice_verify(transport, 1)
        evidence = _local_json(evidence_path)
    source = evidence.get("round2_candidates")
    if (not isinstance(source, dict) or source.get("protocol") != PROTOCOL
            or source.get("round") != 2 or not isinstance(source.get("candidates"), list)
            or len(source["candidates"]) != 3):
        raise RuntimeError("Verified round 1 chooser did not publish three round 2 candidates")
    rows = source["candidates"]
    if [row.get("run_id") for row in rows if isinstance(row, dict)] != [
            "r2-r000", "r2-r001", "r2-r002"]:
        raise RuntimeError("Round 2 must use the three frozen sequential run IDs")
    if any(not isinstance(row, dict) or not RUN_ID.fullmatch(str(row.get("run_id", "")))
           or not row["run_id"].startswith("r2-")
           or not isinstance(row.get("rho"), (int, float))
           or not math.isfinite(row["rho"]) or row["rho"] < 0 for row in rows):
        raise RuntimeError("Round 2 candidate binding is malformed")
    if len({row["run_id"] for row in rows}) != 3:
        raise RuntimeError("Round 2 candidate run IDs are not unique")
    return rows


def _candidate_inputs(transport, round_number: int) -> dict:
    ref = reference_verify(transport)
    if ref.get("status") != "complete":
        raise RuntimeError("Global choice requires the complete verified reference summary")
    if round_number == 1:
        run_ids = list(R1_RUNS)
    else:
        run_ids = list(R1_RUNS) + [row["run_id"] for row in _round2_candidates(transport)]
    candidates = []
    for run_id in run_ids:
        evidence = _verify_search(transport, run_id, require_complete=True)
        summary = evidence["summary"]
        expected_round = 1 if run_id.startswith("r1-") else 2
        if summary.get("round") != expected_round:
            raise RuntimeError("Candidate summary round does not match its run ID")
        if expected_round == 1 and float(summary.get("rho", float("nan"))) != R1_RUNS[run_id]:
            raise RuntimeError("Round 1 candidate rho differs from its frozen point")
        candidates.append({"run_id": run_id, "job_id": evidence["job_id"], "rho": summary["rho"],
                           "round": expected_round,
                           "summary_path": evidence["summary_path"],
                           "result_dir": evidence["result_dir"], "verified_terminal": True})
    return {"protocol": PROTOCOL, "round": round_number,
            "reference_summary_path": ROOT + "/reference_summary.json", "candidates": candidates}


def choice_submit(transport, round_number: int, attempt: str) -> dict:
    if round_number not in (1, 2) or not ATTEMPT.fullmatch(attempt):
        raise ValueError("Choice round must be 1 or 2")
    inputs = _candidate_inputs(transport, round_number)
    input_path = f"{ROOT}/choice_inputs_round{round_number}.json"
    _atomic_remote_json(transport, input_path, inputs)
    intent = HERE / f"choice_r{round_number}_intent_{attempt}.json"
    handle = HERE / f"choice_r{round_number}_handle.json"
    name = _job_name(f"c{round_number}", attempt)
    variables = {"FW_CHOICE_ROUND": str(round_number)}
    return _submit(transport, name=name, script="choose.pbs", intent_path=intent,
                   handle_path=handle, variables=variables,
                   select="select=1:ncpus=4:mem=16gb", walltime="00:10:00")


def choice_verify(transport, round_number: int) -> dict:
    handle = _handle(HERE / f"choice_r{round_number}_handle.json")
    job = handle["job_id"]
    artifact = f"{ROOT}/artifacts/{job}"
    observed, artifact_summary = _terminal(transport, job, artifact, "PIPELINE_COMPLETE")
    result = f"{ROOT}/results/choice-r{round_number}-{job}"
    check = f"test -f {shlex.quote(result + '/CHOICE_COMPLETE')}"
    rc, _ = remote.command(transport, check, timeout=25)
    if rc:
        raise RuntimeError("Chooser result directory is missing CHOICE_COMPLETE")
    summary = _remote_json(transport, result + "/summary.json")
    choice = _remote_json(transport, result + "/choice.json")
    inputs = _remote_json(transport, ROOT + f"/choice_inputs_round{round_number}.json")
    ref = reference_verify(transport)["summary"]
    count, rounds_complete = ((len(R1_RUNS), 1) if round_number == 1 else (len(R1_RUNS) + 3, 2))
    if (summary != artifact_summary or summary.get("status") != "complete"
            or summary.get("protocol") != PROTOCOL or summary.get("job_id") != job
            or summary.get("round") != round_number or summary.get("candidate_count") != count
            or summary.get("rounds_complete") != rounds_complete or summary.get("model_forward_count") != 0
            or summary.get("normalization_c") != ref["loss_normalization"]["normalization_c"]
            or inputs.get("round") != round_number
            or summary.get("selection_source_jobs") != [row.get("job_id") for row in inputs.get("candidates", [])]
            or choice.get("status") != "complete" or choice.get("protocol") != PROTOCOL
            or choice.get("selection_job_id") != job
            or choice.get("rounds_complete") != round_number
            or choice.get("candidate_count") != count or choice.get("model_forward_count") != 0
            or choice.get("normalization_c") != summary.get("normalization_c")
            or choice.get("selection_source_jobs") != summary.get("selection_source_jobs")
            or any(choice.get(key) != summary.get(key) for key in
                   ("selected_run_id", "selected_epoch", "rho", "latent_weight"))):
        raise RuntimeError("Global chooser summary or choice binding is invalid")
    output_name = "round2_candidates.json" if round_number == 1 else "optimal_selection.json"
    output = _remote_json(transport, ROOT + "/" + output_name)
    if output.get("protocol") != PROTOCOL or output.get("selection_job_id") != job:
        raise RuntimeError(f"Published {output_name} is not bound to the verified choice job")
    if round_number == 1:
        rows = output.get("candidates")
        if (output.get("round") != 2 or output.get("selection_job_id") != job
                or output.get("selection_source_jobs") != summary.get("selection_source_jobs")
                or not isinstance(rows, list) or len(rows) != 3
                or any(not isinstance(row, dict) or not RUN_ID.fullmatch(str(row.get("run_id", "")))
                       or not row["run_id"].startswith("r2-")
                       or not isinstance(row.get("rho"), (int, float))
                       or not math.isfinite(row["rho"]) or row["rho"] < 0 for row in rows)
                or len({row["run_id"] for row in rows}) != 3):
            raise RuntimeError("Round 1 chooser did not publish exactly three round 2 candidates")
    else:
        for key in ("selected_run_id", "selected_job_id", "selected_result_dir", "selected_epoch",
                    "selected_round", "rho", "latent_weight", "normalization_c", "rounds_complete",
                    "selection_job_id", "selection_source_jobs"):
            if output.get(key) != choice.get(key):
                raise RuntimeError(f"Optimal selection does not match chooser field {key}")
        if output.get("rounds_complete") != 2:
            raise RuntimeError("Final selection was not frozen after round 2")
    evidence = {**observed, "status": "complete", "summary": summary, "choice": choice,
                "round2_candidates": output if round_number == 1 else None,
                "optimal_selection": output if round_number == 2 else None,
                "result_dir": result, "verified_terminal": True}
    save(HERE / f"choice_round{round_number}_terminal.json", evidence)
    return evidence


def _verify_selected_source(transport, selected: dict) -> dict:
    run_id = selected.get("selected_run_id")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise RuntimeError("Frozen selected run ID is invalid")
    source = _verify_search(transport, run_id, require_complete=True)
    summary = source["summary"]
    if (selected.get("selected_job_id") != source.get("job_id")
            or selected.get("selected_result_dir") != source.get("result_dir")
            or selected.get("selected_round") != summary.get("round")
            or selected.get("rho") != summary.get("rho")
            or selected.get("normalization_c") != summary.get("normalization_c")
            or selected.get("selected_epoch") not in (0, 1, 2)):
        raise RuntimeError("Frozen final selection no longer matches its completed search run")
    return source


def final_submit(transport, attempt: str) -> dict:
    selected = choice_verify(transport, 2)["optimal_selection"]
    _verify_selected_source(transport, selected)
    if selected.get("rounds_complete") != 2:
        raise RuntimeError("Final evaluation requires the verified round 2 frozen choice")
    variables = {"FW_RUN_ID": "final", "FW_RHO": str(float(selected["rho"])),
                 "FW_ROUND": "2", "FW_MODE": "final", "FW_FREEZE_FILE": ROOT + "/optimal_selection.json"}
    return _submit(transport, name=_job_name("fn", attempt), script="gpu.pbs",
                   intent_path=HERE / f"final_intent_{attempt}.json",
                   handle_path=HERE / "final_handle.json", variables=variables,
                   select="select=1:ncpus=16:mem=110gb:ngpus=1", walltime="02:00:00")


def final_verify(transport) -> dict:
    p = protocol()
    handle = _handle(HERE / "final_handle.json")
    job = handle["job_id"]
    artifact = f"{ROOT}/artifacts/{job}"
    observed, summary = _terminal(transport, job, artifact, "PIPELINE_COMPLETE")
    result = f"{ROOT}/results/final-{job}"
    result_summary = _remote_json(transport, result + "/summary.json")
    selected = choice_verify(transport, 2)["optimal_selection"]
    _verify_selected_source(transport, selected)
    if summary != result_summary:
        raise RuntimeError("Final artifact and result summaries differ")
    binding = ("selected_run_id", "selected_job_id", "selected_result_dir", "selected_epoch",
               "selected_round", "rho", "latent_weight", "normalization_c",
               "selection_job_id", "selection_source_jobs")
    if (summary.get("status") != "complete" or summary.get("protocol") != PROTOCOL
            or summary.get("mode") != "final" or summary.get("run_id") != "final"
            or summary.get("job_id") != job or summary.get("rounds_complete") != 2
            or summary.get("calibration_context_count") != 8
            or summary.get("test_context_count") != 44 or summary.get("final_test_context_count") != 44
            or summary.get("test_forward_count") != 44
            or summary.get("teacher_forward_count") != 0
            or summary.get("parent_test_future_forward_count") != 0
            or summary.get("training_performed") is not False or summary.get("updates") != 0
            or summary.get("target_linears") != 614
            or summary.get("protocol") != p["protocol"]
            or any(summary.get(key) != selected.get(key) for key in binding)):
        raise RuntimeError("Final evaluation does not match the frozen choice or 8/44 context gates")
    final_markers = f"test -f {shlex.quote(result + '/FINAL_COMPLETE')}"
    rc, _ = remote.command(transport, final_markers, timeout=25)
    if rc:
        raise RuntimeError("Final result directory lacks FINAL_COMPLETE")
    evidence = {**observed, "status": "complete", "summary": summary, "choice": selected,
                "result_dir": result, "verified_terminal": True}
    save(HERE / "final_terminal.json", evidence)
    return evidence


def analysis_submit(transport, attempt: str) -> dict:
    final = final_verify(transport)
    final_job = final["job_id"]
    return _submit(transport, name=_job_name("an", attempt), script="analysis.pbs",
                   intent_path=HERE / f"analysis_intent_{attempt}.json",
                   handle_path=HERE / "analysis_handle.json",
                   variables={"FW_FINAL_JOB": final_job},
                   select="select=1:ncpus=2:mem=8gb", walltime="00:10:00")


def analysis_status(transport) -> dict:
    handle = _handle(HERE / "analysis_handle.json")
    observed = scheduler(transport, handle["job_id"])
    save(HERE / "analysis_scheduler.json", observed)
    print(json.dumps(observed))
    return observed


def analysis_verify(transport) -> dict:
    handle = _handle(HERE / "analysis_handle.json")
    job = handle["job_id"]
    artifact = f"{ROOT}/artifacts/analysis-{job}"
    observed, artifact_summary = _terminal(transport, job, artifact, "PIPELINE_COMPLETE")
    result = f"{ROOT}/results/analysis-{job}"
    final_evidence = final_verify(transport)
    final, choice = final_evidence["summary"], final_evidence["choice"]
    checks = (f"test -f {shlex.quote(artifact + '/SEARCH_REPORT_COMPLETE')} && "
              f"test -f {shlex.quote(result + '/SEARCH_REPORT_COMPLETE')} && "
              f"test -s {shlex.quote(result + '/summary.json')} && "
              f"test -s {shlex.quote(result + '/report.zh.md')} && "
              f"test -s {shlex.quote(result + '/comparison.json')}")
    rc, _ = remote.command(transport, checks, timeout=25)
    if rc:
        raise RuntimeError("Analysis report outputs or SEARCH_REPORT_COMPLETE marker are missing")
    summary = _remote_json(transport, result + "/summary.json")
    selected_fields = ("selected_run_id", "selected_job_id", "selected_epoch", "selected_round",
                       "rho", "latent_weight", "normalization_c")
    if (summary != artifact_summary or summary.get("status") != "complete"
            or summary.get("protocol") != PROTOCOL or summary.get("job_id") != job
            or summary.get("final_job_id") != final.get("job_id")
            or summary.get("choice_job_id") != choice.get("selection_job_id")
            or summary.get("calibration_context_count") != 8
            or summary.get("selection_context_count") != 8
            or summary.get("selection_point_count") != (len(R1_RUNS) + 3) * 3
            or summary.get("test_context_count") != 44
            or summary.get("model_forward_count") != 0
            or summary.get("teacher_forward_count") != 0
            or summary.get("parent_test_future_forward_count") != 0
            or summary.get("test_previously_seen") is not True
            or any(summary.get(key) != choice.get(key) for key in selected_fields)):
        raise RuntimeError("Analysis summary does not match final/choice bindings or frozen coverage")
    evidence = {**observed, "status": "complete", "summary": summary,
                "result_dir": result, "verified_terminal": True}
    save(HERE / "analysis_terminal.json", evidence)
    return evidence


def continue_final(transport) -> dict:
    handle = _handle(HERE / "final_handle.json")
    old_job = handle["job_id"]
    observed = scheduler(transport, old_job)
    if (observed.get("query_rc") != 0 or observed.get("job_state") not in ("F", "X")
            or observed.get("Exit_status") != "0"):
        raise RuntimeError(f"Final job is not a successful terminal allocation: {observed}")
    artifact = f"{ROOT}/artifacts/{old_job}"
    result = f"{ROOT}/results/final-{old_job}"
    summary = _remote_json(transport, result + "/summary.json")
    artifact_summary = _remote_json(transport, artifact + "/summary.json")
    selected = choice_verify(transport, 2)["optimal_selection"]
    _verify_selected_source(transport, selected)
    binding = ("selected_run_id", "selected_job_id", "selected_result_dir", "selected_epoch",
               "selected_round", "rho", "latent_weight", "normalization_c",
               "selection_job_id", "selection_source_jobs")
    if (summary != artifact_summary or summary.get("status") != "resumable"
            or summary.get("mode") != "final" or summary.get("job_id") != old_job
            or summary.get("run_id") != "final" or summary.get("round") != 2
            or summary.get("rounds_complete") != 2 or summary.get("target_linears") != 614
            or any(summary.get(key) != selected.get(key) for key in binding)):
        raise RuntimeError("Final continuation is not bound to the exact frozen selection")
    checks = (f"test -f {shlex.quote(artifact + '/RESUME_READY')} && "
              f"test -f {shlex.quote(artifact + '/exit_code.txt')} && "
              f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0 && "
              f"test -f {shlex.quote(result + '/CHECKPOINT_READY')} && "
              f"test -s {shlex.quote(result + '/checkpoint.pt')} && "
              f"test -s {shlex.quote(result + '/checkpoint_receipt.json')}")
    rc, _ = remote.command(transport, checks, timeout=25)
    if rc:
        raise RuntimeError("Final resumable job lacks its checkpoint and continuation markers")
    checkpoint = _remote_json(transport, result + "/checkpoint_receipt.json")
    if (checkpoint.get("schema_version") != 2 or checkpoint.get("job_id") != old_job
            or checkpoint.get("mode") != "final"
            or checkpoint.get("run_id") != "final"
            or checkpoint.get("candidate_run_id") != selected.get("selected_run_id")
            or checkpoint.get("round") != 2
            or checkpoint.get("selected_epoch") != selected.get("selected_epoch")
            or checkpoint.get("selected_round") != selected.get("selected_round")
            or checkpoint.get("target_linears") != 614
            or checkpoint.get("updates_completed") != 0
            or checkpoint.get("rho") != selected.get("rho")
            or checkpoint.get("latent_weight") != selected.get("latent_weight")
            or checkpoint.get("normalization_c") != selected.get("normalization_c")
            or checkpoint.get("checkpoint") != result + "/checkpoint.pt"):
        raise RuntimeError("Final checkpoint receipt does not preserve the frozen choice")
    save(HERE / "final_terminal.json", {**observed, "status": "resumable",
                                         "summary": summary, "checkpoint_receipt": checkpoint,
                                         "resume_ready": True})
    attempt = _resume_attempt("final", old_job, lambda label: _job_name("fn", label))
    if not ATTEMPT.fullmatch(attempt):
        raise RuntimeError("Too many final continuation attempts")
    variables = {"FW_RUN_ID": "final", "FW_RHO": str(float(selected["rho"])),
                 "FW_ROUND": "2", "FW_MODE": "final", "FW_FREEZE_FILE": ROOT + "/optimal_selection.json",
                 "FW_RESUME_JOB": old_job}
    return _submit(transport, name=_job_name("fn", attempt), script="gpu.pbs",
                   intent_path=HERE / f"final_intent_{attempt}.json",
                   handle_path=HERE / "final_handle.json", variables=variables,
                   select="select=1:ncpus=16:mem=110gb:ngpus=1", walltime="02:00:00",
                   previous_job=old_job)


def _status(transport, handle_path: Path, snapshot_path: Path) -> dict:
    handle = _handle(handle_path)
    job = handle["job_id"]
    observed = scheduler(transport, job)
    save(snapshot_path, observed)
    print(json.dumps(observed))
    if observed.get("job_state") in ("R", "F", "X"):
        log = f"{ROOT}/artifacts/{job}/job.log"
        rc, output = remote.command(
            transport, f"test ! -f {shlex.quote(log)} || tail -c 6000 {shlex.quote(log)}", timeout=25
        )
        print("LOG_TAIL_RC", rc)
        print(output[-6000:])
    return observed


def pull(transport, names: list[str]) -> None:
    if not names:
        raise ValueError("pull requires --files")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() == ".pt":
            raise ValueError("Only small non-tensor control and summary files may be pulled")
    trusted.pull(transport, names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "upload", "pull", "selfcheck-submit", "selfcheck-status", "selfcheck-verify",
        "reference-submit", "reference-status", "reference-verify", "submit", "status", "verify",
        "continue", "choose-submit", "choose-status", "choose-verify", "final-submit",
        "final-status", "final-verify", "analysis-submit", "analysis-status", "analysis-verify"))
    parser.add_argument("--run-id")
    parser.add_argument("--rho", type=float)
    parser.add_argument("--round", dest="round_number", type=int)
    parser.add_argument("--mode", choices=("search",))
    parser.add_argument("--attempt", default="a")
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    if not ATTEMPT.fullmatch(args.attempt):
        parser.error("--attempt must be a short lowercase token")
    if args.action in ("upload", "pull"):
        if args.action == "pull" and not args.files:
            parser.error("pull requires --files")
    elif args.action.startswith("selfcheck-"):
        pass
    elif args.action.startswith("reference-"):
        if args.action == "reference-submit":
            pass
    elif args.action in ("submit", "status", "verify", "continue"):
        if not args.run_id:
            parser.error(f"{args.action} requires --run-id")
        if args.action == "continue" and args.run_id == "final":
            pass
        elif not RUN_ID.fullmatch(args.run_id):
            parser.error("run-id must use r1-rNNN or r2-rNNN")
        if args.action == "submit" and (args.rho is None or args.round_number is None or args.mode != "search"):
            parser.error("submit requires --rho, --round, and --mode search")
    elif args.action.startswith("choose-"):
        if args.round_number not in (1, 2):
            parser.error("chooser actions require --round 1 or 2")
    elif args.action.startswith("analysis-"):
        pass
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        if args.action == "upload":
            trusted.upload(transport, args.files)
        elif args.action == "pull":
            pull(transport, args.files)
        elif args.action == "selfcheck-submit":
            print(json.dumps(selfcheck_submit(transport, args.attempt)))
        elif args.action == "selfcheck-status":
            _status(transport, HERE / "selfcheck_handle.json", HERE / "selfcheck_scheduler.json")
        elif args.action == "selfcheck-verify":
            print(json.dumps(selfcheck_verify(transport)))
        elif args.action == "reference-submit":
            print(json.dumps(reference_submit(transport, args.attempt)))
        elif args.action == "reference-status":
            _status(transport, HERE / "reference_handle.json", HERE / "reference_scheduler.json")
        elif args.action == "reference-verify":
            print(json.dumps(reference_verify(transport)))
        elif args.action == "submit":
            print(json.dumps(search_submit(transport, args.run_id, args.rho, args.round_number, args.attempt)))
        elif args.action == "status":
            _status(transport, HERE / f"run_{args.run_id}_handle.json",
                    HERE / f"run_{args.run_id}_scheduler.json")
        elif args.action == "verify":
            print(json.dumps(search_verify(transport, args.run_id)))
        elif args.action == "continue":
            if args.run_id == "final":
                print(json.dumps(continue_final(transport)))
            else:
                print(json.dumps(continue_search(transport, args.run_id)))
        elif args.action == "choose-submit":
            print(json.dumps(choice_submit(transport, args.round_number, args.attempt)))
        elif args.action == "choose-status":
            _status(transport, HERE / f"choice_r{args.round_number}_handle.json",
                    HERE / f"choice_r{args.round_number}_scheduler.json")
        elif args.action == "choose-verify":
            print(json.dumps(choice_verify(transport, args.round_number)))
        elif args.action == "final-submit":
            print(json.dumps(final_submit(transport, args.attempt)))
        elif args.action == "final-status":
            _status(transport, HERE / "final_handle.json", HERE / "final_scheduler.json")
        elif args.action == "analysis-submit":
            print(json.dumps(analysis_submit(transport, args.attempt)))
        elif args.action == "analysis-status":
            print(json.dumps(analysis_status(transport)))
        elif args.action == "analysis-verify":
            print(json.dumps(analysis_verify(transport)))
        else:
            print(json.dumps(final_verify(transport)))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
