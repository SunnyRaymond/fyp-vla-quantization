"""One-off CPU schema check and bounded PBS control for action-latent analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket

HERE = Path(__file__).resolve().parent
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-action-latent-20261014"
JOB_NAME = "fwvzl1014chk"
JOB_NAME_B = "fwvzl1014ckb"
OLD_ACTION_ROOT = Path("/scratch/users/ntu/yguo017/fastwam-vq-action-distill-20261011")
OLD_ACTION_JOB = "25764781.pbs101"
LIMIT = 256 * 1024


def worker_guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not os.environ.get("PBS_JOBID", "").strip() or not nodefile or "login" in host.lower():
        raise RuntimeError("CPU analysis check requires a genuine PBS compute allocation")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError("This host is outside the PBS allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("CPU analysis check must not expose CUDA devices")


def summarize_future(analyze, rows, cases, seeds, label):
    aggregated = analyze.aggregate_future_rows(rows, cases, seeds, label)
    return {metric: analyze.mean_summary(aggregated[metric]) for metric in analyze.FUTURE_METRICS}


def worker(protocol_path):
    worker_guard()
    from test_action_latent_analysis import self_check

    self_check()
    import torch
    import analyze_results as analyze

    protocol = analyze.read_json(protocol_path)
    analyze.require(protocol.get("protocol") == "fastwam-vq-action-latent-v1",
                    "Unexpected action-latent protocol")
    analyze.require(protocol.get("remote_root") == ROOT, "Remote protocol root mismatch")
    cases, seeds = protocol["test_cases"], protocol["evaluation_seed_indices"]
    calibration_cases = protocol["calibration_cases"]
    calibration_seeds = protocol["calibration_seed_indices"]
    selection_cases = protocol["selection_cases"]
    selection_seeds = protocol["evaluation_seed_indices"]
    analyze.require(len(cases) == 22 and len(seeds) == 2, "Frozen test cohort must be 22 cases x 2 seeds")
    analyze.require(len(calibration_cases) == 8 and calibration_seeds == [0],
                    "Frozen calibration cohort must contain eight seed-0 contexts")
    analyze.require(len(selection_cases) == 4 and len(selection_seeds) == 2,
                    "Frozen selection cohort must contain four cases x 2 seeds")

    reference_summary, teacher_actions = analyze.read_reference_cache(protocol, torch)
    (source_summary, source_arms, _metadata, _source_rows, source_values, source_selection,
     _selection_teacher, _selection_values, source_cal_rows, _source_cal_values) = analyze.read_source_actions(
        protocol, cases, seeds, teacher_actions)
    source_test = {}
    for arm in source_arms:
        source_test[arm] = analyze.summarize_action_rows(source_values[arm], cases, seeds)
        expected = source_summary["test"][arm]["mean_motor_rmse_first10"]
        actual = source_test[arm]["first10motor_rmse"]["mean"]
        analyze.close(actual, expected, f"source {arm} summary first10 motor RMSE", 1e-6)
    analyze.require(len(source_arms) == 7, "Expected all seven source reference arms")
    parent_arm = protocol["initial_arm"]
    analyze.require(len(source_cal_rows[parent_arm]) == 8, "Parent calibration helper coverage is not 8")
    analyze.require(len(source_selection["bf16"]) == 8 and len(source_selection[parent_arm]) == 8,
                    "BF16/parent selection helper coverage is not 8 each")

    old_result = OLD_ACTION_ROOT / "results" / f"full-{OLD_ACTION_JOB}"
    old_summary = analyze.read_json(old_result / "summary.json")
    analyze.require(old_summary.get("status") == "complete" and old_summary.get("job_id") == OLD_ACTION_JOB,
                    "Old action-only summary identity or status mismatch")
    analyze.require(old_summary.get("selected_epoch") == 2,
                    "Frozen old action-only evaluation must use selected epoch 2")
    payload = torch.load(old_result / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("run_state")
    analyze.require(isinstance(state, dict) and isinstance(state.get("evaluation_rows"), list),
                    "Old action-only checkpoint has no evaluation_rows")
    old_rows = list(state["evaluation_rows"])
    del payload
    old_test = analyze.rows_for_eval(old_rows, "test", 2, cases, seeds, "old action-only test")
    old_cal = analyze.rows_for_eval(old_rows, "calibration", 2, calibration_cases, calibration_seeds,
                                    "old action-only calibration")
    old_test_values = analyze.values_from_eval(old_test, teacher_actions, "old action-only test")
    old_cal_values = analyze.values_from_eval(old_cal, teacher_actions, "old action-only calibration")
    old_test_summary = analyze.summarize_action_rows(old_test_values, cases, seeds)
    old_cal_summary = analyze.summarize_action_rows(old_cal_values, calibration_cases, calibration_seeds)
    analyze.close(old_test_summary["first10motor_rmse"]["mean"],
                  old_summary["test"]["mean_motor_rmse_first10"],
                  "old action-only summary test first10 motor RMSE", 1e-6)
    analyze.close(old_cal_summary["first10motor_rmse"]["mean"],
                  old_summary["calibration"]["mean_motor_rmse_first10"],
                  "old action-only summary calibration first10 motor RMSE", 1e-6)

    diagnostic = analyze.read_diagnostic_future(protocol)
    analyze.require(len(diagnostic["parent_self"]) == 60 and len(diagnostic["distilled_self"]) == 60,
                    "Parent/distilled future diagnostics must cover all 60 contexts")
    test_keys = lambda rows: {key: row for key, row in rows.items() if key[0] in cases}
    calibration_keys = lambda rows: {key: row for key, row in rows.items() if key[0] in calibration_cases}
    parent_future_test = summarize_future(analyze, test_keys(diagnostic["parent_self"]), cases, seeds,
                                          "parent future test")
    old_future_test = summarize_future(analyze, test_keys(diagnostic["distilled_self"]), cases, seeds,
                                       "old action-only future test")
    parent_future_cal = summarize_future(analyze, calibration_keys(diagnostic["parent_self"]),
                                         calibration_cases, calibration_seeds, "parent future calibration")
    old_future_cal = summarize_future(analyze, calibration_keys(diagnostic["distilled_self"]),
                                      calibration_cases, calibration_seeds,
                                      "old action-only future calibration")

    job_id = os.environ["PBS_JOBID"]
    summary = {
        "status": "complete",
        "job_id": job_id,
        "protocol": protocol["protocol"],
        "source_reference_arms": source_arms,
        "source_reference_arm_count": len(source_arms),
        "test_case_count": len(cases),
        "test_context_count": len(cases) * len(seeds),
        "calibration_context_count": len(calibration_cases) * len(calibration_seeds),
        "selection_context_count": len(selection_cases) * len(selection_seeds),
        "reference_contexts": reference_summary["context_count"],
        "old_action_only_test_contexts": len(old_test),
        "old_future_contexts": len(diagnostic["distilled_self"]),
        "parent_future_contexts": len(diagnostic["parent_self"]),
        "self_check": "passed",
        "new_model_forwards": 0,
        "source_action_test_means": source_test,
        "old_action_only_test_means": old_test_summary,
        "old_action_only_calibration_means": old_cal_summary,
        "future_test_means": {"parent_self": parent_future_test,
                               "old_action_only": old_future_test},
        "future_calibration_means": {"parent_self": parent_future_cal,
                                     "old_action_only": old_future_cal},
    }
    artifact_dir = Path(os.environ["ARTIFACTS"])
    encoded = (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    analyze.require(len(encoded) <= LIMIT, "CPU analysis summary exceeds 256 KiB")
    (artifact_dir / "summary.json").write_bytes(encoded)
    print("CPU_ANALYSIS_CHECK_READY", json.dumps({key: summary[key] for key in (
        "job_id", "protocol", "source_reference_arm_count", "test_case_count", "test_context_count",
        "calibration_context_count", "reference_contexts", "self_check", "new_model_forwards")}))


def local_files():
    files = (HERE / "cpu_analysis_check.py", HERE / "cpu_analysis_check.pbs")
    for path in files:
        if not path.is_file() or path.stat().st_size > LIMIT:
            raise RuntimeError(f"Missing or oversized control file: {path.name}")
    return files


def load_control():
    spec = importlib.util.spec_from_file_location("cpu_analysis_existing_control", HERE / "control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submit():
    control = load_control()
    remote = control.remote
    intent_path = HERE / "cpu_analysis_check_intent.json"
    handle_path = HERE / "cpu_analysis_check_handle.json"
    if handle_path.is_file():
        print(handle_path.read_text(encoding="utf-8"))
        return
    intent = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.is_file() else None
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        rc, output = remote.command(transport,
                                    f"qselect -x -u yguo017 -N {shlex.quote(JOB_NAME)} 2>&1", timeout=35)
        if rc not in (0, 1) or (rc == 1 and output.strip()):
            raise RuntimeError("Scheduler submission history is unavailable")
        matches = output.split()
        if matches:
            if len(matches) != 1 or not control.trusted.HANDLE.fullmatch(matches[0]):
                raise RuntimeError("Ambiguous existing CPU-check submission")
            receipt = {**(intent or {}), "job_id": matches[0], "job_name": JOB_NAME, "status": "recovered"}
            control.save(handle_path, receipt)
            control.save(intent_path, receipt)
            print(json.dumps(receipt))
            return
        if intent:
            raise RuntimeError("Unconfirmed CPU-check intent exists; refusing duplicate qsub")

        files = local_files()
        rc, output = remote.command(transport, "mkdir -p " + shlex.quote(ROOT), timeout=25)
        if rc:
            raise RuntimeError(f"Unable to create remote experiment directory: {output}")
        with remote.sftp_client(transport) as sftp:
            for path in files:
                target = f"{ROOT}/{path.name}"
                data = path.read_bytes().replace(b"\r\n", b"\n")
                with sftp.file(target, "wb") as stream:
                    stream.write(data)
        receipt = {"status": "intent_before_qsub", "job_name": JOB_NAME, "remote_root": ROOT,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "queue": "normal",
                   "project": "personal-yguo017", "select": "select=1:ncpus=2:mem=8gb",
                   "walltime": "00:10:00"}
        control.save(intent_path, receipt)
        rc, output = remote.command(transport, "qsub " + shlex.quote(ROOT + "/cpu_analysis_check.pbs"),
                                    timeout=40)
        if rc or not control.trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status="unknown_or_rejected", message=output[:1000])
            control.save(intent_path, receipt)
            raise RuntimeError("qsub outcome is unknown or rejected; do not resubmit")
        receipt.update(job_id=output.strip(), status="confirmed")
        control.save(intent_path, receipt)
        control.save(handle_path, receipt)
        print(json.dumps(receipt))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


def submit_b():
    control = load_control()
    remote = control.remote
    intent_path = HERE / "cpu_analysis_check_intent_b.json"
    handle_path = HERE / "cpu_analysis_check_handle.json"
    old_job = "25798769.pbs101"
    if not handle_path.is_file() or handle_path.stat().st_size > LIMIT:
        raise RuntimeError("The confirmed attempt-a handle is missing or oversized")
    old_handle = json.loads(handle_path.read_text(encoding="utf-8"))
    if old_handle.get("job_id") != old_job:
        raise RuntimeError("Current handle no longer identifies attempt-a")
    intent = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.is_file() else None
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        observed = control.scheduler(transport, old_job)
        if observed["query_rc"] or observed["job_state"] not in ("F", "X") or observed["Exit_status"] != "1":
            raise RuntimeError(f"Attempt-a is not confirmed F/Exit_status=1: {observed}")
        rc, output = remote.command(transport,
                                    f"qselect -x -u yguo017 -N {shlex.quote(JOB_NAME_B)} 2>&1", timeout=35)
        if rc not in (0, 1) or (rc == 1 and output.strip()):
            raise RuntimeError("Attempt-b submission history is unavailable")
        matches = output.split()
        if matches:
            if len(matches) != 1 or not control.trusted.HANDLE.fullmatch(matches[0]):
                raise RuntimeError("Ambiguous attempt-b submission history")
            receipt = {**(intent or {}), "job_id": matches[0], "job_name": JOB_NAME_B,
                       "previous_job": old_job, "status": "recovered"}
            control.save(intent_path, receipt)
            control.save(handle_path, receipt)
            print(json.dumps(receipt))
            return
        if intent:
            raise RuntimeError("Attempt-b intent exists but qselect found no job; refusing duplicate qsub")

        rc, output = remote.command(transport, "mkdir -p " + shlex.quote(ROOT), timeout=25)
        if rc:
            raise RuntimeError(f"Unable to access remote experiment directory: {output}")
        with remote.sftp_client(transport) as sftp:
            for path in local_files():
                target = f"{ROOT}/{path.name}"
                data = path.read_bytes().replace(b"\r\n", b"\n")
                with sftp.file(target, "wb") as stream:
                    stream.write(data)
        receipt = {"status": "intent_before_qsub", "job_name": JOB_NAME_B, "remote_root": ROOT,
                   "previous_job": old_job, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "queue": "normal", "project": "personal-yguo017",
                   "select": "select=1:ncpus=2:mem=8gb", "walltime": "00:10:00"}
        control.save(intent_path, receipt)
        rc, output = remote.command(transport, "qsub " + shlex.quote(ROOT + "/cpu_analysis_check.pbs"),
                                    timeout=40)
        if rc or not control.trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status="unknown_or_rejected", message=output[:1000])
            control.save(intent_path, receipt)
            raise RuntimeError("Attempt-b qsub outcome is unknown or rejected; recover with qselect, do not resubmit")
        receipt.update(job_id=output.strip(), status="confirmed")
        control.save(intent_path, receipt)
        control.save(handle_path, receipt)
        print(json.dumps(receipt))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


def terminal():
    control = load_control()
    remote = control.remote
    handle_path = HERE / "cpu_analysis_check_handle.json"
    if not handle_path.is_file() or handle_path.stat().st_size > LIMIT:
        raise RuntimeError("No bounded CPU-check handle is available")
    handle = json.loads(handle_path.read_text(encoding="utf-8"))
    job_id = handle.get("job_id")
    if not isinstance(job_id, str) or not control.trusted.HANDLE.fullmatch(job_id):
        raise RuntimeError("CPU-check handle is invalid")
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        observed = control.scheduler(transport, job_id)
        print("SCHEDULER", json.dumps(observed))
        if observed["query_rc"] or observed["job_state"] not in ("F", "X"):
            return
        artifact = f"{ROOT}/artifacts/cpu-check-{job_id}"
        log_rc, log_tail = remote.command(transport,
                                          f"test ! -f {shlex.quote(artifact + '/job.log')} || "
                                          f"tail -c 6000 {shlex.quote(artifact + '/job.log')}", timeout=25)
        print("JOB_LOG_TAIL_RC", log_rc, log_tail)
        if observed["Exit_status"] != "0":
            control.save(HERE / "cpu_analysis_check_terminal.json", {
                "status": "failed", "job_id": job_id, "protocol": "fastwam-vq-action-latent-v1",
                "scheduler": observed, "job_log_tail": log_tail[:6000],
            })
            raise RuntimeError(f"CPU-check job ended unsuccessfully: {observed}")
        checks = (f"test -f {shlex.quote(artifact + '/CPU_ANALYSIS_CHECK_COMPLETE')} && "
                  f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0 && "
                  f"test -s {shlex.quote(artifact + '/summary.json')} && "
                  f"test \"$(wc -c < {shlex.quote(artifact + '/summary.json')})\" -le {LIMIT}")
        rc, _ = remote.command(transport, checks, timeout=25)
        if rc:
            raise RuntimeError("CPU-check marker, artifact exit receipt, or summary is missing")
        summary = control.read_remote_json(transport, artifact + "/summary.json")
        expected = {
            "status": "complete", "job_id": job_id, "protocol": "fastwam-vq-action-latent-v1",
            "source_reference_arm_count": 7, "test_case_count": 22, "test_context_count": 44,
            "calibration_context_count": 8, "reference_contexts": 60,
            "old_action_only_test_contexts": 44, "old_future_contexts": 60,
            "parent_future_contexts": 60, "self_check": "passed", "new_model_forwards": 0,
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                raise RuntimeError(f"CPU-check summary {key} mismatch: {summary.get(key)!r} != {value!r}")
        if len(summary.get("source_reference_arms", [])) != 7:
            raise RuntimeError("CPU-check summary does not list all seven source reference arms")
        evidence = {**summary, "terminal_evidence": {"scheduler": observed, "marker": True,
                                                      "artifact_exit_code": 0, "job_log_tail": log_tail}}
        control.save(HERE / "cpu_analysis_check_terminal.json", evidence)
        print(json.dumps(evidence))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=("submit", "submit-b", "status", "verify"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--protocol", default=ROOT + "/protocol.json")
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol)
    elif args.action == "submit":
        submit()
    elif args.action == "submit-b":
        submit_b()
    elif args.action in ("status", "verify"):
        terminal()
    else:
        parser.error("choose submit/submit-b/status/verify or --worker")


if __name__ == "__main__":
    main()
