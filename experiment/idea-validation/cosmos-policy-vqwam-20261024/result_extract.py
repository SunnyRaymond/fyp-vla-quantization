#!/usr/bin/env python3
"""Export a compact, completion-gated report from the finished PBS results."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

MAX_BYTES = 256 * 1024
JOB_NAME = "cpvq24extract"
JOB_NAMES = {JOB_NAME, "cpvq24grip"}
EVALUATION_NAME = "cpvq24eval"
ARMS = ("bf16", "spinquant_w4a4", "quarot_w4a4", "svdquant_w4a4", "viditq_w4a4", "ours_vqwam")
NEW_ARMS = ARMS[2:]
REUSED_ARMS = ARMS[:2]
WORKERS = tuple(f"cosmos-gpu-worker-{i}" for i in range(8))
PBS_ID = re.compile(r"^\d+\.[A-Za-z0-9_.-]+$")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def counts(value: Any, arms: tuple[str, ...], expected: int, label: str) -> None:
    require(isinstance(value, dict) and set(value) == set(arms), f"{label} map incomplete")
    require(all(type(value[a]) is int and value[a] == expected for a in arms),
            f"{label} must be {expected} for every arm")


def no_missing(value: Any, arms: tuple[str, ...], label: str) -> None:
    require(isinstance(value, dict) and set(value) == set(arms), f"{label} map incomplete")
    require(all(value[a] == [] for a in arms), f"{label} contains missing slots")


def short_host(value: Any) -> str:
    return str(value).strip().lower().split(".", 1)[0]


def validate_nodes(protocol: dict[str, Any], summary: dict[str, Any], job_id: str,
                   launch: dict[str, Any], receipts: list[dict[str, Any]],
                   recovery: bool = False) -> list[dict[str, Any]]:
    codes = summary.get("worker_exitcodes")
    require(summary.get("expected_worker_count") == 8
            and summary.get("node_receipts_complete") is (False if recovery else True),
            "Evaluation worker/node receipt status does not match extraction mode")
    require(isinstance(codes, dict) and set(codes) == set(WORKERS)
            and all(type(code) is int and code in ((0, 1) if recovery else (0,)) for code in codes.values()),
            "Evaluation worker exit codes are incomplete or invalid")
    require(launch.get("protocol") == protocol["protocol"] and launch.get("pbs_job_id") == job_id
            and launch.get("expected_workers") == 8, "Multinode launch receipt does not bind to evaluation")
    nodes = launch.get("nodes")
    require(isinstance(nodes, list) and len(nodes) == 2, "Launch receipt must contain two nodes")
    nodes = sorted(nodes, key=lambda n: n.get("rank", -1))
    hosts = [short_host(n.get("hostname")) for n in nodes]
    require([n.get("rank") for n in nodes] == [0, 1] and all(hosts) and len(set(hosts)) == 2,
            "Launch node ranks/hosts are incomplete")
    require(len(receipts) == 2, "Exactly two node receipts are required")

    seen, compact, merged = set(), [], {}
    for receipt in receipts:
        rank = receipt.get("node_rank")
        require(type(rank) is int and rank in (0, 1) and rank not in seen, "Node receipt rank invalid")
        seen.add(rank)
        require(receipt.get("protocol") == protocol["protocol"] and receipt.get("pbs_job_id") == job_id
                and short_host(receipt.get("host")) == hosts[rank]
                and type(receipt.get("exit_code")) is int
                and receipt["exit_code"] in ((0, 1) if recovery else (0,)),
                f"Node {rank} receipt identity/exit code invalid")
        local = receipt.get("worker_exitcodes")
        expected = {f"cosmos-gpu-worker-{i}" for i in range(4 * rank, 4 * rank + 4)}
        require(isinstance(local, dict) and set(local) == expected
                and all(type(code) is int and code in ((0, 1) if recovery else (0,))
                        for code in local.values()),
                f"Node {rank} worker receipts incomplete")
        require(receipt.get("exit_code") == int(any(code != 0 for code in local.values())),
                f"Node {rank} exit code does not match its worker failures")
        merged.update(local)
        compact.append({"node_rank": rank, "host": hosts[rank], "exit_code": receipt["exit_code"],
                        "worker_exitcodes": dict(sorted(local.items()))})
    require(seen == {0, 1} and merged == codes, "Node receipts disagree with evaluation worker codes")
    require(bool(any(codes.values())) is recovery, "Worker failures do not match extraction mode")
    return sorted(compact, key=lambda x: x["node_rank"])


def select_bucket(bucket: dict[str, Any], label: str) -> dict[str, Any]:
    rates, pairs = bucket.get("arms"), bucket.get("paired")
    require(isinstance(rates, dict) and set(rates) == set(ARMS), f"{label} arm rates incomplete")
    ours = pairs.get("ours_vqwam") if isinstance(pairs, dict) else None
    require(isinstance(ours, dict) and set(ours) == set(ARMS) - {"ours_vqwam"},
            f"{label} ours paired comparisons incomplete")
    for other, pair in ours.items():
        n = pair.get("episodes")
        require(type(n) is int and n > 0 and n == pair.get("expected_episodes"),
                f"{label}/ours_vqwam/{other} paired denominator incomplete")
    return {"arms": rates, "paired": {"ours_vqwam": ours}}


def select_performance(protocol: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
    design = performance.get("query_design", {})
    spec = protocol["performance"]
    require(performance.get("protocol") == protocol["protocol"] and performance.get("status") == "complete"
            and performance.get("reference_arm") == "bf16"
            and performance.get("timing_scope") == spec["latency_scope"],
            "Six-arm performance summary is incomplete or mismatched")
    require(design.get("case_ids") == spec["case_ids"] and design.get("seed_indices") == spec["seed_indices"]
            and design.get("blocks") == spec["blocks"] and design.get("repetitions") == spec["repetitions"]
            and len(spec["case_ids"]) * len(spec["seed_indices"]) == 80
            and spec["blocks"] * spec["repetitions"] == 9, "Performance panel is not the frozen 80 x 9 design")
    source = performance.get("arms")
    require(isinstance(source, dict) and set(source) == set(ARMS), "Six-arm performance map incomplete")
    selected = {}
    for arm in ARMS:
        item = source[arm]
        coverage = item.get("coverage", {})
        require(all(type(coverage.get(k)) is int and coverage[k] == v for k, v in {
            "queries": 80, "completed_queries": 80, "samples": 720,
            "expected_samples": 720, "reused_samples": 0,
        }.items()), f"{arm} performance coverage is not 80 queries/720 samples")
        selected[arm] = {k: item[k] for k in (
            "denoising_wall_seconds", "complete_endpoint_query_seconds", "peak_allocated_gb_decimal",
            "packed_storage_bytes", "packed_storage_gib_binary", "runtime_label", "speedup_over_bf16")}
        selected[arm]["coverage"] = {k: coverage[k] for k in (
            "queries", "completed_queries", "samples", "expected_samples", "reused_samples")}
    return {k: performance[k] for k in (
        "protocol", "reference_arm", "model", "mode", "hardware", "timing_scope",
        "query_design", "execution_boundary") if k in performance} | {"status": "complete", "arms": selected}


def gripper_diagnostics(records: dict) -> dict:
    """Compare generated first chunks from the same frozen initial-state/seed pairs."""
    output = {}
    for arm in ARMS:
        errors, flips, flipped_chunks = [], 0, 0
        for index in range(280):
            candidate, reference = records[arm, index], records['bf16', index]
            for key in ('variant_id', 'suite', 'dimension', 'task_id', 'state_id', 'env_seed', 'sampler_seed'):
                require(candidate.get(key) == reference.get(key), f'Unpaired first chunk: {arm}/{index}/{key}')
            chunks = [item.get('first_action_chunk') for item in (candidate, reference)]
            require(all(isinstance(chunk, list) and len(chunk) == 16
                        and all(isinstance(row, list) and len(row) == 7 for row in chunk)
                        for chunk in chunks), f'Missing 16x7 first action chunk: {arm}/{index}')
            chunk_flips = 0
            for row, base in zip(*chunks):
                value, target = float(row[6]), float(base[6])
                require(math.isfinite(value) and math.isfinite(target), 'Nonfinite gripper scalar')
                errors.append(value - target)
                chunk_flips += ((value > 0) - (value < 0)) != ((target > 0) - (target < 0))
            flips += chunk_flips
            flipped_chunks += bool(chunk_flips)
        output[arm] = {'paired_chunks': 280, 'generated_gripper_values': len(errors),
                       'mae': math.fsum(abs(x) for x in errors) / len(errors),
                       'rmse': math.sqrt(math.fsum(x*x for x in errors) / len(errors)),
                       'sign_disagreements': flips, 'sign_disagreement_rate': flips / len(errors),
                       'chunks_with_sign_disagreement': flipped_chunks}
    return {'reference_arm': 'bf16', 'scope': 'first generated16x7 chunk only; paired initial states/seeds; not full-trajectory error',
            'command_rule': 'sign of raw seventh scalar; no extra OpenVLA remapping; zero is a distinct command',
            'new_model_forwards': 0, 'new_episodes': 0, 'arms': output}


def validate_queue_recovery(receipt: dict[str, Any], source_job: str, extractor_job: str,
                            root: Path, worker_codes: dict[str, int],
                            progress: dict[int, dict[str, Any]]) -> dict[str, str]:
    require(receipt.get("source_evaluation_job") == source_job
            and receipt.get("extractor_pbs_job_id") == extractor_job
            and receipt.get("error_type") == "queue_lock_timeout",
            "Queue recovery receipt does not match the source/extractor jobs and approved error")
    failed = {i for i in range(8) if worker_codes.get(f"cosmos-gpu-worker-{i}") != 0}
    require(bool(failed) and receipt.get("failed_workers") == sorted(failed)
            and set(progress) == failed,
            "Recovery worker set differs from the recorded queue-lock failures")
    expected_error = f"TimeoutError: Timed out waiting for queue lock {root / 'results/evaluation/claim_queue.json.lock'}"
    errors = {}
    for worker in sorted(failed):
        item = progress[worker]
        require(item.get("worker_id") == worker and item.get("run_id") == source_job
                and item.get("status") == "failed" and item.get("error") == expected_error,
                f"Worker {worker} failure is not the authorized queue-lock timeout")
        errors[str(worker)] = item["error"]
    return errors


def build_summary(protocol: dict[str, Any], evaluation: dict[str, Any], performance: dict[str, Any],
                  handle: dict[str, Any], qstat: dict[str, Any], marker: bool,
                  launch: dict[str, Any], receipts: list[dict[str, Any]], extractor_job: str,
                  recovery_receipt: dict[str, Any] | None = None,
                  worker_errors: dict[int, dict[str, Any]] | None = None,
                  recovery_root: Path | None = None) -> dict[str, Any]:
    recovery = recovery_receipt is not None
    require(tuple(protocol.get("arms", ())) == ARMS and tuple(protocol.get("new_arms", ())) == NEW_ARMS
            and tuple(protocol.get("reused_arms", ())) == REUSED_ARMS, "Protocol arm set is not the frozen six-arm set")
    cohort = protocol["cohort"]
    require((cohort.get("episodes_per_arm"), cohort.get("total_episodes"), cohort.get("reused_episodes"),
             cohort.get("comparison_episodes")) == (280, 1120, 560, 1680), "Frozen cohort denominators mismatch")
    job_id = handle.get("job_id")
    require(handle.get("stage") == "evaluation" and handle.get("job_name") == EVALUATION_NAME
            and handle.get("remote_root") == protocol.get("remote_root")
            and isinstance(job_id, str) and PBS_ID.fullmatch(job_id), "Current evaluation handle identity invalid")
    require(evaluation.get("protocol") == protocol["protocol"] and evaluation.get("pbs_job_id") == job_id,
            "Evaluation summary does not belong to the current handle")
    require(qstat.get("query_rc") == 0 and qstat.get("Job_Name") == EVALUATION_NAME
            and qstat.get("job_state") == "F", "Evaluation PBS job is not verified terminal F")
    if recovery:
        require(evaluation.get("status") == "failed" and evaluation.get("complete") is False
                and qstat.get("Exit_status") == "1" and marker is False,
                "Queue recovery requires source F/Exit 1, failed summary, and no source marker")
    else:
        require(evaluation.get("status") == "complete" and evaluation.get("complete") is True
                and qstat.get("Exit_status") == "0" and marker is True,
                "Normal extraction requires source F/Exit 0 and PIPELINE_COMPLETE")
    require(isinstance(extractor_job, str) and PBS_ID.fullmatch(extractor_job) and extractor_job != job_id,
            "Current extractor PBS job ID invalid")
    require(evaluation.get("new_arms") == list(NEW_ARMS) and evaluation.get("reused_arms") == list(REUSED_ARMS),
            "Evaluation arm mapping differs from protocol")
    require(evaluation.get("new_arms_status") == "complete", "New arms are not complete")
    for key in ("episode_slots", "expected_episode_slots", "new_episode_slots", "expected_new_episode_slots"):
        require(evaluation.get(key) == 1120, f"Evaluation {key} is not 1120")
    counts(evaluation.get("new_arm_episodes"), NEW_ARMS, 280, "New-arm episode counts")
    counts(evaluation.get("arm_episodes"), NEW_ARMS, 280, "New-arm slot counts")
    no_missing(evaluation.get("missing_new_slots_by_arm"), NEW_ARMS, "New-arm missing slots")
    no_missing(evaluation.get("missing_slots_by_arm"), NEW_ARMS, "New-slot missing slots")
    counts(evaluation.get("reused_arm_episodes"), REUSED_ARMS, 280, "Reused-arm episode counts")
    no_missing(evaluation.get("missing_reused_slots_by_arm"), REUSED_ARMS, "Reused-arm missing slots")
    all_methods = evaluation.get("all_methods", {})
    require(all_methods.get("episode_slots") == 1680 and all_methods.get("expected_episode_slots") == 1680,
            "All-method denominator is not 1680")
    counts(all_methods.get("arm_episodes"), ARMS, 280, "All-method episode counts")
    no_missing(all_methods.get("missing_slots_by_arm"), ARMS, "All-method missing slots")
    node_receipts = validate_nodes(protocol, evaluation, job_id, launch, receipts, recovery=recovery)
    if recovery:
        require(recovery_root is not None and isinstance(worker_errors, dict), "Recovery evidence is missing")
        worker_errors = validate_queue_recovery(recovery_receipt, job_id, extractor_job,
                                                recovery_root, evaluation["worker_exitcodes"], worker_errors)

    strata = {"overall": select_bucket(all_methods["overall"], "overall")}
    for field, expected in (("suites", cohort["suites"]), ("dimensions", cohort["perturbation_dimensions"])):
        source = all_methods.get(field)
        require(isinstance(source, dict) and len(source) == expected, f"{field} strata incomplete")
        strata[field] = {key: select_bucket(value, f"{field}/{key}") for key, value in source.items()}
    cells = all_methods.get("cells")
    require(isinstance(cells, dict) and len(cells) == cohort["suites"] * cohort["perturbation_dimensions"],
            "Cell strata incomplete")
    strata["cells"] = {key: select_bucket(value, f"cells/{key}") for key, value in cells.items()}
    require(set(all_methods.get("timing_and_memory", {})) == set(ARMS)
            and set(all_methods.get("encoding_evidence", {})) == set(ARMS),
            "All-method timing or encoding evidence map incomplete")
    provenance = evaluation.get("reused_provenance")
    require(isinstance(provenance, dict) and set(provenance) == set(REUSED_ARMS), "Reused provenance incomplete")

    result = {
        "protocol": protocol["protocol"], "status": "complete", "complete": True,
        "source_evaluation_job": job_id, "extractor_pbs_job_id": extractor_job,
        "source_execution_complete": not recovery, "source_exit_status": 1 if recovery else 0,
        "source_pipeline_complete": marker, "data_complete": True,
        "completion": {"new": {"slots": 1120, "by_arm": evaluation["new_arm_episodes"]},
                       "reused": {"slots": 560, "by_arm": evaluation["reused_arm_episodes"]},
                       "all_methods": {"slots": 1680, "by_arm": all_methods["arm_episodes"]},
                       "workers": {"count": 8, "exitcodes": evaluation["worker_exitcodes"],
                                   "node_receipts_verified": True,
                                   "node_workers_successful": not recovery,
                                   "node_receipts": node_receipts}},
        "all_methods": {"provenance_boundary": all_methods.get("provenance_boundary"),
                        **strata, "timing_and_memory": all_methods["timing_and_memory"],
                        "encoding_evidence": all_methods["encoding_evidence"],
                        "reused_provenance": provenance},
        "performance": select_performance(protocol, performance),
        "source_qstat": {k: qstat[k] for k in ("Job_Name", "job_state", "Exit_status")},
    }
    if recovery:
        result.update({"recovery_type": "queue_control_failure_after_full_record_validation",
                       "new_model_forwards": 0, "new_episodes": 0,
                       "source_worker_errors": worker_errors})
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> int:
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError(f"Compact report exceeds {MAX_BYTES} bytes ({len(data)} bytes)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return len(data)


def qstat(job_id: str, full: bool = False) -> tuple[dict[str, Any], str]:
    result = subprocess.run(["qstat", "-xf" if full else "-f", job_id],
                            capture_output=True, text=True, timeout=20, check=False)
    text = result.stdout + "\n" + result.stderr
    fields = {key: (m.group(1).strip() if (m := re.search(
        rf"(?m)^\s*{key}\s*=\s*([^\s]+)", text)) else None)
        for key in ("Job_Name", "job_state", "queue", "Exit_status")}
    fields["query_rc"] = result.returncode
    return fields, text


def guard() -> tuple[str, dict[str, Any]]:
    job, nodefile = os.environ.get("PBS_JOBID", ""), os.environ.get("PBS_NODEFILE", "")
    host = socket.gethostname().lower().split(".", 1)[0]
    require(bool(job and nodefile) and "login" not in host, "Missing PBS allocation or login host")
    require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "Refusing GPU-visible extraction")
    with Path(nodefile).open(encoding="utf-8") as stream:
        hosts = {line.strip().split(".", 1)[0].lower() for line in stream if line.strip()}
    require(host in hosts, "Current host absent from PBS_NODEFILE")
    current, raw = qstat(job)
    require(current.get("query_rc") == 0 and current.get("job_state") == "R"
            and current.get("Job_Name") in JOB_NAMES and current.get("queue") in {"qdev", "q1", "q2"}
            and not re.search(r"ngpus\s*=\s*[1-9]", raw),
            "Current job is not a verified CPU cpvq24extract allocation")
    return job, current


def recover_evaluation(root: Path, protocol: dict[str, Any], handle: dict[str, Any],
                       performance: dict[str, Any], extractor_job: str,
                       recovery_receipt: dict[str, Any]) -> dict[str, Any]:
    job = handle.get("job_id")
    source = read_json(root / "results" / "evaluation" / "summary.json")
    require(source.get("protocol") == protocol["protocol"] and source.get("pbs_job_id") == job
            and source.get("status") == "failed" and source.get("complete") is False,
            "Recovery source summary must be the failed summary for this evaluation job")
    source_qstat, _ = qstat(job, full=True)
    require(source_qstat.get("query_rc") == 0 and source_qstat.get("Job_Name") == EVALUATION_NAME
            and source_qstat.get("job_state") == "F" and source_qstat.get("Exit_status") == "1",
            "Recovery source job must be verified F/Exit_status 1")
    base = root / "artifacts" / job
    require(not (base / "PIPELINE_COMPLETE").exists(), "Failed source unexpectedly has PIPELINE_COMPLETE")
    launch = read_json(base / "evaluation_multinode_launch.json")
    receipts = [read_json(base / "nodes" / f"node_{rank}" / "node_receipt.json") for rank in (0, 1)]
    validate_nodes(protocol, source, job, launch, receipts, recovery=True)
    failed_workers = [i for i in range(8) if source["worker_exitcodes"][f"cosmos-gpu-worker-{i}"] != 0]
    progress = {worker: read_json(base / f"worker_{worker}" / "progress.json") for worker in failed_workers}
    validate_queue_recovery(recovery_receipt, job, extractor_job, root,
                            source["worker_exitcodes"], progress)

    # These imports are stdlib-only on entry. The evaluator uses the frozen helper under protocol.asset_root.
    import evaluate
    import aggregate

    protocol = evaluate.validate_protocol(protocol)
    manifest_path = Path(protocol["cohort"]["manifest"])
    _, rows = evaluate.load_manifest(manifest_path, protocol)
    evaluation_root = root / "results" / "evaluation"
    new_records = evaluate.load_completed(evaluation_root, rows, protocol)
    reused_records = evaluate.load_reused_records(root, rows, protocol, manifest_path)
    require(len(new_records) == 1120 and len(reused_records) == 560,
            "Episode validation did not recover the full 1120 + 560 cohort")
    rebuilt = aggregate.summarize(protocol, rows, new_records, reused_records,
                                  source["worker_exitcodes"], job, node_receipts_complete=False)
    report = build_summary(protocol, rebuilt, performance, handle, source_qstat, False,
                           launch, receipts, extractor_job, recovery_receipt, progress, root)
    report['gripper_diagnostics'] = gripper_diagnostics({**new_records, **reused_records})
    return report


def run(args: argparse.Namespace) -> int:
    current_job, _ = guard()
    protocol, handle = read_json(args.protocol), read_json(args.evaluation_handle)
    root = args.root
    performance = read_json(root / "results" / "performance" / "summary.json")
    if args.queue_recovery_receipt is not None:
        recovery_receipt = read_json(args.queue_recovery_receipt)
        result = recover_evaluation(root, protocol, handle, performance, current_job, recovery_receipt)
        size = atomic_json(args.out, result)
        print(f"RESULT_EXTRACT status=complete recovery=queue_lock_timeout bytes={size}/{MAX_BYTES} "
              f"evaluation={handle.get('job_id')} export={current_job} episodes=1120+560",
              flush=True)
        return 0

    job = handle.get("job_id")
    require(isinstance(job, str) and PBS_ID.fullmatch(job), "Current evaluation handle lacks job ID")
    base = root / "artifacts" / job
    evaluation = read_json(root / "results" / "evaluation" / "summary.json")
    source_qstat, _ = qstat(job, full=True)
    launch = read_json(base / "evaluation_multinode_launch.json")
    receipts = [read_json(base / "nodes" / f"node_{rank}" / "node_receipt.json") for rank in (0, 1)]
    result = build_summary(protocol, evaluation, performance, handle, source_qstat,
                           (base / "PIPELINE_COMPLETE").is_file(), launch, receipts, current_job)
    size = atomic_json(args.out, result)
    print(f"RESULT_EXTRACT complete bytes={size}/{MAX_BYTES} evaluation={job} export={current_job}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=here / "protocol.json")
    parser.add_argument("--evaluation-handle", type=Path, default=here / "evaluation_handle.json")
    parser.add_argument("--root", type=Path, default=Path("/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--queue-recovery-receipt", type=Path)
    args = parser.parse_args(argv)
    args.out = args.out or args.root / "results" / "reporting" / "summary.json"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
