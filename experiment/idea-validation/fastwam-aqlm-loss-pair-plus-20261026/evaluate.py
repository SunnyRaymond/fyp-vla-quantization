"""Resumable paired 280+280 AQLM closed-loop evaluator.

The manifest, seed schedule, shard prefix recovery, and episode identity gates
follow the validated loss-pair evaluator.  No training, teacher query, or
external reference episode is performed here.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

from runtime import expected_install_metadata


PROTOCOL = "fastwam-aqlm-loss-pair-plus-v1"
COHORT_PROTOCOL = "fastwam-rotation-baselines-v1"
ARMS = ("reconstruction", "ours")
WORKER_ARMS = ("reconstruction", "reconstruction", "ours", "ours")
TIMING_FIELDS = (
    "denoising_latency_ms", "full_query_wall_ms", "peak_allocated_gpu_GB",
    "baseline_allocated_gpu_GB", "peak_reserved_gpu_GB",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(directory):
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def worker_arm(worker_id, arm=None):
    if type(worker_id) is not int or worker_id not in range(4):
        raise ValueError("worker_id must be 0, 1, 2, or 3")
    assigned = WORKER_ARMS[worker_id]
    if arm is not None and arm != assigned:
        raise ValueError(f"worker {worker_id} is assigned to {assigned}")
    return assigned


def worker_indices(worker_id, arm=None):
    worker_arm(worker_id, arm)
    return list(range(worker_id % 2, 280, 2))


def worker_coverage(worker_id, completed):
    return {"indices": worker_indices(worker_id), "completed_indices": list(completed)}


def environment_seed(index):
    return 800_000_000 + index * 1000


def sampler_seed(index, replan=0):
    return 900_000_000 + index * 1000 + replan


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _require_text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing bound source field: {name}")


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("protocol") != PROTOCOL:
        raise ValueError("AQLM closed-loop protocol identity differs")
    if protocol.get("arms") != list(ARMS) or protocol.get("soft_deadline_seconds") != 56700:
        raise ValueError("Frozen AQLM arms or soft deadline changed")

    cohort = protocol.get("cohort", {})
    expected_cohort = {
        "manifest_protocol": COHORT_PROTOCOL, "variants_per_arm": 280,
        "total_new_episodes": 560, "suites": 4, "dimensions": 7,
        "variants_per_suite": 70, "variants_per_dimension": 40,
        "variants_per_cell": 10, "state_id": 0, "env_seed_base": 800_000_000,
        "sampler_seed_base": 900_000_000, "seed_stride": 1000,
        "paired_seeds_across_arms": True, "sampling_changed": False,
    }
    if any(cohort.get(key) != value for key, value in expected_cohort.items()):
        raise ValueError("Frozen 280-episode paired cohort changed")

    quant = protocol.get("quantization", {})
    expected_quant = {
        "target_linears": 614, "weight_bits": 4, "activation_bits": 4,
        "execution": "decoded_bf16_dense_aqlm", "vq_enabled": True,
        "native_low_bit_kernel": False, "kv_cache_dtype": "bf16",
        "transform_implementation": "smooth05_h128_r0_aqlm_g8_m4_k256_v1",
        "smooth_alpha": 0.5, "hadamard_block": 128, "D": 1,
        "clipping_ratio": 1, "streams": ["action", "video"],
        "rotation_shape": [128, 128], "codebooks": 4,
        "codebook_size": 256, "in_group_size": 8,
        "decoded_operands": "bf16 dense",
    }
    if any(quant.get(key) != value for key, value in expected_quant.items()):
        raise ValueError("Frozen decoded-BF16 AQLM execution contract changed")

    inference = {
        "mode": "idm", "video_steps": 10, "action_steps": 10,
        "sigma_shift": 1.0, "action_horizon": 32, "execute_per_replan": 10,
        "action_ensembler": False, "settling_steps": 30,
        "max_steps_normal": 400, "max_steps_long": 700,
    }
    if protocol.get("inference") != inference:
        raise ValueError("Frozen inference contract changed")

    workers = protocol.get("worker_assignment", {})
    expected_mapping = {
        "0": {"arm": "reconstruction", "shard": 0},
        "1": {"arm": "reconstruction", "shard": 1},
        "2": {"arm": "ours", "shard": 0},
        "3": {"arm": "ours", "shard": 1},
    }
    if (workers.get("workers") != 4 or workers.get("mapping") != expected_mapping
            or workers.get("indices_per_worker") != "index % 2 == worker_id % 2"
            or workers.get("variants_per_worker") != 140):
        raise ValueError("Four-worker assignment or frozen shard changed")

    allocation = protocol.get("allocation", {})
    if (allocation.get("jobs") != 1 or allocation.get("gpus") != 4
            or allocation.get("queue") != "normal"
            or allocation.get("execution_queue") != "g3"
            or allocation.get("walltime") != "16:00:00"
            or allocation.get("episodes_per_job") != 560
            or allocation.get("gpu_monitor_seconds") != 15):
        raise ValueError("Frozen PBS allocation or telemetry interval changed")

    startup = protocol.get("startup_checks", {})
    if (startup.get("case_id") != 2000 or startup.get("seed_index") != 0
            or startup.get("sampler_seed") != 910_004_000
            or startup.get("expected_modules") != 614
            or startup.get("startup_queries_per_worker") != 2
            or startup.get("source_anchor_action_max_abs_tolerance") != 0.0
            or startup.get("source_anchor_future_max_abs_tolerance") != 0.0
            or startup.get("actual_infer_action_max_abs_tolerance") != 1e-6
            or startup.get("orthogonality_max_abs_tolerance") != 5e-5):
        raise ValueError("Frozen AQLM startup gates changed")

    references = protocol.get("references", {})
    if references.get("arms") != [] or references.get("reference_episode_slots") != 0:
        raise ValueError("This evaluation must not silently add external references")
    boundary = protocol.get("boundary", {})
    if (boundary.get("closed_loop_authorized") is not True
            or boundary.get("episodes_per_arm") != 280
            or boundary.get("total_episodes") != 560):
        raise ValueError("AQLM closed-loop boundary differs")

    sources = protocol.get("sources", {})
    for key in ("parent_root", "parent_protocol", "parent_pair_job",
                "parent_pair_artifacts", "parent_summary", "parent_protocol_file",
                "previous_episode_artifacts", "calibration_plan", "official_vendor"):
        _require_text(sources.get(key), f"sources.{key}")
    if not sources["parent_protocol"].startswith("fastwam-aqlm-loss-pair"):
        raise ValueError("Unexpected AQLM parent protocol")

    arm_sources = protocol.get("arm_sources", {})
    if set(arm_sources) != set(ARMS):
        raise ValueError("AQLM source bindings must cover exactly both arms")
    for arm in ARMS:
        source = arm_sources[arm]
        for key in ("source_root", "source_protocol", "source_job",
                    "source_protocol_file", "source_summary", "deployment",
                    "endpoint", "anchor_tensor", "source_bank"):
            _require_text(source.get(key), f"arm_sources.{arm}.{key}")
        if (source["source_job"] != sources["parent_pair_job"]
                or source["source_protocol"] != sources["parent_protocol"]
                or source.get("endpoint_arm") != arm
                or source.get("updates") != 8
                or source.get("training_exposures") != 80
                or source["source_protocol_file"] != sources["parent_protocol_file"]):
            raise ValueError(f"{arm} source binding is not the completed final8 pair")

    rotation = protocol.get("source_rotation", {})
    for key in ("root", "protocol", "job", "lane", "arm", "deployment"):
        _require_text(rotation.get(key), f"source_rotation.{key}")
    if rotation.get("updates") != 8 or rotation.get("training_exposures") != 80:
        raise ValueError("Frozen R0 source binding changed")
    return protocol


def load_manifest(path):
    doc = read_json(path)
    rows = doc.get("variants")
    if doc.get("protocol") != COHORT_PROTOCOL or not isinstance(rows, list) or len(rows) != 280:
        raise ValueError("Expected the original frozen 280-variant manifest")
    identities, cells = set(), collections.Counter()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("index")) is not int
                or row["index"] != index or row.get("state_id", 0) != 0):
            raise ValueError(f"Invalid frozen manifest row {index}")
        identity = str(row.get("variant_id", ""))
        if not identity or identity in identities or "task_id" not in row:
            raise ValueError("Duplicate/missing frozen variant identity")
        identities.add(identity)
        cells[row.get("suite"), row.get("dimension")] += 1
    if (len(cells) != 28 or set(cells.values()) != {10}
            or len({suite for suite, _ in cells}) != 4
            or len({dimension for _, dimension in cells}) != 7):
        raise ValueError("Expected frozen 4-suite x 7-dimension x 10-variant cohort")
    return doc, rows


def validate_install_metadata(metadata, protocol, arm):
    if arm not in ARMS or not isinstance(metadata, dict):
        raise ValueError("Missing AQLM install metadata")
    expected = expected_install_metadata(protocol, arm)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Install metadata differs from frozen {arm} AQLM deployment")
    orthogonality = metadata.get("orthogonality")
    movement = metadata.get("rotation_movement")
    tolerance = protocol["startup_checks"]["orthogonality_max_abs_tolerance"]
    if (not isinstance(orthogonality, dict) or set(orthogonality) != {"action", "video"}
            or any(not _finite_number(orthogonality[key]) or abs(orthogonality[key]) > tolerance
                   for key in orthogonality)):
        raise ValueError("Invalid AQLM rotation orthogonality evidence")
    if (not isinstance(movement, dict) or set(movement) != {"action", "video"}
            or any(not _finite_number(movement[key]) or movement[key] <= 0 for key in movement)):
        raise ValueError("Invalid AQLM rotation movement evidence")
    if set(metadata) != set(expected) | {"orthogonality", "rotation_movement"}:
        raise ValueError("Install metadata contains untracked AQLM fields")
    return metadata


def validate_setup_receipt(receipt, protocol, worker_id, arm, install_metadata=None):
    worker_arm(worker_id, arm)
    if not isinstance(receipt, dict):
        raise ValueError("Missing setup receipt")
    source = protocol["arm_sources"][arm]
    if (receipt.get("protocol") != PROTOCOL or receipt.get("worker_id") != worker_id
            or receipt.get("arm") != arm or receipt.get("passed") is not True
            or receipt.get("observed_modules") != 614
            or receipt.get("case_id") != 2000 or receipt.get("seed_index") != 0
            or receipt.get("sampler_seed") != 910_004_000
            or not isinstance(receipt.get("pbs_job_id"), str)
            or not receipt["pbs_job_id"]
            or receipt.get("source_job") != source["source_job"]
            or receipt.get("source_endpoint") != source["endpoint"]
            or receipt.get("source_anchor_passed") is not True
            or receipt.get("new_training_updates") != 0
            or receipt.get("startup_queries") != 2):
        raise ValueError(f"Worker {worker_id} setup/source-anchor gate failed")
    for key, tolerance in (
        ("source_action_max_abs", protocol["startup_checks"]["source_anchor_action_max_abs_tolerance"]),
        ("source_future_max_abs", protocol["startup_checks"]["source_anchor_future_max_abs_tolerance"]),
        ("actual_infer_action_max_abs", protocol["startup_checks"]["actual_infer_action_max_abs_tolerance"]),
    ):
        value = receipt.get(key)
        if not _finite_number(value) or value < 0 or value > tolerance:
            raise ValueError(f"Worker {worker_id} startup parity failed: {key}")
    if receipt.get("source_action_max_abs") != 0.0 or receipt.get("source_future_max_abs") != 0.0:
        raise ValueError(f"Worker {worker_id} final8 anchor must match exactly")
    if not isinstance(receipt.get("source_anchor_job"), str) or not receipt["source_anchor_job"]:
        raise ValueError(f"Worker {worker_id} anchor PBS identity is missing")
    installed = validate_install_metadata(receipt.get("install_metadata"), protocol, arm)
    if install_metadata is not None and installed != install_metadata:
        raise ValueError(f"Worker {worker_id} setup/install metadata disagree")
    return installed


def validate_record(record, rows, worker_id, protocol, install_metadata, arm):
    worker_arm(worker_id, arm)
    validate_install_metadata(install_metadata, protocol, arm)
    index = record.get("index")
    if type(index) is not int or index not in worker_indices(worker_id, arm):
        raise ValueError("Record is outside its frozen worker shard")
    row = rows[index]
    expected = {
        "protocol": PROTOCOL, "arm": arm, "worker_id": worker_id, "complete": True,
        "suite": row["suite"], "dimension": row["dimension"],
        "task_id": int(row["task_id"]), "state_id": 0,
        "env_seed": environment_seed(index), "sampler_seed": sampler_seed(index),
    }
    if (any(record.get(key) != value for key, value in expected.items())
            or str(record.get("variant_id")) != str(row["variant_id"])
            or type(record.get("success")) is not bool):
        raise ValueError(f"Frozen episode identity/seed differs: {arm}/{index}")
    if record.get("teacher_queries", 0) != 0 or record.get("new_training_updates", 0) != 0:
        raise ValueError("Closed-loop evaluation must not query a teacher or train")
    implementation = install_metadata["transform_implementation"]
    job_id = record.get("pbs_job_id")
    if record.get("transform_implementation") != implementation or not isinstance(job_id, str) or not job_id:
        raise ValueError("Episode transform or PBS identity differs from installation")
    evidence = record.get("vq_evidence")
    if (not isinstance(evidence, dict)
            or any(evidence.get(key) != value for key, value in install_metadata.items())
            or evidence.get("observed_module_count") != 614
            or evidence.get("native_int4_gemm_calls") != 0
            or evidence.get("kv_packed_prefills") != 0
            or evidence.get("kv_layer_reads") != 0
            or type(evidence.get("quantized_linear_calls")) is not int
            or evidence["quantized_linear_calls"] <= 0):
        raise ValueError(f"Decoded-BF16 AQLM episode evidence differs: {arm}/{index}")
    seconds = record.get("episode_seconds")
    if not _finite_number(seconds) or seconds < 0:
        raise ValueError("Invalid episode wall time")
    replans, metrics = record.get("replans"), record.get("metrics")
    if type(replans) is not int or replans <= 0 or not isinstance(metrics, list) or len(metrics) != replans:
        raise ValueError("Missing or inconsistent action-query metrics")
    for replan, metric in enumerate(metrics):
        query = metric.get("query_index_in_arm")
        if (type(query) is not int or query < 0
                or type(metric.get("first_query_in_arm")) is not bool
                or metric["first_query_in_arm"] != (query == 0)
                or metric.get("replan") != replan
                or metric.get("sampler_seed") != sampler_seed(index, replan)
                or metric.get("transform_implementation") != implementation
                or metric.get("pbs_job_id") != job_id
                or type(metric.get("first_query_in_run")) is not bool):
            raise ValueError("Invalid resumed implementation/run/query marker")
        for key in TIMING_FIELDS:
            if not _finite_number(metric.get(key)) or metric[key] < 0:
                raise ValueError(f"Invalid {key}")
    return index


def recover_records(path, rows, worker_id, protocol, install_metadata, arm):
    path = Path(path)
    if not path.exists():
        return {}
    if install_metadata is None:
        raise ValueError("Cannot resume episodes without the matching setup receipt")
    payload = path.read_bytes()
    chunks = payload.splitlines(keepends=True)
    if chunks and not chunks[-1].endswith(b"\n"):
        try:
            json.loads(chunks[-1])
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = b"".join(chunks[:-1])
        else:
            payload += b"\n"
        temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.recovered.tmp")
        with temp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        _fsync_dir(path.parent)

    found, queries, seen_jobs = {}, [], set()
    for line in payload.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index = validate_record(record, rows, worker_id, protocol, install_metadata, arm)
        if index in found:
            raise ValueError("Duplicate completed episode")
        found[index] = record
        for metric in record["metrics"]:
            queries.append(metric["query_index_in_arm"])
            job = metric["pbs_job_id"]
            if metric["first_query_in_run"] != (job not in seen_jobs):
                raise ValueError("Invalid first formal query marker within worker run")
            seen_jobs.add(job)
    indices = list(found)
    if indices != worker_indices(worker_id, arm)[:len(indices)]:
        raise ValueError("Saved episodes are not the frozen worker prefix")
    if queries != list(range(len(queries))):
        raise ValueError("Saved query sequence is not a consecutive worker prefix")
    return found


def append_episode(path, record):
    with Path(path).open("ab") as stream:
        stream.write((json.dumps(record, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def allocation_start_epoch():
    raw = os.environ.get("FW_JOB_START_EPOCH")
    if raw is None:
        raise RuntimeError("PBS runner must export allocation-wide FW_JOB_START_EPOCH")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("FW_JOB_START_EPOCH must be a Unix epoch in seconds") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("FW_JOB_START_EPOCH must be a positive finite Unix epoch")
    return value


def allocation_elapsed_seconds(start_epoch, now_epoch):
    if not _finite_number(start_epoch) or not _finite_number(now_epoch) or start_epoch <= 0 or now_epoch < start_epoch:
        raise ValueError("Invalid allocation start/now epoch")
    return float(now_epoch) - float(start_epoch)


def run_worker(args):
    import runner

    runner.guard()
    protocol = validate_protocol(read_json(args.protocol))
    arm, indices = worker_arm(args.worker_id), worker_indices(args.worker_id)
    _, rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / "protocol.json"
    if frozen.exists() and read_json(frozen) != protocol:
        raise ValueError("Cannot resume with a changed AQLM closed-loop protocol")
    save(frozen, protocol)

    setup_path, episode_path = args.out / "setup_receipt.json", args.out / "episodes.jsonl"
    old_setup = read_json(setup_path) if setup_path.is_file() else None
    old_metadata = (validate_setup_receipt(old_setup, protocol, args.worker_id, arm)
                    if old_setup is not None else None)
    completed = recover_records(episode_path, rows, args.worker_id, protocol, old_metadata, arm)
    if completed and old_metadata is None:
        raise ValueError("Saved episodes exist without their setup receipt")
    (args.out / "failure.json").unlink(missing_ok=True)

    start_epoch = allocation_start_epoch()
    results_root = args.out.parent.parent
    runtime, metadata, soft_stop = None, old_metadata, False

    def receipt(status, **extra):
        return {
            "protocol": PROTOCOL, "worker_id": args.worker_id, "arm": arm,
            "status": status, "complete": status == "complete",
            "episodes": len(completed), "completed": len(completed), "expected": 140,
            "soft_deadline_reached": soft_stop, **worker_coverage(args.worker_id, completed),
            "arm_episodes": {arm: len(completed)}, "install_metadata": metadata,
            "setup_receipt_file": "setup_receipt.json", **extra,
        }

    try:
        if len(completed) != len(indices):
            from runtime import R128Runtime

            save(args.out / "progress.json", receipt("running", stage="model_loading"))
            runtime = R128Runtime(args.out, protocol, args.worker_id)
            setup = runtime.prepare(results_root)
            metadata = validate_install_metadata(runtime.install_metadata, protocol, arm)
            validate_setup_receipt(setup, protocol, args.worker_id, arm, metadata)
            if old_metadata is not None and metadata != old_metadata:
                raise ValueError("Resumed worker loaded different AQLM metadata")
            save(setup_path, setup)
            if not hasattr(runtime, "arm_calls"):
                runtime.arm_calls = {}
            runtime.arm_calls[arm] = sum(len(record["metrics"]) for record in completed.values())
            runtime.calls = runtime.arm_calls[arm]
            save(args.out / "progress.json", receipt("running", stage="episode"))
            for index in indices:
                if index in completed:
                    continue
                if allocation_elapsed_seconds(start_epoch, time.time()) >= protocol["soft_deadline_seconds"]:
                    soft_stop = True
                    break
                save(args.out / "progress.json", receipt("running", stage="episode", index=index))
                record = runtime.episode(rows[index], index)
                if not isinstance(record, dict):
                    raise TypeError("runtime.episode must return an episode dictionary")
                record.setdefault("worker_id", args.worker_id)
                validate_record(record, rows, args.worker_id, protocol, metadata, arm)
                append_episode(episode_path, record)
                completed[index] = record
                save(args.out / "progress.json", receipt("running", stage="episode_saved", index=index))
                print(f"SLOT_COMPLETE worker={args.worker_id} arm={arm} index={index} "
                      f"success={record['success']}", flush=True)
        else:
            if old_setup is None:
                raise ValueError("Completed worker is missing its setup receipt")
            metadata = validate_setup_receipt(old_setup, protocol, args.worker_id, arm)

        status = "checkpointed" if soft_stop else "complete"
        if not soft_stop and set(completed) != set(indices):
            raise RuntimeError("Worker completed coverage differs from its frozen 140-slot shard")
        result = receipt(status)
        save(args.out / "summary.json", result)
        save(args.out / "progress.json", result)
        print(f"WORKER_{status.upper()} worker={args.worker_id} episodes={len(completed)}", flush=True)
    except Exception as exc:
        save(args.out / "failure.json", {
            "protocol": PROTOCOL, "worker_id": args.worker_id, "arm": arm,
            "type": type(exc).__name__, "message": str(exc), "completed": len(completed),
        })
        raise
    finally:
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, choices=range(4), required=True)
    run_worker(parser.parse_args())


if __name__ == "__main__":
    main()
