"""Run one parity-sharded worker for the native Smooth+Scalar W4A4 arm."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

PROTOCOL = "fastwam-smooth-scalar-plus-v1"
COHORT_PROTOCOL = "fastwam-rotation-baselines-v1"
ARM = "smooth05_hadamard_scalar_w4a4"
EXPECTED_VARIANTS = 280
EXPECTED_MODULE_COUNT = 614
EXPECTED_EVIDENCE = {
    "source": "scalar_bank",
    "source_recipe": "smooth05_hadamard_scalar_a4",
    "smooth_alpha": 0.5,
    "transform_seed": 20261006,
    "transform_reused": True,
    "weight_codes_reused": True,
}
TIMING_FIELDS = (
    "denoising_latency_ms",
    "full_query_wall_ms",
    "peak_allocated_gpu_GB",
    "baseline_allocated_gpu_GB",
    "peak_reserved_gpu_GB",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def worker_indices(worker_id: int) -> list[int]:
    if type(worker_id) is not int or worker_id not in (0, 1):
        raise ValueError("worker_id must be 0 or 1")
    return list(range(worker_id, EXPECTED_VARIANTS, 2))


def validate_manifest(path: Path, worker_id: int):
    doc = read_json(path)
    rows = doc.get("variants")
    if doc.get("protocol") != COHORT_PROTOCOL or not isinstance(rows, list) or len(rows) != EXPECTED_VARIANTS:
        raise ValueError("Expected the frozen 280-variant rotation-baselines manifest")
    ids, cells = set(), {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index:
            raise ValueError(f"Invalid global variant index {index}")
        variant_id = str(row.get("variant_id", ""))
        if not variant_id or variant_id in ids:
            raise ValueError(f"Missing or duplicate variant_id at index {index}")
        ids.add(variant_id)
        cells[(row.get("suite"), row.get("dimension"))] = cells.get((row.get("suite"), row.get("dimension")), 0) + 1
        if "task_id" not in row:
            raise ValueError(f"Missing task_id at index {index}")
    if len({suite for suite, _ in cells}) != 4 or len({dimension for _, dimension in cells}) != 7:
        raise ValueError("Expected the original four-suite by seven-dimension cohort")
    if len(cells) != 28 or set(cells.values()) != {10}:
        raise ValueError("Expected 10 variants in each suite-by-dimension cell")
    return doc, rows, worker_indices(worker_id)


def validate_install_metadata(metadata, bank_path: Path) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("scalar_native.install must return metadata")
    module_count = metadata.get("module_count")
    if type(module_count) is not int or module_count != EXPECTED_MODULE_COUNT:
        raise ValueError(f"Expected {EXPECTED_MODULE_COUNT} installed target modules")
    if metadata.get("arm") != ARM:
        raise ValueError("Installed Scalar bank arm does not match this experiment")
    if (metadata.get("source_recipe") != "smooth05_hadamard_scalar_a4"
            or metadata.get("smooth_alpha") != 0.5
            or metadata.get("transform_seed") != 20261006):
        raise ValueError("Installed Scalar bank recipe does not match the frozen configuration")
    if metadata.get("transforms_reused") is not True or metadata.get("weight_codes_reused") is not True:
        raise ValueError("Install must reuse the calibrated transforms and packed weight codes")
    source_bank = metadata.get("source_bank")
    if not isinstance(source_bank, (str, os.PathLike)):
        raise ValueError("Install metadata is missing source_bank")
    expected = Path(bank_path).resolve()
    actual = Path(source_bank).resolve()
    if actual != expected:
        raise ValueError(f"Installed bank path mismatch: {actual} != {expected}")
    return {
        **metadata,
        "arm": ARM,
        "module_count": module_count,
        "source_bank": str(actual),
        "source_recipe": "smooth05_hadamard_scalar_a4",
        "smooth_alpha": 0.5,
        "transform_seed": 20261006,
        "transforms_reused": True,
        "weight_codes_reused": True,
    }


def _finite_nonnegative(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _replace_bytes(path: Path, payload: bytes):
    temp = path.with_name(path.name + ".resume.tmp")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def validate_native_evidence(evidence, source_bank: str):
    if not isinstance(evidence, dict):
        raise ValueError("Episode is missing native quantization evidence")
    checks = {
        "real_quant": True,
        "activation_bits": 4,
        "native_int4_tensorcore": True,
        "bf16_target_weight_absent": True,
        **EXPECTED_EVIDENCE,
        "source_bank": source_bank,
    }
    for key, expected in checks.items():
        if evidence.get(key) != expected:
            raise ValueError(f"Native evidence mismatch for {key}: {evidence.get(key)!r}")
    for key in ("integer_gemm_calls", "native_int4_gemm_calls", "packed_weight_tensor_bytes"):
        value = evidence.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Native evidence requires positive {key}")
    for key in ("kv_packed_prefills", "kv_layer_reads"):
        if evidence.get(key) != 0:
            raise ValueError(f"Scalar W4A4 must keep KV in BF16; expected {key}=0")


def validate_episode_record(record, rows, indices, worker_id: int, source_bank: str):
    if not isinstance(record, dict):
        raise ValueError("Episode record must be an object")
    index = record.get("index")
    if type(index) is not int or index not in indices:
        raise ValueError(f"Unexpected episode index: {index!r}")
    row = rows[index]
    if (record.get("protocol") != PROTOCOL or record.get("complete") is not True
            or record.get("arm") != ARM or record.get("worker_id") != worker_id
            or not isinstance(record.get("success"), bool)
            or str(record.get("variant_id")) != str(row["variant_id"])
            or record.get("suite") != row.get("suite")
            or record.get("dimension") != row.get("dimension")
            or record.get("task_id") != int(row["task_id"])
            or record.get("state_id") != 0
            or record.get("env_seed") != 800_000_000 + index * 1000
            or record.get("sampler_seed") != 900_000_000 + index * 1000):
        raise ValueError(f"Episode identity or seed mismatch at index {index}")
    _finite_nonnegative(record.get("episode_seconds"), "episode_seconds")
    validate_native_evidence(record.get("real_quant_evidence"), source_bank)
    metrics = record.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"Episode {index} has no query metrics")
    for metric in metrics:
        query_index = metric.get("query_index_in_arm")
        if (type(query_index) is not int or query_index < 0
                or type(metric.get("first_query_in_arm")) is not bool
                or metric["first_query_in_arm"] != (query_index == 0)):
            raise ValueError(f"Invalid first/subsequent query marker at episode {index}")
        for key in TIMING_FIELDS:
            _finite_nonnegative(metric.get(key), key)
    return index


def recover_records(path: Path, rows, indices, worker_id: int, source_bank: str):
    if not path.exists():
        return {}
    payload = path.read_bytes()
    chunks = payload.splitlines(keepends=True)
    if chunks and not chunks[-1].endswith(b"\n"):
        try:
            json.loads(chunks[-1])
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = b"".join(chunks[:-1])
            _replace_bytes(path, payload)
            print("RESUME_DROPPED_TORN_FINAL_LINE", flush=True)
        else:
            payload += b"\n"
            _replace_bytes(path, payload)
    completed = {}
    query_indices = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed resume record at {path}:{line_number}") from exc
            index = validate_episode_record(record, rows, indices, worker_id, source_bank)
            if index in completed:
                raise ValueError(f"Duplicate completed slot for index {index}")
            completed[index] = record
            query_indices.extend(metric["query_index_in_arm"] for metric in record["metrics"])
    if len(query_indices) != len(set(query_indices)):
        raise ValueError("Duplicate query_index_in_arm in resume records")
    if sorted(query_indices) != list(range(len(query_indices))):
        raise ValueError("Resume query indices are not a consecutive prefix")
    return completed


def append_episode(path: Path, record):
    encoded = (json.dumps(record, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _record_for_episode(runtime, row, index, worker_id: int, seed_for, install_metadata):
    record = runtime.episode(row, index)
    evidence = dict(record.get("real_quant_evidence") or {})
    evidence.update({
        **EXPECTED_EVIDENCE,
        "source_bank": install_metadata["source_bank"],
    })
    record.update({
        "protocol": PROTOCOL,
        "arm": ARM,
        "worker_id": worker_id,
        "sampler_seed": seed_for(index),
        "real_quant_evidence": evidence,
    })
    return record


def run_worker(worker_id: int, manifest_path: Path, bank_path: Path, out: Path):
    import runner

    runner.guard()
    runner.PROTOCOL = PROTOCOL
    from scalar_native import close as close_scalar, install, validate_runtime

    if not bank_path.is_dir():
        raise FileNotFoundError(f"Scalar bank directory is unavailable: {bank_path}")
    manifest, rows, indices = validate_manifest(manifest_path, worker_id)
    source_bank = str(bank_path.resolve())
    out.mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes.jsonl"
    completed = recover_records(episodes_path, rows, indices, worker_id, source_bank)
    expected = len(indices)
    runtime = None
    install_metadata = None
    try:
        runtime = runner.Runtime(out)
        native_metadata = install(runtime.q, bank_path)
        install_metadata = validate_install_metadata(native_metadata, bank_path)
        runtime.current_arm = ARM
        runtime.q.enable("w4a4")
        setup_receipt = validate_runtime(runtime, native_metadata)
        if not isinstance(setup_receipt, dict):
            raise ValueError("scalar_native.validate_runtime must return a JSON receipt")
        write_json(out / "setup_receipt.json", setup_receipt)
        resumed_queries = sum(len(record["metrics"]) for record in completed.values())
        runtime.arm_calls[ARM] = resumed_queries
        runtime.calls = resumed_queries
        write_json(out / "progress.json", {
            "protocol": PROTOCOL, "worker_id": worker_id, "status": "running",
            "stage": "setup_validation", "completed": len(completed), "expected": expected,
            "install_metadata": install_metadata, "setup_receipt_file": "setup_receipt.json",
        })
        pending = [index for index in indices if index not in completed]
        for index in pending:
            row = rows[index]
            write_json(out / "progress.json", {
                "protocol": PROTOCOL, "worker_id": worker_id, "status": "running",
                "arm": ARM, "variant_id": str(row["variant_id"]), "index": index,
                "completed": len(completed), "expected": expected,
                "install_metadata": install_metadata, "setup_receipt_file": "setup_receipt.json",
            })
            record = _record_for_episode(runtime, row, index, worker_id, runner.seed_for, install_metadata)
            record["env_seed"] = runner.seed_for(index, environment=True)
            validate_episode_record(record, rows, indices, worker_id, source_bank)
            append_episode(episodes_path, record)
            completed[index] = record
            write_json(out / "progress.json", {
                "protocol": PROTOCOL, "worker_id": worker_id, "status": "running",
                "arm": ARM, "variant_id": str(row["variant_id"]), "index": index,
                "completed": len(completed), "expected": expected,
                "install_metadata": install_metadata, "setup_receipt_file": "setup_receipt.json",
            })
            print(f"SLOT_COMPLETE worker={worker_id} index={index} variant={row['variant_id']} success={record['success']}", flush=True)

        if set(completed) != set(indices):
            raise RuntimeError(f"Worker completed {len(completed)} of {expected} episodes")
        summary = {
            "protocol": PROTOCOL, "complete": True, "worker_id": worker_id,
            "episodes": len(completed), "expected": expected,
            "indices": indices, "arm_episodes": {ARM: len(completed)},
            "install_metadata": install_metadata,
            "setup_receipt_file": "setup_receipt.json",
            "quantization_runtime": runtime.q.summary(),
        }
        write_json(out / "summary.json", summary)
        write_json(out / "progress.json", {
            "protocol": PROTOCOL, "worker_id": worker_id, "status": "complete",
            "completed": expected, "expected": expected, "install_metadata": install_metadata,
            "setup_receipt_file": "setup_receipt.json",
        })
        print(f"WORKER_COMPLETE worker={worker_id} episodes={expected}", flush=True)
    finally:
        if runtime is not None:
            if runtime.env is not None:
                runtime.env.close()
                runtime.env = None
            close_scalar(runtime.q)
            runtime.q.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="This worker's output directory")
    args = parser.parse_args()
    run_worker(args.worker_id, args.manifest, args.bank, args.out)


if __name__ == "__main__":
    main()
