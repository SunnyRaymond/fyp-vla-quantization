"""Validate paired scalar episodes against the frozen three-arm cohort."""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

PROTOCOL = "fastwam-smooth-scalar-plus-v1"
COHORT_PROTOCOL = "fastwam-rotation-baselines-v1"
ARM = "smooth05_hadamard_scalar_w4a4"
REFERENCE_ARMS = ("bf16", "quarot_adapted_w4a4", "spinquant_adapted_w4a4")
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
MEMORY_FIELDS = (
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


def load_manifest(path: Path):
    doc = read_json(path)
    rows = doc.get("variants")
    if doc.get("protocol") != COHORT_PROTOCOL or not isinstance(rows, list) or len(rows) != EXPECTED_VARIANTS:
        raise ValueError("Expected the frozen 280-variant rotation-baselines manifest")
    ids, cells = set(), collections.Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index:
            raise ValueError(f"Invalid global variant index {index}")
        variant_id = str(row.get("variant_id", ""))
        if not variant_id or variant_id in ids or "task_id" not in row:
            raise ValueError(f"Missing or duplicate variant identity at index {index}")
        ids.add(variant_id)
        cells[(row.get("suite"), row.get("dimension"))] += 1
    if len({suite for suite, _ in cells}) != 4 or len({dimension for _, dimension in cells}) != 7:
        raise ValueError("Expected the original four-suite by seven-dimension cohort")
    if len(cells) != 28 or set(cells.values()) != {10}:
        raise ValueError("Expected 10 variants in each suite-by-dimension cell")
    shards = doc.get("shards")
    if not isinstance(shards, list) or len(shards) != 4:
        raise ValueError("Original cohort must retain its four frozen shards")
    shard_map = {}
    for shard in shards:
        worker, indices = shard.get("worker_id"), shard.get("indices")
        if type(worker) is not int or worker not in range(4) or worker in shard_map:
            raise ValueError(f"Invalid original worker shard: {worker!r}")
        if (not isinstance(indices, list) or len(indices) != 70
                or any(type(i) is not int or not 0 <= i < EXPECTED_VARIANTS for i in indices)
                or len(set(indices)) != 70):
            raise ValueError(f"Invalid original worker shard {worker}")
        shard_map[worker] = set(indices)
    if set(shard_map) != {0, 1, 2, 3} or set.union(*shard_map.values()) != set(range(EXPECTED_VARIANTS)):
        raise ValueError("Original four worker shards must partition the 280 variants")
    return doc, rows, shard_map


def _positive_int(value, name: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Native evidence requires positive {name}")


def validate_scalar_metadata(metadata):
    if (not isinstance(metadata, dict) or metadata.get("arm") != ARM
            or type(metadata.get("module_count")) is not int
            or metadata.get("module_count") != EXPECTED_MODULE_COUNT
            or not isinstance(metadata.get("source_bank"), str)
            or metadata.get("source_recipe") != "smooth05_hadamard_scalar_a4"
            or metadata.get("smooth_alpha") != 0.5
            or metadata.get("transform_seed") != 20261006
            or metadata.get("transforms_reused") is not True
            or metadata.get("weight_codes_reused") is not True):
        raise ValueError("Worker has invalid Scalar bank install metadata")
    return metadata


def _validate_identity(record, row, index: int, protocol: str, arm: str, worker_id=None):
    if (record.get("protocol") != protocol or record.get("complete") is not True
            or record.get("arm") != arm or not isinstance(record.get("success"), bool)
            or str(record.get("variant_id")) != str(row["variant_id"])
            or record.get("suite") != row.get("suite")
            or record.get("dimension") != row.get("dimension")
            or record.get("task_id") != int(row["task_id"])
            or record.get("state_id") != 0
            or record.get("env_seed") != 800_000_000 + index * 1000
            or record.get("sampler_seed") != 900_000_000 + index * 1000):
        raise ValueError(f"Episode identity or seed mismatch at index {index} for {arm}")
    if worker_id is not None and record.get("worker_id", worker_id) != worker_id:
        raise ValueError(f"Episode worker mismatch at index {index} for {arm}")


def _validate_scalar_record(record, row, index: int, worker_id: int, source_bank: str):
    if not isinstance(record, dict):
        raise ValueError(f"Episode record {index} must be an object")
    _validate_identity(record, row, index, PROTOCOL, ARM, worker_id)
    episode_seconds = float(record.get("episode_seconds"))
    if not math.isfinite(episode_seconds) or episode_seconds < 0:
        raise ValueError(f"Invalid episode wall time at index {index}")
    evidence = record.get("real_quant_evidence")
    required = {
        "real_quant": True,
        "activation_bits": 4,
        "native_int4_tensorcore": True,
        "bf16_target_weight_absent": True,
        **EXPECTED_EVIDENCE,
        "source_bank": source_bank,
    }
    if not isinstance(evidence, dict) or any(evidence.get(key) != value for key, value in required.items()):
        raise ValueError(f"Native Scalar evidence is incomplete at index {index}")
    for key in ("integer_gemm_calls", "native_int4_gemm_calls", "packed_weight_tensor_bytes"):
        _positive_int(evidence.get(key), key)
    if evidence.get("kv_packed_prefills") != 0 or evidence.get("kv_layer_reads") != 0:
        raise ValueError(f"Scalar W4A4 episode {index} unexpectedly used KV4")
    metrics = record.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"Episode {index} has no query metrics")
    for metric in metrics:
        query_index = metric.get("query_index_in_arm")
        if (type(query_index) is not int or query_index < 0
                or type(metric.get("first_query_in_arm")) is not bool
                or metric["first_query_in_arm"] != (query_index == 0)):
            raise ValueError(f"Invalid query marker at episode {index}")
        for key in TIMING_FIELDS:
            value = float(metric.get(key))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} at episode {index}")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed record at {path}:{line_number}") from exc


def load_scalar_episodes(results_root: Path, rows):
    expected = {(ARM, index) for index in range(EXPECTED_VARIANTS)}
    found, worker_metadata, setup_receipts = {}, {}, {}
    query_indices_by_worker = {0: [], 1: []}
    for worker in (0, 1):
        indices = list(range(worker, EXPECTED_VARIANTS, 2))
        out = results_root / f"worker_{worker}"
        progress = read_json(out / "progress.json")
        summary = read_json(out / "summary.json")
        count = len(indices)
        if (progress.get("protocol") != PROTOCOL or progress.get("worker_id") != worker
                or progress.get("status") != "complete" or progress.get("completed") != count
                or progress.get("expected") != count):
            raise ValueError(f"Scalar worker {worker} progress is incomplete")
        metadata = validate_scalar_metadata(summary.get("install_metadata"))
        if (summary.get("protocol") != PROTOCOL or summary.get("complete") is not True
                or summary.get("worker_id") != worker or summary.get("episodes") != count
                or summary.get("expected") != count or summary.get("arm_episodes") != {ARM: count}
                or summary.get("indices") != indices
                or summary.get("setup_receipt_file") != "setup_receipt.json"
                or progress.get("setup_receipt_file") != "setup_receipt.json"):
            raise ValueError(f"Scalar worker {worker} summary is incomplete")
        if progress.get("install_metadata") != metadata:
            raise ValueError(f"Scalar worker {worker} progress/install metadata mismatch")
        worker_metadata[worker] = metadata
        setup_receipt = read_json(out / "setup_receipt.json")
        if not isinstance(setup_receipt, dict):
            raise ValueError(f"Scalar worker {worker} setup receipt is invalid")
        setup_receipts[worker] = setup_receipt
        for line_number, record in _read_jsonl(out / "episodes.jsonl"):
            if not isinstance(record, dict):
                raise ValueError(f"Invalid scalar record at worker {worker}, line {line_number}")
            index = record.get("index")
            if type(index) is not int or index not in indices:
                raise ValueError(f"Unexpected scalar slot at worker {worker}, line {line_number}")
            _validate_scalar_record(record, rows[index], index, worker, metadata["source_bank"])
            slot = (ARM, index)
            if slot in found:
                raise ValueError(f"Duplicate scalar episode slot: {slot}")
            found[slot] = record
            query_indices_by_worker[worker].extend(
                metric["query_index_in_arm"] for metric in record["metrics"]
            )
    if found.keys() != expected:
        raise ValueError(f"Expected 280 unique Scalar episodes, found {len(found)}")
    for worker, query_indices in query_indices_by_worker.items():
        if sorted(query_indices) != list(range(len(query_indices))):
            raise ValueError(f"Worker {worker} query indices are not a unique consecutive sequence")
        if sum(index == 0 for index in query_indices) != 1:
            raise ValueError(f"Worker {worker} must have exactly one first query")
    if worker_metadata[0] != worker_metadata[1]:
        raise ValueError("Scalar bank install metadata differs between workers")
    return (
        {index: found[(ARM, index)] for index in range(EXPECTED_VARIANTS)},
        worker_metadata[0],
        setup_receipts,
    )


def load_reference_episodes(results_root: Path, rows, shards):
    expected = {(arm, index) for arm in REFERENCE_ARMS for index in range(EXPECTED_VARIANTS)}
    found = {}
    for worker, indices in sorted(shards.items()):
        out = results_root / f"worker_{worker}"
        progress = read_json(out / "progress.json")
        summary = read_json(out / "summary.json")
        count = len(indices) * len(REFERENCE_ARMS)
        per_arm = {arm: len(indices) for arm in REFERENCE_ARMS}
        if (progress.get("protocol") != COHORT_PROTOCOL or progress.get("worker_id") != worker
                or progress.get("status") != "complete" or progress.get("completed") != count
                or progress.get("expected") != count):
            raise ValueError(f"Reference worker {worker} progress is incomplete")
        if (summary.get("protocol") != COHORT_PROTOCOL or summary.get("complete") is not True
                or summary.get("worker_id") != worker or summary.get("episodes") != count
                or summary.get("expected") != count or summary.get("arm_episodes") != per_arm):
            raise ValueError(f"Reference worker {worker} summary is incomplete")
        for line_number, record in _read_jsonl(out / "episodes.jsonl"):
            if not isinstance(record, dict):
                raise ValueError(f"Invalid reference record at worker {worker}, line {line_number}")
            arm, index = record.get("arm"), record.get("index")
            if arm not in REFERENCE_ARMS or type(index) is not int or index not in indices:
                raise ValueError(f"Unexpected reference slot at worker {worker}, line {line_number}")
            _validate_identity(record, rows[index], index, COHORT_PROTOCOL, arm, worker)
            if arm != "bf16":
                evidence = record.get("real_quant_evidence", {})
                required = {
                    "rotation_bank": arm,
                    "activation_bits": 4,
                    "native_int4_tensorcore": True,
                    "bf16_target_weight_absent": True,
                }
                if not isinstance(evidence, dict) or any(evidence.get(key) != value for key, value in required.items()):
                    raise ValueError(f"Reference native evidence is incomplete at {worker}/{arm}/{index}")
                for key in ("integer_gemm_calls", "native_int4_gemm_calls", "packed_weight_tensor_bytes"):
                    _positive_int(evidence.get(key), key)
            slot = (arm, index)
            if slot in found:
                raise ValueError(f"Duplicate reference episode slot: {slot}")
            found[slot] = record
    if found.keys() != expected:
        raise ValueError(f"Expected 840 unique reference episodes, found {len(found)}")
    return found


def _rate(records, indices):
    successes = sum(bool(records[index]["success"]) for index in indices)
    return {"successes": successes, "episodes": len(indices), "success_rate": successes / len(indices)}


def _paired(candidate, reference, arm, indices):
    wins = losses = ties = 0
    for index in indices:
        new_success = candidate[index]["success"]
        ref_success = reference[(arm, index)]["success"]
        if new_success and not ref_success:
            wins += 1
        elif ref_success and not new_success:
            losses += 1
        else:
            ties += 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "episodes": len(indices),
        "success_rate_difference": (wins - losses) / len(indices),
    }


def _bucket_summary(candidate, reference, indices):
    return {
        "arms": {ARM: _rate(candidate, indices)},
        "reference_arms": {arm: _rate({i: reference[(arm, i)] for i in indices}, indices)
                           for arm in REFERENCE_ARMS},
        "paired_vs_reference_arms": {arm: _paired(candidate, reference, arm, indices)
                                     for arm in REFERENCE_ARMS},
    }


def _stats(values, unit):
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0, f"mean_{unit}": None, f"median_{unit}": None,
                f"p95_{unit}": None, f"max_{unit}": None}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Summary values must be finite and non-negative")
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "n": len(values),
        f"mean_{unit}": statistics.mean(values),
        f"median_{unit}": statistics.median(values),
        f"p95_{unit}": values[p95_index],
        f"max_{unit}": values[-1],
    }


def summarize_timing(records):
    query_samples = {
        name: {key: [] for key in TIMING_FIELDS}
        for name in ("first_query", "subsequent_queries")
    }
    episode_samples, first_episode_samples, later_episode_samples = [], [], []
    for index in range(EXPECTED_VARIANTS):
        record = records[index]
        episode_ms = float(record["episode_seconds"]) * 1000.0
        episode_samples.append(episode_ms)
        had_first = False
        for metric in record["metrics"]:
            kind = "first_query" if metric["first_query_in_arm"] else "subsequent_queries"
            for key in TIMING_FIELDS:
                query_samples[kind][key].append(float(metric[key]))
            had_first |= metric["first_query_in_arm"]
        (first_episode_samples if had_first else later_episode_samples).append(episode_ms)
    if sum(map(len, (query_samples["first_query"]["denoising_latency_ms"],))) != 2:
        raise ValueError("Expected exactly two first queries, one for each worker")
    if len(first_episode_samples) != 2 or len(later_episode_samples) != 278:
        raise ValueError("Expected two first-query episodes and 278 subsequent-query episodes")
    query_timing = {}
    for kind, fields in query_samples.items():
        queries = len(fields["denoising_latency_ms"])
        query_timing[kind] = {
            "queries": queries,
            "continuous_denoising_gpu_interval_ms": _stats(fields["denoising_latency_ms"], "ms"),
            "full_action_query_wall_ms": _stats(fields["full_query_wall_ms"], "ms"),
            "memory": {key: _stats(fields[key], "GB") for key in MEMORY_FIELDS},
        }
    return {
        "query_timing": query_timing,
        "episode_wall_including_environment_ms": _stats(episode_samples, "ms"),
        "episode_wall_for_first_query_episodes_ms": _stats(first_episode_samples, "ms"),
        "episode_wall_for_subsequent_query_episodes_ms": _stats(later_episode_samples, "ms"),
    }


def summarize(rows, candidate, reference, install_metadata, setup_receipts):
    all_indices = list(range(EXPECTED_VARIANTS))
    groups = {
        "suites": {suite: [i for i, row in enumerate(rows) if row["suite"] == suite]
                   for suite in sorted({row["suite"] for row in rows})},
        "dimensions": {dimension: [i for i, row in enumerate(rows) if row["dimension"] == dimension]
                       for dimension in sorted({row["dimension"] for row in rows})},
        "cells": {
            f"{suite}::{dimension}": [i for i, row in enumerate(rows)
                                       if row["suite"] == suite and row["dimension"] == dimension]
            for suite in sorted({row["suite"] for row in rows})
            for dimension in sorted({row["dimension"] for row in rows})
        },
    }
    output = {
        "protocol": PROTOCOL,
        "complete": True,
        "manifest_variants": EXPECTED_VARIANTS,
        "episode_slots": len(candidate),
        "reference_episode_slots": len(reference),
        "arm": ARM,
        "install_metadata": install_metadata,
        "setup_receipts": {str(worker): receipt for worker, receipt in sorted(setup_receipts.items())},
        "timing": {ARM: summarize_timing(candidate)},
        "overall": _bucket_summary(candidate, reference, all_indices),
    }
    for level, buckets in groups.items():
        output[level] = {name: _bucket_summary(candidate, reference, indices)
                         for name, indices in buckets.items()}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="New scalar root/results directory")
    parser.add_argument("--reference-results", type=Path, required=True, help="Original rotation-baselines results")
    parser.add_argument("--out", type=Path, required=True, help="Output summary JSON file")
    args = parser.parse_args()

    import runner
    runner.guard()

    manifest, rows, shards = load_manifest(args.manifest)
    reference_manifest, _reference_rows, _reference_shards = load_manifest(args.reference_results / "manifest.json")
    if manifest != reference_manifest:
        raise ValueError("New run must use the exact original frozen manifest contents")
    candidate, metadata, setup_receipts = load_scalar_episodes(args.results, rows)
    reference = load_reference_episodes(args.reference_results, rows, shards)
    summary = summarize(rows, candidate, reference, metadata, setup_receipts)
    write_json(args.out, summary)
    print(f"AGGREGATION_COMPLETE scalar_slots=280 reference_slots=840 output={args.out}", flush=True)


if __name__ == "__main__":
    main()
