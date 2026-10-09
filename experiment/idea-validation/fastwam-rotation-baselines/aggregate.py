"""Validate and summarize the paired three-arm LIBERO-Plus cohort."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics

from evaluate import validate_prepared

PROTOCOL = "fastwam-rotation-baselines-v1"
ARMS = ("bf16", "quarot_adapted_w4a4", "spinquant_adapted_w4a4")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_manifest(path: Path):
    doc = read_json(path)
    if doc.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected manifest protocol: {doc.get('protocol')}")
    rows = doc.get("variants")
    if not isinstance(rows, list) or len(rows) != 280:
        raise ValueError("Expected exactly 280 frozen variants")
    ids, cells = set(), collections.Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("index")) is not int or row.get("index") != index:
            raise ValueError(f"Variant index is not global/contiguous at row {index}")
        variant_id = str(row.get("variant_id", ""))
        if not variant_id or variant_id in ids:
            raise ValueError(f"Missing or duplicate variant_id at index {index}")
        ids.add(variant_id)
        cells[(row.get("suite"), row.get("dimension"))] += 1
    if len({suite for suite, _ in cells}) != 4 or len({dim for _, dim in cells}) != 7:
        raise ValueError("Expected four suites and seven perturbation dimensions")
    if len(cells) != 28 or set(cells.values()) != {10}:
        raise ValueError("Expected 10 variants in each suite-by-dimension cell")

    shards = doc.get("shards")
    if not isinstance(shards, list) or len(shards) != 4:
        raise ValueError("Expected exactly four worker shards")
    assigned = {}
    for shard in shards:
        worker = shard.get("worker_id")
        indices = shard.get("indices")
        if isinstance(worker, bool) or not isinstance(worker, int) or worker in assigned:
            raise ValueError(f"Invalid or duplicate worker_id: {worker}")
        if not isinstance(indices, list) or len(indices) != 70:
            raise ValueError(f"Worker {worker} must own 70 variants")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(rows) for i in indices):
            raise ValueError(f"Worker {worker} has an invalid variant index")
        if len(set(indices)) != len(indices):
            raise ValueError(f"Worker {worker} has duplicate indices")
        assigned[worker] = set(indices)
    if set(assigned) != {0, 1, 2, 3} or set.union(*assigned.values()) != set(range(len(rows))):
        raise ValueError("Worker shards must cover all 280 variants")
    if sum(map(len, assigned.values())) != len(rows):
        raise ValueError("Worker shards overlap")
    return doc, rows, assigned


def load_episodes(results_root: Path, rows, assigned):
    try:
        validate_prepared(results_root)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    expected = {(arm, index) for arm in ARMS for index in range(len(rows))}
    found = {}
    for worker, indices in sorted(assigned.items()):
        out = results_root / f"worker_{worker}"
        progress = read_json(out / "progress.json")
        expected_worker = len(indices) * len(ARMS)
        if (progress.get("protocol") != PROTOCOL or progress.get("worker_id") != worker
                or progress.get("status") != "complete" or progress.get("completed") != expected_worker
                or progress.get("expected") != expected_worker):
            raise ValueError(f"Worker {worker} progress is not complete")
        summary = read_json(out / "summary.json")
        if (summary.get("protocol") != PROTOCOL or summary.get("complete") is not True
                or summary.get("worker_id") != worker or summary.get("episodes") != expected_worker
                or summary.get("expected") != expected_worker
                or summary.get("arm_episodes") != {arm: len(indices) for arm in ARMS}):
            raise ValueError(f"Worker {worker} summary is incomplete")
        path = out / "episodes.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed {path}:{line_number}") from exc
                arm, index = record.get("arm"), record.get("index")
                slot = (arm, index)
                if arm not in ARMS or type(index) is not int or index not in indices or slot not in expected:
                    raise ValueError(f"Unexpected slot in {path}:{line_number}: {slot}")
                row = rows[index]
                if (record.get("protocol") != PROTOCOL or record.get("complete") is not True
                        or not isinstance(record.get("success"), bool)
                        or str(record.get("variant_id")) != str(row["variant_id"])
                        or record.get("suite") != row.get("suite")
                        or record.get("dimension") != row.get("dimension")
                        or record.get("task_id") != int(row["task_id"])
                        or record.get("state_id") != 0
                        or record.get("env_seed") != 800_000_000 + index * 1000
                        or record.get("sampler_seed") != 900_000_000 + index * 1000):
                    raise ValueError(f"Episode identity/outcome mismatch in {path}:{line_number}")
                if arm != "bf16":
                    evidence = record.get("real_quant_evidence", {})
                    if (evidence.get("rotation_bank") != arm or evidence.get("activation_bits") != 4
                            or evidence.get("integer_gemm_calls", 0) <= 0
                            or evidence.get("native_int4_gemm_calls", 0) <= 0
                            or evidence.get("native_int4_tensorcore") is not True
                            or evidence.get("packed_weight_tensor_bytes", 0) <= 0
                            or evidence.get("bf16_target_weight_absent") is not True):
                        raise ValueError(f"Native W4A4 evidence is incomplete in {path}:{line_number}")
                if slot in found:
                    raise ValueError(f"Duplicate completed slot: {slot}")
                found[slot] = record
        worker_slots = {(arm, index) for arm in ARMS for index in indices}
        if not worker_slots.issubset(found):
            missing = sorted(worker_slots - found.keys())[:5]
            raise ValueError(f"Worker {worker} is missing completed slots: {missing}")
    if found.keys() != expected:
        raise ValueError(f"Expected 840 unique episodes, found {len(found)}")
    for arm in ARMS:
        if sum(slot_arm == arm for slot_arm, _ in found) != 280:
            raise ValueError(f"Arm {arm} does not cover exactly 280 episodes")
    return found


def rate(records, arm, indices):
    success = sum(records[(arm, index)]["success"] for index in indices)
    total = len(indices)
    return {"successes": success, "episodes": total, "success_rate": success / total}


def paired(records, arm, indices, reference_arm="bf16"):
    wins = losses = ties = 0
    for index in indices:
        candidate = records[(arm, index)]["success"]
        reference = records[(reference_arm, index)]["success"]
        if candidate and not reference:
            wins += 1
        elif reference and not candidate:
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


def timing_stats(values):
    values = sorted(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Timing summary requires finite samples")
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[p95_index],
        "max_ms": values[-1],
    }


def summarize_timing(records):
    output = {}
    for arm in ARMS:
        query_samples = {
            name: {"denoising": [], "full_query": []}
            for name in ("first_query", "subsequent_queries")
        }
        episode_samples = []
        first_query_episode_samples = []
        later_episode_samples = []
        for index in range(280):
            record = records[(arm, index)]
            episode_ms = float(record["episode_seconds"]) * 1000.0
            if not math.isfinite(episode_ms) or episode_ms < 0:
                raise ValueError(f"Invalid episode wall time for {arm}/{index}")
            episode_samples.append(episode_ms)
            metrics = record.get("metrics")
            if not isinstance(metrics, list) or not metrics:
                raise ValueError(f"Missing query timing metrics for {arm}/{index}")
            had_first_query = False
            for metric in metrics:
                query_index = metric.get("query_index_in_arm")
                first = metric.get("first_query_in_arm")
                if (type(query_index) is not int or query_index < 0 or type(first) is not bool
                        or first != (query_index == 0)):
                    raise ValueError(f"Invalid first/steady query marker for {arm}/{index}")
                key = "first_query" if first else "subsequent_queries"
                values = (
                    float(metric["denoising_latency_ms"]),
                    float(metric["full_query_wall_ms"]),
                )
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"Invalid query timing for {arm}/{index}")
                query_samples[key]["denoising"].append(values[0])
                query_samples[key]["full_query"].append(values[1])
                had_first_query |= first
            (first_query_episode_samples if had_first_query else later_episode_samples).append(episode_ms)

        by_query_kind = {}
        for kind, values in query_samples.items():
            by_query_kind[kind] = {
                "queries": len(values["denoising"]),
                "continuous_denoising_gpu_interval_ms": timing_stats(values["denoising"]),
                "full_action_query_wall_ms": timing_stats(values["full_query"]),
            }
        output[arm] = {
            "query_timing": by_query_kind,
            "episode_wall_including_environment_ms": timing_stats(episode_samples),
            "episode_wall_for_first_query_episodes_ms": timing_stats(first_query_episode_samples),
            "episode_wall_for_subsequent_query_episodes_ms": timing_stats(later_episode_samples),
        }
    return output


def summarize(rows, records):
    suite_names = sorted({row["suite"] for row in rows})
    dimensions = sorted({row["dimension"] for row in rows})
    all_indices = list(range(len(rows)))
    groups = {
        "suites": {suite: [i for i, row in enumerate(rows) if row["suite"] == suite] for suite in suite_names},
        "dimensions": {dim: [i for i, row in enumerate(rows) if row["dimension"] == dim] for dim in dimensions},
        "cells": {
            f"{suite}::{dim}": [i for i, row in enumerate(rows) if row["suite"] == suite and row["dimension"] == dim]
            for suite in suite_names for dim in dimensions
        },
    }
    output = {"protocol": PROTOCOL, "complete": True, "manifest_variants": len(rows), "episode_slots": len(records)}
    output["timing"] = summarize_timing(records)
    output["overall"] = {
        "arms": {arm: rate(records, arm, all_indices) for arm in ARMS},
        "paired_vs_bf16": {arm: paired(records, arm, all_indices) for arm in ARMS[1:]},
        "paired_spinquant_vs_quarot": paired(records, ARMS[2], all_indices, ARMS[1]),
    }
    for level, buckets in groups.items():
        output[level] = {}
        for name, indices in buckets.items():
            output[level][name] = {
                "arms": {arm: rate(records, arm, indices) for arm in ARMS},
                "paired_vs_bf16": {arm: paired(records, arm, indices) for arm in ARMS[1:]},
                "paired_spinquant_vs_quarot": paired(records, ARMS[2], indices, ARMS[1]),
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="Directory containing worker_0..worker_3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _doc, rows, assigned = load_manifest(args.manifest)
    records = load_episodes(args.results, rows, assigned)
    summary = summarize(rows, records)
    write_json(args.out, summary)
    print(f"AGGREGATION_COMPLETE slots=840 arms=3 cells=28 output={args.out}", flush=True)


if __name__ == "__main__":
    main()
