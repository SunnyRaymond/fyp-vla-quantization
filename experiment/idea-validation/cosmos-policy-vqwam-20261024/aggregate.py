"""Paired six-method summary for new Cosmos episodes and the two frozen reused arms."""
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path
import statistics
import os

import evaluate

ARMS = evaluate.ARMS
NEW_ARMS = evaluate.NEW_ARMS
REUSED_ARMS = evaluate.REUSED_ARMS
ENCODING_FIELDS = (
    "arm", "execution", "execution_scope", "native_int4_kernel_used", "weight_format",
    "activation_format", "transform", "target_module_count", "target_weight_numel",
    "same_module_effective_bpw", "whole_dit_effective_bpw", "bpw_match_relative_delta",
    "effective_bpw", "encoded_target_bpw", "artifact_dir", "artifact_manifest",
    "implementation", "weight_bits", "protected_weight_bits", "activation_bits",
    "protected_activation_bits", "weight_quantizer", "activation_quantizer",
    "weight_group_size", "global_svd_rank", "packed_gpu_resident", "dense_target_weights",
)


def _rate(records, arm, indices):
    values = [records[arm, index] for index in indices if (arm, index) in records]
    successes = sum(bool(record["success"]) for record in values)
    return {"successes": successes, "episodes": len(values), "expected_episodes": len(indices),
            "success_rate": successes / len(values) if values else None}


def _paired(records, arm, other, indices):
    paired = [(bool(records[arm, index]["success"]), bool(records[other, index]["success"]))
              for index in indices if (arm, index) in records and (other, index) in records]
    wins = sum(candidate and not reference for candidate, reference in paired)
    losses = sum(reference and not candidate for candidate, reference in paired)
    count = len(paired)
    return {"wins": wins, "losses": losses, "ties": count - wins - losses,
            "episodes": count, "expected_episodes": len(indices),
            "success_rate_difference": (wins - losses) / count if count else None}


def _bucket(records, indices, arms):
    return {"arms": {arm: _rate(records, arm, indices) for arm in arms},
            "paired": {arm: {other: _paired(records, arm, other, indices)
                             for other in arms if other != arm} for arm in arms}}


def _stats(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timing or memory samples must be finite and nonnegative")
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)], "max": values[-1]}


def _timing(records, arms):
    output = {}
    for arm in arms:
        groups = {"first_query_in_run": [], "warm_queries": []}
        episodes = []
        for (slot_arm, _), record in records.items():
            if slot_arm != arm:
                continue
            episodes.append(record["episode_seconds"] * 1000.0)
            for metric in record["query_metrics"]:
                key = "first_query_in_run" if metric["first_query_in_run"] else "warm_queries"
                groups[key].append(metric)
        by_group = {}
        for name, metrics in groups.items():
            by_group[name] = {
                "queries": len(metrics),
                "query_wall_ms": _stats([metric["query_seconds"] * 1000.0 for metric in metrics]),
                "denoising_ms": _stats([metric["denoising_seconds"] * 1000.0 for metric in metrics
                                         if metric["denoising_seconds"] is not None]),
                "peak_allocated_gpu_gb": _stats([metric["peak_allocated_gb"] for metric in metrics]),
                "peak_reserved_gpu_gb": _stats([metric["peak_reserved_gb"] for metric in metrics]),
            }
        output[arm] = {"queries": by_group, "episode_wall_ms": _stats(episodes),
                       "query_timing_scope": (
                           "CosmosRuntime query_seconds is the complete action-query interval; "
                           "denoising_seconds is reported only when separately measurable.")}
    return output


def _encoding(records, arms):
    output = {}
    for arm in arms:
        items = [record["quantization_evidence"] for (slot_arm, _), record in records.items()
                 if slot_arm == arm]
        if not items:
            output[arm] = None
            continue
        canonical = {key: items[0].get(key) for key in ENCODING_FIELDS if key in items[0]}
        for item in items[1:]:
            current = {key: item.get(key) for key in canonical}
            if current != canonical:
                raise ValueError(f"Stable quantization evidence changes within arm {arm}")
        output[arm] = canonical
    return output


def _strata(protocol, rows, records, arms):
    indices = list(range(len(rows)))
    suites = sorted({row["suite"] for row in rows})
    dimensions = sorted({row["dimension"] for row in rows})
    by_suite = {suite: [row["index"] for row in rows if row["suite"] == suite] for suite in suites}
    by_dimension = {dimension: [row["index"] for row in rows if row["dimension"] == dimension]
                    for dimension in dimensions}
    return {"overall": _bucket(records, indices, arms),
            "suites": {key: _bucket(records, values, arms) for key, values in by_suite.items()},
            "dimensions": {key: _bucket(records, values, arms) for key, values in by_dimension.items()},
            "cells": {f"{suite}::{dimension}": _bucket(
                records, [row["index"] for row in rows
                          if row["suite"] == suite and row["dimension"] == dimension], arms)
                for suite in suites for dimension in dimensions}}


def _counts(records, arms, rows):
    expected = {(arm, index) for arm in arms for index in range(len(rows))}
    unexpected = set(records) - expected
    if unexpected:
        raise ValueError(f"Unexpected episode slots: {sorted(unexpected)[:5]}")
    counts = collections.Counter(arm for arm, _ in records)
    missing = {arm: sorted(index for candidate, index in expected - set(records) if candidate == arm)
               for arm in arms}
    return counts, missing


def _worker_state(worker_exitcodes, expected_workers):
    codes = dict(worker_exitcodes or {})
    failed = any(code not in (0, None) for code in codes.values())
    complete = (set(codes) == {f"cosmos-gpu-worker-{i}" for i in range(expected_workers)}
                and all(type(code) is int and code == 0 for code in codes.values()))
    return codes, failed, complete


def summarize(protocol, rows, new_records, reused_records, worker_exitcodes=None, pbs_job_id=None,
              node_receipts_complete=True):
    if len(rows) != 280:
        raise ValueError("Final aggregation requires all 280 frozen manifest rows")
    new_counts, new_missing = _counts(new_records, NEW_ARMS, rows)
    reused_counts, reused_missing = _counts(reused_records, REUSED_ARMS, rows)
    all_records = dict(new_records)
    if set(all_records).intersection(reused_records):
        raise ValueError("New and reused arms overlap in episode slots")
    all_records.update(reused_records)
    all_counts, all_missing = _counts(all_records, ARMS, rows)
    expected_workers = evaluate.expected_worker_count(protocol)
    exitcodes, worker_failed, workers_complete = _worker_state(worker_exitcodes, expected_workers)
    new_complete = not any(new_missing.values())
    reuse_complete = not any(reused_missing.values())
    complete = new_complete and reuse_complete and workers_complete and node_receipts_complete
    status = ("failed" if worker_failed or not node_receipts_complete
              else "complete" if complete else "checkpointed")
    new_strata = _strata(protocol, rows, new_records, NEW_ARMS)
    all_strata = _strata(protocol, rows, all_records, ARMS)
    reuse_provenance = {}
    for arm in REUSED_ARMS:
        items = [record for (candidate, _), record in reused_records.items() if candidate == arm]
        reuse_provenance[arm] = {
            "protocols": sorted({record.get("protocol") for record in items if record.get("protocol")}),
            "pbs_job_ids": sorted({record.get("pbs_job_id") for record in items if record.get("pbs_job_id")}),
            "episodes": len(items), "expected_episodes": len(rows),
            "source_root": protocol.get("reuse", {}).get("root"),
        }
    summary = {
        "protocol": protocol["protocol"], "status": status, "complete": complete,
        "pbs_job_id": pbs_job_id, "worker_exitcodes": exitcodes,
        "expected_worker_count": expected_workers,
        "node_receipts_complete": bool(node_receipts_complete),
        "new_arms": list(NEW_ARMS), "reused_arms": list(REUSED_ARMS),
        "episode_slots": len(new_records), "expected_episode_slots": len(NEW_ARMS) * len(rows),
        "arm_episodes": {arm: new_counts[arm] for arm in NEW_ARMS},
        "missing_slots_by_arm": new_missing,
        "new_episode_slots": len(new_records), "expected_new_episode_slots": len(NEW_ARMS) * len(rows),
        "new_arms_status": "complete" if new_complete else "checkpointed",
        "new_arm_episodes": {arm: new_counts[arm] for arm in NEW_ARMS},
        "missing_new_slots_by_arm": new_missing,
        "execution_boundary": protocol.get("execution_boundary"),
        "model": protocol.get("model"), "mode": protocol.get("mode"),
        "cohort_protocol": protocol.get("cohort", {}).get("manifest_protocol"),
        "new_arm_claim_scope": (
            "Four new Cosmos-adapted arms evaluated over the frozen 280-episode paired cohort; "
            "baseline arms use method-specific eager decoded-BF16/fake-A4 and ours uses the packed "
            "GPU-resident engine with decoded-BF16 GEMM; no native INT4 kernel is claimed."),
        **new_strata,
        "timing_and_memory": _timing(new_records, NEW_ARMS),
        "encoding_evidence": _encoding(new_records, NEW_ARMS),
        "reused_arm_episodes": {arm: reused_counts[arm] for arm in REUSED_ARMS},
        "missing_reused_slots_by_arm": reused_missing,
        "reused_provenance": reuse_provenance,
        "all_methods": {
            "episode_slots": len(all_records), "expected_episode_slots": len(ARMS) * len(rows),
            "arm_episodes": {arm: all_counts[arm] for arm in ARMS},
            "missing_slots_by_arm": all_missing,
            "provenance_boundary": {
                "new_arms_protocol": protocol["protocol"],
                "reused_arms_protocols": {arm: reuse_provenance[arm]["protocols"] for arm in REUSED_ARMS},
                "reused_arm_pbs_job_ids": {arm: reuse_provenance[arm]["pbs_job_ids"] for arm in REUSED_ARMS},
                "reused_records_relabelled_as_new_engine_evidence": False,
            },
            **all_strata,
            "timing_and_memory": _timing(all_records, ARMS),
            "encoding_evidence": _encoding(all_records, ARMS),
        },
    }
    return summary


def run(args):
    evaluate._guard_allocation()
    protocol = evaluate.validate_protocol(evaluate.read_json(args.protocol))
    manifest_path = args.manifest or Path(protocol["cohort"]["manifest"])
    _, rows = evaluate.load_manifest(manifest_path, protocol)
    root = Path(args.root or protocol["remote_root"])
    out = Path(args.out or root / "results" / "evaluation")
    new_records = evaluate.load_completed(out, rows, protocol)
    reused_records = evaluate.load_reused_records(root, rows, protocol, manifest_path)
    run_id = os.environ["PBS_JOBID"]
    worker_receipt_path = out / "worker_exitcodes.json"
    worker_receipt = evaluate.read_json(worker_receipt_path) if worker_receipt_path.is_file() else {}
    worker_exitcodes = (worker_receipt.get("worker_exitcodes", {})
                        if worker_receipt.get("pbs_job_id") == run_id else {})
    summary = summarize(protocol, rows, new_records, reused_records, worker_exitcodes, run_id)
    evaluate.save_json(out / "summary.json", summary)
    print(f"COSMOS_AGGREGATE status={summary['status']} "
          f"new={summary['new_episode_slots']}/1120 all={summary['all_methods']['episode_slots']}/1680",
          flush=True)
    return summary


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=here / "protocol.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--soft-deadline-seconds", type=float)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
