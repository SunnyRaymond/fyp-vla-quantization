"""Aggregate paired Cosmos Policy results without importing model dependencies."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics

from evaluate import ARMS, PROTOCOL, load_completed, load_manifest, read_json, save_json, validate_protocol


def _rate(records, arm, indices):
    values = [records[arm, index] for index in indices if (arm, index) in records]
    successes = sum(record["success"] for record in values)
    return {"successes": successes, "episodes": len(values),
            "expected_episodes": len(indices),
            "success_rate": successes / len(values) if values else None}


def _paired(records, arm, other, indices):
    paired = [(records[arm, index]["success"], records[other, index]["success"])
              for index in indices if (arm, index) in records and (other, index) in records]
    wins = sum(candidate and not reference for candidate, reference in paired)
    losses = sum(reference and not candidate for candidate, reference in paired)
    ties = len(paired) - wins - losses
    n = len(paired)
    return {"wins": wins, "losses": losses, "ties": ties, "episodes": n,
            "success_rate_difference": (wins - losses) / n if n else None}


def _bucket(records, indices):
    return {"arms": {arm: _rate(records, arm, indices) for arm in ARMS},
            "paired": {arm: {other: _paired(records, arm, other, indices)
                             for other in ARMS if other != arm} for arm in ARMS}}


def _stats(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timing or memory samples must be finite and nonnegative")
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)], "max": values[-1]}


def _timing(records):
    output = {}
    for arm in ARMS:
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
        output[arm] = {"queries": by_group,
                       "episode_wall_ms": _stats(episodes),
                       "query_timing_scope": "CosmosRuntime query_seconds includes the complete action-query wall interval; denoising is reported only when separately measurable."}
    return output


def _encoding(records):
    output = {}
    keys = ("arm", "execution", "execution_scope", "native_int4_kernel_used",
            "weight_format", "activation_format", "transform", "target_module_count",
            "target_weight_numel", "same_module_effective_bpw", "whole_dit_effective_bpw",
            "bpw_match_relative_delta", "artifact_dir", "artifact_manifest")
    for arm in ARMS:
        evidence = [record["quantization_evidence"] for (slot_arm, _), record in records.items()
                    if slot_arm == arm]
        if not evidence:
            output[arm] = None
            continue
        canonical = {key: evidence[0].get(key) for key in keys}
        for item in evidence[1:]:
            current = {key: item.get(key) for key in keys}
            if current != canonical:
                raise ValueError(f"Quantization evidence changes within arm {arm}")
        output[arm] = canonical
    return output


def summarize(protocol, rows, records):
    expected = {(arm, index) for arm in ARMS for index in range(len(rows))}
    unexpected = set(records) - expected
    if unexpected:
        raise ValueError(f"Unexpected episode slots: {sorted(unexpected)[:5]}")
    counts = collections.Counter(arm for arm, _ in records)
    complete = set(records) == expected and all(counts[arm] == 280 for arm in ARMS)
    all_indices = list(range(len(rows)))
    suites = sorted({row["suite"] for row in rows})
    dimensions = sorted({row["dimension"] for row in rows})
    by_suite = {suite: [row["index"] for row in rows if row["suite"] == suite] for suite in suites}
    by_dimension = {dimension: [row["index"] for row in rows if row["dimension"] == dimension]
                    for dimension in dimensions}
    summary = {
        "protocol": PROTOCOL, "status": "complete" if complete else "checkpointed",
        "complete": complete, "episode_slots": len(records), "expected_episode_slots": len(expected),
        "arm_episodes": {arm: counts[arm] for arm in ARMS},
        "missing_slots_by_arm": {arm: sorted(index for candidate, index in expected - set(records)
                                              if candidate == arm) for arm in ARMS},
        "execution_boundary": protocol["execution_boundary"],
        "model": protocol["model"], "mode": protocol["mode"],
        "calibration_source": protocol["calibration"],
        "cohort_protocol": protocol["cohort"]["manifest_protocol"],
        "claim_scope": ("Paired closed-loop success on the frozen 280-variant LIBERO-Plus cohort. "
                        "Low-bit arms are decoded-BF16/fake-A4 numerical comparisons; native low-bit kernel "
                        "speedup is not measured."),
        "overall": _bucket(records, all_indices),
        "suites": {key: _bucket(records, indices) for key, indices in by_suite.items()},
        "dimensions": {key: _bucket(records, indices) for key, indices in by_dimension.items()},
        "cells": {f"{suite}::{dimension}": _bucket(records, [row["index"] for row in rows
                              if row["suite"] == suite and row["dimension"] == dimension])
                  for suite in suites for dimension in dimensions},
        "timing_and_memory": _timing(records),
        "encoding_evidence": _encoding(records),
    }
    return summary


def render_results(summary):
    lines = ["# Cosmos Policy C8 W4A4 闭环评测", "",
             f"状态：`{summary['status']}`；完成 {summary['episode_slots']}/{summary['expected_episode_slots']} 个 episode。", "",
             "| Arm | Successes / Episodes | Success rate | Paired wins/losses/ties vs BF16 |",
             "|---|---:|---:|---:|"]
    for arm in ARMS:
        score = summary["overall"]["arms"][arm]
        rate = "—" if score["success_rate"] is None else f"{score['success_rate']:.2%}"
        pair = summary["overall"]["paired"].get(arm, {}).get("bf16")
        paired = "—" if pair is None else f"{pair['wins']}/{pair['losses']}/{pair['ties']}"
        lines.append(f"| {arm} | {score['successes']}/{score['episodes']} | {rate} | {paired} |")
    lines += ["", "Latency 按每个 worker/process/arm 的首个正式 query 与后续 warm queries 分开；显存记录 query peak allocated/reserved。",
              "两种量化 arm 仅报告 decoded-BF16/fake-A4 数值评测，不代表 native INT4 kernel 性能。",
              "Suite、扰动维度、suite×dimension 单元及所有配对统计详见 `summary.json`。", ""]
    return "\n".join(lines)


def run(args):
    protocol = validate_protocol(read_json(args.protocol))
    _, rows = load_manifest(args.manifest or Path(protocol["cohort"]["manifest"]), protocol)
    records = load_completed(args.results, rows, protocol)
    summary = summarize(protocol, rows, records)
    save_json(args.results / "summary.json", summary)
    (args.results / "RESULTS.zh.md").write_text(render_results(summary), encoding="utf-8")
    print(f"COSMOS_AGGREGATE status={summary['status']} slots={len(records)}/840", flush=True)
    return summary


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=here / "protocol.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
