"""Summarize hierarchical DINO-WM Wall latency events without extra dependencies."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
out = args.output.resolve()

events = [
    json.loads(line)
    for line in (out / "latency_events.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not events:
    raise SystemExit("No latency events found")


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


children_ms = defaultdict(float)
for event in events:
    if event["parent_id"] is not None:
        children_ms[event["parent_id"]] += event["duration_ms"]
for event in events:
    event["exclusive_ms"] = max(
        0.0, event["duration_ms"] - children_ms[event["event_id"]]
    )

root_events = [event for event in events if event["label"] == "pipeline.total"]
if len(root_events) != 1:
    raise SystemExit(f"Expected one pipeline.total event, found {len(root_events)}")
pipeline_ms = root_events[0]["duration_ms"]

groups = defaultdict(list)
for event in events:
    groups[event["label"]].append(event)

by_label = []
for label, rows in groups.items():
    durations = [row["duration_ms"] for row in rows]
    exclusive = sum(row["exclusive_ms"] for row in rows)
    by_label.append(
        {
            "label": label,
            "count": len(rows),
            "inclusive_total_ms": sum(durations),
            "exclusive_total_ms": exclusive,
            "exclusive_pipeline_percent": 100.0 * exclusive / pipeline_ms,
            "mean_ms": statistics.fmean(durations),
            "p50_ms": percentile(durations, 0.50),
            "p95_ms": percentile(durations, 0.95),
            "max_ms": max(durations),
        }
    )
by_label.sort(key=lambda row: row["exclusive_total_ms"], reverse=True)

rounds = defaultdict(lambda: {"events": 0, "inclusive_ms": 0.0})
cem_steps = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "values": []})
for event in events:
    mpc_iter = event.get("context", {}).get("mpc_iter")
    if mpc_iter is None:
        continue
    row = rounds[str(mpc_iter)]
    row["events"] += 1
    if event["label"] == "mpc.round.total":
        row["inclusive_ms"] += event["duration_ms"]
    cem_opt_step = event.get("context", {}).get("cem_opt_step")
    if cem_opt_step is not None and event["label"] == "cem.optimization_step.total":
        step_row = cem_steps[str(cem_opt_step)]
        step_row["count"] += 1
        step_row["total_ms"] += event["duration_ms"]
        step_row["values"].append(event["duration_ms"])

for row in cem_steps.values():
    values = row.pop("values")
    row["mean_ms"] = statistics.fmean(values)
    row["p95_ms"] = percentile(values, 0.95)

evaluator_kinds = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "values": []})
for event in events:
    if event["label"] != "evaluator.total":
        continue
    kind = event.get("context", {}).get("evaluator_kind", "unknown")
    evaluator_kinds[kind]["count"] += 1
    evaluator_kinds[kind]["total_ms"] += event["duration_ms"]
    evaluator_kinds[kind]["values"].append(event["duration_ms"])
for row in evaluator_kinds.values():
    row["mean_ms"] = statistics.fmean(row.pop("values"))

run_summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
summary = {
    "pipeline_total_ms": pipeline_ms,
    "observed_process_elapsed_ms": run_summary["elapsed_seconds"] * 1000.0,
    "profiler_jsonl_write_ms": run_summary["profiler_jsonl_write_seconds"] * 1000.0,
    "n_events": len(events),
    "n_evals": run_summary["n_evals"],
    "successes": run_summary["successes"],
    "max_mpc_rounds": run_summary["max_mpc_rounds"],
    "profiling_boundary": run_summary["profiling_boundary"],
    "by_label": by_label,
    "by_mpc_round": dict(sorted(rounds.items(), key=lambda item: int(item[0]))),
    "by_cem_optimization_step": dict(
        sorted(cem_steps.items(), key=lambda item: int(item[0]))
    ),
    "evaluator_kinds": evaluator_kinds,
}
(out / "latency_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

lines = [
    "# DINO-WM Wall latency profile",
    "",
    f"- Complete cases: {run_summary['n_evals']}; successes: {run_summary['successes']}",
    f"- Pipeline: {pipeline_ms / 1000:.3f} s; observed process: {run_summary['elapsed_seconds']:.3f} s",
    f"- JSONL write overhead: {run_summary['profiler_jsonl_write_seconds'] * 1000:.3f} ms",
    f"- Boundary: {run_summary['profiling_boundary']}",
    "",
    "## Largest exclusive contributors",
    "",
    "| label | count | exclusive s | pipeline % | inclusive mean ms | p95 ms |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in by_label[:25]:
    lines.append(
        f"| {row['label']} | {row['count']} | {row['exclusive_total_ms'] / 1000:.3f} | "
        f"{row['exclusive_pipeline_percent']:.2f} | {row['mean_ms']:.3f} | {row['p95_ms']:.3f} |"
    )
lines.extend(
    [
        "",
        "## MPC rounds",
        "",
        "| round | wall s | events |",
        "|---:|---:|---:|",
    ]
)
for round_id, row in summary["by_mpc_round"].items():
    lines.append(f"| {int(round_id) + 1} | {row['inclusive_ms'] / 1000:.3f} | {row['events']} |")
lines.extend(
    [
        "",
        "## CEM optimization steps",
        "",
        "| step | count | total s | mean ms | p95 ms |",
        "|---:|---:|---:|---:|---:|",
    ]
)
for step_id, row in summary["by_cem_optimization_step"].items():
    lines.append(
        f"| {int(step_id) + 1} | {row['count']} | {row['total_ms'] / 1000:.3f} | "
        f"{row['mean_ms']:.3f} | {row['p95_ms']:.3f} |"
    )
lines.extend(
    [
        "",
        "## Evaluator calls",
        "",
        "| kind | count | total s | mean ms |",
        "|---|---:|---:|---:|",
    ]
)
for kind, row in sorted(evaluator_kinds.items()):
    lines.append(
        f"| {kind} | {row['count']} | {row['total_ms'] / 1000:.3f} | {row['mean_ms']:.3f} |"
    )
(out / "LATENCY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"pipeline_total_ms": pipeline_ms, "n_events": len(events)}, indent=2))
