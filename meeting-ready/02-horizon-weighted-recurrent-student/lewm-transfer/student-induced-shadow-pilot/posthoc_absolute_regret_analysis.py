#!/usr/bin/env python3
"""Descriptive post-hoc summaries from one frozen seeded pilot's small outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PILOT_JOB_ID = "25537667.pbs101"
SUMMARY_SCHEMA = "lewm-pusht-student-induced-shadow-seeded-pilot-result"
FREEZE_SCHEMA = "lewm-pusht-student-induced-shadow-seeded-pilot-freeze"
ROUNDS = (10, 20, 30)
SOLVE_STEPS = (0, 25)
EXPECTED_SHAPE = [1, 300, 5, 10]
EXPECTED_INPUT_FREEZE = (
    "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/"
    "02-horizon-weighted-recurrent-student/lewm-transfer/"
    "student-induced-shadow-pilot/SEEDED_PILOT_FREEZE.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_compute_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id:
        raise RuntimeError("PBS_JOBID is required")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_NODEFILE must name an existing nodefile")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    node_hosts = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in node_hosts:
        raise RuntimeError(f"hostname {host} is absent from PBS_NODEFILE")
    if any(any(token in node for token in ("login", "head", "submit")) for node in node_hosts):
        raise RuntimeError("PBS_NODEFILE contains a probable login/submit host")
    return job_id, host


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def flatten_numeric(value: Any, name: str) -> list[float]:
    flattened: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        flattened.append(finite_number(item, name))

    visit(value)
    if len(flattened) != 300:
        raise RuntimeError(f"{name} must contain exactly 300 finite costs")
    return flattened


def quantile_type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("cannot summarize an empty episode sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def task_identity(task: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(task[key]) for key in ("selection_order", "episode_idx", "row_index", "start_step"))


def main() -> int:
    args = parse_args()
    analysis_job_id, compute_host = require_compute_allocation()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    for path in (args.candidate_scores, args.summary, args.freeze):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = load_json(args.summary)
    freeze = load_json(args.freeze)
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("pbs_job_id") != PILOT_JOB_ID:
        raise RuntimeError("pilot summary identity does not match the frozen source job")
    if summary.get("status") != "NO_GO":
        raise RuntimeError("source pilot summary is not the expected frozen NO_GO result")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("schema_version") != 1:
        raise RuntimeError("seeded pilot freeze schema mismatch")
    if os.path.normpath(str(args.freeze)) != os.path.normpath(EXPECTED_INPUT_FREEZE):
        raise RuntimeError("freeze path differs from the expected seeded pilot freeze")
    if os.path.normpath(str(summary.get("seeded_pilot_freeze", ""))) != os.path.normpath(EXPECTED_INPUT_FREEZE):
        raise RuntimeError("summary does not identify the expected seeded pilot freeze")
    if tuple(int(value) for value in freeze.get("shadow_scoring_rounds", [])) != ROUNDS:
        raise RuntimeError("frozen CEM scoring rounds differ from this descriptive analysis")
    if int(freeze.get("metrics", {}).get("recall_k", -1)) != 120:
        raise RuntimeError("frozen recall metric is not recall@120")

    collection_tasks = freeze.get("collection_tasks")
    if not isinstance(collection_tasks, list) or len(collection_tasks) != 16:
        raise RuntimeError("freeze must contain the 16 collection tasks")
    tasks_by_id = {int(task["episode_idx"]): task for task in collection_tasks}
    if len(tasks_by_id) != 16:
        raise RuntimeError("frozen collection episode IDs are not unique")
    expected_order = [int(task["episode_idx"]) for task in sorted(collection_tasks, key=lambda x: int(x["selection_order"]))]
    if summary.get("protocol", {}).get("collection_order") != expected_order:
        raise RuntimeError("summary collection order differs from the freeze")
    if summary.get("selection_manifest", {}).get("reserved_holdout_evaluated") is not False:
        raise RuntimeError("summary does not confirm reserved holdout remained unevaluated")

    aggregate_rows = summary.get("aggregate", {}).get("episode_rows")
    if not isinstance(aggregate_rows, list) or len(aggregate_rows) != 16:
        raise RuntimeError("summary must contain one aggregate row per collection episode")
    aggregate_by_id: dict[int, dict[str, Any]] = {}
    for row in aggregate_rows:
        episode_id = int(row["task"]["episode_idx"])
        if episode_id in aggregate_by_id or episode_id not in tasks_by_id:
            raise RuntimeError("summary episode identities are duplicate or outside collection")
        if task_identity(row["task"]) != task_identity(tasks_by_id[episode_id]):
            raise RuntimeError(f"summary task identity differs from freeze for episode {episode_id}")
        aggregate_by_id[episode_id] = row
    matched_ids = {
        episode_id for episode_id, row in aggregate_by_id.items()
        if row.get("status") == "matched_t0_t25"
    }
    nonmatched_ids = set(aggregate_by_id) - matched_ids
    if len(matched_ids) != 15 or len(nonmatched_ids) != 1:
        raise RuntimeError("summary matched/nonmatched coverage differs from the source pilot")

    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    candidate_count = 0
    if args.candidate_scores.stat().st_size > 100_000_000:
        raise RuntimeError("candidate score file exceeds the pre-set 100 MB input ceiling")
    with args.candidate_scores.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise RuntimeError(f"blank candidate-score line at {line_number}")
            row = json.loads(line)
            task = row.get("task")
            if not isinstance(task, dict):
                raise RuntimeError(f"candidate-score line {line_number} lacks task identity")
            episode_id = int(task["episode_idx"])
            if episode_id not in tasks_by_id or task_identity(task) != task_identity(tasks_by_id[episode_id]):
                raise RuntimeError(f"candidate-score task identity differs from freeze at line {line_number}")
            sim_steps = int(row["sim_steps_before_solve"])
            round_number = int(row["round"])
            if sim_steps not in SOLVE_STEPS or round_number not in ROUNDS:
                raise RuntimeError(f"unexpected solve step/round at candidate-score line {line_number}")
            if sim_steps == 25 and episode_id not in matched_ids:
                raise RuntimeError(f"nonmatched episode {episode_id} has a t25 candidate row")
            if int(row.get("solve_ordinal", -1)) != (0 if sim_steps == 0 else 1):
                raise RuntimeError(f"unexpected solve ordinal at candidate-score line {line_number}")
            if row.get("candidate_shape") != EXPECTED_SHAPE:
                raise RuntimeError(f"candidate shape mismatch at candidate-score line {line_number}")
            flatten_numeric(row.get("student_costs"), "student_costs")
            flatten_numeric(row.get("teacher_costs"), "teacher_costs")
            metrics = row.get("metrics")
            if not isinstance(metrics, dict) or metrics.get("valid") is not True:
                raise RuntimeError(f"invalid metrics at candidate-score line {line_number}")
            regret = finite_number(metrics.get("standardized_elite_regret"), "standardized_elite_regret")
            recall = finite_number(metrics.get("recall_at_120"), "recall_at_120")
            if not 0.0 <= recall <= 1.0:
                raise RuntimeError(f"recall@120 outside [0, 1] at line {line_number}")
            key = (episode_id, sim_steps, round_number)
            if key in records:
                raise RuntimeError(f"duplicate candidate-score row for {key}")
            records[key] = {
                "regret": regret,
                "recall_at_120": recall,
                "line_number": line_number,
            }
            candidate_count += 1

    expected_keys = {
        (episode_id, sim_steps, round_number)
        for episode_id in tasks_by_id
        for sim_steps in ((0, 25) if episode_id in matched_ids else (0,))
        for round_number in ROUNDS
    }
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        unexpected = sorted(set(records) - expected_keys)
        raise RuntimeError(f"candidate row coverage mismatch; missing={missing[:3]} unexpected={unexpected[:3]}")

    paired_rows: list[dict[str, Any]] = []
    for episode_id in sorted(matched_ids, key=lambda value: int(tasks_by_id[value]["selection_order"])):
        per_round: dict[str, Any] = {}
        for round_number in ROUNDS:
            t0 = records[(episode_id, 0, round_number)]
            t25 = records[(episode_id, 25, round_number)]
            delta = t25["regret"] - t0["regret"]
            per_round[str(round_number)] = {
                "t0_standardized_elite_regret": t0["regret"],
                "t25_standardized_elite_regret": t25["regret"],
                "t25_minus_t0_standardized_elite_regret": delta,
                "t0_recall_at_120": t0["recall_at_120"],
                "t25_recall_at_120": t25["recall_at_120"],
            }
            frozen_row = aggregate_by_id[episode_id]
            frozen_delta = finite_number(
                frozen_row["standardized_regret_delta_by_round"][str(round_number)],
                "summary standardized regret delta",
            )
            if not math.isclose(delta, frozen_delta, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"candidate-score delta differs from summary for episode {episode_id}, round {round_number}")
            frozen_recall = frozen_row["recall_at_120_by_round"][str(round_number)]
            if not math.isclose(t0["recall_at_120"], float(frozen_recall["t0"]), rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"t0 recall differs from summary for episode {episode_id}, round {round_number}")
            if not math.isclose(t25["recall_at_120"], float(frozen_recall["t25"]), rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"t25 recall differs from summary for episode {episode_id}, round {round_number}")
        paired_rows.append({"task": tasks_by_id[episode_id], "rounds": per_round})

    by_round: dict[str, Any] = {}
    for round_number in ROUNDS:
        by_state: dict[str, Any] = {}
        for sim_steps, state_name in ((0, "t0"), (25, "t25")):
            regrets = [records[(episode_id, sim_steps, round_number)]["regret"] for episode_id in matched_ids]
            recalls = [records[(episode_id, sim_steps, round_number)]["recall_at_120"] for episode_id in matched_ids]
            q1 = quantile_type7(regrets, 0.25)
            median = quantile_type7(regrets, 0.50)
            q3 = quantile_type7(regrets, 0.75)
            by_state[state_name] = {
                "n_episodes": len(regrets),
                "standardized_elite_regret": {
                    "q1_type7": q1,
                    "median": median,
                    "q3_type7": q3,
                    "iqr_width": q3 - q1,
                    "count_gt_0": sum(value > 0.0 for value in regrets),
                    "count_gt_0_5": sum(value > 0.5 for value in regrets),
                },
                "recall_at_120_median": quantile_type7(recalls, 0.50),
            }
        by_round[str(round_number)] = by_state

    output = {
        "schema": "lewm-pusht-student-induced-shadow-absolute-regret-posthoc-result",
        "schema_version": 1,
        "analysis_type": "descriptive_exploratory_posthoc",
        "source_pilot": {
            "pbs_job_id": PILOT_JOB_ID,
            "schema": summary["schema"],
            "frozen_status": summary["status"],
            "candidate_scores_path": str(args.candidate_scores),
            "candidate_scores_size_bytes": args.candidate_scores.stat().st_size,
            "summary_path": str(args.summary),
            "freeze_path": str(args.freeze),
            "freeze_schema": freeze["schema"],
            "candidate_score_rows_read": candidate_count,
            "matched_episode_count": len(matched_ids),
            "excluded_nonmatched_episodes": [
                {"episode_idx": episode_id, "status": aggregate_by_id[episode_id]["status"]}
                for episode_id in sorted(nonmatched_ids)
            ],
        },
        "analysis_job": {"pbs_job_id": analysis_job_id, "compute_host": compute_host},
        "unit": "source episode; one paired row per matched episode",
        "rounds": list(ROUNDS),
        "state_labels": {"t0": 0, "t25": 25},
        "quantile_method": "Hyndman-Fan Type 7 linear interpolation",
        "matched_sample_rule": "Use only the 15 source episodes marked matched_t0_t25 by the existing pilot summary, for both t0 and t25 aggregates.",
        "descriptive_outputs_by_round_and_state": by_round,
        "paired_episode_rows_file": "paired_episode_rows.jsonl",
        "claim_boundary": (
            "Descriptive post-hoc analysis only; the source NO-GO decision and its thresholds are unchanged. "
            "The t25-minus-t0 contrast combines planner context, simulator state, action history, and native CEM candidate/RNG changes; "
            "it does not isolate simulator-state causation and does not establish a GO claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "paired_episode_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in paired_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.output_dir / "posthoc_absolute_regret_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "POSTHOC_ANALYSIS_COMPLETE", "matched_episodes": len(matched_ids), "candidate_score_rows": candidate_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
