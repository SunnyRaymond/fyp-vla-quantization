#!/usr/bin/env python3
"""Read-only CPU posthoc of saved student-CEM candidate banks."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RANKER_DIR = HERE.parent / "onpolicy-fullbank-ranker"
SOURCE_JOB_ID = "25538135.pbs101"
ROUNDS = (10, 20, 30)
STEPS = (0, 25)
TASK_FIELDS = ("selection_order", "episode_idx", "row_index", "start_step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = RANKER_DIR / "artifacts" / SOURCE_JOB_ID
    parser.add_argument("--freeze", type=Path, default=RANKER_DIR / "FREEZE.json")
    parser.add_argument("--train-banks", type=Path, default=source / "train_banks.pt")
    parser.add_argument("--validation-banks", type=Path, default=source / "validation_banks.pt")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_cpu_compute_node() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("requires PBS_JOBID and a real PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes or any(any(token in node for token in ("login", "submit", "head")) for node in nodes):
        raise RuntimeError("current host must be a PBS compute node listed in PBS_NODEFILE")
    ngpus = os.environ.get("PBS_NGPUS", "0").strip()
    if ngpus.isdigit() and int(ngpus) > 0:
        raise RuntimeError(f"refusing GPU allocation PBS_NGPUS={ngpus}")
    gpufile = os.environ.get("PBS_GPUFILE", "").strip()
    if gpufile and Path(gpufile).is_file() and Path(gpufile).stat().st_size > 0:
        raise RuntimeError("refusing a non-empty PBS_GPUFILE")
    if re.search(r"(?:^|[:,])\s*ngpus\s*=\s*[1-9]\d*", os.environ.get("PBS_RESOURCE_LIST", ""), re.I):
        raise RuntimeError("refusing a GPU PBS resource request")
    return job_id, host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def task_id(row: dict[str, Any]) -> tuple[int, int, int, int]:
    try:
        return tuple(int(row[key]) for key in TASK_FIELDS)  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid task identity: {row}") from exc


def load_frozen_split(path: Path) -> dict[str, list[dict[str, int]]]:
    freeze = read_json(path)
    if freeze.get("schema") != "lewm-pusht-onpolicy-fullbank-ranker-freeze" or freeze.get("schema_version") != 1:
        raise ValueError("unexpected source freeze schema")
    split = freeze["task_split"]
    tasks = {name: split[name]["tasks"] for name in ("training", "validation", "untouched")}
    if tuple(map(len, tasks.values())) != (16, 8, 8):
        raise ValueError("source task split must contain 16/8/8 tasks")
    if len({task_id(row) for group in tasks.values() for row in group}) != 32:
        raise ValueError("source task identities overlap")
    return tasks


def finite_tensor(record: dict[str, Any], key: str, shape: tuple[int, ...]) -> Any:
    value = torch.as_tensor(record[key], dtype=torch.float32, device="cpu")
    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{key} must be finite CPU data with shape {shape}")
    return value


def metric_row(record: dict[str, Any], split: str, expected_tasks: set[tuple[int, ...]]) -> dict[str, Any]:
    identity = task_id(record)
    if identity not in expected_tasks or record.get("split") != split:
        raise ValueError(f"bank task/split differs from the frozen {split} split")
    if record.get("schema") != "lewm-pusht-onpolicy-fullbank-bank-v1":
        raise ValueError("unexpected full-bank record schema")
    step, round_index = int(record.get("replan_step", -1)), int(record.get("cem_round", -1))
    if step not in STEPS or round_index not in ROUNDS:
        raise ValueError("unexpected replan_step or CEM round")
    candidates = finite_tensor(record, "candidates", (300, 5, 10))
    student = finite_tensor(record, "student_costs", (300,))
    teacher = finite_tensor(record, "teacher_costs", (300,))

    student_top = torch.argsort(student)[:30]
    full_std = candidates.std(dim=0, unbiased=False)
    top_std = candidates[student_top].std(dim=0, unbiased=False)
    teacher_min = float(teacher.min())
    student_top_teacher_min = float(teacher[student_top].min())
    teacher_scale = float(teacher.std(unbiased=False))
    if not math.isfinite(teacher_scale) or teacher_scale <= 1e-12:
        raise ValueError("teacher cost population std must be positive and finite")
    full_values, top_values = full_std.reshape(-1).tolist(), top_std.reshape(-1).tolist()
    return {
        "split": split,
        "task": {name: identity[i] for i, name in enumerate(TASK_FIELDS)},
        "episode_idx": identity[1],
        "replan_step": step,
        "cem_round": round_index,
        "full_bank_action_std_mean": float(statistics.mean(full_values)),
        "full_bank_action_std_by_coordinate": full_std.tolist(),
        "student_top30_action_std_mean": float(statistics.mean(top_values)),
        "student_top30_action_std_by_coordinate": top_std.tolist(),
        "teacher_best_cost_min": teacher_min,
        "student_top30_best_teacher_cost": student_top_teacher_min,
        "raw_best_cost_gap": student_top_teacher_min - teacher_min,
        "teacher_cost_std_this_round": teacher_scale,
    }


def load_split(path: Path, name: str, tasks: list[dict[str, int]], expected_count: int) -> list[dict[str, Any]]:
    try:
        records = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older cluster PyTorch.
        records = torch.load(path, map_location="cpu")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"{name} banks must contain exactly {expected_count} records")
    expected = {task_id(row) for row in tasks}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int, int]] = set()
    rounds_by_solve: dict[tuple[tuple[int, ...], int], set[int]] = defaultdict(set)
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("every bank must be a dictionary")
        row = metric_row(record, name, expected)
        identity = task_id(row["task"])
        key = (identity, row["replan_step"], row["cem_round"])
        if key in seen:
            raise ValueError(f"duplicate bank: {key}")
        seen.add(key)
        rounds_by_solve[(identity, row["replan_step"])].add(row["cem_round"])
        rows.append(row)

    if {task_id(row) for row in tasks} != {task_id(row["task"]) for row in rows}:
        raise ValueError(f"{name} banks do not cover the frozen task split")
    for task in tasks:
        identity = task_id(task)
        if rounds_by_solve.get((identity, 0)) != set(ROUNDS):
            raise ValueError(f"{name} task {identity} lacks complete t0 rounds")
        t25 = rounds_by_solve.get((identity, 25), set())
        if t25 and t25 != set(ROUNDS):
            raise ValueError(f"{name} task {identity} has incomplete t25 rounds")
    return rows


def add_round10_scaled_gap(rows: list[dict[str, Any]]) -> None:
    scales: dict[tuple[str, tuple[int, ...], int], float] = {}
    for row in rows:
        if row["cem_round"] == 10:
            key = (row["split"], task_id(row["task"]), row["replan_step"])
            scales[key] = row["teacher_cost_std_this_round"]
    for row in rows:
        key = (row["split"], task_id(row["task"]), row["replan_step"])
        if key not in scales:
            raise ValueError(f"round10 teacher scale missing for {key}")
        row["teacher_cost_std_round10_same_solve"] = scales[key]
        row["standardized_best_cost_gap_round10_scale"] = row["raw_best_cost_gap"] / scales[key]
        row.pop("teacher_cost_std_this_round")


def median_matrix(rows: list[dict[str, Any]], key: str) -> list[list[float]]:
    return [
        [statistics.median(float(row[key][i][j]) for row in rows) for j in range(10)]
        for i in range(5)
    ]


def stratum_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["replan_step"], row["cem_round"])].append(row)
    output = []
    scalar_keys = ("full_bank_action_std_mean", "student_top30_action_std_mean", "teacher_best_cost_min", "student_top30_best_teacher_cost", "raw_best_cost_gap", "standardized_best_cost_gap_round10_scale")
    for (split, step, round_index), group in sorted(grouped.items()):
        output.append({
            "split": split,
            "replan_step": step,
            "cem_round": round_index,
            "source_episode_count": len(group),
            "median_metrics": {key: statistics.median(float(row[key]) for row in group) for key in scalar_keys},
            "median_full_bank_action_std_by_coordinate": median_matrix(group, "full_bank_action_std_by_coordinate"),
            "median_student_top30_action_std_by_coordinate": median_matrix(group, "student_top30_action_std_by_coordinate"),
        })
    return output


def paired_trends(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_solve: dict[tuple[str, tuple[int, ...], int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["split"], task_id(row["task"]), row["replan_step"])
        by_solve[key][row["cem_round"]] = row
    deltas: dict[tuple[str, int, int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    fields = ("full_bank_action_std_mean", "student_top30_action_std_mean", "teacher_best_cost_min", "standardized_best_cost_gap_round10_scale")
    for (split, _task, step), rounds in by_solve.items():
        for first, second in ((10, 20), (20, 30), (10, 30)):
            if first in rounds and second in rounds:
                for field in fields:
                    deltas[(split, step, first, second)][field].append(float(rounds[second][field]) - float(rounds[first][field]))
    return [
        {
            "split": split,
            "replan_step": step,
            "from_round": first,
            "to_round": second,
            "paired_episode_solve_count": len(metrics["full_bank_action_std_mean"]),
            "median_paired_delta": {key: statistics.median(values) for key, values in metrics.items()},
        }
        for (split, step, first, second), metrics in sorted(deltas.items())
    ]


def pattern_readout(strata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    medians = {(row["split"], row["replan_step"], row["cem_round"]): row["median_metrics"] for row in strata}
    output = []
    for split in ("train", "validation"):
        for step in STEPS:
            early, late = medians.get((split, step, 10)), medians.get((split, step, 30))
            if early is None or late is None:
                continue
            contraction = late["full_bank_action_std_mean"] < early["full_bank_action_std_mean"] and late["student_top30_action_std_mean"] < early["student_top30_action_std_mean"]
            worsening = late["standardized_best_cost_gap_round10_scale"] > early["standardized_best_cost_gap_round10_scale"]
            output.append({
                "split": split,
                "replan_step": step,
                "sampled_bank_and_top30_dispersion_contract": contraction,
                "student_top30_best_cost_gap_worsens": worsening,
                "pattern": "both" if contraction and worsening else "dispersion_only" if contraction else "best_cost_gap_only" if worsening else "neither",
                "interpretation": "descriptive only; no threshold, gate, or causal claim",
            })
    return output


def main() -> None:
    args = parse_args()
    job_id, host = require_cpu_compute_node()
    global torch
    import torch

    splits = load_frozen_split(args.freeze)
    train_path, validation_path = args.train_banks.resolve(), args.validation_banks.resolve()
    if train_path.parent != validation_path.parent or train_path == validation_path:
        raise ValueError("train and validation banks must share the frozen source directory")
    output_path = args.output.resolve()
    if train_path.parent == output_path.parent or train_path.parent in output_path.parents:
        raise ValueError("output must be outside the immutable source bank directory")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")

    rows = load_split(train_path, "train", splits["training"], 93)
    rows += load_split(validation_path, "validation", splits["validation"], 48)
    add_round10_scaled_gap(rows)
    rows.sort(key=lambda row: (row["split"], row["episode_idx"], row["replan_step"], row["cem_round"]))
    strata = stratum_summaries(rows)
    report = {
        "schema": "lewm-pusht-cem-proposal-dispersion-posthoc-result-v1",
        "report_kind": "READ_ONLY_DESCRIPTIVE_MECHANISM_POSTHOC",
        "pbs_job_id": job_id,
        "compute_host": host,
        "source_job_id": SOURCE_JOB_ID,
        "bank_counts": {"train": 93, "validation_previously_used": 48},
        "dispersion_definition": "Population std over 300 candidates, separately for each coordinate in [5,10]; values are raw saved CEM proposals without clipping because proposal coordinates are unconstrained.",
        "best_cost_gap_definition": "Minimum teacher cost among student-cost top30 minus the minimum teacher cost in the full bank; standardized for every round by the same episode/replan solve's round10 teacher-cost population std.",
        "per_episode_solve_banks": rows,
        "split_step_round_medians": strata,
        "paired_round_trends": paired_trends(rows),
        "descriptive_pattern_readout": pattern_readout(strata),
        "validation_use": "The validation split was already used by the frozen residual-ranker gate; descriptive only, not a new independent gate or tuning signal.",
        "claim_boundary": "Describes sampled bank width and teacher-best reachability on existing baseline-generated banks. Initial proposal parameters are absent, so dispersion cannot identify its cause. No mixture-CEM, planner, or closed-loop result is established.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"wrote read-only proposal-dispersion posthoc: {output_path}", flush=True)


if __name__ == "__main__":
    main()
