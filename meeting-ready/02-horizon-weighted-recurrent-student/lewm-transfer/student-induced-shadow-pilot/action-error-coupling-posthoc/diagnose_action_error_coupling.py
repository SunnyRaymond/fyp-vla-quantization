#!/usr/bin/env python3
"""Read-only episode-level coupling diagnostic for frozen LeWM PushT banks."""

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


TASK_FIELDS = ("selection_order", "episode_idx", "row_index", "start_step")
EXPECTED_PROBES = {
    "two_task": {
        "job_id": "25542215.pbs101",
        "schema": "lewm-pusht-real-observation-latent-alignment-probe-result",
        "count": 2,
    },
    "fourteen_task": {
        "job_id": "25542467.pbs101",
        "schema": "lewm-pusht-real-observation-latent-alignment-14-task-extension-result",
        "count": 14,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--train-banks", type=Path, required=True)
    parser.add_argument("--probe-two-task", type=Path, required=True)
    parser.add_argument("--probe-fourteen-task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_cpu_compute_node() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("requires a real PBS_JOBID and PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes or any(
        any(token in node for token in ("login", "submit", "head")) for node in nodes
    ):
        raise RuntimeError("host/PBS_NODEFILE does not identify a compute allocation")

    ngpus = os.environ.get("PBS_NGPUS", "0").strip()
    if ngpus and ngpus != "0" and (not ngpus.isdigit() or int(ngpus) > 0):
        raise RuntimeError(f"refusing non-CPU allocation PBS_NGPUS={ngpus}")
    gpufile = os.environ.get("PBS_GPUFILE", "").strip()
    if gpufile and Path(gpufile).is_file() and Path(gpufile).stat().st_size > 0:
        raise RuntimeError("refusing non-empty PBS_GPUFILE")
    resources = os.environ.get("PBS_RESOURCE_LIST", "")
    if re.search(r"(?:^|[:,])\s*ngpus\s*=\s*[1-9]\d*", resources, re.IGNORECASE):
        raise RuntimeError("refusing a GPU PBS resource request")
    return job_id, host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def task_identity(task: dict[str, Any]) -> tuple[int, int, int, int]:
    if not isinstance(task, dict):
        raise TypeError("task identity must be a JSON object")
    try:
        return tuple(int(task[key]) for key in TASK_FIELDS)  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid task identity: {task}") from exc


def require_finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{label} must be finite")
    return result


def load_freeze(path: Path) -> dict[str, Any]:
    freeze = read_json(path)
    if freeze.get("schema") != "lewm-pusht-action-error-coupling-posthoc-freeze":
        raise ValueError("unexpected analysis freeze schema")
    if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen_before_analysis":
        raise ValueError("analysis freeze version/status mismatch")
    if freeze.get("scope", {}).get("bank_split") != "train":
        raise ValueError("only the frozen train split may be analyzed")
    if freeze.get("scope", {}).get("reserved_holdout_or_validation_access") is not False:
        raise ValueError("freeze must explicitly forbid validation/holdout access")
    tasks = freeze.get("training_tasks")
    if not isinstance(tasks, list) or len(tasks) != 16:
        raise ValueError("freeze must contain exactly 16 training task identities")
    identities = [task_identity(task) for task in tasks]
    if len(set(identities)) != 16:
        raise ValueError("training task identities are duplicated")
    allowed = freeze.get("allowed_missing_solve", {})
    if allowed.get("episode_idx") != 15946 or allowed.get("replan_step") != 25:
        raise ValueError("the only allowed missing solve must be frozen episode 15946 at t25")
    if not any(identity[1] == int(allowed["episode_idx"]) for identity in identities):
        raise ValueError("the allowed missing-solve episode is not in the frozen train tasks")
    return freeze


def load_probe_file(path: Path, key: str) -> dict[int, dict[str, Any]]:
    expected = EXPECTED_PROBES[key]
    probe = read_json(path)
    if probe.get("schema") != expected["schema"] or probe.get("schema_version") != 1:
        raise ValueError(f"{key} probe schema/version mismatch")
    if probe.get("pbs_job_id") != expected["job_id"]:
        raise ValueError(f"{key} probe job identity mismatch")
    if probe.get("status") != "COMPLETED_ALL_REACHED_SOLVE_HORIZONS":
        raise ValueError(f"{key} probe is not complete")
    if probe.get("reserved_holdout_evaluated_or_scored") is not False:
        raise ValueError(f"{key} probe does not explicitly certify holdout isolation")
    episodes = probe.get("episodes")
    frozen_tasks = probe.get("tasks_frozen")
    if not isinstance(episodes, list) or len(episodes) != expected["count"]:
        raise ValueError(f"{key} probe episode count mismatch")
    if not isinstance(frozen_tasks, list) or len(frozen_tasks) != expected["count"]:
        raise ValueError(f"{key} frozen task list count mismatch")
    frozen_ids = {task_identity(task) for task in frozen_tasks}
    actual_ids: set[tuple[int, int, int, int]] = set()
    parsed: dict[int, dict[str, Any]] = {}
    for episode in episodes:
        identity = task_identity(episode.get("task"))
        if identity not in frozen_ids or identity in actual_ids:
            raise ValueError(f"{key} episode identity is absent/duplicated: {identity}")
        actual_ids.add(identity)
        episode_idx = identity[1]
        if episode_idx in parsed:
            raise ValueError(f"duplicate probe episode_idx {episode_idx}")
        if episode.get("solve_count") != len(episode.get("solves", [])):
            raise ValueError(f"{key} solve_count mismatch for episode {episode_idx}")
        expected_solve_count = 1 if episode_idx == 15946 else 2
        if episode.get("solve_count") != expected_solve_count:
            raise ValueError(f"unexpected solve count for frozen episode {episode_idx}")
        solves: dict[int, dict[str, float]] = {}
        for solve in episode.get("solves", []):
            step = int(solve.get("global_start_env_step", -1))
            if step not in (0, 25) or step in solves:
                raise ValueError(f"unexpected/duplicate solve step {step} for {episode_idx}")
            horizons = solve.get("horizons")
            if not isinstance(horizons, list) or len(horizons) != 5:
                raise ValueError(f"horizon rows are incomplete for episode {episode_idx}, t{step}")
            if [int(row.get("horizon_index_1based", -1)) for row in horizons] != [1, 2, 3, 4, 5]:
                raise ValueError(f"horizon order mismatch for episode {episode_idx}, t{step}")
            selected: dict[str, float] = {}
            for index in (0, 4):
                row = horizons[index]
                if row.get("available") is not True:
                    raise ValueError(f"h{index + 1} is unavailable for completed solve {episode_idx}, t{step}")
                for model in ("student", "teacher"):
                    metric = row.get(model)
                    if not isinstance(metric, dict) or "relative_mse" not in metric:
                        raise ValueError(f"missing {model} relative_mse for episode {episode_idx}, t{step}, h{index + 1}")
                    selected[f"{model}_h{index + 1}"] = require_finite(
                        metric["relative_mse"], f"{model} h{index + 1} relative_mse"
                    )
            solves[step] = selected

        if 0 not in solves:
            raise ValueError(f"every probe episode must have a t0 solve: {episode_idx}")
        if 25 not in solves:
            allowed = episode_idx == 15946 and episode.get("t25_status") == "terminal_at_t25_before_replan"
            terminal = episode.get("termination_at_final_step", {})
            if not allowed or terminal.get("terminated") is not True:
                raise ValueError(f"unexpected missing t25 solve for episode {episode_idx}")
        elif episode.get("t25_status") != "reached":
            raise ValueError(f"t25 solve/status mismatch for episode {episode_idx}")
        parsed[episode_idx] = {"identity": identity, "solves": solves}

    if actual_ids != frozen_ids:
        raise ValueError(f"{key} actual episodes do not exactly match tasks_frozen")
    return parsed


def load_probe_data(two_path: Path, fourteen_path: Path, freeze: dict[str, Any]) -> dict[int, dict[str, Any]]:
    two = load_probe_file(two_path, "two_task")
    fourteen = load_probe_file(fourteen_path, "fourteen_task")
    if set(two) & set(fourteen):
        raise ValueError("two-task and fourteen-task probe episode sets overlap")
    combined = {**two, **fourteen}
    expected = {task_identity(task) for task in freeze["training_tasks"]}
    actual = {row["identity"] for row in combined.values()}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"probe identities do not match frozen train tasks; missing={missing}; unexpected={unexpected}")
    if len(combined) != 16:
        raise ValueError("combined probe must contain exactly 16 unique train episodes")
    return combined


def validate_bank(record: dict[str, Any], expected_identities: set[tuple[int, int, int, int]], torch: Any) -> tuple[tuple[int, int, int, int], int, int]:
    if not isinstance(record, dict) or record.get("split") != "train":
        raise ValueError("only train bank records are allowed")
    identity = task_identity(record)
    if identity not in expected_identities:
        raise ValueError(f"bank contains a non-train or unknown task identity: {identity}")
    step = int(record.get("replan_step", -1))
    round_index = int(record.get("cem_round", -1))
    if step not in (0, 25) or round_index not in (10, 20, 30):
        raise ValueError(f"unexpected bank solve/round: t{step}, round {round_index}")
    if int(record.get("solver_seed", -1)) != 42:
        raise ValueError("bank solver seed is not the frozen seed 42")
    if "candidate_indices" not in record:
        raise ValueError("bank is missing native candidate indices")
    indices = torch.as_tensor(record["candidate_indices"], dtype=torch.int64, device="cpu").reshape(-1)
    if tuple(indices.shape) != (300,) or not torch.equal(indices, torch.arange(300)):
        raise ValueError("candidate indices are not native order 0..299")
    actions = torch.as_tensor(record.get("candidates"), dtype=torch.float64, device="cpu")
    student = torch.as_tensor(record.get("student_costs"), dtype=torch.float64, device="cpu")
    teacher = torch.as_tensor(record.get("teacher_costs"), dtype=torch.float64, device="cpu")
    if tuple(actions.shape) != (300, 5, 10):
        raise ValueError("candidate actions must have shape [300,5,10]")
    if tuple(student.shape) != (300,) or tuple(teacher.shape) != (300,):
        raise ValueError("student and teacher costs must each have shape [300]")
    for label, tensor in (("actions", actions), ("student costs", student), ("teacher costs", teacher)):
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"bank has non-finite {label}")
    if float(student.std(unbiased=False)) <= 1e-6 or float(teacher.std(unbiased=False)) <= 1e-8:
        raise ValueError("student/teacher cost population std is too small")
    if not bool(torch.isfinite(actions[:, 0, :].std(unbiased=False, dim=0)).all().item()):
        raise FloatingPointError("candidate first-action coordinate std is non-finite")
    return identity, step, round_index


def bank_metrics(record: dict[str, Any], torch: Any) -> dict[str, float]:
    actions = torch.as_tensor(record["candidates"], dtype=torch.float64, device="cpu")
    student = torch.as_tensor(record["student_costs"], dtype=torch.float64, device="cpu")
    teacher = torch.as_tensor(record["teacher_costs"], dtype=torch.float64, device="cpu")
    student_order = torch.argsort(student)
    teacher_order = torch.argsort(teacher)
    student_top30 = student_order[:30]
    teacher_top30 = teacher_order[:30]
    first_actions = actions[:, 0, :]
    student_mean = first_actions[student_top30].mean(dim=0)
    teacher_mean = first_actions[teacher_top30].mean(dim=0)
    displacement = student_mean - teacher_mean
    displacement_l2 = float(torch.linalg.vector_norm(displacement).item())
    candidate_coordinate_std = first_actions.std(unbiased=False, dim=0)
    scale_l2 = float(torch.linalg.vector_norm(candidate_coordinate_std).item())
    if not math.isfinite(scale_l2) or scale_l2 <= 1e-8:
        raise ValueError("candidate first-action coordinate std norm must exceed 1e-8")
    teacher_cost_std = float(teacher.std(unbiased=False).item())
    if not math.isfinite(teacher_cost_std) or teacher_cost_std <= 1e-8:
        raise ValueError("teacher objective population std must exceed 1e-8")
    regret = float(((teacher[student_top30].mean() - teacher[teacher_top30].mean()) / teacher_cost_std).item())
    values = {
        "first_action_elite_mean_delta_l2": displacement_l2,
        "first_action_elite_mean_delta_l2_over_candidate_std_l2": displacement_l2 / scale_l2,
        "candidate_first_action_coordinate_std_l2": scale_l2,
        "teacher_elite_regret_normalized": regret,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite bank metric")
    return values


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    if len(x_values) != len(y_values):
        raise ValueError("Spearman inputs must have equal lengths")
    n = len(x_values)
    result: dict[str, Any] = {"n_episodes": n, "rho": None, "status": "insufficient_episode_count"}
    if n < 3:
        return result
    x = average_ranks(x_values)
    y = average_ranks(y_values)
    mx = statistics.mean(x)
    my = statistics.mean(y)
    covariance = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denom = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    if denom <= 0.0:
        result["status"] = "constant_rank"
        return result
    result["rho"] = covariance / denom
    result["status"] = "descriptive_only"
    return result


def episode_bank_summary(banks: list[dict[str, Any]], torch: Any) -> dict[tuple[int, int], dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in banks:
        key = (int(record["episode_idx"]), int(record["replan_step"]))
        grouped[key].append({"round": int(record["cem_round"]), **bank_metrics(record, torch)})
    summaries: dict[tuple[int, int], dict[str, Any]] = {}
    metric_names = (
        "first_action_elite_mean_delta_l2",
        "first_action_elite_mean_delta_l2_over_candidate_std_l2",
        "candidate_first_action_coordinate_std_l2",
        "teacher_elite_regret_normalized",
    )
    for key, rows in grouped.items():
        rounds = sorted(row["round"] for row in rows)
        if rounds != [10, 20, 30]:
            raise ValueError(f"episode {key[0]} t{key[1]} bank rounds are incomplete: {rounds}")
        summaries[key] = {
            "rounds_aggregated": rounds,
            "bank_count": len(rows),
            **{name: statistics.median(row[name] for row in rows) for name in metric_names},
        }
    return summaries


def latent_metrics(solve: dict[str, float]) -> dict[str, float]:
    d1 = solve["student_h1"] - solve["teacher_h1"]
    d5 = solve["student_h5"] - solve["teacher_h5"]
    return {
        "h1_student_minus_teacher_relative_mse": d1,
        "h5_student_minus_teacher_relative_mse": d5,
        "h5_minus_h1_student_minus_teacher_relative_mse": d5 - d1,
    }


def build_episode_rows(
    identities: list[tuple[int, int, int, int]],
    probe_data: dict[int, dict[str, Any]],
    bank_summaries: dict[tuple[int, int], dict[str, Any]],
    freeze: dict[str, Any],
) -> list[dict[str, Any]]:
    missing = freeze["allowed_missing_solve"]
    rows: list[dict[str, Any]] = []
    for identity in sorted(identities, key=lambda item: item[0]):
        selection_order, episode_idx, row_index, start_step = identity
        probe = probe_data[episode_idx]
        for step in (0, 25):
            row: dict[str, Any] = {
                "selection_order": selection_order,
                "episode_idx": episode_idx,
                "row_index": row_index,
                "start_step": start_step,
                "replan_step": step,
            }
            solve = probe["solves"].get(step)
            summary = bank_summaries.get((episode_idx, step))
            if solve is None:
                if not (episode_idx == missing["episode_idx"] and step == missing["replan_step"]):
                    raise ValueError(f"unexpected absent solve for episode {episode_idx} t{step}")
                row.update({
                    "availability": "missing_expected_terminal_before_t25_replan",
                    "t25_status": missing["probe_t25_status"],
                    "latent": None,
                    "cem_round_medians": None,
                })
                if summary is not None:
                    raise ValueError("a bank exists for the episode whose t25 solve is absent")
            else:
                if summary is None:
                    raise ValueError(f"matching fullbank summary missing for episode {episode_idx} t{step}")
                row.update({
                    "availability": "available",
                    "latent": latent_metrics(solve),
                    "cem_round_medians": summary,
                })
            rows.append(row)
    return rows


def correlation_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    y_fields = (
        "first_action_elite_mean_delta_l2",
        "first_action_elite_mean_delta_l2_over_candidate_std_l2",
        "teacher_elite_regret_normalized",
    )
    x_field = "h5_minus_h1_student_minus_teacher_relative_mse"
    for step in (0, 25):
        group = [row for row in rows if row["replan_step"] == step and row["availability"] == "available"]
        output[f"t{step}"] = {}
        for y_field in y_fields:
            output[f"t{step}"][y_field + "_vs_latent_error_increment"] = spearman(
                [float(row["latent"][x_field]) for row in group],
                [float(row["cem_round_medians"][y_field]) for row in group],
            )
    return output


def main() -> None:
    args = parse_args()
    job_id, host = require_cpu_compute_node()

    # Import torch and read every experiment artifact only after the PBS guard.
    import torch

    torch.set_num_threads(2)
    freeze = load_freeze(args.freeze)
    task_rows = [task_identity(task) for task in freeze["training_tasks"]]
    expected_identities = set(task_rows)
    probe_data = load_probe_data(args.probe_two_task, args.probe_fourteen_task, freeze)

    try:
        banks = torch.load(args.train_banks, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older cluster PyTorch.
        banks = torch.load(args.train_banks, map_location="cpu")
    if not isinstance(banks, list) or len(banks) != int(freeze["scope"]["bank_count_expected"]):
        raise ValueError("train bank count/type does not match the frozen source")

    seen: set[tuple[tuple[int, int, int, int], int, int]] = set()
    for record in banks:
        identity, step, round_index = validate_bank(record, expected_identities, torch)
        key = (identity, step, round_index)
        if key in seen:
            raise ValueError(f"duplicate bank identity/step/round: {key}")
        seen.add(key)
    bank_keys = {(identity, step, round_index) for identity, step, round_index in seen}
    expected_keys = {
        (identity, step, round_index)
        for identity in expected_identities
        for step in (0, 25)
        for round_index in (10, 20, 30)
        if not (identity[1] == 15946 and step == 25)
    }
    if bank_keys != expected_keys:
        missing = sorted(expected_keys - bank_keys)
        unexpected = sorted(bank_keys - expected_keys)
        raise ValueError(f"train banks do not match frozen coverage; missing={missing}; unexpected={unexpected}")
    if len([key for key in bank_keys if key[1] == 0]) != 48 or len([key for key in bank_keys if key[1] == 25]) != 45:
        raise ValueError("t0/t25 bank coverage differs from the frozen 48/45 counts")

    bank_summaries = episode_bank_summary(banks, torch)
    episode_rows = build_episode_rows(task_rows, probe_data, bank_summaries, freeze)
    output = {
        "schema": "lewm-pusht-action-error-coupling-posthoc-result",
        "schema_version": 1,
        "status": "READ_ONLY_DESCRIPTIVE_POSTHOC_COMPLETE",
        "pbs_job_id": job_id,
        "compute_host": host,
        "source_jobs": freeze["source_jobs"],
        "source_split": "train_only_16_episodes",
        "source_bank_counts": {"t0": 48, "t25": 45, "total": len(banks)},
        "metric_definitions": freeze["metrics"],
        "missing_solve": {
            "episode_idx": 15946,
            "replan_step": 25,
            "reason": "terminal_at_t25_before_replan",
            "imputed": False,
        },
        "aggregation": freeze["metrics"]["episode_aggregation"],
        "correlation_method": freeze["metrics"]["correlation"],
        "episode_solve_rows": episode_rows,
        "descriptive_spearman": correlation_summaries(episode_rows),
        "claim_boundary": freeze["claim_boundary"],
        "validation_or_reserved_holdout_read": False,
        "training_or_cem_or_gpu_used": False,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote read-only episode-level diagnostic: {args.output}", flush=True)


if __name__ == "__main__":
    main()
