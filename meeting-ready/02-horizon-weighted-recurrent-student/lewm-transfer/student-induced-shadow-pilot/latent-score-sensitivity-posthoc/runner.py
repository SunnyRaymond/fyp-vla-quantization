#!/usr/bin/env python3
"""Describe per-horizon latent discrepancy and official-objective sensitivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_RUNNER_PATH = HERE.parent / "onpolicy-fullbank-ranker" / "partial-horizon-teacher-hybrid" / "runner.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_source_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("partial_horizon_teacher_hybrid_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_compute_node() -> str:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("runner requires PBS_JOBID and a real PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes:
        raise RuntimeError(f"host {host} is not present in PBS_NODEFILE")
    return host


def validate_freeze(freeze: dict[str, Any], source_freeze: dict[str, Any], source_freeze_path: Path) -> None:
    if freeze.get("schema") != "lewm-pusht-latent-score-sensitivity-posthoc-freeze":
        raise ValueError("freeze schema mismatch")
    if freeze.get("status") != "frozen_before_execution":
        raise ValueError("freeze is not frozen_before_execution")
    source = freeze["source"]
    if str(source_freeze_path.resolve()) != str(Path(source["source_freeze_path"]).resolve()):
        raise ValueError("source freeze path differs from the frozen partial-horizon runner")
    if source["collection_job"] != "25538135.pbs101" or int(source["train_bank_count"]) != 93:
        raise ValueError("train-bank provenance/count mismatch")
    for name in ("student_checkpoint_path", "teacher_checkpoint_path"):
        if str(Path(freeze["models"][name]).resolve()) != str(Path(source_freeze["models"][name]).resolve()):
            raise ValueError(f"{name} differs from the frozen partial-horizon model path")


def predict_both(base: Any, student: Any, teacher: Any, bank: dict[str, Any], torch: Any) -> tuple[Any, Any, Any, Any]:
    actions = bank["candidates"].to(device="cuda", dtype=torch.float32)
    initial = bank["initial_emb"].to(device="cuda", dtype=torch.float32).reshape(1, 1, 192)
    history = initial.expand(actions.shape[0], -1, -1)
    goal = bank["goal_emb"].to(device="cuda", dtype=torch.float32).reshape(1, 192)
    info = {"latent_history": initial, "goal_emb": goal}

    with torch.inference_mode():
        student_prediction = student(history, actions)
        teacher_prediction = base.official_teacher_targets(teacher, history, actions)
        student_costs = base._official_objective(teacher, info, student_prediction)
        teacher_costs = base._official_objective(teacher, info, teacher_prediction)

    expected_prediction = (300, 5, 192)
    if tuple(student_prediction.shape) != expected_prediction or tuple(teacher_prediction.shape) != expected_prediction:
        raise RuntimeError(f"prediction shape mismatch for bank {(bank['episode_idx'], bank['replan_step'], bank['cem_round'])}")
    if tuple(student_costs.shape) != (300,) or tuple(teacher_costs.shape) != (300,):
        raise RuntimeError("official objective returned an unexpected cost shape")
    for tensor in (student_prediction, teacher_prediction, student_costs, teacher_costs):
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError("non-finite model prediction or objective cost")
    return student_prediction, teacher_prediction, student_costs, teacher_costs


def relative_latent_mse(student_prediction: Any, teacher_prediction: Any, torch: Any) -> list[float]:
    values = []
    for horizon in range(5):
        target = teacher_prediction[:, horizon, :]
        mse = (student_prediction[:, horizon, :] - target).square().mean()
        denominator = target.square().mean().clamp_min(1e-8)
        values.append(float((mse / denominator).item()))
    return values


def bank_metrics(costs: Any, teacher_costs: Any, source_runner: Any, torch: Any) -> tuple[float, float]:
    return source_runner.bank_metrics(costs, teacher_costs, torch)


def bank_diagnostics(
    bank: dict[str, Any],
    student_prediction: Any,
    teacher_prediction: Any,
    student_costs: Any,
    teacher_costs: Any,
    source_runner: Any,
    base: Any,
    teacher: Any,
    torch: Any,
) -> dict[str, Any]:
    initial = bank["initial_emb"].to(device="cuda", dtype=torch.float32).reshape(1, 1, 192)
    goal = bank["goal_emb"].to(device="cuda", dtype=torch.float32).reshape(1, 192)
    info = {"latent_history": initial, "goal_emb": goal}
    student_regret, student_recall = bank_metrics(student_costs, teacher_costs, source_runner, torch)
    scale = teacher_costs.std(unbiased=False).clamp_min(1e-6)
    latent_errors = relative_latent_mse(student_prediction, teacher_prediction, torch)
    horizon_rows = []

    with torch.inference_mode():
        for horizon in range(5):
            mixed = student_prediction.clone()
            mixed[:, horizon, :] = teacher_prediction[:, horizon, :]
            counterfactual_costs = base._official_objective(teacher, info, mixed)
            if tuple(counterfactual_costs.shape) != (300,) or not bool(torch.isfinite(counterfactual_costs).all().item()):
                raise RuntimeError(f"invalid counterfactual objective at horizon {horizon + 1}")
            regret, recall = bank_metrics(counterfactual_costs, teacher_costs, source_runner, torch)
            score_delta = counterfactual_costs - student_costs
            horizon_rows.append(
                {
                    "horizon": horizon + 1,
                    "relative_latent_mse": latent_errors[horizon],
                    "regret_delta_vs_student": regret - student_regret,
                    "top30_recall_delta_vs_student": recall - student_recall,
                    "score_perturbation_rms_teacher_std": float(
                        (score_delta.square().mean().sqrt() / scale).item()
                    ),
                    "score_perturbation_mean_teacher_std": float((score_delta.mean() / scale).item()),
                }
            )

    return {
        "episode_idx": int(bank["episode_idx"]),
        "replan_step": int(bank["replan_step"]),
        "cem_round": int(bank["cem_round"]),
        "student_regret": student_regret,
        "student_top30_recall": student_recall,
        "horizons": horizon_rows,
    }


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def aggregate(rows: list[dict[str, Any]], episodes: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_step_round": [], "by_step": [], "per_episode": []}
    metrics = (
        "relative_latent_mse",
        "regret_delta_vs_student",
        "top30_recall_delta_vs_student",
        "score_perturbation_rms_teacher_std",
        "score_perturbation_mean_teacher_std",
    )
    def episode_values(step: int | None, round_number: int | None, horizon: int, metric: str) -> dict[int, float]:
        values: dict[int, list[float]] = defaultdict(list)
        for bank in rows:
            if step is not None and bank["replan_step"] != step:
                continue
            if round_number is not None and bank["cem_round"] != round_number:
                continue
            values[bank["episode_idx"]].append(bank["horizons"][horizon - 1][metric])
        return {episode: median(items) for episode, items in values.items()}

    for step in (0, 25):
        for round_number in (10, 20, 30):
            for horizon in range(1, 6):
                item: dict[str, Any] = {"replan_step": step, "cem_round": round_number, "horizon": horizon}
                for metric in metrics:
                    per_episode = episode_values(step, round_number, horizon, metric)
                    item[metric] = {
                        "n_episodes": len(per_episode),
                        "episode_median": median(list(per_episode.values())),
                        "episode_values": {str(key): value for key, value in sorted(per_episode.items())},
                    }
                result["by_step_round"].append(item)

        for horizon in range(1, 6):
            item = {"replan_step": step, "horizon": horizon}
            for metric in metrics:
                per_episode = episode_values(step, None, horizon, metric)
                item[metric] = {
                    "n_episodes": len(per_episode),
                    "episode_median": median(list(per_episode.values())),
                    "episode_values": {str(key): value for key, value in sorted(per_episode.items())},
                }
            result["by_step"].append(item)

    for episode in episodes:
        item: dict[str, Any] = {"episode_idx": episode, "replan_steps": {}}
        for step in (0, 25):
            banks = [bank for bank in rows if bank["episode_idx"] == episode and bank["replan_step"] == step]
            if not banks:
                item["replan_steps"][str(step)] = {"available": False}
                continue
            horizon_summary = []
            for horizon in range(1, 6):
                horizon_item = {"horizon": horizon}
                for metric in metrics:
                    horizon_item[metric] = median(
                        [bank["horizons"][horizon - 1][metric] for bank in banks]
                    )
                horizon_summary.append(horizon_item)
            item["replan_steps"][str(step)] = {
                "available": True,
                "bank_count": len(banks),
                "horizons": horizon_summary,
            }
        result["per_episode"].append(item)

    for step in (0, 25):
        comparison: dict[str, Any] = {"replan_step": step}
        available = [
            episode for episode in episodes
            if any(bank["episode_idx"] == episode and bank["replan_step"] == step for bank in rows)
        ]
        h5_gt_h1 = {}
        for episode in available:
            banks = [bank for bank in rows if bank["episode_idx"] == episode and bank["replan_step"] == step]
            h1 = median([bank["horizons"][0]["relative_latent_mse"] for bank in banks])
            h5 = median([bank["horizons"][4]["relative_latent_mse"] for bank in banks])
            h5_gt_h1[str(episode)] = h5 > h1
        comparison["episodes_with_h5_relative_latent_mse_gt_h1"] = sum(h5_gt_h1.values())
        comparison["episodes_available"] = len(available)
        comparison["episode_signs"] = h5_gt_h1
        result.setdefault("descriptive_readout", []).append(comparison)
    return result


def main() -> None:
    args = parse_args()
    # This guard precedes every project artifact read and all model/data loading.
    host = require_compute_node()
    source_runner = load_source_runner(SOURCE_RUNNER_PATH)

    freeze = read_json(args.freeze)
    source_freeze_path = Path(freeze["source"]["source_freeze_path"])
    source_freeze = read_json(source_freeze_path)
    validate_freeze(freeze, source_freeze, source_freeze_path)
    train_banks_path = Path(freeze["source"]["train_banks_path"])
    if str(train_banks_path.resolve()) != str(Path(source_freeze["source"]["train_banks_path"]).resolve()):
        raise ValueError("train bank path differs from the frozen 25538135 train-bank path")

    import torch

    banks = torch.load(train_banks_path, map_location="cpu", weights_only=False)
    source_runner.validate_banks(banks, source_freeze, torch)
    base, student, teacher = source_runner.load_models(source_freeze, torch)

    cached: list[tuple[dict[str, Any], Any, Any, Any, Any]] = []
    control_errors: list[dict[str, Any]] = []
    for bank in banks:
        student_prediction, teacher_prediction, student_costs, teacher_costs = predict_both(
            base, student, teacher, bank, torch
        )
        key = (int(bank["episode_idx"]), int(bank["replan_step"]), int(bank["cem_round"]))
        control_errors.append(
            source_runner.compare_costs("k0", student_costs, bank["student_costs"], source_freeze, torch, key)
        )
        control_errors.append(
            source_runner.compare_costs("k5", teacher_costs, bank["teacher_costs"], source_freeze, torch, key)
        )
        cached.append(
            (
                bank,
                student_prediction.detach().cpu(),
                teacher_prediction.detach().cpu(),
                student_costs.detach().cpu(),
                teacher_costs.detach().cpu(),
            )
        )

    if len(cached) != 93 or len(control_errors) != 186:
        raise RuntimeError("expected 93 aligned banks and 186 k0/k5 controls before scoring")

    rows = []
    for bank, student_prediction, teacher_prediction, student_costs, teacher_costs in cached:
        rows.append(
            bank_diagnostics(
                bank,
                student_prediction.to("cuda"),
                teacher_prediction.to("cuda"),
                student_costs.to("cuda"),
                teacher_costs.to("cuda"),
                source_runner,
                base,
                teacher,
                torch,
            )
        )

    episode_ids = [int(value) for value in freeze["source"]["train_episode_ids"]]
    output = {
        "schema": freeze["schema"] + ".result",
        "status": "DESCRIPTIVE_POSTHOC_COMPLETE",
        "pbs_job_id": os.environ.get("PBS_JOBID", ""),
        "compute_host": host,
        "source_collection_job": freeze["source"]["collection_job"],
        "source_partial_horizon_job": freeze["source"]["partial_horizon_pilot"],
        "bank_count": len(rows),
        "episode_count": len(episode_ids),
        "control_alignment": {
            "comparisons": len(control_errors),
            "allclose": all(item["allclose"] for item in control_errors),
            "max_abs_error": max(item["max_abs_error"] for item in control_errors),
            "rtol": source_freeze["validity"]["rtol"],
            "atol": source_freeze["validity"]["atol"],
        },
        "aggregation": "episode is the unit; banks are nested; no p-values",
        "claim_boundary": "Teacher predictions are a reference, not ground truth. Single-horizon swaps are score-sensitivity counterfactuals with physically inconsistent mixed latent sequences. No criterion-correctness, GO, CEM, planner, or closed-loop claim.",
        "bank_rows": rows,
        "summary": aggregate(rows, episode_ids),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "latent_score_sensitivity_summary.json"
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {target}; 93 banks; 186 cost controls; descriptive only")


if __name__ == "__main__":
    main()
