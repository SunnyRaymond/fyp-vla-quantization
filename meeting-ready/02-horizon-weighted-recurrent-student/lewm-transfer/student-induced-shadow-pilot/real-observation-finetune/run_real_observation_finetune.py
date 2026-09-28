#!/usr/bin/env python3
"""One frozen real-observation latent fine-tune and paired predictor gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TRANSFER_DIR = HERE.parents[1]
sys.path.insert(0, str(TRANSFER_DIR))

import run_lewm_recurrent_student as student_api  # noqa: E402


FREEZE_SCHEMA = "lewm-pusht-real-observation-student-finetune-freeze"
SUMMARY_SCHEMA = "lewm-pusht-student-driven-real-observation-training-collection-result-v1"
ARRAY_KEYS = (
    "episode_idx", "split", "solve_start_env_step", "z_start", "packed_actions",
    "poststep_latents", "max_plan_action_abs_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--collection-summary", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def require_compute_node() -> str:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required for checkpoint and NPZ access")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_NODEFILE is required for compute-allocation verification")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    if os.environ.get("PBS_NGPUS", "1").strip() in ("0", ""):
        raise RuntimeError("one allocated GPU is required")
    return host


def validate_protocol(freeze: dict[str, Any]) -> None:
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("schema_version") != 1:
        raise ValueError("fine-tune freeze schema mismatch")
    if freeze.get("status") != "LOCAL_FROZEN_PROTOCOL_NOT_SUBMITTED":
        raise ValueError("unexpected fine-tune protocol status")
    training = freeze["training"]
    if (int(training["updates"]), int(training["batch_episodes"])) != (500, 8):
        raise ValueError("frozen training budget drifted")
    if not math.isclose(float(training["learning_rate"]), 3e-5) or not math.isclose(float(training["weight_decay"]), 0.01):
        raise ValueError("frozen optimizer settings drifted")
    if freeze["scope"]["planner_ranking_training_or_evaluation"] is not False:
        raise ValueError("this runner does not support planner-ranking work")


def validate_inputs(
    freeze: dict[str, Any], summary: dict[str, Any], arrays: dict[str, Any], np: Any,
) -> tuple[list[int], dict[int, list[int]], dict[int, list[int]]]:
    source = freeze["source_data"]
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("schema_version") != 1:
        raise ValueError("collection summary schema mismatch")
    if summary.get("pbs_job_id") != source["collection_job_id"]:
        raise ValueError("collection summary is not from the frozen capture job")
    if summary.get("status") not in (
        "COMPLETE_WITH_ALL_TASKS_CAPTURED",
        "COMPLETE_WITH_TERMINAL_OR_BUDGET_LIMITED_WINDOWS",
    ):
        raise ValueError(f"collection is not complete: {summary.get('status')!r}")
    recovery = summary.get("recovery_provenance", {})
    if (recovery.get("source_job_id"), recovery.get("source_pbs_exit_status"),
            recovery.get("recovery_pbs_job_id"), recovery.get("original_summary_and_archive_modified")) != (
            source["collection_job_id"], 1, source["recovery_job_id"], False):
        raise ValueError("collection lacks the frozen independent CPU recovery proof")
    if summary.get("tasks_frozen") != 80 or summary.get("reserved_holdout_evaluated_or_scored") is not False:
        raise ValueError("collection task count or reserved-holdout boundary drifted")
    if summary.get("teacher_forecasts_costs_or_shadow_calls") is not False:
        raise ValueError("training capture unexpectedly used teacher forecasts/costs")

    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 80:
        raise ValueError("collection summary must contain all 80 frozen tasks")
    train_ids: list[int] = []
    validation_ids: list[int] = []
    expected_rows: set[tuple[int, str, int]] = set()
    for episode in episodes:
        task = episode["task"]
        episode_id = int(task["episode_idx"])
        split = str(task["split"])
        if split == "collection_train":
            train_ids.append(episode_id)
        elif split == "collection_validation":
            validation_ids.append(episode_id)
        else:
            raise ValueError(f"unexpected collection split: {split}")
        for window in episode.get("solve_windows", []):
            if bool(window.get("full_window_available")):
                expected_rows.add((episode_id, split, int(window["solve_start_env_step"])))
    if len(train_ids) != 64 or len(validation_ids) != 16 or len(set(train_ids)) != 64 or len(set(validation_ids)) != 16 or set(train_ids) & set(validation_ids):
        raise ValueError("collection episode split is not disjoint 64/16")

    count = len(arrays["episode_idx"])
    expected_shapes = {
        "episode_idx": (count,), "split": (count,), "solve_start_env_step": (count,),
        "z_start": (count, 192), "packed_actions": (count, 5, 10),
        "poststep_latents": (count, 5, 192), "max_plan_action_abs_error": (count,),
    }
    if count == 0 or any(tuple(arrays[key].shape) != shape for key, shape in expected_shapes.items()):
        raise ValueError("real-observation NPZ shapes differ from the frozen tuple schema")
    for key in ("z_start", "packed_actions", "poststep_latents", "max_plan_action_abs_error"):
        if not bool(np.isfinite(arrays[key]).all()):
            raise ValueError(f"non-finite values in archive field {key}")
    if bool((arrays["max_plan_action_abs_error"] > 1e-5).any()):
        raise ValueError("a captured action tuple failed the frozen plan/action alignment gate")
    actual_rows = [
        (int(arrays["episode_idx"][i]), str(arrays["split"][i]), int(arrays["solve_start_env_step"][i]))
        for i in range(count)
    ]
    if len(set(actual_rows)) != len(actual_rows) or set(actual_rows) != expected_rows:
        raise ValueError("NPZ tuples do not match complete solve windows in the collection summary")

    train_row_map = {episode_id: [] for episode_id in train_ids}
    validation_row_map = {episode_id: [] for episode_id in validation_ids}
    train_id_set = set(train_ids)
    for index, (episode_id, split, _start) in enumerate(actual_rows):
        expected_split = "collection_train" if episode_id in train_id_set else "collection_validation"
        if split != expected_split:
            raise ValueError("NPZ episode/split identity differs from the frozen summary")
        target = train_row_map if split == "collection_train" else validation_row_map
        target[episode_id].append(index)
    usable_train_ids = [episode_id for episode_id in train_ids if train_row_map[episode_id]]
    if len(usable_train_ids) < int(freeze["training"]["batch_episodes"]):
        raise ValueError("too few frozen train episodes have a complete five-token window")
    if sum(bool(rows) for rows in validation_row_map.values()) < 12:
        raise ValueError("fewer than 12 validation episodes have a complete window")
    if summary.get("sample_count") != count:
        raise ValueError("NPZ row count differs from collection summary")
    return usable_train_ids, train_row_map, validation_row_map


def load_initial_state(checkpoint: Path, freeze: dict[str, Any], torch: Any) -> dict[str, Any]:
    expected_path = Path(freeze["initial_checkpoint"]["path"]).resolve()
    if checkpoint.resolve() != expected_path:
        raise ValueError("checkpoint path differs from the frozen treatment_step1000 artifact")
    raw = torch.load(checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("state_dict"), dict):
        raise ValueError("frozen checkpoint must contain a state_dict")
    expected_provenance = {"source": "cem_distribution_distill", "arm": "treatment", "extra_updates": 1000}
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("checkpoint provenance differs from frozen treatment_step1000")
    return {str(key): tensor.detach().cpu().clone() for key, tensor in raw["state_dict"].items()}


def train(
    arrays: dict[str, Any], train_ids: list[int], train_rows: dict[int, list[int]],
    initial_state: dict[str, Any], freeze: dict[str, Any], output: Path,
    torch: Any, device: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    config = freeze["training"]
    randomizer = random.Random(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    student = student_api.make_student("baseline").to(device)
    student.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), betas=tuple(config["betas"]),
        eps=float(config["epsilon"]),
    )
    history: list[dict[str, Any]] = []
    student.train()
    for step in range(1, int(config["updates"]) + 1):
        selected_episodes = randomizer.sample(train_ids, int(config["batch_episodes"]))
        row_indices = [randomizer.choice(train_rows[episode_id]) for episode_id in selected_episodes]
        context = torch.as_tensor(arrays["z_start"][row_indices], dtype=torch.float32, device=device).unsqueeze(1)
        actions = torch.as_tensor(arrays["packed_actions"][row_indices], dtype=torch.float32, device=device)
        targets = torch.as_tensor(arrays["poststep_latents"][row_indices], dtype=torch.float32, device=device)
        prediction = student(context, actions)
        loss, per_horizon = student_api.recurrent_loss(prediction, targets)
        if not bool(torch.isfinite(loss).item()) or not bool(torch.isfinite(per_horizon).all().item()):
            raise FloatingPointError(f"non-finite real-observation training loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({
            "step": step,
            "weighted_mse": float(loss.detach().cpu()),
            "per_horizon_mse": [float(value) for value in per_horizon.detach().cpu()],
        })
        if step % 100 == 0:
            print(f"training_update={step} weighted_real_mse={history[-1]['weighted_mse']:.8g}", flush=True)

    student.eval()
    checkpoint_path = output / "real_observation_student_step0500.pt"
    temporary_checkpoint = output / "real_observation_student_step0500.pt.tmp"
    torch.save({
        "state_dict": {key: value.detach().cpu().clone() for key, value in student.state_dict().items()},
        "provenance": {
            "source": "real_observation_student_finetune",
            "freeze_path": str(freeze["_path"]),
            "initial_checkpoint": str(freeze["initial_checkpoint"]["path"]),
            "collection_job_id": freeze["source_data"]["collection_job_id"],
            "updates": int(config["updates"]),
            "seed": int(config["seed"]),
            "objective": config["objective"],
        },
    }, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    return student, history


def evaluate(
    arrays: dict[str, Any], validation_rows: dict[int, list[int]], initial_state: dict[str, Any],
    terminal_student: Any, torch: Any, device: Any,
) -> dict[str, Any]:
    initial_student = student_api.make_student("baseline").to(device)
    initial_student.load_state_dict(initial_state, strict=True)
    initial_student.eval()
    terminal_student.eval()
    per_episode: dict[int, dict[str, list[dict[str, float]]]] = {
        episode_id: {"all": [], "t0": [], "t25": []} for episode_id in validation_rows
    }
    with torch.no_grad():
        for episode_id, row_indices in validation_rows.items():
            for index in row_indices:
                context = torch.as_tensor(arrays["z_start"][index:index + 1], dtype=torch.float32, device=device).unsqueeze(1)
                actions = torch.as_tensor(arrays["packed_actions"][index:index + 1], dtype=torch.float32, device=device)
                targets = torch.as_tensor(arrays["poststep_latents"][index:index + 1], dtype=torch.float32, device=device)
                base_prediction = initial_student(context, actions)
                terminal_prediction = terminal_student(context, actions)
                if not bool(torch.isfinite(base_prediction).all().item() and torch.isfinite(terminal_prediction).all().item()):
                    raise FloatingPointError("non-finite validation prediction")
                denominator = targets.square().mean().clamp_min(1e-8)
                base_mse = float(((base_prediction - targets).square().mean() / denominator).cpu())
                terminal_mse = float(((terminal_prediction - targets).square().mean() / denominator).cpu())
                if not math.isfinite(base_mse) or not math.isfinite(terminal_mse) or base_mse <= 0:
                    raise FloatingPointError("invalid relative MSE in validation")
                record = {
                    "initial_relative_mse": base_mse,
                    "terminal_relative_mse": terminal_mse,
                    "improvement": 1.0 - terminal_mse / base_mse,
                }
                per_episode[episode_id]["all"].append(record)
                start = int(arrays["solve_start_env_step"][index])
                label = "t0" if start == 0 else "t25" if start == 25 else None
                if label is None:
                    raise ValueError(f"unexpected validation solve start: {start}")
                per_episode[episode_id][label].append(record)

    def episode_improvements(group: str) -> dict[int, float]:
        return {
            episode_id: statistics.mean(row["improvement"] for row in values[group])
            for episode_id, values in per_episode.items() if values[group]
        }

    overall = episode_improvements("all")
    t0 = episode_improvements("t0")
    t25 = episode_improvements("t25")
    return {
        "per_episode": per_episode,
        "overall": {
            "coverage_episodes": len(overall),
            "median_episode_improvement": statistics.median(overall.values()) if overall else None,
            "strictly_improved_episodes": sum(value > 0 for value in overall.values()),
        },
        "t0": {
            "coverage_episodes": len(t0),
            "median_episode_improvement": statistics.median(t0.values()) if t0 else None,
        },
        "t25": {
            "coverage_episodes": len(t25),
            "median_episode_improvement": statistics.median(t25.values()) if t25 else None,
        },
    }


def decide_gate(metrics: dict[str, Any], validation_episode_count: int, freeze: dict[str, Any]) -> dict[str, Any]:
    conditions = freeze["predictor_gate"]["pass_conditions"]
    checks = {
        "minimum_episodes_with_any_complete_window": metrics["overall"]["coverage_episodes"] >= int(conditions["minimum_episodes_with_any_complete_window"]),
        "median_episode_improvement": (
            metrics["overall"]["median_episode_improvement"] is not None
            and metrics["overall"]["median_episode_improvement"] >= float(conditions["median_episode_improvement_min"])
        ),
        "strictly_improved_episodes": metrics["overall"]["strictly_improved_episodes"] >= int(conditions["episodes_strictly_improved_min"]),
        "minimum_t0_episode_coverage": metrics["t0"]["coverage_episodes"] >= int(conditions["minimum_t0_episode_coverage"]),
        "t0_median_episode_improvement": (
            metrics["t0"]["median_episode_improvement"] is not None
            and metrics["t0"]["median_episode_improvement"] > float(conditions["t0_median_episode_improvement_min_exclusive"])
        ),
        "minimum_t25_episode_coverage": metrics["t25"]["coverage_episodes"] >= int(conditions["minimum_t25_episode_coverage"]),
        "t25_median_episode_improvement": (
            metrics["t25"]["median_episode_improvement"] is not None
            and metrics["t25"]["median_episode_improvement"] > float(conditions["t25_median_episode_improvement_min_exclusive"])
        ),
        "all_16_validation_episodes_declared": validation_episode_count == 16,
        "all_outputs_finite": True,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "NO-GO",
        "checks": checks,
        "all_pass": passed,
        "planner_ranking": "NOT_RUN_BY_THIS_JOB",
        "official_cem": "NOT_RUN_BY_THIS_JOB",
    }


def main() -> int:
    args = parse_args()
    host = require_compute_node()
    freeze = read_json(args.freeze.resolve(strict=True))
    freeze["_path"] = str(args.freeze.resolve())
    validate_protocol(freeze)
    if args.archive.resolve() != Path(freeze["source_data"]["archive_path"]).resolve():
        raise ValueError("archive path differs from the frozen collection output")
    if args.collection_summary.resolve() != Path(freeze["source_data"]["summary_path"]).resolve():
        raise ValueError("collection-summary path differs from the frozen collection output")
    if args.output.name != os.environ["PBS_JOBID"] or args.output.parent.name != "real-observation-finetune":
        raise ValueError("output must be artifacts/.../real-observation-finetune/<PBS_JOBID>")
    protected_outputs = ("real_observation_student_step0500.pt", "result.json")
    if any((args.output / name).exists() for name in protected_outputs):
        raise FileExistsError(f"refusing to overwrite a checkpoint or result under {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the requested PBS GPU allocation")
    summary = read_json(args.collection_summary.resolve(strict=True))
    with np.load(args.archive.resolve(strict=True), allow_pickle=False) as archive:
        missing = sorted(set(ARRAY_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"training archive is missing fields: {missing}")
        arrays = {key: archive[key] for key in ARRAY_KEYS}
    train_ids, train_rows, validation_rows = validate_inputs(freeze, summary, arrays, np)
    initial_state = load_initial_state(args.initial_checkpoint, freeze, torch)

    started = time.time()
    terminal_student, history = train(
        arrays, train_ids, train_rows, initial_state, freeze, args.output, torch, torch.device("cuda")
    )
    # The validation split is first used after the fixed training budget and terminal save.
    metrics = evaluate(arrays, validation_rows, initial_state, terminal_student, torch, torch.device("cuda"))
    gate = decide_gate(metrics, len(validation_rows), freeze)
    result = {
        "schema": "lewm-pusht-real-observation-student-finetune-result-v1",
        "schema_version": 1,
        "status": "COMPLETE_PREDICTOR_MSE_GO" if gate["all_pass"] else "COMPLETE_PREDICTOR_MSE_NO_GO",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "freeze": str(args.freeze.resolve()),
        "collection_summary": str(args.collection_summary.resolve()),
        "archive": str(args.archive.resolve()),
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "terminal_checkpoint": str((args.output / "real_observation_student_step0500.pt").resolve()),
        "train_episodes_selected": 64,
        "train_episodes_usable": len(train_ids),
        "train_windows": sum(len(rows) for rows in train_rows.values()),
        "validation_episodes": len(validation_rows),
        "validation_windows": sum(len(rows) for rows in validation_rows.values()),
        "training": {
            "updates": len(history),
            "first_weighted_mse": history[0]["weighted_mse"],
            "terminal_weighted_mse": history[-1]["weighted_mse"],
            "last10_weighted_mse_median": statistics.median(row["weighted_mse"] for row in history[-10:]),
            "history": history,
        },
        "predictor_validation": metrics,
        "gate": gate,
        "scope": {
            "teacher_forecast_or_scoring": False,
            "planner_ranking": False,
            "cem_proposal_or_deployment": False,
            "closed_loop_treatment_evaluation": False,
            "untouched_reserved_8_read": False,
            "next_action": "Only a separate frozen planner-ranking gate may be considered after a predictor gate PASS.",
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output / "result.json", result)
    print(json.dumps({"status": result["status"], "gate": gate, "result": str(args.output / "result.json")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
