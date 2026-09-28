"""Frozen first-round runner for block-local action-conditioned dynamics."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import itertools
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_json(path: Path, value: Any, compact: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    options = {"separators": (",", ":")} if compact else {"indent": 2}
    tmp.write_text(json.dumps(value, sort_keys=True, **options), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_checkpoint(torch: Any, path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen FREEZE.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--contrast-name",
        help="Required label when selecting a separate, explicitly identified contrast.",
    )
    args = parser.parse_args()
    if any(getattr(args, key) is not None for key in ("conditions", "arms", "seeds")) and not args.contrast_name:
        parser.error("--conditions/--arms/--seeds subsets require --contrast-name")
    return args


def _select(values: list[Any], requested: list[Any] | None, name: str) -> list[Any]:
    if requested is None:
        return list(values)
    unknown = sorted(set(requested) - set(values))
    if unknown:
        raise ValueError(f"unknown {name}: {unknown}; frozen choices are {values}")
    chosen = list(dict.fromkeys(requested))
    if not chosen:
        raise ValueError(f"{name} selection cannot be empty")
    return chosen


def _bootstrap_contrast(
    left: np.ndarray,
    right: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    paired = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, paired.size, size=(1000, paired.size))
    bootstrap_means = paired[indices].mean(axis=1)
    return {
        "n_episodes": int(paired.size),
        "difference_definition": "left_minus_right; negative favors left",
        "mean_paired_difference": float(paired.mean()),
        "median_paired_difference": float(np.median(paired)),
        "bootstrap_95pct_mean_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "bootstrap_draws": 1000,
        "bootstrap_seed": int(seed),
        "inference_scope": "pilot_episode_inference_within_training_seed; not training_seed_population",
    }


def _sync(torch: Any) -> None:
    torch.cuda.synchronize()


def _train_one(
    torch: Any,
    nn: Any,
    config: dict[str, Any],
    model: Any,
    train_data: dict[str, Any],
    delta_energy: float,
    training_seed: int,
    log_writer: csv.DictWriter,
    log_file: Any,
    identity: dict[str, Any],
) -> tuple[Any, float, int]:
    training = config["training"]
    device = train_data["z"].device
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    z_all, actions = train_data["z"], train_data["action"]
    episodes = z_all.shape[0]
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    horizon = int(training["rollout_horizon"])
    generator = torch.Generator(device="cpu").manual_seed(int(training_seed))
    offset = torch.arange(horizon, device=device)
    normalizer = max(float(delta_energy), 1e-6)
    log_every = 10
    _sync(torch)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        episode_index = torch.randint(episodes, (batch_size,), generator=generator).to(device)
        time_index = torch.randint(
            int(config["data"]["trajectory_steps"]) - horizon + 1,
            (batch_size,),
            generator=generator,
        ).to(device)
        z0 = z_all[episode_index, time_index]
        action_seq = actions[episode_index[:, None], time_index[:, None] + offset[None, :]]
        target_seq = z_all[episode_index[:, None], time_index[:, None] + offset[None, :] + 1]

        optimizer.zero_grad(set_to_none=True)
        one_step_pred = model.step(z0, action_seq[:, 0])
        one_step_mse = nn.functional.mse_loss(one_step_pred, target_seq[:, 0])
        free_pred = model.rollout(z0, action_seq)
        free_mse = nn.functional.mse_loss(free_pred, target_seq)
        loss = (one_step_mse + 0.5 * free_mse) / normalizer
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at step {step}: {identity}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
        optimizer.step()
        if step % log_every == 0 or step == steps:
            log_writer.writerow(
                {
                    **identity,
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "one_step_mse": float(one_step_mse.detach().item()),
                    "free_running_h1_to_5_mse": float(free_mse.detach().item()),
                }
            )
            log_file.flush()
    _sync(torch)
    seconds = time.perf_counter() - started
    peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return optimizer, seconds, peak_mib


def _true_rollout_final(system: Any, z0: Any, actions: Any) -> Any:
    state = z0
    for index in range(actions.shape[1]):
        state = system.step_z(state, actions[:, index])
    return state


def _evaluate_rollouts(
    torch: Any,
    model: Any,
    split_data: dict[str, Any],
    horizons: list[int],
    offsets: list[int],
    trajectory_steps: int,
    normalizer: float,
    batch_size: int,
) -> np.ndarray:
    z, actions = split_data["z"], split_data["action"]
    episode_count = int(z.shape[0])
    errors = np.full((episode_count, len(horizons), len(offsets)), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for hi, horizon in enumerate(horizons):
            for oi, start in enumerate(offsets):
                if start + horizon > trajectory_steps:
                    continue
                for begin in range(0, episode_count, batch_size):
                    end = min(begin + batch_size, episode_count)
                    z0 = z[begin:end, start]
                    action_seq = actions[begin:end, start : start + horizon]
                    prediction = model.rollout_final(z0, action_seq)
                    truth = z[begin:end, start + horizon]
                    episode_mse = (prediction - truth).square().mean(dim=-1) / normalizer
                    errors[begin:end, hi, oi] = episode_mse.detach().cpu().numpy()
    return errors


def _evaluate_action_response(
    np_module: Any,
    torch: Any,
    model: Any,
    system: Any,
    split_data: dict[str, Any],
    split_seed: int,
    horizon: int,
    pair_count: int,
    delta_energy: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    z, actions = split_data["z"], split_data["action"]
    episode_count = int(z.shape[0])
    base = actions[:, :horizon].clamp(-0.9, 0.9)
    directions_np = np_module.random.default_rng(int(split_seed) + 92026).choice(
        np_module.array([-1.0, 1.0], dtype=np_module.float32),
        size=(episode_count, pair_count, horizon, int(actions.shape[-1])),
    )
    directions = torch.as_tensor(directions_np, dtype=actions.dtype, device=actions.device)
    base = base[:, None].expand(-1, pair_count, -1, -1)
    plus_actions = (base + 0.1 * directions).reshape(episode_count * pair_count, horizon, -1)
    minus_actions = (base - 0.1 * directions).reshape(episode_count * pair_count, horizon, -1)
    initial = z[:, 0, None, :].expand(-1, pair_count, -1).reshape(episode_count * pair_count, -1)
    with torch.inference_mode():
        plus_pred = model.rollout_final(initial, plus_actions)
        minus_pred = model.rollout_final(initial, minus_actions)
        predicted_response = plus_pred - minus_pred
        plus_true = _true_rollout_final(system, initial, plus_actions)
        minus_true = _true_rollout_final(system, initial, minus_actions)
        true_response = plus_true - minus_true
        pair_mse = (predicted_response - true_response).square().mean(dim=-1)
        true_response_energy = true_response.square().mean(dim=-1)
    errors = pair_mse.reshape(episode_count, pair_count).detach().cpu().numpy()
    true_energy = true_response_energy.detach().mean().item()
    return (
        (errors / max(float(delta_energy), 1e-6)).astype(np.float32),
        (errors / max(float(true_energy), 1e-6)).astype(np.float32),
        float(true_energy),
    )


def _benchmark(
    torch: Any,
    model: Any,
    split_data: dict[str, Any],
    batch_sizes: list[int],
    horizon: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    z, actions = split_data["z"], split_data["action"]
    episode_count = int(z.shape[0])
    output: dict[str, Any] = {}
    model.prepare_inference()
    with torch.inference_mode():
        for batch_size in batch_sizes:
            index = torch.arange(batch_size, device=z.device) % episode_count
            z0 = z[index, 0]
            action_seq = actions[index, :horizon]
            for name, call in (
                ("complete_rollout_final", model.rollout_final),
                ("each_step_reconstruction", model.rollout),
            ):
                for _ in range(warmup):
                    call(z0, action_seq)
                _sync(torch)
                timings = []
                for _ in range(repeats):
                    _sync(torch)
                    started = time.perf_counter()
                    call(z0, action_seq)
                    _sync(torch)
                    timings.append((time.perf_counter() - started) * 1000.0)
                output[f"{name}_batch{batch_size}_ms"] = {
                    "median": float(np.median(timings)),
                    "mean": float(np.mean(timings)),
                    "p90": float(np.quantile(timings, 0.90)),
                    "warmup": int(warmup),
                    "repeats": int(repeats),
                }
    return output


def _run_one(
    torch: Any,
    nn: Any,
    config: dict[str, Any],
    condition: str,
    arm: str,
    seed: int,
    system: Any,
    datasets: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
    log_writer: csv.DictWriter,
    log_file: Any,
    schedule_index: int,
    total_runs: int,
    schedule_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from models import build_model, matched_dense_hidden

    mean = metadata["mean"]
    dense_hidden = matched_dense_hidden(config, mean) if arm == "dense" else None
    model = build_model(arm, config, seed, mean, dense_hidden=dense_hidden).to("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    identity = {"condition": condition, "arm": arm, "training_seed": int(seed)}
    run_dir = output / condition / arm / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    optimizer, train_seconds, peak_memory_mib = _train_one(
        torch,
        nn,
        config,
        model,
        datasets["train"],
        float(metadata["delta_energy"]),
        seed,
        log_writer,
        log_file,
        identity,
    )
    _sync(torch)
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "condition": condition,
        "arm": arm,
        "training_seed": int(seed),
        "steps": int(config["training"]["steps"]),
        "checkpoint_selection": "last_step",
        "dense_hidden": dense_hidden,
        "parameter_count": int(parameter_count),
    }
    _atomic_checkpoint(torch, run_dir / "checkpoint.pt", checkpoint)

    model.eval()
    model.prepare_inference()
    evaluation = config["evaluation"]
    normalizer = max(float(metadata["delta_energy"]), 1e-6)
    episodes: dict[str, np.ndarray] = {}
    split_summaries: dict[str, Any] = {}
    eval_batch = int(config["training"]["batch_size"])
    for split in ("dev", "test"):
        terminal_error = _evaluate_rollouts(
            torch,
            model,
            datasets[split],
            [int(x) for x in evaluation["horizons"]],
            [int(x) for x in evaluation["episode_start_offsets"]],
            int(config["data"]["trajectory_steps"]),
            normalizer,
            eval_batch,
        )
        response_delta, response_relative, true_energy = _evaluate_action_response(
            np,
            torch,
            model,
            system,
            datasets[split],
            int(config["data"][f"{split}_seed"]),
            int(evaluation["action_response_design"]["horizon"]),
            int(evaluation["action_response_pairs_per_episode"]),
            normalizer,
        )
        episodes[f"{split}_terminal_error"] = terminal_error
        episodes[f"{split}_action_response_delta_normalized"] = response_delta
        episodes[f"{split}_action_response_true_energy_normalized"] = response_relative
        _atomic_npz(
            run_dir / f"{split}_per_episode.npz",
            terminal_error=terminal_error,
            action_response_delta_normalized=response_delta,
            action_response_true_energy_normalized=response_relative,
            true_response_energy=np.asarray(true_energy, dtype=np.float32),
        )
        episode_by_horizon = {
            int(horizon): np.nanmean(terminal_error[:, horizon_index, :], axis=1)
            for horizon_index, horizon in enumerate(evaluation["horizons"])
        }
        split_summaries[split] = {
            "primary_horizon": int(evaluation["primary_horizon"]),
            "horizon_episode_aggregates": {
                str(horizon): {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.90)),
                }
                for horizon, values in episode_by_horizon.items()
            },
            "primary_episode_mean": float(np.mean(episode_by_horizon[int(evaluation["primary_horizon"])])),
            "primary_episode_median": float(np.median(episode_by_horizon[int(evaluation["primary_horizon"])])),
            "primary_episode_p90": float(np.quantile(episode_by_horizon[int(evaluation["primary_horizon"])], 0.90)),
            "action_response_delta_normalized_mean": float(np.mean(response_delta)),
            "action_response_true_energy_normalized_mean": float(np.mean(response_relative)),
            "mean_true_response_energy": true_energy,
        }

    q_alignment = None
    if hasattr(model, "orthogonal_matrix"):
        with torch.inference_mode():
            q = model.orthogonal_matrix()
            known = system.M
            block_dim = int(config["dimensions"]["block_state"])
            blocks = int(config["dimensions"]["blocks"])
            overlap = torch.empty((blocks, blocks), dtype=q.dtype, device=q.device)
            for learned_block in range(blocks):
                q_block = q[:, learned_block * block_dim : (learned_block + 1) * block_dim]
                for true_block in range(blocks):
                    true_basis = known[:, true_block * block_dim : (true_block + 1) * block_dim]
                    overlap[learned_block, true_block] = (q_block.T @ true_basis).square().sum() / block_dim
            overlap_values = overlap.cpu().numpy()
            best_permutation = max(
                itertools.permutations(range(blocks)),
                key=lambda permutation: sum(float(overlap_values[i, permutation[i]]) for i in range(blocks)),
            )
            q_alignment = {
                "learned_to_known_block_subspace_overlap": overlap_values.tolist(),
                "mean_diagonal_overlap": float(np.diag(overlap_values).mean()),
                "best_known_block_assignment": list(best_permutation),
                "mean_overlap_after_best_block_permutation": float(
                    sum(float(overlap_values[i, best_permutation[i]]) for i in range(blocks)) / blocks
                ),
                "interpretation": "descriptive_coordinate_alignment_only",
            }

    latency = _benchmark(
        torch,
        model,
        datasets["test"],
        [int(x) for x in evaluation["latency_batch_sizes"]],
        int(evaluation["latency_horizon"]),
        int(evaluation["latency_warmup"]),
        int(evaluation["latency_repeats"]),
    )
    model.clear_inference_cache()
    record = {
        **identity,
        "schedule_index": int(schedule_index),
        "schedule_total": int(total_runs),
        "schedule_seed": int(schedule_seed),
        "dense_hidden": dense_hidden,
        "parameter_count": int(parameter_count),
        "train_delta_energy": normalizer,
        "data_identity": {
            "system_seed": int(metadata["system_seed"]),
            "split_seeds": metadata["split_seeds"],
            "episode_counts": metadata["episode_counts"],
        },
        "train_seconds": float(train_seconds),
        "peak_memory_mib": int(peak_memory_mib),
        "q_alignment": q_alignment,
        "dev": split_summaries["dev"],
        "test": split_summaries["test"],
        "latency": latency,
        "checkpoint": str((run_dir / "checkpoint.pt").relative_to(output)),
        "episode_files": {
            split: str((run_dir / f"{split}_per_episode.npz").relative_to(output))
            for split in ("dev", "test")
        },
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return record, episodes


def main() -> None:
    args = _parse_args()
    # The guard must run before importing tensor/model code or creating workload data.
    from allocation_guard import ensure_allocation

    allocation = ensure_allocation(require_gpu=True)
    import torch
    from torch import nn

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    conditions = _select(config["conditions"], args.conditions, "conditions")
    arms = _select(config["arms"], args.arms, "arms")
    seeds = _select(config["training"]["seeds"], args.seeds, "seeds")
    is_additional = bool(args.contrast_name)
    output = args.output.resolve()
    run_traces = ("training.tsv", "summary_progress.json", "summary.json", "DONE.json")
    if output.exists() and any((output / name).exists() for name in run_traces):
        raise FileExistsError(f"output already contains runner traces: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "FREEZE.json")

    from controlled_system import make_datasets

    schedule_seed = int(config["execution_order"]["shuffle_seed"])
    schedule = [
        {"condition": condition, "arm": arm, "training_seed": int(seed)}
        for condition in conditions
        for arm in arms
        for seed in seeds
    ]
    random.Random(schedule_seed).shuffle(schedule)

    cached: dict[str, tuple[Any, dict[str, dict[str, Any]], dict[str, Any]]] = {}
    for condition in conditions:
        cached[condition] = make_datasets(config, condition, torch.device("cuda"))

    log_path = output / "training.tsv"
    log_fields = ["condition", "arm", "training_seed", "step", "loss", "one_step_mse", "free_running_h1_to_5_mse"]
    completed: list[dict[str, Any]] = []
    episode_metrics: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    progress: dict[str, Any] = {
        "protocol_id": config["protocol_id"],
        "scope": "additional_contrast" if is_additional else "frozen_primary",
        "contrast_name": args.contrast_name,
        "allocation": allocation,
        "schedule_seed": schedule_seed,
        "schedule": schedule,
        "expected_runs": len(schedule),
        "completed_runs": completed,
    }
    _atomic_json(output / "summary_progress.json", progress)
    _atomic_json(
        output / "progress.json",
        {"status": "ready", "expected_runs": len(schedule), "completed_runs": 0, "current_run": None},
    )

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields, delimiter="\t")
        writer.writeheader()
        log_file.flush()
        for index, item in enumerate(schedule, start=1):
            condition, arm, seed = item["condition"], item["arm"], int(item["training_seed"])
            _atomic_json(
                output / "progress.json",
                {
                    "status": "running",
                    "expected_runs": len(schedule),
                    "completed_runs": len(completed),
                    "schedule_index": index,
                    "current_run": {"condition": condition, "arm": arm, "training_seed": seed},
                },
            )
            system, datasets, metadata = cached[condition]
            record, episodes = _run_one(
                torch,
                nn,
                config,
                condition,
                arm,
                seed,
                system,
                datasets,
                metadata,
                output,
                writer,
                log_file,
                index,
                len(schedule),
                schedule_seed,
            )
            completed.append(record)
            episode_metrics[(condition, arm, seed)] = episodes
            progress["completed_runs"] = completed
            _atomic_json(output / "summary_progress.json", progress)
            _atomic_json(
                output / "progress.json",
                {
                    "status": "running",
                    "expected_runs": len(schedule),
                    "completed_runs": len(completed),
                    "schedule_index": index,
                    "current_run": None,
                    "last_completed_run": {"condition": condition, "arm": arm, "training_seed": seed},
                },
            )

    contrasts: list[dict[str, Any]] = []
    paired_arrays: dict[str, np.ndarray] = {}
    contrast_specs = (
        ("learned_global4", "learned_local4"),
        ("learned_global4", "learned_block"),
        ("learned_block", "random_block"),
    )
    selected_arm_set = set(arms)
    for condition in conditions:
        for seed in seeds:
            for left_arm, right_arm in contrast_specs:
                if left_arm not in selected_arm_set or right_arm not in selected_arm_set:
                    continue
                left_key, right_key = (condition, left_arm, int(seed)), (condition, right_arm, int(seed))
                if left_key not in episode_metrics or right_key not in episode_metrics:
                    continue
                left = np.nanmean(episode_metrics[left_key]["test_terminal_error"][:, [int(x) for x in config["evaluation"]["horizons"]].index(int(config["evaluation"]["primary_horizon"])), :], axis=1)
                right = np.nanmean(episode_metrics[right_key]["test_terminal_error"][:, [int(x) for x in config["evaluation"]["horizons"]].index(int(config["evaluation"]["primary_horizon"])), :], axis=1)
                boot_seed = 20260926 + 10000 * conditions.index(condition) + 100 * int(seed) + 17 * contrast_specs.index((left_arm, right_arm))
                contrast = _bootstrap_contrast(left, right, boot_seed)
                contrast_key = f"{condition}__seed-{seed}__{left_arm}-minus-{right_arm}"
                paired_arrays[contrast_key] = (left - right).astype(np.float32)
                contrast.update(
                    {
                        "episode_array_key": contrast_key,
                        "condition": condition,
                        "training_seed": int(seed),
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "metric": "test_episode_aggregated_normalized_rollout_mse_h10",
                    }
                )
                contrasts.append(contrast)

    by_identity = {(row["condition"], row["arm"], row["training_seed"]): row for row in completed}
    if "dense" in selected_arm_set:
        for row in completed:
            dense = by_identity.get((row["condition"], "dense", row["training_seed"]))
            if dense:
                reference = max(float(dense["test"]["primary_episode_mean"]), float(config["gates"]["reference_error_floor"]))
                tolerance = max(
                    float(config["gates"]["relative_quality_increase_max"]) * reference,
                    float(config["gates"]["absolute_normalized_error_tolerance"]),
                )
                row["test"]["dense_reference_error"] = reference
                row["test"]["dense_quality_tolerance"] = tolerance
                row["test"]["dense_quality_gate_pass"] = (
                    float(row["test"]["primary_episode_mean"]) <= reference + tolerance
                )
                row["dense_parameter_match_relative_error_vs_learned_global4"] = None
                row["dense_parameter_match_gate_pass"] = None
                global4 = by_identity.get((row["condition"], "learned_global4", row["training_seed"]))
                if global4:
                    row["dense_parameter_match_relative_error_vs_learned_global4"] = abs(
                        int(dense["parameter_count"]) - int(global4["parameter_count"])
                    ) / max(1, int(global4["parameter_count"]))
                    row["dense_parameter_match_gate_pass"] = (
                        row["dense_parameter_match_relative_error_vs_learned_global4"]
                        <= float(config["gates"]["dense_budget_match_tolerance"])
                    )

    if paired_arrays:
        _atomic_npz(output / "paired_contrasts_per_episode.npz", **paired_arrays)
    for row in completed:
        dense = by_identity.get((row["condition"], "dense", row["training_seed"]))
        if dense:
            for batch_size in (1, 300):
                key = f"complete_rollout_final_batch{batch_size}_ms"
                arm_ms = row["latency"][key]["median"]
                dense_ms = dense["latency"][key]["median"]
                reduction = 1.0 - arm_ms / max(dense_ms, 1e-12)
                row["latency"][key]["latency_reduction_vs_dense"] = reduction
                row["latency"][key]["speed_gate_pass"] = (
                    reduction >= float(config["gates"]["minimum_complete_rollout_latency_reduction"])
                )

    communication_seeds = [
        item
        for item in contrasts
        if item["condition"] == "lowrank_coupled"
        and item["left_arm"] == "learned_global4"
        and item["right_arm"] == "learned_local4"
    ]
    communication_effect = None
    if communication_seeds:
        all_directions_favor_global = all(item["mean_paired_difference"] < 0 for item in communication_seeds)
        all_intervals_favor_global = all(item["bootstrap_95pct_mean_ci"][1] < 0 for item in communication_seeds)
        communication_effect = {
            "per_seed_mean_difference_and_ci": [
                {
                    "training_seed": item["training_seed"],
                    "mean_paired_difference": item["mean_paired_difference"],
                    "bootstrap_95pct_mean_ci": item["bootstrap_95pct_mean_ci"],
                }
                for item in communication_seeds
            ],
            "all_seed_mean_directions_favor_global4": all_directions_favor_global,
            "all_seed_episode_bootstrap_intervals_favor_global4": all_intervals_favor_global,
            "pilot_status": (
                "consistent_episode_pilot_support"
                if all_directions_favor_global and all_intervals_favor_global
                else "inconclusive_or_no_go"
            ),
            "inference_scope": "three_training_seeds_are_descriptive_not_a_population_sample",
        }

    summary = {
        "protocol_id": config["protocol_id"],
        "scope": "additional_contrast" if is_additional else "frozen_primary",
        "contrast_name": args.contrast_name,
        "config": str(args.config.resolve()),
        "allocation": allocation,
        "schedule_seed": schedule_seed,
        "schedule": schedule,
        "runs": completed,
        "paired_episode_bootstrap_contrasts": contrasts,
        "paired_episode_contrast_file": "paired_contrasts_per_episode.npz" if paired_arrays else None,
        "lowrank_communication_effect": communication_effect,
        "limits": [
            "Pilot episodes are the bootstrap unit within each training seed.",
            "Three training seeds do not support training-seed-population inference.",
            "Controlled 64D systems do not establish visual-world, LeWM, CEM, or closed-loop claims.",
        ],
    }
    summary_path = output / "summary.json"
    _atomic_json(summary_path, summary, compact=True)
    summary_bytes = summary_path.stat().st_size
    if summary_bytes >= 200 * 1024:
        raise RuntimeError(f"summary.json exceeded the 200 KiB lightweight retrieval limit: {summary_bytes}")
    done = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "scope": progress["scope"],
        "contrast_name": args.contrast_name,
        "expected_runs": len(schedule),
        "completed_runs": len(completed),
        "schedule_seed": schedule_seed,
        "summary": summary_path.name,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _atomic_json(output / "DONE.json", done)
    _atomic_json(
        output / "progress.json",
        {
            "status": "complete",
            "expected_runs": len(schedule),
            "completed_runs": len(completed),
            "current_run": None,
        },
    )


if __name__ == "__main__":
    main()
