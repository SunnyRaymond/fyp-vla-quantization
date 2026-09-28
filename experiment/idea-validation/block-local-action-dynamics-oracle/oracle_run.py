"""Run the frozen known-coordinate oracle pilot inside an approved PBS allocation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any


PRIMARY_ARMS = ("oracle_block", "oracle_global4", "oracle_local4")
ARM_TO_BASE = {
    "oracle_block": "learned_block",
    "oracle_global4": "learned_global4",
    "oracle_local4": "learned_local4",
    "oracle_global16": "learned_global16",
}


def load_configs(config_path: Path, global16_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if global16_path is None:
        global16_path = config_path.with_name("GLOBAL16_FREEZE.json")
    addon = json.loads(global16_path.read_text(encoding="utf-8"))
    global16 = copy.deepcopy(config)
    global16["protocol_id"] = addon["protocol_id"]
    global16["conditions"] = addon["conditions"]
    global16["arms"] = addon["arms"]
    global16["dimensions"] = {**config["dimensions"], **addon["dimensions"]}
    global16["reference"] = {**config["reference"], **addon["reference"]}
    return config, global16


def prepare_reference_helpers(config: dict[str, Any]) -> Path:
    helper_root = Path(config["reference"]["source_snapshot"])
    if not (helper_root / "models.py").is_file() or not (helper_root / "run_experiment.py").is_file():
        raise FileNotFoundError(f"first-round helper snapshot missing: {helper_root}")
    root = str(helper_root)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return helper_root


def _assert_frozen_reference(config: dict[str, Any], reference_config: dict[str, Any]) -> None:
    for section in ("data", "training", "evaluation", "generator"):
        for key, value in reference_config[section].items():
            if config[section].get(key) != value:
                raise ValueError(f"freeze mismatch against first-round {section}.{key}")


def _horizon_episode_mean(np: Any, array: Any, config: dict[str, Any]) -> Any:
    horizons = [int(x) for x in config["evaluation"]["horizons"]]
    index = horizons.index(int(config["evaluation"]["primary_horizon"]))
    return np.nanmean(array[:, index, :], axis=1)


def _summarize_split(np: Any, terminal: Any, response_delta: Any, response_relative: Any,
                     true_energy: float, config: dict[str, Any]) -> dict[str, Any]:
    horizons = [int(x) for x in config["evaluation"]["horizons"]]
    episode = {h: np.nanmean(terminal[:, i, :], axis=1) for i, h in enumerate(horizons)}
    primary = episode[int(config["evaluation"]["primary_horizon"])]
    return {
        "primary_horizon": int(config["evaluation"]["primary_horizon"]),
        "horizon_episode_aggregates": {
            str(h): {"mean": float(np.mean(v)), "median": float(np.median(v)), "p90": float(np.quantile(v, .90))}
            for h, v in episode.items()
        },
        "primary_episode_mean": float(np.mean(primary)),
        "primary_episode_median": float(np.median(primary)),
        "primary_episode_p90": float(np.quantile(primary, .90)),
        "action_response_delta_normalized_mean": float(np.mean(response_delta)),
        "action_response_true_energy_normalized_mean": float(np.mean(response_relative)),
        "mean_true_response_energy": float(true_energy),
    }


def _run_oracle(torch: Any, nn: Any, np: Any, config: dict[str, Any], arm: str, condition: str,
                seed: int, system: Any, datasets: dict[str, Any], metadata: dict[str, Any],
                output: Path, writer: Any, log_file: Any, run_index: int, run_total: int,
                schedule_seed: int, runner: Any, build_oracle_model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    mean = metadata["mean"]
    base_arm = ARM_TO_BASE[arm]
    model = build_oracle_model(arm, config, seed, mean, system.M).to("cuda")
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    identity = {"condition": condition, "arm": arm, "initializer_arm": base_arm, "training_seed": int(seed)}
    run_dir = output / condition / arm / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    optimizer, train_seconds, peak_memory_mib = runner._train_one(
        torch, nn, config, model, datasets["train"], float(metadata["delta_energy"]), seed,
        writer, log_file, {k: identity[k] for k in ("condition", "arm", "training_seed")},
    )
    runner._sync(torch)
    q_error = float((model.orthogonal_matrix() - system.M).abs().max().item())
    if q_error != 0.0 or model.transform.weight.grad is not None:
        raise AssertionError(f"fixed oracle coordinate changed: max_abs={q_error}")
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "condition": condition,
        "arm": arm,
        "initializer_arm": base_arm,
        "training_seed": int(seed),
        "steps": int(config["training"]["steps"]),
        "checkpoint_selection": "last_step",
        "parameter_count": int(parameter_count),
        "fixed_coordinate": "generator M; s=(z-mean_z)@M",
    }
    runner._atomic_checkpoint(torch, run_dir / "checkpoint.pt", checkpoint)
    model.eval().prepare_inference()
    evaluation = config["evaluation"]
    horizons = [int(x) for x in evaluation["horizons"]]
    offsets = [int(x) for x in evaluation["episode_start_offsets"]]
    normalizer = max(float(metadata["delta_energy"]), 1e-6)
    split_metrics: dict[str, Any] = {}
    episode_metrics: dict[str, Any] = {}
    for split in ("dev", "test"):
        terminal = runner._evaluate_rollouts(
            torch, model, datasets[split], horizons, offsets,
            int(config["data"]["trajectory_steps"]), normalizer, int(config["training"]["batch_size"]),
        )
        response_delta, response_relative, true_energy = runner._evaluate_action_response(
            np, torch, model, system, datasets[split], int(config["data"][f"{split}_seed"]),
            int(evaluation["action_response_design"]["horizon"]),
            int(evaluation["action_response_pairs_per_episode"]), normalizer,
        )
        split_metrics[split] = _summarize_split(np, terminal, response_delta, response_relative, true_energy, config)
        episode_metrics[f"{split}_terminal_error"] = terminal
        episode_metrics[f"{split}_action_response_delta_normalized"] = response_delta
        episode_metrics[f"{split}_action_response_true_energy_normalized"] = response_relative
        runner._atomic_npz(
            run_dir / f"{split}_per_episode.npz",
            terminal_error=terminal,
            action_response_delta_normalized=response_delta,
            action_response_true_energy_normalized=response_relative,
            true_response_energy=np.asarray(true_energy, dtype=np.float32),
        )
    model.clear_inference_cache()
    record = {
        **identity,
        "schedule_index": int(run_index),
        "schedule_total": int(run_total),
        "schedule_seed": int(schedule_seed),
        "parameter_count": int(parameter_count),
        "fixed_transform_parameters": int(system.M.numel()),
        "train_delta_energy": normalizer,
        "data_identity": {
            "system_seed": int(metadata["system_seed"]),
            "split_seeds": metadata["split_seeds"],
            "episode_counts": metadata["episode_counts"],
        },
        "train_seconds": float(train_seconds),
        "peak_memory_mib": int(peak_memory_mib),
        "fixed_coordinate_max_abs_error_after_training": q_error,
        "dev": split_metrics["dev"],
        "test": split_metrics["test"],
        "checkpoint": str((run_dir / "checkpoint.pt").relative_to(output)),
        "episode_files": {s: str((run_dir / f"{s}_per_episode.npz").relative_to(output)) for s in ("dev", "test")},
    }
    del optimizer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return record, episode_metrics


def _load_reference_model(torch: Any, root: Path, record: dict[str, Any], arm: str,
                          config: dict[str, Any], mean: Any, build_model: Any) -> Any:
    checkpoint = torch.load(root / record["checkpoint"], map_location="cpu", weights_only=True)
    model = build_model(arm, config, int(record["training_seed"]), mean,
                        dense_hidden=checkpoint.get("dense_hidden"))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to("cuda").eval()


def _reproduce_reference(torch: Any, np: Any, config: dict[str, Any], addon: dict[str, Any],
                         primary_summary: dict[str, Any], global16_summary: dict[str, Any],
                         systems: dict[str, Any], datasets: dict[str, Any], metadata: dict[str, Any],
                         primary_root: Path, global16_root: Path, evaluate: Any,
                         build_model: Any, output: Path) -> tuple[dict[tuple[str, str, int], Any], list[dict[str, Any]], dict[tuple[str, str, int], Any]]:
    primary_records = {(r["condition"], r["arm"], int(r["training_seed"])): r for r in primary_summary["runs"]}
    global16_records = {int(r["training_seed"]): r for r in global16_summary["runs"]}
    primary_reference_config = json.loads((primary_root / "FREEZE.json").read_text(encoding="utf-8"))
    global16_reference_config = json.loads((global16_root / "FREEZE.json").read_text(encoding="utf-8"))
    _assert_frozen_reference(config, primary_reference_config)
    _assert_frozen_reference(config, global16_reference_config)
    data_cache: dict[tuple[str, str, int], Any] = {}
    timing_cache: dict[tuple[str, str, int], Any] = {}
    checks: list[dict[str, Any]] = []

    needed = []
    for condition in config["conditions"]:
        for seed in config["training"]["seeds"]:
            for arm in ("learned_block", "learned_global4", "learned_local4", "dense"):
                needed.append((condition, arm, int(seed)))
    for seed in config["training"]["seeds"]:
        needed.append(("lowrank_coupled", "learned_global16", int(seed)))

    for condition, arm, seed in needed:
        is_global16 = arm == "learned_global16"
        ref_root = global16_root if is_global16 else primary_root
        ref = global16_records[seed] if is_global16 else primary_records[(condition, arm, seed)]
        ref_config = global16_reference_config if is_global16 else primary_reference_config
        ref_model = _load_reference_model(
            torch, ref_root, ref, arm, ref_config, metadata[condition]["mean"], build_model,
        )
        errors = evaluate(
            torch, ref_model, datasets[condition]["test"],
            [int(x) for x in config["evaluation"]["horizons"]],
            [int(x) for x in config["evaluation"]["episode_start_offsets"]],
            int(config["data"]["trajectory_steps"]), metadata[condition]["delta_energy"],
            int(config["training"]["batch_size"]),
        )
        if abs(float(ref["train_delta_energy"]) - float(metadata[condition]["delta_energy"])) > 1e-8:
            raise AssertionError(f"train normalization mismatch: {condition}/{arm}/{seed}")
        with np.load(ref_root / ref["episode_files"]["test"]) as old:
            expected = old["terminal_error"]
            if errors.shape != expected.shape:
                raise AssertionError(f"episode/horizon shape mismatch: {condition}/{arm}/{seed}")
            diff = np.abs(errors - expected)
            max_diff = float(np.nanmax(diff))
            np.testing.assert_allclose(errors, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
        key = (condition, arm, seed)
        data_cache[key] = errors
        checks.append({
            "condition": condition, "arm": arm, "training_seed": seed,
            "reference_job": addon["reference"]["job_id"] if is_global16 else config["reference"]["primary_job_id"],
            "original_episode_errors_reproduced": True,
            "max_abs_difference": max_diff,
        })
        del ref_model
        torch.cuda.empty_cache()
    (output / "reference_reproduction.json").write_text(json.dumps(checks, separators=(",", ":")), encoding="utf-8")
    return data_cache, checks, timing_cache


def _add_contrasts(torch: Any, np: Any, config: dict[str, Any], addon: dict[str, Any],
                   completed: list[dict[str, Any]], episodes: dict[tuple[str, str, int], dict[str, Any]],
                   baseline_errors: dict[tuple[str, str, int], Any], output: Path,
                   bootstrap: Any) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    h10_index = [int(x) for x in config["evaluation"]["horizons"]].index(int(config["evaluation"]["primary_horizon"]))
    for row in completed:
        cond, arm, seed = row["condition"], row["arm"], int(row["training_seed"])
        current = np.nanmean(episodes[(cond, arm, seed)]["test_terminal_error"][:, h10_index, :], axis=1)
        comparisons = [row["initializer_arm"], "dense"]
        if arm == "oracle_global16":
            comparisons = ["learned_global16", "dense"]
        for ref_arm in comparisons:
            ref_key = (cond, ref_arm, seed)
            ref_error = np.nanmean(baseline_errors[ref_key][:, h10_index, :], axis=1)
            contrast_seed = 20260926 + 10000 * config["conditions"].index(cond) + 100 * seed + len(contrasts)
            item = bootstrap(current, ref_error, contrast_seed)
            name = f"{cond}__seed-{seed}__{arm}-minus-{ref_arm}"
            paired[name] = (current - ref_error).astype(np.float32)
            contrasts.append({**item, "condition": cond, "training_seed": seed, "left_arm": arm,
                              "right_arm": ref_arm, "episode_array_key": name,
                              "metric": "test_episode_aggregated_normalized_rollout_mse_h10"})
    if paired:
        import run_experiment as runner
        runner._atomic_npz(output / "paired_contrasts_per_episode.npz", **paired)
    return contrasts


def _benchmark_group(torch: Any, np: Any, config: dict[str, Any], global16: dict[str, Any],
                     condition: str, seed: int, systems: dict[str, Any], datasets: dict[str, Any],
                     completed_by_key: dict[tuple[str, str, int], dict[str, Any]], output: Path,
                     primary_root: Path, primary_records: dict[tuple[str, str, int], dict[str, Any]],
                     primary_config: dict[str, Any], build_model: Any, build_oracle_model: Any,
                     benchmark: Any, dense_latency_store: dict[tuple[str, int], Any]) -> None:
    names = ["oracle_block", "oracle_global4", "oracle_local4"]
    if condition == "lowrank_coupled":
        names.append("oracle_global16")
    names.append("dense")
    random.Random(92026 + 1000 * config["conditions"].index(condition) + seed).shuffle(names)
    dense_timing = None
    for arm in names:
        if arm == "dense":
            key = (condition, "dense", seed)
            ref = primary_records[key]
            model = _load_reference_model(torch, primary_root, ref, "dense", primary_config,
                                          datasets[condition]["train"]["z"].mean(dim=(0, 1)), build_model)
        else:
            key = (condition, arm, seed)
            row = completed_by_key[key]
            record_config = global16 if arm == "oracle_global16" else config
            checkpoint = torch.load(output / row["checkpoint"], map_location="cpu", weights_only=True)
            model = build_oracle_model(arm, record_config, seed, checkpoint["state_dict"]["mean"], systems[condition].M)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to("cuda").eval()
        timing = benchmark(torch, model, datasets[condition]["test"], [1, 300], 10, 20, 60)
        if arm == "dense":
            dense_timing = timing
            dense_latency_store[(condition, seed)] = timing
        else:
            row = completed_by_key[(condition, arm, seed)]
            row["latency"] = timing
        del model
        torch.cuda.empty_cache()
    if dense_timing is None:
        raise RuntimeError("same-allocation dense benchmark was not collected")
    dense_record = primary_records[(condition, "dense", seed)]
    for arm in names:
        if arm == "dense":
            continue
        row = completed_by_key[(condition, arm, seed)]
        dense_error = float(dense_record["test"]["primary_episode_mean"])
        tolerance = max(float(config["gates"]["relative_quality_increase_max"]) * max(
            dense_error, float(config["gates"]["reference_error_floor"])),
            float(config["gates"]["absolute_normalized_error_tolerance"]))
        row["test"]["dense_reference_error"] = dense_error
        row["test"]["dense_quality_tolerance"] = tolerance
        row["test"]["dense_quality_gate_pass"] = row["test"]["primary_episode_mean"] <= dense_error + tolerance
        row["latency"]["dense_reference_same_allocation"] = dense_timing
        for batch in (1, 300):
            key = f"complete_rollout_final_batch{batch}_ms"
            reduction = 1.0 - row["latency"][key]["median"] / max(dense_timing[key]["median"], 1e-12)
            row["latency"][key]["latency_reduction_vs_dense"] = reduction
            row["latency"][key]["speed_gate_pass"] = reduction >= float(config["gates"]["minimum_complete_rollout_latency_reduction"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from allocation_guard import ensure_allocation

    allocation = ensure_allocation(require_gpu=True)
    import numpy as np
    import torch
    from torch import nn

    from write_oracle_report import generate as generate_report

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    config, global16_config = load_configs(args.config)
    helper_root = prepare_reference_helpers(config)
    import controlled_system
    import models
    import run_experiment as runner
    from oracle_models import build_oracle_model
    from test_oracle import run_checks

    if tuple(config["arms"]) != PRIMARY_ARMS or int(config["execution_order"]["expected_total_runs"]) != 30:
        raise ValueError("unexpected frozen oracle run matrix")
    if len(config["conditions"]) * len(config["arms"]) * len(config["training"]["seeds"]) != 27:
        raise ValueError("primary freeze must define exactly 27 runs")
    if len(global16_config["conditions"]) * len(config["training"]["seeds"]) != 3:
        raise ValueError("global16 addon must define exactly 3 runs")
    reference = json.loads((Path(config["reference"]["primary_run"]) / "FREEZE.json").read_text(encoding="utf-8"))
    _assert_frozen_reference(config, reference)
    args.output = args.output.resolve()
    if args.output.exists() and any((args.output / name).exists() for name in (
        "training.tsv", "summary.json", "DONE.json", "summary_progress.json")):
        raise FileExistsError(f"output already contains oracle traces: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, args.output / "FREEZE.json")
    shutil.copyfile(args.config.with_name("GLOBAL16_FREEZE.json"), args.output / "GLOBAL16_FREEZE.json")
    helper_copy = args.output / "helper_source_snapshot"
    helper_copy.mkdir()
    for source_name in ("models.py", "controlled_system.py", "run_experiment.py", "allocation_guard.py"):
        shutil.copyfile(helper_root / source_name, helper_copy / source_name)

    mechanism = run_checks(config, global16_config, torch)
    (args.output / "mechanism_tests.json").write_text(
        json.dumps({"allocation": allocation, "checks": mechanism}, indent=2), encoding="utf-8")

    primary_root = Path(config["reference"]["primary_run"])
    global16_root = Path(global16_config["reference"]["remote_run"])
    primary_summary = json.loads((primary_root / "summary.json").read_text(encoding="utf-8"))
    global16_summary = json.loads((global16_root / "summary.json").read_text(encoding="utf-8"))
    global16_reference_config = json.loads((global16_root / "FREEZE.json").read_text(encoding="utf-8"))
    _assert_frozen_reference(global16_config, global16_reference_config)
    if len(primary_summary["runs"]) != 45 or len(global16_summary["runs"]) != 3:
        raise ValueError("reference jobs do not contain the frozen 45+3 runs")

    systems: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for condition in config["conditions"]:
        system, split_data, meta = controlled_system.make_datasets(config, condition, torch.device("cuda"))
        systems[condition], datasets[condition], metadata[condition] = system, split_data, meta

    schedule = [
        {"condition": c, "arm": a, "training_seed": int(s)}
        for c in config["conditions"] for a in config["arms"] for s in config["training"]["seeds"]
    ] + [
        {"condition": "lowrank_coupled", "arm": "oracle_global16", "training_seed": int(s)}
        for s in config["training"]["seeds"]
    ]
    schedule_seed = int(config["execution_order"]["shuffle_seed"])
    random.Random(schedule_seed).shuffle(schedule)
    completed: list[dict[str, Any]] = []
    episode_metrics: dict[tuple[str, str, int], dict[str, Any]] = {}
    runner._atomic_json(args.output / "progress.json", {
        "stage": "training", "status": "ready", "expected_runs": len(schedule),
        "completed_count": 0, "current_run": None,
    })
    runner._atomic_json(args.output / "summary_progress.json", {
        "protocol_id": config["protocol_id"], "scope": "known_coordinate_oracle",
        "allocation": allocation, "schedule_seed": schedule_seed,
        "expected_runs": 30, "completed_runs": completed,
    })
    with (args.output / "training.tsv").open("w", encoding="utf-8", newline="") as log_file:
        fields = ["condition", "arm", "training_seed", "step", "loss", "one_step_mse", "free_running_h1_to_5_mse"]
        writer = csv.DictWriter(log_file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        log_file.flush()
        for index, item in enumerate(schedule, 1):
            condition, arm, seed = item["condition"], item["arm"], int(item["training_seed"])
            runner._atomic_json(args.output / "progress.json", {
                "stage": "training", "status": "running", "expected_runs": len(schedule),
                "completed_count": len(completed), "current_run": item,
            })
            print(f"RUN {index}/{len(schedule)} {condition}/{arm}/seed-{seed}", flush=True)
            cfg = global16_config if arm == "oracle_global16" else config
            record, arrays = _run_oracle(
                torch, nn, np, cfg, arm, condition, seed, systems[condition], datasets[condition], metadata[condition],
                args.output, writer, log_file, index, len(schedule), schedule_seed, runner, build_oracle_model,
            )
            completed.append(record)
            episode_metrics[(condition, arm, seed)] = arrays
            runner._atomic_json(args.output / "summary_progress.json", {
                "protocol_id": config["protocol_id"], "scope": "known_coordinate_oracle",
                "allocation": allocation, "schedule_seed": schedule_seed,
                "expected_runs": 30, "completed_runs": completed,
                "current_run": item,
            })
            runner._atomic_json(args.output / "progress.json", {
                "stage": "training", "status": "running", "expected_runs": len(schedule),
                "completed_count": len(completed), "current_run": None,
                "last_completed_run": item,
            })

    baseline_errors, reproduction_checks, _ = _reproduce_reference(
        torch, np, config, global16_config, primary_summary, global16_summary,
        systems, datasets, metadata, primary_root, global16_root,
        runner._evaluate_rollouts, models.build_model, args.output,
    )
    if len(reproduction_checks) != 39 or not all(x["original_episode_errors_reproduced"] for x in reproduction_checks):
        raise AssertionError("expected all 39 original model episode-error reproductions")
    contrasts = _add_contrasts(
        torch, np, config, global16_config, completed, episode_metrics, baseline_errors,
        args.output, runner._bootstrap_contrast,
    )
    primary_records = {(r["condition"], r["arm"], int(r["training_seed"])): r for r in primary_summary["runs"]}
    completed_by_key = {(r["condition"], r["arm"], int(r["training_seed"])): r for r in completed}
    dense_timing_store: dict[tuple[str, int], Any] = {}
    for condition in config["conditions"]:
        for seed in config["training"]["seeds"]:
            _benchmark_group(
                torch, np, config, global16_config, condition, int(seed), systems, datasets,
                completed_by_key, args.output, primary_root, primary_records,
                reference, models.build_model, build_oracle_model, runner._benchmark, dense_timing_store,
            )

    summary = {
        "protocol_id": config["protocol_id"],
        "scope": "known_coordinate_oracle_primary_plus_global16_addon",
        "allocation": allocation,
        "helper_source_snapshot": str(helper_root),
        "helper_snapshot_copy": "helper_source_snapshot",
        "reference_jobs": {"primary": config["reference"]["primary_job_id"], "global16": global16_config["reference"]["job_id"]},
        "schedule_seed": schedule_seed,
        "schedule": schedule,
        "run_counts": {"primary": 27, "global16_addon": 3, "total": len(completed)},
        "runs": completed,
        "paired_episode_bootstrap_contrasts": contrasts,
        "paired_episode_contrast_file": "paired_contrasts_per_episode.npz",
        "reference_reproduction_file": "reference_reproduction.json",
        "mechanism_checks_file": "mechanism_tests.json",
        "reproduction_checks": {"expected": 39, "passed": len(reproduction_checks)},
        "global16_confounds": "global16 increases both message dimension and predictor input width/parameter count; it does not isolate bandwidth",
        "limits": [
            "Oracle coordinates use the known generator M and are unavailable in an ordinary learned visual latent.",
            "Three training seeds are descriptive pilot evidence, not a training-seed population sample.",
            "Controlled 64D systems provide no visual, C-SWM, LeWM, CEM, or closed-loop evidence.",
        ],
    }
    runner._atomic_json(args.output / "summary.json", summary, compact=True)
    runner._atomic_json(args.output / "progress.json", {
        "stage": "report", "status": "running", "expected_runs": 30,
        "completed_count": len(completed), "current_run": None,
        "reference_reproductions": len(reproduction_checks),
    })
    decision = generate_report(args.output, allocation)
    done = {
        "status": "complete", "protocol_id": config["protocol_id"],
        "expected_runs": 30, "completed_runs": len(completed),
        "primary_runs": 27, "global16_addon_runs": 3,
        "reference_reproduction_checks": len(reproduction_checks),
        "report": "REPORT.zh.md", "decision": decision,
    }
    runner._atomic_json(args.output / "DONE.json", done)
    runner._atomic_json(args.output / "progress.json", {
        "stage": "complete", "status": "complete", "expected_runs": 30,
        "completed_count": len(completed), "current_run": None,
        "reference_reproductions": len(reproduction_checks),
    })
    print(json.dumps({"status": "complete", "runs": len(completed), "reference_reproductions": len(reproduction_checks)}))


if __name__ == "__main__":
    main()
