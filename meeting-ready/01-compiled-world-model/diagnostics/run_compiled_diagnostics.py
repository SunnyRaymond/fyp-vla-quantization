#!/usr/bin/env python3
"""No-retraining mechanism diagnostics for the failed compiled LeWM model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
COMPILED_SCRIPT = HERE.parent / "run_compiled_world_model.py"
TRANSFER = HERE.parents[1] / "02-horizon-weighted-recurrent-student" / "lewm-transfer"
ANCHOR_SCRIPT = TRANSFER / "anchor-aligned-bank" / "run_anchor_aligned_bank.py"
SCHEMA = "lewm.compiled-world-model.diagnostics-summary"
HORIZON = 5
ACTION_DIM = 10
LATENT_DIM = 192
DEV_SLICE = (552, 560)
DEV_SEEDS = (20300967, 20300968)
NEW_SEED_BASE = 20301301
TRAIN_CONTEXTS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--temporal-freeze", type=Path, required=True)
    parser.add_argument("--b1-checkpoint", type=Path, required=True)
    parser.add_argument("--b3-checkpoint", type=Path, required=True)
    parser.add_argument("--positive-control-checkpoint", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("status", "run"), default="status")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def validate_freeze(freeze: Mapping[str, Any]) -> None:
    if freeze.get("schema") != "lewm.compiled-world-model.diagnostics-freeze" or int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unexpected diagnostics freeze")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("diagnostics were not frozen before results")
    if freeze["data"]["development_slice"] != "valid[552:560]" or freeze["data"]["development_seeds"] != list(DEV_SEEDS):
        raise ValueError("development contract drifted")
    if int(freeze["data"]["training_original_contexts"]) != TRAIN_CONTEXTS or int(freeze["data"]["training_new_seed_base"]) != NEW_SEED_BASE:
        raise ValueError("training-context diagnostic contract drifted")
    scope = freeze["scope"]
    if not scope.get("no_network_training") or scope.get("official_cem") != "NOT_RUN_BY_SCOPE" or scope.get("closed_loop") != "NOT_RUN_BY_SCOPE":
        raise ValueError("scope expanded beyond diagnostics")


def load_models(compiled: Any, reference: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    models = {
        "B1_dense": compiled.make_model("B1_dense"),
        "B3_r192": compiled.make_model("B3_r192"),
        "anchor_aligned_positive_control": reference.base.make_student("baseline"),
    }
    checkpoints = {
        "B1_dense": args.b1_checkpoint,
        "B3_r192": args.b3_checkpoint,
        "anchor_aligned_positive_control": args.positive_control_checkpoint,
    }
    metadata = {}
    for label, model in models.items():
        payload = torch.load(checkpoints[label].resolve(), map_location="cpu", weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state, strict=True)
        model.to("cuda").eval().requires_grad_(False)
        metadata[label] = {
            "checkpoint": str(checkpoints[label].resolve()),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "checkpoint_label": payload.get("label") if isinstance(payload, dict) else None,
            "provenance": payload.get("provenance") if isinstance(payload, dict) else None,
        }
    return models, metadata


def make_block(row: Mapping[str, Any], actions: Any, targets: Any, costs: Any, category: str, bank: int, topk: int) -> dict[str, Any]:
    return {
        "row": row,
        "actions": actions,
        "targets": targets,
        "teacher_cost": costs,
        "category": category,
        "bank": int(bank),
        "topk": int(topk),
        "context_id": str(row["context_id"]),
        "episode_id": int(row["episode_id"]),
        "stratum": str(row.get("stratum", "unknown")),
    }


def build_development_blocks(compiled: Any, anchor: Any, reference: Any, official: Any, args: argparse.Namespace, manifest: Mapping[str, Any], temporal: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[Any]]:
    ids, selection = compiled.select_episode_ids(args.dataset.resolve(), manifest, DEV_SLICE)
    previous = tuple(anchor.FRESH_SEEDS)
    anchor.FRESH_SEEDS = DEV_SEEDS
    try:
        rows, metadata = anchor.build_fresh_rows(reference, official, args.dataset.resolve(), ids, temporal)
    finally:
        anchor.FRESH_SEEDS = previous
    if len(rows) != 24 or int(metadata.get("blocks", -1)) != 48:
        raise ValueError("development bank contract drifted")
    blocks = []
    for row in rows:
        for bank in range(2):
            blocks.append(make_block(row, row["future_actions"][bank], row["teacher_targets"][bank], row["teacher_objective"][bank], "unseen_development_contexts", bank, 30))
    return blocks, {**selection, **metadata}, rows


def build_training_blocks(reference: Any, official: Any, args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], temporal: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    selected = list(rows[:TRAIN_CONTEXTS])
    counts = Counter(str(row.get("stratum")) for row in selected)
    if counts != Counter({"early": 8, "middle": 8, "late": 8}):
        raise ValueError(f"training context strata drifted: {counts}")
    original = [make_block(row, row["future_actions"], row["teacher_targets"], row["teacher_objective"], "training_original_candidates", 0, 10) for row in selected]
    low = torch.tensor(temporal["evaluation"]["official_action_low"], dtype=torch.float32).reshape(1, 1, 2)
    high = torch.tensor(temporal["evaluation"]["official_action_high"], dtype=torch.float32).reshape(1, 1, 2)
    gaussian_std = float(temporal["evaluation"]["gaussian_std"])
    generated = []
    with reference.base.HDF5EpisodeSliceReader(args.dataset.resolve()) as reader:
        for ordinal, row in enumerate(selected):
            anchor_step = int(row["anchor"])
            manifest_row = {
                "episode_id": int(row["episode_id"]),
                "history_steps": [anchor_step],
                "history_action_starts": [anchor_step],
                "future_action_start": anchor_step,
            }
            raw = reader.read_manifest_row(manifest_row)
            logged = torch.from_numpy(reference.base.pack_raw_actions(raw["future_actions_raw"]))
            generator = torch.Generator(device="cpu").manual_seed(NEW_SEED_BASE + ordinal)
            actions = reference.base._slate_actions(logged, generator, low, high, gaussian_std, 64).to("cuda")
            context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda").expand(64, -1, -1)
            with torch.no_grad():
                targets = reference.base.official_teacher_targets(official, context, actions)
                costs = reference.base._official_objective(official, row, targets)
            generated.append(make_block(row, actions.detach().cpu(), targets.detach().cpu(), costs.detach().cpu(), "training_new_candidates", 1, 10))
    return original, generated, {
        "contexts": TRAIN_CONTEXTS,
        "strata": dict(counts),
        "new_seed_base": NEW_SEED_BASE,
        "original_candidates_per_context": 64,
        "new_candidates_per_context": 64,
    }


def prediction_metrics(reference: Any, official: Any, row: Mapping[str, Any], prediction: Any, target: Any, teacher_cost: Any, topk: int) -> dict[str, Any]:
    import torch

    cost = reference.base._official_objective(official, row, prediction)
    return {
        "spearman": reference.base._spearman(teacher_cost, cost),
        "topk_overlap": reference.base._topk_overlap(teacher_cost, cost, topk),
        "relative_latent_mse": float(((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).detach().cpu()),
        "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(cost).all().item()),
    }


def diagnose_block(reference: Any, official: Any, model: Any, block: Mapping[str, Any], low_contrast_threshold: float) -> dict[str, Any]:
    import torch

    actions = reference.base._tensor(block["actions"], dtype=torch.float32).to("cuda")
    target = reference.base._tensor(block["targets"], dtype=torch.float32).to("cuda")
    teacher_cost = reference.base._tensor(block["teacher_cost"], dtype=torch.float32).to("cuda")
    context_one = reference.base._tensor(block["row"]["latent_history"], dtype=torch.float32).to("cuda")
    context = context_one.expand(actions.shape[0], -1, -1)
    reference_action = torch.zeros((1, HORIZON, ACTION_DIM), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        prediction = model(context, actions)
        teacher_reference = reference.base.official_teacher_targets(official, context_one, reference_action)
        student_reference = model(context_one, reference_action)
    teacher_mean = target.mean(dim=0, keepdim=True)
    student_mean = prediction.mean(dim=0, keepdim=True)
    teacher_residual = target - teacher_mean
    student_residual = prediction - student_mean
    common_oracle = teacher_mean + student_residual
    contrast_oracle = student_mean + teacher_residual
    anchored = teacher_reference + prediction - student_reference
    total_mse = (prediction - target).square().mean()
    common_mse = (student_mean - teacher_mean).square().mean()
    contrast_mse = (student_residual - teacher_residual).square().mean()
    teacher_contrast_energy = teacher_residual.square().sum()
    teacher_total_energy = target.square().sum().clamp_min(1e-12)
    contrast_ratio = teacher_contrast_energy / teacher_total_energy
    gamma = student_residual.square().sum() / teacher_contrast_energy.clamp_min(1e-12)
    contrast_relative_error = (student_residual - teacher_residual).square().sum() / teacher_contrast_energy.clamp_min(1e-12)
    per_horizon = []
    for horizon in range(HORIZON):
        tr = teacher_residual[:, horizon]
        sr = student_residual[:, horizon]
        denom = tr.square().sum().clamp_min(1e-12)
        per_horizon.append({
            "horizon": horizon + 1,
            "common_mse": float((student_mean[:, horizon] - teacher_mean[:, horizon]).square().mean().cpu()),
            "contrast_mse": float((sr - tr).square().mean().cpu()),
            "contrast_relative_error": float(((sr - tr).square().sum() / denom).cpu()),
            "gamma": float((sr.square().sum() / denom).cpu()),
            "teacher_contrast_energy": float(tr.square().mean().cpu()),
        })
    return {
        "context_id": block["context_id"],
        "episode_id": block["episode_id"],
        "stratum": block["stratum"],
        "bank": block["bank"],
        "candidate_count": int(actions.shape[0]),
        "topk": block["topk"],
        "total_mse": float(total_mse.cpu()),
        "common_mse": float(common_mse.cpu()),
        "contrast_mse": float(contrast_mse.cpu()),
        "decomposition_residual_abs": float((total_mse - common_mse - contrast_mse).abs().cpu()),
        "contrast_relative_error": float(contrast_relative_error.cpu()),
        "contrast_energy_ratio_gamma": float(gamma.cpu()),
        "teacher_contrast_to_total_energy_ratio": float(contrast_ratio.cpu()),
        "low_teacher_contrast": bool(float(contrast_ratio.cpu()) < low_contrast_threshold),
        "per_horizon": per_horizon,
        "predictions": {
            "raw": prediction_metrics(reference, official, block["row"], prediction, target, teacher_cost, block["topk"]),
            "teacher_mean_plus_student_residual": prediction_metrics(reference, official, block["row"], common_oracle, target, teacher_cost, block["topk"]),
            "student_mean_plus_teacher_residual": prediction_metrics(reference, official, block["row"], contrast_oracle, target, teacher_cost, block["topk"]),
            "anchored_zero_reference": prediction_metrics(reference, official, block["row"], anchored, target, teacher_cost, block["topk"]),
        },
    }


def median(values: Sequence[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def aggregate_diagnostics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {
        "blocks": len(items),
        "low_teacher_contrast_blocks": sum(bool(item["low_teacher_contrast"]) for item in items),
        "decomposition_residual_max_abs": max(float(item["decomposition_residual_abs"]) for item in items),
    }
    for key in ("total_mse", "common_mse", "contrast_mse", "contrast_relative_error", "contrast_energy_ratio_gamma", "teacher_contrast_to_total_energy_ratio"):
        output[f"{key}_median"] = median([item[key] for item in items])
    output["per_horizon"] = []
    for horizon in range(HORIZON):
        output["per_horizon"].append({
            "horizon": horizon + 1,
            **{f"{key}_median": median([item["per_horizon"][horizon][key] for item in items]) for key in ("common_mse", "contrast_mse", "contrast_relative_error", "gamma", "teacher_contrast_energy")},
        })
    output["predictions"] = {}
    for variant in ("raw", "teacher_mean_plus_student_residual", "student_mean_plus_teacher_residual", "anchored_zero_reference"):
        rows = [item["predictions"][variant] for item in items]
        output["predictions"][variant] = {
            "spearman_median": median([row["spearman"] for row in rows]),
            "spearman_minimum": min(float(row["spearman"]) for row in rows),
            "topk_overlap_median": median([row["topk_overlap"] for row in rows]),
            "topk_overlap_minimum": min(float(row["topk_overlap"]) for row in rows),
            "relative_latent_mse_median": median([row["relative_latent_mse"] for row in rows]),
            "finite": all(bool(row["finite"]) for row in rows),
        }
    return output


def episode_anchor_improvement(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for item in items:
        grouped[int(item["episode_id"])].append(item)
    per_episode = []
    for episode, group in sorted(grouped.items()):
        raw_s = median([item["predictions"]["raw"]["spearman"] for item in group])
        raw_t = median([item["predictions"]["raw"]["topk_overlap"] for item in group])
        anc_s = median([item["predictions"]["anchored_zero_reference"]["spearman"] for item in group])
        anc_t = median([item["predictions"]["anchored_zero_reference"]["topk_overlap"] for item in group])
        per_episode.append({
            "episode_id": episode,
            "spearman_delta": anc_s - raw_s,
            "topk_delta": anc_t - raw_t,
            "joint_strict_improvement": (anc_s > raw_s and anc_t >= raw_t) or (anc_t > raw_t and anc_s >= raw_s),
        })
    return {"joint_strict_improvement_episodes": sum(bool(item["joint_strict_improvement"]) for item in per_episode), "per_episode": per_episode}


def ridge_fit(phi: Any, target: Any, relative_lambda: float) -> tuple[Any, dict[str, float]]:
    import torch

    x = torch.cat((torch.ones((phi.shape[0], 1), dtype=torch.float64, device=phi.device), phi.to(torch.float64)), dim=1)
    y = target.to(torch.float64)
    gram = x.T @ x
    feature_scale = float(torch.diagonal(gram)[1:].mean().clamp_min(1e-12).cpu())
    penalty = torch.zeros_like(gram)
    penalty[1:, 1:] = torch.eye(phi.shape[1], dtype=torch.float64, device=phi.device) * (relative_lambda * feature_scale)
    system = gram + penalty
    coefficients = torch.linalg.solve(system, x.T @ y)
    return coefficients, {"feature_scale": feature_scale, "absolute_lambda": relative_lambda * feature_scale, "condition_number": float(torch.linalg.cond(system).cpu())}


def apply_affine(phi: Any, coefficients: Any) -> Any:
    import torch

    x = torch.cat((torch.ones((phi.shape[0], 1), dtype=torch.float64, device=phi.device), phi.to(torch.float64)), dim=1)
    return (x @ coefficients).to(torch.float32)


def terminal_metrics(reference: Any, official: Any, row: Mapping[str, Any], prediction: Any, target: Any, teacher_cost: Any, topk: int) -> dict[str, Any]:
    import torch

    dummy = torch.zeros((prediction.shape[0], HORIZON, LATENT_DIM), dtype=prediction.dtype, device=prediction.device)
    dummy[:, -1] = prediction
    cost = reference.base._official_objective(official, row, dummy)
    return {
        "spearman": reference.base._spearman(teacher_cost, cost),
        "topk_overlap": reference.base._topk_overlap(teacher_cost, cost, topk),
        "relative_terminal_mse": float(((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).cpu()),
        "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(cost).all().item()),
    }


def local_phi_diagnostic(reference: Any, official: Any, b3: Any, rows: Sequence[Mapping[str, Any]], relative_lambda: float) -> dict[str, Any]:
    import torch

    directions = []
    for row in rows:
        banks = [reference.base._tensor(row["future_actions"][index], dtype=torch.float32).to("cuda") for index in range(2)]
        targets = [reference.base._tensor(row["teacher_targets"][index], dtype=torch.float32).to("cuda")[:, -1] for index in range(2)]
        costs = [reference.base._tensor(row["teacher_objective"][index], dtype=torch.float32).to("cuda") for index in range(2)]
        with torch.no_grad():
            phis = [b3.action_features(actions.unsqueeze(0))[0, :, -1] for actions in banks]
        for source, destination in ((0, 1), (1, 0)):
            duplicate = (banks[destination][:, None] == banks[source][None, :]).all(dim=-1).all(dim=-1).any(dim=1)
            keep = ~duplicate
            if int(keep.sum()) < 30:
                raise ValueError("too many cross-bank duplicates for top30 evaluation")
            coefficients, conditioning = ridge_fit(phis[source], targets[source], relative_lambda)
            with torch.no_grad():
                fit_prediction = apply_affine(phis[source], coefficients)
                cross_prediction = apply_affine(phis[destination][keep], coefficients)
            fit_metrics = terminal_metrics(reference, official, row, fit_prediction, targets[source], costs[source], 30)
            cross_metrics = terminal_metrics(reference, official, row, cross_prediction, targets[destination][keep], costs[destination][keep], 30)
            directions.append({
                "context_id": row["context_id"],
                "episode_id": int(row["episode_id"]),
                "stratum": row["stratum"],
                "direction": f"bank{source}_to_bank{destination}",
                "source_candidates": int(banks[source].shape[0]),
                "destination_candidates_before_duplicates": int(banks[destination].shape[0]),
                "exact_cross_bank_duplicates_removed": int(duplicate.sum()),
                "destination_candidates_after_duplicates": int(keep.sum()),
                "conditioning": conditioning,
                "fit": fit_metrics,
                "cross": cross_metrics,
            })
    aggregate = {
        "directions": len(directions),
        "exact_cross_bank_duplicates_removed": sum(int(item["exact_cross_bank_duplicates_removed"]) for item in directions),
        "condition_number_median": median([item["conditioning"]["condition_number"] for item in directions]),
        "fit_bank": {
            "spearman_median": median([item["fit"]["spearman"] for item in directions]),
            "top30_overlap_median": median([item["fit"]["topk_overlap"] for item in directions]),
            "relative_terminal_mse_median": median([item["fit"]["relative_terminal_mse"] for item in directions]),
        },
        "cross_bank": {
            "spearman_median": median([item["cross"]["spearman"] for item in directions]),
            "spearman_minimum": min(float(item["cross"]["spearman"]) for item in directions),
            "top30_overlap_median": median([item["cross"]["topk_overlap"] for item in directions]),
            "top30_overlap_minimum": min(float(item["cross"]["topk_overlap"]) for item in directions),
            "relative_terminal_mse_median": median([item["cross"]["relative_terminal_mse"] for item in directions]),
        },
    }
    return {"aggregate": aggregate, "per_direction": directions}


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def time_call(function: Any) -> float:
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(); function(); end.record(); torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def anchored_timing(reference: Any, official: Any, b3: Any, rows: Sequence[Mapping[str, Any]], repeats: int) -> dict[str, Any]:
    import torch

    samples = {"official_teacher_30": [], "B3_raw_30": [], "B3_anchored_30": []}
    for row in rows[:8]:
        actions = reference.base._tensor(row["future_actions"][0], dtype=torch.float32).to("cuda")
        context_one = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
        context = context_one.expand(actions.shape[0], -1, -1)
        reference_action = torch.zeros((1, HORIZON, ACTION_DIM), dtype=torch.float32, device="cuda")

        def teacher30() -> Any:
            value = None
            with torch.no_grad():
                for _ in range(30):
                    prediction = reference.base.official_teacher_targets(official, context, actions)
                    value = reference.base._official_objective(official, row, prediction)
            return value

        def raw30() -> Any:
            value = None
            with torch.no_grad():
                prepared = b3.prepare(context_one)
                for _ in range(30):
                    prediction = b3.query(prepared, actions.unsqueeze(0))[0]
                    value = reference.base._official_objective(official, row, prediction)
            return value

        def anchored30() -> Any:
            value = None
            with torch.no_grad():
                teacher_reference = reference.base.official_teacher_targets(official, context_one, reference_action)
                prepared = b3.prepare(context_one)
                student_reference = b3.query(prepared, reference_action.unsqueeze(0))[0]
                for _ in range(30):
                    prediction = teacher_reference + b3.query(prepared, actions.unsqueeze(0))[0] - student_reference
                    value = reference.base._official_objective(official, row, prediction)
            return value

        functions = {"official_teacher_30": teacher30, "B3_raw_30": raw30, "B3_anchored_30": anchored30}
        for _ in range(3):
            for function in functions.values():
                time_call(function)
        for repeat in range(repeats):
            order = list(functions)
            random.Random(20301350 + len(samples["official_teacher_30"]) + repeat).shuffle(order)
            for name in order:
                samples[name].append(time_call(functions[name]))
    summary = {name: {"p50_ms": median(values), "p95_ms": percentile(values, 0.95), "samples": len(values)} for name, values in samples.items()}
    summary["speedups"] = {
        "anchored_vs_teacher_p50": summary["official_teacher_30"]["p50_ms"] / summary["B3_anchored_30"]["p50_ms"],
        "anchored_vs_raw_B3_p50": summary["B3_raw_30"]["p50_ms"] / summary["B3_anchored_30"]["p50_ms"],
    }
    return summary


def route_decision(freeze: Mapping[str, Any], development: Mapping[str, Any], phi: Mapping[str, Any], timing: Mapping[str, Any], b3_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    threshold = freeze["routing"]["oracle_recovery_thresholds"]
    predictions = development["B3_r192"]["predictions"]
    common_oracle = predictions["teacher_mean_plus_student_residual"]
    action_oracle = predictions["student_mean_plus_teacher_residual"]
    anchored = predictions["anchored_zero_reference"]
    common_recovery = common_oracle["spearman_median"] >= float(threshold["spearman_median_min"]) and common_oracle["topk_overlap_median"] >= float(threshold["top30_overlap_median_min"])
    action_recovery = action_oracle["spearman_median"] >= float(threshold["spearman_median_min"]) and action_oracle["topk_overlap_median"] >= float(threshold["top30_overlap_median_min"])
    phi_cfg = freeze["diagnostic_2"]["phi_useful_gate"]
    phi_agg = phi["aggregate"]
    phi_conditions = {
        "fit_bank_spearman": phi_agg["fit_bank"]["spearman_median"] >= float(phi_cfg["fit_bank_spearman_median_min"]),
        "cross_bank_spearman": phi_agg["cross_bank"]["spearman_median"] >= float(phi_cfg["cross_bank_spearman_median_min"]),
        "cross_bank_top30": phi_agg["cross_bank"]["top30_overlap_median"] >= float(phi_cfg["cross_bank_top30_median_min"]),
        "cross_bank_relative_mse": phi_agg["cross_bank"]["relative_terminal_mse_median"] <= float(phi_cfg["cross_bank_relative_terminal_mse_median_max"]),
    }
    anchor_cfg = freeze["diagnostic_3"]["quality_gate"]
    improvement = episode_anchor_improvement(b3_items)
    anchor_conditions = {
        "spearman": anchored["spearman_median"] >= float(anchor_cfg["spearman_median_min"]),
        "top30": anchored["topk_overlap_median"] >= float(anchor_cfg["top30_overlap_median_min"]),
        "improved_episodes": improvement["joint_strict_improvement_episodes"] >= int(anchor_cfg["strictly_improved_episodes_min"]),
        "timing_speedup": timing["speedups"]["anchored_vs_teacher_p50"] >= float(freeze["diagnostic_3"]["timing"]["speedup_vs_teacher_min"]),
        "timing_p95": timing["B3_anchored_30"]["p95_ms"] <= timing["official_teacher_30"]["p95_ms"],
    }
    phi_useful = all(phi_conditions.values())
    anchor_viable = all(anchor_conditions.values())
    if common_recovery and anchor_viable:
        decision = "ANCHORED_RESIDUAL_GO_SIGNAL"
    elif phi_useful:
        decision = "COMPILER_BOTTLENECK_SIGNAL"
    elif action_recovery:
        decision = "ACTION_RESPONSE_BOTTLENECK"
    else:
        decision = "STOP_CURRENT_SHARED_PHI_PARAMETERIZATION"
    return {
        "decision": decision,
        "common_oracle_recovery": common_recovery,
        "action_oracle_recovery": action_recovery,
        "phi_useful_cross_bank": phi_useful,
        "anchored_viable": anchor_viable,
        "phi_conditions": phi_conditions,
        "anchored_conditions": anchor_conditions,
        "anchored_episode_improvement": improvement,
        "historical_no_go_unchanged": True,
        "official_cem": "NOT_RUN_BY_SCOPE",
        "closed_loop": "NOT_RUN_BY_SCOPE",
    }


def run(args: argparse.Namespace, freeze: Mapping[str, Any]) -> dict[str, Any]:
    require_compute_node()
    import torch

    compiled = load_module("compiled_world_model_reference", COMPILED_SCRIPT)
    anchor = load_module("compiled_anchor_reference_diagnostics", ANCHOR_SCRIPT)
    reference = anchor.load_reference()
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.requires_grad_(False)
    models, model_metadata = load_models(compiled, reference, args)
    manifest = load_json(args.manifest_512.resolve())
    temporal = load_json(args.temporal_freeze.resolve())
    training_rows = torch.load(args.training_rows.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(training_rows, list) or len(training_rows) != 512:
        raise ValueError("anchor-aligned training rows must contain 512 contexts")
    dev_blocks, dev_metadata, dev_rows = build_development_blocks(compiled, anchor, reference, official, args, manifest, temporal)
    train_original, train_new, train_metadata = build_training_blocks(reference, official, args, training_rows, temporal)
    categories = {
        "training_original_candidates": train_original,
        "training_new_candidates": train_new,
        "unseen_development_contexts": dev_blocks,
    }
    diagnostics: dict[str, Any] = {}
    detailed_dev: dict[str, list[Any]] = {}
    low_threshold = float(freeze["data"]["teacher_low_contrast_ratio_threshold"])
    for label, model in models.items():
        diagnostics[label] = {}
        for category, blocks in categories.items():
            print(f"DIAGNOSE_START {label} {category}", flush=True)
            items = [diagnose_block(reference, official, model, block, low_threshold) for block in blocks]
            diagnostics[label][category] = aggregate_diagnostics(items)
            if category == "unseen_development_contexts":
                detailed_dev[label] = items
            print(f"DIAGNOSE_DONE {label} {category}", flush=True)
    phi = local_phi_diagnostic(reference, official, models["B3_r192"], dev_rows, float(freeze["diagnostic_2"]["ridge_relative_lambda"]))
    timing = anchored_timing(reference, official, models["B3_r192"], dev_rows, int(freeze["diagnostic_3"]["timing"]["repeats"]))
    development_aggregate = {label: diagnostics[label]["unseen_development_contexts"] for label in diagnostics}
    routing = route_decision(freeze, development_aggregate, phi, timing, detailed_dev["B3_r192"])
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "DIAGNOSTICS_COMPLETE",
        "decision": routing["decision"],
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "source": {
            "stablewm_home": str(args.stablewm_home.resolve()),
            "dataset": str(args.dataset.resolve()),
            "training_rows": str(args.training_rows.resolve()),
        },
        "models": model_metadata,
        "data": {"development": dev_metadata, "training": train_metadata},
        "diagnostic_1": diagnostics,
        "diagnostic_2_local_phi": phi,
        "diagnostic_3_timing": timing,
        "routing": routing,
        "scope": {"network_training": "NOT_RUN", "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE", "new_final_test": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Exploratory no-retraining mechanism diagnostics. Oracle replacements and local ridge fits are not deployable model results; development data are not a new final test.",
    }
    write_json(args.output.resolve() / "compiled_diagnostics_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = load_json(args.freeze.resolve())
    validate_freeze(freeze)
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "network_training": False, "development": "valid[552:560]", "training_contexts": 24, "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "preflight_status.json", value)
        print(json.dumps(value))
        return 0
    for path in (args.stablewm_home, args.dataset, args.manifest_512, args.training_rows, args.temporal_freeze, args.b1_checkpoint, args.b3_checkpoint, args.positive_control_checkpoint, args.protocol):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = run(args, freeze)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
