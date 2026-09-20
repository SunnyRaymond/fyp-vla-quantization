#!/usr/bin/env python3
"""Run the frozen WIDE+QUERY hidden-256 capacity contrast.

This runner is intentionally separate from the historical optimization-length
runner.  It changes only the NativeDinoPrefixStudent width to 256 and keeps the
same 128-context WIDE+QUERY bank, 1500 updates, optimizer, seeds, and held-out
candidate contract.  The baseline is read from the completed optimization
summary and is never retrained or loaded as a checkpoint.

The runner is predictor-level only.  It never changes or calls the student
observation encoder and does not execute closed-loop control.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_dino_pusht_optimization_length as opt  # noqa: E402


CAPACITY_SCHEMA = "jepa-action-prefix-compiler.capacity-contrast-freeze"
WIDTH = 256
BASELINE_WIDTH = 128
SNAPSHOT_STEPS = (500, 1000, 1500)
SNAPSHOT_NAMES = tuple(f"step_{step}" for step in SNAPSHOT_STEPS)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--deps-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be an object")
    return value


def _exact(parent: Mapping[str, Any], key: str, expected: Any, path: str) -> Any:
    if key not in parent:
        raise ValueError(f"missing frozen field {path}.{key}")
    actual = parent[key]
    if isinstance(expected, float):
        ok = float(actual) == expected
    else:
        ok = actual == expected
    if not ok:
        raise ValueError(f"frozen field mismatch at {path}.{key}: expected {expected!r}, got {actual!r}")
    return actual


def _capacity_settings(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete capacity freeze and expose only frozen values."""

    if str(freeze.get("schema", "")) != CAPACITY_SCHEMA:
        raise ValueError(f"unexpected capacity freeze schema: {freeze.get('schema')!r}")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("capacity freeze schema_version must be 1")

    baseline = _mapping(freeze, "baseline", "freeze")
    _exact(baseline, "hidden_dim", BASELINE_WIDTH, "freeze.baseline")
    _exact(
        baseline,
        "summary_schema",
        "jepa-action-prefix-compiler.dino-pusht-optimization-length-summary",
        "freeze.baseline",
    )
    model = _mapping(freeze, "treatment", "freeze")
    training = _mapping(freeze, "training", "freeze")
    randomness = _mapping(freeze, "randomness_and_schedule", "freeze")
    compatibility = _mapping(randomness, "context_schedule_compatibility", "freeze.randomness_and_schedule")
    heldout = _mapping(freeze, "heldout_evaluation", "freeze")
    ranking = _mapping(heldout, "ranking", "freeze.heldout_evaluation")
    latency = _mapping(heldout, "latency", "freeze.heldout_evaluation")
    causality = _mapping(heldout, "causality", "freeze.heldout_evaluation")
    bank = _mapping(freeze, "frozen_training_bank", "freeze")
    action_distribution = _mapping(bank, "action_query_distribution", "freeze.frozen_training_bank")
    gates = _mapping(freeze, "gates", "freeze")
    capacity_gate = _mapping(gates, "capacity_and_integrity", "freeze.gates")
    effect_gate = _mapping(gates, "capacity_effect", "freeze.gates")
    absolute_gate = _mapping(gates, "absolute_fidelity", "freeze.gates")
    relative_gate = _mapping(gates, "logged_fidelity_noninferiority", "freeze.gates")
    causality_gate = _mapping(gates, "causality", "freeze.gates")
    latency_gate = _mapping(gates, "latency", "freeze.gates")
    execution = _mapping(freeze, "execution_constraints", "freeze")

    _exact(model, "student_class", "NativeDinoPrefixStudent", "freeze.treatment")
    _exact(model, "arm", "WIDE-QUERY", "freeze.treatment")
    _exact(model, "hidden_dim", WIDTH, "freeze.treatment")
    _exact(model, "horizon", 5, "freeze.treatment")
    _exact(model, "frameskip", 5, "freeze.treatment")
    _exact(model, "primitive_action_dim", 2, "freeze.treatment")
    _exact(model, "packed_action_token_dim", 10, "freeze.treatment")
    _exact(model, "visual_dim", 384, "freeze.treatment")
    _exact(model, "proprio_dim", 10, "freeze.treatment")
    _exact(training, "steps", 1500, "freeze.training")
    _exact(training, "batch_size", 32, "freeze.training")
    _exact(training, "optimizer", "AdamW", "freeze.training")
    _exact(training, "learning_rate", 3e-4, "freeze.training")
    _exact(compatibility, "historical_steps", 500, "freeze.randomness_and_schedule.context_schedule_compatibility")
    _exact(randomness, "training_seed", 20260925, "freeze.randomness_and_schedule")
    _exact(randomness, "context_schedule_seed", 20260925, "freeze.randomness_and_schedule")
    _exact(randomness, "action_query_seed", 20260926, "freeze.randomness_and_schedule")
    _exact(randomness, "role_schedule_seed", 20260927, "freeze.randomness_and_schedule")
    _exact(randomness, "timing_action_prefix_seed", 20270925, "freeze.randomness_and_schedule")
    heldout_seeds = [int(value) for value in randomness.get("heldout_action_prefix_seeds", [])]
    if heldout_seeds != [20264925, 20264926]:
        raise ValueError("capacity freeze heldout action-prefix seeds must be [20264925, 20264926]")
    _exact(bank, "train_contexts", 128, "freeze.frozen_training_bank")
    _exact(bank, "train_episodes", 32, "freeze.frozen_training_bank")
    _exact(bank, "contexts_per_episode", 4, "freeze.frozen_training_bank")
    _exact(ranking, "block_count", 16, "freeze.heldout_evaluation.ranking")
    _exact(ranking, "heldout_contexts", 8, "freeze.heldout_evaluation.ranking")
    _exact(ranking, "topk", 30, "freeze.heldout_evaluation.ranking")
    _exact(ranking, "candidates_per_block", 300, "freeze.heldout_evaluation.ranking")
    _exact(latency, "batch_size", 300, "freeze.heldout_evaluation.latency")
    _exact(latency, "warmup_repeats", 3, "freeze.heldout_evaluation.latency")
    _exact(latency, "technical_repeats", 10, "freeze.heldout_evaluation.latency")
    _exact(execution, "gpu_telemetry_seconds", 30, "freeze.execution_constraints")
    _exact(effect_gate, "median_spearman_delta_min", 0.05, "freeze.gates.capacity_effect")
    _exact(effect_gate, "median_top30_delta_min", 0.10, "freeze.gates.capacity_effect")
    _exact(effect_gate, "positive_spearman_blocks_min", 12, "freeze.gates.capacity_effect")
    _exact(effect_gate, "positive_top30_blocks_min", 12, "freeze.gates.capacity_effect")
    _exact(capacity_gate, "future_action_leakage_max_abs", 1e-6, "freeze.gates.capacity_and_integrity")
    _exact(capacity_gate, "last10_to_first_training_mse_ratio_max", 0.8, "freeze.gates.capacity_and_integrity")
    _exact(absolute_gate, "median_spearman_min", 0.99, "freeze.gates.absolute_fidelity_final")
    _exact(absolute_gate, "minimum_spearman_min", 0.95, "freeze.gates.absolute_fidelity_final")
    _exact(absolute_gate, "median_top30_min", 0.95, "freeze.gates.absolute_fidelity_final")
    _exact(absolute_gate, "minimum_top30_min", 0.8, "freeze.gates.absolute_fidelity_final")
    _exact(relative_gate, "ratio_max", 1.25, "freeze.gates.logged_fidelity_noninferiority")
    _exact(causality_gate, "maximum_abs_delta" if "maximum_abs_delta" in causality_gate else "maximum_future_action_leakage_abs", 1e-6, "freeze.gates.causality")
    _exact(latency_gate, "predictor_only_reduction_min", 0.2, "freeze.gates.latency")
    update_schedule = _mapping(randomness, "update_schedule", "freeze.randomness_and_schedule") if "update_schedule" in randomness else None
    if update_schedule is not None:
        _exact(update_schedule, "optimizer_updates", 1500, "freeze.randomness_and_schedule.update_schedule")

    mixture = {key: float(action_distribution[key]) for key in ("logged", "gaussian_planner_init", "one_step_cem_resample")}
    cem = _mapping(bank, "one_step_cem_resample", "freeze.frozen_training_bank")
    query = {
        "fractions": mixture,
        "cem_candidates": int(cem["candidate_count_M"]),
        "cem_topk": int(cem["elite_count_K"]),
        "cem_variance_floor": float(cem["variance_floor"]),
        "evaluation_contexts": 8,
        "evaluation_action_seeds": heldout_seeds,
    }
    if query["fractions"] != {"logged": 0.5, "gaussian_planner_init": 0.25, "one_step_cem_resample": 0.25}:
        raise ValueError("capacity freeze query mixture is not the frozen 50/25/25 bank")
    if (query["cem_candidates"], query["cem_topk"], query["cem_variance_floor"]) != (64, 8, 0.05):
        raise ValueError("capacity freeze CEM bank is not M=64/K=8/variance_floor=0.05")
    if query["evaluation_contexts"] != 8 or query["evaluation_action_seeds"][:2] != heldout_seeds:
        raise ValueError("capacity freeze held-out query contract mismatch")
    return {
        "steps": 1500,
        "batch": 32,
        "lr": 3e-4,
        "hidden_dim": WIDTH,
        "visual_dim": 384,
        "proprio_dim": 10,
        "horizon": 5,
        "frame_skip": 5,
        "action_dim": 2,
        "packed_action_token_dim": 10,
        "historical_steps": 500,
        "checkpoints": SNAPSHOT_STEPS,
        "eval_batch": 300,
        "evaluation_contexts": 8,
        "evaluation_blocks": 16,
        "evaluation_topk": 30,
        "timing_batch": 300,
        "warmup": 3,
        "repeats": 10,
        "telemetry_seconds": 30,
        "train_seed": 20260925,
        "context_schedule_seed": 20260925,
        "action_query_seed": 20260926,
        "role_schedule_seed": 20260927,
        "timing_seed": 20270925,
        "evaluation_action_seeds": heldout_seeds,
        "future_action_tolerance": 1e-6,
        "capacity_gate": dict(capacity_gate),
        "effect_gate": dict(effect_gate),
        "absolute_gate": dict(absolute_gate),
        "relative_gate": dict(relative_gate),
        "causality_gate": dict(causality_gate),
        "latency_gate": dict(latency_gate),
        "baseline": dict(baseline),
        "query": query,
    }


def _parameter_report(student: Any, optimizer: Any | None = None) -> dict[str, Any]:
    import torch

    parameters = list(student.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(int(parameter.numel()) * int(parameter.element_size()) for parameter in parameters)
    optimizer_bytes = 0
    optimizer_tensors = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    optimizer_tensors += 1
                    optimizer_bytes += int(value.numel()) * int(value.element_size())
    return {
        "hidden_dim": WIDTH,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "parameter_bytes": parameter_bytes,
        "parameter_mib": parameter_bytes / (1024.0 * 1024.0),
        "optimizer_state_tensors": optimizer_tensors,
        "optimizer_state_bytes": optimizer_bytes,
        "optimizer_state_mib": optimizer_bytes / (1024.0 * 1024.0),
        "architecture_signature": {
            "student_class": "NativeDinoPrefixStudent",
            "prefix_transformer_layers": 2,
            "prefix_transformer_heads": 4,
            "prefix_feedforward_multiplier": 4,
            "dropout": 0.0,
            "observation_encoder_in_student": False,
        },
    }


def _evaluate_logged_dynamic(
    snapshots: Mapping[str, Mapping[str, Any]],
    model: Any,
    encoded_cache: Mapping[str, Any],
    action_dim: int,
    context_count: int,
    device: Any,
    hidden_dim: int,
) -> dict[str, Any]:
    import torch

    if context_count != 8 or context_count > int(encoded_cache["count"]):
        raise ValueError("capacity contrast requires exactly eight held-out contexts")
    visual_dim = int(encoded_cache["context"]["visual"].shape[-1])
    proprio_dim = int(encoded_cache["context"]["proprio"].shape[-1])
    students: dict[str, Any] = {}
    for name, state in snapshots.items():
        student = opt.NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, hidden_dim).to(device)
        student.load_state_dict(state, strict=True)
        student.eval()
        students[name] = student
    cases: dict[str, list[dict[str, Any]]] = {name: [] for name in snapshots}
    for context_index in range(context_count):
        context = {key: value[context_index : context_index + 1].to(device) for key, value in encoded_cache["context"].items()}
        actions = encoded_cache["actions"][context_index : context_index + 1].to(device)
        with torch.no_grad():
            target = opt._teacher_targets(model, context, actions)
            predictions = {name: student(context, actions) for name, student in students.items()}
        for name, prediction in predictions.items():
            opt._prediction_target_contract(prediction, target)
            cases[name].append({"context_index": context_index, **opt._metrics_by_horizon(prediction, target)})
    result: dict[str, Any] = {}
    for name, values in cases.items():
        result[name] = {
            "context_count": len(values),
            "per_context": values,
            "per_horizon": {
                "relative_mse": [opt._mean([float(item["relative_mse"][index]) for item in values]) for index in range(opt.HORIZON)],
                "cosine": [opt._mean([float(item["cosine"][index]) for item in values]) for index in range(opt.HORIZON)],
            },
            "mean_relative_mse": opt._mean([float(value) for item in values for value in item["relative_mse"]]),
        }
    return result


def _evaluate_snapshots_dynamic(
    snapshots: Mapping[str, Mapping[str, Any]],
    model: Any,
    encoded_cache: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    context_count: int,
    eval_seeds: Sequence[int],
    batch_size: int,
    device: Any,
    hidden_dim: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    blocks = opt._prepare_eval_blocks(encoded_cache, context_count, eval_seeds, batch_size, action_dim, device)
    visual_dim = int(encoded_cache["context"]["visual"].shape[-1])
    proprio_dim = int(encoded_cache["context"]["proprio"].shape[-1])
    students = {
        name: opt.NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, hidden_dim).to(device)
        for name in snapshots
    }
    for name, state in snapshots.items():
        students[name].load_state_dict(state, strict=True)
        students[name].eval()
    per_snapshot: dict[str, list[dict[str, Any]]] = {name: [] for name in snapshots}
    for block in blocks:
        with torch.no_grad():
            teacher_target = opt._teacher_targets(model, block["context"], block["actions"])
            predictions = {name: student(block["context"], block["actions"]) for name, student in students.items()}
        teacher_cost = opt._objective_cost(objective_fn, teacher_target, block["goal"])
        for name, prediction in predictions.items():
            opt._prediction_target_contract(prediction, teacher_target)
            fidelity = opt._metrics_by_horizon(prediction, teacher_target)
            per_snapshot[name].append(
                {
                    "seed": int(block["seed"]),
                    "context_index": int(block["context_index"]),
                    "candidate_count": int(batch_size),
                    "topk": min(30, int(batch_size)),
                    "ranking": {
                        "spearman": opt._spearman(teacher_cost, opt._objective_cost(objective_fn, prediction, block["goal"])),
                        "top30_overlap": opt._topk_overlap(teacher_cost, opt._objective_cost(objective_fn, prediction, block["goal"])),
                    },
                    "teacher_relative": fidelity,
                }
            )
    result: dict[str, Any] = {}
    for name, cases in per_snapshot.items():
        rankings = [case["ranking"] for case in cases]
        relative = [case["teacher_relative"] for case in cases]
        spearman = [float(item["spearman"]) for item in rankings]
        top30 = [float(item["top30_overlap"]) for item in rankings]
        result[name] = {
            "block_count": len(cases),
            "per_block": cases,
            "terminal_ranking": {
                "spearman_mean": opt._mean(spearman),
                "spearman_median": opt._median(spearman),
                "spearman_minimum": min(spearman),
                "top30_overlap_mean": opt._mean(top30),
                "top30_overlap_median": opt._median(top30),
                "top30_overlap_minimum": min(top30),
            },
            "teacher_relative_fidelity": {
                "per_horizon": {
                    "relative_mse": [opt._mean([float(item["relative_mse"][index]) for item in relative]) for index in range(opt.HORIZON)],
                    "cosine": [opt._mean([float(item["cosine"][index]) for item in relative]) for index in range(opt.HORIZON)],
                },
                "mean_relative_mse": opt._mean([float(value) for item in relative for value in item["relative_mse"]]),
            },
        }
    return result, blocks


def _block_map(per_block: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[int, int], Mapping[str, Any]]:
    mapped: dict[tuple[int, int], Mapping[str, Any]] = {}
    for case in per_block:
        key = (int(case["seed"]), int(case["context_index"]))
        if key in mapped:
            raise ValueError(f"{label} has duplicate paired block {key}")
        mapped[key] = case
    return mapped


def _paired_capacity_effect(
    variant_snapshots: Mapping[str, Any], baseline_snapshots: Mapping[str, Any], expected_blocks: int
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    exact_keys = None
    for snapshot in SNAPSHOT_NAMES:
        variant_cases = _block_map(variant_snapshots[snapshot]["per_block"], f"variant {snapshot}")
        baseline_cases = _block_map(baseline_snapshots[snapshot]["per_block"], f"baseline {snapshot}")
        if len(variant_cases) != expected_blocks or len(baseline_cases) != expected_blocks:
            raise ValueError(f"paired block count mismatch at {snapshot}")
        if set(variant_cases) != set(baseline_cases):
            raise ValueError(f"missing/mismatched paired block keys at {snapshot}")
        if exact_keys is None:
            exact_keys = sorted(variant_cases)
        elif sorted(variant_cases) != exact_keys:
            raise ValueError("paired block key order/set changes across snapshots")
        paired = []
        for key in exact_keys:
            variant = variant_cases[key]
            baseline = baseline_cases[key]
            if int(variant["candidate_count"]) != int(baseline["candidate_count"]) or int(variant["topk"]) != int(baseline["topk"]):
                raise ValueError(f"candidate count/top-k mismatch for block {key}")
            paired.append(
                {
                    "block_id": f"seed={key[0]}:context={key[1]}",
                    "seed": key[0],
                    "context_index": key[1],
                    "candidate_count": int(variant["candidate_count"]),
                    "topk": int(variant["topk"]),
                    "spearman_delta": float(variant["ranking"]["spearman"]) - float(baseline["ranking"]["spearman"]),
                    "top30_overlap_delta": float(variant["ranking"]["top30_overlap"]) - float(baseline["ranking"]["top30_overlap"]),
                    "mean_relative_mse_delta": opt._mean(variant["teacher_relative"]["relative_mse"])
                    - opt._mean(baseline["teacher_relative"]["relative_mse"]),
                }
            )
        effects[snapshot] = {
            "comparison": f"hidden_{WIDTH} {snapshot} - hidden_{BASELINE_WIDTH} {snapshot}",
            "per_block": paired,
            "spearman_delta_median": opt._median([item["spearman_delta"] for item in paired]),
            "spearman_delta_minimum": min(item["spearman_delta"] for item in paired),
            "spearman_delta_maximum": max(item["spearman_delta"] for item in paired),
            "top30_overlap_delta_median": opt._median([item["top30_overlap_delta"] for item in paired]),
            "top30_overlap_delta_minimum": min(item["top30_overlap_delta"] for item in paired),
            "top30_overlap_delta_maximum": max(item["top30_overlap_delta"] for item in paired),
            "mean_relative_mse_delta_median": opt._median([item["mean_relative_mse_delta"] for item in paired]),
            "spearman_positive_blocks": sum(item["spearman_delta"] > 0 for item in paired),
            "top30_overlap_positive_blocks": sum(item["top30_overlap_delta"] > 0 for item in paired),
        }
    return {"paired_blocks": expected_blocks, "block_ids": [f"seed={key[0]}:context={key[1]}" for key in exact_keys or []], "snapshots": effects}


def _status(ok: bool, **details: Any) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", **details}


def _gate_summary(
    settings: Mapping[str, Any], result: Mapping[str, Any], pairing: Mapping[str, Any]
) -> dict[str, Any]:
    integrity = result["student_integrity"]
    finite = bool(result["finite_outputs_and_training"])
    leakage_max = max(float(item["max_abs_delta"]) for item in result["causality_final"]["cases"])
    training_ratio = float(result["train"]["latent_last10_to_first_ratio"])
    capacity_spec = settings["capacity_gate"]
    pairing_ok = pairing["status"] == "PASS"
    capacity_ok = (
        finite
        and bool(result["snapshot_outputs_finite"])
        and bool(result["snapshot_state_dicts_finite"])
        and bool(result["shape_dtype_device_match"])
        and bool(integrity["passed"])
        and not bool(result["fallback_used"])
        and pairing_ok
        and leakage_max <= float(capacity_spec["future_action_leakage_max_abs"])
        and training_ratio <= float(capacity_spec["last10_to_first_training_mse_ratio_max"])
    )
    capacity = _status(
        capacity_ok,
        pairing_integrity=pairing,
        all_snapshot_outputs_finite=bool(result["snapshot_outputs_finite"]),
        snapshot_state_dicts_finite=bool(result["snapshot_state_dicts_finite"]),
        final_training_finite=finite,
        future_action_leakage_max_abs=leakage_max,
        future_action_leakage_max_abs_maximum=float(capacity_spec["future_action_leakage_max_abs"]),
        training_last10_to_first_ratio=training_ratio,
        training_last10_to_first_ratio_maximum=float(capacity_spec["last10_to_first_training_mse_ratio_max"]),
        hidden_encode_obs=bool(integrity["encode_obs_entry_points"]),
        source_encoder_reference=bool(integrity["source_teacher_reference"]),
        shared_teacher_module_entries=integrity["shared_teacher_module_entries"],
        no_hidden_encode_obs=not bool(integrity["encode_obs_entry_points"]),
        no_source_encoder_reference=not bool(integrity["source_teacher_reference"]),
        shape_dtype_device_match=bool(result["shape_dtype_device_match"]),
        fallback_used=bool(result["fallback_used"]),
        oom_nan_or_silent_fallback=bool(result["fallback_used"]),
    )

    effect = result["capacity_effect"]["snapshots"]["step_1500"]
    effect_spec = settings["effect_gate"]
    effect_ok = (
        float(effect["spearman_delta_median"]) >= float(effect_spec["median_spearman_delta_min"])
        and float(effect["top30_overlap_delta_median"]) >= float(effect_spec["median_top30_delta_min"])
        and int(effect["spearman_positive_blocks"]) >= int(effect_spec["positive_spearman_blocks_min"])
        and int(effect["top30_overlap_positive_blocks"]) >= int(effect_spec["positive_top30_blocks_min"])
    )
    capacity_effect = _status(
        effect_ok,
        comparison="hidden_256 step_1500 - hidden_128 step_1500",
        median_spearman_delta=effect["spearman_delta_median"],
        median_spearman_delta_min=float(effect_spec["median_spearman_delta_min"]),
        median_top30_delta=effect["top30_overlap_delta_median"],
        median_top30_delta_min=float(effect_spec["median_top30_delta_min"]),
        spearman_positive_blocks=effect["spearman_positive_blocks"],
        positive_spearman_blocks_min=int(effect_spec["positive_spearman_blocks_min"]),
        top30_positive_blocks=effect["top30_overlap_positive_blocks"],
        positive_top30_blocks_min=int(effect_spec["positive_top30_blocks_min"]),
    )

    final_ranking = result["snapshots"]["step_1500"]["terminal_ranking"]
    absolute_spec = settings["absolute_gate"]
    absolute_ok = (
        float(final_ranking["spearman_median"]) >= float(absolute_spec["median_spearman_min"])
        and float(final_ranking["spearman_minimum"]) >= float(absolute_spec["minimum_spearman_min"])
        and float(final_ranking["top30_overlap_median"]) >= float(absolute_spec["median_top30_min"])
        and float(final_ranking["top30_overlap_minimum"]) >= float(absolute_spec["minimum_top30_min"])
    )
    absolute = _status(absolute_ok, snapshot=1500, **final_ranking, thresholds=dict(absolute_spec))

    logged = result["logged_action_teacher_relative"]
    baseline_logged_mse = float(result["baseline_metrics"]["logged_teacher_relative_mse"])
    logged_ratio = opt._safe_ratio(logged["step_1500"]["mean_relative_mse"], baseline_logged_mse)
    relative_spec = settings["relative_gate"]
    relative = _status(
        logged_ratio <= float(relative_spec["ratio_max"]),
        comparison="hidden_256 step1500 / authoritative hidden_128 step1500 on eight logged held-out contexts",
        hidden_256_step1500_mean_relative_mse=logged["step_1500"]["mean_relative_mse"],
        hidden_128_step1500_mean_relative_mse=baseline_logged_mse,
        ratio=logged_ratio,
        ratio_max=float(relative_spec["ratio_max"]),
    )
    causality_max = leakage_max
    causality_spec = settings["causality_gate"]
    causality = _status(
        causality_max <= float(causality_spec["maximum_future_action_leakage_abs"]),
        snapshot=1500,
        maximum_abs_delta=causality_max,
        maximum_future_action_leakage_abs=float(causality_spec["maximum_future_action_leakage_abs"]),
    )
    latency_spec = settings["latency_gate"]
    latency = _status(
        float(result["latency_final"]["median_reduction"]) >= float(latency_spec["predictor_only_reduction_min"]),
        snapshot=1500,
        median_reduction=result["latency_final"]["median_reduction"],
        predictor_only_reduction_min=float(latency_spec["predictor_only_reduction_min"]),
        student_ms_vs_baseline_h128_ratio=result["latency_diagnostic"]["student_ms_vs_baseline_h128_ratio"],
    )
    full_ok = all(item["status"] == "PASS" for item in (capacity, absolute, relative, causality, latency))
    return {
        "schema": "jepa-action-prefix-compiler.capacity-contrast-gates",
        "capacity_and_integrity": capacity,
        "capacity_effect": capacity_effect,
        "absolute_fidelity_final": absolute,
        "teacher_relative_mse_final": relative,
        "causality_final": causality,
        "latency_final": latency,
        "decision_levels": {
            "capacity_effect": "PASS" if effect_ok else "FAIL",
            "full_replacement": "GO" if full_ok else "NO-GO",
        },
        "capacity_effect_decision": "PASS" if effect_ok else "FAIL",
        "full_replacement_decision": "GO" if full_ok else "NO-GO",
        "overall": "GO" if full_ok else "NO-GO",
    }


def _validate_baseline(summary: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("schema") != settings["baseline"]["summary_schema"]:
        raise ValueError("baseline summary schema mismatch")
    source = summary.get("source", {})
    if not isinstance(source, Mapping) or "hidden_dim=128" not in str(source.get("student", "")):
        raise ValueError("baseline summary is not the frozen hidden-128 optimization result")
    contracts = summary.get("contracts", {})
    expected = {
        "total_steps": 1500,
        "heldout_evaluation_contexts": 8,
        "candidate_count_per_block": 300,
        "paired_blocks": 16,
    }
    for key, value in expected.items():
        if int(contracts.get(key, -1)) != value:
            raise ValueError(f"baseline contract mismatch for {key}")
    if [int(value) for value in contracts.get("heldout_evaluation_seeds", [])] != settings["evaluation_action_seeds"]:
        raise ValueError("baseline held-out seeds mismatch")
    query_mixture = contracts.get("query_mixture", {})
    if not isinstance(query_mixture, Mapping):
        raise ValueError("baseline query mixture missing")
    expected_mixture = settings["query"]["fractions"]
    for key, expected_value in expected_mixture.items():
        if float(query_mixture.get(key, -1.0)) != float(expected_value):
            raise ValueError(f"baseline query mixture mismatch for {key}")
    snapshots = summary.get("snapshots", {})
    if not isinstance(snapshots, Mapping) or any(name not in snapshots for name in SNAPSHOT_NAMES):
        raise ValueError("baseline summary is missing 500/1000/1500 snapshots")
    expected_metrics = settings["baseline"].get("expected_metrics", {})
    if not isinstance(expected_metrics, Mapping):
        raise ValueError("freeze.baseline.expected_metrics must be an object")
    final_ranking = snapshots["step_1500"].get("terminal_ranking", {})
    logged = summary.get("logged_action_teacher_relative", {}).get("step_1500", {})
    latency = summary.get("latency_final", {})
    observed_metrics = {
        "spearman_median": final_ranking.get("spearman_median"),
        "spearman_minimum": final_ranking.get("spearman_minimum"),
        "top30_overlap_median": final_ranking.get("top30_overlap_median"),
        "top30_overlap_minimum": final_ranking.get("top30_overlap_minimum"),
        "logged_teacher_relative_mse": logged.get("mean_relative_mse"),
        "predictor_only_student_median_ms": latency.get("student_median_ms"),
        "predictor_only_teacher_median_ms": latency.get("teacher_median_ms"),
        "predictor_only_reduction": latency.get("median_reduction"),
    }
    for key, expected_value in expected_metrics.items():
        if key not in observed_metrics or observed_metrics[key] is None:
            raise ValueError(f"baseline metric is missing: {key}")
        if not math.isclose(float(observed_metrics[key]), float(expected_value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"baseline metric mismatch for {key}: expected {expected_value!r}, got {observed_metrics[key]!r}"
            )
    return {
        "schema": summary["schema"],
        "contracts": dict(contracts),
        "snapshots": snapshots,
        "metrics": observed_metrics,
    }


def main() -> int:
    args = _args()
    opt._require_compute_node()
    root = args.root.resolve()
    asset_root = (args.asset_root or root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = opt._load_json(args.freeze.resolve())
    settings = _capacity_settings(freeze)
    baseline_summary = opt._load_json(args.baseline_summary.resolve())
    baseline = _validate_baseline(baseline_summary, settings)
    if not args.manifest.exists():
        raise FileNotFoundError(f"capacity manifest does not exist: {args.manifest}")
    manifest = opt._load_query_manifest(args.manifest.resolve(), settings)
    opt._require_assets(asset_root, freeze)
    opt._gpu_snapshot(output, "start")

    import torch

    opt._set_seed(settings["train_seed"])
    model, workspace, _anchors, _goals, objective_fn, action_dim, device = opt._load_official(
        root,
        asset_root,
        freeze,
        output,
        settings["train_seed"],
        config_path=args.config.resolve() if args.config else None,
        checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None,
        checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None,
        data_root=args.data_root.resolve() if args.data_root else None,
    )
    if int(action_dim) != settings["packed_action_token_dim"]:
        raise ValueError("official packed action token dim does not match capacity freeze")
    del workspace, _anchors, _goals
    train_dset, heldout_dset = opt._load_trajectory_datasets(root, asset_root, freeze, args.checkpoint, args.checkpoint_config)
    train_examples = manifest["splits"]["train"]["examples"]
    heldout_examples = manifest["splits"]["heldout"]["examples"]
    cache = opt._RawEpisodeCache(8)
    train_encoded = opt._preencode_manifest(train_dset, "train", train_examples, 5, 8, cache, model, device)
    heldout_encoded = opt._preencode_manifest(heldout_dset, "heldout", heldout_examples, 5, 8, cache, model, device)
    visual_dim = int(train_encoded["context"]["visual"].shape[-1])
    proprio_dim = int(train_encoded["context"]["proprio"].shape[-1])
    if visual_dim != settings["visual_dim"] or proprio_dim != settings["proprio_dim"]:
        raise ValueError(
            f"runtime native latent dims {(visual_dim, proprio_dim)} do not match frozen {(settings['visual_dim'], settings['proprio_dim'])}"
        )
    initialization_seed_observed = int(torch.initial_seed())
    student = opt.NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=settings["lr"])
    narrow_episode_ids: list[int] = []
    for example in train_examples:
        episode_id = int(example["episode_id"])
        if episode_id not in narrow_episode_ids:
            narrow_episode_ids.append(episode_id)
        if len(narrow_episode_ids) == 2:
            break
    narrow_indices = [index for index, example in enumerate(train_examples) if int(example["episode_id"]) in narrow_episode_ids]
    if len(narrow_episode_ids) < 2 or not narrow_indices:
        raise ValueError("capacity manifest must contain two historical narrow episodes")
    wide_schedule = opt._wide_schedule(train_examples, narrow_indices, 1500, 32, settings["context_schedule_seed"], 500)
    role_schedule = opt._make_role_schedule(1500, 32, settings["role_schedule_seed"], settings["query"])
    action_banks = opt._precompute_action_banks(model, train_encoded, objective_fn, action_dim, settings["query"], settings["action_query_seed"], device, 8)
    opt._gpu_snapshot(output, "precompute_complete")

    losses: list[dict[str, float]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    checkpoint_paths: dict[str, str] = {}
    for step_index in range(1500):
        context, actions, target = opt._materialize_bank_batch(train_encoded, action_banks, wide_schedule[step_index], role_schedule[step_index], device)
        losses.append(opt._train_one_step(student, optimizer, context, actions, target, step_index + 1))
        step = step_index + 1
        if step in SNAPSHOT_STEPS:
            state = opt._cpu_state_dict(student)
            snapshots[f"step_{step}"] = state
            filename = f"capacity_h256_step{step:04d}.pt"
            torch.save(
                {
                    "schema": "jepa-action-prefix-compiler.dino-pusht-capacity-contrast-checkpoint",
                    "global_step": step,
                    "step": step,
                    "hidden_dim": WIDTH,
                    "state_dict": state,
                    "optimizer_state_dict": opt._cpu_clone(optimizer.state_dict()),
                    "manifest": str(args.manifest.resolve()),
                    "parent_freeze": str(args.freeze.resolve()),
                    "initialization_seed_observed": initialization_seed_observed,
                    "parameter_report": _parameter_report(student, optimizer),
                },
                output / filename,
            )
            checkpoint_paths[f"step_{step}"] = filename
        if step == 1 or step % 150 == 0:
            opt._gpu_snapshot(output, f"train_step_{step}")
    if tuple(snapshots) != SNAPSHOT_NAMES:
        raise RuntimeError(f"missing required snapshots: {tuple(snapshots)}")

    planner, blocks = _evaluate_snapshots_dynamic(snapshots, model, heldout_encoded, objective_fn, action_dim, 8, settings["evaluation_action_seeds"], 300, device, WIDTH)
    logged = _evaluate_logged_dynamic(snapshots, model, heldout_encoded, action_dim, 8, device, WIDTH)
    final_causality = opt._leakage_test(student, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["evaluation_action_seeds"][0] + 9000, device, 1e-6)
    final_latency = opt._latency(student, model, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["timing_seed"], 300, 3, 10, device)
    opt._gpu_snapshot(output, "complete")
    baseline_snapshots = baseline["snapshots"]
    capacity_effect = _paired_capacity_effect(planner, baseline_snapshots, 16)
    baseline_latency = baseline_summary.get("latency_final", {})
    baseline_student_ms = float(baseline_latency.get("student_median_ms", float("nan")))
    latency_diagnostic = {
        "hidden_256_student_median_ms": float(final_latency["student_median_ms"]),
        "hidden_128_baseline_student_median_ms": baseline_student_ms,
        "student_ms_vs_baseline_h128_ratio": opt._safe_ratio(float(final_latency["student_median_ms"]), baseline_student_ms),
        "same_boundary": True,
        "same_timing_seed": settings["timing_seed"],
    }
    baseline_parameter_model = opt.NativeDinoPrefixStudent(
        action_dim, visual_dim, proprio_dim, BASELINE_WIDTH
    )
    baseline_parameter_report = _parameter_report(baseline_parameter_model)
    del baseline_parameter_model
    treatment_parameter_report = _parameter_report(student, optimizer)
    parameter_diagnostic = {
        "hidden_256": treatment_parameter_report,
        "hidden_128_same_class": baseline_parameter_report,
        "trainable_parameter_ratio_h256_to_h128": opt._safe_ratio(
            float(treatment_parameter_report["trainable_parameters"]),
            float(baseline_parameter_report["trainable_parameters"]),
        ),
        "gate": None,
    }
    values = [float(item["latent_mse"]) for item in losses]
    result: dict[str, Any] = {
        "train": {
            "steps": 1500,
            "batch": 32,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "optimizer_reinitialized": False,
            "loss_first": values[0],
            "loss_last": values[-1],
            "latent_mse_median_last_10": opt._median(values[-10:]),
            "latent_last10_to_first_ratio": opt._safe_ratio(opt._median(values[-10:]), values[0]),
            "per_step": losses,
        },
        "snapshots": planner,
        "logged_action_teacher_relative": logged,
        "capacity_effect": capacity_effect,
        "baseline_metrics": baseline["metrics"],
        "causality_final": final_causality,
        "latency_final": final_latency,
        "latency_diagnostic": latency_diagnostic,
        "snapshot_outputs_finite": opt._all_finite(planner) and opt._all_finite(logged),
        "snapshot_state_dicts_finite": opt._all_finite(snapshots),
        "student_integrity": opt._student_integrity(student, model),
        "shape_dtype_device_match": all(bool(item["shape_dtype_device_match"]) for item in losses),
        "fallback_used": False,
    }
    result["finite_outputs_and_training"] = opt._all_finite(result)
    pairing = {"status": "PASS", "paired_blocks": 16, "block_ids": capacity_effect["block_ids"], "candidate_contract": {"count": 300, "topk": 30, "seeds": settings["evaluation_action_seeds"], "generator": "CPU torch.Generator", "order": "seed outer, context_index inner"}}
    result["pairing_integrity"] = pairing
    gates = _gate_summary(settings, result, pairing)
    summary = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-capacity-contrast-summary",
        "schema_version": 1,
        "protocol": str(args.freeze.resolve()),
        "freeze": str(args.freeze.resolve()),
        "manifest": str(args.manifest.resolve()),
        "baseline_summary": str(args.baseline_summary.resolve()),
        "source": {
            "student": f"NativeDinoPrefixStudent hidden_dim={WIDTH}",
            "teacher_target": "same frozen teacher rollout dense targets as hidden-128 baseline",
            "action_bank": "same precomputed frozen WIDE+QUERY bank contract",
        },
        "contracts": {
            "capacity_variant": "hidden_256",
            "baseline_variant": "hidden_128",
            "single_arm": "WIDE-QUERY",
            "total_steps": 1500,
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "heldout_evaluation_contexts": 8,
            "heldout_evaluation_seeds": settings["evaluation_action_seeds"],
            "candidate_count_per_block": 300,
            "paired_blocks": 16,
            "query_mixture": settings["query"]["fractions"],
            "same_initialization_procedure": True,
            "initialization_seed_observed": initialization_seed_observed,
            "initialization_procedure": "existing _load_official RNG path followed by NativeDinoPrefixStudent construction; cross-width state_dict equality is not applicable",
            "checkpoint_filenames": checkpoint_paths,
            "gpu_telemetry_seconds": 30,
        },
        "dimensions": {"action_dim": action_dim, "visual_dim": visual_dim, "proprio_dim": proprio_dim, "horizon": 5},
        "parameter_report": treatment_parameter_report,
        "parameter_diagnostic": parameter_diagnostic,
        "train": result["train"],
        "snapshots": result["snapshots"],
        "logged_action_teacher_relative": result["logged_action_teacher_relative"],
        "baseline_metrics": result["baseline_metrics"],
        "capacity_effect": result["capacity_effect"],
        "pairing_integrity": pairing,
        "causality_final": result["causality_final"],
        "latency_final": result["latency_final"],
        "latency_diagnostic": latency_diagnostic,
        "student_integrity": result["student_integrity"],
        "snapshot_outputs_finite": result["snapshot_outputs_finite"],
        "snapshot_state_dicts_finite": result["snapshot_state_dicts_finite"],
        "shape_dtype_device_match": result["shape_dtype_device_match"],
        "fallback_used": False,
        "finite_outputs_and_training": result["finite_outputs_and_training"],
        "gates": gates,
        "timing_boundary": "predictor-level cached native observation latent plus normalized action prefix; excludes observation encoder, CEM execution, environment interaction and closed-loop control",
        "unverified": [
            "No closed-loop CEM or environment execution is included.",
            "Capacity comparison is limited to the frozen 16 paired held-out blocks.",
            "Candidate tensor identity is established by the frozen generator/seed/order contract; tensors are not stored in the summary.",
        ],
    }
    summary_path = args.summary.resolve() if args.summary else output / "capacity_contrast_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, default=opt._json_default)
    summary_path.write_text(text, encoding="utf-8")
    default_summary = output / "capacity_contrast_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
