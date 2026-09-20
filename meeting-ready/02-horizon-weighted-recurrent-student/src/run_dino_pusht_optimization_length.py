#!/usr/bin/env python3
"""Measure optimization length for the frozen WIDE-QUERY student.

This runner is deliberately a single-arm continuation of the existing query
coverage experiment.  It keeps the same frozen action bank and teacher
targets, burns the old 500 x 32 NARROW context draws before generating a 1500
step WIDE schedule, and uses one AdamW instance throughout.  Snapshots are
captured on CPU at steps 500, 1000, and 1500; all snapshots are evaluated only
after training against the same 8 x 2 x 300 held-out candidate blocks.

The runner performs no model/data work outside a PBS compute allocation and
does not download, install, compile, or hash anything.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_dino_pusht_grounded_prefix import (  # noqa: E402
    _RawEpisodeCache,
    _latent_mse,
    _preencode_manifest,
    _load_trajectory_datasets,
)
from run_dino_pusht_query_coverage import (  # noqa: E402
    _ensure_query_freeze,
    _frozen_randomness,
    _load_query_manifest,
    _make_role_schedule,
    _materialize_bank_batch,
    _precompute_action_banks,
    _query_settings,
    _query_spec,
)
from run_dino_pusht_stage_a import (  # noqa: E402
    HORIZON,
    NativeDinoPrefixStudent,
    _gpu_snapshot,
    _json_default,
    _latency,
    _leakage_test,
    _load_json,
    _load_official,
    _metrics_by_horizon,
    _objective_cost,
    _repeat_anchor,
    _require_assets,
    _require_compute_node,
    _sample_actions,
    _set_seed,
    _spearman,
    _teacher_targets,
    _topk_overlap,
)


SNAPSHOT_STEPS = (500, 1000, 1500)
SNAPSHOT_NAMES = tuple(f"step_{step}" for step in SNAPSHOT_STEPS)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--deps-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def _all_finite(value: Any) -> bool:
    """Recursively validate JSON-like values, tensors, and numpy values."""

    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return True
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all().item())
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return bool(np.isfinite(value).all())
        if isinstance(value, np.generic):
            return bool(np.isfinite(value))
    except ImportError:
        pass
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _optimization_view(freeze: Mapping[str, Any]) -> Mapping[str, Any]:
    value = freeze.get("optimization_length", {})
    return value if isinstance(value, Mapping) else {}


def _optimization_gate_source(freeze: Mapping[str, Any]) -> Mapping[str, Any]:
    optimization = _optimization_view(freeze)
    value = optimization.get("gates", freeze.get("gates", {}))
    return value if isinstance(value, Mapping) else {}


def _action_query_bank_contract(freeze: Mapping[str, Any]) -> dict[str, Any] | None:
    """Translate the flat optimization freeze's canonical action bank.

    The optimization freeze intentionally does not repeat the older
    ``query_coverage``/``action_query_generation`` aliases.  Requiring every
    numerical field here prevents the compatibility translation from silently
    falling back to the query-coverage runner's defaults.
    """

    bank = freeze.get("action_query_bank")
    if bank is None:
        return None
    if not isinstance(bank, Mapping):
        raise ValueError("optimization freeze action_query_bank must be an object")
    if int(bank.get("training_contexts", -1)) != 128:
        raise ValueError("optimization freeze action_query_bank must cover exactly 128 training contexts")
    mixture = bank.get("mixture")
    rows = bank.get("rows_per_batch")
    cem = bank.get("one_step_cem_resample")
    if not all(isinstance(value, Mapping) for value in (mixture, rows, cem)):
        raise ValueError("optimization freeze action_query_bank requires mixture, rows_per_batch, and one_step_cem_resample")
    expected_mixture = {
        "logged": 0.50,
        "gaussian_planner_init": 0.25,
        "one_step_cem_resample": 0.25,
    }
    for key, expected in expected_mixture.items():
        if key not in mixture or float(mixture[key]) != expected:
            raise ValueError(f"optimization freeze action mixture must set {key}={expected}")
    expected_rows = {"logged": 16, "gaussian_planner_init": 8, "one_step_cem_resample": 8, "total": 32}
    for key, expected in expected_rows.items():
        if key not in rows or int(rows[key]) != expected:
            raise ValueError(f"optimization freeze action rows must set {key}={expected}")
    expected_cem = {
        "candidate_count_M": 64,
        "elite_count_K": 8,
        "variance_floor": 0.05,
    }
    for key, expected in expected_cem.items():
        if key not in cem or float(cem[key]) != float(expected):
            raise ValueError(f"optimization freeze CEM bank must set {key}={expected}")
    return {
        "fractions": dict(expected_mixture),
        "cem_candidates": expected_cem["candidate_count_M"],
        "cem_topk": expected_cem["elite_count_K"],
        "cem_variance_floor": expected_cem["variance_floor"],
    }


def _require_mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be an object")
    return value


def _require_exact(parent: Mapping[str, Any], key: str, expected: Any, path: str) -> Any:
    if key not in parent:
        raise ValueError(f"missing frozen field {path}.{key}")
    actual = parent[key]
    if isinstance(expected, float):
        matches = float(actual) == expected
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"frozen field mutation at {path}.{key}: expected {expected!r}, got {actual!r}")
    return actual


def _validate_flat_optimization_contract(freeze: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bind every executable numerical setting to the new flat freeze."""

    schema = str(freeze.get("schema", ""))
    if not schema.startswith("jepa-action-prefix-compiler.optimization"):
        return None
    training = _require_mapping(freeze, "training", "freeze")
    model = _require_mapping(freeze, "model_contract", "freeze")
    randomness = _require_mapping(freeze, "randomness_and_schedule", "freeze")
    compatibility = _require_mapping(randomness, "context_schedule_compatibility", "freeze.randomness_and_schedule")
    update_schedule = _require_mapping(randomness, "update_schedule", "freeze.randomness_and_schedule")
    heldout = _require_mapping(freeze, "heldout_evaluation", "freeze")
    ranking = _require_mapping(heldout, "ranking", "freeze.heldout_evaluation")
    latency = _require_mapping(heldout, "latency", "freeze.heldout_evaluation")
    causality = _require_mapping(heldout, "causality", "freeze.heldout_evaluation")
    source_manifest = _require_mapping(freeze, "source_and_manifest", "freeze")
    train_contract = _require_mapping(source_manifest, "train_contract", "freeze.source_and_manifest")
    heldout_contract = _require_mapping(source_manifest, "heldout_contract", "freeze.source_and_manifest")
    metrics = _require_mapping(freeze, "metrics_and_diagnostics", "freeze")
    optimization_effect = _require_mapping(metrics, "optimization_effect", "freeze.metrics_and_diagnostics")
    gates = _require_mapping(freeze, "gates", "freeze")
    capacity_gate = _require_mapping(gates, "capacity_and_integrity", "freeze.gates")
    effect_gate = _require_mapping(gates, "optimization_materially_helps", "freeze.gates")
    absolute_gate = _require_mapping(gates, "absolute_fidelity_final", "freeze.gates")
    relative_gate = _require_mapping(gates, "teacher_relative_mse_final", "freeze.gates")
    causality_gate = _require_mapping(gates, "causality_final", "freeze.gates")
    latency_gate = _require_mapping(gates, "latency_final", "freeze.gates")
    execution = _require_mapping(freeze, "execution_constraints", "freeze")
    _action_query_bank_contract(freeze)

    _require_exact(training, "steps", 1500, "freeze.training")
    _require_exact(training, "batch_size", 32, "freeze.training")
    _require_exact(training, "optimizer", "AdamW", "freeze.training")
    _require_exact(training, "learning_rate", 3e-4, "freeze.training")
    _require_exact(update_schedule, "single_arm", True, "freeze.randomness_and_schedule.update_schedule")
    _require_exact(update_schedule, "optimizer_updates", 1500, "freeze.randomness_and_schedule.update_schedule")
    _require_exact(compatibility, "historical_steps", 500, "freeze.randomness_and_schedule.context_schedule_compatibility")
    _require_exact(model, "hidden_dim", 128, "freeze.model_contract")
    _require_exact(model, "horizon", 5, "freeze.model_contract")
    _require_exact(model, "frameskip", 5, "freeze.model_contract")
    _require_exact(model, "primitive_action_dim", 2, "freeze.model_contract")
    _require_exact(model, "packed_action_token_dim", 10, "freeze.model_contract")
    _require_exact(train_contract, "contexts", 128, "freeze.source_and_manifest.train_contract")
    _require_exact(train_contract, "episodes", 32, "freeze.source_and_manifest.train_contract")
    _require_exact(train_contract, "examples_per_episode", 4, "freeze.source_and_manifest.train_contract")
    _require_exact(heldout_contract, "contexts", 8, "freeze.source_and_manifest.heldout_contract")
    _require_exact(heldout_contract, "episodes", 8, "freeze.source_and_manifest.heldout_contract")
    _require_exact(ranking, "block_count", 16, "freeze.heldout_evaluation.ranking")
    _require_exact(ranking, "topk", 30, "freeze.heldout_evaluation.ranking")
    _require_exact(ranking, "candidates_per_block", 300, "freeze.heldout_evaluation.ranking")
    _require_exact(causality, "snapshot", 1500, "freeze.heldout_evaluation.causality")
    _require_exact(latency, "snapshot", 1500, "freeze.heldout_evaluation.latency")
    _require_exact(latency, "batch_size", 300, "freeze.heldout_evaluation.latency")
    _require_exact(latency, "warmup_repeats", 3, "freeze.heldout_evaluation.latency")
    _require_exact(latency, "technical_repeats", 10, "freeze.heldout_evaluation.latency")
    _require_exact(execution, "gpu_telemetry_seconds", 30, "freeze.execution_constraints")
    for key, expected in (
        ("training_seed", 20260925),
        ("context_schedule_seed", 20260925),
        ("action_query_seed", 20260926),
        ("role_schedule_seed", 20260927),
        ("timing_action_prefix_seed", 20270925),
    ):
        _require_exact(randomness, key, expected, "freeze.randomness_and_schedule")
    if "heldout_action_prefix_seeds" not in randomness or [int(value) for value in randomness["heldout_action_prefix_seeds"]] != [20264925, 20264926]:
        raise ValueError("heldout action-prefix seed mutation is not allowed")
    _require_exact(optimization_effect, "paired_blocks", 16, "freeze.metrics_and_diagnostics.optimization_effect")
    _require_exact(capacity_gate, "future_action_leakage_max_abs", 1e-6, "freeze.gates.capacity_and_integrity")
    _require_exact(capacity_gate, "last10_to_first_training_mse_ratio_max", 0.8, "freeze.gates.capacity_and_integrity")
    _require_exact(effect_gate, "median_spearman_delta_min", 0.05, "freeze.gates.optimization_materially_helps")
    _require_exact(effect_gate, "median_top30_delta_min", 0.1, "freeze.gates.optimization_materially_helps")
    _require_exact(effect_gate, "positive_spearman_blocks_min", 12, "freeze.gates.optimization_materially_helps")
    _require_exact(effect_gate, "positive_top30_blocks_min", 12, "freeze.gates.optimization_materially_helps")
    _require_exact(absolute_gate, "median_spearman_min", 0.99, "freeze.gates.absolute_fidelity_final")
    _require_exact(absolute_gate, "minimum_spearman_min", 0.95, "freeze.gates.absolute_fidelity_final")
    _require_exact(absolute_gate, "median_top30_min", 0.95, "freeze.gates.absolute_fidelity_final")
    _require_exact(absolute_gate, "minimum_top30_min", 0.8, "freeze.gates.absolute_fidelity_final")
    _require_exact(relative_gate, "ratio_max", 1.25, "freeze.gates.teacher_relative_mse_final")
    _require_exact(causality_gate, "maximum_future_action_leakage_abs", 1e-6, "freeze.gates.causality_final")
    _require_exact(latency_gate, "predictor_only_reduction_min", 0.2, "freeze.gates.latency_final")
    schedule = training.get("checkpoint_schedule")
    if not isinstance(schedule, list):
        raise ValueError("freeze.training.checkpoint_schedule must be an explicit list")
    expected_schedule = {
        500: "wide_query_step0500.pt",
        1000: "wide_query_step1000.pt",
        1500: "wide_query_step1500.pt",
    }
    actual_schedule = {
        int(item.get("step", -1)): str(item.get("filename", ""))
        for item in schedule
        if isinstance(item, Mapping)
    }
    if actual_schedule != expected_schedule:
        raise ValueError(f"checkpoint schedule mutation is not allowed: {actual_schedule}")
    return {
        "steps": int(training["steps"]),
        "batch": int(training["batch_size"]),
        "lr": float(training["learning_rate"]),
        "hidden_dim": int(model["hidden_dim"]),
        "horizon": int(model["horizon"]),
        "frame_skip": int(model["frameskip"]),
        "action_dim": int(model["primitive_action_dim"]),
        "packed_action_token_dim": int(model["packed_action_token_dim"]),
        "historical_steps": int(compatibility["historical_steps"]),
        "checkpoints": tuple(sorted(actual_schedule)),
        "eval_batch": int(ranking["candidates_per_block"]),
        "evaluation_contexts": int(heldout_contract["contexts"]),
        "evaluation_blocks": int(ranking["block_count"]),
        "evaluation_topk": int(ranking["topk"]),
        "timing_batch": int(latency["batch_size"]),
        "warmup": int(latency["warmup_repeats"]),
        "repeats": int(latency["technical_repeats"]),
        "telemetry_seconds": int(execution["gpu_telemetry_seconds"]),
    }


def _optimization_settings(freeze: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Read a staged optimization-length freeze while retaining old aliases."""

    flat_contract = _validate_flat_optimization_contract(freeze)
    base = _query_settings(freeze, protocol)
    bank_contract = _action_query_bank_contract(freeze)
    if str(freeze.get("schema", "")).startswith("jepa-action-prefix-compiler.optimization") and bank_contract is None:
        raise ValueError("optimization-length freeze must declare the flat action_query_bank contract")
    optimization = _optimization_view(freeze)
    protocol_optimization = protocol.get("optimization_length", {})
    if not isinstance(protocol_optimization, Mapping):
        protocol_optimization = {}
    merged = {**protocol_optimization, **optimization}
    checkpoints_value = flat_contract["checkpoints"] if flat_contract is not None else merged.get("checkpoints", freeze.get("checkpoints", SNAPSHOT_STEPS))
    if isinstance(checkpoints_value, Mapping):
        checkpoints_value = checkpoints_value.keys()
    checkpoints = tuple(sorted({int(value) for value in checkpoints_value}))
    if checkpoints != SNAPSHOT_STEPS:
        raise ValueError(f"optimization-length freeze requires checkpoints {SNAPSHOT_STEPS}, got {checkpoints}")
    steps = int(flat_contract["steps"] if flat_contract is not None else merged.get("steps", merged.get("total_steps", 1500)))
    if steps != SNAPSHOT_STEPS[-1]:
        raise ValueError(f"optimization-length freeze requires total_steps=1500, got {steps}")
    base.update(
        {
            "steps": steps,
            "checkpoints": checkpoints,
            "batch": int(flat_contract["batch"] if flat_contract is not None else merged.get("batch_size", base["batch"])),
            "lr": float(flat_contract["lr"] if flat_contract is not None else merged.get("learning_rate", base["lr"])),
            "hidden_dim": int(flat_contract["hidden_dim"] if flat_contract is not None else merged.get("hidden_dim", base["hidden_dim"])),
            "frame_skip": int(flat_contract["frame_skip"] if flat_contract is not None else base["frame_skip"]),
            "horizon": int(flat_contract["horizon"] if flat_contract is not None else HORIZON),
            "action_dim": int(flat_contract["action_dim"] if flat_contract is not None else 2),
            "packed_action_token_dim": int(flat_contract["packed_action_token_dim"] if flat_contract is not None else 10),
            "eval_batch": int(flat_contract["eval_batch"] if flat_contract is not None else merged.get("evaluation_candidates", base["eval_batch"])),
            "evaluation_contexts": int(flat_contract["evaluation_contexts"] if flat_contract is not None else 8),
            "evaluation_blocks": int(flat_contract["evaluation_blocks"] if flat_contract is not None else 16),
            "evaluation_topk": int(flat_contract["evaluation_topk"] if flat_contract is not None else 30),
            "timing_batch": int(flat_contract["timing_batch"] if flat_contract is not None else merged.get("timing_batch_size", base["timing_batch"])),
            "warmup": int(flat_contract["warmup"] if flat_contract is not None else merged.get("warmup_repeats", base["warmup"])),
            "repeats": int(flat_contract["repeats"] if flat_contract is not None else merged.get("technical_repeats", base["repeats"])),
            "telemetry_seconds": int(flat_contract["telemetry_seconds"] if flat_contract is not None else 30),
            "historical_steps": int(flat_contract["historical_steps"] if flat_contract is not None else 500),
        }
    )
    if base["hidden_dim"] != 128:
        raise ValueError("optimization-length experiment is frozen to hidden_dim=128")
    if base["batch"] != 32:
        raise ValueError("optimization-length experiment is frozen to batch_size=32")
    training_contract = freeze.get("training", {})
    checkpoint_schedule = training_contract.get("checkpoint_schedule") if isinstance(training_contract, Mapping) else None
    if checkpoint_schedule is not None:
        if not isinstance(checkpoint_schedule, list):
            raise ValueError("optimization freeze training.checkpoint_schedule must be a list")
        expected_filenames = {
            500: "wide_query_step0500.pt",
            1000: "wide_query_step1000.pt",
            1500: "wide_query_step1500.pt",
        }
        actual = {
            int(item.get("step", -1)): str(item.get("filename", ""))
            for item in checkpoint_schedule
            if isinstance(item, Mapping)
        }
        if actual != expected_filenames:
            raise ValueError(f"optimization freeze checkpoint filenames mismatch: {actual}")
    if bank_contract is not None:
        capacity = _optimization_gate_source(freeze).get("capacity_and_integrity", {})
        if isinstance(capacity, Mapping) and "future_action_leakage_max_abs" in capacity:
            base["future_action_tolerance"] = float(capacity["future_action_leakage_max_abs"])
    return base


def _query_view_for_optimization(freeze: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Expose nested query coverage to the existing canonical parser."""

    view = dict(freeze)
    optimization = _optimization_view(freeze)
    bank_contract = _action_query_bank_contract(freeze)
    if bank_contract is not None:
        view["query_coverage"] = bank_contract
    elif "query_coverage" not in view and "query" not in view:
        for key in ("query_coverage", "query"):
            if isinstance(optimization.get(key), Mapping):
                view[key] = optimization[key]
                break
    if "action_query_generation" not in view and isinstance(optimization.get("action_query_generation"), Mapping):
        view["action_query_generation"] = optimization["action_query_generation"]
    if "randomness_and_schedule" not in view and isinstance(optimization.get("randomness_and_schedule"), Mapping):
        view["randomness_and_schedule"] = optimization["randomness_and_schedule"]
    return view


def _ensure_optimization_freeze(
    freeze: Mapping[str, Any], query: Mapping[str, Any], settings: Mapping[str, Any]
) -> None:
    schema = str(freeze.get("schema", ""))
    if schema in {
        "jepa-action-prefix-compiler.query-coverage-freeze",
        "jepa-action-prefix-compiler.grounded-prefix-freeze",
    }:
        _ensure_query_freeze(freeze, query)
    elif schema not in {
        "jepa-action-prefix-compiler.optimization-length-freeze",
        "jepa-action-prefix-compiler.optimization_length_freeze",
    }:
        raise ValueError(f"unexpected optimization-length freeze schema: {schema!r}")
    else:
        # The new schema may nest the unchanged query contract below
        # ``optimization_length``; validate it with the existing canonical
        # consistency checks after the caller has exposed that view.
        compatibility = dict(freeze)
        compatibility["schema"] = "jepa-action-prefix-compiler.query-coverage-freeze"
        optimization = _optimization_view(freeze)
        bank_contract = _action_query_bank_contract(freeze)
        if bank_contract is not None:
            compatibility["query_coverage"] = bank_contract
        for key in ("query_coverage", "query", "action_query_generation"):
            if key not in compatibility and isinstance(optimization.get(key), Mapping):
                compatibility[key] = optimization[key]
        _ensure_query_freeze(compatibility, query)
    if int(settings["steps"]) != 1500 or tuple(settings["checkpoints"]) != SNAPSHOT_STEPS:
        raise ValueError("optimization-length freeze must expose 1500 steps and 500/1000/1500 checkpoints")


def _cross_check_runtime_contract(query: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    if int(settings["horizon"]) != HORIZON or int(settings["frame_skip"]) != 5:
        raise ValueError("runtime horizon/frame-skip contract is not frozen to H=5/frame_skip=5")
    if int(query["evaluation_contexts"]) != int(settings["evaluation_contexts"]):
        raise ValueError("query evaluation contexts do not match the flat freeze")
    if int(settings["evaluation_blocks"]) != int(settings["evaluation_contexts"]) * len(query["evaluation_action_seeds"]):
        raise ValueError("held-out block count does not match contexts x fresh seeds")
    if int(settings["evaluation_topk"]) != 30 or int(settings["eval_batch"]) != 300:
        raise ValueError("held-out top-k/candidate count does not match the flat freeze")
    if int(settings["timing_batch"]) != 300 or int(settings["warmup"]) != 3 or int(settings["repeats"]) != 10:
        raise ValueError("latency settings do not match the flat freeze")
    if int(settings["telemetry_seconds"]) != 30:
        raise ValueError("GPU telemetry interval does not match the flat freeze")


def _wide_schedule(
    train_examples: Sequence[Mapping[str, Any]],
    narrow_indices: Sequence[int],
    steps: int,
    batch: int,
    context_seed: int,
    historical_steps: int,
) -> list[list[int]]:
    """Reproduce old WIDE-QUERY's first 500 schedules exactly.

    The old runner consumed 500 x 32 NARROW draws before constructing its
    500-step WIDE schedule.  Repeating that burn-in and then drawing 1500 WIDE
    batches preserves the old first 500 WIDE batches byte-for-byte at the RNG
    sequence level without materialising the unused narrow schedule.
    """

    generator = random.Random(int(context_seed))
    for _ in range(historical_steps):
        for _ in range(batch):
            generator.randrange(len(narrow_indices))
    return [
        [generator.randrange(len(train_examples)) for _ in range(batch)]
        for _ in range(steps)
    ]


def _train_one_step(
    student: Any,
    optimizer: Any,
    context: Mapping[str, Any],
    actions: Any,
    target: Mapping[str, Any],
    step: int,
) -> dict[str, float]:
    prediction = student(context, actions)
    contract = _prediction_target_contract(prediction, target)
    latent_loss = _latent_mse(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    latent_loss.backward()
    optimizer.step()
    return {
        "step": step,
        "latent_mse": float(latent_loss.detach().cpu()),
        "shape_dtype_device_match": bool(contract["match"]),
    }


def _prediction_target_contract(
    prediction: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    """Assert the native output contract before calculating loss or metrics."""

    expected_keys = ("visual", "proprio")
    prediction_keys = tuple(sorted(prediction))
    target_keys = tuple(sorted(target))
    expected_key_set = set(expected_keys)
    keys_match = set(prediction_keys) == expected_key_set and set(target_keys) == expected_key_set
    shape_match = keys_match and all(prediction[key].shape == target[key].shape for key in expected_keys)
    dtype_match = keys_match and all(prediction[key].dtype == target[key].dtype for key in expected_keys)
    device_match = keys_match and all(prediction[key].device == target[key].device for key in expected_keys)
    match = bool(keys_match and shape_match and dtype_match and device_match)
    if not match:
        raise RuntimeError(
            "student/teacher native output contract mismatch: "
            f"keys={keys_match}, shape={shape_match}, dtype={dtype_match}, device={device_match}"
        )
    return {
        "match": match,
        "keys_match": bool(keys_match),
        "shape_match": bool(shape_match),
        "dtype_match": bool(dtype_match),
        "device_match": bool(device_match),
        "prediction_shapes": {key: tuple(prediction[key].shape) for key in expected_keys},
        "target_shapes": {key: tuple(target[key].shape) for key in expected_keys},
        "prediction_dtypes": {key: str(prediction[key].dtype) for key in expected_keys},
        "target_dtypes": {key: str(target[key].dtype) for key in expected_keys},
        "prediction_devices": {key: str(prediction[key].device) for key in expected_keys},
        "target_devices": {key: str(target[key].device) for key in expected_keys},
    }


def _student_integrity(student: Any, teacher: Any) -> dict[str, Any]:
    """Check that the post-hoc student has no teacher/encoder entry point."""

    import torch

    named_modules = [name.lower() for name, _ in student.named_modules()]
    teacher_module_ids = {id(module) for _, module in teacher.named_modules()}
    shared_teacher_modules = [
        name for name, module in student.named_modules() if id(module) in teacher_module_ids
    ]
    encode_obs_entries = [name for name in named_modules if "encode_obs" in name]
    encoder_entries = [name for name in named_modules if "encoder" in name]
    teacher_reference = student is teacher
    for value in vars(student).values():
        if value is teacher:
            teacher_reference = True
        if isinstance(value, torch.nn.Module) and value is teacher:
            teacher_reference = True
    evidence = {
        "student_is_teacher": bool(student is teacher),
        "encode_obs_entry_points": encode_obs_entries,
        "source_encoder_module_entries": encoder_entries,
        "shared_teacher_module_entries": shared_teacher_modules,
        "source_teacher_reference": bool(teacher_reference),
    }
    evidence["passed"] = not any(
        (
            evidence["student_is_teacher"],
            encode_obs_entries,
            encoder_entries,
            shared_teacher_modules,
            teacher_reference,
        )
    )
    if not evidence["passed"]:
        raise RuntimeError(f"student integrity contract failed: {evidence}")
    return evidence


def _cpu_state_dict(student: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}


def _cpu_clone(value: Any) -> Any:
    """Move checkpoint tensors to CPU without changing the live optimizer."""

    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _checkpoint_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_cpu_state": torch.get_rng_state().detach().cpu().clone(),
    }
    state["torch_cuda_state_all"] = (
        [item.detach().cpu().clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return state


def _schedule_metadata(
    settings: Mapping[str, Any],
    randomness: Mapping[str, Any],
    query: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "training_seed": int(settings["train_seed"]),
        "context_schedule_seed": int(randomness["context_schedule_seed"]),
        "role_schedule_seed": int(randomness["role_schedule_seed"]),
        "action_query_seed": int(randomness["action_query_seed"]),
        "context_burn_in_steps": int(settings["historical_steps"]),
        "context_burn_in_draws": int(settings["historical_steps"]) * int(settings["batch"]),
        "wide_schedule_steps": int(settings["steps"]),
        "batch_size": int(settings["batch"]),
        "query_mixture": {
            "logged": float(query["fractions"]["logged"]),
            "gaussian_planner_init": float(query["fractions"]["gaussian_planner_init"]),
            "one_step_cem_resample": float(query["fractions"]["one_step_cem_resample"]),
        },
        "cem": {
            "candidate_count_M": int(query["cem_candidates"]),
            "elite_count_K": int(query["cem_topk"]),
            "variance_floor": float(query["cem_variance_floor"]),
        },
    }


def _prepare_eval_blocks(
    encoded_cache: Mapping[str, Any],
    context_count: int,
    eval_seeds: Sequence[int],
    batch_size: int,
    action_dim: int,
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    if context_count > int(encoded_cache["count"]):
        raise ValueError("held-out manifest has fewer contexts than evaluation requires")
    blocks: list[dict[str, Any]] = []
    for seed in eval_seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for context_index in range(context_count):
            context_one = {
                key: value[context_index : context_index + 1].to(device)
                for key, value in encoded_cache["context"].items()
            }
            grounded_one = {
                key: value[context_index : context_index + 1].to(device)
                for key, value in encoded_cache["grounded"].items()
            }
            context = _repeat_anchor(context_one, batch_size)
            goal = _repeat_anchor({key: value[:, -1:] for key, value in grounded_one.items()}, batch_size)
            actions = _sample_actions(generator, batch_size, action_dim, device)
            blocks.append(
                {
                    "seed": int(seed),
                    "context_index": int(context_index),
                    "context": context,
                    "goal": goal,
                    "actions": actions,
                }
            )
    return blocks


def _evaluate_logged_action_snapshots(
    snapshots: Mapping[str, Mapping[str, Any]],
    model: Any,
    encoded_cache: Mapping[str, Any],
    action_dim: int,
    context_count: int,
    device: Any,
) -> dict[str, Any]:
    """Evaluate all eight held-out real logged prefixes after training only."""

    import torch

    if context_count != 8 or context_count > int(encoded_cache["count"]):
        raise ValueError("optimization-length logged-action fidelity requires exactly 8 held-out contexts")
    visual_dim = int(encoded_cache["context"]["visual"].shape[-1])
    proprio_dim = int(encoded_cache["context"]["proprio"].shape[-1])
    students: dict[str, Any] = {}
    for name, state in snapshots.items():
        student = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, 128).to(device)
        student.load_state_dict(state, strict=True)
        student.eval()
        students[name] = student

    cases: dict[str, list[dict[str, Any]]] = {name: [] for name in snapshots}
    for context_index in range(context_count):
        context = {
            key: value[context_index : context_index + 1].to(device)
            for key, value in encoded_cache["context"].items()
        }
        actions = encoded_cache["actions"][context_index : context_index + 1].to(device)
        with torch.no_grad():
            teacher_target = _teacher_targets(model, context, actions)
            predictions = {name: student(context, actions) for name, student in students.items()}
            for prediction in predictions.values():
                _prediction_target_contract(prediction, teacher_target)
        for name, prediction in predictions.items():
            fidelity = _metrics_by_horizon(prediction, teacher_target)
            cases[name].append({"context_index": context_index, **fidelity})

    result: dict[str, Any] = {}
    for name, values in cases.items():
        result[name] = {
            "context_count": len(values),
            "per_context": values,
            "per_horizon": {
                "relative_mse": [
                    _mean([float(item["relative_mse"][index]) for item in values])
                    for index in range(HORIZON)
                ],
                "cosine": [
                    _mean([float(item["cosine"][index]) for item in values])
                    for index in range(HORIZON)
                ],
            },
            "mean_relative_mse": _mean(
                [float(value) for item in values for value in item["relative_mse"]]
            ),
        }
    return result


def _evaluate_snapshots(
    snapshots: Mapping[str, Mapping[str, Any]],
    model: Any,
    encoded_cache: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    context_count: int,
    eval_seeds: Sequence[int],
    batch_size: int,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate every snapshot on one shared, pre-generated candidate bank."""

    import torch

    visual_dim = int(encoded_cache["context"]["visual"].shape[-1])
    proprio_dim = int(encoded_cache["context"]["proprio"].shape[-1])
    blocks = _prepare_eval_blocks(
        encoded_cache, context_count, eval_seeds, batch_size, action_dim, device
    )
    students: dict[str, Any] = {}
    for name, state in snapshots.items():
        student = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, 128).to(device)
        student.load_state_dict(state, strict=True)
        student.eval()
        students[name] = student

    per_snapshot: dict[str, list[dict[str, Any]]] = {name: [] for name in snapshots}
    for block in blocks:
        with torch.no_grad():
            teacher_target = _teacher_targets(model, block["context"], block["actions"])
            predictions = {
                name: student(block["context"], block["actions"])
                for name, student in students.items()
            }
            for prediction in predictions.values():
                _prediction_target_contract(prediction, teacher_target)
        teacher_cost = _objective_cost(objective_fn, teacher_target, block["goal"])
        for name, prediction in predictions.items():
            student_cost = _objective_cost(objective_fn, prediction, block["goal"])
            ranking = {
                "spearman": _spearman(teacher_cost, student_cost),
                "top30_overlap": _topk_overlap(teacher_cost, student_cost),
            }
            fidelity = _metrics_by_horizon(prediction, teacher_target)
            per_snapshot[name].append(
                {
                    "seed": block["seed"],
                    "context_index": block["context_index"],
                    "candidate_count": batch_size,
                    "topk": min(30, batch_size),
                    "ranking": ranking,
                    "teacher_relative": fidelity,
                }
            )

    result: dict[str, Any] = {}
    for name, cases in per_snapshot.items():
        rankings = [case["ranking"] for case in cases]
        relative = [case["teacher_relative"] for case in cases]
        spearman_values = [float(item["spearman"]) for item in rankings]
        top30_values = [float(item["top30_overlap"]) for item in rankings]
        result[name] = {
            "block_count": len(cases),
            "per_block": cases,
            "terminal_ranking": {
                "spearman_mean": _mean(spearman_values),
                "spearman_median": _median(spearman_values),
                "spearman_minimum": min(spearman_values),
                "top30_overlap_mean": _mean(top30_values),
                "top30_overlap_median": _median(top30_values),
                "top30_overlap_minimum": min(top30_values),
            },
            "teacher_relative_fidelity": {
                "per_horizon": {
                    "relative_mse": [
                        _mean([float(item["relative_mse"][index]) for item in relative])
                        for index in range(HORIZON)
                    ],
                    "cosine": [
                        _mean([float(item["cosine"][index]) for item in relative])
                        for index in range(HORIZON)
                    ],
                },
                "mean_relative_mse": _mean(
                    [float(value) for item in relative for value in item["relative_mse"]]
                ),
            },
        }

    effects: dict[str, Any] = {}
    for left, right, label in (("step_1000", "step_500", "1000-500"), ("step_1500", "step_500", "1500-500"), ("step_1500", "step_1000", "1500-1000")):
        left_cases = per_snapshot[left]
        right_cases = per_snapshot[right]
        paired: list[dict[str, Any]] = []
        for left_case, right_case in zip(left_cases, right_cases):
            paired.append(
                {
                    "seed": left_case["seed"],
                    "context_index": left_case["context_index"],
                    "spearman_delta": left_case["ranking"]["spearman"] - right_case["ranking"]["spearman"],
                    "top30_overlap_delta": left_case["ranking"]["top30_overlap"] - right_case["ranking"]["top30_overlap"],
                    "mean_relative_mse_delta": _mean(left_case["teacher_relative"]["relative_mse"])
                    - _mean(right_case["teacher_relative"]["relative_mse"]),
                }
            )
        effects[label] = {
            "comparison": f"{left} - {right}",
            "per_block": paired,
            "spearman_delta_median": _median([item["spearman_delta"] for item in paired]),
            "top30_overlap_delta_median": _median([item["top30_overlap_delta"] for item in paired]),
            "mean_relative_mse_delta_median": _median([item["mean_relative_mse_delta"] for item in paired]),
            "spearman_positive_blocks": sum(item["spearman_delta"] > 0 for item in paired),
            "top30_overlap_positive_blocks": sum(item["top30_overlap_delta"] > 0 for item in paired),
        }
    return result, {"shared_candidate_blocks": blocks, "paired_effects": effects}


def _frozen_gates(
    freeze: Mapping[str, Any],
    result: Mapping[str, Any],
    causality: Mapping[str, Any],
    latency: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    source = _optimization_gate_source(freeze)

    def group(name: str) -> Mapping[str, Any]:
        value = source.get(name, {})
        return value if isinstance(value, Mapping) else {}

    def threshold(values: Mapping[str, Any], key: str, default: float) -> float:
        return float(values.get(key, default))

    def integer_threshold(values: Mapping[str, Any], key: str, default: int) -> int:
        return int(values.get(key, default))

    def status(ok: bool, **details: Any) -> dict[str, Any]:
        return {"status": "PASS" if ok else "FAIL", **details}

    capacity_spec = group("capacity_and_integrity")
    effect_spec = group("optimization_materially_helps")
    absolute_spec = group("absolute_fidelity_final")
    relative_spec = group("teacher_relative_mse_final")
    causality_spec = group("causality_final")
    latency_spec = group("latency_final")

    effect = result["paired_effects"]["1500-500"]
    final_ranking = result["snapshots"]["step_1500"]["terminal_ranking"]
    logged = result["logged_action_teacher_relative"]
    logged_ratio = _safe_ratio(
        logged["step_1500"]["mean_relative_mse"],
        logged["step_500"]["mean_relative_mse"],
    )
    leakage_max = max(float(item["max_abs_delta"]) for item in causality["cases"])
    training_ratio = float(result["train"]["latent_last10_to_first_ratio"])
    finite = bool(result["finite_outputs_and_training"])
    integrity = result["student_integrity"]
    integrity_ok = bool(integrity["passed"])
    shape_dtype_device_match = bool(result["shape_dtype_device_match"])
    fallback_used = bool(result["fallback_used"])
    capacity_ok = (
        finite
        and bool(result["snapshot_state_dicts_finite"])
        and bool(result["snapshot_outputs_finite"])
        and integrity_ok
        and shape_dtype_device_match
        and not fallback_used
        and leakage_max <= threshold(capacity_spec, "future_action_leakage_max_abs", 1e-6)
        and training_ratio <= threshold(capacity_spec, "last10_to_first_training_mse_ratio_max", 0.8)
    )
    capacity = status(
        capacity_ok,
        all_snapshot_outputs_finite=bool(result["snapshot_outputs_finite"]),
        snapshot_state_dicts_finite=bool(result["snapshot_state_dicts_finite"]),
        final_training_finite=finite,
        future_action_leakage_max_abs=leakage_max,
        future_action_leakage_max_abs_maximum=threshold(capacity_spec, "future_action_leakage_max_abs", 1e-6),
        training_last10_to_first_ratio=training_ratio,
        training_last10_to_first_ratio_maximum=threshold(capacity_spec, "last10_to_first_training_mse_ratio_max", 0.8),
        hidden_encode_obs=(len(integrity["encode_obs_entry_points"]) > 0),
        source_encoder_reference=bool(integrity["source_teacher_reference"]),
        no_hidden_encode_obs=(len(integrity["encode_obs_entry_points"]) == 0),
        no_source_encoder_reference=(not bool(integrity["source_teacher_reference"])),
        shape_dtype_device_match=shape_dtype_device_match,
        fallback_used=fallback_used,
        oom_nan_or_silent_fallback=fallback_used,
    )

    effect_ok = (
        float(effect["spearman_delta_median"]) >= threshold(effect_spec, "median_spearman_delta_min", 0.05)
        and float(effect["top30_overlap_delta_median"]) >= threshold(effect_spec, "median_top30_delta_min", 0.1)
        and int(effect["spearman_positive_blocks"]) >= integer_threshold(effect_spec, "positive_spearman_blocks_min", 12)
        and int(effect["top30_overlap_positive_blocks"]) >= integer_threshold(effect_spec, "positive_top30_blocks_min", 12)
    )
    optimization_materially_helps = status(
        effect_ok,
        comparison="step1500 - step500",
        median_spearman_delta=effect["spearman_delta_median"],
        median_spearman_delta_min=threshold(effect_spec, "median_spearman_delta_min", 0.05),
        median_top30_delta=effect["top30_overlap_delta_median"],
        median_top30_delta_min=threshold(effect_spec, "median_top30_delta_min", 0.1),
        spearman_positive_blocks=effect["spearman_positive_blocks"],
        positive_spearman_blocks_min=integer_threshold(effect_spec, "positive_spearman_blocks_min", 12),
        top30_positive_blocks=effect["top30_overlap_positive_blocks"],
        positive_top30_blocks_min=integer_threshold(effect_spec, "positive_top30_blocks_min", 12),
    )

    absolute_ok = (
        float(final_ranking["spearman_median"]) >= threshold(absolute_spec, "median_spearman_min", 0.99)
        and float(final_ranking["spearman_minimum"]) >= threshold(absolute_spec, "minimum_spearman_min", 0.95)
        and float(final_ranking["top30_overlap_median"]) >= threshold(absolute_spec, "median_top30_min", 0.95)
        and float(final_ranking["top30_overlap_minimum"]) >= threshold(absolute_spec, "minimum_top30_min", 0.8)
    )
    absolute_fidelity_final = status(
        absolute_ok,
        snapshot=1500,
        spearman_median=final_ranking["spearman_median"],
        spearman_median_min=threshold(absolute_spec, "median_spearman_min", 0.99),
        spearman_minimum=final_ranking["spearman_minimum"],
        spearman_minimum_min=threshold(absolute_spec, "minimum_spearman_min", 0.95),
        top30_overlap_median=final_ranking["top30_overlap_median"],
        top30_overlap_median_min=threshold(absolute_spec, "median_top30_min", 0.95),
        top30_overlap_minimum=final_ranking["top30_overlap_minimum"],
        top30_overlap_minimum_min=threshold(absolute_spec, "minimum_top30_min", 0.8),
    )

    ratio_max = threshold(relative_spec, "ratio_max", 1.25)
    teacher_relative_mse_final = status(
        logged_ratio <= ratio_max,
        comparison="step1500 / step500 on all 8 held-out logged action prefixes",
        step1500_mean_relative_mse=logged["step_1500"]["mean_relative_mse"],
        step500_mean_relative_mse=logged["step_500"]["mean_relative_mse"],
        ratio=logged_ratio,
        ratio_max=ratio_max,
    )
    causality_max = threshold(causality_spec, "maximum_future_action_leakage_abs", 1e-6)
    causality_final = status(
        leakage_max <= causality_max,
        snapshot=1500,
        maximum_abs_delta=leakage_max,
        maximum_future_action_leakage_abs=causality_max,
    )
    latency_min = threshold(latency_spec, "predictor_only_reduction_min", 0.2)
    latency_final = status(
        float(latency["median_reduction"]) >= latency_min,
        snapshot=1500,
        median_reduction=latency["median_reduction"],
        predictor_only_reduction_min=latency_min,
    )
    full_ok = all(
        item["status"] == "PASS"
        for item in (capacity, absolute_fidelity_final, teacher_relative_mse_final, causality_final, latency_final)
    )
    return {
        "schema": "jepa-action-prefix-compiler.optimization-length-gates",
        "frozen_spec": source,
        "capacity_and_integrity": capacity,
        "optimization_materially_helps": optimization_materially_helps,
        "absolute_fidelity_final": absolute_fidelity_final,
        "teacher_relative_mse_final": teacher_relative_mse_final,
        "causality_final": causality_final,
        "latency_final": latency_final,
        "decision_levels": {
            "optimization_materially_helps": "PASS" if effect_ok else "FAIL",
            "full_replacement": "GO" if full_ok else "NO-GO",
        },
        "optimization_materially_helps_decision": "PASS" if effect_ok else "FAIL",
        "full_replacement_decision": "GO" if full_ok else "NO-GO",
        "overall": "GO" if full_ok else "NO-GO",
    }


def main() -> int:
    args = _args()
    _require_compute_node()
    root = args.root.resolve()
    asset_root = (args.asset_root or root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    protocol = _load_json(args.protocol.resolve())
    settings = _optimization_settings(freeze, protocol)
    freeze_view = _query_view_for_optimization(freeze, protocol)
    query = _query_spec(protocol, freeze_view)
    _ensure_optimization_freeze(freeze, query, settings)
    _cross_check_runtime_contract(query, settings)
    randomness = _frozen_randomness(protocol, freeze_view, settings)
    manifest = _load_query_manifest(args.manifest.resolve(), settings)
    _require_assets(asset_root, freeze)
    if settings["eval_batch"] < 30:
        raise ValueError("held-out candidate batch must be at least top-k=30")
    _gpu_snapshot(output, "start")

    import torch

    _set_seed(settings["train_seed"])
    model, workspace, _anchors, _goals, objective_fn, action_dim, device = _load_official(
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
    if int(settings["packed_action_token_dim"]) != int(settings["frame_skip"]) * int(settings["action_dim"]):
        raise RuntimeError("frozen packed_action_token_dim does not equal frameskip x primitive_action_dim")
    expected_action_dim = int(settings["packed_action_token_dim"])
    if int(action_dim) != expected_action_dim:
        raise ValueError(
            f"official packed action token dim {action_dim} does not match frozen {expected_action_dim}"
        )
    del workspace, _anchors, _goals
    train_dset, heldout_dset = _load_trajectory_datasets(
        root, asset_root, freeze, args.checkpoint, args.checkpoint_config
    )
    train_examples = manifest["splits"]["train"]["examples"]
    heldout_examples = manifest["splits"]["heldout"]["examples"]
    if len(heldout_examples) < query["evaluation_contexts"]:
        raise ValueError("held-out manifest does not contain the required evaluation contexts")
    cache = _RawEpisodeCache(settings["cache_episodes"])
    train_encoded = _preencode_manifest(
        train_dset,
        "train",
        train_examples,
        settings["frame_skip"],
        settings["preencode_chunk"],
        cache,
        model,
        device,
    )
    heldout_encoded = _preencode_manifest(
        heldout_dset,
        "heldout",
        heldout_examples,
        settings["frame_skip"],
        settings["preencode_chunk"],
        cache,
        model,
        device,
    )
    visual_dim = int(train_encoded["context"]["visual"].shape[-1])
    proprio_dim = int(train_encoded["context"]["proprio"].shape[-1])
    student = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, settings["hidden_dim"]).to(device)
    student_integrity = _student_integrity(student, model)
    optimizer = torch.optim.AdamW(student.parameters(), lr=settings["lr"])
    narrow_episode_ids: list[int] = []
    for example in train_examples:
        episode_id = int(example["episode_id"])
        if episode_id not in narrow_episode_ids:
            narrow_episode_ids.append(episode_id)
        if len(narrow_episode_ids) == 2:
            break
    narrow_indices = [
        index for index, example in enumerate(train_examples)
        if int(example["episode_id"]) in narrow_episode_ids
    ]
    if len(narrow_episode_ids) < 2 or not narrow_indices:
        raise ValueError("manifest train split must contain at least two episodes for schedule burn-in")

    # Generate the exact old 500-step WIDE-QUERY prefix, then continue it.
    wide_schedule = _wide_schedule(
        train_examples,
        narrow_indices,
        settings["steps"],
        settings["batch"],
        randomness["context_schedule_seed"],
        settings["historical_steps"],
    )
    role_schedule = _make_role_schedule(
        settings["steps"], settings["batch"], randomness["role_schedule_seed"], query
    )
    action_banks = _precompute_action_banks(
        model,
        train_encoded,
        objective_fn,
        action_dim,
        query,
        randomness["action_query_seed"],
        device,
        settings["preencode_chunk"],
    )
    _gpu_snapshot(output, "precompute_complete")

    losses: list[dict[str, float]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    checkpoint_paths: dict[str, str] = {}
    schedule_metadata = _schedule_metadata(settings, randomness, query)
    telemetry_period = max(1, settings["steps"] // 10)
    for step_index in range(settings["steps"]):
        context, actions, target = _materialize_bank_batch(
            train_encoded,
            action_banks,
            wide_schedule[step_index],
            role_schedule[step_index],
            device,
        )
        losses.append(_train_one_step(student, optimizer, context, actions, target, step_index + 1))
        step = step_index + 1
        if step in settings["checkpoints"]:
            state = _cpu_state_dict(student)
            snapshots[f"step_{step}"] = state
            filename = f"wide_query_step{step:04d}.pt"
            checkpoint_payload = {
                "schema": "jepa-action-prefix-compiler.dino-pusht-optimization-length-checkpoint",
                "global_step": step,
                "step": step,
                "hidden_dim": settings["hidden_dim"],
                "state_dict": state,
                "optimizer_state_dict": _cpu_clone(optimizer.state_dict()),
                "frozen_schedule_metadata": schedule_metadata,
                "rng_state": _checkpoint_rng_state(),
                "manifest": str(args.manifest.resolve()),
                "parent_freeze": str(args.freeze.resolve()),
                "parent_freeze_schema": str(freeze.get("schema", "")),
                "parent_freeze_experiment_id": freeze.get("experiment_id"),
            }
            torch.save(
                checkpoint_payload,
                output / filename,
            )
            checkpoint_paths[f"step_{step}"] = filename
        if step == 1 or step % telemetry_period == 0:
            _gpu_snapshot(output, f"train_step_{step}")
    if tuple(snapshots) != SNAPSHOT_NAMES:
        raise RuntimeError(f"missing required snapshots: {tuple(snapshots)}")

    planner, effect_data = _evaluate_snapshots(
        snapshots,
        model,
        heldout_encoded,
        objective_fn,
        action_dim,
        query["evaluation_contexts"],
        query["evaluation_action_seeds"][:2],
        settings["eval_batch"],
        device,
    )
    logged_action_fidelity = _evaluate_logged_action_snapshots(
        snapshots,
        model,
        heldout_encoded,
        action_dim,
        query["evaluation_contexts"],
        device,
    )
    final_causality = _leakage_test(
        student,
        {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()},
        action_dim,
        query["evaluation_action_seeds"][0] + 9000,
        device,
        settings["future_action_tolerance"],
    )
    final_latency = _latency(
        student,
        model,
        {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()},
        action_dim,
        settings["timing_seed"],
        settings["timing_batch"],
        settings["warmup"],
        settings["repeats"],
        device,
    )
    _gpu_snapshot(output, "complete")

    values = [float(item["latent_mse"]) for item in losses]
    result: dict[str, Any] = {
        "train": {
            "steps": settings["steps"],
            "batch": settings["batch"],
            "optimizer": "AdamW",
            "learning_rate": settings["lr"],
            "optimizer_reinitialized": False,
            "loss_first": values[0],
            "loss_last": values[-1],
            "latent_mse_median_last_10": _median(values[-10:]),
            "latent_last10_to_first_ratio": _safe_ratio(_median(values[-10:]), values[0]),
            "per_step": losses,
        },
        "snapshots": planner,
        "logged_action_teacher_relative": logged_action_fidelity,
        "paired_effects": effect_data["paired_effects"],
        "causality_final": final_causality,
        "latency_final": final_latency,
        "snapshot_outputs_finite": _all_finite(planner) and _all_finite(logged_action_fidelity),
        "snapshot_state_dicts_finite": _all_finite(snapshots),
        "student_integrity": student_integrity,
        "shape_dtype_device_match": all(bool(item["shape_dtype_device_match"]) for item in losses),
        "fallback_used": False,
        "finite_outputs_and_training": _all_finite(
            {
                "losses": losses,
                "snapshots": planner,
                "logged_action_teacher_relative": logged_action_fidelity,
                "effects": effect_data["paired_effects"],
                "causality": final_causality,
                "latency": final_latency,
            }
        ),
    }
    gates = _frozen_gates(freeze, result, final_causality, final_latency, settings)
    result["gates"] = gates
    summary = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-optimization-length-summary",
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "freeze": str(args.freeze.resolve()),
        "manifest": str(args.manifest.resolve()),
        "source": {
            "student": "NativeDinoPrefixStudent hidden_dim=128",
            "teacher_target": "same frozen teacher rollout dense targets as WIDE-QUERY",
            "action_bank": "precomputed once before the first student update",
        },
        "contracts": {
            "single_arm": "WIDE-QUERY",
            "continuous_optimizer": True,
            "optimizer_reinitialized": False,
            "total_steps": settings["steps"],
            "snapshot_steps": list(settings["checkpoints"]),
            "checkpoint_filenames": checkpoint_paths,
            "checkpoint_use": "audit artifacts only; training continued with the same live optimizer and no snapshot reload",
            "context_schedule_seed": randomness["context_schedule_seed"],
            "old_wide_query_burn_in": {
                "narrow_draws": settings["historical_steps"] * settings["batch"],
                "narrow_steps": settings["historical_steps"],
            },
            "wide_schedule_steps": settings["steps"],
            "gpu_telemetry_seconds": settings["telemetry_seconds"],
            "role_schedule_seed": randomness["role_schedule_seed"],
            "first_500_role_permutations_reused": True,
            "heldout_evaluation_contexts": query["evaluation_contexts"],
            "heldout_evaluation_seeds": query["evaluation_action_seeds"][:2],
            "candidate_count_per_block": settings["eval_batch"],
            "paired_blocks": query["evaluation_contexts"] * 2,
            "query_mixture": {
                "logged": query["fractions"]["logged"],
                "gaussian_planner_init": query["fractions"]["gaussian_planner_init"],
                "one_step_cem_resample": query["fractions"]["one_step_cem_resample"],
            },
        },
        "train": result["train"],
        "snapshots": result["snapshots"],
        "logged_action_teacher_relative": result["logged_action_teacher_relative"],
        "paired_effects": result["paired_effects"],
        "causality_final": result["causality_final"],
        "latency_final": result["latency_final"],
        "snapshot_outputs_finite": result["snapshot_outputs_finite"],
        "snapshot_state_dicts_finite": result["snapshot_state_dicts_finite"],
        "student_integrity": result["student_integrity"],
        "shape_dtype_device_match": result["shape_dtype_device_match"],
        "fallback_used": result["fallback_used"],
        "finite_outputs_and_training": result["finite_outputs_and_training"],
        "gates": gates,
        "unverified": [
            "No closed-loop CEM or environment execution is included.",
            "Held-out candidates are repeated measurements within 16 paired blocks, not independent candidates.",
            "Optimization length is diagnosed only for this frozen WIDE-QUERY student and hidden_dim=128.",
        ],
    }
    summary_path = args.summary.resolve() if args.summary else output / "optimization_length_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2, default=_json_default)
    summary_path.write_text(summary_text, encoding="utf-8")
    default_summary = output / "optimization_length_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(summary_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
