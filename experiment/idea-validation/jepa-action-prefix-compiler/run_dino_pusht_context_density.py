#!/usr/bin/env python3
"""Run the frozen h256 context-support density contrast.

The low-density arm is the read-only h256 result from job 24456882.  This
runner trains only the high-density h256 arm, using the exact density schedule
declared by ``CONTEXT_DENSITY_FREEZE.json``.  It never edits or reruns the
low-density arm and never enters closed-loop control.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_dino_pusht_optimization_length as opt  # noqa: E402
import run_dino_pusht_capacity_contrast as cap  # noqa: E402


DENSITY_SCHEMA = "jepa-action-prefix-compiler.context-support-density-freeze"
SUMMARY_SCHEMA = "jepa-action-prefix-compiler.dino-pusht-capacity-contrast-summary"
WIDTH = 256
LOW_CONTEXTS = 128
HIGH_CONTEXTS = 256
SNAPSHOT_STEPS = (500, 1000, 1500)
SNAPSHOT_NAMES = tuple(f"step_{step}" for step in SNAPSHOT_STEPS)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, default=None)
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
    ok = float(actual) == expected if isinstance(expected, float) else actual == expected
    if not ok:
        raise ValueError(f"frozen field mismatch at {path}.{key}: expected {expected!r}, got {actual!r}")
    return actual


def _first(mapping: Mapping[str, Any], names: Sequence[str], path: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise ValueError(f"missing frozen field {path}.{names[0]}")


def _density_settings(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all fixed values; schedule semantics stay in the freeze."""

    if str(freeze.get("schema", "")) != DENSITY_SCHEMA:
        raise ValueError(f"unexpected context-density freeze schema: {freeze.get('schema')!r}")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("context-density freeze schema_version must be 1")

    baseline = _mapping(freeze, "baseline", "freeze")
    treatment = _mapping(freeze, "treatment", "freeze")
    training = _mapping(freeze, "training", "freeze")
    randomness = _mapping(freeze, "randomness_and_schedule", "freeze")
    bank = _mapping(freeze, "frozen_training_bank", "freeze")
    action_dist = _mapping(bank, "action_query_distribution", "freeze.frozen_training_bank")
    cem = _mapping(bank, "one_step_cem_resample", "freeze.frozen_training_bank")
    heldout = _mapping(freeze, "heldout_evaluation", "freeze")
    ranking = _mapping(heldout, "ranking", "freeze.heldout_evaluation")
    latency = _mapping(heldout, "latency", "freeze.heldout_evaluation")
    gates = _mapping(freeze, "gates", "freeze")
    integrity_gate = _mapping(gates, "context_manifest_integrity", "freeze.gates")
    effect_gate = _mapping(gates, "density_effect", "freeze.gates")
    absolute_gate = _mapping(gates, "absolute_fidelity", "freeze.gates")
    relative_gate = _mapping(gates, "logged_fidelity_noninferiority", "freeze.gates")
    causality_gate = _mapping(gates, "causality", "freeze.gates")
    latency_gate = _mapping(gates, "latency", "freeze.gates")
    execution = _mapping(freeze, "execution_constraints", "freeze")

    selector = _mapping(baseline, "summary_selector", "freeze.baseline")
    _exact(selector, "student_hidden_dim", WIDTH, "freeze.baseline.summary_selector")
    _exact(selector, "context_support_density", "low", "freeze.baseline.summary_selector")
    _exact(selector, "train_contexts", LOW_CONTEXTS, "freeze.baseline.summary_selector")
    _exact(selector, "train_episodes", 32, "freeze.baseline.summary_selector")
    _exact(selector, "contexts_per_episode", 4, "freeze.baseline.summary_selector")
    _exact(baseline, "remote_job_id", "24456882.pbs101", "freeze.baseline")
    _exact(baseline, "summary_schema", SUMMARY_SCHEMA, "freeze.baseline")
    _exact(treatment, "hidden_dim", WIDTH, "freeze.treatment")
    _exact(treatment, "student_class", "NativeDinoPrefixStudent", "freeze.treatment")
    _exact(training, "steps", 1500, "freeze.training")
    _exact(training, "batch_size", 32, "freeze.training")
    _exact(training, "optimizer", "AdamW", "freeze.training")
    _exact(training, "learning_rate", 3e-4, "freeze.training")
    _exact(randomness, "training_seed", 20260925, "freeze.randomness_and_schedule")
    _exact(randomness, "context_schedule_seed", 20260925, "freeze.randomness_and_schedule")
    _exact(randomness, "action_query_seed", 20260926, "freeze.randomness_and_schedule")
    _exact(randomness, "role_schedule_seed", 20260927, "freeze.randomness_and_schedule")
    _exact(randomness, "timing_action_prefix_seed", 20270925, "freeze.randomness_and_schedule")
    heldout_seeds = [int(value) for value in randomness.get("heldout_action_prefix_seeds", [])]
    if heldout_seeds != [20264925, 20264926]:
        raise ValueError("density freeze held-out action-prefix seeds must be [20264925, 20264926]")

    _exact(bank, "train_contexts", HIGH_CONTEXTS, "freeze.frozen_training_bank")
    _exact(bank, "train_episodes", 32, "freeze.frozen_training_bank")
    _exact(bank, "contexts_per_episode", 8, "freeze.frozen_training_bank")
    if {key: float(action_dist[key]) for key in ("logged", "gaussian_planner_init", "one_step_cem_resample")} != {
        "logged": 0.5, "gaussian_planner_init": 0.25, "one_step_cem_resample": 0.25
    }:
        raise ValueError("density freeze query mixture is not frozen 50/25/25")
    if (int(cem["candidate_count_M"]), int(cem["elite_count_K"]), float(cem["variance_floor"])) != (64, 8, 0.05):
        raise ValueError("density freeze CEM bank is not M=64/K=8/variance_floor=0.05")
    _exact(ranking, "block_count", 16, "freeze.heldout_evaluation.ranking")
    _exact(ranking, "heldout_contexts", 8, "freeze.heldout_evaluation.ranking")
    _exact(ranking, "candidates_per_block", 300, "freeze.heldout_evaluation.ranking")
    _exact(ranking, "topk", 30, "freeze.heldout_evaluation.ranking")
    _exact(latency, "batch_size", 300, "freeze.heldout_evaluation.latency")
    _exact(latency, "warmup_repeats", 3, "freeze.heldout_evaluation.latency")
    _exact(latency, "technical_repeats", 10, "freeze.heldout_evaluation.latency")
    _exact(execution, "gpu_telemetry_seconds", 30, "freeze.execution_constraints")
    _exact(effect_gate, "median_spearman_delta_min", 0.05, "freeze.gates.density_effect")
    _exact(effect_gate, "median_top30_delta_min", 0.10, "freeze.gates.density_effect")
    _exact(effect_gate, "positive_spearman_blocks_min", 12, "freeze.gates.density_effect")
    _exact(effect_gate, "positive_top30_blocks_min", 12, "freeze.gates.density_effect")
    _exact(relative_gate, "ratio_max", 1.25, "freeze.gates.logged_fidelity_noninferiority")
    integrity_contract = {
        "all_outputs_finite": True,
        "final_training_finite": True,
        "future_action_leakage_max_abs": 1e-6,
        "last10_to_first_training_mse_ratio_max": 0.8,
        "hidden_encode_obs": False,
        "source_encoder_reference": False,
        "old_context_rows_preserved": True,
        "old_context_order_preserved": True,
        "new_contexts_exclude_old_starts": True,
        "train_episode_set_unchanged": True,
        "validation_split_unchanged": True,
        "heldout_split_unchanged": True,
        "new_trajectory_count": 0,
        "shape_dtype_device_match": True,
        "oom_nan_or_silent_fallback": False,
    }
    for key, expected in integrity_contract.items():
        _exact(integrity_gate, key, expected, "freeze.gates.context_manifest_integrity")
    _exact(causality_gate, "maximum_future_action_leakage_abs", 1e-6, "freeze.gates.causality")
    _exact(latency_gate, "predictor_only_reduction_min", 0.2, "freeze.gates.latency")

    # A density contrast is uninterpretable unless the freeze specifies how
    # the old schedule is preserved and how additional support is introduced.
    schedule = (
        randomness.get("context_density_schedule")
        or randomness.get("density_schedule")
        or freeze.get("context_density_schedule")
    )
    if not isinstance(schedule, Mapping):
        raise ValueError(
            "SCHEDULE_BLOCKER: CONTEXT_DENSITY_FREEZE does not define how the "
            "128 appended contexts enter the 1500 update batches; refusing to "
            "invent a support schedule because it would confound support density "
            "with old-context exposure"
        )
    _exact(schedule, "low_density_schedule_preserved", True, "freeze.context_density_schedule")
    _exact(schedule, "low_density_contexts", LOW_CONTEXTS, "freeze.context_density_schedule")
    _exact(schedule, "high_density_contexts", HIGH_CONTEXTS, "freeze.context_density_schedule")
    support_rows = int(_first(schedule, ("support_rows_per_batch", "additional_rows_per_batch"), "freeze.context_density_schedule"))
    if support_rows != 16:
        raise ValueError("density support_rows_per_batch must remain exactly 16")
    support_seed = int(_first(schedule, ("support_schedule_seed", "high_density_schedule_seed"), "freeze.context_density_schedule"))
    position_policy = str(_first(schedule, ("support_position_policy", "position_policy"), "freeze.context_density_schedule"))
    index_policy = str(_first(schedule, ("support_index_policy", "index_policy"), "freeze.context_density_schedule"))
    if position_policy != "random_without_replacement" or index_policy != "random_with_replacement":
        raise ValueError(
            "density schedule policy is not the explicitly frozen random_without_replacement/random_with_replacement contract"
        )
    historical_steps = int(_first(schedule, ("historical_steps",), "freeze.context_density_schedule"))
    if historical_steps != 500:
        raise ValueError("density schedule historical_steps must remain 500")
    support_range = _first(schedule, ("support_index_range",), "freeze.context_density_schedule")
    if [int(value) for value in support_range] != [LOW_CONTEXTS, HIGH_CONTEXTS - 1]:
        raise ValueError("density schedule support_index_range must be exactly [128, 255]")
    _exact(schedule, "low_schedule_row_count", 32, "freeze.context_density_schedule")
    _exact(schedule, "high_schedule_row_count", 32, "freeze.context_density_schedule")
    _exact(schedule, "role_schedule_independent_and_unchanged", True, "freeze.context_density_schedule")
    _exact(schedule, "position_schedule_independent_of_role_schedule", True, "freeze.context_density_schedule")

    return {
        "steps": 1500, "batch": 32, "lr": 3e-4, "hidden_dim": WIDTH,
        "visual_dim": 384, "proprio_dim": 10, "horizon": 5, "frame_skip": 5,
        "action_dim": 2, "packed_action_token_dim": 10, "historical_steps": 500,
        "checkpoints": SNAPSHOT_STEPS, "evaluation_contexts": 8, "evaluation_blocks": 16,
        "evaluation_topk": 30, "eval_batch": 300, "timing_batch": 300, "warmup": 3,
        "repeats": 10, "telemetry_seconds": 30, "train_seed": 20260925,
        "context_schedule_seed": 20260925, "action_query_seed": 20260926,
        "role_schedule_seed": 20260927, "timing_seed": 20270925,
        "evaluation_action_seeds": heldout_seeds, "capacity_gate": dict(integrity_gate),
        "effect_gate": dict(effect_gate), "absolute_gate": dict(absolute_gate),
        "relative_gate": dict(relative_gate), "causality_gate": dict(causality_gate),
        "latency_gate": dict(latency_gate), "teacher_latency_gate": dict(latency_gate),
        "baseline": dict(baseline), "query": {
            "fractions": {"logged": 0.5, "gaussian_planner_init": 0.25, "one_step_cem_resample": 0.25},
            "cem_candidates": 64, "cem_topk": 8, "cem_variance_floor": 0.05,
        },
        "schedule": {"support_rows": support_rows, "support_seed": support_seed,
                      "position_policy": position_policy, "index_policy": index_policy,
                      "historical_steps": historical_steps,
                      "support_range": [LOW_CONTEXTS, HIGH_CONTEXTS - 1],
                      "role_schedule_independent_and_unchanged": True},
    }


def _canonical_context(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Fields that identify a context without depending on JSON key order."""
    return (
        str(item.get("episode_key", f"{item.get('split', 'train')}:{item['episode_id']}")),
        int(item["episode_id"]), int(item.get("start_step", item.get("start"))),
        tuple(int(value) for value in item["target_steps"]),
        str(item.get("observation_file", "")),
    )


def _contexts_from_split(split: Mapping[str, Any], expected: int, name: str) -> list[dict[str, Any]]:
    contexts = split.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != expected:
        raise ValueError(f"density manifest {name} must contain exactly {expected} contexts")
    result: list[dict[str, Any]] = []
    for raw in contexts:
        if not isinstance(raw, Mapping):
            raise ValueError(f"density manifest {name} contains a non-object context")
        item = dict(raw)
        if "start_step" not in item and "start" in item:
            item["start_step"] = item["start"]
        for key in ("episode_id", "start_step", "target_steps"):
            if key not in item:
                raise ValueError(f"density manifest context missing {key}")
        expected_steps = [int(item["start_step"]) + 5 * (index + 1) for index in range(5)]
        if [int(value) for value in item["target_steps"]] != expected_steps:
            raise ValueError("density manifest target_steps are not contiguous H=5/frame_skip=5 targets")
        result.append(item)
    return result


def _load_density_manifest(path: Path, settings: Mapping[str, Any], legacy_path: Path | None) -> dict[str, Any]:
    manifest = opt._load_json(path.resolve())
    if str(manifest.get("schema", "")) not in {
        "jepa-action-prefix-compiler.query-coverage-manifest",
        "jepa-action-prefix-compiler.context-density-manifest",
    }:
        raise ValueError("density runner requires query-coverage/context-density manifest schema")
    protocol = _mapping(manifest, "protocol", "manifest")
    if int(protocol.get("horizon", -1)) != 5 or int(protocol.get("frame_skip", -1)) != 5:
        raise ValueError("density manifest protocol must remain H=5/frame_skip=5")
    splits = _mapping(manifest, "splits", "manifest")
    train = _contexts_from_split(_mapping(splits, "train", "manifest.splits"), HIGH_CONTEXTS, "train")
    heldout = _contexts_from_split(_mapping(splits, "heldout", "manifest.splits"), 8, "heldout")
    ordinals = [int(item.get("context_ordinal", -1)) for item in train]
    if ordinals != list(range(HIGH_CONTEXTS)):
        raise ValueError("density manifest train ordinals must be exactly 0..255")
    if len({_canonical_context(item) for item in train}) != HIGH_CONTEXTS:
        raise ValueError("density manifest train contexts contain duplicates")
    counts: dict[int, int] = {}
    for item in train:
        counts[int(item["episode_id"])] = counts.get(int(item["episode_id"]), 0) + 1
    if len(counts) != 32 or set(counts.values()) != {8}:
        raise ValueError("density manifest must contain 32 train episodes with exactly eight contexts each")
    heldout_keys = {_canonical_context(item) for item in heldout}
    if heldout_keys & {_canonical_context(item) for item in train}:
        raise ValueError("density manifest train/held-out contexts overlap")

    if legacy_path is None:
        legacy_ref = protocol.get("legacy_manifest") or manifest.get("legacy_manifest")
        if legacy_ref:
            legacy_path = (path.parent / str(legacy_ref)).resolve()
    if legacy_path is None or not legacy_path.exists():
        raise ValueError("density manifest requires an existing legacy 128-context manifest for prefix verification")
    legacy = opt._load_json(legacy_path.resolve())
    legacy_splits = _mapping(legacy, "splits", "legacy_manifest")
    old_train = _contexts_from_split(_mapping(legacy_splits, "train", "legacy_manifest.splits"), LOW_CONTEXTS, "legacy train")
    old_ordinals = [int(item.get("context_ordinal", -1)) for item in old_train]
    if old_ordinals != list(range(LOW_CONTEXTS)):
        raise ValueError("legacy manifest train ordinals are not exactly 0..127")
    old_rows_preserved = train[:LOW_CONTEXTS] == old_train
    if not old_rows_preserved:
        raise ValueError("density manifest first 128 train entries are not the immutable legacy prefix")
    if any(int(item.get("context_ordinal", -1)) != 128 + index for index, item in enumerate(train[LOW_CONTEXTS:])):
        raise ValueError("density manifest appended train ordinals are not 128..255")
    old_episode_ids = {int(item["episode_id"]) for item in old_train}
    train_episode_ids = {int(item["episode_id"]) for item in train}
    old_starts_by_episode = {
        episode_id: {int(item["start_step"]) for item in old_train if int(item["episode_id"]) == episode_id}
        for episode_id in old_episode_ids
    }
    new_starts_disjoint = all(
        int(item["start_step"]) not in old_starts_by_episode[int(item["episode_id"])]
        for item in train[LOW_CONTEXTS:]
    )
    legacy_heldout = _contexts_from_split(
        _mapping(legacy_splits, "heldout", "legacy_manifest.splits"), 8, "legacy heldout"
    )
    heldout_unchanged = heldout == legacy_heldout
    if not new_starts_disjoint or train_episode_ids != old_episode_ids or not heldout_unchanged:
        raise ValueError("density manifest changes old starts, train episodes, or held-out rows")

    normalized = dict(manifest)
    normalized["splits"] = {
        **splits,
        "train": {**_mapping(splits, "train", "manifest.splits"), "examples": train},
        "heldout": {**_mapping(splits, "heldout", "manifest.splits"), "examples": heldout},
    }
    normalized["density_validation"] = {
        "status": "PASS", "legacy_manifest": str(legacy_path.resolve()),
        "legacy_prefix_contexts": LOW_CONTEXTS, "new_support_contexts": LOW_CONTEXTS,
        "train_ordinals": "0..127 prefix + 128..255 appended", "contexts_per_episode": 8,
        "duplicate_check": "PASS",
        "old_context_rows_preserved": old_rows_preserved,
        "old_context_order_preserved": old_rows_preserved,
        "new_contexts_exclude_old_starts": new_starts_disjoint,
        "train_episode_set_unchanged": train_episode_ids == old_episode_ids,
        "validation_split_unchanged": heldout_unchanged,
        "heldout_split_unchanged": heldout_unchanged,
        "new_trajectory_count": 0,
    }
    return normalized


def _density_schedules(
    high_examples: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    """Create paired schedules only from the explicit frozen support contract."""
    old_examples = high_examples[:LOW_CONTEXTS]
    episode_ids: list[int] = []
    for item in old_examples:
        episode = int(item["episode_id"])
        if episode not in episode_ids:
            episode_ids.append(episode)
        if len(episode_ids) == 2:
            break
    narrow_indices = [i for i, item in enumerate(old_examples) if int(item["episode_id"]) in episode_ids]
    if len(episode_ids) < 2 or not narrow_indices:
        raise ValueError("density manifest cannot establish the historical narrow schedule prefix")
    schedule = settings["schedule"]
    low = opt._wide_schedule(old_examples, narrow_indices, 1500, 32, settings["context_schedule_seed"], schedule["historical_steps"])
    rng = random.Random(schedule["support_seed"])
    high: list[list[int]] = []
    support_start, support_end = settings["schedule"]["support_range"]
    new_indices = list(range(support_start, support_end + 1))
    if new_indices != list(range(128, 256)):
        raise ValueError("density schedule support range is not exactly 128..255")
    support_rows = int(schedule["support_rows"])
    for low_batch in low:
        positions = sorted(rng.sample(range(32), support_rows))
        support = [new_indices[rng.randrange(len(new_indices))] for _ in positions]
        batch = list(low_batch)
        for position, index in zip(positions, support):
            batch[position] = index
        if len(positions) != support_rows or len(set(positions)) != support_rows:
            raise ValueError(f"density schedule must replace exactly {support_rows} distinct batch positions")
        if any(index < 128 or index > 255 for index in support):
            raise ValueError("density schedule support index escaped 128..255")
        if [batch[i] for i in range(32) if i not in positions] != [low_batch[i] for i in range(32) if i not in positions]:
            raise ValueError("density schedule changed a low-density row outside support positions")
        high.append(batch)
    return low, high, {
        "low_schedule": "exact _wide_schedule over immutable first 128 entries",
        "high_schedule": "low schedule with explicitly frozen support rows replaced",
        "support_rows_per_batch": support_rows, "support_seed": schedule["support_seed"],
        "support_position_policy": schedule["position_policy"],
        "support_index_policy": schedule["index_policy"],
        "support_index_range": [128, 255], "low_schedule_row_count": 32,
        "high_schedule_row_count": 32, "low_rows_unchanged_per_batch": 16,
        "low_schedule_prefix_preserved": True,
    }


def _block_map(per_block: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for case in per_block:
        key = (int(case["seed"]), int(case["context_index"]))
        if key in result:
            raise ValueError(f"{label} has duplicate paired block {key}")
        result[key] = case
    return result


def _paired_density_effect(high: Mapping[str, Any], low: Mapping[str, Any], expected_blocks: int = 16) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    exact_keys: list[tuple[int, int]] | None = None
    for snapshot in SNAPSHOT_NAMES:
        high_map = _block_map(high[snapshot]["per_block"], f"high-density {snapshot}")
        low_map = _block_map(low[snapshot]["per_block"], f"low-density {snapshot}")
        if len(high_map) != expected_blocks or set(high_map) != set(low_map):
            raise ValueError(f"paired density block mismatch at {snapshot}")
        keys = sorted(high_map)
        if exact_keys is None:
            exact_keys = keys
        elif keys != exact_keys:
            raise ValueError("paired density block key set changes across snapshots")
        paired = []
        for key in keys:
            h, l = high_map[key], low_map[key]
            if (int(h["candidate_count"]), int(h["topk"])) != (int(l["candidate_count"]), int(l["topk"])):
                raise ValueError(f"candidate contract mismatch for paired block {key}")
            paired.append({
                "block_id": f"seed={key[0]}:context={key[1]}", "seed": key[0], "context_index": key[1],
                "candidate_count": int(h["candidate_count"]), "topk": int(h["topk"]),
                "spearman_delta": float(h["ranking"]["spearman"]) - float(l["ranking"]["spearman"]),
                "top30_overlap_delta": float(h["ranking"]["top30_overlap"]) - float(l["ranking"]["top30_overlap"]),
                "mean_relative_mse_delta": opt._mean(h["teacher_relative"]["relative_mse"])
                - opt._mean(l["teacher_relative"]["relative_mse"]),
            })
        effects[snapshot] = {
            "comparison": f"h256-Dhigh {snapshot} - authoritative h256-Dlow {snapshot}",
            "per_block": paired,
            "spearman_delta_median": opt._median([x["spearman_delta"] for x in paired]),
            "top30_overlap_delta_median": opt._median([x["top30_overlap_delta"] for x in paired]),
            "mean_relative_mse_delta_median": opt._median([x["mean_relative_mse_delta"] for x in paired]),
            "spearman_positive_blocks": sum(x["spearman_delta"] > 0 for x in paired),
            "top30_overlap_positive_blocks": sum(x["top30_overlap_delta"] > 0 for x in paired),
        }
    return {"paired_blocks": expected_blocks, "block_ids": [f"seed={k[0]}:context={k[1]}" for k in exact_keys or []], "snapshots": effects}


def _validate_baseline(summary: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("density baseline must be a capacity-contrast summary")
    source = _mapping(summary, "source", "baseline")
    if str(source.get("student")) != "NativeDinoPrefixStudent hidden_dim=256":
        raise ValueError("density baseline is not the h256 low-density summary")
    contracts = _mapping(summary, "contracts", "baseline")
    for key, expected in {"capacity_variant": "hidden_256", "total_steps": 1500,
                          "heldout_evaluation_contexts": 8, "candidate_count_per_block": 300,
                          "paired_blocks": 16}.items():
        if contracts.get(key) != expected:
            raise ValueError(f"low-density baseline contract mismatch for {key}")
    if [int(x) for x in contracts.get("heldout_evaluation_seeds", [])] != settings["evaluation_action_seeds"]:
        raise ValueError("low-density baseline held-out seeds mismatch")
    if contracts.get("query_mixture") != settings["query"]["fractions"]:
        raise ValueError("low-density baseline query mixture mismatch")
    snapshots = _mapping(summary, "snapshots", "baseline")
    if any(name not in snapshots for name in SNAPSHOT_NAMES):
        raise ValueError("low-density baseline is missing 500/1000/1500 snapshots")
    return {"snapshots": snapshots, "contracts": dict(contracts),
            "metrics": summary.get("logged_action_teacher_relative", {}).get("step_1500", {}),
            "latency": summary.get("latency_final", {})}


def _status(ok: bool, **details: Any) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", **details}


def _gates(settings: Mapping[str, Any], result: Mapping[str, Any], pairing: Mapping[str, Any]) -> dict[str, Any]:
    integrity = result["student_integrity"]
    effect = result["density_effect"]["snapshots"]["step_1500"]
    effect_spec = settings["effect_gate"]
    effect_ok = (
        float(effect["spearman_delta_median"]) >= float(effect_spec["median_spearman_delta_min"])
        and float(effect["top30_overlap_delta_median"]) >= float(effect_spec["median_top30_delta_min"])
        and int(effect["spearman_positive_blocks"]) >= int(effect_spec["positive_spearman_blocks_min"])
        and int(effect["top30_overlap_positive_blocks"]) >= int(effect_spec["positive_top30_blocks_min"])
    )
    density = _status(effect_ok, **effect)
    final = result["snapshots"]["step_1500"]["terminal_ranking"]
    absolute_spec = settings["absolute_gate"]
    absolute_ok = (
        float(final["spearman_median"]) >= float(absolute_spec["median_spearman_min"])
        and float(final["spearman_minimum"]) >= float(absolute_spec["minimum_spearman_min"])
        and float(final["top30_overlap_median"]) >= float(absolute_spec["median_top30_min"])
        and float(final["top30_overlap_minimum"]) >= float(absolute_spec["minimum_top30_min"])
    )
    absolute = _status(absolute_ok, snapshot=1500, metrics=final, thresholds=dict(absolute_spec))
    low_mse = float(result["baseline_metrics"]["mean_relative_mse"])
    high_mse = float(result["logged_action_teacher_relative"]["step_1500"]["mean_relative_mse"])
    ratio = opt._safe_ratio(high_mse, low_mse)
    relative = _status(ratio <= float(settings["relative_gate"]["ratio_max"]), comparison="Dhigh/Dlow logged teacher-relative MSE", high=high_mse, low=low_mse, ratio=ratio, ratio_max=float(settings["relative_gate"]["ratio_max"]))
    leakage = max(float(x["max_abs_delta"]) for x in result["causality_final"]["cases"])
    causality = _status(leakage <= float(settings["causality_gate"]["maximum_future_action_leakage_abs"]), maximum_abs_delta=leakage, threshold=float(settings["causality_gate"]["maximum_future_action_leakage_abs"]))
    latency = result["latency_final"]
    teacher = _status(float(latency["median_reduction"]) >= float(settings["latency_gate"]["predictor_only_reduction_min"]), median_reduction=latency["median_reduction"], threshold=float(settings["latency_gate"]["predictor_only_reduction_min"]))
    integrity_spec = settings["capacity_gate"]
    manifest_integrity = result["manifest_integrity"]
    training_ratio = float(result["train"]["latent_last10_to_first_ratio"])
    manifest_ok = all(
        manifest_integrity[key] == integrity_spec[key]
        for key in (
            "old_context_rows_preserved", "old_context_order_preserved",
            "new_contexts_exclude_old_starts", "train_episode_set_unchanged",
            "validation_split_unchanged", "heldout_split_unchanged",
            "new_trajectory_count",
        )
    )
    integrity_ok = bool(
        result["finite_outputs_and_training"]
        and result["snapshot_outputs_finite"]
        and result["snapshot_state_dicts_finite"]
        and result["shape_dtype_device_match"]
        and integrity["passed"]
        and not result["fallback_used"]
        and pairing["status"] == "PASS"
        and manifest_ok
        and training_ratio <= float(integrity_spec["last10_to_first_training_mse_ratio_max"])
        and leakage <= float(integrity_spec["future_action_leakage_max_abs"])
    )
    cap_gate = _status(
        integrity_ok,
        finite=result["finite_outputs_and_training"],
        student_integrity=integrity,
        pairing=pairing,
        fallback_used=result["fallback_used"],
        manifest_integrity=manifest_integrity,
        manifest_integrity_passed=manifest_ok,
        last10_to_first_training_mse_ratio=training_ratio,
        last10_to_first_training_mse_ratio_max=float(integrity_spec["last10_to_first_training_mse_ratio_max"]),
        future_action_leakage_max_abs=leakage,
    )
    full = all(x["status"] == "PASS" for x in (cap_gate, absolute, relative, causality, teacher))
    density_supported = effect_ok and full
    return {"schema": "jepa-action-prefix-compiler.context-density-gates", "capacity_and_integrity": cap_gate, "density_effect": density, "absolute_fidelity_final": absolute, "teacher_relative_mse_final": relative, "causality_final": causality, "teacher_latency_final": teacher, "decision_levels": {"density_effect": "PASS" if effect_ok else "FAIL", "full_replacement": "GO" if full else "NO-GO", "density_supported_replacement": "GO" if density_supported else "NO-GO"}, "overall": "GO" if full else "NO-GO"}


def main() -> int:
    args = _args()
    opt._require_compute_node()
    root = args.root.resolve(); asset_root = (args.asset_root or root).resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    freeze = opt._load_json(args.freeze.resolve()); settings = _density_settings(freeze)
    baseline_summary = opt._load_json(args.baseline_summary.resolve()); baseline = _validate_baseline(baseline_summary, settings)
    manifest = _load_density_manifest(args.manifest.resolve(), settings, args.legacy_manifest.resolve() if args.legacy_manifest else None)
    opt._require_assets(asset_root, freeze); opt._gpu_snapshot(output, "start")
    import torch
    opt._set_seed(settings["train_seed"])
    model, workspace, _anchors, _goals, objective_fn, action_dim, device = opt._load_official(root, asset_root, freeze, output, settings["train_seed"], config_path=args.config.resolve() if args.config else None, checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None, checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None, data_root=args.data_root.resolve() if args.data_root else None)
    if int(action_dim) != settings["packed_action_token_dim"]: raise ValueError("official action token dim mismatch")
    del workspace, _anchors, _goals
    train_dset, heldout_dset = opt._load_trajectory_datasets(root, asset_root, freeze, args.checkpoint, args.checkpoint_config)
    train_examples = manifest["splits"]["train"]["examples"]; heldout_examples = manifest["splits"]["heldout"]["examples"]
    cache = opt._RawEpisodeCache(8)
    train_encoded = opt._preencode_manifest(train_dset, "train", train_examples, 5, 8, cache, model, device)
    heldout_encoded = opt._preencode_manifest(heldout_dset, "heldout", heldout_examples, 5, 8, cache, model, device)
    visual_dim = int(train_encoded["context"]["visual"].shape[-1]); proprio_dim = int(train_encoded["context"]["proprio"].shape[-1])
    if (visual_dim, proprio_dim) != (384, 10): raise ValueError("runtime native latent dims do not match frozen contract")
    initialization_seed_observed = int(torch.initial_seed())
    student = opt.NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, WIDTH).to(device); optimizer = torch.optim.AdamW(student.parameters(), lr=settings["lr"])
    low_schedule, high_schedule, schedule_meta = _density_schedules(train_examples, settings)
    role_schedule = opt._make_role_schedule(1500, 32, settings["role_schedule_seed"], settings["query"])
    action_banks = opt._precompute_action_banks(model, train_encoded, objective_fn, action_dim, settings["query"], settings["action_query_seed"], device, 8)
    opt._gpu_snapshot(output, "precompute_complete")
    losses: list[dict[str, float]] = []; snapshots: dict[str, dict[str, Any]] = {}; checkpoint_paths: dict[str, str] = {}
    for step_index in range(1500):
        context, actions, target = opt._materialize_bank_batch(train_encoded, action_banks, high_schedule[step_index], role_schedule[step_index], device)
        losses.append(opt._train_one_step(student, optimizer, context, actions, target, step_index + 1)); step = step_index + 1
        if step in SNAPSHOT_STEPS:
            state = opt._cpu_state_dict(student); snapshots[f"step_{step}"] = state; filename = f"context_density_h256_step{step:04d}.pt"
            torch.save({"schema": "jepa-action-prefix-compiler.dino-pusht-context-density-checkpoint", "step": step, "hidden_dim": WIDTH, "state_dict": state, "optimizer_state_dict": opt._cpu_clone(optimizer.state_dict()), "manifest": str(args.manifest.resolve()), "parent_freeze": str(args.freeze.resolve()), "initialization_seed_observed": initialization_seed_observed}, output / filename); checkpoint_paths[f"step_{step}"] = filename
        if step == 1 or step % 150 == 0: opt._gpu_snapshot(output, f"train_step_{step}")
    if tuple(snapshots) != SNAPSHOT_NAMES: raise RuntimeError(f"missing snapshots: {tuple(snapshots)}")
    planner, _blocks = cap._evaluate_snapshots_dynamic(snapshots, model, heldout_encoded, objective_fn, action_dim, 8, settings["evaluation_action_seeds"], 300, device, WIDTH)
    logged = cap._evaluate_logged_dynamic(snapshots, model, heldout_encoded, action_dim, 8, device, WIDTH)
    causality = opt._leakage_test(student, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["evaluation_action_seeds"][0] + 9000, device, 1e-6)
    latency = opt._latency(student, model, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["timing_seed"], 300, 3, 10, device); opt._gpu_snapshot(output, "complete")
    density_effect = _paired_density_effect(planner, baseline["snapshots"])
    low_latency = float(baseline["latency"].get("student_median_ms", "nan")); low_teacher = float(baseline["latency"].get("teacher_median_ms", "nan"))
    latency_diagnostic = {"h256_Dhigh_student_median_ms": float(latency["student_median_ms"]), "h256_Dlow_student_median_ms": low_latency, "Dhigh_Dlow_student_latency_ratio": opt._safe_ratio(float(latency["student_median_ms"]), low_latency), "h256_Dhigh_teacher_median_ms": float(latency["teacher_median_ms"]), "h256_Dlow_teacher_median_ms": low_teacher, "teacher_latency_ratio": opt._safe_ratio(float(latency["teacher_median_ms"]), low_teacher), "student_ratio_gate": None, "teacher_same_boundary": True}
    values = [float(item["latent_mse"]) for item in losses]
    result: dict[str, Any] = {"train": {"steps": 1500, "batch": 32, "optimizer": "AdamW", "learning_rate": 3e-4, "loss_first": values[0], "loss_last": values[-1], "latent_mse_median_last_10": opt._median(values[-10:]), "latent_last10_to_first_ratio": opt._safe_ratio(opt._median(values[-10:]), values[0]), "per_step": losses}, "snapshots": planner, "logged_action_teacher_relative": logged, "density_effect": density_effect, "baseline_metrics": baseline["metrics"], "causality_final": causality, "latency_final": latency, "latency_diagnostic": latency_diagnostic, "manifest_integrity": manifest["density_validation"], "student_integrity": opt._student_integrity(student, model), "snapshot_outputs_finite": opt._all_finite(planner) and opt._all_finite(logged), "snapshot_state_dicts_finite": opt._all_finite(snapshots), "shape_dtype_device_match": all(bool(item["shape_dtype_device_match"]) for item in losses), "fallback_used": False}
    result["finite_outputs_and_training"] = opt._all_finite(result)
    pairing = {"status": "PASS", "paired_blocks": 16, "block_ids": density_effect["block_ids"], "candidate_contract": {"count": 300, "topk": 30, "seeds": settings["evaluation_action_seeds"], "generator": "CPU torch.Generator", "order": "seed outer, context_index inner"}}
    gates = _gates(settings, result, pairing)
    summary = {"schema": "jepa-action-prefix-compiler.dino-pusht-context-density-summary", "schema_version": 1, "protocol": str(args.freeze.resolve()), "freeze": str(args.freeze.resolve()), "manifest": str(args.manifest.resolve()), "legacy_manifest": str((args.legacy_manifest or Path("unknown")).resolve()), "baseline_summary": str(args.baseline_summary.resolve()), "source": {"student": "NativeDinoPrefixStudent hidden_dim=256", "teacher_target": "same frozen teacher rollout dense targets", "density_variant": "Dhigh", "baseline_variant": "Dlow"}, "contracts": {"density_variant": "h256-Dhigh", "baseline_variant": "h256-Dlow", "train_contexts": 256, "train_episodes": 32, "contexts_per_episode": 8, "total_steps": 1500, "snapshot_steps": list(SNAPSHOT_STEPS), "heldout_evaluation_contexts": 8, "heldout_evaluation_seeds": settings["evaluation_action_seeds"], "candidate_count_per_block": 300, "paired_blocks": 16, "query_mixture": settings["query"]["fractions"], "same_initialization_procedure": True, "initialization_seed_observed": initialization_seed_observed, "gpu_telemetry_seconds": 30}, "density_validation": manifest["density_validation"], "schedule": schedule_meta, "parameter_report": cap._parameter_report(student, optimizer), "train": result["train"], "snapshots": result["snapshots"], "logged_action_teacher_relative": result["logged_action_teacher_relative"], "baseline_metrics": result["baseline_metrics"], "density_effect": result["density_effect"], "pairing_integrity": pairing, "causality_final": result["causality_final"], "latency_final": result["latency_final"], "latency_diagnostic": latency_diagnostic, "student_integrity": result["student_integrity"], "snapshot_outputs_finite": result["snapshot_outputs_finite"], "snapshot_state_dicts_finite": result["snapshot_state_dicts_finite"], "shape_dtype_device_match": result["shape_dtype_device_match"], "fallback_used": False, "finite_outputs_and_training": result["finite_outputs_and_training"], "checkpoint_filenames": checkpoint_paths, "gates": gates, "timing_boundary": "predictor-level cached native observation latent plus normalized action prefix; excludes encoder, CEM execution, environment and closed-loop control"}
    summary_path = (args.summary or output / "context_density_summary.json").resolve(); summary_path.parent.mkdir(parents=True, exist_ok=True); text = json.dumps(summary, indent=2, default=opt._json_default); summary_path.write_text(text, encoding="utf-8"); default_summary = output / "context_density_summary.json"; default_summary.write_text(text, encoding="utf-8") if summary_path != default_summary else None
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
