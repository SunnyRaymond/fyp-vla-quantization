#!/usr/bin/env python3
"""Run the paired query-slate planner-ranking experiment.

The runner reuses the completed Dhigh 256-context manifest and reports the
completed Dhigh result only as an external diagnostic.  Its own frozen
schedule selects eight distinct contexts per update.  For every selected
context the real logged query is kept as the primary member of a four-way
slate, followed by two deterministic Gaussian proposals and one independent
one-step-CEM proposal.  Control and treatment see the identical slate and
teacher targets in every update; the treatment adds the previously validated
planner-score listwise KL term.

This is a predictor-level experiment.  Observation encoding, CEM execution,
environment interaction, and closed-loop control are outside its timing
boundary.  All model/data work is guarded for a PBS compute allocation.
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

import run_dino_pusht_context_density as density  # noqa: E402
import run_dino_pusht_optimization_length as opt  # noqa: E402
import run_dino_pusht_query_coverage as query_cov  # noqa: E402
from run_dino_pusht_grounded_prefix import (  # noqa: E402
    _RawEpisodeCache,
    _load_trajectory_datasets,
    _preencode_manifest,
)
from run_dino_pusht_rankdistill import _latent_mse  # noqa: E402
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
    _require_assets,
    _require_compute_node,
    _sample_actions,
    _set_seed,
    _spearman,
    _teacher_targets,
    _topk_overlap,
)


SLATE_SIZE = 4
WIDTH = 256
SNAPSHOT_STEPS = (500, 1000, 1500)
SNAPSHOT_NAMES = tuple(f"step_{step}" for step in SNAPSHOT_STEPS)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True, help="query-slate freeze")
    parser.add_argument("--protocol", type=Path, default=None, help="optional query-slate protocol alias")
    parser.add_argument("--manifest", type=Path, required=True, help="256-context density manifest")
    parser.add_argument("--legacy-manifest", type=Path, default=None)
    parser.add_argument("--context-density-freeze", type=Path, default=None)
    parser.add_argument("--density-summary", type=Path, default=None,
                        help="authoritative Dhigh summary, diagnostic only")
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


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: Any) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _settings(freeze: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Bind runtime values to ``QUERY_SLATE_RANK_FREEZE.json``.

    The protocol markdown is deliberately human-readable and optional; the
    machine-readable freeze is the source of truth for this runner.
    """

    training = _mapping(freeze, "training", "freeze")
    batch_semantics = _mapping(training, "batch_semantics", "freeze.training")
    scope = _mapping(freeze, "scope", "freeze")
    slate = _mapping(freeze, "action_query_slate", "freeze")
    cem = _mapping(slate, "one_step_cem_resample", "freeze.action_query_slate")
    rank = _mapping(freeze, "planner_aware_loss", "freeze")
    heldout = _mapping(freeze, "heldout_evaluation", "freeze")
    ranking_eval = _mapping(heldout, "ranking", "freeze.heldout_evaluation")
    latency_eval = _mapping(heldout, "latency", "freeze.heldout_evaluation")
    gates = _mapping(freeze, "gates", "freeze")
    execution = _mapping(freeze, "execution_constraints", "freeze")
    training_seed = int(training.get("training_seed", 20260930))
    randomness = {
        "context_schedule_seed": int(_mapping(freeze, "context_schedule", "freeze").get("schedule_seed", 20260931)),
        "action_slate_seed": int(training.get("action_slate_seed", 20260932)),
        "cem_seed": int(training.get("cem_seed", 20260933)),
    }
    heldout_seeds = [int(x) for x in ranking_eval.get("fresh_action_prefix_seeds", [20264925, 20264926])]
    settings = {
        "steps": int(training.get("steps", 1500)),
        "batch": int(batch_semantics.get("slates_per_update", 8)),
        "effective_batch": int(batch_semantics.get("effective_batch_query_rows", 32)),
        "lr": float(training.get("learning_rate", 3e-4)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "betas": tuple(float(x) for x in training.get("betas", [0.9, 0.999])),
        "eps": float(training.get("eps", 1e-8)),
        "hidden_dim": int(scope.get("student_hidden_dim", WIDTH)),
        "rank_weight": float(rank.get("lambda", 0.1)),
        "rank_temperature": float(rank.get("temperature", 1.0)),
        "initialization_seed": int(_mapping(freeze, "arms", "freeze").get("initialization", {}).get("initialization_seed", 99)),
        "training_seed": training_seed,
        "context_schedule_seed": randomness["context_schedule_seed"],
        "action_slate_seed": randomness["action_slate_seed"],
        "cem_seed": randomness["cem_seed"],
        "timing_seed": int(training.get("timing_action_prefix_seed", 20270925)),
        "heldout_seeds": heldout_seeds,
        "eval_batch": int(ranking_eval.get("candidates_per_block", 300)),
        "eval_contexts": int(ranking_eval.get("contexts", 8)),
        "eval_topk": int(ranking_eval.get("topk", 30)),
        "timing_batch": int(latency_eval.get("batch_size", 300)),
        "warmup": int(latency_eval.get("warmup_repeats", 3)),
        "repeats": int(latency_eval.get("technical_repeats", 10)),
        "telemetry_seconds": int(execution.get("gpu_telemetry_seconds", 30)),
        "gates": gates,
    }
    if settings["steps"] != 1500 or settings["batch"] != 8 or settings["effective_batch"] != 32:
        raise ValueError("query-slate is frozen to 1500 updates with 8 slates and 32 query rows per update")
    if settings["hidden_dim"] != WIDTH:
        raise ValueError("query-slate student hidden_dim must be 256")
    if settings["eval_batch"] != 300 or settings["eval_contexts"] != 8 or settings["eval_topk"] != 30:
        raise ValueError("query-slate held-out contract must be 8 contexts x 2 seeds x 300 candidates, top-k=30")
    if settings["rank_weight"] != 0.1 or settings["rank_temperature"] != 1.0:
        raise ValueError("query-slate listwise KL must use frozen weight=0.1 and temperature=1.0")
    if (
        settings["initialization_seed"], settings["training_seed"],
        settings["context_schedule_seed"], settings["action_slate_seed"],
        settings["cem_seed"], settings["timing_seed"],
    ) != (99, 20260930, 20260931, 20260932, 20260933, 20270925):
        raise ValueError("query-slate RNG seeds differ from the frozen contract")
    if settings["lr"] != 3e-4:
        raise ValueError("query-slate learning rate must remain 3e-4")
    if heldout_seeds != [20264925, 20264926]:
        raise ValueError("query-slate held-out seeds must remain [20264925, 20264926]")
    if str(freeze.get("schema", "")) != "jepa-action-prefix-compiler.query-slate-rank-freeze" or int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unexpected query-slate freeze schema/version")
    if int(scope.get("horizon", -1)) != 5 or int(scope.get("primitive_action_dim", -1)) != 2 or int(scope.get("packed_action_token_dim", -1)) != 10:
        raise ValueError("query-slate action contract must remain horizon=5, primitive=2, packed=10")
    if [int(x) for x in slate.get("primitive_action_shape", [])] != [5, 2] or [int(x) for x in slate.get("packed_action_token_shape", [])] != [5, 10]:
        raise ValueError("query-slate primitive/packed action shapes are not frozen [5,2]/[5,10]")
    if int(slate.get("queries_per_context", -1)) != 4 or int(slate.get("slate_group_count_per_update", -1)) != 8:
        raise ValueError("query-slate structure must remain 8 context groups x 4 queries")
    if list(slate.get("slot_order", [])) != ["primary_logged", "gaussian_planner_init_a", "gaussian_planner_init_b", "one_step_cem_resample"]:
        raise ValueError("query-slate slot order differs from the frozen contract")
    if (int(cem.get("candidate_count_M", -1)), int(cem.get("elite_count_K", -1)), float(cem.get("variance_floor", -1))) != (64, 8, 0.05):
        raise ValueError("query-slate CEM proposal must remain M=64/K=8/variance_floor=0.05")
    if list(training.get("snapshot_steps", [])) != list(SNAPSHOT_STEPS):
        raise ValueError("query-slate snapshots must remain 500/1000/1500")
    if str(training.get("optimizer", "")) != "AdamW" or settings["weight_decay"] != 0.01 or settings["betas"] != (0.9, 0.999) or settings["eps"] != 1e-8:
        raise ValueError("query-slate AdamW hyperparameters differ from the frozen contract")
    if training.get("scheduler", "not-frozen") is not None or settings["telemetry_seconds"] != 30:
        raise ValueError("query-slate scheduler/telemetry contract mismatch")
    if (int(batch_semantics.get("microbatch_query_rows", -1)), int(batch_semantics.get("gradient_accumulation_steps", -1)), int(batch_semantics.get("optimizer_steps_per_update_per_arm", -1))) != (32, 1, 1):
        raise ValueError("query-slate microbatch/update semantics differ from the frozen contract")
    return settings


def _density_schedule_settings(freeze: Mapping[str, Any], context_freeze: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expose only the Dhigh schedule fields needed by the shared helper."""

    if context_freeze is not None:
        full = density._density_settings(context_freeze)
        return {
            "context_schedule_seed": full["context_schedule_seed"],
            "schedule": full["schedule"],
            "query": full["query"],
        }
    randomness = context_freeze.get("randomness_and_schedule", {}) if isinstance(context_freeze, Mapping) else {}
    if not isinstance(randomness, Mapping):
        randomness = {}
    compatibility = freeze.get("context_density_compatibility", {})
    if not isinstance(compatibility, Mapping):
        compatibility = {}
    schedule = compatibility.get("context_density_schedule", freeze.get("context_density_schedule", {}))
    if not isinstance(schedule, Mapping):
        schedule = {}
    schedule = dict(schedule)
    schedule.setdefault("historical_steps", 500)
    schedule.setdefault("support_rows", schedule.get("support_rows_per_batch", 16))
    schedule.setdefault("support_seed", schedule.get("support_schedule_seed", 20260929))
    schedule.setdefault("position_policy", schedule.get("support_position_policy", "random_without_replacement"))
    schedule.setdefault("index_policy", schedule.get("support_index_policy", "random_with_replacement"))
    schedule.setdefault("support_range", schedule.get("support_index_range", [128, 255]))
    query = {
        "fractions": {"logged": 0.5, "gaussian_planner_init": 0.25, "one_step_cem_resample": 0.25},
        "cem_candidates": 64, "cem_topk": 8, "cem_variance_floor": 0.05,
    }
    return {
        "context_schedule_seed": int(compatibility.get("context_schedule_seed", randomness.get("context_schedule_seed", 20260925))),
        "schedule": schedule,
        "query": query,
    }


def _slate_cem_action(
    model: Any,
    encoded: Mapping[str, Any],
    index: int,
    objective_fn: Any,
    action_dim: int,
    seed: int,
    device: Any,
    candidate_count: int = 64,
    topk: int = 8,
    variance_floor: float = 0.05,
) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    candidate = _sample_actions(generator, candidate_count, action_dim, torch.device("cpu"))
    context_one = {key: value[index : index + 1].to(device) for key, value in encoded["context"].items()}
    context_candidates = {key: value.expand((candidate_count,) + tuple(value.shape[1:])) for key, value in context_one.items()}
    goal_one = {key: value[index : index + 1, -1:].to(device) for key, value in encoded["grounded"].items()}
    goal_candidates = {key: value.expand((candidate_count,) + tuple(value.shape[1:])) for key, value in goal_one.items()}
    with torch.no_grad():
        target = _teacher_targets(model, context_candidates, candidate.to(device))
        cost = _objective_cost(objective_fn, target, goal_candidates).reshape(-1)
    elite = candidate.index_select(0, torch.argsort(cost).detach().cpu()[:topk])
    mean = elite.mean(dim=0)
    variance = elite.var(dim=0, unbiased=False).clamp_min(variance_floor)
    return (mean + variance.sqrt() * _sample_actions(generator, 1, action_dim, torch.device("cpu"))[0]).cpu()


def _precompute_slate_bank(
    model: Any,
    encoded: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    action_seed: int,
    cem_seed: int,
    device: Any,
) -> dict[str, Any]:
    """Create four deterministic query members and detached teacher targets."""

    import torch

    count = int(encoded["count"])
    primary = encoded["actions"].detach().cpu().float()
    gaussian_a: list[Any] = []
    gaussian_b: list[Any] = []
    cem_rows: list[Any] = []
    for index in range(count):
        gen_a = torch.Generator(device="cpu").manual_seed(int(action_seed) + 1009 * index)
        gen_b = torch.Generator(device="cpu").manual_seed(int(action_seed) + 100000 + 1009 * index)
        gaussian_a.append(_sample_actions(gen_a, 1, action_dim, torch.device("cpu"))[0].cpu())
        gaussian_b.append(_sample_actions(gen_b, 1, action_dim, torch.device("cpu"))[0].cpu())
        cem_rows.append(_slate_cem_action(model, encoded, index, objective_fn, action_dim, int(cem_seed) + 1009 * index, device))
    gaussian_a_tensor = torch.stack(gaussian_a)
    gaussian_b_tensor = torch.stack(gaussian_b)
    cem_tensor = torch.stack(cem_rows)
    slate_actions = torch.stack([primary, gaussian_a_tensor, gaussian_b_tensor, cem_tensor], dim=1)
    distinct = []
    for index in range(count):
        rows = slate_actions[index]
        distinct.append(len({tuple(row.reshape(-1).tolist()) for row in rows}) == SLATE_SIZE)
    if not all(distinct):
        raise RuntimeError("query-slate bank contains an exact duplicate; frozen run is invalid")

    # Expand the cached context once so the 1024 teacher targets are produced
    # before step 1 and can be shared byte-for-byte by both arms.
    encoded_flat = {
        "context": {key: value.repeat_interleave(SLATE_SIZE, dim=0) for key, value in encoded["context"].items()},
        "count": count * SLATE_SIZE,
    }
    targets_flat = query_cov._action_bank_targets(
        model, encoded_flat, slate_actions.reshape(count * SLATE_SIZE, *slate_actions.shape[2:]), device, 8
    )
    targets = {key: value.reshape(count, SLATE_SIZE, *value.shape[1:]) for key, value in targets_flat.items()}
    return {
        "actions": slate_actions,
        "targets": targets,
        "query_count": SLATE_SIZE,
        "generation": {
            "primary": "exact logged action prefix from the selected manifest context",
            "additional": ["independent Gaussian planner init A", "independent Gaussian planner init B", "one-step-CEM resample"],
            "candidate_count_M": 64, "elite_count_K": 8, "variance_floor": 0.05,
            "action_slate_seed": int(action_seed), "cem_seed": int(cem_seed),
            "seed_formulas": {"gaussian_a": "action_slate_seed + 1009 * context_ordinal", "gaussian_b": "action_slate_seed + 100000 + 1009 * context_ordinal", "cem": "cem_seed + 1009 * context_ordinal"},
            "distinctness": "exact tensor equality validated before step 1; collision invalidates run",
            "context_count": count, "query_slot_counts": {"primary_logged": count, "gaussian_planner_init_a": count, "gaussian_planner_init_b": count, "one_step_cem_resample": count},
            "teacher_target_cache_rows": count * SLATE_SIZE,
            "pairwise_distinct_passed": True,
        },
    }


def _materialize_slate_batch(
    encoded: Mapping[str, Any], bank: Mapping[str, Any], selected: Sequence[int], device: Any
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    import torch

    indices = torch.as_tensor(list(selected), dtype=torch.long)
    batch = len(selected)
    slate_actions = bank["actions"].index_select(0, indices)
    context = {key: value.index_select(0, indices).repeat_interleave(SLATE_SIZE, dim=0).to(device, non_blocking=True) for key, value in encoded["context"].items()}
    actions = slate_actions.reshape(batch * SLATE_SIZE, *slate_actions.shape[2:]).to(device, non_blocking=True)
    targets: dict[str, Any] = {}
    for key in ("visual", "proprio"):
        targets[key] = bank["targets"][key].index_select(0, indices).reshape(batch * SLATE_SIZE, *bank["targets"][key].shape[2:]).to(device, non_blocking=True)
    goal = {key: value.index_select(0, indices)[:, -1:].repeat_interleave(SLATE_SIZE, dim=0).to(device, non_blocking=True) for key, value in encoded["grounded"].items()}
    return context, actions, targets, goal


def _slate_listwise_kl(objective_fn: Any, target: Mapping[str, Any], prediction: Mapping[str, Any], goal: Mapping[str, Any], temperature: float) -> Any:
    import torch

    teacher = _objective_cost(objective_fn, target, goal).reshape(-1, SLATE_SIZE).detach()
    student = _objective_cost(objective_fn, prediction, goal).reshape(-1, SLATE_SIZE)
    teacher_z = (teacher - teacher.mean(dim=1, keepdim=True)) / teacher.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    student_z = (student - student.mean(dim=1, keepdim=True)) / student.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    teacher_log = torch.log_softmax(-teacher_z / temperature, dim=1).detach()
    student_log = torch.log_softmax(-student_z / temperature, dim=1)
    return (teacher_log.exp() * (teacher_log - student_log)).sum(dim=1).mean()


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _safe_ratio(a: float, b: float) -> float:
    return float(a / max(b, 1e-12))


def _context_schedule(steps: int, slates_per_update: int, seed: int) -> tuple[list[list[int]], dict[str, Any]]:
    """Generate the frozen eight-distinct-context permutation schedule."""

    if slates_per_update != 8:
        raise ValueError("query-slate schedule requires exactly eight context slates per update")
    rng = random.Random(int(seed))
    schedule: list[list[int]] = []
    permutation_blocks = math.ceil(int(steps) / 32)
    for block_index in range(permutation_blocks):
        permutation = list(range(256))
        rng.shuffle(permutation)
        updates = min(32, int(steps) - block_index * 32)
        for update_index in range(updates):
            selected = permutation[update_index * slates_per_update : (update_index + 1) * slates_per_update]
            if len(selected) != slates_per_update or len(set(selected)) != slates_per_update:
                raise RuntimeError("query-slate context schedule contains a duplicate or incomplete group")
            schedule.append(selected)
    if len(schedule) != steps:
        raise RuntimeError(f"query-slate context schedule has {len(schedule)} updates, expected {steps}")
    return schedule, {
        "pool_size": 256,
        "contexts_per_update": 8,
        "schedule_seed": int(seed),
        "generation": "fresh CPU permutation of 0..255 every 32 updates; consume eight consecutive ordinals",
        "permutation_blocks": permutation_blocks,
        "no_within_update_context_duplicates": True,
        "schedule_precomputed_before_training": True,
    }


def _aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rankings = [item["ranking"] for item in cases]
    return {
        "per_case": list(cases),
        "per_horizon": {
            key: [_mean([item[key][h] for item in cases]) for h in range(HORIZON)]
            for key in ("relative_mse", "cosine")
        },
        "terminal_ranking": {
            "spearman_mean": _mean([float(item["spearman"]) for item in rankings]),
            "spearman_median": _median([float(item["spearman"]) for item in rankings]),
            "spearman_minimum": min(float(item["spearman"]) for item in rankings),
            "top30_overlap_mean": _mean([float(item["top30_overlap"]) for item in rankings]),
            "top30_overlap_median": _median([float(item["top30_overlap"]) for item in rankings]),
            "top30_overlap_minimum": min(float(item["top30_overlap"]) for item in rankings),
        },
        "mean_relative_mse": _mean([float(x) for item in cases for x in item["relative_mse"]]),
    }


def _evaluate_snapshots(snapshots: Mapping[str, Mapping[str, Mapping[str, Any]]], model: Any, encoded: Mapping[str, Any], objective_fn: Any, action_dim: int, eval_seeds: Sequence[int], batch_size: int, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    blocks = opt._prepare_eval_blocks(encoded, 8, eval_seeds, batch_size, action_dim, device)
    result: dict[str, Any] = {arm: {} for arm in snapshots}
    paired: dict[str, Any] = {}
    for snap in SNAPSHOT_NAMES:
        students = {}
        for arm in snapshots:
            student = NativeDinoPrefixStudent(action_dim, int(encoded["context"]["visual"].shape[-1]), int(encoded["context"]["proprio"].shape[-1]), WIDTH).to(device)
            student.load_state_dict(snapshots[arm][snap], strict=True)
            student.eval(); students[arm] = student
        cases = {arm: [] for arm in snapshots}
        paired_cases = []
        for block in blocks:
            with torch.no_grad():
                target = _teacher_targets(model, block["context"], block["actions"])
                predictions = {arm: student(block["context"], block["actions"]) for arm, student in students.items()}
            teacher_cost = _objective_cost(objective_fn, target, block["goal"])
            ranked = {}
            for arm, prediction in predictions.items():
                cost = _objective_cost(objective_fn, prediction, block["goal"])
                ranking = {"spearman": _spearman(teacher_cost, cost), "top30_overlap": _topk_overlap(teacher_cost, cost)}
                item = _metrics_by_horizon(prediction, target)
                item.update({"seed": int(block["seed"]), "context_index": int(block["context_index"]), "candidate_count": batch_size, "topk": min(30, batch_size), "ranking": ranking})
                cases[arm].append(item); ranked[arm] = ranking
            paired_cases.append({"seed": int(block["seed"]), "context_index": int(block["context_index"]), "spearman_delta_treatment_minus_control": ranked["treatment"]["spearman"] - ranked["control"]["spearman"], "top30_overlap_delta_treatment_minus_control": ranked["treatment"]["top30_overlap"] - ranked["control"]["top30_overlap"]})
        for arm in snapshots:
            result[arm][snap] = _aggregate_cases(cases[arm])
        paired[snap] = {
            "per_case": paired_cases,
            "spearman_mean": _mean([x["spearman_delta_treatment_minus_control"] for x in paired_cases]),
            "spearman_median": _median([x["spearman_delta_treatment_minus_control"] for x in paired_cases]),
            "top30_overlap_mean": _mean([x["top30_overlap_delta_treatment_minus_control"] for x in paired_cases]),
            "top30_overlap_median": _median([x["top30_overlap_delta_treatment_minus_control"] for x in paired_cases]),
            "positive_spearman_blocks": sum(x["spearman_delta_treatment_minus_control"] > 0 for x in paired_cases),
            "positive_top30_blocks": sum(x["top30_overlap_delta_treatment_minus_control"] > 0 for x in paired_cases),
        }
    return result, paired


def _evaluate_logged(snapshots: Mapping[str, Mapping[str, Mapping[str, Any]]], model: Any, encoded: Mapping[str, Any], action_dim: int, device: Any) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {arm: {} for arm in snapshots}
    for snap in SNAPSHOT_NAMES:
        for arm in snapshots:
            student = NativeDinoPrefixStudent(action_dim, int(encoded["context"]["visual"].shape[-1]), int(encoded["context"]["proprio"].shape[-1]), WIDTH).to(device)
            student.load_state_dict(snapshots[arm][snap], strict=True); student.eval()
            cases = []
            for index in range(8):
                context = {key: value[index:index + 1].to(device) for key, value in encoded["context"].items()}
                actions = encoded["actions"][index:index + 1].to(device)
                with torch.no_grad():
                    target = _teacher_targets(model, context, actions); prediction = student(context, actions)
                cases.append({"context_index": index, **_metrics_by_horizon(prediction, target)})
            result[arm][snap] = {"context_count": len(cases), "per_context": cases, "mean_relative_mse": _mean([x for case in cases for x in case["relative_mse"]])}
    return result


def _diagnostic(summary_path: Path | None, query_eval: Mapping[str, Any]) -> dict[str, Any]:
    if summary_path is None:
        return {"status": "NOT_SUPPLIED", "purpose": "external Dhigh diagnostic only"}
    summary = _load_json(summary_path.resolve())
    reference = summary.get("snapshots", {}).get("step_1500", {}).get("terminal_ranking", {})
    final = query_eval["treatment"]["step_1500"]["terminal_ranking"]
    reference_logged = summary.get("logged_action_teacher_relative", {}).get("step_1500", {})
    return {
        "status": "PASS" if reference else "FAIL",
        "reference_summary": str(summary_path.resolve()),
        "reference_variant": "authoritative context-density Dhigh",
        "treatment_minus_dhigh": {
            "spearman_median": final.get("spearman_median") - reference.get("spearman_median", float("nan")),
            "top30_overlap_median": final.get("top30_overlap_median") - reference.get("top30_overlap_median", float("nan")),
        },
        "reference_absolute_ranking": reference,
        "reference_logged_action_relative_mse": reference_logged,
        "reference_causality": summary.get("causality_final", {}),
        "reference_predictor_latency": summary.get("latency_final", {}),
        "not_used_as_primary_gate": True,
    }


def _gates(settings: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    gates = settings["gates"]
    capacity = gates.get("paired_integrity", gates.get("capacity", {})); fidelity = gates.get("absolute_fidelity", gates.get("fidelity", {})); effect = gates.get("slate_ranking_effect", gates.get("paired_improvement", {})); noninf = gates.get("latent_noninferiority", gates.get("noninferiority", {})); latency = gates.get("latency", {})
    for name, value in (("paired_integrity", capacity), ("absolute_fidelity", fidelity), ("slate_ranking_effect", effect), ("latent_noninferiority", noninf), ("latency", latency)):
        if not isinstance(value, Mapping):
            raise ValueError(f"query-slate gate group {name} must be an object")
    control = result["evaluation"]["control"]["step_1500"]["terminal_ranking"]; treatment = result["evaluation"]["treatment"]["step_1500"]["terminal_ranking"]; delta = result["paired_deltas"]["step_1500"]
    ratio = float(result["latent_noninferiority"]["logged_treatment_to_control_ratio"])
    train = result["train"]
    max_loss = float(capacity.get("both_arms_training_last10_to_first_mse_ratio_max", capacity.get("last10_to_first_training_mse_ratio_max", 0.8)))
    finite = bool(result["finite_outputs_and_training"]); shape = bool(result["shape_dtype_device_match"]); no_fallback = not bool(result["fallback_used"])
    integrity = result["pairing_integrity"]
    integrity_ok = bool(
        finite and shape and no_fallback
        and bool(integrity["initial_state_equal"])
        and bool(integrity["context_schedule_equal"])
        and bool(integrity["slate_bank_equal"])
        and bool(integrity["heldout_candidate_bank_equal"])
        and bool(integrity["all_slate_groups_complete"])
        and bool(integrity["all_slate_queries_pairwise_distinct"])
        and bool(integrity["heldout_split_unchanged"])
        and bool(integrity["student_goal_input_hidden"])
        and bool(integrity["no_oom_nan_or_silent_fallback"])
        and str(integrity["manifest_integrity"].get("status", "")) == "PASS"
        and all(bool(item["passed"]) for item in result["student_integrity"].values())
        and all(float(train[arm]["latent_last10_to_first_ratio"]) <= max_loss for arm in ("control", "treatment"))
        and all(bool(x["passed"]) for x in result["causality"].values())
    )
    absolute_ok = (treatment["spearman_median"] >= float(fidelity.get("median_spearman_min", 0.99)) and treatment["spearman_minimum"] >= float(fidelity.get("minimum_spearman_min", 0.95)) and treatment["top30_overlap_median"] >= float(fidelity.get("median_top30_min", 0.95)) and treatment["top30_overlap_minimum"] >= float(fidelity.get("minimum_top30_min", 0.8)))
    effect_ok = (delta["spearman_median"] >= float(effect.get("median_spearman_delta_min", 0.05)) and delta["top30_overlap_median"] >= float(effect.get("median_top30_overlap_delta_min", effect.get("median_top30_delta_min", 0.1))) and delta["positive_spearman_blocks"] >= int(effect.get("positive_spearman_blocks_min", 12)) and delta["positive_top30_blocks"] >= int(effect.get("positive_top30_blocks_min", 12)))
    ratio_max = float(noninf.get("ratio_max", noninf.get("treatment_to_control_mean_relative_mse_ratio_max", 1.25)))
    ratio_ok = ratio <= ratio_max
    latency_ok = float(result["latency"]["treatment"]["median_reduction"]) >= float(latency.get("predictor_only_reduction_min", latency.get("rankdistill_predictor_reduction_vs_teacher_min", 0.2)))
    full = all((integrity_ok, absolute_ok, effect_ok, ratio_ok, latency_ok))
    return {
        "paired_integrity": {"status": "PASS" if integrity_ok else "FAIL", "finite": finite, "shape_dtype_device_match": shape, "no_fallback": no_fallback, "last10_to_first_ratio_max": max_loss, "pairing": integrity},
        "absolute_fidelity": {"status": "PASS" if absolute_ok else "FAIL", "metrics": treatment, "thresholds": dict(fidelity)},
        "slate_ranking_effect": {"status": "PASS" if effect_ok else "FAIL", "metrics": delta, "thresholds": dict(effect)},
        "latent_noninferiority": {"status": "PASS" if ratio_ok else "FAIL", "treatment_to_control_logged_relative_mse_ratio": ratio, "thresholds": dict(noninf)},
        "predictor_latency": {"status": "PASS" if latency_ok else "FAIL", "treatment_reduction": result["latency"]["treatment"]["median_reduction"], "thresholds": dict(latency)},
        "decision_levels": {"paired_effect": "PASS" if integrity_ok and effect_ok else "FAIL", "full_replacement": "GO" if full else "NO-GO", "slate_supported_replacement": "GO" if full and integrity_ok and effect_ok else "NO-GO"},
        "overall": "GO" if full else "NO-GO",
    }


def main() -> int:
    args = _args()
    _require_compute_node()
    root = args.root.resolve(); asset_root = (args.asset_root or root).resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve()); protocol = _load_json(args.protocol.resolve()) if args.protocol else {}; settings = _settings(freeze, protocol)
    context_freeze = _load_json(args.context_density_freeze.resolve()) if args.context_density_freeze else None
    density_settings = _density_schedule_settings(freeze, context_freeze)
    legacy = args.legacy_manifest.resolve() if args.legacy_manifest else None
    manifest = density._load_density_manifest(args.manifest.resolve(), density_settings, legacy)
    _require_assets(asset_root, freeze); _gpu_snapshot(output, "start")
    import torch

    _set_seed(settings["training_seed"])
    model, workspace, _anchors, _goals, objective_fn, action_dim, device = _load_official(root, asset_root, freeze, output, settings["training_seed"], config_path=args.config.resolve() if args.config else None, checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None, checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None, data_root=args.data_root.resolve() if args.data_root else None)
    del workspace, _anchors, _goals
    if int(action_dim) != 10:
        raise ValueError("official packed action token dim must remain 10")
    train_dset, heldout_dset = _load_trajectory_datasets(root, asset_root, freeze, args.checkpoint, args.checkpoint_config)
    train_examples = manifest["splits"]["train"]["examples"]; heldout_examples = manifest["splits"]["heldout"]["examples"]
    cache = _RawEpisodeCache(8)
    train_encoded = _preencode_manifest(train_dset, "train", train_examples, 5, 8, cache, model, device)
    heldout_encoded = _preencode_manifest(heldout_dset, "heldout", heldout_examples, 5, 8, cache, model, device)
    if tuple(int(x) for x in train_encoded["context"]["visual"].shape[-1:]) != (384,):
        raise ValueError("visual native latent dimension must remain 384")
    visual_dim = int(train_encoded["context"]["visual"].shape[-1]); proprio_dim = int(train_encoded["context"]["proprio"].shape[-1])
    _set_seed(settings["initialization_seed"])
    template = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    students = {name: NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, WIDTH).to(device) for name in ("control", "treatment")}
    for student in students.values(): student.load_state_dict(initial_state)
    initial_state_equal = all(
        torch.equal(students["control"].state_dict()[key], students["treatment"].state_dict()[key])
        for key in students["control"].state_dict()
    )
    optimizers = {
        name: torch.optim.AdamW(
            student.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"],
            betas=settings["betas"], eps=settings["eps"],
        )
        for name, student in students.items()
    }
    high_schedule, schedule_meta = _context_schedule(settings["steps"], settings["batch"], settings["context_schedule_seed"])
    bank = _precompute_slate_bank(model, train_encoded, objective_fn, action_dim, settings["action_slate_seed"], settings["cem_seed"], device)
    _gpu_snapshot(output, "precompute_complete")
    losses = {name: [] for name in students}; snapshots = {name: {} for name in students}; checkpoint_paths = {name: {} for name in students}; telemetry_period = max(1, settings["steps"] // 10); shape_dtype_device_match = True
    for step_index in range(settings["steps"]):
        context, actions, target, goal = _materialize_slate_batch(train_encoded, bank, high_schedule[step_index], device)
        step = step_index + 1
        order = ("control", "treatment") if step % 2 == 0 else ("treatment", "control")
        for name in order:
            prediction = students[name](context, actions)
            shape_dtype_device_match = shape_dtype_device_match and all(
                prediction[key].shape == target[key].shape
                and prediction[key].dtype == target[key].dtype
                and prediction[key].device == target[key].device
                for key in ("visual", "proprio")
            )
            latent = _latent_mse(prediction, target)
            rank_loss = _slate_listwise_kl(objective_fn, target, prediction, goal, settings["rank_temperature"]) if name == "treatment" else latent.detach().new_zeros(())
            total = latent + settings["rank_weight"] * rank_loss if name == "treatment" else latent
            optimizers[name].zero_grad(set_to_none=True); total.backward(); optimizers[name].step()
            losses[name].append({"step": step_index + 1, "total": float(total.detach().cpu()), "latent_mse": float(latent.detach().cpu()), "planner_listwise_kl": float(rank_loss.detach().cpu()), "query_groups": settings["batch"], "queries_per_group": SLATE_SIZE})
        if step in SNAPSHOT_STEPS:
            for name, student in students.items():
                state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}; snapshots[name][f"step_{step}"] = state
                filename = f"{name}_query_slate_step{step:04d}.pt"; torch.save({"schema": "jepa-action-prefix-compiler.dino-pusht-query-slate-rank-checkpoint", "step": step, "arm": name, "hidden_dim": WIDTH, "state_dict": state, "parent_freeze": str(args.freeze.resolve())}, output / filename); checkpoint_paths[name][f"step_{step}"] = filename
        if step == 1 or step % telemetry_period == 0: _gpu_snapshot(output, f"train_step_{step}")
    evaluation, paired = _evaluate_snapshots(snapshots, model, heldout_encoded, objective_fn, action_dim, settings["heldout_seeds"], settings["eval_batch"], device)
    logged = _evaluate_logged(snapshots, model, heldout_encoded, action_dim, device)
    causality = {name: _leakage_test(student, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["heldout_seeds"][0] + 9000, device, 1e-6) for name, student in students.items()}
    latency = {name: _latency(student, model, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["timing_seed"], settings["timing_batch"], settings["warmup"], settings["repeats"], device) for name, student in students.items()}
    _gpu_snapshot(output, "complete")
    train_result = {}
    for name, values in losses.items():
        totals = [x["total"] for x in values]; latent_values = [x["latent_mse"] for x in values]
        train_result[name] = {"steps": settings["steps"], "batch_contexts": settings["batch"], "effective_query_samples": settings["batch"] * SLATE_SIZE, "loss_first": totals[0], "loss_last": totals[-1], "loss_median_last_10": _median(totals[-10:]), "last10_to_first_loss_ratio": _safe_ratio(_median(totals[-10:]), totals[0]), "latent_mse_first": latent_values[0], "latent_mse_median_last_10": _median(latent_values[-10:]), "latent_last10_to_first_ratio": _safe_ratio(_median(latent_values[-10:]), latent_values[0]), "per_step": values}
    control_eval_mse = evaluation["control"]["step_1500"]["mean_relative_mse"]; treatment_eval_mse = evaluation["treatment"]["step_1500"]["mean_relative_mse"]
    control_logged_mse = logged["control"]["step_1500"]["mean_relative_mse"]; treatment_logged_mse = logged["treatment"]["step_1500"]["mean_relative_mse"]
    density_validation = manifest.get("density_validation", {})
    pairing_integrity = {
        "initial_state_equal": initial_state_equal,
        "context_schedule_equal": True,
        "slate_bank_equal": True,
        "heldout_candidate_bank_equal": True,
        "paired_block_ids": [f"seed={item['seed']}:context={item['context_index']}" for item in paired["step_1500"]["per_case"]],
        "all_slate_groups_complete": len(high_schedule) == settings["steps"] and all(len(x) == settings["batch"] for x in high_schedule),
        "all_slate_queries_pairwise_distinct": bool(bank["generation"]["pairwise_distinct_passed"]),
        "heldout_split_unchanged": bool(density_validation.get("heldout_split_unchanged", True)),
        "student_goal_input_hidden": True,
        "no_oom_nan_or_silent_fallback": True,
        "manifest_integrity": density_validation,
    }
    result = {"train": train_result, "evaluation": evaluation, "paired_deltas": paired, "logged_teacher_relative": logged, "causality": causality, "latency": latency, "pairing_integrity": pairing_integrity, "latent_noninferiority": {"logged_control_mean_relative_mse": control_logged_mse, "logged_treatment_mean_relative_mse": treatment_logged_mse, "logged_treatment_to_control_ratio": _safe_ratio(treatment_logged_mse, control_logged_mse), "evaluation_control_mean_relative_mse": control_eval_mse, "evaluation_treatment_mean_relative_mse": treatment_eval_mse, "evaluation_treatment_to_control_ratio": _safe_ratio(treatment_eval_mse, control_eval_mse)}, "finite_outputs_and_training": _all_finite({"train": train_result, "evaluation": evaluation, "paired": paired, "logged": logged, "latency": latency}), "shape_dtype_device_match": shape_dtype_device_match, "fallback_used": False}
    result["student_integrity"] = {name: opt._student_integrity(student, model) for name, student in students.items()}
    result["shape_dtype_device_match"] = bool(
        result["shape_dtype_device_match"]
        and all(bool(x["passed"]) for x in result["student_integrity"].values())
    )
    gates = _gates(settings, result)
    summary = {"schema": "jepa-action-prefix-compiler.dino-pusht-query-slate-rank-summary", "schema_version": 1, "freeze": str(args.freeze.resolve()), "protocol": str(args.protocol.resolve()) if args.protocol else None, "manifest": str(args.manifest.resolve()), "density_summary": str(args.density_summary.resolve()) if args.density_summary else None, "source": {"student": "NativeDinoPrefixStudent hidden_dim=256", "teacher_target": "shared detached official DINO-WM rollout target", "context_schedule": "frozen query-slate permutation schedule", "external_dhigh": "read-only descriptive diagnostic; not the paired control", "observation_encoder": "outside scope; cached native context"}, "contracts": {"train_contexts": 256, "contexts_per_update": settings["batch"], "slates_per_update": settings["batch"], "queries_per_context": SLATE_SIZE, "effective_query_samples_per_update": settings["effective_batch"], "updates": settings["steps"], "snapshot_steps": list(SNAPSHOT_STEPS), "heldout_blocks": 16, "heldout_contexts": 8, "heldout_seeds": settings["heldout_seeds"], "candidates_per_block": settings["eval_batch"], "same_initialization": True, "paired_slate_and_teacher_targets": True, "rank_weight": settings["rank_weight"], "rank_temperature": settings["rank_temperature"], "training_seed": settings["training_seed"], "initialization_seed": settings["initialization_seed"], "context_schedule_seed": settings["context_schedule_seed"], "action_slate_seed": settings["action_slate_seed"], "cem_seed": settings["cem_seed"], "gpu_telemetry_seconds": settings["telemetry_seconds"]}, "slate": bank["generation"], "schedule": schedule_meta, "pairing_integrity": pairing_integrity, "train": train_result, "training": {"per_step_latent_mse_control": [x["latent_mse"] for x in losses["control"]], "per_step_latent_mse_treatment": [x["latent_mse"] for x in losses["treatment"]], "per_step_rank_kl_treatment": [x["planner_listwise_kl"] for x in losses["treatment"]], "last10_to_first_ratio_control": train_result["control"]["latent_last10_to_first_ratio"], "last10_to_first_ratio_treatment": train_result["treatment"]["latent_last10_to_first_ratio"], "snapshot_steps": list(SNAPSHOT_STEPS)}, "evaluation": evaluation, "paired_deltas": paired, "absolute_treatment": evaluation["treatment"]["step_1500"]["terminal_ranking"], "logged_teacher_relative": logged, "logged_mse_ratio": _safe_ratio(treatment_logged_mse, control_logged_mse), "authoritative_dhigh_diagnostic": _diagnostic(args.density_summary, evaluation), "latent_noninferiority": result["latent_noninferiority"], "causality": causality, "latency": latency, "student_integrity": result["student_integrity"], "finite_outputs_and_training": result["finite_outputs_and_training"], "shape_dtype_device_match": result["shape_dtype_device_match"], "fallback_used": False, "checkpoint_filenames": checkpoint_paths, "gates": gates, "decisions": gates["decision_levels"], "claim_boundary": {"paired_effect": "Only if paired_integrity and slate_ranking_effect pass: within this frozen DINO-WM PushT predictor-level cell, the treatment improves held-out ranking over the identical slate-trained latent-MSE control.", "external_dhigh": "Descriptive reference only; not a causal or paired comparison.", "forbidden": ["encode_obs effect", "closed-loop CEM or environment success", "LeWM/Fast-LeWM transfer", "all-JEPA universality"]}, "timing_boundary": "predictor-level: cached native observation latent plus normalized action prefix; excludes encode_obs, CEM execution, environment interaction and closed-loop control", "unverified": ["No closed-loop CEM or environment execution is included", "This DINO-WM runner does not establish LeWM transfer"]}
    summary_path = (args.summary or output / "query_slate_rank_summary.json").resolve(); summary_path.parent.mkdir(parents=True, exist_ok=True); text = json.dumps(summary, indent=2, default=_json_default); summary_path.write_text(text, encoding="utf-8"); default_summary = output / "query_slate_rank_summary.json"; default_summary.write_text(text, encoding="utf-8") if summary_path != default_summary else None
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
