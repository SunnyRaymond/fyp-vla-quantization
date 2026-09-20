#!/usr/bin/env python3
"""Run the frozen 2x2 context/action-query coverage experiment.

The four arms use the same frozen DINO-WM teacher, NativeDinoPrefixStudent,
initial state, optimizer, update count, batch size and update order:

``NARROW-LOGGED``
    The first two manifest train episodes and their logged action prefixes.
``WIDE-LOGGED``
    All manifest train contexts and their logged action prefixes.
``NARROW-QUERY`` / ``WIDE-QUERY``
    The same narrow/wide context pools, with a frozen 50/25/25 action-query
    mixture: logged actions, Gaussian CEM-initialisation actions, and
    one-step teacher-scored elite-resampled prefixes.

All training targets are dense frozen-teacher rollouts.  Goals are used only
to construct the training query distribution and to score held-out candidates;
they are never passed to the student.  This runner refuses to load models or
decode data outside a PBS compute-node allocation and performs no downloads,
installation, or environment setup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    _batch_from_cache,
    _latent_mse,
    _load_trajectory_datasets,
    _preencode_manifest,
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


ARM_NAMES = (
    "narrow_logged",
    "wide_logged",
    "narrow_query",
    "wide_query",
)


def _all_finite(value: Any) -> bool:
    """Recursively check the scalar/nested result tree used by this runner."""

    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


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


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def _query_settings(freeze: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the dedicated query freeze without relying on grounded layout."""

    training = freeze.get("training", {})
    model_contract = freeze.get("model_contract", {})
    randomness = freeze.get("randomness_and_schedule", {})
    heldout = freeze.get("heldout_evaluation", {})
    ranking = heldout.get("ranking", {}) if isinstance(heldout, Mapping) else {}
    latency = heldout.get("latency", {}) if isinstance(heldout, Mapping) else {}
    gates = freeze.get("gates", {})
    capacity = gates.get("capacity", {}) if isinstance(gates, Mapping) else {}
    if not all(isinstance(item, Mapping) for item in (training, model_contract, randomness, ranking, latency, capacity)):
        raise ValueError("query freeze has malformed training/model/randomness/evaluation objects")
    # A protocol JSON can provide local aliases, but the dedicated freeze is
    # authoritative for this runner's immutable numerical contract.
    return {
        "steps": int(training.get("steps", 500)),
        "batch": int(training.get("batch_size", 32)),
        "lr": float(training.get("learning_rate", 3e-4)),
        "hidden_dim": int(model_contract.get("hidden_dim", 128)),
        "frame_skip": int(model_contract.get("frameskip", 5)),
        "cache_episodes": 8,
        "preencode_chunk": 8,
        "train_seed": int(randomness.get("training_seed", 20260925)),
        "heldout_seeds": [int(value) for value in randomness.get("heldout_action_prefix_seeds", [20264925, 20264926])],
        "timing_seed": int(randomness.get("timing_action_prefix_seed", 20270925)),
        "eval_batch": int(ranking.get("candidates_per_block", 300)),
        "timing_batch": int(latency.get("batch_size", 300)),
        "warmup": int(latency.get("warmup_repeats", 3)),
        "repeats": int(latency.get("technical_repeats", 10)),
        "val_batches": 1,
        "train_segments": 128,
        "heldout_segments": 8,
        "future_action_tolerance": float(capacity.get("future_action_leakage_max_abs", 1e-6)),
        "capacity_gate": dict(capacity),
    }


def _query_spec(protocol: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Read the machine-readable action-mixture contract.

    New query-coverage freezes place the object under ``query_coverage``.
    ``query`` is accepted as a short spelling so a staged protocol can use a
    compact name, while the defaults remain the frozen 50/25/25 design.
    """

    query: dict[str, Any] = {}
    for source in (freeze, protocol):
        candidate = source.get("query_coverage", source.get("query", {}))
        if isinstance(candidate, Mapping):
            query.update(candidate)
    frozen_generation = freeze.get("action_query_generation", {})
    if isinstance(frozen_generation, Mapping):
        query.update(frozen_generation)
    protocol_generation = protocol.get("action_query_generation", {})
    if isinstance(protocol_generation, Mapping):
        query.update(protocol_generation)
    cem_generation = query.get(
        "one_step_cem",
        query.get("one_step_cem_resample", query.get("cem", query.get("one_step_cem_resample", {}))),
    )
    if not isinstance(cem_generation, Mapping):
        cem_generation = {}
    cem_precompute = cem_generation.get("per_training_context_precompute", {})
    if not isinstance(cem_precompute, Mapping):
        cem_precompute = {}
    role_schedule = query.get("role_schedule", {})
    if not isinstance(role_schedule, Mapping):
        role_schedule = {}
    mixture = query.get("mixture", role_schedule.get("rows_per_batch", {}))
    if not isinstance(mixture, Mapping):
        mixture = {}
    raw_logged = float(mixture.get("logged", mixture.get("logged_fraction", 0.50)))
    raw_gaussian = float(
        mixture.get("gaussian_planner_init", mixture.get("gaussian", mixture.get("gaussian_fraction", 0.25)))
    )
    raw_elite = float(
        mixture.get("elite_perturbation", mixture.get("one_step_cem_resample", mixture.get("bounded_perturbation", mixture.get("elite", mixture.get("elite_fraction", 0.25)))))
    )
    raw_total = raw_logged + raw_gaussian + raw_elite
    if raw_total > 1.000001:
        raw_logged, raw_gaussian, raw_elite = (
            raw_logged / raw_total,
            raw_gaussian / raw_total,
            raw_elite / raw_total,
        )
    logged_fraction = float(query.get("logged_fraction", raw_logged))
    gaussian_fraction = float(
        query.get("gaussian_fraction", raw_gaussian)
    )
    elite_fraction = float(
        query.get("elite_fraction", raw_elite)
    )
    if any(value < 0.0 for value in (logged_fraction, gaussian_fraction, elite_fraction)):
        raise ValueError("query action-mixture fractions must be non-negative")
    if abs(logged_fraction + gaussian_fraction + elite_fraction - 1.0) > 1e-6:
        raise ValueError("query action-mixture fractions must sum to one")
    candidate_default = cem_precompute.get(
        "candidate_count_M", cem_generation.get("candidates", cem_generation.get("M", 64))
    )
    topk_default = cem_precompute.get(
        "elite_count_K", cem_generation.get("topk", cem_generation.get("K", 8))
    )
    variance_default = cem_precompute.get(
        "variance_floor", cem_generation.get("variance_floor", 0.05)
    )
    cem_candidates = int(query.get("cem_candidates", query.get("candidate_count", query.get("M", candidate_default))))
    cem_topk = int(query.get("cem_topk", query.get("topk", query.get("K", topk_default))))
    cem_variance_floor = float(query.get("cem_variance_floor", query.get("variance_floor", variance_default)))
    if cem_candidates < 2 or cem_topk < 1 or cem_topk > cem_candidates:
        raise ValueError("one-step CEM requires 1 <= cem_topk <= cem_candidates")
    if cem_variance_floor <= 0.0:
        raise ValueError("cem_variance_floor must be positive")
    evaluation = protocol.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    heldout = protocol.get("heldout", {})
    if not isinstance(heldout, Mapping):
        heldout = {}
    eval_contexts = int(
        query.get("evaluation_contexts", evaluation.get("contexts", heldout.get("contexts", 8)))
    )
    eval_seeds_value = query.get(
        "evaluation_action_seeds",
        evaluation.get("action_seeds", heldout.get("action_prefix_seeds", [20263920, 20263921])),
    )
    randomness = freeze.get("randomness_and_schedule", {})
    if isinstance(randomness, Mapping) and "heldout_action_prefix_seeds" in randomness:
        eval_seeds_value = randomness["heldout_action_prefix_seeds"]
    eval_seeds = [int(value) for value in eval_seeds_value]
    if eval_contexts < 8:
        raise ValueError("query-coverage evaluation requires at least eight held-out contexts")
    if len(eval_seeds) < 2:
        raise ValueError("query-coverage evaluation requires at least two fresh action seeds")
    return {
        "fractions": {
            "logged": logged_fraction,
            "gaussian_planner_init": gaussian_fraction,
            "one_step_cem_resample": elite_fraction,
        },
        "cem_candidates": cem_candidates,
        "cem_topk": cem_topk,
        "cem_variance_floor": cem_variance_floor,
        "evaluation_contexts": eval_contexts,
        "evaluation_action_seeds": eval_seeds,
        "training_query_seed_offset": int(query.get("training_query_seed_offset", 1000003)),
        "elite_action_distribution": "one-step CEM: M Normal(0, I) candidates, teacher objective top-K, per-coordinate mean/variance, variance floor, one resample",
        "student_goal_input": False,
    }


def _ensure_query_freeze(freeze: Mapping[str, Any], query: Mapping[str, Any]) -> None:
    schema = str(freeze.get("schema", ""))
    allowed = {
        "jepa-action-prefix-compiler.query-coverage-freeze",
        "jepa-action-prefix-compiler.grounded-prefix-freeze",
    }
    if schema not in allowed:
        raise ValueError(f"unexpected freeze schema: {schema!r}")
    frozen_query = freeze.get("query_coverage", freeze.get("query", {}))
    generation = freeze.get("action_query_generation", {})
    if isinstance(generation, Mapping):
        precompute = generation.get("one_step_cem_resample", generation.get("one_step_cem", {}))
        if not isinstance(precompute, Mapping):
            precompute = {}
        precompute = precompute.get("per_training_context_precompute", precompute)
        if not isinstance(precompute, Mapping):
            precompute = {}
        for key, expected in (
            ("candidate_count_M", query["cem_candidates"]),
            ("elite_count_K", query["cem_topk"]),
            ("variance_floor", query["cem_variance_floor"]),
        ):
            if key in precompute and float(precompute[key]) != float(expected):
                raise ValueError(f"freeze/protocol CEM generation mismatch for {key}")
    if isinstance(frozen_query, Mapping):
        fractions = frozen_query.get("fractions", frozen_query.get("mixture", {}))
        if isinstance(fractions, Mapping):
            expected = query["fractions"]
            aliases = {
                "logged": ("logged", "logged_fraction"),
                "gaussian_planner_init": ("gaussian_planner_init", "gaussian", "gaussian_fraction"),
                "one_step_cem_resample": (
                    "one_step_cem_resample",
                    "elite_perturbation",
                    "elite",
                    "elite_fraction",
                ),
            }
            for name, keys in aliases.items():
                for key in keys:
                    if key in fractions and abs(float(fractions[key]) - float(expected[name])) > 1e-6:
                        raise ValueError(f"freeze/protocol query fraction mismatch for {name}")


def _load_query_manifest(path: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the dedicated query-coverage manifest.

    The preflight manifest intentionally calls entries ``contexts`` rather
    than ``examples``.  The existing grounded pre-encoder is reusable once
    those immutable references are normalized in memory; the source JSON is
    never rewritten.
    """

    manifest = _load_json(path.resolve())
    if manifest.get("schema") != "jepa-action-prefix-compiler.query-coverage-manifest":
        raise ValueError("query runner requires the dedicated query-coverage manifest schema")
    protocol = manifest.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("query manifest has no protocol object")
    if int(protocol.get("horizon", -1)) != HORIZON:
        raise ValueError("query manifest horizon does not match H=5")
    if int(protocol.get("frame_skip", -1)) != int(settings["frame_skip"]):
        raise ValueError("query manifest frame_skip does not match frozen protocol")
    splits = manifest.get("splits", {})
    if not isinstance(splits, Mapping):
        raise ValueError("query manifest has no splits object")
    normalized = dict(manifest)
    normalized_splits: dict[str, Any] = {}
    expected_counts = {"train": 128, "heldout": 8}
    for split_name, expected_count in expected_counts.items():
        split = splits.get(split_name)
        if not isinstance(split, Mapping):
            raise ValueError(f"query manifest has no {split_name} split")
        contexts = split.get("contexts")
        if not isinstance(contexts, list) or len(contexts) != expected_count:
            raise ValueError(
                f"query manifest {split_name} must contain exactly {expected_count} contexts"
            )
        examples: list[dict[str, Any]] = []
        for context in contexts:
            if not isinstance(context, Mapping):
                raise ValueError(f"query manifest {split_name} contains a non-object context")
            item = dict(context)
            if "start_step" not in item and "start" in item:
                item["start_step"] = item["start"]
            for key in ("episode_id", "start_step", "target_steps"):
                if key not in item:
                    raise ValueError(f"query manifest context missing {key}")
            expected_steps = [
                int(item["start_step"]) + (index + 1) * int(settings["frame_skip"])
                for index in range(HORIZON)
            ]
            if [int(value) for value in item["target_steps"]] != expected_steps:
                raise ValueError("query manifest target_steps are not contiguous frozen frame-skip targets")
            examples.append(item)
        normalized_splits[split_name] = {**split, "examples": examples}
    train_episode_ids = {int(item["episode_id"]) for item in normalized_splits["train"]["examples"]}
    heldout_episode_ids = {int(item["episode_id"]) for item in normalized_splits["heldout"]["examples"]}
    train_episode_keys = {str(item.get("episode_key", f"train:{item['episode_id']}")) for item in normalized_splits["train"]["examples"]}
    heldout_episode_keys = {str(item.get("episode_key", f"heldout:{item['episode_id']}")) for item in normalized_splits["heldout"]["examples"]}
    if train_episode_keys & heldout_episode_keys:
        raise ValueError("query manifest train and held-out episode IDs overlap")
    if len(train_episode_ids) != 32 or len(heldout_episode_ids) != 8:
        raise ValueError("query manifest must contain 32 train and 8 held-out episodes")
    if any(
        sum(int(item["episode_id"]) == episode_id for item in normalized_splits["train"]["examples"]) != 4
        for episode_id in train_episode_ids
    ):
        raise ValueError("query manifest must contain four contexts per train episode")
    if any(
        sum(int(item["episode_id"]) == episode_id for item in normalized_splits["heldout"]["examples"]) != 1
        for episode_id in heldout_episode_ids
    ):
        raise ValueError("query manifest must contain one context per held-out episode")
    normalized["splits"] = normalized_splits
    return normalized


def _frozen_randomness(protocol: Mapping[str, Any], freeze: Mapping[str, Any], settings: dict[str, Any]) -> dict[str, int]:
    randomness = freeze.get("randomness_and_schedule", {})
    if not isinstance(randomness, Mapping):
        randomness = {}
    protocol_randomness = protocol.get("randomness_and_schedule", {})
    if not isinstance(protocol_randomness, Mapping):
        protocol_randomness = {}
    merged = {**randomness, **protocol_randomness}
    settings["train_seed"] = int(merged.get("training_seed", settings["train_seed"]))
    settings["timing_seed"] = int(merged.get("timing_action_prefix_seed", settings["timing_seed"]))
    heldout = merged.get("heldout_action_prefix_seeds", settings["heldout_seeds"])
    settings["heldout_seeds"] = [int(value) for value in heldout]
    return {
        "context_schedule_seed": int(merged.get("context_schedule_seed", settings["train_seed"])),
        "action_query_seed": int(
            merged.get("action_query_seed", merged.get("action_query_generation_seed", settings["train_seed"] + 1))
        ),
        "role_schedule_seed": int(merged.get("role_schedule_seed", settings["train_seed"] + 2)),
    }


def _make_role_schedule(steps: int, batch: int, seed: int, query: Mapping[str, Any]) -> list[Any]:
    import torch

    fractions = query["fractions"]
    counts = [
        int(round(batch * float(fractions["logged"]))),
        int(round(batch * float(fractions["gaussian_planner_init"]))),
    ]
    counts.append(batch - sum(counts))
    if counts != [16, 8, 8] or batch != 32:
        raise ValueError(f"frozen query role schedule requires batch 32 with 16/8/8 rows, got {counts}")
    base = torch.tensor([0] * counts[0] + [1] * counts[1] + [2] * counts[2], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return [base.index_select(0, torch.randperm(batch, generator=generator)) for _ in range(steps)]


def _action_bank_targets(
    model: Any,
    encoded_cache: Mapping[str, Any],
    actions_cpu: Any,
    device: Any,
    chunk_size: int,
) -> dict[str, Any]:
    """Teacher-rollout targets for one frozen per-context action bank."""

    import torch

    pieces: dict[str, list[Any]] = {"visual": [], "proprio": []}
    count = int(actions_cpu.shape[0])
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        indices = torch.arange(start, stop, dtype=torch.long)
        context = {
            key: value.index_select(0, indices).to(device, non_blocking=True)
            for key, value in encoded_cache["context"].items()
        }
        actions = actions_cpu.index_select(0, indices).to(device, non_blocking=True)
        with torch.no_grad():
            target = _teacher_targets(model, context, actions)
        for key in pieces:
            pieces[key].append(target[key].detach().cpu())
    return {key: torch.cat(values, dim=0) for key, values in pieces.items()}


def _precompute_action_banks(
    model: Any,
    encoded_cache: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    query: Mapping[str, Any],
    action_seed: int,
    device: Any,
    target_chunk: int,
) -> dict[str, Any]:
    """Precompute logged/Gaussian/one-step-CEM actions and dense targets.

    The only adaptive operation is the frozen one-step CEM score used while
    constructing the training bank.  Once this function returns, the 500-step
    loop performs no teacher rollout and no action sampling.
    """

    import torch

    count = int(encoded_cache["count"])
    logged = encoded_cache["actions"].detach().cpu().float()
    gaussian_rows: list[Any] = []
    cem_rows: list[Any] = []
    candidate_count = int(query["cem_candidates"])
    topk = int(query["cem_topk"])
    variance_floor = float(query["cem_variance_floor"])
    for index in range(count):
        # The per-context seed is frozen by the query protocol.  Keeping a
        # separate CPU generator also makes the bank independent of context
        # chunking and of CUDA RNG state.
        action_generator = torch.Generator(device="cpu").manual_seed(
            int(action_seed) + 1009 * index
        )
        gaussian = _sample_actions(action_generator, 1, action_dim, torch.device("cpu"))[0].cpu()
        gaussian_rows.append(gaussian)
        candidate = _sample_actions(action_generator, candidate_count, action_dim, torch.device("cpu"))
        context_one = {
            key: value[index : index + 1].to(device)
            for key, value in encoded_cache["context"].items()
        }
        context_candidates = _repeat_anchor(context_one, candidate_count)
        goal_one = {
            key: value[index : index + 1, -1:].to(device)
            for key, value in encoded_cache["grounded"].items()
        }
        goal_candidates = _repeat_anchor(goal_one, candidate_count)
        with torch.no_grad():
            candidate_target = _teacher_targets(model, context_candidates, candidate.to(device))
            candidate_cost = _objective_cost(objective_fn, candidate_target, goal_candidates).reshape(-1)
        elite_index = torch.argsort(candidate_cost)[:topk].detach().cpu()
        elite = candidate.index_select(0, elite_index)
        elite_mean = elite.mean(dim=0)
        elite_variance = elite.var(dim=0, unbiased=False).clamp_min(variance_floor)
        cem = elite_mean + elite_variance.sqrt() * _sample_actions(
            action_generator, 1, action_dim, torch.device("cpu")
        )[0].cpu()
        cem_rows.append(cem)
    actions = {
        "logged": logged,
        "gaussian_planner_init": torch.stack(gaussian_rows),
        "cem_elite_resample": torch.stack(cem_rows),
    }
    targets = {
        role: _action_bank_targets(model, encoded_cache, values, device, target_chunk)
        for role, values in actions.items()
    }
    return {
        "actions": actions,
        "targets": targets,
        "cem": {
            "candidates": candidate_count,
            "topk": topk,
            "variance_floor": variance_floor,
            "action_seed": int(action_seed),
        },
    }


def _materialize_bank_batch(
    encoded_cache: Mapping[str, Any],
    bank: Mapping[str, Any],
    selected: Sequence[int],
    roles: Any,
    device: Any,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import torch

    indices = torch.as_tensor(list(selected), dtype=torch.long)
    context = {
        key: value.index_select(0, indices).to(device, non_blocking=True)
        for key, value in encoded_cache["context"].items()
    }
    role_names = ("logged", "gaussian_planner_init", "cem_elite_resample")
    actions_stack = torch.stack([bank["actions"][name] for name in role_names], dim=1)
    action_rows = actions_stack.index_select(0, indices)
    role_index = roles.to(dtype=torch.long).view(-1, 1, 1, 1).expand(-1, 1, action_rows.shape[2], action_rows.shape[3])
    actions = action_rows.gather(1, role_index).squeeze(1).to(device, non_blocking=True)
    targets: dict[str, Any] = {}
    for key in ("visual", "proprio"):
        target_stack = torch.stack([bank["targets"][name][key] for name in role_names], dim=1)
        target_rows = target_stack.index_select(0, indices)
        shape = (target_rows.shape[0], 1) + tuple(target_rows.shape[2:])
        target_index = roles.to(dtype=torch.long).view(-1, 1, *([1] * (target_rows.ndim - 2))).expand(*shape)
        targets[key] = target_rows.gather(1, target_index).squeeze(1).to(device, non_blocking=True)
    return context, actions, targets


def _train_arm(
    student: Any,
    optimizer: Any,
    context: Mapping[str, Any],
    actions: Any,
    target: Mapping[str, Any],
    step: int,
) -> dict[str, float]:
    prediction = student(context, actions)
    latent_loss = _latent_mse(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    latent_loss.backward()
    optimizer.step()
    return {"step": step + 1, "latent_mse": float(latent_loss.detach().cpu())}


def _evaluate_real_and_teacher_relative(
    students: Mapping[str, Any],
    model: Any,
    encoded_cache: Mapping[str, Any],
    batch_size: int,
    batches: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    generator = random.Random(seed)
    names = ("teacher_rollout",) + tuple(students)
    cases: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    teacher_relative: dict[str, list[dict[str, Any]]] = {name: [] for name in students}
    count = int(encoded_cache["count"])
    for batch_index in range(batches):
        selected = list(range(count)) if batch_size == count else [generator.randrange(count) for _ in range(batch_size)]
        context, actions, grounded = _batch_from_cache(encoded_cache, selected, device)
        with torch.no_grad():
            teacher = _teacher_targets(model, context, actions)
            predictions = {name: student(context, actions) for name, student in students.items()}
        teacher_case = _metrics_by_horizon(teacher, grounded)
        teacher_case["batch"] = batch_index
        cases["teacher_rollout"].append(teacher_case)
        for name, prediction in predictions.items():
            real_case = _metrics_by_horizon(prediction, grounded)
            real_case["batch"] = batch_index
            cases[name].append(real_case)
            relative_case = _metrics_by_horizon(prediction, teacher)
            relative_case["batch"] = batch_index
            teacher_relative[name].append(relative_case)

    def aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "batches": list(items),
            "per_horizon": {
                key: [_mean([float(item[key][h]) for item in items]) for h in range(HORIZON)]
                for key in ("relative_mse", "cosine")
            },
        }

    return {
        "real_future": {name: aggregate(values) for name, values in cases.items()},
        "teacher_relative": {name: aggregate(values) for name, values in teacher_relative.items()},
    }


def _evaluate_planner_blocks(
    students: Mapping[str, Any],
    model: Any,
    encoded_cache: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    context_count: int,
    eval_seeds: Sequence[int],
    batch_size: int,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate 8 contexts x 2 fresh seeds with one paired candidate batch."""

    import torch

    if context_count > int(encoded_cache["count"]):
        raise ValueError("held-out manifest has fewer contexts than evaluation requires")
    cases: dict[str, list[dict[str, Any]]] = {name: [] for name in students}
    blocks: list[dict[str, Any]] = []
    for seed in eval_seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for context_index in range(context_count):
            context_one = {
                key: value[context_index : context_index + 1].to(device)
                for key, value in encoded_cache["context"].items()
            }
            actions_one = encoded_cache["actions"][context_index : context_index + 1].to(device)
            grounded_one = {
                key: value[context_index : context_index + 1].to(device)
                for key, value in encoded_cache["grounded"].items()
            }
            context = _repeat_anchor(context_one, batch_size)
            goal = _repeat_anchor({key: value[:, -1:] for key, value in grounded_one.items()}, batch_size)
            actions = _sample_actions(generator, batch_size, action_dim, device)
            with torch.no_grad():
                teacher_target = _teacher_targets(model, context, actions)
                predictions = {name: student(context, actions) for name, student in students.items()}
            teacher_cost = _objective_cost(objective_fn, teacher_target, goal)
            rankings: dict[str, dict[str, float]] = {}
            for name, prediction in predictions.items():
                student_cost = _objective_cost(objective_fn, prediction, goal)
                ranking = {
                    "spearman": _spearman(teacher_cost, student_cost),
                    "top30_overlap": _topk_overlap(teacher_cost, student_cost),
                }
                rankings[name] = ranking
                fidelity = _metrics_by_horizon(prediction, teacher_target)
                fidelity["seed"] = int(seed)
                fidelity["context_index"] = int(context_index)
                fidelity["ranking"] = ranking
                cases[name].append(fidelity)
            block: dict[str, Any] = {
                "seed": int(seed),
                "context_index": int(context_index),
                "candidate_count": int(batch_size),
                "topk": min(30, int(batch_size)),
                "rankings": rankings,
            }
            for left, right, label in (
                ("wide_logged", "narrow_logged", "context_effect"),
                ("narrow_query", "narrow_logged", "query_effect_narrow"),
                ("wide_query", "wide_logged", "query_effect_wide"),
            ):
                block[f"spearman_delta_{label}"] = rankings[left]["spearman"] - rankings[right]["spearman"]
                block[f"top30_delta_{label}"] = rankings[left]["top30_overlap"] - rankings[right]["top30_overlap"]
            block["spearman_delta_interaction"] = (
                block["spearman_delta_query_effect_wide"] - block["spearman_delta_query_effect_narrow"]
            )
            block["top30_delta_interaction"] = (
                block["top30_delta_query_effect_wide"] - block["top30_delta_query_effect_narrow"]
            )
            blocks.append(block)

    ranking_result: dict[str, Any] = {}
    for name, arm_cases in cases.items():
        ranking = [item["ranking"] for item in arm_cases]
        ranking_result[name] = {
            "per_case": arm_cases,
            "per_horizon_teacher_relative": {
                key: [_mean([float(item[key][h]) for item in arm_cases]) for h in range(HORIZON)]
                for key in ("relative_mse", "cosine")
            },
            "terminal_ranking": {
                "spearman_mean": _mean([float(item["spearman"]) for item in ranking]),
                "spearman_median": _median([float(item["spearman"]) for item in ranking]),
                "spearman_minimum": min(float(item["spearman"]) for item in ranking),
                "top30_overlap_mean": _mean([float(item["top30_overlap"]) for item in ranking]),
                "top30_overlap_median": _median([float(item["top30_overlap"]) for item in ranking]),
                "top30_overlap_minimum": min(float(item["top30_overlap"]) for item in ranking),
                "block_count": len(arm_cases),
            },
        }

    def deltas(label: str, metric: str) -> list[float]:
        return [float(item[f"{metric}_delta_{label}"]) for item in blocks]

    effect_result = {
        "per_block": blocks,
        "context_effect": {
            "spearman": deltas("context_effect", "spearman"),
            "top30_overlap": deltas("context_effect", "top30"),
        },
        "query_effect_narrow": {
            "spearman": deltas("query_effect_narrow", "spearman"),
            "top30_overlap": deltas("query_effect_narrow", "top30"),
        },
        "query_effect_wide": {
            "spearman": deltas("query_effect_wide", "spearman"),
            "top30_overlap": deltas("query_effect_wide", "top30"),
        },
        "interaction": {
            "spearman": [float(item["spearman_delta_interaction"]) for item in blocks],
            "top30_overlap": [float(item["top30_delta_interaction"]) for item in blocks],
        },
    }
    for effect in ("context_effect", "query_effect_narrow", "query_effect_wide", "interaction"):
        for metric in ("spearman", "top30_overlap"):
            values = effect_result[effect][metric]
            effect_result[effect][f"{metric}_mean"] = _mean(values)
            effect_result[effect][f"{metric}_median"] = _median(values)
            effect_result[effect][f"{metric}_positive_blocks"] = sum(value > 0 for value in values)
    return ranking_result, effect_result


def _threshold(group: Mapping[str, Any], default: float, *keys: str) -> float:
    for key in keys:
        if key in group:
            return float(group[key])
    return default


def _integer_threshold(group: Mapping[str, Any], default: int, *keys: str) -> int:
    for key in keys:
        if key in group:
            return int(group[key])
    return default


def _gate_summary(protocol: Mapping[str, Any], result: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, Any]:
    gates = protocol.get("gates", {})
    if not isinstance(gates, Mapping):
        gates = {}
    capacity = gates.get("capacity", {})
    fidelity = gates.get("absolute_fidelity", gates.get("fidelity", {}))
    real_gate = gates.get("real_future", gates.get("teacher_relative", gates.get("noninferiority", {})))
    if not real_gate:
        real_gate = gates.get("latent_noninferiority", {})
    latency_gate = gates.get("latency", {})
    effects = gates.get("effects", {})
    if not isinstance(effects, Mapping):
        effects = {}
    if "action_query_effect" in gates and "query_effect_wide" not in effects:
        effects = {**effects, "query_effect_wide": gates["action_query_effect"]}
    if not all(isinstance(item, Mapping) for item in (capacity, fidelity, real_gate, latency_gate, effects)):
        raise ValueError("query-coverage frozen gates must be objects")
    blocks = result["planner_evaluation"]["effects"]["per_block"]
    block_count = len(blocks)
    positive_default = max(1, math.ceil(0.75 * block_count))
    max_loss_ratio = _threshold(capacity, 0.8, "latent_last10_to_first_ratio_max", "last10_to_first_training_mse_ratio_max")
    causal_pass = all(bool(value["passed"]) for value in result["causality"].values())
    finite = bool(result["finite_outputs_and_training"])
    train_pass = all(float(result["train"][name]["latent_last10_to_first_ratio"]) <= max_loss_ratio for name in ARM_NAMES)
    capacity_pass = finite and causal_pass and train_pass
    primary_name = "wide_query"
    primary_rank = result["planner_evaluation"]["arms"][primary_name]["terminal_ranking"]
    spearman_min = _threshold(
        fidelity, 0.99, "median_spearman_min", "spearman_median_min", "objective_spearman_min"
    )
    spearman_block_min = _threshold(fidelity, 0.95, "spearman_minimum_min", "minimum_spearman_min", "minimum_block_spearman_min")
    top30_min = _threshold(fidelity, 0.95, "top30_overlap_median_min", "median_top30_min", "top30_min")
    top30_block_min = _threshold(fidelity, 0.80, "top30_overlap_minimum_min", "minimum_top30_min", "minimum_block_top30_overlap_min")
    fidelity_pass = (
        primary_rank["spearman_median"] >= spearman_min
        and primary_rank["spearman_minimum"] >= spearman_block_min
        and primary_rank["top30_overlap_median"] >= top30_min
        and primary_rank["top30_overlap_minimum"] >= top30_block_min
    )
    real_ratio_max = _threshold(real_gate, 1.25, "student_to_teacher_relative_mse_ratio_max", "mean_relative_mse_ratio_max", "relative_mse_ratio_max")
    real_ratios = result["real_future_fidelity"]["real_future_ratios"]
    teacher_relative_means = result["teacher_relative_mean_relative_mse"]
    wide_query_to_logged = _safe_ratio(teacher_relative_means["wide_query"], teacher_relative_means["wide_logged"])
    real_pass = wide_query_to_logged <= real_ratio_max
    reduction_min = _threshold(latency_gate, 0.20, "query_predictor_reduction_vs_teacher_min", "predictor_only_reduction_min", "predictor_reduction_vs_teacher_min")
    reduction = float(result["latency"]["wide_query"]["median_reduction"])
    latency_pass = reduction >= reduction_min

    def effect_gate(name: str) -> dict[str, Any]:
        values = result["planner_evaluation"]["effects"][name]
        group = effects.get(name, {})
        if not isinstance(group, Mapping):
            group = {}
        spearman_threshold = _threshold(group, 0.05, "median_spearman_delta_min", "required_median_spearman_delta", "spearman_median_min")
        top30_threshold = _threshold(group, 0.10, "median_top30_delta_min", "required_median_top30_delta", "top30_median_min")
        positive_threshold = _integer_threshold(
            group,
            positive_default,
            "positive_spearman_blocks_min",
            "positive_blocks_min",
            "required_positive_spearman_blocks",
        )
        positive_top30_threshold = _integer_threshold(group, positive_threshold, "positive_top30_blocks_min", "required_positive_top30_blocks")
        spearman_ok = values["spearman_median"] >= spearman_threshold
        top30_ok = values["top30_overlap_median"] >= top30_threshold
        positive_ok = (
            values["spearman_positive_blocks"] >= positive_threshold
            and values["top30_overlap_positive_blocks"] >= positive_top30_threshold
        )
        return {
            "status": "PASS" if spearman_ok and top30_ok and positive_ok else "FAIL",
            "median_spearman_delta": values["spearman_median"],
            "median_spearman_delta_minimum": spearman_threshold,
            "median_top30_delta": values["top30_overlap_median"],
            "median_top30_delta_minimum": top30_threshold,
            "spearman_positive_blocks": values["spearman_positive_blocks"],
            "top30_positive_blocks": values["top30_overlap_positive_blocks"],
            "positive_blocks_minimum": positive_threshold,
            "positive_top30_blocks_minimum": positive_top30_threshold,
        }

    effect_gates = {name: effect_gate(name) for name in ("context_effect", "query_effect_narrow", "query_effect_wide")}
    action_effect_pass = effect_gates["query_effect_wide"]["status"] == "PASS"
    primary_pass = capacity_pass and fidelity_pass and action_effect_pass and real_pass and latency_pass
    return {
        "capacity": {
            "status": "PASS" if capacity_pass else "FAIL",
            "finite_outputs_and_training": finite,
            "causality": causal_pass,
            "training_loss_ratio_maximum": max_loss_ratio,
            "all_arms_loss_ratio_within_gate": train_pass,
        },
        "absolute_fidelity": {
            "primary_arm": primary_name,
            "status": "PASS" if fidelity_pass else "FAIL",
            "spearman_median": primary_rank["spearman_median"],
            "spearman_median_minimum": spearman_min,
            "spearman_block_minimum": primary_rank["spearman_minimum"],
            "spearman_block_minimum_threshold": spearman_block_min,
            "top30_overlap_median": primary_rank["top30_overlap_median"],
            "top30_overlap_median_minimum": top30_min,
            "top30_overlap_block_minimum": primary_rank["top30_overlap_minimum"],
            "top30_overlap_block_minimum_threshold": top30_block_min,
        },
        "real_future_teacher_relative": {
            "status": "PASS" if real_pass else "FAIL",
            "student_to_teacher_real_future_relative_mse_ratios": real_ratios,
            "wide_query_to_wide_logged_teacher_relative_mse_ratio": wide_query_to_logged,
            "maximum_ratio": real_ratio_max,
        },
        "latency": {
            "status": "PASS" if latency_pass else "FAIL",
            "primary_arm": "wide_query",
            "median_reduction": reduction,
            "minimum_reduction": reduction_min,
        },
        "effects": effect_gates,
        "overall": "PASS" if primary_pass else "FAIL",
        "primary_effect": "query_effect_wide",
        "context_and_narrow_query_effects_are_diagnostic": True,
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
    settings = _query_settings(freeze, protocol)
    query = _query_spec(protocol, freeze)
    _ensure_query_freeze(freeze, query)
    randomness = _frozen_randomness(protocol, freeze, settings)
    manifest = _load_query_manifest(args.manifest.resolve(), settings)
    _require_assets(asset_root, freeze)
    if settings["batch"] < 1 or settings["steps"] < 1:
        raise ValueError("frozen training steps and batch must be positive")
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
    template = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, settings["hidden_dim"]).to(device)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    students = {
        name: NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, settings["hidden_dim"]).to(device)
        for name in ARM_NAMES
    }
    for student in students.values():
        student.load_state_dict(initial_state)
    optimizers = {name: torch.optim.AdamW(student.parameters(), lr=settings["lr"]) for name, student in students.items()}
    losses: dict[str, list[dict[str, float]]] = {name: [] for name in ARM_NAMES}
    mixture_counts = {"logged": 16, "gaussian_planner_init": 8, "cem_elite_resample": 8}
    telemetry_period = max(1, settings["steps"] // 10)
    narrow_episode_ids: list[int] = []
    for example in train_examples:
        episode_id = int(example["episode_id"])
        if episode_id not in narrow_episode_ids:
            narrow_episode_ids.append(episode_id)
        if len(narrow_episode_ids) == 2:
            break
    narrow_indices = [
        index for index, example in enumerate(train_examples) if int(example["episode_id"]) in narrow_episode_ids
    ]
    if len(narrow_episode_ids) < 2 or not narrow_indices:
        raise ValueError("manifest train split must contain at least two episodes for narrow arms")

    # Freeze all context indices, role permutations, action proposals, and
    # dense teacher targets before the first optimizer update.  The training
    # loop below therefore contains no teacher rollout or action sampling.
    context_generator = random.Random(randomness["context_schedule_seed"])
    narrow_schedule = [
        [narrow_indices[context_generator.randrange(len(narrow_indices))] for _ in range(settings["batch"])]
        for _ in range(settings["steps"])
    ]
    wide_schedule = [
        [context_generator.randrange(len(train_examples)) for _ in range(settings["batch"])]
        for _ in range(settings["steps"])
    ]
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

    for step in range(settings["steps"]):
        narrow_selected = narrow_schedule[step]
        wide_selected = wide_schedule[step]
        narrow_logged_roles = torch.zeros(settings["batch"], dtype=torch.long)
        wide_logged_roles = torch.zeros(settings["batch"], dtype=torch.long)
        narrow_query_roles = role_schedule[step]
        wide_query_roles = role_schedule[step]
        narrow_logged = _materialize_bank_batch(
            train_encoded, action_banks, narrow_selected, narrow_logged_roles, device
        )
        wide_logged = _materialize_bank_batch(
            train_encoded, action_banks, wide_selected, wide_logged_roles, device
        )
        narrow_query = _materialize_bank_batch(
            train_encoded, action_banks, narrow_selected, narrow_query_roles, device
        )
        wide_query = _materialize_bank_batch(
            train_encoded, action_banks, wide_selected, wide_query_roles, device
        )
        batches = {
            "narrow_logged": narrow_logged,
            "wide_logged": wide_logged,
            "narrow_query": narrow_query,
            "wide_query": wide_query,
        }
        order = ARM_NAMES if step % 2 == 0 else tuple(reversed(ARM_NAMES))
        for name in order:
            context, actions, target = batches[name]
            losses[name].append(
                _train_arm(students[name], optimizers[name], context, actions, target, step)
            )
        if step == 0 or (step + 1) % telemetry_period == 0:
            _gpu_snapshot(output, f"train_step_{step + 1}")

    planner_arms, effects = _evaluate_planner_blocks(
        students,
        model,
        heldout_encoded,
        objective_fn,
        action_dim,
        query["evaluation_contexts"],
        query["evaluation_action_seeds"][:2],
        settings["eval_batch"],
        device,
    )
    real_eval = _evaluate_real_and_teacher_relative(
        students,
        model,
        heldout_encoded,
        len(heldout_examples),
        1,
        settings["train_seed"] + 1,
        device,
    )
    real_future_ratios: dict[str, float] = {}
    real_future_means: dict[str, float] = {}
    teacher_relative_means: dict[str, float] = {}
    teacher_real = _mean(real_eval["real_future"]["teacher_rollout"]["per_horizon"]["relative_mse"])
    for name in ARM_NAMES:
        student_real = _mean(real_eval["real_future"][name]["per_horizon"]["relative_mse"])
        teacher_relative_mean = _mean(real_eval["teacher_relative"][name]["per_horizon"]["relative_mse"])
        real_future_ratios[name] = _safe_ratio(student_real, teacher_real)
        real_future_means[name] = student_real
        teacher_relative_means[name] = teacher_relative_mean
    causality = {
        name: _leakage_test(
            student,
            {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()},
            action_dim,
            query["evaluation_action_seeds"][0] + 9000,
            device,
            settings["future_action_tolerance"],
        )
        for name, student in students.items()
    }
    latency = {
        name: _latency(
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
        for name, student in students.items()
    }
    _gpu_snapshot(output, "complete")

    def arm_train(name: str) -> dict[str, Any]:
        values = [item["latent_mse"] for item in losses[name]]
        last10 = values[-10:]
        return {
            "steps": settings["steps"],
            "batch": settings["batch"],
            "loss_first": values[0],
            "loss_last": values[-1],
            "latent_mse_median_last_10": _median(last10),
            "latent_last10_to_first_ratio": _safe_ratio(_median(last10), values[0]),
            "per_step": losses[name],
        }

    result: dict[str, Any] = {
        "train": {name: arm_train(name) for name in ARM_NAMES},
        "planner_evaluation": {"arms": planner_arms, "effects": effects},
        "real_future_fidelity": real_eval["real_future"],
        "teacher_relative_fidelity": real_eval["teacher_relative"],
        "real_future_teacher_relative_ratios": real_future_ratios,
        "real_future_mean_relative_mse": real_future_means,
        "teacher_relative_mean_relative_mse": teacher_relative_means,
        "teacher_relative_mse": teacher_relative_means,
        "real_future_fidelity_summary": {
            "teacher_rollout_mean_relative_mse": teacher_real,
            "student_to_teacher_real_future_relative_mse_ratios": real_future_ratios,
            "student_to_teacher_target_relative_mse": teacher_relative_means,
        },
        "causality": causality,
        "latency": latency,
    }
    result["finite_outputs_and_training"] = _all_finite(result)
    result["real_future_fidelity"]["real_future_ratios"] = real_future_ratios
    gates = _gate_summary(protocol, result, query)
    checkpoint_common = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-query-coverage-checkpoint",
        "horizon": HORIZON,
        "native_output_fields": ["visual", "proprio"],
        "action_dim": action_dim,
        "visual_dim": visual_dim,
        "proprio_dim": proprio_dim,
        "hidden_dim": settings["hidden_dim"],
        "train_seed": settings["train_seed"],
        "manifest": str(args.manifest.resolve()),
        "query_coverage": query,
    }
    for name, student in students.items():
        torch.save({**checkpoint_common, "arm": name, "state_dict": student.state_dict()}, output / f"{name}_student.pt")

    summary = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-query-coverage-summary",
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "freeze": str(args.freeze.resolve()),
        "manifest": str(args.manifest.resolve()),
        "source": {
            "official_loader": "run_dino_pusht_stage_a._load_official",
            "student": "NativeDinoPrefixStudent shared with grounded-prefix runner",
            "teacher_target": "frozen teacher rollout dense latent MSE for every arm",
            "preencoding": "train and heldout manifest segments encoded once and retained as CPU native latents",
        },
        "contracts": {
            "arm_labels": {
                "NARROW-LOGGED": "narrow_logged",
                "WIDE-LOGGED": "wide_logged",
                "NARROW-QUERY": "narrow_query",
                "WIDE-QUERY": "wide_query",
            },
            "same_initial_state_dict": True,
            "same_steps_batch_optimizer": True,
            "goal_in_student_input": False,
            "query_mixture": query,
            "query_mixture_counts_per_batch": mixture_counts,
            "narrow_train_episode_ids": narrow_episode_ids,
            "narrow_train_examples": len(narrow_indices),
            "wide_train_examples": len(train_examples),
            "heldout_examples": len(heldout_examples),
            "heldout_evaluation_contexts": query["evaluation_contexts"],
            "heldout_evaluation_seeds": query["evaluation_action_seeds"][:2],
            "heldout_paired_blocks": query["evaluation_contexts"] * 2,
            "candidate_count_per_block": settings["eval_batch"],
        },
        "train": result["train"],
        "planner_evaluation": result["planner_evaluation"],
        "real_future_fidelity": result["real_future_fidelity"],
        "teacher_relative_fidelity": result["teacher_relative_fidelity"],
        "real_future_teacher_relative_ratios": result["real_future_teacher_relative_ratios"],
        "real_future_mean_relative_mse": result["real_future_mean_relative_mse"],
        "teacher_relative_mean_relative_mse": result["teacher_relative_mean_relative_mse"],
        "causality": causality,
        "latency": latency,
        "gates": gates,
        "timing_boundary": "predictor-level cached native observation latent plus normalized action prefix; excludes observation encoding, CEM execution, environment interaction and closed-loop control",
        "unverified": [
            "No closed-loop CEM or environment execution is included.",
            "Held-out rankings use each context's real terminal future latent as the official objective goal.",
            "The finite manifest does not support population-level inference or a full-data claim.",
        ],
    }
    summary_path = args.summary.resolve() if args.summary else output / "query_coverage_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2, default=_json_default)
    summary_path.write_text(summary_text, encoding="utf-8")
    default_summary = output / "query_coverage_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(summary_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
