#!/usr/bin/env python3
"""LeWM balanced_base Stage A teacher-screening experiment.

This runner deliberately keeps the new screening hypothesis separate from the
older predictor-level runners.  It imports their tested model/data adapters,
but it never edits or reuses their result artifacts as if they were a saved
balanced_base checkpoint.  If the Phase 6 checkpoint is absent, the balanced
base is reconstructed once from the frozen Phase 5 bank and its step-3000
state is saved in the current PBS output with an explicit ``reconstructed``
provenance marker.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
REFERENCE_DIR = TRANSFER / "state-action-prefix-gru"
SCHEMA = "lewm-recurrent-student.teacher-screening-runner"
LATENT_DIM = 192
HORIZON = 5
ACTION_DIM = 10
HIDDEN_DIM = 256
TRAIN_CONTEXTS = 512
TRAIN_CANDIDATES = 64
BATCH_CONTEXTS = 8
TRAIN_STEPS = 3000
FRESH_EPISODES = 8
FRESH_ANCHORS = ("early", "middle", "late")
FRESH_SEEDS = (20300947, 20300948)
SELECTION_SEED = 20300903
FRESH_SLICE = (544, 552)
NUM_CANDIDATES = 300
ELITE_COUNT = 30
SHORTLIST_SIZES = (60, 120)
RECONSTRUCTION_SEED = 20300901
TRAINING_SEED = 20300902
CONTEXT_SCHEDULE_SEED = 20300904
CANDIDATE_SLATE_SEED = 20300905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--prepared-rows-512", type=Path, required=True)
    parser.add_argument("--prepared-balanced-rows", type=Path, default=None)
    parser.add_argument("--balanced-base-checkpoint", type=Path, default=None)
    parser.add_argument("--temporal-freeze", type=Path, default=None)
    parser.add_argument("--reference-summary", type=Path, default=None)
    parser.add_argument("--phase2-freeze", type=Path, default=None)
    parser.add_argument("--temporal-summary", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--mode", choices=("status", "preflight", "run"), default="status")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference_modules() -> Any:
    """Import the existing tested LeWM adapters without copying their code."""
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    return importlib.import_module("run_lewm_state_action_prefix_gru")


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required for model/HDF5/benchmark work")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "lewm-recurrent-student.teacher-screening-freeze":
        raise ValueError("unexpected teacher-screening freeze schema")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unsupported teacher-screening freeze schema version")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("teacher-screening freeze must be frozen_before_results")
    student = freeze.get("student", {})
    if student.get("architecture") != "LeWMCompactRecurrentTransitionStudent" or int(student.get("hidden_dim", -1)) != HIDDEN_DIM:
        raise ValueError("Stage A must use the unchanged h256 LeWM student")
    if student.get("snapshot") != "step_3000" or student.get("training_changes") is not False:
        raise ValueError("Stage A snapshot/training contract drifted")
    stage = freeze.get("stage_a", {})
    if stage.get("selection_slice") != list(FRESH_SLICE) or int(stage.get("excluded_valid_prefix", -1)) != FRESH_SLICE[0]:
        raise ValueError("fresh valid[544:552] selection contract drifted")
    if int(stage.get("selection_seed", -1)) != SELECTION_SEED:
        raise ValueError("selection seed drifted")
    if list(stage.get("action_prefix_seeds", [])) != list(FRESH_SEEDS):
        raise ValueError("fresh action-prefix seeds drifted")
    if int(stage.get("episodes", -1)) != FRESH_EPISODES or int(stage.get("blocks", -1)) != 48:
        raise ValueError("fresh episode/block contract drifted")
    if int(stage.get("num_candidates", -1)) != NUM_CANDIDATES or int(stage.get("elite_count", -1)) != ELITE_COUNT:
        raise ValueError("candidate/elite count drifted")
    if tuple(int(x) for x in stage.get("shortlist_sizes", [])) != SHORTLIST_SIZES:
        raise ValueError("shortlist sizes drifted")
    gate = stage.get("gate", {})
    for key, expected in {
        "overall_median_recall_min": 0.95,
        "minimum_block_recall_min": 0.80,
        "each_stratum_median_recall_min": 0.95,
        "hybrid_latency_reduction_min": 0.20,
    }.items():
        if not math.isclose(float(gate.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Stage A gate drifted: {key}")
    timing = stage.get("timing", {})
    if int(timing.get("warmup", -1)) != 3 or int(timing.get("repeats", -1)) != 10:
        raise ValueError("timing warmup/repeat contract drifted")
    if int(timing.get("order_seed", -1)) != 20300949:
        raise ValueError("timing order seed drifted")
    if timing.get("cuda_synchronize") is not True or timing.get("interleaved_arm_order") is not True:
        raise ValueError("CUDA/interleaved timing contract drifted")
    if timing.get("shadow_teacher_excluded_from_hybrid_timing") is not True:
        raise ValueError("shadow teacher must be excluded from hybrid timing")
    if freeze.get("scope", {}).get("closed_loop") != "NOT_RUN_BY_SCOPE":
        raise ValueError("closed-loop scope boundary drifted")
    return dict(freeze)


def validate_protocol(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = ("valid[544:552]", "20300947/20300948", "top-30 recall@K", "closed-loop")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol is missing frozen Stage A terms: {missing}")


def load_candidate_generation_contract(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Read the old temporal-balanced freeze only for its frozen slate values.

    The new teacher-screening freeze intentionally has no legacy ``evaluation``
    section.  The shared row builder still expects that small legacy section, so
    this adapter supplies it from the frozen temporal-balanced file and leaves
    all screening seeds controlled by this runner.
    """
    path = args.temporal_freeze.resolve() if args.temporal_freeze else TRANSFER / "temporal-balanced-train" / "LEWM_TEMPORAL_BALANCED_TRAIN_FREEZE.json"
    temporal = load_json(path)
    if temporal.get("schema") != "lewm-recurrent-student.temporal-balanced-train-freeze" or temporal.get("status") != "frozen":
        raise ValueError("temporal-balanced candidate freeze is not frozen")
    evaluation = temporal.get("evaluation")
    required = ("official_action_low", "official_action_high", "gaussian_std")
    if not isinstance(evaluation, Mapping) or any(key not in evaluation for key in required):
        raise ValueError("temporal-balanced freeze lacks the candidate generation contract")
    if list(evaluation["official_action_low"]) != [-1.0, -1.0] or list(evaluation["official_action_high"]) != [1.0, 1.0]:
        raise ValueError("official action bounds drifted from the frozen temporal-balanced contract")
    if not math.isclose(float(evaluation["gaussian_std"]), 0.22360679774997896, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("candidate Gaussian std drifted from the frozen temporal-balanced contract")
    return dict(evaluation), path


def validate_interface(contract: Any) -> dict[str, Any]:
    expected = {
        "observation_history_h": 1,
        "official_future_prediction_count": HORIZON,
        "student_probe_future_count": HORIZON,
        "official_predicted_emb_shape": (1, 2, 6, LATENT_DIM),
        "candidate_shape": (1, 2, HORIZON, ACTION_DIM),
        "semantics_equal": True,
        "raw_action_dim": 2,
    }
    for key, value in expected.items():
        if getattr(contract, key) != value:
            raise ValueError(f"interface contract drifted: {key}={getattr(contract, key)!r}")
    return {key: (list(value) if isinstance(value, tuple) else value) for key, value in expected.items()}


def validate_training_rows(rows: Sequence[Mapping[str, Any]], reference: Any) -> dict[str, Any]:
    import torch

    train = [row for row in rows if row.get("split") == "train"]
    if len(train) != TRAIN_CONTEXTS:
        raise ValueError(f"Phase2 prepared rows must contain {TRAIN_CONTEXTS} train rows")
    first = train[0]
    actions = torch.as_tensor(first["future_actions"])
    if tuple(actions.shape) != (TRAIN_CANDIDATES, HORIZON, ACTION_DIM):
        raise ValueError(f"prepared candidate shape drifted: {tuple(actions.shape)}")
    return {
        "rows": len(rows),
        "train_contexts": len(train),
        "train_candidates": int(actions.shape[0]),
        "latent_shape": list(torch.as_tensor(first["latent_history"]).shape),
        "candidate_shape": list(actions.shape),
        "source": "phase2_prepared_rows",
    }


def validate_balanced_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    train = [row for row in rows if row.get("split") == "train"]
    if len(train) != TRAIN_CONTEXTS:
        raise ValueError(f"balanced rows must contain {TRAIN_CONTEXTS} train rows")
    counts = {name: sum(row.get("stratum") == name for row in train) for name in FRESH_ANCHORS}
    if counts != {"early": 171, "middle": 171, "late": 170}:
        raise ValueError(f"balanced anchor counts drifted: {counts}")
    action_shape = tuple(torch.as_tensor(train[0]["future_actions"]).shape)
    target_shape = tuple(torch.as_tensor(train[0]["teacher_targets"]).shape)
    if action_shape != (TRAIN_CANDIDATES, HORIZON, ACTION_DIM) or target_shape != (TRAIN_CANDIDATES, HORIZON, LATENT_DIM):
        raise ValueError(f"balanced row shapes drifted: actions={action_shape}, targets={target_shape}")
    return {
        "train_contexts": len(train),
        "anchor_counts": counts,
        "candidate_shape": list(action_shape),
        "target_shape": list(target_shape),
        "source": "phase5_prepared_balanced_rows",
    }


def select_fresh_episodes(dataset_path: Path, manifest: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        lengths = [int(x) for x in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= 26]
    random.Random(SELECTION_SEED).shuffle(valid)
    old_heldout = [int(row["episode_id"]) for row in manifest["splits"]["heldout"]]
    old_train = [int(row["episode_id"]) for row in manifest["splits"]["train"]]
    prefix = old_heldout + old_train
    if valid[: len(prefix)] != prefix:
        raise ValueError("selection prefix does not reproduce Phase2 manifest")
    fresh = valid[FRESH_SLICE[0] : FRESH_SLICE[1]]
    if len(fresh) != FRESH_EPISODES or set(fresh) & set(valid[: FRESH_SLICE[0]]):
        raise ValueError("fresh valid[544:552] selection overlaps excluded prefix")
    return fresh, {
        "selection_seed": SELECTION_SEED,
        "valid_count": len(valid),
        "old_heldout_episode_ids": old_heldout,
        "old_train_episode_count": len(old_train),
        "excluded_valid_prefix": FRESH_SLICE[0],
        "fresh_episode_ids": fresh,
        "selection_slice": "valid[544:552]",
        "excluded_prior_fresh_slices": ["valid[520:528]", "valid[528:536]", "valid[536:544]"],
        "result_dependent_selection": False,
    }


def load_or_reconstruct_checkpoint(
    args: argparse.Namespace,
    freeze: Mapping[str, Any],
    reference: Any,
    official_model: Any,
    balanced_rows: Sequence[Mapping[str, Any]] | None,
    initial_state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], Sequence[Mapping[str, Any]] | None]:
    import torch

    checkpoint_path = args.balanced_base_checkpoint.resolve() if args.balanced_base_checkpoint else None
    if checkpoint_path is not None and checkpoint_path.is_file():
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        provenance = raw.get("provenance", {}) if isinstance(raw, Mapping) else {}
        state = raw["state_dict"] if isinstance(raw, Mapping) and "state_dict" in raw else raw
        if not isinstance(state, Mapping):
            raise ValueError("balanced-base checkpoint must contain a state dict")
        student = reference.base.make_student("baseline")
        student.load_state_dict({key: value.detach().clone() for key, value in state.items()}, strict=True)
        if sum(parameter.numel() for parameter in student.parameters()) != 775872:
            raise ValueError("exact checkpoint parameter count drifted")
        student = student.to("cuda").eval()
        reconstructed = bool(isinstance(provenance, Mapping) and (provenance.get("reconstructed") is True or provenance.get("source") == "reconstructed"))
        return student, {
            "source": "reconstructed" if reconstructed else "exact_checkpoint",
            "path": str(checkpoint_path),
            "reconstructed": reconstructed,
            "source_job": freeze["student"]["source_job"],
            "original_checkpoint_available": not reconstructed,
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else {},
        }, balanced_rows

    if balanced_rows is None:
        raise ValueError("balanced rows are required to reconstruct the missing checkpoint")
    settings = {
        "training_seed": TRAINING_SEED,
        "context_schedule_seed": CONTEXT_SCHEDULE_SEED,
        "prefix_initialization_seed": 20300906,
    }
    training = reference.train_arm(balanced_rows, initial_state, official_model, settings, "balanced_base")
    state = training["snapshots"]["step_3000"]
    output_checkpoint = args.output.resolve() / "balanced_base_step3000_reconstructed.pt"
    torch.save(
        {
            "state_dict": state,
            "provenance": {
                "source": "reconstructed",
                "reconstructed": True,
                "original_checkpoint_available": False,
                "source_job": freeze["student"]["source_job"],
            },
        },
        output_checkpoint,
    )
    student = reference.instantiate_student("balanced_base")
    reference.load_full_state(student, state)
    student = student.to("cuda").eval()
    return student, {
        "source": "reconstructed",
        "path": str(output_checkpoint),
        "reconstructed": True,
        "source_job": freeze["student"]["source_job"],
        "original_checkpoint_available": False,
        "reconstruction": {
            "balanced_rows": "phase5_prepared_balanced_rows_or_deterministic_rebuild",
            "initialization_seed": RECONSTRUCTION_SEED,
            "training_seed": TRAINING_SEED,
            "context_schedule_seed": CONTEXT_SCHEDULE_SEED,
            "candidate_slate_seed": CANDIDATE_SLATE_SEED,
            "updates": TRAIN_STEPS,
            "architecture": "LeWMCompactRecurrentTransitionStudent h256",
            "objective": "horizon-weighted free-running latent MSE + 0.1 * context-normalized teacher-score SmoothL1",
        },
        "training_trace": {key: value for key, value in training.items() if key != "snapshots"},
    }, balanced_rows


def score_student(reference: Any, official_model: Any, student: Any, row: Mapping[str, Any], actions: Any) -> Any:
    import torch

    context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
    context = context.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        prediction = student(context, actions)
        costs = reference.base._official_objective(official_model, row, prediction)
    return costs


def score_teacher(reference: Any, official_model: Any, row: Mapping[str, Any], actions: Any) -> Any:
    import torch

    context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
    context = context.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        targets = reference.base.official_teacher_targets(official_model, context, actions)
        costs = reference.base._official_objective(official_model, row, targets)
    return costs


def build_fresh_rows_compat(
    args: argparse.Namespace,
    reference: Any,
    official_model: Any,
    dataset: Path,
    fresh_ids: Sequence[int],
    freeze: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Use the tested builder with an explicit legacy-contract adapter."""
    import copy

    evaluation, source_path = load_candidate_generation_contract(args)
    adapter_freeze = copy.deepcopy(dict(freeze))
    adapter_freeze["evaluation"] = evaluation
    previous_seeds = tuple(reference.ema.FRESH_SEEDS)
    previous_anchors = tuple(reference.ema.FRESH_ANCHORS)
    reference.ema.FRESH_SEEDS = FRESH_SEEDS
    reference.ema.FRESH_ANCHORS = FRESH_ANCHORS
    try:
        rows, metadata = reference.ema.build_fresh_rows(official_model, dataset, fresh_ids, adapter_freeze)
    finally:
        reference.ema.FRESH_SEEDS = previous_seeds
        reference.ema.FRESH_ANCHORS = previous_anchors
    if metadata.get("candidate_seeds") is not None and list(metadata["candidate_seeds"]) != list(FRESH_SEEDS):
        raise ValueError("fresh-row builder did not use the Stage A action-prefix seeds")
    if len(rows) != FRESH_EPISODES * len(FRESH_ANCHORS):
        raise ValueError("fresh-row builder returned an unexpected anchor count")
    if metadata.get("blocks") != len(rows) * len(FRESH_SEEDS):
        raise ValueError("fresh-row builder returned an unexpected block count")
    details = metadata.get("contexts_detail", [])
    if len(details) != len(rows) or {item.get("stratum") for item in details} != set(FRESH_ANCHORS):
        raise ValueError("fresh-row builder anchor/stratum metadata drifted")
    if any(list(item.get("candidate_seeds", [])) != list(FRESH_SEEDS) or int(item.get("candidate_count_per_block", -1)) != NUM_CANDIDATES for item in details):
        raise ValueError("fresh-row builder candidate seed/count metadata drifted")
    return rows, {
        **metadata,
        "candidate_generation_contract_source": str(source_path),
        "candidate_generation_contract": {
            "official_action_low": list(evaluation["official_action_low"]),
            "official_action_high": list(evaluation["official_action_high"]),
            "gaussian_std": float(evaluation["gaussian_std"]),
        },
        "seeds_overridden_only_for_this_call": list(FRESH_SEEDS),
        "anchors_overridden_only_for_this_call": list(FRESH_ANCHORS),
    }


def stable_order(values: Any) -> Any:
    import torch

    return torch.argsort(values, stable=True)


def hybrid_block(reference: Any, official_model: Any, student: Any, row: Mapping[str, Any], block: int, k: int, full_teacher_cost: Any | None = None) -> dict[str, Any]:
    import torch

    actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
    student_cost = score_student(reference, official_model, student, row, actions)
    student_order = stable_order(student_cost)
    shortlist = student_order[:k]
    shortlisted_actions = actions.index_select(0, shortlist)
    verified_cost = score_teacher(reference, official_model, row, shortlisted_actions)
    local_elites = stable_order(verified_cost)[:ELITE_COUNT]
    hybrid_indices = shortlist.index_select(0, local_elites)
    if full_teacher_cost is None:
        full_teacher_cost = score_teacher(reference, official_model, row, actions)
    full_order = stable_order(full_teacher_cost)
    full_top30 = full_order[:ELITE_COUNT]
    captured = torch.isin(full_top30, shortlist).sum()
    teacher_mean = full_teacher_cost.index_select(0, full_top30).mean()
    hybrid_mean = full_teacher_cost.index_select(0, hybrid_indices).mean()
    finite = bool(torch.isfinite(student_cost).all() and torch.isfinite(full_teacher_cost).all() and torch.isfinite(verified_cost).all())
    return {
        "teacher_top30_recall_at_k": float(captured.detach().cpu()) / ELITE_COUNT,
        "full_elite_containment": bool(int(captured.detach().cpu()) == ELITE_COUNT),
        "teacher_elite_mean_cost_regret": float((hybrid_mean - teacher_mean).detach().cpu()),
        "screened_teacher_cost_regret": float((verified_cost.index_select(0, local_elites).mean() - teacher_mean).detach().cpu()),
        "student_shortlist_count": int(k),
        "candidate_count": int(actions.shape[0]),
        "elite_count": ELITE_COUNT,
        "finite": finite,
    }


def aggregate_metric(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot aggregate empty Stage A blocks")
    recalls = [float(item["teacher_top30_recall_at_k"]) for item in items]
    regrets = [float(item["teacher_elite_mean_cost_regret"]) for item in items]
    return {
        "blocks": len(items),
        "recall_median": float(statistics.median(recalls)),
        "recall_minimum": min(recalls),
        "full_elite_containment_rate": float(sum(bool(item["full_elite_containment"]) for item in items) / len(items)),
        "teacher_elite_mean_cost_regret_median": float(statistics.median(regrets)),
        "teacher_elite_mean_cost_regret_maximum": max(regrets),
        "finite": all(bool(item["finite"]) for item in items),
    }


def group_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strata = {name: aggregate_metric([item for item in items if item["stratum"] == name]) for name in FRESH_ANCHORS}
    episode_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        episode_groups[str(item["episode_id"])].append(item)
    episodes = {episode: aggregate_metric(group) for episode, group in sorted(episode_groups.items(), key=lambda pair: int(pair[0]))}
    return {"overall": aggregate_metric(items), "strata": strata, "episodes": episodes}


def time_native(fn: Any) -> tuple[dict[str, float], Any]:
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    torch.cuda.synchronize()
    return {"wall_ms": (time.perf_counter() - wall_start) * 1000.0, "cuda_ms": float(start.elapsed_time(end))}, value


def timing_for_block(reference: Any, official_model: Any, student: Any, row: Mapping[str, Any], block: int, freeze: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    stage = freeze["stage_a"]
    warmup = int(stage["timing"]["warmup"])
    repeats = int(stage["timing"]["repeats"])
    order_rng = random.Random(int(stage["timing"]["order_seed"]))
    order = [("teacher", 0)] + [("hybrid", int(k)) for k in SHORTLIST_SIZES]
    order_rng.shuffle(order)
    actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")

    def teacher_run() -> Any:
        teacher_cost = score_teacher(reference, official_model, row, actions)
        return stable_order(teacher_cost)[:ELITE_COUNT]

    def hybrid_run(k: int) -> Any:
        student_cost = score_student(reference, official_model, student, row, actions)
        shortlist = stable_order(student_cost)[:k]
        shortlist_actions = actions.index_select(0, shortlist)
        verified = score_teacher(reference, official_model, row, shortlist_actions)
        local = stable_order(verified)[:ELITE_COUNT]
        return shortlist.index_select(0, local)

    for _ in range(warmup):
        for arm, k in order:
            if arm == "teacher":
                time_native(teacher_run)
            else:
                time_native(lambda k=k: hybrid_run(k))
    measurements: dict[str, list[dict[str, float]]] = {"teacher300_ms": [], "hybrid60_ms": [], "hybrid120_ms": []}
    for _ in range(repeats):
        for arm, k in order:
            elapsed, _ = time_native(teacher_run if arm == "teacher" else lambda k=k: hybrid_run(k))
            key = "teacher300_ms" if arm == "teacher" else f"hybrid{k}_ms"
            measurements[key].append(elapsed)
    output: dict[str, Any] = {"warmup": warmup, "repeats": repeats, "order": [f"{arm}{k if arm == 'hybrid' else 300}" for arm, k in order], "context_id": row["context_id"], "block": int(block), "boundary": stage["timing"]["boundary"], "shadow_teacher_excluded": True}
    for key, values in measurements.items():
        wall_values = [float(item["wall_ms"]) for item in values]
        cuda_values = [float(item["cuda_ms"]) for item in values]
        output[key] = {
            "wall_samples_ms": wall_values,
            "cuda_samples_ms": cuda_values,
            "median_ms": float(statistics.median(wall_values)),
            "cuda_median_ms": float(statistics.median(cuda_values)),
        }
    teacher_ms = output["teacher300_ms"]["median_ms"]
    for k in SHORTLIST_SIZES:
        hybrid_ms = output[f"hybrid{k}_ms"]["median_ms"]
        output[f"hybrid{k}_latency_reduction"] = float(1.0 - hybrid_ms / max(teacher_ms, 1e-12))
    return output


def timing_for_all_blocks(reference: Any, official_model: Any, student: Any, rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, Any]:
    per_block = [timing_for_block(reference, official_model, student, row, block, freeze) for row in rows for block in range(len(FRESH_SEEDS))]
    if len(per_block) != 48:
        raise ValueError("Stage A timing must cover all 48 nested blocks")
    output: dict[str, Any] = {
        "blocks": len(per_block),
        "warmup": per_block[0]["warmup"],
        "repeats": per_block[0]["repeats"],
        "boundary": per_block[0]["boundary"],
        "interleaved_arm_order": True,
        "shadow_teacher_excluded": True,
        "per_block": [],
    }
    for key in ("teacher300_ms", "hybrid60_ms", "hybrid120_ms"):
        medians = [float(item[key]["median_ms"]) for item in per_block]
        cuda_medians = [float(item[key]["cuda_median_ms"]) for item in per_block]
        output[key] = {"block_median_ms_median": float(statistics.median(medians)), "block_cuda_median_ms_median": float(statistics.median(cuda_medians))}
    for k in SHORTLIST_SIZES:
        reductions = [float(item[f"hybrid{k}_latency_reduction"]) for item in per_block]
        output[f"hybrid{k}_latency_reduction"] = float(statistics.median(reductions))
        output[f"hybrid{k}_latency_reduction_minimum"] = min(reductions)
    for item in per_block:
        output["per_block"].append({
            "context_id": item["context_id"],
            "block": item["block"],
            "teacher300_median_ms": item["teacher300_ms"]["median_ms"],
            "hybrid60_median_ms": item["hybrid60_ms"]["median_ms"],
            "hybrid120_median_ms": item["hybrid120_ms"]["median_ms"],
            "hybrid60_latency_reduction": item["hybrid60_latency_reduction"],
            "hybrid120_latency_reduction": item["hybrid120_latency_reduction"],
        })
    return output


def run_stage_a(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any, contract: Any) -> dict[str, Any]:
    require_compute_node()
    import torch

    dataset = (args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve()
    manifest_path = args.manifest_512.resolve()
    prepared_path = args.prepared_rows_512.resolve()
    for path in (dataset, manifest_path, prepared_path, args.stablewm_home.resolve() / "pusht" / "lewm_object.ckpt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = load_json(manifest_path)
    if len(manifest.get("splits", {}).get("train", [])) != TRAIN_CONTEXTS:
        raise ValueError("Phase2 manifest must contain 512 train contexts")
    control_rows = torch.load(prepared_path, map_location="cpu", weights_only=False)
    control_meta = validate_training_rows(control_rows, reference)
    official_model = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    balanced_rows: Sequence[Mapping[str, Any]] | None = None
    balanced_source = "not_needed_exact_checkpoint"
    if args.balanced_base_checkpoint is None or not args.balanced_base_checkpoint.is_file():
        balanced_path = args.prepared_balanced_rows.resolve() if args.prepared_balanced_rows else None
        if balanced_path is not None and balanced_path.is_file():
            balanced_rows = torch.load(balanced_path, map_location="cpu", weights_only=False)
            balanced_source = "phase5_prepared_balanced_rows"
        else:
            balanced_rows, _ = reference.temporal.build_balanced_train_rows(official_model, dataset, manifest, control_rows)
            balanced_source = "reconstructed_on_compute_node_from_phase2_prepared_rows"
            torch.save(balanced_rows, args.output.resolve() / "prepared_balanced_rows_reconstructed.pt")
        balanced_meta = validate_balanced_rows(balanced_rows)
        balanced_meta["source"] = balanced_source
    else:
        balanced_meta = {"source": balanced_source, "train_contexts": None}
    torch.manual_seed(RECONSTRUCTION_SEED)
    template = reference.base.make_student("baseline")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    torch.cuda.empty_cache()
    student, checkpoint_meta, _ = load_or_reconstruct_checkpoint(args, freeze, reference, official_model, balanced_rows, initial_state)
    fresh_ids, selection = select_fresh_episodes(dataset, manifest)
    fresh_rows, fresh_meta = build_fresh_rows_compat(args, reference, official_model, dataset, fresh_ids, freeze)
    if len(fresh_rows) != 24:
        raise ValueError("fresh Stage A row count must be 24")
    all_results: dict[str, Any] = {}
    timing: dict[str, Any] | None = None
    for k in SHORTLIST_SIZES:
        blocks: list[dict[str, Any]] = []
        for row in fresh_rows:
            for block, seed in enumerate(FRESH_SEEDS):
                actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
                full_teacher_cost = score_teacher(reference, official_model, row, actions)
                item = hybrid_block(reference, official_model, student, row, block, k, full_teacher_cost)
                item.update({"context_id": row["context_id"], "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}"})
                blocks.append(item)
        grouped = group_metrics(blocks)
        overall = grouped["overall"]
        strata_pass = all(float(grouped["strata"][name]["recall_median"]) >= 0.95 for name in FRESH_ANCHORS)
        all_results[str(k)] = {"shortlist_size": k, "per_block": blocks, **grouped, "stratum_gate": strata_pass}
    timing = timing_for_all_blocks(reference, official_model, student, fresh_rows, freeze)
    gate_cfg = freeze["stage_a"]["gate"]
    for k in SHORTLIST_SIZES:
        result = all_results[str(k)]
        overall = result["overall"]
        latency_reduction = float(timing[f"hybrid{k}_latency_reduction"])
        conditions = {
            "overall_median_recall": overall["recall_median"] >= float(gate_cfg["overall_median_recall_min"]),
            "minimum_block_recall": overall["recall_minimum"] >= float(gate_cfg["minimum_block_recall_min"]),
            "each_stratum_median_recall": result["stratum_gate"],
            "hybrid_latency_reduction": latency_reduction >= float(gate_cfg["hybrid_latency_reduction_min"]),
            "finite": bool(overall["finite"] and all(result["strata"][name]["finite"] for name in FRESH_ANCHORS)),
            "interface_correct": True,
        }
        result["gate"] = {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions, "hybrid_latency_reduction": latency_reduction}
    passed = [k for k in SHORTLIST_SIZES if all_results[str(k)]["gate"]["status"] == "PASS"]
    selected = min(passed) if passed else None
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "STAGE_A_COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "source": {"lewm_root": str(args.lewm_root.resolve()), "stablewm_home": str(args.stablewm_home.resolve()), "dataset": str(dataset), "manifest_512": str(manifest_path), "prepared_rows_512": str(prepared_path), "balanced_rows_source": balanced_source},
        "interface_contract": validate_interface(contract),
        "checkpoint": checkpoint_meta,
        "training_bank": {"control": control_meta, "balanced": balanced_meta, "rows_saved_on_compute_node": balanced_source != "not_needed_exact_checkpoint"},
        "fresh_selection": selection,
        "fresh_evaluation": fresh_meta,
        "screening": {"candidate_count": NUM_CANDIDATES, "elite_count": ELITE_COUNT, "shortlist_sizes": list(SHORTLIST_SIZES), "arms": all_results, "selected_shortlist_size": selected, "selection_rule": "smallest K passing every frozen Stage A gate"},
        "timing": timing,
        "stage_a_gate": {"status": "PASS" if selected is not None else "FAIL", "selected_shortlist_size": selected, "tested_shortlist_sizes": list(SHORTLIST_SIZES)},
        "stage_b": {"status": "PENDING_STAGE_A_PASS" if selected is not None else "NOT_RUN_BY_GATE", "official_cem": "PENDING_STAGE_A_PASS" if selected is not None else "NOT_RUN_BY_GATE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Stage A fixed-candidate teacher-screening evidence only; reconstructed checkpoint is not the original saved Phase 6 checkpoint; no adaptive CEM or closed-loop claim is produced.",
    }
    write_json(args.output.resolve() / "teacher_screening_stage_a_summary.json", summary)
    return summary


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    freeze = validate_freeze(load_json(args.freeze.resolve()))
    validate_protocol(args.protocol.resolve())
    reference = load_reference_modules()
    import torch

    contract = None
    interface_status = "not_loaded"
    if args.interface_probe.is_file():
        contract = reference.base.load_interface_contract(args.interface_probe.resolve())
        interface_meta = validate_interface(contract)
        interface_status = "PASS"
    else:
        interface_meta = None
    value = {
        "schema": SCHEMA,
        "status": "PASS" if interface_status == "PASS" else "READY_WITHOUT_INTERFACE_PROBE",
        "model_work_started": False,
        "freeze_schema": freeze["schema"],
        "protocol": str(args.protocol.resolve()),
        "reference_runner": str((REFERENCE_DIR / "run_lewm_state_action_prefix_gru.py").resolve()),
        "reference_import": "PASS",
        "torch_import": "PASS",
        "interface": interface_meta,
        "stage_a": {"selection_slice": list(FRESH_SLICE), "shortlist_sizes": list(SHORTLIST_SIZES), "blocks": 48, "candidates_per_block": NUM_CANDIDATES, "action_prefix_seeds": list(FRESH_SEEDS)},
        "checkpoint_policy": freeze["student"]["checkpoint_policy"],
        "pbs_compute_only": True,
    }
    write_json(args.output.resolve() / "preflight_status.json", value)
    return value


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "selection_slice": "valid[544:552]", "fresh_blocks": 48, "candidates_per_block": 300, "shortlist_sizes": list(SHORTLIST_SIZES), "terminal_snapshot": "step_3000", "official_cem": "NOT_RUN_BY_GATE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    freeze = validate_freeze(load_json(args.freeze.resolve()))
    if args.mode == "preflight":
        value = preflight(args)
        print(json.dumps(value, ensure_ascii=False))
        return 0 if value["status"] in ("PASS", "READY_WITHOUT_INTERFACE_PROBE") else 1
    validate_protocol(args.protocol.resolve())
    reference = load_reference_modules()
    if not args.interface_probe.is_file():
        raise FileNotFoundError(args.interface_probe)
    contract = reference.base.load_interface_contract(args.interface_probe.resolve())
    validate_interface(contract)
    summary = run_stage_a(args, freeze, reference, contract)
    print(json.dumps({"status": summary["status"], "stage_a_gate": summary["stage_a_gate"], "output": str((args.output / "teacher_screening_stage_a_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
