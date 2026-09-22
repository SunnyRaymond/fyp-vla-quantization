#!/usr/bin/env python3
"""Paired anchor-aligned training-bank experiment for LeWM balanced_base."""

from __future__ import annotations

import argparse
import copy
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
SCHEMA = "lewm.anchor-aligned-bank.runner"
TRAIN_CONTEXTS = 512
TRAIN_CANDIDATES = 64
HORIZON = 5
ACTION_DIM = 10
LATENT_DIM = 192
FRESH_SEEDS = (20300967, 20300968)
FRESH_ANCHORS = ("early", "middle", "late")
FRESH_SLICE = (552, 560)
SELECTION_SEED = 20300903
NUM_CANDIDATES = 300
ELITE_COUNT = 30
SHORTLISTS = (60, 120)
INIT_SEED = 20300901
TRAIN_SEED = 20300902
SCHEDULE_SEED = 20300904
SLATE_SEED = 20300905
TIMING_SEED = 20300969


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--temporal-freeze", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--prepared-rows-512", type=Path, required=True)
    parser.add_argument("--prepared-balanced-rows", type=Path, required=True)
    parser.add_argument("--balanced-base-checkpoint", type=Path, required=True)
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


def load_reference() -> Any:
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    return importlib.import_module("run_lewm_state_action_prefix_gru")


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required for model/HDF5/training work")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "lewm.anchor-aligned-bank.freeze" or int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unexpected anchor-aligned-bank freeze")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("freeze must be frozen_before_results")
    training = freeze.get("training", {})
    expected_training = {
        "contexts": TRAIN_CONTEXTS,
        "batch_contexts": 8,
        "candidates": TRAIN_CANDIDATES,
        "updates": 3000,
        "initialization_seed": INIT_SEED,
        "training_seed": TRAIN_SEED,
        "context_schedule_seed": SCHEDULE_SEED,
        "candidate_slate_seed": SLATE_SEED,
    }
    for key, expected in expected_training.items():
        if int(training.get(key, -1)) != expected:
            raise ValueError(f"training contract drifted: {key}")
    if int(training.get("hidden_dim", -1)) != 256 or training.get("snapshot") != "step_3000":
        raise ValueError("student architecture/snapshot drifted")
    if training.get("anchor_counts") != {"early": 171, "middle": 171, "late": 170}:
        raise ValueError("ordinal anchor counts drifted")
    if not training.get("save_treatment_checkpoint_and_rows_on_compute"):
        raise ValueError("treatment compute artifacts must be saved on compute")
    evaluation = freeze.get("evaluation", {})
    for key, expected in {
        "selection_seed": SELECTION_SEED,
        "episodes": 8,
        "candidates": NUM_CANDIDATES,
        "elite_count": ELITE_COUNT,
        "primary_k": 120,
        "descriptive_k": 60,
    }.items():
        if int(evaluation.get(key, -1)) != expected:
            raise ValueError(f"evaluation contract drifted: {key}")
    if evaluation.get("selection_slice") != list(FRESH_SLICE) or int(evaluation.get("excluded_prefix", -1)) != FRESH_SLICE[0]:
        raise ValueError("fresh valid[552:560] contract drifted")
    if list(evaluation.get("action_prefix_seeds", [])) != list(FRESH_SEEDS) or int(evaluation.get("blocks", -1)) != 48:
        raise ValueError("fresh action-prefix/block contract drifted")
    if evaluation.get("same_candidate_banks_for_both_arms") is not True:
        raise ValueError("paired fresh-bank contract drifted")
    timing = freeze.get("timing", {})
    if list(timing.get("arms", [])) != ["teacher300", "teacher120", "teacher60", "student300_control", "student300_treatment", "hybrid120_control", "hybrid120_treatment"]:
        raise ValueError("timing arms drifted")
    for key, expected in {"blocks": 48, "warmup": 3, "repeats": 10, "order_seed": TIMING_SEED}.items():
        if int(timing.get(key, -1)) != expected:
            raise ValueError(f"timing contract drifted: {key}")
    if timing.get("same_gpu_cached_state_goal") is not True or timing.get("shadow_teacher_excluded_from_hybrid_timing") is not True:
        raise ValueError("timing cache/shadow contract drifted")
    scope = freeze.get("scope", {})
    if scope.get("official_cem") != "NOT_RUN_BY_SCOPE" or scope.get("closed_loop") != "NOT_RUN_BY_SCOPE":
        raise ValueError("scope boundary drifted")
    return dict(freeze)


def validate_protocol(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = ("valid[552:560]", "20300967/20300968", "paired", "closed-loop")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol missing frozen terms: {missing}")


def validate_temporal_freeze(path: Path) -> dict[str, Any]:
    freeze = load_json(path.resolve())
    if freeze.get("schema") != "lewm-recurrent-student.temporal-balanced-train-freeze" or freeze.get("status") != "frozen":
        raise ValueError("temporal-balanced candidate freeze is not frozen")
    evaluation = freeze.get("evaluation", {})
    if list(evaluation.get("official_action_low", [])) != [-1.0, -1.0] or list(evaluation.get("official_action_high", [])) != [1.0, 1.0]:
        raise ValueError("official action bounds drifted")
    if not math.isclose(float(evaluation.get("gaussian_std", float("nan"))), 0.22360679774997896, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("official Gaussian std drifted")
    return freeze


def validate_interface(reference: Any, probe: Path) -> dict[str, Any]:
    contract = reference.base.load_interface_contract(probe.resolve())
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
            raise ValueError(f"interface drifted: {key}")
    return {key: list(value) if isinstance(value, tuple) else value for key, value in expected.items()}


def anchor_spec(length: int) -> dict[str, int]:
    late = int(length) - HORIZON * 5 - 1
    if late < 0:
        raise ValueError(f"episode length {length} cannot hold the frozen action span")
    return {"early": 0, "middle": late // 2, "late": late}


def select_fresh(dataset: Path, manifest: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    with h5py.File(dataset, "r") as handle:
        lengths = [int(x) for x in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON * 5 + 1]
    random.Random(SELECTION_SEED).shuffle(valid)
    heldout = [int(row["episode_id"]) for row in manifest["splits"]["heldout"]]
    train = [int(row["episode_id"]) for row in manifest["splits"]["train"]]
    prefix = heldout + train
    if valid[: len(prefix)] != prefix:
        raise ValueError("Phase2 selection prefix cannot be reproduced")
    fresh = valid[FRESH_SLICE[0] : FRESH_SLICE[1]]
    if len(fresh) != 8 or set(fresh) & set(valid[: FRESH_SLICE[0]]):
        raise ValueError("fresh valid[552:560] overlaps excluded prefix")
    return fresh, {
        "selection_seed": SELECTION_SEED,
        "valid_count": len(valid),
        "fresh_episode_ids": fresh,
        "selection_slice": "valid[552:560]",
        "excluded_valid_prefix": 552,
        "excluded_prior_slices": ["valid[520:528]", "valid[528:536]", "valid[536:544]", "valid[544:552]"],
        "result_dependent_selection": False,
    }


def build_anchor_aligned_rows(reference: Any, official_model: Any, dataset: Path, manifest: Mapping[str, Any], control_rows: Sequence[Mapping[str, Any]], temporal_freeze: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recreate the original CPU slate stream and alter only its action center."""
    import torch

    base = reference.base
    low = torch.tensor(temporal_freeze["evaluation"]["official_action_low"], dtype=torch.float32).reshape(1, 1, 2)
    high = torch.tensor(temporal_freeze["evaluation"]["official_action_high"], dtype=torch.float32).reshape(1, 1, 2)
    gaussian_std = float(temporal_freeze["evaluation"]["gaussian_std"])
    train_manifest = list(manifest["splits"]["train"])
    controls = [row for row in control_rows if row.get("split") == "train"]
    if len(train_manifest) != TRAIN_CONTEXTS or len(controls) != TRAIN_CONTEXTS:
        raise ValueError("aligned-bank construction requires 512 train rows")
    generator = torch.Generator(device="cpu").manual_seed(SLATE_SEED)
    rows: list[dict[str, Any]] = []
    counts = {name: 0 for name in FRESH_ANCHORS}
    replay_equal = True
    with base.HDF5EpisodeSliceReader(dataset) as reader:
        for ordinal, (manifest_row, control) in enumerate(zip(train_manifest, controls)):
            if int(control.get("ordinal", ordinal)) != ordinal or int(control["episode_id"]) != int(manifest_row["episode_id"]):
                raise ValueError(f"control row alignment drifted at ordinal {ordinal}")
            episode = reader.episode_slice(int(manifest_row["episode_id"]))
            anchors = anchor_spec(int(episode["length"]))
            stratum = FRESH_ANCHORS[ordinal % 3]
            anchor = anchors[stratum]
            counts[stratum] += 1
            control_raw = reader.read_manifest_row(manifest_row)
            logged_control = torch.from_numpy(base.pack_raw_actions(control_raw["future_actions_raw"]))
            base._slate_actions(logged_control, generator, low, high, gaussian_std, 1)
            base._slate_actions(logged_control, generator, low, high, gaussian_std, 1)
            state_before_count64 = generator.get_state().clone()
            replay_control = base._slate_actions(logged_control, generator, low, high, gaussian_std, TRAIN_CANDIDATES)
            torch.randn((HORIZON, ACTION_DIM), generator=generator)
            cached_control = torch.as_tensor(control["future_actions"], dtype=torch.float32).cpu()
            if tuple(cached_control.shape) != (TRAIN_CANDIDATES, HORIZON, ACTION_DIM):
                raise ValueError(f"cached control bank shape drifted at ordinal {ordinal}")
            equal = bool(torch.equal(replay_control.cpu(), cached_control))
            replay_equal = replay_equal and equal
            if not equal:
                raise ValueError(f"original control bank RNG replay mismatch at ordinal {ordinal}")
            treatment_generator = torch.Generator(device="cpu")
            treatment_generator.set_state(state_before_count64)
            treatment_manifest = {"episode_id": int(manifest_row["episode_id"]), "history_steps": [anchor], "history_action_starts": [anchor], "future_action_start": anchor, "goal_step_offset": anchor + HORIZON * 5}
            raw = reader.read_manifest_row(treatment_manifest)
            logged_treatment = torch.from_numpy(base.pack_raw_actions(raw["future_actions_raw"]))
            treatment_actions = base._slate_actions(logged_treatment, treatment_generator, low, high, gaussian_std, TRAIN_CANDIDATES)
            current = base._normalise_pixels(raw["pixels"][:1]).unsqueeze(1).to("cuda")
            goal_pixels = base._normalise_pixels(episode["pixels"][anchor + HORIZON * 5 : anchor + HORIZON * 5 + 1]).unsqueeze(1).to("cuda")
            action_history = logged_treatment[:1].unsqueeze(0).to("cuda")
            with torch.no_grad():
                latent = official_model.encode({"pixels": current, "action": action_history})["emb"]
                goal_emb = official_model.encode({"pixels": goal_pixels})["emb"]
                actions_cuda = treatment_actions.to("cuda")
                targets = base.official_teacher_targets(official_model, latent.expand(TRAIN_CANDIDATES, -1, -1), actions_cuda)
                objective = base._official_objective(official_model, {"latent_history": latent, "goal_emb": goal_emb}, targets)
            row = {
                "split": "train",
                # Keep the original context identity so the treatment differs only
                # in its action-bank center, not in episode/ordinal bookkeeping.
                "context_id": control["context_id"],
                "ordinal": ordinal,
                "episode_id": int(manifest_row["episode_id"]),
                "anchor": int(anchor),
                "stratum": stratum,
                "latent_history": latent.detach().cpu(),
                "action_history": action_history.detach().cpu(),
                "future_actions": treatment_actions.detach().cpu(),
                "teacher_targets": targets.detach().cpu(),
                "teacher_objective": objective.detach().cpu(),
                "goal_emb": goal_emb.detach().cpu(),
                "bank_variant": "anchor_aligned",
                "control_context_id": control["context_id"],
            }
            if "rank_shuffle_indices" in control:
                row["rank_shuffle_indices"] = torch.as_tensor(control["rank_shuffle_indices"]).clone()
            rows.append(row)
    if counts != {"early": 171, "middle": 171, "late": 170} or len(rows) != TRAIN_CONTEXTS:
        raise ValueError(f"anchor counts drifted: {counts}")
    return rows, {
        "train_contexts": len(rows),
        "anchor_counts": counts,
        "paired_noise_exact": replay_equal,
        "noise_schedule": "count1,count1,count64,randn(HORIZON,ACTION_DIM) per ordinal",
        "center_change_only": True,
        "clip_noise_reconstruction": False,
        "teacher_targets_recomputed_at_current_anchor": True,
        "rows_generated_on_compute_node": True,
        "rows_returned": False,
    }


def load_reconstructed_checkpoint(reference: Any, path: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    raw = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    provenance = raw.get("provenance", {}) if isinstance(raw, Mapping) else {}
    if not isinstance(provenance, Mapping) or provenance.get("reconstructed") is not True:
        raise ValueError("control checkpoint must retain reconstructed provenance")
    state = raw["state_dict"] if isinstance(raw, Mapping) and "state_dict" in raw else raw
    if not isinstance(state, Mapping):
        raise ValueError("control checkpoint lacks state_dict")
    student = reference.base.make_student("baseline")
    student.load_state_dict({key: value.detach().clone() for key, value in state.items()}, strict=True)
    return student.to("cuda").eval(), {"path": str(path.resolve()), "source": "reconstructed", "provenance": dict(provenance), "original_checkpoint_available": False}


def build_fresh_rows(reference: Any, official_model: Any, dataset: Path, episode_ids: Sequence[int], temporal_freeze: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    import copy as copy_module

    adapter_freeze = {"evaluation": copy_module.deepcopy(temporal_freeze["evaluation"])}
    old_seeds = tuple(reference.ema.FRESH_SEEDS)
    old_anchors = tuple(reference.ema.FRESH_ANCHORS)
    reference.ema.FRESH_SEEDS = FRESH_SEEDS
    reference.ema.FRESH_ANCHORS = FRESH_ANCHORS
    try:
        rows, metadata = reference.ema.build_fresh_rows(official_model, dataset, episode_ids, adapter_freeze)
    finally:
        reference.ema.FRESH_SEEDS = old_seeds
        reference.ema.FRESH_ANCHORS = old_anchors
    if len(rows) != 24 or metadata.get("blocks") != 48:
        raise ValueError("fresh rows/blocks drifted")
    details = metadata.get("contexts_detail", [])
    if len(details) != 24 or any(list(item.get("candidate_seeds", [])) != list(FRESH_SEEDS) or int(item.get("candidate_count_per_block", -1)) != NUM_CANDIDATES for item in details):
        raise ValueError("fresh candidate metadata drifted")
    return rows, {**metadata, "candidate_contract_source": "temporal-balanced-freeze", "seeds": list(FRESH_SEEDS)}


def _student_costs(reference: Any, official_model: Any, student: Any, row: Mapping[str, Any], actions: Any) -> tuple[Any, Any]:
    import torch

    context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
    context = context.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        prediction = student(context, actions)
        costs = reference.base._official_objective(official_model, row, prediction)
    return prediction, costs


def _block_metrics(reference: Any, official_model: Any, student: Any, row: Mapping[str, Any], block: int) -> dict[str, Any]:
    import torch

    actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
    target = reference.base._tensor(row["teacher_targets"][block], dtype=torch.float32).to("cuda")
    prediction, student_cost = _student_costs(reference, official_model, student, row, actions)
    teacher_cost = reference.base._tensor(row["teacher_objective"][block], dtype=torch.float32).to("cuda")
    teacher_order = torch.argsort(teacher_cost, stable=True)
    student_order = torch.argsort(student_cost, stable=True)
    teacher_top = teacher_order[:ELITE_COUNT]
    teacher_mean = teacher_cost.index_select(0, teacher_top).mean()
    std = teacher_cost.std(unbiased=False).clamp_min(1e-6)
    output: dict[str, Any] = {
        "spearman": reference.base._spearman(teacher_cost, student_cost),
        "top30_overlap": reference.base._topk_overlap(teacher_cost, student_cost, ELITE_COUNT),
        "relative_latent_mse": float(((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).detach().cpu()),
        "teacher_cost_std_population": float(std.detach().cpu()),
        "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(student_cost).all().item() and torch.isfinite(teacher_cost).all().item()),
    }
    for k in SHORTLISTS:
        shortlist = student_order[:k]
        captured = torch.isin(teacher_top, shortlist).sum()
        verified_order = torch.argsort(teacher_cost.index_select(0, shortlist), stable=True)[:ELITE_COUNT]
        hybrid = shortlist.index_select(0, verified_order)
        raw_regret = teacher_cost.index_select(0, hybrid).mean() - teacher_mean
        output[f"recall_at_{k}"] = float(captured.detach().cpu()) / ELITE_COUNT
        output[f"full_elite_containment_at_{k}"] = bool(int(captured.detach().cpu()) == ELITE_COUNT)
        output[f"elite_mean_cost_regret_at_{k}"] = float(raw_regret.detach().cpu())
        output[f"standardized_elite_mean_cost_regret_at_{k}"] = float((raw_regret / std).detach().cpu())
    return output


def aggregate(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("cannot aggregate empty blocks")
    keys = ("spearman", "top30_overlap", "relative_latent_mse", "recall_at_60", "recall_at_120", "elite_mean_cost_regret_at_60", "elite_mean_cost_regret_at_120", "standardized_elite_mean_cost_regret_at_60", "standardized_elite_mean_cost_regret_at_120")
    out = {"blocks": len(blocks), "finite": all(bool(item["finite"]) for item in blocks)}
    for key in keys:
        values = [float(item[key]) for item in blocks]
        out[f"{key}_median"] = float(statistics.median(values))
        out[f"{key}_minimum"] = min(values)
        out[f"{key}_maximum"] = max(values)
    for k in SHORTLISTS:
        out[f"full_elite_containment_at_{k}_rate"] = float(sum(bool(item[f"full_elite_containment_at_{k}"]) for item in blocks) / len(blocks))
    return out


def group_evaluation(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strata = {name: aggregate([item for item in blocks if item["stratum"] == name]) for name in FRESH_ANCHORS}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in blocks:
        grouped[str(item["episode_id"])].append(item)
    episodes = {episode: aggregate(group) for episode, group in sorted(grouped.items(), key=lambda item: int(item[0]))}
    return {"overall": aggregate(blocks), "strata": strata, "episodes": episodes, "worst_blocks": sorted(blocks, key=lambda item: (float(item["recall_at_120"]), int(item["episode_id"])))[:5]}


def evaluate_arm(reference: Any, official_model: Any, student: Any, rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for row in rows:
        for block, seed in enumerate(FRESH_SEEDS):
            item = _block_metrics(reference, official_model, student, row, block)
            item.update({"label": label, "context_id": row["context_id"], "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}", "teacher_full_score_reused": True})
            blocks.append(item)
    return {"label": label, "per_block": blocks, **group_evaluation(blocks)}


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


def timing_for_block(reference: Any, official_model: Any, students: Mapping[str, Any], row: Mapping[str, Any], block: int, freeze: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
    latent = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
    goal = reference.base._tensor(row["goal_emb"], dtype=torch.float32).to("cuda")
    row_gpu = dict(row)
    row_gpu["latent_history"] = latent
    row_gpu["goal_emb"] = goal
    context = latent.expand(NUM_CANDIDATES, -1, -1)
    def teacher(n: int) -> Any:
        small_actions, small_context = actions[:n], context[:n]
        with torch.no_grad():
            targets = reference.base.official_teacher_targets(official_model, small_context, small_actions)
            costs = reference.base._official_objective(official_model, row_gpu, targets)
        return torch.argsort(costs, stable=True)[:ELITE_COUNT]
    def student_only(student: Any) -> Any:
        with torch.no_grad():
            prediction = student(context, actions)
            costs = reference.base._official_objective(official_model, row_gpu, prediction)
        return torch.argsort(costs, stable=True)[:ELITE_COUNT]
    def hybrid(student: Any) -> Any:
        with torch.no_grad():
            prediction = student(context, actions)
            student_cost = reference.base._official_objective(official_model, row_gpu, prediction)
            shortlist = torch.argsort(student_cost, stable=True)[:120]
            small_actions = actions.index_select(0, shortlist)
            small_context = context.index_select(0, shortlist)
            targets = reference.base.official_teacher_targets(official_model, small_context, small_actions)
            teacher_cost = reference.base._official_objective(official_model, row_gpu, targets)
        return shortlist.index_select(0, torch.argsort(teacher_cost, stable=True)[:ELITE_COUNT])
    timing_cfg = freeze["timing"]
    order = ["teacher300", "teacher120", "teacher60", "student300_control", "student300_treatment", "hybrid120_control", "hybrid120_treatment"]
    random.Random(TIMING_SEED).shuffle(order)
    functions = {
        "teacher300": lambda: teacher(300),
        "teacher120": lambda: teacher(120),
        "teacher60": lambda: teacher(60),
        "student300_control": lambda: student_only(students["control"]),
        "student300_treatment": lambda: student_only(students["treatment"]),
        "hybrid120_control": lambda: hybrid(students["control"]),
        "hybrid120_treatment": lambda: hybrid(students["treatment"]),
    }
    for _ in range(int(timing_cfg["warmup"])):
        for name in order:
            time_native(functions[name])
    samples = {name: [] for name in order}
    for _ in range(int(timing_cfg["repeats"])):
        for name in order:
            elapsed, _ = time_native(functions[name])
            samples[name].append(elapsed)
    return {"context_id": row["context_id"], "block": int(block), "order": order, "warmup": int(timing_cfg["warmup"]), "repeats": int(timing_cfg["repeats"]), "samples": samples}


def summarize_timing(reference: Any, official_model: Any, students: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, Any]:
    blocks = [timing_for_block(reference, official_model, students, row, block, freeze) for row in rows for block in range(2)]
    if len(blocks) != 48:
        raise ValueError("timing must cover 48 blocks")
    output: dict[str, Any] = {"blocks": 48, "warmup": 3, "repeats": 10, "order_seed": TIMING_SEED, "per_block": []}
    names = ["teacher300", "teacher120", "teacher60", "student300_control", "student300_treatment", "hybrid120_control", "hybrid120_treatment"]
    for name in names:
        wall = [float(statistics.median(item["samples"][name][i]["wall_ms"] for i in range(len(item["samples"][name])))) for item in blocks]
        cuda = [float(statistics.median(item["samples"][name][i]["cuda_ms"] for i in range(len(item["samples"][name])))) for item in blocks]
        output[name] = {"block_median_wall_ms_median": float(statistics.median(wall)), "block_median_cuda_ms_median": float(statistics.median(cuda))}
    for item in blocks:
        output["per_block"].append({"context_id": item["context_id"], "block": item["block"], **{name: float(statistics.median(sample["wall_ms"] for sample in item["samples"][name])) for name in names}})
    return output


def paired_gate(control: Mapping[str, Any], treatment: Mapping[str, Any], *, interface_ok: bool, paired_noise_ok: bool) -> dict[str, Any]:
    deltas = []
    for episode in sorted(control["episodes"], key=int):
        c = control["episodes"][episode]
        t = treatment["episodes"][episode]
        deltas.append({"episode_id": int(episode), "recall120_delta_treatment_minus_control": float(t["recall_at_120_median"] - c["recall_at_120_median"]), "standardized_regret120_delta_treatment_minus_control": float(t["standardized_elite_mean_cost_regret_at_120_median"] - c["standardized_elite_mean_cost_regret_at_120_median"]), "strict_improvement": bool(t["recall_at_120_median"] > c["recall_at_120_median"])})
    recall_deltas = [item["recall120_delta_treatment_minus_control"] for item in deltas]
    regret_deltas = [item["standardized_regret120_delta_treatment_minus_control"] for item in deltas]
    control_bad = sum(float(item["recall_at_120"]) < 0.8 for item in control["per_block"])
    treatment_bad = sum(float(item["recall_at_120"]) < 0.8 for item in treatment["per_block"])
    control_worst = min(float(item["recall_at_120"]) for item in control["per_block"])
    treatment_worst = min(float(item["recall_at_120"]) for item in treatment["per_block"])
    conditions = {
        "median_episode_recall120_delta_strictly_positive": statistics.median(recall_deltas) > 0.0,
        "strictly_improved_episodes_min": sum(item["strict_improvement"] for item in deltas) >= 5,
        "treatment_bad_block_count_strictly_less_than_control": treatment_bad < control_bad,
        "treatment_worst_block_recall_not_lower": treatment_worst >= control_worst,
        "median_episode_paired_standardized_regret_delta_max": statistics.median(regret_deltas) <= 0.0,
        "finite_interface_and_paired_noise_required": bool(control["overall"]["finite"] and treatment["overall"]["finite"] and interface_ok and paired_noise_ok),
    }
    return {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions, "per_episode": deltas, "median_recall120_delta": float(statistics.median(recall_deltas)), "median_standardized_regret120_delta": float(statistics.median(regret_deltas)), "control_bad_blocks": control_bad, "treatment_bad_blocks": treatment_bad, "control_worst_block_recall120": control_worst, "treatment_worst_block_recall120": treatment_worst, "paired_unit": "episode after median over six nested blocks"}


def absolute_gate(treatment: Mapping[str, Any]) -> dict[str, Any]:
    strata = treatment["strata"]
    conditions = {"median_recall120_min": treatment["overall"]["recall_at_120_median"] >= 0.95, "minimum_block_recall120_min": treatment["overall"]["recall_at_120_minimum"] >= 0.80, "each_stratum_median_recall120_min": all(strata[name]["recall_at_120_median"] >= 0.95 for name in FRESH_ANCHORS), "finite": treatment["overall"]["finite"]}
    return {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions, "not_a_predictor_replacement_gate": True}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any, interface: Mapping[str, Any], temporal_freeze: Mapping[str, Any]) -> dict[str, Any]:
    require_compute_node()
    import torch

    dataset = (args.dataset or args.stablewm_home / "pusht" / "pusht_expert_train.h5").resolve()
    manifest = load_json(args.manifest_512.resolve())
    control_rows = torch.load(args.prepared_balanced_rows.resolve(), map_location="cpu", weights_only=False)
    train_control = [row for row in control_rows if row.get("split") == "train"]
    if len(train_control) != TRAIN_CONTEXTS:
        raise ValueError("reused balanced control rows are not 512 contexts")
    official_model = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    treatment_rows, bank_meta = build_anchor_aligned_rows(reference, official_model, dataset, manifest, control_rows, temporal_freeze)
    torch.save(treatment_rows, args.output.resolve() / "anchor_aligned_train_rows.pt")
    control_student, control_meta = load_reconstructed_checkpoint(reference, args.balanced_base_checkpoint)
    torch.manual_seed(INIT_SEED)
    template = reference.base.make_student("baseline")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    training = reference.train_arm(
        treatment_rows,
        initial_state,
        official_model,
        {"training_seed": TRAIN_SEED, "context_schedule_seed": SCHEDULE_SEED, "prefix_initialization_seed": INIT_SEED},
        "balanced_base",
    )
    treatment_state = training["snapshots"]["step_3000"]
    treatment_checkpoint = args.output.resolve() / "anchor_aligned_bank_step3000.pt"
    torch.save({"state_dict": treatment_state, "provenance": {"source": "anchor_aligned_bank_treatment", "reconstructed": False, "training_seed": TRAIN_SEED, "context_schedule_seed": SCHEDULE_SEED, "initialization_seed": INIT_SEED, "updates": 3000}}, treatment_checkpoint)
    treatment_student = reference.base.make_student("baseline")
    treatment_student.load_state_dict(treatment_state, strict=True)
    treatment_student = treatment_student.to("cuda").eval()
    fresh_ids, selection = select_fresh(dataset, manifest)
    fresh_rows, fresh_meta = build_fresh_rows(reference, official_model, dataset, fresh_ids, temporal_freeze)
    control_eval = evaluate_arm(reference, official_model, control_student, fresh_rows, "balanced_base_control_reused")
    treatment_eval = evaluate_arm(reference, official_model, treatment_student, fresh_rows, "anchor_aligned_bank_treatment")
    timing = summarize_timing(reference, official_model, {"control": control_student, "treatment": treatment_student}, fresh_rows, freeze)
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_contract": interface,
        "source": {"lewm_root": str(args.lewm_root.resolve()), "stablewm_home": str(args.stablewm_home.resolve()), "dataset": str(dataset), "manifest_512": str(args.manifest_512.resolve()), "prepared_balanced_rows": str(args.prepared_balanced_rows.resolve()), "temporal_freeze": str(args.temporal_freeze.resolve())},
        "control": {"checkpoint": control_meta, "rows_source": "reused_job25213164_reconstructed_balanced_rows", "historical_training": True, "concurrent_control_training": False, "evaluation": control_eval},
        "treatment": {"checkpoint": {"path": str(treatment_checkpoint), "source": "anchor_aligned_bank_treatment", "reconstructed": False}, "rows_path": str((args.output.resolve() / "anchor_aligned_train_rows.pt")), "training_contract": {"initialization_seed": INIT_SEED, "training_seed": TRAIN_SEED, "context_schedule_seed": SCHEDULE_SEED, "updates": 3000, "batch_contexts": 8, "candidates": 64}, "training_trace": {key: value for key, value in training.items() if key != "snapshots"}, "evaluation": treatment_eval},
        "paired_noise": bank_meta,
        "fresh_selection": selection,
        "fresh_evaluation": fresh_meta,
        "paired_mechanism_gate": paired_gate(
            control_eval,
            treatment_eval,
            interface_ok=bool(interface.get("semantics_equal") is True),
            paired_noise_ok=bool(bank_meta.get("paired_noise_exact") is True),
        ),
        "separate_absolute_gate": absolute_gate(treatment_eval),
        "timing": timing,
        "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Anchor-aligned action-bank evidence is a predictor-level training-mechanism result; it does not establish predictor replacement, adaptive CEM, hybrid speedup, or closed-loop success.",
    }
    write_json(args.output.resolve() / "anchor_aligned_bank_summary.json", summary)
    return summary


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    freeze = validate_freeze(load_json(args.freeze.resolve()))
    validate_protocol(args.protocol.resolve())
    temporal = validate_temporal_freeze(args.temporal_freeze.resolve())
    reference = load_reference()
    import torch  # noqa: F401

    interface = validate_interface(reference, args.interface_probe.resolve())
    value = {"schema": SCHEMA, "status": "PASS", "model_work_started": False, "reference_import": "PASS", "torch_import": "PASS", "interface": interface, "fresh_selection": "valid[552:560]", "fresh_blocks": 48, "training_contexts": 512, "training_candidates": 64, "paired_noise_contract": freeze["training"]["paired_noise"], "temporal_candidate_contract": {"official_action_low": temporal["evaluation"]["official_action_low"], "official_action_high": temporal["evaluation"]["official_action_high"], "gaussian_std": temporal["evaluation"]["gaussian_std"]}, "checkpoint_policy": "reused reconstructed control checkpoint; treatment trained from seed 20300901", "pbs_compute_only": True}
    write_json(args.output.resolve() / "preflight_status.json", value)
    return value


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "training_contexts": TRAIN_CONTEXTS, "fresh_selection": "valid[552:560]", "fresh_blocks": 48, "primary_k": 120, "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    freeze = validate_freeze(load_json(args.freeze.resolve()))
    if args.mode == "preflight":
        value = preflight(args)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    validate_protocol(args.protocol.resolve())
    temporal = validate_temporal_freeze(args.temporal_freeze.resolve())
    reference = load_reference()
    interface = validate_interface(reference, args.interface_probe.resolve())
    summary = run(args, freeze, reference, interface, temporal)
    print(json.dumps({"status": summary["status"], "paired_mechanism_gate": summary["paired_mechanism_gate"], "separate_absolute_gate": summary["separate_absolute_gate"], "output": str((args.output / "anchor_aligned_bank_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
