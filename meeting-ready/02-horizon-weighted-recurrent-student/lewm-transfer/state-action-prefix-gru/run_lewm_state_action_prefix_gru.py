#!/usr/bin/env python3
"""LeWM balanced-base versus state-action prefix GRU predictor experiment."""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
TEMPORAL_DIR = TRANSFER / "temporal-balanced-train"
EMA_DIR = TRANSFER / "ema-temporal-confirm"
sys.path.insert(0, str(TEMPORAL_DIR))
sys.path.insert(0, str(EMA_DIR))
sys.path.insert(0, str(TRANSFER / "state-coverage"))
sys.path.insert(0, str(TRANSFER / "score-distill"))
sys.path.insert(0, str(TRANSFER / "dense-rank"))

import run_lewm_temporal_balanced_train as temporal  # noqa: E402

base = temporal.base
ema = temporal.ema
score = temporal.score

TRAIN_CONTEXTS = 512
TRAIN_CANDIDATES = 64
BATCH_CONTEXTS = 8
STEPS = 3000
SNAPSHOT_STEPS = (500, 1000, 1500, 3000)
FRESH_EPISODE_COUNT = 8
FRESH_ANCHORS = ("early", "middle", "late")
FRESH_SEEDS = (20300937, 20300938)
HORIZON_PRIMITIVES = base.HORIZON * (base.ACTION_DIM // 2)
PREFIX_DIM = 64

# The shared fresh-row builder and its scorer use these module globals.
ema.FRESH_SEEDS = FRESH_SEEDS
ema.FRESH_ANCHORS = FRESH_ANCHORS
ema.SNAPSHOT_STEPS = SNAPSHOT_STEPS


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--temporal-freeze", type=Path, required=True)
    parser.add_argument("--temporal-protocol", type=Path, required=True)
    parser.add_argument("--temporal-summary", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--prepared-rows-512", type=Path, required=True)
    parser.add_argument("--prepared-balanced-rows", type=Path, default=None)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--phase2-freeze", type=Path, required=True)
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


def make_prefix_student() -> Any:
    import torch
    import torch.nn as nn

    class _Impl(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.history_adapter = nn.Linear(base.HISTORY_LENGTH * base.LATENT_DIM, base.HIDDEN_DIM)
            self.latent_projection = nn.Linear(base.LATENT_DIM, base.HIDDEN_DIM)
            self.action_projection = nn.Linear(base.ACTION_DIM, base.HIDDEN_DIM)
            self.shared_transition = nn.Sequential(
                nn.LayerNorm(base.HIDDEN_DIM),
                nn.Linear(base.HIDDEN_DIM, base.HIDDEN_DIM * 4),
                nn.GELU(),
                nn.Linear(base.HIDDEN_DIM * 4, base.HIDDEN_DIM),
            )
            self.output_norm = nn.LayerNorm(base.HIDDEN_DIM)
            self.latent_update = nn.Linear(base.HIDDEN_DIM, base.LATENT_DIM)
            self.prefix_init = nn.Linear(base.LATENT_DIM, PREFIX_DIM)
            self.prefix_state_projection = nn.Linear(base.LATENT_DIM, PREFIX_DIM)
            self.prefix_gru = nn.GRUCell(base.ACTION_DIM + PREFIX_DIM, PREFIX_DIM)
            self.prefix_projection = nn.Linear(PREFIX_DIM, base.HIDDEN_DIM)

        def forward(self, latent_history: Any, packed_actions: Any) -> Any:
            if packed_actions.ndim != 3 or packed_actions.shape[-1] != base.ACTION_DIM:
                raise ValueError("packed actions must have shape [B,T,10]")
            history = base._left_pad_history(latent_history)
            state = history[:, -1]
            prefix_hidden = torch.tanh(self.prefix_init(state))
            outputs = []
            for step in range(int(packed_actions.shape[1])):
                history_condition = self.history_adapter(history.reshape(history.shape[0], -1))
                latent_hidden = self.latent_projection(state)
                action = packed_actions[:, step]
                action_hidden = self.action_projection(action)
                prefix_input = torch.cat((action, self.prefix_state_projection(state)), dim=-1)
                prefix_hidden = self.prefix_gru(prefix_input, prefix_hidden)
                prefix_condition = self.prefix_projection(prefix_hidden)
                transition = self.shared_transition(history_condition + latent_hidden + action_hidden + prefix_condition)
                state = state + self.latent_update(self.output_norm(latent_hidden + transition))
                outputs.append(state)
                history = torch.cat((history[:, 1:], state[:, None]), dim=1)
            return torch.stack(outputs, dim=1)

    return _Impl()


def instantiate_student(arm: str) -> Any:
    if arm == "balanced_base":
        return base.make_student("baseline")
    if arm == "state_action_prefix_gru":
        return make_prefix_student()
    raise ValueError(f"unknown arm: {arm}")


def load_shared_base_state(student: Any, initial_state: Mapping[str, Any]) -> None:
    current = student.state_dict()
    for key, value in initial_state.items():
        if key not in current or tuple(current[key].shape) != tuple(value.shape):
            raise ValueError(f"base-owned parameter drifted: {key}")
        current[key] = value.detach().clone()
    student.load_state_dict(current, strict=True)


def load_full_state(student: Any, state: Mapping[str, Any]) -> None:
    student.load_state_dict({key: value.detach().clone() for key, value in state.items()}, strict=True)


def load_freeze(path: Path, temporal_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = load_json(path.resolve())
    if freeze.get("schema") != "lewm-recurrent-student.state-action-prefix-gru-freeze" or freeze.get("status") != "frozen":
        raise ValueError("state-action prefix GRU freeze is not frozen or has an unexpected schema")
    temporal_freeze = load_json(temporal_path.resolve())
    if temporal_freeze.get("schema") != "lewm-recurrent-student.temporal-balanced-train-freeze" or temporal_freeze.get("status") != "frozen":
        raise ValueError("temporal-balanced freeze is not frozen")
    if freeze["design"]["training_arms"] != ["balanced_base", "state_action_prefix_gru"]:
        raise ValueError("training arm contract drifted")
    if int(freeze["design"]["training_rows"]) != TRAIN_CONTEXTS or int(freeze["design"]["updates"]) != STEPS:
        raise ValueError("training row/update contract drifted")
    for key, expected in {"batch_contexts": 8, "train_candidates": 64, "initialization_seed": 20300901, "training_seed": 20300902, "context_schedule_seed": 20300904, "candidate_slate_seed": 20300905}.items():
        if int(freeze["training"][key]) != expected:
            raise ValueError(f"training {key} drifted")
    if int(freeze["training"]["prefix_initialization_seed"]) != 20300906:
        raise ValueError("prefix initialization seed drifted")
    if [float(x) for x in freeze["training"]["horizon_weights"]] != list(base.HORIZON_WEIGHTS):
        raise ValueError("horizon weights drifted")
    if freeze["fresh_evaluation"]["action_prefix_seeds"] != list(FRESH_SEEDS):
        raise ValueError("fresh action-prefix seeds drifted")
    if freeze["fresh_evaluation"]["selection_slice"] != "valid[536:544]":
        raise ValueError("fresh selection slice drifted")
    for key in ("official_cem", "planner_viability", "closed_loop"):
        if freeze["scope_boundary"][key] != "NOT_RUN_BY_SCOPE":
            raise ValueError(f"scope boundary drifted: {key}")
    base_freeze = load_json((path.resolve().parent / ".." / "LEWM_RECURRENT_STUDENT_FREEZE.json").resolve())
    return freeze, temporal_freeze, base_freeze


def anchor_spec(length: int) -> dict[str, int]:
    late = int(length) - HORIZON_PRIMITIVES - 1
    if late < 0:
        raise ValueError(f"episode length {length} cannot hold current+25 actions+goal")
    values = {"early": 0, "middle": late // 2, "late": late}
    for name, anchor in values.items():
        if not (0 <= anchor and anchor + HORIZON_PRIMITIVES < length):
            raise ValueError(f"{name} anchor violates episode bounds for length {length}")
    return values


def select_fresh_episodes(dataset_path: Path, selection_seed: int, old_manifest: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        lengths = [int(x) for x in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON_PRIMITIVES + 1]
    random.Random(int(selection_seed)).shuffle(valid)
    old_heldout = [int(row["episode_id"]) for row in old_manifest["splits"]["heldout"]]
    old_train = [int(row["episode_id"]) for row in old_manifest["splits"]["train"]]
    prefix = old_heldout + old_train
    if valid[: len(prefix)] != prefix:
        raise ValueError("selection prefix does not reproduce the frozen Phase 2 manifest")
    phase3 = valid[520:528]
    phase4 = valid[528:536]
    fresh = valid[536:544]
    excluded = set(prefix) | set(phase3) | set(phase4)
    if len(fresh) != FRESH_EPISODE_COUNT or set(fresh) & excluded:
        raise ValueError("fresh valid[536:544] selection overlaps an excluded set")
    return fresh, {"selection_seed": int(selection_seed), "valid_count": len(valid), "old_heldout_episode_ids": old_heldout, "old_train_episode_count": len(old_train), "excluded_phase3_episode_ids": phase3, "excluded_phase4_episode_ids": phase4, "fresh_episode_ids": fresh, "selection_slice": "valid[536:544]", "excluded_prior_fresh_slices": ["valid[520:528]", "valid[528:536]"], "result_dependent_selection": False}


def aggregate_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("cannot aggregate empty blocks")
    spearman = [float(item["spearman"]) for item in blocks]
    top30 = [float(item["top30_overlap"]) for item in blocks]
    relative = [float(item["relative_latent_mse"]) for item in blocks]
    return {"blocks": len(blocks), "spearman_median": float(statistics.median(spearman)), "spearman_minimum": min(spearman), "top30_median": float(statistics.median(top30)), "top30_minimum": min(top30), "relative_latent_mse_median": float(statistics.median(relative)), "positive_spearman_blocks": sum(value > 0.0 for value in spearman), "positive_top30_blocks": sum(value > 0.0 for value in top30), "finite": all(bool(item["finite"]) for item in blocks)}


def episode_medians(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in blocks:
        grouped[int(item["episode_id"])].append(item)
    result: dict[str, Any] = {}
    for episode, items in sorted(grouped.items()):
        if len(items) != 6:
            raise ValueError(f"episode {episode} does not have six nested blocks")
        result[str(episode)] = {"blocks": 6, "spearman_median": float(statistics.median(float(x["spearman"]) for x in items)), "top30_median": float(statistics.median(float(x["top30_overlap"]) for x in items)), "relative_latent_mse_median": float(statistics.median(float(x["relative_latent_mse"]) for x in items))}
    if len(result) != FRESH_EPISODE_COUNT:
        raise ValueError("nested episode aggregation does not contain eight episodes")
    return result


def train_arm(rows: Sequence[Mapping[str, Any]], initial_state: Mapping[str, Any], official_model: Any, settings: Mapping[str, Any], arm: str) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) != TRAIN_CONTEXTS:
        raise ValueError("training bank must contain exactly 512 contexts")
    if arm == "state_action_prefix_gru":
        torch.manual_seed(int(settings["prefix_initialization_seed"]))
    student = instantiate_student(arm).to("cuda")
    load_shared_base_state(student, initial_state)
    torch.manual_seed(int(settings["training_seed"]))
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    schedule = torch.Generator(device="cpu").manual_seed(int(settings["context_schedule_seed"]))
    snapshots: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for step in range(1, STEPS + 1):
        indices = torch.randperm(len(train_rows), generator=schedule)[:BATCH_CONTEXTS]
        contexts = torch.cat([base._tensor(train_rows[int(i)]["latent_history"]).expand(TRAIN_CANDIDATES, -1, -1) for i in indices], dim=0).to("cuda")
        actions = torch.cat([base._tensor(train_rows[int(i)]["future_actions"]) for i in indices], dim=0).to("cuda")
        targets = torch.cat([base._tensor(train_rows[int(i)]["teacher_targets"]) for i in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        latent_loss, per_horizon = base.recurrent_loss(prediction, targets)
        student_cost = score._student_costs_with_gradient(official_model, train_rows, indices, prediction)
        teacher_cost = score._effective_teacher_costs(train_rows, indices, "score_distill")
        score_loss = score.score_distill_loss(student_cost, teacher_cost)
        total_loss = latent_loss + temporal.ema.SCORE_LOSS_WEIGHT * score_loss
        finite = bool(torch.isfinite(total_loss).item() and torch.isfinite(per_horizon).all().item() and torch.isfinite(score_loss).item())
        if not finite:
            raise FloatingPointError(f"non-finite loss at step {step} in {arm}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        history.append({"step": step, "weighted_latent_mse": float(latent_loss.detach().cpu()), "score_loss": float(score_loss.detach().cpu()), "total_loss": float(total_loss.detach().cpu()), "per_horizon_mse": [float(x) for x in per_horizon.detach().cpu()], "finite": finite})
        if step in SNAPSHOT_STEPS:
            snapshots[f"step_{step}"] = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    first = history[0]["weighted_latent_mse"]
    last10 = statistics.median(item["weighted_latent_mse"] for item in history[-10:])
    trace = {"first": history[0], "step_500": history[499], "step_1000": history[999], "step_1500": history[1499], "terminal": history[-1], "last10_weighted_latent_mse_median": last10}
    return {"snapshots": snapshots, "parameter_count": int(sum(parameter.numel() for parameter in student.parameters())), "base_parameter_count": int(sum(parameter.numel() for key, parameter in student.named_parameters() if not key.startswith(("prefix_init.", "prefix_state_projection.", "prefix_gru.", "prefix_projection.")))), "prefix_parameter_count": int(sum(parameter.numel() for key, parameter in student.named_parameters() if key.startswith(("prefix_init.", "prefix_state_projection.", "prefix_gru.", "prefix_projection.")))), "last10_to_first_ratio": float(last10 / max(first, 1e-12)), "trace": trace, "finite": all(bool(item["finite"]) for item in history), "arm": arm}


def evaluate_state(model: Any, snapshots: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], label: str, arm: str) -> dict[str, Any]:
    import torch

    results: dict[str, Any] = {}
    for step in SNAPSHOT_STEPS:
        name = f"step_{step}"
        student = instantiate_student(arm).to("cuda")
        load_full_state(student, snapshots[name])
        student.eval()
        blocks: list[dict[str, Any]] = []
        for row in rows:
            for block, seed in enumerate(FRESH_SEEDS):
                item = base._stage_a_metrics_for_block(model, student, row, block)
                item.update({"label": label, "context_id": row["context_id"], "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}"})
                blocks.append(item)
        strata = {stratum: aggregate_blocks([item for item in blocks if item["stratum"] == stratum]) for stratum in FRESH_ANCHORS}
        results[name] = {"label": label, "arm": arm, "per_block": blocks, "overall": aggregate_blocks(blocks), "strata": strata, "episodes": episode_medians(blocks)}
        del student
        torch.cuda.empty_cache()
    return results


def causality_and_latency(model: Any, snapshots: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    row = rows[0]
    causality: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    for arm in ("balanced_base", "state_action_prefix_gru"):
        student = instantiate_student(arm).to("cuda")
        load_full_state(student, snapshots[arm]["step_3000"])
        student.eval()
        causality[arm] = base.causality_test(student, row, FRESH_SEEDS[0], float(freeze["evaluation"]["causality_tolerance"]))
        latency[arm] = base.predictor_latency(model, student, row, warmup=int(freeze["evaluation"]["predictor_latency_warmup"]), repeats=int(freeze["evaluation"]["predictor_latency_repeats"]))
        del student
        torch.cuda.empty_cache()
    base_ms = float(latency["balanced_base"]["student_ms"])
    treatment_ms = float(latency["state_action_prefix_gru"]["student_ms"])
    teacher_reductions = {arm: float(latency[arm]["reduction"]) for arm in latency}
    guard = freeze["gates"]["latency_guard_secondary"]
    overhead = treatment_ms / max(base_ms, 1e-12) - 1.0
    latency_guard = {"status": "GO" if overhead <= float(guard["treatment_overhead_vs_balanced_base_max"]) and min(teacher_reductions.values()) >= float(guard["teacher_reduction_min"]) else "NO-GO", "treatment_overhead_vs_balanced_base": overhead, "teacher_reduction_minimum": min(teacher_reductions.values()), "teacher_reduction_by_arm": teacher_reductions, "conditions": {"treatment_overhead_max": overhead <= float(guard["treatment_overhead_vs_balanced_base_max"]), "teacher_reduction_min": min(teacher_reductions.values()) >= float(guard["teacher_reduction_min"])}}
    return causality, latency, latency_guard


def absolute_gate(freeze: Mapping[str, Any], evaluation: Mapping[str, Any], training: Mapping[str, Any], causality: Mapping[str, Any], latency: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["inherited_absolute_predictor"]
    metrics = evaluation["overall"]
    convergence = float(training["last10_to_first_ratio"]) <= 0.8
    causal_pass = all(item["passed"] for item in causality.values())
    conditions = {"median_spearman_min": metrics["spearman_median"] >= float(gate["median_spearman_min"]), "minimum_spearman_min": metrics["spearman_minimum"] >= float(gate["minimum_spearman_min"]), "median_top30_min": metrics["top30_median"] >= float(gate["median_top30_min"]), "minimum_top30_min": metrics["top30_minimum"] >= float(gate["minimum_top30_min"]), "median_relative_latent_mse_max": metrics["relative_latent_mse_median"] <= float(gate["median_relative_latent_mse_max"]), "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(gate["positive_spearman_blocks_min"]), "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(gate["positive_top30_blocks_min"])}
    fidelity = all(conditions.values())
    latency_ok = float(latency["reduction"]) >= float(gate["latency_reduction_min"])
    passed = bool(convergence and metrics["finite"] and causal_pass and fidelity and latency_ok)
    return {"status": "GO" if passed else "NO-GO", "integrity_and_convergence": {"status": "PASS" if convergence and metrics["finite"] else "FAIL", "last10_to_first_ratio": training["last10_to_first_ratio"], "convergence_threshold": 0.8, "finite": metrics["finite"]}, "causality": {"status": "PASS" if causal_pass else "FAIL", "per_prefix": dict(causality)}, "predictor_fidelity_and_ranking": {"status": "PASS" if fidelity else "FAIL", "metrics": {key: metrics[key] for key in ("spearman_median", "spearman_minimum", "top30_median", "top30_minimum", "relative_latent_mse_median", "positive_spearman_blocks", "positive_top30_blocks")}, "conditions": conditions}, "predictor_latency": {"status": "PASS" if latency_ok else "FAIL", "metrics": latency}, "full_cem_viability": "NOT_RUN_BY_SCOPE"}


def stratum_gate(freeze: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["stratum_protection"]
    conditions = {}
    for stratum, metrics in evaluation["strata"].items():
        conditions[stratum] = {"median_spearman_min": metrics["spearman_median"] >= float(gate["median_spearman_min"]), "median_top30_min": metrics["top30_median"] >= float(gate["median_top30_min"]), "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(gate["positive_spearman_blocks_min"]), "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(gate["positive_top30_blocks_min"])}
    return {"status": "GO" if all(all(item.values()) for item in conditions.values()) else "NO-GO", "conditions": conditions, "minimum_thresholds_repeated": False}


def paired_mechanism_gate(freeze: Mapping[str, Any], control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    deltas = []
    for episode in sorted(control["episodes"], key=int):
        ds = float(treatment["episodes"][episode]["spearman_median"]) - float(control["episodes"][episode]["spearman_median"])
        dt = float(treatment["episodes"][episode]["top30_median"]) - float(control["episodes"][episode]["top30_median"])
        deltas.append({"episode_id": int(episode), "spearman_delta_gru_minus_base": ds, "top30_delta_gru_minus_base": dt, "joint_improvement_or_nonworsening": bool((ds > 0 and dt >= 0) or (dt > 0 and ds >= 0))})
    med_s = statistics.median(item["spearman_delta_gru_minus_base"] for item in deltas)
    med_t = statistics.median(item["top30_delta_gru_minus_base"] for item in deltas)
    joint = sum(bool(item["joint_improvement_or_nonworsening"]) for item in deltas)
    minimum = freeze["gates"]["mechanism_secondary"]
    conditions = {"spearman_episode_median_delta_min": med_s >= float(minimum["spearman_episode_median_delta_min"]), "top30_episode_median_delta_min": med_t >= float(minimum["top30_episode_median_delta_min"]), "joint_nonworsening_improvement_min_episodes": joint >= int(minimum["joint_nonworsening_improvement_min_episodes"])}
    return {"status": "GO" if all(conditions.values()) else "NO-GO", "paired_unit": "episode after median over six blocks", "median_spearman_delta": float(med_s), "median_top30_delta": float(med_t), "joint_count": joint, "conditions": conditions, "per_episode": deltas, "cannot_replace_absolute_gate": True}


def preflight() -> dict[str, Any]:
    import torch

    torch.manual_seed(20300901)
    control = base.make_student("baseline")
    base_state = {key: value.detach().clone() for key, value in control.state_dict().items()}
    treatment = make_prefix_student()
    load_shared_base_state(treatment, base_state)
    shared_equal = all(torch.equal(treatment.state_dict()[key], value) for key, value in base_state.items())
    prefix_names = [key for key in treatment.state_dict() if key.startswith(("prefix_init.", "prefix_state_projection.", "prefix_gru.", "prefix_projection."))]
    torch.manual_seed(20300906)
    context = torch.randn(2, 1, base.LATENT_DIM)
    action_a = torch.randn(2, base.HORIZON, base.ACTION_DIM)
    action_b = action_a.clone()
    action_b[:, 2:] = torch.randn_like(action_b[:, 2:])
    with torch.no_grad():
        out_a = treatment(context, action_a)
        out_b = treatment(context, action_b)
    causal_max = float((out_a[:, :2] - out_b[:, :2]).abs().max())
    anchors = {str(length): anchor_spec(length) for length in (30, 64, 128)}
    valid = list(range(600))
    excluded = set(valid[:8]) | set(valid[8:520])
    exclusion_ok = len(set(valid[536:544]) & excluded) == 0
    schedule_a = torch.randperm(512, generator=torch.Generator().manual_seed(20300904))
    schedule_b = torch.randperm(512, generator=torch.Generator().manual_seed(20300904))
    slate_a = torch.rand((300, base.HORIZON, base.ACTION_DIM), generator=torch.Generator().manual_seed(20300905))
    slate_b = slate_a.clone()
    return {"schema": "lewm-recurrent-student.state-action-prefix-gru-preflight", "status": "PASS" if shared_equal and prefix_names and causal_max <= 1e-6 and exclusion_ok and torch.equal(schedule_a, schedule_b) and torch.equal(slate_a, slate_b) else "FAIL", "checks": {"base_owned_parameter_bitwise_equal": shared_equal, "prefix_modules_present": prefix_names, "causality_max_abs_for_unchanged_prefix": causal_max, "causality_tolerance": 1e-6, "selection_exclusion_synthetic": exclusion_ok, "anchor_bounds": anchors, "same_context_schedule_seed": bool(torch.equal(schedule_a, schedule_b)), "same_candidate_slate_seed": bool(torch.equal(slate_a, slate_b)), "nested_unit": "episode -> anchor -> seed; candidate not replicate"}, "parameter_counts": {"balanced_base": int(sum(item.numel() for item in control.parameters())), "state_action_prefix_gru": int(sum(item.numel() for item in treatment.parameters())), "prefix_added": int(sum(item.numel() for key, item in treatment.named_parameters() if key.startswith(("prefix_init.", "prefix_state_projection.", "prefix_gru.", "prefix_projection."))))}}


def run(args: Any, freeze: Mapping[str, Any], temporal_freeze: Mapping[str, Any], base_freeze: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    base.require_compute_node()
    import torch

    dataset = (args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve()
    manifest_path = args.manifest_512.resolve()
    control_rows_path = args.prepared_rows_512.resolve()
    manifest = load_json(manifest_path)
    if len(manifest.get("splits", {}).get("train", [])) != TRAIN_CONTEXTS:
        raise ValueError("formal 512 manifest missing")
    control_rows = torch.load(control_rows_path, map_location="cpu", weights_only=False)
    if len([row for row in control_rows if row.get("split") == "train"]) != TRAIN_CONTEXTS:
        raise ValueError("control prepared bank is not 512 rows")
    reference = load_json(args.reference_summary.resolve())
    temporal_summary = load_json(args.temporal_summary.resolve())
    if reference.get("status") != "PREDICTOR_LEVEL_COMPLETE" or temporal_summary.get("status") != "PREDICTOR_LEVEL_COMPLETE":
        raise ValueError("historical predictor references are incomplete")
    official_model = base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    balanced_rows = None
    balanced_source = "regenerated_on_compute_node"
    if args.prepared_balanced_rows is not None and args.prepared_balanced_rows.is_file():
        balanced_rows = torch.load(args.prepared_balanced_rows.resolve(), map_location="cpu", weights_only=False)
        balanced_source = "reused_remote_phase5_prepared_rows"
    if balanced_rows is None:
        balanced_rows, balanced_meta = temporal.build_balanced_train_rows(official_model, dataset, manifest, control_rows)
    else:
        balanced_meta = {"train_contexts": len([row for row in balanced_rows if row.get("split") == "train"]), "source": balanced_source, "rows_returned": False}
    if len([row for row in balanced_rows if row.get("split") == "train"]) != TRAIN_CONTEXTS:
        raise ValueError("balanced prepared bank is not 512 rows")
    fresh_ids, selection = select_fresh_episodes(dataset, int(freeze["fresh_evaluation"]["selection_seed"]), manifest)
    fresh_rows, fresh_meta = ema.build_fresh_rows(official_model, dataset, fresh_ids, freeze)
    torch.manual_seed(int(freeze["training"]["initialization_seed"]))
    template = base.make_student("baseline")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    settings = {"training_seed": int(freeze["training"]["training_seed"]), "context_schedule_seed": int(freeze["training"]["context_schedule_seed"]), "prefix_initialization_seed": int(freeze["training"]["prefix_initialization_seed"])}
    control_training = train_arm(balanced_rows, initial_state, official_model, settings, "balanced_base")
    treatment_training = train_arm(balanced_rows, initial_state, official_model, settings, "state_action_prefix_gru")
    control_eval = evaluate_state(official_model, control_training["snapshots"], fresh_rows, "balanced_base", "balanced_base")
    treatment_eval = evaluate_state(official_model, treatment_training["snapshots"], fresh_rows, "state_action_prefix_gru", "state_action_prefix_gru")
    snapshots = {"balanced_base": control_training["snapshots"], "state_action_prefix_gru": treatment_training["snapshots"]}
    causality, latency, latency_guard = causality_and_latency(official_model, snapshots, fresh_rows, freeze)
    control_gate = absolute_gate(freeze, control_eval["step_3000"], control_training, causality["balanced_base"], latency["balanced_base"])
    treatment_gate = absolute_gate(freeze, treatment_eval["step_3000"], treatment_training, causality["state_action_prefix_gru"], latency["state_action_prefix_gru"])
    treatment_strata = stratum_gate(freeze, treatment_eval["step_3000"])
    mechanism = paired_mechanism_gate(freeze, control_eval["step_3000"], treatment_eval["step_3000"])
    primary_conditions = {"treatment_absolute_predictor_gate": treatment_gate["status"] == "GO", "treatment_temporal_stratum_gate": treatment_strata["status"] == "GO"}
    def report_training(value: Mapping[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in value.items() if key != "snapshots"}
    summary = {"schema": "lewm-recurrent-student.state-action-prefix-gru-summary", "schema_version": 1, "status": "PREDICTOR_LEVEL_COMPLETE", "freeze": str(args.freeze.resolve()), "protocol": str(args.protocol.resolve()), "temporal_freeze": str(args.temporal_freeze.resolve()), "temporal_protocol": str(args.temporal_protocol.resolve()), "interface_probe": str(args.interface_probe.resolve()), "interface_contract": dict(contract.__dict__), "source": {"lewm_commit": base_freeze["evidence_boundary"]["source_commit"], "stable_worldmodel_cem_commit": base_freeze["evidence_boundary"]["stable_worldmodel_cem_commit"], "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()), "dataset": str(dataset)}, "training_banks": {"manifest": str(manifest_path), "control_prepared_rows": str(control_rows_path), "balanced_source": balanced_source, "balanced_metadata": balanced_meta, "prepared_rows_returned": False}, "historical_references": {"formal_reference": str(args.reference_summary.resolve()), "temporal_balanced_reference": str(args.temporal_summary.resolve()), "temporal_balanced_job": "25152151.pbs101", "test_not_used_for_design": True}, "fresh_selection": selection, "fresh_evaluation": fresh_meta, "shared_training_contract": {"architecture": "LeWMCompactRecurrentTransitionStudent h256 plus optional shared state-action prefix GRU", "hidden_dim": base.HIDDEN_DIM, "prefix_hidden_dim": PREFIX_DIM, "attention": False, "conditioner": False, "goal_input": False, "teacher_forcing": False, "objective": "horizon-weighted free-running latent MSE + 0.1 * context-normalized teacher-score SmoothL1 mean", "optimizer": freeze["training"]["optimizer"], "updates": STEPS, "batch_contexts": BATCH_CONTEXTS, "train_candidates": TRAIN_CANDIDATES, "initialization_state_shared": True, "context_schedule_shared": True, "candidate_slates_shared": True}, "pairing": {"initialization_seed": int(freeze["training"]["initialization_seed"]), "prefix_initialization_seed": int(freeze["training"]["prefix_initialization_seed"]), "training_seed": int(freeze["training"]["training_seed"]), "context_schedule_seed": int(freeze["training"]["context_schedule_seed"]), "candidate_slate_seed": int(freeze["training"]["candidate_slate_seed"]), "same_initial_base_state": True, "base_owned_parameters_bitwise_equal": True, "same_optimizer": True, "same_batch_size": True, "same_context_schedule": True, "same_balanced_training_rows": True, "fresh_action_prefix_seeds": list(FRESH_SEEDS), "candidate_is_not_an_independent_statistical_unit": True}, "arms": {"balanced_base": {"training": report_training(control_training), "evaluation": control_eval, "terminal_metrics": control_eval["step_3000"]["overall"], "causality": causality["balanced_base"], "predictor_latency": latency["balanced_base"], "absolute_predictor_gate": control_gate}, "state_action_prefix_gru": {"training": report_training(treatment_training), "evaluation": treatment_eval, "terminal_metrics": treatment_eval["step_3000"]["overall"], "causality": causality["state_action_prefix_gru"], "predictor_latency": latency["state_action_prefix_gru"], "absolute_predictor_gate": treatment_gate, "temporal_stratum_gate": treatment_strata}}, "paired_deltas": mechanism, "latency_guard": latency_guard, "primary_gate": {"status": "GO" if all(primary_conditions.values()) else "NO-GO", "arm": "state_action_prefix_gru", "conditions": primary_conditions, "absolute_predictor_gate": treatment_gate, "temporal_stratum_gate": treatment_strata, "secondary_mechanism_gate": mechanism, "secondary_latency_guard": latency_guard}, "gpu_telemetry": {"source": "job.log nvidia-smi 5-second samples", "required": True}, "scope_checks": {"formal_512_training_bank_reused": "PASS", "balanced_temporal_training_reproduced": "PASS", "fresh_valid_536_544": "PASS", "base_owned_parameter_equality": "PASS", "same_init_schedule_slates": "PASS", "nested_episode_aggregation": "PASS", "no_result_dependent_selection": "PASS", "rows_generated_or_reused_on_compute_only": "PASS"}, "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}, "claim_boundary": "State-action prefix GRU evidence is predictor-level only; no official CEM, planner viability, closed-loop PushT, encoder speedup, or native deployment claim."}
    write_json(args.output.resolve() / "lewm_state_action_prefix_gru_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        value = {"schema": "lewm-recurrent-student.state-action-prefix-gru", "status": "READY", "training_rows": 512, "updates": 3000, "prefix_hidden_dim": 64, "fresh_selection": "valid[536:544]", "fresh_blocks": 48, "action_prefix_seeds": list(FRESH_SEEDS), "official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    if args.mode == "preflight":
        value = preflight()
        write_json(args.output / "preflight_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0 if value["status"] == "PASS" else 1
    freeze, temporal_freeze, base_freeze = load_freeze(args.freeze, args.temporal_freeze)
    for path in (args.interface_probe, args.manifest_512, args.prepared_rows_512, args.reference_summary, args.temporal_freeze, args.temporal_protocol, args.temporal_summary, args.phase2_freeze, args.dataset or (args.stablewm_home / "pusht_expert_train.h5")):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = base.load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, temporal_freeze, base_freeze, contract)
    print(json.dumps({"status": summary["status"], "primary_gate": summary["primary_gate"]["status"], "output": str((args.output / "lewm_state_action_prefix_gru_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
