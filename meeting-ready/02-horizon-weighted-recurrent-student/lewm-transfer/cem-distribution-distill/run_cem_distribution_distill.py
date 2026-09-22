#!/usr/bin/env python3
"""Fixed-observation LeWM CEM-distribution score-distillation pilot."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
REFERENCE_DIR = TRANSFER / "state-action-prefix-gru"
ANCHOR_DIR = TRANSFER / "anchor-aligned-bank"
SCHEMA = "lewm-recurrent-student.cem-distribution-distill-runner"
TRAIN_CONTEXTS = 512
TRAIN_CANDIDATES = 64
BATCH_CONTEXTS = 8
HORIZON = 5
ACTION_DIM = 10
LATENT_DIM = 192
FRESH_ANCHORS = ("early", "middle", "late")
FRESH_SEEDS = (20300987, 20300988)
FRESH_SLICE = (560, 568)
NUM_CANDIDATES = 300
ELITE_COUNT = 30
CEM_ITERATIONS = 30
CEM_CHECKPOINTS = (10, 20, 30)
TRAIN_TAIL_INDICES = {
    10: (0, 30, 60, 90, 120, 150, 179, 209, 239, 269, 299),
    20: (0, 30, 60, 90, 120, 150, 179, 209, 239, 269, 299),
    30: (0, 33, 66, 100, 133, 166, 199, 233, 266, 299),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--anchor-rows", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--temporal-freeze", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
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


def load_anchor_module() -> Any:
    if str(ANCHOR_DIR) not in sys.path:
        sys.path.insert(0, str(ANCHOR_DIR))
    return importlib.import_module("run_anchor_aligned_bank")


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required for model/HDF5/training work")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and Path(nodefile).is_file():
        nodes = {line.strip().lower() for line in Path(nodefile).read_text().splitlines() if line.strip()}
        short_host = host.split(".", 1)[0]
        short_nodes = {node.split(".", 1)[0] for node in nodes}
        if nodes and host not in nodes and short_host not in short_nodes:
            raise RuntimeError(f"current host {host} is not in PBS_NODEFILE allocation")


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "lewm-recurrent-student.cem-distribution-distill-freeze":
        raise ValueError("unexpected CEM-distribution freeze schema")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("freeze must be frozen_before_results")
    training = freeze.get("training", {})
    expected = {"contexts": TRAIN_CONTEXTS, "batch_contexts": BATCH_CONTEXTS, "candidates_per_context": TRAIN_CANDIDATES, "extra_updates": 1000, "training_seed": 20300982, "context_schedule_seed": 20300984}
    for key, value in expected.items():
        if int(training.get(key, -1)) != value:
            raise ValueError(f"training contract drifted: {key}")
    if int(freeze["cem_collection"].get("iterations", -1)) != CEM_ITERATIONS:
        raise ValueError("CEM iteration count drifted")
    cem = freeze["cem_collection"]
    for key, value in {"saved_checkpoints": list(CEM_CHECKPOINTS), "num_samples": NUM_CANDIDATES, "topk": ELITE_COUNT, "horizon": HORIZON, "packed_action_dim": ACTION_DIM}.items():
        if cem.get(key) != value:
            raise ValueError(f"CEM contract drifted: {key}")
    if cem.get("candidate_zero_is_pre_update_mu") is not True or cem.get("std_unbiased") is not True:
        raise ValueError("candidate-zero/std contract drifted")
    if cem.get("selection_operator") != "torch.topk(cost, k=30, largest=False), matching pinned stable-worldmodel LeWM solver":
        raise ValueError("LeWM topk operator drifted")
    fresh = freeze.get("fresh_evaluation", {})
    if fresh.get("selection_slice") != "valid[560:568]" or int(fresh.get("selection_seed", -1)) != 20300903:
        raise ValueError("fresh selection contract drifted")
    if list(fresh.get("action_prefix_seeds", [])) != list(FRESH_SEEDS) or int(fresh.get("blocks", -1)) != 48:
        raise ValueError("fresh block contract drifted")
    gates = freeze.get("gates", {})
    primary = gates.get("primary_mechanism", {})
    if float(primary.get("treatment_minus_control_median_delta_max", 1.0)) > -0.05 or int(primary.get("strictly_improved_episodes_min", -1)) != 5:
        raise ValueError("primary mechanism gate drifted")
    required_scope = ("fixed observation", "student-driven", "closed-loop", "on-policy")
    return dict(freeze)


def validate_protocol(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    required = ("valid[560:568]", "20300903", "student-driven", "teacher shadow", "closed-loop", "torch.topk")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol missing frozen terms: {missing}")


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


def load_train_rows(path: Path) -> list[Mapping[str, Any]]:
    import torch

    rows = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError("anchor rows must be a list")
    train = [row for row in rows if row.get("split") == "train"]
    if len(train) != TRAIN_CONTEXTS:
        raise ValueError("anchor rows must contain exactly 512 train contexts")
    for ordinal, row in enumerate(train):
        if int(row.get("ordinal", ordinal)) != ordinal:
            raise ValueError(f"anchor row ordinal drifted at {ordinal}")
        if tuple(row["future_actions"].shape) != (TRAIN_CANDIDATES, HORIZON, ACTION_DIM):
            raise ValueError(f"anchor action bank shape drifted at {ordinal}")
        if tuple(row["teacher_targets"].shape) != (TRAIN_CANDIDATES, HORIZON, LATENT_DIM):
            raise ValueError(f"anchor target bank shape drifted at {ordinal}")
        if tuple(row["teacher_objective"].shape) != (TRAIN_CANDIDATES,):
            raise ValueError(f"anchor objective bank shape drifted at {ordinal}")
    return train


def load_start_state(reference: Any, checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    raw = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("provenance"), Mapping):
        raise ValueError("anchor checkpoint provenance is missing")
    provenance = dict(raw["provenance"])
    if provenance.get("source") != "anchor_aligned_bank_treatment" or int(provenance.get("updates", -1)) != 3000:
        raise ValueError("start checkpoint is not the frozen anchor-aligned step3000 checkpoint")
    state = raw.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("anchor checkpoint lacks state_dict")
    copied = {str(key): value.detach().clone() for key, value in state.items()}
    reference.base.make_student("baseline").load_state_dict(copied, strict=True)
    return copied, {"path": str(checkpoint.resolve()), "source": provenance.get("source"), "provenance": provenance}


def select_fresh(dataset: Path, manifest: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    with h5py.File(dataset.resolve(), "r") as handle:
        lengths = [int(value) for value in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON * 5 + 1]
    random.Random(20300903).shuffle(valid)
    prefix = [int(row["episode_id"]) for row in manifest["splits"]["heldout"] + manifest["splits"]["train"]]
    if valid[: len(prefix)] != prefix:
        raise ValueError("valid shuffle prefix cannot be reproduced")
    selected = valid[FRESH_SLICE[0] : FRESH_SLICE[1]]
    if len(selected) != 8 or set(selected) & set(valid[: FRESH_SLICE[0]]):
        raise ValueError("fresh valid[560:568] overlaps excluded prefix")
    return selected, {"selection_seed": 20300903, "selection_slice": "valid[560:568]", "excluded_valid_prefix": 560, "fresh_episode_ids": selected, "result_dependent_selection": False}


def tensor(value: Any, device: str = "cuda") -> Any:
    import torch

    return value if torch.is_tensor(value) and str(value.device) == device else torch.as_tensor(value, dtype=torch.float32, device=device)


def student_costs(reference: Any, official: Any, student: Any, row: Mapping[str, Any], actions: Any) -> Any:
    import torch

    latent = tensor(row["latent_history"])
    context = latent.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        prediction = student(context, actions)
        return reference.base._official_objective(official, row, prediction)


def teacher_targets_and_costs(reference: Any, official: Any, row: Mapping[str, Any], actions: Any) -> tuple[Any, Any]:
    import torch

    latent = tensor(row["latent_history"])
    context = latent.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        targets = reference.base.official_teacher_targets(official, context, actions)
        costs = reference.base._official_objective(official, row, targets)
    return targets, costs


def run_student_cem(reference: Any, official: Any, student: Any, row: Mapping[str, Any], innovations: Any, checkpoints: Sequence[int], tail_indices: Mapping[int, Sequence[int]] | None = None) -> tuple[dict[int, Any], dict[str, Any], dict[int, Any], dict[int, Any]]:
    import torch

    device = torch.device("cuda")
    mu = torch.zeros((HORIZON, ACTION_DIM), device=device)
    sigma = torch.ones_like(mu)
    saved_actions: dict[int, Any] = {}
    saved_targets: dict[int, Any] = {}
    saved_costs: dict[int, Any] = {}
    metadata: dict[str, Any] = {"rounds": []}
    for round_index, epsilon_cpu in enumerate(innovations, start=1):
        epsilon = epsilon_cpu.to(device)
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        actions[0] = mu
        if not bool(torch.equal(actions[0], mu)):
            raise RuntimeError("CEM candidate zero drifted from pre-update mean")
        if not bool(torch.isfinite(actions).all().item()):
            raise RuntimeError(f"non-finite CEM actions at round {round_index}")
        costs = student_costs(reference, official, student, row, actions)
        if not bool(torch.isfinite(costs).all().item()):
            raise RuntimeError(f"non-finite student CEM costs at round {round_index}")
        selected = torch.topk(costs, k=ELITE_COUNT, largest=False, sorted=True).indices
        next_mu = actions.index_select(0, selected).mean(dim=0)
        next_sigma = actions.index_select(0, selected).std(dim=0, unbiased=True)
        if not bool(torch.isfinite(next_mu).all().item() and torch.isfinite(next_sigma).all().item()):
            raise RuntimeError(f"non-finite CEM update at round {round_index}")
        metadata["rounds"].append({"round": round_index, "pre_mu": mu.detach().cpu(), "pre_sigma": sigma.detach().cpu(), "elite_indices": selected.detach().cpu(), "post_mu": next_mu.detach().cpu(), "post_sigma": next_sigma.detach().cpu()})
        if round_index in checkpoints:
            if tail_indices is None:
                chosen = actions.detach().cpu()
                targets, teacher_cost = teacher_targets_and_costs(reference, official, row, actions)
                saved_actions[round_index] = chosen
                saved_targets[round_index] = targets.detach().cpu()
                saved_costs[round_index] = teacher_cost.detach().cpu()
            else:
                indices = torch.as_tensor(list(tail_indices[round_index]), dtype=torch.long)
                chosen = actions.detach().cpu().index_select(0, indices)
                target, teacher_cost = teacher_targets_and_costs(reference, official, row, chosen.to(device))
                saved_actions[round_index] = chosen
                saved_targets[round_index] = target.detach().cpu()
                saved_costs[round_index] = teacher_cost.detach().cpu()
        mu, sigma = next_mu, next_sigma
    return saved_actions, metadata, saved_targets, saved_costs


def build_treatment_rows(reference: Any, official: Any, driver: Any, control_rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    cem = freeze["cem_collection"]
    rows: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    innovation_base = int(cem["innovation_seed_base"])
    for ordinal, control in enumerate(control_rows):
        generator = torch.Generator(device="cpu").manual_seed(innovation_base + ordinal)
        innovations = torch.randn((CEM_ITERATIONS, NUM_CANDIDATES, HORIZON, ACTION_DIM), generator=generator)
        saved, metadata, targets, costs = run_student_cem(reference, official, driver, control, innovations, CEM_CHECKPOINTS, TRAIN_TAIL_INDICES)
        tail_actions = torch.cat([saved[round_index] for round_index in CEM_CHECKPOINTS], dim=0)
        tail_targets = torch.cat([targets[round_index] for round_index in CEM_CHECKPOINTS], dim=0)
        tail_costs = torch.cat([costs[round_index] for round_index in CEM_CHECKPOINTS], dim=0)
        if tuple(tail_actions.shape) != (32, HORIZON, ACTION_DIM):
            raise RuntimeError("CEM tail action shape drifted")
        row = dict(control)
        row["future_actions"] = torch.cat([torch.as_tensor(control["future_actions"][:32]), tail_actions], dim=0)
        row["teacher_targets"] = torch.cat([torch.as_tensor(control["teacher_targets"][:32]), tail_targets], dim=0)
        row["teacher_objective"] = torch.cat([torch.as_tensor(control["teacher_objective"][:32]), tail_costs], dim=0)
        row["bank_variant"] = "cem_distribution_mixed_original32_plus_round10_20_30_tail32"
        row["cem_tail_provenance"] = {"driver": "anchor_aligned_step3000_student", "round_counts": {"10": 11, "20": 11, "30": 10}, "teacher_shadow_only": True}
        rows.append(row)
        all_metadata.append({"ordinal": ordinal, "context_id": control.get("context_id"), "rounds": metadata["rounds"]})
        if (ordinal + 1) % 32 == 0:
            print(json.dumps({"phase": "train_cem_collection", "contexts_completed": ordinal + 1, "contexts_total": TRAIN_CONTEXTS}), flush=True)
    torch.save(all_metadata, output / "cem_collection_metadata.pt")
    torch.save(rows, output / "cem_distribution_train_rows.pt")
    return rows, {"contexts": len(rows), "iterations": CEM_ITERATIONS, "samples_per_iteration": NUM_CANDIDATES, "saved_rounds": list(CEM_CHECKPOINTS), "tail_counts": {"round_10": 11, "round_20": 11, "round_30": 10}, "teacher_labelled_tail_candidates": 512 * 32, "teacher_did_not_select_or_update": True, "metadata_path": str((output / "cem_collection_metadata.pt").resolve())}


def continue_train(reference: Any, rows: Sequence[Mapping[str, Any]], initial_state: Mapping[str, Any], official: Any, training_seed: int, schedule_seed: int, updates: int) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) != TRAIN_CONTEXTS:
        raise ValueError("continuation bank must contain 512 contexts")
    student = reference.base.make_student("baseline").to("cuda")
    student.load_state_dict(copy.deepcopy(initial_state), strict=True)
    torch.manual_seed(int(training_seed))
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    schedule = torch.Generator(device="cpu").manual_seed(int(schedule_seed))
    history: list[dict[str, float]] = []
    for step in range(1, updates + 1):
        indices = torch.randperm(len(train_rows), generator=schedule)[:BATCH_CONTEXTS]
        contexts = torch.cat([reference.base._tensor(train_rows[int(index)]["latent_history"]).expand(TRAIN_CANDIDATES, -1, -1) for index in indices], dim=0).to("cuda")
        actions = torch.cat([reference.base._tensor(train_rows[int(index)]["future_actions"]) for index in indices], dim=0).to("cuda")
        targets = torch.cat([reference.base._tensor(train_rows[int(index)]["teacher_targets"]) for index in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        latent_loss, per_horizon = reference.base.recurrent_loss(prediction, targets)
        student_cost = reference.score._student_costs_with_gradient(official, train_rows, indices, prediction)
        teacher_cost = reference.score._effective_teacher_costs(train_rows, indices, "score_distill")
        score_loss = reference.score.score_distill_loss(student_cost, teacher_cost)
        total_loss = latent_loss + float(reference.ema.SCORE_LOSS_WEIGHT) * score_loss
        if not bool(torch.isfinite(total_loss).item() and torch.isfinite(per_horizon).all().item() and torch.isfinite(score_loss).item()):
            raise FloatingPointError(f"non-finite continuation loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        history.append({"step": step, "weighted_latent_mse": float(latent_loss.detach().cpu()), "score_loss": float(score_loss.detach().cpu()), "total_loss": float(total_loss.detach().cpu())})
    last10 = statistics.median(item["weighted_latent_mse"] for item in history[-10:])
    return {"student": student, "state_dict": {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}, "trace": {"first": history[0], "terminal": history[-1], "last10_weighted_latent_mse_median": last10, "last10_to_first_ratio": float(last10 / max(history[0]["weighted_latent_mse"], 1e-12)), "updates": updates}}


def make_fresh_rows(reference: Any, dataset: Path, manifest: Mapping[str, Any], temporal_freeze: Mapping[str, Any], official: Any, fresh_ids: Sequence[int]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    anchor = load_anchor_module()
    anchor.FRESH_SEEDS = FRESH_SEEDS
    anchor.FRESH_SLICE = FRESH_SLICE
    anchor.NUM_CANDIDATES = NUM_CANDIDATES
    rows, metadata = anchor.build_fresh_rows(reference, official, dataset, fresh_ids, temporal_freeze)
    if len(rows) != 24 or int(metadata.get("blocks", -1)) != 48:
        raise ValueError("fresh standard rows/blocks drifted")
    return rows, {**metadata, "selection_slice": "valid[560:568]", "selection_seed": 20300903, "action_prefix_seeds": list(FRESH_SEEDS)}


def collect_fresh_cem_banks(reference: Any, official: Any, driver: Any, rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    import torch

    base_seed = int(freeze["cem_collection"]["eval_innovation_seed_base"])
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for block, seed in enumerate(FRESH_SEEDS):
            generator = torch.Generator(device="cpu").manual_seed(base_seed + row_index * len(FRESH_SEEDS) + block)
            innovations = torch.randn((CEM_ITERATIONS, NUM_CANDIDATES, HORIZON, ACTION_DIM), generator=generator)
            saved, metadata, targets, costs = run_student_cem(reference, official, driver, row, innovations, CEM_CHECKPOINTS)
            for round_index in CEM_CHECKPOINTS:
                records.append({"row_index": row_index, "context_id": row.get("context_id"), "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "round": round_index, "actions": saved[round_index], "teacher_targets": targets[round_index], "teacher_cost": costs[round_index]})
            if (row_index * len(FRESH_SEEDS) + block + 1) % 8 == 0:
                print(json.dumps({"phase": "fresh_cem_collection", "blocks_completed": row_index * len(FRESH_SEEDS) + block + 1, "blocks_total": 48}), flush=True)
    torch.save(records, output / "fresh_cem_trajectory_banks.pt")
    return records


def metric_for_bank(reference: Any, official: Any, student: Any, row: Mapping[str, Any], bank: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    actions = tensor(bank["actions"])
    target = tensor(bank["teacher_targets"])
    teacher_cost = tensor(bank["teacher_cost"])
    latent = tensor(row["latent_history"])
    with torch.no_grad():
        prediction = student(latent.expand(actions.shape[0], -1, -1), actions)
        student_cost = reference.base._official_objective(official, row, prediction)
    teacher_order = torch.topk(teacher_cost, k=ELITE_COUNT, largest=False, sorted=True).indices
    student_order = torch.topk(student_cost, k=ELITE_COUNT, largest=False, sorted=True).indices
    teacher_top = set(int(value) for value in teacher_order.detach().cpu())
    student_top = set(int(value) for value in student_order.detach().cpu())
    teacher_mean = teacher_cost.index_select(0, teacher_order).mean()
    student_teacher_mean = teacher_cost.index_select(0, student_order).mean()
    regret = student_teacher_mean - teacher_mean
    std = teacher_cost.std(unbiased=False).clamp_min(1e-6)
    output: dict[str, Any] = {
        "spearman": float(reference.base._spearman(teacher_cost, student_cost)),
        "top30_overlap": float(len(teacher_top & student_top) / ELITE_COUNT),
        "relative_latent_mse": float(((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).detach().cpu()),
        "teacher_cost_std_population": float(std.detach().cpu()),
        "raw_elite_mean_cost_regret": float(regret.detach().cpu()),
        "standardized_elite_mean_cost_regret": float((regret / std).detach().cpu()),
        "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(student_cost).all().item() and torch.isfinite(teacher_cost).all().item()),
    }
    for shortlist in (60, 120):
        order = torch.topk(student_cost, k=shortlist, largest=False, sorted=True).indices
        captured = len(teacher_top & set(int(value) for value in order.detach().cpu()))
        output[f"recall_at_{shortlist}"] = float(captured / ELITE_COUNT)
        output[f"full_elite_containment_at_{shortlist}"] = bool(captured == ELITE_COUNT)
    return output


def aggregate(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("cannot aggregate empty blocks")
    keys = ("spearman", "top30_overlap", "relative_latent_mse", "recall_at_60", "recall_at_120", "raw_elite_mean_cost_regret", "standardized_elite_mean_cost_regret")
    result: dict[str, Any] = {"blocks": len(blocks), "finite": all(bool(item["finite"]) for item in blocks)}
    for key in keys:
        values = [float(item[key]) for item in blocks]
        result[f"{key}_median"] = float(statistics.median(values))
        result[f"{key}_minimum"] = min(values)
        result[f"{key}_maximum"] = max(values)
    result["full_elite_containment_at_60_rate"] = float(sum(bool(item["full_elite_containment_at_60"]) for item in blocks) / len(blocks))
    result["full_elite_containment_at_120_rate"] = float(sum(bool(item["full_elite_containment_at_120"]) for item in blocks) / len(blocks))
    return result


def grouped_summary(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_round: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_round[str(block.get("round", "standard"))].append(block)
        by_stratum[str(block["stratum"])].append(block)
        by_episode[str(block["episode_id"])].append(block)
    return {"overall": aggregate(blocks), "rounds": {key: aggregate(value) for key, value in by_round.items()}, "strata": {key: aggregate(value) for key, value in by_stratum.items()}, "episodes": {key: aggregate(value) for key, value in sorted(by_episode.items(), key=lambda item: int(item[0]))}, "worst_blocks": sorted(blocks, key=lambda item: (float(item["recall_at_120"]), int(item["episode_id"])))[:5]}


def evaluate_cem_arm(reference: Any, official: Any, student: Any, rows: Sequence[Mapping[str, Any]], banks: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for bank in banks:
        row = rows[int(bank["row_index"])]
        item = metric_for_bank(reference, official, student, row, bank)
        item.update({"label": label, "context_id": row["context_id"], "episode_id": int(bank["episode_id"]), "anchor": int(bank["anchor"]), "stratum": bank["stratum"], "action_prefix_seed": int(bank["action_prefix_seed"]), "round": int(bank["round"]), "pairing_key": f"episode={int(bank['episode_id'])}:anchor={bank['stratum']}:seed={int(bank['action_prefix_seed'])}:round={int(bank['round'])}"})
        blocks.append(item)
    return {"label": label, "per_block": blocks, **grouped_summary(blocks)}


def evaluate_standard_arm(reference: Any, official: Any, student: Any, rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for row in rows:
        for block, seed in enumerate(FRESH_SEEDS):
            bank = {"actions": row["future_actions"][block], "teacher_targets": row["teacher_targets"][block], "teacher_cost": row["teacher_objective"][block]}
            item = metric_for_bank(reference, official, student, row, bank)
            item.update({"label": label, "context_id": row["context_id"], "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "round": "standard", "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}"})
            blocks.append(item)
    return {"label": label, "per_block": blocks, **grouped_summary(blocks)}


def paired_gates(control_cem: Mapping[str, Any], treatment_cem: Mapping[str, Any], control_standard: Mapping[str, Any], treatment_standard: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    def episode_deltas(control: Mapping[str, Any], treatment: Mapping[str, Any], rounds: Sequence[str] | None = None) -> dict[str, float]:
        c_blocks = control["per_block"]
        t_blocks = treatment["per_block"]
        if rounds is not None:
            c_blocks = [item for item in c_blocks if str(item.get("round")) in rounds]
            t_blocks = [item for item in t_blocks if str(item.get("round")) in rounds]
        c = defaultdict(list)
        t = defaultdict(list)
        for item in c_blocks:
            c[str(item["episode_id"])].append(float(item["standardized_elite_mean_cost_regret"]))
        for item in t_blocks:
            t[str(item["episode_id"])].append(float(item["standardized_elite_mean_cost_regret"]))
        return {episode: float(statistics.median(t[episode]) - statistics.median(c[episode])) for episode in sorted(c, key=int)}

    primary = episode_deltas(control_cem, treatment_cem, ["10", "20", "30"])
    per_round = {str(round_index): episode_deltas(control_cem, treatment_cem, [str(round_index)]) for round_index in CEM_CHECKPOINTS}
    forgetting = episode_deltas(control_standard, treatment_standard, ["standard"])
    primary_values = list(primary.values())
    forgetting_values = list(forgetting.values())
    primary_cfg = freeze["gates"]["primary_mechanism"]
    guard_cfg = freeze["gates"]["forgetting_guard"]
    conditions = {
        "primary_median_delta_le_threshold": statistics.median(primary_values) <= float(primary_cfg["treatment_minus_control_median_delta_max"]),
        "primary_strictly_improved_episodes_min": sum(value < 0.0 for value in primary_values) >= int(primary_cfg["strictly_improved_episodes_min"]),
        "each_round_episode_median_delta_le_zero": all(statistics.median(list(values.values())) <= float(primary_cfg["each_round_episode_median_delta_max"]) for values in per_round.values()),
        "forgetting_median_delta_le_threshold": statistics.median(forgetting_values) <= float(guard_cfg["treatment_minus_control_median_delta_max"]),
        "forgetting_no_episode_above_threshold": max(forgetting_values) <= float(guard_cfg["maximum_episode_delta"]),
        "finite": bool(control_cem["overall"]["finite"] and treatment_cem["overall"]["finite"] and control_standard["overall"]["finite"] and treatment_standard["overall"]["finite"]),
    }
    return {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions, "primary_episode_deltas": primary, "primary_median_delta": float(statistics.median(primary_values)), "strictly_improved_episodes": int(sum(value < 0.0 for value in primary_values)), "per_round_episode_deltas": per_round, "per_round_median_delta": {key: float(statistics.median(list(value.values()))) for key, value in per_round.items()}, "forgetting_episode_deltas": forgetting, "forgetting_median_delta": float(statistics.median(forgetting_values)), "forgetting_max_episode_delta": float(max(forgetting_values)), "paired_unit": "episode; primary median over 18 nested CEM blocks per episode"}


def save_checkpoint(path: Path, state: Mapping[str, Any], start_meta: Mapping[str, Any], arm: str) -> None:
    import torch

    torch.save({"state_dict": dict(state), "provenance": {"source": "cem_distribution_distill", "arm": arm, "reconstructed": False, "start_checkpoint": start_meta, "extra_updates": 1000, "training_seed": 20300982, "context_schedule_seed": 20300984}}, path)


def run(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any) -> dict[str, Any]:
    require_compute_node()
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.resolve()
    manifest = load_json(args.manifest_512.resolve())
    temporal_freeze = load_json(args.temporal_freeze.resolve())
    control_rows = load_train_rows(args.anchor_rows)
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.requires_grad_(False)
    start_state, start_meta = load_start_state(reference, args.anchor_checkpoint)
    driver = reference.base.make_student("baseline").to("cuda")
    driver.load_state_dict(start_state, strict=True)
    driver.eval()
    treatment_rows, bank_meta = build_treatment_rows(reference, official, driver, control_rows, freeze, output)
    control_training = continue_train(reference, control_rows, start_state, official, 20300982, 20300984, 1000)
    treatment_training = continue_train(reference, treatment_rows, start_state, official, 20300982, 20300984, 1000)
    control_path = output / "control_step1000.pt"
    treatment_path = output / "treatment_step1000.pt"
    save_checkpoint(control_path, control_training["state_dict"], start_meta, "control")
    save_checkpoint(treatment_path, treatment_training["state_dict"], start_meta, "treatment")
    control_student = control_training["student"].eval()
    treatment_student = treatment_training["student"].eval()
    fresh_ids, selection = select_fresh(dataset, manifest)
    fresh_rows, fresh_meta = make_fresh_rows(reference, dataset, manifest, temporal_freeze, official, fresh_ids)
    fresh_banks = collect_fresh_cem_banks(reference, official, driver, fresh_rows, freeze, output)
    control_cem = evaluate_cem_arm(reference, official, control_student, fresh_rows, fresh_banks, "control")
    treatment_cem = evaluate_cem_arm(reference, official, treatment_student, fresh_rows, fresh_banks, "treatment")
    control_standard = evaluate_standard_arm(reference, official, control_student, fresh_rows, "control_standard_current_anchor")
    treatment_standard = evaluate_standard_arm(reference, official, treatment_student, fresh_rows, "treatment_standard_current_anchor")
    gates = paired_gates(control_cem, treatment_cem, control_standard, treatment_standard, freeze)
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_contract": validate_interface(reference, args.interface_probe.resolve()),
        "start_checkpoint": start_meta,
        "source": {"dataset": str(dataset), "manifest_512": str(args.manifest_512.resolve()), "anchor_rows": str(args.anchor_rows.resolve()), "temporal_freeze": str(args.temporal_freeze.resolve())},
        "bank_construction": bank_meta,
        "training": {"control": {"checkpoint": str(control_path), "trace": control_training["trace"], "rows": "anchor_aligned 64-candidate bank"}, "treatment": {"checkpoint": str(treatment_path), "trace": treatment_training["trace"], "rows": str((output / "cem_distribution_train_rows.pt").resolve())}},
        "fresh_selection": selection,
        "fresh_evaluation": fresh_meta,
        "cem_trajectory_evaluation": {"control": control_cem, "treatment": treatment_cem},
        "standard_current_anchor_evaluation": {"control": control_standard, "treatment": treatment_standard},
        "gates": gates,
        "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Fixed-observation candidate-ranking evidence only; CEM trajectory banks are frozen-start student-driven actions with teacher shadow labels, not student on-policy environment data. No planner deployment or closed-loop claim.",
    }
    write_json(output / "cem_distribution_distill_summary.json", summary)
    return summary


def preflight(args: argparse.Namespace, freeze: Mapping[str, Any]) -> dict[str, Any]:
    reference = load_reference()
    interface = validate_interface(reference, args.interface_probe.resolve())
    value = {"schema": SCHEMA, "status": "PASS", "model_work_started": False, "reference_import": "PASS", "interface": interface, "training_contexts": TRAIN_CONTEXTS, "training_candidates": TRAIN_CANDIDATES, "extra_updates": 1000, "cem": {"iterations": CEM_ITERATIONS, "samples": NUM_CANDIDATES, "topk": ELITE_COUNT, "saved_rounds": list(CEM_CHECKPOINTS)}, "fresh_selection": "valid[560:568]", "selection_seed": 20300903, "fresh_blocks": 48, "pbs_compute_only": True}
    write_json(args.output.resolve() / "preflight_status.json", value)
    return value


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "training_contexts": TRAIN_CONTEXTS, "extra_updates": 1000, "cem_iterations": CEM_ITERATIONS, "fresh_selection": "valid[560:568]", "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output.resolve() / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    freeze = validate_freeze(load_json(args.freeze.resolve()))
    validate_protocol(args.protocol.resolve())
    if args.mode == "preflight":
        print(json.dumps(preflight(args, freeze), ensure_ascii=False))
        return 0
    reference = load_reference()
    summary = run(args, freeze, reference)
    print(json.dumps({"status": summary["status"], "gates": summary["gates"], "output": str((args.output / "cem_distribution_distill_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
