#!/usr/bin/env python3
"""Single-trajectory EMA confirmatory predictor-only LeWM PushT experiment."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(TRANSFER / "state-coverage"))
sys.path.insert(0, str(TRANSFER / "score-distill"))
sys.path.insert(0, str(TRANSFER / "dense-rank"))

import run_lewm_state_coverage as coverage  # noqa: E402
import run_lewm_score_distill as score  # noqa: E402

base = score.base
dense = score.dense

TRAIN_CANDIDATES = 64
BATCH_CONTEXTS = 8
SCORE_TOP_COUNT = 12
SCORE_TOP_WEIGHT = 2.0
SCORE_OTHER_WEIGHT = 1.0
SCORE_STD_FLOOR = 1e-6
SCORE_SMOOTH_L1_BETA = 1.0
SCORE_LOSS_WEIGHT = 0.1
EMA_DECAY = 0.999
SNAPSHOT_STEPS = (500, 1000, 1500, 3000)
FRESH_EPISODE_COUNT = 8
FRESH_ANCHORS = ("early", "middle", "late")
FRESH_SEEDS = (20300917, 20300918)
HORIZON_PRIMITIVES = base.HORIZON * (base.ACTION_DIM // 2)


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
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--phase2-freeze", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
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


def close(actual: Any, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} drifted: {actual} != {expected}")


def load_freeze(path: Path, phase2_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = load_json(path.resolve())
    if freeze.get("schema") != "lewm-recurrent-student.ema-temporal-confirm-freeze" or freeze.get("status") != "frozen":
        raise ValueError("EMA temporal confirm freeze is not frozen or has an unexpected schema")
    base_freeze = base.load_freeze((path.resolve().parent / str(freeze["base_freeze"])).resolve())
    phase2 = load_json(phase2_path.resolve())
    if phase2.get("schema") != "lewm-recurrent-student.state-context-coverage-freeze" or phase2.get("status") != "frozen":
        raise ValueError("unexpected Phase 2 freeze")
    design = freeze["design"]
    if design.get("training_arms") != ["single_trajectory_online_control_and_ema_primary"] or int(design["training_rows"]) != 512 or int(design["updates"]) != 3000:
        raise ValueError("single-trajectory 512/3000 design drifted")
    training = freeze["training"]
    for key, expected in {"batch_contexts": BATCH_CONTEXTS, "train_candidates": TRAIN_CANDIDATES, "score_top_count": SCORE_TOP_COUNT, "initialization_seed": 20300901, "training_seed": 20300902, "context_schedule_seed": 20300904}.items():
        if int(training[key]) != expected:
            raise ValueError(f"training {key} drifted")
    for key, expected in {"score_top_weight": SCORE_TOP_WEIGHT, "score_other_weight": SCORE_OTHER_WEIGHT, "score_std_floor": SCORE_STD_FLOOR, "score_smooth_l1_beta": SCORE_SMOOTH_L1_BETA, "score_loss_weight": SCORE_LOSS_WEIGHT, "ema_decay": EMA_DECAY}.items():
        close(training[key], expected, key)
    if [float(x) for x in training["horizon_weights"]] != list(base.HORIZON_WEIGHTS):
        raise ValueError("horizon weights drifted")
    fresh = freeze["fresh_evaluation"]
    if fresh["action_prefix_seeds"] != list(FRESH_SEEDS) or set(fresh["anchors"]) != {"early", "middle", "late", "validity"}:
        raise ValueError("fresh anchor/seed contract drifted")
    if set(FRESH_SEEDS) & {20300907, 20300908}:
        raise ValueError("fresh seeds overlap prior heldout seeds")
    if freeze["scope_boundary"]["official_cem"] != "NOT_RUN_BY_SCOPE" or freeze["scope_boundary"]["planner_viability"] != "NOT_RUN_BY_SCOPE" or freeze["scope_boundary"]["closed_loop"] != "NOT_RUN_BY_SCOPE":
        raise ValueError("out-of-scope planner boundary drifted")
    absolute = freeze["gates"]["inherited_absolute_predictor"]
    for key, expected in {"median_spearman_min": 0.95, "minimum_spearman_min": 0.80, "median_top30_min": 0.75, "minimum_top30_min": 0.50, "median_relative_latent_mse_max": 0.25, "positive_spearman_blocks_min": 36, "positive_top30_blocks_min": 36, "latency_reduction_min": 0.20}.items():
        close(absolute[key], expected, key)
    strata = freeze["gates"]["stratum_protection"]
    for key, expected in {"median_spearman_min": 0.95, "median_top30_min": 0.75, "positive_spearman_blocks_min": 12, "positive_top30_blocks_min": 12}.items():
        close(strata[key], expected, f"stratum.{key}")
    mechanism = freeze["gates"]["ema_mechanism_secondary"]
    if float(mechanism["spearman_episode_median_delta_min"]) != 0.0 or float(mechanism["top30_episode_median_delta_min"]) != 0.0 or int(mechanism["joint_nonworsening_improvement_min_episodes"]) != 5:
        raise ValueError("EMA secondary gate drifted")
    if phase2.get("context_manifest", {}).get("heldout_manifest_reused_verbatim") is False:
        raise ValueError("Phase 2 manifest reuse contract is false")
    return freeze, base_freeze, phase2


def phase2_authority(phase2_path: Path) -> dict[str, Any]:
    phase2 = load_json(phase2_path.resolve())
    spec = phase2.get("context_manifest", {})
    if int(spec.get("new_train_target", -1)) != 512 or int(spec.get("heldout_target", -1)) != 8:
        raise ValueError("Phase 2 is not the formal 512/8 context contract")
    if int(phase2.get("training", {}).get("train_candidates", -1)) != TRAIN_CANDIDATES:
        raise ValueError("Phase 2 train candidate count drifted")
    return phase2


def select_fresh_episodes(dataset_path: Path, selection_seed: int, old_manifest: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        lengths = [int(x) for x in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON_PRIMITIVES + 1]
    random.Random(int(selection_seed)).shuffle(valid)
    old_heldout = [int(row["episode_id"]) for row in old_manifest["splits"]["heldout"]]
    old_train = [int(row["episode_id"]) for row in old_manifest["splits"]["train"]]
    if valid[: len(old_heldout)] != old_heldout or valid[len(old_heldout) : len(old_heldout) + len(old_train)] != old_train:
        raise ValueError("selection prefix does not reproduce the frozen Phase 2 manifest")
    fresh = valid[len(old_heldout) + len(old_train) : len(old_heldout) + len(old_train) + FRESH_EPISODE_COUNT]
    if len(fresh) != FRESH_EPISODE_COUNT or set(fresh) & (set(old_heldout) | set(old_train)):
        raise ValueError("fresh selection overlaps old heldout/train episodes")
    return fresh, {"selection_seed": int(selection_seed), "valid_count": len(valid), "old_heldout_episode_ids": old_heldout, "old_train_episode_count": len(old_train), "fresh_episode_ids": fresh, "selection_slice": "valid[520:528]", "result_dependent_selection": False}


def anchor_spec(length: int) -> dict[str, int]:
    late = int(length) - HORIZON_PRIMITIVES - 1
    if late < 0:
        raise ValueError(f"episode length {length} cannot hold current+25 actions+goal")
    anchors = {"early": 0, "middle": late // 2, "late": late}
    for name, anchor in anchors.items():
        goal = anchor + HORIZON_PRIMITIVES
        if not (0 <= anchor <= length - 1 and anchor + HORIZON_PRIMITIVES - 1 < length and goal < length):
            raise ValueError(f"{name} anchor violates episode bounds")
    return anchors


def build_fresh_rows(model: Any, dataset_path: Path, episode_ids: Sequence[int], freeze: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    low = torch.tensor(freeze["evaluation"]["official_action_low"], dtype=torch.float32).reshape(1, 1, 2)
    high = torch.tensor(freeze["evaluation"]["official_action_high"], dtype=torch.float32).reshape(1, 1, 2)
    gaussian_std = float(freeze["evaluation"]["gaussian_std"])
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    with base.HDF5EpisodeSliceReader(dataset_path) as reader:
        for episode_id in episode_ids:
            episode = reader.episode_slice(int(episode_id))
            anchors = anchor_spec(int(episode["length"]))
            for stratum in FRESH_ANCHORS:
                anchor = anchors[stratum]
                manifest_row = {"episode_id": int(episode_id), "history_steps": [anchor], "history_action_starts": [anchor], "future_action_start": anchor, "goal_step_offset": anchor + HORIZON_PRIMITIVES}
                raw = reader.read_manifest_row(manifest_row)
                current = base._normalise_pixels(raw["pixels"][:1]).unsqueeze(1).to("cuda")
                goal_pixels = base._normalise_pixels(episode["pixels"][anchor + HORIZON_PRIMITIVES : anchor + HORIZON_PRIMITIVES + 1]).unsqueeze(1).to("cuda")
                logged = torch.from_numpy(base.pack_raw_actions(raw["future_actions_raw"]))
                encode_action = logged[:1].unsqueeze(0).to("cuda")
                with torch.no_grad():
                    latent = model.encode({"pixels": current, "action": encode_action})["emb"]
                    goal_emb = model.encode({"pixels": goal_pixels})["emb"]
                if tuple(latent.shape) != (1, 1, base.LATENT_DIM):
                    raise RuntimeError(f"fresh latent shape drifted: {tuple(latent.shape)}")
                banks, targets, objectives = [], [], []
                for seed in FRESH_SEEDS:
                    generator = torch.Generator(device="cpu").manual_seed(int(seed))
                    actions = base._slate_actions(logged, generator, low, high, gaussian_std, 300).to("cuda")
                    context = latent.expand(300, -1, -1)
                    with torch.no_grad():
                        target = base.official_teacher_targets(model, context, actions)
                        objective = base._official_objective(model, {"latent_history": latent, "goal_emb": goal_emb}, target)
                    banks.append(actions.detach().cpu())
                    targets.append(target.detach().cpu())
                    objectives.append(objective.detach().cpu())
                context_id = f"fresh_ep{int(episode_id):04d}_{stratum}"
                rows.append({"split": "fresh_eval", "context_id": context_id, "episode_id": int(episode_id), "anchor": int(anchor), "stratum": stratum, "future_action_start": int(anchor), "goal_step": int(anchor + HORIZON_PRIMITIVES), "latent_history": latent.detach().cpu(), "action_history": encode_action.detach().cpu(), "future_actions": torch.stack(banks), "teacher_targets": torch.stack(targets), "teacher_objective": torch.stack(objectives), "goal_emb": goal_emb.detach().cpu()})
                metadata.append({"context_id": context_id, "episode_id": int(episode_id), "stratum": stratum, "anchor": int(anchor), "goal_step": int(anchor + HORIZON_PRIMITIVES), "candidate_seeds": list(FRESH_SEEDS), "candidate_count_per_block": 300})
    if len(rows) != 24:
        raise ValueError(f"fresh row count is {len(rows)}, expected 24")
    return rows, {"contexts": len(rows), "episodes": len(episode_ids), "blocks": len(rows) * len(FRESH_SEEDS), "candidates_per_block": 300, "rows_generated_on_compute_node": True, "rows_returned": False, "contexts_detail": metadata}


def update_ema(ema_state: dict[str, Any], online: Any) -> None:
    import torch

    for name, value in online.state_dict().items():
        current = value.detach().cpu()
        if name not in ema_state or not torch.is_floating_point(current):
            ema_state[name] = current.clone()
        else:
            ema_state[name].mul_(EMA_DECAY).add_(current, alpha=1.0 - EMA_DECAY)


def train_once(rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any], initial_state: Mapping[str, Any], official_model: Any) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) != 512:
        raise ValueError("training bank must contain exactly 512 contexts")
    torch.manual_seed(int(settings["initialization_seed"]))
    online = base.make_student("baseline").to("cuda")
    online.load_state_dict(copy.deepcopy(initial_state), strict=True)
    torch.manual_seed(int(settings["training_seed"]))
    optimizer = torch.optim.AdamW(online.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    schedule = torch.Generator(device="cpu").manual_seed(int(settings["context_schedule_seed"]))
    ema_state = {key: value.detach().cpu().clone() for key, value in initial_state.items()}
    online_snapshots: dict[str, Any] = {}
    ema_snapshots: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for step in range(1, int(settings["steps"]) + 1):
        indices = torch.randperm(len(train_rows), generator=schedule)[:BATCH_CONTEXTS]
        contexts = torch.cat([base._tensor(train_rows[int(i)]["latent_history"]).expand(TRAIN_CANDIDATES, -1, -1) for i in indices], dim=0).to("cuda")
        actions = torch.cat([base._tensor(train_rows[int(i)]["future_actions"]) for i in indices], dim=0).to("cuda")
        targets = torch.cat([base._tensor(train_rows[int(i)]["teacher_targets"]) for i in indices], dim=0).to("cuda")
        prediction = online(contexts, actions)
        latent_loss, per_horizon = base.recurrent_loss(prediction, targets)
        student_cost = score._student_costs_with_gradient(official_model, train_rows, indices, prediction)
        teacher_cost = score._effective_teacher_costs(train_rows, indices, "score_distill")
        objective_loss = score.score_distill_loss(student_cost, teacher_cost)
        total_loss = latent_loss + SCORE_LOSS_WEIGHT * objective_loss
        finite = bool(torch.isfinite(total_loss).item() and torch.isfinite(per_horizon).all().item() and torch.isfinite(objective_loss).item())
        if not finite:
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        update_ema(ema_state, online)
        history.append({"step": step, "weighted_latent_mse": float(latent_loss.detach().cpu()), "score_loss": float(objective_loss.detach().cpu()), "total_loss": float(total_loss.detach().cpu()), "per_horizon_mse": [float(x) for x in per_horizon.detach().cpu()], "finite": finite})
        if step in SNAPSHOT_STEPS:
            online_snapshots[f"step_{step}"] = {key: value.detach().cpu().clone() for key, value in online.state_dict().items()}
            ema_snapshots[f"step_{step}"] = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
    first = history[0]["weighted_latent_mse"]
    last10 = statistics.median(item["weighted_latent_mse"] for item in history[-10:])
    return {"online_snapshots": online_snapshots, "ema_snapshots": ema_snapshots, "parameter_count": int(sum(parameter.numel() for parameter in online.parameters())), "per_step": history, "last10_to_first_ratio": float(last10 / max(first, 1e-12)), "ema_isolation": True, "ema_decay": EMA_DECAY, "same_trajectory": True}


def aggregate_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spearman = [float(x["spearman"]) for x in blocks]
    top30 = [float(x["top30_overlap"]) for x in blocks]
    rel = [float(x["relative_latent_mse"]) for x in blocks]
    return {"blocks": len(blocks), "spearman_median": float(statistics.median(spearman)), "spearman_minimum": min(spearman), "top30_median": float(statistics.median(top30)), "top30_minimum": min(top30), "relative_latent_mse_median": float(statistics.median(rel)), "positive_spearman_blocks": sum(value > 0.0 for value in spearman), "positive_top30_blocks": sum(value > 0.0 for value in top30), "finite": all(bool(x["finite"]) for x in blocks)}


def evaluate_state(model: Any, snapshots: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    import torch

    for step in SNAPSHOT_STEPS:
        name = f"step_{step}"
        student = base.make_student("baseline").to("cuda")
        student.load_state_dict(snapshots[name], strict=True)
        student.eval()
        blocks: list[dict[str, Any]] = []
        for row in rows:
            for block, seed in enumerate(FRESH_SEEDS):
                item = base._stage_a_metrics_for_block(model, student, row, block)
                item.update({"label": label, "context_id": row["context_id"], "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}"})
                blocks.append(item)
        by_stratum = {stratum: aggregate_blocks([item for item in blocks if item["stratum"] == stratum]) for stratum in FRESH_ANCHORS}
        results[name] = {"label": label, "per_block": blocks, "overall": aggregate_blocks(blocks), "strata": by_stratum, "episodes": episode_medians(blocks)}
        del student
        torch.cuda.empty_cache()
    return results


def episode_medians(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in blocks:
        grouped[int(item["episode_id"])].append(item)
    output: dict[str, Any] = {}
    for episode, items in sorted(grouped.items()):
        if len(items) != 6:
            raise ValueError(f"episode {episode} does not have six nested blocks")
        output[str(episode)] = {"blocks": 6, "spearman_median": float(statistics.median(float(x["spearman"]) for x in items)), "top30_median": float(statistics.median(float(x["top30_overlap"]) for x in items)), "relative_latent_mse_median": float(statistics.median(float(x["relative_latent_mse"]) for x in items))}
    return output


def causality_and_latency(model: Any, snapshots: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    row = rows[0]
    output_causality: dict[str, Any] = {}
    output_latency: dict[str, Any] = {}
    for label, states in (("online", snapshots["online"]), ("ema", snapshots["ema"])):
        student = base.make_student("baseline").to("cuda")
        student.load_state_dict(states["step_3000"], strict=True)
        student.eval()
        output_causality[label] = base.causality_test(student, row, FRESH_SEEDS[0], float(freeze["evaluation"]["causality_tolerance"]))
        output_latency[label] = base.predictor_latency(model, student, row, warmup=int(freeze["evaluation"]["predictor_latency_warmup"]), repeats=int(freeze["evaluation"]["predictor_latency_repeats"]))
        del student
        torch.cuda.empty_cache()
    return output_causality, output_latency


def absolute_gate(freeze: Mapping[str, Any], evaluation: Mapping[str, Any], training: Mapping[str, Any], causality: Mapping[str, Any], latency: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["inherited_absolute_predictor"]
    metrics = evaluation["overall"]
    convergence = float(training["last10_to_first_ratio"]) <= float(base.load_freeze((HERE.parent / "LEWM_RECURRENT_STUDENT_FREEZE.json").resolve())["gates"]["convergence"]["last10_to_first_training_mse_ratio_max"])
    causal_pass = all(item["passed"] for item in causality.values())
    integrity = convergence and metrics["finite"] and bool(training["ema_isolation"]) and bool(training["same_trajectory"])
    fidelity_conditions = {"median_spearman_min": metrics["spearman_median"] >= float(gate["median_spearman_min"]), "minimum_spearman_min": metrics["spearman_minimum"] >= float(gate["minimum_spearman_min"]), "median_top30_min": metrics["top30_median"] >= float(gate["median_top30_min"]), "minimum_top30_min": metrics["top30_minimum"] >= float(gate["minimum_top30_min"]), "median_relative_latent_mse_max": metrics["relative_latent_mse_median"] <= float(gate["median_relative_latent_mse_max"]), "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(gate["positive_spearman_blocks_min"]), "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(gate["positive_top30_blocks_min"])}
    fidelity = all(fidelity_conditions.values())
    latency_ok = float(latency["reduction"]) >= float(gate["latency_reduction_min"])
    passed = bool(integrity and causal_pass and fidelity and latency_ok)
    return {"status": "GO" if passed else "NO-GO", "integrity_and_convergence": {"status": "PASS" if integrity else "FAIL", "last10_to_first_ratio": training["last10_to_first_ratio"], "ema_isolation": training["ema_isolation"], "same_trajectory": training["same_trajectory"]}, "causality": {"status": "PASS" if causal_pass else "FAIL", "per_prefix": dict(causality)}, "predictor_fidelity_and_ranking": {"status": "PASS" if fidelity else "FAIL", "metrics": {key: metrics[key] for key in ("spearman_median", "spearman_minimum", "top30_median", "top30_minimum", "relative_latent_mse_median", "positive_spearman_blocks", "positive_top30_blocks")}, "conditions": fidelity_conditions}, "predictor_latency": {"status": "PASS" if latency_ok else "FAIL", "metrics": latency}, "full_cem_viability": "NOT_RUN_BY_SCOPE"}


def stratum_gate(freeze: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["stratum_protection"]
    conditions = {}
    for stratum, metrics in evaluation["strata"].items():
        conditions[stratum] = {"median_spearman_min": metrics["spearman_median"] >= float(gate["median_spearman_min"]), "median_top30_min": metrics["top30_median"] >= float(gate["median_top30_min"]), "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(gate["positive_spearman_blocks_min"]), "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(gate["positive_top30_blocks_min"])}
    return {"status": "GO" if all(all(item.values()) for item in conditions.values()) else "NO-GO", "conditions": conditions, "minimum_thresholds_repeated": False}


def paired_mechanism_gate(freeze: Mapping[str, Any], online: Mapping[str, Any], ema: Mapping[str, Any]) -> dict[str, Any]:
    online_eps, ema_eps = online["episodes"], ema["episodes"]
    deltas = []
    for episode in sorted(online_eps, key=int):
        ds = float(ema_eps[episode]["spearman_median"]) - float(online_eps[episode]["spearman_median"])
        dt = float(ema_eps[episode]["top30_median"]) - float(online_eps[episode]["top30_median"])
        deltas.append({"episode_id": int(episode), "spearman_delta_ema_minus_online": ds, "top30_delta_ema_minus_online": dt, "joint_improvement_or_nonworsening": bool((ds > 0 and dt >= 0) or (dt > 0 and ds >= 0))})
    med_s = statistics.median(item["spearman_delta_ema_minus_online"] for item in deltas)
    med_t = statistics.median(item["top30_delta_ema_minus_online"] for item in deltas)
    joint = sum(bool(item["joint_improvement_or_nonworsening"]) for item in deltas)
    minimum = freeze["gates"]["ema_mechanism_secondary"]
    conditions = {"spearman_episode_median_delta_min": med_s >= float(minimum["spearman_episode_median_delta_min"]), "top30_episode_median_delta_min": med_t >= float(minimum["top30_episode_median_delta_min"]), "joint_nonworsening_improvement_min_episodes": joint >= int(minimum["joint_nonworsening_improvement_min_episodes"])}
    return {"status": "GO" if all(conditions.values()) else "NO-GO", "paired_unit": "episode after median over six blocks", "median_spearman_delta": float(med_s), "median_top30_delta": float(med_t), "joint_count": joint, "conditions": conditions, "per_episode": deltas, "cannot_replace_absolute_gate": True}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], base_freeze: Mapping[str, Any], phase2: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    base.require_compute_node()
    import torch

    dataset = (args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve()
    manifest_path = args.manifest_512.resolve()
    rows_path = args.prepared_rows_512.resolve()
    old_manifest = load_json(manifest_path)
    phase_freeze = copy.deepcopy(dict(base_freeze))
    phase_freeze["shared_data_and_schedule"]["context_manifest"]["train_context_target"] = 512
    manifest = base.validate_manifest(manifest_path, phase_freeze)
    rows = torch.load(rows_path, map_location="cpu", weights_only=False)
    row_meta = dense.validate_dense_rows(rows, phase_freeze)
    if row_meta.get("train_contexts") != 512:
        raise ValueError("formal training bank is not 512 contexts")
    reference = load_json(args.reference_summary.resolve())
    if reference.get("status") != "PREDICTOR_LEVEL_COMPLETE" or reference.get("arms", {}).get("512x3000") is None:
        raise ValueError("historical 512x3000 reference is missing")
    if reference.get("source", {}).get("lewm_commit") != base_freeze["evidence_boundary"]["source_commit"]:
        raise ValueError("historical reference source commit drifted")
    official_model = base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    fresh_ids, selection = select_fresh_episodes(dataset, int(freeze["fresh_evaluation"]["selection_seed"]), old_manifest)
    fresh_rows, fresh_meta = build_fresh_rows(official_model, dataset, fresh_ids, freeze)
    torch.manual_seed(int(freeze["training"]["initialization_seed"]))
    template = base.make_student("baseline").to("cuda")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    torch.cuda.empty_cache()
    settings = {"steps": 3000, "initialization_seed": int(freeze["training"]["initialization_seed"]), "training_seed": int(freeze["training"]["training_seed"]), "context_schedule_seed": int(freeze["training"]["context_schedule_seed"])}
    training = train_once(rows, settings, initial_state, official_model)
    online_eval = evaluate_state(official_model, training["online_snapshots"], fresh_rows, "online")
    ema_eval = evaluate_state(official_model, training["ema_snapshots"], fresh_rows, "ema")
    snapshots = {"online": training["online_snapshots"], "ema": training["ema_snapshots"]}
    causality, latency = causality_and_latency(official_model, snapshots, fresh_rows, freeze)
    online_gate = absolute_gate(freeze, online_eval["step_3000"], training, causality["online"], latency["online"])
    ema_absolute = absolute_gate(freeze, ema_eval["step_3000"], training, causality["ema"], latency["ema"])
    ema_strata = stratum_gate(freeze, ema_eval["step_3000"])
    primary_conditions = {"ema_absolute_predictor_gate": ema_absolute["status"] == "GO", "ema_temporal_stratum_gate": ema_strata["status"] == "GO"}
    mechanism = paired_mechanism_gate(freeze, online_eval["step_3000"], ema_eval["step_3000"])
    summary = {"schema": "lewm-recurrent-student.ema-temporal-confirm", "schema_version": 1, "status": "PREDICTOR_LEVEL_COMPLETE", "freeze": str(args.freeze.resolve()), "protocol": str(args.protocol.resolve()), "interface_probe": str(args.interface_probe.resolve()), "interface_contract": dict(contract.__dict__), "source": {"lewm_commit": base_freeze["evidence_boundary"]["source_commit"], "stable_worldmodel_cem_commit": base_freeze["evidence_boundary"]["stable_worldmodel_cem_commit"], "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()), "dataset": str(dataset)}, "historical_reference": {"formal_job": "24926383.pbs101", "summary": str(args.reference_summary.resolve()), "retrain": False, "terminal_metrics": reference["arms"]["512x3000"]["terminal_metrics"]}, "old_training_bank": {"manifest": str(manifest_path), "prepared_rows": str(rows_path), "manifest_counts": {"train": len(manifest["splits"]["train"]), "heldout": len(manifest["splits"]["heldout"])}, "prepared_row_counts": row_meta, "reused_verbatim": True, "regenerated": False, "returned": False}, "fresh_selection": selection, "fresh_evaluation": fresh_meta, "shared_training_contract": {"architecture": "LeWMCompactRecurrentTransitionStudent", "hidden_dim": base.HIDDEN_DIM, "attention": False, "conditioner": False, "goal_input": False, "teacher_forcing": False, "objective": "horizon-weighted free-running latent MSE + 0.1 * context-normalized teacher-score SmoothL1 mean", "optimizer": freeze["training"]["optimizer"], "updates": 3000, "batch_contexts": 8, "train_candidates": 64, "initialization_state_shared": True, "context_schedule_shared": True, "same_trajectory_online_ema": True}, "pairing": {"initialization_seed": settings["initialization_seed"], "training_seed": settings["training_seed"], "context_schedule_seed": settings["context_schedule_seed"], "fresh_action_prefix_seeds": list(FRESH_SEEDS), "same_initial_state": True, "same_training_trajectory": True, "same_fresh_rows_and_candidate_bank": True, "candidate_is_not_an_independent_statistical_unit": True}, "training": {"parameter_count": training["parameter_count"], "ema_decay": EMA_DECAY, "ema_isolation": training["ema_isolation"], "same_trajectory": training["same_trajectory"], "last10_to_first_ratio": training["last10_to_first_ratio"], "per_step": training["per_step"], "snapshot_steps": list(SNAPSHOT_STEPS), "early_stop": False, "online_terminal": "step_3000", "ema_terminal": "step_3000"}, "arms": {"online": online_eval, "ema": ema_eval}, "terminal_metrics": {"online": online_eval["step_3000"]["overall"], "ema": ema_eval["step_3000"]["overall"]}, "causality": causality, "predictor_latency": latency, "absolute_predictor_gates": {"online_control": online_gate, "ema_primary": ema_absolute}, "temporal_stratum_gate": ema_strata, "paired_deltas": mechanism, "primary_gate": {"status": "GO" if all(primary_conditions.values()) else "NO-GO", "arm": "ema", "conditions": primary_conditions, "absolute_predictor_gate": ema_absolute, "temporal_stratum_gate": ema_strata, "secondary_ema_mechanism_gate": mechanism}, "scope_checks": {"formal_512_training_bank_reused": "PASS", "fresh_episode_selection": "PASS", "fresh_rows_generated_compute_only": "PASS", "same_trajectory_online_ema": "PASS", "ema_isolation": "PASS", "nested_episode_aggregation": "PASS", "no_result_dependent_selection": "PASS"}, "gpu_telemetry": {"source": "job.log nvidia-smi 5-second samples", "required": True}, "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}, "claim_boundary": "EMA temporal multi-anchor confirmatory evidence is predictor-level only; no official CEM, planner viability, closed-loop PushT, encoder speedup, or native deployment claim."}
    write_json(args.output.resolve() / "lewm_ema_temporal_confirm_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze, base_freeze, phase2 = load_freeze(args.freeze, args.phase2_freeze)
    if args.mode == "status":
        value = {"schema": "lewm-recurrent-student.ema-temporal-confirm", "status": "READY", "training_rows": 512, "fresh_episodes": 8, "fresh_contexts": 24, "fresh_blocks": 48, "action_prefix_seeds": list(FRESH_SEEDS), "terminal_snapshot": "step_3000", "primary_arm": "ema", "official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    for path in (args.interface_probe, args.manifest_512, args.prepared_rows_512, args.reference_summary, args.phase2_freeze, args.dataset or (args.stablewm_home / "pusht_expert_train.h5")):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = base.load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, base_freeze, phase2_authority(args.phase2_freeze), contract)
    print(json.dumps({"status": summary["status"], "primary_gate": summary["primary_gate"]["status"], "output": str((args.output / "lewm_ema_temporal_confirm_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
