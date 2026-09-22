#!/usr/bin/env python3
"""LeWM PushT Phase 3 tail-robust EMA score-distillation experiment.

This runner deliberately consumes the frozen 512-context rows produced by the
Phase 2 state-coverage job.  It has no data-generation or checkpoint-export
path: the two new arms differ only in the context-level score aggregation and
both are evaluated on their terminal EMA snapshot.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

TAIL_DIR = Path(__file__).resolve().parent
TRANSFER_DIR = TAIL_DIR.parent
sys.path.insert(0, str(TRANSFER_DIR / "state-coverage"))
sys.path.insert(0, str(TRANSFER_DIR / "score-distill"))
sys.path.insert(0, str(TRANSFER_DIR / "dense-rank"))

import run_lewm_state_coverage as coverage  # noqa: E402
import run_lewm_score_distill as score  # noqa: E402

base = score.base

TRAIN_CANDIDATES = 64
BATCH_CONTEXTS = 8
TOP_COUNT = 12
TAIL_TOP_COUNT = 2
SCORE_TOP_WEIGHT = 2.0
SCORE_OTHER_WEIGHT = 1.0
SCORE_STD_FLOOR = 1e-6
SCORE_SMOOTH_L1_BETA = 1.0
SCORE_LOSS_WEIGHT = 0.1
EMA_DECAY = 0.999
SNAPSHOT_STEPS = (500, 1000, 1500, 3000)
ARMS = ("ema_score", "ema_tail_score")


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


def _close(actual: Any, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} drifted: {actual} != {expected}")


def _phase2_authority(path: Path) -> dict[str, Any]:
    """Validate the Phase 2 source contract without loading data or models."""
    phase2 = load_json(path.resolve())
    if phase2.get("schema") != "lewm-recurrent-student.state-context-coverage-freeze":
        raise ValueError("unexpected Phase 2 state-coverage freeze")
    if phase2.get("status") != "frozen":
        raise ValueError("Phase 2 freeze is not frozen")
    if phase2.get("experiment_name") != "lewm_score_distill_state_context_coverage_phase2":
        raise ValueError("unexpected Phase 2 experiment name")
    manifest = phase2.get("context_manifest", {})
    if manifest.get("new_train_target") != 512 or manifest.get("heldout_target") != 8:
        raise ValueError("Phase 2 512/8 context contract drifted")
    training = phase2.get("training", {})
    expected = {
        "train_candidates": TRAIN_CANDIDATES,
        "score_top_count": TOP_COUNT,
        "score_top_weight": SCORE_TOP_WEIGHT,
        "score_other_weight": SCORE_OTHER_WEIGHT,
        "batch_contexts": BATCH_CONTEXTS,
        "same_context_schedule_seed": 20300904,
        "initialization_seed": 20300901,
        "training_seed": 20300902,
    }
    for key, value in expected.items():
        actual = training.get(key)
        if isinstance(value, float):
            _close(actual, value, f"Phase 2 {key}")
        elif int(actual) != value:
            raise ValueError(f"Phase 2 {key} drifted: {actual} != {value}")
    if [float(x) for x in training.get("horizon_weights", [])] != list(base.HORIZON_WEIGHTS):
        raise ValueError("Phase 2 horizon weights drifted")
    return phase2


def load_freeze(path: Path, phase2_freeze: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = load_json(path.resolve())
    if freeze.get("schema") != "lewm-recurrent-student.tail-robust-freeze":
        raise ValueError("unexpected tail-robust freeze schema")
    if freeze.get("status") != "frozen":
        raise ValueError("tail-robust freeze is not frozen")

    base_freeze_path = (path.resolve().parent / str(freeze["base_freeze"])).resolve()
    base_freeze = base.load_freeze(base_freeze_path)
    phase2 = _phase2_authority(phase2_freeze)
    design = freeze["design"]
    if design.get("new_arms") != list(ARMS) or int(design["train_contexts"]) != 512 or int(design["updates"]) != 3000:
        raise ValueError("tail-robust arm/count/update contract drifted")
    training = freeze["training"]
    if int(training["initialization_seed"]) != 20300901 or int(training["training_seed"]) != 20300902 or int(training["context_schedule_seed"]) != 20300904:
        raise ValueError("tail-robust seed contract drifted")
    if int(training["batch_contexts"]) != BATCH_CONTEXTS or int(training["train_candidates"]) != TRAIN_CANDIDATES or int(training["score_top_count"]) != TOP_COUNT:
        raise ValueError("tail-robust score/batch contract drifted")
    for key, value in {
        "score_top_weight": SCORE_TOP_WEIGHT,
        "score_other_weight": SCORE_OTHER_WEIGHT,
        "score_std_floor": SCORE_STD_FLOOR,
        "score_smooth_l1_beta": SCORE_SMOOTH_L1_BETA,
        "score_loss_weight": SCORE_LOSS_WEIGHT,
        "ema_decay": EMA_DECAY,
    }.items():
        _close(training[key], value, key)
    if [float(x) for x in training["horizon_weights"]] != list(base.HORIZON_WEIGHTS):
        raise ValueError("tail-robust horizon weights drifted from authoritative base freeze")
    if training.get("optimizer") != "AdamW(lr=0.0003, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8)":
        raise ValueError("tail-robust optimizer contract drifted")
    if freeze["tail_aggregation"]["top_count"] != TAIL_TOP_COUNT:
        raise ValueError("tail top-2 count drifted")
    if freeze["evaluation"]["primary_snapshot"] != "ema_step_3000":
        raise ValueError("terminal snapshot must be EMA step 3000")
    if freeze["scope_boundary"]["official_cem"] != "NOT_RUN_BY_SCOPE" or freeze["scope_boundary"]["closed_loop"] != "NOT_RUN_BY_SCOPE":
        raise ValueError("scope boundary permits an out-of-scope planner run")
    absolute = freeze["gates"]["inherited_absolute_predictor"]
    for key, value in {
        "median_spearman_min": 0.95,
        "minimum_spearman_min": 0.80,
        "median_top30_min": 0.75,
        "minimum_top30_min": 0.50,
        "median_relative_latent_mse_max": 0.25,
        "latency_reduction_min": 0.20,
        "positive_spearman_blocks_min": 12,
        "positive_top30_blocks_min": 12,
    }.items():
        _close(absolute[key], value, key)
    no_reg = freeze["gates"]["no_regression_vs_512x3000"]
    for key, value in {"spearman_median_min": 0.978317, "top30_median_min": 0.866667, "relative_latent_mse_median_max": 0.0175}.items():
        _close(no_reg[key], value, f"no_regression.{key}")
    if phase2["context_manifest"]["heldout_manifest_reused_verbatim"] is not True:
        raise ValueError("Phase 2 held-out manifest reuse is not frozen")
    return freeze, base_freeze, phase2


def per_context_score_loss(student_cost: Any, teacher_cost: Any) -> Any:
    """Return the Phase 2 score-distillation SmoothL1 scalar per context."""
    import torch
    import torch.nn.functional as F

    if student_cost.ndim != 2 or teacher_cost.shape != student_cost.shape or student_cost.shape[1] != TRAIN_CANDIDATES:
        raise ValueError("score loss expects [batch_contexts,64] matrices")
    teacher_mean = teacher_cost.mean(dim=1, keepdim=True)
    teacher_std = teacher_cost.std(dim=1, unbiased=False, keepdim=True).clamp_min(SCORE_STD_FLOOR)
    student_norm = (student_cost - student_cost.mean(dim=1, keepdim=True)) / teacher_std
    teacher_norm = (teacher_cost - teacher_mean) / teacher_std
    element_loss = F.smooth_l1_loss(student_norm, teacher_norm, beta=SCORE_SMOOTH_L1_BETA, reduction="none")
    order = torch.argsort(teacher_cost, dim=1, stable=True)
    weights = torch.full_like(element_loss, SCORE_OTHER_WEIGHT)
    weights.scatter_(1, order[:, :TOP_COUNT], SCORE_TOP_WEIGHT)
    return (element_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(SCORE_STD_FLOOR)


def tail_score_aggregate(per_context: Any, arm: str) -> Any:
    import torch

    if per_context.ndim != 1 or per_context.shape[0] != BATCH_CONTEXTS:
        raise ValueError("tail aggregation expects exactly eight context losses")
    mean = per_context.mean()
    if arm == "ema_score":
        return mean
    if arm != "ema_tail_score":
        raise ValueError(f"unknown arm: {arm}")
    return 0.75 * mean + 0.25 * torch.topk(per_context, k=TAIL_TOP_COUNT, largest=True).values.mean()


def update_ema(ema_state: dict[str, Any], student: Any, decay: float = EMA_DECAY) -> None:
    import torch

    for name, value in student.state_dict().items():
        current = value.detach().cpu()
        if name not in ema_state:
            ema_state[name] = current.clone()
        elif torch.is_floating_point(current):
            ema_state[name].mul_(decay).add_(current, alpha=1.0 - decay)
        else:
            ema_state[name] = current.clone()


def train_arm(rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any], official_model: Any, initial_state: Mapping[str, Any], arm: str) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    torch.manual_seed(int(settings["seeds"]["initialization"]))
    student = base.make_student("baseline").to("cuda")
    student.load_state_dict(copy.deepcopy(initial_state), strict=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    schedule_generator = torch.Generator(device="cpu").manual_seed(int(settings["seeds"]["context_schedule"]))
    ema_state = {key: value.detach().cpu().clone() for key, value in initial_state.items()}
    histories: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    steps = int(settings["steps"])
    for step in range(1, steps + 1):
        indices = torch.randperm(len(train_rows), generator=schedule_generator)[:BATCH_CONTEXTS]
        contexts = torch.cat([base._tensor(train_rows[int(i)]["latent_history"]).expand(TRAIN_CANDIDATES, -1, -1) for i in indices], dim=0).to("cuda")
        actions = torch.cat([base._tensor(train_rows[int(i)]["future_actions"]) for i in indices], dim=0).to("cuda")
        targets = torch.cat([base._tensor(train_rows[int(i)]["teacher_targets"]) for i in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        latent_loss, per_horizon = base.recurrent_loss(prediction, targets)
        student_cost = score._student_costs_with_gradient(official_model, train_rows, indices, prediction)
        teacher_cost = score._effective_teacher_costs(train_rows, indices, "ema_score")
        context_losses = per_context_score_loss(student_cost, teacher_cost)
        objective_loss = tail_score_aggregate(context_losses, arm)
        total_loss = latent_loss + SCORE_LOSS_WEIGHT * objective_loss
        finite = bool(torch.isfinite(total_loss).item() and torch.isfinite(per_horizon).all().item() and torch.isfinite(context_losses).all().item())
        if not finite:
            raise FloatingPointError(f"non-finite training loss at step {step} arm={arm}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        update_ema(ema_state, student)
        histories.append({
            "step": step,
            "weighted_latent_mse": float(latent_loss.detach().cpu()),
            "score_loss": float(objective_loss.detach().cpu()),
            "score_context_mean": float(context_losses.mean().detach().cpu()),
            "score_context_top2_mean": float(torch.topk(context_losses.detach(), k=TAIL_TOP_COUNT).values.mean().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "per_horizon_mse": [float(item) for item in per_horizon.detach().cpu()],
            "finite": finite,
        })
        if step in SNAPSHOT_STEPS:
            snapshots[f"step_{step}"] = {key: value.clone() for key, value in ema_state.items()}
    first = histories[0]["weighted_latent_mse"]
    last10 = statistics.median(item["weighted_latent_mse"] for item in histories[-10:])
    return {
        "snapshots": snapshots,
        "parameter_count": int(sum(parameter.numel() for parameter in student.parameters())),
        "per_step": histories,
        "last10_to_first_ratio": float(last10 / max(first, 1e-12)),
        "ema_decay": EMA_DECAY,
        "ema_terminal": "step_3000",
        "ema_isolation": True,
    }


def terminal_metrics(evaluation: Mapping[str, Any], step: int = 3000) -> dict[str, Any]:
    final = evaluation[f"step_{step}"]
    ranking = final["terminal_ranking"]
    return {
        "spearman_median": float(ranking["spearman_median"]),
        "spearman_minimum": float(ranking["spearman_minimum"]),
        "top30_median": float(ranking["top30_overlap_median"]),
        "top30_minimum": float(ranking["top30_overlap_minimum"]),
        "relative_latent_mse_median": float(final["relative_latent_mse_median"]),
        "positive_spearman_blocks": sum(float(item["spearman"]) > 0.0 for item in final["per_block"]),
        "positive_top30_blocks": sum(float(item["top30_overlap"]) > 0.0 for item in final["per_block"]),
        "finite": bool(final["finite"]),
    }


def absolute_gate(freeze: Mapping[str, Any], result: Mapping[str, Any], metrics: Mapping[str, Any], causality: Mapping[str, Any], latency: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["inherited_absolute_predictor"]
    integrity = bool(result["last10_to_first_ratio"] <= float(freeze["training"].get("convergence_ratio_max", 0.8)) and metrics["finite"] and result["ema_isolation"] and all(item["passed"] for item in causality.values()))
    ranking = {
        "spearman_median": metrics["spearman_median"],
        "spearman_minimum": metrics["spearman_minimum"],
        "top30_overlap_median": metrics["top30_median"],
        "top30_overlap_minimum": metrics["top30_minimum"],
    }
    fidelity_conditions = {
        "median_spearman_min": metrics["spearman_median"] >= float(gate["median_spearman_min"]),
        "minimum_spearman_min": metrics["spearman_minimum"] >= float(gate["minimum_spearman_min"]),
        "median_top30_min": metrics["top30_median"] >= float(gate["median_top30_min"]),
        "minimum_top30_min": metrics["top30_minimum"] >= float(gate["minimum_top30_min"]),
        "median_relative_latent_mse_max": metrics["relative_latent_mse_median"] <= float(gate["median_relative_latent_mse_max"]),
        "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(gate["positive_spearman_blocks_min"]),
        "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(gate["positive_top30_blocks_min"]),
    }
    fidelity = all(fidelity_conditions.values())
    latency_ok = float(latency["reduction"]) >= float(gate["latency_reduction_min"])
    passed = bool(integrity and fidelity and latency_ok)
    return {
        "integrity_and_convergence": {"status": "PASS" if integrity else "FAIL", "ema_isolation": bool(result["ema_isolation"]), "last10_to_first_ratio": result["last10_to_first_ratio"]},
        "predictor_fidelity_and_ranking": {"status": "PASS" if fidelity else "FAIL", "metrics": ranking, "relative_latent_mse_median": metrics["relative_latent_mse_median"], "positive_spearman_blocks": metrics["positive_spearman_blocks"], "positive_top30_blocks": metrics["positive_top30_blocks"], "conditions": fidelity_conditions},
        "causality": {"status": "PASS" if all(item["passed"] for item in causality.values()) else "FAIL", "per_prefix": dict(causality)},
        "predictor_latency": {"status": "PASS" if latency_ok else "FAIL", "metrics": latency},
        "predictor_feasibility": "GO" if passed else "NO-GO",
        "full_cem_viability": "NOT_RUN_BY_SCOPE",
    }


def no_regression_gate(freeze: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["no_regression_vs_512x3000"]
    conditions = {
        "spearman_median_min": metrics["spearman_median"] >= float(gate["spearman_median_min"]),
        "top30_median_min": metrics["top30_median"] >= float(gate["top30_median_min"]),
        "relative_latent_mse_median_max": metrics["relative_latent_mse_median"] <= float(gate["relative_latent_mse_median_max"]),
    }
    return {"status": "GO" if all(conditions.values()) else "NO-GO", "reference": "historical 512x3000 from 24926383.pbs101", "metrics": {key: metrics[key] for key in ("spearman_median", "top30_median", "relative_latent_mse_median")}, "thresholds": dict(gate), "conditions": conditions}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], base_freeze: Mapping[str, Any], phase2: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    base.require_compute_node()
    import torch

    output = args.output.resolve()
    manifest_path = args.manifest_512.resolve()
    rows_path = args.prepared_rows_512.resolve()
    reference_path = args.reference_summary.resolve()
    phase_freeze = copy.deepcopy(dict(base_freeze))
    phase_freeze["shared_data_and_schedule"]["context_manifest"]["train_context_target"] = 512
    manifest = base.validate_manifest(manifest_path, phase_freeze)
    rows = torch.load(rows_path, map_location="cpu", weights_only=False)
    row_meta = score.dense.validate_dense_rows(rows, phase_freeze)
    if row_meta["train_contexts"] != 512 or row_meta["heldout_contexts"] != 8 or row_meta["heldout_blocks"] != 16:
        raise ValueError("prepared 512 rows do not match frozen counts")
    reference = load_json(reference_path)
    if reference.get("status") != "PREDICTOR_LEVEL_COMPLETE" or reference.get("arms", {}).get("512x3000") is None:
        raise ValueError("historical 512x3000 reference summary is missing")
    if reference.get("source", {}).get("lewm_commit") != base_freeze["evidence_boundary"]["source_commit"]:
        raise ValueError("historical reference source commit does not match base freeze")

    official_model = base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    torch.manual_seed(int(freeze["training"]["initialization_seed"]))
    template = base.make_student("baseline").to("cuda")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    torch.cuda.empty_cache()
    settings = {
        "steps": int(freeze["design"]["updates"]),
        "batch_contexts": BATCH_CONTEXTS,
        "seeds": {
            "initialization": int(freeze["training"]["initialization_seed"]),
            "training": int(freeze["training"]["training_seed"]),
            "context_schedule": int(freeze["training"]["context_schedule_seed"]),
            "heldout_action_prefix": [int(x) for x in freeze["evaluation"]["heldout_action_prefix_seeds"]],
        },
    }
    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        result = train_arm(rows, settings, official_model, initial_state, arm)
        evaluation = coverage.evaluate_snapshots(official_model, result["snapshots"], rows, SNAPSHOT_STEPS)
        final_student = base.make_student("baseline").to("cuda")
        final_student.load_state_dict(result["snapshots"]["step_3000"], strict=True)
        final_student.eval()
        heldout = [row for row in rows if row.get("split") == "heldout"]
        causality = base.causality_test(final_student, heldout[0], settings["seeds"]["heldout_action_prefix"][0], float(freeze["evaluation"]["causality_tolerance"]))
        latency = base.predictor_latency(official_model, final_student, heldout[0], warmup=int(freeze["evaluation"]["predictor_latency_warmup"]), repeats=int(freeze["evaluation"]["predictor_latency_repeats"]))
        metrics = terminal_metrics(evaluation)
        gates = absolute_gate(freeze, result, metrics, causality, latency)
        arm_results[arm] = {
            "parameter_count": result["parameter_count"],
            "terminal_snapshot": "ema_step_3000",
            "ema": {"decay": EMA_DECAY, "initialization": "shared step-0 state", "updated_after_optimizer_step": True, "terminal_evaluation": True, "online_not_used_for_terminal_selection": True},
            "training": {"last10_to_first_ratio": result["last10_to_first_ratio"], "per_step": result["per_step"]},
            "evaluation": evaluation,
            "terminal_metrics": metrics,
            "causality": causality,
            "predictor_latency": latency,
            "absolute_predictor_gates": gates,
            "no_regression_vs_512x3000": no_regression_gate(freeze, metrics) if arm == "ema_tail_score" else {"status": "DIAGNOSTIC_ONLY"},
        }
        del final_student
        torch.cuda.empty_cache()

    primary = arm_results["ema_tail_score"]
    primary_conditions = {
        "absolute_predictor_feasibility": primary["absolute_predictor_gates"]["predictor_feasibility"] == "GO",
        "no_regression_vs_512x3000": primary["no_regression_vs_512x3000"]["status"] == "GO",
    }
    primary_gate = {"status": "GO" if all(primary_conditions.values()) else "NO-GO", "arm": "ema_tail_score", "conditions": primary_conditions, "absolute_predictor_gate": primary["absolute_predictor_gates"], "no_regression_gate": primary["no_regression_vs_512x3000"]}
    summary = {
        "schema": "lewm-recurrent-student.tail-robust",
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_probe": str(args.interface_probe.resolve()),
        "interface_contract": {key: value for key, value in contract.__dict__.items()},
        "source": {"lewm_commit": base_freeze["evidence_boundary"]["source_commit"], "stable_worldmodel_cem_commit": base_freeze["evidence_boundary"]["stable_worldmodel_cem_commit"], "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()), "dataset": str((args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve())},
        "historical_reference": {"formal_job": "24926383.pbs101", "summary": str(reference_path), "retrain": False, "terminal_metrics": reference["arms"]["512x3000"]["terminal_metrics"]},
        "manifest": {"path": str(manifest_path), "counts": {"train": len(manifest["splits"]["train"]), "heldout": len(manifest["splits"]["heldout"])}, "reused_verbatim": True},
        "prepared_rows": {"path": str(rows_path), "counts": row_meta, "reused_from_formal_job": "24926383.pbs101", "regenerated": False, "transferred_to_local": False},
        "authoritative_phase2": {"freeze": str(args.phase2_freeze.resolve()), "schema": phase2["schema"], "new_train_target": 512, "heldout_target": 8},
        "shared_training_contract": {"architecture": "LeWMCompactRecurrentTransitionStudent", "hidden_dim": base.HIDDEN_DIM, "attention": False, "conditioner": False, "goal_input": False, "teacher_forcing": False, "latent_loss": "unchanged horizon-weighted free-running latent MSE", "optimizer": freeze["training"]["optimizer"], "batch_contexts": BATCH_CONTEXTS, "train_candidates": TRAIN_CANDIDATES, "score_top_count": TOP_COUNT, "score_top_weight": SCORE_TOP_WEIGHT, "score_other_weight": SCORE_OTHER_WEIGHT, "score_loss_weight": SCORE_LOSS_WEIGHT, "horizon_weights": list(base.HORIZON_WEIGHTS), "initialization_state_shared": True, "context_schedule_shared": True, "heldout_candidate_bank_shared": True},
        "pairing": {"initialization_seed": settings["seeds"]["initialization"], "training_seed": settings["seeds"]["training"], "context_schedule_seed": settings["seeds"]["context_schedule"], "heldout_action_prefix_seeds": settings["seeds"]["heldout_action_prefix"], "paired_unit": "heldout context x fresh action-prefix seed block", "candidate_is_not_an_independent_statistical_unit": True, "same_initial_state": True, "same_context_schedule": True, "same_optimizer": True, "same_heldout_candidate_bank": True},
        "arms": arm_results,
        "primary_gate": primary_gate,
        "scope_checks": {"prepared_rows_512": "PASS", "manifest_counts": "PASS", "authoritative_phase2_consistency": "PASS", "ema_isolation": "PASS", "tail_aggregation": "PASS"},
        "stage_b": {"status": "NOT_RUN_BY_SCOPE", "full_cem_viability": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Predictor-level EMA/tail score-distillation comparison only. This does not claim official CEM viability, planner viability, closed-loop PushT success, encode_obs speedup, or native deployment benefit.",
    }
    write_json(output / "lewm_tail_robust_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze, base_freeze, phase2 = load_freeze(args.freeze, args.phase2_freeze)
    if args.mode == "status":
        value = {"schema": "lewm-recurrent-student.tail-robust", "status": "READY", "arms": list(ARMS), "updates": 3000, "snapshot_steps": list(SNAPSHOT_STEPS), "terminal_snapshot": "ema_step_3000", "official_cem": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value))
        return 0
    for path in (args.interface_probe, args.manifest_512, args.prepared_rows_512, args.reference_summary, args.phase2_freeze):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = base.load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, base_freeze, phase2, contract)
    print(json.dumps({"status": summary["status"], "primary_gate": summary["primary_gate"]["status"], "output": str((args.output / "lewm_tail_robust_summary.json").resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
