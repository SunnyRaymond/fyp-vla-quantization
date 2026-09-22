#!/usr/bin/env python3
"""LeWM PushT bounded score-distillation versus pairwise-rank experiment.

The three arms share the prepared dense-rank candidate bank, initial state, and
context schedule.  The student architecture is unchanged; only the external
training objective differs.
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

# Reuse the validated baseline runner and dense-rank evaluator.  The prepared
# rows are loaded from the prior formal job; this runner intentionally has no
# row-generation path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dense-rank"))
import run_lewm_dense_rank as dense  # noqa: E402

base = dense.base

TRAIN_CANDIDATES = 64
SCORE_TOP_COUNT = 12
SCORE_TOP_WEIGHT = 2.0
SCORE_OTHER_WEIGHT = 1.0
SCORE_STD_FLOOR = 1e-6
SCORE_SMOOTH_L1_BETA = 1.0
SCORE_LOSS_WEIGHT = 0.1
ARMS = ("pairwise_rank", "score_distill", "shuffled_score")

SCREEN_SPEARMAN_MIN = 0.701261
SCREEN_TOP30_MIN = 0.433333
SCREEN_RELATIVE_MSE_MAX = 0.0175
SCREEN_POSITIVE_TOP30_EXACT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared-rows", type=Path, required=True)
    parser.add_argument("--action-low", type=float, nargs=2, required=True)
    parser.add_argument("--action-high", type=float, nargs=2, required=True)
    parser.add_argument("--gaussian-std", type=float, required=True)
    parser.add_argument("--mode", choices=("status", "run"), default="status")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_freeze(path: Path) -> dict[str, Any]:
    return dense.load_freeze(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def validate_score_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    settings = base.validate_freeze(freeze)
    overlay = freeze.get("experiment_overlay", {})
    if overlay.get("experiment_name") != "lewm_score_distillation_vs_pairwise_rank":
        raise ValueError("unexpected score-distill freeze")
    train = overlay.get("training", {})
    checks = {
        "train_candidates": (int(train.get("train_candidates", -1)), TRAIN_CANDIDATES),
        "score_top_count": (int(train.get("score_top_count", -1)), SCORE_TOP_COUNT),
        "rank_shuffle_seed": (int(train.get("rank_shuffle_seed", -1)), dense.RANK_SHUFFLE_SEED),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{name} drifted: {actual} != {expected}")
    float_checks = {
        "score_top_weight": (float(train.get("score_top_weight", -1)), SCORE_TOP_WEIGHT),
        "score_other_weight": (float(train.get("score_other_weight", -1)), SCORE_OTHER_WEIGHT),
        "score_std_floor": (float(train.get("score_std_floor", -1)), SCORE_STD_FLOOR),
        "score_smooth_l1_beta": (float(train.get("score_smooth_l1_beta", -1)), SCORE_SMOOTH_L1_BETA),
        "score_loss_weight": (float(train.get("score_loss_weight", -1)), SCORE_LOSS_WEIGHT),
        "rank_temperature": (float(train.get("rank_temperature", -1)), dense.RANK_TEMPERATURE),
    }
    for name, (actual, expected) in float_checks.items():
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name} drifted: {actual} != {expected}")
    return settings


def _effective_teacher_costs(rows: Sequence[Mapping[str, Any]], indices: Any, arm: str, device: str = "cuda") -> Any:
    import torch

    values = []
    for index in indices:
        cost = base._tensor(rows[int(index)]["teacher_objective"], dtype=torch.float32)
        if arm == "shuffled_score":
            permutation = base._tensor(rows[int(index)]["rank_shuffle_indices"], dtype=torch.long)
            cost = cost[permutation]
        values.append(cost)
    return torch.stack(values).to(device)


def score_distill_loss(student_cost: Any, teacher_cost: Any) -> Any:
    """Context-normalized, bounded teacher-score regression.

    Both costs are lower-is-better official objectives.  Normalizing student
    and teacher with the same teacher standard deviation preserves ordering
    while avoiding an unbounded pairwise margin incentive.
    """
    import torch
    import torch.nn.functional as F

    if student_cost.ndim != 2 or teacher_cost.shape != student_cost.shape or student_cost.shape[1] != TRAIN_CANDIDATES:
        raise ValueError("score loss expects [batch_contexts,64] cost matrices")
    teacher_mean = teacher_cost.mean(dim=1, keepdim=True)
    teacher_std = teacher_cost.std(dim=1, unbiased=False, keepdim=True).clamp_min(SCORE_STD_FLOOR)
    student_norm = (student_cost - student_cost.mean(dim=1, keepdim=True)) / teacher_std
    teacher_norm = (teacher_cost - teacher_mean) / teacher_std
    element_loss = F.smooth_l1_loss(student_norm, teacher_norm, beta=SCORE_SMOOTH_L1_BETA, reduction="none")
    order = torch.argsort(teacher_cost, dim=1, stable=True)
    weights = torch.full_like(element_loss, SCORE_OTHER_WEIGHT)
    weights.scatter_(1, order[:, :SCORE_TOP_COUNT], SCORE_TOP_WEIGHT)
    return (element_loss * weights).sum() / weights.sum().clamp_min(SCORE_STD_FLOOR)


def _student_costs_with_gradient(official_model: Any, rows: Sequence[Mapping[str, Any]], indices: Any, prediction: Any) -> Any:
    import torch

    costs = []
    offset = 0
    for index in indices:
        context = base._tensor(rows[int(index)]["latent_history"], dtype=prediction.dtype).to(prediction.device)
        goal = base._tensor(rows[int(index)]["goal_emb"], dtype=prediction.dtype).to(prediction.device)
        local_row = {"latent_history": context, "goal_emb": goal}
        costs.append(base._official_objective(official_model, local_row, prediction[offset : offset + TRAIN_CANDIDATES]))
        offset += TRAIN_CANDIDATES
    result = torch.stack(costs)
    if result.shape != (len(indices), TRAIN_CANDIDATES):
        raise RuntimeError("student objective shape is not [batch,64]")
    return result


def train_arm(rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any], output: Path, official_model: Any, initial_state: Mapping[str, Any], arm: str) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    torch.manual_seed(int(settings["seeds"]["initialization"]))
    student = base.make_student("baseline").to("cuda")
    student.load_state_dict(copy.deepcopy(initial_state), strict=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    schedule_generator = torch.Generator(device="cpu").manual_seed(int(settings["seeds"]["context_schedule"]))
    histories, snapshots = [], {}
    for step in range(1, int(settings["steps"]) + 1):
        indices = torch.randperm(len(train_rows), generator=schedule_generator)[: settings["batch_contexts"]]
        contexts = torch.cat([base._tensor(train_rows[int(i)]["latent_history"]).expand(TRAIN_CANDIDATES, -1, -1) for i in indices], dim=0).to("cuda")
        actions = torch.cat([base._tensor(train_rows[int(i)]["future_actions"]) for i in indices], dim=0).to("cuda")
        targets = torch.cat([base._tensor(train_rows[int(i)]["teacher_targets"]) for i in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        latent_loss, per_horizon = base.recurrent_loss(prediction, targets)
        student_cost = _student_costs_with_gradient(official_model, train_rows, indices, prediction)
        if arm == "pairwise_rank":
            teacher_cost = _effective_teacher_costs(train_rows, indices, "pairwise_rank")
            objective_loss = dense.pairwise_rank_loss(student_cost, teacher_cost)
            objective_name = "pairwise_rank_loss"
        else:
            teacher_cost = _effective_teacher_costs(train_rows, indices, arm)
            objective_loss = score_distill_loss(student_cost, teacher_cost)
            objective_name = "score_distill_loss"
        total_loss = latent_loss + SCORE_LOSS_WEIGHT * objective_loss
        if not torch.isfinite(total_loss) or not bool(torch.isfinite(per_horizon).all()) or not torch.isfinite(objective_loss):
            raise FloatingPointError(f"non-finite training loss at step {step} arm={arm}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        histories.append({
            "step": step,
            "weighted_latent_mse": float(latent_loss.detach().cpu()),
            "objective_loss": float(objective_loss.detach().cpu()),
            "objective_name": objective_name,
            "total_loss": float(total_loss.detach().cpu()),
            "per_horizon_mse": [float(item) for item in per_horizon.detach().cpu()],
        })
        if step in base.SNAPSHOT_STEPS:
            state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
            snapshots[f"step_{step}"] = state
    first = histories[0]["weighted_latent_mse"]
    last10 = statistics.median(item["weighted_latent_mse"] for item in histories[-10:])
    return {"student": student, "snapshots": snapshots, "parameter_count": int(sum(parameter.numel() for parameter in student.parameters())), "per_step": histories, "last10_to_first_ratio": float(last10 / max(first, 1e-12))}


def _paired_delta(first: Mapping[str, Any], second: Mapping[str, Any], key: str, lower_is_better: bool = False) -> dict[str, Any]:
    a = {item["pairing_key"]: float(item[key]) for item in first["step_1500"]["per_block"]}
    b = {item["pairing_key"]: float(item[key]) for item in second["step_1500"]["per_block"]}
    deltas = [b[key_] - a[key_] for key_ in a]
    if lower_is_better:
        improve = sum(delta < 0 for delta in deltas)
        worse = sum(delta > 0 for delta in deltas)
    else:
        improve = sum(delta > 0 for delta in deltas)
        worse = sum(delta < 0 for delta in deltas)
    return {"improve": improve, "worse": worse, "tie": len(deltas) - improve - worse, "median_delta": float(statistics.median(deltas)), "deltas": deltas}


def _screen_metrics(arm_result: Mapping[str, Any]) -> dict[str, Any]:
    terminal = arm_result["evaluation"]["step_1500"]["terminal_ranking"]
    per_block = arm_result["evaluation"]["step_1500"]["per_block"]
    return {
        "spearman_median": float(terminal["spearman_median"]),
        "top30_median": float(terminal["top30_overlap_median"]),
        "top30_minimum": float(terminal["top30_overlap_minimum"]),
        "relative_latent_mse_median": float(arm_result["evaluation"]["step_1500"]["relative_latent_mse_median"]),
        "positive_top30_blocks": sum(float(item["top30_overlap"]) > 0 for item in per_block),
    }


def _meets_score_screen(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "spearman_median_min": metrics["spearman_median"] >= SCREEN_SPEARMAN_MIN,
        "top30_median_min": metrics["top30_median"] >= SCREEN_TOP30_MIN,
        "relative_latent_mse_median_max": metrics["relative_latent_mse_median"] <= SCREEN_RELATIVE_MSE_MAX,
        "positive_top30_blocks_exact": metrics["positive_top30_blocks"] == SCREEN_POSITIVE_TOP30_EXACT,
        "minimum_top30_strictly_positive": metrics["top30_minimum"] > 0.0,
    }


def screening_gate(arm_results: Mapping[str, Any]) -> dict[str, Any]:
    pairwise = _screen_metrics(arm_results["pairwise_rank"])
    score = _screen_metrics(arm_results["score_distill"])
    shuffled = _screen_metrics(arm_results["shuffled_score"])
    score_conditions = _meets_score_screen(score)
    shuffled_conditions = _meets_score_screen(shuffled)
    score_vs_pairwise = {
        "top30": _paired_delta(arm_results["pairwise_rank"]["evaluation"], arm_results["score_distill"]["evaluation"], "top30_overlap"),
        "spearman": _paired_delta(arm_results["pairwise_rank"]["evaluation"], arm_results["score_distill"]["evaluation"], "spearman"),
        "relative_latent_mse": _paired_delta(arm_results["pairwise_rank"]["evaluation"], arm_results["score_distill"]["evaluation"], "relative_latent_mse", lower_is_better=True),
    }
    shuffled_vs_pairwise = {
        "top30": _paired_delta(arm_results["pairwise_rank"]["evaluation"], arm_results["shuffled_score"]["evaluation"], "top30_overlap"),
        "spearman": _paired_delta(arm_results["pairwise_rank"]["evaluation"], arm_results["shuffled_score"]["evaluation"], "spearman"),
        "relative_latent_mse": _paired_delta(arm_results["pairwise_rank"]["evaluation"], arm_results["shuffled_score"]["evaluation"], "relative_latent_mse", lower_is_better=True),
    }
    score_conditions["shuffled_score_cannot_match_score_distill"] = not all(shuffled_conditions.values())
    return {
        "status": "GO" if all(score_conditions.values()) else "NO-GO",
        "reference_arm": "pairwise_rank",
        "score_distill_metrics": score,
        "shuffled_score_metrics": shuffled,
        "pairwise_reference_metrics": pairwise,
        "score_distill_conditions": score_conditions,
        "shuffled_score_same_threshold_conditions": shuffled_conditions,
        "score_distill_vs_pairwise_rank": score_vs_pairwise,
        "shuffled_score_vs_pairwise_rank": shuffled_vs_pairwise,
    }


def run(args: argparse.Namespace, freeze: Mapping[str, Any], settings: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    base.require_compute_node()
    import torch

    dataset_path = (args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve()
    manifest_path = args.manifest.resolve()
    manifest = base.validate_manifest(manifest_path, freeze)
    rows_path = args.prepared_rows.resolve()
    if not rows_path.is_file():
        raise FileNotFoundError(f"prepared rows must be reused from the completed dense-rank job: {rows_path}")
    rows = torch.load(rows_path, map_location="cpu", weights_only=False)
    row_meta = dense.validate_dense_rows(rows, freeze)
    official_model = base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    torch.manual_seed(int(settings["seeds"]["initialization"]))
    template = base.make_student("baseline").to("cuda")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        result = train_arm(rows, settings, args.output.resolve(), official_model, initial_state, arm)
        evaluation = base.stage_a_evaluate(official_model, result["snapshots"], rows, settings, "baseline")
        final_student = base.make_student("baseline").to("cuda")
        final_student.load_state_dict(result["snapshots"]["step_1500"], strict=True)
        final_student.eval()
        heldout = [row for row in rows if row.get("split") == "heldout"]
        causality = base.causality_test(final_student, heldout[0], int(settings["seeds"]["heldout_action_prefix"][0]), 1e-6)
        latency = base.predictor_latency(official_model, final_student, heldout[0])
        gates = base.predictor_gates(freeze, {"last10_to_first_ratio": result["last10_to_first_ratio"]}, evaluation, causality, latency)
        arm_results[arm] = {
            "parameter_count": result["parameter_count"],
            "training": {"last10_to_first_ratio": result["last10_to_first_ratio"], "per_step": result["per_step"]},
            "evaluation": evaluation,
            "causality": causality,
            "predictor_latency": latency,
            "absolute_predictor_gates": gates,
        }
    screening = screening_gate(arm_results)
    summary = {
        "schema": "lewm-recurrent-student.score-distill",
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_probe": str(args.interface_probe.resolve()),
        "interface_contract": jsonable(contract.__dict__),
        "source": {
            "lewm_commit": freeze["evidence_boundary"]["source_commit"],
            "stable_worldmodel_cem_commit": freeze["evidence_boundary"]["stable_worldmodel_cem_commit"],
            "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()),
            "dataset": str(dataset_path),
        },
        "manifest": str(manifest_path),
        "prepared_rows": {"path": str(rows_path), "reused_from_formal_job": "24908446.pbs101", "regenerated": False, "transferred_to_local": False},
        "prepared_row_counts": row_meta,
        "shared_training_contract": {
            "architecture": "LeWMCompactRecurrentTransitionStudent",
            "hidden_dim": base.HIDDEN_DIM,
            "attention": False,
            "conditioner": False,
            "goal_input": False,
            "teacher_forcing": False,
            "initialization_state_shared": True,
            "context_schedule_shared": True,
            "train_candidate_bank_shared": True,
            "score_top_count": SCORE_TOP_COUNT,
            "score_top_weight": SCORE_TOP_WEIGHT,
            "score_other_weight": SCORE_OTHER_WEIGHT,
            "score_std_floor": SCORE_STD_FLOOR,
            "score_smooth_l1_beta": SCORE_SMOOTH_L1_BETA,
            "score_loss_weight": SCORE_LOSS_WEIGHT,
            "rank_reference_temperature": dense.RANK_TEMPERATURE,
        },
        "pairing": {
            "initialization_seed": settings["seeds"]["initialization"],
            "training_seed": settings["seeds"]["training"],
            "context_schedule_seed": settings["seeds"]["context_schedule"],
            "rank_shuffle_seed": dense.RANK_SHUFFLE_SEED,
            "same_initial_state": True,
            "same_context_schedule": True,
            "same_train_candidate_bank": True,
            "same_heldout_candidate_bank": True,
        },
        "arms": arm_results,
        "screening_gate": screening,
        "stage_b": {"status": "NOT_RUN_BY_SCOPE", "full_cem_viability": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "This is a predictor-level three-arm training comparison only. It does not claim official CEM viability, planner viability, closed-loop PushT success, encode_obs speedup, or Fast-LeWM comparison.",
    }
    write_json(args.output.resolve() / "lewm_score_distill_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = load_freeze(args.freeze.resolve())
    settings = validate_score_freeze(freeze)
    if args.mode == "status":
        value = {"schema": "lewm-recurrent-student.score-distill", "status": "READY", "train_candidates": TRAIN_CANDIDATES, "arms": list(ARMS), "prepared_rows_required": True, "interface_probe": str(args.interface_probe.resolve())}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value))
        return 0
    if not args.interface_probe.is_file():
        raise FileNotFoundError(args.interface_probe)
    if not args.prepared_rows.is_file():
        raise FileNotFoundError(args.prepared_rows)
    contract = base.load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, settings, contract)
    print(json.dumps({"status": summary["status"], "screening_gate": summary["screening_gate"]["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
