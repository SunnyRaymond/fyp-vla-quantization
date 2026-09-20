#!/usr/bin/env python3
"""Paired planner-aware distillation experiment for the DINO-WM PushT compiler.

This runner deliberately reuses the Stage-A implementation.  It trains two
students from one identical initialization and one shared stream of teacher
targets: the control uses latent MSE, while the treatment adds a listwise KL
loss over the frozen planner's within-anchor candidate costs.  The goal is
used only to construct that training loss; it is not an input to either
student.

The runner is predictor-level only.  It loads the official model through the
existing Stage-A loader and refuses to load a model outside a PBS compute
allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

# Keep the model/loader/teacher/evaluation/timing implementation in one place.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_dino_pusht_stage_a import (  # noqa: E402
    ANCHOR_COUNT,
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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM reproduction root")
    parser.add_argument("--asset-root", type=Path, default=None, help="staged checkpoint/data root")
    parser.add_argument("--output", type=Path, required=True, help="rank-distillation artifact directory")
    parser.add_argument("--freeze", type=Path, required=True, help="frozen PushT transfer configuration")
    parser.add_argument("--protocol", type=Path, required=True, help="new rank-distillation protocol JSON")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--deps-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _required(mapping: Mapping[str, Any], *keys: str, path: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"protocol is missing {path}; accepted fields: {', '.join(keys)}")


def _training_settings(protocol: Mapping[str, Any]) -> dict[str, Any]:
    training = protocol.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("rank-distillation protocol must contain a training object")
    ranking = training.get("rank_loss")
    if not isinstance(ranking, Mapping):
        ranking = protocol.get("ranking")
    if not isinstance(ranking, Mapping):
        loss = protocol.get("loss")
        ranking = loss.get("planner_aware_listwise_kl") if isinstance(loss, Mapping) else None
    if not isinstance(ranking, Mapping):
        raise ValueError("rank-distillation protocol must contain ranking or loss.planner_aware_listwise_kl")
    return {
        "steps": int(_required(training, "steps", path="training.steps")),
        "batch": int(_required(training, "batch_size", "batch", path="training.batch_size")),
        "lr": float(_required(training, "learning_rate", "lr", path="training.learning_rate")),
        "hidden_dim": int(_required(training, "hidden_dim", path="training.hidden_dim")),
        "rank_weight": float(_required(ranking, "weight", "lambda", "lambda_rank", path="ranking.weight")),
        "rank_temperature": float(
            _required(ranking, "temperature", "tau", path="ranking.temperature")
        ),
    }


def _seed_settings(protocol: Mapping[str, Any]) -> tuple[int, list[int], int]:
    seeds = protocol.get("anchors_and_seeds")
    if isinstance(seeds, Mapping):
        training_seed = int(
            _required(seeds, "training_action_prefix_seed", path="anchors_and_seeds.training_action_prefix_seed")
        )
        heldout = _required(
            seeds,
            "fresh_heldout_action_prefix_seeds",
            "heldout_action_prefix_seeds",
            path="anchors_and_seeds.fresh_heldout_action_prefix_seeds",
        )
        timing_seed = int(
            _required(seeds, "timing_action_prefix_seed", path="anchors_and_seeds.timing_action_prefix_seed")
        )
    else:
        training = protocol.get("training")
        heldout_settings = protocol.get("heldout")
        timing = protocol.get("timing")
        if not all(isinstance(value, Mapping) for value in (training, heldout_settings, timing)):
            raise ValueError("protocol must contain training, heldout and timing objects")
        training_seed = int(_required(training, "seed", path="training.seed"))
        heldout = _required(heldout_settings, "action_prefix_seeds", path="heldout.action_prefix_seeds")
        timing_seed = int(_required(timing, "action_prefix_seed", path="timing.action_prefix_seed"))
    eval_seeds = [int(value) for value in heldout]
    if not eval_seeds:
        raise ValueError("fresh held-out seed list must not be empty")
    return training_seed, eval_seeds, timing_seed


def _timing_settings(protocol: Mapping[str, Any]) -> dict[str, int]:
    timing = protocol.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("rank-distillation protocol must contain timing")
    return {
        "batch": int(_required(timing, "batch_size", path="timing.batch_size")),
        "warmup": int(_required(timing, "warmup_repeats", path="timing.warmup_repeats")),
        "repeats": int(_required(timing, "technical_repeats", "repeats", path="timing.technical_repeats")),
    }


def _listwise_kl(
    objective_fn: Any,
    teacher_target: Mapping[str, Any],
    prediction: Mapping[str, Any],
    goal: Mapping[str, Any],
    temperature: float,
) -> Any:
    import torch
    if temperature <= 0:
        raise ValueError("planner-aware listwise KL temperature must be positive")
    teacher_cost = _objective_cost(objective_fn, teacher_target, goal).reshape(-1).detach()
    student_cost = _objective_cost(objective_fn, prediction, goal).reshape(-1)
    if teacher_cost.numel() != student_cost.numel():
        raise RuntimeError("teacher and student planner costs have different candidate counts")
    teacher_std = teacher_cost.std(unbiased=False).clamp_min(1e-6)
    student_std = student_cost.std(unbiased=False).clamp_min(1e-6)
    teacher_z = (teacher_cost - teacher_cost.mean()) / teacher_std
    student_z = (student_cost - student_cost.mean()) / student_std
    teacher_log_prob = torch.log_softmax(-teacher_z / temperature, dim=0).detach()
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = torch.log_softmax(-student_z / temperature, dim=0)
    return torch.sum(teacher_prob * (teacher_log_prob - student_log_prob))


def _latent_mse(prediction: Mapping[str, Any], target: Mapping[str, Any]) -> Any:
    import torch.nn.functional as F

    return 0.5 * (
        F.mse_loss(prediction["visual"], target["visual"])
        + F.mse_loss(prediction["proprio"], target["proprio"])
    )


def _evaluate_pair(
    students: Mapping[str, Any],
    model: Any,
    anchors: Mapping[str, Any],
    goals: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    eval_seeds: list[int],
    eval_batch: int,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate both arms on the same freshly generated actions and targets."""

    import torch

    for student in students.values():
        student.eval()
    cases: dict[str, list[dict[str, Any]]] = {name: [] for name in students}
    paired: list[dict[str, Any]] = []
    for eval_seed in eval_seeds:
        generator = torch.Generator(device="cpu").manual_seed(eval_seed)
        for anchor_index in range(ANCHOR_COUNT):
            context = _repeat_anchor(
                {key: value[anchor_index : anchor_index + 1] for key, value in anchors.items()},
                eval_batch,
            )
            goal = _repeat_anchor(
                {key: value[anchor_index : anchor_index + 1, :1] for key, value in goals.items()},
                eval_batch,
            )
            actions = _sample_actions(generator, eval_batch, action_dim, device)
            target = _teacher_targets(model, context, actions)
            with torch.no_grad():
                predictions = {name: student(context, actions) for name, student in students.items()}
            ranked: dict[str, dict[str, float]] = {}
            for name, prediction in predictions.items():
                metrics = _metrics_by_horizon(prediction, target)
                teacher_cost = _objective_cost(objective_fn, target, goal)
                student_cost = _objective_cost(objective_fn, prediction, goal)
                rank_metrics = {
                    "spearman": _spearman(teacher_cost, student_cost),
                    "top30_overlap": _topk_overlap(teacher_cost, student_cost),
                }
                metrics.update({"seed": eval_seed, "anchor": anchor_index})
                metrics["ranking"] = rank_metrics
                cases[name].append(metrics)
                ranked[name] = rank_metrics
            paired.append(
                {
                    "seed": eval_seed,
                    "anchor": anchor_index,
                    "spearman_delta_treatment_minus_control": ranked["treatment"]["spearman"]
                    - ranked["control"]["spearman"],
                    "top30_overlap_delta_treatment_minus_control": ranked["treatment"]["top30_overlap"]
                    - ranked["control"]["top30_overlap"],
                }
            )

    result: dict[str, Any] = {}
    for name, arm_cases in cases.items():
        def _mean_series(key: str) -> list[float]:
            return [
                sum(item[key][index] for item in arm_cases) / len(arm_cases)
                for index in range(HORIZON)
            ]

        ranking_cases = [item["ranking"] for item in arm_cases]
        result[name] = {
            "per_horizon": {
                "relative_mse": _mean_series("relative_mse"),
                "cosine": _mean_series("cosine"),
            },
            "per_case": arm_cases,
            "terminal_ranking": {
                "spearman_mean": sum(item["spearman"] for item in ranking_cases) / len(ranking_cases),
                "spearman_median": _median([item["spearman"] for item in ranking_cases]),
                "spearman_minimum": min(item["spearman"] for item in ranking_cases),
                "top30_overlap_mean": sum(item["top30_overlap"] for item in ranking_cases)
                / len(ranking_cases),
                "top30_overlap_median": _median([item["top30_overlap"] for item in ranking_cases]),
                "top30_overlap_minimum": min(item["top30_overlap"] for item in ranking_cases),
                "per_case": ranking_cases,
            },
        }
    return result, paired


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _all_finite_numbers(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _gate_status(protocol: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate only thresholds explicitly frozen in the new protocol."""

    gates = protocol.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("rank-distillation protocol must contain frozen gates")
    capacity = gates.get("capacity")
    fidelity = gates.get("absolute_fidelity")
    improvement = gates.get("paired_improvement")
    noninferiority = gates.get("latent_noninferiority")
    latency = gates.get("latency")
    if not all(isinstance(value, Mapping) for value in (capacity, fidelity, improvement, noninferiority, latency)):
        raise ValueError("protocol is missing one or more frozen gate groups")

    control = results["evaluation"]["control"]["terminal_ranking"]
    treatment = results["evaluation"]["treatment"]["terminal_ranking"]
    deltas = results["paired_deltas"]
    control_loss = results["train"]["control"]["latent_last10_to_first_ratio"]
    treatment_loss = results["train"]["treatment"]["latent_last10_to_first_ratio"]
    causal = results["causality"]
    latency_result = results["latency"]

    max_loss_ratio = float(_required(capacity, "latent_last10_to_first_ratio_max", path="gates.capacity.latent_last10_to_first_ratio_max"))
    causal_pass = all(bool(value["passed"]) for value in causal.values())
    capacity_pass = (
        causal_pass
        and bool(results["finite_outputs_and_training"])
        and math.isfinite(control_loss)
        and math.isfinite(treatment_loss)
        and control_loss <= max_loss_ratio
        and treatment_loss <= max_loss_ratio
    )

    min_treat_spearman = float(_required(fidelity, "rankdistill_objective_spearman_min", path="gates.absolute_fidelity.rankdistill_objective_spearman_min"))
    min_treat_top30 = float(_required(fidelity, "rankdistill_top30_overlap_min", path="gates.absolute_fidelity.rankdistill_top30_overlap_min"))
    min_block_spearman = float(_required(fidelity, "minimum_block_spearman_min", path="gates.absolute_fidelity.minimum_block_spearman_min"))
    min_block_top30 = float(_required(fidelity, "minimum_block_top30_overlap_min", path="gates.absolute_fidelity.minimum_block_top30_overlap_min"))
    min_delta_spearman = float(_required(improvement, "median_spearman_delta_min", path="gates.paired_improvement.median_spearman_delta_min"))
    min_delta_top30 = float(_required(improvement, "median_top30_overlap_delta_min", path="gates.paired_improvement.median_top30_overlap_delta_min"))
    min_positive_spearman = int(_required(improvement, "positive_spearman_blocks_min", path="gates.paired_improvement.positive_spearman_blocks_min"))
    min_positive_top30 = int(_required(improvement, "positive_top30_blocks_min", path="gates.paired_improvement.positive_top30_blocks_min"))
    latent_ratio_max = float(_required(noninferiority, "rankdistill_to_latent_only_mean_relative_mse_ratio_max", path="gates.latent_noninferiority.rankdistill_to_latent_only_mean_relative_mse_ratio_max"))
    latent_ratio = float(results["latent_noninferiority_ratio"])
    fidelity_pass = (
        treatment["spearman_median"] >= min_treat_spearman
        and treatment["top30_overlap_median"] >= min_treat_top30
        and treatment["spearman_minimum"] >= min_block_spearman
        and treatment["top30_overlap_minimum"] >= min_block_top30
        and deltas["spearman_median"] >= min_delta_spearman
        and deltas["top30_overlap_median"] >= min_delta_top30
        and deltas["positive_spearman_blocks"] >= min_positive_spearman
        and deltas["positive_top30_blocks"] >= min_positive_top30
        and latent_ratio <= latent_ratio_max
    )

    min_reduction = float(
        _required(latency, "rankdistill_predictor_reduction_vs_teacher_min", path="gates.latency.rankdistill_predictor_reduction_vs_teacher_min")
    )
    latency_pass = float(latency_result["treatment"]["median_reduction"]) >= min_reduction
    return {
        "capacity": {
            "status": "PASS" if capacity_pass else "FAIL",
            "causality": causal_pass,
            "finite_outputs_and_training": bool(results["finite_outputs_and_training"]),
            "control_last10_to_first_loss_ratio": control_loss,
            "treatment_last10_to_first_loss_ratio": treatment_loss,
            "maximum_loss_ratio": max_loss_ratio,
        },
        "fidelity": {
            "status": "PASS" if fidelity_pass else "FAIL",
            "treatment_spearman": treatment["spearman_median"],
            "treatment_spearman_minimum": min_treat_spearman,
            "treatment_minimum_block_spearman": treatment["spearman_minimum"],
            "treatment_top30_overlap": treatment["top30_overlap_median"],
            "treatment_top30_overlap_minimum": min_treat_top30,
            "treatment_minimum_block_top30_overlap": treatment["top30_overlap_minimum"],
            "spearman_delta": deltas["spearman_median"],
            "spearman_delta_minimum": min_delta_spearman,
            "top30_overlap_delta": deltas["top30_overlap_median"],
            "top30_overlap_delta_minimum": min_delta_top30,
            "positive_spearman_blocks": deltas["positive_spearman_blocks"],
            "positive_top30_blocks": deltas["positive_top30_blocks"],
            "latent_noninferiority_ratio": latent_ratio,
            "latent_noninferiority_ratio_maximum": latent_ratio_max,
        },
        "predictor_latency": {
            "status": "PASS" if latency_pass else "FAIL",
            "minimum_reduction": min_reduction,
            "treatment_reduction": latency_result["treatment"]["median_reduction"],
        },
        "overall": "PASS" if capacity_pass and fidelity_pass and latency_pass else "FAIL",
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
    settings = _training_settings(protocol)
    training_seed, eval_seeds, timing_seed = _seed_settings(protocol)
    timing = _timing_settings(protocol)
    if settings["batch"] != 32:
        raise ValueError("rank-distillation protocol must freeze batch_size=32 for within-anchor ranking")
    if settings["steps"] < 1 or settings["rank_weight"] < 0:
        raise ValueError("protocol training steps must be positive and rank weight non-negative")
    _require_assets(asset_root, freeze)
    _gpu_snapshot(output, "start")

    import torch

    _set_seed(training_seed)
    model, workspace, anchors, goals, objective_fn, action_dim, device = _load_official(
        root,
        asset_root,
        freeze,
        output,
        training_seed,
        config_path=args.config.resolve() if args.config else None,
        checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None,
        checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None,
        data_root=args.data_root.resolve() if args.data_root else None,
    )
    del workspace
    visual_dim = int(anchors["visual"].shape[-1])
    proprio_dim = int(anchors["proprio"].shape[-1])
    template = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, settings["hidden_dim"]).to(device)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    students = {
        name: NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, settings["hidden_dim"]).to(device)
        for name in ("control", "treatment")
    }
    for student in students.values():
        student.load_state_dict(initial_state)
    optimizers = {
        name: torch.optim.AdamW(student.parameters(), lr=settings["lr"])
        for name, student in students.items()
    }
    train_generator = torch.Generator(device="cpu").manual_seed(training_seed)
    train_losses: dict[str, list[dict[str, float]]] = {"control": [], "treatment": []}
    telemetry_period = max(1, settings["steps"] // 10)

    for step in range(settings["steps"]):
        anchor_index = step % ANCHOR_COUNT
        context = _repeat_anchor(
            {key: value[anchor_index : anchor_index + 1] for key, value in anchors.items()}, settings["batch"]
        )
        goal = _repeat_anchor(
            {key: value[anchor_index : anchor_index + 1, :1] for key, value in goals.items()}, settings["batch"]
        )
        actions = _sample_actions(train_generator, settings["batch"], action_dim, device)
        target = _teacher_targets(model, context, actions)
        order = ("control", "treatment") if step % 2 == 0 else ("treatment", "control")
        for name in order:
            prediction = students[name](context, actions)
            latent_loss = _latent_mse(prediction, target)
            if name == "treatment":
                rank_loss = _listwise_kl(
                    objective_fn, target, prediction, goal, settings["rank_temperature"]
                )
                total_loss = latent_loss + settings["rank_weight"] * rank_loss
            else:
                rank_loss = latent_loss.detach().new_zeros(())
                total_loss = latent_loss
            optimizers[name].zero_grad(set_to_none=True)
            total_loss.backward()
            optimizers[name].step()
            train_losses[name].append(
                {
                    "step": step + 1,
                    "anchor": anchor_index,
                    "total": float(total_loss.detach().cpu()),
                    "latent_mse": float(latent_loss.detach().cpu()),
                    "planner_listwise_kl": float(rank_loss.detach().cpu()),
                }
            )
        if step == 0 or (step + 1) % telemetry_period == 0:
            _gpu_snapshot(output, f"train_step_{step + 1}")

    evaluation, paired_cases = _evaluate_pair(
        students, model, anchors, goals, objective_fn, action_dim, eval_seeds, timing["batch"], device
    )
    causality = {
        name: _leakage_test(
            student,
            anchors,
            action_dim,
            eval_seeds[0] + 9000,
            device,
            float(protocol.get("controls", {}).get("future_action_tolerance", 1e-6)),
        )
        for name, student in students.items()
    }
    latency = {
        name: _latency(
            student,
            model,
            anchors,
            action_dim,
            timing_seed,
            timing["batch"],
            timing["warmup"],
            timing["repeats"],
            device,
        )
        for name, student in students.items()
    }
    _gpu_snapshot(output, "complete")

    def _arm_train(name: str) -> dict[str, Any]:
        losses = train_losses[name]
        totals = [item["total"] for item in losses]
        latent = [item["latent_mse"] for item in losses]
        last10 = totals[-10:]
        latent_last10 = latent[-10:]
        return {
            "steps": settings["steps"],
            "batch": settings["batch"],
            "loss_first": totals[0],
            "loss_last": totals[-1],
            "loss_median_last_10": _median(last10),
            "last10_to_first_loss_ratio": _median(last10) / max(totals[0], 1e-12),
            "latent_mse_first": latent[0],
            "latent_mse_median_last_10": _median(latent_last10),
            "latent_last10_to_first_ratio": _median(latent_last10) / max(latent[0], 1e-12),
            "per_step": losses,
        }

    paired_spearman = [item["spearman_delta_treatment_minus_control"] for item in paired_cases]
    paired_top30 = [item["top30_overlap_delta_treatment_minus_control"] for item in paired_cases]
    paired_deltas = {
        "spearman_mean": _mean(paired_spearman),
        "spearman_median": _median(paired_spearman),
        "top30_overlap_mean": _mean(paired_top30),
        "top30_overlap_median": _median(paired_top30),
        "positive_spearman_blocks": sum(value > 0 for value in paired_spearman),
        "positive_top30_blocks": sum(value > 0 for value in paired_top30),
        "per_case": paired_cases,
    }
    control_relative_mse = _mean(evaluation["control"]["per_horizon"]["relative_mse"])
    treatment_relative_mse = _mean(evaluation["treatment"]["per_horizon"]["relative_mse"])
    latent_noninferiority_ratio = treatment_relative_mse / max(control_relative_mse, 1e-12)
    result = {
        "evaluation": evaluation,
        "paired_deltas": paired_deltas,
        "train": {name: _arm_train(name) for name in students},
        "causality": causality,
        "latency": latency,
        "latent_noninferiority_ratio": latent_noninferiority_ratio,
    }
    result["finite_outputs_and_training"] = _all_finite_numbers(result)
    gates = _gate_status(protocol, result)
    checkpoint_common = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-rank-distill-checkpoint",
        "horizon": HORIZON,
        "native_output_fields": ["visual", "proprio"],
        "action_dim": action_dim,
        "visual_dim": visual_dim,
        "proprio_dim": proprio_dim,
        "hidden_dim": settings["hidden_dim"],
        "train_seed": training_seed,
        "rank_weight": settings["rank_weight"],
        "rank_temperature": settings["rank_temperature"],
    }
    for name, student in students.items():
        torch.save({**checkpoint_common, "arm": name, "state_dict": student.state_dict()}, output / f"{name}_student.pt")

    summary = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-rank-distill-summary",
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "freeze": str(args.freeze.resolve()),
        "source": {
            "reproduction_root": str(root),
            "official_loader": "run_dino_pusht_stage_a._load_official",
            "teacher_targets": "shared frozen teacher target per step and held-out case",
        },
        "contracts": {
            "within_anchor_batch": settings["batch"],
            "goal_in_student_input": False,
            "goal_used_for": "planner-aware training loss and evaluation objective only",
            "same_initial_state_dict": True,
            "alternating_update_order": "control_then_treatment on even steps; treatment_then_control on odd steps",
            "heldout_seeds": eval_seeds,
        },
        "train": {name: _arm_train(name) for name in students},
        "causality": causality,
        "evaluation": evaluation,
        "paired_deltas": paired_deltas,
        "latency": latency,
        "latent_noninferiority": {
            "control_mean_relative_mse": control_relative_mse,
            "treatment_mean_relative_mse": treatment_relative_mse,
            "treatment_to_control_ratio": latent_noninferiority_ratio,
        },
        "timing_boundary": "predictor-level: cached native observation latent plus normalized action prefix; excludes encoder, CEM, environment interaction and closed-loop execution",
        "loss": {
            "control": "latent_mse",
            "treatment": "latent_mse + rank_weight * listwise_kl",
            "rank_weight": settings["rank_weight"],
            "rank_temperature": settings["rank_temperature"],
            "listwise_scope": "within one anchor over the batch of 32 actions",
        },
        "gates": gates,
        "unverified": [
            "No closed-loop CEM or environment execution is included.",
            "This DINO-only runner does not establish LeWM transfer.",
        ],
    }
    summary_path = (args.summary.resolve() if args.summary else output / "rank_distill_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    default_summary = output / "rank_distill_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
