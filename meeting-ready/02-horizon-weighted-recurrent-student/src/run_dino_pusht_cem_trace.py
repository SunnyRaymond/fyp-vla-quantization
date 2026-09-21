#!/usr/bin/env python3
"""Diagnose recurrent-student cost ranking and CEM distribution drift."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--plan-targets", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing execution outside PBS")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _rank(values: Any) -> Any:
    import torch

    return torch.argsort(torch.argsort(values, stable=True), stable=True).float()


def _spearman(first: Any, second: Any) -> float:
    import torch

    a = _rank(first.detach().float().cpu())
    b = _rank(second.detach().float().cpu())
    a -= a.mean()
    b -= b.mean()
    denom = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-12))
    return float((a * b).sum() / denom)


def _pearson(first: Any, second: Any) -> float:
    import torch

    a = first.detach().float().cpu()
    b = second.detach().float().cpu()
    a -= a.mean()
    b -= b.mean()
    denom = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-12))
    return float((a * b).sum() / denom)


def _rms(value: Any) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _overlap(first: Any, second: Any, k: int) -> float:
    a = set(first[:k].detach().cpu().tolist())
    b = set(second[:k].detach().cpu().tolist())
    return float(len(a.intersection(b)) / k)


def _pool_metrics(reference_cost: Any, probe_cost: Any, actions: Any, topk: int) -> dict[str, Any]:
    import torch

    # Match the pinned official CEM implementation exactly: plain argsort and
    # torch.std's default unbiased estimator.
    ref_order = reference_cost.argsort()
    probe_order = probe_cost.argsort()
    ref_idx = ref_order[:topk]
    probe_idx = probe_order[:topk]
    ref_elite_mean = actions.index_select(0, ref_idx).mean(dim=0)
    probe_elite_mean = actions.index_select(0, probe_idx).mean(dim=0)
    ref_cost = float(reference_cost.index_select(0, ref_idx).mean().detach().cpu())
    selected_reference_costs = reference_cost.index_select(0, probe_idx)
    selected_cost = float(selected_reference_costs.mean().detach().cpu())
    inverse_rank = reference_cost.new_empty(reference_cost.shape, dtype=torch.long)
    inverse_rank[ref_order] = torch.arange(reference_cost.numel(), device=reference_cost.device)
    selected_ranks = inverse_rank.index_select(0, probe_idx) + 1
    return {
        "spearman": _spearman(reference_cost, probe_cost),
        "pearson": _pearson(reference_cost, probe_cost),
        "topk_overlap": _overlap(ref_idx, probe_idx, topk),
        "reference_elite_mean_cost": ref_cost,
        "reference_cost_of_probe_elites": selected_cost,
        "reference_cost_regret": selected_cost - ref_cost,
        "probe_elite_reference_costs": selected_reference_costs.detach().cpu().tolist(),
        "probe_elite_reference_ranks_1based": selected_ranks.detach().cpu().tolist(),
        "probe_elite_best_reference_rank_1based": int(selected_ranks.min().detach().cpu()),
        "probe_elite_mean_reference_rank_1based": float(selected_ranks.float().mean().detach().cpu()),
        "probe_elite_best_reference_cost_gap": float(
            (selected_reference_costs.min() - reference_cost.min()).detach().cpu()
        ),
        "elite_action_mean_rms": _rms(probe_elite_mean - ref_elite_mean),
        "elite_first_action_rms": _rms(probe_elite_mean[0] - ref_elite_mean[0]),
        "reference_elite_indices": ref_idx.detach().cpu().tolist(),
        "probe_elite_indices": probe_idx.detach().cpu().tolist(),
    }


def _repeat_case(mapping: Mapping[str, Any], index: int, batch: int) -> dict[str, Any]:
    return {
        key: value[index : index + 1].expand((batch,) + tuple(value.shape[1:]))
        for key, value in mapping.items()
    }


def _teacher_cost(teacher: Any, context: Mapping[str, Any], actions: Any, goal: Mapping[str, Any], objective: Any) -> Any:
    from cache_core import rollout_from_encoded_obs

    predicted, _ = rollout_from_encoded_obs(teacher, context, actions)
    return objective(predicted, goal).reshape(-1)


def _student_cost(student: Any, context: Mapping[str, Any], actions: Any, goal: Mapping[str, Any], objective: Any) -> Any:
    predicted = student(context, actions)
    return objective(predicted, goal).reshape(-1)


def _validate_identity(freeze: Mapping[str, Any], parent: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    if freeze.get("schema") != "horizon-weighted-recurrent-student.cem-trace-diagnosis-freeze":
        raise ValueError("unexpected trace freeze schema")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unsupported trace freeze schema version")
    if parent.get("schema") != "horizon-weighted-recurrent-student.pusht-closed-loop-summary":
        raise ValueError("unexpected parent closed-loop summary schema")
    if str(freeze["parent_closed_loop_job"]) != "24544733.pbs101":
        raise ValueError("unexpected parent closed-loop job")
    indices = [int(value) for value in freeze["cases"]["indices"]]
    seeds = [int(value) for value in freeze["cases"]["eval_seeds"]]
    teacher_rows = parent["arms"]["official_dino_wm_teacher"]["episodes"]
    student_rows = parent["arms"]["horizon_weighted_recurrent_student"]["episodes"]
    observed = [
        index
        for index, (teacher, student) in enumerate(zip(teacher_rows, student_rows, strict=True))
        if bool(teacher["success"]) and not bool(student["success"])
    ]
    observed_seeds = [int(student_rows[index]["eval_seed"]) for index in observed]
    if indices != observed or seeds != observed_seeds:
        raise ValueError("frozen cases are not exactly the parent pilot's teacher-only successes")
    return indices, seeds


def _aggregate(records: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = (
        ("shared_pool", "spearman"),
        ("shared_pool", "topk_overlap"),
        ("shared_pool", "reference_cost_regret"),
        ("student_pool_shadow", "spearman"),
        ("student_pool_shadow", "topk_overlap"),
        ("student_pool_shadow", "reference_cost_regret"),
        ("student_pool_shadow", "probe_elite_best_reference_rank_1based"),
        ("student_pool_shadow", "probe_elite_mean_reference_rank_1based"),
        ("post_update", "mu_rms"),
        ("post_update", "sigma_rms"),
        ("post_update", "first_action_rms"),
    )
    for checkpoint in checkpoints:
        rows = [record["snapshots"][str(checkpoint)] for record in records]
        summary: dict[str, Any] = {}
        for group, field in fields:
            values = [float(row[group][field]) for row in rows]
            summary[f"{group}.{field}"] = {
                "mean": float(sum(values) / len(values)),
                "median": float(statistics.median(values)),
                "minimum": float(min(values)),
                "maximum": float(max(values)),
            }
        result[str(checkpoint)] = summary
    return result


def main() -> int:
    args = _args()
    _require_compute_node()
    import torch
    from omegaconf import OmegaConf

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    parent = _load_json(args.parent_summary.resolve())
    case_indices, case_seeds = _validate_identity(freeze, parent)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_dino_pusht_closed_loop as closed_loop

    teacher, student_wm, model_cfg, dataset = closed_loop._load_models(args, freeze)
    student = student_wm.student
    source = args.root.resolve() / "source"
    sys.path.insert(0, str(source))
    from planning.objectives import create_objective_fn
    from preprocessor import Preprocessor
    from utils import move_to_device

    preprocessor = Preprocessor(
        action_mean=dataset.action_mean,
        action_std=dataset.action_std,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        proprio_mean=dataset.proprio_mean,
        proprio_std=dataset.proprio_std,
        transform=dataset.transform,
    )
    with args.plan_targets.resolve().open("rb") as handle:
        targets = pickle.load(handle)
    if not isinstance(targets, dict) or int(targets.get("goal_H", -1)) != 5:
        raise ValueError("unexpected plan_targets payload")
    obs_0 = targets["obs_0"]
    obs_g = targets["obs_g"]
    if int(obs_0["visual"].shape[0]) != 8:
        raise ValueError("parent plan_targets must contain the frozen eight cases")
    device = torch.device("cuda:0")
    transformed_obs = move_to_device(preprocessor.transform_obs(obs_0), device)
    transformed_goal = move_to_device(preprocessor.transform_obs(obs_g), device)
    with torch.no_grad():
        encoded_obs = teacher.encode_obs(transformed_obs)
        encoded_goal = teacher.encode_obs(transformed_goal)

    cem = freeze["cem"]
    iterations = int(cem["iterations"])
    checkpoints = [int(value) for value in cem["checkpoints"]]
    samples = int(cem["num_samples"])
    topk = int(cem["topk"])
    horizon = int(cem["horizon"])
    action_dim = int(cem["packed_action_dim"])
    if action_dim != int(dataset.action_dim * model_cfg.frameskip):
        raise ValueError("packed action dimension differs from official dataset contract")
    objective = create_objective_fn(alpha=1, base=2, mode="last")
    records: list[dict[str, Any]] = []

    for case_index, eval_seed in zip(case_indices, case_seeds, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(int(cem["innovation_seed_base"]) + case_index)
        innovations = torch.randn((iterations, samples, horizon, action_dim), generator=generator)
        context = _repeat_case(encoded_obs, case_index, samples)
        goal = _repeat_case(encoded_goal, case_index, samples)
        teacher_mu = torch.zeros((horizon, action_dim), device=device)
        student_mu = torch.zeros_like(teacher_mu)
        teacher_sigma = torch.ones_like(teacher_mu)
        student_sigma = torch.ones_like(teacher_mu)
        snapshots: dict[str, Any] = {}

        for iteration_index in range(iterations):
            epsilon = innovations[iteration_index].to(device)
            teacher_actions = teacher_mu.unsqueeze(0) + teacher_sigma.unsqueeze(0) * epsilon
            student_actions = student_mu.unsqueeze(0) + student_sigma.unsqueeze(0) * epsilon
            teacher_actions[0] = teacher_mu
            student_actions[0] = student_mu
            if not torch.equal(teacher_actions[0], teacher_mu) or not torch.equal(student_actions[0], student_mu):
                raise RuntimeError("candidate zero must equal the pre-update mean")
            if not torch.isfinite(teacher_actions).all() or not torch.isfinite(student_actions).all():
                raise RuntimeError("non-finite CEM candidate")
            with torch.no_grad():
                teacher_own_cost = _teacher_cost(teacher, context, teacher_actions, goal, objective)
                student_own_cost = _student_cost(student, context, student_actions, goal, objective)
            if not torch.isfinite(teacher_own_cost).all() or not torch.isfinite(student_own_cost).all():
                raise RuntimeError("non-finite native objective cost")

            teacher_idx = teacher_own_cost.argsort()[:topk]
            student_idx = student_own_cost.argsort()[:topk]
            teacher_mu_next = teacher_actions.index_select(0, teacher_idx).mean(dim=0)
            student_mu_next = student_actions.index_select(0, student_idx).mean(dim=0)
            teacher_sigma_next = teacher_actions.index_select(0, teacher_idx).std(dim=0)
            student_sigma_next = student_actions.index_select(0, student_idx).std(dim=0)
            iteration = iteration_index + 1

            if iteration in checkpoints:
                identical_pool = bool(torch.equal(teacher_actions, student_actions))
                if iteration == 1 and not identical_pool:
                    raise RuntimeError("iteration-one candidate pools must be identical")
                with torch.no_grad():
                    student_on_teacher = student_own_cost if identical_pool else _student_cost(student, context, teacher_actions, goal, objective)
                    teacher_on_student = teacher_own_cost if identical_pool else _teacher_cost(teacher, context, student_actions, goal, objective)
                snapshots[str(iteration)] = {
                    "iteration": iteration,
                    "identical_teacher_student_pool": identical_pool,
                    "pre_update": {
                        "teacher_mu": teacher_mu.detach().cpu().tolist(),
                        "student_mu": student_mu.detach().cpu().tolist(),
                        "teacher_sigma": teacher_sigma.detach().cpu().tolist(),
                        "student_sigma": student_sigma.detach().cpu().tolist(),
                    },
                    "shared_pool": {
                        **_pool_metrics(teacher_own_cost, student_on_teacher, teacher_actions, topk),
                        "teacher_costs": teacher_own_cost.detach().cpu().tolist(),
                        "student_costs": student_on_teacher.detach().cpu().tolist(),
                    },
                    "student_pool_shadow": {
                        **_pool_metrics(teacher_on_student, student_own_cost, student_actions, topk),
                        "teacher_shadow_costs": teacher_on_student.detach().cpu().tolist(),
                        "student_costs": student_own_cost.detach().cpu().tolist(),
                    },
                    "post_update": {
                        "mu_rms": _rms(student_mu_next - teacher_mu_next),
                        "sigma_rms": _rms(student_sigma_next - teacher_sigma_next),
                        "first_action_rms": _rms(student_mu_next[0] - teacher_mu_next[0]),
                        "teacher_first_action": teacher_mu_next[0].detach().cpu().tolist(),
                        "student_first_action": student_mu_next[0].detach().cpu().tolist(),
                        "teacher_mu": teacher_mu_next.detach().cpu().tolist(),
                        "student_mu": student_mu_next.detach().cpu().tolist(),
                        "teacher_sigma": teacher_sigma_next.detach().cpu().tolist(),
                        "student_sigma": student_sigma_next.detach().cpu().tolist(),
                    },
                }

            teacher_mu, student_mu = teacher_mu_next, student_mu_next
            teacher_sigma, student_sigma = teacher_sigma_next, student_sigma_next

        record = {
            "case_index": case_index,
            "eval_seed": eval_seed,
            "innovation_seed": int(cem["innovation_seed_base"]) + case_index,
            "snapshots": snapshots,
        }
        records.append(record)
        (output / f"case_{case_index:02d}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"case_index": case_index, "eval_seed": eval_seed, "completed": True}))

    aggregate = _aggregate(records, checkpoints)
    thresholds = freeze["diagnostic_thresholds"]
    first = aggregate[str(checkpoints[0])]
    final = aggregate[str(checkpoints[-1])]
    direct_mismatch = (
        first["shared_pool.spearman"]["median"] < float(thresholds["shared_pool_spearman_min"])
        or first["shared_pool.topk_overlap"]["median"] < float(thresholds["shared_pool_top30_min"])
    )
    first_action_initial = first["post_update.first_action_rms"]["median"]
    first_action_final = final["post_update.first_action_rms"]["median"]
    amplification = (
        first_action_final >= float(thresholds["iter30_first_action_rms_floor"])
        and first_action_final >= float(thresholds["iter30_to_iter1_first_action_rms_ratio_min"]) * max(first_action_initial, 1e-12)
    )
    shadow_mismatch = final["student_pool_shadow.topk_overlap"]["median"] < float(thresholds["shadow_pool_top30_min"])
    decisions = {
        "direct_initial_pool_scoring_mismatch": "SUPPORTED" if direct_mismatch else "NOT_SUPPORTED",
        "iterative_cem_amplification": "SUPPORTED" if amplification else "NOT_SUPPORTED",
        "student_distribution_shadow_misranking_at_iter30": "SUPPORTED" if shadow_mismatch else "NOT_SUPPORTED",
        "spatial_mean_pooling_causal_attribution": "UNTESTED",
        "closed_loop_ood_causal_attribution": "UNTESTED",
    }
    summary = {
        "schema": "horizon-weighted-recurrent-student.cem-trace-diagnosis-summary",
        "schema_version": 1,
        "freeze": str(args.freeze.resolve()),
        "parent_summary": str(args.parent_summary.resolve()),
        "plan_targets": str(args.plan_targets.resolve()),
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "protocol": {
            "closed_loop": False,
            "diagnostic_only": True,
            "fixed_observation": "first MPC observation from the parent paired pilot",
            "candidate_coupling": "common_standard_normal_per_case_iteration",
            "official_cem_semantics": "plain torch.argsort; torch.std unbiased=True; candidate zero equals pre-update mean",
        },
        "cases": records,
        "aggregate": aggregate,
        "decisions": decisions,
        "claim_boundary": freeze["claim_boundary"],
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "cem_trace_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "decisions": decisions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
