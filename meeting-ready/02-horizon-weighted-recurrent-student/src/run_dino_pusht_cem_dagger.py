#!/usr/bin/env python3
"""Offline CEM-DAgger fine-tuning and fixed-observation predictor diagnosis.

The collector runs CEM only on frozen train-split native observation contexts;
it never constructs an environment or executes an action.  The teacher labels
the warm-start student's selected proposal rows, two recurrent students are
fine-tuned from the same horizon-weighted checkpoint, and evaluation is limited
to predictor-level held-out ranking plus the fixed-observation CEM trace.
"""

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


HORIZON = 5
WIDTH = 256
LABEL_CHECKPOINTS = (1, 5, 10, 20, 30)
STAGE2_CHECKPOINTS = (1, 5, 10, 30)
STAGE2_CASES = (0, 1, 2, 4, 5, 7)
STAGE2_SEEDS = (1, 100, 199, 397, 496, 694)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--plan-targets", type=Path, required=True)
    parser.add_argument("--historical-cem-summary", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing execution outside PBS")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and Path(nodefile).exists():
        nodes = {line.strip().lower() for line in Path(nodefile).read_text().splitlines() if line.strip()}
        short_host = host.split(".", 1)[0]
        short_nodes = {node.split(".", 1)[0] for node in nodes}
        if nodes and host not in nodes and short_host not in short_nodes:
            raise RuntimeError(f"current host {host} is not in PBS_NODEFILE allocation")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _assert_finite(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite {label}")


def _select_top_with_zero(cost: Any, count: int) -> Any:
    """Plain argsort top-M with candidate zero retained exactly once."""

    import torch

    order = cost.argsort()
    if bool((order[:count] == 0).any()):
        selected = order[:count]
    else:
        selected = torch.cat((order.new_tensor([0]), order[order != 0][: count - 1]))
    if selected.numel() != count or len(set(selected.detach().cpu().tolist())) != count:
        raise RuntimeError("CEM selected rows do not have exact cardinality")
    if not bool((selected == 0).any()):
        raise RuntimeError("CEM selected rows lost candidate zero")
    return selected


def _update_distribution(actions: Any, indices: Any) -> tuple[Any, Any]:
    elites = actions.index_select(0, indices)
    return elites.mean(dim=0), elites.std(dim=0)


def _weighted_horizon_mse(prediction: Mapping[str, Any], target: Mapping[str, Any], weights: Sequence[float]) -> Any:
    import torch
    import torch.nn.functional as F

    values = []
    for index in range(HORIZON):
        values.append(0.5 * (
            F.mse_loss(prediction["visual"][:, index], target["visual"][:, index])
            + F.mse_loss(prediction["proprio"][:, index], target["proprio"][:, index])
        ))
    per_horizon = torch.stack(values)
    weight = torch.as_tensor(list(weights), dtype=per_horizon.dtype, device=per_horizon.device)
    return (per_horizon * weight).mean()


def _concat_mapping(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    return {key: torch.cat((first[key], second[key]), dim=0) for key in first}


def _validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "horizon-weighted-recurrent-student.cem-dagger-freeze":
        raise ValueError("unexpected CEM-DAgger freeze schema")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unsupported CEM-DAgger freeze schema version")
    scope = freeze["scope"]
    collector = freeze["collector"]
    training = freeze["training"]
    stage1 = freeze["stage1_predictor_gate"]
    stage2 = freeze["stage2_fixed_observation_gate"]
    if bool(scope["closed_loop"]) or bool(scope["environment_interaction"]):
        raise ValueError("CEM-DAgger must remain offline and closed-loop free")
    if int(collector["episode_count"]) != 8 or int(collector["contexts_per_episode"]) != 1:
        raise ValueError("collector must use exactly 8 episodes and one context per episode")
    if [int(value) for value in collector["label_checkpoints"]] != list(LABEL_CHECKPOINTS):
        raise ValueError("collector label checkpoints drifted")
    if int(collector["cem_iterations"]) != 30 or int(collector["candidate_count_M"]) != 300 or int(collector["elite_count_K"]) != 30:
        raise ValueError("collector CEM dimensions drifted")
    if int(collector["student_label_top_m"]) != 120 or int(collector["expected_label_rows"]) != 4800:
        raise ValueError("collector label cardinality drifted")
    if int(training["updates"]) != 500 or int(training["formal_snapshot"]) != 500:
        raise ValueError("CEM-DAgger formal training endpoint must be step 500")
    if int(training["treatment_dagger_rows"]) != 16 or int(training["treatment_replay_rows"]) != 16 or int(training["control_replay_rows"]) != 32:
        raise ValueError("CEM-DAgger arm batch sizes drifted")
    if [float(value) for value in training["horizon_weights"]] != [1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3]:
        raise ValueError("CEM-DAgger horizon weights drifted")
    if bool(training["rank_loss"]) or bool(training["teacher_forcing"]) or bool(training["architecture_change"]):
        raise ValueError("CEM-DAgger must not change loss, teacher forcing, or architecture")
    if [int(value) for value in stage2["cases"]] != list(STAGE2_CASES) or [int(value) for value in stage2["eval_seeds"]] != list(STAGE2_SEEDS):
        raise ValueError("fixed-observation cases or seeds drifted")
    if int(stage2["cem_iterations"]) != 30 or int(stage2["candidate_count_M"]) != 300 or int(stage2["elite_count_K"]) != 30 or [int(value) for value in stage2["checkpoints"]] != list(STAGE2_CHECKPOINTS):
        raise ValueError("fixed-observation CEM dimensions drifted")
    if bool(stage2["closed_loop"]):
        raise ValueError("stage2 must remain fixed-observation only")
    return {
        "collector": collector,
        "training": training,
        "stage1": stage1,
        "stage2": stage2,
        "warm_start": freeze["warm_start"],
        "claim_boundary": freeze["claim_boundary"],
    }


def _normalize_manifest_context(raw: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    item = dict(raw)
    if "start_step" not in item and "start" in item:
        item["start_step"] = item["start"]
    if "target_steps" not in item:
        raise ValueError(f"{split_name} manifest context is missing target_steps")
    steps = item["target_steps"]
    if isinstance(steps, str):
        steps = steps.replace(",", " ").split()
    item["target_steps"] = [int(value) for value in steps]
    item.setdefault("split", split_name)
    item.setdefault("episode_key", f"{split_name}:episode_{int(item['episode_id']):03d}")
    return item


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    schema = manifest.get("schema")
    if schema not in {
        "jepa-action-prefix-compiler.grounded-prefix-manifest",
        "jepa-action-prefix-compiler.context-density-manifest",
    }:
        raise ValueError("unexpected train manifest schema")
    protocol = manifest.get("protocol", {})
    if int(protocol.get("horizon", -1)) != HORIZON or int(protocol.get("frame_skip", -1)) != 5:
        raise ValueError("train manifest must use H=5 and frame_skip=5")
    train = manifest.get("splits", {}).get("train", {})
    heldout = manifest.get("splits", {}).get("heldout", {})
    field = "examples" if schema == "jepa-action-prefix-compiler.grounded-prefix-manifest" else "contexts"
    train_examples = [_normalize_manifest_context(item, "train") for item in train.get(field, [])]
    heldout_examples = [_normalize_manifest_context(item, "heldout") for item in heldout.get(field, [])]
    if len(train_examples) != 256 or len(heldout_examples) != 8:
        raise ValueError("CEM-DAgger requires the exact 256-train/8-heldout manifest")
    train_keys = {str(item["episode_key"]) for item in train_examples}
    heldout_keys = {str(item["episode_key"]) for item in heldout_examples}
    if train_keys.intersection(heldout_keys):
        raise ValueError("train and heldout episode sets overlap")
    train_counts: dict[str, int] = {}
    for item in train_examples:
        train_counts[str(item["episode_key"])] = train_counts.get(str(item["episode_key"]), 0) + 1
    if len(train_counts) != 32 or set(train_counts.values()) != {8}:
        raise ValueError("CEM-DAgger requires 32 train episodes with exactly eight contexts each")
    if len(heldout_keys) != 8:
        raise ValueError("CEM-DAgger requires eight distinct heldout episodes")
    for example in list(train_examples) + list(heldout_examples):
        start = int(example["start_step"])
        expected = [start + (index + 1) * 5 for index in range(HORIZON)]
        if [int(value) for value in example["target_steps"]] != expected:
            raise ValueError("manifest target steps are not contiguous")
    return {"train": train_examples, "heldout": heldout_examples}


def _teacher_cost(teacher: Any, context: Mapping[str, Any], actions: Any, goal: Mapping[str, Any], objective: Any) -> Any:
    from cache_core import rollout_from_encoded_obs

    predicted, _ = rollout_from_encoded_obs(teacher, context, actions)
    return objective(predicted, goal).reshape(-1)


def _student_cost(student: Any, context: Mapping[str, Any], actions: Any, goal: Mapping[str, Any], objective: Any) -> Any:
    predicted = student(context, actions)
    return objective(predicted, goal).reshape(-1)


def _cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _repeat_one(mapping: Mapping[str, Any], index: int, count: int, device: Any) -> dict[str, Any]:
    return {
        key: value[index : index + 1].expand((count,) + tuple(value.shape[1:])).to(device)
        for key, value in mapping.items()
    }


def _collect_labels(
    torch: Any,
    teacher: Any,
    warm_student: Any,
    train_context: Mapping[str, Any],
    train_goal: Mapping[str, Any],
    episode_ids: Sequence[int],
    objective: Any,
    collector: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels_context: list[Any] = []
    labels_actions: list[Any] = []
    labels_visual: list[Any] = []
    labels_proprio: list[Any] = []
    labels_cost: list[Any] = []
    labels_context_index: list[Any] = []
    labels_case_index: list[Any] = []
    labels_episode_id: list[Any] = []
    labels_iteration: list[Any] = []
    labels_candidate_index: list[Any] = []
    case_records: list[dict[str, Any]] = []
    samples = int(collector["candidate_count_M"])
    topk = int(collector["elite_count_K"])
    label_top_m = int(collector["student_label_top_m"])
    iterations = int(collector["cem_iterations"])
    for context_index, episode_id in enumerate(episode_ids):
        generator = torch.Generator(device="cpu").manual_seed(int(collector["innovation_seed_base"]) + 1009 * context_index)
        innovations = torch.randn((iterations, samples, HORIZON, 10), generator=generator)
        context = _repeat_one(train_context, context_index, samples, device)
        goal = _repeat_one(train_goal, context_index, samples, device)
        mu = torch.zeros((HORIZON, 10), device=device)
        sigma = torch.ones_like(mu)
        snapshots: dict[str, Any] = {}
        for iteration_index in range(iterations):
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * innovations[iteration_index].to(device)
            actions[0] = mu
            _assert_finite(torch, actions, f"collector actions episode {episode_id} iteration {iteration_index + 1}")
            with torch.no_grad():
                student_cost = _student_cost(warm_student, context, actions, goal, objective)
            _assert_finite(torch, student_cost, "collector student cost")
            order = student_cost.argsort()
            elites = order[:topk]
            mu_next, sigma_next = _update_distribution(actions, elites)
            _assert_finite(torch, mu_next, "collector mu")
            _assert_finite(torch, sigma_next, "collector sigma")
            iteration = iteration_index + 1
            if iteration in LABEL_CHECKPOINTS:
                selected = _select_top_with_zero(student_cost, label_top_m)
                selected_actions = actions.index_select(0, selected)
                selected_context = _repeat_one(train_context, context_index, label_top_m, device)
                with torch.no_grad():
                    teacher_target = __import__("run_dino_pusht_stage_a", fromlist=["_teacher_targets"])._teacher_targets(
                        teacher, selected_context, selected_actions
                    )
                _assert_finite(torch, teacher_target["visual"], "collector teacher visual labels")
                _assert_finite(torch, teacher_target["proprio"], "collector teacher proprio labels")
                selected_goal = _repeat_one(train_goal, context_index, label_top_m, device)
                with torch.no_grad():
                    teacher_cost = objective(teacher_target, selected_goal).reshape(-1)
                _assert_finite(torch, teacher_cost, "collector teacher label cost")
                labels_actions.append(selected_actions.detach().cpu())
                labels_visual.append(teacher_target["visual"].detach().cpu())
                labels_proprio.append(teacher_target["proprio"].detach().cpu())
                labels_cost.append(teacher_cost.detach().cpu())
                labels_context_index.append(torch.full((label_top_m,), context_index, dtype=torch.long))
                labels_case_index.append(torch.full((label_top_m,), context_index, dtype=torch.long))
                labels_episode_id.append(torch.full((label_top_m,), int(episode_id), dtype=torch.long))
                labels_iteration.append(torch.full((label_top_m,), iteration, dtype=torch.long))
                labels_candidate_index.append(selected.detach().cpu().to(dtype=torch.long))
                snapshots[str(iteration)] = {
                    "iteration": iteration,
                    "student_top120_indices": selected.detach().cpu().tolist(),
                    "student_top30_indices": elites.detach().cpu().tolist(),
                    "label_rows": int(selected.numel()),
                    "pre_update_mu": mu.detach().cpu().tolist(),
                    "post_update_mu": mu_next.detach().cpu().tolist(),
                }
            mu, sigma = mu_next, sigma_next
        case_records.append({"context_index": context_index, "episode_id": int(episode_id), "snapshots": snapshots})
    dataset = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-cem-dagger-labels",
        "schema_version": 1,
        "context_visual": train_context["visual"].detach().cpu(),
        "context_proprio": train_context["proprio"].detach().cpu(),
        "goal_visual": train_goal["visual"].detach().cpu(),
        "goal_proprio": train_goal["proprio"].detach().cpu(),
        "context_index": torch.cat(labels_context_index),
        "label_case_index": torch.cat(labels_case_index),
        "label_episode_id": torch.cat(labels_episode_id),
        "label_iteration": torch.cat(labels_iteration),
        "label_candidate_index": torch.cat(labels_candidate_index),
        "actions": torch.cat(labels_actions),
        "target_visual": torch.cat(labels_visual),
        "target_proprio": torch.cat(labels_proprio),
        "teacher_cost": torch.cat(labels_cost),
        "episode_ids": [int(value) for value in episode_ids],
        "label_checkpoints": list(LABEL_CHECKPOINTS),
    }
    if int(dataset["actions"].shape[0]) != int(collector["expected_label_rows"]):
        raise RuntimeError("CEM-DAgger label row count is not 4800")
    return dataset, case_records


def _materialize_dagger_batch(dataset: Mapping[str, Any], indices: Any, device: Any) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    context_index = dataset["context_index"].index_select(0, indices.cpu())
    context = {
        "visual": dataset["context_visual"].index_select(0, context_index).to(device, non_blocking=True),
        "proprio": dataset["context_proprio"].index_select(0, context_index).to(device, non_blocking=True),
    }
    actions = dataset["actions"].index_select(0, indices.cpu()).to(device, non_blocking=True)
    target = {
        "visual": dataset["target_visual"].index_select(0, indices.cpu()).to(device, non_blocking=True),
        "proprio": dataset["target_proprio"].index_select(0, indices.cpu()).to(device, non_blocking=True),
    }
    return context, actions, target


def _evaluate_predictor(
    torch: Any,
    models: Mapping[str, Any],
    teacher: Any,
    heldout_encoded: Mapping[str, Any],
    objective: Any,
    eval_seeds: Sequence[int],
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    import run_dino_pusht_optimization_length as opt
    import run_dino_pusht_stage_a as stage_a

    blocks = opt._prepare_eval_blocks(heldout_encoded, 8, eval_seeds, batch_size, 10, device)
    result: dict[str, Any] = {}
    for name, model in models.items():
        model.eval()
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for block in blocks:
                target = stage_a._teacher_targets(teacher, block["context"], block["actions"])
                prediction = model(block["context"], block["actions"])
                teacher_cost = objective(target, block["goal"])
                student_cost = objective(prediction, block["goal"])
                _assert_finite(torch, prediction["visual"], f"{name} predictor visual")
                _assert_finite(torch, prediction["proprio"], f"{name} predictor proprio")
                ranking = {
                    "spearman": stage_a._spearman(teacher_cost, student_cost),
                    "top30_overlap": stage_a._topk_overlap(teacher_cost, student_cost),
                }
                rows.append({
                    "seed": int(block["seed"]),
                    "context_index": int(block["context_index"]),
                    "candidate_count": int(batch_size),
                    "topk": 30,
                    "ranking": ranking,
                    **stage_a._metrics_by_horizon(prediction, target),
                })
        spearman = [float(row["ranking"]["spearman"]) for row in rows]
        overlap = [float(row["ranking"]["top30_overlap"]) for row in rows]
        result[name] = {
            "per_block": rows,
            "spearman": {"median": _median(spearman), "mean": _mean(spearman), "minimum": min(spearman)},
            "top30_overlap": {"median": _median(overlap), "mean": _mean(overlap), "minimum": min(overlap)},
        }
        result[name]["causality"] = __import__("run_dino_pusht_recurrent_student", fromlist=["_leakage_test"])._leakage_test(
            model,
            {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()},
            10,
            int(eval_seeds[0]) + 9000,
            device,
            1e-6,
        )
    return result


def _historical_reference(path: Path, parent: Mapping[str, Any]) -> dict[str, Any]:
    historical = _load_json(path)
    if historical.get("schema") != "horizon-weighted-recurrent-student.cem-trace-diagnosis-summary":
        raise ValueError("historical CEM reference has unexpected schema")
    episodes_teacher = parent["arms"]["official_dino_wm_teacher"]["episodes"]
    episodes_student = parent["arms"]["horizon_weighted_recurrent_student"]["episodes"]
    if [int(item["case_index"]) for item in historical["cases"]] != list(STAGE2_CASES):
        raise ValueError("historical CEM reference cases differ")
    observed = [
        index for index, (teacher, student) in enumerate(zip(episodes_teacher, episodes_student, strict=True))
        if bool(teacher["success"]) and not bool(student["success"])
    ]
    if observed != list(STAGE2_CASES):
        raise ValueError("parent summary no longer identifies the frozen six failed cases")
    aggregate = historical["aggregate"]["30"]["student_pool_shadow.reference_cost_regret"]["median"]
    per_case_regret = {}
    teacher_mu = {}
    for case in historical["cases"]:
        index = int(case["case_index"])
        snapshot = case["snapshots"]
        per_case_regret[str(index)] = float(snapshot["30"]["student_pool_shadow"]["reference_cost_regret"])
        teacher_mu[str(index)] = {
            str(checkpoint): snapshot[str(checkpoint)]["post_update"]["teacher_mu"]
            for checkpoint in (1, 5, 10, 30)
        }
    return {
        "summary_path": str(path.resolve()),
        "schema": historical["schema"],
        "student_only_regret_median_iter30": float(aggregate),
        "student_only_regret_by_case_iter30": per_case_regret,
        "teacher_full_reference_mu_by_case": teacher_mu,
        "historical_checkpoints_with_teacher_mu": [1, 5, 10, 30],
        "historical_teacher_full_rerun": False,
    }


def _run_fixed_arm(
    torch: Any,
    model: Any,
    teacher: Any,
    context: Mapping[str, Any],
    goal: Mapping[str, Any],
    innovations: Any,
    objective: Any,
    checkpoints: Sequence[int],
    historical: Mapping[str, Any],
    case_index: int,
    device: Any,
) -> dict[str, Any]:
    model.eval()
    mu = torch.zeros((HORIZON, 10), device=device)
    sigma = torch.ones_like(mu)
    snapshots: dict[str, Any] = {}
    for iteration_index, epsilon_cpu in enumerate(innovations, start=1):
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon_cpu.to(device)
        actions[0] = mu
        _assert_finite(torch, actions, f"fixed CEM {case_index} actions")
        with torch.no_grad():
            student_cost = _student_cost(model, context, actions, goal, objective)
        _assert_finite(torch, student_cost, f"fixed CEM {case_index} student cost")
        student_order = student_cost.argsort()
        selected = student_order[:30]
        mu_next, sigma_next = _update_distribution(actions, selected)
        _assert_finite(torch, mu_next, "fixed CEM updated mu")
        _assert_finite(torch, sigma_next, "fixed CEM updated sigma")
        if iteration_index in checkpoints:
            with torch.no_grad():
                teacher_cost = _teacher_cost(teacher, context, actions, goal, objective)
            _assert_finite(torch, teacher_cost, f"fixed CEM {case_index} teacher shadow cost")
            teacher_order = teacher_cost.argsort()
            teacher_top30 = teacher_order[:30]
            selected_regret = float(
                teacher_cost.index_select(0, selected).mean().detach().cpu()
                - teacher_cost.index_select(0, teacher_top30).mean().detach().cpu()
            )
            reference_mu = historical.get("teacher_full_reference_mu_by_case", {}).get(str(case_index), {}).get(str(iteration_index))
            if reference_mu is None:
                drift = None
                coordinate = None
            else:
                reference = torch.as_tensor(reference_mu, dtype=mu_next.dtype, device=device)
                drift_value = mu_next[0] - reference[0]
                drift = float(drift_value.square().mean().sqrt().detach().cpu())
                coordinate = float(drift_value.abs().max().detach().cpu())
            snapshots[str(iteration_index)] = {
                "iteration": iteration_index,
                "selected_top30_overlap_with_teacher_shadow": float(len(set(selected.detach().cpu().tolist()).intersection(set(teacher_top30.detach().cpu().tolist()))) / 30),
                "teacher_cost_regret": selected_regret,
                "first_action_rms_drift": drift,
                "first_action_coordinate_abs_max": coordinate,
                "pre_update_mu": mu.detach().cpu().tolist(),
                "post_update_mu": mu_next.detach().cpu().tolist(),
                "post_update_sigma": sigma_next.detach().cpu().tolist(),
                "selected_indices": selected.detach().cpu().tolist(),
                "teacher_shadow_top30_indices": teacher_top30.detach().cpu().tolist(),
                "teacher_shadow_calls": 1,
                "teacher_shadow_candidates": 300,
            }
        mu, sigma = mu_next, sigma_next
    return {"snapshots": snapshots, "all_rounds_complete": True, "all_outputs_finite": True, "silent_fallback": False}


def _aggregate_fixed(records: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for checkpoint in checkpoints:
        rows = [record["snapshots"][str(checkpoint)] for record in records]
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in rows if row[key] is not None]
        drift_values = values("first_action_rms_drift")
        coordinate_values = values("first_action_coordinate_abs_max")
        result[str(checkpoint)] = {
            "first_action_rms_drift": None if not drift_values else {"median": _median(drift_values), "mean": _mean(drift_values)},
            "first_action_coordinate_abs_max": None if not coordinate_values else {"maximum": max(coordinate_values)},
            "teacher_cost_regret": {"median": _median(values("teacher_cost_regret")), "mean": _mean(values("teacher_cost_regret"))},
            "selected_top30_overlap_with_teacher_shadow": {"median": _median(values("selected_top30_overlap_with_teacher_shadow"))},
        }
    return result


def main() -> int:
    args = _args()
    _require_compute_node()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    protocol = args.protocol.read_text(encoding="utf-8")
    if "CEM-DAgger" not in protocol or "offline train-split" not in protocol:
        raise ValueError("protocol does not describe the frozen offline CEM-DAgger boundary")
    settings = _validate_freeze(freeze)
    manifest = _validate_manifest(_load_json(args.manifest.resolve()))
    parent = _load_json(args.parent_summary.resolve())
    if parent.get("schema") != "horizon-weighted-recurrent-student.pusht-closed-loop-summary":
        raise ValueError("unexpected parent closed-loop summary schema")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_dino_pusht_closed_loop as closed_loop
    import run_dino_pusht_grounded_prefix as grounded
    import run_dino_pusht_query_slate_rank as slate
    import run_dino_pusht_recurrent_student as recurrent
    import run_dino_pusht_stage_a as stage_a
    import run_dino_pusht_optimization_length as opt
    from planning.objectives import create_objective_fn

    loader_args = argparse.Namespace(
        root=args.root,
        student_checkpoint=args.student_checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        checkpoint_config=args.checkpoint_config,
        data_root=args.data_root,
    )
    runtime_freeze = {
        "student": {
            "checkpoint_schema": settings["warm_start"]["checkpoint_schema"],
            "step": int(settings["warm_start"]["step"]),
            "hidden_dim": WIDTH,
            "visual_dim": 384,
            "proprio_dim": 10,
            "packed_action_dim": 10,
        }
    }
    teacher, warm_world_model, _model_cfg, dataset = closed_loop._load_models(loader_args, runtime_freeze)
    warm_student = warm_world_model.student
    train_dset, heldout_dset = grounded._load_trajectory_datasets(
        args.root.resolve(), (args.asset_root or args.root).resolve(), runtime_freeze, args.student_checkpoint.resolve(), args.checkpoint_config.resolve() if args.checkpoint_config else None
    )
    device = torch.device("cuda:0")
    cache = grounded._RawEpisodeCache(8)
    train_examples = manifest["train"]
    heldout_examples = manifest["heldout"]
    train_encoded = grounded._preencode_manifest(train_dset, "train", train_examples, 5, 8, cache, teacher, device)
    heldout_encoded = grounded._preencode_manifest(heldout_dset, "heldout", heldout_examples, 5, 8, cache, teacher, device)
    selected_by_episode: dict[int, Mapping[str, Any]] = {}
    for example in train_examples:
        selected_by_episode.setdefault(int(example["episode_id"]), example)
    episode_ids = sorted(selected_by_episode)[: int(settings["collector"]["episode_count"])]
    if len(episode_ids) != 8:
        raise ValueError("train manifest has fewer than 8 distinct episodes")
    selected_examples = [selected_by_episode[episode_id] for episode_id in episode_ids]
    selected_context, _selected_actions, selected_grounded = grounded._segment_batch(
        train_dset, "train", selected_examples, list(range(8)), 5, cache, teacher, device
    )
    selected_goal = {key: value[:, -1:] for key, value in selected_grounded.items()}
    objective = create_objective_fn(alpha=1, base=2, mode="last")
    label_dataset, collector_cases = _collect_labels(
        torch, teacher, warm_student, selected_context, selected_goal, episode_ids, objective, settings["collector"], device
    )
    torch.save(label_dataset, output / "cem_dagger_dataset.pt")
    (output / "collector_cases.json").write_text(json.dumps(collector_cases, indent=2), encoding="utf-8")

    visual_dim = int(train_encoded["context"]["visual"].shape[-1])
    proprio_dim = int(train_encoded["context"]["proprio"].shape[-1])
    if visual_dim != 384 or proprio_dim != 10:
        raise ValueError("native latent dimensions drifted")
    warm_state = {key: value.detach().cpu().clone() for key, value in warm_student.state_dict().items()}
    torch.manual_seed(int(settings["training"]["initialization_seed"]))
    torch.cuda.manual_seed_all(int(settings["training"]["initialization_seed"]))
    treatment = recurrent._student(10, visual_dim, proprio_dim, WIDTH).to(device)
    replay_control = recurrent._student(10, visual_dim, proprio_dim, WIDTH).to(device)
    treatment.load_state_dict(warm_state, strict=True)
    replay_control.load_state_dict(warm_state, strict=True)
    treatment_opt = torch.optim.AdamW(treatment.parameters(), lr=float(settings["training"]["learning_rate"]), weight_decay=float(settings["training"]["weight_decay"]), betas=tuple(settings["training"]["betas"]), eps=float(settings["training"]["eps"]))
    control_opt = torch.optim.AdamW(replay_control.parameters(), lr=float(settings["training"]["learning_rate"]), weight_decay=float(settings["training"]["weight_decay"]), betas=tuple(settings["training"]["betas"]), eps=float(settings["training"]["eps"]))
    bank = slate._precompute_slate_bank(
        teacher,
        train_encoded,
        objective,
        10,
        int(settings["training"]["action_slate_seed"]),
        int(settings["training"]["cem_seed"]),
        device,
    )
    replay_schedule, replay_schedule_meta = slate._context_schedule(500, 8, int(settings["training"]["replay_schedule_seed"]))
    dagger_generator = torch.Generator(device="cpu").manual_seed(int(settings["training"]["dagger_sample_seed"]))
    dagger_schedule = [torch.randperm(int(label_dataset["actions"].shape[0]), generator=dagger_generator)[:16] for _ in range(500)]
    losses = {"treatment": [], "replay_control": []}
    for step_index in range(500):
        selected_contexts = replay_schedule[step_index]
        replay_treatment = slate._materialize_slate_batch(train_encoded, bank, selected_contexts[:4], device)
        replay_control_batch = slate._materialize_slate_batch(train_encoded, bank, selected_contexts, device)
        dagger_batch = _materialize_dagger_batch(label_dataset, dagger_schedule[step_index], device)
        treatment_context = _concat_mapping(dagger_batch[0], replay_treatment[0])
        treatment_actions = torch.cat((dagger_batch[1], replay_treatment[1]), dim=0)
        treatment_target = _concat_mapping(dagger_batch[2], replay_treatment[2])
        treatment.train()
        treatment_prediction = treatment(treatment_context, treatment_actions)
        treatment_loss = _weighted_horizon_mse(treatment_prediction, treatment_target, settings["training"]["horizon_weights"])
        treatment_opt.zero_grad(set_to_none=True)
        treatment_loss.backward()
        treatment_opt.step()
        replay_control.train()
        control_prediction = replay_control(replay_control_batch[0], replay_control_batch[1])
        control_loss = _weighted_horizon_mse(control_prediction, replay_control_batch[2], settings["training"]["horizon_weights"])
        control_opt.zero_grad(set_to_none=True)
        control_loss.backward()
        control_opt.step()
        _assert_finite(torch, treatment_loss.detach(), "treatment loss")
        _assert_finite(torch, control_loss.detach(), "replay control loss")
        losses["treatment"].append(float(treatment_loss.detach().cpu()))
        losses["replay_control"].append(float(control_loss.detach().cpu()))
    checkpoints = {
        "treatment": output / "cem_dagger_treatment_step0500.pt",
        "replay_control": output / "cem_dagger_replay_control_step0500.pt",
    }
    for name, model in (("treatment", treatment), ("replay_control", replay_control)):
        torch.save({
            "schema": "jepa-action-prefix-compiler.dino-pusht-cem-dagger-recurrent-checkpoint",
            "schema_version": 1,
            "step": 500,
            "arm": name,
            "architecture": "RecurrentNativeLatentTransitionStudent",
            "hidden_dim": WIDTH,
            "visual_dim": visual_dim,
            "proprio_dim": proprio_dim,
            "packed_action_dim": 10,
            "warm_start_checkpoint": str(args.student_checkpoint.resolve()),
            "label_dataset": str((output / "cem_dagger_dataset.pt").resolve()),
            "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        }, checkpoints[name])

    stage1 = _evaluate_predictor(torch, {"treatment": treatment, "replay_control": replay_control}, teacher, heldout_encoded, objective, [20264925, 20264926], 300, device)
    for name, model in (("treatment", treatment), ("replay_control", replay_control)):
        stage1[name]["finite_outputs"] = bool(_all_finite(stage1[name]))
        stage1[name]["silent_fallback"] = False
    treatment_gate = stage1["treatment"]
    treatment_causality_max = max(
        float(row["max_abs_delta"])
        for row in treatment_gate["causality"].get("cases", [])
    )
    control_gate = stage1["replay_control"]
    control_causality_max = max(
        float(row["max_abs_delta"])
        for row in control_gate["causality"].get("cases", [])
    )
    thresholds = {
        "spearman_median_min": float(settings["stage1"]["spearman_median_min"]),
        "spearman_minimum_min": float(settings["stage1"]["spearman_minimum_min"]),
        "top30_median_min": float(settings["stage1"]["top30_median_min"]),
        "top30_minimum_min": float(settings["stage1"]["top30_minimum_min"]),
        "causality_max_abs": float(settings["stage1"]["causality_max_abs"]),
    }
    def predictor_gate_passed(values: Mapping[str, Any]) -> bool:
        return bool(
            values["spearman"]["median"] >= thresholds["spearman_median_min"]
            and values["spearman"]["minimum"] >= thresholds["spearman_minimum_min"]
            and values["top30_overlap"]["median"] >= thresholds["top30_median_min"]
            and values["top30_overlap"]["minimum"] >= thresholds["top30_minimum_min"]
            and values["causality"].get("passed", False)
            and values["finite_outputs"]
            and not values["silent_fallback"]
        )
    stage1["treatment_gate"] = {
        "spearman_median": treatment_gate["spearman"]["median"],
        "spearman_minimum": treatment_gate["spearman"]["minimum"],
        "top30_median": treatment_gate["top30_overlap"]["median"],
        "top30_minimum": treatment_gate["top30_overlap"]["minimum"],
        "causality_max_abs": treatment_causality_max,
        "status": "PASS" if predictor_gate_passed(treatment_gate) else "FAIL",
        "thresholds": thresholds,
    }
    stage1["replay_control_gate"] = {
        "spearman_median": control_gate["spearman"]["median"],
        "spearman_minimum": control_gate["spearman"]["minimum"],
        "top30_median": control_gate["top30_overlap"]["median"],
        "top30_minimum": control_gate["top30_overlap"]["minimum"],
        "causality_max_abs": control_causality_max,
        "status": "PASS" if predictor_gate_passed(control_gate) else "FAIL",
        "thresholds": thresholds,
    }

    stage2: dict[str, Any] = {"status": "SKIPPED", "reason": "one or both stage1 arm gates failed"}
    historical = None
    if stage1["treatment_gate"]["status"] == "PASS" and stage1["replay_control_gate"]["status"] == "PASS":
        historical = _historical_reference(args.historical_cem_summary.resolve(), parent)
        with args.plan_targets.resolve().open("rb") as handle:
            targets = pickle.load(handle)
        if not isinstance(targets, dict) or int(targets.get("goal_H", -1)) != HORIZON:
            raise ValueError("unexpected fixed-observation plan_targets payload")
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
        obs = move_to_device(preprocessor.transform_obs(targets["obs_0"]), device)
        goal_obs = move_to_device(preprocessor.transform_obs(targets["obs_g"]), device)
        with torch.no_grad():
            encoded_obs = teacher.encode_obs(obs)
            encoded_goal = teacher.encode_obs(goal_obs)
        fixed_records = {"treatment": [], "replay_control": []}
        for case_index, eval_seed in zip(STAGE2_CASES, STAGE2_SEEDS, strict=True):
            generator = torch.Generator(device="cpu").manual_seed(20261100 + case_index)
            innovations = torch.randn((30, 300, HORIZON, 10), generator=generator)
            context = {key: value[case_index : case_index + 1].expand((300,) + tuple(value.shape[1:])) for key, value in encoded_obs.items()}
            goal = {key: value[case_index : case_index + 1].expand((300,) + tuple(value.shape[1:])) for key, value in encoded_goal.items()}
            case_payload = {"case_index": case_index, "eval_seed": eval_seed, "arms": {}}
            for name, model in (("treatment", treatment), ("replay_control", replay_control)):
                arm_record = _run_fixed_arm(torch, model, teacher, context, goal, innovations, objective, STAGE2_CHECKPOINTS, historical, case_index, device)
                fixed_records[name].append(arm_record)
                case_payload["arms"][name] = arm_record
            (output / f"case_{case_index:02d}.json").write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
        treatment_aggregate = _aggregate_fixed(fixed_records["treatment"], STAGE2_CHECKPOINTS)
        control_aggregate = _aggregate_fixed(fixed_records["replay_control"], STAGE2_CHECKPOINTS)
        final_treatment = treatment_aggregate["30"]
        old_regrets = [float(historical["student_only_regret_by_case_iter30"][str(case)]) for case in STAGE2_CASES]
        treatment_regrets = [float(record["snapshots"]["30"]["teacher_cost_regret"]) for record in fixed_records["treatment"]]
        treatment_drifts = [float(record["snapshots"]["30"]["first_action_rms_drift"]) for record in fixed_records["treatment"]]
        control_drifts = [float(record["snapshots"]["30"]["first_action_rms_drift"]) for record in fixed_records["replay_control"]]
        stage2 = {
            "status": "PASS" if (
                final_treatment["first_action_rms_drift"]["median"] <= 0.15
                and final_treatment["first_action_coordinate_abs_max"]["maximum"] <= 0.25
                and final_treatment["teacher_cost_regret"]["median"] <= float(historical["student_only_regret_median_iter30"])
                and sum(a <= b for a, b in zip(treatment_regrets, old_regrets, strict=True)) >= 4
                and _median(treatment_drifts) < _median(control_drifts)
                and sum(a <= b for a, b in zip(treatment_drifts, control_drifts, strict=True)) >= 4
            ) else "FAIL",
            "historical_reference": historical,
            "treatment": {"aggregate": treatment_aggregate, "cases": fixed_records["treatment"]},
            "replay_control": {"aggregate": control_aggregate, "cases": fixed_records["replay_control"]},
            "treatment_support_cases_not_worse_than_historical_student": int(sum(a <= b for a, b in zip(treatment_regrets, old_regrets, strict=True))),
            "treatment_vs_replay_control_drift_cases_not_worse": int(sum(a <= b for a, b in zip(treatment_drifts, control_drifts, strict=True))),
            "treatment_vs_replay_control_drift_median": {"treatment": _median(treatment_drifts), "replay_control": _median(control_drifts)},
            "claim_boundary": "fixed-observation mechanism diagnosis only",
        }

    label_tensors = (
        label_dataset["context_visual"], label_dataset["context_proprio"],
        label_dataset["actions"], label_dataset["target_visual"],
        label_dataset["target_proprio"], label_dataset["teacher_cost"],
    )
    label_outputs_finite = all(bool(torch.isfinite(value).all().item()) for value in label_tensors)
    candidate_cardinality = all(
        len(snapshot["student_top120_indices"]) == 120
        and 0 in snapshot["student_top120_indices"]
        for case in collector_cases
        for snapshot in case["snapshots"].values()
    )
    treatment_cases = stage2.get("treatment", {}).get("cases", []) if isinstance(stage2, Mapping) else []
    control_cases = stage2.get("replay_control", {}).get("cases", []) if isinstance(stage2, Mapping) else []
    all_six_cases_complete = (
        stage2.get("status") in {"PASS", "FAIL"}
        and len(treatment_cases) == len(STAGE2_CASES)
        and len(control_cases) == len(STAGE2_CASES)
    )

    summary = {
        "schema": "horizon-weighted-recurrent-student.cem-dagger-summary",
        "schema_version": 1,
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "manifest": str(args.manifest.resolve()),
        "warm_start_checkpoint": str(args.student_checkpoint.resolve()),
        "collector": {
            "episodes": collector_cases,
            "episode_ids": [int(value) for value in episode_ids],
            "label_rows": int(label_dataset["actions"].shape[0]),
            "label_checkpoints": list(LABEL_CHECKPOINTS),
            "environment_interaction": False,
            "excluded_eval_cases": list(STAGE2_CASES),
        },
        "training": {
            "updates": 500,
            "losses": losses,
            "formal_snapshot": 500,
            "replay_schedule": replay_schedule_meta,
            "arms": {"treatment": "16 CEM-DAgger + 16 exact old offline replay", "replay_control": "32 exact old offline replay"},
        },
        "checkpoints": {name: str(path.resolve()) for name, path in checkpoints.items()},
        "stage1_predictor_gate": stage1,
        "stage2_fixed_observation_gate": stage2,
        "validity": {
            "all_six_cases_complete": bool(all_six_cases_complete),
            "all_outputs_finite": bool(
                _all_finite(collector_cases)
                and label_outputs_finite
                and _all_finite(stage1)
                and _all_finite(stage2)
                and all(_all_finite(values) for values in losses.values())
            ),
            "silent_fallback": False,
            "environment_interaction": False,
            "label_rows_exact": int(label_dataset["actions"].shape[0]) == 4800,
            "superset_cardinality_correct": bool(candidate_cardinality),
            "candidate_zero_and_cardinality_checked": True,
        },
        "claim_boundary": settings["claim_boundary"],
        "gpu": torch.cuda.get_device_name(0),
    }
    summary_path = (args.summary or output / "cem_dagger_summary.json").resolve()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary_path != output / "cem_dagger_summary.json":
        (output / "cem_dagger_summary.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output": str(output), "stage1": stage1["treatment_gate"], "stage2": stage2["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
