#!/usr/bin/env python3
"""Grouped CEM-proposal score distillation on an existing DINO-WM label set.

The runner never collects new CEM labels.  It reuses the 4,800-row label set
from the frozen CEM-DAgger job, trains three matched recurrent students, and
stops at a predictor-level gate unless the preregistered Stage 1 conditions
pass.  Stage 2, when enabled, is the existing fixed-observation CEM trace only.
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
ARMS = ("latent_only", "score_distill", "shuffled_score")
LABEL_CHECKPOINTS = (1, 5, 10, 20, 30)
STAGE2_CASES = (0, 1, 2, 4, 5, 7)
STAGE2_SEEDS = (1, 100, 199, 397, 496, 694)
STAGE2_CHECKPOINTS = (1, 5, 10, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--parent-summary", type=Path)
    parser.add_argument("--plan-targets", type=Path)
    parser.add_argument("--historical-cem-summary", type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _assert_finite(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite {label}")


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing execution outside PBS")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_NODEFILE is required for compute-node guard")
    nodes = {line.strip().lower() for line in Path(nodefile).read_text().splitlines() if line.strip()}
    short_host = host.split(".", 1)[0]
    short_nodes = {node.split(".", 1)[0] for node in nodes}
    if nodes and host not in nodes and short_host not in short_nodes:
        raise RuntimeError(f"current host {host} is not in PBS_NODEFILE allocation")


def _validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "horizon-weighted-recurrent-student.cem-score-distill-freeze":
        raise ValueError("unexpected CEM score-distill freeze schema")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unsupported CEM score-distill freeze version")
    scope = freeze["scope"]
    source = freeze["source_dataset"]
    training = freeze["training"]
    stage1 = freeze["stage1_predictor_gate"]
    stage2 = freeze["stage2_fixed_observation_gate"]
    if bool(scope["closed_loop"]) or bool(scope["environment_interaction"]):
        raise ValueError("score-distill must remain offline and closed-loop free")
    if int(source["expected_rows"]) != 4800 or int(source["contexts"]) != 8:
        raise ValueError("score-distill source dataset dimensions drifted")
    if [int(value) for value in source["label_checkpoints"]] != list(LABEL_CHECKPOINTS):
        raise ValueError("score-distill label checkpoints drifted")
    if int(source["candidates_per_group"]) != 120:
        raise ValueError("score-distill group cardinality drifted")
    if [str(value) for value in training["arms"]] != list(ARMS):
        raise ValueError("score-distill arm order drifted")
    if int(training["updates"]) != 500 or int(training["formal_snapshot"]) != 500:
        raise ValueError("score-distill training endpoint drifted")
    if int(training["contexts_per_update"]) != 8 or int(training["sampled_candidates_per_context"]) != 32:
        raise ValueError("score-distill grouped batch dimensions drifted")
    if int(training["replay_queries_per_context"]) != 4:
        raise ValueError("replay slate cardinality drifted")
    if [float(value) for value in training["horizon_weights"]] != [1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3]:
        raise ValueError("horizon weights drifted")
    if float(training["latent_dagger_weight"]) != 0.5 or float(training["latent_replay_weight"]) != 0.5:
        raise ValueError("latent source weights drifted")
    if float(training["score_loss_weight"]) != 0.1 or int(training["score_top_count"]) != 8:
        raise ValueError("score loss settings drifted")
    if float(training["score_top_weight"]) != 2.0 or float(training["score_other_weight"]) != 1.0:
        raise ValueError("score rank weights drifted")
    if float(training["score_std_floor"]) != 1e-6 or float(training["score_smooth_l1_beta"]) != 1.0:
        raise ValueError("score normalization settings drifted")
    if int(training["replay_schedule_seed"]) != 20261201:
        raise ValueError("replay schedule seed drifted")
    if [int(value) for value in stage1["eval_seeds"]] != [20264925, 20264926]:
        raise ValueError("held-out evaluation seeds drifted")
    if [int(value) for value in stage2["cases"]] != list(STAGE2_CASES) or [int(value) for value in stage2["eval_seeds"]] != list(STAGE2_SEEDS):
        raise ValueError("fixed-observation cases or seeds drifted")
    if [int(value) for value in stage2["checkpoints"]] != list(STAGE2_CHECKPOINTS):
        raise ValueError("fixed-observation checkpoints drifted")
    if bool(stage2["closed_loop"]):
        raise ValueError("Stage 2 must remain fixed-observation only")
    return {"source": source, "training": training, "stage1": stage1, "stage2": stage2, "warm_start": freeze["warm_start"], "claim_boundary": freeze["claim_boundary"]}


def _validate_protocol(text: str) -> None:
    required = ("CEM proposal", "cem_dagger_dataset.pt", "[8,32]", "shuffled_score", "closed-loop")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol is missing frozen boundary clauses: {missing}")


def _load_label_dataset(torch: Any, path: Path) -> tuple[dict[str, Any], dict[tuple[int, int], Any]]:
    try:
        dataset = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(path.resolve(), map_location="cpu")
    if not isinstance(dataset, dict) or dataset.get("schema") != "jepa-action-prefix-compiler.dino-pusht-cem-dagger-labels":
        raise ValueError("unexpected CEM-DAgger label dataset schema")
    required = ("context_visual", "context_proprio", "goal_visual", "goal_proprio", "context_index", "label_iteration", "actions", "target_visual", "target_proprio", "teacher_cost")
    if any(key not in dataset for key in required):
        raise ValueError("CEM-DAgger label dataset is missing a required field")
    if int(dataset["actions"].shape[0]) != 4800:
        raise ValueError("CEM-DAgger label dataset does not contain 4800 rows")
    if tuple(dataset["actions"].shape[1:]) != (HORIZON, 10):
        raise ValueError("CEM-DAgger action shape drifted")
    if int(dataset["context_visual"].shape[0]) != 8 or int(dataset["context_proprio"].shape[0]) != 8:
        raise ValueError("CEM-DAgger context count drifted")
    for key in ("context_index", "label_iteration", "actions", "target_visual", "target_proprio", "teacher_cost"):
        if not bool(torch.isfinite(dataset[key].float()).all().item()):
            raise ValueError(f"non-finite label dataset field: {key}")
    groups: dict[tuple[int, int], list[int]] = {}
    for row, (context, iteration) in enumerate(zip(dataset["context_index"].tolist(), dataset["label_iteration"].tolist(), strict=True)):
        groups.setdefault((int(context), int(iteration)), []).append(row)
    expected = {(context, iteration) for context in range(8) for iteration in LABEL_CHECKPOINTS}
    if set(groups) != expected or any(len(rows) != 120 for rows in groups.values()):
        raise ValueError("CEM-DAgger label groups are not 8x5x120")
    group_tensors = {key: torch.as_tensor(rows, dtype=torch.long) for key, rows in groups.items()}
    return dataset, group_tensors


def _build_candidate_schedule(torch: Any, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    updates = int(settings["updates"])
    sampled = int(settings["sampled_candidates_per_context"])
    total = 120
    candidate_generator = torch.Generator(device="cpu").manual_seed(int(settings["candidate_schedule_seed"]))
    shuffle_generator = torch.Generator(device="cpu").manual_seed(int(settings["score_shuffle_seed"]))
    schedule: list[dict[str, Any]] = []
    for step in range(updates):
        iteration = LABEL_CHECKPOINTS[step % len(LABEL_CHECKPOINTS)]
        candidates = [torch.randperm(total, generator=candidate_generator)[:sampled] for _ in range(8)]
        shuffles = [torch.randperm(sampled, generator=shuffle_generator) for _ in range(8)]
        schedule.append({"iteration": int(iteration), "candidate_indices": candidates, "shuffle_indices": shuffles})
    return schedule


def _materialize_dagger_batch(torch: Any, dataset: Mapping[str, Any], groups: Mapping[tuple[int, int], Any], schedule_row: Mapping[str, Any], device: Any) -> dict[str, Any]:
    context_ids = list(range(8))
    row_ids = []
    for context_index, candidate_indices in zip(context_ids, schedule_row["candidate_indices"], strict=True):
        group_rows = groups[(context_index, int(schedule_row["iteration"]))]
        row_ids.append(group_rows.index_select(0, candidate_indices))
    rows = torch.cat(row_ids)
    context = {
        "visual": dataset["context_visual"].index_select(0, torch.as_tensor(context_ids)).repeat_interleave(32, dim=0).to(device, non_blocking=True),
        "proprio": dataset["context_proprio"].index_select(0, torch.as_tensor(context_ids)).repeat_interleave(32, dim=0).to(device, non_blocking=True),
    }
    goal = {
        "visual": dataset["goal_visual"].index_select(0, torch.as_tensor(context_ids)).repeat_interleave(32, dim=0).to(device, non_blocking=True),
        "proprio": dataset["goal_proprio"].index_select(0, torch.as_tensor(context_ids)).repeat_interleave(32, dim=0).to(device, non_blocking=True),
    }
    return {
        "context": context,
        "actions": dataset["actions"].index_select(0, rows).to(device, non_blocking=True),
        "target": {
            "visual": dataset["target_visual"].index_select(0, rows).to(device, non_blocking=True),
            "proprio": dataset["target_proprio"].index_select(0, rows).to(device, non_blocking=True),
        },
        "goal": goal,
        "teacher_cost": dataset["teacher_cost"].index_select(0, rows).reshape(8, 32).to(device, non_blocking=True),
    }


def _score_loss(torch: Any, student_cost: Any, teacher_cost: Any, shuffle_indices: Sequence[Any], settings: Mapping[str, Any], shuffled: bool) -> Any:
    import torch.nn.functional as F

    if tuple(student_cost.shape) != (8, 32) or tuple(teacher_cost.shape) != (8, 32):
        raise ValueError("grouped score loss expects [8,32] costs")
    if shuffled:
        permutation = torch.stack([item.to(teacher_cost.device) for item in shuffle_indices])
        teacher_for_loss = teacher_cost.gather(1, permutation)
    else:
        teacher_for_loss = teacher_cost
    teacher_mean = teacher_for_loss.mean(dim=1, keepdim=True)
    teacher_std = teacher_for_loss.std(dim=1, unbiased=False, keepdim=True).clamp_min(float(settings["score_std_floor"]))
    student_norm = (student_cost - student_cost.mean(dim=1, keepdim=True)) / teacher_std
    teacher_norm = (teacher_for_loss - teacher_mean) / teacher_std
    element = F.smooth_l1_loss(student_norm, teacher_norm, beta=float(settings["score_smooth_l1_beta"]), reduction="none")
    order = torch.argsort(teacher_for_loss, dim=1, stable=True)
    weights = torch.full_like(element, float(settings["score_other_weight"]))
    weights.scatter_(1, order[:, : int(settings["score_top_count"])], float(settings["score_top_weight"]))
    per_context = (element * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(float(settings["score_std_floor"]))
    return per_context.mean()


def _train(torch: Any, models: Mapping[str, Any], optimizers: Mapping[str, Any], dataset: Mapping[str, Any], groups: Mapping[tuple[int, int], Any], candidate_schedule: Sequence[Mapping[str, Any]], replay_schedule: Sequence[Sequence[int]], replay_bank: Mapping[str, Any], train_encoded: Mapping[str, Any], settings: Mapping[str, Any], objective: Any, device: Any, slate: Any) -> dict[str, list[dict[str, float]]]:
    import run_dino_pusht_cem_dagger as dagger

    losses: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    horizon_weights = settings["horizon_weights"]
    for step, schedule_row in enumerate(candidate_schedule):
        dagger_batch = _materialize_dagger_batch(torch, dataset, groups, schedule_row, device)
        replay_batch = slate._materialize_slate_batch(train_encoded, replay_bank, replay_schedule[step], device)
        for arm in ARMS:
            model = models[arm]
            model.train()
            dagger_prediction = model(dagger_batch["context"], dagger_batch["actions"])
            replay_prediction = model(replay_batch[0], replay_batch[1])
            dagger_latent = dagger._weighted_horizon_mse(dagger_prediction, dagger_batch["target"], horizon_weights)
            replay_latent = dagger._weighted_horizon_mse(replay_prediction, replay_batch[2], horizon_weights)
            latent_loss = float(settings["latent_dagger_weight"]) * dagger_latent + float(settings["latent_replay_weight"]) * replay_latent
            if arm == "latent_only":
                score_loss = latent_loss.detach().new_zeros(())
            else:
                student_cost = objective(dagger_prediction, dagger_batch["goal"]).reshape(8, 32)
                score_loss = _score_loss(torch, student_cost, dagger_batch["teacher_cost"], schedule_row["shuffle_indices"], settings, arm == "shuffled_score")
            total_loss = latent_loss + float(settings["score_loss_weight"]) * score_loss
            _assert_finite(torch, total_loss.detach(), f"{arm} total loss")
            optimizers[arm].zero_grad(set_to_none=True)
            total_loss.backward()
            optimizers[arm].step()
            losses[arm].append({"step": float(step + 1), "latent_dagger": float(dagger_latent.detach().cpu()), "latent_replay": float(replay_latent.detach().cpu()), "latent_loss": float(latent_loss.detach().cpu()), "score_loss": float(score_loss.detach().cpu()), "total_loss": float(total_loss.detach().cpu())})
    return losses


def _paired(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]], metric: str, higher_is_better: bool = True) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[int, int]:
        return int(row["seed"]), int(row["context_index"])
    first = {key(row): row for row in rows_a}
    second = {key(row): row for row in rows_b}
    deltas: list[float] = []
    for pairing in first:
        if metric == "spearman":
            left = float(first[pairing]["ranking"][metric])
            right = float(second[pairing]["ranking"][metric])
        elif metric == "top30_overlap":
            left = float(first[pairing]["ranking"][metric])
            right = float(second[pairing]["ranking"][metric])
        elif metric == "terminal_relative_mse":
            left = float(first[pairing]["relative_mse"][-1])
            right = float(second[pairing]["relative_mse"][-1])
        else:
            raise ValueError(f"unknown paired metric: {metric}")
        deltas.append(left - right)
    improve = sum(delta > 0 if higher_is_better else delta < 0 for delta in deltas)
    worse = sum(delta < 0 if higher_is_better else delta > 0 for delta in deltas)
    return {"improve": int(improve), "worse": int(worse), "tie": len(deltas) - improve - worse, "median_delta": _median(deltas), "mean_delta": _mean(deltas), "deltas": deltas}


def _absolute_gate(arm_result: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    absolute = settings["absolute_gate"]
    causality_values = [float(item["max_abs_delta"]) for item in arm_result["causality"].get("cases", [])]
    causality_max = max(causality_values) if causality_values else float("inf")
    finite_outputs = _finite(arm_result)
    passed = bool(
        arm_result["spearman"]["median"] >= float(absolute["spearman_median_min"])
        and arm_result["spearman"]["minimum"] >= float(absolute["spearman_minimum_min"])
        and arm_result["top30_overlap"]["median"] >= float(absolute["top30_median_min"])
        and arm_result["top30_overlap"]["minimum"] >= float(absolute["top30_minimum_min"])
        and causality_max <= float(absolute["causality_max_abs"])
        and finite_outputs
    )
    return {"spearman_median": float(arm_result["spearman"]["median"]), "spearman_minimum": float(arm_result["spearman"]["minimum"]), "top30_median": float(arm_result["top30_overlap"]["median"]), "top30_minimum": float(arm_result["top30_overlap"]["minimum"]), "causality_max_abs": causality_max, "finite_outputs": bool(finite_outputs), "silent_fallback": False, "status": "PASS" if passed else "FAIL", "thresholds": dict(absolute)}


def _run_stage2(torch: Any, args: argparse.Namespace, settings: Mapping[str, Any], models: Mapping[str, Any], teacher: Any, official_dataset: Any, device: Any, objective: Any, dagger: Any) -> dict[str, Any]:
    parent = _load_json(args.parent_summary.resolve())
    historical = dagger._historical_reference(args.historical_cem_summary.resolve(), parent)
    with args.plan_targets.resolve().open("rb") as handle:
        targets = pickle.load(handle)
    if not isinstance(targets, dict) or int(targets.get("goal_H", -1)) != HORIZON:
        raise ValueError("unexpected fixed-observation plan_targets payload")
    from preprocessor import Preprocessor
    from utils import move_to_device

    preprocessor = Preprocessor(action_mean=official_dataset.action_mean, action_std=official_dataset.action_std, state_mean=official_dataset.state_mean, state_std=official_dataset.state_std, proprio_mean=official_dataset.proprio_mean, proprio_std=official_dataset.proprio_std, transform=official_dataset.transform)
    obs = move_to_device(preprocessor.transform_obs(targets["obs_0"]), device)
    goal_obs = move_to_device(preprocessor.transform_obs(targets["obs_g"]), device)
    with torch.no_grad():
        encoded_obs = teacher.encode_obs(obs)
        encoded_goal = teacher.encode_obs(goal_obs)
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    case_payloads = []
    for case_index, eval_seed in zip(STAGE2_CASES, STAGE2_SEEDS, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(20261100 + case_index)
        innovations = torch.randn((30, 300, HORIZON, 10), generator=generator)
        context = {key: value[case_index : case_index + 1].expand((300,) + tuple(value.shape[1:])) for key, value in encoded_obs.items()}
        goal = {key: value[case_index : case_index + 1].expand((300,) + tuple(value.shape[1:])) for key, value in encoded_goal.items()}
        payload = {"case_index": int(case_index), "eval_seed": int(eval_seed), "arms": {}}
        for arm in ARMS:
            record = dagger._run_fixed_arm(torch, models[arm], teacher, context, goal, innovations, objective, STAGE2_CHECKPOINTS, historical, case_index, device)
            records[arm].append(record)
            payload["arms"][arm] = record
        case_payloads.append(payload)
        (args.output.resolve() / f"case_{case_index:02d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    aggregates = {arm: dagger._aggregate_fixed(records[arm], STAGE2_CHECKPOINTS) for arm in ARMS}
    primary = aggregates["score_distill"]["30"]
    old_regrets = [float(historical["student_only_regret_by_case_iter30"][str(case)]) for case in STAGE2_CASES]
    score_regrets = [float(row["snapshots"]["30"]["teacher_cost_regret"]) for row in records["score_distill"]]
    score_drifts = [float(row["snapshots"]["30"]["first_action_rms_drift"]) for row in records["score_distill"]]
    latent_regrets = [float(row["snapshots"]["30"]["teacher_cost_regret"]) for row in records["latent_only"]]
    latent_drifts = [float(row["snapshots"]["30"]["first_action_rms_drift"]) for row in records["latent_only"]]
    shuffled_regrets = [float(row["snapshots"]["30"]["teacher_cost_regret"]) for row in records["shuffled_score"]]
    shuffled_drifts = [float(row["snapshots"]["30"]["first_action_rms_drift"]) for row in records["shuffled_score"]]
    score_support_pass = bool(primary["first_action_rms_drift"]["median"] <= float(settings["stage2"]["first_action_rms_median_max"]) and primary["first_action_coordinate_abs_max"]["maximum"] <= float(settings["stage2"]["first_action_coordinate_abs_max_max"]) and primary["teacher_cost_regret"]["median"] <= float(historical["student_only_regret_median_iter30"]) and sum(a <= b for a, b in zip(score_regrets, old_regrets, strict=True)) >= int(settings["stage2"]["support_cases_not_worse_min"]))
    score_vs_latent = {"drift_median": {"score_distill": _median(score_drifts), "latent_only": _median(latent_drifts)}, "drift_cases_not_worse": int(sum(a <= b for a, b in zip(score_drifts, latent_drifts, strict=True))), "drift_median_strictly_lower": _median(score_drifts) < _median(latent_drifts), "teacher_regret_median": {"score_distill": _median(score_regrets), "latent_only": _median(latent_regrets)}, "teacher_regret_cases_not_worse": int(sum(a <= b for a, b in zip(score_regrets, latent_regrets, strict=True))), "teacher_regret_median_strictly_lower": _median(score_regrets) < _median(latent_regrets)}
    score_mechanism_pass = bool(score_vs_latent["drift_median_strictly_lower"] and score_vs_latent["drift_cases_not_worse"] >= int(settings["stage2"]["score_vs_latent_drift_cases_not_worse_min"]) and score_vs_latent["teacher_regret_median_strictly_lower"] and score_vs_latent["teacher_regret_cases_not_worse"] >= int(settings["stage2"]["score_vs_latent_teacher_regret_cases_not_worse_min"]))
    shuffled_reproduces = bool(_median(shuffled_drifts) < _median(latent_drifts) and sum(a <= b for a, b in zip(shuffled_drifts, latent_drifts, strict=True)) >= 4 and _median(shuffled_regrets) < _median(latent_regrets) and sum(a <= b for a, b in zip(shuffled_regrets, latent_regrets, strict=True)) >= 4)
    return {"status": "PASS" if score_support_pass and score_mechanism_pass and not shuffled_reproduces else "FAIL", "historical_reference": historical, "aggregates": aggregates, "arms": records, "score_support_gate": {"status": "PASS" if score_support_pass else "FAIL", "cases_not_worse_than_historical": int(sum(a <= b for a, b in zip(score_regrets, old_regrets, strict=True)))}, "score_vs_latent": score_vs_latent, "shuffled_reproduces_mechanism_gain": shuffled_reproduces, "case_payloads": case_payloads, "claim_boundary": "fixed-observation mechanism diagnosis only"}


def _runtime_required(args: argparse.Namespace) -> None:
    names = ("root", "output", "manifest", "dataset", "student_checkpoint", "teacher_checkpoint", "checkpoint_config", "data_root", "parent_summary", "plan_targets", "historical_cem_summary")
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"run mode is missing arguments: {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    freeze = _load_json(args.freeze.resolve())
    settings = _validate_freeze(freeze)
    _validate_protocol(args.protocol.read_text(encoding="utf-8"))
    if args.mode == "preflight":
        print(json.dumps({"status": "READY", "schema": freeze["schema"], "arms": list(ARMS), "source_rows": 4800, "loads_model": False, "loads_dataset": False}, indent=2))
        return 0
    _runtime_required(args)
    _require_compute_node()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_dino_pusht_cem_dagger as dagger
    import run_dino_pusht_closed_loop as closed_loop
    import run_dino_pusht_grounded_prefix as grounded
    import run_dino_pusht_query_slate_rank as slate
    import run_dino_pusht_recurrent_student as recurrent
    from planning.objectives import create_objective_fn

    label_dataset, groups = _load_label_dataset(torch, args.dataset)
    manifest = dagger._validate_manifest(_load_json(args.manifest.resolve()))
    loader_args = argparse.Namespace(root=args.root, student_checkpoint=args.student_checkpoint, teacher_checkpoint=args.teacher_checkpoint, checkpoint_config=args.checkpoint_config, data_root=args.data_root)
    runtime_freeze = {"student": {"checkpoint_schema": settings["warm_start"]["checkpoint_schema"], "step": int(settings["warm_start"]["step"]), "hidden_dim": WIDTH, "visual_dim": 384, "proprio_dim": 10, "packed_action_dim": 10}}
    teacher, warm_world_model, _model_cfg, official_dataset = closed_loop._load_models(loader_args, runtime_freeze)
    train_dset, heldout_dset = grounded._load_trajectory_datasets(args.root.resolve(), (args.asset_root or args.root).resolve(), runtime_freeze, args.student_checkpoint.resolve(), args.checkpoint_config.resolve())
    device = torch.device("cuda:0")
    cache = grounded._RawEpisodeCache(8)
    train_encoded = grounded._preencode_manifest(train_dset, "train", manifest["train"], 5, 8, cache, teacher, device)
    heldout_encoded = grounded._preencode_manifest(heldout_dset, "heldout", manifest["heldout"], 5, 8, cache, teacher, device)
    selected_by_episode: dict[int, Mapping[str, Any]] = {}
    for example in manifest["train"]:
        selected_by_episode.setdefault(int(example["episode_id"]), example)
    episode_ids = sorted(selected_by_episode)[:8]
    if list(label_dataset.get("episode_ids", episode_ids)) != episode_ids:
        raise ValueError("label dataset collector episode order does not match frozen manifest")
    selected_examples = [selected_by_episode[episode_id] for episode_id in episode_ids]
    selected_context, _selected_actions, selected_grounded = grounded._segment_batch(train_dset, "train", selected_examples, list(range(8)), 5, cache, teacher, device)
    if int(selected_context["visual"].shape[-1]) != 384 or int(selected_context["proprio"].shape[-1]) != 10:
        raise ValueError("native latent dimensions drifted")
    objective = create_objective_fn(alpha=1, base=2, mode="last")
    bank = slate._precompute_slate_bank(teacher, train_encoded, objective, 10, int(settings["training"]["action_slate_seed"]), int(settings["training"]["cem_seed"]), device)
    replay_schedule, replay_schedule_meta = slate._context_schedule(500, 8, int(settings["training"]["replay_schedule_seed"]))
    candidate_schedule = _build_candidate_schedule(torch, settings["training"])
    warm_student = warm_world_model.student
    warm_state = {key: value.detach().cpu().clone() for key, value in warm_student.state_dict().items()}
    torch.manual_seed(int(settings["training"]["initialization_seed"]))
    torch.cuda.manual_seed_all(int(settings["training"]["initialization_seed"]))
    models: dict[str, Any] = {}
    optimizers: dict[str, Any] = {}
    for arm in ARMS:
        model = recurrent._student(10, 384, 10, WIDTH).to(device)
        model.load_state_dict(warm_state, strict=True)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=float(settings["training"]["learning_rate"]), weight_decay=float(settings["training"]["weight_decay"]), betas=tuple(settings["training"]["betas"]), eps=float(settings["training"]["eps"]))
    losses = _train(torch, models, optimizers, label_dataset, groups, candidate_schedule, replay_schedule, bank, train_encoded, settings["training"], objective, device, slate)
    checkpoints = {}
    for arm in ARMS:
        path = output / f"cem_score_distill_{arm}_step0500.pt"
        torch.save({"schema": "horizon-weighted-recurrent-student.cem-score-distill-checkpoint", "schema_version": 1, "step": 500, "arm": arm, "architecture": "RecurrentNativeLatentTransitionStudent", "state_dict": {key: value.detach().cpu().clone() for key, value in models[arm].state_dict().items()}}, path)
        checkpoints[arm] = str(path)
    eval_seeds = [int(value) for value in settings["stage1"]["eval_seeds"]]
    eval_results = dagger._evaluate_predictor(torch, models, teacher, heldout_encoded, objective, eval_seeds, 300, device)
    for arm in ARMS:
        eval_results[arm]["finite_outputs"] = bool(_finite(eval_results[arm]))
        eval_results[arm]["silent_fallback"] = False
        eval_results[arm]["absolute_gate"] = _absolute_gate(eval_results[arm], settings["stage1"])
    paired = {}
    for arm in ("score_distill", "shuffled_score"):
        paired[arm] = {"vs_latent_only": {"spearman": _paired(eval_results[arm]["per_block"], eval_results["latent_only"]["per_block"], "spearman"), "top30_overlap": _paired(eval_results[arm]["per_block"], eval_results["latent_only"]["per_block"], "top30_overlap"), "terminal_relative_mse": _paired(eval_results[arm]["per_block"], eval_results["latent_only"]["per_block"], "terminal_relative_mse", False)}}
    pair_thresholds = settings["stage1"]["paired_score_vs_latent"]
    score_pair = paired["score_distill"]["vs_latent_only"]
    shuffled_pair = paired["shuffled_score"]["vs_latent_only"]
    def pair_pass(values: Mapping[str, Any]) -> bool:
        return bool(values["spearman"]["median_delta"] >= float(pair_thresholds["spearman_median_delta_min"]) and values["spearman"]["improve"] >= int(pair_thresholds["spearman_improve_blocks_min"]) and values["top30_overlap"]["median_delta"] >= float(pair_thresholds["top30_median_delta_min"]) and values["top30_overlap"]["improve"] >= int(pair_thresholds["top30_improve_blocks_min"]))
    score_paired_pass = pair_pass(score_pair)
    shuffled_reproduces = pair_pass(shuffled_pair)
    stage1_pass = bool(eval_results["score_distill"]["absolute_gate"]["status"] == "PASS" and score_paired_pass and not shuffled_reproduces)
    stage1 = {"status": "PASS" if stage1_pass else "FAIL", "arms": eval_results, "paired_deltas": paired, "score_distill_paired_gate": {"status": "PASS" if score_paired_pass else "FAIL", "thresholds": dict(pair_thresholds)}, "shuffled_reproduces_paired_gain": shuffled_reproduces, "claim_boundary": settings["claim_boundary"]}
    stage2 = {"status": "SKIPPED", "reason": "Stage 1 score absolute/paired/negative-control gate failed"}
    if stage1_pass:
        stage2 = _run_stage2(torch, args, settings, models, teacher, official_dataset, device, objective, dagger)
    summary = {"schema": "horizon-weighted-recurrent-student.cem-score-distill-summary", "schema_version": 1, "freeze": str(args.freeze.resolve()), "protocol": str(args.protocol.resolve()), "source_dataset": str(args.dataset.resolve()), "source_job": "24928207.pbs101", "training": {"updates": 500, "losses": losses, "replay_schedule": replay_schedule_meta, "replay_contexts_are_old_256_bank": True, "candidate_schedule": {"contexts_per_update": 8, "candidates_per_context": 32, "source_group_candidates": 120, "group_rule": "step modulo five", "precomputed_before_training": True}}, "checkpoints": checkpoints, "stage1_predictor_gate": stage1, "stage2_fixed_observation_gate": stage2, "validity": {"all_outputs_finite": bool(_finite(summary_placeholder := {"losses": losses, "stage1": stage1, "stage2": stage2})), "silent_fallback": False, "environment_interaction": False, "teacher_labels_recollected": False, "source_rows_exact": int(label_dataset["actions"].shape[0]) == 4800}, "claim_boundary": settings["claim_boundary"], "gpu": torch.cuda.get_device_name(0)}
    summary_path = (args.summary or output / "cem_score_distill_summary.json").resolve()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary_path != output / "cem_score_distill_summary.json":
        (output / "cem_score_distill_summary.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output": str(output), "stage1": stage1["status"], "stage2": stage2["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
