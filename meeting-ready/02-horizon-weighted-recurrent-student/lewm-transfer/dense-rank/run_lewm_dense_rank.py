#!/usr/bin/env python3
"""LeWM PushT dense-query and elite-boundary rank-distillation experiment.

The three arms share one precomputed 64-query train bank, one held-out bank,
one initial state, and one context schedule.  Only the training objective
differs: dense latent loss, latent plus frozen teacher-rank loss, or the same
rank loss with fixed within-context label shuffling.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_lewm_recurrent_student as base


TRAIN_CANDIDATES = 64
RANK_TOP_COUNT = 6
RANK_BOUNDARY_COUNT = 6
RANK_TEMPERATURE = 0.5
RANK_LOSS_WEIGHT = 0.1
RANK_SHUFFLE_SEED = 20300910
ARMS = ("dense_latent", "dense_rank", "shuffled_rank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prepared-rows", type=Path, default=None)
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
    value = load_json(path)
    base_name = value.get("base_freeze")
    if not base_name:
        return value
    base_path = (path.parent / str(base_name)).resolve()
    result = load_json(base_path)
    result["experiment_overlay"] = {key: item for key, item in value.items() if key != "base_freeze"}
    return result


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


def validate_dense_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    settings = base.validate_freeze(freeze)
    overlay = freeze.get("experiment_overlay", {})
    if overlay.get("experiment_name") != "lewm_dense_query_rank_distillation":
        raise ValueError("unexpected dense-rank freeze")
    train = overlay.get("training", {})
    if int(train.get("train_candidates", -1)) != TRAIN_CANDIDATES:
        raise ValueError("dense train candidate count drifted")
    if int(train.get("rank_top_count", -1)) != RANK_TOP_COUNT or int(train.get("rank_boundary_count", -1)) != RANK_BOUNDARY_COUNT:
        raise ValueError("rank pair strata drifted")
    if not math.isclose(float(train.get("rank_temperature", -1)), RANK_TEMPERATURE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("rank temperature drifted")
    if not math.isclose(float(train.get("rank_loss_weight", -1)), RANK_LOSS_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("rank loss weight drifted")
    if int(train.get("rank_shuffle_seed", -1)) != RANK_SHUFFLE_SEED:
        raise ValueError("rank shuffle seed drifted")
    return settings


def prepare_dense_rows(model: Any, dataset_path: Path, manifest: Mapping[str, Any], freeze: Mapping[str, Any], args: argparse.Namespace, output: Path) -> list[dict[str, Any]]:
    """Prepare a shared 64-candidate train bank and the frozen 16-block test bank."""
    import torch

    low, high, gaussian_std = base._parse_action_bounds(args, freeze)
    schedule = freeze["shared_data_and_schedule"]
    seeds = schedule["seeds"]
    action_generator = torch.Generator(device="cpu").manual_seed(int(seeds["action_slate"]))
    rank_generator = torch.Generator(device="cpu").manual_seed(RANK_SHUFFLE_SEED)
    heldout_seeds = [int(item) for item in seeds["heldout_action_prefix"]]
    rows: list[dict[str, Any]] = []
    with base.HDF5EpisodeSliceReader(dataset_path) as reader:
        for manifest_row in list(manifest["splits"]["train"]) + list(manifest["splits"]["heldout"]):
            raw = reader.read_manifest_row(manifest_row)
            current = base._normalise_pixels(raw["pixels"][:1]).unsqueeze(1).to("cuda")
            episode = reader.episode_slice(int(manifest_row["episode_id"]))
            goal_index = int(manifest_row.get("goal_step_offset", base.HORIZON * 5))
            goal_pixels = base._normalise_pixels(episode["pixels"][goal_index : goal_index + 1]).unsqueeze(1).to("cuda")
            logged = torch.from_numpy(base.pack_raw_actions(raw["future_actions_raw"]))
            encode_action = logged[:1].unsqueeze(0).to("cuda")
            with torch.no_grad():
                latent = model.encode({"pixels": current, "action": encode_action})["emb"]
                goal_emb = model.encode({"pixels": goal_pixels})["emb"]
            if tuple(latent.shape) != (1, 1, base.LATENT_DIM):
                raise RuntimeError(f"official H=1 encoder output is not [1,1,192]: {tuple(latent.shape)}")
            if manifest_row["split"] == "train":
                # Consume the same frozen slate prefix as the original runner;
                # retain its candidate64 bank instead of using the one-step CEM row.
                base._slate_actions(logged, action_generator, low, high, gaussian_std, 1)
                base._slate_actions(logged, action_generator, low, high, gaussian_std, 1)
                actions = base._slate_actions(logged, action_generator, low, high, gaussian_std, TRAIN_CANDIDATES).to("cuda")
                # The original frozen slate generator consumes a CEM-resample
                # noise draw after candidate64, even though dense-rank does not
                # retain that one-step CEM row.  Consume the same draw so the
                # next context's candidate64 bank remains bitwise identical.
                torch.randn((base.HORIZON, base.ACTION_DIM), generator=action_generator)
                context = latent.expand(TRAIN_CANDIDATES, -1, -1)
                with torch.no_grad():
                    targets = base.official_teacher_targets(model, context, actions)
                    objective = base._official_objective(model, {"latent_history": latent, "goal_emb": goal_emb}, targets)
                permutation = torch.randperm(TRAIN_CANDIDATES, generator=rank_generator)
                rows.append({
                    "split": "train",
                    "context_id": manifest_row["context_id"],
                    "latent_history": latent.detach().cpu(),
                    "action_history": encode_action.detach().cpu(),
                    "future_actions": actions.detach().cpu(),
                    "teacher_targets": targets.detach().cpu(),
                    "teacher_objective": objective.detach().cpu(),
                    "rank_shuffle_indices": permutation,
                    "goal_emb": goal_emb.detach().cpu(),
                })
            else:
                banks, targets, objectives = [], [], []
                for heldout_seed in heldout_seeds:
                    generator = torch.Generator(device="cpu").manual_seed(heldout_seed)
                    actions = base._slate_actions(logged, generator, low, high, gaussian_std, 300).to("cuda")
                    context = latent.expand(300, -1, -1)
                    with torch.no_grad():
                        target = base.official_teacher_targets(model, context, actions)
                        objective = base._official_objective(model, {"latent_history": latent, "goal_emb": goal_emb}, target)
                    banks.append(actions.detach().cpu())
                    targets.append(target.detach().cpu())
                    objectives.append(objective.detach().cpu())
                rows.append({
                    "split": "heldout",
                    "context_id": manifest_row["context_id"],
                    "latent_history": latent.detach().cpu(),
                    "action_history": encode_action.detach().cpu(),
                    "future_actions": torch.stack(banks),
                    "teacher_targets": torch.stack(targets),
                    "teacher_objective": torch.stack(objectives),
                    "goal_emb": goal_emb.detach().cpu(),
                })
    torch.save(rows, output / "prepared_rows.pt")
    return rows


def validate_dense_rows(rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, int]:
    spec = freeze["shared_data_and_schedule"]["context_manifest"]
    train = [row for row in rows if row.get("split") == "train"]
    heldout = [row for row in rows if row.get("split") == "heldout"]
    if len(train) != int(spec["train_context_target"]) or len(heldout) != int(spec["heldout_contexts"]):
        raise ValueError("prepared dense rows do not match the frozen context counts")
    for split, group in (("train", train), ("heldout", heldout)):
        for row in group:
            context = base._tensor(row["latent_history"])
            actions = base._tensor(row["future_actions"])
            targets = base._tensor(row["teacher_targets"])
            if tuple(context.shape) not in ((1, 1, base.LATENT_DIM), (1, base.LATENT_DIM)):
                raise ValueError(f"{split} latent history shape drifted")
            expected_actions = (TRAIN_CANDIDATES, base.HORIZON, base.ACTION_DIM) if split == "train" else (2, 300, base.HORIZON, base.ACTION_DIM)
            expected_targets = (TRAIN_CANDIDATES, base.HORIZON, base.LATENT_DIM) if split == "train" else (2, 300, base.HORIZON, base.LATENT_DIM)
            if tuple(actions.shape) != expected_actions or tuple(targets.shape) != expected_targets:
                raise ValueError(f"{split} dense rows have an unexpected action/target shape")
            objective = base._tensor(row.get("teacher_objective"))
            expected_objective = (TRAIN_CANDIDATES,) if split == "train" else (2, 300)
            if tuple(objective.shape) != expected_objective:
                raise ValueError(f"{split} teacher objective shape drifted")
            if split == "train" and tuple(base._tensor(row.get("rank_shuffle_indices")).shape) != (TRAIN_CANDIDATES,):
                raise ValueError("train rank shuffle indices are missing or malformed")
            if split == "heldout" and ("goal_emb" not in row or "action_history" not in row):
                raise ValueError("heldout row lacks official external criterion inputs")
    return {"train_contexts": len(train), "heldout_contexts": len(heldout), "train_candidates_per_context": TRAIN_CANDIDATES, "heldout_blocks": len(heldout) * 2}


def _effective_teacher_costs(rows: Sequence[Mapping[str, Any]], indices: Any, arm: str, device: str = "cuda") -> Any:
    import torch

    values = []
    for index in indices:
        cost = base._tensor(rows[int(index)]["teacher_objective"], dtype=torch.float32)
        if arm == "shuffled_rank":
            permutation = base._tensor(rows[int(index)]["rank_shuffle_indices"], dtype=torch.long)
            cost = cost[permutation]
        values.append(cost)
    return torch.stack(values).to(device)


def pairwise_rank_loss(student_cost: Any, teacher_cost: Any, temperature: float = RANK_TEMPERATURE) -> Any:
    import torch
    import torch.nn.functional as F

    if student_cost.ndim != 2 or teacher_cost.shape != student_cost.shape or student_cost.shape[1] != TRAIN_CANDIDATES:
        raise ValueError("rank loss expects [batch_contexts,64] cost matrices")
    losses = []
    for context_index in range(student_cost.shape[0]):
        teacher = teacher_cost[context_index]
        order = torch.argsort(teacher, stable=True)
        positive = order[:RANK_TOP_COUNT]
        negative = order[RANK_TOP_COUNT : RANK_TOP_COUNT + RANK_BOUNDARY_COUNT]
        scale = teacher.std(unbiased=False).clamp_min(1e-6)
        normalized_student = student_cost[context_index] / scale
        logits = (normalized_student[positive, None] - normalized_student[None, negative]) / temperature
        losses.append(F.softplus(logits).mean())
    return torch.stack(losses).mean()


def _student_costs_with_gradient(official_model: Any, rows: Sequence[Mapping[str, Any]], indices: Any, prediction: Any) -> Any:
    import torch

    costs = []
    offset = 0
    for index in indices:
        # Keep the official criterion's original goal shape and let its
        # established broadcasting contract handle the candidate batch.  The
        # call is per context so goals never get mixed across rows.
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
    arm_dir = output / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, int(settings["steps"]) + 1):
        indices = torch.randperm(len(train_rows), generator=schedule_generator)[: settings["batch_contexts"]]
        contexts = torch.cat([base._tensor(train_rows[int(i)]["latent_history"]).expand(TRAIN_CANDIDATES, -1, -1) for i in indices], dim=0).to("cuda")
        actions = torch.cat([base._tensor(train_rows[int(i)]["future_actions"]) for i in indices], dim=0).to("cuda")
        targets = torch.cat([base._tensor(train_rows[int(i)]["teacher_targets"]) for i in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        latent_loss, per_horizon = base.recurrent_loss(prediction, targets)
        rank_loss = torch.zeros((), dtype=latent_loss.dtype, device=latent_loss.device)
        if arm != "dense_latent":
            student_cost = _student_costs_with_gradient(official_model, train_rows, indices, prediction)
            teacher_cost = _effective_teacher_costs(train_rows, indices, arm)
            rank_loss = pairwise_rank_loss(student_cost, teacher_cost)
        total_loss = latent_loss + RANK_LOSS_WEIGHT * rank_loss
        if not torch.isfinite(total_loss) or not bool(torch.isfinite(per_horizon).all()) or not torch.isfinite(rank_loss):
            raise FloatingPointError(f"non-finite training loss at step {step} arm={arm}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        histories.append({
            "step": step,
            "weighted_latent_mse": float(latent_loss.detach().cpu()),
            "rank_loss": float(rank_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "per_horizon_mse": [float(item) for item in per_horizon.detach().cpu()],
        })
        if step in base.SNAPSHOT_STEPS:
            state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
            snapshots[f"step_{step}"] = state
            torch.save(state, arm_dir / f"lewm_dense_rank_{arm}_step{step:04d}.pt")
    first = histories[0]["weighted_latent_mse"]
    last10 = statistics.median(item["weighted_latent_mse"] for item in histories[-10:])
    return {"student": student, "snapshots": snapshots, "parameter_count": int(sum(parameter.numel() for parameter in student.parameters())), "per_step": histories, "last10_to_first_ratio": float(last10 / max(first, 1e-12))}


def _metric_medians(evaluation: Mapping[str, Any]) -> dict[str, float]:
    return dict(evaluation["step_1500"]["terminal_ranking"])


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


def screening_gate(arm_results: Mapping[str, Any]) -> dict[str, Any]:
    a = arm_results["dense_latent"]["evaluation"]["step_1500"]
    b = arm_results["dense_rank"]["evaluation"]["step_1500"]
    c = arm_results["shuffled_rank"]["evaluation"]["step_1500"]
    b_delta = _paired_delta(arm_results["dense_latent"]["evaluation"], arm_results["dense_rank"]["evaluation"], "top30_overlap")
    c_delta = _paired_delta(arm_results["dense_latent"]["evaluation"], arm_results["shuffled_rank"]["evaluation"], "top30_overlap")
    b_spearman = _paired_delta(arm_results["dense_latent"]["evaluation"], arm_results["dense_rank"]["evaluation"], "spearman")
    a_rank, b_rank, c_rank = (item["terminal_ranking"] for item in (a, b, c))
    a_positive = sum(float(item["top30_overlap"]) > 0 for item in a["per_block"])
    b_positive = sum(float(item["top30_overlap"]) > 0 for item in b["per_block"])
    c_positive = sum(float(item["top30_overlap"]) > 0 for item in c["per_block"])
    b_conditions = {
        "top30_median_delta_min": b_delta["median_delta"] >= 0.10,
        "top30_improved_blocks_min": b_delta["improve"] >= 10,
        "minimum_top30_non_decrease": b_rank["top30_overlap_minimum"] >= a_rank["top30_overlap_minimum"],
        "positive_top30_non_decrease": b_positive >= a_positive,
        "spearman_median_non_decrease": b_rank["spearman_median"] >= a_rank["spearman_median"],
    }
    c_same_gain = bool(c_delta["median_delta"] >= 0.10 and c_delta["improve"] >= 10 and c_rank["top30_overlap_minimum"] >= a_rank["top30_overlap_minimum"] and c_positive >= a_positive and c_rank["spearman_median"] >= a_rank["spearman_median"])
    b_conditions["shuffled_control_not_same_gain"] = not c_same_gain
    return {"status": "GO" if all(b_conditions.values()) else "NO-GO", "conditions": b_conditions, "dense_rank_vs_dense_latent": {"top30": b_delta, "spearman": b_spearman}, "shuffled_rank_vs_dense_latent": {"top30": c_delta}, "positive_top30_blocks": {"dense_latent": a_positive, "dense_rank": b_positive, "shuffled_rank": c_positive}}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], settings: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    base.require_compute_node()
    import torch

    dataset_path = (args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve()
    manifest_path = (args.manifest or (args.output / "context_manifest.json")).resolve()
    if manifest_path.is_file():
        manifest = base.validate_manifest(manifest_path, freeze)
    else:
        manifest = base.build_context_manifest(dataset_path, manifest_path, freeze)
        manifest = base.validate_manifest(manifest_path, freeze)
    official_model = base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    rows_path = (args.prepared_rows or (args.output / "prepared_rows.pt")).resolve()
    if rows_path.is_file():
        rows = torch.load(rows_path, map_location="cpu", weights_only=False)
    else:
        rows = prepare_dense_rows(official_model, dataset_path, manifest, freeze, args, args.output.resolve())
    row_meta = validate_dense_rows(rows, freeze)
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
        arm_results[arm] = {"parameter_count": result["parameter_count"], "training": {"last10_to_first_ratio": result["last10_to_first_ratio"], "per_step": result["per_step"]}, "evaluation": evaluation, "causality": causality, "predictor_latency": latency, "absolute_predictor_gates": gates}
    screening = screening_gate(arm_results)
    summary = {
        "schema": "lewm-recurrent-student.dense-rank",
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_probe": str(args.interface_probe.resolve()),
        "interface_contract": jsonable(contract.__dict__),
        "source": {"lewm_commit": freeze["evidence_boundary"]["source_commit"], "stable_worldmodel_cem_commit": freeze["evidence_boundary"]["stable_worldmodel_cem_commit"], "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()), "dataset": str(dataset_path)},
        "manifest": str(manifest_path),
        "prepared_row_counts": row_meta,
        "shared_training_contract": {"architecture": "LeWMCompactRecurrentTransitionStudent", "hidden_dim": base.HIDDEN_DIM, "attention": False, "conditioner": False, "goal_input": False, "teacher_forcing": False, "initialization_state_shared": True, "context_schedule_shared": True, "train_candidate_bank_shared": True, "rank_top_count": RANK_TOP_COUNT, "rank_boundary_count": RANK_BOUNDARY_COUNT, "rank_temperature": RANK_TEMPERATURE, "rank_loss_weight": RANK_LOSS_WEIGHT},
        "pairing": {"initialization_seed": settings["seeds"]["initialization"], "training_seed": settings["seeds"]["training"], "context_schedule_seed": settings["seeds"]["context_schedule"], "rank_shuffle_seed": RANK_SHUFFLE_SEED, "same_initial_state": True, "same_context_schedule": True, "same_train_candidate_bank": True, "same_heldout_candidate_bank": True},
        "arms": arm_results,
        "screening_gate": screening,
        "stage_b": {"status": "NOT_RUN_BY_SCOPE", "full_cem_viability": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "This is a predictor-level three-arm training comparison only. It does not claim official CEM viability, planner viability, closed-loop PushT success, encode_obs speedup, or Fast-LeWM comparison.",
    }
    write_json(args.output.resolve() / "lewm_dense_rank_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = load_freeze(args.freeze.resolve())
    settings = validate_dense_freeze(freeze)
    if args.mode == "status":
        value = {"schema": "lewm-recurrent-student.dense-rank", "status": "READY", "train_candidates": TRAIN_CANDIDATES, "arms": list(ARMS), "interface_probe": str(args.interface_probe.resolve())}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value))
        return 0
    if not args.interface_probe.is_file():
        raise FileNotFoundError(args.interface_probe)
    contract = base.load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, settings, contract)
    print(json.dumps({"status": summary["status"], "screening_gate": summary["screening_gate"]["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
