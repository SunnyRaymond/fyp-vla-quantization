#!/usr/bin/env python3
"""LeWM temporal-balanced training confirmatory predictor-only experiment."""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
EMA_DIR = TRANSFER / "ema-temporal-confirm"
sys.path.insert(0, str(EMA_DIR))
sys.path.insert(0, str(TRANSFER / "state-coverage"))
sys.path.insert(0, str(TRANSFER / "score-distill"))
sys.path.insert(0, str(TRANSFER / "dense-rank"))

import run_lewm_ema_temporal_confirm as ema  # noqa: E402

base = ema.base
score = ema.score
dense = ema.dense

TRAIN_CONTEXTS = 512
TRAIN_CANDIDATES = 64
BATCH_CONTEXTS = 8
STEPS = 3000
SNAPSHOT_STEPS = (500, 1000, 1500, 3000)
FRESH_EPISODE_COUNT = 8
FRESH_ANCHORS = ("early", "middle", "late")
FRESH_SEEDS = (20300927, 20300928)
HORIZON_PRIMITIVES = base.HORIZON * (base.ACTION_DIM // 2)

# Reuse the tested fresh-row/evaluation implementation with this experiment's
# frozen seed and snapshot contracts.
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


def load_freeze(path: Path, phase2_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = load_json(path.resolve())
    if freeze.get("schema") != "lewm-recurrent-student.temporal-balanced-train-freeze" or freeze.get("status") != "frozen":
        raise ValueError("temporal-balanced freeze is not frozen or has an unexpected schema")
    phase2 = load_json(phase2_path.resolve())
    if phase2.get("schema") != "lewm-recurrent-student.state-context-coverage-freeze" or phase2.get("status") != "frozen":
        raise ValueError("unexpected Phase 2 freeze")
    design = freeze["design"]
    if design.get("training_arms") != ["step0_train", "balanced_temporal_train"] or int(design["training_rows"]) != 512 or int(design["updates"]) != 3000:
        raise ValueError("training arm/count/update contract drifted")
    training = freeze["training"]
    for key, expected in {"batch_contexts": 8, "train_candidates": 64, "initialization_seed": 20300901, "training_seed": 20300902, "context_schedule_seed": 20300904, "candidate_slate_seed": 20300905}.items():
        if int(training[key]) != expected:
            raise ValueError(f"training {key} drifted")
    if [float(x) for x in training["horizon_weights"]] != list(base.HORIZON_WEIGHTS):
        raise ValueError("horizon weights drifted")
    if freeze["fresh_evaluation"]["action_prefix_seeds"] != list(FRESH_SEEDS):
        raise ValueError("fresh action-prefix seeds drifted")
    if set(FRESH_SEEDS) & {20300907, 20300908, 20300917, 20300918}:
        raise ValueError("fresh action-prefix seeds overlap forbidden seeds")
    for key in ("official_cem", "planner_viability", "closed_loop"):
        if freeze["scope_boundary"][key] != "NOT_RUN_BY_SCOPE":
            raise ValueError(f"scope boundary drifted: {key}")
    phase_spec = phase2.get("context_manifest", {})
    if int(phase_spec.get("new_train_target", -1)) != 512 or int(phase_spec.get("heldout_target", -1)) != 8:
        raise ValueError("Phase 2 is not the formal 512/8 context contract")
    return freeze, phase2, load_json((path.resolve().parent / ".." / "LEWM_RECURRENT_STUDENT_FREEZE.json").resolve())


def anchor_spec(length: int) -> dict[str, int]:
    late = int(length) - HORIZON_PRIMITIVES - 1
    if late < 0:
        raise ValueError(f"episode length {length} cannot hold current+25 actions+goal")
    anchors = {"early": 0, "middle": late // 2, "late": late}
    for name, anchor in anchors.items():
        if not (0 <= anchor and anchor + HORIZON_PRIMITIVES < length):
            raise ValueError(f"{name} anchor violates episode bounds for length {length}")
    return anchors


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
    fresh = valid[528:536]
    if len(fresh) != FRESH_EPISODE_COUNT or set(fresh) & (set(prefix) | set(phase3)):
        raise ValueError("fresh valid[528:536] selection overlaps an excluded set")
    return fresh, {"selection_seed": int(selection_seed), "valid_count": len(valid), "old_heldout_episode_ids": old_heldout, "old_train_episode_count": len(old_train), "excluded_phase3_episode_ids": phase3, "fresh_episode_ids": fresh, "selection_slice": "valid[528:536]", "result_dependent_selection": False}


def _anchor_for_ordinal(episode_length: int, ordinal: int) -> tuple[str, int]:
    stratum = ("early", "middle", "late")[int(ordinal) % 3]
    return stratum, anchor_spec(episode_length)[stratum]


def build_balanced_train_rows(model: Any, dataset_path: Path, manifest: Mapping[str, Any], control_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-encode only treatment states while reusing control actions by ordinal."""
    import torch

    train_manifest = list(manifest["splits"]["train"])
    controls = [row for row in control_rows if row.get("split") == "train"]
    if len(train_manifest) != TRAIN_CONTEXTS or len(controls) != TRAIN_CONTEXTS:
        raise ValueError("balanced training requires exactly 512 aligned control contexts")
    balanced: list[dict[str, Any]] = []
    counts = {name: 0 for name in FRESH_ANCHORS}
    with base.HDF5EpisodeSliceReader(dataset_path) as reader:
        for ordinal, (manifest_row, control) in enumerate(zip(train_manifest, controls)):
            if int(control.get("ordinal", ordinal)) != ordinal and "ordinal" in control:
                raise ValueError("control ordinal metadata drifted")
            episode = reader.episode_slice(int(manifest_row["episode_id"]))
            stratum, anchor = _anchor_for_ordinal(int(episode["length"]), ordinal)
            counts[stratum] += 1
            treatment_manifest = {"episode_id": int(manifest_row["episode_id"]), "history_steps": [anchor], "history_action_starts": [anchor], "future_action_start": anchor, "goal_step_offset": anchor + HORIZON_PRIMITIVES}
            raw = reader.read_manifest_row(treatment_manifest)
            current = base._normalise_pixels(raw["pixels"][:1]).unsqueeze(1).to("cuda")
            goal_pixels = base._normalise_pixels(episode["pixels"][anchor + HORIZON_PRIMITIVES : anchor + HORIZON_PRIMITIVES + 1]).unsqueeze(1).to("cuda")
            logged = torch.from_numpy(base.pack_raw_actions(raw["future_actions_raw"]))
            action_history = logged[:1].unsqueeze(0).to("cuda")
            with torch.no_grad():
                latent = model.encode({"pixels": current, "action": action_history})["emb"]
                goal_emb = model.encode({"pixels": goal_pixels})["emb"]
            actions = base._tensor(control["future_actions"], dtype=torch.float32).to("cuda")
            if tuple(actions.shape) != (TRAIN_CANDIDATES, base.HORIZON, base.ACTION_DIM):
                raise ValueError("control candidate bank shape drifted")
            context = latent.expand(TRAIN_CANDIDATES, -1, -1)
            with torch.no_grad():
                targets = base.official_teacher_targets(model, context, actions)
                objective = base._official_objective(model, {"latent_history": latent, "goal_emb": goal_emb}, targets)
            balanced.append({"split": "train", "context_id": f"balanced_{ordinal:04d}_{stratum}", "ordinal": ordinal, "episode_id": int(manifest_row["episode_id"]), "anchor": anchor, "stratum": stratum, "latent_history": latent.detach().cpu(), "action_history": action_history.detach().cpu(), "future_actions": actions.detach().cpu(), "teacher_targets": targets.detach().cpu(), "teacher_objective": objective.detach().cpu(), "rank_shuffle_indices": base._tensor(control["rank_shuffle_indices"]).clone(), "goal_emb": goal_emb.detach().cpu()})
    if len(balanced) != TRAIN_CONTEXTS or counts != {"early": 171, "middle": 171, "late": 170}:
        raise ValueError(f"balanced counts drifted: {counts}")
    return balanced, {"train_contexts": len(balanced), "anchor_counts": counts, "candidate_bank": "control future_actions reused verbatim by ordinal", "candidate_bank_same_by_ordinal": True, "teacher_targets_recomputed_at_balanced_state": True, "rows_generated_on_compute_node": True, "rows_returned": False}


def train_arm(rows: Sequence[Mapping[str, Any]], initial_state: Mapping[str, Any], official_model: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) != TRAIN_CONTEXTS:
        raise ValueError("training bank must contain exactly 512 contexts")
    student = base.make_student("baseline").to("cuda")
    student.load_state_dict(copy.deepcopy(initial_state), strict=True)
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
        total_loss = latent_loss + ema.SCORE_LOSS_WEIGHT * score_loss
        finite = bool(torch.isfinite(total_loss).item() and torch.isfinite(per_horizon).all().item() and torch.isfinite(score_loss).item())
        if not finite:
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        history.append({"step": step, "weighted_latent_mse": float(latent_loss.detach().cpu()), "score_loss": float(score_loss.detach().cpu()), "total_loss": float(total_loss.detach().cpu()), "per_horizon_mse": [float(x) for x in per_horizon.detach().cpu()], "finite": finite})
        if step in SNAPSHOT_STEPS:
            snapshots[f"step_{step}"] = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    first = history[0]["weighted_latent_mse"]
    last10 = statistics.median(item["weighted_latent_mse"] for item in history[-10:])
    return {"snapshots": snapshots, "parameter_count": int(sum(parameter.numel() for parameter in student.parameters())), "last10_to_first_ratio": float(last10 / max(first, 1e-12)), "trace": {"first": history[0], "steps_500": history[499], "steps_1000": history[999], "steps_1500": history[1499], "terminal": history[-1], "last10_weighted_latent_mse_median": last10}}


def absolute_gate(freeze: Mapping[str, Any], evaluation: Mapping[str, Any], training: Mapping[str, Any], causality: Mapping[str, Any], latency: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["inherited_absolute_predictor"]
    metrics = evaluation["overall"]
    convergence = float(training["last10_to_first_ratio"]) <= 0.8
    causal_pass = all(item["passed"] for item in causality.values())
    conditions = {"median_spearman_min": metrics["spearman_median"] >= float(gate["median_spearman_min"]), "minimum_spearman_min": metrics["spearman_minimum"] >= float(gate["minimum_spearman_min"]), "median_top30_min": metrics["top30_median"] >= float(gate["median_top30_min"]), "minimum_top30_min": metrics["top30_minimum"] >= float(gate["minimum_top30_min"]), "median_relative_latent_mse_max": metrics["relative_latent_mse_median"] <= float(gate["median_relative_latent_mse_max"]), "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(gate["positive_spearman_blocks_min"]), "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(gate["positive_top30_blocks_min"])}
    integrity = bool(convergence and metrics["finite"])
    fidelity = all(conditions.values())
    latency_ok = float(latency["reduction"]) >= float(gate["latency_reduction_min"])
    passed = bool(integrity and causal_pass and fidelity and latency_ok)
    return {"status": "GO" if passed else "NO-GO", "integrity_and_convergence": {"status": "PASS" if integrity else "FAIL", "last10_to_first_ratio": training["last10_to_first_ratio"], "convergence_threshold": 0.8}, "causality": {"status": "PASS" if causal_pass else "FAIL", "per_prefix": causality}, "predictor_fidelity_and_ranking": {"status": "PASS" if fidelity else "FAIL", "metrics": {key: metrics[key] for key in ("spearman_median", "spearman_minimum", "top30_median", "top30_minimum", "relative_latent_mse_median", "positive_spearman_blocks", "positive_top30_blocks")}, "conditions": conditions}, "predictor_latency": {"status": "PASS" if latency_ok else "FAIL", "metrics": latency}, "full_cem_viability": "NOT_RUN_BY_SCOPE"}


def stratum_gate(freeze: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["stratum_protection"]
    conditions: dict[str, dict[str, bool]] = {}
    for stratum, metrics in evaluation["strata"].items():
        conditions[stratum] = {"median_spearman_min": metrics["spearman_median"] >= float(gate["median_spearman_min"]), "median_top30_min": metrics["top30_median"] >= float(gate["median_top30_min"]), "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(gate["positive_spearman_blocks_min"]), "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(gate["positive_top30_blocks_min"])}
    return {"status": "GO" if all(all(values.values()) for values in conditions.values()) else "NO-GO", "conditions": conditions, "minimum_thresholds_repeated": False}


def paired_mechanism_gate(freeze: Mapping[str, Any], control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    deltas = []
    for episode in sorted(control["episodes"], key=int):
        ds = float(treatment["episodes"][episode]["spearman_median"]) - float(control["episodes"][episode]["spearman_median"])
        dt = float(treatment["episodes"][episode]["top30_median"]) - float(control["episodes"][episode]["top30_median"])
        deltas.append({"episode_id": int(episode), "spearman_delta_balanced_minus_control": ds, "top30_delta_balanced_minus_control": dt, "joint_improvement_or_nonworsening": bool((ds > 0 and dt >= 0) or (dt > 0 and ds >= 0))})
    med_s = statistics.median(item["spearman_delta_balanced_minus_control"] for item in deltas)
    med_t = statistics.median(item["top30_delta_balanced_minus_control"] for item in deltas)
    joint = sum(bool(item["joint_improvement_or_nonworsening"]) for item in deltas)
    minimum = freeze["gates"]["balanced_mechanism_secondary"]
    conditions = {"spearman_episode_median_delta_min": med_s >= float(minimum["spearman_episode_median_delta_min"]), "top30_episode_median_delta_min": med_t >= float(minimum["top30_episode_median_delta_min"]), "joint_nonworsening_improvement_min_episodes": joint >= int(minimum["joint_nonworsening_improvement_min_episodes"])}
    return {"status": "GO" if all(conditions.values()) else "NO-GO", "paired_unit": "episode after median over six blocks", "median_spearman_delta": float(med_s), "median_top30_delta": float(med_t), "joint_count": joint, "conditions": conditions, "per_episode": deltas, "cannot_replace_absolute_gate": True}


def run(args: Any, freeze: Mapping[str, Any], phase2: Mapping[str, Any], base_freeze: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    base.require_compute_node()
    import torch

    dataset = (args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve()
    manifest_path = args.manifest_512.resolve()
    control_rows_path = args.prepared_rows_512.resolve()
    manifest = load_json(manifest_path)
    if len(manifest.get("splits", {}).get("train", [])) != TRAIN_CONTEXTS:
        raise ValueError("formal 512 manifest missing")
    control_rows = torch.load(control_rows_path, map_location="cpu", weights_only=False)
    control_train = [row for row in control_rows if row.get("split") == "train"]
    if len(control_train) != TRAIN_CONTEXTS:
        raise ValueError("formal control prepared bank is not 512 rows")
    reference = load_json(args.reference_summary.resolve())
    if reference.get("status") != "PREDICTOR_LEVEL_COMPLETE":
        raise ValueError("historical reference is incomplete")
    official_model = base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    balanced_rows, balanced_meta = build_balanced_train_rows(official_model, dataset, manifest, control_rows)
    fresh_ids, selection = select_fresh_episodes(dataset, int(freeze["fresh_evaluation"]["selection_seed"]), manifest)
    fresh_rows, fresh_meta = ema.build_fresh_rows(official_model, dataset, fresh_ids, freeze)
    torch.manual_seed(int(freeze["training"]["initialization_seed"]))
    template = base.make_student("baseline").to("cuda")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    torch.cuda.empty_cache()
    settings = {"training_seed": int(freeze["training"]["training_seed"]), "context_schedule_seed": int(freeze["training"]["context_schedule_seed"])}
    control_training = train_arm(control_rows, initial_state, official_model, settings)
    treatment_training = train_arm(balanced_rows, initial_state, official_model, settings)
    control_eval = ema.evaluate_state(official_model, control_training["snapshots"], fresh_rows, "step0_train")
    treatment_eval = ema.evaluate_state(official_model, treatment_training["snapshots"], fresh_rows, "balanced_temporal_train")
    arms: dict[str, Any] = {}
    for name, training, evaluation in (("step0_train", control_training, control_eval), ("balanced_temporal_train", treatment_training, treatment_eval)):
        student = base.make_student("baseline").to("cuda")
        student.load_state_dict(training["snapshots"]["step_3000"], strict=True)
        student.eval()
        causality = base.causality_test(student, fresh_rows[0], FRESH_SEEDS[0], float(freeze["evaluation"]["causality_tolerance"]))
        latency = base.predictor_latency(official_model, student, fresh_rows[0], warmup=int(freeze["evaluation"]["predictor_latency_warmup"]), repeats=int(freeze["evaluation"]["predictor_latency_repeats"]))
        absolute = absolute_gate(freeze, evaluation["step_3000"], training, causality, latency)
        strata = stratum_gate(freeze, evaluation["step_3000"])
        training_report = {key: value for key, value in training.items() if key != "snapshots"}
        arms[name] = {"training": training_report, "evaluation": evaluation, "terminal_metrics": evaluation["step_3000"]["overall"], "causality": causality, "predictor_latency": latency, "absolute_predictor_gate": absolute, "temporal_stratum_gate": strata}
        del student
        torch.cuda.empty_cache()
    mechanism = paired_mechanism_gate(freeze, control_eval["step_3000"], treatment_eval["step_3000"])
    primary_conditions = {"balanced_absolute_predictor_gate": arms["balanced_temporal_train"]["absolute_predictor_gate"]["status"] == "GO", "balanced_temporal_stratum_gate": arms["balanced_temporal_train"]["temporal_stratum_gate"]["status"] == "GO"}
    summary = {"schema": "lewm-recurrent-student.temporal-balanced-train-summary", "schema_version": 1, "status": "PREDICTOR_LEVEL_COMPLETE", "freeze": str(args.freeze.resolve()), "protocol": str(args.protocol.resolve()), "interface_probe": str(args.interface_probe.resolve()), "interface_contract": dict(contract.__dict__), "source": {"lewm_commit": base_freeze["evidence_boundary"]["source_commit"], "stable_worldmodel_cem_commit": base_freeze["evidence_boundary"]["stable_worldmodel_cem_commit"], "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()), "dataset": str(dataset)}, "hypothesis_provenance": "Phase 3 fresh valid[520:528] observation; valid[528:536] test was not used for design", "training_banks": {"control_manifest": str(manifest_path), "control_prepared_rows": str(control_rows_path), "control_rows_reused_verbatim": True, "balanced": balanced_meta, "prepared_rows_returned": False}, "fresh_selection": selection, "fresh_evaluation": fresh_meta, "shared_training_contract": {"architecture": "LeWMCompactRecurrentTransitionStudent", "hidden_dim": base.HIDDEN_DIM, "attention": False, "conditioner": False, "goal_input": False, "teacher_forcing": False, "objective": "horizon-weighted free-running latent MSE + 0.1 * context-normalized teacher-score SmoothL1 mean", "optimizer": freeze["training"]["optimizer"], "updates": STEPS, "batch_contexts": BATCH_CONTEXTS, "train_candidates": TRAIN_CANDIDATES, "initialization_state_shared": True, "context_schedule_shared": True}, "pairing": {"initialization_seed": settings["training_seed"] - 1, "training_seed": settings["training_seed"], "context_schedule_seed": settings["context_schedule_seed"], "same_initial_state": True, "same_optimizer": True, "same_batch_size": True, "same_context_schedule": True, "same_control_candidate_bank_by_ordinal": True, "candidate_is_not_an_independent_statistical_unit": True, "fresh_action_prefix_seeds": list(FRESH_SEEDS)}, "arms": arms, "paired_deltas": mechanism, "primary_gate": {"status": "GO" if all(primary_conditions.values()) else "NO-GO", "arm": "balanced_temporal_train", "conditions": primary_conditions, "absolute_predictor_gate": arms["balanced_temporal_train"]["absolute_predictor_gate"], "temporal_stratum_gate": arms["balanced_temporal_train"]["temporal_stratum_gate"], "secondary_balanced_mechanism_gate": mechanism}, "gpu_telemetry": {"source": "job.log nvidia-smi 5-second samples", "required": True}, "scope_checks": {"formal_512_control_reused": "PASS", "balanced_rows_compute_only": "PASS", "fresh_valid_528_536": "PASS", "same_init_schedule": "PASS", "same_candidate_bank_by_ordinal": "PASS", "nested_episode_aggregation": "PASS", "no_result_dependent_selection": "PASS"}, "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}, "claim_boundary": "Temporal-balanced training evidence is predictor-level only; no official CEM, planner viability, closed-loop PushT, encoder speedup, or native deployment claim."}
    write_json(args.output.resolve() / "lewm_temporal_balanced_train_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze, phase2, base_freeze = load_freeze(args.freeze, args.phase2_freeze)
    if args.mode == "status":
        value = {"schema": "lewm-recurrent-student.temporal-balanced-train", "status": "READY", "training_rows": 512, "updates": 3000, "balanced_counts": {"early": 171, "middle": 171, "late": 170}, "fresh_selection": "valid[528:536]", "fresh_blocks": 48, "action_prefix_seeds": list(FRESH_SEEDS), "official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    for path in (args.interface_probe, args.manifest_512, args.prepared_rows_512, args.reference_summary, args.phase2_freeze, args.dataset or (args.stablewm_home / "pusht_expert_train.h5")):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = base.load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, phase2, base_freeze, contract)
    print(json.dumps({"status": summary["status"], "primary_gate": summary["primary_gate"]["status"], "output": str((args.output / "lewm_temporal_balanced_train_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
