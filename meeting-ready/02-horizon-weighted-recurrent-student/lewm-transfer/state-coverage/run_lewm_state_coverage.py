#!/usr/bin/env python3
"""LeWM PushT Phase 2 state/context coverage, predictor-level only."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

STATE_DIR = Path(__file__).resolve().parent
TRANSFER_DIR = STATE_DIR.parent
sys.path.insert(0, str(TRANSFER_DIR / "score-distill"))
sys.path.insert(0, str(TRANSFER_DIR / "dense-rank"))
import run_lewm_score_distill as score  # noqa: E402
import run_lewm_dense_rank as dense  # noqa: E402

base = dense.base

TRAIN_CANDIDATES = 64
HELDOUT_CONTEXTS = 8
HELDOUT_BLOCKS = 16
SNAPSHOT_STEPS = (500, 1000, 1500, 3000)
NEW_ARMS = {
    "256x3000": (256, 3000),
    "512x1500": (512, 1500),
    "512x3000": (512, 3000),
}
TREATMENT_GATE = {
    "spearman_median_min": 0.974011,
    "top30_median_min": 0.816667,
    "relative_latent_mse_median_max": 0.0175,
    "positive_top30_blocks_exact": 16,
    "minimum_top30_strictly_positive": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-256", type=Path, required=True)
    parser.add_argument("--prepared-rows-256", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--action-low", type=float, nargs=2, required=True)
    parser.add_argument("--action-high", type=float, nargs=2, required=True)
    parser.add_argument("--gaussian-std", type=float, required=True)
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


def load_freeze(path: Path) -> dict[str, Any]:
    value = dense.load_freeze(path.resolve())
    base.validate_freeze(value)
    overlay = value.get("experiment_overlay", {})
    if overlay.get("experiment_name") != "lewm_score_distill_state_context_coverage_phase2":
        raise ValueError("unexpected state-coverage freeze")
    train = overlay.get("training", {})
    expected = {
        "train_candidates": TRAIN_CANDIDATES,
        "score_top_count": score.SCORE_TOP_COUNT,
        "batch_contexts": 8,
        "initialization_seed": 20300901,
        "same_context_schedule_seed": 20300904,
    }
    for key, wanted in expected.items():
        if int(train.get(key, -1)) != wanted:
            raise ValueError(f"state-coverage training field drifted: {key}")
    if float(train.get("score_loss_weight", -1.0)) != score.SCORE_LOSS_WEIGHT:
        raise ValueError("score loss weight drifted")
    if [float(x) for x in train.get("horizon_weights", [])] != list(base.HORIZON_WEIGHTS):
        raise ValueError("horizon weights drifted")
    if overlay.get("context_manifest", {}).get("old_train_target") != 256 or overlay.get("context_manifest", {}).get("new_train_target") != 512:
        raise ValueError("context targets drifted")
    return value


def phase_settings(freeze: Mapping[str, Any]) -> dict[str, Any]:
    overlay = freeze["experiment_overlay"]
    train = overlay["training"]
    seeds = freeze["shared_data_and_schedule"]["seeds"]
    return {
        "batch_contexts": int(train["batch_contexts"]),
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "seeds": {
            "initialization": int(train["initialization_seed"]),
            "training": int(train["training_seed"]),
            "context_schedule": int(train["same_context_schedule_seed"]),
            "heldout_action_prefix": [int(x) for x in seeds["heldout_action_prefix"]],
        },
    }


def freeze_for_context_count(freeze: Mapping[str, Any], count: int) -> dict[str, Any]:
    value = copy.deepcopy(dict(freeze))
    value["shared_data_and_schedule"]["context_manifest"]["train_context_target"] = int(count)
    return value


def manifest_prefix_check(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    old_train = list(old["splits"]["train"])
    new_train = list(new["splits"]["train"])
    old_heldout = list(old["splits"]["heldout"])
    new_heldout = list(new["splits"]["heldout"])
    checks = {
        "heldout_verbatim": old_heldout == new_heldout,
        "old_train_is_new_train_prefix": old_train == new_train[: len(old_train)],
        "new_train_count": len(new_train) == 512,
        "heldout_count": len(new_heldout) == HELDOUT_CONTEXTS,
        "selection_seed_equal": old.get("selection_seed") == new.get("selection_seed") == 20300903,
        "history_contract_equal": old.get("history_contract") == new.get("history_contract"),
        "temporal_anchor_unchanged": all(
            row.get("history_steps") == [0]
            and row.get("history_action_starts") == [0]
            and row.get("future_action_start") == 0
            and row.get("goal_step_offset") == 25
            for row in new_train + new_heldout
        ),
        "new_suffix_episode_disjoint_from_heldout": not (
            {int(row["episode_id"]) for row in new_train[256:]}
            & {int(row["episode_id"]) for row in new_heldout}
        ),
    }
    checks["status"] = "PASS" if all(value for key, value in checks.items() if key != "status") else "FAIL"
    if checks["status"] != "PASS":
        raise RuntimeError(f"manifest prefix invariance failed: {checks}")
    return checks


def tensor_equal(first: Any, second: Any) -> bool:
    import torch

    if not torch.is_tensor(first) or not torch.is_tensor(second):
        return first == second
    return bool(torch.equal(first, second))


def prepared_row_equivalence(old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old_train = [row for row in old_rows if row.get("split") == "train"]
    new_train = [row for row in new_rows if row.get("split") == "train"]
    old_heldout = [row for row in old_rows if row.get("split") == "heldout"]
    new_heldout = [row for row in new_rows if row.get("split") == "heldout"]
    train_keys = ("context_id", "latent_history", "action_history", "future_actions", "teacher_targets", "teacher_objective", "goal_emb", "rank_shuffle_indices")
    heldout_keys = ("context_id", "latent_history", "action_history", "future_actions", "teacher_targets", "teacher_objective", "goal_emb")

    def compare(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> bool:
        if len(left) != len(right):
            return False
        return all(all(tensor_equal(a.get(key), b.get(key)) for key in keys) for a, b in zip(left, right))

    checks = {
        "train_prefix_256_bitwise_equal": compare(old_train, new_train[:256], train_keys),
        "heldout_bank_bitwise_equal": compare(old_heldout, new_heldout, heldout_keys),
        "new_train_count": len(new_train) == 512,
        "heldout_count": len(new_heldout) == HELDOUT_CONTEXTS,
    }
    checks["status"] = "PASS" if all(value for key, value in checks.items() if key != "status") else "FAIL"
    if checks["status"] != "PASS":
        raise RuntimeError(f"prepared-row scope check failed: {checks}")
    return checks


def evaluate_snapshots(model: Any, snapshots: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], snapshot_steps: Sequence[int]) -> dict[str, Any]:
    import torch

    heldout = [row for row in rows if row.get("split") == "heldout"]
    evaluation: dict[str, Any] = {}
    for step in snapshot_steps:
        snapshot_name = f"step_{step}"
        student = base.make_student("baseline").to("cuda")
        student.load_state_dict(snapshots[snapshot_name], strict=True)
        student.eval()
        blocks = []
        for row in heldout:
            for block in range(2):
                item = base._stage_a_metrics_for_block(model, student, row, block)
                item["context_id"] = row.get("context_id")
                item["pairing_key"] = f"seed_index={block}:context={row.get('context_id')}"
                blocks.append(item)
        spearman = [float(item["spearman"]) for item in blocks]
        top30 = [float(item["top30_overlap"]) for item in blocks]
        evaluation[snapshot_name] = {
            "per_block": blocks,
            "terminal_ranking": {
                "spearman_median": float(statistics.median(spearman)),
                "spearman_minimum": min(spearman),
                "top30_overlap_median": float(statistics.median(top30)),
                "top30_overlap_minimum": min(top30),
            },
            "relative_latent_mse_median": float(statistics.median(float(item["relative_latent_mse"]) for item in blocks)),
            "finite": all(bool(item["finite"]) for item in blocks),
        }
        del student
        torch.cuda.empty_cache()
    return evaluation


def paired_delta(reference: Mapping[str, Any], treatment: Mapping[str, Any], key: str, lower_is_better: bool = False) -> dict[str, Any]:
    left = {item["pairing_key"]: float(item[key]) for item in reference["per_block"]}
    right = {item["pairing_key"]: float(item[key]) for item in treatment["per_block"]}
    if set(left) != set(right):
        raise ValueError(f"pairing keys differ for {key}")
    deltas = {name: right[name] - left[name] for name in left}
    values = list(deltas.values())
    improve = sum(value < 0 for value in values) if lower_is_better else sum(value > 0 for value in values)
    worse = sum(value > 0 for value in values) if lower_is_better else sum(value < 0 for value in values)
    return {"improve": improve, "worse": worse, "tie": len(values) - improve - worse, "median_delta": float(statistics.median(values)), "deltas_by_pairing_key": deltas}


def treatment_metrics(evaluation: Mapping[str, Any], terminal_step: int) -> dict[str, Any]:
    final = evaluation[f"step_{terminal_step}"]
    terminal = final["terminal_ranking"]
    return {
        "spearman_median": float(terminal["spearman_median"]),
        "spearman_minimum": float(terminal["spearman_minimum"]),
        "top30_median": float(terminal["top30_overlap_median"]),
        "top30_minimum": float(terminal["top30_overlap_minimum"]),
        "relative_latent_mse_median": float(final["relative_latent_mse_median"]),
        "positive_spearman_blocks": sum(float(item["spearman"]) > 0 for item in final["per_block"]),
        "positive_top30_blocks": sum(float(item["top30_overlap"]) > 0 for item in final["per_block"]),
    }


def treatment_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "spearman_median_min": metrics["spearman_median"] >= TREATMENT_GATE["spearman_median_min"],
        "top30_median_min": metrics["top30_median"] >= TREATMENT_GATE["top30_median_min"],
        "relative_latent_mse_median_max": metrics["relative_latent_mse_median"] <= TREATMENT_GATE["relative_latent_mse_median_max"],
        "positive_top30_blocks_exact": metrics["positive_top30_blocks"] == TREATMENT_GATE["positive_top30_blocks_exact"],
        "minimum_top30_strictly_positive": metrics["top30_minimum"] > 0.0,
    }
    return {"status": "GO" if all(conditions.values()) else "NO-GO", "arm": "512x1500", "metrics": dict(metrics), "conditions": conditions}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], contract: Any) -> dict[str, Any]:
    base.require_compute_node()
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    base_freeze = copy.deepcopy(dict(freeze))
    base_freeze.pop("experiment_overlay", None)
    old_manifest = base.validate_manifest(args.manifest_256.resolve(), base_freeze)
    phase_freeze = freeze_for_context_count(freeze, 512)
    manifest_512_path = output / "context_manifest_512.json"
    if manifest_512_path.is_file():
        manifest_512 = base.validate_manifest(manifest_512_path, phase_freeze)
    else:
        manifest_512 = base.build_context_manifest(dataset, manifest_512_path, phase_freeze)
        manifest_512 = base.validate_manifest(manifest_512_path, phase_freeze)
    manifest_check = manifest_prefix_check(old_manifest, manifest_512)

    old_rows_path = args.prepared_rows_256.resolve()
    if not old_rows_path.is_file():
        raise FileNotFoundError(old_rows_path)
    old_rows = torch.load(old_rows_path, map_location="cpu", weights_only=False)
    old_meta = dense.validate_dense_rows(old_rows, base_freeze)

    official_model = base.load_official_checkpoint(args.stablewm_home.resolve())
    official_model.requires_grad_(False)
    prep_dir = output / "prepared_512"
    prep_dir.mkdir(parents=True, exist_ok=True)
    prep_args = argparse.Namespace(action_low=args.action_low, action_high=args.action_high, gaussian_std=args.gaussian_std)
    new_rows = dense.prepare_dense_rows(official_model, dataset, manifest_512, phase_freeze, prep_args, prep_dir)
    new_meta = dense.validate_dense_rows(new_rows, phase_freeze)
    row_check = prepared_row_equivalence(old_rows, new_rows)

    reference = load_json(args.reference_summary.resolve())
    reference_arm = reference["arms"]["score_distill"]
    reference_eval = reference_arm["evaluation"]
    settings_base = phase_settings(freeze)
    torch.manual_seed(int(settings_base["seeds"]["initialization"]))
    template = base.make_student("baseline").to("cuda")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    torch.cuda.empty_cache()
    # score.train_arm reads this validated module-level list for snapshots.
    base.SNAPSHOT_STEPS = SNAPSHOT_STEPS

    rows_by_count = {256: old_rows, 512: new_rows}
    arm_results: dict[str, Any] = {}
    for arm_name, (context_count, steps) in NEW_ARMS.items():
        settings = dict(settings_base)
        settings["steps"] = int(steps)
        arm_output = output / "training" / arm_name
        result = score.train_arm(rows_by_count[context_count], settings, arm_output, official_model, initial_state, "score_distill")
        evaluation = evaluate_snapshots(official_model, result["snapshots"], rows_by_count[context_count], [step for step in SNAPSHOT_STEPS if step <= steps])
        terminal_student = base.make_student("baseline").to("cuda")
        terminal_student.load_state_dict(result["snapshots"][f"step_{steps}"], strict=True)
        terminal_student.eval()
        heldout = [row for row in rows_by_count[context_count] if row.get("split") == "heldout"]
        causality = base.causality_test(terminal_student, heldout[0], int(settings_base["seeds"]["heldout_action_prefix"][0]), 1e-6)
        latency = base.predictor_latency(official_model, terminal_student, heldout[0])
        gates = base.predictor_gates(freeze, {"last10_to_first_ratio": result["last10_to_first_ratio"]}, {"step_1500": evaluation[f"step_{steps}"]}, causality, latency)
        metrics = treatment_metrics(evaluation, steps)
        arm_results[arm_name] = {
            "train_contexts": context_count,
            "updates": steps,
            "terminal_snapshot": f"step_{steps}",
            "parameter_count": result["parameter_count"],
            "training": {"last10_to_first_ratio": result["last10_to_first_ratio"], "per_step": result["per_step"]},
            "evaluation": evaluation,
            "terminal_metrics": metrics,
            "causality": causality,
            "predictor_latency": latency,
            "absolute_predictor_gates": gates,
        }
        del terminal_student
        torch.cuda.empty_cache()

    ref_step = reference_eval["step_1500"]
    paired = {}
    for arm_name, result in arm_results.items():
        terminal = result["evaluation"][result["terminal_snapshot"]]
        paired[arm_name] = {
            "spearman": paired_delta(ref_step, terminal, "spearman"),
            "top30_overlap": paired_delta(ref_step, terminal, "top30_overlap"),
            "relative_latent_mse": paired_delta(ref_step, terminal, "relative_latent_mse", lower_is_better=True),
        }
    primary_gate = treatment_gate(arm_results["512x1500"]["terminal_metrics"])
    summary = {
        "schema": "lewm-recurrent-student.state-context-coverage-summary",
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_probe": str(args.interface_probe.resolve()),
        "interface_contract": {key: value for key, value in contract.__dict__.items()},
        "source": {
            "lewm_commit": freeze["evidence_boundary"]["source_commit"],
            "stable_worldmodel_cem_commit": freeze["evidence_boundary"]["stable_worldmodel_cem_commit"],
            "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()),
            "dataset": str(dataset),
        },
        "reference": {
            "arm": "256x1500",
            "formal_job": "24916520.pbs101",
            "summary": str(args.reference_summary.resolve()),
            "manifest": str(args.manifest_256.resolve()),
            "prepared_rows": str(old_rows_path),
            "score_distill_terminal_metrics": treatment_metrics(reference_eval, 1500),
        },
        "manifests": {
            "256": str(args.manifest_256.resolve()),
            "512": str(manifest_512_path),
            "prefix_invariance": manifest_check,
        },
        "prepared_rows": {
            "256": str(old_rows_path),
            "512": str(prep_dir / "prepared_rows.pt"),
            "old_meta": old_meta,
            "new_meta": new_meta,
            "scope_checks": row_check,
        },
        "shared_training_contract": {
            "architecture": "LeWMCompactRecurrentTransitionStudent",
            "hidden_dim": base.HIDDEN_DIM,
            "attention": False,
            "conditioner": False,
            "goal_input": False,
            "teacher_forcing": False,
            "score_top_count": score.SCORE_TOP_COUNT,
            "score_top_weight": score.SCORE_TOP_WEIGHT,
            "score_other_weight": score.SCORE_OTHER_WEIGHT,
            "score_std_floor": score.SCORE_STD_FLOOR,
            "score_smooth_l1_beta": score.SCORE_SMOOTH_L1_BETA,
            "score_loss_weight": score.SCORE_LOSS_WEIGHT,
            "horizon_weights": list(base.HORIZON_WEIGHTS),
            "initialization_state_shared": True,
            "optimizer_shared": True,
            "batch_contexts": settings_base["batch_contexts"],
            "candidate_generation_and_evaluation_unchanged": True,
        },
        "pairing": {
            "initialization_seed": settings_base["seeds"]["initialization"],
            "training_seed": settings_base["seeds"]["training"],
            "context_schedule_seed": settings_base["seeds"]["context_schedule"],
            "same_initial_state": True,
            "same_optimizer": True,
            "same_batch_size": True,
            "256x3000_schedule_prefix_is_256x1500_schedule": True,
            "512x3000_schedule_prefix_is_512x1500_schedule": True,
            "same_heldout_candidate_bank": True,
        },
        "arms": arm_results,
        "paired_deltas_vs_256x1500_reference": paired,
        "primary_treatment_gate": primary_gate,
        "scope_checks": {"manifest_prefix_invariance": manifest_check, "prepared_row_equivalence": row_check, "scope_leakage": "PASS"},
        "stage_b": {"status": "NOT_RUN_BY_SCOPE", "full_cem_viability": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Predictor-level state/context coverage comparison only. This does not claim official CEM viability, planner viability, closed-loop PushT success, encode_obs speedup, or native deployment benefit.",
    }
    write_json(output / "lewm_state_coverage_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = load_freeze(args.freeze)
    if args.mode == "status":
        value = {
            "schema": "lewm-recurrent-student.state-context-coverage",
            "status": "READY",
            "new_arms": list(NEW_ARMS),
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "new_512_data_generated_only_in_pbs": True,
            "official_cem": "NOT_RUN_BY_SCOPE",
        }
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value))
        return 0
    for path in (args.interface_probe, args.manifest_256, args.prepared_rows_256, args.reference_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = base.load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, contract)
    print(json.dumps({"status": summary["status"], "primary_treatment_gate": summary["primary_treatment_gate"]["status"], "output": str((args.output / "lewm_state_coverage_summary.json").resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
