"""Independent post-calibration tie-break evaluation for selective teacher correction."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

OLD_DIR = Path(__file__).resolve().parents[1] / "selective-teacher-correction"
if str(OLD_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_DIR))
import run_selective_teacher_correction as old  # noqa: E402

cem = old.cem
boundary = old.boundary
SCHEMA = "lewm-recurrent-student.selective-teacher-correction-tiebreak-runner"
SELECTION_SEED = 20300903
TEST_SLICE = (584, 592)
CALIBRATION_SLICE = (576, 584)
FRESH_SEEDS = (20301101, 20301102)
TEST_INNOVATION_BASE = 20301104
ELITE_COUNT = 30
CEM_CHECKPOINTS = (10, 20, 30)
NUM_CANDIDATES = 300
STD_FLOOR = 1e-6
TIMING_WARMUPS = 3
TIMING_REPEATS = 10
CALIBRATION_CALLS = 36
CALIBRATION_BLOCKS = 144
CALIBRATION_U1_BLOCKS = 37
EXPECTED_G36 = 0.6875929732552584
EXPECTED_G37 = 0.6465405171168568
EXPECTED_TAU_G = 0.6670667451860576
CALIBRATION_REASON = "no calibration midpoint has 0 < call_rate <= 0.25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--main-checkpoint", type=Path, required=True)
    parser.add_argument("--sentinel-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--temporal-freeze", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("status", "preflight", "run"), default="status")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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


def norm_path(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "lewm-recurrent-student.selective-teacher-correction-tiebreak-freeze":
        raise ValueError("unexpected tiebreak freeze schema")
    if freeze.get("status") != "frozen_after_calibration_before_test":
        raise ValueError("tiebreak freeze status drifted")
    model = freeze.get("model_and_artifacts", {})
    if model.get("historical_job") != "25239551.pbs101" or model.get("main_arm") != "treatment" or model.get("sentinel_arm") != "control":
        raise ValueError("historical model arms drifted")
    artifact_dir = norm_path(model.get("historical_artifact_dir", ""))
    for key, filename in {
        "historical_summary_path": "cem_distribution_distill_summary.json",
        "main_checkpoint_path": "treatment_step1000.pt",
        "sentinel_checkpoint_path": "control_step1000.pt",
    }.items():
        if norm_path(model.get(key, "")) != f"{artifact_dir}/{filename}":
            raise ValueError(f"historical artifact path drifted: {key}")
    if norm_path(model.get("anchor_driver_checkpoint_path", "")) != "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/anchor-aligned-bank/artifacts/25223859.pbs101/anchor_aligned_bank_step3000.pt":
        raise ValueError("anchor driver path drifted")
    if model.get("anchor_driver_source") != "anchor_aligned_bank_treatment" or int(model.get("anchor_driver_updates", -1)) != 3000:
        raise ValueError("anchor driver provenance drifted")

    calibration = freeze.get("calibration_source", {})
    if calibration.get("job") != "25269182.pbs101" or calibration.get("status") != "INCONCLUSIVE" or calibration.get("reason") != CALIBRATION_REASON:
        raise ValueError("calibration source binding drifted")
    if norm_path(calibration.get("source_runner", "")) != "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/run_selective_teacher_correction.py" or norm_path(calibration.get("source_protocol", "")) != "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/PROTOCOL.zh.md":
        raise ValueError("calibration source runner/protocol drifted")
    if calibration.get("slice") != "valid[576:584]" or calibration.get("selection_seed") != SELECTION_SEED:
        raise ValueError("calibration selection binding drifted")
    if calibration.get("contexts") != 24 or calibration.get("context_seed_trajectories") != 48 or calibration.get("trajectory_blocks") != CALIBRATION_BLOCKS:
        raise ValueError("calibration bank dimensions drifted")
    if calibration.get("saved_rounds") != list(CEM_CHECKPOINTS) or calibration.get("teacher_shadow_only") is not True:
        raise ValueError("calibration bank provenance drifted")

    tie = freeze.get("calibration_tiebreak", {})
    if tie.get("tie_value_u") != 1.0 or tie.get("tie_blocks") != CALIBRATION_U1_BLOCKS or tie.get("cutoff_rank") != CALIBRATION_CALLS or tie.get("next_rank") != CALIBRATION_CALLS + 1:
        raise ValueError("tie-break ranks drifted")
    for key, expected in {"cutoff_g": EXPECTED_G36, "next_g": EXPECTED_G37, "tau_g": EXPECTED_TAU_G}.items():
        if not math.isclose(float(tie.get(key)), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"tie-break constant drifted: {key}")
    if tie.get("rule") != "call full teacher iff u > 1.0 OR (u == 1.0 AND g > tau_g)" or tie.get("strict_inequality") is not True:
        raise ValueError("tie-break rule drifted")
    if tie.get("teacher_free_metric") is not True or tie.get("test_threshold_not_tuned") is not True or tie.get("block_id_or_cross_test_rank_used") is not False:
        raise ValueError("tie-break leakage contract drifted")

    selection = freeze.get("selection_and_banks", {})
    if selection.get("selection_seed") != SELECTION_SEED or selection.get("test_slice") != "valid[584:592]" or selection.get("test_excluded_valid_prefix") != 584:
        raise ValueError("test selection drifted")
    if selection.get("action_prefix_seeds") != list(FRESH_SEEDS) or selection.get("test_innovation_seed_base") != TEST_INNOVATION_BASE or selection.get("trajectory_blocks_per_set") != CALIBRATION_BLOCKS:
        raise ValueError("test bank contract drifted")

    cem_cfg = freeze.get("cem_collection", {})
    for key, expected in {"iterations": cem.CEM_ITERATIONS, "saved_checkpoints": list(CEM_CHECKPOINTS), "num_samples": NUM_CANDIDATES, "topk": ELITE_COUNT, "horizon": cem.HORIZON, "packed_action_dim": cem.ACTION_DIM}.items():
        if cem_cfg.get(key) != expected:
            raise ValueError(f"CEM contract drifted: {key}")
    if cem_cfg.get("candidate_zero_is_pre_update_mu") is not True or cem_cfg.get("std_unbiased") is not True or cem_cfg.get("teacher_shadow_only") is not True:
        raise ValueError("CEM semantics drifted")

    timing = freeze.get("timing", {})
    if timing.get("scope") != "all 144 test trajectory blocks (8 episodes x 3 anchors x 2 seeds x 3 rounds)" or timing.get("warmups") != TIMING_WARMUPS or timing.get("repeats") != TIMING_REPEATS:
        raise ValueError("timing contract drifted")
    gates = freeze.get("gates", {})
    for key, expected in {"test_call_rate_max": 0.35, "primary_selective_minus_main_episode_median_max": -0.1, "strictly_improved_episodes_min": 5, "risk_capture_min": 0.75, "catastrophic_blocks_min": 4, "selective_minus_expected_random_episode_median_max": 0.0, "latency_reduction_vs_teacher300_min": 0.3}.items():
        if gates.get(key) != expected:
            raise ValueError(f"gate drifted: {key}")
    scope = freeze.get("scope", {})
    if scope.get("official_cem_deployment") != "NOT_RUN_BY_SCOPE" or scope.get("closed_loop") != "NOT_RUN_BY_SCOPE" or scope.get("old_25269182_result_rewritten") is not False:
        raise ValueError("scope drifted")
    return dict(freeze)


def validate_protocol(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    required = ("valid[576:584]", "valid[584:592]", "tau_g", "g > tau_g", "teacher-free", "inconclusive", "scientific agent skills")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol missing frozen terms: {missing}")


def validate_source_metric_definition(freeze: Mapping[str, Any]) -> dict[str, Any]:
    calibration = freeze["calibration_source"]
    expected_runner = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/run_selective_teacher_correction.py"
    expected_protocol = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/PROTOCOL.zh.md"
    if norm_path(calibration.get("source_runner", "")) != expected_runner or norm_path(calibration.get("source_protocol", "")) != expected_protocol:
        raise ValueError("calibration source runner/protocol binding drifted")
    actual_runner = norm_path(old.__file__)
    if actual_runner != expected_runner:
        raise ValueError("calibration source runner path drifted")
    source = inspect.getsource(old.evaluate_block)
    marker = '"global_rank_disagreement_secondary": 1.0 - float(reference.base._spearman(main_cost, sentinel_cost))'
    if marker not in source:
        raise ValueError("source runner does not expose the frozen teacher-free g definition")
    return {"source_runner": actual_runner, "definition": "g=1-_spearman(main_cost,sentinel_cost) from old evaluate_block; teacher-free", "verified_static": True, "source_summary": norm_path(calibration["summary_path"]), "source_bank": norm_path(calibration["bank_path"])}


def validate_historical(summary: Mapping[str, Any], freeze: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return old.validate_historical_summary(summary, freeze, args.anchor_checkpoint, args.main_checkpoint, args.sentinel_checkpoint, args.historical_summary)


def validate_calibration_summary(summary: Mapping[str, Any], freeze: Mapping[str, Any], summary_path: Path) -> dict[str, Any]:
    source = freeze["calibration_source"]
    if norm_path(summary_path.resolve()) != norm_path(source["summary_path"]):
        raise ValueError("calibration summary path is not the exact 25269182 artifact")
    if summary.get("status") != source["status"] or summary.get("reason") != source["reason"]:
        raise ValueError("calibration summary status/reason drifted")
    selection = summary.get("calibration_selection", {})
    if selection.get("selection_seed") != SELECTION_SEED or selection.get("selection_slice") != "valid[576:584]" or selection.get("excluded_valid_prefix") != 576:
        raise ValueError("calibration summary selection drifted")
    bank = summary.get("calibration_evaluation", {}).get("bank", {})
    expected_bank = norm_path(source["bank_path"])
    if norm_path(bank.get("path", "")) != expected_bank or bank.get("trajectory_blocks") != CALIBRATION_BLOCKS or bank.get("contexts") != 24 or bank.get("saved_rounds") != list(CEM_CHECKPOINTS) or bank.get("samples_per_iteration") != NUM_CANDIDATES:
        raise ValueError("calibration summary bank provenance drifted")
    blocks = summary.get("calibration_evaluation", {}).get("blocks", [])
    if len(blocks) != CALIBRATION_BLOCKS:
        raise ValueError("calibration summary block count drifted")
    for block in blocks:
        if not math.isfinite(float(block["uncertainty"])) or not math.isfinite(float(block["global_rank_disagreement_secondary"])):
            raise ValueError("calibration summary contains non-finite routing metric")
    u1_count = sum(float(block["uncertainty"]) == 1.0 for block in blocks)
    if u1_count != CALIBRATION_U1_BLOCKS:
        raise ValueError(f"calibration u=1 block count drifted: {u1_count}")
    return {"summary_status": summary["status"], "reason": summary["reason"], "selection": selection, "bank": bank, "blocks": len(blocks), "u1_blocks": u1_count}


def recompute_routing_metrics(reference: Any, official: Any, main_student: Any, sentinel_student: Any, row: Mapping[str, Any], bank: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    actions = cem.tensor(bank["actions"])
    with torch.no_grad():
        main_cost = cem.student_costs(reference, official, main_student, row, actions)
        sentinel_cost = cem.student_costs(reference, official, sentinel_student, row, actions)
    main_top = old.stable_top30(main_cost)
    sentinel_top = old.stable_top30(sentinel_cost)
    main_set = set(int(value) for value in main_top.detach().cpu())
    sentinel_set = set(int(value) for value in sentinel_top.detach().cpu())
    return {
        "uncertainty": 1.0 - float(len(main_set & sentinel_set)) / ELITE_COUNT,
        "global_rank_disagreement_secondary": 1.0 - float(reference.base._spearman(main_cost, sentinel_cost)),
        "finite": bool(torch.isfinite(main_cost).all().item() and torch.isfinite(sentinel_cost).all().item()),
    }


def recompute_calibration_blocks(reference: Any, manifest: Mapping[str, Any], temporal_freeze: Mapping[str, Any], official: Any, main_student: Any, sentinel_student: Any, dataset: Path, calibration_summary: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    source = freeze["calibration_source"]
    bank_path = Path(source["bank_path"])
    if not bank_path.is_file():
        raise FileNotFoundError(f"exact 25269182 calibration bank is missing: {bank_path}")
    calibration_ids, selection = old.select_fresh(dataset, manifest, CALIBRATION_SLICE, "calibration_recompute")
    expected_ids = calibration_summary["calibration_selection"]["fresh_episode_ids"]
    if [int(value) for value in calibration_ids] != [int(value) for value in expected_ids]:
        raise ValueError("recomputed calibration episode selection differs from exact 25269182 summary")
    rows, metadata = old.make_fresh_rows(reference, dataset, manifest, temporal_freeze, official, calibration_ids, CALIBRATION_SLICE, "calibration_recompute")
    bank_records = torch.load(bank_path, map_location="cpu", weights_only=False)
    if not isinstance(bank_records, list) or len(bank_records) != CALIBRATION_BLOCKS:
        raise ValueError("exact 25269182 calibration bank record count drifted")
    summary_blocks = calibration_summary["calibration_evaluation"]["blocks"]
    recomputed: list[dict[str, Any]] = []
    matched = 0
    for index, bank in enumerate(bank_records):
        row = rows[int(bank["row_index"])]
        metrics = recompute_routing_metrics(reference, official, main_student, sentinel_student, row, bank)
        pairing_key = f"episode={int(bank['episode_id'])}:anchor={bank['stratum']}:seed={int(bank['action_prefix_seed'])}:round={int(bank['round'])}"
        expected = summary_blocks[index]
        if expected.get("pairing_key") != pairing_key:
            raise ValueError(f"calibration bank pairing drifted at index {index}")
        if not math.isclose(float(expected["uncertainty"]), float(metrics["uncertainty"]), rel_tol=0.0, abs_tol=1e-9) or not math.isclose(float(expected["global_rank_disagreement_secondary"]), float(metrics["global_rank_disagreement_secondary"]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"calibration routing recomputation differs from exact summary at index {index}")
        recomputed.append({**metrics, "pairing_key": pairing_key, "episode_id": int(bank["episode_id"]), "round": int(bank["round"]), "stratum": str(bank["stratum"]), "action_prefix_seed": int(bank["action_prefix_seed"])})
        matched += 1
    return {"blocks": recomputed, "metadata": metadata, "selection": selection, "bank_path": norm_path(bank_path), "bank_records": len(bank_records), "summary_metric_matches": matched, "teacher_free": True, "teacher_costs_used_for_routing": False}


def compute_tiebreak(summary: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    blocks = summary["calibration_evaluation"]["blocks"]
    tied = [block for block in blocks if float(block["uncertainty"]) == 1.0]
    ranked = sorted(tied, key=lambda block: (-float(block["global_rank_disagreement_secondary"]), str(block.get("pairing_key", ""))))
    if len(ranked) != CALIBRATION_U1_BLOCKS:
        raise ValueError("calibration tied block count changed")
    cutoff_g = float(ranked[CALIBRATION_CALLS - 1]["global_rank_disagreement_secondary"])
    next_g = float(ranked[CALIBRATION_CALLS]["global_rank_disagreement_secondary"])
    tau_g = (cutoff_g + next_g) / 2.0
    calls = sum(float(block["uncertainty"]) > 1.0 or (float(block["uncertainty"]) == 1.0 and float(block["global_rank_disagreement_secondary"]) > tau_g) for block in blocks)
    if not math.isclose(cutoff_g, EXPECTED_G36, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(next_g, EXPECTED_G37, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(tau_g, EXPECTED_TAU_G, rel_tol=0.0, abs_tol=1e-12) or calls != CALIBRATION_CALLS:
        raise ValueError(f"calibration tie-break recomputation drifted: cutoff={cutoff_g}, next={next_g}, tau={tau_g}, calls={calls}")
    frozen = freeze["calibration_tiebreak"]
    if not math.isclose(float(frozen["tau_g"]), tau_g, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("recomputed tau does not match freeze")
    return {"tie_value_u": 1.0, "tie_blocks": len(tied), "cutoff_rank": CALIBRATION_CALLS, "cutoff_g": cutoff_g, "next_rank": CALIBRATION_CALLS + 1, "next_g": next_g, "tau_g": tau_g, "calls": calls, "call_rate": calls / CALIBRATION_BLOCKS, "rule": "u > 1.0 OR (u == 1.0 AND g > tau_g)", "teacher_free": True}


def apply_tiebreak_policy(blocks: Sequence[Mapping[str, Any]], tau_g: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in blocks:
        record = dict(item)
        u = float(item["uncertainty"])
        g = float(item["global_rank_disagreement_secondary"])
        called = u > 1.0 or (u == 1.0 and g > tau_g)
        record["selective_teacher_call"] = called
        record["selective_regret"] = 0.0 if called else float(item["main_regret"])
        record["routing_g"] = g
        record["routing_rule"] = "u > 1.0 OR (u == 1.0 AND g > tau_g)"
        output.append(record)
    calls = sum(bool(item["selective_teacher_call"]) for item in output)
    return output, {"calls": calls, "blocks": len(output), "call_rate": calls / max(len(output), 1), "tau_g": tau_g, "call_rule": "u > 1.0 OR (u == 1.0 AND g > tau_g)", "teacher_free": True}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any) -> dict[str, Any]:
    cem.require_compute_node()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = old.load_json(args.manifest_512.resolve())
    temporal_freeze = old.load_json(args.temporal_freeze.resolve())
    historical_summary = old.load_json(args.historical_summary.resolve())
    calibration_summary = old.load_json(args.calibration_summary.resolve())
    historical = validate_historical(historical_summary, freeze, args)
    source_metric = validate_source_metric_definition(freeze)
    calibration_source = validate_calibration_summary(calibration_summary, freeze, args.calibration_summary)
    start_state, start_meta = cem.load_start_state(reference, args.anchor_checkpoint.resolve())
    main_state, main_meta = boundary.load_checkpoint(reference, args.main_checkpoint.resolve(), "treatment")
    sentinel_state, sentinel_meta = boundary.load_checkpoint(reference, args.sentinel_checkpoint.resolve(), "control")
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.requires_grad_(False)
    main_student = reference.base.make_student("baseline").to("cuda")
    sentinel_student = reference.base.make_student("baseline").to("cuda")
    main_student.load_state_dict(copy.deepcopy(main_state), strict=True)
    sentinel_student.load_state_dict(copy.deepcopy(sentinel_state), strict=True)
    main_student.eval()
    sentinel_student.eval()
    driver = reference.base.make_student("baseline").to("cuda")
    driver.load_state_dict(copy.deepcopy(start_state), strict=True)
    driver.eval()

    calibration_recompute = recompute_calibration_blocks(reference, manifest, temporal_freeze, official, main_student, sentinel_student, args.dataset, calibration_summary, freeze)
    recomputed_calibration_summary = {"calibration_evaluation": {"blocks": calibration_recompute["blocks"]}}
    tiebreak = compute_tiebreak(recomputed_calibration_summary, freeze)

    test_ids, test_selection = old.select_fresh(args.dataset, manifest, TEST_SLICE, "test")
    test_rows, test_meta = old.make_fresh_rows(reference, args.dataset, manifest, temporal_freeze, official, test_ids, TEST_SLICE, "test")
    test_banks, test_bank_meta = old.collect_fresh_banks(reference, official, driver, test_rows, TEST_INNOVATION_BASE, "test", output)
    test_blocks = [old.evaluate_block(reference, official, main_student, sentinel_student, test_rows[int(bank["row_index"])], bank) for bank in test_banks]
    policy_blocks, policy_meta = apply_tiebreak_policy(test_blocks, float(tiebreak["tau_g"]))
    quality = old.quality_summary(policy_blocks, policy_meta)
    timing = old.timing_summary(reference, official, main_student, sentinel_student, test_rows, test_banks, policy_blocks)
    gates = freeze["gates"]
    risk = quality["risk"]
    random_delta = statistics.median(list(quality["selective_minus_expected_random_episode_mean"].values()))
    conditions = {
        "calibration_recomputed_calls_exact": int(tiebreak["calls"]) == CALIBRATION_CALLS and math.isclose(float(tiebreak["call_rate"]), 0.25, rel_tol=0.0, abs_tol=1e-12),
        "test_call_rate_le_max": float(policy_meta["call_rate"]) <= float(gates["test_call_rate_max"]),
        "primary_selective_minus_main_le_threshold": float(quality["primary_median_delta"]) <= float(gates["primary_selective_minus_main_episode_median_max"]),
        "strictly_improved_episodes_min": int(quality["strictly_improved_episodes"]) >= int(gates["strictly_improved_episodes_min"]),
        "risk_catastrophic_block_count_min": int(risk["test_catastrophic_blocks"]) >= int(gates["catastrophic_blocks_min"]),
        "risk_capture_min": risk["capture_rate"] is not None and float(risk["capture_rate"]) >= float(gates["risk_capture_min"]),
        "selective_not_worse_than_expected_random": random_delta <= float(gates["selective_minus_expected_random_episode_median_max"]),
        "latency_reduction_vs_teacher300_min": float(timing["latency_reduction_vs_teacher300"]) >= float(gates["latency_reduction_vs_teacher300_min"]),
        "finite": bool(quality["finite"] and timing["teacher_only"]["finite"] and timing["selective"]["finite"]),
        "shared_provenance": bool(historical["non_concurrent"] and historical["teacher_shadow_only"] and calibration_source["blocks"] == CALIBRATION_BLOCKS and len(test_blocks) == CALIBRATION_BLOCKS),
    }
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "COMPLETE",
        "post_calibration_refinement": True,
        "disclosure": "Rule refined after calibration discreteness; only test is independent evidence.",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_contract": cem.validate_interface(reference, args.interface_probe.resolve()),
        "historical": historical,
        "calibration_source": {**calibration_source, "summary_path": norm_path(args.calibration_summary.resolve()), "metric_definition": source_metric},
        "calibration_recompute": calibration_recompute,
        "calibration_tiebreak": tiebreak,
        "start_checkpoint": start_meta,
        "main_checkpoint": main_meta,
        "sentinel_checkpoint": sentinel_meta,
        "test_selection": test_selection,
        "test_evaluation": {"metadata": test_meta, "bank": test_bank_meta, "blocks": policy_blocks, "quality": quality},
        "timing": timing,
        "gates": {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions, "risk_threshold": old.CATASTROPHIC_REGRET, "paired_unit": "episode; mean over 18 trajectory blocks"},
        "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Independent fixed-observation post-calibration selective candidate-ranking evidence only; no official CEM deployment or closed-loop claim.",
    }
    write_json(output / "selective_teacher_correction_tiebreak_summary.json", jsonable(summary))
    return summary


def preflight(args: argparse.Namespace, freeze: Mapping[str, Any]) -> dict[str, Any]:
    reference = cem.load_reference()
    interface = cem.validate_interface(reference, args.interface_probe.resolve())
    value = {
        "schema": SCHEMA,
        "status": "PASS",
        "model_work_started": False,
        "reference_import": "PASS",
        "interface": interface,
        "post_calibration_refinement": True,
        "calibration_source_job": "25269182.pbs101",
        "calibration": "exact summary; u=1 ties; recomputed g36/g37/tau/calls in compute run",
        "test": "valid[584:592]",
        "test_trajectory_blocks": CALIBRATION_BLOCKS,
        "timing_warmups": TIMING_WARMUPS,
        "timing_repeats": TIMING_REPEATS,
        "pbs_compute_only": True,
        "official_cem": "NOT_RUN_BY_SCOPE",
        "closed_loop": "NOT_RUN_BY_SCOPE",
    }
    write_json(args.output.resolve() / "preflight_status.json", value)
    return value


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = validate_freeze(load_json(args.freeze.resolve()))
    validate_protocol(args.protocol.resolve())
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "post_calibration_refinement": True, "calibration_source_job": "25269182.pbs101", "test": "valid[584:592]", "tie_rule": "u > 1.0 OR (u == 1.0 AND g > tau_g)", "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
    elif args.mode == "preflight":
        value = preflight(args, freeze)
    else:
        reference = cem.load_reference()
        value = run(args, freeze, reference)
    print(json.dumps(jsonable(value), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
