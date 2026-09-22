#!/usr/bin/env python3
"""Fixed-observation selective full-teacher correction for LeWM CEM ranking."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

BOUNDARY_DIR = Path(__file__).resolve().parents[1] / "cem-boundary-ranking"
if str(BOUNDARY_DIR) not in sys.path:
    sys.path.insert(0, str(BOUNDARY_DIR))
import run_cem_boundary_ranking as boundary  # noqa: E402

cem = boundary.cem

SCHEMA = "lewm-recurrent-student.selective-teacher-correction-runner"
SELECTION_SEED = 20300903
FRESH_SEEDS = (20301101, 20301102)
CALIBRATION_SLICE = (576, 584)
TEST_SLICE = (584, 592)
CALIBRATION_INNOVATION_BASE = 20301103
TEST_INNOVATION_BASE = 20301104
CEM_CHECKPOINTS = (10, 20, 30)
NUM_CANDIDATES = 300
ELITE_COUNT = 30
UNCERTAINTY_TARGET = 0.25
TEST_CALL_RATE_MAX = 0.35
CATASTROPHIC_REGRET = 1.5
STD_FLOOR = 1e-6
RANDOM_BASELINE = "analytic_same_budget_expectation"
TIMING_WARMUPS = 3
TIMING_REPEATS = 10


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


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "lewm-recurrent-student.selective-teacher-correction-freeze":
        raise ValueError("unexpected selective-correction freeze schema")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("freeze must be frozen_before_results")
    model = freeze.get("model_and_artifacts", {})
    if model.get("historical_job") != "25239551.pbs101":
        raise ValueError("historical job drifted")
    if model.get("main_arm") != "treatment" or model.get("sentinel_arm") != "control":
        raise ValueError("main/sentinel arms drifted")
    artifact_dir = str(model.get("historical_artifact_dir", "")).replace("\\", "/").rstrip("/")
    exact_paths = {
        "main_checkpoint_path": "treatment_step1000.pt",
        "sentinel_checkpoint_path": "control_step1000.pt",
        "historical_summary_path": "cem_distribution_distill_summary.json",
    }
    for key, filename in exact_paths.items():
        expected = f"{artifact_dir}/{filename}"
        if str(model.get(key, "")).replace("\\", "/").rstrip("/") != expected:
            raise ValueError(f"historical artifact path drifted: {key}")
    selection = freeze.get("selection_and_banks", {})
    expected_selection = {
        "selection_seed": SELECTION_SEED,
        "calibration_slice": "valid[576:584]",
        "calibration_excluded_valid_prefix": 576,
        "test_slice": "valid[584:592]",
        "test_excluded_valid_prefix": 584,
        "action_prefix_seeds": list(FRESH_SEEDS),
        "trajectory_blocks_per_set": 144,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"selection contract drifted: {key}")
    cem_cfg = freeze.get("cem_collection", {})
    for key, expected in {
        "iterations": cem.CEM_ITERATIONS,
        "saved_checkpoints": list(CEM_CHECKPOINTS),
        "num_samples": NUM_CANDIDATES,
        "topk": ELITE_COUNT,
        "horizon": cem.HORIZON,
        "packed_action_dim": cem.ACTION_DIM,
    }.items():
        if cem_cfg.get(key) != expected:
            raise ValueError(f"CEM contract drifted: {key}")
    if cem_cfg.get("candidate_zero_is_pre_update_mu") is not True or cem_cfg.get("std_unbiased") is not True:
        raise ValueError("CEM candidate-zero/std contract drifted")
    if cem_cfg.get("teacher_shadow_only") is not True:
        raise ValueError("teacher shadow provenance drifted")
    uncertainty = freeze.get("uncertainty_rule", {})
    if uncertainty.get("scalar") != "u = 1 - |top30_main intersection top30_sentinel| / 30":
        raise ValueError("uncertainty scalar drifted")
    if uncertainty.get("ranking") != "torch.topk(student_cost, k=30, largest=False, sorted=True).indices; pinned LeWM solver semantics":
        raise ValueError("uncertainty ranking semantics drifted")
    if uncertainty.get("global_rank_disagreement") != "secondary report only; never combined with u, thresholded, or used in a sweep":
        raise ValueError("secondary global-rank diagnostic drifted")
    if uncertainty.get("target_call_rate") != UNCERTAINTY_TARGET or uncertainty.get("maximum_test_call_rate") != TEST_CALL_RATE_MAX:
        raise ValueError("uncertainty call-rate contract drifted")
    threshold = uncertainty.get("threshold_calibration", {})
    if threshold.get("call_rule") != "call full teacher iff u > tau" or threshold.get("no_feasible_threshold") != "INCONCLUSIVE_STOP":
        raise ValueError("threshold rule drifted")
    metrics = freeze.get("metrics", {})
    if metrics.get("catastrophic_main_regret_threshold") != CATASTROPHIC_REGRET or metrics.get("minimum_test_catastrophic_blocks") != 4:
        raise ValueError("risk threshold drifted")
    timing = freeze.get("timing", {})
    if timing.get("scope") != "all 144 test trajectory blocks (8 episodes x 3 anchors x 2 seeds x 3 rounds)":
        raise ValueError("timing scope drifted")
    if timing.get("warmups") != TIMING_WARMUPS or timing.get("repeats") != TIMING_REPEATS:
        raise ValueError("timing repeats drifted")
    gates = freeze.get("gates", {})
    for key, expected in {
        "test_call_rate_max": TEST_CALL_RATE_MAX,
        "primary_selective_minus_main_episode_median_max": -0.1,
        "strictly_improved_episodes_min": 5,
        "risk_capture_min": 0.75,
        "catastrophic_blocks_min": 4,
        "selective_minus_expected_random_episode_median_max": 0.0,
        "latency_reduction_vs_teacher300_min": 0.3,
    }.items():
        if gates.get(key) != expected:
            raise ValueError(f"gate drifted: {key}")
    if freeze.get("scope", {}).get("official_cem_deployment") != "NOT_RUN_BY_SCOPE" or freeze.get("scope", {}).get("closed_loop") != "NOT_RUN_BY_SCOPE":
        raise ValueError("scope drifted")
    return dict(freeze)


def validate_protocol(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    required = ("valid[576:584]", "valid[584:592]", "midpoint", "u >", "analytic", "closed-loop", "scientific agent skills")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol missing frozen terms: {missing}")


def validate_historical_summary(summary: Mapping[str, Any], freeze: Mapping[str, Any], anchor_path: Path, main_path: Path, sentinel_path: Path, summary_path: Path) -> dict[str, Any]:
    if summary.get("status") != "COMPLETE":
        raise ValueError("historical summary is not COMPLETE")
    model = freeze["model_and_artifacts"]
    expected = {
        "anchor": str(model["anchor_driver_checkpoint_path"]).replace("\\", "/").rstrip("/"),
        "main": str(model["main_checkpoint_path"]).replace("\\", "/").rstrip("/"),
        "sentinel": str(model["sentinel_checkpoint_path"]).replace("\\", "/").rstrip("/"),
        "summary": str(model["historical_summary_path"]).replace("\\", "/").rstrip("/"),
    }
    actual = {
        "anchor": str(anchor_path.resolve()).replace("\\", "/").rstrip("/"),
        "main": str(main_path.resolve()).replace("\\", "/").rstrip("/"),
        "sentinel": str(sentinel_path.resolve()).replace("\\", "/").rstrip("/"),
        "summary": str(summary_path.resolve()).replace("\\", "/").rstrip("/"),
    }
    if actual != expected:
        raise ValueError("historical checkpoint/summary args are outside frozen 25239551 artifact directory")
    training = summary.get("training", {})
    for arm, key in (("treatment", "main"), ("control", "sentinel")):
        entry = training.get(arm, {})
        if str(entry.get("checkpoint", "")).replace("\\", "/").rstrip("/") != expected[key]:
            raise ValueError(f"historical {arm} checkpoint path drifted")
        if int(entry.get("trace", {}).get("updates", -1)) != 1000:
            raise ValueError(f"historical {arm} update count drifted")
    start_checkpoint = summary.get("start_checkpoint", {})
    start_path = str(start_checkpoint.get("path", "")).replace("\\", "/").rstrip("/")
    if start_path != expected["anchor"]:
        raise ValueError("historical start checkpoint path drifted from frozen anchor driver")
    if start_checkpoint.get("source") != "anchor_aligned_bank_treatment":
        raise ValueError("historical start checkpoint source drifted")
    start_provenance = start_checkpoint.get("provenance", {})
    if start_provenance.get("source") != "anchor_aligned_bank_treatment" or int(start_provenance.get("updates", -1)) != 3000:
        raise ValueError("historical start checkpoint provenance drifted")
    bank = summary.get("bank_construction", {})
    if bank.get("teacher_did_not_select_or_update") is not True or bank.get("tail_counts") != {"round_10": 11, "round_20": 11, "round_30": 10}:
        raise ValueError("historical teacher-shadow bank provenance drifted")
    start = summary.get("source", {})
    if not str(start.get("anchor_rows", "")).endswith("anchor_aligned_train_rows.pt"):
        raise ValueError("historical anchor source missing")
    return {
        "summary_status": summary.get("status"),
        "job": "25239551.pbs101",
        "main_arm": "treatment",
        "sentinel_arm": "control",
        "non_concurrent": True,
        "main_checkpoint": expected["main"],
        "sentinel_checkpoint": expected["sentinel"],
        "training_updates": {"main": 1000, "sentinel": 1000},
        "teacher_shadow_only": True,
    }


def select_fresh(dataset: Path, manifest: Mapping[str, Any], fresh_slice: tuple[int, int], label: str) -> tuple[list[int], dict[str, Any]]:
    import h5py

    with h5py.File(dataset.resolve(), "r") as handle:
        lengths = [int(value) for value in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= cem.HORIZON * 5 + 1]
    random.Random(SELECTION_SEED).shuffle(valid)
    prefix = [int(row["episode_id"]) for row in manifest["splits"]["heldout"] + manifest["splits"]["train"]]
    if valid[: len(prefix)] != prefix:
        raise ValueError("valid shuffle prefix cannot be reproduced")
    selected = valid[fresh_slice[0] : fresh_slice[1]]
    if len(selected) != 8 or set(selected) & set(valid[: fresh_slice[0]]):
        raise ValueError(f"{label} slice overlaps excluded valid prefix")
    return selected, {
        "label": label,
        "selection_seed": SELECTION_SEED,
        "selection_slice": f"valid[{fresh_slice[0]}:{fresh_slice[1]}]",
        "excluded_valid_prefix": fresh_slice[0],
        "fresh_episode_ids": selected,
        "result_dependent_selection": False,
    }


def make_fresh_rows(reference: Any, dataset: Path, manifest: Mapping[str, Any], temporal_freeze: Mapping[str, Any], official: Any, fresh_ids: Sequence[int], fresh_slice: tuple[int, int], label: str) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    cem.FRESH_SEEDS = FRESH_SEEDS
    cem.FRESH_SLICE = fresh_slice
    cem.NUM_CANDIDATES = NUM_CANDIDATES
    rows, metadata = cem.make_fresh_rows(reference, dataset, manifest, temporal_freeze, official, fresh_ids)
    if len(rows) != 24 or int(metadata.get("blocks", -1)) != 48:
        raise ValueError(f"{label} fresh rows/blocks drifted")
    return rows, {**metadata, "label": label, "selection_slice": f"valid[{fresh_slice[0]}:{fresh_slice[1]}]", "selection_seed": SELECTION_SEED, "action_prefix_seeds": list(FRESH_SEEDS)}


def collect_fresh_banks(reference: Any, official: Any, driver: Any, rows: Sequence[Mapping[str, Any]], innovation_base: int, label: str, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for block, seed in enumerate(FRESH_SEEDS):
            generator = torch.Generator(device="cpu").manual_seed(innovation_base + row_index * len(FRESH_SEEDS) + block)
            innovations = torch.randn((cem.CEM_ITERATIONS, NUM_CANDIDATES, cem.HORIZON, cem.ACTION_DIM), generator=generator)
            saved, metadata, targets, costs = cem.run_student_cem(reference, official, driver, row, innovations, CEM_CHECKPOINTS)
            for round_index in CEM_CHECKPOINTS:
                records.append({
                    "row_index": row_index,
                    "context_id": row.get("context_id"),
                    "episode_id": int(row["episode_id"]),
                    "anchor": int(row["anchor"]),
                    "stratum": row["stratum"],
                    "action_prefix_seed": int(seed),
                    "round": round_index,
                    "actions": saved[round_index],
                    "teacher_targets": targets[round_index],
                    "teacher_cost": costs[round_index],
                })
            if (row_index * len(FRESH_SEEDS) + block + 1) % 8 == 0:
                print(json.dumps({"phase": f"{label}_cem_collection", "blocks_completed": row_index * len(FRESH_SEEDS) + block + 1, "blocks_total": 48}), flush=True)
    path = output / f"fresh_{label}_cem_trajectory_banks.pt"
    torch.save(records, path)
    return records, {"label": label, "trajectory_blocks": len(records), "contexts": len(rows), "iterations": cem.CEM_ITERATIONS, "samples_per_iteration": NUM_CANDIDATES, "saved_rounds": list(CEM_CHECKPOINTS), "innovation_seed_base": innovation_base, "path": str(path)}


def stable_top30(cost: Any) -> Any:
    import torch

    return torch.topk(cost, k=ELITE_COUNT, largest=False, sorted=True).indices


def evaluate_block(reference: Any, official: Any, main_student: Any, sentinel_student: Any, row: Mapping[str, Any], bank: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    actions = cem.tensor(bank["actions"])
    teacher_cost = cem.tensor(bank["teacher_cost"])
    with torch.no_grad():
        main_cost = cem.student_costs(reference, official, main_student, row, actions)
        sentinel_cost = cem.student_costs(reference, official, sentinel_student, row, actions)
    teacher_top = stable_top30(teacher_cost)
    main_top = stable_top30(main_cost)
    sentinel_top = stable_top30(sentinel_cost)
    teacher_set = set(int(value) for value in teacher_top.detach().cpu())
    main_set = set(int(value) for value in main_top.detach().cpu())
    sentinel_set = set(int(value) for value in sentinel_top.detach().cpu())
    std = teacher_cost.std(unbiased=False)
    if float(std.detach().cpu()) <= STD_FLOOR:
        raise FloatingPointError("degenerate teacher population std")
    teacher_mean = teacher_cost.index_select(0, teacher_top).mean()
    main_regret = (teacher_cost.index_select(0, main_top).mean() - teacher_mean) / std
    uncertainty = 1.0 - float(len(main_set & sentinel_set)) / ELITE_COUNT
    return {
        "episode_id": int(bank["episode_id"]),
        "anchor": int(bank["anchor"]),
        "stratum": str(bank["stratum"]),
        "action_prefix_seed": int(bank["action_prefix_seed"]),
        "round": int(bank["round"]),
        "context_id": row.get("context_id"),
        "pairing_key": f"episode={int(bank['episode_id'])}:anchor={bank['stratum']}:seed={int(bank['action_prefix_seed'])}:round={int(bank['round'])}",
        "main_regret": float(main_regret.detach().cpu()),
        "uncertainty": uncertainty,
        "teacher_cost_std_population": float(std.detach().cpu()),
        "main_top30_overlap": float(len(teacher_set & main_set) / ELITE_COUNT),
        "sentinel_top30_overlap": float(len(teacher_set & sentinel_set) / ELITE_COUNT),
        "global_rank_disagreement_secondary": 1.0 - float(reference.base._spearman(main_cost, sentinel_cost)),
        "finite": bool(torch.isfinite(main_cost).all().item() and torch.isfinite(sentinel_cost).all().item() and torch.isfinite(teacher_cost).all().item()),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return float("nan")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def percentile90(values: Sequence[float]) -> float:
    return percentile(values, 0.90)


def percentile95(values: Sequence[float]) -> float:
    return percentile(values, 0.95)


def metric_distribution(blocks: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(item[key]) for item in blocks]
    by_episode: dict[str, list[float]] = defaultdict(list)
    by_round: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in blocks:
        episode = str(item["episode_id"])
        round_key = str(item["round"])
        by_episode[episode].append(float(item[key]))
        by_round[round_key][episode].append(float(item[key]))
    episode_means = {episode: statistics.mean(items) for episode, items in sorted(by_episode.items(), key=lambda item: int(item[0]))}
    round_episode_means = {
        round_key: {episode: statistics.mean(items) for episode, items in sorted(episode_values.items(), key=lambda item: int(item[0]))}
        for round_key, episode_values in sorted(by_round.items(), key=lambda item: int(item[0]))
    }
    return {
        "blocks": len(values),
        "block_median": statistics.median(values),
        "block_p90": percentile90(values),
        "episode_mean": episode_means,
        "episode_median": statistics.median(list(episode_means.values())),
        "episode_p90": percentile90(list(episode_means.values())),
        "round_episode_mean": round_episode_means,
        "round_episode_median": {round_key: statistics.median(list(values_by_episode.values())) for round_key, values_by_episode in round_episode_means.items()},
        "finite": all(math.isfinite(value) for value in values),
    }


def apply_policy(blocks: Sequence[Mapping[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in blocks:
        record = dict(item)
        called = float(item["uncertainty"]) > threshold
        record["selective_teacher_call"] = called
        record["selective_regret"] = 0.0 if called else float(item["main_regret"])
        output.append(record)
    calls = sum(bool(item["selective_teacher_call"]) for item in output)
    return output, {"calls": calls, "blocks": len(output), "call_rate": calls / max(len(output), 1), "threshold": threshold, "call_rule": "u > tau"}


def quality_summary(blocks: Sequence[Mapping[str, Any]], policy_meta: Mapping[str, Any]) -> dict[str, Any]:
    selective = metric_distribution(blocks, "selective_regret")
    main = metric_distribution(blocks, "main_regret")
    expected_random_episode_mean: dict[str, float] = {}
    selective_calls: dict[str, int] = defaultdict(int)
    for item in blocks:
        if bool(item["selective_teacher_call"]):
            selective_calls[str(item["episode_id"])] += 1
    for episode, mean in main["episode_mean"].items():
        expected_random_episode_mean[episode] = (1.0 - selective_calls[episode] / 18.0) * mean
    expected_random = {
        "episode_mean": expected_random_episode_mean,
        "episode_median": statistics.median(list(expected_random_episode_mean.values())),
        "episode_p90": percentile90(list(expected_random_episode_mean.values())),
        "blocks": len(blocks),
        "analytic": True,
        "formula": "(1 - selective_calls_in_episode / 18) * main_episode_mean",
    }
    selective_delta = {episode: selective["episode_mean"][episode] - main["episode_mean"][episode] for episode in main["episode_mean"]}
    random_delta = {episode: selective["episode_mean"][episode] - expected_random_episode_mean[episode] for episode in main["episode_mean"]}
    catastrophic = [item for item in blocks if float(item["main_regret"]) >= CATASTROPHIC_REGRET]
    captured = sum(bool(item["selective_teacher_call"]) for item in catastrophic)
    return {
        "policy": dict(policy_meta),
        "main_student_only": main,
        "selective_teacher_correction": selective,
        "expected_random_same_budget": expected_random,
        "global_rank_disagreement_secondary": metric_distribution(blocks, "global_rank_disagreement_secondary"),
        "selective_minus_main_episode_mean": selective_delta,
        "selective_minus_expected_random_episode_mean": random_delta,
        "primary_median_delta": statistics.median(list(selective_delta.values())),
        "primary_p90_delta": percentile90(list(selective_delta.values())),
        "strictly_improved_episodes": sum(value < 0.0 for value in selective_delta.values()),
        "risk": {
            "main_regret_threshold": CATASTROPHIC_REGRET,
            "test_catastrophic_blocks": len(catastrophic),
            "captured_by_teacher": captured,
            "capture_rate": (captured / len(catastrophic)) if catastrophic else None,
            "status": "INCONCLUSIVE" if len(catastrophic) < 4 else "EVALUATED",
        },
        "finite": all(bool(item["finite"]) and math.isfinite(float(item["selective_regret"])) for item in blocks),
        "shared_pairing": len({item["pairing_key"] for item in blocks}) == len(blocks),
    }


class ThresholdInconclusive(RuntimeError):
    def __init__(self, details: Mapping[str, Any]):
        super().__init__("no calibration midpoint has 0 < call_rate <= 0.25")
        self.details = dict(details)


def choose_threshold(calibration_blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    levels = sorted({float(item["uncertainty"]) for item in calibration_blocks})
    candidates = [(levels[index - 1] + levels[index]) / 2.0 for index in range(1, len(levels))]
    records = []
    for tau in candidates:
        calls = sum(float(item["uncertainty"]) > tau for item in calibration_blocks)
        rate = calls / max(len(calibration_blocks), 1)
        records.append({"tau": tau, "calls": calls, "blocks": len(calibration_blocks), "call_rate": rate})
    feasible = [item for item in records if 0.0 < float(item["call_rate"]) <= UNCERTAINTY_TARGET]
    if not feasible:
        raise ThresholdInconclusive({"levels": levels, "candidates": records, "rule": "midpoints; 0 < qcal <= 0.25; maximize qcal"})
    chosen = max(feasible, key=lambda item: (float(item["call_rate"]), float(item["tau"])))
    return {"levels": levels, "candidates": records, "chosen": chosen, "rule": "u > tau; maximize qcal subject to 0 < qcal <= 0.25"}


def _timed_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _student_cost_timed(reference: Any, official: Any, student: Any, row: Mapping[str, Any], latent: Any, actions_cpu: Any) -> Any:
    import torch

    actions = actions_cpu.to(device="cuda", dtype=torch.float32)
    context = latent.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        prediction = student(context, actions)
        return reference.base._official_objective(official, row, prediction)


def _teacher_cost_timed(reference: Any, official: Any, row: Mapping[str, Any], latent: Any, actions_cpu: Any) -> Any:
    import torch

    actions = actions_cpu.to(device="cuda", dtype=torch.float32)
    context = latent.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        targets = reference.base.official_teacher_targets(official, context, actions)
        return reference.base._official_objective(official, row, targets)


def timing_summary(reference: Any, official: Any, main_student: Any, sentinel_student: Any, rows: Sequence[Mapping[str, Any]], banks: Sequence[Mapping[str, Any]], policy_blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    prepared = []
    for bank in banks:
        row = rows[int(bank["row_index"])]
        latent = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
        actions_cpu = torch.as_tensor(bank["actions"], dtype=torch.float32, device="cpu")
        prepared.append((row, latent, actions_cpu))
    call_flags = [bool(item["selective_teacher_call"]) for item in policy_blocks]
    arms = ("teacher_only", "main_student_only", "two_student_uncertainty", "selective")
    samples: dict[str, list[float]] = {arm: [] for arm in arms}

    def invoke(arm: str, index: int) -> None:
        row, latent, actions_cpu = prepared[index]
        if arm == "teacher_only":
            teacher_cost = _teacher_cost_timed(reference, official, row, latent, actions_cpu)
            _ = torch.topk(teacher_cost, k=ELITE_COUNT, largest=False, sorted=True).indices
        elif arm == "main_student_only":
            main_cost = _student_cost_timed(reference, official, main_student, row, latent, actions_cpu)
            _ = torch.topk(main_cost, k=ELITE_COUNT, largest=False, sorted=True).indices
        else:
            main_cost = _student_cost_timed(reference, official, main_student, row, latent, actions_cpu)
            sentinel_cost = _student_cost_timed(reference, official, sentinel_student, row, latent, actions_cpu)
            main_top = torch.topk(main_cost, k=ELITE_COUNT, largest=False, sorted=True).indices
            sentinel_top = torch.topk(sentinel_cost, k=ELITE_COUNT, largest=False, sorted=True).indices
            _ = 1.0 - float(len(set(int(value) for value in main_top.detach().cpu()) & set(int(value) for value in sentinel_top.detach().cpu()))) / ELITE_COUNT
            if arm == "selective" and call_flags[index]:
                teacher_cost = _teacher_cost_timed(reference, official, row, latent, actions_cpu)
                _ = torch.topk(teacher_cost, k=ELITE_COUNT, largest=False, sorted=True).indices

    for repeat_index in range(TIMING_WARMUPS + TIMING_REPEATS):
        order = arms if repeat_index % 2 == 0 else tuple(reversed(arms))
        for index in range(len(prepared)):
            for arm in order:
                _timed_synchronize()
                start = time.perf_counter()
                invoke(arm, index)
                _timed_synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if repeat_index >= TIMING_WARMUPS:
                    samples[arm].append(elapsed_ms)
    result = {}
    for arm, values in samples.items():
        result[arm] = {"samples": len(values), "mean_ms": statistics.mean(values), "p95_ms": percentile95(values), "finite": all(math.isfinite(value) for value in values)}
    teacher_mean = float(result["teacher_only"]["mean_ms"])
    selective_mean = float(result["selective"]["mean_ms"])
    result["latency_reduction_vs_teacher300"] = (teacher_mean - selective_mean) / max(teacher_mean, 1e-12)
    result["scope"] = "144 test trajectory blocks; 3 warmups x 10 repeats; fixed alternating arm order"
    return result


def run(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any) -> dict[str, Any]:
    cem.require_compute_node()
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest_512.resolve())
    temporal_freeze = load_json(args.temporal_freeze.resolve())
    historical_summary = load_json(args.historical_summary.resolve())
    historical = validate_historical_summary(historical_summary, freeze, args.anchor_checkpoint, args.main_checkpoint, args.sentinel_checkpoint, args.historical_summary)
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

    calibration_ids, calibration_selection = select_fresh(args.dataset, manifest, CALIBRATION_SLICE, "calibration")
    calibration_rows, calibration_meta = make_fresh_rows(reference, args.dataset, manifest, temporal_freeze, official, calibration_ids, CALIBRATION_SLICE, "calibration")
    calibration_banks, calibration_bank_meta = collect_fresh_banks(reference, official, driver, calibration_rows, CALIBRATION_INNOVATION_BASE, "calibration", output)
    calibration_blocks = [evaluate_block(reference, official, main_student, sentinel_student, calibration_rows[int(bank["row_index"])], bank) for bank in calibration_banks]
    try:
        threshold = choose_threshold(calibration_blocks)
    except ThresholdInconclusive as exc:
        summary = {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "INCONCLUSIVE",
            "reason": str(exc),
            "threshold_calibration": exc.details,
            "freeze": str(args.freeze.resolve()),
            "protocol": str(args.protocol.resolve()),
            "historical": historical,
            "start_checkpoint": start_meta,
            "main_checkpoint": main_meta,
            "sentinel_checkpoint": sentinel_meta,
            "calibration_selection": calibration_selection,
            "calibration_evaluation": {"metadata": calibration_meta, "bank": calibration_bank_meta, "blocks": calibration_blocks},
            "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        }
        write_json(output / "selective_teacher_correction_summary.json", jsonable(summary))
        return summary

    test_ids, test_selection = select_fresh(args.dataset, manifest, TEST_SLICE, "test")
    test_rows, test_meta = make_fresh_rows(reference, args.dataset, manifest, temporal_freeze, official, test_ids, TEST_SLICE, "test")
    test_banks, test_bank_meta = collect_fresh_banks(reference, official, driver, test_rows, TEST_INNOVATION_BASE, "test", output)
    test_blocks = [evaluate_block(reference, official, main_student, sentinel_student, test_rows[int(bank["row_index"])], bank) for bank in test_banks]
    policy_blocks, policy_meta = apply_policy(test_blocks, float(threshold["chosen"]["tau"]))
    quality = quality_summary(policy_blocks, policy_meta)
    timing = timing_summary(reference, official, main_student, sentinel_student, test_rows, test_banks, policy_blocks)
    gates_cfg = freeze["gates"]
    primary_delta = float(quality["primary_median_delta"])
    risk = quality["risk"]
    conditions = {
        "threshold_calibration_feasible": True,
        "test_call_rate_le_max": float(policy_meta["call_rate"]) <= float(gates_cfg["test_call_rate_max"]),
        "primary_selective_minus_main_le_threshold": primary_delta <= float(gates_cfg["primary_selective_minus_main_episode_median_max"]),
        "strictly_improved_episodes_min": int(quality["strictly_improved_episodes"]) >= int(gates_cfg["strictly_improved_episodes_min"]),
        "risk_catastrophic_block_count_min": int(risk["test_catastrophic_blocks"]) >= int(gates_cfg["catastrophic_blocks_min"]),
        "risk_capture_min": risk["capture_rate"] is not None and float(risk["capture_rate"]) >= float(gates_cfg["risk_capture_min"]),
        "selective_not_worse_than_expected_random": True,
        "latency_reduction_vs_teacher300_min": float(timing["latency_reduction_vs_teacher300"]) >= float(gates_cfg["latency_reduction_vs_teacher300_min"]),
        "finite": bool(quality["finite"] and timing["teacher_only"]["finite"] and timing["selective"]["finite"]),
        "shared_provenance": bool(historical["non_concurrent"] and historical["teacher_shadow_only"] and len(test_blocks) == 144),
    }
    # The random comparison gate is the median of episode deltas, not an arbitrary block.
    random_delta_values = list(quality["selective_minus_expected_random_episode_mean"].values())
    conditions["selective_not_worse_than_expected_random"] = statistics.median(random_delta_values) <= float(gates_cfg["selective_minus_expected_random_episode_median_max"])
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_contract": cem.validate_interface(reference, args.interface_probe.resolve()),
        "historical": historical,
        "start_checkpoint": start_meta,
        "main_checkpoint": main_meta,
        "sentinel_checkpoint": sentinel_meta,
        "calibration_selection": calibration_selection,
        "calibration_evaluation": {"metadata": calibration_meta, "bank": calibration_bank_meta, "blocks": calibration_blocks, "threshold": threshold},
        "test_selection": test_selection,
        "test_evaluation": {"metadata": test_meta, "bank": test_bank_meta, "blocks": policy_blocks, "quality": quality},
        "timing": timing,
        "gates": {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions, "risk_threshold": CATASTROPHIC_REGRET, "paired_unit": "episode; mean over 18 trajectory blocks"},
        "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Fixed-observation selective candidate-ranking evidence only; teacher correction is block-level and does not update CEM; no official CEM deployment or closed-loop claim.",
    }
    write_json(output / "selective_teacher_correction_summary.json", jsonable(summary))
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
        "historical_job": "25239551.pbs101",
        "main_arm": "treatment",
        "sentinel_arm": "control",
        "uncertainty": "pinned topk top30 disagreement u=1-intersection/30; midpoint tau; strict u>tau",
        "calibration": "valid[576:584]",
        "test": "valid[584:592]",
        "test_trajectory_blocks": 144,
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
        value = {"schema": SCHEMA, "status": "READY", "historical_job": "25239551.pbs101", "calibration": "valid[576:584]", "test": "valid[584:592]", "uncertainty": "pinned topk top30 disagreement", "threshold": "midpoint strict u>tau", "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        print(json.dumps(value, ensure_ascii=False))
        return 0
    if args.mode == "preflight":
        value = preflight(args, freeze)
    else:
        reference = cem.load_reference()
        value = run(args, freeze, reference)
    print(json.dumps(jsonable(value), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
