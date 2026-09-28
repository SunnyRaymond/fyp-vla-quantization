#!/usr/bin/env python3
"""Fixed-observation LeWM adaptive-CEM teacher-schedule comparison."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
OLD_DIR = TRANSFER / "cem-distribution-distill"
SCHEMA = "lewm-recurrent-student.adaptive-teacher-schedule-runner"
HORIZON = 5
ACTION_DIM = 10
NUM_CANDIDATES = 300
ELITE_COUNT = 30
CEM_ITERATIONS = 30
CHECKPOINTS = (10, 20, 30)
FRESH_SLICE = (592, 600)
FRESH_SEEDS = (20301105, 20301106)
INNOVATION_SEED_BASE = 20301112
PROBE_SEED_BASE = 20301120
ARM_ORDER = ("student_only", "uniform_teacher7", "late_teacher7", "teacher_only")
TEACHER_ROUNDS = {
    "student_only": frozenset(),
    "uniform_teacher7": frozenset((4, 8, 12, 16, 20, 24, 28)),
    "late_teacher7": frozenset((24, 25, 26, 27, 28, 29, 30)),
    "teacher_only": frozenset(range(1, 31)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--main-checkpoint", type=Path, required=True)
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


def load_old_runner() -> Any:
    if str(OLD_DIR) not in sys.path:
        sys.path.insert(0, str(OLD_DIR))
    return importlib.import_module("run_cem_distribution_distill")


def load_reference(old: Any) -> Any:
    return old.load_reference()


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required for model/HDF5/inference work")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and Path(nodefile).is_file():
        nodes = {line.strip().lower() for line in Path(nodefile).read_text().splitlines() if line.strip()}
        short_host = host.split(".", 1)[0]
        short_nodes = {node.split(".", 1)[0] for node in nodes}
        if nodes and host not in nodes and short_host not in short_nodes:
            raise RuntimeError(f"current host {host} is not in PBS_NODEFILE allocation")


def validate_freeze(freeze: Mapping[str, Any], checkpoint_arg: Path) -> dict[str, Any]:
    if freeze.get("schema") != "lewm-recurrent-student.adaptive-teacher-schedule-freeze":
        raise ValueError("unexpected adaptive teacher schedule freeze schema")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("freeze must be frozen_before_results")
    model = freeze.get("model", {})
    expected_checkpoint = str(model.get("main_checkpoint_path", "")).replace("\\", "/")
    actual_checkpoint = str(checkpoint_arg).replace("\\", "/")
    if actual_checkpoint != expected_checkpoint:
        raise ValueError("main checkpoint argument does not exactly match frozen path")
    for key, expected in {
        "main_checkpoint_job": "25239551.pbs101",
        "main_checkpoint_filename": "treatment_step1000.pt",
        "main_checkpoint_source": "cem_distribution_distill",
        "main_checkpoint_arm": "treatment",
        "main_checkpoint_updates": 1000,
    }.items():
        if model.get(key) != expected:
            raise ValueError(f"main checkpoint provenance drifted: {key}")
    cem = freeze.get("cem", {})
    expected_cem = {
        "iterations": CEM_ITERATIONS,
        "num_samples": NUM_CANDIDATES,
        "topk": ELITE_COUNT,
        "horizon": HORIZON,
        "packed_action_dim": ACTION_DIM,
        "initial_mu": 0.0,
        "initial_sigma": 1.0,
        "candidate_zero_is_pre_update_mu": True,
        "std_unbiased": True,
        "clip": "none",
        "innovation_seed_base": INNOVATION_SEED_BASE,
    }
    for key, expected in expected_cem.items():
        if cem.get(key) != expected:
            raise ValueError(f"CEM contract drifted: {key}")
    if cem.get("selection_operator") != "torch.topk(cost, k=30, largest=False, sorted=True)":
        raise ValueError("topk contract drifted")
    frozen_schedules = {
        key: list(value)
        for key, value in {
            "student_only": (),
            "uniform_teacher7": (4, 8, 12, 16, 20, 24, 28),
            "late_teacher7": (24, 25, 26, 27, 28, 29, 30),
            "teacher_only": tuple(range(1, 31)),
        }.items()
    }
    if cem.get("teacher_rounds") != frozen_schedules:
        raise ValueError("teacher schedule drifted")
    if cem.get("exact_teacher_schedule_calls") != {key: len(value) for key, value in frozen_schedules.items()}:
        raise ValueError("teacher call counts drifted")
    fresh = freeze.get("fresh_evaluation", {})
    for key, expected in {"selection_slice": "valid[592:600]", "excluded_valid_prefix": 592, "selection_seed": 20300903, "episodes": 8, "contexts_per_episode": 3, "trajectory_blocks": 48}.items():
        if fresh.get(key) != expected:
            raise ValueError(f"fresh contract drifted: {key}")
    if fresh.get("primary_round") != 30 or fresh.get("diagnostic_rounds") != [10, 20, 30]:
        raise ValueError("primary/diagnostic round contract drifted")
    if fresh.get("action_prefix_seeds") != list(FRESH_SEEDS):
        raise ValueError("fresh action-prefix seeds drifted")
    gates = freeze.get("gates", {}).get("schedule_quality", {})
    if float(gates.get("median_schedule_minus_student_round30_delta_max", 1.0)) != -0.1 or int(gates.get("strictly_improved_episodes_min", -1)) != 5:
        raise ValueError("quality gate drifted")
    return dict(freeze)


def validate_protocol(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    required = ("valid[592:600]", "20300903", "student_only", "uniform_teacher7", "late_teacher7", "teacher_only", "candidate zero", "closed-loop", "scientific agent skills")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol missing frozen terms: {missing}")


def validate_interface(reference: Any, probe: Path) -> dict[str, Any]:
    contract = reference.base.load_interface_contract(probe.resolve())
    expected = {
        "observation_history_h": 1,
        "official_future_prediction_count": HORIZON,
        "student_probe_future_count": HORIZON,
        "official_predicted_emb_shape": (1, 2, 6, 192),
        "candidate_shape": (1, 2, HORIZON, ACTION_DIM),
        "semantics_equal": True,
        "raw_action_dim": 2,
    }
    for key, value in expected.items():
        if getattr(contract, key) != value:
            raise ValueError(f"interface drifted: {key}")
    return {key: list(value) if isinstance(value, tuple) else value for key, value in expected.items()}


def tensor(value: Any, device: str = "cuda") -> Any:
    import torch

    return value if torch.is_tensor(value) and str(value.device) == device else torch.as_tensor(value, dtype=torch.float32, device=device)


def select_fresh(dataset: Path, manifest: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    with h5py.File(dataset.resolve(), "r") as handle:
        lengths = [int(value) for value in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON * 5 + 1]
    random.Random(20300903).shuffle(valid)
    prefix = [int(row["episode_id"]) for row in manifest["splits"]["heldout"] + manifest["splits"]["train"]]
    if valid[: len(prefix)] != prefix:
        raise ValueError("valid shuffle prefix cannot be reproduced")
    selected = valid[FRESH_SLICE[0] : FRESH_SLICE[1]]
    if len(selected) != 8 or set(selected) & set(valid[: FRESH_SLICE[0]]):
        raise ValueError("fresh valid[592:600] overlaps excluded prefix")
    return selected, {"selection_seed": 20300903, "selection_slice": "valid[592:600]", "excluded_valid_prefix": 592, "fresh_episode_ids": selected, "result_dependent_selection": False}


def make_fresh_rows(old: Any, reference: Any, official: Any, dataset: Path, manifest: Mapping[str, Any], temporal_freeze: Mapping[str, Any], fresh_ids: Sequence[int]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    old.FRESH_SEEDS = tuple(FRESH_SEEDS)
    old.FRESH_SLICE = tuple(FRESH_SLICE)
    old.NUM_CANDIDATES = NUM_CANDIDATES
    rows, _ = old.make_fresh_rows(reference, dataset, manifest, temporal_freeze, official, fresh_ids)
    if len(rows) != 24:
        raise ValueError("fresh rows must contain 24 episode-anchor contexts")
    return rows, {"selection_slice": "valid[592:600]", "selection_seed": 20300903, "action_prefix_seeds": list(FRESH_SEEDS), "episodes": 8, "contexts": 24, "trajectory_blocks": 48}


def load_main_student(reference: Any, checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    import copy
    import torch

    raw = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("provenance"), Mapping):
        raise ValueError("main checkpoint provenance missing")
    provenance = dict(raw["provenance"])
    expected = {"source": "cem_distribution_distill", "arm": "treatment", "extra_updates": 1000}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"main checkpoint provenance mismatch: {key}")
    start = provenance.get("start_checkpoint")
    if not isinstance(start, Mapping) or start.get("source") != "anchor_aligned_bank_treatment" or int(start.get("provenance", {}).get("updates", -1)) != 3000:
        raise ValueError("main checkpoint start provenance mismatch")
    state = raw.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("main checkpoint lacks state_dict")
    student = reference.base.make_student("baseline").to("cuda")
    student.load_state_dict({str(key): copy.deepcopy(value) for key, value in state.items()}, strict=True)
    student.eval()
    return student, {"path": str(checkpoint.resolve()), "provenance": provenance}


def student_costs(old: Any, reference: Any, official: Any, student: Any, row: Mapping[str, Any], actions: Any) -> Any:
    return old.student_costs(reference, official, student, row, actions)


def teacher_costs(old: Any, reference: Any, official: Any, row: Mapping[str, Any], actions: Any) -> Any:
    return old.teacher_targets_and_costs(reference, official, row, actions)[1]


def build_pair_specs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import torch

    specs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for block, seed in enumerate(FRESH_SEEDS):
            generator = torch.Generator(device="cpu").manual_seed(INNOVATION_SEED_BASE + row_index * len(FRESH_SEEDS) + block)
            innovations = torch.randn((CEM_ITERATIONS, NUM_CANDIDATES, HORIZON, ACTION_DIM), generator=generator)
            specs.append({"row_index": row_index, "row": row, "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}", "innovations": innovations})
    if len(specs) != 48:
        raise ValueError("paired trajectory count drifted")
    return specs


def validate_pair_specs(specs: Sequence[Mapping[str, Any]]) -> bool:
    import torch

    keys = [str(spec["pairing_key"]) for spec in specs]
    return bool(len(specs) == 48 and len(set(keys)) == 48 and all(torch.is_tensor(spec["innovations"]) and tuple(spec["innovations"].shape) == (CEM_ITERATIONS, NUM_CANDIDATES, HORIZON, ACTION_DIM) for spec in specs))


def run_adaptive_arm(old: Any, reference: Any, official: Any, student: Any, spec: Mapping[str, Any], arm: str, capture: bool = True) -> dict[str, Any]:
    import torch

    row = spec["row"]
    mu = torch.zeros((HORIZON, ACTION_DIM), device="cuda")
    sigma = torch.ones_like(mu)
    teacher_schedule = TEACHER_ROUNDS[arm]
    checkpoints: dict[int, dict[str, Any]] = {}
    round_records: list[dict[str, Any]] = []
    teacher_calls = 0
    round1_population_std: float | None = None
    finite = True
    for round_index, epsilon_cpu in enumerate(spec["innovations"], start=1):
        epsilon = epsilon_cpu.to("cuda")
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        actions[0] = mu
        use_teacher = round_index in teacher_schedule
        if use_teacher:
            costs = teacher_costs(old, reference, official, row, actions)
            teacher_calls += 1
        else:
            costs = student_costs(old, reference, official, student, row, actions)
        finite = finite and bool(torch.isfinite(actions).all().item() and torch.isfinite(costs).all().item())
        if not finite:
            raise FloatingPointError(f"non-finite adaptive CEM state at round {round_index}")
        if arm == "teacher_only" and round_index == 1:
            round1_population_std = float(costs.std(unbiased=False).clamp_min(1e-6).detach().cpu())
        selected = torch.topk(costs, k=ELITE_COUNT, largest=False, sorted=True).indices
        next_mu = actions.index_select(0, selected).mean(dim=0)
        next_sigma = actions.index_select(0, selected).std(dim=0, unbiased=True)
        if not bool(torch.isfinite(next_mu).all().item() and torch.isfinite(next_sigma).all().item()):
            raise FloatingPointError(f"non-finite adaptive CEM update at round {round_index}")
        if capture and round_index in CHECKPOINTS:
            checkpoints[round_index] = {"post_mu": next_mu.detach().cpu(), "post_sigma": next_sigma.detach().cpu(), "teacher_round": use_teacher}
        if capture:
            round_records.append({"round": round_index, "teacher_round": use_teacher, "post_mu": next_mu.detach().cpu(), "post_sigma": next_sigma.detach().cpu()})
        mu, sigma = next_mu, next_sigma
    expected_calls = len(teacher_schedule)
    if teacher_calls != expected_calls:
        raise RuntimeError(f"schedule calls drifted for {arm}: {teacher_calls} != {expected_calls}")
    return {"arm": arm, "teacher_calls": teacher_calls, "round1_population_std": round1_population_std, "checkpoints": checkpoints, "rounds": round_records, "final_mu": mu.detach().cpu(), "final_sigma": sigma.detach().cpu(), "finite": finite}


def score_post_mu(old: Any, reference: Any, official: Any, row: Mapping[str, Any], mu: Any) -> float:
    import torch

    return float(teacher_costs(old, reference, official, row, tensor(mu).unsqueeze(0))[0].detach().cpu())


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute percentile of empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_episode(values: Sequence[Mapping[str, Any]], key: str, round_value: int | None = None) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in values:
        if round_value is None or int(item["round"]) == round_value:
            grouped[str(item["episode_id"])].append(float(item[key]))
    return {episode: float(statistics.mean(items)) for episode, items in sorted(grouped.items(), key=lambda item: int(item[0]))}


def evaluate_quality(old: Any, reference: Any, official: Any, models: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    raw_results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_ORDER}
    pair_results: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs, start=1):
        runs = {arm: run_adaptive_arm(old, reference, official, models[arm], spec, arm, capture=True) for arm in ARM_ORDER}
        teacher_std = float(runs["teacher_only"]["round1_population_std"] or 1e-6)
        pair = {"pairing_key": spec["pairing_key"], "episode_id": spec["episode_id"], "anchor": spec["anchor"], "stratum": spec["stratum"], "action_prefix_seed": spec["action_prefix_seed"], "teacher_only_round1_population_std": teacher_std, "arms": {}}
        teacher_final_mu = runs["teacher_only"]["checkpoints"][30]["post_mu"]
        for arm in ARM_ORDER:
            arm_result = runs[arm]
            arm_blocks: list[dict[str, Any]] = []
            arm_summary: dict[str, Any] = {"teacher_calls": arm_result["teacher_calls"], "rounds": {}}
            for round_index in CHECKPOINTS:
                checkpoint = arm_result["checkpoints"][round_index]
                objective = score_post_mu(old, reference, official, spec["row"], checkpoint["post_mu"])
                teacher_only_objective = score_post_mu(old, reference, official, spec["row"], runs["teacher_only"]["checkpoints"][round_index]["post_mu"])
                gap = (objective - teacher_only_objective) / teacher_std
                drift = float(torch.linalg.vector_norm(tensor(checkpoint["post_mu"])[0] - tensor(teacher_final_mu)[0]).detach().cpu()) if round_index == 30 else None
                block = {"arm": arm, "pairing_key": spec["pairing_key"], "episode_id": spec["episode_id"], "anchor": spec["anchor"], "stratum": spec["stratum"], "action_prefix_seed": spec["action_prefix_seed"], "round": round_index, "teacher_round": bool(checkpoint["teacher_round"]), "teacher_round1_population_std": teacher_std, "post_mu": checkpoint["post_mu"].tolist(), "post_sigma": checkpoint["post_sigma"].tolist(), "teacher_objective_post_mu": objective, "teacher_only_objective_post_mu": teacher_only_objective, "standardized_gap_vs_teacher_only": gap, "final_first_action_l2_drift_vs_teacher_only": drift, "finite": bool(torch.isfinite(checkpoint["post_mu"]).all().item() and torch.isfinite(checkpoint["post_sigma"]).all().item() and all(map(lambda x: x == x and abs(x) != float("inf"), (objective, teacher_only_objective, gap))))}
                raw_results[arm].append(block)
                arm_blocks.append(block)
                arm_summary["rounds"][str(round_index)] = {"teacher_objective_post_mu": objective, "teacher_only_objective_post_mu": teacher_only_objective, "standardized_gap_vs_teacher_only": gap}
            pair["arms"][arm] = arm_summary
        pair_results.append(pair)
        if spec_index % 8 == 0:
            print(json.dumps({"phase": "adaptive_quality", "paired_trajectories_completed": spec_index, "paired_trajectories_total": len(specs)}), flush=True)
    episode_gap = {arm: aggregate_episode(blocks, "standardized_gap_vs_teacher_only", 30) for arm, blocks in raw_results.items()}
    episode_delta_vs_student = {arm: {episode: float(values - episode_gap["student_only"][episode]) for episode, values in episode_gap[arm].items()} for arm in ("uniform_teacher7", "late_teacher7")}
    uniform_late = {episode: float(episode_gap["uniform_teacher7"][episode] - episode_gap["late_teacher7"][episode]) for episode in episode_gap["student_only"]}
    arm_summaries: dict[str, Any] = {}
    for arm in ARM_ORDER:
        final_values = list(episode_gap[arm].values())
        diagnostic = {str(round_index): aggregate_episode(raw_results[arm], "standardized_gap_vs_teacher_only", round_index) for round_index in CHECKPOINTS}
        arm_summaries[arm] = {"blocks": len(raw_results[arm]), "expected_schedule_rounds": sorted(TEACHER_ROUNDS[arm]), "final_episode_gap_vs_teacher_only": episode_gap[arm], "final_gap_median": float(statistics.median(final_values)), "diagnostic_episode_gap_by_round": diagnostic, "final_first_action_l2_drift_median": float(statistics.median([float(item["final_first_action_l2_drift_vs_teacher_only"]) for item in raw_results[arm] if item["round"] == 30])), "finite": all(bool(item["finite"]) for item in raw_results[arm]), "per_block": raw_results[arm]}
    quality = {"arms": arm_summaries, "schedule_delta_vs_student": episode_delta_vs_student, "schedule_delta_median_vs_student": {arm: float(statistics.median(list(values.values()))) for arm, values in episode_delta_vs_student.items()}, "schedule_strictly_improved_episodes": {arm: int(sum(value < 0.0 for value in values.values())) for arm, values in episode_delta_vs_student.items()}, "uniform_minus_late_final_episode_delta": uniform_late, "uniform_minus_late_final_median_delta": float(statistics.median(list(uniform_late.values()))), "paired_unit": "episode; round30 mean over 3 anchors x 2 action-prefix seeds", "round10_round20": "diagnostic only; excluded from primary gate"}
    observed_calls = {arm: [int(pair["arms"][arm]["teacher_calls"]) for pair in pair_results] for arm in ARM_ORDER}
    expected_calls = {arm: len(TEACHER_ROUNDS[arm]) for arm in ARM_ORDER}
    quality["schedule_calls_expected"] = expected_calls
    quality["schedule_calls_observed_per_trajectory"] = observed_calls
    quality["schedule_calls_exact"] = {arm: all(value == expected_calls[arm] for value in values) for arm, values in observed_calls.items()}
    return {"quality": quality, "pair_results": pair_results}


def distribution_probe(old: Any, reference: Any, official: Any, models: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], quality_pair_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_ORDER}
    for spec_index, (spec, pair_result) in enumerate(zip(specs, quality_pair_results)):
        generator = torch.Generator(device="cpu").manual_seed(PROBE_SEED_BASE + spec_index)
        epsilon = torch.randn((NUM_CANDIDATES, HORIZON, ACTION_DIM), generator=generator)
        arm_actions: dict[str, Any] = {}
        for arm in ARM_ORDER:
            block = next(item for item in pair_result["arms"][arm]["blocks"] if int(item["round"]) == 30)
            mu = tensor(block["post_mu"])
            sigma = tensor(block["post_sigma"])
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon.to("cuda")
            actions[0] = mu
            costs = teacher_costs(old, reference, official, spec["row"], actions)
            order = torch.topk(costs, k=ELITE_COUNT, largest=False, sorted=True).indices
            arm_actions[arm] = {"mean_top30": float(costs.index_select(0, order).mean().detach().cpu()), "finite": bool(torch.isfinite(costs).all().item())}
        baseline = arm_actions["teacher_only"]["mean_top30"]
        std = float(pair_result.get("teacher_only_round1_population_std", 1e-6))
        for arm in ARM_ORDER:
            record = {"arm": arm, "pairing_key": spec["pairing_key"], "episode_id": spec["episode_id"], "stratum": spec["stratum"], "action_prefix_seed": spec["action_prefix_seed"], "teacher_top30_mean_cost": arm_actions[arm]["mean_top30"], "teacher_only_top30_mean_cost": baseline, "standardized_gap_vs_teacher_only": (arm_actions[arm]["mean_top30"] - baseline) / max(std, 1e-6), "finite": bool(arm_actions[arm]["finite"])}
            records[arm].append(record)
    result: dict[str, Any] = {}
    for arm, blocks in records.items():
        episode = aggregate_episode(blocks, "standardized_gap_vs_teacher_only")
        result[arm] = {"blocks": blocks, "episode_mean_gap": episode, "median_episode_gap": float(statistics.median(list(episode.values()))), "finite": all(bool(item["finite"]) for item in blocks)}
    return {"seed_base": PROBE_SEED_BASE, "independent": True, "arms": result}


def time_trajectories(old: Any, reference: Any, official: Any, models: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], warmups: int = 3, repeats: int = 5) -> dict[str, Any]:
    import torch

    samples: dict[str, list[float]] = {arm: [] for arm in ARM_ORDER}
    for repeat_index in range(warmups + repeats):
        for arm in ARM_ORDER:
            arm_samples: list[float] = []
            for spec in specs:
                torch.cuda.synchronize()
                start = time.perf_counter()
                run_adaptive_arm(old, reference, official, models[arm], spec, arm, capture=False)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000.0
                arm_samples.append(elapsed)
            if repeat_index >= warmups:
                samples[arm].extend(arm_samples)
        print(json.dumps({"phase": "adaptive_timing", "repeat_completed": repeat_index + 1, "warmups": warmups, "repeats": repeats}), flush=True)
    result: dict[str, Any] = {}
    for arm, values in samples.items():
        result[arm] = {"samples": len(values), "mean_ms": float(statistics.mean(values)), "p95_ms": float(percentile(values, 0.95)), "finite": all(value == value and abs(value) != float("inf") for value in values)}
    teacher_mean = result["teacher_only"]["mean_ms"]
    for arm in ("uniform_teacher7", "late_teacher7"):
        result[arm]["reduction_vs_teacher_only"] = float(1.0 - result[arm]["mean_ms"] / teacher_mean)
    result["student_only"]["reduction_vs_teacher_only"] = float(1.0 - result["student_only"]["mean_ms"] / teacher_mean)
    return {"warmups": warmups, "repeats": repeats, "scope": "48 trajectories per arm; complete 30-round adaptive trajectory", "arms": result}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], old: Any, reference: Any) -> dict[str, Any]:
    require_compute_node()
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest_512.resolve())
    temporal_freeze = load_json(args.temporal_freeze.resolve())
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.requires_grad_(False)
    main_student, checkpoint_meta = load_main_student(reference, args.main_checkpoint)
    fresh_ids, selection = select_fresh(args.dataset, manifest)
    fresh_rows, fresh_meta = make_fresh_rows(old, reference, official, args.dataset, manifest, temporal_freeze, fresh_ids)
    specs = build_pair_specs(fresh_rows)
    shared_pairing_check = validate_pair_specs(specs)
    if not shared_pairing_check:
        raise RuntimeError("shared paired innovations/provenance check failed")
    models = {arm: main_student for arm in ARM_ORDER}
    quality_result = evaluate_quality(old, reference, official, models, specs)
    # Build a direct pair/arm block index for the independent distribution probe.
    block_index: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for arm in ARM_ORDER:
        for block in quality_result["quality"]["arms"][arm]["per_block"]:
            block_index[(str(block["pairing_key"]), arm)].append(block)
    quality_pairs_for_probe: list[dict[str, Any]] = []
    for spec in specs:
        pair = {"pairing_key": spec["pairing_key"], "episode_id": spec["episode_id"], "stratum": spec["stratum"], "action_prefix_seed": spec["action_prefix_seed"], "teacher_only_round1_population_std": next(block["teacher_round1_population_std"] for block in quality_result["quality"]["arms"]["teacher_only"]["per_block"] if block["pairing_key"] == spec["pairing_key"]), "arms": {}}
        for arm in ARM_ORDER:
            pair["arms"][arm] = {"blocks": block_index[(str(spec["pairing_key"]), arm)]}
        quality_pairs_for_probe.append(pair)
    probe = distribution_probe(old, reference, official, models, specs, quality_pairs_for_probe)
    timing = time_trajectories(old, reference, official, models, specs)
    quality = quality_result["quality"]
    eligible: dict[str, bool] = {}
    for arm in ("uniform_teacher7", "late_teacher7"):
        eligible[arm] = bool(quality["schedule_delta_median_vs_student"][arm] <= -0.1 and quality["schedule_strictly_improved_episodes"][arm] >= 5 and timing["arms"][arm]["reduction_vs_teacher_only"] >= 0.3)
    validity_pass = bool(all(quality["schedule_calls_exact"].values()) and all(quality["arms"][arm]["finite"] for arm in ARM_ORDER) and all(probe["arms"][arm]["finite"] for arm in ARM_ORDER) and all(timing["arms"][arm]["finite"] for arm in ARM_ORDER) and shared_pairing_check)
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_contract": validate_interface(reference, args.interface_probe.resolve()),
        "main_checkpoint": checkpoint_meta,
        "fresh_selection": selection,
        "fresh_evaluation": fresh_meta,
        "pairing": {"trajectory_blocks": len(specs), "shared_innovation_seed_base": INNOVATION_SEED_BASE, "shared_innovations_across_arms": shared_pairing_check, "all_pairs_have_same_innovations": shared_pairing_check},
        "quality": quality,
        "secondary_distribution_probe": probe,
        "timing": timing,
        "gates": {"schedule_quality_and_latency": eligible, "schedule_calls_exact": all(quality["schedule_calls_exact"].values()), "schedule_calls_expected": quality["schedule_calls_expected"], "schedule_calls_observed_per_trajectory": quality["schedule_calls_observed_per_trajectory"], "finite": all(quality["arms"][arm]["finite"] for arm in ARM_ORDER) and all(probe["arms"][arm]["finite"] for arm in ARM_ORDER) and all(timing["arms"][arm]["finite"] for arm in ARM_ORDER), "shared_provenance": shared_pairing_check, "validity": validity_pass, "overall": "PASS" if validity_pass and any(eligible.values()) else "FAIL", "claim_boundary": "Fixed-observation fully adaptive CEM mechanism evidence only; official CEM deployment and closed-loop are NOT_RUN_BY_SCOPE."},
        "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
    }
    write_json(output / "adaptive_teacher_schedule_summary.json", summary)
    return summary


def preflight(args: argparse.Namespace, freeze: Mapping[str, Any], old: Any) -> dict[str, Any]:
    reference = load_reference(old)
    interface = validate_interface(reference, args.interface_probe.resolve())
    value = {"schema": SCHEMA, "status": "PASS", "model_work_started": False, "reference_import": "PASS", "interface": interface, "main_checkpoint_binding": str(args.main_checkpoint).replace("\\", "/"), "fresh_selection": "valid[592:600]", "selection_seed": 20300903, "trajectory_blocks": 48, "arms": list(ARM_ORDER), "teacher_rounds": {arm: sorted(TEACHER_ROUNDS[arm]) for arm in ARM_ORDER}, "timing": {"warmups": 3, "repeats": 5, "cuda_synchronized": True}, "pbs_compute_only": True, "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
    write_json(args.output.resolve() / "preflight_status.json", value)
    return value


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "fresh_selection": "valid[592:600]", "trajectory_blocks": 48, "arms": list(ARM_ORDER), "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output.resolve() / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    freeze = validate_freeze(load_json(args.freeze.resolve()), args.main_checkpoint)
    validate_protocol(args.protocol.resolve())
    old = load_old_runner()
    if args.mode == "preflight":
        print(json.dumps(preflight(args, freeze, old), ensure_ascii=False))
        return 0
    reference = load_reference(old)
    summary = run(args, freeze, old, reference)
    print(json.dumps({"status": summary["status"], "gates": summary["gates"], "output": str((args.output / "adaptive_teacher_schedule_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
