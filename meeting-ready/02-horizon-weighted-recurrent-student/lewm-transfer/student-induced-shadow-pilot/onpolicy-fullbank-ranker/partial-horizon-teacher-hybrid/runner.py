#!/usr/bin/env python3
"""Score fixed student-generated train banks with a frozen partial-horizon hybrid."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TRANSFER_DIR = HERE.parents[2]
ARMS = (0, 1, 2, 5)
ROUNDS = (10, 20, 30)
ELITES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    parser.add_argument("--train-banks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> str:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("runner requires PBS_JOBID and a real PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes:
        raise RuntimeError(f"host {host} is not present in PBS_NODEFILE")
    return host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_freeze(freeze: dict[str, Any], train_banks: Path) -> None:
    if freeze.get("schema") != "lewm-pusht-partial-horizon-teacher-hybrid-freeze":
        raise ValueError("freeze schema mismatch")
    if freeze.get("status") != "frozen_before_execution":
        raise ValueError("freeze is not frozen_before_execution")
    source = freeze["source"]
    if str(train_banks.resolve()) != str(Path(source["train_banks_path"]).resolve()):
        raise ValueError("train bank path differs from the frozen 25538135 train-bank path")
    if source["collection_job"] != "25538135.pbs101" or int(source["train_bank_count"]) != 93:
        raise ValueError("train-bank provenance/count mismatch")
    if freeze["arms"]["k_values"] != list(ARMS):
        raise ValueError("k arms drifted")
    if freeze["validity"]["required_rounds"] != list(ROUNDS):
        raise ValueError("required CEM rounds drifted")


def validate_banks(banks: Any, freeze: dict[str, Any], torch: Any) -> None:
    source = freeze["source"]
    expected_episodes = {int(value) for value in source["train_episode_ids"]}
    if not isinstance(banks, list) or len(banks) != int(source["train_bank_count"]):
        raise ValueError("expected exactly the frozen 93 train banks")
    seen: set[tuple[int, int, int]] = set()
    counts: dict[int, int] = defaultdict(int)
    for bank in banks:
        if not isinstance(bank, dict) or bank.get("schema") != source["bank_schema"]:
            raise ValueError("bank schema mismatch")
        if bank.get("split") != "train":
            raise ValueError("non-train bank encountered; fail closed")
        episode = int(bank["episode_idx"])
        step = int(bank["replan_step"])
        round_number = int(bank["cem_round"])
        if episode not in expected_episodes or step not in (0, 25) or round_number not in ROUNDS:
            raise ValueError(f"bank outside frozen episode/replan/round set: {(episode, step, round_number)}")
        key = (episode, step, round_number)
        if key in seen:
            raise ValueError(f"duplicate train bank {key}")
        seen.add(key)
        counts[episode] += 1
        if tuple(bank["candidates"].shape) != (300, 5, 10):
            raise ValueError(f"candidate shape mismatch for {key}")
        if tuple(bank["student_costs"].shape) != (300,) or tuple(bank["teacher_costs"].shape) != (300,):
            raise ValueError(f"cost shape mismatch for {key}")
        if not torch.equal(bank["candidate_indices"], torch.arange(300, dtype=torch.int64)):
            raise ValueError(f"candidate ordering mismatch for {key}")
        for field in ("candidates", "initial_emb", "goal_emb", "student_costs", "teacher_costs"):
            tensor = bank[field]
            if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"non-finite or non-tensor {field} in {key}")
        if bank["candidates"].dtype != torch.float32 or bank["student_costs"].dtype != torch.float32 or bank["teacher_costs"].dtype != torch.float32:
            raise ValueError(f"bank action/cost dtype mismatch for {key}")
        teacher_std = bank["teacher_costs"].std(unbiased=False)
        if not bool(torch.isfinite(teacher_std).item()) or float(teacher_std.item()) <= 0.0:
            raise ValueError(f"invalid teacher-cost population standard deviation in {key}")

    if set(counts) != expected_episodes:
        raise ValueError("train banks do not cover exactly the frozen 16 episodes")
    if counts[15946] != 3 or any(counts[episode] != 6 for episode in expected_episodes - {15946}):
        raise ValueError("train-bank episode coverage differs from the frozen t0/t25 protocol")
    for round_number in ROUNDS:
        if any((episode, 0, round_number) not in seen for episode in expected_episodes):
            raise ValueError(f"missing t0 bank at round {round_number}")
        if any((episode, 25, round_number) not in seen for episode in expected_episodes - {15946}):
            raise ValueError(f"missing t25 bank at round {round_number}")
    if any((15946, 25, round_number) in seen for round_number in ROUNDS):
        raise ValueError("episode 15946 must remain t25-unavailable")


def load_models(freeze: dict[str, Any], torch: Any) -> tuple[Any, Any, Any]:
    if str(TRANSFER_DIR) not in sys.path:
        sys.path.insert(0, str(TRANSFER_DIR))
    base = importlib.import_module("run_lewm_recurrent_student")

    student_path = Path(freeze["models"]["student_checkpoint_path"])
    raw_student = torch.load(student_path, map_location="cpu", weights_only=False)
    provenance = raw_student.get("provenance", {})
    for key, expected in {"source": "cem_distribution_distill", "arm": "treatment", "extra_updates": 1000}.items():
        if provenance.get(key) != expected:
            raise ValueError(f"student checkpoint provenance mismatch: {key}")
    start = provenance.get("start_checkpoint", {})
    if start.get("source") != "anchor_aligned_bank_treatment" or int(start.get("provenance", {}).get("updates", -1)) != 3000:
        raise ValueError("student start-checkpoint provenance mismatch")
    student = base.make_student("baseline").to("cuda")
    student.load_state_dict(raw_student["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in student.parameters()) != 775872:
        raise ValueError("h256 student parameter count mismatch")
    student.eval()

    teacher = torch.load(Path(freeze["models"]["teacher_checkpoint_path"]), map_location="cpu", weights_only=False)
    if not hasattr(teacher, "predict") or not hasattr(teacher, "criterion") or not hasattr(teacher, "action_encoder"):
        raise TypeError("checkpoint is not an official LeWM predictor and criterion object")
    teacher = teacher.to("cuda").eval()
    return base, student, teacher


def predict_and_score(base: Any, student: Any, teacher: Any, bank: dict[str, Any], k: int, torch: Any) -> Any:
    actions = bank["candidates"].to(device="cuda", dtype=torch.float32)
    initial = bank["initial_emb"].to(device="cuda", dtype=torch.float32).reshape(1, 1, 192)
    history = initial.expand(actions.shape[0], -1, -1)
    goal = bank["goal_emb"].to(device="cuda", dtype=torch.float32).reshape(1, 192)

    with torch.inference_mode():
        prefix = base.official_teacher_targets(teacher, history, actions[:, :k]) if k else None
        if k == 5:
            prediction = prefix
        else:
            student_start = history if k == 0 else prefix[:, -1:, :]
            continuation = student(student_start, actions[:, k:])
            prediction = continuation if k == 0 else torch.cat((prefix, continuation), dim=1)
        costs = base._official_objective(
            teacher,
            {"latent_history": initial, "goal_emb": goal},
            prediction,
        )
    if tuple(prediction.shape) != (300, 5, 192) or tuple(costs.shape) != (300,):
        raise RuntimeError(f"prediction/criterion shape mismatch for k={k}")
    return costs


def compare_costs(label: str, actual: Any, expected: Any, freeze: dict[str, Any], torch: Any, key: tuple[int, int, int]) -> dict[str, Any]:
    cfg = freeze["validity"]
    expected = expected.to(device=actual.device, dtype=actual.dtype)
    delta = (actual - expected).abs()
    max_abs = float(delta.max().item())
    if not torch.allclose(actual, expected, rtol=float(cfg["rtol"]), atol=float(cfg["atol"])):
        raise RuntimeError(f"{label} saved-cost control mismatch for bank {key}; max_abs={max_abs:.9g}")
    return {"arm": label, "max_abs_error": max_abs, "allclose": True}


def bank_metrics(costs: Any, teacher_costs: Any, torch: Any) -> tuple[float, float]:
    teacher_top = torch.topk(teacher_costs, ELITES, largest=False, sorted=True).indices
    selected = torch.topk(costs, ELITES, largest=False, sorted=True).indices
    top_mean = teacher_costs.index_select(0, teacher_top).mean()
    selected_mean = teacher_costs.index_select(0, selected).mean()
    regret = (selected_mean - top_mean) / teacher_costs.std(unbiased=False)
    recall = torch.isin(teacher_top, selected).to(torch.float32).mean()
    return float(regret.item()), float(recall.item())


def timed_score(base: Any, student: Any, teacher: Any, bank: dict[str, Any], k: int, torch: Any) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    predict_and_score(base, student, teacher, bank, k, torch)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def summarize(rows: list[dict[str, Any]], freeze: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["episode_idx"]), int(row["k"]))].append(row)

    episode_metrics: dict[str, dict[str, Any]] = {}
    for k in (1, 2):
        deltas: dict[int, list[float]] = {}
        latency_reductions: dict[int, list[float]] = {}
        for episode in freeze["source"]["train_episode_ids"]:
            bank_rows = grouped[(int(episode), k)]
            deltas[int(episode)] = [row["regret"] - row["k0_regret"] for row in bank_rows]
            latency_reductions[int(episode)] = [
                1.0 - row["latency_ms"] / row["k5_latency_ms"] for row in bank_rows
            ]
        per_episode = {str(episode): median(values) for episode, values in deltas.items()}
        per_episode_latency = {str(episode): median(values) for episode, values in latency_reductions.items()}
        episode_median_delta = median(list(per_episode.values()))
        latency_median_reduction = median(list(per_episode_latency.values()))
        improved = sum(value < 0.0 for value in per_episode.values())

        per_round: dict[str, float] = {}
        for round_number in (20, 30):
            episode_round_deltas = []
            for episode in freeze["source"]["train_episode_ids"]:
                matched = [
                    row["regret"] - row["k0_regret"]
                    for row in grouped[(int(episode), k)]
                    if int(row["cem_round"]) == round_number
                ]
                episode_round_deltas.append(median(matched))
            per_round[str(round_number)] = median(episode_round_deltas)

        conditions = {
            "minimum_episodes_improved": improved >= int(freeze["exploration_gate"]["episodes_with_regret_improvement_min"]),
            "episode_median_regret_delta": episode_median_delta <= float(freeze["exploration_gate"]["episode_median_regret_delta_max"]),
            "episode_median_latency_reduction": latency_median_reduction >= float(freeze["exploration_gate"]["episode_median_latency_reduction_vs_k5_min"]),
            "round20_nonworsening": per_round["20"] <= 0.0,
            "round30_nonworsening": per_round["30"] <= 0.0,
        }
        episode_metrics[str(k)] = {
            "regret_delta_by_episode": per_episode,
            "latency_reduction_vs_k5_by_episode": per_episode_latency,
            "episodes_improved": improved,
            "median_episode_regret_delta": episode_median_delta,
            "median_episode_latency_reduction_vs_k5": latency_median_reduction,
            "round20_median_episode_delta": per_round["20"],
            "round30_median_episode_delta": per_round["30"],
            "gate": {"status": "GO" if all(conditions.values()) else "NO-GO", "conditions": conditions},
        }

    strata: dict[str, Any] = {}
    for stratum, step in (("t0", 0), ("t25", 25)):
        strata[stratum] = {}
        for round_number in ROUNDS:
            strata[stratum][str(round_number)] = {}
            for k in ARMS:
                selected_rows = [row for row in rows if int(row["replan_step"]) == step and int(row["cem_round"]) == round_number]
                arm_rows = [row for row in selected_rows if int(row["k"]) == k]
                regret_by_episode: dict[int, list[float]] = defaultdict(list)
                delta_by_episode: dict[int, list[float]] = defaultdict(list)
                recall_by_episode: dict[int, list[float]] = defaultdict(list)
                latency_by_episode: dict[int, list[float]] = defaultdict(list)
                for row in arm_rows:
                    regret_by_episode[int(row["episode_idx"])].append(float(row["regret"]))
                    delta_by_episode[int(row["episode_idx"])].append(float(row["regret"] - row["k0_regret"]))
                    recall_by_episode[int(row["episode_idx"])].append(float(row["top30_recall"]))
                    latency_by_episode[int(row["episode_idx"])].append(float(row["latency_ms"]))
                strata[stratum][str(round_number)][str(k)] = {
                    "episode_count": len(regret_by_episode),
                    "median_episode_regret": median([median(v) for v in regret_by_episode.values()]),
                    "median_episode_regret_delta_vs_k0": median([median(v) for v in delta_by_episode.values()]),
                    "median_episode_top30_recall": median([median(v) for v in recall_by_episode.values()]),
                    "median_episode_latency_ms": median([median(v) for v in latency_by_episode.values()]),
                }

    decision = "EXPLORATORY_GO_TO_NEW_INDEPENDENT_BANK_GATE" if any(item["gate"]["status"] == "GO" for item in episode_metrics.values()) else "NO-GO"
    return {"by_k": episode_metrics, "by_stratum_and_round": strata, "decision": decision}


def run(args: argparse.Namespace) -> int:
    host = require_compute_node()
    freeze = read_json(args.freeze)
    validate_freeze(freeze, args.train_banks)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated PBS job")
    banks_path = args.train_banks.resolve(strict=True)
    banks = torch.load(banks_path, map_location="cpu", weights_only=False)
    validate_banks(banks, freeze, torch)
    base, student, teacher = load_models(freeze, torch)

    control_matches: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    timing = freeze["timing"]
    for bank_index, bank in enumerate(banks, start=1):
        key = (int(bank["episode_idx"]), int(bank["replan_step"]), int(bank["cem_round"]))
        device_bank = dict(bank)
        for field in ("candidates", "initial_emb", "goal_emb"):
            device_bank[field] = bank[field].to(device="cuda", dtype=torch.float32)
        teacher_costs = bank["teacher_costs"].to(device="cuda", dtype=torch.float32)
        computed: dict[int, Any] = {}
        metric_rows: dict[int, tuple[float, float]] = {}
        for k in ARMS:
            computed[k] = predict_and_score(base, student, teacher, device_bank, k, torch)
            if k == 0:
                control_matches.append(compare_costs("k0", computed[k], bank["student_costs"], freeze, torch, key))
            elif k == 5:
                control_matches.append(compare_costs("k5", computed[k], bank["teacher_costs"], freeze, torch, key))
            metric_rows[k] = bank_metrics(computed[k], teacher_costs, torch)

        samples: dict[int, list[float]] = {k: [] for k in ARMS}
        for k in ARMS:
            for _ in range(int(timing["warmups"])):
                timed_score(base, student, teacher, device_bank, k, torch)
        for repeat in range(int(timing["repeats"])):
            order = list(ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)])
            for k in order:
                samples[k].append(timed_score(base, student, teacher, device_bank, k, torch))

        common = {
            "episode_idx": key[0],
            "replan_step": key[1],
            "stratum": "t0" if key[1] == 0 else "t25",
            "cem_round": key[2],
        }
        by_arm = {
            str(k): {
                "regret": metric_rows[k][0],
                "top30_recall": metric_rows[k][1],
                "latency_ms": median(samples[k]),
            }
            for k in ARMS
        }
        for k in ARMS:
            row = {**common, "k": k, **by_arm[str(k)]}
            row["k0_regret"] = by_arm["0"]["regret"]
            row["k5_latency_ms"] = by_arm["5"]["latency_ms"]
            bank_rows.append(row)
        print(f"scored_train_bank={bank_index}/{len(banks)} episode={key[0]} step={key[1]} round={key[2]}", flush=True)

    if len(control_matches) != 2 * len(banks):
        raise RuntimeError("k0/k5 saved-cost alignment did not cover every train bank")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "lewm-pusht-partial-horizon-teacher-hybrid-result",
        "status": "COMPLETE",
        "pbs_job_id": os.environ["PBS_JOBID"].strip(),
        "compute_host": host,
        "freeze": str(args.freeze.resolve()),
        "source": {"collection_job": freeze["source"]["collection_job"], "train_bank_count": len(banks), "episode_count": 16, "split_used": "train_only"},
        "controls": {
            "saved_cost_matches": len(control_matches),
            "all_k0_student_and_k5_teacher_costs_matched": True,
            "rtol": freeze["validity"]["rtol"],
            "atol": freeze["validity"]["atol"],
            "maximum_absolute_error": max(item["max_abs_error"] for item in control_matches),
        },
        "timing": {"warmups_per_arm_per_bank": int(timing["warmups"]), "repeats_per_arm_per_bank": int(timing["repeats"]), "cuda_synchronized": True},
        "metrics": summarize(bank_rows, freeze),
        "per_bank_arms": bank_rows,
        "scope": {"new_training": False, "validation_or_holdout_banks_read_or_scored": False, "CEM": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE", "fresh_episode_and_independent_bank_gate": "REQUIRED_BEFORE_ANY_FURTHER_ESCALATION"},
    }
    destination = output / "partial_horizon_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"decision={summary['metrics']['decision']} summary={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
