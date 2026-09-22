#!/usr/bin/env python3
"""Fixed-observation CEM with periodic full-pool teacher anchors.

The positive teacher-full trajectory is recomputed once per case with the
official teacher and the same common CPU innovations.  Its reference budget is
reported separately from the intervention arms; all new arm calls are either
the P5 mechanism anchor or a checkpoint oracle for that arm's current pool.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


CASES = [0, 1, 2, 4, 5, 7]
SEEDS = [1, 100, 199, 397, 496, 694]
CHECKPOINTS = [1, 5, 10, 15, 20, 25, 30]
ANCHORS = [5, 10, 15, 20, 25]
ITERATIONS = 30
SAMPLES = 300
TOPK = 30
HORIZON = 5
ACTION_DIM = 10
INNOVATION_SEED_BASE = 20261100
ARMS = ("student_only", "periodic_teacher_anchor_P5", "shuffled_anchor_P5")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "summarize"), default="run")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path)
    parser.add_argument("--plan-targets", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--data-root", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_freeze(freeze: Mapping[str, Any], protocol: str) -> dict[str, Any]:
    if freeze.get("schema") != "horizon-weighted-recurrent-student.cem-periodic-anchor-freeze":
        raise ValueError("unexpected periodic-anchor freeze schema")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unsupported periodic-anchor freeze schema version")
    cases = freeze.get("cases", {})
    cem = freeze.get("cem", {})
    if [int(v) for v in cases.get("indices", [])] != CASES or [int(v) for v in cases.get("eval_seeds", [])] != SEEDS:
        raise ValueError("frozen cases or eval seeds drifted")
    exact = {
        "iterations": ITERATIONS,
        "num_samples": SAMPLES,
        "topk": TOPK,
        "horizon": HORIZON,
        "packed_action_dim": ACTION_DIM,
        "innovation_seed_base": INNOVATION_SEED_BASE,
    }
    for key, expected in exact.items():
        if int(cem.get(key, -1)) != expected:
            raise ValueError(f"cem.{key} drifted")
    if [int(v) for v in cem.get("checkpoints", [])] != CHECKPOINTS:
        raise ValueError("diagnostic checkpoints drifted")
    if [int(v) for v in cem.get("anchor_rounds_1_based", [])] != ANCHORS or bool(cem.get("anchor_round_30")):
        raise ValueError("anchor rounds drifted or round 30 was enabled")
    if cem.get("candidate_zero_is_mu") is not True or cem.get("std_unbiased") is not True:
        raise ValueError("official CEM candidate-zero/std semantics drifted")
    shuffle = freeze.get("shuffle", {})
    if int(shuffle.get("permutation_seed_base", -1)) != 2026110000 or shuffle.get("device") != "cpu":
        raise ValueError("shuffle permutation contract drifted")
    required_phrases = ("fixed observation", "anchor", "historical", "closed-loop")
    if any(phrase not in protocol.lower() for phrase in required_phrases):
        raise ValueError("protocol text does not state the frozen scope")
    return dict(freeze)


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


def _assert_finite(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite {label}")


def _timed_cost(torch: Any, cost_fn: Any, model: Any, context: Mapping[str, Any], actions: Any, goal: Mapping[str, Any], objective: Any) -> tuple[Any, float]:
    import time

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        cost = cost_fn(model, context, actions, goal, objective)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return cost, (time.perf_counter() - started) * 1000.0


def _update_distribution(torch: Any, actions: Any, indices: Any) -> tuple[Any, Any]:
    elites = actions.index_select(0, indices)
    return elites.mean(dim=0), elites.std(dim=0)


def _rms(value: Any) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _coordinate_abs_max(value: Any) -> float:
    return float(value.detach().float().abs().max().cpu())


def _mean_cost(cost: Any, indices: Any) -> float:
    return float(cost.index_select(0, indices).mean().detach().cpu())


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "minimum": float("nan"), "maximum": float("nan")}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _run_teacher_reference(
    torch: Any,
    trace: Any,
    teacher: Any,
    context: Mapping[str, Any],
    goal: Mapping[str, Any],
    innovations: Any,
    objective: Any,
    device: Any,
) -> dict[str, Any]:
    """Run the official teacher-full reference exactly once for this case."""

    mu = torch.zeros((HORIZON, ACTION_DIM), device=device)
    sigma = torch.ones_like(mu)
    updates: dict[int, dict[str, Any]] = {}
    reference_rounds: dict[str, Any] = {}
    total_ms = 0.0
    for round_index, epsilon_cpu in enumerate(innovations, start=1):
        epsilon = epsilon_cpu.to(device)
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        actions[0] = mu
        _assert_finite(torch, actions, f"teacher reference actions round {round_index}")
        teacher_cost, elapsed = _timed_cost(torch, trace._teacher_cost, teacher, context, actions, goal, objective)
        _assert_finite(torch, teacher_cost, f"teacher reference cost round {round_index}")
        order = teacher_cost.argsort()
        mu_next, sigma_next = _update_distribution(torch, actions, order[:TOPK])
        _assert_finite(torch, mu_next, f"teacher reference mu round {round_index}")
        _assert_finite(torch, sigma_next, f"teacher reference sigma round {round_index}")
        updates[round_index] = {"mu": mu_next.detach(), "sigma": sigma_next.detach()}
        reference_rounds[str(round_index)] = {
            "round": round_index,
            "mu_rms": 0.0,
            "sigma_rms": 0.0,
            "first_action_rms": 0.0,
            "first_action_coordinate_abs_max": 0.0,
            "teacher_top30_overlap": 1.0,
            "teacher_cost_regret": 0.0,
            "pre_update": {"mu": mu.detach().cpu().tolist(), "sigma": sigma.detach().cpu().tolist()},
            "post_update": {"mu": mu_next.detach().cpu().tolist(), "sigma": sigma_next.detach().cpu().tolist()},
            "selected_indices": order[:TOPK].detach().cpu().tolist(),
            "teacher_top30_indices": order[:TOPK].detach().cpu().tolist(),
            "counts": {"reference_teacher_calls": 1, "reference_teacher_candidates": SAMPLES},
            "timing_ms": {"reference_teacher_native": elapsed},
        }
        total_ms += elapsed
        mu, sigma = mu_next, sigma_next
    return {
        "updates": updates,
        "rounds": reference_rounds,
        "total_counts": {"reference_teacher_calls": ITERATIONS, "reference_teacher_candidates": ITERATIONS * SAMPLES},
        "total_timing_ms": total_ms,
    }


def _reference_arm(torch: Any, reference: Mapping[str, Any]) -> dict[str, Any]:
    rounds: dict[str, Any] = {}
    for round_index in range(1, ITERATIONS + 1):
        update = reference["updates"][round_index]
        mu = update["mu"]
        sigma = update["sigma"]
        rounds[str(round_index)] = {
            "round": round_index,
            "mu_rms": 0.0,
            "sigma_rms": 0.0,
            "first_action_rms": 0.0,
            "first_action_coordinate_abs_max": 0.0,
            "teacher_top30_overlap": 1.0,
            "teacher_cost_regret": 0.0,
            "post_update": {"mu": mu.detach().cpu().tolist(), "sigma": sigma.detach().cpu().tolist()},
            "counts": {"student_candidates": 0, "mechanism_teacher_candidates": 0, "diagnostic_oracle_candidates": 0},
            "timing_ms": {"arm_scoring_total": 0.0, "student_native": 0.0, "teacher_native": 0.0, "teacher_oracle_reference": 0.0},
        }
    return {
        "mode": "teacher_full_reference_recomputed_once",
        "rounds": rounds,
        "trajectory_auc": 0.0,
        "total_counts": dict(reference["total_counts"]),
        "total_timing_ms": float(reference["total_timing_ms"]),
        "validity": {"all_rounds_complete": True, "all_outputs_finite": True, "silent_fallback": False, "teacher_reference_zero_drift": True},
    }


def _run_arm(
    torch: Any,
    trace: Any,
    teacher: Any,
    student: Any,
    context: Mapping[str, Any],
    goal: Mapping[str, Any],
    innovations: Any,
    objective: Any,
    reference: Mapping[str, Any],
    mode: str,
    case_index: int,
    device: Any,
) -> dict[str, Any]:
    mu = torch.zeros((HORIZON, ACTION_DIM), device=device)
    sigma = torch.ones_like(mu)
    rounds: dict[str, Any] = {}
    trajectory: list[float] = []
    total_counts = {"student_candidates": 0, "mechanism_teacher_candidates": 0, "diagnostic_oracle_candidates": 0, "student_calls": 0, "mechanism_teacher_calls": 0, "diagnostic_oracle_calls": 0}
    shuffle_base = 2026110000
    for round_index, epsilon_cpu in enumerate(innovations, start=1):
        permutation_list = None
        epsilon = epsilon_cpu.to(device)
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        actions[0] = mu
        _assert_finite(torch, actions, f"{mode} actions round {round_index}")
        student_cost, student_ms = _timed_cost(torch, trace._student_cost, student, context, actions, goal, objective)
        _assert_finite(torch, student_cost, f"{mode} student cost round {round_index}")
        student_order = student_cost.argsort()
        teacher_cost = None
        teacher_ms = 0.0
        mechanism_teacher = False
        selected = student_order[:TOPK]
        if mode != "student_only" and round_index in ANCHORS:
            teacher_cost, teacher_ms = _timed_cost(torch, trace._teacher_cost, teacher, context, actions, goal, objective)
            _assert_finite(torch, teacher_cost, f"{mode} anchor teacher cost round {round_index}")
            mechanism_teacher = True
            teacher_order = teacher_cost.argsort()
            if mode == "shuffled_anchor_P5":
                generator = torch.Generator(device="cpu").manual_seed(shuffle_base + case_index * 100 + round_index)
                permutation = torch.randperm(SAMPLES, generator=generator)
                shuffled_cost = teacher_cost.index_select(0, permutation.to(device))
                selected = shuffled_cost.argsort()[:TOPK]
                permutation_list = permutation.tolist()
            else:
                selected = teacher_order[:TOPK]
                permutation_list = None
        mu_next, sigma_next = _update_distribution(torch, actions, selected)
        _assert_finite(torch, mu_next, f"{mode} mu round {round_index}")
        _assert_finite(torch, sigma_next, f"{mode} sigma round {round_index}")
        reference_mu = reference["updates"][round_index]["mu"]
        reference_sigma = reference["updates"][round_index]["sigma"]
        first_delta = mu_next[0] - reference_mu[0]
        first_action_rms = _rms(first_delta)
        trajectory.append(first_action_rms)
        diagnostic_cost = None
        diagnostic_ms = 0.0
        reused = False
        if round_index in CHECKPOINTS:
            if teacher_cost is not None:
                diagnostic_cost = teacher_cost
                diagnostic_ms = 0.0
                reused = True
            else:
                diagnostic_cost, diagnostic_ms = _timed_cost(torch, trace._teacher_cost, teacher, context, actions, goal, objective)
                _assert_finite(torch, diagnostic_cost, f"{mode} diagnostic teacher cost round {round_index}")
            oracle_order = diagnostic_cost.argsort()
            oracle_top30 = oracle_order[:TOPK]
            regret = _mean_cost(diagnostic_cost, selected) - _mean_cost(diagnostic_cost, oracle_top30)
            overlap = float(len(set(selected.detach().cpu().tolist()).intersection(set(oracle_top30.detach().cpu().tolist()))) / TOPK)
            diagnostic_oracle_candidates = 0 if reused else SAMPLES
            diagnostic_oracle_calls = 0 if reused else 1
        else:
            regret = None
            overlap = None
            diagnostic_oracle_candidates = 0
            diagnostic_oracle_calls = 0
        counts = {
            "student_calls": 1,
            "student_candidates": SAMPLES,
            "mechanism_teacher_calls": 1 if mechanism_teacher else 0,
            "mechanism_teacher_candidates": SAMPLES if mechanism_teacher else 0,
            "diagnostic_oracle_calls": diagnostic_oracle_calls,
            "diagnostic_oracle_candidates": diagnostic_oracle_candidates,
            "diagnostic_oracle_reused_mechanism_call": reused,
        }
        for key in total_counts:
            if key in counts:
                total_counts[key] += int(counts[key])
        row = {
            "round": round_index,
            "mu_rms": _rms(mu_next - reference_mu),
            "sigma_rms": _rms(sigma_next - reference_sigma),
            "first_action_rms": first_action_rms,
            "first_action_coordinate_abs_max": _coordinate_abs_max(first_delta),
            "teacher_top30_overlap": overlap,
            "teacher_cost_regret": regret,
            "pre_update": {"mu": mu.detach().cpu().tolist(), "sigma": sigma.detach().cpu().tolist()},
            "post_update": {"mu": mu_next.detach().cpu().tolist(), "sigma": sigma_next.detach().cpu().tolist()},
            "selected_indices": selected.detach().cpu().tolist(),
            "student_top30_indices": student_order[:TOPK].detach().cpu().tolist(),
            "teacher_top30_indices": None if diagnostic_cost is None else diagnostic_cost.argsort()[:TOPK].detach().cpu().tolist(),
            "shuffle_permutation": permutation_list,
            "counts": counts,
            "timing_ms": {
                "arm_scoring_total": student_ms + teacher_ms,
                "student_native": student_ms,
                "teacher_native": teacher_ms,
                "teacher_oracle_reference": diagnostic_ms,
            },
        }
        rounds[str(round_index)] = row
        mu, sigma = mu_next, sigma_next
    return {
        "mode": mode,
        "rounds": rounds,
        "trajectory_first_action_rms": trajectory,
        "trajectory_auc": float(statistics.fmean(trajectory)),
        "total_counts": total_counts,
        "validity": {"all_rounds_complete": set(rounds) == {str(i) for i in range(1, 31)}, "all_outputs_finite": True, "silent_fallback": False, "teacher_reference_zero_drift": True},
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("mu_rms", "sigma_rms", "first_action_rms", "first_action_coordinate_abs_max", "teacher_top30_overlap", "teacher_cost_regret")
    result: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        rows = [record["rounds"][str(checkpoint)] for record in records]
        result[str(checkpoint)] = {}
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            result[str(checkpoint)][field] = _stats(values)
    result["trajectory_auc"] = _stats([float(record["arm"]["trajectory_auc"]) for record in records])
    return result


def _gate_decisions(records: Mapping[str, Sequence[Mapping[str, Any]]], aggregate: Mapping[str, Any], validity: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    student = records["student_only"]
    treatment = records["periodic_teacher_anchor_P5"]
    shuffled = records["shuffled_anchor_P5"]
    gates = freeze["gates"]
    final = str(ITERATIONS)
    def values(arm: Sequence[Mapping[str, Any]], key: str) -> list[float]:
        return [float(row["rounds"][final][key]) for row in arm]
    def not_worse(a: Sequence[float], b: Sequence[float]) -> int:
        return int(sum(x <= y for x, y in zip(a, b, strict=True)))
    student_rms = values(student, "first_action_rms")
    treatment_rms = values(treatment, "first_action_rms")
    shuffled_rms = values(shuffled, "first_action_rms")
    student_auc = [float(row["arm"]["trajectory_auc"]) for row in student]
    treatment_auc = [float(row["arm"]["trajectory_auc"]) for row in treatment]
    shuffled_auc = [float(row["arm"]["trajectory_auc"]) for row in shuffled]
    student_regret = values(student, "teacher_cost_regret")
    treatment_regret = values(treatment, "teacher_cost_regret")
    checks = {
        "validity_complete_finite_no_fallback": bool(validity["all_cases_complete"] and validity["all_rounds_complete"] and validity["teacher_reference_rounds_complete"] and validity["all_outputs_finite"] and not validity["silent_fallback"] and validity["teacher_reference_zero_drift"]),
        "iteration30_first_action_rms_median": _stats(treatment_rms)["median"] <= float(gates["iteration30_first_action_rms_median_max"]),
        "iteration30_first_action_coordinate_abs_maximum": aggregate["periodic_teacher_anchor_P5"]["per_checkpoint"][final]["first_action_coordinate_abs_max"]["maximum"] <= float(gates["iteration30_first_action_coordinate_abs_max_max"]),
        "iteration30_rms_median_strictly_lower": _stats(treatment_rms)["median"] < _stats(student_rms)["median"],
        "iteration30_rms_cases_not_worse": not_worse(treatment_rms, student_rms) >= int(gates["treatment_vs_student_iter30_rms_cases_not_worse_min"]),
        "trajectory_auc_median_strictly_lower": _stats(treatment_auc)["median"] < _stats(student_auc)["median"],
        "trajectory_auc_cases_not_worse": not_worse(treatment_auc, student_auc) >= int(gates["treatment_vs_student_trajectory_auc_cases_not_worse_min"]),
        "iteration30_regret_median_le_student": _stats(treatment_regret)["median"] <= _stats(student_regret)["median"],
        "iteration30_regret_cases_not_worse": not_worse(treatment_regret, student_regret) >= int(gates["treatment_vs_student_iter30_regret_cases_not_worse_min"]),
    }
    shuffled_reproduces = bool(
        _stats(shuffled_rms)["median"] < _stats(student_rms)["median"]
        and not_worse(shuffled_rms, student_rms) >= int(gates["treatment_vs_student_iter30_rms_cases_not_worse_min"])
        and _stats(shuffled_auc)["median"] < _stats(student_auc)["median"]
        and not_worse(shuffled_auc, student_auc) >= int(gates["treatment_vs_student_trajectory_auc_cases_not_worse_min"])
    )
    treatment_decision = {"status": "GO" if all(checks.values()) and not shuffled_reproduces else "NO-GO", "checks": checks, "negative_control_reproduces_both": shuffled_reproduces, "claim_boundary": freeze["claim_boundary"]["allowed"]}
    return {
        "student_only": {"status": "REFERENCE", "claim_boundary": freeze["claim_boundary"]["allowed"]},
        "periodic_teacher_anchor_P5": treatment_decision,
        "shuffled_anchor_P5": {"status": "FAIL" if shuffled_reproduces else "PASS_NEGATIVE_CONTROL", "reproduces_both": shuffled_reproduces, "iteration30_rms_median": _stats(shuffled_rms)["median"], "trajectory_auc_median": _stats(shuffled_auc)["median"]},
        "negative_control_gate": {"status": "PASS" if not shuffled_reproduces else "FAIL", "reproduces_both": shuffled_reproduces},
    }


def _validity(records: Mapping[str, Sequence[Mapping[str, Any]]], references: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected = set(CASES)
    all_cases = True
    all_rounds = True
    all_finite = True
    fallback = False
    for arm_records in records.values():
        all_cases = all_cases and {int(row["case_index"]) for row in arm_records} == expected
        for row in arm_records:
            all_rounds = all_rounds and set(row["rounds"]) == {str(i) for i in range(1, 31)}
            v = row["arm"]["validity"]
            all_finite = all_finite and bool(v.get("all_outputs_finite", False))
            fallback = fallback or bool(v.get("silent_fallback", False))
    zero = True
    reference_rounds_complete = True
    for reference in references.values():
        reference_rounds_complete = reference_rounds_complete and set(reference["rounds"]) == {str(i) for i in range(1, 31)}
        for update in reference["updates"].values():
            zero = zero and bool((update["mu"] == update["mu"]).all().item()) and bool((update["sigma"] == update["sigma"]).all().item())
    return {"all_cases_complete": all_cases, "all_rounds_complete": all_rounds, "all_outputs_finite": all_finite, "silent_fallback": fallback, "teacher_reference_zero_drift": zero, "teacher_reference_rounds_complete": reference_rounds_complete}


def _runtime_required(args: argparse.Namespace) -> None:
    names = ("root", "output", "parent_summary", "plan_targets", "student_checkpoint", "teacher_checkpoint", "checkpoint_config", "data_root")
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"run mode requires: {', '.join(missing)}")


def _summarize_existing(args: argparse.Namespace, freeze: Mapping[str, Any]) -> int:
    """Recover the deterministic summary from completed per-case JSON files."""

    if args.output is None:
        raise ValueError("summarize mode requires --output")
    output = args.output.resolve()
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ARMS}
    references: dict[str, dict[str, Any]] = {}
    for case_index, eval_seed in zip(CASES, SEEDS, strict=True):
        payload = _load_json(output / f"case_{case_index:02d}.json")
        if int(payload.get("case_index", -1)) != case_index or int(payload.get("eval_seed", -1)) != eval_seed:
            raise ValueError(f"case_{case_index:02d}.json identity drifted")
        reference = payload.get("teacher_full_reference")
        arms = payload.get("arms")
        if not isinstance(reference, dict) or not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ValueError(f"case_{case_index:02d}.json is incomplete")
        references[str(case_index)] = reference
        for name in ARMS:
            arm = arms[name]
            records[name].append({"case_index": case_index, "eval_seed": eval_seed, "rounds": arm["rounds"], "arm": arm})

    expected_rounds = {str(i) for i in range(1, ITERATIONS + 1)}
    validity = {
        "all_cases_complete": all({int(row["case_index"]) for row in arm_records} == set(CASES) for arm_records in records.values()),
        "all_rounds_complete": all(set(row["rounds"]) == expected_rounds for arm_records in records.values() for row in arm_records),
        "all_outputs_finite": all(bool(row["arm"]["validity"].get("all_outputs_finite", False)) for arm_records in records.values() for row in arm_records),
        "silent_fallback": any(bool(row["arm"]["validity"].get("silent_fallback", False)) for arm_records in records.values() for row in arm_records),
        "teacher_reference_zero_drift": all(bool(reference.get("validity", {}).get("teacher_reference_zero_drift", False)) for reference in references.values()),
        "teacher_reference_rounds_complete": all(set(reference.get("rounds", {})) == expected_rounds for reference in references.values()),
    }
    aggregate = {
        name: {
            "per_checkpoint": _aggregate(records[name]),
            "validity": records[name][0]["arm"]["validity"],
            "total_counts": {
                key: int(sum(row["arm"]["total_counts"][key] for row in records[name]))
                for key in records[name][0]["arm"]["total_counts"]
            },
        }
        for name in ARMS
    }
    decisions = _gate_decisions(records, aggregate, validity, freeze)
    for name in ARMS:
        aggregate[name]["decision"] = decisions[name]
    gpu_info = output / "gpu_info.csv"
    gpu = gpu_info.read_text(encoding="utf-8").splitlines()[0].split(",", 1)[0] if gpu_info.is_file() else "recorded by PBS wrapper"
    identity_path = output / "execution_identity.txt"
    identity = {}
    if identity_path.is_file():
        identity = dict(
            line.split("=", 1)
            for line in identity_path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
    summary = {
        "schema": "horizon-weighted-recurrent-student.cem-periodic-anchor-summary",
        "schema_version": 1,
        "freeze": identity.get("freeze", str(args.freeze.resolve())),
        "protocol": identity.get("protocol", str(args.protocol.resolve())),
        "parent_summary": identity.get("parent_summary"),
        "historical_reference": {"job": "24910110.pbs101", "role": "provenance_only_case_seed_and_checkpoint_identity", "runtime_dependency": False, "case_json_loaded": False},
        "teacher_reference_budget": {
            "calls": int(sum(reference["total_counts"]["reference_teacher_calls"] for reference in references.values())),
            "candidates": int(sum(reference["total_counts"]["reference_teacher_candidates"] for reference in references.values())),
            "shared_by_arms": True,
            "excluded_from_intervention_arm_budgets": True,
        },
        "protocol_contract": {"closed_loop": False, "fixed_observation": True, "cases": CASES, "eval_seeds": SEEDS, "common_cpu_innovations": True, "official_cem_semantics": "plain torch.argsort; torch.std unbiased=True; candidate zero equals pre-update mean", "anchor_rounds_1_based": ANCHORS, "diagnostic_checkpoints": CHECKPOINTS},
        "arms": aggregate,
        "decisions": decisions,
        "validity": validity,
        "claim_boundary": freeze["claim_boundary"]["allowed"],
        "closed_loop": "NOT_RUN_BY_SCOPE",
        "gpu": gpu,
        "recovery": {"source": "completed case_*.json", "reason": "original run completed all cases but failed during summary aggregation"},
    }
    (output / "cem_periodic_anchor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": decisions["periodic_teacher_anchor_P5"]["status"], "recovered": True}, indent=2))
    return 0


def main() -> int:
    args = _args()
    freeze = _load_json(args.freeze.resolve())
    protocol = args.protocol.resolve().read_text(encoding="utf-8")
    _validate_freeze(freeze, protocol)
    if args.mode == "preflight":
        print(json.dumps({"status": "READY", "mode": "preflight", "schema": freeze["schema"], "cases": CASES, "checkpoints": CHECKPOINTS, "anchor_rounds": ANCHORS, "models_loaded": False, "datasets_loaded": False}, indent=2))
        return 0
    if args.mode == "summarize":
        return _summarize_existing(args, freeze)
    _runtime_required(args)
    _require_compute_node()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.root.resolve() / "source"))
    import run_dino_pusht_cem_trace as trace
    import run_dino_pusht_closed_loop as closed_loop

    parent = _load_json(args.parent_summary.resolve())
    case_indices, case_seeds = trace._validate_identity(freeze | {"schema": "horizon-weighted-recurrent-student.cem-trace-diagnosis-freeze"}, parent)
    if case_indices != CASES or case_seeds != SEEDS:
        raise ValueError("parent identity differs from periodic-anchor cases")
    teacher, student_wm, model_cfg, dataset = closed_loop._load_models(args, freeze)
    student = student_wm.student
    from planning.objectives import create_objective_fn
    from preprocessor import Preprocessor
    from utils import move_to_device

    with args.plan_targets.resolve().open("rb") as handle:
        targets = pickle.load(handle)
    if not isinstance(targets, dict) or int(targets.get("goal_H", -1)) != HORIZON:
        raise ValueError("unexpected plan_targets payload")
    preprocessor = Preprocessor(action_mean=dataset.action_mean, action_std=dataset.action_std, state_mean=dataset.state_mean, state_std=dataset.state_std, proprio_mean=dataset.proprio_mean, proprio_std=dataset.proprio_std, transform=dataset.transform)
    device = torch.device("cuda:0")
    transformed_obs = move_to_device(preprocessor.transform_obs(targets["obs_0"]), device)
    transformed_goal = move_to_device(preprocessor.transform_obs(targets["obs_g"]), device)
    with torch.no_grad():
        encoded_obs = teacher.encode_obs(transformed_obs)
        encoded_goal = teacher.encode_obs(transformed_goal)
    objective = create_objective_fn(alpha=1, base=2, mode="last")
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ARMS}
    references: dict[str, dict[str, Any]] = {}
    for case_index, eval_seed in zip(CASES, SEEDS, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(INNOVATION_SEED_BASE + case_index)
        innovations = torch.randn((ITERATIONS, SAMPLES, HORIZON, ACTION_DIM), generator=generator)
        context = trace._repeat_case(encoded_obs, case_index, SAMPLES)
        goal = trace._repeat_case(encoded_goal, case_index, SAMPLES)
        reference = _run_teacher_reference(torch, trace, teacher, context, goal, innovations, objective, device)
        references[str(case_index)] = reference
        case_payload: dict[str, Any] = {"case_index": case_index, "eval_seed": eval_seed, "teacher_full_reference": _reference_arm(torch, reference), "arms": {}}
        for mode in ARMS:
            arm = _run_arm(torch, trace, teacher, student, context, goal, innovations, objective, reference, mode, case_index, device)
            records[mode].append({"case_index": case_index, "eval_seed": eval_seed, "rounds": arm["rounds"], "arm": arm})
            case_payload["arms"][mode] = arm
        (output / f"case_{case_index:02d}.json").write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
        print(json.dumps({"case_index": case_index, "eval_seed": eval_seed, "completed": True}), flush=True)
    validity = _validity(records, references)
    aggregate = {name: {"per_checkpoint": _aggregate(records[name]), "validity": records[name][0]["arm"]["validity"], "total_counts": {key: int(sum(row["arm"]["total_counts"][key] for row in records[name])) for key in records[name][0]["arm"]["total_counts"]}} for name in ARMS}
    decisions = _gate_decisions(records, aggregate, validity, freeze)
    for name in ARMS:
        aggregate[name]["decision"] = decisions[name]
    summary = {
        "schema": "horizon-weighted-recurrent-student.cem-periodic-anchor-summary",
        "schema_version": 1,
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "parent_summary": str(args.parent_summary.resolve()),
        "historical_reference": {"job": "24910110.pbs101", "role": "provenance_only_case_seed_and_checkpoint_identity", "runtime_dependency": False, "case_json_loaded": False},
        "teacher_reference_budget": {"calls": int(sum(reference["total_counts"]["reference_teacher_calls"] for reference in references.values())), "candidates": int(sum(reference["total_counts"]["reference_teacher_candidates"] for reference in references.values())), "shared_by_arms": True, "excluded_from_intervention_arm_budgets": True},
        "protocol_contract": {"closed_loop": False, "fixed_observation": True, "cases": CASES, "eval_seeds": SEEDS, "common_cpu_innovations": True, "official_cem_semantics": "plain torch.argsort; torch.std unbiased=True; candidate zero equals pre-update mean", "anchor_rounds_1_based": ANCHORS, "diagnostic_checkpoints": CHECKPOINTS},
        "arms": aggregate,
        "decisions": decisions,
        "validity": validity,
        "claim_boundary": freeze["claim_boundary"]["allowed"],
        "closed_loop": "NOT_RUN_BY_SCOPE",
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "cem_periodic_anchor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": decisions["periodic_teacher_anchor_P5"]["status"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
