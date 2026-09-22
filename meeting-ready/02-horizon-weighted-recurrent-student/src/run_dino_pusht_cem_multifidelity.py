#!/usr/bin/env python3
"""Fixed-observation teacher-verified multi-fidelity CEM mechanism experiment.

This runner is deliberately separate from the historical CEM trace runner.  It
uses the same frozen cases, innovation coupling, cost functions, and official
``argsort``/``std`` semantics, but adds a student-prefilter arm: the student
scores all candidates, the teacher scores only the top-M superset, and the
teacher-scored elites update the proposal distribution.

Teacher scores used only to compute diagnostic containment are reported as
``teacher_oracle`` and are excluded from arm timing and teacher-call budgets.
They are necessary because containment is defined against the teacher's true
top-30 over all 300 candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--plan-targets", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--prefilter-m",
        type=int,
        nargs="+",
        default=[60, 120],
        help="teacher-scored superset sizes; every value must be > topk and <= num_samples",
    )
    return parser.parse_args()


def _require_compute_node() -> None:
    """Refuse local/login execution before importing torch or loading models."""

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


def _cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _assert_finite(torch: Any, value: Any, label: str) -> None:
    """Fail immediately when a per-round model/CEM output is non-finite."""

    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite {label}")


def _timed_cost(torch: Any, cost_fn: Any, model: Any, context: Mapping[str, Any], actions: Any, goal: Mapping[str, Any], objective: Any) -> tuple[Any, float]:
    _cuda_sync(torch)
    started = time.perf_counter()
    with torch.no_grad():
        cost = cost_fn(model, context, actions, goal, objective)
    _cuda_sync(torch)
    return cost, (time.perf_counter() - started) * 1000.0


def _select_prefilter(student_cost: Any, m: int) -> Any:
    """Return top-M indices while always retaining official candidate zero."""

    order = student_cost.argsort()
    zero = order.new_tensor([0])
    if bool((order[:m] == 0).any()):
        selected = order[:m]
    else:
        import torch

        selected = torch.cat((zero, order[order != 0][: m - 1]))
    if selected.numel() != m or not bool((selected == 0).any()):
        raise RuntimeError("prefilter must retain candidate zero with exact cardinality")
    return selected


def _update_distribution(torch: Any, actions: Any, indices: Any) -> tuple[Any, Any]:
    elites = actions.index_select(0, indices)
    # Keep the official implementation's default unbiased estimator.
    return elites.mean(dim=0), elites.std(dim=0)


def _rms(value: Any) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _coordinate_abs_max(value: Any) -> float:
    return float(value.detach().float().abs().max().cpu())


def _overlap(first: Any, second: Any, k: int) -> float:
    a = set(first[:k].detach().cpu().tolist())
    b = set(second[:k].detach().cpu().tolist())
    return float(len(a.intersection(b)) / k)


def _containment(superset: Any, reference_topk: Any, k: int) -> float:
    a = set(superset.detach().cpu().tolist())
    b = set(reference_topk[:k].detach().cpu().tolist())
    return float(len(a.intersection(b)) / k)


def _mean_cost(cost: Any, indices: Any) -> float:
    return float(cost.index_select(0, indices).mean().detach().cpu())


def _validate_prefilter(values: Sequence[int], topk: int, samples: int) -> list[int]:
    result = sorted({int(value) for value in values})
    if not result:
        raise ValueError("--prefilter-m must contain at least one value")
    if any(value <= topk or value > samples for value in result):
        raise ValueError(f"every prefilter M must satisfy topk < M <= num_samples ({topk} < M <= {samples})")
    return result


def _multifidelity_config(freeze: Mapping[str, Any], prefilter_ms: Sequence[int]) -> Mapping[str, Any]:
    config = freeze.get("multifidelity")
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("freeze.multifidelity must be an object")
    expected = config.get("prefilter_m")
    if expected is not None:
        expected_values = sorted({int(value) for value in expected})
        if list(prefilter_ms) != expected_values:
            raise ValueError(f"--prefilter-m {list(prefilter_ms)} differs from frozen multifidelity.prefilter_m {expected_values}")
    gates = config.get("gates", {})
    if gates is not None and not isinstance(gates, Mapping):
        raise ValueError("freeze.multifidelity.gates must be an object")
    return config


def _aggregate(records: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]) -> dict[str, Any]:
    fields = (
        "selected_top30_overlap",
        "prefilter_top_m_containment",
        "student_top30_overlap",
        "teacher_cost_regret",
        "mu_rms",
        "sigma_rms",
        "first_action_rms",
        "first_action_coordinate_abs_max",
        "timing_ms.arm_scoring_total",
        "timing_ms.student_native",
        "timing_ms.teacher_native",
        "timing_ms.teacher_oracle_reference",
    )
    result: dict[str, Any] = {}
    for checkpoint in checkpoints:
        rows = [record["rounds"][str(checkpoint)] for record in records]
        summary: dict[str, Any] = {}
        for field in fields:
            values: list[float] = []
            for row in rows:
                current: Any = row
                for part in field.split("."):
                    current = current[part]
                if current is not None:
                    values.append(float(current))
            if values:
                summary[field] = {
                    "mean": float(sum(values) / len(values)),
                    "median": float(statistics.median(values)),
                    "minimum": float(min(values)),
                    "maximum": float(max(values)),
                }
        result[str(checkpoint)] = summary
    return result


def _gate_value(gates: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in gates:
            return gates[name]
    return None


def _decisions(
    aggregate: Mapping[str, Any],
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    checkpoints: Sequence[int],
    multifidelity: Mapping[str, Any],
) -> dict[str, Any]:
    gates = multifidelity.get("gates", {})
    if not isinstance(gates, Mapping) or not gates:
        return {name: {"status": "NOT_CONFIGURED"} for name in aggregate}
    rms_limit = _gate_value(gates, "iteration30_first_action_rms_median_max", "first_action_rms_median_max")
    coordinate_limit = _gate_value(
        gates,
        "iteration30_first_action_coordinate_abs_max_max",
        "first_action_coordinate_abs_max_max",
    )
    cases_limit = _gate_value(gates, "cases_not_worse_than_student_min")
    regret_rule = _gate_value(gates, "selected_teacher_cost_regret_median_le_student_only")
    required = {
        "iteration30_first_action_rms_median_max": rms_limit,
        "iteration30_first_action_coordinate_abs_max_max": coordinate_limit,
        "cases_not_worse_than_student_min": cases_limit,
        "selected_teacher_cost_regret_median_le_student_only": regret_rule,
    }
    if any(value is None for value in required.values()):
        return {
            name: {"status": "NOT_CONFIGURED", "missing_gates": [key for key, value in required.items() if value is None]}
            for name in aggregate
        }
    final_round = str(max(checkpoints))
    student_regrets = [float(row["rounds"][final_round]["teacher_cost_regret"]) for row in records_by_arm["student_only"]]
    student_regret_median = float(statistics.median(student_regrets))
    decisions: dict[str, Any] = {}
    for name, arm_summary in aggregate.items():
        final = arm_summary["per_round"][final_round]
        arm_regrets = [float(row["rounds"][final_round]["teacher_cost_regret"]) for row in records_by_arm[name]]
        cases_not_worse = sum(
            arm_value <= student_value
            for arm_value, student_value in zip(arm_regrets, student_regrets, strict=True)
        )
        checks = {
            "first_action_rms_median": final["first_action_rms"]["median"] <= float(rms_limit),
            "first_action_coordinate_abs_maximum": final["first_action_coordinate_abs_max"]["maximum"] <= float(coordinate_limit),
            "selected_teacher_cost_regret_median": (
                final["teacher_cost_regret"]["median"] <= student_regret_median
                if bool(regret_rule)
                else True
            ),
            "cases_not_worse_than_student": cases_not_worse >= int(cases_limit),
        }
        decisions[name] = {
            "status": "GO" if all(checks.values()) else "NO-GO",
            "checks": checks,
            "metrics": {
                "iteration30_first_action_rms_median": final["first_action_rms"]["median"],
                "iteration30_first_action_coordinate_abs_maximum": final["first_action_coordinate_abs_max"]["maximum"],
                "iteration30_selected_teacher_cost_regret_median": final["teacher_cost_regret"]["median"],
                "student_only_teacher_cost_regret_median": student_regret_median,
                "cases_not_worse_than_student": int(cases_not_worse),
                "case_count": len(student_regrets),
            },
            "claim_boundary": "fixed-observation mechanism diagnosis only",
        }
    return decisions


def _validity_summary(
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    case_indices: Sequence[int],
    checkpoints: Sequence[int],
) -> dict[str, Any]:
    expected_cases = sorted(int(value) for value in case_indices)
    all_cases_complete = len(expected_cases) == 6
    all_outputs_finite = True
    silent_fallback = False
    superset_cardinality_correct = True
    for records in records_by_arm.values():
        observed_cases = sorted(int(record["case_index"]) for record in records)
        all_cases_complete = all_cases_complete and observed_cases == expected_cases
        for record in records:
            all_cases_complete = all_cases_complete and set(record["rounds"]) == {str(value) for value in checkpoints}
            validity = record.get("validity", {})
            all_outputs_finite = all_outputs_finite and bool(validity.get("all_outputs_finite", False))
            silent_fallback = silent_fallback or bool(validity.get("silent_fallback", False))
            superset_cardinality_correct = superset_cardinality_correct and bool(
                validity.get("superset_cardinality_correct", False)
            )
    return {
        "all_six_cases_complete": bool(all_cases_complete),
        "all_outputs_finite": bool(all_outputs_finite),
        "silent_fallback": bool(silent_fallback),
        "superset_cardinality_correct": bool(superset_cardinality_correct),
    }


def _run_teacher_full_case(
    torch: Any,
    innovations: Any,
    samples: int,
    topk: int,
    horizon: int,
    action_dim: int,
    checkpoints: Sequence[int],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    device = innovations.device
    mu = torch.zeros((horizon, action_dim), device=device)
    sigma = torch.ones_like(mu)
    rounds: dict[str, Any] = {}
    for round_index, epsilon_cpu in enumerate(innovations, start=1):
        epsilon = epsilon_cpu.to(device)
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        actions[0] = mu
        _assert_finite(torch, actions, f"teacher_full actions at round {round_index}")
        # The reference pass is the actual teacher-full arm. Reusing it keeps
        # the diagnostic oracle from doubling the expensive teacher rollout.
        costs = reference["costs"][round_index - 1]
        _assert_finite(torch, costs, f"teacher_full cost at round {round_index}")
        teacher_ms = float(reference["timing_ms"][round_index - 1])
        teacher_order = costs.argsort()
        elite_indices = teacher_order[:topk]
        mu_next, sigma_next = _update_distribution(torch, actions, elite_indices)
        _assert_finite(torch, mu_next, f"teacher_full mu at round {round_index}")
        _assert_finite(torch, sigma_next, f"teacher_full sigma at round {round_index}")
        if round_index in checkpoints:
            rounds[str(round_index)] = {
                "round": round_index,
                "selected_top30_overlap": 1.0,
                "prefilter_top_m_containment": None,
                "student_top30_overlap": None,
                "teacher_cost_regret": 0.0,
                "mu_rms": 0.0,
                "sigma_rms": 0.0,
                "first_action_rms": 0.0,
                "first_action_coordinate_abs_max": 0.0,
                "pre_update": {"mu": mu.detach().cpu().tolist(), "sigma": sigma.detach().cpu().tolist()},
                "post_update": {"mu": mu_next.detach().cpu().tolist(), "sigma": sigma_next.detach().cpu().tolist()},
                "selected_indices": elite_indices.detach().cpu().tolist(),
                "teacher_top30_indices": elite_indices.detach().cpu().tolist(),
                "prefilter_top_m_indices": None,
                "counts": {
                    "student_calls": 0,
                    "student_candidates": 0,
                    "teacher_calls": 1,
                    "teacher_candidates": samples,
                    "teacher_oracle_calls": 0,
                    "teacher_oracle_candidates": 0,
                },
                "timing_ms": {
                    "arm_scoring_total": teacher_ms,
                    "student_native": 0.0,
                    "teacher_native": teacher_ms,
                    "teacher_oracle_reference": 0.0,
                },
            }
        mu, sigma = mu_next, sigma_next
    return {
        "mode": "teacher_full",
        "prefilter_m": None,
        "rounds": rounds,
        "validity": {
            "all_rounds_complete": True,
            "all_outputs_finite": True,
            "silent_fallback": False,
            "superset_cardinality_correct": True,
        },
    }


def _run_student_or_prefilter_case(
    torch: Any,
    trace: Any,
    teacher: Any,
    student: Any,
    context: Mapping[str, Any],
    goal: Mapping[str, Any],
    innovations: Any,
    objective: Any,
    samples: int,
    topk: int,
    horizon: int,
    action_dim: int,
    checkpoints: Sequence[int],
    mode: str,
    prefilter_m: int | None,
    teacher_reference: Mapping[str, Any],
) -> dict[str, Any]:
    device = innovations.device
    mu = torch.zeros((horizon, action_dim), device=device)
    sigma = torch.ones_like(mu)
    rounds: dict[str, Any] = {}
    for round_index, epsilon_cpu in enumerate(innovations, start=1):
        epsilon = epsilon_cpu.to(device)
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        actions[0] = mu
        _assert_finite(torch, actions, f"{mode} actions at round {round_index}")
        student_cost, student_ms = _timed_cost(torch, trace._student_cost, student, context, actions, goal, objective)
        _assert_finite(torch, student_cost, f"{mode} student cost at round {round_index}")
        student_order = student_cost.argsort()
        if mode == "student_only":
            mechanism_teacher_ms = 0.0
            mechanism_teacher_candidates = 0
            prefilter_indices = None
            selected_indices = student_order[:topk]
            selected_teacher_cost = None
        else:
            prefilter_indices = _select_prefilter(student_cost, int(prefilter_m))
            if prefilter_indices.numel() != int(prefilter_m) or not bool((prefilter_indices == 0).any()):
                raise RuntimeError(f"prefilter cardinality/candidate-zero violation at round {round_index}")
            subset_actions = actions.index_select(0, prefilter_indices)
            subset_context = {key: value.index_select(0, prefilter_indices) for key, value in context.items()}
            subset_goal = {key: value.index_select(0, prefilter_indices) for key, value in goal.items()}
            subset_teacher_cost, mechanism_teacher_ms = _timed_cost(
                torch,
                trace._teacher_cost,
                teacher,
                subset_context,
                subset_actions,
                subset_goal,
                objective,
            )
            _assert_finite(torch, subset_teacher_cost, f"{mode} teacher subset cost at round {round_index}")
            subset_order = subset_teacher_cost.argsort()
            selected_indices = prefilter_indices.index_select(0, subset_order[:topk])
            selected_teacher_cost = subset_teacher_cost.index_select(0, subset_order[:topk])
            mechanism_teacher_candidates = int(prefilter_m)
        mu_next, sigma_next = _update_distribution(torch, actions, selected_indices)
        _assert_finite(torch, mu_next, f"{mode} mu at round {round_index}")
        _assert_finite(torch, sigma_next, f"{mode} sigma at round {round_index}")

        if round_index in checkpoints:
            # The diagnostic oracle must score this arm's current candidate
            # pool. The teacher-reference costs belong to the teacher arm's
            # different candidate pool after round one.
            oracle_cost, oracle_ms = _timed_cost(
                torch,
                trace._teacher_cost,
                teacher,
                context,
                actions,
                goal,
                objective,
            )
            _assert_finite(torch, oracle_cost, f"{mode} oracle cost at round {round_index}")
            oracle_order = oracle_cost.argsort()
            oracle_topk = oracle_order[:topk]
            student_topk = student_order[:topk]
            teacher_cost_regret = _mean_cost(oracle_cost, selected_indices) - _mean_cost(oracle_cost, oracle_topk)
            reference_mu = teacher_reference["updates"][round_index - 1]["mu"]
            reference_sigma = teacher_reference["updates"][round_index - 1]["sigma"]
            rounds[str(round_index)] = {
                "round": round_index,
                "selected_top30_overlap": _overlap(selected_indices, oracle_topk, topk),
                "student_top30_overlap": _overlap(student_topk, oracle_topk, topk),
                "prefilter_top_m_containment": None if prefilter_indices is None else _containment(prefilter_indices, oracle_topk, topk),
                "teacher_cost_regret": float(teacher_cost_regret),
                "mu_rms": _rms(mu_next - reference_mu),
                "sigma_rms": _rms(sigma_next - reference_sigma),
                "first_action_rms": _rms(mu_next[0] - reference_mu[0]),
                "first_action_coordinate_abs_max": _coordinate_abs_max(mu_next[0] - reference_mu[0]),
                "pre_update": {"mu": mu.detach().cpu().tolist(), "sigma": sigma.detach().cpu().tolist()},
                "post_update": {"mu": mu_next.detach().cpu().tolist(), "sigma": sigma_next.detach().cpu().tolist()},
                "selected_indices": selected_indices.detach().cpu().tolist(),
                "student_top30_indices": student_topk.detach().cpu().tolist(),
                "teacher_top30_indices": oracle_topk.detach().cpu().tolist(),
                "prefilter_top_m_indices": None if prefilter_indices is None else prefilter_indices.detach().cpu().tolist(),
                "counts": {
                    "student_calls": 1,
                    "student_candidates": samples,
                    "teacher_calls": 0 if mode == "student_only" else 1,
                    "teacher_candidates": mechanism_teacher_candidates,
                    "teacher_oracle_calls": 1,
                    "teacher_oracle_candidates": samples,
                },
                "timing_ms": {
                    "arm_scoring_total": student_ms + mechanism_teacher_ms,
                    "student_native": student_ms,
                    "teacher_native": mechanism_teacher_ms,
                    "teacher_oracle_reference": oracle_ms,
                },
            }
            if selected_teacher_cost is not None:
                rounds[str(round_index)]["mechanism_teacher_elite_cost_mean"] = float(selected_teacher_cost.mean().detach().cpu())
        mu, sigma = mu_next, sigma_next
    return {
        "mode": mode,
        "prefilter_m": prefilter_m,
        "rounds": rounds,
        "validity": {
            "all_rounds_complete": True,
            "all_outputs_finite": True,
            "silent_fallback": False,
            "superset_cardinality_correct": True,
        },
    }


def _teacher_reference(
    torch: Any,
    trace: Any,
    teacher: Any,
    context: Mapping[str, Any],
    goal: Mapping[str, Any],
    innovations: Any,
    objective: Any,
    horizon: int,
    action_dim: int,
    topk: int,
) -> dict[str, Any]:
    device = innovations.device
    mu = torch.zeros((horizon, action_dim), device=device)
    sigma = torch.ones_like(mu)
    costs: list[Any] = []
    updates: list[dict[str, Any]] = []
    timing_ms: list[float] = []
    for epsilon_cpu in innovations:
        epsilon = epsilon_cpu.to(device)
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        actions[0] = mu
        _assert_finite(torch, actions, "teacher reference actions")
        cost, elapsed = _timed_cost(torch, trace._teacher_cost, teacher, context, actions, goal, objective)
        _assert_finite(torch, cost, "teacher reference cost")
        order = cost.argsort()
        mu_next, sigma_next = _update_distribution(torch, actions, order[:topk])
        _assert_finite(torch, mu_next, "teacher reference mu")
        _assert_finite(torch, sigma_next, "teacher reference sigma")
        costs.append(cost)
        updates.append({"mu": mu_next, "sigma": sigma_next})
        timing_ms.append(elapsed)
        mu, sigma = mu_next, sigma_next
    return {"costs": costs, "updates": updates, "timing_ms": timing_ms}


def main() -> int:
    args = _args()
    _require_compute_node()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    parent = _load_json(args.parent_summary.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_dino_pusht_cem_trace as trace
    import run_dino_pusht_closed_loop as closed_loop

    case_indices, case_seeds = trace._validate_identity(freeze, parent)
    teacher, student_wm, model_cfg, dataset = closed_loop._load_models(args, freeze)
    student = student_wm.student
    source = args.root.resolve() / "source"
    sys.path.insert(0, str(source))
    from planning.objectives import create_objective_fn
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
    with args.plan_targets.resolve().open("rb") as handle:
        targets = pickle.load(handle)
    if not isinstance(targets, dict) or int(targets.get("goal_H", -1)) != 5:
        raise ValueError("unexpected plan_targets payload")
    obs_0 = targets["obs_0"]
    obs_g = targets["obs_g"]
    if int(obs_0["visual"].shape[0]) != 8:
        raise ValueError("parent plan_targets must contain the frozen eight cases")
    device = torch.device("cuda:0")
    transformed_obs = move_to_device(preprocessor.transform_obs(obs_0), device)
    transformed_goal = move_to_device(preprocessor.transform_obs(obs_g), device)
    with torch.no_grad():
        encoded_obs = teacher.encode_obs(transformed_obs)
        encoded_goal = teacher.encode_obs(transformed_goal)

    cem = freeze["cem"]
    iterations = int(cem["iterations"])
    checkpoints = [int(value) for value in cem["checkpoints"]]
    if 30 not in checkpoints:
        raise ValueError("multifidelity gates require diagnostic checkpoint 30")
    samples = int(cem["num_samples"])
    topk = int(cem["topk"])
    horizon = int(cem["horizon"])
    action_dim = int(cem["packed_action_dim"])
    if action_dim != int(dataset.action_dim * model_cfg.frameskip):
        raise ValueError("packed action dimension differs from official dataset contract")
    prefilter_ms = _validate_prefilter(args.prefilter_m, topk, samples)
    multifidelity = _multifidelity_config(freeze, prefilter_ms)
    objective = create_objective_fn(alpha=1, base=2, mode="last")
    arm_names = ["teacher_full", "student_only"] + [f"student_prefilter_teacher_m{value:03d}" for value in prefilter_ms]
    all_records: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_names}

    for case_index, eval_seed in zip(case_indices, case_seeds, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(int(cem["innovation_seed_base"]) + case_index)
        innovations_cpu = torch.randn((iterations, samples, horizon, action_dim), generator=generator)
        innovations = innovations_cpu.to(device)
        context = trace._repeat_case(encoded_obs, case_index, samples)
        goal = trace._repeat_case(encoded_goal, case_index, samples)
        reference = _teacher_reference(torch, trace, teacher, context, goal, innovations, objective, horizon, action_dim, topk)
        teacher_arm = _run_teacher_full_case(
            torch, innovations, samples, topk, horizon, action_dim, checkpoints, reference
        )
        all_records["teacher_full"].append({"case_index": case_index, "eval_seed": eval_seed, **teacher_arm})
        case_payload: dict[str, Any] = {"case_index": case_index, "eval_seed": eval_seed, "arms": {"teacher_full": teacher_arm}}
        student_arm = _run_student_or_prefilter_case(
            torch, trace, teacher, student, context, goal, innovations, objective, samples, topk, horizon, action_dim,
            checkpoints, "student_only", None, reference,
        )
        all_records["student_only"].append({"case_index": case_index, "eval_seed": eval_seed, **student_arm})
        case_payload["arms"]["student_only"] = student_arm
        for value in prefilter_ms:
            name = f"student_prefilter_teacher_m{value:03d}"
            arm = _run_student_or_prefilter_case(
                torch, trace, teacher, student, context, goal, innovations, objective, samples, topk, horizon, action_dim,
                checkpoints, "student_prefilter_teacher", value, reference,
            )
            all_records[name].append({"case_index": case_index, "eval_seed": eval_seed, **arm})
            case_payload["arms"][name] = arm
        (output / f"case_{case_index:02d}.json").write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
        print(json.dumps({"case_index": case_index, "eval_seed": eval_seed, "completed": True}))

    aggregate: dict[str, Any] = {}
    for name, records in all_records.items():
        rounds_total = len(records) * iterations
        mode = records[0]["mode"]
        prefilter_m = records[0]["prefilter_m"]
        aggregate[name] = {
            "mode": mode,
            "prefilter_m": prefilter_m,
            "per_round": _aggregate(records, checkpoints),
            "validity": records[0]["validity"],
            "total_counts": {
                "student_candidates": rounds_total * samples if mode != "teacher_full" else 0,
                "student_calls": rounds_total if mode != "teacher_full" else 0,
                "teacher_candidates": (
                    rounds_total * samples
                    if mode == "teacher_full"
                    else rounds_total * int(prefilter_m or 0)
                ),
                "teacher_calls": rounds_total if mode == "teacher_full" else (rounds_total if prefilter_m else 0),
                "teacher_oracle_candidates": 0 if mode == "teacher_full" else len(records) * len(checkpoints) * samples,
                "teacher_oracle_calls": 0 if mode == "teacher_full" else len(records) * len(checkpoints),
            },
        }
    decisions = _decisions(aggregate, all_records, checkpoints, multifidelity)
    for name, decision in decisions.items():
        aggregate[name]["decision"] = decision
    validity = _validity_summary(all_records, case_indices, checkpoints)
    summary = {
        "schema": "horizon-weighted-recurrent-student.cem-multifidelity-summary",
        "schema_version": 1,
        "freeze": str(args.freeze.resolve()),
        "parent_summary": str(args.parent_summary.resolve()),
        "plan_targets": str(args.plan_targets.resolve()),
        "protocol": {
            "closed_loop": False,
            "diagnostic_only": True,
            "fixed_observation": "first MPC observation from the parent paired pilot",
            "cases": case_indices,
            "candidate_coupling": "common CPU-generated standard-normal innovations per case and round",
            "official_cem_semantics": "plain torch.argsort; torch.std unbiased=True; candidate zero equals pre-update mean",
            "reported_rounds": checkpoints,
            "prefilter_m": prefilter_ms,
            "teacher_oracle_boundary": "full teacher scoring is diagnostic-only for containment and is excluded from arm teacher budgets and arm timing",
        },
        "arms": aggregate,
        "decisions": decisions,
        "validity": validity,
        "claim_boundary": freeze["claim_boundary"],
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "cem_multifidelity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "arms": arm_names}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
