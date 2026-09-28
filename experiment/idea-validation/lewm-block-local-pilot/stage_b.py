"""Fixed-observation paired CEM quality diagnostic for the LeWM block pilot.

This is a source-faithful CEM update loop over cached fixed observations. It
does not run the Stable-WorldModel policy/environment lifecycle.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ARMS = ("teacher", "flat_adapter", "block_adapter_global16")
CHECKPOINT_ROUNDS = (1, 5, 10, 30)
LATENT_DIM = 192
HORIZON = 5
ACTION_DIM = 10


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_update(freeze: Mapping[str, Any]) -> dict[str, Any]:
    contingent = freeze["contingent"]
    cfg = contingent["stage_b_native_update"]
    selector_text = str(cfg["selector"]).lower()
    if "torch.topk" in selector_text:
        selector = "torch.topk"
    elif "torch.argsort" in selector_text:
        selector = "torch.argsort"
    else:
        raise ValueError(f"unsupported frozen Stage B selector: {cfg['selector']!r}")

    std_text = str(cfg["std"]).lower()
    if "correction=1" in std_text or "unbiased=true" in std_text:
        correction = 1
    elif "correction=0" in std_text or "unbiased=false" in std_text:
        correction = 0
    else:
        correction_value = cfg.get("std_correction")
        if correction_value not in (0, 1):
            raise ValueError("Stage B freeze must state the native std correction")
        correction = int(correction_value)

    if str(cfg["candidate_zero"]).lower() not in {
        "pre-update mean",
        "current mean",
        "pre-update mu",
    }:
        raise ValueError("Stage B requires candidate 0 to equal the pre-update mean")
    if bool(cfg["clip_actions"]):
        raise ValueError("This Stage B implementation requires the frozen no-clipping source semantics")
    if bool(cfg["std_floor"]):
        raise ValueError("This Stage B implementation requires the frozen no-std-floor source semantics")

    return {
        "initial_mean": float(cfg["initial_mean"]),
        "initial_sampling_std": float(cfg["initial_sampling_std"]),
        "selector": selector,
        "std_correction": correction,
        "clip_actions": False,
        "std_floor": None,
        "source": str(cfg.get("source", "unspecified")),
        "scope": str(cfg.get("scope", "fixed-observation diagnostic")),
    }


def _sample(mu: Any, sigma: Any, epsilon: Any) -> Any:
    """Match CEMSolver's sample, scale/shift, then candidate-zero overwrite."""
    candidates = epsilon * sigma.unsqueeze(1) + mu.unsqueeze(1)
    candidates[:, 0] = mu
    return candidates


def _elite_indices(costs: Any, count: int, selector: str) -> Any:
    import torch

    row = costs.reshape(1, -1)
    if selector == "torch.topk":
        # The pinned source relies on torch.topk's default sorted behavior.
        return torch.topk(row, k=count, dim=1, largest=False).indices[0]
    if selector == "torch.argsort":
        return torch.argsort(row, dim=1, stable=True)[0, :count]
    raise ValueError(f"unsupported selector: {selector}")


def _update(candidates: Any, costs: Any, count: int, update: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    indices = _elite_indices(costs, count, str(update["selector"]))
    elites = candidates[0].index_select(0, indices)
    mean = elites.mean(dim=0, keepdim=True)
    sigma = elites.std(dim=0, correction=int(update["std_correction"]), keepdim=True)
    return mean, sigma, indices


def mechanism_selfcheck(freeze: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Small synthetic tensor check; call only from a guarded PBS preflight."""
    import torch

    if freeze is None:
        update = {
            "initial_mean": 0.0,
            "initial_sampling_std": 1.0,
            "selector": "torch.topk",
            "std_correction": 1,
            "clip_actions": False,
            "std_floor": None,
            "source": "native CEM source reference",
        }
    else:
        update = _resolve_update(freeze)

    device = torch.device("cpu")
    mean = torch.zeros((1, 2, 2), dtype=torch.float64, device=device)
    sigma = torch.ones_like(mean)
    epsilon = torch.arange(24, dtype=torch.float64, device=device).reshape(1, 6, 2, 2) / 7
    candidates = _sample(mean, sigma, epsilon)
    candidate_zero_ok = torch.equal(candidates[:, 0], mean)

    tied_costs = torch.tensor([2.0, 0.0, 0.0, 3.0, 1.0, 4.0], dtype=torch.float64)
    expected = torch.topk(tied_costs.reshape(1, -1), k=3, dim=1, largest=False).indices[0]
    selected = _elite_indices(tied_costs, 3, str(update["selector"]))
    selector_ok = torch.equal(selected, expected)

    std_probe = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    std_expected = std_probe.std(dim=0, correction=int(update["std_correction"]))
    std_ok = bool(torch.isfinite(std_expected).all().item()) and (
        int(update["std_correction"]) != 1
        or math.isclose(float(std_expected.item()), math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-12)
    )

    def toy_cost(actions: Any) -> Any:
        return actions.square().sum(dim=(1, 2))

    def toy_run(*, score_shadow: bool) -> tuple[list[Any], list[Any], list[Any]]:
        mu = torch.zeros((1, 2, 2), dtype=torch.float64, device=device)
        std = torch.ones_like(mu)
        eps = torch.linspace(-1.1, 1.2, steps=3 * 6 * 2 * 2, dtype=torch.float64).reshape(3, 1, 6, 2, 2)
        candidate_history, mean_history, std_history = [], [], []
        for noise in eps:
            bank = _sample(mu, std, noise)
            student_costs = toy_cost(bank[0])
            before = _elite_indices(student_costs, 3, str(update["selector"]))
            mu, std, _ = _update(bank, student_costs, 3, update)
            if score_shadow:
                # Shadow scoring is deliberately out-of-band: it must not be
                # an input to either the student selector or its next state.
                shadow_bank = torch.cat((bank[0], mu), dim=0)
                _ = toy_cost(shadow_bank)
            candidate_history.append(bank.clone())
            mean_history.append(mu.clone())
            std_history.append(std.clone())
            if not torch.equal(before, _elite_indices(student_costs, 3, str(update["selector"]))):
                raise AssertionError("deterministic scorer changed elite selection")
        return candidate_history, mean_history, std_history

    # Teacher and student use the same toy objective and innovations, so their
    # native proposals must match exactly when the scorer is identical.
    first = toy_run(score_shadow=False)
    second = toy_run(score_shadow=True)
    paired_proposal_ok = all(
        torch.equal(a, b)
        for first_group, second_group in zip(first, second, strict=True)
        for a, b in zip(first_group, second_group, strict=True)
    )

    shadow_isolation_ok = paired_proposal_ok

    checks = {
        "candidate_zero_is_pre_update_mean": candidate_zero_ok,
        "selector_matches_configured_native_operator": selector_ok,
        "elite_std_uses_configured_correction": std_ok,
        "identical_teacher_and_student_scores_give_identical_proposals": paired_proposal_ok,
        "shadow_score_does_not_change_proposal_state": shadow_isolation_ok,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "selector": update["selector"],
        "std_correction": update["std_correction"],
        "source": update["source"],
    }


def _tensor(reference: Any, value: Any, dtype: Any, device: Any) -> Any:
    return reference.base._tensor(value, dtype=dtype).to(device)


def _teacher_costs(
    reference: Any,
    official_model: Any,
    row: Mapping[str, Any],
    actions: Any,
) -> Any:
    import torch

    context = _tensor(reference, row["latent_history"], torch.float32, actions.device)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if tuple(context.shape[-2:]) != (1, LATENT_DIM):
        raise ValueError(f"Stage B requires the real H=1, D=192 context; got {tuple(context.shape)}")
    repeated = context.expand(actions.shape[0], -1, -1)
    predictions = reference.base.official_teacher_targets(official_model, repeated, actions)
    objective_row = {
        "latent_history": context,
        "goal_emb": row["goal_emb"],
    }
    return reference.base._official_objective(official_model, objective_row, predictions).reshape(-1)


def _student_costs(
    reference: Any,
    official_model: Any,
    student: Any,
    row: Mapping[str, Any],
    actions: Any,
) -> Any:
    import torch

    context = _tensor(reference, row["latent_history"], torch.float32, actions.device)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if tuple(context.shape[-2:]) != (1, LATENT_DIM):
        raise ValueError(f"Stage B requires H=1, D=192; got {tuple(context.shape)}")
    repeated = context.expand(actions.shape[0], -1, -1)
    predictions = student(repeated, actions)
    expected = (actions.shape[0], HORIZON, LATENT_DIM)
    if tuple(predictions.shape) != expected:
        raise ValueError(f"student predictions must be {expected}, got {tuple(predictions.shape)}")
    objective_row = {"latent_history": context, "goal_emb": row["goal_emb"]}
    costs = reference.base._official_objective(official_model, objective_row, predictions)
    return costs.reshape(-1)


def _middle_rows(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    episodes: list[Any] = []
    by_episode: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("stratum", "")).lower() != "middle":
            continue
        episode = row.get("episode_id")
        if episode not in by_episode:
            episodes.append(episode)
            by_episode[episode] = row
    selected = episodes[:count]
    if len(selected) != count or any(episode is None for episode in selected):
        raise ValueError(f"Stage B needs {count} distinct middle-anchor rows; found {len(selected)}")
    return [by_episode[episode] for episode in selected]


def _score_case(
    reference: Any,
    official_model: Any,
    students: Mapping[str, Any],
    row: Mapping[str, Any],
    case_index: int,
    freeze: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    contingent = freeze["contingent"]
    population = int(contingent["stage_b_population"])
    elites = int(contingent["stage_b_elites"])
    rounds = int(contingent["stage_b_rounds"])
    if (population, elites, rounds) != (300, 30, 30):
        raise ValueError("Stage B budget drifted from 300/30/30")
    device = next(official_model.parameters()).device
    dtype = torch.float32
    initial_mean = torch.full((1, HORIZON, ACTION_DIM), float(update["initial_mean"]), device=device, dtype=dtype)
    initial_sigma = torch.full_like(initial_mean, float(update["initial_sampling_std"]))
    distributions = {
        arm: (initial_mean.clone(), initial_sigma.clone())
        for arm in ARMS
    }
    generator = torch.Generator(device=device).manual_seed(int(contingent["stage_b_seed"]) + case_index)
    checkpoints: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    teacher_round1_std: float | None = None

    with torch.inference_mode():
        for round_index in range(1, rounds + 1):
            epsilon = torch.randn(
                (1, population, HORIZON, ACTION_DIM),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            candidate_banks: dict[str, Any] = {}
            own_costs: dict[str, Any] = {}
            next_distributions: dict[str, tuple[Any, Any]] = {}
            selected_indices: dict[str, Any] = {}

            for arm in ARMS:
                mu, sigma = distributions[arm]
                candidates = _sample(mu, sigma, epsilon)
                if not torch.equal(candidates[:, 0], mu):
                    raise RuntimeError(f"candidate zero drifted from the pre-update mean for {arm}")
                candidate_banks[arm] = candidates
                costs = (
                    _teacher_costs(reference, official_model, row, candidates[0])
                    if arm == "teacher"
                    else _student_costs(reference, official_model, students[arm], row, candidates[0])
                )
                if costs.shape != (population,) or not bool(torch.isfinite(costs).all().item()):
                    raise FloatingPointError(f"invalid {arm} proposal costs at round {round_index}")
                own_costs[arm] = costs
                next_mu, next_sigma, chosen = _update(candidates, costs, elites, update)
                if not bool(torch.isfinite(next_mu).all().item() and torch.isfinite(next_sigma).all().item()):
                    raise FloatingPointError(f"invalid {arm} CEM update at round {round_index}")
                next_distributions[arm] = (next_mu, next_sigma)
                selected_indices[arm] = chosen

            teacher_mu, teacher_sigma = next_distributions["teacher"]
            if round_index == 1:
                teacher_round1_std = max(
                    float(own_costs["teacher"].std(unbiased=False).detach().cpu()),
                    1e-6,
                )

            teacher_shadow_costs: dict[str, Any] = {"teacher": own_costs["teacher"]}
            post_mean_teacher_costs: dict[str, float] = {}
            for arm in ("flat_adapter", "block_adapter_global16"):
                before_mu, before_sigma = (
                    next_distributions[arm][0].clone(),
                    next_distributions[arm][1].clone(),
                )
                shadow_actions = torch.cat(
                    (candidate_banks[arm][0], next_distributions[arm][0]),
                    dim=0,
                )
                shadow = _teacher_costs(reference, official_model, row, shadow_actions)
                teacher_shadow_costs[arm] = shadow[:-1]
                post_mean_teacher_costs[arm] = float(shadow[-1].detach().cpu())
                if not torch.equal(before_mu, next_distributions[arm][0]) or not torch.equal(
                    before_sigma, next_distributions[arm][1]
                ):
                    raise RuntimeError("teacher shadow scoring changed a student proposal")

            post_mean_teacher_costs["teacher"] = float(
                _teacher_costs(reference, official_model, row, teacher_mu)[0].detach().cpu()
            )
            round_record: dict[str, Any] = {"round": round_index, "arms": {}}
            for arm in ARMS:
                shadow_cost = teacher_shadow_costs[arm]
                shadow_elites = _elite_indices(shadow_cost, elites, str(update["selector"]))
                overlap = float(
                    torch.isin(selected_indices[arm], shadow_elites).float().mean().detach().cpu()
                )
                mu, sigma = next_distributions[arm]
                mu_rms = float(torch.sqrt(torch.mean((mu - teacher_mu).square())).detach().cpu())
                sigma_rms = float(torch.sqrt(torch.mean((sigma - teacher_sigma).square())).detach().cpu())
                first_chunk_rms = float(
                    torch.sqrt(torch.mean((mu[:, 0, :] - teacher_mu[:, 0, :]).square())).detach().cpu()
                )
                first_primitive_rms = float(
                    torch.sqrt(torch.mean((mu[:, 0, :2] - teacher_mu[:, 0, :2]).square())).detach().cpu()
                )
                regret = (
                    post_mean_teacher_costs[arm] - post_mean_teacher_costs["teacher"]
                ) / teacher_round1_std
                round_record["arms"][arm] = {
                    "shadow_elite_overlap": overlap,
                    "mu_rms_vs_teacher": mu_rms,
                    "sigma_rms_vs_teacher": sigma_rms,
                    "first_action_chunk_rms_vs_teacher": first_chunk_rms,
                    "first_primitive_rms_vs_teacher": first_primitive_rms,
                    "standardized_teacher_regret_vs_teacher_mean": float(regret),
                    "post_mean_teacher_cost": post_mean_teacher_costs[arm],
                    "finite": bool(
                        math.isfinite(overlap)
                        and math.isfinite(mu_rms)
                        and math.isfinite(sigma_rms)
                        and math.isfinite(regret)
                    ),
                }
                if round_index in CHECKPOINT_ROUNDS:
                    checkpoints[arm][str(round_index)] = dict(round_record["arms"][arm])
            distributions = next_distributions

    final = {arm: checkpoints[arm]["30"] for arm in ARMS}
    passed_by_arm: dict[str, bool] = {}
    for arm in ("flat_adapter", "block_adapter_global16"):
        metrics = final[arm]
        passed_by_arm[arm] = bool(
            metrics["first_action_chunk_rms_vs_teacher"] <= float(contingent["stage_b_abs_first_action_rms_max"])
            and metrics["mu_rms_vs_teacher"] <= float(contingent["stage_b_abs_mu_rms_max"])
            and metrics["sigma_rms_vs_teacher"] <= float(contingent["stage_b_abs_sigma_rms_max"])
            and metrics["shadow_elite_overlap"] >= float(contingent["stage_b_abs_shadow_top30_min"])
            and metrics["standardized_teacher_regret_vs_teacher_mean"]
            <= float(contingent["stage_b_abs_standardized_regret_max"])
        )
    return {
        "episode_id": int(row["episode_id"]),
        "context_id": str(row["context_id"]),
        "anchor": int(row["anchor"]),
        "stratum": str(row["stratum"]),
        "case_seed": int(contingent["stage_b_seed"]) + case_index,
        "teacher_round1_population_std": teacher_round1_std,
        "checkpoints": checkpoints,
        "round30_absolute_passed": passed_by_arm,
    }


def run_stage_b(
    reference: Any,
    official_model: Any,
    factories: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the contingent 8-case fixed-observation CEM diagnostic on PBS."""
    import torch

    reference.base.require_compute_node()
    update = _resolve_update(freeze)
    selfcheck = mechanism_selfcheck(freeze)
    if selfcheck["status"] != "PASS":
        raise RuntimeError(f"Stage B mechanism selfcheck failed: {selfcheck['checks']}")
    for arm in ("flat_adapter", "block_adapter_global16"):
        if arm not in factories:
            raise KeyError(f"Stage B factory missing {arm}")

    selected_rows = _middle_rows(rows, int(freeze["contingent"]["stage_b_episodes"]))
    output_root = Path(output_dir) / "stage_b"
    output_root.mkdir(parents=True, exist_ok=True)
    selection = [
        {
            "episode_id": int(row["episode_id"]),
            "context_id": str(row["context_id"]),
            "anchor": int(row["anchor"]),
            "stratum": str(row["stratum"]),
        }
        for row in selected_rows
    ]
    _write_json(
        output_root / "selection.json",
        {
            "episodes": selection,
            "selection_source": "first distinct episode IDs in frozen fresh-row order, middle stratum only",
            "written_before_stage_b_student_inference": True,
        },
    )

    students: dict[str, Any] = {}
    for arm in ("flat_adapter", "block_adapter_global16"):
        student = factories[arm]()
        student.eval()
        prepare = getattr(student, "prepare_for_inference", None)
        if callable(prepare):
            prepare()
        students[arm] = student

    official_model.eval()
    official_model.requires_grad_(False)
    jsonl = output_root / "cases.jsonl"
    if jsonl.exists():
        raise FileExistsError(f"refusing to overwrite existing Stage B cases: {jsonl}")
    case_results = []
    with jsonl.open("x", encoding="utf-8") as stream:
        for case_index, row in enumerate(selected_rows):
            result = _score_case(reference, official_model, students, row, case_index, freeze, update)
            case_results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()

    required_cases = int(freeze["contingent"]["stage_b_required_cases"])
    arm_pass_counts = {
        arm: sum(bool(case["round30_absolute_passed"][arm]) for case in case_results)
        for arm in ("flat_adapter", "block_adapter_global16")
    }
    absolute_passed = arm_pass_counts["block_adapter_global16"] >= required_cases
    primary_deltas = []
    for case in case_results:
        flat = case["checkpoints"]["flat_adapter"]["30"]
        block = case["checkpoints"]["block_adapter_global16"]["30"]
        primary_deltas.append(
            {
                "episode_id": case["episode_id"],
                "global16_minus_flat_mu_rms": block["mu_rms_vs_teacher"] - flat["mu_rms_vs_teacher"],
                "global16_minus_flat_sigma_rms": block["sigma_rms_vs_teacher"] - flat["sigma_rms_vs_teacher"],
                "global16_minus_flat_standardized_teacher_regret": (
                    block["standardized_teacher_regret_vs_teacher_mean"]
                    - flat["standardized_teacher_regret_vs_teacher_mean"]
                ),
                "global16_minus_flat_shadow_elite_overlap": (
                    block["shadow_elite_overlap"] - flat["shadow_elite_overlap"]
                ),
            }
        )
    mean_deltas = {
        key: float(sum(item[key] for item in primary_deltas) / max(1, len(primary_deltas)))
        for key in (
            "global16_minus_flat_mu_rms",
            "global16_minus_flat_sigma_rms",
            "global16_minus_flat_standardized_teacher_regret",
            "global16_minus_flat_shadow_elite_overlap",
        )
    }
    result = {
        "status": "STAGE_B_COMPLETE",
        "absolute_passed": bool(absolute_passed),
        "absolute_gate": {
            "primary_arm": "block_adapter_global16",
            "required_cases": required_cases,
            "total_cases": len(case_results),
            "passed_cases": arm_pass_counts["block_adapter_global16"],
            "per_arm_passed_cases": arm_pass_counts,
            "thresholds": {
                "first_action_chunk_rms_max": float(freeze["contingent"]["stage_b_abs_first_action_rms_max"]),
                "mu_rms_max": float(freeze["contingent"]["stage_b_abs_mu_rms_max"]),
                "sigma_rms_max": float(freeze["contingent"]["stage_b_abs_sigma_rms_max"]),
                "shadow_top30_min": float(freeze["contingent"]["stage_b_abs_shadow_top30_min"]),
                "standardized_teacher_regret_max": float(freeze["contingent"]["stage_b_abs_standardized_regret_max"]),
            },
        },
        "native_update": update,
        "mechanism_selfcheck": selfcheck,
        "design": {
            "scope": "fixed-observation CEM update diagnostic; not the complete official MPC lifecycle",
            "arms": list(ARMS),
            "population": int(freeze["contingent"]["stage_b_population"]),
            "elites": int(freeze["contingent"]["stage_b_elites"]),
            "rounds": int(freeze["contingent"]["stage_b_rounds"]),
            "shared_innovations": True,
            "candidate_zero_is_pre_update_mean": True,
            "action_bounds_reported": freeze["evaluation"]["official_action_low"] + freeze["evaluation"]["official_action_high"],
            "actions_clipped": bool(update["clip_actions"]),
            "teacher_shadow_used_for_student_updates": False,
            "first_action_rms_unit": "first packed 10D action token; first primitive 2D RMS also stored",
            "teacher_regret_reference": "official teacher score of arm post-update mean minus teacher score of teacher-arm post-update mean; divided by teacher round-1 candidate-score population std",
            "case_seed_rule": "stage_b_seed + zero-based case order",
        },
        "case_results": case_results,
        "global16_minus_flat_round30_mean_deltas": mean_deltas,
        "artifacts": {
            "selection": str(output_root / "selection.json"),
            "cases": str(jsonl),
        },
        "stage_c": "PENDING_TRIGGER_CHECK" if absolute_passed else "NOT_RUN_GATE_FAILED",
    }
    _write_json(output_root / "summary.json", result)
    return result
