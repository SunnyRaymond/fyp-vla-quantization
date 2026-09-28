#!/usr/bin/env python3
"""Read-only fresh-bank cost-error decomposition for the frozen LeWM students."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
PREVIOUS_RUNNER = HERE.parent / "terminal-response-loss" / "runner.py"
SCHEMA = "lewm-recurrent-student.fresh-cost-error-diagnostic-freeze"
ARMS = ("balanced_base", "terminal_response")
ORACLES = ("remove_common", "remove_contrast", "correct_parallel_gain", "remove_orthogonal_response")
MATCHED_METRICS = ("spearman", "top30_recall", "standardized_elite_regret", "normalized_response_mse", "teacher_terminal_contrast_mse")
CONTINUOUS_TOLERANCE = 1e-4
ENERGY_FLOOR = 1e-6


def load_previous():
    spec = importlib.util.spec_from_file_location("terminal_response_source", PREVIOUS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("terminal response source runner unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_sources(freeze: Mapping[str, Any], previous: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if freeze.get("schema") != SCHEMA or freeze.get("status") != "frozen_before_execution":
        raise ValueError("diagnostic freeze identity mismatch")
    source = freeze["source"]
    if source["previous_job_id"] != "25549480.pbs101" or source["episode_ids_in_order"] != [10569, 7242, 13109, 2193, 3218, 2289, 8316, 1085]:
        raise ValueError("previous job or episode identities drifted")
    if source["action_prefix_seeds"] != [20301007, 20301008] or source["anchors"] != ["early", "middle", "late"]:
        raise ValueError("fresh bank anchors/seeds drifted")
    if (source["contexts"], source["banks"], source["candidates_per_bank"]) != (24, 48, 300):
        raise ValueError("fresh coverage drifted")
    if freeze["execution"]["training"] is not False or freeze["execution"]["official_cem"] != "NOT_RUN_BY_SCOPE":
        raise ValueError("diagnostic crossed a frozen scope boundary")
    prior_freeze = previous.load_json(Path(source["previous_freeze"]))
    previous.validate_freeze(prior_freeze)
    selection = previous.load_json(Path(source["selection"]))
    summary = previous.load_json(Path(source["summary"]))
    if summary.get("pbs_job_id") != source["previous_job_id"] or summary.get("status") != "PREDICTOR_LEVEL_COMPLETE":
        raise ValueError("previous summary identity/status mismatch")
    ids = source["episode_ids_in_order"]
    if selection.get("ordered_episode_ids") != ids or selection.get("selection", {}).get("fresh_episode_ids") != ids or summary.get("fresh_selection", {}).get("fresh_episode_ids") != ids:
        raise ValueError("previous selection and summary disagree")
    if summary["training"]["checkpoints"] != source["checkpoints"]:
        raise ValueError("checkpoint paths differ from the previous result")
    if Path(source["previous_dir"]).resolve() != Path(source["summary"]).resolve().parent:
        raise ValueError("previous artifact directory mismatch")
    if prior_freeze["evaluation"]["action_prefix_seeds"] != source["action_prefix_seeds"]:
        raise ValueError("previous freeze action-prefix seeds differ")
    return prior_freeze, selection, summary


def prior_block_maps(summary: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    for arm, branch in (("balanced_base", "control"), ("terminal_response", "terminal_response")):
        blocks = summary["fresh_evaluation"][branch]["blocks"]
        mapping = {str(item["pairing_key"]): item for item in blocks}
        if len(blocks) != 48 or len(mapping) != 48:
            raise ValueError(f"previous {arm} has duplicate or missing banks")
        maps[arm] = mapping
    if set(maps[ARMS[0]]) != set(maps[ARMS[1]]):
        raise ValueError("previous arm banks do not pair")
    return maps


def align_block(current: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, float]:
    for key in ("episode_id", "context_id", "stratum", "anchor", "action_prefix_seed", "pairing_key"):
        if current[key] != prior[key]:
            raise ValueError(f"fresh bank identity drifted: {key}")
    errors: dict[str, float] = {}
    for metric in MATCHED_METRICS:
        gap = abs(float(current[metric]) - float(prior[metric]))
        tolerance = 0.0 if metric == "top30_recall" else CONTINUOUS_TOLERANCE
        if not math.isfinite(gap) or gap > tolerance:
            raise ValueError(f"prior bank metric mismatch for {current['pairing_key']} {metric}: {gap}")
        errors[metric] = gap
    return errors


def ranking_metrics(cost: Any, teacher_cost: Any, torch: Any) -> dict[str, float]:
    teacher_order = torch.argsort(teacher_cost)
    scorer_order = torch.argsort(cost)
    teacher_top = teacher_order[:30]
    scorer_top = scorer_order[:30]
    scale = teacher_cost.std(unbiased=False).clamp_min(1e-6)
    upper, lower = teacher_order[20:30], teacher_order[30:40]
    return {
        "top30_recall": float(torch.isin(teacher_top, scorer_top).sum().item()) / 30,
        "standardized_elite_regret": float(((teacher_cost[scorer_top].mean() - teacher_cost[teacher_top].mean()) / scale).item()),
        "boundary_inversion_20_40": float((cost[upper, None] >= cost[None, lower]).to(torch.float32).mean().item()),
    }


def decompose_bank(row: Mapping[str, Any], block: int, seed: int, prediction: Any, student_cost: Any, torch: Any) -> dict[str, Any]:
    target = row["teacher_targets"][block].to(prediction.device)
    teacher_cost = row["teacher_objective"][block].to(prediction.device)
    zt = target[:, -1, :].to(torch.float64)
    zs = prediction[:, -1, :].to(torch.float64)
    goal = row["goal_emb"].to(device=prediction.device, dtype=torch.float64).reshape(1, 192)
    residual = zt - goal
    error = zs - zt
    common = error.mean(dim=0, keepdim=True)
    contrast = error - common
    teacher_response = zt - zt.mean(dim=0, keepdim=True)
    student_response = zs - zs.mean(dim=0, keepdim=True)
    teacher_energy_sum = teacher_response.square().sum()
    projection_gain = (teacher_response * student_response).sum() / teacher_energy_sum.clamp_min(1e-20)
    orthogonal = student_response - projection_gain * teacher_response
    parallel_error = (projection_gain - 1) * teacher_response
    if float((contrast - parallel_error - orthogonal).abs().max()) > 1e-8:
        raise RuntimeError("response projection did not close")
    teacher_energy = teacher_response.square().mean()
    if abs(float((contrast.square().sum() - parallel_error.square().sum() - orthogonal.square().sum()).item())) > 1e-7:
        raise RuntimeError("response error energy did not close")
    error_energy = error.square().sum(dim=-1).mean()
    if abs(float((error_energy - common.square().sum() - contrast.square().sum(dim=-1).mean()).item())) > 1e-7:
        raise RuntimeError("latent error energy did not close")

    teacher_manual = residual.square().sum(dim=-1)
    student_manual = (zs - goal).square().sum(dim=-1)
    if not torch.allclose(teacher_manual, teacher_cost.to(torch.float64), rtol=1e-5, atol=1e-4):
        raise RuntimeError("official teacher criterion differs from terminal squared distance")
    if not torch.allclose(student_manual, student_cost.to(torch.float64), rtol=1e-5, atol=1e-4):
        raise RuntimeError("official student criterion differs from terminal squared distance")
    goal_common = 2 * (residual * common).sum(dim=-1)
    goal_contrast = 2 * (residual * contrast).sum(dim=-1)
    cross = 2 * (common * contrast).sum(dim=-1)
    contrast_norm = contrast.square().sum(dim=-1)
    common_score = goal_common
    contrast_score = goal_contrast + cross + contrast_norm
    score_error = student_manual - teacher_manual
    if float((score_error - common_score - common.square().sum() - contrast_score).abs().max()) > 1e-7:
        raise RuntimeError("goal-cost error decomposition did not close")
    cost_scale = teacher_cost.std(unbiased=False).to(torch.float64).clamp_min(1e-6)

    def centered_rms(values: Any) -> float:
        return float(((values - values.mean()).square().mean().sqrt() / cost_scale).item())

    def cost_from_latent(values: Any) -> Any:
        return (values - goal).square().sum(dim=-1).to(torch.float32)

    counterfactual_latents = {
        "remove_common": zs - common,
        "remove_contrast": zs - contrast,
        "correct_parallel_gain": zs - parallel_error,
        "remove_orthogonal_response": zs - orthogonal,
    }
    original = ranking_metrics(student_cost, teacher_cost, torch)
    counterfactuals = {
        name: ranking_metrics(cost_from_latent(latent), teacher_cost, torch)
        for name, latent in counterfactual_latents.items()
    }
    response_denominator = teacher_energy_sum.clamp_min(zt.numel() * ENERGY_FLOOR)
    projection = {
        "teacher_contrast_mse": float(teacher_energy.item()),
        "teacher_energy_floor_active": bool(teacher_energy < ENERGY_FLOOR),
        "gain_along_teacher_response": float(projection_gain.item()),
        "cosine": float(((teacher_response * student_response).sum() / (teacher_energy_sum.sqrt() * student_response.square().sum().sqrt()).clamp_min(1e-20)).item()),
        "scale": float((student_response.square().sum().sqrt() / teacher_energy_sum.sqrt().clamp_min(1e-20)).item()),
        "parallel_error_relative_energy": float((parallel_error.square().sum() / response_denominator).item()),
        "orthogonal_error_relative_energy": float((orthogonal.square().sum() / response_denominator).item()),
        "total_response_error_relative_energy": float((contrast.square().sum() / response_denominator).item()),
    }
    if not all(math.isfinite(float(value)) for value in projection.values()):
        raise FloatingPointError("non-finite response projection")
    score_terms = {
        "common_centered_rms_teacher_std": centered_rms(common_score),
        "contrast_centered_rms_teacher_std": centered_rms(contrast_score),
        "goal_common_centered_rms_teacher_std": centered_rms(goal_common),
        "goal_contrast_centered_rms_teacher_std": centered_rms(goal_contrast),
        "cross_centered_rms_teacher_std": centered_rms(cross),
        "contrast_norm_centered_rms_teacher_std": centered_rms(contrast_norm),
        "total_centered_rms_teacher_std": centered_rms(score_error),
    }
    teacher_order = torch.argsort(teacher_cost)
    return {
        "episode_id": int(row["episode_id"]),
        "context_id": str(row["context_id"]),
        "stratum": str(row["stratum"]),
        "anchor": int(row["anchor"]),
        "action_prefix_seed": int(seed),
        "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}",
        "contrast_latent_error_energy_fraction": float((contrast.square().sum(dim=-1).mean() / error_energy.clamp_min(1e-20)).item()),
        "teacher_boundary_30_31_gap_teacher_std": float(((teacher_cost[teacher_order[30]] - teacher_cost[teacher_order[29]]) / cost_scale).item()),
        "response_geometry": projection,
        "score_terms": score_terms,
        "original": original,
        "counterfactuals": counterfactuals,
        "failed_absolute_bank": False,
    }


def value_at(row: Mapping[str, Any], dotted: str) -> float:
    value: Any = row
    for key in dotted.split("."):
        value = value[key]
    return float(value)


SUMMARIZE_METRICS = (
    "contrast_latent_error_energy_fraction",
    "teacher_boundary_30_31_gap_teacher_std",
    "response_geometry.gain_along_teacher_response",
    "response_geometry.cosine",
    "response_geometry.scale",
    "response_geometry.parallel_error_relative_energy",
    "response_geometry.orthogonal_error_relative_energy",
    "response_geometry.total_response_error_relative_energy",
    "score_terms.common_centered_rms_teacher_std",
    "score_terms.contrast_centered_rms_teacher_std",
    "score_terms.goal_common_centered_rms_teacher_std",
    "score_terms.goal_contrast_centered_rms_teacher_std",
    "score_terms.cross_centered_rms_teacher_std",
    "score_terms.contrast_norm_centered_rms_teacher_std",
    "score_terms.total_centered_rms_teacher_std",
    "original.standardized_elite_regret",
    "original.top30_recall",
    "original.boundary_inversion_20_40",
    *(f"counterfactuals.{name}.standardized_elite_regret" for name in ORACLES),
    *(f"counterfactuals.{name}.top30_recall" for name in ORACLES),
    *(f"counterfactuals.{name}.boundary_inversion_20_40" for name in ORACLES),
)


def nested_summary(rows: list[Mapping[str, Any]], metrics: tuple[str, ...] = SUMMARIZE_METRICS) -> dict[str, Any]:
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_id"])].append(row)
    if len(rows) != 48 or len(by_episode) != 8 or any(len(items) != 6 for items in by_episode.values()):
        raise ValueError("diagnostic requires 48 banks nested in eight episodes")
    episode_values = {
        str(episode): {metric: float(statistics.median(value_at(row, metric) for row in items)) for metric in metrics}
        for episode, items in sorted(by_episode.items())
    }
    overall = {metric: float(statistics.median(values[metric] for values in episode_values.values())) for metric in metrics}
    return {"episode_medians": episode_values, "overall_episode_median": overall}


def matched_summary(control: list[Mapping[str, Any]], treatment: list[Mapping[str, Any]]) -> dict[str, Any]:
    left = {str(row["pairing_key"]): row for row in control}
    right = {str(row["pairing_key"]): row for row in treatment}
    if len(left) != 48 or len(right) != 48 or set(left) != set(right):
        raise ValueError("decomposition arms do not pair")
    deltas = [
        {"episode_id": int(left[key]["episode_id"]), "pairing_key": key,
         **{metric: value_at(right[key], metric) - value_at(left[key], metric) for metric in SUMMARIZE_METRICS}}
        for key in left
    ]
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in deltas:
        by_episode[row["episode_id"]].append(row)
    episode_values = {
        str(episode): {metric: float(statistics.median(row[metric] for row in items)) for metric in SUMMARIZE_METRICS}
        for episode, items in sorted(by_episode.items())
    }
    return {
        "paired_unit": "episode after median of six matched bank deltas",
        "episode_median_deltas": episode_values,
        "overall_episode_median_delta": {metric: float(statistics.median(values[metric] for values in episode_values.values())) for metric in SUMMARIZE_METRICS},
    }


def oracle_benefits(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_id"])].append(row)
    result: dict[str, Any] = {}
    for oracle in ORACLES:
        comparisons = {
            "regret_reduction": ("standardized_elite_regret", -1),
            "top30_recall_increase": ("top30_recall", 1),
            "boundary_inversion_reduction": ("boundary_inversion_20_40", -1),
        }
        benefit = {
            str(episode): {
                label: float(statistics.median(
                    sign * (value_at(row, f"counterfactuals.{oracle}.{metric}") - value_at(row, f"original.{metric}"))
                    for row in items
                ))
                for label, (metric, sign) in comparisons.items()
            }
            for episode, items in sorted(by_episode.items())
        }
        result[oracle] = {
            "per_episode": benefit,
            "episode_median": {label: float(statistics.median(values[label] for values in benefit.values())) for label in comparisons},
            "episodes_with_positive_benefit": {label: sum(values[label] > 0 for values in benefit.values()) for label in comparisons},
        }
    return result


def worst_bank_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    extrema = {
        "maximum_regret": ("original.standardized_elite_regret", max),
        "minimum_top30_recall": ("original.top30_recall", min),
        "maximum_boundary_inversion": ("original.boundary_inversion_20_40", max),
        "maximum_common_score_rms": ("score_terms.common_centered_rms_teacher_std", max),
        "maximum_contrast_score_rms": ("score_terms.contrast_centered_rms_teacher_std", max),
        "maximum_parallel_response_error": ("response_geometry.parallel_error_relative_energy", max),
        "maximum_orthogonal_response_error": ("response_geometry.orthogonal_error_relative_energy", max),
    }
    return {
        label: {"pairing_key": selected["pairing_key"], "episode_id": selected["episode_id"], "stratum": selected["stratum"], "value": value_at(selected, metric)}
        for label, (metric, choice) in extrema.items()
        for selected in (choice(rows, key=lambda row: value_at(row, metric)),)
    }


def observed_subset_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"bank_count": 0, "episode_count": 0, "episode_median_metrics": {}, "oracle_paired_effects": {}}
    metrics = (
        "response_geometry.parallel_error_relative_energy",
        "response_geometry.orthogonal_error_relative_energy",
        "score_terms.common_centered_rms_teacher_std",
        "score_terms.contrast_centered_rms_teacher_std",
        "original.standardized_elite_regret",
        "original.top30_recall",
    )
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_id"])].append(row)
    summaries = {
        metric: float(statistics.median(
            statistics.median(value_at(row, metric) for row in items)
            for items in by_episode.values()
        ))
        for metric in metrics
    }
    return {"bank_count": len(rows), "episode_count": len(by_episode), "episode_median_metrics": summaries, "oracle_paired_effects": oracle_benefits(rows)}


def run(freeze_path: Path, output_dir: Path) -> dict[str, Any]:
    previous = load_previous()
    host = previous.require_compute_node()  # before loading models, HDF5, checkpoints, or banks
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    prior_freeze, selection, prior_summary = require_sources(freeze, previous)
    source = freeze["source"]
    maps = prior_block_maps(prior_summary)
    reference = previous.load_reference_modules()
    import torch

    official_model = reference.base.load_official_checkpoint(Path(prior_freeze["source"]["stablewm_home"]))
    official_model.requires_grad_(False)
    fresh_rows, fresh_meta = previous.build_fresh_rows(
        reference, official_model, prior_freeze,
        Path(prior_freeze["source"]["dataset_path"]), source["episode_ids_in_order"],
    )
    if fresh_meta != prior_summary["fresh_evaluation"]["metadata"]:
        raise ValueError("reconstructed context metadata differs from the previous result")
    results: dict[str, list[dict[str, Any]]] = {}
    max_alignment = {metric: 0.0 for metric in MATCHED_METRICS}
    for arm in ARMS:
        checkpoint = torch.load(source["checkpoints"][arm], map_location="cpu", weights_only=False)
        if checkpoint.get("arm") != arm or checkpoint.get("step") != 3000 or checkpoint.get("schema") != prior_freeze["schema"] + ".checkpoint":
            raise ValueError(f"{arm} checkpoint identity mismatch")
        provenance = checkpoint.get("provenance", {})
        if provenance.get("freeze") != source["previous_freeze"] or provenance.get("training_rows_source_job") != "25213164.pbs101" or provenance.get("teacher_targets_reused_from_cached_rows") is not True:
            raise ValueError(f"{arm} checkpoint provenance mismatch")
        if sum(value.numel() for value in checkpoint["state_dict"].values()) != 775872:
            raise ValueError(f"{arm} checkpoint parameter count drifted")
        student = reference.instantiate_student("balanced_base").to("cuda")
        reference.load_full_state(student, checkpoint["state_dict"])
        student.eval()
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for fresh in fresh_rows:
                for block, seed in enumerate(source["action_prefix_seeds"]):
                    current = previous.evaluate_block(reference, official_model, student, fresh, block, int(seed))
                    key = str(current["pairing_key"])
                    if key not in maps[arm]:
                        raise ValueError(f"regenerated bank is absent from prior result: {key}")
                    errors = align_block(current, maps[arm][key])
                    for metric, gap in errors.items():
                        max_alignment[metric] = max(max_alignment[metric], gap)
                    actions = fresh["future_actions"][block].to("cuda")
                    context = fresh["latent_history"].to("cuda").expand(300, -1, -1)
                    prediction = student(context, actions)
                    cost = reference.base._official_objective(official_model, fresh, prediction)
                    decomposed = decompose_bank(fresh, block, int(seed), prediction, cost, torch)
                    if decomposed["pairing_key"] != key:
                        raise ValueError("decomposition bank key disagrees with prior evaluation")
                    prior_treatment = maps["terminal_response"][key]
                    decomposed["failed_absolute_bank"] = bool(prior_treatment["spearman"] < 0.80 or prior_treatment["top30_recall"] < 0.50)
                    rows.append(decomposed)
        if len(rows) != 48:
            raise ValueError(f"{arm} decomposition must contain 48 banks")
        results[arm] = rows
        del student
        torch.cuda.empty_cache()

    all_keys = {row["pairing_key"] for row in results["balanced_base"]}
    if all_keys != set(maps["balanced_base"]):
        raise ValueError("diagnostic did not cover every previous bank")
    failed = [row["pairing_key"] for row in results["terminal_response"] if row["failed_absolute_bank"]]
    output = {
        "schema": SCHEMA + ".result",
        "status": "READ_ONLY_PREDICTOR_DIAGNOSTIC_COMPLETE",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "previous_job_id": source["previous_job_id"],
        "identity_alignment": {"banks_per_arm": 48, "max_abs_metric_differences": max_alignment, "selection_episode_ids": selection["ordered_episode_ids"], "candidate_bank_rows_saved": False},
        "failed_absolute_bank_keys": failed,
        "failed_absolute_bank_count": len(failed),
        "arms": {arm: {"bank_rows": results[arm], "summary": nested_summary(results[arm]), "oracle_paired_effects": oracle_benefits(results[arm]), "worst_banks": worst_bank_rows(results[arm]), "response_floor_active_banks": sum(row["response_geometry"]["teacher_energy_floor_active"] for row in results[arm])} for arm in ARMS},
        "treatment_observed_subgroups": {
            "failed_absolute_bank": observed_subset_summary([row for row in results["terminal_response"] if row["failed_absolute_bank"]]),
            "other_bank": observed_subset_summary([row for row in results["terminal_response"] if not row["failed_absolute_bank"]]),
            "selection_is_not_independent_validation": True,
        },
        "matched_treatment_minus_control": matched_summary(results["balanced_base"], results["terminal_response"]),
        "scope": {"training": "NOT_RUN_BY_SCOPE", "official_cem": "NOT_RUN_BY_SCOPE", "planner": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": freeze["claim_boundary"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "fresh_cost_error_diagnostic_summary.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite previous result: {target}")
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {target}; 48 fresh banks x two arms; previous-bank metrics aligned")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.freeze, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
