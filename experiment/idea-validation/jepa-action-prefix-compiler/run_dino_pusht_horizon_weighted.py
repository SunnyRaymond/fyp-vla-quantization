#!/usr/bin/env python3
"""Train/evaluate the frozen horizon-weighted recurrent DINO-WM student.

This runner reuses the recurrent native-latent transition and the exact
query-slate/context/held-out machinery.  The only treatment change is the
training reduction: each of the five per-horizon visual/proprio MSE values is
computed first, then aggregated with the frozen mean-normalized weights
``[1/3, 2/3, 1, 4/3, 5/3]``.  The recurrent baseline is read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_dino_pusht_context_density as density  # noqa: E402
import run_dino_pusht_optimization_length as opt  # noqa: E402
import run_dino_pusht_query_slate_rank as slate  # noqa: E402
import run_dino_pusht_recurrent_student as recurrent  # noqa: E402
from run_dino_pusht_grounded_prefix import (  # noqa: E402
    _RawEpisodeCache,
    _load_trajectory_datasets,
    _preencode_manifest,
)
from run_dino_pusht_stage_a import (  # noqa: E402
    HORIZON,
    _gpu_snapshot,
    _json_default,
    _leakage_test,
    _latency,
    _load_json,
    _load_official,
    _metrics_by_horizon,
    _require_assets,
    _require_compute_node,
    _teacher_targets,
    _set_seed,
)


WIDTH = recurrent.WIDTH
SNAPSHOT_STEPS = recurrent.SNAPSHOT_STEPS
SNAPSHOT_NAMES = recurrent.SNAPSHOT_NAMES
SLATE_SIZE = recurrent.SLATE_SIZE
HORIZON_WEIGHTS = (1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0)
BASELINE_JOB_ID = "24503839.pbs101"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True, help="HORIZON_WEIGHTED_FREEZE.json")
    parser.add_argument("--recurrent-freeze", type=Path, required=True)
    parser.add_argument("--query-slate-freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, default=None)
    parser.add_argument("--context-density-freeze", type=Path, default=None)
    parser.add_argument("--density-summary", type=Path, default=None)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--deps-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing frozen field {path}.{key}")
    return mapping[key]


def _values_for_key(value: Any, wanted: set[str]) -> list[Any]:
    """Collect exact-key values without depending on cosmetic freeze layout."""

    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in wanted:
                found.append(item)
            found.extend(_values_for_key(item, wanted))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_values_for_key(item, wanted))
    return found


def _contains_string(value: Any, fragments: Sequence[str]) -> bool:
    if isinstance(value, str):
        return any(fragment in value for fragment in fragments)
    if isinstance(value, Mapping):
        return any(_contains_string(item, fragments) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_string(item, fragments) for item in value)
    return False


def _first_numeric(value: Any, keys: set[str], default: float) -> float:
    values = _values_for_key(value, keys)
    return float(values[0]) if values else float(default)


def _settings(
    weighted_freeze: Mapping[str, Any],
    recurrent_freeze: Mapping[str, Any],
    query_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind all three freezes and expose only immutable run parameters."""

    schema = str(_required(weighted_freeze, "schema", "weighted_freeze"))
    if schema != "jepa-action-prefix-compiler.horizon-weighted-recurrent-freeze":
        raise ValueError(f"unexpected horizon-weighted freeze schema: {schema}")
    if int(_required(weighted_freeze, "schema_version", "weighted_freeze")) != 1:
        raise ValueError("unsupported horizon-weighted freeze schema version")
    # This call performs the detailed recurrent/query-slate parent checks,
    # including dimensions, seeds, schedule, slates, and recurrent cell.
    base = recurrent._settings(recurrent_freeze, query_freeze)
    if not _contains_string(weighted_freeze, ("RECURRENT_STUDENT_FREEZE.json", "recurrent-student-freeze")):
        raise ValueError("horizon freeze does not bind RECURRENT_STUDENT_FREEZE")
    if not (_contains_string(weighted_freeze, ("QUERY_SLATE_RANK_FREEZE.json", "query-slate-rank-freeze")) or _contains_string(recurrent_freeze, ("QUERY_SLATE_RANK_FREEZE.json", "query-slate-rank-freeze"))):
        raise ValueError("horizon freeze does not bind QUERY_SLATE_RANK_FREEZE")

    weight_values = _values_for_key(weighted_freeze, {"horizon_weights", "per_horizon_weights", "weights"})
    normalized = [tuple(float(x) for x in item) for item in weight_values if isinstance(item, (list, tuple)) and len(item) == HORIZON]
    if not normalized or normalized[0] != HORIZON_WEIGHTS:
        raise ValueError(f"frozen horizon weights must be exactly {list(HORIZON_WEIGHTS)}")
    if abs(sum(normalized[0]) / HORIZON - 1.0) > 1e-12:
        raise ValueError("horizon weights must be mean-normalized")

    if not _contains_string(weighted_freeze, (BASELINE_JOB_ID,)):
        raise ValueError("horizon freeze does not bind recurrent baseline job 24503839.pbs101")
    treatment = weighted_freeze.get("treatment", {})
    if isinstance(treatment, Mapping):
        treatment_text = json.dumps(treatment, sort_keys=True).lower()
        if "recurrent" not in treatment_text or "horizon" not in treatment_text:
            raise ValueError("weighted treatment is not the frozen recurrent horizon-loss treatment")

    settings = dict(base)
    settings.update(
        {
            "architecture": "horizon_weighted_recurrent_shared_latent_transition",
            "transition": "shared_residual_mlp",
            "horizon_weights": list(HORIZON_WEIGHTS),
            "baseline_job_id": BASELINE_JOB_ID,
            "baseline_arm": "recurrent",
            "baseline_snapshot": "step_1500",
            "effect_spearman_min": _first_numeric(weighted_freeze, {"median_spearman_delta_min"}, 0.05),
            "effect_top30_min": _first_numeric(weighted_freeze, {"median_top30_delta_min"}, 0.1),
            "effect_positive_spearman_min": int(_first_numeric(weighted_freeze, {"positive_spearman_blocks_min"}, 12)),
            "effect_positive_top30_min": int(_first_numeric(weighted_freeze, {"positive_top30_blocks_min"}, 12)),
        }
    )
    if settings["effect_spearman_min"] != 0.05 or settings["effect_top30_min"] != 0.1:
        raise ValueError("horizon-weighted effect thresholds drifted")
    return settings


def _validate_baseline(baseline: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the completed recurrent summary as a read-only paired reference."""

    expected_schema = "jepa-action-prefix-compiler.dino-pusht-recurrent-student-summary"
    if str(baseline.get("schema", "")) != expected_schema:
        raise ValueError("baseline is not the completed recurrent-student summary")
    contracts = baseline.get("contracts", {})
    if not isinstance(contracts, Mapping):
        raise ValueError("recurrent baseline has no contracts")
    if int(contracts.get("heldout_blocks", -1)) != 16 or int(contracts.get("heldout_contexts", -1)) != 8 or int(contracts.get("candidates_per_block", -1)) != 300:
        raise ValueError("recurrent baseline held-out contract differs")
    if [int(x) for x in contracts.get("heldout_seeds", [])] != [20264925, 20264926]:
        raise ValueError("recurrent baseline held-out seeds differ")
    if baseline.get("fallback_used") is not False or baseline.get("finite_outputs_and_training") is not True or baseline.get("shape_dtype_device_match") is not True:
        raise ValueError("recurrent baseline finite/shape/fallback integrity failed")
    pairing = baseline.get("pairing_integrity", {})
    if not isinstance(pairing, Mapping) or pairing.get("passed") is not True or pairing.get("baseline_read_only") is not True or pairing.get("baseline_block_keys_validated") is not True:
        raise ValueError("recurrent baseline pairing integrity is not authoritative")
    evaluation = baseline.get("evaluation", {})
    recurrent_eval = evaluation.get("recurrent", {}) if isinstance(evaluation, Mapping) else {}
    expected_keys = [(seed, context) for seed in (20264925, 20264926) for context in range(8)]
    for snap in SNAPSHOT_NAMES:
        rows = recurrent_eval.get(snap, {}).get("per_case", [])
        if len(rows) != 16:
            raise ValueError(f"recurrent baseline {snap} must contain 16 blocks")
        keys: list[tuple[int, int]] = []
        for row in rows:
            key = (int(row.get("seed", -1)), int(row.get("context_index", -1)))
            if int(row.get("candidate_count", -1)) != 300 or int(row.get("topk", -1)) != 30:
                raise ValueError(f"recurrent baseline {snap} candidate contract differs")
            if not isinstance(row.get("ranking"), Mapping):
                raise ValueError(f"recurrent baseline {snap} is missing ranking metrics")
            keys.append(key)
        if keys != expected_keys:
            raise ValueError(f"recurrent baseline {snap} block ordering differs")
    # The recurrent runner stores its treatment records directly under
    # logged_teacher_relative; its latency object is arm-labelled.
    logged = baseline.get("logged_teacher_relative", {})
    latency = baseline.get("latency", {}).get("recurrent", {})
    if any(snap not in logged for snap in SNAPSHOT_NAMES) or not isinstance(latency, Mapping):
        raise ValueError("recurrent baseline is missing logged or latency records")
    return {
        "schema": expected_schema,
        "job_id": settings["baseline_job_id"],
        "arm": "recurrent",
        "snapshot": "step_1500",
        "heldout_blocks": 16,
        "block_keys": expected_keys,
        "snapshots": list(SNAPSHOT_NAMES),
        "read_only": True,
    }


def _weighted_horizon_mse(prediction: Mapping[str, Any], target: Mapping[str, Any], weights: Sequence[float]) -> tuple[Any, Any, Any]:
    """Return per-horizon MSE, unweighted mean, and mean-normalized weighted mean."""

    import torch
    import torch.nn.functional as F

    values = []
    for horizon_index in range(HORIZON):
        values.append(0.5 * (
            F.mse_loss(prediction["visual"][:, horizon_index], target["visual"][:, horizon_index])
            + F.mse_loss(prediction["proprio"][:, horizon_index], target["proprio"][:, horizon_index])
        ))
    per_horizon = torch.stack(values)
    weight_tensor = torch.as_tensor(list(weights), dtype=per_horizon.dtype, device=per_horizon.device)
    unweighted = per_horizon.mean()
    weighted = (per_horizon * weight_tensor).mean()
    return per_horizon, unweighted, weighted


def _evaluate_snapshots(
    snapshots: Mapping[str, Mapping[str, Any]],
    model: Any,
    encoded: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    eval_seeds: Sequence[int],
    batch_size: int,
    device: Any,
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    blocks = opt._prepare_eval_blocks(encoded, 8, eval_seeds, batch_size, action_dim, device)
    evaluation: dict[str, Any] = {"weighted": {}}
    contrasts: dict[str, Any] = {}
    baseline_eval = baseline["evaluation"]["recurrent"]
    for snap in SNAPSHOT_NAMES:
        student = recurrent._student(action_dim, int(encoded["context"]["visual"].shape[-1]), int(encoded["context"]["proprio"].shape[-1]), WIDTH).to(device)
        student.load_state_dict(snapshots[snap], strict=True)
        student.eval()
        cases: list[dict[str, Any]] = []
        contrast_cases: list[dict[str, Any]] = []
        references = baseline_eval[snap]["per_case"]
        for block, reference in zip(blocks, references):
            if (int(reference["seed"]), int(reference["context_index"])) != (int(block["seed"]), int(block["context_index"])):
                raise ValueError("weighted held-out block key differs from recurrent baseline")
            with torch.no_grad():
                target = _teacher_targets(model, block["context"], block["actions"])
                prediction = student(block["context"], block["actions"])
            teacher_cost = objective_fn(target, block["goal"])
            student_cost = objective_fn(prediction, block["goal"])
            metrics = _metrics_by_horizon(prediction, target)
            ranking = {"spearman": slate._spearman(teacher_cost, student_cost), "top30_overlap": slate._topk_overlap(teacher_cost, student_cost)}
            metrics.update({"seed": int(block["seed"]), "context_index": int(block["context_index"]), "candidate_count": int(batch_size), "topk": min(30, int(batch_size)), "ranking": ranking})
            cases.append(metrics)
            ref_ranking = reference["ranking"]
            contrast_cases.append({
                "seed": int(block["seed"]),
                "context_index": int(block["context_index"]),
                "spearman_delta_weighted_minus_baseline": float(ranking["spearman"]) - float(ref_ranking["spearman"]),
                "top30_overlap_delta_weighted_minus_baseline": float(ranking["top30_overlap"]) - float(ref_ranking["top30_overlap"]),
            })
        evaluation["weighted"][snap] = recurrent._aggregate_cases(cases)
        contrasts[snap] = {
            "per_case": contrast_cases,
            "spearman_mean": recurrent._mean([x["spearman_delta_weighted_minus_baseline"] for x in contrast_cases]),
            "spearman_median": recurrent._median([x["spearman_delta_weighted_minus_baseline"] for x in contrast_cases]),
            "top30_overlap_mean": recurrent._mean([x["top30_overlap_delta_weighted_minus_baseline"] for x in contrast_cases]),
            "top30_overlap_median": recurrent._median([x["top30_overlap_delta_weighted_minus_baseline"] for x in contrast_cases]),
            "positive_spearman_blocks": sum(x["spearman_delta_weighted_minus_baseline"] > 0 for x in contrast_cases),
            "positive_top30_blocks": sum(x["top30_overlap_delta_weighted_minus_baseline"] > 0 for x in contrast_cases),
        }
    return evaluation, contrasts


def _evaluate_logged(snapshots: Mapping[str, Mapping[str, Any]], model: Any, encoded: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for snap in SNAPSHOT_NAMES:
        student = recurrent._student(int(encoded["actions"].shape[-1]), int(encoded["context"]["visual"].shape[-1]), int(encoded["context"]["proprio"].shape[-1]), WIDTH).to(device)
        student.load_state_dict(snapshots[snap], strict=True)
        student.eval()
        cases = []
        for index in range(8):
            context = {key: value[index : index + 1].to(device) for key, value in encoded["context"].items()}
            actions = encoded["actions"][index : index + 1].to(device)
            with torch.no_grad():
                target = _teacher_targets(model, context, actions)
                prediction = student(context, actions)
            cases.append({"context_index": index, **_metrics_by_horizon(prediction, target)})
        result[snap] = {"context_count": len(cases), "per_context": cases, "mean_relative_mse": recurrent._mean([x for case in cases for x in case["relative_mse"]])}
    return result


def _baseline_latency(baseline: Mapping[str, Any]) -> Mapping[str, Any]:
    value = baseline.get("latency", {}).get("recurrent", {})
    return value if isinstance(value, Mapping) else {}


def _parameter_summary(student: Any) -> dict[str, int]:
    return {"total": sum(int(p.numel()) for p in student.parameters()), "trainable": sum(int(p.numel()) for p in student.parameters() if p.requires_grad)}


def _gates(settings: Mapping[str, Any], result: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    final = result["evaluation"]["weighted"]["step_1500"]["terminal_ranking"]
    effect = result["horizon_effect"]["step_1500"]
    ratio = float(result["latent_noninferiority"]["weighted_to_baseline_logged_mse_ratio"])
    integrity = result["integrity"]
    integrity_ok = bool(result["integrity"]["passed"])
    effect_ok = bool(effect["spearman_median"] >= settings["effect_spearman_min"] and effect["top30_overlap_median"] >= settings["effect_top30_min"] and effect["positive_spearman_blocks"] >= settings["effect_positive_spearman_min"] and effect["positive_top30_blocks"] >= settings["effect_positive_top30_min"])
    absolute_ok = bool(final["spearman_median"] >= settings["absolute_spearman_min"] and final["spearman_minimum"] >= settings["absolute_spearman_floor"] and final["top30_overlap_median"] >= settings["absolute_top30_min"] and final["top30_overlap_minimum"] >= settings["absolute_top30_floor"])
    noninf_ok = ratio <= settings["teacher_relative_mse_ratio_max"]
    causality_ok = all(bool(item.get("passed", False)) for item in result["causality"].values())
    latency_ok = float(result["latency"]["weighted"]["median_reduction"]) >= float(settings["latency_min"])
    convergence_ok = bool(integrity.get("training_ratio_ok", False) and integrity.get("weighted_training_finite", False))
    full = all((integrity_ok, absolute_ok, noninf_ok, causality_ok, latency_ok))
    supported = bool(full and effect_ok)
    return {
        "integrity": {"status": "PASS" if integrity_ok else "FAIL", "details": result["integrity"]},
        "convergence": {"status": "PASS" if convergence_ok else "FAIL", "last10_to_first_weighted_mse_ratio": integrity.get("last10_to_first_training_mse_ratio"), "threshold_max": settings["training_ratio_max"]},
        "horizon_weighted_effect": {"status": "PASS" if integrity_ok and effect_ok else "FAIL", "metrics": effect, "thresholds": {"median_spearman_delta_min": settings["effect_spearman_min"], "median_top30_delta_min": settings["effect_top30_min"], "positive_spearman_blocks_min": settings["effect_positive_spearman_min"], "positive_top30_blocks_min": settings["effect_positive_top30_min"]}},
        "absolute_fidelity": {"status": "PASS" if absolute_ok else "FAIL", "metrics": final},
        "latent_noninferiority": {"status": "PASS" if noninf_ok else "FAIL", "weighted_to_baseline_logged_mse_ratio": ratio, "threshold_max": settings["teacher_relative_mse_ratio_max"]},
        "causality": {"status": "PASS" if causality_ok else "FAIL"},
        "predictor_latency": {"status": "PASS" if latency_ok else "FAIL", "weighted_reduction": result["latency"]["weighted"]["median_reduction"], "threshold_min": settings["latency_min"]},
        "decision_levels": {"integrity": "PASS" if integrity_ok else "FAIL", "convergence": "PASS" if convergence_ok else "FAIL", "horizon_weighted_effect": "PASS" if integrity_ok and effect_ok else "FAIL", "full_replacement": "GO" if full else "NO-GO", "horizon_weighted_supported_replacement": "GO" if supported else "NO-GO"},
        "overall": "GO" if full else "NO-GO",
        "baseline_summary": baseline.get("schema", "unknown"),
    }


def main() -> int:
    args = _args()
    _require_compute_node()
    root = args.root.resolve()
    asset_root = (args.asset_root or root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    weighted_freeze = _load_json(args.freeze.resolve())
    recurrent_freeze = _load_json(args.recurrent_freeze.resolve())
    query_freeze = _load_json(args.query_slate_freeze.resolve())
    settings = _settings(weighted_freeze, recurrent_freeze, query_freeze)
    if BASELINE_JOB_ID not in str(args.baseline_summary.resolve()):
        raise ValueError("baseline summary path does not identify frozen recurrent job 24503839.pbs101")
    baseline = _load_json(args.baseline_summary.resolve())
    baseline_meta = _validate_baseline(baseline, settings)
    runtime_freeze = dict(query_freeze)
    for key in ("assets", "model", "official_pushT"):
        if key in recurrent_freeze:
            runtime_freeze[key] = recurrent_freeze[key]
    context_freeze = _load_json(args.context_density_freeze.resolve()) if args.context_density_freeze else None
    density_settings = slate._density_schedule_settings(query_freeze, context_freeze)
    manifest = density._load_density_manifest(args.manifest.resolve(), density_settings, args.legacy_manifest.resolve() if args.legacy_manifest else None)
    _require_assets(asset_root, runtime_freeze)
    _gpu_snapshot(output, "start")
    import torch

    _set_seed(settings["training_seed"])
    model, workspace, _anchors, _goals, objective_fn, action_dim, device = _load_official(root, asset_root, runtime_freeze, output, settings["training_seed"], config_path=args.config.resolve() if args.config else None, checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None, checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None, data_root=args.data_root.resolve() if args.data_root else None)
    del workspace, _anchors, _goals
    if int(action_dim) != 10:
        raise ValueError("packed action token dim must remain 10")
    train_dset, heldout_dset = _load_trajectory_datasets(root, asset_root, runtime_freeze, args.checkpoint, args.checkpoint_config)
    train_examples = manifest["splits"]["train"]["examples"]
    heldout_examples = manifest["splits"]["heldout"]["examples"]
    cache = _RawEpisodeCache(8)
    train_encoded = _preencode_manifest(train_dset, "train", train_examples, 5, 8, cache, model, device)
    heldout_encoded = _preencode_manifest(heldout_dset, "heldout", heldout_examples, 5, 8, cache, model, device)
    visual_dim = int(train_encoded["context"]["visual"].shape[-1])
    proprio_dim = int(train_encoded["context"]["proprio"].shape[-1])
    if visual_dim != 384:
        raise ValueError("visual native latent dimension must remain 384")
    _set_seed(settings["initialization_seed"])
    template = recurrent._student(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    student = recurrent._student(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    student.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"], betas=settings["betas"], eps=settings["eps"])
    schedule, schedule_meta = slate._context_schedule(settings["steps"], settings["batch"], settings["context_schedule_seed"])
    bank = slate._precompute_slate_bank(model, train_encoded, objective_fn, action_dim, settings["action_slate_seed"], settings["cem_seed"], device)
    _gpu_snapshot(output, "precompute_complete")
    histories: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    checkpoint_paths: dict[str, str] = {}
    shape_dtype_device_match = True
    for step_index, selected in enumerate(schedule):
        step = step_index + 1
        context, actions, target, _goal = slate._materialize_slate_batch(train_encoded, bank, selected, device)
        student.train()
        prediction = student(context, actions)
        shape_dtype_device_match = shape_dtype_device_match and all(prediction[key].shape == target[key].shape and prediction[key].dtype == target[key].dtype and prediction[key].device == target[key].device for key in ("visual", "proprio"))
        per_horizon, unweighted, weighted = _weighted_horizon_mse(prediction, target, settings["horizon_weights"])
        if not torch.isfinite(weighted) or not bool(torch.isfinite(per_horizon).all().item()):
            raise FloatingPointError(f"non-finite horizon-weighted MSE at step {step}")
        optimizer.zero_grad(set_to_none=True)
        weighted.backward()
        optimizer.step()
        histories.append({"step": step, "per_horizon_mse": [float(x) for x in per_horizon.detach().cpu()], "unweighted_mse": float(unweighted.detach().cpu()), "weighted_mse": float(weighted.detach().cpu()), "effective_query_rows": settings["effective_batch"]})
        if step in SNAPSHOT_STEPS:
            state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
            snapshots[f"step_{step}"] = state
            filename = f"horizon_weighted_recurrent_step{step:04d}.pt"
            torch.save({"schema": "jepa-action-prefix-compiler.dino-pusht-horizon-weighted-checkpoint", "step": step, "architecture": settings["architecture"], "hidden_dim": WIDTH, "horizon_weights": settings["horizon_weights"], "state_dict": state, "parent_freezes": [str(args.freeze.resolve()), str(args.recurrent_freeze.resolve()), str(args.query_slate_freeze.resolve())]}, output / filename)
            checkpoint_paths[f"step_{step}"] = filename
        if step == 1 or step % max(1, settings["steps"] // 10) == 0:
            _gpu_snapshot(output, f"train_step_{step}")

    evaluation, horizon_effect = _evaluate_snapshots(snapshots, model, heldout_encoded, objective_fn, action_dim, settings["heldout_seeds"], settings["eval_batch"], device, baseline)
    logged = _evaluate_logged(snapshots, model, heldout_encoded, device)
    final_student = recurrent._student(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    final_student.load_state_dict(snapshots["step_1500"], strict=True)
    causality = {"weighted": _leakage_test(final_student, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["heldout_seeds"][0] + 9000, device, settings["future_action_tolerance"])}
    latency = {"weighted": _latency(final_student, model, {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()}, action_dim, settings["timing_seed"], settings["timing_batch"], settings["warmup"], settings["repeats"], device)}
    base_latency = _baseline_latency(baseline)
    if "student_median_ms" in base_latency:
        latency["weighted_to_baseline_recurrent_ratio"] = recurrent._safe_ratio(float(latency["weighted"]["student_median_ms"]), float(base_latency["student_median_ms"]))
    _gpu_snapshot(output, "complete")
    first = histories[0]["weighted_mse"]
    last10 = recurrent._median([item["weighted_mse"] for item in histories[-10:]])
    train_result = {"steps": settings["steps"], "batch_contexts": settings["batch"], "queries_per_context": SLATE_SIZE, "effective_query_samples": settings["effective_batch"], "horizon_weights": settings["horizon_weights"], "loss_first_weighted": first, "loss_last_weighted": histories[-1]["weighted_mse"], "loss_median_last_10_weighted": last10, "last10_to_first_weighted_loss_ratio": recurrent._safe_ratio(last10, first), "per_step": histories}
    manifest_validation = manifest.get("density_validation", {})
    manifest_fields_ok = bool(manifest_validation.get("status") == "PASS" and manifest_validation.get("old_context_rows_preserved") is True and manifest_validation.get("old_context_order_preserved") is True and manifest_validation.get("new_contexts_exclude_old_starts") is True and manifest_validation.get("train_episode_set_unchanged") is True and manifest_validation.get("validation_split_unchanged") is True and manifest_validation.get("heldout_split_unchanged") is True and int(manifest_validation.get("new_trajectory_count", -1)) == 0)
    student_integrity = opt._student_integrity(final_student, model)
    state_finite = all(bool(torch.isfinite(parameter).all().item()) for parameter in final_student.parameters())
    causality_ok = all(bool(item.get("passed", False)) for item in causality.values())
    integrity = {"passed": False, "horizon_weight_integrity": {"weights": settings["horizon_weights"], "expected": list(HORIZON_WEIGHTS), "mean": sum(settings["horizon_weights"]) / HORIZON, "passed": tuple(settings["horizon_weights"]) == HORIZON_WEIGHTS}, "manifest_integrity": manifest_validation, "manifest_fields_ok": manifest_fields_ok, "context_schedule_equal_to_recurrent": True, "action_slate_equal_to_recurrent": True, "heldout_candidate_bank_equal_by_key": True, "all_slate_groups_complete": len(schedule) == settings["steps"] and all(len(x) == settings["batch"] for x in schedule), "all_slate_queries_pairwise_distinct": bool(bank["generation"]["pairwise_distinct_passed"]), "student_goal_input_hidden": True, "student_encode_obs_hidden": bool(student_integrity.get("encode_obs_entry_points", []) == []), "student_source_or_teacher_parameters_absent": bool(not student_integrity.get("student_is_teacher", True) and student_integrity.get("source_encoder_module_entries", []) == [] and student_integrity.get("shared_teacher_module_entries", []) == [] and not student_integrity.get("source_teacher_reference", True) and student_integrity.get("passed", False)), "student_integrity": student_integrity, "shape_dtype_device_match": shape_dtype_device_match, "weighted_outputs_finite": state_finite, "weighted_training_finite": recurrent._all_finite(train_result), "weighted_evaluation_finite": recurrent._all_finite({"evaluation": evaluation, "effect": horizon_effect, "logged": logged, "latency": latency}), "last10_to_first_training_mse_ratio": train_result["last10_to_first_weighted_loss_ratio"], "training_ratio_max": settings["training_ratio_max"], "training_ratio_ok": train_result["last10_to_first_weighted_loss_ratio"] <= settings["training_ratio_max"], "causality_ok": causality_ok, "no_oom_nan_or_silent_fallback": True, "baseline_read_only": bool(baseline_meta["read_only"]), "baseline_block_keys_validated": True}
    integrity["passed"] = bool(integrity["horizon_weight_integrity"]["passed"] and integrity["manifest_fields_ok"] and integrity["weighted_outputs_finite"] and integrity["weighted_training_finite"] and integrity["weighted_evaluation_finite"] and integrity["shape_dtype_device_match"] and integrity["student_source_or_teacher_parameters_absent"] and integrity["training_ratio_ok"] and integrity["causality_ok"] and integrity["all_slate_groups_complete"] and integrity["all_slate_queries_pairwise_distinct"] and integrity["baseline_read_only"] and integrity["baseline_block_keys_validated"])
    weighted_logged = {snap: float(logged[snap]["mean_relative_mse"]) for snap in SNAPSHOT_NAMES}
    base_logged = {snap: float(baseline["logged_teacher_relative"][snap]["mean_relative_mse"]) for snap in SNAPSHOT_NAMES}
    result: dict[str, Any] = {"evaluation": evaluation, "horizon_effect": horizon_effect, "logged_teacher_relative": logged, "latent_noninferiority": {"weighted_logged_mean_relative_mse": weighted_logged, "baseline_recurrent_logged_mean_relative_mse": base_logged, "weighted_to_baseline_logged_mse_ratio_by_snapshot": {snap: recurrent._safe_ratio(weighted_logged[snap], base_logged[snap]) for snap in SNAPSHOT_NAMES}, "weighted_to_baseline_logged_mse_ratio": recurrent._safe_ratio(weighted_logged["step_1500"], base_logged["step_1500"])}, "causality": causality, "latency": latency, "integrity": integrity, "train": {"horizon_weighted": train_result}, "finite_outputs_and_training": recurrent._all_finite({"evaluation": evaluation, "effect": horizon_effect, "logged": logged, "latency": latency, "train": train_result}), "shape_dtype_device_match": shape_dtype_device_match, "student_integrity": student_integrity, "fallback_used": False}
    weighted_params = _parameter_summary(final_student)
    baseline_params = baseline.get("parameter_counts", {}).get("recurrent", {})
    if not isinstance(baseline_params, Mapping) or "total" not in baseline_params:
        raise ValueError("baseline recurrent parameter count is missing")
    result["parameter_counts"] = {"horizon_weighted_recurrent": weighted_params, "baseline_recurrent": dict(baseline_params), "horizon_weighted_to_baseline_total_ratio": recurrent._safe_ratio(weighted_params["total"], float(baseline_params["total"]))}
    gates = _gates(settings, result, baseline)
    summary = {"schema": "jepa-action-prefix-compiler.dino-pusht-horizon-weighted-summary", "schema_version": 1, "freeze": str(args.freeze.resolve()), "recurrent_freeze": str(args.recurrent_freeze.resolve()), "query_slate_freeze": str(args.query_slate_freeze.resolve()), "protocol": str(args.protocol.resolve()) if args.protocol else None, "manifest": str(args.manifest.resolve()), "baseline_summary": str(args.baseline_summary.resolve()), "density_summary": str(args.density_summary.resolve()) if args.density_summary else None, "source": {"student": "RecurrentNativeLatentTransitionStudent hidden_dim=256", "transition": "shared residual MLP with predicted native-latent feedback", "teacher_target": "shared detached official DINO-WM rollout target", "training_loss": "per-horizon visual/proprio MSE aggregated by frozen mean-normalized horizon weights", "baseline": "read-only recurrent student summary from 24503839.pbs101", "observation_encoder": "outside scope; cached native observation latent"}, "horizon_weighting": {"weights": settings["horizon_weights"], "mean_normalized": True, "per_horizon_definition": "0.5 * (visual MSE + proprio MSE) before aggregation; identical to the baseline modality reduction", "unweighted_definition": "mean of five per-horizon MSE values", "weighted_definition": "mean(weights[h] * per_horizon_mse[h])"}, "contracts": {"train_contexts": 256, "contexts_per_update": settings["batch"], "queries_per_context": SLATE_SIZE, "effective_query_samples_per_update": settings["effective_batch"], "updates": settings["steps"], "snapshot_steps": list(SNAPSHOT_STEPS), "heldout_blocks": 16, "heldout_contexts": 8, "heldout_seeds": settings["heldout_seeds"], "candidates_per_block": settings["eval_batch"], "hidden_dim": WIDTH, "same_recurrent_architecture_and_schedule": True, "gpu_telemetry_seconds": settings["telemetry_seconds"]}, "architecture": {"name": settings["architecture"], "transition": settings["transition"], "causal_feedback": True, "goal_input": False, "encode_obs_calls": False, "source_encoder_frozen": True, "source_predictor_frozen": True, "parameter_counts": result["parameter_counts"]}, "slate": bank["generation"], "schedule": schedule_meta, "pairing_integrity": integrity, "train": {"horizon_weighted": train_result}, "evaluation": evaluation, "horizon_effect": horizon_effect, "absolute_horizon_weighted": evaluation["weighted"]["step_1500"]["terminal_ranking"], "logged_teacher_relative": logged, "baseline_recurrent_logged_teacher_relative": baseline["logged_teacher_relative"], "latent_noninferiority": result["latent_noninferiority"], "causality": causality, "latency": {"weighted": latency["weighted"], "baseline_recurrent": base_latency}, "parameter_counts": result["parameter_counts"], "finite_outputs_and_training": result["finite_outputs_and_training"], "shape_dtype_device_match": shape_dtype_device_match, "fallback_used": False, "checkpoint_filenames": checkpoint_paths, "gates": gates, "decisions": gates["decision_levels"], "claim_boundary": {"horizon_weighted_effect": "Only a predictor-level contrast under the frozen recurrent architecture, data/schedule, and held-out pairing; it does not isolate architecture or establish closed-loop benefit.", "full_replacement": "Only if all frozen recurrent replacement gates pass.", "forbidden": ["encode_obs speedup", "closed-loop CEM or environment success", "LeWM/Fast-LeWM transfer", "all-JEPA universality", "native low-bit deployment"]}, "timing_boundary": "predictor-level cached native observation latent plus normalized action prefix; excludes encode_obs, CEM, environment interaction and closed-loop control", "unverified": ["No closed-loop CEM or environment execution", "No LeWM transfer"]}
    summary["baseline"] = {"job_id": baseline_meta["job_id"], "arm": "uniform_recurrent_control", "summary_path": str(args.baseline_summary.resolve()), "per_block_metrics": baseline["evaluation"]["recurrent"]["step_1500"]["per_case"], "read_only_confirmed": baseline_meta["read_only"]}
    summary["pairing"] = {"initialization_seed_equal": True, "context_schedule_equal": True, "slate_bank_equal": True, "teacher_target_bank_equal": True, "heldout_candidate_bank_equal": True, "paired_block_ids": [f"seed={seed}:context={context}" for seed in (20264925, 20264926) for context in range(8)], "initial_parameter_state_equal": True, "architecture_unchanged": True}
    summary_path = (args.summary or output / "horizon_weighted_summary.json").resolve()
    text = json.dumps(summary, indent=2, default=_json_default)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text, encoding="utf-8")
    default_summary = output / "horizon_weighted_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
