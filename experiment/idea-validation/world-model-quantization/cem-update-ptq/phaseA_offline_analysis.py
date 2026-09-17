#!/usr/bin/env python3
"""Stage-A CPU-only audit for CEM-Update PTQ on frozen development pools.

The script reads the previously produced joint-fidelity score NPZ files and
the FP32 development workload.  It never loads the model, imports torch, or
contacts ASPIRE2A.  All aggregation is pool -> MPC point -> episode with
equal weight at each level.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METHODS = (
    "FP32",
    "all_W4",
    "all_W8",
    "RankCal",
    "LocalMSE",
    "ScoreError",
    "Random1",
    "Random2",
)
OLD_METRICS = (
    "e2_pairwise_disagreement",
    "stable30_elite_overlap",
    "stable30_elite_overlap_fraction",
    "elite_mean_normalized_action_mse",
    "score_nmse",
)
LOSS_METRICS = ("L_mu", "L_sigma", "L_update")
EXPECTED_CASES = 26
EXPECTED_EPISODES = 8
EXPECTED_CANDIDATES = 300
ELITE_COUNT = 30
HORIZON = 5
ACTION_DIM = 10
TIE_RTOL = 2e-5
TIE_ATOL = 1e-7


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "pool"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_evidence(path: Path, line_number: int, fragment: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    text = lines[line_number - 1] if 0 < line_number <= len(lines) else None
    return {
        "path": str(path.resolve()),
        "line": line_number,
        "text": text,
        "expected_fragment": fragment,
        "verified": text is not None and fragment in text,
    }


def _stable_elite(scores: np.ndarray) -> np.ndarray:
    return np.argsort(scores, kind="stable")[:ELITE_COUNT].astype(np.int64, copy=False)


def _tie_summary(scores: np.ndarray, elite: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(scores, return_counts=True)
    repeated = counts[counts > 1]
    boundary_value = scores[int(elite[-1])]
    boundary_count = int(np.count_nonzero(scores == boundary_value))
    below_boundary = int(np.count_nonzero(scores < boundary_value))
    return {
        "unique_score_values": int(values.size),
        "tie_groups": int(repeated.size),
        "tie_candidate_pairs": int(np.sum(repeated * (repeated - 1) // 2)),
        "max_equal_score_group": int(repeated.max()) if repeated.size else 1,
        "top30_boundary_value": float(boundary_value),
        "top30_boundary_group_size": boundary_count,
        "top30_boundary_crossing": bool(
            boundary_count > 1 and below_boundary < ELITE_COUNT < below_boundary + boundary_count
        ),
    }


def _pairwise_e2(reference: np.ndarray, quantized: np.ndarray) -> float:
    i, j = np.triu_indices(reference.size, k=1)
    return float(np.mean(np.sign(reference[i] - reference[j]) != np.sign(quantized[i] - quantized[j])))


def _score_nmse(reference: np.ndarray, quantized: np.ndarray) -> float:
    return float(np.mean((quantized - reference) ** 2) / (np.mean(reference**2) + 1e-12))


def _old_pool_metrics(reference: np.ndarray, quantized: np.ndarray, candidates: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    reference_elite = _stable_elite(reference)
    quantized_elite = _stable_elite(quantized)
    reference_mean = candidates[reference_elite].mean(axis=0)
    quantized_mean = candidates[quantized_elite].mean(axis=0)
    overlap = int(np.intersect1d(reference_elite, quantized_elite).size)
    return (
        {
            "e2_pairwise_disagreement": _pairwise_e2(reference, quantized),
            "stable30_elite_overlap": float(overlap),
            "stable30_elite_overlap_fraction": overlap / float(ELITE_COUNT),
            "elite_mean_normalized_action_mse": float(np.mean((quantized_mean - reference_mean) ** 2)),
            "score_nmse": _score_nmse(reference, quantized),
        },
        quantized_elite,
    )


def _update_losses(reference: np.ndarray, quantized: np.ndarray, candidates: np.ndarray, iter_weight: float) -> dict[str, float]:
    reference_elite = _stable_elite(reference)
    quantized_elite = _stable_elite(quantized)
    reference_selected = candidates[reference_elite]
    quantized_selected = candidates[quantized_elite]
    # The workload stores float32 actions; float64 accumulation makes the
    # offline diagnostic deterministic while preserving the saved values.
    mu_reference = reference_selected.mean(axis=0, dtype=np.float64)
    mu_quantized = quantized_selected.mean(axis=0, dtype=np.float64)
    sigma_reference = reference_selected.std(axis=0, ddof=1, dtype=np.float64)
    sigma_quantized = quantized_selected.std(axis=0, ddof=1, dtype=np.float64)
    l_mu = float(np.mean((mu_quantized - mu_reference) ** 2))
    l_sigma = float(np.mean((sigma_quantized - sigma_reference) ** 2))
    return {"L_mu": l_mu, "L_sigma": l_sigma, "L_update": l_mu + iter_weight * l_sigma}


def _mean_records(records: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> dict[str, dict[str, float]]:
    methods: dict[str, dict[str, float]] = {}
    for method in METHODS:
        methods[method] = {
            name: float(
                np.mean(
                    [
                        float(
                            row["methods"][method][name]
                            if name in row["methods"][method]
                            else row["methods"][method]["old_metrics"][name]
                        )
                        for row in records
                    ]
                )
            )
            for name in metric_names
        }
    return methods


def _aggregate(pool_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_point: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        by_point[(str(row["episode_id"]), int(row["mpc_point"]))].append(row)
    metric_names = LOSS_METRICS + OLD_METRICS
    point_rows: list[dict[str, Any]] = []
    for (episode_id, point), rows in sorted(by_point.items()):
        point_rows.append(
            {
                "episode_id": episode_id,
                "mpc_point": point,
                "pool_ids": [str(row["pool_id"]) for row in rows],
                "pool_count": len(rows),
                "methods": _mean_records(rows, metric_names),
            }
        )
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in point_rows:
        by_episode[str(row["episode_id"])].append(row)
    episode_rows: list[dict[str, Any]] = []
    for episode_id, rows in sorted(by_episode.items()):
        episode_rows.append(
            {
                "episode_id": episode_id,
                "point_count": len(rows),
                "pool_count": int(sum(int(row["pool_count"]) for row in rows)),
                "methods": _mean_records(rows, metric_names),
            }
        )
    overall = _mean_records(episode_rows, metric_names)
    return {
        "point_metrics": point_rows,
        "episode_metrics": episode_rows,
        "overall_equal_episode_weight": overall,
        "aggregation": "mean pools within MPC point, mean visited points within episode, then equal-weight mean across episodes",
    }


def _rankings(overall: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    fields = {
        "L_mu": "lower",
        "L_sigma": "lower",
        "L_update": "lower",
        "e2_pairwise_disagreement": "lower",
        "elite_mean_normalized_action_mse": "lower",
        "stable30_elite_overlap_fraction": "higher",
    }
    mixed = [method for method in METHODS if method not in {"FP32", "all_W4", "all_W8"}]
    result: dict[str, Any] = {}
    for field, direction in fields.items():
        for label, names in (("all_methods", list(METHODS)), ("mixed_allocations", mixed)):
            ordered = sorted(
                names,
                key=lambda method: (
                    float(overall[method][field]) if direction == "lower" else -float(overall[method][field]),
                    method,
                ),
            )
            result.setdefault(field, {})[label] = [
                {"method": method, "value": float(overall[method][field])} for method in ordered
            ]
    return result


def _resolve_defaults(script_path: Path) -> dict[str, Path]:
    experiment_root = script_path.resolve().parents[1]
    repo_root = experiment_root.parents[2]
    old_root = experiment_root / "dino-wm-wall"
    artifacts = old_root / "artifacts" / "screen" / "artifacts" / "16177209.pbs101" / "fidelity"
    return {
        "workload": old_root / "artifacts" / "screen" / "artifacts" / "16176810.pbs101" / "evaluation" / "development" / "FP32" / "workload.pkl",
        "scores_dir": artifacts / "pools",
        "joint_verification": old_root / "artifacts" / "screen" / "joint_verification.json",
        "fidelity_summary": artifacts / "summary.json",
        "old_pool_metrics": artifacts / "pool_metrics.jsonl",
        "old_episode_metrics": artifacts / "episode_metrics.json",
        "handoff": experiment_root.parent / "CEM_UPDATE_PTQ_HANDOFF.zh.md",
        "screen_runner": old_root / "screen_runner.py",
        "source_cem": repo_root / "reproduction" / "dino-wm-wall" / "source" / "planning" / "cem.py",
        "output_json": experiment_root / "cem-update-ptq" / "phaseA_results.json",
        "output_md": experiment_root / "cem-update-ptq" / "PHASE_A_RESULT.zh.md",
    }


def _load_old_pool_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pool_id = str(row["pool_id"])
        if pool_id in rows:
            raise ValueError(f"duplicate old pool metric row: {pool_id}")
        rows[pool_id] = row
    return rows


def _validate_close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=TIE_RTOL, atol=TIE_ATOL))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    workload_path = args.workload.resolve()
    scores_dir = args.scores_dir.resolve()
    joint_path = args.joint_verification.resolve()
    fidelity_path = args.fidelity_summary.resolve()
    old_pool_path = args.old_pool_metrics.resolve()
    old_episode_path = args.old_episode_metrics.resolve()
    handoff_path = args.handoff.resolve()
    screen_runner_path = args.screen_runner.resolve()
    source_cem_path = args.source_cem.resolve()

    with workload_path.open("rb") as stream:
        workload = __import__("pickle").load(stream)
    if not isinstance(workload, Mapping) or workload.get("schema") != "rankcal-wall-screen-workload-v1":
        raise ValueError("unexpected workload schema")
    cases = workload.get("cases")
    if not isinstance(cases, Sequence) or len(cases) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} workload cases")
    planner = workload.get("planner", {})
    required_planner = {
        "num_samples": EXPECTED_CANDIDATES,
        "elite_count": ELITE_COUNT,
        "cem_iterations": 5,
        "stable_argsort": True,
    }
    planner_checks = {key: planner.get(key) == value for key, value in required_planner.items()}
    if not all(planner_checks.values()):
        raise ValueError(f"workload planner mismatch: {planner_checks}")

    old_pool = _load_old_pool_rows(old_pool_path)
    if len(old_pool) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} old pool rows, got {len(old_pool)}")
    joint = _load_json(joint_path)
    fidelity = _load_json(fidelity_path)
    for evidence, name in ((joint, "joint_verification"), (fidelity, "fidelity_summary")):
        if evidence.get("pool_count") != EXPECTED_CASES or evidence.get("episode_count") != EXPECTED_EPISODES:
            raise ValueError(f"{name} count mismatch")
        if tuple(evidence.get("methods", ())) != METHODS:
            raise ValueError(f"{name} method order mismatch")

    errors: list[str] = []
    pool_rows: list[dict[str, Any]] = []
    score_file_count = 0
    reference_match_count = 0
    stored_elite_match_count = 0
    old_pool_match_count = 0
    old_pool_checks = 0
    tie_by_method: dict[str, dict[str, int]] = {
        method: {
            "pools_with_any_ties": 0,
            "pools_with_top30_boundary_tie": 0,
            "tie_groups": 0,
            "tie_candidate_pairs": 0,
            "max_equal_score_group": 1,
        }
        for method in METHODS
    }

    for case in cases:
        pool_id = str(case["pool_id"])
        episode_id = str(case["episode_id"])
        candidates = np.asarray(case["candidates"])
        if candidates.shape != (EXPECTED_CANDIDATES, HORIZON, ACTION_DIM):
            errors.append(f"{pool_id}: candidate shape {candidates.shape}")
        if candidates.dtype.kind not in "fiu" or not np.isfinite(candidates).all():
            errors.append(f"{pool_id}: candidate actions are not finite numeric values")
        candidates64 = candidates.astype(np.float64, copy=False)
        score_path = scores_dir / f"pool_{_safe_id(pool_id)}.scores.npz"
        if not score_path.is_file():
            errors.append(f"{pool_id}: missing score file {score_path}")
            continue
        score_file_count += 1
        with np.load(score_path, allow_pickle=False) as archive:
            missing = [method for method in METHODS if method not in archive.files]
            if missing:
                errors.append(f"{pool_id}: missing score arrays {missing}")
                continue
            scores = {method: np.asarray(archive[method], dtype=np.float64).reshape(-1) for method in METHODS}
            stored_elites = {
                method: np.asarray(archive[f"{method}_elite_indices"], dtype=np.int64).reshape(-1)
                for method in METHODS
                if f"{method}_elite_indices" in archive.files
            }
        for method, score in scores.items():
            if score.shape != (EXPECTED_CANDIDATES,) or not np.isfinite(score).all():
                errors.append(f"{pool_id}/{method}: malformed or non-finite score vector")
        case_reference = np.asarray(case.get("reference_scores"), dtype=np.float64).reshape(-1)
        if case_reference.shape == scores["FP32"].shape and np.array_equal(case_reference, scores["FP32"]):
            reference_match_count += 1
        else:
            errors.append(f"{pool_id}: workload reference_scores differs from NPZ FP32")

        old_row = old_pool.get(pool_id)
        if old_row is None:
            errors.append(f"{pool_id}: missing old pool metric row")
        methods: dict[str, Any] = {}
        iter_weight = {1: 1.0, 5: 0.0}.get(int(case["cem_iteration"]))
        if iter_weight is None:
            errors.append(f"{pool_id}: unexpected CEM iteration {case['cem_iteration']}")
            iter_weight = 0.0
        for method in METHODS:
            score = scores[method]
            stable_elite = _stable_elite(score)
            stored_elite = stored_elites.get(method)
            elite_match = stored_elite is not None and np.array_equal(stable_elite, stored_elite)
            if elite_match:
                stored_elite_match_count += 1
            else:
                errors.append(f"{pool_id}/{method}: stored elite indices are not stable top-30 on NPZ scores")
            ties = _tie_summary(score, stable_elite)
            tie_by_method[method]["pools_with_any_ties"] += int(ties["tie_groups"] > 0)
            tie_by_method[method]["pools_with_top30_boundary_tie"] += int(ties["top30_boundary_crossing"])
            tie_by_method[method]["tie_groups"] += ties["tie_groups"]
            tie_by_method[method]["tie_candidate_pairs"] += ties["tie_candidate_pairs"]
            tie_by_method[method]["max_equal_score_group"] = max(
                tie_by_method[method]["max_equal_score_group"], ties["max_equal_score_group"]
            )
            old_metrics, _ = _old_pool_metrics(scores["FP32"], score, candidates64)
            losses = _update_losses(scores["FP32"], score, candidates64, float(iter_weight))
            old_expected = old_row.get("methods", {}).get(method) if old_row else None
            old_match = True
            if old_expected is None:
                old_match = False
            else:
                for field in OLD_METRICS:
                    old_pool_checks += 1
                    if field == "stable30_elite_overlap":
                        field_match = int(round(old_metrics[field])) == int(round(float(old_expected[field])))
                    else:
                        field_match = _validate_close(old_metrics[field], float(old_expected[field]))
                    old_match = old_match and field_match
            if old_match:
                old_pool_match_count += 1
            else:
                errors.append(f"{pool_id}/{method}: recomputed old metrics differ from old pool_metrics.jsonl")
            methods[method] = {
                **losses,
                "iter_weight": float(iter_weight),
                "old_metrics": old_metrics,
                "ties": ties,
                "stable_top30_matches_stored": bool(elite_match),
                "finite": bool(score.shape == (EXPECTED_CANDIDATES,) and np.isfinite(score).all()),
            }
        pool_rows.append(
            {
                "pool_id": pool_id,
                "episode_id": episode_id,
                "mpc_point": int(case["mpc_point"]),
                "cem_iteration": int(case["cem_iteration"]),
                "candidate_shape": list(candidates.shape),
                "methods": methods,
            }
        )

    if score_file_count != EXPECTED_CASES:
        errors.append(f"score file count {score_file_count} != {EXPECTED_CASES}")
    if len(pool_rows) != EXPECTED_CASES:
        errors.append(f"usable pool row count {len(pool_rows)} != {EXPECTED_CASES}")
    episode_ids = sorted({str(row["episode_id"]) for row in pool_rows})
    if len(episode_ids) != EXPECTED_EPISODES:
        errors.append(f"episode count {len(episode_ids)} != {EXPECTED_EPISODES}")

    aggregate = _aggregate(pool_rows)
    old_episode = _load_json(old_episode_path)
    old_episode_rows = {str(row["episode_id"]): row for row in old_episode.get("episodes", [])}
    episode_old_checks = 0
    episode_old_max_abs_diff = 0.0
    for row in aggregate["episode_metrics"]:
        expected = old_episode_rows.get(str(row["episode_id"]))
        if expected is None:
            errors.append(f"missing old episode metric row {row['episode_id']}")
            continue
        for method in METHODS:
            for field in OLD_METRICS:
                episode_old_checks += 1
                actual = float(row["methods"][method][field])
                wanted = float(expected["methods"][method][field])
                episode_old_max_abs_diff = max(episode_old_max_abs_diff, abs(actual - wanted))
                if field == "stable30_elite_overlap":
                    match = int(round(actual)) == int(round(wanted))
                else:
                    match = _validate_close(actual, wanted)
                if not match:
                    errors.append(f"{row['episode_id']}/{method}/{field}: episode metric mismatch")
    old_overall = old_episode.get("overall_equal_episode_weight", {})
    overall_old_checks = 0
    overall_old_max_abs_diff = 0.0
    for method in METHODS:
        for field in OLD_METRICS:
            overall_old_checks += 1
            actual = float(aggregate["overall_equal_episode_weight"][method][field])
            wanted = float(old_overall[method][field])
            overall_old_max_abs_diff = max(overall_old_max_abs_diff, abs(actual - wanted))
            if field == "stable30_elite_overlap":
                match = int(round(actual)) == int(round(wanted))
            else:
                match = _validate_close(actual, wanted)
            if not match:
                errors.append(f"overall/{method}/{field}: overall metric mismatch")

    semantic_evidence = [
        _line_evidence(screen_runner_path, 654, "candidate = torch.randn"),
        _line_evidence(screen_runner_path, 655, "candidate[0] = mu"),
        _line_evidence(screen_runner_path, 663, "torch.argsort(scores, stable=True)"),
        _line_evidence(screen_runner_path, 667, "topk_action.std(dim=0)"),
        _line_evidence(screen_runner_path, 609, "candidate_actions"),
        _line_evidence(screen_runner_path, 816, '"candidates": pool["candidate_actions"]'),
        _line_evidence(screen_runner_path, 772, "planned_actions_normalized"),
        _line_evidence(source_cem_path, 67, "actions: normalized"),
        _line_evidence(handoff_path, 59, "L_update = L_mu + w_iter * L_sigma"),
        _line_evidence(handoff_path, 64, "`w_iter=1`"),
    ]
    semantic_verified = all(item["verified"] for item in semantic_evidence)
    if not semantic_verified:
        errors.append("one or more adapter/normalization source evidence checks failed")

    overall = aggregate["overall_equal_episode_weight"]
    ranking = _rankings(overall)
    mixed_methods = [method for method in METHODS if method not in {"FP32", "all_W4", "all_W8"}]
    nondegenerate: dict[str, Any] = {}
    for field in LOSS_METRICS:
        values = [float(overall[method][field]) for method in mixed_methods]
        nondegenerate[field] = {
            "finite": bool(np.isfinite(values).all()),
            "min": float(min(values)),
            "max": float(max(values)),
            "distinct_values": int(len({round(value, 15) for value in values})),
            "spread": float(max(values) - min(values)),
        }
    nondegenerate["mechanism_measurement_gate"] = bool(
        all(item["finite"] for field, item in nondegenerate.items() if field in LOSS_METRICS)
        and any(item["distinct_values"] > 1 for field, item in nondegenerate.items() if field in LOSS_METRICS)
    )

    payload: dict[str, Any] = {
        "schema": "cem-update-ptq-stageA-offline-v1",
        "status": "complete" if not errors else "validation_failed",
        "stage": "A",
        "execution": {
            "mode": "CPU-only offline",
            "model_loaded": False,
            "torch_imported": False,
            "gpu_used": False,
            "ssh_or_remote_connection": False,
            "download_or_hash": False,
            "weights_fit_or_success_tuned": False,
        },
        "inputs": {
            "workload": str(workload_path),
            "score_directory": str(scores_dir),
            "joint_verification": str(joint_path),
            "joint_fidelity_summary": str(fidelity_path),
            "old_pool_metrics": str(old_pool_path),
            "old_episode_metrics": str(old_episode_path),
        },
        "input_identity": {
            "workload_schema": workload.get("schema"),
            "workload_split": workload.get("split"),
            "screen_split": workload.get("screen_split"),
            "planner_checks": planner_checks,
            "case_count": len(cases),
            "episode_count": len(episode_ids),
            "methods": list(METHODS),
            "candidate_shape": [EXPECTED_CANDIDATES, HORIZON, ACTION_DIM],
        },
        "semantic_audit": {
            "stable_top30": "np.argsort(kind='stable')[:30], matching the actual adapter's torch.argsort(..., stable=True)",
            "std_correction": 1,
            "std_definition": "sample standard deviation with ddof=1, matching torch.std(dim=0) default correction=1",
            "iteration_weights": {"cem_iteration_1": 1.0, "cem_iteration_5": 0.0},
            "scale_s": 1.0,
            "scale_interpretation": "fixed unit scale in action-normalized coordinates; source evidence establishes the coordinate convention, not raw environment units",
            "evidence": semantic_evidence,
            "all_evidence_verified": semantic_verified,
        },
        "data_integrity": {
            "workload_cases": len(cases),
            "score_files_checked": score_file_count,
            "reference_scores_equal_npz_fp32": reference_match_count,
            "stable_top30_equal_stored_elites": stored_elite_match_count,
            "expected_stable_top30_checks": EXPECTED_CASES * len(METHODS),
            "old_pool_metric_rows_matching": old_pool_match_count,
            "expected_old_pool_metric_checks": EXPECTED_CASES * len(METHODS),
            "old_metric_tolerance": {"rtol": TIE_RTOL, "atol": TIE_ATOL, "overlap": "exact integer"},
            "old_episode_metric_checks": episode_old_checks,
            "old_episode_max_abs_diff": episode_old_max_abs_diff,
            "old_overall_metric_checks": overall_old_checks,
            "old_overall_max_abs_diff": overall_old_max_abs_diff,
            "errors": errors,
            "passed": not errors,
        },
        "tie_summary": tie_by_method,
        "pool_metrics": pool_rows,
        "aggregation": aggregate,
        "overall_ranking": ranking,
        "nondegenerate": nondegenerate,
        "interpretation": {
            "data_sufficient_for_stageA": not errors,
            "update_fidelity_is_measurable_and_nondegenerate": bool(nondegenerate["mechanism_measurement_gate"]),
            "continue_mechanism_screening": bool(not errors and nondegenerate["mechanism_measurement_gate"]),
            "success_claim": False,
            "next_gate": "Stage B may be considered for a bounded CAL/DEV mechanism check; this Stage-A result does not authorize GPU execution or prove closed-loop benefit.",
            "weight_policy": "fixed handoff weights only; no success-based weighting or fitting",
        },
        "script": str(Path(__file__).resolve()),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(args.output_json.resolve(), payload)
    _atomic_write(args.output_md.resolve(), _render_markdown(payload))
    return payload


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.3f}"
    return f"{value:.8f}"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    integrity = payload["data_integrity"]
    overall = payload["aggregation"]["overall_equal_episode_weight"]
    lines = [
        "# CEM-Update PTQ 阶段 A：旧 development 数据离线分析",
        "",
        f"状态：`{payload['status']}`。本轮为 `{payload['execution']['mode']}`；没有加载 model、使用 GPU、连接 SSH 或拟合权重。",
        "",
        "## 数据完整性",
        "",
        f"读取 1 个 development `workload.pkl`、{integrity['score_files_checked']}/26 个 joint-fidelity score NPZ、旧 `joint_verification`、pool/episode 汇总。候选池为 26 个、8 个 episode、每池 `(300, 5, 10)`。",
        f"`reference_scores` 与 NPZ FP32 一致：{integrity['reference_scores_equal_npz_fp32']}/26；stable top-30 与 NPZ 保存的 elite indices 一致：{integrity['stable_top30_equal_stored_elites']}/{26 * len(METHODS)}。",
        f"旧 E2/elite-mean/score-NMSE pool rows 复核：{integrity['old_pool_metric_rows_matching']}/{26 * len(METHODS)}；episode 与 overall 最大绝对差分别为 `{integrity['old_episode_max_abs_diff']:.3g}`、`{integrity['old_overall_max_abs_diff']:.3g}`。",
        "",
        "## 固定语义",
        "",
        "`stable top30` 使用实际 adapter 的 stable argsort；`std` 使用 `ddof=1`（PyTorch `torch.std(dim=0)` 默认 correction=1）。`L_update = L_mu + w_iter × L_sigma`，CEM iteration 1/5 的固定权重分别为 1/0；`s=1` 解释为 action-normalized coordinate 的固定单位尺度。完整源码行证据记录在 `phaseA_results.json`。",
        "",
        "## Overall（episode 等权）",
        "",
        "| 方法 | L_mu ↓ | L_sigma ↓ | L_update ↓ | E2 ↓ | Elite overlap ↑ | Elite-mean MSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = overall[method]
        lines.append(
            f"| {method} | {_fmt(float(row['L_mu']))} | {_fmt(float(row['L_sigma']))} | {_fmt(float(row['L_update']))} | {float(row['e2_pairwise_disagreement']):.6f} | {float(row['stable30_elite_overlap_fraction']):.6f} | {_fmt(float(row['elite_mean_normalized_action_mse']))} |"
        )
    lines.extend(
        [
            "",
            "## 每 episode 的 L_update（完整 L_mu/L_sigma/L_update 在 JSON）",
            "",
            "| episode | pools/points | FP32 | all-W4 | all-W8 | RankCal | LocalMSE | ScoreError | Random1 | Random2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["aggregation"]["episode_metrics"]:
        methods = row["methods"]
        values = " | ".join(_fmt(float(methods[method]["L_update"])) for method in METHODS)
        lines.append(f"| {row['episode_id']} | {row['pool_count']}/{row['point_count']} | {values} |")
    ranking = payload["overall_ranking"]
    lines.extend(
        [
            "",
            "## 描述性排序差异",
            "",
            "mixed allocations 的 overall 排名如下（不含 FP32/all-W4/all-W8）：",
            "",
        ]
    )
    for field in ("L_update", "L_mu", "L_sigma", "e2_pairwise_disagreement", "elite_mean_normalized_action_mse", "stable30_elite_overlap_fraction"):
        order = ranking[field]["mixed_allocations"]
        direction = "↓" if field != "stable30_elite_overlap_fraction" else "↑"
        separator = " > " if direction == "↑" else " < "
        lines.append(f"- `{field}` {direction}：" + separator.join(item["method"] for item in order))
    lines.extend(
        [
            "",
            "## 阶段 A 判断",
            "",
            f"三项 update fidelity loss 均 finite 且在 mixed allocations 间有数值差异；measurement gate = `{payload['nondegenerate']['mechanism_measurement_gate']}`。因此旧数据足以完成阶段 A，且可进入有界的 Stage-B CAL/DEV 机制筛选。",
            "这只说明 CEM update 目标在现有固定 pools 上可测并提供不同排序信息；没有闭环新结果，也没有证明新 idea work。不得根据旧 success 反调 `L_mu/L_sigma` 权重；本轮只使用预先冻结的 iter 权重。",
            "",
            "## 输入与限制",
            "",
            f"脚本：`{payload['script']}`。旧输出保持不变；新结果位于本目录的 `phaseA_results.json` 与 `PHASE_A_RESULT.zh.md`。",
            "",
        ]
    )
    if integrity["errors"]:
        lines.extend(["验证错误：", "", *[f"- {error}" for error in integrity["errors"]], ""])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    defaults = _resolve_defaults(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=defaults["workload"])
    parser.add_argument("--scores-dir", type=Path, default=defaults["scores_dir"])
    parser.add_argument("--joint-verification", type=Path, default=defaults["joint_verification"])
    parser.add_argument("--fidelity-summary", type=Path, default=defaults["fidelity_summary"])
    parser.add_argument("--old-pool-metrics", type=Path, default=defaults["old_pool_metrics"])
    parser.add_argument("--old-episode-metrics", type=Path, default=defaults["old_episode_metrics"])
    parser.add_argument("--handoff", type=Path, default=defaults["handoff"])
    parser.add_argument("--screen-runner", type=Path, default=defaults["screen_runner"])
    parser.add_argument("--source-cem", type=Path, default=defaults["source_cem"])
    parser.add_argument("--output-json", type=Path, default=defaults["output_json"])
    parser.add_argument("--output-md", type=Path, default=defaults["output_md"])
    return parser.parse_args()


def main() -> None:
    payload = analyze(_parse_args())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pool_count": payload["input_identity"]["case_count"],
                "episode_count": payload["input_identity"]["episode_count"],
                "stable_top30_checks": payload["data_integrity"]["stable_top30_equal_stored_elites"],
                "measurement_gate": payload["nondegenerate"]["mechanism_measurement_gate"],
                "output_json": str((Path(__file__).resolve().parent / "phaseA_results.json")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if payload["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
