"""Analyze the frozen original-C80, 16-seed BF16 teacher diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
from itertools import combinations
from pathlib import Path
from typing import Any


N_CASES = 80
N_SEEDS = 16
N_STEPS = 10
N_MOTORS = 6
PROJECTION_SEED = 20261021


def require_compute_allocation() -> None:
    """Fail closed unless PBS says this process is on one of its compute nodes."""
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile_name = os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.gethostname().split(".", 1)[0].lower()
    if not job_id or not nodefile_name:
        raise RuntimeError("需要在获批的 PBS allocation 内运行：PBS_JOBID/PBS_NODEFILE 缺失。")
    if "login" in host:
        raise RuntimeError(f"拒绝在 login node 上运行分析：hostname={host}")
    nodefile = Path(nodefile_name)
    if not nodefile.is_file():
        raise RuntimeError(f"PBS_NODEFILE 不存在或不可读：{nodefile}")
    nodes = {line.strip().split(".", 1)[0].lower() for line in nodefile.read_text().splitlines() if line.strip()}
    if host not in nodes:
        raise RuntimeError(f"当前 hostname={host} 不在 PBS_NODEFILE 节点列表中。")


def _deps():
    import numpy as np
    import torch

    return np, torch


def _finite(value: Any) -> Any:
    """Convert numpy values to strict JSON and replace non-finite scalars with null."""
    np, _ = _deps()

    def convert(item):
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if math.isfinite(float(item)) else None
        if isinstance(item, np.bool_):
            return bool(item)
        return item

    return convert(value)


def _stats(values) -> dict[str, float | None]:
    np, _ = _deps()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "q90": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.9)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _relative_frobenius(a, b) -> float:
    np, _ = _deps()
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) + np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(2.0 * np.linalg.norm(a - b) / denom)


def _covariance_and_rank(samples) -> tuple[Any, dict[str, float | int]]:
    np, _ = _deps()
    x = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(x) - 1)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    total = float(eig.sum())
    if total == 0.0:
        return cov, {"effective_rank": 0.0, "numerical_rank": 0, "max_eigenvalue_fraction": 0.0}
    p = eig[eig > 0.0] / total
    effective_rank = float(np.exp(-np.sum(p * np.log(p))))
    rank_tol = float(eig.max()) * 1e-8
    return cov, {
        "effective_rank": effective_rank,
        "numerical_rank": int(np.count_nonzero(eig > rank_tol)),
        "max_eigenvalue_fraction": float(eig.max() / total),
    }


def _projection_matrix():
    np, _ = _deps()
    n_features = N_STEPS * N_MOTORS
    directions = [np.eye(n_features, dtype=np.float64)]
    mean_dirs = np.zeros((N_MOTORS, n_features), dtype=np.float64)
    half_diff_dirs = np.zeros_like(mean_dirs)
    for motor in range(N_MOTORS):
        mean_dirs[motor, motor::N_MOTORS] = 1.0 / N_STEPS
        half_diff_dirs[motor, motor::N_MOTORS][:5] = 1.0 / 5.0
        half_diff_dirs[motor, motor::N_MOTORS][5:] = -1.0 / 5.0
    directions.extend([mean_dirs, half_diff_dirs])
    rng = np.random.default_rng(PROJECTION_SEED)
    random_dirs = rng.standard_normal((16, n_features))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    directions.append(random_dirs)
    matrix = np.concatenate(directions, axis=0)
    if matrix.shape != (88, n_features):
        raise AssertionError(f"expected 88 fixed projections, got {matrix.shape}")
    return matrix


def _histogram(values):
    """Histogram standardized projections into shared bins, retaining both tails."""
    np, _ = _deps()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.linspace(-3.5, 3.5, 71)
    inner, _ = np.histogram(values, bins=edges)
    return np.concatenate(([np.count_nonzero(values < edges[0])], inner, [np.count_nonzero(values > edges[-1])])).astype(np.float64)


def _kernel_matrices(sigmas):
    np, _ = _deps()
    edges = np.linspace(-3.5, 3.5, 71)
    centers = np.concatenate(([-3.55], (edges[:-1] + edges[1:]) / 2.0, [3.55]))
    delta = centers[:, None] - centers[None, :]
    out = {}
    for name, sigma_by_projection in sigmas.items():
        matrices = []
        for sigma in sigma_by_projection:
            sigma = max(float(sigma), 1e-6)
            kernel = np.exp(-0.5 * (delta / sigma) ** 2)
            kernel /= kernel.sum(axis=0, keepdims=True)
            matrices.append(kernel)
        out[name] = np.stack(matrices)
    return out


def _smooth_counts(counts, kernels):
    np, _ = _deps()
    probs = np.empty_like(counts, dtype=np.float64)
    for proj in range(counts.shape[0]):
        probs[proj] = kernels[proj] @ counts[proj]
    probs = np.maximum(probs, 1e-12)
    return probs / probs.sum(axis=1, keepdims=True)


def _projected_distance(p, q) -> dict[str, Any]:
    np, _ = _deps()
    midpoint = 0.5 * (p + q)
    log2 = math.log(2.0)
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=-1) / log2
    js += 0.5 * np.sum(q * np.log(q / midpoint), axis=-1) / log2
    kl_pq = np.sum(p * np.log(p / q), axis=-1) / log2
    kl_qp = np.sum(q * np.log(q / p), axis=-1) / log2
    return {"js": js, "kl_pq": kl_pq, "kl_qp": kl_qp}


def _independent_seed_index_shuffle(actions, rng):
    np, _ = _deps()
    flat = np.asarray(actions, dtype=np.float64).reshape(len(actions), -1)
    shuffled = np.empty_like(flat)
    for feature in range(flat.shape[1]):
        shuffled[:, feature] = flat[rng.permutation(len(flat)), feature]
    return shuffled.reshape(np.asarray(actions).shape)


def _chunk_coherence_fraction(actions) -> float:
    np, _ = _deps()
    centered = actions - actions.mean(axis=0, keepdims=True)
    error_rms = float(np.sqrt(np.mean(centered**2)))
    chunk_means = actions.mean(axis=1)
    chunk_dispersion = float(np.sqrt(np.mean((chunk_means - chunk_means.mean(axis=0, keepdims=True)) ** 2)))
    return chunk_dispersion / error_rms if error_rms else 0.0


def _distribution_stability(actions, projections, center, scale, groups, kernel_sets):
    np, _ = _deps()
    projected = np.einsum("csf,pf->csp", actions.reshape(len(actions), N_SEEDS, -1), projections)
    z = (projected - center[None, None, :]) / scale[None, None, :]
    result = []
    for case_idx in range(len(actions)):
        counts = []
        for group in groups:
            group_values = z[case_idx, group, :]
            counts.append(np.stack([_histogram(group_values[:, p]) for p in range(z.shape[2])]))
        counts = np.stack(counts)
        case_metrics = {}
        for name, matrices in kernel_sets.items():
            smooth = [_smooth_counts(counts[g], matrices) for g in range(len(groups))]
            pair_rows = []
            for a, b in combinations(range(len(groups)), 2):
                dist = _projected_distance(smooth[a], smooth[b])
                pair_rows.append({metric: {"mean_over_88_projections": float(vals.mean()), "q90_over_88_projections": float(np.quantile(vals, 0.9))} for metric, vals in dist.items()})
            case_metrics[name] = {
                "group_pair_count": len(pair_rows),
                "mean_over_group_pairs": {metric: float(np.mean([row[metric]["mean_over_88_projections"] for row in pair_rows])) for metric in ("js", "kl_pq", "kl_qp")},
                "mean_pair_q90_over_projections": {metric: float(np.mean([row[metric]["q90_over_88_projections"] for row in pair_rows])) for metric in ("js", "kl_pq", "kl_qp")},
            }
        result.append(case_metrics)
    return result


def _case_metrics(actions, projections, center, scale, groups4, groups8, kernel_sets, shuffle_seed):
    np, _ = _deps()
    pair_rmse = [float(np.sqrt(np.mean((actions[a] - actions[b]) ** 2))) for a, b in combinations(range(N_SEEDS), 2)]
    within_std_rms = float(np.sqrt(np.var(actions, axis=0, ddof=0).mean()))
    chunk_means = actions.mean(axis=1)
    chunk_mean_dispersion = float(np.sqrt(np.mean((chunk_means - chunk_means.mean(axis=0, keepdims=True)) ** 2)))
    cov, rank = _covariance_and_rank(actions)
    full_mean = actions.mean(axis=0)

    def stability(groups):
        group_means, group_covs = [], []
        for indices in groups:
            block = actions[list(indices)]
            group_means.append(block.mean(axis=0).reshape(-1))
            group_covs.append(_covariance_and_rank(block)[0])
        mean_delta, cov_delta = [], []
        for a, b in combinations(range(len(groups)), 2):
            mean_delta.append(_relative_frobenius(group_means[a], group_means[b]))
            cov_delta.append(_relative_frobenius(group_covs[a], group_covs[b]))
        discrepancy = [float(np.sqrt(np.mean((actions[list(indices)].mean(axis=0) - full_mean) ** 2))) for indices in groups]
        return {
            "group_count": len(groups),
            "pair_count": len(mean_delta),
            "mean_symmetric_relative_frobenius": _stats(mean_delta),
            "covariance_symmetric_relative_frobenius": _stats(cov_delta),
            "group_mean_discrepancy_from_full16_rms_normalized": _stats(discrepancy),
            "group_mean_discrepancy_over_full16_within_state_std": _stats([v / within_std_rms for v in discrepancy]) if within_std_rms > 0 else None,
            "within_state_std_rms_is_zero": within_std_rms == 0.0,
        }

    pooled_k4 = _distribution_stability(actions[None, ...], projections, center, scale, groups4, kernel_sets)[0]
    pooled_k8 = _distribution_stability(actions[None, ...], projections, center, scale, groups8, kernel_sets)[0]
    projected = np.einsum("sf,pf->sp", actions.reshape(N_SEEDS, -1), projections)
    local_center = projected.mean(axis=0)
    local_scale_raw = projected.std(axis=0, ddof=0)
    local_scale = np.where(local_scale_raw > 1e-12, local_scale_raw, 1.0)
    local_k4 = _distribution_stability(actions[None, ...], projections, local_center, local_scale, groups4, kernel_sets)[0]
    local_k8 = _distribution_stability(actions[None, ...], projections, local_center, local_scale, groups8, kernel_sets)[0]
    conditional_std_in_pooled_units = local_scale_raw / scale
    kernel_ratio = {}
    for name in kernel_sets:
        bandwidth = float(name.removeprefix("std_"))
        ratios = bandwidth / np.maximum(conditional_std_in_pooled_units, 1e-12)
        kernel_ratio[name] = {
            "bandwidth_over_within_state_projection_std": {"median": float(np.median(ratios)), "q90": float(np.quantile(ratios, 0.9))},
            "fraction_kernel_width_exceeds_within_state_std": float(np.mean(bandwidth > conditional_std_in_pooled_units)),
        }
    rng = np.random.default_rng(shuffle_seed)
    null_coherence = [_chunk_coherence_fraction(_independent_seed_index_shuffle(actions, rng)) for _ in range(32)]
    return {
        "pairwise_seed_rmse_normalized": _stats(pair_rmse),
        "within_state_std_rms_normalized": within_std_rms,
        "seed_centered_temporal_coordinate_covariance": rank,
        "covariance_max_rank_from_16_centered_seeds": N_SEEDS - 1,
        "chunk_mean_dispersion_rms_normalized": chunk_mean_dispersion,
        "chunk_mean_dispersion_fraction_of_seed_error_rms": _chunk_coherence_fraction(actions),
        "independent_seed_index_shuffle_null": {
            "interpretation": "artificial coupling ablation: independently permute the 16 seed indices at each of 60 coordinates; preserves each scalar marginal, breaks cross-coordinate/time seed pairing; not a policy-quality test",
            "shuffle_count": 32,
            "rng_seed": int(shuffle_seed),
            "observed_chunk_mean_dispersion_fraction": _chunk_coherence_fraction(actions),
            "null_mean": float(np.mean(null_coherence)),
            "null_median": float(np.median(null_coherence)),
            "null_q90": float(np.quantile(null_coherence, 0.9)),
        },
        "k4_stability": stability(groups4),
        "k8_stability": stability(groups8),
        "k4_projected_distribution_stability_pooled_ruler": pooled_k4,
        "k8_projected_distribution_stability_pooled_ruler": pooled_k8,
        "k4_projected_distribution_stability_full16ruler": local_k4,
        "k8_projected_distribution_stability_full16ruler": local_k8,
        "pooled_kernel_vs_within_state_spread": kernel_ratio,
    }


def _load_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ids = protocol.get("calibration", {}).get("case_ids")
    if not isinstance(ids, list) or len(ids) != N_CASES or len({str(x) for x in ids}) != N_CASES:
        raise ValueError(f"protocol calibration.case_ids 必须包含 {N_CASES} 个唯一 case id。")
    return protocol


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _read_manifest(data_root: Path, protocol: dict[str, Any]):
    path = data_root / "results" / "prepare" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"source manifest 不存在：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("cases")
    case_ids = protocol["calibration"]["case_ids"]
    if not isinstance(rows, list) or len(rows) != N_CASES:
        raise ValueError(f"manifest.cases 必须包含 {N_CASES} 个 case。")
    by_id = {str(row["case_id"]): row for row in rows if isinstance(row, dict) and "case_id" in row}
    if len(by_id) != N_CASES or set(by_id) != {str(x) for x in case_ids}:
        raise ValueError("manifest.cases 与 protocol.calibration.case_ids 覆盖不一致。")
    for case_id in case_ids:
        row = by_id[str(case_id)]
        if not all(key in row for key in ("suite", "plan_row", "sampler_seeds")) or len(row["sampler_seeds"]) != N_SEEDS:
            raise ValueError(f"manifest case {case_id} 缺少 suite/plan_row 或 16 个 sampler_seeds。")
    return by_id


def _read_records(data_root: Path, protocol: dict[str, Any], manifest_by_id):
    np, torch = _deps()
    teacher_dir = data_root / "results" / "teacher"
    if not teacher_dir.is_dir():
        raise FileNotFoundError(f"teacher records 目录不存在：{teacher_dir}")
    case_ids = protocol["calibration"]["case_ids"]
    expected = {f"{case_id}_{seed}.pt" for case_id in case_ids for seed in range(N_SEEDS)}
    present = {p.name for p in teacher_dir.glob("*.pt")}
    missing, extra = sorted(expected - present), sorted(present - expected)
    if missing or extra:
        raise ValueError(f"teacher 覆盖不精确：missing={missing[:5]} (n={len(missing)}), extra={extra[:5]} (n={len(extra)})")
    actions, suites, provenance = [], {}, {"collection_job_ids": set(), "source_kinds": set()}
    source_by_seed = {str(i): {} for i in range(N_SEEDS)}
    for case_id in case_ids:
        seed_actions, case_suite = [], None
        manifest_row = manifest_by_id[str(case_id)]
        for seed_index in range(N_SEEDS):
            path = teacher_dir / f"{case_id}_{seed_index}.pt"
            record = torch.load(path, map_location="cpu")
            required = ("protocol", "case_id", "suite", "seed_index", "sampler_seed", "plan_row", "action", "counts", "source_kind", "collection_jobid")
            absent = [key for key in required if key not in record]
            if absent:
                raise ValueError(f"{path} 缺少字段：{absent}")
            if protocol.get("protocol") is not None and record["protocol"] != protocol["protocol"]:
                raise ValueError(f"{path} protocol tag 与 protocol.json 不一致。")
            if str(record["case_id"]) != str(case_id) or int(record["seed_index"]) != seed_index:
                raise ValueError(f"{path} 内 case_id/seed_index 与文件名不一致。")
            if not isinstance(record["action"], torch.Tensor):
                raise TypeError(f"{path} 的 action 必须是 torch.Tensor。")
            action = record["action"].detach().cpu()
            if action.dtype != torch.float32:
                raise TypeError(f"{path} action dtype 应为 float32，实际为 {action.dtype}")
            if tuple(action.shape) != (32, 7):
                raise ValueError(f"{path} action shape 应为 (32,7)，实际为 {tuple(action.shape)}")
            if not bool(torch.isfinite(action).all()):
                raise ValueError(f"{path} action 有非有限值。")
            suite = str(record["suite"])
            if suite != str(manifest_row["suite"]):
                raise ValueError(f"{path} suite 与 prepare/manifest.json 不一致。")
            if _json_key(record["plan_row"]) != _json_key(manifest_row["plan_row"]):
                raise ValueError(f"{path} plan_row 与 prepare/manifest.json 不一致。")
            if _json_key(record["sampler_seed"]) != _json_key(manifest_row["sampler_seeds"][seed_index]):
                raise ValueError(f"{path} sampler_seed 与 prepare/manifest.json 不一致。")
            if case_suite is not None and case_suite != suite:
                raise ValueError(f"case {case_id} 的 suite 在不同 seed 记录间不一致。")
            case_suite = suite
            jobid = str(record["collection_jobid"] or "").strip()
            if not jobid:
                raise ValueError(f"{path} collection_jobid 为空。")
            provenance["collection_job_ids"].add(jobid)
            source_kind = str(record["source_kind"])
            provenance["source_kinds"].add(source_kind)
            source_by_seed[str(seed_index)][source_kind] = source_by_seed[str(seed_index)].get(source_kind, 0) + 1
            seed_actions.append(action[:N_STEPS, :N_MOTORS].to(dtype=torch.float64).numpy())
        suites[str(case_id)] = case_suite
        actions.append(np.stack(seed_actions))
    return np.stack(actions), suites, {**{k: sorted(v) for k, v in provenance.items()}, "source_kind_counts_by_seed_index": source_by_seed}


def _validate_replay(data_root: Path) -> dict[str, Any]:
    path = data_root / "results" / "replay.json"
    if not path.is_file():
        raise FileNotFoundError(f"same-seed replay 记录不存在：{path}")
    replay = json.loads(path.read_text(encoding="utf-8"))
    cached, repeated = replay.get("cached_replays"), replay.get("same_seed_repeats")
    if replay.get("complete") is not True or not isinstance(cached, list) or len(cached) != 8 or not isinstance(repeated, list) or len(repeated) != 8:
        raise ValueError("replay.json 必须含 complete=true、8 个 cached_replays 和 8 个 same_seed_repeats。")
    maxima = []
    for row in cached:
        if not all(key in row for key in ("action_max_abs", "future_max_abs")):
            raise ValueError("cached_replays 缺少 action_max_abs/future_max_abs。")
        maxima.extend((float(row["action_max_abs"]), float(row["future_max_abs"])))
    for row in repeated:
        if "action_max_abs" not in row:
            raise ValueError("same_seed_repeats 缺少 action_max_abs。")
        maxima.append(float(row["action_max_abs"]))
    if any(not math.isfinite(value) or value != 0.0 for value in maxima):
        raise ValueError("replay.json 中全部复播 max-abs 必须有限且等于 0。")
    return {"complete": True, "cached_replay_count": len(cached), "same_seed_repeat_count": len(repeated),
            "cached_action_and_future_max_abs": 0.0, "same_seed_repeat_action_max_abs": 0.0, "source": replay}


def _aggregate_cases(case_rows, suites):
    np, _ = _deps()
    def select(rows, path):
        out = []
        for row in rows:
            value = row
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if isinstance(value, (int, float)) and math.isfinite(value):
                out.append(value)
        return _stats(out)

    metrics = {
        "within_state_std_rms_normalized": ["within_state_std_rms_normalized"],
        "pairwise_seed_rmse_case_mean_normalized": ["pairwise_seed_rmse_normalized", "mean"],
        "chunk_mean_dispersion_rms_normalized": ["chunk_mean_dispersion_rms_normalized"],
        "chunk_mean_dispersion_fraction_of_seed_error_rms": ["chunk_mean_dispersion_fraction_of_seed_error_rms"],
        "shuffle_null_observed_coherence_fraction": ["independent_seed_index_shuffle_null", "observed_chunk_mean_dispersion_fraction"],
        "shuffle_null_mean_coherence_fraction": ["independent_seed_index_shuffle_null", "null_mean"],
        "shuffle_null_q90_coherence_fraction": ["independent_seed_index_shuffle_null", "null_q90"],
        "effective_rank": ["seed_centered_temporal_coordinate_covariance", "effective_rank"],
        "numerical_rank": ["seed_centered_temporal_coordinate_covariance", "numerical_rank"],
        "k4_mean_relative_frobenius": ["k4_stability", "mean_symmetric_relative_frobenius", "mean"],
        "k4_covariance_relative_frobenius": ["k4_stability", "covariance_symmetric_relative_frobenius", "mean"],
        "k8_mean_relative_frobenius": ["k8_stability", "mean_symmetric_relative_frobenius", "mean"],
        "k8_covariance_relative_frobenius": ["k8_stability", "covariance_symmetric_relative_frobenius", "mean"],
        "k4_group_mean_discrepancy_from_full16_rms_normalized": ["k4_stability", "group_mean_discrepancy_from_full16_rms_normalized", "mean"],
        "k8_group_mean_discrepancy_from_full16_rms_normalized": ["k8_stability", "group_mean_discrepancy_from_full16_rms_normalized", "mean"],
        "k4_group_mean_discrepancy_over_within_std": ["k4_stability", "group_mean_discrepancy_over_full16_within_state_std", "mean"],
        "k8_group_mean_discrepancy_over_within_std": ["k8_stability", "group_mean_discrepancy_over_full16_within_state_std", "mean"],
    }
    grouped = {"overall": case_rows}
    for row in case_rows:
        grouped.setdefault(suites[str(row["case_id"])], []).append(row)
    result = {}
    for name, rows in grouped.items():
        item = {"case_count": len(rows), "case_level_metrics": {key: select(rows, path) for key, path in metrics.items()}}
        for k in ("k4_projected_distribution_stability_pooled_ruler", "k8_projected_distribution_stability_pooled_ruler",
                  "k4_projected_distribution_stability_full16ruler", "k8_projected_distribution_stability_full16ruler"):
            bw_names = rows[0][k].keys()
            item[k] = {}
            for bw in bw_names:
                item[k][bw] = {}
                for metric in ("js", "kl_pq", "kl_qp"):
                    item[k][bw][metric] = {
                        "mean_over_case_level_pair_means": select(rows, [k, bw, "mean_over_group_pairs", metric]),
                        "mean_over_case_level_pair_q90s": select(rows, [k, bw, "mean_pair_q90_over_projections", metric]),
                    }
        item["pooled_kernel_vs_within_state_spread"] = {}
        for bw in rows[0]["pooled_kernel_vs_within_state_spread"]:
            item["pooled_kernel_vs_within_state_spread"][bw] = {
                "bandwidth_over_within_state_std_case_median": select(rows, ["pooled_kernel_vs_within_state_spread", bw, "bandwidth_over_within_state_projection_std", "median"]),
                "bandwidth_over_within_state_std_case_q90": select(rows, ["pooled_kernel_vs_within_state_spread", bw, "bandwidth_over_within_state_projection_std", "q90"]),
                "fraction_kernel_width_exceeds_within_state_std": select(rows, ["pooled_kernel_vs_within_state_spread", bw, "fraction_kernel_width_exceeds_within_state_std"]),
            }
        result[name] = item
    return result


def _markdown(summary, suites, per_case):
    lines = [
        "# Fast-WAM original C80 teacher 多 seed 诊断",
        "",
        "本报告只描述 80 个固定 calibration case 上，每个 case 的 16 个 BF16 teacher action 样本。case 是统计单位；seed 是 case 内重复采样，不把 1,280 条记录当成独立 context。无训练、gripper 分析、held-out 或 rollout。",
        "",
        "## 数据与尺度",
        "",
        f"覆盖：{summary['coverage']['case_count']} cases × {summary['coverage']['seed_count_per_case']} seeds；前 10 步、前 6 个 motor coordinates，共 {N_STEPS * N_MOTORS} 维。动作直接采用记录中的原始 normalized representation，不做反归一化；不据此推断物理单位。",
        f"投影：{summary['projections']['count']} 条固定投影（60 axes、6 个时间均值、6 个前半减后半、16 条固定随机单位向量；RNG seed {PROJECTION_SEED}）。投影的中心/尺度由完整 80×16 calibration 样本共同确定，所以组间稳定性以该共享标尺为条件。",
        "",
        "## 按 suite 汇总",
        "",
        "以下每行先在 case 内汇总 seed/group/projected comparisons，再跨 cases 汇总；q90 也是 case-level 指标的分位数。",
        "",
        "| Suite | Cases | Within-state RMS std | Covariance effective rank (max 15) | K4 mean rel. | K4 cov. rel. | K8 mean rel. | K8 cov. rel. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    norm = summary["normalization"]
    if norm["motor_min"] is not None or norm["motor_max"] is not None:
        lines.insert(7, "协议中记录的 motor_min/motor_max 原样列于 `summary.json`；本分析不应用这些范围做反归一化。")
    for name, item in suites.items():
        if name == "overall":
            continue
        m = item["case_level_metrics"]
        lines.append("| {} | {} | {:.5g} | {:.4g} | {:.4g} | {:.4g} | {:.4g} | {:.4g} |".format(
            name, item["case_count"], m["within_state_std_rms_normalized"]["mean"], m["effective_rank"]["mean"],
            m["k4_mean_relative_frobenius"]["mean"], m["k4_covariance_relative_frobenius"]["mean"],
            m["k8_mean_relative_frobenius"]["mean"], m["k8_covariance_relative_frobenius"]["mean"]))
    lines += [
        "",
        "K4 使用四个互不重叠的 4-seed group，并比较每个 case 内全部 6 对；K8 使用 0:8 与 8:16 两组。mean 与 covariance 均采用 symmetric relative Frobenius distance。数值越小表示在当前 seed 划分下更接近；没有预注册阈值时只作描述，不生成 pass/fail。",
        "每个 case 的 K4/K8 group mean 相对 full16 mean 差异也写入 `per_case.json`：同时给出 normalized action 坐标上的 RMS 绝对差，以及除以该 case full16 within-state RMS std 的比值；分母为零时比值记为 null，绝对差仍保留。",
        "",
        "## Projected distributions",
        "",
        "共同 bins 覆盖 standardized projection 的 [-3.5, 3.5]，并保留两侧 tail bins。KL 使用 1e-12 positive floor 后重归一化，单位为 bits；JS bounded in [0,1]。仅使用预设 standardized bandwidth 0.1、0.25、0.5。结果分别报告 pooled C80 ruler 与每 case full16 mean/std ruler；后者检视同一 case 内的条件分布形状，尺度差异另由 RMS 指标呈现。所有行先按 case 聚合；q90 是 case-level mean-pair q90 的跨 case mean。",
        "",
        "| Ruler | K | Bandwidth | JS mean | JS mean pair q90 | KL P→Q mean | KL Q→P mean |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for k, label, ruler in (
        ("k4_projected_distribution_stability_pooled_ruler", "K4", "pooled"),
        ("k8_projected_distribution_stability_pooled_ruler", "K8", "pooled"),
        ("k4_projected_distribution_stability_full16ruler", "K4", "full16ruler"),
        ("k8_projected_distribution_stability_full16ruler", "K8", "full16ruler"),
    ):
        for bw, values in suites["overall"][k].items():
            lines.append("| {} | {} | {} | {:.5g} | {:.5g} | {:.5g} | {:.5g} |".format(
                ruler, label, bw, values["js"]["mean_over_case_level_pair_means"]["mean"],
                values["js"]["mean_over_case_level_pair_q90s"]["mean"],
                values["kl_pq"]["mean_over_case_level_pair_means"]["mean"],
                values["kl_qp"]["mean_over_case_level_pair_means"]["mean"]))
    lines += [
        "",
        "## Pooled kernel 与条件 spread",
        "",
        "下表比较 bandwidth 与每个 case 内 full16 projection standard deviation 的比例。比例越大，pooled-ruler 的 JS/KL 越可能受 oversmoothing 压低；因此不能仅凭 pooled JS 小就推断条件分布稳定。full16ruler 结果作为配对诊断读取，也不作为自动 gate。",
        "",
        "| Bandwidth | Case median(kernel / within-state std) | Case q90 | Mean fraction of projections where kernel > within-state std |",
        "|---|---:|---:|---:|",
    ]
    for bw, values in suites["overall"]["pooled_kernel_vs_within_state_spread"].items():
        lines.append("| {} | {:.5g} | {:.5g} | {:.5g} |".format(
            bw, values["bandwidth_over_within_state_std_case_median"]["mean"],
            values["bandwidth_over_within_state_std_case_q90"]["mean"],
            values["fraction_kernel_width_exceeds_within_state_std"]["mean"]))
    null_metrics = suites["overall"]["case_level_metrics"]
    lines += [
        "",
        "## Trajectory pairing null",
        "",
        "每个 case 都做 32 次 artificial coupling ablation：对 60 个时间/coordinate 位置分别独立重排 16 个 seed，保留每个 scalar 的边际样本，破坏同一 seed 的跨位置配对。下列是 case-level coherence fraction 的跨 case 均值；它是结构诊断，不是策略质量检验。",
        "",
        "| Observed chunk coherence | Independent-shuffle null mean | Null q90 |",
        "|---:|---:|---:|",
        "| {:.5g} | {:.5g} | {:.5g} |".format(
            null_metrics["shuffle_null_observed_coherence_fraction"]["mean"],
            null_metrics["shuffle_null_mean_coherence_fraction"]["mean"],
            null_metrics["shuffle_null_q90_coherence_fraction"]["mean"]),
        "",
        "## 解释边界",
        "",
        "- Covariance effective rank 由 16 个中心化 seed 样本估计，数值秩上限为 15；它不是 teacher 分布的真实 intrinsic dimension。",
        "- 每个 case 只有 16 次采样；投影稳定性、偏度/尾部表现或双峰结构都不能据此作为多模态证据。",
        "- pooled ruler 使用同一 calibration cohort 的全部 16 seeds，是描述性共同坐标系；K4/K8 结论条件于该 ruler。若 kernel width 大于 case 内 full16 conditional spread，pooled JS/KL 可能因 oversmoothing 偏低；应与 full16ruler 和 kernel/spread 比例配对解读。",
        "- full16ruler 对每个 case 用同一组 16 seeds 计算每条投影的均值和标准差，再比较 K4/K8 group；它衡量条件 shape，不能替代 action-scale 的 RMS mean discrepancy。没有使用 held-out 样本或 group-specific scaling。",
        "- 这是固定输入、BF16 teacher 的采样诊断，不代表量化收益、策略质量或闭环性能。",
    ]
    if summary.get("replay") is not None:
        lines += ["", "Same-seed replay 记录见 `summary.json` 的 `replay` 字段；其数值仅作为 collector replay 基线。"]
    return "\n".join(lines) + "\n"


def run_analysis(protocol_path: Path, explicit_root: Path | None = None) -> None:
    require_compute_allocation()
    np, _ = _deps()
    protocol = _load_protocol(protocol_path)
    if explicit_root is not None:
        data_root = explicit_root.resolve()
    else:
        remote_root = protocol.get("remote_root")
        data_root = Path(remote_root) if remote_root else protocol_path.parent
        if not data_root.is_absolute():
            data_root = protocol_path.parent / data_root
        data_root = data_root.resolve()
    manifest_by_id = _read_manifest(data_root, protocol)
    actions, case_suites, provenance = _read_records(data_root, protocol, manifest_by_id)
    projections = _projection_matrix()
    projected_all = np.einsum("csf,pf->csp", actions.reshape(N_CASES, N_SEEDS, -1), projections)
    ruler_center = projected_all.mean(axis=(0, 1))
    ruler_scale = projected_all.std(axis=(0, 1), ddof=0)
    safe_scale = np.where(ruler_scale > 1e-12, ruler_scale, 1.0)
    analysis_cfg = protocol.get("analysis") if isinstance(protocol.get("analysis"), dict) else {}
    bandwidths = [float(x) for x in analysis_cfg.get("only_bandwidths", analysis_cfg.get("bandwidths", [0.1, 0.25, 0.5]))]
    if bandwidths != [0.1, 0.25, 0.5]:
        raise ValueError("当前冻结协议的 standardized bandwidths 必须是 [0.1, 0.25, 0.5]。")
    sigma_by_projection = {f"std_{bandwidth}": np.full(88, bandwidth) for bandwidth in bandwidths}
    kernel_sets = _kernel_matrices(sigma_by_projection)
    groups4 = [tuple(range(i, i + 4)) for i in range(0, 16, 4)]
    groups8 = [tuple(range(i, i + 8)) for i in (0, 8)]
    case_rows = []
    for i, case_id in enumerate(protocol["calibration"]["case_ids"]):
        metrics = _case_metrics(actions[i], projections, ruler_center, safe_scale, groups4, groups8, kernel_sets, PROJECTION_SEED + i)
        case_rows.append({"case_id": case_id, "suite": case_suites[str(case_id)], **metrics})

    within = float(np.sqrt(np.mean(np.var(actions, axis=1, ddof=0))))
    case_means = actions.mean(axis=1)
    between = float(np.sqrt(np.var(case_means, axis=0, ddof=0).mean()))
    normalization_cfg = protocol.get("normalization") if isinstance(protocol.get("normalization"), dict) else {}
    normalization = {
        "action_field": "original normalized action tensor as stored",
        "selection": "action[0:10, 0:6]",
        "additional_transform": "none",
        "motor_min": protocol.get("motor_min", normalization_cfg.get("motor_min")),
        "motor_max": protocol.get("motor_max", normalization_cfg.get("motor_max")),
    }
    replay = _validate_replay(data_root)
    criteria = analysis_cfg.get("stability_criteria", analysis_cfg.get("criteria"))
    summary = {
        "complete": True,
        "pbs_jobid": os.environ["PBS_JOBID"],
        "training_updates": 0,
        "optimizer_steps": 0,
        "heldout_used": False,
        "protocol": protocol.get("protocol"),
        "coverage": {"case_count": len(case_rows), "seed_count_per_case": N_SEEDS, "teacher_record_count": N_CASES * N_SEEDS,
                     "seed_indices": list(range(N_SEEDS)), "case_ids": protocol["calibration"]["case_ids"]},
        "suites": {suite: sum(1 for value in case_suites.values() if value == suite) for suite in sorted(set(case_suites.values()))},
        "provenance": provenance,
        "normalization": normalization,
        "projections": {"count": len(projections), "axis_count": 60, "time_mean_count": 6, "first_half_minus_second_half_count": 6,
                        "fixed_random_count": 16, "rng_seed": PROJECTION_SEED, "ruler": "mean/std pooled over all 80x16 calibration samples; common to all groups",
                        "standardized_scale_min": float(safe_scale.min()), "standardized_scale_max": float(safe_scale.max())},
        "distribution_estimator": {"bins": "shared standardized [-3.5,3.5] with underflow/overflow tail bins", "positive_floor": 1e-12,
                                   "standardized_bandwidths": bandwidths, "divergence_units": "bits",
                                   "rulers": ["pooled C80 full16", "per-case full16ruler"]},
        "between_vs_within_state": {"within_state_std_rms_normalized": within, "between_case_mean_std_rms_normalized": between,
                                    "between_to_within_ratio": between / within if within else None,
                                    "between_state_definition": "coordinate-wise population std across 80 case means, then RMS; contexts are descriptive cases"},
        "predeclared_stability_criteria": criteria,
        "stability_interpretation": "descriptive only; no pass/fail criterion supplied" if criteria is None else "criteria are recorded from protocol; this analyzer does not convert them to a gate",
        "replay": replay,
    }
    suite_summary = _aggregate_cases(case_rows, case_suites)
    out_dir = data_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_case.json").write_text(json.dumps(_finite(case_rows), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "ANALYSIS.zh.md").write_text(_markdown(summary, suite_summary, case_rows), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(_finite({**summary, "aggregates": suite_summary}), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Analysis written to {out_dir}; cases={N_CASES}, seeds/case={N_SEEDS}, records={N_CASES * N_SEEDS}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, help="protocol.json path; defaults to NEW_ROOT/protocol.json")
    parser.add_argument("--root", type=Path, help="data root override; defaults to NEW_ROOT")
    args = parser.parse_args()
    if args.root:
        new_root = args.root
    elif os.environ.get("NEW_ROOT"):
        new_root = Path(os.environ["NEW_ROOT"])
    elif args.protocol:
        new_root = args.protocol.parent
    elif (Path.cwd() / "protocol.json").is_file():
        new_root = Path.cwd()
    else:
        new_root = Path(__file__).resolve().parent
    protocol_path = args.protocol or (new_root / "protocol.json")
    run_analysis(protocol_path.resolve(), new_root.resolve())


if __name__ == "__main__":
    main()
