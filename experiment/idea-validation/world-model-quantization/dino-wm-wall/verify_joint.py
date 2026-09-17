"""Independent CPU verification for joint-fidelity artifacts.

The verifier reads the frozen development workload, ``pool_metrics.jsonl``,
``episode_metrics.json``, ``summary.json``, and every local score NPZ.  It
recomputes all eight methods' pool metrics with NumPy and then reaggregates
pool -> MPC point -> episode with equal episode weight.  It does not import
the measured metric or aggregation helpers from ``joint_fidelity.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

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
METRICS = (
    "e2_pairwise_disagreement",
    "stable30_elite_overlap",
    "stable30_elite_overlap_fraction",
    "elite_mean_normalized_action_mse",
    "score_nmse",
)
JOINT_SCHEMA = "rankcal-wall-joint-fidelity-v1"
POOL_SCHEMA = "rankcal-wall-joint-fidelity-pool-v1"
ALLOCATION_MANIFEST_SCHEMA = "rankcal-wall-joint-fidelity-allocations-v1"
WORKLOAD_SCHEMA = "rankcal-wall-screen-workload-v1"
VERIFICATION_SCHEMA = "rankcal-wall-joint-fidelity-verification-v1"
EXPECTED_EPISODES = 8
EXPECTED_CANDIDATES = 300
ELITE_COUNT = 30
NMSE_EPS = 1e-12
METRIC_RTOL = 2e-5
METRIC_ATOL = 1e-7


class Verification:
    def __init__(self) -> None:
        self.errors: List[str] = []

    def fail(self, message: str) -> None:
        if len(self.errors) < 100:
            self.errors.append(message)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.fail(message)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        return pickle.load(stream)


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "pool"


def _find_summary(artifacts: Path) -> Path:
    artifacts = artifacts.resolve()
    if artifacts.is_file() and artifacts.name == "summary.json":
        return artifacts
    candidates = [artifacts / "summary.json", artifacts / "joint_fidelity" / "summary.json"]
    candidates.extend(sorted(artifacts.rglob("summary.json")) if artifacts.is_dir() else [])
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and payload.get("schema") == JOINT_SCHEMA:
            return path
    raise FileNotFoundError(f"cannot find {JOINT_SCHEMA} summary.json under {artifacts}")


def _resolve_workload(explicit: Path | None, summary: Mapping[str, Any], artifact_dir: Path) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    summary_path = summary.get("workload")
    if summary_path:
        candidates.append(Path(str(summary_path)))
    if artifact_dir.is_dir():
        candidates.extend(sorted(artifact_dir.rglob("workload.pkl")))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("development workload.pkl is required; pass --workload explicitly")


def _load_workload(path: Path, verifier: Verification, expected_pool_count: int) -> Dict[str, Dict[str, Any]]:
    payload = _load(path)
    verifier.require(isinstance(payload, Mapping), f"workload is not a mapping: {path}")
    if not isinstance(payload, Mapping):
        return {}
    verifier.require(payload.get("schema") == WORKLOAD_SCHEMA, f"unexpected workload schema: {payload.get('schema')!r}")
    verifier.require(str(payload.get("split")) == "dev", f"workload split is {payload.get('split')!r}, expected 'dev'")
    cases = payload.get("cases", [])
    verifier.require(isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)), "workload cases is not a sequence")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        return {}
    verifier.require(len(cases) == expected_pool_count, f"workload has {len(cases)} pools, expected {expected_pool_count}")
    pools: Dict[str, Dict[str, Any]] = {}
    for raw in cases:
        if not isinstance(raw, Mapping):
            verifier.fail("workload contains a non-mapping pool")
            continue
        pool = dict(raw)
        pool_id = str(pool.get("pool_id", pool.get("case_id", "")))
        if not pool_id:
            verifier.fail("workload pool has no pool_id/case_id")
            continue
        if pool_id in pools:
            verifier.fail(f"duplicate workload pool ID {pool_id}")
            continue
        episode_id = str(pool.get("episode_id"))
        verifier.require(episode_id in {f"development:{index:03d}" for index in range(EXPECTED_EPISODES)}, f"pool {pool_id} has invalid episode {episode_id}")
        verifier.require(int(pool.get("mpc_point", -1)) in (0, 1), f"pool {pool_id} has invalid mpc_point")
        verifier.require(int(pool.get("cem_iteration", -1)) in (1, 5), f"pool {pool_id} has invalid CEM iteration")
        candidates = np.asarray(pool.get("candidates", []), dtype=np.float64)
        reference = np.asarray(pool.get("reference_scores", []), dtype=np.float64).reshape(-1)
        verifier.require(candidates.shape == (EXPECTED_CANDIDATES, 5, 10), f"pool {pool_id} candidates shape is {candidates.shape}")
        verifier.require(reference.shape == (EXPECTED_CANDIDATES,), f"pool {pool_id} reference_scores shape is {reference.shape}")
        verifier.require(bool(np.isfinite(candidates).all()), f"pool {pool_id} candidates are non-finite")
        verifier.require(bool(np.isfinite(reference).all()), f"pool {pool_id} reference_scores are non-finite")
        pool["pool_id"] = pool_id
        pool["candidates"] = candidates
        pool["reference_scores"] = reference
        pools[pool_id] = pool

    by_episode: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for pool in pools.values():
        by_episode[str(pool["episode_id"])].append(pool)
    expected_episodes = {f"development:{index:03d}" for index in range(EXPECTED_EPISODES)}
    verifier.require(set(by_episode) == expected_episodes, f"workload episodes are {sorted(by_episode)}, expected {sorted(expected_episodes)}")
    for episode_id, episode_pools in sorted(by_episode.items()):
        verifier.require(len(episode_pools) in (2, 4), f"episode {episode_id} has {len(episode_pools)} pools, expected 2 or 4")
        points = {int(pool["mpc_point"]) for pool in episode_pools}
        verifier.require(points in ({0}, {0, 1}), f"episode {episode_id} points are {sorted(points)}, expected 0 or 0/1")
        for point in points:
            iterations = sorted(int(pool["cem_iteration"]) for pool in episode_pools if int(pool["mpc_point"]) == point)
            verifier.require(iterations == [1, 5], f"episode {episode_id} point {point} CEM iterations are {iterations}, expected [1, 5]")
    return pools


def _pairwise_e2(reference: np.ndarray, quantized: np.ndarray) -> float:
    first, second = np.triu_indices(reference.size, k=1)
    return float(np.mean(np.sign(reference[first] - reference[second]) != np.sign(quantized[first] - quantized[second])))


def _score_nmse(reference: np.ndarray, quantized: np.ndarray) -> float:
    return float(np.mean((quantized - reference) ** 2) / (np.mean(reference ** 2) + NMSE_EPS))


def _pool_metrics(reference: np.ndarray, quantized: np.ndarray, candidates: np.ndarray) -> Dict[str, Any]:
    reference_elite = np.argsort(reference, kind="stable")[:ELITE_COUNT]
    quantized_elite = np.argsort(quantized, kind="stable")[:ELITE_COUNT]
    overlap = int(np.intersect1d(reference_elite, quantized_elite).size)
    reference_mean = candidates[reference_elite].mean(axis=0)
    quantized_mean = candidates[quantized_elite].mean(axis=0)
    return {
        "e2_pairwise_disagreement": _pairwise_e2(reference, quantized),
        "stable30_elite_overlap": overlap,
        "stable30_elite_overlap_fraction": overlap / float(ELITE_COUNT),
        "elite_mean_normalized_action_mse": float(np.mean((quantized_mean - reference_mean) ** 2)),
        "score_nmse": _score_nmse(reference, quantized),
        "reference_elite_indices": reference_elite,
        "quantized_elite_indices": quantized_elite,
    }


def _close(actual: Any, expected: Any) -> bool:
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=METRIC_RTOL, atol=METRIC_ATOL))
    except (TypeError, ValueError):
        return False


def _load_jsonl(path: Path, verifier: Verification) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        verifier.fail(f"cannot read {path}: {exc}")
        return rows
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            verifier.fail(f"invalid JSONL row {path}:{line_number}: {exc}")
            continue
        if not isinstance(value, Mapping):
            verifier.fail(f"JSONL row {path}:{line_number} is not an object")
            continue
        rows.append(dict(value))
    return rows


def _index_score_files(
    artifact_dir: Path,
    pool_rows: Sequence[Mapping[str, Any]],
    verifier: Verification,
) -> Dict[str, Tuple[Path, Dict[str, np.ndarray]]]:
    pools_dir = artifact_dir / "pools"
    files = sorted(pools_dir.glob("*.scores.npz")) if pools_dir.is_dir() else []
    by_name = {path.name: path for path in files}
    verifier.require(bool(files), f"no local score NPZ files under {pools_dir}")
    result: Dict[str, Tuple[Path, Dict[str, np.ndarray]]] = {}
    for row in pool_rows:
        pool_id = str(row.get("pool_id"))
        expected_name = f"pool_{_safe_id(pool_id)}.scores.npz"
        path = by_name.get(expected_name)
        if path is None and row.get("score_file"):
            # Only the basename is used: downloaded artifacts can invalidate
            # the recorded absolute path while preserving the local file.
            path = by_name.get(Path(str(row["score_file"])).name)
        if path is None:
            verifier.fail(f"missing local score NPZ for pool {pool_id}")
            continue
        if pool_id in result:
            verifier.fail(f"duplicate pool metric row {pool_id}")
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                result[pool_id] = (path, {key: np.asarray(archive[key]).copy() for key in archive.files})
        except (OSError, ValueError) as exc:
            verifier.fail(f"cannot read score NPZ {path}: {exc}")
    expected_ids = {str(row.get("pool_id")) for row in pool_rows}
    verifier.require(set(result) == expected_ids, "local score NPZ IDs do not match pool metrics")
    verifier.require(len(files) == len(expected_ids), f"score NPZ count is {len(files)}, expected {len(expected_ids)}")
    return result


def _check_pool_rows(
    verifier: Verification,
    pools: Mapping[str, Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    score_index: Mapping[str, Tuple[Path, Mapping[str, np.ndarray]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    expected_keys = set(pools)
    row_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in pool_rows:
        pool_id = str(row.get("pool_id"))
        if pool_id in row_by_id:
            verifier.fail(f"duplicate pool_metrics row {pool_id}")
        row_by_id[pool_id] = row
        verifier.require(row.get("schema") == POOL_SCHEMA, f"pool {pool_id} has unexpected schema {row.get('schema')!r}")
    verifier.require(set(row_by_id) == expected_keys, "pool_metrics rows do not cover exactly workload pools")
    verifier.require(len(pool_rows) == len(pools), f"pool_metrics has {len(pool_rows)} rows, expected {len(pools)}")

    recomputed_rows: List[Dict[str, Any]] = []
    recomputed_methods: Dict[str, Dict[str, Dict[str, Any]]] = {}
    expected_npz_keys = set(METHODS) | {"reference_scores"} | {f"{method}_elite_indices" for method in METHODS}
    for pool_id, pool in pools.items():
        row = row_by_id.get(pool_id)
        score_entry = score_index.get(pool_id)
        if row is None or score_entry is None:
            continue
        score_path, arrays = score_entry
        verifier.require(set(arrays) == expected_npz_keys, f"{pool_id} score NPZ keys differ from the eight-method schema")
        reference = np.asarray(arrays.get("reference_scores", []), dtype=np.float64).reshape(-1)
        workload_reference = np.asarray(pool["reference_scores"], dtype=np.float64).reshape(-1)
        verifier.require(reference.shape == (EXPECTED_CANDIDATES,), f"{pool_id} reference_scores shape is {reference.shape}")
        verifier.require(bool(np.isfinite(reference).all()), f"{pool_id} reference_scores are non-finite")
        verifier.require(np.allclose(reference, workload_reference, rtol=0.0, atol=1e-6), f"{pool_id} score reference differs from workload reference")
        fp32 = np.asarray(arrays.get("FP32", []), dtype=np.float64).reshape(-1)
        verifier.require(np.array_equal(fp32, reference), f"{pool_id} FP32 score is not exactly the stored reference")
        methods: Dict[str, Dict[str, Any]] = {}
        for method in METHODS:
            score = np.asarray(arrays.get(method, []), dtype=np.float64).reshape(-1)
            verifier.require(score.shape == reference.shape, f"{pool_id}/{method} score shape is {score.shape}")
            verifier.require(bool(np.isfinite(score).all()), f"{pool_id}/{method} scores are non-finite")
            if score.shape != reference.shape or not np.isfinite(score).all() or reference.shape != (EXPECTED_CANDIDATES,):
                continue
            metrics = _pool_metrics(reference, score, np.asarray(pool["candidates"], dtype=np.float64))
            stored_indices = np.asarray(arrays.get(f"{method}_elite_indices", []), dtype=np.int64).reshape(-1)
            verifier.require(np.array_equal(stored_indices, metrics["quantized_elite_indices"]), f"{pool_id}/{method} stable elite indices mismatch")
            stored = row.get("methods", {}).get(method) if isinstance(row.get("methods"), Mapping) else None
            verifier.require(isinstance(stored, Mapping), f"{pool_id} pool_metrics missing {method}")
            if isinstance(stored, Mapping):
                for name in METRICS:
                    if name == "stable30_elite_overlap":
                        try:
                            verifier.require(int(stored.get(name, -1)) == int(metrics[name]), f"{pool_id}/{method} {name} mismatch")
                        except (TypeError, ValueError):
                            verifier.fail(f"{pool_id}/{method} {name} is malformed")
                    else:
                        verifier.require(_close(stored.get(name), metrics[name]), f"{pool_id}/{method} {name} mismatch: row={stored.get(name)!r}, recomputed={metrics[name]!r}")
            methods[method] = metrics
        recomputed_methods[pool_id] = methods
        recomputed_rows.append({
            "pool_id": pool_id,
            "episode_id": str(pool["episode_id"]),
            "mpc_point": int(pool["mpc_point"]),
            "cem_iteration": int(pool["cem_iteration"]),
            "score_file": str(score_path.resolve()),
            "methods": methods,
        })
    return recomputed_rows, recomputed_methods


def _aggregate(
    verifier: Verification,
    pools: Mapping[str, Mapping[str, Any]],
    recomputed_methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    by_episode_point: MutableMapping[Tuple[str, int], List[str]] = defaultdict(list)
    for pool_id, pool in pools.items():
        by_episode_point[(str(pool["episode_id"]), int(pool["mpc_point"]))].append(pool_id)
    expected_episodes = {f"development:{index:03d}" for index in range(EXPECTED_EPISODES)}
    episode_rows: List[Dict[str, Any]] = []
    for episode_id in sorted(expected_episodes):
        points = sorted(point for ep, point in by_episode_point if ep == episode_id)
        verifier.require(points in ([0], [0, 1]), f"aggregation episode {episode_id} points are {points}")
        point_values: Dict[int, Dict[str, Dict[str, float]]] = {}
        for point in points:
            pool_ids = by_episode_point[(episode_id, point)]
            verifier.require(len(pool_ids) == 2, f"aggregation episode {episode_id} point {point} has {len(pool_ids)} pools, expected CEM1/5 pair")
            point_values[point] = {}
            for method in METHODS:
                point_values[point][method] = {
                    name: float(np.mean([float(recomputed_methods[pool_id][method][name]) for pool_id in pool_ids]))
                    for name in METRICS
                }
        methods: Dict[str, Dict[str, float]] = {}
        for method in METHODS:
            methods[method] = {
                name: float(np.mean([point_values[point][method][name] for point in points]))
                for name in METRICS
            }
        episode_rows.append({"episode_id": episode_id, "pool_count": sum(len(by_episode_point[(episode_id, point)]) for point in points), "methods": methods})
    overall = {
        method: {
            name: float(np.mean([row["methods"][method][name] for row in episode_rows]))
            for name in METRICS
        }
        for method in METHODS
    }
    return episode_rows, overall


def _compare_aggregates(verifier: Verification, recorded: Any, expected: Sequence[Mapping[str, Any]], label: str) -> None:
    verifier.require(isinstance(recorded, Sequence) and not isinstance(recorded, (str, bytes)), f"{label} episodes is not a sequence")
    if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
        return
    recorded_by_id = {str(row.get("episode_id")): row for row in recorded if isinstance(row, Mapping)}
    verifier.require(len(recorded_by_id) == len(recorded), f"{label} has duplicate or malformed episode rows")
    verifier.require(set(recorded_by_id) == {str(row["episode_id"]) for row in expected}, f"{label} episode IDs differ")
    for expected_row in expected:
        episode_id = str(expected_row["episode_id"])
        actual = recorded_by_id.get(episode_id)
        if actual is None:
            continue
        verifier.require(int(actual.get("pool_count", -1)) == int(expected_row["pool_count"]), f"{label} {episode_id} pool_count mismatch")
        actual_methods = actual.get("methods")
        verifier.require(isinstance(actual_methods, Mapping), f"{label} {episode_id} methods missing")
        if not isinstance(actual_methods, Mapping):
            continue
        for method in METHODS:
            verifier.require(method in actual_methods, f"{label} {episode_id} missing method {method}")
            actual_metrics = actual_methods.get(method, {})
            for name in METRICS:
                verifier.require(_close(actual_metrics.get(name), expected_row["methods"][method][name]), f"{label} {episode_id}/{method}/{name} mismatch")


def _compare_overall(verifier: Verification, recorded: Any, expected: Mapping[str, Mapping[str, float]], label: str) -> None:
    verifier.require(isinstance(recorded, Mapping), f"{label} overall metrics is not a mapping")
    if not isinstance(recorded, Mapping):
        return
    verifier.require(set(str(key) for key in recorded) == set(METHODS), f"{label} overall methods differ")
    for method in METHODS:
        actual = recorded.get(method, {})
        for name in METRICS:
            verifier.require(_close(actual.get(name), expected[method][name]), f"{label} {method}/{name} mismatch")


def _check_identity(verifier: Verification, summary: Mapping[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    verifier.require(summary.get("fp32_restore_exact_checked") is True, "summary does not claim exact FP32 restore check")
    metadata = summary.get("mode_metadata")
    verifier.require(isinstance(metadata, Mapping), "summary mode_metadata is missing")
    if isinstance(metadata, Mapping):
        verifier.require(set(str(key) for key in metadata) == set(METHODS), "mode_metadata methods differ")
        fp32 = metadata.get("FP32", {})
        verifier.require(isinstance(fp32, Mapping) and fp32.get("execution") == "fp32_reference", "FP32 mode identity is not fp32_reference")
        verifier.require(isinstance(fp32, Mapping) and fp32.get("quantization") == [], "FP32 mode has non-empty quantization metadata")
        for method in METHODS[1:]:
            entry = metadata.get(method, {})
            verifier.require(isinstance(entry, Mapping) and entry.get("execution") == "emulation_only", f"{method} execution identity is not emulation_only")

    manifest_path = artifact_dir / "allocation_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = _load(manifest_path)
        except (OSError, ValueError, pickle.PickleError) as exc:
            verifier.fail(f"cannot load allocation_manifest.json: {exc}")
            manifest = {}
        verifier.require(isinstance(manifest, Mapping), "allocation_manifest is not a mapping")
        if isinstance(manifest, Mapping):
            verifier.require(manifest.get("schema") == ALLOCATION_MANIFEST_SCHEMA, f"allocation manifest schema is {manifest.get('schema')!r}")
            verifier.require(manifest.get("fit_split") == "calibration", "allocation manifest fit_split is not calibration")
            verifier.require(manifest.get("development_maps_refit") is False, "allocation manifest allows development refit")
            verifier.require(manifest.get("w8_quota") == {"encoder": 3, "predictor": 2}, "allocation manifest W8 quota differs")
            methods = manifest.get("methods")
            allocation_methods = {"RankCal", "LocalMSE", "ScoreError", "Random1", "Random2"}
            verifier.require(isinstance(methods, Mapping) and set(str(key) for key in methods) == allocation_methods, "allocation manifest methods differ")
            if isinstance(methods, Mapping):
                metadata = summary.get("mode_metadata", {})
                map_fingerprints = set()
                map_costs = set()
                for method in sorted(allocation_methods):
                    entry = methods.get(method, {})
                    mode = metadata.get(method, {}) if isinstance(metadata, Mapping) else {}
                    bits = entry.get("bits") if isinstance(entry, Mapping) else None
                    mode_bits = mode.get("bits") if isinstance(mode, Mapping) else None
                    verifier.require(isinstance(bits, Mapping), f"allocation manifest {method} bits are missing")
                    verifier.require(bits == mode_bits, f"allocation manifest {method} differs from mode_metadata")
                    if isinstance(entry, Mapping):
                        verifier.require(entry.get("w8_counts") == {"encoder": 3, "predictor": 2}, f"allocation manifest {method} W8 counts differ")
                        cost = entry.get("cost")
                        if isinstance(cost, Mapping):
                            map_costs.add((cost.get("logical_weight_bytes"), cost.get("weight_bits")))
                        if isinstance(bits, Mapping):
                            map_fingerprints.add(tuple(sorted((str(key), int(value)) for key, value in bits.items())))
                verifier.require(len(map_fingerprints) == len(allocation_methods), "allocation manifest contains duplicate maps")
                verifier.require(len(map_costs) == 1, "allocation manifest method costs are not common")
    else:
        verifier.fail(f"missing allocation_manifest.json under {artifact_dir}")
    return {"fp32_restore_exact_checked": summary.get("fp32_restore_exact_checked"), "allocation_manifest": str(manifest_path.resolve())}


def _verify(args: argparse.Namespace) -> Dict[str, Any]:
    verifier = Verification()
    artifacts = args.artifacts.resolve()
    summary_path = _find_summary(artifacts)
    artifact_dir = summary_path.parent
    summary = _load(summary_path)
    if not isinstance(summary, Mapping):
        raise ValueError(f"joint summary is not a mapping: {summary_path}")
    verifier.require(summary.get("schema") == JOINT_SCHEMA, f"unexpected joint schema: {summary.get('schema')!r}")
    verifier.require(summary.get("status") == "complete", f"joint status is {summary.get('status')!r}")
    verifier.require(summary.get("split") == "development", f"joint split is {summary.get('split')!r}")
    verifier.require(tuple(summary.get("methods", ())) == METHODS, "joint method order differs from the eight-method protocol")

    workload_path = _resolve_workload(args.workload, summary, artifacts)
    pools = _load_workload(workload_path, verifier, args.expected_pools)
    verifier.require(int(summary.get("pool_count", -1)) == len(pools), "summary pool_count disagrees with workload")
    verifier.require(int(summary.get("episode_count", -1)) == EXPECTED_EPISODES, "summary episode_count is not 8")
    pool_metrics_path = artifact_dir / "pool_metrics.jsonl"
    pool_rows = _load_jsonl(pool_metrics_path, verifier)
    score_index = _index_score_files(artifact_dir, pool_rows, verifier)
    recomputed_rows, recomputed_methods = _check_pool_rows(verifier, pools, pool_rows, score_index)
    expected_episodes, overall = _aggregate(verifier, pools, recomputed_methods)

    episode_metrics_path = artifact_dir / "episode_metrics.json"
    episode_payload = _load(episode_metrics_path) if episode_metrics_path.is_file() else {}
    verifier.require(isinstance(episode_payload, Mapping), "episode_metrics.json is not a mapping")
    if isinstance(episode_payload, Mapping):
        _compare_aggregates(verifier, episode_payload.get("episodes", []), expected_episodes, "episode_metrics")
        _compare_overall(verifier, episode_payload.get("overall_equal_episode_weight", {}), overall, "episode_metrics")
    _compare_overall(verifier, summary.get("overall_equal_episode_weight", {}), overall, "summary")
    identity = _check_identity(verifier, summary, artifact_dir)

    report = {
        "schema": VERIFICATION_SCHEMA,
        "status": "complete" if not verifier.errors else "failed",
        "artifacts": str(artifacts),
        "joint_summary": str(summary_path.resolve()),
        "workload": str(workload_path.resolve()),
        "pool_count": len(pools),
        "expected_pool_count": args.expected_pools,
        "episode_count": len(expected_episodes),
        "methods": list(METHODS),
        "pool_score_files_checked": len(score_index),
        "pool_metric_rows_checked": len(recomputed_rows),
        "metric_formulas": {
            "e2_pairwise_disagreement": "mean(sign(ref_i-ref_j) != sign(q_i-q_j)) for i<j; exact zero signs retained",
            "stable30_elite_overlap": "intersection size of stable argsort(ref)[:30] and stable argsort(q)[:30]",
            "elite_mean_normalized_action_mse": "mean((mean(candidates[q_elite])-mean(candidates[ref_elite]))^2), matching runner field definition",
            "score_nmse": "mean((q-ref)^2)/(mean(ref^2)+1e-12)",
        },
        "metric_tolerance": {"rtol": METRIC_RTOL, "atol": METRIC_ATOL, "stable30_elite_overlap": "exact integer"},
        "aggregation": "mean pools within MPC point, mean visited points within episode, then equal-weight mean across episodes",
        "identity_check": identity,
        "overall_recomputed": overall,
        "errors": verifier.errors,
        "error_count": len(verifier.errors),
    }
    output = args.output.resolve()
    output_path = output if output.suffix.lower() == ".json" else output / "jointverification.json"
    _write_json(output_path, report)
    print(json.dumps({"status": report["status"], "pool_count": len(pools), "methods": list(METHODS), "error_count": len(verifier.errors), "output": str(output_path)}, indent=2), flush=True)
    if verifier.errors:
        raise RuntimeError(f"joint verification found {len(verifier.errors)} error(s); see {output_path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, required=True, help="joint-fidelity job directory or directory containing its summary.json")
    parser.add_argument("--workload", type=Path, default=None, help="FP32 development workload.pkl; defaults to summary path or artifact search")
    parser.add_argument("--output", type=Path, required=True, help="output JSON path or directory for jointverification.json")
    parser.add_argument("--expected-pools", type=int, default=26, help="expected frozen development pool count (default: 26)")
    return parser.parse_args()


def main() -> None:
    _verify(_parse_args())


if __name__ == "__main__":
    main()
