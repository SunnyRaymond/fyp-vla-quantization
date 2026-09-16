"""Independent CPU audit for RankCal/Wall probe artifacts.

The verifier reads ``probes.pkl`` and each per-pool ``.scores.npz`` produced by
``probe_runner.py``.  It deliberately reimplements pairwise E2 disagreement,
planner score NMSE, and the pool-to-point-to-episode site aggregation with NumPy;
none of the measured probe metric or aggregation functions are imported.
Optional calibration allocations are also re-ranked independently from the
stored site metrics and checked for the fixed 3-encoder/2-predictor W8 quota
and equal logical cost.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


PROBE_SCHEMA = "rankcal-wall-calibration-probes-v1"
SCREEN_WORKLOAD_SCHEMA = "rankcal-wall-screen-workload-v1"
EXPECTED_GROUPS = 18
EXPECTED_ENCODER_GROUPS = 12
EXPECTED_PREDICTOR_GROUPS = 6
BITS = (4, 8)
EXPECTED_EPISODES = 8
EXPECTED_CANDIDATES = 300
EXPECTED_CANDIDATE_SHAPE = (300, 5, 10)
EXPECTED_PAIR_COUNT = EXPECTED_CANDIDATES * (EXPECTED_CANDIDATES - 1) // 2
EXPECTED_POOLS_PER_EPISODE = (2, 4)
W8_QUOTA = {"encoder": 3, "predictor": 2}
SIGNALS = ("rank_disagreement", "local_block_output_nmse", "planner_score_nmse")
NMSE_EPS = 1e-12
VERIFICATION_SCHEMA = "rankcal-wall-probe-verification-v1"


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
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        return pickle.load(stream)


def _find_probes_dir(artifacts: Path) -> Tuple[Path, Path]:
    artifacts = artifacts.resolve()
    if artifacts.is_file():
        return artifacts.parent, artifacts
    candidates = [
        (artifacts / "probes", artifacts / "probes" / "probes.pkl"),
        (artifacts, artifacts / "probes.pkl"),
    ]
    for directory, probe_path in candidates:
        if probe_path.is_file():
            return directory, probe_path
    raise FileNotFoundError(f"cannot find probes.pkl under {artifacts}")


def _pool_meta(probe: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    pools = probe.get("pools")
    if isinstance(pools, Sequence) and not isinstance(pools, (str, bytes)):
        return [dict(pool) for pool in pools if isinstance(pool, Mapping)]
    # Older artifacts did not include a pool manifest.  Derive identity only
    # from row metadata; all score vectors are still independently checked.
    seen: Dict[str, Dict[str, Any]] = {}
    for row in records:
        pool_id = str(row.get("pool_id"))
        seen.setdefault(pool_id, {
            "pool_id": pool_id,
            "episode_id": str(row.get("episode_id")),
            "mpc_point": int(row.get("mpc_point", -1)),
            "cem_iteration": int(row.get("cem_iteration", -1)),
            "candidate_shape": [int(row.get("candidate_count", -1))],
        })
    return list(seen.values())


def _pairwise_e2(reference: Sequence[float], quantized: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    quant = np.asarray(quantized, dtype=np.float64).reshape(-1)
    if ref.shape != quant.shape or ref.size < 2:
        raise ValueError("score vectors must have the same length >= 2")
    first, second = np.triu_indices(ref.size, k=1)
    return float(np.mean(np.sign(ref[first] - ref[second]) != np.sign(quant[first] - quant[second])))


def _score_nmse(reference: Sequence[float], quantized: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    quant = np.asarray(quantized, dtype=np.float64).reshape(-1)
    if ref.shape != quant.shape:
        raise ValueError("score vectors must have the same shape")
    return float(np.mean((quant - ref) ** 2) / (np.mean(ref ** 2) + NMSE_EPS))


def _close(actual: Any, expected: Any, atol: float = 1e-8) -> bool:
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return False


def _expected_episode_ids(split: str) -> set[str]:
    prefix = "calibration" if split == "cal" else "development"
    return {f"{prefix}:{index:03d}" for index in range(EXPECTED_EPISODES)}


def _validate_pool_layout(verifier: Verification, split: str, pools: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pool_ids = [str(pool.get("pool_id")) for pool in pools]
    verifier.require(len(pool_ids) == len(set(pool_ids)), "pool manifest contains duplicate pool IDs")
    by_episode: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for pool in pools:
        pool_id = str(pool.get("pool_id"))
        episode_id = str(pool.get("episode_id"))
        by_episode[episode_id].append(dict(pool))
        shape = tuple(pool.get("candidate_shape", ()))
        if shape != EXPECTED_CANDIDATE_SHAPE:
            verifier.fail(
                f"pool {pool_id} candidate_shape is {shape}, expected {EXPECTED_CANDIDATE_SHAPE}"
            )
        verifier.require(int(pool.get("mpc_point", -1)) in (0, 1), f"pool {pool_id} has invalid mpc_point")
        verifier.require(int(pool.get("cem_iteration", -1)) in (1, 5), f"pool {pool_id} has invalid CEM iteration")
    expected_episodes = _expected_episode_ids(split)
    verifier.require(set(by_episode) == expected_episodes, f"pool episodes differ from expected {sorted(expected_episodes)}")
    for episode_id, episode_pools in sorted(by_episode.items()):
        verifier.require(len(episode_pools) in EXPECTED_POOLS_PER_EPISODE, f"episode {episode_id} has {len(episode_pools)} pools, expected 2 or 4")
        points = {int(pool.get("mpc_point", -1)) for pool in episode_pools}
        verifier.require(points in ({0}, {0, 1}), f"episode {episode_id} points are {sorted(points)}, expected {0} or {0, 1}")
        for point in sorted(points):
            iterations = sorted(int(pool.get("cem_iteration", -1)) for pool in episode_pools if int(pool.get("mpc_point", -1)) == point)
            verifier.require(iterations == [1, 5], f"episode {episode_id} point {point} has CEM iterations {iterations}, expected [1, 5]")
    return dict(by_episode)


def _validate_groups(verifier: Verification, groups: Any) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        verifier.fail("probe artifact groups is not a sequence")
        return [], {}
    copied = [dict(group) for group in groups if isinstance(group, Mapping)]
    verifier.require(len(copied) == EXPECTED_GROUPS, f"expected {EXPECTED_GROUPS} groups, got {len(copied)}")
    ids = [str(group.get("group_id")) for group in copied]
    verifier.require(len(ids) == len(set(ids)), "group IDs are not unique")
    families = {group_id: str(group.get("family")) for group_id, group in zip(ids, copied)}
    verifier.require(sum(family == "encoder" for family in families.values()) == EXPECTED_ENCODER_GROUPS, "group manifest lacks 12 encoder groups")
    verifier.require(sum(family == "predictor" for family in families.values()) == EXPECTED_PREDICTOR_GROUPS, "group manifest lacks 6 predictor groups")
    return copied, families


def _index_score_files(probes_dir: Path, verifier: Verification) -> Dict[str, Tuple[Path, Mapping[str, np.ndarray]]]:
    """Index scores through local checkpoint payloads, never remote paths.

    A Windows tar extraction may replace ``:`` in a filename, while the
    checkpoint payload still retains the original pool ID.  The payload is
    therefore the identity source and the sibling ``with_suffix`` path is the
    only score path used.
    """
    checkpoint_dir = probes_dir / "pool_checkpoints"
    result: Dict[str, Tuple[Path, Mapping[str, np.ndarray]]] = {}
    checkpoints = sorted(checkpoint_dir.glob("*.pkl"))
    for checkpoint in checkpoints:
        try:
            payload = _load(checkpoint)
            pool_id = str(payload["pool"]["pool_id"])
        except (OSError, KeyError, TypeError, ValueError, pickle.PickleError) as exc:
            verifier.fail(f"invalid pool checkpoint identity {checkpoint}: {exc}")
            continue
        if pool_id in result:
            verifier.fail(f"duplicate local checkpoint pool ID {pool_id}")
            continue
        score_path = checkpoint.with_suffix(".scores.npz")
        if not score_path.is_file():
            verifier.fail(f"missing sibling score file for checkpoint {checkpoint}")
            continue
        try:
            with np.load(score_path, allow_pickle=False) as archive:
                result[pool_id] = (score_path, {key: np.asarray(archive[key]).copy() for key in archive.files})
        except (OSError, ValueError) as exc:
            verifier.fail(f"cannot read score file {score_path}: {exc}")
    return result


def _check_scores(
    verifier: Verification,
    score_index: Mapping[str, Tuple[Path, Mapping[str, np.ndarray]]],
    pools: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    row_lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    expected_keys = {
        (str(pool.get("pool_id")), str(group.get("group_id")), bits)
        for pool in pools for group in groups for bits in BITS
    }
    for row in records:
        if not isinstance(row, Mapping):
            verifier.fail("probe records contains a non-mapping row")
            continue
        key = (str(row.get("pool_id")), str(row.get("group_id")), int(row.get("bits", -1)))
        if key in row_lookup:
            verifier.fail(f"duplicate probe row {key}")
        row_lookup[key] = dict(row)
    verifier.require(set(row_lookup) == expected_keys, "probe rows do not cover exactly pool x group x bit combinations")

    checked_pool_rows: List[Dict[str, Any]] = []
    pool_by_id = {str(pool.get("pool_id")): pool for pool in pools}
    group_by_id = {str(group.get("group_id")): group for group in groups}
    verifier.require(
        set(score_index) == set(pool_by_id),
        "local checkpoint/score IDs do not exactly match the pool manifest",
    )
    for pool_id, pool in pool_by_id.items():
        score_entry = score_index.get(pool_id)
        if score_entry is None:
            verifier.fail(f"missing local checkpoint/score file for {pool_id}")
            continue
        score_path, scores = score_entry
        reference = np.asarray(scores.get("reference_scores", []), dtype=np.float64).reshape(-1)
        verifier.require(reference.shape == (EXPECTED_CANDIDATES,), f"{pool_id} reference_scores shape is {reference.shape}")
        verifier.require(bool(np.isfinite(reference).all()), f"{pool_id} reference_scores are non-finite")
        pool_checked = {
            "pool_id": pool_id,
            "episode_id": str(pool.get("episode_id")),
            "score_file": str(score_path.resolve()),
            "reference_scores": reference,
            "methods": {},
        }
        used_score_keys = {"reference_scores"}
        expected_score_keys = {
            "reference_scores",
            *{
                f"g{int(group.get('index', -1)):02d}_{group.get('family')}_W{bits}"
                for group in groups
                for bits in BITS
            },
        }
        verifier.require(
            set(scores) == expected_score_keys,
            f"{pool_id} score keys do not exactly cover reference plus 18 groups x 2 bits",
        )
        for group_id, group in group_by_id.items():
            for bits in BITS:
                key = (pool_id, group_id, bits)
                row = row_lookup.get(key)
                if row is None:
                    verifier.fail(f"missing row {key}")
                    continue
                verifier.require(str(row.get("family")) == str(group.get("family")), f"row family mismatch for {key}")
                verifier.require(int(row.get("site_index", -1)) == int(group.get("index", -2)), f"row site_index mismatch for {key}")
                verifier.require(str(row.get("episode_id")) == str(pool.get("episode_id")), f"row episode mismatch for {key}")
                verifier.require(int(row.get("mpc_point", -1)) == int(pool.get("mpc_point", -2)), f"row mpc_point mismatch for {key}")
                verifier.require(int(row.get("cem_iteration", -1)) == int(pool.get("cem_iteration", -2)), f"row CEM iteration mismatch for {key}")
                verifier.require(int(row.get("candidate_count", -1)) == EXPECTED_CANDIDATES, f"row candidate_count mismatch for {key}")
                verifier.require(int(row.get("pair_count", -1)) == EXPECTED_PAIR_COUNT, f"row pair_count mismatch for {key}")
                verifier.require(bool(row.get("finite")), f"row finite=false for {key}")
                for signal in SIGNALS:
                    try:
                        verifier.require(math.isfinite(float(row.get(signal))), f"row {key} signal {signal} is non-finite")
                    except (TypeError, ValueError):
                        verifier.fail(f"row {key} signal {signal} is malformed")
                score_key = str(row.get("score_vector_key", ""))
                expected_score_key = f"g{int(group.get('index', -1)):02d}_{group.get('family')}_W{bits}"
                verifier.require(
                    score_key == expected_score_key,
                    f"score key mismatch for {key}: row={score_key!r} expected={expected_score_key!r}",
                )
                used_score_keys.add(score_key)
                quantized = np.asarray(scores.get(score_key, []), dtype=np.float64).reshape(-1)
                verifier.require(score_key in scores, f"{pool_id} missing score vector {score_key}")
                verifier.require(quantized.shape == reference.shape, f"{pool_id}/{group_id}/W{bits} score shape is {quantized.shape}")
                if quantized.shape == reference.shape and np.isfinite(quantized).all() and reference.shape == (EXPECTED_CANDIDATES,):
                    recomputed_e2 = _pairwise_e2(reference, quantized)
                    recomputed_nmse = _score_nmse(reference, quantized)
                    verifier.require(_close(row.get("rank_disagreement"), recomputed_e2), f"E2 mismatch for {key}: row={row.get('rank_disagreement')} recomputed={recomputed_e2}")
                    verifier.require(_close(row.get("planner_score_nmse"), recomputed_nmse), f"score NMSE mismatch for {key}: row={row.get('planner_score_nmse')} recomputed={recomputed_nmse}")
                    pool_checked["methods"][f"{group_id}/W{bits}"] = {
                        "group_id": group_id,
                        "bits": bits,
                        "rank_disagreement": recomputed_e2,
                        "planner_score_nmse": recomputed_nmse,
                        "local_block_output_nmse": float(row["local_block_output_nmse"]),
                    }
        unexpected = sorted(key for key in scores if key not in used_score_keys and key != "reference_scores")
        verifier.require(not unexpected, f"{pool_id} score file has unreferenced vectors: {unexpected}")
        checked_pool_rows.append(pool_checked)
    return checked_pool_rows


def _reaggregate(
    verifier: Verification,
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    pools_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    expected_episode_ids = set(pools_by_episode)
    by_site_point_episode: MutableMapping[Tuple[str, int, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("group_id")),
            int(row.get("bits", -1)),
            str(row.get("episode_id")),
            int(row.get("mpc_point", -1)),
        )
        by_site_point_episode[key].append(row)
    expected_keys = {(str(group.get("group_id")), bits, episode_id) for group in groups for bits in BITS for episode_id in expected_episode_ids}
    verifier.require(
        {(group_id, bits, episode_id) for group_id, bits, episode_id, _ in by_site_point_episode}
        == expected_keys,
        "site aggregation input does not cover every group x bit x episode",
    )
    site_metrics: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for group in groups:
        group_id = str(group.get("group_id"))
        for bits in BITS:
            episode_values = []
            for episode_id in sorted(expected_episode_ids):
                expected_points = {
                    int(pool.get("mpc_point", -1))
                    for pool in pools_by_episode.get(episode_id, [])
                }
                point_values = []
                for point in sorted(expected_points):
                    point_rows = by_site_point_episode.get((group_id, bits, episode_id, point), [])
                    verifier.require(
                        bool(point_rows),
                        f"no probe rows for {group_id}/W{bits}/{episode_id}/point{point}",
                    )
                    point_values.append(
                        {
                            signal: float(np.mean([float(row[signal]) for row in point_rows]))
                            for signal in SIGNALS
                        }
                    )
                actual_points = {
                    point
                    for group_key, bit_key, ep_key, point in by_site_point_episode
                    if group_key == group_id and bit_key == bits and ep_key == episode_id
                }
                verifier.require(
                    actual_points == expected_points,
                    f"point aggregation mismatch for {group_id}/W{bits}/{episode_id}",
                )
                episode_values.append(
                    {
                        signal: float(np.mean([value[signal] for value in point_values]))
                        for signal in SIGNALS
                    }
                    if point_values
                    else {signal: float("nan") for signal in SIGNALS}
                )
            aggregate = {
                "group_id": group_id,
                "bits": bits,
                "episode_count": len(episode_values),
                "pool_count": sum(
                    sum(
                        len(by_site_point_episode.get((group_id, bits, episode_id, point), []))
                        for point in {
                            int(pool.get("mpc_point", -1))
                            for pool in pools_by_episode.get(episode_id, [])
                        }
                    )
                    for episode_id in expected_episode_ids
                ),
            }
            for signal in SIGNALS:
                aggregate[signal] = float(np.mean([value[signal] for value in episode_values]))
            site_metrics[(group_id, bits)] = aggregate
    return site_metrics


def _compare_site_metrics(
    verifier: Verification,
    recorded: Any,
    recomputed: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> None:
    if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
        verifier.fail("probe artifact site_metrics is not a sequence")
        return
    recorded_lookup: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for row in recorded:
        if not isinstance(row, Mapping):
            verifier.fail("site_metrics contains a non-mapping row")
            continue
        key = (str(row.get("group_id")), int(row.get("bits", -1)))
        if key in recorded_lookup:
            verifier.fail(f"duplicate site_metrics row {key}")
        recorded_lookup[key] = row
    verifier.require(set(recorded_lookup) == set(recomputed), "site_metrics does not cover exactly 18 groups x 2 bits")
    for key, expected in recomputed.items():
        actual = recorded_lookup.get(key)
        if actual is None:
            continue
        verifier.require(int(actual.get("episode_count", -1)) == EXPECTED_EPISODES, f"site_metrics episode_count mismatch for {key}")
        expected_pool_count = int(expected.get("pool_count", -1))
        verifier.require(int(actual.get("pool_count", -1)) == expected_pool_count, f"site_metrics pool_count mismatch for {key}")
        for signal in SIGNALS:
            verifier.require(_close(actual.get(signal), expected[signal]), f"site_metrics {signal} mismatch for {key}: row={actual.get(signal)} recomputed={expected[signal]}")


def _group_cost(group: Mapping[str, Any], bits: int) -> int:
    field = f"logical_weight_bytes_W{bits}"
    if field in group:
        return int(group[field])
    return int(math.ceil(int(group["numel"]) * bits / 8) + int(group.get("scale_count", 0)) * 4)


def _allocation_cost(groups: Sequence[Mapping[str, Any]], bits: Mapping[str, int]) -> Dict[str, int]:
    return {
        "logical_weight_bytes": sum(_group_cost(group, int(bits[str(group["group_id"])])) for group in groups),
        "weight_bits": sum(int(group["numel"]) * int(bits[str(group["group_id"])]) for group in groups),
    }


def _extract_method_payloads(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    methods = payload.get("methods")
    if isinstance(methods, Mapping):
        result.update({str(name): value for name, value in methods.items() if isinstance(value, Mapping)})
    allocations = payload.get("allocations")
    if isinstance(allocations, Sequence) and not isinstance(allocations, (str, bytes)):
        for item in allocations:
            if isinstance(item, Mapping):
                name = str(item.get("method", item.get("name", "")))
                if name:
                    result.setdefault(name, item)
    return result


def _allocation_bits(payload: Mapping[str, Any], group_ids: Sequence[str]) -> Dict[str, int]:
    raw = payload.get("allocation", payload.get("bits"))
    if not isinstance(raw, Mapping):
        raise ValueError("allocation entry has no allocation/bits mapping")
    result = {}
    for key, value in raw.items():
        key = str(key)
        if isinstance(value, Mapping):
            value = value.get("bits")
        result[key] = int(value)
    if set(result) != set(group_ids) or any(bits not in BITS for bits in result.values()):
        raise ValueError("allocation mapping does not exactly cover runtime groups with W4/W8")
    return result


def _select_from_site_metrics(
    site_metrics: Mapping[Tuple[str, int], Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    signal: str,
) -> List[str]:
    selected = []
    for family, quota in W8_QUOTA.items():
        ranking = []
        for group in groups:
            if str(group.get("family")) != family:
                continue
            site = str(group["group_id"])
            benefit = float(site_metrics[(site, 4)][signal]) - float(site_metrics[(site, 8)][signal])
            if not math.isfinite(benefit):
                raise ValueError(f"non-finite allocation signal for {site}")
            ranking.append((-benefit, site))
        ranking.sort(key=lambda item: (item[0], item[1]))
        selected.extend(site for _, site in ranking[:quota])
    return sorted(selected)


def _find_allocation_file(artifacts: Path, probes_dir: Path) -> Path | None:
    candidates = [
        probes_dir / "allocations.json",
        probes_dir.parent / "allocations.json",
        artifacts.resolve() / "allocations.json",
        artifacts.resolve() / "allocations" / "allocations.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _verify_allocations(
    verifier: Verification,
    allocation_path: Path,
    groups: Sequence[Mapping[str, Any]],
    recomputed_site_metrics: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = _load(allocation_path)
    if not isinstance(payload, Mapping):
        verifier.fail(f"allocation artifact is not a mapping: {allocation_path}")
        return {"status": "failed", "path": str(allocation_path.resolve())}
    methods = _extract_method_payloads(payload)
    expected_methods = {"RankCal", "LocalMSE", "ScoreError", "Random1", "Random2"}
    verifier.require(set(methods) == expected_methods, f"allocation methods are {sorted(methods)}, expected {sorted(expected_methods)}")
    group_ids = [str(group["group_id"]) for group in groups]
    selected_maps: Dict[str, List[str]] = {}
    costs: Dict[str, Dict[str, int]] = {}
    signal_by_method = {"RankCal": "rank_disagreement", "LocalMSE": "local_block_output_nmse", "ScoreError": "planner_score_nmse"}
    for method in sorted(expected_methods):
        entry = methods.get(method)
        if entry is None:
            continue
        try:
            bits = _allocation_bits(entry, group_ids)
            cost = _allocation_cost(groups, bits)
        except (KeyError, TypeError, ValueError) as exc:
            verifier.fail(f"invalid allocation {method}: {exc}")
            continue
        family_counts = {
            family: sum(bits[str(group["group_id"])] == 8 for group in groups if str(group["family"]) == family)
            for family in W8_QUOTA
        }
        verifier.require(family_counts == W8_QUOTA, f"allocation {method} W8 quota is {family_counts}, expected {W8_QUOTA}")
        selected = sorted(site for site, value in bits.items() if value == 8)
        selected_maps[method] = selected
        costs[method] = cost
        if "selected_sites" in entry:
            verifier.require(sorted(str(site) for site in entry["selected_sites"]) == selected, f"allocation {method} selected_sites disagrees with bits")
        if isinstance(entry.get("cost"), Mapping):
            verifier.require(int(entry["cost"].get("logical_weight_bytes", -1)) == cost["logical_weight_bytes"], f"allocation {method} logical cost mismatch")
            verifier.require(int(entry["cost"].get("weight_bits", -1)) == cost["weight_bits"], f"allocation {method} weight cost mismatch")
        if method in signal_by_method:
            expected_selected = _select_from_site_metrics(recomputed_site_metrics, groups, signal_by_method[method])
            verifier.require(selected == expected_selected, f"allocation {method} does not match independent W4-W8 selection")
    signatures = {(cost["logical_weight_bytes"], cost["weight_bits"]) for cost in costs.values()}
    verifier.require(len(signatures) == 1 and len(costs) == len(expected_methods), f"allocation maps do not have one common cost: {costs}")
    verifier.require(selected_maps.get("Random1") != selected_maps.get("Random2"), "Random1 and Random2 maps are not distinct")
    verifier.require(len({tuple(value) for value in selected_maps.values()}) == len(selected_maps), "allocation maps contain duplicates")
    return {
        "status": "verified" if not verifier.errors else "checked_with_errors",
        "path": str(allocation_path.resolve()),
        "methods": sorted(methods),
        "selected_sites": selected_maps,
        "computed_costs": costs,
        "w8_quota": W8_QUOTA,
        "independent_signal_selection_checked": True,
        "random_maps_distinct": selected_maps.get("Random1") != selected_maps.get("Random2"),
    }


def _verify(args: argparse.Namespace) -> Dict[str, Any]:
    verifier = Verification()
    artifacts = args.artifacts.resolve()
    probes_dir, probe_path = _find_probes_dir(artifacts)
    payload = _load(probe_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"probe artifact is not a mapping: {probe_path}")
    verifier.require(payload.get("schema") == PROBE_SCHEMA, f"unexpected probe schema: {payload.get('schema')!r}")
    verifier.require(str(payload.get("split")) == args.split, f"probe split {payload.get('split')!r} != requested {args.split!r}")
    if payload.get("source_workload_schema") is not None:
        verifier.require(payload.get("source_workload_schema") == SCREEN_WORKLOAD_SCHEMA, "probe source workload schema is not screen workload v1")
    records = payload.get("records", [])
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        records = []
        verifier.fail("probe artifact records is not a sequence")
    groups, family_by_id = _validate_groups(verifier, payload.get("groups", []))
    pools = _pool_meta(payload, records)
    pools_by_episode = _validate_pool_layout(verifier, args.split, pools)
    verifier.require(int(payload.get("pool_count", len(pools))) == len(pools), "probe pool_count disagrees with pool manifest")
    verifier.require(int(payload.get("episode_count", len(pools_by_episode))) == len(pools_by_episode), "probe episode_count disagrees with pool manifest")
    verifier.require(int(payload.get("probe_record_count", len(records))) == len(records), "probe_record_count disagrees with records")
    expected_record_count = len(pools) * EXPECTED_GROUPS * len(BITS)
    verifier.require(len(records) == expected_record_count, f"expected {expected_record_count} probe rows, got {len(records)}")
    score_index = _index_score_files(probes_dir, verifier)
    checked_pools = _check_scores(verifier, score_index, pools, groups, records)
    recomputed_site_metrics = _reaggregate(verifier, records, groups, pools_by_episode)
    _compare_site_metrics(verifier, payload.get("site_metrics", []), recomputed_site_metrics)

    allocation_path = _find_allocation_file(artifacts, probes_dir) if args.split == "cal" else None
    allocation_check: Dict[str, Any]
    if allocation_path is None:
        allocation_check = {"status": "not_present"}
    else:
        allocation_check = _verify_allocations(verifier, allocation_path, groups, recomputed_site_metrics)
    report = {
        "schema": VERIFICATION_SCHEMA,
        "status": "complete" if not verifier.errors else "failed",
        "artifacts": str(artifacts),
        "probe_file": str(probe_path.resolve()),
        "split": args.split,
        "probe_schema": payload.get("schema"),
        "pool_count": len(pools),
        "episode_count": len(pools_by_episode),
        "pools_verified_from_score_files": len(checked_pools),
        "checkpoint_count_indexed": len(score_index),
        "record_count": len(records),
        "expected_record_count": expected_record_count,
        "group_count": len(groups),
        "group_family_counts": {
            "encoder": sum(family == "encoder" for family in family_by_id.values()),
            "predictor": sum(family == "predictor" for family in family_by_id.values()),
        },
        "site_metric_count": len(recomputed_site_metrics),
        "signals_checked": list(SIGNALS),
        "score_validation": {
            "pairwise_formula": "mean(sign(ref_i-ref_j) != sign(q_i-q_j)) for i<j; exact zero signs retained",
            "score_nmse_formula": "mean((q-ref)^2)/(mean(ref^2)+1e-12)",
            "stable_score_vectors": True,
            "all_rows_finite_checked": True,
            "all_score_files_checked": len(checked_pools) == len(pools),
        },
        "site_aggregation": "mean pools within MPC point, mean visited points within episode, then equal-weight mean across episodes",
        "allocation_check": allocation_check,
        "errors": verifier.errors,
        "error_count": len(verifier.errors),
    }
    output = args.output.resolve()
    output_path = output if output.suffix.lower() == ".json" else output / "compactverification.json"
    _write_json(output_path, report)
    print(json.dumps({"status": report["status"], "split": args.split, "pool_count": len(pools), "record_count": len(records), "error_count": len(verifier.errors), "output": str(output_path)}, indent=2), flush=True)
    if verifier.errors:
        raise RuntimeError(f"probe verification found {len(verifier.errors)} error(s); see {output_path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, required=True, help="probe job directory containing probes/probes.pkl")
    parser.add_argument("--split", choices=("cal", "dev"), required=True, help="probe split")
    parser.add_argument("--output", type=Path, required=True, help="output JSON path or directory for compactverification.json")
    return parser.parse_args()


def main() -> None:
    _verify(_parse_args())


if __name__ == "__main__":
    main()
