"""Analyze calibration ranking stability against held-out development probes.

This is a CPU-only diagnostic.  It does not create an allocation artifact:
development rankings are reported only as top-five diagnostics, while the
calibration leave-one-episode-out rankings are used to describe stability of
the already specified W4/W8 site-selection rule.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


PROBE_SCHEMA = "rankcal-wall-calibration-probes-v1"
ANALYSIS_SCHEMA = "rankcal-wall-stability-analysis-v1"
EXPECTED_EPISODES = 8
BITS = (4, 8)
W8_QUOTA = {"encoder": 3, "predictor": 2}
EXPECTED_FAMILY_COUNTS = {"encoder": 12, "predictor": 6}
SIGNALS = ("rank_disagreement", "local_block_output_nmse", "planner_score_nmse")
SPLIT_ALIASES = {"cal": "cal", "calibration": "cal", "dev": "dev", "development": "dev"}


def _load(path: Path) -> Any:
    path = path.resolve()
    if path.is_dir():
        candidates = (path / "probes.pkl", path / "calibration_probes.pkl")
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        return pickle.load(stream)


def _expected_episode_ids(split: str) -> List[str]:
    prefix = "calibration" if split == "cal" else "development"
    return [f"{prefix}:{index:03d}" for index in range(EXPECTED_EPISODES)]


def _groups(probe: Mapping[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    raw = probe.get("groups")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("probe groups must be a sequence")
    group_ids: List[str] = []
    families: Dict[str, str] = {}
    for group in raw:
        if not isinstance(group, Mapping):
            raise ValueError("probe group is not a mapping")
        group_id = str(group.get("group_id"))
        family = str(group.get("family"))
        if group_id in families:
            raise ValueError(f"duplicate probe group {group_id}")
        if family not in EXPECTED_FAMILY_COUNTS:
            raise ValueError(f"unexpected probe group family {family!r}")
        group_ids.append(group_id)
        families[group_id] = family
    if len(group_ids) != sum(EXPECTED_FAMILY_COUNTS.values()):
        raise ValueError(f"expected 18 groups, got {len(group_ids)}")
    counts = {family: sum(value == family for value in families.values()) for family in EXPECTED_FAMILY_COUNTS}
    if counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"probe group counts are {counts}")
    return sorted(group_ids), families


def _pools(probe: Mapping[str, Any], split: str) -> Tuple[List[str], Dict[str, str]]:
    raw = probe.get("pools")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("probe pool manifest must be a sequence")
    episode_ids = _expected_episode_ids(split)
    pool_ids: List[str] = []
    pool_episode: Dict[str, str] = {}
    by_episode: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for pool in raw:
        if not isinstance(pool, Mapping):
            raise ValueError("probe pool is not a mapping")
        pool_id = str(pool.get("pool_id"))
        episode_id = str(pool.get("episode_id"))
        if pool_id in pool_episode:
            raise ValueError(f"duplicate probe pool {pool_id}")
        if episode_id not in episode_ids:
            raise ValueError(f"unexpected probe episode {episode_id}")
        pool_ids.append(pool_id)
        pool_episode[pool_id] = episode_id
        by_episode[episode_id].append(pool)
    if set(by_episode) != set(episode_ids):
        raise ValueError("probe pool manifest does not contain exactly 8 expected episodes")
    for episode_id, episode_pools in by_episode.items():
        if len(episode_pools) not in (2, 4):
            raise ValueError(f"episode {episode_id} has {len(episode_pools)} pools; expected 2 or 4")
        points = {int(pool.get("mpc_point", -1)) for pool in episode_pools}
        if points not in ({0}, {0, 1}):
            raise ValueError(f"episode {episode_id} has invalid MPC points {sorted(points)}")
        for point in points:
            iterations = sorted(
                int(pool.get("cem_iteration", -1))
                for pool in episode_pools
                if int(pool.get("mpc_point", -1)) == point
            )
            if iterations != [1, 5]:
                raise ValueError(f"episode {episode_id} point {point} must contain CEM iterations 1 and 5")
    return pool_ids, pool_episode


def _validate_probe(probe: Mapping[str, Any], split: str) -> Tuple[List[str], Dict[str, str], List[str], Dict[str, str]]:
    if probe.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"expected {PROBE_SCHEMA}, got {probe.get('schema')!r}")
    if str(probe.get("split")) != split:
        raise ValueError(f"probe split {probe.get('split')!r} does not match {split!r}")
    group_ids, families = _groups(probe)
    pool_ids, pool_episode = _pools(probe, split)
    expected_keys = {(pool_id, group_id, bits) for pool_id in pool_ids for group_id in group_ids for bits in BITS}
    records = probe.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("probe records must be a sequence")
    if len(records) != len(expected_keys):
        raise ValueError(f"expected {len(expected_keys)} records, got {len(records)}")
    observed_keys = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("probe record is not a mapping")
        pool_id = str(row.get("pool_id"))
        group_id = str(row.get("group_id"))
        bits = int(row.get("bits", -1))
        key = (pool_id, group_id, bits)
        if key not in expected_keys or key in observed_keys:
            raise ValueError(f"invalid or duplicate probe record key {key}")
        if str(row.get("episode_id")) != pool_episode[pool_id]:
            raise ValueError(f"probe record episode mismatch for {key}")
        if str(row.get("family")) != families[group_id] or not bool(row.get("finite")):
            raise ValueError(f"probe record family/finite mismatch for {key}")
        for signal in SIGNALS:
            try:
                value = float(row[signal])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed {signal} for {key}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite {signal} for {key}")
        observed_keys.add(key)
    if observed_keys != expected_keys:
        raise ValueError("probe records do not cover every pool x group x bit combination")

    site_metrics = probe.get("site_metrics")
    expected_metric_keys = {(group_id, bits) for group_id in group_ids for bits in BITS}
    if not isinstance(site_metrics, Sequence) or isinstance(site_metrics, (str, bytes)):
        raise ValueError("probe site_metrics must be a sequence")
    metric_keys = set()
    for row in site_metrics:
        if not isinstance(row, Mapping):
            raise ValueError("probe site metric is not a mapping")
        key = (str(row.get("group_id")), int(row.get("bits", -1)))
        if key not in expected_metric_keys or key in metric_keys:
            raise ValueError(f"invalid or duplicate site metric key {key}")
        if "family" in row and str(row.get("family")) != families[key[0]]:
            raise ValueError(f"site metric family mismatch for {key}")
        for signal in SIGNALS:
            if not math.isfinite(float(row[signal])):
                raise ValueError(f"non-finite site metric {signal} for {key}")
        metric_keys.add(key)
    if metric_keys != expected_metric_keys:
        raise ValueError("site_metrics do not cover every group x bit combination")
    return group_ids, families, pool_ids, pool_episode


def _episode_signal_table(
    probe: Mapping[str, Any],
    group_ids: Sequence[str],
    pool_episode: Mapping[str, str],
) -> Dict[Tuple[str, int, str], Dict[str, float]]:
    values: MutableMapping[Tuple[str, int, str], MutableMapping[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in probe["records"]:
        key = (str(row["group_id"]), int(row["bits"]), pool_episode[str(row["pool_id"])])
        for signal in SIGNALS:
            values[key][signal].append(float(row[signal]))
    episodes = sorted({key[2] for key in values})
    result: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for group_id in group_ids:
        for bits in BITS:
            for episode_id in episodes:
                key = (group_id, bits, episode_id)
                if key not in values:
                    raise ValueError(f"missing episode metric for {key}")
                result[key] = {
                    signal: float(np.mean(values[key][signal]))
                    for signal in SIGNALS
                }
    return result


def _aggregate_episode_values(
    episode_values: Mapping[Tuple[str, int, str], Mapping[str, float]],
    group_ids: Sequence[str],
    episode_ids: Sequence[str],
) -> Dict[Tuple[str, int], Dict[str, float]]:
    aggregate: Dict[Tuple[str, int], Dict[str, float]] = {}
    for group_id in group_ids:
        for bits in BITS:
            aggregate[(group_id, bits)] = {
                signal: float(np.mean([
                    episode_values[(group_id, bits, episode_id)][signal]
                    for episode_id in episode_ids
                ]))
                for signal in SIGNALS
            }
    return aggregate


def _site_metrics(probe: Mapping[str, Any]) -> Dict[Tuple[str, int], Dict[str, float]]:
    return {
        (str(row["group_id"]), int(row["bits"])): {
            signal: float(row[signal])
            for signal in SIGNALS
        }
        for row in probe["site_metrics"]
    }


def _assert_metrics_match(
    recorded: Mapping[Tuple[str, int], Mapping[str, float]],
    recomputed: Mapping[Tuple[str, int], Mapping[str, float]],
) -> None:
    for key in recomputed:
        for signal in SIGNALS:
            if not np.isclose(recorded[key][signal], recomputed[key][signal], rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"site_metrics mismatch for {key}/{signal}: "
                    f"recorded={recorded[key][signal]} recomputed={recomputed[key][signal]}"
                )


def _select_sites(
    aggregate: Mapping[Tuple[str, int], Mapping[str, float]],
    families: Mapping[str, str],
    signal: str,
) -> List[str]:
    ranking: List[Tuple[str, float, str]] = []
    for group_id, family in families.items():
        benefit = float(aggregate[(group_id, 4)][signal] - aggregate[(group_id, 8)][signal])
        if not math.isfinite(benefit):
            raise ValueError(f"non-finite W4-W8 benefit for {group_id}/{signal}")
        ranking.append((family, -benefit, group_id))
    selected: List[str] = []
    for family, quota in W8_QUOTA.items():
        family_rank = sorted(
            (item for item in ranking if item[0] == family),
            key=lambda item: (item[1], item[2]),
        )
        selected.extend(item[2] for item in family_rank[:quota])
    return sorted(selected)


def _set_overlap(left: Sequence[str], right: Sequence[str]) -> Dict[str, Any]:
    left_set, right_set = set(left), set(right)
    intersection = left_set & right_set
    union = left_set | right_set
    return {
        "overlap_count": len(intersection),
        "jaccard": float(len(intersection) / len(union)) if union else 1.0,
    }


def _analyze(cal_path: Path, dev_path: Path) -> Dict[str, Any]:
    cal = _load(cal_path)
    dev = _load(dev_path)
    if not isinstance(cal, Mapping) or not isinstance(dev, Mapping):
        raise TypeError("cal and dev probe artifacts must be mappings")
    cal_group_ids, cal_families, _, cal_pool_episode = _validate_probe(cal, "cal")
    dev_group_ids, dev_families, _, dev_pool_episode = _validate_probe(dev, "dev")
    if cal_group_ids != dev_group_ids or cal_families != dev_families:
        raise ValueError("cal and dev group manifests differ")

    cal_episode_ids = _expected_episode_ids("cal")
    dev_episode_ids = _expected_episode_ids("dev")
    cal_episode_values = _episode_signal_table(cal, cal_group_ids, cal_pool_episode)
    dev_episode_values = _episode_signal_table(dev, dev_group_ids, dev_pool_episode)
    cal_aggregate = _aggregate_episode_values(cal_episode_values, cal_group_ids, cal_episode_ids)
    dev_aggregate = _aggregate_episode_values(dev_episode_values, dev_group_ids, dev_episode_ids)
    _assert_metrics_match(_site_metrics(cal), cal_aggregate)
    _assert_metrics_match(_site_metrics(dev), dev_aggregate)

    full_selected = {
        signal: _select_sites(cal_aggregate, cal_families, signal)
        for signal in SIGNALS
    }
    dev_diagnostic = {}
    for signal in SIGNALS:
        dev_selected = _select_sites(dev_aggregate, dev_families, signal)
        dev_diagnostic[signal] = {
            "cal_selected_sites": full_selected[signal],
            "dev_diagnostic_top5_sites": dev_selected,
            **_set_overlap(full_selected[signal], dev_selected),
        }

    loo: Dict[str, Any] = {}
    for signal in SIGNALS:
        frequency = {group_id: 0 for group_id in cal_group_ids}
        runs = []
        exact_match_count = 0
        for omitted_episode in cal_episode_ids:
            retained = [episode_id for episode_id in cal_episode_ids if episode_id != omitted_episode]
            loo_aggregate = _aggregate_episode_values(cal_episode_values, cal_group_ids, retained)
            selected = _select_sites(loo_aggregate, cal_families, signal)
            if selected == full_selected[signal]:
                exact_match_count += 1
            for group_id in selected:
                frequency[group_id] += 1
            runs.append({
                "omitted_episode": omitted_episode,
                "selected_sites": selected,
                **_set_overlap(full_selected[signal], selected),
            })
        loo[signal] = {
            "full_selected_sites": full_selected[signal],
            "exact_match_count": exact_match_count,
            "run_count": len(cal_episode_ids),
            "site_selection_frequency": {
                group_id: {
                    "count": frequency[group_id],
                    "frequency_over_8": float(frequency[group_id] / len(cal_episode_ids)),
                }
                for group_id in cal_group_ids
            },
            "runs": runs,
        }

    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete",
        "calibration_probe": str(cal_path.resolve()),
        "development_probe": str(dev_path.resolve()),
        "selection_rule": {
            "benefit": "site signal W4 minus W8",
            "family_w8_quota": dict(W8_QUOTA),
            "tie_rule": "descending benefit; exact ties use lexicographic site ID within family",
            "aggregation": "pool mean within episode, then equal-weight episode mean",
        },
        "calibration": {
            "episode_count": len(cal_episode_ids),
            "episode_ids": cal_episode_ids,
            "full_selected_sites": full_selected,
            "leave_one_episode_out": loo,
        },
        "development_diagnostic": {
            "episode_count": len(dev_episode_ids),
            "episode_ids": dev_episode_ids,
            "allocation_generated": False,
            "purpose": "diagnostic top-five comparison only; never an input allocation for test",
            "signals": dev_diagnostic,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", type=Path, required=True, help="calibration probes.pkl")
    parser.add_argument("--dev", type=Path, required=True, help="development probes.pkl")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = _analyze(args.cal, args.dev)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
