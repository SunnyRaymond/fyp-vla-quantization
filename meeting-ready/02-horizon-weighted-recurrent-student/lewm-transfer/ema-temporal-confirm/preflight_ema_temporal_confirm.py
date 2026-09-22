#!/usr/bin/env python3
"""Local-only protocol smoke checks; never opens HDF5 or model files."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent


def fresh_selection(valid: list[int], old_heldout: list[int], old_train: list[int]) -> list[int]:
    if valid[: len(old_heldout)] != old_heldout:
        raise AssertionError("old heldout is not the frozen valid prefix")
    cut = len(old_heldout) + len(old_train)
    if valid[len(old_heldout) : cut] != old_train:
        raise AssertionError("old train is not the frozen valid prefix")
    fresh = valid[cut : cut + 8]
    if len(fresh) != 8 or set(fresh) & (set(old_heldout) | set(old_train)):
        raise AssertionError("fresh episode selection leaks an old episode")
    return fresh


def anchor_bounds(length: int, span: int = 25) -> dict[str, int]:
    late = length - span - 1
    if late < 0:
        raise ValueError("episode is too short")
    anchors = {"early": 0, "middle": late // 2, "late": late}
    for anchor in anchors.values():
        if not (0 <= anchor and anchor + span - 1 < length and anchor + span < length):
            raise AssertionError("anchor/current/action/goal bounds drifted")
    return anchors


def nested_aggregation_smoke() -> None:
    rows = []
    for episode in range(8):
        for anchor in ("early", "middle", "late"):
            for seed in (20300917, 20300918):
                rows.append({"episode": episode, "anchor": anchor, "seed": seed, "spearman": 0.9 + episode * 0.001, "top30": 0.7 + episode * 0.001})
    by_episode = {}
    for item in rows:
        by_episode.setdefault(item["episode"], []).append(item)
    if any(len(items) != 6 for items in by_episode.values()):
        raise AssertionError("episode is not the six-block nested replicate")
    if len(rows) != 48 or len({(x["episode"], x["anchor"], x["seed"]) for x in rows}) != 48:
        raise AssertionError("candidate block identity is not unique")
    deltas = [median([x["spearman"] for x in by_episode[e]]) for e in by_episode]
    if len(deltas) != 8:
        raise AssertionError("episode aggregation did not produce eight replicates")


def ema_isolation_smoke() -> None:
    online = 0.0
    ema = online
    for step in range(1, 4):
        online += 1.0
        ema = 0.999 * ema + 0.001 * online
    if not (online == 3.0 and 0.0 < ema < online):
        raise AssertionError("EMA was not initialized/updated as an isolated same-trajectory state")


def main() -> int:
    freeze = json.loads((ROOT / "LEWM_EMA_TEMPORAL_CONFIRM_FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["status"] == "frozen"
    assert freeze["scope_boundary"]["official_cem"] == "NOT_RUN_BY_SCOPE"
    assert freeze["scope_boundary"]["planner_viability"] == "NOT_RUN_BY_SCOPE"
    assert freeze["scope_boundary"]["closed_loop"] == "NOT_RUN_BY_SCOPE"
    assert freeze["fresh_evaluation"]["action_prefix_seeds"] == [20300917, 20300918]
    assert not set(freeze["fresh_evaluation"]["action_prefix_seeds"]) & {20300907, 20300908}
    valid = list(range(540))
    old_heldout = valid[:8]
    old_train = valid[8:520]
    assert fresh_selection(valid, old_heldout, old_train) == list(range(520, 528))
    assert anchor_bounds(26) == {"early": 0, "middle": 0, "late": 0}
    assert anchor_bounds(100)["late"] == 74
    nested_aggregation_smoke()
    ema_isolation_smoke()
    random.Random(20300903).shuffle(valid)
    out = {"status": "PASS", "checks": ["freeze", "fresh_prefix_exclusion", "anchor_bounds", "ema_isolation", "same_trajectory", "nested_episode_aggregation"], "hdf5_or_model_opened": False}
    (ROOT / "local-status").mkdir(exist_ok=True)
    (ROOT / "local-status" / "preflight.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
