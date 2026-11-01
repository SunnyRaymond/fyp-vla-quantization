"""Panel B: a second balanced 280-variant LIBERO-Plus cohort, disjoint from panel A, from the same 1,400-row pool.

Stdlib only (runs on the compute node's host python3). The balance rule is panel A's (fastwam-rotation-baselines
cohort.py `_balanced_sample`); only the seed and the candidate rows (the 40 unused variants per cell) differ.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random

SOURCE_PROTOCOL = "fastwam-optional-idm-plus-pilot-v1"
PANEL_A_PROTOCOL = "fastwam-rotation-baselines-v1"
CELLS, SOURCE_PER_CELL = 28, 50


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def balanced_sample(rows, seed, per_cell):
    """Greedily balance task, subtype, task/subtype, and difficulty (panel A's rule, verbatim)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    priority = {row["variant_id"]: i for i, row in enumerate(shuffled)}
    selected, remaining = [], list(shuffled)
    task_counts, subtype_counts, pair_counts, difficulty_counts = Counter(), Counter(), Counter(), Counter()
    while len(selected) < per_cell:
        row = min(remaining, key=lambda item: (
            task_counts[item["original_task"]],
            subtype_counts[item["subtype"]],
            pair_counts[(item["original_task"], item["subtype"])],
            difficulty_counts[_json_key(item.get("difficulty"))],
            priority[item["variant_id"]],
        ))
        remaining.remove(row)
        selected.append(row)
        task_counts[row["original_task"]] += 1
        subtype_counts[row["subtype"]] += 1
        pair_counts[(row["original_task"], row["subtype"])] += 1
        difficulty_counts[_json_key(row.get("difficulty"))] += 1
    return selected


def build(p):
    """Return the panel-B document for protocol p (deterministic)."""
    cfg = p["panel_b"]
    a = read(p["panel_a"]["manifest"])
    if a.get("protocol") != PANEL_A_PROTOCOL or len(a["variants"]) != 280:
        raise ValueError("Panel A must be the frozen 280-variant rotation-baselines cohort")
    source_path = a["source_manifest"]["path"]
    source = read(source_path)
    if source.get("protocol") != SOURCE_PROTOCOL or len(source["variants"]) != CELLS * SOURCE_PER_CELL:
        raise ValueError("Expected the 1,400-row official pilot pool")
    used = {row["variant_id"] for row in a["variants"]}
    cells = defaultdict(list)
    for row in source["variants"]:
        cells[(row["suite"], row["dimension"])].append(row)
    if len(cells) != CELLS or any(len(rows) != SOURCE_PER_CELL for rows in cells.values()):
        raise ValueError("Pool must have 28 suite/dimension cells of 50")
    selected = {}
    for cell_index, cell in enumerate(cells):
        free = [row for row in cells[cell] if row["variant_id"] not in used]
        if len(free) != SOURCE_PER_CELL - 10:
            raise ValueError(f"Cell {cell} should keep 40 variants after removing panel A, has {len(free)}")
        selected[cell] = balanced_sample(free, cfg["seed"] + cell_index, cfg["per_cell"])
    suites = list(dict.fromkeys(row["suite"] for row in source["variants"]))
    dimensions = list(dict.fromkeys(row["dimension"] for row in source["variants"]))
    variants = []
    for suite in suites:
        for dimension in dimensions:
            for row in selected[(suite, dimension)]:
                position = len(variants)
                variants.append({**row, "position": position, "index": cfg["index_offset"] + position})
    doc = dict(schema_version=1, protocol=p["protocol"], panel="B", seed=cfg["seed"],
               selection=cfg["selection"], index_offset=cfg["index_offset"],
               source_manifest=dict(path=source_path, protocol=source["protocol"], seed=source.get("seed")),
               panel_a=dict(path=p["panel_a"]["manifest"], protocol=a["protocol"], seed=a.get("seed")),
               source=source.get("source"), variants_per_suite_dimension=cfg["per_cell"], variants=variants)
    validate(doc, used, cfg)
    return doc


def validate(doc, used, cfg):
    variants = doc["variants"]
    ids = [row["variant_id"] for row in variants]
    if len(variants) != 280 or len(set(ids)) != 280 or used & set(ids):
        raise ValueError("Panel B must hold 280 unique variants disjoint from panel A")
    if [row["index"] for row in variants] != list(range(cfg["index_offset"], cfg["index_offset"] + 280)):
        raise ValueError("Panel B slot indices must be contiguous from the offset")
    counts = Counter((row["suite"], row["dimension"]) for row in variants)
    if len(counts) != CELLS or set(counts.values()) != {cfg["per_cell"]}:
        raise ValueError("Panel B must have 10 variants per suite/dimension cell")


def write(doc, path):
    """Write once; an existing file must be byte-identical to the deterministic rebuild."""
    path = Path(path)
    data = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise RuntimeError(f"Existing panel-B manifest differs from the rebuild: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    doc = build(p)
    out = Path(p["remote_root"]) / "results" / "panel" / "manifest.json"
    created = write(doc, out)
    tasks = Counter(row["original_task"] for row in doc["variants"])
    print(f"PANEL_B {'created' if created else 'verified'} path={out} variants={len(doc['variants'])} "
          f"original_tasks={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
