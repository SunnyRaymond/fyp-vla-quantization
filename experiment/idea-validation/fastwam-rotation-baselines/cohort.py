"""Freeze a balanced 280-variant cohort and four disjoint worker shards."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random

PROTOCOL = "fastwam-rotation-baselines-v1"
SOURCE_PROTOCOL = "fastwam-optional-idm-plus-pilot-v1"
SEED = 20261008
PER_CELL_SOURCE = 50
PER_CELL = 10
EXPECTED_ROWS = 280
EXPECTED_SHARD_SIZE = 70
ARMS = ("bf16", "quarot_adapted_w4a4", "spinquant_adapted_w4a4")
OFFICIAL_REPOSITORY = "https://github.com/sylvestf/LIBERO-plus"


def _json_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _balanced_sample(rows: list[dict], seed: int) -> list[dict]:
    """Greedily balance task, subtype, task/subtype, and difficulty."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    priority = {row["variant_id"]: i for i, row in enumerate(shuffled)}
    selected = []
    remaining = list(shuffled)
    task_counts, subtype_counts = Counter(), Counter()
    pair_counts, difficulty_counts = Counter(), Counter()
    while len(selected) < PER_CELL:
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


def _worker_quotas(cell_index: int) -> list[int]:
    # Each worker gets 2 or 3 rows in a cell; every dimension totals 10 per worker.
    extra_pairs = ((0, 1), (2, 3), (0, 2), (1, 3))
    extras = extra_pairs[cell_index % 4]
    return [2 + int(worker in extras) for worker in range(4)]


def _assign_workers(rows: list[dict], quotas: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    tie_order = list(range(4))
    rng.shuffle(tie_order)
    tie_rank = {worker: i for i, worker in enumerate(tie_order)}
    task_counts = [Counter() for _ in range(4)]
    subtype_counts = [Counter() for _ in range(4)]
    pair_counts = [Counter() for _ in range(4)]
    difficulty_counts = [Counter() for _ in range(4)]
    assigned = []
    worker_sizes = [0] * 4
    for row in rows:
        task = row["original_task"]
        subtype = row["subtype"]
        pair = (task, subtype)
        difficulty = _json_key(row.get("difficulty"))
        eligible = [worker for worker in range(4) if worker_sizes[worker] < quotas[worker]]
        worker = min(eligible, key=lambda w: (
            task_counts[w][task], subtype_counts[w][subtype],
            pair_counts[w][pair], difficulty_counts[w][difficulty],
            sum(value == w for value in assigned), tie_rank[w],
        ))
        assigned.append(worker)
        worker_sizes[worker] += 1
        task_counts[worker][task] += 1
        subtype_counts[worker][subtype] += 1
        pair_counts[worker][pair] += 1
        difficulty_counts[worker][difficulty] += 1
    return assigned


def _load_source(source_path: str | Path) -> dict:
    document = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if document.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"Expected source protocol {SOURCE_PROTOCOL!r}")
    provenance = document.get("source")
    if (not isinstance(provenance, dict)
            or provenance.get("repository") != OFFICIAL_REPOSITORY
            or not provenance.get("commit")):
        raise ValueError("Source manifest must retain the official LIBERO-Plus provenance")
    variants = document.get("variants")
    if not isinstance(variants, list) or len(variants) != 1400:
        raise ValueError("Expected the 1,400-row official Plus pilot manifest")
    ids = [row.get("variant_id") for row in variants if isinstance(row, dict)]
    if len(ids) != len(variants) or len(set(ids)) != 1400:
        raise ValueError("Source manifest must contain 1,400 rows with unique variant_id values")
    cells = defaultdict(list)
    for row in variants:
        if not all(key in row for key in ("suite", "dimension", "original_task", "subtype", "difficulty", "state_id")):
            raise ValueError("Source variant is missing a task, state, or stratification field")
        cells[(row["suite"], row["dimension"])].append(row)
    if len(cells) != 28 or any(len(rows) != PER_CELL_SOURCE for rows in cells.values()):
        raise ValueError("Source manifest must have 28 suite/dimension cells with 50 rows each")
    suites = list(dict.fromkeys(row["suite"] for row in variants))
    dimensions = list(dict.fromkeys(row["dimension"] for row in variants))
    if len(suites) != 4 or len(dimensions) != 7 or set(cells) != {
        (suite, dimension) for suite in suites for dimension in dimensions
    }:
        raise ValueError("Source cells must form a 4-suite by 7-dimension grid")
    return document, cells, suites, dimensions


def validate_cohort(document: dict) -> None:
    variants = document.get("variants")
    if document.get("protocol") != PROTOCOL or not isinstance(variants, list) or len(variants) != EXPECTED_ROWS:
        raise ValueError("Cohort must use the frozen protocol and contain 280 variants")
    indices = [row.get("index") for row in variants]
    ids = [row.get("variant_id") for row in variants]
    if indices != list(range(EXPECTED_ROWS)) or len(set(ids)) != EXPECTED_ROWS:
        raise ValueError("Cohort indices and variant IDs must each be unique and contiguous")
    counts = Counter((row["suite"], row["dimension"]) for row in variants)
    if len(counts) != 28 or set(counts.values()) != {PER_CELL}:
        raise ValueError("Cohort must contain exactly 10 variants per suite/dimension cell")
    shards = document.get("shards")
    if not isinstance(shards, list) or len(shards) != 4:
        raise ValueError("Cohort must define four worker shards")
    seen = []
    for worker_id, shard in enumerate(shards):
        if shard.get("worker_id") != worker_id or len(shard.get("indices", [])) != EXPECTED_SHARD_SIZE:
            raise ValueError("Each worker shard must have its matching ID and 70 indices")
        seen.extend(shard["indices"])
    if len(seen) != EXPECTED_ROWS or len(set(seen)) != EXPECTED_ROWS or set(seen) != set(indices):
        raise ValueError("Worker shards must be disjoint and cover all 280 cohort indices")
    by_index = {row["index"]: row for row in variants}
    worker_by_index = {index: shard["worker_id"] for shard in shards for index in shard["indices"]}
    cell_worker_counts = Counter(
        (by_index[index]["suite"], by_index[index]["dimension"], worker_by_index[index])
        for index in seen
    )
    if any(count not in (2, 3) for count in cell_worker_counts.values()) or len(cell_worker_counts) != 112:
        raise ValueError("Each cell must contribute two or three variants to each worker")


def build_cohort(source_path: str | Path, out_path: str | Path) -> dict:
    """Select 10 rows per pilot cell, write the frozen cohort, and return it."""
    if Path(source_path).resolve() == Path(out_path).resolve():
        raise ValueError("Output manifest must not replace the source pilot manifest")
    source, cells, suites, dimensions = _load_source(source_path)
    selected_by_cell = {}
    for cell_index, cell in enumerate(cells):
        selected_by_cell[cell] = _balanced_sample(cells[cell], SEED + cell_index)

    variants, shard_indices = [], [[] for _ in range(4)]
    cell_index = 0
    for suite in suites:
        for dimension in dimensions:
            cell_rows = selected_by_cell[(suite, dimension)]
            quotas = _worker_quotas(cell_index)
            workers = _assign_workers(cell_rows, quotas, SEED + 1000 + cell_index)
            for row, worker_id in zip(cell_rows, workers):
                index = len(variants)
                variants.append({**row, "index": index})
                shard_indices[worker_id].append(index)
            cell_index += 1

    source_ref = {
        "path": str(Path(source_path).resolve()),
        "protocol": source["protocol"],
        "seed": source.get("seed"),
    }
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": SEED,
        "selection": "10 per official pilot cell; seeded greedy balance by original_task, subtype, task/subtype pair, and difficulty",
        "source_manifest": source_ref,
        "source": source["source"],
        "arms": list(ARMS),
        "variants_per_suite_dimension": PER_CELL,
        "variants": variants,
        "shards": [
            {"worker_id": worker_id, "indices": indices}
            for worker_id, indices in enumerate(shard_indices)
        ],
    }
    validate_cohort(document)
    output_path = Path(out_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing frozen cohort: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(output_path)
    return document
