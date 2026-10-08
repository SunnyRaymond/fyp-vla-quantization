#!/usr/bin/env python3
"""Build the frozen 1,400-variant LIBERO-Plus cohort from literal source maps."""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import re
import socket
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SEED = 20261005
PER_CELL = 50
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = {
    "Objects Layout": "objects_layout",
    "Camera Viewpoints": "camera_viewpoints",
    "Robot Initial States": "robot_initial_states",
    "Language Instructions": "language_instructions",
    "Light Conditions": "light_conditions",
    "Background Textures": "background_textures",
    "Sensor Noise": "sensor_noise",
}
DIMENSIONS = tuple(CATEGORIES.values())
LAYOUT_SUFFIX = re.compile(r"(?:moved_)?level[1-9][0-9]*_sample[0-9]+", re.IGNORECASE)
SUBTYPE_PREFIX = re.compile(r"(add|table|tb)(?:_|$)", re.IGNORECASE)


def load_literal_task_map(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value = None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == "libero_task_map" for target in targets):
            value = ast.literal_eval(node.value)
            break
    if not isinstance(value, dict):
        raise RuntimeError(f"Could not find a literal libero_task_map dictionary in {path}")
    return value


def is_task_entry(value) -> bool:
    return isinstance(value, str) or isinstance(value, dict)


def order_zero_entries(suite_map, suite: str, path: Path):
    if isinstance(suite_map, dict):
        if 0 in suite_map:
            entries = suite_map[0]
        elif "0" in suite_map:
            entries = suite_map["0"]
        else:
            raise RuntimeError(f"No task_order_index=0 in {path} for {suite}; keys={list(suite_map)[:8]}")
    elif isinstance(suite_map, (list, tuple)):
        if suite_map and all(is_task_entry(item) for item in suite_map):
            entries = suite_map
        elif suite_map and isinstance(suite_map[0], (list, tuple)):
            entries = suite_map[0]
        else:
            raise RuntimeError(f"Unsupported task_order_index map shape in {path} for {suite}")
    else:
        raise RuntimeError(f"Unsupported task map value in {path} for {suite}: {type(suite_map).__name__}")
    if not isinstance(entries, (list, tuple)) or not entries or not all(is_task_entry(item) for item in entries):
        raise RuntimeError(f"Invalid order-zero task list in {path} for {suite}")
    return entries


def task_name_from_entry(entry, suite: str, path: Path) -> str:
    if isinstance(entry, str):
        value = entry
    else:
        value = None
        for key in ("task_name", "name", "bddl_file_name", "bddl_file", "task_file", "file"):
            candidate = entry.get(key)
            if isinstance(candidate, str):
                value = candidate
                break
        if value is None:
            raise RuntimeError(f"Cannot extract task name from {path} entry in {suite}: {list(entry)[:8]}")
    name = PurePosixPath(value.replace("\\", "/")).name
    if name.lower().endswith(".bddl"):
        name = name[:-5]
    if not name:
        raise RuntimeError(f"Empty task name from {path} entry in {suite}")
    return name


def task_names_for_suite(task_map: dict, suite: str, path: Path) -> list[str]:
    if suite not in task_map:
        raise RuntimeError(f"Task map {path} has no suite {suite}")
    names = [task_name_from_entry(item, suite, path) for item in order_zero_entries(task_map[suite], suite, path)]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate order-zero task names in {path} for {suite}")
    return names


def canonical_root(name: str, canonical_tasks: list[str], suite: str) -> str:
    matches = [task for task in canonical_tasks if name == task or name.startswith(task + "_")]
    if not matches:
        raise ValueError(f"No canonical base task prefix: suite={suite} name={name!r}")
    return max(matches, key=len)


def subtype(name: str, category: str, root: str) -> str:
    suffix = name[len(root):].lstrip("_").lower()
    if category == "Objects Layout":
        if LAYOUT_SUFFIX.fullmatch(suffix):
            return "layout"
        match = SUBTYPE_PREFIX.match(suffix)
        if match:
            return match.group(1).lower()
        raise ValueError(f"Unrecognized Objects Layout subtype: name={name!r} root={root!r} suffix={suffix!r}")
    if category == "Background Textures":
        match = SUBTYPE_PREFIX.match(suffix)
        if match:
            return match.group(1).lower()
        raise ValueError(f"Unrecognized Background Textures subtype: name={name!r} root={root!r} suffix={suffix!r}")
    if category == "Camera Viewpoints":
        return "view"
    if category == "Robot Initial States":
        return "initstate"
    if category == "Language Instructions":
        return "language"
    if category == "Light Conditions":
        return "light"
    if category == "Sensor Noise":
        return "noise"
    raise ValueError(f"Unknown perturbation category: {category!r} for name={name!r} root={root!r} suffix={suffix!r}")


def official_difficulty(item: dict, suite: str, name: str):
    if "difficulty_level" not in item:
        raise RuntimeError(f"Official classification entry lacks difficulty_level: suite={suite} name={name!r}")
    value = item["difficulty_level"]
    return None if value is None else int(value)


def balanced_sample(rows: list[dict], suite: str, dimension: str) -> list[dict]:
    rng = random.Random(SEED + SUITES.index(suite) * 10_000 + DIMENSIONS.index(dimension) * 100)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    priority = {row["variant_id"]: i for i, row in enumerate(shuffled)}
    selected: list[dict] = []
    by_task = collections.Counter()
    by_subtype = collections.Counter()
    by_pair = collections.Counter()
    by_difficulty = collections.Counter()
    remaining = list(shuffled)
    while len(selected) < PER_CELL:
        row = min(remaining, key=lambda item: (
            by_task[item["original_task"]],
            by_subtype[item["subtype"]],
            by_pair[(item["original_task"], item["subtype"])],
            by_difficulty[item["difficulty"]],
            priority[item["variant_id"]],
        ))
        remaining.remove(row)
        selected.append(row)
        by_task[row["original_task"]] += 1
        by_subtype[row["subtype"]] += 1
        by_pair[(row["original_task"], row["subtype"])] += 1
        by_difficulty[row["difficulty"]] += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-status", type=Path, required=True)
    args = parser.parse_args()

    import os

    if not os.environ.get("PBS_JOBID") or not os.environ.get("PBS_NODEFILE"):
        raise RuntimeError("Manifest generation requires an approved PBS allocation")
    node = socket.gethostname().split(".")[0]
    nodes = {line.split(".")[0] for line in Path(os.environ["PBS_NODEFILE"]).read_text().split()}
    if "login" in node or node not in nodes:
        raise RuntimeError("Manifest generation requires the allocated compute host")

    source_package = args.repo / "libero" / "libero"
    classification_path = source_package / "benchmark" / "task_classification.json"
    plus_map_path = source_package / "benchmark" / "libero_suite_task_map.py"
    base_map_path = args.base_repo / "libero" / "benchmark" / "libero_suite_task_map.py"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    plus_map = load_literal_task_map(plus_map_path)
    base_map = load_literal_task_map(base_map_path)
    env_status = json.loads(args.env_status.read_text(encoding="utf-8"))

    official_categories = sorted({
        str(item.get("category"))
        for suite in SUITES
        for item in classification.get(suite, [])
        if isinstance(item, dict)
    })
    unknown_categories = sorted(set(official_categories) - set(CATEGORIES))
    print(f"OFFICIAL_PERTURBATION_CATEGORIES {official_categories}", flush=True)
    if unknown_categories:
        raise RuntimeError(f"Unmapped official perturbation categories: {unknown_categories}")

    populations: dict[str, dict[str, int]] = {}
    eligible_populations: dict[str, dict[str, int]] = {}
    skipped_original_counts: dict[str, dict[str, int]] = {}
    missing_difficulty_counts: dict[str, dict[str, int]] = {}
    missing_difficulty_examples: dict[str, list[dict]] = {}
    suite_task_counts: dict[str, int] = {}
    base_task_counts: dict[str, int] = {}
    canonical_exact_presence: dict[str, dict] = {}
    canonical_variant_prefix_counts: dict[str, dict[str, int]] = {}
    cells: dict[tuple[str, str], list[dict]] = {}
    controls: list[dict] = []
    unmapped_subtypes: dict[tuple[str, str], dict] = {}
    for suite in SUITES:
        rows = classification.get(suite)
        if not isinstance(rows, list):
            raise RuntimeError(f"Official Plus classification is missing suite {suite}")
        task_names = task_names_for_suite(plus_map, suite, plus_map_path)
        canonical_tasks = task_names_for_suite(base_map, suite, base_map_path)
        if len(canonical_tasks) != 10:
            raise RuntimeError(f"Expected ten baseline canonical tasks in {suite}; got {len(canonical_tasks)}")
        plus_task_ids = {name: index for index, name in enumerate(task_names)}
        exact_presence = {name: name in plus_task_ids for name in canonical_tasks}
        prefix_counts = {
            name: sum(candidate.startswith(name + "_") for candidate in task_names)
            for name in canonical_tasks
        }
        missing_prefixes = [name for name, count in prefix_counts.items() if count == 0]
        if missing_prefixes:
            raise RuntimeError(f"BASE canonical task prefixes are absent from Plus order-zero map in {suite}: {missing_prefixes[:5]}")
        canonical_exact_presence[suite] = {"count": sum(exact_presence.values()), "by_task": exact_presence}
        canonical_variant_prefix_counts[suite] = prefix_counts
        suite_task_counts[suite] = len(task_names)
        base_task_counts[suite] = len(canonical_tasks)
        population = {dimension: 0 for dimension in DIMENSIONS}
        eligible_population = {dimension: 0 for dimension in DIMENSIONS}
        skipped_original = {dimension: 0 for dimension in DIMENSIONS}
        missing_difficulty = {dimension: 0 for dimension in DIMENSIONS}
        missing_examples: list[dict] = []
        per_dimension: dict[str, list[dict]] = {dimension: [] for dimension in DIMENSIONS}
        unmatched: list[str] = []
        matched_roots = set()
        for item in rows:
            name = item.get("name")
            category = item.get("category")
            if category not in CATEGORIES:
                raise RuntimeError(f"Unmapped official perturbation categories: {official_categories}")
            if name not in plus_task_ids:
                unmatched.append(str(name))
                continue
            root = canonical_root(name, canonical_tasks, suite)
            matched_roots.add(root)
            dimension = CATEGORIES[category]
            population[dimension] += 1
            difficulty = official_difficulty(item, suite, name)
            if difficulty is None:
                missing_difficulty[dimension] += 1
                if len(missing_examples) < 3:
                    missing_examples.append({"dimension": dimension, "name": name, "classification_id": item.get("id")})
            if category == "Objects Layout" and name == root:
                skipped_original[dimension] += 1
                controls.append({
                    "suite": suite,
                    "task_id": plus_task_ids[name],
                    "task_name": name,
                    "dimension": dimension,
                    "subtype": "original",
                    "original_task": root,
                    "variant_id": f"{suite}:{plus_task_ids[name]:04d}",
                    "classification_id": int(item["id"]),
                    "difficulty": difficulty,
                    "state_id": 0,
                })
                continue
            try:
                row_subtype = subtype(name, category, root)
            except ValueError:
                suffix = name[len(root):].lstrip("_").lower()
                pattern = re.sub(r"\d+", "#", suffix)
                key = (category, pattern)
                entry = unmapped_subtypes.setdefault(key, {"count": 0, "examples": []})
                entry["count"] += 1
                if len(entry["examples"]) < 2:
                    entry["examples"].append({"suite": suite, "name": name, "root": root, "suffix": suffix})
                continue
            row = {
                "suite": suite,
                "task_id": plus_task_ids[name],
                "task_name": name,
                "dimension": dimension,
                "subtype": row_subtype,
                "original_task": root,
                "difficulty": difficulty,
                "variant_id": f"{suite}:{plus_task_ids[name]:04d}",
                "classification_id": int(item["id"]),
                "classification_category": category,
                "state_id": 0,
            }
            eligible_population[dimension] += 1
            per_dimension[dimension].append(row)
        if unmatched:
            raise RuntimeError(f"Classification names missing from Plus order-zero suite map for {suite}: {len(unmatched)}; examples={unmatched[:3]}")
        if len(matched_roots) != 10:
            raise RuntimeError(f"Expected ten BASE canonical roots represented in {suite}; parsed {len(matched_roots)}")
        populations[suite] = population
        eligible_populations[suite] = eligible_population
        skipped_original_counts[suite] = skipped_original
        missing_difficulty_counts[suite] = missing_difficulty
        missing_difficulty_examples[suite] = missing_examples
        for dimension, candidates in per_dimension.items():
            cells[(suite, dimension)] = candidates
        print(
            f"SUITE_MAP suite={suite} plus_task_count={len(task_names)} canonical_base_count={len(canonical_tasks)} "
            f"canonical_exact_count={sum(exact_presence.values())} classification_count={len(rows)} "
            f"full_population={population} eligible_population={eligible_population} "
            f"skipped_original={skipped_original} missing_difficulty={missing_difficulty} "
            f"missing_difficulty_examples={missing_examples}",
            flush=True,
        )

    if unmapped_subtypes:
        report = [
            {"category": category, "suffix_pattern": pattern, **details}
            for (category, pattern), details in sorted(unmapped_subtypes.items())
        ]
        print("UNMAPPED_SUBTYPE_SUMMARY " + json.dumps(report, separators=(",", ":"), ensure_ascii=False), flush=True)
        raise RuntimeError(f"Found {len(report)} unmapped category/suffix patterns; refusing to sample or write the manifest")

    insufficient = [
        (suite, dimension, len(candidates))
        for (suite, dimension), candidates in cells.items()
        if len(candidates) < PER_CELL
    ]
    if insufficient:
        raise RuntimeError(f"Insufficient eligible perturbed rows (suite, dimension, count): {insufficient}")

    variants = []
    for suite in SUITES:
        for dimension in DIMENSIONS:
            selected = balanced_sample(cells[(suite, dimension)], suite, dimension)
            for rank, row in enumerate(selected):
                row = dict(row)
                row["selection_rank"] = rank
                variants.append(row)
            print(f"FROZEN_CELL suite={suite} dimension={dimension} n={len(selected)}", flush=True)

    if len(variants) != 1400 or len({row["variant_id"] for row in variants}) != 1400:
        raise RuntimeError(f"Expected 1,400 unique variants; got {len(variants)}")
    count = collections.Counter((row["suite"], row["dimension"]) for row in variants)
    if len(count) != 28 or set(count.values()) != {PER_CELL}:
        raise RuntimeError(f"Frozen cell counts invalid: {count}")

    document = {
        "schema_version": 1,
        "protocol": "fastwam-optional-idm-plus-pilot-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "variants_per_suite_dimension": PER_CELL,
        "state_id": 0,
        "trial_count_per_variant": 1,
        "suite_task_counts": suite_task_counts,
        "base_task_counts": base_task_counts,
        "canonical_exact_presence": canonical_exact_presence,
        "canonical_variant_prefix_counts": canonical_variant_prefix_counts,
        "full_population_counts": populations,
        "eligible_population_counts": eligible_populations,
        "skipped_original_counts": skipped_original_counts,
        "missing_difficulty_counts": missing_difficulty_counts,
        "missing_difficulty_examples": missing_difficulty_examples,
        "canonical_control_tasks": controls,
        "selection_counts": {suite: {dimension: count[(suite, dimension)] for dimension in DIMENSIONS} for suite in SUITES},
        "selection_balance": "greedy round-robin by canonical original_task, then perturbation subtype, task-subtype pair, difficulty; fixed seeded tie order",
        "source": {
            "repository": "https://github.com/sylvestf/LIBERO-plus",
            "commit": env_status["source_commit"],
            "classification_path": str(classification_path),
            "plus_suite_map_path": str(plus_map_path),
            "base_suite_map_path": str(base_map_path),
            "task_order_index": 0,
            "task_id_definition": "zero-based list position in the literal official Plus libero_task_map[suite]; canonical task names were checked against the BASE suite map and matched to Plus task prefixes",
            "classification_metadata": "task_classification.json joined by exact task name to the official order-zero Plus suite map",
        },
        "arms": ["bf16", "w4a8", "w4a4", "w4a4kv4"],
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(document, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    balance = {}
    for suite in SUITES:
        balance[suite] = {}
        for dimension in DIMENSIONS:
            cell = [row for row in variants if row["suite"] == suite and row["dimension"] == dimension]
            balance[suite][dimension] = {
                "original_task": dict(sorted(collections.Counter(row["original_task"] for row in cell).items())),
                "subtype": dict(sorted(collections.Counter(row["subtype"] for row in cell).items())),
                "difficulty": dict(sorted(collections.Counter(json.dumps(row["difficulty"]) for row in cell).items())),
            }
    summary = {
        "schema_version": 1,
        "seed": SEED,
        "variants_per_suite_dimension": PER_CELL,
        "variant_count": len(variants),
        "source_commit": env_status["source_commit"],
        "asset_commit": env_status["assets"]["commit"],
        "suite_task_counts": suite_task_counts,
        "base_task_counts": base_task_counts,
        "canonical_exact_presence": {suite: value["count"] for suite, value in canonical_exact_presence.items()},
        "full_population_counts": populations,
        "eligible_population_counts": eligible_populations,
        "skipped_original_counts": skipped_original_counts,
        "missing_difficulty_counts": missing_difficulty_counts,
        "missing_difficulty_examples": missing_difficulty_examples,
        "selection_counts": document["selection_counts"],
        "selection_balance_counts": balance,
        "canonical_control_count": len(controls),
        "manifest_bytes": args.output.stat().st_size,
    }
    summary_path = args.output.with_name("manifest_summary.json")
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_tmp.replace(summary_path)
    print(f"MANIFEST_SUMMARY path={summary_path} bytes={summary_path.stat().st_size}", flush=True)
    print(f"MANIFEST_COMPLETE path={args.output} bytes={args.output.stat().st_size} variants={len(variants)} unique=1400 seed={SEED}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
