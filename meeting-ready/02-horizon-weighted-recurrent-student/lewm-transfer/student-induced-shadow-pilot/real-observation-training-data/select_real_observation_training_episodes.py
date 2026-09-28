#!/usr/bin/env python3
"""Select the frozen 80-task real-observation dataset from HDF5 metadata only."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path
from typing import Any


FREEZE_SCHEMA = "lewm-pusht-student-driven-real-observation-training-data-freeze"
SEEDED_SCHEMA = "lewm-pusht-student-induced-shadow-seeded-pilot-freeze"
RANKER_SCHEMA = "lewm-pusht-onpolicy-fullbank-ranker-freeze"
RANKER_MANIFEST_SCHEMA = "lewm-pusht-onpolicy-fullbank-collection-manifest-v1"
SELECTION_SCHEMA = "lewm-pusht-student-induced-shadow-episode-selection"
OUTPUT_SCHEMA = "lewm-pusht-student-driven-real-observation-training-selection-manifest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--seeded-freeze", type=Path, required=True)
    parser.add_argument("--ranker-freeze", type=Path, required=True)
    parser.add_argument("--ranker-manifest", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--baseline-tasks", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_compute_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("selector requires PBS_JOBID and an allocated PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    node_hosts = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in node_hosts or any(any(token in node for token in ("login", "head", "submit")) for node in node_hosts):
        raise RuntimeError("actual hostname and PBS_NODEFILE do not establish a compute allocation")
    return job_id, host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require_same_path(actual: Path, expected: Path, label: str) -> Path:
    actual_resolved = actual.resolve(strict=True)
    expected_resolved = expected.resolve(strict=True)
    if os.path.normcase(str(actual_resolved)) != os.path.normcase(str(expected_resolved)):
        raise RuntimeError(f"{label} path differs from its frozen source")
    return actual_resolved


def unique_ids(rows: list[dict[str, Any]], label: str, count: int) -> set[int]:
    ids = [int(row["episode_idx"]) for row in rows]
    if len(rows) != count or len(set(ids)) != count:
        raise RuntimeError(f"{label} must contain exactly {count} unique episode IDs")
    return set(ids)


def load_exclusions(args: argparse.Namespace, freeze: dict[str, Any]) -> tuple[dict[str, set[int]], dict[str, Any]]:
    base = args.freeze.resolve(strict=True).parent
    project = base.parent
    seeded_path = require_same_path(args.seeded_freeze, project / "SEEDED_PILOT_FREEZE.json", "seeded freeze")
    ranker_freeze_path = require_same_path(args.ranker_freeze, project / "onpolicy-fullbank-ranker" / "FREEZE.json", "ranker freeze")
    ranker_manifest_path = require_same_path(
        args.ranker_manifest,
        project / "onpolicy-fullbank-ranker" / "results" / "25538135.pbs101" / "collection_manifest.json",
        "ranker collection manifest",
    )
    selection_freeze_path = require_same_path(args.selection_freeze, project / "SELECTION_FREEZE.json", "episode selection freeze")
    seeded = read_json(seeded_path)
    ranker = read_json(ranker_freeze_path)
    ranker_manifest = read_json(ranker_manifest_path)
    selection = read_json(selection_freeze_path)
    if seeded.get("schema") != SEEDED_SCHEMA or seeded.get("schema_version") != 1:
        raise RuntimeError("seeded-pilot freeze schema mismatch")
    if ranker.get("schema") != RANKER_SCHEMA or ranker.get("schema_version") != 1:
        raise RuntimeError("ranker freeze schema mismatch")
    if ranker_manifest.get("schema") != RANKER_MANIFEST_SCHEMA:
        raise RuntimeError("ranker collection manifest schema mismatch")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("schema_version") != 1:
        raise RuntimeError("legacy episode-selection freeze schema mismatch")

    seeded_train = seeded.get("collection_tasks", [])
    seeded_reserved = seeded.get("reserved_holdout_tasks", [])
    seeded_collection_ids = unique_ids(seeded_train, "seeded collection", 16)
    seeded_reserved_ids = unique_ids(seeded_reserved, "seeded reserved holdout", 16)
    if seeded_collection_ids & seeded_reserved_ids:
        raise RuntimeError("seeded collection and reserved IDs overlap")

    expected_ranker_validation = ranker.get("task_split", {}).get("validation", {}).get("tasks", [])
    ranker_validation_ids = unique_ids(expected_ranker_validation, "ranker validation split", 8)
    manifest_validation = [row["task"] for row in ranker_manifest.get("episodes", []) if row.get("split") == "validation"]
    manifest_validation_ids = unique_ids(manifest_validation, "ranker manifest validation split", 8)
    if manifest_validation_ids != ranker_validation_ids or not ranker_validation_ids <= seeded_reserved_ids:
        raise RuntimeError("ranker validation IDs must match its freeze and be included in seeded reserved holdout")

    untouched_rows = ranker.get("task_split", {}).get("untouched", {}).get("tasks", [])
    untouched_ids = unique_ids(untouched_rows, "ranker untouched split", 8)
    if untouched_ids != seeded_reserved_ids - ranker_validation_ids:
        raise RuntimeError("ranker untouched IDs must be exactly the remaining seeded reserved tasks")

    legacy = selection.get("legacy_exclusions", {}).get("closed_loop_50", {})
    baseline_ids = [int(value) for value in legacy.get("episode_ids", [])]
    if legacy.get("task_count") != 50 or len(baseline_ids) != 50 or len(set(baseline_ids)) != 50:
        raise RuntimeError("legacy freeze must contain exactly 50 unique closed-loop episode IDs")
    expected_baseline_path = Path(str(legacy.get("source_file", "")))
    baseline_path = require_same_path(args.baseline_tasks, expected_baseline_path, "closed-loop baseline task list")
    baseline_doc = read_json(baseline_path)
    baseline_rows = baseline_doc.get("tasks")
    observed_baseline_ids = unique_ids(baseline_rows if isinstance(baseline_rows, list) else [], "closed-loop baseline task list", 50)
    if observed_baseline_ids != set(baseline_ids):
        raise RuntimeError("baseline task IDs differ from the frozen 50-episode exclusion")

    prior = selection.get("legacy_exclusions", {}).get("prior_valid_prefix", {})
    if (prior.get("episode_length_threshold_inclusive"), prior.get("shuffle_seed"), prior.get("slice_start_inclusive"), prior.get("slice_stop_exclusive")) != (26, 20300903, 0, 600):
        raise RuntimeError("legacy valid-prefix exclusion rule differs from its selection freeze")

    sources = {
        "seeded_collection": seeded_collection_ids,
        "seeded_reserved_holdout": seeded_reserved_ids,
        "ranker_validation": ranker_validation_ids,
        "ranker_untouched": untouched_ids,
        "closed_loop_50": set(baseline_ids),
    }
    source_identity = {
        "seeded_freeze": {"path": str(seeded_path), "schema": seeded["schema"], "collection_count": len(seeded_collection_ids), "reserved_count": len(seeded_reserved_ids)},
        "ranker_freeze": {"path": str(ranker_freeze_path), "schema": ranker["schema"], "validation_count": len(ranker_validation_ids), "untouched_count": len(untouched_ids)},
        "ranker_manifest": {"path": str(ranker_manifest_path), "schema": ranker_manifest["schema"], "validation_count": len(manifest_validation_ids)},
        "selection_freeze": {"path": str(selection_freeze_path), "schema": selection["schema"], "closed_loop_50_count": len(baseline_ids), "closed_loop_source_job_id": legacy["source_job_id"]},
        "baseline_task_list": {"path": str(baseline_path), "task_count": len(observed_baseline_ids)},
    }
    return sources, {"identity": source_identity, "prior_valid_rule": prior}


def main() -> int:
    args = parse_args()
    job_id, host = require_compute_allocation()
    output_path = args.output.resolve()
    if output_path.name != "selection_manifest.json" or output_path.parent.name != job_id or output_path.parent.parent.name != "selection":
        raise RuntimeError("selection output must be selection/<PBS_JOBID>/selection_manifest.json")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite selection manifest: {args.output}")
    for path in (args.freeze, args.seeded_freeze, args.ranker_freeze, args.ranker_manifest, args.selection_freeze, args.baseline_tasks, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)

    freeze = read_json(args.freeze.resolve(strict=True))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("schema_version") != 1:
        raise RuntimeError("real-observation data freeze schema mismatch")
    dataset_binding = freeze.get("provenance", {}).get("dataset_path")
    dataset_path = require_same_path(args.dataset, Path(str(dataset_binding)), "HDF5 dataset")
    selection_spec = freeze.get("episode_selection", {})
    if (selection_spec.get("selected_episode_count"), selection_spec.get("selection_seed"), selection_spec.get("rng")) != (80, 20260925, "random.Random"):
        raise RuntimeError("80-episode selection count/seed/RNG mismatch")
    split_spec = selection_spec.get("split", {})
    if (split_spec.get("collection_train"), split_spec.get("collection_validation")) != (64, 16):
        raise RuntimeError("collection split must remain 64 train plus 16 validation episodes")
    expected_source_paths = [
        "../SEEDED_PILOT_FREEZE.json",
        "../onpolicy-fullbank-ranker/results/25538135.pbs101/collection_manifest.json",
        "../onpolicy-fullbank-ranker/FREEZE.json",
        "../SELECTION_FREEZE.json",
    ]
    frozen_sources = selection_spec.get("exclusion_sources", [])
    if [row.get("path") for row in frozen_sources] != expected_source_paths:
        raise RuntimeError("frozen exclusion source paths are missing, unexpected, or reordered")
    if selection_spec.get("candidate_rule") != "episode IDs with ep_len >= 26":
        raise RuntimeError("episode eligibility threshold must remain ep_len >= 26")
    manifest_binding = selection_spec.get("selection_manifest_schema")
    if manifest_binding != OUTPUT_SCHEMA:
        raise RuntimeError("selection manifest schema differs from the frozen schema")
    selection_job_id = selection_spec.get("selection_manifest_job_id")
    if selection_job_id != job_id:
        raise RuntimeError("the 80-task manifest is already materialized; do not resubmit selection under this freeze")
    expected_output = f"/scratch/users/ntu/yguo017/lewm-pusht-iteration/artifacts/student-induced-shadow-pilot/real-observation-training-data/selection/{selection_job_id}/selection_manifest.json"
    if selection_spec.get("selection_manifest_path") != expected_output:
        raise RuntimeError("selection manifest artifact path differs from the freeze")
    sources, source_identity = load_exclusions(args, freeze)

    # Register existing codecs; this stage reads only the four frozen metadata columns.
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        required = {"ep_len", "ep_offset", "episode_idx", "step_idx"}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise RuntimeError(f"HDF5 is missing frozen scalar metadata columns: {missing}")
        lengths_column = handle["ep_len"]
        offsets_column = handle["ep_offset"]
        episode_column = handle["episode_idx"]
        step_column = handle["step_idx"]
        if lengths_column.ndim != 1 or offsets_column.ndim != 1 or lengths_column.shape != offsets_column.shape:
            raise RuntimeError("ep_len and ep_offset must be aligned one-dimensional episode metadata")
        if episode_column.ndim != 1 or step_column.ndim != 1 or episode_column.shape != step_column.shape:
            raise RuntimeError("episode_idx and step_idx must be aligned one-dimensional row metadata")
        if any(column.dtype.kind not in "iu" for column in (lengths_column, offsets_column, episode_column, step_column)):
            raise RuntimeError("all frozen HDF5 metadata columns must use integer dtypes")
        episode_count = int(lengths_column.shape[0])
        row_count = int(episode_column.shape[0])
        lengths = [int(value) for value in lengths_column[:]]
        offsets = [int(value) for value in offsets_column[:]]

        valid_ids = [episode_id for episode_id, length in enumerate(lengths) if length >= 26]
        prior_rule = source_identity["prior_valid_rule"]
        prior_order = list(valid_ids)
        random.Random(int(prior_rule["shuffle_seed"])).shuffle(prior_order)
        prior_ids = set(prior_order[: int(prior_rule["slice_stop_exclusive"])])
        if len(prior_ids) != 600:
            raise RuntimeError("reconstructed valid-prefix exclusion must contain exactly 600 episode IDs")

        exclusion_sets = {**sources, "prior_valid_prefix_600": prior_ids}
        excluded_ids = set().union(*exclusion_sets.values())
        eligible_ids = sorted(episode_id for episode_id in valid_ids if episode_id not in excluded_ids)
        if len(eligible_ids) < 80:
            raise RuntimeError(f"only {len(eligible_ids)} eligible new episodes remain; need 80")
        rng = random.Random(int(selection_spec["selection_seed"]))
        rng.shuffle(eligible_ids)
        chosen = eligible_ids[:80]
        tasks = []
        for order, episode_id in enumerate(chosen):
            start_step = rng.randint(0, lengths[episode_id] - 26)
            row_index = offsets[episode_id] + start_step
            if row_index < 0 or row_index >= row_count:
                raise RuntimeError(f"selected row {row_index} is outside HDF5 row metadata")
            if int(episode_column[row_index]) != episode_id or int(step_column[row_index]) != start_step:
                raise RuntimeError(f"scalar row identity mismatch for selected episode {episode_id}")
            tasks.append({
                "split": "collection_train" if order < 64 else "collection_validation",
                "selection_order": order,
                "episode_idx": episode_id,
                "row_index": int(row_index),
                "start_step": start_step,
            })

    selected_ids = {int(task["episode_idx"]) for task in tasks}
    if len(tasks) != 80 or len(selected_ids) != 80 or selected_ids & excluded_ids:
        raise RuntimeError("selection is not 80 unique episodes disjoint from all frozen exclusions")
    if sum(task["split"] == "collection_train" for task in tasks) != 64 or sum(task["split"] == "collection_validation" for task in tasks) != 16:
        raise RuntimeError("frozen train/validation split count mismatch")

    exclusion_counts = {name: len(ids) for name, ids in exclusion_sets.items()}
    exclusion_counts["union_unique_episode_count"] = len(excluded_ids)
    output = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "status": "COMPLETE_METADATA_ONLY_SELECTION",
        "pbs_job_id": job_id,
        "compute_hostname": host,
        "freeze_path": str(args.freeze.resolve()),
        "dataset_path": str(dataset_path),
        "selection_seed": int(selection_spec["selection_seed"]),
        "selection_rng": "random.Random",
        "selected_episode_count": len(tasks),
        "split_counts": {"collection_train": 64, "collection_validation": 16},
        "metadata_only": {
            "columns_read": ["ep_len", "ep_offset", "episode_idx[row_index]", "step_idx[row_index]"],
            "pixels_actions_or_model_weights_read": False,
            "episode_count": episode_count,
            "hdf5_row_count": row_count,
            "candidate_count_ep_len_ge_26": len(valid_ids),
            "eligible_count_after_exclusions": len(eligible_ids),
        },
        "exclusion_sources": source_identity["identity"],
        "exclusion_counts": exclusion_counts,
        "ranker_validation_subset_of_reserved_assertion": True,
        "selected_overlap_with_exclusions": 0,
        "split_overlap_episode_count": 0,
        "tasks": tasks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary selection file: {temporary}")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({"status": output["status"], "tasks": len(tasks), "train": 64, "validation": 16, "output": str(output_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
