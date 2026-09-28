#!/usr/bin/env python3
"""Select fixed PushT source episodes from HDF5 scalar metadata only."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--baseline-tasks", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_compute_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id:
        raise RuntimeError("PBS_JOBID is required")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_NODEFILE must name an existing nodefile")

    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    node_hosts = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in node_hosts:
        raise RuntimeError(f"hostname {host} is not present in PBS_NODEFILE")
    if any(any(token in node for token in ("login", "head", "submit")) for node in node_hosts):
        raise RuntimeError("PBS_NODEFILE contains a probable login/submit host")
    return job_id, host


def require_freeze(freeze: dict[str, Any]) -> None:
    if freeze.get("schema") != "lewm-pusht-student-induced-shadow-episode-selection":
        raise RuntimeError("selection freeze schema mismatch")
    if freeze.get("schema_version") != 1:
        raise RuntimeError("unsupported selection freeze version")

    dataset = freeze.get("dataset", {})
    if dataset.get("goal_offset_steps") != 25:
        raise RuntimeError("frozen goal_offset_steps must remain 25")
    if dataset.get("path") != "/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home/pusht_expert_train.h5":
        raise RuntimeError("frozen dataset path mismatch")
    if dataset.get("metadata_columns") != ["ep_len", "ep_offset", "episode_idx[row_index]", "step_idx[row_index]"]:
        raise RuntimeError("frozen HDF5 metadata-column contract mismatch")

    legacy = freeze.get("legacy_exclusions", {})
    baseline = legacy.get("closed_loop_50", {})
    baseline_ids = [int(value) for value in baseline.get("episode_ids", [])]
    if baseline.get("task_count") != 50 or len(baseline_ids) != 50 or len(set(baseline_ids)) != 50:
        raise RuntimeError("freeze must contain the exact 50 unique baseline episode IDs")
    if baseline.get("source_job_id") != "25534994.pbs101":
        raise RuntimeError("baseline selection source job mismatch")
    prior_valid = legacy.get("prior_valid_prefix", {})
    if (prior_valid.get("episode_length_threshold_inclusive"), prior_valid.get("shuffle_seed"), prior_valid.get("slice_stop_exclusive")) != (26, 20300903, 600):
        raise RuntimeError("legacy valid-prefix selector contract mismatch")

    selection = freeze.get("candidate_selection", {})
    if selection.get("seed") != 20260924 or selection.get("rng") != "random.Random":
        raise RuntimeError("candidate selection seed/RNG mismatch")
    if (selection.get("unique_source_episodes"), selection.get("collection_count"), selection.get("reserved_holdout_count")) != (32, 16, 16):
        raise RuntimeError("frozen episode counts must remain 32 = 16 + 16")
    if freeze.get("shadow_diagnostic_freeze", {}).get("minimum_matched_episode_coverage") != 12:
        raise RuntimeError("shadow minimum matched-episode coverage mismatch")


def read_baseline_episode_ids(path: Path, expected_ids: list[int]) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise RuntimeError("baseline selected_tasks.json must contain exactly 50 tasks")
    episode_ids = [int(task["episode_idx"]) for task in tasks]
    if len(set(episode_ids)) != 50:
        raise RuntimeError("baseline selected_tasks.json must contain 50 unique source episodes")
    if sorted(episode_ids) != sorted(expected_ids):
        raise RuntimeError("baseline selected_tasks.json IDs differ from the pre-registered freeze")
    return sorted(episode_ids)


def main() -> int:
    args = parse_args()
    job_id, host = require_compute_allocation()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    for path in (args.freeze, args.baseline_tasks, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    require_freeze(freeze)
    expected_baseline_path = Path(freeze["legacy_exclusions"]["closed_loop_50"]["source_file"])
    if os.path.normpath(str(args.baseline_tasks)) != os.path.normpath(str(expected_baseline_path)):
        raise RuntimeError("baseline selected_tasks.json path differs from the freeze")
    baseline_ids = read_baseline_episode_ids(
        args.baseline_tasks,
        [int(value) for value in freeze["legacy_exclusions"]["closed_loop_50"]["episode_ids"]],
    )

    # Register codecs without installing anything; only small scalar metadata is read below.
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(args.dataset, "r") as handle:
        required = {"ep_len", "ep_offset", "episode_idx", "step_idx"}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise RuntimeError(f"HDF5 is missing required small metadata columns: {missing}")
        ep_len_column = handle["ep_len"]
        ep_offset_column = handle["ep_offset"]
        episode_column = handle["episode_idx"]
        step_column = handle["step_idx"]
        if len(ep_len_column.shape) != 1 or len(ep_offset_column.shape) != 1:
            raise RuntimeError("ep_len and ep_offset must be one-dimensional metadata columns")
        if ep_len_column.shape != ep_offset_column.shape:
            raise RuntimeError("ep_len and ep_offset lengths differ")
        if episode_column.shape != step_column.shape or len(episode_column.shape) != 1:
            raise RuntimeError("episode_idx and step_idx must be aligned one-dimensional row columns")
        if ep_len_column.dtype.kind not in "iu" or ep_offset_column.dtype.kind not in "iu":
            raise RuntimeError("ep_len and ep_offset must have integer dtypes")
        if episode_column.dtype.kind not in "iu" or step_column.dtype.kind not in "iu":
            raise RuntimeError("episode_idx and step_idx must have integer dtypes")

        episode_count = int(ep_len_column.shape[0])
        hdf5_row_count = int(episode_column.shape[0])
        ep_lengths = [int(value) for value in ep_len_column[:]]
        ep_offsets = [int(value) for value in ep_offset_column[:]]
        valid_episodes = [episode for episode, length in enumerate(ep_lengths) if length >= 26]
        valid_rng = random.Random(20300903)
        valid_rng.shuffle(valid_episodes)
        if len(valid_episodes) < 600:
            raise RuntimeError(f"reconstructed valid episode list has only {len(valid_episodes)} IDs; need 600")
        prior_valid_ids = valid_episodes[:600]

        excluded_ids = set(baseline_ids) | set(prior_valid_ids)
        eligible_episodes = sorted(
            episode for episode in valid_episodes if episode not in excluded_ids
        )
        if len(eligible_episodes) < 32:
            raise RuntimeError(f"only {len(eligible_episodes)} eligible source episodes remain; need 32")

        candidate_rng = random.Random(20260924)
        candidate_rng.shuffle(eligible_episodes)
        selected_episode_ids = eligible_episodes[:32]
        selected_tasks = []
        for index, episode_id in enumerate(selected_episode_ids):
            max_start_step = ep_lengths[episode_id] - 26
            start_step = candidate_rng.randint(0, max_start_step)
            row_index = ep_offsets[episode_id] + start_step
            if row_index < 0 or row_index >= int(episode_column.shape[0]):
                raise RuntimeError(f"computed row_index is outside HDF5 rows for episode {episode_id}")
            observed_episode = int(episode_column[row_index])
            observed_step = int(step_column[row_index])
            if observed_episode != episode_id or observed_step != start_step:
                raise RuntimeError(
                    f"row identity mismatch for episode {episode_id}: row {row_index} maps to "
                    f"episode {observed_episode}, step {observed_step}"
                )
            selected_tasks.append(
                {
                    "role": "collection" if index < 16 else "reserved_holdout",
                    "selection_order": index,
                    "episode_idx": int(episode_id),
                    "row_index": int(row_index),
                    "start_step": int(start_step),
                }
            )

    selected_ids = [task["episode_idx"] for task in selected_tasks]
    if len(selected_ids) != 32 or len(set(selected_ids)) != 32:
        raise RuntimeError("selection is not 32 unique source episodes")
    if set(selected_ids) & excluded_ids:
        raise RuntimeError("selection overlaps baseline or prior valid-prefix exclusions")

    output = {
        "schema": "lewm-pusht-student-induced-shadow-episode-selection-result",
        "schema_version": 1,
        "pbs_job_id": job_id,
        "compute_hostname": host,
        "dataset_path": str(args.dataset),
        "selection_freeze": freeze,
        "metadata_only": {
            "episode_metadata_read": ["ep_len", "ep_offset"],
            "row_metadata_scalar_reads": ["episode_idx[row_index]", "step_idx[row_index]"],
            "episode_count": episode_count,
            "hdf5_row_count": hdf5_row_count,
            "pixels_or_actions_read": False,
            "model_or_checkpoint_loaded": False,
        },
        "legacy_exclusions": {
            "baseline_source_job_id": "25534994.pbs101",
            "baseline_episode_ids": baseline_ids,
            "prior_valid_selector": "episode IDs enumerated from ep_len >= 26, shuffled by random.Random(20300903), first 600",
            "prior_valid_episode_ids": prior_valid_ids,
        },
        "selected_task_count": len(selected_tasks),
        "collection_count": sum(task["role"] == "collection" for task in selected_tasks),
        "reserved_holdout_count": sum(task["role"] == "reserved_holdout" for task in selected_tasks),
        "tasks": selected_tasks,
        "holdout_access_policy": "The shadow diagnostic may access collection tasks only; reserved_holdout is not for pilot gating or tuning.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SELECTION_COMPLETE", "tasks": len(selected_tasks), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
