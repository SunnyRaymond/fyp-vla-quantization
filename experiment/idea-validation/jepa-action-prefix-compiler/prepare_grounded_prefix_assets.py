#!/usr/bin/env python3
"""Prepare a small, deterministic grounded-prefix manifest.

This preparation step only inspects already-staged PushT trajectory metadata and
observation filenames.  It never copies trajectory files, loads model weights,
decodes videos, downloads data, or hashes files.  The resulting JSON manifest
contains relative episode/anchor references that a later GPU runner can consume
to obtain real future-observation targets instead of frozen teacher rollouts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_HORIZON = 5
DEFAULT_FRAME_SKIP = 5
DEFAULT_TRAIN_EPISODES = 32
DEFAULT_HELDOUT_EPISODES = 8
DEFAULT_ANCHORS_PER_EPISODE = 4
DEFAULT_EPISODE_SEED = 20260920
DEFAULT_TRAIN_ACTION_SEED = 20260920
DEFAULT_HELDOUT_ACTION_SEEDS = (20262920, 20262921)
DEFAULT_TIMING_ACTION_SEED = 20270921

REQUIRED_FILES = (
    "states.pth",
    "rel_actions.pth",
    "seq_lengths.pkl",
    "velocities.pth",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deps-root", type=Path, default=None)
    parser.add_argument("--deps-marker", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--heldout-episodes", type=int, default=DEFAULT_HELDOUT_EPISODES)
    parser.add_argument("--anchors-per-episode", type=int, default=DEFAULT_ANCHORS_PER_EPISODE)
    parser.add_argument("--episode-seed", type=int, default=DEFAULT_EPISODE_SEED)
    parser.add_argument("--train-action-seed", type=int, default=DEFAULT_TRAIN_ACTION_SEED)
    parser.add_argument(
        "--heldout-action-seeds",
        type=int,
        nargs=2,
        default=list(DEFAULT_HELDOUT_ACTION_SEEDS),
    )
    parser.add_argument("--timing-action-seed", type=int, default=DEFAULT_TIMING_ACTION_SEED)
    return parser.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing staged-data inspection outside a PBS allocation")
    if not os.environ.get("PBS_NODEFILE"):
        raise RuntimeError("PBS_NODEFILE is required; refusing staged-data inspection without a node allocation")
    host = os.uname().nodename.lower() if hasattr(os, "uname") else "unknown"
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/head/submit host: {host}")


def _require_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _read_lengths(path: Path) -> list[int]:
    # seq_lengths.pkl is the small metadata index used by the official loader.
    # It is read only after path resolution and is never written back.
    with path.open("rb") as handle:
        value = pickle.load(handle)

    if isinstance(value, dict):
        for key in ("seq_lengths", "lengths", "episode_lengths"):
            if key in value:
                value = value[key]
                break
        else:
            ordered = []
            for key, item in value.items():
                if isinstance(key, int):
                    ordered.append((key, item))
            if ordered and len(ordered) == len(value):
                value = [item for _, item in sorted(ordered)]
            else:
                raise ValueError(f"unsupported sequence-length mapping in {path}")

    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must contain a list/tuple of episode lengths")

    lengths: list[int] = []
    for index, item in enumerate(value):
        try:
            length = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid episode length at index {index}: {item!r}") from exc
        if length < 1:
            raise ValueError(f"episode {index} has non-positive length {length}")
        lengths.append(length)
    if not lengths:
        raise ValueError(f"{path} contains no episode lengths")
    return lengths


def _check_layout(data_root: Path, deps_root: Path | None, deps_marker: Path | None) -> dict[str, Any]:
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {data_root}")

    split_info: dict[str, Any] = {}
    for split in ("train", "val"):
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing staged split directory: {split_root}")
        missing = [name for name in REQUIRED_FILES if not (split_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing staged metadata in {split_root}: {', '.join(missing)}")
        observations = split_root / "obses"
        if not observations.is_dir():
            raise FileNotFoundError(f"missing staged observation directory: {observations}")
        lengths_path = split_root / "seq_lengths.pkl"
        lengths = _read_lengths(lengths_path)
        split_info[split] = {
            "relative_root": split,
            "episode_count": len(lengths),
            "sequence_length_min": min(lengths),
            "sequence_length_max": max(lengths),
            "metadata_files": list(REQUIRED_FILES),
            "observations_dir": f"{split}/obses",
            "sequence_lengths": lengths,
        }

    dependency_info = {"python": sys.executable, "python_version": sys.version.split()[0]}
    for module_name in ("torch", "numpy", "hydra", "omegaconf"):
        dependency_info[f"find_spec_{module_name}"] = importlib.util.find_spec(module_name) is not None
    if deps_root is not None:
        deps_root = deps_root.resolve()
        dependency_info["deps_root"] = str(deps_root)
        dependency_info["deps_root_present"] = deps_root.is_dir()
    if deps_marker is not None:
        deps_marker = deps_marker.resolve()
        dependency_info["deps_marker"] = str(deps_marker)
        dependency_info["deps_marker_present"] = deps_marker.is_file()

    return {"data_root": str(data_root), "splits": split_info, "dependencies": dependency_info}


def _eligible_episodes(lengths: Iterable[int], horizon: int, frame_skip: int) -> list[int]:
    required_span = horizon * frame_skip
    return [index for index, length in enumerate(lengths) if length > required_span]


def _select_episodes(eligible: list[int], count: int, seed: int, split: str) -> list[int]:
    if len(eligible) < count:
        raise ValueError(f"{split} has only {len(eligible)} eligible episodes; need {count}")
    rng = random.Random(seed)
    selected = sorted(rng.sample(eligible, count))
    return selected


def _anchors_for_episode(
    episode_id: int,
    length: int,
    horizon: int,
    frame_skip: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    max_start = length - horizon * frame_skip - 1
    candidates = list(range(max_start + 1))
    if len(candidates) < count:
        raise ValueError(
            f"episode {episode_id} has only {len(candidates)} valid starts; need {count}"
        )
    starts = sorted(random.Random(seed).sample(candidates, count))
    anchors = []
    for anchor_index, start_step in enumerate(starts):
        target_steps = [start_step + (index + 1) * frame_skip for index in range(horizon)]
        anchors.append(
            {
                "anchor_id": f"episode_{episode_id:03d}_start_{start_step:06d}",
                "anchor_index": anchor_index,
                "episode_id": episode_id,
                "start_step": start_step,
                "target_steps": target_steps,
            }
        )
    return anchors


def _build_manifest(args: argparse.Namespace, layout: dict[str, Any]) -> dict[str, Any]:
    horizon = int(args.horizon)
    frame_skip = int(args.frame_skip)
    _require_positive("horizon", horizon)
    _require_positive("frame_skip", frame_skip)
    _require_positive("train_episodes", args.train_episodes)
    _require_positive("heldout_episodes", args.heldout_episodes)
    _require_positive("anchors_per_episode", args.anchors_per_episode)

    train_lengths = layout["splits"]["train"]["sequence_lengths"]
    val_lengths = layout["splits"]["val"]["sequence_lengths"]
    train_eligible = _eligible_episodes(train_lengths, horizon, frame_skip)
    val_eligible = _eligible_episodes(val_lengths, horizon, frame_skip)
    train_ids = _select_episodes(train_eligible, args.train_episodes, args.episode_seed, "train")
    val_ids = _select_episodes(
        val_eligible,
        args.heldout_episodes,
        args.episode_seed + 1,
        "val",
    )

    train_anchors = []
    for episode_id in train_ids:
        train_anchors.extend(
            _anchors_for_episode(
                episode_id,
                train_lengths[episode_id],
                horizon,
                frame_skip,
                args.anchors_per_episode,
                args.episode_seed + episode_id,
            )
        )
    heldout_anchors = []
    for episode_id in val_ids:
        heldout_anchors.extend(
            _anchors_for_episode(
                episode_id,
                val_lengths[episode_id],
                horizon,
                frame_skip,
                args.anchors_per_episode,
                args.episode_seed + 10000 + episode_id,
            )
        )

    def with_files(split: str, lengths: list[int], anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for anchor in anchors:
            item = dict(anchor)
            episode_id = int(item["episode_id"])
            observation_path = (
                Path(layout["data_root"])
                / split
                / "obses"
                / f"episode_{episode_id:03d}.mp4"
            )
            if not observation_path.is_file():
                raise FileNotFoundError(
                    f"selected episode observation is missing: {observation_path}"
                )
            item["episode_length"] = lengths[episode_id]
            item["observation_file"] = f"{split}/obses/episode_{episode_id:03d}.mp4"
            item["metadata_root"] = split
            result.append(item)
        return result

    train_examples = with_files("train", train_lengths, train_anchors)
    heldout_examples = with_files("val", val_lengths, heldout_anchors)
    return {
        "schema": "jepa-action-prefix-compiler.grounded-prefix-manifest",
        "schema_version": 1,
        "source": {
            "data_root": layout["data_root"],
            "layout_verified": True,
            "files_copied": False,
            "hashes_computed": False,
            "required_metadata_files": list(REQUIRED_FILES),
        },
        "protocol": {
            "horizon": horizon,
            "frame_skip": frame_skip,
            "target_semantics": "real future observation indices from the staged trajectory",
            "episode_selection_seed": args.episode_seed,
            "train_action_prefix_seed": args.train_action_seed,
            "heldout_action_prefix_seeds": [int(value) for value in args.heldout_action_seeds],
            "timing_action_prefix_seed": args.timing_action_seed,
            "train_episode_count": len(train_ids),
            "heldout_episode_count": len(val_ids),
            "anchors_per_episode": args.anchors_per_episode,
        },
        "files": {
            "states": "{split}/states.pth",
            "rel_actions": "{split}/rel_actions.pth",
            "velocities": "{split}/velocities.pth",
            "sequence_lengths": "{split}/seq_lengths.pkl",
            "observations": "{split}/obses/episode_{episode_id:03d}.mp4",
        },
        "splits": {
            "train": {"episode_ids": train_ids, "examples": train_examples},
            "heldout": {"source_split": "val", "episode_ids": val_ids, "examples": heldout_examples},
        },
    }


def main() -> int:
    _require_compute_node()
    args = _args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    layout = _check_layout(args.data_root, args.deps_root, args.deps_marker)
    manifest = _build_manifest(args, layout)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"manifest={output}")
    print(f"train_examples={len(manifest['splits']['train']['examples'])}")
    print(f"heldout_examples={len(manifest['splits']['heldout']['examples'])}")
    print(f"horizon={manifest['protocol']['horizon']} frame_skip={manifest['protocol']['frame_skip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
