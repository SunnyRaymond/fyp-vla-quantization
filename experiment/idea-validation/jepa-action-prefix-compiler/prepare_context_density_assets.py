#!/usr/bin/env python3
"""Prepare the metadata-only context-support density manifest.

This preparation step extends one already-frozen query-coverage manifest.  It
keeps the original 128 train context records byte-for-byte in meaning (the
records are compared field-by-field), then appends four new valid starts for
each of the same 32 train episodes.  The eight held-out records are copied
unchanged.  Only ``seq_lengths.pkl``, manifest JSON, and observation filename
entries are inspected; videos and trajectory contents are never read.

The extra-start rule is deterministic and pre-registered: for episode ``i``
we use ``random.Random(20260928 + i)`` on the legal starts after removing the
four starts already present in the base manifest.  The per-episode seed
follows the selection convention used by ``prepare_query_coverage_assets``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from prepare_query_coverage_assets import (
    _eligible_episode_ids,
    _inspect_split,
    _require_compute_node,
    _verify_observation_filenames,
)


DEFAULT_EXTRA_CONTEXTS_PER_EPISODE = 4
DEFAULT_EXTRA_SELECTION_SEED = 20260928
BASE_TRAIN_CONTEXTS = 128
BASE_TRAIN_CONTEXTS_PER_EPISODE = 4
TRAIN_EPISODES = 32
HELDOUT_CONTEXTS = 8
HELDOUT_EPISODES = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-contexts-per-episode",
        type=int,
        default=DEFAULT_EXTRA_CONTEXTS_PER_EPISODE,
    )
    parser.add_argument(
        "--extra-selection-seed",
        type=int,
        default=DEFAULT_EXTRA_SELECTION_SEED,
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _require_int(mapping: dict[str, Any], key: str, where: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.{key} must be an integer")
    return value


def _context_key(context: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(context["split"]),
        int(context["episode_id"]),
        int(context["start_step"]),
    )


def _validate_context(
    context: dict[str, Any],
    *,
    split: str,
    lengths: list[int],
    horizon: int,
    frame_skip: int,
) -> None:
    """Validate one metadata record without opening its observation file."""

    if context.get("split") != split:
        raise AssertionError(f"context split mismatch: {context!r}")
    episode_id = _require_int(context, "episode_id", "context")
    if episode_id < 0 or episode_id >= len(lengths):
        raise AssertionError(f"episode id is outside seq_lengths.pkl: {episode_id}")
    episode_length = _require_int(context, "episode_length", "context")
    if episode_length != lengths[episode_id]:
        raise AssertionError(
            f"episode length mismatch for {split}:{episode_id}: "
            f"{episode_length} != {lengths[episode_id]}"
        )
    start = _require_int(context, "start", "context")
    start_step = _require_int(context, "start_step", "context")
    if start != start_step or start < 0:
        raise AssertionError(f"invalid start fields: {context!r}")
    target_steps = context.get("target_steps")
    expected_targets = [
        start + (offset + 1) * frame_skip for offset in range(horizon)
    ]
    if target_steps != expected_targets:
        raise AssertionError(
            f"target_steps mismatch for {split}:{episode_id}:{start}: "
            f"{target_steps!r} != {expected_targets!r}"
        )
    if not target_steps or target_steps[-1] >= episode_length:
        raise AssertionError(f"start does not leave a complete horizon: {context!r}")
    if context.get("terminal_frame_index") != target_steps[-1]:
        raise AssertionError(f"terminal frame mismatch: {context!r}")
    if context.get("goal_frame_index") != target_steps[-1]:
        raise AssertionError(f"goal frame mismatch: {context!r}")
    expected_key = f"{split}:episode_{episode_id:03d}"
    if context.get("episode_key") != expected_key:
        raise AssertionError(f"episode key mismatch: {context!r}")
    expected_id = f"{split}_episode_{episode_id:03d}_start_{start:06d}"
    if context.get("context_id") != expected_id:
        raise AssertionError(f"context id mismatch: {context!r}")
    expected_observation = f"{split}/obses/episode_{episode_id:03d}.mp4"
    if context.get("observation_file") != expected_observation:
        raise AssertionError(f"observation filename mismatch: {context!r}")


def _validate_base_manifest(
    manifest: dict[str, Any],
    *,
    train_lengths: list[int],
    heldout_lengths: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int], int, int]:
    if manifest.get("schema") != "jepa-action-prefix-compiler.query-coverage-manifest":
        raise ValueError("--query-manifest is not the frozen query-coverage manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported query-coverage manifest schema version")
    protocol = manifest.get("protocol")
    splits = manifest.get("splits")
    if not isinstance(protocol, dict) or not isinstance(splits, dict):
        raise ValueError("query-coverage manifest is missing protocol or splits")
    horizon = _require_int(protocol, "horizon", "protocol")
    frame_skip = _require_int(protocol, "frame_skip", "protocol")
    if horizon < 1 or frame_skip < 1:
        raise ValueError("query-coverage horizon and frame_skip must be positive")
    if protocol.get("train_episode_count") != TRAIN_EPISODES:
        raise ValueError("base manifest must contain exactly 32 train episodes")
    if protocol.get("train_contexts_per_episode") != BASE_TRAIN_CONTEXTS_PER_EPISODE:
        raise ValueError("base manifest must contain exactly four contexts per train episode")
    if protocol.get("train_context_count") != BASE_TRAIN_CONTEXTS:
        raise ValueError("base manifest must contain exactly 128 train contexts")
    if protocol.get("heldout_episode_count") != HELDOUT_EPISODES:
        raise ValueError("base manifest must contain exactly eight held-out episodes")
    if protocol.get("heldout_context_count") != HELDOUT_CONTEXTS:
        raise ValueError("base manifest must contain exactly eight held-out contexts")

    train = splits.get("train")
    heldout = splits.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError("query-coverage manifest must contain train and heldout splits")
    train_ids = train.get("episode_ids")
    heldout_ids = heldout.get("episode_ids")
    train_contexts = train.get("contexts")
    heldout_contexts = heldout.get("contexts")
    if not isinstance(train_ids, list) or not isinstance(heldout_ids, list):
        raise ValueError("base episode_ids must be lists")
    if not isinstance(train_contexts, list) or not isinstance(heldout_contexts, list):
        raise ValueError("base contexts must be lists")
    if len(train_ids) != TRAIN_EPISODES or len(heldout_ids) != HELDOUT_EPISODES:
        raise ValueError("base manifest episode counts do not match the frozen contract")
    if len(train_contexts) != BASE_TRAIN_CONTEXTS or len(heldout_contexts) != HELDOUT_CONTEXTS:
        raise ValueError("base manifest context counts do not match the frozen contract")
    if len(set(train_ids)) != len(train_ids) or len(set(heldout_ids)) != len(heldout_ids):
        raise ValueError("base episode IDs contain duplicates")
    if train.get("source_split") != "train" or heldout.get("source_split") != "val":
        raise ValueError("base manifest must use train and val source splits")

    eligible_train = set(_eligible_episode_ids(train_lengths, horizon, frame_skip))
    eligible_heldout = set(_eligible_episode_ids(heldout_lengths, horizon, frame_skip))
    if not set(train_ids) <= eligible_train:
        raise ValueError("base train episode selection contains an illegal horizon")
    if not set(heldout_ids) <= eligible_heldout:
        raise ValueError("base held-out episode selection contains an illegal horizon")

    for ordinal, context in enumerate(train_contexts):
        if not isinstance(context, dict):
            raise ValueError(f"train context {ordinal} is not an object")
        if context.get("context_ordinal") != ordinal:
            raise ValueError("base train context ordinals must be exactly 0..127")
        if context.get("context_index_in_episode") != ordinal % BASE_TRAIN_CONTEXTS_PER_EPISODE:
            raise ValueError("base train contexts are not in the frozen episode order")
        if context.get("episode_id") != train_ids[ordinal // BASE_TRAIN_CONTEXTS_PER_EPISODE]:
            raise ValueError("base train contexts are not grouped in episode order")
        _validate_context(
            context,
            split="train",
            lengths=train_lengths,
            horizon=horizon,
            frame_skip=frame_skip,
        )
    for ordinal, context in enumerate(heldout_contexts):
        if not isinstance(context, dict):
            raise ValueError(f"held-out context {ordinal} is not an object")
        if context.get("context_ordinal") != ordinal or context.get("context_index_in_episode") != 0:
            raise ValueError("base held-out context order is not the frozen order")
        if context.get("episode_id") != heldout_ids[ordinal]:
            raise ValueError("base held-out contexts are not grouped in episode order")
        _validate_context(
            context,
            split="val",
            lengths=heldout_lengths,
            horizon=horizon,
            frame_skip=frame_skip,
        )

    base_keys = [_context_key(context) for context in [*train_contexts, *heldout_contexts]]
    if len(base_keys) != len(set(base_keys)):
        raise ValueError("base train/held-out contexts contain duplicate starts")
    base_ids = [str(context["context_id"]) for context in [*train_contexts, *heldout_contexts]]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base train/held-out contexts contain duplicate context IDs")
    train_keys = {(context["split"], context["episode_id"]) for context in train_contexts}
    heldout_keys = {(context["split"], context["episode_id"]) for context in heldout_contexts}
    if train_keys & heldout_keys:
        raise ValueError("held-out episodes overlap train episodes")
    return train_contexts, heldout_contexts, train_ids, heldout_ids, horizon, frame_skip


def _select_extra_starts(
    *,
    episode_id: int,
    episode_length: int,
    horizon: int,
    frame_skip: int,
    old_starts: Iterable[int],
    count: int,
    base_seed: int,
) -> list[int]:
    required_span = horizon * frame_skip
    max_start = episode_length - required_span - 1
    if max_start < 0:
        raise ValueError(f"episode {episode_id} has no legal context start")
    old = set(old_starts)
    candidates = [start for start in range(max_start + 1) if start not in old]
    if len(candidates) < count:
        raise ValueError(
            f"episode {episode_id} has {len(candidates)} legal unused starts; need {count}"
        )
    return sorted(random.Random(base_seed + episode_id).sample(candidates, count))


def _make_extra_context(
    *,
    episode_id: int,
    episode_length: int,
    start: int,
    horizon: int,
    frame_skip: int,
    context_index: int,
    ordinal: int,
    query_seed: int,
    selection_seed: int,
) -> dict[str, Any]:
    target_steps = [start + (offset + 1) * frame_skip for offset in range(horizon)]
    return {
        "context_id": f"train_episode_{episode_id:03d}_start_{start:06d}",
        "split": "train",
        "episode_id": episode_id,
        "episode_key": f"train:episode_{episode_id:03d}",
        "context_index_in_episode": context_index,
        "context_ordinal": ordinal,
        "episode_length": episode_length,
        "observation_file": f"train/obses/episode_{episode_id:03d}.mp4",
        "start": start,
        "start_step": start,
        "target_steps": target_steps,
        "terminal_frame_index": target_steps[-1],
        "goal_frame_index": target_steps[-1],
        "context_selection_seed": selection_seed,
        "query_seeds": [query_seed],
        "query_seed": query_seed,
    }


def _validate_density_manifest(
    result: dict[str, Any],
    *,
    base_train: list[dict[str, Any]],
    base_heldout: list[dict[str, Any]],
    train_ids: list[int],
    heldout_ids: list[int],
    train_lengths: list[int],
    heldout_lengths: list[int],
    horizon: int,
    frame_skip: int,
    extra_count: int,
) -> None:
    train_contexts = result["splits"]["train"]["contexts"]
    heldout_contexts = result["splits"]["heldout"]["contexts"]
    if train_contexts[:BASE_TRAIN_CONTEXTS] != base_train:
        raise AssertionError("old 128 train records were not preserved field-by-field")
    if heldout_contexts != base_heldout:
        raise AssertionError("held-out records were changed")
    if len(train_contexts) != TRAIN_EPISODES * (BASE_TRAIN_CONTEXTS_PER_EPISODE + extra_count):
        raise AssertionError("density train context count is incorrect")
    if [context["context_ordinal"] for context in train_contexts] != list(range(len(train_contexts))):
        raise AssertionError("train context ordinals are not contiguous")
    for ordinal, context in enumerate(train_contexts):
        _validate_context(
            context,
            split="train",
            lengths=train_lengths,
            horizon=horizon,
            frame_skip=frame_skip,
        )
        if ordinal < BASE_TRAIN_CONTEXTS:
            expected_episode = train_ids[ordinal // BASE_TRAIN_CONTEXTS_PER_EPISODE]
        else:
            extra_ordinal = ordinal - BASE_TRAIN_CONTEXTS
            expected_episode = train_ids[extra_ordinal // extra_count]
        if context["episode_id"] != expected_episode:
            raise AssertionError("density context episode order changed within the old or appended segment")
    for ordinal, context in enumerate(heldout_contexts):
        _validate_context(
            context,
            split="val",
            lengths=heldout_lengths,
            horizon=horizon,
            frame_skip=frame_skip,
        )
        if context["episode_id"] != heldout_ids[ordinal]:
            raise AssertionError("held-out episode order changed")

    keys = [_context_key(context) for context in [*train_contexts, *heldout_contexts]]
    if len(keys) != len(set(keys)):
        raise AssertionError("density manifest contains duplicate starts")
    context_ids = [str(context["context_id"]) for context in [*train_contexts, *heldout_contexts]]
    if len(context_ids) != len(set(context_ids)):
        raise AssertionError("density manifest contains duplicate context IDs")
    old_starts_by_episode: dict[int, set[int]] = {}
    for context in base_train:
        old_starts_by_episode.setdefault(int(context["episode_id"]), set()).add(
            int(context["start_step"])
        )
    for episode_id in train_ids:
        contexts = [
            context for context in train_contexts if int(context["episode_id"]) == episode_id
        ]
        if len(contexts) != BASE_TRAIN_CONTEXTS_PER_EPISODE + extra_count:
            raise AssertionError("each train episode does not have the requested density")
        extra = contexts[BASE_TRAIN_CONTEXTS_PER_EPISODE:]
        if any(context["context_index_in_episode"] != BASE_TRAIN_CONTEXTS_PER_EPISODE + i for i, context in enumerate(extra)):
            raise AssertionError("extra context indices are not appended per episode")
        if any(int(context["start_step"]) in old_starts_by_episode[episode_id] for context in extra):
            raise AssertionError("an extra context reused an old start")

    train_episode_keys = {(c["split"], c["episode_id"]) for c in train_contexts}
    heldout_episode_keys = {(c["split"], c["episode_id"]) for c in heldout_contexts}
    if train_episode_keys & heldout_episode_keys:
        raise AssertionError("held-out episode entered the train split")


def _build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.extra_contexts_per_episode != DEFAULT_EXTRA_CONTEXTS_PER_EPISODE:
        raise ValueError("extra_contexts_per_episode is frozen at 4")
    data_root = args.data_root.resolve()
    query_manifest_path = args.query_manifest.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {data_root}")
    if not query_manifest_path.is_file():
        raise FileNotFoundError(f"missing query-coverage manifest: {query_manifest_path}")

    train_layout = _inspect_split(data_root, "train")
    heldout_layout = _inspect_split(data_root, "val")
    base = _load_json(query_manifest_path)
    (
        base_train,
        base_heldout,
        train_ids,
        heldout_ids,
        horizon,
        frame_skip,
    ) = _validate_base_manifest(
        base,
        train_lengths=train_layout["sequence_lengths"],
        heldout_lengths=heldout_layout["sequence_lengths"],
    )

    train_query_seed = int(base["protocol"]["train_query_seed"])
    extra_contexts: list[dict[str, Any]] = []
    for episode_id in train_ids:
        episode_length = train_layout["sequence_lengths"][episode_id]
        old_contexts = [context for context in base_train if context["episode_id"] == episode_id]
        old_starts = [int(context["start_step"]) for context in old_contexts]
        selection_seed = args.extra_selection_seed + episode_id
        starts = _select_extra_starts(
            episode_id=episode_id,
            episode_length=episode_length,
            horizon=horizon,
            frame_skip=frame_skip,
            old_starts=old_starts,
            count=args.extra_contexts_per_episode,
            base_seed=args.extra_selection_seed,
        )
        for index, start in enumerate(starts, start=BASE_TRAIN_CONTEXTS_PER_EPISODE):
            ordinal = BASE_TRAIN_CONTEXTS + len(extra_contexts)
            query_seed = train_query_seed + ordinal
            extra_contexts.append(
                _make_extra_context(
                    episode_id=episode_id,
                    episode_length=episode_length,
                    start=start,
                    horizon=horizon,
                    frame_skip=frame_skip,
                    context_index=index,
                    ordinal=ordinal,
                    query_seed=query_seed,
                    selection_seed=selection_seed,
                )
            )

    _verify_observation_filenames(data_root, [*base_train, *extra_contexts, *base_heldout])

    # Keep the old records as the actual list prefix; only protocol/provenance
    # metadata and the appended train records are new.
    result = dict(base)
    result["schema"] = "jepa-action-prefix-compiler.context-density-manifest"
    result["schema_version"] = 1
    source = dict(base["source"])
    source.update(
        {
            "data_root": str(data_root),
            "base_manifest": str(query_manifest_path),
            "base_manifest_schema": base["schema"],
            "base_train_prefix_preserved": True,
            "files_copied": False,
            "videos_decoded": False,
            "models_loaded": False,
            "hashes_computed": False,
            "observation_filenames_only": True,
        }
    )
    result["source"] = source
    protocol = dict(base["protocol"])
    protocol.update(
        {
            "train_contexts_per_episode": BASE_TRAIN_CONTEXTS_PER_EPISODE + args.extra_contexts_per_episode,
            "train_context_count": len(base_train) + len(extra_contexts),
            "base_train_context_count": BASE_TRAIN_CONTEXTS,
            "extra_contexts_per_episode": args.extra_contexts_per_episode,
            "extra_context_selection_seed": args.extra_selection_seed,
            "context_order": "frozen 128-context prefix followed by four sorted extra starts per episode",
        }
    )
    result["protocol"] = protocol
    result["splits"] = dict(base["splits"])
    result["splits"]["train"] = dict(base["splits"]["train"])
    result["splits"]["heldout"] = dict(base["splits"]["heldout"])
    result["splits"]["train"]["contexts"] = base_train + extra_contexts
    # Explicitly retain the exact base held-out list and episode list.
    result["splits"]["heldout"]["contexts"] = base_heldout
    _validate_density_manifest(
        result,
        base_train=base_train,
        base_heldout=base_heldout,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        train_lengths=train_layout["sequence_lengths"],
        heldout_lengths=heldout_layout["sequence_lengths"],
        horizon=horizon,
        frame_skip=frame_skip,
        extra_count=args.extra_contexts_per_episode,
    )
    return result


def main() -> int:
    _require_compute_node()
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    manifest = _build_manifest(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"manifest={output}")
    print(f"train_episodes={manifest['protocol']['train_episode_count']}")
    print(f"train_contexts={manifest['protocol']['train_context_count']}")
    print(f"heldout_episodes={manifest['protocol']['heldout_episode_count']}")
    print(f"heldout_contexts={manifest['protocol']['heldout_context_count']}")
    print(f"horizon={manifest['protocol']['horizon']} frame_skip={manifest['protocol']['frame_skip']}")
    print(f"extra_selection_seed={manifest['protocol']['extra_context_selection_seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
