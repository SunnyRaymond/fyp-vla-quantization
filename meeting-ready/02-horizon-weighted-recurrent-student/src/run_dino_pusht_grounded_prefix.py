#!/usr/bin/env python3
"""Paired grounded-prefix training for the official DINO-WM PushT model.

The experiment keeps the post-hoc student and its inference interface fixed,
but changes only the dense training target:

``narrow_teacher``
    real manifest segments restricted to the first two train episodes;
``wide_teacher``
    frozen teacher-rollout targets on manifest trajectory segments;
``wide_grounded``
    real future observation latents from the same manifest segments and
    frozen ``encode_obs``.

The two wide arms consume exactly the same manifest examples and action
prefixes; all three arms share an initial state dictionary and update order.
This is a predictor-level
experiment.  Model loading and video decoding are refused outside a PBS
compute allocation; this file performs no downloads or environment setup.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_dino_pusht_stage_a import (  # noqa: E402
    ANCHOR_COUNT,
    HORIZON,
    NativeDinoPrefixStudent,
    _gpu_snapshot,
    _json_default,
    _latency,
    _leakage_test,
    _load_json,
    _load_official,
    _metrics_by_horizon,
    _objective_cost,
    _repeat_anchor,
    _require_assets,
    _require_compute_node,
    _sample_actions,
    _set_seed,
    _spearman,
    _teacher_targets,
    _topk_overlap,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--deps-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _required(mapping: Mapping[str, Any], *keys: str, path: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"protocol is missing {path}; accepted fields: {', '.join(keys)}")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _settings(protocol: Mapping[str, Any]) -> dict[str, Any]:
    training = protocol.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("grounded-prefix protocol must contain training")
    data = protocol.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError("grounded-prefix protocol data must be an object")
    heldout = protocol.get("heldout", {})
    timing = protocol.get("timing", {})
    if not isinstance(heldout, Mapping) or not isinstance(timing, Mapping):
        raise ValueError("grounded-prefix protocol heldout/timing must be objects")
    seeds = protocol.get("anchors_and_seeds", {})
    if not isinstance(seeds, Mapping):
        seeds = {}
    train_seed = int(
        seeds.get("training_action_prefix_seed", training.get("seed", 20260920))
    )
    heldout_seeds = seeds.get(
        "heldout_action_prefix_seeds",
        heldout.get("action_prefix_seeds", [20262920, 20262921]),
    )
    timing_seed = int(
        seeds.get("timing_action_prefix_seed", timing.get("action_prefix_seed", 20270921))
    )
    return {
        "steps": int(_required(training, "steps", path="training.steps")),
        "batch": int(_required(training, "batch_size", "batch", path="training.batch_size")),
        "lr": float(_required(training, "learning_rate", "lr", path="training.learning_rate")),
        "hidden_dim": int(_required(training, "hidden_dim", path="training.hidden_dim")),
        "frame_skip": int(data.get("frame_skip", protocol.get("frame_skip", 5))),
        "cache_episodes": int(data.get("cache_episodes", 8)),
        "preencode_chunk": int(data.get("preencode_chunk", data.get("chunk_size", 8))),
        "train_seed": train_seed,
        "heldout_seeds": [int(value) for value in heldout_seeds],
        "timing_seed": timing_seed,
        "eval_batch": int(heldout.get("candidates_per_block", timing.get("batch_size", 300))),
        "timing_batch": int(timing.get("batch_size", 300)),
        "warmup": int(timing.get("warmup_repeats", 3)),
        "repeats": int(timing.get("technical_repeats", 10)),
        "val_batches": int(data.get("validation_batches", 8)),
        "train_segments": int(data["train_segments"]) if "train_segments" in data else None,
        "heldout_segments": int(data["heldout_segments"]) if "heldout_segments" in data else None,
        "future_action_tolerance": float(
            protocol.get("controls", {}).get("future_action_tolerance", 1e-6)
            if isinstance(protocol.get("controls", {}), Mapping)
            else 1e-6
        ),
    }


def _load_manifest(path: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(path.resolve())
    if manifest.get("schema") != "jepa-action-prefix-compiler.grounded-prefix-manifest":
        raise ValueError("unexpected grounded-prefix manifest schema")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("grounded-prefix manifest has no protocol object")
    if int(protocol.get("horizon", -1)) != HORIZON:
        raise ValueError("manifest horizon does not match H=5")
    if int(protocol.get("frame_skip", -1)) != int(settings["frame_skip"]):
        raise ValueError("manifest frame_skip does not match frozen protocol")
    for split_name in ("train", "heldout"):
        split = manifest.get("splits", {}).get(split_name)
        if not isinstance(split, Mapping) or not split.get("examples"):
            raise ValueError(f"manifest has no non-empty {split_name}.examples")
        for example in split["examples"]:
            for key in ("episode_id", "start_step", "target_steps"):
                if key not in example:
                    raise ValueError(f"manifest example missing {key}")
            expected = [
                int(example["start_step"]) + (index + 1) * int(settings["frame_skip"])
                for index in range(HORIZON)
            ]
            if [int(value) for value in example["target_steps"]] != expected:
                raise ValueError("manifest target_steps are not contiguous frozen frame-skip targets")
    expected_counts = {
        "train": settings.get("train_segments"),
        "heldout": settings.get("heldout_segments"),
    }
    for split_name, expected_count in expected_counts.items():
        if expected_count is not None:
            actual_count = len(manifest["splits"][split_name]["examples"])
            if actual_count != int(expected_count):
                raise ValueError(
                    f"manifest {split_name} segment count {actual_count} does not match frozen {expected_count}"
                )
    return manifest


def _model_config_path(
    root: Path,
    asset_root: Path,
    freeze: Mapping[str, Any],
    checkpoint: Path | None,
    checkpoint_config: Path | None,
) -> Path:
    if checkpoint_config is not None:
        return checkpoint_config.resolve()
    if checkpoint is not None:
        return checkpoint.resolve().parent.parent / "hydra.yaml"
    return (
        asset_root
        / "checkpoints"
        / "outputs"
        / freeze["model"]["model_name"]
        / "hydra.yaml"
    ).resolve()


def _load_trajectory_datasets(
    root: Path,
    asset_root: Path,
    freeze: Mapping[str, Any],
    checkpoint: Path | None,
    checkpoint_config: Path | None,
) -> tuple[Any, Any]:
    """Load the same raw PushT train/valid datasets used by Stage-A's loader."""

    import hydra
    from omegaconf import OmegaConf

    source = root / "source"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    cfg = OmegaConf.load(_model_config_path(root, asset_root, freeze, checkpoint, checkpoint_config))
    _, trajectories = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    try:
        return trajectories["train"], trajectories["valid"]
    except KeyError as exc:
        raise ValueError("official PushT loader did not return train/valid trajectory datasets") from exc


class _RawEpisodeCache:
    """Small bounded uint8 cache; transforms and GPU encoding stay per batch."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("cache_episodes must be positive")
        self.limit = limit
        self.values: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def get(self, split: str, dset: Any, episode_id: int) -> Any:
        import torch
        from decord import VideoReader

        key = (split, int(episode_id))
        if key in self.values:
            value = self.values.pop(key)
            self.values[key] = value
            return value
        length = int(dset.get_seq_length(episode_id))
        video_path = dset.data_path / "obses" / f"episode_{episode_id:03d}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"manifest observation video is missing: {video_path}")
        reader = VideoReader(str(video_path), num_threads=1)
        frames = reader.get_batch(list(range(length)))
        value = frames.detach().cpu().to(dtype=torch.uint8)
        self.values[key] = value
        while len(self.values) > self.limit:
            self.values.popitem(last=False)
        return value


def _segment_batch(
    dset: Any,
    split: str,
    examples: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    frame_skip: int,
    cache: _RawEpisodeCache,
    model: Any,
    device: Any,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Encode one manifest-selected batch and return context/actions/real targets."""

    import torch

    visual_batches = []
    proprio_batches = []
    action_batches = []
    for index in selected:
        example = examples[int(index)]
        episode_id = int(example["episode_id"])
        start = int(example["start_step"])
        frame_ids = [start + offset * frame_skip for offset in range(HORIZON + 1)]
        raw = cache.get(split, dset, episode_id)
        if frame_ids[-1] >= raw.shape[0]:
            raise IndexError(f"manifest frame index exceeds episode {episode_id} length")
        frames = raw.index_select(0, torch.tensor(frame_ids, dtype=torch.long))
        visual = frames.float().permute(0, 3, 1, 2) / 255.0
        if dset.transform is not None:
            visual = dset.transform(visual)
        proprio = dset.proprios[episode_id, frame_ids].float()
        primitive = dset.actions[episode_id, start : start + HORIZON * frame_skip].float()
        if primitive.shape[0] != HORIZON * frame_skip:
            raise ValueError("manifest action segment has an unexpected length")
        actions = primitive.reshape(HORIZON, frame_skip, -1).reshape(HORIZON, -1)
        visual_batches.append(visual)
        proprio_batches.append(proprio)
        action_batches.append(actions)

    obs = {
        "visual": torch.stack(visual_batches).to(device),
        "proprio": torch.stack(proprio_batches).to(device),
    }
    actions = torch.stack(action_batches).to(device)
    with torch.no_grad():
        encoded = model.encode_obs(obs)
    context = {key: value[:, :1].detach() for key, value in encoded.items()}
    grounded = {key: value[:, 1 : HORIZON + 1].detach() for key, value in encoded.items()}
    return context, actions, grounded


def _preencode_manifest(
    dset: Any,
    split: str,
    examples: Sequence[Mapping[str, Any]],
    frame_skip: int,
    chunk_size: int,
    cache: _RawEpisodeCache,
    model: Any,
    device: Any,
) -> dict[str, Any]:
    """Encode every frozen manifest segment once and retain native latents on CPU.

    The source video is decoded only during this preparation pass.  Training
    and real-future evaluation index this cache, so the 500-step loop never
    repeats video I/O or the DINO encoder.
    """

    import torch

    if chunk_size < 1:
        raise ValueError("pre-encode chunk_size must be positive")
    context_chunks: dict[str, list[Any]] = {"visual": [], "proprio": []}
    grounded_chunks: dict[str, list[Any]] = {"visual": [], "proprio": []}
    action_chunks: list[Any] = []
    for start in range(0, len(examples), chunk_size):
        stop = min(start + chunk_size, len(examples))
        selected = list(range(start, stop))
        context, actions, grounded = _segment_batch(
            dset,
            split,
            examples,
            selected,
            frame_skip,
            cache,
            model,
            device,
        )
        for key in context_chunks:
            context_chunks[key].append(context[key].detach().cpu())
            grounded_chunks[key].append(grounded[key].detach().cpu())
        action_chunks.append(actions.detach().cpu())
    return {
        "context": {key: torch.cat(values, dim=0) for key, values in context_chunks.items()},
        "actions": torch.cat(action_chunks, dim=0),
        "grounded": {key: torch.cat(values, dim=0) for key, values in grounded_chunks.items()},
        "count": len(examples),
        "split": split,
    }


def _batch_from_cache(
    encoded_cache: Mapping[str, Any],
    selected: Sequence[int],
    device: Any,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Materialize one cached manifest batch on the model device."""

    import torch

    indices = torch.as_tensor(list(selected), dtype=torch.long)
    context = {
        key: value.index_select(0, indices).to(device, non_blocking=True)
        for key, value in encoded_cache["context"].items()
    }
    actions = encoded_cache["actions"].index_select(0, indices).to(device, non_blocking=True)
    grounded = {
        key: value.index_select(0, indices).to(device, non_blocking=True)
        for key, value in encoded_cache["grounded"].items()
    }
    return context, actions, grounded


def _latent_mse(prediction: Mapping[str, Any], target: Mapping[str, Any]) -> Any:
    import torch.nn.functional as F

    return 0.5 * (
        F.mse_loss(prediction["visual"], target["visual"])
        + F.mse_loss(prediction["proprio"], target["proprio"])
    )


def _evaluate_real_segments(
    students: Mapping[str, Any],
    model: Any,
    encoded_cache: Mapping[str, Any],
    batch_size: int,
    batches: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    generator = random.Random(seed)
    metrics: dict[str, list[dict[str, Any]]] = {
        "teacher_rollout": [],
        **{name: [] for name in students},
    }
    count = int(encoded_cache["count"])
    for batch_index in range(batches):
        selected = [generator.randrange(count) for _ in range(batch_size)]
        context, actions, grounded = _batch_from_cache(encoded_cache, selected, device)
        with torch.no_grad():
            teacher_prediction = _teacher_targets(model, context, actions)
            predictions = {
                "teacher_rollout": teacher_prediction,
                **{name: student(context, actions) for name, student in students.items()},
            }
        for name, prediction in predictions.items():
            case = _metrics_by_horizon(prediction, grounded)
            case["batch"] = batch_index
            metrics[name].append(case)

    result: dict[str, Any] = {}
    for name, cases in metrics.items():
        result[name] = {
            "batches": cases,
            "per_horizon": {
                key: [
                    _mean([case[key][horizon] for case in cases])
                    for horizon in range(HORIZON)
                ]
                for key in ("relative_mse", "cosine")
            },
        }
    return result


def _evaluate_planner_pair(
    students: Mapping[str, Any],
    model: Any,
    anchors: Mapping[str, Any],
    goals: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    eval_seeds: Sequence[int],
    batch_size: int,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    cases: dict[str, list[dict[str, Any]]] = {name: [] for name in students}
    paired: list[dict[str, Any]] = []
    for eval_seed in eval_seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(eval_seed))
        for anchor_index in range(ANCHOR_COUNT):
            context = _repeat_anchor(
                {key: value[anchor_index : anchor_index + 1] for key, value in anchors.items()},
                batch_size,
            )
            goal = _repeat_anchor(
                {key: value[anchor_index : anchor_index + 1, :1] for key, value in goals.items()},
                batch_size,
            )
            actions = _sample_actions(generator, batch_size, action_dim, device)
            target = _teacher_targets(model, context, actions)
            with torch.no_grad():
                predictions = {name: student(context, actions) for name, student in students.items()}
            rank_values = {}
            for name, prediction in predictions.items():
                teacher_cost = _objective_cost(objective_fn, target, goal)
                student_cost = _objective_cost(objective_fn, prediction, goal)
                ranking = {
                    "spearman": _spearman(teacher_cost, student_cost),
                    "top30_overlap": _topk_overlap(teacher_cost, student_cost),
                }
                item = _metrics_by_horizon(prediction, target)
                item.update({"seed": int(eval_seed), "anchor": anchor_index, "ranking": ranking})
                cases[name].append(item)
                rank_values[name] = ranking
            item = {"seed": int(eval_seed), "anchor": anchor_index}
            for left, right, label in (
                ("wide_grounded", "wide_teacher", "wide_grounded_minus_wide_teacher"),
                ("wide_teacher", "narrow_teacher", "wide_teacher_minus_narrow_teacher"),
            ):
                if left in rank_values and right in rank_values:
                    item[f"spearman_delta_{label}"] = (
                        rank_values[left]["spearman"] - rank_values[right]["spearman"]
                    )
                    item[f"top30_delta_{label}"] = (
                        rank_values[left]["top30_overlap"] - rank_values[right]["top30_overlap"]
                    )
            paired.append(item)

    result: dict[str, Any] = {}
    for name, arm_cases in cases.items():
        ranking = [item["ranking"] for item in arm_cases]
        result[name] = {
            "per_case": arm_cases,
            "per_horizon": {
                key: [_mean([case[key][horizon] for case in arm_cases]) for horizon in range(HORIZON)]
                for key in ("relative_mse", "cosine")
            },
            "terminal_ranking": {
                "spearman_mean": _mean([item["spearman"] for item in ranking]),
                "spearman_median": _median([item["spearman"] for item in ranking]),
                "spearman_minimum": min(item["spearman"] for item in ranking),
                "top30_overlap_mean": _mean([item["top30_overlap"] for item in ranking]),
                "top30_overlap_median": _median([item["top30_overlap"] for item in ranking]),
                "top30_overlap_minimum": min(item["top30_overlap"] for item in ranking),
            },
        }
    return result, paired


def _gate_summary(
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    gates = protocol.get("gates", {})
    if not isinstance(gates, Mapping):
        gates = {}
    capacity = gates.get("capacity", {})
    fidelity = gates.get("fidelity", gates.get("absolute_fidelity", {}))
    latency = gates.get("latency", {})
    noninferiority = gates.get("grounded_noninferiority", gates.get("latent_noninferiority", {}))
    coverage = gates.get("coverage", gates.get("coverage_effect", {}))
    grounding = gates.get("grounding_effect", {})
    if not all(
        isinstance(item, Mapping)
        for item in (capacity, fidelity, latency, noninferiority, coverage, grounding)
    ):
        raise ValueError("grounded-prefix protocol gates must be objects")

    def threshold(mapping: Mapping[str, Any], default: float, *keys: str) -> float:
        for key in keys:
            if key in mapping:
                return float(mapping[key])
        return default

    def integer_threshold(mapping: Mapping[str, Any], default: int, *keys: str) -> int:
        for key in keys:
            if key in mapping:
                return int(mapping[key])
        return default

    finite = bool(result["finite_outputs_and_training"])
    causal = all(bool(item["passed"]) for item in result["causality"].values())
    train = result["train"]
    max_ratio = float(capacity.get("latent_last10_to_first_ratio_max", 0.8))
    capacity_pass = finite and causal and all(
        float(train[name]["latent_last10_to_first_ratio"]) <= max_ratio for name in train
    )
    narrow_rank = result["planner_evaluation"]["narrow_teacher"]["terminal_ranking"]
    teacher_rank = result["planner_evaluation"]["wide_teacher"]["terminal_ranking"]
    grounded_rank = result["planner_evaluation"]["wide_grounded"]["terminal_ranking"]
    grounded_eval = result["real_future_evaluation"]
    frozen_teacher = _mean(grounded_eval["teacher_rollout"]["per_horizon"]["relative_mse"])
    narrow_real = _mean(grounded_eval["narrow_teacher"]["per_horizon"]["relative_mse"])
    real_teacher = _mean(grounded_eval["wide_teacher"]["per_horizon"]["relative_mse"])
    real_grounded = _mean(grounded_eval["wide_grounded"]["per_horizon"]["relative_mse"])
    grounded_ratio = real_grounded / max(real_teacher, 1e-12)
    coverage_ratio = real_teacher / max(narrow_real, 1e-12)
    delta_spearman = [
        item["spearman_delta_wide_grounded_minus_wide_teacher"]
        for item in result["paired_deltas"]["per_block"]
    ]
    delta_top30 = [
        item["top30_delta_wide_grounded_minus_wide_teacher"]
        for item in result["paired_deltas"]["per_block"]
    ]
    median_spearman_delta = _median(delta_spearman)
    median_top30_delta = _median(delta_top30)
    positive_spearman = sum(value > 0 for value in delta_spearman)
    positive_top30 = sum(value > 0 for value in delta_top30)
    coverage_spearman = [
        item["spearman_delta_wide_teacher_minus_narrow_teacher"]
        for item in result["paired_deltas"]["per_block"]
    ]
    coverage_top30 = [
        item["top30_delta_wide_teacher_minus_narrow_teacher"]
        for item in result["paired_deltas"]["per_block"]
    ]
    coverage_positive_spearman = sum(value > 0 for value in coverage_spearman)
    coverage_positive_top30 = sum(value > 0 for value in coverage_top30)
    ranking_median_min = threshold(
        fidelity,
        0.0,
        "grounded_dense_spearman_median_min",
        "rankdistill_objective_spearman_min",
        "objective_spearman_min",
    )
    ranking_block_min = threshold(
        fidelity,
        0.0,
        "grounded_dense_spearman_minimum_min",
        "minimum_block_spearman_min",
    )
    overlap_median_min = threshold(
        fidelity,
        0.0,
        "grounded_dense_top30_median_min",
        "rankdistill_top30_overlap_min",
        "top30_overlap_min",
    )
    overlap_block_min = threshold(
        fidelity,
        0.0,
        "grounded_dense_top30_minimum_min",
        "minimum_block_top30_overlap_min",
    )
    delta_spearman_min = threshold(
        fidelity,
        0.05,
        "median_spearman_delta_min",
    )
    delta_top30_min = threshold(
        fidelity,
        0.10,
        "median_top30_overlap_delta_min",
    )
    positive_spearman_min = integer_threshold(
        fidelity, 3, "positive_spearman_blocks_min"
    )
    positive_top30_min = integer_threshold(
        fidelity, 3, "positive_top30_blocks_min"
    )
    real_ratio_max = threshold(
        noninferiority,
        1.25,
        "grounded_to_teacher_dense_mean_relative_mse_ratio_max",
        "grounded_to_wide_teacher_real_future_relative_mse_ratio_max",
    )
    coverage_ratio_max = threshold(
        coverage,
        1.0,
        "wide_teacher_to_narrow_teacher_real_future_relative_mse_ratio_max",
        "wide_to_narrow_real_future_ratio_max",
    )
    coverage_delta_spearman_min = threshold(
        coverage, 0.05, "median_spearman_delta_min"
    )
    coverage_delta_top30_min = threshold(
        coverage, 0.10, "median_top30_overlap_delta_min"
    )
    coverage_positive_spearman_min = integer_threshold(
        coverage, 3, "positive_spearman_blocks_min"
    )
    coverage_positive_top30_min = integer_threshold(
        coverage, 3, "positive_top30_blocks_min"
    )
    latency_min = threshold(
        latency, 0.20, "grounded_dense_predictor_reduction_vs_teacher_min"
    )
    reduction = float(result["latency"]["wide_grounded"]["median_reduction"])
    fidelity_pass = (
        grounded_rank["spearman_median"] >= ranking_median_min
        and grounded_rank["spearman_minimum"] >= ranking_block_min
        and grounded_rank["top30_overlap_median"] >= overlap_median_min
        and grounded_rank["top30_overlap_minimum"] >= overlap_block_min
    )
    grounding_ratio_max = threshold(
        grounding,
        real_ratio_max,
        "grounded_to_wide_teacher_real_future_relative_mse_ratio_max",
    )
    grounding_positive_spearman_min = integer_threshold(
        grounding, positive_spearman_min, "positive_spearman_blocks_min"
    )
    grounding_positive_top30_min = integer_threshold(
        grounding, positive_top30_min, "positive_top30_blocks_min"
    )
    grounding_pass = (
        grounded_ratio <= grounding_ratio_max
        and positive_spearman >= grounding_positive_spearman_min
        and positive_top30 >= grounding_positive_top30_min
        and median_spearman_delta >= threshold(grounding, 0.05, "median_spearman_delta_min")
        and median_top30_delta >= threshold(grounding, 0.10, "median_top30_overlap_delta_min")
    )
    coverage_pass = (
        _median(coverage_spearman) >= coverage_delta_spearman_min
        and _median(coverage_top30) >= coverage_delta_top30_min
        and coverage_positive_spearman >= coverage_positive_spearman_min
        and coverage_positive_top30 >= coverage_positive_top30_min
        and coverage_ratio <= coverage_ratio_max
    )
    latency_pass = reduction >= latency_min
    return {
        "capacity": {
            "status": "PASS" if capacity_pass else "FAIL",
            "finite_outputs_and_training": finite,
            "causality": causal,
            "maximum_loss_ratio": max_ratio,
        },
        "fidelity": {
            "status": "PASS" if fidelity_pass else "FAIL",
            "grounded_dense_spearman_median": grounded_rank["spearman_median"],
            "grounded_dense_spearman_median_minimum": ranking_median_min,
            "grounded_dense_spearman_block_minimum": grounded_rank["spearman_minimum"],
            "grounded_dense_spearman_block_minimum_threshold": ranking_block_min,
            "grounded_dense_top30_median": grounded_rank["top30_overlap_median"],
            "grounded_dense_top30_median_minimum": overlap_median_min,
            "grounded_dense_top30_block_minimum": grounded_rank["top30_overlap_minimum"],
            "grounded_dense_top30_block_minimum_threshold": overlap_block_min,
            "wide_teacher_spearman_median": teacher_rank["spearman_median"],
            "wide_teacher_top30_median": teacher_rank["top30_overlap_median"],
            "narrow_teacher_spearman_median": narrow_rank["spearman_median"],
            "narrow_teacher_top30_median": narrow_rank["top30_overlap_median"],
            "median_spearman_delta": median_spearman_delta,
            "median_spearman_delta_minimum": delta_spearman_min,
            "median_top30_delta": median_top30_delta,
            "median_top30_delta_minimum": delta_top30_min,
            "positive_spearman_blocks": positive_spearman,
            "positive_spearman_blocks_minimum": positive_spearman_min,
            "positive_top30_blocks": positive_top30,
            "positive_top30_blocks_minimum": positive_top30_min,
        },
        "grounded_future": {
            "status": "PASS" if grounding_pass else "FAIL",
            "frozen_teacher_rollout_vs_ground_truth_relative_mse": frozen_teacher,
            "grounded_to_wide_teacher_real_future_relative_mse_ratio": grounded_ratio,
            "maximum_ratio": grounding_ratio_max,
            "positive_spearman_blocks": positive_spearman,
            "positive_spearman_blocks_minimum": grounding_positive_spearman_min,
            "positive_top30_blocks": positive_top30,
            "positive_top30_blocks_minimum": grounding_positive_top30_min,
            "median_spearman_delta": median_spearman_delta,
            "median_spearman_delta_minimum": threshold(grounding, 0.05, "median_spearman_delta_min"),
            "median_top30_delta": median_top30_delta,
            "median_top30_delta_minimum": threshold(grounding, 0.10, "median_top30_overlap_delta_min"),
        },
        "coverage": {
            "status": "PASS" if coverage_pass else "FAIL",
            "wide_teacher_to_narrow_teacher_real_future_relative_mse_ratio": coverage_ratio,
            "maximum_ratio": coverage_ratio_max,
            "median_spearman_delta_wide_teacher_minus_narrow_teacher": _median(coverage_spearman),
            "median_spearman_delta_minimum": coverage_delta_spearman_min,
            "median_top30_delta_wide_teacher_minus_narrow_teacher": _median(coverage_top30),
            "median_top30_delta_minimum": coverage_delta_top30_min,
            "positive_spearman_blocks": coverage_positive_spearman,
            "positive_spearman_blocks_minimum": coverage_positive_spearman_min,
            "positive_top30_blocks": coverage_positive_top30,
            "positive_top30_blocks_minimum": coverage_positive_top30_min,
        },
        "predictor_latency": {
            "status": "PASS" if latency_pass else "FAIL",
            "grounded_dense_reduction": reduction,
            "minimum_reduction": latency_min,
        },
        # Coverage is reported as a diagnostic contrast.  The scientific GO
        # requires WIDE-GT absolute fidelity and its grounding effect, but a
        # coverage effect is not a prerequisite for the grounded hypothesis.
        "overall": "PASS" if capacity_pass and fidelity_pass and grounding_pass and latency_pass else "FAIL",
    }


def main() -> int:
    args = _args()
    _require_compute_node()
    root = args.root.resolve()
    asset_root = (args.asset_root or root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    protocol = _load_json(args.protocol.resolve())
    settings = _settings(protocol)
    manifest = _load_manifest(args.manifest.resolve(), settings)
    _require_assets(asset_root, freeze)
    if settings["batch"] < 1 or settings["steps"] < 1:
        raise ValueError("frozen training steps and batch must be positive")
    _gpu_snapshot(output, "start")

    import torch

    _set_seed(settings["train_seed"])
    model, workspace, anchors, goals, objective_fn, action_dim, device = _load_official(
        root,
        asset_root,
        freeze,
        output,
        settings["train_seed"],
        config_path=args.config.resolve() if args.config else None,
        checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None,
        checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None,
        data_root=args.data_root.resolve() if args.data_root else None,
    )
    del workspace
    train_dset, heldout_dset = _load_trajectory_datasets(
        root, asset_root, freeze, args.checkpoint, args.checkpoint_config
    )
    train_examples = manifest["splits"]["train"]["examples"]
    heldout_examples = manifest["splits"]["heldout"]["examples"]
    visual_dim = int(anchors["visual"].shape[-1])
    proprio_dim = int(anchors["proprio"].shape[-1])
    template = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, settings["hidden_dim"]).to(device)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    students = {
        name: NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, settings["hidden_dim"]).to(device)
        for name in ("narrow_teacher", "wide_teacher", "wide_grounded")
    }
    for student in students.values():
        student.load_state_dict(initial_state)
    optimizers = {
        name: torch.optim.AdamW(student.parameters(), lr=settings["lr"])
        for name, student in students.items()
    }
    train_rng = random.Random(settings["train_seed"])
    cache = _RawEpisodeCache(settings["cache_episodes"])
    train_encoded = _preencode_manifest(
        train_dset,
        "train",
        train_examples,
        settings["frame_skip"],
        settings["preencode_chunk"],
        cache,
        model,
        device,
    )
    heldout_encoded = _preencode_manifest(
        heldout_dset,
        "heldout",
        heldout_examples,
        settings["frame_skip"],
        settings["preencode_chunk"],
        cache,
        model,
        device,
    )
    train_episode_ids = []
    for example in train_examples:
        episode_id = int(example["episode_id"])
        if episode_id not in train_episode_ids:
            train_episode_ids.append(episode_id)
        if len(train_episode_ids) == 2:
            break
    narrow_indices = [
        index
        for index, example in enumerate(train_examples)
        if int(example["episode_id"]) in train_episode_ids
    ]
    if len(train_episode_ids) < 2 or not narrow_indices:
        raise ValueError("manifest train split must contain at least two episodes for NARROW-T")
    losses: dict[str, list[dict[str, float]]] = {name: [] for name in students}
    telemetry_period = max(1, settings["steps"] // 10)

    for step in range(settings["steps"]):
        wide_selected = [train_rng.randrange(len(train_examples)) for _ in range(settings["batch"])]
        narrow_selected = [train_rng.randrange(len(narrow_indices)) for _ in range(settings["batch"])]
        narrow_selected = [narrow_indices[index] for index in narrow_selected]
        wide_context, wide_actions, grounded_target = _batch_from_cache(
            train_encoded, wide_selected, device
        )
        narrow_context, narrow_actions, _ = _batch_from_cache(
            train_encoded, narrow_selected, device
        )
        with torch.no_grad():
            wide_teacher_target = _teacher_targets(model, wide_context, wide_actions)
            narrow_teacher_target = _teacher_targets(model, narrow_context, narrow_actions)
        batches = {
            "narrow_teacher": (narrow_context, narrow_actions, narrow_teacher_target),
            "wide_teacher": (wide_context, wide_actions, wide_teacher_target),
            "wide_grounded": (wide_context, wide_actions, grounded_target),
        }
        order = (
            ("narrow_teacher", "wide_teacher", "wide_grounded")
            if step % 2 == 0
            else ("wide_grounded", "wide_teacher", "narrow_teacher")
        )
        for name in order:
            context, actions, target = batches[name]
            prediction = students[name](context, actions)
            latent_loss = _latent_mse(prediction, target)
            optimizers[name].zero_grad(set_to_none=True)
            latent_loss.backward()
            optimizers[name].step()
            losses[name].append(
                {
                    "step": step + 1,
                    "manifest_batch_seed": settings["train_seed"],
                    "latent_mse": float(latent_loss.detach().cpu()),
                }
            )
        if step == 0 or (step + 1) % telemetry_period == 0:
            _gpu_snapshot(output, f"train_step_{step + 1}")

    planner_evaluation, paired_deltas = _evaluate_planner_pair(
        students,
        model,
        anchors,
        goals,
        objective_fn,
        action_dim,
        settings["heldout_seeds"],
        settings["eval_batch"],
        device,
    )
    real_future_evaluation = _evaluate_real_segments(
        students,
        model,
        heldout_encoded,
        min(settings["batch"], len(heldout_examples)),
        settings["val_batches"],
        settings["train_seed"] + 1,
        device,
    )
    causality = {
        name: _leakage_test(
            student,
            anchors,
            action_dim,
            settings["heldout_seeds"][0] + 9000,
            device,
            settings["future_action_tolerance"],
        )
        for name, student in students.items()
    }
    latency = {
        name: _latency(
            student,
            model,
            anchors,
            action_dim,
            settings["timing_seed"],
            settings["timing_batch"],
            settings["warmup"],
            settings["repeats"],
            device,
        )
        for name, student in students.items()
    }
    _gpu_snapshot(output, "complete")

    def arm_train(name: str) -> dict[str, Any]:
        values = [item["latent_mse"] for item in losses[name]]
        last10 = values[-10:]
        return {
            "steps": settings["steps"],
            "batch": settings["batch"],
            "loss_first": values[0],
            "loss_last": values[-1],
            "latent_mse_median_last_10": _median(last10),
            "latent_last10_to_first_ratio": _median(last10) / max(values[0], 1e-12),
            "per_step": losses[name],
        }

    paired_spearman = [
        item["spearman_delta_wide_grounded_minus_wide_teacher"] for item in paired_deltas
    ]
    paired_top30 = [item["top30_delta_wide_grounded_minus_wide_teacher"] for item in paired_deltas]
    wide_vs_narrow_spearman = [
        item["spearman_delta_wide_teacher_minus_narrow_teacher"] for item in paired_deltas
    ]
    wide_vs_narrow_top30 = [
        item["top30_delta_wide_teacher_minus_narrow_teacher"] for item in paired_deltas
    ]
    result: dict[str, Any] = {
        "train": {name: arm_train(name) for name in students},
        "planner_evaluation": planner_evaluation,
        "real_future_evaluation": real_future_evaluation,
        "paired_deltas": {
            "spearman_mean_grounded_minus_teacher_dense": _mean(paired_spearman),
            "spearman_median_grounded_minus_teacher_dense": _median(paired_spearman),
            "top30_mean_grounded_minus_teacher_dense": _mean(paired_top30),
            "top30_median_grounded_minus_teacher_dense": _median(paired_top30),
            "spearman_median_wide_teacher_minus_narrow_teacher": _median(wide_vs_narrow_spearman),
            "top30_median_wide_teacher_minus_narrow_teacher": _median(wide_vs_narrow_top30),
            "per_block": paired_deltas,
        },
        "causality": causality,
        "latency": latency,
    }
    result["finite_outputs_and_training"] = _all_finite(result)
    gates = _gate_summary(protocol, result)
    checkpoint_common = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-grounded-prefix-checkpoint",
        "horizon": HORIZON,
        "native_output_fields": ["visual", "proprio"],
        "action_dim": action_dim,
        "visual_dim": visual_dim,
        "proprio_dim": proprio_dim,
        "hidden_dim": settings["hidden_dim"],
        "train_seed": settings["train_seed"],
        "manifest": str(args.manifest.resolve()),
    }
    for name, student in students.items():
        torch.save({**checkpoint_common, "arm": name, "state_dict": student.state_dict()}, output / f"{name}_student.pt")

    summary = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-grounded-prefix-summary",
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "freeze": str(args.freeze.resolve()),
        "manifest": str(args.manifest.resolve()),
        "source": {
            "reproduction_root": str(root),
            "official_loader": "run_dino_pusht_stage_a._load_official plus the same env.dataset raw train/valid loader",
            "grounded_target": "frozen model.encode_obs on manifest target frames",
            "teacher_target": "frozen model rollout_from_encoded_obs on manifest action prefixes",
            "preencoding": "all train/heldout manifest segments encoded once in chunks and retained as CPU native latents",
        },
        "contracts": {
            "arm_labels": {
                "NARROW-T": "narrow_teacher",
                "WIDE-T": "wide_teacher",
                "WIDE-GT": "wide_grounded",
            },
            "same_initial_state_dict": True,
            "wide_arms_same_manifest_segments": True,
            "wide_arms_same_action_prefixes": True,
            "narrow_teacher_data": "real manifest segments restricted to the first two selected train episodes",
            "alternating_update_order": "narrow_teacher, wide_teacher, wide_grounded on even steps; reverse on odd steps",
            "goal_in_student_input": False,
            "horizon": HORIZON,
            "frame_skip": settings["frame_skip"],
            "train_examples": len(train_examples),
            "narrow_train_examples": len(narrow_indices),
            "narrow_train_episode_ids": train_episode_ids,
            "heldout_examples": len(heldout_examples),
            "frozen_train_segments": settings["train_segments"],
            "frozen_heldout_segments": settings["heldout_segments"],
            "preencode_chunk": settings["preencode_chunk"],
        },
        "train": result["train"],
        "planner_evaluation": planner_evaluation,
        "real_future_evaluation": real_future_evaluation,
        "teacher_rollout_vs_ground_truth": real_future_evaluation["teacher_rollout"],
        "paired_deltas": result["paired_deltas"],
        "causality": causality,
        "latency": latency,
        "gates": gates,
        "timing_boundary": "predictor-level cached native observation latent plus normalized action prefix; excludes observation encoding, CEM, environment interaction and closed-loop execution",
        "unverified": [
            "No closed-loop CEM or environment execution is included.",
            "This DINO-only experiment does not establish LeWM transfer or all-JEPA universality.",
            "Manifest validation episodes are finite held-out segments, not population-level statistical samples.",
        ],
    }
    summary_path = (args.summary.resolve() if args.summary else output / "grounded_prefix_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2, default=_json_default)
    summary_path.write_text(summary_text, encoding="utf-8")
    default_summary = output / "grounded_prefix_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(summary_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
