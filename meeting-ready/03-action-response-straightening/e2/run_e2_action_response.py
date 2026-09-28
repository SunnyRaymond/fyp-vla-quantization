#!/usr/bin/env python3
"""Matched, predictor-only E2 fine-tuning for the pinned LeWM PushT checkpoint.

The runner only reads staged assets inside an approved PBS GPU allocation.
``--mode status`` is dependency-free and performs no asset I/O.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import socket
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "lewm-pusht.action-response-straightening.e2"
SOURCE_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
LATENT_DIM = 192
HISTORY_SIZE = 3
PACKED_ACTION_DIM = 10
FRAMESKIP = 5
ROLLOUT_HORIZON = 5
ACTION_LOW = -1.0
ACTION_HIGH = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("status", "run"), default="status")
    parser.add_argument("--stablewm-home", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--adapter-script", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-contexts", type=int, default=16)
    parser.add_argument("--heldout-contexts", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--heldout-episodes", type=int, default=2)
    parser.add_argument("--batch-contexts", type=int, default=4)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--branch-weight", type=float, default=1.0)
    parser.add_argument("--lambda-curvature", type=float, default=1e-4)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=26092401)
    parser.add_argument("--schedule-seed", type=int, default=26092402)
    parser.add_argument("--encode-batch-contexts", type=int, default=2)
    return parser.parse_args()


def require_compute_allocation() -> None:
    """Fail before model, HDF5, or output work unless this is a real GPU job."""
    job_id = os.environ.get("PBS_JOBID", "").strip()
    host = socket.gethostname().lower()
    if not job_id:
        raise RuntimeError("run mode requires a non-empty PBS_JOBID")
    if any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("run mode requires PBS-provided CUDA_VISIBLE_DEVICES")


def pack_actions(raw_actions: Any) -> Any:
    import numpy as np

    values = np.asarray(raw_actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] % FRAMESKIP:
        raise ValueError(f"raw actions must be [N,2], N divisible by {FRAMESKIP}; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("sampled recorded actions contain non-finite values")
    if bool((values < ACTION_LOW).any() or (values > ACTION_HIGH).any()):
        raise ValueError("sampled recorded actions are outside the declared normalized [-1,1] box")
    return values.reshape(-1, PACKED_ACTION_DIM)


def symmetric_triplet(
    center: Any,
    low: float,
    high: float,
    delta: float,
    coordinate: int,
) -> tuple[Any, Any, Any, Any]:
    """Make exactly symmetric center/plus/minus plans along one legal coordinate."""
    import torch

    center = torch.as_tensor(center, dtype=torch.float32)
    flat = center.reshape(-1)
    if delta <= 0 or coordinate < 0 or coordinate >= flat.numel():
        raise ValueError("delta and coordinate are invalid")
    if bool(((flat < low) | (flat > high)).any()):
        raise ValueError("center action is outside the declared action box")
    margin = min(float(flat[coordinate] - low), float(high - flat[coordinate]))
    if margin + 1e-12 < delta:
        raise ValueError("requested symmetric perturbation does not fit inside the action box")
    direction = torch.zeros_like(flat)
    direction[coordinate] = 1.0
    plus = flat + delta * direction
    minus = flat - delta * direction
    # This check catches any accidental future clipping or asymmetric change.
    if not torch.allclose((plus + minus) * 0.5, flat, atol=1e-7, rtol=0.0):
        raise RuntimeError("finite-difference action triplet is not exactly symmetric")
    return flat, plus, minus, direction


def curvature_penalty(plus: Any, center: Any, minus: Any, delta: Any) -> Any:
    """Mean squared terminal-latent second derivative, normalized by latent D."""
    second = plus - 2.0 * center + minus
    radius = delta.reshape(-1).to(dtype=second.dtype, device=second.device)
    if bool((radius <= 0).any()):
        raise ValueError("curvature radii must be positive")
    return (second.square().mean(dim=-1) / radius.pow(4)).mean()


def taylor_residuals(
    center: Any,
    plus_delta: Any,
    minus_delta: Any,
    plus_radius: Any,
    minus_radius: Any,
    delta: float,
    radius: float,
) -> tuple[Any, Any]:
    """RMSE and relative RMSE of held-out first-order Taylor predictions."""
    derivative = (plus_delta - minus_delta) / (2.0 * delta)
    import torch

    plus_error = plus_radius - (center + radius * derivative)
    minus_error = minus_radius - (center - radius * derivative)
    residuals = torch.stack((plus_error, minus_error), dim=1)
    rmse = residuals.square().mean(dim=(1, 2)).sqrt()
    scale = torch.stack((plus_radius - center, minus_radius - center), dim=1)
    relative = rmse / (scale.square().mean(dim=(1, 2)).sqrt() + 1e-8)
    return rmse, relative


def split_episode_ids(
    episode_ids: Sequence[int], train_count: int, heldout_count: int, seed: int
) -> tuple[list[int], list[int]]:
    ids = list(dict.fromkeys(int(value) for value in episode_ids))
    if train_count < 1 or heldout_count < 1 or len(ids) < train_count + heldout_count:
        raise ValueError("not enough unique parent episodes for a disjoint train/heldout split")
    random.Random(seed).shuffle(ids)
    return ids[:train_count], ids[train_count : train_count + heldout_count]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    import hdf5plugin  # noqa: F401 - register the staged PushT compression filter
    import h5py
    import numpy as np

    with h5py.File(args.dataset, "r") as h5:
        required = {"pixels", "action", "episode_idx", "step_idx", "ep_len", "ep_offset"}
        missing = sorted(required - set(h5.keys()))
        if missing:
            raise ValueError(f"PushT HDF5 lacks required datasets: {missing}")
        lengths = [int(x) for x in h5["ep_len"][:]]
        valid = [episode for episode, length in enumerate(lengths) if length >= 40]
        train_ids, heldout_ids = split_episode_ids(
            valid, args.train_episodes, args.heldout_episodes, args.split_seed
        )
        picker = random.Random(args.split_seed + 1)

        def sample_rows(split: str, ids: list[int], count: int) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            used: dict[int, set[int]] = {episode: set() for episode in ids}
            for ordinal in range(count):
                episode = ids[ordinal % len(ids)]
                offset = int(h5["ep_offset"][episode])
                length = lengths[episode]
                found = False
                for _ in range(max(100, length)):
                    start = picker.randrange(0, length - 40 + 1)
                    if start in used[episode]:
                        continue
                    raw = np.asarray(h5["action"][offset + start + 15 : offset + start + 40], dtype=np.float32)
                    observed_episodes = np.asarray(h5["episode_idx"][offset + start : offset + start + 40])
                    observed_steps = np.asarray(h5["step_idx"][offset + start : offset + start + 40])
                    if not np.all(observed_episodes == episode) or not np.array_equal(observed_steps, np.arange(start, start + 40)):
                        raise ValueError("sampled HDF5 episode_idx/step_idx do not match the parent-episode slice")
                    if raw.shape != (25, 2) or not np.isfinite(raw).all():
                        continue
                    if bool((raw < ACTION_LOW).any() or (raw > ACTION_HIGH).any()):
                        raise ValueError("sampled HDF5 actions are outside the official normalized [-1,1] box")
                    flat = raw.reshape(-1)
                    margin = np.minimum(flat - ACTION_LOW, ACTION_HIGH - flat)
                    eligible = np.flatnonzero(margin >= 2.0 * args.delta)
                    if not eligible.size:
                        continue
                    used[episode].add(start)
                    coordinate = int(picker.choice(eligible.tolist()))
                    sign = -1 if picker.random() < 0.5 else 1
                    rows.append(
                        {
                            "split": split,
                            "context_id": f"{split}_{ordinal:04d}",
                            "episode_id": episode,
                            "start_index": start,
                            "delta_coordinate": coordinate,
                        }
                    )
                    found = True
                    break
                if not found:
                    raise RuntimeError(
                        f"could not sample a legal symmetric action triplet for {split} context {ordinal}; "
                        "reduce delta or inspect the pinned action normalization"
                    )
            return rows

        train_rows = sample_rows("train", train_ids, args.train_contexts)
        heldout_rows = sample_rows("heldout", heldout_ids, args.heldout_contexts)
    if set(train_ids) & set(heldout_ids):
        raise RuntimeError("parent-episode split overlap")
    return {
        "schema": f"{SCHEMA}.manifest",
        "source_commit": SOURCE_COMMIT,
        "episode_disjoint": True,
        "split_seed": args.split_seed,
        "action_contract": {"bounds": [ACTION_LOW, ACTION_HIGH], "frameskip": FRAMESKIP, "packed_dim": PACKED_ACTION_DIM},
        "train_episode_ids": train_ids,
        "heldout_episode_ids": heldout_ids,
        "train": train_rows,
        "heldout": heldout_rows,
    }


def _read_context_arrays(dataset: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import hdf5plugin  # noqa: F401 - register the staged PushT compression filter
    import h5py
    import numpy as np

    output: list[dict[str, Any]] = []
    with h5py.File(dataset, "r") as h5:
        for row in manifest["train"] + manifest["heldout"]:
            episode = int(row["episode_id"])
            start = int(row["start_index"])
            offset = int(h5["ep_offset"][episode])
            frame_steps = [start + i * FRAMESKIP for i in range(4)]
            pixels = np.stack([h5["pixels"][offset + step] for step in frame_steps])
            transition = h5["action"][offset + start : offset + start + 15]
            branch = h5["action"][offset + start + 15 : offset + start + 40]
            transition_tokens = pack_actions(transition)
            branch_tokens = pack_actions(branch)
            center, plus, minus, direction = symmetric_triplet(
                branch_tokens,
                ACTION_LOW,
                ACTION_HIGH,
                manifest["action_contract"]["delta"],
                int(row["delta_coordinate"]),
            )
            plus = plus.reshape(ROLLOUT_HORIZON, PACKED_ACTION_DIM)
            minus = minus.reshape(ROLLOUT_HORIZON, PACKED_ACTION_DIM)
            center = center.reshape(ROLLOUT_HORIZON, PACKED_ACTION_DIM)
            center_np = center.numpy()
            plus_np = plus.numpy()
            minus_np = minus.numpy()
            delta = float(manifest["action_contract"]["delta"])
            eval_radius = 2.0 * delta
            base_flat = branch_tokens.reshape(-1)
            eval_plus = base_flat + eval_radius * direction.numpy()
            eval_minus = base_flat - eval_radius * direction.numpy()
            if bool((eval_plus < ACTION_LOW).any() or (eval_plus > ACTION_HIGH).any() or (eval_minus < ACTION_LOW).any() or (eval_minus > ACTION_HIGH).any()):
                raise RuntimeError("held-out Taylor evaluation left the action box")
            output.append(
                {
                    **row,
                    "pixels": pixels,
                    "transition_actions": transition_tokens,
                    "branch_prefix_actions": transition_tokens[1:3],
                    "triplet_plans": np.stack((center_np, plus_np, minus_np)),
                    "heldout_plans": np.stack(
                        (
                            center_np.reshape(-1),
                            base_flat + delta * direction.numpy(),
                            base_flat - delta * direction.numpy(),
                            eval_plus,
                            eval_minus,
                        )
                    ).reshape(5, ROLLOUT_HORIZON, PACKED_ACTION_DIM),
                    "delta": delta,
                }
            )
    return output


def _normalize_pixels(pixels: Any, device: Any) -> Any:
    import torch

    value = torch.as_tensor(pixels, dtype=torch.float32, device=device)
    if value.ndim != 5 or value.shape[-1] != 3:
        raise ValueError(f"sampled pixels must be [B,T,H,W,3], got {tuple(value.shape)}")
    value = value.permute(0, 1, 4, 2, 3).contiguous() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    return (value - mean) / std


def _encode_contexts(model: Any, rows: list[dict[str, Any]], device: Any, batch_size: int) -> None:
    import torch

    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pixels = _normalize_pixels([row["pixels"] for row in batch_rows], device)
        with torch.no_grad():
            encoded = model.encode({"pixels": pixels})["emb"]
        if tuple(encoded.shape[1:]) != (4, LATENT_DIM):
            raise RuntimeError(f"pinned encoder expected [B,4,192], got {tuple(encoded.shape)}")
        for index, latent in enumerate(encoded.detach().cpu()):
            batch_rows[index]["latent"] = latent
            batch_rows[index]["pixels_shape"] = list(batch_rows[index]["pixels"].shape)
            del batch_rows[index]["pixels"]


def imagined_rollout(model: Any, latent_history: Any, action_prefix: Any, plan: Any) -> Any:
    """Roll out five imagined latents from three recorded states and two actions."""
    import torch

    if latent_history.ndim != 3 or latent_history.shape[1:] != (HISTORY_SIZE, LATENT_DIM):
        raise ValueError("imagined branch must start from [B,3,192] recorded latent history")
    if action_prefix.shape[1:] != (HISTORY_SIZE - 1, PACKED_ACTION_DIM):
        raise ValueError("imagined branch requires the two recorded action tokens before its first candidate")
    if plan.ndim != 3 or plan.shape[1:] != (ROLLOUT_HORIZON, PACKED_ACTION_DIM):
        raise ValueError("imagined plan must be [B,5,10]")
    history = latent_history
    outputs = []
    for step in range(ROLLOUT_HORIZON):
        action_window = torch.cat((action_prefix, plan[:, : step + 1]), dim=1)[:, -HISTORY_SIZE:]
        prediction = model.predict(history[:, -HISTORY_SIZE:], model.action_encoder(action_window))[:, -1:]
        outputs.append(prediction[:, 0])
        history = torch.cat((history[:, 1:], prediction), dim=1)
    return torch.stack(outputs, dim=1)


def _prepare_cache(model: Any, rows: list[dict[str, Any]], device: Any, encode_batch_contexts: int) -> dict[str, Any]:
    import torch

    model.to(device).eval().requires_grad_(False)
    _encode_contexts(model, rows, device, batch_size=encode_batch_contexts)
    interface_probe = None
    for row in rows:
        z = row["latent"].unsqueeze(0).to(device)
        transition_actions = torch.as_tensor(row["transition_actions"], device=device).unsqueeze(0)
        branch_prefix = torch.as_tensor(row["branch_prefix_actions"], device=device).unsqueeze(0)
        training_plans = torch.as_tensor(row["triplet_plans"], device=device)
        heldout_plans = torch.as_tensor(row["heldout_plans"], device=device)
        branch_history = z[:, 1:4]
        with torch.no_grad():
            if interface_probe is None:
                recorded_prediction = model.predict(
                    z[:, :3], model.action_encoder(transition_actions)
                )
                sample_branch = imagined_rollout(
                    model,
                    branch_history,
                    branch_prefix,
                    torch.as_tensor(row["triplet_plans"], device=device)[:1],
                )
                if recorded_prediction.shape != z[:, 1:4].shape:
                    raise RuntimeError("pinned recorded-transition prediction shape does not align with next-frame targets")
                if tuple(sample_branch.shape) != (1, ROLLOUT_HORIZON, LATENT_DIM):
                    raise RuntimeError(f"pinned imagined rollout expected [1,5,192], got {tuple(sample_branch.shape)}")
                interface_probe = {
                    "status": "PASS",
                    "hdf5_observation_window_shape": row["pixels_shape"],
                    "recorded_frameskip_indices": [0, 5, 10, 15],
                    "recorded_action_tokens_shape": list(transition_actions.shape[1:]),
                    "recorded_predict_shape": list(recorded_prediction.shape),
                    "recorded_target_shape": list(z[:, 1:4].shape),
                    "teacher_imagined_branch_shape": list(sample_branch.shape),
                    "action_bounds": [ACTION_LOW, ACTION_HIGH],
                }
            training_targets = imagined_rollout(
                model,
                branch_history.expand(3, -1, -1),
                branch_prefix.expand(3, -1, -1),
                training_plans,
            )
            heldout_targets = imagined_rollout(
                model,
                branch_history.expand(5, -1, -1),
                branch_prefix.expand(5, -1, -1),
                heldout_plans,
            )
        row["training_teacher_imagined_targets"] = training_targets.detach().cpu()
        row["heldout_teacher_imagined_targets"] = heldout_targets.detach().cpu()
        row["transition_actions"] = torch.as_tensor(row["transition_actions"], dtype=torch.float32)
        row["branch_prefix_actions"] = torch.as_tensor(row["branch_prefix_actions"], dtype=torch.float32)
        row["triplet_plans"] = torch.as_tensor(row["triplet_plans"], dtype=torch.float32)
        row["heldout_plans"] = torch.as_tensor(row["heldout_plans"], dtype=torch.float32)
        row["delta"] = float(row["delta"])
    assert interface_probe is not None
    return interface_probe


def _frozen_parts_and_trainables(model: Any) -> list[Any]:
    required = ("encoder", "projector", "action_encoder", "predictor", "pred_proj")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"pinned LeWM object is missing expected modules: {missing}")
    model.requires_grad_(False)
    for name in ("encoder", "projector"):
        getattr(model, name).eval().requires_grad_(False)
    trainable = []
    for name in ("action_encoder", "predictor", "pred_proj"):
        module = getattr(model, name)
        module.requires_grad_(True)
        module.train()
        trainable.extend(module.parameters())
    trainable = [parameter for parameter in trainable if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters in action_encoder/predictor/pred_proj")
    return trainable


def _batch_tensors(rows: list[dict[str, Any]], indices: Sequence[int], device: Any) -> dict[str, Any]:
    import torch

    selected = [rows[int(index)] for index in indices]
    return {
        "latent": torch.stack([row["latent"] for row in selected]).to(device),
        "transition_actions": torch.stack([row["transition_actions"] for row in selected]).to(device),
        "branch_prefix_actions": torch.stack([row["branch_prefix_actions"] for row in selected]).to(device),
        "triplet_plans": torch.stack([row["triplet_plans"] for row in selected]).to(device),
        "teacher_targets": torch.stack([row["training_teacher_imagined_targets"] for row in selected]).to(device),
        "delta": torch.tensor([row["delta"] for row in selected], dtype=torch.float32, device=device),
    }


def _arm_loss(model: Any, batch: dict[str, Any], branch_weight: float, lambda_curvature: float) -> tuple[Any, dict[str, Any]]:
    import torch.nn.functional as F

    z = batch["latent"]
    pred = model.predict(z[:, :3], model.action_encoder(batch["transition_actions"]))
    if pred.shape != z[:, 1:4].shape:
        raise RuntimeError(f"recorded transition outputs {tuple(pred.shape)} do not align with targets {tuple(z[:, 1:4].shape)}")
    recorded = F.mse_loss(pred, z[:, 1:4])

    batch_size = z.shape[0]
    plans = batch["triplet_plans"].reshape(batch_size * 3, ROLLOUT_HORIZON, PACKED_ACTION_DIM)
    history = z[:, 1:4].unsqueeze(1).expand(-1, 3, -1, -1).reshape(batch_size * 3, 3, LATENT_DIM)
    prefix = batch["branch_prefix_actions"].unsqueeze(1).expand(-1, 3, -1, -1).reshape(batch_size * 3, 2, PACKED_ACTION_DIM)
    branch_pred = imagined_rollout(model, history, prefix, plans).reshape(batch_size, 3, ROLLOUT_HORIZON, LATENT_DIM)
    branch = F.mse_loss(branch_pred, batch["teacher_targets"])
    terminal = branch_pred[:, :, -1]
    curvature = curvature_penalty(
        terminal[:, 1], terminal[:, 0], terminal[:, 2], batch["delta"]
    )
    total = recorded + branch_weight * branch + lambda_curvature * curvature
    return total, {"recorded_mse": recorded, "imagined_branch_distill_mse": branch, "curvature": curvature}


def _evaluate(model: Any, rows: list[dict[str, Any]], device: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    recorded_values = []
    branch_values = []
    teacher_response_errors = []
    response_norms = []
    taylor_abs: dict[str, list[float]] = {"delta": [], "2delta": []}
    taylor_rel: dict[str, list[float]] = {"delta": [], "2delta": []}
    with torch.no_grad():
        for row in rows:
            z = row["latent"].unsqueeze(0).to(device)
            transition_actions = row["transition_actions"].unsqueeze(0).to(device)
            pred = model.predict(z[:, :3], model.action_encoder(transition_actions))
            recorded_values.append(float(F.mse_loss(pred, z[:, 1:4]).cpu()))

            plans = row["heldout_plans"].to(device)
            history = z[:, 1:4].expand(5, -1, -1)
            prefix = row["branch_prefix_actions"].unsqueeze(0).to(device).expand(5, -1, -1)
            predicted = imagined_rollout(model, history, prefix, plans)
            teacher = row["heldout_teacher_imagined_targets"].to(device)
            branch_values.append(float(F.mse_loss(predicted[0], teacher[0]).cpu()))
            student_center = predicted[0, -1]
            student_plus, student_minus = predicted[1, -1], predicted[2, -1]
            teacher_center = teacher[0, -1]
            teacher_plus, teacher_minus = teacher[1, -1], teacher[2, -1]
            student_response = student_plus - student_minus
            teacher_response = teacher_plus - teacher_minus
            teacher_response_errors.append(
                float((torch.linalg.vector_norm(student_response - teacher_response) / (torch.linalg.vector_norm(teacher_response) + 1e-8)).cpu())
            )
            response_norms.append(float((torch.linalg.vector_norm(student_response) / (2.0 * row["delta"])).cpu()))
            rmse, rel = taylor_residuals(
                student_center.unsqueeze(0),
                student_plus.unsqueeze(0),
                student_minus.unsqueeze(0),
                predicted[3, -1].unsqueeze(0),
                predicted[4, -1].unsqueeze(0),
                row["delta"],
                2.0 * row["delta"],
            )
            taylor_abs["2delta"].append(float(rmse[0].cpu()))
            taylor_rel["2delta"].append(float(rel[0].cpu()))
            # Same-radius held-out Taylor residual is computed with a one-step
            # symmetric estimate and is retained separately from the 2-delta test.
            rmse_d, rel_d = taylor_residuals(
                student_center.unsqueeze(0),
                student_plus.unsqueeze(0),
                student_minus.unsqueeze(0),
                student_plus.unsqueeze(0),
                student_minus.unsqueeze(0),
                row["delta"],
                row["delta"],
            )
            taylor_abs["delta"].append(float(rmse_d[0].cpu()))
            taylor_rel["delta"].append(float(rel_d[0].cpu()))
    return {
        "recorded_transition_mse": sum(recorded_values) / len(recorded_values),
        "center_imagined_teacher_distill_mse": sum(branch_values) / len(branch_values),
        "action_response_norm_per_action_l2": sum(response_norms) / len(response_norms),
        "action_response_teacher_relative_error": sum(teacher_response_errors) / len(teacher_response_errors),
        "heldout_taylor_rmse_at_delta": sum(taylor_abs["delta"]) / len(taylor_abs["delta"]),
        "heldout_taylor_relative_rmse_at_delta": sum(taylor_rel["delta"]) / len(taylor_rel["delta"]),
        "heldout_taylor_rmse_at_2delta": sum(taylor_abs["2delta"]) / len(taylor_abs["2delta"]),
        "heldout_taylor_relative_rmse_at_2delta": sum(taylor_rel["2delta"]) / len(taylor_rel["2delta"]),
    }


def _train_arm(
    base_model: Any,
    rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: Any,
    arm: str,
    lambda_curvature: float,
    schedule: list[list[int]],
    output: Path,
) -> dict[str, Any]:
    import torch

    # Reset RNG for both arms: dropout and any stochastic module see a matched stream.
    torch.manual_seed(args.schedule_seed + 100)
    torch.cuda.manual_seed_all(args.schedule_seed + 100)
    model = copy.deepcopy(base_model).to(device)
    trainable = _frozen_parts_and_trainables(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    trace = []
    start_time = time.time()
    for update, indices in enumerate(schedule, start=1):
        batch = _batch_tensors(rows, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = _arm_loss(model, batch, args.branch_weight, lambda_curvature)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{arm} non-finite loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        trace.append(
            {
                "update": update,
                "context_indices": list(indices),
                "total": float(loss.detach().cpu()),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                **{key: float(value.detach().cpu()) for key, value in components.items()},
            }
        )
    metrics = _evaluate(model, heldout_rows, device)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, output / f"{arm}_predictor_state.pt")
    return {
        "arm": arm,
        "lambda_curvature": lambda_curvature,
        "trainable_modules": ["action_encoder", "predictor", "pred_proj"],
        "frozen_modules": ["encoder", "projector"],
        "updates": args.updates,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "betas": [0.9, 0.999], "eps": 1e-8},
        "elapsed_seconds": time.time() - start_time,
        "last_update": trace[-1],
        "heldout": metrics,
        "trace": trace,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_compute_allocation()
    if not all((args.stablewm_home, args.dataset, args.adapter_script, args.freeze, args.output)):
        raise ValueError("run mode needs --stablewm-home, --dataset, --adapter-script, --freeze, and --output")
    if min(args.train_contexts, args.heldout_contexts, args.batch_contexts, args.updates, args.encode_batch_contexts) < 1:
        raise ValueError("context counts, batch size, and update count must be positive")
    if args.batch_contexts > args.train_contexts or args.delta <= 0 or args.lambda_curvature <= 0:
        raise ValueError("batch must fit the train set and delta/lambda-curvature must be positive")
    if args.branch_weight < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("loss weight, learning rate, or weight decay is invalid")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("PBS allocation is present but no CUDA device is available")
    device = torch.device("cuda:0")
    checkpoint = args.stablewm_home / "pusht" / "lewm_object.ckpt"
    if not args.dataset.is_file() or not checkpoint.is_file() or not args.adapter_script.is_file() or not args.freeze.is_file():
        raise FileNotFoundError("staged HDF5, pinned object checkpoint, adapter script, or freeze record is missing")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    evidence = freeze.get("evidence_boundary", {})
    scope = freeze.get("scope", {})
    slate = freeze.get("shared_data_and_schedule", {}).get("action_query_slate", {})
    if evidence.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("freeze record source commit differs from the E2 pinned LeWM commit")
    if (
        int(scope.get("latent_dim", -1)) != LATENT_DIM
        or int(scope.get("predictor_max_history_length", -1)) != HISTORY_SIZE
        or int(scope.get("packed_action_token_dim", -1)) != PACKED_ACTION_DIM
        or int(scope.get("horizon", -1)) != ROLLOUT_HORIZON
        or slate.get("official_action_low") != [ACTION_LOW, ACTION_LOW]
        or slate.get("official_action_high") != [ACTION_HIGH, ACTION_HIGH]
    ):
        raise RuntimeError("frozen LeWM latent/action contract differs from E2 assumptions")
    # Model API is gated below against the staged official object contract. The
    # adapter path identifies the local HDF5/LeWM interface record used here.
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_modules = ("encode", "predict", "action_encoder", "encoder", "projector", "predictor", "pred_proj")
    missing = [name for name in expected_modules if not hasattr(model, name)]
    if missing:
        raise TypeError(f"staged official LeWM checkpoint lacks expected API: {missing}")
    model.eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _create_manifest(args)
    manifest["action_contract"]["delta"] = args.delta
    _write_json(args.output / "context_manifest.json", manifest)
    rows = _read_context_arrays(args.dataset, manifest)
    runtime_interface_probe = _prepare_cache(model, rows, device, args.encode_batch_contexts)
    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    if len(train_rows) != args.train_contexts or len(heldout_rows) != args.heldout_contexts:
        raise RuntimeError("cached context count changed from the frozen manifest")
    # CPU model copy is the common official initialization for both matched arms.
    base_model = model.to("cpu").eval().requires_grad_(False)
    del model
    schedule_rng = random.Random(args.schedule_seed)
    context_schedule = [
        schedule_rng.sample(range(len(train_rows)), args.batch_contexts)
        for _ in range(args.updates)
    ]
    arms = []
    for name, coefficient in (("lambda_0", 0.0), ("lambda_positive", args.lambda_curvature)):
        arms.append(_train_arm(base_model, train_rows, heldout_rows, args, device, name, coefficient, context_schedule, args.output))
        torch.cuda.empty_cache()
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "source_commit": SOURCE_COMMIT,
        "checkpoint": str(checkpoint),
        "dataset": str(args.dataset),
        "adapter_script": str(args.adapter_script),
        "freeze_record": str(args.freeze),
        "runtime_interface_probe": runtime_interface_probe,
        "branch_targets": "frozen teacher imagined autoregressive latents; not observed real counterfactual branches",
        "recorded_transition_target": "frozen official encoder/projector embeddings of sampled HDF5 observations at stride 5",
        "loss": "recorded_transition_mse + branch_weight * frozen_teacher_imagined_branch_mse + lambda_curvature * normalized_three_point_terminal_latent_curvature",
        "sigreg": "not optimized: frozen encoder/projector make the official embedding SIGReg term constant with respect to trained modules",
        "target_encoder": "pinned LeWM training uses the same online encoder embeddings; E2 freezes encoder/projector and detaches their shared embeddings",
        "matched_design": {"common_official_initialization": True, "same_parent_episode_manifest": True, "same_context_schedule": True, "context_schedule_indices": context_schedule, "same_torch_seed_for_dropout": True, "same_teacher_imagined_targets": True, "same_optimizer_hyperparameters": True, "lambda_values": [0.0, args.lambda_curvature]},
        "budget": {"train_contexts": args.train_contexts, "heldout_contexts": args.heldout_contexts, "train_episodes": args.train_episodes, "heldout_episodes": args.heldout_episodes, "batch_contexts": args.batch_contexts, "updates_per_arm": args.updates, "delta_action_l2": args.delta, "branch_weight": args.branch_weight},
        "arms": arms,
    }
    _write_json(args.output / "e2_summary.json", result)
    return result


def main() -> int:
    args = parse_args()
    if args.mode == "status":
        print(json.dumps({"schema": SCHEMA, "status": "CODE_ONLY_NOT_RUN", "source_commit": SOURCE_COMMIT}, indent=2))
        return 0
    result = run(args)
    print(json.dumps({"schema": result["schema"], "status": result["status"], "arms": [arm["arm"] for arm in result["arms"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
