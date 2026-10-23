"""Frozen action-temporal and latent-spatiotemporal relational KL losses."""
from __future__ import annotations

import math


_STD_FLOOR = 0.02
_ACTION_SHAPE = (32, 7)
_FUTURE_SHAPE = (1, 48, 3, 14, 28)


def _frozen_config(config):
    if config is None:
        raise ValueError("relational config with frozen temperatures is required")
    result = {}
    for key in ("action_temperature", "latent_temperature"):
        if key not in config:
            raise ValueError(f"missing relational config field: {key}")
        value = float(config[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = value
    if "std_floor" in config and not math.isclose(
        float(config["std_floor"]), _STD_FLOOR, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"frozen relational config requires std_floor={_STD_FLOOR}")
    result["std_floor"] = _STD_FLOOR
    return result


def _validate_inputs(action, future, target_action=None, target_future=None):
    if tuple(action.shape) != _ACTION_SHAPE:
        raise ValueError(f"action must have shape {_ACTION_SHAPE}")
    if tuple(future.shape) != _FUTURE_SHAPE:
        raise ValueError(f"future must have shape {_FUTURE_SHAPE}")
    if target_action is not None and tuple(target_action.shape) != _ACTION_SHAPE:
        raise ValueError(f"target_action must have shape {_ACTION_SHAPE}")
    if target_future is not None and tuple(target_future.shape) != _FUTURE_SHAPE:
        raise ValueError(f"target_future must have shape {_FUTURE_SHAPE}")


def _channel_stats(torch, values, group, channels):
    try:
        mean = torch.as_tensor(group["mean"], device=values.device, dtype=torch.float32).detach().reshape(-1)
        scale = torch.as_tensor(group["scale"], device=values.device, dtype=torch.float32).detach().reshape(-1)
    except (KeyError, TypeError) as error:
        raise ValueError("stats require action/latent mean and scale vectors") from error
    if mean.numel() != channels or scale.numel() != channels:
        raise ValueError("frozen C80 mean and scale must match the channel count")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("frozen C80 mean and scale must be finite")
    return mean, scale.clamp_min(_STD_FLOOR)


def _action_tokens(torch, action, stats):
    values = action[:10, :6].float()
    mean, scale = _channel_stats(torch, values, stats["action"], 6)
    values = (values - mean.reshape(1, 6)) / scale.reshape(1, 6)
    # Sliding ordered windows retain temporal position within each 18-value token.
    return values.unfold(0, 3, 1).permute(0, 2, 1).reshape(8, 18)


def _latent_tokens(torch, future, stats):
    values = future[:, :, 1:].float()
    mean, scale = _channel_stats(torch, values, stats["latent"], 48)
    values = (values - mean.reshape(1, 48, 1, 1, 1)) / scale.reshape(1, 48, 1, 1, 1)
    # Width is two separate 14-pixel camera views; pool each view on its own.
    views = torch.stack((values[..., :14], values[..., 14:]), dim=3)
    batch, channels, times, cameras, height, width = views.shape
    pooled = torch.nn.functional.avg_pool2d(
        views.permute(0, 2, 3, 1, 4, 5).reshape(batch * times * cameras, channels, height, width),
        kernel_size=2,
        stride=2,
    )
    return pooled.reshape(batch, times, cameras, channels, 7, 7).permute(
        0, 1, 2, 4, 5, 3
    ).reshape(196, 48)


def _latent_neighbor_mask(torch, device):
    # Token order is (time, camera, row, column); camera grids never share edges.
    index = torch.arange(196, device=device)
    column = index % 7
    row = (index // 7) % 7
    camera = (index // 49) % 2
    time = index // 98
    dx = (column[:, None] - column[None, :]).abs()
    dy = (row[:, None] - row[None, :]).abs()
    same_time_local = (
        (time[:, None] == time[None, :])
        & (camera[:, None] == camera[None, :])
        & (torch.maximum(dx, dy) == 1)
    )
    same_position_next_time = (
        (time[:, None] != time[None, :])
        & (camera[:, None] == camera[None, :])
        & (row[:, None] == row[None, :])
        & (column[:, None] == column[None, :])
    )
    return same_time_local | same_position_next_time


def _action_neighbor_mask(torch, device):
    return ~torch.eye(8, dtype=torch.bool, device=device)


def _pairwise_distance(torch, tokens):
    return (tokens[:, None, :] - tokens[None, :, :]).square().mean(dim=-1)


def _masked_log_probs(torch, distances, neighbor_mask, temperature):
    logits = -distances / temperature
    logits = logits.masked_fill(~neighbor_mask, torch.finfo(logits.dtype).min)
    return torch.log_softmax(logits, dim=-1)


def _relational_kl(torch, student_tokens, teacher_tokens, neighbor_mask, temperature):
    student_distances = _pairwise_distance(torch, student_tokens)
    with torch.no_grad():
        teacher_distances = _pairwise_distance(torch, teacher_tokens.detach())
        teacher_logp = _masked_log_probs(torch, teacher_distances, neighbor_mask, temperature)
        teacher_prob = teacher_logp.exp()
    student_logp = _masked_log_probs(torch, student_distances, neighbor_mask, temperature)
    per_row = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
    loss = per_row.mean().clamp_min(0.0)
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("relational KL produced a non-finite value")
    return loss


def teacher_distances(torch, action, future, stats):
    """Return frozen-standardized teacher distances on each fixed neighbor graph.

    Concatenate these 1-D tensors over C80 and take each positive-value median
    outside the loss call; the resulting temperatures stay fixed during training.
    """
    _validate_inputs(action, future)
    action_tokens = _action_tokens(torch, action.detach(), stats)
    latent_tokens = _latent_tokens(torch, future.detach(), stats)
    action_mask = _action_neighbor_mask(torch, action.device)
    latent_mask = _latent_neighbor_mask(torch, future.device)
    action_distances = _pairwise_distance(torch, action_tokens).detach()[action_mask]
    latent_distances = _pairwise_distance(torch, latent_tokens).detach()[latent_mask]
    if not bool(torch.isfinite(action_distances).all()) or not bool(torch.isfinite(latent_distances).all()):
        raise FloatingPointError("teacher relational distances must be finite")
    action_distances = action_distances[action_distances > 0]
    latent_distances = latent_distances[latent_distances > 0]
    if action_distances.numel() == 0 or latent_distances.numel() == 0:
        raise ValueError("C80 temperature fitting requires positive teacher neighbor distances")
    return {"action": action_distances, "latent": latent_distances}


def relational_losses(torch, action, future, target_action, target_future, stats, config=None):
    """Return the two fixed-graph KL terms; teacher values are detached."""
    if config is None:
        config = stats.get("config")
    frozen = _frozen_config(config)
    _validate_inputs(action, future, target_action, target_future)
    if len({action.device, future.device, target_action.device, target_future.device}) != 1:
        raise ValueError("student and teacher tensors must share one device")
    if not bool(torch.isfinite(action.detach()).all()) or not bool(torch.isfinite(future.detach()).all()):
        raise ValueError("student tensors must be finite")

    student_action = _action_tokens(torch, action, stats)
    student_latent = _latent_tokens(torch, future, stats)
    with torch.no_grad():
        teacher_action = _action_tokens(torch, target_action.detach(), stats)
        teacher_latent = _latent_tokens(torch, target_future.detach(), stats)
    action_mask = _action_neighbor_mask(torch, action.device)
    latent_mask = _latent_neighbor_mask(torch, future.device)
    return {
        "action_temporal_kl": _relational_kl(
            torch, student_action, teacher_action, action_mask, frozen["action_temperature"]
        ),
        "latent_spatiotemporal_kl": _relational_kl(
            torch, student_latent, teacher_latent, latent_mask, frozen["latent_temperature"]
        ),
    }
