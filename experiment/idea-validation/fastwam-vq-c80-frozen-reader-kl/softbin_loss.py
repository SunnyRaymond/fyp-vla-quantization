"""Frozen first-10 motor and future-latent losses with soft-bin KL terms."""
from __future__ import annotations

import math


def _frozen_config(config):
    if config is None:
        raise ValueError("soft-bin config is required")
    expected = {
        "bins": 16,
        "interior_min": -3.5,
        "interior_max": 3.5,
        "sigma": 0.375,
        "std_floor": 0.02,
    }
    for key, value in expected.items():
        if key not in config:
            raise ValueError(f"missing soft-bin config field: {key}")
        actual = config[key]
        if isinstance(value, float):
            matches = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = int(actual) == value
        if not matches:
            raise ValueError(f"frozen soft-bin config requires {key}={value}")
    return expected


def softbin_log_probs(torch, x_standardized, edges, sigma):
    """Return FP32 log probabilities for Gaussian-noised values in 16 bins."""
    x = x_standardized.float()
    edges = torch.as_tensor(edges, device=x.device, dtype=torch.float32).detach()
    if edges.ndim != 1 or edges.numel() != 15:
        raise ValueError("16 bins require 15 one-dimensional interior edges")
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    z = (edges - x.unsqueeze(-1)) / sigma
    lower, upper = z[..., :-1], z[..., 1:]

    # Use the CDF on the left and the survival function on the right. Substitute
    # benign intervals into the unused branch so its backward pass stays finite.
    use_cdf = lower < 0
    cdf_upper = torch.where(use_cdf, upper, torch.zeros_like(upper))
    cdf_lower = torch.where(use_cdf, lower, -torch.ones_like(lower))
    log_cdf_upper = torch.special.log_ndtr(cdf_upper)
    log_cdf_lower = torch.special.log_ndtr(cdf_lower)
    log_cdf_interval = log_cdf_upper + torch.log1p(
        -torch.exp(log_cdf_lower - log_cdf_upper)
    )

    survival_lower = torch.where(use_cdf, torch.ones_like(lower), lower)
    survival_upper = torch.where(use_cdf, 2 * torch.ones_like(upper), upper)
    log_survival_lower = torch.special.log_ndtr(-survival_lower)
    log_survival_upper = torch.special.log_ndtr(-survival_upper)
    log_survival_interval = log_survival_lower + torch.log1p(
        -torch.exp(log_survival_upper - log_survival_lower)
    )
    log_middle = torch.where(use_cdf, log_cdf_interval, log_survival_interval)

    log_left_tail = torch.special.log_ndtr(z[..., :1])
    log_right_tail = torch.special.log_ndtr(-z[..., -1:])
    return torch.cat((log_left_tail, log_middle, log_right_tail), dim=-1)


def _channel_stats(torch, x, mean, scale, channel_dim, std_floor):
    if x.ndim == 0:
        raise ValueError("pointwise KL input must have a channel dimension")
    channel_dim %= x.ndim
    mean = torch.as_tensor(mean, device=x.device, dtype=torch.float32).detach().reshape(-1)
    scale = torch.as_tensor(scale, device=x.device, dtype=torch.float32).detach().reshape(-1)
    channels = x.shape[channel_dim]
    if mean.numel() != channels or scale.numel() != channels:
        raise ValueError("calibration mean and scale must match the selected channel axis")
    shape = [1] * x.ndim
    shape[channel_dim] = channels
    return mean.reshape(shape), scale.clamp_min(std_floor).reshape(shape)


def pointwise_kl(torch, student, teacher, mean, scale, channel_dim, config):
    """Mean per-scalar KL(teacher || student), with frozen channel calibration."""
    frozen = _frozen_config(config)
    student = student.float()
    teacher = teacher.detach().float()
    if student.shape != teacher.shape:
        raise ValueError("student and teacher tensors must have the same shape")
    mean, scale = _channel_stats(
        torch, student, mean, scale, channel_dim, frozen["std_floor"]
    )
    student_z = (student - mean) / scale
    teacher_z = (teacher - mean) / scale
    edges = torch.linspace(
        frozen["interior_min"], frozen["interior_max"], 15,
        device=student.device, dtype=torch.float32,
    )
    student_logp = softbin_log_probs(torch, student_z, edges, frozen["sigma"])
    teacher_logp = softbin_log_probs(torch, teacher_z, edges, frozen["sigma"])
    per_scalar = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    if not bool(torch.isfinite(per_scalar.detach()).all()):
        raise FloatingPointError("soft-bin KL produced a non-finite value")
    if bool((per_scalar.detach() < -1e-5).any()):
        raise FloatingPointError("soft-bin KL is materially negative")
    return per_scalar.clamp_min(0).mean()


def losses(torch, action, future, target_action, target_future, stats, config=None):
    """Return the four frozen loss terms; callers choose the requested pair."""
    if config is None:
        config = stats.get("config")
    frozen = _frozen_config(config)
    if tuple(action.shape) != (32, 7) or tuple(target_action.shape) != (32, 7):
        raise ValueError("action and target_action must have shape [32, 7]")
    if action.shape != target_action.shape:
        raise ValueError("action and target_action must have the same shape")
    if future.ndim != 5 or future.shape != target_future.shape or future.shape[2] < 2:
        raise ValueError("future and target_future must match [B, C, T>=2, H, W]")

    target_action = target_action.detach()
    target_future = target_future.detach()
    action_error = action[:10, :6].float() - target_action[:10, :6].float()
    future_student = future[:, :, 1:].float()
    future_teacher = target_future[:, :, 1:].float()
    future_error = future_student - future_teacher
    latent_nmse = future_error.square().sum() / future_teacher.square().sum().clamp_min(1e-12)

    action_stats = stats["action"]
    latent_stats = stats["latent"]
    return {
        "action_mse": action_error.square().mean(),
        "latent_nmse": latent_nmse,
        "action_kl": pointwise_kl(
            torch, action[:10, :6], target_action[:10, :6],
            action_stats["mean"], action_stats["scale"], -1, frozen,
        ),
        "latent_kl": pointwise_kl(
            torch, future_student, future_teacher,
            latent_stats["mean"], latent_stats["scale"], 1, frozen,
        ),
    }
