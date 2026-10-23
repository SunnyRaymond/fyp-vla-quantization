"""Explicitly weighted paired endpoint and frozen-reader objectives.

This module provides loss composition only. It has no experiment weights,
optimizer, model construction, or training loop.
"""
import math
import numbers


VARIANTS = ("clean", "noisy")
CONFIG_KEYS = (
    "action_absolute_weight",
    "response_weight",
    "latent_absolute_weight",
    "control_weight",
    "lambda_latent",
)
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)


def validate_config(config):
    """Validate and return the exact five caller-supplied finite weights."""
    if not isinstance(config, dict) or set(config) != set(CONFIG_KEYS):
        raise ValueError(f"config must contain exactly {CONFIG_KEYS}")
    values = {}
    for key in CONFIG_KEYS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{key} must be a real numeric coefficient")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        values[key] = value
    if values["action_absolute_weight"] <= 0:
        raise ValueError("action_absolute_weight must be positive to retain the absolute anchor")
    if values["lambda_latent"] <= 0:
        raise ValueError("lambda_latent must be positive")
    return values


def _check_tensor(torch, value, shape, label):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{label} must be a Tensor with shape {shape}")


def _check_variants(mapping, label):
    if not isinstance(mapping, dict) or set(mapping) != set(VARIANTS):
        raise ValueError(f"{label} must contain exactly clean and noisy")


def paired_endpoint_objective(torch, actions, futures, targets, stats, config):
    """Return the paired endpoint loss and its three unweighted raw terms.

    Absolute action KL uses the first 10x6 motor values. Absolute latent KL
    excludes future frame zero. The response KL couples clean and noisy actions.
    """
    weights = validate_config(config)
    _check_variants(actions, "actions")
    _check_variants(futures, "futures")
    _check_variants(targets, "targets")

    from softbin_loss import pointwise_kl
    from control_losses import action_response_kl

    student_actions, student_futures, teacher_actions, teacher_futures = {}, {}, {}, {}
    student_devices = set()
    for variant in VARIANTS:
        _check_tensor(torch, actions[variant], ACTION_SHAPE, f"actions[{variant}]")
        _check_tensor(torch, futures[variant], FUTURE_SHAPE, f"futures[{variant}]")
        if (not isinstance(targets[variant], dict)
                or not {"action", "future"}.issubset(targets[variant])):
            raise ValueError(f"targets[{variant}] must contain action and future")
        _check_tensor(torch, targets[variant]["action"], ACTION_SHAPE,
                      f"targets[{variant}].action")
        _check_tensor(torch, targets[variant]["future"], FUTURE_SHAPE,
                      f"targets[{variant}].future")
        student_devices.update((actions[variant].device, futures[variant].device))
        student_actions[variant] = actions[variant]
        student_futures[variant] = futures[variant]
        teacher_actions[variant] = targets[variant]["action"].detach().to(
            device=actions[variant].device)
        teacher_futures[variant] = targets[variant]["future"].detach().to(
            device=futures[variant].device)
    if len(student_devices) != 1:
        raise ValueError("clean/noisy actions and futures must share one device")

    action_stats, latent_stats = stats["action"], stats["latent"]
    softbin_config = stats["config"]
    action_terms, latent_terms = [], []
    for variant in VARIANTS:
        action_terms.append(pointwise_kl(
            torch, student_actions[variant][:10, :6], teacher_actions[variant][:10, :6],
            action_stats["mean"], action_stats["scale"], -1, softbin_config))
        latent_terms.append(pointwise_kl(
            torch, student_futures[variant][:, :, 1:].float(),
            teacher_futures[variant][:, :, 1:].float(),
            latent_stats["mean"], latent_stats["scale"], 1, softbin_config))

    action_absolute = 0.5 * (action_terms[0] + action_terms[1])
    latent_absolute = 0.5 * (latent_terms[0] + latent_terms[1])
    response = action_response_kl(
        torch, student_actions["clean"], student_actions["noisy"],
        teacher_actions["clean"], teacher_actions["noisy"], stats)
    endpoint_loss = (
        weights["action_absolute_weight"] * action_absolute
        + weights["response_weight"] * response
        + weights["lambda_latent"] * weights["latent_absolute_weight"] * latent_absolute
    )
    raw_terms = {
        "action_absolute_kl": action_absolute,
        "action_response_kl": response,
        "latent_absolute_kl": latent_absolute,
    }
    return endpoint_loss, raw_terms


def weighted_reader_objective(torch, reader_action, teacher_action, stats, config):
    """Return one variant's half-weighted fixed-reader control KL and raw KL."""
    weights = validate_config(config)
    _check_tensor(torch, reader_action, ACTION_SHAPE, "reader_action")
    _check_tensor(torch, teacher_action, ACTION_SHAPE, "teacher_action")

    from control_losses import latent_control_kl

    target = teacher_action.detach().to(device=reader_action.device)
    raw_control_kl = latent_control_kl(torch, reader_action, target, stats)
    weighted = 0.5 * weights["lambda_latent"] * weights["control_weight"] * raw_control_kl
    return weighted, raw_control_kl
