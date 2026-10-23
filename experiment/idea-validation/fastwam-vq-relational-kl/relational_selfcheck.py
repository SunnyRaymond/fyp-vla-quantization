"""PBS-only synthetic controls for the fixed relational KL losses."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path


def require_allocation():
    """Require this process to run on a host listed in its active PBS allocation."""
    job_id = os.environ.get("PBS_JOBID")
    hostname = socket.gethostname()
    if not job_id:
        raise RuntimeError("relational selfcheck requires an active PBS allocation")
    if "login" in hostname.lower():
        raise RuntimeError(f"relational selfcheck is forbidden on login hosts: {hostname}")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_NODEFILE must identify the active compute allocation")
    short_host = hostname.split(".", 1)[0].lower()
    allocated_hosts = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if short_host not in allocated_hosts:
        raise RuntimeError(f"hostname {hostname} is absent from PBS_NODEFILE")
    return job_id, hostname


def _float(value):
    return float(value.detach().cpu())


def _relation_values(torch, relational_losses, action, future, target_action, target_future, stats, config):
    terms = relational_losses(torch, action, future, target_action, target_future, stats, config)
    return {key: _float(value) for key, value in terms.items()}


def _softbin_anchor(torch, pointwise_kl, action, future, target_action, target_future, stats, config):
    action_stats = stats["action"]
    latent_stats = stats["latent"]
    return pointwise_kl(
        torch, action[:10, :6], target_action[:10, :6],
        action_stats["mean"], action_stats["scale"], -1, config,
    ) + pointwise_kl(
        torch, future[:, :, 1:], target_future[:, :, 1:],
        latent_stats["mean"], latent_stats["scale"], 1, config,
    )


def _make_fixture(torch, device):
    action_mean = torch.tensor([0.1, -0.2, 0.05, 0.3, -0.1, 0.2], device=device)
    action_scale = torch.tensor([0.55, 0.72, 0.91, 1.08, 1.24, 0.83], device=device)
    latent_mean = torch.linspace(-0.12, 0.13, 48, device=device)
    latent_scale = 0.45 + 0.01 * (torch.arange(48, device=device) % 13)
    stats = {
        "action": {"mean": action_mean, "scale": action_scale},
        "latent": {"mean": latent_mean, "scale": latent_scale},
    }

    time = torch.arange(10, device=device, dtype=torch.float32)[:, None]
    feature = torch.arange(6, device=device, dtype=torch.float32)[None, :]
    action_z = (
        0.58 * torch.sin(0.47 * (time + 1) * (feature + 1))
        + 0.31 * torch.cos(0.29 * (time + 1) * (feature + 2))
        + 0.035 * time * (feature + 1)
    )
    action = torch.zeros((32, 7), device=device, dtype=torch.float32)
    action[:10, :6] = action_mean + action_scale * action_z

    channel = torch.arange(48, device=device, dtype=torch.float32)[:, None, None, None]
    future_time = torch.arange(3, device=device, dtype=torch.float32)[None, :, None, None]
    row = torch.arange(14, device=device, dtype=torch.float32)[None, None, :, None]
    column = torch.arange(28, device=device, dtype=torch.float32)[None, None, None, :]
    camera = (column >= 14).float()
    within_camera_column = column.remainder(14)
    latent_z = (
        0.46 * torch.sin(0.19 * (row + 1) * (channel.remainder(7) + 1))
        + 0.34 * torch.cos(0.23 * (within_camera_column + 1) * (channel.remainder(5) + 1))
        + 0.24 * torch.sin(0.71 * (future_time + 1) * (channel.remainder(3) + 1))
        + 0.12 * camera * (channel.remainder(4) + 1)
        + 0.10 * torch.sin(0.08 * (row + 1) * (within_camera_column + 1))
        + 0.28 * torch.sin(0.83 * (future_time + 1)) * (
            0.55 * torch.sin(0.14 * (row + 1) * (channel.remainder(5) + 1))
            + 0.45 * torch.cos(0.18 * (within_camera_column + 1) * (channel.remainder(6) + 1))
        )
    )
    future = torch.zeros((1, 48, 3, 14, 28), device=device, dtype=torch.float32)
    future[0] = latent_mean[:, None, None, None] + latent_scale[:, None, None, None] * latent_z
    return action, future, stats


def run_checks(torch, relational_losses, teacher_distances, pointwise_kl, device):
    action, future, stats = _make_fixture(torch, device)
    config = {"action_temperature": 0.18, "latent_temperature": 0.16}
    softbin_config = {
        "bins": 16,
        "interior_min": -3.5,
        "interior_max": 3.5,
        "sigma": 0.375,
        "std_floor": 0.02,
    }

    same = _relation_values(torch, relational_losses, action, future, action, future, stats, config)
    assert max(same.values()) <= 1e-7, same

    reversed_action = action.clone()
    reversed_action[:10, :6] = action[:10, :6].flip(0)
    reversed_terms = _relation_values(
        torch, relational_losses, reversed_action, future, action, future, stats, config
    )
    assert reversed_terms["action_temporal_kl"] > 1e-5, reversed_terms

    order = torch.tensor([0, 2, 5, 8, 1, 4, 7, 9, 3, 6], device=device)
    permuted_action = action.clone()
    permuted_action[:10, :6] = action[:10, :6].index_select(0, order)
    permuted_terms = _relation_values(
        torch, relational_losses, permuted_action, future, action, future, stats, config
    )
    assert permuted_terms["action_temporal_kl"] > 1e-5, permuted_terms

    action_bias = action.clone()
    action_bias[:10, :6] += stats["action"]["scale"] * torch.tensor(
        [0.24, -0.21, 0.19, -0.17, 0.23, -0.20], device=device
    )
    action_signflip = action.clone()
    action_signflip[:10, :6] = (
        2 * stats["action"]["mean"] - action[:10, :6]
    )
    bias_terms = _relation_values(
        torch, relational_losses, action_bias, future, action, future, stats, config
    )
    signflip_terms = _relation_values(
        torch, relational_losses, action_signflip, future, action, future, stats, config
    )
    assert bias_terms["action_temporal_kl"] <= 1e-7, bias_terms
    assert signflip_terms["action_temporal_kl"] <= 1e-7, signflip_terms
    action_bias_anchor = _float(
        _softbin_anchor(torch, pointwise_kl, action_bias, future, action, future, stats, softbin_config)
    )
    action_signflip_anchor = _float(
        _softbin_anchor(torch, pointwise_kl, action_signflip, future, action, future, stats, softbin_config)
    )
    assert action_bias_anchor > 1e-5 and action_signflip_anchor > 1e-5

    action_target_mse = (action_bias[:10, :6] - action[:10, :6]).square().mean()
    reversed_delta = action[:10, :6].flip(0) - action[:10, :6]
    reverse_scale = (action_target_mse / reversed_delta.square().mean()).sqrt()
    matched_shuffle = action.clone()
    matched_shuffle[:10, :6] += reverse_scale * reversed_delta
    shuffle_mse = (matched_shuffle[:10, :6] - action[:10, :6]).square().mean()
    shuffle_terms = _relation_values(
        torch, relational_losses, matched_shuffle, future, action, future, stats, config
    )
    torch.testing.assert_close(action_target_mse, shuffle_mse, rtol=1e-6, atol=1e-8)
    assert shuffle_terms["action_temporal_kl"] > bias_terms["action_temporal_kl"] + 1e-5

    linear_action = action.clone()
    linear_slopes = torch.tensor([0.16, -0.11, 0.13, 0.09, -0.14, 0.12], device=device)
    linear_offsets = torch.tensor([-0.25, 0.18, 0.07, -0.13, 0.22, -0.19], device=device)
    linear_z = linear_offsets + torch.arange(10, device=device, dtype=torch.float32)[:, None] * linear_slopes
    linear_action[:10, :6] = (
        stats["action"]["mean"] + stats["action"]["scale"] * linear_z
    )
    reversed_linear_action = linear_action.clone()
    reversed_linear_action[:10, :6] = linear_action[:10, :6].flip(0)
    linear_reverse_terms = _relation_values(
        torch, relational_losses, reversed_linear_action, future, linear_action, future, stats, config
    )
    linear_reverse_anchor = _float(
        _softbin_anchor(
            torch, pointwise_kl, reversed_linear_action, future,
            linear_action, future, stats, softbin_config,
        )
    )
    assert linear_reverse_terms["action_temporal_kl"] <= 1e-6, linear_reverse_terms
    assert linear_reverse_anchor > 1e-5

    swapped_future = future.clone()
    swapped_future[:, :, 1:] = future[:, :, 1:].flip(2)
    swapped_terms = _relation_values(
        torch, relational_losses, action, swapped_future, action, future, stats, config
    )
    assert swapped_terms["latent_spatiotemporal_kl"] > 1e-5, swapped_terms

    uniform_time_future = future.clone()
    uniform_shift = stats["latent"]["scale"] * (
        0.23 + 0.08 * torch.sin(torch.arange(48, device=device, dtype=torch.float32) * 0.27)
    )
    uniform_time_future[:, :, 2] = uniform_time_future[:, :, 1] + uniform_shift.reshape(1, 48, 1, 1)
    swapped_uniform_time = uniform_time_future.clone()
    swapped_uniform_time[:, :, 1:] = uniform_time_future[:, :, 1:].flip(2)
    uniform_swap_terms = _relation_values(
        torch, relational_losses, action, swapped_uniform_time,
        action, uniform_time_future, stats, config,
    )
    uniform_swap_anchor = _float(
        _softbin_anchor(
            torch, pointwise_kl, action, swapped_uniform_time,
            action, uniform_time_future, stats, softbin_config,
        )
    )
    assert uniform_swap_terms["latent_spatiotemporal_kl"] <= 1e-6, uniform_swap_terms
    assert uniform_swap_anchor > 1e-5

    rolled_future = future.clone()
    retained = future[:, :, 1:]
    rolled_future[:, :, 1:] = torch.cat(
        (torch.roll(retained[..., :14], shifts=1, dims=-2),
         torch.roll(retained[..., 14:], shifts=1, dims=-2)), dim=-1
    )
    rolled_terms = _relation_values(
        torch, relational_losses, action, rolled_future, action, future, stats, config
    )
    assert rolled_terms["latent_spatiotemporal_kl"] > 1e-5, rolled_terms

    latent_bias = future.clone()
    channel_shift = stats["latent"]["scale"] * (
        0.20 * torch.sin(torch.arange(48, device=device, dtype=torch.float32) * 0.31) + 0.12
    )
    latent_bias[:, :, 1:] += channel_shift.reshape(1, 48, 1, 1, 1)
    latent_bias_terms = _relation_values(
        torch, relational_losses, action, latent_bias, action, future, stats, config
    )
    assert latent_bias_terms["latent_spatiotemporal_kl"] <= 1e-7, latent_bias_terms

    local_delta = torch.zeros_like(future)
    local_pattern = torch.cos(torch.arange(48, device=device, dtype=torch.float32) * 0.37)
    local_delta[:, :, 1:, 3:5, 4:6] = (
        stats["latent"]["scale"].reshape(1, 48, 1, 1, 1)
        * local_pattern.reshape(1, 48, 1, 1, 1)
    )
    target_latent_mse = (latent_bias[:, :, 1:] - future[:, :, 1:]).square().mean()
    local_scale = (target_latent_mse / local_delta[:, :, 1:].square().mean()).sqrt()
    matched_local = future + local_scale * local_delta
    local_mse = (matched_local[:, :, 1:] - future[:, :, 1:]).square().mean()
    torch.testing.assert_close(target_latent_mse, local_mse, rtol=1e-6, atol=1e-8)
    local_terms = _relation_values(
        torch, relational_losses, action, matched_local, action, future, stats, config
    )
    assert local_terms["latent_spatiotemporal_kl"] > latent_bias_terms["latent_spatiotemporal_kl"] + 1e-5
    latent_bias_anchor = _float(
        _softbin_anchor(torch, pointwise_kl, action, latent_bias, action, future, stats, softbin_config)
    )
    assert latent_bias_anchor > 1e-5

    ignored_action = action.clone()
    ignored_action[:, 6] += 7.0
    ignored_action[10:, :6] -= 9.0
    ignored_future = future.clone()
    ignored_future[:, :, 0] += 11.0
    ignored_terms = _relation_values(
        torch, relational_losses, ignored_action, ignored_future, action, future, stats, config
    )
    assert ignored_terms == same, ignored_terms

    distances = teacher_distances(torch, action, future, stats)
    assert distances["action"].ndim == 1 and bool((distances["action"] > 0).any())
    assert distances["latent"].ndim == 1 and bool((distances["latent"] > 0).any())

    student_action = reversed_action.clone().requires_grad_()
    student_future = rolled_future.clone().requires_grad_()
    teacher_action = action.clone().requires_grad_()
    teacher_future = future.clone().requires_grad_()
    grad_terms = relational_losses(
        torch, student_action, student_future, teacher_action, teacher_future, stats, config
    )
    sum(grad_terms.values()).backward()
    assert student_action.grad is not None and bool(torch.isfinite(student_action.grad).all())
    assert student_future.grad is not None and bool(torch.isfinite(student_future.grad).all())
    assert bool((student_action.grad.abs() > 0).any()) and bool((student_future.grad.abs() > 0).any())
    assert teacher_action.grad is None and teacher_future.grad is None
    assert bool((student_action.grad[:, 6] == 0).all())
    assert bool((student_action.grad[10:] == 0).all())
    assert bool((student_future.grad[:, :, 0] == 0).all())

    shape_rejected = False
    try:
        relational_losses(torch, action[:, :6], future, action, future, stats, config)
    except ValueError:
        shape_rejected = True
    assert shape_rejected

    return {
        "checks": {
            "same_teacher_student_near_zero": True,
            "ordered_action_reverse_and_permutation_penalized": True,
            "action_bias_and_centered_signflip_relation_blindness": True,
            "softbin_absolute_anchor_detects_action_bias_and_signflip": True,
            "action_bias_vs_temporal_shuffle_matched_mse": True,
            "linear_action_reverse_relation_blindness_with_softbin_anchor": True,
            "latent_time_swap_and_spatial_roll_penalized": True,
            "uniform_temporal_shift_swap_relation_blindness_with_softbin_anchor": True,
            "latent_global_channel_bias_relation_blindness": True,
            "softbin_absolute_anchor_detects_latent_channel_bias": True,
            "latent_bias_vs_local_perturbation_matched_mse": True,
            "gripper_late22_and_observed_frame_ignored": True,
            "teacher_distances_return_positive_neighbor_samples": True,
            "teacher_detached_and_student_gradients_finite_nonzero": True,
            "ignored_action_and_frame0_gradients_zero": True,
            "shape_errors_rejected": True,
        },
        "metrics": {
            "same": same,
            "action_reverse": reversed_terms,
            "action_permutation": permuted_terms,
            "action_global_bias": bias_terms,
            "action_signflip": signflip_terms,
            "action_matched_shuffle": shuffle_terms,
            "linear_action_reverse": linear_reverse_terms,
            "linear_action_reverse_softbin_anchor": linear_reverse_anchor,
            "action_bias_softbin_anchor": action_bias_anchor,
            "action_signflip_softbin_anchor": action_signflip_anchor,
            "action_global_bias_mse": _float(action_target_mse),
            "action_matched_shuffle_mse": _float(shuffle_mse),
            "latent_time_swap": swapped_terms,
            "latent_uniform_shift_swap": uniform_swap_terms,
            "latent_uniform_shift_swap_softbin_anchor": uniform_swap_anchor,
            "latent_spatial_roll": rolled_terms,
            "latent_global_bias": latent_bias_terms,
            "latent_matched_local": local_terms,
            "latent_bias_softbin_anchor": latent_bias_anchor,
            "latent_global_bias_mse": _float(target_latent_mse),
            "latent_matched_local_mse": _float(local_mse),
            "action_neighbor_samples": int(distances["action"].numel()),
            "latent_neighbor_samples": int(distances["latent"].numel()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="JSON result path; defaults to ARTIFACTS/relational_selfcheck.json")
    args = parser.parse_args()
    job_id, hostname = require_allocation()

    import torch

    source_dir = Path(__file__).resolve().parents[1] / "fastwam-vq-dual-kl"
    sys.path.insert(0, str(source_dir))
    from softbin_loss import pointwise_kl
    from relational_loss import relational_losses, teacher_distances

    artifact_path = args.output
    if artifact_path is None:
        artifact_dir = os.environ.get("ARTIFACTS")
        if not artifact_dir:
            raise RuntimeError("set ARTIFACTS or pass --output for the selfcheck report")
        artifact_path = Path(artifact_dir) / "relational_selfcheck.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_checks(torch, relational_losses, teacher_distances, pointwise_kl, device)
    report = {
        "passed": True,
        "pbs_jobid": job_id,
        "hostname": hostname,
        "device": str(device),
        **result,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("RELATIONAL_SELFCHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
