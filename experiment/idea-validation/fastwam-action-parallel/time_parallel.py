"""Inference-only Picard solvers for Fast-WAM's discrete action Euler path."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Callable

import torch


def picard_windowed(
    denoise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_action: torch.Tensor,
    timesteps: torch.Tensor,
    deltas: torch.Tensor,
    width: int,
    iterations: int,
    *,
    method: str = "prefix",
    arithmetic: str = "native",
    collect_trace: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve unchanged Euler nodes in non-overlapping time windows.

    ``initial_action`` is one ``[32, 7]`` action. ``denoise`` receives a
    contiguous ``[window_width, 32, 7]`` guess batch and matching 1-D times.
    """
    if initial_action.ndim != 2 or tuple(initial_action.shape) != (32, 7):
        raise ValueError(f"initial_action must have shape [32, 7], got {tuple(initial_action.shape)}")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError(f"iterations must be a positive integer, got {iterations!r}")
    if method not in {"prefix", "triangular"}:
        raise ValueError("method must be 'prefix' or 'triangular'")
    if arithmetic not in {"native", "fp32"}:
        raise ValueError("arithmetic must be 'native' or 'fp32'")
    if method == "triangular" and arithmetic != "native":
        raise ValueError("triangular uses native-dtype Euler updates only")

    times = torch.as_tensor(timesteps, device=initial_action.device, dtype=initial_action.dtype)
    step_deltas = torch.as_tensor(deltas, device=initial_action.device, dtype=initial_action.dtype)
    if times.ndim != 1 or step_deltas.ndim != 1 or times.numel() != step_deltas.numel():
        raise ValueError("timesteps and deltas must be 1-D tensors with equal length")
    if times.numel() == 0:
        raise ValueError("at least one Euler node is required")

    node_count = int(times.numel())
    spans = [(start, min(start + width, node_count)) for start in range(0, node_count, width)]
    current = initial_action
    trace: list[dict[str, Any]] = []
    denoise_calls = 0
    scalar_nfes = 0

    for start, stop in spans:
        size = stop - start
        window_left = current
        guesses = torch.stack([window_left] * size, dim=0)
        window_times = times[start:stop]
        window_deltas = step_deltas[start:stop]
        delta_view = window_deltas.reshape(size, *([1] * current.ndim))

        for round_index in range(iterations):
            velocities = denoise(guesses, window_times)
            denoise_calls += 1
            scalar_nfes += size
            if not isinstance(velocities, torch.Tensor) or velocities.shape != guesses.shape:
                shape = getattr(velocities, "shape", None)
                raise ValueError(f"denoise must return shape {tuple(guesses.shape)}, got {shape}")
            if velocities.device != guesses.device or velocities.dtype != guesses.dtype:
                raise ValueError("denoise output must match the action device and dtype")

            if method == "triangular":
                next_nodes = guesses + velocities * delta_view
            elif arithmetic == "fp32":
                increments = velocities.float() * delta_view.float()
                prefix = torch.cumsum(increments, dim=0)
                next_nodes = (window_left.float().unsqueeze(0) + prefix).to(window_left.dtype)
            else:
                increments = velocities * delta_view
                # Make each prefix addition's action-dtype rounding explicit.
                running = torch.zeros_like(window_left)
                prefix_nodes = []
                for increment in increments.unbind(0):
                    running = running + increment
                    prefix_nodes.append(running)
                prefix = torch.stack(prefix_nodes, dim=0)
                next_nodes = window_left.unsqueeze(0) + prefix

            current = next_nodes[-1]
            if collect_trace:
                trace.append(
                    {
                        "window": (start, stop),
                        "iteration": round_index + 1,
                        "right_action": current.detach().clone(),
                    }
                )
            if round_index + 1 < iterations:
                # Keep the left boundary fixed for every sweep of this window.
                guesses = torch.cat((window_left.unsqueeze(0), next_nodes[:-1]), dim=0)

    metadata: dict[str, Any] = {
        "method": method,
        "arithmetic": arithmetic,
        "width": width,
        "iterations": iterations,
        "num_steps": node_count,
        "window_spans": tuple(spans),
        "window_batch_sizes": tuple(stop - start for start, stop in spans),
        "denoise_calls": denoise_calls,
        "scalar_nfes": scalar_nfes,
        "native_scalar_nfes": node_count,
    }
    if collect_trace:
        metadata["trace"] = trace
    return current, metadata


def self_check() -> dict[str, Any]:
    """Small deterministic checks; run only inside an approved compute allocation."""
    job_id = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    hostname = socket.gethostname().split(".", 1)[0].lower()
    if not job_id or not nodefile or "login" in hostname or not Path(nodefile).is_file():
        raise RuntimeError("self_check requires an allocated PBS compute node and PBS nodefile")
    allocated_hosts = {
        line.split(".", 1)[0].strip().lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if hostname not in allocated_hosts:
        raise RuntimeError(f"current host {hostname!r} is absent from PBS_NODEFILE")

    times = torch.linspace(1000, 100, 10, dtype=torch.bfloat16)
    deltas = torch.full((10,), -0.1, dtype=torch.bfloat16)
    action = torch.ones((32, 7), dtype=torch.bfloat16)

    def affine_bf16(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return x * 0.05

    def serial(
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        schedule_times: torch.Tensor,
        schedule_deltas: torch.Tensor,
    ) -> torch.Tensor:
        for index in range(schedule_times.numel()):
            velocity = denoiser(x.unsqueeze(0), schedule_times[index : index + 1])[0]
            delta = schedule_deltas[index].to(device=x.device, dtype=x.dtype)
            x = x + velocity * delta
        return x

    reference = serial(affine_bf16, action, times, deltas)
    one_step, _ = picard_windowed(affine_bf16, action, times[:1], deltas[:1], 1, 1)
    triangular, triangular_meta = picard_windowed(
        affine_bf16, action, times, deltas, 10, 10, method="triangular"
    )
    one_step_reference = action + affine_bf16(action.unsqueeze(0), times[:1])[0] * deltas[0]
    assert torch.equal(one_step, one_step_reference)
    assert torch.equal(triangular, reference)
    assert triangular_meta["scalar_nfes"] == 100

    x32 = torch.ones((32, 7), dtype=torch.float32)
    d32 = torch.full((10,), -0.1, dtype=torch.float32)

    def contractive(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return 0.2 * x

    times32 = times.float()
    exact = serial(contractive, x32, times32, d32)
    prefix_1, _ = picard_windowed(
        contractive, x32, times32, d32, 10, 1, arithmetic="fp32"
    )
    prefix_5, _ = picard_windowed(
        contractive, x32, times32, d32, 10, 5, arithmetic="fp32"
    )
    error_1 = float((prefix_1 - exact).abs().max().item())
    error_5 = float((prefix_5 - exact).abs().max().item())
    assert error_5 < error_1
    return {
        "width_one_matches_euler": True,
        "triangular_full_rounds_matches_euler": True,
        "triangular_scalar_nfes": triangular_meta["scalar_nfes"],
        "prefix_fp32_max_abs_error_r1": error_1,
        "prefix_fp32_max_abs_error_r5": error_5,
    }
