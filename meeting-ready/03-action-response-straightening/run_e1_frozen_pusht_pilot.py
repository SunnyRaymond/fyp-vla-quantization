#!/usr/bin/env python3
"""Measure local action response on real proposals from the official LeWM CEM."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
MEETING_READY = HERE.parent
OFFICIAL_RUNNER = (
    MEETING_READY
    / "02-horizon-weighted-recurrent-student"
    / "lewm-transfer"
    / "official-pusht-cem"
    / "run_official_pusht_cem.py"
)
OFFICIAL_FREEZE = OFFICIAL_RUNNER.with_name("FREEZE.json")
SEEDS = (4101, 4102)
ROUNDS = (1, 15, 30)
DEFAULT_RADII = (0.05, 0.10, 0.20)  # flattened Euclidean L2 units
SCHEMA = "lewm-pusht.action-response-straightening.e1-pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--official-runner", type=Path, default=OFFICIAL_RUNNER)
    parser.add_argument("--freeze", type=Path, default=OFFICIAL_FREEZE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    parser.add_argument("--max-residual-per-radius", type=int, default=16)
    parser.add_argument("--fd-directions", type=int, default=4)
    return parser.parse_args()


def require_compute_allocation() -> dict[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    if not job_id:
        raise RuntimeError("PBS_JOBID is required before loading LeWM or running PushT")
    host = platform.node().lower()
    short_host = host.split(".", 1)[0]
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    nodefile_value = os.environ.get("PBS_NODEFILE", "").strip()
    if not nodefile_value or not Path(nodefile_value).is_file():
        raise RuntimeError("PBS_NODEFILE must exist for allocation validation")
    nodes = {
        line.strip().lower()
        for line in Path(nodefile_value).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    short_nodes = {node.split(".", 1)[0] for node in nodes}
    if not nodes or (host not in nodes and short_host not in short_nodes):
        raise RuntimeError(f"current host {host} is not a member of PBS_NODEFILE")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is required for this GPU pilot")
    return {
        "pbs_job_id": job_id,
        "hostname": host,
        "pbs_nodefile": str(Path(nodefile_value).resolve()),
        "cuda_visible_devices": visible,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_official_runner(path: Path):
    spec = importlib.util.spec_from_file_location("official_pusht_cem_runner", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official CEM runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_summary(value: Any) -> dict[str, Any] | None:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        tensor = value.detach()
        result: dict[str, Any] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        if tensor.numel():
            result["min"] = float(tensor.min().item())
            result["max"] = float(tensor.max().item())
            if tensor.numel() <= 50:
                result["values"] = tensor.cpu().tolist()
        return result
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        result = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if array.size:
            result["min"] = float(array.min())
            result["max"] = float(array.max())
            if array.size <= 50:
                result["values"] = array.tolist()
        return result
    return None


def box_summary(space: Any) -> dict[str, Any] | None:
    if space is None:
        return None
    low = tensor_summary(getattr(space, "low", None))
    high = tensor_summary(getattr(space, "high", None))
    shape = getattr(space, "shape", None)
    if low is None and high is None and shape is None:
        return None
    return {
        "class": f"{type(space).__module__}.{type(space).__qualname__}",
        "shape": list(shape) if shape is not None else None,
        "low": low,
        "high": high,
    }


def solver_domain_summary(solver: Any, env_action_space: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name, value in vars(solver).items():
        if re.search(r"action|bound|low|high|clip|scale|mean|var", name, re.I):
            summary = tensor_summary(value)
            if summary is not None:
                attrs[name] = summary
            elif hasattr(value, "low") or hasattr(value, "high"):
                attrs[name] = box_summary(value)
    source_lines: list[str] = []
    source_file = None
    source_error = None
    try:
        source_file = inspect.getsourcefile(type(solver))
        source = "\n".join(
            inspect.getsource(method)
            for method in (getattr(type(solver), "configure", None), getattr(type(solver), "solve", None))
            if method is not None
        )
        source_lines = [
            line.strip()
            for line in source.splitlines()
            if re.search(r"clip|clamp|bound|action_space|low|high|mean", line, re.I)
        ][:24]
    except (OSError, TypeError) as exc:
        source_error = str(exc)
    return {
        "environment_action_space": box_summary(env_action_space),
        "solver_attributes": attrs,
        "solver_class": f"{type(solver).__module__}.{type(solver).__qualname__}",
        "solver_source_file": str(Path(source_file).resolve()) if source_file else None,
        "solve_source_bound_lines": source_lines,
        "source_inspection_error": source_error,
        "candidate_coordinate_bounds": "observed extrema are descriptive only; bounds come only from explicit solver low/high attributes",
    }


class PilotTrace:
    """Capture only the three real CEM proposal rounds needed by E1."""

    output_key = "official_cem_trace"

    def __init__(self) -> None:
        self.rows: dict[int, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.input_shapes: dict[str, list[int]] | None = None
        self.cem_input_shapes: dict[str, list[int]] | None = None
        self.rounds_seen: list[int] = []

    def reset(self) -> None:
        self.rows.clear()
        self.history.clear()
        self.input_shapes = None
        self.cem_input_shapes = None
        self.rounds_seen.clear()

    def start_batch(self) -> None:
        return None

    def end_solve(self) -> None:
        return None

    def finite_trace(self) -> bool:
        return all(bool(row["finite_flags"].item()) for row in self.history)

    def __call__(self, **kwargs: Any) -> None:
        import torch

        round_id = int(kwargs["step"]) + 1
        self.rounds_seen.append(round_id)
        finite = torch.stack([
            torch.isfinite(kwargs[key]).all() for key in ("costs", "mean", "var")
        ]).all()
        self.history.append({"step": round_id, "finite_flags": finite})
        if self.input_shapes is None:
            self.input_shapes = {
                "candidates": list(kwargs["candidates"].shape),
                "costs": list(kwargs["costs"].shape),
            }
            self.cem_input_shapes = self.input_shapes
        if round_id not in ROUNDS:
            return
        candidates = kwargs["candidates"]
        costs = kwargs["costs"]
        required = ("candidates", "costs", "topk_inds", "prev_mean", "prev_var", "mean", "var")
        with torch.inference_mode(False):
            row = {key: kwargs[key].detach().clone() for key in required}
        if tuple(row["prev_mean"].shape) == (1, 5, 10):
            row["prev_mean"] = row["prev_mean"].unsqueeze(1)
        row["step"] = round_id
        row["prev_mean_candidate0_equal"] = bool(
            torch.equal(row["candidates"][:, :1], row["prev_mean"])
        )
        self.rows[round_id] = row


class CapturingTeacher:
    """Delegate every CEM score to the frozen teacher and retain one context view."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.round_index = 0
        self.rounds: dict[int, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def parameters(self):
        return self.model.parameters()

    def begin_solve(self) -> None:
        self.round_index = 0
        self.rounds.clear()

    def end_solve(self) -> None:
        return None

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Any):
        import torch

        self.round_index += 1
        round_id = self.round_index
        saved_info = None
        if round_id in ROUNDS:
            sample_count = int(action_candidates.shape[1])
            saved_info = {}
            axis_keys: list[str] = []
            with torch.inference_mode(False):
                for key, value in info_dict.items():
                    if torch.is_tensor(value):
                        if value.ndim >= 2 and int(value.shape[1]) == sample_count:
                            saved_info[key] = value[:, :1].detach().clone()
                            axis_keys.append(key)
                        else:
                            saved_info[key] = value.detach().clone()
                    else:
                        saved_info[key] = value
            self.rounds[round_id] = {
                "single_info": saved_info,
                "candidate_axis_keys": axis_keys,
                "action_shape": list(action_candidates.shape),
            }

        costs = self.model.get_cost(info_dict, action_candidates)
        if round_id in ROUNDS:
            with torch.inference_mode(False):
                self.rounds[round_id]["goal_emb"] = (
                    info_dict["goal_emb"].detach().clone()
                    if "goal_emb" in info_dict and torch.is_tensor(info_dict["goal_emb"])
                    else None
                )
        return costs


def expand_context_info(
    single_info: Mapping[str, Any], axis_keys: list[str], samples: int
) -> dict[str, Any]:
    expanded = dict(single_info)
    for key in axis_keys:
        value = expanded[key]
        expanded[key] = value.expand(value.shape[0], samples, *value.shape[2:])
    return expanded


def action_to_flat(action: Any):
    return action.reshape(action.shape[0], action.shape[1], -1)


def percentile(values: Any, qs: tuple[float, ...]) -> dict[str, float]:
    import torch

    flat = values.detach().reshape(-1).float()
    result = torch.quantile(flat, torch.tensor(qs, device=flat.device))
    return {f"p{int(q * 100):02d}": float(v.item()) for q, v in zip(qs, result)}


def synchronized_call(torch: Any, function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, float(time.perf_counter() - started)


def rollout_function(model: Any, info: Mapping[str, Any]):
    import torch

    def function(flat_action):
        actions = flat_action.reshape(1, 1, 5, 10)
        output = model.rollout(dict(info), actions)
        predicted = output["predicted_emb"]
        if predicted.ndim != 4 or tuple(predicted.shape[:2]) != (1, 1):
            raise RuntimeError(f"unexpected LeWM rollout shape: {tuple(predicted.shape)}")
        return predicted.reshape(-1)

    return function


def full_jacobian(torch: Any, function, center):
    """Use math SDPA for AD; the native efficient kernel has no forward AD."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        method = "torch.func.jacfwd_math_sdpa"
        try:
            jacobian = torch.func.jacfwd(function)(center)
            return jacobian, method
        except (AttributeError, NotImplementedError, RuntimeError) as first_error:
            if "out of memory" in str(first_error).lower():
                raise RuntimeError(f"full-rollout Jacobian exhausted GPU memory: {first_error}") from first_error
            try:
                method = "explicit_forward_jvp_columns_math_sdpa"
                eye = torch.eye(center.numel(), dtype=center.dtype, device=center.device)
                columns = []
                for direction in eye:
                    _, tangent = torch.autograd.functional.jvp(
                        function, center, direction, create_graph=False, strict=False
                    )
                    columns.append(tangent)
                return torch.stack(columns, dim=-1), method
            except Exception as second_error:
                raise RuntimeError(
                    "full-rollout action Jacobian failed with math SDPA for both jacfwd and explicit JVP; "
                    f"jacfwd={first_error}; explicit={second_error}"
                ) from second_error


def goal_embedding(model: Any, info: Mapping[str, Any]):
    import torch

    goal = {
        key: value[:, 0]
        for key, value in info.items()
        if torch.is_tensor(value) and value.ndim >= 2
    }
    if "goal" not in goal:
        raise KeyError("captured official CEM context lacks goal pixels")
    goal["pixels"] = goal.pop("goal")
    for key in list(goal):
        if key.startswith("goal_"):
            goal[key[len("goal_") :]] = goal.pop(key)
    goal.pop("action", None)
    return model.encode(goal)["emb"]


def summarize_bounds(solver: Any, center, candidates):
    import numpy as np
    import torch

    possibilities: list[tuple[str, Any, Any]] = []
    objects = [("solver", solver)]
    for name in ("action_space", "action_bounds", "bounds", "action_spec"):
        value = getattr(solver, name, None)
        if value is not None:
            objects.append((f"solver.{name}", value))
    for prefix, obj in objects:
        low = getattr(obj, "low", None)
        high = getattr(obj, "high", None)
        if low is not None and high is not None:
            possibilities.append((prefix, low, high))
    resolved = []
    for name, low, high in possibilities:
        try:
            lo = torch.as_tensor(low, dtype=center.dtype, device=center.device)
            hi = torch.as_tensor(high, dtype=center.dtype, device=center.device)
            if lo.numel() == 2 and center.numel() == 50:
                lo = lo.repeat(25)
                hi = hi.repeat(25)
            lo = torch.broadcast_to(lo, center.shape).reshape(-1)
            hi = torch.broadcast_to(hi, center.shape).reshape(-1)
            if bool(torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo < hi).all()):
                resolved.append((name, lo, hi))
        except (TypeError, ValueError, RuntimeError):
            continue
    candidate_flat = candidates.reshape(1, candidates.shape[1], -1)
    candidate_min = candidate_flat.amin(dim=(0, 1)).reshape(-1)
    candidate_max = candidate_flat.amax(dim=(0, 1)).reshape(-1)
    result = {
        "candidate_observed_min": float(candidate_min.min().item()),
        "candidate_observed_max": float(candidate_max.max().item()),
        "candidate_observed_coordinate_min_max": [
            [float(lo), float(hi)]
            for lo, hi in zip(candidate_min.detach().cpu().tolist(), candidate_max.detach().cpu().tolist())
        ],
        "solver_bounds_candidates": [],
        "selected_solver_bounds": None,
    }
    for name, lo, hi in resolved:
        contains = bool(
            (candidates >= lo.reshape(1, 1, 5, 10)).all()
            and (candidates <= hi.reshape(1, 1, 5, 10)).all()
            and (center >= lo).all()
            and (center <= hi).all()
        )
        row = {
            "source": name,
            "low_min": float(lo.min().item()),
            "low_max": float(lo.max().item()),
            "high_min": float(hi.min().item()),
            "high_max": float(hi.max().item()),
            "contains_all_observed_candidates": contains,
        }
        result["solver_bounds_candidates"].append(row)
        if contains and result["selected_solver_bounds"] is None:
            result["selected_solver_bounds"] = {
                "source": name,
                "low": lo,
                "high": hi,
            }
    selected = result["selected_solver_bounds"]
    if selected is not None:
        result["selected_solver_bounds"] = {
            "source": selected["source"],
            "flat_low": selected["low"].detach().cpu().tolist(),
            "flat_high": selected["high"].detach().cpu().tolist(),
        }
    return result


def finite_difference_check(
    torch: Any,
    function,
    center,
    jacobian,
    anchor_output,
    bounds,
    count: int,
    curvature_radius: float,
):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # The pinned CEM samples unconstrained coordinates even though its
    # environment action_space is Box(-1, 1). Do not borrow that Box as a
    # fictitious solver bound when no clipping is applied to proposals.
    low, high = bounds if bounds is not None else (None, None)
    generator = torch.Generator(device=center.device).manual_seed(94000 + center.numel())
    rows = []
    tried = 0
    for direction_id in range(max(0, count) * 8):
        if len(rows) >= count:
            break
        tried += 1
        direction = torch.randn(center.shape, generator=generator, device=center.device, dtype=center.dtype)
        if bounds is not None:
            headroom = torch.minimum((high - center).clamp_min(0), (center - low).clamp_min(0))
            direction = direction * (headroom >= curvature_radius).to(direction.dtype)
        if not bool((direction.abs() > 1e-9).any()):
            continue
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
        radius_plus = center + curvature_radius * direction
        radius_minus = center - curvature_radius * direction
        if bounds is not None and not bool(((radius_plus >= low) & (radius_plus <= high)
                                            & (radius_minus >= low) & (radius_minus <= high)).all()):
            continue
        autodiff_jvp = jacobian @ direction
        derivative_checks = []
        for derivative_step in (5e-2, 1e-2):
            plus = center + derivative_step * direction
            minus = center - derivative_step * direction
            if bounds is not None and not bool(((plus >= low) & (plus <= high)
                                                & (minus >= low) & (minus <= high)).all()):
                continue
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                f_plus = function(plus)
                f_minus = function(minus)
            finite_difference = (f_plus - f_minus) / (2.0 * derivative_step)
            abs_error = torch.linalg.vector_norm(finite_difference - autodiff_jvp)
            relative_error = abs_error / torch.linalg.vector_norm(autodiff_jvp).clamp_min(1e-12)
            derivative_checks.append({
                "step": float(derivative_step),
                "finite_difference_jvp_l2": float(torch.linalg.vector_norm(finite_difference).item()),
                "absolute_error_l2": float(abs_error.item()),
                "relative_error": float(relative_error.item()),
            })
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            f_radius_plus = function(radius_plus)
            f_radius_minus = function(radius_minus)
        second_difference = f_radius_plus - 2.0 * anchor_output + f_radius_minus
        second_by_horizon = second_difference.reshape(6, 192)
        rows.append({
            "direction_index": direction_id,
            "derivative_checks": derivative_checks,
            "central_curvature_l2_radius": float(curvature_radius),
            "plus_minus_within_solver_bounds": True if bounds is not None else None,
            "autodiff_jvp_l2": float(torch.linalg.vector_norm(autodiff_jvp).item()),
            "nonzero_autodiff_response": bool(torch.linalg.vector_norm(autodiff_jvp).item() > 1e-10),
            "central_second_difference_l2": float(torch.linalg.vector_norm(second_difference).item()),
            "central_curvature_l2_per_radius_squared": float(
                torch.linalg.vector_norm(second_difference).item() / (curvature_radius ** 2)
            ),
            "central_curvature_l2_by_state_index_per_radius_squared": [
                float(value) / (curvature_radius ** 2)
                for value in torch.linalg.vector_norm(second_by_horizon, dim=-1).cpu().tolist()
            ],
        })
    return {
        "status": "CHECKED" if rows else "NO_FEASIBLE_SYMMETRIC_DIRECTIONS",
        "attempted_directions": tried,
        "required_direction_count": count,
        "all_directions_same_curvature_radius": True,
        "solver_bound_basis": "explicit_solver_bounds" if bounds is not None else "unbounded_solver_proposal_coordinates",
        "finite_difference_forward_kernel": "math_sdpa_matches_jacfwd",
        "directions": rows,
    }


def select_residual_indices(distances, radius: float, limit: int):
    import torch

    indices = torch.nonzero((distances <= radius) & (distances > 1e-9), as_tuple=False).flatten()
    if indices.numel() <= limit:
        return indices
    ordered = indices[torch.argsort(distances[indices])]
    positions = torch.linspace(0, ordered.numel() - 1, steps=limit, device=ordered.device).round().long()
    return ordered[positions]


def process_stage(
    torch: Any,
    official: Any,
    solver: Any,
    trace_row: Mapping[str, Any],
    capture_row: Mapping[str, Any],
    radii: list[float],
    max_residual_per_radius: int,
    fd_directions: int,
) -> dict[str, Any]:
    with torch.inference_mode(False):
        candidates = trace_row["candidates"].detach().to(device="cuda", dtype=torch.float32).clone()
        exact_costs = trace_row["costs"].detach().to(device="cuda", dtype=torch.float32).clone()
        reference = trace_row["prev_mean"].detach().to(device="cuda", dtype=candidates.dtype).clone()
    if tuple(candidates.shape) != (1, 300, 5, 10):
        raise RuntimeError(f"expected true official CEM proposals [1,300,5,10], got {tuple(candidates.shape)}")
    if tuple(reference.shape) != (1, 1, 5, 10):
        raise RuntimeError(f"expected pre-sample CEM mean [1,1,5,10], got {tuple(reference.shape)}")
    if not trace_row["prev_mean_candidate0_equal"]:
        raise RuntimeError("official callback prev_mean is not the current round's candidate-zero proposal")
    if tuple(capture_row["action_shape"]) != tuple(candidates.shape):
        raise RuntimeError("captured official get_cost action shape disagrees with CEM callback")
    info = capture_row["single_info"]
    axis_keys = capture_row["candidate_axis_keys"]
    center = reference.reshape(-1).detach()
    if center.numel() != 50:
        raise RuntimeError(f"expected D=50 optimized action, got D={center.numel()}")
    function = rollout_function(official, info)

    anchor_output, forward_s = synchronized_call(torch, lambda: function(center))
    (jacobian, jacobian_method), jacobian_s = synchronized_call(
        torch, lambda: full_jacobian(torch, function, center)
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        math_anchor, math_anchor_s = synchronized_call(torch, lambda: function(center))
    math_anchor_drift_l2 = float(torch.linalg.vector_norm(math_anchor - anchor_output).item())
    del math_anchor
    if tuple(jacobian.shape) != (anchor_output.numel(), center.numel()):
        raise RuntimeError(f"unexpected full-rollout Jacobian shape: {tuple(jacobian.shape)}")
    anchor_output = anchor_output.detach()
    jacobian = jacobian.detach()
    horizon_count = int(anchor_output.numel() // 192)
    if horizon_count != 6:
        raise RuntimeError(f"expected initial plus five future latents (6x192), got {anchor_output.numel()}")
    anchor_trajectory = anchor_output.reshape(6, 192)
    terminal_anchor = anchor_trajectory[-1]
    flat_candidates = action_to_flat(candidates).squeeze(0)
    flat_centered = reference.expand_as(candidates).reshape(300, 50)
    deltas = flat_candidates - flat_centered
    distances = torch.linalg.vector_norm(deltas, dim=-1)
    bounds_summary = summarize_bounds(solver, center, candidates)

    tangent_started = time.perf_counter()
    tangent_flat = anchor_output.unsqueeze(0) + deltas @ jacobian.T
    tangent_pred = tangent_flat.reshape(1, 300, 6, 192)
    goal_emb = capture_row.get("goal_emb")
    if goal_emb is None:
        goal_emb = goal_embedding(official, info)
    goal_emb = goal_emb.to(device="cuda")
    tangent_costs, surrogate_s = synchronized_call(
        torch,
        lambda: official.criterion({"predicted_emb": tangent_pred, "goal_emb": goal_emb}),
    )
    torch.cuda.synchronize()
    full_surrogate_s = float(time.perf_counter() - tangent_started)
    tangent_costs = tangent_costs.reshape(1, 300)

    expanded_info = expand_context_info(info, axis_keys, 300)
    exact_recheck, exact_score_s = synchronized_call(
        torch,
        lambda: official.get_cost(expanded_info, candidates),
    )
    exact_recheck = exact_recheck.reshape(1, 300)
    exact_recheck_max_abs = float((exact_recheck - exact_costs).abs().max().item())
    exact_recheck_equal = bool(torch.allclose(exact_recheck, exact_costs, rtol=1e-5, atol=1e-5))
    if not exact_recheck_equal:
        raise RuntimeError(
            f"replayed official criterion did not reproduce actual CEM costs; max_abs={exact_recheck_max_abs}"
        )

    exact_order = torch.topk(exact_costs, k=30, dim=1, largest=False).indices
    surrogate_order = torch.topk(tangent_costs, k=30, dim=1, largest=False).indices
    overlap = len(set(exact_order[0].tolist()) & set(surrogate_order[0].tolist()))
    sorted_exact = torch.sort(exact_costs, dim=1).values
    sorted_surrogate = torch.sort(tangent_costs, dim=1).values
    cost_abs = (tangent_costs - exact_costs).abs()
    cost_rel_denom = exact_costs.abs().mean().clamp_min(1e-12)
    candidate_summary = {
        "l2_distance_quantiles": percentile(distances, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)),
        "candidate_count": int(candidates.shape[1]),
        "reference_candidate_included": bool((distances[0] < 1e-8).item()),
    }
    per_radius = []
    for radius in radii:
        within = distances <= float(radius)
        nonreference = within & (distances > 1e-9)
        indices = select_residual_indices(distances, float(radius), max_residual_per_radius)
        row: dict[str, Any] = {
            "radius_l2": float(radius),
            "candidate_coverage_count": int(within.sum().item()),
            "candidate_coverage_fraction": float(within.float().mean().item()),
            "nonreference_coverage_count": int(nonreference.sum().item()),
            "residual_sample_count": int(indices.numel()),
            "residual_sample_selection": "exclude center; sort in-radius proposals by L2 distance and choose evenly spaced ranks",
            "residual_sample_distances_l2": [float(value) for value in distances[indices].detach().cpu().tolist()],
        }
        if indices.numel():
            chosen_actions = candidates[:, indices].reshape(1, int(indices.numel()), 5, 10)
            exact_output, exact_forward_s = synchronized_call(
                torch,
                lambda: official.rollout(info, chosen_actions)["predicted_emb"],
            )
            exact_flat = exact_output.reshape(indices.numel(), 6 * 192)
            selected_delta = deltas[indices]
            selected_tangent = anchor_output.unsqueeze(0) + selected_delta @ jacobian.T
            exact_trajectory = exact_flat.reshape(-1, 6, 192)
            tangent_trajectory = selected_tangent.reshape(-1, 6, 192)
            residual = exact_trajectory - tangent_trajectory
            terminal_residual = residual[:, -1]
            terminal_change = exact_trajectory[:, -1] - terminal_anchor.unsqueeze(0)
            residual_norm = torch.linalg.vector_norm(terminal_residual, dim=-1)
            change_norm = torch.linalg.vector_norm(terminal_change, dim=-1)
            row.update({
                "terminal_residual_l2_mean": float(residual_norm.mean().item()),
                "terminal_residual_l2_p90": float(torch.quantile(residual_norm, 0.90).item()),
                "terminal_residual_rms_mean": float(torch.sqrt(terminal_residual.square().mean(dim=-1).mean()).item()),
                "terminal_relative_residual_mean": float((residual_norm / change_norm.clamp_min(1e-12)).mean().item()),
                "terminal_relative_residual_p90": float(torch.quantile(residual_norm / change_norm.clamp_min(1e-12), 0.90).item()),
                "action_induced_terminal_change_l2_mean": float(change_norm.mean().item()),
                "relative_denominator_floor": 1e-12,
                "all_horizon_residual_l2_mean_by_state_index": [
                    float(value) for value in torch.linalg.vector_norm(residual, dim=-1).mean(dim=0).cpu().tolist()
                ],
                "exact_sample_forward_wall_s": exact_forward_s,
                "sample_candidate_indices": [int(value) for value in indices.detach().cpu().tolist()],
            })
        else:
            row.update({
                "terminal_residual_l2_mean": None,
                "terminal_relative_residual_mean": None,
                "action_induced_terminal_change_l2_mean": None,
                "note": "no non-reference real CEM candidate fell inside this radius",
            })
        per_radius.append(row)

    reference_bounds = bounds_summary.get("selected_solver_bounds")
    fd_result = finite_difference_check(
        torch,
        function,
        center,
        jacobian,
        anchor_output,
        (torch.tensor(reference_bounds["flat_low"], device=center.device, dtype=center.dtype),
         torch.tensor(reference_bounds["flat_high"], device=center.device, dtype=center.dtype))
        if reference_bounds is not None
        else None,
        fd_directions,
        min(radii),
    )
    for direction in fd_result["directions"]:
        direction["validated"] = bool(
            direction.get("nonzero_autodiff_response", False)
            and any(check["relative_error"] <= 0.05 for check in direction["derivative_checks"])
        )
    fd_result["status"] = (
        "CHECKED_PASS"
        if len(fd_result["directions"]) >= fd_directions
        and all(direction["validated"] for direction in fd_result["directions"])
        else "INVALID_JACOBIAN"
    )
    fd_result["relative_error_tolerance"] = 0.05
    fd_result["requires_nonzero_response"] = True
    cost_gap_exact = float((sorted_exact[0, 30] - sorted_exact[0, 29]).abs().item())
    cost_gap_tangent = float((sorted_surrogate[0, 30] - sorted_surrogate[0, 29]).abs().item())
    return {
        "cem_round_1_indexed": int(trace_row["step"]),
        "reference": {
            "source": "official callback prev_mean; runtime-verified equal to candidate 0 before scoring",
            "candidate_zero_equal": bool(trace_row["prev_mean_candidate0_equal"]),
            "shape": list(reference.shape),
            "center_min": float(center.min().item()),
            "center_max": float(center.max().item()),
            "pre_sample_mean_flat": [float(value) for value in center.detach().cpu().tolist()],
            "pre_sample_variance_flat": [
                float(value)
                for value in trace_row["prev_var"].reshape(-1).detach().cpu().tolist()
            ],
            "candidate_zero_flat": [
                float(value)
                for value in candidates[0, 0].reshape(-1).detach().cpu().tolist()
            ],
        },
        "optimized_action": {
            "shape": list(candidates.shape),
            "flattened_dimension": int(center.numel()),
            "coordinate_units": "official solver candidate coordinates; no extra normalization applied by this runner",
            "bounds_and_clipping": bounds_summary,
        },
        "candidate_distances": candidate_summary,
        "taylor_radius_results": per_radius,
        "full_candidate_cost_comparison": {
            "exact_cost_source": "actual cost tensor scored inside pinned official CEMSolver.solve",
            "exact_replay_max_abs_error": exact_recheck_max_abs,
            "exact_replay_allclose": exact_recheck_equal,
            "cost_mae": float(cost_abs.mean().item()),
            "cost_rmse": float(torch.sqrt(cost_abs.square().mean()).item()),
            "cost_max_abs_error": float(cost_abs.max().item()),
            "cost_relative_mae_mean_abs_exact_denominator": float((cost_abs.mean() / cost_rel_denom).item()),
            "top30_overlap_count": int(overlap),
            "top30_overlap_fraction": overlap / 30.0,
            "exact_top30_boundary_gap_cost_31_minus_30": cost_gap_exact,
            "linearized_top30_boundary_gap_cost_31_minus_30": cost_gap_tangent,
        },
        "timing": {
            "reference_full_rollout_forward_s": forward_s,
            "full_rollout_jacobian_s": jacobian_s,
            "math_sdpa_anchor_forward_s_diagnostic": math_anchor_s,
            "jacobian_method": jacobian_method,
            "full_batch_surrogate_s": full_surrogate_s,
            "surrogate_criterion_only_s": surrogate_s,
            "full_batch_exact_official_get_cost_s": exact_score_s,
        },
        "autodiff_jvp_finite_difference_check": fd_result,
        "math_vs_native_anchor_l2": math_anchor_drift_l2,
    }


def run(args: argparse.Namespace, allocation: Mapping[str, str]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated GPU job")
    if not args.radii or any(not math.isfinite(radius) or radius <= 0 for radius in args.radii):
        raise ValueError("all --radii values must be finite positive flattened L2 radii")
    if args.max_residual_per_radius < 1:
        raise ValueError("--max-residual-per-radius must be at least 1")
    if args.fd_directions < 2:
        raise ValueError("--fd-directions must be at least 2 for the action-response pilot")
    output = args.output.resolve()
    summary_path = output / "e1_pilot_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite pilot summary: {summary_path}")
    output.mkdir(parents=True, exist_ok=True)

    helper = load_official_runner(args.official_runner)
    freeze = read_json(args.freeze.resolve())
    helper.validate_freeze(freeze)
    schedule, old, reference, _ = helper.load_modules(args.control_root.resolve(), args.lewm_root.resolve())
    schedule.validate_interface(reference, args.probe.resolve())
    import stable_worldmodel as swm
    import jepa

    if args.stablewm_root.resolve() not in Path(swm.__file__).resolve().parents:
        raise RuntimeError(f"stable_worldmodel imported outside staged pinned source: {swm.__file__}")
    if args.lewm_root.resolve() not in Path(jepa.__file__).resolve().parents:
        raise RuntimeError(f"LeWM imported outside staged source: {jepa.__file__}")
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.interpolate_pos_encoding = True
    official.requires_grad_(False)
    official.eval()
    if any(parameter.requires_grad for parameter in official.parameters()):
        raise RuntimeError("teacher parameters must be frozen; action path remains differentiable")

    context_rows = []
    for context_index, seed in enumerate(SEEDS):
        world = helper.make_world(args.stablewm_home.resolve())
        trace = PilotTrace()
        capture = CapturingTeacher(official)
        solve_records: list[dict[str, Any]] = []
        solver, policy, route = helper.make_solver_and_policy(
            world,
            capture,
            reference,
            None,
            "teacher_only",
            [trace],
            solver_seed=4101000 + context_index,
            instrumented_solve_records=solve_records,
            capture_prepared=True,
        )
        try:
            reset_action_space_seed = helper.seed_reset_action_space(world, seed)
            world.reset(seed=seed)
            initial_state_goal = helper.current_env_state(world)
            env_action_space = getattr(world.envs, "action_space", None)
            returned_action = policy.get_action(world.infos)
            torch.cuda.synchronize()
            if trace.rounds_seen != list(range(1, 31)):
                raise RuntimeError(f"official CEM callback did not show exactly 30 rounds: {trace.rounds_seen}")
            if len(trace.rows) != len(ROUNDS) or set(trace.rows) != set(ROUNDS):
                raise RuntimeError(f"expected real callback captures for rounds {ROUNDS}, got {sorted(trace.rows)}")
            if set(capture.rounds) != set(ROUNDS):
                raise RuntimeError(f"expected official get_cost contexts for rounds {ROUNDS}, got {sorted(capture.rounds)}")
            if len(solve_records) != 1:
                raise RuntimeError("official policy must complete one unmodified 30-round CEM solve")
            domain = solver_domain_summary(solver, env_action_space)
            stages = []
            for round_id in ROUNDS:
                trace_row = trace.rows[round_id]
                if not bool(trace_row["prev_mean_candidate0_equal"]):
                    raise RuntimeError(f"round {round_id}: callback prev_mean does not match official candidate 0")
                capture_row = capture.rounds[round_id]
                stage = process_stage(
                    torch,
                    official,
                    solver,
                    trace_row,
                    capture_row,
                    list(args.radii),
                    args.max_residual_per_radius,
                    args.fd_directions,
                )
                stages.append(stage)
            context_rows.append({
                "context_index": context_index,
                "reset_seed": seed,
                "reset_action_space_seed": reset_action_space_seed,
                "initial_state_goal": initial_state_goal,
                "official_input_shapes": trace.input_shapes,
                "official_solver": {
                    "class": f"{type(solver).__module__}.{type(solver).__qualname__}",
                    "round_count": len(trace.rounds_seen),
                    "captured_rounds": list(ROUNDS),
                    "solver_seed": solve_records[0].get("solver_seed"),
                    "candidate_shapes": solve_records[0].get("cem_input_shapes"),
                    "full_solve_wall_s_instrumentation_only": solve_records[0].get("wall_s"),
                    "finite_trace": solve_records[0].get("finite_trace"),
                },
                "raw_environment_and_solver_domain": domain,
                "policy_first_action_shape": list(returned_action.shape) if hasattr(returned_action, "shape") else None,
                "stages": stages,
            })
        finally:
            world.close()
            del policy, solver, route, world
            torch.cuda.empty_cache()

    stage_rows = [stage for context in context_rows for stage in context["stages"]]
    jacobian_valid = bool(stage_rows) and all(
        stage["autodiff_jvp_finite_difference_check"]["status"] == "CHECKED_PASS"
        for stage in stage_rows
    )
    summary = {
        "schema": SCHEMA,
        "status": "PILOT_COMPLETED_NOT_CONFIRMATORY",
        "jacobian_validation": "PASS_FOR_TAYLOR_INTERPRETATION" if jacobian_valid else "INVALID_JACOBIAN_TAYLOR_METRICS_NOT_INTERPRETABLE",
        "reset_seeds": list(SEEDS),
        "reserved_development_seeds_not_used": list(range(4201, 4213)),
        "final_test_seeds_used": [],
        "proposal_source": "real candidates captured inside pinned official CEMSolver.solve; no random-bank replacement",
        "cem_settings": {"num_samples": 300, "iterations": 30, "topk": 30, "selected_rounds_1_indexed": list(ROUNDS)},
        "reference_radii": {"values": list(args.radii), "metric": "flattened L2 in observed official solver action coordinates"},
        "residual_sample_rule": "per radius, exclude candidate zero, sort in-radius real CEM proposals by L2 distance and select evenly spaced ranks; deterministic, no sampling RNG",
        "runtime": {
            **allocation,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "official_runner": str(args.official_runner.resolve()),
            "freeze": str(args.freeze.resolve()),
            "lewm_root": str(args.lewm_root.resolve()),
            "stablewm_root": str(args.stablewm_root.resolve()),
            "stablewm_home": str(args.stablewm_home.resolve()),
        },
        "contexts": context_rows,
        "claim_boundary": "Two-context interface and locality pilot only; no GO/NO-GO claim about training, planner replacement, speedup, or closed-loop task success.",
    }
    write_json(summary_path, summary)
    return summary


def main() -> int:
    args = parse_args()
    allocation = require_compute_allocation()
    result = run(args, allocation)
    print(json.dumps({
        "status": result["status"],
        "summary": str((args.output.resolve() / "e1_pilot_summary.json")),
        "contexts": len(result["contexts"]),
        "context_stage_rows": sum(len(row["stages"]) for row in result["contexts"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
