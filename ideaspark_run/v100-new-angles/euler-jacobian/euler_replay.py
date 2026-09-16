"""Pure NumPy replay for the frozen Euler-Jacobian raw contract.

This file deliberately has no CLI, model import, filesystem access, or cluster
entry point.  ``replay`` accepts an ``np.load``-style mapping and returns only
JSON-serializable Python values.  The producer/source identity checks remain in
the outer CPU verifier.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

STATE_COUNT = 6
ARMS = 2
TOKENS = 50
DIM = 32
EPISODES = (105, 52, 84, 66, 7, 8)
DT = np.float32(-0.1)
TIMESTEP = np.float32(0.5)
FD_EPSILON = np.float32(0.002)

def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None

def _scalar(value: Any) -> tuple[float, bool]:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("float32"):
        return float("nan"), False
    return float(array.item()), True


def _empty_rows() -> list[dict[str, Any]]:
    return [{"state_index": i, "episode_id": EPISODES[i], "valid": False} for i in range(STATE_COUNT)]

def _result(checks: dict[str, bool], rows: list[dict[str, Any]], decision: str, **extra: Any) -> dict[str, Any]:
    return {"checks": checks, "rows": rows, "decision": decision, **extra}

def replay(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Replay engineering gates and the frozen local Jacobian criteria.

    The function never imputes missing values.  Incomplete producer output is
    reported as ``inconclusive_budget``; malformed or numerically invalid output
    is reported as ``implementation_inconclusive`` before any science gate.
    """

    required = (
        "noise", "x_fp", "v", "jac", "directions", "fd_values",
        "tail_probe_v", "tail_delta", "fp_noop_velocity", "fp_path", "fp_path_v",
        "completed", "episode_ids", "timestep", "dt", "fd_epsilon",
    )
    try:
        available = set(raw.keys())
    except AttributeError:
        available = set()
    missing = [key for key in required if key not in available]
    checks: dict[str, bool] = {"required_keys": not missing}
    if missing:
        checks["array_shapes"] = False
        checks["array_dtypes"] = False
        checks["array_finite"] = False
        checks["completed_all"] = False
        checks["engineering_pass"] = False
        return _result(checks, _empty_rows(), "implementation_inconclusive", missing_keys=missing)

    expected = {
        "noise": ((1, TOKENS, DIM), np.dtype("float32")),
        "x_fp": ((STATE_COUNT, TOKENS, DIM), np.dtype("float32")),
        "v": ((STATE_COUNT, ARMS, TOKENS, DIM), np.dtype("float32")),
        "jac": ((STATE_COUNT, ARMS, DIM, TOKENS, DIM), np.dtype("float32")),
        "directions": ((2, DIM), np.dtype("float32")),
        "fd_values": ((STATE_COUNT, ARMS, 2, 2, DIM), np.dtype("float32")),
        "tail_probe_v": ((STATE_COUNT, ARMS, DIM), np.dtype("float32")),
        "tail_delta": ((TOKENS - 1, DIM), np.dtype("float32")),
        "fp_noop_velocity": ((STATE_COUNT, TOKENS, DIM), np.dtype("float32")),
        "fp_path": ((STATE_COUNT, 6, TOKENS, DIM), np.dtype("float32")),
        "fp_path_v": ((STATE_COUNT, 5, TOKENS, DIM), np.dtype("float32")),
        "completed": ((STATE_COUNT, ARMS), np.dtype("bool")),
        "episode_ids": ((STATE_COUNT,), np.dtype("int64")),
    }
    arrays: dict[str, np.ndarray] = {}
    shape_ok = True
    dtype_ok = True
    finite_ok = True
    for name, (shape, dtype) in expected.items():
        try:
            value = np.asarray(raw[name])
        except Exception:
            shape_ok = dtype_ok = finite_ok = False
            continue
        arrays[name] = value
        shape_ok = shape_ok and value.shape == shape
        dtype_ok = dtype_ok and value.dtype == dtype
        if np.issubdtype(value.dtype, np.inexact):
            finite_ok = finite_ok and bool(np.isfinite(value).all())
    checks["array_shapes"] = bool(shape_ok)
    checks["array_dtypes"] = bool(dtype_ok)
    checks["array_finite"] = bool(finite_ok)

    timestep, timestep_ok = _scalar(raw["timestep"])
    dt, dt_ok = _scalar(raw["dt"])
    fd_epsilon, epsilon_ok = _scalar(raw["fd_epsilon"])
    checks["scalar_parameters"] = bool(
        timestep_ok and dt_ok and epsilon_ok
        and timestep == float(TIMESTEP) and dt == float(DT) and fd_epsilon == float(FD_EPSILON)
    )
    checks["episode_order"] = bool(
        "episode_ids" in arrays and arrays["episode_ids"].shape == (STATE_COUNT,)
        and np.array_equal(arrays["episode_ids"], np.asarray(EPISODES, dtype=np.int64))
    )
    completed = arrays.get("completed")
    completed_all = bool(completed is not None and completed.shape == (STATE_COUNT, ARMS) and completed.all())
    checks["completed_all"] = completed_all

    if not (shape_ok and dtype_ok and checks["scalar_parameters"] and checks["episode_order"]):
        checks["engineering_pass"] = False
        return _result(checks, _empty_rows(), "implementation_inconclusive", missing_keys=[])
    if not completed_all:
        checks["engineering_pass"] = False
        return _result(checks, _empty_rows(), "inconclusive_budget")
    if not finite_ok:
        checks["engineering_pass"] = False
        return _result(checks, _empty_rows(), "implementation_inconclusive")

    noise = arrays["noise"]
    x_fp = arrays["x_fp"]
    velocity = arrays["v"]
    jacobian = arrays["jac"]
    directions = arrays["directions"]
    fd_values = arrays["fd_values"]
    tail_probe = arrays["tail_probe_v"]
    fp_noop = arrays["fp_noop_velocity"]
    fp_path = arrays["fp_path"]
    fp_path_v = arrays["fp_path_v"]

    direction_scale = np.float64(np.sqrt(DIM))
    checks["directions_unit_rademacher"] = bool(
        np.allclose(np.linalg.norm(directions.astype(np.float64), axis=1), 1.0, atol=1e-6, rtol=0.0)
        and np.allclose(np.abs(directions.astype(np.float64)) * direction_scale, 1.0, atol=2e-6, rtol=0.0)
    )
    checks["initial_path_matches_noise"] = bool(np.allclose(fp_path[:, 0], noise[0], atol=1e-6, rtol=0.0))

    reconstructed = np.array(fp_path[:, 0], dtype=np.float32, copy=True)
    path_reconstruction = True
    for step in range(5):
        reconstructed = np.add(
            reconstructed,
            np.multiply(np.float32(dt), fp_path_v[:, step], dtype=np.float32),
            dtype=np.float32,
        )
        path_reconstruction = path_reconstruction and bool(
            np.allclose(reconstructed, fp_path[:, step + 1], atol=1e-6, rtol=0.0)
        )
    checks["fp_path_reconstruction"] = bool(path_reconstruction)
    checks["x_fp_matches_path_last"] = bool(np.allclose(x_fp, fp_path[:, -1], atol=1e-6, rtol=0.0))
    checks["future_block_absmax"] = bool(np.max(np.abs(jacobian[:, :, :, 1:, :])) <= 1e-7)
    checks["tail_first_velocity_noop"] = bool(np.allclose(tail_probe, velocity[:, :, 0, :], atol=1e-6, rtol=0.0))
    checks["fp_grad_noop"] = bool(np.allclose(fp_noop, velocity[:, 0], atol=1e-6, rtol=0.0))

    j_block = jacobian[:, :, :, 0, :].astype(np.float64)
    direction64 = directions.astype(np.float64)
    j_direction = np.einsum("saij,dj->sadi", j_block, direction64)
    fd_plus = fd_values[:, :, :, 0, :].astype(np.float64)
    fd_minus = fd_values[:, :, :, 1, :].astype(np.float64)
    fd_estimate = (fd_plus - fd_minus) / (2.0 * np.float64(fd_epsilon))
    fd_error = np.linalg.norm(fd_estimate - j_direction, axis=-1)
    fd_norm = np.linalg.norm(j_direction, axis=-1)
    fd_limit = 1e-3 + 0.02 * fd_norm
    checks["fd_linearization"] = bool(np.all(fd_error <= fd_limit))

    rows: list[dict[str, Any]] = []
    fp_eligible_count = 0
    flag_count = 0
    for state in range(STATE_COUNT):
        metrics: list[dict[str, Any]] = []
        for arm in range(ARMS):
            matrix = np.eye(DIM, dtype=np.float64) - np.float64(0.1) * j_block[state, arm]
            singular = np.linalg.svd(matrix, compute_uv=False)
            sign, logabsdet = np.linalg.slogdet(matrix)
            sigma_min = float(singular[-1])
            sigma_max = float(singular[0])
            ratio = sigma_min / sigma_max if sigma_max > 0.0 else 0.0
            metrics.append({
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "ratio": ratio,
                "sign": float(sign),
                "logabsdet": _finite_or_none(float(logabsdet)),
            })
        fp = metrics[0]
        q = metrics[1]
        fp_eligible = bool(fp["sign"] > 0.0 and fp["sigma_min"] >= 0.05 and fp["ratio"] >= 0.01)
        q_conditioning = bool(
            q["sigma_min"] <= 0.005
            and q["sigma_min"] <= 0.1 * fp["sigma_min"]
            and q["ratio"] <= 0.1 * fp["ratio"]
        )
        q_reversal = bool(
            q["sign"] < 0.0 and q["sigma_min"] >= 0.005 and q["ratio"] >= 1e-4
        )
        flag = bool(fp_eligible and (q_conditioning or q_reversal))
        fp_eligible_count += int(fp_eligible)
        flag_count += int(flag)
        rows.append({
            "state_index": state,
            "episode_id": int(EPISODES[state]),
            "valid": True,
            "fp": fp,
            "q": q,
            "sigma_min_fp": fp["sigma_min"], "sigma_max_fp": fp["sigma_max"], "ratio_fp": fp["ratio"],
            "sign_fp": fp["sign"], "logabsdet_fp": fp["logabsdet"],
            "sigma_min_q": q["sigma_min"], "sigma_max_q": q["sigma_max"], "ratio_q": q["ratio"],
            "sign_q": q["sign"], "logabsdet_q": q["logabsdet"],
            "fp_eligible": fp_eligible,
            "q_conditioning_flag": q_conditioning,
            "q_orientation_reversal_flag": q_reversal,
            "flag": flag,
            "velocityMSE": float(np.mean((velocity[state, 1].astype(np.float64) - velocity[state, 0].astype(np.float64)) ** 2)),
            "fd_error_norms": fd_error[state].tolist(),
            "fd_limits": fd_limit[state].tolist(),
        })

    checks["fp_eligible_minimum"] = bool(fp_eligible_count >= 4)
    checks["flag_minimum"] = bool(flag_count >= 3)
    engineering_keys = (
        "directions_unit_rademacher", "initial_path_matches_noise", "fp_path_reconstruction",
        "x_fp_matches_path_last", "future_block_absmax", "tail_first_velocity_noop",
        "fp_grad_noop", "fd_linearization",
    )
    checks["engineering_pass"] = bool(all(checks[key] for key in engineering_keys))
    if not checks["engineering_pass"]:
        decision = "implementation_inconclusive"
    elif fp_eligible_count < 4:
        decision = "inconclusive_binding"
    elif flag_count >= 3:
        decision = "scope_limited_preliminary_go"
    else:
        decision = "mechanism_no_go"
    return _result(
        checks,
        rows,
        decision,
        counts={"fp_eligible": fp_eligible_count, "flagged": flag_count},
        thresholds={
            "fd_error": "<=1e-3 + 0.02*||Jd||", "future_absmax": 1e-7,
            "fp_sigma_min": 0.05, "fp_ratio": 0.01, "fp_eligible": 4, "q_flagged": 3,
        },
    )
