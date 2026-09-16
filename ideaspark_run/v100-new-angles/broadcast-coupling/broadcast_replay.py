"""Pure NumPy replay for the frozen broadcast-coupling raw contract."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

STATE_COUNT = 6
ARMS = 2
PATCHES = 196
TAIL = 20
CONCAT = 404
VISUAL = 384
DRAW_COUNT = 3
TRAJECTORY_IDS = (1035, 1534, 1158, 203, 1837, 1095)
VALID_INDICES = (124, 125, 126, 127, 128, 129)
DT = np.float32(-0.1)


def _empty_rows() -> list[dict[str, Any]]:
    return [{"state_index": i, "trajectory_id": TRAJECTORY_IDS[i], "valid": False} for i in range(STATE_COUNT)]


def _result(checks: dict[str, bool], rows: list[dict[str, Any]], decision: str, **extra: Any) -> dict[str, Any]:
    return {"checks": checks, "rows": rows, "decision": decision, **extra}


def _mse(value: np.ndarray) -> float:
    return float(np.mean(np.asarray(value, dtype=np.float64) ** 2))


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def replay(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the producer and replay the fixed SR/multiset/MSE algebra."""
    required = (
        "initial_z", "base_quant", "uniform", "scale", "rtn_vector", "codes",
        "seen_tail", "pred_visual", "fp_visual", "fp_copy_visual", "rtn_visual",
        "offset", "valid_indices", "trajectory_ids", "completed",
    )
    try:
        available = set(raw.keys())
    except AttributeError:
        available = set()
    missing = [name for name in required if name not in available]
    checks: dict[str, bool] = {"required_keys": not missing}
    if missing:
        checks.update({"array_shapes": False, "array_dtypes": False, "array_finite": False, "completed_all": False, "engineering_pass": False})
        return _result(checks, _empty_rows(), "implementation_inconclusive", missing_keys=missing)

    expected = {
        "initial_z": ((STATE_COUNT, PATCHES, CONCAT), np.dtype("float32")),
        "base_quant": ((STATE_COUNT, DRAW_COUNT, TAIL), np.dtype("float32")),
        "uniform": ((DRAW_COUNT, TAIL), np.dtype("float32")),
        "scale": ((STATE_COUNT,), np.dtype("float32")),
        "rtn_vector": ((STATE_COUNT, TAIL), np.dtype("float32")),
        "codes": ((STATE_COUNT, DRAW_COUNT, TAIL), np.dtype("int8")),
        "seen_tail": ((STATE_COUNT, ARMS, DRAW_COUNT, PATCHES, TAIL), np.dtype("float32")),
        "pred_visual": ((STATE_COUNT, ARMS, DRAW_COUNT, PATCHES, VISUAL), np.dtype("float32")),
        "fp_visual": ((STATE_COUNT, PATCHES, VISUAL), np.dtype("float32")),
        "fp_copy_visual": ((STATE_COUNT, PATCHES, VISUAL), np.dtype("float32")),
        "rtn_visual": ((STATE_COUNT, ARMS, PATCHES, VISUAL), np.dtype("float32")),
        "offset": ((PATCHES,), np.dtype("int64")),
        "valid_indices": ((STATE_COUNT,), np.dtype("int64")),
        "trajectory_ids": ((STATE_COUNT,), np.dtype("int64")),
        "completed": ((STATE_COUNT,), np.dtype("bool")),
    }
    arrays: dict[str, np.ndarray] = {}
    shape_ok = dtype_ok = finite_ok = True
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
    checks["valid_indices"] = bool(np.array_equal(arrays["valid_indices"], np.asarray(VALID_INDICES, dtype=np.int64)))
    checks["trajectory_ids"] = bool(np.array_equal(arrays["trajectory_ids"], np.asarray(TRAJECTORY_IDS, dtype=np.int64)))
    checks["offset_range"] = bool(np.all((arrays["offset"] >= 0) & (arrays["offset"] < DRAW_COUNT)))
    checks["uniform_range"] = bool(np.all((arrays["uniform"] >= 0) & (arrays["uniform"] < 1)))
    completed_all = bool(arrays["completed"].all())
    checks["completed_all"] = completed_all
    if not (shape_ok and dtype_ok and checks["valid_indices"] and checks["trajectory_ids"]):
        checks["engineering_pass"] = False
        return _result(checks, _empty_rows(), "implementation_inconclusive", missing_keys=[])
    if not completed_all:
        checks["engineering_pass"] = False
        return _result(checks, _empty_rows(), "inconclusive_budget")
    if not finite_ok or not checks["offset_range"] or not checks["uniform_range"]:
        checks["engineering_pass"] = False
        return _result(checks, _empty_rows(), "implementation_inconclusive")

    initial = arrays["initial_z"]
    tail = initial[:, :, VISUAL:]
    base = tail[:, 0, :]
    scale_expected = np.max(np.abs(base), axis=1).astype(np.float32) / np.float32(7.0)
    scale_expected = np.where(scale_expected == 0, np.float32(1.0), scale_expected).astype(np.float32)
    normalized = (base / scale_expected[:, None]).astype(np.float32)
    lower = np.floor(normalized).astype(np.float32)
    fraction = (normalized - lower).astype(np.float32)
    uniform = arrays["uniform"]
    expected_codes = np.clip(lower[:, None, :] + (uniform[None, :, :] < fraction[:, None, :]).astype(np.float32), -7, 7).astype(np.int8)
    expected_base = (expected_codes.astype(np.float32) * scale_expected[:, None, None]).astype(np.float32)
    expected_rtn = np.clip(np.round(normalized), -7, 7).astype(np.float32) * scale_expected[:, None]
    checks["tail_replication"] = bool(np.array_equal(tail, tail[:, :1, :].repeat(PATCHES, axis=1)))
    checks["scale_cpu_float32"] = bool(np.array_equal(arrays["scale"], scale_expected))
    checks["codes_cpu_float32"] = bool(np.array_equal(arrays["codes"], expected_codes))
    checks["base_quant_readback"] = bool(np.array_equal(arrays["base_quant"], expected_base))
    checks["rtn_vector_readback"] = bool(np.array_equal(arrays["rtn_vector"], expected_rtn))
    checks["code_grid"] = bool(np.all((arrays["codes"] >= -7) & (arrays["codes"] <= 7)))

    expected_seen = np.empty_like(arrays["seen_tail"])
    for state in range(STATE_COUNT):
        for draw in range(DRAW_COUNT):
            expected_seen[state, 0, draw] = arrays["base_quant"][state, draw]
            for patch in range(PATCHES):
                expected_seen[state, 1, draw, patch] = arrays["base_quant"][state, (draw + int(arrays["offset"][patch])) % DRAW_COUNT]
    checks["actual_seen_tail"] = bool(np.array_equal(arrays["seen_tail"], expected_seen))
    checks["patch_draw_multiset"] = bool(np.array_equal(np.sort(arrays["seen_tail"][:, 0], axis=1), np.sort(arrays["seen_tail"][:, 1], axis=1)))
    checks["fp_copy"] = bool(np.allclose(arrays["fp_copy_visual"], arrays["fp_visual"], atol=1e-6, rtol=0.0))
    checks["rtn_before_after_exact"] = bool(np.array_equal(arrays["rtn_visual"][:, 0], arrays["rtn_visual"][:, 1]))

    tail64 = tail.astype(np.float64)
    seen64 = arrays["seen_tail"].astype(np.float64)
    input_shared = np.mean((seen64[:, 0] - tail64[:, None, :, :]) ** 2, axis=(1, 2, 3), dtype=np.float64)
    input_spatial = np.mean((seen64[:, 1] - tail64[:, None, :, :]) ** 2, axis=(1, 2, 3), dtype=np.float64)
    input_diff = np.abs(input_shared - input_spatial)
    checks["input_mse_matched"] = bool(np.all(input_diff <= 1e-12 + 1e-7 * np.abs(input_shared)))

    rows: list[dict[str, Any]] = []
    binding_count = 0
    positive_count = 0
    for state in range(STATE_COUNT):
        shared_prediction = arrays["pred_visual"][state, 0].astype(np.float64)
        spatial_prediction = arrays["pred_visual"][state, 1].astype(np.float64)
        fp = arrays["fp_visual"][state].astype(np.float64)
        shared_error = _mse(shared_prediction - fp)
        spatial_error = _mse(spatial_prediction - fp)
        gain = (shared_error - spatial_error) / shared_error if shared_error > 0 else None
        draws_differ = bool(np.unique(arrays["base_quant"][state], axis=0).shape[0] >= 2)
        arms_differ = bool(np.any(arrays["seen_tail"][state, 0] != arrays["seen_tail"][state, 1]))
        binding = bool(shared_error > 1e-12 and draws_differ and arms_differ)
        positive = bool(binding and gain is not None and gain >= 0.10)
        binding_count += int(binding)
        positive_count += int(positive)
        rows.append({
            "state_index": state, "trajectory_id": int(TRAJECTORY_IDS[state]), "valid": True,
            "errors": {"shared_visual_mse": shared_error, "spatial_visual_mse": spatial_error},
            "input_errors": {"shared_mse": float(input_shared[state]), "spatial_mse": float(input_spatial[state]), "abs_diff": float(input_diff[state])},
            "gain": _finite(gain) if gain is not None else None, "binding": binding,
            "draws_differ": draws_differ, "arms_differ": arms_differ,
        })
    engineering_keys = (
        "array_shapes", "array_dtypes", "array_finite", "valid_indices", "trajectory_ids", "offset_range",
        "uniform_range", "tail_replication", "scale_cpu_float32", "codes_cpu_float32", "base_quant_readback",
        "rtn_vector_readback", "code_grid", "actual_seen_tail", "patch_draw_multiset", "fp_copy",
        "rtn_before_after_exact", "input_mse_matched",
    )
    checks["engineering_pass"] = bool(all(checks[key] for key in engineering_keys))
    if not checks["engineering_pass"]:
        decision = "implementation_inconclusive"
    elif binding_count < STATE_COUNT:
        decision = "inconclusive_binding"
    elif positive_count >= 5:
        decision = "scope_limited_preliminary_go"
    else:
        decision = "mechanism_no_go"
    return _result(
        checks,
        rows,
        decision,
        binding_count=binding_count,
        positive_count=positive_count,
        thresholds={"shared_mse_binding": 1e-12, "gain": 0.10, "positive_count": 5, "input_mse": "abs<=1e-12+rel1e-7"},
    )
