"""Pure-Python/NumPy contracts for the Wall broadcast-coupling experiment.

The model runner is intentionally kept separate from this file.  This module
contains the deterministic quantizers, paired metrics, and summary algebra so
that it can be tested without importing torch or loading a checkpoint.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "broadcast-coupling-wall-exp1-v1"
STATE_SCHEMA = "broadcast-coupling-wall-exp1-state-v1"
SUMMARY_SCHEMA = "broadcast-coupling-wall-exp1-summary-v1"
ARMS = ("FP32", "RTN", "shared_stochastic", "iid_patch_stochastic", "balanced_broadcast")
HORIZONS = (1, 5)
VISUAL_DIM = 384
PATCHES = 196
TAIL_DIM = 20
QMAX = np.float32(7.0)
# The registry in ``idea/v100-new-angles/INDEX.zh.md`` marks the
# complete valid-local prefix 0..129 as viewed, reserved, locked, or used by a
# prior gate.  Keep the exclusion conservative: an episode may be admitted to
# a run only when a later manifest proves that it is outside this registry.
KNOWN_USED_EPISODES = tuple(range(130))
PRIOR_EPISODES = KNOWN_USED_EPISODES
DEFAULT_EPISODES = (130, 131, 132, 133, 134, 135)
BROADCAST_DRAW_SEEDS = (2501, 2502, 2503)
OFFSET_SEED = 2601


def derive_seed(base_seed: int, episode_index: int, arm: str, step: int = 0) -> int:
    """Derive a stable 31-bit seed without Python's process-randomized hash."""
    payload = f"{int(base_seed)}|{int(episode_index)}|{arm}|{int(step)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def validate_episode_indices(indices: Iterable[int], excluded: Iterable[int] = PRIOR_EPISODES) -> tuple[int, ...]:
    values = tuple(int(value) for value in indices)
    if not values:
        raise ValueError("at least one fresh episode index is required")
    if len(set(values)) != len(values):
        raise ValueError("episode indices must be unique")
    blocked = set(int(value) for value in excluded)
    overlap = sorted(set(values).intersection(blocked))
    if overlap:
        raise ValueError(f"episode indices overlap excluded prior episodes: {overlap}")
    if any(value < 0 for value in values):
        raise ValueError("episode indices must be non-negative")
    return values


def requested_horizon(step: int, horizons: Sequence[int] = HORIZONS) -> bool:
    """Return whether an autoregressive step is a requested reporting horizon."""
    return int(step) in tuple(int(value) for value in horizons)


def _uniforms(rng: np.random.Generator, arm: str, patches: int, dims: int) -> np.ndarray:
    """Return one draw of stochastic uniforms for the non-paired baseline."""
    if arm == "shared_stochastic":
        return np.broadcast_to(rng.random(dims, dtype=np.float32), (patches, dims)).copy()
    if arm == "iid_patch_stochastic":
        return rng.random((patches, dims), dtype=np.float32)
    raise ValueError(f"stochastic uniforms are not defined for arm {arm!r}")


def _quantize_with_uniforms(values: np.ndarray, uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.float32(np.max(np.abs(values)))
    scale = np.float32(maximum / QMAX) if maximum > 0 else np.float32(1.0)
    normalized = (values / scale).astype(np.float32, copy=False)
    lower = np.floor(normalized).astype(np.float32)
    fraction = (normalized - lower).astype(np.float32)
    codes = np.clip(lower + (uniforms < fraction).astype(np.float32), -7, 7).astype(np.int8)
    return (codes.astype(np.float32) * scale).astype(np.float32), codes


def quantize_tail(tail: np.ndarray, arm: str, rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantize a ``[patch, 20]`` conditioning tail with FP32 operators.

    The scalar scale is computed from the unquantized input and is shared by
    all arms for this call.  Stochastic arms use signed A4 levels ``[-7, 7]``
    and return dequantized float32 values; this is fake quantization, not a
    packed/native low-bit implementation.
    """
    values = np.asarray(tail, dtype=np.float32)
    if values.shape != (PATCHES, TAIL_DIM):
        raise ValueError(f"tail must have shape {(PATCHES, TAIL_DIM)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("tail contains non-finite values")
    if arm not in ARMS:
        raise ValueError(f"unsupported arm {arm!r}")
    maximum = np.float32(np.max(np.abs(values)))
    scale = np.float32(maximum / QMAX) if maximum > 0 else np.float32(1.0)
    normalized = (values / scale).astype(np.float32, copy=False)
    if arm == "FP32":
        output = values.copy()
        scheme = "identity"
        codes = None
    elif arm == "RTN":
        codes = np.clip(np.rint(normalized), -7, 7).astype(np.int8)
        output = (codes.astype(np.float32) * scale).astype(np.float32)
        scheme = "symmetric_A4_RTN"
    else:
        if rng is None:
            raise ValueError(f"{arm} requires an explicit RNG")
        uniforms = _uniforms(rng, arm, PATCHES, TAIL_DIM)
        output, codes = _quantize_with_uniforms(values, uniforms)
        scheme = "symmetric_A4_stochastic_shared" if arm == "shared_stochastic" else "symmetric_A4_stochastic_iid_patch"
    audit = {
        "arm": arm,
        "scheme": scheme,
        "scale": float(scale),
        "input_mse": float(np.mean((output.astype(np.float64) - values.astype(np.float64)) ** 2)),
        "input_max_abs": float(np.max(np.abs(output.astype(np.float64) - values.astype(np.float64)))),
        "finite": bool(np.isfinite(output).all()),
    }
    if codes is not None:
        audit["code_min"] = int(codes.min())
        audit["code_max"] = int(codes.max())
    return output, audit


def prepare_initial_treatments(tail: np.ndarray, base_seed: int, episode_index: int) -> dict[str, dict[str, Any]]:
    """Create deterministic initial-only interventions for one paired state.

    ``shared_stochastic`` and ``balanced_broadcast`` each use the same three
    base stochastic quantized vectors.  Balanced assigns those vectors by a
    fixed patch offset, so every patch has exactly the same three-value
    multiset as Shared and the total input MSE is identical.  The iid arm is a
    deliberately non-matched negative control.

    Returned arrays have shape ``[n_draws, PATCHES, TAIL_DIM]`` and are plain
    NumPy values suitable for a predictor-boundary intervention.  The caller
    must not use them to overwrite later predicted proprio/action coordinates.
    """
    values = np.asarray(tail, dtype=np.float32)
    if values.shape != (PATCHES, TAIL_DIM):
        raise ValueError(f"tail must have shape {(PATCHES, TAIL_DIM)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("tail contains non-finite values")
    maximum = np.float32(np.max(np.abs(values)))
    scale = np.float32(maximum / QMAX) if maximum > 0 else np.float32(1.0)

    fp = values[None, ...].copy()
    rtn_codes = np.clip(np.rint(values / scale), -7, 7).astype(np.int8)
    rtn = (rtn_codes.astype(np.float32) * scale).astype(np.float32)[None, ...]

    shared_draws = []
    for draw_seed in BROADCAST_DRAW_SEEDS:
        rng = np.random.default_rng(derive_seed(base_seed + draw_seed, episode_index, "shared_stochastic"))
        uniforms = np.broadcast_to(rng.random(TAIL_DIM, dtype=np.float32), (PATCHES, TAIL_DIM)).copy()
        quantized, _ = _quantize_with_uniforms(values, uniforms)
        shared_draws.append(quantized)
    shared = np.stack(shared_draws, axis=0)

    offset_rng = np.random.default_rng(OFFSET_SEED)
    offsets = (np.arange(PATCHES, dtype=np.int64) % len(BROADCAST_DRAW_SEEDS))[offset_rng.permutation(PATCHES)]
    balanced = np.stack(
        [shared[(draw + offsets) % len(BROADCAST_DRAW_SEEDS), np.arange(PATCHES)] for draw in range(len(BROADCAST_DRAW_SEEDS))],
        axis=0,
    )
    iid_draws = []
    for draw in range(len(BROADCAST_DRAW_SEEDS)):
        rng = np.random.default_rng(derive_seed(base_seed, episode_index, "iid_patch_stochastic", draw))
        quantized, _ = _quantize_with_uniforms(values, rng.random((PATCHES, TAIL_DIM), dtype=np.float32))
        iid_draws.append(quantized)
    iid = np.stack(iid_draws, axis=0)

    shared_mse = float(np.mean((shared.astype(np.float64) - values.astype(np.float64)) ** 2))
    balanced_mse = float(np.mean((balanced.astype(np.float64) - values.astype(np.float64)) ** 2))
    if not np.array_equal(np.sort(shared, axis=0), np.sort(balanced, axis=0)):
        raise RuntimeError("balanced broadcast failed the shared per-coordinate multiset gate")
    if not math.isclose(shared_mse, balanced_mse, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("balanced broadcast failed the matched total input-MSE gate")
    common = {
        "scale": float(scale),
        "draw_seeds": list(BROADCAST_DRAW_SEEDS),
        "offset_seed": OFFSET_SEED,
        "offset_histogram": [int(np.count_nonzero(offsets == value)) for value in range(len(BROADCAST_DRAW_SEEDS))],
        "intervention_scope": "initial_exact_broadcast_tail_only",
    }
    return {
        "FP32": {"values": fp, "audit": {**common, "arm": "FP32", "n_draws": 1, "input_mse_mean": 0.0}},
        "RTN": {"values": rtn, "audit": {**common, "arm": "RTN", "n_draws": 1, "input_mse_mean": float(np.mean((rtn.astype(np.float64) - values.astype(np.float64)) ** 2))}},
        "shared_stochastic": {"values": shared, "audit": {**common, "arm": "shared_stochastic", "n_draws": 3, "input_mse_mean": shared_mse, "matched_reference": "balanced_broadcast"}},
        "iid_patch_stochastic": {"values": iid, "audit": {**common, "arm": "iid_patch_stochastic", "n_draws": 3, "input_mse_mean": float(np.mean((iid.astype(np.float64) - values.astype(np.float64)) ** 2)), "matched_reference": None}},
        "balanced_broadcast": {"values": balanced, "audit": {**common, "arm": "balanced_broadcast", "n_draws": 3, "input_mse_mean": balanced_mse, "matched_reference": "shared_stochastic", "shared_balanced_mse_equal": True, "shared_balanced_multiset_equal": True}},
    }


def mse(predicted: np.ndarray, target: np.ndarray) -> float:
    lhs = np.asarray(predicted, dtype=np.float64)
    rhs = np.asarray(target, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise ValueError(f"MSE shape mismatch: {lhs.shape} != {rhs.shape}")
    value = float(np.mean((lhs - rhs) ** 2))
    if not math.isfinite(value):
        raise ValueError("MSE is non-finite")
    return value


def make_state_record(
    *,
    episode_index: int,
    trajectory_id: str,
    current_frame: int,
    future_frame: int,
    horizon: int,
    arm_outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one paired state/horizon row from already measured arm metrics."""
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon {horizon}")
    if set(arm_outputs) != set(ARMS):
        raise ValueError("all five arms are required for every paired row")
    fp_real = float(arm_outputs["FP32"]["real_future_feature_mse"])
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        values = dict(arm_outputs[arm])
        real = float(values["real_future_feature_mse"])
        reference = float(values["fp_reference_mse"])
        if not (math.isfinite(real) and math.isfinite(reference)):
            raise ValueError("arm metrics must be finite")
        values["real_future_feature_mse"] = real
        values["fp_reference_mse"] = reference
        values["delta_real_vs_fp32"] = real - fp_real
        arms[arm] = values
    return {
        "schema": STATE_SCHEMA,
        "state_id": f"WallDataset.valid[{int(episode_index)}]@frame[{int(current_frame)}]",
        "episode_index": int(episode_index),
        "trajectory_id": str(trajectory_id),
        "current_frame": int(current_frame),
        "future_frame": int(future_frame),
        "horizon": int(horizon),
        "arms": arms,
    }


def summarize_records(records: Sequence[Mapping[str, Any]], episodes: Sequence[int]) -> dict[str, Any]:
    expected = len(tuple(episodes)) * len(HORIZONS)
    if len(records) != expected:
        raise ValueError(f"expected {expected} paired rows, got {len(records)}")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (str(record["state_id"]), int(record["horizon"]))
        if key in seen:
            raise ValueError(f"duplicate paired row {key}")
        seen.add(key)
        for arm in ARMS:
            if arm not in record["arms"]:
                raise ValueError(f"missing arm {arm} in {key}")
            grouped[(arm, int(record["horizon"]))].append(record)
    expected_keys = {(arm, horizon) for arm in ARMS for horizon in HORIZONS}
    if set(grouped) != expected_keys:
        raise ValueError("summary groups do not cover every arm/horizon")
    rows = []
    for arm in ARMS:
        for horizon in HORIZONS:
            values = [float(item["arms"][arm]["real_future_feature_mse"]) for item in grouped[(arm, horizon)]]
            refs = [float(item["arms"][arm]["fp_reference_mse"]) for item in grouped[(arm, horizon)]]
            deltas = [float(item["arms"][arm]["delta_real_vs_fp32"]) for item in grouped[(arm, horizon)]]
            if not all(math.isfinite(x) for x in values + refs + deltas):
                raise ValueError(f"non-finite summary values for {arm}/{horizon}")
            rows.append({
                "arm": arm,
                "horizon": horizon,
                "n_states": len(values),
                "mean_real_future_feature_mse": float(np.mean(values)),
                "mean_fp_reference_mse": float(np.mean(refs)),
                "mean_delta_real_vs_fp32": float(np.mean(deltas)),
                "state_ids": [str(item["state_id"]) for item in grouped[(arm, horizon)]],
            })
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "episodes": [int(value) for value in episodes],
        "horizons": list(HORIZONS),
        "arms": list(ARMS),
        "rows": rows,
        "execution": "fake_activation_quantization_fp32_operators",
        "native_low_bit_claim": False,
    }
