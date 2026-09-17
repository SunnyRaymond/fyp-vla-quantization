"""Bounded predictor-only W4 stochastic-rounding ensemble screen.

The screen compares a two-member independent stochastic-rounding ensemble with
a two-member antithetic ensemble on one shared pool per fresh Wall episode.
The first member is shared by both ensembles, so the four stored score maps
are ``[shared, independent_b, shared, antithetic_b]``.  Each ensemble still
uses exactly two score maps.  Weights are dequantized and evaluated by the
ordinary FP32 model operators; this is a fake-quantization screen and makes no
native low-bit deployment or speed claim.

This file is intended to run from a CCDS SLURM compute allocation.  Its first
workload action is the existing ``allocation_guard.require_allocation`` call;
model loading and target/raw-array I/O happen only after that guard succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "antithetic-rounding-ensemble-screen-v1"
TARGET_SCHEMA = "antithetic-rounding-fresh-targets-v1"
RAW_SCHEMA = "antithetic-rounding-raw-v1"
HORIZON = 5
ACTION_DIM = 10
ACTION_FLAT_DIM = HORIZON * ACTION_DIM
NUM_SAMPLES = 300
TOPK = 30
BITS = 4
QMAX = 7
EPISODE_COUNT = 6
DATASET_INDICES = tuple(range(96, 102))
EXCLUDED_DATASET_INDICES = tuple(range(0, 96))
ENV_SEED_NAMESPACE = 930000
CANDIDATE_SEED_NAMESPACE = 940000
ROUNDING_SEEDS = (1301, 1302, 1303)
NMSE_EPS = 1e-12
GATE_TOLERANCE = 1e-12
PRIMARY_IMPROVEMENT_THRESHOLD = 0.05
UNIFORM_GRID_BITS = 24
UNIFORM_GRID_SIZE = 1 << UNIFORM_GRID_BITS
EXPECTED_CHECKPOINT_SHA256 = "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b"
EXPECTED_SMOKE_RUNNER_SHA256 = "de5c5eb19f4b26614e71f9e2af36db21e9fd5bcc65188f3191772552bc750af9"
EXPECTED_SCREEN_RUNNER_SHA256 = "51c2463a92a3bf84eabae735eeaa9771a7202add0b2227fe06c7fe1f2bd69d19"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Any) -> Dict[str, Any]:
    resolved = Path(str(path)).resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def _digest_arrays(*arrays: Any, metadata: Mapping[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    if metadata is not None:
        digest.update(json.dumps(metadata, sort_keys=True, default=_json_default).encode("utf-8"))
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _load_helpers() -> Tuple[Any, Any]:
    """Load the verified smoke/screen helpers without assuming cwd."""
    try:
        import smoke_runner as smoke
    except ModuleNotFoundError:
        smoke_path = Path(__file__).with_name("smoke_runner.py")
        spec = importlib.util.spec_from_file_location("ensemble_smoke_runner", smoke_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load smoke_runner.py from {smoke_path}")
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
    try:
        import screen_runner as screen
    except ModuleNotFoundError:
        screen_path = Path(__file__).with_name("screen_runner.py")
        spec = importlib.util.spec_from_file_location("ensemble_screen_runner", screen_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load screen_runner.py from {screen_path}")
        screen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(screen)
    return smoke, screen


def _assert_helper_identity(smoke: Any, screen: Any) -> Dict[str, str]:
    """Bind this screen to the already reviewed smoke/screen helper bytes."""
    actual = {
        "smoke_runner_sha256": _sha256_file(Path(str(smoke.__file__)).resolve()),
        "screen_runner_sha256": _sha256_file(Path(str(screen.__file__)).resolve()),
    }
    expected = {
        "smoke_runner_sha256": EXPECTED_SMOKE_RUNNER_SHA256,
        "screen_runner_sha256": EXPECTED_SCREEN_RUNNER_SHA256,
    }
    for name, value in expected.items():
        if actual.get(name) != value:
            raise RuntimeError(
                f"{name} does not match the frozen reviewed helper: "
                f"got {actual.get(name)!r}, expected {value!r}"
            )
    return {key: str(value) for key, value in actual.items()}


def _target_manifest_path(targets_dir: Path) -> Path:
    return targets_dir / "episode_manifest_fresh_930000.json"


def _target_path(targets_dir: Path, index: int) -> Path:
    return targets_dir / "fresh" / f"episode_{index:03d}.pkl"


def _prepare_targets(
    runtime: Mapping[str, Any],
    targets_dir: Path,
    smoke: Any,
    screen: Any,
    runtime_identity: Mapping[str, Any],
    model_structure: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Create or validate the six fixed fresh targets under one writer."""
    targets_dir = targets_dir.resolve()
    targets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _target_manifest_path(targets_dir)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _validate_targets(manifest, targets_dir, runtime, smoke, screen, runtime_identity, model_structure)

    lock_path = targets_dir / ".episode_manifest_fresh_930000.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError(f"fresh target generation is already in progress: {lock_path}") from exc
    try:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return _validate_targets(manifest, targets_dir, runtime, smoke, screen, runtime_identity, model_structure)
        episodes: List[Dict[str, Any]] = []
        for local_index, dataset_index in enumerate(DATASET_INDICES):
            env_seed = ENV_SEED_NAMESPACE + local_index
            target, layout = screen._new_target(runtime, dataset_index, env_seed)
            fingerprint = screen._target_fingerprint(target, layout)
            target["target_fingerprint"] = fingerprint
            target_path = _target_path(targets_dir, local_index)
            _atomic_pickle(target_path, target)
            episodes.append({
                "episode_id": f"fresh930000:{local_index:03d}",
                "local_index": local_index,
                "dataset_index": dataset_index,
                "env_seed": env_seed,
                "candidate_seed": CANDIDATE_SEED_NAMESPACE + local_index,
                "layout": layout,
                "target_fingerprint": fingerprint,
                "target_path": str(target_path.relative_to(targets_dir)),
            })
        manifest = {
            "schema": TARGET_SCHEMA,
            "screen_schema": getattr(screen, "SCREEN_SCHEMA", None),
            "split": "fresh_930000",
            "episode_count": EPISODE_COUNT,
            "dataset_indices": list(DATASET_INDICES),
            "dataset_index_strategy": "explicit_96_101; indices_0_95_excluded",
            "excluded_dataset_indices": list(EXCLUDED_DATASET_INDICES),
            "env_seed_namespace": ENV_SEED_NAMESPACE,
            "candidate_seed_namespace": CANDIDATE_SEED_NAMESPACE,
            "normalization_identity": screen._normalization_identity(runtime),
            "checkpoint_identity": dict(runtime_identity),
            "model_structure": dict(model_structure),
            "episodes": episodes,
        }
        _atomic_json(manifest_path, manifest)
        return _validate_targets(manifest, targets_dir, runtime, smoke, screen, runtime_identity, model_structure)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _validate_targets(
    manifest: Mapping[str, Any],
    targets_dir: Path,
    runtime: Mapping[str, Any],
    smoke: Any,
    screen: Any,
    runtime_identity: Mapping[str, Any],
    model_structure: Mapping[str, Any],
) -> Mapping[str, Any]:
    if manifest.get("schema") != TARGET_SCHEMA:
        raise RuntimeError(f"unsupported fresh target schema: {manifest.get('schema')!r}")
    if manifest.get("episode_count") != EPISODE_COUNT:
        raise RuntimeError("fresh target count must be six")
    if list(manifest.get("dataset_indices", [])) != list(DATASET_INDICES):
        raise RuntimeError("fresh target dataset indices must be exactly 96..101")
    if list(manifest.get("excluded_dataset_indices", [])) != list(EXCLUDED_DATASET_INDICES):
        raise RuntimeError("fresh target manifest must exclude historical/locked indices 0..95")
    if manifest.get("env_seed_namespace") != ENV_SEED_NAMESPACE:
        raise RuntimeError("fresh target environment seed namespace mismatch")
    if manifest.get("candidate_seed_namespace") != CANDIDATE_SEED_NAMESPACE:
        raise RuntimeError("fresh target candidate seed namespace mismatch")
    if manifest.get("normalization_identity") != screen._normalization_identity(runtime):
        raise RuntimeError("fresh target normalization identity mismatch")
    if manifest.get("model_structure") != dict(model_structure):
        raise RuntimeError("fresh target model structure identity mismatch")
    current_identity = dict(runtime_identity)
    recorded_identity = manifest.get("checkpoint_identity", {})
    for key in ("recorded_epoch", "checkpoint_size_bytes", "checkpoint_sha256", "source_commit", "dinov2_source_commit", "dtype", "decoder", "execution", "torch"):
        if recorded_identity.get(key) != current_identity.get(key):
            raise RuntimeError(f"fresh target checkpoint identity mismatch at {key!r}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != EPISODE_COUNT:
        raise RuntimeError("fresh target manifest episodes must contain six records")
    loaded: List[Dict[str, Any]] = []
    seen_fingerprints = set()
    for local_index, entry in enumerate(episodes):
        expected_id = f"fresh930000:{local_index:03d}"
        if entry.get("episode_id") != expected_id or int(entry.get("local_index", -1)) != local_index:
            raise RuntimeError(f"fresh target episode id mismatch at {local_index}")
        if int(entry.get("dataset_index", -1)) != DATASET_INDICES[local_index]:
            raise RuntimeError(f"fresh target dataset index mismatch at {local_index}")
        if int(entry.get("env_seed", -1)) != ENV_SEED_NAMESPACE + local_index:
            raise RuntimeError(f"fresh target environment seed mismatch at {local_index}")
        if int(entry.get("candidate_seed", -1)) != CANDIDATE_SEED_NAMESPACE + local_index:
            raise RuntimeError(f"fresh target candidate seed mismatch at {local_index}")
        target_path = targets_dir / str(entry.get("target_path", ""))
        if not target_path.is_file():
            raise RuntimeError(f"fresh target file missing: {target_path}")
        with target_path.open("rb") as stream:
            target = pickle.load(stream)
        if target.get("goal_H") != HORIZON:
            raise RuntimeError(f"fresh target horizon mismatch at {local_index}")
        actual = screen._target_fingerprint(target, entry["layout"])
        if actual != entry.get("target_fingerprint") or target.get("target_fingerprint") != actual:
            raise RuntimeError(f"fresh target fingerprint mismatch at {local_index}")
        if actual in seen_fingerprints:
            raise RuntimeError(f"duplicate fresh target fingerprint at {local_index}")
        seen_fingerprints.add(actual)
        loaded.append({**entry, "target_path": str(target_path.resolve()), "target": target})
    return {**dict(manifest), "episodes": loaded}


def _load_allocation_guard() -> Any:
    try:
        from allocation_guard import require_allocation

        return require_allocation
    except ModuleNotFoundError:
        guard_path = Path(__file__).with_name("allocation_guard.py")
        spec = importlib.util.spec_from_file_location("ensemble_allocation_guard", guard_path)
        if spec is None or spec.loader is None:
            raise ImportError("verified allocation_guard.py is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.require_allocation


def _snapshot_equal(model: Any, snapshot: Mapping[str, Any], smoke: Any) -> None:
    for key, expected in snapshot.items():
        family, index, relative = key.split(".", 2)
        actual = smoke._module_for_path(model, family, int(index), relative).weight.detach()
        if not __import__("torch").equal(actual, expected):
            raise RuntimeError(f"weight snapshot restore mismatch at {key}")


def _assert_encoder_untouched(model: Any, snapshot: Mapping[str, Any], smoke: Any) -> None:
    for key, expected in snapshot.items():
        if not key.startswith("encoder."):
            continue
        family, index, relative = key.split(".", 2)
        actual = smoke._module_for_path(model, family, int(index), relative).weight.detach()
        if not __import__("torch").equal(actual, expected):
            raise RuntimeError(f"predictor-only quantizer changed encoder weight at {key}")


def _predictor_linears(model: Any, groups: Sequence[Mapping[str, Any]], smoke: Any) -> List[Dict[str, Any]]:
    import torch
    import torch.nn as nn

    result: List[Dict[str, Any]] = []
    for group in groups:
        if group["family"] != "predictor":
            continue
        for linear in group["linear"]:
            module = smoke._module_for_path(model, group["family"], group["index"], linear["relative"])
            if not isinstance(module, nn.Linear) or module.weight.ndim != 2:
                raise RuntimeError(f"predictor path is not a 2-D Linear: {linear['path']}")
            if module.weight.dtype != torch.float32:
                raise RuntimeError(f"predictor weights must be float32 for this screen: {linear['path']}")
            result.append({
                "group_id": group["group_id"],
                "family": group["family"],
                "index": int(group["index"]),
                "relative": linear["relative"],
                "path": linear["path"],
                "shape": list(module.weight.shape),
            })
    if len(result) != 24:
        raise RuntimeError(f"expected exactly 24 predictor Linear modules, got {len(result)}")
    return result


def _hash_update_array(digest: Any, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())


def _quantize_sr(
    model: Any,
    linears: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    smoke: Any,
    rng_seed: int,
    complement: bool,
    member_id: str,
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """Apply exact per-output stochastic rounding and return error vectors."""
    import torch

    rng = np.random.default_rng(int(rng_seed))
    integer_digest = hashlib.sha256()
    scale_digest = hashlib.sha256()
    uniform_digest = hashlib.sha256()
    details: List[Dict[str, Any]] = []
    error_vectors: List[np.ndarray] = []
    total_numel = 0
    error_sum = 0.0
    error_sq_sum = 0.0
    zero_rows_total = 0
    q_min = QMAX
    q_max = -QMAX
    with torch.no_grad():
        for spec in linears:
            key = f"{spec['family']}.{spec['index']}.{spec['relative']}"
            reference = snapshot[key].detach().float()
            module = smoke._module_for_path(model, spec["family"], spec["index"], spec["relative"])
            rows = reference.reshape(reference.shape[0], -1)
            max_abs = rows.abs().amax(dim=1, keepdim=True)
            zero_rows = max_abs == 0
            scale = torch.where(zero_rows, torch.ones_like(max_abs), max_abs / float(QMAX))
            # Midpoint grid makes u and 1-u have exactly the same finite
            # marginal while retaining a reproducible discrete Uniform draw.
            uniform_index = rng.integers(0, UNIFORM_GRID_SIZE, size=tuple(reference.shape), dtype=np.uint32)
            uniform_np = (uniform_index.astype(np.float64) + 0.5) / float(UNIFORM_GRID_SIZE)
            if complement:
                uniform_np = 1.0 - uniform_np
            _hash_update_array(uniform_digest, uniform_np)
            uniform = torch.as_tensor(uniform_np, dtype=torch.float64, device=reference.device).reshape(rows.shape)
            scaled = rows / scale
            scaled64 = scaled.to(dtype=torch.float64)
            lower64 = torch.floor(scaled64)
            fraction64 = scaled64 - lower64
            integers64 = torch.clamp(lower64 + (uniform < fraction64).to(dtype=torch.float64), -QMAX, QMAX)
            integers = integers64.to(dtype=rows.dtype)
            dequant = torch.where(zero_rows, torch.zeros_like(rows), integers * scale)
            module.weight.copy_(dequant.reshape_as(module.weight))

            integer_np = integers.detach().cpu().numpy().astype(np.int8, copy=False)
            scale_np = scale.detach().cpu().numpy().astype(np.float32, copy=False)
            _hash_update_array(integer_digest, integer_np)
            _hash_update_array(scale_digest, scale_np)
            error_np = (dequant - rows).detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            error_vectors.append(error_np.copy())
            total_numel += int(error_np.size)
            error_sum += float(error_np.astype(np.float64).sum())
            error_sq_sum += float(np.square(error_np.astype(np.float64)).sum())
            zero_rows_total += int(zero_rows.sum().item())
            if integer_np.size:
                q_min = min(q_min, int(integer_np.min()))
                q_max = max(q_max, int(integer_np.max()))
            details.append({
                "path": spec["path"],
                "shape": list(spec["shape"]),
                "out_channels": int(spec["shape"][0]),
                "integer_sha256": hashlib.sha256(np.ascontiguousarray(integer_np).tobytes()).hexdigest(),
                "scale_sha256": hashlib.sha256(np.ascontiguousarray(scale_np).tobytes()).hexdigest(),
                "uniform_sha256": hashlib.sha256(np.ascontiguousarray(uniform_np).tobytes()).hexdigest(),
                "integer_min": int(integer_np.min()) if integer_np.size else 0,
                "integer_max": int(integer_np.max()) if integer_np.size else 0,
                "zero_rows": int(zero_rows.sum().item()),
            })
    mean_error = error_sum / total_numel if total_numel else 0.0
    audit = {
        "member_id": member_id,
        "scheme": "symmetric_per_output_channel_stochastic_rounding",
        "bits": BITS,
        "qmax": QMAX,
        "scale_definition": "s=max(abs(row))/7; zero row uses s=1 and dequantizes to zero",
        "integer_definition": "q=clamp(floor(x/s)+1[u < (x/s-floor(x/s))], -7, 7)",
        "uniform_grid": f"midpoints k+0.5 over 2^{UNIFORM_GRID_BITS}, k in [0,2^{UNIFORM_GRID_BITS}); antithetic is exact 1-u on the same grid",
        "rng_seed": int(rng_seed),
        "complement": bool(complement),
        "linear_count": len(details),
        "total_numel": total_numel,
        "integer_min": q_min,
        "integer_max": q_max,
        "zero_rows": zero_rows_total,
        "integer_sha256": integer_digest.hexdigest(),
        "scale_sha256": scale_digest.hexdigest(),
        "uniform_sha256": uniform_digest.hexdigest(),
        "weight_error_mean": mean_error,
        "weight_error_mse": error_sq_sum / total_numel if total_numel else 0.0,
        "linears": details,
    }
    return audit, error_vectors


def _pair_error_metrics(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> Dict[str, float]:
    if len(first) != len(second):
        raise RuntimeError("pair error vector module counts differ")
    n = 0
    sum_a = sum_b = sum_a2 = sum_b2 = sum_ab = diff2 = 0.0
    for values_a, values_b in zip(first, second):
        if values_a.shape != values_b.shape:
            raise RuntimeError("pair error vector shapes differ")
        a = values_a.astype(np.float64, copy=False)
        b = values_b.astype(np.float64, copy=False)
        n += int(a.size)
        sum_a += float(a.sum())
        sum_b += float(b.sum())
        sum_a2 += float(np.square(a).sum())
        sum_b2 += float(np.square(b).sum())
        sum_ab += float((a * b).sum())
        diff2 += float(np.square(a - b).sum())
    if n == 0:
        return {"error_covariance": 0.0, "error_correlation": 0.0, "pair_weight_error_mse": 0.0}
    mean_a = sum_a / n
    mean_b = sum_b / n
    cov = sum_ab / n - mean_a * mean_b
    var_a = max(0.0, sum_a2 / n - mean_a * mean_a)
    var_b = max(0.0, sum_b2 / n - mean_b * mean_b)
    corr = cov / math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0.0
    return {
        "error_covariance": cov,
        "error_correlation": corr,
        "pair_weight_error_mse": diff2 / n,
    }


def _score(
    model: Any,
    preprocessor: Any,
    objective_fn: Any,
    target: Mapping[str, Any],
    candidates: np.ndarray,
    smoke: Any,
) -> np.ndarray:
    import torch

    scores = smoke._score_pool(
        model, preprocessor, objective_fn, target, torch.as_tensor(candidates, dtype=torch.float32)
    )
    result = scores.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    if result.shape != (NUM_SAMPLES,) or not np.isfinite(result).all():
        raise FloatingPointError("non-finite or malformed score map")
    return result.copy()


def _selection_metrics(reference: np.ndarray, selected: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    if reference.shape != (NUM_SAMPLES,) or selected.shape != (NUM_SAMPLES,):
        raise ValueError("score maps must contain exactly 300 candidates")
    reference_order = np.argsort(reference, kind="stable")
    selected_order = np.argsort(selected, kind="stable")
    reference_elite = reference_order[:TOPK]
    selected_elite = selected_order[:TOPK]
    reference_mean = actions[reference_elite].mean(axis=0)
    selected_mean = actions[selected_elite].mean(axis=0)
    reference64 = reference.astype(np.float64, copy=False)
    reference_score_mean = float(reference64[reference_elite].mean())
    selected_fp_score_mean = float(reference64[selected_elite].mean())
    raw_regret = selected_fp_score_mean - reference_score_mean
    if raw_regret < -GATE_TOLERANCE:
        raise RuntimeError(f"FP32 top30 regret is materially negative: {raw_regret}")
    return {
        "elite_mean_action_mse": float(np.mean(np.square(selected_mean - reference_mean))),
        "selected_top30_fp_score_regret": max(0.0, raw_regret),
        "selected_top30_fp_score_regret_raw": raw_regret,
        "score_nmse": float(np.mean(np.square(selected.astype(np.float64) - reference.astype(np.float64))) / (np.mean(np.square(reference.astype(np.float64))) + NMSE_EPS)),
        "selected_top30_fp_score_mean": selected_fp_score_mean,
        "reference_top30_fp_score_mean": reference_score_mean,
        "selected_top30_overlap": float(np.intersect1d(reference_elite, selected_elite).size / TOPK),
    }


def _evaluate_sr_member(
    runtime: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    predictor: Sequence[Mapping[str, Any]],
    smoke: Any,
    preprocessor: Any,
    objective_fn: Any,
    target: Mapping[str, Any],
    candidates: np.ndarray,
    seed: int,
    complement: bool,
    member_id: str,
    started: float,
    max_seconds: float,
    label: str,
) -> Tuple[Dict[str, Any], List[np.ndarray], np.ndarray]:
    smoke._restore_weights(runtime["model"], snapshot)
    audit, errors = _quantize_sr(
        runtime["model"], predictor, snapshot, smoke, seed, complement, member_id
    )
    _assert_encoder_untouched(runtime["model"], snapshot, smoke)
    _check_deadline(started, max_seconds, label)
    score_map = _score(runtime["model"], preprocessor, objective_fn, target, candidates, smoke)
    return audit, errors, score_map


def _gpu_evidence(torch: Any) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; refusing accidental CPU model execution")
    index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(index)
    name = str(props.name)
    if "v100" not in name.casefold():
        raise RuntimeError(f"this screen requires a V100 allocation, got GPU {name!r}")
    return {
        "device_index": index,
        "name": name,
        "total_memory_bytes": int(props.total_memory),
        "torch": str(torch.__version__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def _verified_runtime_identity(runtime: Mapping[str, Any], smoke: Any) -> Dict[str, Any]:
    """Hash the loaded checkpoint on compute and bind targets to that file."""
    identity = dict(smoke._checkpoint_identity(runtime))
    checkpoint_sha256 = _sha256_file(Path(runtime["checkpoint"]).resolve())
    if checkpoint_sha256 is None:
        raise FileNotFoundError(f"checkpoint is not hashable: {runtime['checkpoint']}")
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "checkpoint SHA-256 does not match the frozen source identity; "
            f"got {checkpoint_sha256}, expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    identity["checkpoint_sha256"] = checkpoint_sha256
    return identity


def _model_structure(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    """Record live structural fields without assuming a model implementation."""
    model = runtime["model"]
    cfg = runtime.get("model_cfg")

    def value(*objects: Any, name: str) -> Any:
        for obj in objects:
            if obj is None:
                continue
            candidate = getattr(obj, name, None)
            if candidate is not None and not callable(candidate):
                try:
                    return int(candidate)
                except (TypeError, ValueError):
                    return str(candidate)
        return None

    encoder = getattr(model, "encoder", None)
    return {
        "num_hist": value(model, cfg, name="num_hist"),
        "num_pred": value(model, cfg, name="num_pred"),
        "frameskip": value(model, cfg, name="frameskip"),
        "concat_dim": value(model, encoder, cfg, name="concat_dim"),
        "model_proprio_dim": value(model, cfg, name="proprio_dim"),
        "encoder_emb_dim": value(encoder, model, cfg, name="emb_dim"),
        "action_dim": value(model, cfg, name="action_dim"),
    }


def _dataset_mapping_audit(dset: Any) -> Dict[str, Any]:
    """Resolve only Subset-style metadata; never read reserved samples."""
    positions = list(range(max(DATASET_INDICES) + 1))
    current = dset
    layers: List[Dict[str, Any]] = []
    while hasattr(current, "indices") and hasattr(current, "dataset"):
        raw_indices = list(getattr(current, "indices"))
        next_dataset = getattr(current, "dataset")
        mapped: List[int] = []
        for position in positions:
            if position >= len(raw_indices):
                raise RuntimeError(f"dataset subset metadata cannot map local index {position}")
            value = raw_indices[position]
            if hasattr(value, "item"):
                value = value.item()
            mapped.append(int(value))
        layers.append({
            "type": type(current).__name__,
            "dataset_type": type(next_dataset).__name__,
            "index_count": len(raw_indices),
            "metadata_only": True,
        })
        positions = mapped
        current = next_dataset
    fresh_ids = positions[DATASET_INDICES[0] : DATASET_INDICES[-1] + 1]
    excluded_ids = positions[: len(EXCLUDED_DATASET_INDICES)]
    if len(set(fresh_ids)) != EPISODE_COUNT:
        raise RuntimeError("fresh dataset positions map to duplicate underlying episode ids")
    overlap = sorted(set(fresh_ids).intersection(excluded_ids))
    if overlap:
        raise RuntimeError(f"fresh dataset mapping overlaps excluded 0..95 mapping: {overlap}")
    return {
        "dataset_type_after_subset_resolution": type(current).__name__,
        "subset_layers": layers,
        "fresh_positions": list(DATASET_INDICES),
        "fresh_underlying_ids": fresh_ids,
        "excluded_positions": list(EXCLUDED_DATASET_INDICES),
        "excluded_underlying_id_count": len(excluded_ids),
        "fresh_distinct": True,
        "fresh_disjoint_from_excluded": True,
        "historical_fingerprint_registry": {
            "checked": False,
            "coverage": "no registry supplied; this does not claim full historical state disjointness",
        },
    }


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep summary inspectable and bounded; detailed quantizer audit is separate."""
    allowed = {
        "schema", "status", "allocation", "parameters", "source_identity", "runtime_identity",
        "gpu", "model_structure", "dataset_mapping", "allocation_evidence", "target_manifest", "raw_npz", "quantizer_audit", "episode_count", "completed_episodes",
        "metrics", "gates", "decision", "mechanism_verdict", "practical_decision", "elapsed_seconds", "max_seconds", "unresolved", "error",
    }
    return {key: value for key, value in summary.items() if key in allowed}


def _write_progress(
    output: Path,
    actions: np.ndarray,
    fp32: np.ndarray,
    rtn: np.ndarray,
    rtn_w8: np.ndarray,
    independent: np.ndarray,
    antithetic: np.ndarray,
    completed: np.ndarray,
) -> None:
    _atomic_npz(
        output / "raw_scores.npz",
        schema=np.asarray(RAW_SCHEMA),
        actions=actions,
        scores_fp32=fp32,
        scores_rtn=rtn,
        scores_rtn_w8=rtn_w8,
        scores_independent=independent,
        scores_antithetic=antithetic,
        completed=completed,
    )


def _check_deadline(started: float, max_seconds: float, label: str) -> None:
    if time.monotonic() - started >= max_seconds:
        raise TimeoutError(f"max-seconds reached at {label}")


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    smoke, screen = _load_helpers()
    helper_identity = _assert_helper_identity(smoke, screen)
    runtime = smoke._runtime(args.root.resolve())
    model_structure = _model_structure(runtime)
    if model_structure.get("num_hist") != 1:
        raise RuntimeError(
            "structural no-go: frozen checkpoint must use num_hist=1; "
            f"live value is {model_structure.get('num_hist')!r}"
        )
    if model_structure.get("action_dim") not in (None, ACTION_DIM):
        raise RuntimeError(
            "structural no-go: target action_dim=10 disagrees with live model; "
            f"live value is {model_structure.get('action_dim')!r}"
        )
    if len(runtime["dset"]) < max(DATASET_INDICES) + 1:
        raise RuntimeError(
            f"validation dataset has only {len(runtime['dset'])} trajectories; "
            f"fresh screen requires indices through {max(DATASET_INDICES)}"
        )
    dataset_mapping = _dataset_mapping_audit(runtime["dset"])
    torch = runtime["torch"]
    runtime_identity = _verified_runtime_identity(runtime, smoke)
    gpu_start = _gpu_evidence(torch)
    manifest_path = _target_manifest_path((args.targets_dir or (output / "targets")).resolve())
    manifest = _prepare_targets(runtime, manifest_path.parent, smoke, screen, runtime_identity, model_structure)
    episodes = manifest["episodes"]
    groups = smoke._linear_groups(runtime["model"])
    predictor = _predictor_linears(runtime["model"], groups, smoke)
    snapshot = smoke._snapshot_weights(runtime["model"], groups)
    _snapshot_equal(runtime["model"], snapshot, smoke)
    predictor_w4_bytes = int(sum(group["logical_weight_bytes_W4"] for group in groups if group["family"] == "predictor"))
    predictor_w8_bytes = int(sum(group["logical_weight_bytes_W8"] for group in groups if group["family"] == "predictor"))
    preprocessor = smoke._preprocessor(runtime)
    objective_fn = smoke._objective()

    actions = np.full((EPISODE_COUNT, NUM_SAMPLES, ACTION_FLAT_DIM), np.nan, dtype=np.float32)
    scores_fp32 = np.full((EPISODE_COUNT, NUM_SAMPLES), np.nan, dtype=np.float32)
    scores_rtn = np.full((EPISODE_COUNT, NUM_SAMPLES), np.nan, dtype=np.float32)
    scores_rtn_w8 = np.full((EPISODE_COUNT, NUM_SAMPLES), np.nan, dtype=np.float32)
    scores_independent = np.full((EPISODE_COUNT, len(ROUNDING_SEEDS), 2, NUM_SAMPLES), np.nan, dtype=np.float32)
    scores_antithetic = np.full_like(scores_independent, np.nan)
    completed = np.zeros(EPISODE_COUNT, dtype=np.bool_)
    episode_metrics: List[Dict[str, Any]] = []
    quantizer_audit: List[Dict[str, Any]] = []

    source_identity = {
        "ensemble_screen": _file_identity(Path(__file__)),
        "smoke_runner": _file_identity(getattr(smoke, "__file__", "")),
        "screen_runner": _file_identity(getattr(screen, "__file__", "")),
        "expected_smoke_runner_sha256": EXPECTED_SMOKE_RUNNER_SHA256,
        "expected_screen_runner_sha256": EXPECTED_SCREEN_RUNNER_SHA256,
        "verified_helper_sha256": helper_identity,
        "source_commit": getattr(smoke, "SOURCE_COMMIT", None),
        "dinov2_source_commit": getattr(smoke, "DINOV2_COMMIT", None),
        "checkpoint_sha256": runtime_identity["checkpoint_sha256"],
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "runtime_root": str(args.root.resolve()),
    }
    try:
        for episode_index, episode in enumerate(episodes):
            _check_deadline(started, args.max_seconds, f"episode {episode_index} start")
            target = episode["target"]
            candidate_rng = np.random.default_rng(int(episode["candidate_seed"]))
            candidate_flat = candidate_rng.standard_normal((NUM_SAMPLES, ACTION_FLAT_DIM), dtype=np.float32)
            candidate_actions = candidate_flat.reshape(NUM_SAMPLES, HORIZON, ACTION_DIM)
            actions[episode_index] = candidate_flat

            _check_deadline(started, args.max_seconds, f"episode {episode_index} FP32")
            smoke._restore_weights(runtime["model"], snapshot)
            _snapshot_equal(runtime["model"], snapshot, smoke)
            fp_map = _score(runtime["model"], preprocessor, objective_fn, target, candidate_actions, smoke)
            scores_fp32[episode_index] = fp_map

            _check_deadline(started, args.max_seconds, f"episode {episode_index} RTN_W4")
            smoke._restore_weights(runtime["model"], snapshot)
            rtn_records = [smoke._quantize_group(runtime["model"], group, BITS) for group in groups if group["family"] == "predictor"]
            _assert_encoder_untouched(runtime["model"], snapshot, smoke)
            rtn_map = _score(runtime["model"], preprocessor, objective_fn, target, candidate_actions, smoke)
            scores_rtn[episode_index] = rtn_map

            _check_deadline(started, args.max_seconds, f"episode {episode_index} RTN_W8")
            smoke._restore_weights(runtime["model"], snapshot)
            rtn_w8_records = [smoke._quantize_group(runtime["model"], group, 8) for group in groups if group["family"] == "predictor"]
            _assert_encoder_untouched(runtime["model"], snapshot, smoke)
            rtn_w8_map = _score(runtime["model"], preprocessor, objective_fn, target, candidate_actions, smoke)
            scores_rtn_w8[episode_index] = rtn_w8_map

            seed_metrics: List[Dict[str, Any]] = []
            for seed_index, seed in enumerate(ROUNDING_SEEDS):
                _check_deadline(started, args.max_seconds, f"episode {episode_index}, seed {seed} start")
                member_a_id = f"seed_{seed}_shared_member_a"
                audit_a, errors_a, map_a = _evaluate_sr_member(
                    runtime, snapshot, predictor, smoke, preprocessor, objective_fn, target,
                    candidate_actions, seed, False, member_a_id, started, args.max_seconds,
                    f"episode {episode_index}, seed {seed} shared member",
                )
                member_i_id = f"seed_{seed}_independent_member_b"
                member_h_id = f"seed_{seed}_antithetic_member_b"
                if seed_index % 2 == 0:
                    first_id, first_seed, first_complement, first_label = member_i_id, seed + 100000, False, "independent"
                    second_id, second_seed, second_complement, second_label = member_h_id, seed, True, "antithetic"
                else:
                    first_id, first_seed, first_complement, first_label = member_h_id, seed, True, "antithetic"
                    second_id, second_seed, second_complement, second_label = member_i_id, seed + 100000, False, "independent"
                first_audit, first_errors, first_map = _evaluate_sr_member(
                    runtime, snapshot, predictor, smoke, preprocessor, objective_fn, target,
                    candidate_actions, first_seed, first_complement, first_id, started, args.max_seconds,
                    f"episode {episode_index}, seed {seed} {first_label} member",
                )
                second_audit, second_errors, second_map = _evaluate_sr_member(
                    runtime, snapshot, predictor, smoke, preprocessor, objective_fn, target,
                    candidate_actions, second_seed, second_complement, second_id, started, args.max_seconds,
                    f"episode {episode_index}, seed {seed} {second_label} member",
                )
                if first_id == member_i_id:
                    audit_i, errors_i, map_i = first_audit, first_errors, first_map
                    audit_h, errors_h, map_h = second_audit, second_errors, second_map
                else:
                    audit_h, errors_h, map_h = first_audit, first_errors, first_map
                    audit_i, errors_i, map_i = second_audit, second_errors, second_map
                independent_pair = _pair_error_metrics(errors_a, errors_i)
                antithetic_pair = _pair_error_metrics(errors_a, errors_h)
                if audit_a["scale_sha256"] != audit_h["scale_sha256"]:
                    raise RuntimeError("antithetic member did not reuse the shared FP32 W4 grid")

                independent_maps = np.stack([map_a, map_i])
                antithetic_maps = np.stack([map_a.copy(), map_h])
                if not np.array_equal(independent_maps[0], antithetic_maps[0]):
                    raise RuntimeError("independent/antithetic first member was not shared exactly")
                scores_independent[episode_index, seed_index] = independent_maps
                scores_antithetic[episode_index, seed_index] = antithetic_maps
                quantizer_audit.append({
                    "episode_id": episode["episode_id"],
                    "seed": int(seed),
                    "shared_first_member": True,
                    "second_member_execution_order": [first_label, second_label],
                    "independent": {"member_a": audit_a, "member_b": audit_i, "pair": independent_pair},
                    "antithetic": {"member_a_shared": member_a_id, "member_b": audit_h, "pair": antithetic_pair},
                    "rtn_predictor_groups": rtn_records,
                    "rtn_w8_predictor_groups": rtn_w8_records,
                })
                independent_ensemble = scores_independent[episode_index, seed_index].mean(axis=0)
                antithetic_ensemble = scores_antithetic[episode_index, seed_index].mean(axis=0)
                seed_metrics.append({
                    "seed": int(seed),
                    "independent": _selection_metrics(fp_map, independent_ensemble, candidate_flat),
                    "antithetic": _selection_metrics(fp_map, antithetic_ensemble, candidate_flat),
                    "pair_weight": {"independent": independent_pair, "antithetic": antithetic_pair},
                })
                del errors_a, errors_i, errors_h

            smoke._restore_weights(runtime["model"], snapshot)
            _snapshot_equal(runtime["model"], snapshot, smoke)
            episode_metrics.append({
                "episode_id": episode["episode_id"],
                "local_index": int(episode["local_index"]),
                "dataset_index": int(episode["dataset_index"]),
                "env_seed": int(episode["env_seed"]),
                "candidate_seed": int(episode["candidate_seed"]),
                "target_fingerprint": episode["target_fingerprint"],
                "fp32_min_score": float(fp_map.min()),
                "rtn": _selection_metrics(fp_map, rtn_map, candidate_flat),
                "rtn_w8": _selection_metrics(fp_map, rtn_w8_map, candidate_flat),
                "rounding_seeds": seed_metrics,
            })
            completed[episode_index] = True
            _write_progress(output, actions, scores_fp32, scores_rtn, scores_rtn_w8, scores_independent, scores_antithetic, completed)
            interim = {
                "schema": SCHEMA,
                "status": "partial",
                "allocation": dict(allocation),
                "episode_count": EPISODE_COUNT,
                "completed_episodes": int(completed.sum()),
                "raw_npz": str((output / "raw_scores.npz").resolve()),
                "max_seconds": args.max_seconds,
                "elapsed_seconds": time.monotonic() - started,
            }
            _atomic_json(output / "summary.json", _compact_summary(interim))
    finally:
        smoke._restore_weights(runtime["model"], snapshot)
        _snapshot_equal(runtime["model"], snapshot, smoke)

    def aggregate(method: str, metric: str) -> Dict[str, Any]:
        episode_values = []
        for record in episode_metrics:
            if method in {"rtn", "rtn_w8"}:
                episode_values.append(float(record[method][metric]))
            else:
                values = [float(seed_record[method][metric]) for seed_record in record["rounding_seeds"]]
                episode_values.append(float(np.mean(values)))
        return {"episode_means": episode_values, "mean_over_episodes": float(np.mean(episode_values))}

    metric_names = (
        "elite_mean_action_mse",
        "selected_top30_fp_score_regret",
        "score_nmse",
        "selected_top30_overlap",
    )
    aggregate_metrics = {
        method: {metric: aggregate(method, metric) for metric in metric_names}
        for method in ("rtn", "rtn_w8", "independent", "antithetic")
    }
    aggregate_metrics["rounding_difference_antithetic_minus_independent"] = {
        metric: float(aggregate_metrics["antithetic"][metric]["mean_over_episodes"] - aggregate_metrics["independent"][metric]["mean_over_episodes"])
        for metric in metric_names
    }

    independent_regret = aggregate_metrics["independent"]["selected_top30_fp_score_regret"]["mean_over_episodes"]
    antithetic_regret = aggregate_metrics["antithetic"]["selected_top30_fp_score_regret"]["mean_over_episodes"]
    independent_episode_regret = np.asarray(aggregate_metrics["independent"]["selected_top30_fp_score_regret"]["episode_means"], dtype=np.float64)
    antithetic_episode_regret = np.asarray(aggregate_metrics["antithetic"]["selected_top30_fp_score_regret"]["episode_means"], dtype=np.float64)
    improved_episodes = int(np.sum(antithetic_episode_regret < independent_episode_regret))
    independent_seed_regret = np.asarray(
        [[seed_record["independent"]["selected_top30_fp_score_regret"] for seed_record in record["rounding_seeds"]] for record in episode_metrics],
        dtype=np.float64,
    )
    antithetic_seed_regret = np.asarray(
        [[seed_record["antithetic"]["selected_top30_fp_score_regret"] for seed_record in record["rounding_seeds"]] for record in episode_metrics],
        dtype=np.float64,
    )
    improved_seeds = int(np.sum(antithetic_seed_regret.mean(axis=0) < independent_seed_regret.mean(axis=0)))
    if independent_regret > NMSE_EPS:
        regret_reduction = float((independent_regret - antithetic_regret) / independent_regret)
        relative_pass = bool(regret_reduction >= PRIMARY_IMPROVEMENT_THRESHOLD)
    else:
        regret_reduction = None
        relative_pass = False
    regret_not_worse_rtn = bool(
        antithetic_regret <= aggregate_metrics["rtn"]["selected_top30_fp_score_regret"]["mean_over_episodes"] + GATE_TOLERANCE
    )
    primary_pass = bool(
        independent_regret > NMSE_EPS
        and relative_pass
        and improved_episodes >= 4
        and improved_seeds >= 2
        and regret_not_worse_rtn
    )
    independent_action_mse = aggregate_metrics["independent"]["elite_mean_action_mse"]["mean_over_episodes"]
    antithetic_action_mse = aggregate_metrics["antithetic"]["elite_mean_action_mse"]["mean_over_episodes"]
    rtn_action_mse = aggregate_metrics["rtn"]["elite_mean_action_mse"]["mean_over_episodes"]
    action_gate_pass = bool(
        antithetic_action_mse <= independent_action_mse + GATE_TOLERANCE
        and antithetic_action_mse <= rtn_action_mse + GATE_TOLERANCE
    )
    engineering_gate_pass = bool(
        int(completed.sum()) == EPISODE_COUNT
        and all(np.isfinite(array[completed]).all() for array in (scores_fp32, scores_rtn, scores_rtn_w8, scores_independent, scores_antithetic))
    )
    gates = {
        "engineering": {
            "passed": engineering_gate_pass,
            "completed_episodes": int(completed.sum()),
            "required_episodes": EPISODE_COUNT,
            "snapshot_restore": "exact torch.equal checked after every member and at run end",
            "encoder_untouched": True,
        },
        "primary_regret_reduction": {
            "passed": primary_pass,
            "headroom": bool(independent_regret > NMSE_EPS),
            "threshold": PRIMARY_IMPROVEMENT_THRESHOLD,
            "direction": "antithetic lower is better",
            "independent_regret_aggregate": independent_regret,
            "antithetic_regret_aggregate": antithetic_regret,
            "reduction_fraction": regret_reduction,
            "relative_improvement_passed": relative_pass,
            "improved_episodes": improved_episodes,
            "required_improved_episodes": 4,
            "improved_rounding_seeds": improved_seeds,
            "required_improved_rounding_seeds": 2,
            "regret_not_worse_than_rtn": regret_not_worse_rtn,
            "denominator": "先按episode平均3个rounding seeds，再对6 episodes求mean；aggregate independent regret",
            "metric": "one_shot_initial_pool_shortlist_regret",
        },
        "elite_mean_action_mse": {
            "passed": action_gate_pass,
            "tolerance": GATE_TOLERANCE,
            "antithetic": antithetic_action_mse,
            "independent": independent_action_mse,
            "rtn_w4": rtn_action_mse,
            "rule": "overall antithetic <= independent + tau and <= RTN_W4 + tau",
        },
        "overall": {
            "passed": bool(engineering_gate_pass and primary_pass and action_gate_pass),
            "requires": ["engineering", "primary_regret_reduction", "elite_mean_action_mse"],
        },
    }
    w8_practical_dominates = bool(
        aggregate_metrics["rtn_w8"]["selected_top30_fp_score_regret"]["mean_over_episodes"] <= antithetic_regret + GATE_TOLERANCE
        and aggregate_metrics["rtn_w8"]["elite_mean_action_mse"]["mean_over_episodes"] <= antithetic_action_mse + GATE_TOLERANCE
        and predictor_w8_bytes <= 2 * predictor_w4_bytes
    )
    mechanism_verdict = "preliminary_go" if gates["overall"]["passed"] else "mechanism_no_go"
    decision = "practical_no_go" if w8_practical_dominates else mechanism_verdict

    gpu_end = _gpu_evidence(torch)
    raw_path = output / "raw_scores.npz"
    audit_path = output / "quantizer_audit.json"
    _atomic_json(audit_path, {
        "schema": "antithetic-rounding-quantizer-audit-v1",
        "formula": "q=clamp(floor(x/s)+1[u < fraction(x/s)],-7,7); dequant=q*s",
        "uniform_grid": "24-bit midpoint grid; independent and antithetic members share the same finite marginal",
        "scope": "predictor 24 Linear modules only; encoder remains FP32",
        "members": quantizer_audit,
    })
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "allocation": dict(allocation),
        "parameters": {
            "episodes": EPISODE_COUNT,
            "dataset_indices": list(DATASET_INDICES),
            "excluded_dataset_indices": list(EXCLUDED_DATASET_INDICES),
            "env_seed_namespace": ENV_SEED_NAMESPACE,
            "candidate_seed_namespace": CANDIDATE_SEED_NAMESPACE,
            "rounding_seeds": list(ROUNDING_SEEDS),
            "candidate_shape": [NUM_SAMPLES, HORIZON, ACTION_DIM],
            "candidate_flat_actiondim": ACTION_FLAT_DIM,
            "candidate_distribution": "standard_normal",
            "rounding_uniform_grid": "24-bit midpoint grid; u=(k+0.5)/2^24 and antithetic=1-u",
            "topk": TOPK,
            "bits": BITS,
            "rtn_w4_scope": "predictor_only_per_output_channel_RTN",
            "rtn_w8_scope": "predictor_only_per_output_channel_RTN_cost_context_only",
            "predictor_linear_count": len(predictor),
            "logical_tensor_bytes": {
                "predictor_W4_single_member": predictor_w4_bytes,
                "predictor_W8_single_member": predictor_w8_bytes,
                "two_member_W4_independent": 2 * predictor_w4_bytes,
                "two_member_W4_antithetic": 2 * predictor_w4_bytes,
                "W8_cost_context_only": True,
            },
            "shared_first_member": True,
            "second_member_order": "independent_then_antithetic for seed indices 0,2; antithetic_then_independent for seed index 1",
            "score_map_order": ["independent_member_a", "independent_member_b", "antithetic_member_a_shared", "antithetic_member_b"],
            "unique_forwards_per_seed": 3,
            "ensemble_forward_budget": 2,
            "w8_baseline": {"scheme": "predictor_only_per_output_channel_RTN", "bits": 8, "gate_role": "cost_context_only"},
            "gate_tolerance": GATE_TOLERANCE,
            "primary_improvement_threshold": PRIMARY_IMPROVEMENT_THRESHOLD,
            "max_seconds": args.max_seconds,
        },
        "source_identity": source_identity,
        "runtime_identity": runtime_identity,
        "gpu": {"start": gpu_start, "end": gpu_end},
        "model_structure": model_structure,
        "dataset_mapping": dataset_mapping,
        "allocation_evidence": {
            "guard_json": str((output / "allocation.json").resolve()),
            "gpu_identity_txt": str((output / "gpu_identity.txt").resolve()),
            "run_log": str((output / "run.log").resolve()),
        },
        "target_manifest": str(manifest_path.resolve()),
        "raw_npz": str(raw_path.resolve()),
        "quantizer_audit": str(audit_path.resolve()),
        "episode_count": EPISODE_COUNT,
        "completed_episodes": int(completed.sum()),
        "metrics": {
            "per_episode": episode_metrics,
            "aggregate": aggregate_metrics,
            "rounding_seed_unit": "repeated_measurement_within_episode; seeds are not 18 independent samples",
            "regret_definition": "mean FP32 score of selected top30 minus mean FP32 score of FP32 top30; lower is better",
            "elite_mean_definition": "MSE between selected and FP32 top30 action means over flattened actiondim=10",
            "score_nmse_definition": "mean squared score error divided by mean squared FP32 score",
            "w8_gate_role": "single W8 RTN supplies the separate frozen practical dominance gate; it does not tune the mechanism gate",
            "w8_practical_dominance": {
                "dominates_antithetic_on_regret_and_action_mse": w8_practical_dominates,
                "bytes_not_more_than_two_w4": predictor_w8_bytes <= 2 * predictor_w4_bytes,
                "forward_count_not_more_than_two_w4": True,
                "comparison": "W8 RTN one forward versus two-member W4 ensemble; practical_no_go is separate from mechanism verdict",
            },
        },
        "gates": gates,
        "decision": decision,
        "mechanism_verdict": mechanism_verdict,
        "practical_decision": "practical_no_go" if w8_practical_dominates else "w8_not_dominant_on_screen_metrics",
        "elapsed_seconds": time.monotonic() - started,
        "max_seconds": args.max_seconds,
        "unresolved": [
            "fake quantization executes ordinary FP32 operators; no native W4 kernel, memory, speed, training, or teacher-deployment claim",
            "six fresh episodes and three repeated rounding seeds are a bounded screen, not a confirmatory benchmark",
        ],
    }
    compact = _compact_summary(summary)
    _atomic_json(output / "summary.json", compact)
    size = (output / "summary.json").stat().st_size
    if size >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB: {size} bytes")
    print(json.dumps({"status": "complete", "episodes": EPISODE_COUNT, "summary": str(output / "summary.json"), "summary_bytes": size}, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="remote DINO-WM modelroot")
    parser.add_argument("--output", type=Path, required=True, help="one-job artifact directory")
    parser.add_argument("--targets-dir", type=Path, default=None, help="fresh target directory; defaults to output/targets")
    parser.add_argument("--max-seconds", type=float, default=2400.0)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    return args


def main() -> None:
    # Deliberately first: no model, target, or raw-array I/O precedes this call.
    require_allocation = _load_allocation_guard()
    allocation = require_allocation()
    args = _parse_args()
    started = time.monotonic()
    try:
        _run(args, allocation)
    except Exception as exc:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "allocation": allocation,
            "error": repr(exc),
            "raw_npz": str((args.output.resolve() / "raw_scores.npz")),
            "completed_episodes": len(list((output / "targets" / "fresh").glob("episode_*.pkl"))) if (output / "targets" / "fresh").is_dir() else 0,
            "elapsed_seconds": time.monotonic() - started,
            "unresolved": ["results are incomplete; no research no-go/go conclusion is supported"],
        }
        _atomic_json(output / "summary.json", _compact_summary(failure))
        raise


if __name__ == "__main__":
    main()
