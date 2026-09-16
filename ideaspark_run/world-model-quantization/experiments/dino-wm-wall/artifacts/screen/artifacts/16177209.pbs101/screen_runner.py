"""RankCal/Wall fast-screen episode runner.

This file is intentionally a thin adapter around :mod:`smoke_runner` and the
official DINO-WM planning runtime.  It runs one real Wall episode at a time,
with a fixed CEM5/MPC12 budget.  Quantization is the smoke runner's numerical
fake-quantization path: weights are dequantized and evaluated by FP32
operators, so this runner does not make a native INT4/INT8 claim.

Examples::

    python screen_runner.py --root RUNTIME --output ARTIFACTS \
        --split development --mode FP32 --targets-dir ARTIFACTS/targets
    python screen_runner.py --root RUNTIME --output ARTIFACTS \
        --split development --mode all_W4 --targets-dir ARTIFACTS/targets
    python screen_runner.py --root RUNTIME --output ARTIFACTS \
        --split pilot_test --mode allocation --allocation-json allocations.json \
        --targets-dir ARTIFACTS/targets

The first FP32 job (or ``--prepare-targets-only``) creates the frozen target
manifest.  Other model jobs read it and fail clearly if it is absent or
inconsistent; they do not wait while occupying a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

# Importing this module is cheap and gives us the verified loader, target
# helpers, quantizer and planner adapter.  It deliberately imports torch only
# inside runtime functions.  The fallback keeps direct module imports working
# when the caller's current directory is not this experiment directory.
try:
    import smoke_runner as smoke
except ModuleNotFoundError:
    _smoke_spec = importlib.util.spec_from_file_location(
        "rankcal_wall_smoke_runner", Path(__file__).with_name("smoke_runner.py")
    )
    if _smoke_spec is None or _smoke_spec.loader is None:
        raise ImportError("cannot load smoke_runner.py")
    smoke = importlib.util.module_from_spec(_smoke_spec)
    _smoke_spec.loader.exec_module(smoke)


SCREEN_SCHEMA = "rankcal-wall-screen-v1"
TARGET_SCHEMA = "rankcal-wall-targets-v1"
RESULT_SCHEMA = "rankcal-wall-episode-result-v1"
SUMMARY_SCHEMA = "rankcal-wall-screen-summary-v1"

SPLIT_SIZES = {
    "calibration": 8,
    "development": 8,
    "pilot_test": 24,
}
SPLIT_NAMESPACES = {
    "calibration": 200000,
    "development": 300000,
    "pilot_test": 400000,
}
# Dataset indices are assigned once in this fixed order.  This gives every
# split a disjoint, auditable source-trajectory set without random resampling.
SPLIT_DATASET_OFFSETS = {
    # smoke owns dataset indices 0 and 1; screen starts at index 2.
    "calibration": 2,
    "development": 10,
    "pilot_test": 18,
}

HORIZON = 5
EXECUTE_MODEL_ACTIONS = 5
FRAMESKIP = 5
NUM_SAMPLES = 300
TOPK = 30
CEM_STEPS = 5
MPC_ROUNDS = 12
ENV_STEP_LIMIT = MPC_ROUNDS * EXECUTE_MODEL_ACTIONS * FRAMESKIP


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


def _digest_arrays(*arrays: Any, metadata: Mapping[str, Any] | None = None) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    if metadata is not None:
        hasher.update(json.dumps(metadata, sort_keys=True, default=_json_default).encode("utf-8"))
    for value in arrays:
        array = np.asarray(value)
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(repr(tuple(array.shape)).encode("ascii"))
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.copy()
    return value.detach().cpu().numpy().copy()


def _layout_from_info(info: Mapping[str, Any]) -> Dict[str, int]:
    def scalar(value: Any) -> int:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)

    return {
        "fix_door_location": scalar(info["fix_door_location"]),
        "fix_wall_location": scalar(info["fix_wall_location"]),
    }


def _normalization_identity(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    dset = runtime["dset"]
    return {
        "action_mean": _digest_arrays(_to_numpy(dset.action_mean)),
        "action_std": _digest_arrays(_to_numpy(dset.action_std)),
        "state_mean": _digest_arrays(_to_numpy(dset.state_mean)),
        "state_std": _digest_arrays(_to_numpy(dset.state_std)),
        "proprio_mean": _digest_arrays(_to_numpy(dset.proprio_mean)),
        "proprio_std": _digest_arrays(_to_numpy(dset.proprio_std)),
        "transform": repr(dset.transform),
    }


def _target_manifest_path(targets_dir: Path, split: str) -> Path:
    return targets_dir / f"episode_manifest_{split}.json"


def _target_record_path(targets_dir: Path, split: str, local_index: int) -> Path:
    return targets_dir / split / f"episode_{local_index:03d}.pkl"


def _target_fingerprint(target: Mapping[str, Any], layout: Mapping[str, Any]) -> str:
    return _digest_arrays(
        target["obs_0"]["visual"],
        target["obs_0"]["proprio"],
        target["obs_g"]["visual"],
        target["obs_g"]["proprio"],
        target["state_0"],
        target["state_g"],
        metadata={"layout": dict(layout), "goal_H": int(target["goal_H"])},
    )


def _validate_runtime_identity(manifest: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    recorded_identity = manifest.get("checkpoint_identity", {})
    current_identity = smoke._checkpoint_identity(runtime)
    for key in ("recorded_epoch", "checkpoint_size_bytes", "source_commit", "dinov2_source_commit", "dtype", "decoder", "execution", "torch"):
        if recorded_identity.get(key) != current_identity.get(key):
            raise RuntimeError(f"frozen target checkpoint identity mismatch at {key!r}")
    if manifest.get("normalization_identity") != _normalization_identity(runtime):
        raise RuntimeError("frozen target normalization identity does not match this runtime")


def _all_target_manifests(targets_dir: Path) -> Iterable[Tuple[Path, Mapping[str, Any]]]:
    for path in sorted(targets_dir.glob("episode_manifest_*.json")):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError(f"cannot read target manifest: {path}")


def _validate_manifest(
    manifest: Mapping[str, Any],
    targets_dir: Path,
    split: str,
    dset_length: int | None = None,
) -> List[Dict[str, Any]]:
    if manifest.get("schema") != TARGET_SCHEMA:
        raise RuntimeError(f"unsupported target manifest schema: {manifest.get('schema')!r}")
    if manifest.get("split") != split:
        raise RuntimeError(f"target manifest split mismatch: {manifest.get('split')!r} != {split!r}")
    expected_size = SPLIT_SIZES[split]
    expected_namespace = SPLIT_NAMESPACES[split]
    if manifest.get("episode_count") != expected_size:
        raise RuntimeError(f"{split} target count must be {expected_size}")
    if manifest.get("seed_namespace_start") != expected_namespace:
        raise RuntimeError(f"{split} target namespace is not {expected_namespace}")

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_size:
        raise RuntimeError("target manifest episodes must contain the fixed split size")
    dataset_indices: List[int] = []
    fingerprints: List[str] = []
    result: List[Dict[str, Any]] = []
    for local_index, entry in enumerate(episodes):
        if not isinstance(entry, dict):
            raise RuntimeError(f"invalid target entry at {local_index}")
        expected_id = f"{split}:{local_index:03d}"
        if entry.get("episode_id") != expected_id:
            raise RuntimeError(f"target episode id mismatch at {local_index}")
        expected_dataset = SPLIT_DATASET_OFFSETS[split] + local_index
        if int(entry.get("dataset_index", -1)) != expected_dataset:
            raise RuntimeError(f"dataset index mismatch for {expected_id}")
        expected_env = expected_namespace + local_index
        expected_cem = expected_namespace + 10000 + local_index
        if int(entry.get("env_seed", -1)) != expected_env or int(entry.get("cem_seed", -1)) != expected_cem:
            raise RuntimeError(f"seed mismatch for {expected_id}")
        target_rel = entry.get("target_path")
        if not isinstance(target_rel, str):
            raise RuntimeError(f"missing target path for {expected_id}")
        target_path = targets_dir / target_rel
        if not target_path.is_file():
            raise RuntimeError(f"target file missing for {expected_id}: {target_path}")
        with target_path.open("rb") as stream:
            target = pickle.load(stream)
        if target.get("goal_H") != HORIZON:
            raise RuntimeError(f"target horizon mismatch for {expected_id}")
        layout = entry.get("layout")
        actual_fingerprint = _target_fingerprint(target, layout)
        if actual_fingerprint != entry.get("target_fingerprint"):
            raise RuntimeError(f"target fingerprint mismatch for {expected_id}")
        dataset_indices.append(expected_dataset)
        fingerprints.append(actual_fingerprint)
        result.append({
            **entry,
            "local_index": local_index,
            "target_path": str(target_path.resolve()),
            "target": target,
        })
    if len(set(dataset_indices)) != len(dataset_indices):
        raise RuntimeError(f"duplicate dataset indices in {split} target manifest")
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError(f"duplicate (start, goal, layout) targets in {split} target manifest")
    if dset_length is not None and max(dataset_indices, default=-1) >= dset_length:
        raise RuntimeError(f"validation dataset has only {dset_length} trajectories")
    return result


def _validate_cross_split_dedup(targets_dir: Path) -> None:
    dataset_indices: Dict[int, str] = {}
    fingerprints: Dict[str, str] = {}
    for path, manifest in _all_target_manifests(targets_dir):
        split = manifest.get("split")
        if split not in SPLIT_SIZES:
            continue
        for entry in manifest.get("episodes", []):
            dataset_index = int(entry.get("dataset_index", -1))
            episode_id = str(entry.get("episode_id"))
            prior = dataset_indices.get(dataset_index)
            if prior is not None and prior != episode_id:
                raise RuntimeError(f"dataset index {dataset_index} reused by {prior} and {episode_id}")
            dataset_indices[dataset_index] = episode_id
            fingerprint = entry.get("target_fingerprint")
            if fingerprint:
                prior_fp = fingerprints.get(str(fingerprint))
                if prior_fp is not None and prior_fp != episode_id:
                    raise RuntimeError(f"target fingerprint reused by {prior_fp} and {episode_id}")
                fingerprints[str(fingerprint)] = episode_id


def _new_target(runtime: Mapping[str, Any], dataset_index: int, env_seed: int) -> Dict[str, Any]:
    dset = runtime["dset"]
    if dataset_index >= len(dset):
        raise RuntimeError(f"validation dataset has only {len(dset)} trajectories; need index {dataset_index}")
    env = smoke._new_wall_env(runtime, count=1)
    try:
        _, _, _, env_info = dset[dataset_index]
        layout = _layout_from_info(env_info)
        torch = runtime["torch"]
        env.update_env([{
            "fix_door_location": torch.tensor(layout["fix_door_location"]),
            "fix_wall_location": torch.tensor(layout["fix_wall_location"]),
        }])
        init_state, goal_state = env.sample_random_init_goal_states([env_seed])
        obs_0, state_0 = env.prepare([env_seed], init_state)
        obs_g, state_g = env.prepare([env_seed], goal_state)
        target = {
            "obs_0": {key: np.expand_dims(_to_numpy(value), axis=1) for key, value in obs_0.items()},
            "obs_g": {key: np.expand_dims(_to_numpy(value), axis=1) for key, value in obs_g.items()},
            "state_0": _to_numpy(state_0),
            "state_g": _to_numpy(state_g),
            "gt_actions": None,
            "goal_H": HORIZON,
        }
        return target, layout
    finally:
        smoke._close_env(env)


def _prepare_targets(runtime: Mapping[str, Any], targets_dir: Path, split: str) -> Mapping[str, Any]:
    """Create one split manifest under an exclusive lock, or read it back."""
    targets_dir = targets_dir.resolve()
    targets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _target_manifest_path(targets_dir, split)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest, targets_dir, split, len(runtime["dset"]))
        _validate_runtime_identity(manifest, runtime)
        _validate_cross_split_dedup(targets_dir)
        return manifest

    lock_path = targets_dir / f".episode_manifest_{split}.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"target generation is already in progress: {lock_path}; rerun after the FP32 prepare job finishes"
        ) from exc

    try:
        # A second check matters if a previous writer finished between the
        # first stat and lock creation.
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_manifest(manifest, targets_dir, split, len(runtime["dset"]))
            _validate_runtime_identity(manifest, runtime)
            return manifest

        expected_size = SPLIT_SIZES[split]
        namespace = SPLIT_NAMESPACES[split]
        offset = SPLIT_DATASET_OFFSETS[split]
        norm_id = _normalization_identity(runtime)
        episodes: List[Dict[str, Any]] = []
        for local_index in range(expected_size):
            dataset_index = offset + local_index
            env_seed = namespace + local_index
            cem_seed = namespace + 10000 + local_index
            target, layout = _new_target(runtime, dataset_index, env_seed)
            fingerprint = _target_fingerprint(target, layout)
            target_path = _target_record_path(targets_dir, split, local_index)
            target["target_fingerprint"] = fingerprint
            _atomic_pickle(target_path, target)
            episodes.append({
                "episode_id": f"{split}:{local_index:03d}",
                "split": split,
                "local_index": local_index,
                "dataset_index": dataset_index,
                "env_seed": env_seed,
                "cem_seed": cem_seed,
                "layout": layout,
                "target_fingerprint": fingerprint,
                "target_path": str(target_path.relative_to(targets_dir)),
            })
        manifest = {
            "schema": TARGET_SCHEMA,
            "screen_schema": SCREEN_SCHEMA,
            "split": split,
            "episode_count": expected_size,
            "seed_namespace_start": namespace,
            "dataset_index_offset": offset,
            "dataset_index_strategy": "fixed_global_disjoint_ranges_smoke_0_1_calibration_2_9_development_10_17_pilot_test_18_41",
            "target_rng": "environment seed is namespace+local_index; CEM seed is namespace+10000+local_index and never used to generate targets",
            "normalization_identity": norm_id,
            "checkpoint_identity": smoke._checkpoint_identity(runtime),
            "episodes": episodes,
        }
        _atomic_json(manifest_path, manifest)
        _validate_manifest(manifest, targets_dir, split, len(runtime["dset"]))
        _validate_cross_split_dedup(targets_dir)
        return manifest
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _protocol_config() -> Dict[str, Any]:
    return {
        "horizon": HORIZON,
        "execute_model_actions": EXECUTE_MODEL_ACTIONS,
        "frameskip": FRAMESKIP,
        "num_samples": NUM_SAMPLES,
        "elite_count": TOPK,
        "cem_iterations": CEM_STEPS,
        "outer_mpc_round_limit": MPC_ROUNDS,
        "environment_step_limit": ENV_STEP_LIMIT,
        "inner_environment_evaluator": None,
        "stable_argsort": True,
        "objective": {"mode": "last", "alpha": 1, "base": 2},
        "reference": {"dtype": "float32", "decoder": None, "activations": "float32"},
        "success": "real Wall env.eval_state after executed MPC boundary; position xy distance < 4.5",
        "quantization_execution": "emulation_only",
    }


def _allocation_mapping(payload: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    raw: Any = payload.get("allocation", payload.get("bits"))
    if raw is None and isinstance(payload.get("selected_sites"), list):
        selected = payload["selected_sites"]
        raw = {str(group["group_id"]): (8 if group["group_id"] in selected else 4) for group in groups}
    if not isinstance(raw, Mapping):
        raise ValueError("allocation JSON needs allocation/bits mapping or selected_sites")
    known = {str(group["group_id"]) for group in groups}
    mapping: Dict[str, int] = {}
    for key, value in raw.items():
        key = str(key)
        if key not in known:
            raise ValueError(f"allocation contains unknown group: {key}")
        if isinstance(value, Mapping):
            value = value.get("bits")
        bits = int(value)
        if bits not in (4, 8):
            raise ValueError(f"allocation bit width must be 4 or 8: {key}={bits}")
        mapping[key] = bits
    if set(mapping) != known:
        missing = sorted(known - set(mapping))
        raise ValueError(f"allocation must specify every runtime group; missing {missing}")
    name = str(payload.get("name", payload.get("method", payload.get("label", "allocation"))))
    if not name or name in {"FP32", "all_W4", "all_W8"}:
        name = "allocation"
    metadata = {
        "name": name,
        "selected_sites": sorted(key for key, bits in mapping.items() if bits == 8),
        "w8_counts": {
            "encoder": sum(bits == 8 and key.startswith("encoder.") for key, bits in mapping.items()),
            "predictor": sum(bits == 8 and key.startswith("predictor.") for key, bits in mapping.items()),
        },
        "source": payload,
    }
    return name, mapping, metadata


def _configure_mode(runtime: Mapping[str, Any], mode: str, allocation_json: Path | None, groups: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    model = runtime["model"]
    snapshot = smoke._snapshot_weights(model, groups)
    smoke._restore_weights(model, snapshot)
    metadata: Dict[str, Any] = {"mode": mode, "execution": "emulation_only"}
    if mode == "FP32":
        return mode, metadata
    if mode in {"all_W4", "all_W8"}:
        bits = 4 if mode == "all_W4" else 8
        metadata["bits"] = bits
        metadata["quantization"] = smoke._apply_all_quantized(model, groups, bits)
        return mode, metadata
    if allocation_json is None:
        raise ValueError("--mode allocation requires --allocation-json")
    payload = json.loads(allocation_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("allocation JSON must be an object")
    label, mapping, allocation_meta = _allocation_mapping(payload, groups)
    metadata.update({"allocation": mapping, **allocation_meta})
    records = []
    for group in groups:
        records.append(smoke._quantize_group(model, group, mapping[group["group_id"]]))
    metadata["quantization"] = records
    return label, metadata


def _load_targets(targets_dir: Path, split: str, runtime: Mapping[str, Any]) -> Tuple[Path, Mapping[str, Any], List[Dict[str, Any]]]:
    manifest_path = _target_manifest_path(targets_dir, split)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"frozen target manifest not found: {manifest_path}; run FP32 first or use --prepare-targets-only"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = _validate_manifest(manifest, targets_dir, split, len(runtime["dset"]))
    _validate_runtime_identity(manifest, runtime)
    _validate_cross_split_dedup(targets_dir)
    return manifest_path, manifest, episodes


def _parse_episode_selection(args: argparse.Namespace, episodes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected = list(episodes)
    if args.episode_ids:
        wanted = set()
        for token in args.episode_ids.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                wanted.add(int(token))
            else:
                wanted.add(token)
        selected = [entry for entry in selected if entry["local_index"] in wanted or entry["episode_id"] in wanted]
        if len(selected) != len(wanted):
            raise ValueError(f"unknown --episode-ids; available ids are 0..{len(episodes)-1} or split:NNN")
    else:
        start = max(0, args.case_start)
        end = len(selected) if args.case_count is None else start + max(0, args.case_count)
        if start >= len(selected) and args.case_count not in (None, 0):
            raise ValueError("--case-start is outside the split")
        selected = selected[start:end]
    return [dict(entry) for entry in selected]


def _safe_mode_dir(mode_label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", mode_label)


def _record_paths(run_dir: Path, local_index: int) -> Tuple[Path, Path, Path]:
    stem = f"episode_{local_index:03d}"
    return run_dir / "episodes" / f"{stem}.json", run_dir / "episodes" / f"{stem}.npz", run_dir / "pools" / f"{stem}.npz"


def _stable_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: identity.get(key) for key in (
        "recorded_epoch", "checkpoint_size_bytes", "source_commit",
        "dinov2_source_commit", "dtype", "decoder", "execution", "torch",
    )}


def _existing_complete(
    result_path: Path,
    split: str,
    mode_label: str,
    target_fingerprint: str,
    protocol: Mapping[str, Any],
    mode_metadata: Mapping[str, Any],
    record_pools: bool,
    runtime_identity: Mapping[str, Any],
) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read existing result: {result_path}") from exc
    if result.get("status") != "complete":
        return False
    if result.get("schema") != RESULT_SCHEMA or result.get("split") != split or result.get("mode") != mode_label:
        raise RuntimeError(f"existing complete result has incompatible identity: {result_path}")
    if result.get("target_fingerprint") != target_fingerprint or result.get("protocol") != protocol:
        raise RuntimeError(f"existing complete result has incompatible target/protocol: {result_path}")
    if result.get("mode_metadata") != mode_metadata:
        raise RuntimeError(f"existing complete result has incompatible allocation/mode metadata: {result_path}")
    if bool(result.get("record_pools")) != bool(record_pools):
        raise RuntimeError(f"existing complete result has incompatible pool-recording setting: {result_path}")
    if _stable_identity(result.get("checkpoint_identity", {})) != _stable_identity(runtime_identity):
        raise RuntimeError(f"existing complete result has incompatible runtime identity: {result_path}")
    trajectory_path = result.get("trajectory_npz")
    if not trajectory_path or not Path(trajectory_path).is_file():
        raise RuntimeError(f"existing complete result is missing trajectory evidence: {result_path}")
    if record_pools:
        pool_path = result.get("pool_npz")
        if not pool_path or not Path(pool_path).is_file() or int(result.get("pool_count", 0)) <= 0:
            raise RuntimeError(f"existing complete result is missing requested pool evidence: {result_path}")
    return True


class _PoolRecorder:
    def __init__(self, workspace: Any, episode: Mapping[str, Any], enabled: bool):
        self.workspace = workspace
        self.episode = episode
        self.enabled = enabled
        self.pools: List[Dict[str, Any]] = []

    def record(
        self,
        point_index: int,
        iteration: int,
        candidate: Any,
        scores: Any,
        mu_before: Any,
        sigma_before: Any,
        raw_obs_0: Mapping[str, Any],
        raw_obs_g: Mapping[str, Any],
        trans_obs_0: Mapping[str, Any],
        trans_obs_g: Mapping[str, Any],
    ) -> None:
        if not self.enabled or point_index >= 2 or iteration not in (1, CEM_STEPS):
            return
        import torch

        pool_id = f"{self.episode['episode_id']}/mpc_{point_index:02d}/cem_{iteration:02d}"
        self.pools.append({
            "pool_id": pool_id,
            "mpc_point": point_index,
            "cem_iteration": iteration,
            "candidate_ids": np.arange(int(candidate.shape[0]), dtype=np.int64),
            "candidate_actions": candidate.detach().cpu().numpy().copy(),
            "scores": scores.detach().cpu().numpy().copy(),
            "mu_before": mu_before.detach().cpu().numpy().copy(),
            "sigma_before": sigma_before.detach().cpu().numpy().copy(),
            "obs_0_visual": _to_numpy(raw_obs_0["visual"]),
            "obs_0_proprio": _to_numpy(raw_obs_0["proprio"]),
            "obs_g_visual": _to_numpy(raw_obs_g["visual"]),
            "obs_g_proprio": _to_numpy(raw_obs_g["proprio"]),
            "trans_obs_0_visual": trans_obs_0["visual"].detach().cpu().numpy().copy(),
            "trans_obs_0_proprio": trans_obs_0["proprio"].detach().cpu().numpy().copy(),
            "trans_obs_g_visual": trans_obs_g["visual"].detach().cpu().numpy().copy(),
            "trans_obs_g_proprio": trans_obs_g["proprio"].detach().cpu().numpy().copy(),
            "stable_sort": True,
            "candidate_zero_is_mu": True,
        })


def _instrumented_cem_plan(recorder: _PoolRecorder):
    """Return the smoke adapter with optional first-two-point pool capture."""
    def plan(self: Any, obs_0: Mapping[str, Any], obs_g: Mapping[str, Any], actions: Any = None):
        import torch
        from einops import repeat
        from utils import move_to_device

        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        prefix = str(getattr(self, "logging_prefix", "plan_0"))
        point_index = int(prefix.rsplit("_", 1)[-1]) if prefix.rsplit("_", 1)[-1].isdigit() else 0
        for iteration in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }
                mu_before = mu[traj].detach().clone()
                sigma_before = sigma[traj].detach().clone()
                candidate = torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device) * sigma[traj] + mu[traj]
                candidate[0] = mu[traj]
                with torch.no_grad():
                    imagined, _ = self.wm.rollout(obs_0=cur_trans_obs_0, act=candidate)
                    scores = self.objective_fn(imagined, cur_z_obs_g)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError(f"non-finite CEM score at iteration {iteration + 1}")
                if traj == 0:
                    recorder.record(point_index, iteration + 1, candidate, scores, mu_before, sigma_before, obs_0, obs_g, trans_obs_0, trans_obs_g)
                topk_idx = torch.argsort(scores, stable=True)[: self.topk]
                topk_action = candidate[topk_idx]
                losses.append(float(scores[topk_idx[0]].item()))
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)
            self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": iteration + 1})
        return mu, np.full(n_evals, np.inf)

    return plan


def _capture_evaluator(evaluator: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    original = evaluator.eval_actions
    calls: List[Dict[str, Any]] = []

    def eval_actions(actions: Any, action_len: Any = None, filename: str = "output", save_video: bool = False):
        result = original(actions, action_len, filename=filename, save_video=False)
        logs, successes, e_obses, e_states = result
        calls.append({
            "actions": actions.detach().cpu().numpy().copy(),
            "action_len": None if action_len is None else np.asarray(action_len).copy(),
            "logs": logs,
            "successes": np.asarray(successes).copy(),
            "obses": {key: np.asarray(value).copy() for key, value in e_obses.items()},
            "states": np.asarray(e_states).copy(),
        })
        return result

    evaluator.eval_actions = eval_actions
    return original, calls


def _array_for_target(target: Mapping[str, Any], key: str, field: str) -> np.ndarray:
    return np.asarray(target[key][field]).copy()


def _run_episode(
    runtime: Mapping[str, Any],
    episode: Mapping[str, Any],
    case_dir: Path,
    record_pools: bool,
    mode_label: str,
    mode_metadata: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    case_dir.mkdir(parents=True, exist_ok=True)
    target = episode["target"]
    pool_recorder: _PoolRecorder | None = None
    calls: List[Dict[str, Any]] = []
    env = None
    workspace = None
    previous_cwd = os.getcwd()
    started = time.monotonic()
    try:
        os.chdir(case_dir)
        workspace, env, cfg = smoke._make_workspace(runtime, episode, case_dir)
        workspace.planner.max_iter = MPC_ROUNDS
        workspace.planner.n_taken_actions = EXECUTE_MODEL_ACTIONS
        workspace.planner.sub_planner.opt_steps = CEM_STEPS
        workspace.planner.sub_planner.evaluator = None
        workspace.evaluator.n_plot_samples = 0

        pool_recorder = _PoolRecorder(workspace, episode, record_pools)
        cem_module = __import__("planning.cem", fromlist=["CEMPlanner"])
        original_cem = cem_module.CEMPlanner.plan
        cem_module.CEMPlanner.plan = _instrumented_cem_plan(pool_recorder)
        original_eval, calls = _capture_evaluator(workspace.evaluator)
        smoke._set_seed(int(episode["cem_seed"]))
        planned_actions, action_len = workspace.planner.plan(workspace.obs_0, workspace.obs_g, actions=None)
        # The last outer-MPC evaluator call already rolled out exactly the
        # final action prefix.  Reusing it avoids a duplicate 300-step env run.
        if not calls:
            raise RuntimeError("MPC planner returned without a real environment evaluation")
        final_call = calls[-1]
        action_len_np = np.asarray(action_len).copy()
        planned_np = planned_actions.detach().cpu().numpy().copy()
        preprocessor = workspace.data_preprocessor
        primitive = preprocessor.denormalize_actions(planned_actions.detach().cpu().reshape(1, -1, 2)).numpy()
        states = final_call["states"][0]
        env_obs = final_call["obses"]
        if planned_np.ndim != 3 or planned_np.shape[0] != 1 or planned_np.shape[1] > MPC_ROUNDS * EXECUTE_MODEL_ACTIONS:
            raise RuntimeError(f"planned action shape exceeds fixed budget: {planned_np.shape}")
        if primitive.ndim != 3 or primitive.shape[0] != 1 or primitive.shape[1] > ENV_STEP_LIMIT:
            raise RuntimeError(f"executed action shape exceeds fixed budget: {primitive.shape}")
        if states.ndim != 2 or states.shape[0] != primitive.shape[1] + 1:
            raise RuntimeError(f"state/action trajectory length mismatch: states={states.shape}, actions={primitive.shape}")
        if any(np.asarray(value).shape[1] != states.shape[0] for value in env_obs.values()):
            raise RuntimeError("environment observation/state trajectory length mismatch")
        finite_len = action_len_np[0] != np.inf
        boundary_model_actions = int(action_len_np[0]) if finite_len else planned_np.shape[1]
        boundary_env_steps = boundary_model_actions * FRAMESKIP
        if boundary_model_actions < 0 or boundary_env_steps > ENV_STEP_LIMIT or boundary_env_steps >= states.shape[0]:
            raise RuntimeError(f"invalid success boundary {boundary_model_actions} model actions / {boundary_env_steps} env steps")
        final_state = states[boundary_env_steps]
        eval_result = env.eval_state(np.asarray(target["state_g"]), final_state.reshape(1, -1))
        success = bool(np.asarray(eval_result["success"]).reshape(-1)[0])
        state_dist = float(np.asarray(eval_result["state_dist"]).reshape(-1)[0])
        goal_xy_dist = float(np.linalg.norm(np.asarray(target["state_g"])[0][:2] - final_state[:2]))
        last_call_success = bool(np.asarray(final_call["successes"]).reshape(-1)[0])
        planner_success = bool(np.asarray(workspace.planner.is_success).reshape(-1)[0])
        if success != last_call_success or success != planner_success:
            raise RuntimeError(
                f"success evidence mismatch: env={success}, last_call={last_call_success}, planner={planner_success}"
            )
        # Keep the completion marker beside the episode directory so resume
        # checks can find it without scanning nested files.
        result_path = case_dir.parent / f"{case_dir.name}.json"
        trajectory_path = case_dir / "trajectory.npz"
        _atomic_npz(
            trajectory_path,
            planned_actions_normalized=planned_np,
            actions_executed=primitive[0],
            actions_executed_prefix=primitive[0][:boundary_env_steps],
            states=states,
            obs_0_visual=_array_for_target(target, "obs_0", "visual"),
            obs_0_proprio=_array_for_target(target, "obs_0", "proprio"),
            goal_visual=_array_for_target(target, "obs_g", "visual"),
            goal_proprio=_array_for_target(target, "obs_g", "proprio"),
            final_visual=env_obs["visual"][0, boundary_env_steps],
            final_proprio=env_obs["proprio"][0, boundary_env_steps],
        )
        if record_pools and pool_recorder is not None and pool_recorder.pools:
            arrays: Dict[str, Any] = {}
            pool_meta = []
            workload_cases = []
            for index, pool in enumerate(pool_recorder.pools):
                prefix = f"pool_{index:02d}"
                for key, value in pool.items():
                    if key in {"pool_id", "mpc_point", "cem_iteration", "stable_sort", "candidate_zero_is_mu"}:
                        continue
                    arrays[f"{prefix}_{key}"] = value
                pool_meta.append({
                    "pool_index": index,
                    "pool_id": pool["pool_id"],
                    "mpc_point": pool["mpc_point"],
                    "cem_iteration": pool["cem_iteration"],
                    "stable_sort": True,
                    "candidate_zero_is_mu": True,
                })
                workload_cases.append({
                    # ``case_id`` is deliberately the unique pool ID.  The
                    # parent probe runner can therefore identify every pool
                    # without guessing from the episode or CEM indices.
                    "case_id": pool["pool_id"],
                    "pool_id": pool["pool_id"],
                    "episode_id": episode["episode_id"],
                    "mpc_point": pool["mpc_point"],
                    "cem_iteration": pool["cem_iteration"],
                    "env_seed": episode["env_seed"],
                    "cem_seed": episode["cem_seed"],
                    "dataset_index": episode["dataset_index"],
                    "layout": episode["layout"],
                    "obs_0": {"visual": pool["obs_0_visual"], "proprio": pool["obs_0_proprio"]},
                    "obs_g": {"visual": pool["obs_g_visual"], "proprio": pool["obs_g_proprio"]},
                    "candidates": pool["candidate_actions"],
                    "reference_scores": pool["scores"],
                    "scores": {"FP32": pool["scores"]},
                })
            pool_path = case_dir / "pools.npz"
            _atomic_npz(pool_path, **arrays)
            _atomic_json(case_dir / "pools.json", {"schema": "rankcal-wall-pool-record-v1", "pools": pool_meta, "array_file": str(pool_path)})
            _atomic_pickle(
                case_dir / "workload.pkl",
                {
                    "schema": "rankcal-wall-screen-workload-v1",
                    "split": {"calibration": "cal", "development": "dev"}.get(episode["split"], episode["split"]),
                    "screen_split": episode["split"],
                    "reference_score_field": "reference_scores",
                    "planner": _protocol_config(),
                    "cases": workload_cases,
                },
            )
        else:
            pool_meta = []
            pool_path = None
            workload_cases = []

        result = {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "split": episode["split"],
            "mode": mode_label,
            "episode_id": episode["episode_id"],
            "local_index": episode["local_index"],
            "dataset_index": episode["dataset_index"],
            "env_seed": episode["env_seed"],
            "cem_seed": episode["cem_seed"],
            "layout": episode["layout"],
            "target_fingerprint": episode["target_fingerprint"],
            "protocol": _protocol_config(),
            "mode_metadata": mode_metadata,
            "checkpoint_identity": smoke._checkpoint_identity(runtime),
            "record_pools": bool(record_pools),
            "success": success,
            "success_source": "real_env_eval_state",
            "state_dist": state_dist,
            "goal_xy_distance": goal_xy_dist,
            "action_len_model_actions": None if not finite_len else int(action_len_np[0]),
            "executed_env_steps": boundary_env_steps,
            "planned_model_actions": int(planned_np.shape[1]),
            "mpc_points_visited": len(calls),
            "trajectory_npz": str(trajectory_path.resolve()),
            "pool_npz": None if pool_path is None else str(pool_path.resolve()),
            "pool_count": len(pool_meta),
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(result_path, result)
        return result, pool_meta, workload_cases
    except Exception as exc:
        error = {
            "schema": RESULT_SCHEMA,
            "status": "failed",
            "split": episode["split"],
            "mode": mode_label,
            "episode_id": episode["episode_id"],
            "local_index": episode["local_index"],
            "dataset_index": episode["dataset_index"],
            "env_seed": episode["env_seed"],
            "cem_seed": episode["cem_seed"],
            "layout": episode["layout"],
            "target_fingerprint": episode["target_fingerprint"],
            "protocol": _protocol_config(),
            "checkpoint_identity": smoke._checkpoint_identity(runtime),
            "record_pools": bool(record_pools),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "mpc_points_visited": len(calls),
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(case_dir.parent / f"{case_dir.name}.json", error)
        return error, [], []
    finally:
        try:
            if workspace is not None:
                cem_module = __import__("planning.cem", fromlist=["CEMPlanner"])
                # Restore the source method if it was replaced for pool capture.
                if "original_cem" in locals():
                    cem_module.CEMPlanner.plan = original_cem
                if "original_eval" in locals():
                    workspace.evaluator.eval_actions = original_eval
        finally:
            smoke._close_env(env)
            os.chdir(previous_cwd)


def _run_screen(args: argparse.Namespace) -> None:
    runtime = smoke._runtime(args.root)
    targets_dir = (args.targets_dir or (args.output / "targets")).resolve()
    manifest_path, manifest, all_episodes = _load_targets(targets_dir, args.split, runtime)
    episodes = _parse_episode_selection(args, all_episodes)
    groups = smoke._linear_groups(runtime["model"])
    mode_label, mode_metadata = _configure_mode(runtime, args.mode, args.allocation_json, groups)
    if args.record_pools and mode_label != "FP32":
        raise ValueError("--record-pools is only valid for --mode FP32")
    run_dir = args.output.resolve() / args.split / _safe_mode_dir(mode_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol_config()
    completed = 0
    skipped = 0
    failed = 0
    records: List[Dict[str, Any]] = []
    workload_cases: List[Dict[str, Any]] = []
    stopped_on_failure = False
    started = time.monotonic()
    snapshot = smoke._snapshot_weights(runtime["model"], groups)
    runtime_identity = smoke._checkpoint_identity(runtime)
    effective_record_pools = bool(args.record_pools and mode_label == "FP32")
    try:
        for episode in episodes:
            result_path, _, _ = _record_paths(run_dir, episode["local_index"])
            if _existing_complete(
                result_path,
                args.split,
                mode_label,
                episode["target_fingerprint"],
                protocol,
                mode_metadata,
                effective_record_pools,
                runtime_identity,
            ):
                skipped += 1
                records.append(json.loads(result_path.read_text(encoding="utf-8")))
                workload_path = run_dir / "episodes" / f"episode_{episode['local_index']:03d}" / "workload.pkl"
                if effective_record_pools and workload_path.is_file():
                    with workload_path.open("rb") as stream:
                        workload_cases.extend(pickle.load(stream).get("cases", []))
                continue
            smoke._restore_weights(runtime["model"], snapshot)
            result, _, episode_workload = _run_episode(
                runtime,
                episode,
                run_dir / "episodes" / f"episode_{episode['local_index']:03d}",
                effective_record_pools,
                mode_label,
                mode_metadata,
            )
            records.append(result)
            workload_cases.extend(episode_workload)
            if result.get("status") == "complete":
                completed += 1
            else:
                failed += 1
                stopped_on_failure = True
                break
    finally:
        smoke._restore_weights(runtime["model"], snapshot)

    workload_path = None
    if effective_record_pools and workload_cases:
        workload_path = run_dir / "workload.pkl"
        _atomic_pickle(
            workload_path,
            {
                "schema": "rankcal-wall-screen-workload-v1",
                "split": {"calibration": "cal", "development": "dev"}.get(args.split, args.split),
                "screen_split": args.split,
                "reference_score_field": "reference_scores",
                "planner": protocol,
                "cases": workload_cases,
            },
        )
    status = "complete" if failed == 0 else ("partial" if completed or skipped else "failed")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": status,
        "screen_schema": SCREEN_SCHEMA,
        "split": args.split,
        "mode": mode_label,
        "requested_episode_count": len(episodes),
        "newly_completed": completed,
        "skipped_complete": skipped,
        "failed": failed,
        "target_manifest": str(manifest_path.resolve()),
        "target_manifest_schema": manifest.get("schema"),
        "target_manifest_identity": _digest_arrays(json.dumps(manifest, sort_keys=True).encode("utf-8")),
        "protocol": protocol,
        "mode_metadata": mode_metadata,
        "record_pools": effective_record_pools,
        "workload_path": None if workload_path is None else str(workload_path.resolve()),
        "stopped_on_failure": stopped_on_failure,
        "episodes": records,
        "elapsed_seconds": time.monotonic() - started,
        "unresolved": [
            "fake quantization executes FP32 operators and cannot support native kernel, memory or speed claims",
            "this fast screen is not a CEM10 or confirmatory benchmark",
        ],
    }
    _atomic_json(run_dir / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "split", "mode", "requested_episode_count", "newly_completed", "skipped_complete", "failed")}, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"{failed} episode(s) failed; results were preserved for resume")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM Wall runtime root")
    parser.add_argument("--output", type=Path, required=True, help="artifact root; results go under split/mode")
    parser.add_argument("--split", choices=tuple(SPLIT_SIZES), required=True, help="fixed screen split")
    parser.add_argument("--mode", choices=("FP32", "all_W4", "all_W8", "allocation"), default=None, help="model configuration")
    parser.add_argument("--allocation-json", type=Path, default=None, help="complete group_id to 4/8 mapping; implies allocation mode when --mode is omitted")
    parser.add_argument("--targets-dir", type=Path, default=None, help="shared frozen target directory")
    parser.add_argument("--prepare-targets-only", action="store_true", help="create/validate frozen targets and exit")
    parser.add_argument("--record-pools", action="store_true", help="FP32 only: save first two actual MPC points, CEM iterations 1/5")
    parser.add_argument("--case-start", type=int, default=0, help="local episode index for bounded preflight/resume")
    parser.add_argument("--case-count", type=int, default=None, help="number of local episodes to run")
    parser.add_argument("--episode-ids", default=None, help="comma-separated local indices or split:NNN ids")
    args = parser.parse_args()
    if args.mode is None:
        args.mode = "allocation" if args.allocation_json is not None else "FP32"
    if args.mode == "allocation" and args.allocation_json is None and not args.prepare_targets_only:
        parser.error("--mode allocation requires --allocation-json")
    if args.mode != "allocation" and args.allocation_json is not None:
        parser.error("--allocation-json requires --mode allocation (or omit --mode)")
    if args.case_start < 0 or (args.case_count is not None and args.case_count < 0):
        parser.error("--case-start and --case-count must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    if args.prepare_targets_only:
        runtime = smoke._runtime(args.root)
        targets_dir = (args.targets_dir or (args.output / "targets")).resolve()
        manifest = _prepare_targets(runtime, targets_dir, args.split)
        print(json.dumps({"status": "complete", "mode": "prepare_targets", "split": args.split, "manifest": str(_target_manifest_path(targets_dir, args.split))}, indent=2), flush=True)
        return
    _run_screen(args)


if __name__ == "__main__":
    main()
