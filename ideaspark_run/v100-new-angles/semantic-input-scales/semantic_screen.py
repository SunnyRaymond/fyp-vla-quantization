"""Bounded Semantic Input Scales screen for the frozen DINO-WM Wall model.

This is a compute-allocation-only, fake-quantization screen.  It quantizes the
input to the predictor (before the predictor's positional addition) and keeps
the 24 predictor Linear weights on the reviewed W4 per-output-channel RTN path.
The encoder, planner, objective, action pool, and target states are unchanged.

The first workload action in :func:`main` is the campaign allocation guard.
Consequently importing this file, parsing arguments, and static compilation are
safe on a workstation, while model loading, dataset access, hashing, and all
numerical work require a verified CCDS SLURM compute allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "semantic-input-scales-screen-v1"
TARGET_SCHEMA = "semantic-input-scales-targets-v1"
RAW_SCHEMA = "semantic-input-scales-raw-v1"
HORIZON = 5
ACTION_DIM = 10
INPUT_DIM = 404
VISUAL_DIM = 384
PROPRIO_DIM = 10
ACTION_EMB_DIM = 10
CAL_DATASET_INDICES = tuple(range(108, 112))
DEV_DATASET_INDICES = tuple(range(112, 118))
CAL_ENV_NAMESPACE = 950000
DEV_ENV_NAMESPACE = 960000
CAL_CANDIDATE_NAMESPACE = CAL_ENV_NAMESPACE + 10000
DEV_CANDIDATE_NAMESPACE = DEV_ENV_NAMESPACE + 10000
CAL_CANDIDATES = 32
DEV_CANDIDATES = 64
TOPK = 6
BITS = 4
WEIGHT_QMAX = 7
ACTIVATION_QMAX = 127
PERMUTATION_SEED = 1501
NMSE_EPS = 1e-12
NOOP_TOLERANCE = 1e-12
PRIMARY_STRICT_TOLERANCE = 1e-12
GATE_IMPROVEMENT = 0.05
MAX_WORKLOAD_SECONDS = 2400.0

ARMS = (
    "FP32",
    "W4A32",
    "FullInput-A8",
    "Semantic-A8",
    "Permuted-A8",
    "PerChannel-A8",
)

# These identities are frozen by the reviewed antithetic-rounding reference.
# A mismatch is an implementation/source identity failure, never a scientific
# negative result.
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


def _write_small_summary(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the controller-readable summary while keeping it below 64 KiB."""
    encoded = json.dumps(payload, indent=2, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds the controller's 64 KiB read limit ({len(encoded)} bytes)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
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
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_allocation_guard() -> Any:
    """Resolve the campaign guard without importing heavy runtime packages."""
    try:
        module = importlib.import_module("allocation_guard")
        return module.require_allocation
    except ModuleNotFoundError:
        candidates = (
            Path(__file__).with_name("allocation_guard.py"),
            Path(__file__).resolve().parents[1] / "allocation_guard.py",
        )
        for guard_path in candidates:
            if not guard_path.is_file():
                continue
            spec = importlib.util.spec_from_file_location("semantic_allocation_guard", guard_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.require_allocation
        raise ImportError("campaign allocation_guard.py is unavailable")


def _load_helpers() -> Tuple[Any, Any]:
    """Load the reviewed helper pair copied by the campaign controller."""
    try:
        smoke = importlib.import_module("smoke_runner")
        screen = importlib.import_module("screen_runner")
        return smoke, screen
    except ModuleNotFoundError:
        helper_dirs = [
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parents[2]
            / "world-model-quantization"
            / "experiments"
            / "dino-wm-wall",
        ]
        for helper_dir in helper_dirs:
            smoke_path = helper_dir / "smoke_runner.py"
            screen_path = helper_dir / "screen_runner.py"
            if not smoke_path.is_file() or not screen_path.is_file():
                continue
            if str(helper_dir) not in sys.path:
                sys.path.insert(0, str(helper_dir))
            smoke = importlib.import_module("smoke_runner")
            screen = importlib.import_module("screen_runner")
            return smoke, screen
        raise ImportError("reviewed smoke_runner.py/screen_runner.py pair is unavailable")


def _assert_helper_identity(smoke: Any, screen: Any) -> Dict[str, str]:
    actual = {
        "smoke_runner_sha256": _sha256_file(Path(smoke.__file__).resolve()),
        "screen_runner_sha256": _sha256_file(Path(screen.__file__).resolve()),
    }
    expected = {
        "smoke_runner_sha256": EXPECTED_SMOKE_RUNNER_SHA256,
        "screen_runner_sha256": EXPECTED_SCREEN_RUNNER_SHA256,
    }
    for key, frozen in expected.items():
        if actual.get(key) != frozen:
            raise RuntimeError(
                f"{key} does not match the frozen reviewed helper: "
                f"got {actual.get(key)!r}, expected {frozen!r}"
            )
    return {key: str(value) for key, value in actual.items()}


def _check_deadline(started: float, max_seconds: float, label: str) -> None:
    if time.monotonic() - started >= max_seconds:
        raise TimeoutError(f"max-seconds reached at {label}")


def _model_structure(runtime: Mapping[str, Any]) -> Dict[str, Any]:
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
        "proprio_dim": value(model, cfg, name="proprio_dim"),
        "action_dim": value(model, cfg, name="action_dim"),
        "model_emb_dim": value(model, name="emb_dim"),
        "encoder_emb_dim": value(encoder, name="emb_dim"),
        "predictor_type": type(getattr(model, "predictor", None)).__name__,
    }


def _assert_model_structure(structure: Mapping[str, Any]) -> None:
    required = {
        "num_hist": 1,
        "concat_dim": 1,
        "proprio_dim": PROPRIO_DIM,
        "action_dim": ACTION_EMB_DIM,
        "encoder_emb_dim": VISUAL_DIM,
    }
    for key, expected in required.items():
        if structure.get(key) != expected:
            raise RuntimeError(
                f"structural no-go: live {key}={structure.get(key)!r}, expected {expected!r}"
            )
    if structure.get("model_emb_dim") not in (None, INPUT_DIM):
        raise RuntimeError(
            f"structural no-go: live model_emb_dim={structure.get('model_emb_dim')!r}, expected 404"
        )


def _runtime_identity(runtime: Mapping[str, Any], smoke: Any) -> Dict[str, Any]:
    identity = dict(smoke._checkpoint_identity(runtime))
    checkpoint = Path(runtime["checkpoint"]).resolve()
    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_sha256 is None:
        raise FileNotFoundError(f"checkpoint is not hashable: {checkpoint}")
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "checkpoint SHA-256 does not match the frozen source identity: "
            f"got {checkpoint_sha256}, expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    identity["checkpoint_sha256"] = checkpoint_sha256
    source_file = Path(runtime["root"]) / "source" / "models" / "visual_world_model.py"
    identity["visual_world_model_source"] = {
        "declared_repo_path": "reproduction/dino-wm-wall/source/models/visual_world_model.py",
        **_file_identity(source_file),
    }
    return identity


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
    }


def _dataset_mapping_audit(dset: Any) -> Dict[str, Any]:
    """Map only the ten used positions through Subset metadata.

    This deliberately never indexes a sample in the protected 84..95 range.
    """
    positions = list(CAL_DATASET_INDICES + DEV_DATASET_INDICES)
    current = dset
    layers: List[Dict[str, Any]] = []
    while hasattr(current, "indices") and hasattr(current, "dataset"):
        raw_indices = list(getattr(current, "indices"))
        if any(position >= len(raw_indices) for position in positions):
            raise RuntimeError("dataset subset metadata cannot map a requested position")
        positions = [int(raw_indices[position]) for position in positions]
        next_dataset = getattr(current, "dataset")
        layers.append({
            "type": type(current).__name__,
            "dataset_type": type(next_dataset).__name__,
            "index_count": len(raw_indices),
            "metadata_only": True,
        })
        current = next_dataset
    if len(set(positions)) != len(positions):
        raise RuntimeError("requested CAL/DEV positions map to duplicate underlying episode ids")
    return {
        "dataset_type_after_subset_resolution": type(current).__name__,
        "subset_layers": layers,
        "requested_positions": list(CAL_DATASET_INDICES + DEV_DATASET_INDICES),
        "underlying_ids": positions,
        "requested_distinct": True,
        "protected_positions": [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
        "protected_state_reads": False,
    }


def _target_manifest_path(targets_dir: Path, split: str) -> Path:
    return targets_dir / f"episode_manifest_{split}.json"


def _target_path(targets_dir: Path, split: str, local_index: int) -> Path:
    return targets_dir / split / f"episode_{local_index:03d}.pkl"


def _target_fingerprint(screen: Any, target: Mapping[str, Any], layout: Mapping[str, Any]) -> str:
    return screen._target_fingerprint(target, layout)


def _prepare_targets(
    runtime: Mapping[str, Any],
    targets_dir: Path,
    screen: Any,
    runtime_identity: Mapping[str, Any],
    structure: Mapping[str, Any],
    split: str,
    dataset_indices: Sequence[int],
    env_namespace: int,
    candidate_namespace: int,
) -> Mapping[str, Any]:
    """Create or validate one disjoint target manifest under an exclusive lock."""
    targets_dir = targets_dir.resolve()
    targets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _target_manifest_path(targets_dir, split)
    expected_indices = list(dataset_indices)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _validate_targets(
            manifest, targets_dir, runtime, screen, runtime_identity, structure, split,
            expected_indices, env_namespace, candidate_namespace,
        )

    lock_path = targets_dir / f".{manifest_path.name}.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError(f"target generation is already in progress: {lock_path}") from exc
    try:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return _validate_targets(
                manifest, targets_dir, runtime, screen, runtime_identity, structure, split,
                expected_indices, env_namespace, candidate_namespace,
            )
        episodes: List[Dict[str, Any]] = []
        for local_index, dataset_index in enumerate(expected_indices):
            env_seed = env_namespace + local_index
            target, layout = screen._new_target(runtime, int(dataset_index), env_seed)
            fingerprint = _target_fingerprint(screen, target, layout)
            target["target_fingerprint"] = fingerprint
            path = _target_path(targets_dir, split, local_index)
            _atomic_pickle(path, target)
            episodes.append({
                "episode_id": f"{split}:{local_index:03d}",
                "split": split,
                "local_index": local_index,
                "dataset_index": int(dataset_index),
                "env_seed": env_seed,
                "candidate_seed": candidate_namespace + local_index,
                "layout": dict(layout),
                "target_fingerprint": fingerprint,
                "target_path": str(path.relative_to(targets_dir)),
            })
        manifest = {
            "schema": TARGET_SCHEMA,
            "split": split,
            "episode_count": len(expected_indices),
            "dataset_indices": expected_indices,
            "env_seed_namespace": env_namespace,
            "candidate_seed_namespace": candidate_namespace,
            "normalization_identity": screen._normalization_identity(runtime),
            "checkpoint_identity": dict(runtime_identity),
            "model_structure": dict(structure),
            "episodes": episodes,
        }
        _atomic_json(manifest_path, manifest)
        return _validate_targets(
            manifest, targets_dir, runtime, screen, runtime_identity, structure, split,
            expected_indices, env_namespace, candidate_namespace,
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _validate_targets(
    manifest: Mapping[str, Any],
    targets_dir: Path,
    runtime: Mapping[str, Any],
    screen: Any,
    runtime_identity: Mapping[str, Any],
    structure: Mapping[str, Any],
    split: str,
    expected_indices: Sequence[int],
    env_namespace: int,
    candidate_namespace: int,
) -> Mapping[str, Any]:
    if manifest.get("schema") != TARGET_SCHEMA or manifest.get("split") != split:
        raise RuntimeError(f"unsupported target manifest for {split}")
    if list(manifest.get("dataset_indices", [])) != list(expected_indices):
        raise RuntimeError(f"{split} target dataset indices are not frozen")
    if manifest.get("episode_count") != len(expected_indices):
        raise RuntimeError(f"{split} target count mismatch")
    if manifest.get("env_seed_namespace") != env_namespace:
        raise RuntimeError(f"{split} environment namespace mismatch")
    if manifest.get("candidate_seed_namespace") != candidate_namespace:
        raise RuntimeError(f"{split} candidate namespace mismatch")
    if manifest.get("model_structure") != dict(structure):
        raise RuntimeError(f"{split} target model structure mismatch")
    if manifest.get("normalization_identity") != screen._normalization_identity(runtime):
        raise RuntimeError(f"{split} target normalization identity mismatch")
    recorded = manifest.get("checkpoint_identity", {})
    for key in ("recorded_epoch", "checkpoint_size_bytes", "checkpoint_sha256", "source_commit", "dinov2_source_commit", "dtype", "decoder", "execution", "torch"):
        if recorded.get(key) != runtime_identity.get(key):
            raise RuntimeError(f"{split} target checkpoint identity mismatch at {key!r}")
    for key in ("sha256", "size_bytes"):
        if recorded.get("visual_world_model_source", {}).get(key) != runtime_identity.get("visual_world_model_source", {}).get(key):
            raise RuntimeError(f"{split} target source identity mismatch at {key!r}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(expected_indices):
        raise RuntimeError(f"{split} target episodes are incomplete")
    loaded: List[Dict[str, Any]] = []
    fingerprints: set[str] = set()
    for local_index, entry in enumerate(episodes):
        if entry.get("episode_id") != f"{split}:{local_index:03d}":
            raise RuntimeError(f"{split} target episode id mismatch at {local_index}")
        if int(entry.get("dataset_index", -1)) != int(expected_indices[local_index]):
            raise RuntimeError(f"{split} dataset index mismatch at {local_index}")
        if int(entry.get("env_seed", -1)) != env_namespace + local_index:
            raise RuntimeError(f"{split} environment seed mismatch at {local_index}")
        if int(entry.get("candidate_seed", -1)) != candidate_namespace + local_index:
            raise RuntimeError(f"{split} candidate seed mismatch at {local_index}")
        path = targets_dir / str(entry.get("target_path", ""))
        if not path.is_file():
            raise RuntimeError(f"{split} target file is missing: {path}")
        with path.open("rb") as stream:
            target = pickle.load(stream)
        if target.get("goal_H") != HORIZON:
            raise RuntimeError(f"{split} target horizon mismatch at {local_index}")
        actual = _target_fingerprint(screen, target, entry.get("layout", {}))
        if actual != entry.get("target_fingerprint") or target.get("target_fingerprint") != actual:
            raise RuntimeError(f"{split} target fingerprint mismatch at {local_index}")
        if actual in fingerprints:
            raise RuntimeError(f"duplicate {split} target fingerprint at {local_index}")
        fingerprints.add(actual)
        loaded.append({**dict(entry), "target_path": str(path.resolve()), "target": target})
    return {**dict(manifest), "episodes": loaded}


def _normalization_matches(runtime: Mapping[str, Any], screen: Any, manifest: Mapping[str, Any]) -> None:
    current = screen._normalization_identity(runtime)
    if manifest.get("normalization_identity") != current:
        raise RuntimeError(f"{manifest.get('split')} target normalization identity mismatch")


def _candidate_pool(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((count, HORIZON, ACTION_DIM)).astype(np.float32)


def _stable_top(scores: np.ndarray) -> np.ndarray:
    indices = np.arange(scores.shape[0], dtype=np.int64)
    return np.lexsort((indices, scores))[:TOPK].astype(np.int64)


def _score_pool(smoke: Any, runtime: Mapping[str, Any], preprocessor: Any, objective: Any, target: Mapping[str, Any], candidates: np.ndarray) -> np.ndarray:
    scores = smoke._score_pool(runtime["model"], preprocessor, objective, target, candidates)
    result = scores.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    if result.shape != (candidates.shape[0],) or not np.isfinite(result).all():
        raise FloatingPointError("non-finite or malformed terminal objective scores")
    return result.copy()


def _input_groups() -> List[Dict[str, Any]]:
    return [
        {"name": "visual", "indices": np.arange(0, VISUAL_DIM, dtype=np.int64)},
        {"name": "proprio", "indices": np.arange(VISUAL_DIM, VISUAL_DIM + PROPRIO_DIM, dtype=np.int64)},
        {"name": "action", "indices": np.arange(VISUAL_DIM + PROPRIO_DIM, INPUT_DIM, dtype=np.int64)},
    ]


def _scale_vectors(cal_absmax: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if cal_absmax.shape != (INPUT_DIM,) or not np.isfinite(cal_absmax).all():
        raise RuntimeError("calibration absmax does not have the live 404-channel shape")
    semantic_groups = _input_groups()
    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(INPUT_DIM).astype(np.int64)
    perm_groups = [
        {"name": "permuted_visual_block", "indices": permutation[:VISUAL_DIM]},
        {"name": "permuted_proprio_block", "indices": permutation[VISUAL_DIM:VISUAL_DIM + PROPRIO_DIM]},
        {"name": "permuted_action_block", "indices": permutation[VISUAL_DIM + PROPRIO_DIM:]},
    ]

    def group_scale(indices: np.ndarray) -> float:
        value = float(np.max(cal_absmax[indices]))
        return 1.0 if value == 0.0 else value / float(ACTIVATION_QMAX)

    full_scale = group_scale(np.arange(INPUT_DIM, dtype=np.int64))
    semantic_scales = np.ones(INPUT_DIM, dtype=np.float64)
    semantic_scale_defs: List[Dict[str, Any]] = []
    for group in semantic_groups:
        scale = group_scale(group["indices"])
        semantic_scales[group["indices"]] = scale
        semantic_scale_defs.append({"name": group["name"], "indices": group["indices"].tolist(), "scale": scale})
    permuted_scales = np.ones(INPUT_DIM, dtype=np.float64)
    permuted_scale_defs: List[Dict[str, Any]] = []
    for group in perm_groups:
        scale = group_scale(group["indices"])
        permuted_scales[group["indices"]] = scale
        permuted_scale_defs.append({"name": group["name"], "indices": group["indices"].tolist(), "scale": scale})
    per_channel = np.where(cal_absmax == 0.0, 1.0, cal_absmax / float(ACTIVATION_QMAX))
    scales = {
        "FullInput-A8": np.full(INPUT_DIM, full_scale, dtype=np.float64),
        "Semantic-A8": semantic_scales,
        "Permuted-A8": permuted_scales,
        "PerChannel-A8": per_channel.astype(np.float64, copy=False),
    }
    definitions = {
        "activation": {
            "dtype": "float32_emulation",
            "bits": 8,
            "qmin": -ACTIVATION_QMAX,
            "qmax": ACTIVATION_QMAX,
            "symmetric": True,
            "zero_range_scale": 1.0,
            "rounding": "torch.round (round-to-nearest-even)",
            "calibration_source": "W4A32 predictor-input pre-hook, CAL only",
            "dev_reestimation": False,
        },
        "full_group": {"name": "all", "start": 0, "stop": INPUT_DIM, "scale": full_scale},
        "semantic_groups": semantic_scale_defs,
        "permuted_groups": permuted_scale_defs,
        "permutation_seed": PERMUTATION_SEED,
        "permutation": permutation.tolist(),
        "per_channel_scale_digest": _digest_arrays(per_channel),
        "scales_by_channel": {key: value.tolist() for key, value in scales.items()},
    }
    return scales, definitions


class _InputHook:
    """Read-only predictor-input hook with optional fixed A8 replacement."""

    def __init__(self, torch: Any, mode: str, scales: np.ndarray | None):
        self.torch = torch
        self.mode = mode
        self.scales = scales
        self.handle = None
        self.calls = 0
        self.shapes: List[List[int]] = []
        self.raw_numel = 0
        self.clipped_numel = 0
        self.error_sq = 0.0
        self.modality_error_sq = {"visual": 0.0, "proprio": 0.0, "action": 0.0}
        self.modality_numel = {"visual": 0, "proprio": 0, "action": 0}
        self.modality_clipped_numel = {"visual": 0, "proprio": 0, "action": 0}
        self.raw_absmax: np.ndarray | None = None

    def attach(self, predictor: Any) -> None:
        self.handle = predictor.register_forward_pre_hook(self._forward_pre_hook)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _forward_pre_hook(self, _module: Any, args: Tuple[Any, ...]) -> Tuple[Any, ...] | None:
        if not args:
            return None
        x = args[0]
        if not hasattr(x, "ndim") or int(x.ndim) != 3 or int(x.shape[-1]) != INPUT_DIM:
            raise RuntimeError(f"predictor input must be [B,P,404], got {getattr(x, 'shape', None)!r}")
        self.calls += 1
        self.shapes.append([int(value) for value in x.shape])
        raw = x.detach()
        absmax = raw.abs().amax(dim=(0, 1)).float().cpu().numpy()
        self.raw_absmax = absmax if self.raw_absmax is None else np.maximum(self.raw_absmax, absmax)
        if self.scales is None:
            return args
        scale = self.torch.as_tensor(self.scales, dtype=raw.dtype, device=raw.device).reshape(1, 1, INPUT_DIM)
        quantized = self.torch.clamp(self.torch.round(raw / scale), -ACTIVATION_QMAX, ACTIVATION_QMAX) * scale
        diff = quantized - raw
        self.raw_numel += int(raw.numel())
        self.error_sq += float(self.torch.square(diff).sum().item())
        self.clipped_numel += int((raw.abs() > scale * ACTIVATION_QMAX).sum().item())
        for name, start, stop in (("visual", 0, VISUAL_DIM), ("proprio", VISUAL_DIM, VISUAL_DIM + PROPRIO_DIM), ("action", VISUAL_DIM + PROPRIO_DIM, INPUT_DIM)):
            modality_raw = raw[..., start:stop]
            modality_diff = diff[..., start:stop]
            modality_scale = scale[..., start:stop]
            self.modality_numel[name] += int(modality_raw.numel())
            self.modality_error_sq[name] += float(self.torch.square(modality_diff).sum().item())
            self.modality_clipped_numel[name] += int((modality_raw.abs() > modality_scale * ACTIVATION_QMAX).sum().item())
        return (quantized,) + tuple(args[1:])

    def evidence(self) -> Dict[str, Any]:
        if self.scales is None:
            return {
                "mode": self.mode,
                "calls": self.calls,
                "shapes": self.shapes,
                "raw_absmax": None if self.raw_absmax is None else self.raw_absmax.tolist(),
            }
        return {
            "mode": self.mode,
            "calls": self.calls,
            "shapes": self.shapes,
            "raw_numel": self.raw_numel,
            "input_quant_mse": self.error_sq / max(self.raw_numel, 1),
            "clip_ratio": self.clipped_numel / max(self.raw_numel, 1),
            "by_modality": {
                name: {
                    "input_quant_mse": self.modality_error_sq[name] / max(self.modality_numel[name], 1),
                    "clip_ratio": self.modality_clipped_numel[name] / max(self.modality_numel[name], 1),
                    "numel": self.modality_numel[name],
                }
                for name in ("visual", "proprio", "action")
            },
        }


def _quantize_predictor_weights(smoke: Any, model: Any, predictor_groups: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for group in predictor_groups:
        records.append(smoke._quantize_group(model, group, BITS))
    return records


def _write_raw(output: Path, actions: np.ndarray, scores: np.ndarray, completed: np.ndarray) -> None:
    top_indices = np.full((scores.shape[0], scores.shape[1], TOPK), -1, dtype=np.int64)
    for episode in range(scores.shape[0]):
        for arm in range(scores.shape[1]):
            if np.isfinite(scores[episode, arm]).all():
                top_indices[episode, arm] = _stable_top(scores[episode, arm])
    _atomic_npz(
        output / "raw_scores_actions.npz",
        schema=np.asarray(RAW_SCHEMA),
        arm_names=np.asarray(ARMS),
        dataset_indices=np.asarray(DEV_DATASET_INDICES, dtype=np.int64),
        actions=actions,
        scores=scores,
        top_indices=top_indices,
        completed=completed,
    )


def _metrics(actions: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    episode_rows: List[Dict[str, Any]] = []
    primary = np.full((len(DEV_DATASET_INDICES), len(ARMS)), np.nan, dtype=np.float64)
    for episode in range(len(DEV_DATASET_INDICES)):
        fp_top = _stable_top(scores[episode, 0])
        fp_score_mean = float(scores[episode, 0, fp_top].mean())
        row: Dict[str, Any] = {
            "episode_index": episode,
            "dataset_index": int(DEV_DATASET_INDICES[episode]),
            "fp_top6_indices": fp_top.tolist(),
            "arms": {},
        }
        fp_first_mean = actions[episode, fp_top, 0, :].mean(axis=0)
        fp_full_mean = actions[episode, fp_top].mean(axis=0)
        for arm_index, arm in enumerate(ARMS):
            top = _stable_top(scores[episode, arm_index])
            first_mean = actions[episode, top, 0, :].mean(axis=0)
            full_mean = actions[episode, top].mean(axis=0)
            first_mse = float(np.mean(np.square(first_mean - fp_first_mean)))
            full_mse = float(np.mean(np.square(full_mean - fp_full_mean)))
            fp_regret_raw = float(scores[episode, 0, top].mean() - fp_score_mean)
            if fp_regret_raw < -PRIMARY_STRICT_TOLERANCE:
                raise RuntimeError(
                    "FP-score shortlist regret is materially negative; "
                    f"episode={episode}, arm={arm}, value={fp_regret_raw:g}"
                )
            score_nmse = float(
                np.mean(np.square(scores[episode, arm_index] - scores[episode, 0]))
                / max(float(np.var(scores[episode, 0])), NMSE_EPS)
            )
            primary[episode, arm_index] = first_mse
            row["arms"][arm] = {
                "top6_indices": top.tolist(),
                "primary_first_action_mse": first_mse,
                "full_horizon_elite_mean_mse": full_mse,
                "one_shot_top6_fp_score_regret": fp_regret_raw,
                "one_shot_top6_fp_score_regret_raw": fp_regret_raw,
                "score_nmse": score_nmse,
            }
        episode_rows.append(row)
    means = np.nanmean(primary, axis=0)
    secondary_names = (
        "full_horizon_elite_mean_mse",
        "one_shot_top6_fp_score_regret",
        "score_nmse",
    )
    secondary_means = {
        name: {
            arm: float(np.mean([row["arms"][arm][name] for row in episode_rows]))
            for arm in ARMS
        }
        for name in secondary_names
    }
    full_mean = float(means[ARMS.index("FullInput-A8")])
    perm_mean = float(means[ARMS.index("Permuted-A8")])
    semantic_mean = float(means[ARMS.index("Semantic-A8")])
    full_better = (
        primary[:, ARMS.index("Semantic-A8")]
        < primary[:, ARMS.index("FullInput-A8")] - PRIMARY_STRICT_TOLERANCE
    ).tolist()
    perm_better = (
        primary[:, ARMS.index("Semantic-A8")]
        < primary[:, ARMS.index("Permuted-A8")] - PRIMARY_STRICT_TOLERANCE
    ).tolist()
    improvements = {
        "vs_full_input": (full_mean - semantic_mean) / max(abs(full_mean), NMSE_EPS),
        "vs_permuted": (perm_mean - semantic_mean) / max(abs(perm_mean), NMSE_EPS),
    }
    near_zero = full_mean <= NMSE_EPS or perm_mean <= NMSE_EPS
    passed = (
        not near_zero
        and improvements["vs_full_input"] >= GATE_IMPROVEMENT
        and improvements["vs_permuted"] >= GATE_IMPROVEMENT
        and sum(full_better) >= 4
        and sum(perm_better) >= 4
    )
    return {
        "episode_rows": episode_rows,
        "primary_name": "first_action_fidelity_proxy_mse",
        "one_shot_regret_formula": "mean_FP64(score_FP32[selected_top6]) - mean_FP64(score_FP32[FP_top6])",
        "score_nmse_name": "variance_normalized_score_mse",
        "score_nmse_formula": "mean((score_arm-score_FP32)^2)/max(var(score_FP32),1e-12)",
        "primary_mean_by_arm": {arm: float(means[i]) for i, arm in enumerate(ARMS)},
        "secondary_equal_episode_means": secondary_means,
        "primary_improvement": improvements,
        "semantic_better_episode_count": {
            "vs_full_input": int(sum(full_better)),
            "vs_permuted": int(sum(perm_better)),
        },
        "semantic_better_by_episode": {
            "vs_full_input": full_better,
            "vs_permuted": perm_better,
        },
        "baseline_near_zero": near_zero,
        "primary_does_not_claim_sequence_choice": True,
        "decision_gate": "no_binding_locus" if near_zero else ("preliminary_go" if passed else "mechanism_no_go"),
    }


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "allocation": dict(allocation),
        "allocation_evidence": {
            "guard": "ideaspark_run/v100-new-angles/allocation_guard.py",
            "owner_partition_node_verified": True,
            "hostname_normalization": "casefolded DNS short-name comparison",
        },
        "parameters": {
            "arms": list(ARMS),
            "horizon": HORIZON,
            "action_dim": ACTION_DIM,
            "input_dim": INPUT_DIM,
            "topk": TOPK,
            "cal_dataset_indices": list(CAL_DATASET_INDICES),
            "dev_dataset_indices": list(DEV_DATASET_INDICES),
            "cal_env_namespace": CAL_ENV_NAMESPACE,
            "dev_env_namespace": DEV_ENV_NAMESPACE,
            "cal_candidate_namespace": CAL_CANDIDATE_NAMESPACE,
            "dev_candidate_namespace": DEV_CANDIDATE_NAMESPACE,
            "max_seconds": args.max_seconds,
            "weight_scope": "predictor 24 Linear modules only; encoder and other weights FP32",
            "execution": "fake_quantization_emulation_only",
        },
    }
    try:
        smoke, screen = _load_helpers()
        summary["helper_identity"] = _assert_helper_identity(smoke, screen)
        _check_deadline(started, args.max_seconds, "helper identity")
        runtime = smoke._runtime(args.root.resolve(), device="cuda:0")
        structure = _model_structure(runtime)
        _assert_model_structure(structure)
        summary["model_structure"] = structure
        summary["runtime_identity"] = _runtime_identity(runtime, smoke)
        summary["source_identity"] = {
            "source_commit": summary["runtime_identity"].get("source_commit"),
            "dinov2_source_commit": summary["runtime_identity"].get("dinov2_source_commit"),
            "checkpoint_sha256": summary["runtime_identity"].get("checkpoint_sha256"),
            "visual_world_model_source": summary["runtime_identity"].get("visual_world_model_source"),
        }
        summary["gpu"] = _gpu_evidence(runtime["torch"])
        if len(runtime["dset"]) <= max(DEV_DATASET_INDICES):
            raise RuntimeError("validation dataset does not contain the frozen CAL/DEV positions")
        summary["dataset_mapping"] = _dataset_mapping_audit(runtime["dset"])
        _check_deadline(started, args.max_seconds, "runtime identity")

        targets_dir = (args.targets_dir or (output / "targets")).resolve()
        cal_manifest = _prepare_targets(
            runtime, targets_dir, screen, summary["runtime_identity"], structure,
            "cal950000", CAL_DATASET_INDICES, CAL_ENV_NAMESPACE, CAL_CANDIDATE_NAMESPACE,
        )
        _normalization_matches(runtime, screen, cal_manifest)
        dev_manifest = _prepare_targets(
            runtime, targets_dir, screen, summary["runtime_identity"], structure,
            "dev960000", DEV_DATASET_INDICES, DEV_ENV_NAMESPACE, DEV_CANDIDATE_NAMESPACE,
        )
        _normalization_matches(runtime, screen, dev_manifest)
        summary["target_manifests"] = {
            "calibration": str(_target_manifest_path(targets_dir, "cal950000")),
            "development": str(_target_manifest_path(targets_dir, "dev960000")),
            "calibration_fingerprints": [entry["target_fingerprint"] for entry in cal_manifest["episodes"]],
            "development_fingerprints": [entry["target_fingerprint"] for entry in dev_manifest["episodes"]],
        }
        _check_deadline(started, args.max_seconds, "target preparation")

        model = runtime["model"]
        groups = smoke._linear_groups(model)
        predictor_groups = [group for group in groups if group["family"] == "predictor"]
        predictor_linear_count = sum(int(group["linear_count"]) for group in predictor_groups)
        if predictor_linear_count != 24:
            raise RuntimeError(f"expected exactly 24 predictor Linear modules, got {predictor_linear_count}")
        snapshot = smoke._snapshot_weights(model, predictor_groups)
        preprocessor = smoke._preprocessor(runtime)
        objective = smoke._objective()

        dev_targets = [entry["target"] for entry in dev_manifest["episodes"]]
        cal_targets = [entry["target"] for entry in cal_manifest["episodes"]]
        dev_actions = np.stack([
            _candidate_pool(DEV_CANDIDATE_NAMESPACE + index, DEV_CANDIDATES)
            for index in range(len(DEV_DATASET_INDICES))
        ], axis=0)
        cal_actions = [
            _candidate_pool(CAL_CANDIDATE_NAMESPACE + index, CAL_CANDIDATES)
            for index in range(len(CAL_DATASET_INDICES))
        ]
        scores = np.full((len(DEV_DATASET_INDICES), len(ARMS), DEV_CANDIDATES), np.nan, dtype=np.float64)
        completed = np.zeros(len(DEV_DATASET_INDICES), dtype=np.bool_)

        # Protocol gate: on the first CAL pool, FP32 and an identity input hook
        # must have exactly the same score shape, values, stable top6, and
        # action shape.  The hook is then removed and its registration count is
        # checked before calibration begins.
        smoke._restore_weights(model, snapshot)
        hook_count_before = len(getattr(model.predictor, "_forward_pre_hooks", {}))
        fp_preflight = _score_pool(smoke, runtime, preprocessor, objective, cal_targets[0], cal_actions[0])
        noop = _InputHook(runtime["torch"], "NOOP", None)
        noop.attach(model.predictor)
        try:
            noop_preflight = _score_pool(smoke, runtime, preprocessor, objective, cal_targets[0], cal_actions[0])
        finally:
            noop.close()
        noop_diff = float(np.max(np.abs(fp_preflight - noop_preflight)))
        same_top6 = np.array_equal(_stable_top(fp_preflight), _stable_top(noop_preflight))
        same_score_shape = fp_preflight.shape == noop_preflight.shape == (CAL_CANDIDATES,)
        same_action_shape = cal_actions[0].shape == (CAL_CANDIDATES, HORIZON, ACTION_DIM)
        hook_count_after = len(getattr(model.predictor, "_forward_pre_hooks", {}))
        if (
            noop_diff > NOOP_TOLERANCE
            or not np.array_equal(fp_preflight, noop_preflight)
            or not same_top6
            or not same_score_shape
            or not same_action_shape
            or hook_count_after != hook_count_before
        ):
            raise RuntimeError(f"FP32 no-op hook equivalence failed: max_abs={noop_diff:g}")
        if noop.calls <= 0 or any(shape[-1] != INPUT_DIM for shape in noop.shapes):
            raise RuntimeError("predictor input hook did not observe the required live 404-channel tensor")
        summary["no_op"] = {
            "max_abs_score_diff": noop_diff,
            "exact_array_equal": True,
            "score_shape_equal": same_score_shape,
            "action_shape_equal": same_action_shape,
            "stable_top6_equal": same_top6,
            "fp_top6": _stable_top(fp_preflight).tolist(),
            "hook_count_before": hook_count_before,
            "hook_count_after_remove": hook_count_after,
            "removed_without_residue": hook_count_after == hook_count_before,
            "observed_shapes": noop.shapes,
            "calls": noop.calls,
            "required_shape": "[B,P,404]",
        }
        _check_deadline(started, args.max_seconds, "no-op gate")

        # CAL is collected only while predictor weights are W4 and the input is
        # untouched.  The resulting channel ranges are then frozen for every
        # DEV arm; no DEV update is possible in this runner.
        smoke._restore_weights(model, snapshot)
        w4_records = _quantize_predictor_weights(smoke, model, predictor_groups)
        cal_absmax = np.zeros(INPUT_DIM, dtype=np.float64)
        cal_hook_evidence: List[Dict[str, Any]] = []
        for index, target in enumerate(cal_targets):
            collector = _InputHook(runtime["torch"], "W4A32-CAL", None)
            collector.attach(model.predictor)
            try:
                _score_pool(smoke, runtime, preprocessor, objective, target, cal_actions[index])
            finally:
                collector.close()
            if collector.raw_absmax is None or collector.raw_absmax.shape != (INPUT_DIM,):
                raise RuntimeError("CAL hook did not observe a 404-channel range")
            cal_absmax = np.maximum(cal_absmax, collector.raw_absmax)
            cal_hook_evidence.append(collector.evidence())
            _check_deadline(started, args.max_seconds, f"CAL episode {index}")
        scales, scale_definitions = _scale_vectors(cal_absmax)
        scale_payload = {
            "schema": "semantic-input-scales-scale-definitions-v1",
            "calibration_dataset_indices": list(CAL_DATASET_INDICES),
            "calibration_candidate_shape": [CAL_CANDIDATES, HORIZON, ACTION_DIM],
            "shared_cal_absmax_by_channel": cal_absmax.tolist(),
            "shared_cal_absmax_shape": [INPUT_DIM],
            "shared_cal_absmax_digest": _digest_arrays(cal_absmax),
            "calibration_hook_evidence": cal_hook_evidence,
            "definitions": scale_definitions,
        }
        scale_path = output / "scale_definitions.json"
        _atomic_json(scale_path, scale_payload)
        scale_identity = _file_identity(scale_path)
        summary["calibration"] = {
            "source": "W4A32 predictor-input hook",
            "dataset_indices": list(CAL_DATASET_INDICES),
            "candidate_shape": [CAL_CANDIDATES, HORIZON, ACTION_DIM],
            "hook_calls_by_episode": [int(item["calls"]) for item in cal_hook_evidence],
            "total_hook_calls": int(sum(item["calls"] for item in cal_hook_evidence)),
            "shared_absmax_shape": [INPUT_DIM],
            "absmax_digest": _digest_arrays(cal_absmax),
            "scale_definitions_path": str(scale_path.resolve()),
            "scale_definitions_sha256": scale_identity["sha256"],
            "scale_definitions_size_bytes": scale_identity["size_bytes"],
            "scale_modes": ["FullInput-A8", "Semantic-A8", "Permuted-A8", "PerChannel-A8"],
            "dev_reestimation": False,
        }

        # Every arm uses the same candidate arrays and target objects.  Weight
        # restoration from the FP snapshot prevents accidental cumulative W4.
        arm_hook_evidence: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARMS}
        for episode, target in enumerate(dev_targets):
            for arm_index, arm in enumerate(ARMS):
                smoke._restore_weights(model, snapshot)
                hook: _InputHook | None = None
                if arm != "FP32":
                    _quantize_predictor_weights(smoke, model, predictor_groups)
                if arm in scales:
                    hook = _InputHook(runtime["torch"], arm, scales[arm])
                    hook.attach(model.predictor)
                try:
                    scores[episode, arm_index] = _score_pool(
                        smoke, runtime, preprocessor, objective, target, dev_actions[episode]
                    )
                finally:
                    if hook is not None:
                        hook.close()
                        arm_hook_evidence[arm].append(hook.evidence())
                _check_deadline(started, args.max_seconds, f"DEV episode {episode} arm {arm}")
            completed[episode] = True
            _write_raw(output, dev_actions, scores, completed)

        if not bool(completed.all()):
            raise RuntimeError("DEV screen did not complete all six episodes")
        if not np.isfinite(scores).all():
            raise FloatingPointError("raw DEV scores contain non-finite values")
        metric_payload = _metrics(dev_actions, scores)
        summary["gates"] = {
            "engineering_shape_identity": True,
            "noop_exact_scores_and_top6": bool(summary["no_op"]["exact_array_equal"] and summary["no_op"]["stable_top6_equal"]),
            "noop_removed_without_residue": bool(summary["no_op"]["removed_without_residue"]),
            "six_dev_episodes_complete": bool(completed.all()),
            "primary_mean_improvement_vs_full_ge_5pct": bool(metric_payload["primary_improvement"]["vs_full_input"] >= GATE_IMPROVEMENT),
            "primary_mean_improvement_vs_permuted_ge_5pct": bool(metric_payload["primary_improvement"]["vs_permuted"] >= GATE_IMPROVEMENT),
            "primary_strict_episode_count_vs_full_ge_4": bool(metric_payload["semantic_better_episode_count"]["vs_full_input"] >= 4),
            "primary_strict_episode_count_vs_permuted_ge_4": bool(metric_payload["semantic_better_episode_count"]["vs_permuted"] >= 4),
            "baseline_near_zero_any_control": bool(metric_payload["baseline_near_zero"]),
            "regret_is_secondary_only": True,
        }
        summary.update({
            "status": "complete",
            "completed_episodes": int(completed.sum()),
            "quantizer": {
                "weight_bits": BITS,
                "weight_qmax": WEIGHT_QMAX,
                "weight_records_W4_predictor": w4_records,
                "scope": "predictor 24 Linear modules only",
                "input_arms": list(ARMS[2:]),
            },
            "input_quantization_evidence": arm_hook_evidence,
            "metrics": metric_payload,
            "raw_npz": str((output / "raw_scores_actions.npz").resolve()),
            "target_fingerprints": summary["target_manifests"],
            "elapsed_seconds": time.monotonic() - started,
        })
        _write_small_summary(output / "summary.json", summary)
        print(json.dumps({"status": "complete", "output": str(output), "decision": metric_payload["decision_gate"]}, indent=2), flush=True)
    except Exception as exc:
        summary.update({
            "status": "implementation_failure" if isinstance(exc, (RuntimeError, FileNotFoundError, ImportError)) else "inconclusive",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        })
        try:
            _write_small_summary(output / "summary.json", summary)
        except RuntimeError:
            # Preserve a bounded failure record if an unexpected error added
            # too much diagnostic content; detailed raw data remains remote.
            compact = {
                "schema": SCHEMA,
                "status": summary.get("status", "inconclusive"),
                "allocation": summary.get("allocation"),
                "error": summary.get("error"),
                "raw_npz": str((output / "raw_scores_actions.npz").resolve()),
                "scale_definitions": str((output / "scale_definitions.json").resolve()),
            }
            _write_small_summary(output / "summary.json", compact)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM Wall modelroot")
    parser.add_argument("--output", type=Path, required=True, help="allocation-local artifact directory")
    parser.add_argument("--targets-dir", type=Path, default=None, help="optional frozen target directory")
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_WORKLOAD_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_WORKLOAD_SECONDS:g}]")
    return args


def main() -> None:
    args = _parse_args()
    # Keep this call first among workload actions.  It checks actual hostname,
    # owner, running job, partition and nodelist; no environment spoofing is
    # accepted by the campaign guard.
    require_allocation = _load_allocation_guard()
    allocation = require_allocation()
    _run(args, allocation)


if __name__ == "__main__":
    main()
