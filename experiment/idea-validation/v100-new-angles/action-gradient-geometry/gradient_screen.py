"""Minimal differentiable action-gradient geometry screen for DINO-WM Wall.

This runner is an allocation-only diagnostic.  It uses the reviewed FP32
predictor and a dequantized predictor-only W4 RTN copy, differentiates only a
normalized action leaf, and never performs an environment rollout.  The first
workload action in :func:`main` is the campaign allocation guard; importing or
parsing this file therefore does not load a model or touch experiment data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "action-gradient-geometry-screen-v1"
RAW_SCHEMA = "action-gradient-geometry-raw-v1"
HORIZON = 2
ACTION_DIM = 10
ACTION_FLAT_DIM = HORIZON * ACTION_DIM
NUM_CANDIDATES = 64
ANCHOR_INDICES = (0, 1, 2, 3)
EPISODE_COUNT = 6
DATASET_INDICES = tuple(range(118, 124))
EXCLUDED_DATASET_INDICES = tuple(range(118))
ENV_SEED_NAMESPACE = 970000
CANDIDATE_SEED_NAMESPACE = 980000
BITS = 4
QMAX = 7
STEP_LENGTH = 0.10
FD_EPSILON = 0.005
GRADIENT_NORM_TOLERANCE = 1e-8
STEP_IMPROVEMENT_TOLERANCE = 1e-7
FD_RELATIVE_TOLERANCE = 0.1
FD_ABSOLUTE_TOLERANCE = 1e-4
MAX_WORKLOAD_SECONDS = 1080.0
ARM_NAMES = ("FP32", "predictor_W4")

EXPECTED_CHECKPOINT_SHA256 = "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b"
EXPECTED_SOURCE_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
EXPECTED_DINOV2_SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
EXPECTED_SMOKE_RUNNER_SHA256 = "de5c5eb19f4b26614e71f9e2af36db21e9fd5bcc65188f3191772552bc750af9"
EXPECTED_SCREEN_RUNNER_SHA256 = "51c2463a92a3bf84eabae735eeaa9771a7202add0b2227fe06c7fe1f2bd69d19"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB ({len(encoded)} bytes)")
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


def _load_allocation_guard() -> Any:
    try:
        module = importlib.import_module("allocation_guard")
        return module.require_allocation
    except ModuleNotFoundError:
        for guard_path in (
            Path(__file__).with_name("allocation_guard.py"),
            Path(__file__).resolve().parents[1] / "allocation_guard.py",
        ):
            if not guard_path.is_file():
                continue
            spec = importlib.util.spec_from_file_location("gradient_allocation_guard", guard_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.require_allocation
    raise ImportError("campaign allocation_guard.py is unavailable")


def _load_helpers(helper_dir: Path | None = None) -> Tuple[Any, Any]:
    """Load the exact helper pair copied by the SLURM controller."""
    if helper_dir is not None:
        helper_dir = helper_dir.resolve()
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
    try:
        return importlib.import_module("smoke_runner"), importlib.import_module("screen_runner")
    except ModuleNotFoundError:
        candidates = []
        if helper_dir is not None:
            candidates.append(helper_dir)
        candidates.append(Path(__file__).resolve().parents[4] / "experiment" / "idea-validation" / "world-model-quantization" / "dino-wm-wall")
        for directory in candidates:
            smoke_path = directory / "smoke_runner.py"
            screen_path = directory / "screen_runner.py"
            if smoke_path.is_file() and screen_path.is_file():
                if str(directory) not in sys.path:
                    sys.path.insert(0, str(directory))
                return importlib.import_module("smoke_runner"), importlib.import_module("screen_runner")
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
            raise RuntimeError(f"{key} mismatch: got {actual.get(key)!r}, expected {frozen!r}")
    return {key: str(value) for key, value in actual.items()}


def _runtime_identity(runtime: Mapping[str, Any], smoke: Any) -> Dict[str, Any]:
    identity = dict(smoke._checkpoint_identity(runtime))
    checkpoint = Path(runtime["checkpoint"]).resolve()
    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: got {checkpoint_sha256!r}, expected {EXPECTED_CHECKPOINT_SHA256!r}"
        )
    for key, expected in (
        ("source_commit", EXPECTED_SOURCE_COMMIT),
        ("dinov2_source_commit", EXPECTED_DINOV2_SOURCE_COMMIT),
    ):
        if identity.get(key) != expected:
            raise RuntimeError(f"{key} mismatch: got {identity.get(key)!r}, expected {expected!r}")
    identity["checkpoint_sha256"] = checkpoint_sha256
    root = Path(runtime["root"]).resolve()
    identity["source_files"] = {
        "visual_world_model.py": _file_identity(root / "source" / "models" / "visual_world_model.py"),
        "objectives.py": _file_identity(root / "source" / "planning" / "objectives.py"),
        "wall_dset.py": _file_identity(root / "source" / "datasets" / "wall_dset.py"),
        "traj_dset.py": _file_identity(root / "source" / "datasets" / "traj_dset.py"),
    }
    missing = [name for name, record in identity["source_files"].items() if record.get("sha256") is None]
    if missing:
        raise FileNotFoundError(f"runtime source identity is incomplete: {missing}")
    identity["action_graph"] = "model.rollout(obs_0=transformed_obs_0, act=action_leaf); objective is smoke._objective()"
    return identity


def _gpu_evidence(torch: Any) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; refusing accidental CPU model execution")
    index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(index)
    name = str(props.name)
    if "v100" not in name.casefold():
        raise RuntimeError(f"this screen requires a V100 allocation, got {name!r}")
    return {
        "device_index": index,
        "name": name,
        "total_memory_bytes": int(props.total_memory),
        "torch": str(torch.__version__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _model_structure(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    model = runtime["model"]
    cfg = runtime.get("model_cfg")
    dset = runtime["dset"]

    def read(*objects: Any, name: str) -> Any:
        for obj in objects:
            if obj is None:
                continue
            value = getattr(obj, name, None)
            if value is not None and not callable(value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return str(value)
        return None

    encoder = getattr(model, "encoder", None)
    result = {
        "num_hist": read(model, cfg, name="num_hist"),
        "num_pred": read(model, cfg, name="num_pred"),
        "frameskip": read(model, cfg, name="frameskip"),
        "concat_dim": read(model, encoder, cfg, name="concat_dim"),
        "proprio_dim": read(model, cfg, name="proprio_dim"),
        "model_action_dim": read(model, name="action_dim"),
        "dataset_action_dim": read(dset, name="action_dim"),
        "encoder_emb_dim": read(encoder, name="emb_dim"),
        "model_emb_dim": read(model, name="emb_dim"),
        "predictor_type": type(getattr(model, "predictor", None)).__name__,
    }
    raw_action_dim = result["dataset_action_dim"]
    frameskip = result["frameskip"]
    if raw_action_dim is None or frameskip is None:
        raise RuntimeError("runtime contract missing dataset action_dim or config frameskip")
    result["effective_action_dim"] = int(raw_action_dim * frameskip)
    result["expected_action_graph_dim"] = ACTION_DIM
    result["concat_input_identity"] = "encoder visual 384 + proprio 10 + action 10 = 404 when concat_dim=1"
    required = {
        "num_hist": 1,
        "concat_dim": 1,
        "frameskip": 5,
        "effective_action_dim": ACTION_DIM,
        "proprio_dim": 10,
        "encoder_emb_dim": 384,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(f"structural no-go: live {key}={result.get(key)!r}, expected {expected!r}")
    if result.get("model_emb_dim") not in (None, 404):
        raise RuntimeError(f"structural no-go: model_emb_dim={result.get('model_emb_dim')!r}, expected 404")
    return result


def _dataset_mapping_audit(dset: Any) -> Dict[str, Any]:
    """Resolve subset indices using metadata only; no sample is indexed."""
    positions = list(range(max(DATASET_INDICES) + 1))
    current = dset
    layers: List[Dict[str, Any]] = []
    while hasattr(current, "indices") and hasattr(current, "dataset"):
        raw_indices = list(getattr(current, "indices"))
        if any(position >= len(raw_indices) for position in positions):
            raise RuntimeError("dataset subset metadata cannot map positions through this layer")
        positions = [int(value.item() if hasattr(value, "item") else value) for value in (raw_indices[p] for p in positions)]
        next_dataset = getattr(current, "dataset")
        layers.append({
            "type": type(current).__name__,
            "dataset_type": type(next_dataset).__name__,
            "index_count": len(raw_indices),
            "metadata_only": True,
        })
        current = next_dataset
    fresh_ids = positions[DATASET_INDICES[0] : DATASET_INDICES[-1] + 1]
    excluded_ids = positions[: len(EXCLUDED_DATASET_INDICES)]
    if len(set(fresh_ids)) != EPISODE_COUNT:
        raise RuntimeError("fresh dataset positions map to duplicate underlying episode ids")
    overlap = sorted(set(fresh_ids).intersection(excluded_ids))
    if overlap:
        raise RuntimeError(f"fresh 118..123 mapping overlaps excluded 0..117: {overlap}")
    source_ids = [f"{type(current).__name__}:{int(value)}" for value in fresh_ids]
    return {
        "dataset_type_after_subset_resolution": type(current).__name__,
        "subset_layers": layers,
        "fresh_positions": list(DATASET_INDICES),
        "fresh_underlying_ids": [int(value) for value in fresh_ids],
        "source_episode_ids": source_ids,
        "excluded_positions": list(EXCLUDED_DATASET_INDICES),
        "excluded_underlying_id_count": len(excluded_ids),
        "fresh_distinct": True,
        "fresh_disjoint_from_excluded": True,
        "protected_state_reads": False,
    }


def _model_snapshot_equal(model: Any, snapshot: Mapping[str, Any], smoke: Any, torch: Any) -> None:
    for key, expected in snapshot.items():
        family, index, relative = key.split(".", 2)
        actual = smoke._module_for_path(model, family, int(index), relative).weight.detach()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"weight snapshot restore mismatch at {key}")


def _assert_encoder_untouched(model: Any, snapshot: Mapping[str, Any], smoke: Any, torch: Any) -> None:
    for key, expected in snapshot.items():
        if not key.startswith("encoder."):
            continue
        family, index, relative = key.split(".", 2)
        actual = smoke._module_for_path(model, family, int(index), relative).weight.detach()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"predictor quantizer changed encoder weight at {key}")


class _WeightTransaction:
    """Restore a complete model snapshot on every arm exit, including errors."""

    def __init__(self, model: Any, snapshot: Mapping[str, Any], smoke: Any, torch: Any) -> None:
        self.model = model
        self.snapshot = snapshot
        self.smoke = smoke
        self.torch = torch

    def __enter__(self) -> "_WeightTransaction":
        self.smoke._restore_weights(self.model, self.snapshot)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.smoke._restore_weights(self.model, self.snapshot)
        _model_snapshot_equal(self.model, self.snapshot, self.smoke, self.torch)
        return False


def _target_tensors(model: Any, preprocessor: Any, target: Mapping[str, Any], count: int, torch: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    device = next(model.parameters()).device
    trans_0 = {key: value.to(device) for key, value in preprocessor.transform_obs(dict(target["obs_0"])).items()}
    trans_g = {key: value.to(device) for key, value in preprocessor.transform_obs(dict(target["obs_g"])).items()}
    trans_0 = {key: value[0:1].repeat((count,) + (1,) * (value.ndim - 1)) for key, value in trans_0.items()}
    with torch.no_grad():
        z_goal_single = model.encode_obs(trans_g)
    z_goal = {key: value[0:1].repeat((count,) + (1,) * (value.ndim - 1)) for key, value in z_goal_single.items()}
    return trans_0, z_goal


def _score_values(model: Any, preprocessor: Any, objective_fn: Any, target: Mapping[str, Any], actions: np.ndarray, torch: Any) -> np.ndarray:
    if actions.ndim != 3 or actions.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"expected action shape (N,{HORIZON},{ACTION_DIM}), got {actions.shape}")
    with torch.no_grad():
        trans_0, z_goal = _target_tensors(model, preprocessor, target, int(actions.shape[0]), torch)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=next(model.parameters()).device)
        z_obses, _ = model.rollout(obs_0=trans_0, act=action_tensor)
        scores = objective_fn(z_obses, z_goal).reshape(-1)
    if scores.numel() != actions.shape[0] or not bool(torch.isfinite(scores).all().item()):
        raise FloatingPointError("non-finite or mis-shaped objective scores")
    return scores.detach().cpu().numpy().astype(np.float32, copy=True)


def _scores_and_gradients(model: Any, preprocessor: Any, objective_fn: Any, target: Mapping[str, Any], actions: np.ndarray, torch: Any) -> Tuple[np.ndarray, np.ndarray]:
    if actions.ndim != 3 or actions.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"expected action shape (N,{HORIZON},{ACTION_DIM}), got {actions.shape}")
    with torch.enable_grad():
        action_leaf = torch.as_tensor(actions, dtype=torch.float32, device=next(model.parameters()).device).detach().clone()
        action_leaf.requires_grad_(True)
        trans_0, z_goal = _target_tensors(model, preprocessor, target, int(actions.shape[0]), torch)
        z_obses, _ = model.rollout(obs_0=trans_0, act=action_leaf)
        scores = objective_fn(z_obses, z_goal).reshape(-1)
        if scores.numel() != actions.shape[0] or not bool(torch.isfinite(scores).all().item()):
            raise FloatingPointError("non-finite or mis-shaped differentiable objective scores")
        gradients = torch.autograd.grad(scores.sum(), action_leaf, retain_graph=False, create_graph=False, allow_unused=False)[0]
        if not bool(torch.isfinite(gradients).all().item()):
            raise FloatingPointError("non-finite action gradients")
    return scores.detach().cpu().numpy().astype(np.float32, copy=True), gradients.detach().cpu().numpy().astype(np.float32, copy=True)


def _candidate_pool(seed: int, np_random: Any) -> np.ndarray:
    candidates = np_random.default_rng(int(seed)).standard_normal((NUM_CANDIDATES, HORIZON, ACTION_DIM)).astype(np.float32)
    if not np.isfinite(candidates).all() or np.all(candidates[0] == 0.0):
        raise RuntimeError("candidate pool is non-finite or candidate 0 is zero")
    return candidates


def _episode_target(runtime: Mapping[str, Any], screen: Any, dataset_index: int, env_seed: int) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    target, layout = screen._new_target(runtime, int(dataset_index), int(env_seed))
    target["goal_H"] = HORIZON
    fingerprint = screen._target_fingerprint(target, layout)
    target["target_fingerprint"] = fingerprint
    return target, layout, fingerprint


def _write_raw(output: Path, arrays: Mapping[str, Any]) -> None:
    _atomic_npz(output / "raw_gradient.npz", **arrays)


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    smoke, screen = _load_helpers(args.helper_dir)
    helper_identity = _assert_helper_identity(smoke, screen)
    runtime = smoke._runtime(args.root.resolve())
    torch = runtime["torch"]
    gpu = _gpu_evidence(torch)
    runtime_identity = _runtime_identity(runtime, smoke)
    structure = _model_structure(runtime)
    mapping = _dataset_mapping_audit(runtime["dset"])
    if len(runtime["dset"]) <= max(DATASET_INDICES):
        raise RuntimeError("validation dataset is too short for fresh positions 118..123")
    model = runtime["model"]
    model.eval()
    groups = smoke._linear_groups(model)
    predictor_groups = [group for group in groups if group["family"] == "predictor"]
    if len(predictor_groups) != 6:
        raise RuntimeError(f"expected six predictor groups, got {len(predictor_groups)}")
    snapshot = smoke._snapshot_weights(model, groups)
    preprocessor = smoke._preprocessor(runtime)
    objective_fn = smoke._objective()

    parameter_flags = {id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)

    arrays: Dict[str, Any] = {
        "schema": np.asarray(RAW_SCHEMA),
        "arm_names": np.asarray(ARM_NAMES),
        "actions": np.full((EPISODE_COUNT, NUM_CANDIDATES, HORIZON, ACTION_DIM), np.nan, dtype=np.float32),
        "scores": np.full((EPISODE_COUNT, 2, NUM_CANDIDATES), np.nan, dtype=np.float32),
        "gradients": np.full((EPISODE_COUNT, 2, len(ANCHOR_INDICES), HORIZON, ACTION_DIM), np.nan, dtype=np.float32),
        "fp_step_scores": np.full((EPISODE_COUNT, 2, len(ANCHOR_INDICES)), np.nan, dtype=np.float32),
        "base_scores": np.full((EPISODE_COUNT, len(ANCHOR_INDICES)), np.nan, dtype=np.float32),
        "fd_values": np.full((EPISODE_COUNT, 2, 2), np.nan, dtype=np.float32),
        "completed": np.zeros((EPISODE_COUNT,), dtype=np.bool_),
        "dataset_indices": np.asarray(DATASET_INDICES, dtype=np.int64),
        "source_episode_ids": np.asarray(mapping["source_episode_ids"]),
    }
    # The action axis is intentionally (candidate=64, H=2, action_dim=10); spell out the
    # asserted raw shape to prevent a future constant edit from changing it.
    if arrays["actions"].shape != (EPISODE_COUNT, NUM_CANDIDATES, HORIZON, ACTION_DIM):
        raise AssertionError("raw action shape construction drifted")
    _write_raw(output, arrays)

    engineering: Dict[str, Any] = {
        "schema": "action-gradient-geometry-engineering-v1",
        "status": "started",
        "allocation": dict(allocation),
        "helper_identity": helper_identity,
        "runtime_identity": runtime_identity,
        "gpu": gpu,
        "model_structure": structure,
        "dataset_mapping": mapping,
        "targets": [],
        "completed_episodes": 0,
        "raw_npz": str((output / "raw_gradient.npz").resolve()),
        "metrics_deferred_to_root_cpu_verifier": True,
    }
    _atomic_json(output / "engineering.json", engineering)

    target_records: List[Dict[str, Any]] = []
    episode_records: List[Dict[str, Any]] = []
    quantizer_records: List[Dict[str, Any]] = []
    for local_index, dataset_index in enumerate(DATASET_INDICES):
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError(f"max-seconds reached before episode {local_index}")
        env_seed = ENV_SEED_NAMESPACE + local_index
        candidate_seed = CANDIDATE_SEED_NAMESPACE + local_index
        target, layout, fingerprint = _episode_target(runtime, screen, dataset_index, env_seed)
        candidates = _candidate_pool(candidate_seed, np.random)
        arrays["actions"][local_index, :, :, :] = candidates
        target_records.append({
            "local_index": local_index,
            "dataset_index": int(dataset_index),
            "env_seed": int(env_seed),
            "candidate_seed": int(candidate_seed),
            "source_episode_id": mapping["source_episode_ids"][local_index],
            "layout": dict(layout),
            "target_fingerprint": fingerprint,
            "goal_H": HORIZON,
            "construction": "screen._new_target initialization/prepare only; no smoke explicit targets or environment rollout",
        })
        engineering["targets"] = target_records
        engineering["current_episode"] = local_index
        _atomic_json(output / "engineering.json", engineering)

        with _WeightTransaction(model, snapshot, smoke, torch):
            fp_scores = _score_values(model, preprocessor, objective_fn, target, candidates, torch)
            fp_anchor_scores, fp_gradients = _scores_and_gradients(
                model, preprocessor, objective_fn, target, candidates[list(ANCHOR_INDICES)], torch
            )
        arrays["scores"][local_index, 0] = fp_scores
        arrays["gradients"][local_index, 0] = fp_gradients
        with _WeightTransaction(model, snapshot, smoke, torch):
            q_records = [smoke._quantize_group(model, group, BITS) for group in predictor_groups]
            _assert_encoder_untouched(model, snapshot, smoke, torch)
            q_scores = _score_values(model, preprocessor, objective_fn, target, candidates, torch)
            _q_anchor_scores, q_gradients = _scores_and_gradients(
                model, preprocessor, objective_fn, target, candidates[list(ANCHOR_INDICES)], torch
            )
        arrays["scores"][local_index, 1] = q_scores
        arrays["gradients"][local_index, 1] = q_gradients
        quantizer_records.append({
            "episode_index": local_index,
            "arm": "predictor_W4",
            "group_count": len(q_records),
            "bits": BITS,
            "qmax": QMAX,
            "scheme": "smoke._quantize_group symmetric per-output-channel RTN dequantized FP32",
            "groups": q_records,
        })
        engineering["quantizer_episodes"] = [
            {"episode_index": int(record["episode_index"]), "group_count": int(record["group_count"]), "bits": int(record["bits"])}
            for record in quantizer_records
        ]
        _atomic_json(output / "engineering.json", engineering)

        fp_anchor_actions = candidates[list(ANCHOR_INDICES)]
        fp_anchor_grad = arrays["gradients"][local_index, 0].astype(np.float64)
        q_anchor_grad = arrays["gradients"][local_index, 1].astype(np.float64)
        fp_norms = np.linalg.norm(fp_anchor_grad.reshape(len(ANCHOR_INDICES), -1), axis=1)
        q_norms = np.linalg.norm(q_anchor_grad.reshape(len(ANCHOR_INDICES), -1), axis=1)
        if not np.isfinite(fp_norms).all() or not np.isfinite(q_norms).all():
            raise FloatingPointError(f"non-finite gradient norm at episode {local_index}")
        fp_unit = np.divide(
            fp_anchor_grad,
            fp_norms[:, None, None],
            out=np.zeros_like(fp_anchor_grad),
            where=fp_norms[:, None, None] > 0.0,
        )
        q_unit = np.divide(
            q_anchor_grad,
            q_norms[:, None, None],
            out=np.zeros_like(q_anchor_grad),
            where=q_norms[:, None, None] > 0.0,
        )
        if np.any(fp_norms <= GRADIENT_NORM_TOLERANCE) or np.any(q_norms <= GRADIENT_NORM_TOLERANCE):
            local_norm_gate = False
        else:
            local_norm_gate = True
        arrays["base_scores"][local_index] = fp_anchor_scores
        fp_step_actions = fp_anchor_actions - STEP_LENGTH * fp_unit.astype(np.float32)
        q_step_actions = fp_anchor_actions - STEP_LENGTH * q_unit.astype(np.float32)
        with _WeightTransaction(model, snapshot, smoke, torch):
            fp_step = _score_values(model, preprocessor, objective_fn, target, fp_step_actions, torch)
            q_step = _score_values(model, preprocessor, objective_fn, target, q_step_actions, torch)
        arrays["fp_step_scores"][local_index, 0] = fp_step
        arrays["fp_step_scores"][local_index, 1] = q_step
        fp_improvements = arrays["base_scores"][local_index].astype(np.float64) - fp_step.astype(np.float64)
        q_improvements = arrays["base_scores"][local_index].astype(np.float64) - q_step.astype(np.float64)
        local_step_gate = bool(np.all(fp_improvements > STEP_IMPROVEMENT_TOLERANCE))
        cosines = np.sum(fp_unit * q_unit, axis=(1, 2))

        fd_records = []
        for arm_index, (unit, grad) in enumerate(((fp_unit, fp_anchor_grad), (q_unit, q_anchor_grad))):
            fd_actions = np.stack((fp_anchor_actions[0] + FD_EPSILON * unit[0], fp_anchor_actions[0] - FD_EPSILON * unit[0]))
            with _WeightTransaction(model, snapshot, smoke, torch):
                if arm_index == 1:
                    [smoke._quantize_group(model, group, BITS) for group in predictor_groups]
                fd_scores = _score_values(model, preprocessor, objective_fn, target, fd_actions, torch)
            arrays["fd_values"][local_index, arm_index] = fd_scores
            derivative = float((float(fd_scores[0]) - float(fd_scores[1])) / (2.0 * FD_EPSILON))
            dot = float(np.dot(grad[0].reshape(-1), unit[0].reshape(-1)))
            tolerance = FD_RELATIVE_TOLERANCE * abs(dot) + FD_ABSOLUTE_TOLERANCE
            fd_records.append({
                "arm": ARM_NAMES[arm_index],
                "plus_score": float(fd_scores[0]),
                "minus_score": float(fd_scores[1]),
                "finite_difference": derivative,
                "gradient_dot_unit": dot,
                "absolute_error": abs(derivative - dot),
                "tolerance": tolerance,
                "passed": bool(abs(derivative - dot) <= tolerance),
            })
        smoke._restore_weights(model, snapshot)
        _model_snapshot_equal(model, snapshot, smoke, torch)
        all_finite = bool(all(np.isfinite(np.asarray(arrays[key][local_index])).all() for key in ("actions", "scores", "gradients", "fp_step_scores", "base_scores", "fd_values")))
        if not all_finite:
            raise FloatingPointError(f"non-finite raw value at episode {local_index}")
        arrays["completed"][local_index] = True
        episode_records.append({
            "local_index": local_index,
            "dataset_index": int(dataset_index),
            "target_fingerprint": fingerprint,
            "candidate0_l2": float(np.linalg.norm(candidates[0].reshape(-1))),
            "gradient_norms": {ARM_NAMES[0]: fp_norms.tolist(), ARM_NAMES[1]: q_norms.tolist()},
            "cosines": cosines.tolist(),
            "fp_step_improvements": fp_improvements.tolist(),
            "q_step_improvements": q_improvements.tolist(),
            "engineering_local_checks": {
                "gradient_norms_above_1e-8": local_norm_gate,
                "fp_step_improvements_above_1e-7": local_step_gate,
                "fd": fd_records,
            },
        })
        engineering["episode_engineering"] = episode_records
        engineering["completed_episodes"] = int(arrays["completed"].sum())
        engineering["status"] = "partial"
        _atomic_json(output / "engineering.json", engineering)
        _write_raw(output, arrays)
        _write_summary(output / "summary.json", {
            "schema": SCHEMA,
            "status": "partial",
            "allocation": dict(allocation),
            "completed_episodes": int(arrays["completed"].sum()),
            "raw_npz": str((output / "raw_gradient.npz").resolve()),
            "engineering": str((output / "engineering.json").resolve()),
            "metrics_deferred_to_root_cpu_verifier": True,
            "max_seconds": args.max_seconds,
        })

    smoke._restore_weights(model, snapshot)
    _model_snapshot_equal(model, snapshot, smoke, torch)
    for parameter in model.parameters():
        if parameter.grad is not None:
            raise RuntimeError("parameter gradient was populated; action-only graph contract failed")
        parameter.requires_grad_(parameter_flags[id(parameter)])
    _write_raw(output, arrays)
    _atomic_json(output / "quantizer_audit.json", {
        "schema": "action-gradient-geometry-quantizer-audit-v1",
        "scope": "predictor transformer Linear groups only",
        "records": quantizer_records,
    })
    engineering.update({
        "status": "complete",
        "episode_engineering": episode_records,
        "completed_episodes": int(arrays["completed"].sum()),
        "quantizer_audit": str((output / "quantizer_audit.json").resolve()),
        "weights_restored_exactly": True,
        "predictor_only_encoder_untouched": True,
    })
    _atomic_json(output / "engineering.json", engineering)
    _write_summary(output / "summary.json", {
        "schema": SCHEMA,
        "status": "complete",
        "allocation": dict(allocation),
        "parameters": {
            "episode_count": EPISODE_COUNT,
            "dataset_indices": list(DATASET_INDICES),
            "excluded_dataset_indices": list(EXCLUDED_DATASET_INDICES),
            "env_seed_namespace": ENV_SEED_NAMESPACE,
            "candidate_seed_namespace": CANDIDATE_SEED_NAMESPACE,
            "candidate_shape": [NUM_CANDIDATES, HORIZON, ACTION_DIM],
            "candidate_rng": "numpy.default_rng(seed).standard_normal",
            "anchor_indices": list(ANCHOR_INDICES),
            "step_length": STEP_LENGTH,
            "fd_epsilon": FD_EPSILON,
            "bits": BITS,
            "qmax": QMAX,
            "arm_names": list(ARM_NAMES),
            "objective": "official create_objective_fn(alpha=1, base=2, mode=last); terminal visual + proprio",
            "gradient_contract": "torch.enable_grad; action leaf only; all model parameters requires_grad=False; no STE",
        },
        "helper_identity": helper_identity,
        "runtime_identity": runtime_identity,
        "gpu": gpu,
        "model_structure": structure,
        "dataset_mapping": mapping,
        "targets": target_records,
        "episode_engineering": episode_records,
        "quantizer_audit": str((output / "quantizer_audit.json").resolve()),
        "engineering": str((output / "engineering.json").resolve()),
        "raw_npz": str((output / "raw_gradient.npz").resolve()),
        "raw_schema": {
            "schema": RAW_SCHEMA,
            "actions": [EPISODE_COUNT, NUM_CANDIDATES, HORIZON, ACTION_DIM],
            "scores": [EPISODE_COUNT, 2, NUM_CANDIDATES],
            "gradients": [EPISODE_COUNT, 2, len(ANCHOR_INDICES), HORIZON, ACTION_DIM],
            "fp_step_scores": [EPISODE_COUNT, 2, len(ANCHOR_INDICES)],
            "base_scores": [EPISODE_COUNT, len(ANCHOR_INDICES)],
            "fd_values": [EPISODE_COUNT, 2, 2],
            "completed": [EPISODE_COUNT],
        },
        "metrics": {
            "deferred_to_root_cpu_verifier": True,
            "global_binding": "average-rank Spearman and centered score NRMSE per frozen protocol",
            "local_geometry": "FP/Q cosine and FP-evaluated negative-step improvements per frozen protocol",
            "no_scientific_verdict_in_gpu_runner": True,
        },
        "engineering_gates": {
            "completed_episodes": int(arrays["completed"].sum()),
            "all_raw_finite": bool(all(np.isfinite(np.asarray(arrays[key])).all() for key in ("actions", "scores", "gradients", "fp_step_scores", "base_scores", "fd_values"))),
            "weights_restored_exactly": True,
            "predictor_only_encoder_untouched": True,
            "fd_results_are_recorded_per_episode_and_arm": True,
        },
        "unresolved": [
            "fake quantization is emulated FP32 RTN; this is not a native W4 deployment or speed claim",
            "six episodes and one candidate pool are a bounded screen; novelty and generalization remain unverified",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "max_seconds": args.max_seconds,
    })
    print(json.dumps({"status": "complete", "episodes": EPISODE_COUNT, "raw": str(output / "raw_gradient.npz")}, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="remote DINO-WM modelroot")
    parser.add_argument("--output", type=Path, required=True, help="one-job artifact directory")
    parser.add_argument("--helper-dir", type=Path, default=None, help="directory containing reviewed helper pair")
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    return args


def main() -> None:
    # Deliberately first workload action: no model/data/hash/array I/O before
    # the real SLURM owner/hostname/partition/node guard succeeds.
    require_allocation = _load_allocation_guard()
    allocation = require_allocation()
    args = _parse_args()
    try:
        _run(args, allocation)
    except Exception as exc:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        _write_summary(output / "summary.json", {
            "schema": SCHEMA,
            "status": "failed",
            "allocation": allocation,
            "error": repr(exc),
            "raw_npz": str((output / "raw_gradient.npz").resolve()),
            "engineering": str((output / "engineering.json").resolve()),
            "metrics_deferred_to_root_cpu_verifier": True,
            "unresolved": ["incomplete raw arrays do not support a scientific verdict"],
        })
        raise


if __name__ == "__main__":
    main()
