"""Prepare the frozen DINO-WM Wall recorded-future inputs.

This is a CPU-only preparation step.  The allocation guard is imported and
called before third-party imports, dataset construction, checkpoint hashing,
or any data I/O.  It constructs the official dataset only; it never loads a
model, calls the environment, or runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


ROOT_DEFAULT = Path("/tc1home/UG/yguo017/cem_update_ccds/modelroot")
ASSETDIR_DEFAULT = ROOT_DEFAULT.parent.parent / "v100_newangles_ccds" / "teacher_bias_ready"
SOURCE_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CHECKPOINT_SHA256 = "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b"
CHECKPOINT_EPOCH = 65
RAW_SCHEMA = "dino-wm-wall-recorded-future-raw-v1"
MANIFEST_SCHEMA = "teacher-bias-recorded-future-manifest-v1"
VALID_LOCAL_INDICES = tuple(range(124, 130))
LOCKED_VALID_LOCAL_INDICES = frozenset(range(84, 96))
FRAME_INDICES = (0, 5, 25)
ACTION_COUNT = 25
FRAMESKIP = 5
PRIMITIVE_ACTION_DIM = 2
MODEL_ACTION_DIM = FRAMESKIP * PRIMITIVE_ACTION_DIM
NUM_HIST = 1
NUM_PRED = 1
MAX_SECONDS = 270.0
MAX_MANIFEST_BYTES = 64 * 1024


class PreparationError(RuntimeError):
    """A frozen schema or implementation contract failed."""


class ResourceBlocked(PreparationError):
    """A required existing source asset is unavailable or changed."""


def _short(value: Any, limit: int = 900) -> str:
    text = str(value).replace("\x00", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    raise TypeError(f"cannot encode {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inside(root: Path, candidate: Path, label: str) -> Path:
    root = root.resolve()
    path = candidate.expanduser().resolve()
    if path == root or root not in path.parents:
        raise PreparationError(f"{label} escapes root: {path}")
    return path


def _sha256(path: Path, started: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if started is not None and time.monotonic() - started >= MAX_SECONDS:
                raise TimeoutError(f"preparation deadline reached while hashing {path}")
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreparationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{label} is not an object: {path}")
    return value


def _status(output: Path, allocation: Mapping[str, Any], phase: str, state: str = "running", **extra: Any) -> None:
    value: dict[str, Any] = {
        "schema": "teacher-bias-preparation-status-v1",
        "state": state,
        "phase": phase,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": {key: item for key, item in allocation.items() if key != "user"},
        "updated_epoch": int(time.time()),
    }
    for key, item in extra.items():
        value[key] = _short(item) if key in {"error", "detail"} else item
    _write_json(output / "status.json", value)


def _deadline(started: float, label: str) -> None:
    if time.monotonic() - started >= MAX_SECONDS:
        raise TimeoutError(f"preparation deadline reached at {label}")


def _git(source: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        detail = str(getattr(exc, "stderr", "") or exc)[:350]
        raise ResourceBlocked(f"cannot inspect DINO-WM source git checkout: {source}; {detail}") from exc


def _tensor_array(value: Any, np: Any, label: str) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise PreparationError(f"non-finite {label}")
    return array.copy()


def _shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def _stats(dset: Any, np: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std"):
        if not hasattr(dset, name):
            raise PreparationError(f"official Wall dataset has no {name}")
        result[name] = _tensor_array(getattr(dset, name), np, name).tolist()
    return result


def _resolve_trajectory_mapping(valid: Any) -> tuple[Any, list[int]]:
    """Return the unsliced source dataset and valid-local -> source mapping."""
    indices = getattr(valid, "indices", None)
    base = getattr(valid, "dataset", None)
    if indices is None or base is None:
        raise PreparationError("frozen TrajSubset requires explicit indices and underlying dataset")
    try:
        mapping = [int(item) for item in indices]
    except (TypeError, ValueError) as exc:
        raise PreparationError("valid dataset does not expose integer trajectory indices") from exc
    if len(mapping) != len(valid) or len(set(mapping)) != len(mapping) or any(i < 0 for i in mapping):
        raise PreparationError("invalid, duplicated, or incomplete TrajSubset source mapping")
    return base, mapping


def _source_records(root: Path, source: Path, started: float) -> dict[str, Any]:
    actual_commit = _git(source, "rev-parse", "HEAD")
    if actual_commit != SOURCE_COMMIT:
        raise ResourceBlocked(f"DINO-WM source commit mismatch: {actual_commit} != {SOURCE_COMMIT}")
    modified = _git(source, "diff", "--name-only", "HEAD").splitlines()
    allowed_patches = {"models/dino.py", "env/__init__.py"}
    if set(modified) - allowed_patches:
        raise ResourceBlocked("DINO-WM source has changes outside the two approved runtime adapters")
    # These exact adapters were installed by the earlier CCDS CPU preparation.
    # Preserve them; a clean-tree requirement would reject the approved runtime.
    dino_original = _git(source, "cat-file", "blob", "HEAD:models/dino.py")
    dino_expected = dino_original.replace('"facebookresearch/dinov2", name',
        '"facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8", name')
    wall_registration = ('from gym.envs.registration import register\n'
        'register(id="wall", entry_point="env.wall.wall_env_wrapper:WallEnvWrapper", max_episode_steps=300, reward_threshold=1.0)')
    if (source / "models/dino.py").read_text().strip() != dino_expected.strip():
        raise ResourceBlocked("DINO source differs from the approved commit-pinning adapter")
    if (source / "env/__init__.py").read_text().strip() != wall_registration:
        raise ResourceBlocked("environment registration differs from the approved Wall-only adapter")
    relative_files = (
        "datasets/wall_dset.py",
        "datasets/traj_dset.py",
        "datasets/img_transforms.py",
        "models/visual_world_model.py",
        "plan.py",
        "models/dino.py",
        "env/__init__.py",
    )
    files: dict[str, Any] = {}
    for relative in relative_files:
        path = _inside(root, source / relative, f"source file {relative}")
        if not path.is_file():
            raise ResourceBlocked(f"required source file is missing: {path}")
        files[relative] = {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path, started)}
    return {"commit": actual_commit, "expected_commit": SOURCE_COMMIT, "files": files,
            "approved_runtime_adapters": sorted(allowed_patches), "modified_tracked_paths": modified}


def _file_record(root: Path, path: Path, started: float, label: str) -> dict[str, Any]:
    path = _inside(root, path, label)
    if not path.is_file():
        raise ResourceBlocked(f"required file is missing: {path}")
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path, started)}


def _prepare(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    # Third-party imports intentionally occur only after require_allocation().
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    started = time.monotonic()
    root = Path(args.root).expanduser().resolve()
    campaign_root = root.parent.parent / "v100_newangles_ccds"
    output = _inside(campaign_root, Path(args.output), "job output")
    assetdir = _inside(campaign_root, Path(args.assetdir), "asset directory")
    visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_cuda not in {"", "-1"}:
        raise PreparationError("CPU preparation received a visible CUDA device")
    source = _inside(root, Path(args.source), "DINO-WM source")
    config_path = _inside(root, Path(args.config), "checkpoint config")
    checkpoint_path = _inside(root, Path(args.checkpoint), "checkpoint")
    dataset_root = _inside(root, Path(args.dataset_root), "dataset root")
    output.mkdir(parents=True, exist_ok=True)
    if assetdir.exists():
        raise ResourceBlocked(f"refusing to overwrite existing teacher-bias asset: {assetdir}")
    assetdir.mkdir(parents=True, exist_ok=False)
    _write_json(
        assetdir / "PREPARATION.lock",
        {
            "job_id": allocation.get("job_id"),
            "hostname": socket.gethostname().split(".", 1)[0],
            "partition": allocation.get("partition"),
            "created_epoch": int(time.time()),
        },
    )
    _status(output, allocation, "guard_verified", cuda_visible_devices=visible_cuda)
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise ResourceBlocked("pinned checkpoint config or checkpoint is missing")
    if not source.is_dir() or not dataset_root.is_dir():
        raise ResourceBlocked("pinned DINO-WM source or dataset root is missing")

    source_identity = _source_records(root, source, started)
    _deadline(started, "source identity")
    config_record = _file_record(root, config_path, started, "checkpoint config")
    checkpoint_record = _file_record(root, checkpoint_path, started, "checkpoint")
    if checkpoint_record["sha256"].lower() != CHECKPOINT_SHA256:
        raise ResourceBlocked("checkpoint SHA-256 differs from the frozen epoch-65 identity")
    _status(
        output,
        allocation,
        "runtime_identity_verified",
        source_commit=source_identity["commit"],
        checkpoint_sha256=checkpoint_record["sha256"],
        checkpoint_bytes=checkpoint_record["size"],
    )

    os.environ["DATASET_DIR"] = str(dataset_root)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    import sys

    sys.path.insert(0, str(source))
    model_cfg = OmegaConf.load(config_path)
    actual_settings = {
        "num_hist": int(model_cfg.num_hist),
        "num_pred": int(model_cfg.num_pred),
        "frameskip": int(model_cfg.frameskip),
        "action_emb_dim": int(model_cfg.action_emb_dim),
        "normalize_action": bool(model_cfg.normalize_action),
        "split_mode": str(model_cfg.env.dataset.split_mode),
        "split_ratio": float(model_cfg.env.dataset.split_ratio),
        "dataset_target": str(model_cfg.env.dataset._target_),
    }
    expected_settings = {
        "num_hist": NUM_HIST,
        "num_pred": NUM_PRED,
        "frameskip": FRAMESKIP,
        "action_emb_dim": MODEL_ACTION_DIM,
        "normalize_action": True,
        "split_mode": "random",
        "split_ratio": 0.9,
        "dataset_target": "datasets.wall_dset.load_wall_slice_train_val",
    }
    for key, expected in expected_settings.items():
        if actual_settings[key] != expected:
            raise ResourceBlocked(f"hydra setting {key}={actual_settings[key]!r} != {expected!r}")
    _, trajectory_datasets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=NUM_HIST,
        num_pred=NUM_PRED,
        frameskip=FRAMESKIP,
    )
    if not isinstance(trajectory_datasets, Mapping) or "valid" not in trajectory_datasets:
        raise PreparationError("official loader did not return trajectory_datasets['valid']")
    valid = trajectory_datasets["valid"]
    if len(valid) <= max(VALID_LOCAL_INDICES):
        raise ResourceBlocked(f"valid trajectory count is {len(valid)}, need local index 129")
    base, mapping = _resolve_trajectory_mapping(valid)
    selected_source_ids = [mapping[index] for index in VALID_LOCAL_INDICES]
    if len(set(selected_source_ids)) != len(selected_source_ids):
        raise PreparationError("selected valid-local indices map to duplicate source trajectories")
    if set(VALID_LOCAL_INDICES) & LOCKED_VALID_LOCAL_INDICES:
        raise PreparationError("frozen selection overlaps locked valid-local indices")
    data_path = Path(base.data_path).expanduser().resolve()
    if data_path != dataset_root / "wall_single":
        raise ResourceBlocked(f"official loader resolved unexpected dataset path: {data_path}")
    if not bool(getattr(base, "normalize_action", False)):
        raise ResourceBlocked("official WallDataset did not enable normalize_action")
    if getattr(base, "transform", None) is None:
        raise ResourceBlocked("official WallDataset has no configured image transform")
    if int(base.action_dim) != PRIMITIVE_ACTION_DIM:
        raise ResourceBlocked(
            f"official WallDataset action dimension is {int(base.action_dim)}, expected {PRIMITIVE_ACTION_DIM}"
        )
    stats = _stats(base, np)
    stats_path = assetdir / "dataset_stats.json"
    _write_json(stats_path, {"schema": "dino-wm-wall-dataset-stats-v1", "values": stats})
    _status(output, allocation, "dataset_verified", valid_count=len(valid), dataset_path=str(data_path))

    shared_names = ("states.pth", "actions.pth", "door_locations.pth", "wall_locations.pth")
    dataset_files: dict[str, Any] = {}
    for name in shared_names:
        dataset_files[f"wall_single/{name}"] = _file_record(root, data_path / name, started, f"dataset file {name}")
    records: list[dict[str, Any]] = []
    samples_dir = assetdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, valid_local in enumerate(VALID_LOCAL_INDICES, 1):
        _deadline(started, f"sample {valid_local} start")
        source_id = mapping[valid_local]
        if source_id < 0:
            raise PreparationError(f"invalid underlying trajectory id for valid-local {valid_local}")
        sequence_length = int(base.get_seq_length(source_id))
        if sequence_length <= max(FRAME_INDICES) or sequence_length < ACTION_COUNT:
            raise ResourceBlocked(
                f"trajectory {source_id} is too short for frozen frames/actions: length={sequence_length}"
            )
        episode_path = data_path / "obses" / f"episode_{source_id:03d}.pth"
        dataset_files[f"wall_single/obses/episode_{source_id:03d}.pth"] = _file_record(
            root, episode_path, started, f"selected trajectory observation file {source_id}"
        )
        obs, sampled_actions, states, env_info = base.get_frames(source_id, list(FRAME_INDICES))
        visual = _tensor_array(obs["visual"], np, f"visual trajectory {source_id}")
        proprio = _tensor_array(obs["proprio"], np, f"proprio trajectory {source_id}")
        sampled_actions_array = _tensor_array(sampled_actions, np, f"sampled actions trajectory {source_id}")
        states_array = _tensor_array(states, np, f"states trajectory {source_id}")
        actions25 = _tensor_array(base.actions[source_id, :ACTION_COUNT], np, f"actions trajectory {source_id}")
        if visual.ndim != 4 or visual.shape[0] != len(FRAME_INDICES):
            raise PreparationError(f"visual frame contract mismatch for trajectory {source_id}: {visual.shape}")
        if proprio.shape != (len(FRAME_INDICES), int(base.proprio_dim)):
            raise PreparationError(f"proprio shape mismatch for trajectory {source_id}: {proprio.shape}")
        if states_array.shape != (len(FRAME_INDICES), int(base.state_dim)):
            raise PreparationError(f"state shape mismatch for trajectory {source_id}: {states_array.shape}")
        if actions25.shape != (ACTION_COUNT, int(base.action_dim)):
            raise PreparationError(f"action shape mismatch for trajectory {source_id}: {actions25.shape}")
        model_actions_h5 = actions25.reshape(5, FRAMESKIP, int(base.action_dim)).reshape(5, FRAMESKIP * int(base.action_dim))
        model_actions_h1 = model_actions_h5[:1].copy()
        relative_sample = f"samples/sample_{ordinal:02d}_valid{valid_local:03d}_source{source_id:03d}.npz"
        sample_path = assetdir / relative_sample
        metadata = {
            "schema": RAW_SCHEMA,
            "valid_local_index": valid_local,
            "underlying_trajectory_id": source_id,
            "frame_indices": list(FRAME_INDICES),
            "action_indices": list(range(ACTION_COUNT)),
            "frameskip": FRAMESKIP,
            "num_hist": NUM_HIST,
            "num_pred": NUM_PRED,
            "sequence_length": sequence_length,
            "dataset_path": str(data_path),
            "dataset_normalization": "official WallDataset normalize_action=true; proprio normalized once; no second preprocessor",
            "sampled_frame_actions_semantics": "official normalized actions at frame_indices; actions_0_24_normalized is the model input",
            "visual_transform": str(base.transform),
            "env_info_keys": sorted(str(key) for key in env_info),
            "checkpoint_config_sha256": config_record["sha256"],
            "checkpoint_sha256": checkpoint_record["sha256"],
            "source_commit": SOURCE_COMMIT,
            "dinov2_source_commit": DINOV2_COMMIT,
        }
        arrays = {
            "valid_local_index": np.asarray(valid_local, dtype=np.int64),
            "underlying_trajectory_id": np.asarray(source_id, dtype=np.int64),
            "frame_indices": np.asarray(FRAME_INDICES, dtype=np.int64),
            "action_indices": np.arange(ACTION_COUNT, dtype=np.int64),
            "visual_0_5_25": visual.astype(np.float32, copy=False),
            "proprio_0_5_25": proprio.astype(np.float32, copy=False),
            "state_0_5_25": states_array.astype(np.float32, copy=False),
            "sampled_frame_actions": sampled_actions_array.astype(np.float32, copy=False),
            "actions_0_24_normalized": actions25.astype(np.float32, copy=False),
            "model_actions_h5": model_actions_h5.astype(np.float32, copy=False),
            "model_actions_h1": model_actions_h1.astype(np.float32, copy=False),
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        }
        np.savez_compressed(sample_path, **arrays)
        records.append(
            {
                "sample_path": relative_sample,
                "sha256": _sha256(sample_path, started),
                "size": sample_path.stat().st_size,
                "valid_local_index": valid_local,
                "underlying_trajectory_id": source_id,
                "frame_indices": list(FRAME_INDICES),
                "action_indices": [0, ACTION_COUNT - 1],
                "visual_shape": _shape(visual),
                "proprio_shape": _shape(proprio),
                "state_shape": _shape(states_array),
                "actions_0_24_shape": _shape(actions25),
                "model_actions_h5_shape": _shape(model_actions_h5),
                "observation_file": f"wall_single/obses/episode_{source_id:03d}.pth",
                "source_trajectory_file_sha256": dataset_files[f"wall_single/obses/episode_{source_id:03d}.pth"]["sha256"],
            }
        )
        _status(output, allocation, "samples_written", samples_written=len(records), total_samples=len(VALID_LOCAL_INDICES))

    _deadline(started, "manifest")
    selection = {
        "valid_local_indices": list(VALID_LOCAL_INDICES),
        "locked_valid_local_indices": [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
        "underlying_trajectory_ids": selected_source_ids,
        "frame_rule": "raw frame 0 current, 5 H1 target, 25 H5 target",
        "action_rule": "raw action indices 0..24; five consecutive frameskip=5 groups, each concat to model action dim 10",
        "mapping_rule": "valid trajectory dataset is unsliced; record TrajSubset.indices mapping explicitly",
        "source_split_random_seed": 42,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "raw_schema": RAW_SCHEMA,
        "asset_root": str(assetdir),
        "source_identity": source_identity,
        "checkpoint": {
            "config": config_record,
            "file": checkpoint_record,
            "expected_epoch": CHECKPOINT_EPOCH,
            "expected_sha256": CHECKPOINT_SHA256,
            "dinov2_source_commit": DINOV2_COMMIT,
        },
        "dataset": {
            "root": str(dataset_root),
            "data_path": str(data_path),
            "loader_target": actual_settings["dataset_target"],
            "loader_settings": actual_settings,
            "valid_count": len(valid),
            "normalization": "official WallDataset normalize_action=true; returned proprio/actions used once",
            "stats_path": str(stats_path),
            "stats_sha256": _sha256(stats_path, started),
            "files": dataset_files,
        },
        "selection": selection,
        "mapping": {
            "valid_local_to_underlying": {str(index): mapping[index] for index in VALID_LOCAL_INDICES},
            "visual_key": "visual",
            "proprio_key": "proprio",
            "state_dim": int(base.state_dim),
            "primitive_action_dim": int(base.action_dim),
            "model_action_dim": MODEL_ACTION_DIM,
            "frame_count": len(FRAME_INDICES),
        },
        "array_contract": {
            "visual_0_5_25": {"axes": "frame,channel,height,width", "normalization": "official dataset transform once"},
            "proprio_0_5_25": {"axes": "frame,proprio_dim", "normalization": "official WallDataset once"},
            "state_0_5_25": {"axes": "frame,state_dim", "normalization": "raw stored state"},
            "sampled_frame_actions": {"axes": "frame,primitive_action_dim", "normalization": "official WallDataset once"},
            "actions_0_24_normalized": {"axes": "primitive_step,primitive_action_dim", "indices": "0..24"},
            "model_actions_h5": {"axes": "transition,concatenated_action_dim", "shape": [5, MODEL_ACTION_DIM]},
            "model_actions_h1": {"axes": "transition,concatenated_action_dim", "shape": [1, MODEL_ACTION_DIM]},
        },
        "samples": records,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "inference_ran": False,
        "environment_rollout": False,
        "locked_indices_read_as_samples": [],
        "download_ran": False,
    }
    asset_manifest = assetdir / "manifest.json"
    output_manifest = output / "manifest.json"
    _write_json(asset_manifest, manifest)
    if asset_manifest.stat().st_size >= MAX_MANIFEST_BYTES:
        raise PreparationError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes: {asset_manifest.stat().st_size}")
    _write_json(output_manifest, manifest)
    _status(
        output,
        allocation,
        "complete",
        state="complete",
        samples_written=len(records),
        manifest=str(assetdir / "manifest.json"),
        model_loaded=False,
        inference_ran=False,
    )
    print(json.dumps({"state": "complete", "samples": len(records), "manifest": str(assetdir / "manifest.json")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT_DEFAULT))
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--assetdir", default=str(ASSETDIR_DEFAULT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # This is intentionally the first operation that can inspect scheduler state.
    from allocation_guard import require_allocation

    allocation = require_allocation()
    root_for_error = Path(args.root).expanduser().resolve()
    output_for_error = _inside(root_for_error.parent.parent / "v100_newangles_ccds", Path(args.output), "job output")
    try:
        _prepare(args, allocation)
    except ResourceBlocked as exc:
        _status(output_for_error, allocation, "resource_blocked", state="resource_blocked", error=exc)
        raise SystemExit(4)
    except Exception as exc:
        _status(output_for_error, allocation, "failed", state="failed", error=f"{type(exc).__name__}: {_short(exc)}")
        raise


if __name__ == "__main__":
    main()
