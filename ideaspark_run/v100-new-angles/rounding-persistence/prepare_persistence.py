"""Prepare six raw LIBERO frames for the denoising-persistence screen.

This CPU preparation is deliberately model-free. It selects one fixed middle
frame for task indices 0..5, verifies the existing base/conditional identities,
decodes only the selected camera frames, and writes raw state/action/image
samples. No checkpoint, policy output, environment rollout, download, or
installation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import socket
import time
from pathlib import Path, PurePosixPath
from typing import Any


ROOT_DEFAULT = Path("/tc1home/UG/yguo017/v100_newangles_ccds")
BASE_IDENTITY_DEFAULT = ROOT_DEFAULT / "smolvla" / "identity.json"
EXTENSION_IDENTITY_DEFAULT = ROOT_DEFAULT / "conditional_marginal_ready" / "identity.json"
HELPER_DEFAULT = ROOT_DEFAULT / "artifacts" / "64758" / "smolvla_cpu_prepare.py"
ASSETDIR_DEFAULT = ROOT_DEFAULT / "rounding_persistence_ready"
MODEL_REPO = "lerobot/smolvla_libero"
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
IDENTITY_SCHEMA = "smolvla-cpu-preparation-identity-v1"
RAW_SCHEMA = "smolvla-raw-input-sample-v1"
MANIFEST_SCHEMA = "rounding-persistence-raw-input-manifest-v1"
TASKS = tuple(range(6))
EXCLUDED_EPISODES = (
    0, 18, 22, 1, 4, 5, 2, 3, 34, 6, 38, 40,
    33, 58, 11, 19, 35, 44, 45, 48,
    85, 88, 21, 37, 59, 80, 49, 50,
)
EXCLUDED_SET = frozenset(EXCLUDED_EPISODES)
CAMERA_KEYS_EXPECTED = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM = 8
PHYSICAL_ACTION_DIM = 7
PADDED_ACTION_DIM = 32
MAX_SECONDS = 270.0


class PreparationError(RuntimeError):
    """A frozen input or schema contract failed."""


class ResourceBlocked(PreparationError):
    """A required existing asset is unavailable or changed."""


def _short(value: Any, limit: int = 900) -> str:
    text = str(value).replace("\x00", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreparationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{label} is not a JSON object: {path}")
    return value


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_rel(value: Any) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if not text or path.is_absolute() or "\\" in text or ".." in path.parts:
        raise PreparationError(f"unsafe dataset relative path: {text!r}")
    return str(path)


def _inside(root: Path, path: Path, label: str) -> Path:
    root, path = root.resolve(), path.resolve()
    if path == root or root not in path.parents:
        raise PreparationError(f"{label} escapes campaign root: {path}")
    return path


def _write_status(out: Path, allocation: dict[str, Any], phase: str, state: str = "running", **extra: Any) -> None:
    payload: dict[str, Any] = {
        "schema": "rounding-persistence-preparation-status-v1",
        "state": state,
        "phase": phase,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": {key: value for key, value in allocation.items() if key != "user"},
        "updated_epoch": int(time.time()),
    }
    for key, value in extra.items():
        payload[key] = _short(value) if key in {"error", "detail"} else value
    _json_write(out / "status.json", payload)


def _deadline(started: float, label: str) -> None:
    if time.monotonic() - started >= MAX_SECONDS:
        raise TimeoutError(f"preparation deadline reached at {label}")


def _load_helper(path: Path) -> Any:
    if not path.is_file():
        raise ResourceBlocked(f"fixed Flow helper is missing: {path}")
    spec = importlib.util.spec_from_file_location("fixed_flow_cpu_prepare_64758", path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"cannot import fixed Flow helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_task_rows", "_format_path", "_decode_frame"):
        if not hasattr(module, name):
            raise PreparationError(f"fixed Flow helper lacks required function: {name}")
    return module


def _record_map(identity: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    dataset = identity.get("dataset")
    records = dataset.get(section) if isinstance(dataset, dict) else None
    if not isinstance(records, list):
        raise PreparationError(f"identity dataset.{section} is not a list")
    result: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not item.get("rfilename"):
            raise PreparationError(f"identity dataset.{section} contains an invalid record")
        relative = _safe_rel(item["rfilename"])
        if relative in result and result[relative] != item:
            raise PreparationError(f"duplicate identity record: {relative}")
        result[relative] = item
    return result


def _expected_hash(record: dict[str, Any]) -> str:
    for key in ("downloaded_sha256", "lfs_sha256", "sha256"):
        value = record.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return value.lower()
    raise PreparationError(f"identity record has no verified SHA-256: {record.get('rfilename')!r}")


def _verify_record(
    root: Path,
    extension_root: Path,
    relative: str,
    record: dict[str, Any],
    hashes: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    relative = _safe_rel(relative)
    declared = record.get("verified_local_path")
    if declared is not None:
        expected_path = _inside(extension_root, extension_root / "source_extension" / relative, "extension source")
        path = Path(str(declared)).expanduser().resolve()
        if path != expected_path:
            raise PreparationError(f"extension record path is not bound to source_extension: {relative}")
        layer = "conditional_extension"
    else:
        path = _inside(root, root / "smolvla" / "source_subset" / relative, "base source")
        layer = "base"
    if not path.is_file():
        raise ResourceBlocked(f"identity-listed source file is missing: {relative}")
    expected_size = record.get("size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ResourceBlocked(f"source size mismatch: {relative}")
    cache_key = str(path)
    expected_hash = _expected_hash(record)
    identity = hashes.get(cache_key)
    if identity is None:
        actual_hash = _sha256(path).lower()
        if actual_hash != expected_hash:
            raise ResourceBlocked(f"source SHA-256 mismatch: {relative}")
        identity = {
            "rfilename": relative,
            "path": str(path),
            "layer": layer,
            "size": path.stat().st_size,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
        hashes[cache_key] = identity
    elif identity["actual_sha256"] != expected_hash:
        raise PreparationError(f"conflicting source identity for {relative}")
    return path, dict(identity)


def _validate_identities(root: Path, base_path: Path, extension_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not base_path.is_file():
        raise ResourceBlocked(f"base SmolVLA identity is missing: {base_path}")
    if not extension_path.is_file():
        raise ResourceBlocked(f"conditional extension identity is missing: {extension_path}")
    base_sha = _sha256(base_path)
    extension_sha = _sha256(extension_path)
    if base_sha != 'be4a49ebe588a49a29bd26ed98b8a01a648a247e66d45e12a935ac7d8d0c4e64' or extension_sha != 'f1d55051d660d25d7edfb01522870cdef08650be7752bbe7b52942741d1119ee':
        raise PreparationError('Base or extension identity differs from the preserved campaign pins')
    base = _json_load(base_path, "base identity")
    extension = _json_load(extension_path, "conditional extension identity")
    for label, identity in (("base", base), ("extension", extension)):
        if identity.get("schema") != IDENTITY_SCHEMA:
            raise PreparationError(f"{label} identity schema mismatch")
        checkpoint = identity.get("checkpoint")
        dataset = identity.get("dataset")
        if not isinstance(checkpoint, dict) or checkpoint.get("repo") != MODEL_REPO or checkpoint.get("revision") != MODEL_REVISION:
            raise PreparationError(f"{label} checkpoint identity mismatch")
        if not isinstance(dataset, dict) or dataset.get("repo") != DATASET_REPO or dataset.get("revision") != DATASET_REVISION:
            raise PreparationError(f"{label} dataset identity mismatch")
    base_mapping = base.get("mapping")
    extension_mapping = extension.get("mapping")
    if not isinstance(base_mapping, dict) or not isinstance(extension_mapping, dict):
        raise PreparationError("base/extension mapping is missing")
    for mapping in (base_mapping, extension_mapping):
        if (
            mapping.get("state_key") != STATE_KEY
            or mapping.get("dataset_state_shape") != [STATE_DIM]
            or mapping.get("action_shape") != [PHYSICAL_ACTION_DIM]
            # The frozen SmolVLA identity emitted by the old helper names this
            # checkpoint-side bound ``max_action_dim``.  Accept the explicit
            # manifest alias only when it agrees with that same bound.
            or mapping.get("max_action_dim") != PADDED_ACTION_DIM
            or (
                mapping.get("padded_action_dim") is not None
                and mapping.get("padded_action_dim") != PADDED_ACTION_DIM
            )
        ):
            raise PreparationError("identity does not bind state8, physical action7, padded action32")
        if tuple(mapping.get("present_image_keys") or ()) != CAMERA_KEYS_EXPECTED:
            raise PreparationError("identity camera mapping differs from the frozen two-camera contract")
    if any(base_mapping.get(key) != extension_mapping.get(key) for key in ("state_key", "dataset_state_shape", "action_shape", "present_image_keys")):
        raise PreparationError("extension changed the base state/action/camera mapping")
    bounded = extension.get("bounded_extension")
    if not isinstance(bounded, dict):
        raise ResourceBlocked("conditional extension identity has no bounded_extension record")
    parent_path = Path(str(bounded.get("parent_identity_path", ""))).expanduser().resolve()
    if parent_path != base_path.resolve() or str(bounded.get("parent_identity_sha256", "")).casefold() != base_sha.casefold():
        raise PreparationError("conditional extension does not bind the original base identity")

    sections = ("metadata_files", "task_mapping_source_files", "data_files", "video_files")
    base_maps = {section: _record_map(base, section) for section in sections}
    extension_maps = {section: _record_map(extension, section) for section in sections}
    for section in sections:
        for relative, record in base_maps[section].items():
            if extension_maps[section].get(relative) != record:
                raise PreparationError(f"extension changed an existing base identity record: {section}/{relative}")
    extra_videos = sorted(set(extension_maps["video_files"]) - set(base_maps["video_files"]))
    expected_extra = sorted(f"videos/{camera}/chunk-000/file-002.mp4" for camera in CAMERA_KEYS_EXPECTED)
    if extra_videos != expected_extra:
        raise ResourceBlocked(f"conditional extension video set is not exactly the file-002 pair: {extra_videos}")
    extension_records = bounded.get("files")
    if not isinstance(extension_records, list):
        raise ResourceBlocked("conditional extension bounded files are missing")
    bounded_names = sorted(
        _safe_rel(item["rfilename"]) for item in extension_records
        if isinstance(item, dict) and item.get("rfilename")
    )
    if bounded_names != expected_extra:
        raise ResourceBlocked("conditional extension bounded files do not match the two file-002 videos")
    bounded_by_name = {_safe_rel(item["rfilename"]): item for item in extension_records if isinstance(item, dict) and item.get("rfilename")}
    for relative in expected_extra:
        record = extension_maps["video_files"].get(relative)
        bounded_record = bounded_by_name.get(relative)
        if not isinstance(record, dict) or not isinstance(bounded_record, dict):
            raise ResourceBlocked(f"conditional extension identity lacks {relative}")
        if _expected_hash(record) != _expected_hash(bounded_record) or int(record.get("size", -1)) != int(bounded_record.get("size", -2)):
            raise PreparationError(f"extension file identity disagreement: {relative}")
    return base, extension, {
        "base": {"path": str(base_path.resolve()), "sha256": base_sha, "schema": base.get("schema")},
        "extension": {
            "path": str(extension_path.resolve()),
            "sha256": extension_sha,
            "schema": extension.get("schema"),
            "parent_identity_path": str(base_path.resolve()),
            "parent_identity_sha256": base_sha,
            "bounded_files": expected_extra,
            "probe_sha256": bounded.get("probe_sha256"),
        },
    }


def _atomic_npz(path: Path, np: Any, arrays: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _prepare(args: argparse.Namespace, allocation: dict[str, Any]) -> None:
    import numpy as np
    import pyarrow.parquet as parquet

    started = time.monotonic()
    root = Path(args.root).expanduser().resolve()
    output = _inside(root, Path(args.output).expanduser(), "job output")
    assetdir = _inside(root, Path(args.assetdir).expanduser(), "asset directory")
    output.mkdir(parents=True, exist_ok=True)
    if assetdir.exists():
        raise ResourceBlocked(f"refusing to overwrite existing persistence asset: {assetdir}")
    assetdir.mkdir(parents=True, exist_ok=False)
    _json_write(assetdir / "PREPARATION.lock", {
        "job_id": allocation.get("job_id"), "owner": allocation.get("user"),
        "hostname": socket.gethostname().split(".", 1)[0], "partition": allocation.get("partition"),
        "created_epoch": int(time.time()),
    })
    _write_status(output, allocation, "guard_verified", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}:
        raise PreparationError("CPU preparation has a visible CUDA device")
    helper_path = Path(args.helper).expanduser().resolve()
    helper = _load_helper(helper_path)
    base_path = _inside(root, Path(args.base_identity).expanduser(), "base identity")
    extension_path = _inside(root, Path(args.extension_identity).expanduser(), "extension identity")
    base, extension, identity_summary = _validate_identities(root, base_path, extension_path)
    extension_root = extension_path.parent
    source_hashes: dict[str, dict[str, Any]] = {}
    _write_status(output, allocation, "identities_verified", base_identity_sha256=identity_summary["base"]["sha256"], extension_identity_sha256=identity_summary["extension"]["sha256"])

    metadata_map = _record_map(extension, "metadata_files")
    task_map = _record_map(extension, "task_mapping_source_files")
    data_map = _record_map(extension, "data_files")
    video_map = _record_map(extension, "video_files")
    if "meta/info.json" not in metadata_map or "meta/tasks.parquet" not in metadata_map:
        raise ResourceBlocked("dataset metadata identity lacks meta/info.json or meta/tasks.parquet")
    info_path, _ = _verify_record(root, extension_root, "meta/info.json", metadata_map["meta/info.json"], source_hashes)
    tasks_path, _ = _verify_record(root, extension_root, "meta/tasks.parquet", metadata_map["meta/tasks.parquet"], source_hashes)
    info = _json_load(info_path, "dataset info")
    version = str(info.get("codebase_version", ""))
    if not (version.startswith("v3.") or version.startswith("3.")):
        raise PreparationError(f"dataset codebase_version is not v3.*: {version!r}")
    task_by_id, _ = helper._task_rows(parquet.read_table(tasks_path))
    if any(task_id not in task_by_id or not str(task_by_id[task_id]).strip() for task_id in TASKS):
        raise ResourceBlocked("tasks.parquet does not contain textual task rows 0..5")

    episode_rows: list[dict[str, Any]] = []
    seen_episode_ids: set[int] = set()
    episode_relatives = sorted(name for name in metadata_map if name.startswith("meta/episodes/") and name.endswith(".parquet"))
    if not episode_relatives:
        raise ResourceBlocked("dataset identity lists no episode metadata")
    for relative in episode_relatives:
        path, _ = _verify_record(root, extension_root, relative, metadata_map[relative], source_hashes)
        for row in parquet.read_table(path).to_pylist():
            try:
                episode_id = int(row["episode_index"])
                length = int(row["length"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError(f"episode metadata lacks numeric episode_index/length: {relative}") from exc
            if episode_id in seen_episode_ids:
                raise PreparationError(f"duplicate episode metadata: {episode_id}")
            if length <= 0:
                raise PreparationError(f"episode {episode_id} has non-positive length")
            seen_episode_ids.add(episode_id)
            episode_rows.append(row)
    episode_tasks: dict[int, set[int]] = {}
    for relative in sorted(task_map):
        path, _ = _verify_record(root, extension_root, relative, task_map[relative], source_hashes)
        try:
            table = parquet.read_table(path, columns=["episode_index", "task_index"])
        except Exception as exc:
            raise PreparationError(f"task mapping columns unavailable: {relative}") from exc
        for row in table.to_pylist():
            try:
                episode_tasks.setdefault(int(row["episode_index"]), set()).add(int(row["task_index"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError(f"invalid task mapping row: {relative}") from exc
    if any(len(episode_tasks.get(int(row["episode_index"]), set())) != 1 for row in episode_rows):
        raise PreparationError("episode to task mapping is missing or non-unique")

    chosen: list[dict[str, Any]] = []
    for task_id in TASKS:
        candidates = [
            dict(row) for row in episode_rows
            if next(iter(episode_tasks[int(row["episode_index"])])) == task_id
            and int(row["episode_index"]) not in EXCLUDED_SET
        ]
        candidates.sort(key=lambda row: int(row["episode_index"]))
        if not candidates:
            raise ResourceBlocked(f"no unused episode remains for task {task_id}")
        row = candidates[0]
        row.update({
            "task_index": task_id,
            "task_text": str(task_by_id[task_id]),
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "frame_index": int(row["length"]) // 2,
        })
        chosen.append(row)
    if len(chosen) != len(TASKS) or len({(row["task_index"], row["episode_index"]) for row in chosen}) != len(TASKS):
        raise PreparationError("selection did not produce six unique task/episode pairs")
    selection = {
        "schema": MANIFEST_SCHEMA,
        "rule": "task_index 0..5; exclude the frozen global episode set; choose the first remaining episode_index; zero-based middle frame floor(length/2)",
        "task_indices": list(TASKS),
        "excluded_episode_ids": list(EXCLUDED_EPISODES),
        "selected": [{key: row[key] for key in ("task_index", "task_text", "episode_index", "length", "frame_index")} for row in chosen],
    }
    _json_write(output / "selection.json", selection)
    _write_status(output, allocation, "selection", selected_samples=len(chosen), selected_episode_ids=[row["episode_index"] for row in chosen])

    data_template, video_template = info.get("data_path"), info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise PreparationError("dataset info lacks data_path/video_path templates")
    features = info.get("features")
    if not isinstance(features, dict):
        raise PreparationError("dataset info lacks feature metadata")
    expected_frame_shapes: dict[str, list[int]] = {}
    for camera in CAMERA_KEYS_EXPECTED:
        feature = features.get(camera)
        shape = feature.get("shape") if isinstance(feature, dict) else None
        names = feature.get("names") if isinstance(feature, dict) else None
        if not isinstance(shape, list) or len(shape) != 3 or any(int(value) <= 0 for value in shape):
            raise PreparationError(f"dataset feature shape is missing for {camera}")
        expected_frame_shapes[camera] = [int(shape[2]), int(shape[0]), int(shape[1])] if names == ["height", "width", "channel"] else [int(value) for value in shape]
    allowed_data = dict(task_map)
    allowed_data.update(data_map)
    fps = float(info.get("fps", 0))
    if not fps > 0:
        raise PreparationError("dataset fps is missing or non-positive")
    tolerance = max(1e-4, 0.5 / fps)
    sample_records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(chosen, 1):
        _deadline(started, f"sample {ordinal} start")
        try:
            data_relative = helper._format_path(
                data_template,
                chunk_index=row["data/chunk_index"], file_index=row["data/file_index"],
                episode_chunk=row["data/chunk_index"], file_chunk=row["data/file_index"],
            )
        except Exception as exc:
            raise PreparationError(f"cannot resolve data pointer for episode {row['episode_index']}") from exc
        if data_relative not in allowed_data:
            raise ResourceBlocked(f"selected data file is not in verified identity: {data_relative}")
        data_path, data_identity = _verify_record(root, extension_root, data_relative, allowed_data[data_relative], source_hashes)
        columns = ["episode_index", "frame_index", "timestamp", "task_index", STATE_KEY, ACTION_KEY]
        try:
            data_rows = parquet.read_table(data_path, columns=columns).to_pylist()
        except Exception as exc:
            raise ResourceBlocked(f"selected data columns unavailable: {data_relative}") from exc
        matches = [item for item in data_rows if int(item["episode_index"]) == row["episode_index"] and int(item["frame_index"]) == row["frame_index"]]
        if len(matches) != 1:
            raise ResourceBlocked(f"fixed middle frame is unavailable for episode {row['episode_index']}: {len(matches)} matches")
        data_row = matches[0]
        if int(data_row["task_index"]) != row["task_index"]:
            raise PreparationError(f"data task mismatch for episode {row['episode_index']}")
        state = np.asarray(data_row[STATE_KEY], dtype=np.float32)
        action = np.asarray(data_row[ACTION_KEY], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise PreparationError(f"selected state is not finite float32[{STATE_DIM}]: episode {row['episode_index']}")
        if action.shape != (PHYSICAL_ACTION_DIM,) or not np.isfinite(action).all():
            raise PreparationError(f"selected action is not finite float32[{PHYSICAL_ACTION_DIM}]: episode {row['episode_index']}")
        timestamp = float(data_row["timestamp"])
        if not np.isfinite(timestamp) or timestamp < 0:
            raise PreparationError(f"selected timestamp is invalid: episode {row['episode_index']}")
        arrays: dict[str, Any] = {
            STATE_KEY: state,
            ACTION_KEY: action,
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
            "episode_index": np.asarray(row["episode_index"], dtype=np.int64),
            "frame_index": np.asarray(row["frame_index"], dtype=np.int64),
            "timestamp": np.asarray(timestamp, dtype=np.float64),
        }
        video_paths: dict[str, str] = {}
        video_timestamps: dict[str, float] = {}
        source_identities: dict[str, dict[str, Any]] = {data_relative: data_identity}
        for camera in CAMERA_KEYS_EXPECTED:
            prefix = f"videos/{camera}/"
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                if prefix + suffix not in row:
                    raise ResourceBlocked(f"episode {row['episode_index']} lacks video pointer {prefix + suffix}")
            try:
                video_relative = helper._format_path(
                    video_template, video_key=camera,
                    chunk_index=row[prefix + "chunk_index"], file_index=row[prefix + "file_index"],
                    episode_chunk=row[prefix + "chunk_index"], file_chunk=row[prefix + "file_index"],
                )
            except Exception as exc:
                raise PreparationError(f"cannot resolve {camera} path for episode {row['episode_index']}") from exc
            if video_relative not in video_map:
                raise ResourceBlocked(f"selected video is not in verified identity: {video_relative}")
            video_path, video_identity = _verify_record(root, extension_root, video_relative, video_map[video_relative], source_hashes)
            video_timestamp = float(row[prefix + "from_timestamp"]) + timestamp
            video_end = float(row[prefix + "to_timestamp"])
            if not np.isfinite(video_timestamp) or not np.isfinite(video_end) or video_timestamp < -tolerance or video_timestamp > video_end + tolerance:
                raise ResourceBlocked(f"selected frame timestamp is outside video range: {video_relative}")
            frame, conversion = helper._decode_frame(video_path, video_timestamp, tolerance)
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[0] != 3:
                raise PreparationError(f"decoded frame is not uint8 CHW3: {video_relative}")
            if list(frame.shape) != expected_frame_shapes[camera]:
                raise PreparationError(
                    f"decoded frame shape differs from dataset feature for {camera}: "
                    f"{list(frame.shape)} vs {expected_frame_shapes[camera]}"
                )
            arrays[camera] = frame
            video_paths[camera] = video_relative
            video_timestamps[camera] = video_timestamp
            source_identities[video_relative] = video_identity
            _deadline(started, f"sample {ordinal} {camera}")
        metadata = {
            "schema": RAW_SCHEMA,
            "task_index": row["task_index"], "task_text": row["task_text"],
            "episode_index": row["episode_index"], "dataset_episode_index": row["episode_index"],
            "frame_index": row["frame_index"], "frame_rule": "floor(length/2)",
            "length": row["length"], "timestamp": timestamp,
            "state_key": STATE_KEY, "state_shape": [STATE_DIM],
            "action_key": ACTION_KEY, "action_shape": [PHYSICAL_ACTION_DIM],
            "padded_action_dim": PADDED_ACTION_DIM, "camera_keys": list(CAMERA_KEYS_EXPECTED),
            "camera_shapes": expected_frame_shapes,
            "video_timestamps": video_timestamps,
            "data_path": data_relative, "video_paths": video_paths,
            "source_identities": source_identities,
            "dataset_repo": DATASET_REPO, "dataset_revision": DATASET_REVISION,
            "base_identity_sha256": identity_summary["base"]["sha256"],
            "extension_identity_sha256": identity_summary["extension"]["sha256"],
            "processor_note": "GPU runner owns checkpoint preprocessing; this NPZ remains raw.",
            "model_loaded": False, "checkpoint_loaded": False, "environment_steps": 0,
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        sample_path = assetdir / "samples" / f"sample_{ordinal:02d}_task{row['task_index']:03d}_episode{row['episode_index']:04d}.npz"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        if sample_path.exists():
            raise PreparationError(f"refusing to overwrite sample: {sample_path}")
        _atomic_npz(sample_path, np, arrays)
        sample_records.append({
            "sample_path": str(sample_path.relative_to(assetdir)).replace(os.sep, "/"),
            "sha256": _sha256(sample_path), "size": sample_path.stat().st_size,
            "task_index": row["task_index"], "task_text": row["task_text"],
            "episode_index": row["episode_index"], "dataset_episode_index": row["episode_index"],
            "length": row["length"], "frame_index": row["frame_index"], "timestamp": timestamp,
            "state_key": STATE_KEY, "state_shape": [STATE_DIM],
            "action_key": ACTION_KEY, "action_shape": [PHYSICAL_ACTION_DIM],
            "camera_keys": list(CAMERA_KEYS_EXPECTED), "data_path": data_relative, "video_paths": video_paths,
            "video_timestamps": video_timestamps, "source_identities": source_identities,
        })
        _write_status(output, allocation, "samples_written", samples_written=len(sample_records), total_samples=len(TASKS))
    if len(sample_records) != len(TASKS):
        raise PreparationError("sample count is not six")
    source_records = sorted(source_hashes.values(), key=lambda item: item["path"])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "raw_schema": RAW_SCHEMA,
        "asset_root": str(assetdir),
        "base_identity_path": identity_summary["base"]["path"],
        "base_identity_sha256": identity_summary["base"]["sha256"],
        "extension_identity_path": identity_summary["extension"]["path"],
        "extension_identity_sha256": identity_summary["extension"]["sha256"],
        "identities": identity_summary,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "codebase_version": version},
        "selection": selection,
        "mapping": {
            "camera_keys": list(CAMERA_KEYS_EXPECTED), "state_key": STATE_KEY, "state_shape": [STATE_DIM],
            "action_key": ACTION_KEY, "physical_action_dim": PHYSICAL_ACTION_DIM,
            "padded_action_dim": PADDED_ACTION_DIM, "camera_shapes": expected_frame_shapes,
        },
        "decode": {"tolerance_s": tolerance, "frame_contract": "uint8 CHW3", "middle_frame_rule": "floor(length/2)"},
        "fixed_flow_helper": {"path": str(helper_path), "sha256": _sha256(helper_path)},
        "source_files": source_records,
        "samples": sample_records,
        "inference_ran": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "environment_rollout": False,
        "download_ran": False,
    }
    _json_write(assetdir / "manifest.json", manifest)
    _json_write(output / "manifest.json", manifest)
    _write_status(output, allocation, "complete", state="complete", samples_written=len(sample_records), manifest=str(assetdir / "manifest.json"), inference_ran=False)
    print(json.dumps({"state": "complete", "samples": len(sample_records), "manifest": str(assetdir / "manifest.json")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT_DEFAULT))
    parser.add_argument("--base-identity", default=str(BASE_IDENTITY_DEFAULT))
    parser.add_argument("--extension-identity", default=str(EXTENSION_IDENTITY_DEFAULT))
    parser.add_argument("--helper", default=str(HELPER_DEFAULT))
    parser.add_argument("--assetdir", default=str(ASSETDIR_DEFAULT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation

    allocation = require_allocation()
    # Validate the error-report destination before any failure handler can
    # write it.  A malformed outside-root argument must never create a status
    # file outside the campaign root.
    root_for_error = Path(args.root).expanduser().resolve()
    output = _inside(root_for_error, Path(args.output).expanduser(), "job output")
    try:
        _prepare(args, allocation)
    except ResourceBlocked as exc:
        _write_status(output, allocation, "resource_blocked", state="resource_blocked", error=exc)
        raise SystemExit(4)
    except Exception as exc:
        _write_status(output, allocation, "failed", state="failed", error=f"{type(exc).__name__}: {_short(exc)}")
        raise


if __name__ == "__main__":
    main()
