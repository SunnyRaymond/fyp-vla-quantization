"""Prepare eight raw samples for the conditional-action-distribution screen.

This is an offline CPU preparation only.  It never downloads, installs,
loads a policy, or reads model output.  The allocation guard is deliberately
called before importing the snapshot helper or third-party data/video code.
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
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
MODEL_REPO = "lerobot/smolvla_libero"
DATASET_REPO = "lerobot/libero"
RAW_SCHEMA = "smolvla-raw-input-sample-v1"
MANIFEST_SCHEMA = "conditional-marginal-raw-input-manifest-v1"
SAMPLE_COUNT = 8
ASSETDIR_NAME = "conditional_marginal"
PADDED_MANIFEST_NAME = "padded_feedback/manifest.json"


class PreparationError(RuntimeError):
    pass


class ResourceBlocked(PreparationError):
    pass


def _short(value: Any, limit: int = 900) -> str:
    text = str(value).replace("\x00", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_rel(value: Any) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if not text or path.is_absolute() or "\\" in text or ".." in path.parts:
        raise PreparationError(f"unsafe source path: {text!r}")
    return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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


def _root_file(root: Path, value: str, label: str) -> Path:
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if root.resolve() not in path.parents:
        raise PreparationError(f"{label} escapes root: {path}")
    return path


def _excluded_manifest(path: Path, label: str, schema: str, count: int, per_task: int) -> dict[str, Any]:
    if not path.is_file():
        raise ResourceBlocked(f"{label} is missing: {path}")
    raw = _load_json(path, label)
    if raw.get("schema") != schema:
        raise PreparationError(f"{label} schema mismatch: {raw.get('schema')!r}")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) != count:
        raise PreparationError(f"{label} must contain exactly {count} samples")
    pairs: list[tuple[int, int]] = []
    for index, entry in enumerate(samples):
        if not isinstance(entry, dict):
            raise PreparationError(f"{label} sample {index} is malformed")
        try:
            task = int(entry["task_index"])
            episode_value = entry.get("dataset_episode_index")
            if episode_value is None:
                episode_value = entry["episode_index"]
            episode = int(episode_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparationError(f"{label} sample {index} lacks task/episode") from exc
        if task not in range(4) or episode < 0:
            raise PreparationError(f"{label} sample {index} has invalid task/episode")
        pairs.append((task, episode))
    if len(set(pairs)) != count or any(sum(task == value for task, _ in pairs) != per_task for value in range(4)):
        raise PreparationError(f"{label} does not contain {per_task} distinct pairs per task")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema": schema,
        "pairs": [{"task_index": task, "episode_index": episode} for task, episode in sorted(pairs)],
        "episode_ids": sorted({episode for _, episode in pairs}),
    }


def _record_map(identity: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    value = identity.get("dataset", {}).get(section, [])
    if not isinstance(value, list):
        raise PreparationError(f"identity dataset.{section} is not a list")
    result: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict) or not item.get("rfilename"):
            raise PreparationError(f"identity dataset.{section} has an invalid record")
        name = _safe_rel(item["rfilename"])
        if name in result and result[name] != item:
            raise PreparationError(f"duplicate identity record: {name}")
        result[name] = item
    return result


def _verify_source(root: Path, relative: str, record: dict[str, Any], hashes: dict[str, str]) -> Path:
    relative = _safe_rel(relative)
    path = root / "smolvla" / "source_subset" / relative
    if record.get("verified_local_path"):
        path = Path(record["verified_local_path"]).resolve()
        allowed = (root / "conditional_marginal_ready" / "source_extension").resolve()
        if allowed not in path.parents or path != allowed / relative:
            raise PreparationError("source extension path does not bind declared file")
    if not path.is_file():
        raise ResourceBlocked(f"identity-listed source file is missing: {relative}")
    expected_size = record.get("size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ResourceBlocked(f"identity-listed source size mismatch: {relative}")
    actual = hashes.get(relative)
    if actual is None:
        actual = _sha256(path)
        hashes[relative] = actual
    expected = record.get("downloaded_sha256") or record.get("lfs_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise PreparationError(f"identity has no verified SHA-256 for {relative}")
    if actual.lower() != expected.lower():
        raise ResourceBlocked(f"identity-listed source hash mismatch: {relative}")
    return path


def _acquire_lock(assetdir: Path, allocation: dict[str, Any]) -> None:
    lock = assetdir / "PREPARATION.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PreparationError(f"single-writer lock already exists: {lock}") from exc
    owner = {
        "job_id": allocation.get("job_id"),
        "owner": allocation.get("user"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "partition": allocation.get("partition"),
        "created_epoch": int(time.time()),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(owner, sort_keys=True) + "\n")


def _status(out: Path, assetdir: Path, allocation: dict[str, Any], phase: str, state: str = "running", write_asset: bool = True, **extra: Any) -> None:
    value: dict[str, Any] = {
        "schema": "conditional-marginal-preparation-status-v1",
        "state": state,
        "phase": phase,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": {key: item for key, item in allocation.items() if key != "user"},
        "updated_epoch": int(time.time()),
    }
    for key, item in extra.items():
        if key in {"error", "detail"}:
            value[key] = _short(item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            value[key] = item
        elif isinstance(item, list):
            value[key] = [_short(x, 180) for x in item[:16]]
    _json_dump(out / "status.json", value)
    if write_asset:
        _json_dump(assetdir / "status.json", value)


def _load_snapshot_helper(root: Path) -> Any:
    path = root / "artifacts" / "64758" / "smolvla_cpu_prepare.py"
    if not path.is_file():
        raise PreparationError(f"verified helper snapshot is missing: {path}")
    spec = importlib.util.spec_from_file_location("smolvla_prepare_snapshot_64758", path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"cannot load helper snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_task_rows", "_format_path", "_decode_frame"):
        if not hasattr(module, name):
            raise PreparationError(f"helper snapshot lacks required pure helper: {name}")
    return module


def _main(args: argparse.Namespace, allocation: dict[str, Any], helper: Any) -> None:
    import numpy as np
    import pyarrow.parquet as parquet

    root = Path(args.root).expanduser().resolve()
    assetdir = Path(args.assetdir).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    assetdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    _acquire_lock(assetdir, allocation)
    if (assetdir / "manifest.json").exists() or list((assetdir / "samples").glob("*.npz")):
        raise PreparationError("existing manifest/sample found; refusing overwrite")
    _status(out, assetdir, allocation, "guard_verified", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}:
        raise PreparationError("CPU preparation received a visible CUDA device")

    identity_path = Path(args.identity_file).resolve() if args.identity_file else root / "smolvla" / "identity.json"
    if root not in identity_path.parents:
        raise PreparationError("identity path escapes campaign root")
    if not identity_path.is_file():
        raise ResourceBlocked(f"base identity is missing: {identity_path}")
    base_identity_sha256 = _sha256(identity_path)
    identity = _load_json(identity_path, "base identity")
    if identity.get("schema") != "smolvla-cpu-preparation-identity-v1":
        raise PreparationError("base identity schema mismatch")
    checkpoint, dataset = identity.get("checkpoint", {}), identity.get("dataset", {})
    if checkpoint.get("repo") != MODEL_REPO or checkpoint.get("revision") != MODEL_REVISION:
        raise PreparationError("base identity checkpoint pin mismatch")
    if dataset.get("repo") != DATASET_REPO or dataset.get("revision") != DATASET_REVISION:
        raise PreparationError("base identity dataset pin mismatch")
    mapping = identity.get("mapping")
    if not isinstance(mapping, dict) or mapping.get("state_key") != "observation.state":
        raise PreparationError("base identity state mapping is not exact")
    if mapping.get("dataset_state_shape") != [8] or mapping.get("action_shape") != [7]:
        raise PreparationError("base identity does not pin state8/action7")
    cameras = mapping.get("present_image_keys")
    if not isinstance(cameras, list) or len(cameras) != 2 or len(set(cameras)) != 2:
        raise PreparationError("base identity does not pin exactly two present cameras")
    _status(out, assetdir, allocation, "base_identity_verified", base_identity_sha256=base_identity_sha256)

    meta = _record_map(identity, "metadata_files")
    task_sources = _record_map(identity, "task_mapping_source_files")
    data_sources = _record_map(identity, "data_files")
    video_sources = _record_map(identity, "video_files")
    source_hashes: dict[str, str] = {}
    info_rel, tasks_rel = "meta/info.json", "meta/tasks.parquet"
    if info_rel not in meta or tasks_rel not in meta:
        raise ResourceBlocked("base identity does not list required dataset metadata")
    info_path = _verify_source(root, info_rel, meta[info_rel], source_hashes)
    tasks_path = _verify_source(root, tasks_rel, meta[tasks_rel], source_hashes)
    info = _load_json(info_path, "dataset info")
    version = str(info.get("codebase_version", ""))
    if not (version.startswith("v3.") or version.startswith("3.")):
        raise PreparationError(f"dataset codebase_version is not v3.*: {version!r}")
    task_by_id, task_by_text = helper._task_rows(parquet.read_table(tasks_path))
    del task_by_text
    if sorted(task_by_id)[:4] != [0, 1, 2, 3]:
        raise PreparationError("tasks metadata does not contain exact task indices 0..3")

    episode_rows: list[dict[str, Any]] = []
    episode_paths = [name for name in meta if name.startswith("meta/episodes/") and name.endswith(".parquet")]
    if not episode_paths:
        raise ResourceBlocked("base identity lists no episode metadata")
    seen_episode_rows: set[int] = set()
    for relative in sorted(episode_paths):
        path = _verify_source(root, relative, meta[relative], source_hashes)
        for row in parquet.read_table(path).to_pylist():
            try:
                episode_id = int(row["episode_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError("episode metadata lacks numeric episode_index") from exc
            if episode_id in seen_episode_rows:
                raise PreparationError(f"duplicate episode metadata: {episode_id}")
            seen_episode_rows.add(episode_id)
            episode_rows.append(row)

    episode_tasks: dict[int, set[int]] = {}
    for relative in sorted(task_sources):
        path = _verify_source(root, relative, task_sources[relative], source_hashes)
        try:
            table = parquet.read_table(path, columns=["episode_index", "task_index"])
        except Exception as exc:
            raise PreparationError(f"task mapping columns unavailable: {relative}") from exc
        for row in table.to_pylist():
            try:
                episode_id, task_id = int(row["episode_index"]), int(row["task_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError(f"invalid task mapping row: {relative}") from exc
            episode_tasks.setdefault(episode_id, set()).add(task_id)
    if any(len(episode_tasks.get(int(row["episode_index"]), set())) != 1 for row in episode_rows):
        raise PreparationError("episode to task mapping is missing or non-unique")

    flow_manifest_path = root / "smolvla" / "manifest.json"
    padded_manifest_path = _root_file(root, args.padded_manifest, "padded manifest")
    flow_excluded = _excluded_manifest(flow_manifest_path, "Flow old manifest", "smolvla-raw-input-manifest-v1", 12, 3)
    padded_excluded = _excluded_manifest(padded_manifest_path, "padded manifest", "padded-coordinate-raw-input-manifest-v1", 8, 2)
    flow_pairs = {(int(item["task_index"]), int(item["episode_index"])) for item in flow_excluded["pairs"]}
    padded_pairs = {(int(item["task_index"]), int(item["episode_index"])) for item in padded_excluded["pairs"]}
    if flow_pairs & padded_pairs:
        raise PreparationError("Flow and padded exclusion manifests overlap")
    excluded_pairs = flow_pairs | padded_pairs
    if len(excluded_pairs) != 20 or any(sum(task == t for task, _ in excluded_pairs) != 5 for t in range(4)):
        raise PreparationError("combined exclusion does not contain exactly five pairs per task")
    excluded_ids = sorted({episode for _, episode in excluded_pairs})
    excluded_manifests = {"flow": flow_excluded, "padded": padded_excluded}

    chosen: list[dict[str, Any]] = []
    for task_id in range(4):
        candidates = [
            dict(row) for row in episode_rows
            if next(iter(episode_tasks[int(row["episode_index"])])) == task_id
            and (task_id, int(row["episode_index"])) not in excluded_pairs
        ]
        candidates.sort(key=lambda row: int(row["episode_index"]))
        if len(candidates) < 2:
            raise PreparationError(f"task {task_id} has fewer than two unused episodes")
        for row in candidates[:2]:
            episode_id, length = int(row["episode_index"]), int(row["length"])
            if length <= 0:
                raise PreparationError(f"episode {episode_id} has non-positive length")
            row.update({"task_index": task_id, "task_text": task_by_id[task_id], "episode_index": episode_id, "length": length, "frame_index": length // 4})
            chosen.append(row)
    if len(chosen) != SAMPLE_COUNT or len({(r["task_index"], r["episode_index"]) for r in chosen}) != SAMPLE_COUNT:
        raise PreparationError("frozen selection did not produce eight unique samples")
    _json_dump(out / "selection.json", {
        "schema": MANIFEST_SCHEMA,
        "rule": "task_index 0..3; exclude Flow manifest 3 plus padded manifest 2 per task; episode_index ascending first two unused; zero-based floor(length/4)",
        "task_indices": [0, 1, 2, 3],
        "excluded_manifests": excluded_manifests,
        "excluded_episode_ids": excluded_ids,
        "excluded_episode_pairs": [{"task_index": t, "episode_index": e} for t, e in sorted(excluded_pairs)],
        "selected": [{"task_index": r["task_index"], "episode_index": r["episode_index"], "length": r["length"], "frame_index": r["frame_index"]} for r in chosen],
    })
    _status(out, assetdir, allocation, "selection", selected_samples=SAMPLE_COUNT)

    data_template, video_template = info.get("data_path"), info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise PreparationError("dataset info lacks data_path/video_path templates")
    allowed_data = dict(task_sources)
    allowed_data.update(data_sources)
    plans: list[dict[str, Any]] = []
    for row in chosen:
        try:
            data_rel = helper._format_path(
                data_template, chunk_index=row["data/chunk_index"], file_index=row["data/file_index"],
                episode_chunk=row["data/chunk_index"], file_chunk=row["data/file_index"],
            )
        except Exception as exc:
            raise PreparationError(f"cannot resolve data pointer for episode {row['episode_index']}") from exc
        if data_rel not in allowed_data:
            raise ResourceBlocked(f"selected data file is not in verified identity: {data_rel}")
        videos: dict[str, str] = {}
        for camera in cameras:
            prefix = f"videos/{camera}/"
            needed = [prefix + "chunk_index", prefix + "file_index", prefix + "from_timestamp"]
            if any(key not in row for key in needed):
                raise ResourceBlocked(f"selected episode lacks video pointer: {row['episode_index']}:{camera}")
            video_rel = helper._format_path(
                video_template, video_key=camera, chunk_index=row[prefix + "chunk_index"],
                file_index=row[prefix + "file_index"], episode_chunk=row[prefix + "chunk_index"],
                file_chunk=row[prefix + "file_index"],
            )
            if video_rel not in video_sources:
                raise ResourceBlocked(f"selected video is not in verified identity: {video_rel}")
            videos[camera] = video_rel
        plans.append({"row": row, "data_path": data_rel, "videos": videos})

    for plan in plans:
        _verify_source(root, plan["data_path"], allowed_data[plan["data_path"]], source_hashes)
        for relative in plan["videos"].values():
            _verify_source(root, relative, video_sources[relative], source_hashes)
    _status(out, assetdir, allocation, "resource_verified", source_files=len(source_hashes))

    fps = float(info.get("fps", 0))
    if not fps > 0:
        raise PreparationError("dataset fps is missing or non-positive")
    tolerance = max(1e-4, 0.5 / fps)
    samples_dir = assetdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    conversions: set[str] = set()
    for ordinal, plan in enumerate(sorted(plans, key=lambda item: (item["row"]["task_index"], item["row"]["episode_index"])), 1):
        row = plan["row"]
        data_path = root / "smolvla" / "source_subset" / plan["data_path"]
        columns = ["episode_index", "frame_index", "timestamp", "task_index", mapping["state_key"]]
        try:
            rows = parquet.read_table(data_path, columns=columns).to_pylist()
        except Exception as exc:
            raise PreparationError(f"selected data columns unavailable: {plan['data_path']}") from exc
        matches = [item for item in rows if int(item["episode_index"]) == row["episode_index"] and int(item["frame_index"]) == row["frame_index"]]
        if len(matches) != 1:
            raise PreparationError(f"expected one selected data row for episode {row['episode_index']}, got {len(matches)}")
        data_row = matches[0]
        if int(data_row["task_index"]) != row["task_index"]:
            raise PreparationError(f"data task mismatch for episode {row['episode_index']}")
        state = np.asarray(data_row[mapping["state_key"]], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise PreparationError(f"selected state is not finite float32 state8: {row['episode_index']}")
        timestamp = float(data_row["timestamp"])
        arrays: dict[str, Any] = {mapping["state_key"]: state}
        video_timestamps: dict[str, float] = {}
        for camera, relative in plan["videos"].items():
            prefix = f"videos/{camera}/"
            video_timestamp = float(row[prefix + "from_timestamp"]) + timestamp
            video_path = _verify_source(root, relative, video_sources[relative], source_hashes)
            frame, conversion = helper._decode_frame(video_path, video_timestamp, tolerance)
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[0] != 3:
                raise PreparationError(f"decoded frame is not uint8 CHW3: {relative}")
            arrays[camera] = frame
            video_timestamps[camera] = video_timestamp
            conversions.add(conversion)
        metadata = {
            "schema": RAW_SCHEMA, "task_index": row["task_index"], "task_text": row["task_text"],
            "episode_index": row["episode_index"], "frame_index": row["frame_index"], "timestamp": timestamp,
            "state_key": mapping["state_key"], "camera_keys": cameras, "missing_checkpoint_image_keys": mapping.get("missing_checkpoint_image_keys", []),
            "video_timestamps": video_timestamps,
            "data_path": plan["data_path"], "video_paths": plan["videos"], "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION, "checkpoint_repo": MODEL_REPO,
            "checkpoint_revision": MODEL_REVISION, "base_identity_sha256": base_identity_sha256,
            "source_file_sha256": {plan["data_path"]: source_hashes[plan["data_path"]], **{p: source_hashes[p] for p in plan["videos"].values()}},
            "processor_note": "GPU runner applies checkpoint processor; this NPZ remains raw.",
        }
        arrays.update({
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
            "episode_index": np.asarray(row["episode_index"], dtype=np.int64),
            "frame_index": np.asarray(row["frame_index"], dtype=np.int64),
            "timestamp": np.asarray(timestamp, dtype=np.float64),
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        })
        for camera, video_timestamp in video_timestamps.items():
            arrays["video_timestamp__" + re.sub(r"[^A-Za-z0-9_.-]", "_", camera)] = np.asarray(video_timestamp, dtype=np.float64)
        sample_path = samples_dir / f"sample_{ordinal:02d}_task{row['task_index']:03d}_episode{row['episode_index']:04d}.npz"
        if sample_path.exists():
            raise PreparationError(f"refusing to overwrite sample: {sample_path}")
        np.savez_compressed(sample_path, **arrays)
        records.append({
            "sample_path": str(sample_path.relative_to(assetdir)).replace(os.sep, "/"),
            "sha256": _sha256(sample_path), "size": sample_path.stat().st_size,
            "task_index": row["task_index"], "task_text": row["task_text"],
            "episode_index": row["episode_index"], "dataset_episode_index": row["episode_index"],
            "length": row["length"], "frame_index": row["frame_index"], "timestamp": timestamp,
            "camera_keys": cameras, "state_key": mapping["state_key"],
            "data_path": plan["data_path"], "video_paths": plan["videos"],
            "video_timestamps": video_timestamps,
        })
        _status(out, assetdir, allocation, "samples_written", samples_written=len(records), total_samples=SAMPLE_COUNT)

    if len(records) != SAMPLE_COUNT:
        raise PreparationError("sample count is not eight")
    manifest = {
        "schema": MANIFEST_SCHEMA, "raw_schema": RAW_SCHEMA, "asset_root": str(assetdir),
        "base_identity_path": str(identity_path), "base_identity_sha256": base_identity_sha256,
        "checkpoint": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "codebase_version": version},
        "excluded_manifests": excluded_manifests,
        "selection": {
            "task_indices": [0, 1, 2, 3], "episodes_per_task": 2,
            "rule": "exclude Flow old manifest 3 plus padded manifest 2 per task; episode_index ascending first two unused; zero-based floor(length/4)",
            "excluded_manifests": excluded_manifests,
            "excluded_episode_ids": excluded_ids,
            "excluded_episode_pairs": [{"task_index": t, "episode_index": e} for t, e in sorted(excluded_pairs)],
            "selected": [{"task_index": r["task_index"], "episode_index": r["episode_index"], "length": r["length"], "frame_index": r["frame_index"]} for r in chosen],
        },
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "mapping": {"camera_keys": cameras, "state_key": mapping["state_key"], "state_shape": [8], "physical_action_dim": 7, "padded_action_dim": 32},
        "decode": {"tolerance_s": tolerance, "conversions": sorted(conversions), "frame_contract": "uint8 CHW3"},
        "samples": records, "inference_ran": False,
    }
    _json_dump(assetdir / "manifest.json", manifest)
    _json_dump(out / "manifest.json", manifest)
    _status(out, assetdir, allocation, "complete", state="complete", samples_written=SAMPLE_COUNT, manifest="manifest.json", inference_ran=False)
    print(json.dumps({"state": "complete", "samples": SAMPLE_COUNT, "manifest": str(assetdir / "manifest.json")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT_DEFAULT))
    parser.add_argument("--identity-file", default=None)
    parser.add_argument("--assetdir", default=str(ROOT_DEFAULT / ASSETDIR_NAME))
    parser.add_argument("--padded-manifest", default=PADDED_MANIFEST_NAME)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation

    allocation = require_allocation()
    try:
        helper = _load_snapshot_helper(Path(args.root).expanduser().resolve())
        _main(args, allocation, helper)
    except ResourceBlocked as exc:
        _status(Path(args.output).expanduser().resolve(), Path(args.assetdir).expanduser().resolve(), allocation, "resource_blocked", state="resource_blocked", write_asset=False, error=exc)
        raise SystemExit(4)
    except Exception as exc:
        _status(Path(args.output).expanduser().resolve(), Path(args.assetdir).expanduser().resolve(), allocation, "failed", state="failed", write_asset=False, error=f"{type(exc).__name__}: {_short(exc)}")
        raise


if __name__ == "__main__":
    main()
