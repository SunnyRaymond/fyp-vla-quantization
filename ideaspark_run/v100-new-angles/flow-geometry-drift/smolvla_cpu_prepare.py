"""Prepare a bounded, raw-input SmolVLA sample set on a CCDS CPU allocation.

This program deliberately does not import or instantiate a policy.  It resolves
public Hub revisions, downloads only the checkpoint metadata/weights and the
metadata plus files needed by the frozen 12-episode selection, decodes one real
frame per episode, and writes raw ``npz`` samples for a later GPU runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MODEL_REPO = "lerobot/smolvla_libero"
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
BASE_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"

MODEL_CAP = 1_500_000_000
META_CAP = 200_000_000
SUBSET_CAP = 3_000_000_000
TOTAL_CAP = 5_000_000_000
SAMPLE_COUNT = 12
TASK_COUNT = 4
EPISODES_PER_TASK = 3
DOWNLOAD_CHUNK = 1024 * 1024


class PreparationError(RuntimeError):
    """A deliberate fail-closed preparation error."""


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _short(value: Any, limit: int = 900) -> str:
    text = str(value).replace("\x00", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _safe_rel(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value or "\\" in value:
        raise PreparationError(f"Unsafe Hub relative path: {value!r}")
    return str(path)


def _api(repo_kind: str, repo_id: str, revision: str) -> dict[str, Any]:
    if repo_kind not in {"models", "datasets"}:
        raise ValueError(repo_kind)
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    query = urllib.parse.urlencode({"revision": revision, "blobs": "true"})
    url = f"https://huggingface.co/api/{repo_kind}/{encoded_repo}/revision/{urllib.parse.quote(revision, safe='')}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "v100-new-angles-preparation/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001 - status records a bounded failure
        raise PreparationError(f"Hub metadata request failed for {repo_id}: {_short(exc)}") from exc
    if not isinstance(value, dict) or not value.get("sha"):
        raise PreparationError(f"Hub metadata has no commit identity for {repo_id}")
    return value


def _sibling_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in payload.get("siblings", []):
        if not isinstance(item, dict) or not item.get("rfilename"):
            continue
        name = _safe_rel(str(item["rfilename"]))
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        raw_size = item.get("size", lfs.get("size"))
        try:
            size = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError):
            size = None
        records[name] = {
            "rfilename": name,
            "size": size,
            "lfs_sha256": lfs.get("sha256"),
        }
    return records


def _weight_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".safetensors", ".bin", ".pt", ".pth")) and (
        "model" in lower or "pytorch" in lower or "adapter" in lower
    )


def _download_url(kind: str, repo_id: str, revision: str, filename: str) -> str:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_file = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    hub_prefix = "datasets/" if kind == "datasets" else ""
    return f"https://huggingface.co/{hub_prefix}{encoded_repo}/resolve/{urllib.parse.quote(revision, safe='')}/{encoded_file}"


def _download(
    *,
    kind: str,
    repo_id: str,
    revision: str,
    record: dict[str, Any],
    destination: Path,
    cap: int,
) -> dict[str, Any]:
    """Stream one file, with one-writer and exact-size/hash checks."""
    filename = str(record["rfilename"])
    expected_size = record.get("size")
    if expected_size is None or expected_size < 0:
        raise PreparationError(f"No exact Hub size for {repo_id}:{filename}")
    if expected_size > cap:
        raise PreparationError(f"File exceeds bounded download cap: {filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise PreparationError(f"Existing file size mismatch: {destination}")
        digest = _sha256(destination)
        expected_hash = record.get("lfs_sha256")
        if expected_hash and digest != expected_hash:
            raise PreparationError(f"Existing file hash mismatch: {destination}")
        return {**record, "downloaded_sha256": digest, "reused": True}
    part = destination.with_name(destination.name + ".part")
    if part.exists():
        raise PreparationError(f"Unfinished single-writer file exists; inspect before retry: {part}")
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(
        _download_url(kind, repo_id, revision, filename),
        headers={"User-Agent": "v100-new-angles-preparation/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open("xb") as output:
            while True:
                block = response.read(DOWNLOAD_CHUNK)
                if not block:
                    break
                received += len(block)
                if received > expected_size or received > cap:
                    raise PreparationError(f"Downloaded file exceeded declared bound: {filename}")
                digest.update(block)
                output.write(block)
        if received != expected_size:
            raise PreparationError(f"Downloaded size mismatch for {filename}: {received} != {expected_size}")
        downloaded_hash = digest.hexdigest()
        expected_hash = record.get("lfs_sha256")
        if expected_hash and downloaded_hash != expected_hash:
            raise PreparationError(f"Downloaded LFS SHA-256 mismatch for {filename}")
        os.replace(part, destination)
    except Exception:
        # Preserve the .part marker.  Another writer must not silently take over.
        raise
    return {**record, "downloaded_sha256": downloaded_hash, "reused": False}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(DOWNLOAD_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_many(
    *,
    kind: str,
    repo_id: str,
    revision: str,
    records: Iterable[dict[str, Any]],
    root: Path,
    cap: int,
) -> list[dict[str, Any]]:
    result = []
    for record in sorted(records, key=lambda item: str(item["rfilename"])):
        result.append(
            _download(
                kind=kind,
                repo_id=repo_id,
                revision=revision,
                record=record,
                destination=root / _safe_rel(str(record["rfilename"])),
                cap=cap,
            )
        )
    return result


def _numeric(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or value is None:
        raise PreparationError(f"Missing numeric metadata field {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"Non-numeric metadata field {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise PreparationError(f"Non-finite metadata field {field}")
    return int(number) if number.is_integer() else number


def _as_task_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _task_rows(table: Any) -> tuple[dict[int, str], dict[str, int]]:
    rows = table.to_pylist()
    if not rows:
        raise PreparationError("tasks.parquet is empty")
    index_names = ("task_index", "index")
    text_names = ("task", "language_instruction", "instruction")
    pandas_meta = json.loads((table.schema.metadata or {}).get(b"pandas", b"{}"))
    if pandas_meta.get("index_columns") == ["__index_level_0__"]:
        text_names += ("__index_level_0__",)
    by_id: dict[int, str] = {}
    by_text: dict[str, int] = {}
    for ordinal, row in enumerate(rows):
        raw_id = next((row[name] for name in index_names if name in row and row[name] is not None), None)
        try:
            task_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise PreparationError(f"Unclear task index in tasks.parquet: {raw_id!r}") from exc
        raw_text = next((row[name] for name in text_names if name in row), None)
        text = _as_task_text(raw_text)
        if text is None:
            raise PreparationError("tasks.parquet has no unambiguous textual task field")
        if task_id in by_id and by_id[task_id] != text:
            raise PreparationError(f"Task index maps to multiple texts: {task_id}")
        if text in by_text and by_text[text] != task_id:
            raise PreparationError(f"Task text maps to multiple indices: {text!r}")
        by_id[task_id] = text
        by_text[text] = task_id
    return by_id, by_text


def _episode_task(value: Any, task_text_to_id: dict[str, int]) -> int:
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise PreparationError("Episode has empty tasks metadata")
    resolved: set[int] = set()
    for item in values:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            resolved.add(int(item))
        elif isinstance(item, str) and item in task_text_to_id:
            resolved.add(task_text_to_id[item])
        else:
            raise PreparationError(f"Unclear episode task reference: {item!r}")
    if len(resolved) != 1:
        raise PreparationError(f"Episode has multiple task references: {sorted(resolved)}")
    return next(iter(resolved))


def _config_features(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_features = config.get("input_features")
    output_features = config.get("output_features")
    if not isinstance(input_features, dict) or not isinstance(output_features, dict):
        raise PreparationError("Checkpoint config lacks unambiguous input/output_features")
    return input_features, output_features


def _feature_dtype(value: Any) -> str:
    return str(value.get("dtype", "")).lower() if isinstance(value, dict) else ""


def _feature_shape(value: Any) -> list[int] | None:
    if not isinstance(value, dict) or not isinstance(value.get("shape"), (list, tuple)):
        return None
    try:
        return [int(x) for x in value["shape"]]
    except (TypeError, ValueError):
        return None


def _checkpoint_mapping(config: dict[str, Any], dataset_info: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    inputs, outputs = _config_features(config)
    image_keys = sorted(
        key for key, value in inputs.items() if _feature_dtype(value) in {"image", "video"} or key.startswith("observation.images.")
    )
    state_keys = [key for key, value in inputs.items() if key == "observation.state" or _feature_dtype(value) in {"state", "proprioception"}]
    if len(state_keys) != 1:
        raise PreparationError(f"Checkpoint state mapping is not unique: {state_keys}")
    if "action" not in outputs:
        raise PreparationError("Checkpoint output_features has no exact action key")
    vlm_name = config.get("vlm_model_name") or config.get("vlm_model")
    if vlm_name != BASE_VLM_REPO:
        raise PreparationError(f"Unexpected or missing SmolVLM base identity: {vlm_name!r}")
    dataset_features = dataset_info.get("features")
    if not isinstance(dataset_features, dict):
        raise PreparationError("Dataset info has no features mapping")
    dataset_camera_keys = sorted(
        key for key, value in dataset_features.items() if _feature_dtype(value) in {"image", "video"}
    )
    processor = json.loads((model_dir / "policy_preprocessor.json").read_text())
    renamers = [s for s in processor["steps"] if s.get("registry_name") == "rename_observations_processor"]
    if len(renamers) != 1:
        raise PreparationError("Expected the one saved official camera rename processor")
    rename_map = renamers[0]["config"]["rename_map"]
    present = sorted(key for key in dataset_camera_keys if rename_map.get(key, key) in image_keys)
    mapped = [rename_map.get(key, key) for key in present]
    if len(set(mapped)) != len(mapped):
        raise PreparationError("Camera rename mapping is not one-to-one")
    missing = sorted(set(image_keys) - set(mapped))
    if not present:
        raise PreparationError("No checkpoint image key is present in the dataset")
    checkpoint_state = inputs[state_keys[0]]
    dataset_state = dataset_features.get(state_keys[0])
    if not isinstance(dataset_state, dict):
        raise PreparationError(f"Dataset has no exact state feature {state_keys[0]!r}")
    checkpoint_state_shape = _feature_shape(checkpoint_state)
    dataset_state_shape = _feature_shape(dataset_state)
    with (model_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors").open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        if header_length > 65536:
            raise PreparationError("Unexpected normalizer header size")
        header = json.loads(stream.read(header_length))
    stats_shapes = [header.get(state_keys[0] + suffix, {}).get("shape") for suffix in (".mean", ".std")]
    if dataset_state_shape != [8] or stats_shapes != [[8], [8]] or config.get("max_state_dim") != 32:
        raise PreparationError(
            f"State/statistics binding failed: nominal={checkpoint_state_shape}, dataset={dataset_state_shape}, stats={stats_shapes}"
        )
    empty_cameras = config.get("empty_cameras", [])
    return {
        "checkpoint_input_features": inputs,
        "checkpoint_output_features": outputs,
        "checkpoint_image_keys": image_keys,
        "dataset_camera_keys": dataset_camera_keys,
        "present_image_keys": present,
        "missing_checkpoint_image_keys": missing,
        "official_camera_rename_map": rename_map,
        "normalizer_state_shapes": stats_shapes,
        "state_shape_note": "Official config nominal6; saved mean/std and training dataset8. Preserve all8, official processor then pad32; no truncation or fabricated state.",
        "state_key": state_keys[0],
        "checkpoint_state_shape": checkpoint_state_shape,
        "dataset_state_shape": dataset_state_shape,
        "action_shape": _feature_shape(outputs["action"]),
        "chunk_size": config.get("chunk_size"),
        "max_action_dim": config.get("max_action_dim"),
        "max_state_dim": config.get("max_state_dim"),
        "num_inference_steps": config.get("num_steps"),
        "empty_cameras": empty_cameras,
        "missing_camera_handling": "record_only; official preprocessor may mask only configured empty_cameras; no fabricated image is saved",
    }


def _format_path(template: str, **values: Any) -> str:
    try:
        rendered = template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise PreparationError(f"Cannot resolve dataset path template {template!r}: {_short(exc)}") from exc
    return _safe_rel(rendered)


def _parquet_rows(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    schema = parquet.read_schema(path)
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise PreparationError(f"{path.name} lacks required columns: {missing}")
    return parquet.read_table(path, columns=columns).to_pylist()


def _decode_frame(video_path: Path, timestamp: float, tolerance_s: float) -> tuple[Any, str]:
    from lerobot.datasets.video_utils import decode_video_frames
    import numpy as np

    # v0.4.4 explicitly supports PyAV and checks the same timestamp tolerance.
    # Job 64757 established that TorchCodec cannot load libpython on this host.
    decoded = decode_video_frames(video_path, [timestamp], tolerance_s, backend="pyav")
    conversion = "lerobot_pyav_float01_to_uint8"
    if hasattr(decoded, "detach"):
        decoded = decoded.detach().cpu().numpy()
    if isinstance(decoded, dict):
        raise PreparationError(f"Video decoder returned mapping for {video_path}")
    array = np.asarray(decoded)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise PreparationError(f"Decoder did not return one CHW frame: shape={array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all() or array.min() < -1e-6 or array.max() > 1.000001:
            raise PreparationError(f"Decoder frame is not uint8 or [0,1] float: dtype={array.dtype}")
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if array.shape[0] not in {1, 3, 4}:
        raise PreparationError(f"Decoder output is not CHW: shape={array.shape}")
    return array, conversion


def _scalar_array(value: Any, dtype: Any = None) -> Any:
    import numpy as np

    return np.asarray(value if value is not None else None, dtype=dtype)


def _versions() -> dict[str, Any]:
    names = ["torch", "torchvision", "lerobot", "torchcodec", "av", "huggingface-hub", "pyarrow", "numpy"]
    result = {"python": sys.version.split()[0]}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _write_status(
    *,
    out: Path,
    assetdir: Path,
    allocation: dict[str, Any],
    phase: str,
    state: str = "running",
    **extra: Any,
) -> None:
    safe_allocation = {key: value for key, value in allocation.items() if key != "user"}
    value: dict[str, Any] = {
        "schema": "smolvla-cpu-preparation-status-v1",
        "state": state,
        "phase": phase,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": safe_allocation,
        "updated_epoch": int(__import__("time").time()),
    }
    for key, item in extra.items():
        if key in {"error", "detail"}:
            value[key] = _short(item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            value[key] = item
        elif isinstance(item, (list, tuple)):
            value[key] = [_short(x, 160) for x in item[:20]]
        elif isinstance(item, dict):
            value[key] = {str(k): _short(v, 180) for k, v in list(item.items())[:20]}
    _json_dump(out / "status.json", value)
    if assetdir != out.parent:
        _json_dump(assetdir / "status.json", value)


def _acquire_lock(assetdir: Path, allocation: dict[str, Any]) -> None:
    lock = assetdir / "PREPARATION.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PreparationError(f"Asset directory is already claimed by a preparation writer: {lock}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"job_id": allocation.get("job_id"), "hostname": allocation.get("hostname")}) + "\n")


def _retire_failed_lock(assetdir: Path, previous: str | None) -> None:
    if previous is None:
        return
    if not previous.isdigit():
        raise PreparationError("Resume requires one explicit numeric job ID")
    lock = assetdir / "PREPARATION.lock"
    owner = json.loads(lock.read_text())
    if str(owner.get("job_id")) != previous:
        raise PreparationError("Resume lock belongs to a different job")
    result = subprocess.run(["sacct", "-nP", "-j", previous, "--format=JobIDRaw,State"], check=True, text=True, capture_output=True)
    states = [row.split("|")[1] for row in result.stdout.splitlines() if row.split("|")[0] == previous]
    if states != ["FAILED"]:
        raise PreparationError(f"Previous writer not confirmed FAILED: {states}")
    retired = assetdir / f"PREPARATION.lock.{previous}.failed"
    if retired.exists():
        raise PreparationError("Previous lock already archived; inspect before another retry")
    lock.rename(retired)


def _join_episode_tasks(episode_rows, dataset_records, assetdir, used_bytes):
    # This dataset revision omits task IDs from episode metadata.  Read only
    # episode/task columns from the existing tabular source, before any video
    # selection or model outputs.  Each task association must be unanimous.
    plan = [record for name, record in dataset_records.items() if name.startswith("data/") and name.endswith(".parquet")]
    tabular_bytes = sum(int(record["size"] or 0) for record in plan)
    if tabular_bytes > SUBSET_CAP or tabular_bytes + used_bytes > TOTAL_CAP:
        raise PreparationError("Task-mapping tabular files exceed aggregate byte cap")
    downloads = _download_many(kind="datasets", repo_id=DATASET_REPO, revision=DATASET_REVISION,
                              records=plan, root=assetdir / "source_subset", cap=SUBSET_CAP)
    by_episode = {}
    for record in downloads:
        rows = _parquet_rows(assetdir / "source_subset" / record["rfilename"], ["episode_index", "task_index"])
        for row in rows:
            ep, task = int(_numeric(row["episode_index"], "episode_index")), int(_numeric(row["task_index"], "task_index"))
            by_episode.setdefault(ep, set()).add(task)
    for row in episode_rows:
        tasks = by_episode.get(int(row["episode_index"]), set())
        if len(tasks) != 1:
            raise PreparationError("Episode task association missing or not unanimous")
        row["tasks"] = list(tasks)
    return downloads


def _select_episodes(
    episode_rows: list[dict[str, Any]], task_ids: list[int], task_text_to_id: dict[str, int], task_text_by_id: dict[int, str]
) -> list[dict[str, Any]]:
    by_task: dict[int, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    seen_episode_ids: set[int] = set()
    for row in episode_rows:
        if "episode_index" not in row or "length" not in row or "tasks" not in row:
            raise PreparationError("Episode metadata lacks episode_index/length/tasks")
        episode_id = int(_numeric(row["episode_index"], "episode_index"))
        if episode_id in seen_episode_ids:
            raise PreparationError(f"Duplicate episode_index in metadata: {episode_id}")
        seen_episode_ids.add(episode_id)
        task_id = _episode_task(row["tasks"], task_text_to_id)
        if task_id not in by_task:
            continue
        length = int(_numeric(row["length"], "length"))
        if length <= 0:
            raise PreparationError(f"Non-positive episode length: {episode_id}")
        selected = dict(row)
        selected.update({"episode_index": episode_id, "task_index": task_id, "task_text": task_text_by_id[task_id], "length": length})
        by_task[task_id].append(selected)
    chosen: list[dict[str, Any]] = []
    for task_id in task_ids:
        candidates = sorted(by_task[task_id], key=lambda item: int(item["episode_index"]))
        if len(candidates) < EPISODES_PER_TASK:
            raise PreparationError(f"Task {task_id} has fewer than {EPISODES_PER_TASK} distinct episodes")
        chosen.extend(candidates[:EPISODES_PER_TASK])
    if len(chosen) != SAMPLE_COUNT or len({int(row["episode_index"]) for row in chosen}) != SAMPLE_COUNT:
        raise PreparationError("Frozen selection did not produce 12 unique episodes")
    for row in chosen:
        row["frame_index"] = int(row["length"]) // 4
        row["selection_rule"] = "task_index ascending first 4; episode_index ascending first 3 distinct; zero-based frame_index=floor(length/4)"
    return chosen


def _main(args: argparse.Namespace, allocation: dict[str, Any]) -> None:
    import numpy as np
    import pyarrow.parquet as parquet

    assetdir = Path(args.assetdir).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    assetdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    _retire_failed_lock(assetdir, args.resume_from_job)
    _acquire_lock(assetdir, allocation)
    _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="guard_verified", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}:
        raise PreparationError("CPU-only preparation received a visible CUDA device")

    _json_dump(out / "versions.json", _versions())
    _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="dependencies", versions_path="versions.json")

    model_payload = _api("models", MODEL_REPO, MODEL_REVISION)
    if model_payload.get("sha") != MODEL_REVISION:
        raise PreparationError(f"Model revision mismatch: {model_payload.get('sha')} != {MODEL_REVISION}")
    model_records = _sibling_records(model_payload)
    # A SmolVLA checkpoint has processor normalizer/unnormalizer tensors next
    # to model tensors.  Keep every safetensors blob from this checkpoint so a
    # later GPU runner can reproduce the official processor exactly.
    model_weights = [record for name, record in model_records.items() if name.lower().endswith(".safetensors")]
    if not model_weights:
        model_weights = [record for name, record in model_records.items() if _weight_name(name)]
    if not model_weights:
        raise PreparationError("SmolVLA checkpoint has no recognized model weights")
    if any(not record.get("lfs_sha256") for record in model_weights):
        raise PreparationError("Every selected SmolVLA weight must expose an LFS SHA-256")
    model_small = [
        record for name, record in model_records.items()
        if not name.lower().endswith(".safetensors") and not _weight_name(name) and (name.endswith(".json") or name.endswith((".txt", ".model", ".jinja")))
    ]
    model_plan = model_weights + model_small
    model_bytes = sum(int(record["size"] or 0) for record in model_plan)
    if model_bytes > MODEL_CAP:
        raise PreparationError(f"Checkpoint plan exceeds {MODEL_CAP} bytes")
    model_downloads = _download_many(
        kind="models", repo_id=MODEL_REPO, revision=MODEL_REVISION, records=model_plan, root=assetdir / "model", cap=MODEL_CAP
    )
    config_path = assetdir / "model" / "config.json"
    if not config_path.is_file():
        raise PreparationError("SmolVLA checkpoint config.json was not downloaded")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise PreparationError("SmolVLA config.json is not an object")
    _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="model_downloaded", model_files=len(model_downloads), model_bytes=model_bytes)

    # Resolve and retain only base VLM metadata.  No base-model weight is ever selected.
    base_payload = _api("models", BASE_VLM_REPO, "main")
    base_records = _sibling_records(base_payload)
    base_meta = [
        record for name, record in base_records.items()
        if not name.lower().endswith((".safetensors", ".bin", ".pt", ".pth")) and (name.endswith(".json") or name.endswith((".txt", ".model", ".jinja")))
    ]
    base_bytes = sum(int(record["size"] or 0) for record in base_meta)
    if base_bytes > 50_000_000:
        raise PreparationError("Base VLM metadata exceeds bounded 50 MB cap")
    base_downloads = _download_many(
        kind="models", repo_id=BASE_VLM_REPO, revision=str(base_payload["sha"]), records=base_meta,
        root=assetdir / "base_vlm_metadata", cap=50_000_000,
    )
    if any(_weight_name(str(record["rfilename"])) for record in base_downloads):
        raise PreparationError("Base VLM weight was selected accidentally")

    dataset_payload = _api("datasets", DATASET_REPO, DATASET_REVISION)
    if dataset_payload.get("sha") != DATASET_REVISION:
        raise PreparationError(f"Dataset revision mismatch: {dataset_payload.get('sha')} != {DATASET_REVISION}")
    dataset_records = _sibling_records(dataset_payload)
    meta_records = [record for name, record in dataset_records.items() if name.startswith("meta/")]
    meta_bytes = sum(int(record["size"] or 0) for record in meta_records)
    if meta_bytes > META_CAP:
        raise PreparationError(f"Dataset metadata exceeds {META_CAP} bytes")
    meta_downloads = _download_many(
        kind="datasets", repo_id=DATASET_REPO, revision=DATASET_REVISION, records=meta_records,
        root=assetdir / "source_subset", cap=META_CAP,
    )
    info_path = assetdir / "source_subset" / "meta" / "info.json"
    tasks_path = assetdir / "source_subset" / "meta" / "tasks.parquet"
    if not info_path.is_file() or not tasks_path.is_file():
        raise PreparationError("Dataset metadata lacks meta/info.json or meta/tasks.parquet")
    dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(dataset_info.get("codebase_version", "")) if isinstance(dataset_info, dict) else ""
    if not isinstance(dataset_info, dict) or not (version.startswith("v3.") or version.startswith("3.")):
        raise PreparationError("Dataset is not a LeRobot v3 metadata set")
    task_text_by_id, task_text_to_id = _task_rows(parquet.read_table(tasks_path))
    task_ids = sorted(task_text_by_id)[:TASK_COUNT]
    if len(task_ids) != TASK_COUNT:
        raise PreparationError("Dataset has fewer than four task entries")
    episode_paths = sorted((assetdir / "source_subset" / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_paths:
        raise PreparationError("Dataset metadata has no episode parquet files")
    episode_rows: list[dict[str, Any]] = []
    for path in episode_paths:
        episode_rows.extend(parquet.read_table(path).to_pylist())
    task_mapping_downloads = []
    if any("tasks" not in row for row in episode_rows):
        task_mapping_downloads = _join_episode_tasks(episode_rows, dataset_records, assetdir, model_bytes + base_bytes + meta_bytes)
    selected = _select_episodes(episode_rows, task_ids, task_text_to_id, task_text_by_id)
    mapping = _checkpoint_mapping(config, dataset_info, assetdir / "model")
    _json_dump(out / "selection.json", {"task_ids": task_ids, "selected": selected, "mapping": mapping})
    _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="selection", selected_episodes=SAMPLE_COUNT, metadata_files=len(meta_downloads), base_vlm_metadata_files=len(base_downloads))

    data_template = dataset_info.get("data_path")
    video_template = dataset_info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise PreparationError("Dataset info lacks data_path/video_path templates")
    file_by_name = dataset_records
    data_plan: dict[str, dict[str, Any]] = {}
    video_plan: dict[str, dict[str, Any]] = {}
    for row in selected:
        try:
            data_rel = _format_path(data_template, chunk_index=row["data/chunk_index"], file_index=row["data/file_index"], episode_chunk=row["data/chunk_index"], file_chunk=row["data/file_index"])
        except KeyError as exc:
            raise PreparationError(f"Episode metadata lacks data chunk/file pointer: {row['episode_index']}") from exc
        if data_rel not in file_by_name:
            raise PreparationError(f"Selected data file is absent from Hub manifest: {data_rel}")
        data_plan[data_rel] = file_by_name[data_rel]
        row["data_path"] = data_rel
        row["data_source_record"] = file_by_name[data_rel]
        for camera_key in mapping["present_image_keys"]:
            prefix = f"videos/{camera_key}/"
            required = [f"{prefix}chunk_index", f"{prefix}file_index", f"{prefix}from_timestamp"]
            if any(name not in row for name in required):
                raise PreparationError(f"Episode lacks video pointer for camera {camera_key}: {row['episode_index']}")
            video_rel = _format_path(
                video_template, video_key=camera_key, chunk_index=row[f"{prefix}chunk_index"], file_index=row[f"{prefix}file_index"],
                episode_chunk=row[f"{prefix}chunk_index"], file_chunk=row[f"{prefix}file_index"],
            )
            if video_rel not in file_by_name:
                raise PreparationError(f"Selected video file is absent from Hub manifest: {video_rel}")
            video_plan[video_rel] = file_by_name[video_rel]
            row.setdefault("video_paths", {})[camera_key] = video_rel
            row.setdefault("video_source_records", {})[camera_key] = file_by_name[video_rel]
    retained_data = {record["rfilename"]: record for record in task_mapping_downloads}
    retained_data.update(data_plan)
    subset_bytes = sum(int(record["size"] or 0) for record in [*retained_data.values(), *video_plan.values()])
    if subset_bytes > SUBSET_CAP:
        raise PreparationError(f"Selected data/video plan exceeds {SUBSET_CAP} bytes")
    if model_bytes + base_bytes + meta_bytes + subset_bytes > TOTAL_CAP:
        raise PreparationError(f"Model plus selected data/video exceeds {TOTAL_CAP} bytes")
    data_downloads = _download_many(
        kind="datasets", repo_id=DATASET_REPO, revision=DATASET_REVISION, records=data_plan.values(),
        root=assetdir / "source_subset", cap=SUBSET_CAP,
    )
    video_downloads = _download_many(
        kind="datasets", repo_id=DATASET_REPO, revision=DATASET_REVISION, records=video_plan.values(),
        root=assetdir / "source_subset", cap=SUBSET_CAP,
    )
    _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="subset_downloaded", data_files=len(data_downloads), video_files=len(video_downloads), subset_bytes=subset_bytes)

    samples_dir = assetdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    tolerance_s = max(1e-4, 0.5 / float(_numeric(dataset_info.get("fps"), "fps")))
    sample_records: list[dict[str, Any]] = []
    decode_conversions: set[str] = set()
    for ordinal, row in enumerate(sorted(selected, key=lambda item: (int(item["task_index"]), int(item["episode_index"]))), start=1):
        data_path = assetdir / "source_subset" / row["data_path"]
        columns = ["episode_index", "frame_index", "timestamp", "task_index", "index", mapping["state_key"]]
        data_rows = _parquet_rows(data_path, columns)
        matching = [
            item for item in data_rows
            if int(_numeric(item["episode_index"], "episode_index")) == int(row["episode_index"])
            and int(_numeric(item["frame_index"], "frame_index")) == int(row["frame_index"])
        ]
        if len(matching) != 1:
            raise PreparationError(f"Expected exactly one selected data row, got {len(matching)} for episode {row['episode_index']}")
        data_row = matching[0]
        if int(_numeric(data_row["task_index"], "task_index")) != int(row["task_index"]):
            raise PreparationError(f"Data task_index mismatch for episode {row['episode_index']}")
        state = np.asarray(data_row[mapping["state_key"]], dtype=np.float32)
        if state.ndim != 1 or state.shape != tuple(mapping["dataset_state_shape"]) or not np.isfinite(state).all():
            raise PreparationError(f"Selected raw state has unexpected shape/finite status: {state.shape}")
        timestamp = float(_numeric(data_row["timestamp"], "timestamp"))
        arrays: dict[str, Any] = {mapping["state_key"]: state}
        video_timestamps: dict[str, float] = {}
        for camera_key in mapping["present_image_keys"]:
            prefix = f"videos/{camera_key}/"
            video_timestamp = float(_numeric(row[f"{prefix}from_timestamp"], f"{prefix}from_timestamp")) + timestamp
            frame, conversion = _decode_frame(assetdir / "source_subset" / row["video_paths"][camera_key], video_timestamp, tolerance_s)
            feature_shape = _feature_shape(dataset_info["features"][camera_key])
            names = dataset_info["features"][camera_key].get("names")
            if names == ["height", "width", "channel"] and feature_shape:
                feature_shape = [feature_shape[2], feature_shape[0], feature_shape[1]]
            if feature_shape and feature_shape != list(frame.shape):
                raise PreparationError(f"Decoded frame shape differs from dataset feature for {camera_key}: {frame.shape} vs {feature_shape}")
            arrays[camera_key] = frame
            video_timestamps[camera_key] = video_timestamp
            decode_conversions.add(conversion)
        metadata = {
            "schema": "smolvla-raw-input-sample-v1",
            "task_index": int(row["task_index"]),
            "task_text": row["task_text"],
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "timestamp": timestamp,
            "state_key": mapping["state_key"],
            "camera_keys": mapping["present_image_keys"],
            "missing_checkpoint_image_keys": mapping["missing_checkpoint_image_keys"],
            "video_timestamps": video_timestamps,
            "data_path": row["data_path"],
            "video_paths": row["video_paths"],
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "checkpoint_repo": MODEL_REPO,
            "checkpoint_revision": MODEL_REVISION,
            "processor_note": "GPU runner must apply the checkpoint processor for normalization/tokenization; CPU sample is raw.",
        }
        arrays.update({
            "task_index": _scalar_array(int(row["task_index"]), np.int64),
            "episode_index": _scalar_array(int(row["episode_index"]), np.int64),
            "frame_index": _scalar_array(int(row["frame_index"]), np.int64),
            "timestamp": _scalar_array(timestamp, np.float64),
            "metadata_json": _scalar_array(json.dumps(metadata, ensure_ascii=False), None),
        })
        for camera_key, video_timestamp in video_timestamps.items():
            arrays[f"video_timestamp__{re.sub(r'[^A-Za-z0-9_.-]', '_', camera_key)}"] = _scalar_array(video_timestamp, np.float64)
        sample_path = samples_dir / f"sample_{ordinal:02d}_task{int(row['task_index']):03d}_episode{int(row['episode_index']):04d}.npz"
        if sample_path.exists():
            raise PreparationError(f"Refusing to overwrite sample: {sample_path}")
        np.savez_compressed(sample_path, **arrays)
        sample_records.append({
            "sample_path": str(sample_path.relative_to(assetdir)).replace(os.sep, "/"),
            "sha256": _sha256(sample_path),
            "size": sample_path.stat().st_size,
            **{key: row[key] for key in ("task_index", "task_text", "episode_index", "length", "frame_index", "data_path", "video_paths")},
            "timestamp": timestamp,
            "video_timestamps": video_timestamps,
            "camera_keys": mapping["present_image_keys"],
            "state_key": mapping["state_key"],
        })
        _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="samples_written", samples_written=len(sample_records), total_samples=SAMPLE_COUNT)
    if len(sample_records) != SAMPLE_COUNT:
        raise PreparationError(f"Expected {SAMPLE_COUNT} samples, wrote {len(sample_records)}")

    identity = {
        "schema": "smolvla-cpu-preparation-identity-v1",
        "checkpoint": {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": model_downloads},
        "base_vlm": {"repo": BASE_VLM_REPO, "revision": base_payload.get("sha"), "weights_downloaded": False, "metadata_files": base_downloads},
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "metadata_files": meta_downloads, "data_files": data_downloads, "video_files": video_downloads, "task_mapping_source_files": task_mapping_downloads},
        "mapping": mapping,
        "selection": {"task_ids": task_ids, "rule": "task_index ascending first 4; episode_index ascending first 3 distinct; zero-based floor(length/4)", "count": SAMPLE_COUNT},
        "decode": {"tolerance_s": tolerance_s, "conversions": sorted(decode_conversions), "original_frame_contract": "uint8 CHW"},
        "no_inference": True,
    }
    _json_dump(out / "identity.json", identity)
    _json_dump(assetdir / "identity.json", identity)
    _json_dump(out / "manifest.json", {"schema": "smolvla-raw-input-manifest-v1", "samples": sample_records, "identity_path": "identity.json"})
    _json_dump(assetdir / "manifest.json", {"schema": "smolvla-raw-input-manifest-v1", "samples": sample_records, "identity_path": "identity.json"})
    _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="complete", state="complete", samples_written=SAMPLE_COUNT, manifest="manifest.json", inference_ran=False)
    print(json.dumps({"state": "complete", "samples": SAMPLE_COUNT, "output": str(out)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assetdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-from-job", default=None)
    args = parser.parse_args()
    # This call must remain before asset directory creation, imports of heavy
    # third-party packages, Hub access, hashing, downloads, or metadata reads.
    from allocation_guard import require_allocation

    allocation = require_allocation()
    try:
        _main(args, allocation)
    except Exception as exc:  # noqa: BLE001 - bounded failure record then non-zero exit
        out = Path(args.output).expanduser()
        assetdir = Path(args.assetdir).expanduser()
        if out.exists() or assetdir.exists():
            _write_status(out=out, assetdir=assetdir, allocation=allocation, phase="failed", state="failed", error=f"{type(exc).__name__}: {_short(exc)}")
        raise


if __name__ == "__main__":
    main()
