"""Prepare the pinned Rolling Ball LeRobot subset; run only in a PBS allocation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import urllib.request

import numpy as np
import pyarrow.parquet as pq


DATASET_COMMIT = "9295b6e9878609a992047f0b8b65421a493299e7"
TASK_INDEX = 3
TASK_MARKER = "rolling down the ramp"
PREP_RUN = "25578687.pbs101"
EXPECTED_EPISODES = 200
EXPECTED_FRAMES_PER_EPISODE = 26
FPS = 25.0
MAX_ALIGNMENT_ERROR_S = 0.5 / FPS + 1e-6
CAMERA_KEY = "observation.images.fixed_cam"
DATA_OUTPUT_FILES = (
    "frames.npy", "actions.npy", "state.npy", "episode_ids.npy",
    "frame_indices.npy", "timestamps.npy", "split.json",
)


def allocation_guard() -> tuple[str, str]:
    """Reject local/login execution before touching dataset inputs or outputs."""
    job = os.environ.get("PBS_JOBID", "").strip()
    nodefile = Path(os.environ.get("PBS_NODEFILE", ""))
    host = socket.gethostname().split(".")[0].lower()
    if not job or not nodefile.is_file():
        raise RuntimeError("A genuine PBS allocation (PBS_JOBID and PBS_NODEFILE) is required")
    if any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-compute host: {host}")
    allocated = {name.split(".")[0].lower() for name in nodefile.read_text().split()}
    if host not in allocated:
        raise RuntimeError(f"Current host {host!r} is absent from PBS_NODEFILE")
    return job, host


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f"Expected a scalar cell, got {value!r}")
        return value[0]
    return value


def find_column(columns: list[str], candidates: tuple[str, ...], *, required=True) -> str | None:
    found = [name for name in candidates if name in columns]
    if len(found) > 1:
        raise ValueError(f"Ambiguous metadata columns {found}")
    if found:
        return found[0]
    if required:
        raise ValueError(f"Missing required metadata column; tried {candidates}; got {columns}")
    return None


def feature_column(columns: list[str], feature: str, field: str, *, required=True) -> str | None:
    return find_column(
        columns,
        (f"videos/{feature}/{field}", f"video/{feature}/{field}", f"{feature}/{field}"),
        required=required,
    )


def download_atomic(url: str, target: Path, max_bytes: int) -> int:
    """Single-writer streaming download; an interrupted .partial is never reused."""
    if target.exists():
        size = target.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Cached source file is empty: {target}")
        return size
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Incomplete download exists; inspect before retrying: {partial}")
    request = urllib.request.Request(url, headers={"User-Agent": "rolling-ball-lewm-dataset-prep"})
    count = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as stream:
        header = response.headers.get("Content-Length")
        expected = int(header) if header and header.isdigit() else None
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > max_bytes:
                raise RuntimeError(f"Download exceeds bounded size ({max_bytes} bytes): {url}")
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    if count <= 0 or (expected is not None and count != expected):
        raise RuntimeError(f"Incomplete download: got {count}, expected {expected}")
    os.replace(partial, target)
    return count


def package_versions() -> dict:
    result = {"python": os.sys.version.split()[0], "numpy": np.__version__}
    for name in ("pyarrow", "av"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def metadata_inputs(root: Path) -> tuple[dict, dict[int, dict], dict, dict]:
    report_dir = root / "runs" / PREP_RUN
    meta_dir = report_dir / "dataset_metadata" / "meta"
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    prep = json.loads((report_dir / "preparation.json").read_text(encoding="utf-8"))
    if prep.get("dataset_commit") != DATASET_COMMIT:
        raise RuntimeError(f"Prep report dataset pin mismatch: {prep.get('dataset_commit')}")
    task_rows = pq.read_table(meta_dir / "tasks.parquet").to_pylist()
    task = next((row for row in task_rows if int(scalar(row["task_index"])) == TASK_INDEX), None)
    if task is None or TASK_MARKER not in str(task.get("task", "")).lower():
        raise RuntimeError(f"task_index={TASK_INDEX} does not identify Rolling Ball: {task}")
    episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    episode_schema = pq.read_schema(episodes_path)
    episode_columns = episode_schema.names
    episode_rows = json.loads((report_dir / "rolling_episodes.json").read_text(encoding="utf-8"))
    if len(episode_rows) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} Rolling Ball episodes, got {len(episode_rows)}")
    episode_id_col = find_column(episode_columns, ("episode_index",))
    dataset_start_col = find_column(episode_columns, ("dataset_from_index",))
    length_col = find_column(episode_columns, ("length",))
    data_chunk_col = find_column(episode_columns, ("data/chunk_index",))
    data_file_col = find_column(episode_columns, ("data/file_index",))
    video_chunk_col = feature_column(episode_columns, CAMERA_KEY, "chunk_index")
    video_file_col = feature_column(episode_columns, CAMERA_KEY, "file_index")
    video_from_col = feature_column(episode_columns, CAMERA_KEY, "from_timestamp")
    video_to_col = feature_column(episode_columns, CAMERA_KEY, "to_timestamp")
    episode_map: dict[int, dict] = {}
    for row in episode_rows:
        eid = int(scalar(row[episode_id_col]))
        if eid in episode_map:
            raise RuntimeError(f"Duplicate episode metadata for {eid}")
        if int(scalar(row[length_col])) != EXPECTED_FRAMES_PER_EPISODE:
            raise RuntimeError(f"Episode {eid} does not have 26 frames")
        episode_map[eid] = {
            "dataset_from_index": int(scalar(row[dataset_start_col])),
            "length": int(scalar(row[length_col])),
            "data_chunk": int(scalar(row[data_chunk_col])),
            "data_file": int(scalar(row[data_file_col])),
            "video_chunk": int(scalar(row[video_chunk_col])),
            "video_file": int(scalar(row[video_file_col])),
            "video_from": float(scalar(row[video_from_col])),
            "video_to": float(scalar(row[video_to_col])),
        }
    if len(episode_map) != EXPECTED_EPISODES:
        raise RuntimeError("Rolling episode mapping count is inconsistent")
    if len({(r["data_chunk"], r["data_file"]) for r in episode_map.values()}) != 1:
        raise RuntimeError("Rolling episodes do not share one data parquet")
    if len({(r["video_chunk"], r["video_file"]) for r in episode_map.values()}) != 1:
        raise RuntimeError("Rolling episodes do not share one fixed-camera video")
    mapping = {
        "episode_columns": episode_columns,
        "episode_index": episode_id_col,
        "dataset_from_index": dataset_start_col,
        "length": length_col,
        "data_chunk_index": data_chunk_col,
        "data_file_index": data_file_col,
        "fixed_video_chunk_index": video_chunk_col,
        "fixed_video_file_index": video_file_col,
        "fixed_video_from_timestamp": video_from_col,
        "fixed_video_to_timestamp": video_to_col,
    }
    return info, episode_map, mapping, task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="ASPIRE2A task root")
    args = parser.parse_args()

    job, host = allocation_guard()
    root = args.root.resolve()
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    lock = data_root / ".prepare_dataset.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Another or interrupted dataset writer holds {lock}") from exc
    os.write(fd, f"job={job}\nhost={host}\n".encode())
    os.fsync(fd)
    os.close(fd)

    report = {
        "status": "RUNNING", "job_id": job, "hostname": host,
        "task_scope": "Rolling Ball Interception only", "trained": False,
        "closed_loop_evaluated": False, "success_labels": "unknown",
        "dataset_commit": DATASET_COMMIT,
    }
    stage = data_root / f".stage-{job}"
    try:
        existing_outputs = [name for name in DATA_OUTPUT_FILES if (data_root / name).exists()]
        if existing_outputs:
            raise RuntimeError(f"Dataset outputs already exist; refusing to overwrite: {existing_outputs}")
        if stage.exists():
            raise RuntimeError(f"Previous partial stage exists; inspect before retrying: {stage}")
        summary_path = data_root / "summary.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if previous.get("status") != "FAIL":
                raise RuntimeError(f"Existing summary is {previous.get('status')}; refusing to replace it")
            archive = data_root / f"summary.failed.{previous.get('job_id', 'unknown')}.json"
            if archive.exists():
                raise RuntimeError(f"Failed summary archive already exists: {archive}")
            os.replace(summary_path, archive)
        stage.mkdir()
        atomic_json(data_root / "summary.json", report)

        info, episode_map, metadata_mapping, task = metadata_inputs(root)
        shared_data = next(iter(episode_map.values()))
        data_rel = info["data_path"].format(
            chunk_index=shared_data["data_chunk"], file_index=shared_data["data_file"]
        )
        video_ref = next(iter(episode_map.values()))
        video_rel = info["video_path"].format(
            video_key=CAMERA_KEY, chunk_index=video_ref["video_chunk"],
            file_index=video_ref["video_file"],
        )
        source_root = data_root / "_source"
        parquet_path = source_root / data_rel
        video_path = source_root / video_rel
        base = f"https://huggingface.co/datasets/cyx337/ReflexBench_dataset/resolve/{DATASET_COMMIT}/"
        data_bytes = download_atomic(base + data_rel, parquet_path, max_bytes=8_000_000)
        video_bytes = download_atomic(base + video_rel, video_path, max_bytes=250_000_000)

        table = pq.read_table(parquet_path)
        cols = table.column_names
        ep_col = find_column(cols, ("episode_index",))
        task_col = find_column(cols, ("task_index",))
        index_col = find_column(cols, ("index",))
        frame_col = find_column(cols, ("frame_index",))
        timestamp_col = find_column(cols, ("timestamp",))
        action_col = find_column(cols, ("action",))
        state_col = find_column(cols, ("observation.state",), required=False)
        direct_video_col = find_column(
            cols,
            (f"videos/{CAMERA_KEY}/timestamp", f"video/{CAMERA_KEY}/timestamp",
             f"{CAMERA_KEY}/timestamp", "video_timestamp"),
            required=False,
        )
        metadata_mapping["data_columns"] = {
            "episode_index": ep_col, "task_index": task_col, "index": index_col,
            "frame_index": frame_col, "timestamp": timestamp_col,
            "action": action_col, "observation.state": state_col,
            "direct_fixed_video_timestamp": direct_video_col,
        }

        values = {name: table[name].to_pylist() for name in (ep_col, task_col, index_col, frame_col, timestamp_col, action_col)}
        if state_col:
            values[state_col] = table[state_col].to_pylist()
        if direct_video_col:
            values[direct_video_col] = table[direct_video_col].to_pylist()
        rows = []
        selected_ids = set(episode_map)
        for row_i in range(table.num_rows):
            eid = int(scalar(values[ep_col][row_i]))
            if int(scalar(values[task_col][row_i])) != TASK_INDEX or eid not in selected_ids:
                continue
            meta = episode_map[eid]
            local_index = int(scalar(values[index_col][row_i])) - meta["dataset_from_index"]
            frame_index = int(scalar(values[frame_col][row_i]))
            if frame_index != local_index:
                raise RuntimeError(f"frame_index/index metadata mismatch for episode {eid}: {frame_index} vs {local_index}")
            if not 0 <= local_index < EXPECTED_FRAMES_PER_EPISODE:
                raise RuntimeError(f"Invalid local frame index {local_index} for episode {eid}")
            timestamp = float(scalar(values[timestamp_col][row_i]))
            local_video_time = float(scalar(values[direct_video_col][row_i])) if direct_video_col else timestamp
            expected_video_time = meta["video_from"] + timestamp
            rows.append({
                "episode_id": eid, "frame_index": local_index,
                "timestamp": timestamp, "expected_video_time": expected_video_time,
                "direct_video_time": local_video_time if direct_video_col else None,
                "action": np.asarray(values[action_col][row_i], dtype=np.float32).reshape(8),
                "state": (np.asarray(values[state_col][row_i], dtype=np.float32).reshape(8)
                          if state_col else None),
            })
        rows.sort(key=lambda r: (r["episode_id"], r["frame_index"]))
        if len(rows) != EXPECTED_EPISODES * EXPECTED_FRAMES_PER_EPISODE:
            raise RuntimeError(f"Expected 5,200 task-index-3 rows, got {len(rows)}")
        for eid in sorted(selected_ids):
            episode_rows = [r for r in rows if r["episode_id"] == eid]
            if [r["frame_index"] for r in episode_rows] != list(range(EXPECTED_FRAMES_PER_EPISODE)):
                raise RuntimeError(f"Episode {eid} frame rows are incomplete or unordered")
            expected = np.asarray([r["expected_video_time"] for r in episode_rows])
            if np.any(np.diff(expected) <= 0):
                raise RuntimeError(f"Episode {eid} timestamps are not strictly increasing")
            if direct_video_col:
                direct = np.asarray([r["direct_video_time"] for r in episode_rows])
                err_absolute = float(np.max(np.abs(direct - expected)))
                err_offset = float(np.max(np.abs((episode_map[eid]["video_from"] + direct) - expected)))
                if err_absolute <= MAX_ALIGNMENT_ERROR_S:
                    for r, v in zip(episode_rows, direct):
                        r["video_time"] = float(v)
                    metadata_mapping["direct_video_timestamp_interpretation"] = "absolute"
                elif err_offset <= MAX_ALIGNMENT_ERROR_S:
                    for r, v in zip(episode_rows, direct):
                        r["video_time"] = episode_map[eid]["video_from"] + float(v)
                    metadata_mapping["direct_video_timestamp_interpretation"] = "episode_local_plus_video_offset"
                else:
                    raise RuntimeError(f"Video timestamp column disagrees with episode offset for {eid}")
            else:
                for r in episode_rows:
                    r["video_time"] = r["expected_video_time"]
        targets = sorted(((r["video_time"], i) for i, r in enumerate(rows)), key=lambda pair: pair[0])
        target_times = [time for time, _ in targets]
        if any(b <= a for a, b in zip(target_times, target_times[1:])):
            raise RuntimeError("Rolling Ball video timestamps are not unique and strictly increasing")
        image_feature = info["features"][CAMERA_KEY]
        image_info = image_feature.get("info", {})
        camera_fps = info.get("fps", image_info.get("video.fps"))
        if camera_fps is None or abs(float(camera_fps) - FPS) > 1e-3:
            raise RuntimeError(f"Expected 25Hz dataset/camera metadata, got {camera_fps!r}")
        image_shape = image_feature.get("shape", [])
        names = image_feature.get("names", [])
        if "height" in names and "width" in names and len(image_shape) == len(names):
            height, width = int(image_shape[names.index("height")]), int(image_shape[names.index("width")])
        else:
            height = int(image_info.get("video.height", 0))
            width = int(image_info.get("video.width", 0))
        if (height, width) != (224, 224):
            raise RuntimeError(f"Expected fixed camera frames at 224x224, got {height}x{width}")
        frame_path = stage / "frames.npy"
        frames = np.lib.format.open_memmap(
            frame_path, mode="w+", dtype=np.uint8,
            shape=(len(rows), height, width, 3),
        )

        import av

        target_i = 0
        prev = None
        decoded_count = 0
        matched_video_indices = []
        errors = []
        matched_pts = {}
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            def assign(frame, pts_time: float, video_index: int, target_time: float, output_index: int) -> None:
                delta = abs(pts_time - target_time)
                errors.append(delta)
                matched_video_indices.append(video_index)
                matched_pts[output_index] = pts_time
                frames[output_index] = frame.to_ndarray(format="rgb24")
            for frame in container.decode(stream):
                decoded_count += 1
                if decoded_count > 100_000:
                    raise RuntimeError("Video decode exceeded bounded 100,000-frame limit")
                if frame.pts is None:
                    raise RuntimeError("Fixed-camera video contains a frame without a PTS")
                pts_time = float(frame.pts * stream.time_base)
                current = (pts_time, decoded_count - 1, frame)
                if prev is not None:
                    midpoint = (prev[0] + current[0]) / 2.0
                    while target_i < len(targets) and targets[target_i][0] <= midpoint:
                        timestamp, output_i = targets[target_i]
                        assign(prev[2], prev[0], prev[1], timestamp, output_i)
                        target_i += 1
                prev = current
            if prev is None:
                raise RuntimeError("Fixed-camera video decoded no frames")
            while target_i < len(targets):
                timestamp, output_i = targets[target_i]
                assign(prev[2], prev[0], prev[1], timestamp, output_i)
                target_i += 1
        frames.flush()
        del frames
        if len(errors) != len(rows) or len(set(matched_video_indices)) != len(rows):
            raise RuntimeError("Video-frame alignment did not produce one unique frame per sample")
        max_error = float(max(errors))
        if max_error > MAX_ALIGNMENT_ERROR_S:
            raise RuntimeError(f"Nearest video timestamp error {max_error:.6f}s exceeds half a 25Hz frame")
        for row_i, row in enumerate(rows):
            meta = episode_map[row["episode_id"]]
            pts_time = matched_pts[row_i]
            if not meta["video_from"] - MAX_ALIGNMENT_ERROR_S <= pts_time <= meta["video_to"] + MAX_ALIGNMENT_ERROR_S:
                raise RuntimeError(f"Matched video frame crosses episode {row['episode_id']} timestamp bounds")

        actions = np.stack([r["action"] for r in rows]).astype(np.float32)
        states = (np.stack([r["state"] for r in rows]).astype(np.float32)
                  if state_col else None)
        episode_ids = np.asarray([r["episode_id"] for r in rows], dtype=np.int64)
        frame_indices = np.asarray([r["frame_index"] for r in rows], dtype=np.int64)
        timestamps = np.asarray([r["timestamp"] for r in rows], dtype=np.float64)
        for name, array in (("actions.npy", actions), ("episode_ids.npy", episode_ids),
                            ("frame_indices.npy", frame_indices), ("timestamps.npy", timestamps)):
            np.save(stage / name, array, allow_pickle=False)
        if states is not None:
            np.save(stage / "state.npy", states, allow_pickle=False)

        episode_ids_unique = sorted(selected_ids)
        shuffled = episode_ids_unique.copy()
        import random
        random.Random(20260927).shuffle(shuffled)
        validation_ids = sorted(shuffled[:20])
        validation_set = set(validation_ids)
        train_ids = sorted(eid for eid in episode_ids_unique if eid not in validation_set)
        if len(train_ids) != 180 or len(validation_ids) != 20 or set(train_ids) & set(validation_ids):
            raise RuntimeError("Episode split check failed")
        split = {
            "seed": 20260927, "method": "Python random.Random(seed).shuffle; first 20 validation",
            "train_episode_ids": train_ids, "validation_episode_ids": validation_ids,
        }
        atomic_json(stage / "split.json", split)

        def value_range(array):
            return {"min": np.min(array, axis=0).tolist(), "max": np.max(array, axis=0).tolist()}
        report.update({
            "status": "PASS", "dataset_commit": DATASET_COMMIT, "task_index": TASK_INDEX,
            "task_description": task.get("task"), "episode_count": len(episode_ids_unique),
            "frame_count": len(rows), "frames_per_episode_min": EXPECTED_FRAMES_PER_EPISODE,
            "frames_per_episode_max": EXPECTED_FRAMES_PER_EPISODE,
            "video_alignment": {
                "camera": CAMERA_KEY, "video_path": video_rel,
                "mapping": metadata_mapping.get(
                    "direct_video_timestamp_interpretation",
                    "episode video from_timestamp + episode-local timestamp",
                ),
                "metadata_fps": float(camera_fps),
                "direct_timestamp_column": direct_video_col,
                "max_error_s": max_error, "mean_error_s": float(np.mean(errors)),
                "tolerance_s": MAX_ALIGNMENT_ERROR_S, "decoded_video_frames": decoded_count,
                "unique_video_frames_matched": len(set(matched_video_indices)),
                "video_timestamp_min_s": float(min(target_times)),
                "video_timestamp_max_s": float(max(target_times)),
            },
            "metadata_column_mapping": metadata_mapping,
            "source_files": {"data_path": data_rel, "data_bytes": data_bytes,
                             "video_path": video_rel, "video_bytes": video_bytes},
            "arrays": {"frames": {"shape": [len(rows), 224, 224, 3], "dtype": "uint8"},
                       "actions": {"shape": list(actions.shape), "dtype": "float32", **value_range(actions)},
                       "state": ({"shape": list(states.shape), "dtype": "float32", **value_range(states)}
                                 if states is not None else "not present in pinned metadata"),
                       "episode_ids": {"shape": list(episode_ids.shape), "dtype": "int64"},
                       "frame_indices": {"shape": list(frame_indices.shape), "dtype": "int64"},
                       "timestamps": {"shape": list(timestamps.shape), "dtype": "float64",
                                      "min": float(timestamps.min()), "max": float(timestamps.max())}},
            "split": {"train_episodes": len(train_ids), "validation_episodes": len(validation_ids),
                      "seed": 20260927},
            "package_versions": package_versions(),
            "success_labels": "unknown; no per-episode success field was joined",
            "trained": False, "closed_loop_evaluated": False,
        })
        for name in DATA_OUTPUT_FILES:
            staged = stage / name
            if staged.exists():
                os.replace(staged, data_root / name)
        atomic_json(data_root / "summary.json", report)
        stage.rmdir()
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:
        report.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        atomic_json(data_root / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
