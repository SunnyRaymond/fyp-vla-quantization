"""Convert the AgileX Piper HDF5 episodes to the LeRobot v2.1 layout FastWAM reads.

  convert_agilex.py --raw <ROOT>/raw --out <ROOT>/data/agilex_pnp_lerobot --report <results>/convert.json

Input (per run): data/chunk-000/episode_XXXXXX.hdf5 with observation/images/{realsense_fixed,realsense_wrist}
(T,240,320,3 uint8), observation/state (T,7), action (T,7), attrs instruction/fps. raw/manifest.json lists every
uploaded file with its sha256; files missing from it (the two 0-byte episode_000039 files) are not converted.

Output: meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl}, data/chunk-000/episode_*.parquet,
videos/chunk-000/observation.images.{image,wrist_image}/episode_*.mp4 (10 fps, pts = frame index, GOP 2).
Episodes are renumbered 0..N-1 across runs in manifest order; timestamps are frame_index / fps.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import time

import av
import datasets
import h5py
import numpy as np

from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import compute_episode_stats
from fastwam.datasets.lerobot.lerobot.datasets.utils import get_hf_features_from_features, serialize_dict

FPS = 10
CAMERAS = {"image": "realsense_fixed", "wrist_image": "realsense_wrist"}
JOINTS = [f"joint_{i}" for i in range(1, 7)]
CODECS = ("libx264", "libopenh264", "libsvtav1", "mpeg4")  # H.264 first: FFmpeg decodes it natively everywhere


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def pick_codec() -> str:
    for name in CODECS:
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RuntimeError(f"No video encoder among {CODECS} in PyAV {av.__version__}")


def encode(frames: np.ndarray, path: Path, codec: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    with av.open(str(tmp), "w") as box:
        stream = box.add_stream(codec, rate=FPS)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        stream.codec_context.gop_size = 2
        if codec == "libsvtav1":
            stream.options = {"crf": "22", "preset": "8"}
        elif codec == "libx264":
            stream.options = {"crf": "17", "preset": "medium", "bf": "0"}
        else:
            stream.bit_rate = 4_000_000
        for i, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)
    tmp.replace(path)


def features(codec: str) -> dict:
    video_info = {"video.fps": FPS, "video.height": 240, "video.width": 320, "video.channels": 3,
                  "video.codec": codec, "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}
    ft = {f"observation.images.{k}": {"dtype": "video", "shape": (240, 320, 3),
                                      "names": ["height", "width", "channels"], "info": video_info} for k in CAMERAS}
    for prefix in ("observation.state", "action"):
        ft[f"{prefix}.joints"] = {"dtype": "float32", "shape": (6,), "names": JOINTS}
        ft[f"{prefix}.gripper"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    ft["timestamp"] = {"dtype": "float32", "shape": (1,), "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        ft[k] = {"dtype": "int64", "shape": (1,), "names": None}
    return ft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    t0 = time.time()
    manifest = json.loads((args.raw / "manifest.json").read_text(encoding="utf-8"))
    bad = [m["path"] for m in manifest["files"] if sha256(args.raw / m["path"]) != m["sha256"]]
    if bad:
        raise RuntimeError(f"sha256 mismatch after upload: {bad}")
    codec = pick_codec()
    ft = features(codec)
    hf_features = get_hf_features_from_features(ft)
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "meta").mkdir(parents=True)

    tasks: dict[str, int] = {}
    episodes, ep_stats, per_run, checks = [], [], {}, []
    index = 0
    for ep_idx, m in enumerate(manifest["files"]):
        src = args.raw / m["path"]
        with h5py.File(src, "r") as h:
            state = h["observation/state"][:].astype(np.float32)
            action = h["action"][:].astype(np.float32)
            imgs = {k: h[f"observation/images/{v}"][:] for k, v in CAMERAS.items()}
            instruction = str(h.attrs["instruction"])
            fps = float(h.attrs["fps"])
        T = len(state)
        if fps != FPS or state.shape != (T, 7) or action.shape != (T, 7):
            raise RuntimeError(f"{src}: fps {fps}, state {state.shape}, action {action.shape}")
        for k, v in imgs.items():
            if v.shape != (T, 240, 320, 3) or v.dtype != np.uint8:
                raise RuntimeError(f"{src}: {k} {v.shape} {v.dtype}")
        task_index = tasks.setdefault(instruction, len(tasks))
        frame_index = np.arange(T, dtype=np.int64)
        cols = {
            "observation.state.joints": state[:, :6], "observation.state.gripper": state[:, 6],
            "action.joints": action[:, :6], "action.gripper": action[:, 6],
            "timestamp": (frame_index / FPS).astype(np.float32), "frame_index": frame_index,
            "episode_index": np.full(T, ep_idx, np.int64), "index": np.arange(index, index + T, dtype=np.int64),
            "task_index": np.full(T, task_index, np.int64),
        }
        data = {k: (v.tolist() if v.ndim == 2 else v) for k, v in cols.items()}
        path = args.out / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_dict(data, features=hf_features).to_parquet(str(path))
        for key in CAMERAS:
            encode(imgs[key], args.out / f"videos/chunk-000/observation.images.{key}/episode_{ep_idx:06d}.mp4", codec)
        stats = compute_episode_stats({k: v for k, v in cols.items()}, ft, is_compute_episode_stats_image=False)
        ep_stats.append({"episode_index": ep_idx, "stats": serialize_dict(stats)})
        episodes.append({"episode_index": ep_idx, "tasks": [instruction], "length": T})
        run = m["path"].split("/")[0]
        per_run.setdefault(run, dict(episodes=0, frames=0, instruction=instruction))
        per_run[run]["episodes"] += 1
        per_run[run]["frames"] += T
        checks.append(float(np.abs(action[:-1] - state[1:]).max()))
        index += T
        print(f"episode {ep_idx:02d} <- {m['path']} T={T} task={task_index}", flush=True)

    info = {
        "codebase_version": "v2.1", "robot_type": "piper", "total_episodes": len(episodes), "total_frames": index,
        "total_tasks": len(tasks), "total_videos": len(episodes) * len(CAMERAS), "total_chunks": 1,
        "chunks_size": 1000, "fps": FPS, "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {k: {**v, "shape": list(v["shape"])} for k, v in ft.items()},
    }
    meta = args.out / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("episodes.jsonl", episodes), ("episodes_stats.jsonl", ep_stats),
                       ("tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])):
        (meta / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    report = dict(episodes=len(episodes), frames=index, tasks=tasks, per_run=per_run, codec=codec,
                  pyav=av.__version__, skipped=manifest.get("skipped", []),
                  action_equals_next_state_maxdiff=max(checks), seconds=round(time.time() - t0, 1))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "CONVERT_COMPLETE").write_text(json.dumps(report) + "\n", encoding="utf-8")
    print("CONVERT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
