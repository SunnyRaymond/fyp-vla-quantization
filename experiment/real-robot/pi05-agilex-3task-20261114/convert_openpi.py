"""Convert the senior's three AgileX Piper tasks (raw 30 Hz HDF5) to the 10 Hz LeRobot v2.1 dataset openpi trains on.

  convert_openpi.py --raw-root <fastwam raw> --raw-root <run raw> --tasks cluster_grasp cabinet_lemon circle \
      --repo-id local/agilex3_10hz --report <results>/convert.json --marker <run>/data/CONVERT_COMPLETE

Each --raw-root holds manifest.json (path / size / sha256 of every episode file, written by the uploader) and
<task>/data/chunk-000/episode_XXXXXX.hdf5 with observation/images/{realsense_fixed,realsense_wrist} (T,240,320,3
uint8), observation/state (T,7), action (T,7), attrs instruction / fps. Every file of the selected tasks is checked
against its sha256 first.

Each raw episode gives three 10 Hz episodes, one per start offset 0/1/2 of the 30 Hz stream, so every raw frame is used
exactly once. Frame k of phase p is raw frame t = p + 3k; its action is action[min(t + 2, T - 1)] = state[t + 3], the
next 10 Hz state (the Fast-WAM conversion uses the same semantics). Images stay 240x320 and are written as LeRobot
`image` features, i.e. PNG bytes inside the parquet files as in openpi's LIBERO example, so training decodes no video;
openpi resizes with padding to 224x224 itself. The dataset lands in $HF_LEROBOT_HOME/<repo-id>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

RAW_FPS = 30
FPS = 10
STRIDE = RAW_FPS // FPS
CAMERAS = {"image": "realsense_fixed", "wrist_image": "realsense_wrist"}
FEATURES = {
    "image": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (7,), "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
                                                          "joint_6", "gripper"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
                                                            "joint_6", "gripper"]},
}


def sha256(path: str) -> tuple[str, str]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return path, h.hexdigest()


def collect(raw_roots: list[Path], tasks: list[str]) -> dict[str, list[dict]]:
    files = {t: [] for t in tasks}
    for root in raw_roots:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for m in manifest["files"]:
            task = m["path"].split("/")[0]
            if task in files:
                files[task].append(dict(m, abs=str(root / m["path"])))
    for task, entries in files.items():
        if not entries:
            raise RuntimeError(f"{task}: no files in any manifest under {raw_roots}")
        names = [e["path"] for e in entries]
        if len(set(names)) != len(names):
            raise RuntimeError(f"{task}: listed in more than one manifest")
        entries.sort(key=lambda e: e["path"])
    return files


def text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", action="append", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--marker", required=True, type=Path)
    args = ap.parse_args()
    t0 = time.time()
    ncpus = max(1, int(os.environ.get("NCPUS", "8")))

    files = collect(args.raw_root, args.tasks)
    every = [e for entries in files.values() for e in entries]
    with mp.Pool(ncpus) as pool:
        digests = dict(pool.imap_unordered(sha256, [e["abs"] for e in every], chunksize=2))
    bad = [e["path"] for e in every if digests[e["abs"]] != e["sha256"]]
    if bad:
        raise RuntimeError(f"sha256 mismatch: {bad}")
    print(f"sha256 ok: {len(every)} files, {sum(e['size'] for e in every) / 2 ** 30:.2f} GiB, "
          f"{time.time() - t0:.0f} s", flush=True)

    root = HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        shutil.rmtree(root)
    dataset = LeRobotDataset.create(repo_id=args.repo_id, robot_type="piper", fps=FPS, features=FEATURES,
                                    use_videos=False, image_writer_processes=max(1, ncpus - 4),
                                    image_writer_threads=2)
    per_task = {}
    for task in args.tasks:
        lengths, instructions, maxdiff = [], set(), 0.0
        for i, e in enumerate(files[task]):
            with h5py.File(e["abs"], "r") as h:
                state = h["observation/state"][:].astype(np.float32)
                action = h["action"][:].astype(np.float32)
                imgs = {k: h[f"observation/images/{v}"][:] for k, v in CAMERAS.items()}
                instruction = text(h.attrs["instruction"]).strip()
                fps = float(h.attrs["fps"])
            T = len(state)
            if fps != RAW_FPS or state.shape != (T, 7) or action.shape != (T, 7):
                raise RuntimeError(f"{e['path']}: fps {fps}, state {state.shape}, action {action.shape}")
            for k, v in imgs.items():
                if v.shape != (T, 240, 320, 3) or v.dtype != np.uint8:
                    raise RuntimeError(f"{e['path']}: {k} {v.shape} {v.dtype}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise RuntimeError(f"{e['path']}: non-finite state/action")
            instructions.add(instruction)
            for phase in range(STRIDE):
                rows = np.arange(phase, T, STRIDE)
                act = action[np.minimum(rows + STRIDE - 1, T - 1)]
                if len(rows) > 1:
                    maxdiff = max(maxdiff, float(np.abs(act[:-1] - state[rows[1:]]).max()))
                for k, t in enumerate(rows):
                    dataset.add_frame({"image": imgs["image"][t], "wrist_image": imgs["wrist_image"][t],
                                       "state": state[t], "actions": act[k], "task": instruction})
                dataset.save_episode()
                lengths.append(len(rows))
            if i % 10 == 0:
                print(f"{task} {i + 1}/{len(files[task])} T={T} episodes={dataset.meta.total_episodes} "
                      f"frames={dataset.meta.total_frames} {time.time() - t0:.0f} s", flush=True)
        if len(instructions) != 1:
            raise RuntimeError(f"{task}: expected one instruction, found {sorted(instructions)}")
        per_task[task] = dict(raw_episodes=len(files[task]), episodes=len(lengths), frames=int(sum(lengths)),
                              instruction=instructions.pop(), length_min=min(lengths), length_max=max(lengths),
                              action_vs_next_state_maxdiff=maxdiff)
    dataset.stop_image_writer()
    # The parquet files embed the PNG bytes, so the temporary per-frame PNG tree is no longer needed.
    shutil.rmtree(root / "images", ignore_errors=True)

    meta = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    report = dict(repo_id=args.repo_id, root=str(root), fps=FPS, raw_fps=RAW_FPS, phases=STRIDE, per_task=per_task,
                  total_episodes=meta["total_episodes"], total_frames=meta["total_frames"],
                  total_tasks=meta["total_tasks"], codebase_version=meta["codebase_version"],
                  parquet_gib=round(sum(p.stat().st_size for p in root.rglob("*.parquet")) / 2 ** 30, 2),
                  seconds=round(time.time() - t0, 1))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    print("CONVERT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
