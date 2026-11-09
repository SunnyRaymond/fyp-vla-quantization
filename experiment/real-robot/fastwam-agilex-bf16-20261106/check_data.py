"""CPU checks of the converted AgileX dataset, run by prep.pbs inside the FastWAM container.

  check_data.py video   --raw R --data D --out O    decode round trip of 4 episodes x 2 cameras with torchcodec and
                                                     pyav against the source HDF5 frames; writes O/video.json and
                                                     O/backend.txt (torchcodec if it decodes cleanly, else pyav)
  check_data.py dataset --config C --stats S --out O  builds RobotVideoDataset from the composed training config C:
                                                     computes the normalisation stats into S, then checks sample
                                                     shapes, normalised ranges, the relative-joint semantics, the
                                                     state padding and DataLoader throughput; writes O/check_data.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch

CAMERAS = {"image": "realsense_fixed", "wrist_image": "realsense_wrist"}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def video(args) -> None:
    from fastwam.datasets.lerobot.lerobot.datasets.video_utils import (
        decode_video_frames_torchcodec, decode_video_frames_torchvision)

    manifest = json.loads((args.raw / "manifest.json").read_text(encoding="utf-8"))["files"]
    n = len(manifest)
    rows, ok = [], {"torchcodec": True, "pyav": True}
    for ep in sorted({0, n // 2 - 1, n // 2, n - 1}):
        with h5py.File(args.raw / manifest[ep]["path"], "r") as h:
            src = {k: h[f"observation/images/{v}"][:] for k, v in CAMERAS.items()}
        T = len(src["image"])
        ts = [k / 10 for k in range(T)]
        for key in CAMERAS:
            path = args.data / f"videos/chunk-000/observation.images.{key}/episode_{ep:06d}.mp4"
            for backend in ("torchcodec", "pyav"):
                t0 = time.time()
                try:
                    if backend == "torchcodec":
                        frames = decode_video_frames_torchcodec(path, ts, 1e-4)
                    else:
                        frames = decode_video_frames_torchvision(path, ts, 1e-4, backend="pyav")
                    got = (frames.squeeze(0).permute(0, 2, 3, 1).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    if got.shape != src[key].shape:
                        raise RuntimeError(f"shape {got.shape} != {src[key].shape}")
                    value = psnr(got, src[key])
                    row = dict(episode=ep, camera=key, backend=backend, frames=T, psnr=round(value, 2),
                               seconds=round(time.time() - t0, 3))
                    ok[backend] &= value > 30
                except Exception as err:  # record and keep testing the other backend
                    row = dict(episode=ep, camera=key, backend=backend, error=f"{type(err).__name__}: {err}"[:400])
                    ok[backend] = False
                rows.append(row)
                print("VIDEO " + json.dumps(row), flush=True)
    backend = "torchcodec" if ok["torchcodec"] else "pyav" if ok["pyav"] else None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "video.json").write_text(json.dumps(dict(rows=rows, ok=ok, backend=backend), indent=2) + "\n")
    if backend is None:
        raise SystemExit("Neither torchcodec nor pyav reproduces the source frames")
    (args.out / "backend.txt").write_text(backend + "\n")


def dataset(args) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils import misc
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    report = {}

    work = args.out / "stats_work"
    misc.register_work_dir(work)
    t0 = time.time()
    instantiate(cfg.data.train, pretrained_norm_stats=None)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "dataset_stats.json", args.stats)
    report["stats_seconds"] = round(time.time() - t0, 1)
    stats = json.loads(args.stats.read_text())
    report["stats"] = {
        part: {k: {s: stats[part][k][s] for s in ("global_min", "global_max")} for k in ("joints", "gripper")}
        for part in ("state", "action")
    }

    ds = instantiate(cfg.data.train)
    report["len"] = len(ds)
    rng = np.random.default_rng(0)
    idxs = sorted({0, 1, 50, len(ds) - 1, *rng.integers(0, len(ds), 120).tolist()})
    amax = torch.zeros(7)
    pmax = torch.zeros(8)
    rel_err, rel_checked, padded = 0.0, 0, 0
    t0 = time.time()
    for i in idxs:
        s = ds[i]
        assert tuple(s["video"].shape) == (3, 9, 224, 448), s["video"].shape
        assert float(s["video"].min()) >= -1.0001 and float(s["video"].max()) <= 1.0001
        assert tuple(s["action"].shape) == (32, 7), s["action"].shape
        assert tuple(s["proprio"].shape) == (32, 8), s["proprio"].shape
        assert float(s["proprio"][:, 7].abs().max()) == 0.0, "state pad column is not zero"
        assert tuple(s["context"].shape) == (128, 4096), s["context"].shape
        valid = ~s["action_is_pad"]
        padded += int(s["action_is_pad"].any())
        amax = torch.maximum(amax, s["action"][valid].abs().amax(0) if valid.any() else amax)
        pmax = torch.maximum(pmax, s["proprio"].abs().amax(0))
        raw = ds.lerobot_dataset.multi_dataset[i]  # unprocessed window: action[t] must equal state[t+1]
        a, st = raw["action.joints"], raw["observation.state.joints"]
        if not bool(raw["action.joints_is_pad"][0]) and not bool(raw["observation.state.joints_is_pad"][1]):
            rel_err = max(rel_err, float((a[0] - st[1]).abs().max()))
            rel_checked += 1
    report["samples"] = dict(
        checked=len(idxs), with_padding=padded, seconds_per_sample=round((time.time() - t0) / len(idxs), 3),
        normalized_action_absmax=[round(float(x), 3) for x in amax],
        normalized_proprio_absmax=[round(float(x), 3) for x in pmax],
        action0_vs_next_state_maxdiff=rel_err, action0_checked=rel_checked,
        prompt=s["prompt"])
    if float(amax.max()) > 1.05 or float(pmax[:7].max()) > 1.05:
        raise SystemExit(f"normalised values leave [-1, 1]: action {amax.tolist()} proprio {pmax.tolist()}")
    if rel_err > 1e-5:
        raise SystemExit(f"action[t] != state[t+1] in the loaded windows (max diff {rel_err})")

    workers = max(1, int(os.environ.get("NCPUS", "8")) - 1)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=workers,
                                         persistent_workers=False, drop_last=True)
    it = iter(loader)
    next(it)
    t0 = time.time()
    nb = 12
    for _ in range(nb):
        batch = next(it)
    dt = time.time() - t0
    report["loader"] = dict(workers=workers, batch=8, batches=nb, samples_per_second=round(nb * 8 / dt, 2),
                            per_worker=round(nb * 8 / dt / workers, 2), keys=sorted(batch))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check_data.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CHECK_DATA " + json.dumps(report), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("video", "dataset"))
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    video(args) if args.mode == "video" else dataset(args)


if __name__ == "__main__":
    main()
