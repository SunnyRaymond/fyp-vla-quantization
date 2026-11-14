"""Helpers for deliver.pbs (run on a compute node inside the openpi venv).

  release_tools.py stats --raw-root R1 --raw-root R2 --tasks T1 T2 T3 --out robot_stats.json
      joint / gripper ranges of the recorded states and actions, and the 10 Hz per-step joint change (all three
      phases), for the safety limits in the robot README
  release_tools.py env --openpi <repo> --out-dir <bundle>/env
      versions of the cluster environment (installed distributions, jax, GPU, openpi commit)
  release_tools.py sums --bundle <bundle>
      SHA256SUMS of every bundle file except README.md, relative paths, sorted
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

STRIDE = 3


def stats(args):
    import h5py

    states, actions, steps, per_task = [], [], [], {}
    for task in args.tasks:
        files = [f for root in args.raw_root for f in sorted((root / task / "data/chunk-000").glob("episode_*.hdf5"))
                 if f.stat().st_size > 0]
        if not files:
            raise SystemExit(f"{task}: no recordings under {args.raw_root}")
        n_steps = 0
        for f in files:
            with h5py.File(f, "r") as h:
                s = h["observation/state"][:].astype(np.float64)
                a = h["action"][:].astype(np.float64)
            states.append(s)
            actions.append(a)
            for p in range(STRIDE):
                d = np.abs(np.diff(s[p::STRIDE, :6], axis=0))
                steps.append(d)
                n_steps += len(d)
        per_task[task] = dict(recordings=len(files), steps_10hz=n_steps)
    s, a, d = np.concatenate(states), np.concatenate(actions), np.concatenate(steps)
    out = dict(tasks=per_task,
               state_min=s.min(0).round(4).tolist(), state_max=s.max(0).round(4).tolist(),
               action_min=a.min(0).round(4).tolist(), action_max=a.max(0).round(4).tolist(),
               step_10hz=dict(count=len(d), median=np.median(d, 0).round(4).tolist(),
                              p99=np.percentile(d, 99, axis=0).round(4).tolist(), max=d.max(0).round(4).tolist()))
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


def env(args):
    import importlib.metadata as m

    import jax

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions())
    (out_dir / "pip_freeze_cluster.txt").write_text("\n".join(dists) + "\n", encoding="utf-8")
    commit = subprocess.run(["git", "-C", args.openpi, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    if not commit.strip():
        head = (Path(args.openpi) / ".git/HEAD").read_text().strip()
        commit = (Path(args.openpi) / ".git" / head[5:]).read_text() if head.startswith("ref: ") else head

    def version(name):
        try:
            return m.version(name)
        except m.PackageNotFoundError:
            return None

    info = dict(python=platform.python_version(), openpi_commit=commit.strip() or None,
                **{k: version(k) for k in ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "torch")},
                jax_cuda_plugin=version("jax-cuda12-plugin"), gpu=jax.devices()[0].device_kind,
                os_in_container=platform.platform())
    (out_dir / "versions_cluster.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info))


def sums(args):
    bundle = Path(args.bundle)
    lines = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        rel = path.relative_to(bundle).as_posix()
        if rel in ("README.md", "SHA256SUMS") or "__pycache__" in rel:
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {rel}\n")
    (bundle / "SHA256SUMS.tmp").write_text("".join(lines), encoding="utf-8")
    os.replace(bundle / "SHA256SUMS.tmp", bundle / "SHA256SUMS")
    total = sum((bundle / l.split("  ", 1)[1].strip()).stat().st_size for l in lines)
    print(json.dumps(dict(files=len(lines), bytes=total)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    a = sub.add_parser("stats")
    a.add_argument("--raw-root", action="append", type=Path, required=True)
    a.add_argument("--tasks", nargs="+", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("env")
    b.add_argument("--openpi", required=True)
    b.add_argument("--out-dir", required=True)
    c = sub.add_parser("sums")
    c.add_argument("--bundle", required=True)
    args = ap.parse_args()
    {"stats": stats, "env": env, "sums": sums}[args.action](args)


if __name__ == "__main__":
    main()
