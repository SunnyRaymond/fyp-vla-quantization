"""Stdlib helpers of the standard-LIBERO 840 loop: JSON I/O, the episode index, claims and their recovery.

Index i (0..839) is suite-major: i = 210 * suite + 21 * task + trial, suites in protocol order, trial = official
initial-state index. Seeds follow the LIBERO-Plus loops: environment 800000000 + 1000 i, sampler 900000000 + 1000 i + replan.
A claim is the directory claims/<i> holding owner.json (Slurm job id, restart count, worker); recover() drops claims
without an episode whose owner is no longer running, so a requeued or killed job leaves nothing stranded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def results(P, model):
    return Path(P["eee"]["root"]) / "results" / model


def index_rows(P):
    """Index rows without task names (the manifest adds names, languages and initial-state paths)."""
    b = P["benchmark"]
    rows = []
    for s, suite in enumerate(b["suites"]):
        for task in range(b["tasks_per_suite"]):
            for trial in range(b["trials_per_task"]):
                i = (s * b["tasks_per_suite"] + task) * b["trials_per_task"] + trial
                rows.append(dict(index=i, suite=suite, task_id=task, trial=trial,
                                 env_seed=800_000_000 + 1000 * i, sampler_seed=900_000_000 + 1000 * i))
    if [r["index"] for r in rows] != list(range(b["episodes"])):
        raise RuntimeError("Index rows do not cover 0..episodes-1")
    return rows


def manifest_rows(P, model):
    doc = read(results(P, model) / "manifest.json")
    rows = doc["rows"]
    expected = index_rows(P)
    if doc.get("protocol") != P["protocol"] or len(rows) != len(expected) or any(
            {k: r[k] for k in e} != e for r, e in zip(rows, expected)):
        raise RuntimeError("manifest.json differs from the protocol's index rows")
    return rows


def order(rows):
    """libero_10 first (longest episodes), then by index."""
    return sorted(rows, key=lambda r: (r["suite"] != "libero_10", r["index"]))


def owner_tag():
    return dict(job=os.environ["SLURM_JOB_ID"], restart=int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0))


def claim(root, ordered, start, worker):
    """Next row at or after `start` with no episode and no claim, claimed by an atomic mkdir of claims/<index>."""
    claims, episodes = root / "claims", root / "episodes"
    claims.mkdir(parents=True, exist_ok=True)
    for k in range(start, len(ordered)):
        index = ordered[k]["index"]
        if (episodes / f"{index}.json").exists():
            continue
        target = claims / str(index)
        try:
            target.mkdir()
        except FileExistsError:
            continue
        save(target / "owner.json", dict(owner_tag(), worker=worker, host=os.uname().nodename, epoch=time.time()))
        return ordered[k], k + 1
    return None, len(ordered)


def release(root, index):
    shutil.rmtree(root / "claims" / str(index), ignore_errors=True)


def recover(root, running, grace=120):
    """Drop claims without an episode whose owner job is not running (or is this job at an earlier restart)."""
    me = owner_tag()
    dropped = []
    claims = root / "claims"
    if not claims.is_dir():
        return dropped
    for target in sorted(claims.iterdir()):
        if (root / "episodes" / f"{target.name}.json").exists():
            continue
        try:
            owner = read(target / "owner.json")
        except (FileNotFoundError, json.JSONDecodeError):
            # mkdir happened but owner.json not yet written: only stale if old
            if time.time() - target.stat().st_mtime < grace:
                continue
            owner = dict(job=None, restart=None)
        stale = owner["job"] not in running or (owner["job"] == me["job"] and owner["restart"] != me["restart"])
        if stale:
            shutil.rmtree(target, ignore_errors=True)
            dropped.append(int(target.name))
    return dropped


def running_jobs(P, max_age=900):
    """Running Slurm job ids from running.json (written by every job's host monitor), or None if missing or stale."""
    try:
        doc = read(Path(P["eee"]["root"]) / "running.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return set(doc["jobs"]) if time.time() - doc["epoch"] <= max_age else None


def next_row(P, root, ordered, worker):
    """Recover stale claims (when the running-job list is fresh), then claim the first free row in order."""
    running = running_jobs(P)
    if running is not None:
        recover(root, running)
    row, _ = claim(root, ordered, 0, worker)
    return row


def done(root, total):
    return sum(1 for i in range(total) if (root / "episodes" / f"{i}.json").exists())


def walltime_seconds(text):
    parts = [int(x) for x in text.replace("-", ":").split(":")]
    if len(parts) == 4:
        d, h, m, s = parts
        return 86400 * d + 3600 * h + 60 * m + s
    h, m, s = parts
    return 3600 * h + 60 * m + s


def gpu_slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def metrics(action, future, ref_action, ref_future):
    """Action max-abs, first-10-step motor RMSE and future NMSE of (action, future) against a reference."""
    d = action.float() - ref_action.float()
    return dict(action_max_abs=float(d.abs().max()),
                first10_motor_rmse=float(d[:10, :6].pow(2).mean().sqrt()),
                future_nmse=float((future.float() - ref_future.float()).pow(2).sum() / ref_future.float().pow(2).sum()))
