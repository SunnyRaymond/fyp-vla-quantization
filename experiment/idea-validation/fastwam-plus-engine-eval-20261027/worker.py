"""Persistent engine workers on the frozen 280-variant LIBERO-Plus panel; one arm per invocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time

from plus_engine import build_runtime, episode, read, save


def guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if (not os.environ.get("PBS_JOBID") or "login" in host.lower() or not nodefile
            or host not in {x.split(".")[0] for x in Path(nodefile).read_text().split()}):
        raise RuntimeError("Approved PBS compute allocation required")


def arm_slots(cfg, total=280):
    slots = cfg.get("slots", "all")
    slots = list(range(total)) if slots == "all" else list(slots)
    if "shard" in cfg:
        k, n = cfg["shard"]
        slots = [s for s in slots if s % n == k]
    return slots


def episode_dir(root, arm, cfg):
    return root / cfg.get("group", arm) / "episodes"


def init_queue(p, arms, arm, rows):
    """Pending = this arm's slots without a saved episode; long-suite slots first."""
    root = Path(p["remote_root"]) / "results"
    done = {int(f.stem) for f in episode_dir(root, arm, arms[arm]).glob("*.json")}
    pending = [s for s in arm_slots(arms[arm]) if s not in done]
    pending.sort(key=lambda s: (rows[s]["suite"] != "libero_10", s))
    save(root / arm / "tasks.json", pending)
    save(root / arm / "task_queue.json", dict(next=0, claims=[]))
    print(f"QUEUE arm={arm} pending={len(pending)} done={len(done)}", flush=True)


def claim(root, worker):
    import fcntl
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue, tasks = read(root / "task_queue.json"), read(root / "tasks.json")
        if queue["next"] >= len(tasks):
            return None
        index = tasks[queue["next"]]
        queue["next"] += 1
        queue["claims"].append(dict(index=index, worker=worker, epoch=time.time()))
        save(root / "task_queue.json", queue)
        return index


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    p, arms = read(args.protocol), read(args.arms)
    cfg = arms[args.arm]
    rows = read(p["panel"]["manifest"])["variants"]
    if len(rows) != 280:
        raise RuntimeError("Expected the frozen 280-variant manifest")
    root = Path(p["remote_root"]) / "results"
    if args.init:
        init_queue(p, arms, args.arm, rows)
        return
    arm_root, episodes = root / args.arm, episode_dir(root, args.arm, cfg)
    out = arm_root / f"worker_{args.worker}"
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    started = time.monotonic()
    runtime = build_runtime(p, args.arm, cfg, out)
    load_seconds, completed = time.monotonic() - started, []
    try:
        while (index := claim(arm_root, args.worker)) is not None:
            save(out / "progress.json", dict(phase="episode", index=index, completed=len(completed),
                                             updated_epoch=time.time()))
            record = episode(runtime, rows[index], index, args.arm)
            record["worker"] = args.worker
            save(episodes / f"{index}.json", record)
            completed.append(index)
            print(f"EPISODE arm={args.arm} worker={args.worker} index={index} "
                  f"success={record['success']} steps={record['steps']}", flush=True)
    finally:
        if runtime.env is not None:
            runtime.env.close()
    save(out / "summary.json", dict(protocol=p["protocol"], arm=args.arm, pbs_jobid=os.environ["PBS_JOBID"],
         complete=True, worker=args.worker, episodes=completed, load_and_anchor_seconds=load_seconds))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
