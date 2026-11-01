"""Worker-slot scheduler for the panel-C closed loops (stdlib; host python3.11).

State lives on GPFS under results/: loops/<arm>/claims/<index>/ (one atomic mkdir per claimed episode, written by
panel_worker), loops/<arm>/episodes/<index>.json (finished episodes), sched/slot_<k>.json (the arm slot k runs) and
sched/lock (a mkdir lock that serializes arm choices).

  init                 drop claims without a finished episode (a previous job died mid-episode) and all slot state
  choose --slot k      print the arm slot k should load next, or nothing when no arm is worth a (re)load
  status               per-arm claimed / finished / active counts
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import time

PANEL = 280


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def results(p):
    return Path(p["remote_root"]) / "results"


def loops(p, arm):
    return results(p) / "loops" / arm


def total(p, arm):
    """Episodes this job owes the arm: 280 per panel it runs here (W3 arms run B and C, the rest only C)."""
    return PANEL * len(p["arms"][arm]["panels"])


def claimed(p, arm):
    d = loops(p, arm) / "claims"
    return sum(1 for _ in d.iterdir()) if d.exists() else 0


def finished(p, arm):
    d = loops(p, arm) / "episodes"
    return sum(1 for f in d.glob("*.json")) if d.exists() else 0


@contextlib.contextmanager
def lock(p, timeout=300.0):
    path = results(p) / "sched" / "lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stale, start = p["schedule"]["lock_stale_seconds"], time.time()
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale:
                    path.rmdir()
                    continue
            except FileNotFoundError:
                continue
            if time.time() - start > timeout:
                raise TimeoutError(f"Scheduler lock held for more than {timeout:.0f}s: {path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.rmdir()


def slots(p):
    d = results(p) / "sched"
    return {int(f.stem.split("_")[1]): read(f).get("arm") for f in d.glob("slot_*.json")} if d.exists() else {}


def pick(p, active, unclaimed):
    """Arm with the most unclaimed episodes per worker after joining; joining a running arm needs join_min left."""
    best, best_score = None, 0.0
    for arm in p["arm_order"]:
        left, n = unclaimed[arm], active.get(arm, 0)
        if left <= 0 or (n and left / n < p["schedule"]["join_min"]):
            continue
        score = left / (n + 1)
        if score > best_score:
            best, best_score = arm, score
    return best


def choose(p, slot):
    with lock(p):
        current = slots(p)
        current.pop(slot, None)
        active = {}
        for arm in current.values():
            if arm:
                active[arm] = active.get(arm, 0) + 1
        unclaimed = {arm: total(p, arm) - claimed(p, arm) for arm in p["arm_order"]}
        arm = pick(p, active, unclaimed)
        save(results(p) / "sched" / f"slot_{slot}.json", dict(arm=arm, epoch=time.time(), unclaimed=unclaimed,
                                                              active=active))
    return arm


def init(p):
    dropped = {}
    for arm in p["arm_order"]:
        claims, episodes = loops(p, arm) / "claims", loops(p, arm) / "episodes"
        dropped[arm] = 0
        if claims.exists():
            for c in claims.iterdir():
                if not (episodes / f"{c.name}.json").exists():
                    shutil.rmtree(c)
                    dropped[arm] += 1
    sched = results(p) / "sched"
    if sched.exists():
        shutil.rmtree(sched)
    sched.mkdir(parents=True)
    return dropped


def status(p):
    current = slots(p)
    return {arm: dict(total=total(p, arm), claimed=claimed(p, arm), finished=finished(p, arm),
                      active=sum(a == arm for a in current.values())) for arm in p["arm_order"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "choose", "status"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()
    p = read(args.protocol)
    if args.action == "init":
        print("SCHED_INIT dropped_claims=" + json.dumps(init(p)), flush=True)
    elif args.action == "choose":
        if args.slot is None:
            parser.error("--slot is required")
        print(choose(p, args.slot) or "")
    else:
        print(json.dumps(status(p)))


if __name__ == "__main__":
    main()
