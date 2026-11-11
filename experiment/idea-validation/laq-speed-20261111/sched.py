"""Worker-slot scheduler of one model's closed loops (stdlib, host python3.11): each slot asks which arm to load next.

  init               drop claims without a finished episode (a previous job died mid-episode) and all slot state
  choose --slot k    print the arm slot k should load next, or nothing when no arm is worth a (re)load
  status             per-arm claimed / finished / active counts
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import shutil
import time

PANEL = 280


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def res(p, model):
    return Path(p["remote_root"]) / "results" / model


def count(d):
    return sum(1 for _ in d.iterdir()) if d.exists() else 0


@contextlib.contextmanager
def lock(p, model, timeout=300.0):
    path = res(p, model) / "sched" / "lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > 120:
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


def slots(p, model):
    d = res(p, model) / "sched"
    return {int(f.stem.split("_")[1]): read(f).get("arm") for f in d.glob("slot_*.json")} if d.exists() else {}


def pick(arms, active, unclaimed, join_min):
    best, best_score = None, 0.0
    for arm in arms:
        left, n = unclaimed[arm], active.get(arm, 0)
        if left <= 0 or (n and left / n < join_min):
            continue
        score = left / (n + 1)
        if score > best_score:
            best, best_score = arm, score
    return best


def choose(p, model, slot):
    with lock(p, model):
        current = slots(p, model)
        current.pop(slot, None)
        active = {}
        for arm in current.values():
            if arm:
                active[arm] = active.get(arm, 0) + 1
        unclaimed = {a: PANEL - count(res(p, model) / "loops" / a / "claims") for a in p["arms"]}
        arm = pick(p["arms"], active, unclaimed, p["models"][model]["jobs"]["loop"]["join_min"])
        save(res(p, model) / "sched" / f"slot_{slot}.json", dict(arm=arm, unclaimed=unclaimed, active=active))
    return arm


def init(p, model):
    dropped = {}
    for arm in p["arms"]:
        claims, episodes = res(p, model) / "loops" / arm / "claims", res(p, model) / "loops" / arm / "episodes"
        dropped[arm] = 0
        if claims.exists():
            for c in claims.iterdir():
                if not (episodes / f"{c.name}.json").exists():
                    shutil.rmtree(c)
                    dropped[arm] += 1
    sched = res(p, model) / "sched"
    if sched.exists():
        shutil.rmtree(sched)
    sched.mkdir(parents=True)
    return dropped


def status(p, model):
    current = slots(p, model)
    return {a: dict(claimed=count(res(p, model) / "loops" / a / "claims"),
                    finished=count(res(p, model) / "loops" / a / "episodes"),
                    active=sum(v == a for v in current.values())) for a in p["arms"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "choose", "status"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()
    p = read(args.protocol)
    if args.action == "init":
        print("SCHED_INIT dropped_claims=" + json.dumps(init(p, args.model)), flush=True)
    elif args.action == "choose":
        print(choose(p, args.model, args.slot) or "")
    else:
        print(json.dumps(status(p, args.model)))


if __name__ == "__main__":
    main()
