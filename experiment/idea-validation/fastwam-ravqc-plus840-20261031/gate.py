"""Which arms run (stdlib, host python3.11). Waits (up to --wait seconds) for the calibration job's
results/ravqc/DECISION.json, then keeps a calibrated arm when it was trained and its held-out mixed loss is not above its
uncalibrated base (user rule, 2026-10-31); an uncalibrated RAVQU arm always runs. Writes results/arms.json and the job's
protocol copy with arm_order = the kept arms (sched.py and panel_worker.py read that copy)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def decide(p, root):
    rows = {}
    for arm in p["arm_order"]:
        if p["arms"][arm].get("uncalibrated"):
            rows[arm] = dict(run=True, reason="uncalibrated base: no held-out gate")
            continue
        a = root / "results" / "ravqc" / arm / "arm.json"
        if not a.exists() or not read(a).get("tuned"):
            rows[arm] = dict(run=False, reason="not trained")
            continue
        mine = read(root / "results" / "ravqc" / "heldout" / f"{arm}.json")["overall"]
        base = read(root / "results" / "heldout" / f"{p['arms'][arm]['base']}.json")["overall"]
        keys = ("mixed_loss", "future_latent_nmse", "first10motor_rmse", "first10gripper_rmse")
        ratio = {k: mine[k] / base[k] for k in keys}
        rows[arm] = dict(run=ratio["mixed_loss"] <= 1.0, ratio=ratio, rule="held-out mixed loss <= base")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--wait", type=int, default=14400)
    args = parser.parse_args()
    p = read(args.protocol)
    root = Path(p["ravqc"]["root"])
    decision, start = root / "results" / "ravqc" / "DECISION.json", time.monotonic()
    while not decision.exists():
        if time.monotonic() - start > args.wait:
            raise SystemExit(f"GATE no {decision} after {args.wait}s")
        time.sleep(60)
    rows = decide(p, root)
    keep = [a for a in p["arm_order"] if rows[a]["run"]]
    save(Path(p["remote_root"]) / "results" / "arms.json", dict(protocol=p["protocol"], arms=rows, run=keep,
                                                                waited_seconds=time.monotonic() - start))
    save(args.protocol, dict(p, arm_order=keep))
    print("GATE " + json.dumps(dict(run=keep, arms=rows)), flush=True)
    print("RUN", *keep, flush=True)  # run.pbs reads the arms from this line only


if __name__ == "__main__":
    main()
