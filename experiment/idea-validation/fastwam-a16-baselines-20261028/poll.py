"""Background watcher for one or more A16 jobs: silent while nothing changes; prints one JSON line and exits when a
job ends, prep completes, an arm finishes quantization or its closed loop, or status queries fail 3 times in a row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PY = HERE.parents[2] / "nscc-access/.venv/Scripts/python.exe"


def snapshot(job):
    out = subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), "status", "--job", job],
                         capture_output=True, text=True, encoding="utf-8", timeout=240)
    if out.returncode != 0:
        return None
    d = json.loads(out.stdout)
    s = d.get("scheduler") or {}
    return dict(job=job, job_id=(d.get("handle") or {}).get("job_id"), job_state=s.get("job_state"),
                exit_status=s.get("Exit_status"), prep=d.get("prep_complete"), quantized=sorted(d.get("quantized", {})),
                loops_done=d.get("loops_done"), loops=d.get("loops"), log_tail=(d.get("log_tail") or "")[-1500:])


def milestones(snap):
    return (snap["prep"], tuple(snap["quantized"]), tuple(snap["loops_done"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    first, failures = {}, 0
    while True:
        snaps = [snapshot(job) for job in args.jobs]
        if any(s is None for s in snaps):
            failures += 1
            if failures >= 3:
                print(json.dumps(dict(event="query_failed")), flush=True)
                return 1
            time.sleep(args.interval)
            continue
        failures = 0
        for snap in snaps:
            first.setdefault(snap["job"], milestones(snap))
            ended = snap["job_state"] in ("F", "X")
            if ended or milestones(snap) != first[snap["job"]]:
                print(json.dumps(dict(event="ended" if ended else "milestone", **snap), ensure_ascii=False), flush=True)
                if ended:
                    subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), "pull"],
                                   capture_output=True, text=True, encoding="utf-8", timeout=600)
                return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
