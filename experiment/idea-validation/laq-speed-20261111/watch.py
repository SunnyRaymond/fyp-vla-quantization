"""Background watcher: silent while nothing changes; prints one JSON line and exits on an event.

  --phase calib   as soon as one calibration job runs, the other one and the loop job behind it are cancelled (the pair
                  that lost the race); exits when the running calibration job ends, after pulling the small results.
  --phase loop    exits when a closed-loop job ends or writes FAILED, after pulling the small results.
Three failed status queries in a row also end the watcher.
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
PAIRS = {"short": "long", "long": "short"}


def control(*args, timeout=300):
    return subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def brief(d, keys):
    return {k: {f: d["jobs"][k].get(f) for f in ("job_id", "job_state", "Exit_status", "queue", "resources_used.walltime")}
            for k in keys if k in d["jobs"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calib", "loop"), required=True)
    parser.add_argument("--jobs", nargs="*", help="job keys to watch (default: every job of the phase)")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    failures = 0
    dropped = set()
    while True:
        try:
            out = control("status")
            d = json.loads(out.stdout) if out.returncode == 0 else None
            error = out.stderr[-500:]
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            d, error = None, repr(exc)[-500:]
        if d is None:
            failures += 1
            if failures >= 3:
                print(json.dumps(dict(event="query_failed", stderr=error)), flush=True)
                return 1
            time.sleep(args.interval)
            continue
        failures = 0
        jobs = d["jobs"]
        watched = args.jobs or [k for k in jobs if k.startswith(args.phase)]
        if args.phase == "calib":  # the first calibration job to start wins: cancel the other one and its loop job
            running = [k.split("_")[1] for k in ("calib_short", "calib_long") if jobs.get(k, {}).get("job_state") == "R"]
            if len(running) == 1 and running[0] not in dropped:
                loser = PAIRS[running[0]]
                live = [k for k in (f"calib_{loser}", f"loop_{loser}") if jobs.get(k, {}).get("job_state") in ("Q", "H", "W")]
                if live:
                    control("cancel", "--jobs", *live)
                dropped.add(running[0])
                watched = [f"calib_{running[0]}"]
            elif dropped:
                watched = [f"calib_{w}" for w in dropped]
        ended = [k for k in watched if jobs.get(k, {}).get("job_state") in ("F", "X")]
        if "cputest" in jobs and jobs["cputest"].get("job_state") == "F" and jobs["cputest"].get("Exit_status") != "0":
            ended.append("cputest")
        ended += [k for k in d["failed"] if k in watched and k not in ended]
        if ended:
            cancelled = None
            if args.phase == "calib":
                winner = next((k.split("_")[1] for k in ended if k.startswith("calib") and jobs[k].get("Exit_status") == "0"), None)
                if winner:
                    loser = PAIRS[winner]
                    live = [k for k in (f"calib_{loser}", f"loop_{loser}") if jobs.get(k, {}).get("job_state") in ("Q", "H", "R", "W")]
                    if live:
                        cancelled = dict(jobs=live, rc=control("cancel", "--jobs", *live).returncode)
            control("pull", timeout=900)
            print(json.dumps(dict(event="ended", ended=brief(d, ended), cancelled=cancelled, jobs=brief(d, list(jobs)),
                                  failed=d["failed"], calib=d["calib"], episodes=d["episodes"]), ensure_ascii=False), flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
