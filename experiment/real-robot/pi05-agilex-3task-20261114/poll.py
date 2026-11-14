"""Background watcher: quiet while the recorded jobs queue or run; prints one JSON line and exits when the --until job
ends (or, with --running-for, once it has run that long), when a watched job ends non-zero (PBS then drops its afterok
dependents), when a FAILED marker appears, or after 10 h of failed status queries (the jump host needs the NTU VPN).
Pulls small results on the way out.
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


def control(*args, timeout=300):
    return subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", required=True, help="job key in jobs.json: prep or train")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--running-for", type=int, help="also exit once the --until job has run this many seconds")
    args = parser.parse_args()
    failed_queries = 0
    while True:
        try:
            out = control("status")
        except subprocess.TimeoutExpired:
            out = None
        if out is None or out.returncode != 0:
            failed_queries += 1
            if failed_queries * args.interval >= 36000:
                print(json.dumps(dict(event="query_failed")), flush=True)
                return 1
            time.sleep(args.interval)
            continue
        failed_queries = 0
        d = json.loads(out.stdout)
        jobs = d["jobs"]
        watched = {args.until, *{"train": ["prep"]}.get(args.until, [])}
        bad = {k: v for k, v in jobs.items() if k in watched and v.get("job_state") == "F"
               and v.get("Exit_status") not in ("0", None)}
        d["failed"] = {k: v for k, v in d["failed"].items() if k in watched}
        target = jobs.get(args.until, {})
        used = target.get("resources_used.walltime") or "0:0:0"
        ran = sum(int(x) * f for x, f in zip(used.split(":"), (3600, 60, 1)))
        running = bool(args.running_for) and target.get("job_state") == "R" and ran >= args.running_for
        if bad or d["failed"] or target.get("job_state") == "F" or running:
            pulled = control("pull", timeout=900)
            event = "failure" if bad or d["failed"] else "ended" if target.get("job_state") == "F" else "running"
            print(json.dumps(dict(event=event, until=args.until,
                                  jobs={k: dict(state=v.get("job_state"), exit=v.get("Exit_status"),
                                                walltime=v.get("resources_used.walltime")) for k, v in jobs.items()},
                                  bad=sorted(bad), failed=d["failed"], prep=d["prep"], train=d["train"],
                                  pull=pulled.stdout[-300:]), ensure_ascii=False), flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
