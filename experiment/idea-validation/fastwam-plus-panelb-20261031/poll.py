"""Background watcher for the panel-B job: quiet until the job ends, FAILED appears, or status queries fail for 10 h
(the jump host needs the NTU VPN, which has dropped for hours at night). Pulls on the way out; prints one JSON line.
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


def control(*args, timeout=240):
    return subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    attempt = ["--attempt", args.attempt] if args.attempt else []
    failed = 0
    while True:
        try:
            out = control("status", *attempt)
        except subprocess.TimeoutExpired:
            out = None
        if out is None or out.returncode != 0:
            failed += 1
            if failed * args.interval >= 36000:
                print(json.dumps(dict(event="query_failed")), flush=True)
                return 1
            time.sleep(args.interval)
            continue
        failed = 0
        d = json.loads(out.stdout)
        s = d.get("scheduler") or {}
        ended = s.get("job_state") in ("F", "X")
        if ended or d.get("failed"):
            pulled = control("pull", *attempt, timeout=900)
            print(json.dumps(dict(event="ended" if ended else "failure", job_state=s.get("job_state"),
                                  exit_status=s.get("Exit_status"), failed=d.get("failed"), arms=d.get("arms"),
                                  pipeline_complete=d.get("pipeline_complete"),
                                  log_tail=(d.get("log_tail") or "")[-1500:], pull=pulled.stdout[-500:]),
                             ensure_ascii=False), flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
