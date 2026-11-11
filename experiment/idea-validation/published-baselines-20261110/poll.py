"""Background watcher: silent while nothing changes; prints one JSON line and exits when a watched job ends, a job writes
FAILED, or status queries fail three times in a row. On a job end it also pulls the small results."""
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
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    failures = 0
    while True:
        out = control("status")
        if out.returncode != 0:
            failures += 1
            if failures >= 3:
                print(json.dumps(dict(event="query_failed", stderr=out.stderr[-500:])), flush=True)
                return 1
            time.sleep(args.interval)
            continue
        failures = 0
        d = json.loads(out.stdout)
        ended = [k for k in args.jobs if d["jobs"].get(k, {}).get("job_state") in ("F", "X")]
        if ended or d["failed"]:
            control("pull", timeout=900)
            print(json.dumps(dict(event="ended", ended={k: {f: d["jobs"][k].get(f) for f in ("job_id", "job_state", "Exit_status")}
                                                       for k in ended}, failed=d["failed"], models=d["models"]),
                             ensure_ascii=False), flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
