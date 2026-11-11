"""Watch jobs of this experiment: silent while they are queued or running, pull and exit when the named jobs finish.

  poll.py --jobs cputest int4 [--interval 180]
Exit 0 when every job finished with exit 0, 1 when one failed (a failed producer ends the watch at once)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def control(*args):
    out = subprocess.run([sys.executable, "-Xutf8", "-B", str(HERE / "control.py"), *args], capture_output=True, text=True)
    return out.returncode, out.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=180)
    args = parser.parse_args()
    failures = 0
    while True:
        code, out = control("status")
        try:
            jobs = json.loads(out)["jobs"]
        except (ValueError, KeyError):
            failures += 1
            if failures >= 5:
                print("status failed five times in a row:\n" + out[-2000:])
                return 1
            time.sleep(args.interval)
            continue
        failures = 0
        states = {k: (jobs[k]["job_state"], jobs[k]["Exit_status"]) for k in args.jobs}
        failed = [k for k, (s, e) in states.items() if s == "F" and e != "0"]
        if failed or all(s == "F" for s, _ in states.values()):
            print(json.dumps(states))
            print(control("pull")[1])
            return 1 if failed else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
