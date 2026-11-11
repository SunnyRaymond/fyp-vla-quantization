"""Background watcher: quiet while the swap job is queued or running; exits on job end, a FAILED marker or three failed
queries in a row, and pulls the results when the job has ended.

  poll.py --interval 120
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PY = HERE.parents[2] / "nscc-access/.venv/Scripts/python.exe"


def control(action, timeout):
    return subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), action], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    failed = 0
    while True:
        try:
            out = control("status", 170)
            snap = json.loads(out.stdout) if out.returncode == 0 else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            snap = None
        if snap is None:
            failed += 1
            if failed >= 3:
                print(json.dumps(dict(event="query_failed")), flush=True)
                return 1
        else:
            failed = 0
            job = snap["jobs"].get("swap", {})
            if snap["failed"] or job.get("job_state") in ("F", "X") or snap.get("complete"):
                print(json.dumps(dict(event="failure" if snap["failed"] else "terminal", **snap), ensure_ascii=False), flush=True)
                pulled = control("pull", 600)
                print(pulled.stdout[-2000:] or pulled.stderr[-2000:], flush=True)
                results = HERE / "pulled" / "results" / "RESULTS.zh.md"
                if results.exists():
                    print(results.read_text(encoding="utf-8"), flush=True)
                return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
