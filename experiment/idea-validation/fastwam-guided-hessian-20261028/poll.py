"""Block until the job's state or worker phase changes, a worker failure appears, or --window seconds pass.

Prints one JSON line: {event, status, job_state, workers, ...}. event is 'terminal', 'failure', 'changed' or 'idle'.
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
STATE = HERE / "poll_state.json"
ATTEMPT = []


def snapshot():
    out = subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), *ATTEMPT, "status"], capture_output=True,
                         text=True, encoding="utf-8", timeout=170)
    if out.returncode != 0:
        return dict(status="query_failed", error=out.stderr[-400:])
    d = json.loads(out.stdout)
    s = d.get("scheduler") or {}
    workers = {}
    for w, v in (d.get("workers") or {}).items():
        p, f = v.get("progress") or {}, v.get("failure")
        workers[w] = dict(phase=p.get("phase"), arm=p.get("arm"), queries=p.get("queries"),
                          contexts=p.get("contexts"), failure=f)
    return dict(status=d.get("status"), job_state=s.get("job_state"), exit_status=s.get("Exit_status"),
                completion_passed=d.get("completion_passed"), workers=workers)


def key(snap):
    return json.dumps(dict(status=snap.get("status"), job_state=snap.get("job_state"),
                           phases={w: (v["phase"], v["arm"]) for w, v in snap.get("workers", {}).items()}),
                      sort_keys=True)


def until_done(interval):
    """Background watcher: quiet until terminal / worker failure / 10 h of failed queries (the jump host is only
    reachable over the NTU VPN, which has dropped for hours at night); pulls on terminal."""
    failed = 0
    while True:
        snap = snapshot()
        failed = failed + 1 if snap.get("status") == "query_failed" else 0
        terminal = snap.get("status") in ("complete", "terminal_unverified") or snap.get("job_state") in ("F", "X")
        failure = any(v.get("failure") for v in snap.get("workers", {}).values())
        if terminal or failure or failed * interval >= 36000:
            event = "terminal" if terminal else "failure" if failure else "query_failed"
            print(json.dumps(dict(event=event, **snap), ensure_ascii=False, default=str), flush=True)
            if terminal:
                out = subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), *ATTEMPT, "pull"], capture_output=True,
                                     text=True, encoding="utf-8", timeout=600)
                print(out.stdout[-3000:] or out.stderr[-3000:], flush=True)
            return 0
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=540)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--until-done", action="store_true")
    parser.add_argument("--attempt", default="")
    parser.add_argument("--phase", default="hosts")
    args = parser.parse_args()
    ATTEMPT[:] = ["--phase", args.phase] + (["--attempt", args.attempt] if args.attempt else [])
    if args.until_done:
        return until_done(args.interval)
    previous = json.loads(STATE.read_text(encoding="utf-8")).get("key") if STATE.exists() else None
    end = time.time() + args.window
    while True:
        snap = snapshot()
        if snap.get("status") in ("complete", "terminal_unverified") or snap.get("job_state") in ("F", "X"):
            event = "terminal"
        elif any(v.get("failure") for v in snap.get("workers", {}).values()):
            event = "failure"
        elif snap.get("status") != "query_failed" and key(snap) != previous:
            event = "changed"
        else:
            event = None
        if event or time.time() + args.interval > end:
            if snap.get("status") != "query_failed":
                STATE.write_text(json.dumps(dict(key=key(snap))), encoding="utf-8")
            print(json.dumps(dict(event=event or "idle", **snap), ensure_ascii=False, default=str))
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
