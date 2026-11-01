"""Race a multi-GPU tune job against single-GPU jobs that cover the same hosts (tune.json "split"), polling every
--interval s. The first start decides: the big job first (or at the same poll as a small one) -> qdel the small jobs still
waiting or running; a small job first -> qdel the big job and, with --then PHASE, submit that control.py phase (the
hosts the small jobs leave uncovered). A job that ends before any starts is reported and nothing is cancelled. Prints
one JSON line and exits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PY = HERE.parents[2] / "nscc-access/.venv/Scripts/python.exe"
sys.path.insert(0, str(HERE))
import control  # noqa: E402

base = control.base


def qdel(transport, job_id):
    if not base.HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, out = base.remote.command(transport, "qdel " + shlex.quote(job_id), timeout=35)
    return dict(job_id=job_id, rc=code, output=out[-300:],
                after=base._scheduler(transport, job_id).get("job_state"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big", required=True, help="PBS id of the multi-GPU job")
    parser.add_argument("--small", required=True, nargs="+", help="PBS ids of the single-GPU jobs")
    parser.add_argument("--then", help="control.py phase to submit when a small job starts first")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    control.configure("", "tune")
    failed = 0
    while True:
        jump = transport = None
        try:
            jump, transport = base.direct_connect()
            big = base._scheduler(transport, args.big).get("job_state")
            small = {j: base._scheduler(transport, j).get("job_state") for j in args.small}
            if big is None or None in small.values():
                raise RuntimeError("qstat failed")
            failed = 0
            if big == "R":
                done = [qdel(transport, j) for j, s in small.items() if s in ("Q", "H", "R")]
                print(json.dumps(dict(event="big_started", big=big, small=small, cancelled=done)), flush=True)
                return 0
            if "R" in small.values():
                done = qdel(transport, args.big) if big in ("Q", "H") else None
                result = dict(event="small_started", big=big, small=small, cancelled=done)
                if args.then:
                    out = subprocess.run([str(PY), "-Xutf8", "-B", str(HERE / "control.py"), "--phase", args.then, "submit"],
                                         capture_output=True, text=True, encoding="utf-8", timeout=300)
                    result.update(then=args.then, then_rc=out.returncode, then_output=(out.stdout or out.stderr)[-1500:])
                print(json.dumps(result), flush=True)
                return 0
            if "F" in (big, *small.values()):
                print(json.dumps(dict(event="ended_in_queue", big=big, small=small)), flush=True)
                return 0
        except Exception as error:  # the jump host is only reachable over the NTU VPN, which drops at night
            failed += 1
            if failed * args.interval >= 36000:
                print(json.dumps(dict(event="query_failed", error=f"{type(error).__name__}: {error}")), flush=True)
                return 1
        finally:
            if transport:
                transport.close()
            if jump:
                jump.close()
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
