"""Background watcher of the Fig. 3 export jobs: silent while they are queued or running, pulls when all have ended.

  watch.py [--interval 120]
Exit 0 when every recorded job finished with PIPELINE_COMPLETE, 1 when one failed, 2 after repeated query failures.
"""
from __future__ import annotations

import argparse
import json
import time

import control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    failures = 0
    while True:
        try:
            jump, t = control.base.direct_connect()
            try:
                state = control.status(t)
                if state and all(v["job_state"] == "F" for v in state.values()):
                    pulled = control.pull(t)
                    print(json.dumps(dict(state=state, pulled=pulled), indent=2))
                    return 0 if all(v["complete"] and v["Exit_status"] == "0" for v in state.values()) else 1
            finally:
                t.close()
                if jump:
                    jump.close()
            failures = 0
        except Exception as error:  # a dropped connection is retried; five in a row end the watch
            failures += 1
            if failures >= 5:
                print(f"WATCH query failed {failures} times: {error!r}")
                return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
