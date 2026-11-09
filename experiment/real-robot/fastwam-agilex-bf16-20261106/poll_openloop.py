"""Background watcher for the open-loop race (control.py race-openloop).

Polls every --race-interval s until one copy runs, then qdel's the copy still waiting (the in-job lock already stops
a second copy from working if both start together). Then polls every --interval s, quiet, until the running copy
ends; pulls results and prints one JSON line. Exits after 10 h of failed queries (the jump host needs the NTU VPN).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race-interval", type=int, default=60)
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()
    cancelled, winner, failed_queries = {}, None, 0
    while True:
        interval = args.race_interval if winner is None else args.interval
        jump = t = None
        try:
            jump, t = c.base.direct_connect()
            with c.remote.sftp_client(t) as sftp:
                jobs = c.load_jobs(sftp)
            ids = {k: jobs[k]["job_id"] for k in c.RACE}
            states = {k: c.qstat(t, j) for k, j in ids.items()}
            if any(s["job_state"] is None for s in states.values()):
                raise RuntimeError("qstat failed")
            failed_queries = 0
            if winner is None:
                running = [k for k, s in states.items() if s["job_state"] == "R"]
                if running:
                    winner = running[0]
                    for k, s in states.items():
                        if k not in running and s["job_state"] in ("Q", "H", "W"):
                            code, out = c.run(t, "qdel " + shlex.quote(ids[k]), timeout=35)
                            cancelled[k] = dict(job_id=ids[k], rc=code, out=out[-200:])
                    print(json.dumps(dict(event="race_started", winner=winner, job=ids[winner], cancelled=cancelled)),
                          flush=True)
                elif all(s["job_state"] == "F" for s in states.values()):
                    winner = "none"
            if winner is not None and (winner == "none" or states[winner]["job_state"] == "F"):
                st = c.status(t)
                if winner != "none" and st["race_lost"].get(winner):
                    # both started in the same poll and this one lost the in-job lock: follow the other copy
                    winner = next(k for k in c.RACE if k != winner)
                    print(json.dumps(dict(event="race_switched", winner=winner, job=ids[winner])), flush=True)
                else:
                    pulled = c.pull(t)
                    print(json.dumps(dict(event="ended", winner=winner, cancelled=cancelled,
                                          jobs={k: dict(job_id=ids[k], state=s["job_state"], exit=s["Exit_status"],
                                                        queue=s["queue"], walltime=s["resources_used.walltime"])
                                                for k, s in states.items()},
                                          openloop=st["openloop"], race_lost=st["race_lost"], failed=st["failed"],
                                          pulled=pulled["files"]), ensure_ascii=False), flush=True)
                    return 0
        except Exception as error:
            failed_queries += 1
            if failed_queries * interval >= 36000:
                print(json.dumps(dict(event="query_failed", error=f"{type(error).__name__}: {error}")), flush=True)
                return 1
        finally:
            if t:
                t.close()
            if jump:
                jump.close()
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
