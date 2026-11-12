"""Background watcher: poll.py KEY[,KEY...] [--interval 300]

Silent while the jobs (keys of jobs.json) are pending or running; exits when one of them fails (sacct state or a FAILED
marker: exit 2), when all have ended (exit 0, after printing their states, episode counts and log tails), or after five
consecutive failed queries (exit 3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control  # noqa: E402

BAD = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")
ENDED = ("COMPLETED",) + BAD


def query(keys):
    state = control.jobs()
    ids = {k: state[k]["job"] for k in keys}
    t = control.eee.connect()
    try:
        rc, out, err = control.run(t, f"sacct -X -n -P -j {','.join(ids.values())} -o JobID,State,ExitCode,Elapsed; "
                                      f"ls {control.REMOTE}/artifacts/*/*/FAILED 2>/dev/null; "
                                      + "; ".join(f"echo COUNT {m} $(ls {control.REMOTE}/results/{m}/episodes 2>/dev/null | wc -l)"
                                                  for m in control.MODELS))
    finally:
        t.close()
    if rc != 0 and not out:
        raise RuntimeError(err)
    states, failed, counts = {}, [], {}
    for line in out.splitlines():
        if line.startswith("COUNT "):
            _, m, n = line.split()
            counts[m] = int(n)
        elif line.endswith("/FAILED"):
            failed.append(line)
        elif "|" in line:
            job, st, code, elapsed = line.split("|")[:4]
            states[job] = (st.split()[0], code, elapsed)
    return ids, states, failed, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keys")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--until-episodes", type=int, default=None,
                        help="also exit (code 0) once every model has at least this many saved episodes")
    args = parser.parse_args()
    keys = args.keys.split(",")
    misses = 0
    while True:
        try:
            ids, states, failed, counts = query(keys)
            misses = 0
        except Exception as exc:  # network or login hiccup
            misses += 1
            if misses >= 5:
                print(f"POLL_QUERY_FAILED {type(exc).__name__}: {exc}", flush=True)
                sys.exit(3)
            time.sleep(args.interval)
            continue
        mine = [f for f in failed if any(f"/{ids[k]}." in f for k in keys)]
        current = {k: states.get(ids[k], ("UNKNOWN", "", ""))[0] for k in keys}
        if mine or any(s in BAD for s in current.values()):
            print("POLL_FAILED", json.dumps(current), mine, json.dumps(counts), flush=True)
            sys.exit(2)
        if all(s in ENDED for s in current.values()):
            print("POLL_DONE", json.dumps({k: states[ids[k]] for k in keys}), json.dumps(counts), flush=True)
            sys.exit(0)
        if args.until_episodes is not None and counts and min(counts.values()) >= args.until_episodes:
            print("POLL_EPISODES", json.dumps(current), json.dumps(counts), flush=True)
            sys.exit(0)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
