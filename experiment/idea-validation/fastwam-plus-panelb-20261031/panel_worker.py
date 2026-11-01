"""Panel-B closed loop for one arm on one worker slot, on the A16 runtime (fastwam-a16-baselines loop_worker).

The A16 build_runtime runs unchanged against this panel's results root: model without T5 + the panel-B T5 cache must
reproduce the full-T5 BF16 case-2000 anchor bitwise, then the arm's weights go in (dense dequantized BF16 from the A16
quantized/ directory; BF16 installs nothing) and eager vs cache + CUDA Graph must be bitwise equal. The model loads
while slot 0 is still building the T5 cache; the worker waits for prep/T5_COMPLETE.json before using it.

Episodes are claimed across nodes with an atomic mkdir of loops/<arm>/claims/<index>; the slot index (280 + panel
position) drives the env and sampler seeds exactly as panel A's index does.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import loop_worker as lw
from common import guard, read, results, save

SETTLING = "Task completed during settling"


def panel_rows(p):
    doc = read(results(p) / "panel" / "manifest.json")
    rows = doc["variants"]
    offset = p["panel_b"]["index_offset"]
    if doc.get("protocol") != p["protocol"] or [r["index"] for r in rows] != list(range(offset, offset + 280)):
        raise RuntimeError("Panel-B manifest does not match this protocol")
    return rows


def task_order(rows):
    """Longest suite first, as in panel A, so the tail of each queue is short episodes."""
    return sorted(rows, key=lambda r: (r["suite"] != "libero_10", r["index"]))


def wait_t5(p, poll=10.0):
    prep, limit = results(p) / "prep", p["schedule"]["t5_wait_seconds"]
    start = time.monotonic()
    while not (prep / "T5_COMPLETE.json").exists():
        if (prep / "T5_FAILED.json").exists():
            raise RuntimeError("T5 cache build failed (prep/T5_FAILED.json)")
        if time.monotonic() - start > limit:
            raise TimeoutError(f"T5 cache not ready after {limit}s")
        time.sleep(poll)
    return time.monotonic() - start


def patch(p, pa):
    """Point the A16 loop worker at the arm's weight source and make it wait for the T5 cache after loading."""
    original_install, original_runtime = lw.install_weights, lw.bf16_runtime
    waited = {}

    def install_weights(_p, runtime, arm):
        source = p["arms"][arm]["weights"]
        if source is not None:
            original_install(pa, runtime, source)

    def bf16_runtime(out, text_encoder=True):
        runtime = original_runtime(out, text_encoder=text_encoder)
        waited["t5_wait_seconds"] = wait_t5(p)
        return runtime

    lw.install_weights, lw.bf16_runtime = install_weights, bf16_runtime
    return waited


def claim(root, order, start, worker):
    """Next unclaimed slot at or after position start in order; returns (row, next_start) or (None, len)."""
    claims = root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for k in range(start, len(order)):
        target = claims / str(order[k]["index"])
        try:
            target.mkdir()
        except FileExistsError:
            continue
        (target / f"worker_{worker}").touch()
        return order[k], k + 1
    return None, len(order)


def run_episode(runtime, row, arm):
    try:
        return lw.episode(runtime, row, row["index"], arm)
    except RuntimeError as exc:
        if SETTLING not in str(exc):
            raise
        if runtime.env is not None:
            runtime.env.close()
            runtime.env = None
        return dict(arm=arm, variant_id=str(row["variant_id"]), index=row["index"], suite=row["suite"],
                    dimension=row["dimension"], task_id=int(row["task_id"]), success=False, invalid=True,
                    termination="settling_success", steps=0, replans=0, episode_seconds=0.0, policy_call_seconds=[],
                    pbs_jobid=os.environ["PBS_JOBID"])


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    if args.arm not in p["arms"]:
        raise ValueError(args.arm)
    pa = read(p["a16"]["protocol"])
    order = task_order(panel_rows(p))
    root = results(p) / "loops" / args.arm
    out = root / f"worker_{args.worker}"
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    started = time.monotonic()
    waited = patch(p, pa)
    runtime = lw.build_runtime(dict(pa, remote_root=p["remote_root"], protocol=p["protocol"]), args.arm, out)
    load_seconds, completed, position = time.monotonic() - started, [], 0
    try:
        while True:
            row, position = claim(root, order, position, args.worker)
            if row is None:
                break
            save(out / "progress.json", dict(phase="episode", index=row["index"], completed=len(completed),
                                             updated_epoch=time.time()))
            record = run_episode(runtime, row, args.arm)
            record["worker"], record["host"] = args.worker, os.uname().nodename
            save(root / "episodes" / f"{row['index']}.json", record)
            completed.append(row["index"])
            print(f"EPISODE arm={args.arm} worker={args.worker} index={row['index']} success={record['success']} "
                  f"steps={record['steps']}" + (" invalid" if record.get("invalid") else ""), flush=True)
    finally:
        if runtime.env is not None:
            runtime.env.close()
    save(out / "summary.json", dict(protocol=p["protocol"], arm=args.arm, pbs_jobid=os.environ["PBS_JOBID"],
                                    complete=True, worker=args.worker, host=os.uname().nodename, episodes=completed,
                                    load_and_anchor_seconds=load_seconds, **waited))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
