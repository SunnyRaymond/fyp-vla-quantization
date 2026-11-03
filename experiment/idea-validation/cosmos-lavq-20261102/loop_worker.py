"""Closed loop for one LAVQ arm on one worker slot (the A16 baselines' loop_worker, this experiment's roots):
CosmosRuntime.episode on the arm's panels, episodes claimed one by one.

Startup (startup.json): the BF16 model on the merged T5 cache must reproduce the baselines' prep BF16 anchor (case 2000)
bitwise;
then the arm's dense BF16 weights go into the 280 target Linears and the arm anchor is taken; the first worker of an
arm stores it as loops/<arm>/anchor.pt and every later worker of that arm must reproduce it bitwise.

Episodes are claimed with an atomic mkdir of loops/<arm>/claims/<index>; libero_10 first so the tail is short.
A variant whose task completes during the 30 settling steps is saved as invalid (same for every arm). Any other
incomplete episode releases its claim and stops the worker (slots.sh restarts a slot once). Past the soft deadline no
new episode starts; an episode cut by it is dropped.

GPU memory: five workers share a 40 GB card. Model build and startup run under an exclusive per-GPU lock (concurrent
builds peak at ~8 GB each and overflowed the card in 26079917), and only the anchor language plus the current
episode's language sit in the GPU T5 cache instead of all 880 entries (1.8 GB).
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time

from common import (anchor, anchor_errors, arm_panels, calibration_input, dump_pickle, guard, install, load_pickle,
                    merged_cache, panel_rows, prep, read, results, runtime, same_anchor, save, torch_save)

SETTLING = "completed during fixed settling steps"


def arm_rows(p, arm):
    return [row for name in arm_panels(p, arm) for row in panel_rows(p, name)]


def task_order(rows):
    return sorted(rows, key=lambda r: (r["suite"] != "libero_10", int(r["index"])))


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


def gpu_lock():
    """Exclusive lock of this worker's GPU in node-local /dev/shm; the kernel drops it if the worker dies."""
    path = f"/dev/shm/cosmos_lavq_{os.environ['PBS_JOBID']}_{os.environ['CUDA_VISIBLE_DEVICES']}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def t5_cache(p, out):
    """(merged T5 cache on CPU, anchor language, one-entry cache file the runtime is built on).

    The official loader moves every entry of its file to the GPU under a 30 s file lock; with 20 workers most time out
    and silently keep an empty cache. Each worker reads the merged file itself instead (it is never written here)."""
    full = load_pickle(merged_cache(p))
    language = calibration_input(p, p["calibration"]["anchor_case"])[1]["description"]
    path = out / "t5_anchor.pkl"
    dump_pickle({language: full[language]}, path)
    return full, language, path


def stage_language(rt, full, keep, row):
    """The episode's language, resolved as CosmosRuntime.episode does, moved to the GPU beside the anchor entry only
    (same tensor and same .to(device) as the official loader)."""
    language = rt._suite(str(row["suite"])).get_task(int(row["task_id"])).language
    cache = rt.cosmos_utils.t5_text_embeddings_cache
    for key in [k for k in cache if k != keep]:
        del cache[key]
    cache[language] = full[language].to(cache[keep].device)
    return language


def walltime_seconds(text):
    h, m, s = (int(x) for x in text.split(":"))
    return 3600 * h + 60 * m + s


def startup(p, rt, arm, root):
    import torch
    bf16 = anchor(p, rt)
    reference = torch.load(prep(p) / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    if not same_anchor(bf16, reference):
        raise RuntimeError(f"BF16 anchor differs from the prep anchor: {anchor_errors(bf16, reference)}")
    installed = install(p, rt, arm)
    mine = anchor(p, rt)
    path = root / "anchor.pt"
    if not path.exists():
        torch_save(mine, path)
    first = torch.load(path, map_location="cpu", weights_only=True)
    if not same_anchor(mine, first):
        raise RuntimeError(f"{arm} anchor differs from the arm's first worker: {anchor_errors(mine, first)}")
    return dict(bf16_anchor_bitwise=True, installed_layers=installed, arm_anchor_bitwise=True,
                anchor_vs_bf16=anchor_errors(mine, reference))


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    arm, worker = args.arm, args.worker
    if arm not in p["arms"]:
        raise ValueError(arm)
    order = task_order(arm_rows(p, arm))
    root = results(p) / "loops" / arm
    out = root / f"worker_{worker}"
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    deadline = float(os.environ["COSMOS_JOB_START_EPOCH"]) + walltime_seconds(p["loop"]["walltime"]) - 900
    started = time.monotonic()
    full, keep, small = t5_cache(p, out)
    lock = gpu_lock()
    locked = time.monotonic()
    rt = runtime(p, out / "runtime", small)
    rt.soft_deadline_epoch = deadline
    check = startup(p, rt, arm, root)
    import torch
    peak = torch.cuda.max_memory_reserved() / 2**30  # since the last query's reset: the arm anchor query
    torch.cuda.empty_cache()
    os.close(lock)  # releases the GPU lock
    save(out / "startup.json", dict(check, arm=arm, worker=worker, pbs_jobid=os.environ["PBS_JOBID"],
                                    host=os.uname().nodename, lock_wait_seconds=locked - started,
                                    load_and_check_seconds=time.monotonic() - locked,
                                    anchor_query_peak_reserved_gib=peak,
                                    reserved_after_startup_gib=torch.cuda.memory_reserved() / 2**30))
    completed, position, stopped = [], 0, False
    try:
        while time.time() < deadline - 600:
            row, position = claim(root, order, position, worker)
            if row is None:
                break
            index = int(row["index"])
            save(out / "progress.json", dict(phase="episode", index=index, completed=len(completed),
                                             updated_epoch=time.time()))
            stage_language(rt, full, keep, row)
            record = rt.episode(row, index)
            error = record.get("error") or ""
            if record["termination"] == "soft_deadline":
                shutil.rmtree(root / "claims" / str(index))
                stopped = True
                break
            invalid = SETTLING in error
            if not invalid and not (record["complete"] and record["closed_loop_complete"]):
                shutil.rmtree(root / "claims" / str(index))  # free the slot for a retry by another worker
                raise RuntimeError(f"{arm} episode {index} incomplete: {record['termination']} {error}")
            record.update(arm=arm, invalid=invalid, success=bool(record["success"]) and not invalid, worker=worker,
                          host=os.uname().nodename, pbs_jobid=os.environ["PBS_JOBID"])
            save(root / "episodes" / f"{index}.json", record)
            completed.append(index)
            print(f"EPISODE arm={arm} worker={worker} index={index} success={record['success']} "
                  f"steps={record['steps']} seconds={record['episode_seconds']:.0f}" + (" invalid" if invalid else ""),
                  flush=True)
    finally:
        rt.close()
    save(out / "summary.json", dict(protocol=p["protocol"], arm=arm, pbs_jobid=os.environ["PBS_JOBID"],
                                    worker=worker, host=os.uname().nodename, episodes=completed,
                                    stopped_at_deadline=stopped or time.time() >= deadline - 600))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
