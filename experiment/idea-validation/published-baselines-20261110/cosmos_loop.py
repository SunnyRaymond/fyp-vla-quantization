"""LIBERO-Plus panel A closed loop of one published baseline on Cosmos-Policy (eager fake quantization, decoded weights).

Startup (under an exclusive per-GPU lock, since concurrent builds overflow a 40 GB card): the BF16 model must reproduce
the A16 BF16 anchor bitwise; then the arm's weights, forwards and (QuaRot/SpinQuant only) KV quantization go in, every
block-level Linear must be hit by the anchor query, and later workers must reproduce the first worker's arm anchor
bitwise. --smoke stops there.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time

from cos_common import (anchor, anchor_errors, calibration_input, dump_pickle, guard, load_pickle, read, runtime,
                        same_anchor, save, torch_save)
from loopcommon import claim, walltime_seconds

SETTLING = "completed during fixed settling steps"


def gpu_lock():
    path = f"/dev/shm/pub_{os.environ['PBS_JOBID']}_{os.environ['CUDA_VISIBLE_DEVICES']}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def t5_cache(m, out):
    """Merged T5 cache on CPU, the anchor language, and a one-entry cache file for building the runtime."""
    full = load_pickle(Path(m["a16_prep"]) / "base_extra_cache.pkl")
    language = calibration_input(m, m["calibration"]["anchor_case"])[1]["description"]
    path = out / "t5_anchor.pkl"
    dump_pickle({language: full[language]}, path)
    return full, language, path


def stage_language(rt, full, keep, row):
    language = rt._suite(str(row["suite"])).get_task(int(row["task_id"])).language
    cache = rt.cosmos_utils.t5_text_embeddings_cache
    for key in [k for k in cache if k != keep]:
        del cache[key]
    cache[language] = full[language].to(cache[keep].device)


def startup(p, m, rt, arm, root):
    import torch
    import kvsites
    import methods as mt
    prep = Path(m["a16_prep"])
    results = Path(p["remote_root"]) / "results" / "cosmos"
    bf16 = anchor(m, rt)
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    if not same_anchor(bf16, reference):
        raise RuntimeError(f"BF16 anchor differs from the A16 anchor: {anchor_errors(bf16, reference)}")
    bank_dir = results / "banks" / arm
    if not read(bank_dir / "COMPLETE.json").get("complete"):
        raise RuntimeError(f"{arm} is not fitted")
    index_of = {r["module"]: r["index"] for r in read(prep / "modules.json")}
    device = torch.device("cuda")
    arch = read(results / "arch.json")
    rotations = None
    if arm == "spinquant":
        rotations = {k: v.to(device, torch.float32) for k, v in
                     torch.load(results / "rotations.pt", map_location="cpu", weights_only=True).items()}
    seen = mt.install_arm(dict(rt.target_linears), index_of, bank_dir, arm, m, arch, device, rotations)
    hits = 0
    kv = None
    if arm in mt.ROTATED:
        kv = kvsites.KV(arm, arch, rotations)
        kvsites.install_cosmos(rt.model, kv, m["attentions"])
    mine = anchor(m, rt)
    if kv is not None:
        hits = kv.counts["kv"]
    if len(seen) != m["targets"] or (kv is not None and hits == 0):
        raise RuntimeError(f"Coverage failed: {len(seen)} Linears, KV hits {hits}")
    path = root / "anchor.pt"
    if not path.exists():
        torch_save(mine, path)
    first = torch.load(path, map_location="cpu", weights_only=True)
    if not same_anchor(mine, first):
        raise RuntimeError(f"{arm} anchor differs from the arm's first worker: {anchor_errors(mine, first)}")
    return dict(protocol=p["protocol"], arm=arm, bf16_anchor_bitwise=True, linears_seen=len(seen),
                kv_hits_per_query=hits, anchor_vs_bf16=anchor_errors(mine, reference),
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    m, arm, worker = p["models"]["cosmos"], args.arm, args.worker
    if arm not in p["arms"]:
        raise ValueError(arm)
    root = Path(p["remote_root"]) / "results" / "cosmos" / "loops" / arm
    out = root / ("smoke" if args.smoke else f"worker_{worker}")
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    deadline = float(os.environ["COSMOS_JOB_START_EPOCH"]) + walltime_seconds(m["jobs"]["loop"]["walltime"]) - 900
    full, keep, small = t5_cache(m, out)
    lock = gpu_lock()
    rt = runtime(m, out / "runtime", small)
    rt.soft_deadline_epoch = deadline
    check = startup(p, m, rt, arm, root)
    import torch
    torch.cuda.empty_cache()
    os.close(lock)
    save(out / "startup.json", dict(check, worker=worker, pbs_jobid=os.environ["PBS_JOBID"], host=os.uname().nodename))
    if args.smoke:
        rt.close()
        print(f"SMOKE cosmos {arm} {check['anchor_vs_bf16']} peak={check['peak_reserved_gib']:.1f}GiB", flush=True)
        save(out / "progress.json", dict(phase="smoke_passed", updated_epoch=time.time()))
        return
    rows = read(m["panel"]["manifest"])["variants"]
    if sorted(int(r["index"]) for r in rows) != list(range(280)):
        raise RuntimeError("Panel A manifest does not hold slot indices 0-279")
    order = sorted(rows, key=lambda r: (r["suite"] != "libero_10", int(r["index"])))
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
                shutil.rmtree(root / "claims" / str(index))
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
    save(out / "summary.json", dict(protocol=p["protocol"], arm=arm, pbs_jobid=os.environ["PBS_JOBID"], worker=worker,
                                    episodes=completed, stopped_at_deadline=stopped))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
