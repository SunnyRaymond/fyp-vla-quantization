"""Preparation on one GPU: target manifest, BF16 masters, C80 Hessians, the BF16 anchor, and the panel-A reuse replay.

1. modules.json + masters/<i>.pt for the 280 target Linears (runtime order = sorted module names).
2. BF16 anchor (case 2000, seed index 0) twice; the two must be bitwise equal; saved as bf16_anchor.pt.
3. Hessians mean x x^T per target over every call of the 80 C80 queries (forward pre-hooks, FP32 accumulators on the
   GPU, as many layers per pass as memory allows).
4. Replay: the first get_action of panel-A episodes reuse.BF16_A.replay_indices must reproduce the stored
   first_action_chunk of the old BF16 run bitwise -> reuse_check.json (the loop job runs BF16 panel A if it fails).
COMPLETE.json is written last; every step skips what is already on disk.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from common import (anchor, base_extra_cache, calibration_query, dump_pickle, guard, modules_rows, panel_rows, read,
                    results, runtime, same_anchor, save, torch_save)


class _FirstQuery(BaseException):
    """Stops CosmosRuntime.episode right after its first policy query (passes through its `except Exception`)."""


def first_chunk(rt, row):
    original, box = rt.query, {}

    def query(*args, **kwargs):
        box["actions"] = original(*args, **kwargs)["actions"]
        raise _FirstQuery

    rt.query = query
    try:
        rt.episode(row, int(row["index"]))
    except _FirstQuery:
        pass
    finally:
        rt.query = original
        if rt.env is not None:
            rt.env.close()
            rt.env = None
    if "actions" not in box:
        raise RuntimeError(f"Episode {row['index']} ended before its first policy query")
    return box["actions"]


def hessians(p, rt, rows, out):
    import torch
    modules = rt.target_linears
    todo = [r for r in rows if not (out / "hessians" / f"{r['index']:04d}.pt").exists()]
    if not todo:
        return 0
    sums, counts = {}, {}

    def hook(name, in_features):
        def pre(module, args):
            if name in sums:
                x = args[0].detach().float().reshape(-1, in_features)
                sums[name].addmm_(x.t(), x)
                counts[name] += x.shape[0]
        return pre

    handles = [m.register_forward_pre_hook(hook(n, m.in_features)) for n, m in modules.items()]
    first, last = p["calibration"]["cases"]
    cases = list(range(first, last + 1))
    try:
        calibration_query(p, rt, cases[0])
        torch.cuda.synchronize()
        budget = torch.cuda.mem_get_info()[0] - p["calibration"]["headroom_gb"] * 2 ** 30
        passes, current, used = [], [], 0
        for r in todo:
            size = 4 * r["shape"][1] ** 2
            if current and used + size > budget:
                passes.append(current)
                current, used = [], 0
            current.append(r)
            used += size
        passes.append(current)
        print(f"HESSIAN layers={len(todo)} passes={len(passes)} budget_gb={budget / 2 ** 30:.1f}", flush=True)
        for number, group in enumerate(passes):
            sums.clear()
            counts.clear()
            with torch.inference_mode():
                for r in group:
                    sums[r["module"]] = torch.zeros(r["shape"][1], r["shape"][1], device="cuda")
                    counts[r["module"]] = 0
                for k, case in enumerate(cases):
                    calibration_query(p, rt, case)
                    if k % 20 == 19:
                        print(f"HESSIAN pass={number + 1}/{len(passes)} contexts={k + 1}", flush=True)
            for r in group:
                name = r["module"]
                if counts[name] == 0:
                    raise RuntimeError(f"Target never called during C80: {name}")
                torch_save(dict(module=name, rows=counts[name], hessian=(sums[name] / counts[name]).cpu()),
                           out / "hessians" / f"{r['index']:04d}.pt")
            sums.clear()
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()
    return len(todo)


def replay(p, rt):
    reuse = p["reuse"]["BF16_A"]
    rows = {int(r["index"]): r for r in panel_rows(p, "A")}
    checks = []
    for index in reuse["replay_indices"]:
        old = read(Path(reuse["episodes"]) / f"{index:03d}.json")
        if old["variant_id"] != rows[index]["variant_id"] or old["arm"] != "bf16":
            raise RuntimeError(f"Old BF16 record {index} does not match panel A")
        actions = first_chunk(rt, rows[index])
        new = actions.tolist()
        diff = max(abs(a - b) for x, y in zip(new, old["first_action_chunk"]) for a, b in zip(x, y))
        checks.append(dict(index=index, variant_id=old["variant_id"], equal=new == old["first_action_chunk"],
                           max_abs=diff))
        print(f"REPLAY index={index} equal={new == old['first_action_chunk']} max_abs={diff:.3g}", flush=True)
    return checks


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    import torch
    started = time.monotonic()
    out = results(p) / "prep"
    cache = out / "base_extra_cache.pkl"  # C80 descriptions + panel-A instructions, no new encodings
    if not cache.exists():
        dump_pickle(base_extra_cache(p)[0], cache)
    rt = runtime(p, out / "runtime", cache)
    rows = modules_rows(rt)
    save(out / "modules.json", rows)
    for r in rows:
        path = out / "masters" / f"{r['index']:04d}.pt"
        if not path.exists():
            torch_save(dict(module=r["module"], weight=rt.target_linears[r["module"]].weight.detach().cpu().clone()),
                       path)
    if not (out / "bf16_anchor.pt").exists():
        a, b = anchor(p, rt), anchor(p, rt)
        if not same_anchor(a, b):
            raise RuntimeError("BF16 anchor is not bitwise repeatable")
        torch_save(dict(a, case_id=p["calibration"]["anchor_case"]), out / "bf16_anchor.pt")
        print(f"ANCHOR action[0]={a['action'][0].tolist()}", flush=True)
    collected = hessians(p, rt, rows, out)
    if not (out / "reuse_check.json").exists():
        checks = replay(p, rt)
        save(out / "reuse_check.json", dict(passed=all(c["equal"] for c in checks), checks=checks,
                                            reuse_job=p["reuse"]["BF16_A"]["job"], pbs_jobid=os.environ["PBS_JOBID"]))
    rt.close()
    save(out / "COMPLETE.json", dict(protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"], targets=len(rows),
                                     target_weights=sum(r["shape"][0] * r["shape"][1] for r in rows),
                                     hessians_collected=collected, reuse=read(out / "reuse_check.json")["passed"],
                                     seconds=time.monotonic() - started))
    print(f"PREP_COMPLETE targets={len(rows)} hessians={collected} "
          f"reuse={read(out / 'reuse_check.json')['passed']}", flush=True)


if __name__ == "__main__":
    main()
