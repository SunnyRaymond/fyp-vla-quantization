"""Preparation shard: C80 Hessians + BF16 master weights for this shard's layers; shard 0 also builds the T5 cache.

Layers are split across shards by Hessian bytes (greedy, largest first); every shard runs the 80 C80 queries once
per memory pass on the full BF16 model and saves mean x x^T per layer.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from common import capture, guard, max_abs, read, results, save, torch_save, bf16_runtime


def shard_layers(modules, shards):
    loads, owner = [0] * shards, {}
    for name in sorted(modules, key=lambda n: (-modules[n].in_features, n)):
        k = min(range(shards), key=lambda s: (loads[s], s))
        owner[name] = k
        loads[k] += modules[name].in_features ** 2
    return owner


def build_t5_cache(p, runtime, out):
    """Encode every panel instruction (and the anchor prompt) twice; save the cache and the full-T5 anchor."""
    import torch
    rows = read(p["panel"]["manifest"])["variants"]
    suites, prompts = {}, {}
    for row in rows:
        if row["suite"] not in suites:
            suites[row["suite"]] = runtime.benchmark.get_benchmark_dict()[row["suite"]]()
        language = suites[row["suite"]].get_task(int(row["task_id"])).language
        prompts[runtime.ev.DEFAULT_PROMPT.format(task=language)] = None
    action, future = capture(p, runtime)
    prompts[runtime.ev.DEFAULT_PROMPT.format(task=runtime.description)] = None
    cache = {}
    with torch.inference_mode():
        for prompt in prompts:
            first = runtime.model.encode_prompt(prompt)
            second = runtime.model.encode_prompt(prompt)
            if not all(torch.equal(a, b) for a, b in zip(first, second)):
                raise RuntimeError(f"T5 encoding is not bitwise repeatable for {prompt!r}")
            cache[prompt] = tuple(t.detach().cpu().clone() for t in first)
    torch_save(cache, out / "t5_cache.pt")
    torch_save(dict(action=action, future=future, case_id=p["panel"]["anchor_case"]), out / "bf16_anchor.pt")
    save(out / "t5_cache.json", dict(protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"],
                                     prompts=len(cache), panel_rows=len(rows),
                                     embedding_shape=list(next(iter(cache.values()))[0].shape)))
    return len(cache)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import calibration_source
    root = results(p) / "prep"
    status = root / f"shard_{args.shard}.json"
    started = time.monotonic()
    runtime = bf16_runtime(root / f"runtime_{args.shard}")
    modules = runtime.q.modules
    names = list(modules)
    owner = shard_layers(modules, args.shards)
    mine = [n for n in names if owner[n] == args.shard]
    index = {n: i for i, n in enumerate(names)}
    if args.shard == 0:
        save(root / "modules.json", [dict(index=i, module=n, shape=list(modules[n].weight.shape)) for i, n in
                                     enumerate(names)])
        if not (root / "t5_cache.pt").exists():
            prompts = build_t5_cache(p, runtime, root)
            print(f"T5_CACHE prompts={prompts}", flush=True)

    for name in mine:
        path = root / "masters" / f"{index[name]:04d}.pt"
        if not path.exists():
            torch_save(dict(module=name, weight=modules[name].weight.detach().cpu().clone()), path)
    todo = [n for n in mine if not (root / "hessians" / f"{index[n]:04d}.pt").exists()]

    sums, rows = {}, {}
    for name, mod in modules.items():
        def forward(x, _name=name, _mod=mod):
            if _name in sums:
                flat = x.detach().float().reshape(-1, _mod.in_features)
                sums[_name].addmm_(flat.t(), flat)
                rows[_name] += flat.shape[0]
            return F.linear(x, _mod.weight, _mod.bias)
        mod.forward = forward

    plan_path = Path(p["panel"]["calibration_plan"])
    plan = {r["case_id"]: r for r in read(plan_path)["inputs"]}
    first, last = p["hessian"]["c80_cases"]
    cases = list(range(first, last + 1))

    def query(case):
        observation, metadata = calibration_source.load_observation(np, plan_path.parent, plan[case])
        runtime.description = metadata["description"]
        runtime.infer(runtime.datum(observation), plan[case]["sampler_seeds"][0], measure=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    query(cases[0])
    torch.cuda.synchronize()
    budget = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.max_memory_reserved()
              - p["hessian"]["headroom_gb"] * 2 ** 30)
    passes, current, used = [], [], 0
    for name in todo:
        size = 4 * modules[name].in_features ** 2
        if current and used + size > budget:
            passes.append(current)
            current, used = [], 0
        current.append(name)
        used += size
    if current:
        passes.append(current)
    save(status, dict(phase="hessian", layers=len(mine), todo=len(todo), passes=len(passes),
                      budget_gb=budget / 2 ** 30, pbs_jobid=os.environ["PBS_JOBID"]))
    for number, group in enumerate(passes):
        sums.clear()
        rows.clear()
        with torch.inference_mode():
            for name in group:
                sums[name] = torch.zeros(modules[name].in_features, modules[name].in_features, device="cuda")
                rows[name] = 0
            for count, case in enumerate(cases):
                query(case)
                if count % 20 == 19:
                    print(f"HESSIAN shard={args.shard} pass={number + 1}/{len(passes)} contexts={count + 1}",
                          flush=True)
        for name in group:
            if rows[name] == 0:
                raise RuntimeError(f"Layer never called during C80: {name}")
            torch_save(dict(module=name, rows=rows[name], hessian=(sums[name] / rows[name]).cpu()),
                       root / "hessians" / f"{index[name]:04d}.pt")
        sums.clear()
        torch.cuda.empty_cache()
    save(status, dict(phase="complete", complete=True, layers=len(mine), passes=len(passes),
                      seconds=time.monotonic() - started, pbs_jobid=os.environ["PBS_JOBID"]))
    print(f"PREP_SHARD_COMPLETE shard={args.shard} layers={len(mine)}", flush=True)


if __name__ == "__main__":
    main()
