"""T5 cache for this job on one GPU: panel-C instructions and the case-2000 anchor prompt encoded twice by the full
BF16 model (bitwise repeat required), merged with the panel-B cache of job 26070795 (the W3 arms run panel B here).

Checks: the full-T5 case-2000 anchor equals the A16 bf16_anchor.pt bitwise; every encoded prompt also present in the
A16 or panel-B cache equals it bitwise; t5_recheck panel-B-only prompts are re-encoded and must equal the panel-B cache
bitwise (same model, same container as that job); every panel-B prompt must be in the panel-B cache.
Writes prep/T5_COMPLETE.json last, or prep/T5_FAILED.json so waiting workers stop at once.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

from common import bf16_runtime, capture, guard, max_abs, read, results, save, torch_save
import panel


def prompts_for(runtime, rows):
    suites, prompts = {}, {}
    for row in rows:
        if row["suite"] not in suites:
            suites[row["suite"]] = runtime.benchmark.get_benchmark_dict()[row["suite"]]()
        task = suites[row["suite"]].get_task(int(row["task_id"]))
        if row.get("task_name") and task.name != row["task_name"]:
            raise RuntimeError(f"Manifest task identity changed: {row['variant_id']} {task.name}")
        prompts[runtime.ev.DEFAULT_PROMPT.format(task=task.language)] = None
    return list(prompts)


def same(a, b):
    import torch
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def encode_twice(runtime, prompt):
    first = runtime.model.encode_prompt(prompt)
    second = runtime.model.encode_prompt(prompt)
    if not same(first, second):
        raise RuntimeError(f"T5 encoding is not bitwise repeatable for {prompt!r}")
    return tuple(t.detach().cpu().clone() for t in first)


def build(p, pa, out):
    import torch
    rows_c, rows_b = panel.load(p, "C"), panel.load(p, "B")
    started = time.monotonic()
    runtime = bf16_runtime(out / "runtime_t5")
    prompts_c, prompts_b = prompts_for(runtime, rows_c), prompts_for(runtime, rows_b)
    action, future = capture(pa, runtime)
    anchor_prompt = runtime.ev.DEFAULT_PROMPT.format(task=runtime.description)
    reference = torch.load(results(pa) / "prep" / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    anchor = dict(action=max_abs(action, reference["action"]), future=max_abs(future, reference["future"]))
    if anchor["action"] != 0 or anchor["future"] != 0:
        raise RuntimeError(f"Full-T5 BF16 anchor differs from the A16 anchor: {anchor}")
    a16 = torch.load(results(pa) / "prep" / "t5_cache.pt", map_location="cpu", weights_only=True)
    cache_b = torch.load(p["panel_b"]["t5_cache"], map_location="cpu", weights_only=True)
    missing = [q for q in prompts_b + [anchor_prompt] if q not in cache_b]
    if missing:
        raise RuntimeError(f"{len(missing)} panel-B prompts are not in the panel-B cache, e.g. {missing[0]!r}")
    cache, shared_a16, shared_b = {}, 0, 0
    with torch.inference_mode():
        for prompt in prompts_c + [anchor_prompt]:
            if prompt in cache:
                continue
            cache[prompt] = encode_twice(runtime, prompt)
            for old, name in ((a16, "A16"), (cache_b, "panel-B")):
                if prompt in old and not same(cache[prompt], old[prompt]):
                    raise RuntimeError(f"T5 encoding differs from the {name} cache for {prompt!r}")
            shared_a16 += prompt in a16
            shared_b += prompt in cache_b
        recheck = [q for q in prompts_b if q not in cache][:p["panel_b"]["t5_recheck"]]
        for prompt in recheck:
            if not same(encode_twice(runtime, prompt), cache_b[prompt]):
                raise RuntimeError(f"Re-encoded panel-B prompt differs from the panel-B cache: {prompt!r}")
    copied = 0
    for prompt in prompts_b:
        if prompt not in cache:
            cache[prompt] = cache_b[prompt]
            copied += 1
    torch_save(cache, out / "t5_cache.pt")
    torch_save(reference, out / "bf16_anchor.pt")
    save(out / "t5_cache.json", dict(protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"], prompts=len(cache),
                                     panel_c_prompts=len(prompts_c), panel_b_prompts=len(prompts_b),
                                     encoded=len(prompts_c) + (anchor_prompt not in prompts_c),
                                     shared_with_a16=shared_a16, shared_with_panel_b=shared_b,
                                     copied_from_panel_b=copied, panel_b_rechecked=len(recheck),
                                     anchor_vs_a16=anchor, seconds=time.monotonic() - started,
                                     embedding_shape=list(next(iter(cache.values()))[0].shape)))
    save(out / "T5_COMPLETE.json", dict(complete=True, pbs_jobid=os.environ["PBS_JOBID"], prompts=len(cache)))
    print(f"T5_CACHE prompts={len(cache)} panel_c_prompts={len(prompts_c)} copied_from_panel_b={copied} "
          f"rechecked={len(recheck)} shared_with_a16={shared_a16} shared_with_panel_b={shared_b}", flush=True)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    pa = read(p["a16"]["protocol"])
    out = results(p) / "prep"
    if (out / "T5_COMPLETE.json").exists():
        print("T5_CACHE already complete", flush=True)
        return
    (out / "T5_FAILED.json").unlink(missing_ok=True)
    try:
        build(p, pa, out)
    except BaseException:
        save(out / "T5_FAILED.json", dict(pbs_jobid=os.environ.get("PBS_JOBID"), error=traceback.format_exc()[-4000:]))
        raise


if __name__ == "__main__":
    main()
