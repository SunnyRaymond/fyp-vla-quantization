"""T5 cache for panel B on one GPU: every panel-B instruction plus the case-2000 anchor prompt, encoded twice by the
full BF16 model (bitwise repeat required). Prompts shared with the A16 cache must match it bitwise, and the full-T5
case-2000 anchor must equal the A16 bf16_anchor.pt bitwise; the A16 anchor is then copied as this panel's reference.
Writes prep/T5_COMPLETE.json last, or prep/T5_FAILED.json so waiting workers stop at once.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

from common import bf16_runtime, capture, guard, max_abs, read, results, save, torch_save


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


def build(p, pa, out):
    import torch
    rows = read(results(p) / "panel" / "manifest.json")["variants"]
    started = time.monotonic()
    runtime = bf16_runtime(out / "runtime_t5")
    prompts = prompts_for(runtime, rows)
    action, future = capture(pa, runtime)
    anchor_prompt = runtime.ev.DEFAULT_PROMPT.format(task=runtime.description)
    reference = torch.load(results(pa) / "prep" / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    anchor = dict(action=max_abs(action, reference["action"]), future=max_abs(future, reference["future"]))
    if anchor["action"] != 0 or anchor["future"] != 0:
        raise RuntimeError(f"Full-T5 BF16 anchor differs from the A16 anchor: {anchor}")
    old = torch.load(results(pa) / "prep" / "t5_cache.pt", map_location="cpu", weights_only=True)
    cache, shared = {}, 0
    with torch.inference_mode():
        for prompt in prompts + [anchor_prompt]:
            if prompt in cache:
                continue
            first = runtime.model.encode_prompt(prompt)
            second = runtime.model.encode_prompt(prompt)
            if not all(torch.equal(a, b) for a, b in zip(first, second)):
                raise RuntimeError(f"T5 encoding is not bitwise repeatable for {prompt!r}")
            cache[prompt] = tuple(t.detach().cpu().clone() for t in first)
            if prompt in old:
                if not all(torch.equal(a, b) for a, b in zip(cache[prompt], old[prompt])):
                    raise RuntimeError(f"T5 encoding differs from the A16 cache for {prompt!r}")
                shared += 1
    torch_save(cache, out / "t5_cache.pt")
    torch_save(reference, out / "bf16_anchor.pt")
    save(out / "t5_cache.json", dict(protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"], prompts=len(cache),
                                     panel_rows=len(rows), panel_prompts=len(prompts), shared_with_a16=shared,
                                     anchor_vs_a16=anchor, seconds=time.monotonic() - started,
                                     embedding_shape=list(next(iter(cache.values()))[0].shape)))
    save(out / "T5_COMPLETE.json", dict(complete=True, pbs_jobid=os.environ["PBS_JOBID"], prompts=len(cache)))
    print(f"T5_CACHE prompts={len(cache)} panel_prompts={len(prompts)} shared_with_a16={shared}", flush=True)


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
