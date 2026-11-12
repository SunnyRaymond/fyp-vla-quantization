"""Fast-WAM T5 prompts of the 40 standard-LIBERO tasks (GPU, run once before the loop): t5prep840.py --protocol P

The A16 cache (fastwam-a16-baselines-20261028/results/prep/t5_cache.pt) only holds LIBERO-Plus variant prompts and the
calibration prompts. This encodes DEFAULT_PROMPT.format(task=language) of every task with the model's own text encoder,
twice, and requires the two passes to be bitwise equal (the rule of the A16 prep). Prompts already in the A16 cache keep
their A100 entry; the A100 vs this-GPU difference of those entries is recorded. Output: results/fastwam/t5_libero40.pt
(only the prompts the A16 cache lacks) and t5_libero40.json.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import l840
from fw_common import bf16_runtime, guard, read, save, torch_save


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    import torch
    P = l840.read(args.protocol)
    p = read(P["models"]["fastwam"]["laq_protocol"])
    m = p["models"]["fastwam"]
    res = l840.results(P, "fastwam")
    tag = l840.owner_tag()
    runtime = bf16_runtime(res / "t5prep" / f"{tag['job']}.{tag['restart']}")
    rows = l840.manifest_rows(P, "fastwam")
    prompts = sorted({runtime.ev.DEFAULT_PROMPT.format(task=r["language"]) for r in rows})
    if len(prompts) != 40:
        raise RuntimeError(f"Expected 40 prompts, got {len(prompts)}")
    a100 = torch.load(Path(m["a16_prep"]) / "t5_cache.pt", map_location="cpu", weights_only=True)
    ref = next(iter(a100.values()))
    new, overlap = {}, {}
    with torch.inference_mode():
        for prompt in prompts:
            first = runtime.model.encode_prompt(prompt)
            second = runtime.model.encode_prompt(prompt)
            if not all(torch.equal(a, b) for a, b in zip(first, second)):
                raise RuntimeError(f"T5 encoding is not bitwise repeatable for {prompt!r}")
            entry = tuple(t.detach().cpu().clone() for t in first)
            if [(t.shape, t.dtype) for t in entry] != [(t.shape, t.dtype) for t in ref]:
                raise RuntimeError(f"Entry layout differs from the A16 cache: {[(t.shape, t.dtype) for t in entry]}")
            if prompt in a100:
                emb, old = entry[0].float(), a100[prompt][0].float()
                overlap[prompt] = dict(max_abs=float((emb - old).abs().max()),
                                       rel_rmse=float((emb - old).pow(2).mean().sqrt() / old.pow(2).mean().sqrt()),
                                       mask_equal=bool(torch.equal(entry[1], a100[prompt][1])))
            else:
                new[prompt] = entry
    torch_save(new, res / "t5_libero40.pt")
    info = dict(protocol=P["protocol"], job=tag, gpu=torch.cuda.get_device_name(0), prompts=len(prompts),
                encoded_new=len(new), kept_from_a16_cache=len(overlap), a100_vs_this_gpu_on_kept=overlap,
                shapes=[list(t.shape) for t in ref], dtypes=[str(t.dtype) for t in ref])
    save(res / "t5_libero40.json", info)
    print(f"T5PREP new={len(new)} kept={len(overlap)} drift={overlap}", flush=True)


if __name__ == "__main__":
    main()
