"""T5 cache for the loop job: base cache (the one the BF16 panel-A run used) + extra cache (C80 descriptions) + the
panel B/C instructions they lack, encoded on this GPU by CosmosT5TextEncoder exactly as prepare_text_embeddings.py did.

Checks: keys in both base and extra are bitwise equal; every new instruction is encoded twice, bitwise equal;
t5.recheck instructions that an earlier job encoded the same way (in base, not in the official cache) are re-encoded
and compared (reported, not required: B/C entries are new for every arm alike). Writes prep/T5_COMPLETE.json last.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from common import base_extra_cache, dump_pickle, guard, load_pickle, merged_cache, panel_rows, read, results, save


def instructions(p):
    """Task language of every panel row, as CosmosRuntime.episode reads it (LIBERO-Plus benchmark task.language)."""
    from libero.libero import benchmark
    suites, texts = {}, {}
    for name in ("A", "B", "C"):
        for row in panel_rows(p, name):
            if row["suite"] not in suites:
                suites[row["suite"]] = benchmark.get_benchmark_dict()[row["suite"]]()
            task = suites[row["suite"]].get_task(int(row["task_id"]))
            if task.name != row["task_name"]:
                raise RuntimeError(f"Manifest task identity changed: {row['variant_id']} {task.name}")
            texts.setdefault(task.language, name)
    return texts


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    import torch
    started = time.monotonic()
    out = results(p) / "prep"
    cache, base, extra = base_extra_cache(p)
    texts = instructions(p)
    missing_a = [t for t, panel in texts.items() if panel == "A" and t not in base]
    if missing_a:
        raise RuntimeError(f"{len(missing_a)} panel-A instructions are not in the base cache, e.g. {missing_a[:3]}")
    missing = [t for t in texts if t not in cache]
    official = load_pickle(p["t5"]["official_cache"])
    recheck = sorted(k for k in base if k not in official)[:p["t5"]["recheck"]]
    print(f"T5 base={len(base)} extra={len(extra)} merged={len(cache)} panel_instructions={len(texts)} "
          f"missing={len(missing)} recheck={len(recheck)}", flush=True)

    from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder
    assets = read(Path(p["cosmos"]["asset_root"]) / "asset_paths.json")
    encoder = CosmosT5TextEncoder(model_name=assets["text_encoder_snapshot"], device="cuda", local_files_only=True)

    def encode(text):
        with torch.inference_mode():
            first = encoder.encode_prompts([text], max_length=512).detach().cpu()
            second = encoder.encode_prompts([text], max_length=512).detach().cpu()
        if not torch.equal(first, second):
            raise RuntimeError(f"T5 encoding is not bitwise repeatable for {text!r}")
        return first

    rechecked = {}
    for text in recheck:
        new = encode(text)
        old = torch.as_tensor(base[text])
        rechecked[text] = dict(equal=bool(torch.equal(new, old)), max_abs=float((new.float() - old.float()).abs().max()))
    for k, text in enumerate(missing):
        cache[text] = encode(text)
        if k % 20 == 0:
            print(f"T5_ENCODE {k + 1}/{len(missing)}", flush=True)
    path = merged_cache(p)
    dump_pickle(cache, path)
    written = load_pickle(path)
    if any(t not in written for t in texts):
        raise RuntimeError("Written cache does not cover every panel instruction")
    save(out / "T5_COMPLETE.json", dict(
        protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"], entries=len(cache), base=len(base),
        extra=len(extra), panel_instructions=len(texts), encoded=len(missing),
        recheck_equal=sum(v["equal"] for v in rechecked.values()), recheck=rechecked,
        embedding_shape=list(next(iter(cache.values())).shape), seconds=time.monotonic() - started))
    print(f"T5_COMPLETE entries={len(cache)} encoded={len(missing)} "
          f"recheck_equal={sum(v['equal'] for v in rechecked.values())}/{len(rechecked)}", flush=True)


if __name__ == "__main__":
    main()
