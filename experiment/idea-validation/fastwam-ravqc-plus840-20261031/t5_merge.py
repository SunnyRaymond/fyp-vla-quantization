"""T5 cache for the 840-episode loop, no model: the A16 cache (panel-A prompts and the anchor prompt) merged with the
panel-C job's cache (panel-B and C prompts; fastwam-plus-panelc-20261031 encoded C and checked its B copies bitwise).
Every prompt present in both must be bitwise equal. bf16_anchor.pt is copied from the A16 prep. Writes
prep/T5_COMPLETE.json last, or prep/T5_FAILED.json so waiting workers stop at once."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import traceback

from common import read, results, save, torch_save


def same(a, b):
    import torch
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def merge(a16, panelc):
    """Union of two {prompt: tuple of tensors} caches; prompts in both must be bitwise equal."""
    shared = [q for q in panelc if q in a16]
    bad = [q for q in shared if not same(a16[q], panelc[q])]
    if bad:
        raise RuntimeError(f"{len(bad)} prompts differ between the A16 and panel-C caches, e.g. {bad[0]!r}")
    return {**a16, **panelc}, len(shared)


def main():
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
        import torch
        a16 = torch.load(results(pa) / "prep" / "t5_cache.pt", map_location="cpu", weights_only=True)
        panelc = torch.load(p["panel_c"]["t5_cache"], map_location="cpu", weights_only=True)
        cache, shared = merge(a16, panelc)
        anchor = torch.load(results(pa) / "prep" / "bf16_anchor.pt", map_location="cpu", weights_only=True)
        torch_save(cache, out / "t5_cache.pt")
        torch_save(anchor, out / "bf16_anchor.pt")
        save(out / "t5_cache.json", dict(protocol=p["protocol"], pbs_jobid=os.environ.get("PBS_JOBID"), prompts=len(cache),
                                         a16_prompts=len(a16), panel_c_prompts=len(panelc), shared=shared))
        save(out / "T5_COMPLETE.json", dict(complete=True, pbs_jobid=os.environ.get("PBS_JOBID"), prompts=len(cache)))
        print(f"T5_CACHE prompts={len(cache)} a16={len(a16)} panel_c={len(panelc)} shared={shared}", flush=True)
    except BaseException:
        save(out / "T5_FAILED.json", dict(pbs_jobid=os.environ.get("PBS_JOBID"), error=traceback.format_exc()[-4000:]))
        raise


if __name__ == "__main__":
    main()
