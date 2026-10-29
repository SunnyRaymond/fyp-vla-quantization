"""Quantize this shard's layers with one method/bit width from saved masters + C80 Hessians; --finalize checks all 614.

Each layer's output is the dense BF16 weight the closed loop installs, plus its stored-bit count and the
calibration-distribution error tr(dW H dW^T) / tr(W H W^T).
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
import time

from common import TARGETS, guard, read, results, save, torch_save

HERE = Path(__file__).resolve().parent
SEED = 20261028


def arm_name(p, method, bits):
    found = [a for a, c in p["arms"].items() if c["method"] == method and c["bits"] == bits]
    if len(found) != 1:
        raise ValueError(f"No unique arm for {method} W{bits}")
    return found[0]


def shard_layers(modules, shards):
    loads, owner = [0] * shards, {}
    for row in sorted(modules, key=lambda r: -(r["shape"][0] * r["shape"][1] + r["shape"][1] ** 2)):
        k = min(range(shards), key=lambda s: (loads[s], s))
        owner[row["index"]] = k
        loads[k] += row["shape"][0] * row["shape"][1] + row["shape"][1] ** 2
    return owner


def quantize(p, method, bits, shard, shards, reverse=False):
    import torch
    sys.path.insert(0, str(HERE / "methods" / method))
    impl = importlib.import_module("quant")
    prep = results(p) / "prep"
    arm = arm_name(p, method, bits)
    out = results(p) / "quantized" / arm
    modules = read(prep / "modules.json")
    owner = shard_layers(modules, shards)
    mine = [r for r in modules if owner[r["index"]] == shard]
    if reverse:  # a second job walks from the last layer back and meets an index-order job in the middle
        mine.reverse()
    started = time.monotonic()
    for done, row in enumerate(mine):
        i, name = row["index"], row["module"]
        if (out / "stats" / f"{i:04d}.json").exists():
            continue
        master = torch.load(prep / "masters" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        saved = torch.load(prep / "hessians" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        if master["module"] != name or saved["module"] != name:
            raise RuntimeError(f"Prep identity differs at {i}: {name}")
        w = master["weight"].to("cuda", torch.float32)
        h = saved["hessian"].to("cuda", torch.float32)
        begin = time.monotonic()
        result = impl.quantize_layer(w, h, bits, seed=SEED + i)
        dq = result["dequant"].to("cuda", torch.float32)
        if tuple(dq.shape) != tuple(w.shape) or not bool(torch.isfinite(dq).all()):
            raise RuntimeError(f"{method} W{bits} produced an invalid weight at {name}")
        d = dq - w
        proxy = float(((d @ h) * d).sum() / ((w @ h) * w).sum().clamp_min(1e-30))
        stored = dq.to(torch.bfloat16).cpu()
        torch_save(dict(module=name, weight=stored), out / "weights" / f"{i:04d}.pt")
        save(out / "stats" / f"{i:04d}.json", dict(
            index=i, module=name, shape=list(w.shape), bits_total=result["bits_total"],
            bpw=result["bits_total"] / w.numel(), proxy_nmse=proxy, seconds=time.monotonic() - begin,
            config=result["config"]))
        del w, h, dq, d, result
        torch.cuda.empty_cache()
        if done % 25 == 0:
            print(f"QUANT {arm} shard={shard} layer={done + 1}/{len(mine)} proxy={proxy:.5f} "
                  f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    save(out / f"shard_{shard}_of_{shards}.json", dict(complete=True, layers=len(mine),
                                                       seconds=time.monotonic() - started,
                                                       pbs_jobid=os.environ["PBS_JOBID"]))


def finalize(p, method, bits):
    arm = arm_name(p, method, bits)
    out = results(p) / "quantized" / arm
    stats = [read(out / "stats" / f"{i:04d}.json") for i in range(TARGETS)]
    weights = sum(s["shape"][0] * s["shape"][1] for s in stats)
    summary = dict(protocol=p["protocol"], arm=arm, method=method, bits=bits, complete=True,
                   pbs_jobid=os.environ["PBS_JOBID"], target_weights=weights,
                   bpw=sum(s["bits_total"] for s in stats) / weights,
                   proxy_nmse_weighted=sum(s["proxy_nmse"] * s["shape"][0] * s["shape"][1] for s in stats) / weights,
                   proxy_nmse_max=max(s["proxy_nmse"] for s in stats),
                   worst_layers=sorted(((s["proxy_nmse"], s["module"]) for s in stats), reverse=True)[:5],
                   quantize_seconds=sum(s["seconds"] for s in stats), config_example=stats[0]["config"])
    save(out / "COMPLETE.json", summary)
    print(f"QUANT_COMPLETE {arm} bpw={summary['bpw']:.4f} proxy={summary['proxy_nmse_weighted']:.5f}", flush=True)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    if args.finalize:
        finalize(p, args.method, args.bits)
    else:
        import torch
        # FP32 matmuls/Cholesky without TF32; deterministic-algorithm mode stays off so ported ops are not refused.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        quantize(p, args.method, args.bits, args.shard, args.shards, args.reverse)


if __name__ == "__main__":
    main()
