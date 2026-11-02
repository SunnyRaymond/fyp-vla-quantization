"""Quantize this shard's target Linears with one method/bit width from the prep masters + C80 Hessians.

Per layer: weights/<i>.pt (dense BF16 the closed loop installs), codes/<i>.pt (the method's native codes, decodable
without requantizing) and stats/<i>.json (stored bits, proxy tr(dW H dW^T) / tr(W H W^T), config). Layers already on
disk are skipped; the shard stops before the soft deadline and exits 3 if layers are left (a resubmission continues).
The last shard to finish writes COMPLETE.json for the arm.
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
SEED = 20261101


def arm_name(p, method, bits):
    found = [a for a, c in p["arms"].items() if c["method"] == method and c["bits"] == bits]
    if len(found) != 1:
        raise ValueError(f"No unique arm for {method} W{bits}")
    return found[0]


def shard_layers(rows, shards):
    """Greedy split by m*n + n^2 (largest first), as in the Fast-WAM A16 quant worker."""
    loads, owner = [0] * shards, {}
    for row in sorted(rows, key=lambda r: -(r["shape"][0] * r["shape"][1] + r["shape"][1] ** 2)):
        k = min(range(shards), key=lambda s: (loads[s], s))
        owner[row["index"]] = k
        loads[k] += row["shape"][0] * row["shape"][1] + row["shape"][1] ** 2
    return owner


def quantize(p, method, bits, shard, shards, deadline):
    import torch
    sys.path.insert(0, str(HERE / "methods" / method))
    impl = importlib.import_module("quant")
    prep = results(p) / "prep"
    arm = arm_name(p, method, bits)
    out = results(p) / "quantized" / arm
    rows = read(prep / "modules.json")
    owner = shard_layers(rows, shards)
    mine = [r for r in rows if owner[r["index"]] == shard]
    started, left = time.monotonic(), 0
    for done, row in enumerate(mine):
        i, name = row["index"], row["module"]
        if (out / "stats" / f"{i:04d}.json").exists():
            continue
        if time.time() > deadline:
            left = sum(not (out / "stats" / f"{r['index']:04d}.json").exists() for r in mine)
            break
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
        torch_save(dict(module=name, method=method, bits=bits, seed=SEED + i, codes=result["codes"]),
                   out / "codes" / f"{i:04d}.pt")
        torch_save(dict(module=name, weight=dq.to(torch.bfloat16).cpu()), out / "weights" / f"{i:04d}.pt")
        save(out / "stats" / f"{i:04d}.json", dict(
            index=i, module=name, shape=list(w.shape), bits_total=result["bits_total"],
            bpw=result["bits_total"] / w.numel(), proxy_nmse=proxy, seconds=time.monotonic() - begin,
            config=result["config"]))
        del w, h, dq, d, result
        torch.cuda.empty_cache()
        if done % 20 == 0:
            print(f"QUANT {arm} shard={shard}/{shards} layer={done + 1}/{len(mine)} proxy={proxy:.5f} "
                  f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    save(out / f"shard_{shard}_of_{shards}.json", dict(complete=left == 0, layers=len(mine), left=left,
                                                       seconds=time.monotonic() - started,
                                                       pbs_jobid=os.environ["PBS_JOBID"]))
    return arm, left


def finalize(p, arm):
    out = results(p) / "quantized" / arm
    stats = [read(out / "stats" / f"{i:04d}.json") for i in range(TARGETS)]
    weights = sum(s["shape"][0] * s["shape"][1] for s in stats)
    summary = dict(protocol=p["protocol"], arm=arm, method=p["arms"][arm]["method"], bits=p["arms"][arm]["bits"],
                   complete=True, pbs_jobid=os.environ["PBS_JOBID"], target_weights=weights,
                   bpw=sum(s["bits_total"] for s in stats) / weights,
                   proxy_nmse_weighted=sum(s["proxy_nmse"] * s["shape"][0] * s["shape"][1] for s in stats) / weights,
                   proxy_nmse_max=max(s["proxy_nmse"] for s in stats),
                   worst_layers=sorted(((s["proxy_nmse"], s["module"]) for s in stats), reverse=True)[:5],
                   quantize_gpu_hours=sum(s["seconds"] for s in stats) / 3600, config_example=stats[0]["config"])
    save(out / "COMPLETE.json", summary)
    print(f"QUANT_COMPLETE {arm} bpw={summary['bpw']:.4f} proxy={summary['proxy_nmse_weighted']:.5f}", flush=True)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", required=True, help="method:bits:shard:shards, space separated")
    args = parser.parse_args()
    p = read(args.protocol)
    import torch
    # FP32 matmuls/Cholesky without TF32, as in the Fast-WAM A16 quant worker.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deadline = float(os.environ["COSMOS_JOB_START_EPOCH"]) + p["single_gpu"]["soft_deadline_seconds"] - 600
    unfinished = 0
    for item in args.plan.split():
        method, bits, shard, shards = item.split(":")
        arm, left = quantize(p, method, int(bits), int(shard), int(shards), deadline)
        unfinished += left
        out = results(p) / "quantized" / arm
        if all((out / "stats" / f"{i:04d}.json").exists() for i in range(TARGETS)):
            finalize(p, arm)
    if unfinished:
        print(f"QUANT_INCOMPLETE layers_left={unfinished}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
