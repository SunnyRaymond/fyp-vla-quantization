"""Fit the four baselines on one model from the saved BF16 masters, C80 Hessians and channel maxima.

Writes results/<model>/banks/<arm>/NNNN.pt (packed codes, scales, channel vectors, signs, low-rank factors) and
results/<model>/stats/<arm>/NNNN.json; --finalize checks all layers and writes results/<model>/banks/<arm>/COMPLETE.json.
Resumable layer by layer. Runs on a GPU; no model is loaded.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import time

import torch

import baselines as bl
import scalarq as sq


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def shard_owner(modules, shards):
    loads, owner = [0] * shards, {}
    for row in sorted(modules, key=lambda r: -(r["shape"][1] ** 2 + r["shape"][0] * r["shape"][1])):
        k = min(range(shards), key=lambda s: (loads[s], s))
        owner[row["index"]] = k
        loads[k] += row["shape"][1] ** 2 + row["shape"][0] * row["shape"][1]
    return owner


def load_maxima(cfg, names):
    """Per-channel input maxima over the frozen calibration set, keyed by module name."""
    m = cfg["maxima"]
    merged = {}
    if m["kind"] == "shards":
        for path in sorted(glob.glob(m["pattern"])):
            for name, value in torch.load(path, map_location="cpu", weights_only=True)["maxima"].items():
                merged[name] = torch.maximum(merged[name], value) if name in merged else value
    else:
        data = torch.load(m["path"], map_location="cpu", weights_only=True)
        for name, entry in data.items():
            by_step = entry.get("by_timestep")
            if by_step:
                rows = list(by_step.values()) if isinstance(by_step, dict) else list(by_step)
                values = torch.stack([torch.as_tensor(r["absmax"], dtype=torch.float32).reshape(-1) for r in rows])
                merged[name] = values.amax(0)
            else:
                merged[name] = torch.as_tensor(entry["absmax"], dtype=torch.float32).reshape(-1)
    if set(merged) != set(names):
        raise RuntimeError(f"Maxima cover {len(merged)} layers, expected {len(names)}: "
                           f"{sorted(set(names) ^ set(merged))[:5]}")
    return merged


def fit(p, model, shard, shards):
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg, quant, methods = p["models"][model], p["quantization"], p["methods"]
    prep = Path(cfg["a16_prep"])
    modules = read(prep / "modules.json")
    if len(modules) != cfg["targets"]:
        raise RuntimeError(f"Expected {cfg['targets']} target Linears, got {len(modules)}")
    owner = shard_owner(modules, shards)
    mine = [r for r in modules if owner[r["index"]] == shard]
    names = [r["module"] for r in modules]
    shapes = {r["module"]: tuple(r["shape"]) for r in modules}
    device = torch.device("cuda")
    maxima = load_maxima(cfg, names)
    out = Path(p["remote_root"]) / "results" / model
    started = time.monotonic()
    for arm in p["arms"]:
        coords = bl.build_coords(cfg, arm, shapes, device) if arm in bl.ROTATED else None
        for done, row in enumerate(mine):
            i, name = row["index"], row["module"]
            stats = out / "stats" / arm / f"{i:04d}.json"
            if stats.exists():
                continue
            begin = time.monotonic()
            master = torch.load(prep / "masters" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
            hess = torch.load(prep / "hessians" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
            if master["module"] != name or hess["module"] != name:
                raise RuntimeError(f"Prep identity differs at {i}: {name}")
            w = master["weight"].to(device, torch.float32)
            h = hess["hessian"].to(device, torch.float32)
            bank, proxy = bl.fit_layer(arm, w, h, maxima[name], coords, name, i, quant, methods)
            bits = sq.bank_bits(bank)
            stored = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in bank.items()}
            torch_save(dict(stored, module=name, arm=arm), out / "banks" / arm / f"{i:04d}.pt")
            save(stats, dict(index=i, module=name, shape=list(w.shape), bits_total=bits, bpw=bits / w.numel(),
                             proxy_nmse=proxy, seconds=time.monotonic() - begin))
            del w, h, bank, stored
            if done % 25 == 0:
                print(f"FIT {model}/{arm} shard={shard} layer={done + 1}/{len(mine)} proxy={proxy:.5f} "
                      f"elapsed={time.monotonic() - started:.0f}s", flush=True)
        torch.cuda.empty_cache()
    save(out / "shards" / f"shard_{shard}_of_{shards}.json", dict(complete=True, pbs_jobid=os.environ.get("PBS_JOBID"),
                                                                 seconds=time.monotonic() - started))


def finalize(p, model):
    cfg = p["models"][model]
    out = Path(p["remote_root"]) / "results" / model
    modules = read(Path(cfg["a16_prep"]) / "modules.json")
    for arm in p["arms"]:
        stats = [read(out / "stats" / arm / f"{r['index']:04d}.json") for r in modules]
        missing = [r["index"] for r in modules if not (out / "banks" / arm / f"{r['index']:04d}.pt").exists()]
        if missing:
            raise RuntimeError(f"{arm} is missing {len(missing)} banks")
        weights = sum(s["shape"][0] * s["shape"][1] for s in stats)
        summary = dict(protocol=p["protocol"], arm=arm, model=model, complete=True, weights=weights,
                       bpw=sum(s["bits_total"] for s in stats) / weights,
                       proxy_nmse_weighted=sum(s["proxy_nmse"] * s["shape"][0] * s["shape"][1] for s in stats) / weights,
                       proxy_nmse_max=max(s["proxy_nmse"] for s in stats),
                       worst_layers=sorted(((s["proxy_nmse"], s["module"]) for s in stats), reverse=True)[:5],
                       seconds=sum(s["seconds"] for s in stats), pbs_jobid=os.environ.get("PBS_JOBID"))
        if summary["weights"] != cfg["target_weights"]:
            raise RuntimeError(f"{arm}: {summary['weights']} weights, expected {cfg['target_weights']}")
        save(out / "banks" / arm / "COMPLETE.json", summary)
        print(f"FIT_COMPLETE {model}/{arm} bpw={summary['bpw']:.4f} proxy={summary['proxy_nmse_weighted']:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    if args.finalize:
        finalize(p, args.model)
    else:
        fit(p, args.model, args.shard, args.shards)


if __name__ == "__main__":
    main()
