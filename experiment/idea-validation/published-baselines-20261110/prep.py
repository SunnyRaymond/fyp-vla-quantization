"""Fit the four published recipes on one model from the BF16 masters, the C80 Hessians, the calibration stage and the
per-channel maxima. Writes results/<model>/banks/<arm>/NNNN.pt and stats/<arm>/NNNN.json; --finalize checks all layers
and writes banks/<arm>/COMPLETE.json. Runs on a GPU, no model is loaded, resumable group by group.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import time

import torch

import methods as mt
import qops as q


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


def load_maxima(cfg, names):
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
                merged[name] = torch.stack([torch.as_tensor(r["absmax"], dtype=torch.float32).reshape(-1) for r in rows]).amax(0)
            else:
                merged[name] = torch.as_tensor(entry["absmax"], dtype=torch.float32).reshape(-1)
    missing = [n for n in names if n not in merged]
    if missing:
        raise RuntimeError(f"Maxima miss {len(missing)} layers, e.g. {missing[:5]}")
    return merged


def svd_groups(rows):
    """Linears sharing one input, in DeepCompressor's grouping: self q/k/v, cross k/v; everything else alone."""
    groups, order = {}, []
    for r in rows:
        role = mt.parse(r["module"])
        if role.attn == "self" and role.kind in ("q", "k", "v"):
            key = f"{role.expert}.{role.block}.self.qkv"
        elif role.attn == "cross" and role.kind in ("k", "v"):
            key = f"{role.expert}.{role.block}.cross.kv"
        else:
            key = r["module"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: mt.parse(r["module"]).kind)
    return [(k, groups[k]) for k in order]


def owner_of(items, size_of, shards):
    loads, owner = [0] * shards, {}
    for key, cost in sorted(((k, size_of(v)) for k, v in items), key=lambda t: -t[1]):
        k = min(range(shards), key=lambda s: (loads[s], s))
        owner[key] = k
        loads[k] += cost
    return owner


def mp_bits(role, blocks):
    """W bits of ViDiT-Q's W4A8 mixed-precision recipe (examples/pixart/configs/w4a8_mixed_precision.yaml): attention W4,
    feed-forward and the first and last block W8. Activations are A8 everywhere."""
    return 8 if role.kind in ("mlp0", "mlp2") or role.block in (0, blocks - 1) else 4


def fit(p, model, shard, shards):
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg, methods = p["models"][model], p["methods"]
    prep = Path(cfg["a16_prep"])
    all_rows = read(prep / "modules.json")
    rows = [r for r in all_rows if mt.parse(r["module"])]
    if len(rows) != cfg["targets"]:
        raise RuntimeError(f"Expected {cfg['targets']} block-level Linears, got {len(rows)}")
    out = Path(p["remote_root"]) / "results" / model
    arch = read(out / "arch.json")
    device = torch.device("cuda")
    blocks = cfg["blocks"]
    names = [r["module"] for r in rows]
    maxima = load_maxima(cfg, names)
    rotations = {k: v.to(device, torch.float32) for k, v in
                 torch.load(out / "rotations.pt", map_location="cpu", weights_only=True).items()}
    started = time.monotonic()

    def master(i, name):
        m = torch.load(prep / "masters" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        if m["module"] != name:
            raise RuntimeError(f"Master identity differs at {i}")
        return m["weight"].to(device, torch.float32)

    def hessian(i, name):
        h = torch.load(prep / "hessians" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        if h["module"] != name:
            raise RuntimeError(f"Hessian identity differs at {i}")
        return h["hessian"].to(device, torch.float32)

    def write(arm, row, bank, proxy, begin):
        bits = bits_of(bank)
        stored = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in bank.items()}
        torch_save(dict(stored, module=row["module"], arm=arm), out / "banks" / arm / f"{row['index']:04d}.pt")
        save(out / "stats" / arm / f"{row['index']:04d}.json", dict(index=row["index"], module=row["module"], shape=row["shape"],
             bits_total=bits, bpw=bits / (row["shape"][0] * row["shape"][1]), proxy_nmse=proxy, seconds=time.monotonic() - begin))

    owner = owner_of([(r["module"], r) for r in rows], lambda r: r["shape"][1] ** 2 + r["shape"][0] * r["shape"][1], shards)
    for arm in p["arms"]:
        if arm == "svdquant":
            continue
        for row in rows:
            name = row["module"]
            if owner[name] != shard or (out / "stats" / arm / f"{row['index']:04d}.json").exists():
                continue
            begin = time.monotonic()
            role = mt.parse(name)
            w = master(row["index"], name)
            if arm in mt.ROTATED:
                xf = mt.xf_for(arm, role, w.shape[1], arch, cfg["rotation_seed"], rotations, device)
                bank, proxy = mt.fit_rotated(arm, w, hessian(row["index"], name), xf)
            else:
                signs = q.random_signs(w.shape[1], cfg["rotation_seed"] + row["index"] * 17, device)
                bank, proxy = mt.fit_viditq(w, maxima[name], mp_bits(role, blocks), methods["viditq"]["alpha"], signs)
            write(arm, row, bank, proxy, begin)
            del w
        print(f"FIT {model}/{arm} shard={shard} elapsed={time.monotonic() - started:.0f}s", flush=True)
        torch.cuda.empty_cache()
    arm = "svdquant"
    if arm in p["arms"]:
        groups = svd_groups(rows)
        owner_g = owner_of(groups, lambda g: sum(r["shape"][1] ** 2 + r["shape"][0] * r["shape"][1] for r in g), shards)
        for key, group in groups:
            if owner_g[key] != shard or all((out / "stats" / arm / f"{r['index']:04d}.json").exists() for r in group):
                continue
            begin = time.monotonic()
            weights = [master(r["index"], r["module"]) for r in group]
            sample = torch.load(out / "samples" / f"{group[0]['index']:04d}.pt", map_location="cpu", weights_only=True)
            role0 = mt.parse(group[0]["module"])
            unsigned = role0.kind == "mlp2"
            x = sample["x"].to(device, torch.float32) + (mt.SVD_SHIFT if unsigned else 0.0)
            s, a, parts, extra = mt.fit_svdquant(weights, x, unsigned, p["seed"] + group[0]["index"], methods["svdquant"])
            for row, (n, codes, scales, b) in zip(group, parts):
                bank = dict(q.encode_sym(codes, scales, 64), smoothing=s, up=b, down=a,
                            bias_shift=(weights[group.index(row)].sum(1) * mt.SVD_SHIFT if unsigned else None),
                            count_down=row is group[0])
                write(arm, row, bank, extra["objective"], begin)
            del weights
        print(f"FIT {model}/{arm} shard={shard} elapsed={time.monotonic() - started:.0f}s", flush=True)
    save(out / "shards" / f"shard_{shard}_of_{shards}.json", dict(complete=True, pbs_jobid=os.environ.get("PBS_JOBID"),
                                                                 seconds=time.monotonic() - started))


def bits_of(bank):
    bits = bank["packed"].numel() * 8
    if bank["kind"] == "sym":
        bits += bank["scales"].numel() * 32
    else:
        bits += bank["delta"].numel() * 16 + bank["zero"].numel() * 16
    for key, width in (("mask", 32), ("signs", 8), ("up", 16)):
        if bank.get(key) is not None:
            bits += bank[key].numel() * width
    if bank.get("smoothing") is not None:
        bits += bank["smoothing"].numel() * 32
    if bank.get("down") is not None and bank.get("count_down", True):
        bits += bank["down"].numel() * 16
    if bank.get("bias_shift") is not None:
        bits += 0
    return int(bits)


def finalize(p, model):
    cfg = p["models"][model]
    out = Path(p["remote_root"]) / "results" / model
    rows = [r for r in read(Path(cfg["a16_prep"]) / "modules.json") if mt.parse(r["module"])]
    for arm in p["arms"]:
        stats = [read(out / "stats" / arm / f"{r['index']:04d}.json") for r in rows]
        missing = [r["index"] for r in rows if not (out / "banks" / arm / f"{r['index']:04d}.pt").exists()]
        if missing:
            raise RuntimeError(f"{arm} is missing {len(missing)} banks")
        weights = sum(s["shape"][0] * s["shape"][1] for s in stats)
        summary = dict(protocol=p["protocol"], arm=arm, model=model, complete=True, weights=weights,
                       bpw=sum(s["bits_total"] for s in stats) / weights,
                       proxy_nmse_weighted=None if arm == "viditq" or arm == "svdquant" else
                       sum(s["proxy_nmse"] * s["shape"][0] * s["shape"][1] for s in stats) / weights,
                       seconds=sum(s["seconds"] for s in stats), pbs_jobid=os.environ.get("PBS_JOBID"))
        if weights != cfg["target_weights"]:
            raise RuntimeError(f"{arm}: {weights} weights, expected {cfg['target_weights']}")
        save(out / "banks" / arm / "COMPLETE.json", summary)
        print(f"FIT_COMPLETE {model}/{arm} bpw={summary['bpw']:.4f}", flush=True)


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
