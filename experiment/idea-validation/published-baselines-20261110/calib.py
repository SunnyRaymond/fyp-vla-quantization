"""Calibration stage of one model, on one GPU: architecture facts, C80 layer inputs (SVDQuant) and SpinQuant's rotations.

SpinQuant follows scripts/10_optimize_rotation.sh: R1 (one per stream) and R2 (one per attention, head dimension), both
started from random signs times Hadamard, optimized on the Stiefel manifold by SGDG (lr 1.5 cosine to 0, 100 steps) with
the weights at 16 bits and A4/KV4 active (the repository's recipe for a GPTQ target). The objective is the normalized
denoiser-output error to the BF16 teacher on C80 packets, the WAM counterpart of the official next-token loss.
"""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import math
import os
from pathlib import Path
import random
import time
import types

import torch
import torch.nn.functional as F

import kvsites
import methods as mt
import qops as q

METHODS = {"video": "_denoise_video", "action": "_denoise_action_with_video_cache"}


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


def tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {k: tree(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree(v, device) for v in value)
    return value


class FastWAM:
    name = "fastwam"

    def __init__(self, p, m, out):
        import numpy as np
        import calibration_source
        from fw_common import bf16_runtime, guard
        guard()
        self.np, self.src = np, calibration_source
        self.rt = bf16_runtime(out, text_encoder=True)
        self.m, self.model = m, self.rt.model
        self.plan_path = Path(m["panel"]["calibration_plan"])
        self.plan = {r["case_id"]: r for r in read(self.plan_path)["inputs"]}
        self.modules = {n: mod for n, mod in self.rt.q.modules.items() if mt.parse(n)}
        block = self.model.video_expert.blocks[0].self_attn
        self.arch = dict(heads=block.num_heads, head_dim=block.attn_head_dim)

    def query(self, case):
        observation, metadata = self.src.load_observation(self.np, self.plan_path.parent, self.plan[case])
        self.rt.description = metadata["description"]
        self.rt.infer(self.rt.datum(observation), self.plan[case]["sampler_seeds"][0], measure=False)

    def packets(self, cases, indices):
        out, model = [], self.model
        for case in cases:
            caches, counts = {}, Counter()
            originals = {s: getattr(model, name) for s, name in METHODS.items()}

            def capture(stream):
                original = originals[stream]
                signature = inspect.signature(original)

                def wrapped(_model, *args, **kwargs):
                    step = counts[stream]
                    counts[stream] += 1
                    captured = None
                    if step in indices:
                        bound = signature.bind(*args, **kwargs)
                        bound.apply_defaults()
                        captured = dict(bound.arguments)
                        if stream == "action":
                            if not caches:
                                caches.update(tree({k: captured[k] for k in ("video_cache_k", "video_cache_v")}, "cpu"))
                            captured = {k: v for k, v in captured.items() if k not in ("video_cache_k", "video_cache_v")}
                        captured = tree(captured, "cpu")
                    prediction = original(*args, **kwargs)
                    if step in indices:
                        out.append(dict(stream=stream, case=case, step=step, kwargs=captured, cache=caches if stream == "action" else None,
                                        teacher=tree(prediction, "cpu")))
                    return prediction
                return types.MethodType(wrapped, model)

            for stream, name in METHODS.items():
                setattr(model, name, capture(stream))
            try:
                self.query(case)
            finally:
                for stream, name in METHODS.items():
                    setattr(model, name, originals[stream])
        return out

    def install_kv(self, kv):
        kvsites.install_fastwam(self.model, kv, self.m["frame_tokens"], self.m["video_tokens"])

    def replay(self, packet, kv):
        kwargs = tree(packet["kwargs"], "cuda")
        if packet["stream"] == "action":
            kwargs["video_cache_k"] = [kv.k(t, "future") for t in tree(packet["cache"]["video_cache_k"], "cuda")]
            kwargs["video_cache_v"] = [kv.v(t, "future", f"video.{i}.self") for i, t in
                                       enumerate(tree(packet["cache"]["video_cache_v"], "cuda"))]
        return getattr(self.model, METHODS[packet["stream"]])(**kwargs), tree(packet["teacher"], "cuda")


class Cosmos:
    name = "cosmos"

    def __init__(self, p, m, out):
        from cos_common import calibration_query, guard, runtime
        guard()
        self.m, self.calibration_query = m, calibration_query
        self.rt = runtime(m, out, Path(m["a16_prep"]) / "base_extra_cache.pkl")
        self.model = self.rt.model
        self.modules = {n: mod for n, mod in self.rt.target_linears.items() if mt.parse(n)}
        attn = next(a for n, a in self.model.named_modules() if hasattr(a, "compute_qkv"))
        self.arch = dict(heads=attn.n_heads, head_dim=attn.head_dim)

    def query(self, case):
        self.calibration_query(self.m, self.rt, case)

    def packets(self, cases, indices):
        out, net, step = [], self.model.net, Counter()
        original = net.forward

        def forward(*args, **kwargs):
            prediction = original(*args, **kwargs)
            out.append(dict(stream="joint", case=case, step=step[case], args=tree(args, "cpu"), kwargs=tree(kwargs, "cpu"),
                            autocast=(torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()),
                            teacher=tree(prediction, "cpu")))
            step[case] += 1
            return prediction
        net.forward = forward
        try:
            for case in cases:
                self.query(case)
        finally:
            del net.forward
        return out

    def install_kv(self, kv):
        kvsites.install_cosmos(self.model, kv, self.m["attentions"])
        for block in self.model.net.blocks:
            block.forward = block._checkpoint_wrapped_module.forward

    def replay(self, packet, kv):
        enabled, dtype = packet["autocast"]
        with torch.autocast("cuda", enabled=enabled, dtype=dtype):
            prediction = self.model.net(*tree(packet["args"], "cuda"), **tree(packet["kwargs"], "cuda"))
        return prediction, tree(packet["teacher"], "cuda")


def capture_samples(ad, cases, index_of, out, per_call, cap, seed):
    todo = {n: m for n, m in ad.modules.items() if not (out / "samples" / f"{index_of[n]:04d}.pt").exists()}
    if not todo:
        return 0
    store, gen = {n: [] for n in todo}, torch.Generator().manual_seed(seed)
    for n, m in todo.items():
        def forward(x, _n=n, _m=m):
            flat = x.detach().reshape(-1, _m.in_features)
            picks = torch.randperm(flat.shape[0], generator=gen)[:per_call].to(flat.device)
            store[_n].append(flat[picks].cpu())
            return F.linear(x, _m.weight, _m.bias)
        m.forward = forward
    try:
        for i, case in enumerate(cases):
            ad.query(case)
            if i % 20 == 19:
                print(f"SAMPLES {ad.name} contexts={i + 1}", flush=True)
    finally:
        for m in todo.values():
            del m.forward
    for n, rows in store.items():
        x = torch.cat(rows, 0)
        x = x[torch.randperm(x.shape[0], generator=gen)[:cap]]
        torch_save(dict(module=n, x=x), out / "samples" / f"{index_of[n]:04d}.pt")
    return len(store)


def sgdg_step(params, lr, seed_rng):
    """SGDG on the Stiefel manifold (momentum 0): train_utils/optimizer.py of the official repository."""
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            x = p.data
            x = x / x.norm(dim=1, keepdim=True).add(1e-8)
            if seed_rng.randint(1, 101) == 1:
                t = x.t().contiguous()
                qr, r = torch.linalg.qr(t)
                x = (qr * torch.diag(r, 0).sign().expand_as(qr)).t()
            g = p.grad.data
            v = -g.t()
            mx = v @ x
            xmx = x @ mx
            xxmx = x.t() @ xmx
            w_hat = mx - 0.5 * xxmx
            w = w_hat - w_hat.t()
            step = 1.0 / (torch.abs(w).sum(0).max() + 1e-8)
            step = min(float(step), lr)
            y0 = x.t()
            y = y0 + step * v
            for _ in range(5):
                y = y0 + step * (w @ (0.5 * (y0 + y)))
            p.copy_(y.t())
            p.grad = None


def train_rotations(ad, arch, m, cfg, out):
    device = torch.device("cuda")
    rng = random.Random(cfg["seed"])
    params = {}

    def init(tag, width):
        s = q.random_signs(width, mt.seed_of(cfg["seed"], tag), device)
        return torch.nn.Parameter(q.had(torch.diag(s).float()))

    for name, mod in ad.modules.items():
        role = mt.parse(name)
        if role.kind in ("q", "k", "v", "mlp0"):
            stream = mt.stream_of(role)
            params.setdefault(stream, init(stream, mod.in_features))
        if role.kind == "o":
            key = mt.attn_key(role)
            params.setdefault(key, init(key, arch["head_dim"]))
    cases = list(range(2000, 2080))
    indices = tuple(m["packet_indices"])
    started = time.monotonic()
    packets = ad.packets(cases, indices)
    if not packets:
        raise RuntimeError("No teacher packets were captured")
    print(f"PACKETS {ad.name} {Counter(pk['stream'] for pk in packets)} in {time.monotonic() - started:.0f}s", flush=True)
    kv = kvsites.KV("spinquant", arch, params, ste=True)
    ad.install_kv(kv)
    for name, mod in ad.modules.items():
        role = mt.parse(name)
        xf = mt.xf_for("spinquant", role, mod.in_features, arch, m["rotation_seed"], params, device)
        actq = mt.actq_for("spinquant", role, arch)

        def forward(x, _xf=xf, _a=actq, _m=mod):
            r = _xf.apply(x.float())
            return F.linear(_xf.inv(q.ste(_a(r), r)).to(x.dtype), _m.weight, _m.bias)
        mod.forward = forward
    order = list(range(len(packets)))
    rng.shuffle(order)
    steps, batch = cfg["steps"], cfg["batch"]
    params_list = list(params.values())
    start = time.monotonic()
    pos = 0
    log = []
    for step in range(steps):
        lr = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * step / steps))
        losses = []
        for _ in range(batch):
            pk = packets[order[pos % len(order)]]
            pos += 1
            with torch.enable_grad():
                pred, teacher = ad.replay(pk, kv)
                pred = pred[0] if isinstance(pred, tuple) else pred
                teacher = teacher[0] if isinstance(teacher, tuple) else teacher
                loss = ((pred.float() - teacher.float()).square().mean()
                        / teacher.float().square().mean().clamp_min(1e-12)) / batch
                if not (torch.isfinite(loss) and loss.requires_grad):
                    raise RuntimeError(f"Invalid training loss at step {step}: {loss}")
                loss.backward()
            losses.append(float(loss) * batch)
        sgdg_step(params_list, lr, rng)
        log.append(dict(step=step, loss=sum(losses) / len(losses), lr=lr))
        if step % 10 == 0 or step == steps - 1:
            print(f"SPIN {ad.name} step={step} loss={log[-1]['loss']:.5f} lr={lr:.3f} elapsed={time.monotonic() - start:.0f}s",
                  flush=True)
    worst = max(float((p.data @ p.data.t() - torch.eye(p.shape[0], device=device)).abs().max()) for p in params_list)
    torch_save({k: v.detach().cpu() for k, v in params.items()}, out / "rotations.pt")
    save(out / "rotations.json", dict(protocol=ad.m.get("name", ad.name), steps=steps, batch=batch, lr=cfg["lr"], matrices=len(params),
                                      losses=log, orthogonality_max_abs=worst, packets=len(packets),
                                      seconds=time.monotonic() - start, pbs_jobid=os.environ.get("PBS_JOBID")))
    print(f"ROTATIONS {ad.name} matrices={len(params)} orth={worst:.2e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    m = dict(p["models"][args.model])
    out = Path(p["remote_root"]) / "results" / args.model
    index_of = {r["module"]: r["index"] for r in read(Path(m["a16_prep"]) / "modules.json")}
    ad = (FastWAM if args.model == "fastwam" else Cosmos)(p, m, out / "calib_runtime")
    save(out / "arch.json", dict(ad.arch, targets=len(ad.modules)))
    print(f"ARCH {ad.name} {ad.arch} targets={len(ad.modules)}", flush=True)
    cases = list(range(2000, 2080))
    ncap = capture_samples(ad, cases, index_of, out, p["methods"]["svdquant"]["samples_per_call"],
                           p["methods"]["svdquant"]["samples_per_layer"], p["seed"])
    print(f"SAMPLES {ad.name} captured={ncap}", flush=True)
    if not (out / "rotations.pt").exists():
        train_rotations(ad, ad.arch, m, p["methods"]["spinquant"], out)
    save(out / "calib" / "COMPLETE.json", dict(complete=True, pbs_jobid=os.environ.get("PBS_JOBID")))


if __name__ == "__main__":
    main()
