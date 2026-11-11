"""Future swap on Fast-WAM W4A4KV4: the share of the action error that enters through the predicted future (Fig. 1).

Arms on the 16 LAQ calibration contexts, against the BF16 teacher the LAQ calibration saved for the same context and
sampler seed:
  plain        the QuaRot W4A4KV4 arm (LAQ with nothing refined)
  plain.swap   plain, but after the 10 video steps the prefill reads the BF16 future instead of its own
  base0        plain + the action rows refined (LAQ's base0)
  base0.swap   base0 with the BF16 future
rho(base) = 1 - mean A(base.swap) / mean A(base), A = MSE of the first 10 actions, motor dims (laq.errors).

The code under test is the LAQ calibration's own copy (artifacts of job 26196738); this file only adds the swap hook.
Checks before any unit: BF16 bitwise vs the A16 anchor; BF16 + swap of the teacher's future reproduces the teacher's
action bit for bit; 4-bit with nothing refined bitwise vs the QuaRot anchor; the hook fires exactly once per swap query.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import random
import time

from fw_common import bf16_runtime, capture, guard, max_abs, read, save
import calib
import laq

ARMS = ("plain", "plain.swap", "base0", "base0.swap")


def install_swap(st, model):
    """After the last video step the native code calls video_expert.prepare(x=latents_video, timestep=0, ...) once to
    build the prefill tokens. With st.swap set, that call gets the BF16 future instead (same observation frame)."""
    original = model.video_expert.prepare
    signature = inspect.signature(original)
    st.swap, st.swapped = None, 0

    def prepare(*args, **kwargs):
        if st.swap is None or st.n_video != st.steps or st.phase is None or st.phase[0] != "video":
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        x = bound.arguments["x"]
        t = st.swap.to(device=x.device, dtype=x.dtype)
        if t.shape != x.shape:
            raise RuntimeError(f"Teacher future {tuple(t.shape)} differs from the latents {tuple(x.shape)}")
        if not bool(bound.arguments["timestep"].eq(0).all()):
            raise RuntimeError("The swapped prepare call is not the timestep-0 prefill preparation")
        if not torch_equal(t[:, :, :1], x[:, :, :1]):
            raise RuntimeError("The teacher's observation frame differs from this query's")
        bound.arguments["x"] = t
        st.swapped += 1
        return original(*bound.args, **bound.kwargs)
    model.video_expert.prepare = prepare


def torch_equal(a, b):
    import torch
    return bool(torch.equal(a, b))


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def rho(base, swap):
    return 1 - mean(swap) / mean(base)


def bootstrap(base, swap, n=2000, seed=20261111):
    rng = random.Random(seed)
    k = len(base)
    draws = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        draws.append(rho([base[i] for i in idx], [swap[i] for i in idx]))
    draws.sort()
    return draws[int(0.025 * n)], draws[int(0.975 * n) - 1]


def work(p, s):
    import numpy as np
    import torch
    import calibration_source
    from profile_worker import _capture_anchor
    guard()
    m, c = p["models"]["fastwam"], p["laq"]
    out_root = Path(s["remote_root"]) / "results"
    laq_calib = Path(s["source"]["calib"])
    prep = Path(m["a16_prep"])
    job = os.environ["PBS_JOBID"].split(".")[0]
    started = time.monotonic()
    runtime = bf16_runtime(out_root / f"runtime_{job}", text_encoder=True)
    t5 = torch.load(prep / "t5_cache.pt", map_location="cpu", weights_only=True)
    encode = runtime.model.encode_prompt

    def encode_prompt(prompt):
        if prompt in t5:
            emb, mask = t5[prompt]
            return emb.to(runtime.model.device), mask.to(runtime.model.device)
        return encode(prompt)
    runtime.model.encode_prompt = encode_prompt
    device = torch.device("cuda")
    st = laq.State(c["steps"], m["frame_tokens"], m["video_tokens"], read(c["arch"]), m["rotation_seed"], device)
    modules = dict(runtime.q.modules)
    if laq.install(st, runtime.model, modules) != m["targets"]:
        raise RuntimeError("Block-level Linear count differs from the protocol")
    install_swap(st, runtime.model)

    checks = {}
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    action, future = capture(m, runtime)
    checks["bf16_vs_a16_anchor"] = dict(action=max_abs(action, reference["action"]), future=max_abs(future, reference["future"]))
    if any(checks["bf16_vs_a16_anchor"].values()):
        raise RuntimeError(f"BF16 with the hooks differs from the A16 anchor: {checks}")
    metric = laq.Metric(torch.load(c["metric"], map_location="cpu", weights_only=True), device)

    plan_path = Path(m["panel"]["calibration_plan"])
    plan = read(plan_path)
    rows = {r["case_id"]: r for r in plan["inputs"]}
    contexts = plan["calibration_cohorts"]["c80"][:c["contexts"]]
    if contexts != read(laq_calib / "allocation.json")["contexts"]:
        raise RuntimeError("The contexts differ from the LAQ calibration's")
    teacher = {k: torch.load(laq_calib / "teacher" / f"{k}.pt", map_location="cpu", weights_only=True) for k in contexts}
    print(f"SWAP contexts={contexts} load={time.monotonic() - started:.0f}s", flush=True)

    def run(case, swap):
        observation, metadata = calibration_source.load_observation(np, plan_path.parent, rows[case])
        runtime.description = metadata["description"]
        st.seed = c["seed"] + case
        st.reset()
        st.swap = teacher[case]["future"] if swap else None
        st.swapped = 0
        try:
            action, _, future, calls = _capture_anchor(runtime, runtime.datum(observation), rows[case]["sampler_seeds"][0], torch)
        finally:
            st.swap = None
        if calls != 1:
            raise RuntimeError("A query called infer_action more than once")
        if st.swapped != (1 if swap else 0):
            raise RuntimeError(f"The swap hook fired {st.swapped} times")
        return action.detach().float().cpu(), future.float().cpu()

    first = contexts[0]
    st.use(None)
    st.inject = None
    action, _ = run(first, True)
    checks["bf16_swap_vs_teacher"] = dict(case=first, action=max_abs(action, teacher[first]["action"]))
    if checks["bf16_swap_vs_teacher"]["action"] != 0:
        raise RuntimeError(f"BF16 reading the teacher's own future differs from the teacher: {checks}")

    index_of = {r["module"]: r["index"] for r in read(prep / "modules.json")}
    loaded = laq.load_weights(st, modules, index_of, Path(c["bank"]))
    st.use(laq.spec_none(c["steps"]))
    action, future = capture(m, runtime)
    plain = torch.load(c["plain_anchor"], map_location="cpu", weights_only=True)
    checks["plain_vs_quarot_anchor"] = dict(action=max_abs(action, plain["action"]), future=max_abs(future, plain["future"]))
    if loaded != m["targets"] or any(checks["plain_vs_quarot_anchor"].values()):
        raise RuntimeError(f"4-bit with nothing refined differs from the QuaRot anchor: {checks}")
    save(out_root / f"startup_{job}.json", checks)
    print("SWAP_CHECKS " + json.dumps(checks), flush=True)

    for arm in ARMS:
        base, swap = arm.split(".")[0], arm.endswith(".swap")
        for case in contexts:
            path = out_root / "units" / arm / f"{case}.json"
            if path.exists():
                continue
            begin = time.monotonic()
            st.use(calib.spec_of(base, c, m))
            action, future = run(case, swap)
            record = dict(laq.errors(metric, action, future, teacher[case]), arm=arm, case=case,
                          seconds=time.monotonic() - begin, pbs_jobid=os.environ["PBS_JOBID"])
            save(path, record)
            print(f"UNIT {arm} case={case} A={record['A']:.4e} Z={record['Z']:.4e} t={record['seconds']:.1f}s", flush=True)
    print(f"SWAP_DONE elapsed={time.monotonic() - started:.0f}s", flush=True)


def finalize(p, s):
    out_root = Path(s["remote_root"]) / "results"
    laq_calib = Path(s["source"]["calib"])
    contexts = read(laq_calib / "allocation.json")["contexts"]
    unit = {a: [read(out_root / "units" / a / f"{k}.json") for k in contexts] for a in ARMS}
    stored = {a: [read(laq_calib / "units" / a / f"{k}.json") for k in contexts] for a in ("plain", "base0")}
    arms = {a: dict(A=mean(u["A"] for u in v), motor_rmse=mean(u["A"] for u in v) ** 0.5, Z=mean(u["Z"] for u in v),
                    ZM=mean(u["ZM"] for u in v), gripper_flips=sum(u["gripper_flips"] for u in v)) for a, v in unit.items()}
    share = {}
    for base in ("plain", "base0"):
        b, w = [u["A"] for u in unit[base]], [u["A"] for u in unit[base + ".swap"]]
        lo, hi = bootstrap(b, w)
        share[base] = dict(rho=rho(b, w), ci95=[lo, hi], contexts_improved=sum(x < y for x, y in zip(w, b)))
    repro = {a: max(abs(x["A"] - y["A"]) for x, y in zip(unit[a], stored[a])) for a in stored}
    summary = dict(protocol=s["protocol"], contexts=contexts, arms=arms, future_share=share,
                   max_abs_A_vs_laq_units=repro, units={a: unit[a] for a in ARMS})
    save(out_root / "summary.json", summary)
    lines = ["# Fast-WAM W4A4KV4：换入 BF16 future 后去掉的 action error（Fig. 1）", "",
             f"上下文 {len(contexts)} 个（LAQ 校准的 C80 前 16 个），对 LAQ 存下的 BF16 teacher；A = 前 10 步 motor 的 MSE。", "",
             "| 臂 | A | motor RMSE | future NMSE (Z) | Z_M | 夹爪翻转 |", "|---|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        v = arms[a]
        lines.append(f"| {a} | {v['A']:.4e} | {v['motor_rmse']:.4f} | {v['Z']:.4f} | {v['ZM']:.4f} | {v['gripper_flips']} |")
    lines += ["", "| 底座 | ρ = 1 − A(swap)/A | 95% bootstrap（按上下文） | 变好的上下文 |", "|---|---:|---|---:|"]
    for base, v in share.items():
        lines.append(f"| {base} | {100 * v['rho']:.1f}% | [{100 * v['ci95'][0]:.1f}%, {100 * v['ci95'][1]:.1f}%] | "
                     f"{v['contexts_improved']}/{len(contexts)} |")
    lines += ["", f"与 LAQ 校准存下的同名单元的 A 最大差：plain {repro['plain']:.3e}，base0 {repro['base0']:.3e}。"]
    (out_root / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(out_root / "COMPLETE.json", dict(complete=True, rho={b: v["rho"] for b, v in share.items()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--laq", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    s, p = read(args.protocol), read(args.laq)
    if args.finalize:
        finalize(p, s)
    else:
        work(p, s)


if __name__ == "__main__":
    main()
