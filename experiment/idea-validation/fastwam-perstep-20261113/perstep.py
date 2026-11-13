"""Per-phase quantization on Fast-WAM W4A4KV4: what a sensitivity taken one denoising step at a time says about the
pathway through the predicted future (Sec. 3.2, Observation 2).

One query has 21 phases: 10 video denoising steps, the prefill that writes the video cache, 10 action denoising steps.
An arm names the phases that run at W4A4KV4 (LAQ's base0: the QuaRot arm with the action rows refined); every other
phase runs in BF16. Arms, on the 16 LAQ calibration contexts against the BF16 teacher of that calibration:
  full       every phase                          (= base0 of the calibration)
  video      the 10 video steps                   (the future is written at 4 bits, the reader is BF16)
  reader     the prefill and the 10 action steps  (a 4-bit reader on the BF16 future)
  v.s        video step s alone, s = 0..9
  prefill    the prefill alone
  a.t        action step t alone, t = 0..9
A = MSE of the first 10 actions, motor dims; Z = future NMSE (laq.errors).
  per-step share of the future pathway = sum_s A(v.s) / (sum_s A(v.s) + A(prefill) + sum_t A(a.t))
  accumulation over the video steps    = A(video) / sum_s A(v.s), and the same for Z

The code under test is the LAQ calibration's own copy (artifacts of job 26196738); this file adds the phase gate: the
BF16 and the decoded 4-bit weights are both kept, and State.call copies the set of the phase into the Linears.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from fw_common import bf16_runtime, capture, guard, max_abs, read, save
import calib
import laq
import methods as mt


def arms_of(steps):
    return ["full", "video", "reader"] + [f"v.{s}" for s in range(steps)] + ["prefill"] + [f"a.{t}" for t in range(steps)]


def active(arm, kind, step):
    if arm == "full":
        return True
    if arm == "video":
        return kind == "video"
    if arm == "reader":
        return kind != "video"
    if arm == "prefill":
        return kind == "prefill"
    head, index = arm.split(".")
    return kind == ("video" if head == "v" else "action") and step == int(index)


class Gate:
    """Two weight sets for the block-level Linears; `set` installs one and switches the operand quantizers with it."""

    def __init__(self, st, modules, torch):
        self.st, self.torch = st, torch
        self.modules = [m for n, m in modules.items() if mt.parse(n) is not None]
        self.sets, self.mode, self.arm = {}, None, "full"
        self.switches = 0

    def store(self, quant):
        torch = self.torch
        need = sum(m.weight.numel() * m.weight.element_size() for m in self.modules)
        free = torch.cuda.mem_get_info()[0]
        on_gpu = free > need + (6 << 30)
        copies = []
        for m in self.modules:
            w = m.weight.detach()
            copies.append(w.clone() if on_gpu else w.cpu().pin_memory())
        self.sets[quant] = copies
        self.mode = quant
        return dict(quant=quant, gib=need / 2 ** 30, free_gib=free / 2 ** 30, on_gpu=on_gpu)

    def set(self, quant):
        if quant == self.mode:
            return
        with self.torch.no_grad():
            for m, w in zip(self.modules, self.sets[quant]):
                m.weight.copy_(w, non_blocking=True)
        self.st.quant = quant
        self.st.shared = None
        self.mode = quant
        self.switches += 1

    def call(self, kind, step, fn, kwargs):
        self.set(active(self.arm, kind, step))
        return fn(**kwargs)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


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
    gate = Gate(st, modules, torch)

    checks = {}
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    action, future = capture(m, runtime)
    checks["bf16_vs_a16_anchor"] = dict(action=max_abs(action, reference["action"]), future=max_abs(future, reference["future"]))
    if any(checks["bf16_vs_a16_anchor"].values()):
        raise RuntimeError(f"BF16 with the hooks differs from the A16 anchor: {checks}")
    metric = laq.Metric(torch.load(c["metric"], map_location="cpu", weights_only=True), device)
    checks["store_bf16"] = gate.store(False)

    index_of = {r["module"]: r["index"] for r in read(prep / "modules.json")}
    loaded = laq.load_weights(st, modules, index_of, Path(c["bank"]))
    st.use(laq.spec_none(c["steps"]))
    action, future = capture(m, runtime)
    plain = torch.load(c["plain_anchor"], map_location="cpu", weights_only=True)
    checks["plain_vs_quarot_anchor"] = dict(action=max_abs(action, plain["action"]), future=max_abs(future, plain["future"]))
    if loaded != m["targets"] or any(checks["plain_vs_quarot_anchor"].values()):
        raise RuntimeError(f"4-bit with nothing refined differs from the QuaRot anchor: {checks}")
    checks["store_w4"] = gate.store(True)
    st.call = gate.call
    save(out_root / f"startup_{job}.json", checks)
    print("PERSTEP_CHECKS " + json.dumps(checks), flush=True)

    plan_path = Path(m["panel"]["calibration_plan"])
    plan = read(plan_path)
    rows = {r["case_id"]: r for r in plan["inputs"]}
    contexts = plan["calibration_cohorts"]["c80"][:c["contexts"]]
    if contexts != read(laq_calib / "allocation.json")["contexts"]:
        raise RuntimeError("The contexts differ from the LAQ calibration's")
    teacher = {k: torch.load(laq_calib / "teacher" / f"{k}.pt", map_location="cpu", weights_only=True) for k in contexts}
    print(f"PERSTEP contexts={contexts} load={time.monotonic() - started:.0f}s", flush=True)
    st.use(calib.spec_of("base0", c, m))

    for arm in arms_of(c["steps"]):
        for case in contexts:
            path = out_root / "units" / arm / f"{case}.json"
            if path.exists():
                continue
            begin = time.monotonic()
            observation, metadata = calibration_source.load_observation(np, plan_path.parent, rows[case])
            runtime.description = metadata["description"]
            st.seed = c["seed"] + case
            st.reset()
            gate.arm, gate.switches = arm, 0
            action, _, future, calls = _capture_anchor(runtime, runtime.datum(observation), rows[case]["sampler_seeds"][0], torch)
            if calls != 1:
                raise RuntimeError("A query called infer_action more than once")
            record = dict(laq.errors(metric, action.detach().float().cpu(), future.float().cpu(), teacher[case]), arm=arm,
                          case=case, switches=gate.switches, seconds=time.monotonic() - begin, pbs_jobid=os.environ["PBS_JOBID"])
            save(path, record)
            print(f"UNIT {arm} case={case} A={record['A']:.4e} Z={record['Z']:.4e} sw={gate.switches} "
                  f"t={record['seconds']:.1f}s", flush=True)
    print(f"PERSTEP_DONE elapsed={time.monotonic() - started:.0f}s", flush=True)


def finalize(p, s):
    out_root = Path(s["remote_root"]) / "results"
    laq_calib = Path(s["source"]["calib"])
    steps = p["laq"]["steps"]
    contexts = read(laq_calib / "allocation.json")["contexts"]
    arms = arms_of(steps)
    unit = {a: [read(out_root / "units" / a / f"{k}.json") for k in contexts] for a in arms}
    stored = [read(laq_calib / "units" / "base0" / f"{k}.json") for k in contexts]
    A = {a: mean(u["A"] for u in v) for a, v in unit.items()}
    Z = {a: mean(u["Z"] for u in v) for a, v in unit.items()}
    ZM = {a: mean(u["ZM"] for u in v) for a, v in unit.items()}
    vs, ats = [f"v.{i}" for i in range(steps)], [f"a.{i}" for i in range(steps)]
    sum_v, sum_a = sum(A[a] for a in vs), sum(A[a] for a in ats)
    derived = dict(
        per_step_future_share=sum_v / (sum_v + A["prefill"] + sum_a),
        sum_video_steps_A=sum_v, prefill_A=A["prefill"], sum_action_steps_A=sum_a,
        accumulation_A=A["video"] / sum_v,
        accumulation_Z=Z["video"] / sum(Z[a] for a in vs),
        accumulation_ZM=ZM["video"] / sum(ZM[a] for a in vs),
        video_over_full_A=A["video"] / A["full"], reader_over_full_A=A["reader"] / A["full"],
        measured_future_share=1 - A["reader"] / A["full"],
        max_abs_A_full_vs_laq_base0=max(abs(x["A"] - y["A"]) for x, y in zip(unit["full"], stored)))
    save(out_root / "summary.json", dict(protocol=s["protocol"], contexts=contexts, A=A, Z=Z, ZM=ZM, derived=derived,
                                         units={a: unit[a] for a in arms}))
    lines = ["# Fast-WAM W4A4KV4：只让一个阶段跑 4-bit，其余阶段 BF16", "",
             f"上下文 {len(contexts)} 个（LAQ 校准的 C80 前 16 个），对 LAQ 存下的 BF16 teacher；底座是 base0（action 行精修）。", "",
             "| 4-bit 的阶段 | A | future NMSE (Z) | Z_M |", "|---|---:|---:|---:|"]
    for a in arms:
        lines.append(f"| {a} | {A[a]:.4e} | {Z[a]:.4e} | {ZM[a]:.4e} |")
    d = derived
    lines += ["", f"- 逐步相加：video 各步 {d['sum_video_steps_A']:.4e}，prefill {d['prefill_A']:.4e}，action 各步 {d['sum_action_steps_A']:.4e}",
              f"- 逐步估计给 future 通路的份额：{100 * d['per_step_future_share']:.1f}%",
              f"- 实测份额 1 − A(reader)/A(full)：{100 * d['measured_future_share']:.1f}%",
              f"- 10 个 video 步一起量化 / 逐步相加：A {d['accumulation_A']:.2f}×，Z {d['accumulation_Z']:.2f}×，Z_M {d['accumulation_ZM']:.2f}×",
              f"- A(video)/A(full) = {d['video_over_full_A']:.3f}，A(reader)/A(full) = {d['reader_over_full_A']:.3f}",
              f"- full 与 LAQ 校准 base0 单元的 A 最大差：{d['max_abs_A_full_vs_laq_base0']:.3e}"]
    (out_root / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(out_root / "COMPLETE.json", dict(complete=True, derived=derived))


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
