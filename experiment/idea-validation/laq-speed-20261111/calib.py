"""LAQ calibration on Fast-WAM: read counts, class benefits, the allocation, and an open-loop check of each loop arm.

Workers (one per GPU, any number of jobs) share the work through claims on (arm, context) units:
  rc.<class>.<frozen|redrawn>   BF16 weights; the 4-bit rounding error of one frozen class with random signs, the sign
                                pattern either kept over the denoising steps or redrawn at every read -> read count
  plain                         4-bit weights, activations and caches, nothing refined (the QuaRot arm bit for bit)
  base0                         plain + the action rows refined
  cls.<class>                   base0 + one class refined entirely -> the damage the class removes
  eval.<arm>                    the allocation of a loop arm under the offline gaze
Objective of a unit: A + lambda_M Z_M against the BF16 teacher of the same context and sampler seed.
--finalize checks every unit and writes results/fastwam/calib/{summary.json, RESULTS.zh.md, COMPLETE.json}.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from fw_common import bf16_runtime, capture, guard, max_abs, read, save, torch_save
import laq

CLASSES = ("text.video", "text.action", "obsA", "obsKV", "futKV")


def root_of(p):
    return Path(p["remote_root"]) / "results" / "fastwam" / "calib"


def arms_of(p):
    steps = p["laq"]["steps"]
    rc = [f"rc.{c}.{mode}" for c in laq.FROZEN for mode in ("frozen", "redrawn")]
    quant = ["plain", "base0"] + [f"cls.{c}" for c in CLASSES] + [f"cls.fut.{s}" for s in range(steps + 1)]
    return rc, quant, [f"eval.{arm}" for arm in p["arms"]]


def spec_of(arm, c, m):
    steps, frame = c["steps"], m["frame_tokens"]
    fut = m["video_tokens"] - frame
    spec = laq.spec_none(steps)
    if arm == "plain":
        return spec
    spec["action"] = True
    name = arm[4:] if arm.startswith("cls.") else None
    if name == "text.video":
        spec["text_video"] = True
    elif name == "text.action":
        spec["text_action"] = True
    elif name in ("obsA", "obsKV"):
        spec[name] = frame
    elif name == "futKV":
        spec["futKV"] = fut
    elif name and name.startswith("fut."):
        spec["fut"][int(name[4:])] = fut
    elif arm != "base0":
        raise ValueError(arm)
    return spec


def claim(root, arm, case, stale):
    if (root / "units" / arm / f"{case}.json").exists():
        return False
    d = root / "claims" / arm / str(case)
    try:
        d.mkdir(parents=True)
    except FileExistsError:
        try:
            if time.time() - d.stat().st_mtime < stale:
                return False
            os.utime(d)
        except FileNotFoundError:
            return False
    return True


def missing(root, arms, contexts):
    return [(a, c) for a in arms for c in contexts if not (root / "units" / a / f"{c}.json").exists()]


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-30))


def allocate(p, root, contexts, names_and_widths, shapes, arch, pi):
    """Benefits, read counts, costs and the allocation of every loop arm, from the finished units."""
    import torch
    m, c = p["models"]["fastwam"], p["laq"]
    steps = c["steps"]
    rc, quant, _ = arms_of(p)
    unit = {a: [read(root / "units" / a / f"{k}.json") for k in contexts] for a in rc + quant}
    lam = mean(u["A"] for u in unit["plain"]) / mean(u["ZM"] for u in unit["plain"])

    def loss(arm):
        return mean(u["A"] + lam * u["ZM"] for u in unit[arm])
    benefit = {a[4:]: max(loss("base0") - loss(a), 0.0) for a in quant if a.startswith("cls.")}
    raw = {a[4:]: loss("base0") - loss(a) for a in quant if a.startswith("cls.")}
    counts = {}
    for cls in laq.FROZEN:
        f, r = unit[f"rc.{cls}.frozen"], unit[f"rc.{cls}.redrawn"]
        counts[cls] = dict(ZM=mean(u["ZM"] for u in f) / max(mean(u["ZM"] for u in r), 1e-30),
                           A=mean(u["A"] for u in f) / max(mean(u["A"] for u in r), 1e-30),
                           frozen=dict(A=mean(u["A"] for u in f), ZM=mean(u["ZM"] for u in f)),
                           redrawn=dict(A=mean(u["A"] for u in r), ZM=mean(u["ZM"] for u in r)))
    teachers = [torch.load(root / "teacher" / f"{k}.pt", map_location="cpu", weights_only=True) for k in contexts]
    att_fut = torch.stack([t["gaze_fut"] / t["gaze_fut"].sum() for t in teachers])
    pin = pi / pi.sum()
    hops = torch.stack([pin @ t["hop"] for t in teachers])
    g_obs = (hops / hops.sum(1, keepdim=True)).mean(0)
    k = max(1, pin.numel() // 4)
    top_pi = set(torch.topk(pin, k).indices.tolist())
    gaze = dict(spearman_attention_vs_metric=spearman(att_fut.mean(0), pin),
                per_context_spearman=mean(spearman(row, pin) for row in att_fut),
                top_quarter_overlap=mean(len(top_pi & set(torch.topk(row, k).indices.tolist())) / k for row in att_fut),
                frame_share_metric=[float(pin[:pin.numel() // 2].sum()), float(pin[pin.numel() // 2:].sum())],
                frame_share_attention=[float(att_fut.mean(0)[:pin.numel() // 2].sum()),
                                       float(att_fut.mean(0)[pin.numel() // 2:].sum())],
                direct_share_of_cache_attention=mean(float(t["gaze_direct"].sum() / (t["gaze_direct"].sum() + t["gaze_fut"].sum()))
                                                     for t in teachers))
    cost = laq.costs(names_and_widths, arch, steps, m["frame_tokens"], m["video_tokens"], shapes["action_tokens"],
                     shapes["context_tokens"])
    arms = {arm: laq.solve(benefit, cost, pin.tolist(), g_obs.tolist(), c["fractions"][arm], steps) for arm in p["arms"]}
    out = dict(protocol=p["protocol"], pbs_jobid=os.environ.get("PBS_JOBID"), contexts=contexts, lambda_M=lam,
               loss=dict(plain=loss("plain"), base0=loss("base0")), benefit=benefit, benefit_raw=raw, read_count=counts,
               cost=cost, gaze=gaze, offline_gaze=dict(fut=pin.tolist(), obs=g_obs.tolist()), arms=arms)
    return out


def work(p, worker):
    import numpy as np
    import torch
    import calibration_source
    from profile_worker import _capture_anchor
    guard()
    m, c = p["models"]["fastwam"], p["laq"]
    root = root_of(p)
    prep = Path(m["a16_prep"])
    deadline = float(os.environ.get("DEADLINE_EPOCH", time.time() + 10 ** 9))
    job = os.environ["PBS_JOBID"].split(".")[0]
    started = time.monotonic()
    runtime = bf16_runtime(root / f"runtime_{job}_{worker}", text_encoder=True)
    t5 = torch.load(prep / "t5_cache.pt", map_location="cpu", weights_only=True)
    encode = runtime.model.encode_prompt
    hits = [0, 0]

    def encode_prompt(prompt):  # cached prompts (the anchor, the panel) as the loops see them; the encoder for the rest
        if prompt in t5:
            hits[0] += 1
            emb, mask = t5[prompt]
            return emb.to(runtime.model.device), mask.to(runtime.model.device)
        hits[1] += 1
        return encode(prompt)
    runtime.model.encode_prompt = encode_prompt
    device = torch.device("cuda")
    arch = read(c["arch"])
    st = laq.State(c["steps"], m["frame_tokens"], m["video_tokens"], arch, m["rotation_seed"], device)
    modules = dict(runtime.q.modules)
    if laq.install(st, runtime.model, modules) != m["targets"]:
        raise RuntimeError("Block-level Linear count differs from the protocol")
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    action, future = capture(m, runtime)
    bf16 = dict(action=max_abs(action, reference["action"]), future=max_abs(future, reference["future"]))
    if bf16["action"] != 0 or bf16["future"] != 0:
        raise RuntimeError(f"BF16 with the LAQ hooks differs from the A16 anchor: {bf16}")
    metric_saved = torch.load(c["metric"], map_location="cpu", weights_only=True)
    metric = laq.Metric(metric_saved, device)
    pi = laq.token_gaze(metric_saved)
    if pi.numel() != st.video - st.frame:
        raise RuntimeError(f"The metric covers {pi.numel()} tokens, the future has {st.video - st.frame}")

    plan_path = Path(m["panel"]["calibration_plan"])
    plan = read(plan_path)
    rows = {r["case_id"]: r for r in plan["inputs"]}
    data = {}

    def datum(case):
        if case not in data:
            observation, metadata = calibration_source.load_observation(np, plan_path.parent, rows[case])
            data[case] = (observation, metadata["description"])
        observation, description = data[case]
        runtime.description = description
        return runtime.datum(observation), rows[case]["sampler_seeds"][0]

    contexts = plan["calibration_cohorts"]["c80"][:c["contexts"]]
    print(f"CALIB worker={worker} contexts={contexts} load={time.monotonic() - started:.0f}s", flush=True)

    def run(case):
        value, seed = datum(case)
        st.seed = c["seed"] + case
        st.reset()
        action, _, future, calls = _capture_anchor(runtime, value, seed, torch)
        if calls != 1:
            raise RuntimeError("A query called infer_action more than once")
        return action.detach().float().cpu(), future.float().cpu()

    st.watch = True
    teacher = {}
    for case in contexts:
        path = root / "teacher" / f"{case}.pt"
        if not path.exists():
            action, future = run(case)
            g = st.gaze
            torch_save(dict(case=case, action=action, future=future, gaze_fut=g["fut"].cpu(), gaze_direct=g["direct"].cpu(),
                            hop=g["hop"].cpu()), path)
        teacher[case] = torch.load(path, map_location="cpu", weights_only=True)
    st.watch = False

    def stage(arms, setup):
        while True:
            did = False
            for arm in arms:
                for case in contexts:
                    if time.time() > deadline - 240:
                        print(f"CALIB_DEADLINE worker={worker}", flush=True)
                        raise SystemExit(75)
                    if not claim(root, arm, case, c["stale_seconds"]):
                        continue
                    begin = time.monotonic()
                    setup(arm)
                    action, future = run(case)
                    record = dict(laq.errors(metric, action, future, teacher[case]), arm=arm, case=case,
                                  seconds=time.monotonic() - begin, pbs_jobid=os.environ["PBS_JOBID"], worker=worker)
                    save(root / "units" / arm / f"{case}.json", record)
                    print(f"UNIT {arm} case={case} A={record['A']:.3e} Z={record['Z']:.3e} ZM={record['ZM']:.3e} "
                          f"t={record['seconds']:.1f}s", flush=True)
                    did = True
            if not missing(root, arms, contexts):
                return
            if not did:
                if time.time() > deadline - 240:
                    raise SystemExit(75)
                time.sleep(20)

    rc, quant, evals = arms_of(p)

    def inject(arm):
        _, rest = arm.split(".", 1)
        cls, mode = rest.rsplit(".", 1)
        st.use(None)
        st.inject = (cls, mode)
    stage(rc, inject)
    st.inject = None

    index_of = {r["module"]: r["index"] for r in read(prep / "modules.json")}
    loaded = laq.load_weights(st, modules, index_of, Path(c["bank"]))
    st.seen.clear()
    st.counts.clear()
    st.use(laq.spec_none(c["steps"]))
    action, future = capture(m, runtime)
    plain = torch.load(c["plain_anchor"], map_location="cpu", weights_only=True)
    same = dict(action=max_abs(action, plain["action"]), future=max_abs(future, plain["future"]))
    sites = dict(st.counts)
    if len(st.seen) != m["targets"] or loaded != m["targets"] or not all(sites.get(s, 0) > 0 for s in ("text", "observation", "future")):
        raise RuntimeError(f"Coverage failed: {len(st.seen)} Linears, KV sites {sites}")
    if same["action"] != 0 or same["future"] != 0:
        raise RuntimeError(f"LAQ with nothing selected differs from the QuaRot anchor: {same}")
    save(root / f"startup_{job}_{worker}.json", dict(bf16_vs_a16_anchor=bf16, plain_vs_quarot_anchor=same, linears=len(st.seen),
                                                    kv_hits_per_query=sites, shapes=st.shapes, prompts=dict(cached=hits[0], encoded=hits[1]),
                                                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))
    stage(quant, lambda arm: st.use(spec_of(arm, c, m)))

    path = root / "allocation.json"
    if not path.exists():
        names = [(n, mod.in_features) for n, mod in modules.items()]
        save(path, allocate(p, root, contexts, names, st.shapes, arch, pi))
    alloc = read(path)
    st.offline = (torch.tensor(alloc["offline_gaze"]["fut"], device=device),
                  torch.tensor(alloc["offline_gaze"]["obs"], device=device))
    stage(evals, lambda arm: st.use(alloc["arms"][arm[5:]]["spec"]))
    print(f"CALIB_DONE worker={worker} elapsed={time.monotonic() - started:.0f}s", flush=True)


def finalize(p):
    root = root_of(p)
    alloc = read(root / "allocation.json")
    contexts = alloc["contexts"]
    rc, quant, evals = arms_of(p)
    left = missing(root, rc + quant + evals, contexts)
    if left:
        raise RuntimeError(f"{len(left)} units are missing, e.g. {left[:4]}")
    lam = alloc["lambda_M"]
    table = {}
    for arm in rc + quant + evals:
        u = [read(root / "units" / arm / f"{k}.json") for k in contexts]
        table[arm] = dict(A=mean(x["A"] for x in u), motor_rmse=mean(x["A"] for x in u) ** 0.5, Z=mean(x["Z"] for x in u),
                          ZM=mean(x["ZM"] for x in u), loss=mean(x["A"] + lam * x["ZM"] for x in u),
                          gripper_flips=sum(x["gripper_flips"] for x in u), seconds=mean(x["seconds"] for x in u))
    summary = dict(alloc, units=table, complete=True, finalized_by=os.environ.get("PBS_JOBID"))
    save(root / "summary.json", summary)
    lines = ["# LAQ 校准（Fast-WAM，BF16 teacher，开环）", "",
             f"上下文 {len(contexts)} 个（C80 的前 {len(contexts)} 个）；目标 A + λ_M·Z_M，λ_M = {lam:.4g}。", "",
             "## 各臂的开环误差（对 BF16 teacher）", "",
             "| 臂 | motor RMSE | future NMSE (Z) | Z_M | 目标 | 夹爪翻转 | 秒/次 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in quant + evals:
        t = table[arm]
        lines.append(f"| {arm} | {t['motor_rmse']:.4f} | {t['Z']:.4f} | {t['ZM']:.4f} | {t['loss']:.3e} | {t['gripper_flips']} | {t['seconds']:.1f} |")
    lines += ["", "## 每一类整类精修去掉的损伤（base0 = plain + action 行精修）", "",
              "| 类 | 目标下降 | 占 base0 目标的比例 |", "|---|---:|---:|"]
    for cls, b in alloc["benefit_raw"].items():
        lines.append(f"| {cls} | {b:.3e} | {100 * b / alloc['loss']['base0']:.1f}% |")
    lines += ["", "## 读次数（BF16 权重，只注入该类的 4-bit 误差；符号冻结 / 每次读取重抽）", "",
              "| 类 | n（Z_M） | n（A） | 冻结 Z_M | 重抽 Z_M |", "|---|---:|---:|---:|---:|"]
    for cls, r in alloc["read_count"].items():
        lines.append(f"| {cls} | {r['ZM']:.2f} | {r['A']:.2f} | {r['frozen']['ZM']:.3e} | {r['redrawn']['ZM']:.3e} |")
    lines += ["", "## 分配", "", "| 臂 | 预算 | 平均 A bit | 平均 KV bit | text(video/action) | obsA | obsKV | futKV | 各步 future 行（最后一项为 prefill） | μ |",
              "|---|---:|---:|---:|---|---:|---:|---:|---|---:|"]
    for arm, a in alloc["arms"].items():
        s, acc = a["spec"], a["accounting"]
        lines.append(f"| {arm} | {100 * acc['fraction']:.1f}% | {acc['a_bits']:.2f} | {acc['kv_bits']:.2f} | "
                     f"{int(s['text_video'])}/{int(s['text_action'])} | {s['obsA']} | {s['obsKV']} | {s['futKV']} | {s['fut']} | "
                     f"{'—' if a['mu'] is None else format(a['mu'], '.3e')} |")
    g = alloc["gaze"]
    lines += ["", "## gaze", "",
              f"- action 注意力给出的 future gaze 与读出度量 token 边缘的 Spearman：平均 gaze {g['spearman_attention_vs_metric']:.3f}，逐上下文平均 {g['per_context_spearman']:.3f}",
              f"- 前 25% token 的重合：{g['top_quarter_overlap']:.3f}",
              f"- 两个预测帧的占比：度量 {g['frame_share_metric'][0]:.3f} / {g['frame_share_metric'][1]:.3f}，注意力 {g['frame_share_attention'][0]:.3f} / {g['frame_share_attention'][1]:.3f}",
              f"- action 对缓存的注意力里直接读观测帧的比例：{g['direct_share_of_cache_attention']:.3f}"]
    (root / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(root / "COMPLETE.json", dict(complete=True, pbs_jobid=os.environ.get("PBS_JOBID")))
    print("CALIB_COMPLETE " + " ".join(f"{a}:A={table[a]['motor_rmse']:.4f},ZM={table[a]['ZM']:.4f}" for a in ["plain", "base0"] + evals), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    if args.finalize:
        finalize(p)
    else:
        work(p, args.worker)


if __name__ == "__main__":
    main()
