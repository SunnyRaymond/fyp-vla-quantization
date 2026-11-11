"""The integer LAQ runtime on Fast-WAM: memory, latency of one policy call and open-loop agreement with the
fake-quantized model that the closed loop evaluated. One GPU, one process, a handful of queries.

Order: build the LAQ arm as a closed-loop worker does (fake quantization, dense BF16 weights); record its outputs on
the query sequence for the LAQ spec and for no refinement; one query in BF16 arithmetic on the dense weights gives
the memory of a BF16 policy call; then the dense weights are freed, the packed 4-bit weights take over (int4.Engine)
and each spec runs eager and through CUDA graphs. A last pass keeps the weights unpacked at 8 bits (resident8).
Writes results/fastwam/bench/<job>/summary.json and RESULTS.zh.md.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import time

from fw_common import anchor_datum, guard, read, save
from graph_bench import run, stats

GIB = 2 ** 30


def errors(outs, reference):
    """First-10 motor RMSE and future NMSE of every query against the reference outputs."""
    motor = [float((a[:10, :6] - ra[:10, :6]).pow(2).mean().sqrt()) for (a, _), (ra, _) in zip(outs, reference)]
    future = [float((z - rz).pow(2).sum() / rz.pow(2).sum()) for (_, z), (_, rz) in zip(outs, reference)]
    return dict(motor_rmse=statistics.mean(motor), motor_rmse_max=max(motor), future_nmse=statistics.mean(future),
                future_nmse_max=max(future))


def kernel_checks(engine, torch):
    """Rotations against the original transforms and the kernels against the reference arithmetic, on real layers."""
    import int4
    st, ref = engine.st, int4.TorchOps()
    g = torch.Generator(device=st.device).manual_seed(7)
    out = dict(rotation_max_abs=0.0, code_mismatch=0.0, scale_rel=0.0, gemm_rel=0.0, kv_rel=0.0, layers=0)
    seen = set()
    for name, (xf, n) in engine.pending.items():
        layer = engine.layers[name]
        if (type(xf).__name__, layer.N, layer.K) in seen:
            continue
        seen.add((type(xf).__name__, layer.N, layer.K))
        x = torch.randn(40, n, generator=g, device=st.device) * torch.linspace(0.2, 3, 40, device=st.device)[:, None]
        fast = engine.xf(name, xf, n).apply(x.to(torch.bfloat16))
        slow = xf.apply(x.to(torch.bfloat16).float())
        out["rotation_max_abs"] = max(out["rotation_max_abs"], float((fast - slow).abs().max()))
        idx = torch.tensor([1, 5, 17, 39], device=st.device)
        for rows in (None, idx):
            c, s = engine.ops.quant(fast, rows)
            rc, rs = ref.quant(fast, rows)
            out["code_mismatch"] = max(out["code_mismatch"], float((c != rc).float().mean()))
            out["scale_rel"] = max(out["scale_rel"], float(((s - rs).abs() / rs).max()))
            y = engine.ops.gemm(rc, rs, rows, layer, engine.scratch, torch.bfloat16).float()
            want = ref.gemm(rc, rs, rows, layer, None, torch.bfloat16).float()
            out["gemm_rel"] = max(out["gemm_rel"], float((y - want).abs().max() / want.abs().max()))
        out["layers"] += 1
    r = torch.randn(6 * st.h, st.d, generator=g, device=st.device)
    for kind in ("K", "V"):
        got = engine.ops.kvq(r, 6, st.h, kind, torch.tensor([0, 4], device=st.device))
        want = ref.kvq(r, 6, st.h, kind, torch.tensor([0, 4], device=st.device))
        out["kv_rel"] = max(out["kv_rel"], float((got - want).abs().max() / want.abs().max()))
    if out["rotation_max_abs"] > 1e-2 or out["code_mismatch"] > 1e-3 or out["gemm_rel"] > 1e-2 or out["kv_rel"] > 1e-2:
        raise RuntimeError(f"Kernel checks failed: {out}")
    return out


def main():
    import torch
    import int4
    import laq
    import methods as mt
    from fastwam_loop import build
    from graphs import Graphs
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    b = p["speed"]
    source = read(Path(b["source_root"]) / "protocol.json")
    if source["protocol"] != b["source_protocol"]:
        raise RuntimeError(f"Source protocol {source['protocol']} differs from {b['source_protocol']}")
    m, arm, c = source["models"]["fastwam"], b["arm"], source["laq"]
    root = Path(b["source_root"]) / "results" / "fastwam" / "loops" / arm
    if not (root / "anchor.pt").exists():
        raise RuntimeError("The arm anchor of the LAQ loop is missing; this job never writes it")
    job = os.environ["PBS_JOBID"]
    out = Path(p["remote_root"]) / "results" / "fastwam" / "bench" / job.split(".")[0]
    started = time.monotonic()
    runtime, check = build(source, m, arm, out / "runtime", root)
    st = runtime.laq
    spec_laq, online = st.spec, st.online
    datum, seed0 = anchor_datum(m, runtime)
    n = b["int4"]
    groups = [[seed0 + i for i in range(n["queries"])], [seed0 + 100 + i for i in range(n["queries_after_reset"])]]
    specs = (("laq", spec_laq, online), ("plain", laq.spec_none(st.steps), False))
    prep = Path(m["a16_prep"])
    bf16 = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    bf16 = [(bf16["action"].float(), bf16["future"].float())]
    print(f"INT4 built arm={arm} load={time.monotonic() - started:.0f}s", flush=True)

    # ---- the fake-quantized model (what the closed loop ran)
    fake, results = {}, {}
    for name, spec, use_online in specs:
        st.use(spec, online=use_online)
        fake[name], seconds = run(runtime, datum, groups, torch)
        results[name] = dict(fake=dict(stats(seconds, 1), vs_bf16_query0=errors(fake[name][:1], bf16)))
        print(f"INT4 fake spec={name} median={results[name]['fake']['median_s']:.2f}s", flush=True)

    # ---- memory of a BF16 policy call: BF16 arithmetic on the dense weights
    st.use(laq.spec_none(st.steps), online=False)
    st.quant = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run(runtime, datum, [groups[0][:2]], torch)
    memory = dict(bf16_peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                  bf16_resident_gib=torch.cuda.memory_allocated() / GIB)
    st.quant = True

    # ---- the integer runtime
    modules = dict(runtime.q.modules)
    index_of = {r["module"]: r["index"] for r in read(prep / "modules.json")}
    dense = sum(mod.weight.numel() * mod.weight.element_size() for name, mod in modules.items()
                if mt.parse(name) is not None)
    engine = int4.Engine(st, int4.TritonOps())
    engine.pending = {}
    for name, module in modules.items():
        role = mt.parse(name)
        if role is not None:
            xf = mt.xf_for("quarot", role, module.in_features, st.arch, st.rotation_seed, None, st.device)
            engine.pending[name] = (xf, module.in_features)
    engine.install(modules, index_of, Path(c["bank"]))
    torch.cuda.empty_cache()
    memory.update(dense_target_weights_gib=dense / GIB, packed_weights_gib=engine.weight_bytes() / GIB,
                  scratch_gib=engine.scratch.numel() / GIB, int4_resident_gib=torch.cuda.memory_allocated() / GIB)
    checks = kernel_checks(engine, torch)
    print(f"INT4 installed layers={len(engine.layers)} resident={memory['int4_resident_gib']:.2f}GiB "
          f"(BF16 {memory['bf16_resident_gib']:.2f}GiB) checks={checks}", flush=True)

    def native(name, spec, use_online, label):
        st.use(spec, online=use_online)
        engine.reset()
        torch.cuda.reset_peak_memory_stats()
        eager, eager_s = run(runtime, datum, groups, torch)
        st.freeze()
        graphs = Graphs(st)
        graphs.install()
        engine.reset()
        try:
            graphed, graph_s = run(runtime, datum, groups, torch)
        finally:
            graphs.uninstall()
        row = dict(eager=stats(eager_s, 1), graph=dict(stats(graph_s, 1), regions=len(graphs.regions)),
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
                   graph_vs_eager=errors(graphed, eager), vs_fake=errors(graphed, fake[name]),
                   vs_bf16_query0=errors(graphed[:1], bf16))
        print(f"INT4 {label} spec={name} eager={row['eager']['median_s']:.3f}s graph={row['graph']['median_s']:.3f}s "
              f"peak={row['peak_allocated_gib']:.2f}GiB vs_fake_motor={row['vs_fake']['motor_rmse']:.4f} "
              f"vs_fake_future={row['vs_fake']['future_nmse']:.5f}", flush=True)
        del graphs
        st.static = None
        torch.cuda.empty_cache()
        return row

    for name, spec, use_online in specs:
        results[name]["int4"] = native(name, spec, use_online, "packed")

    # ---- the same runtime with the weights left unpacked at 8 bits
    for layer in engine.layers.values():
        if layer.w8 is None:
            layer.w8 = layer.dense()
    memory["resident8_gib"] = torch.cuda.memory_allocated() / GIB
    results["laq"]["resident8"] = native("laq", spec_laq, online, "resident8")

    blocks = len(runtime.model.video_expert.blocks)
    caches = {k: int4.cache_bytes(st, blocks, dict(video=s["obsKV"] + s["futKV"],
                                                   **{"text.video": st.shapes["context_tokens"] * s["text_video"],
                                                      "text.action": st.shapes["context_tokens"] * s["text_action"]}))
              for k, s, _ in specs}
    summary = dict(protocol=p["protocol"], pbs_jobid=job, arm=arm, gpu=torch.cuda.get_device_name(),
                   torch=torch.__version__, queries=sum(len(g) for g in groups), spec=spec_laq, memory=memory,
                   kernel_checks=checks, results=results, caches=caches, shapes=dict(st.shapes), blocks=blocks,
                   gemms=engine.gemms, startup=check, seconds=time.monotonic() - started)
    save(out / "summary.json", summary)

    q = summary["queries"]
    lines = ["# LAQ 整数运行时（Fast-WAM，1 张 A100，单进程）", "",
             f"作业 {job}，{summary['gpu']}，torch {summary['torch']}。case 2000 的观测，{q} 次查询"
             f"（{len(groups[0])} 次后 reset 一次，再 {len(groups[1])} 次；每次换采样种子），LAQ 用与闭环相同的在线 gaze。"
             "中位数不计第 1 次（录图模式的第 1 次包含录图）。", "",
             "## 一次 policy call", "",
             "| 设置 | 运行时 | eager 中位数 (ms) | 录图中位数 (ms) | 峰值显存 allocated (GiB) |", "|---|---|---:|---:|---:|"]
    labels = dict(laq="LAQ（15% 精修 + 在线 gaze）", plain="不精修")
    for name in ("laq", "plain"):
        f = results[name]["fake"]
        lines.append(f"| {labels[name]} | 假量化（闭环所用） | {1000 * f['median_s']:.0f} | — | — |")
        for key, label in (("int4", "整数，权重 4 bit 常驻"), ("resident8", "整数，权重解包成 8 bit 常驻")):
            r = results[name].get(key)
            if r:
                lines.append(f"| {labels[name]} | {label} | {1000 * r['eager']['median_s']:.0f} | "
                             f"{1000 * r['graph']['median_s']:.0f} | {r['peak_allocated_gib']:.2f} |")
    lines += ["", "## 显存", "",
              f"- BF16 算术、dense 权重的一次调用：峰值 {memory['bf16_peak_allocated_gib']:.2f} GiB，常驻 {memory['bf16_resident_gib']:.2f} GiB",
              f"- 目标 Linear 的 BF16 权重 {memory['dense_target_weights_gib']:.2f} GiB → 4-bit 码加每行 scale 与 bias "
              f"{memory['packed_weights_gib']:.2f} GiB；INT8 scratch {memory['scratch_gib']:.3f} GiB",
              f"- 整数运行时常驻 {memory['int4_resident_gib']:.2f} GiB（解包成 8 bit 常驻时 {memory['resident8_gib']:.2f} GiB）",
              f"- 一次查询的 K/V 缓存（按 token 数计算，{blocks} 层，文本 {st.shapes['context_tokens']} token）：BF16 "
              f"{caches['laq']['bf16'] / 2 ** 20:.1f} MiB，4-bit 码 {caches['laq']['int4'] / 2 ** 20:.1f} MiB"
              f"（不精修 {caches['plain']['int4'] / 2 ** 20:.1f} MiB）。运行时里缓存以反量化后的 BF16 张量保存，这两个数是存储格式的大小。",
              "", "## 开环一致性", "",
              "| 设置 | 对照 | first-10 motor RMSE（均值 / 最大） | future NMSE（均值 / 最大） |", "|---|---|---:|---:|"]
    for name in ("laq", "plain"):
        r = results[name]
        rows = [("假量化 对 BF16（第 1 次查询）", r["fake"]["vs_bf16_query0"]),
                ("整数 对 BF16（第 1 次查询）", r["int4"]["vs_bf16_query0"]),
                (f"整数 对 假量化（{q} 次查询）", r["int4"]["vs_fake"]),
                ("整数录图 对 整数 eager", r["int4"]["graph_vs_eager"])]
        for label, e in rows:
            lines.append(f"| {labels[name]} | {label} | {e['motor_rmse']:.4f} / {e['motor_rmse_max']:.4f} | "
                         f"{e['future_nmse']:.5f} / {e['future_nmse_max']:.5f} |")
    lines += ["", f"kernel 对参考算术：旋转最大绝对差 {checks['rotation_max_abs']:.2e}，码不一致比例 {checks['code_mismatch']:.2e}，"
                  f"GEMM 输出最大相对差 {checks['gemm_rel']:.2e}，K/V 最大相对差 {checks['kv_rel']:.2e}（{checks['layers']} 种层形状）。"]
    (out / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("INT4_DONE " + " ".join(f"{k}:graph={v['int4']['graph']['median_s']:.3f}s" for k, v in results.items()), flush=True)


if __name__ == "__main__":
    main()
