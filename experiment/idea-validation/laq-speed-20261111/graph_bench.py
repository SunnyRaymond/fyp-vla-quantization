"""Speed of one Fast-WAM policy call, eager against CUDA graphs, for the LAQ arm and for its plain base (no refinement:
the QuaRot arm). One GPU, one process. Reads the LAQ experiment (allocation, arm anchor) and writes only under this
experiment's results/fastwam/bench/<job>/.

The arm is built as a closed-loop worker builds it (fastwam_loop.build on the LAQ experiment's protocol): BF16 parity
with the A16 anchor, then the arm anchor must equal the one the loop workers stored. For each spec (laq, then plain)
the same query sequence runs eager and then through graphs: case 2000, one sampler seed per query, two episodes'
worth (a reset between them puts the offline gaze back), online gaze as in the loop for laq. Every graph-mode action
and predicted future must equal the eager one bit for bit, else the job fails.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import time

from fw_common import anchor_datum, guard, read, save


def run(runtime, datum, groups, torch):
    st = runtime.laq
    outs, seconds = [], []
    for group in groups:
        st.reset()
        for seed in group:
            captured = []
            original = runtime.model.infer_action

            def capture(*args, **kwargs):
                out = original(*args, **kwargs)
                captured.append(out["video_latents"].detach().float().cpu())
                return out
            runtime.model.infer_action = capture
            try:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                action, _ = runtime.infer(datum, seed, measure=False)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - begin)
            finally:
                runtime.model.infer_action = original
            if len(captured) != 1:
                raise RuntimeError("A query called infer_action more than once")
            outs.append((action.detach().float().cpu(), captured[0]))
    return outs, seconds


def compare(name, outs, reference, torch):
    for i, ((a, z), (ra, rz)) in enumerate(zip(outs, reference)):
        if not (torch.equal(a, ra) and torch.equal(z, rz)):
            raise RuntimeError(f"{name} differs from eager at query {i}: action {float((a - ra).abs().max())}, "
                               f"future {float((z - rz).abs().max())}")


def stats(seconds, skip):
    timed = seconds[skip:]
    return dict(seconds=seconds, median_s=statistics.median(timed), min_s=min(timed), first_s=seconds[0])


def main():
    import torch
    import laq
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
    m, arm = source["models"]["fastwam"], b["arm"]
    root = Path(b["source_root"]) / "results" / "fastwam" / "loops" / arm
    if not (root / "anchor.pt").exists():
        raise RuntimeError("The arm anchor of the LAQ loop is missing; this job never writes it")
    job = os.environ["PBS_JOBID"]
    out = Path(p["remote_root"]) / "results" / "fastwam" / "bench" / job.split(".")[0]
    started = time.monotonic()
    runtime, check = build(source, m, arm, out / "runtime", root)
    st = runtime.laq
    spec_laq, online = st.spec, st.online
    print(f"GRAPH built arm={arm} anchor_motor_rmse={check['anchor_first10_motor_rmse']:.4f} "
          f"load={time.monotonic() - started:.0f}s", flush=True)
    datum, seed0 = anchor_datum(m, runtime)
    groups = [[seed0 + i for i in range(b["queries"])], [seed0 + 100 + i for i in range(b["queries_after_reset"])]]
    results = {}
    for name, spec, use_online in (("laq", spec_laq, online), ("plain", laq.spec_none(st.steps), False)):
        st.use(spec, online=use_online)
        torch.cuda.reset_peak_memory_stats()
        eager, eager_s = run(runtime, datum, groups, torch)
        eager_peak = torch.cuda.max_memory_reserved() / 2 ** 30
        st.freeze()
        graphs = Graphs(st)
        graphs.install()
        torch.cuda.reset_peak_memory_stats()
        try:
            graphed, graph_s = run(runtime, datum, groups, torch)
        finally:
            graphs.uninstall()
        compare(f"{name} graphs", graphed, eager, torch)
        results[name] = dict(eager=dict(stats(eager_s, 1), peak_reserved_gib=eager_peak),
                             graph=dict(stats(graph_s, 1), peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                                        **graphs.report()))
        r = results[name]
        print(f"GRAPH spec={name} eager={r['eager']['median_s']:.2f}s graph={r['graph']['median_s']:.2f}s "
              f"capture_query={r['graph']['first_s']:.1f}s peak={r['graph']['peak_reserved_gib']:.1f}GiB "
              f"regions={len(graphs.regions)}", flush=True)
        del graphs
        st.static = None
        torch.cuda.empty_cache()
    episodes = [read(f) for f in (root / "episodes").glob("*.json")]
    loop_calls = [s for e in episodes for s in e.get("policy_call_seconds", [])]
    summary = dict(protocol=p["protocol"], pbs_jobid=job, arm=arm, queries=sum(len(g) for g in groups),
                   bitwise_equal=True, results=results, spec=spec_laq,
                   loop_reference=dict(episodes=len(episodes), calls=len(loop_calls),
                                       median_s=statistics.median(loop_calls) if loop_calls else None,
                                       note="the running LAQ loop: old eager code, two workers per GPU"),
                   seconds=time.monotonic() - started)
    save(out / "summary.json", summary)
    n = summary["queries"]
    lines = ["# CUDA Graph 测速（Fast-WAM，1 张 A100，单进程）", "",
             f"作业 {job}。case 2000 的观测，{n} 次查询（{len(groups[0])} 次后 reset 一次，再 {len(groups[1])} 次；每次换采样种子），"
             "LAQ 用与闭环相同的在线 gaze。每种设置下，录图模式每一次查询的动作和预测 future 都与 eager 逐位相同。"
             "中位数不计第 1 次（录图模式的第 1 次包含预热和录图）。", "",
             "| 设置 | eager 中位数 (s) | 录图中位数 (s) | 加速 | 录图那次查询 (s) | eager 峰值显存 (GiB) | 录图峰值显存 (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("laq", "LAQ（15% 精修 + 在线 gaze）"), ("plain", "不精修（= QuaRot 底座）")):
        e, g = results[name]["eager"], results[name]["graph"]
        lines.append(f"| {label} | {e['median_s']:.2f} | {g['median_s']:.2f} | {e['median_s'] / g['median_s']:.2f}× | "
                     f"{g['first_s']:.1f} | {e['peak_reserved_gib']:.1f} | {g['peak_reserved_gib']:.1f} |")
    lr = summary["loop_reference"]
    if lr["median_s"] is not None:
        lines += ["", f"对照：正在跑的 LAQ 闭环（旧 eager 代码，每卡 2 个进程）{lr['episodes']} 局、{lr['calls']} 次调用的中位数 "
                      f"{lr['median_s']:.2f} s。"]
    lines += ["", f"录图模式下文本 K/V 每步在图里重算（eager 下由 Python 端缓存复用），两者数值相同。"]
    (out / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GRAPH_DONE " + " ".join(f"{k}:eager={v['eager']['median_s']:.2f}s,graph={v['graph']['median_s']:.2f}s"
                                   for k, v in results.items()), flush=True)


if __name__ == "__main__":
    main()
