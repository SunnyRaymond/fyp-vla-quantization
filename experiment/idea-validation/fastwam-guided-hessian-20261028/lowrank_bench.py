"""Policy-call latency and anchor numerics of one low-rank install mode (lowrank_loop.py --lr-mode) on the closed-loop
runtime: the loop worker's build_runtime (its T5-cache, install and cache + CUDA Graph parity checks), then --warmup
untimed and --calls timed runtime.infer calls on the case-2000 query, each synchronized, alone on the GPU. Writes
<out>/bench.json and the graph-path anchor <out>/anchor.pt.

--compare DIR reads DIR/<mode>/ for every mode and writes DIR/summary.json: latency against the host alone, anchor
deviation of fused and merged from the exact branch, and the mode the loop adopts (lowrank.json loop "bench").
"""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import lowrank_loop as ll
from common import anchor_datum, read, save


def bench(argv):
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(ll.configure(argv))
    p = read(args.protocol)
    runtime = ll.lw.build_runtime(p, args.arm, args.out)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    seconds = []
    for i in range(args.warmup + args.calls):
        datum, seed = anchor_datum(p, runtime)
        sync()
        begin = time.perf_counter()
        runtime.infer(datum, seed, measure=False)
        sync()
        if i >= args.warmup:
            seconds.append(time.perf_counter() - begin)
    action, future = ll.lw.capture(p, runtime)
    torch.save(dict(action=action, future=future), args.out / "anchor.pt")
    save(args.out / "bench.json", dict(mode=ll.CFG["lr_mode"], arm=args.arm, calls=seconds,
                                       p50_s=statistics.median(seconds), mean_s=statistics.fmean(seconds),
                                       min_s=min(seconds), check=read(ll.CFG["lr_check"])))
    print(f"BENCH mode={ll.CFG['lr_mode']} p50={statistics.median(seconds):.4f}s calls={len(seconds)}", flush=True)


def compare(root, limit=ll.FAST_REL):
    """Adopts merged if its anchor deviation passes, else fused, else the exact branch. A fast mode that failed its
    install check has no bench.json; its row keeps the check's deviations and is not timed."""
    import torch
    rows = {m: read(root / m / "bench.json") for m in ll.MODES if (root / m / "bench.json").exists()}
    if not {"branch", "host"} <= set(rows):
        raise RuntimeError("The exact branch and the host must both be benched")
    anchors = {m: torch.load(root / m / "anchor.pt", map_location="cpu", weights_only=True) for m in rows}
    b, h = anchors["branch"], anchors["host"]
    out = dict(limit=limit, modes={})
    for m in ll.MODES:
        if m not in rows:
            check = read(root / m / "lowrank_check.json")
            out["modes"][m] = dict(timed=False, check_passed=check["passed"], passed=False,
                                   action_rel=check.get("fast_action_rel"), future_rel=check.get("fast_future_rel"))
            continue
        row = dict(timed=True, p50_s=rows[m]["p50_s"], mean_s=rows[m]["mean_s"],
                   vs_host=rows[m]["p50_s"] / rows["host"]["p50_s"], check_passed=rows[m]["check"]["passed"])
        if m in ("fused", "merged"):
            a = anchors[m]
            row.update(action_rel=ll.motor_rmse(a["action"], b["action"]) / ll.motor_rmse(b["action"], h["action"]),
                       future_rel=ll.distance(a["future"], b["future"]) / ll.distance(b["future"], h["future"]))
            row["passed"] = row["check_passed"] and row["action_rel"] <= limit and row["future_rel"] <= limit
        out["modes"][m] = row
    out["adopt"] = next((m for m in ("merged", "fused") if out["modes"][m]["passed"]), "branch")
    save(root / "summary.json", out)
    for m, row in out["modes"].items():
        dev = (f" action_rel={row['action_rel']:.4f} future_rel={row['future_rel']:.4f}"
               if row.get("action_rel") is not None else "")
        timing = f" p50={row['p50_s']:.4f}s x{row['vs_host']:.3f}" if row["timed"] else " untimed (check failed)"
        print(f"MODE {m}{timing}{dev}", flush=True)
    print(f"ADOPT {out['adopt']}", flush=True)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--compare"]:
        compare(Path(argv[1]))
    else:
        bench(argv)


if __name__ == "__main__":
    main()
