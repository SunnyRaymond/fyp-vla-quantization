"""Aggregate one closed-loop arm on the 280 panel, with paired references and the historical arm totals."""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import statistics as stats

from common import read, results, save

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def load_reference(pattern, arm):
    found = {}
    for path in sorted(glob.glob(pattern)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("arm") == arm and r.get("complete", True):
                    found[int(r["index"])] = r
    return found


def paired(new, ref):
    common = set(new) & set(ref)
    wins = sum(new[i]["success"] and not ref[i]["success"] for i in common)
    losses = sum(ref[i]["success"] and not new[i]["success"] for i in common)
    return dict(slots=len(common), wins=wins, losses=losses, ties=len(common) - wins - losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    p, arm = read(args.protocol), args.arm
    root = results(p) / "loops" / arm
    rows = read(p["panel"]["manifest"])["variants"]
    episodes = {int(f.stem): read(f) for f in (root / "episodes").glob("*.json")}
    quant = read(results(p) / "quantized" / arm / "COMPLETE.json")
    by_suite = Counter(rows[i]["suite"] for i, r in episodes.items() if r["success"])
    by_dim = Counter(rows[i]["dimension"] for i, r in episodes.items() if r["success"])
    dims = sorted({r["dimension"] for r in rows})
    refs = {name: load_reference(s["episodes"], s["arm"]) for name, s in p["references"]["paired"].items()}
    history = read(p["references"]["totals_summary"])
    shown = p["references"]["table_arms"]
    calls = [c for r in episodes.values() for c in r["policy_call_seconds"]]
    startup = [read(f) for f in sorted(root.glob("worker_*/startup.json"))]
    summary = dict(
        protocol=p["protocol"], arm=arm, pbs_jobid=os.environ.get("PBS_JOBID"), method=quant["method"],
        bits=quant["bits"], bpw=quant["bpw"], proxy_nmse_weighted=quant["proxy_nmse_weighted"],
        successes=sum(r["success"] for r in episodes.values()), episodes=len(episodes),
        complete=len(episodes) == 280, suites={s: by_suite[s] for s in SUITES}, dimensions={d: by_dim[d] for d in dims},
        paired={name: paired(episodes, ref) for name, ref in refs.items()},
        reference_totals={a: history["overall"]["arms"][a]["successes"] for a in shown},
        reference_dimensions={d: {a: history["dimensions"][d]["arms"][a]["successes"] for a in shown} for d in dims},
        anchor=startup[0] if startup else None,
        policy_call_p50_s=stats.median(calls) if calls else None, policy_calls=len(calls),
        episode_seconds_mean=stats.fmean(r["episode_seconds"] for r in episodes.values()) if episodes else None)
    save(root / "summary.json", summary)
    lines = [f"# LIBERO-Plus 280：`{arm}`（{quant['method']} W{quant['bits']}，A16，{quant['bpw']:.3f} bpw）\n\n",
             f"完成 {len(episodes)}/280，成功 **{summary['successes']}**。"
             f"权重误差代理（加权）{quant['proxy_nmse_weighted']:.5f}。"
             + (f"policy call p50 {summary['policy_call_p50_s']:.3f} s，episode 平均 "
                f"{summary['episode_seconds_mean']:.1f} s。\n\n" if episodes else "\n\n"),
             "| 臂 | 成功/280 |\n|---|---:|\n", f"| **{arm}** | **{summary['successes']}** |\n"]
    lines += [f"| {a} | {summary['reference_totals'][a]} |\n" for a in shown]
    lines += ["\n| 扰动（每组 40） | " + arm + " | " + " | ".join(shown) + " |\n|---|---:|" + "---:|" * len(shown) + "\n"]
    lines += [f"| {d} | {by_dim[d]} | " + " | ".join(str(summary["reference_dimensions"][d][a]) for a in shown)
              + " |\n" for d in dims]
    lines += ["\n| 配对对象 | 胜 | 负 | 平 |\n|---|---:|---:|---:|\n"]
    lines += [f"| {n} | {v['wins']} | {v['losses']} | {v['ties']} |\n" for n, v in summary["paired"].items()]
    (root / "RESULTS.zh.md").write_text("".join(lines), encoding="utf-8")
    print(f"ANALYSIS arm={arm} done={len(episodes)}/280 successes={summary['successes']}", flush=True)


if __name__ == "__main__":
    main()
