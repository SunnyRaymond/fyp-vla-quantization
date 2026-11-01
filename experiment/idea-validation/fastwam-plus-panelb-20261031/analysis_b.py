"""Panel-B aggregation and the 560-episode (panel A + panel B) comparison table. Stdlib only.

Panel A outcomes come from the A16 loops (quantized arms) and the rotation-baselines BF16 episodes; panel B outcomes
from this experiment. Invalid panel-B slots (task completed during settling) are dropped from every arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
from itertools import combinations
import json
from math import comb
import os
from pathlib import Path
import statistics as stats

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def mcnemar(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def pair(a, b):
    common = set(a) & set(b)
    wins = sum(a[i]["success"] and not b[i]["success"] for i in common)
    losses = sum(b[i]["success"] and not a[i]["success"] for i in common)
    return dict(slots=len(common), wins=wins, losses=losses, p=mcnemar(wins, losses))


def panel_a(p, arm):
    """{index: record} for panel A; BF16 from the rotation-baselines JSONL, the rest from the A16 loops."""
    cfg = p["panel_a"]
    if p["arms"][arm]["weights"] is None:
        found = {}
        for path in sorted(glob.glob(cfg["bf16_episodes"])):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("arm") == cfg["bf16_arm"] and r.get("complete", True):
                        found[int(r["index"])] = r
        return found
    return {int(f.stem): read(f) for f in (Path(cfg["loops"]) / p["arms"][arm]["weights"] / "episodes").glob("*.json")}


def panel_b(p, arm):
    d = Path(p["remote_root"]) / "results" / "loops" / arm / "episodes"
    return {int(f.stem): read(f) for f in d.glob("*.json")}


def counts(records, rows):
    suites = Counter(rows[i]["suite"] for i, r in records.items() if r["success"])
    dims = Counter(rows[i]["dimension"] for i, r in records.items() if r["success"])
    return suites, dims


def bpw(p, arm):
    source = p["arms"][arm]["weights"]
    if source is None:
        return 16.0
    return read(Path(p["a16"]["root"]) / "results" / "quantized" / source / "COMPLETE.json")["bpw"]


def analyse(p):
    root = Path(p["remote_root"]) / "results"
    arms = p["arm_order"]
    rows_b = {r["index"]: r for r in read(root / "panel" / "manifest.json")["variants"]}
    rows_a = {r["index"]: r for r in read(p["panel_a"]["manifest"])["variants"]}
    rows = {**rows_a, **rows_b}
    b_all = {arm: panel_b(p, arm) for arm in arms}
    for arm, rec in b_all.items():
        if set(rec) != set(rows_b):
            raise RuntimeError(f"{arm}: panel B has {len(rec)}/280 episodes")
    invalid = {arm: sorted(i for i, r in rec.items() if r.get("invalid")) for arm, rec in b_all.items()}
    dropped = sorted(set().union(*invalid.values()))
    consistent = all(v == dropped for v in invalid.values())
    b = {arm: {i: r for i, r in rec.items() if i not in dropped} for arm, rec in b_all.items()}
    a = {arm: panel_a(p, arm) for arm in arms}
    for arm, rec in a.items():
        if set(rec) != set(rows_a):
            raise RuntimeError(f"{arm}: panel A has {len(rec)}/280 episodes")
    both = {arm: {**a[arm], **b[arm]} for arm in arms}
    reference = next(arm for arm in arms if p["arms"][arm]["weights"] is None)
    dims = list(dict.fromkeys(r["dimension"] for r in rows_a.values()))
    summary = dict(protocol=p["protocol"], pbs_jobid=os.environ.get("PBS_JOBID"), reference=reference,
                   panel_b_valid=len(rows_b) - len(dropped), invalid_slots=dropped, invalid_consistent=consistent,
                   arms={}, pairs={})
    for arm in arms:
        calls = [c for r in b_all[arm].values() for c in r["policy_call_seconds"]]
        workers = sorted((root / "loops" / arm).glob("worker_*/summary.json"))
        sa, da = counts(a[arm], rows)
        sb, db = counts(b[arm], rows)
        entry = dict(
            bpw=bpw(p, arm),
            panel_a=dict(successes=sum(r["success"] for r in a[arm].values()), episodes=len(a[arm])),
            panel_b=dict(successes=sum(r["success"] for r in b[arm].values()), episodes=len(b[arm])),
            suites={s: dict(a=sa[s], b=sb[s]) for s in SUITES},
            dimensions={d: dict(a=da[d], b=db[d]) for d in dims},
            policy_call_p50_s=stats.median(calls) if calls else None,
            episode_seconds_mean=stats.fmean(r["episode_seconds"] for r in b[arm].values()) if b[arm] else None,
            workers=len(workers),
            load_seconds=[read(f)["load_and_anchor_seconds"] for f in workers])
        entry["total"] = dict(successes=entry["panel_a"]["successes"] + entry["panel_b"]["successes"],
                              episodes=entry["panel_a"]["episodes"] + entry["panel_b"]["episodes"])
        if arm != reference:
            entry["vs_reference"] = dict(a=pair(a[arm], a[reference]), b=pair(b[arm], b[reference]),
                                         total=pair(both[arm], both[reference]))
        summary["arms"][arm] = entry
        save(root / "loops" / arm / "summary.json", dict(arm=arm, **entry))
    quantized = [arm for arm in arms if arm != reference]
    for x, y in combinations(quantized, 2):
        summary["pairs"][f"{x}|{y}"] = dict(a=pair(a[x], a[y]), b=pair(b[x], b[y]), total=pair(both[x], both[y]))
    save(root / "summary.json", summary)
    (root / "RESULTS.zh.md").write_text(report(summary, arms, dims), encoding="utf-8")
    return summary


def report(s, arms, dims):
    ref = s["reference"]
    nb = s["panel_b_valid"]
    out = [f"# LIBERO-Plus 面板 B + 面板 A：每臂 {280 + nb} 个 episode\n\n",
           f"面板 B 有效 {nb}/280"
           + (f"（无效槽位 {s['invalid_slots']}，各臂一致：{s['invalid_consistent']}）" if s["invalid_slots"] else "")
           + "。配对检验为精确 McNemar（双侧）。\n\n",
           f"| 臂 | bpw | 面板 A /280 | 面板 B /{nb} | 合计 /{280 + nb} | 对 {ref} 胜/负（合计） | p |\n",
           "|---|---:|---:|---:|---:|---:|---:|\n"]
    for arm in arms:
        e = s["arms"][arm]
        v = e.get("vs_reference", {}).get("total")
        out.append(f"| {arm} | {e['bpw']:.3f} | {e['panel_a']['successes']} | {e['panel_b']['successes']} | "
                   f"**{e['total']['successes']}** | " + (f"{v['wins']} / {v['losses']} | {v['p']:.3g}" if v else "— | —")
                   + " |\n")
    out += ["\n| 配对 | 面板 A 胜/负 | 面板 B 胜/负 | 合计胜/负 | 合计 p |\n|---|---:|---:|---:|---:|\n"]
    for name, v in s["pairs"].items():
        out.append(f"| {name.replace('|', ' 对 ')} | {v['a']['wins']} / {v['a']['losses']} | "
                   f"{v['b']['wins']} / {v['b']['losses']} | {v['total']['wins']} / {v['total']['losses']} | "
                   f"{v['total']['p']:.3g} |\n")
    out += ["\n| 扰动（A+B） | " + " | ".join(arms) + " |\n|---|" + "---:|" * len(arms) + "\n"]
    for d in dims:
        out.append(f"| {d} | " + " | ".join(str(s["arms"][a]["dimensions"][d]["a"] + s["arms"][a]["dimensions"][d]["b"])
                                          for a in arms) + " |\n")
    out += ["\n| suite（A+B） | " + " | ".join(arms) + " |\n|---|" + "---:|" * len(arms) + "\n"]
    for suite in SUITES:
        out.append(f"| {suite} | " + " | ".join(str(s["arms"][a]["suites"][suite]["a"] + s["arms"][a]["suites"][suite]["b"])
                                              for a in arms) + " |\n")
    out += ["\n| 臂（面板 B） | policy call p50 | episode 平均 | worker 次数 | 加载耗时 |\n|---|---:|---:|---:|---|\n"]
    for arm in arms:
        e = s["arms"][arm]
        loads = ", ".join(f"{x:.0f}" for x in e["load_seconds"])
        out.append(f"| {arm} | {e['policy_call_p50_s']:.3f} s | {e['episode_seconds_mean']:.1f} s | {e['workers']} | "
                   f"{loads} s |\n")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    s = analyse(read(args.protocol))
    for arm, e in s["arms"].items():
        print(f"ANALYSIS arm={arm} A={e['panel_a']['successes']} B={e['panel_b']['successes']}/{e['panel_b']['episodes']} "
              f"total={e['total']['successes']}/{e['total']['episodes']}", flush=True)


if __name__ == "__main__":
    main()
