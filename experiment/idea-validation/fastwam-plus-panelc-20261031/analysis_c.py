"""840-episode table (panels A + B + C) for BF16 and the W2/W3 QuIP#, GPTVQ, QTIP arms. Stdlib only.

Panel A: A16 loops (quantized arms) and the rotation-baselines BF16 JSONL. Panel B: this job for the arms that list it
(W3), else fastwam-plus-panelb-20261031. Panel C: this job. Invalid B/C slots (task completed during settling) are
dropped from every arm. Pairs: every arm vs BF16, the hosts within each bit width, and W3 vs W2 of each method.
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

import panel

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
PANELS = ("A", "B", "C")


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


def episodes(d):
    return {int(f.stem): read(f) for f in Path(d).glob("*.json")}


def here(p, arm):
    return Path(p["remote_root"]) / "results" / "loops" / arm


def records(p, arm, name):
    """{index: record} of one arm on one panel, before invalid slots are dropped."""
    cfg = p["arms"][arm]
    if name == "A":
        if cfg["weights"] is None:
            found = {}
            for path in sorted(glob.glob(p["panel_a"]["bf16_episodes"])):
                for line in Path(path).read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        if r.get("arm") == p["panel_a"]["bf16_arm"] and r.get("complete", True):
                            found[int(r["index"])] = r
            return found
        return episodes(Path(p["panel_a"]["loops"]) / cfg["weights"] / "episodes")
    if name in cfg["panels"]:
        offset = p["panel_b" if name == "B" else "panel_c"]["index_offset"]
        return {i: r for i, r in episodes(here(p, arm) / "episodes").items() if offset <= i < offset + panel.PANEL}
    return episodes(Path(p["panel_b"]["loops"]) / arm / "episodes")


def bpw(p, arm):
    source = p["arms"][arm]["weights"]
    if source is None:
        return 16.0
    return read(Path(p["a16"]["root"]) / "results" / "quantized" / source / "COMPLETE.json")["bpw"]


def pairs_of(arms, reference):
    """(x, y) pairs: hosts within a bit width, then W3 vs W2 of the same method."""
    quantized = [a for a in arms if a != reference]
    out = []
    for bits in ("W2", "W3"):
        out += list(combinations([a for a in quantized if a.endswith("_" + bits)], 2))
    for a in quantized:
        if a.endswith("_W3") and a[:-1] + "2" in quantized:
            out.append((a, a[:-1] + "2"))
    return out


def analyse(p):
    root = Path(p["remote_root"]) / "results"
    arms = p["arm_order"]
    rows = {name: {int(r["index"]): r for r in panel.load(p, name)} for name in PANELS}
    allrows = {i: r for name in PANELS for i, r in rows[name].items()}
    raw = {arm: {name: records(p, arm, name) for name in PANELS} for arm in arms}
    for arm in arms:
        for name in PANELS:
            if set(raw[arm][name]) != set(rows[name]):
                raise RuntimeError(f"{arm}: panel {name} has {len(raw[arm][name])}/{len(rows[name])} episodes")
    invalid, consistent = {}, {}
    for name in ("B", "C"):
        per_arm = {arm: sorted(i for i, r in raw[arm][name].items() if r.get("invalid")) for arm in arms}
        invalid[name] = sorted(set().union(*per_arm.values()))
        consistent[name] = all(v == invalid[name] for v in per_arm.values())
    dropped = set(invalid["B"]) | set(invalid["C"])
    rec = {arm: {name: {i: r for i, r in raw[arm][name].items() if i not in dropped} for name in PANELS}
           for arm in arms}
    every = {arm: {i: r for name in PANELS for i, r in rec[arm][name].items()} for arm in arms}
    reference = next(arm for arm in arms if p["arms"][arm]["weights"] is None)
    dims = list(dict.fromkeys(r["dimension"] for r in rows["A"].values()))
    valid = {name: len(rows[name]) - len(set(invalid.get(name, []))) for name in PANELS}
    summary = dict(protocol=p["protocol"], pbs_jobid=os.environ.get("PBS_JOBID"), reference=reference,
                   valid=valid, invalid_slots=invalid, invalid_consistent=consistent, arms={}, pairs={})
    for arm in arms:
        ran = [r for name in p["arms"][arm]["panels"] for r in raw[arm][name].values() if not r.get("invalid")]
        calls = [c for r in ran for c in r["policy_call_seconds"]]
        workers = sorted(here(p, arm).glob("worker_*/summary.json"))
        wins = [i for i, r in every[arm].items() if r["success"]]
        entry = dict(
            bpw=bpw(p, arm),
            panels={name: dict(successes=sum(r["success"] for r in rec[arm][name].values()),
                               episodes=len(rec[arm][name])) for name in PANELS},
            ran_here=p["arms"][arm]["panels"],
            suites=dict(Counter(allrows[i]["suite"] for i in wins)),
            dimensions=dict(Counter(allrows[i]["dimension"] for i in wins)),
            policy_call_p50_s=stats.median(calls) if calls else None,
            episode_seconds_mean=stats.fmean(r["episode_seconds"] for r in ran) if ran else None,
            workers=len(workers),
            load_seconds=[read(f)["load_and_anchor_seconds"] for f in workers])
        entry["total"] = dict(successes=len(wins), episodes=len(every[arm]))
        if arm != reference:
            entry["vs_reference"] = {**{name: pair(rec[arm][name], rec[reference][name]) for name in PANELS},
                                     "total": pair(every[arm], every[reference])}
        summary["arms"][arm] = entry
        save(root / "loops" / arm / "summary.json", dict(arm=arm, **entry))
    for x, y in pairs_of(arms, reference):
        summary["pairs"][f"{x}|{y}"] = {**{name: pair(rec[x][name], rec[y][name]) for name in PANELS},
                                        "total": pair(every[x], every[y])}
    save(root / "summary.json", summary)
    (root / "RESULTS.zh.md").write_text(report(summary, arms, dims), encoding="utf-8")
    return summary


def report(s, arms, dims):
    ref, v = s["reference"], s["valid"]
    n = sum(v.values())
    inv = {k: x for k, x in s["invalid_slots"].items() if x}
    out = [f"# LIBERO-Plus 面板 A + B + C：每臂 {n} 个 episode\n\n",
           f"有效槽位 A {v['A']}、B {v['B']}、C {v['C']}"
           + (f"（无效槽位 {inv}，各臂一致：{s['invalid_consistent']}）" if inv else "")
           + "。配对检验为精确 McNemar（双侧）。\n\n",
           f"| 臂 | bpw | A /{v['A']} | B /{v['B']} | C /{v['C']} | 合计 /{n} | 成功率 | 对 {ref} 胜/负 | p |\n",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for arm in arms:
        e = s["arms"][arm]
        t = e.get("vs_reference", {}).get("total")
        out.append(f"| {arm} | {e['bpw']:.3f} | " + " | ".join(str(e["panels"][k]["successes"]) for k in PANELS)
                   + f" | **{e['total']['successes']}** | {100 * e['total']['successes'] / e['total']['episodes']:.1f}% | "
                   + (f"{t['wins']} / {t['losses']} | {t['p']:.3g}" if t else "— | —") + " |\n")
    out += [f"\n| 对 {ref} 分面板 | A 胜/负 | B 胜/负 | C 胜/负 |\n|---|---:|---:|---:|\n"]
    for arm in arms:
        t = s["arms"][arm].get("vs_reference")
        if t:
            out.append(f"| {arm} | " + " | ".join(f"{t[k]['wins']} / {t[k]['losses']}" for k in PANELS) + " |\n")
    out += ["\n| 配对 | A 胜/负 | B 胜/负 | C 胜/负 | 合计胜/负 | 合计 p |\n|---|---:|---:|---:|---:|---:|\n"]
    for name, t in s["pairs"].items():
        out.append(f"| {name.replace('|', ' 对 ')} | " + " | ".join(f"{t[k]['wins']} / {t[k]['losses']}" for k in PANELS)
                   + f" | {t['total']['wins']} / {t['total']['losses']} | {t['total']['p']:.3g} |\n")
    out += ["\n| 扰动（A+B+C） | " + " | ".join(arms) + " |\n|---|" + "---:|" * len(arms) + "\n"]
    for d in dims:
        out.append(f"| {d} | " + " | ".join(str(s["arms"][a]["dimensions"].get(d, 0)) for a in arms) + " |\n")
    out += ["\n| suite（A+B+C） | " + " | ".join(arms) + " |\n|---|" + "---:|" * len(arms) + "\n"]
    for suite in SUITES:
        out.append(f"| {suite} | " + " | ".join(str(s["arms"][a]["suites"].get(suite, 0)) for a in arms) + " |\n")
    out += ["\n| 臂（本作业） | 跑的面板 | policy call p50 | episode 平均 | worker 次数 | 加载耗时 |\n"
            "|---|---|---:|---:|---:|---|\n"]
    for arm in arms:
        e = s["arms"][arm]
        loads = ", ".join(f"{x:.0f}" for x in e["load_seconds"])
        out.append(f"| {arm} | {'+'.join(e['ran_here'])} | {e['policy_call_p50_s']:.3f} s | "
                   f"{e['episode_seconds_mean']:.1f} s | {e['workers']} | {loads} s |\n")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    s = analyse(read(args.protocol))
    for arm, e in s["arms"].items():
        print(f"ANALYSIS arm={arm} " + " ".join(f"{k}={e['panels'][k]['successes']}" for k in PANELS)
              + f" total={e['total']['successes']}/{e['total']['episodes']}", flush=True)


if __name__ == "__main__":
    main()
