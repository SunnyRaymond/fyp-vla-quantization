"""840-episode table (panels A + B + C) of the RAVQ arms against BF16 and the W2/W3 QTIP, GPTVQ, QuIP# arms.
Stdlib only.

Ours: the arms this job ran, each over every panel: the panels it runs here from this job's loops, the rest from
reference_records (an uncalibrated RAVQU arm's panel A from its fastwam-guided-hessian-20261028 loop). References: panel A
from the A16 loops (BF16: the rotation-baselines JSONL), panel B from fastwam-plus-panelb-20261031 (BF16, W2) or
fastwam-plus-panelc-20261031 (W3), panel C from fastwam-plus-panelc-20261031; a RAVQU base this job did not run has
panel A only. A reference that has not finished a panel is paired on the slots both have and marked incomplete. Invalid
B/C slots (task completed during settling, the same for every arm) are dropped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from math import comb
import os
from pathlib import Path

import panel

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
PANELS = panel.PANELS


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def mcnemar(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def pair(a, b):
    common = set(a) & set(b)
    wins = sum(a[i]["success"] and not b[i]["success"] for i in common)
    losses = sum(b[i]["success"] and not a[i]["success"] for i in common)
    return dict(slots=len(common), wins=wins, losses=losses, p=mcnemar(wins, losses),
                a=sum(a[i]["success"] for i in common), b=sum(b[i]["success"] for i in common))


def episodes(d):
    return {int(f.stem): read(f) for f in Path(d).glob("*.json")}


def in_panel(found, name):
    offset = PANELS.index(name) * panel.PANEL
    return {i: r for i, r in found.items() if offset <= i < offset + panel.PANEL}


def bf16_panel_a(p):
    found = {}
    for path in sorted(glob.glob(p["panel_a"]["bf16_episodes"])):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("arm") == p["panel_a"]["bf16_arm"] and r.get("complete", True):
                    found[int(r["index"])] = r
    return found


def reference_records(p, arm, name):
    if arm.startswith("RAVQU_"):
        return episodes(Path(p["panel_a"]["ravq_loops"]) / arm / "episodes") if name == "A" else {}
    weights = p["references"][arm]["weights"]
    if name == "A":
        return bf16_panel_a(p) if weights is None else episodes(Path(p["panel_a"]["loops"]) / weights / "episodes")
    if name == "B" and not arm.endswith("_W3"):
        return episodes(Path(p["panel_b"]["loops"]) / arm / "episodes")
    return in_panel(episodes(Path(p["panel_c"]["loops"]) / arm / "episodes"), name)


def analyse(p):
    root = Path(p["remote_root"]) / "results"
    rows = {name: {int(r["index"]): r for r in panel.load(p, name)} for name in PANELS}
    allrows = {i: r for name in PANELS for i, r in rows[name].items()}
    ours = list(p["arm_order"])
    refs = list(p["references"]) + sorted({p["arms"][a]["base"] for a in ours if p["arms"][a]["base"]} - set(ours))
    raw = {}
    for arm in ours:
        found = episodes(root / "loops" / arm / "episodes")
        here = p["arms"][arm]["panels"]
        raw[arm] = {name: in_panel(found, name) if name in here else reference_records(p, arm, name) for name in PANELS}
        for name in PANELS:
            if set(raw[arm][name]) != set(rows[name]):
                raise RuntimeError(f"{arm}: panel {name} has {len(raw[arm][name])}/{len(rows[name])} episodes")
    for arm in refs:
        raw[arm] = {name: reference_records(p, arm, name) for name in PANELS}
    invalid = sorted({i for arm in raw for name in ("B", "C") for i, r in raw[arm][name].items() if r.get("invalid")})
    rec = {arm: {name: {i: r for i, r in raw[arm][name].items() if i not in invalid} for name in PANELS} for arm in raw}
    every = {arm: {i: r for name in PANELS for i, r in rec[arm][name].items()} for arm in raw}
    complete = {arm: {name: len(raw[arm][name]) == len(rows[name]) for name in PANELS} for arm in raw}
    dims = list(dict.fromkeys(r["dimension"] for r in rows["A"].values()))
    summary = dict(protocol=p["protocol"], pbs_jobid=os.environ.get("PBS_JOBID"), invalid_slots=invalid,
                   valid=len(allrows) - len(invalid), arms={}, pairs={}, complete=complete)
    for arm in raw:
        wins = [i for i, r in every[arm].items() if r["success"]]
        summary["arms"][arm] = dict(
            panels={name: dict(successes=sum(r["success"] for r in rec[arm][name].values()), episodes=len(rec[arm][name]))
                    for name in PANELS},
            total=dict(successes=len(wins), episodes=len(every[arm])),
            suites=dict(Counter(allrows[i]["suite"] for i in wins)),
            dimensions=dict(Counter(allrows[i]["dimension"] for i in wins)))
    for arm in ours:
        bits = p["arms"][arm]["bits"]
        targets = ["BF16"] + [r for r, c in p["references"].items() if c["bits"] == bits]
        targets += [p["arms"][arm]["base"]] if p["arms"][arm]["base"] else []
        for ref in targets:
            summary["pairs"][f"{arm}|{ref}"] = {**{name: pair(rec[arm][name], rec[ref][name]) for name in PANELS},
                                                "total": pair(every[arm], every[ref])}
        calls = [c for r in every[arm].values() for c in r.get("policy_call_seconds", [])]
        summary["arms"][arm]["policy_call_p50_s"] = sorted(calls)[len(calls) // 2] if calls else None
        save(root / "loops" / arm / "summary.json", dict(arm=arm, **summary["arms"][arm]))
    save(root / "summary.json", summary)
    (root / "RESULTS.zh.md").write_text(report(summary, ours, refs, dims), encoding="utf-8")
    return summary


def report(s, ours, refs, dims):
    out = [f"# RAVQ（RAVQU 未校准 / RAVQC 校准版）：LIBERO-Plus 面板 A + B + C（每臂 {s['valid']} 个有效 episode）\n\n",
           "配对检验为精确 McNemar（双侧），只在两臂都有结果的槽位上配对；参照臂某个面板没跑完时标注。"
           "RAVQU 的面板 A 来自 fastwam-guided-hessian-20261028 的闭环（同一 A16 运行时），B、C 在本作业跑。\n\n",
           "| 臂 | A | B | C | 合计 | 成功率 |\n|---|---:|---:|---:|---:|---:|\n"]
    for arm in ours + refs:
        e = s["arms"][arm]
        note = "" if all(s["complete"][arm].values()) else "（未全）"
        out.append(f"| {arm}{note} | " + " | ".join(str(e["panels"][k]["successes"]) for k in PANELS)
                   + f" | **{e['total']['successes']}** / {e['total']['episodes']} | "
                   + (f"{100 * e['total']['successes'] / e['total']['episodes']:.1f}%" if e['total']['episodes'] else "—")
                   + " |\n")
    out += ["\n| 配对 | 共同槽位 | 成功（我们 / 对照） | 胜 / 负 | p | A 胜/负 | B 胜/负 | C 胜/负 |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for name, t in s["pairs"].items():
        a, b = name.split("|")
        T = t["total"]
        out.append(f"| {a} 对 {b} | {T['slots']} | {T['a']} / {T['b']} | {T['wins']} / {T['losses']} | {T['p']:.3g} | "
                   + " | ".join(f"{t[k]['wins']} / {t[k]['losses']}" for k in PANELS) + " |\n")
    out += ["\n| 扰动（A+B+C） | " + " | ".join(ours + refs) + " |\n|---|" + "---:|" * len(ours + refs) + "\n"]
    for d in dims:
        out.append(f"| {d} | " + " | ".join(str(s["arms"][a]["dimensions"].get(d, 0)) for a in ours + refs) + " |\n")
    out += ["\n| suite（A+B+C） | " + " | ".join(ours + refs) + " |\n|---|" + "---:|" * len(ours + refs) + "\n"]
    for suite in SUITES:
        out.append(f"| {suite} | " + " | ".join(str(s["arms"][a]["suites"].get(suite, 0)) for a in ours + refs) + " |\n")
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
