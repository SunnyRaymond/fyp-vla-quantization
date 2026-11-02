"""Closed-loop table of the Cosmos A16 baselines (stdlib; host python3.11): results/summary.json + RESULTS.zh.md.

BF16 panel A comes from the reused run (25838062) when the prep replay passed, otherwise from this job. Invalid slots
(task completed during settling) are dropped from every arm. Pairs (exact McNemar on the common valid slots): every
arm vs BF16, the hosts within each bit width, and W3 vs W2 of each method. Partial arms are reported as such.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

from common import arm_panels, read, results, save

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
OFFSET = {"A": 0, "B": 280, "C": 560}


def mcnemar(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def episodes(d):
    return {int(f.stem): read(f) for f in Path(d).glob("*.json")} if Path(d).exists() else {}


def panel_of(index):
    return "A" if index < 280 else "B" if index < 560 else "C"


def records(p, arm):
    """{index: record} over the arm's panels; BF16 panel A from the reused run unless this job ran it."""
    found = episodes(results(p) / "loops" / arm / "episodes")
    if arm == "BF16" and "A" not in arm_panels(p, arm):
        for index, r in episodes(p["reuse"]["BF16_A"]["episodes"]).items():
            found[index] = dict(r, invalid=False)
    return found


def pair(a, b, valid):
    common = set(a) & set(b) & valid
    wins = sum(a[i]["success"] and not b[i]["success"] for i in common)
    losses = sum(b[i]["success"] and not a[i]["success"] for i in common)
    return dict(slots=len(common), wins=wins, losses=losses, p=mcnemar(wins, losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    arms = p["arm_order"]
    recs = {arm: records(p, arm) for arm in arms}
    invalid = {i for r in recs.values() for i, e in r.items() if e.get("invalid")}
    every = set(range(840)) - invalid
    rows = {}
    for arm in arms:
        r = recs[arm]
        expected = {i for name in (["A", "B", "C"] if arm == "BF16" else arm_panels(p, arm))
                    for i in range(OFFSET[name], OFFSET[name] + 280)}
        valid = {i for i in r if i in every}
        quant = results(p) / "quantized" / arm / "COMPLETE.json"
        q = read(quant) if quant.exists() else {}
        start = results(p) / "loops" / arm / "anchor.pt"
        rows[arm] = dict(
            method=p["arms"][arm]["method"], bits=p["arms"][arm]["bits"], bpw=q.get("bpw", 16.0),
            proxy_nmse=q.get("proxy_nmse_weighted"), episodes=len(valid), expected=len(expected - invalid),
            complete=expected - invalid <= set(r), successes=sum(r[i]["success"] for i in valid),
            by_panel={name: sum(r[i]["success"] for i in valid if panel_of(i) == name) for name in OFFSET},
            n_by_panel={name: sum(panel_of(i) == name for i in valid) for name in OFFSET},
            by_suite={s: sum(r[i]["success"] for i in valid if r[i]["suite"] == s) for s in SUITES},
            by_dimension=dict(sorted(Counter(r[i]["dimension"] for i in valid if r[i]["success"]).items())),
            mean_episode_seconds=(sum(r[i].get("episode_seconds", 0.0) for i in valid) / len(valid)) if valid else None,
            has_anchor=start.exists())
    bf16 = recs["BF16"]
    vs_bf16 = {arm: pair(recs[arm], bf16, every) for arm in arms if arm != "BF16"}
    hosts = {}
    for bits in (2, 3):
        group = [a for a in arms if p["arms"][a]["bits"] == bits and p["arms"][a]["method"] != "gptq"]
        for a, b in combinations(group, 2):
            hosts[f"{a} vs {b}"] = pair(recs[a], recs[b], every)
    widths = {}
    for method in ("qtip", "quipsharp", "gptvq", "gptq"):
        w3 = next(a for a in arms if p["arms"][a]["method"] == method and p["arms"][a]["bits"] == 3)
        w2 = next(a for a in arms if p["arms"][a]["method"] == method and p["arms"][a]["bits"] == 2)
        widths[f"{w3} vs {w2}"] = pair(recs[w3], recs[w2], every)
    summary = dict(protocol=p["protocol"], invalid_slots=sorted(invalid), arms=rows, vs_bf16=vs_bf16,
                   hosts=hosts, w3_vs_w2=widths,
                   bf16_panel_a_source="this job" if "A" in arm_panels(p, "BF16") else p["reuse"]["BF16_A"]["job"])
    save(results(p) / "summary.json", summary)

    lines = ["# Cosmos-Policy A16 基线：LIBERO-Plus 闭环", "",
             f"协议 `{p['protocol']}`。无效槽位（settling 内完成）{len(invalid)} 个，已从所有臂剔除。"
             f"BF16 面板 A 来源：{summary['bf16_panel_a_source']}。", "",
             "| 臂 | bpw | 误差代理 | 成功 | 局数 | A | B | C | 对 BF16 胜/负 | p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        r = rows[arm]
        v = vs_bf16.get(arm)
        lines.append(f"| {arm}{'' if r['complete'] else '（未完成）'} | {r['bpw']:.3f} | "
                     f"{'—' if r['proxy_nmse'] is None else format(r['proxy_nmse'], '.4f')} | {r['successes']} | "
                     f"{r['episodes']} | " + " | ".join(
                         f"{r['by_panel'][n]}/{r['n_by_panel'][n]}" if r['n_by_panel'][n] else "—" for n in OFFSET)
                     + (f" | {v['wins']} / {v['losses']} | {v['p']:.3g} |" if v else " | — | — |"))
    lines += ["", "## 同位宽方法对比", "", "| 对 | 槽位 | 胜/负 | p |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['slots']} | {v['wins']} / {v['losses']} | {v['p']:.3g} |" for k, v in
              {**hosts, **widths}.items()]
    lines += ["", "## 按 suite 成功数", "", "| 臂 | " + " | ".join(SUITES) + " |", "|---|" + "---:|" * len(SUITES)]
    lines += [f"| {arm} | " + " | ".join(str(rows[arm]["by_suite"][s]) for s in SUITES) + " |" for arm in arms]
    dims = sorted({d for arm in arms for d in rows[arm]["by_dimension"]})
    lines += ["", "## 按扰动维度成功数", "", "| 臂 | " + " | ".join(dims) + " |", "|---|" + "---:|" * len(dims)]
    lines += [f"| {arm} | " + " | ".join(str(rows[arm]["by_dimension"].get(d, 0)) for d in dims) + " |" for arm in arms]
    (results(p) / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ANALYSIS " + " ".join(f"{a}={rows[a]['successes']}/{rows[a]['episodes']}" for a in arms), flush=True)


if __name__ == "__main__":
    main()
