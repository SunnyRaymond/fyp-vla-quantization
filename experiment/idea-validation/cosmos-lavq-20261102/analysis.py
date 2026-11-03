"""Closed-loop table of the LAVQ arms paired with the Cosmos A16 baselines (stdlib; host python3.11).

Episodes: results/loops/<arm>/episodes (this job) and the baselines' results/loops/<arm>/episodes; BF16 panel A from the
run the baselines reused. Invalid slots (task completed during settling) of any arm are dropped from every comparison.
Pairs (exact McNemar on the common valid slots): every LAVQ arm against BF16 and the three VQ hosts of its bit width.
Writes results/loop_summary.json and results/LOOP_RESULTS.zh.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from math import comb
from pathlib import Path

from common import read, results, save

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
OFFSET = {"A": 0, "B": 280, "C": 560}


def mcnemar(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def episodes(d):
    return {int(f.stem): read(f) for f in Path(d).glob("*.json")} if Path(d).exists() else {}


def panel_of(index):
    return "A" if index < 280 else "B" if index < 560 else "C"


def baseline_records(p, arm):
    base = Path(p["baseline_root"])
    found = episodes(base / "results" / "loops" / arm / "episodes")
    if arm == "BF16":
        bp = read(base / "protocol.json")
        for index, r in episodes(bp["reuse"]["BF16_A"]["episodes"]).items():
            found.setdefault(index, dict(r, invalid=False))
    return found


def pair(a, b, valid):
    common = set(a) & set(b) & valid
    wins = sum(a[i]["success"] and not b[i]["success"] for i in common)
    losses = sum(b[i]["success"] and not a[i]["success"] for i in common)
    return dict(slots=len(common), wins=wins, losses=losses, p=mcnemar(wins, losses))


def row(r, valid):
    v = {i for i in r if i in valid}
    return dict(episodes=len(v), successes=sum(r[i]["success"] for i in v),
                by_panel={n: sum(r[i]["success"] for i in v if panel_of(i) == n) for n in OFFSET},
                by_suite={s: sum(r[i]["success"] for i in v if r[i]["suite"] == s) for s in SUITES},
                by_dimension=dict(sorted(Counter(r[i]["dimension"] for i in v if r[i]["success"]).items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    ours = {arm: episodes(results(p) / "loops" / arm / "episodes") for arm in p["arm_order"]}
    refs = {"BF16": baseline_records(p, "BF16")}
    for arm in p["arm_order"]:
        b = arm[-1]
        for host in ("QTIP", "QUIPSHARP", "GPTVQ"):
            refs.setdefault(f"{host}_W{b}", baseline_records(p, f"{host}_W{b}"))
    every_rec = {**ours, **refs}
    invalid = {i for r in every_rec.values() for i, e in r.items() if e.get("invalid")}
    valid = set(range(840)) - invalid
    rows = {arm: row(r, valid) for arm, r in every_rec.items()}
    for arm in ours:
        q = results(p) / "quantized" / arm / "COMPLETE.json"
        rows[arm]["bpw"] = read(q)["bpw"] if q.exists() else None
        rows[arm]["complete"] = valid <= set(ours[arm])
    pairs = {f"{arm} vs {ref}": pair(ours[arm], refs[ref], valid)
             for arm in ours for ref in refs if ref == "BF16" or ref.endswith(arm[-3:])}
    summary = dict(protocol=p["protocol"], invalid_slots=sorted(invalid), arms=rows, pairs=pairs)
    save(results(p) / "loop_summary.json", summary)
    order = list(ours) + list(refs)
    lines = ["# Cosmos-Policy LAVQ：LIBERO-Plus 闭环（与 A16 基线配对）", "",
             f"无效槽位 {len(invalid)} 个，已从所有臂剔除；有效 {len(valid)}。", "",
             "| 臂 | 成功 | 局数 | A | B | C | " + " | ".join(SUITES) + " |", "|---|---:|---:|---:|---:|---:|" + "---:|" * len(SUITES)]
    for arm in order:
        r = rows[arm]
        lines.append(f"| {arm} | {r['successes']} | {r['episodes']} | " + " | ".join(str(r["by_panel"][n]) for n in OFFSET)
                     + " | " + " | ".join(str(r["by_suite"][s]) for s in SUITES) + " |")
    lines += ["", "| 对 | 槽位 | 胜/负 | p |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['slots']} | {v['wins']} / {v['losses']} | {v['p']:.3g} |" for k, v in pairs.items()]
    dims = sorted({d for arm in order for d in rows[arm]["by_dimension"]})
    lines += ["", "| 臂 | " + " | ".join(dims) + " |", "|---|" + "---:|" * len(dims)]
    lines += [f"| {arm} | " + " | ".join(str(rows[arm]["by_dimension"].get(d, 0)) for d in dims) + " |" for arm in order]
    (results(p) / "LOOP_RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ANALYSIS " + " ".join(f"{a}={rows[a]['successes']}/{rows[a]['episodes']}" for a in order), flush=True)


if __name__ == "__main__":
    main()
