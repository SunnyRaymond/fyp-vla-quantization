"""Summary of one model's standard-LIBERO 840 loop: analysis840.py --protocol P --model fastwam|cosmos

Writes results/<model>/summary.json and RESULTS.zh.md (works on a partial run: counts only the saved episodes).
95% interval: Wald on the episode success rate, the same as the LIBERO-Plus tables.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

import l840

NAMES = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}


def wald(k, n):
    if n == 0:
        return None
    p = k / n
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    P = l840.read(args.protocol)
    res = l840.results(P, args.model)
    rows = l840.manifest_rows(P, args.model)
    episodes = {}
    for r in rows:
        path = res / "episodes" / f"{r['index']}.json"
        if path.exists():
            episodes[r["index"]] = l840.read(path)
    valid = {i: e for i, e in episodes.items() if not e.get("invalid")}
    suites = P["benchmark"]["suites"]
    per_suite = {}
    for s in suites:
        xs = [e for i, e in valid.items() if e["suite"] == s]
        per_suite[s] = dict(successes=sum(e["success"] for e in xs), episodes=len(xs))
    k = sum(e["success"] for e in valid.values())
    n = len(valid)
    calls = [c for e in valid.values() for c in (e.get("policy_call_seconds")
                                                   or [q["query_seconds"] for q in e.get("queries", [])])]
    gpus = sorted({e.get("gpu", "?") for e in episodes.values()})
    summary = dict(protocol=P["protocol"], model=args.model, arm=P["models"][args.model]["arm"],
                   complete=len(episodes) == P["benchmark"]["episodes"], saved=len(episodes), invalid=len(episodes) - n,
                   successes=k, episodes=n, success_rate=100 * k / n if n else None, ci95=wald(k, n),
                   per_suite=per_suite, gpus=gpus,
                   policy_call_median_seconds=statistics.median(calls) if calls else None,
                   episode_seconds_mean=statistics.mean(e["episode_seconds"] for e in valid.values()) if n else None)
    l840.save(res / "summary.json", summary)
    lines = [f"# {args.model}：LAQ（W4A4KV4），标准 LIBERO 840 局", "",
             f"40 个任务 × 官方初始状态 0–20；已保存 {len(episodes)}/{P['benchmark']['episodes']} 局"
             + (f"，无效 {summary['invalid']} 局" if summary["invalid"] else "") + f"；GPU：{', '.join(gpus)}。", "",
             "| 臂 | 成功 | 局数 | 成功率 | ±95% | " + " | ".join(NAMES[s] for s in suites) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(suites)]
    cells = []
    for s in suites:
        d = per_suite[s]
        cells.append(f"{100 * d['successes'] / d['episodes']:.1f}" if d["episodes"] else "–")
    lines.append(f"| laq | {k} | {n} | " + (f"{summary['success_rate']:.1f} | {summary['ci95']:.1f}" if n else "– | –")
                 + " | " + " | ".join(cells) + " |")
    lines += ["", "## 各任务成功数（每任务 21 局）", "", "| suite | " + " | ".join(str(t) for t in range(10)) + " |",
              "|---|" + "---:|" * 10]
    for s in suites:
        cnt = []
        for t in range(10):
            xs = [e for e in valid.values() if e["suite"] == s and e["task_id"] == t]
            cnt.append(f"{sum(e['success'] for e in xs)}/{len(xs)}" if xs else "–")
        lines.append(f"| {NAMES[s]} | " + " | ".join(cnt) + " |")
    if calls:
        lines += ["", f"policy call 中位数 {summary['policy_call_median_seconds']:.2f} s，"
                      f"每局平均 {summary['episode_seconds_mean']:.0f} s。"]
    (res / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
