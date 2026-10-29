"""Aggregate one arm: reproduction check against history, or the 280-panel table with paired references."""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import statistics as stats

from plus_engine import read, save
from worker import arm_slots

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def load_episodes(directory):
    return {int(f.stem): read(f) for f in Path(directory).glob("*.json")}


def load_reference(pattern, arm):
    found = {}
    for path in sorted(glob.glob(pattern)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("arm") == arm and r.get("complete", True):
                    found[int(r["index"])] = r
    return found


def repro(p, arm, cfg, episodes):
    history = load_reference(p["history"]["episodes"], p["history"]["arm"])
    rows = []
    for index in arm_slots(cfg):
        new, old = episodes.get(index), history.get(index)
        rows.append(dict(index=index, success=new and new["success"], steps=new and new["steps"],
                         history_success=old and old["success"], history_steps=old and old["steps"],
                         match=bool(new and old and new["success"] == old["success"]
                                    and new["steps"] == old["steps"])))
    passed = len(rows) == len(arm_slots(cfg)) and all(r["match"] for r in rows)
    return dict(repro_passed=passed, history_job=p["history"]["job"], history_arm=p["history"]["arm"],
                slots=rows), (
        f"# 结果复现检查 `{arm}`\n\n"
        f"与历史 `{p['history']['job']}` 的 `{p['history']['arm']}` 逐条比较成败与步数："
        f"**{'通过' if passed else '未通过'}**（{sum(r['match'] for r in rows)}/{len(rows)}）。\n\n"
        "| slot | 成败 | 步数 | 历史成败 | 历史步数 | 一致 |\n|---:|---|---:|---|---:|---|\n"
        + "".join(f"| {r['index']} | {r['success']} | {r['steps']} | {r['history_success']} | "
                  f"{r['history_steps']} | {r['match']} |\n" for r in rows))


def paired(new, ref):
    common = set(new) & set(ref)
    wins = sum(new[i]["success"] and not ref[i]["success"] for i in common)
    losses = sum(ref[i]["success"] and not new[i]["success"] for i in common)
    return dict(slots=len(common), wins=wins, losses=losses, ties=len(common) - wins - losses)


def panel(p, arm, episodes, rows):
    count = lambda keys: Counter(keys(rows[i]) for i, r in episodes.items() if r["success"])
    by_suite = count(lambda r: r["suite"])
    by_dim = count(lambda r: r["dimension"])
    dims = sorted({r["dimension"] for r in rows})
    refs = {name: load_reference(spec["episodes"], spec["arm"]) for name, spec in p["references"]["paired"].items()}
    history = read(p["references"]["totals_summary"])
    calls = [c for r in episodes.values() for c in r["policy_call_seconds"]]
    summary = dict(
        successes=sum(r["success"] for r in episodes.values()), episodes=len(episodes),
        complete=len(episodes) == 280, suites={s: by_suite[s] for s in SUITES},
        dimensions={d: by_dim[d] for d in dims},
        paired={name: paired(episodes, ref) for name, ref in refs.items()},
        reference_totals={a: v["successes"] for a, v in history["overall"]["arms"].items()},
        reference_suites={s: {a: v["successes"] for a, v in history["suites"][s]["arms"].items()} for s in SUITES},
        reference_dimensions={d: {a: v["successes"] for a, v in history["dimensions"][d]["arms"].items()} for d in dims},
        policy_call_p50_s=stats.median(calls) if calls else None,
        policy_calls=len(calls),
        episode_seconds_mean=stats.fmean(r["episode_seconds"] for r in episodes.values()) if episodes else None)
    shown = p["references"]["table_arms"]
    lines = [f"# LIBERO-Plus 280（engine）：`{arm}`\n",
             f"完成 {summary['episodes']}/280，成功 **{summary['successes']}**。"
             f"policy call p50 {summary['policy_call_p50_s']:.3f} s（{summary['policy_calls']} 次），"
             f"episode 平均 {summary['episode_seconds_mean']:.1f} s。\n" if episodes else "尚无 episode。\n",
             "| 臂 | 成功/280 |\n|---|---:|\n", f"| **{arm}** | **{summary['successes']}** |\n"]
    lines += [f"| {a} | {summary['reference_totals'][a]} |\n" for a in shown]
    lines += ["\n| Suite（每组 70） | " + arm + " | " + " | ".join(shown) + " |\n|---|---:|" + "---:|" * len(shown) + "\n"]
    lines += [f"| {s} | {by_suite[s]} | " + " | ".join(str(summary["reference_suites"][s][a]) for a in shown) + " |\n"
              for s in SUITES]
    lines += ["\n| 扰动（每组 40） | " + arm + " | " + " | ".join(shown) + " |\n|---|---:|" + "---:|" * len(shown) + "\n"]
    lines += [f"| {d} | {by_dim[d]} | " + " | ".join(str(summary["reference_dimensions"][d][a]) for a in shown) + " |\n"
              for d in dims]
    lines += ["\n| 配对对象 | 胜 | 负 | 平 | 共同 slot |\n|---|---:|---:|---:|---:|\n"]
    lines += [f"| {name} | {v['wins']} | {v['losses']} | {v['ties']} | {v['slots']} |\n"
              for name, v in summary["paired"].items()]
    return summary, "".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    p, arms = read(args.protocol), read(args.arms)
    cfg = arms[args.arm]
    root = Path(p["remote_root"]) / "results"
    group = cfg.get("group", args.arm)
    episodes = load_episodes(root / group / "episodes")
    slots = set(arm_slots(cfg))
    own = {i: r for i, r in episodes.items() if i in slots}
    base = dict(protocol=p["protocol"], arm=args.arm, group=group, pbs_jobid=os.environ.get("PBS_JOBID"),
                config=cfg, slots_expected=len(slots), slots_done=len(own))
    if cfg.get("compare_history"):
        extra, text = repro(p, args.arm, cfg, own)
    else:
        rows = read(p["panel"]["manifest"])["variants"]
        extra, text = panel(p, group, episodes, rows)
        if group != args.arm:
            save(root / group / "summary.json", dict(base, arm=group, **extra))
            (root / group / "RESULTS.zh.md").write_text(text, encoding="utf-8")
    extra.pop("complete", None)
    save(root / args.arm / "summary.json", dict(base, complete=len(own) == len(slots), **extra))
    (root / args.arm / "RESULTS.zh.md").write_text(text, encoding="utf-8")
    print(f"ANALYSIS arm={args.arm} done={len(own)}/{len(slots)}", flush=True)


if __name__ == "__main__":
    main()
