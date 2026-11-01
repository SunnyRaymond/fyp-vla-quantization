"""Gripper attribution of the RAVQU_W2 vs QTIP_W2 closed loop (ravq.json trace_job): replay the discordant episodes
(one arm succeeded, the other failed) of both arms with the A16 loop worker's runtime and episode logic, logging per
control step the gripper command before binarization, the finger opening and the end-effector position; then label
every replay by what the gripper physically did.

  --worker w   arm pair[w % 2], shard w // 2 of the discordant list; writes results/ravq/trace/<arm>/<index>.json
  --analyze    CPU: labels, per-episode comparison, summary.json + RESULTS.zh.md under results/ravq/trace

Runs from the closed-loop directory (common.py, loop_worker.py and the engine files next to it).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from common import guard, read, results, save

EMPTY_WIDTH = 0.008   # finger opening (m) below which a closed gripper holds nothing (Panda fully closed ~0)
OPEN_WIDTH = 0.07     # opening above which the gripper is open
SETTLE = 8            # steps after a close command before the opening is read
FLICKER = 5           # close or open segments shorter than this many steps are command flicker


def loop_episodes(root):
    return {int(f.stem): read(f)["success"] for f in (Path(root) / "episodes").glob("*.json")}


def discordant(cfg, p):
    a, b = (loop_episodes(cfg["loops"][arm]) for arm in cfg["pair"])
    if len(a) != 280 or len(b) != 280:
        raise RuntimeError(f"Closed loops incomplete: {len(a)} / {len(b)}")
    return sorted(i for i in a if a[i] != b[i])


def width(obs):
    q = obs["robot0_gripper_qpos"]
    return float(q[0] - q[1])


def traced(runtime, row, index, arm):
    """loop_worker.episode with per-step logging; the control flow, seeds and caps are the same."""
    import numpy as np
    import torch
    import runner as pilot
    started = time.monotonic()
    obs = runtime.task(row, index)
    cap = 700 if row["suite"] == "libero_10" else 400
    steps, replans, success = 0, 0, False
    log = dict(raw=[], command=[], width=[width(obs)], eef=[[float(x) for x in obs["robot0_eef_pos"]]])
    while steps < cap and not success:
        datum = runtime.datum(obs)
        action, _ = runtime.infer(datum, pilot.seed_for(index, replans), measure=False)
        torch.cuda.synchronize()
        raw = runtime.ev._denormalize_action(action, runtime.processor)[0][..., -1]
        commands = runtime.command(action)
        if not np.isfinite(commands).all():
            raise RuntimeError("Non-finite controller commands")
        for k, command in enumerate(commands[:min(10, cap - steps)]):
            obs, _, done, _ = runtime.env.step(command.tolist())
            steps += 1
            log["raw"].append(float(raw[k]))
            log["command"].append(float(command[-1]))
            log["width"].append(width(obs))
            log["eef"].append([float(x) for x in obs["robot0_eef_pos"]])
            if done:
                success = True
                break
        replans += 1
    runtime.env.close()
    runtime.env = None
    return dict(arm=arm, index=index, suite=row["suite"], dimension=row["dimension"], task_id=int(row["task_id"]),
                success=success, steps=steps, replans=replans, seconds=time.monotonic() - started,
                pbs_jobid=os.environ["PBS_JOBID"], **log)


def segments(command):
    """Runs of equal binarized command: (sign, start step, length)."""
    out, start = [], 0
    for t in range(1, len(command) + 1):
        if t == len(command) or command[t] != command[start]:
            out.append((command[start], start, t - start))
            start = t
    return out


def label(trace):
    """What the gripper did: grasps (close segments long enough to read), how each ended, flicker and margin."""
    w, cmd, raw = trace["width"], trace["command"], trace["raw"]
    closes = []
    for sign, start, length in segments(cmd):
        if sign <= 0:
            continue
        if length < SETTLE:
            closes.append(dict(start=start, length=length, kind="flicker"))
            continue
        held = [w[t + 1] for t in range(start + SETTLE - 1, start + length)]  # w[0] is before the first step
        if max(held) < EMPTY_WIDTH:
            kind = "empty"
        elif held[-1] < EMPTY_WIDTH:
            kind = "dropped"
        else:
            kind = "held"
        closes.append(dict(start=start, length=length, kind=kind, width_settled=held[0], width_end=held[-1]))
    kinds = Counter(c["kind"] for c in closes)
    flips = sum(1 for _, _, n in segments(cmd)) - 1
    short = sum(1 for _, _, n in segments(cmd)[1:-1] if n < FLICKER)
    if trace["success"]:
        outcome = "success"
    elif not any(c["kind"] != "flicker" for c in closes):
        outcome = "never_closed"
    elif kinds["held"] == 0:
        outcome = "dropped" if kinds["dropped"] else "empty_grasp"
    elif kinds["dropped"]:
        outcome = "held_then_dropped"
    else:
        outcome = "held_no_success"
    margin = sorted(abs(x - 0.5) for x in raw)
    return dict(outcome=outcome, closes=len(closes), kinds=dict(kinds), flips=flips, short_segments=short,
                first_close=next((c["start"] for c in closes if c["kind"] != "flicker"), None),
                low_margin_steps=sum(1 for x in margin if x < 0.1), steps=trace["steps"])


def analyze(cfg, p):
    root = results(p) / "ravq" / "trace"
    a, b = cfg["pair"]
    loops = {arm: loop_episodes(cfg["loops"][arm]) for arm in cfg["pair"]}
    rows, reproduced = [], Counter()
    for i in discordant(cfg, p):
        row = dict(index=i)
        for arm in cfg["pair"]:
            t = read(root / arm / f"{i}.json")
            row[arm] = dict(loop_success=loops[arm][i], replay_success=t["success"], **label(t))
            reproduced[arm] += t["success"] == loops[arm][i]
            row["suite"], row["dimension"] = t["suite"], t["dimension"]
        row["winner"] = a if loops[a][i] else b
        rows.append(row)
    loser = {arm: Counter(r[arm]["outcome"] for r in rows if r["winner"] != arm) for arm in cfg["pair"]}
    gripper_fail = ("never_closed", "empty_grasp", "dropped", "held_then_dropped")
    summary = dict(protocol=p["protocol"], pbs_jobid=os.environ.get("PBS_JOBID"), complete=True, pair=cfg["pair"],
                   episodes=len(rows), reproduced=dict(reproduced), loser_outcomes={k: dict(v) for k, v in loser.items()},
                   gripper_failures={arm: sum(loser[arm][k] for k in gripper_fail) for arm in cfg["pair"]},
                   thresholds=dict(empty_width=EMPTY_WIDTH, settle=SETTLE, flicker=FLICKER), rows=rows)
    save(root / "summary.json", summary)
    names = dict(success="成功", never_closed="始终没合", empty_grasp="空抓", dropped="抓到又掉",
                 held_then_dropped="抓住后又掉", held_no_success="抓住了但没完成")
    lines = [f"# 夹爪归因：{a} 对 {b} 的分歧回合", "", f"作业 `{summary['pbs_jobid']}`，{len(rows)} 个分歧回合"
             f"（{a} 赢 {sum(r['winner'] == a for r in rows)}，{b} 赢 {sum(r['winner'] == b for r in rows)}）。"
             f"重放复现原闭环成败：{a} {reproduced[a]}/{len(rows)}，{b} {reproduced[b]}/{len(rows)}。", "",
             "| 回合 | suite | 维度 | 赢家 | " + " | ".join(f"{x} 结局 | {x} 合/抖动" for x in cfg["pair"]) + " |",
             "|---|---|---|---|" + "---|---|" * 2]
    for r in rows:
        lines.append(f"| {r['index']} | {r['suite']} | {r['dimension']} | {r['winner']} | " + " | ".join(
            f"{names[r[x]['outcome']]} | {r[x]['closes']}/{r[x]['short_segments']}" for x in cfg["pair"]) + " |")
    lines += ["", "输家的结局："]
    for arm in cfg["pair"]:
        lines.append(f"- {arm} 输的 {sum(loser[arm].values())} 回合：" + "，".join(
            f"{names[k]} {v}" for k, v in loser[arm].most_common()) + f"（夹爪类失败 {summary['gripper_failures'][arm]}）")
    (root / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True, help="ravq.json")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    guard()
    p, cfg = read(args.protocol), read(args.trace)["trace_job"]
    if args.analyze:
        print(json.dumps({k: v for k, v in analyze(cfg, p).items() if k != "rows"}, ensure_ascii=False), flush=True)
        return
    import loop_worker as lw
    arm, shard, shards = cfg["pair"][args.worker % 2], args.worker // 2, args.workers // 2
    mine = discordant(cfg, p)[shard::shards]
    rows = read(p["panel"]["manifest"])["variants"]
    out = results(p) / "ravq" / "trace" / arm
    todo = [i for i in mine if not (out / f"{i}.json").exists()]
    print(f"TRACE arm={arm} worker={args.worker} episodes={len(mine)} todo={len(todo)}", flush=True)
    if not todo:
        return
    runtime = lw.build_runtime(p, arm, results(p) / "ravq" / "trace" / f"worker_{args.worker}")
    try:
        for i in todo:
            record = traced(runtime, rows[i], i, arm)
            save(out / f"{i}.json", record)
            print(f"TRACE arm={arm} index={i} success={record['success']} steps={record['steps']}", flush=True)
    finally:
        if runtime.env is not None:
            runtime.env.close()


if __name__ == "__main__":
    main()
