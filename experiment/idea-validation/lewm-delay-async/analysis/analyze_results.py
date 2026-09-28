#!/usr/bin/env python3
"""Summarize per-episode LeWM delay/async results using only Python stdlib."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory


QUANTILES = (0.50, 0.90, 0.95, 0.99)
METRICS = ("latency_ms", "observation_age_ticks", "rtf", "applied_plan_count", "missed_deadline_ticks", "max_tick_lateness_ms")


def load_rows(path):
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("episodes", [data])
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("输入须为 episode 对象、对象数组、含 episodes 数组的对象或 JSONL")
    return data


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0, **{f"p{int(q * 100)}": None for q in QUANTILES}}

    def linear(q):
        pos = q * (len(values) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    return {"n": len(values), **{f"p{int(q * 100)}": linear(q) for q in QUANTILES}}


def mcnemar_exact_two_sided(gains, losses):
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def analyze(rows, baseline_condition="n1_sync_k0", gate_condition="batch50_official_k0", expected_n=50):
    groups = defaultdict(dict)
    for index, row in enumerate(rows, 1):
        for field in ("task", "condition", "task_id", "success"):
            if field not in row:
                raise ValueError(f"第 {index} 行缺少必需字段 {field!r}")
        if not isinstance(row["success"], bool):
            raise ValueError(f"第 {index} 行 success 必须是 JSON boolean")
        task, condition, task_id = str(row["task"]), str(row["condition"]), str(row["task_id"])
        for metric in METRICS:
            value = row.get(metric)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"第 {index} 行 {metric} 必须是有限的非负数或 null")
        episodes = groups[(task, condition)]
        if task_id in episodes:
            raise ValueError(f"重复 episode: task={task}, condition={condition}, task_id={task_id}")
        episodes[task_id] = row

    tasks = sorted({task for task, _ in groups})
    if not tasks:
        raise ValueError("没有 episode 记录")
    if baseline_condition == gate_condition:
        raise ValueError("baseline 与 gate condition 必须不同")

    result = {
        "schema": "lewm-delay-async-analysis-v1",
        "expected_n": expected_n,
        "baseline_condition": baseline_condition,
        "gate_condition": gate_condition,
        "tasks": {},
    }
    for task in tasks:
        conditions = sorted(condition for task_name, condition in groups if task_name == task)
        baseline = groups.get((task, baseline_condition))
        if baseline is None:
            raise ValueError(f"task={task} 缺少配对基线 {baseline_condition}")
        task_result = {"task_ids": sorted(baseline), "conditions": {}}
        for condition in conditions:
            episodes = groups[(task, condition)]
            successes = sum(row["success"] for row in episodes.values())
            summary = {
                "n": len(episodes),
                "n_matches_expected": len(episodes) == expected_n,
                "successes": successes,
                "success_rate": successes / len(episodes),
                **{metric: quantiles(row[metric] for row in episodes.values() if row.get(metric) is not None) for metric in METRICS},
            }
            if condition == gate_condition:
                if set(episodes) != set(baseline):
                    raise ValueError(f"task={task}, gate={condition} task_id 与冻结基线不匹配")
                summary["role"] = "gate_only"
                summary["paired_vs_baseline"] = None
            elif condition == baseline_condition:
                summary["role"] = "paired_baseline"
                summary["paired_vs_baseline"] = None
            else:
                if set(episodes) != set(baseline):
                    missing = sorted(set(baseline) - set(episodes))
                    extra = sorted(set(episodes) - set(baseline))
                    raise ValueError(f"task={task}, condition={condition} task_id 与基线不匹配；missing={missing}, extra={extra}")
                gains = sum(not baseline[tid]["success"] and episodes[tid]["success"] for tid in baseline)
                losses = sum(baseline[tid]["success"] and not episodes[tid]["success"] for tid in baseline)
                matched = len(baseline)
                summary["role"] = "paired_arm"
                summary["paired_vs_baseline"] = {
                    "matched_n": matched,
                    "gains": gains,
                    "gain_rate": gains / matched,
                    "losses": losses,
                    "loss_rate": losses / matched,
                    "discordant_n": gains + losses,
                    "net_gain_percentage_points": 100 * (gains - losses) / matched,
                    "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(gains, losses),
                }
            task_result["conditions"][condition] = summary
        result["tasks"][task] = task_result
    return result


def self_check():
    ids = ("a", "b", "c", "d")
    baseline = (True, False, False, False)
    target = (True, True, True, False)
    rows = []
    for condition, outcomes in (("n1_sync_k0", baseline), ("async_n1_k0", target), ("batch50_official_k0", baseline)):
        for i, (task_id, success) in enumerate(zip(ids, outcomes)):
            rows.append({"task": "synthetic", "condition": condition, "task_id": task_id, "success": success,
                         "latency_ms": 10 * (i + 1), "observation_age_ticks": i, "rtf": 0.5 * (i + 1)})
    with TemporaryDirectory() as temp_dir:
        sample = Path(temp_dir) / "episodes.jsonl"
        sample.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        result = analyze(load_rows(sample), expected_n=4)["tasks"]["synthetic"]["conditions"]
    assert result["async_n1_k0"]["success_rate"] == 0.75
    paired = result["async_n1_k0"]["paired_vs_baseline"]
    assert (paired["gains"], paired["losses"], paired["mcnemar_exact_two_sided_p"]) == (2, 0, 0.5)
    assert result["batch50_official_k0"]["role"] == "gate_only"
    assert result["batch50_official_k0"]["paired_vs_baseline"] is None
    assert result["async_n1_k0"]["latency_ms"]["p50"] == 25
    assert result["async_n1_k0"]["rtf"]["p90"] == 1.85
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON / JSONL episode records")
    parser.add_argument("--output", type=Path, help="write JSON summary here; default stdout")
    parser.add_argument("--baseline-condition", default="n1_sync_k0")
    parser.add_argument("--gate-condition", default="batch50_official_k0")
    parser.add_argument("--expected-n", type=int, default=50)
    parser.add_argument("--self-check", action="store_true", help="run a tiny in-memory synthetic check")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.input is None:
        parser.error("请提供 --input 或使用 --self-check")
    if args.expected_n < 1:
        parser.error("--expected-n 必须大于 0")
    output = analyze(load_rows(args.input), args.baseline_condition, args.gate_condition, args.expected_n)
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
