"""Require exactly the requested 400 distinct completed original-LIBERO episodes."""
import argparse
import json
import os
from pathlib import Path
import statistics

from worker import read, save, task_plan


def aggregate(p, rows):
    expected = {(s, t, e) for s in p["evaluation"]["suites"] for t in range(10) for e in range(10)}
    keys = [(r["suite"], r["task_id"], r["trial_id"]) for r in rows]
    if len(keys) != 400 or len(set(keys)) != 400 or set(keys) != expected:
        raise ValueError(f"Incomplete or duplicate episode coverage: {len(keys)}, {len(set(keys))}")
    for row in rows:
        if (row.get("complete") is not True or type(row.get("success")) is not bool
                or row.get("protocol") != p["protocol"] or row.get("engine") != "c_buffer"
                or row["initial_state_index"] != row["trial_id"]
                or row["seed"] != 42 or row["sampler_seed"] != 42):
            raise ValueError("Episode identity/settings differ")
    suites = {}
    tasks = []
    for suite in p["evaluation"]["suites"]:
        subset = [r for r in rows if r["suite"] == suite]
        successes = sum(r["success"] for r in subset)
        suites[suite] = dict(episodes=100, successes=successes, success_rate=successes/100)
        for task in range(10):
            selected = [r for r in subset if r["task_id"] == task]
            tasks.append(dict(suite=suite, task_id=task, episodes=10,
                              successes=sum(r["success"] for r in selected)))
    return dict(episodes=400, tasks=40, successes=sum(r["success"] for r in rows),
                success_rate=sum(r["success"] for r in rows)/400, suites=suites, task_results=tasks)


def self_check(p):
    assert len(task_plan(p)) == 40
    rows = [dict(suite=s, task_id=t, trial_id=e, initial_state_index=e, complete=True,
                 success=t % 2 == 0, seed=42, sampler_seed=42, engine="c_buffer", protocol=p["protocol"])
            for s in p["evaluation"]["suites"] for t in range(10) for e in range(10)]
    assert aggregate(p, rows)["successes"] == 200
    for invalid in (rows[:-1], rows[:-1] + [rows[0]]):
        try:
            aggregate(p, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing/duplicate episodes must not pass completion")
    print("Coverage and aggregation self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    if args.self_check:
        self_check(p)
        return
    from worker import guard
    guard()
    root = Path(p["remote_root"]) / "results"
    rows = [read(path) for path in sorted(root.glob("worker_*/episodes/*.json"))]
    result = aggregate(p, rows)
    workers = [read(root / f"worker_{i}/summary.json") for i in range(4)]
    startup = [read(root / f"worker_{i}/startup.json") for i in range(4)]
    job = os.environ["PBS_JOBID"]
    if any(r.get("pbs_jobid") != job for r in rows + workers + startup):
        raise ValueError("Results do not belong to this PBS job")
    if (any(not r.get("passed") or r["action_max_abs"] != 0 or r["future_max_abs"] != 0 for r in startup)
            or any(not r.get("complete") or r["model_loads"] != 1 for r in workers)
            or len({r["gpu_uuid"] for r in startup}) != 4
            or sum(r["episodes"] for r in workers) != 400):
        raise ValueError("Four-GPU engine/completion gate failed")
    timings = [x for r in rows for x in r["policy_call_seconds"]]
    result.update(protocol=p["protocol"], pbs_jobid=job, complete=True,
                  benchmark="original LIBERO", engine="c_buffer", engine_workers_passed=4,
                  new_calibration_updates=0, model_loads=4, workers=workers,
                  policy_calls=len(timings), policy_call_p50_seconds=statistics.median(timings),
                  policy_call_scope="warm model call; excludes observation preprocessing, model loading, graph capture and environment stepping")
    save(root / "summary.json", result)
    lines = ["# Fast-WAM C80 reader-KL + inference engine：标准 LIBERO", "",
             f"作业 `{job}`；40 tasks × 10 episodes，共 400。未重新校准；四个常驻模型动态领取任务。", "",
             "| Suite | 成功数 | 成功率 |", "|---|---:|---:|"]
    for name, r in result["suites"].items():
        lines.append(f"| {name} | {r['successes']}/100 | {100*r['success_rate']:.2f}% |")
    lines += [f"| Average | {result['successes']}/400 | {100*result['success_rate']:.2f}% |", "",
              "结果为每 task 前 10 个官方 initial states 的单轮评测，不代表多次重复或显著性结论。",
              "Engine 使用 packed MXVQ → 在线 BF16 decode → BF16 cuBLAS GEMM，并启用 condition cache/CUDA Graph。"]
    (root / "RESULTS.zh.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("episodes", "successes", "success_rate", "suites")}))


if __name__ == "__main__":
    main()
