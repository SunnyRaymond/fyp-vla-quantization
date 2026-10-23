"""Descriptive paired analysis of completed candidates; PBS CPU only."""
import argparse
import os
from pathlib import Path
import statistics

from choose import (close, eligible_points, epoch_metrics, guard, read, require,
                    select, write)

SOURCES = (("r1-r000", "25805582.pbs101", 0.1),
           ("r1-r001", "25805512.pbs101", 0.5),
           ("r1-r003", "25811378.pbs101", 0.7),
           ("r1-r002", "25805536.pbs101", 1.0))
PRIMARY = "mean_motor_rmse_first10"
FUTURE = "future_latent_nmse"
TEMPORAL = "temporal_delta_global_nmse"


def change(current, reference):
    require(reference > 0, "Percent change needs a positive reference")
    return 100 * (current / reference - 1)


def case_metrics(row, case):
    action = row["per_case"][str(case)]
    future = row["future_metrics"]["per_case"][str(case)]
    return {PRIMARY: action["motor_rmse_first10_vs_bf16"],
            FUTURE: future[FUTURE], TEMPORAL: future[TEMPORAL]}


def paired(current, reference):
    deltas = {case: current[case] - reference[case] for case in current}
    return {"differences": deltas, "wins": sum(v < 0 for v in deltas.values()),
            "ties": sum(v == 0 for v in deltas.values()),
            "losses": sum(v > 0 for v in deltas.values())}


def main():
    guard()
    assert change(8, 10) < 0
    assert paired({"a": 1, "b": 2}, {"a": 2, "b": 1})["wins"] == 1
    args = argparse.ArgumentParser()
    args.add_argument("--protocol", type=Path, required=True)
    args.add_argument("--out", type=Path, required=True)
    args = args.parse_args()
    protocol = read(args.protocol)
    root = Path(protocol["remote_root"])
    require(args.out.parent.resolve() == (root / "results").resolve(), "Unexpected output")
    cases = protocol["selection_cases"]
    require(cases == [1000, 1001, 1002, 1003], "Partial analysis cohort changed")
    points = []
    for run, job, rho in SOURCES:
        source = read(root / "results" / f"{run}-{job}" / "summary.json")
        require(source["protocol"] == protocol["protocol"] and
                source["status"] == "search_complete" and source["job_id"] == job and
                source["run_id"] == run and source["rho"] == rho and
                source["completed_epochs"] == 2 and source["updates"] == 16 and
                source["target_linears"] == 614, "Incomplete or mismatched source")
        require([row["epoch"] for row in source["selection_history"]] == [0, 1, 2],
                "Incomplete selection history")
        for row in source["selection_history"]:
            metrics = epoch_metrics(row, cases)
            per_case = {str(case): case_metrics(row, case) for case in cases}
            for name in (FUTURE, TEMPORAL):
                close(metrics[name], statistics.mean(v[name] for v in per_case.values()), name)
            points.append({"run_id": run, "job_id": job, "rho": rho,
                           "latent_weight": source["latent_weight"], "epoch": row["epoch"],
                           "metrics": metrics, "per_case": per_case})
    baselines = [point for point in points if point["epoch"] == 0]
    baseline = baselines[0]
    require(len(baselines) == len(SOURCES) and
            all(point["metrics"] == baseline["metrics"] and
                point["per_case"] == baseline["per_case"]
                for point in baselines[1:]), "Initial parents differ")
    points = eligible_points(points, baseline["metrics"],
                             protocol["search"]["guardrail_fraction"],
                             protocol["search"]["guardrails"])
    for point in points:
        point["change_pct_vs_parent"] = {
            name: change(value, baseline["metrics"][name])
            for name, value in point["metrics"].items()}
        point["paired_vs_parent"] = {}
        for name in (PRIMARY, FUTURE, TEMPORAL):
            point["paired_vs_parent"][name] = paired(
                {case: v[name] for case, v in point["per_case"].items()},
                {case: v[name] for case, v in baseline["per_case"].items()})
    chosen, strict, _ = select(points, protocol["search"]["near_tie_fraction"])
    comparisons = []
    for epoch in (1, 2):
        epoch_points = sorted((point for point in points if point["epoch"] == epoch),
                              key=lambda point: point["rho"])
        for low_index, low in enumerate(epoch_points):
            for high in epoch_points[low_index + 1:]:
                comparisons.append({
                    "epoch": epoch,
                    "direction": f"high rho {high['rho']:g} minus low rho {low['rho']:g}",
                    "change_pct": {name: change(high["metrics"][name], value)
                                   for name, value in low["metrics"].items()},
                    "paired": {name: paired(
                        {case: v[name] for case, v in high["per_case"].items()},
                        {case: v[name] for case, v in low["per_case"].items()})
                        for name in (PRIMARY, FUTURE, TEMPORAL)}})
    progression = []
    for run, _, rho in SOURCES:
        e1, e2 = [p for p in points if p["run_id"] == run and p["epoch"] > 0]
        progression.append({"rho": rho, "epoch2_vs_epoch1_pct": {
            name: change(e2["metrics"][name], value) for name, value in e1["metrics"].items()}})
    gains = {case: baseline["per_case"][case][PRIMARY] - chosen["per_case"][case][PRIMARY]
             for case in baseline["per_case"]}
    gain_sum = sum(gains.values())
    planned_round1_candidates = 5
    remaining_round1_candidates = planned_round1_candidates - len(SOURCES)
    result = {"status": "partial_analysis_complete", "protocol": protocol["protocol"],
              "job_id": os.environ["PBS_JOBID"], "source_jobs": [s[1] for s in SOURCES],
              "completed_candidates": len(SOURCES),
              "planned_round1_candidates": planned_round1_candidates,
              "remaining_round1_candidates": remaining_round1_candidates,
              "case_count": 4, "context_count": 8, "new_model_forward_count": 0,
              "test_forward_count": 0, "global_selection_frozen": False,
              "points": points, "cross_rho": comparisons, "epoch_progression": progression,
              "provisional_best": {k: chosen[k] for k in ("run_id", "rho", "epoch", "latent_weight")},
              "strict_partial_minimum": {k: strict[k] for k in ("run_id", "rho", "epoch")},
              "provisional_action_gain_contribution_pct": {
                  case: 100 * value / gain_sum for case, value in gains.items()},
              "scope": "Descriptive selection-only; four cases are units, not eight independent samples; no action-only attribution, closed-loop or native-kernel claim"}
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "summary.json", result)
    lines = [f"# Microscaling + latent loss：{len(SOURCES)} 个候选的阶段分析", "",
             "仅 selection 4 cases × 2 seeds，先在 case 内平均 seeds，再等权平均 cases。",
             "所有误差均相对同条件 BF16；百分比相对未训练 Micro parent，负值表示误差下降。",
             "", "| ρ / epoch | first10 RMSE | 变化 | full32 | late22 | gripper | future NMSE | 变化 | temporal NMSE | guards |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for point in [baseline] + [p for p in points if p["epoch"] > 0]:
        m, d = point["metrics"], point.get("change_pct_vs_parent", {})
        lines.append(f"| {point['rho']} / {point['epoch']} | {m[PRIMARY]:.6f} | {d.get(PRIMARY, 0):+.2f}% | {m['mean_motor_rmse_32']:.6f} | {m['mean_motor_rmse_late22']:.6f} | {m['mean_gripper_rmse_first10']:.6f} | {m[FUTURE]:.6f} | {d.get(FUTURE, 0):+.2f}% | {m[TEMPORAL]:.6f} | {'pass' if point.get('eligible', True) else 'fail'} |")
    epoch2_by_run = {point["run_id"]: point for point in points if point["epoch"] == 2}
    candidate_labels = ["parent"] + [f"ρ={rho:g}" for _, _, rho in SOURCES]
    case_separator = "|---|" + "|".join(["---:"] * (2 * len(candidate_labels))) + "|"
    lines.extend(["", "## epoch2 逐 case", "",
                  "| case | " + " | ".join(f"{label} action" for label in candidate_labels) +
                  " | " + " | ".join(f"{label} future" for label in candidate_labels) + " |",
                  case_separator])
    e2s = [epoch2_by_run[run] for run, _, _ in SOURCES]
    for case in map(str, cases):
        values = [p["per_case"][case] for p in [baseline] + e2s]
        cells = [v[name] for name in (PRIMARY, FUTURE) for v in values]
        lines.append("| " + case + " | " + " | ".join(f"{v:.6f}" for v in cells) + " |")
    lines.extend(["", f"目前可用候选中的暂时最优：ρ={chosen['rho']}，epoch={chosen['epoch']}，λ={chosen['latent_weight']:.9f}。",
                  f"这不是全局 optimal λ；其余 {remaining_round1_candidates} 个 R1、整个 R2 和最终 test 尚未纳入。",
                  "本次新增模型 forward=0，未读取任何 .pt。ρ0 action-only 对照尚未完成，不能单独归因 latent loss。"])
    (args.out / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "PARTIAL_ANALYSIS_COMPLETE").touch()
    print("PARTIAL_ANALYSIS_COMPLETE", os.environ["PBS_JOBID"])


if __name__ == "__main__":
    main()
