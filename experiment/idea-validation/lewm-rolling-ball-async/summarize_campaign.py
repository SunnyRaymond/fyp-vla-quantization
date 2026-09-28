"""Summarize the frozen eight-condition campaign without loading a model."""
import argparse
import collections
import csv
import json
import math
import pathlib
import statistics

MODES = ("sync", "fixed-delay", "true-async")
SEEDS = list(range(2026092700, 2026092750))


def stats(values):
    values = sorted(float(x) for x in values if x is not None)
    if not values:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite timing in campaign")
    position = (len(values) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"count": len(values), "median": statistics.median(values), "p95": p95,
            "min": values[0], "max": values[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    args = parser.parse_args()
    rows = []
    baseline_successes = None
    stage_evidence = []
    for mode in MODES:
        path = args.root / "results" / f"{mode}-50-physical-001" / "result.json"
        data = json.loads(path.read_text())
        wrapper = args.root / "results" / f"{mode}-50-physical-wrapper-001"
        exits = {name: int((wrapper / f"{name}_exit_status.txt").read_text())
                 for name in ("runner", "wrapper")}
        assert data["status"] == "COMPLETED" and not data.get("error") and all(x == 0 for x in exits.values())
        assert data["closed_loop_evaluated"] and data["paired"] and not data["smoke_only"]
        assert data["pairing_protocol"] == "physical-v2"
        expected_delays = [0, 1, 2, 4, 8, 16] if mode == "fixed-delay" else [None]
        assert [c["condition"]["fixed_delay_ticks"] for c in data["conditions"]] == expected_delays
        stage_evidence.append({"mode": mode, "result": str(path), "exit_status": exits})
        for condition in data["conditions"]:
            episodes = condition["episodes"]
            assert [ep["fixture_seed"] for ep in episodes] == SEEDS
            assert len(episodes) == condition["summary"]["episodes"] == 50
            requests = [r for ep in episodes for r in ep["requests"]]
            assert len(requests) == sum(ep["request_count"] for ep in episodes)
            applied = [r for r in requests if r["status"] == "applied"]
            assert all(r["status"] in ("applied", "dropped") for r in requests)
            if mode != "true-async":
                expected_age = (condition["condition"]["fixed_delay_ticks"] or 0) * 0.04
                assert all(math.isclose(r["observation_age_sim_s"], expected_age, abs_tol=1e-9) for r in applied)
            else:
                assert all(r["observation_age_sim_s"] >= 0.04 for r in applied)
            successes = {ep["fixture_seed"] for ep in episodes if ep["success"]}
            assert len(successes) == condition["summary"]["native_successes"]
            if mode == "sync":
                baseline_successes = successes
            cleanup_total = sum(ep["terminal_post_step_cleanup_wall_s"] for ep in episodes)
            row = {**condition["condition"], **condition["summary"],
                   "success_seed_ids": sorted(successes),
                   "gained_vs_sync": sorted(successes - baseline_successes),
                   "lost_vs_sync": sorted(baseline_successes - successes),
                   "requests": len(requests), "applied_requests": len(applied),
                   "request_status_counts": dict(collections.Counter(r["status"] for r in requests)),
                   "request_errors": sum(bool(r.get("request_error")) for r in requests),
                   "control_ticks": sum(ep["control_ticks"] for ep in episodes),
                   "total_terminal_cleanup_wall_s": cleanup_total,
                   "aggregate_rtf_with_terminal_cleanup": condition["summary"]["total_sim_elapsed_s"] /
                       (condition["summary"]["total_wall_elapsed_s"] + cleanup_total),
                   "native_end_reason_counts": dict(collections.Counter(ep.get("end_reason") for ep in episodes)),
                   "native_max_phase_counts": dict(collections.Counter(str(ep.get("max_phase")) for ep in episodes)),
                   "native_terminated_phase_counts": dict(collections.Counter(str(ep.get("terminated_phase")) for ep in episodes)),
                   "terminal_cleanup_wall_s": stats(ep["terminal_post_step_cleanup_wall_s"] for ep in episodes),
                   "pacing_overrun_s": stats(x for ep in episodes for x in ep["wall_pacing_overrun_s"])}
            for key in ("server_latency_s", "cem_latency_s", "client_http_latency_s"):
                row[key] = stats(r.get(key) for r in requests)
            for key in ("observation_age_wall_s", "observation_age_sim_s"):
                row[key] = stats(r.get(key) for r in applied)
            rows.append(row)
    summary = {"status": "COMPLETED", "pairing_protocol": "physical-v2",
               "independent_fixture_count": 50, "episode_runs": 400,
               "strict_rgb_pairing_passed": False,
               "observation_age_denominator": "applied requests only; dropped requests counted separately",
               "rtf_boundary": "through native terminal physics step; terminal cleanup/drain reported separately",
               "fixed_delay_rtf_limit": "unpaced synthetic simulation delay; fewer planner calls can raise RTF without speeding up the planner",
               "baseline_limit": "Rolling Ball task adapter uses training-only goals whose catch success labels were not individually verified",
               "quantiles": "linear interpolation of sorted request-level observations",
               "stage_evidence": stage_evidence, "conditions": rows}
    output = args.root / "results" / "campaign-physical-001"
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["mode", "fixed_delay_ticks", "fixed_delay_ms", "episodes", "native_successes",
              "native_success_rate", "requests", "applied_requests", "request_errors", "aggregate_rtf",
              "aggregate_rtf_with_terminal_cleanup", "total_terminal_cleanup_wall_s",
              "planner_deadline_misses", "wall_pacing_overruns"]
    timing = ["server_latency_s", "cem_latency_s", "client_http_latency_s",
              "observation_age_wall_s", "observation_age_sim_s"]
    fields += [f"{key}_{quantile}" for key in timing for quantile in ("count", "median", "p95")]
    with (output / "SUMMARY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields if key in row}
            flat.update({f"{key}_{quantile}": row[key][quantile]
                         for key in timing for quantile in ("count", "median", "p95")})
            writer.writerow(flat)
    print(json.dumps({"status": summary["status"], "conditions": len(rows),
                      "success_counts": [r["native_successes"] for r in rows]}))


if __name__ == "__main__":
    assert math.isclose(stats([1, 2, 3, 4])["p95"], 3.85)
    assert stats([])["count"] == 0
    main()
