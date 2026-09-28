"""Summarize frozen paired task outcomes; run within the campaign allocation."""
import argparse
import json
import os
import platform
from pathlib import Path


def allocation_guard():
    nodes = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    host = platform.node().split(".")[0].lower()
    if not os.environ.get("PBS_JOBID") or not nodes.is_file():
        raise RuntimeError("A real PBS compute allocation is required")
    if host not in {x.split(".")[0].lower() for x in nodes.read_text().split()} or any(x in host for x in ("login", "head", "submit")):
        raise RuntimeError("Refusing control or unallocated host")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    allocation_guard()
    import numpy as np

    freeze = json.loads(args.freeze.read_text())
    arms = {}
    for spec in freeze["arms"]:
        result = json.loads((args.out / "arms" / spec["id"] / "result.json").read_text())
        outcomes = np.asarray(result["episode_successes"], dtype=float)
        samples = np.asarray(result["steady_times_seconds"], dtype=float)
        assert outcomes.shape == (50,) and np.isin(outcomes, [0, 1]).all()
        assert samples.shape == (3,) and np.isfinite(samples).all() and (samples > 0).all()
        assert result["steady_active_environments"] == 50
        assert result["steady_cost_calls"] == [50 * spec["iterations"]] * 3
        assert result["steady_candidate_scores"] == [50 * spec["iterations"] * 300] * 3
        arms[(spec["model"], spec["iterations"], spec["seed"])] = {
            "id": spec["id"], "outcomes": outcomes, "successes": int(outcomes.sum()),
            "steady_median_seconds": float(np.median(samples)),
            "steady_range_seconds": [float(samples.min()), float(samples.max())],
            "evaluation_seconds": result["evaluation_seconds"],
        }
    seeds = [42, 43, 44]
    rng = np.random.default_rng(20260926)
    # Resample source tasks and retain all three seed repeats within each task.
    task_samples = rng.integers(0, 50, size=(10000, 50))

    def summarize_difference(values):
        values = np.asarray(values, dtype=float)
        assert values.shape == (3, 50)
        task_mean = values.mean(axis=0)
        boot = task_mean[task_samples].mean(axis=1)
        return {
            "mean_difference_pp": float(100 * task_mean.mean()),
            "per_seed_difference_pp": (100 * values.mean(axis=1)).tolist(),
            "task_cluster_bootstrap_95_percentile_interval_pp": (100 * np.quantile(boot, [0.025, 0.975])).tolist(),
            "tasks_positive": int((task_mean > 0).sum()),
            "tasks_negative": int((task_mean < 0).sum()),
            "tasks_equal": int((task_mean == 0).sum()),
        }

    budget_differences = {
        m: summarize_difference([arms[m, 10, s]["outcomes"] - arms[m, 30, s]["outcomes"] for s in seeds])
        for m in ("lewm", "fastlewm")
    }
    interaction = summarize_difference([
        (arms["fastlewm", 10, s]["outcomes"] - arms["fastlewm", 30, s]["outcomes"])
        - (arms["lewm", 10, s]["outcomes"] - arms["lewm", 30, s]["outcomes"])
        for s in seeds
    ])
    model_differences = {
        str(i): summarize_difference([arms["fastlewm", i, s]["outcomes"] - arms["lewm", i, s]["outcomes"] for s in seeds])
        for i in (10, 30)
    }
    timing_ratios = {}
    for iterations in (10, 30):
        ratios = [arms["lewm", iterations, s]["steady_median_seconds"] / arms["fastlewm", iterations, s]["steady_median_seconds"] for s in seeds]
        timing_ratios[str(iterations)] = {"lewm_over_fastlewm_per_seed": ratios, "median_ratio": float(np.median(ratios))}
    for model in ("lewm", "fastlewm"):
        ratios = [arms[model, 30, s]["steady_median_seconds"] / arms[model, 10, s]["steady_median_seconds"] for s in seeds]
        timing_ratios[model + "_30_over_10"] = {"per_seed": ratios, "median_ratio": float(np.median(ratios))}
    expected = np.asarray(freeze["validity"]["fast_seed42_i30_expected_successes"], dtype=float)
    fast_exact = bool(np.array_equal(expected, arms["fastlewm", 30, 42]["outcomes"]))
    lewm_usable = all(arms["lewm", 30, s]["successes"] >= 45 for s in seeds)
    rows = [{k: v for k, v in arms[m, i, s].items() if k != "outcomes"} | {"model": m, "iterations": i, "seed": s} for m in ("lewm", "fastlewm") for i in (10, 30) for s in seeds]
    analysis = {
        "status": "COMPLETE", "source_tasks": 50, "solver_seeds": seeds,
        "replication_boundary": "150 outcomes per condition are repeats of 50 tasks, not 150 IID tasks",
        "validity": {"complete_arms": 12, "fast_seed42_i30_exact_vector_match": fast_exact, "lewm_i30_usable": lewm_usable},
        "arms": rows, "budget_10_minus_30": budget_differences,
        "interaction_fast_budget_difference_minus_lewm_budget_difference": interaction,
        "fast_minus_lewm_by_budget": model_differences,
        "steady_timing_ratios": timing_ratios,
        "uncertainty": "Exploratory bootstrap resamples the 50 already-used source tasks. No multiplicity-adjusted confirmatory tests or noninferiority/equivalence claims.",
        "insight_boundary": "Describe stable paired differences first. A mechanism or deployable adaptive-budget claim requires a separate frozen pilot; no follow-up is automatically accepted by this summary.",
    }
    (args.out / "paired_analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"analysis": "COMPLETE", "validity": analysis["validity"], "interaction": interaction}))


if __name__ == "__main__":
    main()
