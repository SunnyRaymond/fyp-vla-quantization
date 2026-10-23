"""CPU-allocation-only two-round selection from immutable search summaries."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics

PROTOCOL = "fastwam-vq-micro-latent-search-v1"
MAX_BYTES = 256 * 1024
METRICS = {
    "mean_motor_rmse_first10": "motor_rmse_first10_vs_bf16",
    "mean_motor_rmse_late22": "motor_rmse_late22_vs_bf16",
    "mean_motor_rmse_32": "motor_rmse_32_vs_bf16",
    "mean_gripper_rmse_first10": "gripper_rmse_first10_vs_bf16",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE", "")
    require(os.environ.get("PBS_JOBID") and nodefile and "login" not in host.lower(),
            "Numerical selection requires a genuine PBS compute allocation")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text().split()}
    require(host in nodes, "Selection host is outside its PBS allocation")


def read(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_BYTES, f"Oversized control JSON: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    require(len(content.encode("utf-8")) <= MAX_BYTES, "Selection output exceeds 256 KiB")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def scalar(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Invalid {label}")
    value = float(value)
    require(math.isfinite(value) and value >= 0, f"Nonfinite or negative {label}")
    return value


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-8),
            f"Stored metric differs from paired case means: {label}")


def p90(values):
    values = sorted(values)
    require(values, "P90 requires nonempty values")
    position = (len(values) - 1) * 0.9
    left = int(math.floor(position))
    right = int(math.ceil(position))
    return values[left] + (values[right] - values[left]) * (position - left)


def epoch_metrics(row, cases):
    require(row.get("case_count") == 4 and row.get("context_count") == 8,
            "Selection must cover four cases and both seeds")
    per_case = row.get("per_case", {})
    require(set(per_case) == {str(case) for case in cases}, "Selection case IDs differ")
    measured = {}
    for aggregate, key in METRICS.items():
        values = [scalar(per_case[str(case)][key], key) for case in cases]
        measured[aggregate] = statistics.mean(values)
        close(row[aggregate], measured[aggregate], aggregate)
    measured["p90_motor_rmse_first10"] = p90(
        [scalar(per_case[str(case)][METRICS["mean_motor_rmse_first10"]], "first10")
         for case in cases])
    close(row["p90_motor_rmse_first10"], measured["p90_motor_rmse_first10"], "p90")
    future = row.get("future_metrics", {})
    require(future.get("case_count") == 4 and future.get("context_count") == 8,
            "Selection future coverage differs")
    measured["future_latent_nmse"] = scalar(future["metrics"]["future_latent_nmse"], "future NMSE")
    measured["temporal_delta_global_nmse"] = scalar(
        future["metrics"]["temporal_delta_global_nmse"], "temporal NMSE")
    return measured


def eligible_points(points, baseline, fraction, guard_names):
    result = []
    for item in points:
        point = dict(item)
        point["guard_failures"] = [name for name in guard_names
                                   if point["metrics"][name] > baseline[name] * (1 + fraction) + 1e-12]
        point["eligible"] = not point["guard_failures"]
        result.append(point)
    return result


def select(points, tie_fraction):
    eligible = [point for point in points if point["eligible"]]
    require(eligible, "No eligible checkpoint, including the epoch0 fallback")
    minimum = min(point["metrics"]["mean_motor_rmse_first10"] for point in eligible)
    near = [point for point in eligible
            if point["metrics"]["mean_motor_rmse_first10"] <= minimum * (1 + tie_fraction) + 1e-12]
    chosen = min(near, key=lambda point: (point["rho"], point["epoch"], point["run_id"]))
    strict = min(eligible, key=lambda point: (point["metrics"]["mean_motor_rmse_first10"],
                                           point["rho"], point["epoch"], point["run_id"]))
    unconstrained = min(points, key=lambda point: (point["metrics"]["mean_motor_rmse_first10"],
                                                  point["rho"], point["epoch"], point["run_id"]))
    return chosen, strict, unconstrained


def first_round_candidates(protocol):
    specs = [{"run_id": f"r1-r{index:03d}", "rho": scalar(rho, "first-round rho")}
             for index, rho in enumerate(protocol["search"]["round1_rhos"])]
    additional = protocol.get("additional_round1_candidates", [])
    require(isinstance(additional, list), "Additional first-round candidates must be a list")
    specs.extend({"run_id": item.get("run_id"), "rho": scalar(item.get("rho"), "additional rho")}
                 for item in additional if isinstance(item, dict))
    require(len(specs) == len(protocol["search"]["round1_rhos"]) + len(additional)
            and all(isinstance(item["run_id"], str) and item["run_id"].startswith("r1-r") for item in specs)
            and len({item["run_id"] for item in specs}) == len(specs)
            and len({item["rho"] for item in specs}) == len(specs),
            "First-round candidate identities or rhos are duplicate or malformed")
    return specs


def round2_candidates(chosen, first_rhos=(0.1, 0.5, 1.0)):
    rhos = sorted(float(rho) for rho in first_rhos)
    require(len(rhos) >= 2 and len(set(rhos)) == len(rhos), "Invalid frozen first-round rho grid")
    index = next((i for i, rho in enumerate(rhos)
                  if math.isclose(chosen["rho"], rho, rel_tol=0.0, abs_tol=1e-12)), None)
    require(index is not None, "Selected rho is not a first-round candidate")
    if chosen["epoch"] == 0 or index == 0:
        extra = [("r2-r001", 0.03), ("r2-r002", math.sqrt(rhos[0] * rhos[1]))]
    elif index == len(rhos) - 1:
        extra = [("r2-r001", math.sqrt(rhos[-2] * rhos[-1])), ("r2-r002", 2.0 * rhos[-1])]
    else:
        extra = [("r2-r001", math.sqrt(rhos[index - 1] * rhos[index])),
                 ("r2-r002", math.sqrt(rhos[index] * rhos[index + 1]))]
    return [{"run_id": run, "rho": rho} for run, rho in [("r2-r000", 0.0), *extra]]


def publish(path, value, identity_keys):
    path = Path(path)
    if path.exists():
        previous = read(path)
        require(all(previous.get(key) == value.get(key) for key in identity_keys),
                "Refusing to change an already published numerical choice")
    write(path, value)


def run(args):
    guard()
    self_check()
    protocol = read(args.protocol)
    inputs = read(args.inputs)
    require(protocol["protocol"] == PROTOCOL and inputs["protocol"] == PROTOCOL,
            "Selection protocol differs")
    require(inputs["round"] == args.round and args.round in (1, 2), "Selection round differs")
    first_specs = first_round_candidates(protocol)
    first_rhos = [item["rho"] for item in first_specs]
    first_count = len(first_specs)
    round2_count = int(protocol["search"]["round2_count"])
    require(round2_count == 3, "This chooser requires exactly three round-two points")
    root = Path(protocol["remote_root"])
    require(args.out.parent.resolve() == (root / "results").resolve(), "Unexpected choice output path")
    args.out.mkdir(parents=True, exist_ok=False)
    require(Path(inputs["reference_summary_path"]).resolve() == Path(protocol["reference_summary"]).resolve(),
            "Reference summary path differs")
    reference = read(inputs["reference_summary_path"])
    require(reference["status"] == "complete" and reference["protocol"] == PROTOCOL,
            "Reference preparation is not complete")
    normalization = reference["loss_normalization"]
    c = scalar(normalization["normalization_c"], "normalization c")
    require(c > 0 and normalization["calibration_ids"] == protocol["calibration_cases"],
            "Loss normalization does not use the frozen calibration cohort")
    a0 = scalar(normalization["mean_cal_initial_action_mse"], "A0")
    z0 = scalar(normalization["mean_cal_initial_future_nmse"], "Z0")
    require(z0 > 0, "Z0 must be positive")
    close(c, a0 / z0, "calibration normalization")
    candidates = inputs["candidates"]
    require(len(candidates) == (first_count if args.round == 1 else first_count + round2_count),
            "Incomplete two-round candidate set")
    require(len({item["run_id"] for item in candidates}) == len(candidates), "Duplicate candidate runs")
    first = [item for item in candidates if item["round"] == 1]
    require({(item["run_id"], scalar(item["rho"], "first-round rho")) for item in first}
            == {(item["run_id"], item["rho"]) for item in first_specs}, "First-round identities or points changed")
    if args.round == 2:
        declaration = read(root / "round2_candidates.json")
        expected = {(item["run_id"], item["rho"]) for item in declaration["candidates"]}
        actual = {(item["run_id"], item["rho"]) for item in candidates if item["round"] == 2}
        require(actual == expected and len(actual) == round2_count, "Second-round points differ from frozen choice")
    points, baseline, source_jobs = [], None, []
    for entry in candidates:
        require(entry.get("verified_terminal") is True, "Candidate lacks controller terminal verification")
        result = Path(entry["result_dir"])
        require(result.parent.resolve() == (root / "results").resolve()
                and result.name == f"{entry['run_id']}-{entry['job_id']}", "Candidate result binding differs")
        require(Path(entry["summary_path"]).resolve() == (result / "summary.json").resolve(),
                "Candidate summary path differs")
        summary = read(entry["summary_path"])
        require(summary["status"] == "search_complete" and summary["protocol"] == PROTOCOL
                and summary["job_id"] == entry["job_id"] and summary["run_id"] == entry["run_id"]
                and summary["rho"] == entry["rho"] and summary["round"] == entry["round"],
                "Candidate summary identity differs")
        require(summary["target_linears"] == 614 and summary["completed_epochs"] == 2
                and summary["updates"] == 16 and summary["test_forward_count"] == 0
                and summary["teacher_forward_count"] == 0 and summary["final_test_context_count"] == 0
                and summary["selection_context_count"] == 16
                and summary["initial_selection_context_count"] == 8, "Candidate training/selection coverage differs")
        close(summary["normalization_c"], c, "candidate normalization")
        close(summary["latent_weight"], entry["rho"] * c, "candidate lambda")
        history = summary["selection_history"]
        require(len(history) == 3 and [row["epoch"] for row in history] == [0, 1, 2],
                "All epoch0/1/2 selection points are required")
        source_jobs.append(entry["job_id"])
        for row in history:
            measured = epoch_metrics(row, protocol["selection_cases"])
            if row["epoch"] == 0:
                if baseline is None:
                    baseline = measured
                else:
                    for name in measured:
                        close(measured[name], baseline[name], "common micro parent " + name)
            points.append({"run_id": entry["run_id"], "job_id": entry["job_id"],
                           "result_dir": str(result), "round": entry["round"], "rho": entry["rho"],
                           "latent_weight": summary["latent_weight"], "epoch": row["epoch"],
                           "metrics": measured})
    search = protocol["search"]
    require(search["guardrail_fraction"] == 0.2 and search["near_tie_fraction"] == 0.01,
            "Approved 20% guards or near-tie policy changed")
    points = eligible_points(points, baseline, search["guardrail_fraction"], search["guardrails"])
    chosen, strict, unconstrained = select(points, search["near_tie_fraction"])
    job = os.environ["PBS_JOBID"]
    choice = {
        "status": "complete", "protocol": PROTOCOL, "selection_job_id": job,
        "rounds_complete": args.round, "candidate_count": len(candidates),
        "normalization_c": c, "loss_normalization": normalization,
        "guardrail_fraction": 0.2, "near_tie_fraction": 0.01,
        "selection_source_jobs": source_jobs, "baseline": baseline,
        "selected_run_id": chosen["run_id"], "selected_job_id": chosen["job_id"],
        "selected_result_dir": chosen["result_dir"], "selected_epoch": chosen["epoch"], "selected_round": chosen["round"],
        "rho": chosen["rho"], "latent_weight": chosen["latent_weight"],
        "selected_metrics": chosen["metrics"], "strict_eligible_minimum": strict,
        "unconstrained_minimum": unconstrained, "points": points,
        "epoch0_fallback": chosen["epoch"] == 0,
        "effective_latent_training": chosen["epoch"] > 0 and chosen["rho"] > 0,
        "model_forward_count": 0, "test_used_for_selection": False,
    }
    write(args.out / "choice.json", choice)
    if args.round == 1:
        next_round = {"protocol": PROTOCOL, "round": 2, "selection_job_id": job,
                      "selection_source_jobs": source_jobs,
                      "candidates": round2_candidates(chosen, first_rhos)}
        write(args.out / "round2_candidates.json", next_round)
        publish(root / "round2_candidates.json", next_round, ("protocol", "round", "candidates", "selection_source_jobs"))
    else:
        publish(root / "optimal_selection.json", choice,
                ("protocol", "rounds_complete", "selected_run_id", "selected_job_id", "selected_epoch", "rho",
                 "latent_weight", "normalization_c", "selection_source_jobs"))
    summary = {"status": "complete", "protocol": PROTOCOL, "job_id": job, "round": args.round,
               "candidate_count": len(candidates), "normalization_c": c, "selected_run_id": chosen["run_id"],
               "selected_epoch": chosen["epoch"], "rho": chosen["rho"], "latent_weight": chosen["latent_weight"],
               "rounds_complete": args.round, "model_forward_count": 0, "selection_source_jobs": source_jobs}
    write(args.out / "summary.json", summary)
    (args.out / "CHOICE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, allow_nan=False))


def self_check():
    # Numeric tests belong to the same CPU PBS allocation as the chooser.
    guard()
    specs = first_round_candidates({"search": {"round1_rhos": [0.1, 0.5, 1.0]},
                                    "additional_round1_candidates": [
                                        {"run_id": "r1-r003", "rho": 0.7},
                                        {"run_id": "r1-r004", "rho": 1.5}]})
    assert [(item["run_id"], item["rho"]) for item in specs[-2:]] == [
        ("r1-r003", 0.7), ("r1-r004", 1.5)]
    assert math.isclose(p90([1, 2, 3, 4]), 3.7)
    base = {"mean_motor_rmse_late22": 1.0, "mean_gripper_rmse_first10": 1.0,
            "p90_motor_rmse_first10": 1.0}
    point = {"run_id": "a", "rho": 0.1, "epoch": 1,
             "metrics": {**base, "mean_motor_rmse_first10": 0.5}}
    rejected = {**point, "run_id": "b", "rho": 1.0,
                "metrics": {**point["metrics"], "mean_motor_rmse_first10": 0.1,
                            "mean_motor_rmse_late22": 1.2001}}
    boundary = {**point, "run_id": "c", "rho": 0.5,
                "metrics": {**point["metrics"], "mean_motor_rmse_first10": 0.498,
                            "mean_motor_rmse_late22": 1.2}}
    points = eligible_points([point, rejected, boundary], base, 0.2, list(base))
    assert points[0]["eligible"] and not points[1]["eligible"] and points[2]["eligible"]
    chosen, strict, unconstrained = select(points, 0.01)
    assert chosen["run_id"] == "a" and strict["run_id"] == "c" and unconstrained["run_id"] == "b"
    assert [item["rho"] for item in round2_candidates({"epoch": 2, "rho": 0.1})][:2] == [0.0, 0.03]
    assert round2_candidates({"epoch": 2, "rho": 1.0})[-1]["rho"] == 2.0
    assert round2_candidates({"epoch": 0, "rho": 1.0})[1]["rho"] == 0.03
    expanded = [0.1, 0.5, 0.7, 1.0, 1.5]
    interior = round2_candidates({"epoch": 2, "rho": 0.7}, expanded)
    assert math.isclose(interior[1]["rho"], math.sqrt(0.5 * 0.7))
    assert math.isclose(interior[2]["rho"], math.sqrt(0.7 * 1.0))
    maximum = round2_candidates({"epoch": 2, "rho": 1.5}, expanded)
    assert math.isclose(maximum[1]["rho"], math.sqrt(1.0 * 1.5)) and maximum[2]["rho"] == 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--round", type=int, choices=(1, 2))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("CHOOSER_SELF_CHECK_PASSED")
    else:
        require(all((args.protocol, args.inputs, args.round, args.out)), "Missing selection arguments")
        run(args)


if __name__ == "__main__":
    main()
