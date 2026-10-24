"""Validate staged runtime parity and summarize the frozen full-panel profile."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import traceback


CANDIDATE_MODES = ("runtime_a", "runtime_b", "runtime_c", "runtime_compile")
REFERENCE_MODES = ("bf16", "eager", "input_fused", "layer_decode_fp32")
REQUIRED_NUMERICS = {
    "deterministic_algorithms": True,
    "fill_uninitialized_memory": True,
    "allow_tf32": False,
    "allow_bf16_reduced_precision_reduction": True,
}


def expected_keys(p):
    profile = p["profiling"]
    cases = list(range(3000, 3040))
    if (profile.get("case_ids") != cases or profile.get("seed_indices") != [0, 1]
            or profile.get("queries_per_mode") != 80):
        raise ValueError("Protocol must preserve contexts 3000-3039 x seeds 0/1 = 80")
    return {f"{case}_{seed}" for case in cases for seed in (0, 1)}


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize empty values")
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parity_pass(max_abs, atol):
    return (isinstance(max_abs, (int, float)) and math.isfinite(max_abs)
            and isinstance(atol, (int, float)) and math.isfinite(atol)
            and max_abs <= atol)


def stage_allowed(mode, gates):
    dependencies = {
        "runtime_a": (),
        "runtime_b": ("runtime_a",),
        "runtime_c": ("runtime_b",),
        "runtime_compile": ("runtime_b",),
    }
    return mode in dependencies and all(gates.get(name, {}).get("passed") is True
                                        for name in dependencies.get(mode, ()))


def complete_modes(modes, gates, reused_modes=(), reuse_source_job=None):
    def candidate_ok(name):
        row = modes[name]
        provenance = row.get("provenance", {})
        if name in reused_modes:
            binding = (
                row.get("status") == "reused_candidate"
                and provenance.get("reused") is True
                and provenance.get("source_job") == reuse_source_job
            )
        else:
            binding = row.get("status") == "measured" and provenance.get("reused") is False
        return (
            binding
            and row.get("queries") == 80
            and row.get("parity_vs_primary", {}).get("passed") is True
            and row.get("validation_checks", {}).get("passed") is True
        )

    return (
        set(modes) == set(REFERENCE_MODES + CANDIDATE_MODES)
        and all(modes[name].get("status") == "cached" for name in REFERENCE_MODES)
        and all(candidate_ok(name) for name in CANDIDATE_MODES)
        and all(gates.get(name, {}).get("passed") is True for name in CANDIDATE_MODES)
    )


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, (dict, list)):
        raise ValueError("Expected JSON object or list: " + str(path))
    return value


def reference_identity(p, mode):
    reuse = p["reference_reuse"]
    if mode == reuse["primary_mode"]:
        return (reuse["primary_profiling_protocol"], reuse["primary_profiling_job"])
    return reuse["profiling_protocol"], reuse["profiling_job"]


def mode_checks(summary, validation, mode):
    flags = {key: summary.get(key) is expected
             for key, expected in REQUIRED_NUMERICS.items()}
    flags["training_updates_zero"] = type(summary.get("new_training_updates")) is int \
        and summary.get("new_training_updates") == 0
    flags["closed_loop_episodes_zero"] = type(summary.get("closed_loop_episodes")) is int \
        and summary.get("closed_loop_episodes") == 0
    flags["compiled_flag_present"] = type(summary.get("compiled")) is bool
    flags["cuda_graph_flag_present"] = type(summary.get("cuda_graph")) is bool
    accuracy = summary.get("performance_accuracy", {})
    anchor = validation.get("source_anchor_parity", {})
    for field, short in (
        ("anchor_action_max_abs", "action_max_abs"),
        ("anchor_future_max_abs", "future_max_abs"),
    ):
        value = summary.get(field, accuracy.get(field, anchor.get(short)))
        flags[field + "_le_1e-6"] = (
            isinstance(value, (int, float)) and math.isfinite(value) and value <= 1e-6
        )
    if mode in ("runtime_c", "runtime_compile"):
        optimization = summary.get("optimization", {})
        measured = optimization.get("after_primary", {}) if isinstance(optimization, dict) else {}
        graph = measured.get("graph", {}) if isinstance(measured, dict) else {}
        regions = graph.get("regions", {}) if isinstance(graph, dict) else {}
        expected = {"video_step": 800, "video_prefill": 80, "action_step": 800}
        flags["expected_runtime_kind"] = graph.get("kind") == (
            "cuda_graph" if mode == "runtime_c" else "torch_compile"
        )
        flags["only_expected_regions"] = isinstance(regions, dict) and set(regions) == set(expected)
        if isinstance(regions, dict):
            for label, calls in expected.items():
                row = regions.get(label, {})
                if mode == "runtime_c":
                    flags[label + "_captured"] = (
                        type(row.get("captures")) is int and row.get("captures") >= 1
                    )
                    flags[label + "_replays_exact"] = (
                        type(row.get("replays")) is int and row.get("replays") == calls
                    )
                else:
                    flags[label + "_compile_calls_exact"] = (
                        type(row.get("compile_calls")) is int and row.get("compile_calls") == calls
                    )
        flags["runtime_mode_flag"] = (
            summary.get("cuda_graph") is True and summary.get("compiled") is False
            if mode == "runtime_c"
            else summary.get("compiled") is True and summary.get("cuda_graph") is False
        )
        if mode == "runtime_compile":
            unique = graph.get("unique_compiled_graphs") if isinstance(graph, dict) else None
            flags["compiled_graphs_positive"] = type(unique) is int and unique > 0
    return {"passed": all(flags.values()), "checks": flags}


def load_mode(p, root, mode, keys, torch, current_job):
    directory = root / mode
    summary = read_json(directory / "summary.json")
    if mode in REFERENCE_MODES:
        protocol, job_id = reference_identity(p, mode)
    elif mode in p.get("candidate_reuse", {}).get("modes", []):
        reuse = p["candidate_reuse"]
        protocol, job_id = reuse["source_protocol"], reuse["source_job"]
    else:
        protocol, job_id = p["protocol"], current_job
    if (summary.get("protocol") != protocol or summary.get("pbs_jobid") != job_id
            or summary.get("mode") != mode or summary.get("complete") is not True
            or summary.get("queries") != 80):
        raise ValueError("Incomplete or incorrectly bound summary: " + mode)

    samples = read_json(directory / "samples.json")
    if not isinstance(samples, list) or len(samples) != 80:
        raise ValueError("Expected 80 latency samples: " + mode)
    sample_keys = set()
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Malformed latency sample: " + mode)
        sample_keys.add(f"{row.get('case_id')}_{row.get('seed_index')}")
        for metric in ("full_query_wall_ms", "peak_allocated_gpu_GB"):
            value = row.get(metric)
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0):
                raise ValueError(f"Invalid {metric} sample: {mode}")
    if sample_keys != keys:
        raise ValueError("Wrong or duplicated latency cohort: " + mode)

    actions = torch.load(directory / "actions.pt", map_location="cpu", weights_only=True)
    shape = tuple(p["profiling"]["action_shape"])
    if not isinstance(actions, dict) or set(actions) != keys:
        raise ValueError("Wrong action cohort: " + mode)
    for key, action in actions.items():
        if (not torch.is_tensor(action) or tuple(action.shape) != shape
                or not bool(torch.isfinite(action).all())):
            raise ValueError("Action shape or finite-value check failed: " + mode + "/" + key)

    validation_path = directory / "validation.json"
    if mode in CANDIDATE_MODES and not validation_path.is_file():
        raise ValueError("Candidate validation receipt is missing: " + mode)
    validation = read_json(validation_path) if validation_path.is_file() else {}
    if mode in CANDIDATE_MODES and not isinstance(validation, dict):
        raise ValueError("Candidate validation receipt is malformed: " + mode)
    if mode in CANDIDATE_MODES and validation.get("mode", mode) != mode:
        raise ValueError("Candidate validation mode differs: " + mode)
    checks = mode_checks(summary, validation, mode) if mode in CANDIDATE_MODES else {"passed": True, "checks": {}}
    if mode in CANDIDATE_MODES:
        checks["checks"].update(
            full80_cohort=True,
            action_shape_finite=True,
            latency_and_memory_samples_finite=True,
        )
    load_memory = summary.get("load_transient_memory", {})
    load_peak = load_memory.get("peak_allocated_bytes") if isinstance(load_memory, dict) else None
    if (not isinstance(load_peak, (int, float)) or not math.isfinite(load_peak)
            or load_peak < 0):
        raise ValueError("Missing or invalid load peak memory: " + mode)
    values = {
        "full_query_wall_ms": [float(row["full_query_wall_ms"]) for row in samples],
        "peak_allocated_gpu_GB": [float(row["peak_allocated_gpu_GB"]) for row in samples],
    }
    performance = {
        metric: {
            "mean": statistics.fmean(rows),
            "p50": percentile(rows, .50),
            "p95": percentile(rows, .95),
            "maximum": max(rows),
        }
        for metric, rows in values.items()
    }
    return {
        "summary": summary,
        "samples": samples,
        "actions": actions,
        "validation": validation,
        "validation_checks": checks,
        "load_peak_allocated_bytes": int(load_peak),
        "performance": performance,
    }


def compare_actions(torch, actual, reference, keys, atol):
    max_abs = 0.0
    squared_sum = 0.0
    value_count = 0
    exact = True
    for key in sorted(keys):
        left = actual[key].detach().float().cpu()
        right = reference[key].detach().float().cpu()
        delta = left - right
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Non-finite full-panel action difference: " + key)
        max_abs = max(max_abs, float(delta.abs().max()))
        squared_sum += float(delta.square().sum())
        value_count += delta.numel()
        exact = exact and bool(torch.equal(actual[key], reference[key]))
    rmse = math.sqrt(squared_sum / value_count) if value_count else math.nan
    return {
        "exact": exact,
        "max_abs": max_abs,
        "rmse": rmse,
        "atol": atol,
        "passed": parity_pass(max_abs, atol),
        "queries": len(keys),
        "comparison": "full80_actions_vs_cached_layer_decode_fp32",
    }


def read_gates(path):
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
        return value.get("modes", {}) if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def write_gates(path, p, current_job, gates):
    from reference_profile import save
    save(path, dict(protocol=p["protocol"], pbs_jobid=current_job, modes=gates))


def stage_gate(p, root, mode, current_job, torch):
    gates_path = root / "stage_gates.json"
    gates = read_gates(gates_path)
    row = {
        "mode": mode,
        "attempted": True,
        "passed": False,
        "primary_mode": p["stages"]["parity"]["primary_mode"],
        "atol": p["stages"]["parity"]["max_abs_action_atol"],
    }
    try:
        if mode not in CANDIDATE_MODES:
            raise ValueError("Unknown candidate mode: " + mode)
        if not stage_allowed(mode, gates):
            missing = [
                dep for dep in p["stages"]["dependencies"][mode]
                if gates.get(dep, {}).get("passed") is not True
            ]
            row.update(status="blocked", blocked_by=missing, reason="Previous parity gate did not pass")
        else:
            keys = expected_keys(p)
            reference = load_mode(p, root, p["stages"]["parity"]["primary_mode"],
                                  keys, torch, current_job)
            candidate = load_mode(p, root, mode, keys, torch, current_job)
            comparison = compare_actions(
                torch, candidate["actions"], reference["actions"], keys, row["atol"]
            )
            row.update(
                status="measured",
                parity_vs_primary=comparison,
                validation_checks=candidate["validation_checks"],
                passed=(comparison["passed"] and candidate["validation_checks"]["passed"]),
            )
    except Exception as exc:
        row.update(
            status="invalid",
            reason=f"{type(exc).__name__}: {exc}",
            traceback_tail=traceback.format_exc().splitlines()[-8:],
        )
    gates[mode] = row
    write_gates(gates_path, p, current_job, gates)
    print("STAGE_PARITY_GATE " + json.dumps(row, allow_nan=False), flush=True)
    return 0 if row["passed"] else 3


def diagnostics(summary, validation):
    optimization = summary.get("optimization", {})
    measured = optimization.get("after_primary", {}) if isinstance(optimization, dict) else {}
    runtime = {
        key: measured[key]
        for key in ("graph", "compiled", "logical_coverage")
        if isinstance(measured, dict) and key in measured
    }
    validation_rows = {
        key: validation[key]
        for key in ("source_anchor_parity", "full80_action_parity_vs_layer_decode_fp32",
                    "output_shapes", "cohort")
        if isinstance(validation, dict) and key in validation
    }
    return {
        "after_primary_runtime_report": runtime,
        "validation": validation_rows,
        "diagnostic_profiler": summary.get("diagnostic_profiler"),
    }


def performance_report(loaded, reference):
    def ratio(numerator, denominator):
        if denominator == 0:
            return None
        value = numerator / denominator
        return value if math.isfinite(value) else None

    perf = loaded["performance"]
    baseline = reference["performance"]
    peak = perf["peak_allocated_gpu_GB"]["maximum"]
    baseline_peak = baseline["peak_allocated_gpu_GB"]["maximum"]
    load_gb = loaded["load_peak_allocated_bytes"] / 1e9
    baseline_load_gb = reference["load_peak_allocated_bytes"] / 1e9
    latency = perf["full_query_wall_ms"]
    baseline_latency = baseline["full_query_wall_ms"]
    return {
        "full_query_wall_ms": latency,
        "peak_allocated_gpu_GB": {"maximum": peak},
        "load_peak_allocated_bytes": loaded["load_peak_allocated_bytes"],
        "load_peak_allocated_GB": load_gb,
        "load_seconds": loaded["summary"].get("load_seconds"),
        "load_peak_reserved_bytes": loaded["summary"].get(
            "load_transient_memory", {}
        ).get("peak_reserved_bytes"),
        "setup_validation_warmup_seconds": loaded["summary"].get(
            "setup_validation_warmup_seconds"
        ),
        "setup_peak_allocated_bytes": loaded["summary"].get("setup_peak_allocated_bytes"),
        "setup_peak_reserved_bytes": loaded["summary"].get("setup_peak_reserved_bytes"),
        "ratios_vs_primary": {
            "mean_latency": ratio(latency["mean"], baseline_latency["mean"]),
            "p50_latency": ratio(latency["p50"], baseline_latency["p50"]),
            "p95_latency": ratio(latency["p95"], baseline_latency["p95"]),
            "peak_allocated_gpu_memory": ratio(peak, baseline_peak),
            "load_peak_allocated_memory": ratio(load_gb, baseline_load_gb),
        },
        "regressions_are_reported_not_gate_failures": True,
    }


def storage_report(p, root):
    source = Path(p["reference_reuse"]["primary_result_root"]) / "storage.json"
    target = root / "storage.json"
    if not source.is_file() or not target.is_file():
        return {"unchanged": False, "status": "missing", "source": str(source)}
    source_bytes = source.read_bytes()
    target_bytes = target.read_bytes()
    try:
        storage = json.loads(target_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"unchanged": False, "status": "invalid_json", "source": str(source)}
    return {
        "unchanged": source_bytes == target_bytes,
        "status": "unchanged" if source_bytes == target_bytes else "changed",
        "source": str(source),
        "receipt_bytes": len(target_bytes),
        "receipt_fields": sorted(storage) if isinstance(storage, dict) else None,
    }


def final_report(p, root, current_job, torch):
    keys = expected_keys(p)
    atol = p["stages"]["parity"]["max_abs_action_atol"]
    primary_mode = p["stages"]["parity"]["primary_mode"]
    gate_rows = read_gates(root / "stage_gates.json")
    candidate_reuse = p.get("candidate_reuse", {})
    reused_modes = set(candidate_reuse.get("modes", ()))
    loaded, modes, gates = {}, {}, {}

    for mode in REFERENCE_MODES:
        try:
            row = load_mode(p, root, mode, keys, torch, current_job)
            loaded[mode] = row
            modes[mode] = {
                "status": "cached",
                "queries": 80,
                "provenance": {
                    "protocol": row["summary"]["protocol"],
                    "pbs_jobid": row["summary"]["pbs_jobid"],
                    "reused": True,
                },
                "performance": performance_report(row, row) if mode == primary_mode else {
                    "full_query_wall_ms": row["performance"]["full_query_wall_ms"],
                    "peak_allocated_gpu_GB": {
                        "maximum": row["performance"]["peak_allocated_gpu_GB"]["maximum"]
                    },
                    "load_peak_allocated_bytes": row["load_peak_allocated_bytes"],
                },
                "validation_checks": {"passed": True, "checks": {
                    "full80_cohort": True, "action_shape_finite": True,
                    "latency_and_memory_samples_finite": True,
                }},
            }
        except Exception as exc:
            modes[mode] = {
                "status": "invalid",
                "queries": 0,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    primary_actions = loaded[primary_mode]["actions"] if primary_mode in loaded else None
    if primary_actions is not None:
        for mode in REFERENCE_MODES:
            if mode in loaded:
                modes[mode]["parity_vs_primary"] = compare_actions(
                    torch, loaded[mode]["actions"], primary_actions, keys, atol
                )
    for mode in CANDIDATE_MODES:
        failure_path = root / mode / "failure.json"
        failure = read_json(failure_path) if failure_path.is_file() else None
        try:
            row = load_mode(p, root, mode, keys, torch, current_job)
            loaded[mode] = row
            parity = (
                compare_actions(torch, row["actions"], primary_actions, keys, atol)
                if primary_actions is not None
                else {"passed": False, "reason": "Primary reference is unavailable"}
            )
            mode_gate = gate_rows.get(mode)
            if not isinstance(mode_gate, dict):
                mode_gate = {
                    "mode": mode, "attempted": True, "passed": False,
                    "status": "gate_missing", "reason": "No persisted stage parity gate",
                }
            elif mode_gate.get("passed") is True:
                mode_gate = dict(mode_gate)
            gates[mode] = mode_gate
            modes[mode] = {
                "status": "reused_candidate" if mode in reused_modes else "measured",
                "queries": 80,
                "provenance": {
                    "protocol": row["summary"]["protocol"],
                    "pbs_jobid": row["summary"]["pbs_jobid"],
                    "gpu": row["summary"].get("gpu"),
                    "gpu_uuid": row["summary"].get("gpu_uuid"),
                    "reused": mode in reused_modes,
                    "source_job": row["summary"]["pbs_jobid"] if mode in reused_modes else None,
                    "source_result_root": candidate_reuse.get("result_root") if mode in reused_modes else None,
                },
                "validation_checks": row["validation_checks"],
                "numerical_flags": {
                    key: row["summary"].get(key) for key in REQUIRED_NUMERICS
                } | {
                    "compiled": row["summary"].get("compiled"),
                    "cuda_graph": row["summary"].get("cuda_graph"),
                    "new_training_updates": row["summary"].get("new_training_updates"),
                    "closed_loop_episodes": row["summary"].get("closed_loop_episodes"),
                },
                "performance": performance_report(row, loaded[primary_mode])
                    if primary_mode in loaded else None,
                "parity_vs_primary": parity,
                "stage_gate": mode_gate,
                "diagnostics": diagnostics(row["summary"], row["validation"]),
            }
        except Exception as exc:
            status = "failed" if failure is not None else "not_measured"
            modes[mode] = {
                "status": status,
                "queries": 0,
                "reason": f"{type(exc).__name__}: {exc}",
                "failure": failure,
                "stage_gate": gate_rows.get(mode, {
                    "mode": mode, "attempted": failure is not None,
                    "passed": False, "status": status,
                }),
            }
            gates[mode] = modes[mode]["stage_gate"]

    storage = storage_report(p, root)
    complete = complete_modes(
        modes, gates, reused_modes, candidate_reuse.get("source_job")
    ) and storage["unchanged"]
    attempted = [
        mode for mode in CANDIDATE_MODES
        if modes[mode]["status"] in ("measured", "reused_candidate", "failed")
    ]
    measured = [mode for mode in CANDIDATE_MODES if modes[mode]["status"] == "measured"]
    return {
        "protocol": p["protocol"],
        "pbs_jobid": current_job,
        "complete": complete,
        "deliverable_complete": complete,
        "pipeline_status": "complete" if complete else "attempted_incomplete",
        "candidate_modes": list(CANDIDATE_MODES),
        "stage_order": list(p["stages"]["order"]),
        "stage_dependencies": p["stages"]["dependencies"],
        "candidate_reuse": {
            "modes": list(candidate_reuse.get("modes", ())),
            "source_protocol": candidate_reuse.get("source_protocol"),
            "source_job": candidate_reuse.get("source_job"),
            "source_job_name": candidate_reuse.get("source_job_name"),
            "source_result_root": candidate_reuse.get("result_root"),
            "whole_pipeline_completion_required": candidate_reuse.get(
                "whole_pipeline_completion_required"
            ),
        },
        "attempted_candidate_modes": attempted,
        "measured_candidate_modes": measured,
        "reused_candidate_modes": [mode for mode in CANDIDATE_MODES if mode in reused_modes],
        "full_queries_per_mode": 80,
        "cohort": {
            "contexts": p["profiling"]["case_ids"],
            "seed_indices": p["profiling"]["seed_indices"],
            "queries": 80,
        },
        "primary_comparator": {
            "mode": primary_mode,
            "protocol": p["reference_reuse"]["primary_profiling_protocol"],
            "pbs_jobid": p["reference_reuse"]["primary_profiling_job"],
            "result_root": p["reference_reuse"]["primary_result_root"],
            "parity_atol": atol,
        },
        "stage_gates": gates,
        "modes": modes,
        "storage_reuse": storage,
        "new_training_updates": 0 if all(
            modes.get(mode, {}).get("numerical_flags", {}).get("new_training_updates") == 0
            for mode in measured
        ) else None,
        "closed_loop_episodes": 0 if all(
            modes.get(mode, {}).get("numerical_flags", {}).get("closed_loop_episodes") == 0
            for mode in measured
        ) else None,
        "performance_regressions_fatal": False,
        "execution_boundary": p["execution_boundary"],
    }


def main():
    from reference_profile import guard, save
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-gate", choices=CANDIDATE_MODES)
    parser.add_argument("--phase", choices=("final",))
    args = parser.parse_args()
    if bool(args.stage_gate) == bool(args.phase):
        parser.error("choose exactly one of --stage-gate or --phase final")
    p = read_json(Path(os.environ["ARTIFACTS"]) / "protocol.json")
    root = Path(p["remote_root"]) / "results"
    import torch
    current_job = os.environ["PBS_JOBID"]
    if args.stage_gate:
        return stage_gate(p, root, args.stage_gate, current_job, torch)
    report = final_report(p, root, current_job, torch)
    save(root / "stage_gates.json", dict(
        protocol=p["protocol"], pbs_jobid=current_job, modes=report["stage_gates"]
    ))
    save(root / "summary.json", report)
    rows = [
        "# Fast-WAM R128 C80 Runtime ABC profile",
        "",
        "Frozen full panel: contexts 3000-3039 x seeds 0/1 (80 queries per mode).",
        "All candidate action outputs are compared against the cached layer_decode_fp32 panel.",
        "Exact equality and max absolute error <= 1e-6 are reported; latency and memory regressions are not parity failures.",
        "The configured diagnostic profiler uses one extra query after the measured 80 and does not enter primary latency.",
        "No baseline reruns, new training updates, or closed-loop episodes.",
        "",
        "| Mode | Status | Mean ms | p50 ms | p95 ms | Query peak GB | Load peak GB | Max abs vs primary | Parity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in REFERENCE_MODES + CANDIDATE_MODES:
        row = report["modes"][mode]
        perf = row.get("performance") or {}
        latency = perf.get("full_query_wall_ms", {})
        peak = perf.get("peak_allocated_gpu_GB", {}).get("maximum")
        load_peak = perf.get("load_peak_allocated_GB")
        parity = row.get("parity_vs_primary", {})
        rows.append(
            f"| {mode} | {row.get('status')} | "
            f"{latency.get('mean', math.nan):.6f} | "
            f"{latency.get('p50', math.nan):.6f} | "
            f"{latency.get('p95', math.nan):.6f} | "
            f"{peak if peak is not None else math.nan:.6f} | "
            f"{load_peak if load_peak is not None else math.nan:.6f} | "
            f"{parity.get('max_abs', math.nan):.9g} | "
            f"{parity.get('passed', False)} |"
        )
    rows.extend([
        "",
        "Deliverable complete: " + str(report["complete"]),
        "Attempted modes: " + ", ".join(report["attempted_candidate_modes"]),
        "Measured modes: " + ", ".join(report["measured_candidate_modes"]),
        "Per-mode validation flags and CUDA graph/compile diagnostics are in summary.json.",
        "",
    ])
    for mode in ("runtime_c", "runtime_compile"):
        row = report["modes"].get(mode, {})
        graph = (row.get("diagnostics", {}).get("after_primary_runtime_report", {})
                 .get("graph", {}))
        regions = graph.get("regions", {}) if isinstance(graph, dict) else {}
        evidence = {
            "kind": graph.get("kind"),
            "regions": regions,
            "unique_compiled_graphs": graph.get("unique_compiled_graphs"),
            "dynamo_counters": graph.get("dynamo_counters"),
            "gate_passed": row.get("validation_checks", {}).get("passed", False),
        }
        rows.append(mode + " execution evidence: " + json.dumps(evidence, ensure_ascii=False))
    rows.append("")
    (root / "RESULTS.zh.md").write_text("\n".join(rows), encoding="utf-8")
    print("RUNTIME_ABC_FINAL " + json.dumps({
        "complete": report["complete"],
        "attempted": report["attempted_candidate_modes"],
        "measured": report["measured_candidate_modes"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
