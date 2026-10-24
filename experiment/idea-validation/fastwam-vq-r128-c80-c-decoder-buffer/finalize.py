"""Validate staged runtime parity and summarize the frozen full-panel profile."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import traceback


REFERENCE_MODES = ("bf16", "eager", "input_fused", "layer_decode_fp32")
REUSED_MODES = ("runtime_c",)
CANDIDATE_MODES = ("c_decoder", "c_buffer")
ALL_MODES = REFERENCE_MODES + REUSED_MODES + CANDIDATE_MODES
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
            or profile.get("queries_per_mode") != 80
            or profile.get("warmup_queries") != 3
            or profile.get("steps_video") != 10
            or profile.get("steps_action") != 10
            or profile.get("action_horizon") != 32
            or profile.get("video_frames") != 9
            or profile.get("action_shape") != [32, 7]):
        raise ValueError("Protocol must preserve the frozen 80-case profile and 3 warmups")
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
        "c_decoder": (),
        "c_buffer": ("c_decoder",),
    }
    return mode in dependencies and all(gates.get(name, {}).get("passed") is True
                                        for name in dependencies.get(mode, ()))


def complete_modes(modes, gates, reused_modes=(), reuse_sources=None):
    reuse_sources = reuse_sources or {}

    def candidate_ok(name):
        row = modes[name]
        provenance = row.get("provenance", {})
        if name in reused_modes:
            source = reuse_sources.get(name, {})
            binding = (
                row.get("status") == "reused_candidate"
                and provenance.get("reused") is True
                and provenance.get("source_job") == source.get("source_job")
                and provenance.get("protocol") == source.get("source_protocol")
            )
        else:
            binding = row.get("status") == "measured" and provenance.get("reused") is False
        return (
            binding
            and row.get("queries") == 80
            and row.get("parity_vs_primary", {}).get("passed") is True
            and row.get("parity_vs_primary", {}).get("max_abs") == 0.0
            and row.get("validation_checks", {}).get("passed") is True
        )

    return (
        set(modes) == set(ALL_MODES)
        and all(modes[name].get("status") == "cached" for name in REFERENCE_MODES)
        and all(candidate_ok(name) for name in REUSED_MODES + CANDIDATE_MODES)
        and all(gates.get(name, {}).get("passed") is True for name in CANDIDATE_MODES)
    )


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, (dict, list)):
        raise ValueError("Expected JSON object or list: " + str(path))
    return value


def decoder_checkpoint_matches(decoder_report, buffer_report):
    """Bind the reused tuning producer independently of the buffer executor."""
    return (
        isinstance(decoder_report, dict) and isinstance(buffer_report, dict)
        and bool(decoder_report.get('checkpoint_identity'))
        and decoder_report.get('checkpoint_identity') == buffer_report.get('checkpoint_identity')
        and decoder_report.get('protocol') == buffer_report.get('protocol')
        and bool(decoder_report.get('pbs_jobid'))
        and decoder_report.get('pbs_jobid') == buffer_report.get('reuse_source_pbs_jobid')
    )


def reference_identity(p, mode):
    reuse = p["reference_reuse"]
    if mode == reuse["primary_mode"]:
        return (reuse["primary_profiling_protocol"], reuse["primary_profiling_job"])
    return reuse["profiling_protocol"], reuse["profiling_job"]


COHORT_FIELDS = (
    "case_ids", "seed_indices", "queries", "warmup_queries", "steps_video",
    "steps_action", "batch_size", "action_horizon", "video_frames",
)


def expected_cohort(p):
    profile = p["profiling"]
    return {
        "case_ids": profile["case_ids"],
        "seed_indices": profile["seed_indices"],
        "queries": profile["queries_per_mode"],
        "warmup_queries": profile["warmup_queries"],
        "steps_video": profile["steps_video"],
        "steps_action": profile["steps_action"],
        "batch_size": profile["batch_size"],
        "action_horizon": profile["action_horizon"],
        "video_frames": profile["video_frames"],
    }


def validate_cohort(p, summary, mode):
    expected = expected_cohort(p)
    if "cohort" in summary:
        cohort = summary["cohort"]
        if (not isinstance(cohort, dict)
                or any(cohort.get(key) != value for key, value in expected.items())):
            raise ValueError("Mode summary cohort or inference geometry differs: " + mode)
        return "mode_summary"

    if mode not in REFERENCE_MODES:
        raise ValueError("Candidate mode summary lacks its cohort receipt: " + mode)
    reuse = p["reference_reuse"]
    artifact_root = (
        reuse["primary_artifact_root"] if mode == reuse["primary_mode"]
        else reuse["artifact_root"]
    )
    protocol_path = Path(artifact_root) / "protocol.json"
    source_protocol = read_json(protocol_path)
    protocol_id, _ = reference_identity(p, mode)
    if (not isinstance(source_protocol, dict)
            or source_protocol.get("protocol") != protocol_id
            or not isinstance(source_protocol.get("profiling"), dict)):
        raise ValueError("Legacy reference protocol is missing or misbound: " + mode)
    profile = source_protocol["profiling"]
    source_geometry = {
        "case_ids": profile.get("case_ids"),
        "seed_indices": profile.get("seed_indices"),
        "queries": profile.get("queries_per_mode"),
        "warmup_queries": profile.get("warmup_queries"),
        "steps_video": profile.get("steps_video"),
        "steps_action": profile.get("steps_action"),
        "batch_size": profile.get("batch_size"),
        "action_horizon": profile.get("action_horizon"),
        "video_frames": profile.get("video_frames"),
    }
    if source_geometry != expected:
        raise ValueError("Legacy reference protocol cohort or inference geometry differs: " + mode)
    return "source_protocol:" + str(protocol_path)


def mode_checks(summary, validation, mode):
    flags = {key: summary.get(key) is expected
             for key, expected in REQUIRED_NUMERICS.items()}
    flags["training_updates_zero"] = type(summary.get("new_training_updates")) is int \
        and summary.get("new_training_updates") == 0
    flags["closed_loop_episodes_zero"] = type(summary.get("closed_loop_episodes")) is int \
        and summary.get("closed_loop_episodes") == 0
    flags["compiled_disabled"] = summary.get("compiled") is False
    flags["manual_cuda_graph_enabled"] = summary.get("cuda_graph") is True
    accuracy = summary.get("performance_accuracy", {})
    anchor = validation.get("source_anchor_parity", {})
    for field, short in (
        ("anchor_action_max_abs", "action_max_abs"),
        ("anchor_future_max_abs", "future_max_abs"),
    ):
        value = summary.get(field, accuracy.get(field, anchor.get(short)))
        flags[field + "_eq_0"] = (
            isinstance(value, (int, float)) and math.isfinite(value) and value == 0.0
        )
        flags["validation_" + field + "_eq_0"] = (
            isinstance(anchor.get(short), (int, float))
            and math.isfinite(anchor[short]) and anchor[short] == 0.0
        )
    flags["strict_anchor_gate_passed"] = (
        validation.get("strict_anchor_gate") is True
        and anchor.get("passed") is True
        and accuracy.get("anchor_passed") is True
    )
    if mode in CANDIDATE_MODES:
        optimization = summary.get("optimization", {})
        measured = optimization.get("after_primary", {}) if isinstance(optimization, dict) else {}
        graph = measured.get("graph", {}) if isinstance(measured, dict) else {}
        regions = graph.get("regions", {}) if isinstance(graph, dict) else {}
        expected = {"video_step": 800, "video_prefill": 80, "action_step": 800}
        flags["expected_runtime_kind"] = graph.get("kind") == "cuda_graph"
        flags["only_expected_graph_regions"] = isinstance(regions, dict) and set(regions) == set(expected)
        if isinstance(regions, dict):
            for label, calls in expected.items():
                row = regions.get(label, {})
                flags[label + "_captured"] = (
                    type(row.get("captures")) is int and row.get("captures") >= 1
                )
                flags[label + "_replays_exact"] = (
                    type(row.get("replays")) is int and row.get("replays") == calls
                )
        logical_per_query = optimization.get("logical_target_linear_calls_per_query") \
            if isinstance(optimization, dict) else None
        flags["logical_coverage_5258_per_query"] = (
            isinstance(logical_per_query, (int, float)) and logical_per_query == 5258.0
        )
        for metric in ("full80_action_max_abs_vs_layer_decode_fp32",
                       "full80_action_rmse_vs_layer_decode_fp32"):
            value = accuracy.get(metric)
            flags[metric + "_eq_0"] = (
                isinstance(value, (int, float)) and math.isfinite(value) and value == 0.0
            )
        parity = validation.get("full80_action_parity_vs_layer_decode_fp32", {})
        flags["validation_full80_action_max_abs_eq_0"] = parity.get("max_abs") == 0.0
        flags["validation_full80_action_rmse_eq_0"] = parity.get("rmse") == 0.0
        flags["validation_full80_exact_80"] = (
            parity.get("exact_queries") == 80 and parity.get("passed") is True
        )
        decoder_optimization = summary.get("decoder_optimization")
        decoder_receipt_present = isinstance(decoder_optimization, dict)
        if not decoder_receipt_present:
            decoder_optimization = {}
        parity = decoder_optimization.get("parity", {})
        frequency = decoder_optimization.get("frequency_panel", {})
        weighted = decoder_optimization.get("frequency_weighted", {})
        module_rows = decoder_optimization.get("modules", [])
        flags["decoder_optimization_receipt_present"] = decoder_receipt_present
        flags["decoder_bitwise_all_modules"] = (
            parity.get("checked_modules") == 614
            and parity.get("bitwise_passed") == 614
            and parity.get("candidate_failures") == []
        )
        flags["decoder_receipt_bound_to_job"] = (
            decoder_optimization.get("protocol") == summary.get("protocol")
            and decoder_optimization.get("pbs_jobid") == summary.get("pbs_jobid")
            and decoder_optimization.get("mode") == mode
            and isinstance(decoder_optimization.get("checkpoint_identity"), dict)
        )
        reuse_source = decoder_optimization.get("reuse_source")
        flags["decoder_reuse_source_bound"] = (
            reuse_source is None if mode == "c_decoder" else
            isinstance(reuse_source, str)
            and Path(reuse_source).name == "tuning.json"
            and Path(reuse_source).parent.name == "c_decoder"
        )
        flags["decoder_reuse_source_job_recorded"] = (
            mode != "c_buffer"
            or (isinstance(decoder_optimization.get("reuse_source_pbs_jobid"), str)
                and bool(decoder_optimization["reuse_source_pbs_jobid"]))
        )
        flags["decoder_frequency_weights_full80"] = (
            decoder_optimization.get("call_frequency_status") == "complete"
            and frequency.get("query_count") == 80
            and frequency.get("logical_decode_calls_full_panel") == 420640
            and frequency.get("logical_decode_calls_per_query") == 5258.0
            and isinstance(module_rows, list) and len(module_rows) == 614
            and all(isinstance(row, dict) and type(row.get("calls")) is int
                    and row["calls"] >= 0 for row in module_rows)
            and sum(row["calls"] for row in module_rows) == 420640
        )
        flags["decoder_weighted_estimate_reported"] = (
            weighted.get("query_count") == 80
            and isinstance(weighted.get("reference_decode_ms_full_panel"), (int, float))
            and math.isfinite(weighted["reference_decode_ms_full_panel"])
            and weighted["reference_decode_ms_full_panel"] > 0
            and isinstance(weighted.get("selected_decode_ms_full_panel"), (int, float))
            and math.isfinite(weighted["selected_decode_ms_full_panel"])
            and weighted["selected_decode_ms_full_panel"] > 0
            and isinstance(weighted.get("selected_over_reference"), (int, float))
            and math.isfinite(weighted["selected_over_reference"])
        )
        flags["validation_decoder_receipt_matches"] = (
            validation.get("decoder_optimization") == summary.get("decoder_optimization")
        )
        layout = summary.get("installation", {}).get("activation_workspace_layout", {})
        if mode == "c_decoder":
            expected_layout = {
                "logical_bytes": 42147840,
                "borrowed_weight_scratch_bytes": 0,
                "independent_bytes": 42147840,
                "alias_weight_scratch": False,
                "h_uses_weight_storage": False,
                "u_uses_weight_storage": False,
                "a4_independent_of_weight_storage": True,
            }
        else:
            expected_layout = {
                "logical_bytes": 42147840,
                "borrowed_weight_scratch_bytes": 33718272,
                "independent_bytes": 8429568,
                "alias_weight_scratch": True,
                "h_uses_weight_storage": True,
                "u_uses_weight_storage": True,
                "a4_independent_of_weight_storage": True,
            }
        flags["activation_workspace_layout_exact"] = all(
            layout.get(key) == value for key, value in expected_layout.items()
        )
        flags["mode_runtime_flags"] = summary.get("cuda_graph") is True and summary.get("compiled") is False
        parameter_execution = summary.get("parameter_execution", {})
        scope = parameter_execution.get("coverage_scope", "")
        flags["parameter_execution_probe_scope_is_limited"] = (
            isinstance(parameter_execution, dict)
            and isinstance(scope, str)
            and "anchor" in scope.lower() and "warmup" in scope.lower()
        )
    return {"passed": all(flags.values()), "checks": flags}


def load_mode(p, root, mode, keys, torch, current_job):
    directory = root / mode
    summary = read_json(directory / "summary.json")
    completed_reuse = p.get("completed_candidate_reuse", {})
    if mode in REFERENCE_MODES:
        protocol, job_id = reference_identity(p, mode)
    elif mode in completed_reuse.get("modes", []):
        protocol, job_id = completed_reuse["source_protocol"], completed_reuse["source_job"]
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

    cohort_source = validate_cohort(p, summary, mode)

    actions = torch.load(directory / "actions.pt", map_location="cpu", weights_only=True)
    shape = tuple(p["profiling"]["action_shape"])
    if not isinstance(actions, dict) or set(actions) != keys:
        raise ValueError("Wrong action cohort: " + mode)
    for key, action in actions.items():
        if (not torch.is_tensor(action) or tuple(action.shape) != shape
                or not bool(torch.isfinite(action).all())):
            raise ValueError("Action shape or finite-value check failed: " + mode + "/" + key)

    validation_path = directory / "validation.json"
    parity_path = directory / "parity.json"
    if mode in CANDIDATE_MODES + REUSED_MODES and not validation_path.is_file():
        raise ValueError("Candidate validation receipt is missing: " + mode)
    validation = read_json(validation_path) if validation_path.is_file() else {}
    if mode in CANDIDATE_MODES + REUSED_MODES and not isinstance(validation, dict):
        raise ValueError("Candidate validation receipt is malformed: " + mode)
    if mode in CANDIDATE_MODES + REUSED_MODES:
        if (validation.get("protocol") != protocol or validation.get("pbs_jobid") != job_id
                or validation.get("mode") != mode or validation.get("complete") is not True):
            raise ValueError("Candidate validation receipt is incomplete or misbound: " + mode)
        if not parity_path.is_file():
            raise ValueError("Candidate parity receipt is missing: " + mode)
        parity_receipt = read_json(parity_path)
        if (not isinstance(parity_receipt, dict)
                or parity_receipt.get("full80_queries") != 80
                or parity_receipt.get("full80_action_max_abs_vs_layer_decode_fp32") != 0.0
                or parity_receipt.get("full80_action_rmse_vs_layer_decode_fp32") != 0.0
                or parity_receipt.get("anchor_action_max_abs") != 0.0
                or parity_receipt.get("anchor_future_max_abs") != 0.0):
            raise ValueError("Candidate exact parity receipt failed: " + mode)
    else:
        parity_receipt = {}
    checks = mode_checks(summary, validation, mode) if mode in CANDIDATE_MODES else {"passed": True, "checks": {}}
    tuning = None
    if mode in CANDIDATE_MODES:
        tuning_path = directory / "tuning.json"
        if not tuning_path.is_file():
            raise ValueError("Candidate decoder tuning receipt is missing: " + mode)
        tuning = read_json(tuning_path)
        if (not isinstance(tuning, dict)
                or tuning != summary.get("decoder_optimization")
                or not isinstance(summary.get("decoder_optimization"), dict)):
            raise ValueError("Candidate decoder optimization receipt is malformed: " + mode)
        parameter_path = directory / "parameter_execution.json"
        if not parameter_path.is_file():
            raise ValueError("Candidate parameter-execution probe receipt is missing: " + mode)
        parameter_execution = read_json(parameter_path)
        if (not isinstance(parameter_execution, dict)
                or parameter_execution != summary.get("parameter_execution")):
            raise ValueError("Parameter-execution summary differs from its receipt: " + mode)
        checks["checks"].update(
            full80_cohort=True,
            action_shape_finite=True,
            latency_and_memory_samples_finite=True,
            tuning_receipt_present=True,
            parameter_execution_receipt_matches=True,
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
        "cohort_source": cohort_source,
        "samples": samples,
        "actions": actions,
        "validation": validation,
        "parity_receipt": parity_receipt,
        "tuning": tuning,
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
        "decoder_optimization": compact_decoder_report(summary.get("decoder_optimization")),
        "parameter_execution_probe": summary.get("parameter_execution"),
        "diagnostic_profiler": summary.get("diagnostic_profiler"),
    }


def compact_decoder_report(report):
    """Keep aggregate control receipts small; full per-module evidence stays in tuning.json."""
    if not isinstance(report, dict):
        return None
    shapes = []
    for row in report.get("shapes", []):
        if not isinstance(row, dict):
            continue
        selected = row.get("selected")
        selected_row = row.get("reference", {}) if selected == "reference" else next(
            (candidate for candidate in row.get("candidates", [])
             if isinstance(candidate, dict) and candidate.get("name") == selected), {}
        )
        shapes.append({
            "shape": row.get("shape"),
            "module_count": row.get("module_count"),
            "call_count": row.get("call_count"),
            "selected": selected,
            "decision_reason": row.get("decision_reason"),
            "reference_median_ms": row.get("reference", {}).get("median_ms"),
            "selected_median_ms": selected_row.get("median_ms"),
            "selected_bitwise": selected_row.get("bitwise", True if selected == "reference" else None),
        })
    keys = (
        "schema", "protocol", "pbs_jobid", "mode", "checkpoint_identity", "device",
        "expected_modules", "module_count", "tuning_method", "call_frequency_status",
        "call_frequency_source", "frequency_panel", "frequency_weighted", "reuse_source",
        "selected_by_shape", "parity",
    )
    return {**{key: report[key] for key in keys if key in report}, "shapes": shapes}


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


def storage_inventory_report(p, root):
    target = root / "storage_inventory.json"
    modules_path = root / "storage_inventory_modules.json"
    markdown_path = root / "inventory.md"
    try:
        receipt = read_json(target)
        modules = read_json(modules_path)
        markdown_bytes = markdown_path.stat().st_size
    except (OSError, ValueError, json.JSONDecodeError):
        return {"passed": False, "status": "missing_or_invalid"}
    module_files = modules.get("files", []) if isinstance(modules, dict) else []
    module_files_complete = (
        isinstance(modules, dict) and isinstance(module_files, list) and bool(module_files)
        and all(isinstance(row, dict) and isinstance(row.get("path"), str)
                and (root / row["path"]).is_file() for row in module_files)
    )
    model_state = receipt.get("model_state", {}) if isinstance(receipt, dict) else {}
    manifest = receipt.get("storage_manifest_validation", {}) if isinstance(receipt, dict) else {}
    residency_scope = receipt.get("residency_scope", "") if isinstance(receipt, dict) else ""
    passed = (
        isinstance(receipt, dict)
        and receipt.get("phase") == "pre"
        and receipt.get("protocol") == p["protocol"]
        and receipt.get("bundle_path") == p["export_reuse"]["bundle_path"]
        and receipt.get("expected_bytes_match") is True
        and receipt.get("expected_unique_storage_bytes") == p["storage_inventory"]["expected_unique_storage_bytes"]
        and model_state.get("unique_storage_bytes") == p["storage_inventory"]["expected_unique_storage_bytes"]
        and manifest.get("model_state_match") is True
        and isinstance(receipt.get("components"), list)
        and isinstance(modules, (dict, list))
        and module_files_complete
        and markdown_bytes > 0
        and isinstance(residency_scope, str)
        and "CPU-mapped" in residency_scope
        and "not a CUDA allocator measurement" in residency_scope
        and p.get("storage_inventory", {}).get("no_model_forward") is True
        and p.get("storage_inventory", {}).get("no_compile") is True
    )
    return {
        "passed": passed,
        "status": "valid" if passed else "invalid",
        "bundle_path": receipt.get("bundle_path") if isinstance(receipt, dict) else None,
        "expected_bytes_match": receipt.get("expected_bytes_match") if isinstance(receipt, dict) else None,
        "component_count": len(receipt.get("components", [])) if isinstance(receipt, dict) else None,
        "module_detail_files": len(module_files) if isinstance(module_files, list) else None,
        "residency_scope": residency_scope,
        "receipt_bytes": target.stat().st_size,
        "modules_receipt_bytes": modules_path.stat().st_size,
        "markdown_bytes": markdown_bytes,
    }


def final_report(p, root, current_job, torch):
    keys = expected_keys(p)
    atol = p["stages"]["parity"]["max_abs_action_atol"]
    primary_mode = p["stages"]["parity"]["primary_mode"]
    gate_rows = read_gates(root / "stage_gates.json")
    candidate_reuse = p.get("candidate_reuse", {})
    completed_reuse = p.get("completed_candidate_reuse", {})
    base_reuse_modes = set(candidate_reuse.get("modes", ()))
    completed_reuse_modes = set(completed_reuse.get("modes", ()))
    reused_modes = base_reuse_modes | completed_reuse_modes
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

    source_stage_gate = {}
    source_gate_path = root / "runtime_c" / "source_stage_gates.json"
    if source_gate_path.is_file():
        try:
            source_stage_gate = read_json(source_gate_path).get("modes", {}).get("runtime_c", {})
        except (OSError, ValueError, json.JSONDecodeError, AttributeError):
            source_stage_gate = {}

    for mode in REUSED_MODES + CANDIDATE_MODES:
        is_base_reused = mode in base_reuse_modes
        is_completed_reuse = mode in completed_reuse_modes
        is_reused = is_base_reused or is_completed_reuse
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
            if is_base_reused:
                mode_gate = {
                    "mode": mode,
                    "attempted": True,
                    "passed": source_stage_gate.get("passed") is True,
                    "source_protocol": candidate_reuse.get("source_protocol"),
                    "source_job": candidate_reuse.get("source_job"),
                    "source_stage_gate": source_stage_gate,
                }
            else:
                mode_gate = gate_rows.get(mode)
                if not isinstance(mode_gate, dict):
                    mode_gate = {
                        "mode": mode, "attempted": True, "passed": False,
                        "status": "gate_missing", "reason": "No persisted stage parity gate",
                    }
                elif mode_gate.get("passed") is True:
                    mode_gate = dict(mode_gate)
                gates[mode] = mode_gate
            validation_checks = dict(row["validation_checks"])
            if is_base_reused:
                validation_checks["passed"] = (
                    validation_checks.get("passed") is True
                    and source_stage_gate.get("passed") is True
                )
                validation_checks["checks"] = dict(validation_checks.get("checks", {}))
                validation_checks["checks"]["source_individual_stage_gate_passed"] = (
                    source_stage_gate.get("passed") is True
                )
            source = candidate_reuse if is_base_reused else completed_reuse if is_completed_reuse else {}
            modes[mode] = {
                "status": "reused_candidate" if is_reused else "measured",
                "queries": 80,
                "provenance": {
                    "protocol": row["summary"]["protocol"],
                    "pbs_jobid": row["summary"]["pbs_jobid"],
                    "gpu": row["summary"].get("gpu"),
                    "gpu_uuid": row["summary"].get("gpu_uuid"),
                    "reused": is_reused,
                    "source_job": row["summary"]["pbs_jobid"] if is_reused else None,
                    "source_result_root": source.get("result_root") if is_reused else None,
                },
                "validation_checks": validation_checks,
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
            if is_base_reused:
                modes[mode]["source_stage_gate"] = source_stage_gate
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
            if not is_reused:
                gates[mode] = modes[mode]["stage_gate"]

    # c_buffer must reuse the decoder choices selected by c_decoder.
    if "c_decoder" in loaded and "c_buffer" in loaded:
        decoder_report = loaded["c_decoder"]["summary"].get("decoder_optimization", {})
        buffer_report = loaded["c_buffer"]["summary"].get("decoder_optimization", {})
        def selected_shapes(report):
            return {
                json.dumps(row.get("shape"), sort_keys=True): row.get("selected")
                for row in report.get("shapes", []) if isinstance(row, dict)
            } if isinstance(report, dict) else {}
        selection_match = bool(selected_shapes(decoder_report)) and (
            selected_shapes(decoder_report) == selected_shapes(buffer_report)
        )
        def frequency_signature(report):
            rows = report.get("modules", []) if isinstance(report, dict) else []
            return {row.get("name"): row.get("calls") for row in rows if isinstance(row, dict)}
        frequency_match = bool(frequency_signature(decoder_report)) and (
            frequency_signature(decoder_report) == frequency_signature(buffer_report)
        )
        checkpoint_match = decoder_checkpoint_matches(decoder_report, buffer_report)
        source_job_match = (
            isinstance(buffer_report, dict)
            and buffer_report.get("reuse_source_pbs_jobid") == (
                completed_reuse.get("source_job") if completed_reuse.get('modes')
                else decoder_report.get('pbs_jobid'))
        )
        modes["c_buffer"].setdefault("validation_checks", {"passed": False, "checks": {}})
        modes["c_buffer"]["validation_checks"]["checks"]["decoder_tuning_matches_c_decoder"] = selection_match
        modes["c_buffer"]["validation_checks"]["checks"]["decoder_frequency_panel_matches_c_decoder"] = frequency_match
        modes["c_buffer"]["validation_checks"]["checks"]["decoder_checkpoint_matches_c_decoder"] = checkpoint_match
        modes["c_buffer"]["validation_checks"]["checks"]["decoder_source_pbs_job_matches_completed_reuse"] = source_job_match
        modes["c_buffer"]["validation_checks"]["passed"] = (
            modes["c_buffer"]["validation_checks"].get("passed") is True
            and selection_match and frequency_match and checkpoint_match and source_job_match
        )

    inventory = storage_inventory_report(p, root)
    reuse_sources = {}
    for source in (candidate_reuse, completed_reuse):
        for name in source.get("modes", ()):
            reuse_sources[name] = {
                "source_job": source.get("source_job"),
                "source_protocol": source.get("source_protocol"),
            }
    complete = complete_modes(
        modes, gates, reused_modes, reuse_sources
    ) and inventory["passed"]
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
        "reused_base_modes": list(REUSED_MODES),
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
        "completed_candidate_reuse": {
            "modes": list(completed_reuse.get("modes", ())),
            "source_protocol": completed_reuse.get("source_protocol"),
            "source_job": completed_reuse.get("source_job"),
            "source_job_name": completed_reuse.get("source_job_name"),
            "source_result_root": completed_reuse.get("result_root"),
        },
        "attempted_candidate_modes": attempted,
        "measured_candidate_modes": measured,
        "reused_candidate_modes": [mode for mode in CANDIDATE_MODES if mode in completed_reuse_modes],
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
        "storage_inventory": inventory,
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
        "# Fast-WAM R128 C80 C decoder/buffer profile",
        "",
        "Frozen full panel: contexts 3000-3039 x seeds 0/1 (80 queries per mode).",
        "The BF16/eager/input_fused/layer_decode_fp32 and runtime_c panels are reused; runtime_c is not rerun.",
        "Both new modes are compared to the cached layer_decode_fp32 panel with exact full80 action parity and exact anchor action/future parity.",
        "Decoder operand checks cover all 614 modules. The c_buffer stage runs only after c_decoder passes its parity gate.",
        "Latency and memory regressions are reported as outcomes and do not determine parity success.",
        "Storage inventory runs before profiling without a model forward or compilation. No training or closed-loop episodes.",
        "",
        "| Mode | Status | Mean ms | p50 ms | p95 ms | Query peak GB | Load peak GB | Max abs vs primary | Parity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in ALL_MODES:
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
        "Measured candidate modes: " + ", ".join(report["measured_candidate_modes"]),
        "Storage inventory passed: " + str(report["storage_inventory"].get("passed")),
        "Per-mode decoder tuning, Graph, workspace, validation, and performance reports are in summary.json and tuning.json.",
        "",
    ])
    rows.append("")
    (root / "RESULTS.zh.md").write_text("\n".join(rows), encoding="utf-8")
    print("C_DECODER_BUFFER_FINAL " + json.dumps({
        "complete": report["complete"],
        "measured": report["measured_candidate_modes"],
        "inventory_passed": report["storage_inventory"].get("passed"),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
