"""Aggregate completed Fast-WAM phase-1/2 diagnostics inside a PBS CPU allocation.

Input contract: ROOT/plan.json, ROOT/terminal_evidence.json, and
ROOT/results/case_00..21/{case_summary.json,queries.jsonl,traces/...}. The
terminal evidence contains one entry per case with case_id, job_id,
exit_status, marker_path, and exit_code_path. No model or third-party package
is imported here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket
import statistics
from typing import Any


PROTOCOL = "fastwam-a4-phases12-v1"
PLUS_DIMENSIONS = (
    "camera_viewpoints", "light_conditions", "background_textures",
    "objects_layout", "robot_initial_states", "language_instructions",
)
BITS = tuple((v, a, p) for v in (4, 8) for a in (4, 8) for p in (4, 8))
PHASE1_TYPES = {"bf16", "all_a8", "all_a4", "independent_reference"}
COUNTERS = ("integer_gemm_calls", "native_int4_gemm_calls", "kv_packed_prefills", "kv_layer_reads")
TRACE_METRICS = (
    "input_absmax", "input_rms",
    "a4_scale_min", "a4_scale_mean", "a4_scale_max",
    "a4_zero_code_fraction", "a4_input_relative_rmse",
    "a8_scale_min", "a8_scale_mean", "a8_scale_max",
    "a8_zero_code_fraction", "a8_input_relative_rmse",
    "native_vs_a4_reference_rmse", "native_vs_a4_reference_max_abs",
    "native_vs_a4_reference_normalized_rmse",
    "a4_vs_a8_reference_rmse", "a4_vs_a8_reference_normalized_rmse",
)


class AggregationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregationError(message)


def allocation_guard() -> None:
    job_id = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    host = socket.gethostname().split(".", 1)[0]
    if not job_id or not nodefile or "login" in host.lower():
        raise AggregationError("aggregation requires an active non-login PBS allocation")
    nodes = {line.split(".", 1)[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise AggregationError(f"host {host} is absent from PBS_NODEFILE")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read valid JSON {path}: {exc}") from exc


def _rooted(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise AggregationError(f"{label} must be relative to the diagnostic root: {value}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AggregationError(f"{label} escapes the diagnostic root: {value}") from exc
    return resolved


def _artifact_path(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AggregationError(f"{label} escapes the diagnostic root: {value}") from exc
    return resolved


def _check_terminal_evidence(root: Path, evidence_path: Path) -> list[dict[str, Any]]:
    evidence = _json(evidence_path)
    cases = evidence.get("cases") if isinstance(evidence, dict) else None
    _require(isinstance(cases, list) and len(cases) == 22,
             "terminal_evidence.json must contain exactly 22 case entries")
    by_id: dict[int, dict[str, Any]] = {}
    for entry in cases:
        _require(isinstance(entry, dict), "terminal evidence entry must be an object")
        case_id = entry.get("case_id")
        _require(type(case_id) is int and 0 <= case_id < 22, f"invalid terminal evidence case_id: {case_id!r}")
        _require(case_id not in by_id, f"duplicate terminal evidence for case_{case_id:02d}")
        _require(isinstance(entry.get("job_id"), str) and bool(entry["job_id"].strip()),
                 f"case_{case_id:02d} has no PBS job_id")
        _require(type(entry.get("exit_status")) is int and entry["exit_status"] == 0,
                 f"case_{case_id:02d} PBS Exit_status is not 0")
        marker = _artifact_path(root, entry.get("marker_path", ""), f"case_{case_id:02d} marker")
        exit_code = _artifact_path(root, entry.get("exit_code_path", ""), f"case_{case_id:02d} exit code")
        _require(marker.name == "PIPELINE_COMPLETE" and marker.is_file(),
                 f"case_{case_id:02d} is missing its PIPELINE_COMPLETE marker")
        _require(exit_code.name == "exit_code.txt" and exit_code.is_file(),
                 f"case_{case_id:02d} is missing exit_code.txt")
        _require(marker.parent == exit_code.parent and marker.parent.name == entry["job_id"],
                 f"case_{case_id:02d} terminal artifacts do not match job_id")
        _require(exit_code.read_text(encoding="utf-8").strip() == "0",
                 f"case_{case_id:02d} artifact exit_code.txt is not 0")
        by_id[case_id] = {
            "case_id": case_id,
            "job_id": entry["job_id"],
            "exit_status": 0,
            "pipeline_complete": True,
            "marker_path": marker.relative_to(root.resolve()).as_posix(),
            "exit_code_path": exit_code.relative_to(root.resolve()).as_posix(),
        }
    _require(set(by_id) == set(range(22)), "terminal evidence does not cover cases 0..21")
    return [by_id[index] for index in range(22)]


def _load_plan(root: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    plan = _json(root / "plan.json")
    _require(isinstance(plan, dict) and plan.get("protocol") == PROTOCOL,
             f"plan.json protocol must be {PROTOCOL}")
    rows = plan.get("inputs")
    _require(isinstance(rows, list) and len(rows) == 22, "plan.json must contain 22 inputs")
    by_id: dict[int, dict[str, Any]] = {}
    dimensions: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    all_seeds: set[int] = set()
    for row in rows:
        _require(isinstance(row, dict), "plan input must be an object")
        case_id = row.get("case_id")
        _require(type(case_id) is int and 0 <= case_id < 22 and case_id not in by_id,
                 f"invalid or duplicate plan case_id: {case_id!r}")
        _require(row.get("domain") in ("original", "plus"), f"case_{case_id:02d} has invalid domain")
        domains[row["domain"]] += 1
        if row["domain"] == "original":
            _require(row.get("dimension") == "unperturbed", f"original case_{case_id:02d} is not unperturbed")
            _require(row.get("source_index") is None, f"original case_{case_id:02d} has a Plus source index")
        else:
            dimension = row.get("dimension")
            _require(dimension in PLUS_DIMENSIONS, f"case_{case_id:02d} has invalid Plus dimension")
            dimensions[dimension] += 1
            _require(type(row.get("source_index")) is int, f"Plus case_{case_id:02d} lacks source_index")
        _require(row.get("state_id") == 0, f"case_{case_id:02d} plan state_id is not 0")
        seeds = row.get("sampler_seeds")
        _require(isinstance(seeds, list) and len(seeds) == 2 and all(type(s) is int for s in seeds)
                 and seeds[0] != seeds[1], f"case_{case_id:02d} must have two distinct integer sampler seeds")
        _require(not (set(seeds) & all_seeds), f"sampler seeds are reused across cases at case_{case_id:02d}")
        all_seeds.update(seeds)
        metadata_file = row.get("metadata_file")
        _require(isinstance(metadata_file, str) and bool(metadata_file),
                 f"case_{case_id:02d} has no metadata_file")
        meta_path = _rooted(root, metadata_file, f"case_{case_id:02d} metadata")
        metadata = _json(meta_path)
        for key in ("case_id", "domain", "dimension", "task_id", "environment_seed", "state_id"):
            _require(metadata.get(key) == row.get(key),
                     f"case_{case_id:02d} metadata {key} disagrees with plan.json")
        _require(metadata.get("state_id") == 0 and metadata.get("settling_steps") == 30,
                 f"case_{case_id:02d} metadata must use state 0 and 30 settling steps")
        _require(metadata.get("evaluation_episodes") == 0 and metadata.get("episode_count") == 0,
                 f"case_{case_id:02d} metadata claims counted episodes")
        _require(isinstance(metadata.get("statepath"), str) and bool(metadata["statepath"]),
                 f"case_{case_id:02d} metadata lacks statepath provenance")
        _require(isinstance(metadata.get("dataset_import_paths"), dict)
                 and bool(metadata["dataset_import_paths"]),
                 f"case_{case_id:02d} metadata lacks dataset provenance")
        by_id[case_id] = {**row, "_metadata": metadata}
    _require(set(by_id) == set(range(22)), "plan case IDs must be exactly 0..21")
    _require(domains == {"original": 10, "plus": 12}, f"expected 10 original and 12 Plus cases, got {dict(domains)}")
    _require(dimensions == {dimension: 2 for dimension in PLUS_DIMENSIONS},
             f"expected two cases per Plus dimension, got {dict(dimensions)}")
    return plan, by_id


def _read_queries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AggregationError(f"missing query records: {path}")
    records = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    raise AggregationError(f"blank JSONL line at {path}:{line_no}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AggregationError(f"JSONL record at {path}:{line_no} is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read query JSONL {path}: {exc}") from exc
    return records


def _bits_label(bits: tuple[int, int, int]) -> str:
    return f"v{bits[0]}a{bits[1]}p{bits[2]}"


def _action_metrics(action: list[Any], baseline: list[Any], context: str) -> dict[str, float | None]:
    def values(rows: list[Any], label: str) -> list[list[float]]:
        _require(isinstance(rows, list) and len(rows) == 32
                 and all(isinstance(row, list) and len(row) == 7 for row in rows),
                 f"{context} {label} action must have shape 32x7")
        result = []
        for row in rows:
            converted = []
            for value in row:
                _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value),
                         f"{context} {label} action contains a non-finite/non-numeric value")
                converted.append(float(value))
            result.append(converted)
        return result

    action_values, reference_values = values(action, "query"), values(baseline, "reference")
    delta = [[value - ref for value, ref in zip(row, ref_row)]
             for row, ref_row in zip(action_values, reference_values)]

    def rmse(items: list[float]) -> float:
        return math.sqrt(sum(value * value for value in items) / len(items))

    def flatten(rows: list[list[float]]) -> list[float]:
        return [value for row in rows for value in row]

    motor_first10 = flatten([row[:6] for row in delta[:10]])
    gripper_first10 = [row[6] for row in delta[:10]]
    motor_32 = flatten([row[:6] for row in delta])
    gripper_32 = [row[6] for row in delta]
    all_delta = flatten(delta)
    reference_motor_first10 = flatten([row[:6] for row in reference_values[:10]])
    reference_all = flatten(reference_values)
    motor_rmse = rmse(motor_first10)
    global_rmse = rmse(all_delta)
    motor_rms = rmse(reference_motor_first10)
    global_rms = rmse(reference_all)
    return {
        "motor_rmse_first10": motor_rmse,
        "normalized_motor_rmse_first10": motor_rmse / motor_rms if motor_rms > 0 else None,
        "gripper_rmse_first10": rmse(gripper_first10),
        "motor_rmse_32": rmse(motor_32),
        "gripper_rmse_32": rmse(gripper_32),
        "rmse_32x7": global_rmse,
        "normalized_rmse_32x7": global_rmse / global_rms if global_rms > 0 else None,
        "reference_motor_rms_first10": motor_rms,
        "reference_action_rms_32x7": global_rms,
        "max_abs_first10": max(abs(value) for value in flatten(delta[:10])),
        "max_abs_32": max(abs(value) for value in all_delta),
    }


def _verify_saved_action_metrics(saved: Any, measured: dict[str, float | None], suffix: str,
                                 context: str) -> None:
    _require(isinstance(saved, dict), f"{context} lacks saved action metrics")
    mapping = {
        "motor_rmse_first10": f"motor_rmse_first10_vs_{suffix}",
        "gripper_rmse_first10": f"gripper_rmse_first10_vs_{suffix}",
        "motor_rmse_32": f"motor_rmse_32_vs_{suffix}",
        "gripper_rmse_32": f"gripper_rmse_32_vs_{suffix}",
        "max_abs_first10": f"max_abs_first10_vs_{suffix}",
        "max_abs_32": f"max_abs_32_vs_{suffix}",
    }
    for key, saved_key in mapping.items():
        value = saved.get(saved_key)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                 f"{context} saved metric {saved_key} is missing/non-finite")
        _require(math.isclose(float(value), float(measured[key]), rel_tol=1e-5, abs_tol=1e-6),
                 f"{context} saved metric {saved_key} disagrees with raw actions")


def _delta_metrics(record: dict[str, Any], context: str) -> dict[str, int]:
    counters = record.get("runtime_counters")
    _require(isinstance(counters, dict), f"{context} lacks runtime_counters")
    before, after, delta = counters.get("before"), counters.get("after"), counters.get("delta")
    _require(all(isinstance(value, dict) for value in (before, after, delta)),
             f"{context} runtime_counters needs before/after/delta")
    output = {}
    for key in COUNTERS:
        values = (before.get(key), after.get(key), delta.get(key))
        _require(all(type(value) is int and value >= 0 for value in values),
                 f"{context} counter {key} must be a nonnegative integer")
        _require(after[key] - before[key] == delta[key], f"{context} counter delta mismatch for {key}")
        output[key] = delta[key]
    return output


def _validate_scheduler_coverage(record: dict[str, Any], context: str) -> None:
    coverage = record.get("stage_scheduler_coverage")
    _require(isinstance(coverage, dict), f"{context} lacks stage_scheduler_coverage")
    wanted = list(range(10))
    for stage in ("video", "action"):
        item = coverage.get(stage)
        _require(isinstance(item, dict) and item.get("schedule_calls") == 1
                 and item.get("step_calls") == 10 and item.get("observed_steps") == wanted,
                 f"{context} lacks full executed {stage} scheduler coverage")


def _scope_for_module(module: str) -> str | None:
    parts = module.split(".")
    if "video_expert" in parts:
        return "video"
    if "action_expert" in parts:
        return "action"
    if "proprio_encoder" in parts:
        return "proprio"
    return None


def _trace_file(root: Path, case_dir: Path, trace_path: str, context: str) -> Path:
    _require(isinstance(trace_path, str) and bool(trace_path), f"{context} has no trace_path")
    path = Path(trace_path)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        first = (case_dir / path).resolve()
        resolved = first if first.is_file() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AggregationError(f"{context} trace_path escapes the diagnostic root") from exc
    _require(resolved.is_file(), f"{context} missing trace file {trace_path}")
    return resolved


def _process_a4_trace(root: Path, case_dir: Path, row: dict[str, Any], record: dict[str, Any],
                      local_values: dict[tuple[str, str, int], dict[str, list[float]]],
                      layer_values: dict[tuple[str, str, int, str], list[float]]) -> None:
    case_id, seed, query_id = row["case_id"], row["sampler_seed"], row["query_id"]
    context = f"case_{case_id:02d}/{query_id}"
    info = record.get("trace_coverage")
    _require(isinstance(info, dict), f"{context} lacks trace_coverage")
    trace_file = _trace_file(root, case_dir, info.get("trace_path", ""), context)
    trace = _json(trace_file)
    _require(trace.get("diagnostic") == "fastwam-stage-quantization-trace-v1"
             and trace.get("completed") is True, f"{context} trace is incomplete or has an unknown schema")
    trace_context = trace.get("context", {})
    _require(trace_context.get("case_id") == case_id and trace_context.get("query_id") == query_id
             and trace_context.get("sampler_seed") == seed,
             f"{context} trace context does not match query record")
    trace_records, coverage = trace.get("records"), trace.get("coverage")
    _require(isinstance(trace_records, list) and trace_records and isinstance(coverage, list),
             f"{context} trace lacks per-call records or coverage")
    _require(type(trace.get("calls")) is int and trace["calls"] == len(trace_records),
             f"{context} trace calls does not match records")
    observed_modules = set()
    observed_scopes: dict[str, set[str]] = defaultdict(set)
    actual_stage_steps: dict[str, set[int]] = {"video": set(), "action": set()}
    calls_by_key: Counter[tuple[Any, ...]] = Counter()
    trace_call_scopes: dict[str, set[str]] = defaultdict(set)
    previous_call_index = None
    for local in trace_records:
        call_index = local.get("call_index")
        _require(type(call_index) is int and call_index >= 0,
                 f"{context} trace call_index is invalid")
        if previous_call_index is not None:
            _require(call_index == previous_call_index + 1,
                     f"{context} trace call_index is not continuous")
        previous_call_index = call_index
        module, scope, stage, step = (local.get(key) for key in ("module", "scope", "stage", "step"))
        _require(isinstance(module, str) and bool(module) and scope in ("video", "action", "proprio"),
                 f"{context} trace has an invalid module/scope")
        _require(_scope_for_module(module) == scope, f"{context} trace module/scope mismatch: {module}/{scope}")
        _require(isinstance(stage, str) and type(step) is int, f"{context} trace lacks stage/step")
        if stage in actual_stage_steps:
            _require(0 <= step < 10, f"{context} trace has invalid {stage} step {step}")
            actual_stage_steps[stage].add(step)
        else:
            _require(stage in ("conditioning", "video_conditioning_prefill") and step == -1,
                     f"{context} trace has unexpected stage/step {stage}/{step}")
        in_shape, out_shape = local.get("input_shape"), local.get("output_shape")
        _require(isinstance(in_shape, list) and isinstance(out_shape, list),
                 f"{context} trace lacks input/output shapes")
        key = (module, scope, stage, step, tuple(in_shape), tuple(out_shape))
        calls_by_key[key] += 1
        observed_modules.add(module)
        observed_scopes[scope].add(module)
        trace_call_scopes[stage].add(scope)
        metrics = local.get("metrics")
        _require(isinstance(metrics, dict), f"{context} trace lacks scalar metrics for {module}")
        for metric in TRACE_METRICS:
            value = metrics.get(metric)
            _require(isinstance(value, (int, float)) and math.isfinite(value),
                     f"{context} trace metric {metric} missing/non-finite for {module}")
            local_values[scope, stage, step].setdefault(metric, []).append(float(value))
        layer_values[module, scope, stage, step].append(
            float(metrics["native_vs_a4_reference_normalized_rmse"])
        )
    _require(actual_stage_steps == {"video": set(range(10)), "action": set(range(10))},
             f"{context} trace misses real video/action scheduler steps")
    _require("video" in trace_call_scopes and "action" in trace_call_scopes,
             f"{context} trace contains no video or action linear calls")

    reported_coverage = Counter()
    for item in coverage:
        _require(isinstance(item, dict), f"{context} trace coverage row is not an object")
        key = (
            item.get("module"), item.get("scope"), item.get("stage"), item.get("step"),
            tuple(item.get("input_shape", [])), tuple(item.get("output_shape", [])),
        )
        count = item.get("calls")
        _require(type(count) is int and count > 0, f"{context} trace coverage calls must be positive")
        reported_coverage[key] += count
    _require(reported_coverage == calls_by_key and sum(reported_coverage.values()) == len(trace_records),
             f"{context} trace coverage does not account for every per-call record")
    _require(info.get("coverage_records") == len(coverage), f"{context} trace coverage count mismatch")
    stage_steps = info.get("stage_steps")
    _require(isinstance(stage_steps, dict)
             and stage_steps.get("video") == list(range(10))
             and stage_steps.get("action") == list(range(10)),
             f"{context} query summary misses executed stage steps")
    module_counts = Counter()
    for item in coverage:
        module_counts[item["module"]] += item["calls"]
    _require(info.get("modules") == dict(module_counts), f"{context} query trace module counts mismatch")
    scope_report = {scope: sorted(names) for scope, names in observed_scopes.items()}
    _require(info.get("scopes") == scope_report, f"{context} query trace scope coverage mismatch")

    row_modules = record.get("observed_linear_modules")
    _require(isinstance(row_modules, list) and set(row_modules) == observed_modules,
             f"{context} trace does not match executed Linear modules")
    _require(record.get("observed_linear_call_count") == len(trace_records),
             f"{context} observed Linear call count mismatch")
    row_scopes = record.get("observed_linear_scopes")
    _require(isinstance(row_scopes, dict)
             and {scope: set(names) for scope, names in row_scopes.items()} == observed_scopes,
             f"{context} trace does not match executed Linear scopes")


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _effect_summary(values: list[float]) -> dict[str, Any]:
    result = _summary(values)
    result["min"] = min(values) if values else None
    return result


def _factorial_summary(context_cells: dict[str, dict[tuple[int, int, int], float]]) -> dict[str, Any]:
    names = ("video", "action", "proprio")
    indices = {name: index for index, name in enumerate(names)}
    main = {}
    for name, index in indices.items():
        pooled4, pooled8, paired = [], [], []
        for values in context_cells.values():
            at4 = [value for bits, value in values.items() if bits[index] == 4]
            at8 = [value for bits, value in values.items() if bits[index] == 8]
            mean4, mean8 = sum(at4) / len(at4), sum(at8) / len(at8)
            pooled4.extend(at4)
            pooled8.extend(at8)
            paired.append(mean4 - mean8)
        main[name] = {
            "a4_mean_motor_rmse_first10": sum(pooled4) / len(pooled4),
            "a8_mean_motor_rmse_first10": sum(pooled8) / len(pooled8),
            "a4_minus_a8_motor_rmse_first10": sum(paired) / len(paired),
            "paired_context_effect": _effect_summary(paired),
        }
    interactions = {}
    for first, second in (("video", "action"), ("video", "proprio"), ("action", "proprio")):
        i, j = indices[first], indices[second]
        effects = []
        for values in context_cells.values():
            cells = {}
            for a in (4, 8):
                for b in (4, 8):
                    sample = [value for bits, value in values.items() if bits[i] == a and bits[j] == b]
                    cells[a, b] = sum(sample) / len(sample)
            effects.append(cells[4, 4] - cells[4, 8] - cells[8, 4] + cells[8, 8])
        interactions[f"{first}_x_{second}"] = {
            "contrast": "mean(4,4)-mean(4,8)-mean(8,4)+mean(8,8), averaged over third factor",
            "paired_context_difference_in_differences": _effect_summary(effects),
        }
    third_order = []
    for values in context_cells.values():
        third_order.append(sum(
            (1 if (bits[0] == 4) else -1)
            * (1 if (bits[1] == 4) else -1)
            * (1 if (bits[2] == 4) else -1) * value
            for bits, value in values.items()
        ))
    interactions["video_x_action_x_proprio"] = {
        "contrast": "signed sum over the complete 2x2x2 cells (4=+1, 8=-1)",
        "paired_context_third_order_difference": _effect_summary(third_order),
    }
    return {
        "response": "absolute RMSE over first 10 steps x 6 motor coordinates versus same-context BF16",
        "primary_metric": "motor_rmse_first10",
        "contexts": len(context_cells),
        "method": "complete 2^3 descriptive contrasts, paired within case and sampler seed; no p-values",
        "main_effects": main,
        "interactions": interactions,
    }


def _action_report(plan_rows: dict[int, dict[str, Any]],
                   actions: dict[tuple[int, int], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, float | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_types = set()
    for case_id, row in plan_rows.items():
        for seed in row["sampler_seeds"]:
            records = actions[case_id, seed]
            baseline = records["bf16"]["raw_normalized_action_32x7"]
            for query_type, record in records.items():
                if query_type == "bf16":
                    continue
                measured = _action_metrics(record["raw_normalized_action_32x7"], baseline,
                                           f"case_{case_id:02d}/{query_type}/seed{seed}")
                _verify_saved_action_metrics(record.get("action_metrics_vs_bf16"), measured,
                                             "bf16", f"case_{case_id:02d}/{query_type}/seed{seed}")
                group_keys = [(row["domain"], row["dimension"])]
                if row["domain"] == "plus":
                    group_keys.append(("plus", "all_dimensions"))
                for key in group_keys:
                    grouped[key][query_type].append(measured)
                all_types.add(query_type)
    groups = []
    ordered_keys = [("original", "unperturbed"), ("plus", "all_dimensions")]
    ordered_keys.extend(("plus", dimension) for dimension in PLUS_DIMENSIONS)
    for domain, dimension in ordered_keys:
        by_type = grouped.get((domain, dimension), {})
        groups.append({
            "domain": domain,
            "dimension": dimension,
            "contexts": len({(case_id, seed) for case_id, row in plan_rows.items()
                             if row["domain"] == domain and (dimension == "all_dimensions" or row["dimension"] == dimension)
                             for seed in row["sampler_seeds"]}),
            "by_query_type": {
                query_type: {
                    "motor_rmse_first10_vs_bf16": _summary([item["motor_rmse_first10"] for item in values]),
                    "normalized_motor_rmse_first10_vs_bf16": _summary(
                        [item["normalized_motor_rmse_first10"] for item in values
                         if item["normalized_motor_rmse_first10"] is not None]
                    ),
                    "first10_gripper_rmse_vs_bf16": _summary(
                        [item["gripper_rmse_first10"] for item in values]),
                    "motor_rmse_32_vs_bf16": _summary([item["motor_rmse_32"] for item in values]),
                    "gripper_rmse_32_vs_bf16": _summary([item["gripper_rmse_32"] for item in values]),
                    "global_rmse_32x7_vs_bf16": _summary([item["rmse_32x7"] for item in values]),
                    "normalized_global_rmse_32x7_vs_bf16": _summary(
                        [item["normalized_rmse_32x7"] for item in values
                         if item["normalized_rmse_32x7"] is not None]
                    ),
                }
                for query_type, values in sorted(by_type.items())
            },
        })
    return {
        "comparison_scope": "each action is compared only with BF16 from the same case_id and sampler_seed",
        "action_shape": [32, 7],
        "primary_metric": "first 10 steps x 6 motor coordinates RMSE in normalized action units",
        "secondary_metrics": {
            "first10_gripper_rmse": "first 10 steps x gripper coordinate",
            "motor_rmse_32": "all 32 steps x 6 motor coordinates",
            "gripper_rmse_32": "all 32 steps x gripper coordinate",
            "global_rmse_32x7": "all 32 steps x all 7 coordinates",
        },
        "normalization": "primary and secondary RMSE may be divided by the same-region BF16 RMS; undefined when that RMS is zero",
        "groups": groups,
        "query_types": sorted(all_types),
    }


def _reference_action_report(values: list[dict[str, float | None]]) -> dict[str, Any]:
    return {
        "comparison_scope": "native all-A4 raw action versus independent-reference raw action in the same case and sampler seed",
        "contexts": len(values),
        "primary_metric": "first 10 steps x 6 motor coordinates RMSE",
        "metrics": {
            "motor_rmse_first10": _summary([item["motor_rmse_first10"] for item in values]),
            "gripper_rmse_first10": _summary([item["gripper_rmse_first10"] for item in values]),
            "motor_rmse_32": _summary([item["motor_rmse_32"] for item in values]),
            "gripper_rmse_32": _summary([item["gripper_rmse_32"] for item in values]),
            "global_rmse_32x7": _summary([item["rmse_32x7"] for item in values]),
        },
    }


def build_report(root: Path, evidence_path: Path | None = None) -> tuple[dict[str, Any], str]:
    """Read and validate a finished campaign; tests call this only on synthetic data."""
    root = root.resolve()
    evidence_path = evidence_path or root / "terminal_evidence.json"
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    evidence_path = evidence_path.resolve()
    terminal = _check_terminal_evidence(root, evidence_path)
    terminal_by_id = {entry["case_id"]: entry for entry in terminal}
    plan, plan_rows = _load_plan(root)
    actions: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    query_rows: list[dict[str, Any]] = []
    factorial_cells: dict[str, dict[tuple[int, int, int], float]] = {}
    reference_action_differences: list[dict[str, float | None]] = []
    phase1_count = mixed_count = factorial_count = reused_endpoints = 0
    local_values: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(dict)
    layer_values: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    trace_count = 0

    for case_id in range(22):
        plan_row = plan_rows[case_id]
        case_dir = root / "results" / f"case_{case_id:02d}"
        summary_path, query_path = case_dir / "case_summary.json", case_dir / "queries.jsonl"
        case_summary = _json(summary_path)
        context = f"case_{case_id:02d}"
        _require(case_summary.get("protocol") == PROTOCOL and case_summary.get("case_id") == case_id,
                 f"{context} case_summary identity/protocol mismatch")
        _require(case_summary.get("pbs_jobid") == terminal_by_id[case_id]["job_id"],
                 f"{context} case_summary pbs_jobid does not match terminal evidence")
        _require(case_summary.get("status") == "complete" and case_summary.get("query_count") == 20,
                 f"{context} case_summary is not complete with 20 queries")
        _require(case_summary.get("context_count") == 2 and case_summary.get("sampler_seeds") == plan_row["sampler_seeds"],
                 f"{context} case_summary does not match its two planned contexts")
        _require(case_summary.get("factorial_cell_count") == 16
                 and isinstance(case_summary.get("factorial_cells"), list)
                 and len(case_summary["factorial_cells"]) == 16,
                 f"{context} case_summary lacks 16 factorial entries")
        _require(case_summary.get("episodes") == 0 and case_summary.get("post_query_env_steps") == 0,
                 f"{context} case_summary claims episodes or post-query environment steps")
        for key in ("domain", "dimension", "task_id", "environment_seed", "original_task", "source_index"):
            _require(case_summary.get(key) == plan_row.get(key), f"{context} case_summary {key} mismatch")
        meta = plan_row["_metadata"]
        summary_input = case_summary.get("input", {})
        if isinstance(summary_input, dict) and "settling_steps" in summary_input:
            _require(summary_input["settling_steps"] == 30, f"{context} case_summary settling_steps mismatch")
        _require(meta.get("state_id") == 0 and meta.get("settling_steps") == 30,
                 f"{context} plan metadata does not establish state 0 / 30 settling steps")

        records = _read_queries(query_path)
        _require(len(records) == 20 and case_summary.get("query_count") == len(records),
                 f"{context} needs exactly 20 JSONL query records")
        _require([record.get("query_count") for record in records] == list(range(1, 21)),
                 f"{context} query_count sequence is incomplete or reordered")
        by_seed: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in plan_row["sampler_seeds"]}
        id_seen = set()
        context_cells_local: dict[int, dict[tuple[int, int, int], float]] = {
            seed: {} for seed in plan_row["sampler_seeds"]
        }
        for record in records:
            qtype = record.get("query_type")
            seed = record.get("sampler_seed")
            query_id = record.get("query_id")
            qctx = f"{context}/{query_id}"
            _require(type(seed) is int and seed in by_seed and isinstance(qtype, str),
                     f"{qctx} has an unexpected sampler seed/type")
            _require(query_id == f"{qtype}_seed{seed}" and query_id not in id_seen,
                     f"{qctx} query_id is inconsistent or duplicated")
            id_seen.add(query_id)
            _require(record.get("protocol") == PROTOCOL and record.get("case_id") == case_id
                     and record.get("context_id") == f"case_{case_id:02d}_seed{seed}",
                     f"{qctx} record identity mismatch")
            for key in ("domain", "dimension", "task_id", "environment_seed", "original_task", "source_index"):
                _require(record.get(key) == plan_row.get(key), f"{qctx} {key} mismatch")
            _require(record.get("episodes") == 0 and record.get("post_query_env_steps") == 0,
                     f"{qctx} claims counted episodes or post-query environment steps")
            _validate_scheduler_coverage(record, qctx)
            action = record.get("raw_normalized_action_32x7")
            # Validate every output, including BF16, before computing any contrasts.
            _action_metrics(action, action, qctx)
            counters = _delta_metrics(record, qctx)
            _require(counters["kv_packed_prefills"] == 0 and counters["kv_layer_reads"] == 0,
                     f"{qctx} unexpectedly used the KV4 path")
            bits_value = record.get("bits")
            if qtype == "bf16":
                _require(bits_value is None and all(counters[key] == 0 for key in COUNTERS),
                         f"{qctx} BF16 query used quantized counters")
            elif qtype == "independent_reference":
                _require(bits_value == [4, 4, 4] and all(counters[key] == 0 for key in COUNTERS),
                         f"{qctx} independent reference used quantized counters or wrong bits")
            else:
                _require(isinstance(bits_value, list) and len(bits_value) == 3
                         and all(type(bit) is int and bit in (4, 8) for bit in bits_value),
                         f"{qctx} quantized query has invalid activation bits")
                bits = tuple(bits_value)
                if qtype == "all_a4":
                    _require(bits == (4, 4, 4), f"{qctx} all_a4 label/bits mismatch")
                elif qtype == "all_a8":
                    _require(bits == (8, 8, 8), f"{qctx} all_a8 label/bits mismatch")
                else:
                    _require(qtype == f"mixed_{_bits_label(bits)}" and bits not in ((4, 4, 4), (8, 8, 8)),
                             f"{qctx} mixed query label/bits mismatch")
                _require(counters["integer_gemm_calls"] > 0,
                         f"{qctx} packed query did not execute integer GEMM")
                if bits == (8, 8, 8):
                    _require(counters["native_int4_gemm_calls"] == 0,
                             f"{qctx} all-A8 query executed native INT4")
                else:
                    _require(counters["native_int4_gemm_calls"] > 0,
                             f"{qctx} A4 query did not execute native INT4")
                context_cells_local[seed][bits] = 0.0
            trace_info = record.get("trace_coverage")
            if qtype == "all_a4":
                _process_a4_trace(root, case_dir, {**record, "case_id": case_id}, record,
                                  local_values, layer_values)
                trace_count += 1
            else:
                _require(trace_info is None, f"{qctx} unexpectedly contains an A4 trace")
            by_seed[seed][qtype] = record
            query_rows.append(record)

        for seed in plan_row["sampler_seeds"]:
            seed_records = by_seed[seed]
            _require(set(seed_records) == PHASE1_TYPES | {
                f"mixed_{_bits_label(bits)}" for bits in BITS if bits not in ((4, 4, 4), (8, 8, 8))
            }, f"{context}/seed{seed} does not contain the 10 expected unique queries")
            _require(len(seed_records) == 10, f"{context}/seed{seed} has duplicate or extra query types")
            phase1_count += len(PHASE1_TYPES)
            mixed_count += 6
            actions[case_id, seed] = seed_records
            for record in seed_records.values():
                if record["query_type"] == "bf16":
                    continue
                measured = _action_metrics(record["raw_normalized_action_32x7"],
                                           seed_records["bf16"]["raw_normalized_action_32x7"],
                                           f"{context}/{record['query_id']}")
                if record["query_type"] != "independent_reference":
                    bits = tuple(record["bits"])
                    context_cells_local[seed][bits] = measured["motor_rmse_first10"]
            native_a4 = seed_records["all_a4"]["raw_normalized_action_32x7"]
            independent = seed_records["independent_reference"]
            direct = _action_metrics(
                native_a4, independent["raw_normalized_action_32x7"],
                f"{context}/all_a4-vs-independent_reference/seed{seed}",
            )
            _verify_saved_action_metrics(
                independent.get("native_all_a4_vs_independent_reference"), direct,
                "independent_reference", f"{context}/all_a4-vs-independent_reference/seed{seed}",
            )
            reference_action_differences.append(direct)
            _require(set(context_cells_local[seed]) == set(BITS),
                     f"{context}/seed{seed} lacks one or more factorial bit triples")
            factorial_cells[f"case_{case_id:02d}_seed{seed}"] = context_cells_local[seed]

        factorial_entries = case_summary["factorial_cells"]
        seen_factorial = set()
        for cell in factorial_entries:
            _require(isinstance(cell, dict), f"{context} factorial cell is not an object")
            seed = cell.get("sampler_seed")
            bits_value = cell.get("bits")
            _require(type(seed) is int and seed in by_seed and isinstance(bits_value, list) and len(bits_value) == 3
                     and all(type(bit) is int and bit in (4, 8) for bit in bits_value),
                     f"{context} factorial cell has invalid seed/bits")
            bits = tuple(bits_value)
            _require(bits in BITS and (seed, bits) not in seen_factorial,
                     f"{context} has invalid or duplicate factorial cell")
            seen_factorial.add((seed, bits))
            expected = next(row for row in by_seed[seed].values()
                            if row.get("bits") == list(bits) and row.get("query_type") != "independent_reference")
            _require(cell.get("query_id") == expected["query_id"] and cell.get("reused_query") is True,
                     f"{context} factorial cell does not map to its executed query")
            factorial_count += 1
            if bits in ((4, 4, 4), (8, 8, 8)):
                reused_endpoints += 1
        _require(seen_factorial == {(seed, bits) for seed in by_seed for bits in BITS},
                 f"{context} factorial cells do not cover both seeds and all 8 bit triples")

    _require(len(query_rows) == 440 and phase1_count == 176 and factorial_count == 352
             and reused_endpoints == 88 and mixed_count == 264,
             "query/phase denominators differ from the authorized 176/352/88/264/440 plan")
    _require(trace_count == 44, f"expected 44 completed all-A4 traces, got {trace_count}")
    action_report = _action_report(plan_rows, actions)
    reference_action_report = _reference_action_report(reference_action_differences)
    factorial_report = _factorial_summary(factorial_cells)

    local_summary = []
    for (scope, stage, step), metrics in sorted(local_values.items()):
        local_summary.append({
            "scope": scope, "stage": stage, "step": step,
            "calls": len(next(iter(metrics.values()))) if metrics else 0,
            "metrics": {name: _summary(values) for name, values in sorted(metrics.items())},
        })
    worst = []
    for (module, scope, stage, step), values in layer_values.items():
        worst.append({
            "module": module, "scope": scope, "stage": stage, "step": step,
            "calls": len(values),
            "mean_native_vs_reference_normalized_rmse": sum(values) / len(values),
            "max_native_vs_reference_normalized_rmse": max(values),
        })
    worst.sort(key=lambda row: (row["mean_native_vs_reference_normalized_rmse"],
                                row["max_native_vs_reference_normalized_rmse"]), reverse=True)

    summary = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scope_limit": "Mechanism diagnostic only; no success-rate or formal-latency claim.",
        "completion_evidence": terminal,
        "denominators": {
            "cases": 22, "contexts": 44, "actual_full_queries": 440,
            "phase1_queries": 176, "factorial_cell_entries": 352,
            "phase2_reused_endpoints": 88, "phase2_new_mixed_queries": 264,
            "completed_case_summaries": 22, "complete_all_a4_traces": trace_count,
            "episodes": 0, "post_query_environment_steps": 0,
            "case_domain_counts": {"original": 10, "plus": 12},
            "plus_dimension_counts": {dimension: 2 for dimension in PLUS_DIMENSIONS},
        },
        "relative_action_rmse": action_report,
        "native_a4_vs_independent_reference_action": reference_action_report,
        "factorial_effects": factorial_report,
        "local_trace": {
            "trace_queries": trace_count,
            "grouping": "scope x actually observed denoising stage x scheduler step",
            "summary": local_summary,
            "worst_layers": worst[:20],
        },
    }
    markdown = _markdown(summary)
    return summary, markdown


def _markdown(summary: dict[str, Any]) -> str:
    den = summary["denominators"]
    lines = [
        "# Fast-WAM A4 Phases 1/2 机制诊断汇总", "",
        f"状态：`{summary['status']}`。本结果覆盖 {den['cases']} cases、{den['contexts']} sampler contexts、",
        f"{den['actual_full_queries']} full queries（Phase 1 {den['phase1_queries']}；factorial {den['factorial_cell_entries']}，",
        f"其中端点复用 {den['phase2_reused_endpoints']}、新增 mixed queries {den['phase2_new_mixed_queries']}）。",
        "episodes = 0，query 后环境步数 = 0。此诊断不提供成功率或 formal latency 结论。", "",
        "## 同 case BF16 action 对照", "",
        "每个 query 仅与相同 case 和 sampler seed 的 BF16 action 比较。Primary 为前 10 步 × 6 motor 维 RMSE；",
        "first10 gripper RMSE 单独报告，secondary global RMSE 覆盖 32×7。", "",
        "| Domain | Dimension | Query | Contexts | Primary motor RMSE | First10 gripper RMSE | Secondary global 32×7 RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in summary["relative_action_rmse"]["groups"]:
        for qtype, values in group["by_query_type"].items():
            primary = values["motor_rmse_first10_vs_bf16"]
            lines.append(
                f"| {group['domain']} | {group['dimension']} | {qtype} | {primary['n']} | "
                f"{_fmt(primary['mean'])} | "
                f"{_fmt(values['first10_gripper_rmse_vs_bf16']['mean'])} | "
                f"{_fmt(values['global_rmse_32x7_vs_bf16']['mean'])} |"
            )
    reference = summary["native_a4_vs_independent_reference_action"]["metrics"]
    lines.extend(["", "## Native all-A4 对 independent reference 的直接 action 差", "",
                  "逐 case/seed 直接比较两组 raw action；不通过它们各自相对 BF16 的 RMSE 推算。", "",
                  f"44 contexts：first10 motor RMSE mean {_fmt(reference['motor_rmse_first10']['mean'])}，"
                  f"first10 gripper RMSE mean {_fmt(reference['gripper_rmse_first10']['mean'])}，"
                  f"secondary global 32×7 RMSE mean {_fmt(reference['global_rmse_32x7']['mean'])}。", "",
                  "## A4 factorial 描述效应", "",
                  "Primary response 是相对同 context BF16 的前 10 步 × 6 motor 维绝对 RMSE；对每个 case/seed 完整配对 2³ cells。仅作描述统计，不做 p-values。", "",
                  "| Factor | Mean at A4 | Mean at A8 | A4 − A8 | Paired contexts |",
                  "|---|---:|---:|---:|---:|"])
    for name, effect in summary["factorial_effects"]["main_effects"].items():
        lines.append(f"| {name} | {_fmt(effect['a4_mean_motor_rmse_first10'])} | "
                     f"{_fmt(effect['a8_mean_motor_rmse_first10'])} | "
                     f"{_fmt(effect['a4_minus_a8_motor_rmse_first10'])} | "
                     f"{effect['paired_context_effect']['n']} |")
    lines.extend(["", "Interactions (paired within context):", ""])
    for name, effect in summary["factorial_effects"]["interactions"].items():
        payload = effect.get("paired_context_difference_in_differences",
                             effect.get("paired_context_third_order_difference", {}))
        lines.append(f"- `{name}`: mean contrast {_fmt(payload.get('mean'))}; median {_fmt(payload.get('median'))}; "
                     f"range [{_fmt(payload.get('min'))}, {_fmt(payload.get('max'))}].")
    lines.extend(["", "## Per-layer trace summaries", "",
                  "仅统计 44 个 native all-A4 query 实际调用的 Linear；不要求未调用模块有 coverage。指标在同一层调用级别累计。", "",
                  "| Scope | Stage | Step | Calls | A4 scale mean | A8 scale mean | A4 zero fraction | A4 input rel. RMSE | A4/A8 local RMSE | A4/A8 local NRMSE | Native/reference NRMSE |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for item in summary["local_trace"]["summary"]:
        metrics = item["metrics"]
        lines.append(
            f"| {item['scope']} | {item['stage']} | {item['step']} | {item['calls']} | "
            f"{_fmt(metrics['a4_scale_mean']['mean'])} | {_fmt(metrics['a8_scale_mean']['mean'])} | "
            f"{_fmt(metrics['a4_zero_code_fraction']['mean'])} | "
            f"{_fmt(metrics['a4_input_relative_rmse']['mean'])} | "
            f"{_fmt(metrics['a4_vs_a8_reference_rmse']['mean'])} | "
            f"{_fmt(metrics['a4_vs_a8_reference_normalized_rmse']['mean'])} | "
            f"{_fmt(metrics['native_vs_a4_reference_normalized_rmse']['mean'])} |"
        )
    lines.extend(["", "Worst observed module/stage/step local normalized errors:", ""])
    for item in summary["local_trace"]["worst_layers"][:10]:
        lines.append(f"- `{item['module']}` ({item['scope']}, {item['stage']} step {item['step']}): "
                     f"mean NRMSE {_fmt(item['mean_native_vs_reference_normalized_rmse'])}, "
                     f"max {_fmt(item['max_native_vs_reference_normalized_rmse'])}, calls {item['calls']}.")
    lines.extend(["", "所有 case 的 PBS exit status、artifact exit code、PIPELINE_COMPLETE marker、case summary、query 计数、", 
                  "counter 边界、actions 形状/有限性与 all-A4 实际 trace coverage 均通过聚合门禁。", ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6g}"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="diagnostic root containing plan/results/artifacts")
    parser.add_argument("--array-evidence", type=Path, help="default: ROOT/terminal_evidence.json")
    parser.add_argument("--out", type=Path, help="default: ROOT")
    args = parser.parse_args()
    allocation_guard()
    root = args.root.resolve()
    evidence = args.array_evidence or root / "terminal_evidence.json"
    if not evidence.is_absolute():
        evidence = root / evidence
    summary, markdown = build_report(root, evidence)
    out = (args.out or root).resolve()
    _write_atomic(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write_atomic(out / "analysis.zh.md", markdown)
    print(f"AGGREGATION_COMPLETE cases=22 contexts=44 queries=440 out={out}", flush=True)


if __name__ == "__main__":
    main()
