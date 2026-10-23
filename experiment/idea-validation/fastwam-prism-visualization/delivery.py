#!/usr/bin/env python3
"""Write a compact JSON-only delivery summary after the PBS renderer completes."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

MAX_OUTPUT_BYTES = 220 * 1024
PROTOCOL_NAME = "fastwam-prism-visualization-v1"


def require_compute_allocation() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("delivery summary requires an active PBS_JOBID")
    host = socket.gethostname().strip().lower()
    if not host or "login" in host:
        raise RuntimeError(f"refusing to summarize on login/unknown host {host!r}")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("delivery summary requires a valid PBS_NODEFILE")
    hosts = {line.strip().lower().split(".", 1)[0]
             for line in Path(nodefile).read_text(encoding="utf-8").splitlines() if line.strip()}
    if host.split(".", 1)[0] not in hosts:
        raise RuntimeError(f"PBS_NODEFILE does not contain actual host {host!r}")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def as_int(value, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be an integer") from exc
    check(not isinstance(value, bool) and (not isinstance(value, float) or result == value),
          f"{label} must be an integer")
    return result


def record_kind(row: dict) -> str:
    kind = str(row.get("kind", "")).lower()
    if "weight" in kind:
        return "weight"
    if "paired" in kind:
        return "paired"
    if "activation" in kind:
        return "actual"
    raise RuntimeError(f"unrecognized record kind: {row.get('record_id')!r}")


def unique_id_map(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        record_id = row.get("record_id")
        check(isinstance(record_id, str) and record_id and record_id not in result,
              f"{label} has an empty or duplicate record_id {record_id!r}")
        result[record_id] = row
    return result


def axis_cv(profile, label: str) -> float | None:
    check(isinstance(profile, list) and profile, f"metrics lack {label}")
    values = [float(value) for value in profile]
    check(all(math.isfinite(value) for value in values), f"non-finite values in {label}")
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean


def compact_record(row: dict, metric: dict, kind: str) -> dict:
    names = {"actual": ("pre_quant", "post_quant", "output"),
             "paired": ("paired_raw", "paired_learned_spinquant", "paired_smooth05_hadamard"),
             "weight": ("weight", "reference")}[kind]
    profiles = (("per_output_channel_rms", "per_input_channel_rms") if kind == "weight"
                else ("per_token_rms", "per_channel_rms"))
    source = metric.get("arrays", {})
    arrays = {}
    for name in names:
        item = source.get(name)
        check(isinstance(item, dict), f"metrics omit {name} for {row['record_id']!r}")
        maximum, rms = float(item["max_abs"]), float(item["rms"])
        check(math.isfinite(maximum) and math.isfinite(rms), f"non-finite metrics for {row['record_id']!r}")
        arrays[name] = {"shape": item.get("shape"), "window_shape": item.get("window_shape"),
            "window_start": item.get("window_start"), "max_abs": maximum, "rms": rms,
            "max_abs_to_rms": maximum / rms if rms else None,
            "axis_rms_cv": {"row_axis": axis_cv(item.get(profiles[0]), profiles[0]),
                            "column_axis": axis_cv(item.get(profiles[1]), profiles[1])}}
    nmse_names = {"actual": ("pre_quant_to_post_quant",),
                  "paired": ("paired_raw_to_qdq", "paired_learned_spinquant_to_qdq", "paired_smooth05_hadamard_to_qdq"),
                  "weight": ("weight_to_reference",)}[kind]
    nmse = metric.get("nmse", {})
    check(all(name in nmse for name in nmse_names), f"metrics omit NMSE for {row['record_id']!r}")
    identity = {name: row.get(name) for name in (
        "record_id", "kind", "case_id", "seed_index", "arm", "transform", "stream", "module",
        "block_index", "stage", "step", "call_index") if name in row}
    identity.update(category=kind, shape=row.get("weight_shape", row.get("shape")))
    return {**identity, "arrays": arrays, "nmse": {name: nmse[name] for name in nmse_names}}


def primary_rows(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    selected = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        group = groups[key]
        low = min(as_int(row.get("call_index"), "call_index") for row in group)
        matches = [row for row in group if as_int(row.get("call_index"), "call_index") == low]
        check(len(matches) == 1, f"tied lowest call_index for {key!r}")
        selected.append(matches[0])
    return selected


def build_summary(out: Path) -> dict:
    protocol = read_json(out / "protocol.json")
    capture = read_json(out / "summary.json")
    raw = read_json(out / "raw" / "manifest.json")
    plots = read_json(out / "plots" / "manifest.json")
    metrics = read_json(out / "plots" / "metrics.json")
    check(protocol.get("protocol") == PROTOCOL_NAME, "unexpected capture protocol")
    check(capture.get("protocol") == protocol["protocol"] and capture.get("status") == "complete" and
          as_int(capture.get("query_count"), "query_count") == 10 and
          as_int(capture.get("weight_records"), "weight_records") == 80,
          "capture summary protocol/status/query/weight gate failed")
    check(raw.get("protocol") == protocol["protocol"] and raw.get("status") == "complete" and
          as_int(raw.get("query_count"), "raw query_count") == 10,
          "raw capture manifest is incomplete")
    check(plots.get("protocol") == protocol["protocol"] and plots.get("protocol_snapshot") == protocol and
          plots.get("status") == "complete" and plots.get("bundle_count") == 12 and
          plots.get("expected_bundle_count") == 12 and plots.get("metrics_ready") is True,
          "plot manifest protocol/status/bundle/metrics gate failed")

    arms, cases = protocol["arms"], [as_int(v, "case") for v in protocol["cases"]]
    seed = as_int(protocol["seed_index"], "seed_index")
    expected_queries = set(product(cases, arms, [seed]))
    query_keys, query_ids, queries_out = set(), set(), []
    native_calls = inference_calls = 0
    queries = raw.get("query_records", [])
    for row in queries:
        key = (as_int(row.get("case_id"), "query case_id"), row.get("arm"), as_int(row.get("seed_index"), "query seed_index"))
        query_id = row.get("query_id")
        check(key in expected_queries and key not in query_keys and isinstance(query_id, str) and query_id not in query_ids,
              f"unexpected or duplicate query tuple/id {key!r}")
        query_keys.add(key)
        query_ids.add(query_id)
        native_calls += as_int(row.get("native_int4_calls"), "native_int4_calls")
        inference_calls += as_int(row.get("actual_inference_calls_this_query"), "actual_inference_calls_this_query")
        queries_out.append({name: row.get(name) for name in (
            "query_id", "case_id", "seed_index", "arm", "native_int4_calls",
            "actual_inference_calls_this_query", "reference_source", "reference_comparison",
            "collector_hooks_observer_only")})
    check(query_keys == expected_queries and len(queries) == 10, "query records do not exactly cover protocol tuples")
    check(inference_calls == as_int(raw.get("runtime_inference_calls"), "runtime_inference_calls") ==
          as_int(capture.get("runtime_inference_calls"), "summary runtime_inference_calls"),
          "runtime inference call totals differ")

    records, metric_rows = raw.get("records", []), metrics.get("records", [])
    check(len(records) == 752 and len(metric_rows) == 752, "expected 752 capture records and metric summaries")
    record_map, metric_map = unique_id_map(records, "raw records"), unique_id_map(metric_rows, "metrics")
    check(record_map.keys() == metric_map.keys(), "manifest and metrics record IDs differ")
    by_kind = {kind: [row for row in records if record_kind(row) == kind]
               for kind in ("actual", "paired", "weight")}
    check(len(by_kind["weight"]) == 80, "manifest does not contain 80 weight records")

    transforms = raw.get("transform_records", [])
    transform_counts = Counter(row.get("metadata", {}).get("basis") for row in transforms)
    main_case = as_int(protocol.get("main_case", 1004), "main_case")
    main_step = as_int(protocol.get("main_step", 4), "main_step")
    selected_actual = primary_rows([row for row in by_kind["actual"]
        if row.get("case_id") == main_case and row.get("step") == main_step],
        ("arm", "stream", "module", "block_index", "stage", "step"))
    selected_paired = primary_rows([row for row in by_kind["paired"]
        if row.get("case_id") == main_case and row.get("step") == main_step],
        ("stream", "module", "block_index", "stage", "step"))
    check(len(selected_actual) == 80 and len(selected_paired) == 16,
          "main case/step does not have the expected actual and paired primary records")
    actual_summaries = [compact_record(row, metric_map[row["record_id"]], "actual") for row in selected_actual]
    paired_summaries = [compact_record(row, metric_map[row["record_id"]], "paired") for row in selected_paired]
    weight_summaries = [compact_record(row, metric_map[row["record_id"]], "weight") for row in by_kind["weight"]]

    streams, suffixes = protocol["streams"], protocol["module_suffixes"]
    blocks, steps = protocol["blocks"], protocol["snapshot_steps"]
    stage_for = {"video_expert": "video", "action_expert": "action"}
    snapshot_steps = {as_int(step, "snapshot step") for step in steps}

    def is_scheduler_snapshot(row: dict) -> bool:
        return (row.get("stage") == stage_for.get(row.get("stream")) and
                as_int(row.get("step"), "record step") in snapshot_steps)

    target_slots = {(row.get("stream"), row.get("module"), row.get("block_index")) for row in by_kind["actual"]}
    scheduler_actual = [row for row in by_kind["actual"] if is_scheduler_snapshot(row)]
    scheduler_paired = [row for row in by_kind["paired"] if is_scheduler_snapshot(row)]
    additional_actual = [row for row in by_kind["actual"] if not is_scheduler_snapshot(row)]
    additional_paired = [row for row in by_kind["paired"] if not is_scheduler_snapshot(row)]
    actual_groups = {(row.get("case_id"), row.get("seed_index"), row.get("arm"), row.get("stream"),
                      row.get("module"), row.get("block_index"), row.get("stage"), row.get("step"))
                     for row in scheduler_actual}
    paired_groups = {(row.get("case_id"), row.get("seed_index"), row.get("stream"), row.get("module"),
                      row.get("block_index"), row.get("stage"), row.get("step")) for row in scheduler_paired}
    weight_groups = {(row.get("arm"), row.get("stream"), row.get("module"), row.get("block_index"))
                     for row in by_kind["weight"]}
    expected_slot_count = len(streams) * len(suffixes) * len(blocks)
    expected_actual = len(cases) * len(arms) * expected_slot_count * len(steps)
    expected_paired = len(cases) * expected_slot_count * len(steps)
    check(len(target_slots) == expected_slot_count and len(actual_groups) == expected_actual and
          len(paired_groups) == expected_paired and len(weight_groups) == 80,
          "observed module/stage/step coverage counts differ from the frozen protocol")
    additional_by_stage_step = Counter(f"{row.get('stage')}:{row.get('step')}" for row in additional_actual + additional_paired)
    source_ids = raw.get("source_identities", {})
    calibration_refs = {name: value["calibration_reference"]
        for name, value in (("protocol", protocol), ("manifest", raw), ("source_identities", source_ids))
        if isinstance(value, dict) and value.get("calibration_reference") is not None}
    absent_parity = sum(row.get("reference_comparison", {}).get("status") == "parity_reference_absent"
                        for row in queries_out)
    metric_window = metrics.get("metric_window", {})
    return {
        "schema": "fastwam-prism-delivery-summary-v1", "status": "complete", "protocol": protocol["protocol"],
        "result_root_relative_to_remote_experiment_root": "results/full",
        "result_paths_relative_to_remote_experiment_root": {"capture_summary": "results/full/summary.json",
            "raw_manifest": "results/full/raw/manifest.json", "plot_manifest": "results/full/plots/manifest.json",
            "complete_metrics": "results/full/plots/metrics.json", "delivery_summary": "results/full/plots/delivery_summary.json"},
        "capture": {"status": capture["status"], "query_count": len(queries),
            "runtime_inference_calls": inference_calls, "native_int4_calls_total": native_calls,
            "records": len(records), "actual_records": len(by_kind["actual"]),
            "paired_local_records": len(by_kind["paired"]), "weight_records": len(by_kind["weight"])},
        "plots": {"status": plots["status"], "bundle_count": plots["bundle_count"],
            "expected_bundle_count": plots["expected_bundle_count"], "metrics_ready": plots["metrics_ready"],
            "metric_record_count": len(metric_rows), "metric_window": metric_window,
        "npz_validation_provenance": "metrics.json is the cached_metrics output from plot.py's first calculate_all_metrics pass, which validated NPZ arrays, metadata and shapes; this summary reads JSON only.",
            "record_file_paths": "Each summarized record_id resolves to its full relative file path in raw/manifest.json."},
        "coverage_counts": {"query_tuples": {"observed": len(query_keys), "expected": 10},
            "record_ids": {"observed": len(record_map), "metrics": len(metric_map), "duplicates": 0, "match": True},
            "target_module_slots": {"observed": len(target_slots), "expected": expected_slot_count},
            "transform_records_by_basis": dict(transform_counts),
            "actual_groups": {"observed": len(actual_groups), "expected": expected_actual},
            "paired_local_groups": {"observed": len(paired_groups), "expected": expected_paired},
            "weight_groups": {"observed": len(weight_groups), "expected": 80},
            "additional_saved_records": {"actual": len(additional_actual), "paired_local": len(additional_paired),
                "by_stage_step": dict(additional_by_stage_step)}},
        "query_records": sorted(queries_out, key=lambda row: (arms.index(row["arm"]), cases.index(row["case_id"]))),
        "main_activation_records": {"case_id": main_case, "step": main_step,
            "call_rule": "lowest call_index per group", "actual": actual_summaries, "paired_local": paired_summaries},
        "all_weight_records": weight_summaries,
        "shape_and_window": {"capture_tokens_max": protocol.get("capture_tokens", 128),
            "input_channels": "Activation and paired-local input arrays retain the full selected Linear input width; actual output arrays retain up to plot_channels.",
            "weight_shape": "Full dimensions are in weight_shape; stored weight/reference arrays use up to weight_output_count × plot_channels.",
            "plot_window": metric_window,
            "statistics_scope": "All max_abs, RMS, axis RMS CV and NMSE values are scoped to the plot window. A 512-channel slice does not preserve full-channel token CV; orthogonal rotations preserve total energy only across the complete channel vector."},
        "source_and_parity_boundary": {"source_identities": source_ids,
            "paired_local": "Three transforms share the same captured BF16 Linear input; their per-row A4 QDQ values are visualization candidates, not runtime model execution.",
            "actual_activation": "Each arm uses its own model execution, so differences from paired-local include upstream arm effects.",
            "native_int4_arms": ["spinquant_native_w4a4", "smooth_scalar_native_w4a4"],
            "decoded_bf16_vq_arms": ["row_scale_vq_a4", "micro_scale_vq_a4"],
            "parity_reference_absent_queries": absent_parity,
            "calibration_reference": {"status": "present" if calibration_refs else "missing", "references": calibration_refs,
                "detail": None if calibration_refs else "No calibration reference is recorded in the capture protocol, manifest or source identities."},
            "accuracy_claim": None, "latency_claim": None},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="completed capture result root")
    args = parser.parse_args()
    require_compute_allocation()
    report = build_summary(args.out.resolve())
    payload = (json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    check(len(payload) <= MAX_OUTPUT_BYTES,
          f"delivery summary is {len(payload)} bytes; limit is {MAX_OUTPUT_BYTES}; refusing to truncate")
    target = args.out.resolve() / "plots" / "delivery_summary.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    print(f"Wrote {target} ({len(payload)} bytes); 10 queries, 752 records, 12 plot bundles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
