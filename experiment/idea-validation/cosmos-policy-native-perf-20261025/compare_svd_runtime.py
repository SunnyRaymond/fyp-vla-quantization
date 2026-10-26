"""Same-GPU engineering comparison of SVDQuant current and fused candidates."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from datetime import datetime, timezone

import perf_native as perf


ARM = "svdquant_w4a4"
VARIANTS = ("current", "native_fused")
DIAGNOSTIC_KEYS = ((3000, 0), (3009, 1), (3020, 0), (3039, 1))
REPEATS = 3
WARMUP_KEYS = DIAGNOSTIC_KEYS[:3]
SCOPE_STAGES = (("sampler", VARIANTS), ("dit_replay", VARIANTS))
FORMAT = "cosmos-svd-native-fusion-diagnostic-v1"


def expected_measurement_ids():
    return tuple((key, repeat) for repeat in range(REPEATS) for key in DIAGNOSTIC_KEYS)


def validate_native_panel(old_protocol, native_protocol):
    validated = perf.validate_protocol(old_protocol, ARM)
    panel = native_protocol.get("panel", {})
    if (native_protocol.get("source_protocol") != old_protocol.get("protocol")
            or panel.get("case_ids_start") != 3000 or panel.get("case_ids_end_inclusive") != 3039
            or panel.get("seed_indices") != [0, 1] or panel.get("queries") != 80
            or panel.get("blocks") != 3 or panel.get("repetitions") != 3
            or panel.get("samples_per_arm_per_scope") != 720
            or panel.get("batch_size") != 1 or panel.get("nfe") != 5):
        raise ValueError("Native protocol is not the frozen batch-1 80-query x 3 x 3 x 5 panel")
    return validated


def validate_measurement_records(records):
    expected = set(expected_measurement_ids())
    observed = []
    for row in records:
        key = tuple(row.get("query", ()))
        repeat = row.get("repeat")
        if len(key) != 2 or not all(isinstance(value, int) for value in key):
            raise ValueError("Diagnostic sample query must be a (case_id, seed_index) pair")
        if isinstance(repeat, bool) or not isinstance(repeat, int):
            raise ValueError("Diagnostic sample repeat must be an integer")
        observed.append((key, repeat))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("A diagnostic variant must contain exactly four fixed keys x three repeats")
    return len(expected)


def summarize_variant_records(scope, records):
    count = validate_measurement_records(records)
    latency_field = "denoising_seconds" if scope == "sampler" else "dit_replay_seconds"
    latency = [float(row[latency_field]) for row in records]
    peaks = [int(row["peak_allocated_bytes"]) for row in records]
    if any(not math.isfinite(value) or value <= 0 for value in latency) or any(value <= 0 for value in peaks):
        raise ValueError("Diagnostic latency and peak-allocation samples must be finite and positive")
    result = {
        "measurement_count": count,
        "latency_metric": latency_field,
        "latency_seconds": perf.summarize(latency),
        "peak_allocated_bytes": perf.summarize(peaks),
        "samples": records,
    }
    if scope == "sampler":
        result["complete_endpoint_query_seconds"] = perf.summarize(
            [float(row["complete_endpoint_query_seconds"]) for row in records])
    return result


def mean_speedup(current, native_fused):
    current_mean = float(current["latency_seconds"]["mean"])
    fused_mean = float(native_fused["latency_seconds"]["mean"])
    if not math.isfinite(current_mean) or not math.isfinite(fused_mean) or fused_mean <= 0:
        raise ValueError("Cannot compute speedup from invalid latency means")
    return current_mean / fused_mean


def replay_input_calls(capture):
    """Keep only inputs; captured reference outputs never enter timed replay logic."""
    calls = capture.get("calls")
    if not isinstance(calls, list) or len(calls) != 5:
        raise ValueError("Fixed DiT replay requires five captured NFE calls")
    result = []
    for nfe, call in enumerate(calls):
        if not isinstance(call, dict) or "args" not in call or "kwargs" not in call:
            raise ValueError(f"Captured NFE {nfe} is missing forward inputs")
        result.append({"args": call["args"], "kwargs": call["kwargs"]})
    return result


def require_variant_gates(worker, root, variant, binding):
    worker._require_gate(root, ARM, variant, "sampler")
    worker._require_gate(root, ARM, variant, "dit_replay")
    check = getattr(worker, "assert_native_gate_binding", None)
    if not callable(check):
        raise RuntimeError("perf_native.assert_native_gate_binding is required for this diagnostic")
    check(root, ARM, variant, binding)


def _check_hardware_uuid(worker, torch, device, expected):
    current = worker._gpu_receipt(torch, device)
    if current.get("gpu_uuid") != expected.get("gpu_uuid"):
        raise RuntimeError("GPU UUID changed during the current-vs-fused diagnostic")
    return current


def _install_variant(worker, runtime, protocol, old_root, root, out, variant):
    metadata, restore = worker._install_native(
        runtime, protocol, old_root, out, ARM, variant, {}, native_root=root)
    if not isinstance(metadata, dict) or not callable(restore):
        raise RuntimeError(f"SVD {variant} install must return metadata and a restore hook")
    try:
        binding = worker._binding(metadata, ARM, variant, root)
        require_variant_gates(worker, root, variant, binding)
        return metadata, binding, restore
    except Exception:
        worker._restore_to_bf16(runtime, protocol, old_root, restore)
        raise


def _measure_sampler_variant(worker, common, runtime, protocol, old_root, root, out,
                             variant, torch, device, hardware):
    import benchmark as legacy_benchmark

    contexts = {key: worker._load_context(common, old_root, *key) for key in DIAGNOSTIC_KEYS}
    metadata, binding, restore = _install_variant(
        worker, runtime, protocol, old_root, root, out, variant)
    records = []
    warmup_rows = []
    try:
        _check_hardware_uuid(worker, torch, device, hardware)
        for key in WARMUP_KEYS:
            row = legacy_benchmark.run_one(common, runtime, contexts[key],
                                           torch=torch, device=device)
            warmup_rows.append({"query": list(key), **row})
        for repeat in range(REPEATS):
            for key in DIAGNOSTIC_KEYS:
                torch.cuda.reset_peak_memory_stats(device)
                measurement = legacy_benchmark.run_one(
                    common, runtime, contexts[key], torch=torch, device=device)
                peak = int(torch.cuda.max_memory_allocated(device))
                if peak <= 0 or measurement.get("sampler_evaluations") != 5:
                    raise RuntimeError("Sampler diagnostic did not complete five NFE with a valid GPU peak")
                records.append({
                    "query": list(key), "repeat": repeat,
                    "denoising_seconds": float(measurement["denoising_seconds"]),
                    "complete_endpoint_query_seconds": float(measurement["complete_endpoint_query_seconds"]),
                    "sampler_evaluations": int(measurement["sampler_evaluations"]),
                    "denoiser_calls": int(measurement["denoiser_calls"]),
                    "peak_allocated_bytes": peak,
                    "gpu_uuid": hardware["gpu_uuid"], "hostname": socket.gethostname(),
                    "pbs_job_id": os.environ.get("PBS_JOBID"),
                    "backend_binding": binding,
                    "memory_scope": "full common.endpoint including resident non-DiT modules and endpoint temporaries",
                })
        summary = summarize_variant_records("sampler", records)
        summary.update({"variant": variant, "backend_binding": binding,
                        "install_metadata": metadata,
                        "warmup": {"query_count": len(warmup_rows),
                                   "queries": warmup_rows,
                                   "measurements_excluded": True},
                        "timing_scope": "synchronized legacy benchmark.run_one sampler.forward wall interval"})
        return summary
    finally:
        worker._restore_to_bf16(runtime, protocol, old_root, restore)
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _measure_dit_variant(worker, runtime, protocol, old_root, root, out, variant,
                         torch, device, hardware, offload):
    metadata, binding, restore = _install_variant(
        worker, runtime, protocol, old_root, root, out, variant)
    bundles = None
    capture_template = None
    captures = {}
    warmup_rows = []
    records = []
    dispatch_name = None
    try:
        _check_hardware_uuid(worker, torch, device, hardware)
        capture_manifest = worker._read_capture_manifest(root, protocol, hardware)
        if capture_manifest.get("capture_scope") != "actual common.endpoint five-NFE model.net.forward arguments and outputs":
            raise RuntimeError("DiT replay inputs are not the frozen actual-endpoint FP captures")
        capture_template = worker._load_capture(root, *DIAGNOSTIC_KEYS[0], torch)
        template_calls = replay_input_calls(capture_template)
        del capture_template
        capture_template = None
        with worker._dit_replay_dispatch(runtime) as dispatch_name:
            bundles = worker._make_graph_bundle(
                runtime.model.net, template_calls, device, torch)
        del template_calls
        if len(bundles) != 5 or [row["nfe"] for row in bundles] != list(range(5)):
            raise RuntimeError("Generic DiT replay must capture five ordered NFE graphs")
        if len({id(row["graph"]) for row in bundles}) != 5:
            raise RuntimeError("Each NFE requires its own graph object")
        if any("output" not in row for row in bundles):
            raise RuntimeError("Every NFE graph must retain its own output buffer")

        for key in WARMUP_KEYS:
            capture = worker._load_capture(root, *key, torch)
            calls = replay_input_calls(capture)
            del capture
            captures[key] = calls
            seconds, peak = worker._replay_five(bundles, calls, device, torch)
            warmup_rows.append({"query": list(key), "dit_replay_seconds": seconds,
                                "peak_allocated_bytes": peak})
        for repeat in range(REPEATS):
            for key in DIAGNOSTIC_KEYS:
                calls = captures.get(key)
                if calls is None:
                    capture = worker._load_capture(root, *key, torch)
                    calls = replay_input_calls(capture)
                    del capture
                    captures[key] = calls
                seconds, peak = worker._replay_five(bundles, calls, device, torch)
                if not math.isfinite(seconds) or seconds <= 0 or peak <= 0:
                    raise RuntimeError("Generic five-NFE DiT replay produced invalid timing or peak memory")
                records.append({
                    "query": list(key), "repeat": repeat,
                    "dit_replay_seconds": float(seconds),
                    "peak_allocated_bytes": int(peak),
                    "gpu_uuid": hardware["gpu_uuid"], "hostname": socket.gethostname(),
                    "pbs_job_id": os.environ.get("PBS_JOBID"),
                    "backend_binding": binding,
                    "memory_scope": "DiT resident parameters + fixed replay inputs and outputs + shared-pool graph activations/temporaries",
                })
        summary = summarize_variant_records("dit_replay", records)
        summary.update({"variant": variant, "backend_binding": binding,
                        "install_metadata": metadata,
                        "warmup": {"query_count": len(warmup_rows),
                                   "queries": warmup_rows,
                                   "graph_capture_warmups_per_nfe": 3,
                                   "graph_capture_warmup_forward_count": 15,
                                   "measurements_excluded": True},
                        "graph_capture_count": len(bundles),
                        "graph_backend": bundles[0]["graph_backend"],
                        "graph_pool_contract": worker.DIT_GRAPH_POOL_CONTRACT,
                        "dit_replay_dispatch": dispatch_name,
                        "reference_outputs_used_for_timing": False,
                        "non_dit_offload": offload,
                        "timing_scope": "CUDA events around five sequential generic CUDA-graph replays; condition and sampler preparation excluded"})
        return summary
    finally:
        # Destroy graph executables before restoring the BF16 target weights they captured.
        if bundles is not None:
            del bundles
        if capture_template is not None:
            del capture_template
        captures.clear()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        worker._restore_to_bf16(runtime, protocol, old_root, restore)
        gc.collect()
        torch.cuda.empty_cache()


def _scope_comparison(scope, variants):
    current = variants["current"]
    fused = variants["native_fused"]
    return {
        "scope": scope,
        "diagnostic_measurements_per_variant": 12,
        "variants": variants,
        "native_fused_mean_speedup_vs_current": mean_speedup(current, fused),
        "speedup_basis": "current mean latency / native_fused mean latency",
        "formal_variant_selection_changed": False,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    old_root, root, out = perf.validate_paths(args.old_root, args.root, args.out)
    protocol = perf.read_json(old_root / "protocol.json")
    native_protocol_path = root / "native_protocol.json"
    if not native_protocol_path.is_file():
        raise FileNotFoundError(f"Frozen native protocol is missing: {native_protocol_path}")
    native_protocol = perf.read_json(native_protocol_path)
    protocol_contract = validate_native_panel(protocol, native_protocol)
    code_root = perf._prepare_code_imports(old_root, None)
    import common
    perf._allocation_guard(common)
    for variant in VARIANTS:
        for scope in ("sampler", "dit_replay"):
            perf._require_gate(root, ARM, variant, scope)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    runtime = None
    report = {
        "format": FORMAT, "status": "running", "diagnostic_only": True,
        "protocol": protocol["protocol"], "arm": ARM,
        "variants": list(VARIANTS), "scope_order": [row[0] for row in SCOPE_STAGES],
        "diagnostic_keys": [list(key) for key in DIAGNOSTIC_KEYS],
        "warmup_keys": [list(key) for key in WARMUP_KEYS],
        "warmup_queries_per_variant_per_scope": len(WARMUP_KEYS),
        "measurements_per_variant_per_scope": len(expected_measurement_ids()),
        "formal_protocol": {"frozen_queries": protocol_contract["queries"],
                            "blocks": protocol_contract["blocks"],
                            "repetitions": protocol_contract["repetitions"],
                            "samples_per_arm": protocol_contract["samples_per_arm"]},
        "training_updates": 0, "closed_loop_episodes": 0,
        "selection_policy": "engineering diagnostic only; does not change the formal selected SVD variant",
        "scopes": {}, "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = root / "results" / "svd_fusion_diagnostic" / "summary.json"
    try:
        runtime = common.prepare_runtime(protocol, old_root, out)
        device = next(runtime.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Same-GPU diagnostic requires CUDA; runtime is on {device}")
        hardware = perf._gpu_receipt(torch, device)
        report["hardware"] = hardware
        report["same_gpu_contract"] = "all four variant/scope stages run in this process on the same GPU UUID"
        sampler_variants = {}
        for variant in VARIANTS:
            sampler_variants[variant] = _measure_sampler_variant(
                perf, common, runtime, protocol, old_root, root, out,
                variant, torch, device, hardware)
        report["scopes"]["sampler"] = _scope_comparison("sampler", sampler_variants)

        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        offload = perf._offload_non_dit(runtime, torch, device)
        gc.collect()
        torch.cuda.empty_cache()
        dit_variants = {}
        for variant in VARIANTS:
            dit_variants[variant] = _measure_dit_variant(
                perf, runtime, protocol, old_root, root, out,
                variant, torch, device, hardware, offload)
        report["scopes"]["dit_replay"] = _scope_comparison("dit_replay", dit_variants)
        report["status"] = "complete"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["summary_bytes"] = perf._bounded_summary(summary_path, report)
        print(json.dumps({"status": report["status"], "summary_path": str(summary_path),
                          "same_gpu_uuid": hardware["gpu_uuid"],
                          "sampler_speedup": report["scopes"]["sampler"]["native_fused_mean_speedup_vs_current"],
                          "dit_replay_speedup": report["scopes"]["dit_replay"]["native_fused_mean_speedup_vs_current"]},
                         ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        if root.exists():
            report["summary_bytes"] = perf._bounded_summary(summary_path, report)
        raise
    finally:
        if runtime is not None:
            for name in ("restore_native_diffusion", "restore_native_rotation", "_cosmos_native_restore"):
                restore = getattr(runtime, name, None)
                if callable(restore):
                    restore()
            import runtime as legacy_runtime
            legacy_runtime.install_arm(runtime, protocol, old_root, "bf16", optimized=False)
        _ = code_root


if __name__ == "__main__":
    raise SystemExit(main())
