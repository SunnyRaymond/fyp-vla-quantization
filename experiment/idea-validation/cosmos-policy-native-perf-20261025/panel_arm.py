"""Run one frozen native arm's sampler and DiT replay panels in one GPU process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import panel_native
import perf_native
from startup_trace import startup_trace


DRIVER_FORMAT = "cosmos_native_single_arm_panel_v1"
SCOPES = ("sampler", "dit_replay")
SOFT_DEADLINE_SECONDS = int(os.environ.get("COSMOS_SOFT_DEADLINE_SECONDS", "6600"))
ROTATION_BACKEND_ARMS = frozenset(("spinquant_w4a4", "quarot_w4a4", "viditq_w4a4"))
LEGACY_SCOPE_SAMPLES = 720
LEGACY_DESIGN = "legacy_80q_3x3"


def validate_arm_request(native_protocol: dict, arm: str, variant: str,
                         root: Path, gate_checker=None) -> dict:
    """Allow a frozen arm to run while other arms' global selection is pending."""
    if native_protocol.get("arms") != list(perf_native.ARMS):
        raise RuntimeError("native_protocol arm list differs from the frozen six-arm panel")
    if arm not in perf_native.ARMS:
        raise ValueError(f"Unknown frozen panel arm: {arm}")
    perf_native.validate_variant(arm, variant)
    if arm == "bf16":
        return {"gate_prechecked": False, "fidelity_policy": "not_applicable",
                "fidelity_failure_blocks_panel": False,
                "fidelity_policy_note": "BF16 is the reference arm; no native fidelity gate applies.",
                "selection_status": native_protocol.get("selection_status")}

    selected = native_protocol.get("selected_variants")
    if not isinstance(selected, dict) or selected.get(arm) != variant:
        raise RuntimeError(f"{arm} / {variant} is not the selected variant in native_protocol.json")
    policy_helper = getattr(perf_native, "baseline_fidelity_is_report_only", None)
    if not callable(policy_helper):
        raise RuntimeError("perf_native.baseline_fidelity_is_report_only is required")
    report_only = bool(policy_helper(native_protocol, arm))
    checker = gate_checker or perf_native._require_gate
    # The shared checker enforces policy, selected backend, and build bindings.
    checker(root, arm, variant, "sampler")
    policy = "report_only" if report_only else "strict"
    note = ("Numeric native-vs-decoded fidelity is observational for this baseline; any existing gate failure does not block the performance panel."
            if report_only else
            "Strict numerical fidelity preflight is required; both full80 sampler and DiT gates must pass.")
    return {"gate_prechecked": not report_only,
            "fidelity_policy": policy,
            "fidelity_failure_blocks_panel": not report_only,
            "fidelity_policy_note": note,
            "selection_status": native_protocol.get("selection_status")}


def scope_coverage_is_full(summary: dict) -> bool:
    coverage = summary.get("coverage", {})
    return (summary.get("performance_design") == perf_native.PERFORMANCE_DESIGN
            and summary.get("complete") is True
            and coverage.get("queries") == perf_native.PERFORMANCE_SAMPLES
            and coverage.get("samples") == perf_native.PERFORMANCE_SAMPLES
            and coverage.get("expected_samples") == perf_native.PERFORMANCE_SAMPLES
            and coverage.get("blocks") == perf_native.PERFORMANCE_BLOCKS
            and coverage.get("repetitions") == perf_native.PERFORMANCE_REPETITIONS)


def _legacy_summary_path(root: Path, scope: str, arm: str, variant: str) -> Path:
    return root / "results" / "native_performance" / scope / variant / arm / "summary.json"


def _normalized_binding(binding: dict) -> dict:
    return {key: value for key, value in binding.items() if key != "dit_replay_dispatch"}


def _finite_tree(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _legacy_warmup_keys(summary: dict, scope: str) -> list[tuple[int, int]]:
    rows = summary.get("warmup_queries")
    if not isinstance(rows, list):
        return []
    try:
        if scope == "sampler":
            return [tuple(int(v) for v in row["query"]) for row in rows]
        return [tuple(int(v) for v in row) for row in rows]
    except (KeyError, TypeError, ValueError):
        return []


def _positive_stat(summary: dict, key: str) -> bool:
    stats = summary.get(key)
    return (isinstance(stats, dict) and stats.get("count") == LEGACY_SCOPE_SAMPLES
            and all(isinstance(stats.get(field), (int, float))
                    and math.isfinite(stats[field]) and stats[field] > 0
                    for field in ("mean", "median", "p95", "max")))


def _check_gate_binding(root: Path, native_protocol: dict, protocol_id: str, arm: str, variant: str,
                        binding: dict, scope: str) -> None:
    if arm == "bf16":
        return
    perf_native.assert_native_gate_binding(root, arm, variant, binding)
    if perf_native.baseline_fidelity_is_report_only(native_protocol, arm):
        return
    gate_path = (root / "results" / "native_gate" / scope / variant / arm
                 / "full80.json")
    gate = perf_native.read_json(gate_path)
    if (gate.get("protocol") != protocol_id or gate.get("arm") != arm
            or gate.get("variant") != variant or gate.get("scope") != scope
            or _normalized_binding(gate.get("backend_binding", {}))
            != _normalized_binding(binding)):
        raise RuntimeError(f"Legacy timing summary is not bound to the current {scope} gate")


def _read_legacy_scope(root: Path, protocol: dict, native_protocol: dict,
                       arm: str, variant: str, scope: str) -> tuple[dict | None, str | None]:
    """Accept only a complete old 80x3x3 summary; never inspect its sample files."""
    path = _legacy_summary_path(root, scope, arm, variant)
    if not path.is_file():
        return None, "missing"
    try:
        if path.stat().st_size > perf_native.MAX_SUMMARY_BYTES:
            raise ValueError("summary exceeds the bounded size limit")
        summary = perf_native.read_json(path)
        coverage = summary.get("coverage", {})
        expected_keys = perf_native.query_keys(protocol)[:int(protocol["performance"]["warmup_queries"])]
        if not _finite_tree(summary):
            raise ValueError("legacy summary contains non-finite values")
        if (summary.get("format") != perf_native.FORMAT
                or summary.get("protocol") != protocol.get("protocol")
                or summary.get("status") != "complete" or summary.get("complete") is not True
                or summary.get("arm") != arm or summary.get("variant") != variant
                or summary.get("scope") != scope
                or coverage.get("queries") != 80
                or coverage.get("samples") != LEGACY_SCOPE_SAMPLES
                or coverage.get("expected_samples") != LEGACY_SCOPE_SAMPLES
                or coverage.get("blocks") != 3 or coverage.get("repetitions") != 3
                or _legacy_warmup_keys(summary, scope) != expected_keys
                or summary.get("training_updates") != 0
                or summary.get("closed_loop_episodes") != 0):
            raise ValueError("legacy scope identity or 720-sample coverage is incomplete")
        expected_timing_scope = ("synchronized legacy benchmark.run_one sampler.forward wall interval"
                                 if scope == "sampler" else
                                 "CUDA events around exactly five graph replays of frozen model.net.forward inputs; condition/sampler preparation excluded")
        if summary.get("timing_scope") != expected_timing_scope:
            raise ValueError("legacy timing scope differs from the active measurement boundary")
        if scope == "dit_replay" and summary.get("warmup_forwards_per_query") != 5:
            raise ValueError("legacy DiT warmup did not cover five NFE forwards")
        metrics = (("denoising_wall_seconds", "complete_endpoint_query_seconds",
                    "peak_allocated_gb_decimal") if scope == "sampler" else
                   ("dit_replay_seconds", "peak_allocated_gb_decimal"))
        if any(not _positive_stat(summary, key) for key in metrics):
            raise ValueError("legacy summary has missing or non-finite aggregate measurements")
        hardware = summary.get("hardware", {})
        if not all(isinstance(hardware.get(key), str) and hardware[key]
                   for key in ("gpu_uuid", "gpu_name", "hostname", "pbs_job_id")):
            raise ValueError("legacy summary lacks actual GPU/job provenance")
        perf_native.validate_hardware_compatibility(native_protocol, hardware)
        if not isinstance(hardware.get("total_memory_bytes"), int) or hardware["total_memory_bytes"] <= 0:
            raise ValueError("legacy summary lacks GPU memory provenance")
        provenance = summary.get("sample_hardware_provenance", {})
        devices = provenance.get("devices")
        if provenance.get("sample_count") != LEGACY_SCOPE_SAMPLES or not isinstance(devices, list) or not devices:
            raise ValueError("legacy sample provenance does not cover all 720 measurements")
        if any(not device.get("gpu_uuid") or not device.get("hostnames")
               or not device.get("pbs_job_ids") for device in devices):
            raise ValueError("legacy sample provenance is missing device, host, or job identifiers")
        binding = summary.get("backend_binding")
        if (not isinstance(binding, dict) or binding.get("arm") != arm
                or binding.get("requested_variant") != variant):
            raise ValueError("legacy summary lacks the selected backend binding")
        if arm == "bf16":
            if binding.get("native_int4_kernel_used") is not False:
                raise ValueError("BF16 legacy scope has unexpected native backend metadata")
        else:
            _check_gate_binding(root, native_protocol, protocol.get("protocol"),
                                arm, variant, binding, scope)
        return summary, None
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        return None, f"rejected: {type(exc).__name__}: {exc}"


def _scope_pair_compatible(left: dict, right: dict) -> bool:
    return (perf_native.compatible_gpu_key(left.get("hardware", {}))
            == perf_native.compatible_gpu_key(right.get("hardware", {}))
            and _normalized_binding(left.get("backend_binding", {}))
            == _normalized_binding(right.get("backend_binding", {})))


def _receipt_path(root: Path, arm: str, variant: str) -> Path:
    return perf_native.performance_root(root) / "panel_arm" / arm / variant / "summary.json"


def _write_receipt(root: Path, arm: str, variant: str, receipt: dict) -> int:
    return perf_native._bounded_summary(_receipt_path(root, arm, variant), receipt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=perf_native.ARMS, required=True)
    parser.add_argument("--variant", choices=("current", "optimized", "native_fused", "native_grouped"),
                        required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    old_root, root, out = perf_native.validate_paths(args.old_root, args.root, args.out)
    native_protocol = perf_native.read_json(root / "native_protocol.json")
    protocol = perf_native.read_json(old_root / "protocol.json")
    perf_native.validate_protocol(protocol, args.arm)
    panel_native.validate_panel_spec(native_protocol, protocol)
    request = validate_arm_request(native_protocol, args.arm, args.variant, root)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    reused, rejected = {}, {}
    for scope in SCOPES:
        summary, reason = _read_legacy_scope(
            root, protocol, native_protocol, args.arm, args.variant, scope)
        if summary is not None:
            reused[scope] = summary
        else:
            rejected[scope] = reason
    if len(reused) == 2 and not _scope_pair_compatible(reused["sampler"], reused["dit_replay"]):
        rejected.update({scope: "rejected: legacy scopes have incompatible hardware/backend bindings"
                         for scope in SCOPES})
        reused.clear()

    receipt = {
        "format": DRIVER_FORMAT,
        "status": "reusing_legacy" if len(reused) == len(SCOPES) else "loading_runtime",
        "complete": False,
        "protocol": protocol["protocol"],
        "native_protocol": native_protocol.get("protocol"),
        "selection_status_at_start": request["selection_status"],
        "arm": args.arm,
        "variant": args.variant,
        "gate_prechecked": request["gate_prechecked"],
        "baseline_fidelity_policy": native_protocol.get("baseline_fidelity_policy"),
        "fidelity_policy": request["fidelity_policy"],
        "fidelity_failure_blocks_panel": request["fidelity_failure_blocks_panel"],
        "fidelity_policy_note": request["fidelity_policy_note"],
        "performance_design": perf_native.PERFORMANCE_DESIGN,
        "scope_order": list(SCOPES),
        "expected_coverage": {"queries": perf_native.PERFORMANCE_SAMPLES,
                              "blocks": perf_native.PERFORMANCE_BLOCKS,
                              "repetitions": perf_native.PERFORMANCE_REPETITIONS,
                              "samples_per_scope": perf_native.PERFORMANCE_SAMPLES,
                              "scope_count": 2,
                              "expected_samples_total": perf_native.PERFORMANCE_SAMPLES * 2,
                              "coverage_source": "old complete 720-sample scope reused where compatible; otherwise 40ctx_once_v3"},
        "legacy_scope_rejections": rejected,
        "scope_results": [],
        "training_updates": 0,
        "closed_loop_episodes": 0,
        "deadline_seconds_from_job_start": SOFT_DEADLINE_SECONDS,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_receipt(root, args.arm, args.variant, receipt)

    if len(reused) == len(SCOPES):
        for scope in SCOPES:
            summary = reused[scope]
            receipt["scope_results"].append(panel_native._scope_record(
                summary, root, scope, args.arm, args.variant, [],
                summary_path=_legacy_summary_path(root, scope, args.arm, args.variant),
                reused=True, source_design=LEGACY_DESIGN))
        receipt.update({"status": "complete", "complete": True,
                        "completed_scope_rows": len(SCOPES),
                        "completed_samples_total": sum(
                            row["samples"] for row in receipt["scope_results"]),
                        "runtime_loaded": False,
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        _write_receipt(root, args.arm, args.variant, receipt)
        return 0

    # No model code or Torch is imported until summary-only reuse is ruled out.
    code_root = perf_native._prepare_code_imports(old_root, None)
    if args.arm in ROTATION_BACKEND_ARMS:
        perf_native._prepare_rotation_backend_import(root)
    import common
    perf_native._allocation_guard(common)
    deadline = perf_native._deadline(SimpleNamespace(
        soft_deadline_seconds=SOFT_DEADLINE_SECONDS))

    runtime = torch = device = hardware = None
    original_offload = None
    try:
        with startup_trace(out):
            torch = common.torch_module()
            runtime = common.prepare_runtime(protocol, old_root, out)
        device = next(runtime.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Expected CUDA model, found {device}")
        hardware = perf_native._gpu_receipt(torch, device)
        panel_native.validate_hardware_contract(native_protocol, hardware)
        perf_native.validate_hardware_compatibility(native_protocol, hardware)
        for reused_scope, old_summary in reused.items():
            if perf_native.compatible_gpu_key(old_summary["hardware"]) != perf_native.compatible_gpu_key(hardware):
                raise RuntimeError(f"Reused {reused_scope} summary is from an incompatible GPU/software class")
        receipt.update({"status": "running", "hardware": hardware,
                        "runtime_loaded": True,
                        "hostname": hardware.get("hostname"),
                        "runtime_load_completed_utc": datetime.now(timezone.utc).isoformat()})
        _write_receipt(root, args.arm, args.variant, receipt)

        scope_summaries = dict(reused)
        for scope in SCOPES:
            if scope in reused:
                summary = reused[scope]
                cleanup = []
                record = panel_native._scope_record(
                    summary, root, scope, args.arm, args.variant, cleanup,
                    summary_path=_legacy_summary_path(root, scope, args.arm, args.variant),
                    reused=True, source_design=LEGACY_DESIGN)
                receipt["scope_results"].append(record)
                receipt.update({"status": "running", "last_completed_scope": record})
                _write_receipt(root, args.arm, args.variant, receipt)
                continue

            if time.monotonic() >= deadline:
                receipt.update({"status": "checkpointed", "complete": False,
                                "stop_reason": "job soft deadline reached between scopes"})
                _write_receipt(root, args.arm, args.variant, receipt)
                return 2

            if scope == "dit_replay":
                # The previous scope has restored BF16 and dropped all native/graph refs.
                offload_receipt = perf_native._offload_non_dit(runtime, torch, device)
                gc.collect()
                torch.cuda.empty_cache()
                original_offload = perf_native._offload_non_dit

                def reuse_initial_offload(_runtime, _torch, _device):
                    if _runtime is not runtime or _torch is not torch or _device != device:
                        raise RuntimeError("DiT worker requested offload for a different runtime/device")
                    return offload_receipt

                perf_native._offload_non_dit = reuse_initial_offload
                perf_native._read_capture_manifest(root, protocol, hardware)

            summary, cleanup = panel_native._run_scope(
                common, runtime, protocol, old_root, root, out,
                args.arm, args.variant, scope, torch, device, hardware,
                deadline, root)
            if scope_summaries and any(not _scope_pair_compatible(summary, previous)
                                       for previous in scope_summaries.values()):
                raise RuntimeError("Sampler and DiT scopes differ in hardware class or backend binding")
            record = panel_native._scope_record(
                summary, root, scope, args.arm, args.variant, cleanup)
            scope_summaries[scope] = summary
            receipt["scope_results"].append(record)
            receipt.update({"status": "running", "last_completed_scope": record})
            _write_receipt(root, args.arm, args.variant, receipt)
            if not scope_coverage_is_full(summary):
                receipt.update({"status": "checkpointed", "complete": False,
                                "stop_reason": f"{scope} did not complete the active 40-context once-only panel"})
                _write_receipt(root, args.arm, args.variant, receipt)
                return 2

        if len(receipt["scope_results"]) != 2:
            raise RuntimeError("Single-arm panel completed fewer than two frozen scope rows")
        receipt.update({"status": "complete", "complete": True,
                        "completed_scope_rows": 2,
                        "completed_samples_total": sum(
                            row["samples"] for row in receipt["scope_results"]),
                        "runtime_loaded": True,
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        _write_receipt(root, args.arm, args.variant, receipt)
        return 0
    except Exception as exc:
        receipt.update({"status": "failed", "complete": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        _write_receipt(root, args.arm, args.variant, receipt)
        raise
    finally:
        if original_offload is not None:
            perf_native._offload_non_dit = original_offload
        if runtime is not None and torch is not None:
            try:
                perf_native._restore_to_bf16(runtime, protocol, old_root)
            finally:
                panel_native._clear_known_arm_caches(runtime)
                gc.collect()
                torch.cuda.empty_cache()
        _ = code_root


if __name__ == "__main__":
    raise SystemExit(main())
