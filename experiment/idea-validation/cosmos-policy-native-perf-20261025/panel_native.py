"""Run the frozen six-arm/two-scope native panel in one GPU process.

Each of the 40 active performance contexts is measured once per arm and scope. The driver
does not train, recalibrate, or run closed-loop episodes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import perf_native
from startup_trace import startup_trace


DRIVER_FORMAT = "cosmos_native_single_process_panel_v1"
FORMAL_SCOPES = ("sampler", "dit_replay")
SOFT_DEADLINE_SECONDS = 6600
FORMAL_PANEL_READY_STATUS = "ready_for_formal_panel"
INT4_ARMS = frozenset(("spinquant_w4a4", "quarot_w4a4",
                       "svdquant_w4a4", "viditq_w4a4"))


def selected_variants(native_protocol: dict) -> dict[str, str]:
    """Require the frozen, explicitly selected six-arm panel before loading a model."""
    if native_protocol.get("selection_status") != FORMAL_PANEL_READY_STATUS:
        raise RuntimeError("native_protocol.selection_status must be ready_for_formal_panel")
    if native_protocol.get("arms") != list(perf_native.ARMS):
        raise RuntimeError("native_protocol arm order differs from the frozen six-arm panel")
    selected = native_protocol.get("selected_variants")
    if not isinstance(selected, dict) or set(selected) != set(perf_native.ARMS):
        raise RuntimeError("native_protocol.selected_variants must name all six frozen arms")
    ordered = {arm: selected[arm] for arm in perf_native.ARMS}
    for arm, variant in ordered.items():
        perf_native.validate_variant(arm, variant)
    return ordered


def validate_panel_spec(native_protocol: dict, source_protocol: dict) -> None:
    if native_protocol.get("protocol") != perf_native.FORMAT:
        raise RuntimeError("native_protocol protocol id differs from this performance worker")
    if native_protocol.get("source_protocol") != source_protocol.get("protocol"):
        raise RuntimeError("native_protocol is not bound to the read-only source protocol")
    expected = {"case_ids_start": 3000, "case_ids_end_inclusive": 3039,
                "seed_indices": [0], "queries": perf_native.PERFORMANCE_SAMPLES,
                "blocks": perf_native.PERFORMANCE_BLOCKS,
                "repetitions": perf_native.PERFORMANCE_REPETITIONS,
                "batch_size": 1, "nfe": 5}
    panel = native_protocol.get("panel", {})
    if any(panel.get(key) != value for key, value in expected.items()):
        raise RuntimeError("native_protocol panel differs from active 40-context x once x five-NFE design")
    source_performance = source_protocol.get("performance", {})
    source_inference = source_protocol.get("inference", {})
    if (source_performance.get("case_ids") != list(range(3000, 3040))
            or source_performance.get("seed_indices") != [0, 1]
            or source_performance.get("queries") != 80
            or source_inference.get("denoising_steps") != 5):
        raise RuntimeError("read-only source protocol cohort or five-NFE sampler differs")


def formal_plan(variants: dict[str, str]) -> list[dict]:
    """Six sampler arms first, then six DiT replay arms; each covers 40 contexts."""
    if set(variants) != set(perf_native.ARMS):
        raise ValueError("Formal plan requires exactly the six frozen arms")
    return [{"scope": scope, "arm": arm, "variant": variants[arm],
             "expected_samples": perf_native.PERFORMANCE_SAMPLES}
            for scope in FORMAL_SCOPES for arm in perf_native.ARMS]


def expected_formal_coverage() -> dict:
    return {"design": perf_native.PERFORMANCE_DESIGN,
            "queries_per_arm_scope": perf_native.PERFORMANCE_SAMPLES,
            "blocks": perf_native.PERFORMANCE_BLOCKS,
            "repetitions": perf_native.PERFORMANCE_REPETITIONS,
            "samples_per_arm_scope": perf_native.PERFORMANCE_SAMPLES,
            "arm_count": 6, "scope_count": 2,
            "expected_arm_scope_rows": 12,
            "expected_samples_total": (perf_native.PERFORMANCE_SAMPLES * 6 * 2)}


def scope_coverage_is_active(summary: dict) -> bool:
    coverage = summary.get("coverage", {})
    return (summary.get("performance_design") == perf_native.PERFORMANCE_DESIGN
            and summary.get("complete") is True
            and coverage.get("queries") == perf_native.PERFORMANCE_SAMPLES
            and coverage.get("samples") == perf_native.PERFORMANCE_SAMPLES
            and coverage.get("expected_samples") == perf_native.PERFORMANCE_SAMPLES
            and coverage.get("blocks") == perf_native.PERFORMANCE_BLOCKS
            and coverage.get("repetitions") == perf_native.PERFORMANCE_REPETITIONS)


def validate_hardware_contract(spec: dict, hardware: dict) -> None:
    expected = spec.get("hardware", {})
    requirements = {
        "gpu_name": expected.get("gpu"),
        "capability": expected.get("capability"),
        "torch_version": expected.get("torch"),
        "cuda_version": expected.get("cuda_runtime"),
    }
    for key, value in requirements.items():
        if value is not None and hardware.get(key) != value:
            raise RuntimeError(f"Active hardware differs from native_protocol.{key}: "
                               f"expected {value!r}, got {hardware.get(key)!r}")
    if not hardware.get("gpu_uuid"):
        raise RuntimeError("Formal panel requires a concrete physical GPU UUID")


def _clear_known_arm_caches(runtime) -> list[str]:
    """Drop known stale arm/rotation references after restore, preserving CPU shadows."""
    removed = []
    target_linears = getattr(runtime, "target_linears", {})
    for name, module in target_linears.items():
        weight = getattr(module, "weight", None)
        if weight is None or str(getattr(weight, "dtype", None)) != "torch.bfloat16":
            raise RuntimeError(f"Arm cleanup requires restored BF16 target weight at {name}")
    restored_graphs = set()

    def release_graphs(graphs):
        if graphs is None or id(graphs) in restored_graphs:
            return
        restored_graphs.add(id(graphs))
        restore_graphs = getattr(graphs, "restore", None)
        if callable(restore_graphs):
            restore_graphs()
        records = getattr(graphs, "records", None)
        if hasattr(records, "clear"):
            records.clear()

    engine = getattr(runtime, "ours_vqwam_engine", None)
    if engine is not None:
        if any(getattr(module, "weight", None) is None
               for module in target_linears.values()):
            raise RuntimeError("Cannot release the Ours engine before BF16 target weights are restored")
        graphs = getattr(engine, "graphs", None)
        release_graphs(graphs)
        banks = getattr(engine, "banks", None)
        if banks is not None:
            workspaces = getattr(banks, "graph_workspaces", None)
            if hasattr(workspaces, "clear"):
                workspaces.clear()
            if hasattr(banks, "default_workspace"):
                banks.workspace = banks.default_workspace
        runtime.__dict__.pop("ours_vqwam_engine", None)
        removed.append("runtime.ours_vqwam_engine")
    optimized_graphs = getattr(runtime, "optimized_vq_graphs", None)
    if optimized_graphs is not None:
        release_graphs(optimized_graphs)
        runtime.__dict__.pop("optimized_vq_graphs", None)
        removed.append("runtime.optimized_vq_graphs")
    for attribute in ("restore_ours_vqwam", "restore_optimized_engine",
                      "restore_native_rotation", "restore_native_diffusion"):
        if attribute in runtime.__dict__:
            runtime.__dict__.pop(attribute, None)
            removed.append(f"runtime.{attribute}")
    for owner_name, owner in (("runtime", runtime),
                              ("controller", getattr(runtime, "controller", None))):
        if owner is None or not hasattr(owner, "__dict__"):
            continue
        for attribute in ("_device_rotation_cache", "_ours_reference_holds"):
            if attribute in owner.__dict__:
                cached = owner.__dict__[attribute]
                if attribute == "_device_rotation_cache" and isinstance(cached, dict):
                    cached.clear()
                    removed.append(f"{owner_name}.{attribute}.clear")
                else:
                    owner.__dict__.pop(attribute, None)
                    removed.append(f"{owner_name}.{attribute}")
    for name, module in target_linears.items():
        for attribute in ("_cosmos_spin_state", "_native_grouped_context",
                          "_native_rotation_context", "_native_rotation_scales",
                          "_native_rotation_signs", "_native_activation_dtype",
                          "_cosmos_adapted_baseline_arm", "_cosmos_native_arm"):
            if attribute in module.__dict__:
                module.__dict__.pop(attribute, None)
                removed.append(f"target_linears[{name}].{attribute}")
        buffers = getattr(module, "_buffers", None)
        if isinstance(buffers, dict):
            for attribute in ("_native_rotation_scales", "_native_rotation_signs"):
                if attribute in buffers:
                    buffers.pop(attribute, None)
                    removed.append(f"target_linears[{name}]._buffers[{attribute}]")
    return removed


def _finish_arm(runtime, protocol, old_root: Path, restore_hook, torch) -> list[str]:
    """Run arm hook, explicitly reinstall BF16, and release transient GPU references."""
    try:
        if callable(restore_hook):
            restore_hook()
    finally:
        try:
            perf_native._restore_to_bf16(runtime, protocol, old_root)
        finally:
            removed = _clear_known_arm_caches(runtime)
            gc.collect()
            torch.cuda.empty_cache()
    return removed


def _summary_path(root: Path) -> Path:
    return perf_native.performance_root(root) / "panel_driver" / "summary.json"


def _write_receipt(root: Path, receipt: dict) -> int:
    return perf_native._bounded_summary(_summary_path(root), receipt)


def _scope_record(summary: dict, root: Path, scope: str, arm: str,
                  variant: str, cleanup: list[str], *, summary_path: Path | None = None,
                  reused: bool = False, source_design: str | None = None) -> dict:
    path = (summary_path if summary_path is not None else
            perf_native.performance_root(root) / scope / variant / arm / "summary.json")
    coverage = summary.get("coverage", {})
    return {"scope": scope, "arm": arm, "variant": variant,
            "status": summary.get("status"), "complete": summary.get("complete") is True,
            "performance_design": summary.get("performance_design"),
            "samples": coverage.get("samples"),
            "expected_samples": coverage.get("expected_samples", perf_native.PERFORMANCE_SAMPLES),
            "summary_path": str(path),
            "reused": reused,
            "source_design": source_design or summary.get("performance_design"),
            "gpu_uuid": summary.get("hardware", {}).get("gpu_uuid"),
            "cleanup_released_known_caches": cleanup}


def _run_scope(common, runtime, protocol, old_root: Path, root: Path, out: Path,
               arm: str, variant: str, scope: str, torch, device, hardware,
               deadline: float, native_root: Path) -> tuple[dict, list[str]]:
    restore_hook = None
    cleanup = []
    try:
        if arm != "bf16":
            perf_native._require_gate(root, arm, variant, scope)
        metadata, restore_hook = perf_native._install_native(
            runtime, protocol, old_root, out, arm, variant, {}, native_root=native_root)
        if arm in INT4_ARMS and metadata.get("native_int4_kernel_used") is not True:
            raise RuntimeError(f"{arm}/{variant} lacks positive native INT4 kernel evidence")
        binding = perf_native._binding(metadata, arm, variant, root)
        perf_native.assert_native_gate_binding(root, arm, variant, binding)
        storage = perf_native._packed_storage_receipt(runtime, metadata, arm, torch, device)
        if scope == "sampler":
            summary = perf_native._run_sampler_benchmark(
                common, runtime, protocol, old_root, root, out, arm, variant,
                torch, device, hardware, metadata, binding, storage, deadline)
        else:
            perf_native._read_capture_manifest(root, protocol, hardware)
            summary = perf_native._run_dit_benchmark(
                runtime, protocol, root, out, arm, variant, torch, device,
                hardware, metadata, binding, storage, deadline)
    finally:
        cleanup = _finish_arm(runtime, protocol, old_root, restore_hook, torch)
        # The caller writes cleanup details after this function returns. The
        # summary dictionary is immutable evidence and contains no CUDA tensors.
    return summary, cleanup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-gpu-uuid")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_gpu_uuid is not None and not args.require_gpu_uuid.strip():
        raise ValueError("--require-gpu-uuid must be a non-empty physical GPU UUID")
    old_root, root, out = perf_native.validate_paths(args.old_root, args.root, args.out)
    native_protocol = perf_native.read_json(root / "native_protocol.json")
    variants = selected_variants(native_protocol)
    protocol = perf_native.read_json(old_root / "protocol.json")
    perf_native.validate_protocol(protocol, "bf16")
    validate_panel_spec(native_protocol, protocol)
    plan = formal_plan(variants)

    code_root = perf_native._prepare_code_imports(old_root, None)
    if any(arm in ("spinquant_w4a4", "quarot_w4a4") for arm in perf_native.ARMS):
        perf_native._prepare_rotation_backend_import(root)
    import common
    perf_native._allocation_guard(common)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    deadline = perf_native._deadline(SimpleNamespace(
        soft_deadline_seconds=SOFT_DEADLINE_SECONDS))
    started_utc = datetime.now(timezone.utc).isoformat()
    receipt = {
        "format": DRIVER_FORMAT,
        "status": "loading_runtime",
        "complete": False,
        "protocol": protocol["protocol"],
        "native_protocol": native_protocol.get("protocol"),
        "selection_status": native_protocol.get("selection_status"),
        "selected_variants": variants,
        "coverage": expected_formal_coverage(),
        "performance_design": perf_native.PERFORMANCE_DESIGN,
        "scope_order": list(FORMAL_SCOPES),
        "scope_results": [],
        "training_updates": 0,
        "closed_loop_episodes": 0,
        "deadline_seconds_from_job_start": SOFT_DEADLINE_SECONDS,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "created_utc": started_utc,
    }
    _write_receipt(root, receipt)

    # The deadline starts before the one cold runtime load, so initialization
    # time cannot silently extend the allocation budget.
    runtime = None
    torch = device = hardware = None
    original_offload = None
    offload_receipt = None
    try:
        with startup_trace(out):
            torch = common.torch_module()
            runtime = common.prepare_runtime(protocol, old_root, out)
        device = next(runtime.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Expected CUDA model, found {device}")
        hardware = perf_native._gpu_receipt(torch, device, args.require_gpu_uuid)
        validate_hardware_contract(native_protocol, hardware)
        receipt.update({"status": "running", "hardware": hardware,
                        "hostname": hardware.get("hostname"),
                        "runtime_load_completed_utc": datetime.now(timezone.utc).isoformat()})
        _write_receipt(root, receipt)

        if time.monotonic() >= deadline:
            receipt.update({"status": "checkpointed", "complete": False,
                            "stop_reason": "job soft deadline reached during cold runtime load"})
            _write_receipt(root, receipt)
            return 2

        # Formal sampler wall measurements are complete before any non-DiT
        # module is offloaded for the DiT-only memory scope.
        for item in plan[:6]:
            if time.monotonic() >= deadline:
                receipt.update({"status": "checkpointed", "complete": False,
                                "stop_reason": "job soft deadline reached between arm scopes"})
                _write_receipt(root, receipt)
                return 2
            summary, cleanup = _run_scope(
                common, runtime, protocol, old_root, root, out,
                item["arm"], item["variant"], item["scope"], torch, device,
                hardware, deadline, root)
            record = _scope_record(summary, root, item["scope"], item["arm"],
                                   item["variant"], cleanup)
            receipt["scope_results"].append(record)
            receipt.update({"status": "running", "last_completed_scope": record})
            _write_receipt(root, receipt)
            coverage = summary.get("coverage", {})
            if not scope_coverage_is_active(summary):
                receipt.update({"status": "checkpointed", "complete": False,
                                "stop_reason": "sampler arm did not complete the active 40-context once-only design"})
                _write_receipt(root, receipt)
                return 2

        # Offload once at the sampler->DiT boundary. The existing DiT worker
        # asks for an offload receipt on every arm; return this frozen receipt
        # for its later calls without repeating .to('cpu')/scope mutation.
        offload_receipt = perf_native._offload_non_dit(runtime, torch, device)
        gc.collect()
        torch.cuda.empty_cache()
        original_offload = perf_native._offload_non_dit

        def reuse_initial_offload(_runtime, _torch, _device):
            if _runtime is not runtime or _torch is not torch or _device != device:
                raise RuntimeError("DiT worker requested offload for a different runtime/device")
            return offload_receipt

        perf_native._offload_non_dit = reuse_initial_offload
        try:
            perf_native._read_capture_manifest(root, protocol, hardware)
            for item in plan[6:]:
                if time.monotonic() >= deadline:
                    receipt.update({"status": "checkpointed", "complete": False,
                                    "stop_reason": "job soft deadline reached between arm scopes"})
                    _write_receipt(root, receipt)
                    return 2
                summary, cleanup = _run_scope(
                    common, runtime, protocol, old_root, root, out,
                    item["arm"], item["variant"], item["scope"], torch, device,
                    hardware, deadline, root)
                record = _scope_record(summary, root, item["scope"], item["arm"],
                                       item["variant"], cleanup)
                receipt["scope_results"].append(record)
                receipt.update({"status": "running", "last_completed_scope": record})
                _write_receipt(root, receipt)
                coverage = summary.get("coverage", {})
                if not scope_coverage_is_active(summary):
                    receipt.update({"status": "checkpointed", "complete": False,
                                    "stop_reason": "DiT arm did not complete the active 40-context once-only design"})
                    _write_receipt(root, receipt)
                    return 2
        finally:
            perf_native._offload_non_dit = original_offload
            original_offload = None

        if len(receipt["scope_results"]) != 12:
            raise RuntimeError("Formal panel completed fewer than 12 arm/scope rows")
        if any(row.get("performance_design") != perf_native.PERFORMANCE_DESIGN
               or row.get("samples") != perf_native.PERFORMANCE_SAMPLES
               or row.get("expected_samples") != perf_native.PERFORMANCE_SAMPLES
               or row.get("complete") is not True
               for row in receipt["scope_results"]):
            raise RuntimeError("Formal panel summary has incomplete arm/scope coverage")
        receipt.update({"status": "complete", "complete": True,
                        "completed_scope_rows": 12,
                        "completed_samples_total": sum(row["samples"]
                                                        for row in receipt["scope_results"]),
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        _write_receipt(root, receipt)
        return 0
    except Exception as exc:
        receipt.update({"status": "failed", "complete": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        _write_receipt(root, receipt)
        raise
    finally:
        if original_offload is not None:
            perf_native._offload_non_dit = original_offload
        if runtime is not None and torch is not None:
            try:
                perf_native._restore_to_bf16(runtime, protocol, old_root)
            finally:
                _clear_known_arm_caches(runtime)
                gc.collect()
                torch.cuda.empty_cache()
        _ = code_root


if __name__ == "__main__":
    raise SystemExit(main())
