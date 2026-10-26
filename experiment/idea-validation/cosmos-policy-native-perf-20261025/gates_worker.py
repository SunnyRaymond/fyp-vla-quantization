"""Run independent native fidelity candidates on one loaded Cosmos runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
from types import SimpleNamespace

import perf_native as perf


ARM_ALIASES = {
    "spinquant": "spinquant_w4a4",
    "quarot": "quarot_w4a4",
    "svdquant": "svdquant_w4a4",
    "viditq": "viditq_w4a4",
}
OURS_ARM = "ours_vqwam"
OURS_ALIASES = {"ours": OURS_ARM}
BASELINE_ARMS = tuple(ARM_ALIASES.values())
GATE_SCOPES = ("sampler", "dit_replay")
SOFT_DEADLINE_SECONDS = int(os.environ.get("COSMOS_SOFT_DEADLINE_SECONDS", "6600"))
WORKER_FORMAT = "cosmos_native_gate_worker_v1"


def normalize_arm(value: str) -> str:
    arm = OURS_ALIASES.get(value, ARM_ALIASES.get(value, value))
    if arm not in (*BASELINE_ARMS, OURS_ARM):
        raise ValueError(f"Unsupported baseline gate arm: {value}")
    return arm


def candidate_variants(arm: str) -> tuple[str, ...]:
    arm = normalize_arm(arm)
    if arm == OURS_ARM:
        return ("optimized",)
    if arm in ("spinquant_w4a4", "quarot_w4a4"):
        return ("current", "native_grouped")
    if arm == "svdquant_w4a4":
        return ("current", "native_fused")
    return ("current",)


def candidate_passed(scope_rows: list[dict]) -> bool:
    by_scope = {row.get("scope"): row for row in scope_rows}
    return (set(by_scope) == set(GATE_SCOPES)
            and all(by_scope[scope].get("full80_passed") is True
                    and by_scope[scope].get("status") == "passed"
                    for scope in GATE_SCOPES))


def run_post_gate_stages(ours_gate, svd_diagnostic, *, arm=None):
    """Keep SVD's DiT diagnostic after Ours, since it offloads non-DiT modules."""
    if arm == OURS_ARM:
        return ({"status": "not_applicable",
                 "reason": "dedicated Ours worker already ran the optimized candidate"},
                {"status": "not_applicable"})
    ours_result = ours_gate()
    diagnostic_result = svd_diagnostic()
    return ours_result, diagnostic_result


def evaluate_candidates(variants, scopes, run_scope, *, preflight=None, on_scope=None):
    """Keep candidate failures local and never reduce the required scope set."""
    results = []
    for variant in variants:
        try:
            check = {"status": "ready"} if preflight is None else preflight(variant)
        except Exception as exc:
            check = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:1000]}
        row = {"variant": variant, "preflight": check, "scopes": []}
        if check.get("status") != "ready":
            state = "deferred" if check.get("status") == "deferred" else "failed"
            row["scopes"] = [{"scope": scope, "status": "skipped",
                              "full80_passed": False,
                              "reason": check.get("reason", "candidate preflight failed")}
                             for scope in scopes]
            row["status"] = state
            results.append(row)
            if on_scope is not None:
                for scope_row in row["scopes"]:
                    on_scope(row, scope_row)
            continue
        for scope in scopes:
            try:
                scope_row = run_scope(variant, scope)
            except Exception as exc:
                scope_row = {"scope": scope, "status": "error", "full80_passed": False,
                             "error": {"type": type(exc).__name__, "message": str(exc)[:1000]}}
            row["scopes"].append(scope_row)
            if on_scope is not None:
                on_scope(row, scope_row)
        row["status"] = "passed" if candidate_passed(row["scopes"]) else "failed"
        results.append(row)
    return results


def _safe_job_component(job_id: str | None) -> str:
    if not job_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise RuntimeError("A valid PBS_JOBID is required for gate worker receipts")
    return job_id


def worker_summary_path(root: Path, arm: str, job_id: str) -> Path:
    return root / "results" / "native_gate_workers" / arm / _safe_job_component(job_id) / "summary.json"


def acquire_ours_claim(root: Path, merged_marker: Path, job_id: str) -> dict:
    """Atomically grant one worker in this PBS allocation the Ours gate slot."""
    if not merged_marker.is_file():
        return {"status": "not_requested", "claimed": False}
    marker = perf.read_json(merged_marker)
    if not isinstance(marker, dict):
        raise RuntimeError("Merged Ours gate marker must be a JSON object")
    if "enabled" in marker and not isinstance(marker["enabled"], bool):
        raise RuntimeError("Merged Ours gate marker enabled must be a JSON boolean")
    if marker.get("enabled") is False:
        return {"status": "disabled", "claimed": False,
                "marker_path": str(merged_marker)}
    component = _safe_job_component(job_id)
    claim = root / "artifacts" / component / "ours_gate_claim"
    try:
        claim.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return {"status": "already_claimed", "claimed": False,
                "claim_path": str(claim)}
    return {"status": "claimed", "claimed": True, "claim_path": str(claim),
            "pbs_job_id": component, "hostname": socket.gethostname()}


def _write_summary(path: Path, report: dict) -> None:
    report["updated_utc"] = datetime.now(timezone.utc).isoformat()
    report["summary_bytes"] = perf._bounded_summary(path, report)


def _scope_row_from_gate(scope: str, gate: dict) -> dict:
    passed = gate.get("full80_passed") is True and gate.get("full80_status") == "passed"
    return {"scope": scope, "status": "passed" if passed else gate.get("status", "not_passed"),
            "full80_passed": passed, "full80_status": gate.get("full80_status"),
            "representative4_status": gate.get("representative4_status"),
            "gate_status": gate.get("status"), "backend_binding": gate.get("backend_binding", {})}


def _run_gate_scope(common, panel, runtime, protocol, old_root, root, out,
                    arm, variant, scope, torch, device, hardware, deadline):
    gate = None
    error = None
    try:
        gate = perf._gate_panel(common, runtime, protocol, old_root, root, out,
            arm, variant, scope, torch, device, hardware, {}, deadline)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1000],
                 "traceback": traceback.format_exc()[-4000:]}
    try:
        # _gate_panel restores target weights; this also clears arm-level caches.
        panel._finish_arm(runtime, protocol, old_root, _active_restore_hook(runtime), torch)
    except Exception as exc:
        cleanup = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        error = {"gate_error": error, "cleanup_error": cleanup} if error else cleanup
    if error:
        return {"scope": scope, "status": "error", "full80_passed": False, "error": error}
    return _scope_row_from_gate(scope, gate or {})


def _active_restore_hook(runtime):
    for name in ("restore_optimized_engine", "restore_ours_vqwam",
                 "restore_native_rotation", "restore_native_diffusion",
                 "_cosmos_native_restore"):
        hook = getattr(runtime, name, None)
        if callable(hook):
            return hook
    return None


def _read_diffusion_build(root: Path, arm: str):
    filename = ('native_viditq_general_build.json' if arm == 'viditq_w4a4'
                else 'native_diffusion_build.json')
    paths = (root / "results" / "preparation" / filename,
             root / "retrieved" / ('preparation.' + filename))
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        return None, f"Native diffusion build receipt missing for {arm}"
    try:
        receipt = perf.read_json(path)
    except Exception as exc:
        return None, f"Cannot read diffusion build receipt: {exc}"
    entry = receipt if arm == 'viditq_w4a4' else receipt.get("arms", {}).get(arm)
    module = Path(entry.get("module", "")) if isinstance(entry, dict) else Path()
    if (receipt.get("status") != "complete" or not receipt.get("pbs_job_id")
            or not isinstance(entry, dict) or not entry.get("source_commit")
            or not module.is_file()):
        return None, f"Native diffusion build receipt/module is not ready for {arm}"
    return {"path": str(path), "pbs_job_id": receipt["pbs_job_id"],
            "source_commit": entry["source_commit"], "module": str(module)}, None


def _probe_receipt_matches(probe: dict, build: dict, arm: dict, probe_module) -> bool:
    producer, source = probe.get("producer", {}), probe.get("source", {})
    cases = probe.get("cases", [])
    if (probe.get("status") != "complete"
            or probe.get("probe") != "nunchaku_official_svdquant_rank16_lora_fusion"
            or producer.get("build_pbs_job_id") != build.get("pbs_job_id")
            or source.get("nunchaku_commit") != probe_module.NUNCHAKU_COMMIT
            or source.get("deepcompressor_packing_commit") != probe_module.DEEPCOMPRESSOR_COMMIT
            or source.get("module_name") != probe_module.MODULE_NAME
            or Path(source.get("module", "")).resolve() != Path(arm.get("module", "")).resolve()
            or probe.get("policy_forward_calls") != 0
            or probe.get("training_steps") != 0
            or probe.get("closed_loop_episodes") != 0):
        return False
    if [case.get("true_m") for case in cases] != list(probe_module.M_CASES):
        return False
    required_phases = ("default_current_stream", "non_default_current_stream", "cuda_graph")
    for case in cases:
        phases = case.get("stream_tests", {})
        if any(phases.get(name, {}).get("passed") is not True for name in required_phases[:2]):
            return False
        graph = phases.get("cuda_graph", {})
        if graph.get("capture_passed") is not True or graph.get("replay_passed") is not True:
            return False
    return True


def _ensure_svd_fused_probe(root: Path) -> dict:
    import probe_svd_fused

    build, arm = probe_svd_fused._load_build_receipt(root)
    path = root / "results" / "preparation" / "svd_fused_kernel_probe.json"
    prior = perf.read_json(path) if path.is_file() else None
    if prior is not None and _probe_receipt_matches(prior, build, arm, probe_svd_fused):
        return {"status": "ready", "source": "existing_probe_receipt",
                "probe_path": str(path), "build_pbs_job_id": build["pbs_job_id"]}
    result = probe_svd_fused._run(root, path)
    if not _probe_receipt_matches(result, build, arm, probe_svd_fused):
        raise RuntimeError("SVD fused native kernel probe did not produce a matching complete receipt")
    return {"status": "ready", "source": "executed_probe_helper",
            "probe_path": str(path), "build_pbs_job_id": build["pbs_job_id"]}


def _candidate_preflight(root: Path, arm: str, variant: str) -> dict:
    if arm == "svdquant_w4a4":
        if variant == "native_fused":
            try:
                return _ensure_svd_fused_probe(root)
            except FileNotFoundError as exc:
                return {"status": "deferred", "reason": str(exc)[:1000]}
            except Exception as exc:
                return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:1000]}
        receipt, reason = _read_diffusion_build(root, "svdquant_w4a4")
        return ({"status": "ready", "build": receipt} if receipt else
                {"status": "deferred", "reason": reason})
    if arm == "viditq_w4a4":
        receipt, reason = _read_diffusion_build(root, "viditq_w4a4")
        return ({"status": "ready", "build": receipt} if receipt else
                {"status": "deferred", "reason": reason})
    if variant == "native_grouped":
        receipt, reason = _read_diffusion_build(root, "svdquant_w4a4")
        return ({"status": "ready", "build": receipt} if receipt else
                {"status": "deferred", "reason": reason})
    return {"status": "ready"}


def _run_svd_diagnostic(common, panel, runtime, protocol, old_root, root, out,
                        torch, device, hardware, report_path):
    import compare_svd_runtime as compare

    result = {"format": compare.FORMAT, "status": "running", "diagnostic_only": True,
        "protocol": protocol["protocol"], "arm": "svdquant_w4a4",
        "variants": list(compare.VARIANTS), "scope_order": ["sampler", "dit_replay"],
        "diagnostic_keys": [list(key) for key in compare.DIAGNOSTIC_KEYS],
        "warmup_keys": [list(key) for key in compare.WARMUP_KEYS],
        "warmup_queries_per_variant_per_scope": len(compare.WARMUP_KEYS),
        "measurements_per_variant_per_scope": 12,
        "formal_protocol": {"frozen_queries": 80, "blocks": 3, "repetitions": 3,
                             "samples_per_arm": 720},
        "hardware": hardware, "training_updates": 0, "closed_loop_episodes": 0,
        "selection_policy": "engineering diagnostic only; does not change formal SVD variant",
        "formal_variant_selection_changed": False,
        "scopes": {}, "created_utc": datetime.now(timezone.utc).isoformat()}

    def sample_scope(scope, variant, offload=None):
        try:
            if scope == "sampler":
                measurement = compare._measure_sampler_variant(
                    perf, common, runtime, protocol, old_root, root, out,
                    variant, torch, device, hardware)
            else:
                measurement = compare._measure_dit_variant(
                    perf, runtime, protocol, old_root, root, out,
                    variant, torch, device, hardware, offload)
            row = {"status": "complete", **measurement}
        except Exception as exc:
            row = {"status": "failed", "error": {"type": type(exc).__name__,
                    "message": str(exc)[:1000]}}
        try:
            panel._finish_arm(runtime, protocol, old_root, _active_restore_hook(runtime), torch)
        except Exception as exc:
            row = {"status": "failed", "measurement": row,
                   "cleanup_error": {"type": type(exc).__name__,
                                     "message": str(exc)[:1000]}}
        return row

    sampler = {}
    for variant in compare.VARIANTS:
        sampler[variant] = sample_scope("sampler", variant)
    if all(row.get("status") == "complete" for row in sampler.values()):
        result["scopes"]["sampler"] = compare._scope_comparison("sampler", sampler)
    else:
        result["scopes"]["sampler"] = {"scope": "sampler", "status": "failed",
                                        "variants": sampler,
                                        "diagnostic_measurements_per_variant": 12}
    _write_summary(report_path, result)

    try:
        offload = perf._offload_non_dit(runtime, torch, device)
        gc.collect()
        torch.cuda.empty_cache()
        dit = {variant: sample_scope("dit_replay", variant, offload)
               for variant in compare.VARIANTS}
    except Exception as exc:
        dit = {variant: {"status": "failed", "error": {"type": type(exc).__name__,
                "message": str(exc)[:1000]}} for variant in compare.VARIANTS}
    if all(row.get("status") == "complete" for row in dit.values()):
        result["scopes"]["dit_replay"] = compare._scope_comparison("dit_replay", dit)
    else:
        result["scopes"]["dit_replay"] = {"scope": "dit_replay", "status": "failed",
                                           "variants": dit,
                                           "diagnostic_measurements_per_variant": 12}
    result["status"] = ("complete" if all(
        result["scopes"][scope].get("diagnostic_measurements_per_variant") == 12
        and all(variant.get("status") == "complete"
                for variant in result["scopes"][scope].get("variants", {}).values())
        for scope in GATE_SCOPES) else "failed")
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_summary(report_path, result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARM_ALIASES.keys(), *BASELINE_ARMS,
                                            *OURS_ALIASES.keys(), OURS_ARM), required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    arm = normalize_arm(args.arm)
    old_root, root, out = perf.validate_paths(args.old_root, args.root, args.out)
    protocol = perf.read_json(old_root / "protocol.json")
    perf.validate_protocol(protocol, arm)
    native_protocol_path = root / "native_protocol.json"
    native_protocol = perf.read_json(native_protocol_path)
    if native_protocol.get("source_protocol") != protocol.get("protocol"):
        raise RuntimeError("Native gate root is not bound to the read-only source protocol")
    code_root = perf._prepare_code_imports(old_root, None)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("PBS_JOBID")
    summary_path = worker_summary_path(root, arm, job_id or "unallocated")
    report = {"format": WORKER_FORMAT, "status": "starting", "complete": False,
        "arm": arm, "candidates": [], "ours_gate": {"status": "not_requested"},
        "ours_gate_claimed": False, "svd_runtime_diagnostic": {"status": "not_applicable"},
        "scope_order": list(GATE_SCOPES), "training_updates": 0,
        "closed_loop_episodes": 0,
        "formal_panel": {"queries": 80, "blocks": 3, "repetitions": 3,
                          "samples_per_candidate_scope": 720},
        "pbs_job_id": job_id, "created_utc": datetime.now(timezone.utc).isoformat()}
    runtime = torch = device = hardware = None
    passed = False
    try:
        import common
        perf._allocation_guard(common)
        if arm in ("spinquant_w4a4", "quarot_w4a4", "viditq_w4a4"):
            perf._prepare_rotation_backend_import(root)
        from startup_trace import startup_trace
        with startup_trace(out):
            torch = common.torch_module()
            runtime = common.prepare_runtime(protocol, old_root, out)
        device = next(runtime.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Native gate worker requires CUDA, found {device}")
        hardware = perf._gpu_receipt(torch, device)
        perf.validate_hardware_compatibility(native_protocol, hardware)
        deadline = perf._deadline(SimpleNamespace(soft_deadline_seconds=SOFT_DEADLINE_SECONDS))
        report.update({"status": "running", "hardware": hardware,
                       "hostname": hardware.get("hostname"),
                       "runtime_prepare_calls": 1,
                       "runtime_loaded_utc": datetime.now(timezone.utc).isoformat()})
        _write_summary(summary_path, report)

        def run_candidates(variants, candidate_arm=arm):
            def run_scope(variant, scope):
                return _run_gate_scope(common, panel_native, runtime, protocol, old_root,
                    root, out, candidate_arm, variant, scope, torch, device, hardware, deadline)

            def candidate_preflight(variant):
                return _candidate_preflight(root, candidate_arm, variant)

            def after_scope(candidate, _scope):
                report["active_candidate"] = candidate
                report["status"] = "running"
                _write_summary(summary_path, report)

            rows = evaluate_candidates(variants, GATE_SCOPES, run_scope,
                preflight=candidate_preflight, on_scope=after_scope)
            report["candidates"].extend(rows)
            report.pop("active_candidate", None)
            _write_summary(summary_path, report)
            return rows

        import panel_native

        baseline_results = run_candidates(candidate_variants(arm))
        def run_ours_gate_stage():
            claim = acquire_ours_claim(root, root / "ours_gate_merged.json", job_id)
            if claim.get("claimed"):
                ours_results = run_candidates(("optimized",), candidate_arm="ours_vqwam")
                claim.update({"candidate": ours_results[0],
                              "passed": ours_results[0].get("status") == "passed"})
                report["ours_gate_claimed"] = True
            _write_summary(summary_path, report)
            return claim

        def run_svd_diagnostic_stage():
            if arm != "svdquant_w4a4":
                return {"status": "not_applicable"}
            full_gate_variants = {row["variant"] for row in baseline_results
                                  if row.get("status") == "passed"}
            if not {"current", "native_fused"}.issubset(full_gate_variants):
                return {"status": "not_run",
                        "reason": "both current and native_fused need sampler and DiT full80 passes"}
            # This DiT-only diagnostic offloads non-DiT modules, so it is the
            # final GPU stage after any claimed Ours gate.
            diagnostic_path = root / "results" / "svd_fusion_diagnostic" / "summary.json"
            return _run_svd_diagnostic(common, panel_native, runtime, protocol,
                old_root, root, out, torch, device, hardware, diagnostic_path)

        report["ours_gate"], report["svd_runtime_diagnostic"] = run_post_gate_stages(
            run_ours_gate_stage, run_svd_diagnostic_stage, arm=arm)
        _write_summary(summary_path, report)

        base_pass = any(row.get("status") == "passed" for row in baseline_results)
        ours_required_pass = (not report.get("ours_gate_claimed")
                              or report["ours_gate"].get("passed") is True)
        passed = base_pass and ours_required_pass
    except Exception as exc:
        report["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        if runtime is not None and torch is not None:
            try:
                import panel_native
                panel_native._finish_arm(runtime, protocol, old_root, None, torch)
            except Exception as exc:
                report["final_restore_error"] = {"type": type(exc).__name__,
                                                  "message": str(exc)[:1000]}
                passed = False
        report.update({"status": "complete" if passed else "failed",
                       "complete": bool(passed),
                       "finished_utc": datetime.now(timezone.utc).isoformat()})
        try:
            _write_summary(summary_path, report)
        except Exception as exc:
            print(f"Failed to save gate-worker receipt {summary_path}: {exc}", file=sys.stderr)
        _ = code_root
    print(json.dumps({"status": report["status"], "summary": str(summary_path),
                      "arm": arm, "candidate_count": len(report["candidates"]),
                      "ours_claimed": report.get("ours_gate_claimed", False)},
                     ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
