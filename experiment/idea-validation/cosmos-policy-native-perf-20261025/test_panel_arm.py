from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_arm
import panel_native
import perf_native


def _protocol():
    return {
        "protocol": "cosmos-policy-vqwam-v1",
        "mode": "libero_direct_policy",
        "arms": list(perf_native.ARMS),
        "inference": {"denoising_steps": 5},
        "performance": {"case_ids": list(range(3000, 3040)),
                        "seed_indices": [0, 1], "queries": 80,
                        "blocks": 3, "repetitions": 3, "warmup_queries": 3},
    }


def _native_protocol(status="pending"):
    return {
        "protocol": perf_native.FORMAT,
        "source_protocol": "cosmos-policy-vqwam-v1",
        "arms": list(perf_native.ARMS),
        "selection_status": status,
        "baseline_fidelity_policy": "strict",
        "selected_variants": {"bf16": "current", "spinquant_w4a4": "current",
                              "quarot_w4a4": "current", "svdquant_w4a4": "current",
                              "viditq_w4a4": "current", "ours_vqwam": "optimized"},
        "panel": {"case_ids_start": 3000, "case_ids_end_inclusive": 3039,
                  "seed_indices": [0], "queries": 40, "blocks": 1,
                  "repetitions": 1, "batch_size": 1, "nfe": 5},
        "hardware": {"gpu": "NVIDIA A100-SXM4-40GB", "capability": [8, 0],
                     "torch": "2.7.0+cu128", "cuda_runtime": "12.8"},
    }


def _legacy_summary(scope, hardware=None, binding=None):
    hardware = hardware or {
        "gpu_uuid": "GPU-old", "gpu_name": "NVIDIA A100-SXM4-40GB",
        "total_memory_bytes": 42_406_903_808, "capability": [8, 0],
        "torch_version": "2.7.0+cu128", "cuda_version": "12.8",
        "hostname": "compute-old", "pbs_job_id": "old.pbs"}
    binding = binding or {"execution": "bf16_reference",
                          "execution_scope": "decoded_bf16_numerical_comparison",
                          "native_int4_kernel_used": False,
                          "arm": "bf16", "requested_variant": "current"}
    coverage = {"queries": 80, "samples": 720, "expected_samples": 720,
                "reused_samples": 0, "blocks": 3, "repetitions": 3}
    provenance = {"sample_count": 720, "devices": [{
        "gpu_uuid": "GPU-sample", "hostnames": ["compute-sample"],
        "pbs_job_ids": ["sample.pbs"]}]}
    common = {"format": perf_native.FORMAT, "protocol": _protocol()["protocol"],
              "status": "complete", "complete": True, "arm": "bf16",
              "variant": "current", "scope": scope, "hardware": hardware,
              "backend_binding": binding, "coverage": coverage,
              "sample_hardware_provenance": provenance,
              "training_updates": 0, "closed_loop_episodes": 0}
    if scope == "sampler":
        common.update({"warmup_queries": [
            {"query": [3000, 0], "denoising_seconds": 0.1,
             "complete_endpoint_query_seconds": 0.2},
            {"query": [3000, 1], "denoising_seconds": 0.1,
             "complete_endpoint_query_seconds": 0.2},
            {"query": [3001, 0], "denoising_seconds": 0.1,
             "complete_endpoint_query_seconds": 0.2}],
            "timing_scope": "synchronized legacy benchmark.run_one sampler.forward wall interval",
            "denoising_wall_seconds": {"count": 720, "mean": 0.1, "median": 0.1,
                                        "p95": 0.1, "max": 0.1},
            "complete_endpoint_query_seconds": {"count": 720, "mean": 0.2,
                                                  "median": 0.2, "p95": 0.2, "max": 0.2},
            "peak_allocated_gb_decimal": {"count": 720, "mean": 5.0, "median": 5.0,
                                           "p95": 5.0, "max": 5.0}})
    else:
        common.update({"warmup_queries": [[3000, 0], [3000, 1], [3001, 0]],
            "warmup_forwards_per_query": 5,
            "timing_scope": "CUDA events around exactly five graph replays of frozen model.net.forward inputs; condition/sampler preparation excluded",
            "dit_replay_seconds": {"count": 720, "mean": 0.25, "median": 0.25,
                                   "p95": 0.25, "max": 0.25},
            "peak_allocated_gb_decimal": {"count": 720, "mean": 5.0, "median": 5.0,
                                           "p95": 5.0, "max": 5.0}})
    return common


def _write_legacy_scope(root, scope, summary):
    path = root / "results" / "native_performance" / scope / "current" / "bf16" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    return path


class PanelArmTests(unittest.TestCase):
    def test_pending_global_selection_allows_bf16_without_native_gates(self):
        events = []
        request = panel_arm.validate_arm_request(
            _native_protocol("pending"), "bf16", "current", Path("root"),
            gate_checker=lambda *args: events.append(args))
        self.assertFalse(request["gate_prechecked"])
        self.assertEqual(request["selection_status"], "pending")
        self.assertEqual(events, [])

    def test_pending_global_selection_allows_selected_native_arm_if_both_gates_pass(self):
        events = []
        with patch.object(perf_native, "baseline_fidelity_is_report_only",
                          return_value=False, create=True):
            request = panel_arm.validate_arm_request(
                _native_protocol("pending"), "spinquant_w4a4", "current", Path("root"),
                gate_checker=lambda *args: events.append(args))
        self.assertTrue(request["gate_prechecked"])
        self.assertEqual(request["fidelity_policy"], "strict")
        self.assertEqual(request["selection_status"], "pending")
        self.assertEqual(events, [(Path("root"), "spinquant_w4a4", "current", "sampler")])

    def test_report_only_baseline_calls_policy_checker_but_is_not_gate_prechecked(self):
        native_protocol = _native_protocol("pending")
        native_protocol["baseline_fidelity_policy"] = "report_only"
        events = []
        with patch.object(perf_native, "baseline_fidelity_is_report_only",
                          return_value=True, create=True):
            request = panel_arm.validate_arm_request(
                native_protocol, "spinquant_w4a4", "current", Path("root"),
                gate_checker=lambda *args: events.append(args))
        self.assertFalse(request["gate_prechecked"])
        self.assertEqual(request["fidelity_policy"], "report_only")
        self.assertFalse(request["fidelity_failure_blocks_panel"])
        self.assertIn("observational", request["fidelity_policy_note"])
        self.assertEqual(events, [(Path("root"), "spinquant_w4a4", "current", "sampler")])

    def test_ours_remains_strict_under_report_only_baseline_policy(self):
        native_protocol = _native_protocol("pending")
        native_protocol["baseline_fidelity_policy"] = "report_only"
        events = []
        with patch.object(perf_native, "baseline_fidelity_is_report_only",
                          return_value=False, create=True):
            request = panel_arm.validate_arm_request(
                native_protocol, "ours_vqwam", "optimized", Path("root"),
                gate_checker=lambda *args: events.append(args))
        self.assertTrue(request["gate_prechecked"])
        self.assertEqual(request["fidelity_policy"], "strict")
        self.assertTrue(request["fidelity_failure_blocks_panel"])
        self.assertEqual(events, [(Path("root"), "ours_vqwam", "optimized", "sampler")])

    def test_unselected_variant_or_failed_gates_reject_before_model_load(self):
        with self.assertRaisesRegex(RuntimeError, "not the selected variant"):
            panel_arm.validate_arm_request(_native_protocol(), "svdquant_w4a4",
                                           "native_fused", Path("root"), lambda *args: None)
        with patch.object(perf_native, "baseline_fidelity_is_report_only",
                          return_value=False, create=True):
            with self.assertRaisesRegex(RuntimeError, "full80 gates"):
                panel_arm.validate_arm_request(
                    _native_protocol(), "spinquant_w4a4", "current", Path("root"),
                    lambda *args: (_ for _ in ()).throw(RuntimeError("both full80 gates must pass")))

    def test_coverage_accepts_40_once_design_and_rejects_legacy_720(self):
        passing = {"complete": True, "performance_design": perf_native.PERFORMANCE_DESIGN,
                   "coverage": {"queries": 40, "samples": 40,
                    "expected_samples": 40, "blocks": 1, "repetitions": 1,
                    }}
        self.assertTrue(panel_arm.scope_coverage_is_full(passing))
        legacy = json.loads(json.dumps(passing))
        legacy["coverage"].update({"samples": 720, "expected_samples": 720,
                                   "blocks": 3, "repetitions": 3})
        self.assertFalse(panel_arm.scope_coverage_is_full(legacy))
        for field, value in (("samples", 4), ("queries", 4), ("expected_samples", 4),
                             ("blocks", 3), ("repetitions", 3)):
            partial = json.loads(json.dumps(passing))
            partial["coverage"][field] = value
            self.assertFalse(panel_arm.scope_coverage_is_full(partial))

    def test_legacy_scope_reuse_requires_complete_720_identity_and_finite_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid = _legacy_summary("sampler")
            _write_legacy_scope(root, "sampler", valid)
            reused, reason = panel_arm._read_legacy_scope(
                root, _protocol(), _native_protocol(), "bf16", "current", "sampler")
            self.assertIsNone(reason)
            self.assertEqual(reused["coverage"]["samples"], 720)
            partial = dict(valid, coverage=dict(valid["coverage"], samples=719))
            _write_legacy_scope(root, "sampler", partial)
            rejected, reason = panel_arm._read_legacy_scope(
                root, _protocol(), _native_protocol(), "bf16", "current", "sampler")
            self.assertIsNone(rejected)
            self.assertIn("720-sample", reason)
            bad_metric = dict(valid, denoising_wall_seconds={
                "count": 720, "mean": float("nan"), "median": 0.1, "p95": 0.1, "max": 0.1})
            bad_path = _write_legacy_scope(root, "sampler", valid)
            bad_path.write_text(json.dumps(bad_metric, allow_nan=True), encoding="utf-8")
            rejected, reason = panel_arm._read_legacy_scope(
                root, _protocol(), _native_protocol(), "bf16", "current", "sampler")
            self.assertIsNone(rejected)
            self.assertIn("non-finite", reason)

    def test_both_complete_legacy_scopes_finish_without_runtime_import(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old_root, root, out = base / "old", base / "new", base / "new" / "logs"
            old_root.mkdir()
            root.mkdir()
            (old_root / "protocol.json").write_text(json.dumps(_protocol()), encoding="utf-8")
            (root / "native_protocol.json").write_text(json.dumps(_native_protocol()), encoding="utf-8")
            _write_legacy_scope(root, "sampler", _legacy_summary("sampler"))
            _write_legacy_scope(root, "dit_replay", _legacy_summary("dit_replay"))
            with patch.object(perf_native, "_prepare_code_imports",
                              side_effect=AssertionError("runtime import must be skipped")):
                result = panel_arm.main(["--arm", "bf16", "--variant", "current",
                                         "--old-root", str(old_root), "--root", str(root),
                                         "--out", str(out)])
            receipt_path = root / "results/native_performance_n40/panel_arm/bf16/current/summary.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertTrue(receipt["complete"])
            self.assertFalse(receipt["runtime_loaded"])
            self.assertEqual(receipt["completed_samples_total"], 1440)
            self.assertEqual([row["samples"] for row in receipt["scope_results"]], [720, 720])
            self.assertTrue(all(row["reused"] for row in receipt["scope_results"]))
            self.assertTrue(all(Path(row["summary_path"]).is_relative_to(root / "results" / "native_performance")
                                for row in receipt["scope_results"]))

    def test_mixed_legacy_sampler_and_new_dit_receipt_reports_actual_760(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old_root, root, out = base / "old", base / "new", base / "new" / "logs"
            old_root.mkdir()
            root.mkdir()
            (old_root / "protocol.json").write_text(json.dumps(_protocol()), encoding="utf-8")
            (root / "native_protocol.json").write_text(json.dumps(_native_protocol()), encoding="utf-8")
            _write_legacy_scope(root, "sampler", _legacy_summary("sampler"))

            class Device:
                type = "cuda"

                def __str__(self):
                    return "cuda:0"

            events = []
            runtime = SimpleNamespace(model=SimpleNamespace(
                parameters=lambda: iter([SimpleNamespace(device=Device())])))
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                empty_cache=lambda: events.append("empty_cache")))
            common = SimpleNamespace(
                torch_module=lambda: events.append("torch") or fake_torch,
                prepare_runtime=lambda *args: events.append("prepare_runtime") or runtime)
            hardware = {"gpu_uuid": "GPU-new", "gpu_name": "NVIDIA A100-SXM4-40GB",
                        "total_memory_bytes": 42_406_903_808, "capability": [8, 0],
                        "torch_version": "2.7.0+cu128", "cuda_version": "12.8",
                        "hostname": "compute-new", "pbs_job_id": "new.pbs"}
            binding = _legacy_summary("sampler")["backend_binding"]
            new_dit = {"status": "complete", "complete": True,
                       "performance_design": perf_native.PERFORMANCE_DESIGN,
                       "protocol": _protocol()["protocol"], "arm": "bf16",
                       "variant": "current", "scope": "dit_replay",
                       "coverage": {"queries": 40, "samples": 40,
                                    "expected_samples": 40, "blocks": 1, "repetitions": 1},
                       "hardware": hardware, "backend_binding": binding}

            class Trace:
                def __enter__(self):
                    events.append("trace_enter")

                def __exit__(self, *_exc):
                    events.append("trace_exit")

            with patch.dict(sys.modules, {"common": common}), \
                 patch.object(perf_native, "_prepare_code_imports", return_value=Path("code")), \
                 patch.object(perf_native, "_allocation_guard", side_effect=lambda _common: events.append("guard")), \
                 patch.object(perf_native, "_gpu_receipt", return_value=hardware), \
                 patch.object(perf_native, "_offload_non_dit", side_effect=lambda *args: events.append("offload") or {"ok": True}), \
                 patch.object(perf_native, "_read_capture_manifest", side_effect=lambda *args: events.append("capture") or {}), \
                 patch.object(perf_native, "_deadline", return_value=999999.0), \
                 patch.object(perf_native, "_restore_to_bf16", side_effect=lambda *args: events.append("final_bf16")), \
                 patch.object(panel_native, "_clear_known_arm_caches", side_effect=lambda _runtime: events.append("clear") or []), \
                 patch.object(panel_native, "_run_scope", side_effect=lambda *args: events.append("dit_replay") or (new_dit, [])), \
                 patch.object(panel_arm, "startup_trace", side_effect=lambda _out: Trace()), \
                 patch.object(panel_arm.time, "monotonic", return_value=0.0):
                result = panel_arm.main(["--arm", "bf16", "--variant", "current",
                                         "--old-root", str(old_root), "--root", str(root),
                                         "--out", str(out)])
            receipt = json.loads((root / "results/native_performance_n40/panel_arm/bf16/current/summary.json").read_text())
            self.assertEqual(result, 0)
            self.assertEqual(events.count("prepare_runtime"), 1)
            self.assertLess(events.index("offload"), events.index("dit_replay"))
            self.assertEqual(receipt["completed_samples_total"], 760)
            self.assertEqual([row["samples"] for row in receipt["scope_results"]], [720, 40])
            self.assertEqual(receipt["scope_results"][0]["source_design"], panel_arm.LEGACY_DESIGN)
            self.assertFalse(receipt["scope_results"][1]["reused"])

    def test_receipt_path_uses_new_n40_namespace(self):
        root = Path("root")
        performance_root = root / "results" / "native_performance_n40"
        with patch.object(perf_native, "performance_root", return_value=performance_root) as helper:
            self.assertEqual(panel_arm._receipt_path(root, "bf16", "current"),
                             performance_root / "panel_arm" / "bf16" / "current" / "summary.json")
            helper.assert_called_once_with(root)

    def test_single_arm_main_loads_one_runtime_and_runs_sampler_before_dit(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old_root, root, out = base / "old", base / "new", base / "new" / "logs"
            old_root.mkdir()
            root.mkdir()
            (old_root / "protocol.json").write_text(json.dumps(_protocol()), encoding="utf-8")
            (root / "native_protocol.json").write_text(json.dumps(_native_protocol()), encoding="utf-8")

            class Device:
                type = "cuda"

                def __str__(self):
                    return "cuda:0"

            events = []
            runtime = SimpleNamespace(model=SimpleNamespace(
                parameters=lambda: iter([SimpleNamespace(device=Device())])))
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                empty_cache=lambda: events.append("empty_cache")))
            common = SimpleNamespace(
                torch_module=lambda: events.append("torch") or fake_torch,
                prepare_runtime=lambda *args: events.append("prepare_runtime") or runtime)
            hardware = {"gpu_uuid": "GPU-test", "gpu_name": "NVIDIA A100-SXM4-40GB",
                        "total_memory_bytes": 40_000_000_000, "capability": [8, 0],
                        "torch_version": "2.7.0+cu128", "cuda_version": "12.8",
                        "hostname": "compute", "pbs_job_id": "test.pbs"}
            coverage = {"queries": 40, "samples": 40, "expected_samples": 40,
                        "blocks": 1, "repetitions": 1}
            summaries = []
            runtime_ids = []

            def run_scope(_common, _runtime, _protocol, _old, _root, _out,
                          _arm, _variant, scope, *_rest):
                events.append(scope)
                runtime_ids.append(id(_runtime))
                summary = {"complete": True, "status": "complete",
                           "performance_design": perf_native.PERFORMANCE_DESIGN,
                           "coverage": coverage,
                           "hardware": hardware}
                summaries.append(summary)
                return summary, ["restored"]

            class Trace:
                def __enter__(self):
                    events.append("trace_enter")

                def __exit__(self, *_exc):
                    events.append("trace_exit")

            fake_common_module = common
            env = {"PBS_JOBID": "test.pbs"}
            with patch.dict(os.environ, env, clear=False), \
                 patch.dict(sys.modules, {"common": fake_common_module}), \
                 patch.object(perf_native, "_prepare_code_imports", return_value=Path("code")), \
                 patch.object(perf_native, "_allocation_guard", side_effect=lambda _common: events.append("guard")), \
                 patch.object(perf_native, "_gpu_receipt", return_value=hardware), \
                 patch.object(perf_native, "_offload_non_dit", side_effect=lambda *args: events.append("offload") or {"ok": True}), \
                 patch.object(perf_native, "_read_capture_manifest", side_effect=lambda *args: events.append("capture_manifest") or {}), \
                 patch.object(perf_native, "_restore_to_bf16", side_effect=lambda *args: events.append("final_bf16")), \
                 patch.object(panel_native, "_clear_known_arm_caches", side_effect=lambda _runtime: events.append("clear_caches") or []), \
                 patch.object(panel_native, "_run_scope", side_effect=run_scope), \
                 patch.object(panel_native.time, "monotonic", side_effect=[0.0, 0.0, 0.0, 0.0, 0.0]), \
                 patch.object(perf_native, "performance_root",
                              side_effect=lambda value: value / "results" / "native_performance_n40"), \
                 patch.object(panel_arm, "startup_trace", side_effect=lambda _out: Trace()):
                result = panel_arm.main(["--arm", "bf16", "--variant", "current",
                                         "--old-root", str(old_root), "--root", str(root),
                                         "--out", str(out)])

            self.assertEqual(result, 0)
            self.assertEqual(events.count("prepare_runtime"), 1)
            self.assertLess(events.index("guard"), events.index("trace_enter"))
            self.assertLess(events.index("trace_enter"), events.index("torch"))
            self.assertLess(events.index("torch"), events.index("prepare_runtime"))
            self.assertLess(events.index("prepare_runtime"), events.index("trace_exit"))
            self.assertLess(events.index("sampler"), events.index("offload"))
            self.assertLess(events.index("offload"), events.index("dit_replay"))
            self.assertEqual(runtime_ids[0], runtime_ids[1])
            receipt_path = root / "results/native_performance_n40/panel_arm/bf16/current/summary.json"
            self.assertTrue(receipt_path.is_file())
            receipt = json.loads(receipt_path.read_text())
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["performance_design"], "40ctx_once_v3")
            self.assertEqual(receipt["expected_coverage"]["queries"], 40)
            self.assertEqual(receipt["expected_coverage"]["samples_per_scope"], 40)
            self.assertEqual(receipt["expected_coverage"]["blocks"], 1)
            self.assertEqual(receipt["expected_coverage"]["repetitions"], 1)
            self.assertEqual(receipt["completed_samples_total"], 80)


if __name__ == "__main__":
    unittest.main()
