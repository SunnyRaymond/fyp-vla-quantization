import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_native
import perf_native


def _ready_protocol():
    return {
        "selection_status": "ready_for_formal_panel",
        "arms": list(perf_native.ARMS),
        "selected_variants": {
            "bf16": "current",
            "spinquant_w4a4": "current",
            "quarot_w4a4": "current",
            "svdquant_w4a4": "current",
            "viditq_w4a4": "current",
            "ours_vqwam": "optimized",
        },
    }


class _Cuda:
    def __init__(self, events):
        self.events = events

    def empty_cache(self):
        self.events.append("empty_cache")


class PanelNativeDriverTests(unittest.TestCase):
    def test_default_panel_does_not_require_a_specific_physical_gpu(self):
        args = panel_native.build_parser().parse_args(
            ['--old-root', 'old', '--root', 'root', '--out', 'out'])
        self.assertIsNone(args.require_gpu_uuid)

    def test_selection_must_be_ready_and_cover_all_frozen_arms(self):
        spec = _ready_protocol()
        self.assertEqual(list(panel_native.selected_variants(spec)), list(perf_native.ARMS))
        pending = dict(spec, selection_status="pending")
        with self.assertRaisesRegex(RuntimeError, "ready_for_formal_panel"):
            panel_native.selected_variants(pending)
        legacy_ready = dict(spec, selection_status="ready")
        with self.assertRaisesRegex(RuntimeError, "ready_for_formal_panel"):
            panel_native.selected_variants(legacy_ready)
        missing = dict(spec, selected_variants={"bf16": "current"})
        with self.assertRaisesRegex(RuntimeError, "all six"):
            panel_native.selected_variants(missing)

    def test_native_and_source_protocols_stay_bound_to_the_frozen_panel(self):
        native = dict(_ready_protocol(), protocol=perf_native.FORMAT,
                      source_protocol="source-v1", panel={
                          "case_ids_start": 3000, "case_ids_end_inclusive": 3039,
                          "seed_indices": [0], "queries": 40, "blocks": 1,
                          "repetitions": 1, "batch_size": 1, "nfe": 5})
        source = {"protocol": "source-v1", "mode": "libero_direct_policy",
                  "inference": {"denoising_steps": 5},
                  "performance": {"case_ids": list(range(3000, 3040)),
                                  "seed_indices": [0, 1], "queries": 80,
                                  "blocks": 3, "repetitions": 3}}
        panel_native.validate_panel_spec(native, source)
        native["panel"]["blocks"] = 3
        with self.assertRaisesRegex(RuntimeError, "40-context"):
            panel_native.validate_panel_spec(native, source)
        native["panel"]["blocks"] = 1
        source["performance"]["case_ids"] = list(range(2999, 3039))
        with self.assertRaisesRegex(RuntimeError, "source protocol cohort"):
            panel_native.validate_panel_spec(native, source)
        native["panel"]["queries"] = 4
        with self.assertRaisesRegex(RuntimeError, "40-context"):
            panel_native.validate_panel_spec(native, source)

    def test_formal_plan_keeps_both_scopes_and_one_sample_per_active_context(self):
        variants = panel_native.selected_variants(_ready_protocol())
        plan = panel_native.formal_plan(variants)
        self.assertEqual(len(plan), 12)
        self.assertEqual([row["scope"] for row in plan[:6]], ["sampler"] * 6)
        self.assertEqual([row["scope"] for row in plan[6:]], ["dit_replay"] * 6)
        self.assertEqual([row["arm"] for row in plan[:6]], list(perf_native.ARMS))
        self.assertTrue(all(row["expected_samples"] == 40 for row in plan))
        coverage = panel_native.expected_formal_coverage()
        self.assertEqual(coverage["design"], "40ctx_once_v3")
        self.assertEqual(coverage["blocks"], 1)
        self.assertEqual(coverage["repetitions"], 1)
        self.assertEqual(coverage["samples_per_arm_scope"], 40)
        self.assertEqual(coverage["expected_samples_total"], 480)
        self.assertNotEqual(coverage["expected_samples_total"], 4320)

    def test_scope_coverage_accepts_n40_and_rejects_old_720_receipt(self):
        passing = {"complete": True, "performance_design": "40ctx_once_v3",
                   "coverage": {"queries": 40, "samples": 40,
                                "expected_samples": 40, "blocks": 1,
                                "repetitions": 1}}
        self.assertTrue(panel_native.scope_coverage_is_active(passing))
        legacy = {"complete": True, "performance_design": "legacy_720",
                  "coverage": {"queries": 80, "samples": 720,
                               "expected_samples": 720, "blocks": 3,
                               "repetitions": 3}}
        self.assertFalse(panel_native.scope_coverage_is_active(legacy))

    def test_hardware_contract_binds_a100_torch_cuda_and_uuid(self):
        hardware = {"gpu_name": "NVIDIA A100-SXM4-40GB", "capability": [8, 0],
                    "torch_version": "2.7.0+cu128", "cuda_version": "12.8",
                    "gpu_uuid": "GPU-test"}
        panel_native.validate_hardware_contract({"hardware": {
            "gpu": hardware["gpu_name"], "capability": [8, 0],
            "torch": "2.7.0+cu128", "cuda_runtime": "12.8"}}, hardware)
        hardware["gpu_uuid"] = None
        with self.assertRaisesRegex(RuntimeError, "physical GPU UUID"):
            panel_native.validate_hardware_contract({"hardware": {}}, hardware)

    def test_cleanup_restores_bf16_then_drops_known_caches_not_cpu_shadow(self):
        events = []
        target = SimpleNamespace()
        target.__dict__.update({"_cosmos_spin_state": object(),
                                "_native_grouped_context": object(),
                                "_buffers": {"_native_rotation_scales": object()},
                                "weight": SimpleNamespace(dtype="torch.bfloat16")})
        controller = SimpleNamespace(_device_rotation_cache=object(),
                                     _base_weights_cpu={"kept": "cpu-shadow"})
        runtime = SimpleNamespace(controller=controller, target_linears={"target": target})
        runtime._ours_reference_holds = object()
        torch = SimpleNamespace(cuda=_Cuda(events))
        with patch.object(panel_native.perf_native, "_restore_to_bf16",
                          side_effect=lambda *args: events.append("bf16")), \
             patch.object(panel_native.gc, "collect",
                          side_effect=lambda: events.append("gc")):
            removed = panel_native._finish_arm(
                runtime, {}, Path("old-root"), lambda: events.append("restore-hook"), torch)
        self.assertEqual(events, ["restore-hook", "bf16", "gc", "empty_cache"])
        self.assertEqual(controller._base_weights_cpu, {"kept": "cpu-shadow"})
        self.assertNotIn("_ours_reference_holds", runtime.__dict__)
        self.assertNotIn("_device_rotation_cache", controller.__dict__)
        self.assertNotIn("_cosmos_spin_state", target.__dict__)
        self.assertNotIn("_native_grouped_context", target.__dict__)
        self.assertNotIn("_native_rotation_scales", target._buffers)
        self.assertIn("runtime._ours_reference_holds", removed)

    def test_cleanup_releases_stale_graph_and_packed_engine_references(self):
        events = []
        target = SimpleNamespace(weight=SimpleNamespace(dtype="torch.bfloat16"))
        graph = SimpleNamespace(records={"capture": object()},
                                restore=lambda: events.append("graph_restore"))
        banks = SimpleNamespace(graph_workspaces={"nfe": object()},
                                default_workspace="default", workspace="temporary")
        engine = SimpleNamespace(graphs=graph, banks=banks)
        runtime = SimpleNamespace(target_linears={"target": target},
                                  ours_vqwam_engine=engine,
                                  optimized_vq_graphs=graph,
                                  restore_ours_vqwam=lambda: None,
                                  restore_optimized_engine=lambda: None)
        removed = panel_native._clear_known_arm_caches(runtime)
        self.assertEqual(events, ["graph_restore"])
        self.assertEqual(graph.records, {})
        self.assertEqual(banks.graph_workspaces, {})
        self.assertEqual(banks.workspace, "default")
        self.assertFalse(hasattr(runtime, "ours_vqwam_engine"))
        self.assertFalse(hasattr(runtime, "optimized_vq_graphs"))
        self.assertFalse(hasattr(runtime, "restore_ours_vqwam"))
        self.assertIn("runtime.ours_vqwam_engine", removed)

    def test_structural_device_rotation_cache_is_cleared_but_attribute_remains(self):
        cache = {"rotation-key": object()}
        controller = SimpleNamespace(_device_rotation_cache=cache,
                                     _base_weights_cpu={"kept": "cpu-shadow"})
        module = SimpleNamespace(weight=SimpleNamespace(dtype="torch.bfloat16"))
        runtime = SimpleNamespace(controller=controller, target_linears={"target": module})
        removed = panel_native._clear_known_arm_caches(runtime)
        self.assertIs(controller._device_rotation_cache, cache)
        self.assertEqual(cache, {})
        self.assertIn("controller._device_rotation_cache.clear", removed)
        self.assertEqual(controller._base_weights_cpu, {"kept": "cpu-shadow"})

    def test_scope_worker_gates_then_installs_measures_and_restores(self):
        events = []
        runtime = SimpleNamespace(controller=SimpleNamespace(), target_linears={})
        hardware = {"gpu_uuid": "GPU-test"}
        torch = SimpleNamespace(cuda=_Cuda(events))
        summary = {"complete": True, "status": "complete",
                   "performance_design": perf_native.PERFORMANCE_DESIGN,
                   "coverage": {"queries": 40, "samples": 40, "expected_samples": 40,
                                "blocks": 1, "repetitions": 1},
                   "hardware": hardware}

        def install(*args, **kwargs):
            events.append("install")
            return {"native_int4_kernel_used": True}, lambda: events.append("restore_hook")

        with patch.object(panel_native.perf_native, "_require_gate",
                          side_effect=lambda *args: events.append("gate")), \
             patch.object(panel_native.perf_native, "_install_native", side_effect=install), \
             patch.object(panel_native.perf_native, "_binding",
                          side_effect=lambda *args: events.append("binding") or {}), \
             patch.object(panel_native.perf_native, "assert_native_gate_binding",
                          side_effect=lambda *args: events.append("assert_binding")), \
             patch.object(panel_native.perf_native, "_packed_storage_receipt",
                          side_effect=lambda *args: events.append("storage") or {}), \
             patch.object(panel_native.perf_native, "_run_sampler_benchmark",
                          side_effect=lambda *args: events.append("measure") or summary), \
             patch.object(panel_native.perf_native, "_restore_to_bf16",
                          side_effect=lambda *args: events.append("bf16_restore")), \
             patch.object(panel_native.gc, "collect",
                          side_effect=lambda: events.append("gc")):
            result, _ = panel_native._run_scope(
                object(), runtime, {}, Path("old"), Path("root"), Path("out"),
                "spinquant_w4a4", "current", "sampler", torch, "cuda:0", hardware,
                time.monotonic() + 60, Path("root"))
        self.assertIs(result, summary)
        self.assertEqual(events, ["gate", "install", "binding", "assert_binding",
                                  "storage", "measure", "restore_hook", "bf16_restore",
                                  "gc", "empty_cache"])

    def test_driver_summary_path_is_small_separate_receipt_location(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            performance_root = root / "results" / "native_performance_n40"
            with patch.object(perf_native, "performance_root",
                              return_value=performance_root) as path_helper:
                self.assertEqual(panel_native._summary_path(root),
                                 performance_root / "panel_driver" / "summary.json")
                path_helper.assert_called_once_with(root)
                record = panel_native._scope_record(
                    {"status": "complete", "complete": True,
                     "performance_design": perf_native.PERFORMANCE_DESIGN,
                     "coverage": {"samples": 40, "expected_samples": 40},
                     "hardware": {"gpu_uuid": "GPU-test"}},
                    root, "sampler", "bf16", "current", [])
                self.assertEqual(Path(record["summary_path"]),
                                 performance_root / "sampler" / "current" / "bf16" / "summary.json")
                self.assertNotIn("results/native_performance/", record["summary_path"])


if __name__ == "__main__":
    unittest.main()
