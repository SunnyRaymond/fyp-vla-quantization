import ast
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gates_worker as worker
import perf_native


class GatesWorkerTests(unittest.TestCase):
    def test_vidit_preflight_requires_the_general_scale_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'results/preparation/native_viditq_general_build.json'
            path.parent.mkdir(parents=True)
            binary = root / 'viditq_atom_per_channel_native.so'
            binary.write_text('test path only; never imported')
            path.write_text(json.dumps({'status': 'complete', 'pbs_job_id': 'build-producer',
                'source_commit': 'pinned-source', 'module': str(binary)}))
            self.assertEqual(worker._candidate_preflight(root, 'viditq_w4a4', 'current')['status'], 'ready')
            binary.unlink()
            self.assertEqual(worker._candidate_preflight(root, 'viditq_w4a4', 'current')['status'], 'deferred')

    def test_arm_aliases_and_candidate_sets_are_frozen(self):
        self.assertEqual(worker.normalize_arm("spinquant"), "spinquant_w4a4")
        self.assertEqual(worker.candidate_variants("spinquant"), ("current", "native_grouped"))
        self.assertEqual(worker.candidate_variants("quarot_w4a4"), ("current", "native_grouped"))
        self.assertEqual(worker.candidate_variants("svdquant"), ("current", "native_fused"))
        self.assertEqual(worker.candidate_variants("viditq"), ("current",))
        self.assertEqual(worker.normalize_arm("ours"), "ours_vqwam")
        self.assertEqual(worker.candidate_variants("ours"), ("optimized",))
        self.assertEqual(worker.candidate_variants("ours_vqwam"), ("optimized",))

    def test_dedicated_ours_worker_runs_only_optimized_across_both_full_scopes(self):
        calls = []
        rows = worker.evaluate_candidates(
            worker.candidate_variants("ours_vqwam"), worker.GATE_SCOPES,
            lambda variant, scope: (calls.append((variant, scope)) or
                {"scope": scope, "status": "passed", "full80_passed": True}))
        self.assertEqual(calls, [("optimized", "sampler"), ("optimized", "dit_replay")])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant"], "optimized")
        self.assertEqual(rows[0]["status"], "passed")

    def test_candidate_failure_does_not_skip_other_variant_or_scope(self):
        calls = []

        def run_scope(variant, scope):
            calls.append((variant, scope))
            if variant == "current" and scope == "sampler":
                raise RuntimeError("expected candidate failure")
            return {"scope": scope, "status": "passed", "full80_passed": True}

        result = worker.evaluate_candidates(
            ("current", "native_grouped"), worker.GATE_SCOPES, run_scope)
        self.assertEqual(calls, [(variant, scope) for variant in ("current", "native_grouped")
                                for scope in worker.GATE_SCOPES])
        self.assertEqual(result[0]["status"], "failed")
        self.assertEqual(result[1]["status"], "passed")

    def test_preflight_deferred_candidate_does_not_skip_next(self):
        calls = []
        result = worker.evaluate_candidates(
            ("native_fused", "current"), worker.GATE_SCOPES,
            lambda variant, scope: (calls.append((variant, scope)) or
                {"scope": scope, "status": "passed", "full80_passed": True}),
            preflight=lambda variant: ({"status": "deferred", "reason": "no receipt"}
                                       if variant == "native_fused" else {"status": "ready"}))
        self.assertEqual(result[0]["status"], "deferred")
        self.assertEqual(result[1]["status"], "passed")
        self.assertEqual(calls, [("current", scope) for scope in worker.GATE_SCOPES])

    def test_candidate_requires_both_full80_scopes_not_representative4(self):
        rows = [{"scope": "sampler", "status": "passed", "full80_passed": True},
                {"scope": "dit_replay", "status": "passed", "full80_passed": False,
                 "representative4_passed": True}]
        self.assertFalse(worker.candidate_passed(rows))
        rows[1]["full80_passed"] = True
        self.assertTrue(worker.candidate_passed(rows))
        self.assertEqual(worker.GATE_SCOPES, ("sampler", "dit_replay"))

    def test_gate_scope_always_runs_panel_cleanup_and_preserves_full_protocol(self):
        protocol = {"performance": {"queries": 80, "blocks": 3, "repetitions": 3}}
        calls = []
        fake_panel = type("Panel", (), {"_finish_arm": staticmethod(
            lambda *args: calls.append("cleanup"))})

        def fake_gate(common, runtime, passed_protocol, old_root, root, out,
                      arm, variant, scope, torch, device, hardware, binding, deadline):
            calls.append((arm, variant, scope, passed_protocol["performance"]["queries"]))
            raise RuntimeError("gate failed")

        with patch.object(perf_native, "_gate_panel", side_effect=fake_gate):
            result = worker._run_gate_scope(None, fake_panel, object(), protocol,
                Path("old"), Path("root"), Path("out"), "spinquant_w4a4", "current",
                "sampler", object(), object(), {}, 0.0)
        self.assertEqual(result["status"], "error")
        self.assertEqual(calls, [("spinquant_w4a4", "current", "sampler", 80), "cleanup"])

    def test_claim_is_atomic_and_only_one_worker_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "ours_gate_merged.json"
            marker.write_text("{}", encoding="utf-8")
            results = []
            barrier = threading.Barrier(8)

            def claim():
                barrier.wait()
                results.append(worker.acquire_ours_claim(root, marker, "12345.pbs101"))

            threads = [threading.Thread(target=claim) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sum(row["claimed"] for row in results), 1)
            claim_dir = root / "artifacts" / "12345.pbs101" / "ours_gate_claim"
            self.assertTrue(claim_dir.is_dir())

    def test_claim_is_not_taken_without_merged_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = worker.acquire_ours_claim(root, root / "ours_gate_merged.json", "job-1")
            self.assertEqual(result["status"], "not_requested")
            self.assertFalse((root / "artifacts").exists())

    def test_disabled_merge_marker_never_claims_or_merges_ours(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "ours_gate_merged.json"
            marker.write_text(json.dumps({"enabled": False, "status": "complete"}),
                              encoding="utf-8")
            result = worker.acquire_ours_claim(root, marker, "job-1")
            self.assertEqual(result["status"], "disabled")
            self.assertFalse(result["claimed"])
            self.assertFalse((root / "artifacts").exists())

    def test_terminal_stage_order_runs_ours_before_svd_offload_diagnostic(self):
        calls = []
        ours, diagnostic = worker.run_post_gate_stages(
            lambda: calls.append("ours") or "ours-result",
            lambda: calls.append("svd-diagnostic") or "diagnostic-result")
        self.assertEqual(calls, ["ours", "svd-diagnostic"])
        self.assertEqual((ours, diagnostic), ("ours-result", "diagnostic-result"))

    def test_dedicated_ours_worker_skips_postclaim_ours_and_svd_diagnostic(self):
        calls = []
        ours, diagnostic = worker.run_post_gate_stages(
            lambda: calls.append("duplicate-ours") or "ours-result",
            lambda: calls.append("svd-diagnostic") or "diagnostic-result",
            arm="ours_vqwam")
        self.assertEqual(calls, [])
        self.assertEqual(ours["status"], "not_applicable")
        self.assertEqual(diagnostic["status"], "not_applicable")

    def test_svd_candidate_is_deferred_without_build_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            check = worker._candidate_preflight(Path(temp), "svdquant_w4a4", "current")
            self.assertEqual(check["status"], "deferred")

    def test_svd_probe_receipt_must_match_build_and_both_stream_paths(self):
        module = SimpleNamespace(NUNCHAKU_COMMIT="commit", MODULE_NAME="mod",
                                 M_CASES=(129, 1764), DEEPCOMPRESSOR_COMMIT="pack-commit")
        build = {"pbs_job_id": "build-1"}
        arm = {"module": "/build/mod.so"}
        phases = {"default_current_stream": {"passed": True},
                  "non_default_current_stream": {"passed": True},
                  "cuda_graph": {"capture_passed": True, "replay_passed": True}}
        receipt = {"status": "complete", "probe": "nunchaku_official_svdquant_rank16_lora_fusion",
            "producer": {"build_pbs_job_id": "build-1"},
            "source": {"module": "/build/mod.so", "module_name": "mod",
                       "nunchaku_commit": "commit",
                       "deepcompressor_packing_commit": "pack-commit"},
            "cases": [{"true_m": value, "stream_tests": phases} for value in (129, 1764)],
            "policy_forward_calls": 0, "training_steps": 0, "closed_loop_episodes": 0}
        self.assertTrue(worker._probe_receipt_matches(receipt, build, arm, module))
        receipt["producer"]["build_pbs_job_id"] = "older-build"
        self.assertFalse(worker._probe_receipt_matches(receipt, build, arm, module))

    def test_worker_has_one_runtime_prepare_call_and_guards_before_startup_load(self):
        tree = ast.parse((HERE / "gates_worker.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "prepare_runtime"]
        self.assertEqual(len(calls), 1)
        source = (HERE / "gates_worker.py").read_text(encoding="utf-8")
        self.assertLess(source.index("perf._allocation_guard(common)"),
                        source.index("with startup_trace(out):"))
        self.assertIn('"samples_per_candidate_scope": 720', source)

    def test_parser_accepts_full_and_short_arm_names(self):
        parser = worker.build_parser()
        args = parser.parse_args(["--arm", "quarot", "--old-root", "old",
                                  "--root", "root", "--out", "root/out"])
        self.assertEqual(worker.normalize_arm(args.arm), "quarot_w4a4")
        for arm in ("ours", "ours_vqwam"):
            with self.subTest(arm=arm):
                args = parser.parse_args(["--arm", arm, "--old-root", "old",
                                          "--root", "root", "--out", "root/out"])
                self.assertEqual(worker.normalize_arm(args.arm), "ours_vqwam")


if __name__ == "__main__":
    unittest.main()
