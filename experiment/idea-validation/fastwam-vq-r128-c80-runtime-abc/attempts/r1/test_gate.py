"""Small standard-library checks for runtime ABC orchestration gates."""
import ast
import json
from pathlib import Path
import re
import runpy
import unittest


HERE = Path(__file__).resolve().parent
finalize = runpy.run_path(str(HERE / "finalize.py"))
CANDIDATES = finalize["CANDIDATE_MODES"]
REFERENCES = finalize["REFERENCE_MODES"]


class GateTests(unittest.TestCase):
    def test_exact_and_tolerance_boundary_pass(self):
        parity_pass = finalize["parity_pass"]
        self.assertTrue(parity_pass(0.0, 1e-6))
        self.assertTrue(parity_pass(1e-6, 1e-6))
        self.assertFalse(parity_pass(1.000001e-6, 1e-6))
        self.assertFalse(parity_pass(float("nan"), 1e-6))

    def test_anchor_action_and_future_errors_are_both_gated(self):
        checks = finalize["mode_checks"]
        summary = {
            "deterministic_algorithms": True,
            "fill_uninitialized_memory": True,
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": True,
            "new_training_updates": 0,
            "closed_loop_episodes": 0,
            "compiled": False,
            "cuda_graph": False,
            "performance_accuracy": {"anchor_action_max_abs": 0.0},
        }
        validation = {"source_anchor_parity": {"future_max_abs": 1e-6}}
        self.assertTrue(checks(summary, validation, "runtime_a")["passed"])
        validation["source_anchor_parity"]["future_max_abs"] = 1.000001e-6
        result = checks(summary, validation, "runtime_a")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["anchor_future_max_abs_le_1e-6"])

    def test_cuda_graph_and_compile_regions_need_full_query_evidence(self):
        checks = finalize["mode_checks"]
        base = {
            "deterministic_algorithms": True,
            "fill_uninitialized_memory": True,
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": True,
            "new_training_updates": 0,
            "closed_loop_episodes": 0,
            "performance_accuracy": {
                "anchor_action_max_abs": 0.0,
                "anchor_future_max_abs": 0.0,
            },
        }
        regions_c = {
            "video_step": {"captures": 1, "replays": 800},
            "video_prefill": {"captures": 1, "replays": 80},
            "action_step": {"captures": 1, "replays": 800},
        }
        runtime_c = dict(base, compiled=False, cuda_graph=True, optimization={
            "after_primary": {"graph": {"kind": "cuda_graph", "regions": regions_c}}
        })
        self.assertTrue(checks(runtime_c, {"mode": "runtime_c"}, "runtime_c")["passed"])
        regions_c["action_step"]["replays"] = 799
        self.assertFalse(checks(runtime_c, {"mode": "runtime_c"}, "runtime_c")["passed"])

        regions_compile = {
            name: {"compile_calls": calls}
            for name, calls in (("video_step", 800), ("video_prefill", 80), ("action_step", 800))
        }
        runtime_compile = dict(base, compiled=True, cuda_graph=False, optimization={
            "after_primary": {"graph": {
                "kind": "torch_compile", "regions": regions_compile,
                "unique_compiled_graphs": 3,
            }}
        })
        self.assertTrue(checks(runtime_compile, {"mode": "runtime_compile"}, "runtime_compile")["passed"])
        runtime_compile["optimization"]["after_primary"]["graph"]["unique_compiled_graphs"] = 0
        self.assertFalse(checks(runtime_compile, {"mode": "runtime_compile"}, "runtime_compile")["passed"])

    def test_stage_order_and_compile_can_follow_c_operational_failure(self):
        allowed = finalize["stage_allowed"]
        self.assertTrue(allowed("runtime_a", {}))
        self.assertFalse(allowed("runtime_b", {}))
        self.assertFalse(allowed("runtime_c", {"runtime_a": {"passed": True}}))
        gates = {
            "runtime_a": {"passed": True},
            "runtime_b": {"passed": True},
            "runtime_c": {"passed": False, "status": "failed"},
        }
        self.assertTrue(allowed("runtime_c", gates))
        self.assertTrue(allowed("runtime_compile", gates))

    def test_completion_requires_all_four_measured_parity_passes(self):
        modes = {
            name: {
                "status": "cached" if name in REFERENCES else "measured",
                "queries": 80,
                "parity_vs_primary": {"passed": True},
                "validation_checks": {"passed": True},
                "performance": {"p50_ratio_vs_primary": 1000.0},
            }
            for name in REFERENCES + CANDIDATES
        }
        gates = {name: {"passed": True} for name in CANDIDATES}
        complete = finalize["complete_modes"]
        self.assertTrue(complete(modes, gates))
        modes["runtime_compile"]["status"] = "failed"
        self.assertFalse(complete(modes, gates))
        modes["runtime_compile"]["status"] = "measured"
        gates["runtime_c"]["passed"] = False
        self.assertFalse(complete(modes, gates))

    def test_frozen_full_panel_is_exactly_80_rows(self):
        cases = list(range(3000, 3040))
        keys = finalize["expected_keys"]({
            "profiling": {
                "case_ids": cases,
                "seed_indices": [0, 1],
                "queries_per_mode": 80,
            }
        })
        self.assertEqual(len(keys), 80)
        self.assertIn("3000_0", keys)
        self.assertIn("3039_1", keys)
        with self.assertRaises(ValueError):
            finalize["expected_keys"]({
                "profiling": {
                    "case_ids": cases[:-1],
                    "seed_indices": [0, 1],
                    "queries_per_mode": 78,
                }
            })

    def test_both_lane_protocols_share_frozen_panel_and_stage_gate(self):
        long = json.loads((HERE / "long" / "protocol.json").read_text(encoding="utf-8"))
        short = json.loads((HERE / "short" / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(long["protocol"], "fastwam-vq-r128-c80-runtime-abc-v1")
        self.assertEqual(short["protocol"], long["protocol"])
        self.assertEqual(long["race_root"], "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-runtime-abc-20261024-r1")
        self.assertEqual(long["remote_root"], long["race_root"] + "-long")
        self.assertEqual(short["remote_root"], short["race_root"] + "-short")
        self.assertEqual(long["job_name"], "vqab0001")
        self.assertEqual(short["job_name"], "vqas0001")
        self.assertEqual(long["allocation"]["execution_queue"], "g1")
        self.assertEqual(short["allocation"]["execution_queue"], "gdev")
        self.assertEqual(long["allocation"]["walltime"], "06:00:00")
        self.assertEqual(short["allocation"]["walltime"], "01:59:00")
        self.assertEqual(long["allocation"]["ncpus"], 16)
        self.assertEqual(long["allocation"]["memory_gb"], 110)
        self.assertEqual(long["profiling"]["case_ids"], short["profiling"]["case_ids"])
        self.assertEqual(long["profiling"]["seed_indices"], [0, 1])
        self.assertEqual(long["profiling"]["queries_per_mode"], 80)
        self.assertEqual(long["profiling"]["action_shape"], [32, 7])
        self.assertEqual(long["stages"]["order"], list(CANDIDATES))
        self.assertEqual(long["stages"]["parity"]["primary_mode"], "layer_decode_fp32")
        self.assertEqual(long["stages"]["parity"]["max_abs_action_atol"], 1e-6)
        self.assertFalse(long["stages"]["performance_or_memory_regression_fatal"])
        self.assertEqual(long["reference_reuse"]["primary_profiling_job"], "25954757.pbs101")
        self.assertEqual(long["reference_reuse"]["profiling_job"], "25943391.pbs101")
        self.assertEqual(
            long["reference_reuse"]["primary_result_root"],
            "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-layerdecode-bf16-20261024-r3-long/results",
        )
        self.assertEqual(
            long["reference_reuse"]["result_root"],
            "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-fused-tuning-20261023-r2/results",
        )

    def test_source_manifest_and_pbs_guards_cover_required_dependencies(self):
        tree = ast.parse((HERE / "control.py").read_text(encoding="utf-8"))
        assignment = next(node for node in tree.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "SOURCES"
                                  for target in node.targets))
        sources = set(ast.literal_eval(assignment.value))
        self.assertTrue({
            "protocol.json", "run.pbs", "environment.sh", "reference_profile.py",
            "export_runtime.py", "fused_input.py", "fused_vq.py", "kernel_runtime.py",
            "layer_runtime.py", "condition_cache.py", "abc_runtime.py", "graph_runtime.py",
            "profile_worker.py", "finalize.py",
        }.issubset(sources))
        script = (HERE / "run.pbs").read_text(encoding="utf-8")
        self.assertRegex(script, r"test -n .*PBS_JOBID")
        self.assertIn("Refusing login-node computation", script)
        self.assertIn("job_state", script)
        self.assertIn("EXPECTED_QUEUE", script)
        self.assertIn("winner.lock", script)
        self.assertIn("RACE_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-runtime-abc-20261024-r1", script)
        self.assertIn("PRIMARY_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-layerdecode-bf16-20261024-r3-long", script)
        self.assertIn("BASELINE_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-fused-tuning-20261023-r2", script)
        self.assertNotIn("fastwam-vq-r128-c80-runtime-abc-20261024-r2", script)
        self.assertIn('"$PRIMARY_RESULTS/storage.json"', script)
        source_loop = set(re.search(r"^for f in (.+); do$", script, re.M).group(1).split())
        self.assertTrue(sources.issubset(source_loop))
        self.assertIn("storage_reuse", (HERE / "finalize.py").read_text(encoding="utf-8"))
        self.assertIn("GPU_SAMPLE", script)
        self.assertIn("sleep 15", script)
        self.assertIn("run_python \"$ARTIFACTS/profile_worker.py\" --mode runtime_a", script)
        self.assertIn("run_optional_mode runtime_c", script)
        self.assertIn("run_optional_mode runtime_compile", script)
        self.assertIn("run_python \"$ARTIFACTS/finalize.py\" --phase final", script)


if __name__ == "__main__":
    unittest.main()
