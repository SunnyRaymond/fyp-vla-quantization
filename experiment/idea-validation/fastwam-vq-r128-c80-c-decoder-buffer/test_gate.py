"""Small standard-library checks for C decoder/buffer orchestration gates."""
import ast
import json
from pathlib import Path
import re
import runpy
import shutil
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
finalize = runpy.run_path(str(HERE / "finalize.py"))
control = runpy.run_path(str(HERE / "control.py"))
REFERENCE_MODES = finalize["REFERENCE_MODES"]
REUSED_MODES = finalize["REUSED_MODES"]
CANDIDATE_MODES = finalize["CANDIDATE_MODES"]


def candidate_receipts(mode):
    layout = {
        "logical_bytes": 42147840,
        "borrowed_weight_scratch_bytes": 0 if mode == "c_decoder" else 33718272,
        "independent_bytes": 42147840 if mode == "c_decoder" else 8429568,
        "alias_weight_scratch": mode == "c_buffer",
        "h_uses_weight_storage": mode == "c_buffer",
        "u_uses_weight_storage": mode == "c_buffer",
        "a4_independent_of_weight_storage": True,
    }
    decoder = {
        "schema": "fastwam_decoder_tuning_v1",
        "protocol": "fastwam-vq-r128-c80-c-decoder-buffer-v1",
        "pbs_jobid": "25960000.pbs101",
        "mode": mode,
        "checkpoint_identity": {"source_job": "25921795.pbs101", "bundle_path": "/frozen/inference_bundle.pt"},
        "reuse_source": None if mode == "c_decoder" else "/new/results/c_decoder/tuning.json",
        "reuse_source_pbs_jobid": None if mode == "c_decoder" else "25959865.pbs101",
        "call_frequency_status": "complete",
        "frequency_panel": {
            "query_count": 80,
            "logical_decode_calls_full_panel": 420640,
            "logical_decode_calls_per_query": 5258.0,
        },
        "frequency_weighted": {
            "query_count": 80,
            "logical_decode_calls_full_panel": 420640,
            "reference_decode_ms_full_panel": 1.0,
            "selected_decode_ms_full_panel": 1.0,
            "selected_over_reference": 1.0,
        },
        "modules": [{"name": f"m{i}", "calls": 686 if i < 160 else 685}
                    for i in range(614)],
        "shapes": [{"shape": [16, 16], "selected": "reference"}],
        "parity": {"checked_modules": 614, "bitwise_passed": 614, "candidate_failures": []},
    }
    # Keep the synthetic full-panel call total aligned with the protocol's exact graph count.
    calls = [row["calls"] for row in decoder["modules"]]
    decoder["modules"][-1]["calls"] += 420640 - sum(calls)
    validation = {
        "strict_anchor_gate": True,
        "source_anchor_parity": {"action_max_abs": 0.0, "future_max_abs": 0.0, "passed": True},
        "full80_action_parity_vs_layer_decode_fp32": {
            "max_abs": 0.0, "rmse": 0.0, "exact_queries": 80, "passed": True,
        },
        "decoder_optimization": decoder,
    }
    summary = {
        "protocol": decoder["protocol"],
        "pbs_jobid": decoder["pbs_jobid"],
        "compiled": False,
        "cuda_graph": True,
        "deterministic_algorithms": True,
        "fill_uninitialized_memory": True,
        "allow_tf32": False,
        "allow_bf16_reduced_precision_reduction": True,
        "new_training_updates": 0,
        "closed_loop_episodes": 0,
        "performance_accuracy": {
            "anchor_action_max_abs": 0.0,
            "anchor_future_max_abs": 0.0,
            "anchor_passed": True,
            "full80_action_max_abs_vs_layer_decode_fp32": 0.0,
            "full80_action_rmse_vs_layer_decode_fp32": 0.0,
        },
        "optimization": {
            "after_primary": {"graph": {
                "kind": "cuda_graph",
                "regions": {
                    "video_step": {"captures": 1, "replays": 800},
                    "video_prefill": {"captures": 1, "replays": 80},
                    "action_step": {"captures": 1, "replays": 800},
                },
            }},
            "logical_target_linear_calls_per_query": 5258.0,
        },
        "decoder_optimization": decoder,
        "installation": {"activation_workspace_layout": layout},
        "parameter_execution": {"coverage_scope": "candidate anchor and 3 warmups including graph capture/warmup; no extra forward"},
    }
    return summary, validation, decoder


class GateTests(unittest.TestCase):
    def test_action_parity_requires_exact_zero(self):
        parity_pass = finalize["parity_pass"]
        self.assertTrue(parity_pass(0.0, 0.0))
        self.assertFalse(parity_pass(1e-15, 0.0))
        self.assertFalse(parity_pass(float("nan"), 0.0))

    def test_candidate_checks_cover_graph_decoder_and_exact_anchor(self):
        checks = finalize["mode_checks"]
        for mode in CANDIDATE_MODES:
            summary, validation, _ = candidate_receipts(mode)
            result = checks(summary, validation, mode)
            self.assertTrue(result["passed"], result["checks"])
            summary["performance_accuracy"]["anchor_future_max_abs"] = 1e-15
            self.assertFalse(checks(summary, validation, mode)["passed"])

    def test_buffer_stage_waits_for_decoder_parity(self):
        allowed = finalize["stage_allowed"]
        self.assertTrue(allowed("c_decoder", {}))
        self.assertFalse(allowed("c_buffer", {}))
        self.assertFalse(allowed("c_buffer", {"c_decoder": {"passed": False}}))
        self.assertTrue(allowed("c_buffer", {"c_decoder": {"passed": True}}))

    def test_complete_requires_reused_c_and_both_exact_candidates(self):
        modes = {
            name: {"status": "cached", "queries": 80}
            for name in REFERENCE_MODES
        }
        modes["runtime_c"] = {
            "status": "reused_candidate", "queries": 80,
            "provenance": {"reused": True,
                           "protocol": "fastwam-vq-r128-c80-runtime-abc-v1",
                           "source_job": "25956803.pbs101"},
            "parity_vs_primary": {"passed": True, "max_abs": 0.0},
            "validation_checks": {"passed": True},
        }
        modes["c_decoder"] = {
            "status": "reused_candidate", "queries": 80,
            "provenance": {"reused": True, "protocol": "fastwam-vq-r128-c80-c-decoder-buffer-v1",
                           "source_job": "25959865.pbs101"},
            "parity_vs_primary": {"passed": True, "max_abs": 0.0},
            "validation_checks": {"passed": True},
        }
        modes["c_buffer"] = {
            "status": "measured", "queries": 80,
            "provenance": {"reused": False},
            "parity_vs_primary": {"passed": True, "max_abs": 0.0},
            "validation_checks": {"passed": True},
        }
        gates = {name: {"passed": True} for name in CANDIDATE_MODES}
        complete = finalize["complete_modes"]
        reused = REUSED_MODES + ("c_decoder",)
        sources = {
            "runtime_c": {"source_job": "25956803.pbs101", "source_protocol": "fastwam-vq-r128-c80-runtime-abc-v1"},
            "c_decoder": {"source_job": "25959865.pbs101", "source_protocol": "fastwam-vq-r128-c80-c-decoder-buffer-v1"},
        }
        self.assertTrue(complete(modes, gates, reused, sources))
        modes["c_decoder"]["provenance"]["source_job"] = "wrong-job"
        self.assertFalse(complete(modes, gates, reused, sources))
        modes["c_decoder"]["provenance"]["source_job"] = "25959865.pbs101"
        modes["c_buffer"]["parity_vs_primary"]["max_abs"] = 1e-15
        self.assertFalse(complete(modes, gates, reused, sources))
        modes["c_buffer"]["parity_vs_primary"]["max_abs"] = 0.0
        gates["c_decoder"]["passed"] = False
        self.assertFalse(complete(modes, gates, reused, sources))

    def test_protocol_preserves_frozen_panel_reuse_and_allocations(self):
        long = json.loads((HERE / "long" / "protocol.json").read_text(encoding="utf-8"))
        short = json.loads((HERE / "short" / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(long["protocol"], "fastwam-vq-r128-c80-c-decoder-buffer-v1")
        self.assertEqual(short["protocol"], long["protocol"])
        self.assertEqual(long["race_root"], "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-c-decoder-buffer-20261024-r3")
        self.assertEqual(long["remote_root"], long["race_root"] + "-long")
        self.assertEqual(short["remote_root"], long["race_root"] + "-short")
        self.assertEqual(long["job_name"], "vqcdl003")
        self.assertEqual(short["job_name"], "vqcds003")
        self.assertEqual((long["allocation"]["queue"], long["allocation"]["execution_queue"]), ("normal", "g1"))
        self.assertEqual((short["allocation"]["queue"], short["allocation"]["execution_queue"]), ("normal", "gdev"))
        self.assertEqual(long["allocation"]["walltime"], "06:00:00")
        self.assertEqual(short["allocation"]["walltime"], "01:59:00")
        self.assertEqual((long["allocation"]["ncpus"], long["allocation"]["memory_gb"]), (16, 110))
        profile = long["profiling"]
        self.assertEqual((profile["case_ids"][0], profile["case_ids"][-1], len(profile["case_ids"])), (3000, 3039, 40))
        self.assertEqual(profile["seed_indices"], [0, 1])
        self.assertEqual(profile["queries_per_mode"], 80)
        self.assertEqual(profile["warmup_queries"], 3)
        self.assertEqual((profile["steps_video"], profile["steps_action"], profile["action_horizon"], profile["video_frames"]), (10, 10, 32, 9))
        self.assertEqual(profile["action_shape"], [32, 7])
        self.assertFalse(profile["torch_compile"])
        self.assertEqual(long["stages"]["order"], list(CANDIDATE_MODES))
        self.assertEqual(profile["mode_execution_order"], ["c_buffer"])
        self.assertEqual(profile["stage1_mode_order"], [])
        self.assertEqual(profile["stage2_mode_order"], ["c_buffer"])
        self.assertEqual(long["stages"]["dependencies"], {"c_decoder": ["runtime_c"], "c_buffer": ["c_decoder"]})
        self.assertEqual(long["stages"]["parity"]["max_abs_action_atol"], 0.0)
        self.assertEqual(long["reference_reuse"]["primary_profiling_job"], "25954757.pbs101")
        self.assertEqual(long["reference_reuse"]["profiling_job"], "25943391.pbs101")
        reuse = long["candidate_reuse"]
        self.assertEqual(reuse["modes"], ["runtime_c"])
        self.assertEqual(reuse["source_protocol"], "fastwam-vq-r128-c80-runtime-abc-v1")
        self.assertEqual(reuse["source_job"], "25956803.pbs101")
        self.assertEqual(reuse["source_job_exit_status"], 141)
        self.assertFalse(reuse["whole_pipeline_completion_required"])
        self.assertEqual(reuse["result_root"], "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-runtime-abc-20261024-r1-long/results")
        self.assertEqual(long["storage_inventory"]["expected_unique_storage_bytes"], 12779984326)
        completed = long["completed_candidate_reuse"]
        self.assertEqual(completed["modes"], ["c_decoder"])
        self.assertEqual(completed["source_protocol"], long["protocol"])
        self.assertEqual(completed["source_job"], "25959865.pbs101")
        self.assertEqual(completed["source_job_name"], "vqcdl002")
        self.assertEqual(completed["result_root"], "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-c-decoder-buffer-20261024-r2-long/results")

    def test_real_legacy_layer_decode_summary_uses_source_protocol_geometry(self):
        legacy_path = (HERE.parent / "fastwam-vq-r128-c80-runtime-abc" / "pulled"
                       / "25956803.pbs101" / "layer_decode_fp32" / "summary.json")
        summary = json.loads(legacy_path.read_text(encoding="utf-8"))
        self.assertNotIn("cohort", summary)
        protocol_path = (HERE.parent / "fastwam-vq-r128-c80-layerdecode-bf16"
                         / "long" / "protocol.json")
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            shutil.copyfile(protocol_path, source_root / "protocol.json")
            p = json.loads((HERE / "long" / "protocol.json").read_text(encoding="utf-8"))
            p["reference_reuse"]["primary_artifact_root"] = str(source_root)
            source = finalize["validate_cohort"](p, summary, "layer_decode_fp32")
            self.assertTrue(source.startswith("source_protocol:"))
            self.assertNotIn("cohort", summary)
            wrong = json.loads((source_root / "protocol.json").read_text(encoding="utf-8"))
            wrong["profiling"]["steps_video"] = 9
            (source_root / "protocol.json").write_text(json.dumps(wrong), encoding="utf-8")
            with self.assertRaises(ValueError):
                finalize["validate_cohort"](p, summary, "layer_decode_fp32")

    def test_control_completion_accepts_individual_c_receipt_after_compile_cancel(self):
        p = control["configure"]("long")
        source_job = p["candidate_reuse"]["source_job"]
        completed = p["completed_candidate_reuse"]
        current_job = "25960000.pbs101"
        modes = {name: {"status": "cached", "queries": 80} for name in REFERENCE_MODES}
        modes["runtime_c"] = {
            "status": "reused_candidate", "queries": 80,
            "provenance": {"reused": True, "protocol": p["candidate_reuse"]["source_protocol"],
                           "pbs_jobid": source_job, "source_job": source_job},
            "parity_vs_primary": {"passed": True, "max_abs": 0.0},
            "validation_checks": {"passed": True},
            "source_stage_gate": {"passed": True},
        }
        required = {
            "expected_runtime_kind", "logical_coverage_5258_per_query", "validation_full80_exact_80",
            "decoder_receipt_bound_to_job", "decoder_reuse_source_bound", "decoder_frequency_weights_full80",
            "decoder_bitwise_all_modules", "anchor_action_max_abs_eq_0", "anchor_future_max_abs_eq_0",
            "parameter_execution_probe_scope_is_limited", "activation_workspace_layout_exact",
            "decoder_reuse_source_job_recorded",
        }
        for name in CANDIDATE_MODES:
            checks = {key: True for key in required}
            if name == "c_buffer":
                checks.update(decoder_tuning_matches_c_decoder=True,
                              decoder_frequency_panel_matches_c_decoder=True,
                              decoder_checkpoint_matches_c_decoder=True,
                              decoder_source_pbs_job_matches_completed_reuse=True)
            modes[name] = {
                "status": "reused_candidate" if name == "c_decoder" else "measured", "queries": 80,
                "provenance": {
                    "reused": name == "c_decoder",
                    "protocol": completed["source_protocol"] if name == "c_decoder" else p["protocol"],
                    "pbs_jobid": completed["source_job"] if name == "c_decoder" else current_job,
                    "source_job": completed["source_job"] if name == "c_decoder" else None,
                    "source_result_root": completed["result_root"] if name == "c_decoder" else None,
                },
                "parity_vs_primary": {"passed": True, "max_abs": 0.0},
                "validation_checks": {"passed": True, "checks": checks},
                "stage_gate": {"passed": True, "parity_vs_primary": {"max_abs": 0.0}},
            }
        summary = {
            "protocol": p["protocol"], "pbs_jobid": current_job, "complete": True,
            "new_training_updates": 0, "closed_loop_episodes": 0, "full_queries_per_mode": 80,
            "candidate_modes": list(CANDIDATE_MODES),
            "stage_dependencies": p["stages"]["dependencies"], "performance_regressions_fatal": False,
            "modes": modes,
            "candidate_reuse": {
                "source_job": source_job, "source_job_name": p["candidate_reuse"]["source_job_name"],
                "modes": ["runtime_c"], "source_protocol": p["candidate_reuse"]["source_protocol"],
                "source_result_root": p["candidate_reuse"]["result_root"],
                "whole_pipeline_completion_required": False,
            },
            "completed_candidate_reuse": {
                "modes": ["c_decoder"], "source_protocol": completed["source_protocol"],
                "source_job": completed["source_job"], "source_job_name": completed["source_job_name"],
                "source_result_root": completed["result_root"],
            },
            "reused_candidate_modes": ["c_decoder"],
            "storage_inventory": {"passed": True},
            "stage_gates": {name: {"passed": True} for name in CANDIDATE_MODES},
        }
        value = {
            "handle": {"remote_root": p["remote_root"], "protocol": p["protocol"],
                       "job_name": p["job_name"], "job_id": current_job},
            "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0",
                          "Job_Name": p["job_name"], "queue": p["allocation"]["execution_queue"],
                          "Resource_List.walltime": p["allocation"]["walltime"],
                          "Resource_List.select": "1:ncpus=16:mem=110gb:ngpus=1"},
            "artifact": {"exit_code": "0", "pipeline_complete": True},
            "summary": summary,
        }
        self.assertEqual(p["candidate_reuse"]["source_job_exit_status"], 141)
        self.assertTrue(control["complete_gate"](value))

    def test_source_manifest_and_pbs_guards_cover_required_dependencies(self):
        tree = ast.parse((HERE / "control.py").read_text(encoding="utf-8"))
        assignment = next(node for node in tree.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "SOURCES"
                                  for target in node.targets))
        sources = set(ast.literal_eval(assignment.value))
        self.assertTrue({"protocol.json", "run.pbs", "environment.sh", "decoder_runtime.py",
                         "inventory.py", "profile_worker.py", "finalize.py"}.issubset(sources))
        script = (HERE / "run.pbs").read_text(encoding="utf-8")
        self.assertRegex(script, r"test -n .*PBS_JOBID")
        self.assertIn("Refusing login-node computation", script)
        self.assertIn("winner.lock", script)
        self.assertIn("GPU_SAMPLE", script)
        self.assertIn("sleep 15", script)
        self.assertIn("fastwam-vq-r128-c80-c-decoder-buffer-20261024-r3", script)
        self.assertIn("fastwam-vq-r128-c80-runtime-abc-20261024-r1-long", script)
        self.assertIn("25956803.pbs101", script)
        self.assertIn("vqcdl003", script)
        self.assertIn("vqcds003", script)
        self.assertIn("inventory.py", script)
        self.assertIn("INVENTORY_RESULTS=", script)
        self.assertIn("25959495.pbs101", script)
        self.assertLess(script.index("from decoder_runtime import smoke"), script.index("--stage-gate c_decoder"))
        self.assertLess(script.index("--stage-gate c_decoder"), script.index("--mode c_buffer"))
        self.assertNotIn("--mode c_decoder", script)
        self.assertIn("--mode c_buffer", script)
        self.assertNotIn("--mode runtime_c", script)
        self.assertNotIn("runtime_compile", script)
        self.assertIn("--stage-gate c_decoder", script)
        self.assertIn("--stage-gate c_buffer", script)
        self.assertIn("--phase final", script)
        self.assertIn("PRIMARY_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-layerdecode-bf16-20261024-r3-long", script)
        self.assertIn("BASELINE_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-fused-tuning-20261023-r2", script)


if __name__ == "__main__":
    unittest.main()
