"""Small CPU-only checks for episode recovery and aggregate coverage gates."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aggregate import ARMS, PROTOCOL, load_episodes, load_manifest, summarize, write_json
from evaluate import recover_records


def fixture_manifest():
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    dimensions = ("camera_viewpoints", "camera_poses", "lighting", "backgrounds",
                 "robot_initial_states", "object_positions", "sensor_noise")
    rows = []
    for suite in suites:
        for dimension in dimensions:
            for repeat in range(10):
                index = len(rows)
                rows.append({"index": index, "variant_id": f"v{index:03d}",
                             "suite": suite, "dimension": dimension, "task_id": repeat})
    return {"protocol": PROTOCOL, "variants": rows,
            "shards": [{"worker_id": worker, "indices": list(range(worker * 70, (worker + 1) * 70))}
                       for worker in range(4)]}


class EvaluationTests(unittest.TestCase):
    def test_complete_paired_aggregate_and_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            manifest = fixture_manifest()
            write_json(results / "manifest.json", manifest)
            banks = {}
            for arm in ARMS[1:]:
                bank = results / "banks" / arm
                bank.mkdir(parents=True)
                write_json(bank / "manifest.json", {"arm": arm})
                banks[arm] = str(bank)
            write_json(results / "PREPARED.json", {
                "protocol": PROTOCOL, "ready": True,
                "numeric_check": {"passed": True}, "native_int4_check": {"passed": True},
                "banks": banks,
            })
            for worker in range(4):
                indices = manifest["shards"][worker]["indices"]
                out = results / f"worker_{worker}"
                out.mkdir()
                expected = len(indices) * len(ARMS)
                write_json(out / "progress.json", {
                    "protocol": PROTOCOL, "worker_id": worker, "status": "complete",
                    "completed": expected, "expected": expected,
                })
                write_json(out / "summary.json", {
                    "protocol": PROTOCOL, "complete": True, "worker_id": worker,
                    "episodes": expected, "expected": expected,
                    "arm_episodes": {arm: len(indices) for arm in ARMS},
                })
                with (out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
                    for position, index in enumerate(indices):
                        row = manifest["variants"][index]
                        for arm_index, arm in enumerate(ARMS):
                            success = (index % 2 == 0 if arm == "bf16" else
                                       index % 4 == 0 if arm == ARMS[1] else
                                       index % 2 == 0 or index % 5 == 0)
                            record = {
                                "protocol": PROTOCOL, "complete": True, "index": index,
                                "variant_id": row["variant_id"], "suite": row["suite"],
                                "dimension": row["dimension"], "task_id": row["task_id"],
                                "state_id": 0, "env_seed": 800_000_000 + index * 1000,
                                "sampler_seed": 900_000_000 + index * 1000,
                                "arm": arm, "success": success,
                                "episode_seconds": 0.1,
                                "metrics": [{
                                    "query_index_in_arm": position,
                                    "first_query_in_arm": position == 0,
                                    "denoising_latency_ms": 10.0 + arm_index,
                                    "full_query_wall_ms": 20.0 + arm_index,
                                }],
                            }
                            if arm != "bf16":
                                record["real_quant_evidence"] = {
                                    "rotation_bank": arm, "activation_bits": 4,
                                    "integer_gemm_calls": 1, "native_int4_gemm_calls": 1,
                                    "native_int4_tensorcore": True, "packed_weight_tensor_bytes": 128,
                                    "bf16_target_weight_absent": True,
                                }
                            stream.write(json.dumps(record) + "\n")

            _doc, rows, assigned = load_manifest(results / "manifest.json")
            episodes = load_episodes(results, rows, assigned)
            summary = summarize(rows, episodes)
            self.assertEqual(summary["episode_slots"], 840)
            self.assertEqual(summary["overall"]["arms"]["bf16"], {
                "successes": 140, "episodes": 280, "success_rate": 0.5,
            })
            self.assertEqual(summary["overall"]["paired_vs_bf16"][ARMS[1]]["success_rate_difference"], -0.25)
            self.assertAlmostEqual(summary["overall"]["paired_vs_bf16"][ARMS[2]]["success_rate_difference"], 0.1)
            self.assertEqual({cell["arms"]["bf16"]["episodes"] for cell in summary["cells"].values()}, {10})
            self.assertEqual({bucket["arms"]["bf16"]["episodes"] for bucket in summary["suites"].values()}, {70})
            self.assertEqual({bucket["arms"]["bf16"]["episodes"] for bucket in summary["dimensions"].values()}, {40})
            for arm_index, arm in enumerate(ARMS):
                timing = summary["timing"][arm]
                first = timing["query_timing"]["first_query"]
                steady = timing["query_timing"]["subsequent_queries"]
                self.assertEqual(first["queries"], 4)
                self.assertEqual(steady["queries"], 276)
                self.assertEqual(first["continuous_denoising_gpu_interval_ms"]["mean_ms"], 10.0 + arm_index)
                self.assertEqual(steady["continuous_denoising_gpu_interval_ms"]["mean_ms"], 10.0 + arm_index)
                self.assertEqual(first["full_action_query_wall_ms"]["mean_ms"], 20.0 + arm_index)
                self.assertEqual(steady["full_action_query_wall_ms"]["mean_ms"], 20.0 + arm_index)
                self.assertEqual(timing["episode_wall_including_environment_ms"]["n"], 280)
                self.assertEqual(timing["episode_wall_including_environment_ms"]["mean_ms"], 100.0)
                self.assertEqual(timing["episode_wall_for_first_query_episodes_ms"]["n"], 4)
                self.assertEqual(timing["episode_wall_for_subsequent_query_episodes_ms"]["n"], 276)

    def test_recovery_drops_only_a_torn_final_line(self):
        rows = fixture_manifest()["variants"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episodes.jsonl"
            record = {"protocol": PROTOCOL, "complete": True, "index": 0,
                      "variant_id": rows[0]["variant_id"], "suite": rows[0]["suite"],
                      "dimension": rows[0]["dimension"], "task_id": rows[0]["task_id"],
                      "state_id": 0, "env_seed": 800_000_000, "sampler_seed": 900_000_000,
                      "arm": "bf16", "success": True}
            path.write_text(json.dumps(record) + "\n{" + '"partial":', encoding="utf-8")
            loaded = recover_records(path, rows, {0})
            self.assertEqual(set(loaded), {("bf16", 0)})
            self.assertTrue(path.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
