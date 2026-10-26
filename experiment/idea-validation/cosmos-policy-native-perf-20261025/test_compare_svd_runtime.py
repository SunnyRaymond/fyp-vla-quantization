from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_svd_runtime as compare
import perf_native


def frozen_old_protocol():
    return {
        "protocol": "cosmos-policy-vqwam-v1",
        "mode": "libero_direct_policy",
        "arms": list(perf_native.ARMS),
        "inference": {"denoising_steps": 5},
        "performance": {
            "case_ids": list(range(3000, 3040)), "seed_indices": [0, 1],
            "queries": 80, "blocks": 3, "repetitions": 3, "warmup_queries": 3,
        },
    }


def frozen_native_protocol():
    return {
        "source_protocol": "cosmos-policy-vqwam-v1",
        "panel": {
            "case_ids_start": 3000, "case_ids_end_inclusive": 3039,
            "seed_indices": [0, 1], "queries": 80, "blocks": 3,
            "repetitions": 3, "samples_per_arm_per_scope": 720,
            "batch_size": 1, "nfe": 5,
        },
    }


def records(scope, values):
    latency_field = "denoising_seconds" if scope == "sampler" else "dit_replay_seconds"
    return [
        {"query": list(key), "repeat": repeat, latency_field: value,
         "complete_endpoint_query_seconds": value + 0.1,
         "peak_allocated_bytes": 1000 + index}
        for index, ((key, repeat), value) in enumerate(zip(compare.expected_measurement_ids(), values))
    ]


class CompareSVDRuntimeTests(unittest.TestCase):
    def test_fixed_diagnostic_keys_and_denominator_do_not_replace_formal_panel(self):
        self.assertEqual(compare.DIAGNOSTIC_KEYS,
                         ((3000, 0), (3009, 1), (3020, 0), (3039, 1)))
        contract = compare.validate_native_panel(frozen_old_protocol(), frozen_native_protocol())
        self.assertEqual(contract["samples_per_arm"], 40)
        self.assertEqual(len(compare.expected_measurement_ids()), 12)
        self.assertEqual(len(compare.WARMUP_KEYS), 3)
        self.assertEqual(compare.REPEATS, 3)

    def test_native_panel_rejects_wrong_batch_or_formal_denominator(self):
        panel = frozen_native_protocol()
        panel["panel"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "batch-1"):
            compare.validate_native_panel(frozen_old_protocol(), panel)
        panel = frozen_native_protocol()
        panel["panel"]["samples_per_arm_per_scope"] = 12
        with self.assertRaisesRegex(ValueError, "batch-1"):
            compare.validate_native_panel(frozen_old_protocol(), panel)

    def test_sample_validator_requires_exactly_four_keys_times_three_repeats(self):
        valid = records("sampler", [1.0] * 12)
        self.assertEqual(compare.validate_measurement_records(valid), 12)
        with self.assertRaisesRegex(ValueError, "exactly four fixed keys"):
            compare.validate_measurement_records(valid[:-1])
        with self.assertRaisesRegex(ValueError, "exactly four fixed keys"):
            compare.validate_measurement_records(valid + [dict(valid[0])])

    def test_speedup_uses_means_not_medians(self):
        current_values = [2.0] * 11 + [100.0]
        fused_values = [1.0] * 11 + [100.0]
        current = compare.summarize_variant_records("sampler", records("sampler", current_values))
        fused = compare.summarize_variant_records("sampler", records("sampler", fused_values))
        speedup = compare.mean_speedup(current, fused)
        expected = current["latency_seconds"]["mean"] / fused["latency_seconds"]["mean"]
        median_ratio = current["latency_seconds"]["median"] / fused["latency_seconds"]["median"]
        self.assertEqual(speedup, expected)
        self.assertNotEqual(speedup, median_ratio)

    def test_each_variant_checks_both_full80_gates_before_binding_assertion(self):
        calls = []

        class Worker:
            @staticmethod
            def _require_gate(root, arm, variant, scope):
                calls.append(("gate", arm, variant, scope))

            @staticmethod
            def assert_native_gate_binding(root, arm, variant, binding):
                calls.append(("binding", arm, variant, binding))

        compare.require_variant_gates(Worker, Path("root"), "native_fused", {"commit": "x"})
        self.assertEqual([row[0] for row in calls], ["gate", "gate", "binding"])
        self.assertEqual([row[-1] for row in calls[:2]], ["sampler", "dit_replay"])
        self.assertEqual(calls[-1][-1], {"commit": "x"})

    def test_sampler_stage_precedes_dit_stage_and_variant_order_is_fixed(self):
        self.assertEqual(compare.SCOPE_STAGES,
                         (("sampler", ("current", "native_fused")),
                          ("dit_replay", ("current", "native_fused"))))
        self.assertEqual(compare.VARIANTS, ("current", "native_fused"))

    def test_dit_replay_keeps_only_forward_inputs_not_reference_outputs(self):
        reference_output = object()
        capture = {"calls": [{"args": (object(),), "kwargs": {},
                              "output": reference_output} for _ in range(5)]}
        calls = compare.replay_input_calls(capture)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(set(call) == {"args", "kwargs"} for call in calls))
        self.assertTrue(all("output" not in call for call in calls))
        with self.assertRaisesRegex(ValueError, "five captured NFE"):
            compare.replay_input_calls({"calls": capture["calls"][:4]})


if __name__ == "__main__":
    unittest.main()
