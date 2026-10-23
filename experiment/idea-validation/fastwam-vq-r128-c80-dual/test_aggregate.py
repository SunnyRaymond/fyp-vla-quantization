"""Small standard-library tests; all metrics below are synthetic fixtures."""
import copy
import unittest

import aggregate


def _protocol():
    refs = {
        "H_VQ8": {
            "root": "/fixture/hvq8", "protocol": "fastwam-vq-r128-joint-c8-v1", "arm": "H_VQ8",
            "split": "heldout", "job": "100.pbs101",
        },
        "R0_W16A4": {
            "root": "/fixture/r0", "protocol": "fastwam-r128-improvements-v1", "arm": "C80_once_lr1e4",
            "lane": "coverage_c80", "split": "heldout", "job": "101.pbs101",
        },
    }
    return {
        "protocol": aggregate.PROTOCOL,
        "target_linears": 614,
        "suites": list(aggregate.SUITES),
        "calibration": {
            "case_ids": list(aggregate.CALIBRATION_CASE_IDS),
            "c80_case_ids": list(aggregate.CALIBRATION_CASE_IDS),
            "seed_indices": [0],
        },
        "heldout": {
            "case_ids": list(aggregate.HELDOUT_CASE_IDS),
            "seed_indices": [0, 1], "queries": 80,
        },
        "evaluation": {
            "case_ids": list(aggregate.HELDOUT_CASE_IDS), "seed_indices": [0, 1],
        },
        "training": {
            "latent_weight": aggregate.LATENT_WEIGHT,
            "vq_updates": 8, "rotation_updates": 8, "contexts_per_update": 10,
            "vq_learning_rate": 5e-4, "rotation_learning_rate": 1e-5,
        },
        "endpoints": list(aggregate.NEW_ARMS),
        "reference_arms": refs,
    }


def _record(protocol, arm, split, case, seed, value, legacy=False):
    if split == "calibration":
        suite = aggregate.SUITES[(case - 2000) // 20]
    else:
        suite = aggregate.SUITES[(case - 3000) // 10]
    mse = value * value
    future = value / 10
    row = {
        "protocol": protocol["reference_arms"][arm]["protocol"] if legacy else aggregate.PROTOCOL,
        "arm": arm, "split": split, "case_id": case, "seed_index": seed, "suite": suite,
        "first10motor_rmse": value, "first10motor_mse": mse,
        "future_latent_nmse": future,
        "mixed_loss": mse + aggregate.LATENT_WEIGHT * future,
        "full32motor_rmse": value + 0.1,
        "late22motor_rmse": value + 0.2,
        "first10gripper_rmse": value + 0.3,
    }
    if arm in aggregate.REFERENCE_ARMS:
        row["source_binding"] = copy.deepcopy(protocol["reference_arms"][arm])
    return row


def _fixture(protocol=None, legacy_refs=False):
    protocol = protocol or _protocol()
    calibration = {}
    for arm_index, arm in enumerate(aggregate.CALIBRATION_ARMS):
        calibration[arm] = [
            _record(protocol, arm, "calibration", case, 0,
                    0.01 * (arm_index + 1) + (case - 2000) // 20)
            for case in aggregate.CALIBRATION_CASE_IDS
        ]
    heldout = {}
    for arm_index, arm in enumerate(aggregate.HELDOUT_ARMS):
        heldout[arm] = [
            _record(protocol, arm, "heldout", case, seed,
                    0.02 * (arm_index + 1) + (case - 3000) // 10 + seed * 0.2,
                    legacy=legacy_refs and arm in aggregate.REFERENCE_ARMS)
            for case in aggregate.HELDOUT_CASE_IDS
            for seed in aggregate.SEED_INDICES
        ]
    return protocol, calibration, heldout


class AggregateTests(unittest.TestCase):
    def test_builds_exact_compact_comparison_and_markdown(self):
        protocol, calibration, heldout = _fixture()
        result = aggregate.build_comparison(protocol, calibration, heldout, {"fixture": "synthetic"})
        self.assertTrue(result["complete"])
        self.assertEqual(result["calibration"]["Seq8"]["queries"], 80)
        self.assertEqual(result["heldout"]["Joint8"]["queries"], 80)
        self.assertEqual(result["heldout"]["Joint8"]["contexts"], 40)
        self.assertNotIn("by_case", result["heldout"]["Joint8"])
        self.assertEqual(set(result["comparisons"]), {
            "Joint8_vs_Seq8", "Seq8_vs_R0_VQ0", "Joint8_vs_R0_VQ0",
            "Seq8_vs_H_VQ8", "Seq8_vs_R0_W16A4",
            "Joint8_vs_H_VQ8", "Joint8_vs_R0_W16A4",
        })
        for metric in aggregate.METRICS:
            self.assertEqual(result["comparisons"]["Joint8_vs_Seq8"]
                             ["paired_context_win_loss"][metric]["contexts"], 40)
        markdown = aggregate.render_markdown(result)
        self.assertIn("C80 calibration：按 suite", markdown)
        self.assertIn("Joint8 includes 8 additional R updates", result["claim_scope"])
        self.assertIn("final update 8", markdown)
        self.assertIn("Context improved / worsened / tied", markdown)
        self.assertIn("H_VQ8", markdown)
        self.assertIn("First10 motor MSE", markdown)

    def test_mean_seed_then_equal_suite_weighting(self):
        protocol, calibration, heldout = _fixture()
        result = aggregate.build_comparison(protocol, calibration, heldout, {})
        # Seq8 has base .04, suite offsets 0..3, and a .1 mean seed offset.
        expected = (0.14 + 1.14 + 2.14 + 3.14) / 4
        self.assertAlmostEqual(
            result["heldout"]["Seq8"]["overall"]["first10motor_rmse"], expected
        )
        self.assertAlmostEqual(
            result["heldout"]["Seq8"]["by_suite"][aggregate.SUITES[2]]["first10motor_rmse"], 2.14
        )

    def test_legacy_reference_protocol_requires_matching_source_binding(self):
        protocol, calibration, heldout = _fixture(legacy_refs=True)
        aggregate.build_comparison(protocol, calibration, heldout, {})
        heldout["R0_W16A4"][0].pop("source_binding")
        with self.assertRaisesRegex(ValueError, "source_binding"):
            aggregate.build_comparison(protocol, calibration, heldout, {})

        protocol, calibration, heldout = _fixture()
        heldout["H_VQ8"][0]["source_binding"]["job"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source_binding job"):
            aggregate.build_comparison(protocol, calibration, heldout, {})

    def test_rejects_wrong_binding_coverage_nonfinite_and_loss(self):
        mutations = (
            (lambda row: row.update(arm="Joint8"), "arm/split"),
            (lambda row: row.update(split="calibration"), "arm/split"),
            (lambda row: row.update(protocol="other"), "protocol binding"),
            (lambda row: row.update(first10motor_rmse=float("nan")), "finite"),
            (lambda row: row.update(mixed_loss=9.0), "mixed_loss"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                protocol, calibration, heldout = _fixture()
                mutate(heldout["Seq8"][0])
                with self.assertRaisesRegex(ValueError, message):
                    aggregate.build_comparison(protocol, calibration, heldout, {})

        protocol, calibration, heldout = _fixture()
        heldout["Seq8"].append(copy.deepcopy(heldout["Seq8"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate.build_comparison(protocol, calibration, heldout, {})

        protocol, calibration, heldout = _fixture()
        calibration["R0_VQ0"] = calibration["R0_VQ0"][:-1]
        with self.assertRaisesRegex(ValueError, "exactly 80 records"):
            aggregate.build_comparison(protocol, calibration, heldout, {})

    def test_rejects_protocol_or_suite_cohort_drift(self):
        protocol, calibration, heldout = _fixture()
        protocol["training"]["vq_updates"] = 7
        with self.assertRaisesRegex(ValueError, "vq_updates"):
            aggregate.build_comparison(protocol, calibration, heldout, {})

        protocol, calibration, heldout = _fixture()
        heldout["Seq8"][0]["suite"] = aggregate.SUITES[1]
        with self.assertRaisesRegex(ValueError, "inconsistent suite"):
            aggregate.build_comparison(protocol, calibration, heldout, {})


if __name__ == "__main__":
    unittest.main()
