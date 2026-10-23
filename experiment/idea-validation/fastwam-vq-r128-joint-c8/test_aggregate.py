"""stdlib-only logic checks; no model or Torch imports."""
import json
import tempfile
import unittest
from pathlib import Path

import aggregate


def _record(case_id, seed_index, suite, value):
    rmse = float(value)
    future = float(value) / 10
    mse = rmse * rmse
    return {
        "case_id": case_id,
        "seed_index": seed_index,
        "suite": suite,
        "first10motor_rmse": rmse,
        "first10motor_mse": mse,
        "future_latent_nmse": future,
        "mixed_loss": mse + aggregate.LATENT_WEIGHT * future,
        "full32motor_rmse": rmse + 0.1,
        "late22motor_rmse": rmse + 0.2,
        "first10gripper_rmse": rmse + 0.3,
    }


def _heldout_records(offset=0.0):
    suite_by_case = {
        case: aggregate.SUITES[(case - 3000) // 10]
        for case in aggregate.CASE_IDS
    }
    return [
        _record(case, seed, suite_by_case[case], offset + aggregate.SUITES.index(suite_by_case[case]) + 1)
        for case in aggregate.CASE_IDS
        for seed in aggregate.SEED_INDICES
    ]


def _c8_records(offset=0.0):
    return [
        _record(case, 0, aggregate.C8_SUITE_BY_CASE[case],
                offset + aggregate.SUITES.index(aggregate.C8_SUITE_BY_CASE[case]) + 1)
        for case in aggregate.C8_CASE_IDS
    ]


class AggregateTests(unittest.TestCase):
    def test_strict_heldout_coverage_and_duplicate_rejection(self):
        records = _heldout_records()
        self.assertEqual(aggregate.aggregate_records(records)["queries"], 80)
        with self.assertRaisesRegex(ValueError, "missing"):
            aggregate.aggregate_records(records[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate.aggregate_records(records + [records[0]])

    def test_missing_mse_and_mixed_loss_are_normalized(self):
        records = _heldout_records()
        for row in records:
            row.pop("first10motor_mse")
            row.pop("mixed_loss")
        result = aggregate.aggregate_records(records)
        first = result["by_case"]["3000"]
        self.assertAlmostEqual(first["first10motor_mse"], first["first10motor_rmse"] ** 2)
        expected = first["first10motor_mse"] + aggregate.LATENT_WEIGHT * first["future_latent_nmse"]
        self.assertAlmostEqual(first["mixed_loss"], expected)

    def test_equal_suite_weighting(self):
        result = aggregate.aggregate_records(_heldout_records())
        suite_means = [result["by_suite"][suite]["first10motor_rmse"] for suite in aggregate.SUITES]
        self.assertAlmostEqual(result["overall"]["first10motor_rmse"], sum(suite_means) / 4)
        self.assertAlmostEqual(result["overall"]["first10motor_rmse"], 2.5)

    def test_report_interface_and_c8_mapping(self):
        p = {
            "protocol": aggregate.PROTOCOL,
            "target_linears": 614,
            "suites": list(aggregate.SUITES),
            "evaluation": {"case_ids": list(aggregate.CASE_IDS), "seed_indices": [0, 1]},
            "calibration": {"case_ids": list(aggregate.C8_CASE_IDS), "seed_indices": [0]},
            "training": {"latent_weight": aggregate.LATENT_WEIGHT},
        }
        heldout = {arm: _heldout_records(i * 0.01) for i, arm in enumerate(aggregate.HELDOUT_ARMS)}
        calibration = {arm: _c8_records(i * 0.01) for i, arm in enumerate(aggregate.CALIBRATION_ARMS)}
        provenance = {"reuse": {"arms": ["H_VQ8", "R8_VQ8"]}, "training_cost": {"gpu_hours": 1.25}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            report = aggregate.write_report(temp, p, heldout, calibration, provenance)
            saved = json.loads(Path(temp, "comparison.json").read_text(encoding="utf-8"))
            markdown = Path(temp, "RESULTS.zh.md").read_text(encoding="utf-8")
        self.assertEqual(set(report["comparisons"]), {
            "noVQ_R8_vs_H", "Joint8_vs_H_VQ8", "Joint8_vs_R8_VQ8"
        })
        self.assertEqual(saved["calibration"]["H_VQ0"]["queries"], 8)
        self.assertAlmostEqual(saved["calibration"]["H_VQ0"]["overall"]["first10motor_rmse"], 2.55)
        self.assertIn("training_cost", markdown)
        self.assertIn("直接对 compressed student 做 coadaptation", markdown)


if __name__ == "__main__":
    unittest.main()
