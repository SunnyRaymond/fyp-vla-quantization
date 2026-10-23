"""Synthetic, standard-library tests for the transport cohort aggregator."""
import copy
import json
import math
import pathlib
import unittest

import aggregate


HERE = pathlib.Path(__file__).resolve().parent
with (HERE / "protocol.json").open(encoding="utf-8") as handle:
    PROTOCOL = json.load(handle)
ARMS = PROTOCOL["probe_arms"] + PROTOCOL["training_arms"]
BASE = {
    "Seq_A4": (1.0, 1.0),
    "Seq_to_Joint_A4": (1.05, 1.05),
    "Joint_A4": (1.2, 1.2),
    "Joint_to_Seq_A4": (1.15, 1.15),
    "Seq_noA4": (1.0, 1.0),
    "Seq_to_Joint_noA4": (1.0001, 1.0001),
    "Joint_noA4": (1.2, 1.2),
    "Joint_to_Seq_noA4": (1.2001, 1.2001),
    "Comp8": (0.9, 0.9),
    "Uncomp8": (1.2, 1.2),
}


def _record(protocol, arm, split, case, seed, suite, mse, latent):
    return {
        "protocol": protocol,
        "arm": arm,
        "split": split,
        "case_id": case,
        "seed_index": seed,
        "suite": suite,
        "sampler_seed": case * 10 + seed,
        "first10motor_rmse": math.sqrt(mse),
        "future_latent_nmse": latent,
        "mixed_loss": mse + aggregate.LATENT_WEIGHT * latent,
        "full32motor_rmse": math.sqrt(mse + 0.2),
        "late22motor_rmse": math.sqrt(mse + 0.1),
        "first10gripper_rmse": math.sqrt(mse / 4),
        "first10motor_mse": mse,
    }


def synthetic_inputs():
    records = {arm: {"calibration": [], "heldout": []} for arm in ARMS}
    pair_records = []
    for split, start, count, seeds in (
        ("calibration", 2000, 80, (0,)),
        ("heldout", 3000, 40, (0, 1)),
    ):
        for offset in range(count):
            case = start + offset
            suite = PROTOCOL["cohorts"]["suites"][offset // (count // 4)]
            for seed in seeds:
                noise = offset * 1e-5 + seed * 1e-6
                for arm in ARMS:
                    mse, latent = BASE[arm]
                    records[arm][split].append(
                        _record(PROTOCOL["protocol"], arm, split, case, seed, suite,
                                mse + noise, latent + noise)
                    )
                for anchor in ("Seq", "Joint"):
                    pair_records.append({
                        "anchor": anchor,
                        "split": split,
                        "case_id": case,
                        "seed_index": seed,
                        "suite": suite,
                        "action_max_abs": 1e-4 + noise,
                        "future_max_abs": 2e-4 + noise,
                        "action_nmse": 0.01 + noise,
                        "future_nmse": 0.02 + noise,
                    })
    return records, pair_records


def build(records=None, pair_records=None, provenance=None):
    if records is None or pair_records is None:
        fresh_records, fresh_pairs = synthetic_inputs()
        records = fresh_records if records is None else records
        pair_records = fresh_pairs if pair_records is None else pair_records
    return aggregate.build_comparison(PROTOCOL, records, pair_records, provenance or {})


class AggregateTests(unittest.TestCase):
    def test_historical_reference_is_not_reported_as_retraining_replay(self):
        records, pairs = synthetic_inputs()
        provenance = {'uncomp8_reference': 'historical_Joint8'}
        with self.assertRaisesRegex(ValueError, 'Historical Uncomp8'):
            aggregate.build_comparison(PROTOCOL, records, pairs, provenance)
        for rows in records['Uncomp8'].values():
            for row in rows:
                row['source_binding'] = dict(kind='historical_reuse', arm='Joint8', job='25912619.pbs101')
        result = aggregate.build_comparison(PROTOCOL, records, pairs, provenance)
        self.assertIsNone(result['replay_gate']['pass'])
        self.assertFalse(result['replay_gate']['available'])
        self.assertTrue(result['replay_gate']['identity_check_pass'])
        self.assertIn('未做同期Uncomp8重训', result['decision']['conclusion'])
        self.assertIn('不是同期训练replay通过', aggregate.render_markdown(result))

    def setUp(self):
        self.records, self.pair_records = synthetic_inputs()

    def test_aggregation_pair_counts_decision_and_compact_output(self):
        result = build(self.records, self.pair_records,
                       {"source_job": "synthetic", "training_history": ["omitted"] * 10000})
        held = result["heldout"]["Seq_A4"]
        self.assertEqual(held["queries"], 80)
        self.assertEqual(held["contexts"], 40)
        self.assertEqual(held["seed_indices"], [0, 1])
        self.assertEqual(result["calibration"]["Seq_A4"]["contexts"], 80)
        self.assertAlmostEqual(held["overall"]["first10motor_mse"], 1 + 0.0001955)
        counts = result["comparisons"]["Comp8_vs_Uncomp8"]["paired_context_win_loss"]
        self.assertEqual(counts["overall"]["mixed_loss"]["contexts"], 40)
        self.assertEqual(counts["by_suite"]["libero_spatial"]["mixed_loss"]["contexts"], 10)
        self.assertTrue(result["replay_gate"]["pass"])
        self.assertTrue(result["decision"]["primary_criteria_pass"])
        self.assertNotIn("training_history", result["provenance"])
        self.assertLess(len(json.dumps(result, ensure_ascii=False).encode("utf-8")), 256 * 1024)
        report = aggregate.render_markdown(result)
        self.assertIn("Heldout 配对比较", report)
        self.assertIn("native VQ", report)

    def test_rejects_missing_cohort(self):
        records = copy.deepcopy(self.records)
        records["Comp8"]["heldout"].pop()
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            build(records, self.pair_records)

    def test_rejects_wrong_seed(self):
        records = copy.deepcopy(self.records)
        records["Comp8"]["heldout"][0]["seed_index"] = 2
        with self.assertRaisesRegex(ValueError, "unexpected record identity"):
            build(records, self.pair_records)

    def test_rejects_duplicate_record(self):
        records = copy.deepcopy(self.records)
        records["Comp8"]["heldout"].append(copy.deepcopy(records["Comp8"]["heldout"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate record"):
            build(records, self.pair_records)

    def test_rejects_mixed_formula_mismatch(self):
        records = copy.deepcopy(self.records)
        records["Seq_A4"]["calibration"][0]["mixed_loss"] += 0.1
        with self.assertRaisesRegex(ValueError, "mixed_loss"):
            build(records, self.pair_records)

    def test_rejects_nonfinite_metric(self):
        records = copy.deepcopy(self.records)
        records["Joint_A4"]["heldout"][0]["future_latent_nmse"] = float("inf")
        records["Joint_A4"]["heldout"][0]["mixed_loss"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            build(records, self.pair_records)

    def test_rejects_suite_binding_mismatch(self):
        records = copy.deepcopy(self.records)
        records["Joint_A4"]["heldout"][0]["suite"] = "libero_object"
        with self.assertRaisesRegex(ValueError, "suite"):
            build(records, self.pair_records)

    def test_rejects_missing_pair_query(self):
        pair_records = copy.deepcopy(self.pair_records)
        pair_records.pop()
        with self.assertRaisesRegex(ValueError, "pair residual requires"):
            build(self.records, pair_records)

    def test_replay_failure_forces_inconclusive(self):
        records = copy.deepcopy(self.records)
        row = records["Uncomp8"]["heldout"][0]
        row["first10motor_mse"] += 1e-4
        row["first10motor_rmse"] = math.sqrt(row["first10motor_mse"])
        row["mixed_loss"] = row["first10motor_mse"] + aggregate.LATENT_WEIGHT * row["future_latent_nmse"]
        result = build(records, self.pair_records)
        self.assertFalse(result["replay_gate"]["pass"])
        self.assertIn("inconclusive", result["decision"]["conclusion"])

    def test_recovery_below_half_does_not_pass_primary_gate(self):
        records = copy.deepcopy(self.records)
        for split in ("calibration", "heldout"):
            for row in records["Comp8"][split]:
                mse = 1.11 + (row["case_id"] % 100) * 1e-5 + row["seed_index"] * 1e-6
                latent = mse
                row.update({
                    "first10motor_mse": mse,
                    "first10motor_rmse": math.sqrt(mse),
                    "future_latent_nmse": latent,
                    "mixed_loss": mse + aggregate.LATENT_WEIGHT * latent,
                    "full32motor_rmse": math.sqrt(mse + 0.2),
                    "late22motor_rmse": math.sqrt(mse + 0.1),
                    "first10gripper_rmse": math.sqrt(mse / 4),
                })
        result = build(records, self.pair_records)
        self.assertLess(result["decision"]["recovery_fraction"], 0.5)
        self.assertFalse(result["decision"]["primary_criteria_pass"])

    def test_sampler_seed_must_match_across_arms(self):
        records = copy.deepcopy(self.records)
        records["Comp8"]["heldout"][0]["sampler_seed"] += 1
        with self.assertRaisesRegex(ValueError, "sampler_seed assignments differ"):
            build(records, self.pair_records)

    def test_large_noa4_residual_is_reported_as_numeric_confound(self):
        records = copy.deepcopy(self.records)
        for row in records["Seq_to_Joint_noA4"]["heldout"]:
            row["first10motor_mse"] += 0.2
            row["first10motor_rmse"] = math.sqrt(row["first10motor_mse"])
            row["mixed_loss"] = row["first10motor_mse"] + aggregate.LATENT_WEIGHT * row["future_latent_nmse"]
            row["full32motor_rmse"] = math.sqrt(row["first10motor_mse"] + 0.2)
            row["late22motor_rmse"] = math.sqrt(row["first10motor_mse"] + 0.1)
            row["first10gripper_rmse"] = math.sqrt(row["first10motor_mse"] / 4)
        result = build(records, self.pair_records)
        gate = result["decision"]["geometry_attribution"]
        self.assertFalse(gate["pass"])
        self.assertIn("numeric confound", gate["status"])
        self.assertGreater(result["pair_residuals"]["Seq"]["heldout"]["metrics"]["action_max_abs"]["max"], 0)

    def test_nonpositive_gap_makes_recovery_undefined(self):
        records = copy.deepcopy(self.records)
        for arm in ("Joint_A4", "Uncomp8"):
            for split in ("calibration", "heldout"):
                for row in records[arm][split]:
                    row["first10motor_mse"] = 0.8
                    row["future_latent_nmse"] = 0.8
                    row["mixed_loss"] = 0.8 + aggregate.LATENT_WEIGHT * 0.8
                    row["first10motor_rmse"] = math.sqrt(0.8)
                    row["full32motor_rmse"] = math.sqrt(1.0)
                    row["late22motor_rmse"] = math.sqrt(0.9)
                    row["first10gripper_rmse"] = math.sqrt(0.2)
        result = build(records, self.pair_records)
        self.assertFalse(result["decision"]["recovery_defined"])
        self.assertIsNone(result["decision"]["recovery_fraction"])
        self.assertIn("gap≤0", aggregate.render_markdown(result))


if __name__ == "__main__":
    unittest.main()
