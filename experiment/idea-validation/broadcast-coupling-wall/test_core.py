import unittest

import numpy as np

from core import (
    ARMS,
    HORIZONS,
    PATCHES,
    TAIL_DIM,
    prepare_initial_treatments,
    quantize_tail,
    requested_horizon,
    summarize_records,
)


class CoreContractTest(unittest.TestCase):
    def test_only_requested_horizons_are_recorded(self):
        self.assertTrue(requested_horizon(1))
        self.assertFalse(requested_horizon(2))
        self.assertFalse(requested_horizon(3))
        self.assertFalse(requested_horizon(4))
        self.assertTrue(requested_horizon(5))

    def test_three_draw_matched_broadcast(self):
        tail = np.linspace(-1.0, 1.0, PATCHES * TAIL_DIM, dtype=np.float32).reshape(PATCHES, TAIL_DIM)
        first = prepare_initial_treatments(tail, 910001, 130)
        second = prepare_initial_treatments(tail, 910001, 130)
        for arm in ARMS:
            np.testing.assert_array_equal(first[arm]["values"], second[arm]["values"])
        shared = first["shared_stochastic"]["values"]
        balanced = first["balanced_broadcast"]["values"]
        np.testing.assert_array_equal(np.sort(shared, axis=0), np.sort(balanced, axis=0))
        self.assertEqual(first["balanced_broadcast"]["audit"]["n_draws"], 3)
        self.assertTrue(first["balanced_broadcast"]["audit"]["shared_balanced_mse_equal"])
        self.assertTrue(first["balanced_broadcast"]["audit"]["shared_balanced_multiset_equal"])
        self.assertFalse(first["iid_patch_stochastic"]["audit"]["matched_reference"])

    def test_quantizer_stochastic_scheme_is_assigned(self):
        tail = np.ones((PATCHES, TAIL_DIM), dtype=np.float32)
        for arm in ("shared_stochastic", "iid_patch_stochastic", "balanced_broadcast"):
            # Balanced is intentionally only available through the paired
            # constructor; quantize_tail must reject the old LHS accident.
            if arm == "balanced_broadcast":
                continue
            output, audit = quantize_tail(tail, arm, np.random.default_rng(3))
            self.assertEqual(output.shape, tail.shape)
            self.assertTrue(audit["scheme"].startswith("symmetric_A4_stochastic"))

    def test_summary_contract(self):
        records = []
        for episode in (130, 131):
            for horizon in HORIZONS:
                arms = {
                    arm: {"real_future_feature_mse": float(episode + horizon), "fp_reference_mse": 0.0}
                    for arm in ARMS
                }
                from core import make_state_record
                records.append(make_state_record(
                    episode_index=episode,
                    trajectory_id=f"WallDataset:test#{episode}",
                    current_frame=0,
                    future_frame=horizon * 5,
                    horizon=horizon,
                    arm_outputs=arms,
                ))
        summary = summarize_records(records, (130, 131))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len(summary["rows"]), len(ARMS) * len(HORIZONS))


if __name__ == "__main__":
    unittest.main()
