import unittest

import numpy as np

import original_projection as diagnostic


class OriginalProjectionTests(unittest.TestCase):
    def test_old_probes_can_miss_a_real_error(self):
        probes = np.eye(60, dtype=np.float64)[:4]
        delta = np.zeros(60, dtype=np.float64)
        delta[10] = 1.0
        self.assertEqual(diagnostic.projected_values(delta, probes).tolist(), [0.0] * 4)
        self.assertAlmostEqual(diagnostic.motor_rmse(delta), 1.0 / np.sqrt(60.0))
        self.assertEqual(diagnostic.span_energy_fraction(delta, probes), (0.0, 4))

    def test_nonorthogonal_probe_span_uses_orthonormal_basis(self):
        probes = np.zeros((2, 60), dtype=np.float64)
        probes[0, 0] = 1.0
        probes[1, 0:2] = 1.0
        delta = np.zeros(60, dtype=np.float64)
        delta[0] = 1.0
        delta[2] = 2.0
        fraction, rank = diagnostic.span_energy_fraction(delta, probes)
        self.assertEqual(rank, 2)
        self.assertAlmostEqual(fraction, 1.0 / 5.0)

    def test_projection_and_motor_rmse_use_their_own_normalization(self):
        delta = np.ones(60, dtype=np.float64)
        probes = np.ones((4, 60), dtype=np.float64) / np.sqrt(60.0)
        self.assertAlmostEqual(diagnostic.motor_rmse(delta), 1.0)
        self.assertAlmostEqual(diagnostic.rms(diagnostic.projected_values(delta, probes)), np.sqrt(60.0))

    def test_calibration_coverage_requires_every_case_arm_pair(self):
        cases = [0, 2]
        arms = ["row_scale_vq_a4", "row_output_vq_a4"]
        good = {(case, arm): {} for case in cases for arm in arms}
        diagnostic.validate_coverage(good, cases, arms)
        missing = dict(good)
        del missing[(2, "row_output_vq_a4")]
        with self.assertRaises(ValueError):
            diagnostic.validate_coverage(missing, cases, arms)


if __name__ == "__main__":
    diagnostic.guard()
    unittest.main()
