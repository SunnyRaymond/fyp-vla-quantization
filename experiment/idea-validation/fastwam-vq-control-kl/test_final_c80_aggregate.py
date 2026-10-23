"""CPU-only tests of full-C80 reduction and the paired reader coefficient."""
import unittest

from control_common import guard
from final_c80_aggregate import compose, summarize, SUITES, RAW_KEYS, METRIC_KEYS


CONFIG = {"action_absolute_weight": 2.0, "response_weight": 3.0,
          "latent_absolute_weight": 4.0, "control_weight": 5.0, "lambda_latent": 0.1}
RAW = dict(zip(RAW_KEYS, (1.0, 2.0, 3.0, 4.0, 6.0)))


class FinalC80SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard()

    def test_joint_formula_averages_both_reader_variants_once(self):
        self.assertAlmostEqual(compose(RAW, CONFIG), 11.7)
        absolute = {**CONFIG, "response_weight": 0.0, "control_weight": 0.0}
        self.assertAlmostEqual(compose(RAW, absolute), 3.2)
        with self.assertRaises(ValueError):
            compose({**RAW, "action_response_kl": float("nan")}, CONFIG)

    def test_original_80_cases_and_equal_suite_reduction(self):
        rows = [{"case_id": case, "suite": SUITES[(case - 2000) // 20],
                 **RAW, **{key: (case - 2000) // 20 + 1 for key in METRIC_KEYS}}
                for case in range(2000, 2080)]
        result = summarize(rows, {"joint": CONFIG})
        self.assertEqual(result["cases"], 80)
        self.assertEqual(result["views_per_case"], 2)
        self.assertEqual(result["overall"]["clean_motor_rmse"], 2.5)
        self.assertAlmostEqual(result["overall"]["objective_by_loss_config"]["joint"], 11.7)
        for invalid in (rows[:-1], rows[:-1] + [rows[0]], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize(invalid, {"joint": CONFIG})
        changed = [{**row, "suite": SUITES[0]} for row in rows]
        with self.assertRaises(ValueError):
            summarize(changed, {"joint": CONFIG})


if __name__ == "__main__":
    unittest.main()
