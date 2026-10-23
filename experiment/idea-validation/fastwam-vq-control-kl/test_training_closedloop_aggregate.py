"""CPU-allocation checks of paired success counts and unchanged denominators."""
import copy
import unittest

from control_common import guard
from training_closedloop_aggregate import summarize


def fixture():
    rows = []
    for suite in range(4):
        for dimension in range(7):
            for _ in range(10):
                rows.append({"index": len(rows), "suite": f"suite{suite}",
                             "dimension": f"dimension{dimension}"})
    candidate = {index: {"success": index % 2 == 0} for index in range(280)}
    references = {
        "all_fail": {index: {"success": False} for index in range(280)},
        "opposite": {index: {"success": index % 2 == 1} for index in range(280)},
    }
    return rows, candidate, references


class PairedSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard()

    def test_full_counts_and_paired_wins_losses(self):
        result = summarize(*fixture())
        overall = result["overall"]
        self.assertEqual(overall["successes"], 140)
        self.assertEqual(overall["episodes"], 280)
        self.assertEqual(overall["paired_vs"]["all_fail"]["wins"], 140)
        self.assertEqual(overall["paired_vs"]["all_fail"]["losses"], 0)
        self.assertEqual(overall["paired_vs"]["all_fail"]["ties"], 140)
        self.assertEqual(overall["paired_vs"]["opposite"]["losses"], 140)
        self.assertEqual(overall["paired_vs"]["opposite"]["ties"], 0)
        self.assertEqual(overall["paired_vs"]["opposite"]["net_successes"], 0)
        self.assertTrue(all(value["episodes"] == 70 for value in result["by_suite"].values()))
        self.assertTrue(all(value["episodes"] == 40 for value in result["by_dimension"].values()))
        self.assertTrue(all(value["episodes"] == 10 for suite in result["by_suite_dimension"].values()
                            for value in suite.values()))

    def test_incomplete_or_changed_cohort_is_rejected(self):
        rows, candidate, references = fixture()
        missing = dict(candidate)
        missing.pop(279)
        with self.assertRaises(ValueError):
            summarize(rows, missing, references)
        bad_reference = copy.deepcopy(references)
        bad_reference["all_fail"].pop(0)
        with self.assertRaises(ValueError):
            summarize(rows, candidate, bad_reference)
        changed = copy.deepcopy(rows)
        changed[0]["dimension"] = "other"
        with self.assertRaises(ValueError):
            summarize(changed, candidate, references)
        nonboolean = copy.deepcopy(candidate)
        nonboolean[0]["success"] = 1
        with self.assertRaises(ValueError):
            summarize(rows, nonboolean, references)


if __name__ == "__main__":
    unittest.main()
