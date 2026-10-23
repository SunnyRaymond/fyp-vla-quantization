import unittest

from metrics import aggregate, frozen_contexts


class MetricsTests(unittest.TestCase):
    def test_aggregate_weights_cases_equally_after_averaging_seeds(self):
        rows = [
            {"case_id": "a", "seed_index": 0, "split": "test", "arm": "ref", "rmse": 0.0},
            {"case_id": "a", "seed_index": 1, "split": "test", "arm": "ref", "rmse": 2.0},
            {"case_id": "b", "seed_index": 0, "split": "test", "arm": "ref", "rmse": 10.0},
        ]
        result = aggregate(rows)[0]
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["context_count"], 3)
        self.assertEqual(result["metrics"]["rmse"], 5.5)
        self.assertEqual([row["metrics"]["rmse"] for row in result["per_case"]], [1.0, 10.0])
        self.assertNotEqual(result["metrics"]["rmse"], sum(row["rmse"] for row in rows) / len(rows))

    def test_aggregate_accepts_arm_specific_metrics_and_ignores_metadata(self):
        rows = [
            {"case_id": 1, "seed_index": 0, "split": "test", "arm": "parent_self",
             "domain": "cube", "dimension": "latent", "latent_shape": [1, 2, 3],
             "future_rmse": [0.2, 0.4], "future_latent_rmse": 0.3, "future_latent_nmse": 0.5},
            {"case_id": 1, "seed_index": 0, "split": "test", "arm": "distilled_self",
             "domain": "cube", "dimension": "latent", "latent_shape": [1, 2, 3],
             "future_rmse": [0.1, 0.3], "first10motor_rmse": 0.25},
        ]
        result = aggregate(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["metrics"], {"future_latent_rmse": 0.3, "future_latent_nmse": 0.5})
        self.assertEqual(result[1]["metrics"], {"first10motor_rmse": 0.25})
        self.assertEqual(aggregate([]), [])

    def test_frozen_contexts_cover_60_pairs_with_30_per_shard(self):
        protocol = {
            "calibration_cases": list(range(8)),
            "selection_cases": list(range(8, 12)),
            "test_cases": list(range(12, 34)),
        }
        contexts = frozen_contexts(protocol)
        pairs = {(row["split"], row["case_id"], row["seed_index"]) for row in contexts}
        self.assertEqual(len(contexts), 60)
        self.assertEqual(len(pairs), 60)
        self.assertEqual([sum(row["shard"] == shard for row in contexts) for shard in (0, 1)], [30, 30])
        for shard in (0, 1):
            rows = [row for row in contexts if row["shard"] == shard]
            self.assertEqual({split: sum(row["split"] == split for row in rows)
                              for split in ("calibration", "selection", "test")},
                             {"calibration": 4, "selection": 4, "test": 22})

    def test_frozen_contexts_reject_duplicate_case_ids(self):
        protocol = {
            "calibration_cases": [0, 1, 2, 3, 4, 5, 6, 7],
            "selection_cases": [7, 9, 10, 11],
            "test_cases": list(range(12, 34)),
        }
        with self.assertRaises(ValueError):
            frozen_contexts(protocol)


if __name__ == "__main__":
    unittest.main()
