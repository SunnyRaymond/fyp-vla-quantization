"""Small CPU-allocation checks of original heldout coverage and saved endpoints."""
import copy
import unittest

from control_common import guard
import training_evaluation as evaluation


def parent():
    return {"source_relational_protocol": {"evaluation": {
        "case_ids": list(range(3000, 3040)), "seed_indices": [0, 1]}}}


def records():
    return [{"case_id": case, "seed_index": seed,
             "suite": evaluation.SUITES[(case - 3000) // 10],
             **{key: float(seed + 1) for key in evaluation.METRICS}}
            for case, seed in evaluation.heldout_queries(parent())]


class TrainingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard()
        import torch
        cls.torch = torch

    def test_both_shards_form_exact_original_80_queries(self):
        left = evaluation.heldout_queries(parent(), 0)
        right = evaluation.heldout_queries(parent(), 1)
        self.assertEqual(len(left), 40)
        self.assertEqual(len(right), 40)
        self.assertFalse(set(left) & set(right))
        self.assertEqual(set(left + right), set(evaluation.heldout_queries(parent())))
        modified = parent()
        modified["source_relational_protocol"]["evaluation"]["seed_indices"] = [1, 2]
        with self.assertRaises(ValueError):
            evaluation.heldout_queries(modified)

    def test_equal_case_suite_aggregation_requires_full_paired_cohort(self):
        rows = records()
        result = evaluation.aggregate_records(parent(), rows)
        self.assertEqual(result["queries"], 80)
        self.assertEqual(result["cases"], 40)
        self.assertEqual(result["overall"]["first10motor_rmse"], 1.5)
        for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                evaluation.aggregate_records(parent(), invalid)
        bad = copy.deepcopy(rows)
        bad[0]["suite"] = evaluation.SUITES[1]
        with self.assertRaises(ValueError):
            evaluation.aggregate_records(parent(), bad)
        bad = copy.deepcopy(rows)
        bad[0]["action_kl"] = float("nan")
        with self.assertRaises(ValueError):
            evaluation.aggregate_records(parent(), bad)

    def test_endpoint_binds_checkpoint_input_and_full_finite_tensors(self):
        torch = self.torch
        binding = {"split": "heldout", "case_id": 3000, "seed_index": 0,
                   "deployment_binding": {"arm": "toy", "producer_job": "123.pbs"},
                   "input_identity": {"image": "toy-image", "proprio": "toy-proprio"}}
        saved = {**binding, "counts": dict(evaluation.COUNTS),
                 "action": torch.zeros(32, 7), "future": torch.zeros(1, 48, 3, 14, 28),
                 "action_initial_noise": torch.zeros(1, 32, 7),
                 "commands": torch.zeros(32, 7), "teacher_commands": torch.zeros(32, 7)}
        self.assertIs(evaluation.validate_endpoint(torch, saved, binding), saved)
        for field, value in (("action", torch.zeros(10, 6)),
                             ("future", torch.full((1, 48, 3, 14, 28), float("nan"))),
                             ("counts", {"video_steps": 10, "action_steps": 9}),
                             ("teacher_commands", torch.zeros(10, 7)),
                             ("input_identity", {"image": "foreign"}),
                             ("deployment_binding", {"arm": "other"})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluation.validate_endpoint(torch, {**saved, field: value}, binding)


if __name__ == "__main__":
    unittest.main()
