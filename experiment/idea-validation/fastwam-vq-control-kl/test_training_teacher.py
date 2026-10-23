"""CPU-only tensor checks for teacher-cache binding and reuse rules."""
import unittest
from types import SimpleNamespace

import training_teacher as worker
from control_common import guard


def _record(torch):
    return {
        "counts": dict(worker.COUNTS),
        "action": torch.zeros(worker.ACTION_SHAPE, dtype=torch.float32, device="cpu"),
        "future": torch.zeros(worker.FUTURE_SHAPE, dtype=torch.float32, device="cpu"),
        "action_initial_noise": torch.zeros(worker.NOISE_SHAPE, dtype=torch.float32, device="cpu"),
    }


class TrainingTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def setUp(self):
        self.binding = {
            "protocol": "fastwam-vq-control-kl-v1", "unit": "training_teacher",
            "index": 2000, "arm": "bf16", "variant": "noisy", "seed": 760000,
            "description": "toy", "input_identity": {"image": "x", "proprio": "y"},
        }
        self.record = {**self.binding, **_record(self.torch)}
        self.metadata = {
            **self.binding, "counts": dict(worker.COUNTS),
            "action_shape": list(worker.ACTION_SHAPE),
            "future_shape": list(worker.FUTURE_SHAPE),
            "action_initial_noise_shape": list(worker.NOISE_SHAPE),
        }

    def test_cached_payload_binding_shape_finiteness_and_provenance(self):
        outputs = worker._validate_bound_record(
            self.torch, self.record, self.metadata, self.binding)
        self.assertEqual(tuple(outputs["action"].shape), worker.ACTION_SHAPE)

        drift = dict(self.metadata, seed=self.binding["seed"] + 1)
        with self.assertRaises(ValueError):
            worker._validate_bound_record(self.torch, self.record, drift, self.binding)
        bad_shape = {**self.record, "action": self.torch.zeros((1, 1))}
        with self.assertRaises(ValueError):
            worker._validate_bound_record(self.torch, bad_shape, self.metadata, self.binding)
        bad_finite = {**self.record, "future": self.record["future"].clone()}
        bad_finite["future"][0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            worker._validate_bound_record(self.torch, bad_finite, self.metadata, self.binding)

        eligibility = {
            "clean_teacher_reuse_eligible": True,
            "noisy_observations_exactly_equal": False,
            "noisy_teacher_reuse_eligible": False,
            "new_noisy_teacher_query_required": True,
        }
        extras = {
            "data_config": {"protocol": "control-kl-training-data-v1"},
            "teacher_plan_pbs_jobid": "25895162.pbs101",
            "source_calibration_job": "25894124.pbs101",
            "producer_job": "25899999.pbs101",
            "source_file": None, "source_eligibility": eligibility,
            "source_kind": "new_query",
        }
        payload, metadata = {**self.record, **extras}, {**self.metadata, **extras}
        worker._validate_training_record(
            self.torch, payload, metadata, self.binding, extras["data_config"],
            extras["teacher_plan_pbs_jobid"], extras["source_calibration_job"],
            None, eligibility)
        with self.assertRaises(ValueError):
            worker._validate_training_record(
                self.torch, payload, {**metadata, "producer_job": "25888888.pbs101"},
                self.binding, extras["data_config"], extras["teacher_plan_pbs_jobid"],
                extras["source_calibration_job"], None, eligibility)

    def test_reuse_eligibility_is_explicit(self):
        row = {
            "case_id": 2000, "clean_teacher_reuse_eligible": True,
            "clean_teacher_source": "clean.pt", "noisy_observations_exactly_equal": True,
            "noisy_teacher_reuse_eligible": True, "noisy_teacher_source": "noisy.pt",
            "new_noisy_teacher_query_required": False,
        }
        self.assertEqual(worker._reuse_source(row, "clean")[0], "clean.pt")
        self.assertEqual(worker._reuse_source(row, "noisy")[0], "noisy.pt")
        row.update(noisy_observations_exactly_equal=False,
                   noisy_teacher_reuse_eligible=False, noisy_teacher_source=None,
                   new_noisy_teacher_query_required=True)
        self.assertIsNone(worker._reuse_source(row, "noisy")[0])
        row["new_noisy_teacher_query_required"] = False
        with self.assertRaises(ValueError):
            worker._reuse_source(row, "noisy")

    def test_clean_noisy_noise_equality_is_required(self):
        noise = self.torch.zeros(worker.NOISE_SHAPE, dtype=self.torch.float32, device="cpu")
        self.assertTrue(worker._noise_equal(self.torch, noise, noise.clone()))
        changed = noise.clone()
        changed[0, 0, 0] = 1.0
        with self.assertRaises(RuntimeError):
            worker._noise_equal(self.torch, noise, changed)

    def test_frozen_check_accepts_scalar_bank_parameters(self):
        torch = self.torch
        model = torch.nn.Linear(1, 1, device="cpu")
        model.requires_grad_(False)
        scalar = torch.nn.Parameter(torch.tensor(1.0, device="cpu"), requires_grad=False)
        banks = SimpleNamespace(books={"book": scalar}, logs={},
                                ratios={"ratio": torch.nn.Parameter(
                                    torch.tensor(1.0, device="cpu"), requires_grad=False)},
                                shadows={})
        switching = SimpleNamespace(runtime=SimpleNamespace(model=model), banks=banks)
        self.assertTrue(worker._all_frozen(switching))
        banks.ratios["ratio"].requires_grad_(True)
        self.assertFalse(worker._all_frozen(switching))

    def test_retry_attempts_require_repair_link_and_remain_counted(self):
        progress = {"query_attempts": []}
        worker._start_query_attempt(progress, "2000:noisy", "job-a", 48)
        with self.assertRaises(RuntimeError):
            worker._start_query_attempt(progress, "2000:noisy", "job-a", 48)
        with self.assertRaises(RuntimeError):
            worker._start_query_attempt(progress, "2000:noisy", "job-b", 48, "other-job")
        worker._start_query_attempt(progress, "2000:noisy", "job-b", 48, "job-a")
        self.assertEqual(len(progress["query_attempts"]), 2)
        self.assertEqual(progress["query_attempts"][0]["status"],
                         "retried_after_terminal_repair")
        worker._complete_query_attempt(progress, "2000:noisy", "job-b")
        self.assertEqual(progress["query_attempts"][1]["status"], "completed")
        progress["new_query_completed_total"] = 1

        summary = worker._new_summary("fastwam-vq-control-kl-v1", "job-b")
        worker._sync_counts(
            summary, progress,
            {"new_query_attempted": 1, "new_query_completed": 1,
             "reuse_calibration": 0, "reuse_existing": 0},
            records_seen=1, planned_queries=48)
        self.assertEqual(summary["new_model_queries"], 2)
        self.assertEqual(summary["new_teacher_queries"], 2)
        self.assertEqual(summary["successful_new_teacher_queries"], 1)
        self.assertEqual(summary["unique_new_teacher_outputs"], 1)


if __name__ == "__main__":
    guard()
    unittest.main()
