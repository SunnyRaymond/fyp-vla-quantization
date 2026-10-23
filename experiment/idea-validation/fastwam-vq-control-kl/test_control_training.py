"""CPU-allocation tests for training config, deadline, resume, and atomic save."""
import os
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from control_common import Deadline, guard
import control_training as worker


CASE_ORDER = list(range(2000, 2080))


def _config():
    parent = {
        "protocol": "toy-parent",
        "source_closed_protocol": {"calibration": {"update_order": list(CASE_ORDER)}},
    }
    data = {"protocol": "toy-data", "parent_protocol": "toy-parent"}
    gate = {
        "protocol": "toy-gate", "fractions": [0.1],
        "objective_keys": {"response": "action_response_kl", "latent": "latent_control_kl"},
    }
    training = {
        "protocol": worker.TRAINING_PROTOCOL,
        "parent_protocol": "toy-parent",
        "data_config": data,
        "checkpoint_arm": "per_channel_clip",
        "update_order": list(CASE_ORDER),
        "source_jobs": {
            "training_teacher": "1001.server",
            "training_loss_selfcheck": "1002.server",
            "aggregate_c80_parameter_gate": "1003.server",
        },
        "arms": {"toy": {
            "action_absolute_weight": 1.0,
            "response_weight": 0.0,
            "latent_absolute_weight": 1.0,
            "control_weight": 0.0,
            "lambda_latent": 1.0,
        }},
        "optimizer": {
            "betas": [0.9, 0.999], "eps": 1e-8,
            "gradient_norm_clip": 1.0, "reset_after_update1": True,
            "gate_fraction": 0.1,
        },
    }
    return parent, data, gate, training


class ControlTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard()
        import torch
        cls.torch = torch

    def test_explicit_config_and_gate_fraction(self):
        parent, data, gate, config = _config()
        loss, optimizer = worker.validate_training_protocol(parent, data, config, "toy", gate)
        self.assertEqual(loss["action_absolute_weight"], 1.0)
        self.assertEqual(optimizer["gate_fraction"], 0.1)
        bad = {**config, "optimizer": {**config["optimizer"], "gate_fraction": 0.2}}
        with self.assertRaises(ValueError):
            worker.validate_training_protocol(parent, data, bad, "toy", gate)
        bad_order = {**config, "update_order": list(reversed(CASE_ORDER))}
        with self.assertRaises(ValueError):
            worker.validate_training_protocol(parent, data, bad_order, "toy", gate)

    def test_prerequisite_receipts_require_both_full_c80_gate_descents(self):
        parent, _data, gate_config, config = _config()
        receipts = {
            "training_teacher": {
                "protocol": "toy-parent", "unit": "training_teacher",
                "pbs_jobid": "1001.server", "complete": True,
                "expected_records": 160, "completed_records": 160, "contexts": 80,
                "planned_new_teacher_queries": 48, "unique_new_teacher_outputs": 48,
                "total_new_query_completed": 48, "new_teacher_queries": 48,
                "total_new_query_attempted": 48, "successful_new_teacher_queries": 48,
                "initial_noise_parity_valid": True, "initial_noise_parity_contexts": 80,
                "frozen_model_unchanged": True, "training_updates": 0,
                "heldout_inputs_used": False, "evaluation_trajectory_inputs_used": False,
            },
            "training_loss_selfcheck": {
                "protocol": "toy-parent", "unit": "training_loss_selfcheck",
                "pbs_jobid": "1002.server", "complete": True,
                "passed_checks": ["test_training_objective.py"],
                "model_queries": 0, "training_updates": 0,
            },
            "aggregate_c80_parameter_gate": {
                "protocol": "toy-parent", "unit": "aggregate_c80_parameter_gate",
                "pbs_jobid": "1003.server", "complete": True,
                "gate_protocol": "toy-gate", "full_c80_finite_step_evaluated": True,
                "training_updates": 0, "heldout_inputs_used": False,
                "sensor_evaluation_inputs_used": False,
                "objectives": {
                    objective: {"cases": 80, "candidates": [
                        {"fraction": 0.1, "cases": 80, "actual_descent": True}
                    ]}
                    for objective in ("action_response_kl", "latent_control_kl")
                },
            },
        }
        self.assertTrue(worker.validate_prerequisite_receipts(
            parent, config, gate_config, receipts))
        for field, value in (("actual_descent", False), ("cases", 79), ("fraction", 0.2)):
            bad = copy.deepcopy(receipts)
            bad["aggregate_c80_parameter_gate"]["objectives"]["latent_control_kl"]["candidates"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.validate_prerequisite_receipts(parent, config, gate_config, bad)
        foreign = copy.deepcopy(receipts)
        foreign["training_teacher"]["pbs_jobid"] = "other.server"
        with self.assertRaises(ValueError):
            worker.validate_prerequisite_receipts(parent, config, gate_config, foreign)
        foreign_gate = copy.deepcopy(receipts)
        foreign_gate["aggregate_c80_parameter_gate"]["pbs_jobid"] = "other.server"
        with self.assertRaises(ValueError):
            worker.validate_prerequisite_receipts(parent, config, gate_config, foreign_gate)

    def test_resume_requires_full_bindings_and_exact_history_prefix(self):
        contract = {
            "protocol": {"protocol": "toy-parent"}, "unit": "train_toy",
            "training_protocol": {"protocol": worker.TRAINING_PROTOCOL},
            "data_config": {"protocol": "toy-data"},
            "source_jobs": {"training_teacher": "1001.server"},
            "teacher_bindings": {"2000:clean": {"input_identity": "toy-id"}},
        }
        checkpoint = {
            **contract, "update_order": CASE_ORDER, "update_index": 2,
            "history": [{"update": 1, "case_id": CASE_ORDER[0]},
                        {"update": 2, "case_id": CASE_ORDER[1]}],
            "ratios": {"toy": self.torch.tensor([1.0])},
            "shadow_log2_scales": {"toy": self.torch.tensor([0.0])},
            "optimizer": {"state": {}}, "rng": {"cpu": self.torch.tensor([1])},
        }
        self.assertEqual(worker.validate_resume_checkpoint(checkpoint, contract, CASE_ORDER), 2)
        drifted = {**checkpoint, "source_jobs": {"training_teacher": "other.server"}}
        with self.assertRaises(ValueError):
            worker.validate_resume_checkpoint(drifted, contract, CASE_ORDER)
        reordered = {**checkpoint, "history": list(reversed(checkpoint["history"]))}
        with self.assertRaises(ValueError):
            worker.validate_resume_checkpoint(reordered, contract, CASE_ORDER)

    def test_real_deadline_near_api(self):
        with patch.dict(os.environ, {"FW_JOB_START_EPOCH": "0"}):
            deadline = Deadline(6600)
            self.assertTrue(deadline.near(300))

    def test_atomic_torch_checkpoint_roundtrip(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            value = {"update_index": 3, "tensor": torch.tensor([1.25, -2.0])}
            worker._atomic_torch_save(torch, path, value)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["update_index"], 3)
            self.assertTrue(torch.equal(loaded["tensor"], value["tensor"]))

    def test_bf16_future_cotangents_combine_in_endpoint_dtype(self):
        torch = self.torch
        leaf = torch.linspace(-0.8, 0.9, 16, dtype=torch.bfloat16).requires_grad_(True)
        absolute = 0.17 * leaf.float().square().mean()
        control = 0.31 * leaf.float().sin().sum()
        expected, = torch.autograd.grad(absolute + control, leaf)
        first = leaf.detach().clone().requires_grad_(True)
        second = leaf.detach().clone().requires_grad_(True)
        absolute_vector, = torch.autograd.grad(0.17 * first.float().square().mean(), first)
        control_vector, = torch.autograd.grad(0.31 * second.float().sin().sum(), second)
        combined = absolute_vector + control_vector
        self.assertEqual(combined.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(expected, combined))


if __name__ == "__main__":
    guard()
    unittest.main()
