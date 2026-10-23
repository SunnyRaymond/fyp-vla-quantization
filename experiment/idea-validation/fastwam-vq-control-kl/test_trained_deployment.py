"""Allocation-guarded CPU tests for the training deployment installer."""
import copy
import unittest


class FakeBanks:
    def __init__(self, torch):
        self.books = {name: torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
                      for name in ("a", "b")}
        self.logs = {name: torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
                     for name in ("a", "b")}
        self.ratios = {name: torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
                       for name in ("a", "b")}
        self.shadows = {name: torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
                        for name in ("a", "b")}

    def restore_clips(self, values):
        for name, value in values.items():
            self.ratios[name].data.copy_(value)

    def restore_scales(self, values):
        for name, value in values.items():
            self.shadows[name].data.copy_(value)


class FakeSwitching:
    def __init__(self, torch, parent):
        self.torch = torch
        self.p = parent
        self.banks = FakeBanks(torch)
        self.runtime = type("Runtime", (), {})()
        self.runtime.model = torch.nn.Linear(1, 1, bias=False)
        self.runtime.model.weight.requires_grad_(False)
        self.original_weights = {name: torch.tensor([[2.0]]) for name in ("a", "b")}
        self.decoded_base_weights = {name: value.clone()
                                     for name, value in self.original_weights.items()}
        self.weights = {name: value.clone() for name, value in self.original_weights.items()}
        self.deployments = {"per_channel_clip": {"ratios": {}, "log2_scales": {}}}
        self.deployment_bindings = {}
        self.active_arm = "bf16"

    def activate(self, arm):
        if arm == "bf16":
            self.weights = {name: value.clone()
                            for name, value in self.decoded_base_weights.items()}
        else:
            saved = self.deployments[arm]
            self.banks.restore_clips(saved["ratios"])
            self.banks.restore_scales(saved["log2_scales"])
            self.weights = {
                name: self.decoded_base_weights[name] * self.torch.pow(
                    2.0, self.banks.shadows[name].detach())
                for name in self.decoded_base_weights
            }
        self.active_arm = arm


class TrainedDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from control_common import guard
        guard()
        import torch
        import trained_deployment
        cls.torch = torch
        cls.installer = staticmethod(trained_deployment.install_training_deployment)

    def setUp(self):
        self.parent = {
            "protocol": "parent-protocol-v1",
            "source_closed_protocol": {
                "clipping": {"minimum_ratio": 0.5, "maximum_ratio": 1.5},
                "scaling": {"minimum_log2": -1.0, "maximum_log2": 1.0},
                "calibration": {"update_order": list(range(2000, 2080))},
            },
        }
        self.config = {
            "protocol": "control-kl-training-v1",
            "parent_protocol": self.parent["protocol"],
            "data_config": {"protocol": "training-data-v1"},
            "checkpoint_arm": "per_channel_clip",
            "update_order": list(range(2000, 2080)),
            "source_jobs": {
                "training_teacher": "teacher-1.pbs",
                "training_loss_selfcheck": "loss-1.pbs",
                "aggregate_c80_parameter_gate": "gate-1.pbs",
            },
            "arms": {"candidate": {"explicit": True}},
            "optimizer": {"gate_fraction": 0.25},
        }
        self.jobs = {
            **self.config["source_jobs"],
            "calibration_0": "cal0.pbs",
            "calibration_1": "cal1.pbs",
            "training_pairs": "pairs.pbs",
            "training_teacher_plan": "plan.pbs",
            "training_runtime_selfcheck": "runtime.pbs",
        }
        self.deployment_path = "/out/deployment.pt"
        self.reference_path = "/out/reference_endpoint.pt"
        self.switching = FakeSwitching(self.torch, self.parent)
        self.deployment = {
            "protocol": copy.deepcopy(self.parent),
            "training_protocol": copy.deepcopy(self.config),
            "data_config": copy.deepcopy(self.config["data_config"]),
            "unit": "train_candidate", "arm": "candidate",
            "checkpoint_arm": "per_channel_clip", "updates": 80,
            "update_order": list(self.config["update_order"]),
            "source_jobs": dict(self.jobs),
            "reference_endpoint": self.reference_path,
            "ratios": {name: self.torch.tensor(1.25) for name in ("a", "b")},
            "log2_scales": {name: self.torch.tensor(0.5) for name in ("a", "b")},
        }
        self.summary = {
            "protocol": self.parent["protocol"],
            "training_protocol": copy.deepcopy(self.config),
            "data_config": copy.deepcopy(self.config["data_config"]),
            "unit": "train_candidate", "arm": "candidate",
            "complete": True, "status": "complete",
            "completed_updates": 80, "training_updates": 80,
            "source_jobs": dict(self.jobs),
            "deployment": self.deployment_path,
            "reference_endpoint": self.reference_path,
            "pbs_jobid": "final-8.pbs",
            "this_job": {"pbs_jobid": "final-8.pbs"},
            "gate_fraction": 0.25,
            "native_cached_parity": {
                "action_max_abs": 0.0, "future_max_abs": 0.0,
                "initial_noise_max_abs": 0.0,
            },
        }

    def run_install(self, deployment=None, summary=None, config=None, path=None, arm="candidate"):
        return self.installer(
            self.switching, self.parent, config or self.config, arm,
            deployment or self.deployment, summary or self.summary,
            path or self.deployment_path)

    def test_install_and_reactivation_use_immutable_base(self):
        binding = self.run_install()
        self.assertEqual(binding["producer_job"], "final-8.pbs")
        self.assertEqual(binding["source_jobs"], self.jobs)
        expected = self.torch.tensor([[2.0 * (2.0 ** 0.5)]])
        self.assertTrue(self.torch.equal(self.switching.weights["a"], expected))
        self.switching.activate("bf16")
        self.switching.activate("per_channel_clip")
        first = self.switching.weights["a"].clone()
        self.switching.activate("per_channel_clip")
        self.assertTrue(self.torch.equal(first, self.switching.weights["a"]))
        self.assertTrue(self.torch.equal(self.switching.banks.ratios["a"].detach(),
                                         self.deployment["ratios"]["a"]))

    def assert_rejected_without_install(self, *args, **kwargs):
        original_slot = self.switching.deployments["per_channel_clip"]
        with self.assertRaises((ValueError, RuntimeError)):
            self.installer(*args, **kwargs)
        self.assertIs(self.switching.deployments["per_channel_clip"], original_slot)

    def test_rejects_foreign_arm_source_and_paths(self):
        foreign_config = copy.deepcopy(self.config)
        foreign_config["data_config"]["protocol"] = "foreign"
        self.assert_rejected_without_install(
            self.switching, self.parent, foreign_config, "candidate",
            self.deployment, self.summary, self.deployment_path)
        wrong_arm = copy.deepcopy(self.deployment)
        wrong_arm["arm"] = "other"
        self.assert_rejected_without_install(
            self.switching, self.parent, self.config, "candidate",
            wrong_arm, self.summary, self.deployment_path)
        wrong_jobs = copy.deepcopy(self.deployment)
        wrong_jobs["source_jobs"]["training_pairs"] = "foreign.pbs"
        self.assert_rejected_without_install(
            self.switching, self.parent, self.config, "candidate",
            wrong_jobs, self.summary, self.deployment_path)
        wrong_path = copy.deepcopy(self.summary)
        wrong_path["deployment"] = "/out/other.pt"
        self.assert_rejected_without_install(
            self.switching, self.parent, self.config, "candidate",
            self.deployment, wrong_path, self.deployment_path)
        wrong_reference = copy.deepcopy(self.deployment)
        wrong_reference["reference_endpoint"] = "/out/other-reference.pt"
        self.assert_rejected_without_install(
            self.switching, self.parent, self.config, "candidate",
            wrong_reference, self.summary, self.deployment_path)
        partial = copy.deepcopy(self.summary)
        partial["completed_updates"] = 79
        self.assert_rejected_without_install(
            self.switching, self.parent, self.config, "candidate",
            self.deployment, partial, self.deployment_path)

    def test_rejects_bad_tensor_shape_dtype_finiteness_and_range(self):
        cases = []
        malformed = copy.deepcopy(self.deployment)
        malformed["ratios"].pop("b")
        cases.append(malformed)
        malformed = copy.deepcopy(self.deployment)
        malformed["ratios"]["a"] = self.torch.tensor([1.0, 1.0])
        cases.append(malformed)
        malformed = copy.deepcopy(self.deployment)
        malformed["ratios"]["a"] = self.torch.tensor(1.0, dtype=self.torch.float64)
        cases.append(malformed)
        malformed = copy.deepcopy(self.deployment)
        malformed["log2_scales"]["a"] = self.torch.tensor(float("nan"))
        cases.append(malformed)
        malformed = copy.deepcopy(self.deployment)
        malformed["log2_scales"]["a"] = self.torch.tensor(2.0)
        cases.append(malformed)
        for deployment in cases:
            with self.subTest(deployment=deployment):
                self.assert_rejected_without_install(
                    self.switching, self.parent, self.config, "candidate",
                    deployment, self.summary, self.deployment_path)


if __name__ == "__main__":
    from control_common import guard
    guard()
    unittest.main()
