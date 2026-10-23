"""Allocation-guarded CPU tensor checks for the explicit training objective."""
import sys
import unittest
from pathlib import Path

from control_common import guard


ROOT = Path(__file__).resolve().parent
DUAL_KL = ROOT.parent / "fastwam-vq-dual-kl"
if (DUAL_KL / "softbin_loss.py").is_file() and str(DUAL_KL) not in sys.path:
    sys.path.insert(0, str(DUAL_KL))

_GUARDED = False


def _ensure_guarded():
    global _GUARDED
    if not _GUARDED:
        guard()
        _GUARDED = True


def _toy_config():
    # Toy-only coefficients for unit tests; these are not experiment settings.
    return {
        "action_absolute_weight": 1.2,
        "response_weight": 0.35,
        "latent_absolute_weight": 0.8,
        "control_weight": 0.6,
        "lambda_latent": 0.7,
    }


def _toy_stats():
    return {
        "action": {"mean": [0.0] * 6, "scale": [1.0] * 6},
        "latent": {"mean": [0.0] * 48, "scale": [1.0] * 48},
        "config": {
            "bins": 16, "interior_min": -3.5, "interior_max": 3.5,
            "sigma": 0.375, "std_floor": 0.02,
        },
    }


class TrainingObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_guarded()
        import torch
        import training_objective
        from parameter_gate import paired_vjp

        cls.torch = torch
        cls.objective = training_objective
        cls.paired_vjp = staticmethod(paired_vjp)

    def setUp(self):
        torch = self.torch
        self.stats = _toy_stats()
        self.config = _toy_config()
        action_base = torch.linspace(-0.35, 0.35, 32 * 7, dtype=torch.float32).reshape(32, 7)
        future_base = torch.linspace(-0.2, 0.2, 1 * 48 * 3 * 14 * 28,
                                     dtype=torch.float32).reshape(self.objective.FUTURE_SHAPE)
        self.targets = {
            "clean": {
                "action": action_base.clone().requires_grad_(True),
                "future": future_base.clone().requires_grad_(True),
            },
            "noisy": {
                "action": (action_base * 0.9 + 0.025).clone().requires_grad_(True),
                "future": (future_base * 0.9 + 0.015).clone().requires_grad_(True),
            },
        }
        for variant in ("clean", "noisy"):
            self.targets[variant].update({
                "counts": {"video_steps": 10, "action_steps": 10},
                "action_initial_noise": torch.zeros((1, 32, 7)),
                "producer_job": "toy-teacher-job",
            })
        self.action_shift = torch.nn.Parameter(torch.tensor(0.17))
        self.future_scale = torch.nn.Parameter(torch.tensor(0.08))
        self.reader_parameter = torch.nn.Parameter(torch.tensor(0.7))
        self.reader_template = torch.linspace(-0.2, 0.2, 32 * 7).reshape(32, 7)

    def _student(self, variant):
        offsets = {"clean": 0.02, "noisy": 0.11}
        action = self.targets[variant]["action"].detach() + self.action_shift + offsets[variant]
        future = (self.targets[variant]["future"].detach() * (1.0 + self.future_scale)
                  + offsets[variant] * 0.1)
        return action, future

    def _toy_reader(self, future):
        tail_mean = future[:, :, 1:].float().mean()
        return self.reader_template + tail_mean * self.reader_parameter.detach()

    def _endpoint_and_readers(self, actions, futures):
        endpoint, raw = self.objective.paired_endpoint_objective(
            self.torch, actions, futures, self.targets, self.stats, self.config)
        reader_terms, raw_control = {}, {}
        for variant in ("clean", "noisy"):
            reader_terms[variant], raw_control[variant] = self.objective.weighted_reader_objective(
                self.torch, self._toy_reader(futures[variant]),
                self.targets[variant]["action"], self.stats, self.config)
        return endpoint + sum(reader_terms.values()), raw, raw_control

    def test_joint_autograd_matches_endpoint_cotangents_and_paired_vjp(self):
        torch = self.torch
        # Existing teacher payloads may carry counts, noise, and provenance too.
        self.assertIn("producer_job", self.targets["clean"])
        direct_outputs = {variant: self._student(variant) for variant in ("clean", "noisy")}
        direct_actions = {variant: direct_outputs[variant][0] for variant in direct_outputs}
        direct_futures = {variant: direct_outputs[variant][1] for variant in direct_outputs}
        direct_loss, _, raw_control = self._endpoint_and_readers(direct_actions, direct_futures)
        self.assertTrue(all(torch.isfinite(value) for value in raw_control.values()))
        direct_inputs = [self.action_shift, self.future_scale]
        direct_inputs.extend(self.targets[v][key] for v in ("clean", "noisy")
                             for key in ("action", "future"))
        direct_inputs.append(self.reader_parameter)
        direct_grads = torch.autograd.grad(direct_loss, direct_inputs, allow_unused=True)
        expected_parameter_grads = direct_grads[:2]
        self.assertTrue(all(value is None for value in direct_grads[2:]))

        leaves_actions = {
            v: direct_actions[v].detach().clone().requires_grad_(True)
            for v in ("clean", "noisy")
        }
        leaves_futures = {
            v: direct_futures[v].detach().clone().requires_grad_(True)
            for v in ("clean", "noisy")
        }
        endpoint_loss, _ = self.objective.paired_endpoint_objective(
            torch, leaves_actions, leaves_futures, self.targets, self.stats, self.config)
        cotangents = list(torch.autograd.grad(
            endpoint_loss,
            [leaves_actions["clean"], leaves_actions["noisy"],
             leaves_futures["clean"], leaves_futures["noisy"]]))
        # Match the real memory schedule: release each reader graph separately.
        for index, variant in enumerate(("clean", "noisy")):
            reader_leaf = direct_futures[variant].detach().clone().requires_grad_(True)
            reader_loss, _ = self.objective.weighted_reader_objective(
                torch, self._toy_reader(reader_leaf), self.targets[variant]["action"],
                self.stats, self.config)
            reader_vector, = torch.autograd.grad(reader_loss, reader_leaf)
            cotangents[2 + index] = cotangents[2 + index] + reader_vector
            del reader_leaf, reader_loss, reader_vector
        for action_grad in cotangents[:2]:
            self.assertEqual(float(action_grad[10:, :].abs().max()), 0.0)
            self.assertEqual(float(action_grad[:10, 6].abs().max()), 0.0)
        for future_grad in cotangents[2:]:
            self.assertEqual(float(future_grad[:, :, 0].abs().max()), 0.0)
        per_variant_cotangents = {
            "clean": (cotangents[0], cotangents[2]),
            "noisy": (cotangents[1], cotangents[3]),
        }
        staged_grads = self.paired_vjp(
            torch, [self.action_shift, self.future_scale],
            lambda variant: self._student(variant), per_variant_cotangents)
        for direct, staged in zip(expected_parameter_grads, staged_grads):
            self.assertIsNotNone(staged)
            self.assertTrue(torch.allclose(direct, staged, rtol=1e-5, atol=1e-6))

        self.assertTrue(all(parameter.grad is None for parameter in (
            self.reader_parameter, *[self.targets[v][k] for v in ("clean", "noisy")
                                     for k in ("action", "future")])))

    def test_shared_action_bias_requires_absolute_anchor(self):
        torch = self.torch
        zeros_action = torch.zeros(self.objective.ACTION_SHAPE)
        zeros_future = torch.zeros(self.objective.FUTURE_SHAPE)
        targets = {variant: {"action": zeros_action, "future": zeros_future}
                   for variant in ("clean", "noisy")}
        shared_bias = torch.nn.Parameter(torch.tensor(0.3))
        actions = {variant: zeros_action + shared_bias
                   for variant in ("clean", "noisy")}
        futures = {variant: zeros_future for variant in ("clean", "noisy")}
        _, terms = self.objective.paired_endpoint_objective(
            torch, actions, futures, targets, self.stats, self.config)
        response_grad, = torch.autograd.grad(
            terms["action_response_kl"], shared_bias, retain_graph=True)
        absolute_grad, = torch.autograd.grad(terms["action_absolute_kl"], shared_bias)
        self.assertEqual(float(response_grad), 0.0)
        self.assertGreater(float(absolute_grad.abs()), 0.0)

    def test_reader_term_is_half_weighted_and_teacher_detached(self):
        torch = self.torch
        reader_action = torch.zeros(self.objective.ACTION_SHAPE, requires_grad=True)
        teacher = torch.ones(self.objective.ACTION_SHAPE, requires_grad=True)
        weighted, raw = self.objective.weighted_reader_objective(
            torch, reader_action, teacher, self.stats, self.config)
        expected = (0.5 * self.config["lambda_latent"]
                    * self.config["control_weight"] * raw)
        self.assertTrue(torch.equal(weighted, expected))
        target_grad, = torch.autograd.grad(weighted, teacher, allow_unused=True)
        self.assertIsNone(target_grad)

    def test_config_requires_exact_finite_nonnegative_coefficients(self):
        validate = self.objective.validate_config
        self.assertEqual(validate(self.config), self.config)
        for candidate in (
            {key: value for key, value in self.config.items() if key != "response_weight"},
            {**self.config, "extra": 1.0},
            {**self.config, "response_weight": -0.1},
            {**self.config, "latent_absolute_weight": float("nan")},
            {**self.config, "control_weight": float("inf")},
            {**self.config, "control_weight": "0.5"},
            {**self.config, "action_absolute_weight": 0.0},
            {**self.config, "lambda_latent": 0.0},
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate(candidate)


if __name__ == "__main__":
    _ensure_guarded()
    unittest.main()
