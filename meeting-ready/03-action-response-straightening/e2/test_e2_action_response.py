"""Small CPU-only contract tests for E2's finite-difference math and split."""

import unittest

import torch

from run_e2_action_response import (
    curvature_penalty,
    split_episode_ids,
    symmetric_triplet,
    taylor_residuals,
)


class E2MathTests(unittest.TestCase):
    def test_triplet_is_symmetric_and_remains_inside_the_action_box(self) -> None:
        center = torch.zeros(5, 10)
        center.reshape(-1)[7] = 0.4
        base, plus, minus, direction = symmetric_triplet(center, -1.0, 1.0, 0.02, 7)
        self.assertTrue(torch.equal(direction.nonzero().flatten(), torch.tensor([7])))
        self.assertTrue(torch.allclose((plus + minus) / 2, base, atol=1e-7, rtol=0.0))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(plus - base)), 0.02, places=6)
        self.assertGreaterEqual(float(minus.min()), -1.0)
        self.assertLessEqual(float(plus.max()), 1.0)

    def test_triplet_rejects_boundary_clipping(self) -> None:
        center = torch.zeros(5, 10)
        center.reshape(-1)[3] = 1.0
        with self.assertRaisesRegex(ValueError, "does not fit"):
            symmetric_triplet(center, -1.0, 1.0, 0.01, 3)

    def test_normalized_three_point_curvature_matches_quadratic(self) -> None:
        delta = torch.tensor([0.1, 0.2])
        center = torch.zeros(2, 2)
        # f(x)=[x^2, 2x^2] gives dimension-normalized squared curvature 10.
        plus = torch.stack((delta.square(), 2 * delta.square()), dim=1)
        minus = plus.clone()
        value = curvature_penalty(plus, center, minus, delta)
        self.assertAlmostEqual(float(value), 10.0, places=5)

    def test_linear_map_has_zero_heldout_taylor_residual(self) -> None:
        delta, radius = 0.1, 0.2
        slope = torch.tensor([[2.0, -3.0]])
        center = torch.tensor([[0.5, 0.25]])
        plus_d = center + delta * slope
        minus_d = center - delta * slope
        plus_r = center + radius * slope
        minus_r = center - radius * slope
        rmse, relative = taylor_residuals(center, plus_d, minus_d, plus_r, minus_r, delta, radius)
        self.assertLess(float(rmse[0]), 1e-6)
        self.assertLess(float(relative[0]), 1e-6)

    def test_parent_episode_split_is_deterministic_and_disjoint(self) -> None:
        train_a, heldout_a = split_episode_ids(range(20), 5, 3, 23)
        train_b, heldout_b = split_episode_ids(range(20), 5, 3, 23)
        self.assertEqual((train_a, heldout_a), (train_b, heldout_b))
        self.assertFalse(set(train_a) & set(heldout_a))
        self.assertEqual(len(train_a), 5)
        self.assertEqual(len(heldout_a), 3)


if __name__ == "__main__":
    unittest.main()
