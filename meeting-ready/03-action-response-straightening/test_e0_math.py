"""Small, dependency-free checks for action-direction finite differences."""

from __future__ import annotations

import math
import unittest


Vector = tuple[float, ...]


def add(a: Vector, b: Vector) -> Vector:
    return tuple(x + y for x, y in zip(a, b))


def subtract(a: Vector, b: Vector) -> Vector:
    return tuple(x - y for x, y in zip(a, b))


def scale(value: float, vector: Vector) -> Vector:
    return tuple(value * x for x in vector)


def squared_norm(vector: Vector) -> float:
    return sum(x * x for x in vector)


def central_second_difference(function, center: Vector, delta: Vector) -> Vector:
    plus = function(add(center, delta))
    middle = function(center)
    minus = function(subtract(center, delta))
    return tuple(p - 2.0 * m + n for p, m, n in zip(plus, middle, minus))


def taylor_residual(function, center: Vector, delta: Vector, jvp: Vector) -> Vector:
    return subtract(subtract(function(add(center, delta)), function(center)), jvp)


def affine(value: Vector) -> Vector:
    x, y = value
    return (2.0 * x - 3.0 * y + 1.0, -x + 4.0 * y - 2.0)


def affine_jvp(direction: Vector) -> Vector:
    dx, dy = direction
    return (2.0 * dx - 3.0 * dy, -dx + 4.0 * dy)


def quadratic(value: Vector) -> Vector:
    (x,) = value
    return (x * x,)


def quadratic_jvp(center: Vector, direction: Vector) -> Vector:
    (x,) = center
    (dx,) = direction
    return (2.0 * x * dx,)


def cubic(value: Vector) -> Vector:
    (x,) = value
    return (x * x * x,)


def cross_term(value: Vector) -> Vector:
    x, y = value
    return (x * y,)


class ActionResponseMathTests(unittest.TestCase):
    def test_affine_map_has_zero_curvature_and_exact_tangent_scores(self) -> None:
        center = (0.25, -0.5)
        delta = (0.08, 0.12)
        second_difference = central_second_difference(affine, center, delta)
        residual = taylor_residual(affine, center, delta, affine_jvp(delta))

        self.assertLess(squared_norm(second_difference), 1e-28)
        self.assertLess(squared_norm(residual), 1e-28)

        goal = (0.4, -1.2)
        exact = affine(add(center, delta))
        tangent = add(affine(center), affine_jvp(delta))
        exact_cost = squared_norm(subtract(exact, goal))
        tangent_cost = squared_norm(subtract(tangent, goal))
        self.assertAlmostEqual(exact_cost, tangent_cost, places=14)

    def test_quadratic_curvature_and_taylor_residual_scale_as_expected(self) -> None:
        center = (0.3,)
        small_delta = (0.1,)
        large_delta = (0.2,)

        small_second = central_second_difference(quadratic, center, small_delta)
        large_second = central_second_difference(quadratic, center, large_delta)
        small_radius = math.sqrt(squared_norm(small_delta))
        large_radius = math.sqrt(squared_norm(large_delta))

        # For f(x)=x^2, the normalized second difference is 2 and its
        # squared, radius-to-the-fourth normalized loss is 4.
        small_curvature = small_second[0] / (small_radius**2)
        large_curvature = large_second[0] / (large_radius**2)
        small_loss = squared_norm(small_second) / (small_radius**4)
        large_loss = squared_norm(large_second) / (large_radius**4)
        self.assertAlmostEqual(small_curvature, 2.0, places=12)
        self.assertAlmostEqual(large_curvature, 2.0, places=12)
        self.assertAlmostEqual(small_loss, 4.0, places=12)
        self.assertAlmostEqual(large_loss, 4.0, places=12)

        small_residual = taylor_residual(
            quadratic, center, small_delta, quadratic_jvp(center, small_delta)
        )[0]
        large_residual = taylor_residual(
            quadratic, center, large_delta, quadratic_jvp(center, large_delta)
        )[0]
        self.assertAlmostEqual(large_residual, 4.0 * small_residual, places=14)

    def test_cubic_at_zero_has_zero_central_second_difference_but_nonzero_residual(self) -> None:
        center = (0.0,)
        delta = (0.2,)

        second_difference = central_second_difference(cubic, center, delta)
        # The derivative of x^3 at zero is zero, so the tangent predicts zero.
        residual = taylor_residual(cubic, center, delta, (0.0,))

        self.assertEqual(second_difference, (0.0,))
        self.assertEqual(squared_norm(second_difference) / (delta[0] ** 4), 0.0)
        self.assertAlmostEqual(residual[0], delta[0] ** 3, places=15)
        self.assertGreater(squared_norm(residual), 0.0)

    def test_cross_term_is_missed_on_axes_and_seen_on_mixed_direction(self) -> None:
        center = (0.0, 0.0)
        radius = 0.3
        axis_x = (radius, 0.0)
        axis_y = (0.0, radius)
        mixed = (radius / math.sqrt(2.0), radius / math.sqrt(2.0))

        second_x = central_second_difference(cross_term, center, axis_x)
        second_y = central_second_difference(cross_term, center, axis_y)
        second_mixed = central_second_difference(cross_term, center, mixed)

        self.assertAlmostEqual(second_x[0], 0.0, places=15)
        self.assertAlmostEqual(second_y[0], 0.0, places=15)
        self.assertAlmostEqual(second_mixed[0], radius**2, places=15)


if __name__ == "__main__":
    unittest.main()
