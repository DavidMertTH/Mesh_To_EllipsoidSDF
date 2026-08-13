"""Regression tests for the CPU-side superquadric geometry strategy."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from superquadric_geometry import (  # noqa: E402
    beta_and_gradient_local,
    interior_points,
    interior_points_local,
    quaternion_matrix,
    signed_distance,
    signed_distance_local,
    surface_points,
    volume,
)


class SuperquadricGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.center = np.array([0.2, -0.1, 0.4], dtype=np.float32)
        self.radii = np.array([1.7, 0.8, 0.45], dtype=np.float32)
        angle = np.deg2rad(31.0)
        self.rotation = np.array(
            [0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)],
            dtype=np.float32,
        )
        rng = np.random.default_rng(7)
        self.directions = rng.normal(size=(512, 3)).astype(np.float32)

    def test_parameterized_surface_is_zero_for_extreme_eps_and_bend(self) -> None:
        for eps in (
            np.array([0.1, 0.1], np.float32),
            np.array([0.35, 0.55], np.float32),
            np.array([1.0, 1.0], np.float32),
            np.array([2.0, 2.0], np.float32),
        ):
            bend = np.array([0.8, -0.45], dtype=np.float32)
            points = surface_points(
                self.center, self.radii, self.rotation, eps,
                self.directions, bend)
            distance = signed_distance(
                self.center, self.radii, self.rotation, eps, points, bend)
            self.assertTrue(np.all(np.isfinite(distance)))
            self.assertLess(float(np.max(np.abs(distance))), 2.0e-4)

    def test_epsilon_one_has_ellipsoid_first_order_distance(self) -> None:
        # On a principal ray, the ellipsoid distance is exact and gives a sharp
        # equality check for the gradient-normalised implicit formulation.
        points = np.array(
            [[self.radii[0] + offset, 0.0, 0.0]
             for offset in (-0.05, -0.01, 0.0, 0.01, 0.05)],
            dtype=np.float32,
        )
        distance = signed_distance_local(
            points, self.radii, np.ones(2, dtype=np.float32))
        expected = points[:, 0] - self.radii[0]
        np.testing.assert_allclose(distance, expected, atol=2.0e-6, rtol=2.0e-6)

    def test_axes_center_and_far_points_remain_finite(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [self.radii[0], 0.0, 0.0],
                [0.0, self.radii[1], 0.0],
                [0.0, 0.0, self.radii[2]],
                [100.0, -50.0, 25.0],
                [1.0e-10, -1.0e-12, 0.0],
            ],
            dtype=np.float32,
        )
        for eps in (np.array([0.1, 2.0]), np.array([2.0, 0.1])):
            beta, gradient = beta_and_gradient_local(
                points, self.radii, eps, np.array([1.2, -0.7]))
            distance = signed_distance_local(
                points, self.radii, eps, np.array([1.2, -0.7]))
            self.assertTrue(np.all(np.isfinite(beta)))
            self.assertTrue(np.all(np.isfinite(gradient)))
            self.assertTrue(np.all(np.isfinite(distance)))

    def test_volume_matches_ellipsoid_and_scale_cubed(self) -> None:
        ellipsoid = volume(self.radii, np.ones(2, dtype=np.float32))
        expected = 4.0 * np.pi * float(np.prod(self.radii)) / 3.0
        self.assertAlmostEqual(ellipsoid, expected, places=6)

        eps = np.array([0.37, 0.62], dtype=np.float32)
        base = volume(self.radii, eps)
        scaled = volume(3.25 * self.radii, eps)
        self.assertAlmostEqual(scaled / base, 3.25 ** 3, places=5)

    def test_interior_sampling_is_deterministic_uniform_and_shape_aware(self) -> None:
        eps = np.array([0.2, 0.35], dtype=np.float32)
        count = 4096
        beta_limit = 0.97
        first = interior_points_local(
            self.radii, eps, count, seed=23, beta_limit=beta_limit)
        repeat = interior_points_local(
            self.radii, eps, count, seed=23, beta_limit=beta_limit)
        different = interior_points_local(
            self.radii, eps, count, seed=24, beta_limit=beta_limit)

        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(first.shape, (count, 3))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(first.flags.c_contiguous)

        beta, _gradient = beta_and_gradient_local(first, self.radii, eps)
        self.assertTrue(np.all(np.isfinite(beta)))
        self.assertLessEqual(float(np.max(beta)), beta_limit + 2.0e-5)
        self.assertTrue(np.all(signed_distance_local(first, self.radii, eps) < 0.0))

        # For any homogeneous 3-D body, beta^3 is uniform on [0, 1] when points
        # are uniform in volume.  This catches angular/radial sphere warps whose
        # samples are inside but strongly biased for box-like superquadrics.
        radial_volume_coordinate = (beta / beta_limit) ** 3
        self.assertAlmostEqual(
            float(np.mean(radial_volume_coordinate)), 0.5, delta=0.03)

        # A boxy SQ owns substantial corner volume outside its same-radii
        # ellipsoid.  Seeing those probes verifies that this is shape-aware SQ
        # sampling rather than the former scaled unit-ball cloud.
        ellipsoid_level = np.sum((first / self.radii[None, :]) ** 2, axis=1)
        self.assertGreater(float(np.mean(ellipsoid_level > 1.0)), 0.10)

    def test_bent_interior_sampling_uses_exact_forward_warp(self) -> None:
        eps = np.array([0.32, 1.65], dtype=np.float32)
        bend = np.array([1.25, -0.7], dtype=np.float32)
        count = 768
        plain = interior_points_local(
            self.radii, eps, count, seed=11, beta_limit=0.9)
        bent = interior_points_local(
            self.radii, eps, count, bend, seed=11, beta_limit=0.9)

        expected = plain.astype(np.float64)
        z2 = expected[:, 2] ** 2
        expected[:, 0] += 0.5 * float(bend[0]) * z2
        expected[:, 1] += 0.5 * float(bend[1]) * z2
        np.testing.assert_allclose(bent, expected, rtol=0.0, atol=2.0e-7)

        beta, _gradient = beta_and_gradient_local(
            bent, self.radii, eps, bend)
        self.assertLessEqual(float(np.max(beta)), 0.9 + 2.0e-5)
        self.assertTrue(np.all(
            signed_distance_local(bent, self.radii, eps, bend) < 0.0))

        world = interior_points(
            self.center, self.radii, self.rotation, eps, count, bend,
            seed=11, beta_limit=0.9)
        expected_world = (
            self.center.astype(np.float64)[None, :]
            + bent.astype(np.float64) @ quaternion_matrix(self.rotation).T)
        np.testing.assert_allclose(world, expected_world, rtol=0.0, atol=2.0e-7)
        self.assertTrue(np.all(signed_distance(
            self.center, self.radii, self.rotation, eps, world, bend) < 0.0))

    def test_interior_sampling_extremes_and_validation(self) -> None:
        bend = np.array([6.0, -6.0], dtype=np.float32)
        for eps in (
            np.array([0.1, 0.1], dtype=np.float32),
            np.array([0.1, 2.0], dtype=np.float32),
            np.array([2.0, 0.1], dtype=np.float32),
            np.array([2.0, 2.0], dtype=np.float32),
        ):
            points = interior_points_local(
                self.radii, eps, 512, bend, seed=5)
            beta, gradient = beta_and_gradient_local(
                points, self.radii, eps, bend)
            self.assertTrue(np.all(np.isfinite(points)))
            self.assertTrue(np.all(np.isfinite(beta)))
            self.assertTrue(np.all(np.isfinite(gradient)))
            self.assertLess(float(np.max(beta)), 1.0 + 2.0e-5)

        empty = interior_points_local(self.radii, np.ones(2), 0)
        self.assertEqual(empty.shape, (0, 3))
        with self.assertRaises(ValueError):
            interior_points_local(self.radii, np.ones(2), -1)
        with self.assertRaises(ValueError):
            interior_points_local(self.radii, np.ones(2), 8, beta_limit=0.0)
        with self.assertRaises(ValueError):
            interior_points_local(
                np.array([1.0, np.nan, 1.0]), np.ones(2), 8)


if __name__ == "__main__":
    unittest.main()
