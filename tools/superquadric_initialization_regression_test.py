"""Formal regression tests for deterministic superquadric initialization.

The tests use analytic SDFs so that orientation, anisotropy and containment can
be checked independently from the sampled grid used by ``OptimizationWorker``.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization import OptimizationWorker  # noqa: E402
from superquadric_geometry import quaternion_matrix, surface_points  # noqa: E402


GRID_SIZE = 49
DX = 0.05


def _world_grid(
    n: int = GRID_SIZE, dx: float = DX,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.full(3, -0.5 * n * dx, dtype=np.float32)
    coordinates = origin[0] + (np.arange(n, dtype=np.float64) + 0.5) * dx
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return origin, np.stack([x, y, z], axis=-1)


def _capped_cylinder_sdf(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    half_length: float,
) -> np.ndarray:
    """Exact Euclidean SDF of a finite cylinder with flat end caps."""
    offset = np.asarray(points, dtype=np.float64) - np.asarray(center, np.float64)
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    axial = np.sum(offset * direction, axis=-1)
    radial = np.linalg.norm(offset - axial[..., None] * direction, axis=-1)
    q = np.stack([radial - radius, np.abs(axial) - half_length], axis=-1)
    return (
        np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
    )


def _surface_directions(count: int = 2048) -> np.ndarray:
    """Deterministic sphere coverage augmented by axes and cube corners."""
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    azimuth = index * np.pi * (3.0 - np.sqrt(5.0))
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fibonacci = np.column_stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), z])
    axes = np.vstack([np.eye(3), -np.eye(3)])
    corners = np.array(
        [[x, y, z_] for x in (-1.0, 1.0)
         for y in (-1.0, 1.0) for z_ in (-1.0, 1.0)],
        dtype=np.float64,
    )
    return np.vstack([fibonacci, axes, corners])


def _worker(
    sdf: np.ndarray,
    origin: np.ndarray,
    primitive_count: int,
    dx: float = DX,
) -> OptimizationWorker:
    return OptimizationWorker(
        sdf_target_np=np.asarray(sdf, dtype=np.float32),
        origin=np.asarray(origin, dtype=np.float32),
        dx=dx,
        n=int(sdf.shape[0]),
        num_ellipsoids=primitive_count,
        max_ellipsoids=max(4, primitive_count),
        num_steps=2,
        report_every=1,
        sample_budget=64,
        maintenance_every=0,
        local_fit=False,
        primitive_shape="superquadric",
        sq_eps1=1.0,
        sq_eps2=1.0,
    )


class SuperquadricInitializationRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.origin, cls.grid = _world_grid()
        cls.directions = _surface_directions()

    def _initialize(
        self, sdf: np.ndarray, count: int, *, dx: float = DX,
        origin: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if origin is None:
            origin = self.origin
        worker = _worker(sdf, origin, count, dx)
        worker.blockSignals(True)
        first = worker._init_inside_mesh(count)
        second = worker._init_inside_mesh(count)

        for actual, repeated in zip(first, second):
            np.testing.assert_array_equal(
                actual, repeated, err_msg="initialization must be deterministic")
            self.assertTrue(np.all(np.isfinite(actual)))

        centers, radii, rotations, eps = first
        self.assertEqual(centers.shape, (count, 3))
        self.assertEqual(radii.shape, (count, 3))
        self.assertEqual(rotations.shape, (count, 4))
        self.assertEqual(eps.shape, (count, 2))
        self.assertTrue(np.all(radii > 0.0))
        np.testing.assert_allclose(
            np.linalg.norm(rotations, axis=1), 1.0, atol=2.0e-6)
        self.assertTrue(np.all((eps >= 0.1) & (eps <= 2.0)))
        return first

    def _assert_no_gross_protrusion(
        self,
        initialized: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        analytic_sdf,
        tolerance: float = 1.5 * DX,
    ) -> None:
        centers, radii, rotations, eps = initialized
        for index in range(len(centers)):
            samples = surface_points(
                centers[index], radii[index], rotations[index], eps[index],
                self.directions,
            )
            maximum = float(np.max(analytic_sdf(samples)))
            self.assertLessEqual(
                maximum,
                tolerance,
                msg=f"primitive {index} protrudes by {maximum:.6g}",
            )

    def test_rotated_cylinder_recovers_axis_and_anisotropy(self) -> None:
        center = np.zeros(3, dtype=np.float64)
        axis = np.array([0.64, -0.31, 0.70], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        radius = 0.27
        half_length = 0.78

        def analytic(points):
            return _capped_cylinder_sdf(
                points, center, axis, radius, half_length)

        initialized = self._initialize(analytic(self.grid), 1)
        centers, radii, rotations, _eps = initialized
        local_long_axis = quaternion_matrix(rotations[0])[:, 2]

        self.assertLessEqual(float(analytic(centers)[0]), 0.0)
        self.assertGreater(abs(float(local_long_axis @ axis)), 0.95)
        self.assertGreater(float(radii[0, 2] / np.max(radii[0, :2])), 1.7)
        self.assertLess(float(np.max(radii[0, :2]) / np.min(radii[0, :2])), 1.15)
        self._assert_no_gross_protrusion(initialized, analytic)

    def test_isotropic_sphere_stays_isotropic_and_contained(self) -> None:
        center = np.zeros(3, dtype=np.float64)
        radius = 0.48

        def analytic(points):
            return np.linalg.norm(
                np.asarray(points, dtype=np.float64) - center, axis=-1) - radius

        initialized = self._initialize(analytic(self.grid), 1)
        centers, radii, _rotations, _eps = initialized

        self.assertLess(np.linalg.norm(centers[0] - center), DX)
        self.assertLess(float(np.max(radii[0]) / np.min(radii[0])), 1.10)
        self._assert_no_gross_protrusion(initialized, analytic)

    def test_disconnected_components_receive_independent_frames(self) -> None:
        component_centers = np.array(
            [[-0.65, 0.0, 0.0], [0.65, 0.0, 0.0]], dtype=np.float64)
        component_axes = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        radius = 0.22
        half_length = 0.52

        def component_sdfs(points):
            return np.stack([
                _capped_cylinder_sdf(
                    points, component_centers[index], component_axes[index],
                    radius, half_length,
                )
                for index in range(2)
            ], axis=-1)

        def analytic_union(points):
            return np.min(component_sdfs(points), axis=-1)

        initialized = self._initialize(analytic_union(self.grid), 2)
        centers, radii, rotations, _eps = initialized
        assignments = np.argmin(component_sdfs(centers), axis=1)

        np.testing.assert_array_equal(np.sort(assignments), [0, 1])
        for primitive, component in enumerate(assignments):
            self.assertLessEqual(
                float(component_sdfs(centers)[primitive, component]), 0.0)
            local_long_axis = quaternion_matrix(rotations[primitive])[:, 2]
            self.assertGreater(
                abs(float(local_long_axis @ component_axes[component])), 0.90)
            self.assertGreater(
                float(radii[primitive, 2] / np.max(radii[primitive, :2])), 1.5)
        self._assert_no_gross_protrusion(initialized, analytic_union)

    def test_empty_and_degenerate_sdfs_have_finite_deterministic_fallbacks(
        self,
    ) -> None:
        n = 17
        dx = 0.1
        origin = np.full(3, -0.5 * n * dx, dtype=np.float32)
        cases = {
            "empty": np.ones((n, n, n), dtype=np.float32),
            "non_finite": np.full((n, n, n), np.nan, dtype=np.float32),
        }

        for label, sdf in cases.items():
            with self.subTest(label=label):
                centers, radii, rotations, _eps = self._initialize(
                    sdf, 3, dx=dx, origin=origin)
                np.testing.assert_allclose(radii, 0.5 * dx, atol=1.0e-7)
                np.testing.assert_array_equal(
                    rotations,
                    np.tile(
                        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        (3, 1),
                    ),
                )
                lower = origin
                upper = origin + n * dx
                self.assertTrue(np.all(centers >= lower))
                self.assertTrue(np.all(centers <= upper))

        # One isolated negative voxel exercises the non-PCA degenerate path.
        isolated = np.ones((n, n, n), dtype=np.float32)
        isolated[n // 2, n // 2, n // 2] = -0.01
        self._initialize(isolated, 1, dx=dx, origin=origin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
