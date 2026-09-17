"""Regression checks for the direct GPU superquadric grid prediction."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization import (  # noqa: E402
    OptimizationWorker,
    _plain_superquadric_sdf_grid_chunk,
    _superquadric_sdf_grid_chunk,
    device,
)


class SuperquadricGridPredictionTest(unittest.TestCase):
    def _worker(self, primitive_shape: str) -> OptimizationWorker:
        return OptimizationWorker(
            sdf_target_np=np.zeros((3, 4, 5), dtype=np.float32),
            origin=np.array([-0.7, -0.8, -0.9], dtype=np.float32),
            dx=0.27,
            n=5,
            num_ellipsoids=2,
            max_ellipsoids=2,
            num_steps=1,
            report_every=1,
            sample_budget=32,
            maintenance_every=0,
            local_fit=False,
            primitive_shape=primitive_shape,
        )

    def test_grid_matches_exact_point_path_for_plain_and_bent(self) -> None:
        centers = np.array([[-0.2, 0.1, 0.2], [0.4, -0.1, -0.25]], np.float32)
        radii = np.array([[0.55, 0.35, 0.7], [0.3, 0.5, 0.6]], np.float32)
        rotations = np.array([[0, 0, 0, 1], [0.1, 0.2, 0.3, 0.927]], np.float32)
        eps = np.array([[0.6, 1.4], [1.5, 0.7]], np.float32)
        bend = np.array([[0.25, -0.15], [-0.2, 0.1]], np.float32)

        for kind in ("superquadric", "bent_superquadric"):
            with self.subTest(kind=kind):
                worker = self._worker(kind)
                flat = np.arange(np.prod(worker._shape), dtype=np.int64)
                points = worker._grid_points_from_flat(flat)
                expected = worker._pred_points_from_params(
                    points, centers, radii, rotations, eps, bend)
                actual = worker._pred_grid_from_params(
                    centers, radii, rotations, eps, bend)
                self.assertEqual(actual.shape, (3, 4, 5))
                np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-5, atol=1e-6)
                with mock.patch("optimization._SQ_GRID_PRED_CHUNK_SIZE", 7):
                    chunked = worker._pred_grid_from_params(
                        centers, radii, rotations, eps, bend)
                np.testing.assert_allclose(
                    chunked.ravel(), expected, rtol=1e-5, atol=1e-6)

                # Force a non-zero flat offset across row and slice boundaries.
                offset, count = 17, 23
                out = wp.empty(count, dtype=wp.float32, device=device)
                args = [
                    wp.array(centers, dtype=wp.vec3, device=device),
                    wp.array(radii, dtype=wp.vec3, device=device),
                    wp.array(rotations.reshape(-1), dtype=wp.float32, device=device),
                    wp.array(eps.reshape(-1), dtype=wp.float32, device=device),
                ]
                if kind == "bent_superquadric":
                    args.append(wp.array(bend.reshape(-1), dtype=wp.float32, device=device))
                args.extend([
                    len(centers), wp.vec3(*worker._origin), worker._dx,
                    worker._nx, worker._ny, offset, out,
                ])
                wp.launch(
                    _superquadric_sdf_grid_chunk if kind == "bent_superquadric"
                    else _plain_superquadric_sdf_grid_chunk,
                    dim=count, inputs=args, device=device)
                np.testing.assert_allclose(
                    out.numpy(), expected[offset:offset + count],
                    rtol=1e-5, atol=1e-6)

    def test_empty_population_preserves_empty_union(self) -> None:
        worker = self._worker("superquadric")
        actual = worker._pred_grid_from_params(
            np.empty((0, 3), np.float32), np.empty((0, 3), np.float32),
            np.empty((0, 4), np.float32))
        self.assertEqual(actual.shape, (3, 4, 5))
        np.testing.assert_array_equal(actual, np.full((3, 4, 5), 1.0e6, np.float32))

    def test_empty_grid_does_not_allocate_zero_length_gpu_buffer(self) -> None:
        worker = self._worker("superquadric")
        worker._nz = 0
        worker._shape = (0, 4, 5)
        actual = worker._pred_grid_from_params(
            np.zeros((1, 3), np.float32), np.ones((1, 3), np.float32),
            np.array([[0, 0, 0, 1]], np.float32))
        self.assertEqual(actual.shape, (0, 4, 5))


if __name__ == "__main__":
    unittest.main()
