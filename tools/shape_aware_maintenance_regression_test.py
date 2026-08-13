"""Regression tests for shape-aware superquadric population maintenance."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization import (  # noqa: E402
    OptimizationWorker,
    _quat_to_rot_matrix,
    _rot_matrix_to_quat,
)


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _identity_quats(count: int) -> np.ndarray:
    return np.repeat(IDENTITY_QUAT[None, :], int(count), axis=0)


def _worker(
    sdf_target: np.ndarray | None = None,
    *,
    origin: np.ndarray | None = None,
    dx: float = 0.1,
    **overrides,
) -> OptimizationWorker:
    if sdf_target is None:
        sdf_target = np.zeros((11, 11, 11), dtype=np.float32)
    sdf_target = np.asarray(sdf_target, dtype=np.float32)
    if origin is None:
        origin = np.full(3, -0.5 * sdf_target.shape[0] * dx, dtype=np.float32)
    kwargs = dict(
        sdf_target_np=sdf_target,
        origin=np.asarray(origin, dtype=np.float32),
        dx=float(dx),
        n=max(sdf_target.shape),
        num_ellipsoids=2,
        max_ellipsoids=4,
        num_steps=2,
        report_every=1,
        sample_budget=64,
        maintenance_every=0,
        local_fit=False,
        primitive_shape="bent_superquadric",
        sq_eps1=0.5,
        sq_eps2=0.5,
        fuse_samples=96,
    )
    kwargs.update(overrides)
    return OptimizationWorker(**kwargs)


class ShapeAwareMaintenanceRegressionTest(unittest.TestCase):
    def test_boxy_sq_coverage_includes_corner_missed_by_ellipsoid_proxy(self) -> None:
        dx = 0.2
        origin = np.full(3, -1.1, dtype=np.float32)
        target = np.ones((11, 11, 11), dtype=np.float32)
        # Voxel centre (x, y, z) = (0.8, 0.8, 0.0): outside a unit sphere,
        # but comfortably inside a boxy unit superquadric.
        target[5, 9, 9] = -1.0
        worker = _worker(
            target,
            origin=origin,
            dx=dx,
            primitive_shape="superquadric",
            num_ellipsoids=1,
            coverage_sample_size=8,
        )
        centers = np.zeros((1, 3), dtype=np.float32)
        radii = np.ones((1, 3), dtype=np.float32)
        rotations = _identity_quats(1)
        eps = np.array([[0.2, 0.2]], dtype=np.float32)
        bend = np.zeros((1, 2), dtype=np.float32)

        coverage = worker._compute_coverage_info(
            centers, radii, rotations, eps, bend)

        self.assertTrue(coverage["valid"])
        self.assertEqual(int(coverage["total_coverage"][0]), 1)
        point = coverage["pts"]
        actual = worker._primitive_sdf_np(
            centers[0], radii[0], rotations[0], point, eps[0], bend[0])
        proxy = worker._ellipsoid_sdf_np(
            centers[0], radii[0], rotations[0], point)
        self.assertLess(float(actual[0]), 0.0)
        self.assertGreater(float(proxy[0]), 0.0)

    def test_bent_aabb_contains_true_volume_uniform_interior_probes(self) -> None:
        worker = _worker(num_ellipsoids=1)
        center = np.zeros(3, dtype=np.float32)
        radii = np.array([0.4, 0.3, 1.0], dtype=np.float32)
        eps = np.array([0.35, 0.55], dtype=np.float32)
        bend = np.array([1.2, -0.8], dtype=np.float32)

        low, high = worker._primitive_aabbs(
            center[None, :], radii[None, :], IDENTITY_QUAT[None, :],
            bend[None, :])
        probes = worker._primitive_interior_points(
            center, radii, IDENTITY_QUAT, 768, eps, bend,
            seed=20260721, beta_limit=0.98)

        # Quadratic bend shifts only the positive x side and negative y side
        # for this identity-frame primitive.
        np.testing.assert_allclose(low[0], [-0.4, -0.7, -1.0], atol=1e-6)
        np.testing.assert_allclose(high[0], [1.0, 0.3, 1.0], atol=1e-6)
        self.assertTrue(np.isfinite(probes).all())
        self.assertTrue(np.all(probes >= low[0] - 2e-6))
        self.assertTrue(np.all(probes <= high[0] + 2e-6))
        probe_sdf = worker._primitive_sdf_np(
            center, radii, IDENTITY_QUAT, probes, eps, bend)
        self.assertLessEqual(float(np.max(probe_sdf)), 2e-5)

    def test_real_protrusion_detector_uses_bent_surface(self) -> None:
        size = 31
        dx = 0.1
        origin = np.full(3, -1.55, dtype=np.float32)
        x_centers = origin[0] + (np.arange(size, dtype=np.float32) + 0.5) * dx
        target = np.broadcast_to(
            (x_centers - 0.45)[None, None, :], (size, size, size)).copy()
        worker = _worker(
            target,
            origin=origin,
            dx=dx,
            num_ellipsoids=1,
            min_split_radius_vox=1.0,
        )
        centers = np.zeros((1, 3), dtype=np.float32)
        radii = np.array([[0.35, 0.3, 0.8]], dtype=np.float32)
        rotations = _identity_quats(1)
        eps = np.array([[0.35, 0.4]], dtype=np.float32)
        straight = np.zeros((1, 2), dtype=np.float32)
        bent = np.array([[1.6, 0.0]], dtype=np.float32)

        straight_result = worker._detect_protruding_ellipsoids(
            centers, radii, rotations, eps, straight)
        bent_result = worker._detect_protruding_ellipsoids(
            centers, radii, rotations, eps, bent)

        self.assertEqual(straight_result.tolist(), [])
        self.assertEqual(bent_result.tolist(), [0])
        score = worker._primitive_protrusion_scores(
            centers, radii, rotations, eps, bent)
        self.assertGreater(float(score[0]), 0.0)

    def test_real_redundancy_detector_removes_only_one_duplicate_sq(self) -> None:
        worker = _worker(fuse_overlap_frac=0.9)
        centers = np.zeros((2, 3), dtype=np.float32)
        radii = np.repeat(
            np.array([[0.45, 0.32, 0.7]], dtype=np.float32), 2, axis=0)
        rotations = _identity_quats(2)
        eps = np.repeat(
            np.array([[0.3, 0.5]], dtype=np.float32), 2, axis=0)
        bend = np.repeat(
            np.array([[0.8, -0.35]], dtype=np.float32), 2, axis=0)

        removed = worker._detect_redundant_ellipsoids(
            centers, radii, rotations, 2, eps, bend)

        # The greedy real-volume path may remove either identical row, but must
        # retain one survivor so a mutually redundant pair cannot make a hole.
        self.assertEqual(len(removed), 1)
        self.assertIn(int(removed[0]), (0, 1))

    def test_bent_z_split_returns_complete_tangent_aligned_shape_state(self) -> None:
        worker = _worker(num_ellipsoids=1)
        angle = 0.41
        co, si = np.cos(angle), np.sin(angle)
        parent_matrix = np.array(
            [[co, 0.0, si], [0.0, 1.0, 0.0], [-si, 0.0, co]],
            dtype=np.float64,
        )
        center = np.array([0.15, -0.2, 0.35], dtype=np.float32)
        radii = np.array([0.45, 0.35, 1.2], dtype=np.float32)
        rotation = _rot_matrix_to_quat(parent_matrix)
        eps = np.array([0.34, 0.68], dtype=np.float32)
        bend = np.array([0.8, -0.45], dtype=np.float32)

        child_c, child_r, child_q, child_e, child_b = worker._split_primitive(
            center, radii, rotation, eps, bend)

        self.assertEqual(child_c.shape, (2, 3))
        self.assertEqual(child_r.shape, (2, 3))
        self.assertEqual(child_q.shape, (2, 4))
        self.assertEqual(child_e.shape, (2, 2))
        self.assertEqual(child_b.shape, (2, 2))
        self.assertTrue(np.isfinite(np.concatenate([
            child_c.ravel(), child_r.ravel(), child_q.ravel(),
            child_e.ravel(), child_b.ravel(),
        ])).all())
        np.testing.assert_allclose(child_r[:, 2], 0.6, atol=1e-7)
        np.testing.assert_allclose(child_e, np.repeat(eps[None, :], 2, axis=0))
        np.testing.assert_allclose(np.linalg.norm(child_q, axis=1), 1.0, atol=1e-6)

        for row, z0 in enumerate((0.6, -0.6)):
            centerline = np.array(
                [0.5 * bend[0] * z0**2, 0.5 * bend[1] * z0**2, z0])
            expected_center = center + parent_matrix @ centerline
            tangent_local = np.array([bend[0] * z0, bend[1] * z0, 1.0])
            tangent_local /= np.linalg.norm(tangent_local)
            expected_tangent = parent_matrix @ tangent_local
            child_matrix = _quat_to_rot_matrix(child_q[row])
            np.testing.assert_allclose(child_c[row], expected_center, atol=2e-6)
            np.testing.assert_allclose(
                child_matrix[:, 2], expected_tangent, atol=2e-6)
            self.assertGreater(float(np.linalg.det(child_matrix)), 0.99999)
        self.assertFalse(np.allclose(child_b, np.repeat(bend[None, :], 2, axis=0)))

    def test_merge_candidate_is_finite_and_shape_state_consistent(self) -> None:
        worker = _worker(merge_enabled=True)
        centers = np.array(
            [[-0.18, 0.0, 0.0], [0.22, 0.02, 0.0]], dtype=np.float32)
        radii = np.array(
            [[0.45, 0.32, 0.55], [0.4, 0.35, 0.5]], dtype=np.float32)
        rotations = _identity_quats(2)
        eps = np.array([[0.32, 0.52], [0.48, 0.7]], dtype=np.float32)
        bend = np.array([[0.3, -0.15], [0.22, -0.08]], dtype=np.float32)

        merged = worker._merge_two_primitives(
            0, 1, centers, radii, rotations, eps, bend)
        c_m, r_m, q_m, e_m, b_m = merged

        self.assertTrue(np.isfinite(np.concatenate(merged)).all())
        self.assertTrue(np.all(r_m > 0.0))
        self.assertAlmostEqual(float(np.linalg.norm(q_m)), 1.0, places=5)
        volumes = worker._primitive_volume_proxies(radii, eps)
        expected_eps = np.average(eps, axis=0, weights=volumes)
        np.testing.assert_allclose(e_m, expected_eps, rtol=1e-6, atol=1e-7)

        merged_matrix = _quat_to_rot_matrix(q_m)
        expected_world_curvature = np.average(bend, axis=0, weights=volumes)
        expected_world_curvature = np.append(expected_world_curvature, 0.0)
        expected_local_curvature = (
            merged_matrix[:, :2].T @ expected_world_curvature)
        np.testing.assert_allclose(
            b_m, expected_local_curvature, rtol=1e-5, atol=1e-6)
        actual_world_curvature = merged_matrix[:, :2] @ b_m
        expected_projection = (
            merged_matrix[:, :2] @ merged_matrix[:, :2].T
            @ expected_world_curvature)
        np.testing.assert_allclose(
            actual_world_curvature, expected_projection,
            rtol=1e-5, atol=1e-6)

        directions = np.vstack([
            np.eye(3, dtype=np.float32),
            -np.eye(3, dtype=np.float32),
            worker._unit_sphere_samples(),
        ])
        surface = worker._primitive_surface_points(
            c_m, r_m, q_m, directions, e_m, b_m)
        surface_sdf = worker._primitive_sdf_np(
            c_m, r_m, q_m, surface, e_m, b_m)
        self.assertLessEqual(float(np.max(np.abs(surface_sdf))), 5e-5)

        interior = worker._primitive_interior_points(
            c_m, r_m, q_m, 256, e_m, b_m,
            seed=913, beta_limit=0.95)
        interior_sdf = worker._primitive_sdf_np(
            c_m, r_m, q_m, interior, e_m, b_m)
        self.assertLessEqual(float(np.max(interior_sdf)), 2e-5)
        low, high = worker._primitive_aabbs(
            c_m[None, :], r_m[None, :], q_m[None, :], b_m[None, :])
        self.assertTrue(np.all(surface >= low[0] - 2e-5))
        self.assertTrue(np.all(surface <= high[0] + 2e-5))


if __name__ == "__main__":
    unittest.main()
