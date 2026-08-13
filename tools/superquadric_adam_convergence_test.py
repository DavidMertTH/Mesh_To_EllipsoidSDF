"""Deterministic end-to-end convergence tests for global SQ Adam fitting.

Known rotated superquadric and bent-superquadric targets are sampled at radial
offsets around their surfaces.  Each fit starts from deliberately perturbed
geometry/shape parameters and must lower the fixed validation loss materially.

Run: .venv/Scripts/python.exe tools/superquadric_adam_convergence_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import MethodType
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warp as wp  # noqa: E402

from fit_validation import (  # noqa: E402
    evaluate_validation_loss,
    stratified_validation_from_samples,
)
from optimization import (  # noqa: E402
    OptimizationWorker,
    _superquadric_sdf_kernel_points,
    device,
)
from sdf_samples import SdfSampleSet  # noqa: E402


DX = 0.03
SURFACE_BAND = 3.0 * DX


def _quat_axis_angle(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = np.deg2rad(float(degrees)) * 0.5
    return np.array([
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
        np.cos(half),
    ], dtype=np.float32)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hamilton product for project-order xyzw quaternions."""
    x1, y1, z1, w1 = np.asarray(lhs, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(rhs, dtype=np.float64)
    result = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])
    return (result / np.linalg.norm(result)).astype(np.float32)


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - w*z), 2.0 * (x*z + w*y)],
        [2.0 * (x*y + w*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - w*x)],
        [2.0 * (x*z - w*y), 2.0 * (y*z + w*x), 1.0 - 2.0 * (x*x + y*y)],
    ], dtype=np.float64)


def _hard_sq_values(
    points: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    eps: np.ndarray,
    bend: np.ndarray,
) -> np.ndarray:
    count = int(len(points))
    wp_center = wp.array(center.reshape(1, 3), dtype=wp.vec3, device=device)
    wp_radii = wp.array(radii.reshape(1, 3), dtype=wp.vec3, device=device)
    wp_rotation = wp.array(
        rotation.reshape(-1), dtype=wp.float32, device=device)
    wp_eps = wp.array(eps.reshape(-1), dtype=wp.float32, device=device)
    wp_bend = wp.array(bend.reshape(-1), dtype=wp.float32, device=device)
    wp_points = wp.array(points, dtype=wp.vec3, device=device)
    wp_indices = wp.array(
        np.arange(count, dtype=np.int32), dtype=wp.int32, device=device)
    scan = wp.zeros((count, 2), dtype=wp.float32, device=device)
    output = wp.empty(count, dtype=wp.float32, device=device)
    wp.launch(
        _superquadric_sdf_kernel_points,
        dim=count,
        inputs=[
            wp_center, wp_radii, wp_rotation, wp_eps, wp_bend,
            scan, 1, wp_points, wp_indices, output,
        ],
        device=device,
    )
    return output.numpy().copy()


def _target_samples(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    eps: np.ndarray,
    bend: np.ndarray,
    *,
    seed: int,
) -> SdfSampleSet:
    """Sample deterministic local rays before applying bend and rotation."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(48, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    e1, e2 = (float(eps[0]), float(eps[1]))
    scaled = np.abs(directions / radii[None, :])
    f = ((scaled[:, 0] ** (2.0 / e2)
          + scaled[:, 1] ** (2.0 / e2)) ** (e2 / e1)
         + scaled[:, 2] ** (2.0 / e1))
    beta = f ** (0.5 * e1)
    surface = directions / beta[:, None]

    offsets = np.array(
        [-0.20, -0.10, -0.035, 0.0, 0.035, 0.10, 0.20],
        dtype=np.float64,
    )
    local = (surface[:, None, :] * (1.0 + offsets[None, :, None])).reshape(-1, 3)
    z = local[:, 2].copy()
    local[:, 0] += 0.5 * float(bend[0]) * z * z
    local[:, 1] += 0.5 * float(bend[1]) * z * z
    world = (local @ _quat_matrix(rotation).T + center[None, :]).astype(np.float32)
    values = _hard_sq_values(world, center, radii, rotation, eps, bend)
    thickness = np.full(len(world), 0.12, dtype=np.float32)
    return SdfSampleSet(
        points=world,
        values=values,
        thickness=thickness,
        dx=DX,
        source="known-rotated-superquadric",
    )


class SuperquadricAdamConvergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def _run_case(self, *, bent: bool) -> None:
        true_center = np.array([0.035, -0.025, 0.020], dtype=np.float32)
        true_radii = np.array([0.31, 0.225, 0.175], dtype=np.float32)
        true_rotation = _quat_axis_angle([0.35, 0.80, 0.48], 34.0)
        true_eps = np.array(
            [0.42, 0.36] if bent else [0.38, 0.52], dtype=np.float32)
        true_bend = np.array(
            [1.15, -0.78] if bent else [0.0, 0.0], dtype=np.float32)
        samples = _target_samples(
            true_center, true_radii, true_rotation, true_eps, true_bend,
            seed=91 if bent else 37,
        )

        initial_center = true_center + np.array(
            [0.055, -0.040, 0.032], dtype=np.float32)
        initial_radii = true_radii * np.array(
            [0.84, 1.16, 0.90], dtype=np.float32)
        rotation_error = _quat_axis_angle([0.70, -0.25, 0.55], 13.0)
        initial_rotation = _quat_multiply(rotation_error, true_rotation)
        initial_eps = np.array([0.76, 0.72], dtype=np.float32)
        initial_bend = np.array(
            [-0.25, 0.20] if bent else [0.0, 0.0], dtype=np.float32)

        worker = OptimizationWorker(
            sdf_target_np=np.zeros((32, 32, 32), dtype=np.float32),
            sdf_samples=samples,
            origin=np.array([-0.48, -0.48, -0.48], dtype=np.float32),
            dx=DX,
            n=32,
            num_ellipsoids=1,
            initial_centers=initial_center.reshape(1, 3),
            initial_radii=initial_radii.reshape(1, 3),
            initial_rotations=initial_rotation.reshape(1, 4),
            primitive_shape="bent_superquadric" if bent else "superquadric",
            sq_eps1=float(initial_eps[0]),
            sq_eps2=float(initial_eps[1]),
            sq_eps_mode="per_primitive",
            sq_unlock_frac=0.0,
            sq_bend_unlock_frac=0.0,
            sq_eps_lr_mult=0.45,
            sq_bend_lr_mult=0.25,
            num_steps=220,
            report_every=55,
            validation_every=20,
            validation_sample_size=samples.size,
            validation_patience=None,
            batch_size=192,
            surface_fraction=0.80,
            maintenance_every=0,
            superfit=False,
            local_fit=False,
            symmetry_enabled=False,
            soft_union=False,
            containment_weight=0.0,
            flat_weight=0.0,
            thin_loss_weight=0.0,
            miss_penalty_weight=3.0,
            outside_penalty_weight=8.0,
            surface_weight=4.0,
            surface_sigma_vox=1.5,
            loss_huber_delta_vox=0.5,
            lr_init=0.006,
            lr_final=0.0004,
            lr_decay_k=3.0,
            lr_mult_radii=1.5,
            lr_mult_rot=0.7,
        )
        worker._rng = np.random.default_rng(12345 if bent else 54321)
        if bent:
            worker._init_bend = MethodType(
                lambda _self, count: np.repeat(
                    initial_bend.reshape(1, 2), int(count), axis=0),
                worker,
            )

        validation = stratified_validation_from_samples(
            samples,
            sample_count=samples.size,
            surface_band=SURFACE_BAND,
            surface_fraction=0.80,
            coarse_fraction=0.0,
            seed=0,
        )
        initial_prediction = worker._pred_points_from_params(
            validation.points,
            initial_center.reshape(1, 3),
            initial_radii.reshape(1, 3),
            initial_rotation.reshape(1, 4),
            initial_eps.reshape(1, 2),
            initial_bend.reshape(1, 2),
        )
        initial_loss = evaluate_validation_loss(
            initial_prediction,
            validation,
            huber_delta=0.5 * DX,
            miss_weight=3.0,
            surface_weight=4.0,
            surface_sigma=1.5 * DX,
            outside_weight=8.0,
            thin_weight=0.0,
            thin_max_factor=6.0,
            coarse_far_weight=0.0,
        ).total

        final_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        worker.step_visual.connect(
            lambda _step, _loss, centers, radii, rotations, extra:
            final_frames.append((
                np.asarray(centers, dtype=np.float32).copy(),
                np.asarray(radii, dtype=np.float32).copy(),
                np.asarray(rotations, dtype=np.float32).copy(),
                np.asarray(extra, dtype=np.float32).copy(),
            )))
        worker._reset_stale_tape()
        worker._run_adam()

        self.assertTrue(worker.validation_history)
        self.assertTrue(final_frames)
        self.assertTrue(np.isfinite(worker.best_validation_loss))
        self.assertLess(
            worker.best_validation_loss,
            0.65 * float(initial_loss),
            msg=(f"{'Bent-SQ' if bent else 'SQ'} did not converge clearly: "
                 f"initial={initial_loss:.6g}, best={worker.best_validation_loss:.6g}"),
        )
        # The deterministic metric must also improve over the first post-update
        # validation check, not only over the deliberately perturbed seed.
        self.assertLess(
            worker.best_validation_loss,
            0.75 * float(worker.validation_history[0][1]),
        )

        final_center, final_radii, final_rotation, final_extra = final_frames[-1]
        for name, values in (
            ("center", final_center), ("radii", final_radii),
            ("rotation", final_rotation), ("shape", final_extra),
        ):
            self.assertTrue(np.isfinite(values).all(), msg=f"non-finite {name}")
        self.assertGreater(float(np.min(final_radii)), 0.0)
        eps = final_extra[:, :2]
        self.assertTrue(np.all((eps >= 0.1) & (eps <= 2.0)))
        if bent:
            self.assertEqual(final_extra.shape, (1, 4))
            kappa = final_extra[:, 2:] * final_radii[:, 2:3]
            self.assertTrue(np.all(
                np.abs(kappa) <= worker._bend_kappa_max + 1.0e-5))
        else:
            self.assertEqual(final_extra.shape, (1, 2))

        final_prediction = worker._pred_points_from_params(
            validation.points,
            final_center, final_radii, final_rotation,
            eps,
            final_extra[:, 2:] if bent else np.zeros((1, 2), dtype=np.float32),
        )
        restored_loss = evaluate_validation_loss(
            final_prediction,
            validation,
            huber_delta=0.5 * DX,
            miss_weight=3.0,
            surface_weight=4.0,
            surface_sigma=1.5 * DX,
            outside_weight=8.0,
            thin_weight=0.0,
            thin_max_factor=6.0,
            coarse_far_weight=0.0,
        ).total
        np.testing.assert_allclose(
            restored_loss, worker.best_validation_loss,
            rtol=2.0e-5, atol=2.0e-7,
        )

    def test_rotated_superquadric_converges(self) -> None:
        self._run_case(bent=False)

    def test_rotated_bent_superquadric_converges(self) -> None:
        self._run_case(bent=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
