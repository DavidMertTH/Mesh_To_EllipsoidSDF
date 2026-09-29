"""Regression tests for transform channels locked during pose refitting."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization import OptimizationWorker  # noqa: E402


INITIAL_CENTER = np.array([[0.25, 0.27, 0.29]], dtype=np.float32)
INITIAL_RADII = np.array([[0.13, 0.11, 0.09]], dtype=np.float32)
INITIAL_ROTATION = np.array([[0.0, 0.0, 0.25881904, 0.9659258]],
                            dtype=np.float32)


def _run_fit(*, return_worker=False, **overrides):
    n = 7
    dx = 0.1
    axis = (np.arange(n, dtype=np.float32) + 0.5) * dx
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    target = (np.sqrt(
        (x - 0.43) ** 2 + (y - 0.34) ** 2 + (z - 0.33) ** 2
    ) - 0.16).astype(np.float32)
    kwargs = dict(
        sdf_target_np=target,
        origin=np.zeros(3, dtype=np.float32),
        dx=dx,
        n=n,
        num_ellipsoids=1,
        initial_centers=INITIAL_CENTER.copy(),
        initial_radii=INITIAL_RADII.copy(),
        initial_rotations=INITIAL_ROTATION.copy(),
        primitive_shape="ellipsoid",
        num_steps=4,
        report_every=1,
        validation_every=1,
        validation_patience=None,
        validation_sample_size=64,
        sample_budget=64,
        maintenance_every=0,
        superfit=False,
        local_fit=False,
        symmetry_enabled=False,
        containment_weight=0.0,
        flat_weight=0.0,
        normal_loss_weight=0.0,
        lr_init=0.01,
        lr_final=0.01,
    )
    kwargs.update(overrides)
    worker = OptimizationWorker(**kwargs)
    frames = []
    worker.step_visual.connect(
        lambda _step, _loss, centers, radii, rotations, _extra:
        frames.append((
            np.asarray(centers).copy(),
            np.asarray(radii).copy(),
            np.asarray(rotations).copy(),
        )))
    worker._reset_stale_tape()
    if worker._method == "adam":
        worker._run_adam()
    else:
        worker._run_naive()
    assert frames
    return (frames[-1], worker) if return_worker else frames[-1]


class PoseRefitTransformLockTest(unittest.TestCase):
    def test_default_still_optimizes_all_transform_channels(self) -> None:
        centers, radii, rotations = _run_fit()
        self.assertGreater(
            np.max(np.abs(centers - INITIAL_CENTER)), 1.0e-4)
        self.assertGreater(
            np.max(np.abs(radii - INITIAL_RADII)), 1.0e-4)
        self.assertGreater(
            np.max(np.abs(rotations - INITIAL_ROTATION)), 1.0e-4)

    def test_each_disabled_channel_preserves_its_starting_value(self) -> None:
        for channel, option, initial in (
            ("position", "optimize_centers", INITIAL_CENTER),
            ("rotation", "optimize_rotations", INITIAL_ROTATION),
            ("scale", "optimize_radii", INITIAL_RADII),
        ):
            with self.subTest(channel=channel):
                result = _run_fit(**{option: False})
                observed = {
                    "position": result[0],
                    "rotation": result[2],
                    "scale": result[1],
                }[channel]
                np.testing.assert_allclose(observed, initial,
                                           rtol=1.0e-6, atol=1.0e-6)

    def test_zero_center_lr_multiplier_freezes_only_position(self) -> None:
        centers, radii, rotations = _run_fit(lr_mult_centers=0.0)
        np.testing.assert_allclose(centers, INITIAL_CENTER,
                                   rtol=1.0e-6, atol=1.0e-6)
        self.assertGreater(
            np.max(np.abs(radii - INITIAL_RADII)), 1.0e-4)
        self.assertGreater(
            np.max(np.abs(rotations - INITIAL_ROTATION)), 1.0e-4)

    def test_center_lr_multiplier_scales_position_updates(self) -> None:
        unit_centers, _, _ = _run_fit(
            lr_mult_centers=1.0, center_step_radius_frac=0.0)
        fast_centers, _, _ = _run_fit(
            lr_mult_centers=2.0, center_step_radius_frac=0.0)
        unit_delta = float(np.linalg.norm(unit_centers - INITIAL_CENTER))
        fast_delta = float(np.linalg.norm(fast_centers - INITIAL_CENTER))
        self.assertGreater(fast_delta, 1.5 * unit_delta)

    def test_all_disabled_channels_preserve_full_transform(self) -> None:
        centers, radii, rotations = _run_fit(
            optimize_centers=False,
            optimize_rotations=False,
            optimize_radii=False,
        )
        np.testing.assert_allclose(centers, INITIAL_CENTER,
                                   rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(radii, INITIAL_RADII,
                                   rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(rotations, INITIAL_ROTATION,
                                   rtol=1.0e-6, atol=1.0e-6)

    def test_bone_local_fit_keeps_locked_offsets_and_follows_transform(self) -> None:
        linear = np.array([[[1.1, 0.0, 0.0],
                            [0.0, 0.9, 0.0],
                            [0.0, 0.0, 1.0]]], dtype=np.float32)
        offset = np.array([[0.06, 0.02, -0.01]], dtype=np.float32)
        prefix = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        (world_centers, _, _), worker = _run_fit(
            return_worker=True,
            optimize_centers=False,
            optimize_rotations=False,
            optimize_radii=False,
            parameter_linear_np=linear,
            parameter_offset_np=offset,
            parameter_rotation_prefix_np=prefix,
            parameter_anchor_centers=INITIAL_CENTER.copy(),
            parameter_anchor_radii=INITIAL_RADII.copy(),
            parameter_anchor_rotations=INITIAL_ROTATION.copy(),
            parameter_center_trust_radius_factor=1.75,
            parameter_radii_trust_factor=2.5,
        )
        self.assertIsNotNone(worker.optimized_parameter_result)
        local_centers, local_radii, local_rotations = (
            worker.optimized_parameter_result)
        np.testing.assert_allclose(local_centers, INITIAL_CENTER,
                                   rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(local_radii, INITIAL_RADII,
                                   rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(local_rotations, INITIAL_ROTATION,
                                   rtol=1.0e-6, atol=1.0e-6)
        expected_world = np.einsum(
            "nij,nj->ni", linear, INITIAL_CENTER) + offset
        np.testing.assert_allclose(world_centers, expected_world,
                                   rtol=1.0e-6, atol=1.0e-6)

    def test_sgd_fallback_honors_transform_locks(self) -> None:
        centers, radii, rotations = _run_fit(
            method="sgd",
            optimize_centers=False,
            optimize_rotations=False,
            optimize_radii=False,
        )
        np.testing.assert_allclose(centers, INITIAL_CENTER,
                                   rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(radii, INITIAL_RADII,
                                   rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(rotations, INITIAL_ROTATION,
                                   rtol=1.0e-6, atol=1.0e-6)

    def test_sgd_fallback_honors_zero_center_lr_multiplier(self) -> None:
        centers, radii, rotations = _run_fit(
            method="sgd", lr_mult_centers=0.0)
        np.testing.assert_allclose(centers, INITIAL_CENTER,
                                   rtol=1.0e-6, atol=1.0e-6)
        self.assertGreater(
            np.max(np.abs(radii - INITIAL_RADII)), 1.0e-4)
        self.assertGreater(
            np.max(np.abs(rotations - INITIAL_ROTATION)), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
