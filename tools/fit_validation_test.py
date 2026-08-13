"""Focused regression tests for deterministic fit validation.

Run: .venv/Scripts/python.exe tools/fit_validation_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit_validation import (  # noqa: E402
    BestCheckpoint,
    STRATUM_COARSE_FAR,
    STRATUM_INSIDE,
    STRATUM_OUTSIDE,
    STRATUM_SURFACE,
    ValidationSample,
    evaluate_validation_loss,
    stratified_validation_from_grid,
    stratified_validation_from_samples,
)
from sdf_samples import SdfSampleSet  # noqa: E402


class StratifiedValidationTests(unittest.TestCase):
    def test_dense_sample_is_reproducible_and_uses_voxel_centres(self) -> None:
        profile = np.array(
            [-0.5, -0.3, -0.02, 0.0, 0.02, 0.2, 0.4, 0.6],
            dtype=np.float32,
        )
        grid = np.broadcast_to(profile, (4, 5, profile.size)).copy()
        thickness = np.linspace(0.01, 0.8, grid.size, dtype=np.float32).reshape(grid.shape)
        origin = np.array([-1.0, 2.0, 3.0], dtype=np.float32)

        first = stratified_validation_from_grid(
            grid, origin, 0.1, thickness=thickness,
            sample_count=40, surface_band=0.05, seed=1701)
        second = stratified_validation_from_grid(
            grid, origin, 0.1, thickness=thickness,
            sample_count=40, surface_band=0.05, seed=1701)
        other_seed = stratified_validation_from_grid(
            grid, origin, 0.1, thickness=thickness,
            sample_count=40, surface_band=0.05, seed=1702)

        np.testing.assert_array_equal(first.source_indices, second.source_indices)
        np.testing.assert_array_equal(first.points, second.points)
        self.assertFalse(np.array_equal(first.source_indices, other_seed.source_indices))
        self.assertEqual(np.unique(first.source_indices).size, first.size)
        self.assertEqual(np.count_nonzero(first.strata == STRATUM_SURFACE), 20)
        self.assertEqual(np.count_nonzero(first.strata == STRATUM_INSIDE), 10)
        self.assertEqual(np.count_nonzero(first.strata == STRATUM_OUTSIDE), 10)

        index = first.source_indices
        iz, rem = np.divmod(index, grid.shape[1] * grid.shape[2])
        iy, ix = np.divmod(rem, grid.shape[2])
        expected_points = np.column_stack((
            origin[0] + (ix + 0.5) * 0.1,
            origin[1] + (iy + 0.5) * 0.1,
            origin[2] + (iz + 0.5) * 0.1,
        )).astype(np.float32)
        np.testing.assert_allclose(first.points, expected_points, atol=1.0e-7)
        np.testing.assert_array_equal(first.values, grid.ravel()[index])
        np.testing.assert_array_equal(first.thickness, thickness.ravel()[index])
        self.assertAlmostEqual(
            first.thickness_reference,
            float(np.median(thickness[thickness > 0.0])),
            places=7,
        )

    def test_sparse_sample_reserves_coarse_far_field_quota(self) -> None:
        values = np.concatenate((
            np.linspace(-0.02, 0.02, 30, dtype=np.float32),
            np.full(30, -0.5, dtype=np.float32),
            np.full(30, 0.5, dtype=np.float32),
            np.tile(np.array([-0.8, 0.8], dtype=np.float32), 15),
        ))
        points = np.column_stack((
            np.arange(values.size, dtype=np.float32),
            np.zeros(values.size, dtype=np.float32),
            np.ones(values.size, dtype=np.float32),
        ))
        coarse = np.zeros(values.size, dtype=np.bool_)
        coarse[-30:] = True
        source = SdfSampleSet(
            points=points,
            values=values,
            thickness=np.full(values.size, 0.25, dtype=np.float32),
            dx=0.01,
            coarse_mask=coarse,
        )

        sample = stratified_validation_from_samples(
            source, sample_count=40, surface_band=0.05,
            surface_fraction=0.5, coarse_fraction=0.2, seed=44)
        self.assertEqual(np.count_nonzero(sample.strata == STRATUM_SURFACE), 20)
        self.assertEqual(np.count_nonzero(sample.strata == STRATUM_INSIDE), 6)
        self.assertEqual(np.count_nonzero(sample.strata == STRATUM_OUTSIDE), 6)
        self.assertEqual(np.count_nonzero(sample.strata == STRATUM_COARSE_FAR), 8)
        np.testing.assert_array_equal(sample.points, source.points[sample.source_indices])
        np.testing.assert_array_equal(sample.coarse_mask, coarse[sample.source_indices])
        self.assertTrue(np.all(sample.coarse_mask[
            sample.strata == STRATUM_COARSE_FAR]))


class ValidationLossTests(unittest.TestCase):
    @staticmethod
    def _sample() -> ValidationSample:
        values = np.array([-0.02, 0.04, 0.0, 0.5], dtype=np.float32)
        return ValidationSample(
            points=np.zeros((4, 3), dtype=np.float32),
            values=values,
            source_indices=np.arange(4),
            strata=np.zeros(4, dtype=np.uint8),
            dx=0.01,
            thickness=np.array([0.4, 0.2, 0.0, 0.4], dtype=np.float32),
            thickness_reference=0.4,
            coarse_mask=np.array([False, False, False, True]),
        )

    def test_loss_matches_production_formula_term_by_term(self) -> None:
        sample = self._sample()
        prediction = np.array([0.03, -0.06, 0.025, 0.8], dtype=np.float64)
        delta = 0.01
        limit = 0.1
        sigma = 0.05

        result = evaluate_validation_loss(
            prediction,
            sample,
            huber_delta=delta,
            clamp_limit=limit,
            miss_weight=3.0,
            surface_weight=2.0,
            surface_sigma=sigma,
            outside_weight=4.0,
            thin_weight=1.0,
            thin_max_factor=3.0,
            coarse_far_weight=0.15,
            coarse_huber_delta=0.04,
        )

        target = sample.values.astype(np.float64)
        raw = np.clip(prediction, -10.0, 10.0)
        weights = 1.0 + 2.0 * np.exp(-(target * target) / sigma**2)
        thin = np.array([1.0, 2.0, 1.0, 1.0])
        weights *= thin
        error = np.abs(limit * np.tanh(raw / limit)
                       - limit * np.tanh(target / limit))
        huber = np.where(error < delta, 0.5 * error**2 / delta,
                         error - 0.5 * delta)
        expected_reconstruction = float(np.mean(weights * huber))

        miss = np.zeros(4)
        miss[0] = weights[0] * 3.0 * (raw[0] - target[0])
        outside = np.zeros(4)
        outside[1] = weights[1] * 4.0 * (target[1] - raw[1])**2 / sigma
        coarse_error = abs(raw[3] - np.clip(target[3], -10.0, 10.0))
        coarse_huber = (coarse_error - 0.5 * 0.04)
        expected_coarse = 0.15 * coarse_huber / 4.0

        np.testing.assert_allclose(
            result.reconstruction, expected_reconstruction, rtol=1.0e-7, atol=1.0e-10)
        np.testing.assert_allclose(
            result.miss, float(np.mean(miss)), rtol=1.0e-7, atol=1.0e-10)
        np.testing.assert_allclose(
            result.outside, float(np.mean(outside)), rtol=1.0e-7, atol=1.0e-10)
        np.testing.assert_allclose(
            result.coarse_far_field, expected_coarse, rtol=1.0e-7, atol=1.0e-10)
        self.assertAlmostEqual(
            result.total,
            result.reconstruction + result.miss + result.outside
            + result.coarse_far_field,
            places=14,
        )

    def test_huber_delta_is_explicit_and_nonfinite_prediction_is_rejected(self) -> None:
        sample = self._sample()
        with self.assertRaises(TypeError):
            evaluate_validation_loss(np.zeros(4), sample)  # type: ignore[call-arg]
        invalid = evaluate_validation_loss(
            np.array([0.0, np.nan, 0.0, 0.0]), sample, huber_delta=0.005)
        self.assertTrue(np.isinf(invalid.total))


class BestCheckpointTests(unittest.TestCase):
    def test_checkpoint_is_strict_deep_copied_and_patience_aware(self) -> None:
        checkpoint = BestCheckpoint(patience=2, min_delta=0.1)
        centres = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.assertTrue(checkpoint.update(1.0, {"centres": centres}, step=10))
        centres[:] = -1.0

        # A change smaller than min_delta and an equal loss are not genuine
        # improvements; two consecutive misses exhaust patience.
        self.assertFalse(checkpoint.update(
            0.95, {"centres": np.full((1, 3), 9.0, dtype=np.float32)}, step=20))
        self.assertFalse(checkpoint.update(
            1.0, {"centres": np.full((1, 3), 8.0, dtype=np.float32)}, step=30))
        self.assertTrue(checkpoint.should_stop)

        restored = checkpoint.restore()
        np.testing.assert_array_equal(
            restored["centres"], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        restored["centres"][:] = 100.0
        np.testing.assert_array_equal(
            checkpoint.restore()["centres"],
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        )

        self.assertTrue(checkpoint.update(
            0.8, {"centres": np.zeros((1, 3), dtype=np.float32)}, step=40))
        self.assertFalse(checkpoint.should_stop)
        self.assertEqual(checkpoint.failed_checks, 0)
        self.assertEqual(checkpoint.best_step, 40)
        self.assertAlmostEqual(checkpoint.best_loss, 0.8)

    def test_failed_state_copy_does_not_modify_checkpoint(self) -> None:
        checkpoint = BestCheckpoint(patience=None)
        self.assertTrue(checkpoint.update(2.0, {"x": np.array([2.0])}))
        with self.assertRaises(TypeError):
            checkpoint.update(1.0, {"x": [1.0]})  # type: ignore[dict-item]
        self.assertAlmostEqual(checkpoint.best_loss, 2.0)
        np.testing.assert_array_equal(checkpoint.restore()["x"], np.array([2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
