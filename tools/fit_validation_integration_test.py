"""Small end-to-end regression tests for optimizer validation integration.

Run: .venv/Scripts/python.exe tools/fit_validation_integration_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warp as wp  # noqa: E402

from fit_validation import (  # noqa: E402
    STRATUM_COARSE_FAR,
    STRATUM_SURFACE,
)
from optimization import OptimizationWorker  # noqa: E402
from sdf_samples import SdfSampleSet  # noqa: E402


def _identity_quats(count: int) -> np.ndarray:
    return np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (int(count), 1),
    )


def _worker(**overrides) -> OptimizationWorker:
    """A tiny deterministic Adam fit whose geometry is valid from step zero."""
    kwargs = dict(
        sdf_target_np=np.zeros((5, 5, 5), dtype=np.float32),
        origin=np.zeros(3, dtype=np.float32),
        dx=0.1,
        n=5,
        num_ellipsoids=1,
        initial_centers=np.array([[0.25, 0.25, 0.25]], dtype=np.float32),
        initial_radii=np.array([[0.14, 0.11, 0.09]], dtype=np.float32),
        initial_rotations=_identity_quats(1),
        primitive_shape="ellipsoid",
        num_steps=3,
        report_every=1,
        validation_every=1,
        validation_sample_size=32,
        validation_patience=None,
        sample_budget=64,
        maintenance_every=0,
        superfit=False,
        local_fit=False,
        symmetry_enabled=False,
        containment_weight=0.0,
        flat_weight=0.0,
        lr_init=0.0,
        lr_final=0.0,
    )
    kwargs.update(overrides)
    return OptimizationWorker(**kwargs)


def _capture_frames(worker: OptimizationWorker) -> list[dict]:
    frames: list[dict] = []

    def capture(step, loss, centers, radii, rotations, extra) -> None:
        frames.append({
            "step": int(step),
            "loss": float(loss),
            "centers": np.asarray(centers, dtype=np.float32).copy(),
            "radii": np.asarray(radii, dtype=np.float32).copy(),
            "rotations": np.asarray(rotations, dtype=np.float32).copy(),
            "extra": None if extra is None else np.asarray(extra).copy(),
        })

    worker.step_visual.connect(capture)
    return frames


def _run_direct(worker: OptimizationWorker) -> None:
    # Calling the implementation directly lets failures propagate into unittest;
    # QThread.run() intentionally catches and prints worker exceptions.
    worker._reset_stale_tape()
    worker._run_adam()


class ValidationIntegrationTest(unittest.TestCase):
    def test_patience_is_not_delayed_by_disabled_densification(self) -> None:
        worker = _worker(
            num_steps=8,
            report_every=1,
            validation_every=1,
            validation_patience=1,
            superfit=False,
            local_fit=False,
        )

        with patch(
            "optimization.evaluate_validation_loss",
            return_value=SimpleNamespace(total=1.0),
        ):
            _run_direct(worker)

        # Two validation intervals are deliberately reserved for settling.
        # At step 2 the patience monitor gets its baseline; the unchanged check
        # at step 3 must stop the otherwise eight-step run.
        np.testing.assert_array_equal(
            [step for step, _loss in worker.validation_history],
            [0, 1, 2, 3],
        )

    def test_sparse_surface_fraction_above_point_eight_is_capped(self) -> None:
        values = np.concatenate((
            np.zeros(50, dtype=np.float32),
            np.full(50, 0.5, dtype=np.float32),
        ))
        points = np.column_stack((
            np.linspace(0.0, 1.0, values.size, dtype=np.float32),
            np.zeros(values.size, dtype=np.float32),
            np.zeros(values.size, dtype=np.float32),
        ))
        coarse = np.zeros(values.size, dtype=np.bool_)
        coarse[50:] = True
        samples = SdfSampleSet(
            points=points,
            values=values,
            dx=0.01,
            coarse_mask=coarse,
        )
        worker = _worker(
            sdf_samples=samples,
            surface_fraction=0.95,
            validation_sample_size=40,
        )

        validation = worker._build_validation_sample()

        self.assertEqual(validation.size, 40)
        self.assertEqual(
            np.count_nonzero(validation.strata == STRATUM_SURFACE), 32)
        self.assertEqual(
            np.count_nonzero(validation.strata == STRATUM_COARSE_FAR), 8)
        self.assertTrue(np.all(validation.coarse_mask[
            validation.strata == STRATUM_COARSE_FAR]))

    def test_final_signal_restores_best_population_ids_shape_step_and_validation_loss(self) -> None:
        worker = _worker(
            superfit=True,
            max_ellipsoids=2,
            num_steps=3,
        )
        frames = _capture_frames(worker)

        def force_population_growth(
            self,
            step,
            pred_centers,
            pred_radii,
            pred_rot_flat,
            pred_eps=None,
            pred_bend=None,
        ):
            if int(step) != 1:
                return None
            wp.synchronize_device()
            centers = pred_centers.numpy().reshape(-1, 3).copy()
            radii = pred_radii.numpy().reshape(-1, 3).copy()
            rotations = pred_rot_flat.numpy().reshape(-1, 4).copy()
            eps = pred_eps.numpy().reshape(-1, 2).copy()
            bend = pred_bend.numpy().reshape(-1, 2).copy()
            self._last_population_lineage = np.array([0, -1], dtype=np.int64)
            return (
                np.concatenate([
                    centers,
                    np.array([[0.40, 0.25, 0.25]], dtype=np.float32),
                ]),
                np.concatenate([
                    radii,
                    np.array([[0.05, 0.04, 0.03]], dtype=np.float32),
                ]),
                np.concatenate([rotations, _identity_quats(1)]),
                np.concatenate([eps, self._init_eps(1)]),
                np.concatenate([bend, self._init_bend(1)]),
            )

        worker._maybe_superfit = MethodType(force_population_growth, worker)
        validation_values = iter((123.0, 124.0, 125.0))

        with patch(
            "optimization.evaluate_validation_loss",
            side_effect=lambda *_args, **_kwargs: SimpleNamespace(
                total=next(validation_values)),
        ):
            _run_direct(worker)

        self.assertGreaterEqual(len(frames), 4)
        first_checkpoint_frame = frames[0]
        final_frame = frames[-1]
        self.assertTrue(any(len(frame["centers"]) == 2 for frame in frames[1:-1]))
        self.assertEqual(len(final_frame["centers"]), 1)
        np.testing.assert_allclose(
            final_frame["centers"], first_checkpoint_frame["centers"])
        np.testing.assert_allclose(
            final_frame["radii"], first_checkpoint_frame["radii"])
        np.testing.assert_allclose(
            final_frame["rotations"], first_checkpoint_frame["rotations"])
        np.testing.assert_array_equal(worker._primitive_ids, np.array([0]))
        self.assertEqual(worker.best_validation_step, 0)
        self.assertEqual(worker.best_validation_loss, 123.0)

        # The restored frame is labelled with the exact metric and step that
        # selected its geometry, rather than a stale pre-update mini-batch loss.
        self.assertEqual(final_frame["step"], 0)
        self.assertAlmostEqual(final_frame["loss"], 123.0, places=5)

    def test_nonfinite_late_geometry_is_not_checkpointed_and_best_is_restored(self) -> None:
        worker = _worker(num_steps=2)
        frames = _capture_frames(worker)
        projection_calls = 0

        def inject_nan_after_second_step(
            self, pred_log_radii, _pred_rot_flat, _num_e,
        ) -> None:
            nonlocal projection_calls
            projection_calls += 1
            if projection_calls == 2:
                values = pred_log_radii.numpy().copy()
                values[0, 0] = np.nan
                pred_log_radii.assign(np.ascontiguousarray(values))

        worker._project_isotropic = MethodType(
            inject_nan_after_second_step, worker)

        with patch(
            "optimization.evaluate_validation_loss",
            return_value=SimpleNamespace(total=0.25),
        ) as evaluate:
            _run_direct(worker)

        self.assertGreaterEqual(len(frames), 3)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(worker.best_validation_step, 0)
        self.assertEqual(worker.best_validation_loss, 0.25)
        self.assertTrue(np.isinf(worker.validation_history[-1][1]))
        self.assertFalse(np.isfinite(frames[-2]["radii"]).all())
        self.assertTrue(np.isfinite(frames[-1]["radii"]).all())
        np.testing.assert_allclose(frames[-1]["centers"], frames[0]["centers"])
        np.testing.assert_allclose(frames[-1]["radii"], frames[0]["radii"])
        np.testing.assert_array_equal(worker._primitive_ids, np.array([0]))
        self.assertEqual(frames[-1]["step"], 0)
        self.assertAlmostEqual(frames[-1]["loss"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
