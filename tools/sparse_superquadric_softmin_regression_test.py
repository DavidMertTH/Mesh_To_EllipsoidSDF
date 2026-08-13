"""Regression coverage for sparse Bent-SQ soft-union fitting.

The test intentionally uses box-like superquadric exponents and non-zero bend.
It covers the point-backed soft-min kernel under autodiff, one Adam update,
sparse thickness upload/importance sampling, and the global worker dispatch.

Run: .venv/Scripts/python.exe tools/sparse_superquadric_softmin_regression_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import MethodType
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warp as wp  # noqa: E402

import optimization as optimization_module  # noqa: E402
from optimization import (  # noqa: E402
    BandSampler,
    OptimizationWorker,
    _PopulationAdam,
    _rmse_loss_kernel_batch,
    _superquadric_softmin_kernel_points,
    device,
)
from sdf_samples import SdfSampleSet  # noqa: E402


SOFT_K = 35.0
SURFACE_BAND = 0.08


def _identity_quats(count: int) -> np.ndarray:
    return np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (int(count), 1),
    )


def _assert_finite(test: unittest.TestCase, name: str, values) -> None:
    array = np.asarray(values)
    test.assertTrue(
        np.isfinite(array).all(),
        msg=f"{name} contains non-finite values",
    )


def _softmin_values(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    rotations: np.ndarray,
    eps: np.ndarray,
    bend: np.ndarray,
) -> np.ndarray:
    count = int(len(points))
    num_e = int(len(centers))
    wp_centers = wp.array(centers, dtype=wp.vec3, device=device)
    wp_radii = wp.array(radii, dtype=wp.vec3, device=device)
    wp_rotations = wp.array(
        np.ascontiguousarray(rotations.reshape(-1)),
        dtype=wp.float32, device=device)
    wp_eps = wp.array(
        np.ascontiguousarray(eps.reshape(-1)),
        dtype=wp.float32, device=device)
    wp_bend = wp.array(
        np.ascontiguousarray(bend.reshape(-1)),
        dtype=wp.float32, device=device)
    wp_points = wp.array(points, dtype=wp.vec3, device=device)
    wp_indices = wp.array(
        np.arange(count, dtype=np.int32), dtype=wp.int32, device=device)
    min_cache = wp.zeros((count, num_e + 1), dtype=wp.float32, device=device)
    sum_cache = wp.zeros((count, num_e + 1), dtype=wp.float32, device=device)
    prediction = wp.empty(count, dtype=wp.float32, device=device)
    wp.launch(
        _superquadric_softmin_kernel_points,
        dim=count,
        inputs=[
            wp_centers, wp_radii, wp_rotations, wp_eps, wp_bend,
            min_cache, sum_cache, num_e, wp_points, wp_indices,
            prediction, float(SOFT_K),
        ],
        device=device,
    )
    return prediction.numpy().copy()


def _sparse_fixture() -> tuple[SdfSampleSet, dict[str, np.ndarray]]:
    """Return deterministic sparse targets sampled from a boxy bent SQ union."""
    true = {
        "centers": np.array(
            [[-0.17, 0.00, 0.00], [0.18, 0.01, 0.01]], dtype=np.float32),
        "radii": np.array(
            [[0.24, 0.18, 0.17], [0.21, 0.16, 0.19]], dtype=np.float32),
        "rotations": _identity_quats(2),
        # Both rows are deliberately far below epsilon=1 (box-like).
        "eps": np.array([[0.20, 0.26], [0.32, 0.22]], dtype=np.float32),
        "bend": np.array([[1.45, -0.85], [-1.10, 0.70]], dtype=np.float32),
    }
    rng = np.random.default_rng(0x5DF)
    candidates = rng.uniform(
        low=np.array([-0.55, -0.38, -0.34]),
        high=np.array([0.55, 0.38, 0.34]),
        size=(2048, 3),
    ).astype(np.float32)
    candidate_values = _softmin_values(candidates, **true)

    surface = np.flatnonzero(np.abs(candidate_values) < SURFACE_BAND)
    far = np.flatnonzero(np.abs(candidate_values) >= SURFACE_BAND)
    if surface.size < 192 or far.size < 96:
        raise AssertionError(
            f"fixture lacks strata: surface={surface.size}, far={far.size}")
    surface = surface[:256]
    far = far[:128]
    selected = np.concatenate([surface, far])
    points = candidates[selected]
    values = candidate_values[selected]

    # Half of the surface points form an unmistakable thin-feature population;
    # the other half supplies the mesh-specific median used by BandSampler.
    thickness = np.zeros(len(selected), dtype=np.float32)
    surface_count = len(surface)
    split = surface_count // 2
    thickness[:split] = 0.01
    thickness[split:surface_count] = 0.10

    coarse = np.zeros(len(selected), dtype=np.bool_)
    coarse[surface_count:surface_count + 64] = True
    samples = SdfSampleSet(
        points=points,
        values=values,
        thickness=thickness,
        dx=0.04,
        source="boxy-bent-sq-regression",
        coarse_mask=coarse,
    )
    return samples, true


class SparseSuperquadricSoftminRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        cls.samples, cls.true = _sparse_fixture()

    def test_boxy_bent_softmin_gradients_and_adam_update_are_finite(self) -> None:
        samples = self.samples
        count = samples.size
        num_e = 2
        initial = {
            "centers": self.true["centers"]
            + np.array([[0.025, -0.010, 0.005], [-0.020, 0.012, -0.006]],
                       dtype=np.float32),
            "radii": self.true["radii"] * np.array(
                [[0.91, 1.06, 0.94], [1.07, 0.92, 1.04]], dtype=np.float32),
            "rotations": _identity_quats(num_e),
            "eps": np.array([[0.28, 0.34], [0.38, 0.29]], dtype=np.float32),
            "bend": np.array([[1.05, -0.55], [-0.75, 0.45]], dtype=np.float32),
        }

        centers = wp.array(
            initial["centers"], dtype=wp.vec3, device=device, requires_grad=True)
        radii = wp.array(
            initial["radii"], dtype=wp.vec3, device=device, requires_grad=True)
        rotations = wp.array(
            initial["rotations"].reshape(-1), dtype=wp.float32,
            device=device, requires_grad=True)
        eps = wp.array(
            initial["eps"].reshape(-1), dtype=wp.float32,
            device=device, requires_grad=True)
        bend = wp.array(
            initial["bend"].reshape(-1), dtype=wp.float32,
            device=device, requires_grad=True)
        points = wp.array(samples.points, dtype=wp.vec3, device=device)
        targets = wp.array(samples.values, dtype=wp.float32, device=device)
        thickness = wp.array(samples.thickness, dtype=wp.float32, device=device)
        indices = wp.array(
            np.arange(count, dtype=np.int32), dtype=wp.int32, device=device)
        min_cache = wp.zeros(
            (count, num_e + 1), dtype=wp.float32,
            device=device, requires_grad=True)
        sum_cache = wp.zeros(
            (count, num_e + 1), dtype=wp.float32,
            device=device, requires_grad=True)
        prediction = wp.empty(
            count, dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(
            1, dtype=wp.float32, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                _superquadric_softmin_kernel_points,
                dim=count,
                inputs=[
                    centers, radii, rotations, eps, bend,
                    min_cache, sum_cache, num_e, points, indices,
                    prediction, float(SOFT_K),
                ],
                device=device,
            )
            wp.launch(
                _rmse_loss_kernel_batch,
                dim=count,
                inputs=[
                    prediction, targets, indices, loss, count,
                    3.0, 4.0, 0.06, 8.0,
                    thickness, 0.10, 1.0, 6.0, 0.01,
                ],
                device=device,
            )
        tape.backward(loss)

        _assert_finite(self, "softmin prediction", prediction.numpy())
        _assert_finite(self, "loss", loss.numpy())
        parameter_arrays = {
            "centers": centers,
            "radii": radii,
            "rotations": rotations,
            "eps": eps,
            "bend": bend,
        }
        before = {name: array.numpy().copy()
                  for name, array in parameter_arrays.items()}
        for name, array in parameter_arrays.items():
            gradient = array.grad.numpy()
            _assert_finite(self, f"{name} gradient", gradient)
            self.assertGreater(
                float(np.linalg.norm(gradient)), 1.0e-9,
                msg=f"{name} did not receive a useful gradient")

        for array in parameter_arrays.values():
            optimizer = _PopulationAdam(array, lr=5.0e-4)
            optimizer.step([array.grad.flatten()])
        wp.synchronize_device(device)

        for name, array in parameter_arrays.items():
            updated = array.numpy()
            _assert_finite(self, f"updated {name}", updated)
            self.assertGreater(
                float(np.max(np.abs(updated - before[name]))), 1.0e-7,
                msg=f"{name} did not update")
        self.assertGreater(float(np.min(radii.numpy())), 0.0)

    def test_sparse_thickness_upload_and_thin_sampling_quota(self) -> None:
        samples = self.samples
        worker = OptimizationWorker(
            sdf_target_np=np.zeros((8, 8, 8), dtype=np.float32),
            sdf_samples=samples,
            origin=np.array([-0.6, -0.6, -0.6], dtype=np.float32),
            dx=0.04,
            n=8,
            num_ellipsoids=2,
            sample_budget=128,
            maintenance_every=0,
            primitive_shape="bent_superquadric",
        )
        uploaded = worker._ensure_sample_targets_wp()
        np.testing.assert_array_equal(uploaded.thickness.numpy(), samples.thickness)
        self.assertAlmostEqual(
            worker._thick_ref,
            float(np.median(samples.thickness[samples.thickness > 0.0])),
            places=7,
        )
        self.assertEqual(worker._thin_weight_eff, worker._thin_loss_weight)

        sampler = BandSampler(
            samples.values,
            batch_size=128,
            band=SURFACE_BAND,
            surface_fraction=0.70,
            rng=np.random.default_rng(17),
            flat_thickness=samples.thickness,
            thin_bias=1.0,
            coarse_mask=samples.coarse_mask,
        )
        self.assertIsNotNone(sampler._band_thin)
        expected_thin = int(round(sampler.n_surf * sampler._thin_quota))
        self.assertGreater(expected_thin, 0)
        batch = sampler.next_batch()
        thin_pool = set(int(index) for index in sampler._band_thin)
        self.assertTrue(all(int(index) in thin_pool
                            for index in batch[:expected_thin]))
        self.assertTrue(np.all(samples.thickness[batch[:expected_thin]] == 0.01))
        self.assertGreaterEqual(
            np.count_nonzero(samples.coarse_mask[batch]), sampler.n_far)

    def test_global_sparse_bent_worker_dispatches_softmin_and_stays_finite(self) -> None:
        samples = self.samples
        initial_centers = self.true["centers"] + np.array(
            [[0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]], dtype=np.float32)
        initial_radii = self.true["radii"] * 0.92
        initial_bend = np.array(
            [[0.95, -0.45], [-0.65, 0.35]], dtype=np.float32)
        worker = OptimizationWorker(
            sdf_target_np=np.zeros((32, 32, 32), dtype=np.float32),
            sdf_samples=samples,
            origin=np.array([-0.64, -0.64, -0.64], dtype=np.float32),
            dx=0.04,
            n=32,
            num_ellipsoids=2,
            initial_centers=initial_centers,
            initial_radii=initial_radii,
            initial_rotations=_identity_quats(2),
            primitive_shape="bent_superquadric",
            sq_eps1=0.31,
            sq_eps2=0.30,
            sq_eps_mode="per_primitive",
            sq_unlock_frac=0.0,
            sq_bend_unlock_frac=0.0,
            sq_eps_lr_mult=0.25,
            sq_bend_lr_mult=0.10,
            soft_union=True,
            densify_until_frac=1.0,
            num_steps=4,
            report_every=1,
            validation_every=4,
            validation_sample_size=64,
            validation_patience=None,
            sample_budget=128,
            maintenance_every=0,
            superfit=False,
            local_fit=False,
            symmetry_enabled=False,
            containment_weight=0.0,
            flat_weight=0.0,
            lr_init=0.004,
            lr_final=0.001,
        )
        worker._init_bend = MethodType(
            lambda _self, count: np.resize(
                initial_bend, (int(count), 2)).astype(np.float32),
            worker,
        )
        frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        worker.step_visual.connect(
            lambda _step, _loss, centers, radii, rotations, extra:
            frames.append((
                np.asarray(centers, dtype=np.float32).copy(),
                np.asarray(radii, dtype=np.float32).copy(),
                np.asarray(rotations, dtype=np.float32).copy(),
                np.asarray(extra, dtype=np.float32).copy(),
            )))

        original_launch = wp.launch
        softmin_launches = 0

        def tracking_launch(kernel, *args, **kwargs):
            nonlocal softmin_launches
            if kernel is _superquadric_softmin_kernel_points:
                softmin_launches += 1
            return original_launch(kernel, *args, **kwargs)

        worker._reset_stale_tape()
        with patch.object(
            optimization_module.wp, "launch", new=tracking_launch,
        ):
            worker._run_adam()

        self.assertGreaterEqual(softmin_launches, worker._num_steps)
        self.assertGreaterEqual(len(frames), worker._num_steps)
        for frame in frames:
            for name, values in zip(
                    ("centers", "radii", "rotations", "eps/bend"), frame,
                    strict=True):
                _assert_finite(self, f"worker {name}", values)
            self.assertEqual(frame[3].shape, (2, 4))
            self.assertTrue(np.all((frame[3][:, :2] >= 0.1)
                                   & (frame[3][:, :2] <= 2.0)))
            kappa = frame[3][:, 2:] * frame[1][:, 2:3]
            self.assertTrue(np.all(np.abs(kappa) <= worker._bend_kappa_max + 1.0e-5))

        changes = []
        for before, after in zip(frames, frames[1:], strict=False):
            changes.append(max(
                float(np.max(np.abs(after[0] - before[0]))),
                float(np.max(np.abs(after[1] - before[1]))),
                float(np.max(np.abs(after[3] - before[3]))),
            ))
        self.assertGreater(max(changes), 1.0e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
