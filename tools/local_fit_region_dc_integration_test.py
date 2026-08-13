"""Real Local-Fit and Region-D&C integration tests for superquadrics."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_settings import defaults as app_setting_defaults  # noqa: E402
from ellipsoid import best_device  # noqa: E402
from optimization import OptimizationWorker, _PopulationAdam  # noqa: E402
from sdf_compute import SdfComputer, SdfResult  # noqa: E402


def _cube_mesh(half_extent: float = 0.32) -> tuple[np.ndarray, np.ndarray]:
    h = np.float32(half_extent)
    vertices = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ],
        dtype=np.float32,
    )
    # Counter-clockwise as viewed from outside; every edge occurs twice.
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _worker(
    result: SdfResult,
    sdf_computer: SdfComputer,
    eps_mode: str,
    **overrides,
) -> OptimizationWorker:
    options = dict(
        sdf_target_np=result.grid,
        origin=result.origin,
        dx=result.dx,
        n=result.n,
        num_ellipsoids=2,
        max_ellipsoids=3,
        num_steps=10,
        report_every=10,
        batch_size=64,
        sample_budget=64,
        maintenance_every=0,
        superfit=False,
        local_fit=True,
        sdf_computer=sdf_computer,
        primitive_shape="bent_superquadric",
        sq_eps_mode=eps_mode,
        sq_eps1=0.9,
        sq_eps2=1.1,
        sq_unlock_frac=0.0,
        sq_bend_unlock_frac=0.0,
        sq_eps_lr_mult=1.0,
        sq_bend_lr_mult=1.0,
        local_steps=4,
        region_steps=4,
        region_dc_cycles=2,
        split_enabled=True,
        split_per_round=1,
        spawn_per_round=0,
        prune_enabled=False,
        merge_enabled=False,
        fuse_per_round=0,
        min_split_radius_vox=1.0,
        split_margin_vox=0.25,
        # The global schedule is deliberately zero: any local update proves
        # that the exposed local_lr, rather than _lr_at(gstep), drives the fit.
        lr_init=0.0,
        lr_final=0.0,
        local_lr=0.003,
        flat_weight=0.0,
    )
    options.update(overrides)
    worker = OptimizationWorker(**options)
    worker._rng = np.random.default_rng(123)
    return worker


def _initial_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Row 0 is outside the local box and therefore remains a frozen contributor.
    # Row 1 fits inside the box but protrudes through the cube.  The real D&C pass
    # splits it along x, producing two descendants with source lineage 1.
    centers = np.array(
        [[1.0, 0.0, 0.0], [0.05, -0.03, 0.0]], dtype=np.float32)
    radii = np.array(
        [[0.08, 0.08, 0.08], [0.50, 0.42, 0.36]], dtype=np.float32)
    rotations = np.repeat(
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), 2, axis=0)
    bend = np.array([[0.0, 0.0], [0.20, -0.12]], dtype=np.float32)
    return centers, radii, rotations, bend


class LocalFitRegionDcIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        vertices, faces = _cube_mesh()
        cls.sdf_computer = SdfComputer(device=best_device())
        cls.sdf_computer.set_mesh(vertices, faces)
        cls.box_result = cls.sdf_computer.compute_box_grid(
            np.full(3, -0.75, dtype=np.float32),
            np.full(3, 0.75, dtype=np.float32),
            n=16,
            compute_thickness=False,
        )

    def _run_local_fit(
        self,
        eps_mode: str,
        eps: np.ndarray,
    ) -> tuple[OptimizationWorker, tuple[np.ndarray, ...], list[np.ndarray]]:
        worker = _worker(self.box_result, self.sdf_computer, eps_mode)
        centers, radii, rotations, bend = _initial_geometry()
        frames: list[np.ndarray] = []
        worker.step_visual.connect(
            lambda _step, _loss, _c, _r, _q, extra:
            frames.append(np.asarray(extra, dtype=np.float32).copy()))

        result = worker._local_fit_regions(
            centers,
            radii,
            rotations,
            [np.zeros(3, dtype=np.float32)],
            [self.box_result],
            gstep=9,
            population_cap=3,
            eps=eps,
            bend=bend,
            allow_region_dc=True,
        )
        return worker, result, frames

    def test_per_primitive_shape_state_survives_real_local_fit_and_dc_split(
        self,
    ) -> None:
        initial_eps = np.array(
            [[0.55, 0.65], [1.25, 1.35]], dtype=np.float32)
        initial_centers, initial_radii, initial_q, initial_bend = _initial_geometry()

        worker, result, frames = self._run_local_fit(
            "per_primitive", initial_eps)
        out_c, out_r, out_q, out_eps, out_bend = result

        self.assertTrue(frames, "the real local Adam loop emitted no frame")
        self.assertEqual(
            [len(out_c), len(out_r), len(out_q), len(out_eps), len(out_bend)],
            [3, 3, 3, 3, 3],
        )
        for values in result:
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(np.linalg.norm(out_q, axis=1), 1.0, atol=2e-5)
        self.assertTrue(np.all((out_eps >= 0.1) & (out_eps <= 2.0)))

        # A real protrusion split occurred between the two optimization cycles.
        np.testing.assert_array_equal(worker._last_region_dc_all_lineage, [0, 0])
        np.testing.assert_array_equal(worker._last_local_fit_lineage, [0, 1, 1])

        # The frozen prefix is bit-for-bit stable, while both shape parameter
        # groups of the trainable descendants received finite optimizer updates.
        np.testing.assert_array_equal(out_c[0], initial_centers[0])
        np.testing.assert_array_equal(out_r[0], initial_radii[0])
        np.testing.assert_array_equal(out_q[0], initial_q[0])
        np.testing.assert_array_equal(out_eps[0], initial_eps[0])
        np.testing.assert_array_equal(out_bend[0], initial_bend[0])
        self.assertGreater(
            float(np.max(np.abs(out_eps[1:] - initial_eps[1]))), 1e-5)
        self.assertGreater(
            float(np.max(np.abs(out_bend[1:] - initial_bend[1]))), 1e-5)

    def test_shared_eps_remains_one_global_value_through_local_fit_and_dc(
        self,
    ) -> None:
        shared_eps = np.repeat(
            np.array([[0.9, 1.1]], dtype=np.float32), 2, axis=0)

        worker, result, frames = self._run_local_fit("shared", shared_eps)
        out_c, out_r, out_q, out_eps, out_bend = result

        self.assertTrue(frames, "the real local Adam loop emitted no frame")
        self.assertEqual(
            [len(out_c), len(out_r), len(out_q), len(out_eps), len(out_bend)],
            [3, 3, 3, 3, 3],
        )
        for values in result:
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(worker._last_region_dc_all_lineage, [0, 0])
        np.testing.assert_array_equal(worker._last_local_fit_lineage, [0, 1, 1])

        # The fixed contributor and both split descendants must decode the same
        # two shared scalars, both during and after the local optimization.
        for extra in frames:
            eps_frame = extra[:, :2]
            np.testing.assert_allclose(
                eps_frame, np.repeat(eps_frame[:1], len(eps_frame), axis=0),
                rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(
            out_eps, np.repeat(out_eps[:1], len(out_eps), axis=0),
            rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(
            out_eps,
            np.repeat(shared_eps[:1], len(out_eps), axis=0),
            rtol=0.0, atol=2e-6,
            err_msg="a region-local fit must not steer the global shared epsilon",
        )

    def test_hostile_many_step_local_fit_stays_inside_cumulative_guards(
        self,
    ) -> None:
        """The former 0.02/400-step path moved one primitive out of its box."""
        center_factor = 0.1
        radii_factor = 1.1
        tolerance = 5.0e-5
        initial_eps = np.array(
            [[0.55, 0.65], [1.25, 1.35]], dtype=np.float32)
        initial_c, initial_r, initial_q, initial_bend = _initial_geometry()
        worker = _worker(
            self.box_result,
            self.sdf_computer,
            "per_primitive",
            local_steps=400,
            region_steps=400,
            region_dc_cycles=1,
            split_enabled=False,
            local_lr=0.02,
            center_step_radius_frac=0.1,
            center_step_min_vox=0.0,
            center_step_max_vox=0.5,
            local_center_trust_radius_factor=center_factor,
            local_radii_trust_factor=radii_factor,
        )
        frames: list[tuple[np.ndarray, ...]] = []
        progress: list[tuple[int, int]] = []

        def capture_frame(
            _step, _loss, centers, radii, rotations, extra,
        ) -> None:
            shape = np.asarray(extra, dtype=np.float32)
            frames.append((
                np.asarray(centers, dtype=np.float32).copy(),
                np.asarray(radii, dtype=np.float32).copy(),
                np.asarray(rotations, dtype=np.float32).copy(),
                shape[:, :2].copy(),
                shape[:, 2:4].copy(),
            ))

        worker.step_visual.connect(capture_frame)
        worker.local_progress.connect(
            lambda current, total: progress.append((int(current), int(total))))
        result = worker._local_fit_regions(
            initial_c,
            initial_r,
            initial_q,
            [np.zeros(3, dtype=np.float32)],
            [self.box_result],
            gstep=9,
            population_cap=2,
            eps=initial_eps,
            bend=initial_bend,
        )
        out_c, out_r, out_q, out_eps, out_bend = result

        self.assertTrue(frames, "the 400-step local fit emitted no frame")
        self.assertTrue(progress, "the 400-step local fit emitted no progress")
        self.assertEqual(progress[-1], (400, 400))
        np.testing.assert_array_equal(worker._last_local_fit_lineage, [0, 1])

        anchor_scale = float(np.mean(initial_r[1]))
        center_limit = center_factor * anchor_scale
        log_radius_limit = float(np.log(radii_factor))
        box_min = self.box_result.aabb_min
        box_max = self.box_result.aabb_max

        def assert_safe_state(state: tuple[np.ndarray, ...]) -> None:
            centers, radii, rotations, eps, bend = state
            for values in state:
                self.assertTrue(
                    np.isfinite(values).all(),
                    "local fit emitted or returned non-finite geometry",
                )
            np.testing.assert_allclose(
                np.linalg.norm(rotations, axis=1), 1.0,
                rtol=0.0, atol=2.0e-5)

            # Row zero was only a union contributor and must remain bit-for-bit
            # unchanged throughout the hostile high-learning-rate refinement.
            np.testing.assert_array_equal(centers[0], initial_c[0])
            np.testing.assert_array_equal(radii[0], initial_r[0])
            np.testing.assert_array_equal(rotations[0], initial_q[0])
            np.testing.assert_array_equal(eps[0], initial_eps[0])
            np.testing.assert_array_equal(bend[0], initial_bend[0])

            center_delta = float(np.linalg.norm(centers[1] - initial_c[1]))
            self.assertLessEqual(
                center_delta,
                center_limit + tolerance,
                "per-step limiting did not enforce the cumulative centre trust region",
            )
            log_radius_delta = np.abs(
                np.log(np.maximum(radii[1], 1.0e-12) / initial_r[1]))
            self.assertLessEqual(
                float(np.max(log_radius_delta)),
                log_radius_limit + tolerance,
                "local radii escaped their cumulative multiplicative trust region",
            )

            # Centre-in-box is insufficient for rotated/bent primitives.  Check
            # the conservative AABB of the complete trainable primitive.
            low, high = worker._primitive_aabbs(
                centers, radii, rotations, bend)
            self.assertGreaterEqual(
                float(np.min(low[1] - box_min)), -tolerance,
                "the trainable primitive's actual AABB escaped below its box",
            )
            self.assertGreaterEqual(
                float(np.min(box_max - high[1])), -tolerance,
                "the trainable primitive's actual AABB escaped above its box",
            )

        for frame in frames:
            assert_safe_state(frame)
        assert_safe_state((out_c, out_r, out_q, out_eps, out_bend))
        self.assertGreater(
            float(np.linalg.norm(out_c[1] - initial_c[1]))
            + float(np.max(np.abs(np.log(out_r[1] / initial_r[1])))),
            1.0e-5,
            "the guard test must still exercise a real local refinement",
        )

    def test_safe_local_fit_defaults_produce_400_effective_region_steps(
        self,
    ) -> None:
        settings = app_setting_defaults()
        self.assertEqual(settings["local_steps"], 400)
        self.assertEqual(settings["region_steps"], 400)
        self.assertAlmostEqual(float(settings["local_lr"]), 0.001)

        # Verify constructor defaults as well as the UI settings.  This catches
        # the historical max(local_steps, region_steps) trap where changing only
        # region_steps still left 1,200 effective local steps.
        worker = OptimizationWorker(
            sdf_target_np=self.box_result.grid,
            origin=self.box_result.origin,
            dx=self.box_result.dx,
            n=self.box_result.n,
            num_ellipsoids=2,
            maintenance_every=0,
            local_fit=False,
        )
        self.assertEqual(worker._local_steps, 400)
        self.assertEqual(worker._region_steps, 400)
        self.assertAlmostEqual(float(worker._local_lr), 0.001)

    def test_nonfinite_local_row_is_restored_and_plain_local_fit_skips_dc(
        self,
    ) -> None:
        initial_eps = np.array(
            [[0.55, 0.65], [1.25, 1.35]], dtype=np.float32)
        initial_c, initial_r, initial_q, initial_bend = _initial_geometry()
        worker = _worker(
            self.box_result,
            self.sdf_computer,
            "per_primitive",
            local_steps=6,
            region_steps=6,
            region_dc_cycles=3,
            split_enabled=True,
        )
        worker._region_dc_all_boxes = lambda *_a, **_k: self.fail(
            "ordinary Local Fit must not run population D&C")
        frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        worker.step_visual.connect(
            lambda _step, _loss, c, r, q, _extra: frames.append((
                np.asarray(c, np.float32).copy(),
                np.asarray(r, np.float32).copy(),
                np.asarray(q, np.float32).copy(),
            )))

        original_step = _PopulationAdam.step
        center_optimizer: list[_PopulationAdam] = []
        center_steps = 0

        def injecting_step(optimizer, gradient):
            nonlocal center_steps
            original_step(optimizer, gradient)
            if not center_optimizer:
                center_optimizer.append(optimizer)
            if optimizer is center_optimizer[0]:
                center_steps += 1
                if center_steps == worker._region_steps:
                    values = optimizer.param.numpy().copy()
                    values[-1] = np.array(
                        [np.nan, np.nan, np.nan], dtype=np.float32)
                    optimizer.param.assign(np.ascontiguousarray(values))

        try:
            _PopulationAdam.step = injecting_step
            result = worker._local_fit_regions(
                initial_c,
                initial_r,
                initial_q,
                [np.zeros(3, dtype=np.float32)],
                [self.box_result],
                gstep=9,
                population_cap=3,
                eps=initial_eps,
                bend=initial_bend,
                allow_region_dc=False,
            )
        finally:
            _PopulationAdam.step = original_step

        self.assertEqual(center_steps, 6)
        self.assertTrue(frames)
        for frame in frames:
            for values in frame:
                self.assertTrue(np.isfinite(values).all())
        for values in result:
            self.assertTrue(np.isfinite(values).all())
        out_c, out_r, out_q, out_eps, out_bend = result
        np.testing.assert_array_equal(out_c[0], initial_c[0])
        np.testing.assert_array_equal(out_r[0], initial_r[0])
        np.testing.assert_array_equal(out_q[0], initial_q[0])
        np.testing.assert_array_equal(out_eps[0], initial_eps[0])
        np.testing.assert_array_equal(out_bend[0], initial_bend[0])
        np.testing.assert_array_equal(out_c[1], initial_c[1])
        np.testing.assert_array_equal(out_r[1], initial_r[1])
        np.testing.assert_array_equal(out_q[1], initial_q[1])
        np.testing.assert_array_equal(out_eps[1], initial_eps[1])
        np.testing.assert_array_equal(out_bend[1], initial_bend[1])


if __name__ == "__main__":
    unittest.main()
