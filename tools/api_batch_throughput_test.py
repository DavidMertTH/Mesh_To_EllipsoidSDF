"""Focused regressions for the API synthetic-pose throughput path.

Run with::

    .venv/Scripts/python.exe tools/api_batch_throughput_test.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest
from unittest import mock

import numpy as np
from PySide6 import QtCore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_window import (  # noqa: E402
    MainWindow,
    SdfWorker,
    _bone_region_sparse_plan,
)
import mesh_io  # noqa: E402
from optimization import OptimizationWorker  # noqa: E402
from sdf_blowup import (  # noqa: E402
    relative_blowup_extent_voxels,
    sparse_band_offsets,
)
from sdf_compute import SdfComputer, SdfResult  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _record


def _box_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([
        [-1.0, -0.5, -0.5], [1.0, -0.5, -0.5],
        [1.0, 0.5, -0.5], [-1.0, 0.5, -0.5],
        [-1.0, -0.5, 0.5], [1.0, -0.5, 0.5],
        [1.0, 0.5, 0.5], [-1.0, 0.5, 0.5],
    ], dtype=np.float32)
    faces = np.asarray([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return vertices, faces


class ApiBatchThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def test_sdf_completion_bypasses_viewer_finalization(self) -> None:
        progress = _Recorder()
        button = _Recorder()
        started: list[bool] = []
        seeded: list[object] = []
        owner = SimpleNamespace(
            _sdf_worker=object(),
            _mesh_settings=SimpleNamespace(blowup_fraction=lambda: 0.0),
            _pending_blowup_capacity_fraction=0.0,
            _api_job_id="a" * 32,
            _api_stage="sdf",
            _api_batch_pipeline=True,
            _last_mesh_result=None,
            _btn_fit=button,
            _progress_set=progress._progress_set,
            _progress_end=progress._progress_end,
            _api_seed_pose_thickness_cache=seeded.append,
            _api_start_fit=lambda: started.append(True),
        )
        result = SimpleNamespace(
            grid=np.zeros((2, 2, 2), dtype=np.float32),
            blowup_thickness=np.ones((2, 2, 2), dtype=np.float32),
            blowup_thickness_extent_vox=3.0,
            blowup_thickness_capacity_fraction=0.1,
            _sdf_blowup_capacity_fraction=0.1,
        )

        # Deliberately omit mesh_sdf_panel/viewer/show_thickness from ``owner``:
        # touching any interactive finalizer makes this call fail immediately.
        MainWindow._on_sdf_done(owner, result)
        self.app.processEvents()

        self.assertIs(owner._last_mesh_result, result)
        self.assertIsNone(result.blowup_thickness)
        self.assertEqual(result.blowup_thickness_extent_vox, 0.0)
        self.assertEqual(seeded, [result])
        self.assertEqual(started, [True])
        self.assertEqual(button.calls, [])
        self.assertEqual(progress.calls, [])

    def test_batch_mesh_swap_avoids_panel_and_rig_rebuilds(self) -> None:
        canceled: list[bool] = []
        owner = SimpleNamespace(
            _cancel_pending_pose_sdf=lambda: canceled.append(True),
            _base_verts=None,
            _base_faces=None,
            _last_mesh_result=object(),
            _mesh_rot_center=None,
            _mesh_rotation=None,
            _pending_rot_mesh=object(),
        )
        vertices = np.array(
            [[-2.0, 0.0, 1.0], [4.0, 2.0, -1.0]], dtype=np.float64)
        faces = np.array([[0, 1, 0]], dtype=np.int64)

        MainWindow._set_api_batch_mesh(owner, vertices, faces)

        self.assertEqual(canceled, [True])
        self.assertEqual(owner._base_verts.dtype, np.float32)
        np.testing.assert_allclose(owner._mesh_rot_center, [1.0, 1.0, 0.0])
        self.assertIsNone(owner._last_mesh_result)
        self.assertIsNone(owner._pending_rot_mesh)

    def test_pose_mesh_reuses_only_identical_oriented_topology(self) -> None:
        owner = SimpleNamespace()
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32)
        faces = np.array(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.int32)
        with mock.patch.object(
            mesh_io.trimesh.repair, "fix_normals",
            wraps=mesh_io.trimesh.repair.fix_normals,
        ) as repair:
            first, _ = MainWindow._api_prepare_mesh_arrays(
                owner, vertices, faces, True)
            second, _ = MainWindow._api_prepare_mesh_arrays(
                owner, vertices * 1.1, faces.copy(), True)
            self.assertEqual(repair.call_count, 1)
            np.testing.assert_array_equal(first.faces, second.faces)
            changed = faces.copy()
            changed[0] = changed[0, ::-1]
            MainWindow._api_prepare_mesh_arrays(
                owner, vertices, changed, True)
            self.assertEqual(repair.call_count, 2)

    @staticmethod
    def _pose_thickness_owner() -> SimpleNamespace:
        vertices = np.array(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
             [-0.5, 0.5, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return SimpleNamespace(
            _api_batch_pipeline=True,
            _api_fit_existing=True,
            _api_batch_id="pose-batch-a",
            _api_pose_index=0,
            _api_pose_thickness_cache=None,
            _api_norm=SimpleNamespace(
                center=np.zeros(3, dtype=np.float64), scale=2.0),
            _base_verts=vertices,
            _base_faces=faces,
            _settings={
                "use_sparse_sdf": True,
                "thickness_max_resolution": 64,
            },
        )

    @staticmethod
    def _uniform_thickness_result(
        value: float = 4.0,
        n: int = 3,
    ) -> SdfResult:
        grid = np.zeros((n, n, n), dtype=np.float32)
        return SdfResult(
            grid=grid,
            n=n,
            dx=1.0,
            origin=np.full(3, -1.0, dtype=np.float32),
            aabb_min=np.full(3, -1.0, dtype=np.float32),
            aabb_max=np.full(3, 1.0, dtype=np.float32),
            thickness=np.full_like(grid, value),
        )

    def test_pose_thickness_cache_reuses_matching_batch_in_new_normalization(
            self) -> None:
        owner = self._pose_thickness_owner()
        result = self._uniform_thickness_result(value=4.0, n=3)

        # The first pose stores normalized lengths (4) in original mesh units
        # using scale 2.  A later pose normalized with scale 3 must therefore
        # receive thickness 6, rather than blindly reusing the old value 4.
        MainWindow._api_seed_pose_thickness_cache(owner, result)
        owner._api_pose_index = 1
        owner._api_norm = SimpleNamespace(
            center=np.ones(3, dtype=np.float64), scale=3.0)
        owner._base_verts = np.ascontiguousarray(
            owner._base_verts * np.array([0.8, 1.1, 1.0], dtype=np.float32)
            + np.array([0.2, -0.1, 0.3], dtype=np.float32)
        )

        reused = MainWindow._api_reusable_pose_face_corner_thickness(
            owner, n=3, blowup_fraction=0.0)

        self.assertIsNotNone(reused)
        self.assertEqual(reused.dtype, np.float32)
        self.assertEqual(reused.shape, owner._base_faces.shape)
        np.testing.assert_allclose(reused, 6.0)
        self.assertIn(
            "original_face_corner_thickness",
            owner._api_pose_thickness_cache,
        )

    def test_pose_thickness_cache_contract_and_supported_modes(
            self) -> None:
        owner = self._pose_thickness_owner()
        MainWindow._api_seed_pose_thickness_cache(
            owner, self._uniform_thickness_result())
        original_faces = owner._base_faces.copy()
        original_vertices = owner._base_verts.copy()

        def lookup(*, n: int = 3, blowup: float = 0.0):
            return MainWindow._api_reusable_pose_face_corner_thickness(
                owner, n=n, blowup_fraction=blowup)

        owner._api_batch_id = "pose-batch-b"
        self.assertIsNone(lookup(), "a new explicit batch must miss")
        owner._api_batch_id = "pose-batch-a"

        owner._base_faces = np.array([[0, 2, 1]], dtype=np.int32)
        self.assertIsNone(lookup(), "changed topology must miss")
        owner._base_faces = original_faces

        owner._base_verts = np.vstack([
            original_vertices,
            np.array([[0.5, 0.5, 0.0]], dtype=np.float32),
        ])
        self.assertIsNone(lookup(), "changed vertex count must miss")
        owner._base_verts = original_vertices

        owner._api_batch_id = None
        self.assertIsNone(lookup(), "requests without an explicit batch must miss")
        owner._api_batch_id = "pose-batch-a"

        self.assertIsNotNone(
            lookup(blowup=0.1),
            "transported thickness must also feed thickness-relative blowup",
        )
        self.assertIsNone(lookup(n=4), "changed SDF resolution must miss")

        owner._settings["thickness_max_resolution"] = 32
        self.assertIsNone(
            lookup(), "changed thickness resolution must miss")
        owner._settings["thickness_max_resolution"] = 64

        owner._settings["use_sparse_sdf"] = False
        self.assertIsNotNone(
            lookup(), "dense SDF fits must use the GPU-transported field")
        owner._settings["use_sparse_sdf"] = True

        owner._api_batch_pipeline = False
        self.assertIsNone(lookup(), "non-batch fits must miss")

    def test_pose_thickness_cache_rejects_unresolved_surface_values(
            self) -> None:
        owner = self._pose_thickness_owner()
        MainWindow._api_seed_pose_thickness_cache(
            owner, self._uniform_thickness_result(value=0.0))
        self.assertIsNone(owner._api_pose_thickness_cache)

    def test_pose_thickness_cache_rejects_one_fully_unresolved_face(
            self) -> None:
        owner = self._pose_thickness_owner()
        owner._base_verts = np.zeros((30, 3), dtype=np.float32)
        owner._base_faces = np.arange(30, dtype=np.int32).reshape(10, 3)
        sampled = np.ones((10, 3), dtype=np.float32)
        sampled[-1] = 0.0
        result = self._uniform_thickness_result()
        result._pose_face_corner_thickness = sampled

        MainWindow._api_seed_pose_thickness_cache(owner, result)

        # A global 90% resolved-value threshold would accept this exact case,
        # even though the last triangle would lose both thin-feature weighting
        # and relative blowup.  Every face must retain a resolved value.
        self.assertIsNone(owner._api_pose_thickness_cache)

    def test_dense_worker_uses_transported_thickness_for_blowup(
            self) -> None:
        vertices, faces = _box_mesh()
        computer = SdfComputer(device="cpu")
        computer.set_mesh(vertices, faces)
        completed: list[SdfResult] = []
        failed: list[str] = []
        worker = SdfWorker(
            computer,
            n=16,
            margin=0.5,
            compute_thickness=False,
            compute_blowup_thickness=True,
            compute_sparse_samples=False,
            transported_face_corner_thickness=np.ones(
                faces.shape, dtype=np.float32),
            max_dist=10.0,
            sdf_blowup_fraction=0.1,
        )
        worker.done.connect(completed.append)
        worker.failed.connect(failed.append)

        worker.run()

        self.assertEqual(failed, [])
        self.assertEqual(len(completed), 1)
        self.assertIsNotNone(completed[0].thickness)
        self.assertIsNotNone(completed[0].blowup_thickness)
        self.assertAlmostEqual(
            completed[0]._sdf_blowup_applied_fraction, 0.1)

    def test_sparse_worker_applies_transported_blowup_to_every_target(
            self) -> None:
        vertices, faces = _box_mesh()
        computer = SdfComputer(device="cpu")
        computer.set_mesh(vertices, faces)
        completed: list[SdfResult] = []
        failed: list[str] = []
        fraction = 0.2
        face_thickness = 4.0
        worker = SdfWorker(
            computer,
            n=16,
            margin=0.5,
            compute_thickness=False,
            compute_blowup_thickness=True,
            compute_sparse_samples=True,
            transported_face_corner_thickness=np.full(
                faces.shape, face_thickness, dtype=np.float32),
            max_dist=10.0,
            sdf_blowup_fraction=fraction,
            thin_surface_fraction=0.0,
        )
        worker.done.connect(completed.append)
        worker.failed.connect(failed.append)

        worker.run()

        self.assertEqual(failed, [])
        self.assertEqual(len(completed), 1)
        result = completed[0]
        samples = getattr(result, "_sparse_samples", None)
        self.assertIsNotNone(samples)
        self.assertIsNotNone(samples.thickness)
        self.assertIsNotNone(samples.coarse_mask)
        self.assertIsNotNone(result.blowup_thickness)

        raw_sdf = computer.query_points(samples.points, max_dist=10.0)
        expected = raw_sdf + np.float32(fraction) * samples.thickness
        band = ~samples.coarse_mask
        positive_coarse = samples.coarse_mask & (samples.thickness > 0.0)
        self.assertTrue(np.any(band))
        self.assertTrue(np.any(positive_coarse))
        np.testing.assert_allclose(
            samples.values[band], expected[band], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(
            samples.values[positive_coarse],
            expected[positive_coarse],
            rtol=0.0,
            atol=2.0e-6,
        )

        blowup_vox = relative_blowup_extent_voxels(
            fraction, result.blowup_thickness, result.dx)
        offsets = sparse_band_offsets(
            blowup_vox, base_offsets=(-2.0, -1.0, 0.0, 1.0, 2.0))
        expanded_clouds = [
            i for i, offset in enumerate(offsets) if abs(offset) > 2.0
        ]
        self.assertTrue(expanded_clouds)
        sparse_plan = _bone_region_sparse_plan(
            int(np.prod(result.grid.shape)), offsets=offsets)
        surface_samples = 128 if sparse_plan is None else sparse_plan[0]
        expanded_indices = np.concatenate([
            np.arange(
                cloud * surface_samples,
                (cloud + 1) * surface_samples,
                dtype=np.int64,
            )
            for cloud in expanded_clouds
        ])
        self.assertTrue(np.all(band[expanded_indices]))
        np.testing.assert_allclose(
            samples.thickness[expanded_indices],
            face_thickness,
            rtol=1.0e-6,
            atol=2.0e-6,
        )

    def test_scalar_fit_progress_reaches_api_without_geometry(self) -> None:
        registry = _Recorder()
        status = _Recorder()
        progress = _Recorder()
        owner = SimpleNamespace(
            _api_job_id="c" * 32,
            _api_stage="fit",
            _api_batch_pipeline=True,
            _api_server=SimpleNamespace(registry=registry),
            _api_progress_last_time=0.0,
            _opt_total_steps=80,
            _status=status,
            _progress_set=progress._progress_set,
        )
        MainWindow._on_opt_step_status(owner, 19, 0.25)
        self.assertEqual(len(registry.calls), 1)
        _, args, fields = registry.calls[0]
        self.assertEqual(args, (owner._api_job_id,))
        self.assertEqual(fields["stage"], "fit")
        self.assertEqual(fields["step"], 20)
        self.assertAlmostEqual(fields["stage_progress"], 0.25)
        self.assertEqual(len(status.calls), 1)
        self.assertEqual(len(progress.calls), 1)
        self.assertAlmostEqual(progress.calls[0][1][0], 72.5)

    def test_batch_sdf_progress_updates_local_bar(self) -> None:
        registry = _Recorder()
        progress = _Recorder()
        owner = SimpleNamespace(
            _api_job_id="d" * 32,
            _api_server=SimpleNamespace(registry=registry),
            _api_batch_pipeline=True,
            _api_progress_last_time=0.0,
            _progress_set=progress._progress_set,
            _status=_Recorder(),
        )
        MainWindow._on_sdf_progress(owner, 0.5, "Thickness spheres")
        self.assertEqual(registry.calls[0][2]["stage"], "sdf")
        self.assertAlmostEqual(progress.calls[0][1][0], 31.5)
        self.assertIn("Thickness spheres", progress.calls[0][1][1])

    def test_batch_reset_hides_local_bar(self) -> None:
        progress = _Recorder()
        owner = SimpleNamespace(
            _api_batch_pipeline=True,
            _progress_end=progress._progress_end,
        )
        MainWindow._api_reset(owner)
        self.assertEqual(len(progress.calls), 1)
        self.assertFalse(owner._api_batch_pipeline)

    def test_api_reset_clears_active_pose_ids_but_keeps_thickness_cache(
            self) -> None:
        progress = _Recorder()
        cache = {"sentinel": object()}
        owner = SimpleNamespace(
            _api_batch_pipeline=True,
            _api_batch_id="pose-batch-a",
            _api_pose_index=7,
            _api_pose_thickness_cache=cache,
            _progress_end=progress._progress_end,
        )

        MainWindow._api_reset(owner)

        self.assertIsNone(owner._api_batch_id)
        self.assertIsNone(owner._api_pose_index)
        self.assertIs(owner._api_pose_thickness_cache, cache)

    def test_batch_progress_keeps_result_without_render_frame(self) -> None:
        registry = _Recorder()
        owner = SimpleNamespace(
            _api_job_id="b" * 32,
            _api_stage="fit",
            _api_batch_pipeline=True,
            _api_last=None,
            _api_server=SimpleNamespace(registry=registry),
            _api_progress_last_time=time.monotonic() - 1.0,
            _opt_total_steps=80,
            _pending_visual="unchanged",
        )
        centers = np.zeros((3, 3), dtype=np.float32)
        radii = np.ones((3, 3), dtype=np.float32)
        rotations = np.zeros((3, 4), dtype=np.float32)

        MainWindow._on_opt_step_visual(
            owner, 20, 0.25, centers, radii, rotations)

        self.assertIs(owner._api_last[0], centers)
        np.testing.assert_allclose(owner._api_last[3], np.ones((3, 2)))
        self.assertEqual(owner._pending_visual, "unchanged")
        self.assertEqual(registry.calls, [])

    def test_optimizer_accepts_final_only_progress_mode(self) -> None:
        worker = OptimizationWorker(
            sdf_target_np=np.zeros((2, 2, 2), dtype=np.float32),
            origin=np.zeros(3, dtype=np.float32),
            dx=1.0,
            n=2,
            num_steps=1,
            emit_intermediate_progress=False,
        )
        self.assertFalse(worker._emit_intermediate_progress)

    def test_optimizer_failure_is_explicit_and_finishes_once(self) -> None:
        worker = OptimizationWorker(
            sdf_target_np=np.zeros((2, 2, 2), dtype=np.float32),
            origin=np.zeros(3, dtype=np.float32),
            dx=1.0,
            n=2,
            num_steps=1,
        )
        failures: list[str] = []
        finishes: list[bool] = []

        def _fail() -> None:
            raise RuntimeError("expected test failure")

        worker._run_adam = _fail
        worker.failed.connect(failures.append)
        worker.finished.connect(lambda: finishes.append(True))
        with mock.patch("traceback.print_exc"):
            worker.start()
            self.assertTrue(worker.wait(5000))
        self.app.processEvents()

        self.assertEqual(len(failures), 1)
        self.assertIn("expected test failure", failures[0])
        self.assertEqual(finishes, [True])
        self.assertEqual(worker.error_message, failures[0])

    def test_api_busy_guard_waits_for_stale_worker_cleanup(self) -> None:
        idle = SimpleNamespace(
            _sdf_worker=None,
            _sdf_finalize_active=False,
            _opt_worker=None,
            _pose_corrective_worker=None,
            _region_sdf_worker=None,
            _region_fit_worker=None,
            _batched_worker=None,
            _bonesep_ctl=None,
            _region_fit_active=False,
        )
        self.assertFalse(MainWindow._api_has_local_work_in_flight(idle))
        idle._sdf_finalize_active = True
        self.assertTrue(MainWindow._api_has_local_work_in_flight(idle))
        idle._sdf_finalize_active = False
        idle._opt_worker = object()
        self.assertTrue(MainWindow._api_has_local_work_in_flight(idle))

    def test_stale_worker_events_and_fit_timer_cannot_touch_new_job(self) -> None:
        old_sdf, new_sdf = object(), object()
        sdf_results: list[object] = []
        owner = SimpleNamespace(
            _sdf_worker=new_sdf,
            _sdf_worker_generation=2,
            _api_job_id="b" * 32,
            _api_stage="sdf",
            _sdf_worker_event_is_current=lambda worker, generation, job_id:
                MainWindow._sdf_worker_event_is_current(
                    owner, worker, generation, job_id),
            _on_sdf_done=sdf_results.append,
        )
        MainWindow._on_sdf_worker_done(
            owner, old_sdf, 1, "a" * 32, object())
        self.assertIs(owner._sdf_worker, new_sdf)
        self.assertEqual(sdf_results, [])

        opt_calls: list[str] = []
        old_opt, new_opt = object(), object()
        owner._opt_worker = new_opt
        owner._opt_worker_generation = 4
        owner._api_stage = "fit"
        owner._opt_worker_event_is_current = (
            lambda worker, generation, job_id:
            MainWindow._opt_worker_event_is_current(
                owner, worker, generation, job_id))
        MainWindow._dispatch_opt_worker_signal(
            owner, old_opt, 3, "a" * 32, opt_calls.append, "stale")
        self.assertIs(owner._opt_worker, new_opt)
        self.assertEqual(opt_calls, [])

        starts: list[bool] = []
        timer_owner = SimpleNamespace(
            _api_job_id="b" * 32,
            _api_stage="sdf",
            _api_start_fit=lambda: starts.append(True),
        )
        MainWindow._api_start_fit_if_current(timer_owner, "a" * 32)
        MainWindow._api_start_fit_if_current(timer_owner, "b" * 32)
        self.assertEqual(starts, [True])

    def test_synchronous_api_start_exception_finishes_through_api_fail(self) -> None:
        job_id = "c" * 32
        failures: list[tuple[str, str]] = []

        def raise_start(_job_id: str) -> None:
            raise RuntimeError("constructor exploded")

        owner = SimpleNamespace(
            _api_server=object(),
            _api_job_id=job_id,
            _api_stage="sdf",
            _api_start_fit_impl=raise_start,
            _api_fail=lambda failed_id, message:
                failures.append((failed_id, message)),
        )
        MainWindow._api_start_fit(owner)
        self.assertEqual(failures[0][0], job_id)
        self.assertIn("constructor exploded", failures[0][1])

    def test_batch_terminal_update_does_not_publish_preview(self) -> None:
        registry = _Recorder()
        reset: list[bool] = []
        result = {"count": 1, "ellipsoids": [{"id": 0}]}
        owner = SimpleNamespace(
            _api_job_id="d" * 32,
            _api_server=SimpleNamespace(registry=registry),
            _api_last=(object(), object(), object()),
            _api_build_result_payload=lambda *_args: result,
            _api_train_correctives=False,
            _api_batch_pipeline=True,
            _api_stage="fit",
            _status=_Recorder(),
            _api_reset=lambda: reset.append(True),
        )

        MainWindow._api_finish_fit(owner)

        final_update = registry.calls[-1]
        self.assertEqual(final_update[0], "update")
        self.assertEqual(final_update[2]["state"], "done")
        self.assertNotIn("preview", final_update[2])
        self.assertEqual(reset, [True])

    def test_sdf_done_exception_fails_only_its_captured_api_job(self) -> None:
        old_job = "e" * 32
        new_job = "f" * 32
        worker = object()
        failures: list[tuple[str, str]] = []
        owner = SimpleNamespace(
            _sdf_worker=worker,
            _sdf_worker_generation=7,
            _api_job_id=old_job,
            _api_stage="sdf",
            _api_batch_pipeline=True,
            _mesh_settings=SimpleNamespace(blowup_fraction=lambda: 0.1),
            _pending_blowup_capacity_fraction=0.0,
            _ensure_blowup_thickness=lambda *_args, **_kwargs:
                (_ for _ in ()).throw(RuntimeError("carrier exploded")),
            _sdf_worker_event_is_current=lambda current, generation, job_id:
                MainWindow._sdf_worker_event_is_current(
                    owner, current, generation, job_id),
            _on_sdf_done=lambda result, job_id=None:
                MainWindow._on_sdf_done(owner, result, job_id),
            _on_sdf_failed=lambda message:
                self.fail(f"API failure escaped to manual handler: {message}"),
            _api_fail=lambda job_id, message:
                failures.append((job_id, message)),
        )
        result = SimpleNamespace(
            grid=np.zeros((2, 2, 2), dtype=np.float32),
            _sdf_blowup_capacity_fraction=0.2,
        )

        MainWindow._on_sdf_worker_done(
            owner, worker, 7, old_job, result)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], old_job)
        self.assertIn("carrier exploded", failures[0][1])
        self.assertIsNone(owner._sdf_worker)

        # A queued callback from that failed worker must not fail a successor.
        successor = object()
        owner._sdf_worker = successor
        owner._sdf_worker_generation = 8
        owner._api_job_id = new_job
        MainWindow._on_sdf_worker_done(
            owner, worker, 7, old_job, result)
        self.assertEqual(len(failures), 1)
        self.assertIs(owner._sdf_worker, successor)

    def test_sdf_finalize_timers_are_generation_and_job_guarded(self) -> None:
        callbacks: list[object] = []
        executed: list[str] = []
        progress_ended: list[bool] = []
        owner = SimpleNamespace(
            _sdf_finalize_active=True,
            _sdf_finalize_generation=11,
            _api_job_id=None,
            _api_stage=None,
            _btn_fit=_Recorder(),
            _progress_set=lambda *_args: None,
            _status=_Recorder(),
            _progress_end=lambda: progress_ended.append(True),
            _sdf_worker=None,
            _opt_worker=None,
            _pose_corrective_worker=None,
            _region_sdf_worker=None,
            _region_fit_worker=None,
            _batched_worker=None,
        )
        result = SimpleNamespace(grid=np.zeros((1, 1, 1), dtype=np.float32))
        steps = [(93.0, "store", lambda: executed.append("old"))]

        with mock.patch.object(
                QtCore.QTimer, "singleShot",
                side_effect=lambda _delay, callback: callbacks.append(callback)):
            MainWindow._run_sdf_finalize_steps(
                owner, result, steps,
                finalize_generation=11)
            self.assertEqual(len(callbacks), 1)
            owner._sdf_finalize_generation = 12
            callbacks.pop()()
            self.assertEqual(executed, [])

            # Completion's delayed progress hide belongs to generation 12 and
            # must not affect a new API job that starts during its 250-ms delay.
            owner._sdf_finalize_active = True
            MainWindow._run_sdf_finalize_steps(
                owner, result, [],
                finalize_generation=12)
            self.assertEqual(len(callbacks), 1)
            owner._api_job_id = "1" * 32
            callbacks.pop()()
            self.assertEqual(progress_ended, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
