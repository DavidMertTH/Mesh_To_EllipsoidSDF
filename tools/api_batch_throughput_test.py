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

from main_window import MainWindow  # noqa: E402
import mesh_io  # noqa: E402
from optimization import OptimizationWorker  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _record


class ApiBatchThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def test_sdf_completion_bypasses_viewer_finalization(self) -> None:
        progress = _Recorder()
        button = _Recorder()
        started: list[bool] = []
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
