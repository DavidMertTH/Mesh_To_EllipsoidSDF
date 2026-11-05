"""Native systematic fitting and live morph preview through real Qt controls.

The panel, pose generation, forward kinematics, skinner and morph driver are
real. A small QWidget host runs MainWindow callbacks while replacing expensive
rendering with captured draw parameters. Real T-Pose worker smoke tests use a
tiny SDF and a single optimizer step.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtCore, QtWidgets

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from ellipsoid import best_device
from joint_pose import JointPoseEdits, joint_pose_angles
from main_window import MainWindow
from mesh_settings import rotation_matrix
from native_morph import NativeMorphDriver
from native_morph_io import save_native_morph_fit
from pose_correctives import PoseCorrectiveKey, PoseCorrectiveLibrary, PoseCorrectiveWorker
from rig_loader import RiggedMesh, load_rigged_mesh
from rig_panel import RigModePanel
from skeleton import Bone, Pose, Skeleton, quat_from_matrix, mat4_compose


def make_rig():
    """An arbitrary two-bone rig proves that authoring needs no humanoid names."""
    root, hinge = Bone("anchor", 0), Bone("custom hinge", 1, parent_index=0)
    hinge.local_bind_transform[0, 3] = .2
    hinge.inverse_bind_matrix[0, 3] = -.2
    for bone in (root, hinge):
        for axis in range(3):
            bone.joint_dof.set_axis(axis, False, 0, 0)
    hinge.joint_dof.set_axis(2, True, -60, 90)
    skeleton = Skeleton([root, hinge])
    vertices = np.array([[.2, -.05, -.05], [.8, -.05, -.05],
                         [.2, .05, -.05], [.2, -.05, .05]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    local = hinge.local_bind_transform.copy()
    local[:3, :3] = rotation_matrix(0, 0, -20)
    return RiggedMesh(vertices, faces, skeleton,
        np.ones((len(vertices), 1), np.float32),
        np.ones((len(vertices), 1), np.int32),
        [Pose.t_pose(), Pose("unrelated imported animation", {1: local})], "custom mesh")


def make_library(rig, primitive_type="superquadric"):
    base = BoneLocalEllipsoids(
        local_centers=np.array([[.25, 0, 0]], np.float32),
        local_radii=np.array([[.15, .05, .07]], np.float32),
        local_rotations=np.array([[0, 0, 0, 1]], np.float32),
        bone_assignments=np.array([1], np.int32),
        attachment_joints=np.array([[1]], np.int32),
        attachment_weights=np.ones((1, 1), np.float32),
        primitive_type=primitive_type,
        shape_exponents=np.array([[.8, 1.1]], np.float32))
    edit = JointPoseEdits(rig.skeleton, Pose.t_pose())
    edit.set_axis(1, 2, 60)
    key = PoseCorrectiveKey("sixty degree support",
        delta_centers=np.array([[.025, .01, 0]], np.float32),
        delta_rotations=np.array([[0, 0, .1, np.sqrt(.99)]], np.float32),
        delta_log_radii=np.log(np.array([[1.4, .8, 1.1]], np.float32)),
        delta_log_shape_exponents=np.log(np.array([[1.1, .9]], np.float32)),
        loss=.03, pose=edit.apply())
    return PoseCorrectiveLibrary(base, [key], base_pose=Pose.t_pose())


class WindowHost(QtWidgets.QWidget):
    """Invoke production callbacks with real panel/data and cheap draw capture."""
    def __init__(self, panel, library=None):
        super().__init__()
        self._rig_panel = panel
        self._pose_correctives = library
        self._morph_driver = None
        self._morph_preview_timer = QtCore.QTimer(self)
        self._morph_preview_timer.setInterval(33)
        self._morph_preview_timer.timeout.connect(lambda: self._on_morph_preview_tick())
        self._morph_last_tick = 0
        self._pose_sdf_timer = QtCore.QTimer(self)
        self._pose_visual_timer = QtCore.QTimer(self)
        self._pending_pose_visual = None
        self._pending_pose_mesh = None
        self._pending_systematic_after_sdf = None
        self._pending_pose_corrective_after_base_fit = False
        self._pending_base_assign_after_fit = False
        self._suppress_pose_sdf_recompute = True
        self._mesh_rotation = (0, 0, 0)
        self._mesh_rot_center = np.zeros(3, np.float32)
        self._viewer = mock.Mock()
        self.draws = []
        self._shape = SimpleNamespace(render=self._capture_draw)
        self._shapes_by_id = {"ellipsoid": self._shape, "superquadric": self._shape}
        self._status = mock.Mock()
        self._api_job_id = None
        self._api_stage = None
        self._api_server = None
        self._api_base_pose = None
        self._api_pending_base_result = None
        self._pose_corrective_worker = None
        self._pose_corrective_cancel_requested = False
        self._pose_corrective_cancel_message = None
        self._pose_corrective_last_error = None
        self._opt_total_steps = 2
        self._last_mesh_result = object()
        self._ellipsoids = None
        self._current_mesh_name = "custom mesh"
        self._btn_fit = QtWidgets.QPushButton(self)
        self._btn_stop = QtWidgets.QPushButton(self)
        self._mesh_sdf_panel = SimpleNamespace(requested_n=8)
        self._slider_margin = QtWidgets.QSlider(self)
        self._slider_margin.setValue(12)
        self._mesh_settings = SimpleNamespace(blowup_fraction=lambda: 0)
        self._settings = {"thickness_max_resolution": 8, "pose_fit_live_preview": True}
        self._device = "cpu"
        self._run_tracker = mock.Mock()
        self._progress_begin = mock.Mock()
        self._progress_set = mock.Mock()
        self._progress_end = mock.Mock()
        self.stop_optimization = mock.Mock()
        self._stop_pose_corrective_fit = mock.Mock()
        self._cancel_pending_pose_sdf = mock.Mock()
        self._ensure_sdf_idle = mock.Mock()
        self._on_compute_all = mock.Mock()
        self._sdf = mock.Mock()
        self._pose_corrective_fit_kwargs = lambda: {"num_steps": 2}
        self._api_fail = mock.Mock()
        self.callback_senders = []
        if library is not None:
            panel.set_bone_local(library.base)
            panel.set_morph_library_count(len(library.keys))

    def __getattr__(self, name):
        method = getattr(MainWindow, name, None)
        if callable(method):
            return MethodType(method, self)
        raise AttributeError(name)

    def _capture_draw(self, _viewer, centers, radii, rotations, exponents):
        self.draws.append(tuple(np.asarray(value).copy()
            for value in (centers, radii, rotations, exponents)))

    @QtCore.Slot(str)
    def deliver_worker_failure(self, message):
        self.callback_senders.append(self.sender())
        MainWindow._on_pose_corrective_worker_failed(self, message)

    @QtCore.Slot(int, int, float, object, object, object, object)
    def deliver_worker_step(self, *args):
        self.callback_senders.append(self.sender())
        MainWindow._on_pose_corrective_step_visual(self, *args)

    @QtCore.Slot()
    def deliver_worker_finished(self):
        self.callback_senders.append(self.sender())
        MainWindow._on_pose_corrective_worker_finished(self)


class CapturedWorker(QtCore.QObject):
    pose_started = QtCore.Signal(int, int, str)
    pose_target_visual = QtCore.Signal(int, str, object, object, object, object, object, object, object)
    pose_sdf_progress = QtCore.Signal(int, float, str)
    pose_fit_progress = QtCore.Signal(int, int, float, object, object, object, object)
    pose_finished = QtCore.Signal(int, str, float)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("parent"))
        self.kwargs = kwargs
        self.result = None
        self.started = False

    def start(self):
        self.started = True


class NativePoseUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.panel = RigModePanel(pose_dir=Path(self.temp.name))
        self.panel.setChecked(True)
        self.rig = make_rig()
        self.panel.set_rigged_mesh(self.rig)
        self.hosts = []
        # Automatic morph-fit saves must never touch the repository.
        autosave = mock.patch("main_window.default_morph_fit_path",
            side_effect=lambda name: Path(self.temp.name) / "auto" / f"{Path(name).stem}_morph_fit.json")
        autosave.start()
        self.addCleanup(autosave.stop)

    def host(self, library=None):
        host = WindowHost(self.panel, library)
        self.hosts.append(host)
        return host

    def tearDown(self):
        for host in self.hosts:
            for timer in (host._morph_preview_timer, host._pose_sdf_timer, host._pose_visual_timer):
                timer.stop()
            host.deleteLater()
        self.panel.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_joint_range_poses_are_the_default_source_and_move_real_mesh(self):
        original = self.rig.poses[1].bone_locals[1].copy()
        self.assertEqual(self.panel.current_source, self.panel.SYSTEMATIC)
        self.panel._spin_sweep_step.setValue(30)
        self.panel._spin_sweep_coverage.setValue(100)
        self.panel._spin_sweep_passes.setValue(1)
        self.assertGreater(len(self.panel.active_poses), 2)
        self.assertEqual(self.panel._systematic_plan.channel_count, 1)
        self.assertEqual(self.panel._slider_pose.maximum(), len(self.panel.active_poses) - 1)
        angles = [joint_pose_angles(self.rig.skeleton, p)[1, 2] for p in self.panel.active_poses]
        self.assertAlmostEqual(min(angles), -60, places=4)
        self.assertAlmostEqual(max(angles), 90, places=4)
        self.assertEqual(self.panel.current_pose_index, 0)
        self.assertIn("base pose", self.panel._lbl_pose_name.text())
        bind_vertices = self.panel.get_deformed_mesh(0)
        self.panel._slider_pose.setValue(1)
        moved = self.panel.get_deformed_mesh()
        self.assertGreater(np.max(np.linalg.norm(moved - bind_vertices, axis=1)), .1)
        np.testing.assert_array_equal(self.rig.poses[1].bone_locals[1], original)
        self.assertIn("1 joints", self.panel._lbl_sweep.text())

    def test_range_change_regenerates_poses_and_discards_only_their_edits(self):
        self.panel.select_source("native")
        self.panel.set_joint_rotation(1, 2, 15)
        self.assertTrue(self.panel.generate_systematic_source())
        self.panel._slider_pose.setValue(1)
        self.panel.set_joint_rotation(1, 2, 50)
        before = self.panel.active_poses
        self.rig.skeleton.bones[1].joint_dof.set_axis(2, True, -10, 20)
        self.panel._spin_sweep_step.setValue(10)
        self.assertIsNot(before, self.panel.active_poses)
        self.assertEqual(self.panel._slider_pose.maximum(), len(self.panel.active_poses) - 1)
        angles = [joint_pose_angles(self.rig.skeleton, p)[1, 2] for p in self.panel.active_poses]
        self.assertGreaterEqual(min(angles), -10 - 1e-4)
        self.assertLessEqual(max(angles), 20 + 1e-4)
        self.assertFalse(any(k[0] == self.panel.SYSTEMATIC for k in self.panel._joint_edits))
        self.panel.select_source("native")
        self.assertAlmostEqual(joint_pose_angles(self.rig.skeleton, self.panel.current_pose)[1, 2], 15, places=4)

    def test_running_fit_keeps_its_poses_until_it_ends(self):
        frozen = self.panel.active_poses
        self.panel.set_auto_pipeline_running(True)
        self.rig.skeleton.bones[1].joint_dof.set_axis(2, True, -10, 20)
        self.panel._update_systematic_summary()
        self.assertIs(self.panel.active_poses, frozen)
        self.panel.set_auto_pipeline_running(False)
        self.assertIsNot(self.panel.active_poses, frozen)

    def test_model_change_invalidates_generated_ranges_manual_edits_and_morph_controls(self):
        self.panel.generate_systematic_source()
        self.panel.set_joint_rotation(1, 2, 40)
        self.panel.set_morph_library_count(1)
        replacement = make_rig()
        replacement.skeleton.bones[1].joint_dof.set_axis(2, False, 0, 0)
        self.panel.set_rigged_mesh(replacement)
        self.assertEqual(self.panel._joint_edits, {})
        self.assertIsNone(self.panel.bone_local)
        self.assertFalse(self.panel._btn_save_morph.isEnabled())
        self.assertFalse(self.panel.generate_systematic_source())
        self.assertEqual(len(self.panel.active_poses), 1)

    def test_training_freezes_pose_authoring_and_releases_only_valid_actions(self):
        host = self.host(make_library(self.rig))
        self.panel.set_joint_rotation(1, 2, 40)
        before = self.panel.current_pose.bone_locals[1].copy()
        self.panel.set_auto_pipeline_running(True)
        for widget in (self.panel._cmb_joint, self.panel._btn_edit_joint,
                self.panel._slider_pose, self.panel._btn_workflow,
                self.panel._btn_fit_base, self.panel._btn_fit_morphs,
                self.panel._spin_morph_steps, self.panel._spin_sweep_step,
                self.panel._btn_save_morph, self.panel._btn_load_morph):
            self.assertFalse(widget.isEnabled(), widget.objectName() or widget.__class__.__name__)
        self.assertFalse(self.panel.generate_systematic_source())
        self.panel.set_joint_rotation(1, 2, -30)
        self.panel.reset_joint_rotation(1)
        self.panel.reset_joint_rotations()
        np.testing.assert_array_equal(self.panel.current_pose.bone_locals[1], before)
        self.panel.set_auto_pipeline_running(False)
        self.assertTrue(self.panel._btn_fit_morphs.isEnabled())
        self.assertTrue(self.panel._btn_save_morph.isEnabled())
        self.assertTrue(self.panel._slider_pose.isEnabled())
        self.panel._bone_local = None
        self.panel.set_morph_library_count(0)
        self.panel.set_auto_pipeline_running(True)
        self.panel.set_auto_pipeline_running(False)
        self.assertFalse(self.panel._btn_save_morph.isEnabled())

    def test_native_fit_is_guarded_while_unity_api_job_is_active(self):
        host = self.host()
        host._api_job_id = "unity in-flight"
        host._on_native_fit_requested(True, True)
        host.stop_optimization.assert_not_called()
        host._on_compute_all.assert_not_called()
        self.assertFalse(self.panel._pose_training_running)
        self.assertIn("Unity API", host._status.showMessage.call_args.args[0])

    def test_workflow_button_requests_bind_sdf_then_fits_base_and_morph_poses(self):
        host = self.host()
        host._start_base_pose_optimization = mock.Mock()
        self.panel.set_joint_rotation(1, 2, 55)
        self.panel.fitRequested.connect(lambda base, morphs: host._on_native_fit_requested(base, morphs))
        self.panel._btn_workflow.click()
        self.assertIs(host._pending_systematic_after_sdf, self.rig)
        self.assertTrue(self.panel._pose_training_running)
        self.assertIsNone(host._last_mesh_result)
        self.assertEqual(self.panel.current_pose_index, 0)
        np.testing.assert_allclose(host._sdf.set_mesh.call_args.args[0], self.rig.vertices)
        host._on_compute_all.assert_called_once()
        host._last_mesh_result = object()
        host._start_systematic_after_sdf()
        self.assertIsNone(host._pending_systematic_after_sdf)
        host._start_base_pose_optimization.assert_called_once()
        self.assertTrue(host._pending_base_assign_after_fit)
        self.assertTrue(host._pending_pose_corrective_after_base_fit)
        self.assertTrue(self.panel._pose_training_running)

    def test_base_button_reuses_a_finished_bind_sdf_and_skips_morph_poses(self):
        host = self.host(make_library(self.rig))
        host._start_base_pose_optimization = mock.Mock()
        host._sdf_worker = None
        host._native_sdf_context = (self.rig, Pose.t_pose(), (0, 0, 0))
        self.panel.fitRequested.connect(lambda base, morphs: host._on_native_fit_requested(base, morphs))
        self.panel._btn_fit_base.click()
        host._on_compute_all.assert_not_called()
        host._start_base_pose_optimization.assert_called_once()
        self.assertIsNone(self.panel.bone_local, "A base refit starts from scratch")
        self.assertIsNone(host._pose_correctives)
        self.assertTrue(host._pending_base_assign_after_fit)
        self.assertFalse(host._pending_pose_corrective_after_base_fit)

    def test_morph_button_needs_a_base_fit_and_otherwise_trains_from_it(self):
        host = self.host()
        host._on_auto_pipeline_clicked = mock.Mock()
        self.panel.fitRequested.connect(lambda base, morphs: host._on_native_fit_requested(base, morphs))
        self.panel._btn_fit_morphs.click()
        host._on_auto_pipeline_clicked.assert_not_called()
        self.assertIn("Fit the base pose first", host._status.showMessage.call_args.args[0])
        self.panel.set_bone_local(make_library(self.rig).base)
        self.panel._btn_fit_morphs.click()
        host._on_auto_pipeline_clicked.assert_called_once()
        host._on_compute_all.assert_not_called()

    def test_model_change_prevents_old_sdf_continuation(self):
        host = self.host()
        host._pending_systematic_after_sdf = self.rig
        host._start_base_pose_optimization = mock.Mock()
        self.panel.set_rigged_mesh(make_rig())
        host._start_systematic_after_sdf()
        host._start_base_pose_optimization.assert_not_called()
        self.assertFalse(self.panel._pose_training_running)

    def test_loading_model_b_retires_model_a_worker_and_clears_all_old_primitives(self):
        library = make_library(self.rig)
        host = self.host(library)
        for name in ("_sdf_worker", "_opt_worker", "_region_sdf_worker",
                "_region_fit_worker", "_batched_worker", "_bonesep_ctl"):
            setattr(host, name, None)
        host._worker_generation = 0
        host._region_fit_active = False
        host._overall_end = mock.Mock()
        host._clear_region_preview_state = mock.Mock()
        host._grp_rig = QtWidgets.QGroupBox(host)
        host._mesh_settings = SimpleNamespace(blowup_fraction=lambda: 0,
            reset=mock.Mock(), set_region_available=mock.Mock())
        host._mesh_sdf_panel.set_blowup_thickness = mock.Mock()
        host._on_sdf_blowup_changed = mock.Mock()
        host._select_loaded_mesh = mock.Mock()
        host._compute_bone_centers = MainWindow._compute_bone_centers
        host._compute_bone_expected_counts = MainWindow._compute_bone_expected_counts
        host._stop_pose_corrective_fit = MethodType(MainWindow._stop_pose_corrective_fit, host)
        host._ellipsoids = object()
        host._fitted_primitive_type = "superquadric"
        host._fitted_shape_exponents = library.base.shape_exponents.copy()
        host._fitted_primitive_rotation = (20, 30, 40)
        host._mesh_rotation = (20, 30, 40)
        host._morph_preview_timer.start()
        old = PoseCorrectiveWorker(rigged_mesh=self.rig, mapper=self.panel.mapper,
            base=library.base, poses=[Pose.t_pose()], grid_n=8, margin=.12,
            fit_kwargs={"num_steps": 1}, device="cpu", parent=host)
        host._pose_corrective_worker = old
        old.finished.connect(host.deliver_worker_finished, QtCore.Qt.QueuedConnection)
        old.finished.emit()
        replacement = make_rig()
        replacement.vertices = replacement.vertices + np.array([0, 1, .3], np.float32)
        replacement.mesh_name = "model B"
        observed_workers = []
        original_set_rig = self.panel.set_rigged_mesh

        def replace_rig(rig):
            observed_workers.append(host._pose_corrective_worker)
            original_set_rig(rig)

        with mock.patch.object(self.panel, "set_rigged_mesh", side_effect=replace_rig):
            host._load_rigged_mesh("model B.fbx", replacement)
        self.assertEqual(observed_workers, [None], "Retire model A work before replacing its skeleton")
        self.assertIs(self.panel.rigged_mesh, replacement)
        self.assertIsNone(self.panel.bone_local)
        self.assertIsNone(host._ellipsoids)
        self.assertIsNone(host._pose_correctives)
        self.assertIsNone(host._morph_driver)
        self.assertIsNone(host._fitted_shape_exponents)
        self.assertEqual(host._fitted_primitive_type, "ellipsoid")
        self.assertEqual(host._fitted_primitive_rotation, (0, 0, 0))
        self.assertEqual(host._mesh_rotation, (0, 0, 0))
        self.assertFalse(host._morph_preview_timer.isActive())
        host._viewer.clear_ellipsoids.assert_called_once()
        host._viewer.set_pose_fit_reference.assert_called_with(None)
        np.testing.assert_array_equal(host._base_verts, replacement.vertices)
        np.testing.assert_array_equal(host._viewer.show_mesh.call_args.args[0], replacement.vertices)
        host._on_compute_all.assert_called_once()
        # The late completion of A cannot reintroduce its fit after B is loaded.
        self.app.processEvents()
        self.assertIs(host.callback_senders[-1], old)
        self.assertIsNone(host._pose_correctives)
        self.assertIsNone(host._ellipsoids)
        self.assertIs(self.panel.rigged_mesh, replacement)

    def test_api_waits_during_native_sdf_to_base_transition_with_no_worker_alive(self):
        host = self.host()
        for name in ("_sdf_worker", "_opt_worker", "_pose_corrective_worker",
                "_region_sdf_worker", "_region_fit_worker", "_batched_worker", "_bonesep_ctl"):
            setattr(host, name, None)
        host._sdf_finalize_active = False
        host._region_fit_active = False
        registry = mock.Mock()
        registry.status_dict.return_value = {"state": "queued"}
        registry.get.return_value = SimpleNamespace(payload={})
        host._api_server = SimpleNamespace(registry=registry)
        self.assertFalse(host._api_has_local_work_in_flight())
        cases = (
            ("_pending_systematic_after_sdf", self.rig),
            ("_pending_pose_corrective_after_base_fit", True),
            ("_pending_base_assign_after_fit", True),
            ("panel_training", True),
        )
        for name, value in cases:
            with self.subTest(phase=name):
                if name == "panel_training":
                    self.panel.set_auto_pipeline_running(True)
                else:
                    setattr(host, name, value)
                self.assertTrue(host._api_has_local_work_in_flight())
                with mock.patch.object(QtCore.QTimer, "singleShot") as retry:
                    host._api_on_fit_requested_impl("queued-api-job")
                retry.assert_called_once()
                self.assertEqual(retry.call_args.args[0], 100)
                registry.update.assert_not_called()
                host._api_fail.assert_not_called()
                self.assertIsNone(host._api_job_id)
                self.assertIn("queued behind local work", host._status.showMessage.call_args.args[0])
                if name == "panel_training":
                    self.panel.set_auto_pipeline_running(False)
                else:
                    setattr(host, name, None if name.endswith("after_sdf") else False)
                self.assertFalse(host._api_has_local_work_in_flight())

    def test_native_training_uses_generated_poses_without_unity_sampled_vertices(self):
        host = self.host(make_library(self.rig))
        self.rig._unity_pose_vertices = [np.ones_like(self.rig.vertices)] * len(self.rig.poses)
        self.panel.generate_systematic_source()
        expected = len(self.panel.active_poses)
        with mock.patch("main_window.PoseCorrectiveWorker", CapturedWorker):
            host._on_pose_corrective_fit_clicked(force_full_window=True)
        worker = host._pose_corrective_worker
        self.assertTrue(worker.started)
        self.assertIsNone(worker.kwargs["target_vertices"])
        self.assertTrue(worker.kwargs["reuse_base_pose"])
        self.assertEqual(len(worker.kwargs["poses"]), expected)
        self.assertIs(worker.kwargs["base"], self.panel.bone_local)
        self.assertTrue(self.panel._pose_training_running)
        self.assertTrue(host._btn_stop.isEnabled())

    def test_unity_sampled_training_remains_a_separate_target_vertex_path(self):
        host = self.host(make_library(self.rig))
        targets = [np.full_like(self.rig.vertices, i) for i in range(len(self.rig.poses))]
        self.rig._unity_pose_vertices = targets
        host._api_job_id = "unity corrective job"
        host._api_stage = "correctives"
        self.panel.select_source("native")   # as _activate_unity_rig_panel does
        with mock.patch("main_window.PoseCorrectiveWorker", CapturedWorker):
            host._on_pose_corrective_fit_clicked(force_full_window=True)
        self.assertIs(host._pose_corrective_worker.kwargs["target_vertices"][0], targets[0])
        self.assertFalse(host._pose_corrective_worker.kwargs["reuse_base_pose"])
        self.assertEqual(self.panel.current_source, self.panel.NATIVE)
        host.stop_optimization.assert_not_called()

    def test_base_fitted_in_an_edited_pose_still_trains_the_neutral_target(self):
        library = make_library(self.rig)
        host = self.host(library)
        host._fitted_primitive_context = (self.rig, library.keys[0].pose, (0, 0, 0))
        self.panel.generate_systematic_source()
        with mock.patch("main_window.PoseCorrectiveWorker", CapturedWorker):
            host._on_pose_corrective_fit_clicked(force_full_window=True)
        self.assertFalse(host._pose_corrective_worker.kwargs["reuse_base_pose"])

    def test_progress_and_success_finish_promote_library_and_restore_pose_preview(self):
        previous = make_library(self.rig)
        host = self.host(previous)
        replacement = copy.deepcopy(previous)
        host._pose_corrective_worker = SimpleNamespace(result=replacement)
        self.panel.set_auto_pipeline_running(True)
        host._on_corrective_pose_started(0, 2, "generated support")
        self.assertIn("1/2", host._status.showMessage.call_args.args[0])
        host._on_corrective_sdf_progress(0, .5, "voxel grid")
        self.assertGreater(host._progress_set.call_args.args[0], 0)
        host._on_corrective_pose_finished(0, "generated support", .01)
        self.assertEqual(self.panel._progress.value(), 1)
        host._on_pose_corrective_worker_finished()
        self.assertIs(host._pose_correctives, replacement)
        saved = Path(self.temp.name) / "auto" / "custom mesh_morph_fit.json"
        self.assertTrue(saved.is_file(), "Finished morph poses are saved automatically")
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["format"], "ellipsdf-pose-correctives")
        self.assertIn(saved.name, self.panel._lbl_morph.text())
        self.assertFalse(self.panel._pose_training_running)
        self.assertFalse(host._btn_stop.isEnabled())
        self.assertTrue(self.panel._btn_save_morph.isEnabled())
        self.assertIsNone(host._pose_corrective_worker)
        self.assertTrue(host.draws, "Current source pose must replace the last training pose on completion")

    def test_cancel_and_failure_preserve_previous_library_and_restore_current_pose(self):
        for canceled in (False, True):
            with self.subTest(canceled=canceled):
                previous = make_library(self.rig)
                host = self.host(previous)
                host._pose_corrective_worker = SimpleNamespace(result=copy.deepcopy(previous))
                self.panel.set_auto_pipeline_running(True)
                if canceled:
                    host._pose_corrective_cancel_requested = True
                    host._pose_corrective_cancel_message = "Stopped."
                else:
                    host._on_pose_corrective_worker_failed("optimizer smoke failure")
                host._on_pose_corrective_worker_finished()
                self.assertIs(host._pose_correctives, previous)
                self.assertFalse((Path(self.temp.name) / "auto").exists())
                self.assertFalse(self.panel._pose_training_running)
                self.assertIsNone(host._pose_corrective_worker)
                self.assertTrue(host.draws, "Stop/failure must restore the user's active pose preview")
                text = host._status.showMessage.call_args.args[0]
                self.assertIn("Stopped" if canceled else "optimizer smoke failure", text)

    def test_manual_joints_and_unrelated_sources_drive_actual_rotations_not_timeline(self):
        library = make_library(self.rig)
        host = self.host(library)
        self.panel._spin_morph_smoothing.setValue(0)
        library.corrected_blend = mock.Mock(side_effect=AssertionError("timeline index used for descriptors"))
        self.panel.poseChanged.connect(lambda index: host._on_rig_pose_changed(index))
        self.panel.set_joint_rotation(1, 2, 60)
        expected = library.corrected_bone_local(library.keys[0])
        np.testing.assert_allclose(host.draws[-1][1], expected.local_radii, atol=1e-6)
        np.testing.assert_allclose(host.draws[-1][3], expected.shape_exponents, atol=1e-6)
        self.panel._library_clips = [SimpleNamespace(name="other independent clip", poses=[library.keys[0].pose])]
        self.assertTrue(self.panel.select_source("other independent clip"))
        self.assertEqual(self.panel.current_pose_frame, 0)
        np.testing.assert_allclose(host.draws[-1][1], expected.local_radii, atol=1e-6)
        library.corrected_blend.assert_not_called()

    def test_morph_toggle_restores_base_and_settings_rebuild_pose_driver(self):
        library = make_library(self.rig)
        host = self.host(library)
        self.panel._spin_morph_smoothing.setValue(0)
        self.panel.set_joint_rotation(1, 2, 60)
        host._render_pose_morph(self.panel.current_pose)
        driver = host._morph_driver
        self.panel.morphPreviewChanged.connect(lambda: host._on_morph_preview_changed())
        self.panel._chk_morph_preview.setChecked(False)
        np.testing.assert_allclose(host.draws[-1][1], library.base.local_radii)
        np.testing.assert_allclose(host.draws[-1][3], library.base.shape_exponents)
        self.assertFalse(host._morph_preview_timer.isActive())
        self.panel._chk_morph_preview.setChecked(True)
        self.assertIsNot(host._morph_driver, driver)
        np.testing.assert_allclose(host.draws[-1][1], library.corrected_bone_local(library.keys[0]).local_radii)
        driver = host._morph_driver
        self.panel._spin_morph_context.setValue(0)
        self.assertIsNot(host._morph_driver, driver)
        self.assertEqual(host._morph_driver.settings.context_depth, 0)

    def test_smoothing_timer_continues_without_pose_changes_and_stops_when_settled(self):
        library = make_library(self.rig)
        host = self.host(library)
        host._render_pose_morph(Pose.t_pose())
        self.panel.set_joint_rotation(1, 2, 60)
        host._render_pose_morph(self.panel.current_pose)
        np.testing.assert_allclose(host.draws[-1][1], library.base.local_radii)
        self.assertTrue(host._morph_preview_timer.isActive())
        # Drive production ticks at deterministic dt; no sleep or heavy redraw.
        import time
        for _ in range(30):
            host._morph_last_tick = time.perf_counter() - .1
            host._on_morph_preview_tick()
            if not host._morph_preview_timer.isActive():
                break
        self.assertFalse(host._morph_preview_timer.isActive())
        np.testing.assert_allclose(host.draws[-1][1], library.corrected_bone_local(library.keys[0]).local_radii, atol=1e-6)
        self.panel.set_auto_pipeline_running(True)
        host._morph_preview_timer.start()
        host._on_morph_preview_tick()
        self.assertFalse(host._morph_preview_timer.isActive())

    def test_morph_timer_stops_when_library_base_no_longer_matches_visible_population(self):
        library = make_library(self.rig)
        host = self.host(library)
        self.panel.set_bone_local(copy.deepcopy(library.base))
        host._morph_preview_timer.start()
        host._on_morph_preview_tick()
        self.assertFalse(host._morph_preview_timer.isActive())
        self.assertEqual(host.draws, [])

    def test_retired_real_worker_queued_signals_cannot_change_new_training_state(self):
        library = make_library(self.rig)
        host = self.host(library)

        def worker():
            return PoseCorrectiveWorker(rigged_mesh=self.rig,
                mapper=self.panel.mapper, base=library.base, poses=[Pose.t_pose()],
                grid_n=8, margin=.12, fit_kwargs={"num_steps": 1}, device="cpu", parent=host)

        old = worker()
        old.failed.connect(host.deliver_worker_failure, QtCore.Qt.QueuedConnection)
        old.pose_fit_progress.connect(host.deliver_worker_step, QtCore.Qt.QueuedConnection)
        old.finished.connect(host.deliver_worker_finished, QtCore.Qt.QueuedConnection)
        host._pose_corrective_worker = old
        old.failed.emit("retired worker error")
        old.pose_fit_progress.emit(0, 1, .3, library.base.local_centers,
            library.base.local_radii, library.base.local_rotations, library.base.shape_exponents)
        old.finished.emit()
        # Signals are already queued; retire before delivering them, as when a
        # quick Stop is followed by a new model or another training run.
        MainWindow._stop_pose_corrective_fit(host)
        current = worker()
        host._pose_corrective_worker = current
        self.panel.set_auto_pipeline_running(True)
        host._btn_stop.setEnabled(True)
        host._pose_visual_timer.start(45)
        marker = ("current training frame",)
        host._pending_pose_visual = marker
        host._status.reset_mock()
        self.app.processEvents()
        self.assertEqual(len(host.callback_senders), 3)
        self.assertTrue(all(sender is old for sender in host.callback_senders),
            "Must exercise the real Qt sender, not a direct Python callback")
        self.assertIs(host._pose_corrective_worker, current)
        self.assertIs(host._pose_correctives, library)
        self.assertIs(host._pending_pose_visual, marker)
        self.assertIsNone(host._pose_corrective_last_error)
        self.assertTrue(self.panel._pose_training_running)
        self.assertTrue(host._btn_stop.isEnabled())
        self.assertTrue(host._pose_visual_timer.isActive())
        self.assertEqual(host.draws, [])
        host._status.showMessage.assert_not_called()

    def test_global_mesh_rotation_applies_to_morph_centers_and_orientations(self):
        library = make_library(self.rig)
        host = self.host(library)
        self.panel._spin_morph_smoothing.setValue(0)
        self.panel.set_joint_rotation(1, 2, 60)
        host._mesh_rotation = (20, -35, 70)
        host._mesh_rot_center = np.array([.15, -.2, .1], np.float32)
        corrected = library.corrected_bone_local(library.keys[0])
        centers, radii, rotations = self.panel.mapper.local_to_world_np(corrected, self.panel.current_pose)
        host._render_pose_morph(self.panel.current_pose)
        shown = host.draws[-1]
        np.testing.assert_allclose(shown[0], host._apply_rotation(centers), atol=1e-6)
        np.testing.assert_allclose(shown[1], radii, atol=1e-6)
        global_frame = rotation_matrix(*host._mesh_rotation)
        for original, actual in zip(rotations, shown[2]):
            expected_frame = global_frame @ mat4_compose(np.zeros(3), original, np.ones(3))[:3, :3]
            actual_frame = mat4_compose(np.zeros(3), actual, np.ones(3))[:3, :3]
            np.testing.assert_allclose(actual_frame, expected_frame, atol=1e-6)

    def test_save_load_roundtrip_and_rejected_load_replace_visible_state_atomically(self):
        library = make_library(self.rig)
        host = self.host(library)
        path = Path(self.temp.name) / "morph.json"
        with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(path), "JSON")):
            host._on_save_morph_fit()
        self.assertTrue(path.is_file())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], "ellipsdf-pose-correctives")
        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(path), "JSON")):
            host._on_load_morph_fit()
        loaded = host._pose_correctives
        self.assertIsNot(loaded, library)
        self.assertIs(self.panel.bone_local, loaded.base)
        self.assertIs(host._morph_driver.library, loaded)
        np.testing.assert_allclose(loaded.base.local_centers, library.base.local_centers)
        self.assertTrue(host.draws)
        path.write_text('{"format": "broken"}', encoding="utf-8")
        before = host.draws[-1]
        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(path), "JSON")):
            host._on_load_morph_fit()
        self.assertIs(host._pose_correctives, loaded)
        self.assertIs(self.panel.bone_local, loaded.base)
        self.assertIs(host.draws[-1], before)
        self.assertIn("could not be loaded", host._status.showMessage.call_args.args[0])

    def test_load_rejects_invalid_runtime_before_replacing_any_visible_data(self):
        library = make_library(self.rig)
        host = self.host(library)
        path = Path(self.temp.name) / "valid.json"
        save_native_morph_fit(library, self.rig, path)
        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(path), "JSON")), \
                mock.patch("main_window.NativeMorphDriver", side_effect=ValueError("runtime rejection")):
            host._on_load_morph_fit()
        self.assertIs(host._pose_correctives, library)
        self.assertIs(self.panel.bone_local, library.base)
        self.assertEqual(host.draws, [])

    def test_failed_save_reports_error_and_keeps_last_successful_file(self):
        library = make_library(self.rig)
        host = self.host(library)
        path = Path(self.temp.name) / "saved.json"
        save_native_morph_fit(library, self.rig, path)
        previous = path.read_bytes()
        library.keys[0].delta_centers += .01
        self.panel.saveMorphRequested.connect(lambda: host._on_save_morph_fit())
        with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(path), "JSON")), \
                mock.patch("native_morph_io.os.replace", side_effect=OSError("disk replacement failed")):
            self.panel._btn_save_morph.click()
        self.assertEqual(path.read_bytes(), previous)
        self.assertIn("disk replacement failed", host._status.showMessage.call_args.args[0])
        self.assertEqual(sorted(p.name for p in Path(self.temp.name).iterdir()), ["saved.json"])

class NativePoseWorkerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_base_reuse_preserves_timeline_and_only_skips_matching_unsampled_neutral_pose(self):
        rig = make_rig()
        base = make_library(rig).base
        explicit_bind = Pose("explicit bind", {
            i: bone.local_bind_transform.copy() for i, bone in enumerate(rig.skeleton.bones)})
        cases = (
            ("ordinary", False, Pose.t_pose(), None, [0, 1]),
            ("native", True, Pose.t_pose(), None, [1]),
            ("explicit bind", True, explicit_bind, None, [1]),
            ("edited first", True, rig.poses[1], None, [0, 1]),
            ("sampled target", True, Pose.t_pose(),
                [rig.vertices + .01, rig.vertices + .02], [0, 1]),
        )
        for name, reuse, first, sampled, expected_targets in cases:
            with self.subTest(name=name):
                worker = PoseCorrectiveWorker(rigged_mesh=rig,
                    mapper=BoneEllipsoidMapper(rig.skeleton), base=base,
                    poses=[first, rig.poses[1]], grid_n=8, margin=.12,
                    reuse_base_pose=reuse, target_vertices=sampled,
                    fit_kwargs={"num_steps": 1, "report_every": 1, "sample_budget": 64,
                        "maintenance_every": 0, "superfit": False, "local_fit": False,
                        "lr_init": .001, "lr_final": .001, "primitive_shape": "superquadric"},
                    device=best_device())
                failures, targets, fits, completed = [], [], [], []
                worker.failed.connect(failures.append)
                worker.pose_target_visual.connect(lambda index, *_: targets.append(index))
                worker.pose_fit_progress.connect(lambda index, *_: fits.append(index))
                worker.pose_finished.connect(lambda index, *_: completed.append(index))
                worker.run()
                self.assertEqual(failures, [])
                self.assertIsNotNone(worker.result)
                self.assertEqual(targets, expected_targets)
                self.assertEqual(sorted(set(fits)), expected_targets)
                self.assertEqual(completed, [0, 1])
                self.assertEqual(len(worker.result.keys), 2)
                self.assertIs(worker.result.keys[0].pose, first)
                if expected_targets == [1]:
                    self.assertEqual(worker.result.keys[0].loss, 0)
                    np.testing.assert_array_equal(worker.result.keys[0].delta_centers, 0)
                    np.testing.assert_array_equal(worker.result.keys[0].delta_log_radii, 0)

    def test_t_pose_generated_targets_fit_ellipsoid_and_superquadric_with_fixed_ids(self):
        from systematic_pose_fitter import generate_systematic_poses, SystematicPoseSettings
        rig = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")
        # Restrict to two actual anatomical joints for a short model-independent sweep.
        for bone in rig.skeleton.bones:
            for axis in range(3):
                bone.joint_dof.set_axis(axis, False, 0, 0)
        elbow = next(b.index for b in rig.skeleton.bones if b.name.endswith(":LeftForeArm"))
        knee = next(b.index for b in rig.skeleton.bones if b.name.endswith(":LeftLeg"))
        rig.skeleton.bones[elbow].joint_dof.set_axis(2, True, 0, 60)
        rig.skeleton.bones[knee].joint_dof.set_axis(0, True, -60, 0)
        plan = generate_systematic_poses(rig.skeleton, SystematicPoseSettings(angle_step=60, coverage=1, passes=1))
        self.assertEqual(len(plan.poses), 2)
        for primitive_type in ("ellipsoid", "superquadric"):
            with self.subTest(primitive_type=primitive_type):
                base = BoneLocalEllipsoids(
                    local_centers=np.array([[0, .025, 0], [0, -.04, 0]], np.float32),
                    local_radii=np.full((2, 3), .035, np.float32),
                    local_rotations=np.tile(np.array([0, 0, 0, 1], np.float32), (2, 1)),
                    bone_assignments=np.array([elbow, knee], np.int32),
                    primitive_type=primitive_type, shape_exponents=np.full((2, 2), .8, np.float32))
                worker = PoseCorrectiveWorker(rigged_mesh=rig,
                    mapper=BoneEllipsoidMapper(rig.skeleton), base=base,
                    poses=plan.poses, grid_n=8, margin=.12,
                    fit_kwargs={"num_steps": 1, "report_every": 1, "sample_budget": 64,
                        "maintenance_every": 0, "superfit": False, "local_fit": False,
                        "lr_init": .001, "lr_final": .001, "primitive_shape": primitive_type},
                    device=best_device())
                failures, meshes, fits = [], [], []
                worker.failed.connect(failures.append)
                worker.pose_target_visual.connect(lambda _i, _name, vertices, *_: meshes.append(vertices.copy()))
                worker.pose_fit_progress.connect(lambda *args: fits.append(args))
                worker.run()
                self.assertEqual(failures, [])
                self.assertIsNotNone(worker.result)
                self.assertEqual(len(worker.result.keys), 2)
                self.assertTrue(fits)
                self.assertGreater(np.max(np.linalg.norm(meshes[1] - meshes[0], axis=1)), .02)
                result = worker.result
                np.testing.assert_array_equal(result.base.bone_assignments, base.bone_assignments)
                self.assertEqual(result.base.primitive_type, primitive_type)
                self.assertTrue(all(key.pose is not None for key in result.keys))
                driver = NativeMorphDriver(rig.skeleton, result)
                for pose in plan.poses:
                    state = driver.evaluate(pose)
                    self.assertEqual(state.num_ellipsoids, 2)
                    self.assertTrue(np.isfinite(state.local_radii).all())
                    self.assertTrue(np.isfinite(state.shape_exponents).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
