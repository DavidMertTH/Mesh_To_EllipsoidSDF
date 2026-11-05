"""Actual native SQ pipeline: fitted coordinates and each visible posed mesh.

Uses the complete MainWindow orchestration, real workers, the T-Pose FBX and
the user's persisted SQ/pose-fit settings. Only numerical budgets are reduced.
Workspace/settings writes are redirected or suppressed without MagicMocks on
the QObject class, which PySide cannot safely introspect as dynamic slots.
"""
from __future__ import annotations

import copy
from functools import wraps
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtCore, QtWidgets

import app_settings
from main_window import MainWindow
from optimization import OptimizationWorker
from rig_loader import load_rigged_mesh
from skeleton import Pose, mat4_compose
from skinning import _deform_numpy


def _rotations_as_matrices(quaternions):
    return np.array([mat4_compose(np.zeros(3), q, np.ones(3))[:3, :3] for q in quaternions])


class NativePoseAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait_for(self, predicate, window, timeout=60):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(predicate(), window._status.currentMessage())

    def test_real_sq_systematic_pipeline_keeps_base_fit_and_renders_every_posed_target(self):
        self._run_real_pipeline(initial_posed_fit=False)

    def test_reused_sq_fit_in_manual_pose_keeps_its_fitted_source_coordinates(self):
        self._run_real_pipeline(initial_posed_fit=True)

    def test_sq_base_maintenance_at_six_hundred_steps_preserves_alignment(self):
        self._run_real_pipeline(initial_posed_fit=False, maintenance=True)

    def test_native_pose_refit_can_lock_rotations_and_radii_while_training_centers(self):
        self._run_real_pipeline(initial_posed_fit=False, transform_flags=(True, False, False))

    def test_default_full_pose_sweep_starts_with_a_moving_mesh_after_base_fit(self):
        self._run_real_pipeline(initial_posed_fit=False, full_sweep=True)

    def _run_real_pipeline(self, *, initial_posed_fit, maintenance=False,
            transform_flags=(False, True, True), full_sweep=False):
        advanced = app_settings.defaults()
        if app_settings._FILE.exists():
            advanced.update(json.loads(app_settings._FILE.read_text(encoding="utf-8")))
        panel_state = json.loads(app_settings._PANEL_FILE.read_text(encoding="utf-8")) \
            if app_settings._PANEL_FILE.exists() else {}
        panel_state = copy.deepcopy(panel_state)
        panel_state["shape"] = "superquadric"
        advanced.update(sample_budget=128, thickness_max_resolution=16,
            region_res=16, region_steps=2, local_steps=2,
            pose_fit_position=transform_flags[0], pose_fit_rotation=transform_flags[1],
            pose_fit_scale=transform_flags[2],
            pose_fit_live_preview=True)
        panel_state.setdefault("shared", {}).update(
            sdf_grid_n=16, num_ellipsoids=8, max_ellipsoids=8, max_steps=600 if maintenance else 100,
            symmetry=True, blowup_fraction=-.04,
            systematic_step=90, systematic_coverage=100, systematic_passes=1,
            morph_smoothing=0)
        panel_state.setdefault("shapes", {}).setdefault("superquadric", {}).update(
            local_fit=True, superfit=True)
        if full_sweep:
            panel_state["shared"].update(systematic_step=30,
                systematic_coverage=90, systematic_passes=4, systematic_roots=False,
                blowup_fraction=0)
            panel_state["shapes"]["superquadric"].update(local_fit=False, superfit=False)
        if maintenance:
            panel_state["shared"].update(sdf_grid_n=32, max_ellipsoids=16)
            panel_state["shapes"]["superquadric"].update(local_fit_every=100, superfit_every=100)
            # Preserve the physical feature/region scale when reducing the
            # user's 128-voxel grid to 32. Eight coarse voxels would reject
            # every limb and silently leave the local-fit path unexercised.
            advanced.update(underrep_min_thickness_vox=2.0,
                thickness_max_resolution=32, region_radius_vox=3.0, min_split_radius_vox=.75)

        optimizer_options, phases, maintenance_events = [], [], []
        original_optimizer_init = OptimizationWorker.__init__

        @wraps(original_optimizer_init)
        def capture_optimizer_init(worker, *args, **kwargs):
            optimizer_options.append({key: kwargs.get(key) for key in (
                "optimize_centers", "optimize_rotations", "optimize_radii",
                "parameter_linear_np", "num_steps", "superfit", "local_fit",
                "superfit_every", "local_fit_every", "local_fit_start_frac", "local_fit_end_frac")})
            original_optimizer_init(worker, *args, **kwargs)
            worker.phase_changed.connect(lambda phase: phases.append(str(phase)), QtCore.Qt.DirectConnection)
            worker.maintenance_done.connect(lambda *event: maintenance_events.append(event), QtCore.Qt.DirectConnection)

        with tempfile.TemporaryDirectory() as tmp:
            # Plain lambdas preserve the QObject class metadata. The app reads
            # the real shape settings; autosave cannot touch the user's files.
            with mock.patch.object(app_settings, "load", return_value=advanced), \
                    mock.patch.object(app_settings, "load_panel", return_value=panel_state), \
                    mock.patch.object(app_settings, "save_panel"), \
                    mock.patch.object(OptimizationWorker, "__init__", new=capture_optimizer_init), \
                    mock.patch.object(MainWindow, "_restore_workspace_layout", new=lambda self: False), \
                    mock.patch.object(MainWindow, "_save_workspace_layout", new=lambda self: None), \
                    mock.patch("main_window.default_morph_fit_path",
                        side_effect=lambda name: Path(tmp) / "morph_fit.json"):
                window = MainWindow(mesh_dir=Path(tmp))
                if full_sweep:
                    window._spin_max_steps.setMinimum(1)
                    window._spin_max_steps.setValue(1)
                # Morph poses have their own budget; keep it at the base budget.
                window._rig_panel._spin_morph_steps.setMinimum(1)
                window._rig_panel._spin_morph_steps.setValue(window._spin_max_steps.value())
                base_snapshots, target_snapshots, audit_errors = [], [], []
                fitted_center_error = []
                fitted_pose = Pose.t_pose()
                original_ensure = window._ensure_pose_corrective_base
                original_target = window._on_corrective_target_visual
                original_fit_visual = window._on_pose_corrective_step_visual

                def capture_base():
                    before = window._ellipsoids
                    expected = None if before is None else (
                        before.centers.copy(), before.radii.copy(), before.rotations.copy())
                    ok = original_ensure()
                    if ok and expected is not None:
                        actual = window._rig_panel.mapper.local_to_world_np(window._rig_panel.bone_local, fitted_pose)
                        fitted_center_error.append(float(np.max(np.linalg.norm(actual[0] - expected[0], axis=1))))
                        base_snapshots.append((expected, actual))
                    return ok

                def capture_target(*args):
                    original_target(*args)
                    index, _name, vertices, _faces, centers, radii, rotations, _eps, pose = args
                    skeleton = window._rig_panel.rigged_mesh.skeleton
                    independent = _deform_numpy(rig.vertices, rig.skin_joints,
                        rig.skin_weights, skeleton.compute_skin_matrices(pose))
                    target_snapshots.append((int(index), vertices.copy(), independent,
                        window._viewer._mesh_verts.copy(), centers.copy(), radii.copy(), rotations.copy()))
                    error = float(np.max(np.linalg.norm(vertices - independent, axis=1)))
                    visible_error = float(np.max(np.linalg.norm(window._viewer._mesh_verts - independent, axis=1)))
                    if error > 1e-5 or visible_error > 1e-5:
                        audit_errors.append((index, "target mesh", error, visible_error))
                    panel = window._rig_panel
                    if (panel.current_pose_index != index or panel.current_pose_frame != index
                            or _name not in panel._lbl_pose_name.text()):
                        audit_errors.append((index, "visible pose controls", panel.current_pose_index,
                            panel.current_pose_frame, panel._lbl_pose_name.text()))

                def capture_fit_visual(*args):
                    original_fit_visual(*args)
                    if target_snapshots:
                        target = target_snapshots[-1][2]
                        error = float(np.max(np.linalg.norm(window._viewer._mesh_verts - target, axis=1)))
                        if error > 1e-5:
                            audit_errors.append((args[0], "mesh during fit", error))

                window._ensure_pose_corrective_base = capture_base
                window._on_corrective_target_visual = capture_target
                window._on_pose_corrective_step_visual = capture_fit_visual
                try:
                    rig = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")
                    # A short but real sweep bends two actual anatomical joints.
                    if not full_sweep:
                        for bone in rig.skeleton.bones:
                            for axis in range(3):
                                bone.joint_dof.set_axis(axis, False, 0, 0)
                    elbow = next(b.index for b in rig.skeleton.bones if b.name.endswith(":LeftForeArm"))
                    knee = next(b.index for b in rig.skeleton.bones if b.name.endswith(":LeftLeg"))
                    if not full_sweep:
                        rig.skeleton.bones[elbow].joint_dof.set_axis(2, True, 0, 90)
                        rig.skeleton.bones[knee].joint_dof.set_axis(0, True, -90, 0)
                    window._load_rigged_mesh(str(ROOT / "meshes" / "T-Pose.fbx"), rig)
                    self.wait_for(lambda: window._sdf_worker is None and not window._sdf_finalize_active,
                        window)
                    self.assertIsNotNone(window._last_mesh_result)
                    self.assertEqual(window._shape.id, "superquadric")
                    self.assertAlmostEqual(window._mesh_settings.blowup_fraction(), 0 if full_sweep else -.04)
                    self.assertEqual(window._settings["pose_fit_position"], transform_flags[0])
                    self.assertEqual(window._shape.fit_kwargs()["local_fit"], not full_sweep)
                    self.assertEqual(window._shape.fit_kwargs()["superfit"], not full_sweep)
                    if initial_posed_fit:
                        root = next(b.index for b in rig.skeleton.bones if b.parent_index < 0)
                        rig.skeleton.bones[root].joint_dof.set_axis(2, True, -90, 90)
                        window._rig_panel.set_joint_rotation(root, 2, 45)
                        window._rig_panel.set_joint_rotation(elbow, 2, 90)
                        window._rig_panel.set_joint_rotation(knee, 0, -90)
                        fitted_pose = copy.deepcopy(window._rig_panel.current_pose)
                        self.wait_for(lambda: window._pending_pose_mesh is None
                            and window._sdf_worker is None and not window._sdf_finalize_active, window)
                        expected_mesh = _deform_numpy(rig.vertices, rig.skin_joints,
                            rig.skin_weights, rig.skeleton.compute_skin_matrices(fitted_pose))
                        np.testing.assert_allclose(window._viewer._mesh_verts, expected_mesh, atol=1e-5)
                        window._btn_fit.click()
                        self.wait_for(lambda: window._opt_worker is None, window)
                        self.assertIsNotNone(window._ellipsoids, window._status.currentMessage())
                    # An existing (posed) base fit is reused by step 2;
                    # otherwise the one-click workflow fits the base first.
                    (window._rig_panel._btn_fit_morphs if initial_posed_fit
                     else window._rig_panel._btn_workflow).click()
                    self.wait_for(lambda: not window._rig_panel._pose_training_running
                        and window._pose_corrective_worker is None and window._opt_worker is None
                        and window._pending_systematic_after_sdf is None
                        and not window._pending_base_assign_after_fit
                        and not window._pending_pose_corrective_after_base_fit, window)
                    self.assertTrue((Path(tmp) / "morph_fit.json").is_file())
                    library = window._pose_correctives
                    self.assertIsNotNone(library, window._status.currentMessage())
                    self.assertEqual(library.base.primitive_type, "superquadric")
                    support_count = len(window._rig_panel.active_poses) if full_sweep else 2
                    self.assertGreater(support_count, 50 if full_sweep else 1)
                    first_target = 0 if initial_posed_fit else 1
                    self.assertEqual(len(library.keys), support_count)
                    self.assertIs(window._morph_curves._library, library)
                    self.assertEqual(len(window._morph_curves._series.samples), support_count + 1)
                    refit_options = [options for options in optimizer_options
                        if options["parameter_linear_np"] is not None]
                    self.assertEqual(len(refit_options), support_count - first_target)
                    if not initial_posed_fit:
                        neutral = library.keys[0]
                        np.testing.assert_array_equal(neutral.delta_centers, 0)
                        np.testing.assert_array_equal(neutral.delta_log_radii, 0)
                        np.testing.assert_array_equal(neutral.delta_log_shape_exponents, 0)
                        np.testing.assert_allclose(_rotations_as_matrices(neutral.delta_rotations),
                            np.broadcast_to(np.eye(3), (library.base.num_ellipsoids, 3, 3)), atol=1e-6)
                        self.assertEqual(neutral.loss, 0)
                    for options in refit_options:
                        self.assertEqual((options["optimize_centers"], options["optimize_rotations"],
                            options["optimize_radii"]), transform_flags)
                        self.assertFalse(options["superfit"])
                        self.assertFalse(options["local_fit"])
                    for key in library.keys:
                        if not transform_flags[0]:
                            np.testing.assert_array_equal(key.delta_centers, 0)
                        if not transform_flags[1]:
                            np.testing.assert_array_equal(key.delta_rotations,
                                np.tile([0, 0, 0, 1], (library.base.num_ellipsoids, 1)))
                        if not transform_flags[2]:
                            np.testing.assert_array_equal(key.delta_log_radii, 0)
                    if maintenance:
                        self.assertEqual(optimizer_options[0]["num_steps"], 600)
                        self.assertIn("local", phases,
                            f"600-step base fit must execute local maintenance: {optimizer_options[0]}, {maintenance_events}")
                    self.assertEqual(audit_errors, [])
                    self.assertEqual([s[0] for s in target_snapshots], list(range(first_target, support_count)))
                    self.assertEqual(window._rig_panel.current_pose_index, support_count - 1)
                    if full_sweep:
                        self.assertGreater(np.max(np.linalg.norm(target_snapshots[0][1] - rig.vertices, axis=1)), .05)
                        for snapshot in target_snapshots:
                            self.assertGreater(np.max(np.linalg.norm(snapshot[1] - rig.vertices, axis=1)), 1e-4)
                    self.assertEqual(len(base_snapshots), 1)
                    print("ALIGNMENT_SOURCE", {"initial_posed_fit": initial_posed_fit,
                        "base_center_roundtrip_error": fitted_center_error}, flush=True)
                    for expected, actual in base_snapshots:
                        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
                        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5)
                        np.testing.assert_allclose(_rotations_as_matrices(actual[2]),
                            _rotations_as_matrices(expected[2]), rtol=1e-5, atol=1e-5)
                    # The last sweep pose may be a small twist; it only has to
                    # be visibly deformed for the alignment check to mean anything.
                    posed_delta = float(np.max(np.linalg.norm(target_snapshots[-1][1] - rig.vertices, axis=1)))
                    self.assertGreater(posed_delta, .1)
                    window._rig_panel._slider_pose.setValue(support_count - 1)
                    shown = window._viewer._mesh_verts.copy()
                    np.testing.assert_allclose(shown, target_snapshots[-1][2], atol=1e-5)
                    self.assertGreater(np.max(np.linalg.norm(shown - rig.vertices, axis=1)), .1)
                    print("ALIGNMENT_AUDIT", {
                        "base_center_roundtrip_error": fitted_center_error,
                        "target_mesh_pose_displacement": posed_delta,
                        "primitive_count": library.base.num_ellipsoids,
                        "shape": library.base.primitive_type,
                        "targets": len(target_snapshots),
                        "errors": audit_errors,
                        "maintenance": maintenance,
                        "local_phases": phases.count("local"),
                    }, flush=True)
                finally:
                    window._panel_save_timer.stop()
                    window.close()
                    window.deleteLater()
                    self.app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
