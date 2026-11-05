"""Captured curve values, selection, and real workspace integration."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtWidgets

import app_settings
from main_window import MainWindow
from morph_curve_data import build_morph_curve_series, relative_rotation_degrees
from morph_curve_panel import MorphCurvePanel, PARAMETERS
from native_morph_io import save_native_morph_fit
from skeleton import Pose
from tools.native_pose_ui_test import make_rig, make_library


class MorphCurvesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.rig = make_rig()
        self.library = make_library(self.rig)
        self.panel = MorphCurvePanel()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def series(self):
        return build_morph_curve_series(self.rig.skeleton, self.library, 1, 0)

    def test_values_are_captured_fit_parameters_relative_to_base(self):
        series = self.series()
        np.testing.assert_allclose(series.angles("z"), [0, 60], atol=1e-5)
        self.assertEqual(series.resolved_axis(), "z")
        np.testing.assert_allclose(series.values("scale"), [[1, 1, 1], [1.4, .8, 1.1]], rtol=1e-6)
        np.testing.assert_allclose(series.values("position_delta"), [[0, 0, 0], [.025, .01, 0]], atol=1e-7)
        np.testing.assert_allclose(series.values("radii")[1], self.library.base.local_radii[0] * [1.4, .8, 1.1], rtol=1e-6)
        np.testing.assert_allclose(series.values("exponents"), [[.8, 1.1], [.88, .99]], rtol=1e-6)
        np.testing.assert_allclose(series.values("rotation")[1], [0, 0, np.degrees(2 * np.arcsin(.1))], atol=1e-5)

    def test_signed_rotation_vectors_match_unity_and_ignore_quaternion_sign(self):
        identity = [0, 0, 0, 1]
        q = np.array([0, 0, np.sin(np.radians(-20) / 2), np.cos(np.radians(-20) / 2)])
        np.testing.assert_allclose(relative_rotation_degrees(identity, q), [0, 0, -20], atol=1e-8)
        np.testing.assert_allclose(relative_rotation_degrees(identity, -q), [0, 0, -20], atol=1e-8)
        for q in ([0, 0, 1, 0], [0, 0, -1, 0]):
            np.testing.assert_allclose(relative_rotation_degrees(identity, q), [0, 0, 180], atol=1e-8)

    def test_joint_angle_uses_captured_base_pose(self):
        self.library.base_pose = self.rig.poses[1]  # -20 degrees
        np.testing.assert_allclose(self.series().angles("z"), [0, 80], atol=1e-5)

    def test_missing_and_invalid_samples_are_skipped_without_inventing_values(self):
        no_pose = copy.deepcopy(self.library.keys[0])
        no_pose.pose = None
        missing = copy.deepcopy(self.library.keys[0])
        missing.delta_centers = np.empty((0, 3))
        invalid = copy.deepcopy(self.library.keys[0])
        invalid.delta_log_radii[0, 0] = np.nan
        self.library.keys.extend([no_pose, missing, invalid])
        series = self.series()
        self.assertEqual(len(series.samples), 2)
        self.assertEqual((series.missing_pose, series.missing_parameters, series.invalid_values), (1, 1, 1))

    def test_equal_angles_keep_distinct_pose_values(self):
        other = copy.deepcopy(self.library.keys[0])
        other.name = "same joint angle with another compound pose"
        other.delta_log_radii[0, 0] = np.log(2)
        self.library.keys.append(other)
        series = self.series()
        np.testing.assert_allclose(series.angles("z"), [0, 60, 60], atol=1e-5)
        np.testing.assert_allclose(series.values("scale")[:, 0], [1, 1.4, 2], rtol=1e-6)

    def test_exponents_are_bounded_and_ellipsoids_remain_one(self):
        self.library.keys[0].delta_log_shape_exponents[0] = [1000, -1000]
        np.testing.assert_allclose(self.series().values("exponents")[1], [2, .1])
        self.library.base.primitive_type = "ellipsoid"
        np.testing.assert_array_equal(self.series().values("exponents"), np.ones((2, 2)))

    def test_plot_and_table_follow_joint_primitive_axis_and_parameter(self):
        self.panel.set_source(self.rig, self.library)
        self.assertEqual(self.panel._joint.currentData(), 1)
        self.assertEqual(self.panel._primitive.currentData(), 0)
        self.assertEqual(self.panel._table.rowCount(), 2)
        self.assertEqual(len(self.panel._curves), 2)
        np.testing.assert_allclose(self.panel._curves[0].getData()[0], [0, 60], atol=1e-5)
        self.panel._parameter.setCurrentIndex(self.panel._parameter.findData("scale"))
        self.assertEqual(len(self.panel._curves), 3)
        np.testing.assert_allclose(self.panel._curves[0].getData()[1], [1, 1.4], rtol=1e-6)
        self.panel._axis.setCurrentIndex(self.panel._axis.findData("x"))
        np.testing.assert_array_equal(self.panel._curves[0].getData()[0], [0, 0])
        self.panel._connect.setChecked(True)
        self.assertIsNotNone(self.panel._curves[0].opts["pen"])

    def test_primitive_filter_can_compare_other_joints(self):
        self.panel.set_source(self.rig, self.library)
        self.panel._joint.setCurrentIndex(self.panel._joint.findData(0))
        self.assertEqual(self.panel._primitive.count(), 0)
        self.assertEqual(self.panel._table.rowCount(), 0)
        self.panel._all_primitives.setChecked(True)
        self.assertEqual(self.panel._primitive.currentData(), 0)
        self.assertEqual(self.panel._table.rowCount(), 2)
        np.testing.assert_array_equal(self.panel._curves[0].getData()[0], [0, 0])

    def test_visible_plot_paints_after_changing_parameters_and_connecting_samples(self):
        self.panel.set_source(self.rig, self.library)
        self.panel.resize(660, 730)
        self.panel.show()
        self.app.processEvents()
        for _label, parameter, _units, _channels in PARAMETERS:
            self.panel._parameter.setCurrentIndex(self.panel._parameter.findData(parameter))
            self.panel._connect.setChecked(True)
            self.panel._table.selectRow(1)
            self.app.processEvents()
            self.assertFalse(self.panel.grab().isNull())

    def test_inspection_does_not_edit_poses_or_fitted_parameters(self):
        expected = copy.deepcopy(self.library)
        native_pose = self.rig.poses[1].bone_locals[1].copy()
        self.panel.set_source(self.rig, self.library)
        for _label, parameter, _units, _channels in PARAMETERS:
            self.panel._parameter.setCurrentIndex(self.panel._parameter.findData(parameter))
            self.panel._table.selectRow(1)
            self.assertIn("sixty degree support", self.panel._detail.text())
            self.assertTrue(self.panel._selection.isVisible())
        np.testing.assert_array_equal(self.library.base.local_centers, expected.base.local_centers)
        np.testing.assert_array_equal(self.library.keys[0].delta_rotations, expected.keys[0].delta_rotations)
        np.testing.assert_array_equal(self.rig.poses[1].bone_locals[1], native_pose)
        self.assertEqual(self.library.to_json(self.rig.skeleton), expected.to_json(self.rig.skeleton))

    def test_replacement_and_unload_clear_stale_curves(self):
        self.panel.set_source(self.rig, self.library)
        replacement = copy.deepcopy(self.library)
        replacement.keys.clear()
        self.panel.set_source(self.rig, replacement)
        self.assertEqual(self.panel._table.rowCount(), 1)
        self.panel.set_source(None, None)
        self.assertEqual(self.panel._table.rowCount(), 0)
        self.assertEqual(self.panel._primitive.count(), 0)
        self.assertFalse(self.panel._parameter.isEnabled())
        self.assertIn("Load a rigged model", self.panel._status.text())

    def test_real_main_window_publishes_completed_library_in_new_dock(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(app_settings, "load", return_value=app_settings.defaults()), \
                mock.patch.object(app_settings, "load_panel", return_value={}), \
                mock.patch.object(app_settings, "save_panel"), \
                mock.patch.object(MainWindow, "_restore_workspace_layout", new=lambda self: False), \
                mock.patch.object(MainWindow, "_save_workspace_layout", new=lambda self: None), \
                mock.patch("main_window.default_morph_fit_path",
                    side_effect=lambda name: Path(tmp) / "auto_morph_fit.json"):
            window = MainWindow(mesh_dir=Path(tmp))
            try:
                window._suppress_pose_sdf_recompute = True
                window._rig_panel.setChecked(True)
                window._rig_panel.set_rigged_mesh(self.rig)
                window._rig_panel.set_bone_local(self.library.base)
                window._pose_corrective_worker = SimpleNamespace(result=self.library)
                window._on_pose_corrective_worker_finished()
                self.assertIs(window._morph_curves._library, self.library)
                self.assertEqual(window._morph_curves._table.rowCount(), 2)
                dock = window._dock_widgets["morph_curves"]
                self.assertEqual(dock.windowTitle(), "Morph Curves")
                self.assertIn(dock, window.tabifiedDockWidgets(window._dock_widgets["dashboard"]))
                action = next(a for a in window._view_menu.actions() if a.text() == "Morph Curves")
                dock.close()
                action.trigger()
                self.assertFalse(dock.isHidden())
                path = Path(tmp) / "morph.json"
                save_native_morph_fit(self.library, self.rig, path)
                with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(path), "JSON")):
                    window._on_load_morph_fit()
                self.assertIs(window._morph_curves._library, window._pose_correctives)
                self.assertIsNot(window._morph_curves._library, self.library)
                self.assertEqual(window._morph_curves._table.rowCount(), 2)
                window._rig_panel.setChecked(False)
                window._set_base_mesh(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), np.empty((0, 3), dtype=np.int32))
                self.assertEqual(window._morph_curves._table.rowCount(), 0)
            finally:
                window._panel_save_timer.stop()
                window.close()
                window.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
