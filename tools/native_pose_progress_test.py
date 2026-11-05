"""Training follows actual worker poses without starting UI-side skinning/SDF."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtWidgets

from rig_loader import load_rigged_mesh
from rig_panel import RigModePanel


class NativePoseProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.rig = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.panel = RigModePanel(pose_dir=Path(self.temp.name))
        self.panel.setChecked(True)
        self.panel.set_rigged_mesh(self.rig)
        self.panel.generate_systematic_source()
        self.changes = []
        self.slider_changes = []
        self.panel.poseChanged.connect(self.changes.append)
        self.panel._slider_pose.valueChanged.connect(self.slider_changes.append)

    def tearDown(self):
        self.panel.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_progress_updates_timeline_and_labels_without_pose_events(self):
        self.panel.set_auto_pipeline_running(True)
        with mock.patch.object(self.panel, "get_deformed_mesh") as skinning:
            self.panel.show_training_pose(3, "actual worker target")
            skinning.assert_not_called()
        self.assertEqual(self.panel.current_pose_index, 3)
        self.assertEqual(self.panel.current_pose_frame, 3)
        self.assertEqual(self.panel._lbl_pose_name.text(), "Pose: actual worker target")
        last = len(self.panel.active_poses) - 1
        self.assertEqual(self.panel._lbl_pose.text(), f"3 / {last}")
        self.assertEqual(self.changes, [])
        self.assertEqual(self.slider_changes, [])
        self.assertTrue(self.panel._pose_training_running)
        self.assertFalse(self.panel._slider_pose.isEnabled())
        for widget in (self.panel._btn_workflow, self.panel._btn_fit_base,
                self.panel._btn_fit_morphs, self.panel._spin_morph_steps):
            self.assertFalse(widget.isEnabled())

    def test_frozen_controls_survive_progress_and_release(self):
        self.changes.clear()
        self.panel.set_auto_pipeline_running(True)
        self.panel.show_training_pose(4)
        self.assertEqual(self.panel.current_pose_frame, 4)
        self.panel.set_auto_pipeline_running(False)
        self.assertTrue(self.panel._slider_pose.isEnabled())
        self.assertEqual(self.panel.current_pose_frame, 4)
        self.assertEqual(self.panel.current_pose.name, self.panel.active_poses[4].name)
        self.assertEqual(self.changes, [])
        self.assertEqual(self.slider_changes, [])
        self.panel._slider_pose.setValue(5)
        self.assertEqual(self.changes, [5])
        self.assertEqual(self.panel.current_pose_frame, 5)

    def test_release_preserves_the_real_deformation_of_the_last_training_pose(self):
        bind = self.panel.get_deformed_mesh(0)
        expected = self.panel.get_deformed_mesh(2)
        self.panel.set_auto_pipeline_running(True)
        self.panel.show_training_pose(2)
        self.panel.set_auto_pipeline_running(False)
        actual = self.panel.get_deformed_mesh()
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertGreater(np.mean(np.linalg.norm(actual - bind, axis=1)), 0.1)

    def test_progress_clamps_indices_and_restores_preexisting_signal_blocks(self):
        self.panel.set_auto_pipeline_running(True)
        self.panel._slider_pose.blockSignals(True)
        self.panel.show_training_pose(-1)
        self.assertEqual(self.panel.current_pose_index, 0)
        self.panel.show_training_pose(999)
        last = len(self.panel.active_poses) - 1
        self.assertGreater(last, 50)
        self.assertEqual(self.panel.current_pose_index, last)
        self.assertTrue(self.panel._slider_pose.signalsBlocked())
        self.assertEqual(self.changes, [])
        self.assertEqual(self.slider_changes, [])


if __name__ == "__main__":
    unittest.main()
