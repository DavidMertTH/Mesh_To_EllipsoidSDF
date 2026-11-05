"""Manual pose authoring through real Qt controls, FK and T-Pose skinning."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtCore, QtWidgets

from joint_pose import JointPoseEdits, joint_pose_angles
from joint_view import JointEditor
from mesh_settings import rotation_matrix
from rig_loader import load_rigged_mesh
from rig_panel import RigModePanel
from skeleton import Bone, Pose, Skeleton


class JointPoseMathTest(unittest.TestCase):
    def test_relative_bind_rotation_preserves_translation_scale_and_shear(self):
        bone = Bone("rotated bind", 0)
        bone.local_bind_transform[:3, :3] = rotation_matrix(31, -22, 16)
        source_local = bone.local_bind_transform.copy()
        stretch = np.array([[2, .2, 0], [.2, 1, .1], [0, .1, 3]])
        source_local[:3, :3] = source_local[:3, :3] @ rotation_matrix(20, 30, 40) @ stretch
        source_local[:3, 3] = [3, 4, 5]
        source = Pose("source", {0: source_local})
        before = source_local.copy()
        skeleton = Skeleton([bone])
        edit = JointPoseEdits(skeleton, source)
        np.testing.assert_allclose(edit.angles[0], [20, 30, 40], atol=1e-4)
        self.assertTrue(edit.set_axis(0, 0, 50))
        local = edit.apply().bone_locals[0]
        np.testing.assert_allclose(local[:3, :3], bone.local_bind_transform[:3, :3] @ rotation_matrix(50, 30, 40) @ stretch, atol=1e-6)
        np.testing.assert_array_equal(local[:3, 3], [3, 4, 5])
        np.testing.assert_array_equal(source.bone_locals[0], before)

    def test_only_authored_axis_is_clamped_and_locked_axes_reject_edits(self):
        bone = Bone("joint", 0)
        bone.joint_dof.set_axis(0, True, -30, 40)
        bone.joint_dof.set_axis(1, False, 0, 0)
        bone.joint_dof.set_axis(2, True, 0, 0)
        local = np.eye(4)
        local[:3, :3] = rotation_matrix(60, 20, 15)
        edit = JointPoseEdits(Skeleton([bone]), Pose("imported", {0: local}))
        self.assertFalse(edit.set_axis(0, 1, 99))
        self.assertFalse(edit.set_axis(0, 2, 99))
        self.assertFalse(edit.set_axis(0, 0, float("nan")))
        self.assertFalse(edit.set_axis(9, 0, 20))
        self.assertTrue(edit.set_axis(0, 0, 999))
        np.testing.assert_allclose(edit.angles[0], [40, 20, 15], atol=1e-5)

    def test_euler_round_trip_at_gimbal_lock(self):
        skeleton = Skeleton([Bone("root", 0)])
        for y in (-90, 90):
            local = np.eye(4)
            local[:3, :3] = rotation_matrix(25, y, -30)
            extracted = joint_pose_angles(skeleton, Pose("singular", {0: local}))[0]
            np.testing.assert_allclose(rotation_matrix(*extracted), local[:3, :3], atol=1e-6)

    def test_cached_angles_survive_crossing_ninety_degrees_and_reset(self):
        skeleton = Skeleton([Bone("root", 0)])
        source = Pose.t_pose()
        edit = JointPoseEdits(skeleton, source)
        for y in (89, 90, 91, 120, 180):
            self.assertTrue(edit.set_axis(0, 1, y))
            np.testing.assert_array_equal(edit.angles[0], [0, y, 0])
        self.assertTrue(edit.reset_joint(0))
        self.assertIs(edit.apply(), source)
        np.testing.assert_allclose(edit.angles[0], 0)


class JointPoseUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.rig = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.panel = RigModePanel(pose_dir=Path(self.temp.name))
        self.rm = copy.deepcopy(self.rig)
        self.rm.poses = [Pose.t_pose(), Pose("second", {})]
        self.panel.setChecked(True)
        self.panel.set_rigged_mesh(self.rm)
        # These tests author edits on the mesh's own frames (API pose source).
        self.panel.select_source("native")

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def bone(self, name):
        return next(i for i, b in enumerate(self.rm.skeleton.bones) if b.name.endswith(":" + name))

    def test_real_elbows_and_knees_bend_in_anatomical_directions(self):
        skeleton = self.rm.skeleton
        base = skeleton.compute_world_transforms(Pose.t_pose())
        for joint, child, axis, angle, direction in (
            ("LeftForeArm", "LeftHand", 2, 90, 1),
            ("RightForeArm", "RightHand", 2, -90, 1),
            ("LeftLeg", "LeftFoot", 0, -90, -1),
            ("RightLeg", "RightFoot", 0, -90, -1),
        ):
            index, end = self.bone(joint), self.bone(child)
            self.panel.set_joint_rotation(index, axis, angle)
            world = skeleton.compute_world_transforms(self.panel.current_pose)
            self.assertGreater(direction * (world[end, 2, 3] - base[end, 2, 3]), .01)
            np.testing.assert_allclose(world[index, :3, 3], base[index, :3, 3], atol=1e-6)
            self.panel.reset_joint_rotations()

    def test_edits_are_per_frame_and_source_and_do_not_modify_imported_poses(self):
        elbow = self.bone("LeftForeArm")
        source = self.panel.active_poses[0]
        self.panel.set_joint_rotation(elbow, 2, 45)
        self.assertEqual(source.bone_locals, {})
        self.panel._slider_pose.setValue(1)
        self.assertNotIn(elbow, self.panel.current_pose.bone_locals)
        self.panel._slider_pose.setValue(0)
        self.assertIn(elbow, self.panel.current_pose.bone_locals)
        self.panel._library_clips = [SimpleNamespace(name="other", poses=[Pose("other", {})])]
        self.assertTrue(self.panel.select_source("other"))
        self.assertNotIn(elbow, self.panel.current_pose.bone_locals)
        self.panel.select_source("native")
        np.testing.assert_allclose(self.panel.joint_rotation_state(self.panel.current_pose)[0][elbow], [0, 0, 45], atol=1e-5)

    def test_reset_one_joint_leaves_other_edits_and_reset_pose_restores_source(self):
        elbow, knee = self.bone("LeftForeArm"), self.bone("LeftLeg")
        self.panel.set_joint_rotation(elbow, 2, 45)
        self.panel.set_joint_rotation(knee, 0, -90)
        self.panel.reset_joint_rotation(elbow)
        self.assertNotIn(elbow, self.panel.current_pose.bone_locals)
        self.assertIn(knee, self.panel.current_pose.bone_locals)
        self.panel.reset_joint_rotations()
        self.assertIs(self.panel.current_pose, self.panel.active_poses[0])

    def test_selection_displays_out_of_range_imported_angles_without_changing_pose(self):
        elbow = self.bone("LeftForeArm")
        local = self.rm.skeleton.bones[elbow].local_bind_transform.copy()
        local[:3, :3] = local[:3, :3] @ rotation_matrix(20, 0, -40)
        self.rm.poses[0] = Pose("imported", {elbow: local})
        editor = JointEditor()
        calls = []
        editor.rotationChanged.connect(lambda *args: calls.append(args))
        angles, _ = self.panel.joint_rotation_state(self.panel.current_pose)
        editor.set_joint("elbow", self.rm.skeleton.bones[elbow].joint_dof, angles[elbow])
        self.assertEqual(calls, [])
        self.assertAlmostEqual(editor._angle_spins[2].value(), -40)
        self.assertFalse(editor._angle_spins[0].isEnabled())
        np.testing.assert_array_equal(self.panel.current_pose.bone_locals[elbow], local)
        editor._angle_spins[2].setValue(-30)
        self.assertEqual(calls, [(2, 0.0)])
        self.assertEqual(editor._angle_spins[2].value(), 0)
        editor.deleteLater()

    def test_new_model_discards_old_joint_edits(self):
        self.panel.set_joint_rotation(0, 1, 45)
        self.panel.set_rigged_mesh(copy.deepcopy(self.rig))
        self.assertEqual(self.panel._joint_edits, {})
        self.assertEqual(self.panel._cmb_joint.count(), 65)

    def test_rotation_controls_refresh_real_mesh_skeleton_and_ellipsoids(self):
        from bone_ellipsoid_mapper import BoneLocalEllipsoids
        from main_window import MainWindow
        from viewer3d import SceneViewer3D

        viewer = SceneViewer3D()
        elbow = self.bone("LeftForeArm")
        bind = self.rm.skeleton.compute_world_transforms(None)
        midpoint = (bind[elbow, :3, 3] + bind[self.bone("LeftHand"), :3, 3]) * .5
        local_center = (np.linalg.inv(bind[elbow]) @ np.append(midpoint, 1))[:3]
        local = BoneLocalEllipsoids(
            local_centers=np.array([local_center], dtype=np.float32),
            local_radii=np.full((1, 3), .02, dtype=np.float32),
            local_rotations=np.array([[0, 0, 0, 1]], dtype=np.float32),
            bone_assignments=np.array([elbow], dtype=np.int32),
        )
        self.panel.set_bone_local(local)
        draws = []
        shape = SimpleNamespace(render=lambda _v, c, r, q, e: draws.append(c.copy()))
        owner = SimpleNamespace(
            _rig_panel=self.panel, _viewer=viewer, _mesh_rotation=(0, 0, 0),
            _apply_rotation=lambda v: v, _pose_correctives=None,
            _shape=shape, _shapes_by_id={"ellipsoid": shape},
            _suppress_pose_sdf_recompute=False, _pose_sdf_timer=QtCore.QTimer(),
            _status=SimpleNamespace(showMessage=lambda _s: None),
        )
        owner._show_skeleton_for_pose = MethodType(MainWindow._show_skeleton_for_pose, owner)
        self.panel.poseChanged.connect(lambda i: MainWindow._on_rig_pose_changed(owner, i))
        self.panel.jointEditRequested.connect(viewer.select_joint)
        viewer.connect_joint_rotation_controls(self.panel.set_joint_rotation,
            self.panel.reset_joint_rotation, self.panel.reset_joint_rotations, self.panel.select_joint)
        try:
            MainWindow._on_rig_pose_changed(owner, 0)
            vertices = viewer._mesh_verts.copy()
            positions = viewer._bone_positions.copy()
            center = draws[-1].copy()
            # Open through the rig-panel button even when joint markers are off.
            viewer._overlay._checks["joints"].setChecked(False)
            self.panel._cmb_joint.setCurrentIndex(elbow)
            self.panel._btn_edit_joint.click()
            editor = viewer._joint_overlay.editor
            self.assertEqual(viewer._joint_overlay.selected, elbow)
            self.assertFalse(viewer._joint_overlay.isHidden())
            self.assertEqual(editor.sizeHint().width(), editor.width())
            self.assertTrue(editor._angle_sliders[2].isEnabled())
            self.assertFalse(editor._angle_sliders[0].isEnabled())
            editor._angle_sliders[2].setValue(900)
            self.assertGreater(np.max(np.linalg.norm(viewer._mesh_verts - vertices, axis=1)), .05)
            self.assertGreater(np.max(np.linalg.norm(viewer._bone_positions - positions, axis=1)), .05)
            self.assertGreater(np.linalg.norm(draws[-1] - center), .01)
            self.assertAlmostEqual(editor._angle_spins[2].value(), 90)
            self.assertTrue(editor._reset_joint.isEnabled())
            np.testing.assert_array_equal(owner._pending_pose_mesh[0], viewer._mesh_verts)
            self.assertTrue(owner._pose_sdf_timer.isActive())
            editor._reset_joint.click()
            np.testing.assert_allclose(viewer._mesh_verts, vertices, atol=1e-6)
            np.testing.assert_allclose(draws[-1], center, atol=1e-6)
            self.assertFalse(editor._reset_joint.isEnabled())
            # Selecting a different joint in the scene synchronizes the panel.
            viewer.select_joint(self.bone("LeftLeg"))
            self.assertEqual(self.panel._cmb_joint.currentIndex(), self.bone("LeftLeg"))
        finally:
            owner._pose_sdf_timer.stop()
            viewer._joint_overlay.set_enabled(False)
            viewer.widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
