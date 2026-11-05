"""Joint geometry and real Qt interaction regressions (no OpenGL rendering)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from joint_dof import (
    JointDof, joint_pointer_angle, joint_rotation_arc,
    joint_rotation_frames, project_joint_points,
)
from joint_view import JointOverlay
from skeleton import Bone, Pose, Skeleton, mat4_compose
from widgets import DropGLView


class JointGeometryTest(unittest.TestCase):
    def test_settings_are_per_bone_and_reject_invalid_limits(self):
        first, second = Bone("first", 0), Bone("second", 1)
        first.joint_dof.set_axis(0, False, -45, 45)
        first.joint_dof.set_axis(1, True, 20, 20)
        self.assertEqual(first.joint_dof.count, 1)
        self.assertEqual(second.joint_dof.count, 3)
        for minimum, maximum in ((50, -50), (-181, 0), (0, float("nan"))):
            with self.assertRaises(ValueError):
                first.joint_dof.set_axis(0, True, minimum, maximum)

    def test_scaled_rotated_frames_remain_orthonormal(self):
        from mesh_settings import rotation_matrix
        rotation = rotation_matrix(31, -16, 44)
        world = np.eye(4)
        world[:3, :3] = rotation @ np.diag([2.0, 0.5, 4.0])
        frame = joint_rotation_frames(world[None])[0]
        np.testing.assert_allclose(frame, rotation, atol=1e-6)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-6)
        world[:3, 0] *= -1
        self.assertAlmostEqual(float(np.linalg.det(joint_rotation_frames(world[None])[0])), 1, places=5)
        self.assertEqual(joint_rotation_frames(np.empty((0, 4, 4))).shape, (0, 3, 3))

    def test_arcs_are_perpendicular_to_axis_and_follow_limits(self):
        center = np.array([2.0, 3.0, 4.0])
        for axis in range(3):
            arc = joint_rotation_arc(center, np.eye(3), axis, 2, 0, 90)
            np.testing.assert_allclose(arc[:, axis], center[axis], atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(arc - center, axis=1), 2, atol=1e-6)
            self.assertAlmostEqual(arc[0, (axis + 1) % 3] - center[(axis + 1) % 3], 2)
            self.assertAlmostEqual(arc[-1, (axis + 2) % 3] - center[(axis + 2) % 3], 2)

    def test_projection_clips_and_uses_qt_vertical_coordinates(self):
        pixels, depth, visible = project_joint_points(
            [[0, 0, 0], [1, 1, 0], [0, 0, 2]], np.eye(4), 800, 600)
        np.testing.assert_allclose(pixels[:2], [[400, 300], [800, 0]])
        np.testing.assert_array_equal(visible, [True, True, False])
        behind = np.eye(4)
        behind[3, 3] = -1
        self.assertFalse(project_joint_points([[0, 0, 0]], behind, 800, 600)[2][0])


class JointInteractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.view = DropGLView()
        self.view.resize(800, 600)
        self.view.setCameraPosition(distance=3, elevation=15, azimuth=45)
        self.layer = JointOverlay(self.view)
        self.view.add_corner_widget(self.layer, "fill", 0)
        self.view.add_corner_widget(self.layer.editor, "br", 52)
        self.rig = object()
        self.dofs = [JointDof(), JointDof()]
        self.positions = np.array([[0, 0, 0], [0, 0, 0.4]], dtype=np.float32)
        self.layer.set_joints(self.positions, names=["Root", "Elbow"], dofs=self.dofs, rig_key=self.rig)
        self.layer.set_enabled(True)
        self.layer.editor.hide()
        self.view._reposition_overlays()

    def tearDown(self):
        self.layer.set_enabled(False)
        self.view.close()
        self.view.deleteLater()
        self.app.processEvents()

    def point(self, index=0):
        pixels, _depth, _valid = project_joint_points(
            self.layer.positions, self.layer._projection(), 800, 600)
        return QtCore.QPointF(*map(float, pixels[index]))

    def mouse(self, kind, point, button=QtCore.Qt.NoButton, buttons=QtCore.Qt.NoButton):
        event = QtGui.QMouseEvent(kind, point, point, button, buttons, QtCore.Qt.NoModifier)
        # Deliver through the real view event filter and camera event handlers.
        self.app.sendEvent(self.view, event)

    def click(self, point):
        self.mouse(QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        self.mouse(QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton)

    def pin_rotation_axis(self, minimum=-60, maximum=60, elevation=90):
        self.view.setCameraPosition(elevation=elevation, azimuth=0)
        # A face-on Z rotation view would put the two original joints at the
        # same screen position; separate them so this test picks the root.
        self.layer.set_joints(self.positions + [[0, 0, 0], [0.7, 0, 0]],
                              names=["Root", "Elbow"], dofs=self.dofs, rig_key=self.rig)
        self.dofs[0].enabled[:] = False
        self.dofs[0].set_axis(2, True, minimum, maximum)
        self.click(self.point())
        self.assertEqual(self.layer.selected, 0)

    def handle_point(self, bound):
        handle = next(handle for handle in self.layer._limit_handles(self.layer._gizmo_geometry())
                      if handle["axis"] == 2 and handle["bound"] == bound)
        self.assertTrue(handle["visible"])
        return QtCore.QPointF(*map(float, handle["pixel"]))

    def angle_point(self, angle, bound="maximum", radius_factor=None):
        geometry = self.layer._gizmo_geometry()
        factor = (0.85 if bound == "minimum" else 1.16) if radius_factor is None else radius_factor
        point = joint_rotation_arc(geometry["center"], geometry["frame"], 2,
                                   geometry["radius"] * factor, angle, angle)[0]
        pixels, _depth, _visible = project_joint_points(
            point[None], geometry["matrix"], 800, 600)
        return QtCore.QPointF(*map(float, pixels[0]))

    def drag_limit(self, bound, angle):
        self.mouse(QtCore.QEvent.MouseButtonPress, self.handle_point(bound),
                   QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        self.assertIsNotNone(self.layer._handle_drag)
        point = self.angle_point(angle, bound)
        self.mouse(QtCore.QEvent.MouseMove, point, buttons=QtCore.Qt.LeftButton)
        self.mouse(QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton)

    def test_pointer_angle_matches_projected_local_rotation_planes(self):
        from mesh_settings import rotation_matrix
        frame = rotation_matrix(0, 0, 30)
        center = np.array([0.1, -0.2, 0.3])
        matrix = self.layer._projection()
        for axis in range(3):
            for angle in (-135, -45, 0, 60, 160):
                with self.subTest(axis=axis, angle=angle):
                    point = joint_rotation_arc(center, frame, axis, 0.2, angle, angle)[0]
                    pixels, _depth, _valid = project_joint_points(point[None], matrix, 800, 600)
                    actual = joint_pointer_angle(pixels[0], matrix, 800, 600, center, frame, axis)
                    self.assertIsNotNone(actual)
                    self.assertAlmostEqual((actual - angle + 180) % 360 - 180, 0, places=3)
        self.assertIsNone(joint_pointer_angle([0, 0], np.zeros((4, 4)), 800, 600,
                                              center, frame, 2))

    def test_dragging_both_limits_syncs_fields_and_keeps_camera_still(self):
        self.pin_rotation_axis()
        camera = (self.view.opts["azimuth"], self.view.opts["elevation"], self.view.opts["distance"])
        self.drag_limit("minimum", -20)
        self.assertAlmostEqual(self.dofs[0].minimum[2], -20, places=1)
        self.assertEqual(self.layer.editor._minima[2].value(), self.dofs[0].minimum[2])
        self.drag_limit("maximum", 110)
        self.assertAlmostEqual(self.dofs[0].maximum[2], 110, places=1)
        self.assertEqual(self.layer.editor._maxima[2].value(), self.dofs[0].maximum[2])
        self.assertEqual(self.layer.selected, 0)
        self.assertFalse(self.layer.editor.isHidden())
        self.assertEqual(camera, (self.view.opts["azimuth"], self.view.opts["elevation"],
                                  self.view.opts["distance"]))
        self.assertIsNone(self.layer._handle_drag)

    def test_handles_exist_at_full_circle_and_closed_span(self):
        self.pin_rotation_axis(-180, 180)
        self.assertGreaterEqual((self.handle_point("minimum") - self.handle_point("maximum")).manhattanLength(), 16)
        self.drag_limit("minimum", -150)
        self.drag_limit("maximum", 150)
        np.testing.assert_allclose([self.dofs[0].minimum[2], self.dofs[0].maximum[2]], [-150, 150], atol=0.1)
        self.dofs[0].set_axis(2, True, 40, 40)
        self.layer.editor._sync()
        self.drag_limit("minimum", 20)
        self.assertAlmostEqual(self.dofs[0].minimum[2], 20, places=1)
        self.assertEqual(self.dofs[0].maximum[2], 40)
        self.assertEqual(self.dofs[0].count, 1)

    def test_dragged_limits_clamp_without_crossing_and_unwrap_the_seam(self):
        self.pin_rotation_axis()
        self.drag_limit("minimum", 75)
        self.assertEqual(self.dofs[0].minimum[2], self.dofs[0].maximum[2])
        self.dofs[0].set_axis(2, True, -100, 175)
        point = self.handle_point("maximum")
        self.mouse(QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        beyond_seam = self.angle_point(-175)
        self.mouse(QtCore.QEvent.MouseMove, beyond_seam, buttons=QtCore.Qt.LeftButton)
        self.assertEqual(self.dofs[0].maximum[2], 180)
        reverse = self.angle_point(175)
        self.mouse(QtCore.QEvent.MouseMove, reverse, buttons=QtCore.Qt.LeftButton)
        self.assertEqual(self.dofs[0].maximum[2], 170)
        self.mouse(QtCore.QEvent.MouseButtonRelease, reverse, QtCore.Qt.LeftButton)

    def test_escape_cancels_drag_before_releasing_the_joint(self):
        self.pin_rotation_axis()
        self.mouse(QtCore.QEvent.MouseButtonPress, self.handle_point("maximum"),
                   QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        self.mouse(QtCore.QEvent.MouseMove, self.angle_point(100), buttons=QtCore.Qt.LeftButton)
        self.assertGreater(self.dofs[0].maximum[2], 60)
        self.app.sendEvent(self.view, QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier))
        self.assertEqual(self.dofs[0].maximum[2], 60)
        self.assertEqual(self.layer.editor._maxima[2].value(), 60)
        self.assertEqual(self.layer.selected, 0)
        self.assertIsNone(self.layer._handle_drag)
        self.mouse(QtCore.QEvent.MouseButtonRelease, self.angle_point(100), QtCore.Qt.LeftButton)
        self.assertEqual(self.layer.selected, 0)
        self.app.sendEvent(self.view, QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier))
        self.assertIsNone(self.layer.selected)

    def test_edge_on_drag_supports_vertical_motion_and_hidden_axes_have_no_handles(self):
        self.pin_rotation_axis(-30, 90, elevation=0)
        point = self.handle_point("maximum")
        self.assertIsNone(self.layer._pointer_angle(point, self.layer._gizmo_geometry(), 2))
        self.mouse(QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        target = point + QtCore.QPointF(0, -15)
        self.mouse(QtCore.QEvent.MouseMove, target, buttons=QtCore.Qt.LeftButton)
        self.mouse(QtCore.QEvent.MouseButtonRelease, target, QtCore.Qt.LeftButton)
        self.assertEqual(self.dofs[0].maximum[2], 105)
        self.dofs[0].enabled[2] = False
        self.assertEqual(self.layer._limit_handles(self.layer._gizmo_geometry()), [])
        self.assertIsNone(self.layer.pick_handle(target))

    def test_drag_survives_leave_and_ends_on_hide(self):
        self.pin_rotation_axis()
        self.mouse(QtCore.QEvent.MouseButtonPress, self.handle_point("maximum"),
                   QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        self.app.sendEvent(self.view, QtCore.QEvent(QtCore.QEvent.Leave))
        self.assertIsNotNone(self.layer._handle_drag)
        self.mouse(QtCore.QEvent.MouseMove, self.angle_point(100), buttons=QtCore.Qt.LeftButton)
        self.layer.set_enabled(False)
        self.assertIsNone(self.layer._handle_drag)
        self.assertIsNone(self.layer.selected)
        self.assertAlmostEqual(self.dofs[0].maximum[2], 100, places=1)

    def test_pie_sector_is_transparent_and_follows_numeric_limits(self):
        self.pin_rotation_axis(-30, 80)
        image = QtGui.QImage(800, 600, QtGui.QImage.Format_ARGB32)
        image.fill(QtCore.Qt.transparent)
        self.layer.render(image, QtCore.QPoint(), QtGui.QRegion(), QtWidgets.QWidget.DrawChildren)
        inside = self.angle_point(40, radius_factor=0.65)
        outside = self.angle_point(160, radius_factor=0.65)
        color = image.pixelColor(round(inside.x()), round(inside.y()))
        self.assertGreater(color.alpha(), 0)
        self.assertLess(color.alpha(), 150)
        self.assertGreater(color.blue(), color.red())
        self.assertEqual(image.pixelColor(round(outside.x()), round(outside.y())).alpha(), 0)
        self.layer.editor._maxima[2].setValue(120)
        handle = self.handle_point("maximum")
        expected = self.angle_point(120)
        self.assertLess((handle - expected).manhattanLength(), 0.1)

    def test_disabled_rings_do_not_inherit_another_axes_sector_fill(self):
        from mesh_settings import rotation_matrix
        self.pin_rotation_axis()
        self.dofs[0].enabled[:] = False
        self.dofs[0].set_axis(0, True, -30, 80)
        self.layer.rotations = np.tile(np.eye(3), (2, 1, 1))
        self.layer.rotations[0] = rotation_matrix(0, 90, 0)
        geometry = self.layer._gizmo_geometry()
        image = QtGui.QImage(800, 600, QtGui.QImage.Format_ARGB32)
        image.fill(QtCore.Qt.transparent)
        self.layer.render(image, QtCore.QPoint(), QtGui.QRegion(), QtWidgets.QWidget.DrawChildren)
        points = np.array([joint_rotation_arc(geometry["center"], geometry["frame"], 0,
                                              geometry["radius"] * 0.65, angle, angle)[0]
                           for angle in (40, 160)])
        pixels, _depth, _valid = project_joint_points(points, geometry["matrix"], 800, 600)
        inside, outside = [image.pixelColor(round(p[0]), round(p[1])) for p in pixels]
        self.assertGreater(inside.alpha(), 0)
        self.assertLess(inside.alpha(), 100)
        self.assertEqual(outside.alpha(), 0)

    def test_hover_leave_pin_edit_switch_and_release(self):
        point = self.point(1)
        self.mouse(QtCore.QEvent.MouseMove, point)
        self.assertEqual(self.layer.hovered, 1)
        self.assertTrue(self.layer.editor.isHidden())
        self.app.sendEvent(self.view, QtCore.QEvent(QtCore.QEvent.Leave))
        self.assertIsNone(self.layer.hovered)
        self.click(point)
        self.assertEqual(self.layer.selected, 1)
        self.assertFalse(self.layer.editor.isHidden())
        self.app.sendEvent(self.view, QtCore.QEvent(QtCore.QEvent.Leave))
        self.assertEqual(self.layer.selected, 1)
        self.layer.editor._checks[0].setChecked(False)
        self.layer.editor._minima[1].setValue(-35)
        self.layer.editor._maxima[1].setValue(80)
        self.assertEqual(self.dofs[1].count, 2)
        self.assertEqual(self.dofs[1].minimum[1], -35)
        self.assertEqual(self.dofs[1].maximum[1], 80)
        self.assertEqual(self.dofs[0].count, 3)
        self.click(self.point(0))
        self.assertEqual(self.layer.selected, 0)
        self.click(QtCore.QPointF(790, 590))
        self.assertIsNone(self.layer.selected)
        self.assertTrue(self.layer.editor.isHidden())

    def test_drag_from_joint_keeps_camera_navigation_and_does_not_pin(self):
        point = self.point()
        old_azimuth = self.view.opts["azimuth"]
        self.mouse(QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        self.mouse(QtCore.QEvent.MouseMove, point + QtCore.QPointF(70, 0), buttons=QtCore.Qt.LeftButton)
        self.mouse(QtCore.QEvent.MouseButtonRelease, point + QtCore.QPointF(70, 0), QtCore.Qt.LeftButton)
        self.assertIsNone(self.layer.selected)
        self.assertNotEqual(old_azimuth, self.view.opts["azimuth"])

    def test_pose_updates_preserve_pin_and_settings_new_rig_releases(self):
        self.click(self.point(1))
        self.layer.editor._checks[2].setChecked(False)
        self.layer.set_joints(self.positions + [0.1, 0, 0], names=["Root", "Elbow"],
                              dofs=self.dofs, rig_key=self.rig)
        self.assertEqual(self.layer.selected, 1)
        self.assertFalse(self.layer.dofs[1].enabled[2])
        self.layer.set_joints(self.positions, names=["Root", "Elbow"], rig_key=object())
        self.assertIsNone(self.layer.selected)
        self.assertEqual(self.layer.dofs[1].count, 3)

    def test_visibility_and_escape_release_and_do_not_pick_hidden(self):
        self.click(self.point())
        self.app.sendEvent(self.view, QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier))
        self.assertIsNone(self.layer.selected)
        self.click(self.point())
        self.layer.set_enabled(False)
        self.assertIsNone(self.layer.selected)
        self.assertIsNone(self.layer.pick_joint(self.point()))
        self.assertTrue(self.layer.editor.isHidden())
        self.assertFalse(self.layer._timer.isActive())

    def test_joint_editor_normalizes_crossed_limits(self):
        self.click(self.point())
        editor = self.layer.editor
        editor._maxima[0].setValue(-40)
        editor._minima[0].setValue(20)
        self.assertEqual(self.dofs[0].minimum[0], 20)
        self.assertEqual(self.dofs[0].maximum[0], 20)
        self.assertEqual(self.dofs[0].count, 2)

    def test_near_joint_wins_at_coincident_screen_position(self):
        camera = np.asarray(self.view.cameraPosition())
        toward_camera = camera / np.linalg.norm(camera)
        self.layer.set_joints(np.array([[0, 0, 0], toward_camera * 0.2]), rig_key=self.rig)
        self.assertEqual(self.layer.pick_joint(self.point(0)), 1)
        # Projection must match Qt's own matrix mapping at an off-centre point.
        viewport = self.view.getViewport()
        matrix = self.view.projectionMatrix(viewport, viewport) * self.view.viewMatrix()
        p = matrix.map(QtGui.QVector3D(0.1, 0.2, 0.3))
        pixels, _depth, _valid = project_joint_points([[0.1, 0.2, 0.3]], self.layer._projection(), 800, 600)
        np.testing.assert_allclose(pixels[0], [(p.x() + 1) * 400, (1 - p.y()) * 300], atol=1e-4)

    def test_gizmo_paints_to_image_without_opengl(self):
        self.click(self.point(1))
        self.dofs[1].set_axis(0, False, -180, 180)
        self.dofs[1].set_axis(1, True, -35, 80)
        image = QtGui.QImage(800, 600, QtGui.QImage.Format_ARGB32)
        image.fill(QtCore.Qt.transparent)
        self.layer.render(image, QtCore.QPoint(), QtGui.QRegion(), QtWidgets.QWidget.DrawChildren)
        self.assertGreater(image.pixelColor(400, 300).alpha(), 0)
        self.assertGreater(image.pixelColor(400, 300).red(), 180)
        self.assertLess(image.pixelColor(400, 300).green(), 200)
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)

    def test_viewer_joint_toggle_is_independent_and_static_mesh_clears(self):
        from viewer3d import SceneViewer3D
        viewer = SceneViewer3D()
        try:
            viewer.show_bones(self.positions, np.array([-1, 0]))
            controls = viewer._overlay._checks
            self.assertEqual(controls["joints"].text(), "Joint")
            self.assertFalse(controls["joints"].isHidden())
            controls["skeleton"].setChecked(False)
            self.assertFalse(viewer._bone_line_item.visible())
            self.assertFalse(viewer._joint_overlay.isHidden())
            controls["joints"].setChecked(False)
            controls["skeleton"].setChecked(True)
            self.assertTrue(viewer._bone_line_item.visible())
            self.assertTrue(viewer._joint_overlay.isHidden())
            line = viewer._bone_line_item
            viewer.show_bones(self.positions + [0.1, 0, 0], np.array([-1, 0]))
            self.assertIs(viewer._bone_line_item, line)
            self.assertTrue(viewer._joint_overlay.isHidden())
            viewer.remove_skeleton()
            self.assertTrue(controls["joints"].isHidden())
            self.assertEqual(len(viewer._joint_overlay.positions), 0)
        finally:
            viewer.widget.deleteLater()

    def test_main_window_passes_posed_local_frames_and_global_rotation(self):
        from main_window import MainWindow
        from mesh_settings import rotation_matrix, rotate_mesh
        root = Bone("Root", 0)
        child = Bone("Elbow", 1, 0)
        child.local_bind_transform[2, 3] = 0.5
        skeleton = Skeleton([root, child])
        quaternion = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        pose = Pose("posed", {1: mat4_compose(np.array([0, 0, 0.5]), quaternion, np.ones(3))})
        calls = []
        harness = SimpleNamespace(
            _mesh_rotation=(0, 90, 0),
            _apply_rotation=lambda p: rotate_mesh(p, 0, 90, 0, center=[0, 0, 0]),
            _viewer=SimpleNamespace(show_bones=lambda *args, **kwargs: calls.append((args, kwargs))))
        MainWindow._show_skeleton_for_pose(harness, SimpleNamespace(skeleton=skeleton), pose)
        args, kwargs = calls[0]
        np.testing.assert_allclose(args[0][1], [0.5, 0, 0], atol=1e-6)
        np.testing.assert_allclose(kwargs["joint_rotations"][1],
                                   rotation_matrix(0, 90, 0) @ rotation_matrix(0, 0, 90), atol=1e-6)
        self.assertIs(kwargs["joint_dofs"][1], child.joint_dof)
        self.assertIs(kwargs["rig_key"], skeleton)


if __name__ == "__main__":
    unittest.main()
