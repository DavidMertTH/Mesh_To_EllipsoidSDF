"""Joint functions, live pose error and the animation preview."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtWidgets

from animation_preview import AnimationTrack, random_motion, random_multi_axis_poses
from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from joint_pose import JointPoseEdits, joint_pose_angles
from main_window import MainWindow
from morph_error import SurfaceErrorMeter, pair_signed_distance
from morph_error_panel import MorphErrorPanel
from morph_function import JointFunctionModel
from pose_correctives import PoseCorrectiveKey, PoseCorrectiveLibrary
from rig_panel import RigModePanel
from rig_world_transform import RigWorldTransform
from skeleton import Pose
from superquadric_geometry import signed_distance_batch
from tools.native_pose_ui_test import WindowHost, make_library, make_rig

ANGLES = (-50, -30, -10, 10, 25, 40, 55, 70, 85)


def hinge_pose(rig, angle):
    edit = JointPoseEdits(rig.skeleton, Pose.t_pose())
    edit.set_axis(1, 2, angle)
    return edit.apply()


def target(angle):
    """A cubic of the quaternion component, which the 13 features represent exactly."""
    s = np.sin(np.radians(angle) / 2)
    return .4 * s + .2 * s * s - .3 * s ** 3


def polynomial_library(rig, angles=ANGLES):
    base = make_library(rig).base
    keys = []
    for angle in angles:
        value = target(angle)
        keys.append(PoseCorrectiveKey(
            f"hinge {angle}", delta_centers=np.zeros((1, 3), np.float32),
            delta_rotations=np.array([[0, 0, np.sin(value / 4), np.cos(value / 4)]], np.float32),
            delta_log_radii=np.array([[value, -value / 2, 0]], np.float32),
            delta_log_shape_exponents=np.array([[value / 5, 0]], np.float32),
            loss=.01, pose=hinge_pose(rig, angle)))
    return PoseCorrectiveLibrary(base, keys, base_pose=Pose.t_pose())


class JointFunctionModelTest(unittest.TestCase):
    def setUp(self):
        self.rig = make_rig()
        self.library = polynomial_library(self.rig)

    def test_fits_and_interpolates_a_joint_polynomial(self):
        model = JointFunctionModel(self.rig.skeleton, self.library)
        self.assertEqual(model.contexts, [[1]])           # the fixed root joint is not an input
        for angle in (-40, 0, 33, 62):
            delta = model.deltas(hinge_pose(self.rig, angle))[0]
            self.assertAlmostEqual(delta[3], target(angle), delta=2e-3)
            self.assertAlmostEqual(delta[4], -target(angle) / 2, delta=2e-3)
        self.assertGreater(model.explained_variance, .99)
        self.assertEqual(model.coefficient_count, 13 * 4)  # 4 outputs move: 2 radii, rotation, exponent

    def test_bind_pose_reproduces_the_base_exactly(self):
        model = JointFunctionModel(self.rig.skeleton, self.library)
        local = model.evaluate(Pose.t_pose())
        base = self.library.base
        np.testing.assert_allclose(local.local_radii, base.local_radii, rtol=1e-6)
        np.testing.assert_allclose(local.local_centers, base.local_centers, atol=1e-7)
        np.testing.assert_allclose(local.shape_exponents, base.shape_exponents, rtol=1e-6)
        np.testing.assert_allclose(np.abs(local.local_rotations), np.abs(base.local_rotations), atol=1e-6)

    def test_matches_the_morph_key_at_a_trained_pose(self):
        model = JointFunctionModel(self.rig.skeleton, self.library)
        local = model.evaluate(self.library.keys[5].pose)
        expected = self.library.base.local_radii[0, 0] * np.exp(target(ANGLES[5]))
        self.assertAlmostEqual(float(local.local_radii[0, 0]), float(expected), delta=1e-3)

    def test_extrapolation_is_clamped_to_the_trained_range(self):
        model = JointFunctionModel(self.rig.skeleton, polynomial_library(self.rig, ANGLES[:5]))
        values = np.array([target(a) for a in ANGLES[:5]])
        span = values.max() - values.min()
        delta = model.deltas(hinge_pose(self.rig, 90))[0, 3]
        self.assertLessEqual(delta, max(values.max() + .1 * span, 0) + 1e-6)   # keys are float32
        self.assertGreaterEqual(delta, min(values.min() - .1 * span, 0) - 1e-6)

    def test_needs_enough_fitted_keys(self):
        with self.assertRaises(ValueError):
            JointFunctionModel(self.rig.skeleton, polynomial_library(self.rig, ANGLES[:3]))


class SurfaceErrorTest(unittest.TestCase):
    def test_row_wise_distance_matches_the_superquadric_sdf(self):
        rng = np.random.default_rng(3)
        radii = rng.uniform(.05, .4, (40, 3))
        eps = rng.uniform(.2, 1.9, (40, 2))
        points = rng.normal(0, .3, (40, 3))
        for i in range(40):
            expected = signed_distance_batch(np.zeros((1, 3)), radii[i:i + 1], [[0, 0, 0, 1]],
                                             eps[i:i + 1], points[i:i + 1])[0, 0]
            got = pair_signed_distance(points[i:i + 1], radii[i:i + 1], eps[i:i + 1])[0]
            self.assertAlmostEqual(float(got), float(expected), places=5)

    def test_meter_equals_the_full_union_distance(self):
        rig = make_rig()
        library = make_library(rig)
        meter = SurfaceErrorMeter(rig, library.base)
        pose = hinge_pose(rig, 35)
        error = meter.measure(pose, library.base)
        vertices = meter.posed_vertices(pose)
        c, r, q = BoneEllipsoidMapper(rig.skeleton).local_to_world_np(library.base, pose)
        exact = np.abs(signed_distance_batch(c, r, q, library.base.shape_exponents, vertices).min(0))
        self.assertAlmostEqual(error.mean_absolute, float(exact.mean()), places=5)
        self.assertAlmostEqual(error.mean_percent, float(exact.mean()) / meter.height * 100, places=4)
        both = meter.measure_many(pose, {"a": library.base, "b": library.base})
        self.assertEqual(both["a"], both["b"])


class RigWorldTransformTest(unittest.TestCase):
    def test_matches_the_mapper_with_blended_attachments(self):
        rig = make_rig()
        rig.skeleton.bones[0].joint_dof.set_axis(1, True, -40, 40)
        layout = BoneLocalEllipsoids(
            local_centers=np.array([[.25, .02, 0], [.05, 0, .01]], np.float32),
            local_radii=np.array([[.15, .05, .07], [.1, .1, .1]], np.float32),
            local_rotations=np.array([[0, 0, .2, .98], [.1, 0, 0, .995]], np.float32),
            bone_assignments=np.array([1, 0], np.int32),
            attachment_joints=np.array([[1, 0], [0, 1]], np.int32),
            attachment_weights=np.array([[.6, .4], [1, 0]], np.float32),
            primitive_type="superquadric", shape_exponents=np.ones((2, 2), np.float32))
        edit = JointPoseEdits(rig.skeleton, Pose.t_pose())
        edit.set_axis(1, 2, 50)
        edit.set_axis(0, 1, -25)
        pose = edit.apply()
        transform = RigWorldTransform(rig.skeleton, layout)
        got = transform.apply(layout, transform.pose_map(pose))
        expected = BoneEllipsoidMapper(rig.skeleton).local_to_world_np(layout, pose)
        np.testing.assert_allclose(got[0], expected[0], atol=1e-6)
        np.testing.assert_allclose(got[1], expected[1])
        sign = np.sign(np.sum(got[2] * expected[2], axis=1, keepdims=True))
        np.testing.assert_allclose(got[2] * sign, expected[2], atol=1e-6)


class AnimationTrackTest(unittest.TestCase):
    def test_random_poses_stay_inside_the_joint_limits(self):
        rig = make_rig()
        poses = random_multi_axis_poses(rig.skeleton, 40, seed=4, coverage=.8)
        angles = np.array([joint_pose_angles(rig.skeleton, pose)[1] for pose in poses])
        self.assertTrue(np.all(angles[:, 2] >= -60 * .8 - 1e-6) and np.all(angles[:, 2] <= 90 * .8 + 1e-6))
        np.testing.assert_allclose(angles[:, :2], 0, atol=1e-6)     # disabled axes stay put
        self.assertTrue(all(0 not in pose.bone_locals for pose in poses))   # root excluded
        self.assertGreater(np.ptp(angles[:, 2]), 60)

    def test_track_loops_and_hits_its_keys(self):
        rig = make_rig()
        keys = [Pose.t_pose(), hinge_pose(rig, 40), hinge_pose(rig, -30)]
        track = AnimationTrack("t", keys, keys_per_second=2.0)
        self.assertEqual(track.duration, 1.0)
        self.assertAlmostEqual(track.frame_at(.25), .5)
        self.assertAlmostEqual(track.frame_at(1.25), .5)
        self.assertIs(track.pose_at(rig.skeleton, .5), keys[1])
        middle = joint_pose_angles(rig.skeleton, track.pose_at(rig.skeleton, .25))[1, 2]
        self.assertAlmostEqual(middle, 20, delta=.5)

    def test_random_motion_starts_and_ends_at_the_bind_pose(self):
        track = random_motion(make_rig().skeleton, keys=5, seed=2)
        self.assertEqual(len(track.poses), 7)
        self.assertFalse(track.poses[0].bone_locals)
        self.assertFalse(track.poses[-1].bone_locals)


class AnimationPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.rig = make_rig()
        self.panel = RigModePanel(pose_dir=ROOT / "tools" / "__no_pose_library__")
        self.panel.setChecked(True)
        self.panel.set_rigged_mesh(self.rig)
        self.states, self.frames = [], []
        self.panel.animationStateChanged.connect(self.states.append)
        self.panel.animationPoseChanged.connect(lambda pose, dt, t: self.frames.append((pose, dt, t)))

    def tearDown(self):
        self.panel.stop_animation(notify=False)
        self.panel.deleteLater()
        self.app.processEvents()

    def test_random_motion_is_the_default_clip(self):
        self.assertEqual(self.panel._cmb_clip.itemData(0), "random")
        self.assertEqual(self.panel._cmb_clip.findData("mesh"), 1)     # two-frame mesh animation
        self.assertGreater(self.panel.animation_track.duration, 0)

    def test_play_replaces_the_timeline_pose_and_stop_restores_it(self):
        timeline = self.panel.current_pose
        self.assertTrue(self.panel.play_animation())
        self.assertTrue(self.panel.animation_playing)
        self.assertEqual(self.states, [True])
        self.assertIs(self.panel.current_pose, self.frames[-1][0])
        self.panel._anim_last -= .2
        self.panel._on_animation_tick()
        self.assertGreater(self.frames[-1][2], .15)
        self.panel._anim_last -= 5      # a stalled frame advances at most 0.25 s
        self.panel._on_animation_tick()
        self.assertLessEqual(self.frames[-1][1], .25)
        self.panel.stop_animation()
        self.assertEqual(self.states, [True, False])
        self.assertFalse(self.panel.animation_active)
        self.assertEqual(self.panel.current_pose.name, timeline.name)

    def test_scrubbing_shows_a_paused_animation_pose(self):
        self.panel._cmb_clip.setCurrentIndex(self.panel._cmb_clip.findData("mesh"))
        self.panel._slider_anim.setValue(1000)
        self.assertTrue(self.panel.animation_active)
        self.assertFalse(self.panel.animation_playing)
        self.assertEqual(self.states, [True])

    def test_fitting_and_timeline_changes_stop_the_animation(self):
        self.panel.play_animation()
        self.panel.set_auto_pipeline_running(True)
        self.assertFalse(self.panel.animation_active)
        self.assertFalse(self.panel._btn_play.isEnabled())
        self.panel.set_auto_pipeline_running(False)
        self.panel.play_animation()
        self.panel.set_joint_rotation(1, 2, 10)
        self.assertFalse(self.panel.animation_active)

    def test_model_and_speed_are_persisted_settings(self):
        keys = [key for key, _widget, _kind in self.panel.pose_tool_setting_specs()]
        self.assertIn("morph_model", keys)
        self.assertIn("animation_speed", keys)
        self.assertEqual(self.panel.morph_model, "keys")
        self.panel._cmb_morph_model.setCurrentIndex(1)
        self.assertEqual(self.panel.morph_model, "function")


class LivePreviewIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.rig = make_rig()
        self.library = polynomial_library(self.rig)
        self.panel = RigModePanel(pose_dir=ROOT / "tools" / "__no_pose_library__")
        self.panel.setChecked(True)
        self.panel.set_rigged_mesh(self.rig)
        self.host = WindowHost(self.panel, self.library)
        self.host._morph_error = MorphErrorPanel()
        self.host._joint_function = None
        self.host._error_meter = None
        self.host._rig_transform = None

    def tearDown(self):
        self.panel.stop_animation(notify=False)
        self.host.deleteLater()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_function_model_drives_the_preview_and_all_errors_are_measured(self):
        self.panel._cmb_morph_model.setCurrentIndex(self.panel._cmb_morph_model.findData("function"))
        pose = hinge_pose(self.rig, 33)
        MainWindow._render_pose_morph(self.host, pose, playback_time=.5)
        radii = self.host.draws[-1][1]
        expected = self.library.base.local_radii[0, 0] * np.exp(target(33))
        self.assertAlmostEqual(float(radii[0, 0]), float(expected), delta=2e-3)
        errors = self.host._morph_error.last_errors
        self.assertEqual(set(errors), {"none", "keys", "function"})
        self.assertTrue(all(np.isfinite(e.mean_percent) for e in errors.values()))
        self.assertIn("Joint functions:", self.panel._lbl_function.text())
        self.assertEqual(self.host._morph_error._counts["function"], 1)

    def test_too_few_keys_fall_back_to_the_key_blend(self):
        library = make_library(self.rig)
        host = WindowHost(self.panel, library)
        host._morph_error = MorphErrorPanel()
        host._joint_function = host._error_meter = host._rig_transform = None
        self.panel._cmb_morph_model.setCurrentIndex(self.panel._cmb_morph_model.findData("function"))
        MainWindow._render_pose_morph(host, library.keys[0].pose)
        self.assertEqual(set(host._morph_error.last_errors), {"none", "keys"})
        self.assertIn("unavailable", self.panel._lbl_function.text())
        host.deleteLater()

    def test_animation_frames_render_without_sdf_work(self):
        self.host._apply_rotation = lambda v: v
        self.host._show_skeleton_for_pose = mock.Mock()
        self.host._on_compute_all.reset_mock()
        MainWindow._on_animation_pose(self.host, hinge_pose(self.rig, 20), .03, 1.0)
        self.host._viewer.show_mesh.assert_called_once()
        self.assertTrue(self.host.draws)
        self.host._on_compute_all.assert_not_called()
        self.assertIn("keys", self.host._morph_error.last_errors)


if __name__ == "__main__":
    unittest.main()
