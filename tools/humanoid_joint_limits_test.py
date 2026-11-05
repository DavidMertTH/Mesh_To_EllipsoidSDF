"""Authored human joint limits, real bind axes and equivalent-rig contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanoid_joint_limits import PROFILE_PATH, apply_humanoid_joint_limits
from joint_dof import JointDof, joint_rotation_frames
from main_window import MainWindow
from mesh_io import NormalizationTransform
from rig_loader import load_rigged_mesh
from skeleton import Bone, Skeleton, mat4_compose


def _fresh(skeleton):
    bones = copy.deepcopy(skeleton.bones)
    for bone in bones:
        bone.joint_dof = JointDof()
    return Skeleton(bones)


def _rotation(axis, angle):
    vector = np.eye(3)[axis]
    half = np.radians(angle) / 2
    return mat4_compose(np.zeros(3), np.r_[vector * np.sin(half), np.cos(half)], np.ones(3))


class HumanoidJointLimitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")
        cls.skeleton = cls.rig.skeleton
        cls.keys = {bone.name.rsplit(":", 1)[-1]: bone.index for bone in cls.skeleton.bones}
        cls.world = cls.skeleton.compute_world_transforms(None)

    def test_profile_covers_every_reference_bone_with_valid_ranges(self):
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(profile["format"], "ellipsdf-joint-limits")
        self.assertEqual(len(profile["bones"]), 65)
        self.assertEqual({entry["name"] for entry in profile["bones"]}, set(self.keys))
        self.assertEqual(self.skeleton.joint_limit_profile, "T-Pose humanoid")
        for entry in profile["bones"]:
            frame = np.asarray(entry["axes"])
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=2e-6)
            dof = self.skeleton.bones[self.keys[entry["name"]]].joint_dof
            np.testing.assert_array_equal(dof.enabled, entry["enabled"])
            np.testing.assert_allclose(dof.minimum, entry["minimum"])
            np.testing.assert_allclose(dof.maximum, entry["maximum"])
            self.assertTrue(np.all(dof.minimum <= 0) and np.all(dof.maximum >= 0))
            self.assertTrue(np.all(dof.minimum >= -180) and np.all(dof.maximum <= 180))

    def test_hinge_directions_bend_elbows_forward_and_knees_backward(self):
        frames = joint_rotation_frames(self.world)
        for name, child, axis, angle, expected_forward in (
            ("LeftForeArm", "LeftHand", 2, 90, True),
            ("RightForeArm", "RightHand", 2, -90, True),
            ("LeftLeg", "LeftFoot", 0, -90, False),
            ("RightLeg", "RightFoot", 0, -90, False),
        ):
            i, j = self.keys[name], self.keys[child]
            dof = self.skeleton.bones[i].joint_dof
            self.assertTrue(dof.enabled[axis])
            self.assertLessEqual(dof.minimum[axis], angle)
            self.assertGreaterEqual(dof.maximum[axis], angle)
            vector = frames[i].T @ (self.world[j, :3, 3] - self.world[i, :3, 3])
            bent = frames[i] @ (_rotation(axis, angle)[:3, :3] @ vector)
            self.assertGreater(bent[2], 0.15) if expected_forward else self.assertLess(bent[2], -0.15)
        for name in ("LeftLeg", "RightLeg"):
            self.assertEqual(self.skeleton.bones[self.keys[name]].joint_dof.count, 1)

    def test_terminals_are_locked_and_fingers_have_suitable_dof(self):
        locked = [bone for bone in self.skeleton.bones if bone.joint_dof.count == 0]
        self.assertEqual(len(locked), 13)
        for side in ("Left", "Right"):
            for digit in ("Index", "Middle", "Ring", "Pinky"):
                for segment, count in ((1, 2), (2, 1), (3, 1), (4, 0)):
                    key = f"{side}Hand{digit}{segment}"
                    self.assertEqual(self.skeleton.bones[self.keys[key]].joint_dof.count, count)
            self.assertEqual(self.skeleton.bones[self.keys[side + "HandThumb1"]].joint_dof.count, 3)
        self.assertEqual(self.skeleton.bones[self.keys["Hips"]].joint_dof.count, 3)

    def test_dancing_and_fight_idle_load_the_same_defaults(self):
        for file in ("Dancing.fbx", "Fight Idle.fbx"):
            skeleton = load_rigged_mesh(ROOT / "meshes" / file).skeleton
            self.assertEqual(skeleton.joint_limit_profile, "T-Pose humanoid")
            for expected, actual in zip(self.skeleton.bones, skeleton.bones):
                np.testing.assert_allclose(actual.joint_dof.minimum, expected.joint_dof.minimum)
                np.testing.assert_allclose(actual.joint_dof.maximum, expected.joint_dof.maximum)

    def test_namespaces_and_global_rotation_scale_translation_are_equivalent(self):
        skeleton = _fresh(self.skeleton)
        for bone in skeleton.bones:
            bone.name = "new_export:" + bone.name.rsplit(":", 1)[-1]
        transform = _rotation(2, 73)
        transform[:3, :3] *= 4
        transform[:3, 3] = [15, -8, 21]
        self.assertEqual(apply_humanoid_joint_limits(skeleton, transform @ self.world), 65)
        for expected, actual in zip(self.skeleton.bones, skeleton.bones):
            np.testing.assert_array_equal(actual.joint_dof.enabled, expected.joint_dof.enabled)
            np.testing.assert_allclose(actual.joint_dof.minimum, expected.joint_dof.minimum)
            np.testing.assert_allclose(actual.joint_dof.maximum, expected.joint_dof.maximum)

    def test_equivalent_joint_axes_are_permuted_and_signed(self):
        skeleton = _fresh(self.skeleton)
        target = self.world.copy()
        elbow = self.keys["LeftForeArm"]
        # New Y is old Z: transfer the hinge to the target Y axis.
        permutation = np.eye(3)[:, [1, 2, 0]]
        target[elbow, :3, :3] = self.world[elbow, :3, :3] @ permutation
        self.assertEqual(apply_humanoid_joint_limits(skeleton, target), 65)
        dof = skeleton.bones[elbow].joint_dof
        np.testing.assert_array_equal(dof.enabled, [True, True, False])
        self.assertEqual((dof.minimum[1], dof.maximum[1]), (0, 145))
        skeleton = _fresh(self.skeleton)
        target[elbow, :3, :3] = self.world[elbow, :3, :3] @ np.diag([1, -1, -1])
        apply_humanoid_joint_limits(skeleton, target)
        dof = skeleton.bones[elbow].joint_dof
        self.assertEqual((dof.minimum[2], dof.maximum[2]), (-145, 0))

    def test_manual_edits_unknown_rigs_and_uncertain_axes_are_preserved(self):
        skeleton = _fresh(self.skeleton)
        elbow = self.keys["LeftForeArm"]
        skeleton.bones[elbow].joint_dof.set_axis(2, True, -5, 120)
        self.assertEqual(apply_humanoid_joint_limits(skeleton), 64)
        self.assertEqual(skeleton.bones[elbow].joint_dof.minimum[2], -5)
        skeleton = Skeleton([Bone("Root", 0), Bone("Elbow", 1, 0)])
        self.assertEqual(apply_humanoid_joint_limits(skeleton), 0)
        self.assertTrue(all(bone.joint_dof.count == 3 for bone in skeleton.bones))
        skeleton = _fresh(self.skeleton)
        target = self.world.copy()
        target[elbow, :3, :3] = self.world[elbow, :3, :3] @ _rotation(1, 45)[:3, :3]
        self.assertEqual(apply_humanoid_joint_limits(skeleton, target), 64)
        self.assertEqual(skeleton.bones[elbow].joint_dof.count, 3)

    def test_allowances_do_not_change_imported_animation_transforms(self):
        skeleton = _fresh(self.skeleton)
        pose = self.rig.poses[-1]
        before = skeleton.compute_skin_matrices(pose).copy()
        apply_humanoid_joint_limits(skeleton)
        np.testing.assert_allclose(skeleton.compute_skin_matrices(pose), before)

    def test_unity_overlay_uses_bind_axes_even_for_a_posed_snapshot(self):
        current = self.world.copy()
        elbow = self.keys["LeftForeArm"]
        current[elbow, :3, :3] = self.world[elbow, :3, :3] @ _rotation(1, 45)[:3, :3]
        rig = {"bones": [dict(name=bone.name, parent=bone.parent_index,
                               matrix=bind.tolist(), currentMatrix=posed.tolist())
                         for bone, bind, posed in zip(self.skeleton.bones, self.world, current)]}
        viewer = mock.Mock()
        owner = SimpleNamespace(_viewer=viewer, _status=mock.Mock())
        transform = NormalizationTransform(center=np.zeros(3), scale=1.0)
        MainWindow._api_show_rig_bones(owner, rig, transform)
        viewer.remove_skeleton.assert_not_called()
        dofs = viewer.show_bones.call_args.kwargs["joint_dofs"]
        self.assertEqual(dofs[elbow].count, 2)
        self.assertEqual(dofs[elbow].maximum[2], 145)

    def test_legacy_unity_overlay_without_bind_matrices_stays_visible(self):
        rig = {"bones": [dict(name=bone.name, parent=bone.parent_index,
                               matrix=None, currentMatrix=world.tolist())
                         for bone, world in zip(self.skeleton.bones, self.world)]}
        viewer = mock.Mock()
        owner = SimpleNamespace(_viewer=viewer, _status=mock.Mock())
        transform = NormalizationTransform(center=np.zeros(3), scale=1.0)
        MainWindow._api_show_rig_bones(owner, rig, transform)
        viewer.remove_skeleton.assert_not_called()
        self.assertTrue(all(dof.count == 3 for dof in
                            viewer.show_bones.call_args.kwargs["joint_dofs"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
