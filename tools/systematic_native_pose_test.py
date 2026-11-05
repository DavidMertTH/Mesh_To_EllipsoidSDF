"""Numerical contracts for model-independent native joint-range sweeps."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint_dof import JointDof, joint_rotation_frames
from joint_pose import _rotation
from rig_loader import load_rigged_mesh
from skeleton import Bone, Skeleton
from systematic_pose_fitter import (
    SystematicPoseSettings, angle_samples, generate_systematic_poses,
    systematic_pose_signature,
)


def _dof(enabled=(True, False, False), minimum=(-30, 0, 0), maximum=(30, 0, 0)):
    return JointDof(np.asarray(enabled, dtype=bool), np.asarray(minimum, dtype=float),
                    np.asarray(maximum, dtype=float))


def _skeleton(*names):
    bones = [Bone("Root", 0, joint_dof=_dof((False,) * 3, (0,) * 3, (0,) * 3))]
    bones += [Bone(name, i, 0, joint_dof=_dof()) for i, name in enumerate(names, 1)]
    return Skeleton(bones)


def _target_key(targets):
    return tuple((t.joint, t.axis, round((t.angle + 180) % 360 - 180, 10)) for t in targets)


class SystematicSamplingTest(unittest.TestCase):
    def test_steps_sweep_each_direction_outward_and_keep_exact_endpoints(self):
        self.assertEqual(angle_samples(-65, 73.5, 30, 1), [-30, -60, -65, 30, 60, 73.5])
        self.assertEqual(angle_samples(-37, 63, 30, 0.5), [-18.5, 30, 31.5])
        self.assertEqual(angle_samples(-60, 60, 30, 1), [-30, -60, 30, 60])
        # The shorter direction first keeps the one jump across neutral small.
        self.assertEqual(angle_samples(-90, 40, 30, 1), [30, 40, -30, -60, -90])
        self.assertEqual(angle_samples(0, 0, 30, 1), [])

    def test_every_sample_is_one_step_from_neutral_or_an_earlier_sample(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            low, high = np.sort(rng.uniform(-180, 180, 2))
            step, coverage = rng.uniform(5, 90), rng.uniform(0.05, 1)
            samples = angle_samples(low, high, step, coverage)
            visited = [float(np.clip(0, low, high))]
            for sample in samples:
                # One-sided ranges start at their nearest-to-zero endpoint.
                self.assertLessEqual(min(abs(sample - v) for v in visited), step + 1e-9,
                                     (low, high, step, coverage, samples))
                visited.append(sample)

    def test_one_sided_ranges_stay_in_the_authored_interval(self):
        self.assertEqual(angle_samples(10, 70, 30, 0.5), [10, 30, 40])
        self.assertEqual(angle_samples(-70, -10, 30, 0.5), [-10, -30, -40])
        self.assertEqual(angle_samples(100, 105, 30, 0.5), [100, 102.5])

    def test_endpoint_just_below_a_step_is_not_rounded_outside_the_range(self):
        high = 30 - 1e-7
        self.assertEqual(angle_samples(0, high, 30, 1), [high])
        low = -30 + 1e-7
        self.assertEqual(angle_samples(low, 0, 30, 1), [low])
        for lo, hi in ((1.12345678, 7.98765432), (-7.98765432, -1.12345678)):
            values = angle_samples(lo, hi, 5, 1)
            self.assertIn(lo, values)
            self.assertIn(hi, values)
            self.assertTrue(all(lo <= value <= hi for value in values))

    def test_sampling_and_settings_reject_invalid_values(self):
        for step in (0, 4.9, 90.1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                angle_samples(-30, 30, step, 1)
            with self.assertRaises(ValueError):
                SystematicPoseSettings(angle_step=step)
        for coverage in (0, 0.049, 1.001, np.inf, np.nan):
            with self.assertRaises(ValueError):
                angle_samples(-30, 30, 30, coverage)
        for limits in ((30, -30), (-181, 30), (-30, 181), (np.nan, 30)):
            with self.assertRaises(ValueError):
                angle_samples(*limits, 30, 1)
        for passes in (0, 13, 1.5, True):
            with self.assertRaises(ValueError):
                SystematicPoseSettings(passes=passes)
        self.assertEqual(SystematicPoseSettings(passes=np.int64(2)).passes, 2)


class SystematicPlanTest(unittest.TestCase):
    def assertPlanRanges(self, skeleton, plan):
        self.assertEqual(len(plan.poses), len(plan.targets))
        self.assertEqual(plan.poses[0].bone_locals, {})
        self.assertEqual(plan.targets[0], ())
        for pose, targets in zip(plan.poses[1:], plan.targets[1:]):
            self.assertEqual(len({t.joint for t in targets}), len(targets))
            self.assertEqual(set(pose.bone_locals), {t.joint for t in targets})
            for target in targets:
                dof = skeleton.bones[target.joint].joint_dof
                self.assertTrue(dof.enabled[target.axis])
                self.assertGreaterEqual(target.angle, dof.minimum[target.axis])
                self.assertLessEqual(target.angle, dof.maximum[target.axis])
        keys = [_target_key(targets) for targets in plan.targets]
        self.assertEqual(len(keys), len(set(keys)))

    def test_first_pass_steps_every_lane_through_its_angles_in_order(self):
        skeleton = _skeleton("ball", "hinge", "short")
        skeleton.bones[1].joint_dof = _dof((True, True, False), (-60, -40, 0), (60, 90, 0))
        skeleton.bones[3].joint_dof = _dof(minimum=(-20, 0, 0), maximum=(10, 0, 0))
        plan = generate_systematic_poses(skeleton, SystematicPoseSettings(coverage=1, passes=1))
        lanes = {index: [(axis, angle) for axis in range(3) if bone.joint_dof.enabled[axis]
                         for angle in angle_samples(bone.joint_dof.minimum[axis],
                                                    bone.joint_dof.maximum[axis], 30, 1)]
                 for index, bone in enumerate(skeleton.bones) if index}
        self.assertEqual(len(plan.poses), 1 + max(map(len, lanes.values())))
        for step, targets in enumerate(plan.targets[1:]):
            expected = {joint: lane[step] for joint, lane in lanes.items() if step < len(lane)}
            self.assertEqual({t.joint: (t.axis, t.angle) for t in targets}, expected)

    def test_parallel_plan_samples_every_channel_without_combining_ball_joint_axes(self):
        skeleton = _skeleton("ball", "hinge")
        skeleton.bones[1].joint_dof = _dof((True, True, False), (-45, 0, -90), (45, 60, 90))
        plan = generate_systematic_poses(skeleton, SystematicPoseSettings(coverage=1, passes=1))
        self.assertEqual((plan.joint_count, plan.channel_count), (2, 3))
        self.assertEqual(len(plan.poses), 7)
        self.assertTrue(any(len(targets) == 2 for targets in plan.targets))
        self.assertPlanRanges(skeleton, plan)
        for index, bone in enumerate(skeleton.bones):
            for axis in range(3):
                if not bone.joint_dof.enabled[axis]:
                    continue
                expected = angle_samples(bone.joint_dof.minimum[axis], bone.joint_dof.maximum[axis], 30, 1)
                actual = {t.angle for targets in plan.targets for t in targets
                          if t.joint == index and t.axis == axis}
                self.assertEqual(actual, set(expected))

    def test_passes_recombine_independent_small_lanes_and_remove_duplicates(self):
        skeleton = _skeleton("Wing1", "Tail5", "Coil2")
        one = generate_systematic_poses(skeleton, SystematicPoseSettings(coverage=1, passes=1))
        many = generate_systematic_poses(skeleton, SystematicPoseSettings(coverage=1, passes=12))
        self.assertGreater(len(many.poses), len(one.poses))
        self.assertLessEqual(len(many.poses), 1 + 2 ** 3)
        self.assertPlanRanges(skeleton, many)
        matrices = [np.round(skeleton.compute_world_transforms(pose), 9).tobytes()
                    for pose in many.poses]
        self.assertEqual(len(matrices), len(set(matrices)))

    def test_half_turn_signs_and_repeated_single_lane_passes_are_deduplicated(self):
        skeleton = _skeleton("mechanism")
        skeleton.bones[1].joint_dof = _dof(minimum=(-180, 0, 0), maximum=(180, 0, 0))
        plan = generate_systematic_poses(skeleton, SystematicPoseSettings(angle_step=90, coverage=1, passes=12))
        self.assertEqual(len(plan.poses), 4)  # bind, -90, +90, one half turn
        self.assertPlanRanges(skeleton, plan)

    def test_root_option_and_locked_channels_work_on_arbitrary_rigs(self):
        skeleton = _skeleton("only-zero-width", "disabled")
        skeleton.bones[0].joint_dof = _dof()
        skeleton.bones[1].joint_dof = _dof(minimum=(15, 0, 0), maximum=(15, 0, 0))
        skeleton.bones[2].joint_dof = _dof((False,) * 3)
        excluded = generate_systematic_poses(skeleton)
        self.assertEqual((excluded.joint_count, excluded.channel_count, len(excluded.poses)), (0, 0, 1))
        included = generate_systematic_poses(skeleton, SystematicPoseSettings(coverage=1, include_roots=True))
        self.assertEqual((included.joint_count, included.channel_count, len(included.poses)), (1, 1, 3))
        self.assertTrue(all(t.joint == 0 for targets in included.targets for t in targets))
        self.assertPlanRanges(skeleton, included)
        empty = generate_systematic_poses(Skeleton([]))
        self.assertEqual(len(empty.poses), 1)

    def test_names_and_namespaces_do_not_filter_nonhuman_joints(self):
        skeleton = _skeleton("asset:Feeler.008", "asset:ピストン", "123")
        plan = generate_systematic_poses(skeleton)
        self.assertEqual((plan.joint_count, plan.channel_count), (3, 3))
        self.assertEqual({t.joint for targets in plan.targets for t in targets}, {1, 2, 3})
        self.assertPlanRanges(skeleton, plan)
        # Non-Latin names retain distinct lane hashes instead of all collapsing
        # to an empty canonical name and repeating the same two packed poses.
        non_latin = _skeleton("ピストン", "触手")
        recombined = generate_systematic_poses(non_latin, SystematicPoseSettings(coverage=1, passes=12))
        self.assertEqual(len(recombined.poses), 5)

    def test_regeneration_is_deterministic_and_does_not_mutate_bind_or_ranges(self):
        skeleton = _skeleton("asset:Left.Feeler", "asset:Right.Feeler", "hinge")
        before = copy.deepcopy(skeleton.bones)
        first, second = generate_systematic_poses(skeleton), generate_systematic_poses(skeleton)
        self.assertEqual(first.targets, second.targets)
        self.assertEqual([p.name for p in first.poses], [p.name for p in second.poses])
        for a, b in zip(first.poses, second.poses):
            np.testing.assert_array_equal(skeleton.compute_world_transforms(a), skeleton.compute_world_transforms(b))
        for old, bone in zip(before, skeleton.bones):
            np.testing.assert_array_equal(old.local_bind_transform, bone.local_bind_transform)
            np.testing.assert_array_equal(old.joint_dof.minimum, bone.joint_dof.minimum)
            np.testing.assert_array_equal(old.joint_dof.maximum, bone.joint_dof.maximum)
            np.testing.assert_array_equal(old.joint_dof.enabled, bone.joint_dof.enabled)

    def test_cache_signature_tracks_settings_ranges_names_hierarchy_and_bind(self):
        original = _skeleton("joint")
        settings = SystematicPoseSettings()
        signature = systematic_pose_signature(original, settings)
        self.assertEqual(signature, systematic_pose_signature(copy.deepcopy(original), settings))
        changes = (
            lambda s: setattr(s.bones[1], "name", "different"),
            lambda s: setattr(s.bones[1], "parent_index", -1),
            lambda s: s.bones[1].joint_dof.set_axis(0, True, -20, 30),
            lambda s: s.bones[1].joint_dof.set_axis(0, False, -30, 30),
            lambda s: s.bones[1].local_bind_transform.__setitem__((1, 3), 2),
        )
        for change in changes:
            skeleton = copy.deepcopy(original)
            change(skeleton)
            self.assertNotEqual(signature, systematic_pose_signature(skeleton, settings))
        self.assertNotEqual(signature, systematic_pose_signature(original, SystematicPoseSettings(passes=2)))

    def test_nonuniform_scale_shear_reflection_and_translation_are_preserved(self):
        skeleton = _skeleton("affine")
        bind = np.eye(4)
        bind[:3, :3] = _rotation([17, -29, 42]) @ np.array([
            [-2.0, 0.3, -0.1], [0.0, 1.3, 0.4], [0.0, 0.0, 0.7]])
        bind[:3, 3] = [0.4, 1.2, -0.8]
        skeleton.bones[1].local_bind_transform = bind
        skeleton.bones[1].joint_dof = _dof((True, True, True), (-60, -45, -30), (90, 75, 60))
        plan = generate_systematic_poses(skeleton, SystematicPoseSettings(coverage=1))
        reference = joint_rotation_frames(np.array([bind]))[0]
        for pose, targets in zip(plan.poses[1:], plan.targets[1:]):
            local = pose.bone_locals[1]
            np.testing.assert_array_equal(local[:3, 3], bind[:3, 3])
            np.testing.assert_array_equal(local[3], bind[3])
            np.testing.assert_allclose(local[:3, :3].T @ local[:3, :3],
                                       bind[:3, :3].T @ bind[:3, :3], atol=2e-6)
            self.assertAlmostEqual(np.linalg.det(local[:3, :3]), np.linalg.det(bind[:3, :3]), places=5)
            target = targets[0]
            angles = np.zeros(3)
            angles[target.axis] = target.angle
            current = joint_rotation_frames(np.array([local]))[0]
            np.testing.assert_allclose(reference.T @ current, _rotation(angles), atol=2e-6)

    def test_generated_rotation_propagates_to_descendant_positions(self):
        skeleton = _skeleton("pivot", "tip")
        skeleton.bones[1].joint_dof = _dof((False, False, True), (0, 0, 0), (0, 0, 90))
        skeleton.bones[2].parent_index = 1
        skeleton.bones[2].joint_dof = _dof((False,) * 3)
        skeleton.bones[2].local_bind_transform[:3, 3] = [1, 0, 0]
        plan = generate_systematic_poses(skeleton, SystematicPoseSettings(coverage=1))
        pose = next(p for p, targets in zip(plan.poses, plan.targets) if targets and targets[0].angle == 90)
        world = skeleton.compute_world_transforms(pose)
        np.testing.assert_allclose(world[2, :3, 3], [0, 1, 0], atol=1e-6)


class RealHumanoidSystematicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")
        cls.settings = SystematicPoseSettings(angle_step=30, coverage=0.9, passes=4)
        cls.plan = generate_systematic_poses(cls.rig.skeleton, cls.settings)

    def test_real_tpose_covers_all_editable_channels_and_locks_terminals(self):
        skeleton, plan = self.rig.skeleton, self.plan
        self.assertEqual(plan.joint_count, 51)
        self.assertEqual(plan.channel_count, sum(b.joint_dof.count for b in skeleton.bones[1:]))
        self.assertGreater(len(plan.poses), 50)
        SystematicPlanTest.assertPlanRanges(self, skeleton, plan)
        for index, bone in enumerate(skeleton.bones):
            for axis in range(3):
                actual = {t.angle for targets in plan.targets for t in targets
                          if t.joint == index and t.axis == axis}
                expected = angle_samples(bone.joint_dof.minimum[axis], bone.joint_dof.maximum[axis],
                                         self.settings.angle_step, self.settings.coverage)
                if index == 0 or not bone.joint_dof.enabled[axis]:
                    expected = []
                # +180/-180 are the same rotation; default coverage avoids them.
                self.assertEqual(actual, set(expected), (bone.name, axis))

    def test_equivalent_mixamo_models_share_targets_and_skin_vertices_move(self):
        for name in ("Dancing.fbx", "Fight Idle.fbx"):
            rig = load_rigged_mesh(ROOT / "meshes" / name)
            plan = generate_systematic_poses(rig.skeleton, self.settings)
            self.assertEqual(plan.targets, self.plan.targets)
            self.assertEqual(len(plan.poses), len(self.plan.poses))
        skeleton = self.rig.skeleton
        base = skeleton.compute_skin_matrices(self.plan.poses[0])
        moving = skeleton.compute_skin_matrices(self.plan.poses[1])
        self.assertGreater(np.max(np.abs(moving - base)), 0.1)
        self.assertTrue(np.all(np.isfinite(moving)))
        # A real bound bone-local point follows the generated pose rather than
        # merely changing support labels/angles.
        bone = next(t.joint for t in self.plan.targets[1] if
                    skeleton.bones[t.joint].name.endswith("LeftForeArm"))
        tip = skeleton.bone_index("mixamorig1:LeftHand")
        bind_world = skeleton.compute_world_transforms(None)
        point = np.r_[bind_world[tip, :3, 3], 1]
        self.assertGreater(np.linalg.norm((moving[bone] @ point - base[bone] @ point)[:3]), 0.02)


if __name__ == "__main__":
    unittest.main()
