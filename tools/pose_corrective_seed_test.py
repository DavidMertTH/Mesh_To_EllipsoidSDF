"""Corrective fits start from, and stay near, the most similar fitted pose."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids  # noqa: E402
from ellipsoid import best_device  # noqa: E402
from joint_pose import JointPoseEdits, _rotation  # noqa: E402
from pose_correctives import (  # noqa: E402
    PoseCorrectiveWorker,
    _gather_local_seed,
    _most_similar_fitted,
)
from skeleton import Bone, Pose, Skeleton  # noqa: E402
from tools.native_pose_ui_test import make_library, make_rig  # noqa: E402


def _chain(count):
    """root -> j1 -> j2 ...: neighbouring joints share a morph context."""
    bones = [Bone("root", 0)]
    for index in range(1, count):
        bone = Bone(f"j{index}", index, index - 1)
        bone.local_bind_transform[0, 3] = .2
        bones.append(bone)
    return Skeleton(bones)


def _pose(skeleton, **degrees):
    locals_ = {}
    for name, angle in degrees.items():
        index = skeleton.bone_index(name)
        local = skeleton.bones[index].local_bind_transform.copy()
        local[:3, :3] = _rotation([0, 0, angle])
        locals_[index] = local
    return Pose(" ".join(f"{k}={v}" for k, v in degrees.items()), locals_)


class MostSimilarFittedPoseTest(unittest.TestCase):
    def setUp(self):
        self.skeleton = _chain(6)
        self.bind = Pose.t_pose()

    def choose(self, pose, fitted, assignments):
        return _most_similar_fitted(self.skeleton, pose, fitted, np.asarray(assignments)).tolist()

    def test_sweep_continues_from_the_adjacent_fitted_angle(self):
        s = self.skeleton
        fitted = [self.bind, _pose(s, j2=30), _pose(s, j2=60)]
        self.assertEqual(self.choose(_pose(s, j2=90), fitted, [2]), [2])
        self.assertEqual(self.choose(_pose(s, j2=-30), fitted, [2]), [0])

    def test_each_primitive_uses_the_pose_closest_for_its_own_joints(self):
        # A packed pose moves distant joints together. No single fitted pose
        # matches both, but each primitive has an exact local match.
        s = self.skeleton
        fitted = [self.bind, _pose(s, j1=30), _pose(s, j4=30)]
        self.assertEqual(self.choose(_pose(s, j1=30, j4=30), fitted, [1, 4, 1]), [1, 2, 1])

    def test_parent_and_child_rotations_are_part_of_the_similarity(self):
        s = self.skeleton
        fitted = [self.bind, _pose(s, j2=35, j3=55), _pose(s, j2=40, j3=0)]
        current = _pose(s, j2=40, j3=55)
        # Counting its child joint, support 1 is closer even though joint 2
        # alone matches support 2 exactly.
        self.assertEqual(self.choose(current, fitted, [2]), [1])
        alone = _most_similar_fitted(s, current, fitted, np.array([2]), context_depth=0)
        self.assertEqual(alone.tolist(), [2])

    def test_ties_prefer_the_base_then_the_overall_closest_then_the_earliest_pose(self):
        s = self.skeleton
        fitted = [self.bind, _pose(s, j4=30), _pose(s, j4=60)]
        # Joint 1 never moved and ties everywhere. The runtime driver shows the
        # plain base for it, so training continues from the base as well.
        self.assertEqual(self.choose(_pose(s, j4=60), fitted, [1, 4]), [0, 2])
        fitted = [self.bind, _pose(s, j2=30, j4=30), _pose(s, j2=30, j4=60)]
        self.assertEqual(self.choose(_pose(s, j2=30, j4=60), fitted, [2]), [2])
        fitted = [self.bind, _pose(s, j2=30), _pose(s, j2=30)]
        self.assertEqual(self.choose(_pose(s, j2=30), fitted, [2]), [1])
        self.assertEqual(self.choose(self.bind, [self.bind, None, self.bind], [1]), [0])

    def test_invalid_assignments_fall_back_to_the_whole_pose(self):
        s = self.skeleton
        fitted = [self.bind, _pose(s, j1=30), _pose(s, j4=30, j5=30)]
        self.assertEqual(self.choose(_pose(s, j4=30, j5=30), fitted, [-1, 99]), [2, 2])


class SeedAnchorTest(unittest.TestCase):
    def test_seed_and_base_share_the_regularization_and_bound_the_step(self):
        library = make_library(make_rig())
        seed = library.corrected_bone_local(library.keys[0])
        worker = SimpleNamespace(_fit_kwargs={"primitive_shape": "superquadric"},
                                 _sdf_blowup_fraction=0.0, _base=library.base)
        sdf = SimpleNamespace(grid=np.zeros((2, 2, 2), np.float32), origin=np.zeros(3, np.float32),
                              dx=1.0, n=2, thickness=None)
        kwargs = PoseCorrectiveWorker._optimizer_kwargs(
            worker, sdf, seed, np.eye(3, dtype=np.float32)[None], np.zeros((1, 3), np.float32),
            np.array([[0, 0, 0, 1]], np.float32))
        for name, attribute in (("centers", "local_centers"), ("radii", "local_radii"),
                                ("rotations", "local_rotations")):
            np.testing.assert_array_equal(kwargs[f"initial_{name}"], getattr(seed, attribute))
            np.testing.assert_array_equal(kwargs[f"parameter_anchor_{name}"], getattr(seed, attribute))
        self.assertFalse(np.array_equal(kwargs["parameter_anchor_centers"], library.base.local_centers))
        self.assertGreater(kwargs["parameter_center_trust_radius_factor"], 0)
        # An equally strong pull toward the base stops drift along pose chains.
        for name, attribute in (("centers", "local_centers"), ("radii", "local_radii"),
                                ("rotations", "local_rotations")):
            np.testing.assert_array_equal(kwargs[f"parameter_neighbor_{name}"],
                                          getattr(library.base, attribute))
        for name in ("center", "radii", "rotation"):
            self.assertGreater(kwargs[f"parameter_neighbor_{name}_regularization"], 0)
            self.assertEqual(kwargs[f"parameter_neighbor_{name}_regularization"],
                             kwargs[f"parameter_{name}_regularization"])
        # Keys never leave the base's usual trust region.
        np.testing.assert_array_equal(kwargs["parameter_bound_centers"], library.base.local_centers)
        np.testing.assert_array_equal(kwargs["parameter_bound_radii"], library.base.local_radii)
        self.assertEqual(kwargs["parameter_bound_center_radius_factor"],
                         kwargs["parameter_center_trust_radius_factor"])
        self.assertEqual(kwargs["parameter_bound_radii_factor"],
                         kwargs["parameter_radii_trust_factor"])


def _hinge_poses(rig, *angles):
    poses = []
    for angle in angles:
        edit = JointPoseEdits(rig.skeleton, Pose.t_pose())
        edit.set_axis(1, 2, angle)
        poses.append(edit.apply())
    return poses


def _worker(rig, base, poses, **options):
    fit_kwargs = {"num_steps": 5, "report_every": 1, "sample_budget": 128,
                  "maintenance_every": 0, "superfit": False, "local_fit": False,
                  "lr_init": .01, "lr_final": .01, "primitive_shape": "superquadric",
                  "sq_eps_mode": "per_primitive", "sq_unlock_frac": 0}
    fit_kwargs.update(options.pop("fit_kwargs", {}))
    return PoseCorrectiveWorker(rigged_mesh=rig, mapper=BoneEllipsoidMapper(rig.skeleton),
                                base=base, poses=poses, grid_n=8, margin=.12,
                                device=best_device(), fit_kwargs=fit_kwargs, **options)


class SweepSeedingWorkerTest(unittest.TestCase):
    def test_real_sweep_starts_each_pose_from_the_previous_fitted_angle(self):
        rig = make_rig()
        base = make_library(rig).base
        poses = _hinge_poses(rig, -30, -60)
        starts, failures = [], []
        worker = _worker(rig, base, poses)
        worker.failed.connect(failures.append)
        worker.pose_target_visual.connect(lambda *args: starts.append(args))
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(len(starts), 2)
        mapper = worker._mapper
        np.testing.assert_allclose(starts[0][4], mapper.local_to_world_np(base, pose=poses[0])[0],
                                   atol=1e-6)
        first = worker.result.corrected_bone_local(worker.result.keys[0])
        self.assertGreater(np.max(np.abs(first.local_centers - base.local_centers)), 1e-6,
                           "the first pose must actually move its primitive")
        expected = mapper.local_to_world_np(first, pose=poses[1])
        for got, want in zip(starts[1][4:7], expected):
            np.testing.assert_allclose(np.abs(got), np.abs(want), atol=1e-5)
        np.testing.assert_allclose(starts[1][7], first.shape_exponents, atol=1e-5)

    def test_base_fitted_in_an_edited_pose_is_the_candidate_for_that_pose(self):
        rig = make_rig()
        base = make_library(rig).base
        edited, bent = _hinge_poses(rig, -60, -55)
        starts, failures = [], []
        worker = _worker(rig, base, [Pose.t_pose(), bent], base_pose=Pose.t_pose(),
                         base_fit_pose=edited)
        worker.failed.connect(failures.append)
        worker.pose_target_visual.connect(lambda *args: starts.append(args))
        worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(len(starts), 2)
        # -55 degrees is next to the -60 degree pose the base was fitted in,
        # not to the freshly fitted bind pose.
        np.testing.assert_allclose(starts[1][4], worker._mapper.local_to_world_np(base, pose=bent)[0],
                                   atol=1e-6)
        self.assertIs(worker.result.base_pose, worker._base_pose)

    def test_chained_fits_stay_inside_the_outer_bound_around_the_base(self):
        rig = make_rig()
        base = make_library(rig).base
        poses = _hinge_poses(rig, -30, -60, 30, 60)
        failures = []
        worker = _worker(rig, base, poses, fit_kwargs={
            "num_steps": 20, "lr_init": .05, "lr_final": .05,
            "parameter_bound_center_radius_factor": .05, "parameter_bound_radii_factor": 1.05})
        worker.failed.connect(failures.append)
        worker.run()
        self.assertEqual(failures, [])
        limit = .05 * np.max(base.local_radii, axis=1)
        for key in worker.result.keys:
            self.assertTrue(np.all(np.linalg.norm(key.delta_centers, axis=1) <= limit + 1e-5),
                            (key.name, key.delta_centers, limit))
            self.assertLessEqual(np.max(np.abs(key.delta_log_radii)), np.log(1.05) + 1e-5)
        self.assertTrue(any(np.max(np.linalg.norm(key.delta_centers, axis=1)) > .5 * limit[0]
                            for key in worker.result.keys), "the bound must be exercised")


if __name__ == "__main__":
    unittest.main(verbosity=2)
