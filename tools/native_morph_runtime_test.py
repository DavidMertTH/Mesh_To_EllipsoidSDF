"""Numerical contracts for native Unity-compatible pose morphing and V4 IO."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from joint_pose import JointPoseEdits
from native_morph import MorphSettings, NativeMorphDriver, rotation_distances
from native_morph_io import load_native_morph_fit, native_morph_from_json, save_native_morph_fit
from pose_correctives import PoseCorrectiveKey, PoseCorrectiveLibrary
from skeleton import Bone, Pose, Skeleton, mat4_compose, mat4_decompose, quat_multiply


def axis_angle(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = np.radians(degrees) * .5
    return np.r_[axis * np.sin(half), np.cos(half)]


def rig():
    skeleton = Skeleton([
        Bone("source:Root", 0),
        Bone("source:Elbow", 1, 0, mat4_compose([0, 1, 0], axis_angle([1, 0, 0], 15), [1, 1, 1])),
        Bone("source:Hand", 2, 1, mat4_compose([0, 1, 0], axis_angle([0, 0, 1], -5), [1, 1, 1])),
        Bone("source:Other", 3, 0, mat4_compose([1, 0, 0], [0, 0, 0, 1], [1, 1, 1])),
    ])
    for bone, world in zip(skeleton.bones, skeleton.compute_world_transforms()):
        bone.inverse_bind_matrix = np.linalg.inv(world)
    return SimpleNamespace(skeleton=skeleton, mesh_name="test",
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        skin_joints=np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int32),
        skin_weights=np.array([[1, 0], [.6, .4], [.8, .2]], dtype=np.float32))


def pose(skeleton, angles, name="Unrelated clip"):
    locals_ = {}
    for index, angle in angles.items():
        local = skeleton.bones[index].local_bind_transform.copy()
        local[:3, :3] = local[:3, :3] @ mat4_compose([0, 0, 0], axis_angle([1, 0, 0], angle), [1, 1, 1])[:3, :3]
        locals_[index] = local
    return Pose(name, locals_)


def library(skeleton, primitive_type="superquadric"):
    base = BoneLocalEllipsoids(
        local_centers=np.array([[.1, .2, .3], [.2, .1, 0], [.3, 0, .1]], dtype=np.float32),
        local_radii=np.array([[.1, .2, .3], [.2, .3, .1], [.3, .1, .2]], dtype=np.float32),
        local_rotations=np.array([axis_angle([0, 1, 0], 20)] * 3, dtype=np.float32),
        bone_assignments=np.array([1, 2, 3], dtype=np.int32),
        attachment_joints=np.array([[1, 2], [2, 1], [3, 0]], dtype=np.int32),
        attachment_weights=np.array([[.8, .2], [.6, .4], [.7, .3]], dtype=np.float32),
        primitive_type=primitive_type,
        shape_exponents=np.array([[.6, .8], [.8, 1.1], [1.3, 1.1]], dtype=np.float32))
    keys = []
    for angle, factor in ((90, 1), (-70, -1)):
        keys.append(PoseCorrectiveKey(f"Elbow {angle}",
            delta_centers=np.array([[.2, -.1, .1], [.1, 0, .2], [.05, .1, -.1]], dtype=np.float32) * factor,
            delta_rotations=np.array([axis_angle([0, 0, 1], factor * 70)] * 3, dtype=np.float32),
            delta_log_radii=np.array([[.5, -.4, .2], [-.1, .3, .2], [.2, -.2, .3]], dtype=np.float32) * factor,
            delta_log_shape_exponents=np.array([[.3, -.2], [-.1, .1], [.2, .1]], dtype=np.float32) * factor,
            pose=pose(skeleton, {1: angle})))
    return PoseCorrectiveLibrary(base, keys, Pose.t_pose())


def unity_weights(skeleton, library_, current_pose, settings):
    """Independent scalar translation of Unity's weight jobs and exact lock.

    Quaternion.Angle uses acos(dot); context traversal is Unity's parent-first
    recursion. This reference intentionally does not use native helper code.
    """
    def rotations(pose_):
        return [mat4_decompose(pose_.bone_locals.get(i, b.local_bind_transform)
                if pose_ is not None else b.local_bind_transform)[1]
                for i, b in enumerate(skeleton.bones)]

    current, neutral = rotations(current_pose), rotations(library_.base_pose)
    targets = [rotations(key.pose) for key in library_.keys if key.pose is not None]
    contexts = []
    for bone in range(skeleton.num_bones):
        seen = []
        def visit(index, depth):
            if index < 0 or index in [i for i, _w in seen]:
                return
            seen.append((index, settings.context_falloff ** depth))
            if depth >= settings.context_depth:
                return
            visit(skeleton.bones[index].parent_index, depth + 1)
            for child, entry in enumerate(skeleton.bones):
                if entry.parent_index == index:
                    visit(child, depth + 1)
        visit(bone, 0)
        contexts.append(seen)

    def distance(target, context):
        sum_, weight = 0, 0
        for index, w in context:
            q, p = current[index], target[index]
            dot = abs(np.dot(q / np.linalg.norm(q), p / np.linalg.norm(p)))
            sum_ += w * 2 * np.arccos(min(1, dot)) / np.pi
            weight += w
        return sum_ / weight

    base_distance = [distance(neutral, c) for c in contexts]
    target_distance = np.array([[distance(t, c) for c in contexts] for t in targets])
    global_distance = np.mean(target_distance, axis=1)
    nearest = int(np.argmin(global_distance))
    second = min(float(np.mean(base_distance)), *(global_distance[i]
        for i in range(len(targets)) if i != nearest))
    if (settings.exact_match_epsilon > 0 and global_distance[nearest] <= settings.exact_match_epsilon
            and second > global_distance[nearest] + settings.exact_match_epsilon):
        result = np.zeros_like(target_distance)
        result[nearest] = 1
        return result
    result = np.zeros_like(target_distance)
    for bone in range(skeleton.num_bones):
        base = settings.base_support * np.exp(-max(.0001, settings.falloff) * base_distance[bone]) / max(base_distance[bone] ** 2, 1e-8)
        raw = []
        for target in range(len(targets)):
            d = target_distance[target, bone]
            is_neutral = (settings.base_support > 0 and settings.base_tie_epsilon > 0
                and base_distance[bone] <= settings.base_tie_epsilon
                and d + settings.base_tie_epsilon >= base_distance[bone])
            raw.append(0 if is_neutral else np.exp(-max(.0001, settings.falloff) * d) / max(d * d, 1e-8))
        retained = sorted(range(len(raw)), key=lambda i: (-raw[i], i))
        if settings.top_k:
            retained = retained[:settings.top_k]
        denominator = base + sum(raw[i] for i in retained)
        if denominator > 1e-6:
            for index in retained:
                result[index, bone] = raw[index] / denominator
    return result


class NativeMorphRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rig = rig()
        self.library = library(self.rig.skeleton)

    def test_numerically_matches_unity_default_per_bone_algorithm(self):
        for depth in (0, 1, 2):
            for top_k in (0, 1, 2):
                for support in (0, .4, 1, 2):
                    settings = MorphSettings(context_depth=depth, top_k=top_k, base_support=support)
                    driver = NativeMorphDriver(self.rig.skeleton, self.library, settings)
                    for angle in (-70, -35, 0, 20, 45, 90, 150):
                        current = pose(self.rig.skeleton, {1: angle, 2: angle * .12})
                        np.testing.assert_allclose(driver.desired_weights(current),
                            unity_weights(self.rig.skeleton, self.library, current, settings), atol=1e-7)

    def test_exact_key_recovers_all_parameters_and_unchanged_joint_deltas(self):
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        for key in self.library.keys:
            actual = driver.evaluate(key.pose)
            np.testing.assert_allclose(actual.local_centers, self.library.base.local_centers + key.delta_centers, atol=1e-7)
            np.testing.assert_allclose(actual.local_radii, self.library.base.local_radii * np.exp(key.delta_log_radii), atol=1e-7)
            np.testing.assert_allclose(actual.shape_exponents, self.library.base.shape_exponents * np.exp(key.delta_log_shape_exponents), atol=1e-7)
            expected = np.array([quat_multiply(q, d) for q, d in zip(self.library.base.local_rotations, key.delta_rotations)])
            np.testing.assert_allclose(rotation_distances(actual.local_rotations, expected), 0, atol=1e-7)

    def test_neutral_keeps_base_exactly_including_coincident_training_key(self):
        self.library.keys.append(replace(self.library.keys[0], pose=Pose.t_pose()))
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        actual = driver.evaluate(Pose.t_pose())
        np.testing.assert_allclose(actual.local_centers, self.library.base.local_centers)
        np.testing.assert_allclose(actual.local_radii, self.library.base.local_radii)
        np.testing.assert_equal(driver.weights, 0)

    def test_duplicate_exact_descriptors_share_support_without_exact_lock(self):
        self.library.keys.append(replace(self.library.keys[0], name="duplicate"))
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        weights = driver.desired_weights(self.library.keys[0].pose)
        self.assertAlmostEqual(weights[0, 1], .5, places=6)
        self.assertAlmostEqual(weights[2, 1], .5, places=6)

    def test_manual_between_key_pose_uses_geometric_log_blend(self):
        edits = JointPoseEdits(self.rig.skeleton, Pose.t_pose())
        self.assertTrue(edits.set_axis(1, 0, 45))
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        result = driver.evaluate(edits.apply())
        weights = driver.weights[:, self.library.base.bone_assignments]
        expected_centers = self.library.base.local_centers.copy().astype(float)
        expected_logs = np.log(self.library.base.local_radii.astype(float))
        expected_rotation_angles = np.zeros(3)
        for index, key in enumerate(self.library.keys):
            expected_centers += weights[index, :, None] * key.delta_centers
            expected_logs += weights[index, :, None] * key.delta_log_radii
            expected_rotation_angles += weights[index] * (70 if index == 0 else -70)
        np.testing.assert_allclose(result.local_centers, expected_centers, atol=1e-7)
        np.testing.assert_allclose(result.local_radii, np.exp(expected_logs), atol=1e-7)
        expected_q = [quat_multiply(q, axis_angle([0, 0, 1], angle))
                      for q, angle in zip(self.library.base.local_rotations, expected_rotation_angles)]
        np.testing.assert_allclose(rotation_distances(result.local_rotations, expected_q), 0, atol=1e-7)

    def test_parent_child_context_differs_from_unrelated_branch(self):
        driver = NativeMorphDriver(self.rig.skeleton, self.library,
            MorphSettings(exact_match_epsilon=0))
        driver.evaluate(pose(self.rig.skeleton, {1: 45}))
        self.assertGreater(driver.weights[0, 2], .1)
        self.assertEqual(driver.weights[0, 3], 0)
        no_context = NativeMorphDriver(self.rig.skeleton, self.library,
            MorphSettings(context_depth=0, exact_match_epsilon=0))
        self.assertEqual(no_context.desired_weights(pose(self.rig.skeleton, {1: 45}))[0, 2], 0)

    def test_top_k_ties_are_stable_by_original_key_id(self):
        self.library.keys = [replace(self.library.keys[0], name=str(i), pose=pose(self.rig.skeleton, {1: 60})) for i in range(8)]
        driver = NativeMorphDriver(self.rig.skeleton, self.library,
            MorphSettings(top_k=3, exact_match_epsilon=0, base_support=0))
        weights = driver.desired_weights(pose(self.rig.skeleton, {1: 30}))
        np.testing.assert_allclose(weights[:3], 1 / 3)
        np.testing.assert_equal(weights[3:], 0)

    def test_time_smoothing_has_unity_90_percent_response(self):
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        driver.evaluate(Pose.t_pose())
        target = self.library.keys[0].pose
        driver.evaluate(target, dt=0)
        np.testing.assert_equal(driver.weights, 0)
        self.assertFalse(driver.settled)
        driver.evaluate(target, dt=.12)
        np.testing.assert_allclose(driver.weights[0], .9, atol=1e-12)
        driver.evaluate(target, dt=2)
        self.assertTrue(driver.settled)
        np.testing.assert_equal(driver.weights[0], 1)

    def test_smoothing_disabled_and_reset_are_immediate(self):
        for smoothing_time in (0, .12):
            driver = NativeMorphDriver(self.rig.skeleton, self.library,
                MorphSettings(smoothing_time=smoothing_time))
            driver.evaluate(Pose.t_pose())
            driver.reset()
            driver.evaluate(self.library.keys[0].pose, dt=0)
            np.testing.assert_equal(driver.weights[0], 1)

    def test_output_ids_and_smooth_attachments_cannot_mutate_the_fit(self):
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        output = driver.evaluate(self.library.keys[0].pose)
        np.testing.assert_equal(output.bone_assignments, self.library.base.bone_assignments)
        np.testing.assert_equal(output.attachment_joints, self.library.base.attachment_joints)
        np.testing.assert_allclose(output.attachment_weights, self.library.base.attachment_weights)
        output.bone_assignments[:] = 0
        output.attachment_joints[:] = 0
        output.attachment_weights[:] = 0
        np.testing.assert_equal(driver.evaluate(Pose.t_pose()).bone_assignments, [1, 2, 3])
        np.testing.assert_equal(self.library.base.attachment_joints[0], [1, 2])

    def test_rigid_and_smooth_mappers_accept_morphed_local_population(self):
        mapper = BoneEllipsoidMapper(self.rig.skeleton)
        current = self.library.keys[0].pose
        result = NativeMorphDriver(self.rig.skeleton, self.library).evaluate(current)
        centers, radii, rotations = mapper.local_to_world_np(result, current)
        self.assertTrue(np.isfinite(centers).all())
        np.testing.assert_equal(radii, result.local_radii)
        np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1, atol=1e-6)
        self.assertGreater(np.max(np.abs(centers - mapper.local_to_world_np(self.library.base, current)[0])), .05)

    def test_antipodes_and_unnormalized_quaternions_are_equivalent(self):
        quaternions = np.array([axis_angle([1, 2, 3], angle) for angle in (-170, 0, 90, 180)])
        np.testing.assert_allclose(rotation_distances(quaternions * 1e300, -quaternions * 1e-300), 0, atol=1e-15)
        first = NativeMorphDriver(self.rig.skeleton, self.library).evaluate(self.library.keys[0].pose)
        self.library.base.local_rotations = -self.library.base.local_rotations.astype(float) * 1e300
        for key in self.library.keys:
            key.delta_rotations = -key.delta_rotations.astype(float) * 1e300
        second = NativeMorphDriver(self.rig.skeleton, self.library).evaluate(self.library.keys[0].pose)
        np.testing.assert_allclose(rotation_distances(first.local_rotations, second.local_rotations), 0, atol=1e-7)

    def test_exact_half_turn_antipodes_have_same_logarithmic_partial_blend(self):
        self.library.keys[0].delta_rotations = np.tile([0, -1, 0, 0], (3, 1))
        first = NativeMorphDriver(self.rig.skeleton, self.library).evaluate(pose(self.rig.skeleton, {1: 45}))
        self.library.keys[0].delta_rotations *= -1
        second = NativeMorphDriver(self.rig.skeleton, self.library).evaluate(pose(self.rig.skeleton, {1: 45}))
        np.testing.assert_allclose(rotation_distances(first.local_rotations, second.local_rotations), 0, atol=1e-7)

    def test_pose_names_translation_and_scale_do_not_drive_weights(self):
        reference = self.library.keys[0].pose
        local = reference.bone_locals[1].copy()
        local[:3, 3] = [100, -20, 15]
        local[:3, :3] = local[:3, :3] @ np.diag([2, .5, 3])
        arbitrary = Pose("No timeline relation", {1: local})
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        np.testing.assert_allclose(driver.desired_weights(arbitrary), driver.desired_weights(reference), atol=1e-7)

    def test_gimbal_crossing_does_not_jump_rotation_descriptor_weights(self):
        self.library.keys[0].pose = Pose("Y90", {1: mat4_compose([0, 1, 0], axis_angle([0, 1, 0], 90), [1, 1, 1])})
        driver = NativeMorphDriver(self.rig.skeleton, self.library, MorphSettings(exact_match_epsilon=0))
        left = Pose("a", {1: mat4_compose([0, 1, 0], axis_angle([0, 1, 0], 89.99), [1, 1, 1])})
        right = Pose("b", {1: mat4_compose([0, 1, 0], axis_angle([0, 1, 0], 90.01), [1, 1, 1])})
        np.testing.assert_allclose(driver.desired_weights(left), driver.desired_weights(right), atol=1e-7)

    def test_missing_descriptors_and_empty_library_keep_base(self):
        for keys in ([], [replace(self.library.keys[0], pose=None)]):
            driver = NativeMorphDriver(self.rig.skeleton, replace(self.library, keys=keys))
            self.assertFalse(driver.has_descriptors)
            actual = driver.evaluate(pose(self.rig.skeleton, {1: 90}))
            np.testing.assert_allclose(actual.local_centers, self.library.base.local_centers)

    def test_ellipsoid_exponents_stay_canonical_and_sq_bounds_are_safe(self):
        for primitive_type in ("ellipsoid", "superquadric"):
            library_ = library(self.rig.skeleton, primitive_type)
            library_.keys[0].delta_log_shape_exponents = np.tile([1e30, -1e30], (3, 1))
            with np.errstate(all="raise"):
                output = NativeMorphDriver(self.rig.skeleton, library_).evaluate(library_.keys[0].pose)
            expected = np.ones((3, 2)) if primitive_type == "ellipsoid" else np.tile([2, .1], (3, 1))
            np.testing.assert_allclose(output.shape_exponents, expected)

    def test_malformed_fit_and_radius_overflow_rejected_before_driver_creation(self):
        for field, value in (("delta_centers", np.full((3, 3), np.nan)),
                             ("delta_centers", np.full((3, 3), 1e300)),
                             ("delta_log_radii", np.full((3, 3), 1e6)),
                             ("delta_log_radii", np.full((3, 3), -1e6)),
                             ("delta_rotations", np.zeros((3, 4))),
                             ("delta_centers", np.zeros((9,)))):
            bad = copy.deepcopy(self.library)
            setattr(bad.keys[0], field, value)
            with self.subTest(field=field, value=value[0]), self.assertRaises(ValueError):
                NativeMorphDriver(self.rig.skeleton, bad)

    def test_invalid_elapsed_time_or_pose_does_not_mutate_weights(self):
        driver = NativeMorphDriver(self.rig.skeleton, self.library)
        driver.evaluate(Pose.t_pose())
        before = driver.weights.copy()
        for dt in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                driver.evaluate(self.library.keys[0].pose, dt)
            np.testing.assert_equal(before, driver.weights)
        with self.assertRaises(ValueError):
            driver.evaluate(Pose("invalid", {1: np.full((4, 4), np.nan)}))
        np.testing.assert_equal(before, driver.weights)

    def test_settings_validate_counts_and_ranges(self):
        for settings in ({"falloff": -1}, {"base_support": np.nan}, {"top_k": 1.2},
                         {"context_depth": 9}, {"context_depth": 1.5}, {"context_falloff": 2}):
            with self.assertRaises(ValueError):
                MorphSettings(**settings)


class NativeMorphIOTests(unittest.TestCase):
    def setUp(self):
        self.rig = rig()
        self.library = library(self.rig.skeleton)

    def payload(self):
        return self.library.to_json(self.rig.skeleton)

    def test_ellipsoid_and_superquadric_v4_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            for primitive in ("ellipsoid", "superquadric"):
                library_ = library(self.rig.skeleton, primitive)
                path = save_native_morph_fit(library_, self.rig, Path(folder) / f"{primitive}.json")
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["version"], 4)
                self.assertIn("skinning_digest", saved["native_model"])
                reopened = load_native_morph_fit(self.rig, path)
                original_driver = NativeMorphDriver(self.rig.skeleton, library_)
                reopened_driver = NativeMorphDriver(self.rig.skeleton, reopened)
                for angle in (0, 45, 90, -70):
                    current = pose(self.rig.skeleton, {1: angle})
                    a, b = original_driver.evaluate(current), reopened_driver.evaluate(current)
                    np.testing.assert_allclose(a.local_centers, b.local_centers, atol=2e-7)
                    np.testing.assert_allclose(a.local_radii, b.local_radii, atol=2e-7)
                    np.testing.assert_allclose(a.shape_exponents, b.shape_exponents, atol=2e-7)
                    np.testing.assert_allclose(rotation_distances(a.local_rotations, b.local_rotations), 0, atol=2e-7)

    def test_existing_unity_v4_object_and_parallel_arrays_can_load(self):
        for use_objects in (True, False):
            data = self.payload()
            if not use_objects:
                for key in data["poses"]:
                    del key["ellipsoids"]
            result = native_morph_from_json(self.rig, data)
            self.assertEqual(len(result.keys), 2)
            np.testing.assert_allclose(result.keys[0].delta_centers, self.library.keys[0].delta_centers)

    def test_real_t_pose_fbx_manual_joints_drive_roundtrip_morph(self):
        from rig_loader import load_rigged_mesh
        rigged = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")
        skeleton = rigged.skeleton
        indices = {bone.name.rsplit(":", 1)[-1]: i for i, bone in enumerate(skeleton.bones)}
        elbow, knee = indices["LeftForeArm"], indices["RightLeg"]
        base = BoneLocalEllipsoids(
            local_centers=np.array([[.01, .02, .03], [.04, .05, .06]], dtype=np.float32),
            local_radii=np.array([[.1, .08, .06], [.2, .1, .08]], dtype=np.float32),
            local_rotations=np.tile([0., 0., 0., 1.], (2, 1)).astype(np.float32),
            bone_assignments=np.array([elbow, knee], dtype=np.int32))
        edits = JointPoseEdits(skeleton, Pose.t_pose())
        self.assertTrue(edits.set_axis(elbow, 2, 90))
        self.assertTrue(edits.set_axis(knee, 0, -90))
        key = PoseCorrectiveKey("Ranges, no clip", np.array([[.03, 0, 0], [0, .04, 0]], dtype=np.float32),
            np.tile(axis_angle([1, 0, 0], 20), (2, 1)).astype(np.float32),
            np.array([[.4, -.2, .1], [-.1, .2, .1]], dtype=np.float32), pose=edits.apply())
        library_ = PoseCorrectiveLibrary(base, [key], Pose.t_pose())
        with tempfile.TemporaryDirectory() as folder:
            saved = save_native_morph_fit(library_, rigged, Path(folder) / "tpose.json")
            loaded = load_native_morph_fit(rigged, saved)
            for current in (key.pose, loaded.keys[0].pose):
                result = NativeMorphDriver(skeleton, loaded).evaluate(current)
                np.testing.assert_allclose(result.local_centers, base.local_centers + key.delta_centers, atol=2e-7)
                np.testing.assert_allclose(result.local_radii, base.local_radii * np.exp(key.delta_log_radii), atol=2e-7)

    def test_out_of_order_primitive_ids_are_restored(self):
        data = self.payload()
        data["base"].reverse()
        for key in data["poses"]:
            key["ellipsoids"].reverse()
        loaded = native_morph_from_json(self.rig, data)
        np.testing.assert_allclose(loaded.base.local_centers, self.library.base.local_centers)
        np.testing.assert_allclose(loaded.keys[0].delta_centers, self.library.keys[0].delta_centers)

    def reordered_rig(self):
        new = copy.deepcopy(self.rig)
        order = [2, 0, 3, 1]
        mapping = {old: index for index, old in enumerate(order)}
        bones = []
        for index, old in enumerate(order):
            bone = copy.deepcopy(self.rig.skeleton.bones[old])
            bone.index = index
            bone.name = "target:" + bone.name.rsplit(":", 1)[-1]
            bone.parent_index = mapping.get(bone.parent_index, -1)
            bones.append(bone)
        new.skeleton = Skeleton(bones)
        new.skin_joints = np.vectorize(mapping.get)(new.skin_joints).astype(np.int32)
        return new, mapping

    def test_name_remapping_preserves_attachment_identity_after_reordering(self):
        for with_attachment_names in (True, False):
            target, mapping = self.reordered_rig()
            data = self.payload()
            if not with_attachment_names:
                for entry in data["base"]:
                    del entry["attachment_bones"]
            reopened = native_morph_from_json(target, data)
            np.testing.assert_equal(reopened.base.bone_assignments, [mapping[i] for i in self.library.base.bone_assignments])
            np.testing.assert_equal(reopened.base.attachment_joints,
                [[mapping[i] for i in row] for row in self.library.base.attachment_joints])
            np.testing.assert_allclose(NativeMorphDriver(target.skeleton, reopened).evaluate(reopened.keys[0].pose).local_centers,
                NativeMorphDriver(self.rig.skeleton, self.library).evaluate(self.library.keys[0].pose).local_centers, atol=2e-7)

    def test_native_metadata_also_accepts_reordering_and_new_namespace(self):
        target, mapping = self.reordered_rig()
        with tempfile.TemporaryDirectory() as folder:
            path = save_native_morph_fit(self.library, self.rig, Path(folder) / "fit.json")
            loaded = load_native_morph_fit(target, path)
            np.testing.assert_equal(loaded.base.bone_assignments, [mapping[1], mapping[2], mapping[3]])

    def test_native_model_rejects_mesh_skinning_bind_and_hierarchy_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_native_morph_fit(self.library, self.rig, Path(folder) / "fit.json")
            for kind in ("mesh", "weights", "joints", "bind", "inverse", "hierarchy"):
                target = copy.deepcopy(self.rig)
                if kind == "mesh": target.vertices[0, 0] += .1
                if kind == "weights": target.skin_weights[1] = [.5, .5]
                if kind == "joints": target.skin_joints[1, 0] = 0
                if kind == "bind": target.skeleton.bones[1].local_bind_transform[0, 3] += .1
                if kind == "inverse": target.skeleton.bones[1].inverse_bind_matrix[0, 3] += .1
                if kind == "hierarchy": target.skeleton.bones[1].parent_index = 3
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    load_native_morph_fit(target, path)

    def test_incomplete_missing_and_ambiguous_descriptors_rejected(self):
        for mode in ("missing", "empty", "partial", "duplicate", "missing_bone", "wrong_parent"):
            data = self.payload()
            if mode == "missing": del data["poses"][0]["pose"]
            if mode == "empty": data["poses"][0]["pose"]["bones"] = []
            if mode == "partial": data["poses"][0]["pose"]["bones"].pop()
            if mode == "duplicate": data["poses"][0]["pose"]["bones"].append(data["poses"][0]["pose"]["bones"][0])
            if mode == "missing_bone": data["base"][0]["bone"] = "absent"
            if mode == "wrong_parent": data["poses"][0]["pose"]["bones"][1]["parent_index"] = 3
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                native_morph_from_json(self.rig, data)
        target = copy.deepcopy(self.rig)
        target.skeleton.bones[2].name = "another:Elbow"
        data = self.payload()
        data["base"][0]["bone"] = "unknown:Elbow"
        with self.assertRaises(ValueError):
            native_morph_from_json(target, data)

    def test_hostile_json_rejected_atomically_without_rig_mutation(self):
        original_worlds = self.rig.skeleton.compute_world_transforms().copy()
        cases = [None, {"format": "unknown"}]
        for mutate in (
            lambda d: d.update(count=3.5),
            lambda d: d.update(version=100),
            lambda d: d.update(quaternion_convention="wxyz"),
            lambda d: d.update(native_model=[]),
            lambda d: d.update(poses=[]),
            lambda d: d["base"][0].update(id=True),
            lambda d: d["base"][1].update(id=0),
            lambda d: d["base"][0].update(radii=[0, 1, 1]),
            lambda d: d["base"][0].update(local_center=[1e300, 0, 0]),
            lambda d: d["base"][0].update(local_rotation=[0, 0, 0, 0]),
            lambda d: d["base"][0].update(attachment_weights=[-1, 2]),
            lambda d: d["base"][0].update(attachment_weights=[0, 0]),
            lambda d: d["poses"][0].update(loss=float("nan")),
            lambda d: d["poses"][0]["ellipsoids"][0].update(delta_log_radii=[1e6, 0, 0]),
            lambda d: d["poses"][0]["ellipsoids"][0].update(delta_log_radii=[-1e6, 0, 0]),
            lambda d: d["poses"][0]["ellipsoids"][0].update(delta_local_center=[float("nan"), 0, 0]),
            lambda d: d["poses"][0]["ellipsoids"][0].update(delta_local_rotation=[0, 0, 0, 0]),
        ):
            data = self.payload()
            mutate(data)
            cases.append(data)
        for index, data in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                native_morph_from_json(self.rig, data)
            np.testing.assert_equal(original_worlds, self.rig.skeleton.compute_world_transforms())

    def test_quaternion_scale_is_normalized_on_read_and_save(self):
        data = self.payload()
        for entry in data["base"]:
            entry["local_rotation"] = (np.asarray(entry["local_rotation"]) * 1e300).tolist()
        for key in data["poses"]:
            for entry in key["ellipsoids"]:
                entry["delta_local_rotation"] = (-np.asarray(entry["delta_local_rotation"]) * 1e-300).tolist()
        loaded = native_morph_from_json(self.rig, data)
        np.testing.assert_allclose(np.linalg.norm(loaded.base.local_rotations, axis=1), 1, atol=1e-7)
        self.library.base.local_rotations = self.library.base.local_rotations.astype(float) * 1e300
        for key in self.library.keys:
            key.delta_rotations = key.delta_rotations.astype(float) * 1e300
        with tempfile.TemporaryDirectory() as folder:
            path = save_native_morph_fit(self.library, self.rig, Path(folder) / "fit.json")
            reopened = load_native_morph_fit(self.rig, path)
            np.testing.assert_allclose(rotation_distances(loaded.base.local_rotations, reopened.base.local_rotations), 0, atol=2e-7)

    def test_failed_save_does_not_replace_existing_file_or_leave_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_native_morph_fit(self.library, self.rig, Path(folder) / "fit.json")
            original = path.read_bytes()
            bad = copy.deepcopy(self.library)
            bad.keys[0].delta_log_radii[:] = 1e6
            with self.assertRaises(ValueError):
                save_native_morph_fit(bad, self.rig, path)
            self.assertEqual(path.read_bytes(), original)
            with patch("native_morph_io.os.replace", side_effect=OSError("simulated replacement failure")):
                with self.assertRaises(OSError):
                    save_native_morph_fit(self.library, self.rig, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
