"""Behavioral regressions for Unity mesh/rig snapshots and shared normalization."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api_rig_space import rig_snapshot_error, validate_unity_rig_snapshot  # noqa: E402
from main_window import MainWindow  # noqa: E402
from mesh_io import NormalizationTransform, load_and_prepare_arrays  # noqa: E402
from skeleton import mat4_compose  # noqa: E402
from skinning import _deform_numpy  # noqa: E402


def _matrix(position=(0, 0, 0), angle=0, scale=(1, 1, 1)):
    half = np.radians(angle) / 2
    return mat4_compose(np.asarray(position), np.array([0, 0, np.sin(half), np.cos(half)]), np.asarray(scale))


def _fixture(renderer=None):
    renderer = np.eye(4) if renderer is None else renderer
    # An asymmetric hand-like surface with its wrist below its AABB midpoint.
    local = np.array([[-.2, 0, 0], [.2, 0, .1], [-.1, .9, 0], [.15, 1.3, .05]], dtype=np.float32)
    bind_local = np.stack([_matrix(), _matrix((0, .7, 0))])
    posed_local = np.stack([_matrix((.05, -.1, .03), 15), _matrix((.1, .8, .05), -25)])
    bind = renderer @ bind_local
    current = renderer @ posed_local
    joints = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    weights = np.array([[1, 0], [.8, .2], [.9, .1], [1, 0]], dtype=np.float32)
    rest = local @ renderer[:3, :3].T + renderer[:3, 3]
    # Reference skinning uses the existing deformation path, independently of the validator.
    actual = _deform_numpy(rest, joints, weights, current @ np.linalg.inv(bind))
    rig = {
        "coordinateSpace": "unity_world",
        "bones": [dict(name=name, parent=parent, matrix=bind[i].tolist(), currentMatrix=current[i].tolist())
                  for i, (name, parent) in enumerate([("Wrist", -1), ("Finger", 0)])],
        "restVertices": rest.tolist(), "boneIndices": joints.tolist(), "boneWeights": weights.tolist(),
    }
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return actual, faces, rig


class UnityRigSnapshotTest(unittest.TestCase):
    def test_transformed_and_nonuniformly_scaled_rigs_are_valid(self):
        for position, angle, scale in [
            ((0, 0, 0), 0, (1, 1, 1)), ((3, -4, 5), 37, (.5, .5, .5)),
            ((3, -4, 5), -29, (.5, 2, 3)), ((-7, 4, 2), 87, (-1, 2, .4)),
            ((0, 0, 0), 12, (.0001, .0001, .0001)),
            ((100000, -100000, 50000), 35, (1, 2, .5)),
        ]:
            with self.subTest(position=position, scale=scale):
                verts, _, rig = _fixture(_matrix(position, angle, scale))
                before = copy.deepcopy(rig)
                result = validate_unity_rig_snapshot(rig, verts)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(rig, before)

    def test_constant_offset_is_measured_without_moving_the_bones(self):
        vertices, _, rig = _fixture()
        offset = np.array([.2, -.4, .1])
        result = validate_unity_rig_snapshot(rig, vertices + offset)
        report = result["snapshots"][0]
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(report["classification"], "translation")
        np.testing.assert_allclose(report["meanOffset"], offset, atol=1e-7)
        self.assertIn("mesh offset [0.2, -0.4, 0.1]", rig_snapshot_error(result))

    def test_double_applied_scale_is_identified(self):
        vertices, _, rig = _fixture(_matrix((3, -4, 5), 22, (.5, .5, .5)))
        doubled = (vertices - np.array([3, -4, 5])) * .5 + np.array([3, -4, 5])
        report = validate_unity_rig_snapshot(rig, doubled)["snapshots"][0]
        self.assertEqual(report["classification"], "scale")
        self.assertAlmostEqual(report["scaleRatio"], .5, places=5)

    def test_bind_skeleton_with_current_mesh_is_rejected(self):
        vertices, _, rig = _fixture()
        for bone in rig["bones"]:
            bone["currentMatrix"] = bone["matrix"]
        report = validate_unity_rig_snapshot(rig, vertices)["snapshots"][0]
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["classification"], "pose_or_transform")

    def test_all_frames_are_checked_and_error_names_the_bad_pose(self):
        vertices, _, rig = _fixture()
        matrices = [b["currentMatrix"] for b in rig["bones"]]
        rig["poseFrames"] = [dict(name="Good", vertices=vertices.tolist(), boneMatrices=matrices),
                             dict(name="Hand_Fist", vertices=(vertices + .3).tolist(), boneMatrices=matrices)]
        result = validate_unity_rig_snapshot(rig, vertices)
        self.assertEqual([r["status"] for r in result["snapshots"]], ["ok", "ok", "mismatch"])
        self.assertIn("Hand_Fist", rig_snapshot_error(result))

    def test_rounding_inside_tolerance_is_accepted(self):
        vertices, _, rig = _fixture()
        result = validate_unity_rig_snapshot(rig, vertices + 1e-6)
        self.assertEqual(result["status"], "ok")

    def test_active_blendshapes_are_explicitly_skipped(self):
        vertices, _, rig = _fixture()
        rig["snapshotValidation"] = dict(status="skipped", reason="active_blend_shapes")
        result = validate_unity_rig_snapshot(rig, vertices + .1)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["snapshots"][0]["reason"], "active_blend_shapes")

    def test_client_measurements_cannot_mask_a_mismatch(self):
        vertices, _, rig = _fixture()
        for declaration in [dict(status="ok", maxError=0), dict(status="skipped", reason="looks_aligned")]:
            rig["snapshotValidation"] = declaration
            self.assertEqual(validate_unity_rig_snapshot(rig, vertices + .1)["status"], "mismatch")

    def test_legacy_and_static_requests_remain_usable(self):
        vertices, _, rig = _fixture()
        self.assertEqual(validate_unity_rig_snapshot(None, vertices)["reason"], "no_rig")
        del rig["restVertices"]
        self.assertEqual(validate_unity_rig_snapshot(rig, vertices)["reason"], "missing_rest_or_bind_data")

    def test_invalid_skinning_data_is_rejected_before_comparison(self):
        vertices, _, original = _fixture()
        def bad_bone(rig): rig["boneIndices"][0][0] = 99
        def bad_weight(rig): rig["boneWeights"][0][0] = -.1
        def zero_weights(rig): rig["boneWeights"][0] = [0, 0]
        def fractional_bone(rig): rig["boneIndices"][0][0] = .5
        def singular(rig): rig["bones"][0]["matrix"][0][0] = 0
        def nonfinite(rig): rig["restVertices"][0][0] = float("nan")
        def wrong_shape(rig): rig["restVertices"] = rig["restVertices"][:-1]
        for mutate in (bad_bone, bad_weight, zero_weights, fractional_bone, singular, nonfinite, wrong_shape):
            with self.subTest(mutation=mutate.__name__), self.assertRaises(ValueError):
                rig = copy.deepcopy(original)
                mutate(rig)
                validate_unity_rig_snapshot(rig, vertices)

    def test_incomplete_pose_frame_is_reported(self):
        vertices, _, rig = _fixture()
        rig["poseFrames"] = [dict(name="Incomplete", vertices=vertices.tolist())]
        with self.assertRaisesRegex(ValueError, r"poseFrames\[0\]"):
            validate_unity_rig_snapshot(rig, vertices)

    def test_zero_weight_padding_indices_are_ignored(self):
        vertices, _, rig = _fixture()
        rig["boneIndices"][0][1] = -99
        self.assertEqual(validate_unity_rig_snapshot(rig, vertices)["status"], "ok")

    def test_import_stops_before_mesh_preparation_on_mismatch(self):
        vertices, faces, rig = _fixture()
        payload = dict(vertices=(vertices + .2).tolist(), faces=faces.tolist(), rig=rig)
        failures = []
        registry = SimpleNamespace(status_dict=lambda _: {"state": "queued"},
            get=lambda _: SimpleNamespace(payload=payload), update=lambda *args, **kwargs: True)
        owner = SimpleNamespace(_api_server=SimpleNamespace(registry=registry), _api_job_id=None,
            _api_has_local_work_in_flight=lambda: False, _api_fail=lambda job, message: failures.append(message),
            _api_resolve_primitive_request=MainWindow._api_resolve_primitive_request)
        # Resolve primitive settings without depending on GUI controls.
        owner._api_resolve_primitive_request = lambda *args: ("ellipsoid", {})
        MainWindow._api_on_fit_requested_impl(owner, "snapshot-test")
        self.assertEqual(len(failures), 1)
        self.assertIn("Mesh/skeleton mismatch", failures[0])

    def test_rest_mesh_current_pose_and_joints_share_one_normalization(self):
        for renderer in [np.eye(4), _matrix((3, -4, 5), 37, (.5, 2, 3))]:
            vertices, faces, rig = _fixture(renderer)
            mesh, transform = load_and_prepare_arrays(vertices, faces)
            owner = SimpleNamespace()
            imported = MainWindow._api_build_rigged_mesh(owner, np.asarray(mesh.vertices), faces, rig, transform)
            positions, _ = imported.skeleton.compute_bone_positions_rotations(owner._api_base_pose)
            expected_positions = transform.to_normalized_point(
                np.array([b["currentMatrix"] for b in rig["bones"]])[:, :3, 3])
            np.testing.assert_allclose(positions, expected_positions, atol=2e-6)
            self.assertGreater(np.linalg.norm(positions[0]), .1)
            skinned = _deform_numpy(imported.vertices, imported.skin_joints, imported.skin_weights,
                imported.skeleton.compute_skin_matrices(owner._api_base_pose))
            np.testing.assert_allclose(skinned, mesh.vertices, atol=3e-6)
            np.testing.assert_allclose(transform.to_original_point(mesh.vertices), vertices, atol=1e-6)

    def test_overlay_uses_current_matrices_and_shared_normalization(self):
        vertices, _, rig = _fixture()
        transform = NormalizationTransform(center=np.array([.1, .6, 0]), scale=2.5)
        draws = []
        owner = SimpleNamespace(_viewer=SimpleNamespace(show_bones=lambda p, parents, **kwargs: draws.append(p),
            remove_skeleton=lambda: self.fail("A valid rig must remain visible")),
            _status=SimpleNamespace(showMessage=self.fail))
        MainWindow._api_show_rig_bones(owner, rig, transform)
        expected = transform.to_normalized_point(np.array([b["currentMatrix"] for b in rig["bones"]])[:, :3, 3])
        np.testing.assert_allclose(draws[0], expected, atol=1e-6)

    def test_pose_library_does_not_replace_current_snapshot_in_viewer(self):
        vertices, faces, rig = _fixture()
        frame_matrices = np.array([b["currentMatrix"] for b in rig["bones"]])
        frame_matrices[:, :3, 3] += [.3, .4, .5]
        frame_vertices = vertices + np.array([.3, .4, .5])
        rig["poseFrames"] = [dict(name="Other pose", vertices=frame_vertices.tolist(),
                                 boneMatrices=frame_matrices.tolist())]
        self.assertEqual(validate_unity_rig_snapshot(rig, vertices)["status"], "ok")
        mesh, transform = load_and_prepare_arrays(vertices, faces)
        owner = SimpleNamespace()
        imported = MainWindow._api_build_rigged_mesh(owner, np.asarray(mesh.vertices), faces, rig, transform)
        current = imported.skeleton.compute_world_transforms(owner._api_base_pose)[:, :3, 3]
        expected = transform.to_normalized_point(np.array([b["currentMatrix"] for b in rig["bones"]])[:, :3, 3])
        np.testing.assert_allclose(current, expected, atol=1e-6)
        self.assertEqual(imported.poses[0].name, "Other pose")

    def test_legacy_self_parent_does_not_erase_the_wrist_origin(self):
        vertices, faces, rig = _fixture(_matrix((3, -4, 5), 37, (.5, 2, 3)))
        rig["bones"][0]["parent"] = 0
        mesh, transform = load_and_prepare_arrays(vertices, faces)
        owner = SimpleNamespace()
        imported = MainWindow._api_build_rigged_mesh(owner, np.asarray(mesh.vertices), faces, rig, transform)
        positions, _ = imported.skeleton.compute_bone_positions_rotations(owner._api_base_pose)
        expected = transform.to_normalized_point(np.array([b["currentMatrix"] for b in rig["bones"]])[:, :3, 3])
        self.assertEqual(imported.skeleton.bones[0].parent_index, -1)
        np.testing.assert_allclose(positions, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
