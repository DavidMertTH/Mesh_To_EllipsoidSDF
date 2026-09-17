"""Regression tests for FBX mesh Model/Geometric transform evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbx_parser import (  # noqa: E402
    FbxNode,
    FbxProperty,
    _geometry_to_world_transform,
    _transform_mesh_geometry,
    extract_rig_data,
)
from rig_loader import load_rigged_mesh  # noqa: E402


HAND_DIR = ROOT / "clothSimulation" / "Character"
HAND_PATHS = {
    "RightHand": HAND_DIR / "RightHand.fbx",
    "LeftHand": HAND_DIR / "LeftHand.fbx",
}


def _property(name: str, *values: float | int) -> FbxNode:
    props = [
        FbxProperty("S", name),
        FbxProperty("S", ""),
        FbxProperty("S", ""),
        FbxProperty("S", "A"),
    ]
    props.extend(FbxProperty("D", value) for value in values)
    return FbxNode("P", properties=props)


def _model(*properties: FbxNode) -> FbxNode:
    return FbxNode(
        "Model",
        properties=[
            FbxProperty("L", 1),
            FbxProperty("S", "SyntheticMesh"),
            FbxProperty("S", "Mesh"),
        ],
        children=[FbxNode("Properties70", children=list(properties))],
    )


class FbxModelTransformUnitTest(unittest.TestCase):
    def test_bind_pose_and_geometric_trs_are_composed_once(self) -> None:
        model = _model(
            _property("Lcl Translation", 100.0, 200.0, 300.0),
            _property("GeometricTranslation", 4.0, 5.0, 6.0),
            _property("GeometricRotation", 0.0, 0.0, 90.0),
            _property("GeometricScaling", 2.0, 3.0, 4.0),
        )
        bind_world = np.eye(4, dtype=np.float64)
        bind_world[:3, 3] = [10.0, 20.0, 30.0]

        transform = _geometry_to_world_transform(
            1,
            {1: model},
            {},
            {1: bind_world},
        )
        point = transform @ np.array([1.0, 0.0, 0.0, 1.0])

        # Geometric S then R then T maps [1,0,0] -> [4,7,6]; the authoritative
        # bind-pose world translation then maps that to [14,27,36]. The large
        # Lcl Translation must not be applied a second time.
        np.testing.assert_allclose(point, [14.0, 27.0, 36.0, 1.0], atol=1e-10)

    def test_model_local_rotation_is_fallback_without_bind_pose(self) -> None:
        model = _model(_property("Lcl Rotation", 90.0, 0.0, 0.0))
        transform = _geometry_to_world_transform(1, {1: model}, {}, {})
        point = transform @ np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(point, [0.0, -1.0, 0.0, 1.0], atol=1e-10)

    def test_mirrored_geometry_reverses_triangle_winding(self) -> None:
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mirror = np.diag([-1.0, 1.0, 1.0, 1.0])

        transformed, transformed_faces = _transform_mesh_geometry(
            vertices, faces, mirror)

        np.testing.assert_allclose(transformed[1], [-1.0, 0.0, 0.0])
        np.testing.assert_array_equal(transformed_faces, [[0, 2, 1]])


@unittest.skipUnless(
    all(path.is_file() for path in HAND_PATHS.values()),
    "Unity RightHand.fbx and LeftHand.fbx assets are not available",
)
class HandFbxIntegrationTest(unittest.TestCase):
    EXPECTED_BOUNDS = {
        "RightHand": (
            [-5.7361612, -9.4270649, -5.9476795],
            [6.7353210, 10.0921116, 1.9175988],
        ),
        "LeftHand": (
            [-6.7353210, -9.4270649, -5.9476795],
            [5.7361612, 10.0921116, 1.9175990],
        ),
    }

    def test_hand_mesh_and_cluster_bindposes_share_world_space(self) -> None:
        expected_model_rotation = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        for hand_name, path in HAND_PATHS.items():
            with self.subTest(hand=hand_name):
                parsed = extract_rig_data(path)
                self.assertIsNotNone(parsed.mesh)
                assert parsed.mesh is not None
                vertices = parsed.mesh.vertices.astype(np.float64)
                cluster_points = np.stack([
                    cluster.transform_link[:3, 3]
                    for cluster in parsed.skin_clusters
                ])

                np.testing.assert_allclose(
                    parsed.mesh.geometry_to_world,
                    expected_model_rotation,
                    atol=1e-7,
                )
                expected_min, expected_max = self.EXPECTED_BOUNDS[hand_name]
                np.testing.assert_allclose(
                    vertices.min(axis=0), expected_min, atol=2e-6)
                np.testing.assert_allclose(
                    vertices.max(axis=0), expected_max, atol=2e-6)

                mesh_extent = np.ptp(vertices, axis=0)
                bone_extent = np.ptp(cluster_points, axis=0)
                self.assertEqual(int(np.argmax(mesh_extent)), 1)
                self.assertEqual(int(np.argmax(bone_extent)), 1)
                inside = np.all(
                    (cluster_points >= vertices.min(axis=0) - 1e-5)
                    & (cluster_points <= vertices.max(axis=0) + 1e-5),
                    axis=1,
                )
                self.assertEqual(int(inside.sum()), 26)
                self.assertEqual(len(inside), 26)

    def test_hand_loader_normalizes_mesh_and_bind_skeleton_together(self) -> None:
        for hand_name, path in HAND_PATHS.items():
            with self.subTest(hand=hand_name):
                parsed = extract_rig_data(path)
                loaded = load_rigged_mesh(path, target_scale=1.0)
                bone_positions, _ = (
                    loaded.skeleton.compute_bone_positions_rotations(None)
                )
                name_by_fbx_id = {
                    bone.fbx_id: bone.name for bone in parsed.bones
                }
                cluster_points = np.stack([
                    bone_positions[loaded.skeleton.bone_index(
                        name_by_fbx_id[cluster.bone_fbx_id]
                    )]
                    for cluster in parsed.skin_clusters
                ])
                mesh_min = loaded.vertices.min(axis=0)
                mesh_max = loaded.vertices.max(axis=0)
                inside = np.all(
                    (cluster_points >= mesh_min - 1e-5)
                    & (cluster_points <= mesh_max + 1e-5),
                    axis=1,
                )

                self.assertEqual(int(np.argmax(mesh_max - mesh_min)), 1)
                self.assertEqual(int(np.argmax(np.ptp(cluster_points, axis=0))), 1)
                self.assertEqual(int(inside.sum()), 26)
                self.assertEqual(len(inside), 26)


if __name__ == "__main__":
    unittest.main(verbosity=2)
