"""Regression coverage for affine bone-local ellipsoid centers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids  # noqa: E402
from rig_ingest import world_to_bone_local_entries  # noqa: E402
from skeleton import Bone, Pose, Skeleton, mat4_compose, quat_multiply  # noqa: E402


def _axis_angle(axis: list[float], degrees: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    half = np.radians(degrees) * 0.5
    return np.r_[direction * np.sin(half), np.cos(half)]


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (np.asarray(transform, dtype=np.float64)
            @ np.r_[np.asarray(point, dtype=np.float64), 1.0])[:3]


class AffineCenterRegressionTest(unittest.TestCase):
    def test_public_world_to_bone_local_entries_preserves_scaled_center(self) -> None:
        bone_rotation = _axis_angle([0.2, 0.7, -0.3], 37.0)
        bone_matrix = mat4_compose(
            np.array([4.0, -2.0, 0.5]),
            bone_rotation,
            np.array([2.0, 0.5, 1.5]),
        )
        expected_local_center = np.array([0.25, -0.8, 1.2])
        expected_local_rotation = _axis_angle([1.0, 0.3, 0.1], -21.0)
        world_center = _transform_point(bone_matrix, expected_local_center)
        world_rotation = quat_multiply(bone_rotation, expected_local_rotation)
        radii = np.array([[0.3, 0.2, 0.1]], dtype=np.float64)
        source = [{
            "id": 17,
            "name": "Sphere_ScaledBone_0",
            "bone_index": 0,
            "bone": "ScaledBone",
        }]

        entries = world_to_bone_local_entries(
            world_centers=world_center[None, :],
            world_radii=radii,
            world_rotations=np.asarray(world_rotation)[None, :],
            bone_assignments=np.array([0], dtype=np.int32),
            rig={"bones": [{
                "name": "ScaledBone",
                "currentMatrix": bone_matrix.tolist(),
            }]},
            source_entries=source,
        )

        self.assertEqual(entries[0]["id"], 17)
        np.testing.assert_allclose(
            entries[0]["local_center"], expected_local_center, atol=2.0e-6)
        np.testing.assert_allclose(entries[0]["radii"], radii[0], atol=1.0e-7)
        local_rotation = np.asarray(entries[0]["local_rotation"], dtype=np.float64)
        self.assertAlmostEqual(
            abs(float(np.dot(local_rotation, expected_local_rotation))),
            1.0,
            places=6,
        )
        self.assertEqual(entries[0]["attachment_bone_indices"], [0])
        np.testing.assert_allclose(entries[0]["attachment_weights"], [1.0])

    def test_blended_attachment_center_roundtrip_uses_full_affines(self) -> None:
        bind = np.stack([
            mat4_compose(
                np.array([1.0, -2.0, 0.5]),
                _axis_angle([0.0, 0.0, 1.0], 20.0),
                np.array([2.0, 0.75, 1.25]),
            ),
            mat4_compose(
                np.array([-0.5, 1.0, 2.0]),
                _axis_angle([0.0, 1.0, 0.0], -15.0),
                np.array([0.6, 1.4, 1.8]),
            ),
        ])
        posed = np.stack([
            mat4_compose(
                np.array([1.5, -1.8, 0.2]),
                _axis_angle([1.0, 0.0, 0.0], 25.0),
                np.array([1.2, 0.8, 1.6]),
            ),
            mat4_compose(
                np.array([-0.3, 1.4, 1.7]),
                _axis_angle([0.0, 0.0, 1.0], -30.0),
                np.array([0.9, 2.0, 0.7]),
            ),
        ])
        skeleton = Skeleton([
            Bone(
                name=f"Bone{i}",
                index=i,
                parent_index=-1,
                local_bind_transform=bind[i],
                inverse_bind_matrix=np.linalg.inv(bind[i]),
            )
            for i in range(2)
        ])
        pose = Pose(name="scaled pose", bone_locals={0: posed[0], 1: posed[1]})
        local = BoneLocalEllipsoids(
            local_centers=np.array([[0.2, -0.35, 0.6]], dtype=np.float32),
            local_radii=np.array([[0.12, 0.08, 0.05]], dtype=np.float32),
            local_rotations=np.array([
                _axis_angle([0.3, 1.0, -0.2], 18.0),
            ], dtype=np.float32),
            bone_assignments=np.array([0], dtype=np.int32),
            attachment_joints=np.array([[0, 1, -1, -1]], dtype=np.int32),
            attachment_weights=np.array([[0.3, 0.7, 0.0, 0.0]], dtype=np.float32),
        )
        mapper = BoneEllipsoidMapper(skeleton)

        world_centers, world_radii, world_rotations = mapper.local_to_world_np(
            local, pose=pose)
        bind_center = _transform_point(bind[0], local.local_centers[0])
        expected_world_center = (
            0.3 * _transform_point(posed[0] @ np.linalg.inv(bind[0]), bind_center)
            + 0.7 * _transform_point(posed[1] @ np.linalg.inv(bind[1]), bind_center)
        )
        np.testing.assert_allclose(
            world_centers[0], expected_world_center, rtol=2.0e-6, atol=2.0e-6)

        recovered = mapper.world_to_local(
            world_centers,
            world_radii,
            world_rotations,
            local.bone_assignments,
            pose=pose,
            attachment_joints=local.attachment_joints,
            attachment_weights=local.attachment_weights,
        )
        np.testing.assert_allclose(
            recovered.local_centers, local.local_centers, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(recovered.local_radii, local.local_radii, atol=0.0)
        dots = np.abs(np.sum(
            recovered.local_rotations * local.local_rotations, axis=1))
        np.testing.assert_allclose(dots, np.ones_like(dots), atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
