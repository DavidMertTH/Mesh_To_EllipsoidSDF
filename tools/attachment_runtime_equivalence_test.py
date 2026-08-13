"""Regression test for Python/Unity smooth ellipsoid attachments."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import (  # noqa: E402
    BoneLocalEllipsoids,
    apply_attachment_parameter_transform,
    attachment_parameter_transform,
)
from skeleton import quat_inverse, quat_multiply  # noqa: E402


def _normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1.0e-12)


def _axis_angle(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = np.radians(degrees) * 0.5
    return _normalize(np.r_[axis * np.sin(half), np.cos(half)])


def _rotate(q: np.ndarray, value: np.ndarray) -> np.ndarray:
    q = _normalize(q)
    xyz = q[:3]
    t = 2.0 * np.cross(xyz, value)
    return value + q[3] * t + np.cross(xyz, t)


def _blend_quaternions(quaternions, weights) -> np.ndarray:
    reference = _normalize(quaternions[0])
    result = np.zeros(4, dtype=np.float64)
    for q, weight in zip(quaternions, weights):
        q = _normalize(q)
        if np.dot(reference, q) < 0.0:
            q = -q
        result += float(weight) * q
    return _normalize(result)


def _unity_reference(
    local: BoneLocalEllipsoids,
    bind_positions: np.ndarray,
    bind_rotations: np.ndarray,
    pose_positions: np.ndarray,
    pose_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent translation of EllipSDFMorphDriver.TryApplySmoothAttachment."""
    centers = np.zeros_like(local.local_centers, dtype=np.float64)
    rotations = np.zeros_like(local.local_rotations, dtype=np.float64)
    for i in range(local.num_ellipsoids):
        primary = int(local.bone_assignments[i])
        base_center = (
            bind_positions[primary]
            + _rotate(bind_rotations[primary], local.local_centers[i])
        )
        joints = local.attachment_joints[i]
        weights = local.attachment_weights[i]
        valid = (joints >= 0) & (weights > 1.0e-8)
        joints = joints[valid]
        weights = weights[valid].astype(np.float64)
        weights /= np.sum(weights)

        delta_quaternions = []
        for joint, weight in zip(joints, weights):
            joint = int(joint)
            delta = _normalize(quat_multiply(
                pose_rotations[joint], quat_inverse(bind_rotations[joint])))
            centers[i] += float(weight) * (
                pose_positions[joint]
                + _rotate(delta, base_center - bind_positions[joint])
            )
            delta_quaternions.append(delta)
        blended_delta = _blend_quaternions(delta_quaternions, weights)
        rotations[i] = _normalize(quat_multiply(
            quat_multiply(blended_delta, bind_rotations[primary]),
            local.local_rotations[i]))
    return centers.astype(np.float32), rotations.astype(np.float32)


def main() -> None:
    bind_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.4, 0.1, 0.0],
        [0.8, 0.15, 0.05],
    ], dtype=np.float32)
    bind_rotations = np.array([
        _axis_angle([0, 0, 1], 5),
        _axis_angle([0, 1, 0], -12),
        _axis_angle([1, 0, 0], 18),
    ], dtype=np.float32)
    pose_positions = np.array([
        [0.05, -0.02, 0.01],
        [0.39, 0.16, -0.03],
        [0.74, 0.27, 0.12],
    ], dtype=np.float32)
    pose_rotations = np.array([
        quat_multiply(_axis_angle([0, 0, 1], 22), bind_rotations[0]),
        quat_multiply(_axis_angle([0, 1, 0], 31), bind_rotations[1]),
        quat_multiply(_axis_angle([1, 0, 0], -27), bind_rotations[2]),
    ], dtype=np.float32)
    local = BoneLocalEllipsoids(
        local_centers=np.array([
            [0.13, 0.02, -0.01],
            [0.17, -0.04, 0.06],
            [0.08, 0.03, 0.02],
        ], dtype=np.float32),
        local_radii=np.array([
            [0.1, 0.08, 0.06],
            [0.12, 0.07, 0.05],
            [0.09, 0.06, 0.04],
        ], dtype=np.float32),
        local_rotations=np.array([
            _axis_angle([1, 1, 0], 11),
            _axis_angle([0, 1, 1], -19),
            _axis_angle([1, 0, 1], 7),
        ], dtype=np.float32),
        bone_assignments=np.array([0, 1, 2], dtype=np.int32),
        attachment_joints=np.array([
            [0, -1, -1, -1],
            [0, 1, 2, -1],
            [1, 2, -1, -1],
        ], dtype=np.int32),
        attachment_weights=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.15, 0.65, 0.20, 0.0],
            [0.35, 0.65, 0.0, 0.0],
        ], dtype=np.float32),
    )

    linear, offset, prefix = attachment_parameter_transform(
        local, bind_positions, bind_rotations, pose_positions, pose_rotations)
    python_centers, python_rotations = apply_attachment_parameter_transform(
        local.local_centers, local.local_rotations, linear, offset, prefix)
    unity_centers, unity_rotations = _unity_reference(
        local, bind_positions, bind_rotations, pose_positions, pose_rotations)

    np.testing.assert_allclose(
        python_centers, unity_centers, rtol=2.0e-6, atol=2.0e-6)
    dots = np.abs(np.sum(python_rotations * unity_rotations, axis=1))
    np.testing.assert_allclose(dots, np.ones(len(dots)), atol=2.0e-6)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
