"""Bounded, non-destructive joint rotations relative to the local bind pose."""

from __future__ import annotations

import numpy as np

from joint_dof import joint_rotation_frames
from skeleton import Pose, Skeleton


def _rotation(angles):
    """Local XYZ angles in degrees, composed as Rz @ Ry @ Rx."""
    x, y, z = np.radians(angles)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def joint_pose_angles(skeleton: Skeleton, pose: Pose | None) -> np.ndarray:
    """Extract bind-relative XYZ angles without including scale or shear."""
    binds = np.asarray([b.local_bind_transform for b in skeleton.bones])
    if not len(binds):
        return np.empty((0, 3))
    locals_ = np.asarray([
        pose.bone_locals.get(i, b.local_bind_transform) if pose is not None
        else b.local_bind_transform for i, b in enumerate(skeleton.bones)
    ])
    reference = joint_rotation_frames(binds).astype(np.float64)
    current = joint_rotation_frames(locals_).astype(np.float64)
    relative = reference.transpose(0, 2, 1) @ current
    cy = np.hypot(relative[:, 0, 0], relative[:, 1, 0])
    angles = np.column_stack((
        np.arctan2(relative[:, 2, 1], relative[:, 2, 2]),
        np.arctan2(-relative[:, 2, 0], cy),
        np.arctan2(relative[:, 1, 0], relative[:, 0, 0]),
    ))
    singular = cy < 1e-6
    angles[singular, 0] = np.arctan2(-relative[singular, 1, 2], relative[singular, 1, 1])
    angles[singular, 2] = 0.0
    degrees = np.degrees(angles)
    degrees[np.abs(degrees) < 1e-5] = 0.0
    return degrees


class JointPoseEdits:
    """A sparse edit layer over one source frame; imported poses stay immutable.

    Cache authored Euler values so crossing 90 degrees does not switch Euler
    branches while dragging. Only the edited axis is clamped: selecting a joint
    or editing a different axis must not silently alter an imported animation.
    """

    def __init__(self, skeleton: Skeleton, source: Pose):
        self.skeleton = skeleton
        self.source = source
        self.angles = joint_pose_angles(skeleton, source)
        self.locals: dict[int, np.ndarray] = {}

    def set_axis(self, index: int, axis: int, value: float) -> bool:
        if not 0 <= index < self.skeleton.num_bones or axis not in (0, 1, 2):
            return False
        if not np.isfinite(value):
            return False
        bone = self.skeleton.bones[index]
        dof = bone.joint_dof
        if not dof.enabled[axis] or dof.maximum[axis] - dof.minimum[axis] <= 1e-6:
            return False
        value = float(np.clip(value, dof.minimum[axis], dof.maximum[axis]))
        if abs(value - self.angles[index, axis]) <= 1e-7:
            return False
        self.angles[index, axis] = value
        original = np.asarray(self.source.bone_locals.get(index, bone.local_bind_transform), dtype=np.float64)
        reference, current = joint_rotation_frames(np.array([bone.local_bind_transform, original])).astype(np.float64)
        # Polar stretch retains source scale, reflection and shear, as well as
        # the local translation. Rotating never changes the bone's length.
        stretch = current.T @ original[:3, :3]
        local = original.copy()
        local[:3, :3] = reference @ _rotation(self.angles[index]) @ stretch
        self.locals[index] = local
        return True

    def reset_joint(self, index: int) -> bool:
        if index not in self.locals:
            return False
        del self.locals[index]
        self.angles[index] = joint_pose_angles(self.skeleton, self.source)[index]
        return True

    def apply(self) -> Pose:
        if not self.locals:
            return self.source
        locals_ = dict(self.source.bone_locals)
        locals_.update(self.locals)
        return Pose(name=f"{self.source.name} (edited)", bone_locals=locals_)
