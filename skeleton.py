"""
skeleton.py — Bone hierarchy, poses, and forward kinematics.

Pure data model + math helpers. No I/O, no UI.

Exports used by other modules:
  - Bone, Pose, Skeleton
  - mat4_compose, mat4_decompose
  - quat_multiply, quat_inverse, quat_rotate, quat_from_matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  Quaternion helpers  (x, y, z, w)  — matches glTF / Warp convention
# ═════════════════════════════════════════════════════════════════════════════

def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (x,y,z,w) quaternions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    q = np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q /= n
    return q


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternion) of (x,y,z,w)."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-D vector *v* by quaternion *q* (x,y,z,w)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x, y, z, w = q
    # t = 2 * cross(q_xyz, v)
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Extract rotation quaternion (x,y,z,w) from a 3×3 or 4×4 matrix."""
    R = np.asarray(m, dtype=np.float64)[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q /= n
    return q


# ═════════════════════════════════════════════════════════════════════════════
#  4×4 matrix compose / decompose
# ═════════════════════════════════════════════════════════════════════════════

def mat4_compose(
    translation: np.ndarray,
    rotation_xyzw: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Build a 4×4 matrix from T, R (quaternion x,y,z,w), S."""
    t = np.asarray(translation, dtype=np.float64)
    q = np.asarray(rotation_xyzw, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)

    # Quaternion → 3×3
    n = np.linalg.norm(q)
    if n > 1e-12:
        q /= n
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1 - 2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1 - 2*(x*x+y*y)],
    ], dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R * s[np.newaxis, :]  # columns scaled
    M[:3, 3] = t
    return M


def mat4_decompose(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4×4 matrix into (translation, rotation_xyzw, scale).

    Assumes no shear.
    """
    m = np.asarray(m, dtype=np.float64)
    t = m[:3, 3].copy()

    # Extract scale from column lengths
    sx = np.linalg.norm(m[:3, 0])
    sy = np.linalg.norm(m[:3, 1])
    sz = np.linalg.norm(m[:3, 2])
    s = np.array([sx, sy, sz], dtype=np.float64)

    # Remove scale to get pure rotation
    R = m[:3, :3].copy()
    if sx > 1e-12:
        R[:, 0] /= sx
    if sy > 1e-12:
        R[:, 1] /= sy
    if sz > 1e-12:
        R[:, 2] /= sz

    # Handle negative determinant (reflection)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
        s[0] *= -1

    q = quat_from_matrix(R)
    return t, q, s


# ═════════════════════════════════════════════════════════════════════════════
#  Bone
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Bone:
    """Single bone in a skeleton.

    Attributes
    ----------
    name : str
    index : int
        Position in skeleton's bone list.
    parent_index : int
        -1 for root bones.
    local_bind_transform : (4,4) float64
        Local transform relative to parent in bind pose.
    inverse_bind_matrix : (4,4) float64
        Transforms a point from world space to this bone's local space
        in the bind pose.
    """
    name: str
    index: int
    parent_index: int = -1
    local_bind_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64),
    )
    inverse_bind_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Pose
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Pose:
    """A pose = per-bone local-transform overrides.

    bone_locals maps bone *index* → 4×4 local transform.
    Bones not in the dict keep their bind-pose local transform.
    """
    name: str = "unnamed"
    bone_locals: Dict[int, np.ndarray] = field(default_factory=dict)

    @classmethod
    def t_pose(cls) -> Pose:
        """The bind / T-pose (no overrides)."""
        return cls(name="T-Pose")


# ═════════════════════════════════════════════════════════════════════════════
#  Skeleton
# ═════════════════════════════════════════════════════════════════════════════

class Skeleton:
    """Ordered bone hierarchy with forward kinematics.

    Bone indices are stable and may come from external runtimes such as Unity.
    Parent indices do not have to be topologically sorted.
    """

    def __init__(self, bones: List[Bone]):
        self.bones = bones
        self._name_to_idx: Dict[str, int] = {b.name: b.index for b in bones}

    @property
    def num_bones(self) -> int:
        return len(self.bones)

    def bone_index(self, name: str) -> int:
        return self._name_to_idx[name]

    # ── Forward kinematics ───────────────────────────────────────────

    def compute_world_transforms(
        self,
        pose: Pose | None = None,
    ) -> np.ndarray:
        """Compute world-space 4×4 transforms for every bone.

        Parameters
        ----------
        pose : Pose, optional
            If None or T-pose, uses bind-pose locals.

        Returns
        -------
        (B, 4, 4) float64
        """
        B = self.num_bones
        world = np.zeros((B, 4, 4), dtype=np.float64)
        done = np.zeros(B, dtype=bool)
        visiting = np.zeros(B, dtype=bool)

        def _local_for(i: int) -> np.ndarray:
            bone = self.bones[i]
            # Local transform: pose override or bind default
            if pose is not None and i in pose.bone_locals:
                return pose.bone_locals[i].astype(np.float64)
            return bone.local_bind_transform.astype(np.float64)

        def _compute(i: int) -> np.ndarray:
            if done[i]:
                return world[i]
            if visiting[i]:
                # Malformed/cyclic hierarchy: keep the bone usable as a root
                # instead of poisoning the whole pose.
                world[i] = _local_for(i)
                done[i] = True
                return world[i]
            visiting[i] = True
            bone = self.bones[i]
            local = _local_for(i)
            pi = int(bone.parent_index)
            if pi < 0 or pi >= B or pi == i:
                world[i] = local
            else:
                world[i] = _compute(pi) @ local
            visiting[i] = False
            done[i] = True
            return world[i]

        for i in range(B):
            _compute(i)

        return world

    def compute_skin_matrices(
        self,
        pose: Pose | None = None,
    ) -> np.ndarray:
        """Skinning matrices: world_transform @ inverse_bind per bone.

        These transform bind-pose vertices to their posed positions.

        Returns
        -------
        (B, 4, 4) float64
        """
        world = self.compute_world_transforms(pose)
        inv_bind = np.stack(
            [b.inverse_bind_matrix.astype(np.float64) for b in self.bones],
        )
        # skin[i] = world[i] @ inv_bind[i]
        return np.einsum('bij,bjk->bik', world, inv_bind)

    def compute_bone_positions_rotations(
        self,
        pose: Pose | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Joint positions and rotations for a given pose.

        Returns
        -------
        positions : (B, 3) float32
        rotations : (B, 4) float32 — quaternions (x,y,z,w)
        """
        world = self.compute_world_transforms(pose)
        positions = world[:, :3, 3].astype(np.float32)
        rotations = np.zeros((self.num_bones, 4), dtype=np.float32)
        for i in range(self.num_bones):
            rotations[i] = quat_from_matrix(world[i]).astype(np.float32)
        return positions, rotations

    def get_bind_positions(self) -> np.ndarray:
        """Joint positions in bind pose — (B, 3) float32."""
        return self.compute_bone_positions_rotations(None)[0]
