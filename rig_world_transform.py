"""Vectorized bone-local → world mapping for live previews.

Produces the same result as ``BoneEllipsoidMapper.local_to_world_np`` (full
affine centre skinning and blended attachment rotations), but computes the
per-pose map once for all primitives without Python loops.  Every primitive
set that shares the base's bone assignments and attachments (base, morph-key
blend, joint functions) can reuse one map per frame.
"""

from __future__ import annotations

import numpy as np

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from native_morph import _multiply_quats
from skeleton import Pose, Skeleton, mat4_decompose


def _normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    ok = (norm > 1e-12) & np.isfinite(norm)
    identity = np.zeros_like(q)
    identity[..., 3] = 1
    return np.where(ok, q / np.where(ok, norm, 1), identity)


class RigWorldTransform:
    """Per-pose local→world maps for one primitive layout on one skeleton."""

    def __init__(self, skeleton: Skeleton, layout: BoneLocalEllipsoids):
        self.skeleton, self.layout = skeleton, layout
        bones, count = skeleton.num_bones, layout.num_ellipsoids
        primary = np.asarray(layout.bone_assignments, dtype=int)
        if np.any(primary < 0) or np.any(primary >= bones):
            raise ValueError("primitive references a missing bone")
        joints = np.repeat(primary[:, None], 1, axis=1)
        weights = np.ones((count, 1))
        if (layout.attachment_joints is not None and layout.attachment_weights is not None
                and layout.attachment_joints.shape[0] == count
                and layout.attachment_joints.shape == layout.attachment_weights.shape):
            raw_j = np.asarray(layout.attachment_joints, dtype=int)
            raw_w = np.asarray(layout.attachment_weights, dtype=np.float64)
            joints = np.zeros_like(raw_j)
            weights = np.zeros_like(raw_w)
            for i in range(count):
                valid = (raw_w[i] > 1e-8) & (raw_j[i] >= 0) & (raw_j[i] < bones)
                j, w = raw_j[i][valid], raw_w[i][valid]
                if len(j) == 0 or w.sum() <= 1e-12:
                    j, w = np.array([primary[i]]), np.array([1.0])
                joints[i, :len(j)] = j
                weights[i, :len(j)] = w / w.sum()
        self._primary, self._joints, self._weights = primary, joints, weights
        self._bind_world = skeleton.compute_world_transforms(None)
        self._bind_q = np.array([mat4_decompose(m)[1] for m in self._bind_world])

    def matches(self, skeleton, layout) -> bool:
        return skeleton is self.skeleton and layout is self.layout

    def pose_map(self, pose: Pose | None):
        """``(linear (N,3,3), offset (N,3), rotation_prefix (N,4))`` for *pose*."""
        bind, posed = self._bind_world, self.skeleton.compute_world_transforms(pose)
        try:
            delta_linear = np.transpose(np.linalg.solve(
                np.transpose(bind[:, :3, :3], (0, 2, 1)), np.transpose(posed[:, :3, :3], (0, 2, 1))), (0, 2, 1))
        except np.linalg.LinAlgError:
            delta_linear = posed[:, :3, :3] @ np.linalg.pinv(bind[:, :3, :3])
        delta_offset = posed[:, :3, 3] - np.einsum("bij,bj->bi", delta_linear, bind[:, :3, 3])
        pose_q = np.array([mat4_decompose(m)[1] for m in posed])
        bind_inv = self._bind_q * [-1, -1, -1, 1]
        delta_q = _normalize(_multiply_quats(pose_q, bind_inv / np.linalg.norm(self._bind_q, axis=1, keepdims=True) ** 2))

        w = self._weights[..., None]
        blended_linear = np.sum(w[..., None] * delta_linear[self._joints], axis=1)            # (N,3,3)
        blended_offset = np.sum(w * delta_offset[self._joints], axis=1)                      # (N,3)
        primary_bind = bind[self._primary]
        linear = blended_linear @ primary_bind[:, :3, :3]
        offset = np.einsum("nij,nj->ni", blended_linear, primary_bind[:, :3, 3]) + blended_offset

        quats = _normalize(delta_q[self._joints])                                           # (N,K,4)
        reference = quats[:, :1]
        sign = np.where(np.sum(reference * quats, axis=-1, keepdims=True) < 0, -1.0, 1.0)
        blended_q = _normalize(np.sum(w * sign * quats, axis=1))
        prefix = _normalize(_multiply_quats(blended_q, self._bind_q[self._primary]))
        return linear, offset, prefix

    @staticmethod
    def apply(local: BoneLocalEllipsoids, pose_map):
        """World ``(centers, radii, rotations)`` of *local* under *pose_map*."""
        linear, offset, prefix = pose_map
        centers = np.einsum("nij,nj->ni", linear, np.asarray(local.local_centers, dtype=np.float64)) + offset
        rotations = _normalize(_multiply_quats(prefix, np.asarray(local.local_rotations, dtype=np.float64)))
        return (centers.astype(np.float32), np.asarray(local.local_radii, dtype=np.float32).copy(),
                rotations.astype(np.float32))


def reference_world(skeleton: Skeleton, local: BoneLocalEllipsoids, pose: Pose | None):
    """The mapper's loop implementation, for tests and fallbacks."""
    return BoneEllipsoidMapper(skeleton).local_to_world_np(local, pose)
