"""
bone_ellipsoid_mapper.py — Map ellipsoids to bones and handle bone-local parameters.

Core concept:
  Ellipsoid parameters (center, radii, rotation) are stored in **bone-local space**.
  For any given pose, bone transforms map them to world space.
  Training gradients flow through the bone transform back to the local parameters.

  world_center   = bone_rot ⊗ local_center + bone_pos
  world_rotation = bone_rot ⊗ local_rotation
  world_radii    = local_radii   (rigid bones — no scaling)

Workflow:
  1. Optimise ellipsoids in T-pose world space (existing pipeline)
  2. assign_to_bones() — determine which bone each ellipsoid belongs to
  3. world_to_local()  — convert world params → bone-local params
  4. For each new pose:
     a. local_to_world() — get world params from bone-local + pose transforms
     b. Compute SDF loss, backprop to bone-local params
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from skeleton import (
    Skeleton, Pose,
    quat_multiply, quat_inverse, quat_rotate, quat_from_matrix,
    mat4_decompose,
)

try:
    import warp as wp
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False


# ── Warp kernel: bone-local → world (differentiable) ────────────────────────

if _HAS_WARP:
    @wp.kernel
    def _local_to_world_kernel(
        local_centers: wp.array(dtype=wp.vec3),
        local_radii: wp.array(dtype=wp.vec3),
        local_rot_flat: wp.array(dtype=wp.float32),
        bone_assignments: wp.array(dtype=wp.int32),
        bone_positions: wp.array(dtype=wp.vec3),
        bone_rotations: wp.array(dtype=wp.quat),
        world_centers: wp.array(dtype=wp.vec3),
        world_radii: wp.array(dtype=wp.vec3),
        world_rot_flat: wp.array(dtype=wp.float32),
    ):
        """Transform bone-local ellipsoid params to world space.

        Differentiable w.r.t. local_centers, local_radii, local_rot_flat.
        Bone transforms are treated as constants (no grad needed).
        """
        tid = wp.tid()
        bone_idx = bone_assignments[tid]

        b_pos = bone_positions[bone_idx]
        b_rot = bone_rotations[bone_idx]

        # ── Center: world = bone_rot ⊗ local + bone_pos ──
        lc = local_centers[tid]
        wc = wp.quat_rotate(b_rot, lc) + b_pos
        world_centers[tid] = wc

        # ── Radii: unchanged (rigid transforms) ──
        world_radii[tid] = local_radii[tid]

        # ── Rotation: world = bone_rot ⊗ local_rot ──
        base = tid * 4
        lq = wp.normalize(wp.quat(
            local_rot_flat[base + 0],
            local_rot_flat[base + 1],
            local_rot_flat[base + 2],
            local_rot_flat[base + 3],
        ))
        wq = wp.mul(b_rot, lq)

        world_rot_flat[base + 0] = wq[0]
        world_rot_flat[base + 1] = wq[1]
        world_rot_flat[base + 2] = wq[2]
        world_rot_flat[base + 3] = wq[3]


# ── Data container ───────────────────────────────────────────────────────────

@dataclass
class BoneLocalEllipsoids:
    """Ellipsoid parameters in bone-local space.

    These are the **trainable** parameters that persist across poses.
    """
    local_centers: np.ndarray     # (N, 3) float32 — offset from bone origin
    local_radii: np.ndarray       # (N, 3) float32 — semi-axis lengths
    local_rotations: np.ndarray   # (N, 4) float32 — quaternion (x,y,z,w)
    bone_assignments: np.ndarray  # (N,) int32     — which bone each belongs to
    attachment_joints: np.ndarray | None = None
    attachment_weights: np.ndarray | None = None
    num_ellipsoids: int = 0

    def __post_init__(self):
        self.num_ellipsoids = len(self.local_centers)
        if self.attachment_joints is None or self.attachment_weights is None:
            self.attachment_joints = None
            self.attachment_weights = None
            return
        joints = np.asarray(self.attachment_joints, dtype=np.int32)
        weights = np.asarray(self.attachment_weights, dtype=np.float32)
        if joints.shape != weights.shape or joints.shape[0] != self.num_ellipsoids:
            self.attachment_joints = None
            self.attachment_weights = None
            return
        sums = np.sum(weights, axis=1, keepdims=True)
        valid = sums[:, 0] > 1.0e-8
        weights = weights.copy()
        weights[valid] /= sums[valid]
        self.attachment_joints = joints
        self.attachment_weights = weights


# ── Mapper class ─────────────────────────────────────────────────────────────

def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1.0e-12:
        q = q / n
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1.0e-12 and np.isfinite(n):
        return q / n
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _blend_quaternions(quats: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not quats:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    ref = _normalize_quat(quats[0])
    acc = np.zeros(4, dtype=np.float64)
    for q, w in zip(quats, weights):
        qq = _normalize_quat(q)
        if np.dot(ref, qq) < 0.0:
            qq = -qq
        acc += float(w) * qq
    return _normalize_quat(acc)


def attachment_parameter_transform(
    bone_local: "BoneLocalEllipsoids",
    bind_positions: np.ndarray,
    bind_rotations: np.ndarray,
    pose_positions: np.ndarray,
    pose_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the fixed local-to-world map for one rig pose.

    For a fixed pose, attachment skinning is affine in each local center and a
    constant quaternion prefix for each local rotation.  Returning that map lets
    the optimizer train the persistent bone-local parameters directly while the
    SDF loss is still evaluated in world space.
    """
    bind_t = np.asarray(bind_positions, dtype=np.float64).reshape(-1, 3)
    bind_q = np.asarray(bind_rotations, dtype=np.float64).reshape(-1, 4)
    pose_t = np.asarray(pose_positions, dtype=np.float64).reshape(-1, 3)
    pose_q = np.asarray(pose_rotations, dtype=np.float64).reshape(-1, 4)
    if not (len(bind_t) == len(bind_q) == len(pose_t) == len(pose_q)):
        raise ValueError("bind and pose transform arrays must have equal length")

    n_bones = len(bind_t)
    n_ellipsoids = bone_local.num_ellipsoids
    linear = np.zeros((n_ellipsoids, 3, 3), dtype=np.float32)
    offset = np.zeros((n_ellipsoids, 3), dtype=np.float32)
    rotation_prefix = np.zeros((n_ellipsoids, 4), dtype=np.float32)
    use_attachments = (
        bone_local.attachment_joints is not None
        and bone_local.attachment_weights is not None
        and bone_local.attachment_joints.shape[0] == n_ellipsoids
        and bone_local.attachment_joints.shape == bone_local.attachment_weights.shape
    )

    for i in range(n_ellipsoids):
        primary = int(bone_local.bone_assignments[i])
        if primary < 0 or primary >= n_bones:
            raise ValueError(f"ellipsoid {i} references invalid primary bone {primary}")

        if use_attachments:
            joints = np.asarray(bone_local.attachment_joints[i], dtype=np.int32)
            weights = np.asarray(bone_local.attachment_weights[i], dtype=np.float64)
            valid = (weights > 1.0e-8) & (joints >= 0) & (joints < n_bones)
            joints = joints[valid]
            weights = weights[valid]
        else:
            joints = np.zeros(0, dtype=np.int32)
            weights = np.zeros(0, dtype=np.float64)
        if len(joints) == 0 or float(weights.sum()) <= 1.0e-12:
            joints = np.array([primary], dtype=np.int32)
            weights = np.array([1.0], dtype=np.float64)
        else:
            weights = weights / float(weights.sum())

        blended_linear = np.zeros((3, 3), dtype=np.float64)
        blended_offset = np.zeros(3, dtype=np.float64)
        delta_quats: list[np.ndarray] = []
        delta_weights: list[float] = []
        for joint, weight in zip(joints, weights):
            joint = int(joint)
            bind_r = _quat_to_matrix(bind_q[joint])
            pose_r = _quat_to_matrix(pose_q[joint])
            delta_r = pose_r @ bind_r.T
            blended_linear += float(weight) * delta_r
            blended_offset += float(weight) * (
                pose_t[joint] - delta_r @ bind_t[joint])
            delta_quats.append(
                quat_multiply(pose_q[joint], quat_inverse(bind_q[joint])))
            delta_weights.append(float(weight))

        primary_bind_r = _quat_to_matrix(bind_q[primary])
        linear[i] = (blended_linear @ primary_bind_r).astype(np.float32)
        offset[i] = (
            blended_linear @ bind_t[primary] + blended_offset
        ).astype(np.float32)
        rotation_prefix[i] = _normalize_quat(quat_multiply(
            _blend_quaternions(delta_quats, delta_weights),
            bind_q[primary],
        )).astype(np.float32)

    return linear, offset, rotation_prefix


def apply_attachment_parameter_transform(
    local_centers: np.ndarray,
    local_rotations: np.ndarray,
    linear: np.ndarray,
    offset: np.ndarray,
    rotation_prefix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a map produced by :func:`attachment_parameter_transform`."""
    centers = np.asarray(local_centers, dtype=np.float64).reshape(-1, 3)
    rotations = np.asarray(local_rotations, dtype=np.float64).reshape(-1, 4)
    matrices = np.asarray(linear, dtype=np.float64).reshape(-1, 3, 3)
    translations = np.asarray(offset, dtype=np.float64).reshape(-1, 3)
    prefixes = np.asarray(rotation_prefix, dtype=np.float64).reshape(-1, 4)
    if not (len(centers) == len(rotations) == len(matrices)
            == len(translations) == len(prefixes)):
        raise ValueError("local parameters and attachment maps must have equal length")
    world_centers = np.einsum("nij,nj->ni", matrices, centers) + translations
    world_rotations = np.array([
        _normalize_quat(quat_multiply(prefixes[i], rotations[i]))
        for i in range(len(centers))
    ], dtype=np.float32)
    return world_centers.astype(np.float32), world_rotations


def _attachment_from_scores(
    scores: np.ndarray,
    primary: int,
    *,
    max_influences: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    primary = int(primary)
    valid = np.flatnonzero(np.isfinite(scores) & (scores > 1.0e-12))
    if len(valid) == 0:
        joints = np.full(max_influences, max(0, primary), dtype=np.int32)
        weights = np.zeros(max_influences, dtype=np.float32)
        weights[0] = 1.0
        return joints, weights
    order = valid[np.argsort(scores[valid])[::-1]]
    chosen = list(order[:max_influences])
    if primary not in chosen:
        chosen[-1] = primary
    joints = np.full(max_influences, int(chosen[0]), dtype=np.int32)
    weights = np.zeros(max_influences, dtype=np.float32)
    vals = np.array([max(float(scores[j]), 0.0) for j in chosen], dtype=np.float64)
    if vals.sum() <= 1.0e-12:
        vals[:] = 0.0
        vals[0] = 1.0
    vals /= vals.sum()
    for i, (j, w) in enumerate(zip(chosen, vals)):
        joints[i] = int(j)
        weights[i] = float(w)
    return joints, weights


def _attachment_from_nearby_vertices(
    center: np.ndarray,
    mesh_vertices: np.ndarray,
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    n_bones: int,
    primary: int,
    *,
    k_nearest: int = 20,
    max_influences: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
    joints = np.asarray(skin_joints, dtype=np.int32)
    weights = np.asarray(skin_weights, dtype=np.float64)
    if len(verts) == 0 or joints.shape != weights.shape:
        return _attachment_from_scores(
            np.zeros(n_bones), primary, max_influences=max_influences)
    dists = np.linalg.norm(
        verts - np.asarray(center, dtype=np.float64)[None, :], axis=1)
    k = min(int(k_nearest), len(dists))
    nearest = np.arange(len(dists)) if k >= len(dists) else np.argpartition(dists, k)[:k]
    inv_d = 1.0 / np.maximum(dists[nearest], 1.0e-8)
    scores = np.zeros(int(n_bones), dtype=np.float64)
    for local_i, vi in enumerate(nearest):
        for slot in range(joints.shape[1]):
            bi = int(joints[vi, slot])
            if 0 <= bi < n_bones:
                scores[bi] += inv_d[local_i] * max(float(weights[vi, slot]), 0.0)
    return _attachment_from_scores(scores, primary, max_influences=max_influences)


class BoneEllipsoidMapper:
    """Maps ellipsoids to bones and converts between local / world space."""

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self._bone_local: Optional[BoneLocalEllipsoids] = None

    @property
    def bone_local(self) -> BoneLocalEllipsoids | None:
        return self._bone_local

    # ── Step 1: Assign ellipsoids to bones ───────────────────────────

    def assign_to_bones(
        self,
        world_centers: np.ndarray,
        world_radii: np.ndarray,
        world_rotations: np.ndarray,
        mesh_vertices: np.ndarray,
        skin_joints: np.ndarray,
        skin_weights: np.ndarray,
        pose: Pose | None = None,
    ) -> BoneLocalEllipsoids:
        """Assign each ellipsoid to its nearest bone and convert to bone-local space.

        Strategy: For each ellipsoid center, find the closest mesh vertices
        and use their skin weights to vote on the most influential bone.

        Parameters
        ----------
        world_centers : (N, 3) — ellipsoid centers in world space
        world_radii : (N, 3)
        world_rotations : (N, 4) — quaternions (x,y,z,w)
        mesh_vertices : (V, 3) — mesh vertices (in same pose as the ellipsoids)
        skin_joints : (V, 4) — bone indices per vertex
        skin_weights : (V, 4) — blend weights per vertex
        pose : Pose, optional — the pose the ellipsoids were optimised in (default: T-pose)

        Returns
        -------
        BoneLocalEllipsoids
        """
        N = len(world_centers)
        bone_assignments = np.zeros(N, dtype=np.int32)
        attachment_joints = np.zeros((N, 4), dtype=np.int32)
        attachment_weights = np.zeros((N, 4), dtype=np.float32)

        for i in range(N):
            c = world_centers[i].astype(np.float64)

            # Find K nearest mesh vertices
            dists = np.linalg.norm(
                mesh_vertices.astype(np.float64) - c[np.newaxis, :],
                axis=1,
            )
            K = min(20, len(dists))
            if K >= len(dists):
                nearest_idx = np.arange(len(dists))
            else:
                nearest_idx = np.argpartition(dists, K)[:K]

            # Weighted vote: closer vertices count more, weighted by skin weight
            inv_dists = 1.0 / np.maximum(dists[nearest_idx], 1e-8)

            bone_scores = np.zeros(self.skeleton.num_bones, dtype=np.float64)
            for ni in range(K):
                vi = nearest_idx[ni]
                w_dist = inv_dists[ni]
                for k in range(skin_joints.shape[1]):
                    bone_idx = skin_joints[vi, k]
                    w_skin = skin_weights[vi, k]
                    bone_scores[bone_idx] += w_dist * w_skin

            bone_assignments[i] = int(np.argmax(bone_scores))
            aj, aw = _attachment_from_scores(
                bone_scores, bone_assignments[i], max_influences=4)
            attachment_joints[i] = aj
            attachment_weights[i] = aw

        # Convert to bone-local space
        bone_local = self.world_to_local(
            world_centers, world_radii, world_rotations,
            bone_assignments, pose,
            attachment_joints=attachment_joints,
            attachment_weights=attachment_weights,
        )

        self._bone_local = bone_local
        return bone_local

    # ── World → Local ────────────────────────────────────────────────

    def world_to_local(
        self,
        world_centers: np.ndarray,
        world_radii: np.ndarray,
        world_rotations: np.ndarray,
        bone_assignments: np.ndarray,
        pose: Pose | None = None,
        attachment_joints: np.ndarray | None = None,
        attachment_weights: np.ndarray | None = None,
    ) -> BoneLocalEllipsoids:
        """Convert world-space ellipsoid params to bone-local space.

        For each ellipsoid assigned to bone B:
          local_center = inv(bone_rot) ⊗ (world_center - bone_pos)
          local_rot    = inv(bone_rot) ⊗ world_rot
          local_radii  = world_radii   (no scaling)
        """
        N = len(world_centers)
        world_transforms = self.skeleton.compute_world_transforms(pose)
        bind_transforms = self.skeleton.compute_world_transforms(None)
        if attachment_joints is None and self._bone_local is not None:
            if self._bone_local.num_ellipsoids == N:
                attachment_joints = self._bone_local.attachment_joints
                attachment_weights = self._bone_local.attachment_weights
        use_attachments = (
            attachment_joints is not None
            and attachment_weights is not None
            and np.asarray(attachment_joints).shape[0] == N
            and np.asarray(attachment_joints).shape == np.asarray(attachment_weights).shape
        )

        local_centers = np.zeros((N, 3), dtype=np.float32)
        local_radii = world_radii.copy().astype(np.float32)
        local_rotations = np.zeros((N, 4), dtype=np.float32)

        bind_t = np.zeros((self.skeleton.num_bones, 3), dtype=np.float64)
        bind_q = np.zeros((self.skeleton.num_bones, 4), dtype=np.float64)
        pose_t = np.zeros((self.skeleton.num_bones, 3), dtype=np.float64)
        pose_q = np.zeros((self.skeleton.num_bones, 4), dtype=np.float64)
        for bi in range(self.skeleton.num_bones):
            bind_t[bi], bind_q[bi], _ = mat4_decompose(bind_transforms[bi])
            pose_t[bi], pose_q[bi], _ = mat4_decompose(world_transforms[bi])

        for i in range(N):
            primary = int(bone_assignments[i])
            q_world = _normalize_quat(world_rotations[i].astype(np.float64))

            if use_attachments:
                joints = np.asarray(attachment_joints[i], dtype=np.int32)
                weights = np.asarray(attachment_weights[i], dtype=np.float64)
                valid = (weights > 1.0e-8) & (joints >= 0) & (joints < self.skeleton.num_bones)
                if not np.any(valid):
                    joints = np.array([primary], dtype=np.int32)
                    weights = np.array([1.0], dtype=np.float64)
                else:
                    joints = joints[valid]
                    weights = weights[valid]
                    weights = weights / max(float(weights.sum()), 1.0e-12)

                A = np.zeros((3, 3), dtype=np.float64)
                b = np.zeros(3, dtype=np.float64)
                delta_quats: list[np.ndarray] = []
                delta_weights: list[float] = []
                for joint, weight in zip(joints, weights):
                    rb = _quat_to_matrix(bind_q[int(joint)])
                    rp = _quat_to_matrix(pose_q[int(joint)])
                    Aj = rp @ rb.T
                    A += float(weight) * Aj
                    b += float(weight) * (pose_t[int(joint)] - Aj @ bind_t[int(joint)])
                    dq = quat_multiply(pose_q[int(joint)], quat_inverse(bind_q[int(joint)]))
                    delta_quats.append(dq)
                    delta_weights.append(float(weight))

                try:
                    bind_center = np.linalg.solve(
                        A, world_centers[i].astype(np.float64) - b)
                except np.linalg.LinAlgError:
                    bind_center = np.linalg.pinv(A) @ (
                        world_centers[i].astype(np.float64) - b)

                q_bind_primary_inv = quat_inverse(bind_q[primary])
                local_c = quat_rotate(q_bind_primary_inv, bind_center - bind_t[primary])
                blend_delta = _blend_quaternions(delta_quats, delta_weights)
                bind_world_q = quat_multiply(quat_inverse(blend_delta), q_world)
                q_local = quat_multiply(q_bind_primary_inv, bind_world_q)
            else:
                # Center: inv_rot(world_center - bone_pos)
                delta = world_centers[i].astype(np.float64) - pose_t[primary]
                q_inv = quat_inverse(pose_q[primary])
                local_c = quat_rotate(q_inv, delta)
                q_local = quat_multiply(q_inv, q_world)

            local_centers[i] = local_c.astype(np.float32)
            local_rotations[i] = _normalize_quat(q_local).astype(np.float32)

        return BoneLocalEllipsoids(
            local_centers=local_centers,
            local_radii=local_radii,
            local_rotations=local_rotations,
            bone_assignments=bone_assignments.astype(np.int32),
            attachment_joints=(None if not use_attachments
                               else np.asarray(attachment_joints, dtype=np.int32).copy()),
            attachment_weights=(None if not use_attachments
                                else np.asarray(attachment_weights, dtype=np.float32).copy()),
        )

    # ── Local → World (NumPy, for visualisation) ─────────────────────

    def local_to_world_np(
        self,
        bone_local: BoneLocalEllipsoids | None = None,
        pose: Pose | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert bone-local params to world space using NumPy.

        Returns (world_centers, world_radii, world_rotations).
        """
        if bone_local is None:
            bone_local = self._bone_local
        assert bone_local is not None, "No bone-local params. Call assign_to_bones first."

        linear, offset, rotation_prefix = self.local_to_world_parameter_transform(
            bone_local, pose=pose)
        world_centers, world_rotations = apply_attachment_parameter_transform(
            bone_local.local_centers,
            bone_local.local_rotations,
            linear,
            offset,
            rotation_prefix,
        )
        return world_centers, bone_local.local_radii.copy(), world_rotations

    def local_to_world_parameter_transform(
        self,
        bone_local: BoneLocalEllipsoids | None = None,
        pose: Pose | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the differentiable affine/quaternion map for ``pose``."""
        if bone_local is None:
            bone_local = self._bone_local
        assert bone_local is not None, "No bone-local params. Call assign_to_bones first."

        bind_world = self.skeleton.compute_world_transforms(None)
        pose_world = self.skeleton.compute_world_transforms(pose)
        bind_t = np.zeros((self.skeleton.num_bones, 3), dtype=np.float64)
        bind_q = np.zeros((self.skeleton.num_bones, 4), dtype=np.float64)
        pose_t = np.zeros((self.skeleton.num_bones, 3), dtype=np.float64)
        pose_q = np.zeros((self.skeleton.num_bones, 4), dtype=np.float64)
        for bi in range(self.skeleton.num_bones):
            bind_t[bi], bind_q[bi], _ = mat4_decompose(bind_world[bi])
            pose_t[bi], pose_q[bi], _ = mat4_decompose(pose_world[bi])
        return attachment_parameter_transform(
            bone_local, bind_t, bind_q, pose_t, pose_q)

    # ── Local → World (Warp, for differentiable training) ────────────

    def local_to_world_warp(
        self,
        local_centers_wp: "wp.array",
        local_radii_wp: "wp.array",
        local_rot_flat_wp: "wp.array",
        bone_assignments: np.ndarray,
        pose: Pose | None = None,
        device: str = "cuda",
    ) -> tuple["wp.array", "wp.array", "wp.array"]:
        """Differentiable bone-local → world transform using Warp.

        The returned arrays are part of the computation graph, so
        gradients will flow back to local_centers_wp, local_radii_wp,
        and local_rot_flat_wp.

        Returns (world_centers, world_radii, world_rot_flat) — all Warp arrays.
        """
        assert _HAS_WARP, "Warp is required for differentiable transforms."

        N = len(bone_assignments)
        world_transforms = self.skeleton.compute_world_transforms(pose)

        # Extract per-bone position and rotation
        bone_pos_np = np.zeros((self.skeleton.num_bones, 3), dtype=np.float32)
        bone_rot_np = np.zeros((self.skeleton.num_bones, 4), dtype=np.float32)

        for bi in range(self.skeleton.num_bones):
            t, q, _ = mat4_decompose(world_transforms[bi])
            bone_pos_np[bi] = t.astype(np.float32)
            bone_rot_np[bi] = q.astype(np.float32)

        # Upload constants
        wp_bone_assign = wp.array(bone_assignments.astype(np.int32),
                                  dtype=wp.int32, device=device)
        wp_bone_pos = wp.array(bone_pos_np, dtype=wp.vec3, device=device)
        wp_bone_rot = wp.array(bone_rot_np, dtype=wp.quat, device=device)

        # Allocate outputs (differentiable)
        world_centers = wp.empty(N, dtype=wp.vec3, device=device,
                                 requires_grad=True)
        world_radii = wp.empty(N, dtype=wp.vec3, device=device,
                               requires_grad=True)
        world_rot_flat = wp.empty(N * 4, dtype=wp.float32, device=device,
                                  requires_grad=True)

        wp.launch(
            _local_to_world_kernel,
            dim=N,
            inputs=[
                local_centers_wp, local_radii_wp, local_rot_flat_wp,
                wp_bone_assign, wp_bone_pos, wp_bone_rot,
                world_centers, world_radii, world_rot_flat,
            ],
            device=device,
        )

        return world_centers, world_radii, world_rot_flat

    # ── Utility ──────────────────────────────────────────────────────

    def get_bone_name(self, ellipsoid_index: int) -> str:
        """Return the bone name for a given ellipsoid."""
        if self._bone_local is None:
            return "?"
        bi = self._bone_local.bone_assignments[ellipsoid_index]
        return self.skeleton.bones[bi].name


# ── Bone-based initialization (no T-pose fit needed) ────────────────────────

def initialize_ellipsoids_from_bones(
    skeleton: Skeleton,
    vertices: np.ndarray,
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    n_ellipsoids: int = 10,
    padding: float = 1.2,
    min_radius: float = 0.02,
    influence_threshold: float = 0.01,
    min_bone_mass: float = 0.25,
    max_points_per_bone: int = 4096,
    kmeans_iters: int = 12,
    random_seed: int = 42,
) -> BoneLocalEllipsoids:
    """Create BoneLocalEllipsoids directly from the skeleton structure.

    Skips the T-pose fit + assign-to-bones workflow entirely.
    For each bone, ellipsoids are placed from the vertices it actually
    influences, weighted by skin weights. Budget is allocated by total
    influence mass per bone.

    Parameters
    ----------
    skeleton       : Skeleton — bone hierarchy
    vertices       : (V, 3) rest-pose mesh vertices
    skin_joints    : (V, K) bone indices per vertex
    skin_weights   : (V, K) blend weights per vertex
    n_ellipsoids   : total number of ellipsoids to create
    padding        : radii scale factor applied to the half-range of the point cloud
    min_radius     : minimum semi-axis length to avoid degenerate ellipsoids
    influence_threshold : ignore per-vertex bone weights smaller than this
    min_bone_mass  : ignore bones whose total influence is below this
    max_points_per_bone : cap per-bone point clouds used for clustering/fitting
    kmeans_iters   : weighted k-means refinement iterations
    random_seed    : for k-means reproducibility
    """
    rng = np.random.default_rng(random_seed)
    n_bones = skeleton.num_bones
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    skin_joints = np.asarray(skin_joints, dtype=np.int32)
    skin_weights = np.asarray(skin_weights, dtype=np.float32)

    # ── 1. Compact skin influence lists: avoids a dense B x V matrix ──
    bone_vertex_ids, bone_vertex_weights, influence_mass = _bone_influence_lists(
        skin_joints, skin_weights, n_bones,
        influence_threshold=float(influence_threshold),
    )
    influence_mass[influence_mass < float(min_bone_mass)] = 0.0

    # ── 2. Budget: how many ellipsoids per bone ──
    budget = _allocate_ellipsoid_budget(influence_mass, n_ellipsoids)

    # ── 3. Bone world transforms in T-pose ──
    world_transforms = skeleton.compute_world_transforms(None)

    all_centers:     list[np.ndarray] = []
    all_radii:       list[np.ndarray] = []
    all_rotations:   list[np.ndarray] = []
    all_assignments: list[int]        = []
    all_attachment_joints: list[np.ndarray] = []
    all_attachment_weights: list[np.ndarray] = []

    for bi in range(n_bones):
        k = int(budget[bi])
        if k == 0:
            continue

        t_bone, q_bone, _ = mat4_decompose(world_transforms[bi])
        q_inv = quat_inverse(q_bone)

        # Vertices influenced by this bone -> bone-local space.  Weights are
        # kept through clustering and fitting so soft blend regions still count.
        vert_ids = bone_vertex_ids[bi]
        bone_weights = bone_vertex_weights[bi]
        if len(vert_ids) == 0:
            continue
        bone_verts_world = vertices[vert_ids]

        cap = max(int(max_points_per_bone), k * 32)
        if cap > 0 and len(bone_verts_world) > cap:
            sample_idx = _weighted_subsample_indices(bone_weights, cap, rng)
            bone_verts_world = bone_verts_world[sample_idx]
            bone_weights = bone_weights[sample_idx]

        local_verts = _quat_rotate_many(
            q_inv, bone_verts_world.astype(np.float64) - t_bone,
        ).astype(np.float32)

        # Split into k clusters and fit one ellipsoid per cluster
        clusters = (
            [(local_verts, bone_weights)]
            if k == 1 else
            _weighted_kmeans_clusters(
                local_verts, bone_weights, k, rng, n_iter=int(kmeans_iters),
            )
        )

        for pts, w in clusters:
            center, radii, rotation = _ellipsoid_from_weighted_points(
                pts, weights=w, padding=padding, min_r=min_radius,
            )
            center_world = quat_rotate(q_bone, center.astype(np.float64)) + t_bone
            aj, aw = _attachment_from_nearby_vertices(
                center_world, vertices, skin_joints, skin_weights,
                n_bones, bi, k_nearest=20, max_influences=4)
            all_centers.append(center)
            all_radii.append(radii)
            all_rotations.append(rotation)
            all_assignments.append(bi)
            all_attachment_joints.append(aj)
            all_attachment_weights.append(aw)

    if not all_centers:
        return BoneLocalEllipsoids(
            local_centers=np.zeros((0, 3), dtype=np.float32),
            local_radii=np.zeros((0, 3), dtype=np.float32),
            local_rotations=np.zeros((0, 4), dtype=np.float32),
            bone_assignments=np.zeros((0,), dtype=np.int32),
        )

    return BoneLocalEllipsoids(
        local_centers=np.array(all_centers,     dtype=np.float32),
        local_radii=np.array(all_radii,         dtype=np.float32),
        local_rotations=np.array(all_rotations, dtype=np.float32),
        bone_assignments=np.array(all_assignments, dtype=np.int32),
        attachment_joints=np.array(all_attachment_joints, dtype=np.int32),
        attachment_weights=np.array(all_attachment_weights, dtype=np.float32),
    )


def _bone_vertex_weight_matrix(
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    n_bones: int,
    influence_threshold: float = 0.01,
) -> np.ndarray:
    """Return ``(B,V)`` influence weights accumulated from all skin slots."""
    if skin_joints.ndim != 2 or skin_weights.ndim != 2:
        raise ValueError("skin_joints and skin_weights must be (V,K) arrays.")
    if skin_joints.shape != skin_weights.shape:
        raise ValueError("skin_joints and skin_weights must have matching shapes.")

    v_count, k_count = skin_joints.shape
    out = np.zeros((int(n_bones), int(v_count)), dtype=np.float32)
    vert_idx = np.arange(v_count, dtype=np.int64)
    for k in range(k_count):
        joints = skin_joints[:, k].astype(np.int64, copy=False)
        weights = skin_weights[:, k].astype(np.float32, copy=False)
        valid = (
            (joints >= 0)
            & (joints < n_bones)
            & np.isfinite(weights)
            & (weights >= float(influence_threshold))
        )
        if np.any(valid):
            np.add.at(out, (joints[valid], vert_idx[valid]), weights[valid])
    return out


def _bone_influence_lists(
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    n_bones: int,
    influence_threshold: float = 0.01,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return compact per-bone vertex ids, weights, and total influence mass."""
    if skin_joints.ndim != 2 or skin_weights.ndim != 2:
        raise ValueError("skin_joints and skin_weights must be (V,K) arrays.")
    if skin_joints.shape != skin_weights.shape:
        raise ValueError("skin_joints and skin_weights must have matching shapes.")

    n_bones = int(n_bones)
    v_count, k_count = skin_joints.shape
    empty_i = [np.empty(0, dtype=np.int32) for _ in range(n_bones)]
    empty_w = [np.empty(0, dtype=np.float32) for _ in range(n_bones)]
    mass = np.zeros(n_bones, dtype=np.float32)
    if n_bones <= 0 or v_count == 0 or k_count == 0:
        return empty_i, empty_w, mass

    flat_joints = skin_joints.reshape(-1).astype(np.int64, copy=False)
    flat_weights = skin_weights.reshape(-1).astype(np.float32, copy=False)
    flat_verts = np.repeat(np.arange(v_count, dtype=np.int32), k_count)
    valid = (
        (flat_joints >= 0)
        & (flat_joints < n_bones)
        & np.isfinite(flat_weights)
        & (flat_weights >= float(influence_threshold))
    )
    if not np.any(valid):
        return empty_i, empty_w, mass

    joints = flat_joints[valid]
    verts = flat_verts[valid]
    weights = flat_weights[valid]
    order = np.argsort(joints, kind="stable")
    joints = joints[order]
    verts = verts[order]
    weights = weights[order]

    split = np.flatnonzero(np.diff(joints)) + 1
    starts = np.r_[0, split]
    ends = np.r_[split, len(joints)]

    by_vertex = empty_i
    by_weight = empty_w
    for start, end in zip(starts, ends):
        bi = int(joints[start])
        vi = verts[start:end]
        wi = weights[start:end]
        if len(vi) > 1:
            unique_vi, inv = np.unique(vi, return_inverse=True)
            summed = np.zeros(len(unique_vi), dtype=np.float32)
            np.add.at(summed, inv, wi)
            vi = unique_vi.astype(np.int32, copy=False)
            wi = summed
        by_vertex[bi] = np.ascontiguousarray(vi.astype(np.int32, copy=False))
        by_weight[bi] = np.ascontiguousarray(wi.astype(np.float32, copy=False))
        mass[bi] = float(np.sum(wi, dtype=np.float64))
    return by_vertex, by_weight, mass


def _weighted_subsample_indices(
    weights: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weighted sample without replacement, falling back to uniform sampling."""
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    n = len(weights)
    max_points = int(max_points)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int32)
    positive = np.maximum(weights, 0.0)
    total = float(positive.sum())
    if total > 0.0:
        probs = positive / total
        return rng.choice(n, size=max_points, replace=False, p=probs).astype(np.int32)
    return rng.choice(n, size=max_points, replace=False).astype(np.int32)


def _quat_rotate_many(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Vectorized quaternion rotation for an (N,3) array."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    qv = q[:3]
    qw = float(q[3])
    t = 2.0 * np.cross(qv[np.newaxis, :], vectors)
    return vectors + qw * t + np.cross(qv[np.newaxis, :], t)


def _allocate_ellipsoid_budget(
    vertex_counts: np.ndarray,
    total: int,
) -> np.ndarray:
    """Distribute `total` ellipsoids proportionally to per-bone importance."""
    vertex_counts = np.asarray(vertex_counts, dtype=np.float64)
    active = vertex_counts > 0
    n_active = int(active.sum())
    budget = np.zeros(len(vertex_counts), dtype=np.int32)

    if n_active == 0 or total == 0:
        return budget

    if n_active >= total:
        # Give 1 to each of the top-`total` bones by vertex count
        top = np.argsort(-vertex_counts)[:total]
        budget[top] = 1
        return budget

    # Give every active bone at least 1, distribute the remainder proportionally
    budget[active] = 1
    remaining = total - n_active

    counts_f = vertex_counts.astype(float)
    counts_f[~active] = 0.0
    total_count = counts_f.sum()

    if remaining > 0 and total_count > 0:
        extra_f = counts_f / total_count * remaining
        extra_floor = np.floor(extra_f).astype(np.int32)
        budget += extra_floor
        leftover = remaining - int(extra_floor.sum())
        fracs = extra_f - extra_floor
        fracs[~active] = -1.0
        top_frac = np.argsort(-fracs)[:leftover]
        budget[top_frac] += 1

    return budget


def _kmeans_clusters(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_iter: int = 12,
) -> list[np.ndarray]:
    """K-means++ clustering; returns list of point arrays (one per cluster)."""
    return [
        pts for pts, _ in _weighted_kmeans_clusters(
            points, np.ones(len(points), dtype=np.float32), k, rng, n_iter,
        )
    ]


def _weighted_kmeans_clusters(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_iter: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Weighted k-means++ clustering for bone-local influence points."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    valid = np.isfinite(weights) & (weights > 0.0)
    points = points[valid]
    weights = weights[valid]
    k = min(k, len(points))
    if k <= 0:
        return []

    # K-means++ seeding with an incremental nearest-distance cache.  The old
    # version rebuilt an N x seeds x 3 temporary for every seed.
    weight_sum = float(weights.sum())
    probs0 = weights / weight_sum if weight_sum > 0 else None
    first = int(rng.choice(len(points), p=probs0))
    seed_idx = [first]
    chosen = np.zeros(len(points), dtype=bool)
    chosen[first] = True
    diff = points - points[first]
    closest_d2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
    for _ in range(k - 1):
        weighted_dists = closest_d2 * weights
        weighted_dists[chosen] = 0.0
        total = float(weighted_dists.sum())
        if total > 0.0:
            next_idx = int(rng.choice(len(points), p=weighted_dists / total))
        else:
            available = np.flatnonzero(~chosen)
            if len(available) == 0:
                break
            next_idx = int(rng.choice(available))
        seed_idx.append(next_idx)
        chosen[next_idx] = True
        diff = points - points[next_idx]
        new_d2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
        closest_d2 = np.minimum(closest_d2, new_d2)

    centroids = points[seed_idx].copy()
    k = len(centroids)

    labels = np.zeros(len(points), dtype=np.int32)
    for _ in range(n_iter):
        labels = _nearest_centroid_labels(points, centroids)
        cluster_weight = np.bincount(labels, weights=weights, minlength=k).astype(np.float32)
        new_centroids = centroids.copy()
        non_empty = cluster_weight > 0.0
        if np.any(non_empty):
            for ax in range(3):
                sums = np.bincount(
                    labels, weights=weights * points[:, ax], minlength=k,
                ).astype(np.float32)
                new_centroids[non_empty, ax] = sums[non_empty] / cluster_weight[non_empty]
        if np.allclose(centroids, new_centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids

    labels = _nearest_centroid_labels(points, centroids)
    return [
        (points[labels == ki], weights[labels == ki])
        for ki in range(k) if np.any(labels == ki)
    ]


def _nearest_centroid_labels(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    centroids = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    n = len(points)
    k = len(centroids)
    labels = np.zeros(n, dtype=np.int32)
    if n == 0 or k == 0:
        return labels
    point_norm = np.einsum("ij,ij->i", points, points, dtype=np.float32)
    centroid_norm = np.einsum("ij,ij->i", centroids, centroids, dtype=np.float32)
    max_cells = 1_000_000
    chunk = max(1, min(n, max_cells // max(k, 1)))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d2 = (
            point_norm[start:stop, None]
            + centroid_norm[None, :]
            - 2.0 * (points[start:stop] @ centroids.T)
        )
        labels[start:stop] = np.argmin(d2, axis=1)
    return labels


def _ellipsoid_from_points(
    points: np.ndarray,
    padding: float = 1.2,
    min_r: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an oriented ellipsoid to an unweighted point cloud."""
    return _ellipsoid_from_weighted_points(points, None, padding, min_r)


def _ellipsoid_from_weighted_points(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    padding: float = 1.2,
    min_r: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an oriented ellipsoid to a weighted bone-local point cloud."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if weights is None:
        weights = np.ones(len(points), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    valid = np.isfinite(weights) & (weights > 0.0)
    points = points[valid]
    weights = weights[valid]

    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if len(points) == 0:
        return (
            np.zeros(3, dtype=np.float32),
            np.full(3, min_r, dtype=np.float32),
            identity,
        )

    center = np.average(points, axis=0, weights=weights).astype(np.float64)
    if len(points) < 2:
        return center.astype(np.float32), np.full(3, min_r, dtype=np.float32), identity

    centered = points.astype(np.float64) - center
    wsum = max(float(weights.sum()), 1e-12)
    cov = (centered * weights[:, None]).T @ centered / wsum

    rotation = identity
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order]
        if np.linalg.det(axes) < 0.0:
            axes[:, -1] *= -1.0

        local = centered @ axes
        lo = np.array([
            _weighted_quantile(local[:, ax], weights, 0.02)
            for ax in range(3)
        ], dtype=np.float64)
        hi = np.array([
            _weighted_quantile(local[:, ax], weights, 0.98)
            for ax in range(3)
        ], dtype=np.float64)
        local_center = (lo + hi) * 0.5
        center = center + axes @ local_center
        half_range = np.maximum((hi - lo) * 0.5, 0.0)
        rotation = quat_from_matrix(axes).astype(np.float32)
    except np.linalg.LinAlgError:
        half_range = (points.max(axis=0) - points.min(axis=0)) / 2.0

    radii = np.maximum(half_range * padding, min_r).astype(np.float32)
    return center.astype(np.float32), radii, rotation


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return float(np.quantile(values, quantile))
    cdf = np.cumsum(weights) / total
    return float(np.interp(float(quantile), cdf, values))
