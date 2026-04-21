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
    num_ellipsoids: int = 0

    def __post_init__(self):
        self.num_ellipsoids = len(self.local_centers)


# ── Mapper class ─────────────────────────────────────────────────────────────

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

        # Convert to bone-local space
        bone_local = self.world_to_local(
            world_centers, world_radii, world_rotations,
            bone_assignments, pose,
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
    ) -> BoneLocalEllipsoids:
        """Convert world-space ellipsoid params to bone-local space.

        For each ellipsoid assigned to bone B:
          local_center = inv(bone_rot) ⊗ (world_center - bone_pos)
          local_rot    = inv(bone_rot) ⊗ world_rot
          local_radii  = world_radii   (no scaling)
        """
        N = len(world_centers)
        bone_pos, bone_rot = self.skeleton.compute_bone_positions_rotations(pose)

        # The bone positions/rotations are from the world transforms.
        # But we need the transforms AFTER applying inverse_bind, so that
        # "local" means relative to the bone's coordinate system.
        # Actually, we want the bone's world transform directly:
        # local_center is the offset from the bone's world position,
        # expressed in the bone's world rotation frame.

        world_transforms = self.skeleton.compute_world_transforms(pose)

        local_centers = np.zeros((N, 3), dtype=np.float32)
        local_radii = world_radii.copy().astype(np.float32)
        local_rotations = np.zeros((N, 4), dtype=np.float32)

        for i in range(N):
            bi = bone_assignments[i]
            t_bone, q_bone, _ = mat4_decompose(world_transforms[bi])

            # Center: inv_rot(world_center - bone_pos)
            delta = world_centers[i].astype(np.float64) - t_bone
            q_inv = quat_inverse(q_bone)
            local_c = quat_rotate(q_inv, delta)
            local_centers[i] = local_c.astype(np.float32)

            # Rotation: inv(bone_rot) * world_rot
            q_world = world_rotations[i].astype(np.float64)
            q_local = quat_multiply(q_inv, q_world)
            q_local /= np.linalg.norm(q_local)
            local_rotations[i] = q_local.astype(np.float32)

        return BoneLocalEllipsoids(
            local_centers=local_centers,
            local_radii=local_radii,
            local_rotations=local_rotations,
            bone_assignments=bone_assignments.astype(np.int32),
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

        world_transforms = self.skeleton.compute_world_transforms(pose)

        N = bone_local.num_ellipsoids
        w_centers = np.zeros((N, 3), dtype=np.float32)
        w_radii = bone_local.local_radii.copy()
        w_rotations = np.zeros((N, 4), dtype=np.float32)

        for i in range(N):
            bi = bone_local.bone_assignments[i]
            t_bone, q_bone, _ = mat4_decompose(world_transforms[bi])

            # Center
            lc = bone_local.local_centers[i].astype(np.float64)
            wc = quat_rotate(q_bone, lc) + t_bone
            w_centers[i] = wc.astype(np.float32)

            # Rotation
            lq = bone_local.local_rotations[i].astype(np.float64)
            wq = quat_multiply(q_bone, lq)
            wq /= np.linalg.norm(wq)
            w_rotations[i] = wq.astype(np.float32)

        return w_centers, w_radii, w_rotations

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

    # ── Reassignment check ───────────────────────────────────────────

    def check_and_reassign(
        self,
        bone_local: BoneLocalEllipsoids,
        pose: Pose,
        deformed_verts: np.ndarray,
        skin_joints: np.ndarray,
        skin_weights: np.ndarray,
    ) -> tuple[BoneLocalEllipsoids, int]:
        """Re-evaluate bone assignments for drifted ellipsoids.

        Gradient updates during training can shift ellipsoid centres into a
        different bone's region.  This method detects such drift and
        re-expresses the affected ellipsoids in the new bone's local frame
        so that subsequent pose transforms remain correct.

        Returns (updated_bone_local, n_reassigned).
        """
        wc, wr, wq = self.local_to_world_np(bone_local, pose)
        N = len(wc)
        old_assignments = bone_local.bone_assignments.copy()
        new_assignments = np.empty(N, dtype=np.int32)

        for i in range(N):
            c = wc[i].astype(np.float64)
            dists = np.linalg.norm(
                deformed_verts.astype(np.float64) - c[np.newaxis, :], axis=1,
            )
            K = min(20, len(dists))
            nearest_idx = np.argpartition(dists, K)[:K] if K < len(dists) else np.arange(len(dists))
            inv_dists = 1.0 / np.maximum(dists[nearest_idx], 1e-8)
            bone_scores = np.zeros(self.skeleton.num_bones, dtype=np.float64)
            for ni in range(K):
                vi = nearest_idx[ni]
                w_dist = inv_dists[ni]
                for k in range(skin_joints.shape[1]):
                    bone_scores[skin_joints[vi, k]] += w_dist * skin_weights[vi, k]
            new_assignments[i] = int(np.argmax(bone_scores))

        changed_mask = new_assignments != old_assignments
        n_reassigned = int(changed_mask.sum())
        if n_reassigned == 0:
            return bone_local, 0

        world_transforms = self.skeleton.compute_world_transforms(pose)
        new_local_centers    = bone_local.local_centers.copy()
        new_local_rotations  = bone_local.local_rotations.copy()

        for i in np.where(changed_mask)[0]:
            bi_new = new_assignments[i]
            t_bone, q_bone, _ = mat4_decompose(world_transforms[bi_new])
            q_inv = quat_inverse(q_bone)
            new_local_centers[i] = quat_rotate(q_inv, wc[i].astype(np.float64) - t_bone).astype(np.float32)
            q_local = quat_multiply(q_inv, wq[i].astype(np.float64))
            new_local_rotations[i] = (q_local / np.linalg.norm(q_local)).astype(np.float32)

        updated = BoneLocalEllipsoids(
            local_centers=new_local_centers,
            local_radii=bone_local.local_radii.copy(),
            local_rotations=new_local_rotations,
            bone_assignments=new_assignments,
        )
        self._bone_local = updated
        return updated, n_reassigned

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
    random_seed: int = 42,
) -> BoneLocalEllipsoids:
    """Create BoneLocalEllipsoids directly from the skeleton structure.

    Skips the T-pose fit + assign-to-bones workflow entirely.
    For each bone, ellipsoids are placed based on the local vertex distribution.
    Budget is allocated proportionally to vertex count per bone.

    Parameters
    ----------
    skeleton       : Skeleton — bone hierarchy
    vertices       : (V, 3) rest-pose mesh vertices
    skin_joints    : (V, K) bone indices per vertex
    skin_weights   : (V, K) blend weights per vertex
    n_ellipsoids   : total number of ellipsoids to create
    padding        : radii scale factor applied to the half-range of the point cloud
    min_radius     : minimum semi-axis length to avoid degenerate ellipsoids
    random_seed    : for k-means reproducibility
    """
    rng = np.random.default_rng(random_seed)
    n_bones = skeleton.num_bones
    V = len(vertices)

    # ── 1. Dominant bone per vertex (argmax of skin weights) ──
    dominant_bone = skin_joints[np.arange(V), np.argmax(skin_weights, axis=1)]

    # ── 2. Budget: how many ellipsoids per bone ──
    counts = np.bincount(dominant_bone, minlength=n_bones)
    budget = _allocate_ellipsoid_budget(counts, n_ellipsoids)

    # ── 3. Bone world transforms in T-pose ──
    world_transforms = skeleton.compute_world_transforms(None)

    all_centers:     list[np.ndarray] = []
    all_radii:       list[np.ndarray] = []
    all_rotations:   list[np.ndarray] = []
    all_assignments: list[int]        = []

    for bi in range(n_bones):
        k = int(budget[bi])
        if k == 0:
            continue

        t_bone, q_bone, _ = mat4_decompose(world_transforms[bi])
        q_inv = quat_inverse(q_bone)

        # Vertices dominated by this bone → bone-local space
        bone_verts_world = vertices[dominant_bone == bi]
        if len(bone_verts_world) == 0:
            continue

        local_verts = np.array([
            quat_rotate(q_inv, v.astype(np.float64) - t_bone)
            for v in bone_verts_world
        ], dtype=np.float32)

        # Split into k clusters and fit one ellipsoid per cluster
        clusters = [local_verts] if k == 1 else _kmeans_clusters(local_verts, k, rng)

        for pts in clusters:
            center, radii = _ellipsoid_from_points(pts, padding, min_radius)
            all_centers.append(center)
            all_radii.append(radii)
            all_rotations.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            all_assignments.append(bi)

    return BoneLocalEllipsoids(
        local_centers=np.array(all_centers,     dtype=np.float32),
        local_radii=np.array(all_radii,         dtype=np.float32),
        local_rotations=np.array(all_rotations, dtype=np.float32),
        bone_assignments=np.array(all_assignments, dtype=np.int32),
    )


def _allocate_ellipsoid_budget(
    vertex_counts: np.ndarray,
    total: int,
) -> np.ndarray:
    """Distribute `total` ellipsoids proportionally to per-bone vertex count."""
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
    n_iter: int = 30,
) -> list[np.ndarray]:
    """K-means++ clustering; returns list of point arrays (one per cluster)."""
    k = min(k, len(points))

    # K-means++ seeding: spread initial centroids
    seed_idx = [int(rng.integers(len(points)))]
    for _ in range(k - 1):
        dists = np.min(
            np.linalg.norm(points[:, None] - points[seed_idx][None], axis=2),
            axis=1,
        ) ** 2
        total = dists.sum()
        probs = dists / total if total > 0 else np.ones(len(points)) / len(points)
        seed_idx.append(int(rng.choice(len(points), p=probs)))

    centroids = points[seed_idx].copy()

    labels = np.zeros(len(points), dtype=np.int32)
    for _ in range(n_iter):
        dists = np.linalg.norm(
            points[:, None, :] - centroids[None, :, :], axis=2,
        )  # (N, k)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            points[labels == ki].mean(axis=0)
            if np.any(labels == ki) else centroids[ki]
            for ki in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return [points[labels == ki] for ki in range(k) if np.any(labels == ki)]


def _ellipsoid_from_points(
    points: np.ndarray,
    padding: float = 1.2,
    min_r: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an axis-aligned ellipsoid (center + radii) to a point cloud."""
    center = points.mean(axis=0).astype(np.float32)
    if len(points) < 2:
        return center, np.full(3, min_r, dtype=np.float32)
    half_range = (points.max(axis=0) - points.min(axis=0)) / 2.0
    radii = np.maximum(half_range * padding, min_r).astype(np.float32)
    return center, radii
