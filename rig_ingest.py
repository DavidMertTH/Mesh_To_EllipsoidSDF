"""
rig_ingest.py — Turn Unity rig data into bone-assigned ellipsoids.

When a fit request carries skinning data, the fitted (world-space, denormalized)
ellipsoids are assigned to bones via :class:`BoneEllipsoidMapper` and re-expressed
in each bone's local frame, so Unity can simply parent ``Sphere_<i>`` under the
matching bone Transform and set ``localPosition`` / ``localRotation``.

Expected ``rig`` payload (all in the *same* coordinate space as the mesh vertices
that were posted — i.e. the original, un-normalized space the result is mapped
back into)::

    {
      "bones": [ { "name": "UpperArm.L",
                   "position": [x, y, z],          # bone bind world position
                   "rotation": [x, y, z, w] },     # bone bind world rotation (xyzw)
                 ... ],
      "boneIndices": [[i0, i1, i2, i3], ...],       # per vertex (V, K)
      "boneWeights": [[w0, w1, w2, w3], ...]        # per vertex (V, K)
    }

Each bone is treated as a root whose ``local_bind_transform`` *is* its bind world
transform, so ``Skeleton.compute_world_transforms`` returns those transforms
directly — the mapper only needs per-bone world transforms, not the hierarchy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bone_ellipsoid_mapper import BoneEllipsoidMapper
from skeleton import Bone, Skeleton, mat4_compose


def build_skeleton_from_bones(bones: list[dict[str, Any]]) -> Skeleton:
    """Build a flat :class:`Skeleton` from Unity bone bind transforms."""
    if not bones:
        raise ValueError("rig.bones is empty")
    sk_bones: list[Bone] = []
    for i, b in enumerate(bones):
        pos = np.asarray(b["position"], dtype=np.float64).reshape(3)
        rot = np.asarray(b["rotation"], dtype=np.float64).reshape(4)   # xyzw
        world = mat4_compose(pos, rot, np.ones(3, dtype=np.float64))
        sk_bones.append(Bone(
            name=str(b["name"]),
            index=i,
            parent_index=-1,
            local_bind_transform=world,
            inverse_bind_matrix=np.linalg.inv(world),
        ))
    return Skeleton(sk_bones)


def assign_ellipsoids_to_bones(
    world_centers: np.ndarray,
    world_radii: np.ndarray,
    world_rotations: np.ndarray,
    mesh_vertices: np.ndarray,
    rig: dict[str, Any],
) -> list[dict[str, Any]]:
    """Assign ellipsoids to bones and return Unity-ready, bone-local entries.

    All geometry inputs must share one coordinate space (the posted mesh's
    original space).  Returns a list of dicts with ``name`` (``Sphere_<i>``),
    ``bone``, ``local_center``, ``radii`` and ``local_rotation``.
    """
    bones = rig.get("bones") or []
    skeleton = build_skeleton_from_bones(bones)

    verts = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
    joints = np.asarray(rig["boneIndices"]).reshape(len(verts), -1).astype(np.int64)
    weights = np.asarray(rig["boneWeights"], dtype=np.float64).reshape(len(verts), -1)
    if joints.shape != weights.shape:
        raise ValueError(
            f"boneIndices {joints.shape} and boneWeights {weights.shape} "
            f"must have the same shape")
    if int(joints.max(initial=0)) >= skeleton.num_bones:
        raise ValueError("boneIndices reference a bone beyond rig.bones")

    mapper = BoneEllipsoidMapper(skeleton)
    bl = mapper.assign_to_bones(
        np.asarray(world_centers, dtype=np.float64),
        np.asarray(world_radii, dtype=np.float64),
        np.asarray(world_rotations, dtype=np.float64),
        verts, joints, weights, pose=None,
    )

    entries: list[dict[str, Any]] = []
    for i in range(bl.num_ellipsoids):
        bi = int(bl.bone_assignments[i])
        entries.append({
            "name": f"Sphere_{i}",
            "bone": skeleton.bones[bi].name,
            "local_center": [round(float(v), 7) for v in bl.local_centers[i]],
            "radii": [round(float(v), 7) for v in bl.local_radii[i]],
            "local_rotation": [round(float(v), 7) for v in bl.local_rotations[i]],
        })
    return entries


def world_to_bone_local_entries(
    world_centers: np.ndarray,
    world_radii: np.ndarray,
    world_rotations: np.ndarray,
    bone_assignments: np.ndarray,
    rig: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert ellipsoids with *known* bone assignments to bone-local entries.

    Like :func:`assign_ellipsoids_to_bones`, but skips the nearest-vertex vote:
    each ellipsoid's bone is already decided (e.g. by Bone Separation training,
    where each bone is fitted in isolation).  Geometry inputs must be in the
    rig's coordinate space (the posted mesh's original space).
    """
    bones = rig.get("bones") or []
    skeleton = build_skeleton_from_bones(bones)
    mapper = BoneEllipsoidMapper(skeleton)

    bl = mapper.world_to_local(
        np.asarray(world_centers, dtype=np.float64),
        np.asarray(world_radii, dtype=np.float64),
        np.asarray(world_rotations, dtype=np.float64),
        np.asarray(bone_assignments).astype(np.int32),
        pose=None,
    )

    entries: list[dict[str, Any]] = []
    for i in range(bl.num_ellipsoids):
        bi = int(bl.bone_assignments[i])
        entries.append({
            "name": f"Sphere_{i}",
            "bone": skeleton.bones[bi].name,
            "local_center": [round(float(v), 7) for v in bl.local_centers[i]],
            "radii": [round(float(v), 7) for v in bl.local_radii[i]],
            "local_rotation": [round(float(v), 7) for v in bl.local_rotations[i]],
        })
    return entries
