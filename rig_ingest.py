"""
rig_ingest.py — Turn Unity rig data into bone-assigned ellipsoids.

When a fit request carries skinning data, the fitted (snapshot-space, denormalized)
ellipsoids are assigned to bones via :class:`BoneEllipsoidMapper` and re-expressed
in each bone's local frame, so Unity can simply parent
``Sphere_[bodypart]_[Number]`` under the matching bone Transform and set
``localPosition`` / ``localRotation``.

Expected ``rig`` payload (all in the *same* coordinate space as the mesh vertices
that were posted — i.e. the original, un-normalized space the result is mapped
back into)::

    {
      "bones": [ { "name": "UpperArm.L",
                   "position": [x, y, z],          # bone snapshot-space position
                   "rotation": [x, y, z, w] },     # bone snapshot-space rotation
                 ... ],
      "boneIndices": [[i0, i1, i2, i3], ...],       # per vertex (V, K)
      "boneWeights": [[w0, w1, w2, w3], ...]        # per vertex (V, K)
    }

Each bone is treated as a root whose ``local_bind_transform`` *is* its supplied
snapshot-space transform, so ``Skeleton.compute_world_transforms`` returns those transforms
directly — the mapper only needs per-bone world transforms, not the hierarchy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from bone_ellipsoid_mapper import BoneEllipsoidMapper
from skeleton import Bone, Skeleton, mat4_compose


def sphere_name(bodypart: str | None, number: int) -> str:
    """Stable Unity object name: Sphere_[bodypart]_[Number]."""
    raw = str(bodypart or "Mesh").strip() or "Mesh"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)
    return f"Sphere_{safe}_{int(number)}"


def attachment_entry_fields(
    bone_local,
    ellipsoid_index: int,
    skeleton: Skeleton,
) -> dict[str, Any]:
    """Return the stable JSON representation of one attachment set."""
    primary = int(bone_local.bone_assignments[ellipsoid_index])
    joints = None
    weights = None
    if (bone_local.attachment_joints is not None
            and bone_local.attachment_weights is not None):
        joints = np.asarray(
            bone_local.attachment_joints[ellipsoid_index], dtype=np.int32)
        weights = np.asarray(
            bone_local.attachment_weights[ellipsoid_index], dtype=np.float64)
        valid = (
            (weights > 1.0e-8)
            & (joints >= 0)
            & (joints < skeleton.num_bones)
        )
        joints = joints[valid]
        weights = weights[valid]
    if joints is None or len(joints) == 0 or float(weights.sum()) <= 1.0e-12:
        joints = np.array([primary], dtype=np.int32)
        weights = np.array([1.0], dtype=np.float64)
    else:
        weights = weights / float(weights.sum())
    return {
        "attachment_bone_indices": [int(v) for v in joints],
        "attachment_bones": [skeleton.bones[int(v)].name for v in joints],
        "attachment_weights": [round(float(v), 7) for v in weights],
    }


def _attachments_from_entries(
    source_entries: list[dict[str, Any]] | None,
    assignments: np.ndarray,
    skeleton: Skeleton,
) -> tuple[np.ndarray, np.ndarray]:
    """Read attachment arrays from JSON, falling back to each primary bone."""
    count = len(assignments)
    joints = np.tile(np.asarray(assignments, dtype=np.int32)[:, None], (1, 4))
    weights = np.zeros((count, 4), dtype=np.float32)
    weights[:, 0] = 1.0
    name_to_index = {b.name: int(i) for i, b in enumerate(skeleton.bones)}
    for i in range(count):
        src = (
            source_entries[i]
            if source_entries is not None and i < len(source_entries)
            and isinstance(source_entries[i], dict)
            else {}
        )
        raw_joints = src.get("attachment_bone_indices")
        if raw_joints is None:
            raw_names = list(src.get("attachment_bones") or [])
            raw_joints = [name_to_index.get(str(name), -1) for name in raw_names]
        raw_weights = src.get("attachment_weights")
        if raw_joints is None or raw_weights is None:
            continue
        try:
            js = np.asarray(raw_joints, dtype=np.int32).reshape(-1)
            ws = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if len(js) != len(ws):
            continue
        valid = (ws > 1.0e-8) & (js >= 0) & (js < skeleton.num_bones)
        js = js[valid][:4]
        ws = ws[valid][:4]
        if len(js) == 0 or float(ws.sum()) <= 1.0e-12:
            continue
        ws = ws / float(ws.sum())
        joints[i, :len(js)] = js
        joints[i, len(js):] = int(assignments[i])
        weights[i] = 0.0
        weights[i, :len(ws)] = ws.astype(np.float32)
    return joints, weights


def build_skeleton_from_bones(bones: list[dict[str, Any]]) -> Skeleton:
    """Build a flat :class:`Skeleton` from Unity bone snapshot transforms."""
    if not bones:
        raise ValueError("rig.bones is empty")
    sk_bones: list[Bone] = []
    for i, b in enumerate(bones):
        raw_matrix = b.get("currentMatrix") or b.get("matrix")
        if raw_matrix is not None:
            world = np.asarray(raw_matrix, dtype=np.float64).reshape(4, 4)
        else:
            pos = np.asarray(b["position"], dtype=np.float64).reshape(3)
            rot = np.asarray(b["rotation"], dtype=np.float64).reshape(4)   # xyzw
            scale = np.asarray(
                b.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64,
            ).reshape(3)
            world = mat4_compose(pos, rot, scale)
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
    original space).  Returns a list of dicts with ``name``
    (``Sphere_[bodypart]_[Number]``), ``bone``, ``local_center``, ``radii`` and
    ``local_rotation``.
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
    counts: dict[str, int] = defaultdict(int)
    for i in range(bl.num_ellipsoids):
        bi = int(bl.bone_assignments[i])
        bone_name = skeleton.bones[bi].name
        local_index = counts[bone_name]
        counts[bone_name] += 1
        entries.append({
            "id": int(i),
            "name": sphere_name(bone_name, local_index),
            "bone_index": bi,
            "bone": bone_name,
            "center": [round(float(v), 7) for v in world_centers[i]],
            "local_center": [round(float(v), 7) for v in bl.local_centers[i]],
            "radii": [round(float(v), 7) for v in bl.local_radii[i]],
            "rotation": [round(float(v), 7) for v in world_rotations[i]],
            "local_rotation": [round(float(v), 7) for v in bl.local_rotations[i]],
            **attachment_entry_fields(bl, i, skeleton),
        })
    return entries


def world_to_bone_local_entries(
    world_centers: np.ndarray,
    world_radii: np.ndarray,
    world_rotations: np.ndarray,
    bone_assignments: np.ndarray,
    rig: dict[str, Any],
    source_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert ellipsoids with *known* bone assignments to bone-local entries.

    Like :func:`assign_ellipsoids_to_bones`, but skips the nearest-vertex vote:
    each ellipsoid's bone is already decided (e.g. by Bone Separation training,
    where each bone is fitted in isolation).  Geometry inputs must be in the
    rig's coordinate space (the posted mesh's original space).

    ``source_entries`` optionally preserves externally stable ids/names/bone
    labels, which is important for Unity pose morph fitting: the fitted
    ellipsoid must be returned as the same logical ellipsoid it came from.
    """
    bones = rig.get("bones") or []
    skeleton = build_skeleton_from_bones(bones)
    mapper = BoneEllipsoidMapper(skeleton)

    assignments = np.asarray(bone_assignments).astype(np.int32)
    attachment_joints, attachment_weights = _attachments_from_entries(
        source_entries, assignments, skeleton)
    bl = mapper.world_to_local(
        np.asarray(world_centers, dtype=np.float64),
        np.asarray(world_radii, dtype=np.float64),
        np.asarray(world_rotations, dtype=np.float64),
        assignments,
        pose=None,
        attachment_joints=attachment_joints,
        attachment_weights=attachment_weights,
    )

    entries: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for i in range(bl.num_ellipsoids):
        bi = int(bl.bone_assignments[i])
        src = (
            source_entries[i]
            if source_entries is not None and i < len(source_entries)
            and isinstance(source_entries[i], dict)
            else {}
        )
        source_bone = str(src.get("bone") or "")
        bone_name = source_bone or skeleton.bones[bi].name
        local_index = counts[bone_name]
        counts[bone_name] += 1
        try:
            entry_id = int(src.get("id", i))
        except Exception:
            entry_id = int(i)
        try:
            entry_bone_index = int(src.get("bone_index", bi))
        except Exception:
            entry_bone_index = bi
        entry_name = str(src.get("name") or "") or sphere_name(bone_name, local_index)
        entries.append({
            "id": entry_id,
            "name": entry_name,
            "bone_index": entry_bone_index,
            "bone": bone_name,
            "center": [round(float(v), 7) for v in world_centers[i]],
            "local_center": [round(float(v), 7) for v in bl.local_centers[i]],
            "radii": [round(float(v), 7) for v in bl.local_radii[i]],
            "rotation": [round(float(v), 7) for v in world_rotations[i]],
            "local_rotation": [round(float(v), 7) for v in bl.local_rotations[i]],
            **attachment_entry_fields(bl, i, skeleton),
        })
    return entries
