"""
bone_separation.py — Split a skinned mesh into per-bone region submeshes.

The "Bone Separation" fit mode trains each bone independently: every bone gets
its *own* SDF, computed from only the mesh region around that bone, and an
isolated ellipsoid fit.  This module does the geometry side — carving the mesh
into one compact submesh per bone and allocating the ellipsoid budget — while
``main_window`` drives the sequential SDF→fit loop.

Region rule
-----------
Membership is by skin *influence*, not just the single dominant bone: a vertex
joins bone *b*'s region when its weight toward *b* clears ``influence_threshold``
(its dominant bone always counts).  Because transition vertices are skinned by
two bones, they land in *both* neighbouring regions, so the two bones' fits
overlap across the seam instead of butting against a hard cut — that hard cut is
what produced the gaps/creases at the joints.  The region is then grown outward
by ``overlap_rings`` triangle rings, which both widens the overlap and pushes
each submesh's open boundary away from the bone so its region SDF stays
well-signed near the joint.  A triangle joins if **any** of its vertices is in
the region.  Each submesh is compacted to its own vertex set so its SDF bounding
box hugs the bone, and re-oriented outward (``fix_normals``).

Budgets are allocated by each bone's dominant *surface area* (not vertex count,
which is a tessellation artefact that over-funds finely-meshed fingers), so a
bone's ellipsoid count reflects the geometry it must cover rather than how
densely it happens to be triangulated or the size of the overlapping collar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from bone_ellipsoid_mapper import _allocate_ellipsoid_budget


@dataclass
class BonePart:
    """One bone's region as a standalone submesh plus its ellipsoid budget."""

    bone_index: int
    vertices: np.ndarray   # (Vi, 3) float32 — compacted submesh vertices
    faces: np.ndarray      # (Fi, 3) int32   — reindexed into ``vertices``
    budget: int            # initial ellipsoid count for this bone
    max_budget: int        # growth cap (>= budget)


def partition_mesh_by_bone(
    vertices: np.ndarray,
    faces: np.ndarray,
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    total_budget: int,
    total_max: int,
    min_faces: int = 4,
    influence_threshold: float = 0.1,
    overlap_rings: int = 1,
) -> list[BonePart]:
    """Carve a skinned mesh into per-bone submeshes with allocated budgets.

    Parameters
    ----------
    vertices : (V, 3) mesh vertices (any consistent coordinate space).
    faces : (F, 3) triangle indices.
    skin_joints : (V, K) bone index per vertex influence.
    skin_weights : (V, K) blend weight per vertex influence.
    total_budget : total initial ellipsoid count, split across bones by their
        vertex share (proportional allocation).
    total_max : total growth budget, split the same way (per-bone max is at
        least the per-bone initial budget).
    min_faces : drop bone regions with fewer than this many triangles.
    influence_threshold : a vertex joins a bone's region when its skin weight
        toward that bone is at least this value (its dominant bone always
        counts).  Larger → tighter, less overlapping regions; smaller →
        wider overlap and smoother seams.
    overlap_rings : after the weight-based selection, grow each region outward
        by this many triangle rings.  Adds extra overlap at the seams and keeps
        the submesh's open boundary away from the bone (cleaner region SDF).

    Returns
    -------
    list[BonePart] — one entry per bone that has a non-empty region and a
    non-zero budget, in ascending bone-index order.
    """
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    V = len(verts)
    joints = np.asarray(skin_joints).reshape(V, -1).astype(np.int64)
    weights = np.asarray(skin_weights, dtype=np.float64).reshape(V, -1)
    if joints.shape != weights.shape:
        raise ValueError(
            f"skin_joints {joints.shape} and skin_weights {weights.shape} "
            f"must have the same shape")

    # Dominant bone per vertex = the influence with the largest weight.
    dominant = joints[np.arange(V), np.argmax(weights, axis=1)]   # (V,)
    n_bones = int(dominant.max(initial=0)) + 1

    # Budgets are allocated by each bone's dominant *surface area*, NOT its
    # vertex count.  Vertex count is a tessellation artefact: a hand/finger is
    # usually far more finely meshed than a torso, so a vertex-count split hands
    # fingers an outsized ellipsoid budget and the fitter then carpets the thin
    # tube with many tiny spheres.  Surface area reflects the actual geometry a
    # bone needs to cover, so a slender finger gets just one or two ellipsoids.
    # Each triangle's area is shared equally among the dominant bones of its
    # three vertices (handles triangles that straddle a seam).
    tri_v = verts[tris]                                          # (F, 3, 3)
    tri_area = 0.5 * np.linalg.norm(
        np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0]),
        axis=1)                                                  # (F,)
    dom_tri = dominant[tris]                                     # (F, 3)
    area_per_bone = np.zeros(n_bones, dtype=np.float64)
    for j in range(3):
        np.add.at(area_per_bone, dom_tri[:, j], tri_area / 3.0)

    budget = _allocate_ellipsoid_budget(area_per_bone, int(total_budget))
    max_bud = _allocate_ellipsoid_budget(area_per_bone, int(total_max))

    rings = max(0, int(overlap_rings))

    parts: list[BonePart] = []
    for bi in range(n_bones):
        if budget[bi] <= 0:
            continue

        # Region membership by skin influence: a vertex is in bone bi's region
        # when its weight toward bi clears the threshold (or bi is its dominant
        # bone).  Transition vertices land in both neighbouring regions, so the
        # two bones' fits overlap across the seam instead of meeting at a hard
        # cut — smoothing the join.
        infl_bi = np.where(joints == bi, weights, 0.0).max(axis=1)   # (V,)
        sel_v = (infl_bi >= influence_threshold) | (dominant == bi)
        face_mask = sel_v[tris].any(axis=1)

        # Grow outward by a few triangle rings for extra overlap, and to push
        # the submesh's open boundary off the bone for a cleaner region SDF.
        for _ in range(rings):
            sel_v[np.unique(tris[face_mask])] = True
            face_mask = sel_v[tris].any(axis=1)

        if int(face_mask.sum()) < min_faces:
            continue

        global_faces = tris[face_mask]
        used = np.unique(global_faces)
        remap = np.full(V, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))

        sub_verts = verts[used]
        sub_faces = remap[global_faces].astype(np.int32)

        mesh = trimesh.Trimesh(vertices=sub_verts, faces=sub_faces, process=False)
        try:                                   # outward-orient for a sane SDF sign
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

        parts.append(BonePart(
            bone_index=bi,
            vertices=mesh.vertices.astype(np.float32, copy=False),
            faces=mesh.faces.astype(np.int32, copy=False),
            budget=int(budget[bi]),
            max_budget=int(max(max_bud[bi], budget[bi])),
        ))
    return parts
