"""
bone_symmetry.py — geometric mirror-plane detection and bone-region pairing.

Pure, dependency-light helpers used by the Bone-Separation pipeline to exploit
left/right symmetry:

  * ``detect_mirror_plane`` — find a world-axis mirror plane of a point cloud via
    a Chamfer-style nearest-neighbour test.
  * ``pair_bone_parts`` — classify per-bone region submeshes against that plane
    into mirror pairs (``{derived_index: source_index}``) and on-plane *centre*
    regions (which are themselves symmetric).

Keeping these here (instead of buried in ``main_window``) lets both the GUI and
the batched fitter share one implementation and unit-test it without Qt/Warp.
"""

from __future__ import annotations

import numpy as np


def detect_mirror_plane(verts: np.ndarray, max_pts: int = 2000,
                        tol_frac: float = 0.02, min_inlier: float = 0.85):
    """Geometrically detect a mirror plane normal to a world axis.

    For each axis the point cloud is reflected about the bounding-box midpoint and
    every reflected point is matched to its nearest original point (a Chamfer-style
    symmetry test).  The fraction of points whose mirror partner lies within
    ``tol_frac`` of the bbox diagonal is the symmetry score; the best-scoring axis
    is returned as ``(axis, coord)`` when it clears ``min_inlier``.  Returns
    ``None`` if the mesh is not symmetric enough about any axis.  ``verts`` is
    subsampled to ``max_pts`` points to bound the O(N²) nearest-neighbour cost.
    """
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if len(v) < 8:
        return None
    if len(v) > max_pts:
        idx = np.random.default_rng(0).choice(len(v), max_pts, replace=False)
        v = v[idx]
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    center = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo))
    if diag < 1e-9:
        return None
    tol = tol_frac * diag
    best_axis, best_inlier = None, 0.0
    for a in range(3):
        refl = v.copy()
        refl[:, a] = 2.0 * center[a] - refl[:, a]
        # nearest-neighbour distance from each reflected point to the cloud
        d2 = ((refl[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
        nn = np.sqrt(d2.min(axis=1))
        inlier = float(np.mean(nn < tol))
        if inlier > best_inlier:
            best_axis, best_inlier = a, inlier
    if best_axis is None or best_inlier < min_inlier:
        return None
    return int(best_axis), float(center[best_axis])


def pair_bone_parts(parts, axis: int, coord: float):
    """Pair mirror-symmetric bone regions across the plane ``(axis, coord)``.

    Each part is represented by the centroid of its submesh vertices.  A part is
    treated as *on-plane* (centre) when its centroid sits within a small tolerance
    of the plane: such a region is itself mirror-symmetric, so it is fitted *with*
    the symmetry constraint (never mirrored to a partner).  Off-plane parts are
    matched by mutual nearest neighbour of their reflected centroids; for each
    matched pair the earlier-indexed part is the *source* (fitted) and the later
    one is *derived* by reflecting the source's fitted ellipsoids.

    Returns ``(mirror_map, center)`` where ``mirror_map`` is
    ``{derived_index: source_index}`` and ``center`` is the set of on-plane part
    indices (both index into ``parts``).
    """
    if len(parts) < 1:
        return {}, set()
    cents = np.array([p.vertices.mean(axis=0) for p in parts], dtype=np.float64)
    P = len(cents)
    span = float(cents[:, axis].max() - cents[:, axis].min())
    eps = 0.05 * span + 1e-9
    on_plane = np.abs(cents[:, axis] - coord) <= eps
    center = {int(i) for i in range(P) if on_plane[i]}
    if P < 2:
        return {}, center

    refl = cents.copy()
    refl[:, axis] = 2.0 * coord - refl[:, axis]
    # Pairwise distance from each reflected centroid to every original centroid.
    diff = refl[:, None, :] - cents[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)

    mirror_map: dict[int, int] = {}
    paired: set[int] = set()
    for i in range(P):
        j = int(nn[i])
        if on_plane[i] or on_plane[j] or i in paired or j in paired:
            continue
        if nn[j] != i or i >= j:        # mutual NN, keep canonical order i<j
            continue
        sep = float(np.linalg.norm(cents[i] - cents[j]))
        if D[i, j] < 0.5 * sep:         # reflected-i lands close to j vs. their gap
            mirror_map[j] = i
            paired.update((i, j))
    return mirror_map, center
