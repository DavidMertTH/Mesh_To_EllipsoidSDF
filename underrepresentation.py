"""
underrepresentation.py — Detect under-represented mesh regions via a
*relative* coverage error.

Motivation
----------
An absolute error |pred − target| systematically under-weights thin
features: a fully-missed finger of a human mesh has a tiny |target|
(shallow interior), so its absolute reconstruction error is small even
when no ellipsoid covers it at all.  Large limbs dominate the absolute
error and thin parts are ignored.

Relative metric
----------------
We measure how far the ellipsoid union surface sits from the mesh,
relative to the local feature thickness, and then **emphasise the mesh
surface (zero level set) strongly** — getting the surface right is the
single most important goal, so a mismatch there counts much more than in
the interior bulk.

    gap(p)   = max(pred(p) − target(p), 0)   # > 0 when ellipsoids leave p uncovered
    depth(p) = max(−target(p), floor)        # local thickness, floored near surface
    rel(p)   = gap(p) / depth(p)

    surface emphasis (peaks on the surface, target → 0):
        sw(p) = 1 + surface_weight · exp(−(target(p) / σ)²),  σ = surface_sigma_vox · dx

    score(p) = rel(p) · sw(p)

  - rel ≈ 0   → ellipsoid surface coincides with the target (well covered)
  - rel ≈ 1   → ellipsoid surface sits at the mesh surface (point fully missed)
  - rel > 1   → ellipsoids are even farther out than the surface

Dividing by local depth keeps a thin finger and a thick limb on the same
scale (both flagged when their interior is unfilled, independent of size).
Unlike the previous version we do **not** drop the surface shell — instead
voxels near the surface are up-weighted by ``sw``, and a thin band just
outside the surface is also considered so the search reacts to the
ellipsoid surface failing to reach the mesh surface.  A perfectly fitted
surface has gap ≈ 0, so it is never flagged regardless of ``sw``.
"""

from __future__ import annotations

import numpy as np


def _underrep_score(
    flat_target: np.ndarray,
    flat_pred: np.ndarray,
    dx: float,
    surface_weight: float,
    surface_sigma_vox: float,
    depth_floor_vox: float,
    outside_margin_vox: float,
    flat_thickness: np.ndarray | None = None,
    min_thickness_vox: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (consider_idx, rel, gap, sw) for all considered voxels.

    Considered = interior voxels plus an optional thin shell just outside the
    surface (target < outside_margin·dx).

      - rel : detection signal = floored relative miss (gap / denom)
      - gap : absolute miss in world units (used to reject sub-voxel noise)
      - sw  : surface-emphasis multiplier (peaks on the zero level set)

    The relative-miss denominator is the **local feature thickness** when a
    precomputed ``flat_thickness`` field is supplied: a given gap on a thin
    feature (small thickness) then yields a much larger relative error than the
    same gap on a thick limb.  Without a thickness field it falls back to the
    per-voxel interior depth ``-target``.

    ``min_thickness_vox`` floors that local thickness (in voxels) before it is
    used as the relative-miss scale.  Features thinner than the grid can resolve
    (e.g. a finger only ~2 voxels across, or sub-voxel on a coarse grid) would
    otherwise drive the denominator toward zero, inflating their relative error
    so far that the region is *permanently* flagged — a union of coarse
    ellipsoids can never close a sub-voxel gap, so spawn fires endlessly and
    carpets the feature with tiny spheres.  Flooring the thickness treats such
    features at the resolution scale, so coverage to within a voxel or two is
    accepted and the spawn loop stops.  ``0.0`` keeps the old behaviour.
    """
    margin = float(outside_margin_vox) * float(dx)
    consider_idx = np.where(flat_target < margin)[0]
    empty = np.empty((0,), dtype=np.float32)
    if consider_idx.size == 0:
        return consider_idx, empty, empty, empty

    t = flat_target[consider_idx]
    gap = np.maximum(flat_pred[consider_idx] - t, 0.0)          # absolute miss (world units)
    if flat_thickness is not None:
        # Feature thickness is a diameter; half of it is the radial reach used
        # as the relative-miss scale (matches the old depth = -target scale).
        # Floor the thickness at ``min_thickness_vox`` voxels first so features
        # below the grid resolution are scored at the resolution scale instead
        # of driving the denominator (and thus the relative error) toward
        # infinity — that runaway is what made thin fingers spawn-thrash into a
        # chain of tiny spheres.
        thick = np.maximum(flat_thickness[consider_idx],
                           float(min_thickness_vox) * float(dx))
        denom = np.maximum(0.5 * thick, 0.5 * float(dx))
    else:
        denom = np.maximum(-t, float(depth_floor_vox) * float(dx))
    rel = (gap / denom).astype(np.float32)                      # detection signal

    sigma = max(float(surface_sigma_vox) * float(dx), 1e-9)
    sw = (1.0 + float(surface_weight) * np.exp(-(t * t) / (sigma * sigma))).astype(np.float32)

    return consider_idx, rel, gap.astype(np.float32), sw


def compute_relative_underrep(
    target_grid: np.ndarray,
    pred_grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    n: int,
    surface_weight: float = 4.0,
    surface_sigma_vox: float = 1.5,
    depth_floor_vox: float = 2.0,
    outside_margin_vox: float = 0.0,
    rel_threshold: float = 0.6,
    min_gap_vox: float = 1.0,
    thickness_grid: np.ndarray | None = None,
    min_thickness_vox: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space points of under-represented voxels (surface-weighted).

    Detection and severity are deliberately decoupled:

      - **Detection** uses the floored relative miss ``rel`` against a single
        threshold, and additionally requires the *absolute* miss to exceed
        ``min_gap_vox`` voxels.  This rejects the sub-voxel quantisation noise
        that otherwise floods the whole surface shell, while still catching
        genuinely missed thin features (their interior ``rel`` is large).
      - **Severity** (the returned ``values``, used for colour and later
        region ranking) is ``rel · sw`` so that misses *on the surface* are
        emphasised far more strongly than equally-deep interior misses.

    Parameters
    ----------
    target_grid, pred_grid : (nz, ny, nx) float32
        Mesh SDF and ellipsoid union SDF (negative inside).  May be anisotropic;
        the voxel shape is taken from ``target_grid.shape``.
    origin : (3,) float32
        World-space corner of the grid.
    dx : float
        Voxel size.
    n : int
        Deprecated / unused — kept for signature compatibility.  The grid shape
        is derived from ``target_grid`` so anisotropic grids work unchanged.
    surface_weight : float
        Strength of the surface emphasis on severity (default 4.0 → up to 5×).
    surface_sigma_vox : float
        Width of the surface bump in voxels (default 1.5).
    depth_floor_vox : float
        Lower bound on the depth denominator in voxels (default 2.0).  Larger
        values make the surface less twitchy to sub-voxel gaps.
    outside_margin_vox : float
        How far outside the surface (in voxels) is still considered (default 0.0
        → interior only).
    rel_threshold : float
        Minimum relative miss for detection (default 0.6).
    min_gap_vox : float
        Minimum absolute miss in voxels for detection (default 1.0) — kills
        sub-voxel surface noise.

    Returns
    -------
    points : (M, 3) float32
        World positions of under-represented voxels.
    values : (M,) float32
        Surface-weighted severity (rel · sw) at those voxels.
    """
    flat_target = target_grid.ravel()
    flat_pred = pred_grid.ravel()
    flat_thickness = thickness_grid.ravel() if thickness_grid is not None else None

    consider_idx, rel, gap, sw = _underrep_score(
        flat_target, flat_pred, dx,
        surface_weight, surface_sigma_vox, depth_floor_vox, outside_margin_vox,
        flat_thickness, min_thickness_vox,
    )
    if consider_idx.size == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    keep = (rel >= rel_threshold) & (gap >= float(min_gap_vox) * float(dx))
    if not np.any(keep):
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    sel_flat = consider_idx[keep]
    values = (rel[keep] * sw[keep]).astype(np.float32)

    # Derive the (possibly anisotropic) shape straight from the grid — ``n`` is
    # kept only for signature compatibility and may not describe every axis.
    iz, iy, ix = np.unravel_index(sel_flat, target_grid.shape)
    points = (origin.astype(np.float32)
              + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * float(dx))
    return points.astype(np.float32), values


def relative_underrep_grid(
    target_grid: np.ndarray,
    pred_grid: np.ndarray,
    dx: float,
    surface_weight: float = 4.0,
    surface_sigma_vox: float = 1.5,
    depth_floor_vox: float = 2.0,
    outside_margin_vox: float = 0.0,
    min_gap_vox: float = 1.0,
    thickness_grid: np.ndarray | None = None,
    min_thickness_vox: float = 0.0,
) -> np.ndarray:
    """Full (n, n, n) surface-weighted under-representation severity grid.

    Severity = rel · sw, zeroed where the absolute miss is below
    ``min_gap_vox`` voxels or outside ``outside_margin_vox``.
    Useful for the SuperFit residual-region clustering later on.
    """
    shape = target_grid.shape
    flat_target = target_grid.ravel()
    flat_pred = pred_grid.ravel()
    flat_thickness = thickness_grid.ravel() if thickness_grid is not None else None
    out = np.zeros(flat_target.shape, dtype=np.float32)

    consider_idx, rel, gap, sw = _underrep_score(
        flat_target, flat_pred, dx,
        surface_weight, surface_sigma_vox, depth_floor_vox, outside_margin_vox,
        flat_thickness, min_thickness_vox,
    )
    valid = gap >= float(min_gap_vox) * float(dx)
    out[consider_idx[valid]] = (rel[valid] * sw[valid])
    return out.reshape(shape)
