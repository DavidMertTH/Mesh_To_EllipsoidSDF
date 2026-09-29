"""
sdf_compute.py — Warp-based SDF computation on triangle meshes.

Provides:
  - GPU/CPU kernels for single-point and voxel-grid SDF queries.
  - SdfComputer class that manages mesh upload and grid computation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

from sdf_blowup import (
    BLOWUP_CARRIER_MARGIN_VOXELS,
    MAX_UI_THICKNESS_FRACTION,
    build_surface_carried_thickness,
    conservative_mirror_min,
    relative_blowup_extent_voxels,
)
from thickness import dilate_zeros, local_thickness
from sdf_samples import SdfSampleSet
from thin_sampling import (
    inverse_thickness_factors,
    positive_weighted_median,
)


# ── Warp kernels ──────────────────────────────────────────────────────────────

@wp.kernel
def _sdf_one_point_kernel(
    mesh_id: wp.uint64,
    p: wp.vec3,
    use_winding: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    if use_winding == 1:
        q = wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6, 2.0, 0.5)
    else:
        q = wp.mesh_query_point(mesh_id, p, 1.0e6)
    if q.result == 0:
        out_sdf[0] = 1.0e6
        return
    closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    d = wp.length(p - closest)
    s = -1.0 if q.sign < 0.0 else 1.0
    out_sdf[0] = d * s


@wp.kernel
def _sdf_voxel_grid_kernel(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    max_dist: float,
    use_winding: int,
    out_offset: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)

    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )

    # ``max_dist`` bounds the BVH search: voxels whose nearest surface point is
    # farther than this terminate early (huge speedup for the empty exterior).
    # It MUST exceed the deepest interior depth, else thick interiors miss and
    # wrongly read as far-outside — the caller sizes it accordingly.
    # ``use_winding`` picks the inside/outside test: the generalised winding
    # number (robust for non-watertight meshes) vs the faster normal-based sign.
    if use_winding == 1:
        q = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_dist, 2.0, 0.5)
    else:
        q = wp.mesh_query_point(mesh_id, p, max_dist)
    if q.result == 0:
        out_sdf[out_offset + tid] = 1.0e6
        return

    closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    d = wp.length(p - closest)
    s = -1.0 if q.sign < 0.0 else 1.0
    out_sdf[out_offset + tid] = d * s


@wp.kernel
def _sdf_surface_thickness_voxel_grid_kernel(
    mesh_id: wp.uint64,
    mesh_indices: wp.array(dtype=wp.int32),
    surface_thickness: wp.array(dtype=wp.float32),
    face_corner_mode: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    max_dist: float,
    use_winding: int,
    out_offset: int,
    out_sdf: wp.array(dtype=wp.float32),
    out_thickness: wp.array(dtype=wp.float32),
):
    """Query signed distance and its nearest-surface feature in one BVH pass.

    Warp's mesh-query coordinates use ``u * v0 + v * v1 +
    (1-u-v) * v2``.  The same weights therefore transport either a scalar on
    the three shared mesh vertices or three independent values owned by the
    queried face.  The latter preserves discontinuities across hard edges.
    """
    tid = wp.tid()
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)

    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )
    if use_winding == 1:
        q = wp.mesh_query_point_sign_winding_number(
            mesh_id, p, max_dist, 2.0, 0.5)
    else:
        q = wp.mesh_query_point(mesh_id, p, max_dist)

    out_index = out_offset + tid
    if q.result == 0:
        out_sdf[out_index] = 1.0e6
        out_thickness[out_index] = 0.0
        return

    closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    d = wp.length(p - closest)
    s = -1.0 if q.sign < 0.0 else 1.0
    out_sdf[out_index] = d * s

    face_offset = q.face * 3
    i0 = mesh_indices[face_offset]
    i1 = mesh_indices[face_offset + 1]
    i2 = mesh_indices[face_offset + 2]
    if face_corner_mode == 1:
        i0 = face_offset
        i1 = face_offset + 1
        i2 = face_offset + 2
    w2 = 1.0 - q.u - q.v
    out_thickness[out_index] = (
        q.u * surface_thickness[i0]
        + q.v * surface_thickness[i1]
        + w2 * surface_thickness[i2]
    )


@wp.kernel
def _sdf_voxel_grid_batch_kernel(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    dxs: wp.array(dtype=wp.float32),
    n_per: int,
    max_dist: float,
    use_winding: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    # One launch over many equal-resolution (n_per³) boxes; box index = tid //
    # voxels-per-box.  Used to compute every local region box in a single launch.
    tid = wp.tid()
    per = n_per * n_per * n_per
    b = tid // per
    local = tid % per
    ix = local % n_per
    iy = (local // n_per) % n_per
    iz = local // (n_per * n_per)

    origin = origins[b]
    dx = dxs[b]
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )

    if use_winding == 1:
        q = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_dist, 2.0, 0.5)
    else:
        q = wp.mesh_query_point(mesh_id, p, max_dist)
    if q.result == 0:
        out_sdf[tid] = 1.0e6
        return

    closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    d = wp.length(p - closest)
    s = -1.0 if q.sign < 0.0 else 1.0
    out_sdf[tid] = d * s


@wp.kernel
def _sdf_points_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    use_winding: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = points[tid]
    if use_winding == 1:
        q = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_dist, 2.0, 0.5)
    else:
        q = wp.mesh_query_point(mesh_id, p, max_dist)
    if q.result == 0:
        out_sdf[tid] = 1.0e6
        return
    closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    d = wp.length(p - closest)
    s = -1.0 if q.sign < 0.0 else 1.0
    out_sdf[tid] = d * s


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SdfResult:
    """Holds the output of a voxel-grid SDF computation.

    The grid may be **anisotropic** in voxel *count*: ``n`` is the resolution of
    the longest axis, while ``nx/ny/nz`` are the per-axis counts (shorter axes
    get fewer voxels at the same ``dx``).  ``grid.shape == (nz, ny, nx)``.
    """
    grid: np.ndarray          # (nz, ny, nx) float32
    n: int                    # longest-axis resolution (= max(nx, ny, nz))
    dx: float
    origin: np.ndarray        # (3,) float32  – world-space corner
    aabb_min: np.ndarray      # (3,) float32
    aabb_max: np.ndarray      # (3,) float32
    thickness: np.ndarray | None = None  # (nz, ny, nx) float32 local feature thickness
    blowup_thickness: np.ndarray | None = None  # thickness carried into exterior blowup band
    nx: int = 0               # per-axis voxel counts; 0 → fall back to ``n``
    ny: int = 0
    nz: int = 0
    thickness_stride_vox: float = 1.0
    blowup_thickness_extent_vox: float = 0.0
    # Largest |thickness fraction| for which ``blowup_thickness`` was carried
    # far enough into the exterior.  ``None`` means legacy/unknown metadata.
    blowup_thickness_capacity_fraction: float | None = None

    def __post_init__(self):
        # Back-fill per-axis counts for cubic results / older call sites.
        if self.nx == 0:
            self.nx = self.n
        if self.ny == 0:
            self.ny = self.n
        if self.nz == 0:
            self.nz = self.n
        if self.blowup_thickness is not None:
            blowup_thickness = np.asarray(
                self.blowup_thickness, dtype=np.float32)
            if blowup_thickness.shape != np.asarray(self.grid).shape:
                raise ValueError(
                    "blowup_thickness must have the same shape as grid")
            self.blowup_thickness = np.ascontiguousarray(blowup_thickness)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape as ``(nz, ny, nx)`` — matches ``grid.shape``."""
        return (self.nz, self.ny, self.nx)


def _sample_voxel_field_trilinear(
    field: np.ndarray,
    origin: np.ndarray,
    spacing: float | np.ndarray,
    points: np.ndarray,
    *,
    chunk_size: int = 262_144,
    progress_cb=None,
) -> np.ndarray:
    """Sample a voxel-centred ``(nz, ny, nx)`` field at world-space points.

    ``origin`` is the grid *corner*, matching :class:`SdfResult`; consequently
    voxel ``(0, 0, 0)`` is centred at ``origin + 0.5 * spacing``.  A scalar
    spacing covers the regular production grids.  A 3-vector is accepted for
    the anisotropic coarse lattice used by the sparse-only fallback.

    Values outside the grid's cell bounds are zero.  Processing in chunks keeps
    temporary index/weight arrays bounded even for million-sample clouds.
    """
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("voxel field must have shape (nz, ny, nx)")
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.empty(0, dtype=np.float32)
    org = np.asarray(origin, dtype=np.float64).reshape(3)
    step = np.asarray(spacing, dtype=np.float64)
    if step.ndim == 0:
        step = np.repeat(step, 3)
    step = step.reshape(3)
    if not np.isfinite(step).all() or np.any(step <= 0.0):
        raise ValueError("voxel spacing must be finite and positive")

    # Coordinates and shape are xyz here; the ndarray itself is indexed zyx.
    shape_xyz = np.asarray(values.shape[::-1], dtype=np.int64)
    out = np.zeros(len(pts), dtype=np.float32)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(pts), chunk_size):
        end = min(len(pts), start + chunk_size)
        coord = (pts[start:end].astype(np.float64) - org) / step - 0.5
        valid_point = np.all(
            (coord >= -0.5) & (coord <= shape_xyz.astype(np.float64) - 0.5),
            axis=1,
        )
        # Cell-boundary points use constant extension of the nearest voxel;
        # points farther outside remain zero via ``valid_point``.
        coord = np.clip(coord, 0.0, shape_xyz.astype(np.float64) - 1.0)
        lo = np.floor(coord).astype(np.int64)
        hi = np.minimum(lo + 1, shape_xyz - 1)
        frac = (coord - lo).astype(np.float32)
        accum = np.zeros(end - start, dtype=np.float32)
        for dz in (0, 1):
            iz = hi[:, 2] if dz else lo[:, 2]
            wz = frac[:, 2] if dz else (1.0 - frac[:, 2])
            for dy in (0, 1):
                iy = hi[:, 1] if dy else lo[:, 1]
                wy = frac[:, 1] if dy else (1.0 - frac[:, 1])
                for dx_i in (0, 1):
                    ix = hi[:, 0] if dx_i else lo[:, 0]
                    wx = frac[:, 0] if dx_i else (1.0 - frac[:, 0])
                    accum += values[iz, iy, ix] * (wx * wy * wz)
        accum[~valid_point] = 0.0
        out[start:end] = accum
        if progress_cb is not None:
            progress_cb(
                end / max(1, len(pts)),
                f"Thickness vertices {end:,}/{len(pts):,}",
            )
    return out


def _sample_zero_dilated_field_trilinear(
    field: np.ndarray,
    origin: np.ndarray,
    spacing: float | np.ndarray,
    points: np.ndarray,
    *,
    dilation_iters: int = 2,
    chunk_size: int = 65_536,
    progress_cb=None,
) -> np.ndarray:
    """Sample ``dilate_zeros(field, iters)`` without materialising that grid.

    ``dilate_zeros`` freezes a voxel as soon as a positive 6-neighbour reaches
    it.  Therefore an initially-zero queried corner receives the maximum value
    on the *nearest* positive Manhattan shell, up to ``dilation_iters``.  Only
    the eight interpolation corners and their small neighbourhoods are read;
    the production 512³ thickness volume is never copied or scanned.
    """
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("voxel field must have shape (nz, ny, nx)")
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.empty(0, dtype=np.float32)
    iters = max(0, int(dilation_iters))
    org = np.asarray(origin, dtype=np.float64).reshape(3)
    step = np.asarray(spacing, dtype=np.float64)
    if step.ndim == 0:
        step = np.repeat(step, 3)
    step = step.reshape(3)
    if not np.isfinite(step).all() or np.any(step <= 0.0):
        raise ValueError("voxel spacing must be finite and positive")

    shape_xyz = np.asarray(values.shape[::-1], dtype=np.int64)
    # Shell order matters: a value propagated on an earlier iteration becomes
    # non-zero and is deliberately not replaced by a larger, more distant one.
    shells = [
        tuple(
            (dz, dy, dx_i)
            for dz in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dx_i in range(-radius, radius + 1)
            if abs(dx_i) + abs(dy) + abs(dz) == radius
        )
        for radius in range(1, iters + 1)
    ]

    def _corner_values(
        iz: np.ndarray,
        iy: np.ndarray,
        ix: np.ndarray,
    ) -> np.ndarray:
        sampled = values[iz, iy, ix].copy()
        unresolved = sampled == 0.0
        for shell in shells:
            if not np.any(unresolved):
                break
            shell_max = np.zeros_like(sampled)
            for dz, dy, dx_i in shell:
                qz = iz + dz
                qy = iy + dy
                qx = ix + dx_i
                valid = (
                    (qz >= 0) & (qz < values.shape[0])
                    & (qy >= 0) & (qy < values.shape[1])
                    & (qx >= 0) & (qx < values.shape[2])
                    & unresolved
                )
                if np.any(valid):
                    shell_max[valid] = np.maximum(
                        shell_max[valid],
                        values[qz[valid], qy[valid], qx[valid]],
                    )
            reached = unresolved & (shell_max > 0.0)
            sampled[reached] = shell_max[reached]
            unresolved[reached] = False
        return sampled

    out = np.zeros(len(pts), dtype=np.float32)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(pts), chunk_size):
        end = min(len(pts), start + chunk_size)
        coord = (pts[start:end].astype(np.float64) - org) / step - 0.5
        valid_point = np.all(
            (coord >= -0.5)
            & (coord <= shape_xyz.astype(np.float64) - 0.5),
            axis=1,
        )
        coord = np.clip(coord, 0.0, shape_xyz.astype(np.float64) - 1.0)
        lo = np.floor(coord).astype(np.int64)
        hi = np.minimum(lo + 1, shape_xyz - 1)
        frac = (coord - lo).astype(np.float32)
        accum = np.zeros(end - start, dtype=np.float32)
        for dz in (0, 1):
            iz = hi[:, 2] if dz else lo[:, 2]
            wz = frac[:, 2] if dz else (1.0 - frac[:, 2])
            for dy in (0, 1):
                iy = hi[:, 1] if dy else lo[:, 1]
                wy = frac[:, 1] if dy else (1.0 - frac[:, 1])
                for dx_i in (0, 1):
                    ix = hi[:, 0] if dx_i else lo[:, 0]
                    wx = frac[:, 0] if dx_i else (1.0 - frac[:, 0])
                    accum += _corner_values(iz, iy, ix) * (wx * wy * wz)
        accum[~valid_point] = 0.0
        out[start:end] = accum
        if progress_cb is not None:
            progress_cb(
                end / max(1, len(pts)),
                f"Thickness vertices {end:,}/{len(pts):,}",
            )
    return out


def sample_thickness_at_vertices(
    result: SdfResult,
    vertices: np.ndarray,
    *,
    progress_cb=None,
) -> np.ndarray:
    """Sample an existing dense feature-thickness field at mesh vertices.

    The exterior surface carrier is preferred when it is available.  Legacy
    :class:`SdfResult` instances that only contain the local thickness field use
    the same two-voxel compatibility dilation as sparse sampling.  This helper
    is intended for seeding a topology-stable per-vertex cache before later
    mesh poses transport those values with their vertices.
    """
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if not np.isfinite(points).all():
        raise ValueError("vertices must be finite")

    carried = getattr(result, "blowup_thickness", None)
    if carried is not None:
        source = np.asarray(carried, dtype=np.float32)
        sample = _sample_voxel_field_trilinear
    else:
        local = getattr(result, "thickness", None)
        if local is None:
            raise ValueError("SdfResult has no feature-thickness field")
        source = np.asarray(local, dtype=np.float32)
        sample = _sample_zero_dilated_field_trilinear

    if source.shape != np.asarray(result.grid).shape:
        raise ValueError("SdfResult grid/thickness shape mismatch")
    if not np.isfinite(source).all() or np.any(source < 0.0):
        raise ValueError("SdfResult thickness must be finite and nonnegative")

    sampled = sample(
        source,
        np.asarray(result.origin, dtype=np.float32),
        float(result.dx),
        points,
        progress_cb=progress_cb,
    )
    return np.ascontiguousarray(sampled, dtype=np.float32)


def sample_thickness_at_face_corners(
    result: SdfResult,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    progress_cb=None,
) -> np.ndarray:
    """Sample three topology-stable, edge-safe feature values per triangle.

    Each value is sampled inside its triangle at barycentric coordinates
    ``(.5, .25, .25)`` and permutations.  Equivalently, a probe lies at
    ``.25 * corner + .75 * centroid``.  A short two-voxel line in the inward
    direction (opposite the oriented face normal) is reduced by maximum; this
    avoids a surface-alignment zero without reaching across the exterior gap to
    a nearby disconnected surface.  The probe values are then inverted back to
    true affine corner values via ``t_i = 4*s_i - sum(s)``.  Noisy inversions
    that approach or cross zero are blended face-locally back toward the robust
    probe values.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.isfinite(verts).all():
        raise ValueError("vertices must be finite")
    if tris.size and (int(tris.min()) < 0 or int(tris.max()) >= len(verts)):
        raise ValueError("faces contain an out-of-range vertex index")

    triangle_vertices = verts[tris]
    triangle_sum = np.sum(
        triangle_vertices, axis=1, keepdims=True, dtype=np.float32)
    probes = 0.25 * (triangle_vertices + triangle_sum)
    face_normals = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    normal_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(normal_lengths, 1.0e-12)
    # ``load_and_prepare_arrays`` orients production faces outwards.  If a
    # caller supplies inconsistent winding, unresolved values fail the later
    # per-face cache validation instead of borrowing from another component.
    normal_offsets = np.asarray((0.0, -1.0, -2.0), dtype=np.float32)
    probe_layers = (
        probes[None, :, :, :]
        + normal_offsets[:, None, None, None]
        * np.float32(result.dx)
        * face_normals[None, :, None, :]
    )
    flat_probes = np.ascontiguousarray(
        probe_layers.reshape(-1, 3), dtype=np.float32)

    def _face_progress(frac: float, _msg: str) -> None:
        if progress_cb is not None:
            completed = min(
                len(flat_probes), int(np.ceil(float(frac) * len(flat_probes))))
            progress_cb(
                float(frac),
                f"Thickness face corners {completed:,}/{len(flat_probes):,}",
            )

    carried = getattr(result, "blowup_thickness", None)
    if carried is not None:
        source = np.asarray(carried, dtype=np.float32)
        sample = _sample_voxel_field_trilinear
    else:
        local = getattr(result, "thickness", None)
        if local is None:
            raise ValueError("SdfResult has no feature-thickness field")
        source = np.asarray(local, dtype=np.float32)
        sample = _sample_zero_dilated_field_trilinear

    if source.shape != np.asarray(result.grid).shape:
        raise ValueError("SdfResult grid/thickness shape mismatch")
    if not np.isfinite(source).all() or np.any(source < 0.0):
        raise ValueError("SdfResult thickness must be finite and nonnegative")

    sampled = sample(
        source,
        np.asarray(result.origin, dtype=np.float32),
        float(result.dx),
        flat_probes,
        progress_cb=_face_progress if progress_cb is not None else None,
    )
    sampled_layers = np.asarray(sampled, dtype=np.float32).reshape(
        len(normal_offsets), len(tris), 3)
    probe_values = np.max(sampled_layers, axis=0).astype(
        np.float64, copy=False)

    # Repair isolated unresolved probes only from their two values on the same
    # face.  A completely unresolved face deliberately remains zero so the
    # batch cache can reject it and recompute thickness for the next pose.
    positive = np.isfinite(probe_values) & (probe_values > 0.0)
    face_has_value = np.any(positive, axis=1)
    positive_count = np.maximum(np.sum(positive, axis=1), 1)
    face_replacement = (
        np.sum(np.where(positive, probe_values, 0.0), axis=1)
        / positive_count
    )
    repaired_probes = np.where(
        positive,
        probe_values,
        face_replacement[:, None],
    )
    repaired_probes[~face_has_value] = 0.0

    probe_sum = np.sum(repaired_probes, axis=1, keepdims=True)
    reconstructed = 4.0 * repaired_probes - probe_sum
    tiny = float(np.finfo(np.float32).tiny)
    face_scale = np.mean(repaired_probes, axis=1)
    min_probe = np.min(repaired_probes, axis=1)
    positive_floor = np.maximum(
        tiny,
        np.minimum(1.0e-6 * face_scale, 0.5 * min_probe),
    )
    unstable = (
        ~np.isfinite(reconstructed)
        | (reconstructed <= positive_floor[:, None])
    )
    denominators = repaired_probes - reconstructed
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        alpha_limit = np.where(
            unstable & (denominators > 0.0),
            (repaired_probes - positive_floor[:, None]) / denominators,
            1.0,
        )
    alpha_limit[~np.isfinite(alpha_limit)] = 0.0
    blend_alpha = np.clip(np.min(alpha_limit, axis=1), 0.0, 1.0)
    # Do not stop at the numerical zero boundary: noisy probe triples would
    # otherwise produce technically-positive but practically useless corners.
    # Move halfway from that boundary back toward the measured face probes.
    blend_alpha[np.any(unstable, axis=1)] *= 0.5
    corner_values = (
        repaired_probes
        + blend_alpha[:, None] * (reconstructed - repaired_probes)
    )
    corner_values = np.maximum(corner_values, positive_floor[:, None])
    float32_max = float(np.finfo(np.float32).max)
    invalid_corner = (
        ~np.isfinite(corner_values) | (corner_values > float32_max)
    )
    corner_values = np.where(
        invalid_corner, repaired_probes, corner_values)
    corner_values[~face_has_value] = 0.0
    return np.ascontiguousarray(corner_values, dtype=np.float32)


# ── SDF computer ──────────────────────────────────────────────────────────────

class SdfComputer:
    """
    Manages a Warp mesh and exposes SDF query methods.

    Usage:
        comp = SdfComputer(device="cuda")
        comp.set_mesh(verts, faces)
        result = comp.compute_voxel_grid(n=128)
        val    = comp.query_point([0.0, 0.0, 0.0])
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if wp.is_cuda_available() else "cpu")
        self._warp_mesh: Optional[wp.Mesh] = None
        self._points_wp = None
        self._indices_wp = None
        self._verts: Optional[np.ndarray] = None
        self._faces: Optional[np.ndarray] = None
        self._topology_key: tuple[tuple[int, ...], bytes] | None = None
        # Whether the mesh is closed (every edge shared by exactly 2 faces).  A
        # non-watertight mesh has an unreliable normal-based inside/outside sign,
        # so we switch to the (slower but robust) winding-number sign for it.
        self._watertight: bool = True
        # These counters make the allocation/readback savings observable without
        # putting timing probes or synchronisation into the hot path.  A single
        # SdfComputer is deliberately serial; sharing it across concurrent Warp
        # streams would make the mutable mesh and scratch buffers race.
        self._reuse_stats = {
            "set_mesh_calls": 0,
            "topology_cache_hits": 0,
            "watertight_checks": 0,
            "mesh_rebuilds": 0,
            "mesh_refits": 0,
            "reused_vertex_bytes": 0,
            "reused_index_bytes": 0,
            "grid_kernel_launches": 0,
            "grid_buffer_allocations": 0,
            "grid_progress_syncs": 0,
            "grid_host_readbacks": 0,
        }

    # ── mesh management ───────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._warp_mesh is not None and self._verts is not None

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray) -> None:
        """
        Upload a triangle mesh to the Warp device.

        Args:
            verts: (V, 3) float32
            faces: (F, 3) int32
        """
        # Own both arrays from the start.  API decoders and pose producers may
        # reuse their input buffers immediately after this call; aliasing them
        # would let an external in-place edit desynchronise CPU sampling from the
        # uploaded points/BVH (and would make refit rollback unreliable).
        vertices = np.array(
            verts, dtype=np.float32, order="C", copy=True).reshape(-1, 3)
        face_array = np.array(
            faces, dtype=np.int32, order="C", copy=True).reshape(-1, 3)
        if len(vertices) == 0:
            raise ValueError("verts must contain at least one vertex")
        if len(face_array) == 0:
            raise ValueError("faces must contain at least one triangle")

        self._reuse_stats["set_mesh_calls"] += 1
        topology_key = self._topology_fingerprint(face_array)
        same_topology = (
            self._topology_key == topology_key
            and self._faces is not None
            and self._indices_wp is not None
        )
        previous_vertex_count = (
            int(self._points_wp.shape[0]) if self._points_wp is not None else -1)
        if ((not same_topology or previous_vertex_count != len(vertices))
                and (np.any(face_array < 0)
                     or np.any(face_array >= len(vertices)))):
            raise ValueError("faces contain an out-of-range vertex index")

        can_refit = (
            same_topology
            and self._warp_mesh is not None
            and previous_vertex_count == len(vertices)
            and callable(getattr(self._warp_mesh, "refit", None))
        )

        if can_refit:
            # ``assign`` reuses the existing device allocation.  Warp guarantees
            # that Mesh.refit() refreshes the BVH after its points are modified;
            # both operations are submitted to the same device stream.  Restore
            # the old points best-effort when either operation raises so the CPU
            # cache never advertises vertices that the BVH did not accept.
            old_vertices = self._verts
            try:
                self._points_wp.assign(vertices)
                self._warp_mesh.refit()
            except Exception:
                if old_vertices is not None:
                    try:
                        self._points_wp.assign(old_vertices)
                        self._warp_mesh.refit()
                    except Exception:
                        # Preserve the original exception.  A second failure can
                        # only mean Warp left this mesh unusable; callers will not
                        # see a falsely committed CPU/topology state.
                        pass
                raise
            self._verts = vertices
            self._reuse_stats["topology_cache_hits"] += 1
            self._reuse_stats["reused_index_bytes"] += int(face_array.nbytes)
            self._reuse_stats["mesh_refits"] += 1
            self._reuse_stats["reused_vertex_bytes"] += int(vertices.nbytes)
            return

        # Build every topology-dependent resource in locals.  Publishing the
        # fields only after Mesh construction succeeds makes a failed upload or
        # BVH build an atomic no-op from the caller's point of view.
        if same_topology:
            new_faces = self._faces
            new_indices_wp = self._indices_wp
            new_watertight = self._watertight
        else:
            new_faces = face_array.copy()
            new_watertight = self._is_watertight(new_faces)
            new_indices_wp = wp.array(
                new_faces.reshape(-1), dtype=wp.int32, device=self.device)
        new_points_wp = wp.array(
            vertices, dtype=wp.vec3, device=self.device)
        new_warp_mesh = wp.Mesh(
            points=new_points_wp,
            indices=new_indices_wp,
            support_winding_number=not new_watertight,
        )

        self._verts = vertices
        self._points_wp = new_points_wp
        self._warp_mesh = new_warp_mesh
        if same_topology:
            self._reuse_stats["topology_cache_hits"] += 1
            self._reuse_stats["reused_index_bytes"] += int(face_array.nbytes)
        else:
            self._faces = new_faces
            self._topology_key = topology_key
            self._watertight = new_watertight
            self._indices_wp = new_indices_wp
            self._reuse_stats["watertight_checks"] += 1
        self._reuse_stats["mesh_rebuilds"] += 1

    @staticmethod
    def _topology_fingerprint(faces: np.ndarray) -> tuple[tuple[int, ...], bytes]:
        """Return a cheap content key for canonical contiguous int32 faces.

        Hashing is linear and allocation-free (apart from the small digest),
        unlike the sort/unique watertightness test.  The strong 128-bit digest
        lets independently decoded per-pose arrays hit the same cache.
        """
        canonical = np.ascontiguousarray(faces, dtype=np.int32).reshape(-1, 3)
        digest = hashlib.blake2b(
            memoryview(canonical).cast("B"), digest_size=16).digest()
        return canonical.shape, digest

    @property
    def reuse_stats(self) -> dict[str, int]:
        """Snapshot of safe reuse and unavoidable device-work counters.

        ``mesh_refits`` measures avoided Mesh/BVH reconstructions.  The grid
        counters expose how many device allocations, kernel submissions and
        blocking host readbacks were needed; chunked progress deliberately uses
        one output allocation/readback for all slabs.
        """
        return dict(self._reuse_stats)

    @staticmethod
    def _is_watertight(faces: np.ndarray) -> bool:
        """True if every undirected edge is shared by exactly two faces."""
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(f) == 0:
            return False
        edges = np.sort(
            np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0),
            axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    @property
    def _winding_flag(self) -> int:
        """0 = fast normal-based sign (watertight), 1 = robust winding number."""
        return 0 if self._watertight else 1

    def clear(self) -> None:
        self._warp_mesh = None
        self._points_wp = None
        self._indices_wp = None
        self._verts = None
        self._faces = None
        self._topology_key = None

    # ── single-point query ────────────────────────────────────────────────

    def query_point(self, p_xyz) -> float:
        """Return the signed distance at a single world-space point."""
        self._check_ready()
        p = wp.vec3(float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2]))
        out = wp.zeros(1, dtype=wp.float32, device=self.device)
        wp.launch(
            kernel=_sdf_one_point_kernel,
            dim=1,
            inputs=[self._warp_mesh.id, p, self._winding_flag, out],
            device=self.device,
        )
        return float(out.numpy()[0])

    def query_points(self, points: np.ndarray, max_dist: float = 1.0e6) -> np.ndarray:
        """Return signed distances for many world-space points."""
        self._check_ready()
        pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            return np.empty(0, dtype=np.float32)
        pts_wp = wp.array(pts, dtype=wp.vec3, device=self.device)
        out = wp.empty(pts.shape[0], dtype=wp.float32, device=self.device)
        wp.launch(
            kernel=_sdf_points_kernel,
            dim=pts.shape[0],
            inputs=[self._warp_mesh.id, pts_wp, float(max_dist),
                    self._winding_flag, out],
            device=self.device,
        )
        return out.numpy().astype(np.float32, copy=False)

    def compute_sparse_samples(
        self,
        n: int,
        margin: float = 0.5,
        surface_samples: int | None = None,
        offsets_vox: tuple[float, ...] = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
        coarse_n: int = 32,
        max_dist: float | None = None,
        seed: int = 12345,
        progress_cb=None,
        thickness_result: SdfResult | None = None,
        vertex_thickness: np.ndarray | None = None,
        face_corner_thickness: np.ndarray | None = None,
        thin_surface_fraction: float | None = 0.0,
        thickness_sampling_power: float | None = None,
    ) -> SdfSampleSet:
        """Compute sparse training samples concentrated around mesh triangles.

        The sample cloud has high density in a narrow band around the surface
        and a low-density coarse lattice through the padded AABB.  It is intended
        for optimizer loss batches, not for UI slicing.

        When ``vertex_thickness`` provides one feature-thickness value per mesh
        vertex, or ``face_corner_thickness`` provides three independent values
        per face, the sampled triangle's barycentric weights transport it to the
        surface point.  The two representations are mutually exclusive.  Every
        normal offset of that point receives the same value.  Coarse points
        remain zero unless ``thickness_result`` contains a transported blowup
        carrier, in which case they sample that bounded field.  This path
        performs no local-thickness computation.

        Otherwise, when ``thickness_result`` provides a dense local-thickness
        field, that field is sampled trilinearly.  Standalone sparse calls
        derive a lower-resolution thickness field from their coarse SDF lattice.

        ``thickness_sampling_power`` continuously changes the triangle draw from
        area-uniform sampling (0) to area / local-thickness (1), or stronger
        inverse powers.  No binary thin/thick classification is involved and
        every triangle with non-zero area retains non-zero probability.
        """
        self._check_ready()
        if self._faces is None or self._verts is None:
            raise RuntimeError("No mesh loaded. Call set_mesh() first.")
        if progress_cb is not None:
            progress_cb(0.0, "Preparing sparse SDF samples ...")

        verts = np.asarray(self._verts, dtype=np.float32)
        faces = np.asarray(self._faces, dtype=np.int32)
        if vertex_thickness is not None and face_corner_thickness is not None:
            raise ValueError(
                "vertex_thickness and face_corner_thickness are mutually "
                "exclusive")
        if thickness_sampling_power is None:
            thickness_sampling_power = (
                0.0 if thin_surface_fraction is None
                else float(np.clip(
                    float(thin_surface_fraction), 0.0, 1.0)))
        sampling_power = float(thickness_sampling_power)
        if not np.isfinite(sampling_power) or sampling_power < 0.0:
            raise ValueError(
                "thickness_sampling_power must be finite and non-negative")
        vertex_thickness_values = None
        if vertex_thickness is not None:
            vertex_thickness_values = np.asarray(
                vertex_thickness, dtype=np.float32).reshape(-1)
            if vertex_thickness_values.shape[0] != verts.shape[0]:
                raise ValueError(
                    "vertex_thickness must contain one value per mesh vertex")
            if not np.isfinite(vertex_thickness_values).all():
                raise ValueError("vertex_thickness must be finite")
            if np.any(vertex_thickness_values < 0.0):
                raise ValueError("vertex_thickness must be nonnegative")
            vertex_thickness_values = np.ascontiguousarray(
                vertex_thickness_values, dtype=np.float32)
        face_corner_thickness_values = None
        if face_corner_thickness is not None:
            face_corner_thickness_values = np.asarray(
                face_corner_thickness, dtype=np.float32).reshape(-1)
            expected = int(faces.shape[0]) * 3
            if face_corner_thickness_values.size != expected:
                raise ValueError(
                    "face_corner_thickness must contain exactly three values "
                    "per mesh face")
            if not np.isfinite(face_corner_thickness_values).all():
                raise ValueError("face_corner_thickness must be finite")
            if np.any(face_corner_thickness_values < 0.0):
                raise ValueError("face_corner_thickness must be nonnegative")
            face_corner_thickness_values = np.ascontiguousarray(
                face_corner_thickness_values.reshape(-1, 3),
                dtype=np.float32,
            )
        vmin = verts.min(axis=0).astype(np.float32)
        vmax = verts.max(axis=0).astype(np.float32)
        extent = vmax - vmin
        max_extent = float(extent.max())
        if max_extent <= 0.0:
            raise ValueError("Degenerate AABB (extent <= 0).")

        if thickness_result is not None:
            # Reuse the dense target's exact lattice.  Its margin may have been
            # enlarged automatically for a relative blowup; recomputing from
            # the UI margin here would give sparse samples a different dx/AABB.
            dx = float(thickness_result.dx)
            counts = np.array(
                [thickness_result.nx,
                 thickness_result.ny,
                 thickness_result.nz],
                dtype=np.int64,
            )
            aabb_min = np.asarray(
                thickness_result.origin, dtype=np.float32).reshape(3)
            aabb_max = (
                aabb_min.astype(np.float64)
                + counts.astype(np.float64) * dx
            ).astype(np.float32)
            padded_max = float(np.max(counts)) * dx
        else:
            padded_max = max_extent * (1.0 + float(margin))
            dx = padded_max / float(n)
            padded = (extent * (1.0 + float(margin))).astype(np.float64)
            counts = np.maximum(1, np.ceil(padded / dx).astype(np.int64))
            center = 0.5 * (vmin + vmax)
            half = 0.5 * counts.astype(np.float64) * dx
            aabb_min = (center - half).astype(np.float32)
            aabb_max = (center + half).astype(np.float32)

        tri = verts[faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        cross = np.cross(e1, e2)
        double_area = np.linalg.norm(cross, axis=1)
        valid = double_area > 1e-12
        if not np.any(valid):
            raise ValueError("Mesh has no non-degenerate triangles.")
        tri = tri[valid]
        if face_corner_thickness_values is not None:
            triangle_thickness = face_corner_thickness_values[valid]
        elif vertex_thickness_values is not None:
            triangle_thickness = vertex_thickness_values[faces[valid]]
        else:
            triangle_thickness = None
        cross = cross[valid]
        area = (double_area[valid].astype(np.float64) * 0.5)
        normals = cross / np.maximum(double_area[valid, None], 1e-12)

        # The coarse lattice is independent of the random surface draw.  Build
        # it first so sparse-only callers can derive a cheap thickness guide for
        # that draw instead of discovering small structures only afterwards.
        coarse_n = max(4, int(coarse_n))
        coarse_counts = np.maximum(
            1,
            np.ceil(counts * (coarse_n / max(float(counts.max()), 1.0))).astype(np.int64),
        )
        gx = np.linspace(aabb_min[0], aabb_max[0], int(coarse_counts[0]), endpoint=False,
                         dtype=np.float32) + 0.5 * (aabb_max[0] - aabb_min[0]) / int(coarse_counts[0])
        gy = np.linspace(aabb_min[1], aabb_max[1], int(coarse_counts[1]), endpoint=False,
                         dtype=np.float32) + 0.5 * (aabb_max[1] - aabb_min[1]) / int(coarse_counts[1])
        gz = np.linspace(aabb_min[2], aabb_max[2], int(coarse_counts[2]), endpoint=False,
                         dtype=np.float32) + 0.5 * (aabb_max[2] - aabb_min[2]) / int(coarse_counts[2])
        zz, yy, xx = np.meshgrid(gz, gy, gx, indexing="ij")
        coarse_points = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
        effective_max_dist = (
            max(16.0 * float(dx), 1.2 * padded_max)
            if max_dist is None else float(max_dist))
        precomputed_coarse_values = None
        sparse_only_thickness_field = None
        sparse_only_thickness_spacing = None

        rng = np.random.default_rng(seed)
        if surface_samples is None:
            # O(n^2) surface density instead of O(n^3) volume density.
            surface_samples = int(np.clip(2 * int(n) * int(n), 8192, 160_000))
        area_sum = float(area.sum(dtype=np.float64))
        if not np.isfinite(area_sum) or area_sum <= 1e-12:
            raise ValueError("Mesh has no finite triangle area.")
        area_probs = area / area_sum
        area_probs = area_probs / float(area_probs.sum(dtype=np.float64))
        surface_samples = int(surface_samples)
        if surface_samples < 0:
            raise ValueError("surface_samples must be non-negative")

        # Determine continuous triangle importance before the area draw.
        # Otherwise a small finger can receive too few distinct sparse points
        # and no later mini-batch weighting can recover its geometry.  Explicit
        # vertex/face-corner thickness is preferred; the dense carrier is the
        # fallback.
        sampling_triangle_thickness = triangle_thickness
        triangle_probe_points = None
        if sampling_triangle_thickness is None and sampling_power > 0.0:
            triangle_probe_points = 0.25 * (
                tri + np.sum(
                    tri, axis=1, keepdims=True, dtype=np.float32))
        if (sampling_triangle_thickness is None and sampling_power > 0.0
                and thickness_result is not None
                and (getattr(thickness_result, "blowup_thickness", None) is not None
                     or getattr(thickness_result, "thickness", None) is not None)):
            def _thickness_weighting_progress(fraction, message):
                if progress_cb is not None:
                    progress_cb(
                        0.10 * float(fraction),
                        f"Weighting surface thickness · {message}",
                    )

            sampled_corners = sample_thickness_at_vertices(
                thickness_result,
                np.ascontiguousarray(
                    triangle_probe_points.reshape(-1, 3), dtype=np.float32),
                progress_cb=_thickness_weighting_progress,
            )
            sampling_triangle_thickness = sampled_corners.reshape(-1, 3)
        elif sampling_triangle_thickness is None and sampling_power > 0.0:
            # Sparse-only path: query the already-required coarse lattice first,
            # derive its bounded-cost thickness field, and use that to stratify
            # triangles.  The values/field are retained below, so this does not
            # duplicate either the coarse mesh queries or thickness analysis.
            if progress_cb is not None:
                progress_cb(0.02, "Weighting surface thickness · coarse SDF")
            precomputed_coarse_values = self.query_points(
                coarse_points, max_dist=effective_max_dist)
            coarse_shape = (
                int(coarse_counts[2]),
                int(coarse_counts[1]),
                int(coarse_counts[0]),
            )
            coarse_grid = precomputed_coarse_values.reshape(coarse_shape)
            sparse_only_thickness_spacing = (
                (aabb_max.astype(np.float64) - aabb_min.astype(np.float64))
                / coarse_counts.astype(np.float64)
            )
            sparse_only_thickness_field = dilate_zeros(
                local_thickness(
                    coarse_grid,
                    float(np.max(sparse_only_thickness_spacing)),
                ),
                iters=2,
            )
            sampled_corners = _sample_voxel_field_trilinear(
                sparse_only_thickness_field,
                aabb_min,
                sparse_only_thickness_spacing,
                np.ascontiguousarray(
                    triangle_probe_points.reshape(-1, 3), dtype=np.float32),
            )
            sampling_triangle_thickness = sampled_corners.reshape(-1, 3)

        sampling_probs = area_probs
        applied_sampling_power = 0.0
        sampling_corner_factors = None
        if (sampling_power > 0.0
                and sampling_triangle_thickness is not None):
            face_thickness = np.asarray(
                sampling_triangle_thickness, dtype=np.float32).reshape(-1, 3)
            positive = np.isfinite(face_thickness) & (face_thickness > 0.0)
            positive_count = np.sum(positive, axis=1)
            representative = np.zeros(len(tri), dtype=np.float64)
            np.divide(
                np.sum(np.where(positive, face_thickness, 0.0), axis=1,
                       dtype=np.float64),
                positive_count,
                out=representative,
                where=positive_count > 0,
            )
            if np.any(positive):
                reference = positive_weighted_median(
                    representative, area)
                corner_factors = inverse_thickness_factors(
                    face_thickness,
                    sampling_power,
                    reference=reference,
                )
                triangle_factors = np.mean(
                    corner_factors, axis=1, dtype=np.float64)
                mass = area.astype(np.float64) * triangle_factors
                mass_sum = float(mass.sum(dtype=np.float64))
                if np.isfinite(mass_sum) and mass_sum > 0.0:
                    sampling_probs = mass / mass_sum
                    sampling_corner_factors = corner_factors
                    applied_sampling_power = sampling_power

        # One continuous categorical draw; no thin/regular quotas or threshold.
        tri_idx = rng.choice(
            len(tri), size=surface_samples, replace=True, p=sampling_probs)
        if sampling_corner_factors is None:
            r1 = rng.random(surface_samples, dtype=np.float32)
            r2 = rng.random(surface_samples, dtype=np.float32)
            sr1 = np.sqrt(r1).astype(np.float32, copy=False)
            b0 = 1.0 - sr1
            b1 = sr1 * (1.0 - r2)
            b2 = sr1 * r2
        else:
            # Approximate the inverse-thickness field linearly over each face.
            # Sampling a linear barycentric density is exact as a mixture of
            # Dirichlet(2,1,1) distributions, one per corner.  This avoids
            # choosing the right triangle but then sampling its thick side just
            # as often as its thin side.
            selected_factors = np.asarray(
                sampling_corner_factors[tri_idx], dtype=np.float64)
            factor_sum = selected_factors.sum(axis=1, dtype=np.float64)
            component_draw = rng.random(surface_samples) * factor_sum
            component = np.sum(
                component_draw[:, None]
                > np.cumsum(selected_factors, axis=1),
                axis=1,
            )
            component = np.minimum(component, 2)
            gamma = rng.exponential(size=(surface_samples, 3))
            rows = np.arange(surface_samples)
            gamma[rows, component] += rng.exponential(size=surface_samples)
            barycentric = gamma / gamma.sum(axis=1, keepdims=True)
            b0 = barycentric[:, 0].astype(np.float32)
            b1 = barycentric[:, 1].astype(np.float32)
            b2 = barycentric[:, 2].astype(np.float32)
        base = (tri[tri_idx, 0] * b0[:, None]
                + tri[tri_idx, 1] * b1[:, None]
                + tri[tri_idx, 2] * b2[:, None]).astype(np.float32)
        base_transported_thickness = None
        if triangle_thickness is not None:
            sampled_triangle_thickness = triangle_thickness[tri_idx]
            base_transported_thickness = (
                sampled_triangle_thickness[:, 0] * b0
                + sampled_triangle_thickness[:, 1] * b1
                + sampled_triangle_thickness[:, 2] * b2
            ).astype(np.float32, copy=False)
        nrm = normals[tri_idx].astype(np.float32)

        offsets = np.asarray(offsets_vox, dtype=np.float32) * np.float32(dx)
        band_points = [
            (base + off * nrm).astype(np.float32)
            for off in offsets
        ]

        band_count = sum(len(part) for part in band_points)
        points = np.concatenate(band_points + [coarse_points], axis=0)
        coarse_mask = np.zeros(points.shape[0], dtype=np.bool_)
        coarse_mask[band_count:] = True
        if progress_cb is not None:
            progress_cb(0.45, f"Querying {len(points):,} sparse SDF samples ...")
        pending_points = (
            points[:band_count]
            if precomputed_coarse_values is not None else points)
        if progress_cb is not None and len(pending_points) > 65_536:
            vals = np.empty(len(pending_points), dtype=np.float32)
            chunk = 65_536
            for start in range(0, len(pending_points), chunk):
                end = min(len(pending_points), start + chunk)
                vals[start:end] = self.query_points(
                    pending_points[start:end],
                    max_dist=effective_max_dist,
                )
                progress_cb(
                    0.45 + 0.50 * (end / max(1, len(pending_points))),
                    f"Sparse query {end:,}/{len(pending_points):,} samples",
                )
            pending_values = vals
        else:
            pending_values = self.query_points(
                pending_points, max_dist=effective_max_dist)
        values = (
            np.concatenate([pending_values, precomputed_coarse_values])
            if precomputed_coarse_values is not None else pending_values)

        # Orient the sampled triangle normals with the signed-distance field.
        # The band is stored as one complete surface cloud per offset, so the
        # SDF slope along each cloud's common triangle normal tells us whether
        # that normal points toward increasing (outside) distance.  Coarse
        # far-field samples keep a zero normal and are ignored by normal loss.
        target_normals = np.zeros((len(points), 3), dtype=np.float32)
        if len(offsets) > 1 and int(surface_samples) > 0:
            band_values = values[:band_count].reshape(
                len(offsets), int(surface_samples))
            offset_world = offsets.astype(np.float64)
            offset_centered = offset_world - float(np.mean(offset_world))
            denom = float(np.dot(offset_centered, offset_centered))
            value_centered = (
                band_values.astype(np.float64)
                - np.mean(band_values, axis=0, dtype=np.float64)[None, :]
            )
            slope = (
                np.sum(offset_centered[:, None] * value_centered, axis=0)
                / max(denom, 1.0e-12)
            )
            orientation = np.where(
                slope >= 0.0, 1.0, -1.0).astype(np.float32)
            oriented = (nrm * orientation[:, None]).astype(
                np.float32, copy=False)
            target_normals[:band_count] = np.concatenate(
                [oriented for _ in offsets], axis=0)
        if progress_cb is not None:
            progress_cb(0.96, "Sampling sparse feature thickness ...")

        thickness_field = None
        thickness_origin = None
        thickness_spacing: float | np.ndarray | None = None
        if base_transported_thickness is not None:
            # ``band_points`` is laid out as one complete surface cloud per
            # offset, so tiling preserves the triangle-point correspondence.
            # With adaptive blowup, the dense transported result is already a
            # full nearest-surface carrier.  Sample it for the coarse lattice as
            # well so every sparse target receives the same relative offset.
            # Without a carrier, coarse far-field points intentionally remain
            # zero: they are only a broad occupancy constraint, not a surface
            # feature sample.
            coarse_thickness = np.zeros(
                len(coarse_points), dtype=np.float32)
            carried_source = (
                None if thickness_result is None
                else getattr(thickness_result, "blowup_thickness", None)
            )
            if carried_source is not None:
                carried_source = np.asarray(
                    carried_source, dtype=np.float32)
                if carried_source.shape != np.asarray(
                        thickness_result.grid).shape:
                    raise ValueError(
                        "thickness_result grid/thickness shape mismatch")
                if (not np.isfinite(carried_source).all()
                        or np.any(carried_source < 0.0)):
                    raise ValueError(
                        "thickness_result thickness must be finite and "
                        "nonnegative")
                coarse_thickness = _sample_voxel_field_trilinear(
                    carried_source,
                    np.asarray(thickness_result.origin, dtype=np.float32),
                    float(thickness_result.dx),
                    coarse_points,
                )
            thickness = np.concatenate([
                np.tile(base_transported_thickness, len(band_points)),
                coarse_thickness,
            ]).astype(np.float32, copy=False)
        elif thickness_result is not None and thickness_result.thickness is not None:
            carried_source = getattr(
                thickness_result, "blowup_thickness", None)
            source = np.asarray(
                carried_source
                if carried_source is not None
                else thickness_result.thickness,
                dtype=np.float32,
            )
            if source.shape != np.asarray(thickness_result.grid).shape:
                raise ValueError("thickness_result grid/thickness shape mismatch")
            # Prefer the normal-projected carrier used by dense fitting and the
            # live preview.  Older results without that cache retain the former
            # two-voxel compatibility extension.
            thickness_field = (
                source if carried_source is not None
                else dilate_zeros(source, iters=2)
            )
            thickness_origin = np.asarray(thickness_result.origin, dtype=np.float32)
            thickness_spacing = float(thickness_result.dx)
        else:
            # Sparse-only callers already queried a regular coarse SDF lattice.
            # Reuse it as a geometrically meaningful, bounded-cost thickness
            # source instead of silently disabling thin-feature weighting.
            if sparse_only_thickness_field is None:
                coarse_shape = (
                    int(coarse_counts[2]),
                    int(coarse_counts[1]),
                    int(coarse_counts[0]),
                )
                coarse_grid = values[band_count:].reshape(coarse_shape)
                sparse_only_thickness_spacing = (
                    (aabb_max.astype(np.float64) - aabb_min.astype(np.float64))
                    / coarse_counts.astype(np.float64)
                )
                sparse_only_thickness_field = dilate_zeros(
                    local_thickness(
                        coarse_grid,
                        float(np.max(sparse_only_thickness_spacing)),
                    ),
                    iters=2,
                )
            thickness_field = sparse_only_thickness_field
            thickness_origin = aabb_min
            thickness_spacing = sparse_only_thickness_spacing

        if base_transported_thickness is None:
            thickness = _sample_voxel_field_trilinear(
                thickness_field,
                thickness_origin,
                thickness_spacing,
                points,
            )
            # All offset samples in one band originate from the same triangle point.
            # If an exterior offset lies beyond the two-voxel dilated grid field,
            # carry that surface point's feature scale along its sampled normal.
            # This is a property of the sampling construction, not nearest-neighbour
            # guessing, and keeps inside/outside residuals weighted symmetrically.
            base_thickness = _sample_voxel_field_trilinear(
                thickness_field,
                thickness_origin,
                thickness_spacing,
                base,
            )
            surface_samples_i = int(surface_samples)
            for band_i in range(len(band_points)):
                start = band_i * surface_samples_i
                end = start + surface_samples_i
                band_thickness = thickness[start:end]
                missing = band_thickness <= 0.0
                if np.any(missing):
                    band_thickness[missing] = base_thickness[missing]
        thickness = np.nan_to_num(
            thickness, nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32, copy=False)
        if progress_cb is not None:
            progress_cb(1.0, "Sparse SDF samples done")
        return SdfSampleSet(
            points,
            values,
            thickness,
            dx=float(dx),
            source="mesh-sparse",
            coarse_mask=coarse_mask,
            normals=target_normals,
            thickness_sampling_power=applied_sampling_power,
        )

    # ── voxel grid ────────────────────────────────────────────────────────

    def _launch_grid(self, aabb_min: np.ndarray, dx: float,
                     shape: int | tuple, max_dist: float) -> np.ndarray:
        """Launch the single-box SDF kernel and return the (nz, ny, nx) grid.

        ``shape`` is either a scalar ``n`` (cubic ``n³``) or an explicit
        ``(nx, ny, nz)`` tuple for an anisotropic grid.
        """
        origin = wp.vec3(float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2]))
        if isinstance(shape, tuple):
            nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
        else:
            nx = ny = nz = int(shape)
        total = nx * ny * nz
        out = wp.empty(total, dtype=wp.float32, device=self.device)
        self._reuse_stats["grid_buffer_allocations"] += 1
        wp.launch(
            kernel=_sdf_voxel_grid_kernel,
            dim=total,
            inputs=[self._warp_mesh.id, origin, float(dx), nx, ny, nz,
                    float(max_dist), self._winding_flag, 0, out],
            device=self.device,
        )
        self._reuse_stats["grid_kernel_launches"] += 1
        host = out.numpy()
        self._reuse_stats["grid_host_readbacks"] += 1
        return host.reshape((nz, ny, nx)).astype(np.float32, copy=False)

    def _launch_grid_with_vertex_thickness(
        self,
        aabb_min: np.ndarray,
        dx: float,
        shape: int | tuple,
        max_dist: float,
        vertex_thickness_wp,
        *,
        face_corner_mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Launch the fused SDF/transport kernel and read both full grids.

        ``face_corner_mode`` is private plumbing shared with the face-owned
        representation.  The public vertex helper keeps its original signature
        and behavior for existing callers.
        """
        origin = wp.vec3(
            float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2]))
        if isinstance(shape, tuple):
            nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
        else:
            nx = ny = nz = int(shape)
        total = nx * ny * nz
        out_sdf = wp.empty(total, dtype=wp.float32, device=self.device)
        out_thickness = wp.empty(
            total, dtype=wp.float32, device=self.device)
        self._reuse_stats["grid_buffer_allocations"] += 2
        wp.launch(
            kernel=_sdf_surface_thickness_voxel_grid_kernel,
            dim=total,
            inputs=[
                self._warp_mesh.id,
                self._indices_wp,
                vertex_thickness_wp,
                int(bool(face_corner_mode)),
                origin,
                float(dx),
                nx,
                ny,
                nz,
                float(max_dist),
                self._winding_flag,
                0,
                out_sdf,
                out_thickness,
            ],
            device=self.device,
        )
        self._reuse_stats["grid_kernel_launches"] += 1
        sdf_host = out_sdf.numpy()
        thickness_host = out_thickness.numpy()
        self._reuse_stats["grid_host_readbacks"] += 2
        grid_shape = (nz, ny, nx)
        return (
            sdf_host.reshape(grid_shape).astype(np.float32, copy=False),
            thickness_host.reshape(grid_shape).astype(np.float32, copy=False),
        )

    def _launch_grid_with_face_corner_thickness(
        self,
        aabb_min: np.ndarray,
        dx: float,
        shape: int | tuple,
        max_dist: float,
        face_corner_thickness_wp,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fused full-grid query for three independent values per face."""
        return self._launch_grid_with_vertex_thickness(
            aabb_min,
            dx,
            shape,
            max_dist,
            face_corner_thickness_wp,
            face_corner_mode=True,
        )

    def _launch_grid_chunked(self, aabb_min: np.ndarray, dx: float,
                             shape: tuple, max_dist: float,
                             progress_cb, p0: float, p1: float) -> np.ndarray:
        """Like :meth:`_launch_grid` but computed in z-slabs, reporting progress.

        Each slab is an independent box launch (origin shifted along z), but all
        launches write into one device buffer.  The complete grid is read back
        once after the final launch instead of once per slab.  With a progress
        callback, groups of four launches are synchronised before reporting;
        this keeps cancellation responsive without starving the device between
        every small slab.  No extra streams or overlapping mesh mutation are
        involved.
        """
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        total = nx * ny * nz
        out = wp.empty(total, dtype=wp.float32, device=self.device)
        self._reuse_stats["grid_buffer_allocations"] += 1
        # ~20 updates over the grid, at least 1 layer per slab.
        layers = max(1, nz // 20)
        z = 0
        submitted_since_sync = 0
        while z < nz:
            cz = min(layers, nz - z)
            slab_min = wp.vec3(
                float(aabb_min[0]),
                float(aabb_min[1]),
                float(aabb_min[2] + z * dx),
            )
            slab_total = nx * ny * cz
            wp.launch(
                kernel=_sdf_voxel_grid_kernel,
                dim=slab_total,
                inputs=[self._warp_mesh.id, slab_min, float(dx), nx, ny, cz,
                        float(max_dist), self._winding_flag, z * nx * ny, out],
                device=self.device,
            )
            self._reuse_stats["grid_kernel_launches"] += 1
            submitted_since_sync += 1
            z += cz
            if (progress_cb is not None
                    and (submitted_since_sync >= 4 or z == nz)):
                wp.synchronize_device(self.device)
                self._reuse_stats["grid_progress_syncs"] += 1
                progress_cb(p0 + (p1 - p0) * (z / nz), f"SDF grid {z}/{nz} layers")
                submitted_since_sync = 0
        host = out.numpy()
        self._reuse_stats["grid_host_readbacks"] += 1
        return host.reshape((nz, ny, nx)).astype(np.float32, copy=False)

    def _launch_grid_chunked_with_vertex_thickness(
        self,
        aabb_min: np.ndarray,
        dx: float,
        shape: tuple,
        max_dist: float,
        vertex_thickness_wp,
        progress_cb,
        p0: float,
        p1: float,
        *,
        face_corner_mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chunked fused SDF/feature transport with one pair of buffers.

        As in :meth:`_launch_grid_chunked`, launches are grouped between
        progress synchronisations.  Both outputs use the same ``out_offset`` so
        every z-slab remains aligned without a second mesh query.
        """
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        total = nx * ny * nz
        out_sdf = wp.empty(total, dtype=wp.float32, device=self.device)
        out_thickness = wp.empty(
            total, dtype=wp.float32, device=self.device)
        self._reuse_stats["grid_buffer_allocations"] += 2
        layers = max(1, nz // 20)
        z = 0
        submitted_since_sync = 0
        while z < nz:
            cz = min(layers, nz - z)
            slab_min = wp.vec3(
                float(aabb_min[0]),
                float(aabb_min[1]),
                float(aabb_min[2] + z * dx),
            )
            slab_total = nx * ny * cz
            wp.launch(
                kernel=_sdf_surface_thickness_voxel_grid_kernel,
                dim=slab_total,
                inputs=[
                    self._warp_mesh.id,
                    self._indices_wp,
                    vertex_thickness_wp,
                    int(bool(face_corner_mode)),
                    slab_min,
                    float(dx),
                    nx,
                    ny,
                    cz,
                    float(max_dist),
                    self._winding_flag,
                    z * nx * ny,
                    out_sdf,
                    out_thickness,
                ],
                device=self.device,
            )
            self._reuse_stats["grid_kernel_launches"] += 1
            submitted_since_sync += 1
            z += cz
            if (progress_cb is not None
                    and (submitted_since_sync >= 4 or z == nz)):
                wp.synchronize_device(self.device)
                self._reuse_stats["grid_progress_syncs"] += 1
                progress_cb(
                    p0 + (p1 - p0) * (z / nz),
                    f"SDF + thickness grid {z}/{nz} layers",
                )
                submitted_since_sync = 0
        sdf_host = out_sdf.numpy()
        thickness_host = out_thickness.numpy()
        self._reuse_stats["grid_host_readbacks"] += 2
        grid_shape = (nz, ny, nx)
        return (
            sdf_host.reshape(grid_shape).astype(np.float32, copy=False),
            thickness_host.reshape(grid_shape).astype(np.float32, copy=False),
        )

    def _launch_grid_chunked_with_face_corner_thickness(
        self,
        aabb_min: np.ndarray,
        dx: float,
        shape: tuple,
        max_dist: float,
        face_corner_thickness_wp,
        progress_cb,
        p0: float,
        p1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chunked fused query for three independent values per face."""
        return self._launch_grid_chunked_with_vertex_thickness(
            aabb_min,
            dx,
            shape,
            max_dist,
            face_corner_thickness_wp,
            progress_cb,
            p0,
            p1,
            face_corner_mode=True,
        )

    def _coarse_probe(self, aabb_min: np.ndarray, n: int,
                      max_extent: float) -> tuple[np.ndarray, float]:
        """Compute a cheap coarse (≤64³) SDF grid; returns ``(grid, coarse_dx)``.

        Reused both to size the BVH search cap and (optionally) to detect mirror
        symmetry, so the coarse pass runs at most once per ``compute_voxel_grid``.
        """
        coarse_n = int(min(64, n))
        coarse_dx = max_extent / float(coarse_n)
        coarse = self._launch_grid(aabb_min, coarse_dx, coarse_n, 1.0e6)
        return coarse, coarse_dx

    @staticmethod
    def _cap_from_coarse(coarse: np.ndarray, coarse_dx: float, dx: float) -> float:
        """Safe BVH search cap from a coarse interior-depth probe.

        ``mesh_query_point`` finds the nearest surface regardless of in/out, so
        the cap must exceed the deepest interior depth (else thick interiors miss
        and read as far-outside).  Add 50% plus a few voxels of headroom (the
        coarse grid slightly under-samples the deepest point); floored so very
        thin meshes still cover the sampling band.
        """
        depth = float(-coarse.min())          # deepest interior (0 if none)
        cap = depth * 1.5 + 4.0 * float(coarse_dx)
        return max(cap, 16.0 * float(dx))

    @staticmethod
    def _detect_mirror_axis(coarse: np.ndarray, coarse_dx: float,
                            rel_thresh: float = 0.15) -> int | None:
        """Detect a mirror plane through the box centre from the coarse grid.

        The box is centred on the mesh, so a symmetric mesh mirrors about the
        box centre; for each numpy axis we compare the grid with its flip over a
        near-surface band and pick the best axis if its relative mismatch is low
        enough.  Returns the numpy axis (0=z, 1=y, 2=x) or ``None`` if the mesh
        is not convincingly symmetric.
        """
        band = np.abs(coarse) < (3.0 * float(coarse_dx))
        if not band.any():
            return None
        scale = max(float(np.abs(coarse[band]).mean()), 1e-6)
        denom = max(int(band.sum()), 1)
        errs = {}
        for ax in (0, 1, 2):
            flipped = np.flip(coarse, axis=ax)
            errs[ax] = float(np.abs(coarse[band] - flipped[band]).sum() / denom)
        best = min(errs, key=errs.get)
        return best if (errs[best] / scale) <= rel_thresh else None

    def _launch_grid_half(self, aabb_min: np.ndarray, dx: float, shape: tuple,
                          max_dist: float, mirror_ax: int,
                          progress_cb=None, p0: float = 0.1,
                          p1: float = 0.9) -> np.ndarray:
        """Compute only one half of the grid along ``mirror_ax`` and mirror it.

        For a mesh symmetric about the box centre, voxel ``i`` and ``nA-1-i``
        along the mirror axis hold the same SDF value, so we evaluate the lower
        half (≈½ the BVH queries) at full resolution and reflect it to fill the
        rest.  ``mirror_ax`` is a numpy axis (0=z, 1=y, 2=x); ``shape`` is the
        full ``(nx, ny, nz)`` count.  Returns the full ``(nz, ny, nx)`` grid.
        """
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        n_along = (nz, ny, nx)[mirror_ax]          # full count on the mirror axis
        h = (n_along + 1) // 2                      # lower half incl. centre slab
        # Reduce the count on the mirror axis in the (nx, ny, nz) tuple.
        tuple_idx = {0: 2, 1: 1, 2: 0}[mirror_ax]
        half_counts = [nx, ny, nz]
        half_counts[tuple_idx] = h
        half_shape = (half_counts[0], half_counts[1], half_counts[2])

        if progress_cb is not None:
            grid_low = self._launch_grid_chunked(
                aabb_min, dx, half_shape, max_dist, progress_cb, p0=p0, p1=p1)
        else:
            grid_low = self._launch_grid(aabb_min, dx, half_shape, max_dist)

        flipped = np.flip(grid_low, axis=mirror_ax)
        if n_along % 2 == 0:
            upper = flipped
        else:
            # Drop the shared centre slab (first layer of the flip).
            sl = [slice(None)] * 3
            sl[mirror_ax] = slice(1, None)
            upper = flipped[tuple(sl)]
        return np.concatenate([grid_low, upper], axis=mirror_ax)

    def _launch_grid_half_with_vertex_thickness(
        self,
        aabb_min: np.ndarray,
        dx: float,
        shape: tuple,
        max_dist: float,
        mirror_ax: int,
        vertex_thickness_wp,
        progress_cb=None,
        p0: float = 0.1,
        p1: float = 0.9,
        *,
        face_corner_mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fused half-grid query, reflecting SDF and feature identically."""
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        n_along = (nz, ny, nx)[mirror_ax]
        h = (n_along + 1) // 2
        tuple_idx = {0: 2, 1: 1, 2: 0}[mirror_ax]
        half_counts = [nx, ny, nz]
        half_counts[tuple_idx] = h
        half_shape = (half_counts[0], half_counts[1], half_counts[2])

        if progress_cb is not None:
            grid_low, thickness_low = (
                self._launch_grid_chunked_with_vertex_thickness(
                    aabb_min,
                    dx,
                    half_shape,
                    max_dist,
                    vertex_thickness_wp,
                    progress_cb,
                    p0,
                    p1,
                    face_corner_mode=face_corner_mode,
                )
            )
        else:
            grid_low, thickness_low = (
                self._launch_grid_with_vertex_thickness(
                    aabb_min,
                    dx,
                    half_shape,
                    max_dist,
                    vertex_thickness_wp,
                    face_corner_mode=face_corner_mode,
                )
            )

        def _reflect(low: np.ndarray) -> np.ndarray:
            flipped = np.flip(low, axis=mirror_ax)
            if n_along % 2 == 0:
                upper = flipped
            else:
                sl = [slice(None)] * 3
                sl[mirror_ax] = slice(1, None)
                upper = flipped[tuple(sl)]
            return np.concatenate([low, upper], axis=mirror_ax)

        return _reflect(grid_low), _reflect(thickness_low)

    def _launch_grid_half_with_face_corner_thickness(
        self,
        aabb_min: np.ndarray,
        dx: float,
        shape: tuple,
        max_dist: float,
        mirror_ax: int,
        face_corner_thickness_wp,
        progress_cb=None,
        p0: float = 0.1,
        p1: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fused symmetric half-grid query for face-owned values."""
        return self._launch_grid_half_with_vertex_thickness(
            aabb_min,
            dx,
            shape,
            max_dist,
            mirror_ax,
            face_corner_thickness_wp,
            progress_cb,
            p0,
            p1,
            face_corner_mode=True,
        )

    def compute_voxel_grid(self, n: int, margin: float = 0.5,
                           compute_thickness: bool = True,
                           thickness_max_resolution: int | None = 128,
                           max_dist: float | None = None,
                           progress_cb=None,
                           symmetry: bool = False,
                           compute_blowup_thickness: bool = False,
                           blowup_thickness_fraction: float | None = None,
                           guard_voxels_per_side: int = 0,
                           vertex_thickness: np.ndarray | None = None,
                           face_corner_thickness: np.ndarray | None = None,
                           ) -> SdfResult:
        """
        Compute an axis-aligned voxel grid SDF from the mesh AABB.

        Args:
            n: total voxel count along the longest axis.  Shorter axes use
                fewer voxels at the same spacing.
            margin: fractional margin added to the bounding box extent (0.0–1.0).
            compute_thickness: also compute the local feature-thickness field
                (used by the relative under-representation metric).  This flag
                controls the expensive voxel analysis only when neither
                transported thickness representation is present.
            vertex_thickness: optional feature thickness attached to every mesh
                vertex.  It takes precedence over ``compute_thickness`` and is
                barycentrically transported by the same nearest-triangle query
                that computes the SDF.  Thus the returned result contains a
                full thickness grid even when ``compute_thickness=False``.
            face_corner_thickness: optional three feature-thickness values per
                mesh face.  Unlike shared vertex values, these remain separate
                across hard edges.  It is mutually exclusive with
                ``vertex_thickness`` and uses the same fused nearest-triangle
                query.
            compute_blowup_thickness: carry that thickness through the exterior
                offset band.  Disable while blowup is zero to avoid retaining a
                second large volume; it can be built lazily from ``thickness``.
            blowup_thickness_fraction: largest absolute local-thickness
                fraction the carried exterior field must support.  ``None``
                reserves the full UI range.
            guard_voxels_per_side: explicit interpolation/optimizer safety
                cells outside the fractionally padded core on every axis.
            thickness_max_resolution: if set, compute the expensive thickness
                field on a downsampled grid whose longest axis is at most this
                value, then upsample to the SDF grid shape.  ``0``/``None`` keeps
                full-resolution thickness.
            max_dist: BVH search cap (world units).  ``None`` → auto-size from a
                coarse interior-depth probe (prunes the empty exterior for a big
                speedup).  Pass ``float('inf')`` to disable the cap.
            progress_cb: optional ``callable(frac: float, msg: str)`` invoked with
                ``frac`` in ``[0, 1]``.  When given, the main grid is computed in
                z-slabs so a worker thread can report fine-grained progress.
            symmetry: if the mesh is detected (on the coarse probe) to be mirror-
                symmetric about the box centre, evaluate only one half at full
                resolution and reflect it — ~halving the BVH query cost.  The
                returned grid is still full-size.

        Returns:
            SdfResult with the 3-D grid and metadata.
        """
        self._check_ready()
        if vertex_thickness is not None and face_corner_thickness is not None:
            raise ValueError(
                "vertex_thickness and face_corner_thickness are mutually "
                "exclusive")
        transported_values = None
        transported_values_wp = None
        transported_face_corner_mode = face_corner_thickness is not None
        transported_source = (
            face_corner_thickness
            if transported_face_corner_mode else vertex_thickness)
        if transported_source is not None:
            transported_values = np.asarray(
                transported_source, dtype=np.float32).reshape(-1)
            if transported_face_corner_mode:
                expected = int(self._faces.shape[0]) * 3
                if transported_values.size != expected:
                    raise ValueError(
                        "face_corner_thickness must contain exactly three "
                        "values per mesh face")
                field_name = "face_corner_thickness"
            else:
                if transported_values.size != int(self._verts.shape[0]):
                    raise ValueError(
                        "vertex_thickness must contain exactly one value per "
                        "mesh vertex")
                field_name = "vertex_thickness"
            if not np.isfinite(transported_values).all():
                raise ValueError(f"{field_name} must be finite")
            if np.any(transported_values < 0.0):
                raise ValueError(f"{field_name} must be nonnegative")
            transported_values = np.ascontiguousarray(
                transported_values, dtype=np.float32)
            transported_values_wp = wp.array(
                transported_values, dtype=wp.float32, device=self.device)
        if progress_cb is not None:
            progress_cb(0.0, "Preparing SDF grid …")

        vmin = self._verts.min(axis=0).astype(np.float32)
        vmax = self._verts.max(axis=0).astype(np.float32)

        extent = vmax - vmin
        max_extent = float(extent.max())
        if max_extent <= 0.0:
            raise ValueError("Degenerate AABB (extent <= 0).")

        n = int(n)
        guard = int(guard_voxels_per_side)
        if guard < 0 or float(guard) != float(guard_voxels_per_side):
            raise ValueError(
                "guard_voxels_per_side must be a non-negative integer")
        if n <= 2 * guard:
            raise ValueError(
                "resolution must exceed twice guard_voxels_per_side")
        margin = float(margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("margin must be finite and non-negative")

        # ``n`` remains the total longest-axis grid count.  Explicit guard cells
        # live outside the fractionally padded core on *every* axis, so a narrow
        # arm/plate receives the same true four-voxel safety band as a long axis.
        # Encoding that guard in ``margin`` used to shrink it with aspect ratio.
        core_n = n - 2 * guard
        padded_max = max_extent * (1.0 + float(margin))
        dx = padded_max / float(core_n)

        padded = (extent * (1.0 + float(margin))).astype(np.float64)
        core_counts = np.maximum(
            1, np.ceil(padded / dx - 1.0e-10).astype(np.int64))
        # Avoid a floating-point ceil turning the longest core axis into n+1.
        core_counts[np.isclose(
            extent, max_extent, rtol=1.0e-7, atol=0.0)] = core_n
        counts = core_counts + 2 * guard
        nx, ny, nz = int(counts[0]), int(counts[1]), int(counts[2])

        center = 0.5 * (vmin + vmax)
        half = 0.5 * counts.astype(np.float64) * dx       # per-axis half-extent
        aabb_min = (center - half).astype(np.float32)
        aabb_max = (center + half).astype(np.float32)

        # One coarse probe serves both the BVH cap and symmetry detection.
        coarse = coarse_dx = None
        if max_dist is None or symmetry:
            if progress_cb is not None:
                progress_cb(0.05, "Probing interior depth …")
            # The cheap probe is cubic; centre that cube independently instead
            # of starting it at the anisotropic grid's short-axis minimum.
            # The previous off-centre probe could reject genuine symmetry.
            full_longest_span = float(np.max(counts)) * float(dx)
            coarse_origin = (
                center - 0.5 * full_longest_span).astype(np.float32)
            coarse, coarse_dx = self._coarse_probe(
                coarse_origin, int(np.max(counts)), full_longest_span)
        if max_dist is None:
            max_dist = self._cap_from_coarse(coarse, coarse_dx, dx)

        mirror_ax = (self._detect_mirror_axis(coarse, coarse_dx)
                     if (symmetry and coarse is not None) else None)
        computes_local_thickness = bool(
            compute_thickness and transported_values is None)
        grid_p1 = 0.78 if computes_local_thickness else 0.9

        if mirror_ax is not None:
            if progress_cb is not None:
                label = (
                    "SDF + thickness grid" if transported_values is not None
                    else "SDF grid"
                )
                progress_cb(
                    0.08,
                    f"{label} (symmetric ½, axis {'zyx'[mirror_ax]}) …",
                )
            if transported_values is not None:
                launch_half = (
                    self._launch_grid_half_with_face_corner_thickness
                    if transported_face_corner_mode
                    else self._launch_grid_half_with_vertex_thickness
                )
                grid, transported_grid = (
                    launch_half(
                        aabb_min,
                        float(dx),
                        (nx, ny, nz),
                        float(max_dist),
                        mirror_ax,
                        transported_values_wp,
                        progress_cb,
                        p0=0.1,
                        p1=grid_p1,
                    )
                )
            else:
                grid = self._launch_grid_half(
                    aabb_min, float(dx), (nx, ny, nz), float(max_dist),
                    mirror_ax, progress_cb, p0=0.1, p1=grid_p1)
        elif progress_cb is not None:
            if transported_values is not None:
                launch_chunked = (
                    self._launch_grid_chunked_with_face_corner_thickness
                    if transported_face_corner_mode
                    else self._launch_grid_chunked_with_vertex_thickness
                )
                grid, transported_grid = (
                    launch_chunked(
                        aabb_min,
                        float(dx),
                        (nx, ny, nz),
                        float(max_dist),
                        transported_values_wp,
                        progress_cb,
                        p0=0.1,
                        p1=grid_p1,
                    )
                )
            else:
                grid = self._launch_grid_chunked(
                    aabb_min, float(dx), (nx, ny, nz), float(max_dist),
                    progress_cb, p0=0.1, p1=grid_p1)
        else:
            if transported_values is not None:
                launch_full = (
                    self._launch_grid_with_face_corner_thickness
                    if transported_face_corner_mode
                    else self._launch_grid_with_vertex_thickness
                )
                grid, transported_grid = (
                    launch_full(
                        aabb_min,
                        float(dx),
                        (nx, ny, nz),
                        float(max_dist),
                        transported_values_wp,
                    )
                )
            else:
                grid = self._launch_grid(
                    aabb_min, float(dx), (nx, ny, nz), float(max_dist))

        thickness = None
        blowup_thickness = None
        thickness_stride_vox = 1.0
        blowup_thickness_extent_vox = 0.0
        blowup_thickness_capacity_fraction = None
        if transported_values is not None:
            # Preserve the legacy/raw-field contract: feature thickness is an
            # interior quantity.  ``transported_grid`` additionally contains
            # the nearest surface feature outside, but exposing that as raw
            # thickness would make dense far-field loss weighting non-zero.
            # Keep a separate, bounded exterior carrier only when blowup needs
            # it.  Both operations are linear masks; neither performs another
            # mesh/BVH query.
            if compute_blowup_thickness:
                thickness = np.zeros_like(transported_grid)
                np.copyto(
                    thickness,
                    transported_grid,
                    where=np.asarray(grid) < 0.0,
                )
                carrier_fraction = (
                    MAX_UI_THICKNESS_FRACTION
                    if blowup_thickness_fraction is None
                    else float(blowup_thickness_fraction)
                )
                blowup_thickness_extent_vox = (
                    relative_blowup_extent_voxels(
                        carrier_fraction, thickness, float(dx))
                    + BLOWUP_CARRIER_MARGIN_VOXELS
                )
                blowup_thickness = transported_grid
                blowup_thickness[
                    np.asarray(grid)
                    > blowup_thickness_extent_vox * float(dx)
                ] = 0.0
                blowup_thickness_capacity_fraction = abs(carrier_fraction)
            else:
                transported_grid[np.asarray(grid) >= 0.0] = 0.0
                thickness = transported_grid
        elif compute_thickness:
            if progress_cb is not None:
                progress_cb(grid_p1, "Computing thickness field …")

                def _thick_progress(frac, msg):
                    progress_cb(
                        grid_p1 + (0.98 - grid_p1) * float(frac),
                        str(msg),
                    )
            else:
                _thick_progress = None
            if (thickness_max_resolution is not None
                    and int(thickness_max_resolution) > 0):
                thickness_stride_vox = float(max(
                    1,
                    int(np.ceil(
                        max(grid.shape)
                        / float(int(thickness_max_resolution)))),
                ))
            thickness = local_thickness(
                grid, float(dx),
                max_resolution=thickness_max_resolution,
                progress_cb=_thick_progress)
            if mirror_ax is not None:
                # Strided low-resolution thickness sampling starts at index 0
                # and can introduce a one-sided phase bias.  Mirror the resolved
                # partner across one-sided holes and otherwise keep the smaller
                # measured pair value, so the cap is exact and conservative.
                thickness = conservative_mirror_min(
                    thickness, axis=mirror_ax)
            if compute_blowup_thickness:
                if progress_cb is not None:
                    progress_cb(
                        0.985, "Preparing adaptive SDF blowup field …")
                carrier_fraction = (
                    MAX_UI_THICKNESS_FRACTION
                    if blowup_thickness_fraction is None
                    else float(blowup_thickness_fraction)
                )
                blowup_thickness_extent_vox = (
                    relative_blowup_extent_voxels(
                        carrier_fraction, thickness, float(dx))
                    + BLOWUP_CARRIER_MARGIN_VOXELS
                )
                blowup_thickness = build_surface_carried_thickness(
                    grid,
                    thickness,
                    float(dx),
                    max_exterior_vox=blowup_thickness_extent_vox,
                    thickness_stride_vox=thickness_stride_vox,
                    device=self.device,
                )
                blowup_thickness_capacity_fraction = abs(carrier_fraction)
        if progress_cb is not None:
            progress_cb(1.0, "SDF done")

        return SdfResult(
            grid=grid,
            n=int(max(nx, ny, nz)),
            dx=float(dx),
            origin=aabb_min.astype(np.float32),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            thickness=thickness,
            blowup_thickness=blowup_thickness,
            thickness_stride_vox=thickness_stride_vox,
            blowup_thickness_extent_vox=blowup_thickness_extent_vox,
            blowup_thickness_capacity_fraction=(
                blowup_thickness_capacity_fraction),
            nx=nx, ny=ny, nz=nz,
        )

    @staticmethod
    def _isotropic_box(aabb_min, aabb_max, n: int):
        """Centre-expand a box to its longest extent → (box_min, box_max, dx)."""
        bmin = np.asarray(aabb_min, dtype=np.float32)
        bmax = np.asarray(aabb_max, dtype=np.float32)
        center = 0.5 * (bmin + bmax)
        max_extent = float((bmax - bmin).max())
        if max_extent <= 0.0:
            raise ValueError("Degenerate box (extent <= 0).")
        half = 0.5 * max_extent
        box_min = (center - half).astype(np.float32)
        box_max = (center + half).astype(np.float32)
        return box_min, box_max, max_extent / float(n), max_extent

    def compute_box_grid(self, aabb_min, aabb_max, n: int = 128,
                         compute_thickness: bool = True,
                         max_dist: float | None = None) -> SdfResult:
        """Compute a high-resolution SDF over an arbitrary axis-aligned box.

        Unlike :meth:`compute_voxel_grid` (which derives its box from the whole
        mesh AABB), this evaluates a fresh ``n³`` grid over the supplied box only,
        so a small region is resolved much more finely (genuinely finer voxels).

        The box is made isotropic by expanding the shortest axes up to the
        longest extent, keeping the box centred, so ``dx`` is uniform.

        Args:
            aabb_min, aabb_max: (3,) world-space box corners.
            n: voxels per axis (default 128).
            compute_thickness: also compute the local feature-thickness field.
            max_dist: BVH search cap (world units).  ``None`` → ``1.2×`` the box
                extent, which safely covers the box interior/band while pruning
                far traversal.

        Returns:
            SdfResult over the box (grid, n, dx, origin, aabb_min, aabb_max,
            thickness).
        """
        self._check_ready()

        box_min, box_max, dx, max_extent = self._isotropic_box(aabb_min, aabb_max, n)
        if max_dist is None:
            max_dist = 1.2 * max_extent

        grid = self._launch_grid(box_min, float(dx), int(n), float(max_dist))
        thickness = local_thickness(grid, float(dx)) if compute_thickness else None

        return SdfResult(
            grid=grid,
            n=n,
            dx=float(dx),
            origin=box_min,
            aabb_min=box_min,
            aabb_max=box_max,
            thickness=thickness,
        )

    def compute_box_grids_batch(self, boxes, n: int = 128,
                                compute_thickness: bool = True,
                                max_dist: float | None = None) -> list[SdfResult]:
        """Compute several region boxes in a *single* kernel launch.

        ``boxes`` is a list of ``(aabb_min, aabb_max)`` pairs.  Each is made
        isotropic and resolved at ``n³``; all are evaluated in one launch (less
        per-launch overhead and fewer host round-trips than calling
        :meth:`compute_box_grid` per region).  Returns one SdfResult per box, in
        input order.

        Args:
            max_dist: shared BVH cap (world units).  ``None`` → ``1.2×`` the
                largest box extent (safe for every box, prunes far traversal).
        """
        self._check_ready()
        n = int(n)
        if not boxes:
            return []

        per = n * n * n
        origins, dxs, metas = [], [], []
        for (bmin, bmax) in boxes:
            box_min, box_max, dx, max_extent = self._isotropic_box(bmin, bmax, n)
            origins.append(box_min)
            dxs.append(np.float32(dx))
            metas.append((box_min, box_max, float(dx), max_extent))

        if max_dist is None:
            max_dist = 1.2 * max(m[3] for m in metas)

        origins_wp = wp.array(np.stack(origins).astype(np.float32),
                              dtype=wp.vec3, device=self.device)
        dxs_wp = wp.array(np.asarray(dxs, dtype=np.float32),
                          dtype=wp.float32, device=self.device)
        total = len(boxes) * per
        out = wp.empty(total, dtype=wp.float32, device=self.device)
        wp.launch(
            kernel=_sdf_voxel_grid_batch_kernel,
            dim=total,
            inputs=[self._warp_mesh.id, origins_wp, dxs_wp, n,
                    float(max_dist), self._winding_flag, out],
            device=self.device,
        )

        flat = out.numpy()
        results = []
        for b, (box_min, box_max, dx, _) in enumerate(metas):
            grid = flat[b * per:(b + 1) * per].reshape((n, n, n)).astype(
                np.float32, copy=False)
            thickness = local_thickness(grid, dx) if compute_thickness else None
            results.append(SdfResult(
                grid=grid, n=n, dx=dx, origin=box_min,
                aabb_min=box_min, aabb_max=box_max, thickness=thickness,
            ))
        return results

    # ── internal ──────────────────────────────────────────────────────────

    def _check_ready(self):
        if not self.is_ready:
            raise RuntimeError("No mesh loaded. Call set_mesh() first.")
