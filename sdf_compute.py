"""
sdf_compute.py — Warp-based SDF computation on triangle meshes.

Provides:
  - GPU/CPU kernels for single-point and voxel-grid SDF queries.
  - SdfComputer class that manages mesh upload and grid computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

from sdf_blowup import (
    build_surface_carried_thickness,
    conservative_mirror_min,
)
from thickness import dilate_zeros, local_thickness
from sdf_samples import SdfSampleSet


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
        out_sdf[tid] = 1.0e6
        return

    closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    d = wp.length(p - closest)
    s = -1.0 if q.sign < 0.0 else 1.0
    out_sdf[tid] = d * s


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
    return out


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
        self._verts: Optional[np.ndarray] = None
        self._faces: Optional[np.ndarray] = None
        # Whether the mesh is closed (every edge shared by exactly 2 faces).  A
        # non-watertight mesh has an unreliable normal-based inside/outside sign,
        # so we switch to the (slower but robust) winding-number sign for it.
        self._watertight: bool = True

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
        self._verts = verts.astype(np.float32, copy=False)
        self._faces = faces.astype(np.int32, copy=False)
        self._watertight = self._is_watertight(faces)
        points_wp = wp.array(self._verts, dtype=wp.vec3, device=self.device)
        indices_wp = wp.array(
            faces.astype(np.int32, copy=False).reshape(-1),
            dtype=wp.int32,
            device=self.device,
        )
        self._warp_mesh = wp.Mesh(points=points_wp, indices=indices_wp)

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
        self._verts = None

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
    ) -> SdfSampleSet:
        """Compute sparse training samples concentrated around mesh triangles.

        The sample cloud has high density in a narrow band around the surface
        and a low-density coarse lattice through the padded AABB.  It is intended
        for optimizer loss batches, not for UI slicing.

        When ``thickness_result`` provides a dense local-thickness field, that
        field is sampled trilinearly.  This is the preferred production path:
        it gives sparse and dense training identical feature scales without
        recomputing thickness.  Standalone sparse calls derive a lower-resolution
        thickness field from their coarse SDF lattice instead.
        """
        self._check_ready()
        if self._faces is None or self._verts is None:
            raise RuntimeError("No mesh loaded. Call set_mesh() first.")
        if progress_cb is not None:
            progress_cb(0.0, "Preparing sparse SDF samples ...")

        verts = np.asarray(self._verts, dtype=np.float32)
        faces = np.asarray(self._faces, dtype=np.int32)
        vmin = verts.min(axis=0).astype(np.float32)
        vmax = verts.max(axis=0).astype(np.float32)
        extent = vmax - vmin
        max_extent = float(extent.max())
        if max_extent <= 0.0:
            raise ValueError("Degenerate AABB (extent <= 0).")

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
        cross = cross[valid]
        area = (double_area[valid].astype(np.float64) * 0.5)
        normals = cross / np.maximum(double_area[valid, None], 1e-12)

        rng = np.random.default_rng(seed)
        if surface_samples is None:
            # O(n^2) surface density instead of O(n^3) volume density.
            surface_samples = int(np.clip(2 * int(n) * int(n), 8192, 160_000))
        area_sum = float(area.sum(dtype=np.float64))
        if not np.isfinite(area_sum) or area_sum <= 1e-12:
            raise ValueError("Mesh has no finite triangle area.")
        probs = area / area_sum
        probs = probs / float(probs.sum(dtype=np.float64))
        tri_idx = rng.choice(len(tri), size=int(surface_samples), replace=True, p=probs)
        r1 = rng.random(int(surface_samples), dtype=np.float32)
        r2 = rng.random(int(surface_samples), dtype=np.float32)
        sr1 = np.sqrt(r1).astype(np.float32, copy=False)
        b0 = 1.0 - sr1
        b1 = sr1 * (1.0 - r2)
        b2 = sr1 * r2
        base = (tri[tri_idx, 0] * b0[:, None]
                + tri[tri_idx, 1] * b1[:, None]
                + tri[tri_idx, 2] * b2[:, None]).astype(np.float32)
        nrm = normals[tri_idx].astype(np.float32)

        offsets = np.asarray(offsets_vox, dtype=np.float32) * np.float32(dx)
        band_points = [
            (base + off * nrm).astype(np.float32)
            for off in offsets
        ]

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
        coarse_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        band_count = sum(len(part) for part in band_points)
        points = np.concatenate(band_points + [coarse_points], axis=0)
        coarse_mask = np.zeros(points.shape[0], dtype=np.bool_)
        coarse_mask[band_count:] = True
        if progress_cb is not None:
            progress_cb(0.45, f"Querying {len(points):,} sparse SDF samples ...")
        if max_dist is None:
            max_dist = max(16.0 * float(dx), 1.2 * padded_max)
        if progress_cb is not None and len(points) > 65_536:
            vals = np.empty(len(points), dtype=np.float32)
            chunk = 65_536
            for start in range(0, len(points), chunk):
                end = min(len(points), start + chunk)
                vals[start:end] = self.query_points(
                    points[start:end],
                    max_dist=float(max_dist),
                )
                progress_cb(
                    0.45 + 0.50 * (end / max(1, len(points))),
                    f"Sparse query {end:,}/{len(points):,} samples",
                )
            values = vals
        else:
            values = self.query_points(points, max_dist=float(max_dist))
        if progress_cb is not None:
            progress_cb(0.96, "Sampling sparse feature thickness ...")

        thickness_field = None
        thickness_origin = None
        thickness_spacing: float | np.ndarray | None = None
        if thickness_result is not None and thickness_result.thickness is not None:
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
            coarse_shape = (
                int(coarse_counts[2]), int(coarse_counts[1]), int(coarse_counts[0]),
            )
            coarse_grid = values[band_count:].reshape(coarse_shape)
            coarse_spacing = (
                (aabb_max.astype(np.float64) - aabb_min.astype(np.float64))
                / coarse_counts.astype(np.float64)
            )
            thickness_field = dilate_zeros(
                local_thickness(coarse_grid, float(np.max(coarse_spacing))),
                iters=2,
            )
            thickness_origin = aabb_min
            thickness_spacing = coarse_spacing

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
        wp.launch(
            kernel=_sdf_voxel_grid_kernel,
            dim=total,
            inputs=[self._warp_mesh.id, origin, float(dx), nx, ny, nz,
                    float(max_dist), self._winding_flag, out],
            device=self.device,
        )
        return out.numpy().reshape((nz, ny, nx)).astype(np.float32, copy=False)

    def _launch_grid_chunked(self, aabb_min: np.ndarray, dx: float,
                             shape: tuple, max_dist: float,
                             progress_cb, p0: float, p1: float) -> np.ndarray:
        """Like :meth:`_launch_grid` but computed in z-slabs, reporting progress.

        Each slab is an independent box launch (origin shifted along z), so the
        kernel is unchanged.  ``progress_cb(frac, msg)`` is called after every
        slab with ``frac`` interpolated in ``[p0, p1]`` — letting a worker thread
        drive a progress bar without the per-voxel kernel knowing anything.
        """
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        grid = np.empty((nz, ny, nx), dtype=np.float32)
        # ~20 updates over the grid, at least 1 layer per slab.
        layers = max(1, nz // 20)
        z = 0
        while z < nz:
            cz = min(layers, nz - z)
            slab_min = np.array(
                [aabb_min[0], aabb_min[1], aabb_min[2] + z * dx], dtype=np.float32)
            grid[z:z + cz] = self._launch_grid(slab_min, dx, (nx, ny, cz), max_dist)
            z += cz
            if progress_cb is not None:
                progress_cb(p0 + (p1 - p0) * (z / nz), f"SDF grid {z}/{nz} layers")
        return grid

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

    def compute_voxel_grid(self, n: int, margin: float = 0.5,
                           compute_thickness: bool = True,
                           thickness_max_resolution: int | None = 128,
                           max_dist: float | None = None,
                           progress_cb=None,
                           symmetry: bool = False,
                           compute_blowup_thickness: bool = False) -> SdfResult:
        """
        Compute an axis-aligned voxel grid SDF from the mesh AABB.

        Args:
            n: number of voxels along each axis.
            margin: fractional margin added to the bounding box extent (0.0–1.0).
            compute_thickness: also compute the local feature-thickness field
                (used by the relative under-representation metric).
            compute_blowup_thickness: carry that thickness through the exterior
                offset band.  Disable while blowup is zero to avoid retaining a
                second large volume; it can be built lazily from ``thickness``.
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
        if progress_cb is not None:
            progress_cb(0.0, "Preparing SDF grid …")

        vmin = self._verts.min(axis=0).astype(np.float32)
        vmax = self._verts.max(axis=0).astype(np.float32)

        extent = vmax - vmin
        max_extent = float(extent.max())
        if max_extent <= 0.0:
            raise ValueError("Degenerate AABB (extent <= 0).")

        # ``n`` resolves the *longest* axis; ``dx`` follows from it.  Each axis
        # then gets just enough voxels to cover its own (margin-padded) extent at
        # that same ``dx`` — so the box hugs the mesh instead of being a cube and
        # short axes don't waste voxels on empty exterior.  Margin is applied per
        # axis relative to each axis' own extent.
        padded_max = max_extent * (1.0 + float(margin))
        dx = padded_max / float(n)

        padded = (extent * (1.0 + float(margin))).astype(np.float64)
        counts = np.maximum(1, np.ceil(padded / dx).astype(np.int64))
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
            coarse, coarse_dx = self._coarse_probe(aabb_min, int(n), padded_max)
        if max_dist is None:
            max_dist = self._cap_from_coarse(coarse, coarse_dx, dx)

        mirror_ax = (self._detect_mirror_axis(coarse, coarse_dx)
                     if (symmetry and coarse is not None) else None)
        grid_p1 = 0.78 if compute_thickness else 0.9

        if mirror_ax is not None:
            if progress_cb is not None:
                progress_cb(0.08, f"SDF grid (symmetric ½, axis {'zyx'[mirror_ax]}) …")
            grid = self._launch_grid_half(
                aabb_min, float(dx), (nx, ny, nz), float(max_dist),
                mirror_ax, progress_cb, p0=0.1, p1=grid_p1)
        elif progress_cb is not None:
            grid = self._launch_grid_chunked(
                aabb_min, float(dx), (nx, ny, nz), float(max_dist),
                progress_cb, p0=0.1, p1=grid_p1)
        else:
            grid = self._launch_grid(aabb_min, float(dx), (nx, ny, nz),
                                     float(max_dist))

        thickness = None
        blowup_thickness = None
        if compute_thickness:
            if progress_cb is not None:
                progress_cb(grid_p1, "Computing thickness field …")

                def _thick_progress(frac, msg):
                    progress_cb(
                        grid_p1 + (0.98 - grid_p1) * float(frac),
                        str(msg),
                    )
            else:
                _thick_progress = None
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
                blowup_thickness = build_surface_carried_thickness(
                    grid, thickness, float(dx))
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
