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


# ── Warp kernels ──────────────────────────────────────────────────────────────

@wp.kernel
def _sdf_one_point_kernel(
    mesh_id: wp.uint64,
    p: wp.vec3,
    out_sdf: wp.array(dtype=wp.float32),
):
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

    q = wp.mesh_query_point(mesh_id, p, 1.0e6)
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
    """Holds the output of a voxel-grid SDF computation."""
    grid: np.ndarray          # (nz, ny, nx) float32
    n: int
    dx: float
    origin: np.ndarray        # (3,) float32  – world-space corner
    aabb_min: np.ndarray      # (3,) float32
    aabb_max: np.ndarray      # (3,) float32


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

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._warp_mesh: Optional[wp.Mesh] = None
        self._verts: Optional[np.ndarray] = None

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
        points_wp = wp.array(self._verts, dtype=wp.vec3, device=self.device)
        indices_wp = wp.array(
            faces.astype(np.int32, copy=False).reshape(-1),
            dtype=wp.int32,
            device=self.device,
        )
        self._warp_mesh = wp.Mesh(points=points_wp, indices=indices_wp)

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
            inputs=[self._warp_mesh.id, p, out],
            device=self.device,
        )
        return float(out.numpy()[0])

    # ── voxel grid ────────────────────────────────────────────────────────

    def compute_voxel_grid(self, n: int) -> SdfResult:
        """
        Compute an axis-aligned voxel grid SDF from the mesh AABB.

        Args:
            n: number of voxels along each axis.

        Returns:
            SdfResult with the 3-D grid and metadata.
        """
        self._check_ready()

        vmin = self._verts.min(axis=0).astype(np.float32)
        vmax = self._verts.max(axis=0).astype(np.float32)

        extent = vmax - vmin
        max_extent = float(extent.max())
        if max_extent <= 0.0:
            raise ValueError("Degenerate AABB (extent <= 0).")

        dx = max_extent / float(n)
        center = 0.5 * (vmin + vmax)
        half = 0.5 * max_extent

        aabb_min = center - half
        aabb_max = center + half
        origin = wp.vec3(float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2]))

        nx = ny = nz = int(n)
        total = nx * ny * nz
        out = wp.empty(total, dtype=wp.float32, device=self.device)

        wp.launch(
            kernel=_sdf_voxel_grid_kernel,
            dim=total,
            inputs=[self._warp_mesh.id, origin, float(dx), nx, ny, nz, out],
            device=self.device,
        )

        grid = out.numpy().reshape((nz, ny, nx)).astype(np.float32, copy=False)

        return SdfResult(
            grid=grid,
            n=n,
            dx=float(dx),
            origin=aabb_min.astype(np.float32),
            aabb_min=aabb_min.astype(np.float32),
            aabb_max=aabb_max.astype(np.float32),
        )

    # ── internal ──────────────────────────────────────────────────────────

    def _check_ready(self):
        if not self.is_ready:
            raise RuntimeError("No mesh loaded. Call set_mesh() first.")