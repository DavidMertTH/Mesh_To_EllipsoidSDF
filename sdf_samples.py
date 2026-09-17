"""Sparse/sample-backed SDF target data shared by optimizers.

The dense SDF grid remains useful for slices, analysis overlays and legacy
maintenance code.  Training, however, only needs batches of target samples.
This module defines that smaller common API so dense grids, narrow-band samples
and future block-sparse/octree backends can feed the same kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from sdf_blowup import (
    DEFAULT_MAX_THICKNESS_FRACTION,
    apply_thickness_relative_blowup,
    apply_thickness_limited_blowup,
)


def sdf_grid_normals(grid: np.ndarray, dx: float) -> np.ndarray:
    """Return unit SDF gradients for a ``(z, y, x)`` voxel grid.

    Degenerate axes and locally flat/invalid gradient regions receive a zero
    vector, which makes them safe to ignore in the normal-loss kernel.
    """
    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("SDF grid must have shape (nz, ny, nx)")
    spacing = float(dx)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("dx must be finite and positive")

    derivatives: list[np.ndarray] = []
    for axis, size in enumerate(values.shape):
        if int(size) <= 1:
            derivative = np.zeros_like(values)
        else:
            derivative = np.gradient(
                values,
                spacing,
                axis=axis,
                edge_order=2 if int(size) >= 3 else 1,
            )
        derivatives.append(np.asarray(derivative, dtype=np.float32))
    gz, gy, gx = derivatives
    gradients = np.stack([gx, gy, gz], axis=-1).astype(np.float32, copy=False)
    lengths = np.linalg.norm(gradients, axis=-1, keepdims=True)
    valid = np.isfinite(lengths[..., 0]) & (lengths[..., 0] > 1.0e-6)
    normals = np.zeros_like(gradients)
    normals[valid] = gradients[valid] / lengths[valid]
    return np.ascontiguousarray(normals, dtype=np.float32)


def sample_sdf_grid_normals(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    points: np.ndarray,
    *,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Sample analytic trilinear-grid normals without allocating a 3-D gradient.

    This is used by sparse fitting after a thickness-relative blowup.  The
    transformed field has spatially varying offsets, so the original triangle
    normals are no longer the exact gradient of the target being optimized.
    """
    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) < 2:
        raise ValueError("grid must have shape (nz, ny, nx), each axis >= 2")
    spacing = float(dx)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("dx must be finite and positive")
    base_origin = np.asarray(origin, dtype=np.float32).reshape(3)
    sample_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(values).all() or not np.isfinite(sample_points).all():
        raise ValueError("grid and points must be finite")

    nz, ny, nx = (int(v) for v in values.shape)
    result = np.zeros((len(sample_points), 3), dtype=np.float32)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(sample_points), chunk_size):
        stop = min(start + chunk_size, len(sample_points))
        grid_pos = (
            (sample_points[start:stop] - base_origin[None, :]) / spacing
            - 0.5
        )
        x0 = np.clip(np.floor(grid_pos[:, 0]).astype(np.int64), 0, nx - 2)
        y0 = np.clip(np.floor(grid_pos[:, 1]).astype(np.int64), 0, ny - 2)
        z0 = np.clip(np.floor(grid_pos[:, 2]).astype(np.int64), 0, nz - 2)
        fx = np.clip(grid_pos[:, 0] - x0, 0.0, 1.0).astype(np.float32)
        fy = np.clip(grid_pos[:, 1] - y0, 0.0, 1.0).astype(np.float32)
        fz = np.clip(grid_pos[:, 2] - z0, 0.0, 1.0).astype(np.float32)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        v000 = values[z0, y0, x0]
        v100 = values[z0, y0, x1]
        v010 = values[z0, y1, x0]
        v110 = values[z0, y1, x1]
        v001 = values[z1, y0, x0]
        v101 = values[z1, y0, x1]
        v011 = values[z1, y1, x0]
        v111 = values[z1, y1, x1]

        one_x, one_y, one_z = 1.0 - fx, 1.0 - fy, 1.0 - fz
        gx = (
            one_z * (one_y * (v100 - v000) + fy * (v110 - v010))
            + fz * (one_y * (v101 - v001) + fy * (v111 - v011))
        ) / spacing
        gy = (
            one_z * (one_x * (v010 - v000) + fx * (v110 - v100))
            + fz * (one_x * (v011 - v001) + fx * (v111 - v101))
        ) / spacing
        gz = (
            one_y * (one_x * (v001 - v000) + fx * (v101 - v100))
            + fy * (one_x * (v011 - v010) + fx * (v111 - v110))
        ) / spacing
        gradients = np.stack([gx, gy, gz], axis=1).astype(
            np.float32, copy=False)
        lengths = np.linalg.norm(gradients, axis=1)
        valid = np.isfinite(lengths) & (lengths > 1.0e-6)
        result[start:stop][valid] = gradients[valid] / lengths[valid, None]
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass
class SdfSampleSet:
    """World-space SDF samples used by differentiable fitting."""

    points: np.ndarray
    values: np.ndarray
    thickness: np.ndarray | None = None
    dx: float = 1.0
    source: str = "samples"
    coarse_mask: np.ndarray | None = None
    normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float32).reshape(-1, 3)
        self.values = np.ascontiguousarray(self.values, dtype=np.float32).reshape(-1)
        if self.points.shape[0] != self.values.shape[0]:
            raise ValueError("SdfSampleSet points/value count mismatch")
        if self.thickness is not None:
            self.thickness = np.ascontiguousarray(
                self.thickness, dtype=np.float32).reshape(-1)
            if self.thickness.shape[0] != self.values.shape[0]:
                raise ValueError("SdfSampleSet thickness/value count mismatch")
        if self.coarse_mask is not None:
            self.coarse_mask = np.ascontiguousarray(
                self.coarse_mask, dtype=np.bool_).reshape(-1)
            if self.coarse_mask.shape[0] != self.values.shape[0]:
                raise ValueError("SdfSampleSet coarse-mask/value count mismatch")
        if self.normals is not None:
            self.normals = np.ascontiguousarray(
                self.normals, dtype=np.float32).reshape(-1, 3)
            if self.normals.shape[0] != self.values.shape[0]:
                raise ValueError("SdfSampleSet normal/value count mismatch")

    @property
    def size(self) -> int:
        return int(self.values.shape[0])

    def with_offset(self, offset: float) -> "SdfSampleSet":
        if float(offset) == 0.0:
            return self
        return SdfSampleSet(
            points=self.points,
            values=(self.values + np.float32(offset)).astype(np.float32),
            thickness=self.thickness,
            dx=self.dx,
            source=self.source,
            coarse_mask=self.coarse_mask,
            normals=self.normals,
        )

    def with_thickness_limited_offset(
        self,
        offset: float,
        max_thickness_fraction: float = DEFAULT_MAX_THICKNESS_FRACTION,
    ) -> "SdfSampleSet":
        """Apply an adaptive offset while preserving valid sample metadata.

        A constant SDF offset leaves its spatial gradient (and therefore its
        normals) unchanged.  Once the requested offset is capped by the local
        thickness, however, the transformed field is
        ``sdf + sign(offset) * fraction * thickness``.  Its normal then also
        depends on the unknown spatial thickness gradient.  An unstructured
        sparse sample set does not contain enough neighbourhood information to
        reconstruct that derivative reliably, so capped and unresolved samples
        receive a zero normal and are ignored by the normal-loss kernel.  Samples
        with a strictly inactive cap retain their original normal.
        """
        if float(offset) == 0.0:
            return self
        adjusted_values = apply_thickness_limited_blowup(
            self.values,
            float(offset),
            self.thickness,
            float(self.dx),
            max_thickness_fraction=max_thickness_fraction,
        )
        adjusted_normals = self.normals
        if adjusted_normals is not None and self.thickness is not None:
            # Strict inequality deliberately excludes the min() transition,
            # where the adaptive offset is non-differentiable.  Unknown
            # thickness (zero) also fails this test and is masked safely.
            constant_offset = (
                float(max_thickness_fraction) * self.thickness
                > abs(float(offset))
            )
            adjusted_normals = adjusted_normals.copy()
            adjusted_normals[~constant_offset] = 0.0
        return SdfSampleSet(
            points=self.points,
            values=adjusted_values,
            thickness=self.thickness,
            dx=self.dx,
            source=self.source,
            coarse_mask=self.coarse_mask,
            normals=adjusted_normals,
        )

    def with_thickness_relative_offset(
        self,
        thickness_fraction: float,
        *,
        normals: np.ndarray | None = None,
    ) -> "SdfSampleSet":
        """Apply a local-diameter fraction while preserving sample metadata.

        If transformed-field normals are not supplied, normals are cleared: a
        spatially varying thickness offset does not preserve the raw mesh SDF
        gradient.  Dense callers can use :func:`sample_sdf_grid_normals` to
        provide the exact trilinear target normals without a huge gradient grid.
        """
        fraction = float(thickness_fraction)
        if fraction == 0.0:
            return self
        adjusted_normals = None
        if normals is not None:
            adjusted_normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
            if adjusted_normals.shape[0] != self.size:
                raise ValueError("normal/sample count mismatch")
        elif self.normals is not None:
            adjusted_normals = np.zeros_like(self.normals, dtype=np.float32)
        return SdfSampleSet(
            points=self.points,
            values=apply_thickness_relative_blowup(
                self.values,
                fraction,
                self.thickness,
            ),
            thickness=self.thickness,
            dx=self.dx,
            source=self.source,
            coarse_mask=self.coarse_mask,
            normals=adjusted_normals,
        )

    @classmethod
    def from_grid(
        cls,
        grid: np.ndarray,
        origin: np.ndarray,
        dx: float,
        thickness: np.ndarray | None = None,
        source: str = "dense-grid",
    ) -> "SdfSampleSet":
        g = np.asarray(grid, dtype=np.float32)
        nz, ny, nx = (int(s) for s in g.shape)
        iz, iy, ix = np.meshgrid(
            np.arange(nz, dtype=np.float32),
            np.arange(ny, dtype=np.float32),
            np.arange(nx, dtype=np.float32),
            indexing="ij",
        )
        o = np.asarray(origin, dtype=np.float32)
        pts = np.stack([
            o[0] + (ix.ravel() + 0.5) * float(dx),
            o[1] + (iy.ravel() + 0.5) * float(dx),
            o[2] + (iz.ravel() + 0.5) * float(dx),
        ], axis=1)
        th = None if thickness is None else np.asarray(thickness, dtype=np.float32).ravel()
        normals = sdf_grid_normals(g, float(dx)).reshape(-1, 3)
        return cls(
            pts, g.ravel(), th, float(dx), source=source,
            normals=normals)


class UploadedSdfSamples:
    """Device arrays for an ``SdfSampleSet``."""

    def __init__(
        self,
        samples: SdfSampleSet,
        device: str,
        *,
        include_normals: bool = True,
    ):
        self.samples = samples
        self.points = wp.array(samples.points, dtype=wp.vec3, device=device)
        self.values = wp.array(samples.values, dtype=wp.float32, device=device)
        thick = samples.thickness
        if thick is None:
            thick = np.zeros(samples.size, dtype=np.float32)
        self.thickness = wp.array(thick, dtype=wp.float32, device=device)
        coarse = samples.coarse_mask
        if coarse is None:
            coarse = np.zeros(samples.size, dtype=np.int32)
        else:
            coarse = np.asarray(coarse, dtype=np.int32)
        self.coarse_mask = wp.array(coarse, dtype=wp.int32, device=device)
        self.normals = None
        if include_normals:
            normals = samples.normals
            if normals is None:
                normals = np.zeros((samples.size, 3), dtype=np.float32)
            self.normals = wp.array(normals, dtype=wp.vec3, device=device)
