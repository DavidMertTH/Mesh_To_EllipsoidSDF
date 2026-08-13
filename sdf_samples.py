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
    apply_thickness_limited_blowup,
)


@dataclass
class SdfSampleSet:
    """World-space SDF samples used by differentiable fitting."""

    points: np.ndarray
    values: np.ndarray
    thickness: np.ndarray | None = None
    dx: float = 1.0
    source: str = "samples"
    coarse_mask: np.ndarray | None = None

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
        )

    def with_thickness_limited_offset(
        self,
        offset: float,
        max_thickness_fraction: float = DEFAULT_MAX_THICKNESS_FRACTION,
    ) -> "SdfSampleSet":
        """Apply an adaptive offset while preserving sparse sample metadata."""
        if float(offset) == 0.0:
            return self
        return SdfSampleSet(
            points=self.points,
            values=apply_thickness_limited_blowup(
                self.values,
                float(offset),
                self.thickness,
                float(self.dx),
                max_thickness_fraction=max_thickness_fraction,
            ),
            thickness=self.thickness,
            dx=self.dx,
            source=self.source,
            coarse_mask=self.coarse_mask,
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
        return cls(pts, g.ravel(), th, float(dx), source=source)


class UploadedSdfSamples:
    """Device arrays for an ``SdfSampleSet``."""

    def __init__(self, samples: SdfSampleSet, device: str):
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
