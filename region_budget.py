"""Stable, size-aware surface regions for local primitive population caps.

The optimiser works in volume space, while the representation shown to the
user lives on the mesh surface.  This module builds both from one deterministic
partition: faces are grouped around well-spaced surface seeds and the global
primitive budget is apportioned by region area.
"""

from __future__ import annotations

from dataclasses import dataclass
import colorsys

import numpy as np


@dataclass(frozen=True)
class MeshRegionBudget:
    """A surface partition and its exact per-region population capacities."""

    centers: np.ndarray
    capacities: np.ndarray
    face_regions: np.ndarray
    face_colors: np.ndarray
    region_colors: np.ndarray
    region_areas: np.ndarray

    def assignments(self, points: np.ndarray) -> np.ndarray:
        """Assign world-space points to their nearest budget-region centre."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(pts) == 0 or len(self.centers) == 0:
            return np.empty((len(pts),), dtype=np.int32)
        out = np.empty(len(pts), dtype=np.int32)
        centers = np.asarray(self.centers, dtype=np.float32)
        for start in range(0, len(pts), 8192):
            block = pts[start:start + 8192]
            d2 = np.sum(
                (block[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            out[start:start + len(block)] = np.argmin(d2, axis=1)
        return out

    def counts(self, points: np.ndarray) -> np.ndarray:
        assignment = self.assignments(points)
        return np.bincount(
            assignment, minlength=len(self.capacities)).astype(np.int32)


def _palette(count: int) -> np.ndarray:
    colors = np.ones((count, 4), dtype=np.float32)
    for index in range(count):
        hue = (0.08 + index * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.98)
        colors[index] = (red, green, blue, 1.0)
    return colors


def _apportion_capacities(
    areas: np.ndarray,
    total: int,
    minimum: int,
    area_power: float,
) -> np.ndarray:
    """Largest-remainder apportionment whose integer result sums to *total*."""
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    count = len(areas)
    if count == 0:
        return np.empty((0,), dtype=np.int32)
    total = max(count, int(total))
    minimum = max(1, min(int(minimum), total // count))
    caps = np.full(count, minimum, dtype=np.int32)
    remaining = int(total - int(np.sum(caps)))
    if remaining <= 0:
        return caps

    safe = np.maximum(areas, np.finfo(np.float64).eps)
    weights = np.power(safe, float(np.clip(area_power, 0.05, 1.0)))
    weights /= float(np.sum(weights))
    exact = weights * remaining
    whole = np.floor(exact).astype(np.int32)
    caps += whole
    leftover = remaining - int(np.sum(whole))
    if leftover:
        order = np.argsort(-(exact - whole), kind="stable")
        caps[order[:leftover]] += 1
    return caps


def build_mesh_region_budget(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_primitives: int,
    *,
    target_capacity: int = 6,
    minimum_capacity: int = 2,
    area_power: float = 0.65,
) -> MeshRegionBudget | None:
    """Partition a mesh and assign size-aware local primitive ceilings.

    ``area_power < 1`` is intentional: large regions still receive the larger
    absolute cap, but small regions receive more primitives per unit area.  It
    prevents the torso from consuming the budget while retaining enough
    capacity for hands, fingers and other fine structures.
    """
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    max_primitives = max(1, int(max_primitives))
    target_capacity = max(1, int(target_capacity))
    minimum_capacity = max(1, int(minimum_capacity))
    if len(verts) == 0 or len(tris) == 0:
        return None
    if np.min(tris) < 0 or np.max(tris) >= len(verts):
        raise ValueError("mesh faces contain invalid vertex indices")

    triangles = verts[tris].astype(np.float64)
    centroids = np.mean(triangles, axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    valid = np.isfinite(centroids).all(axis=1) & np.isfinite(areas) & (areas > 0.0)
    if not np.any(valid):
        return None

    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / twice_area[valid, None]
    desired_regions = max(1, int(np.ceil(max_primitives / target_capacity)))
    affordable_regions = max(1, max_primitives // minimum_capacity)
    region_count = min(int(np.sum(valid)), desired_regions, affordable_regions, 64)

    valid_indices = np.flatnonzero(valid)
    valid_centroids = centroids[valid]
    valid_normals = normals[valid]
    valid_areas = areas[valid]

    # Deterministic farthest-point surface seeds.  The normal term prevents
    # opposite sides of a thin sheet from collapsing into exactly the same
    # patch even when their Euclidean positions are close.
    weighted_center = np.average(valid_centroids, axis=0, weights=valid_areas)
    first = int(np.argmax(np.sum((valid_centroids - weighted_center) ** 2, axis=1)))
    seed_rows = [first]
    min_score = np.full(len(valid_centroids), np.inf, dtype=np.float64)
    normal_scale2 = max(
        float(np.sum(valid_areas)) / (np.pi * max(region_count, 1)),
        np.finfo(np.float64).eps,
    )
    for _ in range(1, region_count):
        seed = seed_rows[-1]
        delta = valid_centroids - valid_centroids[seed]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        normal_delta = 1.0 - np.clip(valid_normals @ valid_normals[seed], -1.0, 1.0)
        score = distance2 + 0.30 * normal_scale2 * normal_delta
        min_score = np.minimum(min_score, score)
        min_score[np.asarray(seed_rows, dtype=np.int64)] = -1.0
        seed_rows.append(int(np.argmax(min_score)))

    seeds = valid_centroids[np.asarray(seed_rows, dtype=np.int64)]
    seed_normals = valid_normals[np.asarray(seed_rows, dtype=np.int64)]
    assignment_valid = np.empty(len(valid_centroids), dtype=np.int32)
    for start in range(0, len(valid_centroids), 8192):
        block = valid_centroids[start:start + 8192]
        block_normals = valid_normals[start:start + 8192]
        d2 = np.sum((block[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
        normal_delta = 1.0 - np.clip(block_normals @ seed_normals.T, -1.0, 1.0)
        assignment_valid[start:start + len(block)] = np.argmin(
            d2 + 0.30 * normal_scale2 * normal_delta, axis=1)

    # Degenerate faces are harmless and inherit the nearest spatial patch so
    # the colour array always remains face-aligned.
    face_regions = np.empty(len(tris), dtype=np.int32)
    face_regions[valid_indices] = assignment_valid
    invalid_indices = np.flatnonzero(~valid)
    if len(invalid_indices):
        d2 = np.sum(
            (centroids[invalid_indices, None, :] - seeds[None, :, :]) ** 2,
            axis=2,
        )
        face_regions[invalid_indices] = np.argmin(d2, axis=1)

    region_areas = np.bincount(
        assignment_valid, weights=valid_areas,
        minlength=region_count).astype(np.float64)
    centers = np.empty((region_count, 3), dtype=np.float64)
    for region in range(region_count):
        mask = assignment_valid == region
        if np.any(mask):
            centers[region] = np.average(
                valid_centroids[mask], axis=0, weights=valid_areas[mask])
        else:
            centers[region] = seeds[region]

    capacities = _apportion_capacities(
        region_areas, max_primitives, minimum_capacity, area_power)
    colors = _palette(region_count)
    face_colors = colors[face_regions]
    return MeshRegionBudget(
        centers=np.ascontiguousarray(centers, dtype=np.float32),
        capacities=np.ascontiguousarray(capacities, dtype=np.int32),
        face_regions=np.ascontiguousarray(face_regions, dtype=np.int32),
        face_colors=np.ascontiguousarray(face_colors, dtype=np.float32),
        region_colors=np.ascontiguousarray(colors, dtype=np.float32),
        region_areas=np.ascontiguousarray(region_areas, dtype=np.float32),
    )
