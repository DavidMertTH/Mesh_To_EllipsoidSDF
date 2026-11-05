"""Detail zones: high-resolution sub-fits for features thinner than the global grid.

Every optimizer length scale (training-sample band, loss width, densify and
split limits, initial placement) is tied to the global SDF voxel.  A finger
that is thinner than one voxel therefore cannot be fitted well in a
whole-body fit, although the same hand fits well when loaded on its own.

This module automates "loading the thin part on its own":

1. ``exact_vertex_diameters`` measures the local feature diameter at every
   vertex with exact mesh-distance queries (largest ball that fits inside the
   surface at that vertex), independent of any voxel grid.
2. ``detect_detail_zones`` groups vertices thinner than ``threshold_vox``
   global voxels into connected zones (for a humanoid: hands and toes) and
   gives each zone a padded box.
3. ``allocate_detail_zones`` splits the primitive budget between the main
   fit and the zones; ``carve_grid`` / ``carve_values`` remove the zone cores
   from the main fit's target, so only the zone fits cover them.
4. Each zone's cropped mesh is fitted in its own normalized frame (a much
   finer grid), mirror-symmetric zones once and reflected, and
   ``merge_zone_results`` combines everything.

The functions are pure numpy; mesh-distance queries are passed in as a
callable so the module stays independent of the GPU backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

Query = Callable[[np.ndarray], np.ndarray]


@dataclass
class DetailZone:
    """One region refitted at its own resolution (world/normalized-mesh frame)."""

    index: int
    lo: np.ndarray            # crop box (core padded)
    hi: np.ndarray
    core_lo: np.ndarray       # box around the thin vertices
    core_hi: np.ndarray
    vertex_indices: np.ndarray
    area: float
    median_diameter: float
    complexity: float = 0.0   # surface area / median diameter^2 (dimensionless)
    reserve: int = 0          # primitives reserved from the total budget
    mirror_of: Optional[int] = None   # source zone this one is reflected from
    mirror_axis: Optional[int] = None
    mirror_plane: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo


@dataclass
class ZonePlan:
    """Budget and zones decided before the global fit starts."""

    zones: list
    total_num: int
    total_max: int
    global_num: int
    global_max: int
    dx: float
    settings: dict = field(default_factory=dict)

    @property
    def sources(self) -> list:
        return [z for z in self.zones if z.mirror_of is None]


# ── geometry helpers ─────────────────────────────────────────────────────────

def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(v)
    for k in range(3):
        np.add.at(normals, f[:, k], cross)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-20)).astype(np.float32)


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    tri = v[f]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(v))
    for k in range(3):
        np.add.at(out, f[:, k], area / 3.0)
    return out


def exact_vertex_diameters(vertices: np.ndarray, faces: np.ndarray, query: Query,
                           *, steps: int = 96, chunk: int = 8192) -> np.ndarray:
    """Local feature diameter per vertex from exact signed-distance queries.

    The diameter is twice the radius of the largest ball that touches the
    vertex from inside: walking inward along the vertex normal, the ball of
    radius t centred at ``p - t n`` stays inside while ``-sdf(p - t n) >= t``.
    ``query`` returns signed distances (negative inside) for (k, 3) points.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    if len(v) == 0:
        return np.zeros(0, np.float32)
    n = vertex_normals(v, faces)
    extent = float(np.max(v.max(axis=0) - v.min(axis=0)))
    if extent <= 0.0:
        return np.zeros(len(v), np.float32)
    # Orient each normal outward with the field itself; meshes with flipped
    # or inconsistent winding still give a correct inward walk.
    probe_step = np.float32(1e-4 * extent)
    outward = (np.asarray(query(v + probe_step * n))
               - np.asarray(query(v - probe_step * n)))
    n = n * np.where(outward >= 0.0, 1.0, -1.0)[:, None].astype(np.float32)
    ts = np.geomspace(2e-4 * extent, 0.5 * extent, int(steps)).astype(np.float32)
    out = np.empty(len(v), np.float32)
    for a in range(0, len(v), int(chunk)):
        b = min(len(v), a + int(chunk))
        pts = (v[a:b, None, :] - ts[None, :, None] * n[a:b, None, :]).reshape(-1, 3)
        depth = -np.asarray(query(np.ascontiguousarray(pts))).reshape(b - a, len(ts))
        inside = depth >= 0.95 * ts[None, :]
        first_fail = np.argmax(~inside, axis=1)
        first_fail[inside.all(axis=1)] = len(ts)
        out[a:b] = 2.0 * ts[np.clip(first_fail - 1, 0, len(ts) - 1)]
    return out


def smooth_vertex_values(values: np.ndarray, faces: np.ndarray,
                         iterations: int = 2) -> np.ndarray:
    """Average each vertex with its one-ring a few times (removes single spikes)."""
    v = np.asarray(values, np.float64).copy()
    f = np.asarray(faces, np.int64)
    for _ in range(int(iterations)):
        acc = np.zeros_like(v)
        count = np.zeros_like(v)
        for k in range(3):
            for j in range(3):
                np.add.at(acc, f[:, k], v[f[:, j]])
                np.add.at(count, f[:, k], 1.0)
        v = acc / np.maximum(count, 1.0)
    return v.astype(np.float32)


def connected_components(n: int, edges: np.ndarray) -> np.ndarray:
    """Component label per vertex (label propagation with pointer jumping)."""
    labels = np.arange(n, dtype=np.int64)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return labels
    a, b = edges[:, 0], edges[:, 1]
    while True:
        low = np.minimum(labels[a], labels[b])
        new = labels.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]                 # pointer jumping speeds up long chains
        if np.array_equal(new, labels):
            return labels
        labels = new


# ── zone detection ───────────────────────────────────────────────────────────

def detect_detail_zones(
    vertices: np.ndarray,
    faces: np.ndarray,
    diameters: np.ndarray,
    dx: float,
    *,
    threshold_vox: float = 1.5,
    seed_vox: float = 1.0,
    box_quantile: float = 0.02,
    pad_fraction: float = 0.10,
    min_area_vox: float = 40.0,
    join_vox: float = 2.0,
    smoothing: int = 2,
) -> list:
    """Regions whose features are thinner than ``threshold_vox`` global voxels.

    Thin vertices are grouped by mesh connectivity.  Groups smaller than
    ``min_area_vox`` voxel faces (stray creases, nostrils) are ignored and
    groups closer than ``join_vox`` voxels are merged.  Each zone's core box
    spans the clearly thin vertices (``seed_vox``), trimmed by
    ``box_quantile`` so a few thin outliers on a neighbouring limb cannot
    stretch it; the crop box adds ``pad_fraction`` of the core diagonal.
    """
    v = np.asarray(vertices, np.float32)
    f = np.asarray(faces, np.int64)
    dx = float(dx)
    if len(v) == 0 or len(f) == 0 or dx <= 0.0:
        return []
    d = (smooth_vertex_values(diameters, f, smoothing)
         if smoothing else np.asarray(diameters, np.float32))
    thin = d < float(threshold_vox) * dx
    if not np.any(thin):
        return []
    varea = vertex_areas(v, f)
    thin_faces = thin[f].all(axis=1)
    tf = f[thin_faces]
    edges = np.concatenate([tf[:, [0, 1]], tf[:, [1, 2]], tf[:, [2, 0]]])
    labels = connected_components(len(v), edges)
    groups = []
    for label in np.unique(labels[thin]):
        idx = np.flatnonzero((labels == label) & thin)
        area = float(varea[idx].sum())
        if area < float(min_area_vox) * dx * dx:
            continue
        groups.append([idx, v[idx].min(axis=0), v[idx].max(axis=0), area])
    join = float(join_vox) * dx
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                gi, gj = groups[i], groups[j]
                gap = np.maximum(0.0, np.maximum(gi[1] - gj[2], gj[1] - gi[2]))
                if float(np.linalg.norm(gap)) <= join:
                    groups[i] = [np.concatenate([gi[0], gj[0]]),
                                 np.minimum(gi[1], gj[1]), np.maximum(gi[2], gj[2]),
                                 gi[3] + gj[3]]
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    zones = []
    for idx, _lo, _hi, area in groups:
        core = idx
        if seed_vox is not None:
            seeds = idx[d[idx] < float(seed_vox) * dx]
            if len(seeds) >= 8:
                core = seeds
        q = float(np.clip(box_quantile, 0.0, 0.49))
        core_lo = np.quantile(v[core], q, axis=0).astype(np.float32)
        core_hi = np.quantile(v[core], 1.0 - q, axis=0).astype(np.float32)
        pad = max(float(pad_fraction) * float(np.linalg.norm(core_hi - core_lo)), 2.0 * dx)
        median = float(np.median(d[idx]))
        zones.append(DetailZone(
            index=len(zones), lo=core_lo - pad, hi=core_hi + pad,
            core_lo=core_lo, core_hi=core_hi, vertex_indices=idx, area=area,
            median_diameter=median,
            complexity=float(area / max(median, 1e-12) ** 2)))
    return zones


def pair_mirrored_zones(zones: Sequence[DetailZone], vertices: np.ndarray,
                        faces: np.ndarray, query: Query, axis: int, plane: float,
                        dx: float, *, tolerance_vox: float = 0.5) -> None:
    """Mark zones that are mirror images of an earlier zone across ``plane``.

    A pair needs matching reflected boxes and mirror-symmetric geometry: the
    reflected crop of the source must lie on the mesh surface (mean |sdf|
    below ``tolerance_vox`` voxels).
    """
    v = np.asarray(vertices, np.float32)
    used = set()
    for zone in zones:
        if zone.index in used:
            continue
        reflected_center = zone.center.copy()
        reflected_center[axis] = 2.0 * plane - reflected_center[axis]
        best, best_distance = None, np.inf
        for other in zones:
            if other.index == zone.index or other.index in used:
                continue
            distance = float(np.linalg.norm(other.center - reflected_center))
            if distance < best_distance:
                best, best_distance = other, distance
        if best is None or best_distance > 3.0 * dx:
            continue
        if np.any(np.abs(best.size - zone.size) > 3.0 * dx):
            continue
        sample = v[zone.vertex_indices]
        if len(sample) > 4000:
            sample = sample[np.linspace(0, len(sample) - 1, 4000).astype(np.int64)]
        mirrored = sample.copy()
        mirrored[:, axis] = 2.0 * plane - mirrored[:, axis]
        error = float(np.mean(np.abs(np.asarray(query(np.ascontiguousarray(mirrored))))))
        if error > float(tolerance_vox) * dx:
            continue
        best.mirror_of = zone.index
        best.mirror_axis = int(axis)
        best.mirror_plane = float(plane)
        used.update((zone.index, best.index))


def find_mirror_pairs(zones: Sequence[DetailZone], vertices: np.ndarray, faces: np.ndarray,
                      query: Query, dx: float) -> Optional[tuple[int, float]]:
    """Pair zones across the first axis-aligned plane through the mesh centre
    that maps them onto each other geometrically; returns (axis, plane)."""
    v = np.asarray(vertices, np.float32)
    middle = 0.5 * (v.min(axis=0) + v.max(axis=0))
    for axis in range(3):
        pair_mirrored_zones(zones, vertices, faces, query, axis, float(middle[axis]), dx)
        if any(z.mirror_of is not None for z in zones):
            return axis, float(middle[axis])
    return None


def allocate_detail_zones(zones: list, vertices: np.ndarray, faces: np.ndarray,
                          diameters: np.ndarray, dx: float, total_num: int, total_max: int, *,
                          weight: float = 1.0, power: float = 1.25, min_zone: int = 8,
                          max_share: float = 0.6, min_global: int = 10,
                          growth: float = 0.1,
                          settings: Optional[dict] = None) -> ZonePlan:
    """Split the primitive budget between the main fit and the zones up front.

    Every region gets primitives in proportion to ``area / diameter**power``:
    surface area times a density that rises as features get thinner.  The main
    fit is weighted by the area and median diameter of everything outside the
    zone cores; ``weight`` scales all zones.  Zones never get fewer than
    ``min_zone`` primitives and together at most ``max_share`` of the budget.
    """
    total_max = max(1, int(total_max))
    total_num = max(1, min(int(total_num), total_max))
    v = np.asarray(vertices, np.float32)
    d = np.asarray(diameters, np.float32)
    areas = vertex_areas(v, faces)
    in_core = np.zeros(len(v), bool)
    for zone in zones:
        in_core |= np.all((v >= zone.core_lo) & (v <= zone.core_hi), axis=1)
    body_area = float(areas[~in_core].sum())
    body_diameter = float(np.median(d[~in_core])) if np.any(~in_core) else 1.0
    p = float(power)
    body_weight = body_area / max(body_diameter, 1e-12) ** p
    zone_weights = np.array([float(weight) * z.area / max(z.median_diameter, 1e-12) ** p
                             for z in zones], float)
    total_weight = body_weight + float(zone_weights.sum())
    budgets = (total_max * zone_weights / max(total_weight, 1e-12)) if len(zones) else zone_weights
    budgets = np.maximum(budgets, float(min_zone))
    cap = float(max_share) * total_max
    if budgets.sum() > cap > 0.0:
        budgets *= cap / budgets.sum()
    budgets = np.floor(budgets + 0.5).astype(int)
    for zone, budget in zip(zones, budgets):
        zone.reserve = int(budget)
    global_max = max(int(min_global), total_max - int(budgets.sum()))
    # The main fit keeps the user's initial count: its final population grows
    # only moderately beyond the start, and the carved mesh is almost all of
    # the original, so a proportionally smaller start would starve the body.
    # ``growth`` keeps room below the maximum so SuperFit can still add
    # primitives where the main fit needs them.
    room = int(np.ceil(float(growth) * global_max))
    global_num = max(1, min(total_num, global_max - room))
    return ZonePlan(zones=list(zones), total_num=total_num, total_max=total_max,
                    global_num=global_num, global_max=global_max, dx=float(dx),
                    settings=dict(settings or {}))


def box_sdf(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Signed distance to an axis-aligned box (negative inside)."""
    p = np.asarray(points, np.float32)
    center = 0.5 * (np.asarray(lo, np.float32) + np.asarray(hi, np.float32))
    half = 0.5 * (np.asarray(hi, np.float32) - np.asarray(lo, np.float32))
    q = np.abs(p - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return (outside + inside).astype(np.float32)


def carve_values(points: np.ndarray, values: np.ndarray, zones: Sequence[DetailZone]):
    """Signed distances of the mesh with every zone core removed.

    ``max(sdf, -box_sdf)`` is the CSG difference "mesh minus box": the main fit
    sees each zone core as empty space, so it spends no primitives there and
    keeps its primitives out of it.  Returns (values, changed_mask).
    """
    out = np.asarray(values, np.float32).copy()
    changed = np.zeros(len(out), bool)
    for zone in zones:
        carved = -box_sdf(points, zone.core_lo, zone.core_hi)
        mask = carved > out
        out[mask] = carved[mask]
        changed |= mask
    return out, changed


def seam_band_mask(points: np.ndarray, values: np.ndarray, zones: Sequence[DetailZone],
                   band: float, vertices: Optional[np.ndarray] = None) -> np.ndarray:
    """Samples inside a zone core that lie next to main-fit surface (the seam).

    Where the main fit's geometry (a forearm, a heel) continues into a zone
    core, the main fit should neither be pushed out of nor pulled into the
    core: its primitives then end naturally a little inside the zone and
    overlap the zone's primitives instead of leaving a gap at the seam.
    With ``vertices`` the band follows the mesh vertices that belong to no
    zone (the actual seam); without them every core face gets the band.
    Only points inside the mesh are freed: sticking out of the skin stays
    penalized everywhere.
    """
    p = np.asarray(points, np.float32)
    v = np.asarray(values, np.float32)
    mask = np.zeros(len(p), bool)
    band = float(band)
    if band <= 0.0 or len(p) == 0:
        return mask
    inside_mesh = v <= 0.0
    main_vertices = None
    if vertices is not None:
        verts = np.asarray(vertices, np.float32)
        in_zone = np.zeros(len(verts), bool)
        for zone in zones:
            in_zone[np.asarray(zone.vertex_indices, np.int64)] = True
        main_vertices = verts[~in_zone]
    for zone in zones:
        depth = -box_sdf(p, zone.core_lo, zone.core_hi)      # > 0 inside the core
        candidates = np.flatnonzero((depth > 0.0) & inside_mesh)
        if len(candidates) == 0:
            continue
        if main_vertices is None:
            mask[candidates[depth[candidates] < band]] = True
            continue
        nearby = main_vertices[np.all((main_vertices >= zone.core_lo - band)
                                      & (main_vertices <= zone.core_hi + band), axis=1)]
        if len(nearby) == 0:
            continue
        for start in range(0, len(candidates), 4096):
            chunk = candidates[start:start + 4096]
            diff = p[chunk, None, :] - nearby[None, :, :]
            nearest = np.sqrt(np.min(np.einsum("ijk,ijk->ij", diff, diff), axis=1))
            mask[chunk[nearest < band]] = True
    return mask


def carve_grid(grid: np.ndarray, origin: np.ndarray, dx: float,
               zones: Sequence[DetailZone]) -> np.ndarray:
    """``carve_values`` for a dense (nz, ny, nx) grid with voxel centres at
    ``origin + (index + 0.5) * dx``; only the slabs touching a core are visited."""
    out = np.array(grid, dtype=np.float32, copy=True)
    nz, ny, nx = out.shape
    origin = np.asarray(origin, np.float32)
    xs = origin[0] + (np.arange(nx, dtype=np.float32) + 0.5) * dx
    ys = origin[1] + (np.arange(ny, dtype=np.float32) + 0.5) * dx
    zs = origin[2] + (np.arange(nz, dtype=np.float32) + 0.5) * dx
    # max(sdf, -box_sdf) changes a voxel inside the core box, or an interior
    # voxel that is closer to the box than to the mesh surface.  The latter
    # lie within the deepest interior distance of the box.
    reach = max(0.0, -float(np.min(out))) if out.size else 0.0
    for zone in zones:
        lo = zone.core_lo - reach
        hi = zone.core_hi + reach
        ix = np.flatnonzero((xs >= lo[0]) & (xs <= hi[0]))
        iy = np.flatnonzero((ys >= lo[1]) & (ys <= hi[1]))
        iz = np.flatnonzero((zs >= lo[2]) & (zs <= hi[2]))
        if not (len(ix) and len(iy) and len(iz)):
            continue
        Z, Y, X = np.meshgrid(zs[iz], ys[iy], xs[ix], indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        block = out[iz[0]:iz[-1] + 1, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1].reshape(-1)
        carved = -box_sdf(pts, zone.core_lo, zone.core_hi)
        out[iz[0]:iz[-1] + 1, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] = np.maximum(
            block, carved).reshape(len(iz), len(iy), len(ix))
    return out


def inside_any_core(centers: np.ndarray, zones: Sequence[DetailZone],
                    shrink: float = 0.0) -> np.ndarray:
    """Primitives whose centre lies inside a zone core (shrunk by ``shrink``)."""
    c = np.asarray(centers, np.float32).reshape(-1, 3)
    mask = np.zeros(len(c), bool)
    for zone in zones:
        mask |= np.all((c >= zone.core_lo + shrink) & (c <= zone.core_hi - shrink), axis=1)
    return mask


# ── zone fitting helpers ─────────────────────────────────────────────────────

def zone_submesh(vertices: np.ndarray, faces: np.ndarray,
                 zone: DetailZone) -> tuple[np.ndarray, np.ndarray]:
    """Faces fully inside the zone box, re-indexed (unreferenced vertices dropped)."""
    v = np.asarray(vertices, np.float32)
    f = np.asarray(faces, np.int64)
    inside = np.all((v >= zone.lo) & (v <= zone.hi), axis=1)
    kept = f[inside[f].all(axis=1)]
    used = np.unique(kept)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    return v[used].copy(), remap[kept].astype(np.int32)


def normalization(vertices: np.ndarray, target_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """(center, scale) of ``mesh_io.normalize_mesh_capture``: p_norm = (p - center) * scale."""
    v = np.asarray(vertices, np.float64)
    center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    extent = float(np.max(v.max(axis=0) - v.min(axis=0)))
    scale = (2.0 * float(target_scale) / extent) if extent > 0.0 else 1.0
    return center.astype(np.float32), float(scale)


def zone_grid_resolution(zone_vertices: np.ndarray, median_diameter: float, *,
                         voxels_per_feature: float = 8.0, n_min: int = 128,
                         n_max: int = 320, margin: float = 0.1) -> int:
    """Grid n so the zone's median feature spans ``voxels_per_feature`` voxels."""
    v = np.asarray(zone_vertices, np.float64)
    if len(v) == 0 or median_diameter <= 0.0:
        return int(n_min)
    extent = float(np.max(v.max(axis=0) - v.min(axis=0))) * (1.0 + float(margin))
    n = extent / (float(median_diameter) / float(voxels_per_feature))
    n = int(np.ceil(n / 16.0) * 16)
    return int(np.clip(n, int(n_min), max(int(n_min), int(n_max))))


def _shape_array(shape, n: int) -> Optional[np.ndarray]:
    if shape is None:
        return None
    a = np.asarray(shape, np.float32)
    return a.reshape(n, -1) if a.size else a.reshape(n, 0)


def from_normalized(params, center: np.ndarray, scale: float):
    """Map (centers, radii, rotations, shape) from a normalized zone frame back.

    Shape columns 2-3 (bend curvature of bent superquadrics, 1/length) scale
    with the inverse length factor; exponents are dimensionless.
    """
    c, r, q, shape = params
    c = np.asarray(c, np.float32) / np.float32(scale) + np.asarray(center, np.float32)
    r = np.asarray(r, np.float32) / np.float32(scale)
    shape = _shape_array(shape, len(c))
    if shape is not None and shape.shape[1] >= 4:
        shape = shape.copy()
        shape[:, 2:4] *= np.float32(scale)
    return c, r, np.asarray(q, np.float32), shape


def mirror_params(params, axis: int, plane: float):
    """Reflect primitives across the plane x[axis] = plane.

    A reflection M = diag(..-1 at axis..) maps a rotation R to M R M, whose
    quaternion keeps the ``axis`` component and negates the two others.  In
    the primitive's local frame the same axis flips, so a bend along that local
    axis (shape columns 2-3) changes sign.
    """
    c, r, q, shape = params
    c = np.asarray(c, np.float32).copy()
    c[:, axis] = 2.0 * np.float32(plane) - c[:, axis]
    q = np.asarray(q, np.float32).copy()
    for k in range(3):
        if k != axis:
            q[:, k] = -q[:, k]
    shape = _shape_array(shape, len(c))
    if shape is not None and shape.shape[1] >= 4 and axis in (0, 1):
        shape = shape.copy()
        shape[:, 2 + axis] = -shape[:, 2 + axis]
    return c, np.asarray(r, np.float32).copy(), q, shape


def merge_zone_results(global_params, keep: np.ndarray, zone_params: Sequence):
    """Global primitives outside every zone core, then each zone's primitives."""
    c, r, q, shape = global_params
    n = len(c)
    shape = _shape_array(shape, n)
    parts = [(np.asarray(c, np.float32)[keep], np.asarray(r, np.float32)[keep],
              np.asarray(q, np.float32)[keep], None if shape is None else shape[keep])]
    parts.extend(zone_params)
    centers = np.concatenate([p[0] for p in parts]).astype(np.float32)
    radii = np.concatenate([p[1] for p in parts]).astype(np.float32)
    rotations = np.concatenate([p[2] for p in parts]).astype(np.float32)
    shapes = [p[3] for p in parts]
    if all(s is None for s in shapes):
        merged_shape = None
    else:
        width = max(s.shape[1] for s in shapes if s is not None)
        filled = []
        for p, s in zip(parts, shapes):
            block = np.zeros((len(p[0]), width), np.float32)
            if width >= 2:
                block[:, :2] = 1.0       # neutral exponents for shape-less parts
            if s is not None and s.size:
                block[:, :s.shape[1]] = s
            filled.append(block)
        merged_shape = np.concatenate(filled).astype(np.float32)
    return centers, radii, rotations, merged_shape


# ── process protocol (used by detail_zone_worker / detail_zone_stage) ────────

MESSAGE_PREFIX = "@@DZ "


def pack_params(centers, radii, rotations, shape) -> dict:
    """Compact, lossless float32 encoding of one primitive population."""
    import base64
    c = np.asarray(centers, np.float32).reshape(-1, 3)
    n = len(c)
    s = _shape_array(shape, n)
    k = 0 if s is None else int(s.shape[1])
    blocks = [c, np.asarray(radii, np.float32).reshape(n, 3),
              np.asarray(rotations, np.float32).reshape(n, 4)]
    if k:
        blocks.append(s)
    data = np.concatenate(blocks, axis=1).astype(np.float32) if n else np.zeros((0, 10 + k), np.float32)
    return {"n": n, "k": k, "data": base64.b64encode(np.ascontiguousarray(data).tobytes()).decode("ascii")}


def unpack_params(message: dict):
    import base64
    n, k = int(message["n"]), int(message["k"])
    flat = np.frombuffer(base64.b64decode(message["data"]), dtype=np.float32)
    data = flat.reshape(n, 10 + k) if n else np.zeros((0, 10 + k), np.float32)
    shape = data[:, 10:10 + k].copy() if k else None
    return data[:, 0:3].copy(), data[:, 3:6].copy(), data[:, 6:10].copy(), shape


# ── zone quality check ───────────────────────────────────────────────────────

def _quat_matrices(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)


def union_inside(points: np.ndarray, params, *, capsule: bool = False,
                 chunk: int = 4096) -> np.ndarray:
    """Whether each point lies inside the primitive union.

    Superquadrics use the implicit ((|x|^(2/e2) + |y|^(2/e2))^(e2/e1) + |z|^(2/e1)) < 1
    after undoing the bend (x -= kx z^2 / 2, y -= ky z^2 / 2); ellipsoids are
    exponents 1.  Capsules: radius r0 around a z segment of half-length r2.
    """
    c, r, q, shape = params
    p = np.asarray(points, np.float64)
    c = np.asarray(c, np.float64)
    r = np.maximum(np.abs(np.asarray(r, np.float64)), 1e-9)
    rot = _quat_matrices(q)
    n = len(c)
    s = _shape_array(shape, n)
    e = np.ones((n, 2)) if s is None or s.shape[1] < 2 else np.asarray(s[:, :2], np.float64)
    e = np.clip(e, 0.05, 4.0)
    k = np.zeros((n, 2)) if s is None or s.shape[1] < 4 else np.asarray(s[:, 2:4], np.float64)
    inside = np.zeros(len(p), bool)
    for start in range(0, len(p), int(chunk)):
        block = p[start:start + int(chunk)]
        local = np.einsum("nji,pnj->pni", rot, block[:, None, :] - c[None, :, :])
        if capsule:
            z = np.clip(local[..., 2], -r[:, 2], r[:, 2])
            d2 = local[..., 0] ** 2 + local[..., 1] ** 2 + (local[..., 2] - z) ** 2
            inside[start:start + len(block)] = np.any(d2 < r[:, 0] ** 2, axis=1)
            continue
        z = local[..., 2]
        x = local[..., 0] - 0.5 * k[:, 0] * z * z
        y = local[..., 1] - 0.5 * k[:, 1] * z * z
        with np.errstate(divide="ignore"):
            lx = (2.0 / e[:, 1]) * np.log(np.maximum(np.abs(x / r[:, 0]), 1e-300))
            ly = (2.0 / e[:, 1]) * np.log(np.maximum(np.abs(y / r[:, 1]), 1e-300))
            lz = (2.0 / e[:, 0]) * np.log(np.maximum(np.abs(z / r[:, 2]), 1e-300))
        lf = np.logaddexp((e[:, 1] / e[:, 0]) * np.logaddexp(lx, ly), lz)
        inside[start:start + len(block)] = np.any(lf < 0.0, axis=1)
    return inside


def surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 0):
    """Area-uniform surface points and their (unoriented) face normals."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    valid = area > 0
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid)[rng.choice(int(valid.sum()), size=int(count),
                                           p=area[valid] / area[valid].sum())]
    r1, r2 = rng.random(int(count)), rng.random(int(count))
    s1 = np.sqrt(r1)
    bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], axis=1)
    pts = np.einsum("pk,pkj->pj", bary, tri[idx])
    normals = cross[idx] / np.maximum(2.0 * area[idx, None], 1e-30)
    return pts.astype(np.float32), normals.astype(np.float32)


def uncovered_fraction(vertices: np.ndarray, faces: np.ndarray, query: Query, params,
                       inset: float, *, capsule: bool = False, count: int = 20000,
                       seed: int = 0) -> float:
    """Share of the surface whose point ``inset`` below the skin lies in no primitive.

    A missing finger or toe shows up as a few percent of the zone surface;
    small rounding at edges stays well below that.
    """
    pts, normals = surface_samples(vertices, faces, count, seed)
    if len(pts) == 0:
        return 0.0
    eps = np.float32(max(1e-4 * float(np.max(np.ptp(pts, axis=0))), 1e-6))
    outward = np.asarray(query(pts + eps * normals)) - np.asarray(query(pts - eps * normals))
    normals = normals * np.where(outward >= 0.0, 1.0, -1.0)[:, None].astype(np.float32)
    probe = pts - np.float32(inset) * normals
    # Only probes that are really inside the mesh can be judged.
    depth = -np.asarray(query(probe))
    valid = depth > 0.25 * float(inset)
    if not np.any(valid):
        return 0.0
    covered = union_inside(probe[valid], params, capsule=capsule)
    return float(1.0 - covered.mean())
