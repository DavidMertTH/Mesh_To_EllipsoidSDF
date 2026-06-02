"""
thickness.py — Local feature-thickness field from a voxel SDF.

Motivation
----------
The under-representation metric needs to know *how thick* each region of the
mesh is, so that a given absolute deviation on a thin feature (a finger) counts
as a much larger *relative* error than the same deviation on a thick limb.

Using the per-voxel SDF interior depth as the denominator is wrong: a voxel
just under the surface of a thick limb has a tiny depth too, so the thick limb
would be over-flagged near its surface.  We instead want a quantity that is
~constant across a feature's cross-section and equals the feature's diameter.

Local thickness (Hildebrand & Rüegsegger)
-----------------------------------------
The local thickness at an interior point ``p`` is the diameter of the *largest
inscribed sphere of the shape that contains p*.  Every maximal inscribed sphere
is centred on the medial axis (the ridge of the distance field) with radius
equal to the interior distance there.  So:

    1.  D(q)        = interior distance-to-surface  (= -sdf, q inside).
    2.  ridge       = local maxima of D  (medial-axis voxels).
    3.  thickness(p)= 2 · max{ D(q) : q on ridge, |p-q| ≤ D(q) }.

Implementation: splat a sphere of radius D(q) and value 2·D(q) from every ridge
voxel, processed largest-radius-first, keeping the per-voxel maximum.  Centres
already covered by a bigger sphere are skipped — this keeps it fast enough for a
one-off preprocessing pass.
"""

from __future__ import annotations

import numpy as np


def dilate_zeros(field: np.ndarray, iters: int = 2) -> np.ndarray:
    """Grow non-zero values into zero voxels (6-neighbour max), ``iters`` times.

    Surface voxels lie just outside the mesh where the interior thickness field
    is 0; dilating fills them with the adjacent interior feature thickness so the
    surface inherits its feature's thickness (used for loss weighting + display).
    """
    f = field.astype(np.float32, copy=True)
    for _ in range(iters):
        zero = f == 0.0
        if not zero.any():
            break
        m = f.copy()
        m[:-1] = np.maximum(m[:-1], f[1:])
        m[1:] = np.maximum(m[1:], f[:-1])
        m[:, :-1] = np.maximum(m[:, :-1], f[:, 1:])
        m[:, 1:] = np.maximum(m[:, 1:], f[:, :-1])
        m[:, :, :-1] = np.maximum(m[:, :, :-1], f[:, :, 1:])
        m[:, :, 1:] = np.maximum(m[:, :, 1:], f[:, :, :-1])
        f = np.where(zero, m, f)
    return f


def _sphere_offset_cache():
    cache: dict[int, np.ndarray] = {}

    def offsets(r_int: int) -> np.ndarray:
        cached = cache.get(r_int)
        if cached is not None:
            return cached
        rng = np.arange(-r_int, r_int + 1)
        zz, yy, xx = np.meshgrid(rng, rng, rng, indexing="ij")
        mask = (xx * xx + yy * yy + zz * zz) <= r_int * r_int
        offs = np.stack([zz[mask], yy[mask], xx[mask]], axis=1).astype(np.int32)
        cache[r_int] = offs
        return offs

    return offsets


def _medial_ridge(Dv: np.ndarray) -> np.ndarray:
    """Boolean mask of distance-field local maxima (medial-axis proxy).

    A voxel is kept if its depth is >= all 6 face-neighbours (plateaus kept).
    Only interior voxels (Dv > 0) qualify.
    """
    interior = Dv > 0.0
    ridge = interior.copy()
    # Compare against each 6-neighbour using padded shifts so borders (outside
    # the mesh) never wrongly suppress an interior ridge voxel.
    pad = np.pad(Dv, 1, mode="constant", constant_values=0.0)
    c = pad[1:-1, 1:-1, 1:-1]
    ridge &= c >= pad[0:-2, 1:-1, 1:-1]
    ridge &= c >= pad[2:, 1:-1, 1:-1]
    ridge &= c >= pad[1:-1, 0:-2, 1:-1]
    ridge &= c >= pad[1:-1, 2:, 1:-1]
    ridge &= c >= pad[1:-1, 1:-1, 0:-2]
    ridge &= c >= pad[1:-1, 1:-1, 2:]
    return ridge


def local_thickness(
    target_grid: np.ndarray,
    dx: float,
    thickness_floor_vox: float = 1.0,
) -> np.ndarray:
    """Return a (nz, ny, nx) local-thickness field in **world units**.

    Interior voxels get the diameter of the largest inscribed sphere containing
    them; exterior voxels get 0.  A floor of ``thickness_floor_vox`` voxels is
    applied to every interior voxel so the value is never sub-voxel.
    """
    Dv = np.maximum(-target_grid / float(dx), 0.0).astype(np.float32)  # radius, voxels
    nz, ny, nx = Dv.shape
    thick = np.zeros_like(Dv)  # holds diameter in voxel units

    ridge = _medial_ridge(Dv)
    idx = np.argwhere(ridge)
    if idx.size == 0:
        return thick.astype(np.float32)

    radii = Dv[ridge]
    order = np.argsort(-radii)
    idx = idx[order]
    radii = radii[order]

    offsets = _sphere_offset_cache()

    for (z, y, x), r in zip(idx, radii):
        diam = 2.0 * float(r)
        if thick[z, y, x] >= diam:
            continue  # already covered by a larger inscribed sphere
        r_int = int(round(r))
        if r_int < 1:
            if thick[z, y, x] < diam:
                thick[z, y, x] = diam
            continue
        offs = offsets(r_int)
        zz = z + offs[:, 0]
        yy = y + offs[:, 1]
        xx = x + offs[:, 2]
        valid = (zz >= 0) & (zz < nz) & (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        zz, yy, xx = zz[valid], yy[valid], xx[valid]
        cur = thick[zz, yy, xx]
        upd = cur < diam
        thick[zz[upd], yy[upd], xx[upd]] = diam

    # Floor every interior voxel and convert to world units.
    interior = Dv > 0.0
    floor = float(thickness_floor_vox)
    thick[interior] = np.maximum(thick[interior], floor)
    return (thick * float(dx)).astype(np.float32)


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic SDF: a thin slab (thickness 4 vox) joined to a thick block
    # (thickness ~20 vox). Local thickness should recover ~those diameters.
    n = 64
    dx = 1.0
    grid = np.full((n, n, n), 10.0, dtype=np.float32)  # outside (+)

    def fill_box(g, z0, z1, y0, y1, x0, x1):
        # crude interior SDF inside the box: -distance to nearest box face
        for z in range(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    d = min(z - z0, z1 - 1 - z, y - y0, y1 - 1 - y, x - x0, x1 - 1 - x) + 1
                    g[z, y, x] = -float(d)

    # thick block: 20 vox tall in z
    fill_box(grid, 10, 30, 20, 44, 20, 44)
    # thin slab: 4 vox tall in z, attached
    fill_box(grid, 24, 28, 20, 44, 44, 60)

    th = local_thickness(grid, dx)
    block_vals = th[15:25, 25:39, 25:39]
    slab_vals = th[24:28, 25:39, 48:58]
    print(f"thick block local thickness (expect ~20): "
          f"median={np.median(block_vals[block_vals > 0]):.1f}  max={block_vals.max():.1f}")
    print(f"thin slab  local thickness (expect ~4) : "
          f"median={np.median(slab_vals[slab_vals > 0]):.1f}  max={slab_vals.max():.1f}")
