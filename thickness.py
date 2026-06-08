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
inscribed sphere of the shape that contains p*.  By definition

    thickness(p) = 2 · max{ D(q) : q interior, |p−q| ≤ D(q) },   D(q) = −sdf(q).

Implementation: splat a sphere of radius D(q) and value 2·D(q) from every
interior voxel, processed largest-radius-first, keeping the per-voxel maximum.
A voxel already covered by a bigger sphere is skipped, so the expensive splats
happen only for the maximal spheres (the medial axis emerges from the skip rule
itself) — no separate, discretisation-fragile ridge detection is needed.

Precision
---------
Two refinements push the result from the ~1.5–2 voxel low bias of the naïve
voxel-sampled version down to sub-0.5 voxel:

  * **Sub-voxel peak depth** — the medial axis generally falls *between* voxel
    centres, so the deepest sampled ``D`` is short of the true inscribed radius
    (e.g. a cylinder axis at a voxel corner samples ``R−√2/2`` instead of ``R``).
    Near a distance-field ridge the central-difference gradient magnitude drops
    below 1; that per-axis deficit is the sub-voxel distance to the true ridge
    and is added back to ``D`` (``_refine_peak_depth``).
  * **Float-radius coverage** — spheres are rasterised at their exact (refined)
    radius instead of an integer-rounded one, so thin features keep their true
    diameter.

Both are one-off costs in the SDF preprocessing pass (~2 s on a 256³ grid).
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


def _sphere_offset_cache(quant_vox: float = 1.0 / 16.0):
    """Cache of integer voxel offsets inside a sphere of (float) radius ``r``.

    Radii are quantised to ``quant_vox`` (sub-voxel) so the cache stays small
    while coverage error from quantisation is well below a tenth of a voxel.
    """
    cache: dict[int, np.ndarray] = {}

    def offsets(r: float) -> np.ndarray:
        key = int(round(float(r) / quant_vox))
        cached = cache.get(key)
        if cached is not None:
            return cached
        rq = key * quant_vox
        ext = int(np.ceil(rq))
        if ext < 1:
            offs = np.zeros((1, 3), dtype=np.int32)  # centre voxel only
        else:
            rng = np.arange(-ext, ext + 1)
            zz, yy, xx = np.meshgrid(rng, rng, rng, indexing="ij")
            mask = (xx * xx + yy * yy + zz * zz) <= rq * rq
            offs = np.stack([zz[mask], yy[mask], xx[mask]], axis=1).astype(np.int32)
        cache[key] = offs
        return offs

    return offsets


def _refine_peak_depth(Dv: np.ndarray) -> np.ndarray:
    """Sub-voxel-refined interior depth (voxel units).

    The medial axis of a feature generally lies between voxel centres, so the
    deepest *sampled* ``D`` underestimates the true inscribed radius.  In a true
    distance field ``|∇D| = 1`` everywhere except on the medial ridge, where the
    central-difference magnitude collapses; that per-axis deficit equals the
    sub-voxel distance from the voxel centre to the ridge, so adding it back
    recovers the peak (exact for slabs, ≲0.5 vox for cylinders/spheres).

    Only voxels that are a local maximum along an axis (concave there) receive a
    gain, each per-axis gain is capped at half a voxel, and the per-axis gains
    are combined *Euclidean* (not summed): the gain is the length of the
    sub-voxel offset vector to the ridge, so an isotropic point ridge (sphere
    centre) is not over-counted the way a plain sum would be.
    """
    sq = np.zeros_like(Dv)
    for ax in range(3):
        a = np.roll(Dv, 1, axis=ax)
        c = np.roll(Dv, -1, axis=ax)
        grad = 0.5 * (c - a)                 # central difference along axis
        concave = (a - 2.0 * Dv + c) < -1e-6  # voxel is a ridge peak on this axis
        gain = np.where(concave, np.clip(np.abs(grad), 0.0, 0.5), 0.0)
        sq = sq + gain * gain
    ref = Dv + np.sqrt(sq).astype(Dv.dtype)
    # Refinement is meaningful only strictly inside the shape.
    ref[Dv <= 0.0] = 0.0
    return ref


def local_thickness(
    target_grid: np.ndarray,
    dx: float,
    thickness_floor_vox: float = 1.0,
) -> np.ndarray:
    """Return a (nz, ny, nx) local-thickness field in **world units**.

    Interior voxels get the diameter of the largest inscribed sphere containing
    them; exterior voxels get 0.  A floor of ``thickness_floor_vox`` voxels is
    applied to every interior voxel so the value is never sub-voxel.

    Precise but one-off: every interior voxel is a candidate sphere centre and
    spheres are rasterised at their exact sub-voxel-refined radius.  The
    largest-first ordering plus the skip-already-covered test keep the expensive
    splats limited to the maximal spheres (~thousands even on a 256³ grid).
    """
    Dv = np.maximum(-target_grid / float(dx), 0.0).astype(np.float32)  # radius, voxels
    nz, ny, nx = Dv.shape
    thick = np.zeros_like(Dv)  # holds diameter in voxel units

    interior = Dv > 0.0
    idx = np.argwhere(interior)
    if idx.size == 0:
        return thick.astype(np.float32)

    radii = _refine_peak_depth(Dv)[interior]   # sub-voxel-refined inscribed radii
    order = np.argsort(-radii)
    idx = idx[order]
    radii = radii[order]

    offsets = _sphere_offset_cache()

    for (z, y, x), r in zip(idx, radii):
        diam = 2.0 * float(r)
        if thick[z, y, x] >= diam:
            continue  # already covered by a larger inscribed sphere
        offs = offsets(float(r))
        zz = z + offs[:, 0]
        yy = y + offs[:, 1]
        xx = x + offs[:, 2]
        valid = (zz >= 0) & (zz < nz) & (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        zz, yy, xx = zz[valid], yy[valid], xx[valid]
        cur = thick[zz, yy, xx]
        upd = cur < diam
        thick[zz[upd], yy[upd], xx[upd]] = diam

    # Floor every interior voxel and convert to world units.
    floor = float(thickness_floor_vox)
    thick[interior] = np.maximum(thick[interior], floor)
    return (thick * float(dx)).astype(np.float32)


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Accuracy regression on *true Euclidean* SDFs (warp produces these), where
    # the analytic local thickness is known exactly.  The deep-interior median
    # must land within ~0.5 voxel of the true diameter — the sub-voxel peak
    # refinement is what makes that possible (the naïve version was ~1.5–2 low).
    n, dx = 96, 1.0
    rng = (np.arange(n) + 0.5) * dx
    zz, yy, xx = np.meshgrid(rng, rng, rng, indexing="ij")
    c = n * dx / 2.0

    def sphere(R):
        return (np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) - R).astype(np.float32)

    def cylinder(R):  # infinite along z
        return (np.sqrt((xx - c) ** 2 + (yy - c) ** 2) - R).astype(np.float32)

    def slab(half):   # infinite in y, z
        return (np.abs(xx - c) - half).astype(np.float32)

    cases = [("sphere R=12", sphere(12.0), -6.0, 24.0),
             ("cylinder R=8", cylinder(8.0), -4.0, 16.0),
             ("cylinder R=2.5", cylinder(2.5), -0.5, 5.0),
             ("slab half=4", slab(4.0), -2.0, 8.0),
             ("slab half=1.5", slab(1.5), -0.3, 3.0)]
    worst = 0.0
    for label, g, thr, truth in cases:
        th = local_thickness(g, dx)
        med = float(np.median(th[g < thr]))
        err = abs(med - truth)
        worst = max(worst, err)
        flag = "OK " if err < 0.5 else "!! "
        print(f"{flag}{label:16s} median={med:7.3f}  truth={truth:5.2f}  |err|={err:.3f}")
    print(f"worst |err| = {worst:.3f} voxel  ({'PASS' if worst < 0.5 else 'FAIL'})")
