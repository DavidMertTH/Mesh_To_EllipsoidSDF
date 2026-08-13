"""
sdf_slice.py — planar slices through a 3-D SDF volume for the movable
texture plane in the 3-D viewport.

The slice plane is perpendicular to one world axis ("normal").  Plane labels map
to that normal axis:  XY → z,  XZ → y,  YZ → x.  A 2-D slice is turned into an
RGBA texture (via the shared SDF colormap LUT) plus a 4×4 world transform that
places and orients the textured quad so its texels line up exactly with the
voxel grid the SDF was sampled on.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from sdf_colormap import make_sdf_lut, colorize_sdf_slice, SLICE_INTERIOR_GAMMA
from ellipsoid import best_device

# Plane label → normal world-axis index (x=0, y=1, z=2).
PLANE_NORMAL = {"XY": 2, "XZ": 1, "YZ": 0}
PLANE_LABELS = ("XY", "XZ", "YZ")

# SDF display range (matches the 2-D SdfSlicePanel for a consistent look).
SLICE_VMIN = -0.2
SLICE_VMAX = 0.2

# Interior colour easing constant lives in sdf_colormap (shared with the 2-D
# SdfSlicePanel); re-exported here as the default for the GPU render functions.

# Slice texture resolution: the longer in-plane axis gets this many pixels (the
# shorter axis is scaled to keep a square pixel / true aspect).
SLICE_TARGET_PX = 700


def slice_resolution(grid_shape, dx: float, normal_idx: int,
                     target: int = SLICE_TARGET_PX):
    """Aspect-correct ``(W, H, pixel_size)`` for a ~2K slice texture.

    The longer in-plane world extent maps to ``target`` pixels; the pixel is
    square (uniform ``pixel_size``), so the shorter axis gets proportionally
    fewer pixels and the slice is not distorted.
    """
    a0, a1 = inplane_axes(normal_idx)
    ext0 = n_slices(grid_shape, a0) * float(dx)
    ext1 = n_slices(grid_shape, a1) * float(dx)
    px = max(ext0, ext1) / float(max(int(target), 1))
    px = max(px, 1e-9)
    W = max(1, int(round(ext0 / px)))
    H = max(1, int(round(ext1 / px)))
    return W, H, float(px)


def resample_bilinear(field: np.ndarray, W: int, H: int,
                      dx: float, pixel_size: float) -> np.ndarray:
    """Bilinearly resample a 2-D ``[a0, a1]`` field (voxel spacing ``dx``) onto a
    regular ``(W, H)`` grid of spacing ``pixel_size`` — i.e. interpolate between
    the known grid pixels.  Both grids share the same origin/voxel-centre
    convention so the in-plane geometry lines up exactly.
    """
    n0, n1 = field.shape
    field = field.astype(np.float32, copy=False)
    # Target pixel centre i → source fractional voxel index.
    fi = (np.arange(W, dtype=np.float32) + 0.5) * pixel_size / dx - 0.5
    fj = (np.arange(H, dtype=np.float32) + 0.5) * pixel_size / dx - 0.5
    fi = np.clip(fi, 0.0, n0 - 1.0)
    fj = np.clip(fj, 0.0, n1 - 1.0)
    i0 = np.floor(fi).astype(np.int64); i1 = np.minimum(i0 + 1, n0 - 1)
    j0 = np.floor(fj).astype(np.int64); j1 = np.minimum(j0 + 1, n1 - 1)
    ti = (fi - i0)[:, None]; tj = (fj - j0)[None, :]
    f00 = field[np.ix_(i0, j0)]; f01 = field[np.ix_(i0, j1)]
    f10 = field[np.ix_(i1, j0)]; f11 = field[np.ix_(i1, j1)]
    top = f00 * (1.0 - tj) + f01 * tj
    bot = f10 * (1.0 - tj) + f11 * tj
    return np.ascontiguousarray(top * (1.0 - ti) + bot * ti)


def inplane_axes(normal_idx: int) -> tuple[int, int]:
    """The two in-plane world axes (ascending) for a given normal axis."""
    a = tuple(x for x in (0, 1, 2) if x != normal_idx)
    return int(a[0]), int(a[1])


def n_slices(grid_shape: tuple[int, int, int], normal_idx: int) -> int:
    """Number of voxels along ``normal_idx`` for a ``(nz, ny, nx)`` grid."""
    nz, ny, nx = grid_shape
    return (nx, ny, nz)[normal_idx]


def extract_slice(grid: np.ndarray, normal_idx: int, k: int) -> np.ndarray:
    """2-D slice of a ``(nz, ny, nx)`` grid perpendicular to ``normal_idx``.

    Returns an array indexed ``[i_a0, j_a1]`` where ``(a0, a1)`` are the two
    in-plane world axes in ascending order — already in GLImageItem's
    ``(width, height)`` layout (local x ↔ a0, local y ↔ a1).  Works for a full
    grid or a single-slab grid (normal axis of length 1, then ``k == 0``).
    """
    if normal_idx == 2:        # XY plane, slice along z
        sl = grid[k, :, :]     # (ny, nx) = [iy, ix]
    elif normal_idx == 1:      # XZ plane, slice along y
        sl = grid[:, k, :]     # (nz, nx) = [iz, ix]
    else:                      # YZ plane, slice along x
        sl = grid[:, :, k]     # (nz, ny) = [iz, iy]
    return np.ascontiguousarray(sl.T)   # → [a0, a1]


def slice_rgba(slice2d: np.ndarray, lut: np.ndarray | None = None,
               depth: float = 1.0, out_band: float = 1.0) -> np.ndarray:
    """Map a 2-D SDF slice to an ``(W, H, 4)`` uint8 RGBA texture via the LUT.

    The interior and exterior are scaled **independently** so the slice reads as
    a crisp cross-section instead of a soft halo:

      * interior (SDF < 0): normalised by ``depth`` (the deepest interior
        magnitude of the volume) — the maximum interior colour is reached only at
        the genuinely deepest point, and the gradient in between reveals depth
        structure;
      * exterior (SDF > 0): normalised by a SMALL ``out_band`` (a few voxels) so
        the colour fades to the background within a few voxels of the surface —
        a sharp boundary, NOT a wide halo whose width grows with ``depth`` (which
        made thin features look like grey smudges).
    """
    # Delegate to the shared (warp-free) colouriser so the 2-D SdfSlicePanel and
    # this 3-D slice render identically.  Matches the GPU _color_sdf_kernel.
    return colorize_sdf_slice(slice2d, lut, depth, out_band)


def slice_rgba_diff(diff: np.ndarray, primary_rgb, secondary_rgb,
                    scale: float = 1.0, mid_rgb=(128, 128, 138)) -> np.ndarray:
    """Opaque diverging colour-ramp RGBA for the difference between two slices.

    The difference is mapped onto a continuous ramp ``secondary ← neutral →
    primary``: ``diff = +scale`` → **primary**, ``diff = −scale`` → **secondary**,
    ``diff = 0`` → the neutral ``mid_rgb`` (so a perfect match is still visible,
    not transparent).  ``primary_rgb`` / ``secondary_rgb`` are 0–255 RGB triples.
    """
    v = np.clip(diff.astype(np.float32) / max(float(scale), 1e-9), -1.0, 1.0)
    prim = np.asarray(primary_rgb, dtype=np.float32)[:3]
    sec = np.asarray(secondary_rgb, dtype=np.float32)[:3]
    mid = np.asarray(mid_rgb, dtype=np.float32)[:3]
    mag = np.abs(v)[..., None]
    target = np.where((v >= 0.0)[..., None], prim, sec)     # (W, H, 3)
    rgb = mid * (1.0 - mag) + target * mag                  # ramp through neutral
    rgba = np.empty(diff.shape + (4,), dtype=np.ubyte)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.ubyte)
    rgba[..., 3] = 255                                      # opaque ramp
    return np.ascontiguousarray(rgba)


def slice_transform(origin, normal_idx: int, k: int,
                    pixel_size: float, dx: float) -> np.ndarray:
    """4×4 world transform placing GLImageItem texels onto the slice plane.

    Texel ``(i, j)`` at local ``(i, j, 0)`` maps to in-plane world position with
    spacing ``pixel_size`` (the 2K texture spacing); the whole quad sits at the
    ``k``-th *grid* voxel centre along the normal axis (spacing ``dx``).  Returns
    a float64 ``(4, 4)`` matrix (row-major, ``M @ v``).
    """
    a0, a1 = inplane_axes(normal_idx)
    ps = float(pixel_size)
    M = np.zeros((4, 4), dtype=np.float64)
    M[a0, 0] = ps              # +1 in local x → pixel_size along a0
    M[a1, 1] = ps              # +1 in local y → pixel_size along a1
    M[normal_idx, 2] = ps      # local z is flat/unused; keep det > 0
    t = np.asarray(origin, dtype=np.float64).copy()
    t[normal_idx] += (float(k) + 0.5) * float(dx)
    M[0, 3], M[1, 3], M[2, 3] = float(t[0]), float(t[1]), float(t[2])
    M[3, 3] = 1.0
    return M


# ── direct per-pixel ellipsoid-union evaluation ───────────────────────────────
# The slice for the ellipsoid source is computed by walking EVERY ellipsoid at
# each pixel and keeping the smallest SDF (the union).  The quaternion is
# normalised here — the trained quaternions drift off unit length (Adam does not
# renormalise the stored value), and without this an ellipsoid would vanish
# (|q|>1) or balloon to fill the slice (|q|<1).

@wp.kernel
def _slice_min_kernel(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),     # flat (num_e * 4), (x,y,z,w)
    num_e: int,
    points: wp.array(dtype=wp.vec3),          # one world position per pixel
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = points[tid]
    min_d = float(1.0e6)
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(             # normalise the (drifted) quat
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        local_p = wp.quat_rotate_inv(q, p - centers[i])
        r = radii[i]
        scaled = wp.vec3(local_p[0] / r[0], local_p[1] / r[1], local_p[2] / r[2])
        k0 = wp.length(scaled)
        d = float(1.0e6)
        if k0 < 1.0:
            r_min = wp.min(wp.min(r[0], r[1]), r[2])
            d = (k0 - 1.0) * r_min
        else:
            scaled2 = wp.vec3(
                local_p[0] / (r[0] * r[0]),
                local_p[1] / (r[1] * r[1]),
                local_p[2] / (r[2] * r[2]))
            k1 = wp.max(wp.length(scaled2), 1.0e-8)
            d = k0 * (k0 - 1.0) / k1
        if d < min_d:
            min_d = d
    out[tid] = min_d


def slice_points(origin, normal_idx: int, k: int, W: int, H: int,
                 pixel_size: float, dx: float) -> np.ndarray:
    """World positions of the ``(W, H)`` 2K in-plane pixels at slice index ``k``.

    In-plane spacing is ``pixel_size`` (the 2K texture); the plane sits at the
    ``k``-th *grid* voxel centre along the normal (spacing ``dx``).  Ordered so
    ``reshape(W, H)`` lines up with the ``[a0, a1]`` layout / :func:`slice_transform`.
    """
    a0, a1 = inplane_axes(normal_idx)
    o = np.asarray(origin, dtype=np.float32)
    ps = float(pixel_size)
    ii = (np.arange(W, dtype=np.float32) + 0.5) * ps
    jj = (np.arange(H, dtype=np.float32) + 0.5) * ps
    Ig, Jg = np.meshgrid(ii, jj, indexing="ij")        # (W, H)
    pts = np.empty((W, H, 3), dtype=np.float32)
    pts[..., a0] = o[a0] + Ig
    pts[..., a1] = o[a1] + Jg
    pts[..., normal_idx] = o[normal_idx] + (float(k) + 0.5) * float(dx)
    return pts.reshape(-1, 3)


def ellipsoid_slice_sdf(centers, radii, rotations, points) -> np.ndarray:
    """Union (min) MertStein SDF over ALL ellipsoids, evaluated per point.

    For each world point in ``points`` (M, 3) the kernel walks every ellipsoid,
    evaluates its SDF (quaternion normalised) and keeps the minimum — exactly the
    "loop all ellipsoids, store the smallest value" the slice needs.  Returns
    ``(M,)`` float32.  Empty population → all-positive (outside).
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    m = int(pts.shape[0])
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    n_e = int(centers.shape[0])
    if n_e == 0 or m == 0:
        return np.full(m, 1.0e6, dtype=np.float32)
    dev = best_device()
    wc = wp.array(centers, dtype=wp.vec3, device=dev)
    wr = wp.array(np.ascontiguousarray(np.abs(radii), dtype=np.float32),
                  dtype=wp.vec3, device=dev)
    wq = wp.array(np.ascontiguousarray(rotations, dtype=np.float32).reshape(-1),
                  dtype=wp.float32, device=dev)
    wp_pts = wp.array(pts, dtype=wp.vec3, device=dev)
    out = wp.empty(m, dtype=wp.float32, device=dev)
    wp.launch(_slice_min_kernel, dim=m,
              inputs=[wc, wr, wq, n_e, wp_pts, out], device=dev)
    return out.numpy()


# ── GPU render pipeline ───────────────────────────────────────────────────────
# Everything below evaluates the 2K slice AND colours it entirely on the GPU,
# reading back only the final RGBA (uint8) so the GUI thread is not blocked by
# 3-megapixel numpy work.  Each pixel's world position is computed in-kernel from
# a plane basis (p0 + i*step0 + j*step1) so no host point array is needed.
# Keep kernel comments ASCII-only (warp transpiles them to a cp1252 .cu file).


@wp.func
def _mertstein(local_p: wp.vec3, r: wp.vec3) -> wp.float32:
    scaled = wp.vec3(local_p[0] / r[0], local_p[1] / r[1], local_p[2] / r[2])
    k0 = wp.length(scaled)
    if k0 < 1.0:
        r_min = wp.min(wp.min(r[0], r[1]), r[2])
        return (k0 - 1.0) * r_min
    scaled2 = wp.vec3(local_p[0] / (r[0] * r[0]),
                      local_p[1] / (r[1] * r[1]),
                      local_p[2] / (r[2] * r[2]))
    k1 = wp.max(wp.length(scaled2), 1.0e-8)
    return k0 * (k0 - 1.0) / k1


@wp.kernel
def _ell_field_kernel(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    num_e: int,
    p0: wp.vec3, step0: wp.vec3, step1: wp.vec3, H: int,
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    i = tid // H
    j = tid - i * H
    p = p0 + float(i) * step0 + float(j) * step1
    min_d = float(1.0e6)
    for e in range(num_e):
        b = e * 4
        q = wp.normalize(wp.quat(rot_flat[b + 0], rot_flat[b + 1],
                                 rot_flat[b + 2], rot_flat[b + 3]))
        d = _mertstein(wp.quat_rotate_inv(q, p - centers[e]), radii[e])
        if d < min_d:
            min_d = d
    out[tid] = min_d


@wp.kernel
def _grid_field_kernel(
    grid: wp.array(dtype=wp.float32),       # flat (nz*ny*nx)
    gorigin: wp.vec3, dx: float, nx: int, ny: int, nz: int,
    p0: wp.vec3, step0: wp.vec3, step1: wp.vec3, H: int,
    offset: float,                          # SDF "blowup": uniform offset added
    out: wp.array(dtype=wp.float32),
):
    # Trilinear sample of the mesh SDF grid (interpolate between known voxels).
    tid = wp.tid()
    i = tid // H
    j = tid - i * H
    p = p0 + float(i) * step0 + float(j) * step1
    gx = (p[0] - gorigin[0]) / dx - 0.5
    gy = (p[1] - gorigin[1]) / dx - 0.5
    gz = (p[2] - gorigin[2]) / dx - 0.5
    ix0 = wp.clamp(int(wp.floor(gx)), 0, nx - 2)
    iy0 = wp.clamp(int(wp.floor(gy)), 0, ny - 2)
    iz0 = wp.clamp(int(wp.floor(gz)), 0, nz - 2)
    fx = wp.clamp(gx - float(ix0), 0.0, 1.0)
    fy = wp.clamp(gy - float(iy0), 0.0, 1.0)
    fz = wp.clamp(gz - float(iz0), 0.0, 1.0)
    nynx = ny * nx
    b = iz0 * nynx + iy0 * nx + ix0
    c00 = grid[b] * (1.0 - fx) + grid[b + 1] * fx
    c10 = grid[b + nx] * (1.0 - fx) + grid[b + nx + 1] * fx
    c01 = grid[b + nynx] * (1.0 - fx) + grid[b + nynx + 1] * fx
    c11 = grid[b + nynx + nx] * (1.0 - fx) + grid[b + nynx + nx + 1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    out[tid] = c0 * (1.0 - fz) + c1 * fz + offset


@wp.func
def _thickness_limited_grid_value(
    value: wp.float32,
    thickness: wp.float32,
    requested_offset: wp.float32,
    dx: wp.float32,
    max_thickness_fraction: wp.float32,
) -> wp.float32:
    effective = requested_offset
    if thickness > 0.0:
        direction = 1.0
        if requested_offset < 0.0:
            direction = -1.0
        effective = direction * wp.min(
            wp.abs(requested_offset),
            max_thickness_fraction * thickness,
        )
    else:
        # Missing thickness can be an unresolved thin feature.  Keep it
        # unchanged everywhere; a large negative offset must not pull a
        # distant unresolved sample into a false exterior surface band.
        effective = 0.0
    return value + effective


@wp.kernel
def _adaptive_grid_field_kernel(
    grid: wp.array(dtype=wp.float32),
    thickness: wp.array(dtype=wp.float32),
    gorigin: wp.vec3, dx: float, nx: int, ny: int, nz: int,
    p0: wp.vec3, step0: wp.vec3, step1: wp.vec3, H: int,
    offset: float,
    max_thickness_fraction: float,
    out: wp.array(dtype=wp.float32),
):
    # Apply the cap independently at all eight grid corners before trilinear
    # interpolation.  This exactly matches interpolating the adjusted CPU grid.
    tid = wp.tid()
    i = tid // H
    j = tid - i * H
    p = p0 + float(i) * step0 + float(j) * step1
    gx = (p[0] - gorigin[0]) / dx - 0.5
    gy = (p[1] - gorigin[1]) / dx - 0.5
    gz = (p[2] - gorigin[2]) / dx - 0.5
    ix0 = wp.clamp(int(wp.floor(gx)), 0, nx - 2)
    iy0 = wp.clamp(int(wp.floor(gy)), 0, ny - 2)
    iz0 = wp.clamp(int(wp.floor(gz)), 0, nz - 2)
    fx = wp.clamp(gx - float(ix0), 0.0, 1.0)
    fy = wp.clamp(gy - float(iy0), 0.0, 1.0)
    fz = wp.clamp(gz - float(iz0), 0.0, 1.0)
    nynx = ny * nx
    b = iz0 * nynx + iy0 * nx + ix0

    v000 = _thickness_limited_grid_value(
        grid[b], thickness[b], offset, dx, max_thickness_fraction)
    v100 = _thickness_limited_grid_value(
        grid[b + 1], thickness[b + 1], offset, dx,
        max_thickness_fraction)
    v010 = _thickness_limited_grid_value(
        grid[b + nx], thickness[b + nx], offset, dx,
        max_thickness_fraction)
    v110 = _thickness_limited_grid_value(
        grid[b + nx + 1], thickness[b + nx + 1], offset, dx,
        max_thickness_fraction)
    v001 = _thickness_limited_grid_value(
        grid[b + nynx], thickness[b + nynx], offset, dx,
        max_thickness_fraction)
    v101 = _thickness_limited_grid_value(
        grid[b + nynx + 1], thickness[b + nynx + 1], offset, dx,
        max_thickness_fraction)
    v011 = _thickness_limited_grid_value(
        grid[b + nynx + nx], thickness[b + nynx + nx], offset, dx,
        max_thickness_fraction)
    v111 = _thickness_limited_grid_value(
        grid[b + nynx + nx + 1], thickness[b + nynx + nx + 1],
        offset, dx, max_thickness_fraction)
    c00 = v000 * (1.0 - fx) + v100 * fx
    c10 = v010 * (1.0 - fx) + v110 * fx
    c01 = v001 * (1.0 - fx) + v101 * fx
    c11 = v011 * (1.0 - fx) + v111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    out[tid] = c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def _color_sdf_kernel(
    field: wp.array(dtype=wp.float32),
    depth: float, out_band: float, gamma: float,
    lut: wp.array(dtype=wp.float32), n_lut: int,   # flat (n_lut*4) RGBA 0..255
    surf_idx: int,                                 # LUT entry for the surface (t=0.5)
    rgba: wp.array(dtype=wp.uint8),                # flat (M*4)
):
    # Exterior (SDF > 0): LUT mapping surface -> far, as before.
    # Interior (SDF < 0): blend the LUT's SURFACE colour -> DEEPEST colour
    # directly across the WHOLE interior, eased by gamma>1.  The shared LUT has
    # an interior plateau (it hits ~full colour just past the surface), so
    # indexing it would saturate early; lerping the two endpoints instead makes
    # the full colour appear ONLY at the deepest point with a long, even blend.
    tid = wp.tid()
    v = field[tid]
    o = tid * 4
    if v < 0.0:
        mag = wp.pow(wp.min(-v / depth, 1.0), gamma)   # 0 surface .. 1 deepest
        si = surf_idx * 4                              # surface LUT entry
        rgba[o + 0] = wp.uint8(lut[si + 0] * (1.0 - mag) + lut[0] * mag)
        rgba[o + 1] = wp.uint8(lut[si + 1] * (1.0 - mag) + lut[1] * mag)
        rgba[o + 2] = wp.uint8(lut[si + 2] * (1.0 - mag) + lut[2] * mag)
        rgba[o + 3] = wp.uint8(lut[si + 3] * (1.0 - mag) + lut[3] * mag)
    else:
        s = wp.min(v / out_band, 1.0)
        t = 0.5 + 0.5 * s
        idx = wp.clamp(int(t * float(n_lut - 1)), 0, n_lut - 1)
        li = idx * 4
        rgba[o + 0] = wp.uint8(lut[li + 0])
        rgba[o + 1] = wp.uint8(lut[li + 1])
        rgba[o + 2] = wp.uint8(lut[li + 2])
        rgba[o + 3] = wp.uint8(lut[li + 3])


@wp.kernel
def _maxabs_diff_kernel(
    ell: wp.array(dtype=wp.float32), mesh: wp.array(dtype=wp.float32),
    out_max: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    wp.atomic_max(out_max, 0, wp.abs(ell[tid] - mesh[tid]))


@wp.kernel
def _color_diff_kernel(
    ell: wp.array(dtype=wp.float32), mesh: wp.array(dtype=wp.float32),
    scale: float, prim: wp.vec3, sec: wp.vec3, mid: wp.vec3,
    rgba: wp.array(dtype=wp.uint8),
):
    # Opaque diverging ramp secondary <- mid -> primary == slice_rgba_diff.
    tid = wp.tid()
    v = wp.clamp((ell[tid] - mesh[tid]) / scale, -1.0, 1.0)
    mag = wp.abs(v)
    if v >= 0.0:
        col = mid * (1.0 - mag) + prim * mag
    else:
        col = mid * (1.0 - mag) + sec * mag
    o = tid * 4
    rgba[o + 0] = wp.uint8(wp.clamp(col[0], 0.0, 255.0))
    rgba[o + 1] = wp.uint8(wp.clamp(col[1], 0.0, 255.0))
    rgba[o + 2] = wp.uint8(wp.clamp(col[2], 0.0, 255.0))
    rgba[o + 3] = wp.uint8(255)


def _plane_basis(origin, normal_idx, k, px, dx):
    """(p0, step0, step1) wp.vec3 for the in-plane pixel basis at slice ``k``."""
    a0, a1 = inplane_axes(normal_idx)
    o = np.asarray(origin, dtype=np.float64).copy()
    o[a0] += 0.5 * px
    o[a1] += 0.5 * px
    o[normal_idx] += (float(k) + 0.5) * float(dx)
    s0 = np.zeros(3); s0[a0] = px
    s1 = np.zeros(3); s1[a1] = px
    return (wp.vec3(float(o[0]), float(o[1]), float(o[2])),
            wp.vec3(float(s0[0]), float(s0[1]), float(s0[2])),
            wp.vec3(float(s1[0]), float(s1[1]), float(s1[2])))


def make_lut_wp(lut: np.ndarray | None = None):
    """Upload the SDF colour LUT to the device; returns (wp_array, n_entries)."""
    if lut is None:
        lut = make_sdf_lut()
    dev = best_device()
    flat = np.ascontiguousarray(lut.astype(np.float32).reshape(-1))
    return wp.array(flat, dtype=wp.float32, device=dev), int(lut.shape[0])


def upload_grid(grid: np.ndarray):
    """Upload the (nz,ny,nx) SDF grid to the device once (cache in the viewer)."""
    dev = best_device()
    # ``astype`` without ``copy=False`` duplicated even an already-contiguous
    # 512³ float32 carrier (another 512 MiB).  Keep the upload staging view
    # zero-copy whenever the caller already supplies production layout/dtype.
    flat = np.ascontiguousarray(
        np.asarray(grid, dtype=np.float32)).reshape(-1)
    return wp.array(flat, dtype=wp.float32, device=dev)


def render_ellipsoid(centers, radii, rotations, origin, normal_idx, k, W, H,
                     px, dx, lut_wp, n_lut, depth, out_band,
                     gamma=SLICE_INTERIOR_GAMMA):
    """GPU-render the ellipsoid-union slice → ``(W, H, 4)`` uint8 RGBA."""
    dev = best_device()
    n_e = int(np.asarray(centers).shape[0])
    M = int(W) * int(H)
    p0, s0, s1 = _plane_basis(origin, normal_idx, k, px, dx)
    field = wp.empty(M, dtype=wp.float32, device=dev)
    if n_e == 0:
        field.fill_(1.0e6)
    else:
        wc = wp.array(np.ascontiguousarray(centers, np.float32), dtype=wp.vec3, device=dev)
        wr = wp.array(np.ascontiguousarray(np.abs(radii), np.float32), dtype=wp.vec3, device=dev)
        wq = wp.array(np.ascontiguousarray(rotations, np.float32).reshape(-1),
                      dtype=wp.float32, device=dev)
        wp.launch(_ell_field_kernel, dim=M,
                  inputs=[wc, wr, wq, n_e, p0, s0, s1, int(H), field], device=dev)
    rgba = wp.empty(M * 4, dtype=wp.uint8, device=dev)
    wp.launch(_color_sdf_kernel, dim=M,
              inputs=[field, float(depth), float(out_band), float(gamma),
                      lut_wp, int(n_lut), (int(n_lut) - 1) // 2, rgba],
              device=dev)
    return rgba.numpy().reshape(int(W), int(H), 4)


def render_mesh(grid_wp, gorigin, dx, nx, ny, nz, origin, normal_idx, k, W, H,
                px, lut_wp, n_lut, depth, out_band, offset=0.0,
                gamma=SLICE_INTERIOR_GAMMA, thickness_wp=None,
                max_thickness_fraction=0.25):
    """GPU-render the (trilinearly interpolated) mesh slice → uint8 RGBA.

    ``offset`` is the requested SDF blowup.  With ``thickness_wp`` its magnitude
    is capped per voxel by ``max_thickness_fraction`` of local feature thickness.
    """
    dev = best_device()
    M = int(W) * int(H)
    p0, s0, s1 = _plane_basis(origin, normal_idx, k, px, dx)
    field = wp.empty(M, dtype=wp.float32, device=dev)
    common = [
        wp.vec3(float(gorigin[0]), float(gorigin[1]), float(gorigin[2])),
        float(dx), int(nx), int(ny), int(nz), p0, s0, s1, int(H),
    ]
    if thickness_wp is None:
        wp.launch(
            _grid_field_kernel, dim=M,
            inputs=[grid_wp, *common, float(offset), field],
            device=dev)
    else:
        wp.launch(
            _adaptive_grid_field_kernel, dim=M,
            inputs=[
                grid_wp, thickness_wp, *common, float(offset),
                float(max_thickness_fraction), field,
            ],
            device=dev)
    rgba = wp.empty(M * 4, dtype=wp.uint8, device=dev)
    wp.launch(_color_sdf_kernel, dim=M,
              inputs=[field, float(depth), float(out_band), float(gamma),
                      lut_wp, int(n_lut), (int(n_lut) - 1) // 2, rgba],
              device=dev)
    return rgba.numpy().reshape(int(W), int(H), 4)


def render_diff(centers, radii, rotations, grid_wp, gorigin, dx, nx, ny, nz,
                origin, normal_idx, k, W, H, px, primary_rgb, secondary_rgb,
                mid_rgb=(128, 128, 138), offset=0.0, thickness_wp=None,
                max_thickness_fraction=0.25):
    """GPU-render ellipsoid−mesh on a diverging two-colour ramp → uint8 RGBA.

    ``offset`` is added to the mesh SDF.  With ``thickness_wp`` the same local
    thickness cap used by the fitting target is applied before interpolation.
    """
    dev = best_device()
    n_e = int(np.asarray(centers).shape[0])
    M = int(W) * int(H)
    p0, s0, s1 = _plane_basis(origin, normal_idx, k, px, dx)
    ell = wp.empty(M, dtype=wp.float32, device=dev)
    if n_e == 0:
        ell.fill_(1.0e6)
    else:
        wc = wp.array(np.ascontiguousarray(centers, np.float32), dtype=wp.vec3, device=dev)
        wr = wp.array(np.ascontiguousarray(np.abs(radii), np.float32), dtype=wp.vec3, device=dev)
        wq = wp.array(np.ascontiguousarray(rotations, np.float32).reshape(-1),
                      dtype=wp.float32, device=dev)
        wp.launch(_ell_field_kernel, dim=M,
                  inputs=[wc, wr, wq, n_e, p0, s0, s1, int(H), ell], device=dev)
    mesh = wp.empty(M, dtype=wp.float32, device=dev)
    common = [
        wp.vec3(float(gorigin[0]), float(gorigin[1]), float(gorigin[2])),
        float(dx), int(nx), int(ny), int(nz), p0, s0, s1, int(H),
    ]
    if thickness_wp is None:
        wp.launch(
            _grid_field_kernel, dim=M,
            inputs=[grid_wp, *common, float(offset), mesh],
            device=dev)
    else:
        wp.launch(
            _adaptive_grid_field_kernel, dim=M,
            inputs=[
                grid_wp, thickness_wp, *common, float(offset),
                float(max_thickness_fraction), mesh,
            ],
            device=dev)
    mx = wp.zeros(1, dtype=wp.float32, device=dev)
    wp.launch(_maxabs_diff_kernel, dim=M, inputs=[ell, mesh, mx], device=dev)
    scale = max(float(mx.numpy()[0]), 1e-4)
    rgba = wp.empty(M * 4, dtype=wp.uint8, device=dev)
    prim = wp.vec3(*[float(c) for c in primary_rgb[:3]])
    sec = wp.vec3(*[float(c) for c in secondary_rgb[:3]])
    mid = wp.vec3(*[float(c) for c in mid_rgb[:3]])
    wp.launch(_color_diff_kernel, dim=M,
              inputs=[ell, mesh, float(scale), prim, sec, mid, rgba], device=dev)
    return rgba.numpy().reshape(int(W), int(H), 4)
