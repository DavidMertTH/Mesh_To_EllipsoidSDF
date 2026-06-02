"""
raymarch.py — GPU sphere-tracer for the ellipsoid-union SDF.

Most SDFs are visualised by *ray marching*: for every screen pixel a ray is
shot from the camera into the scene and advanced by the SDF value (sphere
tracing) until it hits the iso-surface, which is then shaded from the SDF
gradient.  This module does exactly that, fully parallel on the GPU via NVIDIA
Warp — one thread per pixel — and reads back only the final RGBA image.

The camera is supplied as the inverse model-view-projection matrix (taken
straight from the live pyqtgraph viewport), so the marched image lines up
pixel-exact with the rest of the GL scene (axis, mesh, skeleton).  Rays are
clipped to the scene bounding sphere first, so empty pixels cost almost nothing
and the march range is tight.

Keep all kernel comments ASCII-only — Warp transpiles them into a cp1252 .cu
file and non-ASCII characters break the compile.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ellipsoid import best_device


# ── per-ellipsoid SDF (MertStein form, matches sdf_slice / the trained loss) ───

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


@wp.func
def _smin(a: wp.float32, b: wp.float32, k: wp.float32) -> wp.float32:
    # Polynomial smooth minimum: blends the two surfaces over a width ``k``
    # (k=0 -> a hard min/union).  The result is <= min(a,b), so it stays a
    # conservative distance bound and never makes sphere tracing overshoot.
    if k <= 0.0:
        return wp.min(a, b)
    h = wp.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1.0 - h) + a * h) - k * h * (1.0 - h)


@wp.func
def _union_sdf(
    p: wp.vec3,
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rmax: wp.array(dtype=wp.float32),    # largest semi-axis per ellipsoid
    rot_flat: wp.array(dtype=wp.float32),
    num_e: int,
    blend: wp.float32,
) -> wp.float32:
    # Smooth union of all ellipsoids: a running smooth-min of the per-ellipsoid
    # SDFs.  ``blend == 0`` reduces to the plain (hard) union.
    acc = float(1.0e6)
    for e in range(num_e):
        # Bounding-sphere cull: the distance from p to ellipsoid e is at least
        # (|p - center| - rmax).  If that lower bound is already further than
        # the current accumulator (plus the blend reach), ellipsoid e cannot
        # change the smooth-min, so skip its expensive exact evaluation.  This
        # is exact, not an approximation -- smin only mixes inputs within a band
        # of width ``blend``.  Rays in a warp are coherent, so the branch barely
        # diverges (unlike the scattered grid sampler).
        if wp.length(p - centers[e]) - rmax[e] > acc + blend:
            continue
        b = e * 4
        q = wp.normalize(wp.quat(rot_flat[b + 0], rot_flat[b + 1],
                                 rot_flat[b + 2], rot_flat[b + 3]))
        d = _mertstein(wp.quat_rotate_inv(q, p - centers[e]), radii[e])
        acc = _smin(acc, d, blend)
    return acc


@wp.func
def _normal(
    p: wp.vec3, h: float,
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rmax: wp.array(dtype=wp.float32),
    rot_flat: wp.array(dtype=wp.float32),
    num_e: int,
    blend: wp.float32,
) -> wp.vec3:
    # Tetrahedron gradient (4 SDF taps instead of 6) of the union SDF == normal.
    k0 = wp.vec3(1.0, -1.0, -1.0)
    k1 = wp.vec3(-1.0, -1.0, 1.0)
    k2 = wp.vec3(-1.0, 1.0, -1.0)
    k3 = wp.vec3(1.0, 1.0, 1.0)
    n = (k0 * _union_sdf(p + k0 * h, centers, radii, rmax, rot_flat, num_e, blend)
         + k1 * _union_sdf(p + k1 * h, centers, radii, rmax, rot_flat, num_e, blend)
         + k2 * _union_sdf(p + k2 * h, centers, radii, rmax, rot_flat, num_e, blend)
         + k3 * _union_sdf(p + k3 * h, centers, radii, rmax, rot_flat, num_e, blend))
    return wp.normalize(n)


@wp.kernel
def _raymarch_kernel(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rmax: wp.array(dtype=wp.float32),
    rot_flat: wp.array(dtype=wp.float32),
    num_e: int,
    inv_mvp: wp.mat44,                 # NDC -> world (clip space inverse)
    W: int, H: int,
    sph_c: wp.vec3, sph_r: float,      # scene bounding sphere (ray clip)
    surf_eps: float, normal_h: float, relax: float, max_steps: int,
    blend: float,                      # smooth-union width (world units)
    base: wp.vec3,                     # surface colour (0..1)
    bg: wp.vec3, bg_a: float,          # background colour (0..1) + alpha
    rgba: wp.array(dtype=wp.uint8),    # flat (W*H*4), row-major rows of width W
):
    tid = wp.tid()
    px = tid % W
    py = tid // W
    o = tid * 4

    # Background by default (alpha lets the GL scene show through on a miss).
    bg_r = wp.uint8(wp.clamp(bg[0] * 255.0, 0.0, 255.0))
    bg_g = wp.uint8(wp.clamp(bg[1] * 255.0, 0.0, 255.0))
    bg_b = wp.uint8(wp.clamp(bg[2] * 255.0, 0.0, 255.0))
    bg_aa = wp.uint8(wp.clamp(bg_a * 255.0, 0.0, 255.0))
    rgba[o + 0] = bg_r
    rgba[o + 1] = bg_g
    rgba[o + 2] = bg_b
    rgba[o + 3] = bg_aa

    # Pixel centre -> NDC.  Screen y points down, NDC y points up, so flip y.
    ndc_x = 2.0 * (float(px) + 0.5) / float(W) - 1.0
    ndc_y = 1.0 - 2.0 * (float(py) + 0.5) / float(H)

    near4 = inv_mvp * wp.vec4(ndc_x, ndc_y, -1.0, 1.0)
    far4 = inv_mvp * wp.vec4(ndc_x, ndc_y, 1.0, 1.0)
    wn = 1.0 / near4[3]
    wf = 1.0 / far4[3]
    ro = wp.vec3(near4[0] * wn, near4[1] * wn, near4[2] * wn)
    fp = wp.vec3(far4[0] * wf, far4[1] * wf, far4[2] * wf)
    rd = wp.normalize(fp - ro)

    # Clip the ray to the scene bounding sphere: skip empty pixels cheaply and
    # bound the march range.
    oc = ro - sph_c
    b = wp.dot(oc, rd)
    c = wp.dot(oc, oc) - sph_r * sph_r
    disc = b * b - c
    if disc < 0.0:
        return
    sq = wp.sqrt(disc)
    t = wp.max(-b - sq, 0.0)
    tmax = -b + sq
    if tmax <= 0.0:
        return

    # Sphere trace.  ``relax`` (<1) shortens each step so the approximate
    # ellipsoid distance does not overshoot the surface on grazing rays.
    hit = int(0)
    for s in range(max_steps):
        p = ro + rd * t
        d = _union_sdf(p, centers, radii, rmax, rot_flat, num_e, blend)
        if d < surf_eps:
            hit = 1
            break
        t = t + wp.max(d * relax, surf_eps * 0.5)
        if t > tmax:
            break

    if hit == 0:
        return

    p = ro + rd * t
    n = _normal(p, normal_h, centers, radii, rmax, rot_flat, num_e, blend)
    v = wp.vec3(-rd[0], -rd[1], -rd[2])              # toward the camera
    # Flip the normal toward the viewer (stable shading regardless of side).
    if wp.dot(n, v) < 0.0:
        n = wp.vec3(-n[0], -n[1], -n[2])

    # Three-light diffuse + ambient (world space, z up), gentle white specular.
    l0 = wp.normalize(wp.vec3(0.4, 0.5, 0.85))       # key (from above)
    l1 = wp.normalize(wp.vec3(-0.6, -0.35, 0.45))    # fill
    l2 = wp.normalize(wp.vec3(0.15, -0.85, 0.35))    # rim
    diff = 0.34                                       # ambient
    diff = diff + wp.max(wp.dot(n, l0), 0.0) * 0.7
    diff = diff + wp.max(wp.dot(n, l1), 0.0) * 0.35
    diff = diff + wp.max(wp.dot(n, l2), 0.0) * 0.25
    rgb = base * wp.min(diff, 1.25)

    hdir = wp.normalize(l0 + v)
    spec = wp.pow(wp.max(wp.dot(n, hdir), 0.0), 48.0) * 0.5
    rgb = rgb + wp.vec3(spec, spec, spec)

    rgba[o + 0] = wp.uint8(wp.clamp(rgb[0] * 255.0, 0.0, 255.0))
    rgba[o + 1] = wp.uint8(wp.clamp(rgb[1] * 255.0, 0.0, 255.0))
    rgba[o + 2] = wp.uint8(wp.clamp(rgb[2] * 255.0, 0.0, 255.0))
    rgba[o + 3] = wp.uint8(255)


def scene_bounding_sphere(centers: np.ndarray, radii: np.ndarray):
    """(center, radius) world bounding sphere enclosing every ellipsoid.

    Each ellipsoid is bounded by a sphere of its largest semi-axis; the union is
    bounded by a sphere through the farthest such extent from the centroid.
    """
    c = np.ascontiguousarray(centers, dtype=np.float64)
    r = np.abs(np.ascontiguousarray(radii, dtype=np.float64))
    rmax = r.max(axis=1)                                  # per-ellipsoid radius
    mid = c.mean(axis=0)
    reach = np.linalg.norm(c - mid, axis=1) + rmax
    return mid.astype(np.float32), float(reach.max())


class Raymarcher:
    """Stateful sphere-tracer that caches the ellipsoid population on the device.

    The population (centers / radii / rotations) is uploaded once via
    :meth:`update` and reused across frames, so an interactive camera orbit only
    pays for the kernel + readback — not a fresh host->device upload and the
    bounding-sphere / size derivation every frame.  Call :meth:`update` whenever
    the ellipsoids change (e.g. each fit step), then :meth:`render` per frame.
    """

    def __init__(self):
        self._dev = best_device()
        self._n = 0
        self._wc = None
        self._wr = None
        self._wq = None
        self._wrmax = None
        self._sph_c = (0.0, 0.0, 0.0)
        self._sph_r = 1.0
        self._char = 1.0
        # Reused output buffer, re-allocated only when the pixel count grows.
        self._rgba = None
        self._rgba_cap = 0

    def update(self, centers, radii, rotations) -> None:
        """Upload a new ellipsoid population to the device (call on change)."""
        c = np.ascontiguousarray(centers, np.float32) if centers is not None \
            else np.zeros((0, 3), np.float32)
        self._n = int(c.shape[0])
        if self._n == 0:
            return
        r = np.ascontiguousarray(np.abs(radii), np.float32)
        rmax = np.ascontiguousarray(r.max(axis=1))
        self._wc = wp.array(c, dtype=wp.vec3, device=self._dev)
        self._wr = wp.array(r, dtype=wp.vec3, device=self._dev)
        self._wq = wp.array(np.ascontiguousarray(rotations, np.float32).reshape(-1),
                            dtype=wp.float32, device=self._dev)
        self._wrmax = wp.array(rmax, dtype=wp.float32, device=self._dev)
        self._sph_c, self._sph_r = scene_bounding_sphere(c, r)
        self._char = float(np.median(rmax))

    def _out(self, m: int):
        if self._rgba is None or self._rgba_cap < m:
            self._rgba = wp.empty(m, dtype=wp.uint8, device=self._dev)
            self._rgba_cap = m
        return self._rgba

    def render(
        self,
        inv_mvp: np.ndarray,                # (4,4) row-major NDC->world
        W: int, H: int,
        base_rgb,                           # (3,) 0..1 surface colour
        bg_rgb=(0.0, 0.0, 0.0),             # (3,) 0..1 background colour
        bg_alpha: float = 0.0,              # 0 => transparent (over GL)
        blend: float = 0.0,                 # smooth-union amount (fraction)
        max_steps: int = 160,
        relax: float = 0.75,
    ) -> np.ndarray:
        """Render the cached population → ``(H, W, 4)`` uint8 (top row first)."""
        W = max(int(W), 1)
        H = max(int(H), 1)
        M = W * H
        rgba = self._out(M * 4)
        bg = tuple(float(x) for x in bg_rgb)
        if self._n == 0:
            wp.launch(_fill_bg_kernel, dim=M,
                      inputs=[wp.vec3(bg[0], bg[1], bg[2]), float(bg_alpha), rgba],
                      device=self._dev)
            return rgba.numpy()[:M * 4].reshape(H, W, 4)

        surf_eps = max(self._sph_r * 1.5e-3, 1e-6)
        normal_h = max(self._sph_r * 1.0e-3, 1e-6)
        blend_world = max(float(blend), 0.0) * self._char
        m = np.ascontiguousarray(inv_mvp, dtype=np.float32).reshape(-1)
        wm = wp.mat44(
            float(m[0]), float(m[1]), float(m[2]), float(m[3]),
            float(m[4]), float(m[5]), float(m[6]), float(m[7]),
            float(m[8]), float(m[9]), float(m[10]), float(m[11]),
            float(m[12]), float(m[13]), float(m[14]), float(m[15]),
        )
        base = tuple(float(x) for x in base_rgb)
        wp.launch(
            _raymarch_kernel, dim=M,
            inputs=[
                self._wc, self._wr, self._wrmax, self._wq, self._n, wm, W, H,
                wp.vec3(float(self._sph_c[0]), float(self._sph_c[1]), float(self._sph_c[2])),
                float(self._sph_r),
                float(surf_eps), float(normal_h), float(relax), int(max_steps),
                float(blend_world),
                wp.vec3(base[0], base[1], base[2]),
                wp.vec3(bg[0], bg[1], bg[2]), float(bg_alpha),
                rgba,
            ],
            device=self._dev,
        )
        return rgba.numpy()[:M * 4].reshape(H, W, 4)


def render(centers, radii, rotations, inv_mvp, W, H, base_rgb,
           bg_rgb=(0.0, 0.0, 0.0), bg_alpha=0.0, blend=0.0,
           max_steps=160, relax=0.75) -> np.ndarray:
    """One-shot convenience wrapper (uploads + renders a single frame)."""
    rm = Raymarcher()
    rm.update(centers, radii, rotations)
    return rm.render(inv_mvp, W, H, base_rgb, bg_rgb, bg_alpha, blend,
                     max_steps, relax)


@wp.kernel
def _fill_bg_kernel(bg: wp.vec3, bg_a: float, rgba: wp.array(dtype=wp.uint8)):
    tid = wp.tid()
    o = tid * 4
    rgba[o + 0] = wp.uint8(wp.clamp(bg[0] * 255.0, 0.0, 255.0))
    rgba[o + 1] = wp.uint8(wp.clamp(bg[1] * 255.0, 0.0, 255.0))
    rgba[o + 2] = wp.uint8(wp.clamp(bg[2] * 255.0, 0.0, 255.0))
    rgba[o + 3] = wp.uint8(wp.clamp(bg_a * 255.0, 0.0, 255.0))
