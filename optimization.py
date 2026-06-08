"""
optimization.py — Ellipsoid fitting via differentiable SDF.

MINI-BATCH + CONSERVATIVE POPULATION MANAGEMENT:
  - Epoch-based mini-batch sampling (index indirection).
  - Periodic maintenance (every `maintenance_every` iterations):
      1. PRUNE (conservative) — only remove ellipsoids that are truly
         redundant: either (a) degenerate (collapsed to near-zero volume)
         or (b) *contained* inside a larger ellipsoid (center is inside
         the larger one AND all radii are smaller).
         → Large ellipsoids are never removed in favour of small ones.
         → A budget cap limits removals to `max_prune_fraction` of the
           population per round so training stays stable.
      2. SPAWN — fill vacancies at high-error regions via farthest-point
         sampling (same as before).
"""

import contextlib

import warp as wp
import warp.optim
import numpy as np

from PySide6 import QtCore

from ellipsoid import Ellipsoid, EllipsoidSet, best_device
from underrepresentation import relative_underrep_grid, compute_relative_underrep
from thickness import dilate_zeros


# ── Warp kernels ──────────────────────────────────────────────────────────────

@wp.kernel
def _ellipsoid_sdf_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    num_ellipsoids: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
):
    bid = wp.tid()
    tid = indices[bid]

    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)

    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )

    min_d[bid, 0] = 1.0e6

    for i in range(num_ellipsoids):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0],
            rot_flat[base + 1],
            rot_flat[base + 2],
            rot_flat[base + 3],
        ))
        local_p = wp.quat_rotate_inv(q, p - centers[i])
        r = radii[i]

        scaled = wp.vec3(
            local_p[0] / r[0],
            local_p[1] / r[1],
            local_p[2] / r[2],
        )

        k0 = wp.length(scaled)

        # ── MertStein hybrid: Quílez outside, Scaled-Sphere (min r) inside ──
        d = float(1.0e6)
        if k0 < 1.0:
            # Interior → (k0 − 1) · min(r)
            r_min = wp.min(wp.min(r[0], r[1]), r[2])
            d = (k0 - 1.0) * r_min
        else:
            # Exterior → Quílez: k0·(k0−1) / k1
            scaled2 = wp.vec3(
                local_p[0] / (r[0] * r[0]),
                local_p[1] / (r[1] * r[1]),
                local_p[2] / (r[2] * r[2]),
            )
            k1 = wp.length(scaled2)
            k1_safe = wp.max(k1, 1.0e-8)
            d = k0 * (k0 - 1.0) / k1_safe

        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)

    out_sdf[bid] = min_d[bid, num_ellipsoids]


@wp.kernel
def _ellipsoid_sdf_kernel_points(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    num_ellipsoids: int,
    points: wp.array(dtype=wp.vec3),          # pre-computed world sample points
    indices: wp.array(dtype=wp.int32),        # batch indices into ``points``
    out_sdf: wp.array(dtype=wp.float32),
):
    # Same hard-min MertStein union as ``_ellipsoid_sdf_kernel_batch`` but the
    # sample position comes straight from a world-point array instead of being
    # decoded from a single grid.  This lets one kernel evaluate samples drawn
    # from MANY different region boxes (each with its own origin/dx) in a single
    # launch — the boxes' geometry is already baked into ``points``.
    bid = wp.tid()
    p = points[indices[bid]]

    min_d[bid, 0] = 1.0e6
    for i in range(num_ellipsoids):
        base = i * 4
        q = wp.normalize(wp.quat(
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
            k1_safe = wp.max(wp.length(scaled2), 1.0e-8)
            d = k0 * (k0 - 1.0) / k1_safe
        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)

    out_sdf[bid] = min_d[bid, num_ellipsoids]


@wp.kernel
def _ellipsoid_softmin_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    m_cache: wp.array2d(dtype=wp.float32),   # running min   (scan, batch × N+1)
    s_cache: wp.array2d(dtype=wp.float32),   # running LSE sum (scan, batch × N+1)
    num_ellipsoids: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
    k: float,
):
    # SMOOTH union of the per-ellipsoid SDFs via a numerically-stable, online
    # LogSumExp:  softmin = m − log(Σ exp(−k·(dᵢ − m))) / k,  with m = min dᵢ.
    # As k→∞ this → hard min; for finite k it blends several near-minimum
    # ellipsoids so the gradient is SHARED among them (instead of only the single
    # nearest), giving much denser gradients → faster, more complete coverage.
    # Stored as an array scan (one write per slot) so Warp can backprop it; the
    # online rescale keeps every exp() argument ≤ 0 (no overflow).
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx, (float(iy) + 0.5) * dx, (float(iz) + 0.5) * dx)

    m_cache[bid, 0] = 1.0e6
    s_cache[bid, 0] = 0.0

    for i in range(num_ellipsoids):
        base = i * 4
        q = wp.normalize(wp.quat(
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

        m_prev = m_cache[bid, i]
        s_prev = s_cache[bid, i]
        if d < m_prev:
            # new running min: rescale the old sum to the new pivot, add 1
            m_cache[bid, i + 1] = d
            s_cache[bid, i + 1] = s_prev * wp.exp(-k * (m_prev - d)) + 1.0
        else:
            m_cache[bid, i + 1] = m_prev
            s_cache[bid, i + 1] = s_prev + wp.exp(-k * (d - m_prev))

    m = m_cache[bid, num_ellipsoids]
    s = s_cache[bid, num_ellipsoids]
    out_sdf[bid] = m - wp.log(s) / k


@wp.func
def soft_clamp(x: float, limit: float) -> float:
    return limit * wp.tanh(x / limit)


@wp.kernel
def _rmse_loss_kernel_batch(
    sdf_pred: wp.array(dtype=wp.float32),
    sdf_target: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    loss: wp.array(dtype=wp.float32),
    batch_size: int,
    miss_weight: float,
    surface_weight: float,
    surface_sigma: float,
    outside_weight: float,
    thickness: wp.array(dtype=wp.float32),
    thick_ref: float,
    thin_weight: float,
    thin_max_factor: float,
):
    bid = wp.tid()
    tid = indices[bid]
    limit = float(0.1)

    # Bound the predicted SDF to a physically sane range.  The miss / protrusion
    # penalties below use the RAW prediction (not the soft-clamped base term), so
    # a single transiently-diverged primitive (huge predicted SDF) would
    # otherwise blow the loss up to ~1e11.  A normalised mesh's true SDF is far
    # below this cap, so legitimate gradients are untouched while outliers can no
    # longer dominate.
    sp = wp.clamp(sdf_pred[bid], -10.0, 10.0)

    # Surface emphasis: Gaussian bump centred on the zero level set so the
    # ellipsoid surface is pulled onto the mesh surface far more strongly
    # than interior bulk. surface_sigma is in world units (~few voxels).
    t = sdf_target[tid]
    sw = float(1.0) + surface_weight * wp.exp(-(t * t) / (surface_sigma * surface_sigma))

    # Thin-feature emphasis: scale by the *inverse* local feature thickness so a
    # deviation on a thin structure grows the loss faster than the same deviation
    # on a thick limb. thick_ref ≈ median feature thickness → thick regions get
    # tw ≈ 1, thin regions are boosted up to thin_max_factor.
    tw = float(1.0)
    th = thickness[tid]
    if thin_weight > float(0.0) and th > float(0.0):
        boost = thick_ref / th - float(1.0)
        if boost < float(0.0):
            boost = float(0.0)
        tw = float(1.0) + thin_weight * boost
        if tw > thin_max_factor:
            tw = thin_max_factor
    w = sw * tw

    # Base SDF reconstruction loss (surface- + thinness-weighted)
    diff = wp.abs(soft_clamp(sp, limit) - soft_clamp(t, limit))
    wp.atomic_add(loss, 0, w * diff / float(batch_size))

    # Miss penalty: target inside mesh but ellipsoid says outside
    if t < float(0.0) and sp > float(0.0):
        miss = sp - t
        wp.atomic_add(loss, 0, w * miss_weight * miss / float(batch_size))

    # Protrusion penalty: target OUTSIDE the mesh but ellipsoid covers it,
    # i.e. the ellipsoid sticks out past the true surface. Penalised strongly
    # so primitives stay inside the mesh rather than bulging out.
    if t > float(0.0) and sp < float(0.0):
        over = t - sp
        wp.atomic_add(loss, 0, w * outside_weight * over / float(batch_size))


@wp.kernel
def _flatness_penalty_kernel(
    radii: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
    count: int,
    offset: int,
    flat_weight: float,
    flat_min_ratio: float,
):
    # Shape regulariser: penalise ellipsoids that have collapsed into thin flat
    # *disks* (smallest axis ≪ the other two) while still ALLOWING slender
    # *needles* (one long axis, two short-but-equal axes) — those are exactly
    # what delicate thin features need.  Key: compare the smallest axis to the
    # MEDIAN axis, not the mean.  A disk has rmin ≪ rmid → penalised; a needle
    # has rmin ≈ rmid (both short) → ratio ≈ 1 → free.  Scale-invariant.
    e = offset + wp.tid()
    r = radii[e]
    a = r[0]
    b = r[1]
    c = r[2]
    rmax = wp.max(a, wp.max(b, c))
    rmin = wp.min(a, wp.min(b, c))
    rmid = wp.max(a + b + c - rmax - rmin, 1.0e-9)
    pen = float(0.0)
    d = flat_min_ratio - rmin / rmid
    if d > float(0.0):
        pen = d * d
    wp.atomic_add(loss, 0, flat_weight * pen / float(count))


@wp.kernel
def _bone_membership_kernel(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    bone_centers: wp.array(dtype=wp.vec3),
    num_bones: int,
    soft: float,
    soft_count: wp.array(dtype=wp.float32),   # per-ellipsoid accumulator
):
    # One thread per (ellipsoid, bone) pair.  Adds a smooth membership (≈1 well
    # inside the ellipsoid, ≈0 outside) to the ellipsoid's accumulator.  Using a
    # *single* atomic-add into an array (instead of a mutated scalar in a dynamic
    # for-loop) keeps the operation differentiable — Warp cannot backprop through
    # a scalar accumulated across a runtime-bounded loop.
    gid = wp.tid()
    e = gid // num_bones
    b = gid % num_bones
    base = e * 4
    q = wp.normalize(wp.quat(
        rot_flat[base + 0], rot_flat[base + 1],
        rot_flat[base + 2], rot_flat[base + 3]))
    c = centers[e]
    r = radii[e]
    rx = wp.max(r[0], 1.0e-6)
    ry = wp.max(r[1], 1.0e-6)
    rz = wp.max(r[2], 1.0e-6)
    local_p = wp.quat_rotate_inv(q, bone_centers[b] - c)
    sx = local_p[0] / rx
    sy = local_p[1] / ry
    sz = local_p[2] / rz
    k0 = wp.sqrt(sx * sx + sy * sy + sz * sz + 1.0e-12)
    # tanh form of the logistic — no exp overflow for far bones.
    m = 0.5 * (1.0 - wp.tanh((k0 - 1.0) / (2.0 * soft)))
    wp.atomic_add(soft_count, e, m)


@wp.kernel
def _bone_penalty_kernel(
    soft_count: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    count: int,
    weight: float,
    tol: float,
):
    # Penalise enclosing more than ~1 bone: the soft count's excess over
    # (1 + tol) is squared.  Covering one bone (plus a little slack) is free;
    # spanning two or more is pushed back.  Differentiable through soft_count.
    e = wp.tid()
    excess = soft_count[e] - 1.0 - tol
    pen = float(0.0)
    if excess > 0.0:
        pen = excess * excess
    wp.atomic_add(loss, 0, weight * pen / float(count))


@wp.kernel
def _containment_penalty_kernel(
    centers: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.float32),   # flat (nz·ny·nx) mesh SDF
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    loss: wp.array(dtype=wp.float32),
    count: int,
    weight: float,
):
    # Per-ellipsoid containment: penalise an ellipsoid whose CENTRE lies outside
    # the mesh (target SDF > 0).  The target is read by *trilinear interpolation*
    # so the penalty is differentiable in the centre — the gradient points back
    # toward the surface and pulls a wandering ellipsoid inside.  Quadratic, so a
    # centre that drifts far out is punished hard.
    e = wp.tid()
    p = centers[e]
    gx = (p[0] - origin[0]) / dx - 0.5
    gy = (p[1] - origin[1]) / dx - 0.5
    gz = (p[2] - origin[2]) / dx - 0.5

    ix0 = int(wp.floor(gx))
    iy0 = int(wp.floor(gy))
    iz0 = int(wp.floor(gz))
    # Clamp the base corner so all 8 neighbours are in-bounds (a centre beyond
    # the box extrapolates the edge gradient, which still points inward).
    ix0 = wp.clamp(ix0, 0, nx - 2)
    iy0 = wp.clamp(iy0, 0, ny - 2)
    iz0 = wp.clamp(iz0, 0, nz - 2)

    # Clamp the interpolation weights to [0,1] (pure interpolation, no
    # extrapolation): a centre far outside the grid would otherwise extrapolate
    # the trilinear field to a huge value → t² blows the loss up.  Centres are
    # kept near the grid by the per-step clamp, so the inward-pull gradient is
    # unaffected for in-grid centres.
    fx = wp.clamp(gx - float(ix0), 0.0, 1.0)
    fy = wp.clamp(gy - float(iy0), 0.0, 1.0)
    fz = wp.clamp(gz - float(iz0), 0.0, 1.0)

    nynx = ny * nx
    b = iz0 * nynx + iy0 * nx + ix0
    c000 = target[b]
    c100 = target[b + 1]
    c010 = target[b + nx]
    c110 = target[b + nx + 1]
    c001 = target[b + nynx]
    c101 = target[b + nynx + 1]
    c011 = target[b + nynx + nx]
    c111 = target[b + nynx + nx + 1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    t = c0 * (1.0 - fz) + c1 * fz

    if t > 0.0:
        wp.atomic_add(loss, 0, weight * t * t / float(count))


@wp.kernel
def _exp_radii_kernel(
    log_radii: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
):
    # World radii from the trainable log-radii: r = exp(log_r).  Run *inside* the
    # tape so the gradient flows back to log-space (dr/d log_r = r), giving the
    # optimiser scale-invariant, always-positive radius updates.
    e = wp.tid()
    s = log_radii[e]
    radii[e] = wp.vec3(wp.exp(s[0]), wp.exp(s[1]), wp.exp(s[2]))


@wp.kernel
def _sgd_step_vec3(
    param: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    lr: float,
):
    tid = wp.tid()
    param[tid] = param[tid] - lr * grad[tid]


@wp.kernel
def _sgd_step_f32(
    param: wp.array(dtype=wp.float32),
    grad: wp.array(dtype=wp.float32),
    lr: float,
):
    tid = wp.tid()
    param[tid] = param[tid] - lr * grad[tid]


@wp.kernel
def _normalize_flat_quats(
    rot_flat: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    base = tid * 4
    x = rot_flat[base + 0]
    y = rot_flat[base + 1]
    z = rot_flat[base + 2]
    w = rot_flat[base + 3]
    inv_len = 1.0 / wp.max(wp.sqrt(x * x + y * y + z * z + w * w), 1.0e-12)
    rot_flat[base + 0] = x * inv_len
    rot_flat[base + 1] = y * inv_len
    rot_flat[base + 2] = z * inv_len
    rot_flat[base + 3] = w * inv_len


# ── Sphere constraint + dedicated sphere kernels ────────────────────────────────
# A sphere is fitted with a single radius and no rotation.  The dedicated kernels
# below evaluate the EXACT, cheap union SDF  min_i (|p − c_i| − r_i)  — no
# quaternion, no per-axis scaling, no Quílez branch — which is the per-step hot
# loop, so this is a real speedup over re-using the ellipsoid kernel.  The radius
# is ``radii[i][0]``; the projection keeps the other two components equal to it
# and the rotation at identity so rendering / maintenance / exports stay sane.

@wp.kernel
def _broadcast_log_radii(log_radii: wp.array(dtype=wp.vec3)):
    # Keep the (log-)radii isotropic by copying component 0 to all three.  Only
    # component 0 is fed to the sphere kernel, so broadcasting (not averaging)
    # preserves the full radius gradient step.
    i = wp.tid()
    s = log_radii[i]
    log_radii[i] = wp.vec3(s[0], s[0], s[0])


@wp.kernel
def _reset_rot_identity(rot_flat: wp.array(dtype=wp.float32)):
    i = wp.tid()
    base = i * 4
    rot_flat[base + 0] = 0.0
    rot_flat[base + 1] = 0.0
    rot_flat[base + 2] = 0.0
    rot_flat[base + 3] = 1.0


@wp.kernel
def _sphere_sdf_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    min_d: wp.array2d(dtype=wp.float32),
    num_spheres: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
):
    # Exact hard-min sphere union SDF (grid-sampled).  Mirrors
    # ``_ellipsoid_sdf_kernel_batch`` (same scan buffer for backprop) but reduces
    # each primitive to |p − c| − r.
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )
    min_d[bid, 0] = 1.0e6
    for i in range(num_spheres):
        r = radii[i]
        d = wp.length(p - centers[i]) - r[0]
        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)
    out_sdf[bid] = min_d[bid, num_spheres]


@wp.kernel
def _sphere_sdf_kernel_points(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    min_d: wp.array2d(dtype=wp.float32),
    num_spheres: int,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
):
    # Sphere union SDF sampled at pre-computed world points (local-fit path).
    bid = wp.tid()
    p = points[indices[bid]]
    min_d[bid, 0] = 1.0e6
    for i in range(num_spheres):
        r = radii[i]
        d = wp.length(p - centers[i]) - r[0]
        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)
    out_sdf[bid] = min_d[bid, num_spheres]


@wp.kernel
def _sphere_softmin_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    m_cache: wp.array2d(dtype=wp.float32),
    s_cache: wp.array2d(dtype=wp.float32),
    num_spheres: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
    k: float,
):
    # Smooth (LogSumExp) sphere union — same online-rescaled scan as
    # ``_ellipsoid_softmin_kernel_batch`` but with the exact sphere distance.
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx, (float(iy) + 0.5) * dx, (float(iz) + 0.5) * dx)
    m_cache[bid, 0] = 1.0e6
    s_cache[bid, 0] = 0.0
    for i in range(num_spheres):
        r = radii[i]
        d = wp.length(p - centers[i]) - r[0]
        m_prev = m_cache[bid, i]
        s_prev = s_cache[bid, i]
        if d < m_prev:
            m_cache[bid, i + 1] = d
            s_cache[bid, i + 1] = s_prev * wp.exp(-k * (m_prev - d)) + 1.0
        else:
            m_cache[bid, i + 1] = m_prev
            s_cache[bid, i + 1] = s_prev + wp.exp(-k * (d - m_prev))
    m = m_cache[bid, num_spheres]
    s = s_cache[bid, num_spheres]
    out_sdf[bid] = m - wp.log(s) / k


# ── Superquadric SDF (PER-PRIMITIVE roundness exponents) ────────────────────────
# A superquadric generalises the ellipsoid with two roundness exponents:
#   F = (|x/a|^(2/e2) + |y/b|^(2/e2))^(e2/e1) + |z/c|^(2/e1),   surface at F = 1.
# beta = F^(e1/2) is linear along rays (beta = 1 on the surface), the analogue of
# the ellipsoid's k0.  We reuse the same inside/outside hybrid as the ellipsoid
# (scaled-sphere inside, radial distance outside), so for e1 = e2 = 1 this reduces
# to the ellipsoid case and for a sphere it is exact.  e1, e2 are stored PER
# PRIMITIVE in an ``eps`` array (vec2 = (e1, e2)) and are trained like the other
# parameters — the gradient flows to centres / radii / rotation AND eps.

@wp.func
def _sq_shape_distance(lp: wp.vec3, r: wp.vec3, e1: float, e2: float) -> float:
    # Superquadric (pseudo-)distance for a point already in the local frame.
    # The pow bases are CLAMPED (≤ cap): with e ≥ 0.1 the exponent 2/e reaches 20,
    # so an un-clamped base > ~60 overflows float32 — the forward value stays
    # finite (the outside branch saturates) but the gradient through the
    # overflowing pow becomes inf/NaN and corrupts the parameters.  Clamping the
    # base keeps the gradient finite for far / thin primitives (it also guards
    # against a near-zero radius).
    eps = float(1.0e-9)
    cap = float(50.0)
    big = float(1.0e36)
    ax = wp.pow(wp.min(wp.abs(lp[0] / r[0]) + eps, cap), 2.0 / e2)
    ay = wp.pow(wp.min(wp.abs(lp[1] / r[1]) + eps, cap), 2.0 / e2)
    az = wp.pow(wp.min(wp.abs(lp[2] / r[2]) + eps, cap), 2.0 / e1)
    # The SECOND pow can overflow even when the bases above are clamped: when e2
    # is large (so ax+ay is moderate) and e1 is small, the exponent e2/e1 reaches
    # ~20 and (ax+ay)^20 overflows float32 → inf → NaN gradient.  Clamp it so the
    # value (and its gradient) stay finite.
    f = wp.min(wp.pow(ax + ay, e2 / e1), big) + az
    beta = wp.min(wp.pow(f, e1 * 0.5), big)
    pmag = wp.length(lp)
    rmin = wp.min(wp.min(r[0], r[1]), r[2])
    d = float(0.0)
    if beta < 1.0:
        d = (beta - 1.0) * rmin                       # inside: scaled-sphere
    else:
        d = pmag * (beta - 1.0) / wp.max(beta, eps)   # outside: radial distance
    return d


@wp.func
def _sq_distance(p: wp.vec3, c: wp.vec3, r: wp.vec3, q: wp.quat,
                 e1: float, e2: float) -> float:
    lp = wp.quat_rotate_inv(q, p - c)
    return _sq_shape_distance(lp, r, e1, e2)


@wp.func
def _bent_sq_distance(p: wp.vec3, c: wp.vec3, r: wp.vec3, q: wp.quat,
                      e1: float, e2: float, kx: float, ky: float) -> float:
    # Bent superquadric.  A simple, numerically robust quadratic bend warps the
    # local x/y by −½·k·z² (Barr-style global deformation); the inverse warp maps
    # the query point back into the straight superquadric, where we evaluate the
    # base distance, then divide by the local z-stretch to keep it ~metric.
    # kx = ky = 0 reduces EXACTLY to the plain superquadric.
    lp = wp.quat_rotate_inv(q, p - c)
    z = lp[2]
    ulp = wp.vec3(lp[0] - 0.5 * kx * z * z,
                  lp[1] - 0.5 * ky * z * z,
                  z)
    d = _sq_shape_distance(ulp, r, e1, e2)
    denom = wp.sqrt(1.0 + (kx * z) * (kx * z) + (ky * z) * (ky * z))
    return d / denom


@wp.kernel
def _clamp_eps(eps: wp.array(dtype=wp.float32), lo: float, hi: float):
    # Keep per-primitive roundness in a numerically safe range after each step.
    # ``eps`` is flat float32 (2 per primitive) so the Warp Adam optimiser
    # accepts it (it does not support vec2).
    i = wp.tid()
    eps[i] = wp.clamp(eps[i], lo, hi)


@wp.kernel
def _clamp_log_radii(log_radii: wp.array(dtype=wp.vec3), lo: float, hi: float):
    # Safety net for the global loop: bound the (log-)radii so a runaway Adam
    # step can't grow a radius to inf (clamp turns inf → hi).  Generous bounds,
    # so it never interferes with a legitimate fit.
    i = wp.tid()
    s = log_radii[i]
    log_radii[i] = wp.vec3(wp.clamp(s[0], lo, hi),
                           wp.clamp(s[1], lo, hi),
                           wp.clamp(s[2], lo, hi))


@wp.kernel
def _superquadric_sdf_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    eps: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    num_e: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
):
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx, (float(iy) + 0.5) * dx, (float(iz) + 0.5) * dx)
    min_d[bid, 0] = 1.0e6
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        be = i * 2
        d = _bent_sq_distance(p, centers[i], radii[i], q,
                              eps[be], eps[be + 1], bend[be], bend[be + 1])
        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)
    out_sdf[bid] = min_d[bid, num_e]


@wp.kernel
def _superquadric_sdf_kernel_points(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    eps: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    num_e: int,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
):
    bid = wp.tid()
    p = points[indices[bid]]
    min_d[bid, 0] = 1.0e6
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        be = i * 2
        d = _bent_sq_distance(p, centers[i], radii[i], q,
                              eps[be], eps[be + 1], bend[be], bend[be + 1])
        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)
    out_sdf[bid] = min_d[bid, num_e]


@wp.kernel
def _superquadric_softmin_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    eps: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    m_cache: wp.array2d(dtype=wp.float32),
    s_cache: wp.array2d(dtype=wp.float32),
    num_e: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
    k: float,
):
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx, (float(iy) + 0.5) * dx, (float(iz) + 0.5) * dx)
    m_cache[bid, 0] = 1.0e6
    s_cache[bid, 0] = 0.0
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        be = i * 2
        d = _bent_sq_distance(p, centers[i], radii[i], q,
                              eps[be], eps[be + 1], bend[be], bend[be + 1])
        m_prev = m_cache[bid, i]
        s_prev = s_cache[bid, i]
        if d < m_prev:
            m_cache[bid, i + 1] = d
            s_cache[bid, i + 1] = s_prev * wp.exp(-k * (m_prev - d)) + 1.0
        else:
            m_cache[bid, i + 1] = m_prev
            s_cache[bid, i + 1] = s_prev + wp.exp(-k * (d - m_prev))
    m = m_cache[bid, num_e]
    s = s_cache[bid, num_e]
    out_sdf[bid] = m - wp.log(s) / k


# ── Capsule SDF (exact) ─────────────────────────────────────────────────────────
# A capsule is a line segment (along the local z-axis) swept by a sphere.  It maps
# onto the existing parameters: radius = r[0], half-length = r[2] (r[1] is kept
# equal to r[0] by a projection so the cross-section stays circular).  The SDF is
# exact and cheap — distance to the segment minus the radius — and the kernels
# share the ellipsoid signature (centres / radii / rot_flat), so dispatch is a
# straight kernel swap.

@wp.func
def _capsule_distance(p: wp.vec3, c: wp.vec3, r: wp.vec3, q: wp.quat) -> float:
    lp = wp.quat_rotate_inv(q, p - c)
    rad = r[0]
    h = r[2]
    qz = wp.clamp(lp[2], -h, h)
    dvec = wp.vec3(lp[0], lp[1], lp[2] - qz)
    return wp.length(dvec) - rad


@wp.kernel
def _capsule_eq_radii(radii: wp.array(dtype=wp.vec3)):
    # Keep the cross-section circular: r[1] = r[0].  Works on log-radii or world
    # radii (it only copies component 0 into component 1, leaving the length r[2]).
    i = wp.tid()
    s = radii[i]
    radii[i] = wp.vec3(s[0], s[0], s[2])


@wp.kernel
def _capsule_sdf_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    num_e: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
):
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx, (float(iy) + 0.5) * dx, (float(iz) + 0.5) * dx)
    min_d[bid, 0] = 1.0e6
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        d = _capsule_distance(p, centers[i], radii[i], q)
        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)
    out_sdf[bid] = min_d[bid, num_e]


@wp.kernel
def _capsule_sdf_kernel_points(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    num_e: int,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
):
    bid = wp.tid()
    p = points[indices[bid]]
    min_d[bid, 0] = 1.0e6
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        d = _capsule_distance(p, centers[i], radii[i], q)
        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)
    out_sdf[bid] = min_d[bid, num_e]


@wp.kernel
def _capsule_softmin_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    m_cache: wp.array2d(dtype=wp.float32),
    s_cache: wp.array2d(dtype=wp.float32),
    num_e: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
    k: float,
):
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx, (float(iy) + 0.5) * dx, (float(iz) + 0.5) * dx)
    m_cache[bid, 0] = 1.0e6
    s_cache[bid, 0] = 0.0
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        d = _capsule_distance(p, centers[i], radii[i], q)
        m_prev = m_cache[bid, i]
        s_prev = s_cache[bid, i]
        if d < m_prev:
            m_cache[bid, i + 1] = d
            s_cache[bid, i + 1] = s_prev * wp.exp(-k * (m_prev - d)) + 1.0
        else:
            m_cache[bid, i + 1] = m_prev
            s_cache[bid, i + 1] = s_prev + wp.exp(-k * (d - m_prev))
    m = m_cache[bid, num_e]
    s = s_cache[bid, num_e]
    out_sdf[bid] = m - wp.log(s) / k


# ── Range-restricted SGD (SuperFit isolated local fitting) ──────────────────────
# These update only the contiguous ellipsoid range [offset:], freezing every
# ellipsoid before `offset`.  Used when newly-spawned ellipsoids (always appended
# at the END of the population) are fitted in isolation to a residual region.

@wp.kernel
def _sgd_step_vec3_range(
    param: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    lr: float,
    offset: int,
):
    tid = wp.tid()
    i = offset + tid
    param[i] = param[i] - lr * grad[i]


@wp.kernel
def _sgd_step_f32_range(
    param: wp.array(dtype=wp.float32),
    grad: wp.array(dtype=wp.float32),
    lr: float,
    offset: int,
):
    # offset is in float32 elements (= ellipsoid_offset * 4 for rot_flat)
    tid = wp.tid()
    i = offset + tid
    param[i] = param[i] - lr * grad[i]


@wp.kernel
def _normalize_flat_quats_range(
    rot_flat: wp.array(dtype=wp.float32),
    offset: int,
):
    # offset is in ellipsoids; normalise quats for ellipsoids [offset:]
    tid = wp.tid()
    base = (offset + tid) * 4
    x = rot_flat[base + 0]
    y = rot_flat[base + 1]
    z = rot_flat[base + 2]
    w = rot_flat[base + 3]
    inv_len = 1.0 / wp.max(wp.sqrt(x * x + y * y + z * z + w * w), 1.0e-12)
    rot_flat[base + 0] = x * inv_len
    rot_flat[base + 1] = y * inv_len
    rot_flat[base + 2] = z * inv_len
    rot_flat[base + 3] = w * inv_len


@wp.kernel
def _clamp_radii_range(
    radii: wp.array(dtype=wp.vec3),
    rmin: float,
    rmax: float,
    offset: int,
):
    # Keep radii positive and bounded for ellipsoids [offset:].  Plain SGD on the
    # MertStein outside term can blow a radius up (sample points far outside a tiny
    # seed give huge outward gradients) or drive it negative; clamp every step.
    tid = wp.tid()
    i = offset + tid
    r = radii[i]
    rx = wp.clamp(wp.abs(r[0]), rmin, rmax)
    ry = wp.clamp(wp.abs(r[1]), rmin, rmax)
    rz = wp.clamp(wp.abs(r[2]), rmin, rmax)
    radii[i] = wp.vec3(rx, ry, rz)


@wp.kernel
def _clamp_centers_range(
    centers: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    hi: wp.vec3,
    offset: int,
):
    # Keep trainable centres [offset:] inside the region box so a runaway Adam
    # step cannot fling an ellipsoid out of the box ("gesprengt").
    tid = wp.tid()
    i = offset + tid
    c = centers[i]
    cx = wp.clamp(c[0], lo[0], hi[0])
    cy = wp.clamp(c[1], lo[1], hi[1])
    cz = wp.clamp(c[2], lo[2], hi[2])
    centers[i] = wp.vec3(cx, cy, cz)


@wp.kernel
def _clamp_centers_perbox(
    centers: wp.array(dtype=wp.vec3),
    lo: wp.array(dtype=wp.vec3),       # per-trainable lower bound (its own box)
    hi: wp.array(dtype=wp.vec3),       # per-trainable upper bound (its own box)
    offset: int,
):
    # Like ``_clamp_centers_range`` but each trainable [offset+tid] is clamped to
    # ITS OWN region box — needed when many region boxes are fitted together in a
    # single optimisation, so an ellipsoid cannot drift into a neighbouring box.
    tid = wp.tid()
    i = offset + tid
    c = centers[i]
    l = lo[tid]
    h = hi[tid]
    centers[i] = wp.vec3(
        wp.clamp(c[0], l[0], h[0]),
        wp.clamp(c[1], l[1], h[1]),
        wp.clamp(c[2], l[2], h[2]))


@wp.kernel
def _clamp_log_radii_perbox(
    log_radii: wp.array(dtype=wp.vec3),
    logmin: wp.array(dtype=wp.float32),   # per-trainable log(min radius)
    logmax: wp.array(dtype=wp.float32),   # per-trainable log(half box extent)
    offset: int,
):
    # Bound the LOG-space radii of each trainable to [log r_min, log r_max] of its
    # own box.  Clamping in log space (not linear) keeps the radius update
    # scale-invariant and always positive, matching the global optimiser.
    tid = wp.tid()
    i = offset + tid
    s = log_radii[i]
    lo = logmin[tid]
    hi = logmax[tid]
    log_radii[i] = wp.vec3(
        wp.clamp(s[0], lo, hi),
        wp.clamp(s[1], lo, hi),
        wp.clamp(s[2], lo, hi))


@wp.kernel
def _zero_vec3_prefix(grad: wp.array(dtype=wp.vec3)):
    # Zero the gradient of the frozen contributor prefix [0:offset] so that the
    # Adam optimiser (whose moments stay 0 for these) never moves them.
    tid = wp.tid()
    grad[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _zero_f32_prefix(grad: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    grad[tid] = 0.0


device = best_device()


def _soft_clamp_np(x: np.ndarray, limit: float) -> np.ndarray:
    return limit * np.tanh(x / limit)


# ── Epoch-based index sampler ─────────────────────────────────────────────────

class EpochSampler:
    def __init__(self, total: int, batch_size: int,
                 rng: np.random.Generator | None = None):
        self.total = total
        self.batch_size = batch_size
        self._rng = rng or np.random.default_rng()
        self._indices = np.arange(total, dtype=np.int32)
        self._cursor = total

    def next_batch(self) -> np.ndarray:
        if self._cursor + self.batch_size > self.total:
            self._rng.shuffle(self._indices)
            self._cursor = 0
        batch = self._indices[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return np.ascontiguousarray(batch)


class BandSampler:
    """Resolution-independent importance sampler for SDF fitting.

    The cost per optimisation step is set by ``batch_size`` voxels, *not* by the
    grid resolution.  Each batch mixes a fixed fraction drawn from the **surface
    band** (|target| < ``band``) with the remainder drawn uniformly from the
    whole grid.  Sampling near the surface keeps the fit sharp even at high
    resolution, where surface voxels (∝ n²) would otherwise be swamped by
    interior/exterior bulk (∝ n³) under uniform sampling.

    Drawing with replacement makes each batch O(batch_size) regardless of grid
    size, so cranking the resolution no longer slows the loop down.
    """

    def __init__(self, flat_target: np.ndarray, batch_size: int, band: float,
                 surface_fraction: float, rng: np.random.Generator | None = None,
                 flat_thickness: np.ndarray | None = None, thin_bias: float = 0.0):
        self.batch_size = int(batch_size)
        self._rng = rng or np.random.default_rng()
        self._all = np.arange(flat_target.size, dtype=np.int32)
        self._band = np.where(np.abs(flat_target) < band)[0].astype(np.int32)
        if self._band.size == 0:
            self._band = self._all
        sf = float(np.clip(surface_fraction, 0.0, 1.0))
        self.n_surf = min(int(self.batch_size * sf), self.batch_size)
        self.n_rest = self.batch_size - self.n_surf

        # Thin-feature sampling (adaptive — driven by the thickness field).
        #
        # Delicate parts are a *tiny* fraction of all surface voxels, so a soft
        # ∝1/thickness reweighting can't reliably surface them: features rarer
        # than the reweighting's effective tail simply never land in a batch
        # without an exponent so steep it starves the thick bulk.  Instead we
        # reserve a guaranteed *quota* of the surface-band draws for a dedicated
        # thin pool.
        #
        # What makes it adaptive (not a magic absolute threshold):
        #   * "thin" = band voxels whose local thickness is below half the band
        #     *median* — scale-free, so it follows each mesh's own distribution;
        #   * the quota is *gated* to 0 when no such voxels exist, so a roughly
        #     uniform-thickness mesh is sampled exactly as before (no distortion).
        # ``thin_bias`` scales the quota (1.0 = default 30 %, 0 = off), capped so
        # the thick surface always keeps the majority of the band.
        self._band_thin = None
        self._thin_quota = 0.0
        if flat_thickness is not None and thin_bias > 0.0 and self._band.size > 1:
            th = flat_thickness[self._band].astype(np.float64)
            valid = th > 0.0
            if valid.sum() > 1:
                med = float(np.median(th[valid]))
                thin_mask = valid & (th < 0.5 * med)
                if thin_mask.mean() > 1e-3:          # thin features actually present
                    self._band_thin = self._band[thin_mask]
                    self._thin_quota = float(np.clip(0.3 * thin_bias, 0.0, 0.6))

    def next_batch(self) -> np.ndarray:
        parts = []
        if self.n_surf > 0:
            n_thin = 0
            if self._band_thin is not None and self._thin_quota > 0.0:
                n_thin = int(round(self.n_surf * self._thin_quota))
            n_band = self.n_surf - n_thin
            if n_thin > 0:
                parts.append(self._rng.choice(
                    self._band_thin, size=n_thin, replace=True))
            if n_band > 0:
                parts.append(self._rng.choice(
                    self._band, size=n_band, replace=True))
        if self.n_rest > 0:
            parts.append(self._rng.choice(self._all, size=self.n_rest, replace=True))
        batch = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return np.ascontiguousarray(batch.astype(np.int32))


# ── Quaternion helper (numpy) ─────────────────────────────────────────────────

def _quat_to_rot_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """(4,) quaternion → (3,3) rotation matrix."""
    q = quat_xyzw.astype(np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1 - 2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1 - 2*(x*x+y*y)],
    ], dtype=np.float64)


def _rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → (4,) quaternion (x,y,z,w).  Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return (q / np.linalg.norm(q)).astype(np.float32)


def _mirror_quat(quat_xyzw: np.ndarray, axis: int) -> np.ndarray:
    """Mirror an ellipsoid's orientation across the plane normal to ``axis``.

    Reflection ``M`` (−1 on ``axis``) makes ``M·R`` improper (det −1).  An
    ellipsoid is invariant under flipping the direction of any principal axis, so
    we negate one column to restore a proper rotation, then convert back.  The
    resulting quaternion's ellipsoid SDF equals the original's reflected SDF
    (verified numerically over random orientations/axes).
    """
    M = np.eye(3, dtype=np.float64)
    M[axis, axis] = -1.0
    Rm = M @ _quat_to_rot_matrix(quat_xyzw)
    if np.linalg.det(Rm) < 0.0:
        Rm[:, 0] = -Rm[:, 0]
    return _rot_matrix_to_quat(Rm)


def _mirror_quat_mean(quat_xyzw: np.ndarray, axis: int) -> np.ndarray:
    """Mirror-symmetric orientation: nlerp average of ``q`` and ``mirror(q)``.

    Used for on-plane ellipsoids so their orientation is invariant under the
    symmetry reflection (the mirror swaps the two, fixing their mean).
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    qm = _mirror_quat(q, axis).astype(np.float64)
    if np.dot(q, qm) < 0.0:
        qm = -qm
    qs = q + qm
    n = np.linalg.norm(qs)
    if n < 1e-9:                       # antipodal → fall back to the mirror itself
        return _mirror_quat(q, axis)
    return (qs / n).astype(np.float32)


# Vectorised closed-form mirror (M,4) — equivalent to ``_mirror_quat`` per row but
# branch-free: reflecting across the plane normal to ``axis`` negates the two
# quaternion vector components perpendicular to ``axis`` (the axis component and w
# stay).  Verified to yield the same ellipsoid as the matrix method; it is an
# involution (applying it twice is the identity), so the on-plane mean below is
# exactly mirror-invariant.
def _mirror_quats(q: np.ndarray, axis: int) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    perp = [i for i in range(3) if i != axis]
    out[:, perp] *= -1.0
    return out


def _mirror_quats_mean(q: np.ndarray, axis: int) -> np.ndarray:
    """Row-wise mirror-symmetric orientation (nlerp of q and mirror(q))."""
    q = np.asarray(q, dtype=np.float32)
    qm = _mirror_quats(q, axis)
    flip = np.sum(q * qm, axis=1) < 0.0
    qm[flip] *= -1.0
    s = q + qm
    nrm = np.linalg.norm(s, axis=1, keepdims=True)
    bad = nrm[:, 0] < 1e-9
    if bad.any():
        s[bad] = qm[bad]
        nrm = np.linalg.norm(s, axis=1, keepdims=True)
    return (s / np.maximum(nrm, 1e-12)).astype(np.float32)


# ── Worker ────────────────────────────────────────────────────────────────────

class OptimizationWorker(QtCore.QThread):
    """Ellipsoid fitting with mini-batch sampling + conservative population mgmt.

    Pruning philosophy
    ------------------
    Large ellipsoids are the backbone of the approximation and are
    *never* removed in favour of smaller ones.  A small ellipsoid is
    only pruned when it is **contained** inside a larger one — meaning
    its centre lies inside the larger ellipsoid (normalised distance
    < containment_thresh) and all its radii are smaller.

    Per maintenance round at most ``max_prune_fraction`` of the current
    population is removed so training stays stable.

    Parameters
    ----------
    maintenance_every : int
        Prune/spawn cycle frequency (default 200 — frequent but gentle).
    containment_thresh : float
        Normalised-distance threshold for considering a point "inside"
        the larger ellipsoid.  1.0 = exactly on the surface.
        Default 0.8 (well inside).
    max_prune_fraction : float
        At most this fraction of the population may be pruned per round
        (default 0.15 = 15 %).
    min_volume_abs : float
        Absolute volume floor — ellipsoids with prod(radii) below this
        are considered degenerate (default 1e-8).
    """

    step_visual      = QtCore.Signal(int, float, object, object, object, object)
    step_sdf         = QtCore.Signal(int, float, object, object, object)  # (step, loss, ell_grid, ur_points, ur_values)
    maintenance_done = QtCore.Signal(int, int, int, int)
    phase_changed    = QtCore.Signal(str)   # "global" | "local"
    local_progress   = QtCore.Signal(int, int)   # (current, total) local-fit steps
    region_changed   = QtCore.Signal(object)  # (aabb_min, aabb_max) world box, or None
    prep_progress    = QtCore.Signal(float, str)  # (0..1, label) pre-training setup
    op_events        = QtCore.Signal(int, object)  # (step, [(op:str, center:(3,), radius:float), ...])
    analysis_regions = QtCore.Signal(int, object)  # (step, {'over'|'under'|'bridge': [(center, radius), ...]})
    finished         = QtCore.Signal()

    DEFAULT_BATCH_FRACTION = 0.125

    def __init__(
        self,
        sdf_target_np: np.ndarray,
        origin: np.ndarray,
        dx: float,
        n: int,
        num_ellipsoids: int = 10,
        method: str = "adam",
        num_steps: int = 2000,
        report_every: int = 20,
        sdf_mode: int = 4,
        batch_fraction: float | None = None,
        batch_size: int | None = None,
        sample_budget: int = 49152,
        surface_band_vox: float = 3.0,
        surface_fraction: float = 0.75,
        maintenance_every: int = 200,
        miss_penalty_weight: float = 3.0,
        outside_penalty_weight: float = 14.0,
        containment_weight: float = 6.0,
        surface_weight: float = 4.0,
        surface_sigma_vox: float = 1.5,
        underrep_rel_threshold: float = 0.6,
        underrep_min_gap_vox: float = 0.5,
        underrep_min_thickness_vox: float = 4.0,
        max_prune_fraction: float = 0.15,
        min_volume_abs: float = 1e-8,
        coverage_sample_size: int = 20000,
        superfit: bool = False,
        max_ellipsoids: int = 60,
        superfit_every: int = 150,
        densify_start_frac: float = 0.0,
        densify_until_frac: float = 0.75,
        local_steps: int = 1200,
        local_lr: float = 0.02,
        region_radius_vox: float = 6.0,
        spawn_per_round: int = 3,
        spawn_underrep: bool = True,
        split_enabled: bool = True,
        split_per_round: int = 7,
        split_margin_vox: float = 0.5,
        split_size_factor: float = 1.2,
        min_split_radius_vox: float = 2.0,
        bridge_min_outside: float = 0.1,
        fuse_per_round: int = 2,
        fuse_overlap_frac: float = 0.9,
        fuse_samples: int = 96,
        merge_per_round: int = 3,
        merge_tol: float = 0.12,
        merge_enabled: bool = True,
        prune_enabled: bool = True,
        bone_aware: bool = False,
        bone_centers_np: np.ndarray | None = None,
        bone_span_weight: float = 0.4,
        bone_span_tol: float = 0.35,
        bone_span_soft: float = 0.15,
        lr_init: float = 0.01,
        lr_final: float = 0.0002,
        lr_decay_k: float = 7.0,
        lr_mult_radii: float = 2.0,
        lr_mult_rot: float = 1.0,
        soft_union: bool = False,   # experimental — tended to look worse in tests
        soft_union_vox_start: float = 3.0,
        soft_union_vox_end: float = 0.6,
        thickness_np: np.ndarray | None = None,
        thin_loss_weight: float = 1.0,
        thin_max_factor: float = 6.0,
        thin_sample_bias: float = 1.0,
        flat_weight: float = 0.5,
        flat_min_ratio: float = 0.35,
        degenerate_flat_ratio: float = 0.12,
        degenerate_spike_ratio: float = 8.0,
        sdf_computer=None,
        region_res: int | None = None,
        local_fit: bool = True,
        local_fit_start_frac: float = 0.25,
        local_fit_end_frac: float = 1.0,
        local_fit_every: int = 150,
        region_dc_cycles: int = 3,
        region_steps: int = 2000,
        symmetry_enabled: bool = False,
        symmetry_every: int = 100,
        primitive_shape: str = "ellipsoid",
        sq_eps1: float = 1.0,
        sq_eps2: float = 1.0,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._sdf_target_np = sdf_target_np
        self._origin = origin
        self._dx = dx
        self._n = n
        # Per-axis voxel counts — the grid may be anisotropic (n resolves only the
        # longest axis).  Derived from the target grid's own shape so every
        # downstream flat-index / kernel launch / world↔voxel mapping matches it.
        self._nz, self._ny, self._nx = (int(s) for s in self._sdf_target_np.shape)
        self._shape = (self._nz, self._ny, self._nx)
        self._num_ellipsoids = num_ellipsoids
        self._method = method
        self._num_steps = num_steps
        self._report_every = report_every
        self._sdf_mode = sdf_mode
        self._stop_flag = False

        self._maintenance_every = maintenance_every
        self._miss_penalty_weight = miss_penalty_weight
        self._outside_penalty_weight = outside_penalty_weight
        self._containment_weight = containment_weight
        self._surface_weight = surface_weight
        self._surface_sigma = max(surface_sigma_vox * float(dx), 1e-6)
        self._underrep_rel_threshold = float(underrep_rel_threshold)
        self._underrep_min_gap_vox = float(underrep_min_gap_vox)
        self._underrep_min_thickness_vox = float(underrep_min_thickness_vox)
        self._max_prune_fraction = max_prune_fraction
        self._min_volume_abs = min_volume_abs
        self._coverage_sample_size = coverage_sample_size

        self._superfit = superfit
        self._max_ellipsoids = max_ellipsoids
        self._superfit_every = superfit_every
        self._densify_start_frac = float(np.clip(densify_start_frac, 0.0, 1.0))
        self._densify_until_frac = float(np.clip(densify_until_frac, 0.0, 1.0))
        self._local_steps = local_steps
        self._local_lr = local_lr
        self._region_radius_vox = region_radius_vox
        # How many under-represented regions to surface for the analysis overlay
        # (≥ the densify budget so the viewport shows more than just the few that
        # get acted on this cycle).
        self._analysis_region_k = 24
        # Severity floor (relative-miss × surface-emphasis) below which a region
        # is NOT considered under-represented.  Filters the marginal half-voxel
        # gaps that otherwise flood both the overlay and the spawn/split picker.
        # Severity = rel · sw with sw up to (1 + surface_weight) on the surface,
        # so requiring a surface region to reach the configured ``rel_threshold``
        # before it counts means a floor of rel_threshold·(1+sw_max).
        self._analysis_min_severity = (
            self._underrep_rel_threshold * (1.0 + float(self._surface_weight)))
        # SuperFit region detection (the only n³ cost per cycle) runs on a
        # resolution-capped copy of the target grid so its cost is independent
        # of the global grid n.  Longest axis of the detection grid ≤ this cap;
        # the predicted-grid build + relative-underrep scan then scale with the
        # cap, not with n.  The local fit already uses ``region_res`` (n-free).
        self._region_detect_cap = 64
        self._det_cache: dict | None = None
        self._spawn_per_round = spawn_per_round
        self._spawn_underrep = spawn_underrep
        self._split_enabled = split_enabled
        self._split_per_round = split_per_round
        self._split_margin_vox = split_margin_vox
        self._split_size_factor = split_size_factor
        self._min_split_radius_vox = min_split_radius_vox
        self._bridge_min_outside = bridge_min_outside
        self._fuse_per_round = fuse_per_round
        self._fuse_overlap_frac = fuse_overlap_frac
        self._fuse_samples = fuse_samples
        self._fuse_unit_pts = None      # cached unit-ball sample points
        # ── merge step: fuse two overlapping ellipsoids into one when doing so
        # barely changes the union surface (see _detect_merges) ──
        self._merge_per_round = merge_per_round
        self._merge_tol = merge_tol
        self._merge_enabled = merge_enabled
        self._prune_enabled = prune_enabled
        self._merge_sphere_pts = None   # cached unit-sphere surface points
        # ── bone-awareness: penalise ellipsoids spanning multiple bones ──
        self._bone_span_weight = float(bone_span_weight)
        self._bone_span_tol = float(bone_span_tol)
        self._bone_span_soft = float(bone_span_soft)
        self._bone_centers_np = None
        self._num_bones = 0
        if bone_aware and bone_centers_np is not None and len(bone_centers_np) > 0:
            self._bone_centers_np = np.ascontiguousarray(
                bone_centers_np, dtype=np.float32)
            self._num_bones = int(len(self._bone_centers_np))
        self._bone_aware = self._num_bones > 0
        self._lr_init = lr_init
        self._lr_final = lr_final
        self._lr_decay_k = lr_decay_k
        self._lr_mult_radii = lr_mult_radii   # per-group LR (radii in log-space)
        self._lr_mult_rot = lr_mult_rot
        # Soft-min (smooth) union of the ellipsoid SDFs — denser gradients.  The
        # blend width is annealed from ``vox_start`` → ``vox_end`` voxels over
        # training (soft early for coverage, near-hard late for accuracy).
        self._soft_union = bool(soft_union)
        self._soft_vox_start = float(soft_union_vox_start)
        self._soft_vox_end = float(soft_union_vox_end)
        self._thickness_np = thickness_np
        self._thin_loss_weight = thin_loss_weight
        self._thin_max_factor = thin_max_factor
        self._thin_sample_bias = thin_sample_bias
        self._thickness_flat = None     # dilated flat thickness (built lazily)
        self._flat_weight = flat_weight
        self._flat_min_ratio = flat_min_ratio
        # Hard-delete thresholds for degenerate shapes during SuperFit (axis
        # ratios vs the median axis): too-flat disks and too-pointy spikes.
        self._degenerate_flat_ratio = degenerate_flat_ratio
        self._degenerate_spike_ratio = degenerate_spike_ratio
        # High-res per-region local fitting (SuperFit). When a mesh-backed
        # SdfComputer is supplied, each maintained region is re-fitted against a
        # fresh ``region_res³`` SDF box limited to that region (genuinely finer
        # than the global coarse grid).
        self._sdf_computer = sdf_computer
        # Local optimisation ALWAYS runs on a fixed 128³ box (per user request):
        # applied to the small region box this is genuinely finer (much smaller
        # dx) than the global coarse grid, regardless of the global n.
        self._region_res = int(region_res) if region_res is not None else 128
        # Number of divide-and-conquer passes executed *within* one local
        # optimisation of a region, and the total Adam steps spent per region
        # (split across the cycles).  Longer + multi-pass D&C refines the region
        # instead of blowing it up.
        self._region_dc_cycles = max(1, int(region_dc_cycles))
        self._region_steps = max(self._local_steps, int(region_steps))
        # When off, SuperFit still does divide-and-conquer maintenance
        # (delete/fuse/split/spawn) but skips the per-region high-res local fit;
        # the new ellipsoids are left for the global optimiser to refine.
        self._local_fit_enabled = bool(local_fit)
        # Local fit only kicks in after this fraction of training has elapsed
        # (default 25 %): early steps refine the global layout first, then the
        # high-res per-region fit starts once the population has roughly settled.
        self._local_fit_start_frac = float(np.clip(local_fit_start_frac, 0.0, 1.0))
        # Local fit runs on its own window [start, end] and its own elapsed
        # cadence (``local_fit_every`` steps), fully decoupled from the densify
        # window/cadence so the two phases can overlap, be disjoint, or differ in
        # frequency.  ``_last_local_fit_step`` tracks when local fit last fired.
        self._local_fit_end_frac = float(np.clip(local_fit_end_frac, 0.0, 1.0))
        self._local_fit_every = max(1, int(local_fit_every))
        self._last_local_fit_step = -10**9
        # Symmetry constraint: mirror-projection during training onto an
        # auto-detected plane (resolved lazily once the loop starts).
        self._symmetry_enabled = bool(symmetry_enabled)
        self._symmetry_every = max(1, int(symmetry_every))
        # Primitive-shape constraint.  "sphere" → project to isotropic radii +
        # identity rotation after every optimiser step (see _project_isotropic).
        self._primitive_shape = str(primitive_shape).lower()
        self._isotropic = (self._primitive_shape == "sphere")
        # Superquadric: shared roundness exponents (fixed during the fit).  The
        # dedicated kernels evaluate the generalised SDF; radii + rotation are
        # trained as usual (no isotropy/rotation projection).
        # Both "superquadric" and "bent_superquadric" use the superquadric
        # kernels; the bent variant additionally trains a per-primitive bend.
        self._superquadric = self._primitive_shape in (
            "superquadric", "bent_superquadric")
        self._bent = (self._primitive_shape == "bent_superquadric")
        # Capsule: segment + radius.  radius = r[0], half-length = r[2]; r[1] is
        # projected to r[0] each step (circular cross-section).
        self._capsule = (self._primitive_shape == "capsule")
        self._sq_eps1 = float(np.clip(sq_eps1, 0.1, 2.0))
        self._sq_eps2 = float(np.clip(sq_eps2, 0.1, 2.0))
        # Dedicated, (nearly) constant learning rate for the roundness exponents
        # and the bend curvature — the global LR decays to ~lr_final, far too
        # small to move them in the refinement phase, so they would stay stuck.
        self._sq_eps_lr = max(float(lr_init), 0.01)
        # Max |bend curvature| (clamped each step; conservative for stability).
        self._bend_max = 6.0
        self._sym_axis = None         # world axis 0/1/2, set by _detect_symmetry_axis
        self._sym_plane = None        # world-space plane coordinate on that axis
        # Symmetry is auto-gated: detection runs once and may resolve to "no
        # symmetry" (``_sym_axis`` stays None), in which case it is skipped for
        # the whole run.  ``_sym_checked`` prevents re-detecting on a grid that a
        # previous pass already symmetrised.
        self._sym_checked = False
        # Hard-mirror layout: the ellipsoid set is ordered
        #   [on-plane (trainable) | source (trainable) | mirror (derived)]
        # and only the first two blocks are trained; ``mirror`` is re-derived from
        # ``source`` after every step.  These counts track the partition.
        self._sym_n_op = 0            # number of on-plane (trainable) ellipsoids
        self._sym_n_so = 0            # number of source off-plane ellipsoids
        # Thin-feature loss weighting (built lazily on the device once).
        self._wp_thickness = None
        self._thick_ref = 1.0
        self._thin_weight_eff = 0.0
        self._rng = np.random.default_rng()

        self._surface_band_vox = surface_band_vox
        self._surface_fraction = surface_fraction

        total = self._nx * self._ny * self._nz
        if batch_size is not None:
            self._batch_size = min(batch_size, total)
        elif batch_fraction is not None:
            self._batch_size = max(1024, min(int(total * batch_fraction), total))
        else:
            # Resolution-independent budget: cost per step no longer grows with n³.
            self._batch_size = min(int(sample_budget), total)

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        if self._method == "adam":
            self._run_adam()
        else:
            self._run_naive()
        self.finished.emit()

    # ── shape constraint (sphere) ─────────────────────────────────────

    def _project_isotropic(self, log_radii, rot_flat, num_e) -> None:
        """Project params onto the sphere subspace (isotropic + no rotation).

        No-op unless this is a sphere fit.  Cheap enough to run every step and on
        the whole array (already-isotropic primitives are unchanged), so it can
        be called from every optimisation loop (global / local / symmetry).
        """
        if not self._isotropic or num_e <= 0:
            return
        wp.launch(_broadcast_log_radii, dim=num_e, inputs=[log_radii], device=device)
        wp.launch(_reset_rot_identity, dim=num_e, inputs=[rot_flat], device=device)

    def _project_isotropic_np(self, r_np: np.ndarray, q_np: np.ndarray) -> tuple:
        """Numpy projection of a maintenance/local-fit result to spheres."""
        if not self._isotropic:
            return r_np, q_np
        r_iso = np.repeat(np.asarray(r_np, dtype=np.float32).mean(axis=1, keepdims=True),
                          3, axis=1).astype(np.float32)
        q_id = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                       (len(r_np), 1))
        return r_iso, q_id

    def _project_capsule(self, radii, num_e) -> None:
        """Project radii onto the capsule subspace (circular cross-section r1=r0).

        Works on log-radii or world radii; no-op unless this is a capsule fit.
        """
        if not self._capsule or num_e <= 0:
            return
        wp.launch(_capsule_eq_radii, dim=num_e, inputs=[radii], device=device)

    def _project_capsule_np(self, r_np: np.ndarray) -> np.ndarray:
        """Numpy projection of a maintenance result to capsules (r1 = r0)."""
        if not self._capsule:
            return r_np
        r = np.asarray(r_np, dtype=np.float32).copy()
        r[:, 1] = r[:, 0]
        return r

    # ── progress reporting ────────────────────────────────────────────

    def _emit_progress(self, step, loss_wp, pred_centers, pred_radii,
                       pred_rot_flat, num_e, origin, dx, n, pred_eps=None,
                       pred_bend=None):
        wp.synchronize_device(device)
        loss_val = float(loss_wp.numpy()[0])

        c_np = pred_centers.numpy().copy()
        r_np = pred_radii.numpy().copy()
        q_np = pred_rot_flat.numpy().reshape(-1, 4).copy()
        e_np = None
        if pred_eps is not None:
            e_np = pred_eps.numpy().reshape(-1, 2).copy()
            if pred_bend is not None:
                # Pack bend after eps → (N,4) = [e1, e2, kx, ky] for bent shapes.
                b_np = pred_bend.numpy().reshape(-1, 2).copy()
                e_np = np.concatenate([e_np, b_np], axis=1)
        self.step_visual.emit(step, loss_val, c_np, r_np, q_np, e_np)

        # NB: the per-step ellipsoid-SDF grid + under-rep used to be computed
        # here and emitted via ``step_sdf`` for an ellipsoid slice view.  That
        # view was removed (the slice now shows the mesh only), so its consumer
        # is a no-op — computing the n³ grid every report_every·10 steps was
        # pure wasted work (costly on CPU especially) and has been dropped.
        # The spawn/maintenance path computes its own under-rep independently.

    # ── buffer allocation ─────────────────────────────────────────────

    def _init_inside_mesh(self, num_e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial ellipsoid parameters placed inside the mesh.

        Uses farthest-point sampling on interior voxels (sdf_target < 0)
        to get diverse starting positions.  Initial radii are proportional
        to local depth so ellipsoids start at a reasonable size.
        """
        origin = self._origin
        dx = self._dx
        n = self._n

        flat_target = self._sdf_target_np.ravel()
        interior_mask = flat_target < 0.0
        interior_idx = np.where(interior_mask)[0]

        if len(interior_idx) == 0:
            # Fallback: random in bounding box
            centers = (np.random.rand(num_e, 3).astype(np.float32) - 0.5)
            radii = np.ones((num_e, 3), dtype=np.float32) * 0.1
            rots = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_e, 1))
            return centers, radii, rots, self._init_eps(num_e)

        # Convert interior voxels to world positions
        iz, iy, ix = np.unravel_index(interior_idx, self._shape)
        interior_world = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx
        interior_depth = np.abs(flat_target[interior_idx])  # distance to surface

        # Use depth as "importance" for FPS — prefer deep interior points
        selected = self._farthest_point_sample(
            interior_world, interior_depth, num_e,
            existing_centers=np.empty((0, 3), dtype=np.float32),
        )

        centers = interior_world[selected].astype(np.float32)
        local_depth = interior_depth[selected]

        # Initial radii: 60% of local depth, at least 2×dx
        min_r = float(dx) * 2.0
        init_r = np.clip(local_depth * 0.6, min_r, None)
        radii = np.stack([init_r, init_r, init_r], axis=1).astype(np.float32)

        rots = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_e, 1),
        )

        eps = self._init_eps(len(centers))
        # Start in the hard-mirror layout so only the source half is trained from
        # step 0 and the mirror half is its slave.
        if self._symmetry_enabled and self._sym_axis is not None:
            centers, radii, rots, eps = self._build_symmetric_layout(
                centers, radii, rots, eps)
        return centers, radii, rots, eps

    def _init_eps(self, n: int) -> np.ndarray:
        """Per-primitive superquadric exponents, initialised from the UI values."""
        return np.tile(
            np.array([self._sq_eps1, self._sq_eps2], dtype=np.float32), (int(n), 1))

    def _uniform_eps_wp(self, n: int):
        """A device eps array (vec2) of uniform init values — used where the
        trainable per-primitive eps isn't threaded (naive / local-fit paths)."""
        return wp.array(self._init_eps(n).reshape(-1).astype(np.float32),
                        dtype=wp.float32, device=device)

    def _init_bend(self, n: int) -> np.ndarray:
        """Per-primitive bend curvature (kx, ky), initialised straight (0)."""
        return np.zeros((int(n), 2), dtype=np.float32)

    def _zero_bend_wp(self, n: int):
        """A device bend array (flat 2·n, all zero) — straight, for paths that
        don't thread the trainable bend (naive / local-fit)."""
        return wp.zeros(2 * int(n), dtype=wp.float32, device=device)

    # ── symmetry constraint ────────────────────────────────────────────

    def _detect_symmetry_axis(self, grid: np.ndarray):
        """Auto-detect the mirror plane from the (AABB-centred) SDF grid.

        For each world axis, compare the grid with its flip about the grid centre
        over a surface band; the axis with the smallest mismatch is the symmetry
        axis.  Returns ``(world_axis, plane_coord)`` where ``plane_coord`` is the
        AABB centre on that axis.  Logs a warning if even the best axis is a poor
        match (mesh not actually symmetric there).
        """
        dx = float(self._dx)
        band = np.abs(grid) < (3.0 * dx)        # near-surface voxels only
        denom = max(int(band.sum()), 1)
        # world axis -> numpy array axis (grid is (nz, ny, nx))
        arr_of = {0: 2, 1: 1, 2: 0}
        scale = max(float(np.abs(grid[band]).mean()), 1e-6) if band.any() else 1.0
        errs = {}
        for wax, aax in arr_of.items():
            flipped = np.flip(grid, axis=aax)
            errs[wax] = float(np.abs(grid[band] - flipped[band]).sum() / denom)
        best = min(errs, key=errs.get)
        rel = errs[best] / scale
        n_axis = (self._nx, self._ny, self._nz)[best]
        plane = float(self._origin[best] + 0.5 * n_axis * dx)
        axis_name = "XYZ"[best]
        # Only treat the mesh as symmetric when the best axis is a genuinely good
        # match; otherwise return no axis so symmetry is skipped for this run.
        if rel > 0.15:
            print(f"[symmetry] best axis {axis_name} mismatch too high "
                  f"(rel={rel:.3f}); mesh not symmetric — symmetry skipped.")
            return None, None
        print(f"[symmetry] detected axis {axis_name} (rel mismatch {rel:.3f}), "
              f"plane @ {plane:.4f}")
        return best, plane

    def _symmetry_source_side(self, centers: np.ndarray) -> float:
        """Sign of ``centre - plane`` for the source (better-fitting) half.

        Returns ``+1.0`` if the positive side is the source half, ``-1.0`` for
        the negative side — matching the choice ``_build_symmetric_layout`` makes
        (lower mean ``|residual at centre|``; ties → more populous side, then
        positive).  Used to keep the per-region local fit on the half that
        actually survives the re-layout.
        """
        a, p = self._sym_axis, self._sym_plane
        c = np.asarray(centers, dtype=np.float32)
        if a is None or len(c) == 0:
            return 1.0
        tol = 1.5 * float(self._dx)
        signed = c[:, a] - p
        on_plane = np.abs(signed) < tol
        pos = (~on_plane) & (signed >= 0.0)
        neg = (~on_plane) & (signed < 0.0)

        def _side_residual(mask):
            if not np.any(mask):
                return np.inf
            errs = [abs(self._grid_value(self._sdf_target_np, c[i]))
                    for i in np.where(mask)[0]]
            return float(np.mean(errs)) if errs else np.inf
        rp, rn = _side_residual(pos), _side_residual(neg)
        if rp < rn:
            return 1.0
        if rn < rp:
            return -1.0
        return 1.0 if pos.sum() >= neg.sum() else -1.0

    def _build_symmetric_layout(self, centers: np.ndarray, radii: np.ndarray,
                                rotations: np.ndarray, eps: np.ndarray | None = None):
        """Re-order an ellipsoid set into the hard-mirror training layout.

        Output order is ``[on_plane | source | mirror]`` where:
          * on-plane ellipsoids (centre within a voxel-scale tolerance of the
            plane) are pinned to the plane with a mirror-symmetric orientation;
          * ``source`` are the off-plane ellipsoids of the better-fitting half;
          * ``mirror`` are their reflections (1:1, same order).
        Only ``[on_plane | source]`` are trained; ``mirror`` is re-derived from
        ``source`` every step.  Also records ``self._sym_n_op`` / ``self._sym_n_so``.
        Returns ``(centers, radii, rotations)``.
        """
        a, p = self._sym_axis, self._sym_plane
        c = np.asarray(centers, dtype=np.float32)
        r = np.asarray(radii, dtype=np.float32)
        q = np.asarray(rotations, dtype=np.float32)
        if eps is None:
            eps = self._init_eps(len(c))
        e = np.asarray(eps, dtype=np.float32)
        if a is None or len(c) == 0:
            self._sym_n_op, self._sym_n_so = 0, 0
            return c, r, q, e

        tol = 1.5 * float(self._dx)
        signed = c[:, a] - p
        on_plane = np.abs(signed) < tol
        pos = (~on_plane) & (signed >= 0.0)
        neg = (~on_plane) & (signed < 0.0)

        # Source half = the better-fitting side (resolved identically by
        # ``_symmetry_source_side`` so the local fit can pick the same half).
        src_mask = pos if self._symmetry_source_side(c) >= 0.0 else neg

        op_idx = np.where(on_plane)[0]
        so_idx = np.where(src_mask)[0]
        n_op, n_so = int(op_idx.size), int(so_idx.size)

        # on-plane block (pinned + symmetric orientation)
        op_c = c[op_idx].copy()
        if n_op:
            op_c[:, a] = p
        op_r = r[op_idx].copy()
        op_e = e[op_idx].copy()
        op_q = _mirror_quats_mean(q[op_idx], a) if n_op else q[op_idx]

        # source block (unchanged) + mirror block (derived from source)
        so_c, so_r, so_q = c[so_idx].copy(), r[so_idx].copy(), q[so_idx].copy()
        so_e = e[so_idx].copy()
        mi_c = so_c.copy()
        if n_so:
            mi_c[:, a] = 2.0 * p - so_c[:, a]
        mi_q = _mirror_quats(so_q, a)

        out_c = np.concatenate([op_c, so_c, mi_c], axis=0).astype(np.float32)
        out_r = np.concatenate([op_r, so_r, so_r], axis=0).astype(np.float32)
        out_q = np.concatenate([op_q, so_q, mi_q], axis=0).astype(np.float32)
        # eps follows radii exactly: mirror inherits the source's exponents.
        out_e = np.concatenate([op_e, so_e, so_e], axis=0).astype(np.float32)
        if len(out_c) == 0:
            self._sym_n_op, self._sym_n_so = 0, 0
            return c, r, q, e
        self._sym_n_op, self._sym_n_so = n_op, n_so
        return out_c, out_r, out_q, out_e

    def _project_symmetry_inplace(self, pred_centers, pred_radii,
                                  pred_rot_flat, pred_eps=None) -> None:
        """Re-impose exact symmetry on the live device arrays, count unchanged.

        Run after every Adam step: re-derive the ``mirror`` block from ``source``
        and re-pin/​symmetrise the ``on_plane`` block, writing back in place so the
        Adam optimiser (and its moments) are preserved — no rebuild.  This is what
        makes the mirror half a pure slave of the trained source half.
        """
        a, p = self._sym_axis, self._sym_plane
        n_op, n_so = self._sym_n_op, self._sym_n_so
        if a is None or (n_op == 0 and n_so == 0):
            return
        c = pred_centers.numpy()
        r = pred_radii.numpy()
        q = pred_rot_flat.numpy().reshape(-1, 4)

        e = pred_eps.numpy().reshape(-1, 2) if pred_eps is not None else None

        if n_op:
            c[:n_op, a] = p
            q[:n_op] = _mirror_quats_mean(q[:n_op], a)
        if n_so:
            s0, m0 = n_op, n_op + n_so
            src_c, src_q = c[s0:s0 + n_so], q[s0:s0 + n_so]
            mc = src_c.copy(); mc[:, a] = 2.0 * p - src_c[:, a]
            c[m0:m0 + n_so] = mc
            r[m0:m0 + n_so] = r[s0:s0 + n_so]
            q[m0:m0 + n_so] = _mirror_quats(src_q, a)
            if e is not None:
                e[m0:m0 + n_so] = e[s0:s0 + n_so]   # mirror eps = source eps

        pred_centers.assign(np.ascontiguousarray(c))
        pred_radii.assign(np.ascontiguousarray(r))
        pred_rot_flat.assign(np.ascontiguousarray(q.reshape(-1)))
        if e is not None:
            pred_eps.assign(np.ascontiguousarray(e.reshape(-1)))

    def _setup_symmetry(self) -> None:
        """Resolve the symmetry plane and symmetrise the target + thickness grids.

        Run once before fitting starts.  Averaging each grid with its own flip
        removes discretisation asymmetry so both halves share an identical target.
        """
        self._sym_axis, self._sym_plane = self._detect_symmetry_axis(
            self._sdf_target_np)
        self._sym_checked = True
        if self._sym_axis is None:
            # Mesh is not symmetric — leave the target/thickness grids untouched.
            return
        aax = {0: 2, 1: 1, 2: 0}[self._sym_axis]
        g = self._sdf_target_np
        self._sdf_target_np = (0.5 * (g + np.flip(g, axis=aax))).astype(np.float32)
        if self._thickness_np is not None:
            t = self._thickness_np
            self._thickness_np = (0.5 * (t + np.flip(t, axis=aax))).astype(np.float32)

    def _alloc_buffers(
        self,
        num_e: int,
        batch_size: int,
        total: int,
        centers_np: np.ndarray | None = None,
        radii_np: np.ndarray | None = None,
        rot_np: np.ndarray | None = None,
        sdf_target_np: np.ndarray | None = None,
        eps_np: np.ndarray | None = None,
        bend_np: np.ndarray | None = None,
    ) -> dict:
        src = self._sdf_target_np if sdf_target_np is None else sdf_target_np
        sdf_target = wp.array(
            src.flatten(),
            dtype=wp.float32, device=device, requires_grad=False,
        )

        if centers_np is None:
            centers_np, radii_np, rot_np, eps_np = self._init_inside_mesh(num_e)
        if radii_np is None:
            radii_np = np.ones((num_e, 3), dtype=np.float32) * 0.1
        if rot_np is None:
            rot_np = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_e, 1),
            )
        if eps_np is None:
            eps_np = self._init_eps(len(centers_np))
        if bend_np is None:
            bend_np = self._init_bend(len(centers_np))
        # The actual count can differ from the requested ``num_e`` — e.g. the
        # symmetric layout returns on_plane + 2·source.  Size every buffer to it
        # so ``min_d_cache`` matches ``pred_centers`` and the kernel never reads
        # out of bounds.
        num_e = int(centers_np.shape[0])

        pred_centers = wp.array(
            centers_np.astype(np.float32), dtype=wp.vec3,
            device=device, requires_grad=True,
        )
        pred_radii = wp.array(
            radii_np.astype(np.float32), dtype=wp.vec3,
            device=device, requires_grad=True,
        )
        pred_rot_flat = wp.array(
            rot_np.astype(np.float32).flatten(), dtype=wp.float32,
            device=device, requires_grad=True,
        )
        pred_eps = wp.array(
            np.ascontiguousarray(eps_np[:num_e].reshape(-1), dtype=np.float32),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        pred_bend = wp.array(
            np.ascontiguousarray(bend_np[:num_e].reshape(-1), dtype=np.float32),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        min_d_cache = wp.zeros(
            shape=(batch_size, num_e + 1),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        sdf_pred = wp.empty(
            batch_size, dtype=wp.float32, device=device, requires_grad=True,
        )
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        wp_indices = wp.empty(batch_size, dtype=wp.int32, device=device)

        return dict(
            sdf_target=sdf_target,
            pred_centers=pred_centers,
            pred_radii=pred_radii,
            pred_rot_flat=pred_rot_flat,
            pred_eps=pred_eps,
            pred_bend=pred_bend,
            min_d_cache=min_d_cache,
            sdf_pred=sdf_pred,
            loss=loss,
            wp_indices=wp_indices,
        )

    # ══════════════════════════════════════════════════════════════════
    # CONSERVATIVE POPULATION MANAGEMENT
    # ══════════════════════════════════════════════════════════════════

    def _do_maintenance(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, int]:
        """Coverage-based prune + proactive spawn.

        Returns (centers, radii, rotations, changed, n_pruned, n_spawned).
        """
        n_before = len(centers)
        budget = max(1, int(n_before * self._max_prune_fraction))

        # ── 1. Remove clearly degenerate ellipsoids ──────────────────
        volumes = np.prod(np.abs(radii), axis=1)
        vol_ok = volumes > self._min_volume_abs
        n_degenerate = int(np.count_nonzero(~vol_ok))

        if n_degenerate > budget:
            degen_idx = np.where(~vol_ok)[0]
            degen_vols = volumes[degen_idx]
            keep_from_degen = degen_idx[np.argsort(degen_vols)[-max(0, n_degenerate - budget):]]
            vol_ok[keep_from_degen] = True

        centers   = centers[vol_ok]
        radii     = radii[vol_ok]
        rotations = rotations[vol_ok]
        n_removed = n_before - len(centers)
        budget -= n_removed

        # ── 2. Coverage-based pruning ─────────────────────────────────
        cov = self._compute_coverage_info(centers, radii, rotations)

        n_pruned = n_removed
        if self._prune_enabled and budget > 0 and len(centers) >= 2 and cov['valid']:
            to_remove = self._select_prune_candidates(cov, budget)
            if to_remove:
                keep_mask = np.ones(len(centers), dtype=bool)
                keep_mask[to_remove] = False
                centers   = centers[keep_mask]
                radii     = radii[keep_mask]
                rotations = rotations[keep_mask]
                budget   -= len(to_remove)
                n_pruned += len(to_remove)
                cov = self._filter_coverage_info(cov, np.where(keep_mask)[0])

        # ── 3. Proactively replace low-coverage ellipsoids ────────────
        if self._prune_enabled and cov['valid'] and budget > 0:
            n_uncov = cov['n_uncovered']
            n_sample = len(cov['pts'])
            uncov_frac = n_uncov / max(n_sample, 1)

            if n_uncov >= 50 and uncov_frac >= 0.05:
                extra_slots = min(budget, max(1, int(round(uncov_frac * self._num_ellipsoids * 0.5))))
                already_free = self._num_ellipsoids - len(centers)
                extra_prune = max(0, extra_slots - already_free)

                if extra_prune > 0 and len(centers) > 1:
                    total_cov  = cov['total_coverage']
                    unique_cov = cov['unique_coverage']
                    zero_unique_mask = unique_cov == 0
                    order = np.argsort(np.where(zero_unique_mask, total_cov, total_cov + 1e9))
                    swap_candidates = order[:extra_prune].tolist()

                    keep_mask = np.ones(len(centers), dtype=bool)
                    keep_mask[swap_candidates] = False
                    centers   = centers[keep_mask]
                    radii     = radii[keep_mask]
                    rotations = rotations[keep_mask]
                    n_pruned += len(swap_candidates)

        # ── 4. Spawn ─────────────────────────────────────────────────
        num_to_spawn = self._num_ellipsoids - len(centers)
        n_spawned = 0

        if num_to_spawn > 0:
            new_c, new_r, new_q = self._spawn_at_errors(
                centers, radii, rotations, num_to_spawn,
            )
            centers   = np.concatenate([centers, new_c], axis=0)
            radii     = np.concatenate([radii, new_r], axis=0)
            rotations = np.concatenate([rotations, new_q], axis=0)
            n_spawned = num_to_spawn

        changed = n_pruned > 0 or n_spawned > 0
        return centers, radii, rotations, changed, n_pruned, n_spawned

    # ── SDF helper (numpy, single ellipsoid) ──────────────────────────

    @staticmethod
    def _ellipsoid_sdf_np(center, radii, rotation_quat, points):
        """MertStein hybrid SDF for one ellipsoid at (N,3) points."""
        R = _quat_to_rot_matrix(rotation_quat)
        delta = points.astype(np.float64) - center.astype(np.float64)
        local_p = (R.T @ delta.T).T
        r = np.abs(radii).astype(np.float64)
        r_safe = np.maximum(r, 1e-12)
        scaled = local_p / r_safe[np.newaxis, :]
        k0 = np.linalg.norm(scaled, axis=1)
        r_min = float(r.min())
        scaled2 = local_p / np.maximum(r_safe ** 2, 1e-24)[np.newaxis, :]
        k1 = np.maximum(np.linalg.norm(scaled2, axis=1), 1e-8)
        inside  = (k0 - 1.0) * r_min
        outside = k0 * (k0 - 1.0) / k1
        return np.where(k0 < 1.0, inside, outside).astype(np.float32)

    @staticmethod
    def _ellipsoid_sdf_np_batch(centers, radii, rotations, points):
        """Vectorised MertStein hybrid SDF — (E,3)/(E,3)/(E,4) vs (N,3) → (E,N).

        Same math as ``_ellipsoid_sdf_np`` but evaluates *all* ellipsoids at once
        with broadcasting instead of a per-ellipsoid Python loop.  On CPU (no
        Warp acceleration here) this is the difference between E numpy passes and
        a single one — the hot path of coverage-based maintenance.
        """
        E = len(centers)
        # 3×3 rotation per ellipsoid; the E-loop is over tiny matrices (cheap)
        # while the expensive per-point work below stays fully vectorised.
        R = np.stack([_quat_to_rot_matrix(rotations[i]) for i in range(E)])  # (E,3,3)
        pts = points.astype(np.float64)                                      # (N,3)
        delta = pts[None, :, :] - centers[:, None, :].astype(np.float64)     # (E,N,3)
        # local_p = (Rᵀ·deltaᵀ)ᵀ == delta·R, batched over E.
        local_p = np.einsum('eni,eij->enj', delta, R)                        # (E,N,3)
        r = np.abs(radii).astype(np.float64)                                 # (E,3)
        r_safe = np.maximum(r, 1e-12)
        scaled = local_p / r_safe[:, None, :]
        k0 = np.linalg.norm(scaled, axis=2)                                  # (E,N)
        r_min = r.min(axis=1)                                                # (E,)
        scaled2 = local_p / np.maximum(r_safe ** 2, 1e-24)[:, None, :]
        k1 = np.maximum(np.linalg.norm(scaled2, axis=2), 1e-8)               # (E,N)
        inside  = (k0 - 1.0) * r_min[:, None]
        outside = k0 * (k0 - 1.0) / k1
        return np.where(k0 < 1.0, inside, outside).astype(np.float32)        # (E,N)

    # ── coverage computation (shared by pruning + spawn) ──────────────

    def _compute_coverage_info(self, centers, radii, rotations):
        n, dx, origin = self._n, self._dx, self._origin
        flat_target = self._sdf_target_np.ravel()
        interior_idx = np.where(flat_target < 0.0)[0]
        if len(interior_idx) == 0 or len(centers) == 0:
            return {'valid': False}

        sample_size = min(self._coverage_sample_size, len(interior_idx))
        sample_flat_idx = np.random.default_rng(0).choice(interior_idx, size=sample_size, replace=False)
        iz, iy, ix = np.unravel_index(sample_flat_idx, self._shape)
        pts = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx

        per_sdf      = self._ellipsoid_sdf_np_batch(centers, radii, rotations, pts)
        is_inside    = per_sdf < 0.0
        cover_count  = is_inside.sum(axis=0)
        unique_cov   = (is_inside & (cover_count == 1)[None, :]).sum(axis=1).astype(int)
        total_cov    = is_inside.sum(axis=1).astype(int)
        uncov_mask   = cover_count == 0
        return {
            'valid': True, 'pts': pts, 'sample_flat_idx': sample_flat_idx,
            'is_inside': is_inside, 'cover_count': cover_count,
            'unique_coverage': unique_cov, 'total_coverage': total_cov,
            'uncovered_mask': uncov_mask, 'n_uncovered': int(uncov_mask.sum()),
        }

    @staticmethod
    def _filter_coverage_info(cov, keep_indices):
        if not cov['valid']:
            return cov
        is_inside   = cov['is_inside'][keep_indices]
        cover_count = is_inside.sum(axis=0)
        unique_cov  = (is_inside & (cover_count == 1)[None, :]).sum(axis=1).astype(int)
        uncov_mask  = cover_count == 0
        return {**cov, 'is_inside': is_inside, 'cover_count': cover_count,
                'unique_coverage': unique_cov, 'total_coverage': is_inside.sum(axis=1).astype(int),
                'uncovered_mask': uncov_mask, 'n_uncovered': int(uncov_mask.sum())}

    @staticmethod
    def _select_prune_candidates(cov, budget):
        zero_unique = np.where(cov['unique_coverage'] == 0)[0]
        if len(zero_unique) == 0:
            return []
        order = np.argsort(cov['total_coverage'][zero_unique])
        return zero_unique[order].tolist()[:budget]

    # ── spawning ──────────────────────────────────────────────────────

    def _spawn_at_errors(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        num_spawn: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Place new ellipsoids inside the mesh, prioritising missed regions."""
        origin, dx, n = self._origin, self._dx, self._n

        ell_set = EllipsoidSet(device=device)
        if len(centers) > 0:
            ell_set.set_parameters(centers, radii, rotations)
        pred_grid = ell_set.compute_sdf_grid(
            origin, dx, n, shape=(self._nx, self._ny, self._nz))

        target_grid = self._sdf_target_np
        error = np.abs(_soft_clamp_np(pred_grid, 0.1) - _soft_clamp_np(target_grid, 0.1))

        flat_target = target_grid.ravel()
        flat_pred   = pred_grid.ravel()
        flat_error  = error.ravel()
        interior_mask = flat_target < 0.0
        interior_idx  = np.where(interior_mask)[0]

        if len(interior_idx) == 0:
            interior_idx = np.where(np.abs(flat_target) < 2.0 * dx)[0]
        if len(interior_idx) == 0:
            return (np.zeros((num_spawn, 3), dtype=np.float32),
                    np.full((num_spawn, 3), float(dx) * 3.0, dtype=np.float32),
                    np.tile(np.array([0., 0., 0., 1.], dtype=np.float32), (num_spawn, 1)))

        all_centers_list: list[np.ndarray] = []
        all_flat_idx_list: list[np.ndarray] = []
        spawned_so_far = 0

        # Priority tier: missed interior voxels (sdf_target<0, sdf_pred>0)
        missed_idx = np.where(interior_mask & (flat_pred > 0.0))[0]
        if len(missed_idx) > 0:
            n_prio    = min(num_spawn, len(missed_idx))
            pool_size = min(n_prio * 50, len(missed_idx))
            top_local = np.argpartition(flat_error[missed_idx], -pool_size)[-pool_size:]
            pool_idx  = missed_idx[top_local]
            iz, iy, ix = np.unravel_index(pool_idx, self._shape)
            pool_world = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx
            selected = self._farthest_point_sample(pool_world, flat_error[pool_idx], n_prio, centers)
            all_centers_list.append(pool_world[selected].astype(np.float32))
            all_flat_idx_list.append(pool_idx[selected])
            spawned_so_far = len(selected)

        # Fill tier: remaining slots from high-error interior voxels
        n_fill = num_spawn - spawned_so_far
        if n_fill > 0:
            pool_size = min(n_fill * 50, len(interior_idx))
            top_local = np.argpartition(flat_error[interior_idx], -pool_size)[-pool_size:]
            pool_flat = interior_idx[top_local]
            iz, iy, ix = np.unravel_index(pool_flat, self._shape)
            pool_world = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx
            existing = np.concatenate([centers] + all_centers_list, axis=0) if all_centers_list else centers
            selected = self._farthest_point_sample(pool_world, flat_error[pool_flat], n_fill, existing)
            all_centers_list.append(pool_world[selected].astype(np.float32))
            all_flat_idx_list.append(pool_flat[selected])

        new_centers    = np.concatenate(all_centers_list, axis=0)
        spawn_flat_idx = np.concatenate(all_flat_idx_list, axis=0)
        local_depth    = np.abs(flat_target[spawn_flat_idx])
        init_r = np.clip(local_depth * 0.6, float(dx) * 2.0, None)
        new_radii = np.stack([init_r, init_r, init_r], axis=1).astype(np.float32)
        new_rots  = np.tile(np.array([0., 0., 0., 1.], dtype=np.float32), (len(new_centers), 1))
        return new_centers, new_radii, new_rots

    # ══════════════════════════════════════════════════════════════════
    # SUPERFIT — residual-region detection + isolated local fitting
    # ══════════════════════════════════════════════════════════════════

    def _build_detection_grid(self) -> dict | None:
        """Resolution-capped copy of the target/thickness grids for region
        detection, cached across cycles (the target grid never changes).

        Returns ``None`` when the grid is already at/under the cap (no
        downsampling needed → detection runs on the full grid as before).
        The coarse grid keeps voxel-centre alignment with the fine grid so
        ``seed_world`` stays in the same world frame.
        """
        longest = max(self._shape)
        f = int(np.ceil(longest / float(self._region_detect_cap)))
        if f <= 1:
            return None
        if self._det_cache is not None and self._det_cache['f'] == f:
            return self._det_cache

        target = np.ascontiguousarray(self._sdf_target_np[::f, ::f, ::f])
        thickness = (np.ascontiguousarray(self._thickness_np[::f, ::f, ::f])
                     if self._thickness_np is not None else None)
        dx = float(self._dx) * f
        # coarse centre(i) == fine centre(i·f)  ⇒  origin shifts by −½·dx·(f−1)
        origin = (self._origin.astype(np.float64)
                  - 0.5 * float(self._dx) * (f - 1)).astype(self._origin.dtype)
        self._det_cache = dict(
            f=f, target=target, thickness=thickness, dx=dx, origin=origin,
        )
        return self._det_cache

    @contextlib.contextmanager
    def _detection_grid_scope(self):
        """Temporarily swap the grid geometry to the capped detection grid so
        ``_detect_worst_regions`` (and the predicted-grid build inside it) run
        in O(cap³) instead of O(n³).  Restores everything on exit.

        ``region_radius_vox`` is rescaled by 1/f so the world-space suppression
        radius between successive seeds is unchanged by the downsampling.
        """
        det = self._build_detection_grid()
        if det is None:
            yield
            return
        saved = (self._sdf_target_np, self._thickness_np, self._origin,
                 self._dx, self._n, self._nz, self._ny, self._nx, self._shape,
                 self._region_radius_vox)
        try:
            self._sdf_target_np = det['target']
            self._thickness_np = det['thickness']
            self._origin = det['origin']
            self._dx = det['dx']
            self._nz, self._ny, self._nx = det['target'].shape
            self._shape = det['target'].shape
            self._n = max(self._shape)
            self._region_radius_vox = self._region_radius_vox / det['f']
            yield
        finally:
            (self._sdf_target_np, self._thickness_np, self._origin,
             self._dx, self._n, self._nz, self._ny, self._nx, self._shape,
             self._region_radius_vox) = saved

    def _pred_grid_from_params(self, centers, radii, rotations) -> np.ndarray:
        ell_set = EllipsoidSet(device=device)
        if len(centers) > 0:
            ell_set.set_parameters(centers, radii, rotations)
        return ell_set.compute_sdf_grid(
            self._origin, self._dx, self._n, sdf_mode=self._sdf_mode,
            shape=(self._nx, self._ny, self._nz))

    def _detect_worst_regions(self, centers, radii, rotations, k,
                              min_severity: float = 0.0):
        """Find up to ``k`` *spatially-separated* under-represented regions.

        Greedy peak picking on the severity grid: take the argmax, record its
        seed + local interior pool, then suppress a wider ball around it so the
        next pick lands on a different part of the mesh.  Returns a list of
        region dicts (worst-first, possibly empty), each with:
          - ``seed_world``  : (3,) world position of the peak-severity voxel
          - ``pool_flat``   : flat voxel indices of the local interior pool
          - ``seed_depth``  : local feature thickness at the seed (|target|)
          - ``severity``    : peak severity value

        ``min_severity`` stops the search as soon as the next-worst peak falls
        below it (severity = relative miss × surface emphasis).  The grid assigns
        a non-zero severity to *any* voxel missed by ≥ half a voxel, so without a
        floor the picker keeps surfacing marginal, essentially-covered regions;
        the floor restricts it to genuinely under-represented ones.
        """
        nx, ny, nz = self._nx, self._ny, self._nz
        dx, origin = self._dx, self._origin
        pred_grid = self._pred_grid_from_params(centers, radii, rotations)

        sev = relative_underrep_grid(
            self._sdf_target_np, pred_grid, dx,
            surface_weight=self._surface_weight,
            surface_sigma_vox=max(self._surface_sigma / max(dx, 1e-12), 1e-6),
            min_gap_vox=self._underrep_min_gap_vox,
            thickness_grid=self._thickness_np,
            min_thickness_vox=self._underrep_min_thickness_vox,
        )
        flat_sev = sev.ravel()                 # view: suppression below edits sev
        flat_target = self._sdf_target_np.ravel()
        radius = float(self._region_radius_vox)

        def _ball_flat(cx, cy, cz, r):
            rr = int(np.ceil(r))
            lo = np.maximum(np.array([cx, cy, cz]) - rr, 0)
            hi = np.minimum(np.array([cx, cy, cz]) + rr + 1,
                            np.array([nx, ny, nz]))
            gx, gy, gz = np.meshgrid(
                np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
                np.arange(lo[2], hi[2]), indexing="ij",
            )
            d = np.sqrt((gx.ravel() - cx) ** 2 + (gy.ravel() - cy) ** 2
                        + (gz.ravel() - cz) ** 2)
            flat = (gz.ravel() * (nx * ny) + gy.ravel() * nx
                    + gx.ravel()).astype(np.int64)
            return flat[d <= r]

        regions = []
        floor = max(0.0, float(min_severity))
        for _ in range(int(k)):
            seed_flat = int(np.argmax(flat_sev))
            peak = float(flat_sev[seed_flat])
            if peak <= floor:
                break
            siz, siy, six = np.unravel_index(seed_flat, self._shape)
            seed_world = (origin.astype(np.float32)
                          + (np.array([six, siy, siz], dtype=np.float32) + 0.5) * float(dx))

            ball = _ball_flat(six, siy, siz, radius)
            pool_flat = ball[flat_target[ball] < 0.0]
            if pool_flat.size == 0:
                pool_flat = np.array([seed_flat], dtype=np.int64)

            regions.append(dict(
                seed_world=seed_world.astype(np.float32),
                pool_flat=pool_flat.astype(np.int32),
                seed_depth=float(abs(flat_target[seed_flat])),
                severity=peak,
            ))

            # Suppress a wider ball so the next seed is a different region.
            flat_sev[_ball_flat(six, siy, siz, radius * 2.0)] = 0.0

        return regions

    def _spawn_in_regions(self, regions, budget):
        """Spawn new ellipsoids at under-represented region seeds.

        Each new ellipsoid is an isotropic sphere centred on the region's
        peak-severity interior voxel with radius bounded by that voxel's
        interior depth (``|SDF|``), so it is **fully inside the mesh**: a ball of
        radius ``r ≤ |SDF(p)|`` around an interior point ``p`` contains no surface.
        Regions too shallow to host a meaningful ellipsoid are skipped.

        Returns ``(centers, radii, rotations, region_sites)`` where each
        ``region_site`` is ``(centre_world, half_extent)`` for the optional local
        fit.  ``centers`` is empty when nothing was spawned.
        """
        dx = float(self._dx)
        empty = (np.empty((0, 3), np.float32), np.empty((0, 3), np.float32),
                 np.empty((0, 4), np.float32), [])
        if budget <= 0 or not regions:
            return empty

        min_depth = 1.0 * dx                 # need room for a real ellipsoid
        inside_frac = 0.8                    # radius as a fraction of the depth
        r_region_world = float(self._region_radius_vox) * dx

        cs, rs, qs, sites = [], [], [], []
        for reg in regions:
            if len(cs) >= int(budget):
                break
            depth = float(reg["seed_depth"])
            if depth < min_depth:
                continue
            # r < depth ⇒ the whole sphere stays inside the mesh.
            rad = float(min(inside_frac * depth, depth - 0.5 * dx))
            if rad <= 0.0:
                continue
            c = np.asarray(reg["seed_world"], dtype=np.float32)
            cs.append(c)
            rs.append(np.array([rad, rad, rad], dtype=np.float32))
            qs.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            sites.append((c.copy(), max(r_region_world, rad * 2.0)))

        if not cs:
            return empty
        return (np.asarray(cs, dtype=np.float32),
                np.asarray(rs, dtype=np.float32),
                np.asarray(qs, dtype=np.float32),
                sites)

    @staticmethod
    def _farthest_point_sample(
        candidates: np.ndarray,
        errors: np.ndarray,
        k: int,
        existing_centers: np.ndarray,
    ) -> np.ndarray:
        n_cand = len(candidates)
        if n_cand == 0 or k == 0:
            return np.array([], dtype=int)

        if len(existing_centers) > 0:
            dists = np.linalg.norm(
                candidates[:, np.newaxis, :] - existing_centers[np.newaxis, :, :],
                axis=2,
            )
            min_dists = dists.min(axis=1)
        else:
            min_dists = np.full(n_cand, 1e6, dtype=np.float32)

        selected = []
        for _ in range(k):
            scores = min_dists * (errors + 1e-8)
            best = int(np.argmax(scores))
            selected.append(best)
            new_dists = np.linalg.norm(candidates - candidates[best], axis=1)
            min_dists = np.minimum(min_dists, new_dists)

        return np.array(selected, dtype=int)

    # ══════════════════════════════════════════════════════════════════
    # TRAINING LOOPS
    # ══════════════════════════════════════════════════════════════════

    # ── SuperFit: spawn at worst region + isolated local fit ──────────

    def _grid_value(self, grid: np.ndarray, world_pt: np.ndarray) -> float:
        """Nearest-voxel value of a (nz, ny, nx) grid at a world-space point."""
        q = (np.asarray(world_pt, dtype=np.float32) - self._origin) / float(self._dx)
        hi = np.array([self._nx - 1, self._ny - 1, self._nz - 1])
        ijk = np.clip(np.floor(q).astype(np.int64), 0, hi)
        return float(grid[ijk[2], ijk[1], ijk[0]])

    def _interior_ball_pool(self, world_center: np.ndarray, radius_vox: float) -> np.ndarray:
        """Flat indices of interior voxels within ``radius_vox`` of a world point."""
        nx, ny, nz = self._nx, self._ny, self._nz
        q = (np.asarray(world_center, dtype=np.float32) - self._origin) / float(self._dx)
        cx, cy, cz = np.clip(np.floor(q).astype(np.int64), 0,
                             np.array([nx - 1, ny - 1, nz - 1]))
        r = float(radius_vox)
        rr = int(np.ceil(r))
        lo = np.maximum(np.array([cx, cy, cz]) - rr, 0)
        hi = np.minimum(np.array([cx, cy, cz]) + rr + 1, np.array([nx, ny, nz]))
        gx, gy, gz = np.meshgrid(
            np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
            np.arange(lo[2], hi[2]), indexing="ij",
        )
        d = np.sqrt((gx.ravel() - cx) ** 2 + (gy.ravel() - cy) ** 2 + (gz.ravel() - cz) ** 2)
        flat = (gz.ravel() * (nx * ny) + gy.ravel() * nx + gx.ravel()).astype(np.int64)
        flat = flat[d <= r]
        flat_target = self._sdf_target_np.ravel()
        return flat[flat_target[flat] < 0.0]

    def _detect_protruding_ellipsoids(self, centers, radii, rotations) -> np.ndarray:
        """Indices of ellipsoids that stick out past the mesh surface *because
        they are too big in that direction*.  Returned worst-first.

        For each ellipsoid we sample the target SDF at its six principal-axis
        tips (center ± semi-axis·axis).  A tip with ``target > 0`` lies outside
        the mesh.  But a tip grazing the surface is normal and *good* — a
        well-fitting ellipsoid's surface coincides with the mesh, so its tips sit
        right at it.  The old test flagged any tip poking out by half a voxel and
        then gated on ``max_r`` vs. the local thickness; for an elongated
        ellipsoid (a limb / finger) the long axis is naturally far larger than
        the cross-section, so that gate was always true and virtually every good
        elongated fit was flagged.

        Instead we require the overshoot to be a meaningful fraction of *that
        tip's own semi-axis*: ``t / r[k] > rel_thresh``.  This is exactly "the
        ellipsoid is at least ``split_size_factor`` too long along axis k" and is
        scale- and elongation-invariant — a 1-voxel poke on a 20-voxel axis is
        ignored, while a genuinely oversized axis is caught.  ``rel_thresh`` is
        derived from the same ``split_size_factor`` knob: for an axis oversized
        by factor f the centred overshoot is ``(f−1)/f`` of the semi-axis.
        """
        target = self._sdf_target_np
        dx = float(self._dx)
        margin = float(self._split_margin_vox) * dx          # sub-voxel-noise floor
        min_r = float(self._min_split_radius_vox) * dx
        f = max(float(self._split_size_factor), 1.0 + 1e-6)
        rel_thresh = 1.0 - 1.0 / f                            # (f−1)/f

        idx, scores = [], []
        for i in range(len(centers)):
            c = centers[i].astype(np.float32)
            r = radii[i].astype(np.float32)
            Rm = _quat_to_rot_matrix(rotations[i]).astype(np.float32)

            protr = 0.0
            for k in range(3):
                rk = float(r[k])
                if rk <= 0.0:
                    continue
                axis = Rm[:, k]
                for s in (1.0, -1.0):
                    t = self._grid_value(target, c + s * rk * axis)
                    # Outside by both an absolute (noise) and a relative
                    # (vs this semi-axis) margin → genuinely oversized here.
                    if t > margin and (t / rk) > rel_thresh:
                        protr = max(protr, t / rk)
            if protr <= 0.0:
                continue
            if float(np.max(r)) < min_r:
                continue

            idx.append(i)
            scores.append(protr)

        if not idx:
            return np.array([], dtype=int)
        order = np.argsort(-np.asarray(scores))
        return np.asarray(idx, dtype=int)[order]

    def _detect_bridging_ellipsoids(self, centers, radii, rotations) -> np.ndarray:
        """Indices of large ellipsoids that bridge several separate structures.

        A single ellipsoid stretched across two (or more) nearby features has a
        chunk of its *interior* sitting in the empty gap between them — i.e. a
        significant fraction of its interior probe points fall OUTSIDE the mesh,
        even though the ellipsoid as a whole is not fully outside (those get
        deleted earlier in the cycle).  The 6-tip protrusion test misses this
        because the axis tips can land inside the structures while the middle
        straddles the gap.  Such an ellipsoid should be split so each half can
        settle onto one structure.  Returned worst-(most-bridging)-first.
        """
        n_ell = len(centers)
        if n_ell == 0:
            return np.array([], dtype=int)

        unit = self._unit_ball_samples()                  # (M, 3) interior probe
        target = self._sdf_target_np
        thick = self._thickness_np
        origin = self._origin.astype(np.float32)
        dx = float(self._dx)
        n = self._n
        margin = float(self._split_margin_vox) * dx
        min_r = float(self._min_split_radius_vox) * dx

        idx, scores = [], []
        for i in range(n_ell):
            r = radii[i].astype(np.float32)
            max_r = float(np.max(r))
            if max_r < min_r:
                continue
            Rm = _quat_to_rot_matrix(rotations[i]).astype(np.float32)
            pts = centers[i].astype(np.float32) + (unit * r) @ Rm.T
            q = (pts - origin) / dx
            ijk = np.clip(np.floor(q).astype(np.int64), 0,
                          np.array([self._nx - 1, self._ny - 1, self._nz - 1]))
            vals = target[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
            outside_frac = float(np.mean(vals > margin))
            if outside_frac < float(self._bridge_min_outside):
                continue
            # Must be oversized vs. the local feature thickness, so a small
            # ellipsoid merely poking past a surface is left to the protrusion
            # test instead.  (When the centre is in the gap, thickness there is
            # 0 → oversized stays True, which is exactly the bridging case.)
            oversized = True
            if thick is not None:
                th_c = self._grid_value(thick, centers[i])
                if th_c > 0.0:
                    oversized = max_r > self._split_size_factor * (0.5 * th_c)
            if not oversized:
                continue
            idx.append(i)
            scores.append(outside_frac)

        if not idx:
            return np.array([], dtype=int)
        order = np.argsort(-np.asarray(scores))
        return np.asarray(idx, dtype=int)[order]

    def _split_ellipsoid(self, c, r, q):
        """Halve an ellipsoid along its longest axis → two child ellipsoids.

        The children tile the parent exactly: each gets half the longest
        semi-axis and is offset by ±half that semi-axis along it, so together
        they span the original extent with no gap.  Rotation is preserved.
        """
        Rm = _quat_to_rot_matrix(q).astype(np.float32)
        k = int(np.argmax(r))
        axis = Rm[:, k].astype(np.float32)
        half = 0.5 * float(r[k])
        off = half * axis
        c1 = (np.asarray(c, np.float32) + off).astype(np.float32)
        c2 = (np.asarray(c, np.float32) - off).astype(np.float32)
        rc = np.asarray(r, np.float32).copy()
        rc[k] = half
        new_c = np.stack([c1, c2]).astype(np.float32)
        new_r = np.stack([rc, rc]).astype(np.float32)
        new_q = np.stack([np.asarray(q, np.float32), np.asarray(q, np.float32)]).astype(np.float32)
        return new_c, new_r, new_q

    def _split_targets_for_regions(self, centers, regions, budget,
                                   exclude=None, n_protect=0):
        """Pick which existing ellipsoids to *split* to cover under-represented
        regions — the densification path that replaces random spawning.

        For each region (worst-first) the nearest still-eligible ellipsoid is
        chosen; it will be split along its longest semi-axis.  An ellipsoid is
        eligible if its index is ``>= n_protect`` (frozen contributors are off
        limits) and it has not already been picked.  Returns an ordered, unique
        list of indices, capped at ``budget``.
        """
        used = set(int(i) for i in (exclude or set()))
        targets: list[int] = []
        if budget <= 0 or not regions or len(centers) == 0:
            return targets
        cen = np.asarray(centers, dtype=np.float32)
        for region in regions:
            if len(targets) >= budget:
                break
            seed = np.asarray(region['seed_world'], dtype=np.float32)
            d = np.linalg.norm(cen - seed[None, :], axis=1)
            for j in np.argsort(d):
                j = int(j)
                if j < n_protect or j in used:
                    continue
                used.add(j)
                targets.append(j)
                break
        return targets

    def _densify_regions(self, centers, radii, regions, budget, exclude=None,
                         split_enabled=True, spawn_enabled=True):
        """Per under-represented region: **split** the nearest existing ellipsoid
        if one is close, otherwise **spawn** a new (fully-inside) ellipsoid.

        "Close" means the nearest eligible ellipsoid's surface lies within one
        region radius of the region seed — then extending it (split) is better
        than seeding a fresh primitive; an isolated gap with no ellipsoid nearby
        gets a spawn instead.  ``split_enabled`` / ``spawn_enabled`` gate the two
        mechanisms; when only one is on it handles every region.  Returns
        ``(split_targets, spawn_regions)``.
        """
        split_targets: list[int] = []
        spawn_regions: list[dict] = []
        if budget <= 0 or not regions or not (split_enabled or spawn_enabled):
            return split_targets, spawn_regions

        used = set(int(i) for i in (exclude or set()))
        cen = np.asarray(centers, dtype=np.float32)
        mean_r = radii.mean(axis=1) if len(radii) else np.zeros(0, dtype=np.float32)
        near_world = float(self._region_radius_vox) * float(self._dx)

        for region in regions:
            if len(split_targets) + len(spawn_regions) >= int(budget):
                break
            seed = np.asarray(region['seed_world'], dtype=np.float32)
            j = None
            d = None
            if len(cen) > 0:
                dist = np.linalg.norm(cen - seed[None, :], axis=1)
                for k in np.argsort(dist):
                    k = int(k)
                    if k not in used:
                        j, d = k, float(dist[k])
                        break
            near = j is not None and (d - float(mean_r[j])) < near_world

            # Prefer split for a nearby ellipsoid, spawn for an isolated gap;
            # fall back to whichever mechanism is enabled.
            if near and split_enabled:
                used.add(j)
                split_targets.append(j)
            elif spawn_enabled:
                spawn_regions.append(region)
            elif split_enabled and j is not None:
                used.add(j)
                split_targets.append(j)
            # else: both disabled here → leave the region for the optimiser
        return split_targets, spawn_regions

    def _unit_ball_samples(self) -> np.ndarray:
        """Cached (M, 3) points uniformly filling the unit ball (interior probe)."""
        if self._fuse_unit_pts is None:
            m = int(self._fuse_samples)
            rng = np.random.default_rng(0)        # deterministic probe cloud
            dirs = rng.normal(size=(m, 3)).astype(np.float32)
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
            rad = np.cbrt(rng.random(m)).astype(np.float32)[:, None]
            self._fuse_unit_pts = (dirs * rad).astype(np.float32)
        return self._fuse_unit_pts

    # ── merge step ──────────────────────────────────────────────────────

    def _unit_sphere_samples(self) -> np.ndarray:
        """Cached (S, 3) points on the unit sphere (surface probe for merging)."""
        if self._merge_sphere_pts is None:
            s = max(64, int(self._fuse_samples))
            rng = np.random.default_rng(1)
            dirs = rng.normal(size=(s, 3)).astype(np.float32)
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
            self._merge_sphere_pts = dirs.astype(np.float32)
        return self._merge_sphere_pts

    @staticmethod
    def _ellipsoid_k(points, c, r, R) -> np.ndarray:
        """Normalised radial coordinate of ``points`` w.r.t. an ellipsoid.

        ``k <= 1`` inside, ``k == 1`` on the surface (``R`` = rotation matrix from
        :func:`_quat_to_rot_matrix`; local = (world − c) · R).
        """
        local = (points.astype(np.float32) - c.astype(np.float32)) @ R.astype(np.float32)
        qd = local / np.maximum(r.astype(np.float32), 1e-9)
        return np.sqrt(np.sum(qd * qd, axis=1))

    def _merge_two_ellipsoids(self, i, j, centers, radii, rotations):
        """Single ellipsoid matching the volume-weighted second moments of the
        union of ellipsoids ``i`` and ``j``.

        Computed analytically (no sampling noise): each uniform ellipsoid has
        centroid ``c`` and world covariance ``R·diag(r²/5)·Rᵀ``.  The combined
        mass mean and covariance give the merged centre / orientation, and the
        eigenvalues map back to radii via ``a = sqrt(5·λ)``.
        """
        ci = centers[i].astype(np.float64)
        cj = centers[j].astype(np.float64)
        ri = radii[i].astype(np.float64)
        rj = radii[j].astype(np.float64)
        Ri = _quat_to_rot_matrix(rotations[i])
        Rj = _quat_to_rot_matrix(rotations[j])

        vi = float(np.prod(np.maximum(ri, 1e-9)))   # ∝ volume
        vj = float(np.prod(np.maximum(rj, 1e-9)))
        wsum = vi + vj
        c_m = (vi * ci + vj * cj) / wsum

        cov_i = Ri @ np.diag(ri * ri / 5.0) @ Ri.T
        cov_j = Rj @ np.diag(rj * rj / 5.0) @ Rj.T
        di = (ci - c_m)[:, None]
        dj = (cj - c_m)[:, None]
        M = (vi * (cov_i + di @ di.T) + vj * (cov_j + dj @ dj.T)) / wsum

        w, V = np.linalg.eigh(M)              # ascending eigenvalues, columns = axes
        w = np.maximum(w, 1e-12)
        if np.linalg.det(V) < 0.0:            # keep a proper (right-handed) rotation
            V[:, 0] = -V[:, 0]
        r_m = np.sqrt(5.0 * w).astype(np.float32)
        q_m = _rot_matrix_to_quat(V)
        return c_m.astype(np.float32), r_m, q_m

    def _merge_changes_surface(self, i, j, c_m, r_m, q_m,
                               centers, radii, rotations) -> bool:
        """True if replacing ``i, j`` by the merged ellipsoid moves the surface.

        Two-sided boundary check in normalised (radius) units:
          * every union-boundary point (surface of one, *outside* the other)
            should land on the merged surface (``k_m ≈ 1``);
          * every merged-surface point should land on the union boundary
            (``min(k_i, k_j) ≈ 1``).
        The pair must actually overlap.  Returns ``False`` (→ safe to merge) only
        when both worst-case errors are within ``self._merge_tol``.
        """
        Ri = _quat_to_rot_matrix(rotations[i]).astype(np.float32)
        Rj = _quat_to_rot_matrix(rotations[j]).astype(np.float32)
        Rm = _quat_to_rot_matrix(q_m).astype(np.float32)
        sph = self._unit_sphere_samples()

        si = centers[i].astype(np.float32) + (sph * radii[i].astype(np.float32)) @ Ri.T
        sj = centers[j].astype(np.float32) + (sph * radii[j].astype(np.float32)) @ Rj.T
        ki_in_j = self._ellipsoid_k(si, centers[j], radii[j], Rj)
        kj_in_i = self._ellipsoid_k(sj, centers[i], radii[i], Ri)

        # Must overlap (some surface point of one lies inside the other).
        if ki_in_j.min() > 1.0 and kj_in_i.min() > 1.0:
            return True

        bound = np.vstack([si[ki_in_j > 1.0], sj[kj_in_i > 1.0]])
        if len(bound) == 0:                   # one fully contains the other → fuse, not merge
            return True
        km = self._ellipsoid_k(bound, c_m, r_m, Rm)
        err_a = float(np.max(np.abs(km - 1.0)))

        sm = c_m.astype(np.float32) + (sph * r_m) @ Rm.T
        ku = np.minimum(self._ellipsoid_k(sm, centers[i], radii[i], Ri),
                        self._ellipsoid_k(sm, centers[j], radii[j], Rj))
        err_b = float(np.max(np.abs(ku - 1.0)))

        return max(err_a, err_b) > float(self._merge_tol)

    def _merge_increases_loss(self, i, j, c_m, r_m, q_m,
                              centers, radii, rotations,
                              n_samples: int = 1500, rel_eps: float = 0.10) -> bool:
        """True if replacing ``i, j`` by the merged ellipsoid raises the fit loss.

        Compares the union SDF against the target SDF, before vs after the merge,
        over random points in the pair's bounding box.  ``others`` (the union of
        all *other* ellipsoids) is unchanged by the merge, so only ``i, j`` vs the
        merged primitive affect the loss.  Only candidates that already passed the
        cheap geometric gates reach here, so this runs rarely.
        """
        origin = self._origin.astype(np.float64)
        dx = float(self._dx)
        nx, ny, nz = self._nx, self._ny, self._nz

        cs = np.stack([centers[i], centers[j], c_m]).astype(np.float64)
        rs = np.array([np.max(np.abs(radii[i])), np.max(np.abs(radii[j])),
                       np.max(np.abs(r_m))], dtype=np.float64)
        lo = (cs - rs[:, None]).min(axis=0)
        hi = (cs + rs[:, None]).max(axis=0)

        rng = np.random.default_rng(int(i) * 131071 + int(j))   # per-pair, stable
        pts = (lo[None, :] + rng.random((n_samples, 3))
               * (hi - lo)[None, :]).astype(np.float32)

        q = (pts.astype(np.float64) - origin) / dx
        ijk = np.clip(np.floor(q).astype(np.int64), 0,
                      np.array([nx - 1, ny - 1, nz - 1]))
        target = self._sdf_target_np[ijk[:, 2], ijk[:, 1], ijk[:, 0]].astype(np.float32)

        # Other ellipsoids that can be nearest inside this box (prefilter).
        box_c = 0.5 * (lo + hi)
        box_r = 0.5 * float(np.linalg.norm(hi - lo))
        others = np.full(n_samples, 1e6, dtype=np.float32)
        for k in range(len(centers)):
            if k == i or k == j:
                continue
            if np.linalg.norm(centers[k].astype(np.float64) - box_c) \
                    > box_r + float(np.max(np.abs(radii[k]))):
                continue
            others = np.minimum(
                others, self._ellipsoid_sdf_np(centers[k], radii[k], rotations[k], pts))

        sdf_i = self._ellipsoid_sdf_np(centers[i], radii[i], rotations[i], pts)
        sdf_j = self._ellipsoid_sdf_np(centers[j], radii[j], rotations[j], pts)
        sdf_m = self._ellipsoid_sdf_np(c_m, r_m, q_m, pts)

        before = np.minimum(others, np.minimum(sdf_i, sdf_j))
        after = np.minimum(others, sdf_m)
        loss_before = float(np.mean((before - target) ** 2))
        loss_after = float(np.mean((after - target) ** 2))
        # Reject if the loss grows by more than a relative margin AND a small
        # absolute floor (so a negligible absolute rise on an already-good fit
        # doesn't block a genuine redundancy merge).
        abs_floor = (0.1 * dx) ** 2
        return (loss_after - loss_before) > max(rel_eps * loss_before, abs_floor)

    def _detect_merges(self, centers, radii, rotations):
        """Merge overlapping ellipsoid pairs whose fusion barely moves the surface.

        Greedy, most-overlapping pair first; each ellipsoid is used at most once
        per round and at most ``self._merge_per_round`` merges are applied.
        Returns ``(centers, radii, rotations, n_merged)`` — possibly unchanged.
        """
        n = len(centers)
        if not self._merge_enabled or n <= 1 or self._merge_per_round <= 0:
            return centers, radii, rotations, 0

        # Prefilter: keep only pairs that actually overlap along their line of
        # centres.  The half-extent (support) of an ellipsoid along a unit
        # direction ``u`` is ``sqrt(uᵀ·Σ·u)`` with ``Σ = R·diag(r²)·Rᵀ``.  Using
        # this directional reach instead of a mean radius makes the test correct
        # for *elongated* ellipsoids — collinear finger spheres/capsules whose
        # long axes overlap end-to-end are now detected as merge candidates
        # (a mean-radius test underestimates the long-axis reach and misses
        # them, which is why finger sphere-chains were never consolidated).
        sigma = []
        for k in range(n):
            Rk = _quat_to_rot_matrix(rotations[k]).astype(np.float64)
            rk = radii[k].astype(np.float64)
            sigma.append(Rk @ np.diag(rk * rk) @ Rk.T)

        cand = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (centers[i].astype(np.float64)
                        - centers[j].astype(np.float64))
                d = float(np.linalg.norm(diff))
                if d < 1e-9:
                    cand.append((-1e9, i, j))     # coincident → maximal overlap
                    continue
                u = diff / d
                reach_i = float(np.sqrt(max(u @ sigma[i] @ u, 0.0)))
                reach_j = float(np.sqrt(max(u @ sigma[j] @ u, 0.0)))
                slack = d - (reach_i + reach_j)
                if slack < 0.0:                   # surfaces overlap along centre line
                    cand.append((slack, i, j))
        if not cand:
            return centers, radii, rotations, 0
        cand.sort()                            # most overlapping (most negative) first

        consumed: set[int] = set()
        merged_c, merged_r, merged_q = [], [], []
        n_merged = 0
        for _, i, j in cand:
            if n_merged >= int(self._merge_per_round):
                break
            if i in consumed or j in consumed:
                continue
            c_m, r_m, q_m = self._merge_two_ellipsoids(i, j, centers, radii, rotations)
            # Shape-aware guard (replaces the old total-volume guard).  Merging
            # two *collinear* primitives into one ellipsoid is the desired result
            # for a finger sphere-chain — the merged ellipsoid is elongated along
            # the offset (a capsule), so its *volume* necessarily exceeds either
            # input and the old volume guard rejected exactly the merges we want.
            #
            # The runaway it protected against was *isotropic* ballooning (fusing
            # offset/separate blobs into a fat bounding ellipsoid).  That inflates
            # the CROSS-SECTION — the two smallest semi-axes — whereas a genuine
            # collinear merge only grows the long axis.  So gate on cross-section
            # growth: allow the long axis to extend freely, reject cross-section
            # inflation.  Faithfulness is still enforced by the surface- and
            # loss-gates below.
            def _cross_section(r):
                s = np.sort(np.abs(np.asarray(r, dtype=np.float64)))
                return float(s[0] * s[1])         # product of the two smallest
            cs_i = _cross_section(radii[i])
            cs_j = _cross_section(radii[j])
            cs_m = _cross_section(r_m)
            if cs_m > 1.25 * max(cs_i, cs_j):
                continue

            # Is this a *collinear / elongating* merge (a finger sphere-chain
            # consolidating into a capsule)?  Signal: the merged ellipsoid's long
            # axis is aligned with the line of centres, and the centres are
            # meaningfully separated.  For such merges the surface-equality gate
            # below is the wrong test — it compares the merged ellipsoid against
            # the *union of the two inputs* (a peanut), but we deliberately want
            # a smooth capsule that differs from that peanut while fitting the
            # *target* better.  The cross-section gate (above) already forbids
            # cross-section ballooning, and the loss gate (below) verifies the
            # fit against the target SDF, so those two suffice here.
            diff = (centers[i].astype(np.float64)
                    - centers[j].astype(np.float64))
            d = float(np.linalg.norm(diff))
            collinear = False
            if d > 1e-9:
                u = diff / d
                Rm = _quat_to_rot_matrix(q_m).astype(np.float64)
                long_axis = Rm[:, int(np.argmax(np.abs(r_m)))]
                align = abs(float(u @ long_axis))
                reach_i = float(np.sqrt(max(u @ sigma[i] @ u, 0.0)))
                reach_j = float(np.sqrt(max(u @ sigma[j] @ u, 0.0)))
                sep = d / max(reach_i + reach_j, 1e-9)
                collinear = (align > 0.85) and (sep > 0.30)

            if not collinear and self._merge_changes_surface(
                    i, j, c_m, r_m, q_m, centers, radii, rotations):
                continue
            # Final, decisive gate: only merge if it does not raise the fit loss.
            if self._merge_increases_loss(i, j, c_m, r_m, q_m,
                                          centers, radii, rotations):
                continue
            consumed.add(i)
            consumed.add(j)
            merged_c.append(c_m)
            merged_r.append(r_m)
            merged_q.append(q_m)
            n_merged += 1

        if n_merged == 0:
            return centers, radii, rotations, 0

        keep = [k for k in range(n) if k not in consumed]
        out_c = np.vstack([centers[keep], np.asarray(merged_c, dtype=np.float32)])
        out_r = np.vstack([radii[keep], np.asarray(merged_r, dtype=np.float32)])
        out_q = np.vstack([rotations[keep], np.asarray(merged_q, dtype=np.float32)])
        return (out_c.astype(np.float32), out_r.astype(np.float32),
                out_q.astype(np.float32), n_merged)

    def _detect_redundant_ellipsoids(self, centers, radii, rotations, k_max) -> np.ndarray:
        """Up to ``k_max`` ellipsoids whose interior is covered by the others.

        An ellipsoid "has no independent task" when (almost) every point of its
        interior also lies inside at least one *other surviving* ellipsoid — its
        region is redundant, so dropping it leaves the union SDF essentially
        unchanged.  We probe each ellipsoid's interior with a fixed unit-ball
        point cloud (scaled/rotated/translated into world space) and test those
        points against the others' inside-test.

        Removal is **greedy, smallest-volume-first**, re-checking each candidate
        against the *survivors only*.  This collapses overlapping pairs onto the
        larger primitive without ever removing both members of a mutual pair
        (which would punch a hole): once the smaller is dropped, the larger is no
        longer covered and is kept.
        """
        n_ell = len(centers)
        if n_ell <= 1 or k_max <= 0:
            return np.array([], dtype=int)

        unit = self._unit_ball_samples()                  # (M, 3)
        rmats = [_quat_to_rot_matrix(rotations[j]).astype(np.float32) for j in range(n_ell)]
        vols = np.prod(np.maximum(radii, 1e-9), axis=1)    # ∝ ellipsoid volume
        order = np.argsort(vols)                           # smallest first

        alive = np.ones(n_ell, dtype=bool)
        removed = []
        thr = float(self._fuse_overlap_frac)
        for i in order:
            if len(removed) >= int(k_max):
                break
            pts = centers[i].astype(np.float32) + (unit * radii[i].astype(np.float32)) @ rmats[i].T
            covered = np.zeros(len(pts), dtype=bool)
            for j in range(n_ell):
                if j == i or not alive[j]:
                    continue
                local = (pts - centers[j].astype(np.float32)) @ rmats[j]
                qd = local / np.maximum(radii[j].astype(np.float32), 1e-9)
                covered |= np.sum(qd * qd, axis=1) <= 1.0
                if covered.all():
                    break
            if float(covered.mean()) >= thr:
                alive[i] = False
                removed.append(int(i))

        return np.asarray(removed, dtype=int)

    def _detect_outside_ellipsoids(self, centers, radii, rotations) -> np.ndarray:
        """Indices of primitives that should be deleted as 'wasted', i.e. one of:

          (a) entirely outside the mesh — not a single interior probe point falls
              inside the target surface;
          (b) centre clearly outside the mesh (target SDF at the centre beyond a
              small margin) — catches a huge primitive sitting outside that only
              clips the mesh with one tip (the ellipsoid probe misses these for
              boxy / bent superquadrics);
          (c) oversized — a radius larger than 60 % of the grid extent, which can
              only ever be a runaway blob covering everything.

        (b)+(c) are what kill the "very large primitives far outside the mesh"
        that the radial pseudo-distance fails to penalise.
        """
        n_ell = len(centers)
        if n_ell == 0:
            return np.array([], dtype=int)

        unit = self._unit_ball_samples()                  # (M, 3)
        target = self._sdf_target_np
        origin = self._origin.astype(np.float32)
        dx = float(self._dx)
        hi = np.array([self._nx - 1, self._ny - 1, self._nz - 1])
        extent = float(max(self._nx, self._ny, self._nz) * dx)
        center_margin = 3.0 * dx          # centre this far outside → delete
        max_radius = 0.6 * extent         # bigger than this → runaway, delete

        def _grid_val(world_pts):
            q = (np.asarray(world_pts, np.float32) - origin) / dx
            q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
            ijk = np.clip(np.floor(q), 0, hi).astype(np.int64)
            return target[ijk[..., 2], ijk[..., 1], ijk[..., 0]]

        outside = []
        for i in range(n_ell):
            r_i = radii[i].astype(np.float32)
            # (c) oversized runaway
            if float(np.max(r_i)) > max_radius:
                outside.append(int(i))
                continue
            # (b) centre clearly outside the mesh
            cval = float(_grid_val(centers[i].astype(np.float32)[None, :])[0])
            if cval > center_margin:
                outside.append(int(i))
                continue
            # (a) entirely outside (interior probe cloud)
            Rm = _quat_to_rot_matrix(rotations[i]).astype(np.float32)
            pts = centers[i].astype(np.float32) + (unit * r_i) @ Rm.T
            vals = _grid_val(pts)
            if np.all(vals > 0.0):                        # no probe point is inside
                outside.append(int(i))

        return np.asarray(outside, dtype=int)

    def _detect_degenerate_ellipsoids(self, radii: np.ndarray) -> np.ndarray:
        """Indices of ellipsoids whose shape is too degenerate to be useful.

        Two failure modes, both via scale-invariant axis ratios (sorted radii
        r_min ≤ r_mid ≤ r_max):
          * **flat** disk/pancake — the thin axis collapsed: ``r_min/r_mid`` below
            ``degenerate_flat_ratio``;
          * **pointy** needle/spike — one axis runs away: ``r_max/r_mid`` above
            ``degenerate_spike_ratio``.
        Both are compared to the *median* axis so genuine slender features (two
        short equal axes + one long) are judged only by how far past the spike
        threshold they go, not flagged as flat.
        """
        r = np.abs(np.asarray(radii, dtype=np.float64))
        if len(r) == 0:
            return np.array([], dtype=int)
        rmax = r.max(axis=1)
        rmin = r.min(axis=1)
        rmid = np.maximum(r.sum(axis=1) - rmax - rmin, 1e-9)
        flat = (rmin / rmid) < float(self._degenerate_flat_ratio)
        spiky = (rmax / rmid) > float(self._degenerate_spike_ratio)
        return np.where(flat | spiky)[0]

    def _local_fit(self, centers, radii, rotations, offset, pool_flat, gstep=-1):
        """SGD on the appended range [offset:] only, sampled from a local pool.

        Every ellipsoid before ``offset`` is frozen; gradients still flow but the
        range-restricted step kernels leave them untouched.  Returns updated
        (centers, radii, rotations) as numpy arrays.

        Emits ``local_progress`` and a lightweight ``step_visual`` a handful of
        times so the UI keeps updating during the (long) local fit instead of
        appearing frozen.
        """
        origin, dx = self._origin, self._dx
        nx, ny, nz = self._nx, self._ny, self._nz
        total = nx * ny * nz
        self._ensure_thickness_wp(total)
        num_e = len(centers)
        n_active = num_e - offset
        if n_active <= 0 or pool_flat.size == 0:
            return centers, radii, rotations

        bs = int(min(self._batch_size, 4096))
        buf = self._alloc_buffers(num_e, bs, total, centers, radii, rotations)
        pred_centers  = buf['pred_centers']
        pred_radii    = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']
        min_d_cache   = buf['min_d_cache']
        sdf_pred      = buf['sdf_pred']
        loss          = buf['loss']
        sdf_target    = buf['sdf_target']
        wp_indices    = buf['wp_indices']

        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = float(self._local_lr)
        rot_offset = offset * 4
        report_every = max(1, self._local_steps // 20)

        for li in range(self._local_steps):
            if self._stop_flag:
                break
            batch = self._rng.choice(pool_flat, size=bs, replace=True).astype(np.int32)
            wp_indices.assign(np.ascontiguousarray(batch))

            tape = wp.Tape()
            with tape:
                min_d_cache.zero_()
                if self._isotropic:
                    wp.launch(
                        _sphere_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                elif self._superquadric:
                    wp.launch(
                        _superquadric_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat,
                                self._uniform_eps_wp(num_e),
                                self._zero_bend_wp(num_e), min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                elif self._capsule:
                    wp.launch(
                        _capsule_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat, min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                else:
                    wp.launch(
                        _ellipsoid_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat, min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                loss.zero_()
                wp.launch(
                    _rmse_loss_kernel_batch,
                    dim=bs,
                    inputs=[sdf_pred, sdf_target, wp_indices, loss, bs,
                            float(self._miss_penalty_weight),
                            float(self._surface_weight), float(self._surface_sigma),
                            float(self._outside_penalty_weight),
                            self._wp_thickness, float(self._thick_ref),
                            float(self._thin_weight_eff), float(self._thin_max_factor)],
                    device=device,
                )
                if self._flat_weight > 0.0:
                    wp.launch(
                        _flatness_penalty_kernel,
                        dim=n_active,
                        inputs=[pred_radii, loss, n_active, offset,
                                float(self._flat_weight), float(self._flat_min_ratio)],
                        device=device,
                    )

            tape.backward(loss)

            wp.launch(_sgd_step_vec3_range, dim=n_active,
                      inputs=[pred_centers, tape.gradients[pred_centers], lr, offset],
                      device=device)
            wp.launch(_sgd_step_vec3_range, dim=n_active,
                      inputs=[pred_radii, tape.gradients[pred_radii], lr, offset],
                      device=device)
            wp.launch(_sgd_step_f32_range, dim=n_active * 4,
                      inputs=[pred_rot_flat, tape.gradients[pred_rot_flat], lr, rot_offset],
                      device=device)
            wp.launch(_normalize_flat_quats_range, dim=n_active,
                      inputs=[pred_rot_flat, offset], device=device)

            tape.zero()

            if li % report_every == 0 or li == self._local_steps - 1:
                wp.synchronize_device(device)
                c = pred_centers.numpy().copy()
                r = pred_radii.numpy().copy()
                q = pred_rot_flat.numpy().reshape(-1, 4).copy()
                self.local_progress.emit(li + 1, self._local_steps)
                self.step_visual.emit(int(gstep), float(loss.numpy()[0]), c, r, q, None)

        wp.synchronize_device(device)
        c = pred_centers.numpy().copy()
        r = pred_radii.numpy().copy()
        q = pred_rot_flat.numpy().reshape(-1, 4).copy()
        return c, r, q

    def _region_box(self, region_center, half_extent):
        """Padded world-space box for a region (pre-isotropic; the SdfComputer
        expands it to a cube).  Shared by the batch pre-compute and the fit so
        both agree on the box geometry.

        A SMALL, fixed region box — the whole point is a small high-res region.
        It is deliberately NOT grown to swallow large ellipsoids: padded only
        slightly so trainable ellipsoids near the edge stay fully covered.
        """
        c0 = np.asarray(region_center, dtype=np.float32)
        half = float(half_extent)
        pad = 2.0 * float(self._dx)
        return (c0 - half - pad).astype(np.float32), (c0 + half + pad).astype(np.float32)

    def _region_divide_conquer(self, contrib_c, contrib_r, contrib_q,
                               train_c, train_r, train_q, res, n_fixed):
        """One divide-and-conquer pass on the trainable set against the box grid.

        Runs the same adaptive-density moves as the global SuperFit cycle
        (delete-outside / fuse-redundant / split-oversized / spawn-underrep) but
        scoped to the high-res region box ``res`` and applied only to the
        *trainable* suffix; the frozen contributor prefix is protected from every
        move and merely shapes the union for the detectors.

        The existing numpy detectors all read ``self._{sdf_target_np,thickness_np,
        origin,dx,n}``, so we temporarily swap those to the box grid ("grid
        context") and restore them afterwards.

        Returns the (possibly resized) trainable (centers, radii, rotations).
        """
        n_contrib = int(contrib_c.shape[0])

        saved = (self._sdf_target_np, self._thickness_np,
                 self._origin, self._dx, self._n,
                 self._nx, self._ny, self._nz, self._shape)
        self._sdf_target_np = res.grid
        self._thickness_np = res.thickness
        self._origin = res.origin.astype(np.float32)
        self._dx = float(res.dx)
        self._n = int(res.n)
        # Keep the per-axis shape in sync with the swapped-in box grid so every
        # detector's unravel/clip uses the box's shape, not the global grid's.
        self._nz, self._ny, self._nx = (int(s) for s in res.grid.shape)
        self._shape = (self._nz, self._ny, self._nx)
        try:
            c = np.concatenate([contrib_c, train_c], axis=0).astype(np.float32)
            r = np.concatenate([contrib_r, train_r], axis=0).astype(np.float32)
            q = np.concatenate([contrib_q, train_q], axis=0).astype(np.float32)

            def _drop(idx):
                nonlocal c, r, q
                idx = np.asarray(idx, dtype=int)
                idx = idx[idx >= n_contrib]          # never touch the prefix
                if idx.size == 0:
                    return
                keep = np.ones(len(c), dtype=bool)
                keep[idx] = False
                c, r, q = c[keep], r[keep], q[keep]

            # Delete trainables entirely outside the (box) mesh, then fuse
            # trainables whose interior is already covered.
            _drop(self._detect_outside_ellipsoids(c, r, q))
            _drop(self._detect_redundant_ellipsoids(c, r, q, self._fuse_per_round))

            # Remaining global budget (net additions allowed across the whole pop).
            n_train = len(c) - n_contrib
            room = self._max_ellipsoids - (n_fixed + n_train)

            # Split oversized / bridging / protruding trainables (net +1 each).
            if room > 0:
                bridge = self._detect_bridging_ellipsoids(c, r, q)
                protr = self._detect_protruding_ellipsoids(c, r, q)
                seen, split_all = set(), []
                for v in list(bridge) + list(protr):
                    v = int(v)
                    if v >= n_contrib and v not in seen:
                        seen.add(v)
                        split_all.append(v)
                n_split = int(min(len(split_all), self._split_per_round, room))
                if n_split > 0:
                    split_idx = np.sort(np.asarray(split_all[:n_split], dtype=int))
                    child_c, child_r, child_q = [], [], []
                    for i in split_idx:
                        cc, cr, cq = self._split_ellipsoid(c[i], r[i], q[i])
                        child_c.append(cc); child_r.append(cr); child_q.append(cq)
                    keep = np.ones(len(c), dtype=bool)
                    keep[split_idx] = False
                    c = np.concatenate([c[keep]] + child_c, axis=0).astype(np.float32)
                    r = np.concatenate([r[keep]] + child_r, axis=0).astype(np.float32)
                    q = np.concatenate([q[keep]] + child_q, axis=0).astype(np.float32)
                    room -= n_split

            # Under-represented box regions: SPLIT the nearest trainable
            # ellipsoid along its longest semi-axis (no random spawning).  The
            # frozen contributor prefix [:n_contrib] is never split.
            if room > 0:
                n_dens = int(min(self._spawn_per_round, room))
                regions = self._detect_worst_regions(c, r, q, n_dens)
                dens_idx = self._split_targets_for_regions(
                    c, regions, room, exclude=set(), n_protect=n_contrib)
                if dens_idx:
                    split_idx = np.sort(np.asarray(dens_idx, dtype=int))
                    child_c, child_r, child_q = [], [], []
                    for i in split_idx:
                        cc, cr, cq = self._split_ellipsoid(c[i], r[i], q[i])
                        child_c.append(cc); child_r.append(cr); child_q.append(cq)
                    keep = np.ones(len(c), dtype=bool)
                    keep[split_idx] = False
                    c = np.concatenate([c[keep]] + child_c, axis=0).astype(np.float32)
                    r = np.concatenate([r[keep]] + child_r, axis=0).astype(np.float32)
                    q = np.concatenate([q[keep]] + child_q, axis=0).astype(np.float32)
                    room -= len(split_idx)

            return (c[n_contrib:].copy(), r[n_contrib:].copy(), q[n_contrib:].copy())
        finally:
            (self._sdf_target_np, self._thickness_np,
             self._origin, self._dx, self._n,
             self._nx, self._ny, self._nz, self._shape) = saved

    def _region_dc_all_boxes(self, fixed_c, fixed_r, fixed_q,
                             train_c, train_r, train_q, train_box, boxes):
        """Per-box divide-and-conquer for the combined local fit.

        Runs the existing ``_region_divide_conquer`` once per region box, scoped
        to that box's trainables (every other ellipsoid — frozen prefix + the
        other boxes' trainables — is passed as protected contributor so the
        detectors see the full union and the budget stays global).  Returns the
        reassembled, box-grouped trainable set plus a ``changed`` flag (False
        when no box added/removed/moved anything, so the caller can keep its Adam
        state instead of rebuilding).
        """
        new_c, new_r, new_q, new_b = [], [], [], []
        changed = False
        for b in range(len(boxes)):
            sel = (train_box == b)
            if not np.any(sel):
                continue
            tc, tr, tq = train_c[sel], train_r[sel], train_q[sel]
            other = ~sel
            oc = np.concatenate([fixed_c, train_c[other]], axis=0).astype(np.float32)
            orr = np.concatenate([fixed_r, train_r[other]], axis=0).astype(np.float32)
            oq = np.concatenate([fixed_q, train_q[other]], axis=0).astype(np.float32)
            ntc, ntr, ntq = self._region_divide_conquer(
                oc, orr, oq, tc.copy(), tr.copy(), tq.copy(),
                boxes[b]['res'], int(oc.shape[0]))
            if (ntc.shape[0] != tc.shape[0]
                    or not np.array_equal(ntc, tc)
                    or not np.array_equal(ntr, tr)
                    or not np.array_equal(ntq, tq)):
                changed = True
            new_c.append(ntc); new_r.append(ntr); new_q.append(ntq)
            new_b.append(np.full(ntc.shape[0], b, dtype=int))
        if not new_c:
            return train_c, train_r, train_q, train_box, False
        return (np.concatenate(new_c, axis=0).astype(np.float32),
                np.concatenate(new_r, axis=0).astype(np.float32),
                np.concatenate(new_q, axis=0).astype(np.float32),
                np.concatenate(new_b, axis=0), changed)

    def _local_fit_regions(self, centers, radii, rotations, region_sites,
                           box_results, gstep=-1):
        """Combined high-res local fit of ALL maintained regions at once.

        Replaces the older one-region-at-a-time loop (``_local_fit_region`` per
        site).  Every region's trainable ellipsoids are optimised together in a
        SINGLE Adam loop against the union of all pre-computed high-res box grids:
        samples are drawn from every box and evaluated by a point-based kernel, so
        each box's own origin/dx is baked into its sample points.  This

          * runs one optimisation instead of N serial ones (far fewer buffer
            allocations / host syncs → much faster), and
          * lets neighbouring regions coordinate through one shared union min
            instead of freezing each other as contributors (fewer dead gradients
            → less erratic motion).

        Parametrisation matches the GLOBAL optimiser — log-space radii, per-group
        learning rates (centres / radii / rotation), and Adam moments preserved
        across D&C cycles when nothing changed — so the global→local→global
        hand-off is continuous.  Returns updated (centers, radii, rotations).
        """
        if (self._sdf_computer is None or not self._sdf_computer.is_ready
                or not region_sites or not box_results):
            return centers, radii, rotations

        centers = np.asarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        rotations = np.asarray(rotations, dtype=np.float32)

        # ── 1) Per-box geometry + assign each ellipsoid to at most one box ──
        boxes = []
        for res in box_results:
            bmin = res.aabb_min.astype(np.float32)
            bmax = res.aabb_max.astype(np.float32)
            boxes.append(dict(
                res=res, bn=int(res.n), origin=res.origin.astype(np.float32),
                dx=float(res.dx), box_min=bmin, box_max=bmax,
                extent=float((bmax - bmin).max())))

        mr_all = np.max(np.abs(radii), axis=1)
        assigned = np.full(len(centers), -1, dtype=int)
        for b, m in enumerate(boxes):
            bmin, bmax = m['box_min'], m['box_max']
            # Trainable for this box: centre inside AND whole ellipsoid fits (so
            # fitting against the box SDF will not clip its surface).  First box
            # wins when boxes overlap.
            center_in = np.all((centers >= bmin) & (centers <= bmax), axis=1)
            fits = (np.all(centers - mr_all[:, None] >= bmin, axis=1)
                    & np.all(centers + mr_all[:, None] <= bmax, axis=1))
            cand = center_in & fits & (assigned < 0)
            assigned[cand] = b
        train_idx = np.where(assigned >= 0)[0]
        if train_idx.size == 0:
            return centers, radii, rotations
        # Group trainables by box so a no-op D&C round returns an identical set
        # (lets the Adam state be reused across cycles).
        train_idx = train_idx[np.argsort(assigned[train_idx], kind="stable")]
        train_box = assigned[train_idx].copy()

        # ── 2) Combined sample pool: world points + target + thickness ──
        band = float(self._surface_band_vox)
        pts_list, tgt_list, th_list = [], [], []
        any_thick = False
        for m in boxes:
            bn, o, d = m['bn'], m['origin'], m['dx']
            flat_t = m['res'].grid.ravel()
            sel = np.where(flat_t < band * d)[0]          # interior + thin band
            if sel.size == 0:
                continue
            iz = sel // (bn * bn)
            rem = sel - iz * (bn * bn)
            iy = rem // bn
            ix = rem - iy * bn
            px = o[0] + (ix.astype(np.float32) + 0.5) * d
            py = o[1] + (iy.astype(np.float32) + 0.5) * d
            pz = o[2] + (iz.astype(np.float32) + 0.5) * d
            pts_list.append(np.stack([px, py, pz], axis=1).astype(np.float32))
            tgt_list.append(flat_t[sel].astype(np.float32))
            if m['res'].thickness is not None:
                th = dilate_zeros(m['res'].thickness, iters=2).ravel().astype(np.float32)
                th_list.append(th[sel])
                any_thick = True
            else:
                th_list.append(np.zeros(sel.size, dtype=np.float32))
        if not pts_list:
            return centers, radii, rotations
        pool_points = np.concatenate(pts_list, axis=0).astype(np.float32)
        pool_targets = np.concatenate(tgt_list, axis=0).astype(np.float32)
        pool_thick = np.concatenate(th_list, axis=0).astype(np.float32)
        P = int(pool_points.shape[0])

        interior_th = pool_thick[pool_thick > 0.0]
        thick_ref = float(np.median(interior_th)) if interior_th.size else 1.0
        thin_w = float(self._thin_loss_weight) if any_thick else 0.0

        # Pool is fixed for the whole fit → upload once.
        wp_points = wp.array(pool_points, dtype=wp.vec3, device=device)
        wp_targets = wp.array(pool_targets, dtype=wp.float32, device=device)
        wp_thick = wp.array(pool_thick, dtype=wp.float32, device=device)

        # Show every region box that actually holds a trainable — i.e. the small
        # high-res boxes currently being optimised, not one box over the whole
        # object.  (Boxes are fixed geometry, so one emit at the start suffices.)
        active = sorted({int(b) for b in train_box})
        region_boxes = [(boxes[b]['box_min'].copy(), boxes[b]['box_max'].copy())
                        for b in active]
        self.region_changed.emit(region_boxes)

        # ── 3) frozen prefix + trainable suffix ──
        fixed_mask = np.ones(len(centers), dtype=bool)
        fixed_mask[train_idx] = False
        fixed_c = centers[fixed_mask].astype(np.float32).copy()
        fixed_r = radii[fixed_mask].astype(np.float32).copy()
        fixed_q = rotations[fixed_mask].astype(np.float32).copy()
        n_fixed = int(fixed_c.shape[0])

        train_c = centers[train_idx].astype(np.float32).copy()
        train_r = radii[train_idx].astype(np.float32).copy()
        train_q = rotations[train_idx].astype(np.float32).copy()

        bs = int(min(self._batch_size, 4096))
        lr0 = float(self._lr_at(gstep)) if gstep >= 0 else float(self._local_lr)
        n_cycles = self._region_dc_cycles
        steps_per_cycle = max(1, self._region_steps // n_cycles)
        report_every = max(1, steps_per_cycle // 3)

        def _clamp_arrays(tbox):
            lo = np.stack([boxes[b]['box_min'] for b in tbox]).astype(np.float32)
            hi = np.stack([boxes[b]['box_max'] for b in tbox]).astype(np.float32)
            logmin = np.array(
                [np.log(max(boxes[b]['dx'], 1e-9)) for b in tbox], dtype=np.float32)
            logmax = np.array(
                [np.log(max(0.5 * boxes[b]['extent'], 2.0 * boxes[b]['dx']))
                 for b in tbox], dtype=np.float32)
            return (wp.array(lo, dtype=wp.vec3, device=device),
                    wp.array(hi, dtype=wp.vec3, device=device),
                    wp.array(logmin, dtype=wp.float32, device=device),
                    wp.array(logmax, dtype=wp.float32, device=device))

        def _build_state():
            sub_c = np.concatenate([fixed_c, train_c], axis=0).astype(np.float32)
            sub_r = np.concatenate([fixed_r, train_r], axis=0).astype(np.float32)
            sub_q = np.concatenate([fixed_q, train_q], axis=0).astype(np.float32)
            num_e = n_fixed + int(train_c.shape[0])
            buf = self._alloc_buffers(num_e, bs, P, sub_c, sub_r, sub_q,
                                      sdf_target_np=pool_targets)
            log_r = wp.array(
                np.log(np.maximum(buf['pred_radii'].numpy(), 1e-6)),
                dtype=wp.vec3, device=device, requires_grad=True)
            st = dict(
                num_e=num_e, offset=n_fixed,
                pred_centers=buf['pred_centers'], pred_radii=buf['pred_radii'],
                pred_rot_flat=buf['pred_rot_flat'], pred_log_radii=log_r,
                min_d_cache=buf['min_d_cache'], sdf_pred=buf['sdf_pred'],
                loss=buf['loss'], wp_indices=buf['wp_indices'],
                opt_c=wp.optim.Adam([buf['pred_centers']], lr=lr0),
                opt_r=wp.optim.Adam([log_r], lr=lr0 * self._lr_mult_radii),
                opt_q=wp.optim.Adam([buf['pred_rot_flat']], lr=lr0 * self._lr_mult_rot),
            )
            st['grad_c'] = [st['pred_centers'].grad.flatten()]
            st['grad_r'] = [log_r.grad.flatten()]
            st['grad_q'] = [st['pred_rot_flat'].grad.flatten()]
            st['clamps'] = _clamp_arrays(train_box)
            return st

        state = None
        for cycle in range(n_cycles):
            if self._stop_flag or int(train_c.shape[0]) == 0:
                break
            if state is None:
                state = _build_state()

            num_e = state['num_e']
            offset = state['offset']
            n_train = num_e - offset
            rot_offset = offset * 4
            pred_centers = state['pred_centers']
            pred_radii = state['pred_radii']
            pred_rot_flat = state['pred_rot_flat']
            pred_log_radii = state['pred_log_radii']
            min_d_cache = state['min_d_cache']
            sdf_pred = state['sdf_pred']
            loss = state['loss']
            wp_indices = state['wp_indices']
            cl_lo, cl_hi, cl_logmin, cl_logmax = state['clamps']

            for li in range(steps_per_cycle):
                if self._stop_flag:
                    break
                batch = self._rng.integers(0, P, size=bs).astype(np.int32)
                wp_indices.assign(np.ascontiguousarray(batch))

                tape = wp.Tape()
                with tape:
                    # World radii from trainable log-radii (gradient → log-space).
                    wp.launch(_exp_radii_kernel, dim=num_e,
                              inputs=[pred_log_radii, pred_radii], device=device)
                    min_d_cache.zero_()
                    if self._isotropic:
                        wp.launch(
                            _sphere_sdf_kernel_points,
                            dim=bs,
                            inputs=[pred_centers, pred_radii,
                                    min_d_cache, num_e, wp_points, wp_indices, sdf_pred],
                            device=device)
                    elif self._superquadric:
                        wp.launch(
                            _superquadric_sdf_kernel_points,
                            dim=bs,
                            inputs=[pred_centers, pred_radii, pred_rot_flat,
                                    self._uniform_eps_wp(num_e),
                                    self._zero_bend_wp(num_e),
                                    min_d_cache, num_e, wp_points, wp_indices, sdf_pred],
                            device=device)
                    elif self._capsule:
                        wp.launch(
                            _capsule_sdf_kernel_points,
                            dim=bs,
                            inputs=[pred_centers, pred_radii, pred_rot_flat,
                                    min_d_cache, num_e, wp_points, wp_indices, sdf_pred],
                            device=device)
                    else:
                        wp.launch(
                            _ellipsoid_sdf_kernel_points,
                            dim=bs,
                            inputs=[pred_centers, pred_radii, pred_rot_flat,
                                    min_d_cache, num_e, wp_points, wp_indices, sdf_pred],
                            device=device)
                    loss.zero_()
                    wp.launch(
                        _rmse_loss_kernel_batch,
                        dim=bs,
                        inputs=[sdf_pred, wp_targets, wp_indices, loss, bs,
                                float(self._miss_penalty_weight),
                                float(self._surface_weight), float(self._surface_sigma),
                                float(self._outside_penalty_weight),
                                wp_thick, float(thick_ref),
                                float(thin_w), float(self._thin_max_factor)],
                        device=device)
                    if self._flat_weight > 0.0:
                        wp.launch(
                            _flatness_penalty_kernel,
                            dim=n_train,
                            inputs=[pred_radii, loss, n_train, offset,
                                    float(self._flat_weight), float(self._flat_min_ratio)],
                            device=device)

                tape.backward(loss)

                # Freeze the contributor prefix (zero its grads → Adam can't move it).
                if offset > 0:
                    wp.launch(_zero_vec3_prefix, dim=offset,
                              inputs=[pred_centers.grad], device=device)
                    wp.launch(_zero_vec3_prefix, dim=offset,
                              inputs=[pred_log_radii.grad], device=device)
                    wp.launch(_zero_f32_prefix, dim=rot_offset,
                              inputs=[pred_rot_flat.grad], device=device)

                state['opt_c'].lr = lr0
                state['opt_r'].lr = lr0 * self._lr_mult_radii
                state['opt_q'].lr = lr0 * self._lr_mult_rot
                state['opt_c'].step(state['grad_c'])
                state['opt_r'].step(state['grad_r'])
                state['opt_q'].step(state['grad_q'])
                tape.zero()

                # Per-box clamps (log-radii + centres) and unit quats.
                wp.launch(_clamp_log_radii_perbox, dim=n_train,
                          inputs=[pred_log_radii, cl_logmin, cl_logmax, offset],
                          device=device)
                wp.launch(_clamp_centers_perbox, dim=n_train,
                          inputs=[pred_centers, cl_lo, cl_hi, offset], device=device)
                wp.launch(_normalize_flat_quats_range, dim=n_train,
                          inputs=[pred_rot_flat, offset], device=device)

                if li % report_every == 0 or li == steps_per_cycle - 1:
                    wp.launch(_exp_radii_kernel, dim=num_e,
                              inputs=[pred_log_radii, pred_radii], device=device)
                    wp.synchronize_device(device)
                    sc = pred_centers.numpy()[offset:]
                    sr = pred_radii.numpy()[offset:]
                    sq = pred_rot_flat.numpy().reshape(-1, 4)[offset:]
                    vis_c = np.concatenate([fixed_c, sc], axis=0)
                    vis_r = np.concatenate([fixed_r, sr], axis=0)
                    vis_q = np.concatenate([fixed_q, sq], axis=0)
                    done = cycle * steps_per_cycle + li + 1
                    self.local_progress.emit(done, self._region_steps)
                    self.step_visual.emit(int(gstep), float(loss.numpy()[0]),
                                          vis_c.copy(), vis_r.copy(), vis_q.copy(),
                                          None)

            # Pull trainables back (refresh world radii from log first).
            wp.launch(_exp_radii_kernel, dim=num_e,
                      inputs=[pred_log_radii, pred_radii], device=device)
            wp.synchronize_device(device)
            train_c = pred_centers.numpy()[offset:].astype(np.float32).copy()
            train_r = pred_radii.numpy()[offset:].astype(np.float32).copy()
            train_q = pred_rot_flat.numpy().reshape(-1, 4)[offset:].astype(np.float32).copy()

            # Per-box divide-and-conquer between cycles.  Keep the Adam state
            # only when nothing changed (warm-start); else rebuild next cycle.
            if cycle < n_cycles - 1 and not self._stop_flag:
                train_c, train_r, train_q, train_box, changed = \
                    self._region_dc_all_boxes(
                        fixed_c, fixed_r, fixed_q,
                        train_c, train_r, train_q, train_box, boxes)
                if changed:
                    state = None

        out_c = np.concatenate([fixed_c, train_c], axis=0).astype(np.float32)
        out_r = np.concatenate([fixed_r, train_r], axis=0).astype(np.float32)
        out_q = np.concatenate([fixed_q, train_q], axis=0).astype(np.float32)
        return out_c, out_r, out_q

    def _maybe_superfit(self, step, pred_centers, pred_radii, pred_rot_flat):
        """SuperFit cycle (adaptive density control, à la Gaussian Splatting).

        Three moves, then one isolated local fit of everything new:
          - **fuse** (prune) redundant ellipsoids whose interior is already
            covered by the others — they have no independent task, so dropping
            them frees budget without changing the union SDF;
          - **split** oversized ellipsoids that protrude past the mesh surface
            into two halves (over-reconstruction → divide), and
          - **spawn** fresh ellipsoids in under-represented regions
            (under-reconstruction → conquer).

        Returns new (centers, radii, rotations) when the population changed,
        else ``None``.  Net growth stops once ``max_ellipsoids`` is reached
        (a split is net +1: two children replace one parent).
        """
        # Densify and local fit are independent: either one being enabled is
        # enough to enter this method (the run loop dispatches here when
        # ``superfit OR local_fit`` is on).
        if not (self._superfit or self._local_fit_enabled):
            return None
        ns = float(self._num_steps)
        # Densify (population-changing) moves fire on the SuperFit cadence, but
        # only inside the [densify_start, densify_until] window — afterwards the
        # population settles cleanly (pure Adam refinement, à la Gaussian
        # Splatting).  Local fit runs on its OWN window [local_fit_start,
        # local_fit_end] and its own elapsed cadence (local_fit_every),
        # completely decoupled: the two phases may overlap, be disjoint, or run
        # at different frequencies.  Densify requires SuperFit to be enabled;
        # local fit does not.
        densify_active = bool(
            self._superfit
            and self._superfit_every > 0 and step > 0
            and step % self._superfit_every == 0
            and step >= self._densify_start_frac * ns
            and step < self._densify_until_frac * ns)
        local_active = bool(
            self._local_fit_enabled
            and step >= self._local_fit_start_frac * ns
            and step <= self._local_fit_end_frac * ns
            and (step - self._last_local_fit_step) >= self._local_fit_every)
        if not (densify_active or local_active):
            return None

        wp.synchronize_device(device)
        centers = pred_centers.numpy().copy()
        radii = pred_radii.numpy().copy()
        rotations = pred_rot_flat.numpy().reshape(-1, 4).copy()

        n_before = len(centers)

        # Operation gizmo events for the 3-D viewport: each is
        # ``(op, center_world, radius_world)`` recorded at the spot the move
        # happened, so the user can watch where SuperFit acted.
        ops: list[tuple] = []

        def _record(op, idx, cen, rad):
            for i in np.atleast_1d(np.asarray(idx, dtype=int)):
                ops.append((op, cen[i].astype(np.float32).copy(),
                            float(np.max(np.abs(rad[i])))))

        # ── 0a) Delete ellipsoids that sit entirely outside the mesh ──
        # All population-changing moves (delete / fuse / merge / split / spawn)
        # are gated on the densify window; outside it only local fit may run.
        out_idx = (self._detect_outside_ellipsoids(centers, radii, rotations)
                   if densify_active else np.empty(0, dtype=int))
        n_deleted = int(len(out_idx))
        if n_deleted > 0:
            _record('delete', out_idx, centers, radii)
            keep = np.ones(len(centers), dtype=bool)
            keep[out_idx] = False
            centers, radii, rotations = centers[keep], radii[keep], rotations[keep]

        # ── 0a2) Delete degenerate shapes (too flat / too pointy) ──
        deg_idx = (self._detect_degenerate_ellipsoids(radii)
                   if densify_active else np.empty(0, dtype=int))
        # Never wipe the whole population in one round (safety against a bad fit
        # transiently making everything degenerate).
        if 0 < len(deg_idx) < len(centers):
            n_deleted += int(len(deg_idx))
            _record('delete', deg_idx, centers, radii)
            keep = np.ones(len(centers), dtype=bool)
            keep[deg_idx] = False
            centers, radii, rotations = centers[keep], radii[keep], rotations[keep]

        # ── 0b) Fuse redundant ellipsoids (no independent task → drop them) ──
        # Done first so freed slots are reused by the split/spawn moves below.
        # Gated by the Prune toggle (this is the population-shrinking pruning;
        # the 0a/0a2 safety deletes above stay on regardless).
        fuse_idx = (self._detect_redundant_ellipsoids(
            centers, radii, rotations, self._fuse_per_round)
            if (densify_active and self._prune_enabled) else np.empty(0, dtype=int))
        n_fused = int(len(fuse_idx)) + n_deleted
        if len(fuse_idx) > 0:
            _record('fuse', fuse_idx, centers, radii)
            keep = np.ones(len(centers), dtype=bool)
            keep[fuse_idx] = False
            centers, radii, rotations = centers[keep], radii[keep], rotations[keep]

        # ── 0c) Merge overlapping pairs into one when it barely moves the surface
        if densify_active:
            centers, radii, rotations, n_merged = self._detect_merges(
                centers, radii, rotations)
            # _detect_merges appends the merged ellipsoids at the tail of the
            # arrays, so the last ``n_merged`` entries are the fused-pair results.
            if n_merged > 0:
                _record('merge', np.arange(len(centers) - n_merged, len(centers)),
                        centers, radii)
        else:
            n_merged = 0
        n_fused += n_merged

        # ── Analysis snapshot for the 3-D viewport ──────────────────────
        # Run the three densify detectors on the post-fuse/merge population and
        # publish *what they flag*, independent of which mechanisms are enabled
        # or how much budget is left — so the user can see how over-/under-/
        # bridging classification behaves.  The action paths below reuse these
        # results (no detector runs twice).
        viz_bridge = self._detect_bridging_ellipsoids(centers, radii, rotations)
        viz_protr = self._detect_protruding_ellipsoids(centers, radii, rotations)
        # Region detection is the only O(n³) step in SuperFit; run it on the
        # resolution-capped detection grid so the per-cycle cost stays constant
        # as the global grid n grows.  Outputs are world-space (seed_world /
        # seed_depth), so downstream split/spawn is unaffected.
        with self._detection_grid_scope():
            viz_regions = self._detect_worst_regions(
                centers, radii, rotations, self._analysis_region_k,
                min_severity=self._analysis_min_severity)
        if not self.signalsBlocked():
            bridge_set = set(int(i) for i in viz_bridge)
            region_r_world = float(self._region_radius_vox) * float(self._dx)
            analysis = {
                # Bridging takes priority over plain protrusion (as in the action
                # path), so an ellipsoid is shown in at most one over-rep class.
                'over': [(centers[i].astype(np.float32).copy(),
                          float(np.max(np.abs(radii[i]))))
                         for i in viz_protr if int(i) not in bridge_set],
                'bridge': [(centers[i].astype(np.float32).copy(),
                            float(np.max(np.abs(radii[i])))) for i in viz_bridge],
                'under': [(np.asarray(reg['seed_world'], np.float32).copy(),
                           region_r_world) for reg in viz_regions],
            }
            self.analysis_regions.emit(step, analysis)

        n_curr = len(centers)
        # Net additions allowed this cycle.  Zero outside the densify window (or
        # at the cap) so the split/spawn detectors below naturally produce
        # nothing — but local fit may still run afterwards.
        budget = (self._max_ellipsoids - n_curr) if densify_active else 0
        budget = max(0, budget)

        # ── Densify: SPLIT over-represented ellipsoids + SPLIT/SPAWN in under-rep
        # Both mechanisms are independently switchable (split_enabled /
        # spawn_underrep); merge is gated in _detect_merges by merge_enabled.
        # 1) Over-represented ellipsoids to split (net +1 each): a *bridging* one
        #    spanning the gap between structures (prioritised), or one that
        #    *protrudes* past the surface.  Combine both, bridging-first, dedup.
        targets, seen = [], set()
        if self._split_enabled:
            bridge = viz_bridge           # reuse the analysis-snapshot detections
            protr = viz_protr
            cap_over = int(min(self._split_per_round, budget))
            for v in list(bridge) + list(protr):
                v = int(v)
                if v not in seen and len(targets) < cap_over:
                    seen.add(v)
                    targets.append(v)

        # Under-represented regions → SPLIT the nearest ellipsoid when one is
        # close to the gap, else SPAWN a new ellipsoid guaranteed fully inside
        # the mesh.  Detect on the *full* current config so areas still covered
        # by a parent are not double-counted.
        spawn_c = np.empty((0, 3), np.float32)
        spawn_r = np.empty((0, 3), np.float32)
        spawn_q = np.empty((0, 4), np.float32)
        spawn_sites = []
        if self._split_enabled or self._spawn_underrep:
            n_dens = int(min(self._spawn_per_round, budget - len(targets)))
            # The worst-first viz regions are a superset; the action path only
            # needs the top-``n_dens`` (greedy peak order is identical).
            regions = viz_regions[:n_dens] if n_dens > 0 else []
            split_tgts, spawn_regions = self._densify_regions(
                centers, radii, regions, budget - len(targets), exclude=seen,
                split_enabled=self._split_enabled,
                spawn_enabled=self._spawn_underrep)
            targets += split_tgts
            if spawn_regions:
                spawn_c, spawn_r, spawn_q, spawn_sites = self._spawn_in_regions(
                    spawn_regions, len(spawn_regions))
                # Under symmetry the post-maintenance ``_build_symmetric_layout``
                # rebuilds the population from the *source* (better-fitting) half
                # and discards the other half.  Spawns target under-represented
                # (worse-fitting) regions — i.e. the half that gets discarded —
                # so they vanish.  Mirror each spawn so a copy lands on the source
                # half and survives the re-layout (the discarded one is recreated
                # as its reflection anyway).
                if (self._symmetry_enabled and self._sym_axis is not None
                        and len(spawn_c) > 0):
                    a, p = self._sym_axis, self._sym_plane
                    mc = spawn_c.copy()
                    mc[:, a] = 2.0 * float(p) - mc[:, a]
                    spawn_c = np.concatenate([spawn_c, mc], axis=0).astype(np.float32)
                    spawn_r = np.concatenate([spawn_r, spawn_r.copy()], axis=0).astype(np.float32)
                    spawn_q = np.concatenate(
                        [spawn_q, _mirror_quats(spawn_q, a)], axis=0).astype(np.float32)
                    spawn_sites = spawn_sites + [
                        (mc[i].astype(np.float32), h) for i, (_c, h) in enumerate(spawn_sites)]
        n_spawn = int(len(spawn_c))
        if n_spawn > 0:
            _record('spawn', np.arange(n_spawn), spawn_c, spawn_r)

        # Note: when ``budget == 0`` (outside the densify window, or at the cap)
        # ``targets`` is empty and ``n_spawn == 0``, so the child-building below
        # is a no-op and flow falls straight through to the local-fit phase.

        # Build split children (over-rep), recording each split site as a region
        # centre (the parent's location, with a half-extent covering the parent).
        r_radius_world = float(self._region_radius_vox) * float(self._dx)
        split_idx = (np.sort(np.asarray(targets, dtype=int))
                     if targets else np.empty(0, dtype=int))
        n_split = int(len(split_idx))
        child_c, child_r, child_q, pools = [], [], [], []
        region_sites = []   # (center_world, half_extent) per maintained region
        for i in split_idx:
            cc, cr, cq = self._split_ellipsoid(centers[i], radii[i], rotations[i])
            child_c.append(cc); child_r.append(cr); child_q.append(cq)
            pools.append(self._interior_ball_pool(centers[i], self._region_radius_vox))
            parent_max_r = float(np.max(np.abs(radii[i])))
            half = max(r_radius_world, parent_max_r * float(self._split_size_factor))
            region_sites.append((centers[i].astype(np.float32).copy(), half))
            ops.append(('split', centers[i].astype(np.float32).copy(), parent_max_r))

        # Append spawned ellipsoids and their region sites.
        if n_spawn > 0:
            child_c.append(spawn_c); child_r.append(spawn_r); child_q.append(spawn_q)
            region_sites.extend(spawn_sites)
            for site_c, _half in spawn_sites:
                pools.append(self._interior_ball_pool(site_c, self._region_radius_vox))

        keep = np.ones(len(centers), dtype=bool)   # current (post-fusion) length
        keep[split_idx] = False
        centers, radii, rotations = centers[keep], radii[keep], rotations[keep]
        offset = len(centers)        # frozen prefix = surviving originals

        centers   = np.concatenate([centers]   + child_c, axis=0)
        radii     = np.concatenate([radii]     + child_r, axis=0)
        rotations = np.concatenate([rotations] + child_q, axis=0)

        # Local fit with no fresh densify regions (e.g. local window extends past
        # the densify window, or densify added nothing this cycle): source the
        # worst regions from the analysis snapshot and re-fit the EXISTING
        # geometry there.  ``_local_fit_regions`` assigns trainable ellipsoids by
        # region-box membership, so refitting in place needs no appended children.
        if local_active and not region_sites and viz_regions:
            r_radius_world = float(self._region_radius_vox) * float(self._dx)
            for reg in viz_regions:
                sc = np.asarray(reg['seed_world'], np.float32).copy()
                region_sites.append((sc, r_radius_world))
                pools.append(self._interior_ball_pool(sc, self._region_radius_vox))

        # Under symmetry only the source (better-fitting) half survives the
        # post-maintenance ``_build_symmetric_layout``; the other half is
        # discarded and re-derived as the source's reflection.  Locally fitting a
        # discarded-side region is therefore wasted work (incl. its 128³ box SDF)
        # — drop those sites and keep only source-side / on-plane regions.  Each
        # mirrored spawn already has a source-side twin, so its fit is preserved.
        if (self._symmetry_enabled and self._sym_axis is not None
                and region_sites):
            a, p = self._sym_axis, float(self._sym_plane)
            tol = 1.5 * float(self._dx)
            src_sign = self._symmetry_source_side(centers)
            kept = []
            for site_c, site_half in region_sites:
                signed = float(site_c[a]) - p
                on_plane = abs(signed) < tol
                if on_plane or (signed >= 0.0) == (src_sign >= 0.0):
                    kept.append((site_c, site_half))
            region_sites = kept

        # Gate: local fit runs on its own window + elapsed cadence (computed at
        # the top).  Stamp the firing step whenever the cadence is due — even if
        # there is nothing to fit this cycle — so it does not re-trigger every
        # step.  New densify children, when local fit is NOT due, are simply left
        # for the global optimiser to refine.
        did_local = False
        if local_active:
            self._last_local_fit_step = step
            if region_sites and self._sdf_computer is not None \
                    and self._sdf_computer.is_ready:
                # Re-fit every ellipsoid centred in each region against a fresh
                # high-res SDF box limited to that region.  All region boxes are
                # computed up front in a SINGLE batched kernel launch.
                self.phase_changed.emit("local")
                box_geoms = [self._region_box(c, h) for c, h in region_sites]
                box_results = self._sdf_computer.compute_box_grids_batch(
                    box_geoms, n=self._region_res, compute_thickness=True)
                # One combined fit over ALL region boxes (replaces the former
                # per-region serial loop): faster and less erratic, see
                # _local_fit_regions.
                centers, radii, rotations = self._local_fit_regions(
                    centers, radii, rotations, region_sites, box_results, step)
                self.region_changed.emit(None)
                self.phase_changed.emit("global")
                did_local = True
            elif region_sites:
                # Fallback: single coarse-grid local fit over the union of pools.
                self.phase_changed.emit("local")
                pools = [p for p in pools if p.size > 0]
                union_pool = (np.unique(np.concatenate(pools)).astype(np.int32)
                              if pools else np.arange(self._nx * self._ny * self._nz,
                                                      dtype=np.int32))
                centers, radii, rotations = self._local_fit(
                    centers, radii, rotations, offset, union_pool, step,
                )
                self.phase_changed.emit("global")
                did_local = True

        if ops:
            self.op_events.emit(step, ops)
        if n_fused == 0 and n_split == 0 and n_spawn == 0 and not did_local:
            # Nothing changed this cycle (no densify moves, no local fit).
            self.maintenance_done.emit(step, n_before, 0, 0)
            return None
        n_appended = len(centers) - offset
        self.maintenance_done.emit(
            step, n_before, n_fused + n_split + n_spawn, n_appended)
        return centers, radii, rotations

    def _maybe_maintain(self, step, pred_centers, pred_radii, pred_rot_flat):
        if self._maintenance_every <= 0:
            return None
        if step == 0 or step % self._maintenance_every != 0:
            return None
        if step >= self._densify_until_frac * self._num_steps:
            return None     # refinement phase: no more density changes

        wp.synchronize_device(device)
        c = pred_centers.numpy().copy()
        r = pred_radii.numpy().copy()
        q = pred_rot_flat.numpy().reshape(-1, 4).copy()

        n_before = len(c)
        c, r, q, changed, n_pruned, n_spawned = self._do_maintenance(c, r, q)
        self.maintenance_done.emit(step, n_before, n_pruned, n_spawned)

        if not changed:
            return None
        return c, r, q

    def _ensure_thickness_wp(self, total: int) -> None:
        """Build the device thickness array + reference scale once.

        When no thickness field is available the weighting is neutral
        (thin_weight_eff = 0 → tw = 1 everywhere).
        """
        if self._wp_thickness is not None:
            return
        if self._thickness_np is not None:
            flat = dilate_zeros(self._thickness_np, iters=2).ravel().astype(np.float32)
            interior = flat[flat > 0.0]
            self._thick_ref = float(np.median(interior)) if interior.size else 1.0
            self._thin_weight_eff = float(self._thin_loss_weight)
            self._thickness_flat = flat        # reused by the thinness-biased sampler
        else:
            flat = np.zeros(total, dtype=np.float32)
            self._thick_ref = 1.0
            self._thin_weight_eff = 0.0
            self._thickness_flat = None
        self._wp_thickness = wp.array(
            flat, dtype=wp.float32, device=device, requires_grad=False,
        )

    # ── naive SGD ─────────────────────────────────────────────────────

    def _run_naive(self):
        origin = self._origin
        n = self._n
        nx, ny, nz = self._nx, self._ny, self._nz
        dx = self._dx
        total = nx * ny * nz
        self._ensure_thickness_wp(total)
        num_e = self._num_ellipsoids
        bs = self._batch_size

        buf = self._alloc_buffers(num_e, bs, total)
        pred_centers  = buf['pred_centers']
        pred_radii    = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']
        min_d_cache   = buf['min_d_cache']
        sdf_pred      = buf['sdf_pred']
        loss          = buf['loss']
        sdf_target    = buf['sdf_target']
        wp_indices    = buf['wp_indices']
        # Init (e.g. the symmetric layout) may yield a different count than
        # requested — track the actual number so kernel launches match the arrays.
        num_e = int(pred_centers.shape[0])

        sampler = BandSampler(
            self._sdf_target_np.ravel(), bs,
            float(self._surface_band_vox) * float(dx),
            self._surface_fraction, rng=self._rng,
            flat_thickness=self._thickness_flat,
            thin_bias=float(self._thin_sample_bias),
        )
        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = 0.01

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            result = self._maybe_maintain(step, pred_centers, pred_radii, pred_rot_flat)
            if result is not None:
                c_np, r_np, q_np = result
                r_np, q_np = self._project_isotropic_np(r_np, q_np)   # sphere
                num_e = len(c_np)
                buf = self._alloc_buffers(num_e, bs, total, c_np, r_np, q_np)
                buf['sdf_target'] = sdf_target
                pred_centers  = buf['pred_centers']
                pred_radii    = buf['pred_radii']
                pred_rot_flat = buf['pred_rot_flat']
                min_d_cache   = buf['min_d_cache']
                sdf_pred      = buf['sdf_pred']
                loss          = buf['loss']
                wp_indices    = buf['wp_indices']

            wp_indices.assign(sampler.next_batch())

            tape = wp.Tape()
            with tape:
                min_d_cache.zero_()
                if self._isotropic:
                    wp.launch(
                        _sphere_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                elif self._superquadric:
                    wp.launch(
                        _superquadric_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat,
                                self._uniform_eps_wp(num_e),
                                self._zero_bend_wp(num_e), min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                elif self._capsule:
                    wp.launch(
                        _capsule_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat, min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                else:
                    wp.launch(
                        _ellipsoid_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat, min_d_cache,
                                num_e, wp_origin, float(dx), nx, ny, nz,
                                wp_indices, sdf_pred],
                        device=device,
                    )
                loss.zero_()
                wp.launch(
                    _rmse_loss_kernel_batch,
                    dim=bs,
                    inputs=[sdf_pred, sdf_target, wp_indices, loss, bs,
                            float(self._miss_penalty_weight),
                            float(self._surface_weight), float(self._surface_sigma),
                            float(self._outside_penalty_weight),
                            self._wp_thickness, float(self._thick_ref),
                            float(self._thin_weight_eff), float(self._thin_max_factor)],
                    device=device,
                )
                if self._flat_weight > 0.0:
                    wp.launch(
                        _flatness_penalty_kernel,
                        dim=num_e,
                        inputs=[pred_radii, loss, num_e, 0,
                                float(self._flat_weight), float(self._flat_min_ratio)],
                        device=device,
                    )

            tape.backward(loss)

            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_centers, tape.gradients[pred_centers], lr],
                      device=device)
            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_radii, tape.gradients[pred_radii], lr],
                      device=device)
            wp.launch(_sgd_step_f32, dim=num_e * 4,
                      inputs=[pred_rot_flat, tape.gradients[pred_rot_flat], lr],
                      device=device)
            wp.launch(_normalize_flat_quats, dim=num_e,
                      inputs=[pred_rot_flat], device=device)
            # Sphere: naive path trains world radii directly — project them
            # (mean of the 3 components) + reset rotation each step.
            self._project_isotropic(pred_radii, pred_rot_flat, num_e)
            # Capsule: circular cross-section (r1 = r0).
            self._project_capsule(pred_radii, num_e)

            tape.zero()

            if step % self._report_every == 0:
                self._emit_progress(step, loss, pred_centers, pred_radii,
                                    pred_rot_flat, num_e, origin, dx, n)

    # ── Adam ──────────────────────────────────────────────────────────

    def _lr_at(self, step: int) -> float:
        """Cosine schedule with an exponential tail (lr_init → lr_final).

        The cosine keeps the LR high early so big structures snap into place,
        while the extra ``exp(-lr_decay_k · progress)`` factor pulls the whole
        curve down through the middle and end of training so ellipsoids actually
        settle late instead of jittering around the optimum.
        """
        steps = max(self._num_steps - 1, 1)
        progress = min(max(step / steps, 0.0), 1.0)
        cos = 0.5 * (1.0 + np.cos(np.pi * progress))
        tail = np.exp(-float(self._lr_decay_k) * progress)
        return float(self._lr_final + (self._lr_init - self._lr_final) * cos * tail)

    def _soft_k(self, step: int) -> float:
        """Soft-min sharpness k for this step (1 / blend-width in world units).

        The blend width anneals from ``vox_start`` → ``vox_end`` voxels over
        training: soft (wide) early for dense gradients, near-hard (narrow) late
        for an accurate union.
        """
        steps = max(self._num_steps - 1, 1)
        progress = min(max(step / steps, 0.0), 1.0)
        vox = self._soft_vox_start + (self._soft_vox_end - self._soft_vox_start) * progress
        return 1.0 / max(vox * float(self._dx), 1e-9)

    def _run_adam(self):
        origin = self._origin
        n = self._n
        nx, ny, nz = self._nx, self._ny, self._nz
        dx = self._dx
        total = nx * ny * nz
        # ── Pre-training setup (slow for large SDFs) — report it on the progress
        # bar so the gap between pressing Fit and the first step is not dead time.
        # Resolve the symmetry plane + symmetrise the target BEFORE init/thickness
        # so the initial placement and loss already see a symmetric target.
        if self._symmetry_enabled and not self._sym_checked:
            self.prep_progress.emit(0.05, "detecting symmetry")
            self._setup_symmetry()
        self.prep_progress.emit(0.30, "feature thickness")
        self._ensure_thickness_wp(total)
        num_e = self._num_ellipsoids
        bs = self._batch_size

        self.prep_progress.emit(0.55, "allocating buffers")
        buf = self._alloc_buffers(num_e, bs, total)
        pred_centers  = buf['pred_centers']
        pred_radii    = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']
        pred_eps      = buf['pred_eps']
        pred_bend     = buf['pred_bend']
        min_d_cache   = buf['min_d_cache']
        sdf_pred      = buf['sdf_pred']
        loss          = buf['loss']
        sdf_target    = buf['sdf_target']
        wp_indices    = buf['wp_indices']
        # Init (e.g. the symmetric layout) may yield a different count than
        # requested — track the actual number so kernel launches match the arrays.
        num_e = int(pred_centers.shape[0])
        # Second scan buffer for the soft-min (LogSumExp) union.
        soft_s_cache = wp.zeros(
            (bs, num_e + 1), dtype=wp.float32, device=device, requires_grad=True)

        # NB: sample the *whole* grid even for symmetric meshes.  A fixed-half
        # sampler starves the trained source ellipsoids of gradient whenever the
        # source half is the *other* side (the source side is chosen dynamically
        # in _build_symmetric_layout and can flip on maintenance) — and the
        # per-step cost is set by batch_size regardless, so half-sampling saved
        # nothing anyway.
        self.prep_progress.emit(0.80, "building sampler")
        sampler = BandSampler(
            self._sdf_target_np.ravel(), bs,
            float(self._surface_band_vox) * float(dx),
            self._surface_fraction, rng=self._rng,
            flat_thickness=self._thickness_flat,
            thin_bias=float(self._thin_sample_bias),
        )
        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = self._lr_init

        # Generous parameter bounds (safety net for superquadric / bent fits):
        # radii ∈ [0.25·dx, grid extent], centres within the (slightly padded) grid.
        _extent = float(max(self._nx, self._ny, self._nz) * dx)
        _log_rmin = float(np.log(0.25 * dx))
        _log_rmax = float(np.log(_extent))
        _pad = 0.25 * _extent
        _c_lo = wp.vec3(float(origin[0]) - _pad, float(origin[1]) - _pad,
                        float(origin[2]) - _pad)
        _c_hi = wp.vec3(float(origin[0]) + self._nx * dx + _pad,
                        float(origin[1]) + self._ny * dx + _pad,
                        float(origin[2]) + self._nz * dx + _pad)
        # numpy versions of the same bounds, to sanitise maintenance results
        # before they enter the global population.
        _r_min, _r_max = float(0.25 * dx), _extent
        _c_lo_np = origin.astype(np.float32) - _pad
        _c_hi_np = (origin.astype(np.float32)
                    + np.array([self._nx, self._ny, self._nz], np.float32) * dx + _pad)

        # Bone-awareness: upload bone centres once (bind-pose, fixed during fit)
        # and a per-ellipsoid soft-count buffer (resized on maintenance).
        wp_bone_centers = None
        bone_soft_count = None
        if self._bone_aware and self._num_bones > 0:
            wp_bone_centers = wp.array(
                self._bone_centers_np, dtype=wp.vec3, device=device)
            bone_soft_count = wp.zeros(
                num_e, dtype=wp.float32, device=device, requires_grad=True)

        # Radii are trained in LOG space: scale-invariant, always-positive
        # updates.  The world radii the kernels need are derived in-tape via
        # _exp_radii_kernel.  Per-group Adam optimisers (centres / log-radii /
        # rotation) give each parameter type its own learning rate.
        def _make_opts():
            log_r = wp.array(
                np.log(np.maximum(pred_radii.numpy(), 1e-6)),
                dtype=wp.vec3, device=device, requires_grad=True)
            oc = wp.optim.Adam([pred_centers], lr=lr)
            orad = wp.optim.Adam([log_r], lr=lr)
            oq = wp.optim.Adam([pred_rot_flat], lr=lr)
            # Superquadric: per-primitive eps + bend optimisers (trained
            # directly, clamped each step).  Harmless when not used.
            oeps = wp.optim.Adam([pred_eps], lr=lr)
            obend = wp.optim.Adam([pred_bend], lr=lr)
            return (log_r, oc, orad, oq, oeps, obend,
                    [pred_centers.grad.flatten()], [log_r.grad.flatten()],
                    [pred_rot_flat.grad.flatten()], [pred_eps.grad.flatten()],
                    [pred_bend.grad.flatten()])

        self.prep_progress.emit(0.95, "optimizer")
        (pred_log_radii, opt_c, opt_r, opt_q, opt_eps, opt_bend,
         grad_c, grad_r, grad_q, grad_eps, grad_bend) = _make_opts()

        self.prep_progress.emit(1.0, "starting")
        self.phase_changed.emit("global")
        for step in range(self._num_steps):
            if self._stop_flag:
                break

            lr = self._lr_at(step)

            # Keep world radii in sync with the trainable log-radii so the
            # maintenance read-back below sees current values.
            wp.launch(_exp_radii_kernel, dim=num_e,
                      inputs=[pred_log_radii, pred_radii], device=device)

            # SuperFit handles BOTH densification and local fit, so dispatch
            # there whenever either is enabled (local fit can run with
            # densification off).  Plain maintenance only when neither is on.
            if self._superfit or self._local_fit_enabled:
                result = self._maybe_superfit(step, pred_centers, pred_radii, pred_rot_flat)
            else:
                result = self._maybe_maintain(step, pred_centers, pred_radii, pred_rot_flat)
            if result is not None:
                c_np, r_np, q_np = result
                # Sanitise the maintained set BEFORE it enters the global
                # population: a high-res local fit can occasionally return a
                # diverged primitive (NaN/inf or out-of-bounds), and the first
                # global step would otherwise compute its loss before the
                # per-step clamps run → a transient ~1e11 loss spike.
                c_np = np.clip(np.nan_to_num(np.asarray(c_np, np.float32),
                                             nan=0.0, posinf=0.0, neginf=0.0),
                               _c_lo_np, _c_hi_np).astype(np.float32)
                r_np = np.clip(np.nan_to_num(np.asarray(r_np, np.float32),
                                             nan=_r_min, posinf=_r_max, neginf=_r_min),
                               _r_min, _r_max).astype(np.float32)
                q_np = np.nan_to_num(np.asarray(q_np, np.float32),
                                     nan=0.0, posinf=0.0, neginf=0.0)
                _qn = np.linalg.norm(q_np, axis=1)
                q_np[_qn < 1e-6] = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
                # Sphere: project the maintained set to isotropic + no rotation
                # before it re-enters the optimiser (covers spawn/split/merge and
                # any local fit done inside maintenance).
                r_np, q_np = self._project_isotropic_np(r_np, q_np)
                # Capsule: circular cross-section for the maintained set.
                r_np = self._project_capsule_np(r_np)
                # Superquadric: maintenance reshuffles the population without
                # tracking eps, so reset per-primitive eps to the init values for
                # the new set; it then trains per-primitive again (freely in the
                # post-densify refinement phase).
                eps_np = self._init_eps(len(c_np))
                # Maintenance edits the full set (both halves); re-impose the
                # hard-mirror layout so only the source half stays trainable.
                if self._symmetry_enabled and self._sym_axis is not None:
                    c_np, r_np, q_np, eps_np = self._build_symmetric_layout(
                        c_np, r_np, q_np, eps_np)
                num_e = len(c_np)
                # Bend resets to straight on maintenance (not threaded through
                # the reshuffle); it re-trains per-primitive afterwards.
                bend_np = self._init_bend(num_e)
                buf = self._alloc_buffers(num_e, bs, total, c_np, r_np, q_np,
                                          eps_np=eps_np, bend_np=bend_np)
                buf['sdf_target'] = sdf_target
                pred_centers  = buf['pred_centers']
                pred_radii    = buf['pred_radii']
                pred_rot_flat = buf['pred_rot_flat']
                pred_eps      = buf['pred_eps']
                pred_bend     = buf['pred_bend']
                min_d_cache   = buf['min_d_cache']
                sdf_pred      = buf['sdf_pred']
                loss          = buf['loss']
                wp_indices    = buf['wp_indices']

                (pred_log_radii, opt_c, opt_r, opt_q, opt_eps, opt_bend,
                 grad_c, grad_r, grad_q, grad_eps, grad_bend) = _make_opts()
                soft_s_cache = wp.zeros(
                    (bs, num_e + 1), dtype=wp.float32, device=device,
                    requires_grad=True)

                if bone_soft_count is not None:
                    bone_soft_count = wp.zeros(
                        num_e, dtype=wp.float32, device=device, requires_grad=True)

            wp_indices.assign(sampler.next_batch())

            tape = wp.Tape()
            with tape:
                # Derive world radii from the trainable log-radii (gradient flows
                # back to log-space).
                wp.launch(_exp_radii_kernel, dim=num_e,
                          inputs=[pred_log_radii, pred_radii], device=device)
                min_d_cache.zero_()
                # Soft-min union only during the densification phase (dense
                # gradients → faster, more complete coverage); switch to the
                # exact HARD min for the refinement phase so the final fit has no
                # soft-union bias (train-soft / eval-hard mismatch).
                use_soft = (self._soft_union
                            and step < self._densify_until_frac * self._num_steps)
                if use_soft:
                    soft_s_cache.zero_()
                    if self._isotropic:
                        wp.launch(
                            _sphere_softmin_kernel_batch,
                            dim=bs,
                            inputs=[pred_centers, pred_radii,
                                    min_d_cache, soft_s_cache, num_e, wp_origin,
                                    float(dx), nx, ny, nz, wp_indices, sdf_pred,
                                    float(self._soft_k(step))],
                            device=device,
                        )
                    elif self._superquadric:
                        wp.launch(
                            _superquadric_softmin_kernel_batch,
                            dim=bs,
                            inputs=[pred_centers, pred_radii, pred_rot_flat,
                                    pred_eps, pred_bend,
                                    min_d_cache, soft_s_cache, num_e, wp_origin,
                                    float(dx), nx, ny, nz, wp_indices, sdf_pred,
                                    float(self._soft_k(step))],
                            device=device,
                        )
                    elif self._capsule:
                        wp.launch(
                            _capsule_softmin_kernel_batch,
                            dim=bs,
                            inputs=[pred_centers, pred_radii, pred_rot_flat,
                                    min_d_cache, soft_s_cache, num_e, wp_origin,
                                    float(dx), nx, ny, nz, wp_indices, sdf_pred,
                                    float(self._soft_k(step))],
                            device=device,
                        )
                    else:
                        wp.launch(
                            _ellipsoid_softmin_kernel_batch,
                            dim=bs,
                            inputs=[pred_centers, pred_radii, pred_rot_flat,
                                    min_d_cache, soft_s_cache, num_e, wp_origin,
                                    float(dx), nx, ny, nz, wp_indices, sdf_pred,
                                    float(self._soft_k(step))],
                            device=device,
                        )
                elif self._isotropic:
                    wp.launch(
                        _sphere_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii,
                                min_d_cache, num_e, wp_origin, float(dx),
                                nx, ny, nz, wp_indices, sdf_pred],
                        device=device,
                    )
                elif self._superquadric:
                    wp.launch(
                        _superquadric_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat,
                                pred_eps, pred_bend,
                                min_d_cache, num_e, wp_origin, float(dx),
                                nx, ny, nz, wp_indices, sdf_pred],
                        device=device,
                    )
                elif self._capsule:
                    wp.launch(
                        _capsule_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat,
                                min_d_cache, num_e, wp_origin, float(dx),
                                nx, ny, nz, wp_indices, sdf_pred],
                        device=device,
                    )
                else:
                    wp.launch(
                        _ellipsoid_sdf_kernel_batch,
                        dim=bs,
                        inputs=[pred_centers, pred_radii, pred_rot_flat,
                                min_d_cache, num_e, wp_origin, float(dx),
                                nx, ny, nz, wp_indices, sdf_pred],
                        device=device,
                    )
                loss.zero_()
                wp.launch(
                    _rmse_loss_kernel_batch,
                    dim=bs,
                    inputs=[sdf_pred, sdf_target, wp_indices, loss, bs,
                            float(self._miss_penalty_weight),
                            float(self._surface_weight), float(self._surface_sigma),
                            float(self._outside_penalty_weight),
                            self._wp_thickness, float(self._thick_ref),
                            float(self._thin_weight_eff), float(self._thin_max_factor)],
                    device=device,
                )
                if self._flat_weight > 0.0:
                    wp.launch(
                        _flatness_penalty_kernel,
                        dim=num_e,
                        inputs=[pred_radii, loss, num_e, 0,
                                float(self._flat_weight), float(self._flat_min_ratio)],
                        device=device,
                    )
                if self._containment_weight > 0.0:
                    wp.launch(
                        _containment_penalty_kernel,
                        dim=num_e,
                        inputs=[pred_centers, sdf_target, wp_origin, float(dx),
                                nx, ny, nz, loss, num_e,
                                float(self._containment_weight)],
                        device=device,
                    )
                if wp_bone_centers is not None and self._bone_span_weight > 0.0:
                    bone_soft_count.zero_()
                    wp.launch(
                        _bone_membership_kernel,
                        dim=num_e * self._num_bones,
                        inputs=[pred_centers, pred_radii, pred_rot_flat,
                                wp_bone_centers, self._num_bones,
                                float(self._bone_span_soft), bone_soft_count],
                        device=device,
                    )
                    wp.launch(
                        _bone_penalty_kernel,
                        dim=num_e,
                        inputs=[bone_soft_count, loss, num_e,
                                float(self._bone_span_weight),
                                float(self._bone_span_tol)],
                        device=device,
                    )

            tape.backward(loss)
            # Per-group learning rates (centres / log-radii / rotation).
            opt_c.lr = lr
            opt_r.lr = lr * self._lr_mult_radii
            opt_q.lr = lr * self._lr_mult_rot
            opt_c.step(grad_c)
            opt_r.step(grad_r)
            opt_q.step(grad_q)
            if self._superquadric:
                opt_eps.lr = self._sq_eps_lr   # steady, not the decaying schedule
                opt_eps.step(grad_eps)
            if self._bent:
                opt_bend.lr = self._sq_eps_lr
                opt_bend.step(grad_bend)
            tape.zero()

            # Re-normalise the trained quaternions back to unit length.  The
            # batch SDF kernel normalises internally so the fit is unaffected,
            # but the STORED quats drift off unit otherwise — and anything that
            # reads them without normalising (the SDF-slice union grid, exports)
            # would then see ellipsoids vanish (|q|>1) or balloon (|q|<1).
            wp.launch(_normalize_flat_quats, dim=num_e,
                      inputs=[pred_rot_flat], device=device)

            # Hard mirror: the trained source half moved this step → re-derive the
            # mirror half from it and re-pin the on-plane block, in place (Adam
            # state preserved).  This keeps the mirror an exact slave of the
            # source every step, so effectively only one side is ever trained.
            # Radii are projected in log-space (mirror gets the source's log-radius
            # — equivalent to copying the world radius).
            if self._symmetry_enabled and self._sym_axis is not None:
                self._project_symmetry_inplace(pred_centers, pred_log_radii,
                                               pred_rot_flat, pred_eps)

            # Sphere: project to isotropic radii + identity rotation each step.
            self._project_isotropic(pred_log_radii, pred_rot_flat, num_e)
            # Capsule: keep the cross-section circular (r1 = r0) each step.
            self._project_capsule(pred_log_radii, num_e)
            # Superquadric: keep per-primitive roundness in a safe range, and
            # bound centres + (log-)radii so the harsher SQ/bend gradients can't
            # drive a primitive to inf (a generous safety net).
            if self._superquadric:
                wp.launch(_clamp_eps, dim=2 * num_e,
                          inputs=[pred_eps, 0.1, 2.0], device=device)
                wp.launch(_clamp_log_radii, dim=num_e,
                          inputs=[pred_log_radii, _log_rmin, _log_rmax],
                          device=device)
                wp.launch(_clamp_centers_range, dim=num_e,
                          inputs=[pred_centers, _c_lo, _c_hi, 0], device=device)
            # Bent superquadric: keep the bend curvature bounded.
            if self._bent:
                wp.launch(_clamp_eps, dim=2 * num_e,
                          inputs=[pred_bend, -self._bend_max, self._bend_max],
                          device=device)

            if step % self._report_every == 0:
                # Refresh world radii from the (just-updated) log-radii for emit.
                wp.launch(_exp_radii_kernel, dim=num_e,
                          inputs=[pred_log_radii, pred_radii], device=device)
                self._emit_progress(step, loss, pred_centers, pred_radii,
                                    pred_rot_flat, num_e, origin, dx, n,
                                    pred_eps=pred_eps,
                                    pred_bend=pred_bend if self._bent else None)

                wp.synchronize_device(device)
                loss_val = float(loss.numpy()[0])
                if loss_val < 1e-10:
                    break

        # Already symmetric every step; emit the final set for completeness.
        if self._symmetry_enabled and self._sym_axis is not None:
            wp.launch(_exp_radii_kernel, dim=num_e,
                      inputs=[pred_log_radii, pred_radii], device=device)
            wp.synchronize_device(device)
            self.step_visual.emit(
                self._num_steps, float(loss.numpy()[0]),
                pred_centers.numpy().copy(), pred_radii.numpy().copy(),
                pred_rot_flat.numpy().reshape(-1, 4).copy(),
                (np.concatenate([pred_eps.numpy().reshape(-1, 2),
                                 pred_bend.numpy().reshape(-1, 2)], axis=1).copy()
                 if self._bent else
                 (pred_eps.numpy().reshape(-1, 2).copy()
                  if self._superquadric else None)))


# ── Demo helper ───────────────────────────────────────────────────────────────

def create_demo_ellipsoids(device: str = "cpu") -> EllipsoidSet:
    q_id = Ellipsoid.identity_quat()

    angle = np.radians(45.0)
    half = angle * 0.5
    q_tilt_z = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)

    angle_x = np.radians(30.0)
    half_x = angle_x * 0.5
    q_tilt_x = np.array([np.sin(half_x), 0.0, 0.0, np.cos(half_x)], dtype=np.float32)

    ellipsoids = [
        Ellipsoid(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radii=np.array([0.5, 0.3, 0.3], dtype=np.float32),
            rotation=q_id,
        ),
        Ellipsoid(
            center=np.array([0.4, 0.4, 0.0], dtype=np.float32),
            radii=np.array([0.25, 0.15, 0.2], dtype=np.float32),
            rotation=q_id,
        ),
        Ellipsoid(
            center=np.array([-0.3, -0.3, 0.2], dtype=np.float32),
            radii=np.array([0.3, 0.2, 0.15], dtype=np.float32),
            rotation=q_id,
        ),
        Ellipsoid(
            center=np.array([0.0, 0.5, -0.3], dtype=np.float32),
            radii=np.array([0.15, 0.35, 0.15], dtype=np.float32),
            rotation=q_id,
        ),
        Ellipsoid(
            center=np.array([-0.5, 0.1, 0.1], dtype=np.float32),
            radii=np.array([0.2, 0.2, 0.35], dtype=np.float32),
            rotation=q_id,
        ),
    ]

    return EllipsoidSet.from_list(ellipsoids, device=device)