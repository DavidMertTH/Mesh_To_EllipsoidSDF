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
from underrepresentation import relative_underrep_samples
from thickness import dilate_zeros
from sdf_blowup import (
    DEFAULT_MAX_THICKNESS_FRACTION,
    apply_thickness_limited_blowup,
    conservative_mirror_min,
)
from sdf_compute import _sample_voxel_field_trilinear
from sdf_samples import SdfSampleSet, UploadedSdfSamples
from fit_validation import (
    BestCheckpoint,
    Patience,
    ValidationSample,
    evaluate_validation_loss,
    stratified_validation_from_grid,
    stratified_validation_from_samples,
)
from superquadric_geometry import (
    interior_points as _sq_interior_points,
    signed_distance_batch as _sq_signed_distance_batch,
    surface_points as _sq_surface_points,
    volume as _sq_volume,
)


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
    huber_delta: float,
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
    delta = wp.max(huber_delta, float(1.0e-8))
    base = diff - float(0.5) * delta
    if diff < delta:
        base = float(0.5) * diff * diff / delta
    wp.atomic_add(loss, 0, w * base / float(batch_size))

    # Miss penalty: target inside mesh but ellipsoid says outside
    if t < float(0.0) and sp > float(0.0):
        miss = sp - t
        wp.atomic_add(loss, 0, w * miss_weight * miss / float(batch_size))

    # Protrusion penalty: target OUTSIDE the mesh but ellipsoid covers it,
    # i.e. the ellipsoid sticks out past the true surface.  Penalise each
    # sampled protruding location quadratically so a few distant bulges cannot
    # hide among many near-surface samples.  Dividing by the surface-band scale
    # keeps the weight comparable to the former linear term at one sigma while
    # making protrusions beyond that distance increasingly expensive.
    if t > float(0.0) and sp < float(0.0):
        over = t - sp
        over_scale = wp.max(surface_sigma, float(1.0e-6))
        over_penalty = over * over / over_scale
        wp.atomic_add(loss, 0, w * outside_weight * over_penalty / float(batch_size))


@wp.kernel
def _coarse_far_field_loss_kernel(
    sdf_pred: wp.array(dtype=wp.float32),
    sdf_target: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    coarse_mask: wp.array(dtype=wp.int32),
    loss: wp.array(dtype=wp.float32),
    batch_size: int,
    weight: float,
    huber_delta: float,
):
    bid = wp.tid()
    tid = indices[bid]
    if coarse_mask[tid] == 0:
        return

    # The detailed loss above intentionally saturates outside a narrow surface
    # band.  Coarse lattice samples get this weak unsaturated Huber term so the
    # SDF still supplies a direction far from the mesh without letting the low-
    # resolution field dominate surface accuracy.
    pred = wp.clamp(sdf_pred[bid], -10.0, 10.0)
    target = wp.clamp(sdf_target[tid], -10.0, 10.0)
    error = wp.abs(pred - target)
    delta = wp.max(huber_delta, float(1.0e-6))
    penalty = error - float(0.5) * delta
    if error < delta:
        penalty = float(0.5) * error * error / delta
    wp.atomic_add(loss, 0, weight * penalty / float(batch_size))


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
def _parameter_local_to_world_kernel(
    local_centers: wp.array(dtype=wp.vec3),
    local_rot_flat: wp.array(dtype=wp.float32),
    linear_row0: wp.array(dtype=wp.vec3),
    linear_row1: wp.array(dtype=wp.vec3),
    linear_row2: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=wp.vec3),
    rotation_prefixes: wp.array(dtype=wp.quat),
    world_centers: wp.array(dtype=wp.vec3),
    world_rot_flat: wp.array(dtype=wp.float32),
):
    """Fixed-pose attachment map used by bone-local corrective fitting."""
    e = wp.tid()
    lc = local_centers[e]
    world_centers[e] = wp.vec3(
        wp.dot(linear_row0[e], lc),
        wp.dot(linear_row1[e], lc),
        wp.dot(linear_row2[e], lc),
    ) + offsets[e]

    base = e * 4
    local_q = wp.normalize(wp.quat(
        local_rot_flat[base + 0],
        local_rot_flat[base + 1],
        local_rot_flat[base + 2],
        local_rot_flat[base + 3],
    ))
    world_q = wp.normalize(wp.mul(rotation_prefixes[e], local_q))
    world_rot_flat[base + 0] = world_q[0]
    world_rot_flat[base + 1] = world_q[1]
    world_rot_flat[base + 2] = world_q[2]
    world_rot_flat[base + 3] = world_q[3]


@wp.kernel
def _parameter_regularization_kernel(
    local_centers: wp.array(dtype=wp.vec3),
    local_log_radii: wp.array(dtype=wp.vec3),
    local_rot_flat: wp.array(dtype=wp.float32),
    anchor_centers: wp.array(dtype=wp.vec3),
    anchor_log_radii: wp.array(dtype=wp.vec3),
    anchor_rot_flat: wp.array(dtype=wp.float32),
    anchor_scales: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    count: int,
    center_weight: float,
    radii_weight: float,
    rotation_weight: float,
):
    e = wp.tid()
    scale = wp.max(anchor_scales[e], 1.0e-6)
    dc = (local_centers[e] - anchor_centers[e]) / scale
    dlr = local_log_radii[e] - anchor_log_radii[e]

    base = e * 4
    q = wp.normalize(wp.quat(
        local_rot_flat[base + 0], local_rot_flat[base + 1],
        local_rot_flat[base + 2], local_rot_flat[base + 3]))
    qa = wp.normalize(wp.quat(
        anchor_rot_flat[base + 0], anchor_rot_flat[base + 1],
        anchor_rot_flat[base + 2], anchor_rot_flat[base + 3]))
    qdot = wp.clamp(wp.abs(
        q[0] * qa[0] + q[1] * qa[1] + q[2] * qa[2] + q[3] * qa[3]),
        0.0, 1.0)
    penalty = (
        center_weight * wp.dot(dc, dc)
        + radii_weight * wp.dot(dlr, dlr) / 3.0
        + rotation_weight * (1.0 - qdot * qdot)
    )
    wp.atomic_add(loss, 0, penalty / float(count))


@wp.kernel
def _project_parameter_trust_region_kernel(
    local_centers: wp.array(dtype=wp.vec3),
    local_log_radii: wp.array(dtype=wp.vec3),
    anchor_centers: wp.array(dtype=wp.vec3),
    anchor_log_radii: wp.array(dtype=wp.vec3),
    anchor_scales: wp.array(dtype=wp.float32),
    center_radius_factor: float,
    log_radius_limit: float,
):
    e = wp.tid()
    if center_radius_factor > 0.0:
        delta = local_centers[e] - anchor_centers[e]
        length = wp.length(delta)
        limit = wp.max(anchor_scales[e] * center_radius_factor, 1.0e-6)
        if length > limit:
            local_centers[e] = anchor_centers[e] + delta * (limit / length)
    if log_radius_limit > 0.0:
        value = local_log_radii[e]
        anchor = anchor_log_radii[e]
        local_log_radii[e] = wp.vec3(
            wp.clamp(value[0], anchor[0] - log_radius_limit,
                     anchor[0] + log_radius_limit),
            wp.clamp(value[1], anchor[1] - log_radius_limit,
                     anchor[1] + log_radius_limit),
            wp.clamp(value[2], anchor[2] - log_radius_limit,
                     anchor[2] + log_radius_limit),
        )


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
# beta = F^(e1/2) is homogeneous of degree one (beta = 1 on the surface), the
# analogue of the ellipsoid's k0.  The signed first-order geometric distance is
# therefore (beta - 1) / |grad beta|.  Evaluating beta and its gradient through
# nested powers is numerically fragile, so the helpers below work with log(beta)
# and grad(log(beta)) throughout.  For e1 = e2 = 1 this reduces to the same
# Quilez/Taubin distance used by the ellipsoid exterior path.  e1, e2 are stored
# PER PRIMITIVE in an ``eps`` array (vec2 = (e1, e2)) and are trained like the
# other parameters — the gradient flows to centres / radii / rotation AND eps.


@wp.func
def _sq_logaddexp(a: float, b: float) -> float:
    """Two-term log-sum-exp whose exponent arguments are always non-positive."""
    m = wp.max(a, b)
    return m + wp.log(wp.exp(a - m) + wp.exp(b - m))


@wp.func
def _sq_log_beta_grad(lp: wp.vec3, r: wp.vec3,
                      e1: float, e2: float) -> wp.vec4:
    """Return ``(log(beta), grad_local(log(beta)))`` for a superquadric.

    All generalized-power sums are evaluated as log-sum-exp operations.  The
    relative coordinate floor supplies a finite, zero subgradient on symmetry
    axes (and at the centre) without perturbing ordinary surface samples.
    """
    tiny_r = float(1.0e-8)
    tiny_u = float(1.0e-7)
    se1 = wp.clamp(e1, 0.1, 2.0)
    se2 = wp.clamp(e2, 0.1, 2.0)

    rx = wp.max(wp.abs(r[0]), tiny_r)
    ry = wp.max(wp.abs(r[1]), tiny_r)
    rz = wp.max(wp.abs(r[2]), tiny_r)
    ax = wp.abs(lp[0])
    ay = wp.abs(lp[1])
    az = wp.abs(lp[2])
    sx = wp.max(ax, rx * tiny_u)
    sy = wp.max(ay, ry * tiny_u)
    sz = wp.max(az, rz * tiny_u)

    lx = (2.0 / se2) * (wp.log(sx) - wp.log(rx))
    ly = (2.0 / se2) * (wp.log(sy) - wp.log(ry))
    lz = (2.0 / se1) * (wp.log(sz) - wp.log(rz))
    la = _sq_logaddexp(lx, ly)
    lxy = (se2 / se1) * la
    lf = _sq_logaddexp(lxy, lz)
    log_beta = 0.5 * se1 * lf

    # Mixture weights give an overflow-free analytic gradient of log(beta):
    #   d log(beta)/dx = w_xy * w_x / x, and analogously for y/z.
    # x/safe_abs(x)^2 is a finite signed reciprocal with value zero on an axis.
    wx = wp.exp(lx - la)
    wy = wp.exp(ly - la)
    wxy = wp.exp(lxy - lf)
    wz = wp.exp(lz - lf)
    # Divide sequentially so an otherwise harmless very large coordinate cannot
    # overflow while forming safe_abs^2.
    gx = wxy * wx * (lp[0] / sx) / sx
    gy = wxy * wy * (lp[1] / sy) / sy
    gz = wz * (lp[2] / sz) / sz
    return wp.vec4(log_beta, gx, gy, gz)


@wp.func
def _sq_normalized_distance(log_beta: float, grad_log_beta: wp.vec3,
                            r: wp.vec3) -> float:
    """Gradient-normalized implicit distance from log(beta) and its gradient.

    Outside, ``(1 - 1/beta) / |grad log(beta)|`` avoids ever materialising a
    potentially huge beta.  Inside, beta is bounded by one and safe to form.
    Only a tiny neighbourhood of the non-differentiable centre receives a smooth
    metric floor; at and near the surface the expression is exactly
    ``(beta - 1) / |grad beta|``.
    """
    rmin = wp.max(
        wp.min(wp.min(wp.abs(r[0]), wp.abs(r[1])), wp.abs(r[2])),
        float(1.0e-8),
    )
    inv_rmin = 1.0 / rmin
    metric_eps = float(1.0e-8) * inv_rmin
    grad_log_sq = wp.dot(grad_log_beta, grad_log_beta)

    d = float(0.0)
    if log_beta >= 0.0:
        # exp(-log_beta) is in [0, 1], even for arbitrarily distant points.
        inv_beta = wp.exp(wp.max(-log_beta, -80.0))
        denom = wp.sqrt(grad_log_sq + metric_eps * metric_eps)
        d = (1.0 - inv_beta) / denom
    else:
        # beta is in (0, 1), hence cannot overflow.  beta*grad(log beta) is
        # grad(beta).  At the exact centre grad(beta) is undefined; fade in a
        # scale-aware floor below beta=1e-4 so the limiting distance is -rmin.
        beta = wp.exp(wp.max(log_beta, -80.0))
        grad_beta = beta * grad_log_beta
        beta_floor = float(1.0e-4)
        center_gate = wp.max((beta_floor - beta) / beta_floor, 0.0) * inv_rmin
        denom = wp.sqrt(
            wp.dot(grad_beta, grad_beta)
            + center_gate * center_gate
            + metric_eps * metric_eps
        )
        d = (beta - 1.0) / denom
    return d


@wp.func
def _sq_shape_distance(lp: wp.vec3, r: wp.vec3, e1: float, e2: float) -> float:
    data = _sq_log_beta_grad(lp, r, e1, e2)
    grad_log_beta = wp.vec3(data[1], data[2], data[3])
    return _sq_normalized_distance(data[0], grad_log_beta, r)


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
    # the query point back into the straight superquadric.  The inverse-warp
    # Jacobian maps the base implicit gradient into bent-local coordinates before
    # its norm is used, so the first-order distance responds to the actual surface
    # normal rather than to a direction-independent stretch estimate.
    # kx = ky = 0 reduces EXACTLY to the plain superquadric.
    lp = wp.quat_rotate_inv(q, p - c)
    z = lp[2]
    ulp = wp.vec3(lp[0] - 0.5 * kx * z * z,
                  lp[1] - 0.5 * ky * z * z,
                  z)
    data = _sq_log_beta_grad(ulp, r, e1, e2)
    grad_u = wp.vec3(data[1], data[2], data[3])
    # u = (x - .5*kx*z^2, y - .5*ky*z^2, z), hence
    # grad_lp(log beta) = J_u(lp)^T * grad_u(log beta).
    grad_lp = wp.vec3(
        grad_u[0],
        grad_u[1],
        grad_u[2] - kx * z * grad_u[0] - ky * z * grad_u[1],
    )
    return _sq_normalized_distance(data[0], grad_lp, r)


@wp.kernel
def _decode_eps_parameter(
    raw: wp.array(dtype=wp.float32),
    eps: wp.array(dtype=wp.float32),
    shared: int,
    lo: float,
    hi: float,
):
    """Smooth bounded ε parameterisation, optionally shared by all rows."""
    i = wp.tid()
    source = i
    if shared != 0:
        source = i % 2
    value = raw[source]
    sigmoid = 1.0 / (1.0 + wp.exp(-value))
    eps[i] = lo + (hi - lo) * sigmoid


@wp.kernel
def _decode_bend_parameter(
    raw_kappa: wp.array(dtype=wp.float32),
    radii: wp.array(dtype=wp.vec3),
    bend: wp.array(dtype=wp.float32),
    kappa_max: float,
):
    """Decode dimensionless curvature κ into physical k=κ/r_z."""
    i = wp.tid()
    row = i // 2
    rz = wp.max(wp.abs(radii[row][2]), float(1.0e-8))
    bend[i] = kappa_max * wp.tanh(raw_kappa[i]) / rz


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


@wp.kernel
def _superquadric_softmin_kernel_points(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    eps: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    m_cache: wp.array2d(dtype=wp.float32),
    s_cache: wp.array2d(dtype=wp.float32),
    num_e: int,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_sdf: wp.array(dtype=wp.float32),
    k: float,
):
    """Smooth SQ union for sparse point targets (same online LSE as dense)."""
    bid = wp.tid()
    p = points[indices[bid]]
    m_cache[bid, 0] = 1.0e6
    s_cache[bid, 0] = 0.0
    for i in range(num_e):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0], rot_flat[base + 1],
            rot_flat[base + 2], rot_flat[base + 3]))
        be = i * 2
        d = _bent_sq_distance(
            p, centers[i], radii[i], q,
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
def _population_adam_step_vec3(
    param: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    first: wp.array(dtype=wp.vec3),
    second: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=wp.int32),
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
):
    """Adam with an independent bias-correction age for every vec3 row."""
    i = wp.tid()
    g = grad[i]
    m = beta1 * first[i] + (1.0 - beta1) * g
    v = beta2 * second[i] + (1.0 - beta2) * wp.cw_mul(g, g)
    t = age[i] + 1
    corr1 = 1.0 - wp.pow(beta1, float(t))
    corr2 = 1.0 - wp.pow(beta2, float(t))
    mhat = m / corr1
    vhat = v / corr2
    denom = wp.vec3(
        wp.sqrt(vhat[0]) + epsilon,
        wp.sqrt(vhat[1]) + epsilon,
        wp.sqrt(vhat[2]) + epsilon,
    )
    param[i] = param[i] - lr * wp.cw_div(mhat, denom)
    first[i] = m
    second[i] = v
    age[i] = t


@wp.kernel
def _population_adam_step_f32(
    param: wp.array(dtype=wp.float32),
    grad: wp.array(dtype=wp.float32),
    first: wp.array(dtype=wp.float32),
    second: wp.array(dtype=wp.float32),
    age: wp.array(dtype=wp.int32),
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
):
    """Adam with an independent bias-correction age for every scalar."""
    i = wp.tid()
    g = grad[i]
    m = beta1 * first[i] + (1.0 - beta1) * g
    v = beta2 * second[i] + (1.0 - beta2) * g * g
    t = age[i] + 1
    mhat = m / (1.0 - wp.pow(beta1, float(t)))
    vhat = v / (1.0 - wp.pow(beta2, float(t)))
    param[i] = param[i] - lr * mhat / (wp.sqrt(vhat) + epsilon)
    first[i] = m
    second[i] = v
    age[i] = t


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
def _copy_vec3_range(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
    offset: int,
):
    tid = wp.tid()
    i = offset + tid
    dst[i] = src[i]


@wp.kernel
def _limit_center_step_by_radius(
    centers: wp.array(dtype=wp.vec3),
    prev_centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    radius_frac: float,
    min_step: float,
    max_step: float,
    offset: int,
):
    # Adam's centre LR is in world units.  Limit each actual displacement by the
    # primitive's own size so tiny ellipsoids on fingers cannot jump as far as
    # torso-sized primitives in one update.
    tid = wp.tid()
    i = offset + tid
    c0 = prev_centers[i]
    c1 = centers[i]
    d = c1 - c0
    dist = wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    r = radii[i]
    mean_r = (wp.abs(r[0]) + wp.abs(r[1]) + wp.abs(r[2])) / float(3.0)
    allowed = wp.clamp(radius_frac * mean_r, min_step, max_step)
    if dist > allowed and dist > float(1.0e-12):
        centers[i] = c0 + d * (allowed / dist)


@wp.kernel
def _limit_center_step_by_radius_perbox(
    centers: wp.array(dtype=wp.vec3),
    prev_centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    min_steps: wp.array(dtype=wp.float32),
    max_steps: wp.array(dtype=wp.float32),
    radius_frac: float,
    offset: int,
):
    """Local-fit step limiter using each high-resolution box's voxel size."""
    tid = wp.tid()
    i = offset + tid
    c0 = prev_centers[i]
    c1 = centers[i]
    d = c1 - c0
    dist = wp.length(d)
    r = radii[i]
    mean_r = (wp.abs(r[0]) + wp.abs(r[1]) + wp.abs(r[2])) / float(3.0)
    allowed = wp.clamp(radius_frac * mean_r, min_steps[tid], max_steps[tid])
    if dist > allowed and dist > float(1.0e-12):
        centers[i] = c0 + d * (allowed / dist)


@wp.kernel
def _project_local_parameter_trust_region(
    centers: wp.array(dtype=wp.vec3),
    log_radii: wp.array(dtype=wp.vec3),
    anchor_centers: wp.array(dtype=wp.vec3),
    anchor_log_radii: wp.array(dtype=wp.vec3),
    anchor_scales: wp.array(dtype=wp.float32),
    center_radius_factor: float,
    log_radius_limit: float,
    offset: int,
):
    """Keep a local refinement close to the primitive state it started from."""
    tid = wp.tid()
    i = offset + tid
    if center_radius_factor > 0.0:
        delta = centers[i] - anchor_centers[tid]
        length = wp.length(delta)
        limit = wp.max(anchor_scales[tid] * center_radius_factor, 1.0e-6)
        if length > limit:
            centers[i] = anchor_centers[tid] + delta * (limit / length)
    if log_radius_limit > 0.0:
        value = log_radii[i]
        anchor = anchor_log_radii[tid]
        log_radii[i] = wp.vec3(
            wp.clamp(value[0], anchor[0] - log_radius_limit,
                     anchor[0] + log_radius_limit),
            wp.clamp(value[1], anchor[1] - log_radius_limit,
                     anchor[1] + log_radius_limit),
            wp.clamp(value[2], anchor[2] - log_radius_limit,
                     anchor[2] + log_radius_limit),
        )


@wp.kernel
def _project_local_linear_trust_region(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    anchor_centers: wp.array(dtype=wp.vec3),
    anchor_radii: wp.array(dtype=wp.vec3),
    anchor_scales: wp.array(dtype=wp.float32),
    center_radius_factor: float,
    radius_factor: float,
    offset: int,
):
    """Equivalent safety projection for the legacy coarse-grid SGD fallback."""
    tid = wp.tid()
    i = offset + tid
    if center_radius_factor > 0.0:
        delta = centers[i] - anchor_centers[tid]
        length = wp.length(delta)
        limit = wp.max(anchor_scales[tid] * center_radius_factor, 1.0e-6)
        if length > limit:
            centers[i] = anchor_centers[tid] + delta * (limit / length)
    if radius_factor > 1.0:
        value = radii[i]
        anchor = anchor_radii[tid]
        radii[i] = wp.vec3(
            wp.clamp(wp.abs(value[0]), anchor[0] / radius_factor,
                     anchor[0] * radius_factor),
            wp.clamp(wp.abs(value[1]), anchor[1] / radius_factor,
                     anchor[1] * radius_factor),
            wp.clamp(wp.abs(value[2]), anchor[2] / radius_factor,
                     anchor[2] * radius_factor),
        )


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
SPARSE_FAR_FIELD_FRACTION = 0.20
SPARSE_FAR_FIELD_WEIGHT = 0.15


class _PopulationAdam:
    """Adam state that survives population edits by explicit row lineage.

    Warp's stock Adam owns one global time step and reallocates every moment
    buffer when a parameter array changes shape.  Adaptive primitive fitting
    changes that shape routinely.  This optimiser stores a bias-correction age
    per scalar/vec3 element, so survivors and split descendants keep their
    history while genuinely new primitives start at age zero.
    """

    def __init__(self, param, lr: float, state: dict | None = None):
        self.param = param
        self.lr = float(lr)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1.0e-8
        dtype = param.dtype
        if dtype == wp.vec3:
            moment_dtype = wp.vec3
        elif dtype == wp.float32:
            moment_dtype = wp.float32
        else:
            raise TypeError(f"unsupported population Adam dtype: {dtype}")
        self.first = wp.zeros(param.shape, dtype=moment_dtype, device=param.device)
        self.second = wp.zeros(param.shape, dtype=moment_dtype, device=param.device)
        self.age = wp.zeros(param.shape, dtype=wp.int32, device=param.device)
        if state is not None:
            self.first.assign(np.ascontiguousarray(state["first"]))
            self.second.assign(np.ascontiguousarray(state["second"]))
            self.age.assign(np.ascontiguousarray(state["age"], dtype=np.int32))

    def step(self, gradient) -> None:
        grad = gradient[0] if isinstance(gradient, (list, tuple)) else gradient
        kernel = (_population_adam_step_vec3
                  if self.param.dtype == wp.vec3 else _population_adam_step_f32)
        wp.launch(
            kernel,
            dim=len(self.param),
            inputs=[
                self.param, grad, self.first, self.second, self.age,
                float(self.lr), float(self.beta1), float(self.beta2),
                float(self.epsilon),
            ],
            device=self.param.device,
        )

    def snapshot(self) -> dict:
        wp.synchronize_device(self.param.device)
        return {
            "first": self.first.numpy().copy(),
            "second": self.second.numpy().copy(),
            "age": self.age.numpy().copy(),
        }

    @staticmethod
    def remap(snapshot: dict | None, lineage: np.ndarray, width: int = 1) -> dict | None:
        """Map old per-primitive rows to a new population.

        ``lineage[new_row]`` is an old primitive row or ``-1`` for a fresh
        primitive.  ``width`` is 1 for vec3 arrays, 4 for quaternions and 2 for
        epsilon/bend scalar arrays.
        """
        if snapshot is None:
            return None
        lineage = np.asarray(lineage, dtype=np.int64).reshape(-1)
        width = max(1, int(width))
        old_first = np.asarray(snapshot["first"])
        old_second = np.asarray(snapshot["second"])
        old_age = np.asarray(snapshot["age"], dtype=np.int32)
        if width == 1:
            new_shape = (len(lineage),) + old_first.shape[1:]
            first = np.zeros(new_shape, dtype=old_first.dtype)
            second = np.zeros(new_shape, dtype=old_second.dtype)
            age = np.zeros((len(lineage),), dtype=np.int32)
            valid = (lineage >= 0) & (lineage < len(old_first))
            first[valid] = old_first[lineage[valid]]
            second[valid] = old_second[lineage[valid]]
            age[valid] = old_age[lineage[valid]]
        else:
            first = np.zeros(len(lineage) * width, dtype=old_first.dtype)
            second = np.zeros(len(lineage) * width, dtype=old_second.dtype)
            age = np.zeros(len(lineage) * width, dtype=np.int32)
            for new_row, old_row in enumerate(lineage):
                if old_row < 0:
                    continue
                old_start = int(old_row) * width
                old_stop = old_start + width
                if old_stop > len(old_first):
                    continue
                new_start = new_row * width
                new_stop = new_start + width
                first[new_start:new_stop] = old_first[old_start:old_stop]
                second[new_start:new_stop] = old_second[old_start:old_stop]
                age[new_start:new_stop] = old_age[old_start:old_stop]
        return {"first": first, "second": second, "age": age}


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
    whole grid.  Sparse sample sets may additionally mark a coarse lattice; a
    fixed far-field quota then guarantees low-detail gradients away from the
    mesh instead of letting those few samples disappear among surface points.
    Sampling near the surface keeps the fit sharp even at high resolution,
    where surface voxels (∝ n²) would otherwise be swamped by interior/exterior
    bulk (∝ n³) under uniform sampling.

    Drawing with replacement makes each batch O(batch_size) regardless of grid
    size, so cranking the resolution no longer slows the loop down.
    """

    def __init__(self, flat_target: np.ndarray, batch_size: int, band: float,
                 surface_fraction: float, rng: np.random.Generator | None = None,
                 flat_thickness: np.ndarray | None = None, thin_bias: float = 0.0,
                 coarse_mask: np.ndarray | None = None,
                 far_field_fraction: float = SPARSE_FAR_FIELD_FRACTION):
        self.batch_size = int(batch_size)
        self._rng = rng or np.random.default_rng()
        self._all = np.arange(flat_target.size, dtype=np.int32)
        self._band = np.where(np.abs(flat_target) < band)[0].astype(np.int32)
        if self._band.size == 0:
            self._band = self._all

        self._far_field = None
        if coarse_mask is not None:
            coarse = np.asarray(coarse_mask, dtype=np.bool_).reshape(-1)
            if coarse.size != flat_target.size:
                raise ValueError("coarse_mask/target size mismatch")
            coarse_indices = np.flatnonzero(coarse).astype(np.int32)
            if coarse_indices.size > 0:
                far = coarse_indices[
                    np.abs(flat_target[coarse_indices]) >= float(band)]
                self._far_field = far if far.size > 0 else coarse_indices

        sf = float(np.clip(surface_fraction, 0.0, 1.0))
        if self._far_field is not None:
            ff = float(np.clip(far_field_fraction, 0.0, 1.0))
            self.n_far = min(
                max(1, int(round(self.batch_size * ff))), self.batch_size)
        else:
            self.n_far = 0
        self.n_surf = min(
            int(self.batch_size * sf), self.batch_size - self.n_far)
        self.n_rest = self.batch_size - self.n_surf - self.n_far

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
        if self.n_far > 0:
            parts.append(self._rng.choice(
                self._far_field, size=self.n_far, replace=True))
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
    ellipsoid_metrics = QtCore.Signal(int, object)  # (step, {metric_name: np.ndarray(N,)})
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
        loss_huber_delta_vox: float = 0.5,
        validation_sample_size: int = 4096,
        validation_every: int | None = None,
        validation_patience: int | None = 12,
        validation_min_delta: float = 0.0,
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
        local_steps: int = 400,
        local_lr: float = 0.001,
        local_center_trust_radius_factor: float = 0.75,
        local_radii_trust_factor: float = 1.5,
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
        bone_expected_counts_np: np.ndarray | None = None,
        bone_span_weight: float = 0.4,
        bone_span_tol: float = 0.35,
        bone_span_soft: float = 0.15,
        lr_init: float = 0.01,
        lr_final: float = 0.0002,
        lr_decay_k: float = 7.0,
        lr_mult_radii: float = 2.0,
        lr_mult_rot: float = 1.0,
        center_step_radius_frac: float = 0.5,
        center_step_min_vox: float = 0.25,
        center_step_max_vox: float = 4.0,
        soft_union: bool = False,   # experimental — tended to look worse in tests
        soft_union_vox_start: float = 3.0,
        soft_union_vox_end: float = 0.6,
        sdf_samples: SdfSampleSet | None = None,
        thickness_np: np.ndarray | None = None,
        sdf_blowup_offset: float = 0.0,
        sdf_blowup_max_thickness_fraction: float = (
            DEFAULT_MAX_THICKNESS_FRACTION
        ),
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
        region_steps: int = 400,
        symmetry_enabled: bool = False,
        symmetry_every: int = 100,
        primitive_shape: str = "ellipsoid",
        sq_eps1: float = 1.0,
        sq_eps2: float = 1.0,
        sq_eps_mode: str = "per_primitive",
        sq_unlock_frac: float = 0.20,
        sq_bend_unlock_frac: float = 0.40,
        sq_eps_lr_mult: float = 0.25,
        sq_bend_lr_mult: float = 0.10,
        sq_bend_kappa_max: float = 2.5,
        initial_centers: np.ndarray | None = None,
        initial_radii: np.ndarray | None = None,
        initial_rotations: np.ndarray | None = None,
        initial_eps: np.ndarray | None = None,
        initial_bend: np.ndarray | None = None,
        parameter_linear_np: np.ndarray | None = None,
        parameter_offset_np: np.ndarray | None = None,
        parameter_rotation_prefix_np: np.ndarray | None = None,
        parameter_anchor_centers: np.ndarray | None = None,
        parameter_anchor_radii: np.ndarray | None = None,
        parameter_anchor_rotations: np.ndarray | None = None,
        parameter_neighbor_centers: np.ndarray | None = None,
        parameter_neighbor_radii: np.ndarray | None = None,
        parameter_neighbor_rotations: np.ndarray | None = None,
        parameter_center_regularization: float = 0.0,
        parameter_radii_regularization: float = 0.0,
        parameter_rotation_regularization: float = 0.0,
        parameter_neighbor_center_regularization: float = 0.0,
        parameter_neighbor_radii_regularization: float = 0.0,
        parameter_neighbor_rotation_regularization: float = 0.0,
        parameter_center_trust_radius_factor: float = 0.0,
        parameter_radii_trust_factor: float = 0.0,
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
        self._sdf_samples = sdf_samples
        self._uploaded_samples: UploadedSdfSamples | None = None
        self._stop_flag = False
        self._initial_centers = (
            None if initial_centers is None
            else np.asarray(initial_centers, dtype=np.float32).reshape(-1, 3)
        )
        self._initial_radii = (
            None if initial_radii is None
            else np.asarray(initial_radii, dtype=np.float32).reshape(-1, 3)
        )
        self._initial_rotations = (
            None if initial_rotations is None
            else np.asarray(initial_rotations, dtype=np.float32).reshape(-1, 4)
        )
        self._initial_eps = (
            None if initial_eps is None
            else np.asarray(initial_eps, dtype=np.float32).reshape(-1, 2)
        )
        self._initial_bend = (
            None if initial_bend is None
            else np.asarray(initial_bend, dtype=np.float32).reshape(-1, 2)
        )
        if ((self._initial_eps is not None or self._initial_bend is not None)
                and self._initial_centers is None):
            raise ValueError(
                "initial_eps/initial_bend require initial_centers")
        if self._initial_centers is not None:
            n_init = len(self._initial_centers)
            if self._initial_radii is not None and len(self._initial_radii) != n_init:
                raise ValueError("initial_radii must match initial_centers length")
            if self._initial_rotations is not None and len(self._initial_rotations) != n_init:
                raise ValueError("initial_rotations must match initial_centers length")
            if self._initial_eps is not None and len(self._initial_eps) != n_init:
                raise ValueError("initial_eps must match initial_centers length")
            if self._initial_bend is not None and len(self._initial_bend) != n_init:
                raise ValueError("initial_bend must match initial_centers length")
            self._num_ellipsoids = n_init

        transform_parts = (
            parameter_linear_np,
            parameter_offset_np,
            parameter_rotation_prefix_np,
        )
        if any(v is not None for v in transform_parts) and not all(
                v is not None for v in transform_parts):
            raise ValueError("parameter transform requires linear, offset and rotation prefix")
        self._parameterized = all(v is not None for v in transform_parts)
        self._parameter_linear_np = None
        self._parameter_offset_np = None
        self._parameter_rotation_prefix_np = None
        if self._parameterized:
            count = int(self._num_ellipsoids)
            self._parameter_linear_np = np.asarray(
                parameter_linear_np, dtype=np.float32).reshape(count, 3, 3)
            self._parameter_offset_np = np.asarray(
                parameter_offset_np, dtype=np.float32).reshape(count, 3)
            self._parameter_rotation_prefix_np = np.asarray(
                parameter_rotation_prefix_np, dtype=np.float32).reshape(count, 4)
            if self._method != "adam":
                raise ValueError("bone-local parameter transforms require Adam")

        def _optional_parameter_set(centers, radii, rotations, label):
            values = (centers, radii, rotations)
            if not any(v is not None for v in values):
                return None
            if not all(v is not None for v in values):
                raise ValueError(f"{label} requires centers, radii and rotations")
            count = int(self._num_ellipsoids)
            return (
                np.asarray(centers, dtype=np.float32).reshape(count, 3),
                np.asarray(radii, dtype=np.float32).reshape(count, 3),
                np.asarray(rotations, dtype=np.float32).reshape(count, 4),
            )

        self._parameter_anchor = _optional_parameter_set(
            parameter_anchor_centers, parameter_anchor_radii,
            parameter_anchor_rotations, "parameter anchor")
        self._parameter_neighbor = _optional_parameter_set(
            parameter_neighbor_centers, parameter_neighbor_radii,
            parameter_neighbor_rotations, "parameter neighbor")
        self._parameter_regularization = (
            max(0.0, float(parameter_center_regularization)),
            max(0.0, float(parameter_radii_regularization)),
            max(0.0, float(parameter_rotation_regularization)),
        )
        self._parameter_neighbor_regularization = (
            max(0.0, float(parameter_neighbor_center_regularization)),
            max(0.0, float(parameter_neighbor_radii_regularization)),
            max(0.0, float(parameter_neighbor_rotation_regularization)),
        )
        self._parameter_center_trust_radius_factor = max(
            0.0, float(parameter_center_trust_radius_factor))
        radius_trust = max(0.0, float(parameter_radii_trust_factor))
        self._parameter_log_radius_limit = (
            float(np.log(max(radius_trust, 1.0))) if radius_trust > 1.0 else 0.0)
        self.optimized_parameter_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        self._maintenance_every = maintenance_every
        self._miss_penalty_weight = miss_penalty_weight
        self._outside_penalty_weight = outside_penalty_weight
        self._containment_weight = containment_weight
        self._surface_weight = surface_weight
        self._surface_sigma_vox = max(float(surface_sigma_vox), 1.0e-6)
        self._surface_sigma = max(self._surface_sigma_vox * float(dx), 1e-6)
        self._loss_huber_delta = max(
            float(loss_huber_delta_vox) * float(dx), 1.0e-8)
        self._validation_sample_size = max(1, int(validation_sample_size))
        self._validation_every = max(
            1,
            int(validation_every)
            if validation_every is not None
            else max(5 * int(report_every), 100),
        )
        if validation_patience is not None and int(validation_patience) <= 0:
            raise ValueError("validation_patience must be positive or None")
        self._validation_patience = (
            None if validation_patience is None else int(validation_patience))
        if not np.isfinite(validation_min_delta) or float(validation_min_delta) < 0.0:
            raise ValueError("validation_min_delta must be finite and non-negative")
        self._validation_min_delta = float(validation_min_delta)
        self.best_validation_loss = float("inf")
        self.best_validation_step: int | None = None
        self.validation_history: list[tuple[int, float]] = []
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
        self._local_steps = max(1, int(local_steps))
        self._local_lr = max(1.0e-8, float(local_lr))
        self._local_center_trust_radius_factor = max(
            0.0, float(local_center_trust_radius_factor))
        local_radius_trust = max(1.0, float(local_radii_trust_factor))
        self._local_log_radius_limit = float(np.log(local_radius_trust))
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
        self._live_metric = "default"
        self._last_metric_emit_step = -10**9
        # Local fit refines the worst-fitting regions.  When SuperFit is the
        # driver it reuses the densify/analysis regions (severity ≥ the floor
        # above).  But local fit can be enabled WITHOUT SuperFit, or run after the
        # densify window has closed — in those cases there are no densify regions,
        # so local fit detects its OWN worst regions here.  The floor is ~0 (any
        # voxel missed by ≥ half a voxel qualifies) so it always picks the top-K
        # worst regions and therefore fires throughout its window regardless of
        # when the window opens — not only early in training when the still-coarse
        # global fit happens to leave a high-severity region.  ``k`` bounds the
        # per-cycle high-res box fits.  Without this, local fit silently no-ops as
        # soon as the global fit has converged below the analysis severity floor.
        self._local_fit_region_k = 8
        self._local_fit_min_severity = 0.0
        # Local fit should spend its expensive high-res boxes on delicate
        # structures.  The value is the exponent for a thickness-based ranking
        # boost in ``_detect_worst_regions``: thin regions rise, thick torso-like
        # regions fall back unless their miss is overwhelmingly worse.
        self._local_fit_thin_preference = 2.0
        # Region detection evaluates a bounded, cached set of exact points from
        # the original target grid.  Thin and (when available) per-bone quotas
        # keep small structures represented without constructing a predicted n³
        # grid or decimating the target until fingers disappear.
        self._region_candidate_budget = 65_536
        self._region_thin_candidate_fraction = 0.35
        self._region_bone_candidate_fraction = 0.25
        self._region_candidate_cache: dict[tuple, np.ndarray] = {}
        self._spawn_per_round = spawn_per_round
        self._spawn_underrep = spawn_underrep
        self._split_enabled = split_enabled
        self._split_per_round = split_per_round
        self._split_margin_vox = split_margin_vox
        self._split_size_factor = split_size_factor
        self._min_split_radius_vox = min_split_radius_vox
        self._bridge_min_outside = bridge_min_outside
        self._bridge_margin_thickness_frac = 0.08
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
        self._sq_unit_moment_cache: dict[tuple[float, float], np.ndarray] = {}
        # ── bone-awareness: penalise ellipsoids spanning multiple bones ──
        self._bone_span_weight = float(bone_span_weight)
        self._bone_span_tol = float(bone_span_tol)
        self._bone_span_soft = float(bone_span_soft)
        self._bone_centers_np = None
        self._bone_expected_weights_np = None
        self._num_bones = 0
        if bone_aware and bone_centers_np is not None and len(bone_centers_np) > 0:
            self._bone_centers_np = np.ascontiguousarray(
                bone_centers_np, dtype=np.float32)
            self._num_bones = int(len(self._bone_centers_np))
            if bone_expected_counts_np is not None:
                bw = np.asarray(bone_expected_counts_np, dtype=np.float32).reshape(-1)
                if len(bw) == self._num_bones and float(np.sum(bw)) > 0.0:
                    bw = np.maximum(bw, 1.0e-6)
                    self._bone_expected_weights_np = (
                        bw / float(np.sum(bw))).astype(np.float32)
        self._bone_aware = self._num_bones > 0
        self._lr_init = lr_init
        self._lr_final = lr_final
        self._lr_decay_k = lr_decay_k
        self._lr_mult_radii = lr_mult_radii   # per-group LR (radii in log-space)
        self._lr_mult_rot = lr_mult_rot
        self._center_step_radius_frac = max(0.0, float(center_step_radius_frac))
        self._center_step_min_vox = max(0.0, float(center_step_min_vox))
        self._center_step_max_vox = max(
            self._center_step_min_vox, float(center_step_max_vox))
        # Soft-min (smooth) union of the ellipsoid SDFs — denser gradients.  The
        # blend width is annealed from ``vox_start`` → ``vox_end`` voxels over
        # training (soft early for coverage, near-hard late for accuracy).
        self._soft_union = bool(soft_union)
        self._soft_vox_start = float(soft_union_vox_start)
        self._soft_vox_end = float(soft_union_vox_end)
        self._thickness_np = thickness_np
        self._sdf_blowup_offset = float(sdf_blowup_offset)
        self._sdf_blowup_max_thickness_fraction = float(
            sdf_blowup_max_thickness_fraction)
        if not np.isfinite(self._sdf_blowup_offset):
            raise ValueError("sdf_blowup_offset must be finite")
        if (not np.isfinite(self._sdf_blowup_max_thickness_fraction)
                or not 0.0 < self._sdf_blowup_max_thickness_fraction < 0.5):
            raise ValueError(
                "sdf_blowup_max_thickness_fraction must be between 0 and 0.5")
        # Preserve the whole-mesh carrier and its geometry: local-fit methods
        # temporarily swap ``_thickness_np/_origin/_dx`` to each region grid.
        self._sdf_blowup_thickness_np = (
            None if self._sdf_blowup_offset == 0.0 or thickness_np is None
            else np.asarray(thickness_np, dtype=np.float32)
        )
        self._sdf_blowup_origin = np.asarray(origin, dtype=np.float32).reshape(3)
        self._sdf_blowup_dx = float(dx)
        self._thin_loss_weight = thin_loss_weight
        self._thin_max_factor = thin_max_factor
        self._thin_sample_bias = thin_sample_bias
        self._thickness_flat = None     # dilated flat thickness (built lazily)
        self._thickness_margin_np = None
        self._thickness_margin_source_id = None
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
        # The coarse-grid fallback uses ``local_steps``; the high-resolution
        # box fit has its own exact step budget.  Coupling both via ``max`` made
        # reducing Region steps ineffective whenever Local steps was larger.
        self._region_steps = max(1, int(region_steps))
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
        # Superquadric: bounded roundness exponents can be fixed, globally
        # shared, or independent per primitive.  The dedicated kernels evaluate
        # the generalised SDF; radii + rotation are trained as usual.
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
        mode = str(sq_eps_mode).strip().lower()
        if mode not in {"fixed", "shared", "per_primitive"}:
            raise ValueError(
                "sq_eps_mode must be 'fixed', 'shared' or 'per_primitive'")
        self._sq_eps_mode = mode
        self._sq_unlock_frac = float(np.clip(sq_unlock_frac, 0.0, 1.0))
        self._sq_bend_unlock_frac = float(
            np.clip(sq_bend_unlock_frac, 0.0, 1.0))
        self._sq_eps_lr_mult = max(0.0, float(sq_eps_lr_mult))
        self._sq_bend_lr_mult = max(0.0, float(sq_bend_lr_mult))
        # κ = k*r_z is dimensionless.  Bounding κ instead of physical k keeps
        # the same allowable bend across tiny fingers and large body primitives.
        self._bend_kappa_max = max(0.1, float(sq_bend_kappa_max))
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
        sample_total = int(sdf_samples.size) if sdf_samples is not None else total
        if batch_size is not None:
            self._batch_size = min(batch_size, sample_total)
        elif batch_fraction is not None:
            self._batch_size = max(1024, min(int(sample_total * batch_fraction), sample_total))
        else:
            # Resolution-independent budget: cost per step no longer grows with n³.
            self._batch_size = min(int(sample_budget), sample_total)

    def request_stop(self):
        self._stop_flag = True

    @staticmethod
    def _reset_stale_tape():
        """Drop a leftover process-global Warp autodiff tape, if any.

        Warp's tape lives on ``context.runtime.tape`` (one per process).  If a
        previous worker was torn down while a tape was active, the next worker's
        first ``with tape:`` raises "entering a tape while one is already
        active" — and stays broken for the rest of the session.  We serialise
        workers (only one runs at a time), so a tape found active *here* is
        always stale and safe to clear, which self-heals such a session.
        """
        try:
            import warp as wp
            rt = None
            for base in (wp, getattr(wp, "_src", None)):
                ctx = getattr(base, "context", None) if base is not None else None
                if ctx is not None and getattr(ctx, "runtime", None) is not None:
                    rt = ctx.runtime
                    break
            if rt is not None and getattr(rt, "tape", None) is not None:
                rt.tape = None
        except Exception:
            pass

    def run(self):
        # Always emit ``finished`` (even on error) so the host's pipeline — e.g.
        # sequential Bone-Separation — advances instead of hanging on a failed
        # bone.  Clear any stale global tape before starting (see above).
        try:
            self._reset_stale_tape()
            if self._method == "adam":
                self._run_adam()
            else:
                self._run_naive()
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.finished.emit()

    def symmetry_metadata(self) -> dict | None:
        """Describe the exact hard-mirror layout produced by this worker.

        The optimizer keeps its symmetric population in the stable order
        ``[on-plane | source | mirror]``.  Exposing that partition lets API
        clients preserve the actual training pairs instead of trying to infer
        them again from rounded output geometry.
        """
        if self._sym_axis is None or self._sym_plane is None:
            return None
        n_on_plane = max(0, int(self._sym_n_op))
        n_pairs = max(0, int(self._sym_n_so))
        if n_on_plane + 2 * n_pairs <= 0:
            return None
        return {
            "axis": int(self._sym_axis),
            "plane": float(self._sym_plane),
            "on_plane_count": n_on_plane,
            "pair_count": n_pairs,
        }

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
        counts = (len(c_np), len(r_np), len(q_np))
        expected = int(num_e)
        if counts != (expected, expected, expected):
            raise RuntimeError(
                "optimizer population buffers are out of sync at "
                f"step {int(step)}: expected={expected}, "
                f"centers={counts[0]}, radii={counts[1]}, "
                f"rotations={counts[2]}"
            )
        e_np = None
        metric_eps = None
        metric_bend = None
        if pred_eps is not None:
            e_np = pred_eps.numpy().reshape(-1, 2).copy()
            if len(e_np) != expected:
                raise RuntimeError(
                    "optimizer shape buffers are out of sync at "
                    f"step {int(step)}: expected={expected}, eps={len(e_np)}"
                )
            metric_eps = e_np
            if pred_bend is not None:
                # Pack bend after eps → (N,4) = [e1, e2, kx, ky] for bent shapes.
                b_np = pred_bend.numpy().reshape(-1, 2).copy()
                if len(b_np) != expected:
                    raise RuntimeError(
                        "optimizer shape buffers are out of sync at "
                        f"step {int(step)}: expected={expected}, bend={len(b_np)}"
                    )
                metric_bend = b_np
                e_np = np.concatenate([e_np, b_np], axis=1)
        self.step_visual.emit(step, loss_val, c_np, r_np, q_np, e_np)
        self._emit_live_metric_if_needed(
            step, c_np, r_np, q_np, metric_eps, metric_bend)

        # NB: the per-step ellipsoid-SDF grid + under-rep used to be computed
        # here and emitted via ``step_sdf`` for an ellipsoid slice view.  That
        # view was removed (the slice now shows the mesh only), so its consumer
        # is a no-op — computing the n³ grid every report_every·10 steps was
        # pure wasted work (costly on CPU especially) and has been dropped.
        # The spawn/maintenance path computes its own under-rep independently.

    def set_live_metric(self, metric: str) -> None:
        self._live_metric = str(metric or "default")

    def _emit_live_metric_if_needed(
        self,
        step: int,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> None:
        metric = getattr(self, "_live_metric", "default")
        if metric == "default" or self.signalsBlocked():
            return
        cheap = {
            "bridging", "too_large", "redundant", "coverage",
            "unique_coverage", "bone_over_budget",
        }
        if metric in cheap:
            every = max(1, int(self._report_every))
            regions = None
        elif metric == "too_small":
            # Region detection evaluates a bounded exact-point prediction; keep
            # it live, but not every frame.  Still more frequent than maintenance.
            every = max(50, int(self._report_every) * 3)
            if step - getattr(self, "_last_metric_emit_step", -10**9) < every:
                return
            with self._detection_grid_scope():
                regions = self._detect_worst_regions(
                    centers, radii, rotations, self._analysis_region_k,
                    min_severity=self._analysis_min_severity,
                    eps=eps, bend=bend)
        else:
            return
        if step - getattr(self, "_last_metric_emit_step", -10**9) < every:
            return
        metrics = self._compute_ellipsoid_quality_metrics(
            centers, radii, rotations, regions, only={metric},
            eps=eps, bend=bend)
        self._last_metric_emit_step = int(step)
        self.ellipsoid_metrics.emit(step, {metric: metrics.get(metric)})

    def _bone_assignments_np(self, centers: np.ndarray) -> np.ndarray | None:
        """Assign each ellipsoid to the nearest bind-pose bone centre."""
        if (not self._bone_aware or self._bone_centers_np is None
                or self._num_bones <= 0 or len(centers) == 0):
            return None
        cen = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        bones = np.asarray(self._bone_centers_np, dtype=np.float32).reshape(-1, 3)
        d2 = np.sum((cen[:, None, :] - bones[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1).astype(np.int32)

    def _bone_over_budget_scores(
        self,
        centers: np.ndarray,
        *,
        slack: float = 1.25,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return per-ellipsoid pressure when a bone has too many ellipsoids."""
        n = len(centers)
        zero = np.zeros(n, dtype=np.float32)
        assign = self._bone_assignments_np(centers)
        weights = self._bone_expected_weights_np
        if assign is None or weights is None or len(weights) != self._num_bones:
            return zero, assign, None, None

        counts = np.bincount(assign, minlength=self._num_bones).astype(np.float32)
        expected = np.maximum(weights.astype(np.float32) * float(max(n, 1)), 1.0e-6)
        allowed = np.maximum(1.0, expected * float(slack))
        over = np.maximum(0.0, counts - allowed) / np.maximum(allowed, 1.0)
        return over[assign].astype(np.float32), assign, counts, expected

    def _bone_capacity_counts(self, total_count: int | None = None) -> np.ndarray | None:
        """Hard per-bone population cap used by bone-aware spawn/split.

        ``_bone_expected_weights_np`` stores the relative mesh share per bone.
        Convert that share into a maximum count at the global population cap so
        a full bone never receives another spawned or split ellipsoid.
        """
        weights = self._bone_expected_weights_np
        if (not self._bone_aware or weights is None
                or self._num_bones <= 0 or len(weights) != self._num_bones):
            return None
        total = int(total_count if total_count is not None else self._max_ellipsoids)
        total = max(1, total)
        caps = np.ceil(weights.astype(np.float32) * float(total)).astype(np.int32)
        caps = np.maximum(caps, 1)
        return caps

    def _nearest_bone_indices_np(self, points: np.ndarray) -> np.ndarray | None:
        if (not self._bone_aware or self._bone_centers_np is None
                or self._num_bones <= 0):
            return None
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        bones = np.asarray(self._bone_centers_np, dtype=np.float32).reshape(-1, 3)
        d2 = np.sum((pts[:, None, :] - bones[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1).astype(np.int32)

    @staticmethod
    def _bone_has_add_capacity(
        bone_index: int | None,
        counts: np.ndarray | None,
        caps: np.ndarray | None,
    ) -> bool:
        if counts is None or caps is None or bone_index is None:
            return True
        bi = int(bone_index)
        if bi < 0 or bi >= len(caps):
            return True
        return int(counts[bi]) < int(caps[bi])

    def _bone_growth_state(
        self,
        centers: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        caps = self._bone_capacity_counts(self._max_ellipsoids)
        assign = self._bone_assignments_np(centers)
        if caps is None or assign is None:
            return assign, None, caps
        counts = np.bincount(assign, minlength=self._num_bones).astype(np.int32)
        return assign, counts, caps

    def _filter_spawn_candidates_by_bone_capacity(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        counts: np.ndarray | None = None,
        caps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if counts is None or caps is None or len(centers) == 0:
            return centers, radii, rotations
        assign = self._nearest_bone_indices_np(centers)
        if assign is None:
            return centers, radii, rotations
        keep: list[int] = []
        for i, bi in enumerate(assign):
            bi = int(bi)
            if self._bone_has_add_capacity(bi, counts, caps):
                keep.append(i)
                counts[bi] += 1
        if len(keep) == len(centers):
            return centers, radii, rotations
        if not keep:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )
        idx = np.asarray(keep, dtype=int)
        return centers[idx], radii[idx], rotations[idx]

    def _reserve_split_bone_capacity(
        self,
        index: int,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        bone_assign: np.ndarray | None,
        bone_counts: np.ndarray | None,
        bone_caps: np.ndarray | None,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> bool:
        """Reserve the real per-bone population delta of one split.

        A split removes one parent and adds two children.  The parent bone must
        have spare add-capacity before the split starts, and the final child
        placement may not increase any already-full bone.
        """
        if bone_counts is None or bone_caps is None:
            return True
        i = int(index)
        if i < 0 or i >= len(centers):
            return False

        parent_bone = None
        if bone_assign is not None and i < len(bone_assign):
            parent_bone = int(bone_assign[i])
        if not self._bone_has_add_capacity(parent_bone, bone_counts, bone_caps):
            return False

        e = None if eps is None else np.asarray(eps)[i]
        b = None if bend is None else np.asarray(bend)[i]
        child_c, _child_r, _child_q, _child_e, _child_b = self._split_primitive(
            centers[i], radii[i], rotations[i], e, b)
        child_assign = self._nearest_bone_indices_np(child_c)
        if child_assign is None:
            return True

        trial = np.asarray(bone_counts, dtype=np.int32).copy()
        if parent_bone is not None and 0 <= parent_bone < len(trial):
            trial[parent_bone] = max(0, int(trial[parent_bone]) - 1)
        for bi in child_assign:
            bi = int(bi)
            if 0 <= bi < len(trial):
                trial[bi] += 1

        for bi in range(min(len(trial), len(bone_caps))):
            # Existing over-cap populations are tolerated, but growth must not
            # make that bone worse.  Otherwise the hard cap is the ceiling.
            allowed = max(int(bone_counts[bi]), int(bone_caps[bi]))
            if int(trial[bi]) > allowed:
                return False
        bone_counts[:len(trial)] = trial
        return True

    # ── buffer allocation ─────────────────────────────────────────────

    def _init_inside_mesh(
        self,
        num_e: int,
        progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial ellipsoid parameters placed inside the mesh.

        Uses farthest-point sampling on interior voxels (sdf_target < 0)
        to get diverse starting positions.  Initial radii are proportional
        to local depth so ellipsoids start at a reasonable size.
        """
        origin = self._origin
        dx = self._dx
        n = self._n

        if progress_cb is not None:
            progress_cb(0.0, "finding interior voxels")
        flat_target = self._sdf_target_np.ravel()
        interior_mask = flat_target < 0.0
        interior_idx = np.where(interior_mask)[0]

        if len(interior_idx) == 0:
            # Deterministic degraded input handling: use the lowest finite SDF
            # voxels rather than injecting random, non-reproducible centres.
            finite_idx = np.flatnonzero(np.isfinite(flat_target))
            if finite_idx.size:
                order = finite_idx[np.argsort(
                    flat_target[finite_idx], kind="stable")]
                chosen = np.resize(order[:max(1, min(num_e, len(order)))], num_e)
                iz, iy, ix = np.unravel_index(chosen, self._shape)
                centers = origin + (
                    np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5
                ) * dx
            else:
                grid_extent = np.array(
                    [self._nx, self._ny, self._nz], dtype=np.float32) * dx
                centers = np.repeat(
                    (origin + 0.5 * grid_extent)[None, :], num_e, axis=0)
            radii = np.full((num_e, 3), 0.5 * float(dx), dtype=np.float32)
            rots = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_e, 1))
            if progress_cb is not None:
                progress_cb(1.0, "initial ellipsoids ready")
            return centers, radii, rots, self._init_eps(num_e)

        # Cap the FPS candidate cloud before converting to world coordinates.
        # Farthest-point sampling is O(candidates * ellipsoids), so running it
        # over every interior voxel dominates startup on fine SDF grids.  Keep
        # deep interior voxels, but spread the shortlist across depth bands so
        # thin parts still receive initial seeds.
        interior_depth = np.abs(flat_target[interior_idx])
        max_candidates = self._initial_fps_candidate_limit(num_e, len(interior_idx))
        if len(interior_idx) > max_candidates:
            if progress_cb is not None:
                progress_cb(
                    0.20,
                    f"shortlisting {max_candidates:,} of {len(interior_idx):,} interior candidates",
                )
            interior_idx = self._shortlist_initial_interior_candidates(
                interior_idx, interior_depth, max_candidates,
            )
            interior_depth = np.abs(flat_target[interior_idx])

        # Convert shortlisted interior voxels to world positions
        if progress_cb is not None:
            progress_cb(0.25, f"preparing {len(interior_idx):,} interior candidates")
        iz, iy, ix = np.unravel_index(interior_idx, self._shape)
        interior_world = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx

        # Use depth as "importance" for FPS — prefer deep interior points
        if progress_cb is not None:
            progress_cb(0.45, "placing initial ellipsoids")
        selected = self._farthest_point_sample(
            interior_world, interior_depth, num_e,
            existing_centers=np.empty((0, 3), dtype=np.float32),
        )

        centers = interior_world[selected].astype(np.float32)
        local_depth = interior_depth[selected]
        eps = self._init_eps(len(centers))
        if self._superquadric:
            if progress_cb is not None:
                progress_cb(0.70, "estimating local shape frames")
            radii, rots = self._initial_local_pca_shapes(
                centers, local_depth, eps)
        else:
            # Legacy families retain their established spherical start.
            min_r = float(dx) * 2.0
            init_r = np.clip(local_depth * 0.6, min_r, None)
            radii = np.stack([init_r, init_r, init_r], axis=1).astype(np.float32)
            rots = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                (len(centers), 1),
            )
        # Start in the hard-mirror layout so only the source half is trained from
        # step 0 and the mirror half is its slave.
        if progress_cb is not None:
            progress_cb(0.85, "finalizing initial population")
        if self._symmetry_enabled and self._sym_axis is not None:
            centers, radii, rots, eps = self._build_symmetric_layout(
                centers, radii, rots, eps)
        if progress_cb is not None:
            progress_cb(1.0, "initial ellipsoids ready")
        return centers, radii, rots, eps

    def _initial_local_pca_shapes(
        self,
        centers: np.ndarray,
        depths: np.ndarray,
        eps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Connected-component PCA plus surface-traced anisotropic SQ starts."""
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        radii_out = np.empty((len(centers), 3), dtype=np.float32)
        rotations_out = np.empty((len(centers), 4), dtype=np.float32)
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        directions = np.vstack([
            self._unit_sphere_samples(),
            np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                      for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
                     dtype=np.float32),
        ])

        for i, center in enumerate(centers):
            depth = max(float(depths[i]), 0.25 * float(self._dx))
            window_vox = float(np.clip(
                3.5 * depth / max(float(self._dx), 1.0e-12), 6.0, 24.0))
            component, component_sdf = self._initial_connected_component(
                center, window_vox)

            use_pca = len(component) >= 24
            if use_pca:
                local = component.astype(np.float64) - center.astype(np.float64)
                distance2 = np.einsum("ij,ij->i", local, local)
                sigma = max(0.55 * window_vox * float(self._dx), float(self._dx))
                weights = np.exp(-0.5 * distance2 / (sigma * sigma))
                weights *= 1.0 + 2.0 * np.exp(
                    -(component_sdf.astype(np.float64) ** 2)
                    / max((1.5 * float(self._dx)) ** 2, 1.0e-12))
                weight_sum = max(float(np.sum(weights)), 1.0e-12)
                mean = np.sum(local * weights[:, None], axis=0) / weight_sum
                centered = local - mean[None, :]
                covariance = (
                    (centered * weights[:, None]).T @ centered / weight_sum)
                covariance += np.eye(3) * float(self._dx) ** 2 / 12.0
                values, vectors = np.linalg.eigh(covariance)
                values = np.maximum(values, float(self._dx) ** 2 / 12.0)
            else:
                values = np.full(3, float(self._dx) ** 2 / 12.0)
                vectors = np.eye(3, dtype=np.float64)

            # ascending S/M/L; local z=L, x=M and y=S when those directions are
            # actually identifiable.  Degenerate eigenspaces use canonical
            # projected world axes so repeated runs cannot spin arbitrarily.
            lam_s, lam_m, lam_l = (float(v) for v in values)
            gap_long = (lam_l - lam_m) / max(lam_l, 1.0e-12)
            gap_cross = (lam_m - lam_s) / max(lam_m, 1.0e-12)

            def _smooth(value, low=0.08, high=0.25):
                t = float(np.clip((value - low) / (high - low), 0.0, 1.0))
                return t * t * (3.0 - 2.0 * t)

            confidence_long = _smooth(gap_long)
            confidence_cross = _smooth(gap_cross)
            if confidence_long <= 1.0e-6:
                matrix = np.eye(3, dtype=np.float64)
                ratios = np.ones(3, dtype=np.float64)
            else:
                z_axis = vectors[:, 2].astype(np.float64)
                pivot = int(np.argmax(np.abs(z_axis)))
                if z_axis[pivot] < 0.0:
                    z_axis *= -1.0
                reference = np.eye(3)[int(np.argmin(np.abs(z_axis)))]
                canonical_x = reference - z_axis * float(reference @ z_axis)
                canonical_x /= max(float(np.linalg.norm(canonical_x)), 1.0e-12)
                if confidence_cross > 1.0e-6:
                    x_axis = vectors[:, 1].astype(np.float64)
                    if float(x_axis @ canonical_x) < 0.0:
                        x_axis *= -1.0
                    x_axis = (
                        confidence_cross * x_axis
                        + (1.0 - confidence_cross) * canonical_x)
                    x_axis -= z_axis * float(x_axis @ z_axis)
                    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
                else:
                    x_axis = canonical_x
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
                x_axis = np.cross(y_axis, z_axis)
                matrix = np.column_stack([x_axis, y_axis, z_axis])

                lengths = np.sqrt([lam_s, lam_m, lam_l])
                rho_x = min(2.5, (lengths[1] / max(lengths[0], 1.0e-12))
                            ** confidence_cross)
                rho_z = min(5.0, rho_x * (
                    lengths[2] / max(lengths[1], 1.0e-12)) ** confidence_long)
                ratios = np.array([rho_x, 1.0, rho_z], dtype=np.float64)

            q = _rot_matrix_to_quat(matrix)
            if q[3] < 0.0:
                q = -q

            half_distances = np.empty(3, dtype=np.float64)
            max_trace = window_vox * float(self._dx)
            for axis in range(3):
                positive = self._initial_ray_surface_distance(
                    center, matrix[:, axis], max_trace)
                negative = self._initial_ray_surface_distance(
                    center, -matrix[:, axis], max_trace)
                half_distances[axis] = max(
                    min(positive, negative), 0.5 * float(self._dx))
            scale = 0.85 * float(np.min(half_distances / ratios))
            candidate_radii = np.maximum(
                scale * ratios, 0.25 * float(self._dx))

            def _surface_max_sdf(factor: float) -> float:
                points = _sq_surface_points(
                    center, (candidate_radii * factor).astype(np.float32), q,
                    eps[i], directions, np.zeros(2, np.float32))
                return float(np.max(self._grid_values_trilinear(
                    self._sdf_target_np, points)))

            # Exact family verification catches box corners and local
            # non-convexity that six ray distances cannot see.
            if _surface_max_sdf(1.0) > 0.25 * float(self._dx):
                lo, hi = 0.0, 1.0
                for _ in range(14):
                    mid = 0.5 * (lo + hi)
                    if _surface_max_sdf(mid) <= 0.25 * float(self._dx):
                        lo = mid
                    else:
                        hi = mid
                candidate_radii *= max(lo, 0.05)

            radii_out[i] = candidate_radii.astype(np.float32)
            rotations_out[i] = q.astype(np.float32)
        return radii_out, rotations_out

    def _initial_connected_component(
        self, center: np.ndarray, radius_vox: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """6-connected local interior component around one FPS seed."""
        seed = np.floor(
            (np.asarray(center, np.float32) - self._origin) / float(self._dx)
        ).astype(np.int64)
        seed = np.clip(seed, 0, np.array([
            self._nx - 1, self._ny - 1, self._nz - 1]))
        radius = max(1, int(np.ceil(radius_vox)))
        low = np.maximum(seed - radius, 0)
        high = np.minimum(
            seed + radius + 1,
            np.array([self._nx, self._ny, self._nz], dtype=np.int64))
        block = self._sdf_target_np[
            low[2]:high[2], low[1]:high[1], low[0]:high[0]]
        inside = np.isfinite(block) & (block < 0.0)
        local_seed = seed - low
        seed_zyx = (int(local_seed[2]), int(local_seed[1]), int(local_seed[0]))
        if not inside[seed_zyx]:
            return np.empty((0, 3), np.float32), np.empty(0, np.float32)

        visited = np.zeros_like(inside, dtype=bool)
        frontier = np.zeros_like(inside, dtype=bool)
        frontier[seed_zyx] = True
        visited[seed_zyx] = True
        while np.any(frontier):
            neighbor = np.zeros_like(frontier)
            neighbor[1:] |= frontier[:-1]
            neighbor[:-1] |= frontier[1:]
            neighbor[:, 1:] |= frontier[:, :-1]
            neighbor[:, :-1] |= frontier[:, 1:]
            neighbor[:, :, 1:] |= frontier[:, :, :-1]
            neighbor[:, :, :-1] |= frontier[:, :, 1:]
            frontier = neighbor & inside & ~visited
            visited |= frontier

        local_zyx = np.argwhere(visited)
        if len(local_zyx) > 20_000:
            stride = int(np.ceil(len(local_zyx) / 20_000))
            local_zyx = local_zyx[::stride]
        xyz = local_zyx[:, [2, 1, 0]].astype(np.int64) + low[None, :]
        points = (self._origin + (xyz.astype(np.float32) + 0.5)
                  * float(self._dx)).astype(np.float32)
        values = self._sdf_target_np[xyz[:, 2], xyz[:, 1], xyz[:, 0]]
        return points, values.astype(np.float32)

    def _initial_ray_surface_distance(
        self, center: np.ndarray, direction: np.ndarray, max_distance: float,
    ) -> float:
        """Sphere-trace from an interior seed and bisect the first zero crossing."""
        direction = np.asarray(direction, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        center = np.asarray(center, dtype=np.float64)
        previous_t = 0.0
        t = 0.0
        for _ in range(96):
            value = float(self._grid_values_trilinear(
                self._sdf_target_np, center + t * direction)[0])
            if value >= 0.0:
                lo, hi = previous_t, t
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    mid_value = float(self._grid_values_trilinear(
                        self._sdf_target_np, center + mid * direction)[0])
                    if mid_value < 0.0:
                        lo = mid
                    else:
                        hi = mid
                return max(lo, 0.25 * float(self._dx))
            previous_t = t
            step = float(np.clip(
                0.8 * abs(value), 0.35 * float(self._dx),
                2.0 * float(self._dx)))
            t = min(t + step, float(max_distance))
            if t >= float(max_distance):
                return float(max_distance)
        return float(max_distance)

    @staticmethod
    def _initial_fps_candidate_limit(num_e: int, n_available: int) -> int:
        """Bound startup FPS cost while scaling with requested population."""
        if n_available <= 0:
            return 0
        limit = max(4096, int(num_e) * 320)
        limit = max(int(num_e), min(240_000, limit))
        return min(int(n_available), limit)

    def _shortlist_initial_interior_candidates(
        self,
        interior_idx: np.ndarray,
        interior_depth: np.ndarray,
        max_candidates: int,
    ) -> np.ndarray:
        interior_idx = np.asarray(interior_idx, dtype=np.int64)
        depth = np.asarray(interior_depth, dtype=np.float32)
        max_candidates = int(max_candidates)
        if max_candidates <= 0 or len(interior_idx) <= max_candidates:
            return interior_idx

        finite = np.isfinite(depth)
        if not np.any(finite):
            step = max(1, int(np.ceil(len(interior_idx) / max_candidates)))
            return interior_idx[::step][:max_candidates]

        valid_idx = interior_idx[finite]
        valid_depth = depth[finite]
        if len(valid_idx) <= max_candidates:
            return valid_idx

        depth_order = np.argsort(valid_depth)
        n_bins = min(8, max(2, int(np.sqrt(max_candidates / 256.0))))
        bins = np.array_split(depth_order, n_bins)
        picks: list[np.ndarray] = []
        remaining = max_candidates

        for bi, bin_order in enumerate(bins):
            if remaining <= 0 or len(bin_order) == 0:
                continue
            # Later bins contain deeper voxels; give them more quota while still
            # preserving some near-surface/thin-structure candidates.
            weight = float(bi + 1)
            weights_left = sum(float(j + 1) for j in range(bi, len(bins)))
            quota = int(round(remaining * weight / max(weights_left, 1.0)))
            quota = max(1, min(quota, len(bin_order), remaining))
            local_depth = valid_depth[bin_order]
            if len(bin_order) > quota:
                top_local = np.argpartition(local_depth, -quota)[-quota:]
                bin_order = bin_order[top_local]
            picks.append(valid_idx[bin_order])
            remaining -= len(bin_order)

        if not picks:
            return valid_idx[:max_candidates]

        out = np.concatenate(picks)
        if len(out) < max_candidates:
            chosen = set(int(x) for x in out)
            fill = np.array(
                [int(x) for x in valid_idx if int(x) not in chosen],
                dtype=np.int64,
            )
            need = max_candidates - len(out)
            if len(fill) > need:
                fill_depth = np.abs(self._sdf_target_np.ravel()[fill])
                fill = fill[np.argpartition(fill_depth, -need)[-need:]]
            out = np.concatenate([out, fill])
        elif len(out) > max_candidates:
            out_depth = np.abs(self._sdf_target_np.ravel()[out])
            out = out[np.argpartition(out_depth, -max_candidates)[-max_candidates:]]
        return np.ascontiguousarray(out.astype(np.int64, copy=False))

    def _init_eps(self, n: int) -> np.ndarray:
        """Per-primitive superquadric exponents, initialised from the UI values."""
        return np.tile(
            np.array([self._sq_eps1, self._sq_eps2], dtype=np.float32), (int(n), 1))

    def _new_primitive_eps(
        self,
        n: int,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Initial ε rows for genuinely new primitives.

        A shared ε pair is one global model parameter, so topology edits must
        inherit its current learned value.  Reintroducing the UI prior for every
        spawn would move that parameter merely because the population changed.
        Per-primitive and fixed modes intentionally keep their configured prior.
        """
        count = int(n)
        if self._sq_eps_mode == "shared" and reference is not None:
            values = np.asarray(reference, dtype=np.float32).reshape(-1, 2)
            finite = values[np.all(np.isfinite(values), axis=1)]
            if len(finite):
                shared = np.mean(finite, axis=0, dtype=np.float64).astype(np.float32)
                return np.repeat(shared[None, :], count, axis=0)
        return self._init_eps(count)

    def _shape_state_np(
        self,
        n: int,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return validated, owned ``(eps, bend)`` arrays for a population.

        Population-management code used to carry only centers/radii/rotations,
        which silently reset learned superquadric parameters whenever buffers
        were rebuilt.  Keeping this small normalisation helper at the boundary
        makes every maintenance/local-fit path preserve the same row ordering.
        """
        n = int(n)
        if eps is None:
            eps_np = self._init_eps(n)
        else:
            eps_np = np.asarray(eps, dtype=np.float32).reshape(-1, 2)
            if len(eps_np) != n:
                raise ValueError(f"eps population mismatch: {len(eps_np)} != {n}")
            eps_np = eps_np.copy()
        if bend is None:
            bend_np = self._init_bend(n)
        else:
            bend_np = np.asarray(bend, dtype=np.float32).reshape(-1, 2)
            if len(bend_np) != n:
                raise ValueError(f"bend population mismatch: {len(bend_np)} != {n}")
            bend_np = bend_np.copy()
        return eps_np, bend_np

    def _init_bend(self, n: int) -> np.ndarray:
        """Per-primitive bend curvature (kx, ky), initialised straight (0)."""
        return np.zeros((int(n), 2), dtype=np.float32)

    def _eps_raw_np(self, eps: np.ndarray) -> np.ndarray:
        values = np.asarray(eps, dtype=np.float32).reshape(-1, 2)
        if self._sq_eps_mode == "shared":
            values = np.mean(values, axis=0, keepdims=True)
        unit = (values - 0.1) / 1.9
        unit = np.clip(unit, 1.0e-5, 1.0 - 1.0e-5)
        return np.log(unit / (1.0 - unit)).astype(np.float32)

    def _bend_raw_np(self, bend: np.ndarray, radii: np.ndarray) -> np.ndarray:
        physical = np.asarray(bend, dtype=np.float32).reshape(-1, 2)
        rz = np.maximum(
            np.abs(np.asarray(radii, dtype=np.float32).reshape(-1, 3)[:, 2:3]),
            1.0e-8,
        )
        unit = np.clip(
            physical * rz / float(self._bend_kappa_max), -0.99999, 0.99999)
        return np.arctanh(unit).astype(np.float32)

    def _eps_is_trainable(self, step: int) -> bool:
        return bool(
            self._superquadric
            and self._sq_eps_mode != "fixed"
            and int(step) >= self._sq_unlock_frac * float(self._num_steps)
        )

    def _eps_is_locally_trainable(self, step: int) -> bool:
        """Whether a region-local optimiser may update ε.

        Shared ε is global by definition and is therefore learned only from the
        global sample distribution.  A local box may update independent rows,
        but must not steer a parameter used by every primitive in the model.
        """
        return bool(
            self._sq_eps_mode == "per_primitive"
            and self._eps_is_trainable(step)
        )

    def _bend_is_trainable(self, step: int) -> bool:
        return bool(
            self._bent
            and int(step) >= self._sq_bend_unlock_frac * float(self._num_steps)
        )

    def _decode_shape_parameters(
        self,
        raw_eps,
        eps,
        raw_bend,
        bend,
        radii,
        num_e: int,
    ) -> None:
        if self._superquadric:
            wp.launch(
                _decode_eps_parameter,
                dim=2 * int(num_e),
                inputs=[raw_eps, eps, int(self._sq_eps_mode == "shared"), 0.1, 2.0],
                device=device,
            )
        if self._bent:
            wp.launch(
                _decode_bend_parameter,
                dim=2 * int(num_e),
                inputs=[raw_bend, radii, bend, float(self._bend_kappa_max)],
                device=device,
            )

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
                                rotations: np.ndarray, eps: np.ndarray | None = None,
                                bend: np.ndarray | None = None):
        """Re-order an ellipsoid set into the hard-mirror training layout.

        Output order is ``[on_plane | source | mirror]`` where:
          * on-plane ellipsoids (centre within a voxel-scale tolerance of the
            plane) are pinned to the plane with a mirror-symmetric orientation;
          * ``source`` are the off-plane ellipsoids of the better-fitting half;
          * ``mirror`` are their reflections (1:1, same order).
        Only ``[on_plane | source]`` are trained; ``mirror`` is re-derived from
        ``source`` every step.  Also records ``self._sym_n_op`` / ``self._sym_n_so``.
        Returns geometry plus ε, and also bend when the caller supplied bend.
        """
        a, p = self._sym_axis, self._sym_plane
        c = np.asarray(centers, dtype=np.float32)
        r = np.asarray(radii, dtype=np.float32)
        q = np.asarray(rotations, dtype=np.float32)
        return_bend = bend is not None
        e, b = self._shape_state_np(len(c), eps, bend)
        if a is None or len(c) == 0:
            self._sym_n_op, self._sym_n_so = 0, 0
            self._last_symmetry_lineage = np.arange(len(c), dtype=np.int64)
            return (c, r, q, e, b) if return_bend else (c, r, q, e)

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
        requested_count = int(op_idx.size + 2 * so_idx.size)
        max_count = max(1, int(self._max_ellipsoids))
        if requested_count > max_count:
            # Preserve the stable prefix.  Densify/local-fit children are
            # appended, so trimming the tail rejects only growth that did not
            # fit into the user-visible global budget.  Source entries consume
            # two slots because each requires a mirrored partner.
            if op_idx.size >= max_count:
                op_idx = op_idx[:max_count]
                so_idx = so_idx[:0]
            else:
                pair_budget = (max_count - int(op_idx.size)) // 2
                so_idx = so_idx[:pair_budget]
            print(
                "[symmetry] capped mirrored population "
                f"from {requested_count} to "
                f"{int(op_idx.size + 2 * so_idx.size)} "
                f"(max={max_count})"
            )
        n_op, n_so = int(op_idx.size), int(so_idx.size)

        # on-plane block (pinned + symmetric orientation)
        op_c = c[op_idx].copy()
        if n_op:
            op_c[:, a] = p
        op_r = r[op_idx].copy()
        op_e = e[op_idx].copy()
        op_b = b[op_idx].copy()
        # A primitive pinned to an x/y mirror plane can only be self-symmetric
        # when the corresponding quadratic bend component vanishes.  Mirroring
        # z leaves z^2 unchanged, so both bend components remain valid there.
        if n_op and a in (0, 1):
            op_b[:, a] = 0.0
        op_q = _mirror_quats_mean(q[op_idx], a) if n_op else q[op_idx]

        # source block (unchanged) + mirror block (derived from source)
        so_c, so_r, so_q = c[so_idx].copy(), r[so_idx].copy(), q[so_idx].copy()
        so_e = e[so_idx].copy()
        so_b = b[so_idx].copy()
        mi_c = so_c.copy()
        if n_so:
            mi_c[:, a] = 2.0 * p - so_c[:, a]
        mi_q = _mirror_quats(so_q, a)
        mi_b = so_b.copy()
        if a in (0, 1):
            # R' = S_world R S_local.  The same local x/y reflection maps
            # x - .5*kx*z^2 (or y - .5*ky*z^2) only if that curvature flips.
            mi_b[:, a] *= -1.0

        out_c = np.concatenate([op_c, so_c, mi_c], axis=0).astype(np.float32)
        out_r = np.concatenate([op_r, so_r, so_r], axis=0).astype(np.float32)
        out_q = np.concatenate([op_q, so_q, mi_q], axis=0).astype(np.float32)
        # eps follows radii exactly: mirror inherits the source's exponents.
        out_e = np.concatenate([op_e, so_e, so_e], axis=0).astype(np.float32)
        out_b = np.concatenate([op_b, so_b, mi_b], axis=0).astype(np.float32)
        self._last_symmetry_lineage = np.concatenate([
            op_idx.astype(np.int64, copy=False),
            so_idx.astype(np.int64, copy=False),
            so_idx.astype(np.int64, copy=False),
        ])
        if len(out_c) == 0:
            self._sym_n_op, self._sym_n_so = 0, 0
            self._last_symmetry_lineage = np.arange(len(c), dtype=np.int64)
            return (c, r, q, e, b) if return_bend else (c, r, q, e)
        self._sym_n_op, self._sym_n_so = n_op, n_so
        if return_bend:
            return out_c, out_r, out_q, out_e, out_b
        return out_c, out_r, out_q, out_e

    def _project_symmetry_inplace(self, pred_centers, pred_radii,
                                  pred_rot_flat, pred_eps=None,
                                  pred_bend=None) -> None:
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
        b = pred_bend.numpy().reshape(-1, 2) if pred_bend is not None else None

        if n_op:
            c[:n_op, a] = p
            q[:n_op] = _mirror_quats_mean(q[:n_op], a)
            if b is not None and a in (0, 1):
                b[:n_op, a] = 0.0
        if n_so:
            s0, m0 = n_op, n_op + n_so
            src_c, src_q = c[s0:s0 + n_so], q[s0:s0 + n_so]
            mc = src_c.copy(); mc[:, a] = 2.0 * p - src_c[:, a]
            c[m0:m0 + n_so] = mc
            r[m0:m0 + n_so] = r[s0:s0 + n_so]
            q[m0:m0 + n_so] = _mirror_quats(src_q, a)
            if e is not None:
                e[m0:m0 + n_so] = e[s0:s0 + n_so]   # mirror eps = source eps
            if b is not None:
                mirror_b = b[s0:s0 + n_so].copy()
                if a in (0, 1):
                    mirror_b[:, a] *= -1.0
                b[m0:m0 + n_so] = mirror_b

        pred_centers.assign(np.ascontiguousarray(c))
        pred_radii.assign(np.ascontiguousarray(r))
        pred_rot_flat.assign(np.ascontiguousarray(q.reshape(-1)))
        if e is not None:
            pred_eps.assign(np.ascontiguousarray(e.reshape(-1)))
        if b is not None:
            pred_bend.assign(np.ascontiguousarray(b.reshape(-1)))

    def _setup_symmetry(self) -> None:
        """Resolve the symmetry plane and symmetrise the target + thickness grids.

        Run once before fitting starts.  The raw target uses an average to
        remove discretisation noise.  An already blown-up target instead takes
        the less aggressive member of every mirror pair, so symmetrisation can
        never undo thin-feature protection.
        """
        self._sym_axis, self._sym_plane = self._detect_symmetry_axis(
            self._sdf_target_np)
        self._sym_checked = True
        if self._sym_axis is None:
            # Mesh is not symmetric — leave the target/thickness grids untouched.
            return
        aax = {0: 2, 1: 1, 2: 0}[self._sym_axis]
        g = self._sdf_target_np
        mirrored_grid = np.flip(g, axis=aax)
        if self._sdf_blowup_offset < 0.0:
            self._sdf_target_np = np.maximum(
                g, mirrored_grid).astype(np.float32)
        elif self._sdf_blowup_offset > 0.0:
            self._sdf_target_np = np.minimum(
                g, mirrored_grid).astype(np.float32)
        else:
            self._sdf_target_np = (
                0.5 * (g + mirrored_grid)).astype(np.float32)
        if self._thickness_np is not None:
            # A mean could let a thick side raise the permitted blowup on its
            # thinner mirror partner.  Keep the minimum when both partners are
            # resolved, while a lone resolved value repairs only a sampling
            # hole on the otherwise symmetric target.
            self._thickness_np = conservative_mirror_min(
                self._thickness_np, axis=aax)
            if self._sdf_blowup_offset != 0.0:
                self._sdf_blowup_thickness_np = self._thickness_np
        if self._sdf_samples is not None:
            self._sdf_samples = self._paired_symmetric_samples(
                self._sdf_samples,
                axis=int(self._sym_axis),
                plane=float(self._sym_plane),
                tolerance=max(1.0e-6, 1.0e-4 * float(self._dx)),
            )
            self._batch_size = min(
                int(self._batch_size), int(self._sdf_samples.size))
            self._uploaded_samples = None

    @staticmethod
    def _paired_symmetric_samples(
        samples: SdfSampleSet,
        *,
        axis: int,
        plane: float,
        tolerance: float,
    ) -> SdfSampleSet:
        """Keep one sample half and derive its exact mirrored partner half."""
        points = np.asarray(samples.points, dtype=np.float32)
        if points.size == 0:
            return samples
        signed = points[:, int(axis)] - np.float32(plane)
        on_plane = np.abs(signed) <= float(tolerance)
        positive = signed > float(tolerance)
        negative = signed < -float(tolerance)
        # Prefer +axis deterministically; fall back if a clipped sparse source
        # happens to contain almost exclusively the opposite half.
        source_side = positive
        if np.count_nonzero(positive) < max(1, np.count_nonzero(negative) // 4):
            source_side = negative
        source_idx = np.flatnonzero(on_plane | source_side)
        mirror_idx = np.flatnonzero(source_side)
        if source_idx.size == 0:
            return samples

        source_points = points[source_idx]
        mirror_points = points[mirror_idx].copy()
        mirror_points[:, int(axis)] = (
            2.0 * np.float32(plane) - mirror_points[:, int(axis)])
        paired_points = np.concatenate(
            [source_points, mirror_points], axis=0)
        paired_values = np.concatenate(
            [samples.values[source_idx], samples.values[mirror_idx]], axis=0)
        paired_thickness = None
        if samples.thickness is not None:
            paired_thickness = np.concatenate([
                samples.thickness[source_idx],
                samples.thickness[mirror_idx],
            ], axis=0)
        paired_coarse = None
        if samples.coarse_mask is not None:
            paired_coarse = np.concatenate([
                samples.coarse_mask[source_idx],
                samples.coarse_mask[mirror_idx],
            ], axis=0)
        return SdfSampleSet(
            points=paired_points,
            values=paired_values,
            thickness=paired_thickness,
            dx=float(samples.dx),
            source=f"{samples.source}-symmetric",
            coarse_mask=paired_coarse,
        )

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
        progress_cb=None,
        apply_initial_symmetry: bool = False,
    ) -> dict:
        src = self._sdf_target_np if sdf_target_np is None else sdf_target_np
        provided_centers = centers_np is not None
        if progress_cb is not None:
            progress_cb(0.02, "preparing target buffer")
        src_flat = np.ascontiguousarray(src.flatten(), dtype=np.float32)

        if centers_np is None:
            if progress_cb is not None:
                progress_cb(0.10, "initializing ellipsoids")

                def _init_progress(frac, msg):
                    progress_cb(0.10 + 0.42 * float(frac), str(msg))
            else:
                _init_progress = None
            centers_np, radii_np, rot_np, eps_np = self._init_inside_mesh(
                num_e, progress_cb=_init_progress)
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
        if (apply_initial_symmetry and provided_centers
                and self._symmetry_enabled and self._sym_axis is not None):
            centers_np, radii_np, rot_np, eps_np, bend_np = \
                self._build_symmetric_layout(
                    centers_np, radii_np, rot_np, eps_np, bend_np)
        # The actual count can differ from the requested ``num_e`` — e.g. the
        # symmetric layout returns on_plane + 2·source.  Size every buffer to it
        # so ``min_d_cache`` matches ``pred_centers`` and the kernel never reads
        # out of bounds.
        num_e = int(centers_np.shape[0])

        if progress_cb is not None:
            progress_cb(0.56, "uploading SDF target")
        sdf_target = wp.array(
            src_flat,
            dtype=wp.float32, device=device, requires_grad=False,
        )

        if progress_cb is not None:
            progress_cb(0.68, "uploading primitive parameters")
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
        pred_eps_raw = wp.array(
            np.ascontiguousarray(self._eps_raw_np(eps_np[:num_e]).reshape(-1)),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        pred_eps = wp.array(
            np.ascontiguousarray(eps_np[:num_e].reshape(-1), dtype=np.float32),
            dtype=wp.float32, device=device, requires_grad=True)
        pred_bend_raw = wp.array(
            np.ascontiguousarray(
                self._bend_raw_np(bend_np[:num_e], radii_np[:num_e]).reshape(-1)),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        pred_bend = wp.array(
            np.ascontiguousarray(bend_np[:num_e].reshape(-1), dtype=np.float32),
            dtype=wp.float32, device=device, requires_grad=True)
        # Keep host read-back valid before the first tape/maintenance cycle.
        self._decode_shape_parameters(
            pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
            pred_radii, num_e)
        if progress_cb is not None:
            progress_cb(0.84, "allocating work buffers")
        min_d_cache = wp.zeros(
            shape=(batch_size, num_e + 1),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        sdf_pred = wp.empty(
            batch_size, dtype=wp.float32, device=device, requires_grad=True,
        )
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        wp_indices = wp.empty(batch_size, dtype=wp.int32, device=device)
        if progress_cb is not None:
            progress_cb(1.0, "buffers ready")

        return dict(
            sdf_target=sdf_target,
            pred_centers=pred_centers,
            pred_radii=pred_radii,
            pred_rot_flat=pred_rot_flat,
            pred_eps_raw=pred_eps_raw,
            pred_eps=pred_eps,
            pred_bend_raw=pred_bend_raw,
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
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               bool, int, int]:
        """Coverage-based prune + proactive spawn.

        Returns ``(centers, radii, rotations, eps, bend, changed,
        n_pruned, n_spawned)``.  Shape rows follow every keep/spawn operation.
        """
        n_before = len(centers)
        eps, bend = self._shape_state_np(n_before, eps, bend)
        eps_before_edits = eps.copy()
        lineage = np.arange(n_before, dtype=np.int64)
        budget = max(1, int(n_before * self._max_prune_fraction))

        # ── 1. Remove clearly degenerate primitives ──────────────────
        volumes = self._primitive_volume_proxies(radii, eps)
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
        eps       = eps[vol_ok]
        bend      = bend[vol_ok]
        lineage   = lineage[vol_ok]
        n_removed = n_before - len(centers)
        budget -= n_removed

        # ── 2. Coverage-based pruning ─────────────────────────────────
        cov = self._compute_coverage_info(
            centers, radii, rotations, eps, bend)

        n_pruned = n_removed
        if self._prune_enabled and budget > 0 and len(centers) >= 2 and cov['valid']:
            to_remove = self._select_prune_candidates(cov, budget)
            if to_remove:
                keep_mask = np.ones(len(centers), dtype=bool)
                keep_mask[to_remove] = False
                centers   = centers[keep_mask]
                radii     = radii[keep_mask]
                rotations = rotations[keep_mask]
                eps       = eps[keep_mask]
                bend      = bend[keep_mask]
                lineage   = lineage[keep_mask]
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
                    eps       = eps[keep_mask]
                    bend      = bend[keep_mask]
                    lineage   = lineage[keep_mask]
                    n_pruned += len(swap_candidates)

        # ── 4. Spawn ─────────────────────────────────────────────────
        num_to_spawn = self._num_ellipsoids - len(centers)
        n_spawned = 0

        if num_to_spawn > 0:
            _assign, bone_counts, bone_caps = self._bone_growth_state(centers)
            new_c, new_r, new_q = self._spawn_at_errors(
                centers, radii, rotations, num_to_spawn,
                bone_counts=bone_counts, bone_caps=bone_caps,
                eps=eps, bend=bend,
                new_eps_reference=eps_before_edits,
            )
            if len(new_c) > 0:
                centers   = np.concatenate([centers, new_c], axis=0)
                radii     = np.concatenate([radii, new_r], axis=0)
                rotations = np.concatenate([rotations, new_q], axis=0)
                eps       = np.concatenate(
                    [eps, self._new_primitive_eps(
                        len(new_c), eps_before_edits)], axis=0)
                bend      = np.concatenate(
                    [bend, self._init_bend(len(new_c))], axis=0)
                lineage   = np.concatenate([
                    lineage, np.full(len(new_c), -1, dtype=np.int64)
                ])
            n_spawned = int(len(new_c))

        changed = n_pruned > 0 or n_spawned > 0
        self._last_maintenance_lineage = lineage
        return (centers, radii, rotations, eps, bend,
                changed, n_pruned, n_spawned)

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

    def _primitive_sdf_np_batch(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        points: np.ndarray,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the actual configured primitive family on CPU.

        Population management must use the same geometry as optimisation.  In
        particular, a boxy or bent superquadric must never be classified through
        an ellipsoid proxy merely because maintenance runs in NumPy.
        """
        c = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        r = np.asarray(radii, dtype=np.float32).reshape(-1, 3)
        q = np.asarray(rotations, dtype=np.float32).reshape(-1, 4)
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(c) == 0:
            return np.empty((0, len(p)), dtype=np.float32)
        if not (len(c) == len(r) == len(q)):
            raise ValueError("primitive population arrays must have equal length")
        if self._superquadric:
            e, b = self._shape_state_np(len(c), eps, bend)
            return _sq_signed_distance_batch(c, r, q, e, p, b)
        return self._ellipsoid_sdf_np_batch(c, r, q, p)

    def _primitive_sdf_np(
        self,
        center: np.ndarray,
        radii: np.ndarray,
        rotation: np.ndarray,
        points: np.ndarray,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> np.ndarray:
        e = None if eps is None else np.asarray(eps, dtype=np.float32).reshape(1, 2)
        b = None if bend is None else np.asarray(bend, dtype=np.float32).reshape(1, 2)
        return self._primitive_sdf_np_batch(
            np.asarray(center, dtype=np.float32).reshape(1, 3),
            np.asarray(radii, dtype=np.float32).reshape(1, 3),
            np.asarray(rotation, dtype=np.float32).reshape(1, 4),
            points, e, b)[0]

    def _primitive_volume_proxies(
        self,
        radii: np.ndarray,
        eps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scale-compatible volumes (ε=1 equals the historic radii product)."""
        r = np.asarray(radii, dtype=np.float32).reshape(-1, 3)
        if not self._superquadric:
            return np.prod(np.maximum(np.abs(r), 1.0e-30), axis=1)
        e, _ = self._shape_state_np(len(r), eps, None)
        sphere_factor = 4.0 * np.pi / 3.0
        return np.asarray(
            [_sq_volume(r[i], e[i]) / sphere_factor for i in range(len(r))],
            dtype=np.float64,
        )

    def _primitive_surface_points(
        self,
        center: np.ndarray,
        radii: np.ndarray,
        rotation: np.ndarray,
        directions: np.ndarray,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> np.ndarray:
        directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        directions = directions / np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
        if self._superquadric:
            e = self._init_eps(1)[0] if eps is None else np.asarray(eps, np.float32)
            b = self._init_bend(1)[0] if bend is None else np.asarray(bend, np.float32)
            return _sq_surface_points(
                center, radii, rotation, e, directions, b)
        matrix = _quat_to_rot_matrix(rotation).astype(np.float32)
        return np.ascontiguousarray(
            np.asarray(center, np.float32)
            + (directions * np.asarray(radii, np.float32)) @ matrix.T,
            dtype=np.float32,
        )

    def _primitive_interior_points(
        self,
        center: np.ndarray,
        radii: np.ndarray,
        rotation: np.ndarray,
        count: int,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
        *,
        seed: int = 0,
        beta_limit: float = 1.0,
    ) -> np.ndarray:
        if self._superquadric:
            e = self._init_eps(1)[0] if eps is None else np.asarray(eps, np.float32)
            b = self._init_bend(1)[0] if bend is None else np.asarray(bend, np.float32)
            return _sq_interior_points(
                center, radii, rotation, e, int(count), b,
                seed=int(seed), beta_limit=float(beta_limit))
        rng = np.random.default_rng(int(seed))
        dirs = rng.normal(size=(int(count), 3)).astype(np.float32)
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1.0e-12)
        radial = (np.cbrt(rng.random(int(count))) * float(beta_limit)).astype(
            np.float32)[:, None]
        matrix = _quat_to_rot_matrix(rotation).astype(np.float32)
        return np.ascontiguousarray(
            np.asarray(center, np.float32)
            + (dirs * radial * np.asarray(radii, np.float32)) @ matrix.T,
            dtype=np.float32,
        )

    def _primitive_bound_radius(
        self,
        radii: np.ndarray,
        bend: np.ndarray | None = None,
    ) -> float:
        """Conservative world-space radius for broad-phase boxes."""
        r = np.abs(np.asarray(radii, dtype=np.float64).reshape(3))
        bound = float(np.linalg.norm(r))
        if self._superquadric and bend is not None:
            k = np.abs(np.asarray(bend, dtype=np.float64).reshape(2))
            bound += 0.5 * float(np.linalg.norm(k)) * float(r[2] * r[2])
        return bound

    def _primitive_aabbs(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        bend: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conservative world AABBs, including the complete quadratic bend."""
        c = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        r = np.abs(np.asarray(radii, dtype=np.float32).reshape(-1, 3))
        q = np.asarray(rotations, dtype=np.float32).reshape(-1, 4)
        if bend is None:
            b = np.zeros((len(c), 2), dtype=np.float32)
        else:
            b = np.asarray(bend, dtype=np.float32).reshape(-1, 2)
        if self._capsule:
            capsule_half = np.stack(
                [r[:, 0], r[:, 0], r[:, 2] + r[:, 0]], axis=1,
            ).astype(np.float64)
            low = -capsule_half
            high = capsule_half
        else:
            low = -r.astype(np.float64)
            high = r.astype(np.float64)
        if self._superquadric:
            shift = 0.5 * b.astype(np.float64) * r[:, 2:3].astype(np.float64) ** 2
            low[:, :2] += np.minimum(shift, 0.0)
            high[:, :2] += np.maximum(shift, 0.0)
        local_mid = 0.5 * (low + high)
        local_half = 0.5 * (high - low)
        out_low = np.empty_like(low)
        out_high = np.empty_like(high)
        for i in range(len(c)):
            matrix = _quat_to_rot_matrix(q[i]).astype(np.float64)
            world_mid = c[i].astype(np.float64) + matrix @ local_mid[i]
            world_half = np.abs(matrix) @ local_half[i]
            out_low[i] = world_mid - world_half
            out_high[i] = world_mid + world_half
        return out_low.astype(np.float32), out_high.astype(np.float32)

    # ── coverage computation (shared by pruning + spawn) ──────────────

    def _compute_coverage_info(
        self, centers, radii, rotations, eps=None, bend=None,
    ):
        n, dx, origin = self._n, self._dx, self._origin
        flat_target = self._sdf_target_np.ravel()
        interior_idx = np.where(flat_target < 0.0)[0]
        if len(interior_idx) == 0 or len(centers) == 0:
            return {'valid': False}

        sample_size = min(self._coverage_sample_size, len(interior_idx))
        sample_flat_idx = np.random.default_rng(0).choice(interior_idx, size=sample_size, replace=False)
        iz, iy, ix = np.unravel_index(sample_flat_idx, self._shape)
        pts = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx

        per_sdf = self._primitive_sdf_np_batch(
            centers, radii, rotations, pts, eps, bend)
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
        bone_counts: np.ndarray | None = None,
        bone_caps: np.ndarray | None = None,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
        new_eps_reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Place new primitives inside the mesh, prioritising missed regions."""
        origin, dx, n = self._origin, self._dx, self._n

        pred_grid = self._pred_grid_from_params(
            centers, radii, rotations, eps, bend)

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
            new_centers = np.zeros((num_spawn, 3), dtype=np.float32)
            new_radii = np.full((num_spawn, 3), float(dx) * 3.0, dtype=np.float32)
            new_rots = np.tile(
                np.array([0., 0., 0., 1.], dtype=np.float32), (num_spawn, 1))
            return self._filter_spawn_candidates_by_bone_capacity(
                new_centers, new_radii, new_rots, bone_counts, bone_caps)

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
        shape_factor = self._spawn_shape_radius_factor(
            eps if new_eps_reference is None else new_eps_reference)
        desired_r = local_depth * (0.6 / shape_factor)
        max_inside_r = np.maximum(
            (local_depth - 0.5 * float(dx)) / shape_factor,
            0.25 * float(dx))
        init_r = np.minimum(
            np.maximum(desired_r, 0.5 * float(dx)), max_inside_r)
        new_radii = np.stack([init_r, init_r, init_r], axis=1).astype(np.float32)
        new_rots  = np.tile(np.array([0., 0., 0., 1.], dtype=np.float32), (len(new_centers), 1))
        new_centers, new_radii, new_rots = self._filter_spawn_candidates_by_bone_capacity(
            new_centers, new_radii, new_rots, bone_counts, bone_caps)
        return new_centers, new_radii, new_rots

    # ══════════════════════════════════════════════════════════════════
    # SUPERFIT — residual-region detection + isolated local fitting
    # ══════════════════════════════════════════════════════════════════

    @contextlib.contextmanager
    def _detection_grid_scope(self):
        """Compatibility scope for exact point-sampled region detection.

        Callers used to enter a temporary 64³ decimated grid here.  Detection
        is now bounded by its candidate count instead, so retaining the original
        grid is both cheaper than a full predicted grid and preserves thin parts.
        """
        yield

    @staticmethod
    def _sample_candidate_pool(
        pool: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        pool = np.asarray(pool, dtype=np.int64).reshape(-1)
        count = max(0, int(count))
        if count == 0 or pool.size == 0:
            return np.empty((0,), dtype=np.int64)
        if pool.size <= count:
            return pool.copy()
        return np.asarray(rng.choice(pool, size=count, replace=False), dtype=np.int64)

    def _bone_balanced_candidates(
        self,
        pool: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Reserve candidate slots across represented nearest-bone regions."""
        if (count <= 0 or not self._bone_aware or self._bone_centers_np is None
                or self._num_bones <= 0 or len(pool) == 0):
            return np.empty((0,), dtype=np.int64)

        probe_count = min(len(pool), max(int(count) * 4, int(count)))
        probe = self._sample_candidate_pool(pool, probe_count, rng)
        points = self._grid_points_from_flat(probe)
        bones = np.asarray(self._bone_centers_np, dtype=np.float32).reshape(-1, 3)
        assignment = np.empty(len(points), dtype=np.int32)
        for start in range(0, len(points), 8192):
            stop = min(start + 8192, len(points))
            d2 = np.sum(
                (points[start:stop, None, :] - bones[None, :, :]) ** 2,
                axis=2,
            )
            assignment[start:stop] = np.argmin(d2, axis=1).astype(np.int32)

        active = np.unique(assignment)
        if active.size == 0:
            return np.empty((0,), dtype=np.int64)
        base = int(count) // int(active.size)
        remainder = int(count) % int(active.size)
        selected = []
        for pos, bone_idx in enumerate(active):
            quota = base + (1 if pos < remainder else 0)
            group = probe[assignment == int(bone_idx)]
            selected.append(self._sample_candidate_pool(group, quota, rng))
        return (np.concatenate(selected).astype(np.int64)
                if selected else np.empty((0,), dtype=np.int64))

    def _region_candidate_indices(self) -> np.ndarray:
        """Cached exact-grid samples with guaranteed thin/bone representation."""
        key = (
            id(self._sdf_target_np), id(self._thickness_np), self._shape,
            float(self._dx), id(self._bone_centers_np), self._bone_aware,
            int(self._region_candidate_budget),
            float(self._region_thin_candidate_fraction),
            float(self._region_bone_candidate_fraction),
        )
        cached = self._region_candidate_cache.get(key)
        if cached is not None:
            return cached

        target = self._sdf_target_np.ravel()
        inside = target < 0.0
        interior = np.flatnonzero(inside).astype(np.int64)
        if interior.size == 0:
            out = np.empty((0,), dtype=np.int64)
            self._region_candidate_cache[key] = out
            return out

        budget = min(int(self._region_candidate_budget), int(interior.size))
        band_world = max(float(self._surface_band_vox) * float(self._dx),
                         float(self._dx))
        surface = np.flatnonzero(
            inside & (target >= -band_world)
        ).astype(np.int64)
        if surface.size == 0:
            surface = interior

        # A stable per-grid RNG keeps candidate coverage deterministic between
        # SuperFit cycles while avoiding flat-index lattice aliasing.
        nz, ny, nx = self._shape
        seed = (
            int(nx) * 73_856_093
            ^ int(ny) * 19_349_663
            ^ int(nz) * 83_492_791
            ^ int(round(float(self._dx) * 1.0e9))
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        selected_parts = []
        if self._thickness_np is not None:
            thickness = self._thickness_np.ravel()
            valid = thickness[interior]
            valid = valid[valid > 0.0]
            if valid.size:
                ref = float(np.median(valid))
                surface_thickness = thickness[surface]
                thin = surface[
                    (surface_thickness > 0.0)
                    & (surface_thickness < 0.5 * ref)
                ]
                n_thin = min(
                    len(thin),
                    int(round(budget * self._region_thin_candidate_fraction)),
                )
                selected_parts.append(
                    self._sample_candidate_pool(thin, n_thin, rng))

        n_bone = int(round(budget * self._region_bone_candidate_fraction))
        selected_parts.append(
            self._bone_balanced_candidates(surface, n_bone, rng))

        selected = set()
        for part in selected_parts:
            selected.update(int(v) for v in part)

        def _fill(pool: np.ndarray, desired_size: int) -> None:
            desired_size = min(int(desired_size), budget)
            for _ in range(5):
                remaining = desired_size - len(selected)
                if remaining <= 0:
                    return
                draw = self._sample_candidate_pool(
                    pool, min(len(pool), max(remaining * 2, remaining)), rng)
                for value in draw:
                    if len(selected) >= desired_size:
                        return
                    selected.add(int(value))
            # Near saturation, repeated small random draws may keep hitting the
            # existing set.  One wider final draw closes that tail without ever
            # materialising a full-grid set difference.
            remaining = desired_size - len(selected)
            if remaining > 0:
                draw = self._sample_candidate_pool(
                    pool, min(len(pool), max(remaining * 16, 1024)), rng)
                for value in draw:
                    if len(selected) >= desired_size:
                        return
                    selected.add(int(value))

        # Most samples remain close to the zero set; the tail preserves deeper
        # misses needed for safe spawn depth and volumetric coverage.
        _fill(surface, int(round(0.85 * budget)))
        _fill(interior, budget)

        out = np.fromiter(selected, dtype=np.int64, count=len(selected))
        out = np.ascontiguousarray(np.sort(out).astype(np.int64, copy=False))

        # Global + a handful of local box grids are sufficient.  Avoid retaining
        # every transient array identity in very long fitting sessions.
        if len(self._region_candidate_cache) >= 32:
            self._region_candidate_cache.clear()
        self._region_candidate_cache[key] = out
        return out

    def _pred_grid_from_params(
        self, centers, radii, rotations, eps=None, bend=None,
    ) -> np.ndarray:
        if self._superquadric or self._capsule:
            total = int(self._nx * self._ny * self._nz)
            result = np.empty(total, dtype=np.float32)
            chunk_size = 65_536
            for start in range(0, total, chunk_size):
                stop = min(start + chunk_size, total)
                flat = np.arange(start, stop, dtype=np.int64)
                points = self._grid_points_from_flat(flat)
                result[start:stop] = self._pred_points_from_params(
                    points, centers, radii, rotations, eps, bend)
            return result.reshape(self._shape)
        ell_set = EllipsoidSet(device=device)
        if len(centers) > 0:
            ell_set.set_parameters(centers, radii, rotations)
        return ell_set.compute_sdf_grid(
            self._origin, self._dx, self._n, sdf_mode=self._sdf_mode,
            shape=(self._nx, self._ny, self._nz))

    def _pred_points_from_params(
        self,
        points: np.ndarray,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the current primitive union at exact world-space points."""
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        centers = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 3)
        radii = np.ascontiguousarray(radii, dtype=np.float32).reshape(-1, 3)
        rotations = np.ascontiguousarray(rotations, dtype=np.float32).reshape(-1, 4)
        if len(points) == 0:
            return np.empty((0,), dtype=np.float32)
        if len(centers) == 0:
            return np.full(len(points), 1.0e6, dtype=np.float32)

        num_e = int(len(centers))
        wp_centers = wp.array(centers, dtype=wp.vec3, device=device)
        wp_radii = wp.array(radii, dtype=wp.vec3, device=device)
        wp_rot = wp.array(rotations.reshape(-1), dtype=wp.float32, device=device)
        if self._superquadric:
            eps_np, bend_np = self._shape_state_np(num_e, eps, bend)
            wp_eps = wp.array(
                np.ascontiguousarray(eps_np.reshape(-1)),
                dtype=wp.float32, device=device)
            wp_bend = wp.array(
                np.ascontiguousarray(bend_np.reshape(-1)),
                dtype=wp.float32, device=device)
        else:
            wp_eps = None
            wp_bend = None

        # The point kernels retain a per-sample scan over all primitives.  Chunk
        # launches to keep that temporary below roughly 32 MiB even for a large
        # ellipsoid budget.
        max_scan_values = 8_000_000
        chunk_size = max(1, max_scan_values // max(num_e + 1, 1))
        chunk_size = min(int(chunk_size), len(points))
        result = np.empty(len(points), dtype=np.float32)
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            chunk = points[start:stop]
            count = int(len(chunk))
            wp_points = wp.array(chunk, dtype=wp.vec3, device=device)
            wp_indices = wp.array(
                np.arange(count, dtype=np.int32), dtype=wp.int32, device=device)
            min_d = wp.zeros((count, num_e + 1), dtype=wp.float32, device=device)
            out = wp.empty(count, dtype=wp.float32, device=device)

            if self._isotropic:
                wp.launch(
                    _sphere_sdf_kernel_points, dim=count,
                    inputs=[wp_centers, wp_radii, min_d, num_e,
                            wp_points, wp_indices, out], device=device)
            elif self._superquadric:
                wp.launch(
                    _superquadric_sdf_kernel_points, dim=count,
                    inputs=[wp_centers, wp_radii, wp_rot, wp_eps, wp_bend,
                            min_d, num_e, wp_points, wp_indices, out],
                    device=device)
            elif self._capsule:
                wp.launch(
                    _capsule_sdf_kernel_points, dim=count,
                    inputs=[wp_centers, wp_radii, wp_rot, min_d, num_e,
                            wp_points, wp_indices, out], device=device)
            else:
                wp.launch(
                    _ellipsoid_sdf_kernel_points, dim=count,
                    inputs=[wp_centers, wp_radii, wp_rot, min_d, num_e,
                            wp_points, wp_indices, out], device=device)
            result[start:stop] = out.numpy()
        return result

    def _detect_worst_regions(self, centers, radii, rotations, k,
                               min_severity: float = 0.0,
                               thin_preference: float = 0.0,
                               eps: np.ndarray | None = None,
                               bend: np.ndarray | None = None):
        """Find up to ``k`` *spatially-separated* under-represented regions.

        A bounded candidate set is sampled once from exact voxel centres of the
        current target grid.  It reserves explicit quotas for thin surface
        regions and, when rig information is available, nearest-bone regions.
        The primitive union is evaluated only at those points, avoiding both an
        O(n³) predicted grid and the old coarse-grid aliasing of fingers.

        Greedy peak picking takes the worst sampled point and suppresses nearby
        samples in world space.  Returns region dicts (worst-first) with:
          - ``seed_world``  : (3,) world position of the peak-severity voxel
          - ``pool_flat``   : flat voxel indices of the local interior pool
          - ``seed_depth``  : local feature thickness at the seed (|target|)
          - ``severity``    : peak severity value
          - ``rank_score``  : score used for greedy picking
          - ``seed_thickness``: local feature thickness at the seed, if known

        ``min_severity`` stops the search as soon as the next-worst peak falls
        below it (severity = relative miss × surface emphasis).

        ``thin_preference`` only changes the ranking, not the underlying
        severity threshold.  With a thickness grid present, scores are multiplied
        by ``(median_thickness / local_thickness) ** thin_preference``.  This
        pushes local fit toward fingers, tails, ears, etc. and away from large
        thick masses such as a torso/belly.
        """
        candidate_flat = self._region_candidate_indices()
        if candidate_flat.size == 0 or int(k) <= 0:
            return []

        dx = float(self._dx)
        points = self._grid_points_from_flat(candidate_flat)
        flat_target = self._sdf_target_np.ravel()
        target = flat_target[candidate_flat].astype(np.float32, copy=False)
        if eps is None and bend is None:
            # Keep geometry-only detector stubs/callers compatible; SQ paths
            # provide the learned arrays explicitly.
            pred = self._pred_points_from_params(
                points, centers, radii, rotations)
        else:
            pred = self._pred_points_from_params(
                points, centers, radii, rotations, eps=eps, bend=bend)
        flat_thick = (self._thickness_np.ravel()
                      if self._thickness_np is not None else None)
        thickness = (flat_thick[candidate_flat].astype(np.float32, copy=False)
                     if flat_thick is not None else None)

        severity = relative_underrep_samples(
            target, pred, dx,
            surface_weight=self._surface_weight,
            surface_sigma_vox=max(self._surface_sigma / max(dx, 1e-12), 1e-6),
            min_gap_vox=self._underrep_min_gap_vox,
            thickness_values=thickness,
            min_thickness_vox=self._underrep_min_thickness_vox,
        )
        rank = severity.copy()
        if thickness is not None and float(thin_preference) > 0.0:
            valid = thickness > 0.0
            if np.any(valid):
                ref = float(np.median(thickness[valid]))
                if ref > 1e-12:
                    boost = np.ones_like(thickness, dtype=np.float32)
                    thick = np.maximum(thickness[valid], float(dx))
                    boost[valid] = np.clip(
                        (ref / thick) ** float(thin_preference), 0.05, 8.0)
                    rank *= boost

        regions = []
        floor = max(0.0, float(min_severity))
        suppression_radius = max(
            2.0 * float(self._region_radius_vox) * dx, dx)
        for _ in range(int(k)):
            seed_pos = int(np.argmax(rank))
            rank_peak = float(rank[seed_pos])
            peak = float(severity[seed_pos])
            if rank_peak <= 0.0 or peak <= floor:
                break
            seed_flat = int(candidate_flat[seed_pos])
            seed_world = points[seed_pos].astype(np.float32).copy()
            pool_flat = self._interior_ball_pool(
                seed_world, self._region_radius_vox)
            if pool_flat.size == 0:
                pool_flat = np.array([seed_flat], dtype=np.int64)

            regions.append(dict(
                seed_world=seed_world,
                pool_flat=pool_flat.astype(np.int32),
                seed_depth=float(abs(target[seed_pos])),
                severity=peak,
                rank_score=rank_peak,
                seed_thickness=(None if thickness is None
                                else float(thickness[seed_pos])),
            ))

            delta = points - seed_world[None, :]
            suppress = np.einsum("ij,ij->i", delta, delta) <= suppression_radius ** 2
            severity[suppress] = 0.0
            rank[suppress] = 0.0

        return regions

    def _spawn_shape_radius_factor(
        self, reference_eps: np.ndarray | None = None,
    ) -> float:
        """Circumradius of a unit spawn shape (new bends always start at zero)."""
        if not self._superquadric:
            return 1.0
        corners = np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0)
             for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float32)
        directions = np.vstack([self._unit_sphere_samples(), corners])
        unit_surface = _sq_surface_points(
            np.zeros(3, np.float32), np.ones(3, np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], np.float32),
            self._new_primitive_eps(1, reference_eps)[0],
            directions, np.zeros(2, np.float32))
        return max(1.0, 1.01 * float(np.max(np.linalg.norm(
            unit_surface, axis=1))))

    def _spawn_in_regions(
        self,
        regions,
        budget,
        bone_counts: np.ndarray | None = None,
        bone_caps: np.ndarray | None = None,
        reference_eps: np.ndarray | None = None,
    ):
        """Spawn new primitives at under-represented region seeds.

        Each new primitive starts isotropic and straight at the region's
        peak-severity interior voxel.  Its family-specific circumradius is
        bounded by the local interior depth, so boxy SQ corners stay inside too.
        Regions too shallow to host a meaningful primitive are skipped.

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
        shape_radius_factor = self._spawn_shape_radius_factor(reference_eps)

        cs, rs, qs, sites = [], [], [], []
        for reg in regions:
            if len(cs) >= int(budget):
                break
            depth = float(reg["seed_depth"])
            if depth < min_depth:
                continue
            # Family circumradius < depth keeps the complete primitive inside.
            rad = float(min(
                inside_frac * depth / shape_radius_factor,
                (depth - 0.5 * dx) / shape_radius_factor))
            if rad <= 0.0:
                continue
            c = np.asarray(reg["seed_world"], dtype=np.float32)
            bone_idx = None
            if bone_counts is not None and bone_caps is not None:
                assign = self._nearest_bone_indices_np(c.reshape(1, 3))
                if assign is not None:
                    bone_idx = int(assign[0])
                if not self._bone_has_add_capacity(bone_idx, bone_counts, bone_caps):
                    continue
            cs.append(c)
            rs.append(np.array([rad, rad, rad], dtype=np.float32))
            qs.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            sites.append((c.copy(), max(r_region_world, rad * 2.0)))
            if bone_idx is not None:
                bone_counts[bone_idx] += 1

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
        k = min(int(k), int(n_cand))

        candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 3)
        errors = np.asarray(errors, dtype=np.float32).reshape(-1)
        if len(existing_centers) > 0:
            existing_centers = np.asarray(existing_centers, dtype=np.float32).reshape(-1, 3)
            min_dists2 = np.full(n_cand, np.inf, dtype=np.float32)
            for center in existing_centers:
                diff = candidates - center
                d2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
                min_dists2 = np.minimum(min_dists2, d2)
        else:
            min_dists2 = np.full(n_cand, 1e12, dtype=np.float32)

        selected = []
        for _ in range(k):
            scores = min_dists2 * (errors + 1e-8)
            best = int(np.argmax(scores))
            selected.append(best)
            diff = candidates - candidates[best]
            new_dists2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
            min_dists2 = np.minimum(min_dists2, new_dists2)
            min_dists2[best] = 0.0

        return np.array(selected, dtype=int)

    # ══════════════════════════════════════════════════════════════════
    # TRAINING LOOPS
    # ══════════════════════════════════════════════════════════════════

    # ── SuperFit: spawn at worst region + isolated local fit ──────────

    def _grid_value(self, grid: np.ndarray, world_pt: np.ndarray) -> float:
        """Nearest-voxel value of a (nz, ny, nx) grid at a world-space point."""
        return float(self._grid_values(grid, np.asarray(world_pt).reshape(1, 3))[0])

    def _grid_values(self, grid: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
        """Nearest-voxel values for a world-space point array."""
        pts = np.asarray(world_pts, dtype=np.float32).reshape(-1, 3)
        q = (pts - self._origin) / float(self._dx)
        hi = np.array([self._nx - 1, self._ny - 1, self._nz - 1])
        ijk = np.clip(np.floor(q).astype(np.int64), 0, hi)
        return np.asarray(grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]])

    def _grid_values_trilinear(
        self, grid: np.ndarray, world_pts: np.ndarray,
    ) -> np.ndarray:
        """Trilinear values under the project's voxel-centre convention."""
        pts = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
        coord = ((pts - self._origin.astype(np.float64)) / float(self._dx)
                 - 0.5)
        shape_xyz = np.array([self._nx, self._ny, self._nz], dtype=np.int64)
        outside = np.any((coord < 0.0) | (coord > shape_xyz - 1.0), axis=1)
        coord = np.clip(coord, 0.0, shape_xyz.astype(np.float64) - 1.0)
        lower = np.floor(coord).astype(np.int64)
        upper = np.minimum(lower + 1, shape_xyz - 1)
        frac = coord - lower
        result = np.zeros(len(pts), dtype=np.float64)
        data = np.asarray(grid)
        for bx in (0, 1):
            wx = (1.0 - frac[:, 0]) if bx == 0 else frac[:, 0]
            ix = lower[:, 0] if bx == 0 else upper[:, 0]
            for by in (0, 1):
                wy = (1.0 - frac[:, 1]) if by == 0 else frac[:, 1]
                iy = lower[:, 1] if by == 0 else upper[:, 1]
                for bz in (0, 1):
                    wz = (1.0 - frac[:, 2]) if bz == 0 else frac[:, 2]
                    iz = lower[:, 2] if bz == 0 else upper[:, 2]
                    result += wx * wy * wz * data[iz, iy, ix]
        # Leaving the sampled grid cannot be considered safely inside.
        result[outside] = np.maximum(result[outside], float(self._dx))
        return result.astype(np.float32)

    def _sample_global_thickness_points(self, points: np.ndarray) -> np.ndarray:
        """Nearest-neighbour global thickness samples for world-space points."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if self._thickness_np is None or pts.size == 0:
            return np.zeros(pts.shape[0], dtype=np.float32)
        rel = (pts - self._origin.astype(np.float32)) / float(self._dx)
        ijk = np.floor(rel).astype(np.int32)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, self._nx - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, self._ny - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, self._nz - 1)
        return self._thickness_np[
            ijk[:, 2], ijk[:, 1], ijk[:, 0],
        ].astype(np.float32, copy=False)

    def _bridge_margin_values(self, ijk: np.ndarray, base_margin: float) -> np.ndarray | float:
        """Per-sample outside margin for bridging tests.

        The floor is the old voxel-noise margin.  Where a local-thickness field
        exists, thick regions get a proportionally wider tolerance, while thin
        regions stay sensitive.  The thickness map is lightly dilated so samples
        just outside the mesh inherit nearby feature thickness instead of zero.
        """
        thick = self._thickness_np
        if thick is None:
            return float(base_margin)
        source_id = id(thick)
        if (self._thickness_margin_np is None
                or self._thickness_margin_np.shape != thick.shape
                or self._thickness_margin_source_id != source_id):
            self._thickness_margin_np = dilate_zeros(thick, iters=2).astype(np.float32)
            self._thickness_margin_source_id = source_id
        th = self._thickness_margin_np[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        return np.maximum(
            float(base_margin),
            float(self._bridge_margin_thickness_frac) * th,
        ).astype(np.float32)

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

    def _primitive_protrusion_scores(
        self, centers, radii, rotations, eps=None, bend=None,
    ) -> np.ndarray:
        """Actual-surface protrusion score for every configured primitive."""
        count = len(centers)
        scores = np.zeros(count, dtype=np.float32)
        if count == 0:
            return scores
        eps, bend = self._shape_state_np(count, eps, bend)
        target = self._sdf_target_np
        dx = float(self._dx)
        margin = float(self._split_margin_vox) * dx
        min_r = float(self._min_split_radius_vox) * dx
        f = max(float(self._split_size_factor), 1.0 + 1e-6)
        rel_thresh = 1.0 - 1.0 / f
        axes = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
        directions = np.vstack([axes, self._unit_sphere_samples()])
        for i in range(count):
            if self._primitive_bound_radius(radii[i], bend[i]) < min_r:
                continue
            points = self._primitive_surface_points(
                centers[i], radii[i], rotations[i], directions,
                eps[i], bend[i])
            values = self._grid_values(target, points).astype(np.float32)
            support = np.maximum(
                np.linalg.norm(points - centers[i][None, :], axis=1), dx)
            relative = values / support
            valid = (values > margin) & (relative > rel_thresh)
            if not np.any(valid):
                continue
            scores[i] = float(np.max(relative[valid]))
        return scores

    def _detect_protruding_ellipsoids(
        self, centers, radii, rotations, eps=None, bend=None,
    ) -> np.ndarray:
        """Indices of primitives whose true surface protrudes, worst first."""
        scores = self._primitive_protrusion_scores(
            centers, radii, rotations, eps, bend)
        idx = np.flatnonzero(scores > 0.0)
        if idx.size == 0:
            return np.array([], dtype=int)
        return idx[np.argsort(-scores[idx], kind="stable")].astype(int)

    def _primitive_bridging_scores(
        self, centers, radii, rotations, eps=None, bend=None,
    ) -> np.ndarray:
        """Fraction of each primitive's real volume lying outside the mesh.

        The deterministic probes are volume-uniform in SQ space and are bent
        with the exact forward map, so box corners and curved gaps contribute in
        the same proportions as they do to the fitted primitive.
        """
        n_ell = len(centers)
        scores = np.zeros(n_ell, dtype=np.float32)
        if n_ell == 0:
            return scores

        eps, bend = self._shape_state_np(n_ell, eps, bend)
        sample_count = max(64, int(self._fuse_samples))
        target = self._sdf_target_np
        thick = self._thickness_np
        origin = self._origin.astype(np.float32)
        dx = float(self._dx)
        margin = float(self._split_margin_vox) * dx
        min_r = float(self._min_split_radius_vox) * dx

        for i in range(n_ell):
            bound = self._primitive_bound_radius(radii[i], bend[i])
            if bound < min_r:
                continue
            pts = self._primitive_interior_points(
                centers[i], radii[i], rotations[i], sample_count,
                eps[i], bend[i], seed=0xB123 + i)
            q = (pts - origin) / dx
            ijk = np.clip(np.floor(q).astype(np.int64), 0,
                          np.array([self._nx - 1, self._ny - 1, self._nz - 1]))
            vals = target[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
            margins = self._bridge_margin_values(ijk, margin)
            outside_frac = float(np.mean(vals > margins))
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
                    oversized = bound > self._split_size_factor * (0.5 * th_c)
            if not oversized:
                continue
            scores[i] = outside_frac
        return scores

    def _detect_bridging_ellipsoids(
        self, centers, radii, rotations, eps=None, bend=None,
    ) -> np.ndarray:
        """Indices of true primitive volumes bridging empty space, worst first."""
        scores = self._primitive_bridging_scores(
            centers, radii, rotations, eps, bend)
        idx = np.flatnonzero(scores > 0.0)
        if idx.size == 0:
            return np.array([], dtype=int)
        return idx[np.argsort(-scores[idx], kind="stable")].astype(int)

    def _compute_ellipsoid_quality_metrics(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        regions: list | None = None,
        only: set[str] | None = None,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-ellipsoid scalar diagnostics used by the viewport heatmap.

        Values are raw scores; the viewer normalises them for display.  Higher
        means "more of this thing" for every metric.
        """
        n_ell = len(centers)
        zeros = np.zeros(n_ell, dtype=np.float32)
        requested = set(only) if only is not None else {
            "bridging", "too_large", "too_small",
            "redundant", "coverage", "unique_coverage",
            "bone_over_budget",
        }
        if n_ell == 0:
            return {k: zeros for k in requested}
        eps, bend = self._shape_state_np(n_ell, eps, bend)

        bridging = np.zeros(n_ell, dtype=np.float32)
        if "bridging" in requested:
            bridging = self._primitive_bridging_scores(
                centers, radii, rotations, eps, bend)

        too_large = np.zeros(n_ell, dtype=np.float32)
        if "too_large" in requested:
            too_large = self._primitive_protrusion_scores(
                centers, radii, rotations, eps, bend)

        too_small = np.zeros(n_ell, dtype=np.float32)
        if "too_small" in requested and regions and len(centers) > 0:
            cen = np.asarray(centers, dtype=np.float32)
            for reg in regions:
                seed = np.asarray(reg.get("seed_world"), dtype=np.float32)
                if seed.shape != (3,):
                    continue
                idx = int(np.argmin(np.linalg.norm(cen - seed[None, :], axis=1)))
                too_small[idx] = max(too_small[idx], float(reg.get("severity", 0.0)))

        redundant = np.zeros(n_ell, dtype=np.float32)
        coverage = np.zeros(n_ell, dtype=np.float32)
        unique = np.zeros(n_ell, dtype=np.float32)
        if requested & {"redundant", "coverage", "unique_coverage"}:
            try:
                cov = self._compute_coverage_info(
                    centers, radii, rotations, eps, bend)
                if cov.get("valid"):
                    sample_n = max(1, int(cov["is_inside"].shape[1]))
                    total = cov["total_coverage"].astype(np.float32)
                    uniq = cov["unique_coverage"].astype(np.float32)
                    coverage = (total / sample_n).astype(np.float32)
                    unique = (uniq / sample_n).astype(np.float32)
                    redundant = (1.0 - (uniq / np.maximum(total, 1.0))).astype(np.float32)
                    redundant[total <= 0.0] = 1.0
            except Exception:
                pass

        bone_over_budget = np.zeros(n_ell, dtype=np.float32)
        if "bone_over_budget" in requested:
            bone_over_budget, _assign, _counts, _expected = (
                self._bone_over_budget_scores(centers))

        out = {
            "bridging": bridging,
            "too_large": too_large,
            "too_small": too_small,
            "redundant": redundant,
            "coverage": coverage,
            "unique_coverage": unique,
            "bone_over_budget": bone_over_budget,
        }
        return {k: out[k] for k in requested if k in out}

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

    def _split_primitive(self, c, r, q, eps=None, bend=None):
        """Create two complete child states in the configured shape family.

        A z-split of a bent SQ follows the parent centreline and aligns each
        child with its local tangent.  The child curvature is the quadratic
        centreline acceleration expressed in that tangent frame.  This is the
        second-order local model of the parent, rather than a copied bend value
        attached to ellipsoid-style child centres.
        """
        c = np.asarray(c, dtype=np.float32).reshape(3)
        r = np.abs(np.asarray(r, dtype=np.float32).reshape(3))
        q = np.asarray(q, dtype=np.float32).reshape(4)
        e = (self._init_eps(1)[0] if eps is None
             else np.asarray(eps, dtype=np.float32).reshape(2))
        b = (self._init_bend(1)[0] if bend is None
             else np.asarray(bend, dtype=np.float32).reshape(2))
        axis = int(np.argmax(r))
        half = 0.5 * float(r[axis])
        child_r = np.repeat(r[None, :], 2, axis=0)
        child_r[:, axis] = half
        child_e = np.repeat(e[None, :], 2, axis=0).astype(np.float32)

        if not (self._bent and axis == 2):
            matrix = _quat_to_rot_matrix(q).astype(np.float32)
            offset = half * matrix[:, axis]
            child_c = np.stack([c + offset, c - offset]).astype(np.float32)
            child_q = np.repeat(q[None, :], 2, axis=0).astype(np.float32)
            child_b = np.repeat(b[None, :], 2, axis=0).astype(np.float32)
            rz = np.maximum(child_r[:, 2:3], 1.0e-8)
            child_b = np.clip(
                child_b * rz, -self._bend_kappa_max,
                self._bend_kappa_max) / rz
            return child_c, child_r, child_q, child_e, child_b

        parent_matrix = _quat_to_rot_matrix(q).astype(np.float64)
        child_centers, child_rotations, child_bends = [], [], []
        acceleration = np.array([float(b[0]), float(b[1]), 0.0])
        for z0 in (half, -half):
            centerline = np.array([
                0.5 * float(b[0]) * z0 * z0,
                0.5 * float(b[1]) * z0 * z0,
                z0,
            ], dtype=np.float64)
            tangent_raw = np.array(
                [float(b[0]) * z0, float(b[1]) * z0, 1.0],
                dtype=np.float64)
            speed = max(float(np.linalg.norm(tangent_raw)), 1.0e-12)
            tangent = tangent_raw / speed
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            x_axis -= tangent * float(np.dot(x_axis, tangent))
            if np.linalg.norm(x_axis) < 1.0e-8:
                x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                x_axis -= tangent * float(np.dot(x_axis, tangent))
            x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
            y_axis = np.cross(tangent, x_axis)
            y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
            x_axis = np.cross(y_axis, tangent)
            local_frame = np.column_stack([x_axis, y_axis, tangent])
            world_frame = parent_matrix @ local_frame
            if np.linalg.det(world_frame) < 0.0:
                world_frame[:, 0] *= -1.0
            normal_acceleration = acceleration - tangent * float(
                np.dot(acceleration, tangent))
            curvature = np.array([
                float(np.dot(normal_acceleration, x_axis)) / (speed * speed),
                float(np.dot(normal_acceleration, y_axis)) / (speed * speed),
            ], dtype=np.float32)
            child_centers.append(c.astype(np.float64) + parent_matrix @ centerline)
            child_rotations.append(_rot_matrix_to_quat(world_frame))
            child_bends.append(curvature)
        child_bends = np.asarray(child_bends, dtype=np.float32)
        rz = np.maximum(child_r[:, 2:3], 1.0e-8)
        child_bends = np.clip(
            child_bends * rz, -self._bend_kappa_max,
            self._bend_kappa_max) / rz
        return (
            np.asarray(child_centers, dtype=np.float32), child_r,
            np.asarray(child_rotations, dtype=np.float32), child_e,
            child_bends.astype(np.float32),
        )

    def _split_targets_for_regions(
        self,
        centers,
        radii,
        rotations,
        regions,
        budget,
        exclude=None,
        n_protect=0,
        bone_assign: np.ndarray | None = None,
        bone_counts: np.ndarray | None = None,
        bone_caps: np.ndarray | None = None,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ):
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
        eps, bend = self._shape_state_np(len(centers), eps, bend)
        for region in regions:
            if len(targets) >= budget:
                break
            seed = np.asarray(region['seed_world'], dtype=np.float32)
            seed_sdf = self._primitive_sdf_np_batch(
                centers, radii, rotations, seed[None, :], eps, bend)[:, 0]
            for j in np.argsort(np.abs(seed_sdf), kind="stable"):
                j = int(j)
                if j < n_protect or j in used:
                    continue
                if not self._reserve_split_bone_capacity(
                        j, centers, radii, rotations,
                        bone_assign, bone_counts, bone_caps, eps, bend):
                    continue
                used.add(j)
                targets.append(j)
                break
        return targets

    def _densify_regions(self, centers, radii, rotations, regions, budget, exclude=None,
                         split_enabled=True, spawn_enabled=True,
                         bone_assign: np.ndarray | None = None,
                         bone_counts: np.ndarray | None = None,
                         bone_caps: np.ndarray | None = None,
                         eps: np.ndarray | None = None,
                         bend: np.ndarray | None = None):
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
        eps, bend = self._shape_state_np(len(cen), eps, bend)
        near_world = float(self._region_radius_vox) * float(self._dx)

        for region in regions:
            if len(split_targets) + len(spawn_regions) >= int(budget):
                break
            seed = np.asarray(region['seed_world'], dtype=np.float32)
            j = None
            surface_distance = None
            if len(cen) > 0:
                seed_sdf = self._primitive_sdf_np_batch(
                    centers, radii, rotations, seed[None, :], eps, bend)[:, 0]
                for k in np.argsort(np.abs(seed_sdf), kind="stable"):
                    k = int(k)
                    if k not in used:
                        j, surface_distance = k, abs(float(seed_sdf[k]))
                        break
            near = j is not None and surface_distance < near_world

            # Prefer split for a nearby ellipsoid, spawn for an isolated gap;
            # fall back to whichever mechanism is enabled.
            if near and split_enabled:
                if self._reserve_split_bone_capacity(
                        j, centers, radii, rotations,
                        bone_assign, bone_counts, bone_caps, eps, bend):
                    used.add(j)
                    split_targets.append(j)
                elif spawn_enabled:
                    sbi = None
                    if bone_counts is not None and bone_caps is not None:
                        assign = self._nearest_bone_indices_np(seed.reshape(1, 3))
                        if assign is not None:
                            sbi = int(assign[0])
                    if self._bone_has_add_capacity(sbi, bone_counts, bone_caps):
                        spawn_regions.append(region)
                        if (bone_counts is not None and sbi is not None
                                and 0 <= sbi < len(bone_counts)):
                            bone_counts[sbi] += 1
            elif spawn_enabled:
                bi = None
                if bone_counts is not None and bone_caps is not None:
                    assign = self._nearest_bone_indices_np(seed.reshape(1, 3))
                    if assign is not None:
                        bi = int(assign[0])
                if self._bone_has_add_capacity(bi, bone_counts, bone_caps):
                    spawn_regions.append(region)
                    if (bone_counts is not None and bi is not None
                            and 0 <= bi < len(bone_counts)):
                        bone_counts[bi] += 1
            elif split_enabled and j is not None:
                if self._reserve_split_bone_capacity(
                        j, centers, radii, rotations,
                        bone_assign, bone_counts, bone_caps, eps, bend):
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

    def _grid_points_from_flat(self, flat_idx: np.ndarray) -> np.ndarray:
        """World-space voxel centres for flat target-grid indices."""
        flat_idx = np.asarray(flat_idx, dtype=np.int64).reshape(-1)
        if flat_idx.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        iz, iy, ix = np.unravel_index(flat_idx, self._shape)
        return (self._origin.astype(np.float32)
                + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5)
                * float(self._dx)).astype(np.float32)

    def _critical_grid_points_for_indices(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        indices,
        max_points: int = 256,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local target-grid samples that a merge/fuse is not allowed to hurt.

        Candidate points are mesh interior + surface-band voxels inside the
        affected ellipsoid(s).  Ranking prefers thin features, near-surface
        voxels and points uniquely covered by the affected set.  This avoids the
        old failure mode where random/local-volume checks were dominated by thick
        body mass while fingers or other small features lost their owner.
        """
        idx = [int(i) for i in np.atleast_1d(indices)]
        if not idx:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))

        eps, bend = self._shape_state_np(len(centers), eps, bend)
        aabb_lo, aabb_hi = self._primitive_aabbs(
            centers[idx], radii[idx], rotations[idx], bend[idx])
        lo = np.min(aabb_lo, axis=0)
        hi = np.max(aabb_hi, axis=0)
        max_extent = float(np.max(aabb_hi - aabb_lo))
        pad = max(2.0 * float(self._dx), 0.125 * max_extent)
        lo -= pad
        hi += pad

        q0 = np.floor((lo - self._origin) / float(self._dx)).astype(np.int64)
        q1 = np.ceil((hi - self._origin) / float(self._dx)).astype(np.int64) + 1
        q0 = np.clip(q0, 0, np.array([self._nx - 1, self._ny - 1, self._nz - 1]))
        q1 = np.clip(q1, 1, np.array([self._nx, self._ny, self._nz]))
        if np.any(q1 <= q0):
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))

        span = np.maximum(q1 - q0, 1)
        total_cells = int(np.prod(span.astype(np.int64)))
        max_scan_cells = max(20000, int(max_points) * 64)
        if total_cells > max_scan_cells:
            # Large torso-sized candidates can span a huge chunk of the grid.
            # Probe a deterministic subset before any expensive E×N SDF matrix.
            seed = 2166136261
            for v in idx:
                seed = (seed ^ int(v)) * 16777619 & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            sample_n = int(max_scan_cells)
            gx = rng.integers(q0[0], q1[0], size=sample_n, dtype=np.int64)
            gy = rng.integers(q0[1], q1[1], size=sample_n, dtype=np.int64)
            gz = rng.integers(q0[2], q1[2], size=sample_n, dtype=np.int64)
            flat = (gz * (self._nx * self._ny) + gy * self._nx + gx).astype(np.int64)
            flat = np.unique(flat)
        else:
            xs = np.arange(q0[0], q1[0], dtype=np.int64)
            ys = np.arange(q0[1], q1[1], dtype=np.int64)
            zs = np.arange(q0[2], q1[2], dtype=np.int64)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            flat = (gz.ravel() * (self._nx * self._ny)
                    + gy.ravel() * self._nx + gx.ravel()).astype(np.int64)
        target_flat = self._sdf_target_np.ravel()
        band = max(float(self._surface_band_vox), 1.0) * float(self._dx)
        target_vals = target_flat[flat]
        mesh_mask = target_vals < band
        if not np.any(mesh_mask):
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))

        flat = flat[mesh_mask]
        target_vals = target_vals[mesh_mask].astype(np.float32)
        score = np.ones(len(flat), dtype=np.float32)
        score *= (1.0 + 3.0 * np.exp(
            -(target_vals * target_vals) / max((1.5 * float(self._dx)) ** 2, 1e-12)))

        thick = self._thickness_np
        if thick is not None:
            th = thick.ravel()[flat].astype(np.float32)
            valid = th > 0.0
            if np.any(valid):
                ref = float(np.median(th[valid]))
                if ref > 1e-12:
                    score *= np.clip(ref / np.maximum(th, float(self._dx)),
                                     0.25, 8.0).astype(np.float32)

        max_affect_eval = max(2048, int(max_points) * 8)
        if len(flat) > max_affect_eval:
            order = np.argsort(-score, kind="stable")[:max_affect_eval]
            flat = flat[order]
            target_vals = target_vals[order]
            score = score[order]

        pts = self._grid_points_from_flat(flat)
        aff_sdf = self._primitive_sdf_np_batch(
            centers[idx], radii[idx], rotations[idx], pts,
            eps[idx], bend[idx])
        affected_inside = np.any(aff_sdf < 0.0, axis=0)
        if not np.any(affected_inside):
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))

        pts = pts[affected_inside]
        target_vals = target_vals[affected_inside]
        flat = flat[affected_inside]
        score = score[affected_inside]

        max_unique_eval = max(1024, int(max_points) * 4)
        if len(pts) > max_unique_eval:
            order = np.argsort(-score, kind="stable")[:max_unique_eval]
            pts = pts[order]
            target_vals = target_vals[order]
            score = score[order]

        all_sdf = self._primitive_sdf_np_batch(
            centers, radii, rotations, pts, eps, bend)
        cover = all_sdf < 0.0
        cover_count = cover.sum(axis=0)
        affected_only = np.any(cover[idx], axis=0)
        unique_to_affected = affected_only & (cover_count == 1)
        score[unique_to_affected] *= 4.0

        if len(pts) > int(max_points):
            order = np.argsort(-score, kind="stable")[:int(max_points)]
            pts = pts[order]
            target_vals = target_vals[order]
        return pts.astype(np.float32), target_vals.astype(np.float32)

    def _union_sdf_np(
        self, centers, radii, rotations, pts, eps=None, bend=None,
    ) -> np.ndarray:
        if len(centers) == 0:
            return np.full(len(pts), 1e6, dtype=np.float32)
        return np.min(self._primitive_sdf_np_batch(
            centers, radii, rotations, pts, eps, bend), axis=0).astype(np.float32)

    def _critical_replacement_worse(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        remove_indices,
        add_centers: np.ndarray | None = None,
        add_radii: np.ndarray | None = None,
        add_rotations: np.ndarray | None = None,
        add_eps: np.ndarray | None = None,
        add_bend: np.ndarray | None = None,
        *,
        eps: np.ndarray | None = None,
        bend: np.ndarray | None = None,
        rel_eps: float = 0.03,
    ) -> bool:
        """True when a candidate merge/fuse worsens critical local samples."""
        remove = [int(i) for i in np.atleast_1d(remove_indices)]
        eps, bend = self._shape_state_np(len(centers), eps, bend)
        pts, target = self._critical_grid_points_for_indices(
            centers, radii, rotations, remove, eps=eps, bend=bend)
        if len(pts) == 0:
            return False

        before = self._union_sdf_np(
            centers, radii, rotations, pts, eps, bend)
        keep = np.ones(len(centers), dtype=bool)
        keep[remove] = False
        after_c = centers[keep]
        after_r = radii[keep]
        after_q = rotations[keep]
        after_e = eps[keep]
        after_b = bend[keep]
        if add_centers is not None and len(add_centers) > 0:
            after_c = np.vstack([after_c, np.asarray(add_centers, dtype=np.float32)])
            after_r = np.vstack([after_r, np.asarray(add_radii, dtype=np.float32)])
            after_q = np.vstack([after_q, np.asarray(add_rotations, dtype=np.float32)])
            candidate_e, candidate_b = self._shape_state_np(
                len(add_centers), add_eps, add_bend)
            after_e = np.vstack([after_e, candidate_e])
            after_b = np.vstack([after_b, candidate_b])
        after = self._union_sdf_np(
            after_c, after_r, after_q, pts, after_e, after_b)

        before_err = (before - target) ** 2
        after_err = (after - target) ** 2
        mean_before = float(np.mean(before_err))
        mean_after = float(np.mean(after_err))
        abs_floor = (0.05 * float(self._dx)) ** 2
        if (mean_after - mean_before) > max(rel_eps * mean_before, abs_floor):
            return True

        # A small mean can hide a punched hole in a thin feature.  Also guard the
        # upper tail of the error distribution.
        p_before = float(np.percentile(before_err, 95))
        p_after = float(np.percentile(after_err, 95))
        return (p_after - p_before) > max(0.10 * p_before, (0.10 * float(self._dx)) ** 2)

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

    def _sq_unit_variances(self, eps: np.ndarray) -> np.ndarray:
        """Deterministic unit-SQ coordinate variances for moment matching."""
        e = np.clip(np.asarray(eps, dtype=np.float32).reshape(2), 0.1, 2.0)
        key = (round(float(e[0]), 4), round(float(e[1]), 4))
        cached = self._sq_unit_moment_cache.get(key)
        if cached is None:
            points = _sq_interior_points(
                np.zeros(3, np.float32), np.ones(3, np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], np.float32), e,
                8192, np.zeros(2, np.float32), seed=0x51A7)
            cached = np.maximum(np.var(points.astype(np.float64), axis=0), 1.0e-8)
            self._sq_unit_moment_cache[key] = cached
        return np.asarray(cached, dtype=np.float64)

    def _merge_two_primitives(
        self, i, j, centers, radii, rotations, eps, bend,
    ):
        """Moment-matched candidate in the same family as the source pair.

        SQ mass properties are measured from deterministic volume-uniform
        samples of the real bent shapes.  The candidate local z-axis follows
        the pair's source z-axes, ε is volume-weighted, and curvature is first
        combined in world space before being projected into the new frame.
        """
        if not self._superquadric:
            c, r, q = self._merge_two_ellipsoids(
                i, j, centers, radii, rotations)
            e = ((np.asarray(eps[i]) + np.asarray(eps[j])) * 0.5).astype(np.float32)
            b = np.zeros(2, dtype=np.float32)
            return c, r, q, e, b

        volumes = self._primitive_volume_proxies(
            radii[[i, j]], np.asarray(eps)[[i, j]])
        vi, vj = float(volumes[0]), float(volumes[1])
        total_volume = max(vi + vj, 1.0e-20)
        e_m = ((vi * eps[i] + vj * eps[j]) / total_volume).astype(np.float32)

        sample_count = max(512, int(self._fuse_samples) * 4)
        points_i = self._primitive_interior_points(
            centers[i], radii[i], rotations[i], sample_count,
            eps[i], bend[i], seed=0xA110 + int(i))
        points_j = self._primitive_interior_points(
            centers[j], radii[j], rotations[j], sample_count,
            eps[j], bend[j], seed=0xB220 + int(j))
        points = np.vstack([points_i, points_j]).astype(np.float64)
        weights = np.concatenate([
            np.full(sample_count, vi / sample_count, dtype=np.float64),
            np.full(sample_count, vj / sample_count, dtype=np.float64),
        ])
        weight_sum = float(np.sum(weights))
        centroid = np.sum(points * weights[:, None], axis=0) / weight_sum
        delta = points - centroid[None, :]
        covariance = (delta * weights[:, None]).T @ delta / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1.0e-12)

        Ri = _quat_to_rot_matrix(rotations[i]).astype(np.float64)
        Rj = _quat_to_rot_matrix(rotations[j]).astype(np.float64)
        z_tensor = (vi * np.outer(Ri[:, 2], Ri[:, 2])
                    + vj * np.outer(Rj[:, 2], Rj[:, 2])) / total_volume
        z_scores = np.einsum("ik,ij,jk->k", eigenvectors, z_tensor, eigenvectors)
        z_index = int(np.argmax(z_scores))
        remaining = [axis for axis in np.argsort(eigenvalues) if axis != z_index]
        matrix = np.column_stack([
            eigenvectors[:, remaining[0]],
            eigenvectors[:, remaining[1]],
            eigenvectors[:, z_index],
        ])
        if np.linalg.det(matrix) < 0.0:
            matrix[:, 0] *= -1.0
        q_m = _rot_matrix_to_quat(matrix)

        curvature_world = (
            vi * (Ri[:, :2] @ np.asarray(bend[i], dtype=np.float64))
            + vj * (Rj[:, :2] @ np.asarray(bend[j], dtype=np.float64))
        ) / total_volume
        b_m = (matrix[:, :2].T @ curvature_world).astype(np.float32)

        local_about_centroid = (points - centroid[None, :]) @ matrix
        mean_z2 = float(np.sum(
            weights * local_about_centroid[:, 2] ** 2) / weight_sum)
        mean_bend_shift = np.array(
            [0.5 * float(b_m[0]) * mean_z2,
             0.5 * float(b_m[1]) * mean_z2, 0.0], dtype=np.float64)
        c_m = centroid - matrix @ mean_bend_shift

        local = (points - c_m[None, :]) @ matrix
        z2 = local[:, 2] ** 2
        local[:, 0] -= 0.5 * float(b_m[0]) * z2
        local[:, 1] -= 0.5 * float(b_m[1]) * z2
        local_mean = np.sum(local * weights[:, None], axis=0) / weight_sum
        centered = local - local_mean[None, :]
        local_variance = np.sum(
            centered * centered * weights[:, None], axis=0) / weight_sum
        unit_variance = self._sq_unit_variances(e_m)
        r_m = np.sqrt(np.maximum(local_variance / unit_variance, 1.0e-12))
        return (c_m.astype(np.float32), r_m.astype(np.float32),
                q_m.astype(np.float32), e_m, b_m)

    def _merge_changes_surface(
        self, i, j, c_m, r_m, q_m, e_m, b_m,
        centers, radii, rotations, eps, bend,
    ) -> bool:
        """Two-sided boundary check using the exact source/candidate family."""
        directions = self._unit_sphere_samples()
        si = self._primitive_surface_points(
            centers[i], radii[i], rotations[i], directions, eps[i], bend[i])
        sj = self._primitive_surface_points(
            centers[j], radii[j], rotations[j], directions, eps[j], bend[j])
        sdf_i_at_j = self._primitive_sdf_np(
            centers[i], radii[i], rotations[i], sj, eps[i], bend[i])
        sdf_j_at_i = self._primitive_sdf_np(
            centers[j], radii[j], rotations[j], si, eps[j], bend[j])

        if np.all(sdf_i_at_j > 0.0) and np.all(sdf_j_at_i > 0.0):
            return True
        boundary = np.vstack([
            si[sdf_j_at_i > 0.0],
            sj[sdf_i_at_j > 0.0],
        ])
        if len(boundary) == 0:
            return True

        characteristic = max(
            float(np.mean(np.abs(radii[i]))),
            float(np.mean(np.abs(radii[j]))),
            float(np.mean(np.abs(r_m))),
            float(self._dx),
        )
        candidate_at_boundary = self._primitive_sdf_np(
            c_m, r_m, q_m, boundary, e_m, b_m)
        err_source_to_candidate = float(
            np.max(np.abs(candidate_at_boundary)) / characteristic)

        sm = self._primitive_surface_points(
            c_m, r_m, q_m, directions, e_m, b_m)
        source_at_candidate = self._primitive_sdf_np_batch(
            centers[[i, j]], radii[[i, j]], rotations[[i, j]], sm,
            eps[[i, j]], bend[[i, j]])
        err_candidate_to_source = float(
            np.max(np.abs(np.min(source_at_candidate, axis=0))) / characteristic)
        return max(err_source_to_candidate, err_candidate_to_source) > float(
            self._merge_tol)

    def _merge_increases_loss(self, i, j, c_m, r_m, q_m, e_m, b_m,
                              centers, radii, rotations, eps, bend,
                              n_samples: int = 1500, rel_eps: float = 0.10) -> bool:
        """True if replacing a pair raises the production-shaped local loss.

        Compares the union SDF against the target SDF, before vs after the merge,
        over random points in the pair's bounding box.  ``others`` (the union of
        all *other* ellipsoids) is unchanged by the merge, so only ``i, j`` vs the
        merged primitive affect the loss.  Only candidates that already passed the
        cheap geometric gates reach here, so this runs rarely.
        """
        origin = self._origin.astype(np.float64)
        dx = float(self._dx)
        nx, ny, nz = self._nx, self._ny, self._nz

        pair_lo, pair_hi = self._primitive_aabbs(
            centers[[i, j]], radii[[i, j]], rotations[[i, j]], bend[[i, j]])
        merged_lo, merged_hi = self._primitive_aabbs(
            np.asarray(c_m).reshape(1, 3), np.asarray(r_m).reshape(1, 3),
            np.asarray(q_m).reshape(1, 4), np.asarray(b_m).reshape(1, 2))
        lo = np.min(np.vstack([pair_lo, merged_lo]), axis=0).astype(np.float64)
        hi = np.max(np.vstack([pair_hi, merged_hi]), axis=0).astype(np.float64)

        rng = np.random.default_rng(int(i) * 131071 + int(j))   # per-pair, stable
        pts = (lo[None, :] + rng.random((n_samples, 3))
               * (hi - lo)[None, :]).astype(np.float32)

        q = (pts.astype(np.float64) - origin) / dx
        ijk = np.clip(np.floor(q).astype(np.int64), 0,
                      np.array([nx - 1, ny - 1, nz - 1]))
        target = self._sdf_target_np[ijk[:, 2], ijk[:, 1], ijk[:, 0]].astype(np.float32)

        keep = np.ones(len(centers), dtype=bool)
        keep[[i, j]] = False
        if np.any(keep):
            others = np.min(self._primitive_sdf_np_batch(
                centers[keep], radii[keep], rotations[keep], pts,
                eps[keep], bend[keep]), axis=0)
        else:
            others = np.full(n_samples, 1e6, dtype=np.float32)

        pair_sdf = self._primitive_sdf_np_batch(
            centers[[i, j]], radii[[i, j]], rotations[[i, j]], pts,
            eps[[i, j]], bend[[i, j]])
        sdf_i, sdf_j = pair_sdf[0], pair_sdf[1]
        sdf_m = self._primitive_sdf_np(c_m, r_m, q_m, pts, e_m, b_m)

        before = np.minimum(others, np.minimum(sdf_i, sdf_j))
        after = np.minimum(others, sdf_m)
        def _local_loss(prediction: np.ndarray) -> float:
            pred = np.clip(prediction.astype(np.float64), -10.0, 10.0)
            tgt = target.astype(np.float64)
            weight = 1.0 + float(self._surface_weight) * np.exp(
                -(tgt * tgt) / max(float(self._surface_sigma) ** 2, 1.0e-12))
            thickness = self._sample_global_thickness_points(pts).astype(np.float64)
            if self._thin_weight_eff > 0.0:
                valid = thickness > 0.0
                thin_factor = np.ones_like(thickness)
                thin_factor[valid] = np.minimum(
                    1.0 + float(self._thin_weight_eff) * np.maximum(
                        float(self._thick_ref) / thickness[valid] - 1.0, 0.0),
                    float(self._thin_max_factor))
                weight *= thin_factor
            diff = np.abs(
                _soft_clamp_np(pred, 0.1) - _soft_clamp_np(tgt, 0.1))
            delta = max(float(self._loss_huber_delta), 1.0e-8)
            base = np.where(
                diff < delta, 0.5 * diff * diff / delta,
                diff - 0.5 * delta)
            values = weight * base
            miss = (tgt < 0.0) & (pred > 0.0)
            values[miss] += (weight[miss] * float(self._miss_penalty_weight)
                             * (pred[miss] - tgt[miss]))
            outside = (tgt > 0.0) & (pred < 0.0)
            over = tgt[outside] - pred[outside]
            values[outside] += (
                weight[outside] * float(self._outside_penalty_weight)
                * over * over / max(float(self._surface_sigma), 1.0e-6))
            return float(np.mean(values))

        loss_before = _local_loss(before)
        loss_after = _local_loss(after)
        # Reject if the loss grows by more than a relative margin AND a small
        # absolute floor (so a negligible absolute rise on an already-good fit
        # doesn't block a genuine redundancy merge).
        abs_floor = (0.1 * dx) ** 2
        return (loss_after - loss_before) > max(rel_eps * loss_before, abs_floor)

    def _detect_merges(self, centers, radii, rotations, eps=None, bend=None):
        """Merge overlapping ellipsoid pairs whose fusion barely moves the surface.

        Greedy, most-overlapping pair first; each ellipsoid is used at most once
        per round and at most ``self._merge_per_round`` merges are applied.
        Returns ``(centers, radii, rotations, eps, bend, n_merged)`` — possibly
        unchanged.  Shape parameters of an accepted pair are volume-weighted,
        matching the mass weighting used for the merged geometry.
        """
        n = len(centers)
        self._last_merge_lineage = np.arange(n, dtype=np.int64)
        eps, bend = self._shape_state_np(n, eps, bend)
        if not self._merge_enabled or n <= 1 or self._merge_per_round <= 0:
            return centers, radii, rotations, eps, bend, 0

        # Conservative broad phase over the real primitive bounds.  Bent
        # displacement and boxy SQ corners are included in these AABBs.
        aabb_lo, aabb_hi = self._primitive_aabbs(
            centers, radii, rotations, bend)
        volumes = self._primitive_volume_proxies(radii, eps)
        bone_pressure, bone_assign, _counts, _expected = (
            self._bone_over_budget_scores(centers))
        max_pressure = float(np.max(bone_pressure)) if len(bone_pressure) else 0.0
        merge_cap = int(self._merge_per_round)
        if max_pressure > 0.0:
            merge_cap += int(min(3, np.ceil(max_pressure)))

        cand = []
        for i in range(n):
            for j in range(i + 1, n):
                overlap_extent = np.minimum(aabb_hi[i], aabb_hi[j]) - np.maximum(
                    aabb_lo[i], aabb_lo[j])
                if np.any(overlap_extent <= 0.0):
                    continue
                overlap_volume = float(np.prod(overlap_extent.astype(np.float64)))
                overlap_fraction = overlap_volume / max(
                    min(float(volumes[i]), float(volumes[j])), 1.0e-20)
                pair_pressure = 0.0
                if (bone_assign is not None
                        and int(bone_assign[i]) == int(bone_assign[j])):
                    pair_pressure = max(
                        float(bone_pressure[i]), float(bone_pressure[j]))
                cand.append((-(overlap_fraction + pair_pressure), i, j))
        if not cand:
            return centers, radii, rotations, eps, bend, 0
        cand.sort()                            # most overlapping (most negative) first

        consumed: set[int] = set()
        merged_c, merged_r, merged_q, merged_e, merged_b = [], [], [], [], []
        n_merged = 0
        for _, i, j in cand:
            if n_merged >= merge_cap:
                break
            if i in consumed or j in consumed:
                continue
            c_m, r_m, q_m, e_m, b_m = self._merge_two_primitives(
                i, j, centers, radii, rotations, eps, bend)
            e_m = np.clip(np.asarray(e_m, np.float32), 0.1, 2.0)
            rz_m = max(abs(float(r_m[2])), 1.0e-8)
            b_m = (np.clip(
                np.asarray(b_m, np.float32) * rz_m,
                -self._bend_kappa_max, self._bend_kappa_max) / rz_m
            ).astype(np.float32)
            if not np.isfinite(np.concatenate([
                    c_m, r_m, q_m, e_m, b_m])).all():
                continue
            merged_volume = float(self._primitive_volume_proxies(
                r_m[None, :], e_m[None, :])[0])
            if merged_volume > 1.75 * (float(volumes[i]) + float(volumes[j])):
                continue

            surface_changed = self._merge_changes_surface(
                i, j, c_m, r_m, q_m, e_m, b_m,
                centers, radii, rotations, eps, bend)
            if self._merge_increases_loss(
                    i, j, c_m, r_m, q_m, e_m, b_m,
                    centers, radii, rotations, eps, bend,
                    rel_eps=0.0 if surface_changed else 0.10):
                continue
            if self._critical_replacement_worse(
                    centers, radii, rotations, [i, j],
                    c_m[None, :], r_m[None, :], q_m[None, :],
                    e_m[None, :], b_m[None, :], eps=eps, bend=bend):
                continue
            consumed.add(i)
            consumed.add(j)
            merged_c.append(c_m)
            merged_r.append(r_m)
            merged_q.append(q_m)
            merged_e.append(e_m)
            merged_b.append(b_m)
            n_merged += 1

        if n_merged == 0:
            return centers, radii, rotations, eps, bend, 0

        keep = [k for k in range(n) if k not in consumed]
        # Merged rows are new primitives; survivors retain their exact source
        # row.  The global optimiser uses this map to transfer Adam moments.
        self._last_merge_lineage = np.concatenate([
            np.asarray(keep, dtype=np.int64),
            np.full(n_merged, -1, dtype=np.int64),
        ])
        out_c = np.vstack([centers[keep], np.asarray(merged_c, dtype=np.float32)])
        out_r = np.vstack([radii[keep], np.asarray(merged_r, dtype=np.float32)])
        out_q = np.vstack([rotations[keep], np.asarray(merged_q, dtype=np.float32)])
        out_e = np.vstack([eps[keep], np.asarray(merged_e, dtype=np.float32)])
        out_b = np.vstack([bend[keep], np.asarray(merged_b, dtype=np.float32)])
        return (out_c.astype(np.float32), out_r.astype(np.float32),
                out_q.astype(np.float32), out_e.astype(np.float32),
                out_b.astype(np.float32), n_merged)

    def _detect_redundant_ellipsoids(
        self, centers, radii, rotations, k_max, eps=None, bend=None,
    ) -> np.ndarray:
        """Up to ``k_max`` primitives whose real volume is covered by others.

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

        eps, bend = self._shape_state_np(n_ell, eps, bend)
        sample_count = max(64, int(self._fuse_samples))
        vols = self._primitive_volume_proxies(radii, eps)
        bone_pressure, _assign, _counts, _expected = (
            self._bone_over_budget_scores(centers))
        # Over-budget bones get first pass; within that, keep the old
        # smallest-volume-first behaviour.
        order = np.lexsort((vols, -bone_pressure))

        alive = np.ones(n_ell, dtype=bool)
        removed = []
        base_thr = float(self._fuse_overlap_frac)
        for i in order:
            if len(removed) >= int(k_max):
                break
            pts = self._primitive_interior_points(
                centers[i], radii[i], rotations[i], sample_count,
                eps[i], bend[i], seed=0xF053 + int(i))
            others = np.flatnonzero(alive & (np.arange(n_ell) != int(i)))
            if others.size == 0:
                continue
            other_sdf = self._primitive_sdf_np_batch(
                centers[others], radii[others], rotations[others], pts,
                eps[others], bend[others])
            covered = np.any(other_sdf < 0.0, axis=0)
            pressure = float(np.clip(bone_pressure[int(i)], 0.0, 2.0))
            thr = max(0.60, base_thr - 0.15 * pressure)
            if float(covered.mean()) >= thr:
                if self._critical_replacement_worse(
                        centers[alive], radii[alive], rotations[alive],
                        np.where(np.flatnonzero(alive) == i)[0],
                        eps=eps[alive], bend=bend[alive]):
                    continue
                alive[i] = False
                removed.append(int(i))

        return np.asarray(removed, dtype=int)

    def _detect_outside_ellipsoids(
        self, centers, radii, rotations, eps=None, bend=None,
    ) -> np.ndarray:
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

        eps, bend = self._shape_state_np(n_ell, eps, bend)
        sample_count = max(64, int(self._fuse_samples))
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
            if self._primitive_bound_radius(r_i, bend[i]) > max_radius:
                outside.append(int(i))
                continue
            # (b) centre clearly outside the mesh
            cval = float(_grid_val(centers[i].astype(np.float32)[None, :])[0])
            if cval > center_margin:
                outside.append(int(i))
                continue
            # (a) entirely outside (interior probe cloud)
            pts = self._primitive_interior_points(
                centers[i], r_i, rotations[i], sample_count,
                eps[i], bend[i], seed=0x0751 + i)
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

    def _local_fit(self, centers, radii, rotations, offset, pool_flat, gstep=-1,
                   eps=None, bend=None):
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
        eps, bend = self._shape_state_np(num_e, eps, bend)
        n_active = num_e - offset
        if n_active <= 0 or pool_flat.size == 0:
            return centers, radii, rotations, eps, bend

        bs = int(min(self._batch_size, 4096))
        buf = self._alloc_buffers(
            num_e, bs, total, centers, radii, rotations,
            eps_np=eps, bend_np=bend)
        pred_centers  = buf['pred_centers']
        pred_radii    = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']
        pred_eps_raw  = buf['pred_eps_raw']
        pred_eps      = buf['pred_eps']
        pred_bend_raw = buf['pred_bend_raw']
        pred_bend     = buf['pred_bend']
        min_d_cache   = buf['min_d_cache']
        sdf_pred      = buf['sdf_pred']
        loss          = buf['loss']
        sdf_target    = buf['sdf_target']
        wp_indices    = buf['wp_indices']

        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        anchor_centers_np = centers[offset:].astype(np.float32).copy()
        anchor_radii_np = np.maximum(
            np.abs(radii[offset:]).astype(np.float32), 1.0e-8)
        anchor_scales_np = np.mean(anchor_radii_np, axis=1).astype(np.float32)
        anchor_centers = wp.array(
            anchor_centers_np, dtype=wp.vec3, device=device)
        anchor_radii = wp.array(
            anchor_radii_np, dtype=wp.vec3, device=device)
        anchor_scales = wp.array(
            anchor_scales_np, dtype=wp.float32, device=device)
        prev_centers = wp.empty(num_e, dtype=wp.vec3, device=device)
        scheduled_lr = (
            float(self._lr_at(gstep)) if gstep >= 0 else float(self._local_lr))
        lr_peak = float(self._local_lr)
        if np.isfinite(scheduled_lr) and scheduled_lr > 0.0:
            lr_peak = min(lr_peak, scheduled_lr)
        lr_peak = min(lr_peak, 0.5 * float(dx))
        radius_factor = float(np.exp(self._local_log_radius_limit))
        rot_offset = offset * 4
        shape_offset = offset * 2
        report_every = max(1, self._local_steps // 20)

        for li in range(self._local_steps):
            if self._stop_flag:
                break
            progress = float(li) / float(max(self._local_steps - 1, 1))
            anneal = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
            lr = float(lr_peak * anneal)
            batch = self._rng.choice(pool_flat, size=bs, replace=True).astype(np.int32)
            wp_indices.assign(np.ascontiguousarray(batch))

            tape = wp.Tape()
            with tape:
                self._decode_shape_parameters(
                    pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                    pred_radii, num_e)
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
                                pred_eps, pred_bend, min_d_cache,
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
                            float(self._thin_weight_eff), float(self._thin_max_factor),
                            max(0.5 * float(dx), 1.0e-8)],
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

            if self._center_step_radius_frac > 0.0:
                wp.launch(_copy_vec3_range, dim=n_active,
                          inputs=[pred_centers, prev_centers, offset],
                          device=device)
            wp.launch(_sgd_step_vec3_range, dim=n_active,
                      inputs=[pred_centers, tape.gradients[pred_centers], lr, offset],
                      device=device)
            wp.launch(_sgd_step_vec3_range, dim=n_active,
                      inputs=[pred_radii, tape.gradients[pred_radii], lr, offset],
                      device=device)
            wp.launch(_sgd_step_f32_range, dim=n_active * 4,
                      inputs=[pred_rot_flat, tape.gradients[pred_rot_flat], lr, rot_offset],
                      device=device)
            shape_step = int(gstep if gstep >= 0 else self._num_steps)
            shape_base_lr = lr
            if self._eps_is_locally_trainable(shape_step):
                wp.launch(
                    _sgd_step_f32_range, dim=n_active * 2,
                    inputs=[pred_eps_raw, tape.gradients[pred_eps_raw],
                            float(shape_base_lr * self._sq_eps_lr_mult),
                            shape_offset],
                    device=device)
            if self._bend_is_trainable(shape_step):
                wp.launch(
                    _sgd_step_f32_range, dim=n_active * 2,
                    inputs=[pred_bend_raw, tape.gradients[pred_bend_raw],
                            float(shape_base_lr * self._sq_bend_lr_mult),
                            shape_offset],
                    device=device)
            if self._center_step_radius_frac > 0.0:
                wp.launch(
                    _limit_center_step_by_radius,
                    dim=n_active,
                    inputs=[
                        pred_centers, prev_centers, pred_radii,
                        float(self._center_step_radius_frac),
                        float(self._center_step_min_vox) * float(dx),
                        float(self._center_step_max_vox) * float(dx),
                        offset,
                    ],
                    device=device,
                )
            wp.launch(
                _project_local_linear_trust_region,
                dim=n_active,
                inputs=[
                    pred_centers, pred_radii,
                    anchor_centers, anchor_radii, anchor_scales,
                    float(self._local_center_trust_radius_factor),
                    radius_factor, offset,
                ],
                device=device,
            )
            wp.launch(_normalize_flat_quats_range, dim=n_active,
                      inputs=[pred_rot_flat, offset], device=device)

            tape.zero()

            if li % report_every == 0 or li == self._local_steps - 1:
                self._decode_shape_parameters(
                    pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                    pred_radii, num_e)
                wp.synchronize_device(device)
                c = pred_centers.numpy().copy()
                r = pred_radii.numpy().copy()
                q = pred_rot_flat.numpy().reshape(-1, 4).copy()
                e = pred_eps.numpy().reshape(-1, 2).copy()
                b = pred_bend.numpy().reshape(-1, 2).copy()
                extra = (np.concatenate([e, b], axis=1)
                         if self._bent else e) if self._superquadric else None
                self.local_progress.emit(li + 1, self._local_steps)
                self.step_visual.emit(
                    int(gstep), float(loss.numpy()[0]), c, r, q, extra)
                self._emit_live_metric_if_needed(
                    int(gstep), c, r, q, e, b)

        self._decode_shape_parameters(
            pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
            pred_radii, num_e)
        wp.synchronize_device(device)
        c = pred_centers.numpy().copy()
        r = pred_radii.numpy().copy()
        q = pred_rot_flat.numpy().reshape(-1, 4).copy()
        e = pred_eps.numpy().reshape(-1, 2).copy()
        b = pred_bend.numpy().reshape(-1, 2).copy()
        suffix_bad = (
            ~np.isfinite(c[offset:]).all(axis=1)
            | ~np.isfinite(r[offset:]).all(axis=1)
            | ~np.isfinite(q[offset:]).all(axis=1)
            | ~np.isfinite(e[offset:]).all(axis=1)
            | ~np.isfinite(b[offset:]).all(axis=1)
            | np.any(r[offset:] <= 0.0, axis=1)
        )
        if np.any(suffix_bad):
            bad = np.flatnonzero(suffix_bad)
            c[offset + bad] = centers[offset + bad]
            r[offset + bad] = radii[offset + bad]
            q[offset + bad] = rotations[offset + bad]
            e[offset + bad] = eps[offset + bad]
            b[offset + bad] = bend[offset + bad]
        return c, r, q, e, b

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
        # A dilated target may move outside the original local-fit box.  Keep
        # the complete requested distance plus the existing two-voxel guard.
        pad = 2.0 * float(self._dx) + abs(float(self._sdf_blowup_offset))
        return (c0 - half - pad).astype(np.float32), (c0 + half + pad).astype(np.float32)

    def _apply_blowup_to_region_result(self, res) -> None:
        """Apply the whole-mesh thickness cap to one fresh local SDF box."""
        requested = float(self._sdf_blowup_offset)
        if requested == 0.0:
            return

        source = self._sdf_blowup_thickness_np
        if source is None:
            # Missing thickness must fail closed around the movable surface,
            # never silently turn a protected local fit back into uniform blowup.
            local_thickness = np.zeros_like(res.grid, dtype=np.float32)
        else:
            nz, ny, nx = (int(v) for v in np.asarray(res.grid).shape)
            local_thickness = np.empty((nz, ny, nx), dtype=np.float32)
            plane = ny * nx
            chunk = 262_144
            flat_out = local_thickness.ravel()
            for start in range(0, flat_out.size, chunk):
                stop = min(start + chunk, flat_out.size)
                flat = np.arange(start, stop, dtype=np.int64)
                z = flat // plane
                rem = flat - z * plane
                y = rem // nx
                x = rem - y * nx
                points = np.empty((len(flat), 3), dtype=np.float32)
                points[:, 0] = (
                    float(res.origin[0])
                    + (x.astype(np.float32) + 0.5) * float(res.dx)
                )
                points[:, 1] = (
                    float(res.origin[1])
                    + (y.astype(np.float32) + 0.5) * float(res.dx)
                )
                points[:, 2] = (
                    float(res.origin[2])
                    + (z.astype(np.float32) + 0.5) * float(res.dx)
                )
                flat_out[start:stop] = _sample_voxel_field_trilinear(
                    source,
                    self._sdf_blowup_origin,
                    self._sdf_blowup_dx,
                    points,
                )

        res.blowup_thickness = np.ascontiguousarray(
            local_thickness, dtype=np.float32)
        # Region maintenance/loss weighting must cover the moved exterior too.
        res.thickness = res.blowup_thickness
        res.grid = apply_thickness_limited_blowup(
            res.grid,
            requested,
            res.blowup_thickness,
            float(res.dx),
            max_thickness_fraction=(
                self._sdf_blowup_max_thickness_fraction),
        )

    def _region_divide_conquer(self, contrib_c, contrib_r, contrib_q,
                               train_c, train_r, train_q, res, n_fixed,
                               population_cap=None, contrib_eps=None,
                               contrib_bend=None, train_eps=None,
                               train_bend=None):
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
        return_shape_state = any(
            value is not None for value in
            (contrib_eps, contrib_bend, train_eps, train_bend))
        n_contrib = int(contrib_c.shape[0])
        contrib_eps, contrib_bend = self._shape_state_np(
            n_contrib, contrib_eps, contrib_bend)
        train_eps, train_bend = self._shape_state_np(
            len(train_c), train_eps, train_bend)

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
            e = np.concatenate([contrib_eps, train_eps], axis=0).astype(np.float32)
            b = np.concatenate([contrib_bend, train_bend], axis=0).astype(np.float32)
            lineage = np.concatenate([
                np.full(n_contrib, -1, dtype=np.int64),
                np.arange(len(train_c), dtype=np.int64),
            ])

            def _drop(idx):
                nonlocal c, r, q, e, b, lineage
                idx = np.asarray(idx, dtype=int)
                idx = idx[idx >= n_contrib]          # never touch the prefix
                if idx.size == 0:
                    return
                keep = np.ones(len(c), dtype=bool)
                keep[idx] = False
                c, r, q, e, b = c[keep], r[keep], q[keep], e[keep], b[keep]
                lineage = lineage[keep]

            # Delete trainables entirely outside the (box) mesh, then fuse
            # trainables whose interior is already covered.
            _drop(self._detect_outside_ellipsoids(c, r, q, e, b))
            if self._prune_enabled:
                _drop(self._detect_redundant_ellipsoids(
                    c, r, q, self._fuse_per_round, e, b))
            bone_assign, bone_counts, bone_caps = self._bone_growth_state(c)

            # Remaining global budget (net additions allowed across the whole pop).
            n_train = len(c) - n_contrib
            cap = (self._max_ellipsoids if population_cap is None
                   else int(population_cap))
            room = cap - (n_fixed + n_train)

            # Split oversized / bridging / protruding trainables (net +1 each).
            if room > 0:
                bridge = self._detect_bridging_ellipsoids(c, r, q, e, b)
                protr = self._detect_protruding_ellipsoids(c, r, q, e, b)
                seen, split_all = set(), []
                cap_local = int(min(self._split_per_round, room))
                for v in list(bridge) + list(protr):
                    v = int(v)
                    if len(split_all) >= cap_local:
                        break
                    if v >= n_contrib and v not in seen:
                        if not self._reserve_split_bone_capacity(
                                v, c, r, q, bone_assign, bone_counts, bone_caps,
                                e, b):
                            continue
                        seen.add(v)
                        split_all.append(v)
                n_split = int(len(split_all))
                if n_split > 0:
                    split_idx = np.sort(np.asarray(split_all[:n_split], dtype=int))
                    child_c, child_r, child_q, child_e, child_b = [], [], [], [], []
                    for i in split_idx:
                        cc, cr, cq, ce, cb = self._split_primitive(
                            c[i], r[i], q[i], e[i], b[i])
                        child_c.append(cc); child_r.append(cr); child_q.append(cq)
                        child_e.append(ce)
                        child_b.append(cb)
                    keep = np.ones(len(c), dtype=bool)
                    keep[split_idx] = False
                    parent_lineage = lineage[split_idx]
                    c = np.concatenate([c[keep]] + child_c, axis=0).astype(np.float32)
                    r = np.concatenate([r[keep]] + child_r, axis=0).astype(np.float32)
                    q = np.concatenate([q[keep]] + child_q, axis=0).astype(np.float32)
                    e = np.concatenate([e[keep]] + child_e, axis=0).astype(np.float32)
                    b = np.concatenate([b[keep]] + child_b, axis=0).astype(np.float32)
                    lineage = np.concatenate([
                        lineage[keep], np.repeat(parent_lineage, 2)
                    ]).astype(np.int64)
                    room -= n_split
                    bone_assign, bone_counts, bone_caps = self._bone_growth_state(c)

            # Under-represented box regions: SPLIT the nearest trainable
            # ellipsoid along its longest semi-axis (no random spawning).  The
            # frozen contributor prefix [:n_contrib] is never split.
            if room > 0:
                n_dens = int(min(self._spawn_per_round, room))
                regions = self._detect_worst_regions(
                    c, r, q, n_dens, eps=e, bend=b)
                dens_idx = self._split_targets_for_regions(
                    c, r, q, regions, room, exclude=set(), n_protect=n_contrib,
                    bone_assign=bone_assign,
                    bone_counts=bone_counts,
                    bone_caps=bone_caps,
                    eps=e, bend=b)
                if dens_idx:
                    split_idx = np.sort(np.asarray(dens_idx, dtype=int))
                    child_c, child_r, child_q, child_e, child_b = [], [], [], [], []
                    for i in split_idx:
                        cc, cr, cq, ce, cb = self._split_primitive(
                            c[i], r[i], q[i], e[i], b[i])
                        child_c.append(cc); child_r.append(cr); child_q.append(cq)
                        child_e.append(ce)
                        child_b.append(cb)
                    keep = np.ones(len(c), dtype=bool)
                    keep[split_idx] = False
                    parent_lineage = lineage[split_idx]
                    c = np.concatenate([c[keep]] + child_c, axis=0).astype(np.float32)
                    r = np.concatenate([r[keep]] + child_r, axis=0).astype(np.float32)
                    q = np.concatenate([q[keep]] + child_q, axis=0).astype(np.float32)
                    e = np.concatenate([e[keep]] + child_e, axis=0).astype(np.float32)
                    b = np.concatenate([b[keep]] + child_b, axis=0).astype(np.float32)
                    lineage = np.concatenate([
                        lineage[keep], np.repeat(parent_lineage, 2)
                    ]).astype(np.int64)
                    room -= len(split_idx)

            result = (c[n_contrib:].copy(), r[n_contrib:].copy(),
                      q[n_contrib:].copy(), e[n_contrib:].copy(),
                      b[n_contrib:].copy())
            self._last_region_dc_lineage = lineage[n_contrib:].copy()
            return result if return_shape_state else result[:3]
        finally:
            (self._sdf_target_np, self._thickness_np,
             self._origin, self._dx, self._n,
             self._nx, self._ny, self._nz, self._shape) = saved

    def _region_dc_all_boxes(self, fixed_c, fixed_r, fixed_q,
                             train_c, train_r, train_q, train_box, boxes,
                             population_cap=None, fixed_eps=None,
                             fixed_bend=None, train_eps=None,
                             train_bend=None):
        """Per-box divide-and-conquer for the combined local fit.

        Runs the existing ``_region_divide_conquer`` once per region box, scoped
        to that box's trainables (every other ellipsoid — frozen prefix + the
        other boxes' trainables — is passed as protected contributor so the
        detectors see the full union and the budget stays global).  Returns the
        reassembled, box-grouped trainable set plus a ``changed`` flag (False
        when no box added/removed/moved anything, so the caller can keep its Adam
        state instead of rebuilding).
        """
        return_shape_state = any(
            value is not None for value in
            (fixed_eps, fixed_bend, train_eps, train_bend))
        global_cap = (self._max_ellipsoids if population_cap is None
                      else int(population_cap))
        fixed_count = int(len(fixed_c))
        fixed_eps, fixed_bend = self._shape_state_np(
            fixed_count, fixed_eps, fixed_bend)
        train_eps, train_bend = self._shape_state_np(
            len(train_c), train_eps, train_bend)
        original_counts = np.array(
            [np.count_nonzero(train_box == b) for b in range(len(boxes))],
            dtype=np.int32,
        )
        processed_count = 0
        new_c, new_r, new_q, new_e, new_bend, new_box = [], [], [], [], [], []
        new_lineage: list[np.ndarray] = []
        changed = False
        for b in range(len(boxes)):
            sel = (train_box == b)
            if not np.any(sel):
                continue
            tc, tr, tq = train_c[sel], train_r[sel], train_q[sel]
            te, tb = train_eps[sel], train_bend[sel]
            other = ~sel
            oc = np.concatenate([fixed_c, train_c[other]], axis=0).astype(np.float32)
            orr = np.concatenate([fixed_r, train_r[other]], axis=0).astype(np.float32)
            oq = np.concatenate([fixed_q, train_q[other]], axis=0).astype(np.float32)
            oe = np.concatenate([fixed_eps, train_eps[other]], axis=0).astype(np.float32)
            ob = np.concatenate([fixed_bend, train_bend[other]], axis=0).astype(np.float32)
            unprocessed_count = int(original_counts[b + 1:].sum())
            max_box_count = max(
                0,
                global_cap - fixed_count - processed_count - unprocessed_count,
            )
            if return_shape_state:
                ntc, ntr, ntq, nte, ntb = self._region_divide_conquer(
                    oc, orr, oq, tc.copy(), tr.copy(), tq.copy(),
                    boxes[b]['res'], int(oc.shape[0]),
                    population_cap=int(oc.shape[0]) + max_box_count,
                    contrib_eps=oe, contrib_bend=ob,
                    train_eps=te.copy(), train_bend=tb.copy())
            else:
                # Retain the legacy geometry-only private contract for external
                # callers/tests; the optimizer itself always supplies shape state.
                ntc, ntr, ntq = self._region_divide_conquer(
                    oc, orr, oq, tc.copy(), tr.copy(), tq.copy(),
                    boxes[b]['res'], int(oc.shape[0]),
                    population_cap=int(oc.shape[0]) + max_box_count)
                nte, ntb = self._shape_state_np(len(ntc))
            local_map = np.asarray(
                getattr(self, "_last_region_dc_lineage",
                        np.arange(len(ntc), dtype=np.int64)),
                dtype=np.int64,
            )
            source_rows = np.flatnonzero(sel).astype(np.int64)
            mapped_lineage = np.full(len(local_map), -1, dtype=np.int64)
            valid_lineage = ((local_map >= 0) & (local_map < len(source_rows)))
            mapped_lineage[valid_lineage] = source_rows[local_map[valid_lineage]]
            if len(ntc) > max_box_count:
                ntc = ntc[:max_box_count]
                ntr = ntr[:max_box_count]
                ntq = ntq[:max_box_count]
                nte = nte[:max_box_count]
                ntb = ntb[:max_box_count]
                mapped_lineage = mapped_lineage[:max_box_count]
                changed = True
            if (ntc.shape[0] != tc.shape[0]
                    or not np.array_equal(ntc, tc)
                    or not np.array_equal(ntr, tr)
                    or not np.array_equal(ntq, tq)
                    or not np.array_equal(nte, te)
                    or not np.array_equal(ntb, tb)):
                changed = True
            new_c.append(ntc); new_r.append(ntr); new_q.append(ntq)
            new_e.append(nte); new_bend.append(ntb)
            new_box.append(np.full(ntc.shape[0], b, dtype=int))
            new_lineage.append(mapped_lineage)
            processed_count += int(ntc.shape[0])
        if not new_c:
            result = (train_c, train_r, train_q, train_eps, train_bend,
                      train_box, False)
            self._last_region_dc_all_lineage = np.arange(
                len(train_c), dtype=np.int64)
        else:
            result = (np.concatenate(new_c, axis=0).astype(np.float32),
                      np.concatenate(new_r, axis=0).astype(np.float32),
                      np.concatenate(new_q, axis=0).astype(np.float32),
                      np.concatenate(new_e, axis=0).astype(np.float32),
                      np.concatenate(new_bend, axis=0).astype(np.float32),
                      np.concatenate(new_box, axis=0), changed)
            self._last_region_dc_all_lineage = np.concatenate(
                new_lineage, axis=0).astype(np.int64)
        if return_shape_state:
            return result
        return result[0], result[1], result[2], result[5], result[6]

    def _local_fit_regions(self, centers, radii, rotations, region_sites,
                           box_results, gstep=-1, population_cap=None,
                           eps=None, bend=None, allow_region_dc=False):
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
        eps, bend = self._shape_state_np(len(centers), eps, bend)
        self._last_local_fit_lineage = np.arange(len(centers), dtype=np.int64)
        self._last_local_fit_rollback_rows = np.empty(0, dtype=np.int64)
        if (self._sdf_computer is None or not self._sdf_computer.is_ready
                or not region_sites or not box_results):
            return centers, radii, rotations, eps, bend

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

        # Local fit may only edit a primitive if its complete actual AABB,
        # including bend displacement, is inside the local problem box.
        primitive_lo, primitive_hi = self._primitive_aabbs(
            centers, radii, rotations, bend)
        assigned = np.full(len(centers), -1, dtype=int)
        for b, m in enumerate(boxes):
            bmin, bmax = m['box_min'], m['box_max']
            # Trainable for this box: the full ellipsoid must be inside the
            # local problem box.  Overlapping/nearby ellipsoids remain frozen
            # contributors so local fit cannot drag geometry across box borders.
            fits = (np.all(primitive_lo >= bmin, axis=1)
                    & np.all(primitive_hi <= bmax, axis=1))
            cand = fits & (assigned < 0)
            assigned[cand] = b
        train_idx = np.where(assigned >= 0)[0]
        if train_idx.size == 0:
            return centers, radii, rotations, eps, bend
        # Group trainables by box so a no-op D&C round returns an identical set
        # (lets the Adam state be reused across cycles).
        train_idx = train_idx[np.argsort(assigned[train_idx], kind="stable")]
        train_box = assigned[train_idx].copy()
        active = sorted({int(b) for b in train_box})

        # ── 2) Combined sample pool: world points + target + thickness ──
        band = float(self._surface_band_vox)
        pts_list, tgt_list, th_list = [], [], []
        any_thick = False
        # Ignore boxes that do not own a trainable primitive.  Their samples can
        # only pull a different box's primitive through the hard union minimum.
        for box_index in active:
            m = boxes[box_index]
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
                th = self._sample_global_thickness_points(pts_list[-1])
                th_list.append(th)
                any_thick = any_thick or bool(np.any(th > 0.0))
        if not pts_list:
            return centers, radii, rotations, eps, bend
        pool_points = np.concatenate(pts_list, axis=0).astype(np.float32)
        pool_targets = np.concatenate(tgt_list, axis=0).astype(np.float32)
        pool_thick = np.concatenate(th_list, axis=0).astype(np.float32)
        P = int(pool_points.shape[0])
        local_dx = float(min(boxes[b]['dx'] for b in active))
        local_surface_sigma = max(
            self._surface_sigma_vox * local_dx, 1.0e-8)

        interior_th = pool_thick[pool_thick > 0.0]
        thick_ref = float(np.median(interior_th)) if interior_th.size else 1.0
        thin_w = float(self._thin_loss_weight) if any_thick else 0.0

        # Fixed, stratified hold-out points make the final per-primitive
        # acceptance check deterministic rather than dependent on the last
        # stochastic mini-batch.
        validation_parts: list[np.ndarray] = []
        validation_budget = min(2048, P)
        surface_mask = np.abs(pool_targets) <= 2.0 * local_dx
        strata_masks = (
            surface_mask,
            pool_targets < -2.0 * local_dx,
            pool_targets > 2.0 * local_dx,
        )
        strata_quota = (
            validation_budget // 2,
            validation_budget // 4,
            validation_budget
            - validation_budget // 2
            - validation_budget // 4,
        )
        for mask, quota in zip(strata_masks, strata_quota):
            candidates = np.flatnonzero(mask)
            if candidates.size and quota > 0:
                take = min(int(quota), int(candidates.size))
                positions = np.linspace(
                    0, candidates.size - 1, take, dtype=np.int64)
                validation_parts.append(candidates[positions])
        validation_idx = (
            np.unique(np.concatenate(validation_parts))
            if validation_parts else np.empty(0, dtype=np.int64))
        if validation_idx.size < validation_budget:
            remaining = np.setdiff1d(
                np.arange(P, dtype=np.int64), validation_idx,
                assume_unique=False)
            take = min(validation_budget - validation_idx.size, remaining.size)
            if take:
                positions = np.linspace(
                    0, remaining.size - 1, take, dtype=np.int64)
                validation_idx = np.concatenate(
                    [validation_idx, remaining[positions]])
        validation_values = pool_targets[validation_idx]
        validation_strata = np.where(
            np.abs(validation_values) <= 2.0 * local_dx, 0,
            np.where(validation_values < 0.0, 1, 2),
        ).astype(np.uint8)
        validation_sample = ValidationSample(
            points=pool_points[validation_idx],
            values=validation_values,
            source_indices=validation_idx,
            strata=validation_strata,
            dx=local_dx,
            thickness=(pool_thick[validation_idx] if any_thick else None),
            thickness_reference=(thick_ref if any_thick else None),
        )

        # Pool is fixed for the whole fit → upload once.
        wp_points = wp.array(pool_points, dtype=wp.vec3, device=device)
        wp_targets = wp.array(pool_targets, dtype=wp.float32, device=device)
        wp_thick = wp.array(pool_thick, dtype=wp.float32, device=device)

        # Show every region box that actually holds a trainable — i.e. the small
        # high-res boxes currently being optimised, not one box over the whole
        # object.  (Boxes are fixed geometry, so one emit at the start suffices.)
        region_boxes = [(boxes[b]['box_min'].copy(), boxes[b]['box_max'].copy())
                        for b in active]
        self.region_changed.emit(region_boxes)

        # ── 3) frozen prefix + trainable suffix ──
        fixed_mask = np.ones(len(centers), dtype=bool)
        fixed_mask[train_idx] = False
        fixed_idx = np.where(fixed_mask)[0]
        fixed_c = centers[fixed_mask].astype(np.float32).copy()
        fixed_r = radii[fixed_mask].astype(np.float32).copy()
        fixed_q = rotations[fixed_mask].astype(np.float32).copy()
        fixed_e = eps[fixed_mask].astype(np.float32).copy()
        fixed_b = bend[fixed_mask].astype(np.float32).copy()
        n_fixed = int(fixed_c.shape[0])

        train_c = centers[train_idx].astype(np.float32).copy()
        train_r = radii[train_idx].astype(np.float32).copy()
        train_q = rotations[train_idx].astype(np.float32).copy()
        train_e = eps[train_idx].astype(np.float32).copy()
        train_b = bend[train_idx].astype(np.float32).copy()
        anchor_c = train_c.copy()
        anchor_r = train_r.copy()
        anchor_q = train_q.copy()
        anchor_e = train_e.copy()
        anchor_b = train_b.copy()
        local_lineage = np.concatenate([
            fixed_idx.astype(np.int64, copy=False),
            train_idx.astype(np.int64, copy=False),
        ])

        bs = int(min(self._batch_size, 4096))
        # Local LR remains user-controlled, but it may not exceed either the
        # current positive global schedule or half a high-resolution box voxel.
        # This prevents one Adam step from jumping several local voxels.
        lr0 = float(self._local_lr)
        scheduled_lr = (
            float(self._lr_at(gstep)) if gstep >= 0 else lr0)
        if np.isfinite(scheduled_lr) and scheduled_lr > 0.0:
            lr0 = min(lr0, scheduled_lr)
        lr0 = min(lr0, 0.5 * local_dx)
        dc_enabled = bool(
            allow_region_dc and len(active) == 1
            and self._region_dc_cycles > 1)
        n_cycles = min(
            self._region_dc_cycles if dc_enabled else 1,
            self._region_steps,
        )
        cycle_base, cycle_remainder = divmod(self._region_steps, n_cycles)
        cycle_steps = [
            cycle_base + int(cycle < cycle_remainder)
            for cycle in range(n_cycles)
        ]

        def _clamp_arrays(tbox, ac, ar, aq, ab):
            box_lo = np.stack(
                [boxes[b]['box_min'] for b in tbox]).astype(np.float32)
            box_hi = np.stack(
                [boxes[b]['box_max'] for b in tbox]).astype(np.float32)
            anchor_lo, anchor_hi = self._primitive_aabbs(ac, ar, aq, ab)
            radius_factor = float(np.exp(self._local_log_radius_limit))
            low_extent = np.maximum(ac - anchor_lo, 0.0) * radius_factor
            high_extent = np.maximum(anchor_hi - ac, 0.0) * radius_factor
            # Keep each start state feasible even when a very tight box leaves
            # no room for the full configured growth factor.
            lo = np.minimum(box_lo + low_extent, ac).astype(np.float32)
            hi = np.maximum(box_hi - high_extent, ac).astype(np.float32)
            logmin = np.array(
                [np.log(max(boxes[b]['dx'], 1e-9)) for b in tbox], dtype=np.float32)
            logmax = np.array(
                [np.log(max(0.5 * boxes[b]['extent'], 2.0 * boxes[b]['dx']))
                 for b in tbox], dtype=np.float32)
            min_steps = np.array(
                [self._center_step_min_vox * boxes[b]['dx'] for b in tbox],
                dtype=np.float32)
            max_steps = np.array(
                [self._center_step_max_vox * boxes[b]['dx'] for b in tbox],
                dtype=np.float32)
            return (wp.array(lo, dtype=wp.vec3, device=device),
                    wp.array(hi, dtype=wp.vec3, device=device),
                    wp.array(logmin, dtype=wp.float32, device=device),
                    wp.array(logmax, dtype=wp.float32, device=device),
                    wp.array(min_steps, dtype=wp.float32, device=device),
                    wp.array(max_steps, dtype=wp.float32, device=device))

        def _build_state():
            sub_c = np.concatenate([fixed_c, train_c], axis=0).astype(np.float32)
            sub_r = np.concatenate([fixed_r, train_r], axis=0).astype(np.float32)
            sub_q = np.concatenate([fixed_q, train_q], axis=0).astype(np.float32)
            sub_e = np.concatenate([fixed_e, train_e], axis=0).astype(np.float32)
            sub_b = np.concatenate([fixed_b, train_b], axis=0).astype(np.float32)
            num_e = n_fixed + int(train_c.shape[0])
            buf = self._alloc_buffers(
                num_e, bs, P, sub_c, sub_r, sub_q,
                sdf_target_np=pool_targets, eps_np=sub_e, bend_np=sub_b)
            log_r = wp.array(
                np.log(np.maximum(buf['pred_radii'].numpy(), 1e-6)),
                dtype=wp.vec3, device=device, requires_grad=True)
            anchor_log_r = np.log(np.maximum(anchor_r, 1.0e-8)).astype(np.float32)
            anchor_scales = np.mean(
                np.maximum(np.abs(anchor_r), 1.0e-8), axis=1).astype(np.float32)
            st = dict(
                num_e=num_e, offset=n_fixed,
                pred_centers=buf['pred_centers'], pred_radii=buf['pred_radii'],
                pred_rot_flat=buf['pred_rot_flat'], pred_log_radii=log_r,
                pred_eps_raw=buf['pred_eps_raw'], pred_eps=buf['pred_eps'],
                pred_bend_raw=buf['pred_bend_raw'], pred_bend=buf['pred_bend'],
                prev_centers=wp.empty(num_e, dtype=wp.vec3, device=device),
                min_d_cache=buf['min_d_cache'], sdf_pred=buf['sdf_pred'],
                loss=buf['loss'], wp_indices=buf['wp_indices'],
                opt_c=_PopulationAdam(buf['pred_centers'], lr0),
                opt_r=_PopulationAdam(
                    log_r, lr0 * self._lr_mult_radii),
                opt_q=_PopulationAdam(
                    buf['pred_rot_flat'], lr0 * self._lr_mult_rot),
                opt_eps=_PopulationAdam(buf['pred_eps_raw'], lr0),
                opt_bend=_PopulationAdam(buf['pred_bend_raw'], lr0),
                anchor_centers=wp.array(
                    anchor_c, dtype=wp.vec3, device=device),
                anchor_log_radii=wp.array(
                    anchor_log_r, dtype=wp.vec3, device=device),
                anchor_scales=wp.array(
                    anchor_scales, dtype=wp.float32, device=device),
            )
            st['grad_c'] = [st['pred_centers'].grad.flatten()]
            st['grad_r'] = [log_r.grad.flatten()]
            st['grad_q'] = [st['pred_rot_flat'].grad.flatten()]
            st['grad_eps'] = [st['pred_eps_raw'].grad.flatten()]
            st['grad_bend'] = [st['pred_bend_raw'].grad.flatten()]
            st['clamps'] = _clamp_arrays(
                train_box, anchor_c, anchor_r, anchor_q, anchor_b)
            return st

        def _repair_local_rows(tc, tr, tq, te, tb):
            """Return finite, trust-bounded rows whose full AABB fits its box."""
            tc = np.asarray(tc, np.float32).copy()
            tr = np.asarray(tr, np.float32).copy()
            tq = np.asarray(tq, np.float32).copy()
            te = np.asarray(te, np.float32).copy()
            tb = np.asarray(tb, np.float32).copy()
            repaired = np.zeros(len(tc), dtype=bool)

            def _restore(rows):
                rows = np.asarray(rows, dtype=np.int64)
                if rows.size == 0:
                    return
                tc[rows] = anchor_c[rows]
                tr[rows] = anchor_r[rows]
                tq[rows] = anchor_q[rows]
                te[rows] = anchor_e[rows]
                tb[rows] = anchor_b[rows]
                repaired[rows] = True

            qnorm = np.linalg.norm(tq, axis=1)
            invalid = (
                ~np.isfinite(tc).all(axis=1)
                | ~np.isfinite(tr).all(axis=1)
                | ~np.isfinite(tq).all(axis=1)
                | ~np.isfinite(te).all(axis=1)
                | ~np.isfinite(tb).all(axis=1)
                | np.any(tr <= 0.0, axis=1)
                | ~np.isfinite(qnorm)
                | (qnorm < 1.0e-8)
            )
            _restore(np.flatnonzero(invalid))

            qnorm = np.maximum(np.linalg.norm(tq, axis=1, keepdims=True), 1.0e-12)
            tq /= qnorm
            if self._local_center_trust_radius_factor > 0.0:
                scale = np.mean(
                    np.maximum(np.abs(anchor_r), 1.0e-8), axis=1)
                limit = self._local_center_trust_radius_factor * scale
                delta = tc - anchor_c
                dist = np.linalg.norm(delta, axis=1)
                move = dist > np.maximum(limit, 1.0e-8)
                if np.any(move):
                    tc[move] = (
                        anchor_c[move]
                        + delta[move]
                        * (limit[move] / np.maximum(dist[move], 1.0e-12))[:, None]
                    )
                    repaired[move] = True
            if self._local_log_radius_limit > 0.0:
                factor = float(np.exp(self._local_log_radius_limit))
                bounded = np.clip(
                    np.abs(tr), anchor_r / factor, anchor_r * factor)
                repaired |= np.any(bounded != tr, axis=1)
                tr = bounded.astype(np.float32)
            te = np.clip(te, 0.1, 2.0).astype(np.float32)
            rz = np.maximum(np.abs(tr[:, 2:3]), 1.0e-8)
            kappa = np.clip(
                tb * rz, -self._bend_kappa_max, self._bend_kappa_max)
            tb = (kappa / rz).astype(np.float32)

            # Shift a still-valid shape back into its assigned problem box.
            # If it is wider than the box, or trust projection makes the shift
            # infeasible, restore the complete birth/start row.
            prim_lo, prim_hi = self._primitive_aabbs(tc, tr, tq, tb)
            for row, box_index in enumerate(train_box):
                bmin = boxes[int(box_index)]['box_min']
                bmax = boxes[int(box_index)]['box_max']
                if np.any((prim_hi[row] - prim_lo[row]) > (bmax - bmin) + 1.0e-6):
                    _restore([row])
                    continue
                shift = (
                    np.maximum(bmin - prim_lo[row], 0.0)
                    + np.minimum(bmax - prim_hi[row], 0.0)
                )
                if np.any(np.abs(shift) > 0.0):
                    tc[row] += shift.astype(np.float32)
                    repaired[row] = True

            if self._local_center_trust_radius_factor > 0.0:
                scale = np.mean(
                    np.maximum(np.abs(anchor_r), 1.0e-8), axis=1)
                limit = self._local_center_trust_radius_factor * scale
                delta = tc - anchor_c
                dist = np.linalg.norm(delta, axis=1)
                move = dist > np.maximum(limit, 1.0e-8)
                if np.any(move):
                    tc[move] = (
                        anchor_c[move]
                        + delta[move]
                        * (limit[move] / np.maximum(dist[move], 1.0e-12))[:, None]
                    )
                    repaired[move] = True

            prim_lo, prim_hi = self._primitive_aabbs(tc, tr, tq, tb)
            outside = np.array([
                np.any(prim_lo[row] < boxes[int(box_index)]['box_min'] - 5.0e-6)
                or np.any(prim_hi[row] > boxes[int(box_index)]['box_max'] + 5.0e-6)
                for row, box_index in enumerate(train_box)
            ], dtype=bool)
            _restore(np.flatnonzero(outside))
            return tc, tr, tq, te, tb, repaired

        def _refresh_and_repair_state(st):
            num = st['num_e']
            off = st['offset']
            wp.launch(
                _exp_radii_kernel, dim=num,
                inputs=[st['pred_log_radii'], st['pred_radii']], device=device)
            self._decode_shape_parameters(
                st['pred_eps_raw'], st['pred_eps'],
                st['pred_bend_raw'], st['pred_bend'],
                st['pred_radii'], num)
            wp.synchronize_device(device)
            tc = st['pred_centers'].numpy()[off:].astype(np.float32).copy()
            tr = st['pred_radii'].numpy()[off:].astype(np.float32).copy()
            tq = st['pred_rot_flat'].numpy().reshape(-1, 4)[off:].astype(
                np.float32).copy()
            te = st['pred_eps'].numpy().reshape(-1, 2)[off:].astype(
                np.float32).copy()
            tb = st['pred_bend'].numpy().reshape(-1, 2)[off:].astype(
                np.float32).copy()
            tc, tr, tq, te, tb, repaired = _repair_local_rows(
                tc, tr, tq, te, tb)
            if np.any(repaired):
                full_c = np.concatenate([fixed_c, tc], axis=0).astype(np.float32)
                full_r = np.concatenate([fixed_r, tr], axis=0).astype(np.float32)
                full_q = np.concatenate([fixed_q, tq], axis=0).astype(np.float32)
                full_e = np.concatenate([fixed_e, te], axis=0).astype(np.float32)
                full_b = np.concatenate([fixed_b, tb], axis=0).astype(np.float32)
                st['pred_centers'].assign(np.ascontiguousarray(full_c))
                st['pred_log_radii'].assign(np.ascontiguousarray(
                    np.log(np.maximum(full_r, 1.0e-8)).astype(np.float32)))
                st['pred_rot_flat'].assign(np.ascontiguousarray(
                    full_q.reshape(-1)))
                st['pred_eps_raw'].assign(np.ascontiguousarray(
                    self._eps_raw_np(full_e).reshape(-1)))
                st['pred_bend_raw'].assign(np.ascontiguousarray(
                    self._bend_raw_np(full_b, full_r).reshape(-1)))
                wp.launch(
                    _exp_radii_kernel, dim=num,
                    inputs=[st['pred_log_radii'], st['pred_radii']],
                    device=device)
                self._decode_shape_parameters(
                    st['pred_eps_raw'], st['pred_eps'],
                    st['pred_bend_raw'], st['pred_bend'],
                    st['pred_radii'], num)
                # A repaired parameter row no longer matches its accumulated
                # update direction.  Reset the short-lived local moments so the
                # next step cannot immediately replay the rejected motion.
                for optimizer_name in (
                        'opt_c', 'opt_r', 'opt_q', 'opt_eps', 'opt_bend'):
                    optimizer = st[optimizer_name]
                    optimizer.first.zero_()
                    optimizer.second.zero_()
                    optimizer.age.zero_()
                wp.synchronize_device(device)
            return tc, tr, tq, te, tb

        state = None
        completed_steps = 0
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
            pred_eps_raw = state['pred_eps_raw']
            pred_eps = state['pred_eps']
            pred_bend_raw = state['pred_bend_raw']
            pred_bend = state['pred_bend']
            prev_centers = state['prev_centers']
            min_d_cache = state['min_d_cache']
            sdf_pred = state['sdf_pred']
            loss = state['loss']
            wp_indices = state['wp_indices']
            (cl_lo, cl_hi, cl_logmin, cl_logmax,
             cl_min_step, cl_max_step) = state['clamps']
            steps_this_cycle = cycle_steps[cycle]
            report_every = max(1, steps_this_cycle // 3)
            executed_steps = 0

            for li in range(steps_this_cycle):
                if self._stop_flag:
                    break
                global_local_step = completed_steps + li
                progress = (
                    float(global_local_step)
                    / float(max(self._region_steps - 1, 1)))
                anneal = (
                    0.1
                    + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))
                step_lr = float(lr0 * anneal)
                batch = self._rng.integers(0, P, size=bs).astype(np.int32)
                wp_indices.assign(np.ascontiguousarray(batch))

                tape = wp.Tape()
                with tape:
                    # World radii from trainable log-radii (gradient → log-space).
                    wp.launch(_exp_radii_kernel, dim=num_e,
                              inputs=[pred_log_radii, pred_radii], device=device)
                    self._decode_shape_parameters(
                        pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                        pred_radii, num_e)
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
                                    pred_eps, pred_bend,
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
                                float(self._surface_weight),
                                float(local_surface_sigma),
                                float(self._outside_penalty_weight),
                            wp_thick, float(thick_ref),
                            float(thin_w), float(self._thin_max_factor),
                            max(0.5 * local_dx, 1.0e-8)],
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
                    if self._sq_eps_mode != "shared":
                        wp.launch(_zero_f32_prefix, dim=offset * 2,
                                  inputs=[pred_eps_raw.grad], device=device)
                    wp.launch(_zero_f32_prefix, dim=offset * 2,
                              inputs=[pred_bend_raw.grad], device=device)

                state['opt_c'].lr = step_lr
                state['opt_r'].lr = step_lr * self._lr_mult_radii
                state['opt_q'].lr = step_lr * self._lr_mult_rot
                shape_step = int(gstep if gstep >= 0 else self._num_steps)
                state['opt_eps'].lr = step_lr * self._sq_eps_lr_mult
                state['opt_bend'].lr = step_lr * self._sq_bend_lr_mult
                if self._center_step_radius_frac > 0.0:
                    wp.launch(_copy_vec3_range, dim=n_train,
                              inputs=[pred_centers, prev_centers, offset],
                              device=device)
                state['opt_c'].step(state['grad_c'])
                state['opt_r'].step(state['grad_r'])
                state['opt_q'].step(state['grad_q'])
                if self._eps_is_locally_trainable(shape_step):
                    state['opt_eps'].step(state['grad_eps'])
                if self._bend_is_trainable(shape_step):
                    state['opt_bend'].step(state['grad_bend'])
                tape.zero()

                # Refresh the just-updated derived values before movement and
                # trust projection; otherwise the limiter sees pre-step radii.
                wp.launch(
                    _exp_radii_kernel, dim=num_e,
                    inputs=[pred_log_radii, pred_radii], device=device)
                self._decode_shape_parameters(
                    pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                    pred_radii, num_e)

                # Per-box step limits use the HIGH-RES box voxel size.
                if self._center_step_radius_frac > 0.0:
                    wp.launch(
                        _limit_center_step_by_radius_perbox,
                        dim=n_train,
                        inputs=[
                            pred_centers, prev_centers, pred_radii,
                            cl_min_step, cl_max_step,
                            float(self._center_step_radius_frac),
                            offset,
                        ],
                        device=device)
                wp.launch(_clamp_log_radii_perbox, dim=n_train,
                          inputs=[pred_log_radii, cl_logmin, cl_logmax, offset],
                          device=device)
                wp.launch(
                    _project_local_parameter_trust_region,
                    dim=n_train,
                    inputs=[
                        pred_centers, pred_log_radii,
                        state['anchor_centers'],
                        state['anchor_log_radii'],
                        state['anchor_scales'],
                        float(self._local_center_trust_radius_factor),
                        float(self._local_log_radius_limit),
                        offset,
                    ],
                    device=device)
                wp.launch(_clamp_centers_perbox, dim=n_train,
                          inputs=[pred_centers, cl_lo, cl_hi, offset], device=device)
                wp.launch(_normalize_flat_quats_range, dim=n_train,
                          inputs=[pred_rot_flat, offset], device=device)
                self._project_isotropic(pred_log_radii, pred_rot_flat, num_e)
                self._project_capsule(pred_log_radii, num_e)
                executed_steps = li + 1

                if li % report_every == 0 or li == steps_this_cycle - 1:
                    sc, sr, sq, se, sb = _refresh_and_repair_state(state)
                    vis_c = np.concatenate([fixed_c, sc], axis=0)
                    vis_r = np.concatenate([fixed_r, sr], axis=0)
                    vis_q = np.concatenate([fixed_q, sq], axis=0)
                    vis_e = np.concatenate([fixed_e, se], axis=0)
                    vis_b = np.concatenate([fixed_b, sb], axis=0)
                    extra = (np.concatenate([vis_e, vis_b], axis=1)
                             if self._bent else vis_e) if self._superquadric else None
                    done = completed_steps + li + 1
                    self.local_progress.emit(done, self._region_steps)
                    self.step_visual.emit(int(gstep), float(loss.numpy()[0]),
                                          vis_c.copy(), vis_r.copy(), vis_q.copy(),
                                          None if extra is None else extra.copy())
                    self._emit_live_metric_if_needed(
                        int(gstep), vis_c.copy(), vis_r.copy(), vis_q.copy(),
                        vis_e.copy(), vis_b.copy())

            train_c, train_r, train_q, train_e, train_b = \
                _refresh_and_repair_state(state)
            completed_steps += executed_steps

            # Population edits are only allowed for a real densify-triggered,
            # single-box fit.  Ordinary Local Fit therefore cannot silently
            # delete or split an existing ellipsoid.
            if dc_enabled and cycle < n_cycles - 1 and not self._stop_flag:
                previous_train_lineage = local_lineage[len(fixed_c):].copy()
                previous_train = (
                    train_c.copy(), train_r.copy(), train_q.copy(),
                    train_e.copy(), train_b.copy())
                previous_anchors = (
                    anchor_c.copy(), anchor_r.copy(), anchor_q.copy(),
                    anchor_e.copy(), anchor_b.copy())
                (train_c, train_r, train_q, train_e, train_b,
                 train_box, changed) = \
                    self._region_dc_all_boxes(
                        fixed_c, fixed_r, fixed_q,
                        train_c, train_r, train_q, train_box, boxes,
                        population_cap=population_cap,
                        fixed_eps=fixed_e, fixed_bend=fixed_b,
                        train_eps=train_e, train_bend=train_b)
                dc_map = np.asarray(
                    getattr(self, "_last_region_dc_all_lineage",
                            np.arange(len(train_c), dtype=np.int64)),
                    dtype=np.int64,
                )
                mapped_train = np.full(len(dc_map), -1, dtype=np.int64)
                valid_dc = ((dc_map >= 0) & (dc_map < len(previous_train_lineage)))
                mapped_train[valid_dc] = previous_train_lineage[dc_map[valid_dc]]
                local_lineage = np.concatenate([
                    local_lineage[:len(fixed_c)], mapped_train,
                ])
                if changed:
                    # Survivors keep their original trust anchor.  Split,
                    # merged, or otherwise changed rows get a birth anchor and
                    # fresh Adam moments so siblings never inherit one parent's
                    # momentum.
                    anchor_c = train_c.copy()
                    anchor_r = train_r.copy()
                    anchor_q = train_q.copy()
                    anchor_e = train_e.copy()
                    anchor_b = train_b.copy()
                    counts = np.bincount(
                        dc_map[dc_map >= 0],
                        minlength=len(previous_train[0]))
                    for row, parent_row in enumerate(dc_map):
                        parent_row = int(parent_row)
                        if not (0 <= parent_row < len(previous_train[0])):
                            continue
                        unchanged = (
                            counts[parent_row] == 1
                            and np.array_equal(
                                train_c[row], previous_train[0][parent_row])
                            and np.array_equal(
                                train_r[row], previous_train[1][parent_row])
                            and np.array_equal(
                                train_q[row], previous_train[2][parent_row])
                            and np.array_equal(
                                train_e[row], previous_train[3][parent_row])
                            and np.array_equal(
                                train_b[row], previous_train[4][parent_row])
                        )
                        if unchanged:
                            anchor_c[row] = previous_anchors[0][parent_row]
                            anchor_r[row] = previous_anchors[1][parent_row]
                            anchor_q[row] = previous_anchors[2][parent_row]
                            anchor_e[row] = previous_anchors[3][parent_row]
                            anchor_b[row] = previous_anchors[4][parent_row]
                    state = None

        out_c = np.concatenate([fixed_c, train_c], axis=0).astype(np.float32)
        out_r = np.concatenate([fixed_r, train_r], axis=0).astype(np.float32)
        out_q = np.concatenate([fixed_q, train_q], axis=0).astype(np.float32)
        out_e = np.concatenate([fixed_e, train_e], axis=0).astype(np.float32)
        out_b = np.concatenate([fixed_b, train_b], axis=0).astype(np.float32)

        def _validation_loss(c_values, r_values, q_values, e_values, b_values):
            prediction = self._pred_points_from_params(
                validation_sample.points,
                c_values, r_values, q_values, e_values, b_values)
            return float(evaluate_validation_loss(
                prediction,
                validation_sample,
                huber_delta=max(0.5 * local_dx, 1.0e-8),
                miss_weight=float(self._miss_penalty_weight),
                surface_weight=float(self._surface_weight),
                surface_sigma=float(local_surface_sigma),
                outside_weight=float(self._outside_penalty_weight),
                thin_weight=float(thin_w),
                thin_max_factor=float(self._thin_max_factor),
                thickness_reference=(thick_ref if any_thick else None),
                coarse_far_weight=0.0,
            ).total)

        # Counterfactual acceptance is row-local: retain a refined primitive
        # only if the deterministic local union is no worse than replacing that
        # one row by its start/birth anchor.  This catches a single broken
        # ellipsoid without discarding useful changes made by its neighbours.
        rollback_rows: list[int] = []
        current_validation_loss = _validation_loss(
            out_c, out_r, out_q, out_e, out_b)
        for row in range(len(train_c)):
            out_row = n_fixed + row
            unchanged = (
                np.array_equal(out_c[out_row], anchor_c[row])
                and np.array_equal(out_r[out_row], anchor_r[row])
                and np.array_equal(out_q[out_row], anchor_q[row])
                and np.array_equal(out_e[out_row], anchor_e[row])
                and np.array_equal(out_b[out_row], anchor_b[row])
            )
            if unchanged:
                continue
            trial_c = out_c.copy()
            trial_r = out_r.copy()
            trial_q = out_q.copy()
            trial_e = out_e.copy()
            trial_b = out_b.copy()
            trial_c[out_row] = anchor_c[row]
            trial_r[out_row] = anchor_r[row]
            trial_q[out_row] = anchor_q[row]
            trial_e[out_row] = anchor_e[row]
            trial_b[out_row] = anchor_b[row]
            trial_loss = _validation_loss(
                trial_c, trial_r, trial_q, trial_e, trial_b)
            tolerance = max(
                1.0e-8,
                1.0e-5 * abs(current_validation_loss)
                if np.isfinite(current_validation_loss) else 0.0)
            if (not np.isfinite(current_validation_loss)
                    or trial_loss + tolerance < current_validation_loss):
                out_c, out_r, out_q, out_e, out_b = (
                    trial_c, trial_r, trial_q, trial_e, trial_b)
                current_validation_loss = trial_loss
                rollback_rows.append(out_row)

        self._last_local_fit_rollback_rows = np.asarray(
            rollback_rows, dtype=np.int64)
        if rollback_rows:
            extra = (
                np.concatenate([out_e, out_b], axis=1)
                if self._bent else out_e
            ) if self._superquadric else None
            self.step_visual.emit(
                int(gstep), float(current_validation_loss),
                out_c.copy(), out_r.copy(), out_q.copy(),
                None if extra is None else extra.copy())
        self._last_local_fit_lineage = local_lineage
        return out_c, out_r, out_q, out_e, out_b

    def _maybe_superfit(self, step, pred_centers, pred_radii, pred_rot_flat,
                        pred_eps=None, pred_bend=None):
        """SuperFit cycle (adaptive density control, à la Gaussian Splatting).

        Three moves, then one isolated local fit of everything new:
          - **fuse** (prune) redundant ellipsoids whose interior is already
            covered by the others — they have no independent task, so dropping
            them frees budget without changing the union SDF;
          - **split** oversized ellipsoids that protrude past the mesh surface
            into two halves (over-reconstruction → divide), and
          - **spawn** fresh ellipsoids in under-represented regions
            (under-reconstruction → conquer).

        Returns new ``(centers, radii, rotations, eps, bend)`` when the
        population changed, else ``None``.  Net growth stops once
        ``max_ellipsoids`` is reached
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
        eps = (None if pred_eps is None
               else pred_eps.numpy().reshape(-1, 2).copy())
        bend = (None if pred_bend is None
                else pred_bend.numpy().reshape(-1, 2).copy())
        eps, bend = self._shape_state_np(len(centers), eps, bend)
        eps_before_edits = eps.copy()

        n_before = len(centers)
        lineage = np.arange(n_before, dtype=np.int64)
        self._last_population_lineage = None

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
        out_idx = (self._detect_outside_ellipsoids(
            centers, radii, rotations, eps, bend)
                   if densify_active else np.empty(0, dtype=int))
        n_deleted = int(len(out_idx))
        if n_deleted > 0:
            _record('delete', out_idx, centers, radii)
            keep = np.ones(len(centers), dtype=bool)
            keep[out_idx] = False
            centers, radii, rotations = centers[keep], radii[keep], rotations[keep]
            eps, bend = eps[keep], bend[keep]
            lineage = lineage[keep]

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
            eps, bend = eps[keep], bend[keep]
            lineage = lineage[keep]

        # ── 0b) Fuse redundant ellipsoids (no independent task → drop them) ──
        # Done first so freed slots are reused by the split/spawn moves below.
        # Gated by the Prune toggle (this is the population-shrinking pruning;
        # the 0a/0a2 safety deletes above stay on regardless).
        fuse_idx = (self._detect_redundant_ellipsoids(
            centers, radii, rotations, self._fuse_per_round, eps, bend)
            if (densify_active and self._prune_enabled) else np.empty(0, dtype=int))
        n_fused = int(len(fuse_idx)) + n_deleted
        if len(fuse_idx) > 0:
            _record('fuse', fuse_idx, centers, radii)
            keep = np.ones(len(centers), dtype=bool)
            keep[fuse_idx] = False
            centers, radii, rotations = centers[keep], radii[keep], rotations[keep]
            eps, bend = eps[keep], bend[keep]
            lineage = lineage[keep]

        # ── 0c) Merge overlapping pairs into one when it barely moves the surface
        if densify_active:
            lineage_before_merge = lineage
            (centers, radii, rotations, eps, bend,
             n_merged) = self._detect_merges(
                centers, radii, rotations, eps, bend)
            merge_map = np.asarray(
                getattr(self, "_last_merge_lineage", np.arange(len(centers))),
                dtype=np.int64,
            )
            lineage = np.full(len(merge_map), -1, dtype=np.int64)
            valid_merge = ((merge_map >= 0)
                           & (merge_map < len(lineage_before_merge)))
            lineage[valid_merge] = lineage_before_merge[merge_map[valid_merge]]
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
        viz_bridge = self._detect_bridging_ellipsoids(
            centers, radii, rotations, eps, bend)
        viz_protr = self._detect_protruding_ellipsoids(
            centers, radii, rotations, eps, bend)
        # Region detection uses a cached, bounded set of exact target-grid
        # samples.  Outputs are world-space (seed_world / seed_depth), so
        # downstream split/spawn remains independent of grid indexing.
        with self._detection_grid_scope():
            viz_regions = self._detect_worst_regions(
                centers, radii, rotations, self._analysis_region_k,
                min_severity=self._analysis_min_severity,
                eps=eps, bend=bend)
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
            self.ellipsoid_metrics.emit(
                step,
                self._compute_ellipsoid_quality_metrics(
                    centers, radii, rotations, viz_regions,
                    eps=eps, bend=bend))

        n_curr = len(centers)
        # Net additions allowed this cycle.  Zero outside the densify window (or
        # at the cap) so the split/spawn detectors below naturally produce
        # nothing — but local fit may still run afterwards.
        budget = (self._max_ellipsoids - n_curr) if densify_active else 0
        budget = max(0, budget)
        if self._symmetry_enabled and self._sym_axis is not None:
            # Off-plane growth consumes a source slot and its mandatory mirror.
            # Reserve pair slots up front; the final layout enforces the exact
            # cap for the rarer on-plane case.
            budget //= 2
        bone_assign, bone_counts, bone_caps = self._bone_growth_state(centers)

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
                    if not self._reserve_split_bone_capacity(
                            v, centers, radii, rotations,
                            bone_assign, bone_counts, bone_caps, eps, bend):
                        continue
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
                centers, radii, rotations, regions, budget - len(targets), exclude=seen,
                split_enabled=self._split_enabled,
                spawn_enabled=self._spawn_underrep,
                bone_assign=bone_assign,
                bone_counts=bone_counts,
                bone_caps=bone_caps,
                eps=eps, bend=bend)
            targets += split_tgts
            if spawn_regions:
                spawn_c, spawn_r, spawn_q, spawn_sites = self._spawn_in_regions(
                    spawn_regions, len(spawn_regions),
                    reference_eps=eps_before_edits)
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
                    mr = spawn_r.copy()
                    mq = _mirror_quats(spawn_q, a).astype(np.float32)
                    keep_mirror: list[int] = []
                    mirror_assign = self._nearest_bone_indices_np(mc)
                    for mi in range(len(mc)):
                        bi = None
                        if mirror_assign is not None:
                            bi = int(mirror_assign[mi])
                        if self._bone_has_add_capacity(bi, bone_counts, bone_caps):
                            keep_mirror.append(mi)
                            if (bone_counts is not None and bi is not None
                                    and 0 <= bi < len(bone_counts)):
                                bone_counts[bi] += 1
                    if keep_mirror:
                        midx = np.asarray(keep_mirror, dtype=int)
                        old_sites = list(spawn_sites)
                        spawn_c = np.concatenate([spawn_c, mc[midx]], axis=0).astype(np.float32)
                        spawn_r = np.concatenate([spawn_r, mr[midx]], axis=0).astype(np.float32)
                        spawn_q = np.concatenate([spawn_q, mq[midx]], axis=0).astype(np.float32)
                        spawn_sites = spawn_sites + [
                            (mc[i].astype(np.float32), old_sites[i][1])
                            for i in keep_mirror]
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
        child_c, child_r, child_q, child_e, child_b, pools = [], [], [], [], [], []
        region_sites = []   # (center_world, half_extent) per maintained region
        for i in split_idx:
            cc, cr, cq, ce, cb = self._split_primitive(
                centers[i], radii[i], rotations[i], eps[i], bend[i])
            child_c.append(cc); child_r.append(cr); child_q.append(cq)
            child_e.append(ce)
            child_b.append(cb)
            pools.append(self._interior_ball_pool(centers[i], self._region_radius_vox))
            parent_max_r = float(np.max(np.abs(radii[i])))
            half = max(r_radius_world, parent_max_r * float(self._split_size_factor))
            region_sites.append((centers[i].astype(np.float32).copy(), half))
            ops.append(('split', centers[i].astype(np.float32).copy(), parent_max_r))

        # Append spawned ellipsoids and their region sites.
        if n_spawn > 0:
            child_c.append(spawn_c); child_r.append(spawn_r); child_q.append(spawn_q)
            child_e.append(self._new_primitive_eps(
                n_spawn, eps_before_edits))
            child_b.append(self._init_bend(n_spawn))
            region_sites.extend(spawn_sites)
            for site_c, _half in spawn_sites:
                pools.append(self._interior_ball_pool(site_c, self._region_radius_vox))

        keep = np.ones(len(centers), dtype=bool)   # current (post-fusion) length
        keep[split_idx] = False
        split_parent_lineage = lineage[split_idx].copy()
        centers, radii, rotations = centers[keep], radii[keep], rotations[keep]
        eps, bend = eps[keep], bend[keep]
        survivor_lineage = lineage[keep]
        offset = len(centers)        # frozen prefix = surviving originals

        centers   = np.concatenate([centers]   + child_c, axis=0)
        radii     = np.concatenate([radii]     + child_r, axis=0)
        rotations = np.concatenate([rotations] + child_q, axis=0)
        eps       = np.concatenate([eps]       + child_e, axis=0)
        bend      = np.concatenate([bend]      + child_b, axis=0)
        appended_lineage: list[np.ndarray] = [
            np.repeat(value, 2).astype(np.int64)
            for value in split_parent_lineage
        ]
        if n_spawn > 0:
            appended_lineage.append(np.full(n_spawn, -1, dtype=np.int64))
        lineage = np.concatenate(
            [survivor_lineage] + appended_lineage,
            axis=0,
        ) if appended_lineage else survivor_lineage

        # Local fit with no fresh densify regions (e.g. local window extends past
        # the densify window, densify added nothing this cycle, or SuperFit is
        # off entirely): source the worst regions to re-fit the EXISTING geometry
        # there.  ``_local_fit_regions`` assigns trainable ellipsoids by
        # region-box membership, so refitting in place needs no appended children.
        if local_active and not region_sites:
            # Local fit is expensive and intended for delicate details.  Do its
            # own region ranking with a thickness boost instead of reusing the
            # generic analysis/densify list, which can be dominated by large
            # thick structures when their absolute miss is high.
            with self._detection_grid_scope():
                lf_regions = self._detect_worst_regions(
                    centers, radii, rotations, self._local_fit_region_k,
                    min_severity=self._local_fit_min_severity,
                    thin_preference=self._local_fit_thin_preference,
                    eps=eps, bend=bend)
            if lf_regions and len(centers):
                # Keep the box centred on the actual thin/problem region.  Do
                # not snap it to the nearest ellipsoid centre: on characters
                # that often means a large belly/torso primitive wins the
                # nearest-centre test and the expensive local fit moves back to
                # the thick middle of the body.
                r_radius_world = float(self._region_radius_vox) * float(self._dx)
                seen_sites: set[tuple] = set()
                for reg in lf_regions:
                    sc = np.asarray(reg['seed_world'], np.float32).copy()
                    key = tuple(np.round(sc / max(float(self._dx), 1e-9)).astype(int))
                    if key in seen_sites:
                        continue
                    seen_sites.add(key)
                    region_sites.append((sc, r_radius_world))
                    pools.append(self._interior_ball_pool(
                        sc, self._region_radius_vox))

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
                    box_geoms, n=self._region_res, compute_thickness=False)
                for box_result in box_results:
                    self._apply_blowup_to_region_result(box_result)
                # One combined fit over ALL region boxes (replaces the former
                # per-region serial loop): faster and less erratic, see
                # _local_fit_regions.
                local_population_cap = int(self._max_ellipsoids)
                if self._symmetry_enabled and self._sym_axis is not None:
                    free_slots = max(
                        0, int(self._max_ellipsoids) - int(len(centers)))
                    local_population_cap = int(len(centers)) + free_slots // 2
                centers, radii, rotations, eps, bend = self._local_fit_regions(
                    centers, radii, rotations, region_sites, box_results, step,
                    population_cap=local_population_cap,
                    eps=eps, bend=bend,
                    allow_region_dc=bool(
                        densify_active and (n_split > 0 or n_spawn > 0)))
                local_map = np.asarray(
                    getattr(self, "_last_local_fit_lineage",
                            np.arange(len(centers))),
                    dtype=np.int64,
                )
                before_local = lineage
                lineage = np.full(len(local_map), -1, dtype=np.int64)
                valid_local = ((local_map >= 0)
                               & (local_map < len(before_local)))
                lineage[valid_local] = before_local[local_map[valid_local]]
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
                centers, radii, rotations, eps, bend = self._local_fit(
                    centers, radii, rotations, offset, union_pool, step,
                    eps=eps, bend=bend,
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
        self._last_population_lineage = lineage
        return centers, radii, rotations, eps, bend

    def _maybe_maintain(self, step, pred_centers, pred_radii, pred_rot_flat,
                        pred_eps=None, pred_bend=None):
        self._last_population_lineage = None
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
        e = (None if pred_eps is None
             else pred_eps.numpy().reshape(-1, 2).copy())
        b = (None if pred_bend is None
             else pred_bend.numpy().reshape(-1, 2).copy())

        n_before = len(c)
        (c, r, q, e, b, changed, n_pruned,
         n_spawned) = self._do_maintenance(c, r, q, e, b)
        self.maintenance_done.emit(step, n_before, n_pruned, n_spawned)

        if not changed:
            return None
        self._last_population_lineage = np.asarray(
            getattr(self, "_last_maintenance_lineage",
                    np.full(len(c), -1, dtype=np.int64)),
            dtype=np.int64,
        )
        return c, r, q, e, b

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

    def _ensure_sample_targets_wp(self) -> UploadedSdfSamples | None:
        """Upload sparse/sample SDF targets for point-based loss batches."""
        if self._sdf_samples is None:
            return None
        if self._uploaded_samples is None:
            self._uploaded_samples = UploadedSdfSamples(self._sdf_samples, device)
            th = self._sdf_samples.thickness
            if th is not None:
                valid = th[th > 0.0]
                self._thick_ref = float(np.median(valid)) if valid.size else 1.0
                self._thin_weight_eff = (
                    float(self._thin_loss_weight) if valid.size else 0.0)
            else:
                self._thick_ref = 1.0
                self._thin_weight_eff = 0.0
        return self._uploaded_samples

    def _build_validation_sample(self):
        """Create the deterministic hold-out used for checkpoint selection.

        Validation is intentionally independent from ``BandSampler`` and its
        stochastic mini-batches.  Dense fits use the same dilated thickness
        field as the production loss; sparse fits preserve the coarse/far-field
        stratum contained in ``SdfSampleSet``.
        """
        if self._sdf_samples is not None:
            coarse_fraction = (
                float(SPARSE_FAR_FIELD_FRACTION)
                if self._sdf_samples.coarse_mask is not None else 0.0)
            return stratified_validation_from_samples(
                self._sdf_samples,
                sample_count=self._validation_sample_size,
                surface_band=float(self._surface_band_vox) * float(self._dx),
                surface_fraction=min(
                    float(self._surface_fraction),
                    1.0 - coarse_fraction),
                coarse_fraction=coarse_fraction,
                seed=0,
            )

        thickness = None
        if self._thickness_flat is not None:
            thickness = np.asarray(
                self._thickness_flat, dtype=np.float32).reshape(self._shape)
        return stratified_validation_from_grid(
            self._sdf_target_np,
            self._origin,
            self._dx,
            thickness=thickness,
            sample_count=self._validation_sample_size,
            surface_band=float(self._surface_band_vox) * float(self._dx),
            surface_fraction=float(self._surface_fraction),
            seed=0,
        )

    # ── naive SGD ─────────────────────────────────────────────────────

    def _run_naive(self):
        origin = self._origin
        n = self._n
        nx, ny, nz = self._nx, self._ny, self._nz
        dx = self._dx
        total = nx * ny * nz
        if self._symmetry_enabled and not self._sym_checked:
            self._setup_symmetry()
        self._ensure_thickness_wp(total)
        num_e = self._num_ellipsoids
        bs = self._batch_size

        buf = self._alloc_buffers(
            num_e, bs, total,
            self._initial_centers,
            self._initial_radii,
            self._initial_rotations,
            eps_np=self._initial_eps,
            bend_np=self._initial_bend,
            apply_initial_symmetry=True,
        )
        pred_centers  = buf['pred_centers']
        pred_radii    = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']
        pred_eps_raw  = buf['pred_eps_raw']
        pred_eps      = buf['pred_eps']
        pred_bend_raw = buf['pred_bend_raw']
        pred_bend     = buf['pred_bend']
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

            # Maintenance must see the shape state produced by the previous raw
            # update, even on steps where no progress frame was emitted.
            self._decode_shape_parameters(
                pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                pred_radii, num_e)
            result = self._maybe_maintain(
                step, pred_centers, pred_radii, pred_rot_flat,
                pred_eps, pred_bend)
            if result is not None:
                c_np, r_np, q_np, eps_np, bend_np = result
                r_np, q_np = self._project_isotropic_np(r_np, q_np)   # sphere
                r_np = self._project_capsule_np(r_np)
                if self._symmetry_enabled and self._sym_axis is not None:
                    c_np, r_np, q_np, eps_np, bend_np = \
                        self._build_symmetric_layout(
                            c_np, r_np, q_np, eps_np, bend_np)
                num_e = len(c_np)
                buf = self._alloc_buffers(
                    num_e, bs, total, c_np, r_np, q_np,
                    eps_np=eps_np, bend_np=bend_np)
                buf['sdf_target'] = sdf_target
                pred_centers  = buf['pred_centers']
                pred_radii    = buf['pred_radii']
                pred_rot_flat = buf['pred_rot_flat']
                pred_eps_raw  = buf['pred_eps_raw']
                pred_eps      = buf['pred_eps']
                pred_bend_raw = buf['pred_bend_raw']
                pred_bend     = buf['pred_bend']
                min_d_cache   = buf['min_d_cache']
                sdf_pred      = buf['sdf_pred']
                loss          = buf['loss']
                wp_indices    = buf['wp_indices']

            wp_indices.assign(sampler.next_batch())

            tape = wp.Tape()
            with tape:
                self._decode_shape_parameters(
                    pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                    pred_radii, num_e)
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
                                pred_eps, pred_bend, min_d_cache,
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
                            float(self._thin_weight_eff), float(self._thin_max_factor),
                            max(0.5 * float(dx), 1.0e-8)],
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
            if self._eps_is_trainable(step):
                wp.launch(
                    _sgd_step_f32,
                    dim=len(pred_eps_raw),
                    inputs=[pred_eps_raw, tape.gradients[pred_eps_raw],
                            float(lr * self._sq_eps_lr_mult)],
                    device=device)
            if self._bend_is_trainable(step):
                wp.launch(
                    _sgd_step_f32,
                    dim=len(pred_bend_raw),
                    inputs=[pred_bend_raw, tape.gradients[pred_bend_raw],
                            float(lr * self._sq_bend_lr_mult)],
                    device=device)
            wp.launch(_normalize_flat_quats, dim=num_e,
                      inputs=[pred_rot_flat], device=device)
            # Sphere: naive path trains world radii directly — project them
            # (mean of the 3 components) + reset rotation each step.
            self._project_isotropic(pred_radii, pred_rot_flat, num_e)
            # Capsule: circular cross-section (r1 = r0).
            self._project_capsule(pred_radii, num_e)
            if self._symmetry_enabled and self._sym_axis is not None:
                self._project_symmetry_inplace(
                    pred_centers, pred_radii, pred_rot_flat,
                    pred_eps_raw
                    if self._sq_eps_mode == "per_primitive" else None,
                    pred_bend_raw if self._bent else None)

            tape.zero()

            if step % self._report_every == 0:
                self._decode_shape_parameters(
                    pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                    pred_radii, num_e)
                self._emit_progress(step, loss, pred_centers, pred_radii,
                                    pred_rot_flat, num_e, origin, dx, n,
                                    pred_eps=pred_eps if self._superquadric else None,
                                    pred_bend=pred_bend if self._bent else None)

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
        # Symmetry may replace the sparse set with a differently sized,
        # exactly-paired set.  Capture all sample-dependent state only after
        # that replacement so uploads, buffers, masks and sampler indices refer
        # to the same collection.
        sample_targets = self._sdf_samples
        use_sample_targets = sample_targets is not None
        train_total = int(sample_targets.size) if use_sample_targets else total
        if use_sample_targets:
            self.prep_progress.emit(0.30, "uploading sparse samples")
            uploaded_samples = self._ensure_sample_targets_wp()
            target_values_np = sample_targets.values
            target_thickness_np = sample_targets.thickness
            if target_thickness_np is None:
                target_thickness_np = np.zeros(train_total, dtype=np.float32)
        else:
            uploaded_samples = None
            self.prep_progress.emit(0.30, "feature thickness")
            self._ensure_thickness_wp(total)
            target_values_np = self._sdf_target_np.ravel()
            target_thickness_np = self._thickness_flat
        num_e = self._num_ellipsoids
        bs = self._batch_size

        self.prep_progress.emit(0.55, "allocating buffers")
        def _buffer_progress(frac, msg):
            self.prep_progress.emit(
                0.55 + (0.78 - 0.55) * float(frac),
                str(msg),
            )
        buf = self._alloc_buffers(
            num_e, bs, train_total,
            self._initial_centers,
            self._initial_radii,
            self._initial_rotations,
            sdf_target_np=target_values_np if use_sample_targets else None,
            eps_np=self._initial_eps,
            bend_np=self._initial_bend,
            progress_cb=_buffer_progress,
            apply_initial_symmetry=True)
        pred_centers  = buf['pred_centers']
        pred_radii    = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']
        pred_eps_raw  = buf['pred_eps_raw']
        pred_eps      = buf['pred_eps']
        pred_bend_raw = buf['pred_bend_raw']
        pred_bend     = buf['pred_bend']
        min_d_cache   = buf['min_d_cache']
        sdf_pred      = buf['sdf_pred']
        loss          = buf['loss']
        sdf_target    = buf['sdf_target']
        wp_indices    = buf['wp_indices']
        # Init (e.g. the symmetric layout) may yield a different count than
        # requested — track the actual number so kernel launches match the arrays.
        num_e = int(pred_centers.shape[0])
        if self._parameterized and num_e != int(self._num_ellipsoids):
            raise RuntimeError("bone-local corrective fitting cannot change population size")

        fit_centers = pred_centers
        fit_rot_flat = pred_rot_flat
        parameter_rows = None
        parameter_offsets = None
        parameter_prefixes = None
        if self._parameterized:
            fit_centers = wp.empty(
                num_e, dtype=wp.vec3, device=device, requires_grad=True)
            fit_rot_flat = wp.empty(
                num_e * 4, dtype=wp.float32, device=device, requires_grad=True)
            parameter_rows = tuple(
                wp.array(
                    np.ascontiguousarray(self._parameter_linear_np[:, row, :]),
                    dtype=wp.vec3, device=device)
                for row in range(3)
            )
            parameter_offsets = wp.array(
                np.ascontiguousarray(self._parameter_offset_np),
                dtype=wp.vec3, device=device)
            parameter_prefixes = wp.array(
                np.ascontiguousarray(self._parameter_rotation_prefix_np),
                dtype=wp.quat, device=device)

        def _refresh_parameter_world() -> None:
            if not self._parameterized:
                return
            wp.launch(
                _parameter_local_to_world_kernel,
                dim=num_e,
                inputs=[
                    pred_centers, pred_rot_flat,
                    parameter_rows[0], parameter_rows[1], parameter_rows[2],
                    parameter_offsets, parameter_prefixes,
                    fit_centers, fit_rot_flat,
                ],
                device=device,
            )

        def _upload_parameter_anchor(values):
            if values is None:
                return None
            centers_np, radii_np, rotations_np = values
            radii_np = np.maximum(np.asarray(radii_np, dtype=np.float32), 1.0e-7)
            scales_np = np.maximum(
                np.max(radii_np, axis=1), float(dx)).astype(np.float32)
            return (
                wp.array(np.ascontiguousarray(centers_np),
                         dtype=wp.vec3, device=device),
                wp.array(np.ascontiguousarray(np.log(radii_np)),
                         dtype=wp.vec3, device=device),
                wp.array(np.ascontiguousarray(rotations_np).reshape(-1),
                         dtype=wp.float32, device=device),
                wp.array(scales_np, dtype=wp.float32, device=device),
            )

        parameter_anchor_wp = _upload_parameter_anchor(self._parameter_anchor)
        parameter_neighbor_wp = _upload_parameter_anchor(self._parameter_neighbor)
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
            target_values_np, bs,
            float(self._surface_band_vox) * float(dx),
            self._surface_fraction, rng=self._rng,
            flat_thickness=target_thickness_np,
            thin_bias=float(self._thin_sample_bias),
            coarse_mask=sample_targets.coarse_mask if use_sample_targets else None,
        )
        validation_sample = self._build_validation_sample()
        # Checkpoint selection and early-stop patience are separate concerns:
        # the former spans the whole run, while the latter begins only after
        # population edits and shape unlocks have ended.
        best_checkpoint = BestCheckpoint(None, 0.0)
        validation_patience = Patience(
            self._validation_patience, self._validation_min_delta)
        self.best_validation_loss = float("inf")
        self.best_validation_step = None
        self.validation_history = []

        # Population edits and late shape unlocks deliberately make the loss
        # non-stationary.  Patience is therefore observed only once all such
        # phases have ended; checkpoints themselves are valid from step zero.
        stable_fraction = (
            float(self._densify_until_frac) if self._superfit else 0.0)
        if self._local_fit_enabled:
            stable_fraction = max(stable_fraction, float(self._local_fit_end_frac))
        if self._superquadric and self._sq_eps_mode != "fixed":
            stable_fraction = max(stable_fraction, float(self._sq_unlock_frac))
        if self._bent:
            stable_fraction = max(stable_fraction, float(self._sq_bend_unlock_frac))
        validation_stable_step = int(np.ceil(
            np.clip(stable_fraction, 0.0, 1.0) * max(self._num_steps - 1, 0)))
        validation_stable_step = min(
            max(self._num_steps - 1, 0),
            validation_stable_step + 2 * self._validation_every)
        last_discontinuity_step = -10**9
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
        def _make_opts(previous=None, lineage=None):
            log_r = wp.array(
                np.log(np.maximum(pred_radii.numpy(), 1e-6)),
                dtype=wp.vec3, device=device, requires_grad=True)
            previous = previous or {}
            if lineage is None:
                state_c = state_r = state_q = state_eps = state_bend = None
            else:
                state_c = _PopulationAdam.remap(
                    previous.get("centers"), lineage, 1)
                state_r = _PopulationAdam.remap(
                    previous.get("radii"), lineage, 1)
                state_q = _PopulationAdam.remap(
                    previous.get("rotation"), lineage, 4)
                state_eps = (previous.get("eps")
                             if self._sq_eps_mode == "shared"
                             else _PopulationAdam.remap(
                                 previous.get("eps"), lineage, 2))
                state_bend = _PopulationAdam.remap(
                    previous.get("bend"), lineage, 2)
            oc = _PopulationAdam(pred_centers, lr=lr, state=state_c)
            orad = _PopulationAdam(log_r, lr=lr, state=state_r)
            oq = _PopulationAdam(pred_rot_flat, lr=lr, state=state_q)
            # Per-element ages let freshly spawned parameters begin with correct
            # Adam bias correction while survivors retain their full history.
            oeps = _PopulationAdam(pred_eps_raw, lr=lr, state=state_eps)
            obend = _PopulationAdam(pred_bend_raw, lr=lr, state=state_bend)
            return (log_r, oc, orad, oq, oeps, obend,
                    wp.empty(num_e, dtype=wp.vec3, device=device),
                    [pred_centers.grad.flatten()], [log_r.grad.flatten()],
                    [pred_rot_flat.grad.flatten()], [pred_eps_raw.grad.flatten()],
                    [pred_bend_raw.grad.flatten()])

        self.prep_progress.emit(0.95, "optimizer")
        (pred_log_radii, opt_c, opt_r, opt_q, opt_eps, opt_bend,
         prev_centers, grad_c, grad_r, grad_q, grad_eps, grad_bend) = _make_opts()

        primitive_ids = np.arange(num_e, dtype=np.int64)
        next_primitive_id = int(num_e)
        self._primitive_ids = primitive_ids.copy()

        def _snapshot_opts():
            return {
                "centers": opt_c.snapshot(),
                "radii": opt_r.snapshot(),
                "rotation": opt_q.snapshot(),
                "eps": opt_eps.snapshot(),
                "bend": opt_bend.snapshot(),
            }

        def _advance_ids(old_ids: np.ndarray, row_lineage: np.ndarray) -> np.ndarray:
            nonlocal next_primitive_id
            old_ids = np.asarray(old_ids, dtype=np.int64).reshape(-1)
            row_lineage = np.asarray(row_lineage, dtype=np.int64).reshape(-1)
            out = np.empty(len(row_lineage), dtype=np.int64)
            claimed: set[int] = set()
            for new_row, old_row in enumerate(row_lineage):
                if 0 <= int(old_row) < len(old_ids):
                    candidate = int(old_ids[int(old_row)])
                    if candidate not in claimed:
                        out[new_row] = candidate
                        claimed.add(candidate)
                        continue
                out[new_row] = next_primitive_id
                next_primitive_id += 1
            return out

        def _validation_state() -> dict[str, np.ndarray]:
            """Read one coherent post-update population snapshot."""
            wp.synchronize_device(device)
            state = {
                "centers": np.asarray(
                    pred_centers.numpy(), dtype=np.float32).reshape(-1, 3),
                "world_centers": np.asarray(
                    fit_centers.numpy(), dtype=np.float32).reshape(-1, 3),
                "radii": np.asarray(
                    pred_radii.numpy(), dtype=np.float32).reshape(-1, 3),
                "rotations": np.asarray(
                    pred_rot_flat.numpy(), dtype=np.float32).reshape(-1, 4),
                "world_rotations": np.asarray(
                    fit_rot_flat.numpy(), dtype=np.float32).reshape(-1, 4),
                "primitive_ids": np.asarray(primitive_ids, dtype=np.int64),
                "symmetry_partition": np.array(
                    [self._sym_n_op, self._sym_n_so], dtype=np.int64),
            }
            if self._superquadric:
                state["eps"] = np.asarray(
                    pred_eps.numpy(), dtype=np.float32).reshape(-1, 2)
                state["bend"] = np.asarray(
                    pred_bend.numpy(), dtype=np.float32).reshape(-1, 2)
            return state

        def _record_validation(step_index: int) -> float:
            state = _validation_state()
            geometry_keys = [
                "centers", "world_centers", "radii",
                "rotations", "world_rotations",
            ]
            if self._superquadric:
                geometry_keys.extend(["eps", "bend"])
            finite_geometry = all(
                np.isfinite(state[name]).all() for name in geometry_keys)
            if finite_geometry:
                prediction = self._pred_points_from_params(
                    validation_sample.points,
                    state["world_centers"], state["radii"],
                    state["world_rotations"],
                    state.get("eps"), state.get("bend"),
                )
                measured = evaluate_validation_loss(
                    prediction,
                    validation_sample,
                    huber_delta=float(self._loss_huber_delta),
                    miss_weight=float(self._miss_penalty_weight),
                    surface_weight=float(self._surface_weight),
                    surface_sigma=float(self._surface_sigma),
                    outside_weight=float(self._outside_penalty_weight),
                    thin_weight=float(self._thin_weight_eff),
                    thin_max_factor=float(self._thin_max_factor),
                    thickness_reference=float(self._thick_ref),
                    coarse_far_weight=(
                        float(SPARSE_FAR_FIELD_WEIGHT)
                        if use_sample_targets else 0.0),
                    coarse_huber_delta=max(4.0 * float(dx), 0.02),
                )
                value = float(measured.total)
            else:
                value = float("inf")
            best_checkpoint.update(value, state, step=step_index)
            patience_ready = max(
                validation_stable_step,
                last_discontinuity_step + 2 * self._validation_every)
            if step_index >= patience_ready:
                validation_patience.update(value)
            else:
                validation_patience.reset()
            self.validation_history.append((int(step_index), value))
            self.best_validation_loss = float(best_checkpoint.best_loss)
            self.best_validation_step = best_checkpoint.best_step
            return value

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
            self._decode_shape_parameters(
                pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                pred_radii, num_e)

            # SuperFit handles BOTH densification and local fit, so dispatch
            # there whenever either is enabled (local fit can run with
            # densification off).  Plain maintenance only when neither is on.
            self._last_population_lineage = None
            if self._parameterized:
                result = None
            elif self._superfit or self._local_fit_enabled:
                result = self._maybe_superfit(
                    step, pred_centers, pred_radii, pred_rot_flat,
                    pred_eps, pred_bend)
            else:
                result = self._maybe_maintain(
                    step, pred_centers, pred_radii, pred_rot_flat,
                    pred_eps, pred_bend)
            if result is not None:
                last_discontinuity_step = int(step)
                validation_patience.reset()
                old_num_e = int(num_e)
                old_ids = primitive_ids.copy()
                previous_opt_state = _snapshot_opts()
                old_c_np = np.asarray(
                    pred_centers.numpy(), dtype=np.float32).reshape(-1, 3).copy()
                old_r_np = np.asarray(
                    pred_radii.numpy(), dtype=np.float32).reshape(-1, 3).copy()
                old_q_np = np.asarray(
                    pred_rot_flat.numpy(), dtype=np.float32).reshape(-1, 4).copy()
                old_eps_np = np.asarray(
                    pred_eps.numpy(), dtype=np.float32).reshape(-1, 2).copy()
                old_bend_np = np.asarray(
                    pred_bend.numpy(), dtype=np.float32).reshape(-1, 2).copy()
                c_np, r_np, q_np, eps_np, bend_np = result
                row_lineage = getattr(self, "_last_population_lineage", None)
                if row_lineage is None:
                    row_lineage = (np.arange(old_num_e, dtype=np.int64)
                                   if len(c_np) == old_num_e
                                   else np.full(len(c_np), -1, dtype=np.int64))
                row_lineage = np.asarray(row_lineage, dtype=np.int64).reshape(-1)
                if len(row_lineage) != len(c_np):
                    row_lineage = np.full(len(c_np), -1, dtype=np.int64)
                # Sanitise BEFORE the maintained set re-enters global Adam.
                # A non-finite survivor is restored from its lineage row; a
                # non-finite genuinely new row is dropped.  Mapping NaN centres
                # to the origin (the old behaviour) creates a valid-looking but
                # unrelated primitive and is much harder to recover from.
                c_np = np.asarray(c_np, np.float32).reshape(-1, 3).copy()
                r_np = np.asarray(r_np, np.float32).reshape(-1, 3).copy()
                q_np = np.asarray(q_np, np.float32).reshape(-1, 4).copy()
                eps_np = np.asarray(eps_np, np.float32).reshape(-1, 2).copy()
                bend_np = np.asarray(
                    bend_np, np.float32).reshape(-1, 2).copy()
                bad_rows = (
                    ~np.isfinite(c_np).all(axis=1)
                    | ~np.isfinite(r_np).all(axis=1)
                    | ~np.isfinite(q_np).all(axis=1)
                    | ~np.isfinite(eps_np).all(axis=1)
                    | ~np.isfinite(bend_np).all(axis=1)
                    | np.any(r_np <= 0.0, axis=1)
                )
                for bad_row in np.flatnonzero(bad_rows):
                    source = int(row_lineage[bad_row])
                    if 0 <= source < old_num_e:
                        c_np[bad_row] = old_c_np[source]
                        r_np[bad_row] = old_r_np[source]
                        q_np[bad_row] = old_q_np[source]
                        eps_np[bad_row] = old_eps_np[source]
                        bend_np[bad_row] = old_bend_np[source]
                        bad_rows[bad_row] = False
                if np.any(bad_rows):
                    keep = ~bad_rows
                    c_np, r_np, q_np = c_np[keep], r_np[keep], q_np[keep]
                    eps_np, bend_np = eps_np[keep], bend_np[keep]
                    row_lineage = row_lineage[keep]
                if len(c_np) == 0:
                    c_np, r_np, q_np = (
                        old_c_np.copy(), old_r_np.copy(), old_q_np.copy())
                    eps_np, bend_np = old_eps_np.copy(), old_bend_np.copy()
                    row_lineage = np.arange(old_num_e, dtype=np.int64)

                c_np = np.clip(c_np, _c_lo_np, _c_hi_np).astype(np.float32)
                r_np = np.clip(r_np, _r_min, _r_max).astype(np.float32)
                eps_np = np.clip(eps_np, 0.1, 2.0).astype(np.float32)
                rz_np = np.maximum(np.abs(r_np[:, 2:3]), 1.0e-8)
                kappa_np = np.clip(
                    bend_np * rz_np,
                    -self._bend_kappa_max, self._bend_kappa_max)
                bend_np = (kappa_np / rz_np).astype(np.float32)
                _qn = np.linalg.norm(q_np, axis=1)
                q_np[_qn < 1e-6] = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
                q_np /= np.maximum(
                    np.linalg.norm(q_np, axis=1, keepdims=True), 1.0e-12)
                # Sphere: project the maintained set to isotropic + no rotation
                # before it re-enters the optimiser (covers spawn/split/merge and
                # any local fit done inside maintenance).
                r_np, q_np = self._project_isotropic_np(r_np, q_np)
                # Capsule: circular cross-section for the maintained set.
                r_np = self._project_capsule_np(r_np)
                # Maintenance edits the full set (both halves); re-impose the
                # hard-mirror layout so only the source half stays trainable.
                if self._symmetry_enabled and self._sym_axis is not None:
                    c_np, r_np, q_np, eps_np, bend_np = self._build_symmetric_layout(
                        c_np, r_np, q_np, eps_np, bend_np)
                    symmetry_map = np.asarray(
                        getattr(self, "_last_symmetry_lineage",
                                np.arange(len(c_np))),
                        dtype=np.int64,
                    )
                    composed = np.full(len(symmetry_map), -1, dtype=np.int64)
                    valid_symmetry = ((symmetry_map >= 0)
                                      & (symmetry_map < len(row_lineage)))
                    composed[valid_symmetry] = row_lineage[
                        symmetry_map[valid_symmetry]]
                    row_lineage = composed
                optimizer_lineage = row_lineage.copy()
                valid_optimizer_rows = (
                    (optimizer_lineage >= 0)
                    & (optimizer_lineage < old_num_e))
                for new_row in np.flatnonzero(valid_optimizer_rows):
                    old_row = int(optimizer_lineage[new_row])
                    geometry_unchanged = (
                        np.allclose(
                            c_np[new_row], old_c_np[old_row],
                            rtol=1.0e-6, atol=1.0e-7)
                        and np.allclose(
                            r_np[new_row], old_r_np[old_row],
                            rtol=1.0e-6, atol=1.0e-7)
                        and np.allclose(
                            q_np[new_row], old_q_np[old_row],
                            rtol=1.0e-6, atol=1.0e-7)
                        and np.allclose(
                            eps_np[new_row], old_eps_np[old_row],
                            rtol=1.0e-6, atol=1.0e-7)
                        and np.allclose(
                            bend_np[new_row], old_bend_np[old_row],
                            rtol=1.0e-6, atol=1.0e-7)
                    )
                    if not geometry_unchanged:
                        optimizer_lineage[new_row] = -1
                num_e = len(c_np)
                primitive_ids = _advance_ids(old_ids, row_lineage)
                self._primitive_ids = primitive_ids.copy()
                buf = self._alloc_buffers(num_e, bs, train_total, c_np, r_np, q_np,
                                          sdf_target_np=target_values_np
                                          if use_sample_targets else None,
                                          eps_np=eps_np, bend_np=bend_np)
                buf['sdf_target'] = sdf_target
                pred_centers  = buf['pred_centers']
                pred_radii    = buf['pred_radii']
                pred_rot_flat = buf['pred_rot_flat']
                pred_eps_raw  = buf['pred_eps_raw']
                pred_eps      = buf['pred_eps']
                pred_bend_raw = buf['pred_bend_raw']
                pred_bend     = buf['pred_bend']
                # Non-parameterized fits evaluate these arrays directly.  A
                # maintenance pass replaces the population buffers, so refresh
                # the aliases as one atomic population switch.
                fit_centers = pred_centers
                fit_rot_flat = pred_rot_flat
                pred_eps      = buf['pred_eps']
                pred_bend     = buf['pred_bend']
                min_d_cache   = buf['min_d_cache']
                sdf_pred      = buf['sdf_pred']
                loss          = buf['loss']
                wp_indices    = buf['wp_indices']

                (pred_log_radii, opt_c, opt_r, opt_q, opt_eps, opt_bend,
                 prev_centers, grad_c, grad_r, grad_q, grad_eps, grad_bend) = \
                    _make_opts(previous_opt_state, optimizer_lineage)
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
                self._decode_shape_parameters(
                    pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                    pred_radii, num_e)
                _refresh_parameter_world()
                min_d_cache.zero_()
                # Soft-min union only during the densification phase (dense
                # gradients → faster, more complete coverage); switch to the
                # exact HARD min for the refinement phase so the final fit has no
                # soft-union bias (train-soft / eval-hard mismatch).
                use_soft = (self._soft_union
                            and (not use_sample_targets or self._superquadric)
                            and step < self._densify_until_frac * self._num_steps)
                if use_soft:
                    soft_s_cache.zero_()
                    if self._isotropic:
                        wp.launch(
                            _sphere_softmin_kernel_batch,
                            dim=bs,
                            inputs=[fit_centers, pred_radii,
                                    min_d_cache, soft_s_cache, num_e, wp_origin,
                                    float(dx), nx, ny, nz, wp_indices, sdf_pred,
                                    float(self._soft_k(step))],
                            device=device,
                        )
                    elif self._superquadric:
                        if use_sample_targets:
                            wp.launch(
                                _superquadric_softmin_kernel_points,
                                dim=bs,
                                inputs=[fit_centers, pred_radii, fit_rot_flat,
                                        pred_eps, pred_bend,
                                        min_d_cache, soft_s_cache, num_e,
                                        uploaded_samples.points, wp_indices,
                                        sdf_pred, float(self._soft_k(step))],
                                device=device,
                            )
                        else:
                            wp.launch(
                                _superquadric_softmin_kernel_batch,
                                dim=bs,
                                inputs=[fit_centers, pred_radii, fit_rot_flat,
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
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
                                    min_d_cache, soft_s_cache, num_e, wp_origin,
                                    float(dx), nx, ny, nz, wp_indices, sdf_pred,
                                    float(self._soft_k(step))],
                            device=device,
                        )
                    else:
                        wp.launch(
                            _ellipsoid_softmin_kernel_batch,
                            dim=bs,
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
                                    min_d_cache, soft_s_cache, num_e, wp_origin,
                                    float(dx), nx, ny, nz, wp_indices, sdf_pred,
                                    float(self._soft_k(step))],
                            device=device,
                        )
                elif self._isotropic:
                    if use_sample_targets:
                        wp.launch(
                            _sphere_sdf_kernel_points,
                            dim=bs,
                            inputs=[fit_centers, pred_radii,
                                    min_d_cache, num_e, uploaded_samples.points,
                                    wp_indices, sdf_pred],
                            device=device,
                        )
                    else:
                        wp.launch(
                            _sphere_sdf_kernel_batch,
                            dim=bs,
                            inputs=[fit_centers, pred_radii,
                                    min_d_cache, num_e, wp_origin, float(dx),
                                    nx, ny, nz, wp_indices, sdf_pred],
                            device=device,
                        )
                elif self._superquadric:
                    if use_sample_targets:
                        wp.launch(
                            _superquadric_sdf_kernel_points,
                            dim=bs,
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
                                    pred_eps, pred_bend,
                                    min_d_cache, num_e, uploaded_samples.points,
                                    wp_indices, sdf_pred],
                            device=device,
                        )
                    else:
                        wp.launch(
                            _superquadric_sdf_kernel_batch,
                            dim=bs,
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
                                    pred_eps, pred_bend,
                                    min_d_cache, num_e, wp_origin, float(dx),
                                    nx, ny, nz, wp_indices, sdf_pred],
                            device=device,
                        )
                elif self._capsule:
                    if use_sample_targets:
                        wp.launch(
                            _capsule_sdf_kernel_points,
                            dim=bs,
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
                                    min_d_cache, num_e, uploaded_samples.points,
                                    wp_indices, sdf_pred],
                            device=device,
                        )
                    else:
                        wp.launch(
                            _capsule_sdf_kernel_batch,
                            dim=bs,
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
                                    min_d_cache, num_e, wp_origin, float(dx),
                                    nx, ny, nz, wp_indices, sdf_pred],
                            device=device,
                        )
                else:
                    if use_sample_targets:
                        wp.launch(
                            _ellipsoid_sdf_kernel_points,
                            dim=bs,
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
                                    min_d_cache, num_e, uploaded_samples.points,
                                    wp_indices, sdf_pred],
                            device=device,
                        )
                    else:
                        wp.launch(
                            _ellipsoid_sdf_kernel_batch,
                            dim=bs,
                            inputs=[fit_centers, pred_radii, fit_rot_flat,
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
                            uploaded_samples.thickness if use_sample_targets
                            else self._wp_thickness,
                            float(self._thick_ref),
                            float(self._thin_weight_eff), float(self._thin_max_factor),
                            float(self._loss_huber_delta)],
                    device=device,
                )
                if use_sample_targets and sample_targets.coarse_mask is not None:
                    wp.launch(
                        _coarse_far_field_loss_kernel,
                        dim=bs,
                        inputs=[
                            sdf_pred, sdf_target, wp_indices,
                            uploaded_samples.coarse_mask, loss, bs,
                            float(SPARSE_FAR_FIELD_WEIGHT),
                            max(4.0 * float(dx), 0.02),
                        ],
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
                if self._containment_weight > 0.0 and not use_sample_targets:
                    wp.launch(
                        _containment_penalty_kernel,
                        dim=num_e,
                        inputs=[fit_centers, sdf_target, wp_origin, float(dx),
                                nx, ny, nz, loss, num_e,
                                float(self._containment_weight)],
                        device=device,
                    )
                if wp_bone_centers is not None and self._bone_span_weight > 0.0:
                    bone_soft_count.zero_()
                    wp.launch(
                        _bone_membership_kernel,
                        dim=num_e * self._num_bones,
                        inputs=[fit_centers, pred_radii, fit_rot_flat,
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

                if (parameter_anchor_wp is not None
                        and any(weight > 0.0 for weight in self._parameter_regularization)):
                    wp.launch(
                        _parameter_regularization_kernel,
                        dim=num_e,
                        inputs=[
                            pred_centers, pred_log_radii, pred_rot_flat,
                            *parameter_anchor_wp, loss, num_e,
                            *self._parameter_regularization,
                        ],
                        device=device,
                    )
                if (parameter_neighbor_wp is not None
                        and any(weight > 0.0 for weight in self._parameter_neighbor_regularization)):
                    wp.launch(
                        _parameter_regularization_kernel,
                        dim=num_e,
                        inputs=[
                            pred_centers, pred_log_radii, pred_rot_flat,
                            *parameter_neighbor_wp, loss, num_e,
                            *self._parameter_neighbor_regularization,
                        ],
                        device=device,
                    )

            tape.backward(loss)
            # Per-group learning rates (centres / log-radii / rotation).
            opt_c.lr = lr
            opt_r.lr = lr * self._lr_mult_radii
            opt_q.lr = lr * self._lr_mult_rot
            if self._center_step_radius_frac > 0.0:
                wp.launch(_copy_vec3_range, dim=num_e,
                          inputs=[pred_centers, prev_centers, 0],
                          device=device)
            opt_c.step(grad_c)
            opt_r.step(grad_r)
            opt_q.step(grad_q)
            if self._eps_is_trainable(step):
                opt_eps.lr = lr * self._sq_eps_lr_mult
                opt_eps.step(grad_eps)
            if self._bend_is_trainable(step):
                opt_bend.lr = lr * self._sq_bend_lr_mult
                opt_bend.step(grad_bend)
            tape.zero()

            if self._center_step_radius_frac > 0.0:
                wp.launch(
                    _limit_center_step_by_radius,
                    dim=num_e,
                    inputs=[
                        pred_centers, prev_centers, pred_radii,
                        float(self._center_step_radius_frac),
                        float(self._center_step_min_vox) * float(dx),
                        float(self._center_step_max_vox) * float(dx),
                        0,
                    ],
                    device=device)

            if parameter_anchor_wp is not None and (
                    self._parameter_center_trust_radius_factor > 0.0
                    or self._parameter_log_radius_limit > 0.0):
                wp.launch(
                    _project_parameter_trust_region_kernel,
                    dim=num_e,
                    inputs=[
                        pred_centers, pred_log_radii,
                        parameter_anchor_wp[0], parameter_anchor_wp[1],
                        parameter_anchor_wp[3],
                        float(self._parameter_center_trust_radius_factor),
                        float(self._parameter_log_radius_limit),
                    ],
                    device=device,
                )

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
                                               pred_rot_flat,
                                               pred_eps_raw
                                               if self._sq_eps_mode == "per_primitive"
                                               else None,
                                               pred_bend_raw if self._bent else None)

            # Sphere: project to isotropic radii + identity rotation each step.
            self._project_isotropic(pred_log_radii, pred_rot_flat, num_e)
            # Capsule: keep the cross-section circular (r1 = r0) each step.
            self._project_capsule(pred_log_radii, num_e)
            # Superquadric: keep per-primitive roundness in a safe range, and
            # bound centres + (log-)radii so the harsher SQ/bend gradients can't
            # drive a primitive to inf (a generous safety net).
            if self._superquadric:
                wp.launch(_clamp_log_radii, dim=num_e,
                          inputs=[pred_log_radii, _log_rmin, _log_rmax],
                          device=device)
                wp.launch(_clamp_centers_range, dim=num_e,
                          inputs=[pred_centers, _c_lo, _c_hi, 0], device=device)

            report_due = (step % self._report_every == 0)
            validation_due = (
                step % self._validation_every == 0
                or step == self._num_steps - 1)
            if report_due or validation_due:
                # Refresh all derived world parameters once for reporting and
                # deterministic hold-out evaluation.
                wp.launch(_exp_radii_kernel, dim=num_e,
                          inputs=[pred_log_radii, pred_radii], device=device)
                self._decode_shape_parameters(
                    pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                    pred_radii, num_e)
                _refresh_parameter_world()

            stop_for_patience = False
            if validation_due:
                _record_validation(step)
                stop_for_patience = (
                    step >= max(
                        validation_stable_step,
                        last_discontinuity_step + 2 * self._validation_every)
                    and validation_patience.should_stop)

            if report_due:
                self._emit_progress(step, loss, fit_centers, pred_radii,
                                    fit_rot_flat, num_e, origin, dx, n,
                                    pred_eps=(pred_eps
                                              if self._superquadric else None),
                                    pred_bend=pred_bend if self._bent else None)

                wp.synchronize_device(device)
                loss_val = float(loss.numpy()[0])
                if loss_val < 1e-10:
                    break
            if stop_for_patience:
                break

        # Always publish the best deterministic hold-out checkpoint, not merely
        # the last stochastic mini-batch state.  This also provides a clean
        # rollback if a late SQ/bend update becomes non-finite or overfits.
        final_step = int(step if "step" in locals() else self._num_steps)
        if best_checkpoint.has_checkpoint:
            best_state = best_checkpoint.restore()
            best_num = int(len(best_state["radii"]))
            best_centers_wp = wp.array(
                np.ascontiguousarray(best_state["world_centers"]),
                dtype=wp.vec3, device=device)
            best_radii_wp = wp.array(
                np.ascontiguousarray(best_state["radii"]),
                dtype=wp.vec3, device=device)
            best_rot_wp = wp.array(
                np.ascontiguousarray(best_state["world_rotations"].reshape(-1)),
                dtype=wp.float32, device=device)
            best_eps_wp = None
            best_bend_wp = None
            if self._superquadric:
                best_eps_wp = wp.array(
                    np.ascontiguousarray(best_state["eps"].reshape(-1)),
                    dtype=wp.float32, device=device)
                if self._bent:
                    best_bend_wp = wp.array(
                        np.ascontiguousarray(best_state["bend"].reshape(-1)),
                        dtype=wp.float32, device=device)
            best_loss_wp = wp.array(
                np.array([best_checkpoint.best_loss], dtype=np.float32),
                dtype=wp.float32, device=device)
            self._primitive_ids = np.asarray(
                best_state["primitive_ids"], dtype=np.int64).copy()
            symmetry_partition = np.asarray(
                best_state["symmetry_partition"], dtype=np.int64).reshape(2)
            self._sym_n_op = int(symmetry_partition[0])
            self._sym_n_so = int(symmetry_partition[1])
            best_step = (
                final_step if best_checkpoint.best_step is None
                else int(best_checkpoint.best_step))
            self._emit_progress(
                best_step, best_loss_wp, best_centers_wp, best_radii_wp,
                best_rot_wp, best_num, origin, dx, n,
                pred_eps=best_eps_wp, pred_bend=best_bend_wp)
            if self._parameterized:
                self.optimized_parameter_result = (
                    np.asarray(best_state["centers"], dtype=np.float32).copy(),
                    np.asarray(best_state["radii"], dtype=np.float32).copy(),
                    np.asarray(best_state["rotations"], dtype=np.float32).copy(),
                )
        else:
            wp.launch(_exp_radii_kernel, dim=num_e,
                      inputs=[pred_log_radii, pred_radii], device=device)
            self._decode_shape_parameters(
                pred_eps_raw, pred_eps, pred_bend_raw, pred_bend,
                pred_radii, num_e)
            _refresh_parameter_world()
            self._emit_progress(
                final_step, loss, fit_centers, pred_radii, fit_rot_flat,
                num_e, origin, dx, n,
                pred_eps=pred_eps if self._superquadric else None,
                pred_bend=pred_bend if self._bent else None)
            if self._parameterized:
                wp.synchronize_device(device)
                self.optimized_parameter_result = (
                    np.asarray(pred_centers.numpy(), dtype=np.float32).reshape(-1, 3),
                    np.asarray(pred_radii.numpy(), dtype=np.float32).reshape(-1, 3),
                    np.asarray(pred_rot_flat.numpy(), dtype=np.float32).reshape(-1, 4),
                )


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
