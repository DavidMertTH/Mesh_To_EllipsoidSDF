"""
batched_fit.py — fit MANY bone regions simultaneously in ONE GPU loop.

Motivation
----------
Bone-Separation gives every bone its *own* isolated SDF and a small ellipsoid
budget.  Fitting them one after another (the old sequential controller) wastes
the GPU: a 2-ellipsoid finger fills a tiny fraction of the device per step, yet
the whole pipeline pays the per-bone launch/overhead serially.  The cure is to
train *all* bones in a single Adam loop — each bone's ellipsoids are optimised
against its own SDF, but every step issues ONE batched launch that covers all
bones at once, so the GPU runs full and the wall-clock cost collapses toward the
cost of a *single* fit.

Isolation without a shared grid
-------------------------------
Bones overlap in world space (the seam collars are deliberately shared), so a
single world-space SDF would merge neighbours.  Instead each bone contributes a
*pool* of world-space sample points carrying its own region-SDF value.  A
per-sample ``group`` id and a per-ellipsoid ``group`` id make the union kernel
evaluate each sample only against the ellipsoids of *its* bone — so the fits stay
perfectly isolated even though all bones share one launch and one optimiser.

Scope (v1)
----------
This engine does the *global Adam* fit with a fixed per-bone budget (the
area-allocated ``budget`` from ``partition_mesh_by_bone``).  It deliberately does
NOT run the single-bone optimiser's maintenance (prune/spawn/split/merge/local
fit) — those are inherently per-grid and would reintroduce the serial cost.
Symmetry is handled by the *controller* around this engine: paired bones are
mirrored from their source, on-plane (centre) bones are symmetrised after the
fit (:func:`symmetrize_ellipsoids`).

The Warp kernels here mirror the proven analytic ellipsoid SDF used by
``optimization.py`` (MertStein hybrid: scaled-sphere inside, Quílez outside).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp
import warp.optim
from PySide6 import QtCore


# ── Warp kernels ────────────────────────────────────────────────────────────

@wp.func
def _soft_clamp(x: float, limit: float) -> float:
    return limit * wp.tanh(x / limit)


@wp.kernel
def _exp_radii_kernel(
    log_radii: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    lr = log_radii[i]
    radii[i] = wp.vec3(wp.exp(lr[0]), wp.exp(lr[1]), wp.exp(lr[2]))


@wp.kernel
def _normalize_flat_quats(rot_flat: wp.array(dtype=wp.float32)):
    i = wp.tid()
    base = i * 4
    q = wp.normalize(wp.quat(
        rot_flat[base + 0], rot_flat[base + 1],
        rot_flat[base + 2], rot_flat[base + 3]))
    rot_flat[base + 0] = q[0]
    rot_flat[base + 1] = q[1]
    rot_flat[base + 2] = q[2]
    rot_flat[base + 3] = q[3]


@wp.kernel
def _grouped_union_kernel(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    ell_start: wp.array(dtype=wp.int32),     # per-group first ellipsoid index
    ell_count: wp.array(dtype=wp.int32),     # per-group ellipsoid count
    points: wp.array(dtype=wp.vec3),         # all pool points, concatenated
    sample_idx: wp.array(dtype=wp.int32),    # batch → global pool-point index
    sample_grp: wp.array(dtype=wp.int32),    # batch → group id
    min_d: wp.array2d(dtype=wp.float32),     # scratch scan, (batch × max_count+1)
    max_count: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    # One thread per sample.  A sample only ever sees the ellipsoids of its own
    # bone group (ell_start[g] .. ell_start[g]+ell_count[g]), so the per-bone
    # fits stay isolated despite sharing one launch.  Loop runs a fixed
    # ``max_count`` and masks inactive slots (d = +inf) → static loop bound, clean
    # autodiff, identical result to a ragged loop.
    bid = wp.tid()
    g = sample_grp[bid]
    p = points[sample_idx[bid]]
    s = ell_start[g]
    cnt = ell_count[g]

    min_d[bid, 0] = 1.0e6
    for k in range(max_count):
        d = float(1.0e6)
        if k < cnt:
            i = s + k
            base = i * 4
            q = wp.normalize(wp.quat(
                rot_flat[base + 0], rot_flat[base + 1],
                rot_flat[base + 2], rot_flat[base + 3]))
            local_p = wp.quat_rotate_inv(q, p - centers[i])
            r = radii[i]
            scaled = wp.vec3(local_p[0] / r[0], local_p[1] / r[1],
                             local_p[2] / r[2])
            k0 = wp.length(scaled)
            if k0 < 1.0:
                r_min = wp.min(wp.min(r[0], r[1]), r[2])
                d = (k0 - 1.0) * r_min
            else:
                scaled2 = wp.vec3(local_p[0] / (r[0] * r[0]),
                                  local_p[1] / (r[1] * r[1]),
                                  local_p[2] / (r[2] * r[2]))
                k1_safe = wp.max(wp.length(scaled2), 1.0e-8)
                d = k0 * (k0 - 1.0) / k1_safe
        min_d[bid, k + 1] = wp.min(min_d[bid, k], d)

    out_sdf[bid] = min_d[bid, max_count]


@wp.kernel
def _grouped_loss_kernel(
    sdf_pred: wp.array(dtype=wp.float32),
    sample_idx: wp.array(dtype=wp.int32),
    sample_grp: wp.array(dtype=wp.int32),
    target_pool: wp.array(dtype=wp.float32),
    thick_pool: wp.array(dtype=wp.float32),
    grp_sigma: wp.array(dtype=wp.float32),      # per-group surface sigma (world)
    grp_thickref: wp.array(dtype=wp.float32),   # per-group thickness reference
    loss: wp.array(dtype=wp.float32),
    inv_batch: float,
    miss_weight: float,
    surface_weight: float,
    outside_weight: float,
    thin_weight: float,
    thin_max_factor: float,
):
    # Surface- and thinness-weighted SDF reconstruction loss with miss /
    # protrusion penalties — same formulation as optimization._rmse_loss_kernel
    # but the target/thickness come from the per-sample pool (no grid decode).
    bid = wp.tid()
    g = sample_grp[bid]
    pidx = sample_idx[bid]
    limit = float(0.1)

    sp = wp.clamp(sdf_pred[bid], -10.0, 10.0)
    t = target_pool[pidx]

    sigma = grp_sigma[g]
    sw = 1.0 + surface_weight * wp.exp(-(t * t) / (sigma * sigma))

    tw = float(1.0)
    th = thick_pool[pidx]
    thref = grp_thickref[g]
    if thin_weight > 0.0 and th > 0.0 and thref > 0.0:
        boost = thref / th - 1.0
        if boost < 0.0:
            boost = 0.0
        tw = 1.0 + thin_weight * boost
        if tw > thin_max_factor:
            tw = thin_max_factor
    w = sw * tw

    diff = wp.abs(_soft_clamp(sp, limit) - _soft_clamp(t, limit))
    wp.atomic_add(loss, 0, w * diff * inv_batch)

    if t < 0.0 and sp > 0.0:
        wp.atomic_add(loss, 0, w * miss_weight * (sp - t) * inv_batch)
    if t > 0.0 and sp < 0.0:
        wp.atomic_add(loss, 0, w * outside_weight * (t - sp) * inv_batch)


@wp.kernel
def _flatness_penalty_kernel(
    radii: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
    count: int,
    flat_weight: float,
    flat_min_ratio: float,
):
    # Penalise collapsed flat *disks* (rmin ≪ rmid) while allowing slender
    # needles (rmin ≈ rmid).  Scale-invariant.  Same as optimization.py.
    e = wp.tid()
    r = radii[e]
    a = r[0]
    b = r[1]
    c = r[2]
    rmax = wp.max(a, wp.max(b, c))
    rmin = wp.min(a, wp.min(b, c))
    rmid = wp.max(a + b + c - rmax - rmin, 1.0e-9)
    pen = float(0.0)
    d = flat_min_ratio - rmin / rmid
    if d > 0.0:
        pen = d * d
    wp.atomic_add(loss, 0, flat_weight * pen / float(count))


@wp.kernel
def _clamp_log_radii_arr(log_radii: wp.array(dtype=wp.vec3),
                         lo: wp.array(dtype=wp.float32),
                         hi: wp.array(dtype=wp.float32)):
    # Per-ellipsoid log-radius bounds.  Each bone gets bounds derived from ITS
    # OWN region extent, so a small bone's ellipsoid can never grow to the whole
    # skeleton's size (a global bound was the "huge ellipsoid" bug).
    i = wp.tid()
    r = log_radii[i]
    a = lo[i]
    b = hi[i]
    log_radii[i] = wp.vec3(wp.clamp(r[0], a, b),
                           wp.clamp(r[1], a, b),
                           wp.clamp(r[2], a, b))


@wp.kernel
def _clamp_centers_arr(centers: wp.array(dtype=wp.vec3),
                       lo: wp.array(dtype=wp.vec3),
                       hi: wp.array(dtype=wp.vec3)):
    # Per-ellipsoid centre bounds (its bone's region bbox + pad), so an ellipsoid
    # cannot drift across the body into a neighbouring bone's region.
    i = wp.tid()
    c = centers[i]
    a = lo[i]
    b = hi[i]
    centers[i] = wp.vec3(wp.clamp(c[0], a[0], b[0]),
                         wp.clamp(c[1], a[1], b[1]),
                         wp.clamp(c[2], a[2], b[2]))


# ── per-bone job ────────────────────────────────────────────────────────────

@dataclass
class BoneFitJob:
    """Everything the batched engine needs to fit one bone region.

    Pool points/targets/thickness are in *world* space (so groups can be unioned
    in one launch).  ``init_*`` is the starting ellipsoid layout (FPS interior
    placement) and ``budget`` its fixed count.  ``symmetry`` is the plane
    ``(axis, coord)`` for an on-plane (centre) bone whose *result* should be
    symmetrised, else ``None`` (the controller mirrors paired bones itself).
    """

    bone_index: int
    pool_points: np.ndarray     # (P, 3) float32 world sample points
    pool_target: np.ndarray     # (P,)   float32 region SDF at each point
    pool_thick: np.ndarray      # (P,)   float32 local feature thickness (>=0)
    band_local: np.ndarray      # (B,)   int32 indices into the pool (surface band)
    init_centers: np.ndarray    # (E, 3) float32
    init_radii: np.ndarray      # (E, 3) float32
    init_rots: np.ndarray       # (E, 4) float32 (x, y, z, w)
    budget: int
    dx: float                   # voxel size of this bone's grid (world units)
    symmetry: tuple[int, float] | None = None


# ── job factory (numpy, runs on the GUI/host thread) ─────────────────────────

def _voxel_world_positions(flat_idx: np.ndarray, nx: int, ny: int, nz: int,
                           origin: np.ndarray, dx: float) -> np.ndarray:
    """Map C-order flat grid indices (iz·ny·nx + iy·nx + ix) → world centres."""
    ix = flat_idx % nx
    iy = (flat_idx // nx) % ny
    iz = flat_idx // (nx * ny)
    g = np.stack([ix, iy, iz], axis=1).astype(np.float32)
    return (origin.astype(np.float32) + (g + 0.5) * np.float32(dx)).astype(np.float32)


def _fps_init(pool_pts: np.ndarray, pool_depth: np.ndarray, num_e: int,
              dx: float, rng: np.random.Generator) -> tuple:
    """Farthest-point init: spread ``num_e`` centres over deep interior points.

    ``pool_pts`` are interior world points, ``pool_depth`` their |SDF| (depth).
    Radii start proportional to local depth; rotations are identity.
    """
    num_e = int(max(1, num_e))
    if len(pool_pts) == 0:
        centers = (rng.random((num_e, 3)).astype(np.float32) - 0.5) * float(dx)
        radii = np.full((num_e, 3), max(dx, 1e-4), dtype=np.float32)
        rots = np.tile(np.array([0, 0, 0, 1], np.float32), (num_e, 1))
        return centers, radii, rots

    # Greedy farthest-point sampling, seeded at the deepest interior point.
    n = len(pool_pts)
    take = min(num_e, n)
    chosen = np.empty(take, dtype=np.int64)
    chosen[0] = int(np.argmax(pool_depth))
    d2 = ((pool_pts - pool_pts[chosen[0]]) ** 2).sum(axis=1)
    for i in range(1, take):
        chosen[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, ((pool_pts - pool_pts[chosen[i]]) ** 2).sum(axis=1))
    sel = pool_pts[chosen]
    depth = np.maximum(pool_depth[chosen], 0.5 * dx)
    # If fewer interior points than budget, pad by jittering existing centres.
    if take < num_e:
        extra = num_e - take
        pad_idx = rng.integers(0, take, size=extra)
        jitter = (rng.random((extra, 3)).astype(np.float32) - 0.5) * float(dx)
        sel = np.vstack([sel, sel[pad_idx] + jitter])
        depth = np.concatenate([depth, depth[pad_idx]])
    centers = sel.astype(np.float32)
    r = np.clip(depth, 0.5 * dx, None).astype(np.float32)
    radii = np.repeat(r[:, None], 3, axis=1) * np.float32(0.8)
    rots = np.tile(np.array([0, 0, 0, 1], np.float32), (num_e, 1))
    return centers, radii.astype(np.float32), rots


def make_bone_fit_job(
    bone_index: int,
    grid: np.ndarray,            # (nz, ny, nx) region SDF
    origin: np.ndarray,          # (3,) world corner
    dx: float,
    nx: int, ny: int, nz: int,
    budget: int,
    thickness: np.ndarray | None = None,
    symmetry: tuple[int, float] | None = None,
    pool_size: int = 20000,
    surface_band_vox: float = 3.0,
    rng: np.random.Generator | None = None,
) -> BoneFitJob:
    """Build a :class:`BoneFitJob` from a bone region's voxel SDF.

    Samples a bounded *pool* of world points (all interior + surface band +
    some exterior, subsampled to ``pool_size``), records each point's target SDF
    and local thickness, marks the surface-band subset, and FPS-initialises the
    ellipsoid layout from the interior.
    """
    rng = rng or np.random.default_rng()
    flat = np.asarray(grid, dtype=np.float32).ravel()
    flat_th = (np.asarray(thickness, dtype=np.float32).ravel()
               if thickness is not None else np.zeros_like(flat))
    band = float(surface_band_vox) * float(dx)

    interior = np.where(flat < 0.0)[0]
    surface = np.where(np.abs(flat) < band)[0]
    exterior = np.where(flat >= band)[0]

    # Compose the pool.  RESERVE a slice for exterior points up front so the fit
    # always has outside/protrusion gradients: without exterior samples an
    # ellipsoid can grow past the surface unpenalised (a cause of runaway "huge"
    # ellipsoids on bones whose interior+surface alone fills the pool).
    ext_budget = int(0.25 * pool_size)
    n_ext = min(len(exterior), ext_budget)
    ext_sel = (rng.choice(exterior, size=n_ext, replace=False)
               if n_ext > 0 else np.empty(0, dtype=np.int64))

    # Core = ALL surface (the signal that matters) + as much interior as fits.
    core_cap = max(0, pool_size - n_ext)
    keep_surf = surface.astype(np.int64)
    room = core_cap - len(keep_surf)
    int_only = np.setdiff1d(interior.astype(np.int64), keep_surf,
                            assume_unique=False)
    if room <= 0:
        int_only = np.empty(0, dtype=np.int64)
    elif len(int_only) > room:
        int_only = rng.choice(int_only, size=room, replace=False)
    core = np.unique(np.concatenate([keep_surf, int_only]))
    pool_flat = np.unique(
        np.concatenate([core, ext_sel.astype(np.int64)])).astype(np.int64)

    pool_points = _voxel_world_positions(pool_flat, nx, ny, nz, origin, dx)
    pool_target = flat[pool_flat].astype(np.float32)
    pool_thick = flat_th[pool_flat].astype(np.float32)
    band_mask = np.abs(pool_target) < band
    band_local = np.where(band_mask)[0].astype(np.int32)
    if band_local.size == 0:
        band_local = np.arange(len(pool_flat), dtype=np.int32)

    # FPS init from interior pool points (depth = |SDF|).
    int_mask = pool_target < 0.0
    init_c, init_r, init_q = _fps_init(
        pool_points[int_mask], np.abs(pool_target[int_mask]),
        int(budget), float(dx), rng)

    return BoneFitJob(
        bone_index=int(bone_index),
        pool_points=pool_points,
        pool_target=pool_target,
        pool_thick=pool_thick,
        band_local=band_local,
        init_centers=init_c,
        init_radii=init_r,
        init_rots=init_q,
        budget=int(budget),
        dx=float(dx),
        symmetry=symmetry,
    )


# ── result symmetrisation (numpy) ────────────────────────────────────────────

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
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
    q = np.array([x, y, z, w], np.float64)
    return (q / np.linalg.norm(q)).astype(np.float32)


def mirror_quats(q: np.ndarray, axis: int) -> np.ndarray:
    """Closed-form quaternion mirror across the plane normal to ``axis``.

    Reflecting negates the two vector components perpendicular to ``axis`` (the
    axis component and w stay).  Involutive; matches optimization._mirror_quats.
    """
    out = np.asarray(q, np.float32).reshape(-1, 4).copy()
    perp = [i for i in range(3) if i != axis]
    out[:, perp] *= -1.0
    return out


def reflect_ellipsoids(c: np.ndarray, r: np.ndarray, q: np.ndarray,
                       axis: int, coord: float) -> tuple:
    """Reflect an ellipsoid set across the plane ``axis = coord`` (for a mirror
    partner bone): centre's axis-coordinate flips, radii unchanged, orientation
    mirrored."""
    cm = np.asarray(c, np.float32).reshape(-1, 3).copy()
    cm[:, axis] = np.float32(2.0 * coord) - cm[:, axis]
    return (cm.astype(np.float32),
            np.asarray(r, np.float32).reshape(-1, 3).copy(),
            mirror_quats(q, axis))


def symmetrize_ellipsoids(c: np.ndarray, r: np.ndarray, q: np.ndarray,
                          axis: int, coord: float) -> tuple:
    """Make an on-plane (centre) bone's fitted set mirror-symmetric.

    Each ellipsoid is paired with the nearest ellipsoid of the *reflected* set;
    the pair is averaged (centre, radii) and its orientation set to the mirror
    mean, so the result is invariant under the symmetry reflection.  This
    realises "symmetric bones are fitted symmetrically" without per-step device
    work — the unconstrained fit is snapped onto the symmetric manifold once.
    """
    c = np.asarray(c, np.float32).reshape(-1, 3)
    r = np.asarray(r, np.float32).reshape(-1, 3)
    q = np.asarray(q, np.float32).reshape(-1, 4)
    n = len(c)
    if n == 0:
        return c, r, q
    cr, rr, qr = reflect_ellipsoids(c, r, q, axis, coord)
    # Nearest reflected partner for each ellipsoid (greedy, symmetric enough).
    D = np.linalg.norm(c[:, None, :] - cr[None, :, :], axis=2)
    partner = D.argmin(axis=1)
    out_c = np.empty_like(c)
    out_r = np.empty_like(r)
    out_q = np.empty_like(q)
    for i in range(n):
        j = int(partner[i])
        out_c[i] = 0.5 * (c[i] + cr[j])
        out_r[i] = 0.5 * (r[i] + rr[j])
        # mirror-mean orientation: nlerp of q[i] and its mirror
        qi = q[i].astype(np.float64)
        qmi = mirror_quats(q[i:i + 1], axis)[0].astype(np.float64)
        if np.dot(qi, qmi) < 0.0:
            qmi = -qmi
        s = qi + qmi
        nrm = np.linalg.norm(s)
        out_q[i] = (qmi if nrm < 1e-9 else s / nrm).astype(np.float32)
    return out_c.astype(np.float32), out_r.astype(np.float32), out_q.astype(np.float32)


# ── batched worker ──────────────────────────────────────────────────────────

class BatchedFitWorker(QtCore.QThread):
    """Fit all bone groups in ONE Adam loop (each against its own SDF pool).

    Emits ``step_visual(step, loss, centers, radii, rots)`` periodically with the
    *combined* world-space ellipsoids of every group (source/centre bones only —
    mirror partners are derived by the controller after finishing).  On
    completion ``results`` holds ``[(bone_index, centers, radii, rots), ...]``
    and ``fit_finished`` is emitted.

    Note: the completion signal is named ``fit_finished`` (not ``finished``) to
    avoid shadowing :class:`QtCore.QThread`'s built-in ``finished`` signal.
    """

    step_visual = QtCore.Signal(int, float, object, object, object)
    prep_progress = QtCore.Signal(float, str)
    fit_finished = QtCore.Signal()

    def __init__(
        self,
        jobs: list[BoneFitJob],
        *,
        device: str,
        num_steps: int = 1200,
        report_every: int = 20,
        sample_budget: int = 49152,
        surface_fraction: float = 0.75,
        lr_init: float = 0.01,
        lr_final: float = 0.0002,
        lr_decay_k: float = 7.0,
        lr_mult_radii: float = 2.0,
        lr_mult_rot: float = 1.0,
        miss_penalty_weight: float = 3.0,
        outside_penalty_weight: float = 14.0,
        surface_weight: float = 4.0,
        surface_sigma_vox: float = 1.5,
        flat_weight: float = 0.5,
        flat_min_ratio: float = 0.35,
        thin_loss_weight: float = 1.0,
        thin_max_factor: float = 6.0,
        rng: np.random.Generator | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._device = device
        self._num_steps = int(num_steps)
        self._report_every = max(1, int(report_every))
        self._sample_budget = int(sample_budget)
        self._surface_fraction = float(np.clip(surface_fraction, 0.0, 1.0))
        self._lr_init = float(lr_init)
        self._lr_final = float(lr_final)
        self._lr_decay_k = float(lr_decay_k)
        self._lr_mult_radii = float(lr_mult_radii)
        self._lr_mult_rot = float(lr_mult_rot)
        self._miss = float(miss_penalty_weight)
        self._outside = float(outside_penalty_weight)
        self._surface_weight = float(surface_weight)
        self._surface_sigma_vox = float(surface_sigma_vox)
        self._flat_weight = float(flat_weight)
        self._flat_min_ratio = float(flat_min_ratio)
        self._thin_weight = float(thin_loss_weight)
        self._thin_max_factor = float(thin_max_factor)
        self._rng = rng or np.random.default_rng()

        self._stop_flag = False
        self.results: list[tuple] = []     # [(bone_index, c, r, q), ...]

    def request_stop(self):
        self._stop_flag = True

    def _lr_at(self, step: int) -> float:
        frac = step / max(1, self._num_steps - 1)
        return float(self._lr_final + (self._lr_init - self._lr_final)
                     * np.exp(-self._lr_decay_k * frac))

    # ── per-step sampler plan ────────────────────────────────────────────
    def _plan_samples(self):
        """Fixed per-group sample counts + static layout arrays.

        Distributes ``sample_budget`` across groups proportional to ellipsoid
        budget (floor 256), then precomputes the static ``sample_grp`` ids and
        per-group pool offsets / band index pools used every step.
        """
        budgets = np.array([max(1, j.budget) for j in self._jobs], np.float64)
        share = budgets / budgets.sum()
        m = np.maximum(256, np.round(share * self._sample_budget)).astype(int)
        n_surf = (m * self._surface_fraction).astype(int)
        n_rest = m - n_surf

        pool_sizes = [len(j.pool_points) for j in self._jobs]
        offsets = np.concatenate([[0], np.cumsum(pool_sizes)]).astype(np.int64)
        sample_grp = np.concatenate(
            [np.full(int(m[g]), g, np.int32) for g in range(len(self._jobs))])
        return m, n_surf, n_rest, offsets, sample_grp

    def run(self):
        device = self._device
        jobs = self._jobs
        if not jobs:
            self.fit_finished.emit()
            return
        try:
            self._run(device, jobs)
        finally:
            self.fit_finished.emit()

    def _run(self, device, jobs):
        self.prep_progress.emit(0.1, "packing bones")
        G = len(jobs)

        # ── concatenate per-group ellipsoid params ──
        counts = np.array([len(j.init_centers) for j in jobs], np.int32)
        max_count = int(counts.max())
        ell_start_np = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        total_e = int(ell_start_np[-1])
        ell_start = wp.array(ell_start_np[:-1], dtype=wp.int32, device=device)
        ell_count = wp.array(counts, dtype=wp.int32, device=device)

        c0 = np.vstack([j.init_centers for j in jobs]).astype(np.float32)
        r0 = np.vstack([j.init_radii for j in jobs]).astype(np.float32)
        q0 = np.vstack([j.init_rots for j in jobs]).astype(np.float32).reshape(-1)

        pred_centers = wp.array(c0, dtype=wp.vec3, device=device, requires_grad=True)
        pred_radii = wp.array(r0, dtype=wp.vec3, device=device, requires_grad=True)
        pred_log_radii = wp.array(np.log(np.maximum(r0, 1e-6)), dtype=wp.vec3,
                                  device=device, requires_grad=True)
        pred_rot_flat = wp.array(q0, dtype=wp.float32, device=device,
                                 requires_grad=True)

        # ── concatenate per-group pools ──
        self.prep_progress.emit(0.4, "uploading pools")
        all_points = np.vstack([j.pool_points for j in jobs]).astype(np.float32)
        all_target = np.concatenate([j.pool_target for j in jobs]).astype(np.float32)
        all_thick = np.concatenate([j.pool_thick for j in jobs]).astype(np.float32)
        wp_points = wp.array(all_points, dtype=wp.vec3, device=device)
        wp_target = wp.array(all_target, dtype=wp.float32, device=device)
        wp_thick = wp.array(all_thick, dtype=wp.float32, device=device)

        grp_sigma_np = np.array(
            [max(1e-6, self._surface_sigma_vox * j.dx) for j in jobs], np.float32)
        grp_sigma = wp.array(grp_sigma_np, dtype=wp.float32, device=device)
        # Per-group thickness reference = median of positive thickness in its pool.
        thref_np = np.empty(G, np.float32)
        for g, j in enumerate(jobs):
            pos = j.pool_thick[j.pool_thick > 0.0]
            thref_np[g] = float(np.median(pos)) if pos.size else 0.0
        grp_thickref = wp.array(thref_np, dtype=wp.float32, device=device)

        # ── per-step sample plan ──
        m, n_surf, n_rest, pool_off, sample_grp_np = self._plan_samples()
        batch = int(sample_grp_np.size)
        sample_grp = wp.array(sample_grp_np, dtype=wp.int32, device=device)
        # band index pools (global pool indices) + all-index pools per group
        band_pools = [pool_off[g] + jobs[g].band_local for g in range(G)]
        all_pools = [pool_off[g] + np.arange(len(jobs[g].pool_points), dtype=np.int64)
                     for g in range(G)]

        # ── scratch + bounds ──
        min_d = wp.zeros((batch, max_count + 1), dtype=wp.float32,
                         device=device, requires_grad=True)
        sdf_pred = wp.zeros(batch, dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        sample_idx = wp.zeros(batch, dtype=wp.int32, device=device)

        # Per-group parameter bounds, expanded to one entry per ellipsoid.
        # Crucially the bounds are PER BONE and derived from the bone BODY only —
        # NOT from ``pool_points`` (which deliberately includes exterior/margin
        # samples reaching the grid edge).  Binding to the full padded pool box
        # let a single semi-axis equal the whole bone+margin extent, so ellipsoids
        # ballooned to ~2× the bone and stuck far out.  We instead measure the
        # bbox of the interior+surface points (``target <= band``), cap the radius
        # semi-axis to HALF that extent (one ellipsoid can at most span the bone,
        # never exceed it), and pad centres by only a few voxels.
        log_rmin_e = np.empty(total_e, np.float32)
        log_rmax_e = np.empty(total_e, np.float32)
        c_lo_e = np.empty((total_e, 3), np.float32)
        c_hi_e = np.empty((total_e, 3), np.float32)
        for g, j in enumerate(jobs):
            a, b = int(ell_start_np[g]), int(ell_start_np[g + 1])
            band_g = 3.0 * float(j.dx)               # matches make_bone_fit_job
            body_mask = j.pool_target <= band_g      # interior + surface shell
            body = j.pool_points[body_mask]
            if len(body) < 2:                        # degenerate → fall back
                body = j.pool_points
            plo = body.min(axis=0)
            phi = body.max(axis=0)
            ext = float(np.max(phi - plo))
            pad_g = 2.0 * float(j.dx)                # tight, voxel-scaled pad
            log_rmin_e[a:b] = np.log(max(1e-6, 0.25 * j.dx))
            log_rmax_e[a:b] = np.log(max(1e-4, 0.5 * ext))
            c_lo_e[a:b] = plo - pad_g
            c_hi_e[a:b] = phi + pad_g
        wp_log_rmin = wp.array(log_rmin_e, dtype=wp.float32, device=device)
        wp_log_rmax = wp.array(log_rmax_e, dtype=wp.float32, device=device)
        wp_c_lo = wp.array(c_lo_e, dtype=wp.vec3, device=device)
        wp_c_hi = wp.array(c_hi_e, dtype=wp.vec3, device=device)

        # ── optimisers (separate LR per param group) ──
        self.prep_progress.emit(0.7, "optimizer")
        opt_c = wp.optim.Adam([pred_centers], lr=self._lr_init)
        opt_r = wp.optim.Adam([pred_log_radii], lr=self._lr_init)
        opt_q = wp.optim.Adam([pred_rot_flat], lr=self._lr_init)
        grad_c = [pred_centers.grad.flatten()]
        grad_r = [pred_log_radii.grad.flatten()]
        grad_q = [pred_rot_flat.grad.flatten()]

        inv_batch = 1.0 / float(batch)
        self.prep_progress.emit(1.0, "starting")

        for step in range(self._num_steps):
            if self._stop_flag:
                break
            lr = self._lr_at(step)

            # draw a fresh mini-batch per group → global pool indices
            parts = []
            for g in range(G):
                bp = band_pools[g]
                ap = all_pools[g]
                ns = int(n_surf[g])
                nr = int(n_rest[g])
                if ns > 0:
                    parts.append(self._rng.choice(bp if bp.size else ap,
                                                  size=ns, replace=True))
                if nr > 0:
                    parts.append(self._rng.choice(ap, size=nr, replace=True))
            idx_np = np.concatenate(parts).astype(np.int32)
            sample_idx.assign(idx_np)

            tape = wp.Tape()
            with tape:
                wp.launch(_exp_radii_kernel, dim=total_e,
                          inputs=[pred_log_radii, pred_radii], device=device)
                min_d.zero_()
                wp.launch(_grouped_union_kernel, dim=batch,
                          inputs=[pred_centers, pred_radii, pred_rot_flat,
                                  ell_start, ell_count, wp_points,
                                  sample_idx, sample_grp, min_d, max_count,
                                  sdf_pred], device=device)
                loss.zero_()
                wp.launch(_grouped_loss_kernel, dim=batch,
                          inputs=[sdf_pred, sample_idx, sample_grp,
                                  wp_target, wp_thick, grp_sigma, grp_thickref,
                                  loss, inv_batch, self._miss,
                                  self._surface_weight, self._outside,
                                  self._thin_weight, self._thin_max_factor],
                          device=device)
                if self._flat_weight > 0.0:
                    wp.launch(_flatness_penalty_kernel, dim=total_e,
                              inputs=[pred_radii, loss, total_e,
                                      self._flat_weight, self._flat_min_ratio],
                              device=device)

            tape.backward(loss)
            opt_c.lr = lr
            opt_r.lr = lr * self._lr_mult_radii
            opt_q.lr = lr * self._lr_mult_rot
            opt_c.step(grad_c)
            opt_r.step(grad_r)
            opt_q.step(grad_q)
            tape.zero()

            wp.launch(_normalize_flat_quats, dim=total_e,
                      inputs=[pred_rot_flat], device=device)
            wp.launch(_clamp_log_radii_arr, dim=total_e,
                      inputs=[pred_log_radii, wp_log_rmin, wp_log_rmax],
                      device=device)
            wp.launch(_clamp_centers_arr, dim=total_e,
                      inputs=[pred_centers, wp_c_lo, wp_c_hi], device=device)

            if step % self._report_every == 0:
                wp.launch(_exp_radii_kernel, dim=total_e,
                          inputs=[pred_log_radii, pred_radii], device=device)
                wp.synchronize_device(device)
                c_np = pred_centers.numpy().copy()
                r_np = pred_radii.numpy().copy()
                q_np = pred_rot_flat.numpy().reshape(-1, 4).copy()
                self.step_visual.emit(step, float(loss.numpy()[0]),
                                      c_np, r_np, q_np)

        # ── finalise: split the flat param block back per group ──
        wp.launch(_exp_radii_kernel, dim=total_e,
                  inputs=[pred_log_radii, pred_radii], device=device)
        wp.synchronize_device(device)
        c_np = pred_centers.numpy().copy()
        r_np = pred_radii.numpy().copy()
        q_np = pred_rot_flat.numpy().reshape(-1, 4).copy()
        results = []
        for g, j in enumerate(jobs):
            a, b = int(ell_start_np[g]), int(ell_start_np[g + 1])
            cg, rg, qg = c_np[a:b], r_np[a:b], q_np[a:b]
            if j.symmetry is not None and len(cg):
                ax, co = j.symmetry
                cg, rg, qg = symmetrize_ellipsoids(cg, rg, qg, ax, co)
            results.append((j.bone_index, cg, rg, qg))
        self.results = results
