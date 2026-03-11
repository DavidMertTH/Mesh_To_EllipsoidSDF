"""
optimization.py — Ellipsoid fitting via differentiable SDF.

MINI-BATCH + GENTLE POPULATION MANAGEMENT:
  - Epoch-based mini-batch sampling (index indirection).
  - Periodic maintenance (every `maintenance_every` iterations):
      1. PRUNE (gentle) — remove ellipsoids that are truly redundant:
         (a) degenerate (collapsed to near-zero volume), or
         (b) *similar* to another ellipsoid — meaning their centres
             are close relative to their mean size AND their sorted
             radii ratios are all near 1.  When a pair is found,
             the smaller one is removed.
         → A tight budget cap (`max_prune_fraction`) limits removals
           per round so training stays stable.
      2. SPAWN — fill vacancies at high-error regions via
         farthest-point sampling.
"""

import time as _time

import warp as wp
import warp.optim
import numpy as np

from PySide6 import QtCore

from ellipsoid import Ellipsoid, EllipsoidSet


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
        scaled2 = wp.vec3(
            local_p[0] / (r[0] * r[0]),
            local_p[1] / (r[1] * r[1]),
            local_p[2] / (r[2] * r[2]),
        )

        k0 = wp.length(scaled)
        k1 = wp.length(scaled2)
        k1_safe = wp.max(k1, 1.0e-8)
        d = k0 * (k0 - 1.0) / k1_safe

        min_d[bid, i + 1] = wp.min(min_d[bid, i], d)

    out_sdf[bid] = min_d[bid, num_ellipsoids]


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
):
    bid = wp.tid()
    tid = indices[bid]
    limit = 0.1
    diff = wp.abs(soft_clamp(sdf_pred[bid], limit) - soft_clamp(sdf_target[tid], limit))
    wp.atomic_add(loss, 0, diff / float(batch_size))


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


device = "cuda"


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


# ── Quaternion helper (numpy) ─────────────────────────────────────────────────

def _quat_to_rot_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
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


# ── Worker ────────────────────────────────────────────────────────────────────

class OptimizationWorker(QtCore.QThread):
    """Ellipsoid fitting with mini-batch sampling + gentle population mgmt.

    Signals
    -------
    step_visual : (step, loss, centers_np, radii_np, rot_flat_np)
        Emitted every ``report_every`` steps for the 3-D viewer.
    step_metrics : (dict)
        Emitted alongside step_visual.  Keys:
            step, loss, num_ellipsoids, iter_ms, elapsed_s,
            mean_radius, total_volume, min_radius, max_radius
    step_sdf : (step, loss, ell_set, bool, origin, dx, n)
        Emitted less frequently for the SDF slice recompute.
    maintenance_done : (step, n_before, n_pruned, n_spawned)
    finished : ()
    """

    step_visual      = QtCore.Signal(int, float, object, object, object)
    step_sdf         = QtCore.Signal(int, float, object, bool, object, float, int)
    maintenance_done = QtCore.Signal(int, int, int, int)
    step_metrics     = QtCore.Signal(object)
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
        batch_fraction: float | None = None,
        batch_size: int | None = None,
        maintenance_every: int = 500,
        similarity_center_thresh: float = 0.5,
        similarity_shape_thresh: float = 1.6,
        max_prune_fraction: float = 0.08,
        min_volume_abs: float = 1e-9,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._sdf_target_np = sdf_target_np
        self._origin = origin
        self._dx = dx
        self._n = n
        self._num_ellipsoids = num_ellipsoids
        self._method = method
        self._num_steps = num_steps
        self._report_every = report_every
        self._stop_flag = False

        self._maintenance_every = maintenance_every
        self._similarity_center_thresh = similarity_center_thresh
        self._similarity_shape_thresh = similarity_shape_thresh
        self._max_prune_fraction = max_prune_fraction
        self._min_volume_abs = min_volume_abs

        total = n * n * n
        if batch_size is not None:
            self._batch_size = min(batch_size, total)
        else:
            frac = batch_fraction or self.DEFAULT_BATCH_FRACTION
            self._batch_size = max(1024, min(int(total * frac), total))

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        if self._method == "adam":
            self._run_adam()
        else:
            self._run_naive()
        self.finished.emit()

    # ── metrics helper ────────────────────────────────────────────────

    @staticmethod
    def _build_metrics(
        step: int,
        loss_val: float,
        num_e: int,
        radii_np: np.ndarray,
        iter_ms: float,
        elapsed_s: float,
    ) -> dict:
        abs_r = np.abs(radii_np)
        mean_r = float(np.mean(abs_r)) if len(abs_r) > 0 else 0.0
        total_vol = float(np.sum(np.prod(abs_r, axis=1))) if len(abs_r) > 0 else 0.0
        min_r = float(np.min(abs_r)) if len(abs_r) > 0 else 0.0
        max_r = float(np.max(abs_r)) if len(abs_r) > 0 else 0.0
        return dict(
            step=step,
            loss=loss_val,
            num_ellipsoids=num_e,
            iter_ms=iter_ms,
            elapsed_s=elapsed_s,
            mean_radius=mean_r,
            total_volume=total_vol,
            min_radius=min_r,
            max_radius=max_r,
        )

    # ── progress reporting ────────────────────────────────────────────

    def _emit_progress(self, step, loss_wp, pred_centers, pred_radii,
                       pred_rot_flat, num_e, origin, dx, n,
                       iter_ms: float, elapsed_s: float):
        wp.synchronize_device(device)
        loss_val = float(loss_wp.numpy()[0])

        c_np = pred_centers.numpy().copy()
        r_np = pred_radii.numpy().copy()
        q_np = pred_rot_flat.numpy().reshape(-1, 4).copy()
        self.step_visual.emit(step, loss_val, c_np, r_np, q_np)

        m = self._build_metrics(step, loss_val, num_e, r_np, iter_ms, elapsed_s)
        self.step_metrics.emit(m)

        if step % (self._report_every * 10) == 0:
            ell_set = EllipsoidSet()
            ell_set.set_parameters(c_np, r_np, q_np)
            self.step_sdf.emit(step, loss_val, ell_set, True, origin, dx, n)

    # ── buffer allocation ─────────────────────────────────────────────

    def _init_inside_mesh(self, num_e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origin = self._origin
        dx = self._dx
        n = self._n

        flat_target = self._sdf_target_np.ravel()
        interior_mask = flat_target < 0.0
        interior_idx = np.where(interior_mask)[0]

        if len(interior_idx) == 0:
            centers = (np.random.rand(num_e, 3).astype(np.float32) - 0.5)
            radii = np.ones((num_e, 3), dtype=np.float32) * 0.1
            rots = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_e, 1))
            return centers, radii, rots

        iz, iy, ix = np.unravel_index(interior_idx, (n, n, n))
        interior_world = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx
        interior_depth = np.abs(flat_target[interior_idx])

        selected = self._farthest_point_sample(
            interior_world, interior_depth, num_e,
            existing_centers=np.empty((0, 3), dtype=np.float32),
        )

        centers = interior_world[selected].astype(np.float32)
        local_depth = interior_depth[selected]

        min_r = float(dx) * 2.0
        init_r = np.clip(local_depth * 0.6, min_r, None)
        radii = np.stack([init_r, init_r, init_r], axis=1).astype(np.float32)

        rots = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_e, 1),
        )
        return centers, radii, rots

    def _alloc_buffers(self, num_e, batch_size, total,
                       centers_np=None, radii_np=None, rot_np=None):
        sdf_target = wp.array(
            self._sdf_target_np.flatten(),
            dtype=wp.float32, device=device, requires_grad=False,
        )

        if centers_np is None:
            centers_np, radii_np, rot_np = self._init_inside_mesh(num_e)
        if radii_np is None:
            radii_np = np.ones((num_e, 3), dtype=np.float32) * 0.1
        if rot_np is None:
            rot_np = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_e, 1),
            )

        pred_centers = wp.array(centers_np.astype(np.float32), dtype=wp.vec3,
                                device=device, requires_grad=True)
        pred_radii = wp.array(radii_np.astype(np.float32), dtype=wp.vec3,
                              device=device, requires_grad=True)
        pred_rot_flat = wp.array(rot_np.astype(np.float32).flatten(),
                                 dtype=wp.float32, device=device, requires_grad=True)
        min_d_cache = wp.zeros(shape=(batch_size, num_e + 1),
                               dtype=wp.float32, device=device, requires_grad=True)
        sdf_pred = wp.empty(batch_size, dtype=wp.float32,
                            device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        wp_indices = wp.empty(batch_size, dtype=wp.int32, device=device)

        return dict(sdf_target=sdf_target, pred_centers=pred_centers,
                    pred_radii=pred_radii, pred_rot_flat=pred_rot_flat,
                    min_d_cache=min_d_cache, sdf_pred=sdf_pred,
                    loss=loss, wp_indices=wp_indices)

    # ══════════════════════════════════════════════════════════════════
    # GENTLE POPULATION MANAGEMENT  (similarity-based)
    # ══════════════════════════════════════════════════════════════════

    def _do_maintenance(self, centers, radii, rotations):
        n_before = len(centers)
        budget = max(1, int(n_before * self._max_prune_fraction))

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

        if budget > 0 and len(centers) >= 2:
            to_remove = self._find_similar(centers, radii)[:budget]
            if len(to_remove) > 0:
                keep_mask = np.ones(len(centers), dtype=bool)
                keep_mask[to_remove] = False
                centers   = centers[keep_mask]
                radii     = radii[keep_mask]
                rotations = rotations[keep_mask]

        n_pruned = n_before - len(centers)
        num_to_spawn = self._num_ellipsoids - len(centers)
        n_spawned = 0

        if num_to_spawn > 0:
            new_c, new_r, new_q = self._spawn_at_errors(centers, radii, rotations, num_to_spawn)
            centers   = np.concatenate([centers, new_c], axis=0)
            radii     = np.concatenate([radii, new_r], axis=0)
            rotations = np.concatenate([rotations, new_q], axis=0)
            n_spawned = num_to_spawn

        changed = n_pruned > 0 or n_spawned > 0
        return centers, radii, rotations, changed, n_pruned, n_spawned

    def _find_similar(self, centers, radii):
        n = len(centers)
        abs_radii = np.abs(radii)
        volumes = np.prod(abs_radii, axis=1)
        avg_r   = np.mean(abs_radii, axis=1)
        sorted_radii = np.sort(abs_radii, axis=1)

        remove_set: set[int] = set()
        order = np.argsort(volumes)

        for rank_a in range(n):
            i = order[rank_a]
            if i in remove_set:
                continue
            ri = sorted_radii[i]; ci = centers[i]; avg_ri = avg_r[i]

            for rank_b in range(rank_a + 1, n):
                j = order[rank_b]
                if j in remove_set:
                    continue

                mean_avg_r = 0.5 * (avg_ri + avg_r[j])
                if mean_avg_r < 1e-12:
                    continue
                if float(np.linalg.norm(ci - centers[j])) / mean_avg_r > self._similarity_center_thresh:
                    continue

                rj = sorted_radii[j]
                ratio = np.maximum(ri / np.maximum(rj, 1e-12),
                                   rj / np.maximum(ri, 1e-12))
                if np.max(ratio) > self._similarity_shape_thresh:
                    continue

                remove_set.add(i)
                break

        return sorted(remove_set, key=lambda idx: volumes[idx])

    # ── spawning ──────────────────────────────────────────────────────

    def _spawn_at_errors(self, centers, radii, rotations, num_spawn):
        origin = self._origin; dx = self._dx; n = self._n

        ell_set = EllipsoidSet(device=device)
        if len(centers) > 0:
            ell_set.set_parameters(centers, radii, rotations)
        pred_grid = ell_set.compute_sdf_grid(origin, dx, n)
        target_grid = self._sdf_target_np
        error = np.abs(_soft_clamp_np(pred_grid, 0.1) - _soft_clamp_np(target_grid, 0.1))

        flat_target = target_grid.ravel(); flat_error = error.ravel()
        interior_idx = np.where(flat_target < 0.0)[0]
        if len(interior_idx) == 0:
            interior_idx = np.where(np.abs(flat_target) < 2.0 * dx)[0]
        if len(interior_idx) == 0:
            return (np.zeros((num_spawn, 3), dtype=np.float32),
                    np.full((num_spawn, 3), float(dx) * 3.0, dtype=np.float32),
                    np.tile(np.array([0,0,0,1], dtype=np.float32), (num_spawn, 1)))

        interior_errors = flat_error[interior_idx]
        pool_size = min(num_spawn * 50, len(interior_idx))
        top_local = np.argpartition(interior_errors, -pool_size)[-pool_size:]
        pool_flat_idx = interior_idx[top_local]

        iz, iy, ix = np.unravel_index(pool_flat_idx, (n, n, n))
        pool_world = origin + (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * dx
        pool_errors = flat_error[pool_flat_idx]

        selected_idx = self._farthest_point_sample(pool_world, pool_errors, num_spawn, centers)
        new_centers = pool_world[selected_idx].astype(np.float32)
        local_depth = np.abs(flat_target[pool_flat_idx[selected_idx]])
        min_r = float(dx) * 2.0
        init_r = np.clip(local_depth * 0.6, min_r, None)
        new_radii = np.stack([init_r, init_r, init_r], axis=1).astype(np.float32)
        new_rots = np.tile(np.array([0,0,0,1], dtype=np.float32), (num_spawn, 1))
        return new_centers, new_radii, new_rots

    @staticmethod
    def _farthest_point_sample(candidates, errors, k, existing_centers):
        n_cand = len(candidates)
        if n_cand == 0 or k == 0:
            return np.array([], dtype=int)

        if len(existing_centers) > 0:
            dists = np.linalg.norm(
                candidates[:, np.newaxis, :] - existing_centers[np.newaxis, :, :], axis=2)
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

    def _maybe_maintain(self, step, pred_centers, pred_radii, pred_rot_flat):
        if self._maintenance_every <= 0:
            return None
        if step == 0 or step % self._maintenance_every != 0:
            return None

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

    # ── naive SGD ─────────────────────────────────────────────────────

    def _run_naive(self):
        origin = self._origin; n = self._n; dx = self._dx
        total = n * n * n; num_e = self._num_ellipsoids; bs = self._batch_size

        buf = self._alloc_buffers(num_e, bs, total)
        pred_centers  = buf['pred_centers'];  pred_radii  = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']; min_d_cache = buf['min_d_cache']
        sdf_pred = buf['sdf_pred']; loss = buf['loss']
        sdf_target = buf['sdf_target']; wp_indices = buf['wp_indices']

        sampler = EpochSampler(total, bs)
        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = 0.01

        t_start = _time.perf_counter()
        t_last_report = t_start

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            result = self._maybe_maintain(step, pred_centers, pred_radii, pred_rot_flat)
            if result is not None:
                c_np, r_np, q_np = result; num_e = len(c_np)
                buf = self._alloc_buffers(num_e, bs, total, c_np, r_np, q_np)
                buf['sdf_target'] = sdf_target
                pred_centers = buf['pred_centers']; pred_radii = buf['pred_radii']
                pred_rot_flat = buf['pred_rot_flat']; min_d_cache = buf['min_d_cache']
                sdf_pred = buf['sdf_pred']; loss = buf['loss']; wp_indices = buf['wp_indices']

            wp_indices.assign(sampler.next_batch())
            tape = wp.Tape()
            with tape:
                min_d_cache.zero_()
                wp.launch(_ellipsoid_sdf_kernel_batch, dim=bs,
                          inputs=[pred_centers, pred_radii, pred_rot_flat, min_d_cache,
                                  num_e, wp_origin, float(dx), n, n, n, wp_indices, sdf_pred],
                          device=device)
                loss.zero_()
                wp.launch(_rmse_loss_kernel_batch, dim=bs,
                          inputs=[sdf_pred, sdf_target, wp_indices, loss, bs], device=device)

            tape.backward(loss)
            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_centers, tape.gradients[pred_centers], lr], device=device)
            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_radii, tape.gradients[pred_radii], lr], device=device)
            wp.launch(_sgd_step_f32, dim=num_e * 4,
                      inputs=[pred_rot_flat, tape.gradients[pred_rot_flat], lr], device=device)
            wp.launch(_normalize_flat_quats, dim=num_e,
                      inputs=[pred_rot_flat], device=device)
            tape.zero()

            if step % self._report_every == 0:
                t_now = _time.perf_counter()
                elapsed_s = t_now - t_start
                iter_ms = ((t_now - t_last_report) / max(self._report_every, 1)) * 1000.0
                t_last_report = t_now
                self._emit_progress(step, loss, pred_centers, pred_radii,
                                    pred_rot_flat, num_e, origin, dx, n,
                                    iter_ms, elapsed_s)

    # ── Adam ──────────────────────────────────────────────────────────

    def _run_adam(self):
        origin = self._origin; n = self._n; dx = self._dx
        total = n * n * n; num_e = self._num_ellipsoids; bs = self._batch_size

        buf = self._alloc_buffers(num_e, bs, total)
        pred_centers  = buf['pred_centers'];  pred_radii  = buf['pred_radii']
        pred_rot_flat = buf['pred_rot_flat']; min_d_cache = buf['min_d_cache']
        sdf_pred = buf['sdf_pred']; loss = buf['loss']
        sdf_target = buf['sdf_target']; wp_indices = buf['wp_indices']

        sampler = EpochSampler(total, bs)
        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = 0.01

        params = [pred_centers, pred_radii, pred_rot_flat]
        grads = [p.grad.flatten() for p in params]
        optimizer = wp.optim.Adam(params, lr=lr)

        t_start = _time.perf_counter()
        t_last_report = t_start

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            result = self._maybe_maintain(step, pred_centers, pred_radii, pred_rot_flat)
            if result is not None:
                c_np, r_np, q_np = result; num_e = len(c_np)
                buf = self._alloc_buffers(num_e, bs, total, c_np, r_np, q_np)
                buf['sdf_target'] = sdf_target
                pred_centers = buf['pred_centers']; pred_radii = buf['pred_radii']
                pred_rot_flat = buf['pred_rot_flat']; min_d_cache = buf['min_d_cache']
                sdf_pred = buf['sdf_pred']; loss = buf['loss']; wp_indices = buf['wp_indices']
                params = [pred_centers, pred_radii, pred_rot_flat]
                grads = [p.grad.flatten() for p in params]
                optimizer = wp.optim.Adam(params, lr=lr)

            wp_indices.assign(sampler.next_batch())
            tape = wp.Tape()
            with tape:
                min_d_cache.zero_()
                wp.launch(_ellipsoid_sdf_kernel_batch, dim=bs,
                          inputs=[pred_centers, pred_radii, pred_rot_flat, min_d_cache,
                                  num_e, wp_origin, float(dx), n, n, n, wp_indices, sdf_pred],
                          device=device)
                loss.zero_()
                wp.launch(_rmse_loss_kernel_batch, dim=bs,
                          inputs=[sdf_pred, sdf_target, wp_indices, loss, bs], device=device)

            tape.backward(loss)
            optimizer.step(grads)
            tape.zero()

            if step % self._report_every == 0:
                t_now = _time.perf_counter()
                elapsed_s = t_now - t_start
                iter_ms = ((t_now - t_last_report) / max(self._report_every, 1)) * 1000.0
                t_last_report = t_now
                self._emit_progress(step, loss, pred_centers, pred_radii,
                                    pred_rot_flat, num_e, origin, dx, n,
                                    iter_ms, elapsed_s)

                wp.synchronize_device(device)
                loss_val = float(loss.numpy()[0])
                if loss_val < 1e-10:
                    break


# ── Demo helper ───────────────────────────────────────────────────────────────

def create_demo_ellipsoids(device: str = "cpu") -> EllipsoidSet:
    q_id = Ellipsoid.identity_quat()
    angle = np.radians(45.0); half = angle * 0.5
    q_tilt_z = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)
    angle_x = np.radians(30.0); half_x = angle_x * 0.5
    q_tilt_x = np.array([np.sin(half_x), 0.0, 0.0, np.cos(half_x)], dtype=np.float32)

    ellipsoids = [
        Ellipsoid(center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                  radii=np.array([0.5, 0.3, 0.3], dtype=np.float32), rotation=q_id),
        Ellipsoid(center=np.array([0.4, 0.4, 0.0], dtype=np.float32),
                  radii=np.array([0.25, 0.15, 0.2], dtype=np.float32), rotation=q_id),
        Ellipsoid(center=np.array([-0.3, -0.3, 0.2], dtype=np.float32),
                  radii=np.array([0.3, 0.2, 0.15], dtype=np.float32), rotation=q_id),
        Ellipsoid(center=np.array([0.0, 0.5, -0.3], dtype=np.float32),
                  radii=np.array([0.15, 0.35, 0.15], dtype=np.float32), rotation=q_id),
        Ellipsoid(center=np.array([-0.5, 0.1, 0.1], dtype=np.float32),
                  radii=np.array([0.2, 0.2, 0.35], dtype=np.float32), rotation=q_id),
    ]
    return EllipsoidSet.from_list(ellipsoids, device=device)