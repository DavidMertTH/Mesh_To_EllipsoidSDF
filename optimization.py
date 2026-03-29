import warp as wp
import warp.optim
import numpy as np

from PySide6 import QtCore

from ellipsoid import EllipsoidSet
from sdf_methods import sdf_ellipsoid_wp, METHOD_QUILEZ

# ── Loss IDs ──────────────────────────────────────────────────────────────────
LOSS_MAE = 0
LOSS_CLAMP = 1
LOSS_SOFT_CLAMP = 2

LOSS_INFO = [
    (LOSS_MAE,        "MAE",               "Mean absolute error"),
    (LOSS_CLAMP,      "Clamp (±0.1)",      "Clamped absolute error"),
    (LOSS_SOFT_CLAMP, "Soft Clamp (tanh)",  "Soft-clamped absolute error"),
]

# During Warp initialization, the default device is set to "cuda:0" if CUDA is available.
# Otherwise, the default device is "cpu".
device = wp.get_device()

@wp.kernel
def _ellipsoid_union_sdf_kernel(
    centers:        wp.array(dtype=wp.vec3),
    log_radii:      wp.array(dtype=wp.vec3),
    rot_flat:       wp.array(dtype=wp.float32),
    min_d:          wp.array2d(dtype=wp.float32),
    num_ellipsoids: int,
    origin:         wp.vec3,
    dx:             float,
    n:              int,
    method_id:      int,
    out_sdf:        wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    ix = tid % n
    iy = (tid // n) % n
    iz = tid // (n * n)

    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )

    min_d[tid, 0] = 1.0e6

    for i in range(num_ellipsoids):
        base = i * 4
        q = wp.normalize(wp.quat(
            rot_flat[base + 0],
            rot_flat[base + 1],
            rot_flat[base + 2],
            rot_flat[base + 3],
        ))
        local_p = wp.quat_rotate_inv(q, p - centers[i])
        lr = log_radii[i]
        r = wp.vec3(wp.exp(lr[0]), wp.exp(lr[1]), wp.exp(lr[2]))

        d = sdf_ellipsoid_wp(local_p, r, method_id)

        min_d[tid, i + 1] = wp.min(min_d[tid, i], d)

    out_sdf[tid] = min_d[tid, num_ellipsoids]


@wp.kernel
def _mae_loss_kernel(
    sdf_pred:   wp.array(dtype=wp.float32),
    sdf_target: wp.array(dtype=wp.float32),
    loss:       wp.array(dtype=wp.float32),
    n:          int,
):
    tid = wp.tid()
    diff = wp.abs(sdf_pred[tid] - sdf_target[tid])
    wp.atomic_add(loss, 0, diff / float(n))

@wp.kernel
def _clamp_loss_kernel(
    sdf_pred:   wp.array(dtype=wp.float32),
    sdf_target: wp.array(dtype=wp.float32),
    loss:       wp.array(dtype=wp.float32),
    n:          int,
):
    tid = wp.tid()
    delta = 0.1
    diff = wp.abs(wp.clamp(sdf_pred[tid], -delta, delta) - wp.clamp(sdf_target[tid], -delta, delta))
    wp.atomic_add(loss, 0, diff / float(n))

@wp.func
def soft_clamp(
    x: float,
    delta: float
):
    return delta * wp.tanh(x / delta)

@wp.kernel
def _soft_clamp_loss_kernel(
    sdf_pred:   wp.array(dtype=wp.float32),
    sdf_target: wp.array(dtype=wp.float32),
    loss:       wp.array(dtype=wp.float32),
    n:          int,
):
    tid = wp.tid()
    delta = 0.1
    diff = wp.abs(soft_clamp(sdf_pred[tid], delta) - soft_clamp(sdf_target[tid], delta))
    wp.atomic_add(loss, 0, diff / float(n))

@wp.kernel
def _eikonal_loss_kernel(
    sdf_pred: wp.array(dtype=wp.float32),
    loss:     wp.array(dtype=wp.float32),
    n:        int,
    dx:       float,
    weight:   float,
):
    tid = wp.tid()
    ix = tid % n
    iy = (tid // n) % n
    iz = tid // (n * n)

    # Skip boundary voxels — central differences need neighbours
    if ix == 0 or ix == n - 1:
        return
    if iy == 0 or iy == n - 1:
        return
    if iz == 0 or iz == n - 1:
        return

    inv_2dx = 1.0 / (2.0 * dx)

    gx = (sdf_pred[tid + 1]     - sdf_pred[tid - 1])     * inv_2dx
    gy = (sdf_pred[tid + n]     - sdf_pred[tid - n])     * inv_2dx
    gz = (sdf_pred[tid + n * n] - sdf_pred[tid - n * n]) * inv_2dx

    grad_mag = wp.sqrt(gx * gx + gy * gy + gz * gz)
    eik = (grad_mag - 1.0) * (grad_mag - 1.0)

    interior = float((n - 2) * (n - 2) * (n - 2))
    wp.atomic_add(loss, 0, weight * eik / interior)


class OptimizationWorker(QtCore.QThread):
    """Runs ellipsoid optimization in a background thread.

    All GPU → CPU readbacks happen *inside* this thread so the main
    (UI) thread never blocks on a device sync.

    Signals
    -------
    step_visual(int, float, ndarray, ndarray, ndarray)
        Emitted every *report_every* steps with lightweight numpy copies
        of (step, loss, centers(N,3), radii(N,3), rotations(N,4)).
        Intended for the 3-D viewer (fast concatenated mesh update).
    step_sdf(int, float, EllipsoidSet, bool, ndarray, float, int)
        Emitted every *report_every × 5* steps with an EllipsoidSet
        for the heavier SDF-slice recomputation.
    finished()
        Emitted when the loop ends.
    """

    step_visual = QtCore.Signal(int, float, object, object, object)
    step_sdf    = QtCore.Signal(int, float, object, bool, object, float, int)
    finished    = QtCore.Signal()

    def __init__(
        self,
        sdf_target_np: np.ndarray,
        origin: np.ndarray,
        dx: float,
        n: int,
        num_ellipsoids: int = 10,
        num_steps: int = 2000,
        report_every: int = 20,
        sdf_method_id: int = METHOD_QUILEZ,
        loss_id: int = LOSS_MAE,
        eikonal_enabled: bool = False,
        eikonal_weight: float = 0.1,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._sdf_target_np = sdf_target_np
        self._origin = origin
        self._dx = dx
        self._n = n
        self._num_ellipsoids = num_ellipsoids
        self._num_steps = num_steps
        self._report_every = report_every
        self._sdf_method_id = sdf_method_id
        self._loss_id = loss_id
        self._eikonal_enabled = eikonal_enabled
        self._eikonal_weight = eikonal_weight
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        self._run_optimization()
        self.finished.emit()

    # ── helper: async readback + emit ─────────────────────────────────

    def _emit_progress(self, step, loss_wp, pred_centers, pred_log_radii,
                       pred_rot_flat, num_e, origin, dx, n):
        """Read back arrays in this worker thread and emit signals."""
        wp.synchronize_device(device)

        loss_val = float(loss_wp.numpy()[0])

        # Lightweight readback for 3-D viewer (always)
        c_np = pred_centers.numpy().copy()
        r_np = np.exp(pred_log_radii.numpy()).copy()
        q_np = pred_rot_flat.numpy().reshape(-1, 4).copy()
        self.step_visual.emit(step, loss_val, c_np, r_np, q_np)

        # Heavier SDF-slice readback (every 5th report)
        if step % (self._report_every) == 0:
            ell_set = EllipsoidSet()
            ell_set.set_parameters(c_np, r_np, q_np)
            self.step_sdf.emit(step, loss_val, ell_set, True, origin, dx, n)

    def _run_optimization(self):
        origin = self._origin
        n = self._n
        dx = self._dx
        total = n * n * n
        num_e = self._num_ellipsoids

        sdf_target = wp.array(
            self._sdf_target_np.flatten(),
            dtype=wp.float32, device=device, requires_grad=False,
        )

        pred_centers = wp.array(
            (np.random.rand(num_e, 3).astype(np.float32) - 0.5) * 2.0 * 0.5,
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        pred_log_radii = wp.array(
            np.log(np.ones((num_e, 3), dtype=np.float32) * 0.1).astype(np.float32),
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        # Use a flat float32 array for rotations (Adam-compatible)
        unity_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_e, 1))
        pred_rot_flat = wp.array(
            unity_quats.flatten(),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        min_d_cache = wp.zeros(shape=(total, num_e + 1), dtype=wp.float32, device=device, requires_grad=True)

        sdf_pred = wp.empty(total, dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = 0.01

        params = [pred_centers, pred_log_radii, pred_rot_flat]
        grads = [p.grad.flatten() for p in params]
        optimizer = wp.optim.Adam(params, lr=lr)

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            tape = wp.Tape()
            with tape:
                min_d_cache.zero_()
                wp.launch(
                    _ellipsoid_union_sdf_kernel,
                    dim=total,
                    inputs=[pred_centers, pred_log_radii, pred_rot_flat, min_d_cache,
                            num_e, wp_origin, float(dx), n,
                            self._sdf_method_id, sdf_pred],
                    device=device,
                )
                loss.zero_()

                # Primary loss
                if self._loss_id == LOSS_CLAMP:
                    wp.launch(
                        _clamp_loss_kernel,
                        dim=total,
                        inputs=[sdf_pred, sdf_target, loss, total],
                        device=device,
                    )
                elif self._loss_id == LOSS_SOFT_CLAMP:
                    wp.launch(
                        _soft_clamp_loss_kernel,
                        dim=total,
                        inputs=[sdf_pred, sdf_target, loss, total],
                        device=device,
                    )
                else:
                    wp.launch(
                        _mae_loss_kernel,
                        dim=total,
                        inputs=[sdf_pred, sdf_target, loss, total],
                        device=device,
                    )

                # Eikonal regularisation (weighted additive term)
                if self._eikonal_enabled:
                    wp.launch(
                        _eikonal_loss_kernel,
                        dim=total,
                        inputs=[sdf_pred, loss, n, float(dx),
                                self._eikonal_weight],
                        device=device,
                    )

            tape.backward(loss)
            optimizer.step(grads)
            tape.zero()

            if step % self._report_every == 0:
                self._emit_progress(step, loss, pred_centers, pred_log_radii,
                                    pred_rot_flat, num_e, origin, dx, n)

                loss_val = float(loss.numpy()[0])
                if loss_val < 1e-10:
                    break