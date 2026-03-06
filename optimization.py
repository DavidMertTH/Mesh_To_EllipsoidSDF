import time

import warp as wp
import warp.optim
import numpy as np

from PySide6 import QtCore
from numpy.ma.core import shape

from ellipsoid import Ellipsoid
from ellipsoid import EllipsoidSet


@wp.kernel
def _ellipsoid_union_sdf_kernel_flat(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    min_d: wp.array4d(dtype=wp.float32),
    num_ellipsoids: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    """SDF kernel that reads rotations from a flat float32 array
    (4 consecutive floats per ellipsoid: x, y, z, w) so that the
    array is compatible with Warp's Adam optimizer."""
    tid = wp.tid()
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)

    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )

    min_d[ix, iy, iz, 0] = 1.0e6

    for i in range(num_ellipsoids):
        # Read 4 consecutive floats and build a normalised quaternion
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

        # Clamp to avoid division by zero (keeps autodiff stable)
        k1_safe = wp.max(k1, 1.0e-8)
        d = k0 * (k0 - 1.0) / k1_safe

        min_d[ix, iy, iz, i + 1] = wp.min(min_d[ix, iy, iz, i], d)

    out_sdf[tid] = min_d[ix, iy, iz, num_ellipsoids]


@wp.kernel
def _rmse_loss_kernel(
    sdf_pred: wp.array(dtype=wp.float32),
    sdf_target: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    n: int,
):
    tid = wp.tid()
    diff = sdf_pred[tid] - sdf_target[tid]
    # Accumulate squared error; divide by n outside or use atomic
    wp.atomic_add(loss, 0, diff * diff / float(n))

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
    """Normalize every group of 4 consecutive floats to unit length.
    Launch with dim = num_ellipsoids."""
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


class OptimizationWorker(QtCore.QThread):
    """Runs ellipsoid optimisation in a background thread.

    Signals
    -------
    step_done(int, float, EllipsoidSet)
        Emitted every *report_every* steps with (step, loss, current ellipsoids).
    finished()
        Emitted when the loop ends.
    """

    step_done = QtCore.Signal(int, float, object, bool, object, float, int)   # step, loss, EllipsoidSet
    finished = QtCore.Signal()

    def __init__(
        self,
        method: str = "naive",       # "naive" or "adam"
        num_steps: int = 2000,
        report_every: int = 20,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._method = method
        self._num_steps = num_steps
        self._report_every = report_every
        self._stop_flag = False

    # ── public helpers ────────────────────────────────────────────────

    def request_stop(self):
        """Ask the loop to stop at the next check-point."""
        self._stop_flag = True

    # ── thread entry point ────────────────────────────────────────────

    def run(self):
        if self._method == "adam":
            self._run_adam()
        else:
            self._run_naive()
        self.finished.emit()

    # ── naive SGD ─────────────────────────────────────────────────────

    def _run_naive(self):
        gt_set = create_demo_ellipsoids(device=device)
        origin = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        n = 128
        dx = 2.0 / n
        total = n * n * n

        sdf_target = wp.array(
            gt_set.compute_sdf_grid(origin, dx, n).flatten(),
            dtype=wp.float32, device=device, requires_grad=False,
        )

        num_e = gt_set.count
        pred_centers = wp.array(
            gt_set.centers + np.random.randn(*gt_set.centers.shape).astype(np.float32) * 1.0,
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        pred_radii = wp.array(
            gt_set.radii + np.random.randn(*gt_set.radii.shape).astype(np.float32) * 0.5,
            # gt_set.radii.copy(),
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        # Use a flat float32 array for rotations (4 floats per ellipsoid)
        # so that they are differentiable with any Warp optimizer
        pred_rot_flat = wp.array(
            gt_set.rotations.copy().flatten(),
            dtype=wp.float32, device=device, requires_grad=True,
        )

        sdf_pred = wp.empty(total, dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = 0.01

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            tape = wp.Tape()
            with tape:
                wp.launch(
                    _ellipsoid_union_sdf_kernel_flat,
                    dim=total,
                    inputs=[pred_centers, pred_radii, pred_rot_flat,
                            num_e, wp_origin, float(dx), n, n, n, sdf_pred],
                    device=device,
                )
                loss.zero_()
                wp.launch(
                    _rmse_loss_kernel,
                    dim=total,
                    inputs=[sdf_pred, sdf_target, loss, total],
                    device=device,
                )

            tape.backward(loss)

            pred_centers_grad = tape.gradients[pred_centers]
            pred_radii_grad = tape.gradients[pred_radii]
            pred_rot_flat_grad = tape.gradients[pred_rot_flat]

            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_centers, pred_centers_grad, lr], device=device)
            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_radii, pred_radii_grad, lr], device=device)
            wp.launch(_sgd_step_f32, dim=num_e * 4,
                      inputs=[pred_rot_flat, pred_rot_flat_grad, lr], device=device)
            # Re-normalize quaternions after the gradient step
            wp.launch(_normalize_flat_quats, dim=num_e,
                      inputs=[pred_rot_flat], device=device)

            tape.zero()

            if step % self._report_every == 0:
                ell_set = EllipsoidSet()
                centers = np.concatenate((gt_set.centers, pred_centers.numpy()))
                radii = np.concatenate((gt_set.radii, pred_radii.numpy()))
                # Reshape flat float32 back to (N, 4) for EllipsoidSet
                rotations = np.concatenate((gt_set.rotations, pred_rot_flat.numpy().reshape(-1, 4)))
                ell_set.set_parameters(
                    centers,
                    radii,
                    rotations,
                )
                loss_val = float(loss.numpy()[0])
                self.step_done.emit(step, loss_val, ell_set, True, origin, dx, n)

    # ── Adam ──────────────────────────────────────────────────────────

    def _run_adam(self):
        gt_set = create_demo_ellipsoids(device=device)
        origin = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        n = 128
        dx = 2.0 / n
        total = n * n * n

        sdf_target = wp.array(
            gt_set.compute_sdf_grid(origin, dx, n).flatten(),
            dtype=wp.float32, device=device, requires_grad=False,
        )

        num_e = 10 #gt_set.count
        pred_centers = wp.array(
            #gt_set.centers + np.random.randn(*gt_set.centers.shape).astype(np.float32) * 1.0,
            (np.random.rand( num_e, 3 ).astype(np.float32) - 0.5) * 2.0 * 0.5, # range [-0.5, 0.5)
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        pred_radii = wp.array(
            #gt_set.radii + np.random.randn(*gt_set.radii.shape).astype(np.float32) * 0.5,
            np.ones((num_e, 3)) * 0.1,
            #gt_set.radii.copy(),
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        # Use a flat float32 array for rotations (Adam-compatible)
        unity_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_e, 1))
        pred_rot_flat = wp.array(
            unity_quats.flatten(),
            dtype=wp.float32, device=device, requires_grad=True,
        )
        min_d_cache = wp.zeros(shape = (n, n, n, num_e + 1), dtype=wp.float32, device=device, requires_grad=True)

        sdf_pred = wp.empty(total, dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = 0.01

        #params = [pred_centers]
        #params = [pred_centers, pred_radii]
        params = [pred_centers, pred_radii, pred_rot_flat]
        grads = [p.grad.flatten() for p in params]
        optimizer = wp.optim.Adam(params, lr=lr)

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            tape = wp.Tape()
            with tape:
                min_d_cache.zero_()
                wp.launch(
                    _ellipsoid_union_sdf_kernel_flat,
                    dim=total,
                    inputs=[pred_centers, pred_radii, pred_rot_flat, min_d_cache,
                            num_e, wp_origin, float(dx), n, n, n, sdf_pred],
                    device=device,
                )
                loss.zero_()
                wp.launch(
                    _rmse_loss_kernel,
                    dim=total,
                    inputs=[sdf_pred, sdf_target, loss, total],
                    device=device,
                )

            tape.backward(loss)
            optimizer.step(grads)
            # Re-normalize quaternions after Adam step
            # wp.launch(_normalize_flat_quats, dim=num_e,
            #           inputs=[pred_rot_flat], device=device)
            tape.zero()

            if step % self._report_every == 0:
                ell_set = EllipsoidSet()
                centers = np.concatenate((gt_set.centers, pred_centers.numpy()))
                radii = np.concatenate((gt_set.radii, pred_radii.numpy()))
                # Reshape flat float32 back to (N, 4) for EllipsoidSet
                rotations = np.concatenate((gt_set.rotations, pred_rot_flat.numpy().reshape(-1, 4)))

                gt_color = (0.0, 1.0, 0.0, 0.5)
                pred_color = (1.0, 1.0, 0.0, 0.9)
                colors = [gt_color for _ in range(gt_set.count)]
                colors.extend([pred_color for _ in range(num_e)])
                ell_set.set_parameters(
                    centers,
                    radii,
                    rotations,
                    colors
                )
                loss_val = float(loss.numpy()[0])
                self.step_done.emit(step, loss_val, ell_set, True, origin, dx, n)
                #self.step_done.emit(step, loss_val, None, None, None, None, None)
                if loss_val < 1e-10:
                    break
                #time.sleep(0.2)  # Yield to UI thread


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