import time

import warp as wp
import warp.optim
import numpy as np

from PySide6 import QtCore

from ellipsoid import Ellipsoid
from ellipsoid import EllipsoidSet


@wp.kernel
def _ellipsoid_union_sdf_kernel(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quat),
    num_ellipsoids: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)

    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )

    min_d = float(1.0e6)

    for i in range(num_ellipsoids):
        local_p = wp.quat_rotate_inv(rotations[i], p - centers[i])
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

        min_d = wp.min(min_d, d)

    out_sdf[tid] = min_d


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
def _sgd_step_quat(
    param: wp.array(dtype=wp.quat),
    grad: wp.array(dtype=wp.quat),
    lr: float,
):
    tid = wp.tid()
    q = param[tid]
    g = grad[tid]
    # Simple gradient step + renormalize
    new_q = wp.quat(
        q[0] - lr * g[0],
        q[1] - lr * g[1],
        q[2] - lr * g[2],
        q[3] - lr * g[3],
    )
    param[tid] = wp.normalize(new_q)


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
        time.sleep(15)
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
        pred_rotations = wp.array(
            gt_set.rotations.copy(),
            dtype=wp.quat, device=device, requires_grad=False,
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
                    _ellipsoid_union_sdf_kernel,
                    dim=total,
                    inputs=[pred_centers, pred_radii, pred_rotations,
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
            #pred_rotations_grad = tape.gradients[pred_rotations]

            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_centers, pred_centers_grad, lr], device=device)
            wp.launch(_sgd_step_vec3, dim=num_e,
                      inputs=[pred_radii, pred_radii_grad, lr], device=device)
            #wp.launch(_sgd_step_quat, dim=num_e,
            #          inputs=[pred_rotations, pred_rotations_grad, lr], device=device)

            tape.zero()

            if step % self._report_every == 0:
                ell_set = EllipsoidSet()
                centers = np.concatenate((gt_set.centers, pred_centers.numpy()))
                radii = np.concatenate((gt_set.radii, pred_radii.numpy()))
                rotations = np.concatenate((gt_set.rotations, pred_rotations.numpy()))
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

        num_e = gt_set.count
        pred_centers = wp.array(
            #gt_set.centers + np.random.randn(*gt_set.centers.shape).astype(np.float32) * 1.0,
            origin + np.random.rand(*gt_set.centers.shape).astype(np.float32) * 0.5,
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        pred_radii = wp.array(
            #gt_set.radii + np.random.randn(*gt_set.radii.shape).astype(np.float32) * 0.5,
            np.ones_like(gt_set.radii) * 0.1,
            #gt_set.radii.copy(),
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        pred_rotations = wp.array(
            gt_set.rotations.copy(),
            dtype=wp.quat, device=device, requires_grad=False,
        )

        sdf_pred = wp.empty(total, dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        lr = 0.01

        #params = [pred_centers]
        params = [pred_centers, pred_radii]
        #params = [pred_centers, pred_radii, pred_rotations]
        grads = [p.grad.flatten() for p in params]
        optimizer = wp.optim.Adam(params, lr=lr)

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            tape = wp.Tape()
            with tape:
                wp.launch(
                    _ellipsoid_union_sdf_kernel,
                    dim=total,
                    inputs=[pred_centers, pred_radii, pred_rotations,
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
            tape.zero()

            if step % self._report_every == 0:
                ell_set = EllipsoidSet()
                centers = np.concatenate((gt_set.centers, pred_centers.numpy()))
                radii = np.concatenate((gt_set.radii, pred_radii.numpy()))
                rotations = np.concatenate((gt_set.rotations, pred_rotations.numpy()))
                ell_set.set_parameters(
                    centers,
                    radii,
                    rotations,
                )
                loss_val = float(loss.numpy()[0])
                self.step_done.emit(step, loss_val, ell_set, True, origin, dx, n)
                if loss_val < 1e-10:
                    break
                time.sleep(0.2)  # Yield to UI thread


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
        # Ellipsoid(
        #     center=np.array([0.4, 0.4, 0.0], dtype=np.float32),
        #     radii=np.array([0.5, 0.3, 0.3], dtype=np.float32),
        #     rotation=q_id,
        # ),
        # Ellipsoid(
        #     center=np.array([-0.3, -0.3, 0.2], dtype=np.float32),
        #     radii=np.array([0.5, 0.3, 0.3], dtype=np.float32),
        #     rotation=q_id,
        # ),
        # Ellipsoid(
        #     center=np.array([0.0, 0.5, -0.3], dtype=np.float32),
        #     radii=np.array([0.5, 0.3, 0.3], dtype=np.float32),
        #     rotation=q_id,
        # ),
        # Ellipsoid(
        #     center=np.array([-0.5, 0.1, 0.1], dtype=np.float32),
        #     radii=np.array([0.5, 0.3, 0.3], dtype=np.float32),
        #     rotation=q_id,
        # ),
    ]

    return EllipsoidSet.from_list(ellipsoids, device=device)