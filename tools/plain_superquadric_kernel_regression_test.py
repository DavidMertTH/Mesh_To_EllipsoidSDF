"""Compare straight-SQ kernels with the bent kernels at zero curvature.

Run: .venv/Scripts/python.exe tools/plain_superquadric_kernel_regression_test.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization import (  # noqa: E402
    _plain_superquadric_sdf_kernel_batch,
    _plain_superquadric_sdf_kernel_points,
    _plain_superquadric_softmin_kernel_batch,
    _plain_superquadric_softmin_kernel_points,
    _superquadric_sdf_kernel_batch,
    _superquadric_sdf_kernel_points,
    _superquadric_softmin_kernel_batch,
    _superquadric_softmin_kernel_points,
)


DEVICE = os.environ.get("SQ_TEST_DEVICE", "cpu")


@wp.kernel
def _sum_output(values: wp.array(dtype=wp.float32),
                total: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(total, 0, values[i])


def _run(*, soft: bool, points_mode: bool, bent: bool):
    centers = wp.array(
        np.array([[-0.4, 0.1, 0.0], [0.7, -0.2, 0.3]], np.float32),
        dtype=wp.vec3, device=DEVICE, requires_grad=True)
    radii = wp.array(
        np.array([[0.9, 0.5, 0.7], [0.55, 0.75, 0.4]], np.float32),
        dtype=wp.vec3, device=DEVICE, requires_grad=True)
    rot = wp.array(
        np.array([0, 0, 0, 1, 0, 0, 0, 1], np.float32),
        dtype=wp.float32, device=DEVICE)
    eps = wp.array(
        np.array([0.7, 1.3, 1.2, 0.8], np.float32),
        dtype=wp.float32, device=DEVICE, requires_grad=True)
    bend = wp.zeros(4, dtype=wp.float32, device=DEVICE)
    sample_indices = np.array([0, 1, 2, 3, 5, 6, 7], np.int32)
    indices = wp.array(sample_indices, dtype=wp.int32, device=DEVICE)
    origin = wp.vec3(-1.0, -1.0, -1.0)
    dx = 1.0
    points_np = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
    ], np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=DEVICE)
    count = len(sample_indices)
    out = wp.empty(count, dtype=wp.float32, device=DEVICE,
                   requires_grad=True)
    total = wp.zeros(1, dtype=wp.float32, device=DEVICE,
                     requires_grad=True)
    prefix = [centers, radii, rot, eps]
    if bent:
        prefix.append(bend)
    if soft:
        m_cache = wp.zeros((count, 3), dtype=wp.float32, device=DEVICE,
                           requires_grad=True)
        s_cache = wp.zeros((count, 3), dtype=wp.float32, device=DEVICE,
                           requires_grad=True)
        args = prefix + [m_cache, s_cache, 2]
        kernel = (
            _superquadric_softmin_kernel_points if bent else
            _plain_superquadric_softmin_kernel_points
        ) if points_mode else (
            _superquadric_softmin_kernel_batch if bent else
            _plain_superquadric_softmin_kernel_batch
        )
    else:
        m_cache = wp.zeros((count, 3), dtype=wp.float32, device=DEVICE,
                           requires_grad=True)
        args = prefix + [m_cache, 2]
        kernel = (
            _superquadric_sdf_kernel_points if bent else
            _plain_superquadric_sdf_kernel_points
        ) if points_mode else (
            _superquadric_sdf_kernel_batch if bent else
            _plain_superquadric_sdf_kernel_batch
        )
    if points_mode:
        args += [points, indices, out]
    else:
        args += [origin, dx, 2, 2, 2, indices, out]
    if soft:
        args.append(8.0)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=count, inputs=args, device=DEVICE)
        wp.launch(_sum_output, dim=count, inputs=[out, total], device=DEVICE)
    tape.backward(total)
    return (out.numpy().copy(), centers.grad.numpy().copy(),
            radii.grad.numpy().copy(), eps.grad.numpy().copy())


def test_plain_matches_zero_bend() -> None:
    for soft in (False, True):
        for points_mode in (False, True):
            plain = _run(soft=soft, points_mode=points_mode, bent=False)
            zero_bend = _run(soft=soft, points_mode=points_mode, bent=True)
            for label, actual, expected in zip(
                    ("SDF", "centers", "radii", "eps"), plain, zero_bend):
                np.testing.assert_allclose(
                    actual, expected, rtol=2.0e-4, atol=2.0e-5,
                    err_msg=f"{label}: soft={soft}, points={points_mode}")
                if label != "SDF" and not np.any(np.abs(actual) > 1.0e-7):
                    raise AssertionError(
                        f"Missing {label} gradients: soft={soft}, "
                        f"points={points_mode}")
    print("straight vs zero-bend SDF and gradients: PASS")


if __name__ == "__main__":
    test_plain_matches_zero_bend()
