"""Focused regression tests for the differentiable superquadric distance.

Run: .venv/Scripts/python.exe tools/superquadric_distance_regression_test.py
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

from optimization import _bent_sq_distance, _sq_shape_distance  # noqa: E402


DEVICE = os.environ.get("SQ_TEST_DEVICE", "cpu")


@wp.kernel
def _eval_plain(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    eps: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    j = 2 * i
    out[i] = _sq_shape_distance(points[i], radii[i], eps[j], eps[j + 1])


@wp.kernel
def _eval_bent(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    eps: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    j = 2 * i
    out[i] = _bent_sq_distance(
        points[i],
        wp.vec3(0.0, 0.0, 0.0),
        radii[i],
        wp.quat(0.0, 0.0, 0.0, 1.0),
        eps[j], eps[j + 1], bend[j], bend[j + 1],
    )


@wp.kernel
def _mean_square(values: wp.array(dtype=wp.float32),
                 loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, values[i] * values[i] / float(values.shape[0]))


@wp.kernel
def _sum(values: wp.array(dtype=wp.float32),
         loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, values[i])


def _assert_finite(name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        bad = np.argwhere(~np.isfinite(values))
        raise AssertionError(f"{name} contains non-finite values at {bad[:8].tolist()}")


def _plain_values(points: np.ndarray, radii: np.ndarray,
                  eps: np.ndarray) -> np.ndarray:
    count = len(points)
    wp_points = wp.array(points, dtype=wp.vec3, device=DEVICE)
    wp_radii = wp.array(radii, dtype=wp.vec3, device=DEVICE)
    wp_eps = wp.array(eps.reshape(-1), dtype=wp.float32, device=DEVICE)
    out = wp.empty(count, dtype=wp.float32, device=DEVICE)
    wp.launch(
        _eval_plain, dim=count,
        inputs=[wp_points, wp_radii, wp_eps, out], device=DEVICE,
    )
    return out.numpy().copy()


def test_finite_values_and_gradients() -> None:
    points = np.array([
        [0.0, 0.0, 0.0],             # non-differentiable gauge centre
        [1.7, 0.0, 0.0],             # x axis / surface
        [0.0, 0.8, 0.0],             # y axis / surface
        [0.0, 0.0, 0.3],             # z axis / surface
        [1.0e-12, -2.0e-12, 0.1],    # immediately beside two axes
        [1000.0, -500.0, 250.0],     # far field that overflowed nested pow
        [-0.4, 0.2, -0.1],
        [2.5, -1.5, 0.05],
    ], dtype=np.float32)
    radii = np.tile(
        np.array([[1.7, 0.8, 0.3]], dtype=np.float32), (len(points), 1))
    eps = np.array([
        [0.1, 0.1],
        [0.1, 2.0],
        [2.0, 0.1],
        [2.0, 2.0],
        [0.1, 2.0],
        [2.0, 0.1],
        [1.0, 1.0],
        [0.2, 1.8],
    ], dtype=np.float32)
    bends = np.array([
        [0.0, 0.0],
        [6.0, -6.0],
        [-6.0, 6.0],
        [3.0, -2.0],
        [-1.0, 4.0],
        [6.0, 6.0],
        [-3.0, 2.0],
        [0.5, -0.75],
    ], dtype=np.float32)
    count = len(points)

    def evaluate(kernel, include_bend: bool) -> None:
        wp_points = wp.array(
            points, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        wp_radii = wp.array(
            radii, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        wp_eps = wp.array(
            eps.reshape(-1), dtype=wp.float32, device=DEVICE, requires_grad=True)
        out = wp.empty(
            count, dtype=wp.float32, device=DEVICE, requires_grad=True)
        loss = wp.zeros(
            1, dtype=wp.float32, device=DEVICE, requires_grad=True)
        wp_bend = None

        tape = wp.Tape()
        with tape:
            if include_bend:
                wp_bend = wp.array(
                    bends.reshape(-1), dtype=wp.float32,
                    device=DEVICE, requires_grad=True)
                wp.launch(
                    kernel, dim=count,
                    inputs=[wp_points, wp_radii, wp_eps, wp_bend, out],
                    device=DEVICE,
                )
            else:
                wp.launch(
                    kernel, dim=count,
                    inputs=[wp_points, wp_radii, wp_eps, out], device=DEVICE,
                )
            wp.launch(_mean_square, dim=count, inputs=[out, loss], device=DEVICE)
        tape.backward(loss)

        label = "bent" if include_bend else "plain"
        _assert_finite(f"{label} distance", out.numpy())
        _assert_finite(f"{label} point gradient", wp_points.grad.numpy())
        _assert_finite(f"{label} radius gradient", wp_radii.grad.numpy())
        _assert_finite(f"{label} epsilon gradient", wp_eps.grad.numpy())
        if wp_bend is not None:
            _assert_finite("bend gradient", wp_bend.grad.numpy())

    evaluate(_eval_plain, include_bend=False)
    evaluate(_eval_bent, include_bend=True)
    print("finite values/gradients: PASS")


def test_epsilon_one_matches_ellipsoid_formula() -> None:
    radii_one = np.array([1.7, 0.8, 0.3], dtype=np.float32)
    points = np.array([
        [2.2, 0.1, 0.05],
        [0.4, 1.3, -0.2],
        [-1.8, 0.9, 0.7],
        [0.2, -0.2, 0.8],
    ], dtype=np.float32)
    radii = np.tile(radii_one[None, :], (len(points), 1))
    eps = np.ones((len(points), 2), dtype=np.float32)
    actual = _plain_values(points, radii, eps)

    k0 = np.linalg.norm(points / radii, axis=1)
    if not np.all(k0 > 1.0):
        raise AssertionError("ellipsoid comparison points must be exterior")
    k1 = np.linalg.norm(points / (radii * radii), axis=1)
    expected = k0 * (k0 - 1.0) / np.maximum(k1, 1.0e-8)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-6)
    print(f"epsilon=1 ellipsoid max error: {np.max(np.abs(actual - expected)):.3e}")


def test_surface_gradient_and_bend_jacobian() -> None:
    radii_one = np.array([1.4, 0.9, 0.45], dtype=np.float32)
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.7, 0.25],
        [-0.4, 0.8, -0.6],
    ], dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    beta = np.linalg.norm(directions / radii_one[None, :], axis=1)
    unbent_surface = directions / beta[:, None]
    count = len(unbent_surface)
    radii = np.tile(radii_one[None, :], (count, 1)).astype(np.float32)
    eps = np.ones((count, 2), dtype=np.float32)
    bend = np.tile(np.array([[1.3, -0.8]], dtype=np.float32), (count, 1))

    # Forward warp corresponding to the inverse used by _bent_sq_distance.
    z = unbent_surface[:, 2]
    bent_surface = unbent_surface.copy()
    bent_surface[:, 0] += 0.5 * bend[:, 0] * z * z
    bent_surface[:, 1] += 0.5 * bend[:, 1] * z * z

    def surface_eval(kernel, points: np.ndarray, include_bend: bool) -> None:
        wp_points = wp.array(
            points, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        wp_radii = wp.array(radii, dtype=wp.vec3, device=DEVICE)
        wp_eps = wp.array(eps.reshape(-1), dtype=wp.float32, device=DEVICE)
        out = wp.empty(
            count, dtype=wp.float32, device=DEVICE, requires_grad=True)
        loss = wp.zeros(
            1, dtype=wp.float32, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            if include_bend:
                wp_bend = wp.array(
                    bend.reshape(-1), dtype=wp.float32, device=DEVICE)
                wp.launch(
                    kernel, dim=count,
                    inputs=[wp_points, wp_radii, wp_eps, wp_bend, out],
                    device=DEVICE,
                )
            else:
                wp.launch(
                    kernel, dim=count,
                    inputs=[wp_points, wp_radii, wp_eps, out], device=DEVICE,
                )
            wp.launch(_sum, dim=count, inputs=[out, loss], device=DEVICE)
        tape.backward(loss)

        np.testing.assert_allclose(out.numpy(), 0.0, atol=3.0e-6)
        grad_norm = np.linalg.norm(wp_points.grad.numpy(), axis=1)
        np.testing.assert_allclose(grad_norm, 1.0, rtol=5.0e-4, atol=5.0e-4)

    surface_eval(_eval_plain, unbent_surface, include_bend=False)
    surface_eval(_eval_bent, bent_surface, include_bend=True)

    zero_bend = np.zeros_like(bend)
    wp_points = wp.array(unbent_surface, dtype=wp.vec3, device=DEVICE)
    wp_radii = wp.array(radii, dtype=wp.vec3, device=DEVICE)
    wp_eps = wp.array(eps.reshape(-1), dtype=wp.float32, device=DEVICE)
    wp_bend = wp.array(zero_bend.reshape(-1), dtype=wp.float32, device=DEVICE)
    plain = wp.empty(count, dtype=wp.float32, device=DEVICE)
    bent = wp.empty(count, dtype=wp.float32, device=DEVICE)
    wp.launch(
        _eval_plain, dim=count,
        inputs=[wp_points, wp_radii, wp_eps, plain], device=DEVICE)
    wp.launch(
        _eval_bent, dim=count,
        inputs=[wp_points, wp_radii, wp_eps, wp_bend, bent], device=DEVICE)
    np.testing.assert_allclose(bent.numpy(), plain.numpy(), rtol=0.0, atol=1.0e-7)
    print("surface gradient / bend Jacobian: PASS")


def main() -> None:
    wp.init()
    test_finite_values_and_gradients()
    test_epsilon_one_matches_ellipsoid_formula()
    test_surface_gradient_and_bend_jacobian()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
