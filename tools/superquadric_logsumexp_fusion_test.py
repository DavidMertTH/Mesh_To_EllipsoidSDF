"""Regression check for the fused superquadric log-sum-exp/weight helper.

Run: .venv/Scripts/python.exe -B tools/superquadric_logsumexp_fusion_test.py
Set SQ_TEST_DEVICE=cuda:0 to exercise the GPU implementation.
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
    _sq_log_beta_grad,
    _sq_log_safe_ratio,
    _sq_logaddexp_weights,
    _sq_normalized_distance,
)


@wp.func
def _legacy_logaddexp(a: float, b: float) -> float:
    m = wp.max(a, b)
    return m + wp.log(wp.exp(a - m) + wp.exp(b - m))


@wp.func
def _legacy_log_beta_grad(lp: wp.vec3, r: wp.vec3,
                          e1: float, e2: float) -> wp.vec4:
    """Pre-fusion implementation, retained as an AD parity oracle."""
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
    la = _legacy_logaddexp(lx, ly)
    lxy = (se2 / se1) * la
    lf = _legacy_logaddexp(lxy, lz)
    log_beta = 0.5 * se1 * lf

    wx = wp.exp(lx - la)
    wy = wp.exp(ly - la)
    wxy = wp.exp(lxy - lf)
    wz = wp.exp(lz - lf)
    gx = wxy * wx * (lp[0] / sx) / sx
    gy = wxy * wy * (lp[1] / sy) / sy
    gz = wz * (lp[2] / sz) / sz
    return wp.vec4(log_beta, gx, gy, gz)


@wp.kernel
def _evaluate_log_ratio(
    numerator: wp.array(dtype=wp.float32),
    denominator: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    output[i] = _sq_log_safe_ratio(numerator[i], denominator[i])


@wp.kernel
def _evaluate_shape(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    eps: wp.array(dtype=wp.float32),
    data_out: wp.array(dtype=wp.vec4),
    distance_out: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    legacy: bool,
):
    i = wp.tid()
    j = 2 * i
    data = wp.vec4()
    if legacy:
        data = _legacy_log_beta_grad(points[i], radii[i], eps[j], eps[j + 1])
    else:
        data = _sq_log_beta_grad(points[i], radii[i], eps[j], eps[j + 1])
    data_out[i] = data
    d = _sq_normalized_distance(
        data[0], wp.vec3(data[1], data[2], data[3]), radii[i])
    distance_out[i] = d
    # Seed all components, plus the actual distance used in fitting.
    wp.atomic_add(loss, 0, d + 0.125 * data[0] + 0.0625 * data[1]
                  - 0.03125 * data[2] + 0.015625 * data[3])


@wp.kernel
def _evaluate(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    result = _sq_logaddexp_weights(a[i], b[i])
    values[i] = result
    wp.atomic_add(loss, 0, result[0])


def main() -> None:
    wp.init()
    device = os.environ.get("SQ_TEST_DEVICE", "cpu")
    numerator = np.array([1.0e-15, 0.5, 1.0, 1.0e35], dtype=np.float32)
    denominator = np.array([1.0e-8, 1.0, 0.5, 1.0e-8], dtype=np.float32)
    ratio_out = wp.empty(len(numerator), dtype=wp.float32, device=device)
    wp.launch(_evaluate_log_ratio, dim=len(numerator), inputs=[
        wp.array(numerator, device=device),
        wp.array(denominator, device=device),
        ratio_out,
    ], device=device)
    np.testing.assert_allclose(
        ratio_out.numpy(),
        np.log(numerator.astype(np.float64)) - np.log(denominator.astype(np.float64)),
        rtol=2e-6, atol=2e-6,
    )
    print(f"safe log-ratio normal and overflow paths ({device}): PASS")

    rng = np.random.default_rng(5471)
    a = rng.uniform(-100.0, 100.0, 512).astype(np.float32)
    b = rng.uniform(-100.0, 100.0, 512).astype(np.float32)
    a[:6] = np.array([-100.0, -20.0, 0.0, 1.0, 80.0, 100.0])
    b[:6] = np.array([100.0, 20.0, 0.0, 1.0, -80.0, -100.0])

    wp_a = wp.array(a, dtype=wp.float32, device=device, requires_grad=True)
    wp_b = wp.array(b, dtype=wp.float32, device=device, requires_grad=True)
    values = wp.empty(len(a), dtype=wp.vec3, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(_evaluate, dim=len(a), inputs=[wp_a, wp_b, values, loss], device=device)
    tape.backward(loss)

    # The former implementation computed each mixture weight with a second exp.
    m = np.maximum(a, b)
    logsum = m + np.log(np.exp(a - m) + np.exp(b - m))
    old_weights = np.stack((np.exp(a - logsum), np.exp(b - logsum)), axis=1)
    actual = values.numpy()
    np.testing.assert_allclose(actual[:, 0], logsum, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[:, 1:], old_weights, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(wp_a.grad.numpy(), actual[:, 1], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(wp_b.grad.numpy(), actual[:, 2], rtol=2e-5, atol=2e-6)
    print(f"log-sum-exp values, weights and gradients ({device}): PASS")

    points = rng.normal(size=(512, 3)).astype(np.float32)
    points[:8] = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0],
        [1.0e-12, 0.0, 0.1], [1000.0, -500.0, 250.0],
    ], dtype=np.float32)
    radii = np.exp(rng.uniform(-2.0, 1.0, size=(512, 3))).astype(np.float32)
    radii[4] = 1.0  # equal generalized powers / log-sum-exp ties
    eps = rng.uniform(0.1, 2.0, size=(512, 2)).astype(np.float32)
    eps[:8] = np.array([
        [0.1, 0.1], [0.1, 2.0], [2.0, 0.1], [2.0, 2.0],
        [1.0, 1.0], [0.1001, 1.9999], [1.9999, 0.1001], [1.0, 1.0],
    ], dtype=np.float32)

    def evaluate_shape(legacy: bool):
        wp_points = wp.array(points, dtype=wp.vec3, device=device, requires_grad=True)
        wp_radii = wp.array(radii, dtype=wp.vec3, device=device, requires_grad=True)
        wp_eps = wp.array(eps.reshape(-1), dtype=wp.float32,
                          device=device, requires_grad=True)
        data_out = wp.empty(len(points), dtype=wp.vec4,
                            device=device, requires_grad=True)
        distance_out = wp.empty(len(points), dtype=wp.float32,
                                device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(_evaluate_shape, dim=len(points),
                      inputs=[wp_points, wp_radii, wp_eps, data_out,
                              distance_out, loss, legacy], device=device)
        tape.backward(loss)
        return (data_out.numpy(), distance_out.numpy(),
                wp_points.grad.numpy(), wp_radii.grad.numpy(), wp_eps.grad.numpy())

    reference = evaluate_shape(True)
    fused = evaluate_shape(False)
    for name, gradient in zip(("point", "radius", "epsilon"), fused[2:]):
        if not np.all(np.isfinite(gradient)) or not np.any(np.abs(gradient) > 1.0e-8):
            raise AssertionError(f"{device}: missing/non-finite {name} gradient")
    for name, old, new in zip(
        ("log-beta/gradient", "distance", "point AD", "radius AD", "epsilon AD"),
        reference, fused,
    ):
        np.testing.assert_allclose(new, old, rtol=2e-4, atol=2e-5,
                                   err_msg=f"{device}: {name}")
    print(f"full superquadric values and parameter gradients ({device}): PASS")


if __name__ == "__main__":
    main()
