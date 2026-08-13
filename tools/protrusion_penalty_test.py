"""Regression test for the quadratic ellipsoid protrusion penalty.

Run: .venv/Scripts/python.exe tools/protrusion_penalty_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batched_fit import _grouped_loss_kernel  # noqa: E402
from optimization import _rmse_loss_kernel_batch  # noqa: E402


DEVICE = "cpu"


def _main_fitter_increment(target: float, prediction: float, sigma: float) -> float:
    sdf_pred = wp.array([prediction], dtype=wp.float32, device=DEVICE)
    sdf_target = wp.array([target], dtype=wp.float32, device=DEVICE)
    indices = wp.array([0], dtype=wp.int32, device=DEVICE)
    thickness = wp.zeros(1, dtype=wp.float32, device=DEVICE)

    def evaluate(outside_weight: float) -> float:
        loss = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        wp.launch(
            _rmse_loss_kernel_batch,
            dim=1,
            inputs=[
                sdf_pred,
                sdf_target,
                indices,
                loss,
                1,
                0.0,
                0.0,
                sigma,
                outside_weight,
                thickness,
                1.0,
                0.0,
                1.0,
                0.1,
            ],
            device=DEVICE,
        )
        return float(loss.numpy()[0])

    return evaluate(1.0) - evaluate(0.0)


def _grouped_fitter_increment(target: float, prediction: float, sigma: float) -> float:
    sdf_pred = wp.array([prediction], dtype=wp.float32, device=DEVICE)
    sample_idx = wp.array([0], dtype=wp.int32, device=DEVICE)
    sample_grp = wp.array([0], dtype=wp.int32, device=DEVICE)
    target_pool = wp.array([target], dtype=wp.float32, device=DEVICE)
    thick_pool = wp.zeros(1, dtype=wp.float32, device=DEVICE)
    grp_sigma = wp.array([sigma], dtype=wp.float32, device=DEVICE)
    grp_thickref = wp.zeros(1, dtype=wp.float32, device=DEVICE)

    def evaluate(outside_weight: float) -> float:
        loss = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        wp.launch(
            _grouped_loss_kernel,
            dim=1,
            inputs=[
                sdf_pred,
                sample_idx,
                sample_grp,
                target_pool,
                thick_pool,
                grp_sigma,
                grp_thickref,
                loss,
                1.0,
                0.0,
                0.0,
                outside_weight,
                0.0,
                1.0,
            ],
            device=DEVICE,
        )
        return float(loss.numpy()[0])

    return evaluate(1.0) - evaluate(0.0)


def _check_kernel(name: str, evaluate) -> None:
    sigma = 0.1
    near_distance = 0.05
    far_distance = 0.20

    near = evaluate(0.02, -0.03, sigma)
    far = evaluate(0.08, -0.12, sigma)

    expected_near = near_distance**2 / sigma
    expected_far = far_distance**2 / sigma
    np.testing.assert_allclose(near, expected_near, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(far, expected_far, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(far / near, 16.0, rtol=1.0e-5, atol=1.0e-5)
    print(f"{name}: near={near:.6f}, far={far:.6f}, ratio={far / near:.1f}")


def main() -> None:
    wp.init()
    _check_kernel("main fitter", _main_fitter_increment)
    _check_kernel("grouped fitter", _grouped_fitter_increment)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
