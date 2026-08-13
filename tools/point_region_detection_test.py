"""Regression test for exact point-sampled residual-region detection.

The thin strand is deliberately placed between all samples of the former
``grid[::f, ::f, ::f]`` detector.  Exact candidates must still find it.

Run:  .venv/Scripts/python.exe tools/point_region_detection_test.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PySide6 import QtCore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimization import OptimizationWorker  # noqa: E402


def test_strided_alias() -> bool:
    shape = (33, 129, 129)
    dx = 0.01
    origin = np.zeros(3, dtype=np.float32)
    target = np.full(shape, 0.05, dtype=np.float32)
    thickness = np.zeros(shape, dtype=np.float32)

    # Former cap: ceil(129 / 64) == 3.  y=64 and z=16 are both 1 mod 3,
    # therefore the old strided target contained no voxel from this strand.
    z, y = 16, 64
    target[z, y, 78:111] = -0.004
    thickness[z, y, 78:111] = 0.008
    old_decimation_sees_strand = bool(np.any(target[::3, ::3, ::3] < 0.0))

    pred = target.copy()
    pred[z, y, 78:111] += 0.02

    worker = OptimizationWorker(
        sdf_target_np=target,
        origin=origin,
        dx=dx,
        n=max(shape),
        num_ellipsoids=1,
        num_steps=1,
        thickness_np=thickness,
        local_fit=True,
        superfit=False,
        underrep_min_gap_vox=0.25,
        underrep_min_thickness_vox=1.0,
    )

    def _sample_pred(self, points, _c, _r, _q):
        ijk = np.floor((np.asarray(points) - origin[None, :]) / dx).astype(np.int64)
        return pred[ijk[:, 2], ijk[:, 1], ijk[:, 0]]

    worker._pred_points_from_params = types.MethodType(_sample_pred, worker)
    regions = worker._detect_worst_regions(
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.float32),
        np.empty((0, 4), np.float32),
        k=1,
    )

    expected_yz = origin + (np.array([0.0, y, z], np.float32) + 0.5) * dx
    picked = np.asarray(regions[0]["seed_world"], np.float32) if regions else None
    ok = (
        not old_decimation_sees_strand
        and picked is not None
        and abs(float(picked[1] - expected_yz[1])) < 0.5 * dx
        and abs(float(picked[2] - expected_yz[2])) < 0.5 * dx
    )

    print("old decimation sees strand:", old_decimation_sees_strand)
    print("candidate count:", len(worker._region_candidate_indices()))
    print("picked:", None if picked is None else np.round(picked, 4))
    return bool(ok)


def test_thin_and_bone_quotas() -> bool:
    shape = (40, 40, 40)
    dx = 0.01
    target = np.full(shape, -0.004, dtype=np.float32)
    thickness = np.full(shape, 0.1, dtype=np.float32)
    thin_flat = np.arange(500, dtype=np.int64) * 97 % target.size
    thickness.ravel()[thin_flat] = 0.01
    bone_centers = np.array([
        [0.05, 0.20, 0.20],
        [0.15, 0.20, 0.20],
        [0.25, 0.20, 0.20],
        [0.35, 0.20, 0.20],
    ], dtype=np.float32)

    worker = OptimizationWorker(
        sdf_target_np=target,
        origin=np.zeros(3, dtype=np.float32),
        dx=dx,
        n=max(shape),
        num_ellipsoids=1,
        num_steps=1,
        thickness_np=thickness,
        local_fit=False,
        superfit=False,
        bone_aware=True,
        bone_centers_np=bone_centers,
    )
    worker._region_candidate_budget = 1000
    candidates = worker._region_candidate_indices()
    thin_count = int(np.isin(candidates, thin_flat).sum())

    points = worker._grid_points_from_flat(candidates)
    d2 = np.sum(
        (points[:, None, :] - bone_centers[None, :, :]) ** 2,
        axis=2,
    )
    bone_counts = np.bincount(np.argmin(d2, axis=1), minlength=len(bone_centers))
    expected_thin = int(round(
        worker._region_candidate_budget * worker._region_thin_candidate_fraction))
    ok = (
        len(candidates) == worker._region_candidate_budget
        and thin_count >= expected_thin
        and np.all(bone_counts > 0)
    )
    print("thin quota:", thin_count, "/", expected_thin)
    print("bone candidate counts:", bone_counts.tolist())
    return bool(ok)


def main() -> int:
    QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv)
    alias_ok = test_strided_alias()
    quota_ok = test_thin_and_bone_quotas()
    ok = alias_ok and quota_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
