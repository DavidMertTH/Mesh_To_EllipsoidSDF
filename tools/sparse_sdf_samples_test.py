"""Headless regression test for the sparse/sample SDF training API."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids  # noqa: E402
from ellipsoid import best_device  # noqa: E402
from mesh_io import load_and_prepare  # noqa: E402
from optimization import BandSampler, OptimizationWorker  # noqa: E402
from pose_correctives import PoseCorrectiveWorker  # noqa: E402
from rig_loader import load_rigged_mesh  # noqa: E402
from sdf_compute import SdfComputer  # noqa: E402
from sdf_samples import SdfSampleSet  # noqa: E402


def test_sparse_samples() -> None:
    mesh = load_and_prepare(ROOT / "meshes" / "bunny.obj")
    sdf = SdfComputer(device=best_device())
    sdf.set_mesh(mesh.vertices, mesh.faces)
    samples = sdf.compute_sparse_samples(
        n=32,
        margin=0.1,
        surface_samples=1024,
        offsets_vox=(-2.0, 0.0, 2.0),
        coarse_n=8,
    )
    assert samples.size > 0
    assert samples.points.shape == (samples.size, 3)
    assert np.isfinite(samples.values).all()
    assert samples.coarse_mask is not None
    assert samples.coarse_mask.shape == (samples.size,)
    assert 0 < int(samples.coarse_mask.sum()) <= 8 ** 3
    assert np.array_equal(samples.with_offset(0.1).coarse_mask, samples.coarse_mask)
    print("sparse samples:", samples.size)


def test_sparse_far_field_quota() -> None:
    target = np.concatenate([
        np.zeros(1000, dtype=np.float32),
        np.ones(20, dtype=np.float32),
    ])
    coarse = np.zeros(target.size, dtype=np.bool_)
    coarse[-20:] = True
    sampler = BandSampler(
        target,
        batch_size=100,
        band=0.1,
        surface_fraction=0.75,
        rng=np.random.default_rng(7),
        coarse_mask=coarse,
    )
    batch = sampler.next_batch()
    assert sampler.n_far == 20
    assert np.count_nonzero(batch >= 1000) >= 20
    print("sparse far-field quota: ok")


def test_optimizer_sparse_path() -> None:
    mesh = load_and_prepare(ROOT / "meshes" / "bunny.obj")
    sdf = SdfComputer(device=best_device())
    sdf.set_mesh(mesh.vertices, mesh.faces)
    dense = sdf.compute_voxel_grid(n=16, margin=0.1, compute_thickness=False)
    samples = sdf.compute_sparse_samples(
        n=16,
        margin=0.1,
        surface_samples=512,
        offsets_vox=(-2.0, 0.0, 2.0),
        coarse_n=6,
    )
    worker = OptimizationWorker(
        sdf_target_np=dense.grid,
        sdf_samples=samples,
        origin=dense.origin,
        dx=dense.dx,
        n=dense.n,
        num_ellipsoids=2,
        num_steps=3,
        report_every=1,
        sample_budget=256,
        maintenance_every=0,
        superfit=False,
        local_fit=False,
    )
    worker.run()
    print("optimizer sparse path: ok")


def test_sparse_symmetry_pairs_values_and_thickness() -> None:
    samples = SdfSampleSet(
        points=np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float32),
        values=np.array([20.0, 10.0, 0.0, 1.0, 2.0], dtype=np.float32),
        thickness=np.array([8.0, 6.0, 1.0, 2.0, 4.0], dtype=np.float32),
        dx=0.25,
        source="pair-test",
        coarse_mask=np.array([True, False, True, False, True]),
    )
    paired = OptimizationWorker._paired_symmetric_samples(
        samples, axis=0, plane=0.0, tolerance=1.0e-6)

    expected_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
    ], dtype=np.float32)
    np.testing.assert_array_equal(paired.points, expected_points)
    np.testing.assert_array_equal(
        paired.values, np.array([0.0, 1.0, 2.0, 1.0, 2.0],
                                dtype=np.float32))
    np.testing.assert_array_equal(
        paired.thickness, np.array([1.0, 2.0, 4.0, 2.0, 4.0],
                                   dtype=np.float32))
    np.testing.assert_array_equal(
        paired.coarse_mask,
        np.array([True, False, True, False, True]),
    )
    assert paired.source == "pair-test-symmetric"
    print("sparse symmetry pairs: ok")


def test_sparse_symmetry_optimizer_uses_resized_pair_set() -> None:
    n = 8
    dx = 0.25
    origin = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    coords = origin[0] + (
        np.arange(n, dtype=np.float32) + np.float32(0.5)) * dx
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    dense = (np.sqrt(x * x + y * y + z * z) - 0.65).astype(np.float32)

    # Deliberately imbalanced: the deterministic +X source side has five
    # samples, while -X has twenty. Pairing therefore shrinks 25 -> 10. This
    # catches stale pre-symmetry lengths/masks and out-of-range GPU indices.
    negative = [
        [xv, yv, 0.0]
        for xv in (-0.875, -0.625, -0.375, -0.125)
        for yv in (-0.75, -0.25, 0.25, 0.75)
    ]
    negative.extend([
        [-0.75, 0.0, 0.5],
        [-0.50, 0.0, 0.5],
        [-0.25, 0.0, 0.5],
        [-0.10, 0.0, 0.5],
    ])
    positive = [
        [0.125, -0.50, 0.0],
        [0.250, -0.25, 0.25],
        [0.375, 0.00, 0.0],
        [0.625, 0.25, -0.25],
        [0.875, 0.50, 0.0],
    ]
    points = np.asarray(negative + positive, dtype=np.float32)
    samples = SdfSampleSet(
        points=points,
        values=(np.linalg.norm(points, axis=1) - 0.65).astype(np.float32),
        thickness=np.full(points.shape[0], 0.4, dtype=np.float32),
        dx=dx,
        source="imbalanced-pair-test",
        coarse_mask=(np.arange(points.shape[0]) % 3 == 0),
    )
    worker = OptimizationWorker(
        sdf_target_np=dense,
        sdf_samples=samples,
        origin=origin,
        dx=dx,
        n=n,
        num_ellipsoids=1,
        num_steps=1,
        report_every=1,
        sample_budget=64,
        maintenance_every=0,
        superfit=False,
        local_fit=False,
        symmetry_enabled=True,
        initial_centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        initial_radii=np.array([[0.4, 0.4, 0.4]], dtype=np.float32),
        initial_rotations=np.array(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )
    worker._detect_symmetry_axis = lambda _grid: (0, 0.0)
    worker._reset_stale_tape()
    worker._run_adam()

    assert samples.size == 25
    assert worker._sdf_samples.size == 10
    assert worker._uploaded_samples.samples is worker._sdf_samples
    assert worker._batch_size == 10
    print("sparse symmetry resized optimizer target: ok")


def test_pose_corrective_fixed_id_path() -> None:
    rigged = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")
    mapper = BoneEllipsoidMapper(rigged.skeleton)
    bone_local = BoneLocalEllipsoids(
        local_centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        local_radii=np.array([[0.15, 0.15, 0.15]], dtype=np.float32),
        local_rotations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        bone_assignments=np.array([0], dtype=np.int32),
    )
    worker = PoseCorrectiveWorker(
        rigged_mesh=rigged,
        mapper=mapper,
        base=bone_local,
        poses=rigged.poses[:1],
        fit_kwargs={
            "num_steps": 1,
            "report_every": 1,
            "sample_budget": 128,
            "maintenance_every": 0,
            "superfit": False,
            "local_fit": False,
        },
        grid_n=12,
        margin=0.1,
    )
    worker.run()
    assert worker.result is not None
    assert worker.result.base.num_ellipsoids == bone_local.num_ellipsoids
    assert len(worker.result.keys) == 1
    assert worker.result.keys[0].delta_centers.shape == bone_local.local_centers.shape
    print("pose corrective fixed IDs: ok")


def main() -> None:
    test_sparse_samples()
    test_sparse_far_field_quota()
    test_optimizer_sparse_path()
    test_sparse_symmetry_pairs_values_and_thickness()
    test_sparse_symmetry_optimizer_uses_resized_pair_set()
    test_pose_corrective_fixed_id_path()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
