"""Deterministic regression coverage for sparse local-thickness samples."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdf_compute import SdfComputer, _sample_voxel_field_trilinear  # noqa: E402


def _box_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Closed 2 x 1 x 1 box with consistently outward-facing triangles."""
    vertices = np.asarray([
        [-1.0, -0.5, -0.5], [1.0, -0.5, -0.5],
        [1.0, 0.5, -0.5], [-1.0, 0.5, -0.5],
        [-1.0, -0.5, 0.5], [1.0, -0.5, 0.5],
        [1.0, 0.5, 0.5], [-1.0, 0.5, 0.5],
    ], dtype=np.float32)
    faces = np.asarray([
        [0, 2, 1], [0, 3, 2],       # -z
        [4, 5, 6], [4, 6, 7],       # +z
        [0, 1, 5], [0, 5, 4],       # -y
        [3, 7, 6], [3, 6, 2],       # +y
        [0, 4, 7], [0, 7, 3],       # -x
        [1, 2, 6], [1, 6, 5],       # +x
    ], dtype=np.int32)
    return vertices, faces


def test_voxel_centered_trilinear_sampling() -> None:
    z, y, x = np.meshgrid(
        np.arange(3), np.arange(4), np.arange(5), indexing="ij",
    )
    field = (x + 10.0 * y + 100.0 * z).astype(np.float32)
    origin = np.asarray([-2.0, 3.0, 7.0], dtype=np.float32)
    spacing = 2.0
    # First point is voxel (x=2,y=1,z=1); second is halfway to (3,2,2).
    points = np.asarray([
        origin + spacing * np.asarray([2.5, 1.5, 1.5]),
        origin + spacing * np.asarray([3.0, 2.0, 2.0]),
        origin + spacing * np.asarray([-1.0, 0.5, 0.5]),
    ], dtype=np.float32)
    got = _sample_voxel_field_trilinear(field, origin, spacing, points)
    np.testing.assert_allclose(got, [112.0, 167.5, 0.0], atol=1.0e-6)


def test_sparse_thickness_from_dense_source() -> None:
    vertices, faces = _box_mesh()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    dense = sdf.compute_voxel_grid(
        n=28,
        margin=0.25,
        compute_thickness=True,
        compute_blowup_thickness=True,
        thickness_max_resolution=None,
    )
    surface_samples = 512
    offsets = (-2.0, 0.0, 2.0)
    sparse = sdf.compute_sparse_samples(
        n=28,
        margin=0.25,
        surface_samples=surface_samples,
        offsets_vox=offsets,
        coarse_n=8,
        seed=7,
        thickness_result=dense,
    )
    assert sparse.thickness is not None
    assert sparse.thickness.shape == (sparse.size,)
    assert np.isfinite(sparse.thickness).all()
    assert dense.blowup_thickness is not None
    band = sparse.thickness[:surface_samples * len(offsets)]
    positive_ratio = float(np.mean(band > 0.0))
    assert positive_ratio > 0.95, positive_ratio
    base_points = sparse.points[surface_samples:2 * surface_samples]
    expected_base = _sample_voxel_field_trilinear(
        dense.blowup_thickness,
        dense.origin,
        dense.dx,
        base_points,
    )
    np.testing.assert_allclose(
        sparse.thickness[surface_samples:2 * surface_samples],
        expected_base,
        atol=1.0e-6,
    )
    median = float(np.median(band[band > 0.0]))
    # Sharp box edges have smaller local thickness than the one-unit face
    # interior, so the complete surface distribution spans both scales.
    assert 0.10 < median < 1.25, median


def test_sparse_only_thickness_fallback() -> None:
    vertices, faces = _box_mesh()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    surface_samples = 256
    sparse = sdf.compute_sparse_samples(
        n=24,
        margin=0.25,
        surface_samples=surface_samples,
        offsets_vox=(-1.0, 0.0, 1.0),
        coarse_n=12,
        seed=11,
    )
    band = sparse.thickness[:3 * surface_samples]
    assert np.isfinite(band).all()
    assert float(np.mean(band > 0.0)) > 0.75


def main() -> None:
    test_voxel_centered_trilinear_sampling()
    test_sparse_thickness_from_dense_source()
    test_sparse_only_thickness_fallback()
    print("sparse thickness regression: PASS")


if __name__ == "__main__":
    main()
