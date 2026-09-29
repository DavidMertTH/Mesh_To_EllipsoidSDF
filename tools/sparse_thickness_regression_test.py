"""Deterministic regression coverage for sparse local-thickness samples."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sdf_compute as sdf_compute_module  # noqa: E402
from sdf_compute import (  # noqa: E402
    SdfComputer,
    SdfResult,
    _sample_voxel_field_trilinear,
    sample_thickness_at_face_corners,
    sample_thickness_at_vertices,
)
from thickness import dilate_zeros  # noqa: E402


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


def test_vertex_thickness_is_barycentrically_transported() -> None:
    vertices, faces = _box_mesh()
    # An affine scalar field is reproduced exactly by barycentric interpolation.
    vertex_thickness = (
        2.0
        + 0.20 * vertices[:, 0]
        + 0.30 * vertices[:, 1]
        + 0.40 * vertices[:, 2]
    ).astype(np.float32)
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    surface_samples = 384
    offsets = (-3.0, 0.0, 2.0)

    # Prove that this path never falls back to the expensive voxel operation.
    def fail_local_thickness(*_args, **_kwargs):
        raise AssertionError("vertex-thickness path called local_thickness")

    original_local_thickness = sdf_compute_module.local_thickness
    sdf_compute_module.local_thickness = fail_local_thickness
    try:
        sparse = sdf.compute_sparse_samples(
            n=24,
            margin=0.25,
            surface_samples=surface_samples,
            offsets_vox=offsets,
            coarse_n=8,
            seed=19,
            vertex_thickness=vertex_thickness,
        )
    finally:
        sdf_compute_module.local_thickness = original_local_thickness

    base_points = sparse.points[surface_samples:2 * surface_samples]
    expected = (
        2.0
        + 0.20 * base_points[:, 0]
        + 0.30 * base_points[:, 1]
        + 0.40 * base_points[:, 2]
    )
    for band_i in range(len(offsets)):
        start = band_i * surface_samples
        np.testing.assert_allclose(
            sparse.thickness[start:start + surface_samples],
            expected,
            atol=2.0e-6,
        )
    np.testing.assert_array_equal(
        sparse.thickness[surface_samples * len(offsets):],
        0.0,
    )


def test_face_corner_thickness_is_barycentrically_transported() -> None:
    vertices, faces = _box_mesh()
    triangle_vertices = vertices[faces]
    face_corner_thickness = (
        2.0
        + 0.20 * triangle_vertices[:, :, 0]
        + 0.30 * triangle_vertices[:, :, 1]
        + 0.40 * triangle_vertices[:, :, 2]
    ).astype(np.float32)
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    surface_samples = 384
    offsets = (-3.0, 0.0, 2.0)

    def fail_local_thickness(*_args, **_kwargs):
        raise AssertionError("face-corner path called local_thickness")

    original_local_thickness = sdf_compute_module.local_thickness
    sdf_compute_module.local_thickness = fail_local_thickness
    try:
        sparse = sdf.compute_sparse_samples(
            n=24,
            margin=0.25,
            surface_samples=surface_samples,
            offsets_vox=offsets,
            coarse_n=8,
            seed=19,
            face_corner_thickness=face_corner_thickness,
        )
    finally:
        sdf_compute_module.local_thickness = original_local_thickness

    base_points = sparse.points[surface_samples:2 * surface_samples]
    expected = (
        2.0
        + 0.20 * base_points[:, 0]
        + 0.30 * base_points[:, 1]
        + 0.40 * base_points[:, 2]
    )
    for band_i in range(len(offsets)):
        start = band_i * surface_samples
        np.testing.assert_allclose(
            sparse.thickness[start:start + surface_samples],
            expected,
            atol=2.0e-6,
        )
    # Without a dense blowup carrier the coarse lattice is deliberately only
    # an occupancy constraint and receives no surface feature weighting.
    np.testing.assert_array_equal(
        sparse.thickness[surface_samples * len(offsets):],
        0.0,
    )


def test_sparse_face_discontinuity_does_not_cross_shared_vertices() -> None:
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    face_corner_thickness = np.asarray(
        [[1.0, 1.0, 1.0], [7.0, 7.0, 7.0]], dtype=np.float32)
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    surface_samples = 2048
    sparse = sdf.compute_sparse_samples(
        n=8,
        margin=0.1,
        surface_samples=surface_samples,
        offsets_vox=(0.0,),
        coarse_n=4,
        seed=5,
        face_corner_thickness=face_corner_thickness,
    )

    surface_points = sparse.points[:surface_samples]
    surface_thickness = sparse.thickness[:surface_samples]
    first_face = surface_points[:, 0] > surface_points[:, 1] + 0.05
    second_face = surface_points[:, 1] > surface_points[:, 0] + 0.05
    assert np.any(first_face) and np.any(second_face)
    np.testing.assert_allclose(
        surface_thickness[first_face], 1.0, atol=1.0e-5)
    np.testing.assert_allclose(
        surface_thickness[second_face], 7.0, atol=1.0e-5)
    np.testing.assert_array_equal(
        sparse.thickness[surface_samples:], 0.0)


def _assert_invalid_vertex_thickness(values: np.ndarray, message: str) -> None:
    vertices, faces = _box_mesh()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    try:
        sdf.compute_sparse_samples(
            n=8,
            surface_samples=8,
            coarse_n=4,
            vertex_thickness=values,
        )
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("invalid vertex_thickness was accepted")


def test_vertex_thickness_validation() -> None:
    _assert_invalid_vertex_thickness(
        np.ones(7, dtype=np.float32), "one value per mesh vertex")
    nonfinite = np.ones(8, dtype=np.float32)
    nonfinite[3] = np.nan
    _assert_invalid_vertex_thickness(nonfinite, "finite")
    negative = np.ones(8, dtype=np.float32)
    negative[4] = -0.1
    _assert_invalid_vertex_thickness(negative, "nonnegative")


def _assert_invalid_face_corner_thickness(
    values: np.ndarray,
    message: str,
) -> None:
    vertices, faces = _box_mesh()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    try:
        sdf.compute_sparse_samples(
            n=8,
            surface_samples=8,
            coarse_n=4,
            face_corner_thickness=values,
        )
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("invalid face_corner_thickness was accepted")


def test_face_corner_thickness_validation() -> None:
    vertices, faces = _box_mesh()
    valid = np.ones(faces.shape, dtype=np.float32)
    _assert_invalid_face_corner_thickness(
        valid[:-1], "three values per mesh face")
    nonfinite = valid.copy()
    nonfinite[3, 1] = np.nan
    _assert_invalid_face_corner_thickness(nonfinite, "finite")
    negative = valid.copy()
    negative[4, 2] = -0.1
    _assert_invalid_face_corner_thickness(negative, "nonnegative")

    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    try:
        sdf.compute_sparse_samples(
            n=8,
            surface_samples=8,
            coarse_n=4,
            vertex_thickness=np.ones(len(vertices), dtype=np.float32),
            face_corner_thickness=valid,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc), str(exc)
    else:
        raise AssertionError("both transported representations were accepted")


def test_sample_thickness_at_vertices_prefers_surface_carrier() -> None:
    z, y, x = np.meshgrid(
        np.arange(2), np.arange(2), np.arange(2), indexing="ij",
    )
    carried = (x + 10.0 * y + 100.0 * z).astype(np.float32)
    result = SdfResult(
        grid=np.zeros((2, 2, 2), dtype=np.float32),
        n=2,
        dx=1.0,
        origin=np.zeros(3, dtype=np.float32),
        aabb_min=np.zeros(3, dtype=np.float32),
        aabb_max=np.full(3, 2.0, dtype=np.float32),
        thickness=np.full((2, 2, 2), 999.0, dtype=np.float32),
        blowup_thickness=carried,
    )
    vertices = np.asarray([
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5],
    ], dtype=np.float32)
    got = sample_thickness_at_vertices(result, vertices)
    np.testing.assert_allclose(got, [0.0, 55.5, 111.0], atol=1.0e-6)


def test_sample_thickness_at_vertices_dilates_legacy_field() -> None:
    local = np.zeros((5, 5, 5), dtype=np.float32)
    local[2, 2, 2] = 3.5
    result = SdfResult(
        grid=np.zeros_like(local),
        n=5,
        dx=1.0,
        origin=np.zeros(3, dtype=np.float32),
        aabb_min=np.zeros(3, dtype=np.float32),
        aabb_max=np.full(3, 5.0, dtype=np.float32),
        thickness=local,
    )
    vertices = np.asarray([
        [2.5, 2.5, 2.5],
        [0.5, 2.5, 2.5],
        [0.5, 0.5, 0.5],
    ], dtype=np.float32)
    got = sample_thickness_at_vertices(result, vertices)
    expected = _sample_voxel_field_trilinear(
        dilate_zeros(local, iters=2), result.origin, result.dx, vertices)
    np.testing.assert_allclose(got, expected, atol=1.0e-6)


def test_face_corner_seed_does_not_cross_nearby_disconnected_surface() -> None:
    """The short repair line must stay on the inward side of each face."""
    shape = (8, 8, 8)
    carrier = np.zeros(shape, dtype=np.float32)
    carrier[2, :, :] = 1.0
    carrier[4, :, :] = 7.0
    result = SdfResult(
        grid=np.zeros(shape, dtype=np.float32),
        n=8,
        dx=1.0,
        origin=np.zeros(3, dtype=np.float32),
        aabb_min=np.zeros(3, dtype=np.float32),
        aabb_max=np.full(3, 8.0, dtype=np.float32),
        thickness=carrier,
        blowup_thickness=carrier,
    )
    # The faces are only two voxels apart.  Their winding points away from the
    # gap, as it does for two separately closed components.  A symmetric
    # +/-2-voxel maximum would incorrectly copy the value 7 onto the first
    # face; inward-only probes must keep both components independent.
    vertices = np.asarray([
        [1.5, 1.5, 2.5], [5.5, 1.5, 2.5], [1.5, 5.5, 2.5],
        [1.5, 1.5, 4.5], [1.5, 5.5, 4.5], [5.5, 1.5, 4.5],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    sampled = sample_thickness_at_face_corners(result, vertices, faces)

    np.testing.assert_allclose(
        sampled[0], [1.0, 1.0, 1.0], rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        sampled[1], [7.0, 7.0, 7.0], rtol=0.0, atol=1.0e-6)


def test_face_corner_seed_repairs_one_missing_probe_face_locally() -> None:
    grid = np.zeros((3, 3, 3), dtype=np.float32)
    result = SdfResult(
        grid=grid,
        n=3,
        dx=1.0,
        origin=np.zeros(3, dtype=np.float32),
        aabb_min=np.zeros(3, dtype=np.float32),
        aabb_max=np.full(3, 3.0, dtype=np.float32),
        thickness=grid,
        blowup_thickness=np.ones_like(grid),
    )
    vertices = np.asarray([
        [0.5, 0.5, 0.5],
        [1.5, 0.5, 0.5],
        [0.5, 1.5, 0.5],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    def fake_sample(_field, _origin, _spacing, points, **_kwargs):
        values = np.zeros(len(points), dtype=np.float32)
        values[:3] = [0.0, 2.0, 4.0]
        return values

    with mock.patch.object(
            sdf_compute_module,
            "_sample_voxel_field_trilinear",
            side_effect=fake_sample):
        sampled = sample_thickness_at_face_corners(result, vertices, faces)

    assert np.all(np.isfinite(sampled))
    assert np.all(sampled > 0.0), sampled
    np.testing.assert_allclose(float(np.mean(sampled)), 3.0, atol=1.0e-5)


def main() -> None:
    test_voxel_centered_trilinear_sampling()
    test_sparse_thickness_from_dense_source()
    test_sparse_only_thickness_fallback()
    test_vertex_thickness_is_barycentrically_transported()
    test_face_corner_thickness_is_barycentrically_transported()
    test_sparse_face_discontinuity_does_not_cross_shared_vertices()
    test_vertex_thickness_validation()
    test_face_corner_thickness_validation()
    test_sample_thickness_at_vertices_prefers_surface_carrier()
    test_sample_thickness_at_vertices_dilates_legacy_field()
    test_face_corner_seed_does_not_cross_nearby_disconnected_surface()
    test_face_corner_seed_repairs_one_missing_probe_face_locally()
    print("sparse thickness regression: PASS")


if __name__ == "__main__":
    main()
