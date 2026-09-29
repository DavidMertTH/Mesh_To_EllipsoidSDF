"""Regressions for fused pose-to-pose feature-thickness transport."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sdf_compute as sdf_compute_module  # noqa: E402
from sdf_blowup import (  # noqa: E402
    BLOWUP_CARRIER_MARGIN_VOXELS,
    relative_blowup_extent_voxels,
)
from sdf_compute import (  # noqa: E402
    SdfComputer,
    SdfResult,
    _sample_voxel_field_trilinear,
    sample_thickness_at_face_corners,
    sample_thickness_at_vertices,
)


def _triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    return vertices, faces


def _computer() -> tuple[SdfComputer, np.ndarray]:
    vertices, faces = _triangle_mesh()
    computer = SdfComputer(device="cpu")
    computer.set_mesh(vertices, faces)
    # t(x, y, z) = 1 + x + 2y is affine on the triangle, so exact
    # barycentric interpolation has a simple analytic reference.
    return computer, np.asarray([1.0, 2.0, 3.0], dtype=np.float32)


def _box_computer() -> tuple[SdfComputer, np.ndarray]:
    vertices = np.asarray([
        [-1.0, -0.5, -0.5], [1.0, -0.5, -0.5],
        [1.0, 0.5, -0.5], [-1.0, 0.5, -0.5],
        [-1.0, -0.5, 0.5], [1.0, -0.5, 0.5],
        [1.0, 0.5, 0.5], [-1.0, 0.5, 0.5],
    ], dtype=np.float32)
    faces = np.asarray([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    computer = SdfComputer(device="cpu")
    computer.set_mesh(vertices, faces)
    return computer, np.ones(faces.shape, dtype=np.float32)


def test_kernel_barycentrically_transports_vertex_thickness() -> None:
    computer, vertex_thickness = _computer()
    thickness_wp = wp.array(
        vertex_thickness, dtype=wp.float32, device="cpu")
    _, transported = computer._launch_grid_with_vertex_thickness(
        np.asarray([0.0, 0.0, -0.0625], dtype=np.float32),
        0.125,
        (8, 8, 1),
        10.0,
        thickness_wp,
    )

    xs = (np.arange(8) + 0.5) * 0.125
    ys = (np.arange(8) + 0.5) * 0.125
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    inside = (xx + yy) < (1.0 - 1.0e-6)
    expected = 1.0 + xx + 2.0 * yy
    np.testing.assert_allclose(
        transported[0][inside], expected[inside], atol=2.0e-6)


def test_kernel_barycentrically_transports_face_corner_thickness() -> None:
    computer, values = _computer()
    thickness_wp = wp.array(
        values.reshape(-1), dtype=wp.float32, device="cpu")
    _, transported = computer._launch_grid_with_face_corner_thickness(
        np.asarray([0.0, 0.0, -0.0625], dtype=np.float32),
        0.125,
        (8, 8, 1),
        10.0,
        thickness_wp,
    )

    xs = (np.arange(8) + 0.5) * 0.125
    ys = (np.arange(8) + 0.5) * 0.125
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    inside = (xx + yy) < (1.0 - 1.0e-6)
    expected = 1.0 + xx + 2.0 * yy
    np.testing.assert_allclose(
        transported[0][inside], expected[inside], atol=2.0e-6)


def test_face_corner_transport_isolates_shared_hard_edge() -> None:
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    face_corner_thickness = np.asarray(
        [[1.0, 1.0, 1.0], [7.0, 7.0, 7.0]], dtype=np.float32)
    computer = SdfComputer(device="cpu")
    computer.set_mesh(vertices, faces)
    thickness_wp = wp.array(
        face_corner_thickness.reshape(-1), dtype=wp.float32, device="cpu")

    _, transported = computer._launch_grid_with_face_corner_thickness(
        np.asarray([0.0, 0.0, -0.0625], dtype=np.float32),
        0.125,
        (8, 8, 1),
        10.0,
        thickness_wp,
    )

    xy = (np.arange(8) + 0.5) * 0.125
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    first_face = xx > yy + 0.25
    second_face = yy > xx + 0.25
    assert np.any(first_face) and np.any(second_face)
    np.testing.assert_allclose(
        transported[0][first_face], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(
        transported[0][second_face], 7.0, atol=1.0e-6)


def test_voxel_result_keeps_raw_thickness_inside_only() -> None:
    computer, face_corner_thickness = _box_computer()
    with mock.patch.object(
            sdf_compute_module,
            "local_thickness",
            side_effect=AssertionError("transport called local_thickness")):
        result = computer.compute_voxel_grid(
            n=16,
            margin=0.5,
            compute_thickness=True,
            max_dist=10.0,
            face_corner_thickness=face_corner_thickness,
        )

    assert result.thickness is not None
    inside = result.grid < 0.0
    outside = ~inside
    assert np.any(inside) and np.any(outside)
    np.testing.assert_array_equal(result.thickness[inside], 1.0)
    np.testing.assert_array_equal(result.thickness[outside], 0.0)


def test_vertex_thickness_validation_happens_before_grid_work() -> None:
    computer, vertex_thickness = _computer()
    cases = [
        (vertex_thickness[:2], "exactly one value per mesh vertex"),
        (np.asarray([1.0, np.nan, 3.0], dtype=np.float32), "finite"),
        (np.asarray([1.0, -0.1, 3.0], dtype=np.float32), "nonnegative"),
    ]
    for values, expected_message in cases:
        before = computer.reuse_stats["grid_kernel_launches"]
        try:
            computer.compute_voxel_grid(
                n=8,
                compute_thickness=False,
                max_dist=10.0,
                vertex_thickness=values,
            )
        except ValueError as exc:
            assert expected_message in str(exc), str(exc)
        else:
            raise AssertionError("invalid vertex_thickness was accepted")
        assert computer.reuse_stats["grid_kernel_launches"] == before


def test_face_corner_validation_happens_before_grid_work() -> None:
    computer, _ = _box_computer()
    faces = np.asarray(computer._faces, dtype=np.int32)
    valid = np.ones(faces.shape, dtype=np.float32)
    cases = [
        (valid[:-1], "exactly three values per mesh face"),
        (np.where(np.indices(valid.shape)[0] == 0, np.nan, valid), "finite"),
        (-valid, "nonnegative"),
    ]
    for values, expected_message in cases:
        before = computer.reuse_stats["grid_kernel_launches"]
        try:
            computer.compute_voxel_grid(
                n=8,
                compute_thickness=False,
                max_dist=10.0,
                face_corner_thickness=values,
            )
        except ValueError as exc:
            assert expected_message in str(exc), str(exc)
        else:
            raise AssertionError("invalid face_corner_thickness was accepted")
        assert computer.reuse_stats["grid_kernel_launches"] == before

    try:
        computer.compute_voxel_grid(
            n=8,
            compute_thickness=False,
            max_dist=10.0,
            vertex_thickness=np.ones(8, dtype=np.float32),
            face_corner_thickness=valid,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc), str(exc)
    else:
        raise AssertionError("both transported representations were accepted")


def test_vertex_seed_samples_legacy_dilation_lazily() -> None:
    local = np.zeros((5, 5, 5), dtype=np.float32)
    local[2, 2, 2] = 4.0
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
    with mock.patch.object(
            sdf_compute_module,
            "dilate_zeros",
            side_effect=AssertionError("seed dilated the full grid")):
        sampled = sample_thickness_at_vertices(result, vertices)
    np.testing.assert_array_equal(sampled, [4.0, 4.0, 0.0])


def test_affine_face_corner_seed_reconstructs_transport_field() -> None:
    """Interior probes must be inverted before they become corner values."""
    shape = (9, 9, 9)
    origin = np.asarray([-3.0, -3.0, -3.0], dtype=np.float32)
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    x_center = origin[0] + x + 0.5
    y_center = origin[1] + y + 0.5
    # Positive affine field, independent of z so all inward normal probes have
    # the same analytic value.
    carrier = (20.0 + 2.0 * x_center + 3.0 * y_center).astype(np.float32)
    result = SdfResult(
        grid=np.zeros(shape, dtype=np.float32),
        n=shape[0],
        dx=1.0,
        origin=origin,
        aabb_min=origin,
        aabb_max=origin + np.asarray(shape[::-1], dtype=np.float32),
        thickness=carrier,
        blowup_thickness=carrier,
    )
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    seeded = sample_thickness_at_face_corners(result, vertices, faces)

    # The (.5, .25, .25) probes see [22.5, 23.5, 24.0].  Those samples must
    # be inverted to the actual affine corner values before later poses use
    # ordinary barycentric interpolation.
    np.testing.assert_allclose(
        seeded, [[20.0, 24.0, 26.0]], rtol=0.0, atol=1.0e-6)

    computer = SdfComputer(device="cpu")
    computer.set_mesh(vertices, faces)
    thickness_wp = wp.array(
        seeded.reshape(-1), dtype=wp.float32, device="cpu")
    _, transported = computer._launch_grid_with_face_corner_thickness(
        np.asarray([0.0, 0.0, -0.125], dtype=np.float32),
        0.25,
        (8, 8, 1),
        10.0,
        thickness_wp,
    )
    xy = (np.arange(8, dtype=np.float32) + 0.5) * 0.25
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    inside = (xx + yy) < (2.0 - 1.0e-6)
    expected = 20.0 + 2.0 * xx + 3.0 * yy
    np.testing.assert_allclose(
        transported[0][inside], expected[inside], rtol=0.0, atol=2.0e-6)


def test_hard_edge_box_seed_recovers_face_scale() -> None:
    computer, _ = _box_computer()
    vertices = np.asarray(computer._verts, dtype=np.float32)
    faces = np.asarray(computer._faces, dtype=np.int32)
    result = computer.compute_voxel_grid(
        n=32,
        margin=0.25,
        compute_thickness=True,
        compute_blowup_thickness=False,
        thickness_max_resolution=None,
        max_dist=10.0,
    )

    sampled = sample_thickness_at_face_corners(result, vertices, faces)

    assert sampled.shape == faces.shape
    assert sampled.dtype == np.float32
    assert np.isfinite(sampled).all()
    # The box has a one-unit limiting feature size.  Affine reconstruction can
    # amplify voxel noise at individual corners, but must keep every value
    # resolved while preserving the face scale in aggregate.
    assert np.all(sampled > 0.0), sampled
    np.testing.assert_allclose(float(np.median(sampled)), 1.0, atol=0.20)
    np.testing.assert_allclose(float(np.mean(sampled)), 1.0, atol=0.12)


def test_fused_chunked_and_half_grid_paths() -> None:
    computer, values = _computer()
    face_corner_thickness = values.reshape(1, 3)
    thickness_wp = wp.array(
        face_corner_thickness.reshape(-1), dtype=wp.float32, device="cpu")
    origin = np.asarray([-0.25, -0.25, -0.4], dtype=np.float32)
    shape = (7, 6, 5)

    direct_sdf, direct_thickness = (
        computer._launch_grid_with_face_corner_thickness(
            origin, 0.2, shape, 10.0, thickness_wp)
    )
    progress: list[float] = []
    chunked_sdf, chunked_thickness = (
        computer._launch_grid_chunked_with_face_corner_thickness(
            origin,
            0.2,
            shape,
            10.0,
            thickness_wp,
            lambda frac, _msg: progress.append(float(frac)),
            0.1,
            0.9,
        )
    )
    np.testing.assert_allclose(chunked_sdf, direct_sdf, atol=1.0e-6)
    np.testing.assert_allclose(
        chunked_thickness, direct_thickness, atol=1.0e-6)
    assert progress and progress[-1] == 0.9

    half_sdf, half_thickness = (
        computer._launch_grid_half_with_face_corner_thickness(
            origin,
            0.2,
            shape,
            10.0,
            mirror_ax=0,
            face_corner_thickness_wp=thickness_wp,
        )
    )
    assert half_sdf.shape == half_thickness.shape == (5, 6, 7)
    np.testing.assert_array_equal(half_sdf, np.flip(half_sdf, axis=0))
    np.testing.assert_array_equal(
        half_thickness, np.flip(half_thickness, axis=0))


def test_missed_mesh_queries_clear_transported_thickness() -> None:
    computer, values = _computer()
    thickness_wp = wp.array(
        values.reshape(-1), dtype=wp.float32, device="cpu")
    grid, thickness = computer._launch_grid_with_face_corner_thickness(
        np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        0.25,
        (2, 2, 2),
        0.01,
        thickness_wp,
    )
    np.testing.assert_array_equal(grid, np.float32(1.0e6))
    np.testing.assert_array_equal(thickness, np.float32(0.0))


def test_dense_transported_field_is_the_blowup_carrier() -> None:
    computer, face_corner_thickness = _box_computer()
    fraction = 0.2
    with mock.patch.object(
            sdf_compute_module,
            "local_thickness",
            side_effect=AssertionError("transport called local_thickness")), \
         mock.patch.object(
            sdf_compute_module,
            "build_surface_carried_thickness",
            side_effect=AssertionError("transport rebuilt its carrier")):
        result = computer.compute_voxel_grid(
            n=32,
            margin=2.0,
            compute_thickness=False,
            compute_blowup_thickness=True,
            blowup_thickness_fraction=fraction,
            max_dist=10.0,
            face_corner_thickness=face_corner_thickness,
        )

    assert result.thickness is not None
    assert result.blowup_thickness is not result.thickness
    assert result.thickness_stride_vox == 1.0
    assert result.blowup_thickness_capacity_fraction == fraction
    expected_extent = (
        relative_blowup_extent_voxels(
            fraction, result.thickness, result.dx)
        + BLOWUP_CARRIER_MARGIN_VOXELS
    )
    np.testing.assert_allclose(
        result.blowup_thickness_extent_vox, expected_extent, atol=1.0e-6)
    outside = result.grid >= 0.0
    np.testing.assert_array_equal(result.thickness[outside], 0.0)
    carrier_limit = result.blowup_thickness_extent_vox * result.dx
    near_exterior = outside & (result.grid <= carrier_limit - result.dx)
    far_exterior = result.grid > carrier_limit + result.dx
    assert np.any(near_exterior) and np.any(far_exterior)
    assert np.all(result.blowup_thickness[near_exterior] > 0.0)
    np.testing.assert_array_equal(
        result.blowup_thickness[far_exterior], 0.0)

    sparse = computer.compute_sparse_samples(
        n=32,
        margin=2.0,
        surface_samples=32,
        offsets_vox=(-1.0, 0.0, 1.0),
        coarse_n=8,
        seed=7,
        thickness_result=result,
        face_corner_thickness=face_corner_thickness,
    )
    # A non-zero blowup also moves coarse occupancy targets.  Their feature
    # scale therefore comes from the already transported dense carrier.
    assert sparse.coarse_mask is not None
    coarse_values = sparse.values[sparse.coarse_mask]
    coarse_thickness = sparse.thickness[sparse.coarse_mask]
    expected_coarse = _sample_voxel_field_trilinear(
        result.blowup_thickness,
        result.origin,
        result.dx,
        sparse.points[sparse.coarse_mask],
    )
    np.testing.assert_allclose(
        coarse_thickness, expected_coarse, atol=1.0e-6)
    safely_near = coarse_values <= carrier_limit - 2.0 * result.dx
    safely_far = coarse_values >= carrier_limit + 2.0 * result.dx
    assert np.any(safely_near) and np.any(safely_far)
    assert np.all(coarse_thickness[safely_near] > 0.0)
    np.testing.assert_array_equal(coarse_thickness[safely_far], 0.0)


def main() -> None:
    test_kernel_barycentrically_transports_vertex_thickness()
    test_kernel_barycentrically_transports_face_corner_thickness()
    test_face_corner_transport_isolates_shared_hard_edge()
    test_voxel_result_keeps_raw_thickness_inside_only()
    test_vertex_thickness_validation_happens_before_grid_work()
    test_face_corner_validation_happens_before_grid_work()
    test_vertex_seed_samples_legacy_dilation_lazily()
    test_affine_face_corner_seed_reconstructs_transport_field()
    test_hard_edge_box_seed_recovers_face_scale()
    test_fused_chunked_and_half_grid_paths()
    test_missed_mesh_queries_clear_transported_thickness()
    test_dense_transported_field_is_the_blowup_carrier()
    print("dense thickness transport: PASS")


if __name__ == "__main__":
    main()
