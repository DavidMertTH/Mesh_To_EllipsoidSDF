"""Focused regressions for pose-to-pose SDF resource reuse."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sdf_compute  # noqa: E402
from sdf_compute import SdfComputer  # noqa: E402


def _open_triangle() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    return vertices, faces


def test_same_topology_updates_points_and_refits_existing_mesh() -> None:
    vertices, faces = _open_triangle()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    mesh = sdf._warp_mesh
    point_buffer = sdf._points_wp
    index_buffer = sdf._indices_wp
    before = abs(sdf.query_point([0.25, 0.25, 0.1]))

    moved = vertices + np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    # A separately decoded but byte-identical face array must still hit.
    sdf.set_mesh(moved, faces.copy())
    after = abs(sdf.query_point([0.25, 0.25, 0.1]))

    assert sdf._warp_mesh is mesh
    assert sdf._points_wp is point_buffer
    assert sdf._indices_wp is index_buffer
    np.testing.assert_allclose(before, 0.1, atol=1.0e-5)
    np.testing.assert_allclose(after, 0.9, atol=1.0e-5)
    assert sdf.reuse_stats == {
        "set_mesh_calls": 2,
        "topology_cache_hits": 1,
        "watertight_checks": 1,
        "mesh_rebuilds": 1,
        "mesh_refits": 1,
        "reused_vertex_bytes": moved.nbytes,
        "reused_index_bytes": faces.nbytes,
        "grid_kernel_launches": 0,
        "grid_buffer_allocations": 0,
        "grid_progress_syncs": 0,
        "grid_host_readbacks": 0,
    }


def test_changed_topology_rebuilds_and_rechecks_watertightness() -> None:
    vertices, faces = _open_triangle()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    first_mesh = sdf._warp_mesh

    vertices = np.concatenate([
        vertices,
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    ])
    tetra_faces = np.asarray([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ], dtype=np.int32)
    sdf.set_mesh(vertices, tetra_faces)

    assert sdf._warp_mesh is not first_mesh
    assert sdf.reuse_stats["topology_cache_hits"] == 0
    assert sdf.reuse_stats["watertight_checks"] == 2
    assert sdf.reuse_stats["mesh_rebuilds"] == 2
    assert sdf.reuse_stats["mesh_refits"] == 0


def test_chunked_grid_uses_one_buffer_and_one_host_readback() -> None:
    vertices, faces = _open_triangle()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    origin = np.asarray([-0.5, -0.5, -0.5], dtype=np.float32)
    shape = (7, 6, 5)

    direct = sdf._launch_grid(origin, 0.2, shape, 10.0)
    stats_before = sdf.reuse_stats
    progress: list[float] = []
    chunked = sdf._launch_grid_chunked(
        origin,
        0.2,
        shape,
        10.0,
        lambda frac, _msg: progress.append(float(frac)),
        0.1,
        0.9,
    )
    delta = {
        key: sdf.reuse_stats[key] - stats_before[key]
        for key in (
            "grid_kernel_launches",
            "grid_buffer_allocations",
            "grid_progress_syncs",
            "grid_host_readbacks",
        )
    }

    np.testing.assert_allclose(chunked, direct, rtol=0.0, atol=1.0e-6)
    expected_launches = shape[2]  # nz < 20 => one layer per launch
    assert len(progress) == (expected_launches + 3) // 4
    assert progress[-1] == 0.9
    assert delta == {
        "grid_kernel_launches": expected_launches,
        "grid_buffer_allocations": 1,
        "grid_progress_syncs": (expected_launches + 3) // 4,
        "grid_host_readbacks": 1,
    }


def test_committed_mesh_does_not_alias_caller_arrays() -> None:
    vertices, faces = _open_triangle()
    original_vertices = vertices.copy()
    original_faces = faces.copy()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)

    vertices[:] = 99.0
    faces[:] = 0

    np.testing.assert_array_equal(sdf._verts, original_vertices)
    np.testing.assert_array_equal(sdf._faces, original_faces)


def test_failed_topology_rebuild_keeps_previous_cache_atomically() -> None:
    vertices, faces = _open_triangle()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    old_fields = {
        name: getattr(sdf, name)
        for name in (
            "_warp_mesh", "_points_wp", "_indices_wp", "_faces",
            "_topology_key",
        )
    }
    old_vertices = sdf._verts.copy()
    old_watertight = sdf._watertight

    changed_vertices = np.concatenate([
        vertices,
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    ])
    changed_faces = np.asarray([
        [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
    ], dtype=np.int32)
    with mock.patch.object(
            sdf_compute.wp, "Mesh", side_effect=RuntimeError("BVH failed")):
        try:
            sdf.set_mesh(changed_vertices, changed_faces)
        except RuntimeError as exc:
            assert "BVH failed" in str(exc)
        else:
            raise AssertionError("failed Warp mesh construction did not raise")

    for name, value in old_fields.items():
        assert getattr(sdf, name) is value
    np.testing.assert_array_equal(sdf._verts, old_vertices)
    assert sdf._watertight == old_watertight


def test_failed_refit_restores_old_points_and_cpu_vertices() -> None:
    vertices, faces = _open_triangle()
    sdf = SdfComputer(device="cpu")
    sdf.set_mesh(vertices, faces)
    old_vertices = sdf._verts.copy()

    class FakePoints:
        shape = (len(vertices),)

        def __init__(self):
            self.data = old_vertices.copy()

        def assign(self, values):
            self.data = np.asarray(values, dtype=np.float32).copy()

    class FakeMesh:
        def __init__(self):
            self.refit_calls = 0

        def refit(self):
            self.refit_calls += 1
            if self.refit_calls == 1:
                raise RuntimeError("refit failed")

    fake_points = FakePoints()
    fake_mesh = FakeMesh()
    sdf._points_wp = fake_points
    sdf._warp_mesh = fake_mesh
    moved = vertices + np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    try:
        sdf.set_mesh(moved, faces)
    except RuntimeError as exc:
        assert "refit failed" in str(exc)
    else:
        raise AssertionError("failed refit did not raise")

    np.testing.assert_array_equal(sdf._verts, old_vertices)
    np.testing.assert_array_equal(fake_points.data, old_vertices)
    assert fake_mesh.refit_calls == 2


def main() -> None:
    test_same_topology_updates_points_and_refits_existing_mesh()
    test_changed_topology_rebuilds_and_rechecks_watertightness()
    test_chunked_grid_uses_one_buffer_and_one_host_readback()
    test_committed_mesh_does_not_alias_caller_arrays()
    test_failed_topology_rebuild_keeps_previous_cache_atomically()
    test_failed_refit_restores_old_points_and_cpu_vertices()
    print("sdf compute reuse: PASS")


if __name__ == "__main__":
    main()
