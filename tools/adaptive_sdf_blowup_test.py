"""Deterministic regressions for thickness-aware SDF blowup."""

from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdf_blowup import (  # noqa: E402
    BLOWUP_CARRIER_MARGIN_VOXELS,
    MAX_UI_THICKNESS_FRACTION,
    apply_thickness_relative_blowup,
    build_surface_carried_thickness,
    conservative_mirror_min,
    legacy_voxel_blowup_to_thickness_fraction,
    relative_blowup_extent_voxels,
    required_relative_sdf_margin,
    sparse_band_offsets,
    thickness_relative_offsets,
)
from sdf_samples import SdfSampleSet, sample_sdf_grid_normals  # noqa: E402
from thickness import local_thickness  # noqa: E402


class AdaptiveSdfBlowupTest(unittest.TestCase):
    def test_zero_request_is_an_exact_noop(self) -> None:
        values = np.array(
            [[-3.0, -0.25, 0.0], [0.25, 3.0, 30.0]],
            dtype=np.float32,
        )
        thickness = np.array(
            [[0.0, 0.5, 2.0], [8.0, 0.0, 20.0]],
            dtype=np.float32,
        )

        offsets = thickness_relative_offsets(values, 0.0, thickness)
        result = apply_thickness_relative_blowup(values, 0.0, thickness)

        np.testing.assert_array_equal(offsets, np.zeros_like(values))
        np.testing.assert_array_equal(result, values)
        self.assertEqual(offsets.dtype, np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_relative_offset_has_no_absolute_plateau(self) -> None:
        values = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
        thickness = np.array([0.4, 2.0, 8.0], dtype=np.float32)

        for fraction in (0.1, -0.1):
            with self.subTest(fraction=fraction):
                expected_offsets = fraction * thickness
                offsets = thickness_relative_offsets(
                    values, fraction, thickness)
                result = apply_thickness_relative_blowup(
                    values, fraction, thickness)

                np.testing.assert_allclose(
                    offsets, expected_offsets, rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(
                    offsets / thickness, fraction, rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(
                    result, values + expected_offsets,
                    rtol=0.0, atol=1.0e-6)

    def test_unknown_thickness_fails_closed_everywhere(self) -> None:
        values = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
        thickness = np.zeros_like(values)

        for fraction in (0.2, -0.2):
            with self.subTest(fraction=fraction):
                offsets = thickness_relative_offsets(
                    values, fraction, thickness)
                expected = np.zeros_like(values)
                np.testing.assert_array_equal(offsets, expected)

                result = apply_thickness_relative_blowup(
                    values, fraction, thickness)
                np.testing.assert_array_equal(result, values + expected)
                without_field = apply_thickness_relative_blowup(
                    values, fraction, None)
                np.testing.assert_array_equal(without_field, values)

    def test_offset_field_is_odd_in_request_and_reflection_symmetric(
            self) -> None:
        values = np.array(
            [-20.0, -1.0, -0.1, 0.0, -0.1, -1.0, -20.0],
            dtype=np.float32,
        )
        thickness = np.array(
            [0.0, 1.0, 4.0, 16.0, 4.0, 1.0, 0.0],
            dtype=np.float32,
        )

        positive = thickness_relative_offsets(values, 0.2, thickness)
        negative = thickness_relative_offsets(values, -0.2, thickness)
        reflected = thickness_relative_offsets(
            values[::-1], 0.2, thickness[::-1])

        np.testing.assert_array_equal(positive, -negative)
        np.testing.assert_array_equal(reflected, positive[::-1])
        np.testing.assert_array_equal(positive, positive[::-1])

    def test_dense_and_flat_sparse_arrays_have_identical_results(self) -> None:
        values = np.linspace(-6.0, 6.0, 60, dtype=np.float32).reshape(
            3, 4, 5)
        thickness = np.linspace(
            0.25, 12.0, values.size, dtype=np.float32).reshape(values.shape)
        thickness.ravel()[::7] = 0.0
        fraction = -0.175

        dense_offsets = thickness_relative_offsets(
            values, fraction, thickness)
        sparse_offsets = thickness_relative_offsets(
            values.ravel(), fraction, thickness.ravel())
        dense_result = apply_thickness_relative_blowup(
            values, fraction, thickness)
        sparse_result = apply_thickness_relative_blowup(
            values.ravel(), fraction, thickness.ravel())

        np.testing.assert_array_equal(
            dense_offsets, sparse_offsets.reshape(values.shape))
        np.testing.assert_array_equal(
            dense_result, sparse_result.reshape(values.shape))

    def test_relative_blowup_is_world_scale_covariant(self) -> None:
        values = np.array([-0.3, 0.1, 2.0], dtype=np.float32)
        thickness = np.array([0.4, 1.2, 8.0], dtype=np.float32)
        fraction = -0.1375
        reference = apply_thickness_relative_blowup(
            values, fraction, thickness)

        for scale in (0.01, 3.7, 1000.0):
            with self.subTest(scale=scale):
                scaled = apply_thickness_relative_blowup(
                    scale * values, fraction, scale * thickness)
                np.testing.assert_allclose(
                    scaled, scale * reference, rtol=2.0e-6, atol=1.0e-6)

    def test_conservative_mirror_min_repairs_downsample_phase_bias(
            self) -> None:
        thickness = np.array(
            [[1.0, 5.0, 7.0, 3.0], [4.0, 9.0, 2.0, 8.0]],
            dtype=np.float32,
        )
        symmetric = conservative_mirror_min(thickness, axis=1)

        np.testing.assert_array_equal(symmetric, symmetric[:, ::-1])
        self.assertTrue(np.all(symmetric <= thickness))
        np.testing.assert_array_equal(
            symmetric,
            np.array(
                [[1.0, 5.0, 5.0, 1.0], [4.0, 2.0, 2.0, 4.0]],
                dtype=np.float32,
            ),
        )

        with_holes = np.array(
            [[0.0, 3.0, 7.0, 5.0]], dtype=np.float32)
        repaired = conservative_mirror_min(with_holes, axis=1)
        np.testing.assert_array_equal(
            repaired,
            np.array([[5.0, 3.0, 3.0, 5.0]], dtype=np.float32),
        )

    def test_slab_carries_surface_thickness_without_thick_to_thin_leak(
            self) -> None:
        dx = 1.0
        nz, ny, nx = 3, 9, 49
        x = np.arange(nx, dtype=np.float32) - nx // 2
        slab_half_width = 4.0
        grid_line = np.abs(x) - slab_half_width
        grid = np.broadcast_to(
            grid_line[None, None, :], (nz, ny, nx)).copy()

        thin_value = np.float32(2.0)
        thick_value = np.float32(8.0)
        thickness = np.zeros_like(grid)
        interior = grid < 0.0
        thin_rows = np.arange(ny)[None, :, None] < 4
        thickness[interior & np.broadcast_to(thin_rows, grid.shape)] = (
            thin_value)
        thickness[interior & ~np.broadcast_to(thin_rows, grid.shape)] = (
            thick_value)

        carried = build_surface_carried_thickness(
            grid, thickness, dx, max_exterior_vox=12.0)

        z = nz // 2
        thin_y = 3  # directly beside the thick region: catches max dilation
        thick_y = 4
        exterior_x = nx // 2 + 10
        self.assertGreater(grid[z, thin_y, exterior_x] / dx, 2.0)
        self.assertLessEqual(grid[z, thin_y, exterior_x] / dx, 12.0)
        self.assertEqual(carried[z, thin_y, exterior_x], thin_value)
        self.assertEqual(carried[z, thick_y, exterior_x], thick_value)

        # The carrier follows the slab normal and therefore remains symmetric
        # on the two exterior sides.
        np.testing.assert_array_equal(carried, carried[..., ::-1])

        # Interior input is preserved, while samples outside the requested
        # carrier band remain unknown instead of inheriting a remote maximum.
        np.testing.assert_array_equal(carried[interior], thickness[interior])
        beyond_band_x = nx // 2 + 18
        self.assertGreater(grid[z, thin_y, beyond_band_x] / dx, 12.0)
        self.assertEqual(carried[z, thin_y, beyond_band_x], 0.0)

    def test_surface_thickness_is_stable_across_resolution_and_phase(
            self) -> None:
        truth = 0.5
        for n in (48, 64, 96):
            dx = 2.0 / n
            coord = (
                (np.arange(n, dtype=np.float32) + 0.5) * dx - 1.0)
            for phase in (0.0, 0.25, 0.5, 0.75):
                with self.subTest(n=n, phase=phase):
                    line = np.abs(coord - phase * dx) - 0.5 * truth
                    grid = np.broadcast_to(
                        line[None, None, :], (3, 3, n)).copy()
                    thickness = local_thickness(
                        grid, dx, max_resolution=None)
                    self.assertEqual(
                        int(np.count_nonzero(thickness[grid >= 0.0])), 0)
                    carried = build_surface_carried_thickness(
                        grid, thickness, dx, max_exterior_vox=12.0)
                    surface_band = (line >= 0.0) & (line <= dx)
                    measured = carried[1, 1, surface_band]
                    self.assertTrue(measured.size > 0)
                    np.testing.assert_allclose(
                        measured, truth, rtol=0.0, atol=2.0 * dx)

    def test_dynamic_carrier_and_sparse_extent_cover_thick_regions(self) -> None:
        dx = 1.0
        x = np.arange(-70, 71, dtype=np.float32)
        grid_line = np.abs(x) - 40.0
        grid = np.broadcast_to(
            grid_line[None, None, :], (3, 3, x.size)).copy()
        thickness = np.zeros_like(grid)
        thickness[grid < 0.0] = 80.0
        fraction = -MAX_UI_THICKNESS_FRACTION

        extent_vox = relative_blowup_extent_voxels(
            fraction, thickness, dx)
        self.assertEqual(extent_vox, 20.0)
        carried = build_surface_carried_thickness(grid, thickness, dx)
        moved_surface = np.flatnonzero(np.isclose(grid_line, 20.0))
        self.assertGreater(moved_surface.size, 0)
        self.assertTrue(np.all(carried[1, 1, moved_surface] == 80.0))
        self.assertGreaterEqual(
            max(abs(v) for v in sparse_band_offsets(extent_vox)),
            extent_vox + 1.0,
        )
        self.assertGreaterEqual(
            required_relative_sdf_margin(0.0, fraction, 128),
            0.5,
        )

    def test_explicit_grid_guard_is_aspect_ratio_safe(self) -> None:
        from sdf_compute import SdfComputer

        half = np.array([2.0, 0.5, 0.25], dtype=np.float32)
        vertices = np.array([
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ], dtype=np.float32)
        # Vertex index bits are x/y/z respectively.
        faces = np.array([
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ], dtype=np.int32)
        fraction = MAX_UI_THICKNESS_FRACTION
        guard = int(BLOWUP_CARRIER_MARGIN_VOXELS)
        computer = SdfComputer(device="cpu")
        computer.set_mesh(vertices, faces)
        result = computer.compute_voxel_grid(
            n=64,
            margin=required_relative_sdf_margin(0.0, fraction, 64),
            compute_thickness=False,
            max_dist=float("inf"),
            guard_voxels_per_side=guard,
        )

        counts = np.array([result.nx, result.ny, result.nz])
        grid_min = np.asarray(result.origin, dtype=np.float64)
        grid_max = grid_min + counts * float(result.dx)
        extent = 2.0 * half.astype(np.float64)
        moved_min = -half.astype(np.float64) - fraction * extent
        moved_max = half.astype(np.float64) + fraction * extent
        lower_clearance = (moved_min - grid_min) / float(result.dx)
        upper_clearance = (grid_max - moved_max) / float(result.dx)

        self.assertEqual(result.n, 64)
        self.assertEqual(result.n, max(result.grid.shape))
        self.assertTrue(np.all(lower_clearance >= guard - 1.0e-5))
        self.assertTrue(np.all(upper_clearance >= guard - 1.0e-5))
        self.assertTrue(np.all(lower_clearance < guard + 1.0))
        self.assertTrue(np.all(upper_clearance < guard + 1.0))

    def test_unresolved_thin_sheet_cannot_borrow_thick_body_behind_it(
            self) -> None:
        dx = 1.0
        x = np.arange(25, dtype=np.float32)
        thick_body = np.abs(x - 3.0) - 3.0
        unresolved_sheet = np.abs(x - 10.0) - 0.2
        grid_line = np.minimum(thick_body, unresolved_sheet)
        grid = np.broadcast_to(grid_line[None, None, :], (3, 3, 25)).copy()
        thickness = np.zeros_like(grid)
        thick_interior = np.broadcast_to(
            (thick_body < 0.0)[None, None, :], grid.shape)
        thickness[thick_interior] = 6.0
        # The sub-voxel sheet has an interior SDF sample at x=10, but its local
        # thickness is deliberately unresolved (zero).
        self.assertLess(grid[1, 1, 10], 0.0)
        self.assertEqual(thickness[1, 1, 10], 0.0)

        carried = build_surface_carried_thickness(
            grid, thickness, dx, max_exterior_vox=8.0)

        # x=12 projects to the unresolved sheet, not to the thicker body on its
        # inward side.  The full-resolution first-exit chord may recover the
        # sheet's own sub-voxel thickness, but must never borrow the body value.
        self.assertGreater(grid[1, 1, 12], 0.0)
        self.assertGreater(carried[1, 1, 12], 0.0)
        self.assertLess(carried[1, 1, 12], 1.0)
        offset = thickness_relative_offsets(
            grid[1, 1, 12:13], -0.25, carried[1, 1, 12:13])
        self.assertLess(abs(float(offset[0])), 0.25)

    def test_carrier_cannot_cross_one_voxel_air_gap(self) -> None:
        dx = 1.0
        x = np.arange(20, dtype=np.float32)
        thick_body = np.abs(x - 4.5) - 5.0
        unresolved_sheet = np.abs(x - 11.0) - 0.2
        grid_line = np.minimum(thick_body, unresolved_sheet)
        grid = np.broadcast_to(
            grid_line[None, None, :], (3, 3, x.size)).copy()
        thickness = np.zeros_like(grid)
        body_interior = np.broadcast_to(
            (thick_body < 0.0)[None, None, :], grid.shape)
        thickness[body_interior] = 6.0

        # Along the inward normal the samples are body=-0.5, air=+0.5,
        # unresolved sheet=-0.2, exterior=+0.8.  The interpolated +0.5 air
        # crossing must stop the carrier before it reaches the body.
        np.testing.assert_allclose(
            grid[1, 1, 9:13],
            [-0.5, 0.5, -0.2, 0.8],
            rtol=0.0,
            atol=1.0e-6,
        )
        carried = build_surface_carried_thickness(
            grid, thickness, dx, max_exterior_vox=4.0)
        self.assertGreater(carried[1, 1, 12], 0.0)
        self.assertLess(carried[1, 1, 12], 1.0)

    def test_carrier_uses_resolved_corner_across_thickness_stride_hole(
            self) -> None:
        dx = 1.0
        x = np.arange(13, dtype=np.float32) - 6.0
        grid_line = np.abs(x) - 2.5
        grid = np.broadcast_to(
            grid_line[None, None, :], (3, 3, x.size)).copy()
        thickness = np.zeros_like(grid)
        interior = grid < 0.0
        thickness[interior] = 2.0
        # Emulate a coarse/strided thickness pass: one local interior corner is
        # unresolved although the other corners in the same surface cell are
        # valid.
        thickness[1, 1, 8] = 0.0

        carried = build_surface_carried_thickness(
            grid, thickness, dx, max_exterior_vox=3.0)

        self.assertGreater(grid[1, 1, 9], 0.0)
        self.assertEqual(carried[1, 1, 9], 2.0)

    def test_factor_four_floor_does_not_inflate_two_voxel_slab(self) -> None:
        n = 18
        dx = 1.0
        x = np.arange(n, dtype=np.float32) + 0.5 - n / 2
        line = np.abs(x) - 1.0
        grid = np.broadcast_to(
            line[None, None, :], (3, 3, n)).copy()
        thickness = np.zeros_like(grid)
        thickness[grid < 0.0] = 4.0

        carried = build_surface_carried_thickness(
            grid,
            thickness,
            dx,
            max_exterior_vox=4.0,
            thickness_stride_vox=4.0,
            device="cpu",
        )

        np.testing.assert_array_equal(
            thickness[1, 1, 7:11], [0.0, 4.0, 4.0, 0.0])
        np.testing.assert_allclose(
            carried[1, 1, 7:11], 2.0, rtol=0.0, atol=1.0e-5)
        eroded = apply_thickness_relative_blowup(
            grid, 0.25, carried)
        dilated = apply_thickness_relative_blowup(
            grid, -0.25, carried)
        np.testing.assert_allclose(
            eroded[1, 1, 8:10], 0.0, rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(
            dilated[1, 1, [7, 10]], 0.0, rtol=0.0, atol=1.0e-5)

    def test_medial_thin_voxel_and_small_world_scale_are_repaired(self) -> None:
        import warp as wp

        n = 19
        x = np.arange(n, dtype=np.float32) - n // 2
        dimensionless_sdf = np.abs(x) - 1.0
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")

        for dx in (1.0, 1.0e-8):
            grid = np.broadcast_to(
                (dimensionless_sdf * dx)[None, None, :],
                (3, 3, n),
            ).copy()
            thickness = np.zeros_like(grid)
            thickness[grid < 0.0] = 4.0 * dx
            for device in devices:
                with self.subTest(dx=dx, device=device):
                    carried = build_surface_carried_thickness(
                        grid,
                        thickness,
                        dx,
                        max_exterior_vox=4.0,
                        thickness_stride_vox=4.0,
                        device=device,
                    )
                    # The sole interior voxel has zero centred gradient.  An
                    # adjacent normal plus first-exit chord still recovers the
                    # two-voxel width, independent of world-unit scale.
                    np.testing.assert_allclose(
                        carried[1, 1, 7:12] / dx,
                        2.0,
                        rtol=2.0e-5,
                        atol=5.0e-5,
                    )
                    eroded = apply_thickness_relative_blowup(
                        grid, 0.25, carried)
                    self.assertAlmostEqual(
                        float(eroded[1, 1, 9] / dx), -0.5, places=5)

    def test_oblique_float32_ridge_uses_a_stable_neighbour_normal(self) -> None:
        import warp as wp

        n = 64
        dx = 1.0
        coord = np.arange(n, dtype=np.float32) - n / 2 + 0.5
        z, y, x = np.meshgrid(coord, coord, coord, indexing="ij")
        normal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        normal /= np.linalg.norm(normal)
        q = x * normal[0] + y * normal[1] + z * normal[2]
        grid = (np.abs(q) - 0.5).astype(np.float32)
        thickness = local_thickness(grid, dx, max_resolution=16)
        carried_cpu = build_surface_carried_thickness(
            grid,
            thickness,
            dx,
            max_exterior_vox=8.0,
            thickness_stride_vox=4.0,
            device="cpu",
        )
        core = np.zeros_like(grid, dtype=bool)
        core[18:-18, 18:-18, 18:-18] = True
        medial = core & (grid < 0.0) & (np.abs(q) < 1.0e-5)
        self.assertGreater(np.count_nonzero(medial), 0)
        np.testing.assert_allclose(
            carried_cpu[medial], 1.0, rtol=0.0, atol=0.25)

        if wp.is_cuda_available():
            carried_cuda = build_surface_carried_thickness(
                grid,
                thickness,
                dx,
                max_exterior_vox=8.0,
                thickness_stride_vox=4.0,
                device="cuda:0",
            )
            np.testing.assert_allclose(
                carried_cuda[medial],
                carried_cpu[medial],
                rtol=2.0e-5,
                atol=5.0e-5,
            )

    def test_carrier_closes_factor_four_surface_stride_blocks(self) -> None:
        n = 64
        dx = 1.0
        coord = np.arange(n, dtype=np.float32) + 0.5 - n / 2
        z, y, x = np.meshgrid(coord, coord, coord, indexing="ij")
        grid = (
            np.sqrt(x * x + y * y + z * z) - 18.0
        ).astype(np.float32)
        # 64 -> 16 is the same factor-four downsampling used by a
        # 512-grid with the production 128 thickness-resolution limit.
        thickness = local_thickness(
            grid, dx, max_resolution=16)
        thickness = conservative_mirror_min(thickness, axis=2)
        carried = build_surface_carried_thickness(
            grid,
            thickness,
            dx,
            max_exterior_vox=14.0,
            thickness_stride_vox=4.0,
        )

        exterior_band = (grid >= 0.0) & (grid <= 14.0 * dx)
        coverage = float(np.mean(carried[exterior_band] > 0.0))
        self.assertGreater(coverage, 0.99)

    def test_carrier_margin_covers_optimizer_surface_band(self) -> None:
        # Dynamic extent reaches the farthest relative zero surface; the fixed
        # remainder must still cover the optimizer's three-voxel surface band.
        self.assertGreaterEqual(BLOWUP_CARRIER_MARGIN_VOXELS, 4.0)

    def test_cached_carrier_skips_repeated_full_volume_scans(self) -> None:
        from unittest.mock import patch

        import main_window

        raw = np.full((2, 2, 2), 4.0, dtype=np.float32)
        cached = np.full_like(raw, 4.0)

        def result(*, capacity):
            return SimpleNamespace(
                grid=np.zeros_like(raw),
                thickness=raw,
                dx=1.0,
                blowup_thickness=cached,
                blowup_thickness_extent_vox=5.0,
                blowup_thickness_capacity_fraction=capacity,
                _sdf_blowup_capacity_fraction=0.25,
            )

        host = SimpleNamespace(
            _last_mesh_result=None,
            _mesh_settings=SimpleNamespace(
                blowup_fraction=lambda: 0.1),
            _status=SimpleNamespace(showMessage=lambda _msg: None),
            _device="cpu",
        )

        known = result(capacity=0.25)
        with patch(
                "main_window.relative_blowup_extent_voxels",
                side_effect=AssertionError("cache hit rescanned thickness")), \
             patch(
                "main_window.build_surface_carried_thickness",
                side_effect=AssertionError("cache hit rebuilt carrier")):
            first = main_window.MainWindow._ensure_blowup_thickness(
                host, known, thickness_fraction=0.1, update_views=False)
            second = main_window.MainWindow._ensure_blowup_thickness(
                host, known, thickness_fraction=-0.1, update_views=False)
        self.assertIs(first, cached)
        self.assertIs(second, cached)

        # A legacy carrier without support metadata self-heals after exactly
        # one extent check; subsequent slider ticks take the same O(1) path.
        legacy = result(capacity=None)
        original_extent = main_window.relative_blowup_extent_voxels
        with patch(
                "main_window.relative_blowup_extent_voxels",
                wraps=original_extent) as scan, \
             patch(
                "main_window.build_surface_carried_thickness",
                side_effect=AssertionError("wide legacy carrier rebuilt")):
            main_window.MainWindow._ensure_blowup_thickness(
                host, legacy, thickness_fraction=0.1, update_views=False)
            main_window.MainWindow._ensure_blowup_thickness(
                host, legacy, thickness_fraction=-0.1, update_views=False)
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(
            legacy.blowup_thickness_capacity_fraction, 0.25)

    def test_local_fit_resamples_thickness_and_uses_relative_offset(
            self) -> None:
        from optimization import OptimizationWorker

        worker = SimpleNamespace(
            _sdf_blowup_fraction=-0.25,
            _sdf_blowup_thickness_np=np.full(
                (5, 5, 5), 2.0, dtype=np.float32),
            _sdf_blowup_origin=np.zeros(3, dtype=np.float32),
            _sdf_blowup_dx=1.0,
            _dx=0.5,
        )
        region = SimpleNamespace(
            grid=np.zeros((3, 3, 3), dtype=np.float32),
            origin=np.full(3, 1.25, dtype=np.float32),
            dx=0.5,
            thickness=None,
            blowup_thickness=None,
        )
        OptimizationWorker._apply_blowup_to_region_result(worker, region)

        np.testing.assert_allclose(
            region.blowup_thickness, 2.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            region.grid, -0.5, rtol=0.0, atol=1.0e-6)
        self.assertIs(region.thickness, region.blowup_thickness)

        box_min, box_max = OptimizationWorker._region_box(
            worker,
            np.zeros(3, dtype=np.float32),
            half_extent=0.25,
        )
        np.testing.assert_allclose(box_min, -1.75, atol=1.0e-6)
        np.testing.assert_allclose(box_max, 1.75, atol=1.0e-6)

    def test_pose_targets_use_each_current_poses_thickness(self) -> None:
        from pose_correctives import PoseCorrectiveWorker

        geometry = SimpleNamespace(
            num_ellipsoids=1,
            local_centers=np.zeros((1, 3), dtype=np.float32),
            local_radii=np.ones((1, 3), dtype=np.float32),
            local_rotations=np.array(
                [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        )
        worker = SimpleNamespace(
            _fit_kwargs={},
            _sdf_blowup_fraction=-0.1,
            _base=geometry,
        )
        identity_linear = np.eye(3, dtype=np.float32)[None, ...]
        zero_offset = np.zeros((1, 3), dtype=np.float32)
        identity_rotation = np.array(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        def target(thickness_value: float, dx: float):
            sdf = SimpleNamespace(
                grid=np.zeros((2, 2, 2), dtype=np.float32),
                thickness=np.full(
                    (2, 2, 2), thickness_value, dtype=np.float32),
                blowup_thickness=np.full(
                    (2, 2, 2), thickness_value, dtype=np.float32),
                origin=np.zeros(3, dtype=np.float32),
                dx=dx,
                n=2,
            )
            return PoseCorrectiveWorker._optimizer_kwargs(
                worker,
                sdf,
                geometry,
                identity_linear,
                zero_offset,
                identity_rotation,
                None,
                0.0,
            )["sdf_target_np"]

        pose_a = target(2.0, dx=0.5)
        pose_b = target(4.0, dx=0.125)
        np.testing.assert_allclose(pose_a, -0.2, atol=1.0e-6)
        np.testing.assert_allclose(pose_b, -0.4, atol=1.0e-6)
        np.testing.assert_allclose(pose_b, 2.0 * pose_a, atol=1.0e-6)

    def test_optimizer_symmetry_chooses_less_aggressive_blowup_pair(
            self) -> None:
        from optimization import OptimizationWorker

        for fraction, expected in (
            (-0.1, np.array([[[-2.0, -2.0]]], dtype=np.float32)),
            (0.1, np.array([[[-5.0, -5.0]]], dtype=np.float32)),
        ):
            with self.subTest(fraction=fraction):
                worker = SimpleNamespace(
                    _sdf_target_np=np.array(
                        [[[-2.0, -5.0]]], dtype=np.float32),
                    _thickness_np=np.array(
                        [[[1.0, 10.0]]], dtype=np.float32),
                    _sdf_blowup_fraction=fraction,
                    _sdf_blowup_thickness_np=None,
                    _sdf_samples=None,
                    _uploaded_samples=None,
                    _dx=0.25,
                    _detect_symmetry_axis=lambda _grid: (0, 0.0),
                )
                OptimizationWorker._setup_symmetry(worker)

                np.testing.assert_array_equal(
                    worker._sdf_target_np, expected)
                np.testing.assert_array_equal(
                    worker._thickness_np,
                    np.array([[[1.0, 1.0]]], dtype=np.float32),
                )

    def test_sparse_offsets_extend_symmetrically_beyond_large_requests(
            self) -> None:
        base = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
        self.assertEqual(sparse_band_offsets(4.0), base)

        magnitude = 7.25
        positive = sparse_band_offsets(magnitude)
        negative = sparse_band_offsets(-magnitude)

        self.assertEqual(positive, negative)
        self.assertEqual(positive, tuple(sorted(set(positive))))
        self.assertTrue(set(base).issubset(positive))
        for value in (
            -magnitude,
            magnitude,
            -(magnitude + 1.0),
            magnitude + 1.0,
        ):
            self.assertIn(value, positive)
        self.assertLess(min(positive), -magnitude)
        self.assertGreater(max(positive), magnitude)
        self.assertEqual(set(positive), {-value for value in positive})

    def test_sparse_sample_relative_offset_preserves_metadata(self) -> None:
        samples = SdfSampleSet(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            dtype=np.float32),
            values=np.array([0.0, 10.0], dtype=np.float32),
            thickness=np.array([0.8, 40.0], dtype=np.float32),
            dx=0.5,
            source="adaptive-test",
            coarse_mask=np.array([False, True]),
            normals=np.array(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32),
        )
        transformed_normals = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        adjusted = samples.with_thickness_relative_offset(
            -0.1, normals=transformed_normals)

        np.testing.assert_array_equal(adjusted.points, samples.points)
        np.testing.assert_array_equal(adjusted.thickness, samples.thickness)
        np.testing.assert_array_equal(
            adjusted.coarse_mask, samples.coarse_mask)
        np.testing.assert_allclose(
            adjusted.values,
            np.array([-0.08, 6.0], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(adjusted.normals, transformed_normals)
        self.assertEqual(adjusted.dx, samples.dx)
        self.assertEqual(adjusted.source, samples.source)

    def test_sparse_normals_are_sampled_from_transformed_grid(self) -> None:
        n = 9
        dx = 0.25
        origin = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        z, y, x = np.meshgrid(
            np.arange(n, dtype=np.float32),
            np.arange(n, dtype=np.float32),
            np.arange(n, dtype=np.float32),
            indexing="ij",
        )
        # Exact affine field under trilinear interpolation.
        grid = (2.0 * x + 3.0 * y - 4.0 * z).astype(np.float32)
        points = np.array(
            [[-0.6, -0.7, -0.5], [0.2, 0.1, -0.3]], dtype=np.float32)
        normals = sample_sdf_grid_normals(grid, origin, dx, points)
        expected = np.array([2.0, 3.0, -4.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(
            normals, np.repeat(expected[None, :], 2, axis=0),
            rtol=0.0, atol=1.0e-6)

    def test_cpu_and_warp_preview_use_the_same_relative_value(self) -> None:
        import warp as wp
        import sdf_slice

        grid = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        thickness = np.full_like(grid, 2.0)
        fraction = -0.1
        cpu = apply_thickness_relative_blowup(
            grid, fraction, thickness)
        grid_wp = wp.array(
            grid.ravel(), dtype=wp.float32, device="cpu")
        thickness_wp = wp.array(
            thickness.ravel(), dtype=wp.float32, device="cpu")
        out = wp.empty(1, dtype=wp.float32, device="cpu")
        zero_step = wp.vec3(0.0, 0.0, 0.0)
        wp.launch(
            sdf_slice._relative_grid_field_kernel,
            dim=1,
            inputs=[
                grid_wp,
                thickness_wp,
                wp.vec3(0.0, 0.0, 0.0),
                1.0, 4, 4, 4,
                wp.vec3(1.5, 1.5, 1.5),
                zero_step, zero_step, 1,
                fraction,
                out,
            ],
            device="cpu",
        )
        self.assertAlmostEqual(float(out.numpy()[0]), float(cpu[1, 1, 1]))

    def test_legacy_slider_position_migrates_to_relative_strength(self) -> None:
        self.assertEqual(
            legacy_voxel_blowup_to_thickness_fraction(10.0), 0.25)
        self.assertEqual(
            legacy_voxel_blowup_to_thickness_fraction(-10.0), -0.25)
        self.assertAlmostEqual(
            legacy_voxel_blowup_to_thickness_fraction(-2.0), -0.05)

    def test_application_paths_use_the_same_relative_contract(self) -> None:
        main_window = (ROOT / "main_window.py").read_text(encoding="utf-8")
        pose_correctives = (
            ROOT / "pose_correctives.py").read_text(encoding="utf-8")
        viewer = (ROOT / "viewer3d.py").read_text(encoding="utf-8")
        slice_source = (ROOT / "sdf_slice.py").read_text(encoding="utf-8")
        compute = (ROOT / "sdf_compute.py").read_text(encoding="utf-8")
        optimization = (
            ROOT / "optimization.py").read_text(encoding="utf-8")
        widgets = (ROOT / "widgets.py").read_text(encoding="utf-8")

        self.assertIn("apply_thickness_relative_blowup(", main_window)
        self.assertIn("relative_blowup_extent_voxels(", main_window)
        self.assertIn(
            ".with_thickness_relative_offset(", main_window)
        self.assertIn(
            'getattr(mesh_result, "blowup_thickness", None)', main_window)

        self.assertIn(
            "build_surface_carried_thickness(", compute)
        self.assertIn(
            "blowup_thickness=blowup_thickness", compute)

        self.assertIn(
            "compute_thickness=self._has_sdf_blowup",
            pose_correctives,
        )
        self.assertIn(
            "apply_thickness_relative_blowup(", pose_correctives)
        self.assertIn(
            "loss_thickness = (", pose_correctives)

        self.assertIn("sdf_blowup_fraction=", main_window)
        self.assertIn(
            "def _apply_blowup_to_region_result", optimization)
        self.assertIn(
            "self._apply_blowup_to_region_result(box_result)", optimization)
        self.assertIn(
            "relative_blowup_extent_voxels(", optimization)

        self.assertIn(
            "def set_sdf_blowup(", widgets)
        self.assertIn(
            "apply_thickness_relative_blowup(", widgets)
        self.assertIn(
            "self._mesh_sdf_panel.set_sdf_blowup", main_window)
        self.assertIn(
            "def _ensure_blowup_thickness(", main_window)
        self.assertIn(
            "compute_blowup_thickness=(", main_window)
        self.assertIn(
            "def set_blowup_thickness(", widgets)

        self.assertGreaterEqual(
            viewer.count("thickness_wp=self._blowup_thickness_wp"), 2)
        self.assertIn("_relative_grid_field_kernel", slice_source)
        self.assertIn("_thickness_relative_grid_value", slice_source)


if __name__ == "__main__":
    unittest.main()
