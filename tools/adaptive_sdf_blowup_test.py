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
    apply_thickness_limited_blowup,
    build_surface_carried_thickness,
    conservative_mirror_min,
    sparse_band_offsets,
    thickness_limited_offsets,
)
from sdf_samples import SdfSampleSet  # noqa: E402
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

        offsets = thickness_limited_offsets(
            values, 0.0, thickness, dx=0.5)
        result = apply_thickness_limited_blowup(
            values, 0.0, thickness, dx=0.5)

        np.testing.assert_array_equal(offsets, np.zeros_like(values))
        np.testing.assert_array_equal(result, values)
        self.assertEqual(offsets.dtype, np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_both_signs_cap_thin_features_but_leave_thick_ones_full(self) -> None:
        values = np.array([-0.1, 0.1], dtype=np.float32)
        thickness = np.array([0.8, 40.0], dtype=np.float32)

        for requested in (1.6, -1.6):
            with self.subTest(requested=requested):
                direction = np.sign(requested)
                expected_offsets = np.array(
                    [direction * 0.2, requested],
                    dtype=np.float32,
                )
                offsets = thickness_limited_offsets(
                    values,
                    requested,
                    thickness,
                    dx=0.5,
                    max_thickness_fraction=0.25,
                )
                result = apply_thickness_limited_blowup(
                    values,
                    requested,
                    thickness,
                    dx=0.5,
                    max_thickness_fraction=0.25,
                )

                np.testing.assert_allclose(
                    offsets, expected_offsets, rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(
                    result,
                    values + expected_offsets,
                    rtol=0.0,
                    atol=1.0e-6,
                )

    def test_unknown_thickness_fails_closed_everywhere(self) -> None:
        values = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
        thickness = np.zeros_like(values)

        for requested in (1.25, -1.25):
            with self.subTest(requested=requested):
                offsets = thickness_limited_offsets(
                    values, requested, thickness, dx=0.5)
                expected = np.zeros_like(values)
                np.testing.assert_array_equal(offsets, expected)

                result = apply_thickness_limited_blowup(
                    values, requested, thickness, dx=0.5)
                np.testing.assert_array_equal(result, values + expected)

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

        positive = thickness_limited_offsets(
            values, 2.0, thickness, dx=0.5)
        negative = thickness_limited_offsets(
            values, -2.0, thickness, dx=0.5)
        reflected = thickness_limited_offsets(
            values[::-1], 2.0, thickness[::-1], dx=0.5)

        np.testing.assert_array_equal(positive, -negative)
        np.testing.assert_array_equal(reflected, positive[::-1])
        np.testing.assert_array_equal(positive, positive[::-1])

    def test_dense_and_flat_sparse_arrays_have_identical_results(self) -> None:
        values = np.linspace(-6.0, 6.0, 60, dtype=np.float32).reshape(
            3, 4, 5)
        thickness = np.linspace(
            0.25, 12.0, values.size, dtype=np.float32).reshape(values.shape)
        thickness.ravel()[::7] = 0.0
        requested = -2.25
        dx = 0.4

        dense_offsets = thickness_limited_offsets(
            values, requested, thickness, dx)
        sparse_offsets = thickness_limited_offsets(
            values.ravel(), requested, thickness.ravel(), dx)
        dense_result = apply_thickness_limited_blowup(
            values, requested, thickness, dx)
        sparse_result = apply_thickness_limited_blowup(
            values.ravel(), requested, thickness.ravel(), dx)

        np.testing.assert_array_equal(
            dense_offsets, sparse_offsets.reshape(values.shape))
        np.testing.assert_array_equal(
            dense_result, sparse_result.reshape(values.shape))

    def test_world_offset_does_not_change_with_grid_spacing(self) -> None:
        values = np.array([-0.1, 0.1], dtype=np.float32)
        thickness = np.array([1.2, 8.0], dtype=np.float32)
        requested_world = -0.6

        fine = thickness_limited_offsets(
            values, requested_world, thickness, dx=0.05)
        coarse = thickness_limited_offsets(
            values, requested_world, thickness, dx=0.5)

        np.testing.assert_array_equal(fine, coarse)
        np.testing.assert_allclose(
            fine, [-0.3, -0.6], rtol=0.0, atol=1.0e-6)

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
        # inward side.  Failing closed is what prevents an unsafe large blowup.
        self.assertGreater(grid[1, 1, 12], 0.0)
        self.assertEqual(carried[1, 1, 12], 0.0)
        offset = thickness_limited_offsets(
            grid[1, 1, 12:13], -4.0, carried[1, 1, 12:13], dx)
        self.assertEqual(float(offset[0]), 0.0)

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
        self.assertEqual(carried[1, 1, 12], 0.0)

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
            grid, thickness, dx, max_exterior_vox=14.0)

        exterior_band = (grid >= 0.0) & (grid <= 14.0 * dx)
        coverage = float(np.mean(carried[exterior_band] > 0.0))
        self.assertGreater(coverage, 0.99)

    def test_carrier_margin_covers_optimizer_surface_band(self) -> None:
        # Max UI request is ten voxels and the optimizer samples a three-voxel
        # surface band.  One extra voxel avoids a secondary unprotected edge.
        self.assertGreaterEqual(BLOWUP_CARRIER_MARGIN_VOXELS, 4.0)

    def test_local_fit_resamples_world_thickness_and_keeps_world_offset(
            self) -> None:
        from optimization import OptimizationWorker

        worker = SimpleNamespace(
            _sdf_blowup_offset=-1.0,
            _sdf_blowup_max_thickness_fraction=0.25,
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
        np.testing.assert_allclose(box_min, -2.25, atol=1.0e-6)
        np.testing.assert_allclose(box_max, 2.25, atol=1.0e-6)

    def test_optimizer_symmetry_chooses_less_aggressive_blowup_pair(
            self) -> None:
        from optimization import OptimizationWorker

        for requested, expected in (
            (-1.0, np.array([[[-2.0, -2.0]]], dtype=np.float32)),
            (1.0, np.array([[[-5.0, -5.0]]], dtype=np.float32)),
        ):
            with self.subTest(requested=requested):
                worker = SimpleNamespace(
                    _sdf_target_np=np.array(
                        [[[-2.0, -5.0]]], dtype=np.float32),
                    _thickness_np=np.array(
                        [[[1.0, 10.0]]], dtype=np.float32),
                    _sdf_blowup_offset=requested,
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

    def test_sparse_sample_adaptive_offset_preserves_metadata(self) -> None:
        samples = SdfSampleSet(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            dtype=np.float32),
            values=np.array([0.0, 10.0], dtype=np.float32),
            thickness=np.array([0.8, 40.0], dtype=np.float32),
            dx=0.5,
            source="adaptive-test",
            coarse_mask=np.array([False, True]),
        )
        adjusted = samples.with_thickness_limited_offset(-1.5)

        np.testing.assert_array_equal(adjusted.points, samples.points)
        np.testing.assert_array_equal(adjusted.thickness, samples.thickness)
        np.testing.assert_array_equal(
            adjusted.coarse_mask, samples.coarse_mask)
        np.testing.assert_allclose(
            adjusted.values,
            np.array([-0.2, 8.5], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertEqual(adjusted.dx, samples.dx)
        self.assertEqual(adjusted.source, samples.source)

    def test_application_paths_use_the_same_adaptive_contract(self) -> None:
        main_window = (ROOT / "main_window.py").read_text(encoding="utf-8")
        pose_correctives = (
            ROOT / "pose_correctives.py").read_text(encoding="utf-8")
        viewer = (ROOT / "viewer3d.py").read_text(encoding="utf-8")
        slice_source = (ROOT / "sdf_slice.py").read_text(encoding="utf-8")
        compute = (ROOT / "sdf_compute.py").read_text(encoding="utf-8")
        optimization = (
            ROOT / "optimization.py").read_text(encoding="utf-8")
        widgets = (ROOT / "widgets.py").read_text(encoding="utf-8")

        self.assertIn("apply_thickness_limited_blowup(", main_window)
        self.assertIn("sparse_band_offsets(blowup_vox)", main_window)
        self.assertIn(
            ".with_thickness_limited_offset(float(blowup))", main_window)
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
            "apply_thickness_limited_blowup(", pose_correctives)
        self.assertIn(
            "loss_thickness = (", pose_correctives)

        self.assertIn(
            "sdf_blowup_offset=float(blowup)", main_window)
        self.assertIn(
            "sdf_blowup_offset=blowup_offset", main_window)
        self.assertIn(
            "def _apply_blowup_to_region_result", optimization)
        self.assertIn(
            "self._apply_blowup_to_region_result(box_result)", optimization)
        self.assertIn(
            "abs(float(self._sdf_blowup_offset))", optimization)

        self.assertIn(
            "def set_sdf_blowup(", widgets)
        self.assertIn(
            "apply_thickness_limited_blowup(", widgets)
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
        self.assertIn("_adaptive_grid_field_kernel", slice_source)
        self.assertIn("_thickness_limited_grid_value", slice_source)


if __name__ == "__main__":
    unittest.main()
