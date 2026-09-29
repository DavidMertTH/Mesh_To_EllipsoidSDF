"""Regression coverage for continuous inverse-thickness surface sampling."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_settings  # noqa: E402
import sdf_compute as sdf_compute_module  # noqa: E402
from main_window import SdfWorker  # noqa: E402
from optimization import BandSampler, OptimizationWorker  # noqa: E402
from sdf_compute import SdfComputer  # noqa: E402
from sdf_samples import SdfSampleSet  # noqa: E402
from thin_sampling import thickness_sampling_probabilities  # noqa: E402


class ContinuousThicknessSamplingTest(unittest.TestCase):
    def test_setting_exposes_continuous_power(self) -> None:
        defaults = app_settings.defaults()
        self.assertAlmostEqual(defaults["thickness_sampling_power"], 1.0)
        self.assertNotIn("thin_surface_fraction", defaults)
        self.assertNotIn("thin_sample_bias", defaults)
        field = next(
            field
            for _tab, groups in app_settings.SETTINGS_SPEC
            for _group, fields in groups
            for field in fields
            if field[0] == "thickness_sampling_power"
        )
        self.assertEqual(
            field[2:8], ("float", 0.0, 2.0, 0.10, 2, 1.0))
        self.assertIn("no thin/thick split", field[8])

    def test_sdf_worker_preserves_original_positional_argument_order(self) -> None:
        names = list(inspect.signature(SdfWorker.__init__).parameters)
        original_tail = [
            "compute_sparse_samples",
            "max_dist",
            "sdf_blowup_fraction",
            "sdf_blowup_capacity_fraction",
            "sdf_blowup_carrier_fraction",
            "sdf_guard_voxels_per_side",
        ]
        start = names.index("compute_sparse_samples")
        self.assertEqual(names[start:start + len(original_tail)], original_tail)

    def test_recent_fraction_migrates_to_same_numeric_power(self) -> None:
        cases = ((0.0, 0.0), (0.42, 0.42), (1.0, 1.0), ("nan", 1.0))
        for legacy, expected in cases:
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "app_settings.json"
                path.write_text(json.dumps({
                    "thin_surface_fraction": legacy,
                    "unrelated_future_key": 7,
                }), encoding="utf-8")
                with mock.patch.object(app_settings, "_FILE", path):
                    loaded = app_settings.load()
                persisted = json.loads(path.read_text(encoding="utf-8"))
                self.assertAlmostEqual(
                    loaded["thickness_sampling_power"], expected)
                self.assertNotIn("thin_surface_fraction", persisted)
                self.assertAlmostEqual(
                    persisted["thickness_sampling_power"], expected)
                self.assertEqual(persisted["unrelated_future_key"], 7)

    def test_older_bias_migrates_to_approximate_strength(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app_settings.json"
            path.write_text(json.dumps({"thin_sample_bias": 1.0}),
                            encoding="utf-8")
            with mock.patch.object(app_settings, "_FILE", path):
                loaded = app_settings.load()
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(loaded["thickness_sampling_power"], 0.30)
        self.assertNotIn("thin_sample_bias", persisted)

    def test_canonical_setting_wins_over_legacy_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app_settings.json"
            path.write_text(json.dumps({
                "thickness_sampling_power": 1.25,
                "thin_surface_fraction": 0.42,
                "thin_sample_bias": 2.0,
            }), encoding="utf-8")
            with mock.patch.object(app_settings, "_FILE", path):
                loaded = app_settings.load()
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(loaded["thickness_sampling_power"], 1.25)
        self.assertNotIn("thin_surface_fraction", persisted)
        self.assertNotIn("thin_sample_bias", persisted)

    def test_probability_ratios_are_continuous_and_scale_free(self) -> None:
        thickness = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
        uniform = thickness_sampling_probabilities(thickness, 0.0)
        inverse = thickness_sampling_probabilities(thickness, 1.0)
        inverse_square = thickness_sampling_probabilities(thickness, 2.0)
        np.testing.assert_allclose(uniform, np.full(3, 1.0 / 3.0))
        np.testing.assert_allclose(inverse, np.asarray([4.0, 2.0, 1.0]) / 7.0)
        np.testing.assert_allclose(
            inverse_square, np.asarray([16.0, 4.0, 1.0]) / 21.0)
        np.testing.assert_allclose(
            inverse,
            thickness_sampling_probabilities(100.0 * thickness, 1.0),
        )

    def test_unresolved_thickness_is_finite_and_neutral(self) -> None:
        probs = thickness_sampling_probabilities(
            np.asarray([1.0, 0.0, np.nan, np.inf, 2.0]), 1.0)
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(np.all(probs > 0.0))
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertAlmostEqual(probs[1], probs[2])
        self.assertAlmostEqual(probs[2], probs[3])

    def test_band_sampler_uses_one_continuous_distribution(self) -> None:
        target = np.zeros(3, dtype=np.float32)
        thickness = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
        sampler = BandSampler(
            target,
            batch_size=70_000,
            band=0.1,
            surface_fraction=1.0,
            rng=np.random.default_rng(9),
            flat_thickness=thickness,
            thickness_sampling_power=1.0,
        )
        expected = np.asarray([4.0, 2.0, 1.0]) / 7.0
        probabilities = np.diff(np.concatenate([
            np.asarray([0.0]), sampler._band_cdf]))
        np.testing.assert_allclose(probabilities, expected)
        counts = np.bincount(sampler.next_batch(), minlength=3) / 70_000.0
        np.testing.assert_allclose(counts, expected, atol=0.006)
        self.assertTrue(np.all(counts > 0.0))
        self.assertFalse(hasattr(sampler, "_band_thin"))

    def test_band_sampler_keeps_far_field_quota(self) -> None:
        target = np.zeros(100, dtype=np.float32)
        coarse = np.zeros(100, dtype=np.bool_)
        coarse[-20:] = True
        sampler = BandSampler(
            target,
            batch_size=100,
            band=0.1,
            surface_fraction=1.0,
            flat_thickness=np.linspace(1.0, 2.0, 100),
            coarse_mask=coarse,
            thickness_sampling_power=1.0,
        )
        self.assertEqual(sampler.n_surf, 80)
        self.assertEqual(sampler.n_far, 20)
        self.assertEqual(sampler.n_rest, 0)

    def test_low_level_legacy_aliases_remain_accepted(self) -> None:
        target = np.zeros(10, dtype=np.float32)
        thickness = np.linspace(1.0, 2.0, 10, dtype=np.float32)
        recent = BandSampler(
            target, 10, 0.1, 1.0,
            flat_thickness=thickness, thin_fraction=0.4)
        older = BandSampler(
            target, 10, 0.1, 1.0,
            flat_thickness=thickness, thin_bias=1.0)
        self.assertAlmostEqual(recent._thickness_sampling_power, 0.4)
        self.assertAlmostEqual(older._thickness_sampling_power, 0.3)

    def test_negative_power_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            BandSampler(
                np.zeros(3, dtype=np.float32),
                batch_size=3,
                band=0.1,
                surface_fraction=1.0,
                flat_thickness=np.asarray([1.0, 2.0, 4.0]),
                thickness_sampling_power=-1.0,
            )

    def test_mismatched_preweighted_sparse_pool_is_rejected(self) -> None:
        samples = SdfSampleSet(
            points=np.zeros((2, 3), dtype=np.float32),
            values=np.zeros(2, dtype=np.float32),
            thickness=np.asarray([1.0, 2.0], dtype=np.float32),
            thickness_sampling_power=2.0,
        )
        with self.assertRaisesRegex(ValueError, "rebuild the sparse sample"):
            OptimizationWorker(
                sdf_target_np=np.zeros((2, 2, 2), dtype=np.float32),
                sdf_samples=samples,
                origin=np.zeros(3, dtype=np.float32),
                dx=1.0,
                n=2,
                num_ellipsoids=1,
                thickness_sampling_power=1.0,
            )

    @staticmethod
    def _two_triangle_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Areas are 50 and 0.5 (100:1); thicknesses are 100 and 1 (100:1).
        # At power 1, area/thickness gives both triangles equal probability.
        vertices = np.asarray([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
            [20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [20.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        thickness = np.asarray(
            [100.0, 100.0, 100.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return vertices, faces, thickness

    def test_sparse_pool_draw_is_area_over_thickness(self) -> None:
        vertices, faces, thickness = self._two_triangle_mesh()
        computer = SdfComputer(device="cpu")
        computer.set_mesh(vertices, faces)
        samples = computer.compute_sparse_samples(
            n=16,
            margin=0.1,
            surface_samples=20_000,
            offsets_vox=(0.0,),
            coarse_n=4,
            seed=11,
            vertex_thickness=thickness,
            thickness_sampling_power=1.0,
        )
        surface_points = samples.points[:20_000]
        small_count = int(np.count_nonzero(surface_points[:, 0] >= 20.0))
        self.assertAlmostEqual(small_count / 20_000.0, 0.5, delta=0.015)
        self.assertAlmostEqual(samples.thickness_sampling_power, 1.0)
        self.assertGreater(
            len(np.unique(np.round(
                surface_points[surface_points[:, 0] >= 20.0], decimals=5),
                axis=0)),
            1_000,
        )

    def test_sparse_power_zero_preserves_area_distribution(self) -> None:
        vertices, faces, thickness = self._two_triangle_mesh()
        computer = SdfComputer(device="cpu")
        computer.set_mesh(vertices, faces)
        samples = computer.compute_sparse_samples(
            n=16,
            margin=0.1,
            surface_samples=20_000,
            offsets_vox=(0.0,),
            coarse_n=4,
            seed=11,
            vertex_thickness=thickness,
            thickness_sampling_power=0.0,
        )
        small_fraction = float(np.mean(samples.points[:20_000, 0] >= 20.0))
        self.assertAlmostEqual(small_fraction, 1.0 / 101.0, delta=0.003)
        self.assertAlmostEqual(samples.thickness_sampling_power, 0.0)

    def test_sparse_sampling_is_continuous_inside_a_triangle(self) -> None:
        vertices = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        # Corner zero is much thinner than the opposite edge.
        thickness = np.asarray([1.0, 100.0, 100.0], dtype=np.float32)
        computer = SdfComputer(device="cpu")
        computer.set_mesh(vertices, faces)
        samples = computer.compute_sparse_samples(
            n=8,
            margin=0.1,
            surface_samples=20_000,
            offsets_vox=(0.0,),
            coarse_n=4,
            seed=23,
            vertex_thickness=thickness,
            thickness_sampling_power=1.0,
        )
        points = samples.points[:20_000]
        barycentric_at_thin_corner = 1.0 - points[:, 0] - points[:, 1]
        # Uniform triangle sampling has mean 1/3.  The continuous linear
        # inverse-thickness density approaches 1/2 near this thin corner.
        self.assertGreater(float(np.mean(barycentric_at_thin_corner)), 0.48)

    def test_sparse_only_path_weights_before_surface_draw(self) -> None:
        vertices, faces, _thickness = self._two_triangle_mesh()
        computer = SdfComputer(device="cpu")
        computer.set_mesh(vertices, faces)
        computer.query_points = lambda points, max_dist: np.zeros(
            len(points), dtype=np.float32)

        def synthetic_thickness(grid, _dx):
            field = np.full_like(grid, 100.0, dtype=np.float32)
            field[:, :, (3 * field.shape[2]) // 4:] = 1.0
            return field

        with mock.patch.object(
                sdf_compute_module, "local_thickness", synthetic_thickness):
            samples = computer.compute_sparse_samples(
                n=16,
                margin=0.1,
                surface_samples=20_000,
                offsets_vox=(0.0,),
                coarse_n=16,
                seed=13,
                thickness_sampling_power=1.0,
            )

        small_fraction = float(np.mean(samples.points[:20_000, 0] >= 20.0))
        self.assertAlmostEqual(small_fraction, 0.5, delta=0.03)
        self.assertAlmostEqual(samples.thickness_sampling_power, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
