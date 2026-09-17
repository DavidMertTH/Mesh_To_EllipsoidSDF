"""Regression coverage for size-aware mesh-region population budgets."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization import OptimizationWorker  # noqa: E402
from region_budget import build_mesh_region_budget  # noqa: E402
import app_settings  # noqa: E402


class SizeRegionBudgetRegressionTest(unittest.TestCase):
    def test_region_limit_has_a_persistent_training_toggle(self) -> None:
        self.assertIs(
            app_settings.defaults()["size_region_budget_enabled"], True)
        source = (ROOT / "main_window.py").read_text(encoding="utf-8")
        self.assertIn(
            'self._chk_region_budget = QtWidgets.QCheckBox("Regional limit")',
            source,
        )
        self.assertIn(
            "self._chk_region_budget.toggled.connect(\n"
            "            self._on_region_budget_toggled)",
            source,
        )
        self.assertIn(
            'advanced_settings.pop("size_region_budget_enabled", True)',
            source,
        )
        self.assertIn("if region_budget_enabled and not fixed_population:", source)

    def test_area_budget_is_exact_and_favours_small_regions_relatively(self) -> None:
        large = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
        small = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        small.apply_translation((8.0, 0.0, 0.0))
        mesh = trimesh.util.concatenate([large, small])

        budget = build_mesh_region_budget(
            mesh.vertices, mesh.faces, 12,
            target_capacity=6, minimum_capacity=1, area_power=0.5,
        )
        self.assertIsNotNone(budget)
        assert budget is not None
        self.assertEqual(len(budget.centers), 2)
        self.assertEqual(int(np.sum(budget.capacities)), 12)
        self.assertEqual(budget.face_colors.shape, (len(mesh.faces), 4))
        self.assertTrue(np.all(budget.capacities >= 1))

        order = np.argsort(budget.region_areas)
        small_index, large_index = int(order[0]), int(order[-1])
        self.assertGreater(
            int(budget.capacities[large_index]),
            int(budget.capacities[small_index]),
        )
        small_density = (
            budget.capacities[small_index] / budget.region_areas[small_index])
        large_density = (
            budget.capacities[large_index] / budget.region_areas[large_index])
        self.assertGreater(float(small_density), float(large_density))

    def test_spawn_filter_uses_spatial_cap_without_a_rig(self) -> None:
        grid = np.full((8, 8, 8), -0.1, dtype=np.float32)
        worker = OptimizationWorker(
            sdf_target_np=grid,
            origin=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
            dx=0.5,
            n=8,
            num_ellipsoids=2,
            max_ellipsoids=4,
            num_steps=1,
            sample_budget=64,
            local_fit=False,
            spatial_budget_centers_np=np.array(
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            spatial_budget_caps_np=np.array([1, 3], dtype=np.int32),
        )
        existing = np.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        _assignment, counts, caps = worker._bone_growth_state(existing)
        candidates = np.array([
            [-1.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ], dtype=np.float32)
        radii = np.full((4, 3), 0.1, dtype=np.float32)
        rotations = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1))

        kept, _radii, _rotations = \
            worker._filter_spawn_candidates_by_bone_capacity(
                candidates, radii, rotations, counts, caps)

        self.assertEqual(len(kept), 2)
        self.assertTrue(np.all(kept[:, 0] > 0.0))
        np.testing.assert_array_equal(counts, [1, 3])


if __name__ == "__main__":
    unittest.main()
