"""Regression tests for per-primitive superquadric state across population edits."""

from __future__ import annotations

from pathlib import Path
import sys
from types import MethodType
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warp as wp  # noqa: E402

from optimization import OptimizationWorker, _PopulationAdam  # noqa: E402
from superquadric_geometry import quaternion_matrix, volume  # noqa: E402


def _identity_quats(count: int) -> np.ndarray:
    return np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1))


def _worker(**overrides) -> OptimizationWorker:
    kwargs = dict(
        sdf_target_np=np.zeros((5, 5, 5), dtype=np.float32),
        origin=np.zeros(3, dtype=np.float32),
        dx=0.1,
        n=5,
        num_ellipsoids=1,
        max_ellipsoids=4,
        num_steps=2,
        report_every=1,
        sample_budget=64,
        maintenance_every=0,
        local_fit=False,
        primitive_shape="bent_superquadric",
        sq_eps1=1.1,
        sq_eps2=1.2,
    )
    kwargs.update(overrides)
    return OptimizationWorker(**kwargs)


class SuperquadricStateRegressionTest(unittest.TestCase):
    def test_bend_warmup_is_independent_when_epsilon_is_fixed(self) -> None:
        worker = _worker(
            num_steps=10,
            sq_eps_mode="fixed",
            sq_unlock_frac=0.8,
            sq_bend_unlock_frac=0.2,
        )
        self.assertFalse(worker._eps_is_trainable(2))
        self.assertTrue(worker._bend_is_trainable(2))

    def test_provided_initial_population_builds_hard_mirror_layout(self) -> None:
        worker = _worker(symmetry_enabled=True, max_ellipsoids=4)
        worker._sym_axis = 0
        worker._sym_plane = 0.25
        centers = np.array(
            [[0.40, 0.20, 0.20], [0.10, 0.20, 0.20]], dtype=np.float32)
        radii = np.full((2, 3), 0.05, dtype=np.float32)
        rotations = _identity_quats(2)
        eps = np.array([[0.41, 0.73], [1.2, 1.4]], dtype=np.float32)
        bend = np.array([[0.3, -0.2], [-0.7, 0.5]], dtype=np.float32)

        buf = worker._alloc_buffers(
            2, 8, 125, centers, radii, rotations,
            eps_np=eps, bend_np=bend, apply_initial_symmetry=True)
        out_c = buf["pred_centers"].numpy().reshape(-1, 3)
        out_eps = buf["pred_eps"].numpy().reshape(-1, 2)
        out_bend = buf["pred_bend"].numpy().reshape(-1, 2)

        self.assertEqual((worker._sym_n_op, worker._sym_n_so), (0, 1))
        self.assertEqual(len(out_c), 2)
        self.assertAlmostEqual(float(out_c[0, 0] + out_c[1, 0]), 0.5)
        np.testing.assert_allclose(out_eps[1], out_eps[0])
        np.testing.assert_allclose(
            out_bend[1], [-out_bend[0, 0], out_bend[0, 1]])

    def test_naive_path_sets_up_and_projects_shape_symmetry(self) -> None:
        centers = np.array(
            [[0.40, 0.20, 0.20], [0.10, 0.20, 0.20]], dtype=np.float32)
        radii = np.full((2, 3), 0.05, dtype=np.float32)
        rotations = _identity_quats(2)
        eps = np.repeat(
            np.array([[0.52, 0.68]], dtype=np.float32), 2, axis=0)
        bend = np.repeat(
            np.array([[0.25, -0.15]], dtype=np.float32), 2, axis=0)
        worker = _worker(
            method="naive",
            num_steps=1,
            num_ellipsoids=2,
            max_ellipsoids=4,
            symmetry_enabled=True,
            sq_eps_mode="fixed",
            initial_centers=centers,
            initial_radii=radii,
            initial_rotations=rotations,
            initial_eps=eps,
            initial_bend=bend,
        )
        frames: list[tuple[np.ndarray, np.ndarray]] = []
        worker.step_visual.connect(
            lambda _step, _loss, c, _r, _q, extra:
            frames.append((
                np.asarray(c, dtype=np.float32).copy(),
                np.asarray(extra, dtype=np.float32).copy(),
            )))

        worker._run_naive()

        self.assertEqual((worker._sym_axis, worker._sym_n_so), (0, 1))
        self.assertTrue(frames)
        out_c, out_shape = frames[-1]
        np.testing.assert_allclose(out_c[0, 0] + out_c[1, 0], 0.5, atol=1e-6)
        np.testing.assert_allclose(out_shape[1, :2], out_shape[0, :2])
        np.testing.assert_allclose(
            out_shape[1, 2:], [-out_shape[0, 2], out_shape[0, 3]])

    def test_naive_maintenance_observes_latest_raw_shape_update(self) -> None:
        worker = _worker(
            method="naive",
            num_steps=3,
            report_every=100,
            sq_eps_mode="per_primitive",
            sq_unlock_frac=0.0,
            sq_bend_unlock_frac=0.0,
            sq_eps_lr_mult=1.0,
            sq_bend_lr_mult=1.0,
            initial_centers=np.array([[0.25, 0.25, 0.25]], dtype=np.float32),
            initial_radii=np.array([[0.12, 0.09, 0.07]], dtype=np.float32),
            initial_rotations=_identity_quats(1),
            initial_eps=np.array([[0.75, 0.85]], dtype=np.float32),
            initial_bend=np.array([[0.2, -0.15]], dtype=np.float32),
        )
        observed: dict[int, np.ndarray] = {}

        def observe(
            self, step, _centers, _radii, _rotations,
            pred_eps=None, pred_bend=None,
        ):
            wp.synchronize_device()
            observed[int(step)] = np.concatenate([
                pred_eps.numpy().reshape(-1, 2),
                pred_bend.numpy().reshape(-1, 2),
            ], axis=1)
            return None

        worker._maybe_maintain = MethodType(observe, worker)
        worker._run_naive()

        self.assertEqual(sorted(observed), [0, 1, 2])
        self.assertGreater(
            float(np.max(np.abs(observed[2] - observed[1]))), 1.0e-7,
            "maintenance saw stale decoded epsilon/bend after a raw update",
        )

    def test_shared_eps_spawn_inherits_learned_global_value(self) -> None:
        worker = _worker(
            num_ellipsoids=2,
            sq_eps_mode="shared",
            sq_eps1=1.1,
            sq_eps2=1.2,
            prune_enabled=False,
        )
        learned = np.array([[0.31, 0.57]], dtype=np.float32)
        centers = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
        radii = np.array([[0.08, 0.07, 0.06]], dtype=np.float32)
        rotations = _identity_quats(1)
        bend = np.zeros((1, 2), dtype=np.float32)

        worker._compute_coverage_info = lambda *_args, **_kw: {"valid": False}
        worker._spawn_at_errors = lambda *_args, **_kw: (
            np.array([[0.35, 0.2, 0.2]], dtype=np.float32),
            np.array([[0.04, 0.04, 0.04]], dtype=np.float32),
            _identity_quats(1),
        )

        out = worker._do_maintenance(
            centers, radii, rotations, learned, bend)
        out_eps = out[3]
        self.assertEqual(out_eps.shape, (2, 2))
        np.testing.assert_allclose(
            out_eps, np.repeat(learned, 2, axis=0), atol=1.0e-7)
        self.assertFalse(np.allclose(out_eps[1], [1.1, 1.2]))

    def test_shared_eps_survives_complete_prune_for_spawn_sizing(self) -> None:
        worker = _worker(
            num_ellipsoids=1,
            sq_eps_mode="shared",
            min_volume_abs=1.0,
            prune_enabled=False,
        )
        learned = np.array([[0.27, 0.49]], dtype=np.float32)
        captured: dict[str, np.ndarray] = {}

        worker._compute_coverage_info = lambda *_args, **_kw: {"valid": False}

        def spawn(*_args, **kwargs):
            captured["reference"] = np.asarray(
                kwargs["new_eps_reference"], dtype=np.float32).copy()
            return (
                np.array([[0.25, 0.25, 0.25]], dtype=np.float32),
                np.array([[0.04, 0.04, 0.04]], dtype=np.float32),
                _identity_quats(1),
            )

        worker._spawn_at_errors = spawn
        result = worker._do_maintenance(
            np.array([[0.2, 0.2, 0.2]], dtype=np.float32),
            np.array([[0.01, 0.01, 0.01]], dtype=np.float32),
            _identity_quats(1),
            learned,
            np.zeros((1, 2), dtype=np.float32),
        )
        np.testing.assert_allclose(captured["reference"], learned)
        np.testing.assert_allclose(result[3], learned)

    def test_symmetry_transforms_bend_in_local_mirror_frame(self) -> None:
        for axis in range(3):
            with self.subTest(axis=axis):
                worker = _worker(symmetry_enabled=True, max_ellipsoids=3)
                worker._sym_axis = axis
                worker._sym_plane = 0.0
                centers = np.zeros((3, 3), dtype=np.float32)
                centers[0, axis] = 0.2
                centers[1, axis] = -0.2
                # Row 2 lies on the plane and exercises the self-mirror projection.
                radii = np.full((3, 3), 0.05, dtype=np.float32)
                rotations = _identity_quats(3)
                eps = np.array(
                    [[0.3, 0.6], [1.2, 1.5], [0.8, 0.9]], dtype=np.float32)
                bend = np.array(
                    [[0.4, -0.7], [1.0, 1.1], [0.5, -0.6]], dtype=np.float32)

                out_c, out_r, out_q, out_eps, out_bend = \
                    worker._build_symmetric_layout(
                        centers, radii, rotations, eps, bend)
                self.assertEqual(len(out_c), 3)
                # Layout is [on-plane, positive source, reflected mirror].
                np.testing.assert_allclose(out_eps[2], out_eps[1])
                expected_mirror = out_bend[1].copy()
                if axis in (0, 1):
                    expected_mirror[axis] *= -1.0
                    self.assertEqual(float(out_bend[0, axis]), 0.0)
                np.testing.assert_allclose(out_bend[2], expected_mirror)

                buf = worker._alloc_buffers(
                    3, 8, 125, out_c, out_r, out_q,
                    eps_np=out_eps, bend_np=out_bend)
                source_bend = np.array([0.25, -0.35], dtype=np.float32)
                device_bend = buf["pred_bend"].numpy().reshape(-1, 2)
                device_bend[1] = source_bend
                buf["pred_bend"].assign(
                    np.ascontiguousarray(device_bend.reshape(-1)))
                worker._project_symmetry_inplace(
                    buf["pred_centers"], buf["pred_radii"],
                    buf["pred_rot_flat"], buf["pred_eps"], buf["pred_bend"])
                projected = buf["pred_bend"].numpy().reshape(-1, 2)
                expected_projected = source_bend.copy()
                if axis in (0, 1):
                    expected_projected[axis] *= -1.0
                np.testing.assert_allclose(projected[2], expected_projected)

    def test_split_children_inherit_parent_eps_and_bend(self) -> None:
        worker = _worker(
            superfit=True,
            superfit_every=1,
            densify_until_frac=1.0,
            split_enabled=True,
            split_per_round=1,
            spawn_underrep=False,
            merge_enabled=False,
            prune_enabled=False,
            max_ellipsoids=2,
            num_steps=4,
        )
        worker.blockSignals(True)

        centers = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
        radii = np.array([[0.12, 0.09, 0.07]], dtype=np.float32)
        rotations = _identity_quats(1)
        eps = np.array([[0.36, 0.71]], dtype=np.float32)
        bend = np.array([[0.45, -0.2]], dtype=np.float32)
        buf = worker._alloc_buffers(
            1, 8, 125, centers, radii, rotations,
            eps_np=eps, bend_np=bend)

        worker._detect_outside_ellipsoids = lambda *_args, **_kw: np.empty(0, int)
        worker._detect_degenerate_ellipsoids = lambda *_args, **_kw: np.empty(0, int)
        worker._detect_bridging_ellipsoids = lambda *_args, **_kw: np.array([0])
        worker._detect_protruding_ellipsoids = lambda *_args, **_kw: np.empty(0, int)
        worker._detect_worst_regions = lambda *_args, **_kw: []
        worker._reserve_split_bone_capacity = lambda *_args, **_kw: True

        result = worker._maybe_superfit(
            1, buf["pred_centers"], buf["pred_radii"], buf["pred_rot_flat"],
            buf["pred_eps"], buf["pred_bend"])
        self.assertIsNotNone(result)
        out_c, out_r, out_q, out_eps, out_bend = result
        self.assertEqual(len(out_c), 2)
        self.assertEqual(len(out_c), len(out_r))
        self.assertEqual(len(out_c), len(out_q))
        np.testing.assert_allclose(out_eps, np.repeat(eps, 2, axis=0))
        np.testing.assert_allclose(out_bend, np.repeat(bend, 2, axis=0))
        np.testing.assert_array_equal(worker._last_population_lineage, [0, 0])

    def test_adam_lineage_preserves_survivors_and_resets_spawns(self) -> None:
        snapshot = {
            "first": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "second": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            "age": np.array([7, 7, 11, 11], dtype=np.int32),
        }
        # Two scalar components per primitive: row 1 survives, row 0 is split
        # into two descendants, and the final row is a genuine spawn.
        remapped = _PopulationAdam.remap(snapshot, [1, 0, 0, -1], width=2)
        np.testing.assert_array_equal(
            remapped["first"], [3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            remapped["age"], [11, 11, 7, 7, 7, 7, 0, 0])

    def test_merge_volume_weights_eps_and_bend(self) -> None:
        worker = _worker(merge_enabled=True, merge_per_round=1)
        centers = np.zeros((2, 3), dtype=np.float32)
        radii = np.array(
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.2]], dtype=np.float32)
        rotations = _identity_quats(2)
        eps = np.array([[0.2, 0.4], [1.1, 1.6]], dtype=np.float32)
        bend = np.array([[-0.6, 0.2], [0.3, 0.8]], dtype=np.float32)

        worker._merge_changes_surface = lambda *_args, **_kw: False
        worker._merge_increases_loss = lambda *_args, **_kw: False
        worker._critical_replacement_worse = lambda *_args, **_kw: False
        out_c, out_r, out_q, out_eps, out_bend, count = worker._detect_merges(
            centers, radii, rotations, eps, bend)

        self.assertEqual(count, 1)
        self.assertEqual(len(out_c), len(out_r))
        self.assertEqual(len(out_c), len(out_q))
        weights = np.array(
            [volume(radii[i], eps[i]) for i in range(2)], dtype=np.float64)
        expected_eps = np.average(eps, axis=0, weights=weights)
        expected_world_curvature = np.r_[
            np.average(bend, axis=0, weights=weights), 0.0]
        merged_matrix = quaternion_matrix(out_q[0])
        expected_world_curvature -= (
            merged_matrix[:, 2]
            * float(merged_matrix[:, 2] @ expected_world_curvature))
        actual_world_curvature = merged_matrix[:, :2] @ out_bend[0]
        np.testing.assert_allclose(out_eps[0], expected_eps, rtol=1e-6)
        np.testing.assert_allclose(
            actual_world_curvature, expected_world_curvature,
            rtol=2e-5, atol=2e-5)

    def test_global_population_rebuild_does_not_reset_shape_state(self) -> None:
        centers = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
        radii = np.array([[0.12, 0.1, 0.08]], dtype=np.float32)
        rotations = _identity_quats(1)
        custom_eps = np.array([[0.31, 0.63]], dtype=np.float32)
        custom_bend = np.array([[0.4, -0.25]], dtype=np.float32)
        worker = _worker(
            superfit=True,
            superfit_every=1,
            densify_until_frac=1.0,
            lr_init=0.0,
            lr_final=0.0,
            initial_centers=centers,
            initial_radii=radii,
            initial_rotations=rotations,
            initial_eps=custom_eps,
            initial_bend=custom_bend,
            sq_eps1=float(custom_eps[0, 0]),
            sq_eps2=float(custom_eps[0, 1]),
            sq_eps_mode="fixed",
            sq_bend_unlock_frac=1.0,
        )
        frames: list[np.ndarray] = []
        worker.step_visual.connect(
            lambda _step, _loss, _c, _r, _q, extra:
            frames.append(np.asarray(extra, dtype=np.float32).copy()))

        def force_one_noop_rebuild(
            self, step, pred_centers, pred_radii, pred_rot_flat,
            pred_eps=None, pred_bend=None,
        ):
            if step == 1:
                wp.synchronize_device()
                return (
                    pred_centers.numpy().copy(),
                    pred_radii.numpy().copy(),
                    pred_rot_flat.numpy().reshape(-1, 4).copy(),
                    pred_eps.numpy().reshape(-1, 2).copy(),
                    pred_bend.numpy().reshape(-1, 2).copy(),
                )
            return None

        worker._maybe_superfit = MethodType(force_one_noop_rebuild, worker)
        worker.run()

        self.assertTrue(frames)
        np.testing.assert_allclose(frames[-1][:, :2], custom_eps, atol=1e-6)
        np.testing.assert_allclose(frames[-1][:, 2:], custom_bend, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
