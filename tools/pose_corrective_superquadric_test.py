"""Shape-exponent regression tests for pose-corrective libraries/workers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import (  # noqa: E402
    BoneLocalEllipsoids,
    BoneEllipsoidMapper,
    initialize_ellipsoids_from_bones,
)
from pose_correctives import (  # noqa: E402
    PoseCorrectiveLibrary,
    PoseCorrectiveWorker,
    _blend_local_seed,
    corrective_from_optimized_local,
)
from skeleton import Bone, Pose, Skeleton, mat4_compose  # noqa: E402
from rig_loader import load_rigged_mesh  # noqa: E402


def _local(eps, *, primitive_type="superquadric"):
    count = len(eps)
    return BoneLocalEllipsoids(
        local_centers=np.array(
            [[0.1 * i, 0.0, 0.0] for i in range(count)], dtype=np.float32),
        local_radii=np.full((count, 3), 0.25, dtype=np.float32),
        local_rotations=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1)),
        bone_assignments=np.zeros(count, dtype=np.int32),
        primitive_type=primitive_type,
        shape_exponents=np.asarray(eps, dtype=np.float32),
    )


class PoseCorrectiveSuperquadricTest(unittest.TestCase):
    def test_corrective_round_trip_and_v4_json(self) -> None:
        base = _local([[0.4, 0.6], [1.2, 1.4]])
        optimized = _local([[0.8, 0.3], [0.6, 1.8]])
        optimized.local_centers += 0.02
        optimized.local_radii *= 1.1
        target_rotation = np.array(
            [0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)])
        pose = Pose(name="Grip", bone_locals={
            0: mat4_compose(
                np.zeros(3), target_rotation, np.ones(3)),
        })
        key = corrective_from_optimized_local(
            base, optimized, "Grip", 0.125, pose=pose)
        library = PoseCorrectiveLibrary(
            base=base, keys=[key], base_pose=Pose.t_pose())
        corrected = library.corrected_bone_local(key)

        np.testing.assert_allclose(
            corrected.shape_exponents, optimized.shape_exponents,
            rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            corrected.local_radii, optimized.local_radii, atol=1.0e-7)
        self.assertEqual(corrected.primitive_type, "superquadric")

        skeleton = Skeleton([Bone(
            name="Root", index=0, parent_index=-1,
            local_bind_transform=np.eye(4),
            inverse_bind_matrix=np.eye(4))])
        payload = library.to_json(skeleton)
        self.assertEqual(payload["version"], 4)
        self.assertEqual(payload["primitive_type"], "superquadric")
        self.assertTrue(all(
            entry["primitive_type"] == "superquadric"
            and len(entry["shape_exponents"]) == 2
            for entry in payload["base"]))
        self.assertEqual(len(
            payload["poses"][0]["delta_log_shape_exponents"]), 2)
        self.assertEqual(len(payload["poses"][0]["ellipsoids"]), 2)
        self.assertEqual(
            payload["poses"][0]["ellipsoids"][1]["id"], 1)
        self.assertEqual(len(
            payload["poses"][0]["ellipsoids"][0]
            ["delta_log_shape_exponents"]), 2)
        pose_payload = payload["poses"][0]["pose"]
        self.assertEqual(
            payload["base_pose"]["descriptor"],
            "bone_local_rotations",
        )
        self.assertEqual(pose_payload["descriptor"], "bone_local_rotations")
        self.assertEqual(pose_payload["bones"][0]["index"], 0)
        np.testing.assert_allclose(
            pose_payload["bones"][0]["local_rotation"],
            target_rotation,
            atol=1.0e-6,
        )

    def test_seed_blends_exponents_in_log_space(self) -> None:
        base = _local([[0.25, 0.5]])
        neighbor = _local([[1.0, 2.0]])
        blended = _blend_local_seed(base, neighbor, 0.5)
        np.testing.assert_allclose(blended.shape_exponents, [[0.5, 1.0]])
        self.assertEqual(blended.primitive_type, "superquadric")

    def test_worker_passes_per_primitive_initial_eps(self) -> None:
        base = _local([[0.35, 0.55], [1.4, 1.8]])
        worker = SimpleNamespace(
            _fit_kwargs={
                "primitive_shape": "superquadric",
                "sq_eps_mode": "per_primitive",
            },
            _sdf_blowup_fraction=0.0,
            _base=base,
        )
        sdf = SimpleNamespace(
            grid=np.zeros((2, 2, 2), dtype=np.float32),
            origin=np.zeros(3, dtype=np.float32),
            dx=1.0,
            n=2,
            thickness=None,
        )
        count = base.num_ellipsoids
        kwargs = PoseCorrectiveWorker._optimizer_kwargs(
            worker,
            sdf,
            base,
            np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
            np.zeros((count, 3), dtype=np.float32),
            np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                (count, 1)),
            None,
            0.0,
        )
        self.assertEqual(kwargs["primitive_shape"], "superquadric")
        np.testing.assert_allclose(kwargs["initial_eps"], base.shape_exponents)

        worker._fit_kwargs = {"primitive_shape": "ellipsoid"}
        with self.assertRaisesRegex(ValueError, "does not match"):
            PoseCorrectiveWorker._optimizer_kwargs(
                worker,
                sdf,
                base,
                np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
                np.zeros((count, 3), dtype=np.float32),
                np.tile(
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    (count, 1)),
                None,
                0.0,
            )

    def test_real_worker_keeps_sq_state_through_one_pose_fit(self) -> None:
        rig = load_rigged_mesh(ROOT / "meshes" / "T-Pose.fbx")
        mapper = BoneEllipsoidMapper(rig.skeleton)
        ellipse_base = initialize_ellipsoids_from_bones(
            rig.skeleton,
            rig.vertices,
            rig.skin_joints,
            rig.skin_weights,
            n_ellipsoids=3,
            max_points_per_bone=128,
            kmeans_iters=1,
        )
        eps = np.array(
            [[0.4, 0.7], [0.9, 1.3], [1.6, 0.55]], dtype=np.float32)
        base = BoneLocalEllipsoids(
            local_centers=ellipse_base.local_centers,
            local_radii=ellipse_base.local_radii,
            local_rotations=ellipse_base.local_rotations,
            bone_assignments=ellipse_base.bone_assignments,
            attachment_joints=ellipse_base.attachment_joints,
            attachment_weights=ellipse_base.attachment_weights,
            primitive_type="superquadric",
            shape_exponents=eps,
        )
        seen_eps: list[np.ndarray] = []
        worker = PoseCorrectiveWorker(
            rigged_mesh=rig,
            mapper=mapper,
            base=base,
            poses=[rig.poses[0]],
            grid_n=8,
            margin=0.12,
            fit_kwargs={
                "primitive_shape": "superquadric",
                "sq_eps_mode": "per_primitive",
                "num_steps": 1,
                "report_every": 1,
                "sample_budget": 128,
                "maintenance_every": 0,
                "superfit": False,
                "local_fit": False,
            },
        )
        worker.pose_fit_progress.connect(
            lambda _i, _step, _loss, _c, _r, _q, shape:
            seen_eps.append(np.asarray(shape, dtype=np.float32).copy()))
        worker.run()
        self.assertIsNotNone(worker.result)
        self.assertTrue(seen_eps)
        self.assertEqual(worker.result.base.primitive_type, "superquadric")
        self.assertEqual(
            worker.result.keys[0].delta_log_shape_exponents.shape, (3, 2))
        self.assertTrue(np.isfinite(seen_eps[-1]).all())


if __name__ == "__main__":
    unittest.main()
