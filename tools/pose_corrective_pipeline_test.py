"""Headless regression test for sequential pose-corrective fitting.

This intentionally uses real rigged FBX geometry plus multi-frame clips from
poses/ so the test covers the actual target-mesh and bone-motion path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import (  # noqa: E402
    BoneEllipsoidMapper,
    initialize_ellipsoids_from_bones,
)
from ellipsoid import best_device  # noqa: E402
from pose_correctives import PoseCorrectiveWorker  # noqa: E402
from rig_loader import load_rigged_mesh  # noqa: E402
from skeleton import interpolate_pose  # noqa: E402
from skinning import deform_mesh  # noqa: E402
import pose_library  # noqa: E402


def _sample_clip(rig, clip_name: str, count: int):
    clips = pose_library.load_all_clips(rig.skeleton)
    for clip in clips:
        if clip.name == clip_name:
            if len(clip.poses) < 2:
                raise AssertionError(f"clip {clip_name!r} is not multi-frame")
            idxs = np.linspace(0, len(clip.poses) - 1, count, dtype=np.int32)
            return [clip.poses[int(i)] for i in idxs]
    names = ", ".join(c.name for c in clips)
    raise AssertionError(f"clip {clip_name!r} not found. Available: {names}")


def _max_vertex_delta(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    first = np.asarray(frames[0], dtype=np.float32)
    return max(
        float(np.max(np.linalg.norm(np.asarray(f, dtype=np.float32) - first, axis=1)))
        for f in frames[1:]
    )


def run_case(mesh_name: str, clip_name: str, pose_count: int = 3) -> None:
    rig = load_rigged_mesh(ROOT / "meshes" / mesh_name)
    poses = _sample_clip(rig, clip_name, pose_count)
    mapper = BoneEllipsoidMapper(rig.skeleton)
    base = initialize_ellipsoids_from_bones(
        rig.skeleton,
        rig.vertices,
        rig.skin_joints,
        rig.skin_weights,
        n_ellipsoids=8,
        max_points_per_bone=512,
        kmeans_iters=2,
    )
    assert base.num_ellipsoids == 8
    assert base.attachment_joints is not None
    assert base.attachment_weights is not None
    np.testing.assert_allclose(
        np.sum(base.attachment_weights, axis=1),
        np.ones(base.num_ellipsoids, dtype=np.float32),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    events: list[tuple[str, int]] = []
    target_vertices: list[np.ndarray] = []
    target_names: list[str] = []

    worker = PoseCorrectiveWorker(
        rigged_mesh=rig,
        mapper=mapper,
        base=base,
        poses=poses,
        grid_n=10,
        margin=0.12,
        fit_kwargs={
            "num_steps": 2,
            "report_every": 1,
            "sample_budget": 256,
            "maintenance_every": 0,
            "superfit": False,
            "local_fit": False,
            "lr_init": 0.003,
            "lr_final": 0.001,
        },
        device=best_device(),
    )

    def on_target(idx, name, vertices, faces, centers, radii, rotations, pose):
        events.append(("target", int(idx)))
        target_names.append(str(name))
        target_vertices.append(np.asarray(vertices, dtype=np.float32).copy())
        if int(idx) == 0:
            exp_c, exp_r, exp_q = mapper.local_to_world_np(base, pose=pose)
            np.testing.assert_allclose(centers, exp_c, rtol=1.0e-5, atol=1.0e-5)
            np.testing.assert_allclose(radii, exp_r, rtol=1.0e-5, atol=1.0e-5)
            np.testing.assert_allclose(
                np.abs(rotations), np.abs(exp_q), rtol=1.0e-5, atol=1.0e-5)
        roundtrip = mapper.world_to_local(
            centers,
            radii,
            rotations,
            base.bone_assignments,
            pose=pose,
            attachment_joints=base.attachment_joints,
            attachment_weights=base.attachment_weights,
        )
        rt_c, rt_r, rt_q = mapper.local_to_world_np(roundtrip, pose=pose)
        np.testing.assert_allclose(rt_c, centers, rtol=1.0e-4, atol=1.0e-4)
        np.testing.assert_allclose(rt_r, radii, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            np.abs(rt_q), np.abs(rotations), rtol=1.0e-4, atol=1.0e-4)
        assert np.asarray(vertices).shape == rig.vertices.shape
        assert np.asarray(faces).shape == rig.faces.shape

    def on_fit(idx, _step, _loss, centers, radii, rotations):
        idx = int(idx)
        events.append(("fit", idx))
        assert ("target", idx) in events, "fit emitted before target visual"
        assert np.asarray(centers).shape == base.local_centers.shape
        assert np.asarray(radii).shape == base.local_radii.shape
        assert np.asarray(rotations).shape == base.local_rotations.shape

    worker.pose_target_visual.connect(on_target)
    worker.pose_fit_progress.connect(on_fit)
    worker.run()

    assert worker.result is not None
    assert len(worker.result.keys) == len(poses)
    assert worker.result.base.num_ellipsoids == base.num_ellipsoids
    assert len(target_vertices) == len(poses)
    assert _max_vertex_delta(target_vertices) > 1.0e-4
    payload = worker.result.to_json(rig.skeleton)
    assert payload["version"] == 2
    assert len(payload["base"]) == base.num_ellipsoids
    assert all(entry.get("attachment_bone_indices") for entry in payload["base"])
    assert all(entry.get("attachment_weights") for entry in payload["base"])

    for idx, key in enumerate(worker.result.keys):
        assert key.delta_centers.shape == base.local_centers.shape
        assert key.delta_rotations.shape == base.local_rotations.shape
        assert key.delta_log_radii.shape == base.local_radii.shape
        assert np.isfinite(key.delta_centers).all()
        assert np.isfinite(key.delta_rotations).all()
        assert np.isfinite(key.delta_log_radii).all()
        center_limits = 1.75 * np.max(base.local_radii, axis=1)
        assert np.all(np.linalg.norm(key.delta_centers, axis=1) <= center_limits + 2e-5)
        assert np.max(np.abs(key.delta_log_radii)) <= np.log(2.5) + 2e-5
        corrected = worker.result.corrected_bone_local(key)
        wc, wr, wq = mapper.local_to_world_np(corrected, pose=poses[idx])
        assert wc.shape == base.local_centers.shape
        assert wr.shape == base.local_radii.shape
        assert wq.shape == base.local_rotations.shape
        assert np.isfinite(wc).all() and np.isfinite(wr).all() and np.isfinite(wq).all()

    direct0 = worker.result.corrected_bone_local(worker.result.keys[0])
    blend0 = worker.result.corrected_blend(0.0)
    np.testing.assert_allclose(blend0.local_centers, direct0.local_centers)
    np.testing.assert_allclose(blend0.local_radii, direct0.local_radii)
    np.testing.assert_allclose(np.abs(blend0.local_rotations), np.abs(direct0.local_rotations))

    mid_pose = interpolate_pose(
        rig.skeleton,
        poses[0],
        poses[1],
        0.5,
        name="debug blend 0.5",
    )
    mid_vertices = deform_mesh(
        rig.vertices,
        rig.skin_joints,
        rig.skin_weights,
        rig.skeleton.compute_skin_matrices(mid_pose),
        device=best_device(),
    )
    assert mid_vertices.shape == rig.vertices.shape
    assert np.isfinite(mid_vertices).all()
    assert _max_vertex_delta([target_vertices[0], mid_vertices]) > 1.0e-5

    mid_corrected = worker.result.corrected_blend(0.5)
    assert mid_corrected.local_centers.shape == base.local_centers.shape
    assert mid_corrected.local_radii.shape == base.local_radii.shape
    assert mid_corrected.local_rotations.shape == base.local_rotations.shape
    assert mid_corrected.attachment_joints is not None
    assert mid_corrected.attachment_weights is not None
    np.testing.assert_allclose(mid_corrected.attachment_joints, base.attachment_joints)
    np.testing.assert_allclose(mid_corrected.attachment_weights, base.attachment_weights)
    mc, mr, mq = mapper.local_to_world_np(mid_corrected, pose=mid_pose)
    assert mc.shape == base.local_centers.shape
    assert mr.shape == base.local_radii.shape
    assert mq.shape == base.local_rotations.shape
    assert np.isfinite(mc).all() and np.isfinite(mr).all() and np.isfinite(mq).all()

    print(
        f"{mesh_name} / {clip_name}: {len(poses)} frames, "
        f"{base.num_ellipsoids} fixed IDs, targets={target_names}"
    )


def main() -> None:
    run_case("T-Pose.fbx", "Dance Loop", pose_count=3)
    run_case("T-Pose.fbx", "Fight Loop", pose_count=3)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
