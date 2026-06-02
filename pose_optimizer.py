"""
pose_optimizer.py — Multi-pose ellipsoid training with bone-local parameters.

Trains a **single** set of bone-local ellipsoid parameters that generalise
across all given poses.  The training loop:

  1. Pre-compute deformed meshes + SDF grids for every pose (once).
  2. For each training step:
     a. Pick a pose (round-robin or random).
     b. Transform bone-local params → world space (differentiable via Warp).
     c. Compute ellipsoid SDF at mini-batch sample points.
     d. Compare with the pre-computed mesh SDF for that pose.
     e. Backprop gradients → bone-local parameters.
     f. Adam update on bone-local params.

The result is one set of bone-local params that works well across all poses.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import warp as wp
    import warp.optim
except ImportError:
    raise ImportError("Warp is required for pose_optimizer.")

from PySide6 import QtCore

from skeleton import Skeleton, Pose, mat4_decompose
from skinning import deform_mesh
from sdf_compute import SdfComputer, SdfResult
from ellipsoid import EllipsoidSet, best_device
from bone_ellipsoid_mapper import (
    BoneEllipsoidMapper, BoneLocalEllipsoids,
)
from optimization import (
    _ellipsoid_sdf_kernel_batch,
    _rmse_loss_kernel_batch,
    _normalize_flat_quats,
    EpochSampler,
    OptimizationWorker,
)
from bone_ellipsoid_mapper import _local_to_world_kernel


device = "cuda"


# ── Pre-computed pose data ───────────────────────────────────────────────────

class PoseSdfData:
    """SDF grid + bone transforms + deformed mesh for one pose."""

    def __init__(
        self,
        pose: Pose,
        sdf_grid: np.ndarray,    # (nz, ny, nx) float32
        origin: np.ndarray,      # (3,) float32
        dx: float,
        n: int,
        bone_positions: np.ndarray,  # (B, 3) float32
        bone_rotations: np.ndarray,  # (B, 4) float32 — quat xyzw
        deformed_verts: np.ndarray | None = None,  # (V, 3) float32
    ):
        self.pose = pose
        self.sdf_grid = sdf_grid
        self.origin = origin
        self.dx = dx
        self.n = n
        # Per-axis voxel counts (grid may be anisotropic; each pose's deformed
        # mesh has its own AABB → its own shape).  ``grid.shape == (nz, ny, nx)``.
        self.nz, self.ny, self.nx = (int(s) for s in sdf_grid.shape)
        self.bone_positions = bone_positions
        self.bone_rotations = bone_rotations
        self.deformed_verts = deformed_verts

        # Upload SDF to GPU
        self.wp_sdf_target = wp.array(
            sdf_grid.flatten(), dtype=wp.float32, device=device,
            requires_grad=False,
        )
        # Upload bone data
        self.wp_bone_pos = wp.array(
            bone_positions, dtype=wp.vec3, device=device,
        )
        self.wp_bone_rot = wp.array(
            bone_rotations, dtype=wp.quat, device=device,
        )


# ── Multi-Pose Worker ────────────────────────────────────────────────────────

class MultiPoseOptimizationWorker(QtCore.QThread):
    """Train bone-local ellipsoid params across multiple poses.

    Signals
    -------
    precompute_progress(int, int)
        (current_pose_index, total_poses) during SDF pre-computation.
    step_visual(int, float, object, object, object, str)
        (step, loss, world_centers, world_radii, world_rots, pose_name)
    step_sdf(int, float, object, bool, object, float, int)
        Same as OptimizationWorker.step_sdf for SDF panel updates.
    pose_loss(int, str, float)
        (step, pose_name, loss) — per-pose loss for monitoring.
    finished()
        Emitted when training completes.
    """

    precompute_progress = QtCore.Signal(int, int)
    precompute_done = QtCore.Signal()  # all pose data ready
    step_visual = QtCore.Signal(int, float, object, object, object, str)
    step_sdf = QtCore.Signal(int, float, object, bool, object, float, int)
    pose_loss = QtCore.Signal(int, str, float)
    pose_switched = QtCore.Signal(int)  # pose_index — emitted when pose changes
    maintenance_done = QtCore.Signal(int, int, int, int)  # step, n_before, n_pruned, n_spawned
    finished = QtCore.Signal()

    def __init__(
        self,
        # Rig data
        rest_vertices: np.ndarray,
        faces: np.ndarray,
        skeleton: Skeleton,
        skin_joints: np.ndarray,
        skin_weights: np.ndarray,
        poses: List[Pose],
        # Initial ellipsoid params (from T-pose optimisation)
        bone_local: BoneLocalEllipsoids,
        mapper: BoneEllipsoidMapper,
        # Training config
        num_steps: int = 3000,
        steps_per_pose: int | None = None,
        report_every: int = 20,
        grid_n: int = 64,
        margin: float = 0.5,
        batch_fraction: float = 0.125,
        lr: float = 0.005,
        maintenance_every: int = 0,
        miss_penalty_weight: float = 3.0,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._rest_verts = rest_vertices
        self._faces = faces
        self._skeleton = skeleton
        self._skin_joints = skin_joints
        self._skin_weights = skin_weights
        self._poses = poses
        self._bone_local = bone_local
        self._mapper = mapper

        self._num_steps = num_steps
        self._steps_per_pose = steps_per_pose
        self._report_every = report_every
        self._grid_n = grid_n
        self._margin = margin
        self._batch_fraction = batch_fraction
        self._lr = lr
        self._maintenance_every = maintenance_every
        self._miss_penalty_weight = miss_penalty_weight

        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    @property
    def pose_data(self) -> list:
        """Pre-computed pose data (available after precompute_done)."""
        return getattr(self, '_pose_data', [])

    # ── Main entry ───────────────────────────────────────────────────

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            print(f"[MultiPoseOptimizer] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.finished.emit()

    def _run_impl(self):
        # ── Phase 1: Pre-compute SDF for each pose ──
        pose_data = self._precompute_all_poses()
        self._pose_data = pose_data
        if self._stop_flag or not pose_data:
            return

        self.precompute_done.emit()

        # ── Phase 2: Training loop ──
        self._train_loop(pose_data)

    # ── Phase 1: SDF Pre-computation ─────────────────────────────────

    def _precompute_all_poses(self) -> List[PoseSdfData]:
        """Deform mesh + compute SDF grid for every pose."""
        sdf_computer = SdfComputer(device=best_device())
        pose_data: List[PoseSdfData] = []
        n_poses = len(self._poses)

        for pi, pose in enumerate(self._poses):
            if self._stop_flag:
                return pose_data

            self.precompute_progress.emit(pi, n_poses)

            # Deform mesh
            skin_mats = self._skeleton.compute_skin_matrices(pose)
            deformed_verts = deform_mesh(
                self._rest_verts, self._skin_joints, self._skin_weights,
                skin_mats, device=None,  # CPU for reliability
            )

            # Compute SDF
            sdf_computer.set_mesh(deformed_verts, self._faces)
            result = sdf_computer.compute_voxel_grid(
                n=self._grid_n, margin=self._margin, compute_thickness=False,
            )

            # Get bone transforms for this pose
            world_transforms = self._skeleton.compute_world_transforms(pose)
            bone_pos = np.zeros((self._skeleton.num_bones, 3), dtype=np.float32)
            bone_rot = np.zeros((self._skeleton.num_bones, 4), dtype=np.float32)
            for bi in range(self._skeleton.num_bones):
                t, q, _ = mat4_decompose(world_transforms[bi])
                bone_pos[bi] = t.astype(np.float32)
                bone_rot[bi] = q.astype(np.float32)

            pose_data.append(PoseSdfData(
                pose=pose,
                sdf_grid=result.grid,
                origin=result.origin,
                dx=result.dx,
                n=result.n,
                bone_positions=bone_pos,
                bone_rotations=bone_rot,
                deformed_verts=deformed_verts.copy(),
            ))

        self.precompute_progress.emit(n_poses, n_poses)
        return pose_data

    # ── Phase 2: Training ────────────────────────────────────────────

    def _train_loop(self, pose_data: List[PoseSdfData]):
        """Adam training on bone-local params, cycling through poses."""
        bl = self._bone_local
        N = bl.num_ellipsoids
        n = self._grid_n
        # Per-pose grids may be anisotropic with differing voxel counts (each
        # pose's deformed mesh has its own AABB).  Size the shared batch from the
        # SMALLEST pose grid so every per-pose sampler has enough flat indices.
        pose_totals = [int(pd.sdf_grid.size) for pd in pose_data]
        total = min(pose_totals) if pose_totals else n * n * n

        # ── Upload bone-local params (trainable) ──
        pred_local_centers = wp.array(
            bl.local_centers.astype(np.float32),
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        pred_local_radii = wp.array(
            bl.local_radii.astype(np.float32),
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        pred_local_rot_flat = wp.array(
            bl.local_rotations.astype(np.float32).flatten(),
            dtype=wp.float32, device=device, requires_grad=True,
        )

        # Bone assignments (constant)
        wp_bone_assign = wp.array(
            bl.bone_assignments.astype(np.int32),
            dtype=wp.int32, device=device,
        )

        # ── Allocate working buffers ──
        bs = max(1024, min(int(total * self._batch_fraction), total))

        world_centers = wp.empty(N, dtype=wp.vec3, device=device,
                                 requires_grad=True)
        world_radii = wp.empty(N, dtype=wp.vec3, device=device,
                               requires_grad=True)
        world_rot_flat = wp.empty(N * 4, dtype=wp.float32, device=device,
                                  requires_grad=True)
        min_d_cache = wp.zeros((bs, N + 1), dtype=wp.float32,
                               device=device, requires_grad=True)
        sdf_pred = wp.empty(bs, dtype=wp.float32, device=device,
                            requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device,
                        requires_grad=True)
        wp_indices = wp.empty(bs, dtype=wp.int32, device=device)

        # ── Adam on local params ──
        params = [pred_local_centers, pred_local_radii, pred_local_rot_flat]
        optimizer = wp.optim.Adam(params, lr=self._lr)
        grads = [p.grad.flatten() for p in params]

        # ── Per-pose sampler ──
        samplers = [EpochSampler(t, bs) for t in pose_totals]

        n_poses = len(pose_data)

        spp = self._steps_per_pose if self._steps_per_pose else 10
        prev_pi = -1

        # Helper to create a temporary OptimizationWorker for maintenance
        def _make_maintenance_worker(sdf_np, origin, dx, grid_n):
            return OptimizationWorker(
                sdf_target_np=sdf_np, origin=origin, dx=dx, n=grid_n,
                num_ellipsoids=N, num_steps=0,
            )

        for step in range(self._num_steps):
            if self._stop_flag:
                break

            # Pick pose: switch every steps_per_pose steps
            pi = (step // spp) % n_poses
            pd = pose_data[pi]

            # ── Mutex: only ONE special event per step ────────────────
            #    Either a pose switch (+ reassignment) OR a maintenance
            #    step can happen, never both in the same iteration.
            wants_pose_switch = (pi != prev_pi)
            wants_maintenance = (
                self._maintenance_every > 0
                and step > 0
                and step % self._maintenance_every == 0
            )

            if wants_pose_switch:
                # ── Pose switch + reassignment ────────────────────────
                if prev_pi >= 0 and pd.deformed_verts is not None:
                    wp.synchronize_device(device)
                    current_bl = BoneLocalEllipsoids(
                        local_centers=pred_local_centers.numpy().copy(),
                        local_radii=pred_local_radii.numpy().copy(),
                        local_rotations=pred_local_rot_flat.numpy()
                            .reshape(-1, 4).copy(),
                        bone_assignments=bl.bone_assignments.copy(),
                    )
                    new_bl, n_reassigned = self._mapper.check_and_reassign(
                        current_bl, pd.pose, pd.deformed_verts,
                        self._skin_joints, self._skin_weights,
                    )
                    if n_reassigned > 0:
                        bl = new_bl
                        pred_local_centers.assign(wp.array(
                            new_bl.local_centers.astype(np.float32),
                            dtype=wp.vec3, device=device,
                        ))
                        pred_local_radii.assign(wp.array(
                            new_bl.local_radii.astype(np.float32),
                            dtype=wp.vec3, device=device,
                        ))
                        pred_local_rot_flat.assign(wp.array(
                            new_bl.local_rotations.astype(np.float32).flatten(),
                            dtype=wp.float32, device=device,
                        ))
                        wp_bone_assign = wp.array(
                            new_bl.bone_assignments.astype(np.int32),
                            dtype=wp.int32, device=device,
                        )
                        params = [pred_local_centers, pred_local_radii,
                                  pred_local_rot_flat]
                        grads = [p.grad.flatten() for p in params]
                        optimizer = wp.optim.Adam(params, lr=self._lr)

                self.pose_switched.emit(pi)
                prev_pi = pi

            elif wants_maintenance:
                # ── Maintenance (prune + spawn) ───────────────────────
                #    Runs in world space for the current pose, then
                #    converts back to bone-local.
                wp.synchronize_device(device)
                lc = pred_local_centers.numpy().copy()
                lr_np = pred_local_radii.numpy().copy()
                lq = pred_local_rot_flat.numpy().reshape(-1, 4).copy()

                current_bl = BoneLocalEllipsoids(
                    local_centers=lc, local_radii=lr_np,
                    local_rotations=lq,
                    bone_assignments=bl.bone_assignments.copy(),
                )
                # Transform to world for maintenance analysis
                wc, wr, wq = self._mapper.local_to_world_np(current_bl, pd.pose)

                worker = _make_maintenance_worker(
                    pd.sdf_grid, pd.origin, pd.dx, n,
                )
                wc, wr, wq, changed, n_pruned, n_spawned = \
                    worker._do_maintenance(wc, wr, wq)
                self.maintenance_done.emit(step, N, n_pruned, n_spawned)

                if changed:
                    # Re-assign to bones and convert back to bone-local
                    new_bl = self._mapper.assign_to_bones(
                        wc, wr, wq,
                        pd.deformed_verts if pd.deformed_verts is not None
                            else self._rest_verts,
                        self._skin_joints, self._skin_weights,
                        pose=pd.pose,
                    )
                    N = new_bl.num_ellipsoids
                    bl = new_bl

                    pred_local_centers = wp.array(
                        new_bl.local_centers.astype(np.float32),
                        dtype=wp.vec3, device=device, requires_grad=True,
                    )
                    pred_local_radii = wp.array(
                        new_bl.local_radii.astype(np.float32),
                        dtype=wp.vec3, device=device, requires_grad=True,
                    )
                    pred_local_rot_flat = wp.array(
                        new_bl.local_rotations.astype(np.float32).flatten(),
                        dtype=wp.float32, device=device, requires_grad=True,
                    )
                    wp_bone_assign = wp.array(
                        new_bl.bone_assignments.astype(np.int32),
                        dtype=wp.int32, device=device,
                    )
                    # Reallocate working buffers for potentially changed N
                    world_centers = wp.empty(N, dtype=wp.vec3, device=device,
                                             requires_grad=True)
                    world_radii = wp.empty(N, dtype=wp.vec3, device=device,
                                           requires_grad=True)
                    world_rot_flat = wp.empty(N * 4, dtype=wp.float32,
                                              device=device, requires_grad=True)
                    min_d_cache = wp.zeros((bs, N + 1), dtype=wp.float32,
                                           device=device, requires_grad=True)
                    sdf_pred = wp.empty(bs, dtype=wp.float32, device=device,
                                        requires_grad=True)
                    loss = wp.zeros(1, dtype=wp.float32, device=device,
                                    requires_grad=True)

                    params = [pred_local_centers, pred_local_radii,
                              pred_local_rot_flat]
                    grads = [p.grad.flatten() for p in params]
                    optimizer = wp.optim.Adam(params, lr=self._lr)

            # ── Forward pass ──
            tape = wp.Tape()
            with tape:
                # 1. Local → World transform
                wp.launch(
                    _local_to_world_kernel,
                    dim=N,
                    inputs=[
                        pred_local_centers, pred_local_radii,
                        pred_local_rot_flat,
                        wp_bone_assign,
                        pd.wp_bone_pos, pd.wp_bone_rot,
                        world_centers, world_radii, world_rot_flat,
                    ],
                    device=device,
                )

                # 2. Sample batch indices
                wp_indices.assign(samplers[pi].next_batch())

                wp_origin = wp.vec3(
                    float(pd.origin[0]),
                    float(pd.origin[1]),
                    float(pd.origin[2]),
                )

                # 3. Ellipsoid SDF at sample points
                min_d_cache.zero_()
                wp.launch(
                    _ellipsoid_sdf_kernel_batch,
                    dim=bs,
                    inputs=[
                        world_centers, world_radii, world_rot_flat,
                        min_d_cache, N, wp_origin,
                        float(pd.dx), pd.nx, pd.ny, pd.nz,
                        wp_indices, sdf_pred,
                    ],
                    device=device,
                )

                # 4. Loss against mesh SDF
                loss.zero_()
                wp.launch(
                    _rmse_loss_kernel_batch,
                    dim=bs,
                    inputs=[sdf_pred, pd.wp_sdf_target, wp_indices,
                            loss, bs, float(self._miss_penalty_weight),
                            float(0.0), float(1.0), float(0.0),
                            pd.wp_sdf_target, float(1.0), float(0.0), float(1.0)],
                    device=device,
                )

            # ── Backward + update ──
            tape.backward(loss)
            optimizer.step(grads)
            tape.zero()

            # Normalise quaternions
            wp.launch(
                _normalize_flat_quats, dim=N,
                inputs=[pred_local_rot_flat],
                device=device,
            )

            # ── Reporting ──
            if step % self._report_every == 0:
                self._emit_progress(
                    step, loss, pd,
                    pred_local_centers, pred_local_radii,
                    pred_local_rot_flat, N,
                )

        # ── Save final bone-local params ──
        wp.synchronize_device(device)
        self._bone_local.local_centers = pred_local_centers.numpy().copy()
        self._bone_local.local_radii = pred_local_radii.numpy().copy()
        self._bone_local.local_rotations = pred_local_rot_flat.numpy()\
            .reshape(-1, 4).copy()

    # ── Progress emission ────────────────────────────────────────────

    def _emit_progress(
        self, step, loss_wp, pd,
        pred_local_centers, pred_local_radii, pred_local_rot_flat, num_e,
    ):
        wp.synchronize_device(device)
        loss_val = float(loss_wp.numpy()[0])

        # Get world-space params for visualisation
        lc = pred_local_centers.numpy().copy()
        lr = pred_local_radii.numpy().copy()
        lq = pred_local_rot_flat.numpy().reshape(-1, 4).copy()

        # Transform to world for the current pose
        wc, wr, wq = self._mapper.local_to_world_np(
            BoneLocalEllipsoids(
                local_centers=lc, local_radii=lr,
                local_rotations=lq,
                bone_assignments=self._bone_local.bone_assignments,
            ),
            pose=pd.pose,
        )

        self.step_visual.emit(step, loss_val, wc, wr, wq, pd.pose.name)
        self.pose_loss.emit(step, pd.pose.name, loss_val)

        # SDF panel update (less frequent)
        if step % (self._report_every * 5) == 0:
            ell_set = EllipsoidSet()
            ell_set.set_parameters(wc, wr, wq)
            self.step_sdf.emit(
                step, loss_val, ell_set, False,
                pd.origin, pd.dx, pd.n,
            )

    # ── Public: get final result ─────────────────────────────────────

    @property
    def result(self) -> BoneLocalEllipsoids:
        """Final trained bone-local parameters."""
        return self._bone_local


# ── Convenience: Full pipeline ───────────────────────────────────────────────

def run_multipose_pipeline(
    rigged_mesh,   # RiggedMesh from rig_loader
    initial_ellipsoid_set: EllipsoidSet,
    poses: List[Pose] | None = None,
    num_steps: int = 3000,
    grid_n: int = 64,
    lr: float = 0.005,
) -> tuple[BoneEllipsoidMapper, BoneLocalEllipsoids]:
    """Non-threaded convenience function for scripting / testing.

    Parameters
    ----------
    rigged_mesh : RiggedMesh
    initial_ellipsoid_set : EllipsoidSet — from T-pose optimisation
    poses : list[Pose], optional — if None, uses rigged_mesh.poses
    num_steps : int
    grid_n : int
    lr : float

    Returns
    -------
    mapper : BoneEllipsoidMapper
    bone_local : BoneLocalEllipsoids
    """
    from skeleton import Pose

    skeleton = rigged_mesh.skeleton
    mapper = BoneEllipsoidMapper(skeleton)

    if poses is None:
        poses = rigged_mesh.poses

    # Step 1: Assign ellipsoids to bones
    bone_local = mapper.assign_to_bones(
        world_centers=initial_ellipsoid_set.centers,
        world_radii=initial_ellipsoid_set.radii,
        world_rotations=initial_ellipsoid_set.rotations,
        mesh_vertices=rigged_mesh.vertices,
        skin_joints=rigged_mesh.skin_joints,
        skin_weights=rigged_mesh.skin_weights,
        pose=Pose.t_pose(),
    )

    print(f"[MultiPose] Assigned {bone_local.num_ellipsoids} ellipsoids to bones:")
    for i in range(bone_local.num_ellipsoids):
        print(f"  Ellipsoid {i} → {mapper.get_bone_name(i)}")

    # Step 2: Pre-compute pose SDFs
    sdf_computer = SdfComputer(device=best_device())
    pose_data_list = []

    for pi, pose in enumerate(poses):
        skin_mats = skeleton.compute_skin_matrices(pose)
        deformed_verts = deform_mesh(
            rigged_mesh.vertices, rigged_mesh.skin_joints,
            rigged_mesh.skin_weights, skin_mats,
        )
        sdf_computer.set_mesh(deformed_verts, rigged_mesh.faces)
        result = sdf_computer.compute_voxel_grid(n=grid_n, margin=0.5, compute_thickness=False)

        world_transforms = skeleton.compute_world_transforms(pose)
        bone_pos = np.zeros((skeleton.num_bones, 3), dtype=np.float32)
        bone_rot = np.zeros((skeleton.num_bones, 4), dtype=np.float32)
        for bi in range(skeleton.num_bones):
            t, q, _ = mat4_decompose(world_transforms[bi])
            bone_pos[bi] = t.astype(np.float32)
            bone_rot[bi] = q.astype(np.float32)

        pose_data_list.append(PoseSdfData(
            pose=pose, sdf_grid=result.grid,
            origin=result.origin, dx=result.dx, n=result.n,
            bone_positions=bone_pos, bone_rotations=bone_rot,
        ))

        print(f"  Pose {pi}/{len(poses)}: {pose.name} — SDF computed")

    # Step 3: Training
    N = bone_local.num_ellipsoids
    n = grid_n
    # Per-pose grids may be anisotropic with differing voxel counts; size the
    # shared batch from the smallest so every per-pose sampler has enough indices.
    pose_totals = [int(pd.sdf_grid.size) for pd in pose_data_list]
    total = min(pose_totals) if pose_totals else n ** 3
    bs = max(1024, int(total * 0.125))

    pred_lc = wp.array(bone_local.local_centers, dtype=wp.vec3,
                       device=device, requires_grad=True)
    pred_lr = wp.array(bone_local.local_radii, dtype=wp.vec3,
                       device=device, requires_grad=True)
    pred_lrot = wp.array(bone_local.local_rotations.flatten(),
                         dtype=wp.float32, device=device, requires_grad=True)
    wp_ba = wp.array(bone_local.bone_assignments, dtype=wp.int32, device=device)

    world_c = wp.empty(N, dtype=wp.vec3, device=device, requires_grad=True)
    world_r = wp.empty(N, dtype=wp.vec3, device=device, requires_grad=True)
    world_rf = wp.empty(N * 4, dtype=wp.float32, device=device, requires_grad=True)
    min_d = wp.zeros((bs, N + 1), dtype=wp.float32, device=device, requires_grad=True)
    sdf_pred = wp.empty(bs, dtype=wp.float32, device=device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp_idx = wp.empty(bs, dtype=wp.int32, device=device)

    optimizer = wp.optim.Adam([pred_lc, pred_lr, pred_lrot], lr=lr)
    grads = [p.grad.flatten() for p in [pred_lc, pred_lr, pred_lrot]]
    samplers = [EpochSampler(t, bs) for t in pose_totals]

    n_poses = len(pose_data_list)
    for step in range(num_steps):
        pi = step % n_poses
        pd = pose_data_list[pi]

        tape = wp.Tape()
        with tape:
            wp.launch(_local_to_world_kernel, dim=N,
                      inputs=[pred_lc, pred_lr, pred_lrot, wp_ba,
                              pd.wp_bone_pos, pd.wp_bone_rot,
                              world_c, world_r, world_rf], device=device)

            wp_idx.assign(samplers[pi].next_batch())
            wp_o = wp.vec3(float(pd.origin[0]), float(pd.origin[1]),
                           float(pd.origin[2]))
            min_d.zero_()
            wp.launch(_ellipsoid_sdf_kernel_batch, dim=bs,
                      inputs=[world_c, world_r, world_rf, min_d, N,
                              wp_o, float(pd.dx), pd.nx, pd.ny, pd.nz, wp_idx, sdf_pred],
                      device=device)
            loss_buf.zero_()
            wp.launch(_rmse_loss_kernel_batch, dim=bs,
                      inputs=[sdf_pred, pd.wp_sdf_target, wp_idx,
                              loss_buf, bs, float(0.0),
                              float(0.0), float(1.0), float(0.0),
                              pd.wp_sdf_target, float(1.0), float(0.0), float(1.0)], device=device)

        tape.backward(loss_buf)
        optimizer.step(grads)
        tape.zero()
        wp.launch(_normalize_flat_quats, dim=N, inputs=[pred_lrot], device=device)

        if step % 100 == 0:
            wp.synchronize_device(device)
            print(f"  Step {step}/{num_steps}: loss={float(loss_buf.numpy()[0]):.6f}"
                  f"  pose={pd.pose.name}")

    wp.synchronize_device(device)
    bone_local.local_centers = pred_lc.numpy().copy()
    bone_local.local_radii = pred_lr.numpy().copy()
    bone_local.local_rotations = pred_lrot.numpy().reshape(-1, 4).copy()

    print(f"[MultiPose] Training complete.")
    return mapper, bone_local
