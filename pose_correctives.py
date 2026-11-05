"""
pose_correctives.py - Sequential pose-corrective ellipsoid fitting.

This module keeps a fixed base ellipsoid population and trains one relative
corrective layer per pose:

    corrected_local = base_local * pose_delta
    world           = bone_pose * corrected_local

The ellipsoid count, order, and bone assignment never change in this phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from ellipsoid import SDF_MERTSTEIN, best_device
from optimization import OptimizationWorker
from rig_ingest import attachment_entry_fields
from sdf_blowup import (
    BLOWUP_CARRIER_MARGIN_VOXELS,
    apply_thickness_relative_blowup,
    required_relative_sdf_margin,
)
from sdf_compute import SdfComputer
from skeleton import (
    Pose,
    Skeleton,
    mat4_decompose,
    quat_inverse,
    quat_multiply,
    quat_slerp,
)
from skinning import deform_mesh


class _PoseCorrectiveCanceled(RuntimeError):
    """Internal sentinel used to stop corrective training quietly."""


def _normalize_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    out = q / np.maximum(n, 1.0e-9)
    bad = ~np.isfinite(out).all(axis=1)
    if np.any(bad):
        out[bad] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return out.astype(np.float32)


@dataclass
class PoseCorrectiveKey:
    """One relative ellipsoid corrective layer for one pose."""

    name: str
    delta_centers: np.ndarray
    delta_rotations: np.ndarray
    delta_log_radii: np.ndarray
    loss: float = 0.0
    delta_log_shape_exponents: np.ndarray | None = None
    pose: Pose | None = None

    def to_json(self, skeleton: Skeleton | None = None) -> dict[str, Any]:
        count = len(np.asarray(self.delta_centers).reshape(-1, 3))
        delta_eps = self.delta_log_shape_exponents
        if delta_eps is None:
            delta_eps = np.zeros((count, 2), dtype=np.float32)
        delta_centers = np.asarray(
            self.delta_centers, dtype=np.float32).reshape(count, 3)
        delta_rotations = _normalize_quats(self.delta_rotations)
        delta_log_radii = np.asarray(
            self.delta_log_radii, dtype=np.float32).reshape(count, 3)
        delta_eps = np.asarray(
            delta_eps, dtype=np.float32).reshape(count, 2)
        result = {
            "name": self.name,
            "loss": round(float(self.loss), 7),
            # Keep the original parallel arrays for existing consumers.
            "delta_centers": delta_centers.tolist(),
            "delta_rotations": delta_rotations.tolist(),
            "delta_log_radii": delta_log_radii.tolist(),
            "delta_log_shape_exponents": delta_eps.tolist(),
            # Unity's JsonUtility cannot reliably deserialize jagged arrays.
            # The object representation is additive and carries stable ids.
            "ellipsoids": [
                {
                    "id": int(i),
                    "delta_local_center": delta_centers[i].tolist(),
                    "delta_local_rotation": delta_rotations[i].tolist(),
                    "delta_log_radii": delta_log_radii[i].tolist(),
                    "delta_log_shape_exponents": delta_eps[i].tolist(),
                }
                for i in range(count)
            ],
        }
        if skeleton is not None and self.pose is not None:
            result["pose"] = _pose_descriptor_to_json(
                skeleton, self.pose)
        return result


def _pose_descriptor_to_json(
    skeleton: Skeleton,
    pose: Pose,
) -> dict[str, Any]:
    """Serialize the rotation descriptor consumed by Unity's MorphDriver.

    The runtime driver currently compares bone-local rotations. Positions are
    intentionally omitted: Unity reconstructs them from its captured base pose,
    which also avoids guessing a root bone's parent-space translation from the
    world-space API protocol.
    """
    bones: list[dict[str, Any]] = []
    for bone in skeleton.bones:
        index = int(bone.index)
        local = np.asarray(
            pose.bone_locals.get(index, bone.local_bind_transform),
            dtype=np.float64,
        ).reshape(4, 4)
        _translation, rotation, _scale = mat4_decompose(local)
        rotation = _normalize_quats(
            np.asarray(rotation, dtype=np.float32).reshape(1, 4))[0]
        bones.append({
            "index": index,
            "parent_index": int(bone.parent_index),
            "name": str(bone.name),
            "local_rotation": rotation.tolist(),
        })
    return {
        "descriptor": "bone_local_rotations",
        "bones": bones,
    }


@dataclass
class PoseCorrectiveLibrary:
    """Base bone-local ellipsoids plus pose-keyed relative deltas."""

    base: BoneLocalEllipsoids
    keys: list[PoseCorrectiveKey] = field(default_factory=list)
    base_pose: Pose | None = None

    def key(self, index: int) -> PoseCorrectiveKey | None:
        if 0 <= int(index) < len(self.keys):
            return self.keys[int(index)]
        return None

    def corrected_bone_local(
        self,
        key: PoseCorrectiveKey | None,
        weight: float = 1.0,
    ) -> BoneLocalEllipsoids:
        """Return base ellipsoids with a corrective key blended in."""
        base = self.base
        if key is None or weight == 0.0:
            return BoneLocalEllipsoids(
                local_centers=base.local_centers.copy(),
                local_radii=base.local_radii.copy(),
                local_rotations=base.local_rotations.copy(),
                bone_assignments=base.bone_assignments.copy(),
                attachment_joints=(None if base.attachment_joints is None
                                   else base.attachment_joints.copy()),
                attachment_weights=(None if base.attachment_weights is None
                                    else base.attachment_weights.copy()),
                primitive_type=base.primitive_type,
                shape_exponents=base.shape_exponents.copy(),
            )
        w = float(np.clip(weight, 0.0, 1.0))
        centers = (
            base.local_centers.astype(np.float32)
            + np.asarray(key.delta_centers, dtype=np.float32) * w
        )
        radii = (
            base.local_radii.astype(np.float32)
            * np.exp(np.asarray(key.delta_log_radii, dtype=np.float32) * w)
        )
        # First version: nlerp between identity and the full delta quaternion.
        d = _normalize_quats(key.delta_rotations)
        ident = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (len(d), 1))
        flip = np.sum(ident * d, axis=1) < 0.0
        d[flip] *= -1.0
        blended_delta = _normalize_quats((1.0 - w) * ident + w * d)
        rotations = np.array([
            quat_multiply(base.local_rotations[i], blended_delta[i])
            for i in range(base.num_ellipsoids)
        ], dtype=np.float32)
        delta_eps = key.delta_log_shape_exponents
        if delta_eps is None:
            delta_eps = np.zeros_like(base.shape_exponents, dtype=np.float32)
        else:
            delta_eps = np.asarray(delta_eps, dtype=np.float32).reshape(
                base.num_ellipsoids, 2)
        shape_exponents = np.clip(
            base.shape_exponents.astype(np.float32) * np.exp(delta_eps * w),
            0.1,
            2.0,
        ).astype(np.float32)
        return BoneLocalEllipsoids(
            local_centers=centers.astype(np.float32),
            local_radii=radii.astype(np.float32),
            local_rotations=_normalize_quats(rotations),
            bone_assignments=base.bone_assignments.copy(),
            attachment_joints=(None if base.attachment_joints is None
                               else base.attachment_joints.copy()),
            attachment_weights=(None if base.attachment_weights is None
                                else base.attachment_weights.copy()),
            primitive_type=base.primitive_type,
            shape_exponents=shape_exponents,
        )

    def corrected_blend(self, frame: float) -> BoneLocalEllipsoids:
        """Return bone-local ellipsoids for a fractional corrective frame."""
        if not self.keys:
            return self.corrected_bone_local(None)
        if len(self.keys) == 1:
            return self.corrected_bone_local(self.keys[0])

        f = float(np.clip(frame, 0.0, float(len(self.keys) - 1)))
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, len(self.keys) - 1)
        w = f - float(i0)
        if i0 == i1 or w <= 1.0e-6:
            return self.corrected_bone_local(self.keys[i0])
        if w >= 1.0 - 1.0e-6:
            return self.corrected_bone_local(self.keys[i1])

        k0 = self.keys[i0]
        k1 = self.keys[i1]
        d0 = _normalize_quats(k0.delta_rotations)
        d1 = _normalize_quats(k1.delta_rotations)
        blended_delta = np.array([
            quat_slerp(d0[i], d1[i], w)
            for i in range(self.base.num_ellipsoids)
        ], dtype=np.float32)
        key = PoseCorrectiveKey(
            name=f"{k0.name} -> {k1.name} {w:.2f}",
            delta_centers=(
                (1.0 - w) * np.asarray(k0.delta_centers, dtype=np.float32)
                + w * np.asarray(k1.delta_centers, dtype=np.float32)
            ),
            delta_rotations=blended_delta,
            delta_log_radii=(
                (1.0 - w) * np.asarray(k0.delta_log_radii, dtype=np.float32)
                + w * np.asarray(k1.delta_log_radii, dtype=np.float32)
            ),
            loss=(1.0 - w) * float(k0.loss) + w * float(k1.loss),
            delta_log_shape_exponents=(
                (1.0 - w) * np.asarray(
                    k0.delta_log_shape_exponents
                    if k0.delta_log_shape_exponents is not None
                    else np.zeros_like(self.base.shape_exponents),
                    dtype=np.float32,
                )
                + w * np.asarray(
                    k1.delta_log_shape_exponents
                    if k1.delta_log_shape_exponents is not None
                    else np.zeros_like(self.base.shape_exponents),
                    dtype=np.float32,
                )
            ),
        )
        return self.corrected_bone_local(key)

    def to_json(self, skeleton: Skeleton) -> dict[str, Any]:
        base = self.base
        entries = []
        for i in range(base.num_ellipsoids):
            bi = int(base.bone_assignments[i])
            entries.append({
                "id": i,
                "bone": skeleton.bones[bi].name if 0 <= bi < skeleton.num_bones else "",
                "local_center": [round(float(v), 7) for v in base.local_centers[i]],
                "local_rotation": [round(float(v), 7) for v in base.local_rotations[i]],
                "radii": [round(float(v), 7) for v in base.local_radii[i]],
                "primitive_type": base.primitive_type,
                "shape_exponents": [
                    round(float(v), 7) for v in base.shape_exponents[i]
                ],
                **attachment_entry_fields(base, i, skeleton),
            })
        result = {
            "format": "ellipsdf-pose-correctives",
            "version": 4,
            "primitive_type": base.primitive_type,
            "quaternion_convention": "xyzw",
            "count": int(base.num_ellipsoids),
            "base": entries,
            "poses": [k.to_json(skeleton) for k in self.keys],
        }
        if self.base_pose is not None:
            result["base_pose"] = _pose_descriptor_to_json(
                skeleton, self.base_pose)
        return result

    def save_json(self, skeleton: Skeleton, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_json(skeleton), f, indent=2)
        return out


def corrective_from_optimized_local(
    base: BoneLocalEllipsoids,
    optimized: BoneLocalEllipsoids,
    name: str,
    loss: float,
    pose: Pose | None = None,
) -> PoseCorrectiveKey:
    """Compute relative deltas from base local params to optimized local params."""
    if optimized.num_ellipsoids != base.num_ellipsoids:
        raise ValueError("pose corrective must keep the same ellipsoid count")
    if optimized.primitive_type != base.primitive_type:
        raise ValueError(
            "pose corrective must keep the same primitive_type "
            f"({base.primitive_type!r} != {optimized.primitive_type!r})")
    delta_centers = (
        optimized.local_centers.astype(np.float32)
        - base.local_centers.astype(np.float32)
    )
    delta_log_radii = np.log(
        np.maximum(optimized.local_radii.astype(np.float32), 1.0e-7)
        / np.maximum(base.local_radii.astype(np.float32), 1.0e-7)
    ).astype(np.float32)
    delta_log_shape_exponents = np.log(
        np.maximum(optimized.shape_exponents.astype(np.float32), 0.1)
        / np.maximum(base.shape_exponents.astype(np.float32), 0.1)
    ).astype(np.float32)
    delta_rot = np.array([
        quat_multiply(quat_inverse(base.local_rotations[i]), optimized.local_rotations[i])
        for i in range(base.num_ellipsoids)
    ], dtype=np.float32)
    return PoseCorrectiveKey(
        name=str(name or "Pose"),
        delta_centers=delta_centers,
        delta_rotations=_normalize_quats(delta_rot),
        delta_log_radii=delta_log_radii,
        loss=float(loss),
        delta_log_shape_exponents=delta_log_shape_exponents,
        pose=pose,
    )


def _most_similar_fitted(
    skeleton: Skeleton,
    pose: Pose,
    fitted_poses: list[Pose | None],
    assignments: np.ndarray,
    context_depth: int = 1,
    context_falloff: float = 0.35,
) -> np.ndarray:
    """Per primitive, the index of the most similar already fitted pose.

    Similarity is the runtime morph driver's: joint-local rotations of the
    primitive's bone, blended with its parent/child context. Exact ties prefer
    candidate 0, the base fit, which the driver also shows where its pose ties
    with the base; other ties choose the overall most similar pose, then the
    earliest one.
    """
    # native_morph imports this module; share its pose similarity lazily.
    from native_morph import _pose_rotations, _weight_contexts, rotation_distances
    contexts = _weight_contexts(skeleton, context_depth, context_falloff)
    current = _pose_rotations(skeleton, pose)
    fitted = np.array([_pose_rotations(skeleton, p) for p in fitted_poses])
    distances = rotation_distances(current[None], fitted) @ contexts.T
    overall = np.mean(distances, axis=1)
    bones = np.asarray(assignments, dtype=np.int64)
    valid = (bones >= 0) & (bones < skeleton.num_bones)
    local = np.where(valid[None], distances[:, np.where(valid, bones, 0)], overall[:, None])
    tied = local <= np.min(local, axis=0, keepdims=True) + 1.0e-6
    closest = np.argmin(np.where(tied, overall[:, None], np.inf), axis=0)
    return np.where(tied[0], 0, closest)


def _gather_local_seed(
    base: BoneLocalEllipsoids,
    fitted: list[BoneLocalEllipsoids],
    sources: np.ndarray,
) -> BoneLocalEllipsoids:
    """Take each primitive's parameters from its chosen fitted pose."""
    if any(local.primitive_type != base.primitive_type for local in fitted):
        raise ValueError("cannot seed correctives from different primitive types")
    rows = np.arange(base.num_ellipsoids)

    def pick(attribute):
        return np.stack([np.asarray(getattr(local, attribute), dtype=np.float32)
                         for local in fitted])[sources, rows]

    return BoneLocalEllipsoids(
        local_centers=pick("local_centers"),
        local_radii=pick("local_radii"),
        local_rotations=_normalize_quats(pick("local_rotations")),
        bone_assignments=base.bone_assignments.copy(),
        attachment_joints=(None if base.attachment_joints is None
                           else base.attachment_joints.copy()),
        attachment_weights=(None if base.attachment_weights is None
                            else base.attachment_weights.copy()),
        primitive_type=base.primitive_type,
        shape_exponents=pick("shape_exponents"),
    )


class PoseCorrectiveWorker(QtCore.QThread):
    """Sequentially fit relative corrective layers for a list of poses."""

    pose_started = QtCore.Signal(int, int, str)
    pose_target_visual = QtCore.Signal(
        int, str, object, object, object, object, object, object, object)
    pose_sdf_progress = QtCore.Signal(int, float, str)
    pose_fit_progress = QtCore.Signal(
        int, int, float, object, object, object, object)
    pose_finished = QtCore.Signal(int, str, float)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        rigged_mesh,
        mapper: BoneEllipsoidMapper,
        base: BoneLocalEllipsoids,
        poses: list[Pose],
        grid_n: int,
        margin: float,
        fit_kwargs: dict[str, Any],
        base_pose: Pose | None = None,
        base_fit_pose: Pose | None = None,
        reuse_base_pose: bool = False,
        target_vertices: list[np.ndarray] | None = None,
        sdf_blowup_fraction: float = 0.0,
        thickness_max_resolution: int | None = 128,
        context_depth: int = 1,
        context_falloff: float = 0.35,
        device: str | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rigged_mesh = rigged_mesh
        self._mapper = mapper
        self._base = base
        self._poses = list(poses or [])
        self._grid_n = int(grid_n)
        self._margin = float(margin)
        self._fit_kwargs = dict(fit_kwargs or {})
        self._base_pose = base_pose
        # The pose the base parameters were fitted in, if it differs from the
        # runtime's neutral base_pose (e.g. a base fitted in an edited pose).
        self._base_fit_pose = base_fit_pose if base_fit_pose is not None else base_pose
        self._reuse_base_pose = bool(reuse_base_pose)
        # Pose similarity for seeds uses the morph driver's joint context.
        self._context_depth = int(context_depth)
        self._context_falloff = float(context_falloff)
        self._sdf_blowup_fraction = float(sdf_blowup_fraction)
        if (not np.isfinite(self._sdf_blowup_fraction)
                or not -0.5 < self._sdf_blowup_fraction < 0.5):
            raise ValueError(
                "sdf_blowup_fraction magnitude must be smaller than 0.5")
        self._has_sdf_blowup = self._sdf_blowup_fraction != 0.0
        self._thickness_max_resolution = thickness_max_resolution
        self._target_vertices = (
            [np.asarray(v, dtype=np.float32).copy() for v in target_vertices]
            if target_vertices is not None else None
        )
        self._device = device or best_device()
        self._stop = False
        self._stop_reason: str | None = None
        self._active_optimizer: OptimizationWorker | None = None
        self.result: PoseCorrectiveLibrary | None = None

    def request_stop(self, reason: str | None = None) -> None:
        self._stop = True
        if self._stop_reason is None:
            self._stop_reason = reason or "request_stop called"
        if self._active_optimizer is not None:
            self._active_optimizer.request_stop()

    def run(self) -> None:
        try:
            self.result = self._run_all()
        except _PoseCorrectiveCanceled as e:
            self.result = None
            self.failed.emit(str(e) or "pose corrective training canceled")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))
        finally:
            self._active_optimizer = None
            self.finished.emit()

    def _run_all(self) -> PoseCorrectiveLibrary:
        rm = self._rigged_mesh
        skeleton = rm.skeleton
        keys: list[PoseCorrectiveKey] = []
        # Every fit starts from the most similar configuration fitted so far,
        # chosen per primitive (see _optimizer_kwargs for its anchors). The
        # base fit is the first such configuration.
        fitted_poses: list[Pose | None] = [self._base_fit_pose]
        fitted_locals: list[BoneLocalEllipsoids] = [self._base]
        total = len(self._poses)
        for idx, pose in enumerate(self._poses):
            if self._stop:
                raise _PoseCorrectiveCanceled(
                    self._stop_reason or f"canceled before pose {idx}")
            pose_name = pose.name or f"Pose {idx + 1}"
            if (idx == 0 and self._reuse_base_pose
                    and self._target_vertices is None
                    and all(np.allclose(
                        pose.bone_locals.get(bi, bone.local_bind_transform),
                        (self._base_pose.bone_locals.get(bi, bone.local_bind_transform)
                         if self._base_pose is not None else bone.local_bind_transform),
                        rtol=0.0, atol=1.0e-9)
                        for bi, bone in enumerate(skeleton.bones))):
                # The native sweep starts with the already fitted bind pose.
                # Preserve its neutral key and timeline index, then move straight
                # to the first deformed target instead of fitting it all again.
                neutral = corrective_from_optimized_local(
                    self._base, self._base, pose_name, 0.0, pose=pose)
                # q^-1 * q can round to ~1e-18; the neutral key is exact.
                neutral.delta_rotations[:] = np.array([0, 0, 0, 1], dtype=np.float32)
                keys.append(neutral)
                self.pose_finished.emit(idx, pose_name, 0.0)
                continue
            self.pose_started.emit(idx, total, pose_name)

            if (self._target_vertices is not None
                    and idx < len(self._target_vertices)):
                deformed = np.asarray(self._target_vertices[idx],
                                      dtype=np.float32)
            else:
                skin_mats = skeleton.compute_skin_matrices(pose)
                deformed = deform_mesh(
                    rm.vertices, rm.skin_joints, rm.skin_weights, skin_mats,
                    device=self._device,
                )
            sources = _most_similar_fitted(
                skeleton, pose, fitted_poses, self._base.bone_assignments,
                self._context_depth, self._context_falloff)
            initial_local = _gather_local_seed(self._base, fitted_locals, sources)
            self._apply_refit_locks(initial_local)
            parameter_linear, parameter_offset, parameter_rotation_prefix = (
                self._mapper.local_to_world_parameter_transform(self._base, pose=pose)
            )
            start_c, start_r, start_q = self._mapper.local_to_world_np(
                initial_local, pose=pose)
            self.pose_target_visual.emit(
                idx,
                pose_name,
                np.asarray(deformed, dtype=np.float32).copy(),
                np.asarray(rm.faces, dtype=np.int32).copy(),
                np.asarray(start_c, dtype=np.float32).copy(),
                np.asarray(start_r, dtype=np.float32).copy(),
                _normalize_quats(np.asarray(start_q, dtype=np.float32)).copy(),
                initial_local.shape_exponents.copy(),
                pose,
            )

            comp = SdfComputer(device=self._device)
            comp.set_mesh(deformed, rm.faces)
            def _sdf_progress(f, m, pi=idx):
                if self._stop:
                    raise _PoseCorrectiveCanceled(
                        self._stop_reason or f"canceled during SDF for pose {pi}")
                self.pose_sdf_progress.emit(pi, float(f), str(m))
                if self._stop:
                    raise _PoseCorrectiveCanceled(
                        self._stop_reason or f"canceled during SDF for pose {pi}")

            sdf = comp.compute_voxel_grid(
                n=self._grid_n,
                margin=required_relative_sdf_margin(
                    self._margin,
                    self._sdf_blowup_fraction,
                    self._grid_n,
                ),
                compute_thickness=self._has_sdf_blowup,
                compute_blowup_thickness=self._has_sdf_blowup,
                thickness_max_resolution=self._thickness_max_resolution,
                progress_cb=_sdf_progress,
                symmetry=False,
                blowup_thickness_fraction=self._sdf_blowup_fraction,
                guard_voxels_per_side=(
                    int(BLOWUP_CARRIER_MARGIN_VOXELS)
                    if self._has_sdf_blowup else 0),
            )
            if self._stop:
                raise _PoseCorrectiveCanceled(
                    self._stop_reason or f"canceled after SDF for pose {idx}")

            last = {
                "loss": float("inf"),
                "centers": start_c,
                "radii": start_r,
                "rotations": start_q,
                "shape_exponents": initial_local.shape_exponents.copy(),
            }

            kwargs = self._optimizer_kwargs(
                sdf,
                initial_local,
                parameter_linear,
                parameter_offset,
                parameter_rotation_prefix,
            )
            opt = OptimizationWorker(**kwargs)
            self._active_optimizer = opt
            step_error: list[RuntimeError] = []

            def _on_step(step, loss, centers, radii, rotations, _extra, pi=idx):
                last["loss"] = float(loss)
                last["centers"] = np.asarray(centers, dtype=np.float32).copy()
                last["radii"] = np.asarray(radii, dtype=np.float32).copy()
                last["rotations"] = np.asarray(rotations, dtype=np.float32).copy()
                if initial_local.primitive_type == "superquadric":
                    if _extra is None:
                        step_error.append(RuntimeError(
                            "superquadric corrective fit did not emit shape exponents"))
                        opt.request_stop()
                        return
                    step_eps = np.asarray(_extra, dtype=np.float32)
                    if step_eps.ndim != 2 or step_eps.shape[0] != len(centers) \
                            or step_eps.shape[1] < 2:
                        step_error.append(RuntimeError(
                            "superquadric corrective fit emitted invalid shape exponents"))
                        opt.request_stop()
                        return
                    step_eps = step_eps[:, :2]
                    if (not np.isfinite(step_eps).all()
                            or np.any(step_eps < 0.1) or np.any(step_eps > 2.0)):
                        step_error.append(RuntimeError(
                            "superquadric corrective fit emitted out-of-range shape exponents"))
                        opt.request_stop()
                        return
                    last["shape_exponents"] = step_eps.copy()
                else:
                    last["shape_exponents"] = np.ones(
                        (len(centers), 2), dtype=np.float32)
                self.pose_fit_progress.emit(
                    pi, int(step), float(loss),
                    last["centers"], last["radii"], last["rotations"],
                    last["shape_exponents"],
                )

            # This optimizer runs synchronously inside our worker thread, which
            # has no Qt event loop. Consume its snapshots here immediately;
            # pose_fit_progress then queues the copied frame to the GUI thread.
            opt.step_visual.connect(_on_step, QtCore.Qt.DirectConnection)
            opt.run()
            self._active_optimizer = None
            if step_error:
                raise step_error[0]
            if self._stop:
                raise _PoseCorrectiveCanceled(
                    self._stop_reason or f"canceled during fit for pose {idx}")

            if opt.optimized_parameter_result is None:
                raise RuntimeError(
                    f"pose {pose_name!r} did not produce bone-local parameters")
            local_centers, local_radii, local_rotations = opt.optimized_parameter_result
            optimized = BoneLocalEllipsoids(
                local_centers=np.asarray(local_centers, dtype=np.float32),
                local_radii=np.asarray(local_radii, dtype=np.float32),
                local_rotations=_normalize_quats(local_rotations),
                bone_assignments=self._base.bone_assignments.copy(),
                attachment_joints=(None if self._base.attachment_joints is None
                                   else self._base.attachment_joints.copy()),
                attachment_weights=(None if self._base.attachment_weights is None
                                    else self._base.attachment_weights.copy()),
                primitive_type=self._base.primitive_type,
                shape_exponents=np.asarray(
                    last["shape_exponents"], dtype=np.float32).copy(),
            )
            self._apply_refit_locks(optimized)
            key = corrective_from_optimized_local(
                self._base,
                optimized,
                pose_name,
                float(last["loss"]),
                pose=pose,
            )
            if not self._fit_kwargs.get("optimize_rotations", True):
                key.delta_rotations[:] = np.array([0, 0, 0, 1], dtype=np.float32)
            keys.append(key)
            fitted_poses.append(pose)
            fitted_locals.append(optimized)
            self.pose_finished.emit(idx, pose_name, float(last["loss"]))

        if total > 0 and not keys:
            raise RuntimeError(
                f"pose corrective training produced 0 keys before pose 0 "
                f"finished (poses={total}, stop={self._stop}, "
                f"reason={self._stop_reason or 'none'})")
        return PoseCorrectiveLibrary(
            base=self._base,
            keys=keys,
            base_pose=self._base_pose,
        )

    def _apply_refit_locks(self, local: BoneLocalEllipsoids) -> None:
        """Locks apply to the base, including neighbour seeds and round trips."""
        for flag, attribute in (
                ("optimize_centers", "local_centers"),
                ("optimize_rotations", "local_rotations"),
                ("optimize_radii", "local_radii")):
            if not self._fit_kwargs.get(flag, True):
                setattr(local, attribute, getattr(self._base, attribute).copy())
        if (not self._fit_kwargs.get("optimize_shape_exponents", True)
                or self._fit_kwargs.get("sq_eps_mode") == "fixed"):
            local.shape_exponents = self._base.shape_exponents.copy()

    def _optimizer_kwargs(
        self,
        sdf,
        initial_local: BoneLocalEllipsoids,
        parameter_linear: np.ndarray,
        parameter_offset: np.ndarray,
        parameter_rotation_prefix: np.ndarray,
    ) -> dict[str, Any]:
        """Optimizer settings anchored at the seed and at the base.

        ``initial_local`` holds each primitive's most similar fitted state. The
        fit starts there, may move one trust-region step from it, and is pulled
        toward it. An equally strong pull toward the base, and the base's own
        trust region as an outer bound, keep per-fit noise from adding up
        along chains of similar poses, so neighbouring keys stay consistent.
        """
        kw = dict(self._fit_kwargs)
        primitive_type = str(
            getattr(initial_local, "primitive_type", "ellipsoid")
            or "ellipsoid").strip().lower()
        if primitive_type not in ("ellipsoid", "superquadric"):
            raise ValueError(
                "pose correctives support only ellipsoid and superquadric")
        requested_type = str(
            kw.get("primitive_shape", primitive_type)
            or primitive_type).strip().lower()
        if requested_type != primitive_type:
            raise ValueError(
                "pose-corrective optimizer primitive_shape does not match "
                f"the base ({requested_type!r} != {primitive_type!r})")
        count = initial_local.num_ellipsoids
        weights = (
            float(kw.get("parameter_center_regularization", 0.006)),
            float(kw.get("parameter_radii_regularization", 0.003)),
            float(kw.get("parameter_rotation_regularization", 0.002)),
        )
        center_trust = float(kw.get("parameter_center_trust_radius_factor", 1.75))
        radii_trust = float(kw.get("parameter_radii_trust_factor", 2.5))
        sdf_target = np.asarray(sdf.grid, dtype=np.float32)
        blowup_thickness = None
        if self._sdf_blowup_fraction != 0.0:
            blowup_thickness = getattr(sdf, "blowup_thickness", None)
            if blowup_thickness is None:
                blowup_thickness = sdf.thickness
            sdf_target = apply_thickness_relative_blowup(
                sdf_target,
                self._sdf_blowup_fraction,
                blowup_thickness,
            )
        kw.update({
            "sdf_target_np": sdf_target,
            "origin": sdf.origin,
            "dx": float(sdf.dx),
            "n": int(sdf.n),
            "sdf_blowup_fraction": float(self._sdf_blowup_fraction),
            "num_ellipsoids": int(count),
            "max_ellipsoids": int(count),
            "initial_centers": initial_local.local_centers,
            "initial_radii": initial_local.local_radii,
            "initial_rotations": initial_local.local_rotations,
            "parameter_linear_np": parameter_linear,
            "parameter_offset_np": parameter_offset,
            "parameter_rotation_prefix_np": parameter_rotation_prefix,
            "parameter_anchor_centers": initial_local.local_centers,
            "parameter_anchor_radii": initial_local.local_radii,
            "parameter_anchor_rotations": initial_local.local_rotations,
            "parameter_center_regularization": weights[0],
            "parameter_radii_regularization": weights[1],
            "parameter_rotation_regularization": weights[2],
            "parameter_center_trust_radius_factor": center_trust,
            "parameter_radii_trust_factor": radii_trust,
            # The optimizer's second L2 anchor ("neighbor") holds the base.
            "parameter_neighbor_centers": self._base.local_centers,
            "parameter_neighbor_radii": self._base.local_radii,
            "parameter_neighbor_rotations": self._base.local_rotations,
            "parameter_neighbor_center_regularization": float(
                kw.get("parameter_neighbor_center_regularization", weights[0])),
            "parameter_neighbor_radii_regularization": float(
                kw.get("parameter_neighbor_radii_regularization", weights[1])),
            "parameter_neighbor_rotation_regularization": float(
                kw.get("parameter_neighbor_rotation_regularization", weights[2])),
            "parameter_bound_centers": self._base.local_centers,
            "parameter_bound_radii": self._base.local_radii,
            "parameter_bound_center_radius_factor": float(
                kw.get("parameter_bound_center_radius_factor", center_trust)),
            "parameter_bound_radii_factor": float(
                kw.get("parameter_bound_radii_factor", radii_trust)),
            "maintenance_every": 0,
            "superfit": False,
            "local_fit": False,
            "spawn_underrep": False,
            "split_enabled": False,
            "merge_enabled": False,
            "prune_enabled": False,
            "symmetry_enabled": False,
            "primitive_shape": primitive_type,
            "sdf_mode": int(kw.get("sdf_mode", SDF_MERTSTEIN)),
        })
        if primitive_type == "superquadric":
            initial_eps = np.asarray(
                getattr(initial_local, "shape_exponents", None),
                dtype=np.float32,
            ).reshape(count, 2)
            if (not np.isfinite(initial_eps).all()
                    or np.any(initial_eps < 0.1) or np.any(initial_eps > 2.0)):
                raise ValueError("invalid superquadric shape_exponents in pose seed")
            kw["initial_eps"] = initial_eps.copy()
        else:
            kw.pop("initial_eps", None)
        loss_thickness = (
            blowup_thickness
            if self._sdf_blowup_fraction != 0.0 and blowup_thickness is not None
            else sdf.thickness
        )
        if loss_thickness is not None:
            kw["thickness_np"] = np.asarray(
                loss_thickness, dtype=np.float32)
        kw.pop("bone_aware", None)
        kw.pop("bone_centers_np", None)
        kw.pop("bone_expected_counts_np", None)
        return kw
