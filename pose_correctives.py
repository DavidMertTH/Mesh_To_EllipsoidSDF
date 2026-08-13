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
from sdf_blowup import apply_thickness_limited_blowup
from sdf_compute import SdfComputer
from skeleton import Pose, Skeleton, quat_inverse, quat_multiply, quat_slerp
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

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loss": round(float(self.loss), 7),
            "delta_centers": np.asarray(self.delta_centers, dtype=np.float32).tolist(),
            "delta_rotations": np.asarray(self.delta_rotations, dtype=np.float32).tolist(),
            "delta_log_radii": np.asarray(self.delta_log_radii, dtype=np.float32).tolist(),
        }


@dataclass
class PoseCorrectiveLibrary:
    """Base bone-local ellipsoids plus pose-keyed relative deltas."""

    base: BoneLocalEllipsoids
    keys: list[PoseCorrectiveKey] = field(default_factory=list)

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
        return BoneLocalEllipsoids(
            local_centers=centers.astype(np.float32),
            local_radii=radii.astype(np.float32),
            local_rotations=_normalize_quats(rotations),
            bone_assignments=base.bone_assignments.copy(),
            attachment_joints=(None if base.attachment_joints is None
                               else base.attachment_joints.copy()),
            attachment_weights=(None if base.attachment_weights is None
                                else base.attachment_weights.copy()),
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
                **attachment_entry_fields(base, i, skeleton),
            })
        return {
            "format": "ellipsdf-pose-correctives",
            "version": 2,
            "quaternion_convention": "xyzw",
            "count": int(base.num_ellipsoids),
            "base": entries,
            "poses": [k.to_json() for k in self.keys],
        }

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
) -> PoseCorrectiveKey:
    """Compute relative deltas from base local params to optimized local params."""
    if optimized.num_ellipsoids != base.num_ellipsoids:
        raise ValueError("pose corrective must keep the same ellipsoid count")
    delta_centers = (
        optimized.local_centers.astype(np.float32)
        - base.local_centers.astype(np.float32)
    )
    delta_log_radii = np.log(
        np.maximum(optimized.local_radii.astype(np.float32), 1.0e-7)
        / np.maximum(base.local_radii.astype(np.float32), 1.0e-7)
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
    )


def _pose_rotation_distance(skeleton: Skeleton, a: Pose, b: Pose) -> float:
    """RMS angular pose distance, normalized to the [0, 1] half-turn range."""
    _, qa = skeleton.compute_bone_positions_rotations(a)
    _, qb = skeleton.compute_bone_positions_rotations(b)
    qa = _normalize_quats(qa)
    qb = _normalize_quats(qb)
    dots = np.clip(np.abs(np.sum(qa * qb, axis=1)), 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return float(np.sqrt(np.mean(np.square(angles / np.pi))))


def _blend_local_seed(
    base: BoneLocalEllipsoids,
    neighbor: BoneLocalEllipsoids | None,
    weight: float,
) -> BoneLocalEllipsoids:
    if neighbor is None or weight <= 1.0e-6:
        centers = base.local_centers.copy()
        radii = base.local_radii.copy()
        rotations = base.local_rotations.copy()
    else:
        w = float(np.clip(weight, 0.0, 1.0))
        centers = (
            (1.0 - w) * base.local_centers
            + w * neighbor.local_centers
        ).astype(np.float32)
        radii = np.exp(
            (1.0 - w) * np.log(np.maximum(base.local_radii, 1.0e-7))
            + w * np.log(np.maximum(neighbor.local_radii, 1.0e-7))
        ).astype(np.float32)
        rotations = np.array([
            quat_slerp(base.local_rotations[i], neighbor.local_rotations[i], w)
            for i in range(base.num_ellipsoids)
        ], dtype=np.float32)
    return BoneLocalEllipsoids(
        local_centers=centers,
        local_radii=radii,
        local_rotations=_normalize_quats(rotations),
        bone_assignments=base.bone_assignments.copy(),
        attachment_joints=(None if base.attachment_joints is None
                           else base.attachment_joints.copy()),
        attachment_weights=(None if base.attachment_weights is None
                            else base.attachment_weights.copy()),
    )


class PoseCorrectiveWorker(QtCore.QThread):
    """Sequentially fit relative corrective layers for a list of poses."""

    pose_started = QtCore.Signal(int, int, str)
    pose_target_visual = QtCore.Signal(int, str, object, object, object, object, object, object)
    pose_sdf_progress = QtCore.Signal(int, float, str)
    pose_fit_progress = QtCore.Signal(int, int, float, object, object, object)
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
        target_vertices: list[np.ndarray] | None = None,
        sdf_blowup_vox: float = 0.0,
        sdf_blowup_offset: float | None = None,
        thickness_max_resolution: int | None = 128,
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
        self._sdf_blowup_vox = float(sdf_blowup_vox)
        if not np.isfinite(self._sdf_blowup_vox):
            raise ValueError("sdf_blowup_vox must be finite")
        self._sdf_blowup_offset = (
            None if sdf_blowup_offset is None
            else float(sdf_blowup_offset)
        )
        if (self._sdf_blowup_offset is not None
                and not np.isfinite(self._sdf_blowup_offset)):
            raise ValueError("sdf_blowup_offset must be finite")
        self._has_sdf_blowup = (
            self._sdf_blowup_offset != 0.0
            if self._sdf_blowup_offset is not None
            else self._sdf_blowup_vox != 0.0
        )
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
        trained_poses: list[Pose] = []
        trained_locals: list[BoneLocalEllipsoids] = []
        total = len(self._poses)
        for idx, pose in enumerate(self._poses):
            if self._stop:
                raise _PoseCorrectiveCanceled(
                    self._stop_reason or f"canceled before pose {idx}")
            pose_name = pose.name or f"Pose {idx + 1}"
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
            neighbor_local = None
            neighbor_strength = 0.0
            if trained_poses:
                distances = np.asarray([
                    _pose_rotation_distance(skeleton, pose, trained_pose)
                    for trained_pose in trained_poses
                ], dtype=np.float64)
                nearest = int(np.argmin(distances))
                neighbor_local = trained_locals[nearest]
                neighbor_strength = float(np.exp(-np.square(distances[nearest] / 0.25)))
            initial_local = _blend_local_seed(
                self._base, neighbor_local, neighbor_strength)
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
                margin=self._margin,
                compute_thickness=self._has_sdf_blowup,
                compute_blowup_thickness=self._has_sdf_blowup,
                thickness_max_resolution=self._thickness_max_resolution,
                progress_cb=_sdf_progress,
                symmetry=False,
            )
            if self._stop:
                raise _PoseCorrectiveCanceled(
                    self._stop_reason or f"canceled after SDF for pose {idx}")

            last = {
                "loss": float("inf"),
                "centers": start_c,
                "radii": start_r,
                "rotations": start_q,
            }

            kwargs = self._optimizer_kwargs(
                sdf,
                initial_local,
                parameter_linear,
                parameter_offset,
                parameter_rotation_prefix,
                neighbor_local,
                neighbor_strength,
            )
            opt = OptimizationWorker(**kwargs)
            self._active_optimizer = opt

            def _on_step(step, loss, centers, radii, rotations, _extra, pi=idx):
                last["loss"] = float(loss)
                last["centers"] = np.asarray(centers, dtype=np.float32).copy()
                last["radii"] = np.asarray(radii, dtype=np.float32).copy()
                last["rotations"] = np.asarray(rotations, dtype=np.float32).copy()
                self.pose_fit_progress.emit(
                    pi, int(step), float(loss),
                    last["centers"], last["radii"], last["rotations"],
                )

            opt.step_visual.connect(_on_step)
            opt.run()
            self._active_optimizer = None
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
            )
            key = corrective_from_optimized_local(
                self._base, optimized, pose_name, float(last["loss"]))
            keys.append(key)
            trained_poses.append(pose)
            trained_locals.append(optimized)
            self.pose_finished.emit(idx, pose_name, float(last["loss"]))

        if total > 0 and not keys:
            raise RuntimeError(
                f"pose corrective training produced 0 keys before pose 0 "
                f"finished (poses={total}, stop={self._stop}, "
                f"reason={self._stop_reason or 'none'})")
        return PoseCorrectiveLibrary(base=self._base, keys=keys)

    def _optimizer_kwargs(
        self,
        sdf,
        initial_local: BoneLocalEllipsoids,
        parameter_linear: np.ndarray,
        parameter_offset: np.ndarray,
        parameter_rotation_prefix: np.ndarray,
        neighbor_local: BoneLocalEllipsoids | None,
        neighbor_strength: float,
    ) -> dict[str, Any]:
        kw = dict(self._fit_kwargs)
        neighbor_weights = (
            float(kw.pop("parameter_neighbor_center_regularization", 0.004)),
            float(kw.pop("parameter_neighbor_radii_regularization", 0.002)),
            float(kw.pop("parameter_neighbor_rotation_regularization", 0.0015)),
        )
        count = initial_local.num_ellipsoids
        sdf_target = np.asarray(sdf.grid, dtype=np.float32)
        requested_blowup = (
            float(self._sdf_blowup_offset)
            if self._sdf_blowup_offset is not None
            else self._sdf_blowup_vox * float(sdf.dx)
        )
        blowup_thickness = None
        if requested_blowup != 0.0:
            blowup_thickness = getattr(sdf, "blowup_thickness", None)
            if blowup_thickness is None:
                blowup_thickness = sdf.thickness
            sdf_target = apply_thickness_limited_blowup(
                sdf_target,
                requested_blowup,
                blowup_thickness,
                float(sdf.dx),
            )
        kw.update({
            "sdf_target_np": sdf_target,
            "origin": sdf.origin,
            "dx": float(sdf.dx),
            "n": int(sdf.n),
            "sdf_blowup_offset": float(requested_blowup),
            "num_ellipsoids": int(count),
            "max_ellipsoids": int(count),
            "initial_centers": initial_local.local_centers,
            "initial_radii": initial_local.local_radii,
            "initial_rotations": initial_local.local_rotations,
            "parameter_linear_np": parameter_linear,
            "parameter_offset_np": parameter_offset,
            "parameter_rotation_prefix_np": parameter_rotation_prefix,
            "parameter_anchor_centers": self._base.local_centers,
            "parameter_anchor_radii": self._base.local_radii,
            "parameter_anchor_rotations": self._base.local_rotations,
            "parameter_center_regularization": float(
                kw.get("parameter_center_regularization", 0.006)),
            "parameter_radii_regularization": float(
                kw.get("parameter_radii_regularization", 0.003)),
            "parameter_rotation_regularization": float(
                kw.get("parameter_rotation_regularization", 0.002)),
            "parameter_center_trust_radius_factor": float(
                kw.get("parameter_center_trust_radius_factor", 1.75)),
            "parameter_radii_trust_factor": float(
                kw.get("parameter_radii_trust_factor", 2.5)),
            "maintenance_every": 0,
            "superfit": False,
            "local_fit": False,
            "spawn_underrep": False,
            "split_enabled": False,
            "merge_enabled": False,
            "prune_enabled": False,
            "symmetry_enabled": False,
            "primitive_shape": "ellipsoid",
            "sdf_mode": int(kw.get("sdf_mode", SDF_MERTSTEIN)),
        })
        loss_thickness = (
            blowup_thickness
            if requested_blowup != 0.0 and blowup_thickness is not None
            else sdf.thickness
        )
        if loss_thickness is not None:
            kw["thickness_np"] = np.asarray(
                loss_thickness, dtype=np.float32)
        if neighbor_local is not None and neighbor_strength > 1.0e-4:
            kw.update({
                "parameter_neighbor_centers": neighbor_local.local_centers,
                "parameter_neighbor_radii": neighbor_local.local_radii,
                "parameter_neighbor_rotations": neighbor_local.local_rotations,
                "parameter_neighbor_center_regularization": (
                    neighbor_weights[0] * float(neighbor_strength)),
                "parameter_neighbor_radii_regularization": (
                    neighbor_weights[1] * float(neighbor_strength)),
                "parameter_neighbor_rotation_regularization": (
                    neighbor_weights[2] * float(neighbor_strength)),
            })
        kw.pop("bone_aware", None)
        kw.pop("bone_centers_np", None)
        kw.pop("bone_expected_counts_np", None)
        return kw
