"""Pose-driven primitive morphing, matching Unity's default Morph Driver.

Rotation-only descriptors, parent/child context, neutral base support, top-K
inverse-distance RBF weights and quaternion-log delta blending. Primitive IDs,
bone assignments and smooth skin attachments stay fixed. No timeline index is
used, so manual poses and unrelated animation clips drive the same library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bone_ellipsoid_mapper import BoneLocalEllipsoids
from joint_dof import joint_rotation_frames
from pose_correctives import PoseCorrectiveLibrary
from skeleton import Pose, Skeleton, quat_from_matrix


@dataclass(frozen=True)
class MorphSettings:
    falloff: float = 8.0
    context_depth: int = 1
    context_falloff: float = 0.35
    top_k: int = 6
    base_support: float = 1.0
    base_tie_epsilon: float = 0.001
    exact_match_epsilon: float = 0.000001
    smoothing_time: float = 0.12

    def __post_init__(self):
        values = (self.falloff, self.context_falloff, self.base_support,
                  self.base_tie_epsilon, self.exact_match_epsilon, self.smoothing_time)
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Morph settings must be finite and nonnegative")
        if (not isinstance(self.context_depth, (int, np.integer))
                or not isinstance(self.top_k, (int, np.integer))
                or not 0 <= self.context_depth <= 8
                or not 0 <= self.context_falloff <= 1 or self.top_k < 0):
            raise ValueError("Invalid morph context or support count")


def _unit_quaternions(values):
    """Normalize with scaling, so finite unnormalized inputs cannot overflow."""
    q = np.asarray(values, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Morph rotations must be finite quaternions")
    scale = np.max(np.abs(q), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Morph rotations must be nonzero quaternions")
    q = q / scale
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _finite_array(values, shape, label):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid morph {label}: expected finite values of shape {shape}")
    return array.copy()


def _pose_rotations(skeleton, pose):
    matrices = np.asarray([pose.bone_locals.get(i, bone.local_bind_transform)
        if pose is not None else bone.local_bind_transform for i, bone in enumerate(skeleton.bones)])
    matrices = _finite_array(matrices, (skeleton.num_bones, 4, 4), "pose transforms")
    frames = joint_rotation_frames(matrices)
    rotations = np.array([quat_from_matrix(frame) for frame in frames], dtype=np.float64).reshape(-1, 4)
    return _unit_quaternions(rotations)


def rotation_distances(current, targets):
    """Shortest quaternion angle / pi; antipodes and exact matches are stable."""
    current = _unit_quaternions(current)
    targets = _unit_quaternions(targets)
    sign = np.where(np.sum(current * targets, axis=-1, keepdims=True) < 0, -1.0, 1.0)
    aligned = targets * sign
    return (4.0 / np.pi) * np.arctan2(
        np.linalg.norm(current - aligned, axis=-1),
        np.linalg.norm(current + aligned, axis=-1))


def _rotation_logs(quaternions):
    q = _unit_quaternions(quaternions)
    # Unity canonicalizes the 180-degree hemisphere by XYZ as well. Choosing
    # only by W would give opposite logarithms for q and -q at an exact half turn.
    epsilon = 1e-7
    near_half_turn = np.abs(q[..., 3]) <= epsilon
    xyz_negative = ((q[..., 0] < 0)
                    | ((np.abs(q[..., 0]) <= epsilon) & (q[..., 1] < 0))
                    | ((np.abs(q[..., 0]) <= epsilon)
                       & (np.abs(q[..., 1]) <= epsilon) & (q[..., 2] < 0)))
    negate = np.where(near_half_turn, xyz_negative, q[..., 3] < 0)
    q = np.where(negate[..., None], -q, q)
    sine = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    half_angle = np.arctan2(sine, np.clip(q[..., 3:4], -1, 1))
    return q[..., :3] * np.divide(2 * half_angle, sine,
        out=np.full_like(sine, 2), where=sine > 1e-12)


def _rotation_exp(vectors):
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    factor = np.divide(np.sin(angle * .5), angle,
        out=np.full_like(angle, .5), where=angle > 1e-12)
    return np.concatenate((vectors * factor, np.cos(angle * .5)), axis=-1)


def _multiply_quats(a, b):
    xyz = a[:, 3:4] * b[:, :3] + b[:, 3:4] * a[:, :3] + np.cross(a[:, :3], b[:, :3])
    w = a[:, 3:4] * b[:, 3:4] - np.sum(a[:, :3] * b[:, :3], axis=1, keepdims=True)
    q = np.column_stack((xyz, w))
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def _weight_contexts(skeleton, depth, falloff):
    count = skeleton.num_bones
    adjacent = [set() for _ in range(count)]
    for child, bone in enumerate(skeleton.bones):
        parent = bone.parent_index
        if 0 <= parent < count and parent != child:
            adjacent[child].add(parent)
            adjacent[parent].add(child)
    matrix = np.zeros((count, count))
    for bone in range(count):
        seen, frontier = {bone}, {bone}
        for level in range(depth + 1):
            matrix[bone, list(frontier)] = falloff ** level
            next_ = set().union(*(adjacent[i] for i in frontier)) if frontier else set()
            frontier = next_ - seen
            seen.update(frontier)
        matrix[bone] /= np.sum(matrix[bone])
    return matrix


class NativeMorphDriver:
    def __init__(self, skeleton: Skeleton, library: PoseCorrectiveLibrary, settings=None):
        self.skeleton, self.library = skeleton, library
        self.settings = settings or MorphSettings()
        count, bones = library.base.num_ellipsoids, skeleton.num_bones
        if count < 1 or bones < 1:
            raise ValueError("Morphing needs a base population and a skeleton")
        base = library.base
        self._primitive_type = base.primitive_type
        self._base_centers = _finite_array(base.local_centers, (count, 3), "base centers")
        self._base_radii = _finite_array(base.local_radii, (count, 3), "base radii")
        self._base_rotations_local = _unit_quaternions(
            _finite_array(base.local_rotations, (count, 4), "base rotations"))
        self._base_exponents = _finite_array(base.shape_exponents, (count, 2), "base exponents")
        if np.any(self._base_exponents < .1) or np.any(self._base_exponents > 2):
            raise ValueError("Morph base exponents must be inside [0.1, 2]")
        self._assignments = np.asarray(base.bone_assignments)
        if (self._assignments.shape != (count,)
                or self._assignments.dtype.kind not in "iu"
                or np.any(self._assignments < 0) or np.any(self._assignments >= bones)):
            raise ValueError("Morph primitive refers to a missing bone")
        self._assignments = self._assignments.astype(np.int32, copy=True)
        self._attachment_joints = None if base.attachment_joints is None else base.attachment_joints.copy()
        self._attachment_weights = None if base.attachment_weights is None else base.attachment_weights.copy()
        if self._attachment_joints is not None:
            joints, attachment_weights = self._attachment_joints, self._attachment_weights
            if (attachment_weights is None or joints.ndim != 2 or joints.shape[0] != count
                    or joints.shape != attachment_weights.shape
                    or not np.isfinite(attachment_weights).all() or np.any(attachment_weights < 0)
                    or np.any((attachment_weights > 0) & ((joints < 0) | (joints >= bones)))
                    or np.any(np.sum(attachment_weights, axis=1) <= 0)):
                raise ValueError("Invalid morph skin attachments")
        if np.any(self._base_radii <= 0):
            raise ValueError("Morph base radii must be positive")
        self._base_log_radii = np.log(self._base_radii)
        self._check_float32(self._base_centers, "base centers")
        self._check_radii(self._base_log_radii)
        self._indices = [i for i, key in enumerate(library.keys) if key.pose is not None]
        self.has_descriptors = bool(self._indices)
        self._targets = np.array([_pose_rotations(skeleton, library.keys[i].pose)
                                 for i in self._indices]).reshape(len(self._indices), bones, 4)
        self._base_rotations = _pose_rotations(skeleton, library.base_pose)
        self._contexts = _weight_contexts(skeleton, self.settings.context_depth, self.settings.context_falloff)
        self._centers = np.array([_finite_array(library.keys[i].delta_centers, (count, 3), "center deltas")
            for i in self._indices]).reshape(-1, count, 3)
        self._log_radii = np.array([_finite_array(library.keys[i].delta_log_radii, (count, 3), "radius deltas")
            for i in self._indices]).reshape(-1, count, 3)
        self._log_rotations = _rotation_logs(np.array([_finite_array(library.keys[i].delta_rotations,
            (count, 4), "rotation deltas") for i in self._indices]).reshape(-1, count, 4))
        self._log_exponents = np.array([_finite_array(library.keys[i].delta_log_shape_exponents,
            (count, 2), "exponent deltas")
            if library.keys[i].delta_log_shape_exponents is not None else np.zeros((count, 2))
            for i in self._indices], dtype=np.float64).reshape(-1, count, 2)
        if not all(np.isfinite(a).all() for a in (self._centers, self._log_radii, self._log_exponents)):
            raise ValueError("Morph deltas must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            self._check_float32(self._base_centers + self._centers, "target centers")
            self._check_radii(self._base_log_radii + self._log_radii)
        self.weights = None
        self.settled = True

    @staticmethod
    def _check_float32(values, label):
        if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError(f"Morph {label} exceed the finite float32 range")

    @staticmethod
    def _check_radii(log_radii):
        if (not np.isfinite(log_radii).all()
                or np.any(log_radii > np.log(float(np.finfo(np.float32).max)))
                or np.any(log_radii < np.log(float(np.finfo(np.float32).tiny)))):
            raise ValueError("Morph radii exceed the positive float32 range")

    def reset(self):
        """Make the next evaluation immediate, for loading or resetting a fit."""
        self.weights = None
        self.settled = True

    def _kernel(self, distance):
        return np.exp(-max(.0001, self.settings.falloff) * distance) / np.maximum(distance * distance, 1e-8)

    def desired_weights(self, pose: Pose | None):
        count, bones = len(self._indices), self.skeleton.num_bones
        if not count:
            return np.zeros((0, bones))
        current = _pose_rotations(self.skeleton, pose)
        distances = rotation_distances(current[None], self._targets) @ self._contexts.T
        base_distances = rotation_distances(current, self._base_rotations) @ self._contexts.T
        # Unity pins a unique exact whole-pose match, including primitives on
        # joints which did not move. The base support wins coincident ties.
        global_distances = np.mean(distances, axis=1)
        nearest = int(np.argmin(global_distances))
        epsilon = self.settings.exact_match_epsilon
        second = min(float(np.mean(base_distances)),
                     float(np.min(np.delete(global_distances, nearest))) if count > 1 else float("inf"))
        if epsilon > 0 and global_distances[nearest] <= epsilon and second > global_distances[nearest] + epsilon:
            result = np.zeros((count, bones))
            result[nearest] = 1
            return result
        raw = self._kernel(distances)
        tie = self.settings.base_tie_epsilon
        if self.settings.base_support > 0 and tie > 0:
            raw[(base_distances[None] <= tie) & (distances + tie >= base_distances[None])] = 0
        top_k = self.settings.top_k
        if 0 < top_k < count:
            # Stable ties select earlier target IDs, like Unity's top-K cutoff.
            order = np.argsort(-raw, axis=0, kind="stable")
            raw[order[top_k:], np.arange(bones)[None]] = 0
        with np.errstate(over="ignore"):
            base_raw = self.settings.base_support * self._kernel(base_distances)
            denominator = np.sum(raw, axis=0) + base_raw
        return np.divide(raw, denominator[None], out=np.zeros_like(raw), where=denominator[None] > 1e-6)

    def evaluate(self, pose: Pose | None, dt: float | None = None) -> BoneLocalEllipsoids:
        if dt is not None and (not np.isfinite(dt) or dt < 0):
            raise ValueError("Morph elapsed time must be finite and nonnegative")
        desired = self.desired_weights(pose)
        if self.weights is None or dt is None or self.settings.smoothing_time <= 0:
            next_weights = desired
        else:
            with np.errstate(over="ignore"):
                alpha = -np.expm1(-np.log(10) * float(dt) / self.settings.smoothing_time)
            next_weights = self.weights + alpha * (desired - self.weights)
        settled = bool(np.max(np.abs(desired - next_weights), initial=0) < 1e-5)
        if settled:
            next_weights = desired
        weights = next_weights[:, self._assignments]
        centers = self._base_centers + np.einsum("kn,kni->ni", weights, self._centers)
        log_radii = self._base_log_radii + np.einsum("kn,kni->ni", weights, self._log_radii)
        self._check_float32(centers, "centers")
        self._check_radii(log_radii)
        radii = np.exp(log_radii)
        self._check_float32(radii, "radii")
        delta_rotations = _rotation_exp(np.einsum("kn,kni->ni", weights, self._log_rotations))
        rotations = _multiply_quats(self._base_rotations_local, delta_rotations)
        # Clamp in log space before exponentiation: even hostile finite exponent
        # deltas cannot overflow while keeping Unity's final [0.1, 2] bounds.
        exponents = np.exp(np.clip(np.log(self._base_exponents)
            + np.einsum("kn,kni->ni", weights, self._log_exponents), np.log(.1), np.log(2)))
        result = BoneLocalEllipsoids(
            local_centers=np.asarray(centers, dtype=np.float32),
            local_radii=np.asarray(radii, dtype=np.float32),
            local_rotations=np.asarray(rotations, dtype=np.float32),
            bone_assignments=self._assignments.copy(),
            attachment_joints=None if self._attachment_joints is None else self._attachment_joints.copy(),
            attachment_weights=None if self._attachment_weights is None else self._attachment_weights.copy(),
            primitive_type=self._primitive_type,
            shape_exponents=np.asarray(exponents, dtype=np.float32))
        self.weights, self.settled = next_weights, settled
        return result
