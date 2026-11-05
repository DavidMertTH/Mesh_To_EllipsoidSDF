"""Joint-driven primitive functions, a compact alternative to morph-key blending.

Each primitive gets one small polynomial of the rotations of the joints that
move it: its assigned bone, that bone's parent and children, and its skin
attachment joints.  Per joint, the bind-relative rotation quaternion gives 13
features:

    x, y, z, w-1, x², y², z², xy, xz, yz, x³, y³, z³

All of them are zero at the bind pose, so the base fit is reproduced exactly
there.  The coefficients are fitted by ridge regression on the trained morph
keys.  The regularization is chosen per output by the closed-form
leave-one-out error.  The stored model is a fixed coefficient matrix per
primitive, independent of the number of fitted poses.
"""

from __future__ import annotations

import time

import numpy as np

from bone_ellipsoid_mapper import BoneLocalEllipsoids
from native_morph import (_multiply_quats, _pose_rotations, _rotation_exp,
                          _rotation_logs, _unit_quaternions)
from pose_correctives import PoseCorrectiveLibrary
from skeleton import Pose, Skeleton

FEATURES_PER_JOINT = 13
OUTPUTS = 11            # centre/size (3), log radii (3), rotation log (3), log exponents (2)
_LAMBDAS = np.logspace(-6, 1, 15)
_MIN_SAMPLES = 5


def joint_features(logs: np.ndarray) -> np.ndarray:
    """Quaternion polynomial features ``(..., B, 13)`` of rotation logs ``(..., B, 3)``."""
    q = _rotation_exp(np.asarray(logs, dtype=np.float64))
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([x, y, z, w - 1, x * x, y * y, z * z, x * y, x * z, y * z,
                     x ** 3, y ** 3, z ** 3], axis=-1)


def _ridge_loo(features: np.ndarray, targets: np.ndarray):
    """Ridge weights (unscaled feature space) and LOO error per output."""
    count, width = features.shape
    scale = float(np.sqrt(np.mean(features ** 2) + 1e-12))
    scaled = features / scale
    # One eigendecomposition serves every regularization strength.
    eigenvalues, vectors = np.linalg.eigh(scaled.T @ scaled)
    projected = scaled @ vectors
    rhs = vectors.T @ (scaled.T @ targets)
    best = np.full(targets.shape[1], np.inf)
    weights = np.zeros((width, targets.shape[1]))
    for lam in _LAMBDAS:
        inverse = 1.0 / (np.maximum(eigenvalues, 0) + lam * count)
        candidate = vectors @ (inverse[:, None] * rhs)
        leverage = (projected ** 2) @ inverse
        residual = (targets - scaled @ candidate) / np.maximum(1 - leverage, 1e-6)[:, None]
        error = np.mean(residual ** 2, axis=0)
        better = error < best
        best[better] = error[better]
        weights[:, better] = candidate[:, better]
    return weights / scale, best


class JointFunctionModel:
    """Per-primitive quaternion polynomials fitted to a morph-key library."""

    def __init__(self, skeleton: Skeleton, library: PoseCorrectiveLibrary):
        started = time.perf_counter()
        self.skeleton, self.library = skeleton, library
        base = library.base
        count, bones = base.num_ellipsoids, skeleton.num_bones
        keys = [key for key in library.keys if key.pose is not None]
        if len(keys) < _MIN_SAMPLES:
            raise ValueError(f"Joint functions need at least {_MIN_SAMPLES} fitted pose keys")
        self._base = base
        self._size = np.exp(np.mean(np.log(np.asarray(base.local_radii, dtype=np.float64)), axis=1))
        self._base_q = _unit_quaternions(base.local_rotations)
        self._base_log_radii = np.log(np.asarray(base.local_radii, dtype=np.float64))
        exponents = (np.ones((count, 2)) if base.shape_exponents is None
                     else np.asarray(base.shape_exponents, dtype=np.float64))
        self._base_log_exponents = np.log(np.clip(exponents, .1, 2))
        self._bind_inverse = _unit_quaternions(_pose_rotations(skeleton, library.base_pose)) * [-1, -1, -1, 1]

        logs = np.array([self.rotation_logs(key.pose) for key in keys])            # (K, B, 3)
        varying = np.ptp(logs, axis=0).max(axis=1) > 1e-6                            # (B,)
        targets = self._key_deltas(keys)                                             # (K, N, 11)
        features = joint_features(logs)                                              # (K, B, 13)
        assignments = np.asarray(base.bone_assignments, dtype=int)
        parents = np.array([bone.parent_index for bone in skeleton.bones])

        self.contexts, self._outputs, self._weights = [], [], []
        self._low, self._high = np.zeros((count, OUTPUTS)), np.zeros((count, OUTPUTS))
        explained, total = 0.0, 0.0
        for i in range(count):
            joints = self._context(i, assignments, parents, varying, bones)
            active = np.flatnonzero(np.abs(targets[:, i]).max(axis=0) > 1e-9)
            self.contexts.append(joints)
            self._outputs.append(active)
            if not joints or not len(active):
                self._weights.append(np.zeros((len(joints) * FEATURES_PER_JOINT, len(active))))
                continue
            y = targets[:, i, active]
            weights, loo = _ridge_loo(features[:, joints].reshape(len(keys), -1), y)
            self._weights.append(weights)
            span = y.max(axis=0) - y.min(axis=0)
            self._low[i, active] = np.minimum(y.min(axis=0) - .1 * span, 0)
            self._high[i, active] = np.maximum(y.max(axis=0) + .1 * span, 0)
            variance = y.var(axis=0)
            explained += float(np.sum(variance - np.minimum(loo, variance)))
            total += float(np.sum(variance))
        self.sample_count = len(keys)
        self.coefficient_count = int(sum(w.size for w in self._weights))
        self.explained_variance = explained / total if total > 0 else 0.0
        self.fit_seconds = time.perf_counter() - started

    # ── data ──
    def rotation_logs(self, pose: Pose | None) -> np.ndarray:
        """Bind-relative joint rotation vectors ``(B, 3)``."""
        current = _pose_rotations(self.skeleton, pose)
        return _rotation_logs(_multiply_quats(self._bind_inverse, current))

    def _key_deltas(self, keys) -> np.ndarray:
        count = self._base.num_ellipsoids
        rows = []
        for key in keys:
            exponents = (np.zeros((count, 2)) if key.delta_log_shape_exponents is None
                         else np.asarray(key.delta_log_shape_exponents, dtype=np.float64))
            rows.append(np.concatenate([
                np.asarray(key.delta_centers, dtype=np.float64) / self._size[:, None],
                np.asarray(key.delta_log_radii, dtype=np.float64),
                _rotation_logs(_unit_quaternions(key.delta_rotations)),
                exponents], axis=1))
        return np.asarray(rows)

    def _context(self, i, assignments, parents, varying, bones):
        bone = int(assignments[i])
        joints = {bone}
        if parents[bone] >= 0:
            joints.add(int(parents[bone]))
        joints.update(int(child) for child in np.flatnonzero(parents == bone))
        base = self._base
        if base.attachment_joints is not None and base.attachment_weights is not None:
            joints.update(int(j) for j, w in zip(base.attachment_joints[i], base.attachment_weights[i])
                          if w > 1e-6 and 0 <= j < bones)
        return sorted(j for j in joints if varying[j])

    # ── runtime ──
    def deltas(self, pose: Pose | None) -> np.ndarray:
        """Predicted per-primitive deltas ``(N, 11)`` for *pose*."""
        features = joint_features(self.rotation_logs(pose))
        out = np.zeros((self._base.num_ellipsoids, OUTPUTS))
        for i, (joints, active, weights) in enumerate(zip(self.contexts, self._outputs, self._weights)):
            if joints and len(active):
                out[i, active] = features[joints].reshape(-1) @ weights
        return np.clip(out, self._low, self._high)

    def evaluate(self, pose: Pose | None) -> BoneLocalEllipsoids:
        """Bone-local primitives for *pose*, as the morph driver returns them."""
        delta = self.deltas(pose)
        base = self._base
        rotations = _multiply_quats(self._base_q, _rotation_exp(delta[:, 6:9]))
        exponents = np.exp(np.clip(self._base_log_exponents + delta[:, 9:11], np.log(.1), np.log(2)))
        return BoneLocalEllipsoids(
            local_centers=np.asarray(base.local_centers + delta[:, :3] * self._size[:, None], dtype=np.float32),
            local_radii=np.asarray(np.exp(self._base_log_radii + delta[:, 3:6]), dtype=np.float32),
            local_rotations=np.asarray(rotations, dtype=np.float32),
            bone_assignments=np.asarray(base.bone_assignments).copy(),
            attachment_joints=None if base.attachment_joints is None else base.attachment_joints.copy(),
            attachment_weights=None if base.attachment_weights is None else base.attachment_weights.copy(),
            primitive_type=base.primitive_type,
            shape_exponents=np.asarray(exponents, dtype=np.float32))

    def summary(self) -> str:
        return (f"{len(self.contexts)} joint functions · {self.coefficient_count:,} coefficients · "
                f"{self.explained_variance:.0%} of held-out variation explained")
