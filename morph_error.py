"""Live total fit error of posed primitives against the skinned mesh.

The error is the mean absolute signed distance of the primitive union at
posed mesh-surface vertices, in percent of the body height (0 = every sampled
vertex lies exactly on the union surface).  For live use, each vertex is
evaluated only against the primitives nearest to it in the base pose.  The
union minimum almost always comes from one of them, so a frame costs a few
milliseconds instead of a full primitives × vertices evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bone_ellipsoid_mapper import BoneLocalEllipsoids
from rig_world_transform import RigWorldTransform
from skeleton import Pose
from skinning import deform_mesh
from superquadric_geometry import EPS_MAX, EPS_MIN, signed_distance_batch


def _rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-15)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def pair_signed_distance(points_local: np.ndarray, radii: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Row-wise :func:`superquadric_geometry.signed_distance_local` without bending."""
    p = np.asarray(points_local, dtype=np.float64)
    r = np.maximum(np.abs(np.asarray(radii, dtype=np.float64)), 1e-12)
    e = np.clip(np.asarray(eps, dtype=np.float64), EPS_MIN, EPS_MAX)
    e1, e2 = e[:, 0], e[:, 1]
    safe = np.maximum(np.abs(p), r * 1e-7)
    log_n = np.log(safe) - np.log(r)
    log_x = (2 / e2) * log_n[:, 0]
    log_y = (2 / e2) * log_n[:, 1]
    log_a = np.logaddexp(log_x, log_y)
    log_xy = (e2 / e1) * log_a
    log_z = (2 / e1) * log_n[:, 2]
    log_f = np.logaddexp(log_xy, log_z)
    log_beta = .5 * e1 * log_f
    w_xy, w_z = np.exp(log_xy - log_f), np.exp(log_z - log_f)
    gradient = np.column_stack([
        w_xy * np.exp(log_x - log_a) * p[:, 0] / safe[:, 0] ** 2,
        w_xy * np.exp(log_y - log_a) * p[:, 1] / safe[:, 1] ** 2,
        w_z * p[:, 2] / safe[:, 2] ** 2])
    r_min = r.min(axis=1)
    metric = 1e-8 / r_min
    grad_sq = np.einsum("ij,ij->i", gradient, gradient)
    outside = log_beta >= 0
    beta = np.exp(np.clip(log_beta, -80, 80))
    with np.errstate(over="ignore", invalid="ignore"):
        distance_out = (1 - np.exp(np.maximum(-log_beta, -80))) / np.sqrt(grad_sq + metric ** 2)
        gate = np.maximum((1e-4 - beta) / 1e-4, 0) / r_min
        distance_in = (beta - 1) / np.sqrt(beta ** 2 * grad_sq + gate ** 2 + metric ** 2)
    distance = np.where(outside, distance_out, distance_in)
    return np.nan_to_num(distance, nan=0.0, posinf=1e30, neginf=-1e30)


@dataclass(frozen=True)
class SurfaceError:
    mean_percent: float        # mean |SDF| at the sampled vertices, % of height
    p95_percent: float         # 95th percentile, % of height
    mean_absolute: float       # mean |SDF| in model units


class SurfaceErrorMeter:
    """Measures how well posed bone-local primitives cover the skinned mesh."""

    def __init__(self, rigged, base: BoneLocalEllipsoids, base_pose: Pose | None = None,
                 *, sample_count: int = 2500, candidates: int = 10, seed: int = 1):
        self.rigged, self.base = rigged, base
        self.transform = RigWorldTransform(rigged.skeleton, base)
        vertices = np.asarray(rigged.vertices, dtype=np.float64)
        count = len(vertices)
        self._sample = (np.arange(count) if count <= sample_count
                        else np.sort(np.random.default_rng(seed).choice(count, sample_count, replace=False)))
        self._vertices = vertices[self._sample]
        self._joints = np.asarray(rigged.skin_joints)[self._sample]
        self._weights = np.asarray(rigged.skin_weights)[self._sample]
        self.height = float(max(np.ptp(vertices[:, 1]), 1e-9))
        pose = base_pose if base_pose is not None else Pose.t_pose()
        posed = self.posed_vertices(pose)
        c, r, q = self.transform.apply(base, self.transform.pose_map(pose))
        exponents = self._exponents(base)
        full = signed_distance_batch(c, r, q, exponents, posed)                    # (N, S)
        k = min(int(candidates), base.num_ellipsoids)
        self._candidates = np.argpartition(np.abs(full), k - 1, axis=0)[:k].T      # (S, k)

    @staticmethod
    def _exponents(local: BoneLocalEllipsoids) -> np.ndarray:
        if local.shape_exponents is None:
            return np.ones((local.num_ellipsoids, 2), dtype=np.float32)
        return np.asarray(local.shape_exponents)

    def posed_vertices(self, pose: Pose | None) -> np.ndarray:
        return deform_mesh(self._vertices, self._joints, self._weights,
                           self.rigged.skeleton.compute_skin_matrices(pose))

    def matches(self, rigged, base) -> bool:
        return rigged is self.rigged and base is self.base

    def measure(self, pose: Pose | None, local: BoneLocalEllipsoids,
                vertices: np.ndarray | None = None, pose_map=None) -> SurfaceError:
        """Error of *local* (same layout as the base) in *pose*."""
        return self.measure_many(pose, {"_": local}, vertices, pose_map)["_"]

    def measure_many(self, pose: Pose | None, locals_by_name: dict,
                     vertices: np.ndarray | None = None, pose_map=None) -> dict:
        """Errors of several primitive sets sharing the base layout, in one pose."""
        points = self.posed_vertices(pose) if vertices is None else vertices
        pose_map = self.transform.pose_map(pose) if pose_map is None else pose_map
        k = self._candidates.shape[1]
        index = self._candidates.reshape(-1)
        world = np.repeat(np.asarray(points, dtype=np.float64), k, axis=0)
        results = {}
        for name, local in locals_by_name.items():
            c, r, q = self.transform.apply(local, pose_map)
            rotations = _rotation_matrices(q)
            local_points = np.einsum("mi,mij->mj", world - np.asarray(c, dtype=np.float64)[index],
                                     rotations[index])
            distance = pair_signed_distance(local_points, np.asarray(r)[index], self._exponents(local)[index])
            union = np.abs(distance.reshape(self._candidates.shape).min(axis=1))
            results[name] = SurfaceError(mean_percent=float(union.mean()) / self.height * 100,
                                         p95_percent=float(np.percentile(union, 95)) / self.height * 100,
                                         mean_absolute=float(union.mean()))
        return results
