"""Editable rotational degrees of freedom and joint display geometry.

Angles describe offsets from the local bind pose, in degrees. Allowances bound
manual pose authoring; importing and displaying animation does not clamp it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointDof:
    enabled: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=bool))
    minimum: np.ndarray = field(default_factory=lambda: np.full(3, -180.0))
    maximum: np.ndarray = field(default_factory=lambda: np.full(3, 180.0))

    @property
    def count(self) -> int:
        return int(np.count_nonzero(
            self.enabled & (self.maximum - self.minimum > 1e-6)))

    def set_axis(self, axis: int, enabled: bool, minimum: float,
                 maximum: float) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("Joint axis must be X, Y or Z")
        if not (np.isfinite(minimum) and np.isfinite(maximum)
                and -180.0 <= minimum <= maximum <= 180.0):
            raise ValueError("Joint angles must satisfy -180 <= min <= max <= 180")
        self.enabled[axis] = enabled
        self.minimum[axis] = minimum
        self.maximum[axis] = maximum


def joint_rotation_frames(world_matrices: np.ndarray) -> np.ndarray:
    """Extract proper orthonormal frames without scale/shear from bone worlds."""
    matrices = np.asarray(world_matrices, dtype=np.float64).reshape(-1, 4, 4)
    if not len(matrices):
        return np.empty((0, 3, 3), dtype=np.float32)
    u, _s, vh = np.linalg.svd(matrices[:, :3, :3])
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
    correction[:, 2, 2] = np.linalg.det(u @ vh)
    return np.asarray(u @ correction @ vh, dtype=np.float32)


def project_joint_points(points: np.ndarray, matrix: np.ndarray,
                         width: int, height: int):
    """Project to logical Qt pixels, returning pixels, clip depth and visibility."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    clip = homogeneous @ np.asarray(matrix, dtype=np.float64).reshape(4, 4).T
    valid = np.all(np.isfinite(clip), axis=1) & (clip[:, 3] > 1e-9)
    ndc = np.full((len(points), 3), np.nan)
    np.divide(clip[:, :3], clip[:, 3:4], out=ndc, where=valid[:, None])
    pixels = np.column_stack(((ndc[:, 0] + 1) * width * 0.5,
                              (1 - ndc[:, 1]) * height * 0.5))
    valid &= np.all(np.abs(ndc) <= 1.0, axis=1)
    return pixels, ndc[:, 2], valid


def joint_rotation_arc(center: np.ndarray, frame: np.ndarray, axis: int,
                       radius: float, minimum: float = -180.0,
                       maximum: float = 180.0) -> np.ndarray:
    """A positive right-hand rotation arc perpendicular to the selected axis."""
    samples = max(2, int(np.ceil((maximum - minimum) / 4.0)) + 1)
    angles = np.radians(np.linspace(minimum, maximum, samples))
    local = np.zeros((samples, 3), dtype=np.float64)
    local[:, (axis + 1) % 3] = np.cos(angles) * radius
    local[:, (axis + 2) % 3] = np.sin(angles) * radius
    return np.asarray(local @ frame.T + center, dtype=np.float32)


def joint_pointer_angle(position: np.ndarray, matrix: np.ndarray,
                        width: int, height: int, center: np.ndarray,
                        frame: np.ndarray, axis: int) -> float | None:
    """Ray/rotation-plane angle, or None for an edge-on/degenerate view.

    Screen coordinates are logical Qt pixels. The basis matches
    ``joint_rotation_arc`` so dragging stays in the displayed local plane.
    """
    if width <= 0 or height <= 0:
        return None
    x, y = position
    ndc = np.array([[2.0 * x / width - 1, 1 - 2.0 * y / height, z, 1]
                    for z in (-1.0, 1.0)])
    try:
        ray = ndc @ np.linalg.inv(matrix).T
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(ray)) or np.any(np.abs(ray[:, 3]) < 1e-12):
        return None
    ray = ray[:, :3] / ray[:, 3:4]
    direction = ray[1] - ray[0]
    normal = frame[:, axis]
    denominator = float(direction @ normal)
    # Ray/plane intersection becomes unstable when looking along the plane.
    if abs(denominator) < 0.08 * np.linalg.norm(direction):
        return None
    distance = float((center - ray[0]) @ normal) / denominator
    if distance < 0:
        return None
    offset = ray[0] + distance * direction - center
    cosine = float(offset @ frame[:, (axis + 1) % 3])
    sine = float(offset @ frame[:, (axis + 2) % 3])
    if np.hypot(cosine, sine) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(sine, cosine)))
