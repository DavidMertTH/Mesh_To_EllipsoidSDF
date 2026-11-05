"""Read-only captured morph samples, using Unity's signed rotation vectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from native_morph import _multiply_quats, _pose_rotations, _rotation_logs, _unit_quaternions


def relative_rotation_degrees(basis, current):
    basis, current = _unit_quaternions(basis), _unit_quaternions(current)
    inverse = basis.copy()
    inverse[:3] *= -1
    relative = _multiply_quats(inverse[None], current[None])
    return np.degrees(_rotation_logs(relative)[0])


@dataclass
class MorphCurveSample:
    name: str
    key_index: int
    joint_rotation: np.ndarray
    position: np.ndarray
    radii: np.ndarray
    rotation_delta: np.ndarray
    exponents: np.ndarray


@dataclass
class MorphCurveSeries:
    samples: list[MorphCurveSample] = field(default_factory=list)
    missing_pose: int = 0
    missing_parameters: int = 0
    invalid_values: int = 0

    @property
    def skipped(self):
        return self.missing_pose + self.missing_parameters + self.invalid_values

    def resolved_axis(self, requested="auto"):
        if requested != "auto":
            return requested
        if not self.samples:
            return "x"
        vectors = np.array([sample.joint_rotation for sample in self.samples])
        return "xyz"[int(np.argmax(np.ptp(vectors, axis=0)))]

    def angles(self, requested="auto"):
        axis = self.resolved_axis(requested)
        vectors = np.array([sample.joint_rotation for sample in self.samples]).reshape(-1, 3)
        return np.linalg.norm(vectors, axis=1) if axis == "total" else vectors[:, "xyz".index(axis)]

    def values(self, parameter):
        if not self.samples:
            return np.empty((0, 2 if parameter == "exponents" else 3))
        if parameter == "scale":
            return np.array([s.radii / self.samples[0].radii for s in self.samples])
        if parameter == "position_delta":
            return np.array([s.position - self.samples[0].position for s in self.samples])
        attribute = {"exponents": "exponents", "radii": "radii",
                     "position": "position", "rotation": "rotation_delta"}[parameter]
        return np.array([getattr(s, attribute) for s in self.samples])


def _row(values, index, width):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not 0 <= index < len(array):
        raise IndexError("Missing primitive parameters")
    result = array[index].copy()
    if not np.isfinite(result).all():
        raise ValueError("Non-finite primitive parameters")
    return result


def build_morph_curve_series(skeleton, library, joint_index, primitive_index):
    """Preserve every captured sample; never invent unchanged missing targets."""
    result = MorphCurveSeries()
    if (library is None or not 0 <= joint_index < skeleton.num_bones
            or not 0 <= primitive_index < library.base.num_ellipsoids):
        return result
    base = library.base
    try:
        basis = _pose_rotations(skeleton, library.base_pose)[joint_index]
        position = _row(base.local_centers, primitive_index, 3)
        radii = _row(base.local_radii, primitive_index, 3)
        _unit_quaternions(_row(base.local_rotations, primitive_index, 4))
        exponents = (_row(base.shape_exponents, primitive_index, 2)
                     if base.primitive_type == "superquadric" else np.ones(2))
        if np.any(radii <= 0) or np.any(exponents <= 0):
            raise ValueError("Nonpositive primitive parameters")
        exponents = np.clip(exponents, .1, 2)
    except (ValueError, IndexError, TypeError):
        result.invalid_values = 1 + len(library.keys)
        return result
    result.samples.append(MorphCurveSample("Base pose", -1, np.zeros(3),
        position, radii, np.zeros(3), exponents))
    for index, key in enumerate(library.keys):
        if key.pose is None:
            result.missing_pose += 1
            continue
        try:
            current = _pose_rotations(skeleton, key.pose)[joint_index]
            angles = relative_rotation_degrees(basis, current)
            center_delta = _row(key.delta_centers, primitive_index, 3)
            log_radii = _row(key.delta_log_radii, primitive_index, 3)
            rotation = relative_rotation_degrees(np.array([0., 0., 0., 1.]),
                _row(key.delta_rotations, primitive_index, 4))
            log_eps = (np.zeros(2) if key.delta_log_shape_exponents is None
                       else _row(key.delta_log_shape_exponents, primitive_index, 2))
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                corrected_radii = radii * np.exp(log_radii)
                corrected_center = position + center_delta
                corrected_eps = (np.exp(np.clip(np.log(exponents) + log_eps, np.log(.1), np.log(2)))
                                 if base.primitive_type == "superquadric" else np.ones(2))
            if not np.isfinite(corrected_radii).all() or np.any(corrected_radii <= 0):
                raise ValueError("Invalid fitted radii")
            result.samples.append(MorphCurveSample(key.name or f"Pose {index + 1}", index,
                angles, corrected_center, corrected_radii, rotation, corrected_eps))
        except IndexError:
            result.missing_parameters += 1
        except (ValueError, TypeError, FloatingPointError):
            result.invalid_values += 1
    return result
