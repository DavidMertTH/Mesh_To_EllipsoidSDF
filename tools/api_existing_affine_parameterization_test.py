"""Regression for affine centers in the Unity fit-pose parameterization."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import attachment_parameter_transform  # noqa: E402
from main_window import MainWindow  # noqa: E402
from skeleton import mat4_compose, quat_from_matrix  # noqa: E402


def _axis_angle(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = np.radians(degrees) * 0.5
    return np.r_[axis * np.sin(half), np.cos(half)]


class _ApiState:
    _normalize_quat_np = staticmethod(MainWindow._normalize_quat_np)


def main() -> None:
    bind = mat4_compose(
        np.array([2.0, -1.0, 0.5]),
        _axis_angle([0.0, 0.0, 1.0], 20.0),
        np.array([2.0, 0.5, 1.25]),
    )
    current = mat4_compose(
        np.array([2.5, -0.7, 0.2]),
        _axis_angle([1.0, 0.0, 0.0], -30.0),
        np.array([1.4, 0.8, 1.7]),
    )
    center_offset = np.array([1.2, -0.3, 0.1])
    normalization_scale = 2.5
    local_center = np.array([0.3, -0.4, 0.2], dtype=np.float32)

    state = _ApiState()
    state._api_norm = SimpleNamespace(
        center=center_offset, scale=normalization_scale)
    state._api_rig = {"bones": [{
        "name": "ScaledBone",
        "matrix": bind.tolist(),
        "currentMatrix": current.tolist(),
    }]}
    state._api_initial_ellipsoid_meta = [{
        "local_center": local_center,
        "local_rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "local_radii": np.array([0.2, 0.1, 0.05], dtype=np.float32),
        "bone_index": 0,
        "attachment_bone_indices": [0],
        "attachment_weights": np.array([1.0], dtype=np.float32),
        "attachment_reference_positions": None,
        "attachment_reference_rotations": None,
    }]

    local, linear, offset, rotation_prefix = (
        MainWindow._api_existing_local_parameterization(state))
    actual_center = linear[0] @ local.local_centers[0] + offset[0]
    normalized_current = current.copy()
    normalized_current[:3, 3] = (
        (current[:3, 3] - center_offset) * normalization_scale)
    expected_center = (
        normalized_current @ np.r_[local_center.astype(np.float64), 1.0])[:3]
    np.testing.assert_allclose(actual_center, expected_center, atol=2.0e-6)

    reference_position = (
        (bind[:3, 3] - center_offset) * normalization_scale)[None, :]
    current_position = (
        (current[:3, 3] - center_offset) * normalization_scale)[None, :]
    reference_rotation = state._normalize_quat_np(quat_from_matrix(bind))[None, :]
    current_rotation = state._normalize_quat_np(quat_from_matrix(current))[None, :]
    _, _, rigid_prefix = attachment_parameter_transform(
        local,
        reference_position,
        reference_rotation,
        current_position,
        current_rotation,
    )
    np.testing.assert_allclose(
        np.abs(np.sum(rotation_prefix * rigid_prefix, axis=1)), [1.0], atol=1.0e-6)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
