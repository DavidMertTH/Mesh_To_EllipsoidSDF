"""Regression tests for the Unity API rig/mesh coordinate contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_rig_space import correct_unity_rig_space  # noqa: E402


def _mat(tx: float, ty: float, tz: float) -> list[list[float]]:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = [tx, ty, tz]
    return m.tolist()


def test_anatomical_bbox_offset_is_not_guessed() -> None:
    verts = np.array([
        [-0.4, -1.0, -0.2],
        [0.4, -1.0, 0.2],
        [-0.5, 1.0, -0.2],
        [0.5, 1.0, 0.2],
    ], dtype=np.float32)
    rig = {
        "bones": [
            {"name": "Wrist", "matrix": _mat(0.0, 0.0, 0.0), "parent": -1},
            {"name": "FingerTip", "matrix": _mat(0.0, 1.0, 0.0), "parent": 0},
        ],
        "poseFrames": [
            {"name": "pose", "boneMatrices": [_mat(0.0, 0.0, 0.0),
                                               _mat(0.0, 0.8, 0.0)]},
        ],
    }

    fixed, delta, reason = correct_unity_rig_space(rig, verts)

    assert fixed is rig
    assert reason is None
    assert np.allclose(delta, [0.0, 0.0, 0.0])
    wrist = np.asarray(fixed["bones"][0]["matrix"], dtype=np.float64)
    fingertip = np.asarray(fixed["bones"][1]["matrix"], dtype=np.float64)
    frame_tip = np.asarray(
        fixed["poseFrames"][0]["boneMatrices"][1], dtype=np.float64)
    assert np.allclose(wrist[:3, 3], [0.0, 0.0, 0.0])
    assert np.allclose(fingertip[:3, 3], [0.0, 1.0, 0.0])
    assert np.allclose(frame_tip[:3, 3], [0.0, 0.8, 0.0])


def test_aligned_rig_is_left_unchanged() -> None:
    verts = np.array([
        [-0.4, -1.0, -0.2],
        [0.4, -1.0, 0.2],
        [-0.5, 1.0, -0.2],
        [0.5, 1.0, 0.2],
    ], dtype=np.float32)
    rig = {
        "bones": [
            {"name": "Hips", "matrix": _mat(0.0, -0.8, 0.0), "parent": -1},
            {"name": "Head", "matrix": _mat(0.0, 0.8, 0.0), "parent": 0},
        ],
    }

    fixed, delta, reason = correct_unity_rig_space(rig, verts)

    assert fixed is rig
    assert reason is None
    assert np.allclose(delta, [0.0, 0.0, 0.0])


def test_world_to_bone_local_roundtrip_keeps_unity_origin() -> None:
    """A valid Unity bone must reconstruct the original world centre exactly."""
    verts = np.array([
        [-0.5, -1.0, -0.25],
        [0.5, -1.0, 0.25],
        [-0.5, 1.0, -0.25],
        [0.5, 1.0, 0.25],
    ], dtype=np.float32)
    bone_world = np.asarray(_mat(3.0, 0.0, -2.0), dtype=np.float64)
    rig = {
        "coordinateSpace": "unity_world",
        "bones": [{
            "name": "Palm",
            "matrix": bone_world.tolist(),
            "currentMatrix": bone_world.tolist(),
            "parent": -1,
        }],
    }

    fixed, delta, reason = correct_unity_rig_space(rig, verts)

    world_center = np.array([3.2, 0.1, -1.6, 1.0], dtype=np.float64)
    fixed_bone = np.asarray(
        fixed["bones"][0]["currentMatrix"], dtype=np.float64)
    local_center = np.linalg.inv(fixed_bone) @ world_center
    reconstructed = bone_world @ local_center
    assert fixed is rig
    assert reason is None
    assert np.allclose(delta, 0.0)
    assert np.allclose(reconstructed, world_center)


if __name__ == "__main__":
    test_anatomical_bbox_offset_is_not_guessed()
    test_aligned_rig_is_left_unchanged()
    test_world_to_bone_local_roundtrip_keeps_unity_origin()
    print("api_rig_space_test: ok")
