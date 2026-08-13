"""Smoke test for Unity API rig/mesh space correction."""

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


def test_vertical_world_offset_is_corrected() -> None:
    verts = np.array([
        [-0.4, -1.0, -0.2],
        [0.4, -1.0, 0.2],
        [-0.5, 1.0, -0.2],
        [0.5, 1.0, 0.2],
    ], dtype=np.float32)
    rig = {
        "bones": [
            {"name": "Hips", "matrix": _mat(0.0, -2.4, 0.0), "parent": -1},
            {"name": "Head", "matrix": _mat(0.0, -0.8, 0.0), "parent": 0},
        ],
        "poseFrames": [
            {"name": "pose", "boneMatrices": [_mat(0.0, -2.2, 0.0),
                                               _mat(0.0, -0.6, 0.0)]},
        ],
    }

    fixed, delta, reason = correct_unity_rig_space(rig, verts)

    assert reason is not None
    assert np.allclose(delta, [0.0, 1.6, 0.0])
    hips = np.asarray(fixed["bones"][0]["matrix"], dtype=np.float64)
    head = np.asarray(fixed["bones"][1]["matrix"], dtype=np.float64)
    frame_head = np.asarray(
        fixed["poseFrames"][0]["boneMatrices"][1], dtype=np.float64)
    assert np.allclose(hips[:3, 3], [0.0, -0.8, 0.0])
    assert np.allclose(head[:3, 3], [0.0, 0.8, 0.0])
    assert np.allclose(frame_head[:3, 3], [0.0, 1.0, 0.0])
    assert rig["bones"][0]["matrix"][1][3] == -2.4


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


if __name__ == "__main__":
    test_vertical_world_offset_is_corrected()
    test_aligned_rig_is_left_unchanged()
    print("api_rig_space_test: ok")
