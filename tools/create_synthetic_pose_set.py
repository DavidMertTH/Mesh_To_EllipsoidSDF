"""Generate a compact humanoid synthetic coverage pose clip.

The clip is authored on meshes/T-Pose.fbx and saved through pose_library so it
contains both source bone names and canonical humanoid slots.  That makes it
usable on rigs whose bone names differ but still map to the same humanoid slots.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_library import _humanoid_slot, save_animation
from rig_loader import load_rigged_mesh
from skeleton import Pose, mat4_compose, mat4_decompose, quat_multiply


REFERENCE_MESH = ROOT / "meshes" / "T-Pose.fbx"
POSE_DIR = ROOT / "poses"
CLIP_NAME = "Synthetic Coverage"


def _axis_angle(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n <= 1.0e-12 or abs(float(degrees)) <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    a /= n
    half = np.deg2rad(float(degrees)) * 0.5
    s = np.sin(half)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(half)],
                    dtype=np.float64)


def _euler_delta(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Local XYZ delta quaternion in degrees."""
    qx = _axis_angle((1, 0, 0), x)
    qy = _axis_angle((0, 1, 0), y)
    qz = _axis_angle((0, 0, 1), z)
    return quat_multiply(quat_multiply(qx, qy), qz)


def _slot_map(skeleton) -> dict[str, int]:
    out: dict[str, int] = {}
    for bone in skeleton.bones:
        slot = _humanoid_slot(bone.name)
        if slot is not None and slot not in out:
            out[slot] = bone.index
    return out


def _make_pose(skeleton, slots: dict[str, int], name: str,
               deltas: dict[str, tuple[float, float, float]]) -> Pose:
    locals_out = {}
    for slot, xyz in deltas.items():
        idx = slots.get(slot)
        if idx is None:
            continue
        bind = skeleton.bones[idx].local_bind_transform.astype(np.float64)
        t, q_bind, scale = mat4_decompose(bind)
        q = quat_multiply(q_bind, _euler_delta(*xyz))
        locals_out[idx] = mat4_compose(t, q, scale)
    return Pose(name=name, bone_locals=locals_out)


def _mirror_lr(base: dict[str, tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
    """Swap left/right keys and mirror the main sideways channels."""
    out: dict[str, tuple[float, float, float]] = {}
    for slot, (x, y, z) in base.items():
        if slot.startswith("left_"):
            key = "right_" + slot[len("left_"):]
            out[key] = (x, -y, -z)
        elif slot.startswith("right_"):
            key = "left_" + slot[len("right_"):]
            out[key] = (x, -y, -z)
        else:
            out[slot] = (x, -y, -z)
    return out


def _with_fingers(amount: float) -> dict[str, tuple[float, float, float]]:
    out = {}
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for seg, mult in ((1, 1.0), (2, 0.75), (3, 0.55)):
                out[f"{side}_{finger}_{seg}"] = (amount * mult, 0.0, 0.0)
    return out


def build_poses(skeleton) -> list[Pose]:
    slots = _slot_map(skeleton)
    frames: list[tuple[str, dict[str, tuple[float, float, float]]]] = [
        ("Synthetic Coverage 00 T-Pose", {}),
        ("Synthetic Coverage 01 A-Pose Relaxed", {
            "left_upper_arm": (0, 0, -28), "right_upper_arm": (0, 0, 28),
            "left_lower_arm": (0, 0, -8), "right_lower_arm": (0, 0, 8),
            "left_hand": (8, 0, -4), "right_hand": (8, 0, 4),
        }),
        ("Synthetic Coverage 02 Arms Overhead", {
            "left_upper_arm": (-95, 0, -18), "right_upper_arm": (-95, 0, 18),
            "left_lower_arm": (-18, 0, -8), "right_lower_arm": (-18, 0, 8),
            "chest": (-8, 0, 0), "neck": (8, 0, 0),
        }),
        ("Synthetic Coverage 03 Forward Reach", {
            "left_upper_arm": (-58, 24, -4), "right_upper_arm": (-58, -24, 4),
            "left_lower_arm": (-10, 0, -8), "right_lower_arm": (-10, 0, 8),
            "left_hand": (0, 8, 0), "right_hand": (0, -8, 0),
        }),
        ("Synthetic Coverage 04 Wide T Twist Wrists", {
            "left_upper_arm": (8, -18, 8), "right_upper_arm": (8, 18, -8),
            "left_lower_arm": (0, 28, -10), "right_lower_arm": (0, -28, 10),
            "left_hand": (0, 30, 0), "right_hand": (0, -30, 0),
        }),
        ("Synthetic Coverage 05 Guard Elbows Bent", {
            "left_upper_arm": (-30, 32, -20), "right_upper_arm": (-30, -32, 20),
            "left_lower_arm": (-82, 12, 6), "right_lower_arm": (-82, -12, -6),
            "left_hand": (-8, 18, 0), "right_hand": (-8, -18, 0),
            "chest": (0, 0, 6), "head": (0, 0, -4),
        } | _with_fingers(32)),
        ("Synthetic Coverage 06 Cross Body", {
            "left_upper_arm": (-44, 58, 16), "right_upper_arm": (-44, -58, -16),
            "left_lower_arm": (-74, 8, -12), "right_lower_arm": (-74, -8, 12),
            "left_hand": (0, 25, 0), "right_hand": (0, -25, 0),
            "chest": (0, 0, -8),
        }),
        ("Synthetic Coverage 07 Forward Bend", {
            "hips": (16, 0, 0), "spine": (18, 0, 0),
            "upper_spine": (16, 0, 0), "chest": (12, 0, 0),
            "neck": (-10, 0, 0), "head": (-8, 0, 0),
            "left_upper_arm": (20, 0, -18), "right_upper_arm": (20, 0, 18),
        }),
        ("Synthetic Coverage 08 Back Arch", {
            "hips": (-10, 0, 0), "spine": (-15, 0, 0),
            "upper_spine": (-14, 0, 0), "chest": (-10, 0, 0),
            "neck": (8, 0, 0), "head": (8, 0, 0),
            "left_upper_arm": (-42, 0, -10), "right_upper_arm": (-42, 0, 10),
        }),
        ("Synthetic Coverage 09 Twist Left", {
            "hips": (0, 0, -10), "spine": (0, 0, -14),
            "upper_spine": (0, 0, -16), "chest": (0, 0, -18),
            "neck": (0, 0, 12), "head": (0, 0, 10),
            "left_upper_arm": (-12, 10, -12), "right_upper_arm": (-12, -10, 12),
        }),
        ("Synthetic Coverage 10 Twist Right", {
            "hips": (0, 0, 10), "spine": (0, 0, 14),
            "upper_spine": (0, 0, 16), "chest": (0, 0, 18),
            "neck": (0, 0, -12), "head": (0, 0, -10),
            "left_upper_arm": (-12, 10, -12), "right_upper_arm": (-12, -10, 12),
        }),
        ("Synthetic Coverage 11 Side Bend Left", {
            "hips": (0, 0, -6), "spine": (0, -14, 0),
            "upper_spine": (0, -16, 0), "chest": (0, -12, 0),
            "neck": (0, 10, 0), "head": (0, 8, 0),
            "left_upper_arm": (-24, 0, -24), "right_upper_arm": (18, 0, 28),
        }),
        ("Synthetic Coverage 12 Side Bend Right", {
            "hips": (0, 0, 6), "spine": (0, 14, 0),
            "upper_spine": (0, 16, 0), "chest": (0, 12, 0),
            "neck": (0, -10, 0), "head": (0, -8, 0),
            "left_upper_arm": (18, 0, -28), "right_upper_arm": (-24, 0, 24),
        }),
        ("Synthetic Coverage 13 Walk Left Forward", {
            "hips": (0, 0, -4),
            "left_upper_leg": (-34, 4, 2), "left_lower_leg": (22, 0, 0),
            "left_foot": (-8, 0, 0), "left_toe": (10, 0, 0),
            "right_upper_leg": (24, -3, -2), "right_lower_leg": (-18, 0, 0),
            "right_foot": (10, 0, 0),
            "left_upper_arm": (24, 0, -10), "right_upper_arm": (-28, 0, 10),
            "left_lower_arm": (-18, 0, 0), "right_lower_arm": (-24, 0, 0),
        }),
        ("Synthetic Coverage 14 Walk Right Forward", {
            "hips": (0, 0, 4),
            "right_upper_leg": (-34, -4, -2), "right_lower_leg": (22, 0, 0),
            "right_foot": (-8, 0, 0), "right_toe": (10, 0, 0),
            "left_upper_leg": (24, 3, 2), "left_lower_leg": (-18, 0, 0),
            "left_foot": (10, 0, 0),
            "right_upper_arm": (24, 0, 10), "left_upper_arm": (-28, 0, -10),
            "right_lower_arm": (-18, 0, 0), "left_lower_arm": (-24, 0, 0),
        }),
        ("Synthetic Coverage 15 Deep Crouch", {
            "hips": (22, 0, 0), "spine": (14, 0, 0), "chest": (8, 0, 0),
            "left_upper_leg": (-72, 8, 4), "right_upper_leg": (-72, -8, -4),
            "left_lower_leg": (92, 0, 0), "right_lower_leg": (92, 0, 0),
            "left_foot": (-28, 0, 0), "right_foot": (-28, 0, 0),
            "left_upper_arm": (-28, 20, -12), "right_upper_arm": (-28, -20, 12),
            "left_lower_arm": (-45, 0, 0), "right_lower_arm": (-45, 0, 0),
        }),
        ("Synthetic Coverage 16 Wide Squat", {
            "hips": (12, 0, 0), "spine": (8, 0, 0),
            "left_upper_leg": (-48, 24, 12), "right_upper_leg": (-48, -24, -12),
            "left_lower_leg": (68, 0, 0), "right_lower_leg": (68, 0, 0),
            "left_foot": (-18, 10, 0), "right_foot": (-18, -10, 0),
            "left_upper_arm": (-8, 0, -32), "right_upper_arm": (-8, 0, 32),
        }),
        ("Synthetic Coverage 17 Left Lunge", {
            "hips": (10, 0, -8), "spine": (8, 0, 6), "chest": (4, 0, 8),
            "left_upper_leg": (-62, 10, 4), "left_lower_leg": (78, 0, 0),
            "left_foot": (-14, 0, 0),
            "right_upper_leg": (26, -8, -4), "right_lower_leg": (-12, 0, 0),
            "right_foot": (8, 0, 0),
            "left_upper_arm": (-18, 16, -8), "right_upper_arm": (-38, -18, 16),
        }),
        ("Synthetic Coverage 18 Right Lunge", _mirror_lr({
            "left_upper_leg": (-62, 10, 4), "left_lower_leg": (78, 0, 0),
            "left_foot": (-14, 0, 0),
            "right_upper_leg": (26, -8, -4), "right_lower_leg": (-12, 0, 0),
            "right_foot": (8, 0, 0),
            "left_upper_arm": (-18, 16, -8), "right_upper_arm": (-38, -18, 16),
            "hips": (10, 0, -8), "spine": (8, 0, 6), "chest": (4, 0, 8),
        })),
        ("Synthetic Coverage 19 Tiptoe Reach", {
            "left_foot": (34, 0, 0), "right_foot": (34, 0, 0),
            "left_toe": (-18, 0, 0), "right_toe": (-18, 0, 0),
            "left_upper_arm": (-86, 0, -10), "right_upper_arm": (-86, 0, 10),
            "left_lower_arm": (-12, 0, 0), "right_lower_arm": (-12, 0, 0),
            "spine": (-6, 0, 0), "chest": (-8, 0, 0), "head": (6, 0, 0),
        }),
        ("Synthetic Coverage 20 Hands Open Close", {
            "left_upper_arm": (-18, 18, -14), "right_upper_arm": (-18, -18, 14),
            "left_lower_arm": (-68, 8, 0), "right_lower_arm": (-68, -8, 0),
        } | _with_fingers(58)),
        ("Synthetic Coverage 21 Extreme Forward Fold", {
            "hips": (38, 0, 0), "spine": (34, 0, 0),
            "upper_spine": (30, 0, 0), "chest": (24, 0, 0),
            "neck": (-24, 0, 0), "head": (-20, 0, 0),
            "left_upper_arm": (42, 0, -12), "right_upper_arm": (42, 0, 12),
            "left_lower_arm": (-18, 0, -8), "right_lower_arm": (-18, 0, 8),
            "left_upper_leg": (8, 5, 0), "right_upper_leg": (8, -5, 0),
            "left_lower_leg": (-8, 0, 0), "right_lower_leg": (-8, 0, 0),
        }),
        ("Synthetic Coverage 22 Extreme Backbend Overhead", {
            "hips": (-22, 0, 0), "spine": (-32, 0, 0),
            "upper_spine": (-30, 0, 0), "chest": (-26, 0, 0),
            "neck": (18, 0, 0), "head": (20, 0, 0),
            "left_upper_arm": (-122, 0, -20), "right_upper_arm": (-122, 0, 20),
            "left_lower_arm": (-28, 0, -10), "right_lower_arm": (-28, 0, 10),
            "left_foot": (28, 0, 0), "right_foot": (28, 0, 0),
            "left_toe": (-14, 0, 0), "right_toe": (-14, 0, 0),
        }),
        ("Synthetic Coverage 23 Extreme Twist Left Reach", {
            "hips": (0, 0, -22), "spine": (0, 0, -30),
            "upper_spine": (0, 0, -34), "chest": (0, 0, -38),
            "neck": (0, 0, 24), "head": (0, 0, 20),
            "left_upper_arm": (-72, 26, -26), "left_lower_arm": (-36, 10, -14),
            "right_upper_arm": (28, -46, 34), "right_lower_arm": (-80, -12, 10),
            "left_hand": (0, 36, 0), "right_hand": (0, -38, 0),
        }),
        ("Synthetic Coverage 24 Extreme Twist Right Reach", _mirror_lr({
            "hips": (0, 0, -22), "spine": (0, 0, -30),
            "upper_spine": (0, 0, -34), "chest": (0, 0, -38),
            "neck": (0, 0, 24), "head": (0, 0, 20),
            "left_upper_arm": (-72, 26, -26), "left_lower_arm": (-36, 10, -14),
            "right_upper_arm": (28, -46, 34), "right_lower_arm": (-80, -12, 10),
            "left_hand": (0, 36, 0), "right_hand": (0, -38, 0),
        })),
        ("Synthetic Coverage 25 Extreme Side Crunch Left", {
            "hips": (0, 0, -14), "spine": (0, -30, 0),
            "upper_spine": (0, -34, 0), "chest": (0, -30, 0),
            "neck": (0, 20, 0), "head": (0, 18, 0),
            "left_upper_arm": (-50, 0, -48), "left_lower_arm": (-52, 0, -12),
            "right_upper_arm": (38, 0, 52), "right_lower_arm": (-18, 0, 18),
            "left_upper_leg": (-22, 18, 10), "right_upper_leg": (-8, -12, -8),
        }),
        ("Synthetic Coverage 26 Extreme Side Crunch Right", _mirror_lr({
            "hips": (0, 0, -14), "spine": (0, -30, 0),
            "upper_spine": (0, -34, 0), "chest": (0, -30, 0),
            "neck": (0, 20, 0), "head": (0, 18, 0),
            "left_upper_arm": (-50, 0, -48), "left_lower_arm": (-52, 0, -12),
            "right_upper_arm": (38, 0, 52), "right_lower_arm": (-18, 0, 18),
            "left_upper_leg": (-22, 18, 10), "right_upper_leg": (-8, -12, -8),
        })),
        ("Synthetic Coverage 27 Extreme Left High Kick", {
            "hips": (0, 0, -10), "spine": (10, 0, 8), "chest": (6, 0, 10),
            "left_upper_leg": (-118, 6, 2), "left_lower_leg": (18, 0, 0),
            "left_foot": (-22, 0, 0), "left_toe": (12, 0, 0),
            "right_upper_leg": (18, -8, -4), "right_lower_leg": (-8, 0, 0),
            "right_foot": (12, 0, 0),
            "left_upper_arm": (22, 0, -20), "right_upper_arm": (-62, -10, 24),
            "left_lower_arm": (-28, 0, 0), "right_lower_arm": (-36, 0, 0),
        }),
        ("Synthetic Coverage 28 Extreme Right High Kick", _mirror_lr({
            "hips": (0, 0, -10), "spine": (10, 0, 8), "chest": (6, 0, 10),
            "left_upper_leg": (-118, 6, 2), "left_lower_leg": (18, 0, 0),
            "left_foot": (-22, 0, 0), "left_toe": (12, 0, 0),
            "right_upper_leg": (18, -8, -4), "right_lower_leg": (-8, 0, 0),
            "right_foot": (12, 0, 0),
            "left_upper_arm": (22, 0, -20), "right_upper_arm": (-62, -10, 24),
            "left_lower_arm": (-28, 0, 0), "right_lower_arm": (-36, 0, 0),
        })),
        ("Synthetic Coverage 29 Extreme Left Front Split", {
            "hips": (8, 0, -4), "spine": (8, 0, 4), "chest": (4, 0, 5),
            "left_upper_leg": (-88, 4, 0), "left_lower_leg": (6, 0, 0),
            "left_foot": (-18, 0, 0),
            "right_upper_leg": (78, -4, 0), "right_lower_leg": (-8, 0, 0),
            "right_foot": (18, 0, 0),
            "left_upper_arm": (-20, 16, -28), "right_upper_arm": (-24, -16, 28),
        }),
        ("Synthetic Coverage 30 Extreme Right Front Split", _mirror_lr({
            "hips": (8, 0, -4), "spine": (8, 0, 4), "chest": (4, 0, 5),
            "left_upper_leg": (-88, 4, 0), "left_lower_leg": (6, 0, 0),
            "left_foot": (-18, 0, 0),
            "right_upper_leg": (78, -4, 0), "right_lower_leg": (-8, 0, 0),
            "right_foot": (18, 0, 0),
            "left_upper_arm": (-20, 16, -28), "right_upper_arm": (-24, -16, 28),
        })),
        ("Synthetic Coverage 31 Extreme Side Split", {
            "hips": (16, 0, 0), "spine": (10, 0, 0), "chest": (6, 0, 0),
            "left_upper_leg": (-18, 68, 18), "right_upper_leg": (-18, -68, -18),
            "left_lower_leg": (8, 0, 0), "right_lower_leg": (8, 0, 0),
            "left_foot": (-12, 18, 0), "right_foot": (-12, -18, 0),
            "left_upper_arm": (-28, 0, -58), "right_upper_arm": (-28, 0, 58),
            "left_lower_arm": (-18, 0, -8), "right_lower_arm": (-18, 0, 8),
        }),
        ("Synthetic Coverage 32 Extreme Deep Kneel", {
            "hips": (30, 0, 0), "spine": (18, 0, 0), "chest": (12, 0, 0),
            "left_upper_leg": (-98, 8, 2), "right_upper_leg": (-98, -8, -2),
            "left_lower_leg": (128, 0, 0), "right_lower_leg": (128, 0, 0),
            "left_foot": (-42, 0, 0), "right_foot": (-42, 0, 0),
            "left_upper_arm": (-34, 22, -14), "right_upper_arm": (-34, -22, 14),
            "left_lower_arm": (-78, 6, 0), "right_lower_arm": (-78, -6, 0),
        } | _with_fingers(48)),
        ("Synthetic Coverage 33 Extreme Knee To Chest Left", {
            "hips": (18, 0, -8), "spine": (12, 0, 8), "chest": (8, 0, 8),
            "left_upper_leg": (-126, 8, 0), "left_lower_leg": (118, 0, 0),
            "left_foot": (-28, 0, 0),
            "right_upper_leg": (10, -4, 0), "right_lower_leg": (-6, 0, 0),
            "left_upper_arm": (-54, 34, -16), "right_upper_arm": (-28, -20, 18),
            "left_lower_arm": (-92, 8, -6), "right_lower_arm": (-76, -6, 8),
        }),
        ("Synthetic Coverage 34 Extreme Knee To Chest Right", _mirror_lr({
            "hips": (18, 0, -8), "spine": (12, 0, 8), "chest": (8, 0, 8),
            "left_upper_leg": (-126, 8, 0), "left_lower_leg": (118, 0, 0),
            "left_foot": (-28, 0, 0),
            "right_upper_leg": (10, -4, 0), "right_lower_leg": (-6, 0, 0),
            "left_upper_arm": (-54, 34, -16), "right_upper_arm": (-28, -20, 18),
            "left_lower_arm": (-92, 8, -6), "right_lower_arm": (-76, -6, 8),
        })),
        ("Synthetic Coverage 35 Extreme Arms Behind Back", {
            "chest": (-6, 0, 0), "neck": (8, 0, 0), "head": (6, 0, 0),
            "left_upper_arm": (58, -42, -26), "right_upper_arm": (58, 42, 26),
            "left_lower_arm": (-92, -18, -22), "right_lower_arm": (-92, 18, 22),
            "left_hand": (16, -48, -14), "right_hand": (16, 48, 14),
        } | _with_fingers(36)),
        ("Synthetic Coverage 36 Extreme Self Hug", {
            "chest": (4, 0, -4), "neck": (-4, 0, 4), "head": (-2, 0, 4),
            "left_upper_arm": (-56, 76, 20), "right_upper_arm": (-56, -76, -20),
            "left_lower_arm": (-104, 14, -18), "right_lower_arm": (-104, -14, 18),
            "left_hand": (-8, 44, 0), "right_hand": (-8, -44, 0),
        } | _with_fingers(42)),
        ("Synthetic Coverage 37 Extreme Hands Behind Head", {
            "chest": (-8, 0, 0), "neck": (8, 0, 0), "head": (8, 0, 0),
            "left_upper_arm": (-76, 38, -62), "right_upper_arm": (-76, -38, 62),
            "left_lower_arm": (-122, 6, 24), "right_lower_arm": (-122, -6, -24),
            "left_hand": (18, 42, 0), "right_hand": (18, -42, 0),
        } | _with_fingers(26)),
        ("Synthetic Coverage 38 Extreme Claw Fingers", {
            "left_upper_arm": (-22, 24, -18), "right_upper_arm": (-22, -24, 18),
            "left_lower_arm": (-86, 16, -6), "right_lower_arm": (-86, -16, 6),
            "left_hand": (-26, 46, -12), "right_hand": (-26, -46, 12),
        } | _with_fingers(86)),
        ("Synthetic Coverage 39 Extreme Open Fingers", {
            "left_upper_arm": (-16, 18, -22), "right_upper_arm": (-16, -18, 22),
            "left_lower_arm": (-42, 24, -10), "right_lower_arm": (-42, -24, 10),
            "left_hand": (18, 52, 18), "right_hand": (18, -52, -18),
        } | _with_fingers(-34)),
        ("Synthetic Coverage 40 Extreme Asymmetric Balance", {
            "hips": (6, 0, 18), "spine": (0, -10, -12),
            "upper_spine": (-8, -12, -18), "chest": (-10, -12, -22),
            "neck": (6, 8, 10), "head": (8, 8, 12),
            "left_upper_leg": (-92, 22, 8), "left_lower_leg": (96, 0, 0),
            "left_foot": (-32, 0, 0),
            "right_upper_leg": (34, -26, -12), "right_lower_leg": (-26, 0, 0),
            "right_foot": (22, 0, 0),
            "left_upper_arm": (-118, -8, -32), "right_upper_arm": (42, -52, 44),
            "left_lower_arm": (-34, 0, -8), "right_lower_arm": (-88, -20, 14),
        }),
        ("Synthetic Coverage 41 Extreme Crawl Reach", {
            "hips": (34, 0, 0), "spine": (28, 0, 0), "chest": (18, 0, 0),
            "neck": (-22, 0, 0), "head": (-18, 0, 0),
            "left_upper_arm": (72, 12, -24), "left_lower_arm": (-96, 0, -8),
            "right_upper_arm": (-82, -8, 22), "right_lower_arm": (-18, 0, 0),
            "left_upper_leg": (-74, 18, 8), "left_lower_leg": (112, 0, 0),
            "right_upper_leg": (42, -18, -8), "right_lower_leg": (82, 0, 0),
            "left_foot": (-26, 0, 0), "right_foot": (-30, 0, 0),
        }),
        ("Synthetic Coverage 42 Extreme Low Side Lunge Left", {
            "hips": (24, 0, -12), "spine": (18, 0, 10), "chest": (10, 0, 12),
            "left_upper_leg": (-92, 26, 12), "left_lower_leg": (122, 0, 0),
            "left_foot": (-36, 12, 0),
            "right_upper_leg": (24, -58, -18), "right_lower_leg": (-12, 0, 0),
            "right_foot": (14, -22, 0),
            "left_upper_arm": (-32, 28, -18), "right_upper_arm": (-54, -30, 28),
            "left_lower_arm": (-70, 0, 0), "right_lower_arm": (-38, 0, 0),
        }),
        ("Synthetic Coverage 43 Extreme Low Side Lunge Right", _mirror_lr({
            "hips": (24, 0, -12), "spine": (18, 0, 10), "chest": (10, 0, 12),
            "left_upper_leg": (-92, 26, 12), "left_lower_leg": (122, 0, 0),
            "left_foot": (-36, 12, 0),
            "right_upper_leg": (24, -58, -18), "right_lower_leg": (-12, 0, 0),
            "right_foot": (14, -22, 0),
            "left_upper_arm": (-32, 28, -18), "right_upper_arm": (-54, -30, 28),
            "left_lower_arm": (-70, 0, 0), "right_lower_arm": (-38, 0, 0),
        })),
    ]
    return [_make_pose(skeleton, slots, name, deltas) for name, deltas in frames]


def main() -> None:
    rig = load_rigged_mesh(REFERENCE_MESH)
    poses = build_poses(rig.skeleton)
    path = save_animation(poses, rig.skeleton, CLIP_NAME, POSE_DIR)
    print(f"Wrote {len(poses)} synthetic coverage poses to {path}")


if __name__ == "__main__":
    main()
