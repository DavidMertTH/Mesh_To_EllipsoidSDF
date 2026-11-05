"""Authored rotational allowances calibrated to the shipped T-Pose skeleton.

The profile uses bind-local axes, not animation Euler channels or Unity muscle
axes. Equivalent rigs are matched by names and bind frames; axis permutations
and signs are transferred explicitly. These settings describe the joint viewer
and do not clamp skinning, animation or pose-corrective fitting.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
import json
from pathlib import Path
import re

import numpy as np

from joint_dof import joint_rotation_frames


PROFILE_PATH = Path(__file__).parent / "assets" / "joint_limits" / "t_pose.json"


def _bone_key(name: str) -> str:
    """Ignore exporter namespaces and harmless name separators."""
    return re.sub(r"[^a-z0-9]", "", str(name).rsplit(":", 1)[-1].lower())


@lru_cache(maxsize=1)
def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def apply_humanoid_joint_limits(skeleton, reference_world=None) -> int:
    """Initialize unedited joints on a T-Pose-equivalent rig; return count.

    Reference worlds must be bind transforms, even when a Unity snapshot is
    currently posed. Unrecognized rigs and uncertain axis matches stay as-is.
    """
    names = {}
    for bone in skeleton.bones:
        key = _bone_key(bone.name)
        if key in names:
            return 0  # Ambiguous names cannot safely identify fixed joints.
        names[key] = bone
    profile = _profile()
    if not all(_bone_key(name) in names for name in profile["required_bones"]):
        return 0
    if reference_world is None:
        reference_world = skeleton.compute_world_transforms(None)
    frames = joint_rotation_frames(reference_world)
    if len(frames) != skeleton.num_bones:
        raise ValueError("Joint-limit reference must match the skeleton")
    root = names[_bone_key(profile["reference_bone"])]
    # Anatomical directions come from the bind geometry, so a different root
    # axis convention or a globally rotated model does not change the limits.
    positions = np.asarray(reference_world, dtype=np.float64)[:, :3, 3]
    up = positions[names["head"].index] - positions[root.index]
    lateral = (positions[names["leftupleg"].index]
               - positions[names["rightupleg"].index])
    up_length = np.linalg.norm(up)
    if up_length < 1e-9:
        return 0
    up /= up_length
    lateral -= np.dot(lateral, up) * up
    lateral_length = np.linalg.norm(lateral)
    if lateral_length < 1e-9:
        return 0
    lateral /= lateral_length
    body_frame = np.column_stack((lateral, up, np.cross(lateral, up)))
    applied = 0
    for entry in profile["bones"]:
        bone = names.get(_bone_key(entry["name"]))
        if bone is None:
            continue
        dof = bone.joint_dof
        # Never overwrite settings already supplied or edited by the user.
        if not (np.all(dof.enabled) and np.all(dof.minimum == -180.0)
                and np.all(dof.maximum == 180.0)):
            continue
        reference = np.asarray(entry["axes"], dtype=np.float64)
        target = body_frame.T @ frames[bone.index]
        alignment = target.T @ reference
        assignment = max(permutations(range(3)), key=lambda axes: sum(
            abs(alignment[axes[source], source]) for source in range(3)))
        # Do not silently interpret a substantially different joint orientation
        # as the authored hinge axis. Locked end markers need no axis matching.
        if any(entry["enabled"]) and min(
                abs(alignment[assignment[source], source])
                for source in range(3)) < 0.9:
            continue
        for source, axis in enumerate(assignment):
            minimum = float(entry["minimum"][source])
            maximum = float(entry["maximum"][source])
            if alignment[axis, source] < 0.0:
                minimum, maximum = -maximum, -minimum
            dof.set_axis(axis, bool(entry["enabled"][source]), minimum, maximum)
        applied += 1
    if applied:
        skeleton.joint_limit_profile = profile["name"]
    return applied
