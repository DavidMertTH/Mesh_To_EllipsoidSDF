"""Regression check for retargeting saved Mixamo poses to Female Mannequin."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_library import _humanoid_slot, load_pose  # noqa: E402
from rig_loader import load_rigged_mesh  # noqa: E402
from skinning import deform_mesh  # noqa: E402
from skeleton import Pose  # noqa: E402


def main() -> None:
    rig = load_rigged_mesh(ROOT / "meshes" / "Female Mannequin.fbx")
    pose = load_pose(ROOT / "poses" / "Fight_Guard.json", rig.skeleton)

    required = {
        "left_upper_leg",
        "left_lower_leg",
        "left_foot",
        "right_upper_leg",
        "right_lower_leg",
        "right_foot",
        "left_upper_arm",
        "left_lower_arm",
        "right_upper_arm",
        "right_lower_arm",
    }

    target_slots = {_humanoid_slot(b.name) for b in rig.skeleton.bones}
    pose_slots = {
        _humanoid_slot(rig.skeleton.bones[i].name)
        for i in pose.bone_locals
    }

    assert _humanoid_slot("mixamorig1:LeftUpLeg") == "left_upper_leg"
    assert _humanoid_slot("mixamorig1:LeftLeg") == "left_lower_leg"
    assert _humanoid_slot("mixamorig1:RightUpLeg") == "right_upper_leg"
    assert _humanoid_slot("mixamorig1:RightLeg") == "right_lower_leg"
    assert _humanoid_slot("Left_thigh") == "left_upper_leg"
    assert _humanoid_slot("Left_shin") == "left_lower_leg"
    assert _humanoid_slot("Left_ankle") == "left_foot"

    missing_target = sorted(required - target_slots)
    missing_pose = sorted(required - pose_slots)
    assert not missing_target, f"missing target slots: {missing_target}"
    assert not missing_pose, f"missing pose slots: {missing_pose}"

    bind_verts = deform_mesh(
        rig.vertices,
        rig.skin_joints,
        rig.skin_weights,
        rig.skeleton.compute_skin_matrices(Pose.t_pose()),
        device="cpu",
    )
    posed_verts = deform_mesh(
        rig.vertices,
        rig.skin_joints,
        rig.skin_weights,
        rig.skeleton.compute_skin_matrices(pose),
        device="cpu",
    )
    bind_span = np.ptp(bind_verts, axis=0)
    posed_span = np.ptp(posed_verts, axis=0)
    assert posed_span[1] > 0.75 * bind_span[1], (
        f"retargeted mannequin collapsed/twisted: bind span={bind_span}, "
        f"posed span={posed_span}"
    )

    print("Female Mannequin bones:", rig.skeleton.num_bones)
    print("Retargeted pose bones:", len(pose.bone_locals))
    print("Bind span:", np.round(bind_span, 3).tolist())
    print("Pose span:", np.round(posed_span, 3).tolist())
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
