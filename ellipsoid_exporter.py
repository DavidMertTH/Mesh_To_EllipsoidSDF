"""
ellipsoid_exporter.py — Export bone-local ellipsoids to JSON for Unity import.

Coordinate system used in this file (matches our internal representation):
  Right-hand, Y-up (same as FBX default, Blender, Maya).

The Unity importer applies a mirror-X conversion to get into Unity's
left-hand Y-up system:
  position  : (x, y, z) → (-x, y, z)
  quaternion: (x, y, z, w) → (-x, y, z, w)   [negate x component only]

Quaternion convention throughout: [x, y, z, w]
"""

from __future__ import annotations

import json
from pathlib import Path

from bone_ellipsoid_mapper import BoneLocalEllipsoids
from skeleton import Skeleton


def export_ellipsoids(
    bone_local: BoneLocalEllipsoids,
    skeleton: Skeleton,
    filepath: str | Path,
) -> int:
    """Export bone-local ellipsoid data to a JSON file for Unity.

    Parameters
    ----------
    bone_local : trained BoneLocalEllipsoids
    skeleton   : used for bone-name lookup
    filepath   : destination .json path

    Returns
    -------
    Number of ellipsoids written.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(bone_local.num_ellipsoids):
        bi = int(bone_local.bone_assignments[i])
        entries.append({
            "bone":           skeleton.bones[bi].name,
            # offset from bone origin, expressed in bone's orientation frame
            "local_center":   [round(float(v), 7) for v in bone_local.local_centers[i]],
            # ellipsoid semi-axes (half-extents)
            "radii":          [round(float(v), 7) for v in bone_local.local_radii[i]],
            # orientation relative to bone frame — quaternion [x, y, z, w]
            "local_rotation": [round(float(v), 7) for v in bone_local.local_rotations[i]],
        })

    payload = {
        "version":               1,
        "coordinate_system":     "right_hand_y_up",
        "quaternion_convention": "xyzw",
        "count":                 len(entries),
        "ellipsoids":            entries,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Exporter] {len(entries)} ellipsoids → {filepath}")
    return len(entries)
