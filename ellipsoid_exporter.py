"""
ellipsoid_exporter.py — Export bone-local primitives to JSON for Unity import.

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
from collections import defaultdict
from pathlib import Path

from bone_ellipsoid_mapper import BoneLocalEllipsoids
from rig_ingest import attachment_entry_fields, sphere_name
from skeleton import Skeleton


def export_ellipsoids(
    bone_local: BoneLocalEllipsoids,
    skeleton: Skeleton,
    filepath: str | Path,
) -> int:
    """Export bone-local ellipsoid/superquadric data for Unity.

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
    counts = defaultdict(int)
    for i in range(bone_local.num_ellipsoids):
        bi = int(bone_local.bone_assignments[i])
        bone_name = skeleton.bones[bi].name
        local_index = counts[bone_name]
        counts[bone_name] += 1
        primitive_type = str(
            getattr(bone_local, "primitive_type", "ellipsoid") or "ellipsoid")
        shape_exponents = getattr(bone_local, "shape_exponents", None)
        eps = (
            shape_exponents[i]
            if shape_exponents is not None
            else (1.0, 1.0)
        )
        entries.append({
            "id":             int(i),
            "name":           sphere_name(bone_name, local_index),
            "bone":           bone_name,
            "bone_index":     bi,
            "primitive_type": primitive_type,
            "shape_exponents": [round(float(v), 7) for v in eps],
            # offset from bone origin, expressed in bone's orientation frame
            "local_center":   [round(float(v), 7) for v in bone_local.local_centers[i]],
            # ellipsoid semi-axes (half-extents)
            "radii":          [round(float(v), 7) for v in bone_local.local_radii[i]],
            # orientation relative to bone frame — quaternion [x, y, z, w]
            "local_rotation": [round(float(v), 7) for v in bone_local.local_rotations[i]],
            **attachment_entry_fields(bone_local, i, skeleton),
        })

    primitive_type = str(
        getattr(bone_local, "primitive_type", "ellipsoid") or "ellipsoid")
    payload = {
        "version":               4,
        "coordinate_system":     "right_hand_y_up",
        "quaternion_convention": "xyzw",
        "primitive_type":        primitive_type,
        "count":                 len(entries),
        "ellipsoids":            entries,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Exporter] {len(entries)} {primitive_type} primitives -> {filepath}")
    return len(entries)
