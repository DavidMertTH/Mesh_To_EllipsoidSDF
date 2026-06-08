"""
pose_library.py — Save and load humanoid poses to/from disk.

A pose is stored as per-bone local 4×4 transforms keyed by *bone name*
(not index), so a pose authored on one FBX can be re-applied to any
skeleton that shares the same bone names (e.g. all Mixamo rigs).

On load, bone names are remapped to the current skeleton's bone indices;
bones present in the file but missing in the skeleton are silently skipped,
and bones missing from the file keep their bind-pose local transform.

Files are plain JSON in a `poses/` directory so they are human-readable
and diff-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from skeleton import Pose, Skeleton

POSE_FORMAT = "ellipsdf-pose"
POSE_VERSION = 1
DEFAULT_POSE_DIR = Path(__file__).parent / "poses"


def _safe_name(name: str) -> str:
    """Turn a pose name into a safe file stem."""
    keep = "-_. "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned or "pose"


def serialize_pose(pose: Pose, skeleton: Skeleton) -> dict:
    """Convert a Pose (index→4×4) into a name-keyed JSON-able dict."""
    bone_locals: dict[str, list] = {}
    for idx, mat in pose.bone_locals.items():
        if 0 <= idx < skeleton.num_bones:
            name = skeleton.bones[idx].name
            bone_locals[name] = np.asarray(mat, dtype=np.float64).tolist()
    return {
        "format": POSE_FORMAT,
        "version": POSE_VERSION,
        "name": pose.name,
        "bone_locals": bone_locals,
    }


def deserialize_pose(data: dict, skeleton: Skeleton,
                     name: str | None = None) -> Pose:
    """Rebuild a Pose for *skeleton* from a serialized dict.

    Bone names absent from the skeleton are skipped.
    """
    name_to_idx = {b.name: b.index for b in skeleton.bones}
    bone_locals: dict[int, np.ndarray] = {}
    for bone_name, mat in data.get("bone_locals", {}).items():
        idx = name_to_idx.get(bone_name)
        if idx is None:
            continue
        bone_locals[idx] = np.array(mat, dtype=np.float64).reshape(4, 4)
    return Pose(name=name or data.get("name", "pose"), bone_locals=bone_locals)


def save_pose(pose: Pose, skeleton: Skeleton, name: str,
              directory: str | Path = DEFAULT_POSE_DIR) -> Path:
    """Serialize *pose* to ``<directory>/<name>.json`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = serialize_pose(pose, skeleton)
    data["name"] = name
    path = directory / f"{_safe_name(name)}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_pose(path: str | Path, skeleton: Skeleton) -> Pose:
    """Load a single pose file and remap it onto *skeleton*."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return deserialize_pose(data, skeleton, name=data.get("name", path.stem))


def list_pose_files(directory: str | Path = DEFAULT_POSE_DIR) -> List[Path]:
    """Return all ``*.json`` pose files in *directory*, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def load_all_poses(skeleton: Skeleton,
                   directory: str | Path = DEFAULT_POSE_DIR) -> List[Pose]:
    """Load every pose file in *directory*, remapped onto *skeleton*.

    Files that fail to parse are skipped (a warning is printed).
    """
    poses: List[Pose] = []
    for p in list_pose_files(directory):
        try:
            poses.append(load_pose(p, skeleton))
        except Exception as e:  # noqa: BLE001 — keep the library usable
            print(f"[PoseLibrary] Failed to load {p.name}: {e}")
    return poses
