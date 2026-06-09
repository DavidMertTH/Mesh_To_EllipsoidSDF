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
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np

from skeleton import (
    Pose, Skeleton, mat4_compose, mat4_decompose,
    quat_inverse, quat_multiply,
)

POSE_FORMAT = "ellipsdf-pose"
ANIMATION_FORMAT = "ellipsdf-animation"
POSE_VERSION = 1
DEFAULT_POSE_DIR = Path(__file__).parent / "poses"
_REFERENCE_BIND_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _name_parts(name: str) -> tuple[str, list[str], str]:
    """Return raw base name, separator tokens, and compact searchable text."""
    raw = str(name).lower()
    if ":" in raw:
        raw = raw.split(":")[-1]
    tokens = [t for t in re.split(r"[\s()\[\]{}|/\\.\-_]+", raw) if t]
    compact = "".join(tokens)
    return raw, tokens, compact


def _name_tokens(name: str) -> list[str]:
    """Normalize common Mixamo/Unity/FBX bone names into searchable tokens."""
    s = _name_parts(name)[0]
    for ch in "()[]{}|/\\.-_ ":
        s = s.replace(ch, " ")

    # Split CamelCase-ish names after lowercasing by inserting breaks on common
    # humanoid words.  This keeps names like LeftUpperArm usable even when Unity
    # sends no separators.
    words = [
        "left", "right", "upper", "lower", "fore", "up", "arm", "leg", "hand",
        "foot", "toe", "base", "thumb", "index", "middle", "ring", "pinky",
        "little", "spine", "chest", "neck", "head", "hips", "hip", "pelvis",
        "shoulder", "clavicle", "thigh", "shin", "calf", "ankle", "ball",
    ]
    for w in words:
        s = s.replace(w, f" {w} ")
    return [t for t in s.split() if t and not t.startswith("mixamorig")]


def _name_side(name: str, toks: list[str]) -> str | None:
    """Detect left/right from names using words and common L/R suffixes."""
    raw, raw_tokens, compact = _name_parts(name)
    del raw

    all_tokens = set(toks) | set(raw_tokens)
    if any(t in all_tokens for t in ("left", "lf", "lft")):
        return "left"
    if any(t in all_tokens for t in ("right", "rt", "rgt")):
        return "right"
    if "l" in all_tokens or compact.startswith("l") or compact.endswith("l"):
        return "left"
    if "r" in all_tokens or compact.startswith("r") or compact.endswith("r"):
        return "right"
    return None


def _humanoid_slot(name: str) -> str | None:
    """Best-effort canonical humanoid slot for differently named rigs."""
    toks = _name_tokens(name)
    compact = "".join(toks)
    if not toks and not compact:
        return None

    # Terminal/end markers are display helpers, not deforming pose targets.  Keep
    # numbered finger phalanges, but do not let generic tip/end bones claim slots.
    is_terminal = "tip" in toks or "end" in toks
    has_number = any(t.isdigit() for t in toks)
    if is_terminal and not has_number:
        return None

    side = _name_side(name, toks)

    def has(*parts: str) -> bool:
        return any(p in toks or p in compact for p in parts)

    if has("hips", "hip", "pelvis"):
        return "hips"
    if has("head") and "top" not in toks and "end" not in toks:
        return "head"
    if has("neck"):
        return "neck"

    if has("spine", "chest"):
        nums = [int(t) for t in toks if t.isdigit()]
        n = nums[-1] if nums else 0
        if has("chest") or n >= 2:
            return "chest"
        if n == 1:
            return "upper_spine"
        return "spine"

    if side is None:
        return None

    prefix = f"{side}_"
    if has("shoulder", "clavicle"):
        return prefix + "shoulder"
    if has("hand") and not has("thumb", "index", "middle", "ring", "pinky", "little"):
        return prefix + "hand"
    if has("upper") and has("arm"):
        return prefix + "upper_arm"
    if has("fore", "lower") and has("arm"):
        return prefix + "lower_arm"
    if has("arm") and not has("fore", "lower"):
        return prefix + "upper_arm"

    if has("foot") and not has("toe"):
        return prefix + "foot"
    if has("toe", "ball"):
        return prefix + "toe"
    if (has("upper") or has("up")) and has("leg") or has("thigh"):
        return prefix + "upper_leg"
    if has("shin", "calf") or has("leg") and not has("upper", "up"):
        return prefix + "lower_leg"
    if has("ankle"):
        return prefix + "foot"

    finger = None
    if has("thumb"):
        finger = "thumb"
    elif has("index"):
        finger = "index"
    elif has("middle"):
        finger = "middle"
    elif has("ring"):
        finger = "ring"
    elif has("pinky", "little"):
        finger = "pinky"
    if finger is not None:
        nums = [int(t) for t in toks if t.isdigit()]
        seg = nums[-1] if nums else 1
        # Mixamo terminal "*4" bones are usually end bones, not animated phalanges.
        seg = max(1, min(3, seg))
        return f"{prefix}{finger}_{seg}"

    return None


def _reference_bind_locals_for_data(data: dict) -> dict[str, np.ndarray]:
    """Best-effort source bind locals for legacy Mixamo pose files."""
    explicit = data.get("humanoid_bind_locals")
    if isinstance(explicit, dict) and explicit:
        return {
            str(slot): np.array(mat, dtype=np.float64).reshape(4, 4)
            for slot, mat in explicit.items()
        }

    bone_names = tuple(str(n) for n in data.get("bone_locals", {}).keys())
    if not any("mixamorig" in n.lower() for n in bone_names):
        return {}

    cache_key = "mixamo"
    if cache_key in _REFERENCE_BIND_CACHE:
        return _REFERENCE_BIND_CACHE[cache_key]

    refs: dict[str, np.ndarray] = {}
    candidates = [
        Path(__file__).parent / "meshes" / "T-Pose.fbx",
        Path(__file__).parent / "meshes" / "Fight Idle.fbx",
        Path(__file__).parent / "meshes" / "Dancing.fbx",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            from rig_loader import load_rigged_mesh

            rig = load_rigged_mesh(path)
        except Exception:
            continue
        for bone in rig.skeleton.bones:
            refs[bone.name] = bone.local_bind_transform.astype(np.float64)
            slot = _humanoid_slot(bone.name)
            if slot is not None and slot not in refs:
                refs[slot] = bone.local_bind_transform.astype(np.float64)
        if refs:
            break

    _REFERENCE_BIND_CACHE[cache_key] = refs
    return refs


def _rotation_on_target_bind(
    source_local: np.ndarray,
    target_bind: np.ndarray,
    source_bind: np.ndarray | None = None,
) -> np.ndarray:
    """Retarget source animation delta while preserving target bone offsets."""
    _src_t, src_q, _src_s = mat4_decompose(source_local)
    tgt_t, tgt_q, tgt_s = mat4_decompose(target_bind)

    if source_bind is not None:
        _bind_t, bind_q, _bind_s = mat4_decompose(source_bind)
        delta_q = quat_multiply(quat_inverse(bind_q), src_q)
        src_q = quat_multiply(tgt_q, delta_q)

    return mat4_compose(tgt_t, src_q, tgt_s)


@dataclass
class PoseClip:
    """Mesh-independent pose/animation clip remapped to the current skeleton."""
    name: str
    poses: List[Pose]
    path: Path | None = None


def _safe_name(name: str) -> str:
    """Turn a pose name into a safe file stem."""
    keep = "-_. "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned or "pose"


def serialize_pose(pose: Pose, skeleton: Skeleton) -> dict:
    """Convert a Pose (index→4×4) into a name-keyed JSON-able dict."""
    bone_locals: dict[str, list] = {}
    humanoid_slots: dict[str, list] = {}
    humanoid_bind_locals: dict[str, list] = {}
    for idx, mat in pose.bone_locals.items():
        if 0 <= idx < skeleton.num_bones:
            name = skeleton.bones[idx].name
            arr = np.asarray(mat, dtype=np.float64)
            bone_locals[name] = arr.tolist()
            slot = _humanoid_slot(name)
            if slot is not None:
                humanoid_slots[slot] = arr.tolist()
                humanoid_bind_locals[slot] = np.asarray(
                    skeleton.bones[idx].local_bind_transform,
                    dtype=np.float64,
                ).tolist()
    return {
        "format": POSE_FORMAT,
        "version": POSE_VERSION,
        "name": pose.name,
        "bone_locals": bone_locals,
        "humanoid_slots": humanoid_slots,
        "humanoid_bind_locals": humanoid_bind_locals,
    }


def deserialize_pose(data: dict, skeleton: Skeleton,
                     name: str | None = None) -> Pose:
    """Rebuild a Pose for *skeleton* from a serialized dict.

    Exact bone names are used first.  When names differ, fall back to canonical
    humanoid slots inferred from both source and target names.  Slot-retargeted
    bones keep the target skeleton's local translation/scale and only borrow the
    source rotation, so applying a Mixamo/Unity pose does not pull the mesh back
    into the source rig's proportions.
    """
    name_to_idx = {b.name: b.index for b in skeleton.bones}
    slot_to_idx: dict[str, int] = {}
    for b in skeleton.bones:
        slot = _humanoid_slot(b.name)
        if slot is not None and slot not in slot_to_idx:
            slot_to_idx[slot] = b.index

    source_binds = _reference_bind_locals_for_data(data)
    bone_locals: dict[int, np.ndarray] = {}
    for bone_name, mat in data.get("bone_locals", {}).items():
        src = np.array(mat, dtype=np.float64).reshape(4, 4)
        idx = name_to_idx.get(bone_name)
        if idx is not None:
            bone_locals[idx] = src
            continue

        slot = _humanoid_slot(bone_name)
        idx = slot_to_idx.get(slot) if slot is not None else None
        if idx is not None and idx not in bone_locals:
            source_bind = source_binds.get(bone_name)
            if source_bind is None and slot is not None:
                source_bind = source_binds.get(slot)
            bone_locals[idx] = _rotation_on_target_bind(
                src, skeleton.bones[idx].local_bind_transform, source_bind)

    # Newer saved poses may carry explicit canonical slots.  Use them only for
    # target bones that were not already filled by exact/source-name remapping.
    for slot, mat in data.get("humanoid_slots", {}).items():
        idx = slot_to_idx.get(str(slot))
        if idx is None or idx in bone_locals:
            continue
        src = np.array(mat, dtype=np.float64).reshape(4, 4)
        source_bind = source_binds.get(str(slot))
        bone_locals[idx] = _rotation_on_target_bind(
            src, skeleton.bones[idx].local_bind_transform, source_bind)

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


def save_animation(poses: List[Pose], skeleton: Skeleton, name: str,
                   directory: str | Path = DEFAULT_POSE_DIR) -> Path:
    """Serialize a multi-frame animation clip independent of any mesh."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, pose in enumerate(poses):
        frame = serialize_pose(pose, skeleton)
        frame["name"] = pose.name or f"frame_{i}"
        frames.append(frame)
    data = {
        "format": ANIMATION_FORMAT,
        "version": POSE_VERSION,
        "name": name,
        "frames": frames,
    }
    path = directory / f"{_safe_name(name)}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_pose(path: str | Path, skeleton: Skeleton) -> Pose:
    """Load a single pose file and remap it onto *skeleton*."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return deserialize_pose(data, skeleton, name=data.get("name", path.stem))


def load_clip(path: str | Path, skeleton: Skeleton) -> PoseClip:
    """Load a single pose or animation file and remap it onto *skeleton*."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    fmt = data.get("format", POSE_FORMAT)
    if fmt == ANIMATION_FORMAT:
        name = data.get("name", path.stem)
        poses = [
            deserialize_pose(frame, skeleton, name=frame.get("name", f"{name}_{i}"))
            for i, frame in enumerate(data.get("frames", []))
        ]
        return PoseClip(name=name, poses=poses or [Pose.t_pose()], path=path)
    pose = deserialize_pose(data, skeleton, name=data.get("name", path.stem))
    return PoseClip(name=pose.name, poses=[pose], path=path)


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


def load_all_clips(skeleton: Skeleton,
                   directory: str | Path = DEFAULT_POSE_DIR) -> List[PoseClip]:
    """Load every pose/animation clip, remapped onto *skeleton*."""
    clips: List[PoseClip] = []
    for p in list_pose_files(directory):
        try:
            clips.append(load_clip(p, skeleton))
        except Exception as e:  # noqa: BLE001
            print(f"[PoseLibrary] Failed to load {p.name}: {e}")
    return clips
