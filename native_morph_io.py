"""Save/reopen native morph fits using the existing Unity corrective format."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from bone_ellipsoid_mapper import BoneLocalEllipsoids
from joint_dof import joint_rotation_frames
from pose_correctives import PoseCorrectiveKey, PoseCorrectiveLibrary
from skeleton import Pose, mat4_compose


# Where the native workflow stores a finished morph fit automatically.
DEFAULT_MORPH_FIT_DIR = Path(__file__).parent / "morph_fits"


def default_morph_fit_path(mesh_name, directory=None):
    stem = Path(str(mesh_name or "")).stem or "model"
    return Path(directory or DEFAULT_MORPH_FIT_DIR) / f"{stem}_morph_fit.json"


def _canonical_name(name):
    return str(name).rsplit(":", 1)[-1].casefold()


def _geometry_digest(rigged):
    digest = hashlib.sha256()
    for array, dtype in ((rigged.vertices, "<f4"), (rigged.faces, "<i4")):
        data = np.ascontiguousarray(array, dtype=dtype)
        digest.update(str(data.shape).encode("ascii"))
        digest.update(data.tobytes())
    return digest.hexdigest()


def _skinning_digest(rigged):
    """Hash bone identities, so harmless skeleton reordering still remaps."""
    digest = hashlib.sha256()
    joints = np.asarray(rigged.skin_joints)
    weights = np.ascontiguousarray(rigged.skin_weights, dtype="<f4")
    if (joints.shape != weights.shape or joints.ndim != 2
            or joints.shape[0] != len(rigged.vertices) or joints.dtype.kind not in "iu"
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Invalid model skinning")
    names = [_canonical_name(bone.name) for bone in rigged.skeleton.bones]
    canonical_joints = []
    for row_joints, row_weights in zip(joints, weights):
        row_names = []
        for joint, weight in zip(row_joints, row_weights):
            if weight > 0:
                if joint < 0 or joint >= len(names):
                    raise ValueError("Model skinning refers to a missing bone")
                row_names.append(names[int(joint)])
            else:
                row_names.append("")
        canonical_joints.append(row_names)
    digest.update(json.dumps(canonical_joints, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(str(weights.shape).encode("ascii"))
    digest.update(weights.tobytes())
    return digest.hexdigest()


def save_native_morph_fit(library, rigged, path):
    from native_morph import NativeMorphDriver, _unit_quaternions
    NativeMorphDriver(rigged.skeleton, library).evaluate(library.base_pose)
    normalized = replace(library,
        base=replace(library.base, local_rotations=_unit_quaternions(library.base.local_rotations)),
        keys=[replace(key, delta_rotations=_unit_quaternions(key.delta_rotations)) for key in library.keys])
    data = normalized.to_json(rigged.skeleton)
    # These optional fields are ignored by existing Unity JSON consumers. The
    # poses remain rotation descriptors; only a fitted shape is model-specific.
    data["native_model"] = {
        "mesh_name": rigged.mesh_name,
        "geometry_digest": _geometry_digest(rigged),
        "skinning_digest": _skinning_digest(rigged),
        "bind_locals": {b.name: np.asarray(b.local_bind_transform).tolist()
                        for b in rigged.skeleton.bones},
        "inverse_bind_matrices": {b.name: np.asarray(b.inverse_bind_matrix).tolist()
                                  for b in rigged.skeleton.bones},
        "parents": {b.name: (rigged.skeleton.bones[b.parent_index].name
                              if b.parent_index >= 0 else None)
                    for b in rigged.skeleton.bones},
    }
    # Validation and serialization happen before touching a previous saved fit.
    native_morph_from_json(rigged, data)
    serialized = json.dumps(data, indent=2, allow_nan=False)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent,
                                         prefix=f".{out.name}.", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return out


def load_native_morph_fit(rigged, path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return native_morph_from_json(rigged, data)


def native_morph_from_json(rigged, data):
    """Validate completely before returning a new library; never mutate a rig."""
    try:
        return _native_morph_from_json(rigged, data)
    except (KeyError, TypeError, OverflowError, IndexError) as exc:
        raise ValueError("Malformed native morph-fit data") from exc


def _native_morph_from_json(rigged, data):
    if not isinstance(data, dict) or data.get("format") != "ellipsdf-pose-correctives":
        raise ValueError("Select an EllipSDF pose-correctives / morph-fit JSON file")
    if data.get("quaternion_convention", "xyzw") != "xyzw":
        raise ValueError("Morph rotations must use xyzw quaternions")
    version = data.get("version", 4)
    if type(version) is not int or not 1 <= version <= 4:
        raise ValueError("Unsupported morph-fit version")
    skeleton = rigged.skeleton
    exact_names = {bone.name: i for i, bone in enumerate(skeleton.bones)}
    if len(exact_names) != skeleton.num_bones or not exact_names:
        raise ValueError("Morph fitting requires unique skeleton bone names")
    short_names = {}
    for name, index in exact_names.items():
        short_names.setdefault(name.rsplit(":", 1)[-1].casefold(), []).append(index)

    def bone_index(name):
        if not isinstance(name, str) or not name:
            raise ValueError("Morph fit bone names must be nonempty strings")
        if name in exact_names:
            return exact_names[name]
        matches = short_names.get(_canonical_name(name), [])
        if len(matches) != 1:
            raise ValueError(f"Morph fit bone is missing or ambiguous: {name}")
        return matches[0]

    def numeric(values, shape, label, *, float32=True):
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {label}") from exc
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {label}: expected finite values of shape {shape}")
        if float32 and np.any(np.abs(array) > np.finfo(np.float32).max):
            raise ValueError(f"Invalid {label}: values exceed the finite float32 range")
        return array

    def quaternions(values, count, label):
        q = numeric(values, (count, 4), label, float32=False)
        scale = np.max(np.abs(q), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError(f"Invalid {label}: zero quaternion")
        q /= scale
        return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)

    def integer(value, label):
        if type(value) is not int:
            raise ValueError(f"Invalid {label}: expected an integer")
        return value

    def object_list(value, label):
        if not isinstance(value, list) or any(not isinstance(e, dict) for e in value):
            raise ValueError(f"Invalid {label}: expected a list of objects")
        return value

    metadata = data.get("native_model", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid native model metadata")
    if metadata.get("geometry_digest") and metadata["geometry_digest"] != _geometry_digest(rigged):
        raise ValueError("This morph fit belongs to a different mesh")
    if metadata.get("skinning_digest") and metadata["skinning_digest"] != _skinning_digest(rigged):
        raise ValueError("This morph fit belongs to a model with different skinning")
    for field, attribute in (("bind_locals", "local_bind_transform"),
                             ("inverse_bind_matrices", "inverse_bind_matrix")):
        matrices = metadata.get(field)
        if matrices is None:
            continue
        if not isinstance(matrices, dict) or len(matrices) != skeleton.num_bones:
            raise ValueError(f"Incomplete morph fit {field}")
        seen = set()
        for name, matrix in matrices.items():
            index = bone_index(name)
            if index in seen:
                raise ValueError(f"Duplicate morph fit bind bone: {name}")
            seen.add(index)
            saved = numeric(matrix, (4, 4), "bind transform", float32=False)
            if not np.allclose(saved, getattr(skeleton.bones[index], attribute), rtol=1e-6, atol=1e-6):
                raise ValueError(f"Morph fit bind transform does not match: {name}")
    if "parents" in metadata:
        parents = metadata["parents"]
        if not isinstance(parents, dict) or len(parents) != skeleton.num_bones:
            raise ValueError("Incomplete morph fit hierarchy")
        seen = set()
        for name, parent in parents.items():
            index = bone_index(name)
            if index in seen:
                raise ValueError("Duplicate morph fit hierarchy bone")
            seen.add(index)
            if (bone_index(parent) if parent is not None else -1) != skeleton.bones[index].parent_index:
                raise ValueError(f"Morph fit hierarchy does not match: {name}")

    entries = object_list(data.get("base", []), "base population")
    count = len(entries)
    if count < 1 or integer(data.get("count", count), "primitive count") != count:
        raise ValueError("Morph fit contains no valid base population")

    def stable_entries(items):
        items = object_list(items, "primitive deltas")
        ids = [integer(e.get("id", i), "primitive ID") for i, e in enumerate(items)]
        if len(items) != count or sorted(ids) != list(range(count)):
            raise ValueError("Morph primitive IDs must be complete and unique")
        return sorted(enumerate(items), key=lambda pair: ids[pair[0]])

    entries = [e for _i, e in stable_entries(entries)]
    assignments = np.array([bone_index(e["bone"]) for e in entries], dtype=np.int32)
    radii = numeric([e["radii"] for e in entries], (count, 3), "base radii")
    if np.any(radii < np.finfo(np.float32).tiny):
        raise ValueError("Morph fit radii must be positive")
    primitive_type = data.get("primitive_type", "ellipsoid")
    if any(e.get("primitive_type", primitive_type) != primitive_type for e in entries):
        raise ValueError("Morph fit primitive types do not match")
    attachments = []
    source_indices = {}
    descriptor = data.get("base_pose") or next(
        (entry.get("pose") for entry in object_list(data.get("poses", []), "pose keys")
         if isinstance(entry.get("pose"), dict)), {})
    if isinstance(descriptor, dict):
        for ordinal, entry in enumerate(object_list(descriptor.get("bones", []), "pose bones")):
            source_index = integer(entry.get("index", ordinal), "pose bone index")
            if source_index in source_indices:
                raise ValueError("Duplicate pose bone index")
            source_indices[source_index] = bone_index(entry["name"])
    for index, entry in enumerate(entries):
        names = entry.get("attachment_bones")
        joints = entry.get("attachment_bone_indices")
        weights = entry.get("attachment_weights")
        if names is not None:
            if not isinstance(names, list):
                raise ValueError("Morph attachment bone names must be a list")
            joints = [bone_index(name) for name in names]
        if joints is None and weights is None:
            attachments.append(([int(assignments[index])], [1.0]))
            continue
        if joints is None or weights is None or len(joints) != len(weights) or not len(joints):
            raise ValueError("Morph attachment arrays do not match")
        if names is None:
            joints = [integer(joint, "attachment bone index") for joint in joints]
            if any(joint not in source_indices for joint in joints):
                raise ValueError("Morph attachment bone index has no named descriptor")
            joints = [source_indices[joint] for joint in joints]
        j = np.asarray(joints, dtype=np.int64)
        w = numeric(weights, (len(j),), "attachment weights")
        if (j.ndim != 1 or np.any(j < 0) or np.any(j >= skeleton.num_bones)
                or np.any(w < 0) or w.sum() <= 0 or not np.isfinite(w.sum())):
            raise ValueError("Invalid morph skin attachments")
        attachments.append((j, w / w.sum()))
    stride = max(len(j) for j, _w in attachments)
    attachment_joints = np.repeat(assignments[:, None], stride, axis=1)
    attachment_weights = np.zeros((count, stride), dtype=np.float32)
    for index, (joints, weights) in enumerate(attachments):
        attachment_joints[index, :len(joints)] = joints
        attachment_weights[index, :len(weights)] = weights
    base = BoneLocalEllipsoids(
        local_centers=numeric([e["local_center"] for e in entries], (count, 3), "base centers").astype(np.float32),
        local_radii=radii.astype(np.float32),
        local_rotations=quaternions([e["local_rotation"] for e in entries], count, "base rotations"),
        bone_assignments=assignments, attachment_joints=attachment_joints,
        attachment_weights=attachment_weights, primitive_type=primitive_type,
        shape_exponents=numeric([e.get("shape_exponents", [1, 1]) for e in entries], (count, 2), "shape exponents").astype(np.float32))

    def pose_descriptor(descriptor, name):
        if not isinstance(descriptor, dict) or descriptor.get("descriptor") != "bone_local_rotations":
            raise ValueError(f"Morph key {name!r} lacks a bone-rotation descriptor")
        locals_, seen, source_indices = {}, set(), {}
        bones = object_list(descriptor.get("bones", []), "pose bones")
        for ordinal, entry in enumerate(bones):
            index = bone_index(entry["name"])
            if index in seen:
                raise ValueError(f"Duplicate pose bone: {entry['name']}")
            seen.add(index)
            source_index = integer(entry.get("index", ordinal), "pose bone index")
            if source_index < 0 or source_index in source_indices:
                raise ValueError("Duplicate or negative pose bone index")
            source_indices[source_index] = index
            q = quaternions([entry["local_rotation"]], 1, "pose rotation")[0]
            bind = np.asarray(skeleton.bones[index].local_bind_transform, dtype=np.float64)
            frame = joint_rotation_frames(bind[None])[0].astype(np.float64)
            local = bind.copy()
            local[:3, :3] = mat4_compose(np.zeros(3), q, np.ones(3))[:3, :3] @ (frame.T @ bind[:3, :3])
            locals_[index] = local
        if len(seen) != skeleton.num_bones:
            raise ValueError(f"Morph key {name!r} does not describe the complete loaded skeleton")
        for entry in bones:
            if "parent_index" in entry:
                parent = integer(entry["parent_index"], "pose parent index")
                if parent < -1 or (parent >= 0 and parent not in source_indices):
                    raise ValueError("Pose descriptor refers to a missing parent")
                current_parent = source_indices[parent] if parent >= 0 else -1
                if current_parent != skeleton.bones[bone_index(entry["name"])].parent_index:
                    raise ValueError("Pose descriptor hierarchy does not match the loaded model")
        return Pose(name=name, bone_locals=locals_)

    keys = []
    for entry in object_list(data.get("poses", []), "pose keys"):
        name = str(entry.get("name", f"Pose {len(keys) + 1}"))
        if entry.get("ellipsoids") is not None:
            deltas = [e for _i, e in stable_entries(entry["ellipsoids"])]
            dc = [e["delta_local_center"] for e in deltas]
            dq = [e["delta_local_rotation"] for e in deltas]
            dr = [e["delta_log_radii"] for e in deltas]
            de = [e.get("delta_log_shape_exponents", [0, 0]) for e in deltas]
        else:
            dc, dq, dr = entry["delta_centers"], entry["delta_rotations"], entry["delta_log_radii"]
            de = entry.get("delta_log_shape_exponents", np.zeros((count, 2)))
        loss = float(entry.get("loss", 0))
        if not np.isfinite(loss):
            raise ValueError("Morph loss must be finite")
        keys.append(PoseCorrectiveKey(name=name,
            delta_centers=numeric(dc, (count, 3), "center deltas").astype(np.float32),
            delta_rotations=quaternions(dq, count, "rotation deltas"),
            delta_log_radii=numeric(dr, (count, 3), "radius deltas").astype(np.float32),
            delta_log_shape_exponents=numeric(de, (count, 2), "exponent deltas").astype(np.float32),
            loss=loss, pose=pose_descriptor(entry.get("pose"), name)))
    if not keys:
        raise ValueError("Morph fit has no trained pose keys")
    base_pose = (pose_descriptor(data["base_pose"], "Morph base")
                 if data.get("base_pose") is not None else Pose.t_pose())
    library = PoseCorrectiveLibrary(base=base, keys=keys, base_pose=base_pose)
    from native_morph import NativeMorphDriver
    NativeMorphDriver(skeleton, library).evaluate(base_pose)
    return library
