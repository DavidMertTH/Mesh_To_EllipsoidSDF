"""
rig_loader.py — Load rigged meshes from glTF / GLB / FBX files.

Extracts:
  - Triangle mesh (vertices, faces)
  - Skeleton (bone hierarchy + bind poses)
  - Skin weights (sparse vertex → bone weights)
  - Animation poses (keyframe data → list of Pose objects)

Uses trimesh for mesh geometry and parses the glTF JSON / FBX binary
structure directly for skeleton / skin / animation data.

Supported formats: .gltf, .glb, .fbx
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh

from skeleton import Bone, Pose, Skeleton, mat4_compose

# ── Data container ───────────────────────────────────────────────────────────

@dataclass
class RiggedMesh:
    """All data extracted from a rigged mesh file."""
    vertices: np.ndarray              # (V, 3) float32 — bind-pose positions
    faces: np.ndarray                 # (F, 3) int32
    skeleton: Skeleton
    skin_weights: np.ndarray          # (V, max_influences) float32
    skin_joints: np.ndarray           # (V, max_influences) int32 — bone indices
    poses: List[Pose] = field(default_factory=list)
    mesh_name: str = ""


# ── glTF accessor helpers ────────────────────────────────────────────────────

_COMPONENT_TYPE_SIZE = {
    5120: 1,   # BYTE
    5121: 1,   # UNSIGNED_BYTE
    5122: 2,   # SHORT
    5123: 2,   # UNSIGNED_SHORT
    5125: 4,   # UNSIGNED_INT
    5126: 4,   # FLOAT
}

_COMPONENT_TYPE_FORMAT = {
    5120: 'b',   # BYTE
    5121: 'B',   # UNSIGNED_BYTE
    5122: 'h',   # SHORT
    5123: 'H',   # UNSIGNED_SHORT
    5125: 'I',   # UNSIGNED_INT
    5126: 'f',   # FLOAT
}

_TYPE_NUM_COMPONENTS = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16,
}

_COMPONENT_TYPE_NUMPY = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}


class _GltfBuffers:
    """Manages buffer data from a glTF/GLB file."""

    def __init__(self, gltf_json: dict, binary_chunk: bytes | None = None,
                 base_path: Path | None = None):
        self._json = gltf_json
        self._binary_chunk = binary_chunk
        self._base_path = base_path
        self._buffer_cache: dict[int, bytes] = {}

    def _get_buffer(self, buf_index: int) -> bytes:
        if buf_index in self._buffer_cache:
            return self._buffer_cache[buf_index]

        buf_def = self._json['buffers'][buf_index]
        uri = buf_def.get('uri', '')

        if not uri and self._binary_chunk is not None:
            # GLB binary chunk
            data = self._binary_chunk
        elif uri.startswith('data:'):
            # Data URI (base64)
            import base64
            _, b64 = uri.split(',', 1)
            data = base64.b64decode(b64)
        elif self._base_path is not None:
            data = (self._base_path / uri).read_bytes()
        else:
            raise ValueError(f"Cannot resolve buffer URI: {uri!r}")

        self._buffer_cache[buf_index] = data
        return data

    def read_accessor(self, acc_index: int) -> np.ndarray:
        """Read a glTF accessor into a numpy array."""
        acc = self._json['accessors'][acc_index]
        bv_index = acc.get('bufferView')
        count = acc['count']
        comp_type = acc['componentType']
        acc_type = acc['type']

        num_comp = _TYPE_NUM_COMPONENTS[acc_type]
        np_dtype = _COMPONENT_TYPE_NUMPY[comp_type]

        if bv_index is None:
            # Sparse or empty accessor
            return np.zeros((count, num_comp), dtype=np_dtype).squeeze()

        bv = self._json['bufferViews'][bv_index]
        buf_data = self._get_buffer(bv['buffer'])

        byte_offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        byte_stride = bv.get('byteStride', 0)

        comp_size = _COMPONENT_TYPE_SIZE[comp_type]
        elem_size = comp_size * num_comp

        if byte_stride == 0 or byte_stride == elem_size:
            # Tightly packed
            total_bytes = count * elem_size
            raw = buf_data[byte_offset:byte_offset + total_bytes]
            arr = np.frombuffer(raw, dtype=np_dtype).copy()
            if num_comp > 1:
                arr = arr.reshape(count, num_comp)
        else:
            # Strided
            arr = np.empty((count, num_comp), dtype=np_dtype)
            for i in range(count):
                off = byte_offset + i * byte_stride
                row = np.frombuffer(
                    buf_data[off:off + elem_size], dtype=np_dtype,
                )
                arr[i] = row

        return arr


# ── glTF parsing ─────────────────────────────────────────────────────────────

def _load_gltf_json(path: Path) -> tuple[dict, bytes | None, Path]:
    """Load glTF JSON and optional binary chunk."""
    suffix = path.suffix.lower()

    if suffix == '.glb':
        with open(path, 'rb') as f:
            # GLB header
            magic, version, length = struct.unpack('<III', f.read(12))
            assert magic == 0x46546C67, "Not a valid GLB file"

            # Chunk 0: JSON
            chunk_len, chunk_type = struct.unpack('<II', f.read(8))
            assert chunk_type == 0x4E4F534A, "First GLB chunk must be JSON"
            json_bytes = f.read(chunk_len)
            gltf = json.loads(json_bytes)

            # Chunk 1: Binary (optional)
            binary_chunk = None
            if f.tell() < length:
                chunk_len, chunk_type = struct.unpack('<II', f.read(8))
                if chunk_type == 0x004E4942:
                    binary_chunk = f.read(chunk_len)

        return gltf, binary_chunk, path.parent

    elif suffix == '.gltf':
        with open(path) as f:
            gltf = json.load(f)
        return gltf, None, path.parent

    else:
        raise ValueError(f"Unsupported format: {suffix} (expected .gltf or .glb)")


def _parse_skeleton(gltf: dict, buffers: _GltfBuffers,
                    skin_index: int = 0) -> Skeleton:
    """Extract a Skeleton from glTF skin data."""
    skin = gltf['skins'][skin_index]
    joint_indices = skin['joints']  # node indices that are bones
    num_bones = len(joint_indices)

    # Map glTF node index → bone index
    node_to_bone = {node_idx: bone_idx
                    for bone_idx, node_idx in enumerate(joint_indices)}

    # Inverse bind matrices
    ibm_acc = skin.get('inverseBindMatrices')
    if ibm_acc is not None:
        ibm_data = buffers.read_accessor(ibm_acc)  # (N, 16)
        ibm_data = ibm_data.reshape(num_bones, 4, 4).astype(np.float64)
    else:
        ibm_data = np.tile(np.eye(4, dtype=np.float64), (num_bones, 1, 1))

    nodes = gltf['nodes']
    bones: List[Bone] = []

    for bone_idx, node_idx in enumerate(joint_indices):
        node = nodes[node_idx]
        name = node.get('name', f'bone_{bone_idx}')

        # Local bind transform
        if 'matrix' in node:
            local_mat = np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
        else:
            t = np.array(node.get('translation', [0, 0, 0]), dtype=np.float64)
            r = np.array(node.get('rotation', [0, 0, 0, 1]), dtype=np.float64)
            s = np.array(node.get('scale', [1, 1, 1]), dtype=np.float64)
            local_mat = mat4_compose(t, r, s)

        # Parent: find if any joint node lists this node as a child
        parent_bone_idx = -1
        for other_node_idx in joint_indices:
            other_node = nodes[other_node_idx]
            children_of_other = other_node.get('children', [])
            if node_idx in children_of_other:
                parent_bone_idx = node_to_bone.get(other_node_idx, -1)
                break

        bones.append(Bone(
            name=name,
            index=bone_idx,
            parent_index=parent_bone_idx,
            local_bind_transform=local_mat,
            inverse_bind_matrix=ibm_data[bone_idx],
        ))

    return Skeleton(bones)


def _parse_skin_weights(gltf: dict, buffers: _GltfBuffers,
                        mesh_index: int = 0,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Extract skin joints and weights from a glTF mesh primitive.

    Returns
    -------
    joints : (V, 4) int32  — bone indices per vertex
    weights : (V, 4) float32 — corresponding weights
    """
    mesh = gltf['meshes'][mesh_index]
    # Use first primitive that has JOINTS_0
    for prim in mesh['primitives']:
        attrs = prim['attributes']
        if 'JOINTS_0' in attrs:
            joints = buffers.read_accessor(attrs['JOINTS_0'])   # (V, 4)
            weights = buffers.read_accessor(attrs['WEIGHTS_0'])  # (V, 4)
            return joints.astype(np.int32), weights.astype(np.float32)

    raise ValueError("No skin weights (JOINTS_0/WEIGHTS_0) found in mesh.")


def _parse_animations(gltf: dict, buffers: _GltfBuffers,
                      skeleton: Skeleton,
                      joint_indices: list[int],
                      sample_count: int = 10,
                      ) -> List[Pose]:
    """Extract sampled poses from glTF animations.

    Uniformly samples `sample_count` frames from the first animation.
    """
    if 'animations' not in gltf or len(gltf['animations']) == 0:
        return []

    nodes = gltf['nodes']
    node_to_bone = {node_idx: bone_idx
                    for bone_idx, node_idx in enumerate(joint_indices)}

    anim = gltf['animations'][0]
    channels = anim['channels']
    samplers = anim['samplers']

    # Collect all channels per target node
    # channel: {sampler, target: {node, path}}
    # path: "translation", "rotation", "scale"
    bone_channels: dict[int, dict[str, dict]] = {}

    for ch in channels:
        target = ch['target']
        node_idx = target.get('node')
        if node_idx is None or node_idx not in node_to_bone:
            continue
        bone_idx = node_to_bone[node_idx]
        path = target['path']

        sampler = samplers[ch['sampler']]
        times = buffers.read_accessor(sampler['input']).astype(np.float64)
        values = buffers.read_accessor(sampler['output']).astype(np.float64)

        if bone_idx not in bone_channels:
            bone_channels[bone_idx] = {}

        bone_channels[bone_idx][path] = {
            'times': times,
            'values': values,
            'interpolation': sampler.get('interpolation', 'LINEAR'),
        }

    if not bone_channels:
        return []

    # Find global time range
    all_times = []
    for bc in bone_channels.values():
        for ch_data in bc.values():
            all_times.append(ch_data['times'][-1])
    t_max = max(all_times) if all_times else 0.0

    if t_max <= 0:
        return []

    # Sample uniformly
    sample_times = np.linspace(0, t_max, sample_count, endpoint=True)
    poses: List[Pose] = []

    for si, t in enumerate(sample_times):
        pose = Pose(name=f"frame_{si}")

        for bone_idx, ch_dict in bone_channels.items():
            bone = skeleton.bones[bone_idx]
            # Start from bind pose
            t_bind, r_bind, s_bind = np.array([0, 0, 0], dtype=np.float64), \
                np.array([0, 0, 0, 1], dtype=np.float64), \
                np.array([1, 1, 1], dtype=np.float64)

            # Get bind pose defaults from node
            node = nodes[joint_indices[bone_idx]]
            if 'translation' in node:
                t_bind = np.array(node['translation'], dtype=np.float64)
            if 'rotation' in node:
                r_bind = np.array(node['rotation'], dtype=np.float64)
            if 'scale' in node:
                s_bind = np.array(node['scale'], dtype=np.float64)

            trans = _interp_at(ch_dict.get('translation'), t, t_bind)
            rot = _interp_at(ch_dict.get('rotation'), t, r_bind, is_quat=True)
            scale = _interp_at(ch_dict.get('scale'), t, s_bind)

            local_mat = mat4_compose(trans, rot, scale)
            pose.bone_locals[bone_idx] = local_mat

        poses.append(pose)

    return poses


def _interp_at(ch_data: dict | None, t: float, default: np.ndarray,
               is_quat: bool = False) -> np.ndarray:
    """Linear interpolation of a channel at time t."""
    if ch_data is None:
        return default.copy()

    times = ch_data['times']
    values = ch_data['values']

    if t <= times[0]:
        return values[0].copy()
    if t >= times[-1]:
        return values[-1].copy()

    # Find bracketing indices
    idx = np.searchsorted(times, t, side='right') - 1
    idx = max(0, min(idx, len(times) - 2))

    t0, t1 = times[idx], times[idx + 1]
    dt = t1 - t0
    alpha = (t - t0) / dt if dt > 1e-12 else 0.0

    v0 = values[idx]
    v1 = values[idx + 1]

    if is_quat:
        return _slerp(v0, v1, alpha)
    else:
        return v0 * (1.0 - alpha) + v1 * alpha


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation for quaternions."""
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)

    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-12:
        return q0.copy()

    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    result = a * q0 + b * q1
    return result / np.linalg.norm(result)


# ── Public API ───────────────────────────────────────────────────────────────

def load_rigged_mesh(
    path: str | Path,
    target_scale: float = 1.0,
    animation_samples: int = 10,
) -> RiggedMesh:
    """Load a rigged mesh from a glTF/GLB/FBX file.

    Parameters
    ----------
    path : str or Path
        Path to .gltf, .glb, or .fbx file.
    target_scale : float
        Scale the mesh to fit in [-target_scale, target_scale].
    animation_samples : int
        Number of frames to sample from animations.

    Returns
    -------
    RiggedMesh
        Complete rigged mesh with skeleton, skin weights, and poses.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.fbx':
        return _load_fbx_rigged(path, target_scale, animation_samples)
    elif suffix in ('.gltf', '.glb'):
        return _load_gltf_rigged(path, target_scale, animation_samples)
    else:
        raise ValueError(
            f"Unsupported rig format: {suffix}. "
            "Supported: .fbx, .gltf, .glb"
        )


def is_rigged_mesh(path: str | Path) -> bool:
    """Quick check whether a file likely contains a rigged mesh."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.fbx':
        try:
            from fbx_parser import extract_rig_data
            rig = extract_rig_data(path)
            return len(rig.bones) > 0 and len(rig.skin_clusters) > 0
        except Exception:
            return False

    if suffix in ('.gltf', '.glb'):
        try:
            gltf, _, _ = _load_gltf_json(path)
            return 'skins' in gltf and len(gltf['skins']) > 0
        except Exception:
            return False

    return False


# ── glTF loading path ────────────────────────────────────────────────────────

def _load_gltf_rigged(
    path: Path,
    target_scale: float = 1.0,
    animation_samples: int = 10,
) -> RiggedMesh:
    """Load a rigged mesh from glTF/GLB."""
    gltf, binary_chunk, base_path = _load_gltf_json(path)
    buffers = _GltfBuffers(gltf, binary_chunk, base_path)

    if 'skins' not in gltf or len(gltf['skins']) == 0:
        raise ValueError(f"No skeleton/skin found in {path.name}. "
                         "Use the standard mesh loader for static meshes.")

    # ── Skeleton ──
    skeleton = _parse_skeleton(gltf, buffers, skin_index=0)
    joint_indices = gltf['skins'][0]['joints']

    # ── Mesh ──
    from mesh_io import as_trimesh_scene, scene_to_single_mesh, normalize_mesh
    scene = as_trimesh_scene(str(path))
    mesh = scene_to_single_mesh(scene)

    # ── Skin weights ──
    try:
        joints, weights = _parse_skin_weights(gltf, buffers, mesh_index=0)
    except ValueError:
        nv = len(mesh.vertices)
        joints = np.zeros((nv, 4), dtype=np.int32)
        weights = np.zeros((nv, 4), dtype=np.float32)
        weights[:, 0] = 1.0

    # Normalize weights
    w_sum = weights.sum(axis=1, keepdims=True)
    w_sum = np.maximum(w_sum, 1e-8)
    weights = weights / w_sum

    # Vertex count mismatch handling
    nv_mesh = len(mesh.vertices)
    nv_skin = len(joints)
    if nv_mesh != nv_skin:
        if nv_mesh < nv_skin:
            joints = joints[:nv_mesh]
            weights = weights[:nv_mesh]
        else:
            pad_j = np.zeros((nv_mesh - nv_skin, 4), dtype=np.int32)
            pad_w = np.zeros((nv_mesh - nv_skin, 4), dtype=np.float32)
            pad_w[:, 0] = 1.0
            joints = np.concatenate([joints, pad_j])
            weights = np.concatenate([weights, pad_w])

    mesh = normalize_mesh(mesh, target_scale)

    # ── Animations ──
    poses = _parse_animations(
        gltf, buffers, skeleton, joint_indices,
        sample_count=animation_samples,
    )
    poses.insert(0, Pose.t_pose())

    return RiggedMesh(
        vertices=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.int32),
        skeleton=skeleton,
        skin_weights=weights,
        skin_joints=joints,
        poses=poses,
        mesh_name=path.stem,
    )


# ── FBX loading path ────────────────────────────────────────────────────────

def _normalize_mesh_np(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Center at origin and scale to [-target_scale, target_scale].

    Pure numpy — no trimesh dependency.

    Returns (vertices, faces, center, scale_factor) so the same transform
    can be applied to bone world-space matrices.
    """
    v = vertices.astype(np.float32)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center = (vmin + vmax) * 0.5
    v = v - center
    extent = float((v.max(axis=0) - v.min(axis=0)).max())
    scale_factor = (2.0 * target_scale / extent) if extent > 0 else 1.0
    v = v * scale_factor
    return v.astype(np.float32), faces.astype(np.int32), center, scale_factor


def _load_fbx_rigged(
    path: Path,
    target_scale: float = 1.0,
    animation_samples: int = 10,
) -> RiggedMesh:
    """Load a rigged mesh from an FBX file.

    Uses the built-in FBX binary parser — no trimesh/assimp needed.
    """
    from fbx_parser import (
        extract_rig_data, euler_to_quat_xyzw, euler_to_matrix,
    )

    rig = extract_rig_data(path)

    if len(rig.bones) == 0:
        raise ValueError(f"No bones found in {path.name}.")

    # ── Mesh geometry — from FBX parser directly ──
    if rig.mesh is None or len(rig.mesh.vertices) == 0:
        raise ValueError(
            f"Could not extract mesh geometry from {path.name}. "
            f"The FBX file may use an unsupported structure or be ASCII FBX. "
            f"Try re-exporting as binary FBX from Blender/Mixamo."
        )

    vertices = rig.mesh.vertices.copy()
    faces = rig.mesh.faces.copy()
    nv = len(vertices)

    # ── Normalize mesh (pure numpy, no trimesh) ──
    vertices, faces, mesh_center, mesh_scale = _normalize_mesh_np(
        vertices, faces, target_scale,
    )

    # Build the 4×4 normalization matrix: scale * (v - center)
    norm_mat = np.eye(4, dtype=np.float64)
    norm_mat[0, 0] = mesh_scale
    norm_mat[1, 1] = mesh_scale
    norm_mat[2, 2] = mesh_scale
    norm_mat[:3, 3] = -mesh_center.astype(np.float64) * mesh_scale

    # ── Build Skeleton from FBX bone data ──
    #
    # KEY INSIGHT: The TransformLink from skin clusters is the ground-truth
    # bind-pose world transform. We derive local transforms FROM it,
    # NOT from the Euler angles (which are unreliable with pre-rotation,
    # rotation order variations, etc.).
    #
    # We apply the mesh normalization transform to all bone world matrices
    # so bones and vertices live in the same coordinate space.

    fbx_id_to_idx: dict[int, int] = {}
    bones: List[Bone] = []

    for bone_idx, fb in enumerate(rig.bones):
        fbx_id_to_idx[fb.fbx_id] = bone_idx

    # Collect TransformLink (world bind pose) per bone — apply normalization
    bind_world: dict[int, np.ndarray] = {}  # bone_idx → 4×4
    for sc in rig.skin_clusters:
        bi = fbx_id_to_idx.get(sc.bone_fbx_id, -1)
        if bi >= 0:
            bind_world[bi] = norm_mat @ sc.transform_link.astype(np.float64)

    # For bones without a skin cluster, compute world bind from Euler chain
    # (forward kinematics using Euler-based locals as fallback)
    euler_locals: dict[int, np.ndarray] = {}
    for bone_idx, fb in enumerate(rig.bones):
        q_rot = euler_to_quat_xyzw(fb.local_rotation)
        if np.any(np.abs(fb.pre_rotation) > 1e-6):
            R_pre = euler_to_matrix(fb.pre_rotation)
            R_anim = euler_to_matrix(fb.local_rotation)
            R_combined = R_pre @ R_anim
            local_mat = np.eye(4, dtype=np.float64)
            local_mat[:3, :3] = R_combined * fb.local_scaling[np.newaxis, :]
            local_mat[:3, 3] = fb.local_translation
        else:
            local_mat = mat4_compose(
                fb.local_translation, q_rot, fb.local_scaling,
            )
        euler_locals[bone_idx] = local_mat

    # Forward-pass: fill in missing bind_world from Euler chain
    for bone_idx, fb in enumerate(rig.bones):
        if bone_idx in bind_world:
            continue
        parent_idx = fbx_id_to_idx.get(fb.parent_id, -1)
        if parent_idx >= 0 and parent_idx in bind_world:
            bind_world[bone_idx] = bind_world[parent_idx] @ euler_locals[bone_idx]
        else:
            bind_world[bone_idx] = norm_mat @ euler_locals[bone_idx]

    # Derive local_bind_transform = inv(parent_world) @ my_world
    for bone_idx, fb in enumerate(rig.bones):
        parent_idx = fbx_id_to_idx.get(fb.parent_id, -1)

        my_world = bind_world.get(bone_idx, np.eye(4, dtype=np.float64))

        if parent_idx >= 0 and parent_idx in bind_world:
            parent_world = bind_world[parent_idx]
            local_mat = np.linalg.inv(parent_world) @ my_world
        else:
            local_mat = my_world.copy()

        inv_bind = np.linalg.inv(my_world)

        bones.append(Bone(
            name=fb.name,
            index=bone_idx,
            parent_index=parent_idx,
            local_bind_transform=local_mat,
            inverse_bind_matrix=inv_bind,
        ))

    skeleton = Skeleton(bones)

    # Store pre-rotations for animation parsing
    bone_pre_rotations: dict[int, np.ndarray] = {}
    for bone_idx, fb in enumerate(rig.bones):
        if np.any(np.abs(fb.pre_rotation) > 1e-6):
            bone_pre_rotations[bone_idx] = fb.pre_rotation.copy()

    # ── Build skin weights from FBX clusters ──
    max_influences = 4
    vert_influences: dict[int, list[tuple[int, float]]] = {}

    for sc in rig.skin_clusters:
        bone_idx = fbx_id_to_idx.get(sc.bone_fbx_id, -1)
        if bone_idx < 0:
            continue
        for vi, w in zip(sc.indices, sc.weights):
            vi = int(vi)
            if vi >= nv:
                continue
            if vi not in vert_influences:
                vert_influences[vi] = []
            vert_influences[vi].append((bone_idx, float(w)))

    # Pack into (V, 4) arrays
    skin_joints = np.zeros((nv, max_influences), dtype=np.int32)
    skin_weights = np.zeros((nv, max_influences), dtype=np.float32)

    for vi, influences in vert_influences.items():
        influences.sort(key=lambda x: -x[1])
        for k in range(min(len(influences), max_influences)):
            skin_joints[vi, k] = influences[k][0]
            skin_weights[vi, k] = influences[k][1]

    # Normalize weights
    w_sum = skin_weights.sum(axis=1, keepdims=True)
    w_sum = np.maximum(w_sum, 1e-8)
    skin_weights = skin_weights / w_sum

    # Vertices without any influence → bind to root
    unbound = w_sum.ravel() < 1e-6
    skin_weights[unbound, 0] = 1.0

    # ── Parse animations into Poses ──
    poses = _parse_fbx_animations(
        rig, skeleton, fbx_id_to_idx, animation_samples,
        bone_pre_rotations,
        mesh_center=mesh_center, mesh_scale=mesh_scale,
        euler_locals=euler_locals,
    )
    poses.insert(0, Pose.t_pose())

    return RiggedMesh(
        vertices=vertices,
        faces=faces,
        skeleton=skeleton,
        skin_weights=skin_weights,
        skin_joints=skin_joints,
        poses=poses,
        mesh_name=path.stem,
    )


def _parse_fbx_animations(
    rig,
    skeleton: Skeleton,
    fbx_id_to_idx: dict[int, int],
    sample_count: int,
    bone_pre_rotations: dict[int, np.ndarray] | None = None,
    mesh_center: np.ndarray | None = None,
    mesh_scale: float = 1.0,
    euler_locals: dict[int, np.ndarray] | None = None,
) -> List[Pose]:
    """Convert FBX animation curves into sampled Poses.

    FBX animated local transform = T * R_pre * R_anim * S
    where R_pre is constant (from PreRotation property).

    mesh_center / mesh_scale: normalization applied to mesh vertices.
    Root bones need norm_mat applied; child bone locals are in original
    FBX space (norm cancels out in inv(parent) @ child).

    euler_locals: original FBX local transforms per bone index (before
    normalization), used as defaults for animation interpolation.
    """
    from fbx_parser import euler_to_quat_xyzw, euler_to_matrix

    def _matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
        """3×3 rotation matrix → Euler XYZ degrees (inverse of euler_to_matrix)."""
        # euler_to_matrix builds R = Rz @ Ry @ Rx
        sy = -R[2, 0]
        sy = np.clip(sy, -1.0, 1.0)
        y = np.arcsin(sy)
        cy = np.cos(y)
        if abs(cy) > 1e-6:
            x = np.arctan2(R[2, 1] / cy, R[2, 2] / cy)
            z = np.arctan2(R[1, 0] / cy, R[0, 0] / cy)
        else:
            # Gimbal lock
            x = np.arctan2(-R[1, 2], R[1, 1])
            z = 0.0
        return np.degrees(np.array([x, y, z], dtype=np.float64))

    if not rig.anim_curves:
        return []

    if bone_pre_rotations is None:
        bone_pre_rotations = {}

    # Group curves by bone
    bone_curves: dict[int, dict[str, dict[int, tuple]]] = {}

    for ac in rig.anim_curves:
        if ac.bone_fbx_id not in bone_curves:
            bone_curves[ac.bone_fbx_id] = {}
        if ac.channel not in bone_curves[ac.bone_fbx_id]:
            bone_curves[ac.bone_fbx_id][ac.channel] = {}
        bone_curves[ac.bone_fbx_id][ac.channel][ac.axis] = (ac.times, ac.values)

    # Find global time range
    t_max = 0.0
    for bc in bone_curves.values():
        for ch in bc.values():
            for _, (times, _) in ch.items():
                if len(times) > 0:
                    t_max = max(t_max, times[-1])

    if t_max <= 0:
        return []

    sample_times = np.linspace(0, t_max, sample_count, endpoint=True)
    poses: List[Pose] = []

    for si, t in enumerate(sample_times):
        pose = Pose(name=f"frame_{si}")

        for bone_fbx_id, channels in bone_curves.items():
            bone_idx = fbx_id_to_idx.get(bone_fbx_id, -1)
            if bone_idx < 0:
                continue

            bone = skeleton.bones[bone_idx]

            # Start from ORIGINAL FBX local transform defaults
            # (not the normalized bind-pose, which has mesh_scale baked
            # into root bones and would cause double-scaling).
            from skeleton import mat4_decompose
            if euler_locals and bone_idx in euler_locals:
                ref_mat = euler_locals[bone_idx]
            else:
                ref_mat = bone.local_bind_transform
            t_ref, _, s_ref = mat4_decompose(ref_mat)
            trans = t_ref.copy()
            scale = s_ref.copy()

            # Extract R_anim Euler angles from the reference (undo pre-rotation)
            R_ref = ref_mat[:3, :3].copy()
            for col in range(3):
                nrm = np.linalg.norm(R_ref[:, col])
                if nrm > 1e-12:
                    R_ref[:, col] /= nrm
            pre_rot = bone_pre_rotations.get(bone_idx)
            if pre_rot is not None:
                R_pre_inv = euler_to_matrix(pre_rot).T  # inverse = transpose
                R_anim_ref = R_pre_inv @ R_ref
            else:
                R_anim_ref = R_ref
            rot_euler = _matrix_to_euler_xyz(R_anim_ref)

            has_override = False

            if "T" in channels:
                for axis in range(3):
                    if axis in channels["T"]:
                        times_a, vals_a = channels["T"][axis]
                        trans[axis] = _interp_scalar(times_a, vals_a, t,
                                                     trans[axis])
                        has_override = True

            if "R" in channels:
                for axis in range(3):
                    if axis in channels["R"]:
                        times_a, vals_a = channels["R"][axis]
                        rot_euler[axis] = _interp_scalar(times_a, vals_a, t,
                                                          rot_euler[axis])
                        has_override = True

            if "S" in channels:
                for axis in range(3):
                    if axis in channels["S"]:
                        times_a, vals_a = channels["S"][axis]
                        scale[axis] = _interp_scalar(times_a, vals_a, t,
                                                     scale[axis])
                        has_override = True

            if has_override:
                # Build: local = T * R_pre * R_anim * S
                R_anim = euler_to_matrix(rot_euler)

                pre_rot = bone_pre_rotations.get(bone_idx)
                if pre_rot is not None:
                    R_pre = euler_to_matrix(pre_rot)
                    R_combined = R_pre @ R_anim
                else:
                    R_combined = R_anim

                local_mat = np.eye(4, dtype=np.float64)
                local_mat[:3, :3] = R_combined * scale[np.newaxis, :]
                local_mat[:3, 3] = trans

                # Root bones: animation locals are in original FBX space,
                # apply mesh normalization (center + scale).
                # Child bones: norm_mat cancels out in inv(parent) @ child,
                # so their locals stay in original FBX space (correct).
                if bone.parent_index < 0 and mesh_center is not None:
                    norm = np.eye(4, dtype=np.float64)
                    norm[0, 0] = mesh_scale
                    norm[1, 1] = mesh_scale
                    norm[2, 2] = mesh_scale
                    c = mesh_center.astype(np.float64)
                    norm[:3, 3] = -c * mesh_scale
                    local_mat = norm @ local_mat

                pose.bone_locals[bone_idx] = local_mat

        poses.append(pose)

    return poses


def _interp_scalar(
    times: np.ndarray, values: np.ndarray, t: float, default: float,
) -> float:
    """Linearly interpolate a scalar value at time t."""
    if len(times) == 0:
        return default
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])

    idx = int(np.searchsorted(times, t, side='right')) - 1
    idx = max(0, min(idx, len(times) - 2))
    t0, t1 = times[idx], times[idx + 1]
    dt = t1 - t0
    alpha = (t - t0) / dt if dt > 1e-12 else 0.0
    return float(values[idx] * (1.0 - alpha) + values[idx + 1] * alpha)
