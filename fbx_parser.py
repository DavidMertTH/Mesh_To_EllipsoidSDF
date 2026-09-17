"""
fbx_parser.py — Focused FBX binary parser for rigged mesh extraction.

Parses the FBX binary format directly (no Autodesk SDK needed).
Extracts only what we need:
  - Skeleton hierarchy (bone names, transforms, parent-child)
  - Skin weights (vertex → bone assignments + weights)
  - Animation keyframes (bone transforms over time)
  - Mesh geometry in the same FBX world space as the skeleton bind pose

Supports FBX versions 7100–7700 (covers Blender, Maya, Mixamo exports).

References:
  - https://code.blender.org/2013/08/fbx-binary-file-format-specification/
  - https://wiki.rogiken.org/specifications/file-format/fbx/
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  Low-level FBX binary reader
# ═════════════════════════════════════════════════════════════════════════════

FBX_MAGIC = b'Kaydara FBX Binary  \x00\x1a\x00'


@dataclass
class FbxProperty:
    """A single FBX property value."""
    type_code: str
    value: Any


@dataclass
class FbxNode:
    """A node in the FBX tree."""
    name: str
    properties: List[FbxProperty] = field(default_factory=list)
    children: List["FbxNode"] = field(default_factory=list)

    def find(self, name: str) -> Optional["FbxNode"]:
        """Find first direct child with given name."""
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name: str) -> List["FbxNode"]:
        """Find all direct children with given name."""
        return [c for c in self.children if c.name == name]

    def prop_value(self, index: int = 0, default=None):
        """Get property value by index."""
        if index < len(self.properties):
            return self.properties[index].value
        return default

    def find_property_node(self, prop_name: str) -> Optional["FbxNode"]:
        """Find child 'P' or 'Properties70' entry by property name."""
        p70 = self.find("Properties70")
        if p70 is None:
            return None
        for p in p70.find_all("P"):
            if p.prop_value(0) == prop_name:
                return p
        return None


def _read_fbx(path: Path) -> Tuple[List[FbxNode], int]:
    """Read an FBX binary file and return (top_level_nodes, version)."""
    data = path.read_bytes()
    if not data[:len(FBX_MAGIC)] == FBX_MAGIC:
        raise ValueError(f"Not an FBX binary file: {path.name}")

    version = struct.unpack_from('<I', data, len(FBX_MAGIC))[0]
    offset = len(FBX_MAGIC) + 4

    # Version > 7500 uses 64-bit offsets
    use_64bit = version >= 7500

    nodes = []
    while offset < len(data):
        node, offset = _read_node(data, offset, use_64bit)
        if node is None:
            break
        nodes.append(node)

    return nodes, version


def _read_node(data: bytes, offset: int, use_64bit: bool) -> Tuple[Optional[FbxNode], int]:
    """Read a single FBX node from the binary data."""
    if use_64bit:
        if offset + 25 > len(data):
            return None, len(data)
        end_offset = struct.unpack_from('<Q', data, offset)[0]
        num_props = struct.unpack_from('<Q', data, offset + 8)[0]
        prop_list_len = struct.unpack_from('<Q', data, offset + 16)[0]
        name_len = data[offset + 24]
        offset += 25
    else:
        if offset + 13 > len(data):
            return None, len(data)
        end_offset = struct.unpack_from('<I', data, offset)[0]
        num_props = struct.unpack_from('<I', data, offset + 4)[0]
        prop_list_len = struct.unpack_from('<I', data, offset + 8)[0]
        name_len = data[offset + 12]
        offset += 13

    # Null node sentinel
    if end_offset == 0:
        return None, offset

    name = data[offset:offset + name_len].decode('ascii', errors='replace')
    offset += name_len

    # Read properties
    props = []
    prop_end = offset + prop_list_len
    for _ in range(num_props):
        if offset >= prop_end:
            break
        prop, offset = _read_property(data, offset)
        props.append(prop)
    offset = prop_end  # Ensure we're past all properties

    # Read nested nodes
    children = []
    while offset < end_offset:
        child, offset = _read_node(data, offset, use_64bit)
        if child is None:
            break
        children.append(child)

    return FbxNode(name=name, properties=props, children=children), end_offset


def _read_property(data: bytes, offset: int) -> Tuple[FbxProperty, int]:
    """Read a single FBX property."""
    type_code = chr(data[offset])
    offset += 1

    if type_code == 'Y':  # int16
        val = struct.unpack_from('<h', data, offset)[0]
        return FbxProperty(type_code, val), offset + 2

    elif type_code == 'C':  # bool (uint8)
        val = bool(data[offset])
        return FbxProperty(type_code, val), offset + 1

    elif type_code == 'I':  # int32
        val = struct.unpack_from('<i', data, offset)[0]
        return FbxProperty(type_code, val), offset + 4

    elif type_code == 'F':  # float32
        val = struct.unpack_from('<f', data, offset)[0]
        return FbxProperty(type_code, val), offset + 4

    elif type_code == 'D':  # float64
        val = struct.unpack_from('<d', data, offset)[0]
        return FbxProperty(type_code, val), offset + 8

    elif type_code == 'L':  # int64
        val = struct.unpack_from('<q', data, offset)[0]
        return FbxProperty(type_code, val), offset + 8

    elif type_code == 'S':  # string
        length = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        val = data[offset:offset + length].decode('utf-8', errors='replace')
        # FBX uses \x00\x01 as separator in some strings
        val = val.split('\x00')[0]
        return FbxProperty(type_code, val), offset + length

    elif type_code == 'R':  # raw bytes
        length = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        val = data[offset:offset + length]
        return FbxProperty(type_code, val), offset + length

    elif type_code in ('f', 'd', 'i', 'l'):  # arrays
        array_len = struct.unpack_from('<I', data, offset)[0]
        encoding = struct.unpack_from('<I', data, offset + 4)[0]
        comp_len = struct.unpack_from('<I', data, offset + 8)[0]
        offset += 12

        if type_code == 'f':
            dtype, elem_size = np.float32, 4
        elif type_code == 'd':
            dtype, elem_size = np.float64, 8
        elif type_code == 'i':
            dtype, elem_size = np.int32, 4
        elif type_code == 'l':
            dtype, elem_size = np.int64, 8
        else:
            dtype, elem_size = np.uint8, 1

        raw = data[offset:offset + comp_len]
        if encoding == 1:  # zlib compressed
            raw = zlib.decompress(raw)

        arr = np.frombuffer(raw, dtype=dtype, count=array_len).copy()
        return FbxProperty(type_code, arr), offset + comp_len

    else:
        # Unknown type — skip (shouldn't happen with valid FBX)
        return FbxProperty(type_code, None), offset


# ═════════════════════════════════════════════════════════════════════════════
#  High-level extraction
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FbxBoneInfo:
    """Raw bone data extracted from FBX."""
    fbx_id: int
    name: str
    local_translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    local_rotation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )  # Euler degrees (FBX default)
    local_scaling: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64),
    )
    pre_rotation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    parent_id: int = -1


@dataclass
class FbxSkinCluster:
    """Skin cluster: one bone's influence on a set of vertices."""
    bone_fbx_id: int
    indices: np.ndarray       # vertex indices
    weights: np.ndarray       # per-vertex weights
    transform: np.ndarray     # (4,4) — cluster transform
    transform_link: np.ndarray  # (4,4) — bone bind-pose world transform


@dataclass
class FbxAnimCurve:
    """Animation curve for a single bone property channel."""
    bone_fbx_id: int
    channel: str  # "T", "R", "S"
    axis: int     # 0=X, 1=Y, 2=Z
    times: np.ndarray
    values: np.ndarray


@dataclass
class FbxMeshData:
    """Mesh geometry extracted from FBX."""
    vertices: np.ndarray    # (V, 3) float32
    faces: np.ndarray       # (F, 3) int32 — triangulated
    geometry_to_world: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64),
    )
    model_fbx_id: int = 0


@dataclass
class FbxRigData:
    """Complete rig data extracted from an FBX file."""
    bones: List[FbxBoneInfo]
    skin_clusters: List[FbxSkinCluster]
    anim_curves: List[FbxAnimCurve]
    num_mesh_vertices: int
    mesh: Optional[FbxMeshData] = None


_FBX_ROTATION_ORDERS = ("XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX")


def _translation_matrix(value: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(value, dtype=np.float64).reshape(3)
    return out


def _scale_matrix(value: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[0, 0], out[1, 1], out[2, 2] = (
        np.asarray(value, dtype=np.float64).reshape(3)
    )
    return out


def _model_vec3_property(
    model: FbxNode,
    name: str,
    default: tuple[float, float, float],
) -> np.ndarray:
    prop = model.find_property_node(name)
    if prop is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([
        prop.prop_value(4, default[0]),
        prop.prop_value(5, default[1]),
        prop.prop_value(6, default[2]),
    ], dtype=np.float64)


def _model_int_property(model: FbxNode, name: str, default: int) -> int:
    prop = model.find_property_node(name)
    if prop is None:
        return int(default)
    try:
        return int(prop.prop_value(4, default))
    except (TypeError, ValueError):
        return int(default)


def _euler_to_matrix_ordered(
    euler_deg: np.ndarray,
    order_index: int = 0,
) -> np.ndarray:
    """Return an FBX Euler rotation as a homogeneous column-vector matrix."""
    order = (
        _FBX_ROTATION_ORDERS[order_index]
        if 0 <= int(order_index) < len(_FBX_ROTATION_ORDERS)
        else "XYZ"
    )
    radians = np.radians(np.asarray(euler_deg, dtype=np.float64).reshape(3))
    cx, sx = np.cos(radians[0]), np.sin(radians[0])
    cy, sy = np.cos(radians[1]), np.sin(radians[1])
    cz, sz = np.cos(radians[2]), np.sin(radians[2])
    rotations = {
        "X": np.array([
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ], dtype=np.float64),
        "Y": np.array([
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ], dtype=np.float64),
        "Z": np.array([
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
    }
    rotation = np.eye(3, dtype=np.float64)
    for axis in order:
        # FBX's XYZ order applies X, then Y, then Z to a point. With column
        # vectors that is Rz @ Ry @ Rx, hence each new axis pre-multiplies.
        rotation = rotations[axis] @ rotation
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    return out


def _model_local_transform(model: FbxNode) -> np.ndarray:
    """Evaluate the FBX node-local transform including pivots and pre/post R."""
    translation = _model_vec3_property(
        model, "Lcl Translation", (0.0, 0.0, 0.0))
    rotation = _model_vec3_property(
        model, "Lcl Rotation", (0.0, 0.0, 0.0))
    scaling = _model_vec3_property(
        model, "Lcl Scaling", (1.0, 1.0, 1.0))
    rotation_offset = _model_vec3_property(
        model, "RotationOffset", (0.0, 0.0, 0.0))
    rotation_pivot = _model_vec3_property(
        model, "RotationPivot", (0.0, 0.0, 0.0))
    pre_rotation = _model_vec3_property(
        model, "PreRotation", (0.0, 0.0, 0.0))
    post_rotation = _model_vec3_property(
        model, "PostRotation", (0.0, 0.0, 0.0))
    scaling_offset = _model_vec3_property(
        model, "ScalingOffset", (0.0, 0.0, 0.0))
    scaling_pivot = _model_vec3_property(
        model, "ScalingPivot", (0.0, 0.0, 0.0))
    rotation_order = _model_int_property(model, "RotationOrder", 0)

    pre = _euler_to_matrix_ordered(pre_rotation, rotation_order)
    local_rotation = _euler_to_matrix_ordered(rotation, rotation_order)
    post = _euler_to_matrix_ordered(post_rotation, rotation_order)
    try:
        post_inverse = np.linalg.inv(post)
    except np.linalg.LinAlgError:
        post_inverse = np.eye(4, dtype=np.float64)

    # Autodesk FBX transform evaluation order. Geometric transforms are not
    # included here because they affect geometry only and are not inherited.
    return (
        _translation_matrix(translation)
        @ _translation_matrix(rotation_offset)
        @ _translation_matrix(rotation_pivot)
        @ pre
        @ local_rotation
        @ post_inverse
        @ _translation_matrix(-rotation_pivot)
        @ _translation_matrix(scaling_offset)
        @ _translation_matrix(scaling_pivot)
        @ _scale_matrix(scaling)
        @ _translation_matrix(-scaling_pivot)
    )


def _model_geometric_transform(model: FbxNode) -> np.ndarray:
    """Return the non-inherited FBX GeometricTranslation/Rotation/Scaling."""
    translation = _model_vec3_property(
        model, "GeometricTranslation", (0.0, 0.0, 0.0))
    rotation = _model_vec3_property(
        model, "GeometricRotation", (0.0, 0.0, 0.0))
    scaling = _model_vec3_property(
        model, "GeometricScaling", (1.0, 1.0, 1.0))
    rotation_order = _model_int_property(model, "RotationOrder", 0)
    return (
        _translation_matrix(translation)
        @ _euler_to_matrix_ordered(rotation, rotation_order)
        @ _scale_matrix(scaling)
    )


def _extract_bind_pose_matrices(objects: FbxNode) -> Dict[int, np.ndarray]:
    """Extract authoritative Model world matrices from FBX BindPose objects."""
    matrices: Dict[int, np.ndarray] = {}
    for obj in objects.children:
        if obj.name != "Pose":
            continue
        pose_kind = " ".join((
            str(obj.prop_value(1, "")), str(obj.prop_value(2, "")),
        )).replace(" ", "").lower()
        if "bindpose" not in pose_kind:
            continue
        for pose_node in obj.find_all("PoseNode"):
            node = pose_node.find("Node")
            matrix = pose_node.find("Matrix")
            if node is None or matrix is None:
                continue
            raw = matrix.prop_value(0)
            if not isinstance(raw, np.ndarray) or raw.size != 16:
                continue
            world = raw.reshape(4, 4).T.astype(np.float64)
            if np.all(np.isfinite(world)):
                matrices.setdefault(int(node.prop_value(0, 0)), world)
    return matrices


def _model_world_transform(
    model_id: int,
    models: Dict[int, FbxNode],
    parents_map: Dict[int, List[int]],
    bind_pose_matrices: Dict[int, np.ndarray],
    cache: Dict[int, np.ndarray],
    visiting: Optional[set[int]] = None,
) -> np.ndarray:
    """Resolve a Model's world transform, preferring its bind-pose matrix."""
    if model_id in cache:
        return cache[model_id]
    if model_id in bind_pose_matrices:
        cache[model_id] = bind_pose_matrices[model_id].copy()
        return cache[model_id]
    model = models.get(model_id)
    if model is None:
        return np.eye(4, dtype=np.float64)

    active = set() if visiting is None else visiting
    if model_id in active:
        return _model_local_transform(model)
    active.add(model_id)
    parent_id = next(
        (pid for pid in parents_map.get(model_id, []) if pid in models),
        None,
    )
    local = _model_local_transform(model)
    if parent_id is None:
        world = local
    else:
        world = _model_world_transform(
            parent_id, models, parents_map, bind_pose_matrices, cache, active,
        ) @ local
    active.remove(model_id)
    cache[model_id] = world
    return world


def _geometry_to_world_transform(
    model_id: int,
    models: Dict[int, FbxNode],
    parents_map: Dict[int, List[int]],
    bind_pose_matrices: Dict[int, np.ndarray],
    cache: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """Return ModelGlobal @ GeometricTRS for vertices owned by ``model_id``."""
    model = models.get(model_id)
    if model is None:
        return np.eye(4, dtype=np.float64)
    world = _model_world_transform(
        model_id, models, parents_map, bind_pose_matrices,
        {} if cache is None else cache,
    )
    return world @ _model_geometric_transform(model)


def _transform_mesh_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a homogeneous transform and preserve outward triangle winding."""
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    transformed_h = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T
    w = transformed_h[:, 3:4]
    if np.any(np.abs(w) <= 1.0e-12):
        raise ValueError("FBX mesh Model transform produced points at infinity")
    transformed = (transformed_h[:, :3] / w).astype(np.float32)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("FBX mesh Model transform produced non-finite vertices")

    out_faces = np.asarray(faces, dtype=np.int32).copy()
    if np.linalg.det(np.asarray(transform, dtype=np.float64)[:3, :3]) < 0.0:
        out_faces = out_faces[:, [0, 2, 1]]
    return transformed, out_faces


def extract_rig_data(path: Path) -> FbxRigData:
    """Parse an FBX file and extract all rig-relevant data.

    Parameters
    ----------
    path : Path to .fbx file

    Returns
    -------
    FbxRigData with bones, skin clusters, and animation curves.
    """
    nodes, version = _read_fbx(path)

    # Build node lookup
    root_map: Dict[str, FbxNode] = {n.name: n for n in nodes}

    objects = root_map.get("Objects")
    connections = root_map.get("Connections")

    if objects is None:
        raise ValueError("FBX file has no Objects section.")
    if connections is None:
        raise ValueError("FBX file has no Connections section.")

    # ── Parse connections → parent map ──
    # connection: "OO" or "OP", child_id, parent_id [, property]
    # A node can have multiple OO connections (to parent bone, anim nodes,
    # skin clusters, etc.), so we store ALL parents per child.
    parents_map: Dict[int, List[int]] = {}  # child_id → [parent_ids]
    children_map: Dict[int, List[int]] = {}  # parent_id → [child_ids]
    conn_list: List[Tuple[str, int, int]] = []

    for c in connections.find_all("C"):
        conn_type = c.prop_value(0, "")
        child_id = c.prop_value(1, 0)
        parent_id = c.prop_value(2, 0)
        conn_list.append((conn_type, child_id, parent_id))
        if conn_type == "OO":
            if child_id not in parents_map:
                parents_map[child_id] = []
            parents_map[child_id].append(parent_id)
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)

    # Single-parent convenience (first OO parent) — used by non-bone lookups
    parent_map: Dict[int, int] = {
        cid: pids[0] for cid, pids in parents_map.items()
    }

    # ── Parse Models (bones + mesh) ──
    models: Dict[int, FbxNode] = {}
    bone_infos: Dict[int, FbxBoneInfo] = {}
    mesh_model_ids: List[int] = []

    for obj in objects.children:
        if obj.name != "Model":
            continue
        fbx_id = obj.prop_value(0, 0)
        name_raw = obj.prop_value(1, "")
        obj_type = obj.prop_value(2, "")

        # Clean up FBX name (often "Model::BoneName")
        name = name_raw.split('\x00')[0].split('\x01')[0]
        if '::' in name:
            name = name.split('::', 1)[1]

        models[fbx_id] = obj

        if obj_type in ("LimbNode", "Limb", "Root", "Null"):
            info = FbxBoneInfo(fbx_id=fbx_id, name=name)
            _extract_transforms(obj, info)
            bone_infos[fbx_id] = info
        elif obj_type == "Mesh":
            mesh_model_ids.append(fbx_id)

    # ── Resolve bone parent-child via connections ──
    # A bone may have multiple OO parents (anim nodes, skin clusters, etc.)
    # Find the one that is actually another bone (Model with LimbNode/etc. type).
    for bone_id, info in bone_infos.items():
        for pid in parents_map.get(bone_id, []):
            if pid in bone_infos:
                info.parent_id = pid
                break

    # Geometry is stored in its owning Model's local space, while cluster
    # TransformLink matrices are FBX-world bind transforms. Resolve the exact
    # Geometry → Model connection and move vertices into that same world space.
    geometry_ids = {
        int(obj.prop_value(0, 0))
        for obj in objects.children
        if obj.name == "Geometry"
    }
    geometry_to_model: Dict[int, int] = {}
    for conn_type, child_id, parent_id in conn_list:
        if conn_type != "OO":
            continue
        if child_id in models and parent_id in geometry_ids:
            # Reversed connections are accepted for defensive compatibility.
            geometry_to_model.setdefault(parent_id, child_id)
        elif child_id in geometry_ids and parent_id in models:
            geometry_to_model.setdefault(child_id, parent_id)
    bind_pose_matrices = _extract_bind_pose_matrices(objects)
    model_world_cache: Dict[int, np.ndarray] = {}

    # ── Parse Geometry → extract mesh ──
    num_verts = 0
    mesh_data: Optional[FbxMeshData] = None

    for obj in objects.children:
        if obj.name == "Geometry":
            # Type is usually prop[2] = "Mesh", but some exporters
            # omit it or use different casing / encoding.
            geom_type = str(obj.prop_value(2, "")).strip().lower()
            is_mesh = ("mesh" in geom_type) or (geom_type == "")

            if is_mesh:
                verts_node = obj.find("Vertices")
                poly_node = obj.find("PolygonVertexIndex")

                if verts_node is not None:
                    v = verts_node.prop_value(0)
                    if isinstance(v, np.ndarray) and len(v) >= 3:
                        num_verts = len(v) // 3
                        vertices = v.astype(np.float32).reshape(-1, 3)

                        if poly_node is not None:
                            pvi = poly_node.prop_value(0)
                            if isinstance(pvi, np.ndarray) and len(pvi) >= 3:
                                faces = _triangulate_fbx_polygons(pvi)
                                geometry_id = int(obj.prop_value(0, 0))
                                model_id = geometry_to_model.get(geometry_id, 0)
                                if model_id == 0 and len(mesh_model_ids) == 1:
                                    model_id = mesh_model_ids[0]
                                geometry_to_world = _geometry_to_world_transform(
                                    model_id,
                                    models,
                                    parents_map,
                                    bind_pose_matrices,
                                    model_world_cache,
                                )
                                vertices, faces = _transform_mesh_geometry(
                                    vertices, faces, geometry_to_world)
                                mesh_data = FbxMeshData(
                                    vertices=vertices,
                                    faces=faces,
                                    geometry_to_world=geometry_to_world,
                                    model_fbx_id=model_id,
                                )
                if mesh_data is not None:
                    break  # Found usable mesh

    # ── Parse Deformers (Skin + Clusters) ──
    skin_clusters: List[FbxSkinCluster] = []
    deformer_ids: Dict[int, FbxNode] = {}
    cluster_ids: Dict[int, FbxNode] = {}

    for obj in objects.children:
        if obj.name != "Deformer":
            continue
        fbx_id = obj.prop_value(0, 0)
        sub_type = obj.prop_value(2, "")

        if sub_type == "Skin":
            deformer_ids[fbx_id] = obj
        elif sub_type == "Cluster":
            cluster_ids[fbx_id] = obj

    # Resolve cluster → bone via connections
    cluster_to_bone: Dict[int, int] = {}
    for conn_type, child_id, parent_id in conn_list:
        if conn_type == "OO" and child_id in bone_infos and parent_id in cluster_ids:
            cluster_to_bone[parent_id] = child_id
        elif conn_type == "OO" and parent_id in bone_infos and child_id in cluster_ids:
            cluster_to_bone[child_id] = parent_id

    for cluster_id, cluster_node in cluster_ids.items():
        bone_id = cluster_to_bone.get(cluster_id, -1)
        if bone_id < 0:
            continue

        indices_node = cluster_node.find("Indexes")
        weights_node = cluster_node.find("Weights")
        transform_node = cluster_node.find("Transform")
        transform_link_node = cluster_node.find("TransformLink")

        if indices_node is None or weights_node is None:
            continue

        indices = indices_node.prop_value(0, np.array([], dtype=np.int32))
        weights = weights_node.prop_value(0, np.array([], dtype=np.float64))

        transform = np.eye(4, dtype=np.float64)
        if transform_node is not None:
            t_arr = transform_node.prop_value(0)
            if isinstance(t_arr, np.ndarray) and len(t_arr) == 16:
                transform = t_arr.reshape(4, 4).T.astype(np.float64)

        transform_link = np.eye(4, dtype=np.float64)
        if transform_link_node is not None:
            tl_arr = transform_link_node.prop_value(0)
            if isinstance(tl_arr, np.ndarray) and len(tl_arr) == 16:
                transform_link = tl_arr.reshape(4, 4).T.astype(np.float64)

        skin_clusters.append(FbxSkinCluster(
            bone_fbx_id=bone_id,
            indices=indices.astype(np.int32),
            weights=weights.astype(np.float64),
            transform=transform,
            transform_link=transform_link,
        ))

    # ── Parse AnimationCurves ──
    anim_curves: List[FbxAnimCurve] = []
    curve_nodes: Dict[int, FbxNode] = {}
    anim_curve_node_ids: Dict[int, FbxNode] = {}

    for obj in objects.children:
        if obj.name == "AnimationCurveNode":
            fbx_id = obj.prop_value(0, 0)
            anim_curve_node_ids[fbx_id] = obj
        elif obj.name == "AnimationCurve":
            fbx_id = obj.prop_value(0, 0)
            curve_nodes[fbx_id] = obj

    # Build curve_node → bone mapping and curve → curve_node mapping
    curve_node_to_bone: Dict[int, Tuple[int, str]] = {}
    curve_to_curve_node: Dict[int, Tuple[int, str]] = {}

    for conn_type, child_id, parent_id in conn_list:
        # AnimCurveNode → Model (bone) with property name
        if (conn_type == "OP" and child_id in anim_curve_node_ids
                and parent_id in bone_infos):
            # The 4th property is the channel name
            # We need to find the connection with the property
            pass

    # Simplified: parse from OP connections
    for c in connections.find_all("C"):
        conn_type = c.prop_value(0, "")
        child_id = c.prop_value(1, 0)
        parent_id = c.prop_value(2, 0)
        prop_name = c.prop_value(3, "")

        if conn_type == "OP":
            if child_id in anim_curve_node_ids and parent_id in bone_infos:
                # channel is T, R, or S from the property name
                channel = ""
                if "Lcl Translation" in prop_name:
                    channel = "T"
                elif "Lcl Rotation" in prop_name:
                    channel = "R"
                elif "Lcl Scaling" in prop_name:
                    channel = "S"
                if channel:
                    curve_node_to_bone[child_id] = (parent_id, channel)

            elif child_id in curve_nodes and parent_id in anim_curve_node_ids:
                # Axis from property: "d|X", "d|Y", "d|Z"
                axis = -1
                if "X" in prop_name:
                    axis = 0
                elif "Y" in prop_name:
                    axis = 1
                elif "Z" in prop_name:
                    axis = 2
                if axis >= 0:
                    curve_to_curve_node[child_id] = (parent_id, axis)

    # Now extract actual curve data
    for curve_id, (curve_node_id, axis) in curve_to_curve_node.items():
        if curve_node_id not in curve_node_to_bone:
            continue
        bone_id, channel = curve_node_to_bone[curve_node_id]

        curve_node = curve_nodes[curve_id]
        key_time_node = curve_node.find("KeyTime")
        key_value_node = curve_node.find("KeyValueFloat")

        if key_time_node is None or key_value_node is None:
            continue

        times_raw = key_time_node.prop_value(0)
        values = key_value_node.prop_value(0)

        if not isinstance(times_raw, np.ndarray) or not isinstance(values, np.ndarray):
            continue

        # FBX times are in "FBX time units" (1/46186158000 seconds)
        FBX_TIME_UNIT = 46186158000.0
        times = times_raw.astype(np.float64) / FBX_TIME_UNIT

        anim_curves.append(FbxAnimCurve(
            bone_fbx_id=bone_id,
            channel=channel,
            axis=axis,
            times=times,
            values=values.astype(np.float64),
        ))

    # ── Build ordered bone list ──
    bone_list = _order_bones(bone_infos)

    return FbxRigData(
        bones=bone_list,
        skin_clusters=skin_clusters,
        anim_curves=anim_curves,
        num_mesh_vertices=num_verts,
        mesh=mesh_data,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def _triangulate_fbx_polygons(pvi: np.ndarray) -> np.ndarray:
    """Convert FBX PolygonVertexIndex to triangulated (F, 3) int32 faces.

    FBX encodes polygon ends by negating the last index and subtracting 1:
      - Triangle [0, 1, 2] is stored as [0, 1, -3]  (last = -(2+1))
      - Quad [0, 1, 2, 3] is stored as [0, 1, 2, -4]  (last = -(3+1))

    Quads and n-gons are triangulated using a fan from vertex 0.
    """
    faces = []
    poly_start = 0

    for i in range(len(pvi)):
        if pvi[i] < 0:
            # End of polygon — decode the last index
            last_idx = -(pvi[i] + 1)
            poly_indices = []
            for j in range(poly_start, i):
                poly_indices.append(int(pvi[j]))
            poly_indices.append(int(last_idx))

            # Fan triangulation
            if len(poly_indices) >= 3:
                v0 = poly_indices[0]
                for k in range(1, len(poly_indices) - 1):
                    faces.append([v0, poly_indices[k], poly_indices[k + 1]])

            poly_start = i + 1

    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.array(faces, dtype=np.int32)

def _extract_transforms(model_node: FbxNode, info: FbxBoneInfo):
    """Extract Lcl Translation / Rotation / Scaling from a Model node."""
    p70 = model_node.find("Properties70")
    if p70 is None:
        return

    for p in p70.find_all("P"):
        prop_name = p.prop_value(0, "")
        if prop_name == "Lcl Translation":
            info.local_translation = np.array([
                p.prop_value(4, 0.0),
                p.prop_value(5, 0.0),
                p.prop_value(6, 0.0),
            ], dtype=np.float64)
        elif prop_name == "Lcl Rotation":
            info.local_rotation = np.array([
                p.prop_value(4, 0.0),
                p.prop_value(5, 0.0),
                p.prop_value(6, 0.0),
            ], dtype=np.float64)
        elif prop_name == "Lcl Scaling":
            info.local_scaling = np.array([
                p.prop_value(4, 1.0),
                p.prop_value(5, 1.0),
                p.prop_value(6, 1.0),
            ], dtype=np.float64)
        elif prop_name == "PreRotation":
            info.pre_rotation = np.array([
                p.prop_value(4, 0.0),
                p.prop_value(5, 0.0),
                p.prop_value(6, 0.0),
            ], dtype=np.float64)


def _order_bones(bone_infos: Dict[int, FbxBoneInfo]) -> List[FbxBoneInfo]:
    """Sort bones in topological order (parents before children)."""
    ordered = []
    visited = set()
    id_to_info = bone_infos

    def visit(fbx_id):
        if fbx_id in visited:
            return
        info = id_to_info.get(fbx_id)
        if info is None:
            return
        if info.parent_id in id_to_info and info.parent_id not in visited:
            visit(info.parent_id)
        visited.add(fbx_id)
        ordered.append(info)

    for fbx_id in bone_infos:
        visit(fbx_id)

    return ordered


def euler_to_quat_xyzw(euler_deg: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Convert Euler angles (degrees) to quaternion (x,y,z,w).

    FBX default rotation order is XYZ (applied as Rz * Ry * Rx to a point).
    """
    r = np.radians(euler_deg.astype(np.float64))
    cx, sx = np.cos(r[0] / 2), np.sin(r[0] / 2)
    cy, sy = np.cos(r[1] / 2), np.sin(r[1] / 2)
    cz, sz = np.cos(r[2] / 2), np.sin(r[2] / 2)

    # XYZ intrinsic = ZYX extrinsic
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q /= n
    return q


def euler_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """Euler XYZ degrees → 3×3 rotation matrix."""
    r = np.radians(euler_deg.astype(np.float64))
    cx, sx = np.cos(r[0]), np.sin(r[0])
    cy, sy = np.cos(r[1]), np.sin(r[1])
    cz, sz = np.cos(r[2]), np.sin(r[2])

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx
