"""
mesh_io.py — Loading, flattening, and normalizing triangle meshes via trimesh.

Robust loading with fallback strategies:
  1. Try trimesh.load(force="scene") for multi-geometry files (GLB, GLTF, DAE).
  2. If that fails (e.g. PLY → "No module named ply"), fall back to
     trimesh.load(force="mesh") which uses trimesh's built-in loaders.
  3. Wrap the result in a Scene regardless so downstream code stays uniform.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class NormalizationTransform:
    """Maps points/lengths between *original* mesh space and the normalized
    space the optimizer works in.

    ``normalize_mesh`` centers the mesh on its AABB midpoint and uniformly
    scales it so the longest axis spans ``2 * target_scale``.  The forward map
    applied to a vertex is::

        v_norm = (v_orig - center) * scale          scale = 2*target_scale/extent

    Because the scale is uniform and positive there is no reflection, so
    ellipsoid orientations (quaternions) are unchanged by it — only centers and
    radii need converting back.
    """

    center: np.ndarray   # (3,) AABB midpoint of the original mesh
    scale: float         # multiply original → normalized

    def to_original_point(self, p: np.ndarray) -> np.ndarray:
        """Normalized-space point(s) → original mesh space."""
        return np.asarray(p, dtype=np.float64) / self.scale + self.center

    def to_original_length(self, length: np.ndarray) -> np.ndarray:
        """Normalized-space length(s)/radii → original mesh space."""
        return np.asarray(length, dtype=np.float64) / self.scale


def _load_robust(path: str) -> trimesh.Trimesh | trimesh.Scene:
    """Try multiple loading strategies in order of preference.

    Strategy 1: force="scene"  — handles multi-geometry (GLB, GLTF, DAE)
    Strategy 2: force="mesh"   — uses built-in per-format loaders (PLY, STL, OBJ, …)
    Strategy 3: no force       — let trimesh auto-detect
    """
    p = Path(path)
    if p.suffix.lower() == ".fbx":
        return _load_fbx_static_mesh(p)

    errors: list[str] = []

    # Strategy 1: scene (best for multi-geometry)
    try:
        return trimesh.load(path, force="scene")
    except Exception as e:
        errors.append(f"force='scene': {e}")

    # Strategy 2: direct mesh (best for PLY, STL, simple OBJ)
    try:
        return trimesh.load(path, force="mesh")
    except Exception as e:
        errors.append(f"force='mesh': {e}")

    # Strategy 3: auto-detect
    try:
        return trimesh.load(path)
    except Exception as e:
        errors.append(f"auto: {e}")

    raise ValueError(
        f"Could not load mesh from '{path}'.\n"
        + "\n".join(f"  - {err}" for err in errors)
    )


def _load_fbx_static_mesh(path: Path) -> trimesh.Trimesh:
    """Load FBX mesh geometry through the local parser.

    trimesh does not provide an FBX loader in this environment. Rigged FBX
    files are handled before this path by rig_loader; this fallback keeps
    static or partially rigged FBX assets usable.
    """
    try:
        from fbx_parser import extract_rig_data

        rig = extract_rig_data(path)
    except Exception as e:
        raise ValueError(f"Could not parse FBX '{path}': {e}") from e

    if rig.mesh is None or len(rig.mesh.vertices) == 0 or len(rig.mesh.faces) == 0:
        raise ValueError(
            f"No triangle mesh found in '{path}'. "
            "Try re-exporting as binary FBX with mesh geometry."
        )

    return trimesh.Trimesh(
        vertices=rig.mesh.vertices.astype(np.float32, copy=False),
        faces=rig.mesh.faces.astype(np.int32, copy=False),
        process=False,
    )


def as_trimesh_scene(path: str) -> trimesh.Scene:
    """Load any mesh file as a trimesh.Scene (handles multi-geometry).

    Falls back through multiple trimesh loading strategies if the
    preferred one fails (e.g. PLY files with force='scene').
    """
    loaded = _load_robust(path)

    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(loaded)
        return scene
    if isinstance(loaded, trimesh.Scene):
        return loaded

    # Rare: trimesh returned something else (PointCloud, Path, …)
    # Try to extract geometry
    scene = trimesh.Scene()
    if hasattr(loaded, 'geometry'):
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                scene.add_geometry(geom, geom_name=name)
    if len(scene.geometry) == 0:
        raise ValueError(f"No triangle mesh found in '{path}' (got {type(loaded).__name__}).")
    return scene


def scene_to_single_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Flatten a Scene into one Trimesh in world coordinates."""
    # Fast path: single geometry
    if len(scene.geometry) == 1:
        mesh = next(iter(scene.geometry.values()))
        if isinstance(mesh, trimesh.Trimesh):
            mesh.vertices = mesh.vertices.astype(np.float32, copy=False)
            mesh.faces = mesh.faces.astype(np.int32, copy=False)
            return mesh

    # Multi-geometry: try to_geometry() first, then dump()
    mesh = None

    try:
        geom = scene.to_geometry()
        if isinstance(geom, trimesh.Trimesh):
            mesh = geom
    except Exception:
        pass

    if mesh is None:
        try:
            dumped = scene.dump(concatenate=True)
            if isinstance(dumped, trimesh.Trimesh):
                mesh = dumped
            elif hasattr(dumped, 'geometry'):
                parts = [g for g in dumped.geometry.values()
                         if isinstance(g, trimesh.Trimesh)]
                if parts:
                    mesh = trimesh.util.concatenate(parts)
        except Exception:
            pass

    if mesh is None:
        # Last resort: concatenate all Trimesh geometries manually
        parts = [g for g in scene.geometry.values()
                 if isinstance(g, trimesh.Trimesh)]
        if parts:
            mesh = trimesh.util.concatenate(parts)

    if mesh is None:
        raise ValueError("No triangle mesh found in file.")

    if len(mesh.faces.shape) > 1 and mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    mesh.vertices = mesh.vertices.astype(np.float32, copy=False)
    mesh.faces = mesh.faces.astype(np.int32, copy=False)
    return mesh


def normalize_mesh_capture(
    mesh: trimesh.Trimesh, target_scale: float = 1.0,
) -> tuple[trimesh.Trimesh, NormalizationTransform]:
    """Like :func:`normalize_mesh` but also returns the transform that was
    applied, so callers can map fitted ellipsoids back to original mesh space.
    """
    v = mesh.vertices
    center = (v.min(axis=0) + v.max(axis=0)) * 0.5
    v = v - center
    extent = float(np.max(v.max(axis=0) - v.min(axis=0)))
    scale = (2.0 * target_scale / extent) if extent > 0 else 1.0
    v = v * scale
    mesh.vertices = v.astype(np.float32, copy=False)
    transform = NormalizationTransform(
        center=np.asarray(center, dtype=np.float64).copy(), scale=float(scale))
    return mesh, transform


def normalize_mesh(mesh: trimesh.Trimesh, target_scale: float = 1.0) -> trimesh.Trimesh:
    """Center at origin and scale to fit into [-target_scale, target_scale]."""
    mesh, _ = normalize_mesh_capture(mesh, target_scale=target_scale)
    return mesh


def load_and_prepare(path: str, target_scale: float = 1.0) -> trimesh.Trimesh:
    """Convenience: load → flatten → normalize in one call.

    Returns a single Trimesh with float32 verts and int32 faces.
    """
    scene = as_trimesh_scene(path)
    mesh = scene_to_single_mesh(scene)
    mesh = normalize_mesh(mesh, target_scale=target_scale)
    return mesh


def load_and_prepare_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_scale: float = 1.0,
) -> tuple[trimesh.Trimesh, NormalizationTransform]:
    """Build a normalized Trimesh from in-memory vertex/face arrays.

    The array-based counterpart to :func:`load_and_prepare`, used by the API
    server when a mesh arrives over the wire (e.g. from Unity) instead of from a
    file.  Returns the normalized mesh *and* the :class:`NormalizationTransform`
    so fitted ellipsoids can be mapped back to the caller's original space.

    Parameters
    ----------
    vertices : (N, 3) array of vertex positions in the caller's coordinate space.
    faces    : (M, 3) array of triangle indices.
    """
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(
            f"Empty mesh: {len(verts)} vertices, {len(tris)} faces.")

    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)

    # Meshes arriving over the wire (Unity, other DCC tools) can have an
    # inconsistent or inward-facing winding.  The SDF sign is derived from
    # triangle orientation (see sdf_compute._winding_flag), so inward-facing
    # normals invert the field — the interior reads as exterior and the fit
    # never converges.  Reorient faces outward so the sign is correct
    # regardless of the caller's winding convention (handedness, CW/CCW).
    try:
        trimesh.repair.fix_normals(mesh)   # fix_winding + volume-based inversion
    except Exception:                      # non-manifold/degenerate → best effort
        pass

    mesh, transform = normalize_mesh_capture(mesh, target_scale=target_scale)
    mesh.vertices = mesh.vertices.astype(np.float32, copy=False)
    mesh.faces = mesh.faces.astype(np.int32, copy=False)
    return mesh, transform
