"""
mesh_io.py — Loading, flattening, and normalizing triangle meshes via trimesh.

Robust loading with fallback strategies:
  1. Try trimesh.load(force="scene") for multi-geometry files (GLB, GLTF, DAE).
  2. If that fails (e.g. PLY → "No module named ply"), fall back to
     trimesh.load(force="mesh") which uses trimesh's built-in loaders.
  3. Wrap the result in a Scene regardless so downstream code stays uniform.
"""

import numpy as np
import trimesh


def _load_robust(path: str) -> trimesh.Trimesh | trimesh.Scene:
    """Try multiple loading strategies in order of preference.

    Strategy 1: force="scene"  — handles multi-geometry (GLB, GLTF, DAE)
    Strategy 2: force="mesh"   — uses built-in per-format loaders (PLY, STL, OBJ, …)
    Strategy 3: no force       — let trimesh auto-detect
    """
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


def normalize_mesh(mesh: trimesh.Trimesh, target_scale: float = 1.0) -> trimesh.Trimesh:
    """Center at origin and scale to fit into [-target_scale, target_scale]."""
    v = mesh.vertices
    center = (v.min(axis=0) + v.max(axis=0)) * 0.5
    v = v - center
    extent = np.max(v.max(axis=0) - v.min(axis=0))
    if extent > 0:
        v = v * (2.0 * target_scale / extent)
    mesh.vertices = v.astype(np.float32, copy=False)
    return mesh


def load_and_prepare(path: str, target_scale: float = 1.0) -> trimesh.Trimesh:
    """Convenience: load → flatten → normalize in one call.

    Returns a single Trimesh with float32 verts and int32 faces.
    """
    scene = as_trimesh_scene(path)
    mesh = scene_to_single_mesh(scene)
    mesh = normalize_mesh(mesh, target_scale=target_scale)
    return mesh