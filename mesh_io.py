import numpy as np
import trimesh


def as_trimesh_scene(path: str) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(loaded)
        return scene
    return loaded


def scene_to_single_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    geom = scene.to_geometry()

    if isinstance(geom, trimesh.Trimesh):
        mesh = geom
    else:
        dumped = scene.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            mesh = dumped
        else:
            parts = [g for g in dumped.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(parts) if parts else None

    if mesh is None:
        raise ValueError("No triangle mesh found in file.")

    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    mesh.vertices = mesh.vertices.astype(np.float32, copy=False)
    mesh.faces = mesh.faces.astype(np.int32, copy=False)
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, target_scale: float = 1.0) -> trimesh.Trimesh:
    v = mesh.vertices
    center = (v.min(axis=0) + v.max(axis=0)) * 0.5
    v = v - center
    extent = np.max(v.max(axis=0) - v.min(axis=0))
    if extent > 0:
        v = v * (2.0 * target_scale / extent)
    mesh.vertices = v.astype(np.float32, copy=False)
    return mesh


def load_and_prepare(path: str, target_scale: float = 1.0) -> trimesh.Trimesh:
    scene = as_trimesh_scene(path)
    mesh = scene_to_single_mesh(scene)
    mesh = normalize_mesh(mesh, target_scale=target_scale)
    return mesh
