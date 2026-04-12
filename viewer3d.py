"""
viewer3d.py — 3-D scene viewers for meshes and ellipsoids.

Hierarchy:
  _BaseViewer          – shared setup (GL widget, grid, axis)
  MeshViewer3D         – shows a single triangle mesh
  EllipsoidViewer3D    – shows a set of ellipsoids as a single concatenated
                         GLMeshItem with per-face colours.  A unit icosphere
                         is precomputed once; each update only applies
                         NumPy transforms (scale → rotate → translate) and
                         concatenates.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import trimesh
import pyqtgraph.opengl as gl

from widgets import DropGLView


# ── Colour palette for ellipsoids ─────────────────────────────────────────────

ELLIPSOID_PALETTE = np.array([
    [242, 230,  65, 179],
    [ 73,  98, 242, 179],
    [242, 213,  65, 179],
    [ 24,  40,  89, 217],
    [140, 160, 242, 179],
    [200, 195,  55, 179],
    [ 50,  70, 160, 191],
    [220, 220, 180, 166],
], dtype=np.float32) / 255.0

DEFAULT_OPT_COLOR = np.array([73, 98, 242, 179], dtype=np.float32) / 255.0


# ── Precomputed unit icosphere ────────────────────────────────────────────────

def _make_unit_icosphere(subdivisions: int = 3):
    """Create a unit icosphere once.  Returns (verts, faces) as float32/int32."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    verts = np.ascontiguousarray(sphere.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(sphere.faces, dtype=np.int32)
    return verts, faces

_UNIT_VERTS, _UNIT_FACES = _make_unit_icosphere(3)
_UNIT_N_VERTS = _UNIT_VERTS.shape[0]
_UNIT_N_FACES = _UNIT_FACES.shape[0]


def _quat_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert (x,y,z,w) quaternion to a 3×3 rotation matrix (float32)."""
    x, y, z, w = quat_xyzw.astype(np.float64)
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    x /= n; y /= n; z /= n; w /= n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
        [    2*(xy+wz), 1 - 2*(xx+zz),     2*(yz-wx)],
        [    2*(xz-wy),     2*(yz+wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)


def build_concatenated_mesh(
    centers: np.ndarray,
    radii: np.ndarray,
    rotations: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> tuple:
    """Build a single concatenated mesh from N ellipsoid parameters.

    Returns (all_verts, all_faces, all_face_colors).
    """
    n_ell = centers.shape[0]
    if n_ell == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.int32),
                np.empty((0, 4), dtype=np.float32))

    V = _UNIT_N_VERTS
    F = _UNIT_N_FACES

    all_verts = np.empty((n_ell * V, 3), dtype=np.float32)
    all_faces = np.empty((n_ell * F, 3), dtype=np.int32)
    all_face_colors = np.empty((n_ell * F, 4), dtype=np.float32)

    for i in range(n_ell):
        v = _UNIT_VERTS * radii[i]
        R = _quat_to_rotation_matrix(rotations[i])
        v = v @ R.T
        v += centers[i]

        vs = i * V
        fs = i * F
        all_verts[vs:vs + V] = v
        all_faces[fs:fs + F] = _UNIT_FACES + vs

        if colors is not None and len(colors) > 0:
            c = colors[i % len(colors)] if len(colors) > 1 else colors[0]
        else:
            c = ELLIPSOID_PALETTE[i % len(ELLIPSOID_PALETTE)]
        all_face_colors[fs:fs + F] = c

    return all_verts, all_faces, all_face_colors


# ── Base viewer ───────────────────────────────────────────────────────────────

class _BaseViewer:
    """Common setup: GL widget with grid and axis."""

    BG_COLOR = (0, 0, 0, 255)
    def __init__(self):
        self._view = DropGLView()
        self._view.setBackgroundColor(self.BG_COLOR)
        self._view.setCameraPosition(distance=3.0, elevation=15, azimuth=45)

        # grid = gl.GLGridItem()
        # grid.scale(1, 1, 1)
        # self._view.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self._view.addItem(axis)

    @property
    def widget(self) -> DropGLView:
        return self._view


# ── Mesh viewer ───────────────────────────────────────────────────────────────

class MeshViewer3D(_BaseViewer):
    """Shows a single triangle mesh with optional skeleton overlay."""

    MESH_FACE_COLOR = (73 / 256, 98 / 256, 242 / 256, 0.0)
    MESH_EDGE_COLOR = (242 / 256, 230 / 256, 65 / 256, 0.3)
    BONE_COLOR = (1.0, 0.2, 0.2, 1.0)
    JOINT_COLOR = (1.0, 1.0, 0.0, 1.0)

    def __init__(self):
        super().__init__()
        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._bone_items: List[gl.GLLinePlotItem] = []
        self._joint_item: Optional[gl.GLScatterPlotItem] = None

    def show_mesh(self, verts: np.ndarray, faces: np.ndarray) -> None:
        self.clear_mesh()
        from OpenGL import GL as _GL
        self._mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            color=self.MESH_FACE_COLOR,
            smooth=False,
            drawEdges=True,
            drawFaces=True,
            edgeColor=self.MESH_EDGE_COLOR,
        )
        self._mesh_item.setGLOptions({
            _GL.GL_DEPTH_TEST: True,
            _GL.GL_BLEND: True,
            'glBlendFuncSeparate': (
                _GL.GL_SRC_ALPHA, _GL.GL_ONE_MINUS_SRC_ALPHA,
                _GL.GL_ONE, _GL.GL_ONE_MINUS_SRC_ALPHA,
            ),
        })
        self._view.addItem(self._mesh_item)

    def show_bones(self, positions: np.ndarray, parent_indices: np.ndarray) -> None:
        """Draw skeleton bones as lines + joints as dots.

        Parameters
        ----------
        positions : (B, 3) float32 — world-space joint positions
        parent_indices : (B,) int — parent index per bone (-1 for roots)
        """
        self.clear_bones()

        # Lines: one segment per bone that has a parent
        line_pts = []
        for i, pi in enumerate(parent_indices):
            if pi >= 0:
                line_pts.append(positions[pi])
                line_pts.append(positions[i])

        if line_pts:
            pts = np.array(line_pts, dtype=np.float32)
            line_item = gl.GLLinePlotItem(
                pos=pts,
                color=self.BONE_COLOR,
                width=3.0,
                mode='lines',
            )
            self._view.addItem(line_item)
            self._bone_items.append(line_item)

        # Joint dots
        self._joint_item = gl.GLScatterPlotItem(
            pos=positions,
            color=self.JOINT_COLOR,
            size=8.0,
            pxMode=True,
        )
        self._view.addItem(self._joint_item)

    def clear_bones(self) -> None:
        for item in self._bone_items:
            self._view.removeItem(item)
        self._bone_items.clear()
        if self._joint_item is not None:
            self._view.removeItem(self._joint_item)
            self._joint_item = None

    def clear_mesh(self) -> None:
        if self._mesh_item is not None:
            self._view.removeItem(self._mesh_item)
            self._mesh_item = None


# ── Ellipsoid viewer ──────────────────────────────────────────────────────────

class EllipsoidViewer3D(_BaseViewer):
    """
    Shows a set of ellipsoids as **one** GLMeshItem.

    A unit icosphere (subdivisions=3) is precomputed once at module load.
    Each call to ``show_ellipsoids_fast`` transforms the cached vertices
    with per-ellipsoid scale/rotation/translation on CPU, concatenates
    everything, and feeds the result into a single GLMeshItem with
    per-face colours.  No trimesh recreation per ellipsoid.
    """

    def __init__(self):
        super().__init__()
        self._item: Optional[gl.GLMeshItem] = None

    # ── fast path (raw numpy arrays) ──────────────────────────────────

    def show_ellipsoids_fast(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        """Update the display from raw numpy arrays — no EllipsoidSet needed."""
        verts, faces, face_colors = build_concatenated_mesh(
            centers, radii, rotations, colors,
        )
        self._set_mesh(verts, faces, face_colors)

    # ── legacy path (EllipsoidSet object) ─────────────────────────────

    def show_ellipsoids(self, ellipsoid_set) -> None:
        """Update from an EllipsoidSet (calls show_ellipsoids_fast)."""
        colors = None
        if ellipsoid_set.colors is not None:
            colors = np.array(ellipsoid_set.colors, dtype=np.float32)
        self.show_ellipsoids_fast(
            ellipsoid_set.centers,
            ellipsoid_set.radii,
            ellipsoid_set.rotations,
            colors,
        )

    def clear_ellipsoids(self) -> None:
        if self._item is not None:
            self._view.removeItem(self._item)
            self._item = None

    # ── internal ──────────────────────────────────────────────────────

    def _set_mesh(self, verts, faces, face_colors) -> None:
        """Replace the single GLMeshItem with new data."""
        if verts.shape[0] == 0:
            self.clear_ellipsoids()
            return

        md = gl.MeshData(vertexes=verts, faces=faces, faceColors=face_colors)

        if self._item is not None:
            self._item.setMeshData(meshdata=md)
        else:
            from OpenGL import GL as _GL
            self._item = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                drawEdges=False,
                drawFaces=True,
            )
            # Custom GL state: translucent blending + back-face culling.
            # - GL_BLEND + blendFuncSeparate gives the transparency.
            # - GL_DEPTH_TEST keeps correct front-to-back ordering.
            # - GL_CULL_FACE removes interior / back-facing triangles
            #   that caused the see-through artifacts.
            self._item.setGLOptions({
                _GL.GL_DEPTH_TEST: True,
                _GL.GL_BLEND: True,
                _GL.GL_CULL_FACE: True,
                'glBlendFuncSeparate': (
                    _GL.GL_SRC_ALPHA, _GL.GL_ONE_MINUS_SRC_ALPHA,
                    _GL.GL_ONE, _GL.GL_ONE_MINUS_SRC_ALPHA,
                ),
            })
            self._view.addItem(self._item)

