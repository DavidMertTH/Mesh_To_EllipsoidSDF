"""
viewer3d.py — 3-D scene viewers for meshes and ellipsoids.

Hierarchy:
  _BaseViewer          – shared setup (GL widget, grid, axis)
  MeshViewer3D         – shows a single triangle mesh
  EllipsoidViewer3D    – shows a set of ellipsoid meshes with per-ellipsoid colours
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyqtgraph.opengl as gl

from widgets import DropGLView


# ── Colour palette for ellipsoids ─────────────────────────────────────────────

ELLIPSOID_PALETTE = [
    (242 / 255, 230 / 255,  65 / 255, 0.70),   # gold (edge colour)
    ( 73 / 255,  98 / 255, 242 / 255, 0.70),   # blue (mesh face colour)
    (242 / 255, 213 / 255,  65 / 255, 0.70),   # warm gold
    ( 24 / 255,  40 / 255,  89 / 255, 0.85),   # deep navy
    (140 / 255, 160 / 255, 242 / 255, 0.70),   # light periwinkle
    (200 / 255, 195 / 255,  55 / 255, 0.70),   # muted olive-gold
    ( 50 / 255,  70 / 255, 160 / 255, 0.75),   # mid blue
    (220 / 255, 220 / 255, 180 / 255, 0.65),   # pale warm white
]


# ── Base viewer ───────────────────────────────────────────────────────────────

class _BaseViewer:
    """Common setup: GL widget with grid and axis."""

    BG_COLOR = (2, 11, 13, 255)

    def __init__(self):
        self._view = DropGLView()
        self._view.setBackgroundColor(self.BG_COLOR)
        self._view.setCameraPosition(distance=3.0, elevation=15, azimuth=45)

        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        self._view.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self._view.addItem(axis)

    @property
    def widget(self) -> DropGLView:
        return self._view


# ── Mesh viewer ───────────────────────────────────────────────────────────────

class MeshViewer3D(_BaseViewer):
    """Shows a single triangle mesh."""

    MESH_FACE_COLOR = (73 / 256, 98 / 256, 242 / 256, 1.0)
    MESH_EDGE_COLOR = (242 / 256, 230 / 256, 65 / 256, 1.0)

    def __init__(self):
        super().__init__()
        self._mesh_item: Optional[gl.GLMeshItem] = None

    def show_mesh(self, verts: np.ndarray, faces: np.ndarray) -> None:
        self.clear_mesh()
        self._mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            color=self.MESH_FACE_COLOR,
            smooth=False,
            drawEdges=True,
            drawFaces=True,
            edgeColor=self.MESH_EDGE_COLOR,
        )
        self._view.addItem(self._mesh_item)

    def clear_mesh(self) -> None:
        if self._mesh_item is not None:
            self._view.removeItem(self._mesh_item)
            self._mesh_item = None


# ── Ellipsoid viewer ──────────────────────────────────────────────────────────

class EllipsoidViewer3D(_BaseViewer):
    def __init__(self):
        super().__init__()
        self._items: List[gl.GLMeshItem] = []

    def show_ellipsoids(self, ellipsoid_set) -> None:
        """
        Generate and display meshes for every ellipsoid in *ellipsoid_set*.
        """
        self.clear_ellipsoids()

        meshes = ellipsoid_set.generate_meshes(subdivisions=3)

        for i, mesh in enumerate(meshes):
            color = ELLIPSOID_PALETTE[i % len(ELLIPSOID_PALETTE)]
            item = gl.GLMeshItem(
                vertexes=mesh.vertices,
                faces=mesh.faces,
                color=color,
                smooth=True,
                drawEdges=False,
                drawFaces=True,
            )
            self._view.addItem(item)
            self._items.append(item)

    def clear_ellipsoids(self) -> None:
        for item in self._items:
            self._view.removeItem(item)
        self._items.clear()