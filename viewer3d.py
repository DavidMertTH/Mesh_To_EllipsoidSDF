from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph.opengl as gl

from widgets import DropGLView


class MeshViewer3D:
    BG_COLOR = (2, 11, 13, 255)
    MESH_FACE_COLOR = (73 / 256, 98 / 256, 242 / 256, 1.0)
    MESH_EDGE_COLOR = (242 / 256, 230 / 256, 65 / 256, 1.0)

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

        self._mesh_item: Optional[gl.GLMeshItem] = None

    # ── public ────────────────────────────────────────────────────────────

    @property
    def widget(self) -> DropGLView:
        return self._view

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
