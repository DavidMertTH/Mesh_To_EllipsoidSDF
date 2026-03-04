"""
main_window.py — Application main window with 2×2 layout.

  ┌─────────────────────┬─────────────────────┐
  │  Mesh 3-D Viewer    │  Mesh SDF Slice     │
  │  (top-left)         │  (top-right)        │
  ├─────────────────────┼─────────────────────┤
  │  Ellipsoid 3-D      │  Ellipsoid SDF      │
  │  Viewer (bot-left)  │  Slice (bot-right)  │
  └─────────────────────┴─────────────────────┘

Top row:    loaded mesh → SDF from Warp mesh queries
Bottom row: ellipsoid set → analytical SDF (Ínigo Quílez approx.)
Both SDF panels share the same AABB / grid so slices are directly comparable.
"""

from __future__ import annotations

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import warp as wp

from mesh_io import load_and_prepare
from sdf_compute import SdfComputer, SdfResult
from ellipsoid import EllipsoidSet, create_demo_ellipsoids
from viewer3d import MeshViewer3D, EllipsoidViewer3D
from widgets import SdfSlicePanel


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh → Ellipsoid SDF Approximation")

        wp.init()
        self._device = "cuda" if wp.is_cuda_available() else "cpu"

        self._sdf = SdfComputer(device=self._device)
        self._ellipsoids: EllipsoidSet = create_demo_ellipsoids(device=self._device)

        self._last_mesh_result: SdfResult | None = None


        self._mesh_viewer = MeshViewer3D()
        self._mesh_sdf_panel = SdfSlicePanel()

        self._ell_viewer = EllipsoidViewer3D()
        self._ell_sdf_panel = SdfSlicePanel()

        self._build_layout()
        self._build_toolbar()
        self._connect_signals()

        self._status = self.statusBar()
        self._status.showMessage(
            "Drag & drop a mesh onto the top-left view. "
            "SDF + demo ellipsoids will be computed automatically."
        )

        self._ell_viewer.show_ellipsoids(self._ellipsoids)


    def _build_layout(self):
        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top_splitter.addWidget(self._mesh_viewer.widget)
        top_splitter.addWidget(self._mesh_sdf_panel)
        top_splitter.setSizes([650, 650])

        bot_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bot_splitter.addWidget(self._ell_viewer.widget)
        bot_splitter.addWidget(self._ell_sdf_panel)
        bot_splitter.setSizes([650, 650])

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bot_splitter)
        main_splitter.setSizes([450, 450])

        self.setCentralWidget(main_splitter)

    def _build_toolbar(self):
        act_compute = QtGui.QAction("Compute SDF Grid (G)", self)
        act_compute.setShortcut(QtGui.QKeySequence("G"))
        act_compute.triggered.connect(self._on_compute_all)

        tb = self.addToolBar("Tools")
        tb.addAction(act_compute)
        self.addAction(act_compute)

    def _connect_signals(self):
        self._mesh_viewer.widget.fileDropped.connect(self._on_file_dropped)

        self._mesh_sdf_panel.computeRequested.connect(self._on_compute_all)
        self._ell_sdf_panel.computeRequested.connect(self._on_compute_all)

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_file_dropped(self, path: str):
        try:
            mesh = load_and_prepare(path, target_scale=1.0)

            verts: np.ndarray = mesh.vertices.view(np.ndarray)
            faces: np.ndarray = mesh.faces.view(np.ndarray)

            self._mesh_viewer.show_mesh(verts, faces)
            self._sdf.set_mesh(verts, faces)

            self._status.showMessage(
                f"Loaded: {path} | verts={len(verts)} faces={len(faces)} | device={self._device}"
            )

            self._on_compute_all()

        except Exception as e:
            self._status.showMessage(f"Failed to load: {path} ({e})")

    def _on_compute_all(self, n: int | None = None):
        if not self._sdf.is_ready:
            self._status.showMessage("Load a mesh first.")
            return

        if n is None:
            n = self._mesh_sdf_panel.requested_n

        # ── 1. Mesh SDF ──────────────────────────────────────────────────
        self._status.showMessage(f"Computing mesh SDF (n={n}) on {self._device} …")
        try:
            mesh_result = self._sdf.compute_voxel_grid(n=n)
        except Exception as e:
            self._status.showMessage(f"Mesh SDF failed: {e}")
            return

        self._last_mesh_result = mesh_result
        self._mesh_sdf_panel.set_sdf(mesh_result.grid)

        self._status.showMessage(f"Computing ellipsoid SDF (n={n}) on {self._device} …")
        try:
            ell_grid = self._ellipsoids.compute_sdf_grid(
                origin=mesh_result.origin,
                dx=mesh_result.dx,
                n=n,
            )
        except Exception as e:
            self._status.showMessage(f"Ellipsoid SDF failed: {e}")
            return

        self._ell_sdf_panel.set_sdf(ell_grid)

        self._ell_viewer.show_ellipsoids(self._ellipsoids)

        self._status.showMessage(
            f"Done — mesh SDF min={float(np.min(mesh_result.grid)):.4f} "
            f"max={float(np.max(mesh_result.grid)):.4f}  |  "
            f"ellipsoid SDF min={float(np.min(ell_grid)):.4f} "
            f"max={float(np.max(ell_grid)):.4f}"
        )


    def update_ellipsoids(self, ellipsoid_set: EllipsoidSet) -> None:

        self._ellipsoids = ellipsoid_set

        self._ell_viewer.show_ellipsoids(self._ellipsoids)

        if self._last_mesh_result is not None:
            r = self._last_mesh_result
            ell_grid = self._ellipsoids.compute_sdf_grid(
                origin=r.origin,
                dx=r.dx,
                n=r.n,
            )
            self._ell_sdf_panel.set_sdf(ell_grid)