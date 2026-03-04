from __future__ import annotations

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import warp as wp
from mesh_io import load_and_prepare
from sdf_compute import SdfComputer
from viewer3d import MeshViewer3D
from widgets import SdfSlicePanel


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Warp Mesh Viewer + SDF XY Slice (Drag & Drop)")

        # ── Warp ──────────────────────────────────────────────────────────
        wp.init()
        self._device = "cuda" if wp.is_cuda_available() else "cpu"

        # ── Domain objects ────────────────────────────────────────────────
        self._sdf = SdfComputer(device=self._device)

        # ── UI components ─────────────────────────────────────────────────
        self._viewer = MeshViewer3D()
        self._sdf_panel = SdfSlicePanel()

        self._build_layout()
        self._build_toolbar()
        self._connect_signals()

        # ── Status bar ────────────────────────────────────────────────────
        self._status = self.statusBar()
        self._status.showMessage("Drag & drop a mesh. SDF will be computed automatically.")

    # ── layout / toolbar ──────────────────────────────────────────────────

    def _build_layout(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._viewer.widget)
        splitter.addWidget(self._sdf_panel)
        splitter.setSizes([650, 650])
        self.setCentralWidget(splitter)

    def _build_toolbar(self):
        act_compute = QtGui.QAction("Compute SDF Grid (G)", self)
        act_compute.setShortcut(QtGui.QKeySequence("G"))
        act_compute.triggered.connect(self._on_compute)

        tb = self.addToolBar("Tools")
        tb.addAction(act_compute)
        self.addAction(act_compute)

    def _connect_signals(self):
        self._viewer.widget.fileDropped.connect(self._on_file_dropped)
        self._sdf_panel.computeRequested.connect(self._on_compute)

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_file_dropped(self, path: str):
        try:
            mesh = load_and_prepare(path, target_scale=1.0)

            verts: np.ndarray = mesh.vertices.view(np.ndarray)
            faces: np.ndarray = mesh.faces.view(np.ndarray)

            self._viewer.show_mesh(verts, faces)
            self._sdf.set_mesh(verts, faces)

            self._status.showMessage(
                f"Loaded: {path} | verts={len(verts)} faces={len(faces)} | device={self._device}"
            )

            # auto-compute on load
            self._on_compute()

        except Exception as e:
            self._status.showMessage(f"Failed to load: {path} ({e})")

    def _on_compute(self, n: int | None = None):
        if not self._sdf.is_ready:
            self._status.showMessage("Load a mesh first.")
            return

        if n is None:
            n = self._sdf_panel.requested_n

        self._status.showMessage(f"Computing SDF grid n={n} on {self._device} …")
        try:
            result = self._sdf.compute_voxel_grid(n=n)
        except Exception as e:
            self._status.showMessage(f"Compute failed: {e}")
            return

        self._sdf_panel.set_sdf(result.grid)

        self._status.showMessage(
            f"SDF done: shape={result.grid.shape} dx={result.dx:.6f} "
            f"min={float(np.min(result.grid)):.6f} max={float(np.max(result.grid)):.6f}"
        )
