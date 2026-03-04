"""
widgets.py — Reusable Qt widgets for the SDF viewer application.

  - DropGLView:  GLViewWidget that accepts file drag-and-drop.
  - SdfSlicePanel: Right-side panel showing an XY slice of the SDF grid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from sdf_colormap import make_sdf_lut


# ── 3-D viewport with drag-and-drop ──────────────────────────────────────────

class DropGLView(gl.GLViewWidget):
    """GLViewWidget that emits *fileDropped(str)* when a file is dropped on it."""

    fileDropped = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAcceptDrops(True)

    # Qt overrides ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.fileDropped.emit(path)


# ── SDF slice panel ──────────────────────────────────────────────────────────

class SdfSlicePanel(QtWidgets.QWidget):
    """
    Panel that displays an XY slice through a 3-D SDF grid.

    Signals:
        computeRequested(int)  – emitted when the user clicks Compute.
                                 Carries the requested grid resolution *n*.
    """

    computeRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lut = make_sdf_lut()
        self._sdf_grid: Optional[np.ndarray] = None
        self._build_ui()

    # ── public API ────────────────────────────────────────────────────────

    def set_sdf(self, grid: np.ndarray) -> None:
        """
        Provide a new SDF volume (nz, ny, nx) and refresh the view.
        The Z slider is automatically adjusted.
        """
        self._sdf_grid = grid
        n = grid.shape[0]

        self.slider_z.blockSignals(True)
        self.slider_z.setRange(0, n - 1)
        self.slider_z.setValue(n // 2)
        self.slider_z.setSingleStep(1)
        self.slider_z.setPageStep(max(1, n // 32))
        self.slider_z.blockSignals(False)

        self._update_slice()

    @property
    def requested_n(self) -> int:
        return int(self.spin_n.value())

    # ── internal ──────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("SDF XY Slice")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Controls
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(16, 512)
        self.spin_n.setValue(128)
        self.spin_n.setSingleStep(16)

        self.btn_compute = QtWidgets.QPushButton("Compute (G)")
        self.btn_compute.clicked.connect(self._on_compute_clicked)

        form.addRow("Grid n:", self.spin_n)
        form.addRow("", self.btn_compute)
        layout.addLayout(form)

        # Image view
        self.img_xy = pg.ImageView()
        self.img_xy.ui.roiBtn.hide()
        self.img_xy.ui.menuBtn.hide()
        self.img_xy.ui.histogram.gradient.setVisible(False)

        layout.addWidget(QtWidgets.QLabel("XY (Z fixed)"))
        layout.addWidget(self.img_xy, 1)

        # Z slider
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_z.setRange(0, 127)
        self.slider_z.setValue(64)
        self.slider_z.valueChanged.connect(self._update_slice)
        layout.addWidget(self.slider_z)

        # Apply LUT once
        self._apply_lut()

    def _apply_lut(self):
        self.img_xy.getImageItem().setLookupTable(self._lut)
        self.img_xy.getImageItem().setAutoDownsample(True)

    def _update_slice(self):
        if self._sdf_grid is None:
            return
        n = self._sdf_grid.shape[0]
        iz = max(0, min(n - 1, int(self.slider_z.value())))
        xy = self._sdf_grid[iz, :, :]
        self.img_xy.setImage(xy.T, autoLevels=False)
        self.img_xy.setLevels(-0.2, 0.2)
        self._apply_lut()

    def _on_compute_clicked(self):
        self.computeRequested.emit(self.requested_n)