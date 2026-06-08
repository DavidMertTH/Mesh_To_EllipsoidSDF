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

import theme
from sdf_colormap import make_sdf_lut, colorize_sdf_slice


# ── wheel-scroll guard ───────────────────────────────────────────────────────

class WheelGuard(QtCore.QObject):
    """Application-wide event filter that stops the mouse wheel from editing
    value widgets (spin boxes, combo boxes, sliders).

    Scrolling a settings panel would otherwise accidentally change whatever
    field the cursor happens to be over.  Instead the wheel event is forwarded
    to the enclosing scroll area so the panel keeps scrolling, while the
    widget's value is left untouched.  Open combo-box pop-ups are unaffected
    (their list view is a separate widget), and so are the scroll bars
    themselves (``QScrollBar`` is not in the target list).
    """

    # QSlider, not QAbstractSlider, so we never swallow QScrollBar wheel events.
    _TARGETS = (QtWidgets.QAbstractSpinBox,
                QtWidgets.QComboBox,
                QtWidgets.QSlider)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Wheel and isinstance(obj, self._TARGETS):
            area = self._scroll_area(obj)
            if area is not None:
                # Re-dispatch the scroll to the panel so it still scrolls.
                QtWidgets.QApplication.sendEvent(area.viewport(), ev)
            return True  # swallow it for the value widget itself
        return False

    @staticmethod
    def _scroll_area(widget):
        p = widget.parentWidget()
        while p is not None:
            if isinstance(p, QtWidgets.QAbstractScrollArea):
                return p
            p = p.parentWidget()
        return None


# ── 3-D viewport with drag-and-drop ──────────────────────────────────────────

class DropGLView(gl.GLViewWidget):
    """GLViewWidget that emits *fileDropped(str)* when a file is dropped on it.

    Also supports floating *corner overlays*: child widgets pinned to a corner
    of the viewport that stay anchored while the view is resized.  Used for the
    in-viewport display/render menus.
    """

    fileDropped = QtCore.Signal(str)

    # Continuous-motion steps per ~60 fps tick.  WASD and the arrow keys share
    # the same actions: zoom (W/S, ↑/↓) and pan (A/D, ←/→).
    _PAN_PIXELS = 12.0       # screen pixels — pan() scales px → world
    _ZOOM_FACTOR = 0.97      # distance multiplier per tick

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAcceptDrops(True)
        # (widget, corner, margin) — corner ∈ {'tl','tr','bl','br'}
        self._overlays: list = []

        # ── keyboard navigation (WASD = move, arrows = orbit) ──
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._nav_keys: set = set()
        self._nav_timer = QtCore.QTimer(self)
        self._nav_timer.setInterval(16)        # ~60 fps
        self._nav_timer.timeout.connect(self._nav_step)

    # ── keyboard navigation ─────────────────────────────────────────────────

    @property
    def _nav_set(self):
        Q = QtCore.Qt
        return {Q.Key_W, Q.Key_A, Q.Key_S, Q.Key_D,
                Q.Key_Up, Q.Key_Down, Q.Key_Left, Q.Key_Right}

    def keyPressEvent(self, ev):
        if ev.key() in self._nav_set:
            if not ev.isAutoRepeat():
                self._nav_keys.add(ev.key())
                if not self._nav_timer.isActive():
                    self._nav_timer.start()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() in self._nav_set:
            if not ev.isAutoRepeat():
                self._nav_keys.discard(ev.key())
                if not self._nav_keys:
                    self._nav_timer.stop()
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def focusOutEvent(self, ev):
        # Drop held keys when focus leaves so motion doesn't get stuck.
        self._nav_keys.clear()
        self._nav_timer.stop()
        super().focusOutEvent(ev)

    def _nav_step(self):
        keys = self._nav_keys
        if not keys:
            self._nav_timer.stop()
            return
        Q = QtCore.Qt

        # Arrow keys mirror WASD: ↑/↓ = W/S, ←/→ = A/D.
        zoom_in = Q.Key_W in keys or Q.Key_Up in keys
        zoom_out = Q.Key_S in keys or Q.Key_Down in keys
        pan_right = Q.Key_D in keys or Q.Key_Right in keys
        pan_left = Q.Key_A in keys or Q.Key_Left in keys

        # W / S (or ↑/↓) → dolly (zoom in/out by scaling the camera distance).
        dist = float(self.opts.get('distance', 10.0))
        new_dist = dist
        if zoom_in:
            new_dist *= self._ZOOM_FACTOR
        if zoom_out:
            new_dist /= self._ZOOM_FACTOR
        if new_dist != dist:
            self.setCameraPosition(distance=max(new_dist, 1e-3))

        # A / D (or ←/→) → pan the target horizontally (pan() interprets dx in
        # pixels and scales to world units by the current distance internally).
        # A/← pan the view left, D/→ pan right.
        px = (self._PAN_PIXELS if pan_left else 0.0) \
            - (self._PAN_PIXELS if pan_right else 0.0)
        if px:
            self.pan(px, 0.0, 0.0, relative='view-upright')

    # ── corner overlays ─────────────────────────────────────────────────────

    def add_corner_widget(self, widget, corner: str = "tl", margin: int = 8) -> None:
        """Pin *widget* to a corner of the viewport as a floating overlay."""
        widget.setParent(self)
        widget.raise_()
        widget.show()
        self._overlays.append((widget, corner, margin))
        self._reposition_overlays()

    def _reposition_overlays(self) -> None:
        w, h = self.width(), self.height()
        for widget, corner, margin in self._overlays:
            sz = widget.sizeHint()
            ww, wh = sz.width(), sz.height()
            if corner == "fill":
                # Cover the whole viewport (e.g. the raymarch image overlay).
                widget.resize(w, h)
                widget.move(0, 0)
                continue
            if corner == "bottom":
                # Stretch across the bottom edge (e.g. the slice position slider).
                ww = max(ww, w - 2 * margin)
                widget.resize(ww, wh)
                widget.move(margin, max(margin, h - wh - margin))
                continue
            x = margin if corner in ("tl", "bl") else max(margin, w - ww - margin)
            y = margin if corner in ("tl", "tr") else max(margin, h - wh - margin)
            widget.resize(sz)
            widget.move(x, y)

    # Qt overrides ─────────────────────────────────────────────────────────

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reposition_overlays()

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

    def __init__(self, parent=None, default_n: int = 512):
        super().__init__(parent)
        self._lut = make_sdf_lut()
        self._sdf_grid: Optional[np.ndarray] = None
        self._dx: float = 1.0                 # voxel size (for the exterior band)
        # Default grid resolution.  The host lowers this (→ 64) when no CUDA GPU
        # is present, since the n³ SDF runs on the CPU there and 512³ is far too
        # heavy.  Not persisted, so it is re-applied per session.
        self._default_n = int(default_n)
        self._build_ui()

    # ── public API ────────────────────────────────────────────────────────

    def set_sdf(self, grid: np.ndarray, dx: float | None = None) -> None:
        """
        Provide a new SDF volume (nz, ny, nx) and refresh the view.
        The Z slider is automatically adjusted.  ``dx`` (voxel size) sets the
        exterior fade band so the colouring matches the 3-D slice.
        """
        self._sdf_grid = grid
        if dx is not None:
            self._dx = max(float(dx), 1e-9)
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
        # Inherit the theme's text colour (palette) so the heading stays legible
        # in dark mode — a bare stylesheet without a colour falls back to black.
        title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: palette(window-text);")
        layout.addWidget(title)

        # Controls
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(16, 2_000_000_000)   # no practical upper cap
        self.spin_n.setValue(self._default_n)
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
        self.img_xy.ui.histogram.hide()          # we pre-colour; no histogram
        self.img_xy.getImageItem().setAutoDownsample(True)

        layout.addWidget(QtWidgets.QLabel("XY (Z fixed)"))
        layout.addWidget(self.img_xy, 1)

        # Z slider
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_z.setRange(0, 127)
        self.slider_z.setValue(64)
        self.slider_z.valueChanged.connect(self._update_slice)
        layout.addWidget(self.slider_z)

        self.apply_theme()

    def apply_theme(self):
        """Recolour the SDF slice for the current theme.

        Rebuilds the colormap LUT (its dark/light extreme and brand colours
        follow the theme) and re-renders; the slice is pre-coloured into an RGBA
        image (see _update_slice), so no ImageView LUT/levels are involved.
        """
        self.img_xy.ui.graphicsView.setBackground(theme.bg((0, 0, 0)))
        self._lut = make_sdf_lut()
        self._update_slice()

    def _update_slice(self):
        if self._sdf_grid is None:
            return
        n = self._sdf_grid.shape[0]
        iz = max(0, min(n - 1, int(self.slider_z.value())))
        slice2d = np.ascontiguousarray(self._sdf_grid[iz, :, :].T)   # [a0, a1]

        # Colour exactly like the 3-D slice: interior blends surface->deepest
        # across the whole interior (full colour only at the deepest point),
        # exterior fades within a few voxels.  depth is per-volume (deepest
        # interior magnitude of the WHOLE grid, so the colour scale is stable
        # while scrolling through Z).
        depth = max(-float(self._sdf_grid.min()), 1e-4)
        out_band = 3.0 * float(self._dx)
        rgba = colorize_sdf_slice(slice2d, self._lut, depth, out_band)
        # Pre-coloured RGBA → must be shown RAW.  ImageView otherwise applies a
        # LUT and (worse) levels=[0,1] to the uint8 image, scaling every channel
        # value >=1 up to 255 == washed-out white.  Use an identity 0..255 level
        # so the colours come through exactly as computed.  (levels=None would
        # crash newer pyqtgraph once autoDownsample turns the uint8 image into
        # float on render: makeARGB demands levels for float input.)
        item = self.img_xy.getImageItem()
        item.setLookupTable(None)
        self.img_xy.setImage(rgba, autoLevels=False, autoHistogramRange=False,
                             levels=(0, 255))

    def _on_compute_clicked(self):
        self.computeRequested.emit(self.requested_n)