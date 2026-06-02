"""
mesh_settings.py — panel for per-mesh adjustments fed into the SDF pipeline.

Two controls:
  * Rotation (X/Y/Z degrees) — slider AND a number field per axis; rotates the
    loaded mesh.  For a rigged FBX the host applies the rotation across every
    pose (and the skeleton) and re-computes the SDF asynchronously.
  * SDF Blowup — a single offset added uniformly to the SDF at every voxel
    (positive erodes / surface inward, negative dilates).  Shown live in the SDF
    slice and baked into the fit target.

Pure UI: emits ``rotationChanged(rx, ry, rz)`` (degrees) and
``blowupChanged(voxels)``; the MainWindow does the actual work.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets


def rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Z·Y·X intrinsic Euler rotation (degrees) → 3×3 matrix (float32)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def rotate_mesh(verts: np.ndarray, rx: float, ry: float, rz: float,
                center=None) -> np.ndarray:
    """Rotate ``verts`` (V,3) by the Euler angles about ``center``.

    ``center=None`` uses the bounding-box centre of ``verts``; pass a FIXED
    centre (e.g. the bind-pose centre) to keep the pivot stable across poses.
    """
    v = np.asarray(verts, dtype=np.float32)
    if v.size == 0 or (rx == 0.0 and ry == 0.0 and rz == 0.0):
        return v.copy()
    c = (0.5 * (v.min(axis=0) + v.max(axis=0)) if center is None
         else np.asarray(center, dtype=np.float32))
    R = rotation_matrix(rx, ry, rz)
    return ((v - c) @ R.T + c).astype(np.float32)


class MeshSettingsPanel(QtWidgets.QWidget):
    """Rotation (X/Y/Z, slider + number) + SDF blowup controls for the mesh."""

    rotationChanged = QtCore.Signal(float, float, float)   # degrees
    blowupChanged = QtCore.Signal(float)                   # voxels (× dx by viewer)

    _BLOWUP_STEPS = 10          # slider int → /10 voxels
    _BLOWUP_RANGE = 100         # ±10.0 voxels

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Rotation group ──
        rot_box = QtWidgets.QGroupBox("Rotation")
        rform = QtWidgets.QFormLayout(rot_box)
        self._rot_sliders: dict[str, QtWidgets.QSlider] = {}
        self._rot_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for axis in ("X", "Y", "Z"):
            sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sld.setRange(-180, 180)
            sld.setValue(0)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-180.0, 180.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix("°")
            spin.setFixedWidth(72)
            sld.valueChanged.connect(lambda _v, a=axis: self._sync_from_slider(a))
            spin.valueChanged.connect(lambda _v, a=axis: self._sync_from_spin(a))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(sld)
            row.addWidget(spin)
            rform.addRow(f"{axis}:", row)
            self._rot_sliders[axis] = sld
            self._rot_spins[axis] = spin
        self._btn_reset_rot = QtWidgets.QPushButton("Reset rotation")
        self._btn_reset_rot.clicked.connect(self.reset_rotation)
        rform.addRow(self._btn_reset_rot)
        outer.addWidget(rot_box)

        # ── SDF blowup group ──
        blow_box = QtWidgets.QGroupBox("SDF Blowup")
        bform = QtWidgets.QFormLayout(blow_box)
        self._blowup = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._blowup.setRange(-self._BLOWUP_RANGE, self._BLOWUP_RANGE)
        self._blowup.setValue(0)
        self._blowup.setToolTip(
            "Add a uniform offset (in voxels) to the SDF everywhere.\n"
            "Positive erodes (surface inward), negative dilates.\n"
            "Live in the slice and baked into the fit target.")
        self._lbl_blowup = QtWidgets.QLabel("0.0 vox")
        self._lbl_blowup.setMinimumWidth(56)
        self._blowup.valueChanged.connect(self._on_blowup)
        brow = QtWidgets.QHBoxLayout()
        brow.addWidget(self._blowup)
        brow.addWidget(self._lbl_blowup)
        bform.addRow("Offset:", brow)
        self._btn_reset_blowup = QtWidgets.QPushButton("Reset blowup")
        self._btn_reset_blowup.clicked.connect(self.reset_blowup)
        bform.addRow(self._btn_reset_blowup)
        outer.addWidget(blow_box)
        outer.addStretch(1)

    # ── rotation sync (slider ↔ number) ──

    def _sync_from_slider(self, axis: str) -> None:
        spin = self._rot_spins[axis]
        spin.blockSignals(True)
        spin.setValue(float(self._rot_sliders[axis].value()))
        spin.blockSignals(False)
        self._emit_rotation()

    def _sync_from_spin(self, axis: str) -> None:
        sld = self._rot_sliders[axis]
        sld.blockSignals(True)
        sld.setValue(int(round(self._rot_spins[axis].value())))
        sld.blockSignals(False)
        self._emit_rotation()

    def _emit_rotation(self) -> None:
        self.rotationChanged.emit(*self.rotation_deg())

    def _on_blowup(self, v: int) -> None:
        vox = v / float(self._BLOWUP_STEPS)
        self._lbl_blowup.setText(f"{vox:+.1f} vox")
        self.blowupChanged.emit(vox)

    # ── public API ──

    def rotation_deg(self) -> tuple[float, float, float]:
        return (float(self._rot_spins["X"].value()),
                float(self._rot_spins["Y"].value()),
                float(self._rot_spins["Z"].value()))

    def blowup_voxels(self) -> float:
        return self._blowup.value() / float(self._BLOWUP_STEPS)

    def reset_rotation(self) -> None:
        for a in ("X", "Y", "Z"):
            for w in (self._rot_sliders[a], self._rot_spins[a]):
                w.blockSignals(True); w.setValue(0); w.blockSignals(False)
        self._emit_rotation()

    def reset_blowup(self) -> None:
        self._blowup.blockSignals(True)
        self._blowup.setValue(0)
        self._blowup.blockSignals(False)
        self._lbl_blowup.setText("0.0 vox")
        self.blowupChanged.emit(0.0)

    def reset(self) -> None:
        """Reset both controls without emitting (e.g. on loading a new mesh)."""
        for a in ("X", "Y", "Z"):
            for w in (self._rot_sliders[a], self._rot_spins[a]):
                w.blockSignals(True); w.setValue(0); w.blockSignals(False)
        self._blowup.blockSignals(True); self._blowup.setValue(0); self._blowup.blockSignals(False)
        self._lbl_blowup.setText("0.0 vox")
