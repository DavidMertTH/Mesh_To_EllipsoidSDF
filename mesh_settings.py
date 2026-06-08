"""
mesh_settings.py — panel for per-mesh adjustments fed into the SDF pipeline.

Two controls:
  * Rotation (X/Y/Z degrees) — slider AND a number field per axis; rotates the
    loaded mesh.  For a rigged FBX the host applies the rotation across every
    pose (and the skeleton) and re-computes the SDF asynchronously.
  * SDF Blowup — a single offset added uniformly to the SDF at every voxel
    (positive erodes / surface inward, negative dilates).  Shown live in the SDF
    slice and baked into the fit target.
  * Mesh Blowup — an exploded view of the per-bone region submeshes (the carving
    used by Bone-Separation mode): a toggle plus a slider that pushes each bone's
    submesh radially outward so the regions can be inspected/verified.  Only
    available when a rigged mesh is loaded.

Pure UI: emits ``rotationChanged(rx, ry, rz)`` (degrees), ``blowupChanged(voxels)``,
``regionPreviewToggled(on)`` and ``regionBlowupChanged(factor)``; the MainWindow
does the actual work.
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
    regionPreviewToggled = QtCore.Signal(bool)             # exploded preview on/off
    regionBlowupChanged = QtCore.Signal(float)             # explosion factor

    _BLOWUP_STEPS = 10          # slider int → /10 voxels
    _BLOWUP_RANGE = 100         # ±10.0 voxels
    _REGION_STEPS = 100         # slider int → /100 explosion factor
    _REGION_RANGE = 300         # 0.0 … 3.0× explosion

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

        # ── Mesh blowup group (exploded per-bone region preview) ──
        self._region_box = QtWidgets.QGroupBox("Mesh Blowup (Bone Regions)")
        self._region_box.setToolTip(
            "Explode the per-bone region submeshes (the Bone-Separation carving)\n"
            "to verify them: each bone's region is pushed radially outward and\n"
            "tinted a distinct colour.  Requires a rigged mesh.")
        gform = QtWidgets.QFormLayout(self._region_box)
        self._chk_region = QtWidgets.QCheckBox("Preview bone regions (explode)")
        self._chk_region.toggled.connect(self._on_region_toggled)
        gform.addRow(self._chk_region)
        self._region = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._region.setRange(0, self._REGION_RANGE)
        self._region.setValue(0)
        self._region.setToolTip(
            "How far to push each bone region apart (0 = in place).")
        self._lbl_region = QtWidgets.QLabel("0.00×")
        self._lbl_region.setMinimumWidth(56)
        self._region.valueChanged.connect(self._on_region_blowup)
        grow = QtWidgets.QHBoxLayout()
        grow.addWidget(self._region)
        grow.addWidget(self._lbl_region)
        gform.addRow("Spread:", grow)
        self._region_box.setEnabled(False)   # until a rigged mesh is loaded
        outer.addWidget(self._region_box)
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

    def _on_region_toggled(self, on: bool) -> None:
        self._region.setEnabled(on)
        self.regionPreviewToggled.emit(bool(on))

    def _on_region_blowup(self, v: int) -> None:
        factor = v / float(self._REGION_STEPS)
        self._lbl_region.setText(f"{factor:.2f}×")
        self.regionBlowupChanged.emit(factor)

    # ── public API ──

    def rotation_deg(self) -> tuple[float, float, float]:
        return (float(self._rot_spins["X"].value()),
                float(self._rot_spins["Y"].value()),
                float(self._rot_spins["Z"].value()))

    def blowup_voxels(self) -> float:
        return self._blowup.value() / float(self._BLOWUP_STEPS)

    def region_preview_enabled(self) -> bool:
        return self._chk_region.isChecked()

    def region_blowup(self) -> float:
        return self._region.value() / float(self._REGION_STEPS)

    def set_region_available(self, available: bool) -> None:
        """Enable/disable the Mesh-Blowup group (rigged meshes only).

        Disabling also switches the preview off (emitting ``regionPreviewToggled``
        so the host can drop any active preview) and zeroes the spread.
        """
        if not available and self._chk_region.isChecked():
            self._chk_region.setChecked(False)   # emits toggled(False)
        self._region_box.setEnabled(bool(available))
        if not available:
            self._region.blockSignals(True)
            self._region.setValue(0)
            self._region.blockSignals(False)
            self._lbl_region.setText("0.00×")

    def set_blowup_voxels(self, vox: float) -> None:
        """Set the blowup slider without emitting (caller updates the viewer)."""
        v = int(round(float(vox) * self._BLOWUP_STEPS))
        v = max(-self._BLOWUP_RANGE, min(self._BLOWUP_RANGE, v))
        self._blowup.blockSignals(True)
        self._blowup.setValue(v)
        self._blowup.blockSignals(False)
        self._lbl_blowup.setText(f"{v / float(self._BLOWUP_STEPS):+.1f} vox")

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
        self._chk_region.blockSignals(True); self._chk_region.setChecked(False); self._chk_region.blockSignals(False)
        self._region.blockSignals(True); self._region.setValue(0); self._region.blockSignals(False)
        self._region.setEnabled(False)
        self._lbl_region.setText("0.00×")
