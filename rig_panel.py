"""
rig_panel.py — Modular UI panel for rigged humanoid workflow.

Provides the RigModePanel widget which integrates into the main toolbar area.
When disabled, the entire rig workflow is invisible and the static mesh
pipeline works exactly as before.

Features:
  - Toggle: Rig Mode on/off
  - Pose selector / timeline slider
  - Multi-pose training controls
  - Bone assignment visualisation
  - Per-pose loss display
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from PySide6 import QtCore, QtWidgets

from skeleton import Skeleton, Pose
from rig_loader import RiggedMesh, load_rigged_mesh, is_rigged_mesh
from skinning import deform_mesh
from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from ellipsoid import EllipsoidSet
import pose_library


class RigModePanel(QtWidgets.QGroupBox):
    """Collapsible panel for rig-mode controls.

    Signals
    -------
    rigLoaded(object)
        Emitted when a rigged mesh is loaded.  Payload: RiggedMesh.
    poseChanged(int)
        Emitted when the user selects a different pose index.
    multiPoseRequested()
        Emitted when the user clicks "Multi-Pose Fit".
    rigModeToggled(bool)
        Emitted when rig mode is toggled on/off.
    """

    rigLoaded = QtCore.Signal(object)
    poseChanged = QtCore.Signal(int)
    multiPoseRequested = QtCore.Signal()
    rigModeToggled = QtCore.Signal(bool)
    autoPipelineRequested = QtCore.Signal()
    exportUnityRequested = QtCore.Signal()

    # Sentinel stored as combo data for the FBX's own animation entry.
    NATIVE = -1

    def __init__(self, parent=None, pose_dir: Optional[Path] = None):
        super().__init__("Rig Mode", parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._on_toggle)

        self._rigged_mesh: Optional[RiggedMesh] = None
        self._mapper: Optional[BoneEllipsoidMapper] = None
        self._bone_local: Optional[BoneLocalEllipsoids] = None

        # Pose-library state: poses loaded from disk.
        self._pose_dir: Path = Path(pose_dir) if pose_dir \
            else pose_library.DEFAULT_POSE_DIR
        self._library_poses: List[Pose] = []

        self._build_ui()

    @property
    def is_active(self) -> bool:
        return self.isChecked()

    @property
    def rigged_mesh(self) -> Optional[RiggedMesh]:
        return self._rigged_mesh

    @property
    def mapper(self) -> Optional[BoneEllipsoidMapper]:
        return self._mapper

    @property
    def bone_local(self) -> Optional[BoneLocalEllipsoids]:
        return self._bone_local

    @property
    def current_pose_index(self) -> int:
        return self._slider_pose.value()

    @property
    def active_poses(self) -> List[Pose]:
        """The pose list currently driving the timeline.

        ``[native Animation]`` selected → the FBX's own animation frames.
        A saved pose selected → just that single (one-frame) pose.
        """
        data = self._cmb_source.currentData()
        if data is not None and data != self.NATIVE:
            if 0 <= data < len(self._library_poses):
                return [self._library_poses[data]]
            return [Pose.t_pose()]
        if self._rigged_mesh is not None:
            return self._rigged_mesh.poses
        return [Pose.t_pose()]

    def pose_at(self, idx: int) -> Pose:
        """Return the pose at *idx* in the active source (clamped to T-pose)."""
        poses = self.active_poses
        if 0 <= idx < len(poses):
            return poses[idx]
        return Pose.t_pose()

    @property
    def current_pose(self) -> Optional[Pose]:
        if self._rigged_mesh is None:
            return None
        return self.pose_at(self._slider_pose.value())

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Status ──
        self._lbl_status = QtWidgets.QLabel("No rigged mesh loaded")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._lbl_status)

        # ── Animation / pose source ──
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(QtWidgets.QLabel("Source:"))
        self._cmb_source = QtWidgets.QComboBox()
        self._cmb_source.setToolTip(
            "Play the animation baked into the FBX, or apply one of the "
            "saved single-frame poses from the poses/ folder."
        )
        self._cmb_source.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(self._cmb_source, 1)
        layout.addLayout(src_row)

        # ── Pose controls ──
        pose_row = QtWidgets.QHBoxLayout()
        pose_row.addWidget(QtWidgets.QLabel("Pose:"))

        self._slider_pose = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_pose.setRange(0, 0)
        self._slider_pose.setValue(0)
        self._slider_pose.setEnabled(False)
        self._slider_pose.valueChanged.connect(self._on_pose_changed)
        pose_row.addWidget(self._slider_pose, 1)

        self._lbl_pose = QtWidgets.QLabel("0 / 0")
        self._lbl_pose.setFixedWidth(80)
        pose_row.addWidget(self._lbl_pose)

        layout.addLayout(pose_row)

        # ── Pose name ──
        self._lbl_pose_name = QtWidgets.QLabel("")
        self._lbl_pose_name.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(self._lbl_pose_name)

        # ── Bone assignment info ──
        self._lbl_bones = QtWidgets.QLabel("")
        self._lbl_bones.setWordWrap(True)
        self._lbl_bones.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self._lbl_bones)

        # ── Pose library management ──
        lib_row = QtWidgets.QHBoxLayout()
        self._btn_save_pose = QtWidgets.QPushButton("💾 Save Pose")
        self._btn_save_pose.setToolTip(
            "Save the current pose to the pose library (poses/ folder)."
        )
        self._btn_save_pose.setEnabled(False)
        self._btn_save_pose.clicked.connect(self._on_save_pose_clicked)
        lib_row.addWidget(self._btn_save_pose)

        self._btn_reload_lib = QtWidgets.QPushButton("⟳ Reload Library")
        self._btn_reload_lib.setToolTip("Re-scan the poses/ folder from disk.")
        self._btn_reload_lib.setEnabled(False)
        self._btn_reload_lib.clicked.connect(lambda: self.reload_library())
        lib_row.addWidget(self._btn_reload_lib)
        layout.addLayout(lib_row)

        # ── Multi-pose training ──
        # No rig-specific knobs: steps + LR come from the right panel, all loss
        # and sampling settings from the Settings dialog — same as a normal fit.
        layout.addSpacing(8)
        _hint = QtWidgets.QLabel(
            "Multi-Pose Training uses the main Steps / LR and Settings dialog."
        )
        _hint.setWordWrap(True)
        _hint.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(_hint)

        # ── One-click auto pipeline ──
        self._btn_auto = QtWidgets.QPushButton("▶ Auto Fit All Poses")
        self._btn_auto.setToolTip(
            "Automatically: fit T-pose ellipsoids → assign to bones → train all poses"
        )
        self._btn_auto.setEnabled(False)
        self._btn_auto.setStyleSheet(
            "font-weight: bold; background-color: #2d7d32; color: white; padding: 4px;"
        )
        self._btn_auto.clicked.connect(lambda: self.autoPipelineRequested.emit())
        layout.addWidget(self._btn_auto)

        layout.addWidget(QtWidgets.QLabel("— or step-by-step —"))

        btn_row = QtWidgets.QHBoxLayout()

        self._btn_assign = QtWidgets.QPushButton("1. Assign to Bones")
        self._btn_assign.setToolTip(
            "Assign current ellipsoids to skeleton bones (run after T-pose fit)"
        )
        self._btn_assign.setEnabled(False)
        self._btn_assign.clicked.connect(self._on_assign_clicked)
        btn_row.addWidget(self._btn_assign)

        self._btn_multipose = QtWidgets.QPushButton("2. Multi-Pose Fit")
        self._btn_multipose.setToolTip(
            "Train bone-local params across all poses"
        )
        self._btn_multipose.setEnabled(False)
        self._btn_multipose.clicked.connect(self._on_multipose_clicked)
        btn_row.addWidget(self._btn_multipose)

        layout.addLayout(btn_row)

        # ── Unity export ──
        self._btn_export = QtWidgets.QPushButton("⬆ Export for Unity (.json)")
        self._btn_export.setToolTip(
            "Export bone-local ellipsoids to JSON for import into Unity"
        )
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(lambda: self.exportUnityRequested.emit())
        layout.addWidget(self._btn_export)

        # ── Progress ──
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        layout.addStretch()

    # ── Public methods ───────────────────────────────────────────────

    def set_rigged_mesh(self, rigged_mesh: RiggedMesh):
        """Set a loaded rigged mesh and update UI."""
        self._rigged_mesh = rigged_mesh
        self._mapper = BoneEllipsoidMapper(rigged_mesh.skeleton)
        self._bone_local = None

        # Load saved poses for this skeleton and rebuild the source dropdown,
        # defaulting to the FBX's own animation.
        self._library_poses = pose_library.load_all_poses(
            rigged_mesh.skeleton, self._pose_dir,
        )
        self._populate_source_combo(select=self.NATIVE)

        n_bones = rigged_mesh.skeleton.num_bones

        self._btn_assign.setEnabled(True)
        self._btn_auto.setEnabled(True)
        self._btn_save_pose.setEnabled(True)
        self._btn_reload_lib.setEnabled(True)

        self._lbl_status.setText(
            f"Rig: {rigged_mesh.mesh_name} | "
            f"{n_bones} bones | {len(rigged_mesh.poses)} FBX poses | "
            f"{len(self._library_poses)} saved"
        )
        self._lbl_status.setStyleSheet("color: green; font-size: 11px;")

        self._refresh_slider_for_source()

        self.rigLoaded.emit(rigged_mesh)

    def set_bone_local(self, bone_local: BoneLocalEllipsoids):
        """Update bone-local params (after assignment or training)."""
        self._bone_local = bone_local
        self._btn_multipose.setEnabled(True)
        self._btn_export.setEnabled(True)

        # Show bone assignment summary
        if self._mapper:
            bone_counts: dict[str, int] = {}
            for i in range(bone_local.num_ellipsoids):
                name = self._mapper.get_bone_name(i)
                bone_counts[name] = bone_counts.get(name, 0) + 1
            summary = ", ".join(f"{n}:{c}" for n, c in
                                sorted(bone_counts.items(), key=lambda x: -x[1]))
            self._lbl_bones.setText(f"Bone assignment: {summary}")

    def set_auto_pipeline_running(self, running: bool):
        """Enable/disable the auto pipeline button during training."""
        self._btn_auto.setEnabled(not running and self._rigged_mesh is not None)
        if running:
            self._btn_auto.setText("⏳ Running…")
        else:
            self._btn_auto.setText("▶ Auto Fit All Poses")

    def set_progress(self, value: int, maximum: int = 100):
        """Update progress bar."""
        self._progress.setRange(0, maximum)
        self._progress.setValue(value)
        self._progress.setVisible(value < maximum)

    def get_deformed_mesh(self, pose_index: int = -1) -> Optional[np.ndarray]:
        """Return deformed vertices for the given pose (or current)."""
        if self._rigged_mesh is None:
            return None

        if pose_index < 0:
            pose_index = self._slider_pose.value()

        rm = self._rigged_mesh
        pose = self.pose_at(pose_index)
        skin_mats = rm.skeleton.compute_skin_matrices(pose)
        return deform_mesh(
            rm.vertices, rm.skin_joints, rm.skin_weights, skin_mats,
        )

    def get_world_ellipsoids(
        self, pose_index: int = -1,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (centers, radii, rotations) in world space for the given pose."""
        if self._mapper is None or self._bone_local is None:
            return None
        if self._rigged_mesh is None:
            return None

        if pose_index < 0:
            pose_index = self._slider_pose.value()

        pose = self.pose_at(pose_index)
        return self._mapper.local_to_world_np(self._bone_local, pose)

    # ── Slots ────────────────────────────────────────────────────────

    def _on_toggle(self, checked: bool):
        self.rigModeToggled.emit(checked)

    def _refresh_slider_for_source(self):
        """Re-range the timeline slider for the active source and refresh view."""
        n = len(self.active_poses)
        self._slider_pose.blockSignals(True)
        self._slider_pose.setRange(0, max(0, n - 1))
        self._slider_pose.setValue(0)
        self._slider_pose.setEnabled(n > 1)
        self._slider_pose.blockSignals(False)
        self._on_pose_changed(0)

    def _populate_source_combo(self, select=None):
        """Rebuild the source dropdown: [native Animation] + saved poses.

        *select* is the combo data to re-select afterwards (defaults to the
        previously selected entry, falling back to native).
        """
        if select is None:
            select = self._cmb_source.currentData()
        self._cmb_source.blockSignals(True)
        self._cmb_source.clear()
        self._cmb_source.addItem("[native Animation]", self.NATIVE)
        for i, p in enumerate(self._library_poses):
            self._cmb_source.addItem(p.name, i)
        target = self._cmb_source.findData(select)
        self._cmb_source.setCurrentIndex(max(0, target))
        self._cmb_source.blockSignals(False)

    def _on_source_changed(self, _index: int):
        self._refresh_slider_for_source()

    def reload_library(self):
        """Re-scan the poses/ folder and rebuild the dropdown."""
        if self._rigged_mesh is None:
            return
        prev = self._cmb_source.currentData()
        self._library_poses = pose_library.load_all_poses(
            self._rigged_mesh.skeleton, self._pose_dir,
        )
        self._populate_source_combo(select=prev)
        self._refresh_slider_for_source()

    def _on_save_pose_clicked(self):
        if self._rigged_mesh is None:
            return
        pose = self.current_pose
        if pose is None:
            return
        default_name = pose.name if pose.name not in ("", "T-Pose") else "pose"
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Pose", "Pose name:", text=default_name,
        )
        if not ok or not name.strip():
            return
        # Persist under the chosen name so it reloads correctly.
        pose_to_save = Pose(name=name.strip(), bone_locals=dict(pose.bone_locals))
        try:
            path = pose_library.save_pose(
                pose_to_save, self._rigged_mesh.skeleton,
                name.strip(), self._pose_dir,
            )
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(
                self, "Save Pose", f"Failed to save pose:\n{e}")
            return
        self.reload_library()
        # Select the freshly saved pose in the dropdown.
        for i, p in enumerate(self._library_poses):
            if p.name == name.strip():
                self._populate_source_combo(select=i)
                self._refresh_slider_for_source()
                break
        self._lbl_bones.setText(f"Saved pose → {path.name}")

    def _on_pose_changed(self, idx: int):
        if self._rigged_mesh is None:
            return
        n = len(self.active_poses)
        pose = self.pose_at(idx)
        self._lbl_pose.setText(f"{idx} / {max(0, n - 1)}")
        self._lbl_pose_name.setText(f"Pose: {pose.name}")
        self.poseChanged.emit(idx)

    def _on_assign_clicked(self):
        """Triggered by "Assign to Bones" button.

        The actual assignment needs the current ellipsoid set from the
        main window, so we just emit and let main_window handle it.
        """
        # This is handled externally — see main_window integration
        self._btn_assign.setEnabled(False)
        self._btn_assign.setText("Assigning…")
        QtCore.QTimer.singleShot(0, self._do_assign_signal)

    def _do_assign_signal(self):
        """Deferred to allow UI update before heavy computation."""
        # The actual work is done in main_window._on_rig_assign_clicked
        self._btn_assign.setText("1. Assign to Bones")
        self._btn_assign.setEnabled(True)

    def _on_multipose_clicked(self):
        self.multiPoseRequested.emit()


# ── File detection helper ────────────────────────────────────────────────────

def try_load_rigged(path: str, target_scale: float = 1.0) -> Optional[RiggedMesh]:
    """Try to load a file as a rigged mesh. Returns None if not rigged."""
    p = Path(path)
    if not is_rigged_mesh(p):
        return None
    try:
        return load_rigged_mesh(p, target_scale=target_scale)
    except Exception as e:
        print(f"[Rig] Failed to load {p.name}: {e}")
        return None
