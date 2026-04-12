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

    def __init__(self, parent=None):
        super().__init__("Rig Mode", parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._on_toggle)

        self._rigged_mesh: Optional[RiggedMesh] = None
        self._mapper: Optional[BoneEllipsoidMapper] = None
        self._bone_local: Optional[BoneLocalEllipsoids] = None

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
    def current_pose(self) -> Optional[Pose]:
        if self._rigged_mesh is None:
            return None
        idx = self._slider_pose.value()
        if 0 <= idx < len(self._rigged_mesh.poses):
            return self._rigged_mesh.poses[idx]
        return Pose.t_pose()

    @property
    def multi_pose_steps(self) -> int:
        return self._spin_steps.value()

    @property
    def multi_pose_lr(self) -> float:
        return self._spin_lr.value()

    @property
    def multi_pose_steps_per_pose(self) -> int:
        return self._spin_steps_per_pose.value()

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

        # ── Multi-pose training ──
        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Multi-Pose Training:"))

        train_form = QtWidgets.QFormLayout()
        train_form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self._spin_steps = QtWidgets.QSpinBox()
        self._spin_steps.setRange(100, 50000)
        self._spin_steps.setValue(3000)
        self._spin_steps.setSingleStep(500)
        self._spin_steps.setToolTip("Total training steps across all poses")
        train_form.addRow("Steps:", self._spin_steps)

        self._spin_lr = QtWidgets.QDoubleSpinBox()
        self._spin_lr.setRange(0.0001, 0.1)
        self._spin_lr.setValue(0.005)
        self._spin_lr.setSingleStep(0.001)
        self._spin_lr.setDecimals(4)
        self._spin_lr.setToolTip("Learning rate for bone-local parameter training")
        train_form.addRow("LR:", self._spin_lr)

        self._spin_steps_per_pose = QtWidgets.QSpinBox()
        self._spin_steps_per_pose.setRange(1, 1000)
        self._spin_steps_per_pose.setValue(10)
        self._spin_steps_per_pose.setSingleStep(5)
        self._spin_steps_per_pose.setToolTip("Training steps on each pose before switching")
        train_form.addRow("Steps/Pose:", self._spin_steps_per_pose)

        layout.addLayout(train_form)

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

        n_poses = len(rigged_mesh.poses)
        n_bones = rigged_mesh.skeleton.num_bones

        self._slider_pose.setRange(0, max(0, n_poses - 1))
        self._slider_pose.setValue(0)
        self._slider_pose.setEnabled(n_poses > 1)

        self._lbl_status.setText(
            f"Rig: {rigged_mesh.mesh_name} | "
            f"{n_bones} bones | {n_poses} poses"
        )
        self._lbl_status.setStyleSheet("color: green; font-size: 11px;")
        self._btn_assign.setEnabled(True)
        self._on_pose_changed(0)

        self.rigLoaded.emit(rigged_mesh)

    def set_bone_local(self, bone_local: BoneLocalEllipsoids):
        """Update bone-local params (after assignment or training)."""
        self._bone_local = bone_local
        self._btn_multipose.setEnabled(True)

        # Show bone assignment summary
        if self._mapper:
            bone_counts: dict[str, int] = {}
            for i in range(bone_local.num_ellipsoids):
                name = self._mapper.get_bone_name(i)
                bone_counts[name] = bone_counts.get(name, 0) + 1
            summary = ", ".join(f"{n}:{c}" for n, c in
                                sorted(bone_counts.items(), key=lambda x: -x[1]))
            self._lbl_bones.setText(f"Bone assignment: {summary}")

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
        pose = rm.poses[pose_index] if pose_index < len(rm.poses) else Pose.t_pose()
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

        rm = self._rigged_mesh
        pose = rm.poses[pose_index] if pose_index < len(rm.poses) else Pose.t_pose()
        return self._mapper.local_to_world_np(self._bone_local, pose)

    # ── Slots ────────────────────────────────────────────────────────

    def _on_toggle(self, checked: bool):
        self.rigModeToggled.emit(checked)

    def _on_pose_changed(self, idx: int):
        if self._rigged_mesh is None:
            return
        n = len(self._rigged_mesh.poses)
        pose = self._rigged_mesh.poses[idx] if idx < n else Pose.t_pose()
        self._lbl_pose.setText(f"{idx} / {n - 1}")
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
