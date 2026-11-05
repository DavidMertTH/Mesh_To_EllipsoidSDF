"""
rig_panel.py — Modular UI panel for the rigged humanoid workflow.

Shown in the Mesh tab once a rigged mesh is loaded.  The poses always come
from the rig's joint ranges (``systematic_pose_fitter``); frame 0 is the
bind/base pose and every further frame is a nearby joint-range pose.

Workflow:
  1. Fit the base pose (main Steps / LR, ellipsoids are assigned to bones).
  2. Fit the nearby joint-range poses as morph keys (own "Steps per morph
     pose") and save the morph fit automatically.

The animation preview plays a random multi-axis motion inside the joint
limits, a pose-library clip or the mesh's own animation without refitting, so
the morph keys or the joint functions can be watched live.

The Unity API can still select its own pose source (mesh animation or a
pose-library clip) through ``select_source``; the GUI never shows it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from PySide6 import QtCore, QtWidgets

from animation_preview import AnimationTrack, random_motion
from skeleton import Pose, interpolate_pose
from joint_pose import JointPoseEdits, joint_pose_angles
from systematic_pose_fitter import (
    SystematicPoseSettings, generate_systematic_poses, systematic_pose_signature,
)
from native_morph import MorphSettings
from rig_loader import RiggedMesh, load_rigged_mesh, is_rigged_mesh
from skinning import deform_mesh
from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
import pose_library


class RigModePanel(QtWidgets.QGroupBox):
    """Rig controls: joint-range pose preview + base/morph fitting workflow.

    Signals
    -------
    rigLoaded(object)
        Emitted when a rigged mesh is loaded.  Payload: RiggedMesh.
    poseChanged(int)
        Emitted when the visible pose changes.
    fitRequested(bool, bool)
        ``(fit_base, fit_morphs)`` — start the native workflow.
    """

    rigLoaded = QtCore.Signal(object)
    poseChanged = QtCore.Signal(int)
    fitRequested = QtCore.Signal(bool, bool)
    exportUnityRequested = QtCore.Signal()
    jointEditRequested = QtCore.Signal(int)
    morphPreviewChanged = QtCore.Signal()
    # (pose, elapsed seconds since the previous frame, playback seconds)
    animationPoseChanged = QtCore.Signal(object, float, float)
    # True while an animation pose replaces the timeline pose
    animationStateChanged = QtCore.Signal(bool)
    saveMorphRequested = QtCore.Signal()
    loadMorphRequested = QtCore.Signal()

    # Pose sources.  The GUI always uses the joint ranges; the Unity API may
    # select the mesh's own animation or a pose-library clip by name.
    NATIVE = -1
    SYSTEMATIC = "__joint_ranges__"

    def __init__(self, parent=None, pose_dir: Optional[Path] = None):
        super().__init__("Rig · Morph Poses", parent)
        self._active = False

        self._rigged_mesh: Optional[RiggedMesh] = None
        self._mapper: Optional[BoneEllipsoidMapper] = None
        self._bone_local: Optional[BoneLocalEllipsoids] = None
        self._joint_edits: dict[tuple, JointPoseEdits] = {}
        self._source = self.SYSTEMATIC
        self._systematic_plan = None
        self._systematic_preview = None
        self._systematic_signature = None
        self._pose_training_running = False
        self._morph_key_count = 0

        # Pose-library clips: only reachable through the Unity API.
        self._pose_dir: Path = Path(pose_dir) if pose_dir \
            else pose_library.DEFAULT_POSE_DIR
        self._library_clips: List[pose_library.PoseClip] = []

        # Animation preview (never fitted; drives the live morph preview).
        self._tracks: dict[str, AnimationTrack] = {}
        self._random_seed = 0
        self._anim_active = False
        self._anim_pose: Optional[Pose] = None
        self._anim_time = 0.0
        self._anim_elapsed = 0.0
        self._anim_last = 0.0
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)

        self._build_ui()

    # ── State ────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    def setChecked(self, active: bool) -> None:  # noqa: N802 - Qt-style API
        """Activate/deactivate rig handling (the panel is not a checkbox)."""
        self._active = bool(active)
        if not self._active:
            self.stop_animation()

    def isChecked(self) -> bool:  # noqa: N802 - Qt-style API
        return self._active

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
    def current_pose_frame(self) -> float:
        return float(self._slider_pose.value())

    @property
    def current_source(self):
        return self._source

    @property
    def current_source_label(self) -> str:
        if self._source == self.SYSTEMATIC:
            return "[Joint ranges]"
        if self._source == self.NATIVE:
            return "[Mesh Animation]"
        return str(self._source)

    @property
    def shape_fitting_enabled(self) -> bool:
        """Morph shapes are always trained; kept for API callers."""
        return True

    @property
    def morph_steps(self) -> int:
        return int(self._spin_morph_steps.value())

    @property
    def active_poses(self) -> List[Pose]:
        """The pose list currently driving the timeline."""
        if self._source == self.SYSTEMATIC and self._rigged_mesh is not None:
            if self._systematic_plan is None:
                self._systematic_plan = self._systematic_preview_plan()
            return self._systematic_plan.poses
        if self._source != self.NATIVE:
            for clip in self._library_clips:
                if clip.name == self._source:
                    return clip.poses
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

    def pose_pair_at(self, frame: float) -> tuple[int, int, float]:
        """Return integer pose endpoints and blend factor for a fractional frame."""
        poses = self.active_poses
        n = len(poses)
        if n <= 1:
            return 0, 0, 0.0
        f = float(np.clip(frame, 0.0, float(n - 1)))
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, n - 1)
        return i0, i1, f - float(i0)

    def pose_at_frame(self, frame: float) -> Pose:
        """Return the active pose, interpolating when *frame* is fractional."""
        edit = self._joint_edits.get(self._joint_edit_key(frame))
        return edit.apply() if edit is not None else self._source_pose_at_frame(frame)

    def _source_pose_at_frame(self, frame: float) -> Pose:
        poses = self.active_poses
        if not poses:
            return Pose.t_pose()
        i0, i1, w = self.pose_pair_at(frame)
        if i0 == i1 or w <= 1.0e-6 or self._rigged_mesh is None:
            return self.pose_at(i0)
        if w >= 1.0 - 1.0e-6:
            return self.pose_at(i1)
        return interpolate_pose(
            self._rigged_mesh.skeleton,
            poses[i0],
            poses[i1],
            w,
            name=f"{poses[i0].name} -> {poses[i1].name} {w:.2f}",
        )

    def _joint_edit_key(self, frame: float) -> tuple:
        return (self._source,
                round(float(np.clip(frame, 0, max(0, len(self.active_poses) - 1))), 3))

    def set_joint_rotation(self, index: int, axis: int, value: float) -> None:
        if self._rigged_mesh is None or not self.is_active or self._pose_training_running:
            return
        self.stop_animation()
        frame = self.current_pose_frame
        key = self._joint_edit_key(frame)
        edit = self._joint_edits.get(key)
        if edit is None:
            edit = JointPoseEdits(self._rigged_mesh.skeleton, self._source_pose_at_frame(frame))
        if edit.set_axis(index, axis, value):
            self._joint_edits[key] = edit
            self.poseChanged.emit(self.current_pose_index)

    def reset_joint_rotation(self, index: int) -> None:
        if self._pose_training_running:
            return
        edit = self._joint_edits.get(self._joint_edit_key(self.current_pose_frame))
        if edit is not None and edit.reset_joint(index):
            self.poseChanged.emit(self.current_pose_index)

    def reset_joint_rotations(self) -> None:
        if self._pose_training_running:
            return
        edit = self._joint_edits.pop(self._joint_edit_key(self.current_pose_frame), None)
        if edit is not None and edit.locals:
            self.poseChanged.emit(self.current_pose_index)

    def joint_rotation_state(self, pose: Pose | None):
        """Angles and reset availability for the pose currently being rendered."""
        skeleton = self._rigged_mesh.skeleton
        angles = joint_pose_angles(skeleton, pose)
        edited = np.zeros(skeleton.num_bones, dtype=bool)
        edit = self._joint_edits.get(self._joint_edit_key(self.current_pose_frame))
        if edit is not None and pose is not None:
            for index, local in edit.locals.items():
                if index in pose.bone_locals and np.allclose(pose.bone_locals[index], local, atol=1e-7):
                    angles[index] = edit.angles[index]
                    edited[index] = True
        return angles, edited

    def select_joint(self, index: int) -> None:
        """Keep the rig-panel selector aligned with clicks in the viewport."""
        if 0 <= index < self._cmb_joint.count():
            self._cmb_joint.setCurrentIndex(index)

    def reset_pose_to_start(self, emit: bool = True) -> None:
        """Move the visible pose timeline back to frame 0 (the base pose)."""
        self.stop_animation(notify=emit)
        with QtCore.QSignalBlocker(self._slider_pose):
            self._slider_pose.setValue(0)
        self._update_pose_labels(0)
        if emit and self._rigged_mesh is not None:
            self.poseChanged.emit(0)

    @property
    def current_pose(self) -> Optional[Pose]:
        """The rendered pose: the animation pose while one is shown, else the timeline pose."""
        if self._rigged_mesh is None:
            return None
        if self._anim_active and self._anim_pose is not None:
            return self._anim_pose
        return self.pose_at_frame(self.current_pose_frame)

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._lbl_status = QtWidgets.QLabel("No rigged mesh loaded")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._lbl_status)

        self._build_workflow(layout)
        self._build_pose_preview(layout)
        self._build_animation(layout)
        self._build_joint_ranges(layout)
        self._build_morph_preview(layout)

        self._btn_export = QtWidgets.QPushButton("⬆ Export for Unity (.json)")
        self._btn_export.setToolTip(
            "Export bone-local ellipsoids to JSON for import into Unity")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(lambda: self.exportUnityRequested.emit())
        layout.addWidget(self._btn_export)
        layout.addStretch()

    def _build_workflow(self, layout):
        group = QtWidgets.QGroupBox("Workflow")
        v = QtWidgets.QVBoxLayout(group)
        v.setSpacing(4)
        hint = QtWidgets.QLabel(
            "1. Fit the base pose (main Steps / LR).\n"
            "2. Fit the nearby joint-range poses as morph keys; "
            "the morph fit is saved automatically.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: gray;")
        v.addWidget(hint)

        form = QtWidgets.QFormLayout()
        self._spin_morph_steps = QtWidgets.QSpinBox()
        self._spin_morph_steps.setRange(10, 100000)
        self._spin_morph_steps.setSingleStep(100)
        self._spin_morph_steps.setValue(1000)
        self._spin_morph_steps.setToolTip(
            "Optimizer steps for each morph pose fit (the base pose uses the main Steps)")
        form.addRow("Steps per morph pose:", self._spin_morph_steps)
        v.addLayout(form)

        self._btn_workflow = QtWidgets.QPushButton("▶ Fit Base + Morph Poses")
        self._btn_workflow.setToolTip(
            "Fit the base pose, then every joint-range pose as a morph key, then save")
        self._btn_workflow.setStyleSheet(
            "font-weight: bold; background-color: #2d7d32; color: white; padding: 4px;")
        self._btn_workflow.clicked.connect(lambda: self.fitRequested.emit(True, True))
        v.addWidget(self._btn_workflow)

        row = QtWidgets.QHBoxLayout()
        self._btn_fit_base = QtWidgets.QPushButton("1 · Fit Base Pose")
        self._btn_fit_base.setToolTip(
            "Fit the bind pose from scratch and assign the primitives to bones")
        self._btn_fit_base.clicked.connect(lambda: self.fitRequested.emit(True, False))
        row.addWidget(self._btn_fit_base)
        self._btn_fit_morphs = QtWidgets.QPushButton("2 · Fit Morph Poses")
        self._btn_fit_morphs.setToolTip(
            "Fit the joint-range poses from the current base fit and save the morph fit")
        self._btn_fit_morphs.clicked.connect(lambda: self.fitRequested.emit(False, True))
        row.addWidget(self._btn_fit_morphs)
        v.addLayout(row)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        v.addWidget(self._progress)

        self._lbl_bones = QtWidgets.QLabel("")
        self._lbl_bones.setWordWrap(True)
        self._lbl_bones.setStyleSheet("font-size: 10px; color: gray;")
        v.addWidget(self._lbl_bones)
        layout.addWidget(group)

    def _build_pose_preview(self, layout):
        pose_row = QtWidgets.QHBoxLayout()
        pose_row.addWidget(QtWidgets.QLabel("Pose:"))
        self._slider_pose = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_pose.setRange(0, 0)
        self._slider_pose.setValue(0)
        self._slider_pose.setEnabled(False)
        self._slider_pose.setToolTip("Preview the joint-range poses (0 = base pose)")
        self._slider_pose.valueChanged.connect(self._on_pose_changed)
        pose_row.addWidget(self._slider_pose, 1)
        self._lbl_pose = QtWidgets.QLabel("0 / 0")
        self._lbl_pose.setFixedWidth(80)
        pose_row.addWidget(self._lbl_pose)
        layout.addLayout(pose_row)

        self._lbl_pose_name = QtWidgets.QLabel("")
        self._lbl_pose_name.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(self._lbl_pose_name)

        joint_row = QtWidgets.QHBoxLayout()
        joint_row.addWidget(QtWidgets.QLabel("Joint:"))
        self._cmb_joint = QtWidgets.QComboBox()
        self._cmb_joint.setMinimumWidth(0)
        self._cmb_joint.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._cmb_joint.setMinimumContentsLength(12)
        self._cmb_joint.setEnabled(False)
        self._cmb_joint.activated.connect(lambda i: self.jointEditRequested.emit(i))
        joint_row.addWidget(self._cmb_joint, 1)
        self._btn_edit_joint = QtWidgets.QPushButton("Rotate…")
        self._btn_edit_joint.setEnabled(False)
        self._btn_edit_joint.setToolTip(
            "Open live joint rotation and range controls in the 3-D viewer")
        self._btn_edit_joint.clicked.connect(
            lambda: self.jointEditRequested.emit(self._cmb_joint.currentIndex()))
        joint_row.addWidget(self._btn_edit_joint)
        layout.addLayout(joint_row)

    def _build_animation(self, layout):
        group = QtWidgets.QGroupBox("Animation preview")
        v = QtWidgets.QVBoxLayout(group)
        v.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Clip:"))
        self._cmb_clip = QtWidgets.QComboBox()
        self._cmb_clip.setMinimumWidth(0)
        self._cmb_clip.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._cmb_clip.setMinimumContentsLength(12)
        self._cmb_clip.setToolTip("Random motion: random poses inside the joint limits, several axes "
                                  "per joint at once. Clips come from the pose library.")
        self._cmb_clip.currentIndexChanged.connect(self._on_clip_changed)
        row.addWidget(self._cmb_clip, 1)
        self._btn_new_motion = QtWidgets.QPushButton("New")
        self._btn_new_motion.setToolTip("Generate a different random motion")
        self._btn_new_motion.clicked.connect(self._on_new_random_motion)
        row.addWidget(self._btn_new_motion)
        v.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self._btn_play = QtWidgets.QPushButton("▶ Play")
        self._btn_play.setToolTip("Play the clip; the primitives follow the selected morph model live")
        self._btn_play.clicked.connect(self.toggle_animation)
        row.addWidget(self._btn_play)
        self._btn_stop_anim = QtWidgets.QPushButton("■ Stop")
        self._btn_stop_anim.setToolTip("Stop and return to the timeline pose")
        self._btn_stop_anim.clicked.connect(lambda: self.stop_animation())
        row.addWidget(self._btn_stop_anim)
        row.addWidget(QtWidgets.QLabel("Speed:"))
        self._spin_anim_speed = QtWidgets.QDoubleSpinBox()
        self._spin_anim_speed.setRange(.05, 5)
        self._spin_anim_speed.setDecimals(2)
        self._spin_anim_speed.setSingleStep(.25)
        self._spin_anim_speed.setValue(1)
        self._spin_anim_speed.setSuffix("×")
        row.addWidget(self._spin_anim_speed)
        v.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self._slider_anim = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_anim.setRange(0, 1000)
        self._slider_anim.setToolTip("Scrub through the clip")
        self._slider_anim.valueChanged.connect(self._on_animation_scrubbed)
        row.addWidget(self._slider_anim, 1)
        self._lbl_anim = QtWidgets.QLabel("0.0 / 0.0 s")
        self._lbl_anim.setFixedWidth(90)
        row.addWidget(self._lbl_anim)
        v.addLayout(row)
        hint = QtWidgets.QLabel("Nothing is fitted while playing. The Pose Error panel shows the "
                                "total error of no morph, morph keys and joint functions live.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: gray;")
        v.addWidget(hint)
        layout.addWidget(group)

    def _build_joint_ranges(self, layout):
        group = QtWidgets.QGroupBox("Joint-range poses")
        form = QtWidgets.QFormLayout(group)
        self._spin_sweep_step = QtWidgets.QDoubleSpinBox()
        self._spin_sweep_step.setRange(5, 90)
        self._spin_sweep_step.setDecimals(1)
        self._spin_sweep_step.setSingleStep(5)
        self._spin_sweep_step.setSuffix("°")
        self._spin_sweep_step.setValue(30)
        self._spin_sweep_step.setKeyboardTracking(False)
        self._spin_sweep_step.setToolTip(
            "Angle between neighbouring poses; each pose starts from an adjacent, already fitted one")
        form.addRow("Angle step:", self._spin_sweep_step)
        self._spin_sweep_coverage = QtWidgets.QDoubleSpinBox()
        self._spin_sweep_coverage.setRange(5, 100)
        self._spin_sweep_coverage.setDecimals(0)
        self._spin_sweep_coverage.setSuffix("%")
        self._spin_sweep_coverage.setValue(90)
        self._spin_sweep_coverage.setKeyboardTracking(False)
        self._spin_sweep_coverage.setToolTip("Share of each joint range to sample, including its covered endpoints")
        form.addRow("Range coverage:", self._spin_sweep_coverage)
        self._spin_sweep_passes = QtWidgets.QSpinBox()
        self._spin_sweep_passes.setRange(1, 12)
        self._spin_sweep_passes.setValue(4)
        form.addRow("Recombined passes:", self._spin_sweep_passes)
        self._chk_sweep_roots = QtWidgets.QCheckBox("Include root rotations")
        self._chk_sweep_roots.setToolTip("Root orientation normally moves the whole model without deforming it")
        form.addRow(self._chk_sweep_roots)
        self._lbl_sweep = QtWidgets.QLabel("Load a rig to generate poses from its joint ranges.")
        self._lbl_sweep.setWordWrap(True)
        form.addRow(self._lbl_sweep)
        layout.addWidget(group)
        for widget in (self._spin_sweep_step, self._spin_sweep_coverage, self._spin_sweep_passes):
            widget.valueChanged.connect(self._update_systematic_summary)
        self._chk_sweep_roots.toggled.connect(self._update_systematic_summary)

    def _build_morph_preview(self, layout):
        morph = QtWidgets.QGroupBox("Morph fit")
        form = QtWidgets.QFormLayout(morph)
        self._chk_morph_preview = QtWidgets.QCheckBox("Pose-driven primitive morphing")
        self._chk_morph_preview.setChecked(True)
        self._chk_morph_preview.setToolTip("Drive trained morph shapes from live joint rotations, including manual poses")
        form.addRow(self._chk_morph_preview)
        self._cmb_morph_model = QtWidgets.QComboBox()
        self._cmb_morph_model.addItem("Morph keys (pose blend)", "keys")
        self._cmb_morph_model.addItem("Joint functions (polynomial)", "function")
        self._cmb_morph_model.setToolTip(
            "Morph keys: blend the stored per-pose fits (the settings below).\n"
            "Joint functions: one small polynomial per primitive of the rotations of its "
            "nearby joints, fitted to the stored keys (no key database at runtime).")
        form.addRow("Model:", self._cmb_morph_model)
        self._spin_morph_context = QtWidgets.QSpinBox()
        self._spin_morph_context.setRange(0, 4)
        self._spin_morph_context.setValue(1)
        self._spin_morph_context.setToolTip("0: assigned bone; 1: also direct parent and child rotations")
        form.addRow("Joint context:", self._spin_morph_context)
        self._spin_morph_falloff = QtWidgets.QDoubleSpinBox()
        self._spin_morph_falloff.setRange(.1, 100)
        self._spin_morph_falloff.setValue(8)
        self._spin_morph_falloff.setKeyboardTracking(False)
        form.addRow("Pose falloff:", self._spin_morph_falloff)
        self._spin_morph_smoothing = QtWidgets.QDoubleSpinBox()
        self._spin_morph_smoothing.setRange(0, 2)
        self._spin_morph_smoothing.setDecimals(2)
        self._spin_morph_smoothing.setSingleStep(.02)
        self._spin_morph_smoothing.setValue(.12)
        self._spin_morph_smoothing.setSuffix(" s")
        self._spin_morph_smoothing.setKeyboardTracking(False)
        self._spin_morph_smoothing.setToolTip("Time to reach 90% of a new morph blend; 0 updates immediately")
        form.addRow("Transition:", self._spin_morph_smoothing)
        self._lbl_morph = QtWidgets.QLabel("No trained morph fit loaded")
        self._lbl_morph.setWordWrap(True)
        form.addRow(self._lbl_morph)
        self._lbl_function = QtWidgets.QLabel("")
        self._lbl_function.setWordWrap(True)
        self._lbl_function.setStyleSheet("font-size: 10px; color: gray;")
        form.addRow(self._lbl_function)
        row = QtWidgets.QHBoxLayout()
        self._btn_save_morph = QtWidgets.QPushButton("Save as…")
        self._btn_save_morph.setEnabled(False)
        self._btn_save_morph.clicked.connect(self.saveMorphRequested)
        row.addWidget(self._btn_save_morph)
        self._btn_load_morph = QtWidgets.QPushButton("Load…")
        self._btn_load_morph.clicked.connect(self.loadMorphRequested)
        row.addWidget(self._btn_load_morph)
        form.addRow(row)
        layout.addWidget(morph)
        self._chk_morph_preview.toggled.connect(self.morphPreviewChanged)
        self._cmb_morph_model.currentIndexChanged.connect(self.morphPreviewChanged)
        for widget in (self._spin_morph_context, self._spin_morph_falloff, self._spin_morph_smoothing):
            widget.valueChanged.connect(self.morphPreviewChanged)
        self._set_rig_controls_enabled(False)

    def _set_rig_controls_enabled(self, enabled: bool) -> None:
        for widget in self._rig_controls():
            widget.setEnabled(enabled)

    def _rig_controls(self):
        return (self._cmb_joint, self._btn_edit_joint, self._btn_workflow,
                self._btn_fit_base, self._btn_fit_morphs, self._spin_morph_steps,
                self._spin_sweep_step, self._spin_sweep_coverage,
                self._spin_sweep_passes, self._chk_sweep_roots, self._btn_load_morph,
                self._cmb_clip, self._btn_new_motion, self._btn_play, self._btn_stop_anim,
                self._spin_anim_speed, self._slider_anim)

    # ── Settings ─────────────────────────────────────────────────────

    def pose_tool_setting_specs(self):
        return [
            ("systematic_step", self._spin_sweep_step, "float"),
            ("systematic_coverage", self._spin_sweep_coverage, "float"),
            ("systematic_passes", self._spin_sweep_passes, "int"),
            ("systematic_roots", self._chk_sweep_roots, "bool"),
            ("morph_steps", self._spin_morph_steps, "int"),
            ("morph_preview", self._chk_morph_preview, "bool"),
            ("morph_context", self._spin_morph_context, "int"),
            ("morph_falloff", self._spin_morph_falloff, "float"),
            ("morph_smoothing", self._spin_morph_smoothing, "float"),
            ("morph_model", self._cmb_morph_model, "combo_data"),
            ("animation_speed", self._spin_anim_speed, "float"),
        ]

    def systematic_settings(self):
        return SystematicPoseSettings(angle_step=self._spin_sweep_step.value(),
            coverage=self._spin_sweep_coverage.value() / 100,
            passes=self._spin_sweep_passes.value(), include_roots=self._chk_sweep_roots.isChecked())

    def morph_settings(self):
        return MorphSettings(context_depth=self._spin_morph_context.value(),
            falloff=self._spin_morph_falloff.value(), smoothing_time=self._spin_morph_smoothing.value())

    @property
    def morph_preview_enabled(self):
        return self._chk_morph_preview.isChecked()

    @property
    def morph_model(self) -> str:
        """``"keys"`` (pose blend) or ``"function"`` (joint functions)."""
        return self._cmb_morph_model.currentData() or "keys"

    def set_function_summary(self, text: str) -> None:
        self._lbl_function.setText(text)

    # ── Joint-range poses ────────────────────────────────────────────

    def _systematic_preview_plan(self):
        skeleton = self._rigged_mesh.skeleton
        settings = self.systematic_settings()
        signature = systematic_pose_signature(skeleton, settings)
        if signature != self._systematic_signature:
            self._systematic_preview = generate_systematic_poses(skeleton, settings)
            self._systematic_signature = signature
        return self._systematic_preview

    def _update_systematic_summary(self, *_):
        """Re-generate the joint-range poses after a setting or range changed.

        The plan stays frozen while a fit is running; it is refreshed when the
        fit ends.
        """
        if self._rigged_mesh is None:
            return
        plan = self._systematic_preview_plan()
        self._lbl_sweep.setText(f"{len(plan.poses)} poses · {plan.joint_count} joints · "
            f"{plan.channel_count} axes")
        if self._pose_training_running or plan is self._systematic_plan:
            return
        self._systematic_plan = plan
        # Edits belonged to frames of the previous plan.
        self._joint_edits = {key: edit for key, edit in self._joint_edits.items()
                             if key[0] != self.SYSTEMATIC}
        if self._source == self.SYSTEMATIC:
            self._refresh_slider_for_source(keep_index=True)

    def generate_systematic_source(self):
        """Select the up-to-date joint-range poses; False if none can move."""
        if self._rigged_mesh is None or self._pose_training_running:
            return False
        self._update_systematic_summary()
        if self._source != self.SYSTEMATIC:
            self._source = self.SYSTEMATIC
            self._refresh_slider_for_source()
        return self._systematic_plan.channel_count > 0

    # ── Public methods ───────────────────────────────────────────────

    def set_morph_library_count(self, count):
        self._morph_key_count = int(count)
        self._btn_save_morph.setEnabled(count > 0 and not self._pose_training_running)
        self._lbl_morph.setText(f"{count} trained pose keys · live joint-driven shapes" if count
                                else "No trained morph fit loaded")

    def set_morph_saved_path(self, path) -> None:
        self._lbl_morph.setText(
            f"{self._morph_key_count} trained pose keys · saved → {Path(path).name}")
        self._lbl_morph.setToolTip(str(path))

    def set_rigged_mesh(self, rigged_mesh: RiggedMesh):
        """Set a loaded rigged mesh and update UI."""
        self._rigged_mesh = rigged_mesh
        self._pose_training_running = False
        self._joint_edits.clear()
        self._source = self.SYSTEMATIC
        self._systematic_plan = self._systematic_preview = self._systematic_signature = None
        self._mapper = BoneEllipsoidMapper(rigged_mesh.skeleton)
        self._bone_local = None
        self.stop_animation(notify=False)
        self._btn_export.setEnabled(False)
        self._lbl_bones.setText("")
        self._library_clips = pose_library.load_all_clips(
            rigged_mesh.skeleton, self._pose_dir,
        )

        n_bones = rigged_mesh.skeleton.num_bones
        self._cmb_joint.clear()
        for bone in rigged_mesh.skeleton.bones:
            self._cmb_joint.addItem(bone.name.rsplit(":", 1)[-1])
            self._cmb_joint.setItemData(self._cmb_joint.count() - 1, bone.name, QtCore.Qt.ToolTipRole)
        self._set_rig_controls_enabled(n_bones > 0)
        self.set_morph_library_count(0)
        self._update_systematic_summary()
        self._rebuild_tracks()

        self._lbl_status.setText(
            f"Rig: {rigged_mesh.mesh_name} | {n_bones} bones")
        self._lbl_status.setStyleSheet("color: green; font-size: 11px;")

        self._refresh_slider_for_source()

        self.rigLoaded.emit(rigged_mesh)

    def set_bone_local(self, bone_local: BoneLocalEllipsoids | None):
        """Update bone-local params (after assignment or training)."""
        self._bone_local = bone_local
        self.set_morph_library_count(0)
        self._btn_export.setEnabled(bone_local is not None)

        if self._mapper:
            self._mapper._bone_local = bone_local
        if bone_local is None:
            self._lbl_bones.setText("")
            return
        if self._mapper:
            bone_counts: dict[str, int] = {}
            for i in range(bone_local.num_ellipsoids):
                name = self._mapper.get_bone_name(i)
                bone_counts[name] = bone_counts.get(name, 0) + 1
            summary = ", ".join(f"{n}:{c}" for n, c in
                                sorted(bone_counts.items(), key=lambda x: -x[1]))
            self._lbl_bones.setText(f"Base pose: {bone_local.num_ellipsoids} primitives · {summary}")

    def set_auto_pipeline_running(self, running: bool):
        """Freeze pose authoring while a base or morph fit is running."""
        if running:
            self.stop_animation()
        self._pose_training_running = bool(running)
        has_rig = self._rigged_mesh is not None
        self._set_rig_controls_enabled(not running and has_rig)
        self._btn_save_morph.setEnabled(not running and self._morph_key_count > 0)
        self._slider_pose.setEnabled(not running and len(self.active_poses) > 1)
        self._btn_workflow.setText("⏳ Running…" if running else "▶ Fit Base + Morph Poses")
        if not running:
            self._update_systematic_summary()

    def set_progress(self, value: int, maximum: int = 100):
        """Update progress bar."""
        self._progress.setRange(0, maximum)
        self._progress.setValue(value)
        self._progress.setVisible(value < maximum)

    def show_training_pose(self, index: int, pose_name: str | None = None) -> None:
        """Follow the worker's target without triggering another pose render."""
        count = len(self.active_poses)
        index = int(np.clip(index, 0, max(0, count - 1)))
        with QtCore.QSignalBlocker(self._slider_pose):
            self._slider_pose.setValue(index)
        self._update_pose_labels(index)
        if pose_name is not None:
            self._lbl_pose_name.setText(f"Pose: {pose_name}")

    def get_deformed_mesh(self, pose_index: float = -1.0) -> Optional[np.ndarray]:
        """Return deformed vertices for the given pose (or current)."""
        if self._rigged_mesh is None:
            return None

        if pose_index < 0:
            pose_index = self.current_pose_frame

        rm = self._rigged_mesh
        pose = self.pose_at_frame(float(pose_index))
        skin_mats = rm.skeleton.compute_skin_matrices(pose)
        return deform_mesh(
            rm.vertices, rm.skin_joints, rm.skin_weights, skin_mats,
        )

    def get_world_ellipsoids(
        self, pose_index: float = -1.0,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (centers, radii, rotations) in world space for the given pose."""
        if self._mapper is None or self._bone_local is None:
            return None
        if self._rigged_mesh is None:
            return None

        if pose_index < 0:
            pose_index = self.current_pose_frame

        pose = self.pose_at_frame(float(pose_index))
        return self._mapper.local_to_world_np(self._bone_local, pose)

    def select_source(self, source) -> bool:
        """Select a pose source for the Unity API (native, joint ranges or a clip)."""
        if source is None:
            return True
        text = str(source).strip()
        if not text:
            return True
        lowered = text.lower()
        if lowered in ("native", "mesh", "mesh animation", "[mesh animation]"):
            target = self.NATIVE
        elif lowered in ("systematic", "joint ranges", "[joint ranges]"):
            target = self.SYSTEMATIC
        else:
            target = None
            for clip in self._library_clips:
                label = clip.name if len(clip.poses) == 1 else f"{clip.name} ({len(clip.poses)} frames)"
                if lowered in (clip.name.lower(), label.lower()):
                    target = clip.name
                    break
            if target is None:
                return False
        if target != self._source:
            self.stop_animation(notify=False)
            self._source = target
            self._refresh_slider_for_source()
        return True

    # ── Animation preview ──

    @property
    def animation_active(self) -> bool:
        """An animation pose (playing or paused) replaces the timeline pose."""
        return self._anim_active

    @property
    def animation_playing(self) -> bool:
        return self._anim_timer.isActive()

    @property
    def animation_track(self) -> Optional[AnimationTrack]:
        key = self._cmb_clip.currentData()
        if key is None or self._rigged_mesh is None:
            return None
        if key == "random" and key not in self._tracks:
            self._tracks[key] = random_motion(self._rigged_mesh.skeleton, seed=self._random_seed,
                                              include_roots=self._chk_sweep_roots.isChecked())
        return self._tracks.get(key)

    def _rebuild_tracks(self) -> None:
        self._tracks = {}
        with QtCore.QSignalBlocker(self._cmb_clip):
            self._cmb_clip.clear()
            if self._rigged_mesh is None:
                return
            self._cmb_clip.addItem("Random motion (joint limits)", "random")
            if len(self._rigged_mesh.poses) > 1:
                self._tracks["mesh"] = AnimationTrack("Mesh animation", list(self._rigged_mesh.poses), 10.0)
                self._cmb_clip.addItem(f"Mesh animation ({len(self._rigged_mesh.poses)} frames)", "mesh")
            for clip in self._library_clips:
                if len(clip.poses) > 1:
                    key = f"clip:{clip.name}"
                    self._tracks[key] = AnimationTrack(clip.name, list(clip.poses), 10.0)
                    self._cmb_clip.addItem(f"{clip.name} ({len(clip.poses)} frames)", key)
        self._update_animation_label()

    def toggle_animation(self) -> None:
        if self.animation_playing:
            self.pause_animation()
        else:
            self.play_animation()

    def play_animation(self) -> bool:
        track = self.animation_track
        if track is None or self._pose_training_running or not self.is_active:
            return False
        was_active = self._anim_active
        self._anim_active = True
        self._anim_last = time.perf_counter()
        self._anim_timer.start()
        self._btn_play.setText("⏸ Pause")
        if not was_active:
            self._anim_elapsed = 0.0
            self.animationStateChanged.emit(True)
        self._emit_animation_pose(0.0)
        return True

    def pause_animation(self) -> None:
        self._anim_timer.stop()
        self._btn_play.setText("▶ Play")

    def stop_animation(self, notify: bool = True) -> None:
        """Stop playback and show the timeline pose again."""
        self._anim_timer.stop()
        self._btn_play.setText("▶ Play")
        was_active = self._anim_active
        self._anim_active = False
        self._anim_pose = None
        self._anim_time = 0.0
        with QtCore.QSignalBlocker(self._slider_anim):
            self._slider_anim.setValue(0)
        self._update_animation_label()
        if was_active and notify:
            self.animationStateChanged.emit(False)

    def _on_animation_tick(self) -> None:
        now = time.perf_counter()
        dt = min(max(0.0, now - self._anim_last), .25)
        self._anim_last = now
        self._anim_time += dt * self._spin_anim_speed.value()
        self._anim_elapsed += dt
        self._emit_animation_pose(dt)

    def _emit_animation_pose(self, dt: float) -> None:
        track = self.animation_track
        if track is None or self._rigged_mesh is None:
            self.stop_animation()
            return
        self._anim_pose = track.pose_at(self._rigged_mesh.skeleton, self._anim_time)
        if track.duration > 0:
            with QtCore.QSignalBlocker(self._slider_anim):
                self._slider_anim.setValue(int(round((self._anim_time % track.duration)
                                                     / track.duration * 1000)))
        self._update_animation_label()
        self.animationPoseChanged.emit(self._anim_pose, float(dt), float(self._anim_elapsed))

    def _on_animation_scrubbed(self, value: int) -> None:
        track = self.animation_track
        if track is None or self._pose_training_running or not self.is_active:
            return
        self._anim_time = value / 1000 * track.duration
        if not self._anim_active:
            self._anim_active = True
            self._anim_elapsed = 0.0
            self.animationStateChanged.emit(True)
        self._emit_animation_pose(0.0)

    def _on_clip_changed(self, *_):
        self._anim_time = 0.0
        if self._anim_active:
            self._emit_animation_pose(0.0)
        self._update_animation_label()

    def _on_new_random_motion(self) -> None:
        self._random_seed += 1
        self._tracks.pop("random", None)
        index = self._cmb_clip.findData("random")
        if index >= 0 and index != self._cmb_clip.currentIndex():
            self._cmb_clip.setCurrentIndex(index)
        else:
            self._on_clip_changed()

    def _update_animation_label(self) -> None:
        track = self.animation_track if self._cmb_clip.count() else None
        duration = track.duration if track is not None else 0.0
        current = self._anim_time % duration if duration > 0 else 0.0
        self._lbl_anim.setText(f"{current:.1f} / {duration:.1f} s")

    # ── Slots ────────────────────────────────────────────────────────

    def _refresh_slider_for_source(self, keep_index: bool = False):
        """Re-range the timeline slider for the active source and refresh view."""
        n = len(self.active_poses)
        index = min(self._slider_pose.value(), max(0, n - 1)) if keep_index else 0
        with QtCore.QSignalBlocker(self._slider_pose):
            self._slider_pose.setRange(0, max(0, n - 1))
            self._slider_pose.setValue(index)
            self._slider_pose.setEnabled(n > 1 and not self._pose_training_running)
        self._on_pose_changed(index)

    def _on_pose_changed(self, idx: int):
        if self._rigged_mesh is None:
            return
        self.stop_animation()
        self._update_pose_labels(idx)
        self.poseChanged.emit(idx)

    def _update_pose_labels(self, idx: int):
        n = len(self.active_poses)
        idx = int(np.clip(idx, 0, max(0, n - 1)))
        self._lbl_pose.setText(f"{idx} / {max(0, n - 1)}")
        name = self.pose_at(idx).name
        if idx == 0 and self._source == self.SYSTEMATIC:
            name += " · base pose"
        self._lbl_pose_name.setText(f"Pose: {name}")


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
