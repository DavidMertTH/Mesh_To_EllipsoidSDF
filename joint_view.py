"""Camera-aligned joint picking, rotational gizmos and a compact DOF editor."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from joint_dof import (
    JointDof, joint_pointer_angle, joint_rotation_arc, project_joint_points,
)
import theme


AXIS_COLORS = ("#ed625e", "#54c887", "#609ef4")


class JointEditor(QtWidgets.QFrame):
    changed = QtCore.Signal()
    released = QtCore.Signal()
    rotationChanged = QtCore.Signal(int, float)
    resetRequested = QtCore.Signal()
    resetPoseRequested = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("JointEditor")
        self.setFixedWidth(300)
        self._dof = None
        self._angles = None
        self._edited = self._pose_edited = False
        self._updating = False
        self._escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self)
        self._escape_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._escape_shortcut.activated.connect(self.released)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 9)
        layout.setSpacing(5)
        self._name = QtWidgets.QLabel()
        self._name.setMinimumWidth(0)
        self._name.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self._name.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self._name, 0, 0, 1, 2)
        close = QtWidgets.QToolButton()
        close.setText("×")
        close.setToolTip("Release joint selection (Esc)")
        close.clicked.connect(self.released)
        layout.addWidget(close, 0, 2, alignment=QtCore.Qt.AlignRight)
        self._summary = QtWidgets.QLabel()
        layout.addWidget(self._summary, 1, 0, 1, 3)
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, 2, 0, 1, 3)
        rotation_page = QtWidgets.QWidget()
        rotation_layout = QtWidgets.QGridLayout(rotation_page)
        rotation_layout.setContentsMargins(6, 8, 6, 8)
        self._angle_sliders, self._angle_spins = [], []
        for axis, name in enumerate("XYZ"):
            label = QtWidgets.QLabel(name)
            label.setStyleSheet(f"color: {AXIS_COLORS[axis]}; font-weight: bold;")
            rotation_layout.addWidget(label, axis, 0)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setSingleStep(10)
            slider.setPageStep(50)
            slider.setMinimumWidth(65)
            slider.valueChanged.connect(lambda v, a=axis: self._edit_rotation(a, v / 10.0))
            rotation_layout.addWidget(slider, axis, 1)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix("°")
            spin.setKeyboardTracking(False)
            spin.setFixedWidth(83)
            spin.valueChanged.connect(lambda v, a=axis: self._edit_rotation(a, v))
            rotation_layout.addWidget(spin, axis, 2)
            self._angle_sliders.append(slider)
            self._angle_spins.append(spin)
        self._rotation_note = QtWidgets.QLabel()
        self._rotation_note.setWordWrap(True)
        rotation_layout.addWidget(self._rotation_note, 3, 0, 1, 3)
        reset_row = QtWidgets.QHBoxLayout()
        self._reset_joint = QtWidgets.QPushButton("Reset joint")
        self._reset_joint.setToolTip("Restore this joint's original source pose")
        self._reset_joint.clicked.connect(self.resetRequested)
        reset_row.addWidget(self._reset_joint)
        self._reset_pose = QtWidgets.QPushButton("Reset pose edits")
        self._reset_pose.setToolTip("Restore all manually rotated joints at the current frame")
        self._reset_pose.clicked.connect(self.resetPoseRequested)
        reset_row.addWidget(self._reset_pose)
        rotation_layout.addLayout(reset_row, 4, 0, 1, 3)
        self._tabs.addTab(rotation_page, "Rotation")
        limits_page = QtWidgets.QWidget()
        limits_layout = QtWidgets.QGridLayout(limits_page)
        limits_layout.setContentsMargins(6, 8, 6, 8)
        limits_layout.addWidget(QtWidgets.QLabel("Axis"), 0, 0)
        limits_layout.addWidget(QtWidgets.QLabel("Min"), 0, 1)
        limits_layout.addWidget(QtWidgets.QLabel("Max"), 0, 2)
        self._checks, self._minima, self._maxima = [], [], []
        for axis, name in enumerate("XYZ"):
            check = QtWidgets.QCheckBox(name)
            check.setStyleSheet(f"color: {AXIS_COLORS[axis]}; font-weight: bold;")
            check.setToolTip(f"Allow rotation around bone-local {name}")
            spins = []
            for column in (1, 2):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-180.0, 180.0)
                spin.setDecimals(1)
                spin.setSingleStep(5.0)
                spin.setSuffix("°")
                spin.setKeyboardTracking(False)
                spin.setFixedWidth(84)
                spin.setToolTip("Allowed angular offset around the displayed local axis")
                spin.valueChanged.connect(
                    lambda _v, a=axis, c=column: self._edit_axis(a, c))
                limits_layout.addWidget(spin, axis + 1, column)
                spins.append(spin)
            check.toggled.connect(lambda _on, a=axis: self._edit_axis(a, 0))
            limits_layout.addWidget(check, axis + 1, 0)
            self._checks.append(check)
            self._minima.append(spins[0])
            self._maxima.append(spins[1])
        note = QtWidgets.QLabel("Drag the limit handles · Esc cancels a drag")
        note.setWordWrap(True)
        note.setToolTip("Click empty space or press Esc again to release the joint")
        limits_layout.addWidget(note, 4, 0, 1, 3)
        self._tabs.addTab(limits_page, "Limits")
        self.apply_theme()

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(self.width())
        return size

    def apply_theme(self):
        dark = theme.is_dark_mode()
        bg = "rgba(20,22,30,235)" if dark else "rgba(245,247,250,245)"
        text = "#dde2ee" if dark else "#1a1e28"
        field = "#2a2e3a" if dark else "#ffffff"
        self.setStyleSheet(
            "QFrame#JointEditor {"
            f"background: {bg}; border: 1px solid #7a8396; border-radius: 6px; }}"
            f"QLabel, QToolButton {{color: {text}; background: transparent;"
            "border: none; font-size: 11px; }"
            f"QDoubleSpinBox {{color: {text}; background: {field};"
            "border: 1px solid #7a8396; border-radius: 3px; padding: 2px; }"
            "QDoubleSpinBox:disabled {color: #7a8396; }"
        )

    def set_joint(self, name: str, dof: JointDof, angles=None,
                  edited=False, pose_edited=False):
        if self._angles is None and angles is not None:
            self._tabs.setCurrentIndex(0)
        self._dof = dof
        self._angles = None if angles is None else np.asarray(angles, dtype=float).copy()
        self._edited, self._pose_edited = bool(edited), bool(pose_edited)
        self._name.setText(QtGui.QFontMetrics(self._name.font()).elidedText(
            name, QtCore.Qt.ElideMiddle, 218))
        self._name.setToolTip(name)
        self._sync()

    def _sync(self):
        self._updating = True
        for axis in range(3):
            enabled = bool(self._dof.enabled[axis])
            self._checks[axis].setChecked(enabled)
            self._minima[axis].setValue(float(self._dof.minimum[axis]))
            self._maxima[axis].setValue(float(self._dof.maximum[axis]))
            self._minima[axis].setEnabled(enabled)
            self._maxima[axis].setEnabled(enabled)
            minimum, maximum = float(self._dof.minimum[axis]), float(self._dof.maximum[axis])
            value = 0.0 if self._angles is None else float(self._angles[axis])
            available = self._angles is not None and enabled and maximum - minimum > 1e-6
            slider, spin = self._angle_sliders[axis], self._angle_spins[axis]
            slider.setRange(round(minimum * 10), round(maximum * 10))
            slider.setValue(round(value * 10))
            # Imported animation can lie outside the authoring range. Display
            # its actual angle without changing the pose just by selecting it.
            spin.setRange(min(minimum, value), max(maximum, value))
            spin.setValue(value)
            slider.setEnabled(available)
            spin.setEnabled(available)
            tooltip = (f"Local {'XYZ'[axis]} from bind pose · {minimum:g}° to {maximum:g}°"
                       if available else "Rotation locked" if self._angles is not None
                       else "Load a skinned rig to rotate its joints")
            slider.setToolTip(tooltip)
            spin.setToolTip(tooltip)
        count = self._dof.count
        self._summary.setText(f"{count} rotational DOF" if count else "Locked · 0 rotational DOF")
        self._rotation_note.setText(
            "Angles from T-pose · live mesh and ellipsoids"
            if self._angles is not None else "Load a skinned rig to rotate its joints.")
        self._reset_joint.setEnabled(self._edited)
        self._reset_pose.setEnabled(self._pose_edited)
        self._updating = False

    def _edit_rotation(self, axis: int, value: float):
        if self._updating or self._angles is None or not self._angle_spins[axis].isEnabled():
            return
        value = float(np.clip(value, self._dof.minimum[axis], self._dof.maximum[axis]))
        self._angles[axis] = value
        self._sync()
        self.rotationChanged.emit(axis, value)

    def _edit_axis(self, axis: int, source: int):
        if self._updating or self._dof is None:
            return
        minimum = self._minima[axis].value()
        maximum = self._maxima[axis].value()
        if minimum > maximum:
            if source == 1:
                maximum = minimum
            else:
                minimum = maximum
        self._dof.set_axis(axis, self._checks[axis].isChecked(), minimum, maximum)
        self._sync()
        self.changed.emit()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.released.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class JointOverlay(QtWidgets.QWidget):
    """Project 3-D joints into a mouse-transparent overlay above all render modes.

    Picking and painting use the same logical-pixel projection, including on
    high-DPI displays. Joints are intentionally visible through the mesh.
    Click releases select; dragging from a joint still orbits the camera.
    A pinned gizmo's limit handles consume their drag and edit angular bounds.
    """

    rotationChanged = QtCore.Signal(int, int, float)
    resetJointRequested = QtCore.Signal(int)
    resetPoseRequested = QtCore.Signal()
    jointSelected = QtCore.Signal(int)
    limitsChanged = QtCore.Signal()

    def __init__(self, view):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._view = view
        self.editor = JointEditor()
        self.editor.changed.connect(self.update)
        self.editor.changed.connect(self.limitsChanged)
        self.editor.released.connect(self.release_selection)
        self.editor.rotationChanged.connect(self._rotate_selected)
        self.editor.resetRequested.connect(self._reset_selected)
        self.editor.resetPoseRequested.connect(self.resetPoseRequested)
        self.positions = np.empty((0, 3), dtype=np.float32)
        self.rotations = np.empty((0, 3, 3), dtype=np.float32)
        self.names = []
        self.dofs = []
        self.angles = None
        self.edited = np.empty(0, dtype=bool)
        self._rig_key = None
        self.hovered = None
        self.selected = None
        self._press_position = None
        self._press_joint = None
        self._dragged = False
        self._hover_handle = None
        self._handle_drag = None
        self._mouse_position = None
        self._camera_signature = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._camera_changed)
        view.setMouseTracking(True)
        view.installEventFilter(self)

    def set_joints(self, positions, rotations=None, names=None, dofs=None,
                   rig_key=None, angles=None, edited=None):
        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        names = list(names) if names is not None else [f"Joint {i}" for i in range(len(positions))]
        if len(names) != len(positions):
            raise ValueError("Joint names must match joint positions")
        new_rig = rig_key is not self._rig_key or names != self.names
        if new_rig:
            self.release_selection()
        self._rig_key = rig_key
        self.positions = positions
        self.rotations = (np.broadcast_to(np.eye(3), (len(positions), 3, 3))
                          if rotations is None else np.asarray(rotations).reshape(-1, 3, 3))
        if len(self.rotations) != len(positions):
            raise ValueError("Joint rotations must match joint positions")
        self.names = names
        self.angles = None if angles is None else np.asarray(angles).reshape(-1, 3)
        if self.angles is not None and len(self.angles) != len(positions):
            raise ValueError("Joint angles must match joint positions")
        self.edited = np.zeros(len(positions), dtype=bool) if edited is None else np.asarray(edited, dtype=bool)
        if self.edited.shape != (len(positions),):
            raise ValueError("Joint edit flags must match joint positions")
        if dofs is not None:
            if len(dofs) != len(positions):
                raise ValueError("Joint DOF settings must match joint positions")
            self.dofs = list(dofs)
        elif new_rig or len(self.dofs) != len(positions):
            self.dofs = [JointDof() for _ in positions]
        if self.selected is not None:
            self._sync_editor()
        elif self._mouse_position is not None and self._press_position is None:
            self._set_hover(self._mouse_position)
        if not len(positions):
            self.set_enabled(False)
        self.update()

    def _sync_editor(self):
        index = self.selected
        self.editor.set_joint(self.names[index], self.dofs[index],
                              None if self.angles is None else self.angles[index],
                              self.edited[index], np.any(self.edited))

    def select_joint(self, index: int):
        if not 0 <= index < len(self.positions):
            return
        self._finish_handle_drag()
        self.selected = index
        self._sync_editor()
        self.editor.show()
        self.editor.raise_()
        self._view._reposition_overlays()
        self.jointSelected.emit(index)
        self.update()

    def _rotate_selected(self, axis, value):
        if self.selected is not None and self.angles is not None:
            self.rotationChanged.emit(self.selected, axis, value)

    def _reset_selected(self):
        if self.selected is not None:
            self.resetJointRequested.emit(self.selected)

    def set_enabled(self, enabled: bool):
        enabled = bool(enabled and len(self.positions))
        self.setVisible(enabled)
        if enabled:
            self._timer.start()
        else:
            self._timer.stop()
            self.release_selection()

    def release_selection(self):
        self._finish_handle_drag()
        self.selected = self.hovered = None
        self._hover_handle = None
        self._press_position = self._press_joint = None
        self._mouse_position = None
        self._dragged = False
        self.editor.hide()
        self._view.unsetCursor()
        self.update()


    def _projection(self):
        viewport = self._view.getViewport()
        if viewport[2] <= 0 or viewport[3] <= 0:
            return None
        transform = self._view.projectionMatrix(viewport, viewport) * self._view.viewMatrix()
        # QMatrix4x4.data() is column-major; NumPy uses row-major.
        return np.asarray(transform.data(), dtype=np.float64).reshape(4, 4).T

    def pick_joint(self, position):
        if self.isHidden() or not len(self.positions):
            return None
        matrix = self._projection()
        if matrix is None:
            return None
        pixels, depth, valid = project_joint_points(
            self.positions, matrix, self._view.width(), self._view.height())
        distance = np.linalg.norm(pixels - [position.x(), position.y()], axis=1)
        candidates = np.flatnonzero(valid & (distance <= 9.0))
        if not len(candidates):
            return None
        # Prefer the nearest screen point; coincident joints pick front-to-back.
        closest = np.min(distance[candidates])
        candidates = candidates[distance[candidates] <= closest + 0.5]
        return int(candidates[np.argmin(depth[candidates])])

    def _set_hover(self, position):
        handle = self.pick_handle(position)
        index = self.pick_joint(position)
        if index != self.hovered or handle != self._hover_handle:
            self.hovered = index
            self._hover_handle = handle
            self.update()
        if handle is not None:
            self._view.setCursor(QtCore.Qt.OpenHandCursor)
        elif index is None:
            self._view.unsetCursor()
        else:
            self._view.setCursor(QtCore.Qt.PointingHandCursor)

    def _gizmo_geometry(self, matrix=None):
        active = self.selected if self.selected is not None else self.hovered
        if active is None:
            return None
        if self._handle_drag is not None:
            return self._handle_drag["geometry"]
        matrix = self._projection() if matrix is None else matrix
        if matrix is None:
            return None
        center = self.positions[active]
        _pixels, _depth, visible = project_joint_points(
            center[None], matrix, self._view.width(), self._view.height())
        if not visible[0]:
            return None
        return dict(joint=active, center=center.copy(), frame=self.rotations[active].copy(),
                    radius=float(self._view.pixelSize(center)) * 54.0,
                    matrix=matrix, width=self._view.width(), height=self._view.height())

    def _limit_handles(self, geometry):
        """Both bounds stay accessible even for 360-degree or zero-width spans."""
        handles = []
        dof = self.dofs[geometry["joint"]]
        for axis in range(3):
            if not dof.enabled[axis]:
                continue
            pair = []
            for bound, factor in (("minimum", 0.85), ("maximum", 1.16)):
                angle = float(getattr(dof, bound)[axis])
                point = joint_rotation_arc(
                    geometry["center"], geometry["frame"], axis,
                    geometry["radius"] * factor, angle, angle)[0]
                endpoint = joint_rotation_arc(
                    geometry["center"], geometry["frame"], axis,
                    geometry["radius"], angle, angle)[0]
                pixels, depth, valid = project_joint_points(
                    np.array([point, endpoint]), geometry["matrix"],
                    geometry["width"], geometry["height"])
                pair.append(dict(axis=axis, bound=bound, angle=angle,
                                 pixel=pixels[0], endpoint=pixels[1],
                                 depth=depth[0], visible=bool(valid[0])))
            if all(handle["visible"] for handle in pair):
                delta = pair[1]["pixel"] - pair[0]["pixel"]
                separation = np.linalg.norm(delta)
                if separation < 16.0:
                    # Edge-on or coincident bounds need separate screen grips.
                    direction = delta / separation if separation > 1 else np.array([0.0, 1.0])
                    midpoint = (pair[0]["pixel"] + pair[1]["pixel"]) * 0.5
                    pair[0]["pixel"] = midpoint - direction * 8.0
                    pair[1]["pixel"] = midpoint + direction * 8.0
            handles.extend(pair)
        return handles

    def pick_handle(self, position):
        if self.isHidden() or self.selected is None:
            return None
        geometry = self._gizmo_geometry()
        if geometry is None:
            return None
        pointer = np.array([position.x(), position.y()])
        candidates = [(float(np.linalg.norm(handle["pixel"] - pointer)), handle)
                      for handle in self._limit_handles(geometry) if handle["visible"]]
        candidates = [(distance, handle) for distance, handle in candidates if distance <= 10]
        if not candidates:
            return None
        _distance, handle = min(candidates, key=lambda entry: (entry[0], entry[1]["depth"]))
        return handle["axis"], handle["bound"]

    @staticmethod
    def _pointer_angle(position, geometry, axis):
        return joint_pointer_angle(
            np.array([position.x(), position.y()]), geometry["matrix"],
            geometry["width"], geometry["height"], geometry["center"],
            geometry["frame"], axis)

    def _start_handle_drag(self, handle, position):
        geometry = self._gizmo_geometry()
        axis, bound = handle
        dof = self.dofs[geometry["joint"]]
        self._handle_drag = dict(
            geometry=geometry, axis=axis, bound=bound,
            original=(float(dof.minimum[axis]), float(dof.maximum[axis])),
            previous_position=position, previous_angle=self._pointer_angle(position, geometry, axis))
        self._hover_handle = handle
        self._press_position = self._press_joint = None
        self._view.setFocus(QtCore.Qt.MouseFocusReason)
        self._view._nav_keys.clear()
        self._view._nav_timer.stop()
        if self._view.isVisible():
            self._view.grabMouse()
        self._view.setCursor(QtCore.Qt.ClosedHandCursor)
        self.update()

    def _move_handle_drag(self, position):
        drag = self._handle_drag
        geometry, axis, bound = drag["geometry"], drag["axis"], drag["bound"]
        angle = self._pointer_angle(position, geometry, axis)
        previous_angle = drag["previous_angle"]
        if angle is not None and previous_angle is not None:
            delta = (angle - previous_angle + 180.0) % 360.0 - 180.0
        else:
            movement = position - drag["previous_position"]
            # Looking directly at the edge: vertical dragging remains usable.
            delta = -movement.y()
            if abs(movement.x()) > abs(movement.y()):
                current = float(getattr(self.dofs[geometry["joint"]], bound)[axis])
                tangent = joint_rotation_arc(
                    geometry["center"], geometry["frame"], axis,
                    geometry["radius"], current - 1, current + 1)[[0, -1]]
                pixels, _depth, valid = project_joint_points(
                    tangent, geometry["matrix"], geometry["width"], geometry["height"])
                derivative = (pixels[1] - pixels[0]) * 0.5
                squared = float(derivative @ derivative)
                if np.all(valid) and squared > 0.01:
                    delta = float(np.array([movement.x(), movement.y()]) @ derivative) / squared
        dof = self.dofs[geometry["joint"]]
        value = float(getattr(dof, bound)[axis]) + delta
        if bound == "minimum":
            minimum = round(float(np.clip(value, -180.0, dof.maximum[axis])), 1)
            maximum = float(dof.maximum[axis])
        else:
            minimum = float(dof.minimum[axis])
            maximum = round(float(np.clip(value, dof.minimum[axis], 180.0)), 1)
        dof.set_axis(axis, True, minimum, maximum)
        drag["previous_position"] = position
        drag["previous_angle"] = angle
        self.editor._sync()
        self.update()

    def _finish_handle_drag(self, cancel=False):
        drag = self._handle_drag
        if drag is None:
            return
        if cancel:
            self.dofs[drag["geometry"]["joint"]].set_axis(
                drag["axis"], True, *drag["original"])
            self.editor._sync()
        self._handle_drag = None
        self._hover_handle = None
        if QtWidgets.QWidget.mouseGrabber() is self._view:
            self._view.releaseMouse()
        self._view.unsetCursor()
        self.update()
        self.limitsChanged.emit()

    def eventFilter(self, obj, event):
        if obj is not self._view or self.isHidden():
            return False
        kind = event.type()
        if self._handle_drag is not None:
            if kind == QtCore.QEvent.MouseMove:
                self._mouse_position = event.position()
                self._move_handle_drag(event.position())
                return True
            if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                self._move_handle_drag(event.position())
                self._finish_handle_drag()
                self._set_hover(event.position())
                return True
            if kind == QtCore.QEvent.KeyPress:
                if event.key() == QtCore.Qt.Key_Escape:
                    self._finish_handle_drag(cancel=True)
                return True
            if kind in (QtCore.QEvent.Wheel, QtCore.QEvent.MouseButtonPress,
                        QtCore.QEvent.Leave):
                return True
            if kind in (QtCore.QEvent.FocusOut, QtCore.QEvent.Resize):
                self._finish_handle_drag()
        if kind == QtCore.QEvent.MouseMove:
            position = event.position()
            self._mouse_position = position
            if event.buttons() != QtCore.Qt.NoButton:
                if self._press_position is not None:
                    self._dragged |= ((position - self._press_position).manhattanLength()
                                      >= QtWidgets.QApplication.startDragDistance())
                self.hovered = None
                self._view.unsetCursor()
                self.update()
            else:
                self._set_hover(position)
        elif kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonDblClick) and event.button() == QtCore.Qt.LeftButton:
            handle = self.pick_handle(event.position())
            if handle is not None:
                self._start_handle_drag(handle, event.position())
                return True
            self._press_position = event.position()
            self._press_joint = self.pick_joint(event.position())
            self._dragged = False
        elif kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            if self._press_position is not None:
                dragged = self._dragged or (
                    (event.position() - self._press_position).manhattanLength()
                    >= QtWidgets.QApplication.startDragDistance())
                index = self.pick_joint(event.position())
                if not dragged and index == self._press_joint:
                    if index is None:
                        self.release_selection()
                    else:
                        self.select_joint(index)
                self._press_position = self._press_joint = None
        elif kind == QtCore.QEvent.Leave:
            self._mouse_position = None
            self.hovered = None
            self._hover_handle = None
            self._view.unsetCursor()
            self.update()
        elif kind == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
            if self.selected is not None:
                self.release_selection()
                return True
        return False

    def _camera_changed(self):
        matrix = self._projection()
        if matrix is None:
            return
        signature = (self._view.width(), self._view.height(), matrix.tobytes())
        if signature != self._camera_signature:
            self._camera_signature = signature
            if (self._mouse_position is not None and self._press_position is None
                    and self._handle_drag is None):
                self._set_hover(self._mouse_position)
            self.update()

    @staticmethod
    def _draw_polyline(painter, pixels, valid):
        # Open rim/guide paths must never inherit a pie sector's fill brush.
        painter.setBrush(QtCore.Qt.NoBrush)
        path = QtGui.QPainterPath()
        previous_valid = False
        for point, visible in zip(pixels, valid):
            if visible:
                p = QtCore.QPointF(float(point[0]), float(point[1]))
                if previous_valid:
                    path.lineTo(p)
                else:
                    path.moveTo(p)
            previous_valid = bool(visible)
        painter.drawPath(path)

    @staticmethod
    def _draw_label(painter, label, point, width, height):
        box = QtGui.QFontMetrics(painter.font()).boundingRect(label)
        x = np.clip(point.x() + 12, 4, max(4, width - box.width() - 16))
        y = np.clip(point.y() - 23, 4, max(4, height - box.height() - 8))
        rect = QtCore.QRectF(float(x), float(y), box.width() + 12, box.height() + 6)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(24, 27, 35, 225))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QtGui.QColor("#f9fafc"))
        painter.drawText(rect, QtCore.Qt.AlignCenter, label)

    def _draw_limit_handles(self, painter, geometry):
        active_handle = ((self._handle_drag["axis"], self._handle_drag["bound"])
                         if self._handle_drag is not None else self._hover_handle)
        tooltip = None
        for handle in self._limit_handles(geometry):
            if not handle["visible"]:
                continue
            identity = (handle["axis"], handle["bound"])
            highlighted = identity == active_handle
            point = QtCore.QPointF(*map(float, handle["pixel"]))
            endpoint = QtCore.QPointF(*map(float, handle["endpoint"]))
            color = QtGui.QColor(AXIS_COLORS[handle["axis"]])
            painter.setPen(QtGui.QPen(color, 1.2))
            painter.drawLine(endpoint, point)
            painter.setPen(QtGui.QPen(QtGui.QColor("#f9fafc"), 1.6 if highlighted else 1.0))
            painter.setBrush(color)
            radius = 7.0 if highlighted else 5.5
            painter.drawEllipse(point, radius, radius)
            if handle["bound"] == "minimum":
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor("#f9fafc"))
                painter.drawEllipse(point, 1.6, 1.6)
            if highlighted:
                bound = "Min" if handle["bound"] == "minimum" else "Max"
                label = f"{'XYZ'[handle['axis']]} {bound}: {handle['angle']:+.1f}°"
                tooltip = label, point
        if tooltip is not None:
            self._draw_label(painter, *tooltip, self._view.width(), self._view.height())

    def paintEvent(self, event):
        matrix = self._projection()
        if matrix is None or not len(self.positions):
            return
        width, height = self._view.width(), self._view.height()
        pixels, depth, valid = project_joint_points(self.positions, matrix, width, height)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        active = self.selected if self.selected is not None else self.hovered
        geometry = self._gizmo_geometry(matrix)
        if geometry is not None:
            center, frame = geometry["center"], geometry["frame"]
            dof = self.dofs[active]
            radius = geometry["radius"]
            center_pixels, _depth, _valid = project_joint_points(center[None], matrix, width, height)
            for axis in range(3):
                color = QtGui.QColor(AXIS_COLORS[axis])
                full_arc = joint_rotation_arc(center, frame, axis, radius)
                ring_pixels, _depth, ring_valid = project_joint_points(full_arc, matrix, width, height)
                guide_color = QtGui.QColor(color)
                guide_color.setAlpha(80)
                pen = QtGui.QPen(guide_color, 1.0, QtCore.Qt.DashLine)
                painter.setPen(pen)
                self._draw_polyline(painter, ring_pixels, ring_valid)
                enabled = dof.enabled[axis] and dof.maximum[axis] > dof.minimum[axis]
                if enabled:
                    arc = joint_rotation_arc(center, frame, axis, radius,
                                             dof.minimum[axis], dof.maximum[axis])
                    arc_pixels, _depth, arc_valid = project_joint_points(arc, matrix, width, height)
                    # A projected triangle fan is a bone-local pie sector.
                    # Let QPainter clip at viewport edges instead of dropping
                    # an otherwise visible sector when its rim exits the view.
                    in_front = np.isfinite(arc_pixels).all(axis=1) & (np.abs(_depth) <= 1)
                    if np.all(in_front):
                        sector = QtGui.QPainterPath()
                        sector.moveTo(QtCore.QPointF(*map(float, center_pixels[0])))
                        for point in arc_pixels:
                            sector.lineTo(QtCore.QPointF(*map(float, point)))
                        sector.closeSubpath()
                        fill = QtGui.QColor(color)
                        dragging_axis = (self._handle_drag is not None
                                         and self._handle_drag["axis"] == axis)
                        fill.setAlpha(85 if dragging_axis else 48 if self.selected is not None else 30)
                        painter.setPen(QtCore.Qt.NoPen)
                        painter.setBrush(fill)
                        painter.drawPath(sector)
                    if dof.maximum[axis] - dof.minimum[axis] < 359.9:
                        painter.setPen(QtGui.QPen(color, 1.0))
                        for endpoint in arc_pixels[[0, -1]]:
                            if np.all(np.isfinite(endpoint)):
                                painter.drawLine(QtCore.QPointF(*map(float, center_pixels[0])),
                                                 QtCore.QPointF(*map(float, endpoint)))
                    painter.setPen(QtGui.QPen(color, 2.5))
                    self._draw_polyline(painter, arc_pixels, arc_valid)
                axis_line = np.array([center, center + frame[:, axis] * radius * 1.25])
                line_pixels, _depth, line_valid = project_joint_points(axis_line, matrix, width, height)
                if not enabled:
                    color.setAlpha(100)
                painter.setPen(QtGui.QPen(color, 1.5))
                self._draw_polyline(painter, line_pixels, line_valid)
                if line_valid[1]:
                    end = QtCore.QPointF(*map(float, line_pixels[1]))
                    painter.drawText(end + QtCore.QPointF(4, -4), "XYZ"[axis])
        # Front joints paint last, matching the depth tie-break used by picking.
        for index in np.argsort(-depth):
            if not valid[index]:
                continue
            point = QtCore.QPointF(*map(float, pixels[index]))
            highlighted = index == active
            painter.setPen(QtGui.QPen(QtGui.QColor(
                "#f9fafc" if highlighted else "#352719"), 1.3))
            painter.setBrush(QtGui.QColor("#ffe49a" if highlighted else "#ff8b35"))
            painter.drawEllipse(point, 6.0 if highlighted else 4.0, 6.0 if highlighted else 4.0)
        if active is not None and valid[active] and self._hover_handle is None:
            point = QtCore.QPointF(*map(float, pixels[active]))
            count = self.dofs[active].count
            hint = "" if self.selected is not None else " · click to edit"
            name = QtGui.QFontMetrics(painter.font()).elidedText(
                self.names[active], QtCore.Qt.ElideMiddle, 180)
            label = f"{name} · {count} DOF{hint}"
            self._draw_label(painter, label, point, width, height)
        if self.selected is not None and geometry is not None:
            self._draw_limit_handles(painter, geometry)
        painter.end()
