"""Dockable joint-angle versus primitive-parameter plots from stored fits."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from morph_curve_data import MorphCurveSeries, build_morph_curve_series
import theme


PARAMETERS = (
    ("Exponents", "exponents", "Exponent", ("ε1", "ε2")),
    ("Scale / base", "scale", "Radius / base radius", ("X", "Y", "Z")),
    ("Radii (local)", "radii", "Local radius", ("X", "Y", "Z")),
    ("Position (local)", "position", "Local position", ("X", "Y", "Z")),
    ("Position Δ (local)", "position_delta", "Position Δ from base", ("X", "Y", "Z")),
    ("Rotation Δ (°)", "rotation", "Rotation Δ from base (°)", ("X", "Y", "Z")),
)


class MorphCurvePanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rig = self._library = None
        self._series = MorphCurveSeries()
        self._order = np.empty(0, dtype=int)
        self._curves = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        controls = QtWidgets.QGridLayout()
        self._joint = QtWidgets.QComboBox()
        self._primitive = QtWidgets.QComboBox()
        self._axis = QtWidgets.QComboBox()
        for label, value in (("Auto (largest range)", "auto"), ("Local X", "x"),
                ("Local Y", "y"), ("Local Z", "z"), ("Total angle (unsigned)", "total")):
            self._axis.addItem(label, value)
        self._parameter = QtWidgets.QComboBox()
        for label, value, _units, _channels in PARAMETERS:
            self._parameter.addItem(label, value)
        for row, (label, widget) in enumerate((("Joint", self._joint),
                ("Primitive", self._primitive), ("Angle axis", self._axis),
                ("Parameter", self._parameter))):
            widget.setMinimumContentsLength(12)
            widget.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            controls.addWidget(QtWidgets.QLabel(label), row, 0)
            controls.addWidget(widget, row, 1)
        layout.addLayout(controls)
        switches = QtWidgets.QHBoxLayout()
        self._all_primitives = QtWidgets.QCheckBox("All primitives")
        self._connect = QtWidgets.QCheckBox("Connect samples")
        switches.addWidget(self._all_primitives)
        switches.addWidget(self._connect)
        switches.addStretch()
        layout.addLayout(switches)
        self._status = QtWidgets.QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=.25)
        self._plot.setMinimumHeight(130)
        self._plot.setMenuEnabled(False)
        self._legend = self._plot.addLegend()
        self._selection = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#a0a7b4", style=QtCore.Qt.DashLine))
        self._plot.addItem(self._selection)
        self._selection.hide()
        splitter.addWidget(self._plot)
        self._table = QtWidgets.QTableWidget()
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.verticalHeader().hide()
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self._table)
        splitter.setSizes([300, 150])
        layout.addWidget(splitter, 1)
        self._detail = QtWidgets.QLabel("Select a sample to inspect its pose and values.")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)
        hint = QtWidgets.QLabel("Stored fit samples; other joints may vary between poses. Connecting samples is a visual guide.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._joint.currentIndexChanged.connect(self._populate_primitives)
        self._primitive.currentIndexChanged.connect(self._rebuild_series)
        self._all_primitives.toggled.connect(self._populate_primitives)
        self._axis.currentIndexChanged.connect(self._redraw)
        self._parameter.currentIndexChanged.connect(self._redraw)
        self._connect.toggled.connect(self._redraw)
        self._table.itemSelectionChanged.connect(self._select_table_sample)
        self.apply_theme()
        self.set_source(None, None)

    def set_source(self, rig, library):
        if rig is self._rig and library is self._library and self._joint.count():
            return
        previous = self._joint.currentData() if rig is self._rig else None
        had_library = self._library is not None
        self._rig, self._library = rig, library
        with QtCore.QSignalBlocker(self._joint):
            self._joint.clear()
            if rig is not None and rig.skeleton.num_bones:
                counts = (np.bincount(library.base.bone_assignments, minlength=rig.skeleton.num_bones)
                          if library is not None else np.zeros(rig.skeleton.num_bones, dtype=int))
                for index, bone in enumerate(rig.skeleton.bones):
                    self._joint.addItem(f"{bone.name.rsplit(':', 1)[-1]} ({counts[index]} primitives)", index)
                    self._joint.setItemData(index, bone.name, QtCore.Qt.ToolTipRole)
                initial = self._joint.findData(previous) if previous is not None else -1
                if initial >= 0 and not had_library and not counts[initial]:
                    initial = -1
                self._joint.setCurrentIndex(initial if initial >= 0 else int(np.argmax(counts)))
        available = rig is not None and library is not None
        for widget in (self._joint, self._primitive, self._axis, self._parameter, self._all_primitives, self._connect):
            widget.setEnabled(available)
        self._populate_primitives()

    def _populate_primitives(self, *_):
        previous = self._primitive.currentData()
        with QtCore.QSignalBlocker(self._primitive):
            self._primitive.clear()
            if self._library is not None:
                for index, joint in enumerate(self._library.base.bone_assignments):
                    if self._all_primitives.isChecked() or joint == self._joint.currentData():
                        self._primitive.addItem(f"ID {index} · {self._library.base.primitive_type}", index)
                selected = self._primitive.findData(previous)
                self._primitive.setCurrentIndex(selected if selected >= 0 else 0)
        self._rebuild_series()

    def _rebuild_series(self, *_):
        joint, primitive = self._joint.currentData(), self._primitive.currentData()
        self._series = (build_morph_curve_series(self._rig.skeleton, self._library, joint, primitive)
                       if self._rig is not None and self._library is not None
                       and joint is not None and primitive is not None else MorphCurveSeries())
        self._redraw()

    def _redraw(self, *_):
        for curve in self._curves:
            self._plot.removeItem(curve)
        self._curves.clear()
        self._legend.clear()
        self._selection.hide()
        self._detail.setText("Select a sample to inspect its pose and values.")
        parameter = self._parameter.currentData()
        _label, _key, units, channels = next(p for p in PARAMETERS if p[1] == parameter)
        axis = self._series.resolved_axis(self._axis.currentData())
        self._plot.setLabel("bottom", "Joint angle / base (°) · " + ("total" if axis == "total" else axis.upper()))
        self._plot.setLabel("left", units)
        angles = self._series.angles(self._axis.currentData())
        values = self._series.values(parameter)
        self._order = np.argsort(angles, kind="stable")
        colors = ("#38a9ee", "#ef9d40", "#ab7bea")
        for channel, name in enumerate(channels):
            if not len(angles):
                break
            curve = self._plot.plot(angles[self._order], values[self._order, channel],
                name=name, pen=pg.mkPen(colors[channel], width=2) if self._connect.isChecked() else None,
                symbol="o",
                symbolSize=8, symbolBrush=colors[channel], symbolPen=colors[channel],
                data=list(range(len(self._order))))
            curve.sigPointsClicked.connect(self._select_plot_sample)
            self._curves.append(curve)
        self._plot.enableAutoRange()
        with QtCore.QSignalBlocker(self._table):
            self._table.clear()
            self._table.setColumnCount(2 + len(channels))
            self._table.setHorizontalHeaderLabels(["Pose", "Angle (°)", *channels])
            self._table.setRowCount(len(self._order))
            for row, index in enumerate(self._order):
                sample = self._series.samples[index]
                fields = [sample.name, f"{angles[index]:.5g}", *[f"{v:.6g}" for v in values[index]]]
                for column, value in enumerate(fields):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setToolTip(f"{sample.name}\n" + ("Base pose" if sample.key_index < 0 else f"Pose key {sample.key_index}"))
                    self._table.setItem(row, column, item)
        if self._rig is None:
            message = "Load a rigged model to inspect morph curves."
        elif self._library is None:
            message = "Train pose correctives or load a morph fit to see its captured curves."
        elif not self._series.samples:
            message = "No samples for this selection. Choose another joint or enable All primitives."
        else:
            message = f"{len(self._series.samples) - 1} fitted samples + base · ID {self._primitive.currentData()}"
            if self._series.skipped:
                message += f" · {self._series.skipped} skipped (missing pose/parameters or invalid values)"
        self._status.setText(message)

    def _select_plot_sample(self, _curve, points, _event):
        if points:
            self._table.selectRow(int(points[0].data()))

    def _select_table_sample(self):
        row = self._table.currentRow()
        if not 0 <= row < len(self._order):
            return
        index = self._order[row]
        sample = self._series.samples[index]
        angle = self._series.angles(self._axis.currentData())[index]
        values = self._series.values(self._parameter.currentData())[index]
        channels = next(p[3] for p in PARAMETERS if p[1] == self._parameter.currentData())
        self._selection.setValue(float(angle))
        self._selection.show()
        self._detail.setText(f"{sample.name} · {angle:.5g}° · "
            + " · ".join(f"{name} = {value:.6g}" for name, value in zip(channels, values)))

    def apply_theme(self):
        self._plot.setBackground(theme.bg((2, 11, 13)))
        for name in ("left", "bottom"):
            self._plot.getAxis(name).setPen(theme.pg_fg())
            self._plot.getAxis(name).setTextPen(theme.pg_fg())
