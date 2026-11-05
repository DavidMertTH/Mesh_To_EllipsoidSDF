"""Dockable live view of the total primitive fit error in the current pose."""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

import theme

SERIES = (
    ("none", "No morph", "#8a8f98"),
    ("keys", "Morph keys", "#e8a33d"),
    ("function", "Joint functions", "#3fa7d6"),
)
LABELS = {key: label for key, label, _ in SERIES}
HISTORY_SECONDS = 20.0


class MorphErrorPanel(QtWidgets.QWidget):
    """Shows the current total error and its history while an animation plays.

    The error is the mean absolute distance of the primitive union to the posed
    mesh surface, in percent of the body height.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = {key: deque() for key, _, _ in SERIES}
        self._times = {key: deque() for key, _, _ in SERIES}
        self._sums = {key: 0.0 for key, _, _ in SERIES}
        self._counts = {key: 0 for key, _, _ in SERIES}
        self._active = "none"
        self.last_errors: dict = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._headline = QtWidgets.QLabel()
        self._headline.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self._headline)
        self._details = QtWidgets.QLabel()
        self._details.setWordWrap(True)
        self._details.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self._details)
        self._average = QtWidgets.QLabel()
        self._average.setWordWrap(True)
        self._average.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self._average)

        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=.25)
        self._plot.setMinimumHeight(120)
        self._plot.setMenuEnabled(False)
        self._plot.setLabel("left", "Error (% of height)")
        self._plot.getAxis("left").enableAutoSIPrefix(False)
        self._plot.setLabel("bottom", "Playback time (s)")
        self._legend = self._plot.addLegend(offset=(-10, 10))
        self._legend.setZValue(100)
        self._curves = {key: self._plot.plot([], [], name=label, pen=pg.mkPen(color, width=1.5))
                        for key, label, color in SERIES}
        layout.addWidget(self._plot, 1)

        row = QtWidgets.QHBoxLayout()
        hint = QtWidgets.QLabel("Mean distance of the primitive surface to the posed mesh surface "
                                "(2,500 sampled vertices). 0 = perfect cover.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: gray;")
        row.addWidget(hint, 1)
        self._btn_reset = QtWidgets.QPushButton("Reset averages")
        self._btn_reset.clicked.connect(self.clear_history)
        row.addWidget(self._btn_reset)
        layout.addLayout(row)
        self.apply_theme()
        self.set_unavailable("Fit a base pose to measure the error.")

    # ── state ──
    def set_unavailable(self, message: str) -> None:
        self.last_errors = {}
        self._headline.setText(f"<span style='font-size:15px'><b>Total error: –</b></span>")
        self._details.setText(message)
        self._average.setText("")

    def clear_history(self) -> None:
        for key, _, _ in SERIES:
            self._history[key].clear()
            self._times[key].clear()
            self._sums[key] = 0.0
            self._counts[key] = 0
            self._curves[key].setData([], [])
        self._average.setText("")

    def update_errors(self, errors: dict, active: str, *, height: float | None = None,
                      playback_time: float | None = None) -> None:
        """Show *errors* (series key → SurfaceError); append to the history while playing."""
        self.last_errors = dict(errors)
        self._active = active if active in errors else next(iter(errors), "none")
        current = errors.get(self._active)
        if current is None:
            self.set_unavailable("No primitives to measure.")
            return
        color = dict((k, c) for k, _, c in SERIES)[self._active]
        absolute = (f" · mean distance {current.mean_absolute:.5f} model units "
                    f"(height {height:.3g})" if height else "")
        self._headline.setText(
            f"<span style='font-size:15px'><b>Total error: "
            f"<span style='color:{color}'>{current.mean_percent:.4f} %</span></b></span>"
            f" <span style='color:gray'>of height ({LABELS[self._active]}){absolute}</span>")
        parts = []
        none = errors.get("none")
        for key, label, color_ in SERIES:
            if key not in errors:
                continue
            value = errors[key].mean_percent
            gain = ""
            if none is not None and key != "none" and none.mean_percent > 0:
                change = (value / none.mean_percent - 1) * 100
                gain = f" ({'−' if change < 0 else '+'}{abs(change):.0f} % vs. no morph)"
            weight = "b" if key == self._active else "span"
            parts.append(f"<{weight} style='color:{color_}'>{label}: {value:.4f} %{gain}</{weight}>")
        p95 = current.p95_percent
        self._details.setText(" · ".join(parts) + f"<br><span style='color:gray'>95th percentile "
                              f"({LABELS[self._active]}): {p95:.3f} %</span>")
        if playback_time is None:
            return
        for key in errors:
            value = errors[key].mean_percent
            times, history = self._times[key], self._history[key]
            if times and playback_time < times[-1]:
                times.clear()
                history.clear()
            times.append(playback_time)
            history.append(value)
            while times and times[0] < playback_time - HISTORY_SECONDS:
                times.popleft()
                history.popleft()
            self._curves[key].setData(np.fromiter(times, float), np.fromiter(history, float))
            self._sums[key] += value
            self._counts[key] += 1
        for key, _, _ in SERIES:
            if key not in errors and self._counts[key] == 0:
                self._curves[key].setData([], [])
        averages = [f"<span style='color:{c}'>{label} {self._sums[k] / self._counts[k]:.4f} %</span>"
                    for k, label, c in SERIES if self._counts[k]]
        frames = max(self._counts.values())
        self._average.setText(f"Average while playing ({frames} frames): " + " · ".join(averages))

    def averages(self) -> dict:
        return {k: self._sums[k] / self._counts[k] for k, _, _ in SERIES if self._counts[k]}

    def apply_theme(self) -> None:
        background = theme.bg((2, 11, 13))
        self._plot.setBackground(background)
        self._legend.setBrush(pg.mkBrush(*background, 210))
        self._legend.setLabelTextColor("#d0d4da" if theme.is_dark_mode() else "#30343a")
        for name in ("left", "bottom"):
            self._plot.getAxis(name).setPen(theme.pg_fg())
            self._plot.getAxis(name).setTextPen(theme.pg_fg())
