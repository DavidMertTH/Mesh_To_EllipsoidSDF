"""settings_dialog.py — the 'Settings' window.

A small tabbed dialog built automatically from ``app_settings.SETTINGS_SPEC``.
It holds the detailed fitting / maintenance knobs that would otherwise clutter
the main window (how many ellipsoids merge / spawn / split per SuperFit cycle,
local-fit resolution, loss weights, sampling budget, …), plus an *Appearance*
tab that hosts the theme colour pickers.

The dialog only edits a working copy; the caller reads :meth:`values` on accept
and persists them via :func:`app_settings.save`.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

import app_settings


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, values: dict,
                 color_row: QtWidgets.QWidget | None = None,
                 mode_row: QtWidgets.QWidget | None = None,
                 parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self._widgets: dict[str, QtWidgets.QWidget] = {}

        root = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)

        for tab_title, groups in app_settings.SETTINGS_SPEC:
            page = QtWidgets.QWidget()
            pv = QtWidgets.QVBoxLayout(page)
            pv.setContentsMargins(8, 8, 8, 8)
            pv.setSpacing(8)
            for grp_title, fields in groups:
                box = QtWidgets.QGroupBox(grp_title)
                form = QtWidgets.QFormLayout(box)
                form.setLabelAlignment(QtCore.Qt.AlignLeft)
                for field in fields:
                    key, label, kind, mn, mx, step, dec, default, tip = field
                    w = self._make_widget(kind, mn, mx, step, dec)
                    self._set_widget_value(w, values.get(key, default))
                    if tip:
                        w.setToolTip(tip)
                    self._widgets[key] = w
                    form.addRow(f"{label}:", w)
                pv.addWidget(box)
            pv.addStretch(1)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            tabs.addTab(scroll, tab_title)

        # Appearance tab — light/dark mode controls + the theme-colour swatches.
        if color_row is not None or mode_row is not None:
            page = QtWidgets.QWidget()
            pv = QtWidgets.QVBoxLayout(page)
            pv.setContentsMargins(8, 8, 8, 8)
            pv.setSpacing(8)
            if mode_row is not None:
                mbox = QtWidgets.QGroupBox("Mode")
                ml = QtWidgets.QVBoxLayout(mbox)
                ml.addWidget(mode_row)
                pv.addWidget(mbox)
            if color_row is not None:
                box = QtWidgets.QGroupBox("Theme colors")
                bl = QtWidgets.QVBoxLayout(box)
                bl.addWidget(color_row)
                pv.addWidget(box)
            pv.addStretch(1)
            tabs.addTab(page, "Appearance")

        # Buttons: Restore Defaults | Cancel | OK
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.RestoreDefaults)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults)
        root.addWidget(btns)

        self.setMinimumSize(440, 520)

    @staticmethod
    def _make_widget(kind, mn, mx, step, dec) -> QtWidgets.QWidget:
        if kind == "bool":
            return QtWidgets.QCheckBox()
        if kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(mn), int(mx))
            w.setSingleStep(int(step))
        else:
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(float(mn), float(mx))
            w.setSingleStep(float(step))
            w.setDecimals(int(dec))
        return w

    @staticmethod
    def _set_widget_value(w: QtWidgets.QWidget, value) -> None:
        if isinstance(w, QtWidgets.QCheckBox):
            w.setChecked(bool(value))
        else:
            w.setValue(value)

    @staticmethod
    def _widget_value(w: QtWidgets.QWidget):
        if isinstance(w, QtWidgets.QCheckBox):
            return bool(w.isChecked())
        return w.value()

    def load_values(self, values: dict) -> None:
        """Refresh every widget from *values* (used when reopening the dialog)."""
        for key, w in self._widgets.items():
            if key in values:
                self._set_widget_value(w, values[key])

    def _restore_defaults(self) -> None:
        defaults = app_settings.defaults()
        for key, w in self._widgets.items():
            if key in defaults:
                self._set_widget_value(w, defaults[key])

    def values(self) -> dict:
        """The current value of every setting widget."""
        return {key: self._widget_value(w) for key, w in self._widgets.items()}
