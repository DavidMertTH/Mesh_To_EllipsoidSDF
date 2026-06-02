"""
dashboard.py — compact live "at a glance" panel for the running fit.

Complements the RunTrackerPanel (which owns the full loss curve + run history):
this is a quick status board with headline stat cards (step / loss / speed / ETA
/ …) and two small live graphs (loss, ellipsoid count) so the most important
numbers are visible without reading the run tracker.

Fed from MainWindow's existing optimisation signal handlers via:
  begin(total_steps, mesh_name, num_ellipsoids)
  record(step, loss, n_ellipsoids, radii=None)
  set_phase(phase)   finish()   reset()
"""

from __future__ import annotations

import time

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

import theme


def _fmt_time(seconds: float) -> str:
    """Seconds → ``m:ss`` (or ``h:mm:ss``)."""
    if seconds is None or seconds < 0 or not np.isfinite(seconds):
        return "—"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_loss(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.4g}" if v >= 1e-3 else f"{v:.2e}"


class DashboardPanel(QtWidgets.QWidget):
    """Headline stat cards + two compact live graphs for the current fit."""

    # (key, title) for each stat card, laid out left→right, top→bottom.
    _CARDS = (
        ("step", "Step"), ("loss", "Loss"), ("best", "Best loss"),
        ("ellipsoids", "Ellipsoids"),
        ("phase", "Phase"), ("elapsed", "Elapsed"),
        ("speed", "Speed"), ("eta", "ETA"),
    )
    _CARD_COLS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict[str, QtWidgets.QLabel] = {}

        # run state
        self._total = 0
        self._t0: float | None = None        # wall clock of first recorded step
        self._step0 = 0
        self._best: float | None = None
        self._steps: list[int] = []
        self._losses: list[float] = []
        self._counts: list[int] = []

        self._build_ui()
        self.apply_theme()
        self.reset()

    # ── public API (called from MainWindow) ────────────────────────────

    def begin(self, total_steps: int, mesh_name: str = "",
              num_ellipsoids: int = 0) -> None:
        self.reset()
        self._total = max(1, int(total_steps))
        self._set("phase", "PREP")
        self._set("ellipsoids", str(int(num_ellipsoids)))
        self._set("step", f"0 / {self._total}")

    def record(self, step: int, loss: float, n_ellipsoids: int,
               radii: np.ndarray | None = None) -> None:
        now = time.monotonic()
        if self._t0 is None:                  # first real step → start the clock
            self._t0, self._step0 = now, int(step)

        loss = float(loss)
        if np.isfinite(loss) and (self._best is None or loss < self._best):
            self._best = loss

        self._steps.append(int(step))
        self._losses.append(loss)
        self._counts.append(int(n_ellipsoids))

        frac = min(1.0, step / self._total)
        elapsed = now - self._t0
        done = max(step - self._step0, 0)
        speed = done / elapsed if elapsed > 1e-6 and done > 0 else 0.0
        eta = (self._total - step) / speed if speed > 1e-9 else None

        self._set("step", f"{int(step)} / {self._total}  ({frac*100:.0f}%)")
        self._set("loss", _fmt_loss(loss))
        self._set("best", _fmt_loss(self._best))
        self._set("ellipsoids", str(int(n_ellipsoids)))
        self._set("elapsed", _fmt_time(elapsed))
        self._set("speed", f"{speed:.1f}/s" if speed else "—")
        self._set("eta", _fmt_time(eta))

        self._curve_loss.setData(self._steps, self._losses)
        self._curve_count.setData(self._steps, self._counts)

    def set_phase(self, phase: str) -> None:
        self._set("phase", str(phase).upper())

    def finish(self) -> None:
        self._set("phase", "DONE")

    def reset(self) -> None:
        self._total = 0
        self._t0 = None
        self._step0 = 0
        self._best = None
        self._steps.clear(); self._losses.clear(); self._counts.clear()
        self._curve_loss.setData([], [])
        self._curve_count.setData([], [])
        for key, _title in self._CARDS:
            self._set(key, "—")

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("Dashboard")
        title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: palette(window-text);")
        layout.addWidget(title)

        # ── stat cards grid ──
        cards = QtWidgets.QGridLayout()
        cards.setHorizontalSpacing(6)
        cards.setVerticalSpacing(6)
        for i, (key, ttl) in enumerate(self._CARDS):
            cards.addWidget(self._make_card(key, ttl),
                            i // self._CARD_COLS, i % self._CARD_COLS)
        layout.addLayout(cards)

        # ── two compact graphs ──
        self._plot_loss = self._make_plot("Loss", log_y=True)
        self._curve_loss = self._plot_loss.plot(
            [], [], pen=pg.mkPen(theme.YELLOW, width=2))
        layout.addWidget(self._plot_loss, stretch=1)

        self._plot_count = self._make_plot("Ellipsoids", log_y=False)
        self._curve_count = self._plot_count.plot(
            [], [], pen=pg.mkPen(theme.BLUE, width=2))
        layout.addWidget(self._plot_count, stretch=1)

    def _make_card(self, key: str, title: str) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("dashCard")
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(1)
        cap = QtWidgets.QLabel(title)
        cap.setObjectName("dashCap")
        val = QtWidgets.QLabel("—")
        val.setObjectName("dashVal")
        v.addWidget(cap)
        v.addWidget(val)
        self._cards[key] = val
        return card

    def _make_plot(self, ylabel: str, log_y: bool) -> pg.PlotWidget:
        p = pg.PlotWidget()
        p.setLabel("left", ylabel)
        p.setLabel("bottom", "Step")
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setLogMode(x=False, y=log_y)
        p.setMinimumHeight(90)
        p.setMouseEnabled(x=False, y=False)
        p.setMenuEnabled(False)
        return p

    def _set(self, key: str, text: str) -> None:
        lbl = self._cards.get(key)
        if lbl is not None:
            lbl.setText(text)

    def apply_theme(self) -> None:
        dark = theme.is_dark_mode()
        fg = theme.pg_fg()
        if dark:
            card_bg, cap_col, val_col, border = "#0c1622", "#8a93a8", "#dde2ee", "#243348"
        else:
            card_bg, cap_col, val_col, border = "#eef1f6", "#5a6478", "#1a1e28", "#c4ccda"
        self.setStyleSheet(
            f"QFrame#dashCard {{ background: {card_bg};"
            f"  border: 1px solid {border}; border-radius: 6px; }}"
            f"QLabel#dashCap {{ color: {cap_col}; font-size: 10px; }}"
            f"QLabel#dashVal {{ color: {val_col}; font-size: 15px; font-weight: bold; }}"
        )
        for p in (self._plot_loss, self._plot_count):
            p.setBackground(theme.bg((2, 11, 13)))
            for ax_name in ("left", "bottom"):
                ax = p.getAxis(ax_name)
                ax.setPen(fg)
                ax.setTextPen(fg)
        # Brand-coloured curves follow the live colour pickers.
        self._curve_loss.setPen(pg.mkPen(theme.YELLOW, width=2))
        self._curve_count.setPen(pg.mkPen(theme.BLUE, width=2))
