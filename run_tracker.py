"""
run_tracker.py — Auswertungs-Panel mit Live-Loss-Kurven und Run-Verwaltung.

Features:
  - Live-Plot der Loss-Kurve via pyqtgraph
  - Dropdown zur Auswahl vergangener und aktueller Runs
  - Run-Metadaten (Methode, #Ellipsoide, Schritte, …)
  - Runs können als JSON gespeichert werden
  - Gespeicherte Runs überleben die Session
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

import theme


# ── Persistence directory ─────────────────────────────────────────────────────

_DEFAULT_RUNS_DIR = Path(__file__).parent / "saved_runs"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    """All data kept for a single optimisation run."""
    run_id: str
    name: str = ""
    mesh_name: str = ""
    method: str = "adam"
    num_ellipsoids: int = 10
    grid_n: int = 128
    started: str = ""
    finished: bool = False
    saved: bool = False

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = self.run_id.replace(":", "-").replace(" ", "_")
        fp = directory / f"{safe_name}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        self.saved = True
        return fp

    @classmethod
    def load(cls, fp: Path) -> "RunRecord":
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        rec = cls.from_dict(d)
        rec.saved = True
        return rec

    # ── helpers ───────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        label = self.name if self.name else self.run_id
        status = "" if self.finished else " [running]"
        return f"{label}{status}"

    @property
    def final_loss(self) -> float | None:
        return self.losses[-1] if self.losses else None

    @property
    def best_loss(self) -> float | None:
        return min(self.losses) if self.losses else None


# ── Colour cycle for plot lines ───────────────────────────────────────────────

def _plot_colors() -> list:
    """Plot-line colour cycle — brand secondary/primary first (read live)."""
    return [
        theme.YELLOW,        # brand colours first (see theme.py)
        theme.BLUE,
        (242, 100, 80),
        (80, 220, 140),
        (200, 120, 255),
        (255, 180, 60),
        (100, 200, 255),
        (220, 220, 100),
    ]


# ── Widget ────────────────────────────────────────────────────────────────────

class RunTrackerPanel(QtWidgets.QWidget):
    """Evaluation panel: loss convergence plot + run management."""

    def __init__(self, runs_dir: Path | str | None = None, parent=None):
        super().__init__(parent)
        self._runs_dir = Path(runs_dir) if runs_dir else _DEFAULT_RUNS_DIR

        self._runs: list[RunRecord] = []
        self._current_run: RunRecord | None = None
        self._plot_items: dict[str, pg.PlotDataItem] = {}
        self._run_colors: dict[str, tuple] = {}

        self._build_ui()
        self._load_saved_runs()

    # ══════════════════════════════════════════════════════════════════
    # PUBLIC API  (called from MainWindow)
    # ══════════════════════════════════════════════════════════════════

    def begin_run(
        self,
        mesh_name: str = "",
        method: str = "adam",
        num_ellipsoids: int = 10,
        grid_n: int = 128,
    ) -> RunRecord:
        """Start tracking a new run."""
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        rec = RunRecord(
            run_id=run_id,
            mesh_name=mesh_name,
            method=method,
            num_ellipsoids=num_ellipsoids,
            grid_n=grid_n,
            started=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._runs.append(rec)
        self._current_run = rec
        self._add_run_to_combo(rec)
        self._run_combo.setCurrentIndex(self._run_combo.count() - 1)
        self._ensure_plot_curve(rec)
        self._update_info()
        self._refresh_plot_visibility()
        return rec

    def record_step(self, step: int, loss: float) -> None:
        """Append a (step, loss) data point to the current run."""
        if self._current_run is None:
            return
        self._current_run.steps.append(step)
        self._current_run.losses.append(loss)
        self._update_plot_curve(self._current_run)

    def finish_run(self) -> None:
        """Mark the current run as finished."""
        if self._current_run is not None:
            self._current_run.finished = True
            idx = self._find_combo_index(self._current_run.run_id)
            if idx >= 0:
                self._run_combo.setItemText(idx, self._current_run.display_name)
            self._update_info()
            self._refresh_plot_visibility()

    # ══════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("Analysis")
        title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: palette(window-text);")
        layout.addWidget(title)

        # ── Run selector ──────────────────────────────────────────────
        row_sel = QtWidgets.QHBoxLayout()
        row_sel.addWidget(QtWidgets.QLabel("Run:"))
        self._run_combo = QtWidgets.QComboBox()
        self._run_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed,
        )
        self._run_combo.currentIndexChanged.connect(self._on_run_changed)
        row_sel.addWidget(self._run_combo)
        layout.addLayout(row_sel)

        # ── Plot ──────────────────────────────────────────────────────
        self._plot = pg.PlotWidget(title="Loss")
        self._plot.setLabel("bottom", "Step")
        self._plot.setLabel("left", "Loss")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend(offset=(10, 10))
        # The curve is the primary content of this tab; let it consume all
        # vertical space that is not needed by the compact controls below.
        layout.addWidget(self._plot, stretch=1)

        # ── Show-all checkbox ─────────────────────────────────────────
        self._chk_show_all = QtWidgets.QCheckBox("Show all runs")
        self._chk_show_all.setChecked(True)
        self._chk_show_all.toggled.connect(self._refresh_plot_visibility)
        layout.addWidget(self._chk_show_all)

        # ── Info area ─────────────────────────────────────────────────
        self._info_text = QtWidgets.QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(118)
        self._info_text.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum,
        )
        layout.addWidget(self._info_text)

        # Background / foreground colours follow light/dark mode.
        self.apply_theme()

        # ── Action buttons ────────────────────────────────────────────
        row_actions = QtWidgets.QHBoxLayout()

        self._btn_save = QtWidgets.QPushButton("💾 Save")
        self._btn_save.setToolTip("Save run permanently as JSON")
        self._btn_save.clicked.connect(self._on_save)
        row_actions.addWidget(self._btn_save)

        self._btn_delete = QtWidgets.QPushButton("🗑 Delete")
        self._btn_delete.setToolTip("Delete run")
        self._btn_delete.clicked.connect(self._on_delete)
        row_actions.addWidget(self._btn_delete)

        self._btn_export = QtWidgets.QPushButton("📋 Copy CSV")
        self._btn_export.setToolTip("Copy loss data to clipboard")
        self._btn_export.clicked.connect(self._on_copy_csv)
        row_actions.addWidget(self._btn_export)

        row_actions.addStretch()
        layout.addLayout(row_actions)

    def apply_theme(self):
        """Re-colour the loss plot + info box for the current light/dark mode."""
        fg = theme.pg_fg()
        self._plot.setBackground(theme.bg((2, 11, 13)))
        for ax_name in ("left", "bottom"):
            axis = self._plot.getAxis(ax_name)
            axis.setPen(fg)
            axis.setTextPen(fg)
        self._plot.setTitle("Loss", color=fg)
        legend = self._plot.plotItem.legend
        if legend is not None:
            for _sample, label in legend.items:
                try:
                    label.setText(label.text, color=fg)
                except Exception:
                    pass
        if theme.is_dark_mode():
            self._info_text.setStyleSheet(
                "background-color: #0a1520; color: #ccc; "
                "font-family: monospace; font-size: 12px;")
        else:
            self._info_text.setStyleSheet(
                "background-color: #f3f5f8; color: #222; "
                "font-family: monospace; font-size: 12px;")

    # ══════════════════════════════════════════════════════════════════
    # RUN COMBO / SELECTION
    # ══════════════════════════════════════════════════════════════════

    def _add_run_to_combo(self, rec: RunRecord):
        self._run_combo.addItem(rec.display_name, rec.run_id)

    def _find_combo_index(self, run_id: str) -> int:
        for i in range(self._run_combo.count()):
            if self._run_combo.itemData(i) == run_id:
                return i
        return -1

    def _selected_run(self) -> RunRecord | None:
        idx = self._run_combo.currentIndex()
        if idx < 0:
            return None
        run_id = self._run_combo.itemData(idx)
        for r in self._runs:
            if r.run_id == run_id:
                return r
        return None

    def _on_run_changed(self, _idx: int):
        self._update_info()
        self._refresh_plot_visibility()

    # ══════════════════════════════════════════════════════════════════
    # PLOT MANAGEMENT
    # ══════════════════════════════════════════════════════════════════

    def _get_run_color(self, rec: RunRecord) -> tuple:
        if rec.run_id not in self._run_colors:
            palette = _plot_colors()
            self._run_colors[rec.run_id] = palette[len(self._run_colors) % len(palette)]
        return self._run_colors[rec.run_id]

    def _ensure_plot_curve(self, rec: RunRecord) -> pg.PlotDataItem:
        if rec.run_id in self._plot_items:
            return self._plot_items[rec.run_id]
        color = self._get_run_color(rec)
        pen = pg.mkPen(color=color, width=2)
        label = rec.name if rec.name else rec.run_id
        curve = self._plot.plot([], [], pen=pen, name=label)
        self._plot_items[rec.run_id] = curve
        return curve

    def _update_plot_curve(self, rec: RunRecord):
        curve = self._ensure_plot_curve(rec)
        if rec.steps and rec.losses:
            curve.setData(rec.steps, rec.losses)
        self._refresh_plot_visibility()

    def _refresh_plot_visibility(self):
        show_all = self._chk_show_all.isChecked()
        selected = self._selected_run()
        selected_id = selected.run_id if selected is not None else None
        active_id = None
        if self._current_run is not None and not self._current_run.finished:
            active_id = self._current_run.run_id

        for run_id, curve in self._plot_items.items():
            # The live curve remains visible even while an older run is selected.
            visible = show_all or run_id in (selected_id, active_id)
            curve.setVisible(visible)

            color = self._run_colors.get(run_id, theme.YELLOW)
            if run_id == active_id:
                # Current optimisation: saturated, thicker and drawn on top.
                opacity, width, z_value = 1.0, 3.0, 20.0
            elif active_id is not None and run_id == selected_id:
                # An explicitly inspected historic run remains readable without
                # competing visually with the live result.
                opacity, width, z_value = 0.60, 2.0, 10.0
            elif active_id is None and run_id == selected_id:
                opacity, width, z_value = 1.0, 2.5, 20.0
            else:
                opacity, width, z_value = 0.26, 1.25, 0.0
            curve.setOpacity(opacity)
            curve.setPen(pg.mkPen(color=color, width=width))
            curve.setZValue(z_value)

    # ══════════════════════════════════════════════════════════════════
    # INFO PANEL
    # ══════════════════════════════════════════════════════════════════

    def _update_info(self):
        rec = self._selected_run()
        if rec is None:
            self._info_text.setPlainText("No run selected.")
            return

        status = "Completed" if rec.finished else "Running …"
        lines = [
            f"Status:         {status}",
            f"Mesh:           {rec.mesh_name or '—'}",
            f"Method:         {rec.method}",
            f"Ellipsoids:     {rec.num_ellipsoids}",
            f"Grid N:         {rec.grid_n}",
            f"Started:        {rec.started}",
            f"Steps:          {len(rec.steps)}",
        ]
        if rec.final_loss is not None:
            lines.append(f"Last loss:      {rec.final_loss:.6f}")
        if rec.best_loss is not None:
            lines.append(f"Best loss:      {rec.best_loss:.6f}")
        if rec.saved:
            lines.append("Saved:          ✓")

        self._info_text.setPlainText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    # ACTIONS: save, delete, export
    # ══════════════════════════════════════════════════════════════════

    def _on_save(self):
        rec = self._selected_run()
        if rec is None:
            return
        try:
            fp = rec.save(self._runs_dir)
            self._update_info()
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Run saved:\n{fp}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Save failed:\n{e}")

    def _on_delete(self):
        rec = self._selected_run()
        if rec is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Delete",
            f"Really delete run '{rec.display_name}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        if rec.saved:
            safe_name = rec.run_id.replace(":", "-").replace(" ", "_")
            fp = self._runs_dir / f"{safe_name}.json"
            if fp.exists():
                fp.unlink()

        if rec.run_id in self._plot_items:
            self._plot.removeItem(self._plot_items.pop(rec.run_id))
        self._run_colors.pop(rec.run_id, None)

        self._runs = [r for r in self._runs if r.run_id != rec.run_id]
        idx = self._find_combo_index(rec.run_id)
        if idx >= 0:
            self._run_combo.removeItem(idx)

        if rec is self._current_run:
            self._current_run = None
        self._update_info()
        self._refresh_plot_visibility()

    def _on_copy_csv(self):
        rec = self._selected_run()
        if rec is None or not rec.steps:
            return
        lines = ["step,loss"]
        for s, l in zip(rec.steps, rec.losses):
            lines.append(f"{s},{l:.8f}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    # PERSISTENCE — load saved runs on startup
    # ══════════════════════════════════════════════════════════════════

    def _load_saved_runs(self):
        if not self._runs_dir.is_dir():
            return
        for fp in sorted(self._runs_dir.glob("*.json")):
            try:
                rec = RunRecord.load(fp)
                self._runs.append(rec)
                self._add_run_to_combo(rec)
                curve = self._ensure_plot_curve(rec)
                if rec.steps and rec.losses:
                    curve.setData(rec.steps, rec.losses)
                    curve.setVisible(False)
            except Exception as e:
                print(f"[RunTracker] Could not load {fp.name}: {e}")
        self._refresh_plot_visibility()
