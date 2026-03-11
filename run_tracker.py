"""
run_tracker.py — Auswertungs-Panel mit Multi-Metrik-Plots und Run-Verwaltung.

Features:
  - Umschaltbarer Plot: Loss, Best Loss, Ellipsoid-Anzahl, Iterationszeit,
    Gesamtzeit, mittlerer/min/max Radius, Gesamtvolumen
  - Dropdown zur Auswahl vergangener und aktueller Runs
  - "Alle Runs" Overlay-Checkbox
  - Run-Metadaten im Info-Bereich
  - Runs benennen, als JSON speichern, löschen, CSV kopieren
  - Persistenz über Sessions hinweg
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg


# ── Persistence directory ─────────────────────────────────────────────────────

_DEFAULT_RUNS_DIR = Path(__file__).parent / "saved_runs"


# ── Metric definitions ────────────────────────────────────────────────────────
# Each metric is a (key_in_RunRecord.series, display_name, y_label, format_str)

METRICS = [
    ("loss",            "Loss",               "Loss",           "{:.6f}"),
    ("best_loss",       "Best Loss (bisher)", "Best Loss",      "{:.6f}"),
    ("num_ellipsoids",  "Ellipsoid-Anzahl",   "Anzahl",         "{:.0f}"),
    ("iter_ms",         "Iterationszeit",     "ms / Iteration", "{:.2f}"),
    ("elapsed_s",       "Gesamtzeit",         "Sekunden",       "{:.1f}"),
    ("mean_radius",     "Mittlerer Radius",   "Radius",         "{:.4f}"),
    ("min_radius",      "Min. Radius",        "Radius",         "{:.4f}"),
    ("max_radius",      "Max. Radius",        "Radius",         "{:.4f}"),
    ("total_volume",    "Gesamtvolumen",      "Volumen",        "{:.4f}"),
]

# Keys from the metrics dict emitted by OptimizationWorker.step_metrics
_DIRECT_METRIC_KEYS = {
    "loss", "num_ellipsoids", "iter_ms", "elapsed_s",
    "mean_radius", "min_radius", "max_radius", "total_volume",
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    """All data kept for a single optimisation run."""
    run_id: str
    name: str = ""
    mesh_name: str = ""
    method: str = "adam"
    num_ellipsoids_init: int = 10
    grid_n: int = 128
    batch_size: int = 0
    started: str = ""
    finished: bool = False
    saved: bool = False

    # Per-step time series — keys match METRICS above
    steps: list[int] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=lambda: {
        k: [] for k, _, _, _ in METRICS
    })

    # Maintenance log
    maintenance_log: list[dict] = field(default_factory=list)

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        # Handle legacy files that don't have `series`
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in d.items() if k in known}
        rec = cls(**filtered)
        # Ensure all metric keys exist
        for key, _, _, _ in METRICS:
            if key not in rec.series:
                rec.series[key] = []
        return rec

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
        status = "" if self.finished else " [laufend]"
        return f"{label}{status}"

    @property
    def final_loss(self) -> float | None:
        losses = self.series.get("loss", [])
        return losses[-1] if losses else None

    @property
    def best_loss(self) -> float | None:
        losses = self.series.get("loss", [])
        return min(losses) if losses else None


# ── Colour cycle for plot lines ───────────────────────────────────────────────

_PLOT_COLORS = [
    (242, 230, 65),
    (73, 98, 242),
    (242, 100, 80),
    (80, 220, 140),
    (200, 120, 255),
    (255, 180, 60),
    (100, 200, 255),
    (220, 220, 100),
]


# ── Widget ────────────────────────────────────────────────────────────────────

class RunTrackerPanel(QtWidgets.QWidget):
    """Evaluation panel: multi-metric convergence plots + run management."""

    def __init__(self, runs_dir: Path | str | None = None, parent=None):
        super().__init__(parent)
        self._runs_dir = Path(runs_dir) if runs_dir else _DEFAULT_RUNS_DIR

        self._runs: list[RunRecord] = []
        self._current_run: RunRecord | None = None

        # plot_items[run_id] = PlotDataItem  (one curve per run for current metric)
        self._plot_items: dict[str, pg.PlotDataItem] = {}
        # Remember which color each run got
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
        batch_size: int = 0,
    ) -> RunRecord:
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        rec = RunRecord(
            run_id=run_id,
            mesh_name=mesh_name,
            method=method,
            num_ellipsoids_init=num_ellipsoids,
            grid_n=grid_n,
            batch_size=batch_size,
            started=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._runs.append(rec)
        self._current_run = rec
        self._add_run_to_combo(rec)
        self._run_combo.setCurrentIndex(self._run_combo.count() - 1)
        self._ensure_plot_curve(rec)
        self._update_info()
        return rec

    def record_step(self, metrics: dict) -> None:
        """Append a full metrics snapshot.  Called from MainWindow on
        each step_metrics signal from the optimizer."""
        if self._current_run is None:
            return

        rec = self._current_run
        step = metrics.get("step", len(rec.steps))
        rec.steps.append(step)

        # Record all direct metrics
        for key in _DIRECT_METRIC_KEYS:
            if key in metrics and key in rec.series:
                rec.series[key].append(float(metrics[key]))

        # Derived: running best loss
        losses = rec.series.get("loss", [])
        if losses:
            prev_best = rec.series["best_loss"][-1] if rec.series["best_loss"] else losses[-1]
            rec.series["best_loss"].append(min(prev_best, losses[-1]))

        self._update_current_curve()

    def record_maintenance(self, step: int, n_before: int,
                           n_pruned: int, n_spawned: int) -> None:
        if self._current_run is None:
            return
        self._current_run.maintenance_log.append({
            "step": step, "before": n_before,
            "pruned": n_pruned, "spawned": n_spawned,
        })
        self._update_info()

    def finish_run(self) -> None:
        if self._current_run is not None:
            self._current_run.finished = True
            idx = self._find_combo_index(self._current_run.run_id)
            if idx >= 0:
                self._run_combo.setItemText(idx, self._current_run.display_name)
            self._update_info()

    # ══════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("Auswertung")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # ── Run selector row ──────────────────────────────────────────
        row_sel = QtWidgets.QHBoxLayout()
        row_sel.addWidget(QtWidgets.QLabel("Run:"))
        self._run_combo = QtWidgets.QComboBox()
        self._run_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Fixed)
        self._run_combo.currentIndexChanged.connect(self._on_run_changed)
        row_sel.addWidget(self._run_combo)
        layout.addLayout(row_sel)

        # ── Metric selector row ───────────────────────────────────────
        row_metric = QtWidgets.QHBoxLayout()
        row_metric.addWidget(QtWidgets.QLabel("Metrik:"))
        self._metric_combo = QtWidgets.QComboBox()
        for key, display_name, _, _ in METRICS:
            self._metric_combo.addItem(display_name, key)
        self._metric_combo.setCurrentIndex(0)
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        row_metric.addWidget(self._metric_combo)
        layout.addLayout(row_metric)

        # ── Plot ──────────────────────────────────────────────────────
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Schritt")
        self._plot.setLabel("left", "Loss")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setBackground((2, 11, 13))
        self._plot.addLegend(offset=(10, 10))
        layout.addWidget(self._plot, stretch=3)

        # ── Show-all checkbox ─────────────────────────────────────────
        self._chk_show_all = QtWidgets.QCheckBox("Alle Runs anzeigen")
        self._chk_show_all.setChecked(False)
        self._chk_show_all.toggled.connect(self._refresh_plot_visibility)
        layout.addWidget(self._chk_show_all)

        # ── Info area ─────────────────────────────────────────────────
        self._info_text = QtWidgets.QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(170)
        self._info_text.setStyleSheet(
            "background-color: #0a1520; color: #ccc; "
            "font-family: monospace; font-size: 12px;"
        )
        layout.addWidget(self._info_text, stretch=1)

        # ── Name + save row ───────────────────────────────────────────
        row_name = QtWidgets.QHBoxLayout()
        row_name.addWidget(QtWidgets.QLabel("Name:"))
        self._edit_name = QtWidgets.QLineEdit()
        self._edit_name.setPlaceholderText("Run benennen …")
        self._edit_name.returnPressed.connect(self._on_rename)
        row_name.addWidget(self._edit_name)
        self._btn_rename = QtWidgets.QPushButton("Umbenennen")
        self._btn_rename.clicked.connect(self._on_rename)
        row_name.addWidget(self._btn_rename)
        layout.addLayout(row_name)

        # ── Action buttons ────────────────────────────────────────────
        row_actions = QtWidgets.QHBoxLayout()
        self._btn_save = QtWidgets.QPushButton("💾 Speichern")
        self._btn_save.setToolTip("Run dauerhaft als JSON speichern")
        self._btn_save.clicked.connect(self._on_save)
        row_actions.addWidget(self._btn_save)

        self._btn_delete = QtWidgets.QPushButton("🗑 Löschen")
        self._btn_delete.setToolTip("Gespeicherten Run löschen")
        self._btn_delete.clicked.connect(self._on_delete)
        row_actions.addWidget(self._btn_delete)

        self._btn_export = QtWidgets.QPushButton("📋 CSV kopieren")
        self._btn_export.setToolTip("Alle Metrik-Daten in die Zwischenablage kopieren")
        self._btn_export.clicked.connect(self._on_copy_csv)
        row_actions.addWidget(self._btn_export)

        row_actions.addStretch()
        layout.addLayout(row_actions)

    # ══════════════════════════════════════════════════════════════════
    # METRIC SELECTION
    # ══════════════════════════════════════════════════════════════════

    @property
    def _current_metric_key(self) -> str:
        return self._metric_combo.currentData() or "loss"

    def _metric_y_label(self, key: str) -> str:
        for k, _, y_label, _ in METRICS:
            if k == key:
                return y_label
        return ""

    def _metric_fmt(self, key: str) -> str:
        for k, _, _, fmt in METRICS:
            if k == key:
                return fmt
        return "{:.4f}"

    def _on_metric_changed(self, _idx: int):
        """Rebuild all plot curves for the newly selected metric."""
        self._rebuild_all_curves()

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
            idx = len(self._run_colors) % len(_PLOT_COLORS)
            self._run_colors[rec.run_id] = _PLOT_COLORS[idx]
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

    def _update_current_curve(self):
        """Update the plot curve for the current run with the active metric."""
        if self._current_run is None:
            return
        self._set_curve_data(self._current_run)
        self._refresh_plot_visibility()

    def _set_curve_data(self, rec: RunRecord):
        """Set curve data for `rec` using the currently selected metric."""
        curve = self._ensure_plot_curve(rec)
        key = self._current_metric_key
        data = rec.series.get(key, [])
        steps = rec.steps[:len(data)]
        if steps and data:
            curve.setData(steps, data)
        else:
            curve.setData([], [])

    def _rebuild_all_curves(self):
        """Clear and rebuild all curves for the selected metric."""
        # Remove all existing items
        for curve in self._plot_items.values():
            self._plot.removeItem(curve)
        self._plot_items.clear()

        # Update y-axis label
        key = self._current_metric_key
        self._plot.setLabel("left", self._metric_y_label(key))

        # Recreate curves for all runs
        for rec in self._runs:
            self._ensure_plot_curve(rec)
            self._set_curve_data(rec)

        self._refresh_plot_visibility()

    def _refresh_plot_visibility(self):
        show_all = self._chk_show_all.isChecked()
        selected = self._selected_run()
        for run_id, curve in self._plot_items.items():
            if show_all:
                curve.setVisible(True)
            elif selected and run_id == selected.run_id:
                curve.setVisible(True)
            else:
                curve.setVisible(False)

    # ══════════════════════════════════════════════════════════════════
    # INFO PANEL
    # ══════════════════════════════════════════════════════════════════

    def _update_info(self):
        rec = self._selected_run()
        if rec is None:
            self._info_text.setPlainText("Kein Run ausgewählt.")
            self._edit_name.clear()
            return

        self._edit_name.setText(rec.name)

        status = "Abgeschlossen" if rec.finished else "Laufend …"
        n_steps = len(rec.steps)

        lines = [
            f"ID:             {rec.run_id}",
            f"Name:           {rec.name or '(unbenannt)'}",
            f"Status:         {status}",
            f"Mesh:           {rec.mesh_name or '—'}",
            f"Methode:        {rec.method}",
            f"Ellipsoide (i): {rec.num_ellipsoids_init}",
            f"Grid N:         {rec.grid_n}",
            f"Batch Size:     {rec.batch_size}",
            f"Gestartet:      {rec.started}",
            f"Schritte:       {n_steps}",
        ]

        # Latest values of key metrics
        def _last(key):
            v = rec.series.get(key, [])
            return v[-1] if v else None

        loss = _last("loss")
        if loss is not None:
            lines.append(f"Letzter Loss:   {loss:.6f}")
        bl = rec.best_loss
        if bl is not None:
            lines.append(f"Bester Loss:    {bl:.6f}")

        n_ell = _last("num_ellipsoids")
        if n_ell is not None:
            lines.append(f"Ellipsoide:     {int(n_ell)}")

        elapsed = _last("elapsed_s")
        if elapsed is not None:
            lines.append(f"Laufzeit:       {elapsed:.1f} s")

        iter_ms = _last("iter_ms")
        if iter_ms is not None:
            lines.append(f"Iter.-Zeit:     {iter_ms:.2f} ms/step")

        mean_r = _last("mean_radius")
        if mean_r is not None:
            lines.append(f"Mittl. Radius:  {mean_r:.4f}")

        total_v = _last("total_volume")
        if total_v is not None:
            lines.append(f"Gesamtvolumen:  {total_v:.4f}")

        if rec.maintenance_log:
            last_m = rec.maintenance_log[-1]
            lines.append(
                f"Letzte Wartung: Step {last_m['step']} "
                f"(pruned={last_m['pruned']}, spawned={last_m['spawned']})"
            )
            lines.append(f"Wartungen ges.: {len(rec.maintenance_log)}")

        if rec.saved:
            lines.append("Gespeichert:    ✓")

        self._info_text.setPlainText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    # ACTIONS: rename, save, delete, export
    # ══════════════════════════════════════════════════════════════════

    def _on_rename(self):
        rec = self._selected_run()
        if rec is None:
            return
        new_name = self._edit_name.text().strip()
        if not new_name:
            return
        rec.name = new_name
        idx = self._find_combo_index(rec.run_id)
        if idx >= 0:
            self._run_combo.setItemText(idx, rec.display_name)
        # Update legend label
        if rec.run_id in self._plot_items:
            self._plot_items[rec.run_id].opts["name"] = new_name
            self._plot.plotItem.legend.clear()
            for rid, curve in self._plot_items.items():
                r = next((x for x in self._runs if x.run_id == rid), None)
                lbl = r.name if (r and r.name) else rid
                self._plot.plotItem.legend.addItem(curve, lbl)
        self._update_info()

    def _on_save(self):
        rec = self._selected_run()
        if rec is None:
            return
        try:
            fp = rec.save(self._runs_dir)
            self._update_info()
            QtWidgets.QMessageBox.information(
                self, "Gespeichert", f"Run gespeichert:\n{fp}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Fehler", f"Speichern fehlgeschlagen:\n{e}")

    def _on_delete(self):
        rec = self._selected_run()
        if rec is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Löschen",
            f"Run '{rec.display_name}' wirklich löschen?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
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

    def _on_copy_csv(self):
        """Copy ALL metric series for the selected run as CSV."""
        rec = self._selected_run()
        if rec is None or not rec.steps:
            return
        # Build header from available series with data
        cols = ["step"]
        for key, name, _, _ in METRICS:
            if rec.series.get(key):
                cols.append(key)

        lines = [",".join(cols)]
        for i, s in enumerate(rec.steps):
            row = [str(s)]
            for key, _, _, fmt in METRICS:
                data = rec.series.get(key, [])
                if data:
                    val = data[i] if i < len(data) else ""
                    row.append(fmt.format(val) if val != "" else "")
            lines.append(",".join(row))

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
                self._ensure_plot_curve(rec)
                self._set_curve_data(rec)
                # Hidden by default
                self._plot_items[rec.run_id].setVisible(False)
            except Exception as e:
                print(f"[RunTracker] Konnte {fp.name} nicht laden: {e}")