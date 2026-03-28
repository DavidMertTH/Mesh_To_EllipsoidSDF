"""
benchmark_sdf.py — Benchmark-Tool für Ellipsoid-SDF-Approximationen.

Eigenständig ausführbar:
    python benchmark_sdf.py

Vergleicht verschiedene SDF-Approximationen gegen einen exakten
Ground-Truth (Bisection auf dem Lagrange-Multiplikator) für
konfigurierbare Ellipsoid-Achsen.

Methoden:
  - Exakt (Bisection)         →  Ground Truth
  - Quílez                    →  k0·(k0−1) / k1
  - Scaled-Sphere (min r)     →  (|p/r|−1) · min(r)
  - Scaled-Sphere (mean r)    →  (|p/r|−1) · mean(r)
  - Scaled-Sphere (geom mean) →  (|p/r|−1) · ∛(r₁r₂r₃)
  - MertStein                 →  Quílez (außen) + Sc-min (innen)
"""

import sys
import time

import numpy as np

from PySide6 import QtCore, QtWidgets, QtGui

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


# ══════════════════════════════════════════════════════════════════════════════
# SDF METHODS — imported from central registry (sdf_methods.py)
# ══════════════════════════════════════════════════════════════════════════════

from sdf_methods import (
    sdf_exact_np as sdf_exact,
    sdf_quilez_np as sdf_quilez,
    sdf_scaled_min_np as sdf_scaled_min,
    sdf_scaled_mean_np as sdf_scaled_mean,
    sdf_scaled_gmean_np as sdf_scaled_gmean,
    sdf_mertstein_np as sdf_mertstein,
    SDF_METHODS as _SDF_METHODS_INFO,
)

# Method registry for the benchmark: (display_name, function, short_name)
METHODS = [
    (m.name, m.numpy_fn, m.short) for m in _SDF_METHODS_INFO
]


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _make_slice_grid(axis_a, axis_b, extent, grid_n):
    """Create (grid_n²,3) points on a 2-D slice through origin."""
    t = np.linspace(-extent, extent, grid_n)
    aa, bb = np.meshgrid(t, t)
    points = np.zeros((grid_n * grid_n, 3), dtype=np.float64)
    points[:, axis_a] = aa.ravel()
    points[:, axis_b] = bb.ravel()
    return t, points


def _compute_metrics(error, gt, radii):
    """Compute error metrics, overall and per-region."""
    ae = np.abs(error)
    mean_r = np.mean(radii)
    near_mask = np.abs(gt) < 0.15 * mean_r
    int_mask  = gt < 0
    ext_mask  = gt > 0

    def _stats(e):
        if len(e) == 0:
            return dict(mae=0.0, rmse=0.0, l_inf=0.0)
        return dict(
            mae=float(np.mean(np.abs(e))),
            rmse=float(np.sqrt(np.mean(e ** 2))),
            l_inf=float(np.max(np.abs(e))),
        )

    return dict(
        total=_stats(error),
        interior=_stats(error[int_mask]),
        exterior=_stats(error[ext_mask]),
        near_surface=_stats(error[near_mask]),
    )


def run_benchmark(radii_tuple, grid_n=256, timing_repeats=5):
    """Run the full benchmark for a given set of radii.

    Returns a dict with everything needed for plotting.
    """
    radii = np.array(radii_tuple, dtype=np.float64)
    extent = np.max(radii) * 2.0

    # ── 2-D slices ────────────────────────────────────────────────────
    slices = {}
    for label, ax_a, ax_b in [("XY", 0, 1), ("XZ", 0, 2)]:
        coords, points = _make_slice_grid(ax_a, ax_b, extent, grid_n)

        gt = sdf_exact(points, radii)

        methods_data = {}
        for name, func, short in METHODS:
            sdf = func(points, radii)
            error = sdf - gt
            metrics = _compute_metrics(error, gt, radii)
            methods_data[name] = dict(
                sdf=sdf.reshape(grid_n, grid_n),
                error=error.reshape(grid_n, grid_n),
                metrics=metrics,
                short=short,
            )

        slices[label] = dict(
            coords=coords,
            gt=gt.reshape(grid_n, grid_n),
            methods=methods_data,
        )

    # ── Radial error profile (rays from center, XY plane) ──────────────
    n_radial = 500
    # Normalized distance: 0 = center, 1 = surface, 2 = far outside
    t_norm = np.linspace(0.01, 2.0, n_radial)

    # Rays along principal axes + diagonals in XY
    ray_dirs = {
        "x-Achse":   np.array([1.0, 0.0, 0.0]),
        "y-Achse":   np.array([0.0, 1.0, 0.0]),
        "45° (xy)":  np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    }

    radial_data = dict(t_norm=t_norm, rays={})
    for ray_label, direction in ray_dirs.items():
        # Scale direction so that t_norm=1 hits the ellipsoid surface
        # Surface point: p_i = d_i * r_i  such that |p/r| = 1
        # For direction d: surface at t where |(t*d)/r| = 1
        #   t_surface = 1 / |d/r|
        d_over_r = direction / radii
        t_surface = 1.0 / np.linalg.norm(d_over_r)

        points = np.outer(t_norm * t_surface, direction)  # (n_radial, 3)
        gt_rad = sdf_exact(points, radii)

        ray_result = dict(gt=gt_rad)
        for name, func, short in METHODS:
            ray_result[name] = func(points, radii) - gt_rad
        radial_data["rays"][ray_label] = ray_result

    # ── Timing (3-D grid, smaller) ────────────────────────────────────
    tn = 48
    tc = np.linspace(-extent, extent, tn)
    xx, yy, zz = np.meshgrid(tc, tc, tc, indexing="ij")
    pts3d = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    timing = {}
    # Ground truth
    times = []
    for _ in range(timing_repeats):
        t0 = time.perf_counter()
        sdf_exact(pts3d, radii)
        times.append(time.perf_counter() - t0)
    timing["Ground Truth"] = float(np.median(times)) * 1000.0

    for name, func, short in METHODS:
        times = []
        for _ in range(timing_repeats):
            t0 = time.perf_counter()
            func(pts3d, radii)
            times.append(time.perf_counter() - t0)
        timing[name] = float(np.median(times)) * 1000.0

    return dict(
        radii=radii,
        extent=extent,
        grid_n=grid_n,
        slices=slices,
        radial=radial_data,
        timing=timing,
        aspect_ratio=float(np.max(radii) / np.min(radii)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Qt GUI
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ellipsoid SDF Benchmark")
        self.resize(1700, 1100)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Controls ──────────────────────────────────────────────────
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(10)

        bar.addWidget(QtWidgets.QLabel("rₓ:"))
        self._spin_rx = self._make_spin(1.0, 0.05, 20.0)
        bar.addWidget(self._spin_rx)

        bar.addWidget(QtWidgets.QLabel("rᵧ:"))
        self._spin_ry = self._make_spin(0.5, 0.05, 20.0)
        bar.addWidget(self._spin_ry)

        bar.addWidget(QtWidgets.QLabel("r_z:"))
        self._spin_rz = self._make_spin(0.3, 0.05, 20.0)
        bar.addWidget(self._spin_rz)

        bar.addSpacing(20)
        bar.addWidget(QtWidgets.QLabel("Grid N:"))
        self._spin_n = QtWidgets.QSpinBox()
        self._spin_n.setRange(64, 512)
        self._spin_n.setValue(256)
        self._spin_n.setSingleStep(64)
        bar.addWidget(self._spin_n)

        bar.addSpacing(20)
        self._btn_run = QtWidgets.QPushButton("▶  Berechnen")
        self._btn_run.setFixedHeight(32)
        self._btn_run.clicked.connect(self._on_run)
        bar.addWidget(self._btn_run)

        self._chk_interior = QtWidgets.QCheckBox("Nur Interior (SDF < 0)")
        self._chk_interior.setChecked(False)
        self._chk_interior.setToolTip("Fehler nur für Punkte innerhalb des Ellipsoids berechnen")
        self._chk_interior.toggled.connect(self._on_interior_toggled)
        bar.addWidget(self._chk_interior)

        self._btn_pdf = QtWidgets.QPushButton("📄 PDF exportieren")
        self._btn_pdf.setFixedHeight(32)
        self._btn_pdf.setEnabled(False)
        self._btn_pdf.clicked.connect(self._on_export_pdf)
        bar.addWidget(self._btn_pdf)

        bar.addStretch()

        self._lbl_status = QtWidgets.QLabel("")
        self._lbl_status.setStyleSheet("color: #888; font-style: italic;")
        bar.addWidget(self._lbl_status)

        root.addLayout(bar)

        # ── Matplotlib figure ─────────────────────────────────────────
        self._fig = Figure(figsize=(18, 12), dpi=100, facecolor="#0d1117")
        self._canvas = FigureCanvas(self._fig)
        root.addWidget(self._canvas, stretch=4)

        # ── Metrics table ─────────────────────────────────────────────
        self._table = QtWidgets.QTableWidget()
        self._table.setMaximumHeight(180)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { font-family: monospace; font-size: 12px; }"
        )
        root.addWidget(self._table, stretch=1)

        self.setCentralWidget(central)

    @staticmethod
    def _make_spin(val, lo, hi):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(0.05)
        s.setDecimals(3)
        return s

    # ── Run benchmark ─────────────────────────────────────────────────

    def _on_run(self):
        rx = self._spin_rx.value()
        ry = self._spin_ry.value()
        rz = self._spin_rz.value()
        n = self._spin_n.value()

        self._lbl_status.setText("Berechne …")
        self._btn_run.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        t0 = time.perf_counter()
        results = run_benchmark((rx, ry, rz), grid_n=n)
        elapsed = time.perf_counter() - t0

        self._last_results = results
        self._update_figure(results)
        self._update_table(results)
        self._btn_pdf.setEnabled(True)

        ar = results["aspect_ratio"]
        self._lbl_status.setText(
            f"Fertig — κ = {ar:.2f}  |  {elapsed:.1f} s  |  Grid {n}×{n}"
        )
        self._btn_run.setEnabled(True)

    # ── PDF export ────────────────────────────────────────────────────

    def _on_export_pdf(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "PDF speichern", "benchmark_sdf.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if path:
            self._fig.savefig(path, bbox_inches="tight", facecolor=self._fig.get_facecolor())
            self._lbl_status.setText(f"Exportiert: {path}")

    def _on_interior_toggled(self, checked):
        if hasattr(self, "_last_results"):
            self._update_figure(self._last_results)
            self._update_table(self._last_results)

    # ── Update figure ─────────────────────────────────────────────────

    def _update_figure(self, res):
        self._fig.clear()

        n_methods = len(METHODS)
        n_cols = 1 + n_methods  # GT + methods
        interior_only = self._chk_interior.isChecked()

        # Layout: 3 rows (XY only)
        #   Row 0: SDF maps   (GT + each method)
        #   Row 1: Error maps (label + each method)
        #   Row 2: bar charts / angular / timing
        gs = gridspec.GridSpec(
            3, n_cols, figure=self._fig,
            height_ratios=[1, 1, 0.85],
            hspace=0.40, wspace=0.25,
        )

        text_color = "#d0d0d0"
        radii = res["radii"]
        extent = res["extent"]
        ar = res["aspect_ratio"]

        # ── Shared SDF color limits ───────────────────────────────────
        sdf_vmin = -extent * 0.3
        sdf_vmax =  extent * 0.3

        sl = res["slices"]["XY"]
        coords = sl["coords"]
        gt = sl["gt"]
        ext_kw = dict(extent=[-extent, extent, -extent, extent],
                      origin="lower", aspect="equal")

        # ── Interior mask (GT < 0) ────────────────────────────────────
        interior_mask = gt < 0.0  # True where inside

        # ── Prepare masked error arrays + per-method MAE ──────────────
        error_maps = {}
        mae_vals = {}
        max_err = 0.0
        for name, _, _ in METHODS:
            err = sl["methods"][name]["error"].copy()
            if interior_only:
                err[~interior_mask] = np.nan
            error_maps[name] = err
            abs_err = np.abs(err)
            valid = abs_err[~np.isnan(abs_err)]
            mae_vals[name] = float(np.mean(valid)) if len(valid) > 0 else 0.0
            if len(valid) > 0:
                max_err = max(max_err, float(np.max(valid)))
        if max_err < 1e-12:
            max_err = 1e-6

        # ── Row 0: SDF — Ground truth in col 0 ───────────────────────
        ax_gt = self._fig.add_subplot(gs[0, 0])
        ax_gt.imshow(gt, cmap="RdYlBu_r",
                     vmin=sdf_vmin, vmax=sdf_vmax, **ext_kw)
        ax_gt.contour(coords, coords, gt, levels=[0],
                      colors="black", linewidths=0.8)
        ax_gt.set_title("Ground Truth\nXY-Schnitt (z = 0)",
                        fontsize=9, color=text_color)
        ax_gt.set_xlabel("x", fontsize=8, color=text_color)
        ax_gt.set_ylabel("y", fontsize=8, color=text_color)
        ax_gt.tick_params(colors=text_color, labelsize=7)
        ax_gt.set_facecolor("#0d1117")

        # ── Row 0: SDF — each method ─────────────────────────────────
        for col_idx, (name, _, short) in enumerate(METHODS):
            md = sl["methods"][name]
            ax = self._fig.add_subplot(gs[0, col_idx + 1])
            ax.imshow(md["sdf"], cmap="RdYlBu_r",
                      vmin=sdf_vmin, vmax=sdf_vmax, **ext_kw)
            ax.contour(coords, coords, md["sdf"], levels=[0],
                       colors="black", linewidths=0.6)
            ax.set_title(f"{name} — SDF",
                         fontsize=9, color=text_color)
            ax.set_xlabel("x", fontsize=8, color=text_color)
            ax.tick_params(colors=text_color, labelsize=7)
            ax.set_facecolor("#0d1117")

        # ── Row 1: Error — label in col 0 ────────────────────────────
        ax_empty = self._fig.add_subplot(gs[1, 0])
        ax_empty.set_facecolor("#0d1117")
        label_extra = "\n(nur Interior)" if interior_only else ""
        ax_empty.text(0.5, 0.5, f"|Fehler|\nXY-Schnitt (z = 0){label_extra}",
                      ha="center", va="center", fontsize=10,
                      color=text_color, transform=ax_empty.transAxes)
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])
        for spine in ax_empty.spines.values():
            spine.set_color("#333")

        # ── Row 1: Error — each method heatmap (red, absolute) ───────
        for col_idx, (name, _, short) in enumerate(METHODS):
            ax = self._fig.add_subplot(gs[1, col_idx + 1])
            abs_err = np.abs(error_maps[name])
            ax.imshow(abs_err, cmap="Reds", vmin=0, vmax=max_err, **ext_kw)
            ax.contour(coords, coords, gt, levels=[0],
                       colors="black", linewidths=0.6)
            mae = mae_vals[name]
            ax.set_title(f"{name} — |Fehler|\nMAE = {mae:.4f}",
                         fontsize=9, color=text_color)
            ax.set_xlabel("x", fontsize=8, color=text_color)
            ax.tick_params(colors=text_color, labelsize=7)
            ax.set_facecolor("#0d1117")

        # ── Row 2: bar charts + angular profile + timing ──────────────

        ax_bar = self._fig.add_subplot(gs[2, 0:2])
        self._draw_metric_bars(ax_bar, res, text_color)

        ax_rad = self._fig.add_subplot(gs[2, 2:4])
        self._draw_radial_profile(ax_rad, res, text_color)

        if n_cols > 4:
            ax_time = self._fig.add_subplot(gs[2, 4:n_cols])
        else:
            ax_time = self._fig.add_subplot(gs[2, n_cols - 1])
        self._draw_timing_bars(ax_time, res, text_color)

        # ── Suptitle ──────────────────────────────────────────────────
        self._fig.suptitle(
            f"Ellipsoid SDF Benchmark   —   r = ({radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f})"
            f"   κ = {ar:.2f}",
            fontsize=13, color=text_color, y=0.99,
        )

        self._canvas.draw()

    # ── Sub-plots ─────────────────────────────────────────────────────

    def _draw_metric_bars(self, ax, res, tc):
        """Grouped bar chart: MAE, RMSE, L∞ for XY slice."""
        interior_only = self._chk_interior.isChecked()
        region = "interior" if interior_only else "total"
        region_label = "Interior" if interior_only else "gesamt"

        sl = res["slices"]["XY"]
        names, mae, rmse, linf = [], [], [], []
        for name, _, short in METHODS:
            m = sl["methods"][name]["metrics"][region]
            names.append(short)
            mae.append(m["mae"])
            rmse.append(m["rmse"])
            linf.append(m["l_inf"])

        x = np.arange(len(names))
        w = 0.25
        colors = ["#f2e641", "#4962f2", "#f26450"]
        ax.bar(x - w, mae, w, label="MAE", color=colors[0], edgecolor="none")
        ax.bar(x, rmse, w, label="RMSE", color=colors[1], edgecolor="none")
        ax.bar(x + w, linf, w, label="L∞", color=colors[2], edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, color=tc)
        ax.set_ylabel("Fehler", fontsize=9, color=tc)
        ax.set_title(f"Fehlermetriken (XY, {region_label})", fontsize=9, color=tc)
        ax.legend(fontsize=8, loc="upper left", facecolor="#1a2030",
                  edgecolor="#333", labelcolor=tc)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#333")

    def _draw_radial_profile(self, ax, res, tc):
        """Error vs. normalized radial distance along rays from center."""
        rad = res["radial"]
        t_norm = rad["t_norm"]
        method_colors = ["#f2e641", "#4962f2", "#50c878", "#c878ff", "#ff7f50"]
        line_styles = ["-", "--", ":"]

        for i, (name, _, short) in enumerate(METHODS):
            for j, (ray_label, ray_data) in enumerate(rad["rays"].items()):
                label = f"{short} ({ray_label})" if j == 0 or i == 0 else None
                # Only label method on first ray, ray on first method
                if i == 0 and j > 0:
                    label = ray_label
                elif j == 0:
                    label = short
                else:
                    label = None
                ax.plot(t_norm, np.abs(ray_data[name]),
                        color=method_colors[i % len(method_colors)],
                        linestyle=line_styles[j % len(line_styles)],
                        linewidth=1.0, alpha=0.85)

        # Manual legend: methods as colored lines, rays as line styles
        from matplotlib.lines import Line2D
        handles = []
        for i, (name, _, short) in enumerate(METHODS):
            handles.append(Line2D([0], [0], color=method_colors[i % len(method_colors)],
                                  linewidth=1.5, label=short))
        for j, ray_label in enumerate(rad["rays"].keys()):
            handles.append(Line2D([0], [0], color="#aaa",
                                  linestyle=line_styles[j % len(line_styles)],
                                  linewidth=1.2, label=ray_label))

        ax.axvline(1.0, color="#666", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate("Oberfläche", xy=(1.0, 0.95), xycoords=("data", "axes fraction"),
                    fontsize=7, color="#888", ha="left", va="top",
                    xytext=(4, 0), textcoords="offset points")

        ax.set_xlabel("Normierter Abstand (0=Zentrum, 1=Oberfläche)", fontsize=8, color=tc)
        ax.set_ylabel("|Fehler|", fontsize=9, color=tc)
        ax.set_title("Radiales Fehlerprofil", fontsize=9, color=tc)
        ax.legend(handles=handles, fontsize=6, loc="upper right",
                  facecolor="#1a2030", edgecolor="#333", labelcolor=tc, ncol=2)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#333")

    def _draw_timing_bars(self, ax, res, tc):
        """Timing bar chart."""
        timing = res["timing"]
        names = list(timing.keys())
        values = list(timing.values())
        short_names = ["Exakt"] + [s for _, _, s in METHODS]

        colors = ["#888"] + ["#f2e641", "#4962f2", "#50c878", "#c878ff", "#ff7f50"]
        bars = ax.barh(range(len(names)), values, color=colors[:len(names)],
                       edgecolor="none", height=0.6)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=8, color=tc)
        ax.set_xlabel("ms (48³ Grid)", fontsize=9, color=tc)
        ax.set_title("Laufzeit", fontsize=9, color=tc)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#333")
        ax.invert_yaxis()

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=7, color=tc)

    # ── Update table ──────────────────────────────────────────────────

    def _update_table(self, res):
        sl = res["slices"]["XY"]
        regions = ["total", "interior", "exterior", "near_surface"]
        region_labels = ["Gesamt", "Interior", "Exterior", "Oberfläche (±15%)"]

        headers = ["Methode"]
        for rl in region_labels:
            headers += [f"MAE ({rl})", f"RMSE ({rl})", f"L∞ ({rl})"]
        headers.append("Zeit (ms)")

        t = self._table
        t.clear()
        t.setColumnCount(len(headers))
        t.setRowCount(len(METHODS))
        t.setHorizontalHeaderLabels(headers)

        timing = res["timing"]

        for row, (name, _, short) in enumerate(METHODS):
            md = sl["methods"][name]
            t.setItem(row, 0, QtWidgets.QTableWidgetItem(name))

            col = 1
            for region in regions:
                m = md["metrics"][region]
                for key in ["mae", "rmse", "l_inf"]:
                    item = QtWidgets.QTableWidgetItem(f"{m[key]:.6f}")
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                    t.setItem(row, col, item)
                    col += 1

            item = QtWidgets.QTableWidgetItem(f"{timing[name]:.2f}")
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            t.setItem(row, col, item)

        t.resizeColumnsToContents()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(13, 17, 23))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(208, 208, 208))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(22, 27, 35))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(30, 37, 48))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(208, 208, 208))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(33, 40, 52))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(208, 208, 208))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(73, 98, 242))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(palette)

    win = BenchmarkWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()