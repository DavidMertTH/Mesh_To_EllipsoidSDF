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
# SDF METHODS  (pure NumPy, CPU, float64)
# ══════════════════════════════════════════════════════════════════════════════

def sdf_exact(points: np.ndarray, radii: np.ndarray,
              n_bisect: int = 80) -> np.ndarray:
    """Ground-truth signed distance via bisection.

    Finds parameter *t* such that the closest-point equation
        q_i = r_i² · |p_i| / (r_i² + t)
    satisfies  Σ (q_i / r_i)² = 1  (point on surface).

    Signed: negative inside, positive outside.
    """
    r = radii.astype(np.float64)
    r2 = r ** 2
    pts = points.astype(np.float64)

    inside = np.sum((pts / r) ** 2, axis=1) < 1.0

    # Small epsilon relative to smallest axis — keeps bisection stable
    # for points near the origin without distorting other results.
    eps = 1e-10 * np.min(r)
    p = np.maximum(np.abs(pts), eps)

    N = len(p)
    t_lo = np.empty(N, dtype=np.float64)
    t_hi = np.empty(N, dtype=np.float64)

    # Exterior: t ∈ [0, T_max]
    T_max = np.max(r) * np.linalg.norm(p, axis=1) + np.sum(r2)
    t_lo[~inside] = 0.0
    t_hi[~inside] = T_max[~inside]

    # Interior: t ∈ [T_min, 0]  where T_min → −min(r²)
    T_min = -np.min(r2) * (1.0 - 1e-15)
    t_lo[inside] = T_min
    t_hi[inside] = 0.0

    # F(t) = Σ (r·p / (r²+t))² − 1   is monotonically decreasing
    for _ in range(n_bisect):
        t_mid = 0.5 * (t_lo + t_hi)
        denom = r2 + t_mid[:, None]                    # (N, 3)
        F = np.sum((r * p / denom) ** 2, axis=1) - 1.0  # (N,)
        move_lo = F > 0
        t_lo = np.where(move_lo, t_mid, t_lo)
        t_hi = np.where(~move_lo, t_mid, t_hi)

    t = 0.5 * (t_lo + t_hi)
    q = r2 * p / (r2 + t[:, None])
    dist = np.linalg.norm(p - q, axis=1)
    return np.where(inside, -dist, dist)


def sdf_quilez(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Ínigo Quílez approximation:  k0·(k0 − 1) / k1."""
    r = radii.astype(np.float64)
    p = points.astype(np.float64)
    scaled  = p / r
    scaled2 = p / (r * r)
    k0 = np.linalg.norm(scaled, axis=1)
    k1 = np.maximum(np.linalg.norm(scaled2, axis=1), 1e-15)
    return k0 * (k0 - 1.0) / k1


def sdf_scaled_min(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · min(r)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.min(r)


def sdf_scaled_mean(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · mean(r)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.mean(r)


def sdf_scaled_gmean(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · ∛(r₁r₂r₃)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.cbrt(np.prod(r))


# Method registry: (display_name, function, short_name_for_plots)
METHODS = [
    ("Quílez",                sdf_quilez,      "Quílez"),
    ("Scaled (min r)",        sdf_scaled_min,  "Sc-min"),
    ("Scaled (mean r)",       sdf_scaled_mean, "Sc-mean"),
    ("Scaled (geom. mean r)", sdf_scaled_gmean,"Sc-gmean"),
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

    # ── Angular error profile (circle in XY plane) ────────────────────
    n_angles = 720
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    # Sample at ~10 % outside the ellipsoid in each direction
    R_fac = 1.1
    xy_r = np.stack([radii[0] * np.cos(angles),
                     radii[1] * np.sin(angles),
                     np.zeros(n_angles)], axis=1) * R_fac
    gt_ang = sdf_exact(xy_r, radii)
    ang_data = dict(angles=np.degrees(angles), gt=gt_ang)
    for name, func, short in METHODS:
        ang_data[name] = func(xy_r, radii) - gt_ang

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
        angular=ang_data,
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
        self.resize(1700, 1000)
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
        self._fig = Figure(figsize=(18, 10), dpi=100, facecolor="#0d1117")
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

    # ── Update figure ─────────────────────────────────────────────────

    def _update_figure(self, res):
        self._fig.clear()

        n_methods = len(METHODS)
        n_cols = 1 + n_methods  # GT + methods

        gs = gridspec.GridSpec(
            3, n_cols, figure=self._fig,
            height_ratios=[1, 1, 0.85],
            hspace=0.35, wspace=0.25,
        )

        text_color = "#d0d0d0"
        radii = res["radii"]
        extent = res["extent"]
        ar = res["aspect_ratio"]

        # ── Shared error limits across all methods ────────────────────
        max_err = 0.0
        for sl_key in ["XY", "XZ"]:
            for name, _, _ in METHODS:
                me = np.max(np.abs(res["slices"][sl_key]["methods"][name]["error"]))
                max_err = max(max_err, me)
        if max_err < 1e-12:
            max_err = 1e-6
        err_norm = TwoSlopeNorm(vmin=-max_err, vcenter=0, vmax=max_err)

        slice_labels = {"XY": "XY-Schnitt (z = 0)", "XZ": "XZ-Schnitt (y = 0)"}
        axis_labels  = {"XY": ("x", "y"), "XZ": ("x", "z")}

        for row_idx, sl_key in enumerate(["XY", "XZ"]):
            sl = res["slices"][sl_key]
            coords = sl["coords"]
            ext_kw = dict(extent=[-extent, extent, -extent, extent],
                          origin="lower", aspect="equal")
            ax_lbl = axis_labels[sl_key]

            # Ground truth
            ax_gt = self._fig.add_subplot(gs[row_idx, 0])
            im = ax_gt.imshow(sl["gt"], cmap="RdYlBu_r",
                              vmin=-extent * 0.3, vmax=extent * 0.3, **ext_kw)
            ax_gt.contour(coords, coords, sl["gt"], levels=[0],
                          colors="white", linewidths=0.8)
            ax_gt.set_title(f"Ground Truth\n{slice_labels[sl_key]}",
                            fontsize=9, color=text_color)
            ax_gt.set_xlabel(ax_lbl[0], fontsize=8, color=text_color)
            ax_gt.set_ylabel(ax_lbl[1], fontsize=8, color=text_color)
            ax_gt.tick_params(colors=text_color, labelsize=7)
            ax_gt.set_facecolor("#0d1117")

            # Error maps
            for col_idx, (name, _, short) in enumerate(METHODS):
                md = sl["methods"][name]
                ax = self._fig.add_subplot(gs[row_idx, col_idx + 1])
                ax.imshow(md["error"], cmap="RdBu_r", norm=err_norm, **ext_kw)
                ax.contour(coords, coords, sl["gt"], levels=[0],
                           colors="white", linewidths=0.6)
                mae = md["metrics"]["total"]["mae"]
                ax.set_title(f"{name}\nMAE = {mae:.4f}", fontsize=9, color=text_color)
                ax.set_xlabel(ax_lbl[0], fontsize=8, color=text_color)
                ax.tick_params(colors=text_color, labelsize=7)
                ax.set_facecolor("#0d1117")

        # ── Row 2: bar charts + angular profile ───────────────────────

        # Panel 1: grouped MAE / RMSE / L∞
        ax_bar = self._fig.add_subplot(gs[2, 0:2])
        self._draw_metric_bars(ax_bar, res, text_color)

        # Panel 2: angular error profile
        ax_ang = self._fig.add_subplot(gs[2, 2:4])
        self._draw_angular_profile(ax_ang, res, text_color)

        # Panel 3: timing
        ax_time = self._fig.add_subplot(gs[2, 4])
        self._draw_timing_bars(ax_time, res, text_color)

        # ── Suptitle ──────────────────────────────────────────────────
        self._fig.suptitle(
            f"Ellipsoid SDF Benchmark   —   r = ({radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f})"
            f"   κ = {ar:.2f}",
            fontsize=13, color=text_color, y=0.98,
        )

        self._canvas.draw()

    # ── Sub-plots ─────────────────────────────────────────────────────

    def _draw_metric_bars(self, ax, res, tc):
        """Grouped bar chart: MAE, RMSE, L∞ for XY slice, total region."""
        sl = res["slices"]["XY"]
        names, mae, rmse, linf = [], [], [], []
        for name, _, short in METHODS:
            m = sl["methods"][name]["metrics"]["total"]
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
        ax.set_title("Fehlermetriken (XY, gesamt)", fontsize=9, color=tc)
        ax.legend(fontsize=8, loc="upper left", facecolor="#1a2030",
                  edgecolor="#333", labelcolor=tc)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#333")

    def _draw_angular_profile(self, ax, res, tc):
        """Error vs. angle on a circle just outside the ellipsoid (XY plane)."""
        ang = res["angular"]
        angles = ang["angles"]
        colors = ["#f2e641", "#4962f2", "#50c878", "#c878ff"]
        for i, (name, _, short) in enumerate(METHODS):
            ax.plot(angles, np.abs(ang[name]),
                    color=colors[i % len(colors)], linewidth=1.2, label=short)

        ax.set_xlabel("Winkel (°)", fontsize=9, color=tc)
        ax.set_ylabel("|Fehler|", fontsize=9, color=tc)
        ax.set_title("Winkelprofil (1.1 × Oberfläche, XY)", fontsize=9, color=tc)
        ax.set_xlim(0, 360)
        ax.legend(fontsize=7, loc="upper right", facecolor="#1a2030",
                  edgecolor="#333", labelcolor=tc)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#333")

    def _draw_timing_bars(self, ax, res, tc):
        """Timing bar chart."""
        timing = res["timing"]
        names = list(timing.keys())
        values = list(timing.values())
        short_names = ["Exakt"] + [s for _, _, s in METHODS]

        colors = ["#888"] + ["#f2e641", "#4962f2", "#50c878", "#c878ff"]
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
