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

Uses NVIDIA Warp for GPU acceleration when CUDA is available.
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
# WARP SETUP & DEVICE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

try:
    import warp as wp
    wp.init()
    _HAS_WARP = True
    _DEVICE = "cuda" if wp.is_cuda_available() else "cpu"
except Exception:
    _HAS_WARP = False
    _DEVICE = "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# WARP KERNELS
# ══════════════════════════════════════════════════════════════════════════════

if _HAS_WARP:

    @wp.kernel
    def _wp_exact_kernel(
        points: wp.array(dtype=wp.float64, ndim=2),
        r0: wp.float64, r1: wp.float64, r2: wp.float64,
        out: wp.array(dtype=wp.float64),
    ):
        tid = wp.tid()
        px = points[tid, 0]
        py = points[tid, 1]
        pz = points[tid, 2]

        r0_2 = r0 * r0
        r1_2 = r1 * r1
        r2_2 = r2 * r2

        # Inside test
        s0 = px / r0
        s1 = py / r1
        s2 = pz / r2
        inside = (s0 * s0 + s1 * s1 + s2 * s2) < wp.float64(1.0)

        # Epsilon relative to smallest radius
        r_min = wp.min(wp.min(r0, r1), r2)
        r_max = wp.max(wp.max(r0, r1), r2)
        eps = wp.float64(1.0e-10) * r_min

        ax = wp.max(wp.abs(px), eps)
        ay = wp.max(wp.abs(py), eps)
        az = wp.max(wp.abs(pz), eps)

        p_len = wp.sqrt(ax * ax + ay * ay + az * az)

        t_lo = wp.float64(0.0)
        t_hi = wp.float64(0.0)
        if inside:
            t_lo = -r_min * r_min * (wp.float64(1.0) - wp.float64(1.0e-15))
            t_hi = wp.float64(0.0)
        else:
            t_lo = wp.float64(0.0)
            t_hi = r_max * p_len + r0_2 + r1_2 + r2_2

        # Bisection: 80 iterations
        for _i in range(80):
            t_mid = wp.float64(0.5) * (t_lo + t_hi)
            d0 = r0 * ax / (r0_2 + t_mid)
            d1 = r1 * ay / (r1_2 + t_mid)
            d2 = r2 * az / (r2_2 + t_mid)
            F = d0 * d0 + d1 * d1 + d2 * d2 - wp.float64(1.0)
            if F > wp.float64(0.0):
                t_lo = t_mid
            else:
                t_hi = t_mid

        t = wp.float64(0.5) * (t_lo + t_hi)
        qx = r0_2 * ax / (r0_2 + t)
        qy = r1_2 * ay / (r1_2 + t)
        qz = r2_2 * az / (r2_2 + t)
        ddx = ax - qx
        ddy = ay - qy
        ddz = az - qz
        dist = wp.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)

        if inside:
            out[tid] = -dist
        else:
            out[tid] = dist


    @wp.kernel
    def _wp_quilez_kernel(
        points: wp.array(dtype=wp.float64, ndim=2),
        r0: wp.float64, r1: wp.float64, r2: wp.float64,
        out: wp.array(dtype=wp.float64),
    ):
        tid = wp.tid()
        px = points[tid, 0]
        py = points[tid, 1]
        pz = points[tid, 2]

        s0 = px / r0
        s1 = py / r1
        s2 = pz / r2
        k0 = wp.sqrt(s0 * s0 + s1 * s1 + s2 * s2)

        s20 = px / (r0 * r0)
        s21 = py / (r1 * r1)
        s22 = pz / (r2 * r2)
        k1 = wp.max(wp.sqrt(s20 * s20 + s21 * s21 + s22 * s22), wp.float64(1.0e-15))

        out[tid] = k0 * (k0 - wp.float64(1.0)) / k1


    @wp.kernel
    def _wp_scaled_kernel(
        points: wp.array(dtype=wp.float64, ndim=2),
        r0: wp.float64, r1: wp.float64, r2: wp.float64,
        scale: wp.float64,
        out: wp.array(dtype=wp.float64),
    ):
        tid = wp.tid()
        px = points[tid, 0]
        py = points[tid, 1]
        pz = points[tid, 2]

        s0 = px / r0
        s1 = py / r1
        s2 = pz / r2
        k = wp.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
        out[tid] = (k - wp.float64(1.0)) * scale


# ══════════════════════════════════════════════════════════════════════════════
# SDF METHODS — CPU (NumPy, float64) + GPU (Warp) wrappers
# ══════════════════════════════════════════════════════════════════════════════

def _wp_run_exact(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Exact bisection SDF via Warp kernel."""
    pts = wp.array(points.astype(np.float64), dtype=wp.float64, device=_DEVICE)
    out = wp.empty(len(points), dtype=wp.float64, device=_DEVICE)
    r = radii.astype(np.float64)
    wp.launch(_wp_exact_kernel, dim=len(points),
              inputs=[pts, r[0], r[1], r[2], out], device=_DEVICE)
    wp.synchronize_device(_DEVICE)
    return out.numpy()


def _wp_run_quilez(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Quílez SDF via Warp kernel."""
    pts = wp.array(points.astype(np.float64), dtype=wp.float64, device=_DEVICE)
    out = wp.empty(len(points), dtype=wp.float64, device=_DEVICE)
    r = radii.astype(np.float64)
    wp.launch(_wp_quilez_kernel, dim=len(points),
              inputs=[pts, r[0], r[1], r[2], out], device=_DEVICE)
    wp.synchronize_device(_DEVICE)
    return out.numpy()


def _wp_run_scaled(points: np.ndarray, radii: np.ndarray,
                   scale: float) -> np.ndarray:
    """Scaled-sphere SDF via Warp kernel (generic, scale passed in)."""
    pts = wp.array(points.astype(np.float64), dtype=wp.float64, device=_DEVICE)
    out = wp.empty(len(points), dtype=wp.float64, device=_DEVICE)
    r = radii.astype(np.float64)
    wp.launch(_wp_scaled_kernel, dim=len(points),
              inputs=[pts, r[0], r[1], r[2], float(scale), out],
              device=_DEVICE)
    wp.synchronize_device(_DEVICE)
    return out.numpy()


def _wp_scaled_min(points, radii):
    return _wp_run_scaled(points, radii, float(np.min(radii)))

def _wp_scaled_mean(points, radii):
    return _wp_run_scaled(points, radii, float(np.mean(radii)))

def _wp_scaled_gmean(points, radii):
    return _wp_run_scaled(points, radii, float(np.cbrt(np.prod(radii.astype(np.float64)))))


# ── CPU fallbacks (pure NumPy, float64) ──────────────────────────────────────

def _cpu_sdf_exact(points: np.ndarray, radii: np.ndarray,
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


def _cpu_sdf_quilez(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Ínigo Quílez approximation:  k0·(k0 − 1) / k1."""
    r = radii.astype(np.float64)
    p = points.astype(np.float64)
    scaled  = p / r
    scaled2 = p / (r * r)
    k0 = np.linalg.norm(scaled, axis=1)
    k1 = np.maximum(np.linalg.norm(scaled2, axis=1), 1e-15)
    return k0 * (k0 - 1.0) / k1


def _cpu_sdf_scaled_min(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · min(r)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.min(r)


def _cpu_sdf_scaled_mean(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · mean(r)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.mean(r)


def _cpu_sdf_scaled_gmean(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · ∛(r₁r₂r₃)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.cbrt(np.prod(r))


# Method registry: (display_name, function, short_name_for_plots)
# Uses Warp/GPU when available, CPU fallback otherwise.
if _HAS_WARP:
    sdf_exact = _wp_run_exact
    METHODS = [
        ("Quílez",                _wp_run_quilez,   "Quílez"),
        ("Scaled (min r)",        _wp_scaled_min,   "Sc-min"),
        ("Scaled (mean r)",       _wp_scaled_mean,  "Sc-mean"),
        ("Scaled (geom. mean r)", _wp_scaled_gmean, "Sc-gmean"),
    ]
else:
    sdf_exact = _cpu_sdf_exact
    METHODS = [
        ("Quílez",                _cpu_sdf_quilez,      "Quílez"),
        ("Scaled (min r)",        _cpu_sdf_scaled_min,  "Sc-min"),
        ("Scaled (mean r)",       _cpu_sdf_scaled_mean, "Sc-mean"),
        ("Scaled (geom. mean r)", _cpu_sdf_scaled_gmean,"Sc-gmean"),
    ]

print(f"[benchmark_sdf] Device: {_DEVICE}"
      f"  |  Warp: {'yes' if _HAS_WARP else 'no'}"
      f"  |  CUDA: {'yes' if _HAS_WARP and _DEVICE == 'cuda' else 'no'}")


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
        "x-axis":    np.array([1.0, 0.0, 0.0]),
        "y-axis":    np.array([0.0, 1.0, 0.0]),
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
# THEMES
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "dark": dict(
        fig_bg="#0d1117",
        ax_bg="#0d1117",
        text="#d0d0d0",
        spine="#333333",
        grid="#555555",
        legend_bg="#1a2030",
        legend_edge="#333333",
        contour="black",
        status="#888888",
        vline="#666666",
        annotate="#888888",
        ray_legend="#aaaaaa",
        # Qt palette
        qt_window=(13, 17, 23),
        qt_window_text=(208, 208, 208),
        qt_base=(22, 27, 35),
        qt_alt_base=(30, 37, 48),
        qt_text=(208, 208, 208),
        qt_button=(33, 40, 52),
        qt_button_text=(208, 208, 208),
        qt_highlight=(73, 98, 242),
        qt_highlight_text=(255, 255, 255),
    ),
    "light": dict(
        fig_bg="#ffffff",
        ax_bg="#ffffff",
        text="#1a1a1a",
        spine="#cccccc",
        grid="#dddddd",
        legend_bg="#f5f5f5",
        legend_edge="#cccccc",
        contour="#222222",
        status="#666666",
        vline="#999999",
        annotate="#666666",
        ray_legend="#555555",
        # Qt palette
        qt_window=(255, 255, 255),
        qt_window_text=(26, 26, 26),
        qt_base=(255, 255, 255),
        qt_alt_base=(245, 245, 245),
        qt_text=(26, 26, 26),
        qt_button=(235, 235, 235),
        qt_button_text=(26, 26, 26),
        qt_highlight=(73, 98, 242),
        qt_highlight_text=(255, 255, 255),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Qt GUI
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self._theme_name = "dark"
        dev_tag = "GPU" if _DEVICE == "cuda" else "CPU"
        self.setWindowTitle(f"Ellipsoid SDF Benchmark  [{dev_tag}]")
        self.resize(1700, 1100)
        self._build_ui()
        self._apply_qt_palette()

    @property
    def _theme(self):
        return THEMES[self._theme_name]

    def _apply_qt_palette(self):
        t = self._theme
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(*t["qt_window"]))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(*t["qt_window_text"]))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(*t["qt_base"]))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(*t["qt_alt_base"]))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(*t["qt_text"]))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(*t["qt_button"]))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(*t["qt_button_text"]))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(*t["qt_highlight"]))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(*t["qt_highlight_text"]))
        QtWidgets.QApplication.instance().setPalette(palette)
        self._lbl_status.setStyleSheet(f"color: {t['status']}; font-style: italic;")
        self._fig.set_facecolor(t["fig_bg"])

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
        self._btn_run = QtWidgets.QPushButton("▶  Compute")
        self._btn_run.setFixedHeight(32)
        self._btn_run.clicked.connect(self._on_run)
        bar.addWidget(self._btn_run)

        self._chk_interior = QtWidgets.QCheckBox("Interior only (SDF < 0)")
        self._chk_interior.setChecked(False)
        self._chk_interior.setToolTip("Compute error only for points inside the ellipsoid")
        self._chk_interior.toggled.connect(self._on_interior_toggled)
        bar.addWidget(self._chk_interior)

        self._btn_pdf = QtWidgets.QPushButton("📄 Export PDF")
        self._btn_pdf.setFixedHeight(32)
        self._btn_pdf.setEnabled(False)
        self._btn_pdf.clicked.connect(self._on_export_pdf)
        bar.addWidget(self._btn_pdf)

        bar.addSpacing(10)
        self._btn_theme = QtWidgets.QPushButton("☀ Light")
        self._btn_theme.setFixedHeight(32)
        self._btn_theme.setFixedWidth(80)
        self._btn_theme.clicked.connect(self._on_toggle_theme)
        bar.addWidget(self._btn_theme)

        bar.addStretch()

        self._lbl_status = QtWidgets.QLabel("")
        self._lbl_status.setStyleSheet("color: #888; font-style: italic;")
        bar.addWidget(self._lbl_status)

        root.addLayout(bar)

        # ── Matplotlib figure ─────────────────────────────────────────
        self._fig = Figure(figsize=(18, 12), dpi=100, facecolor=self._theme["fig_bg"])
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

        self._lbl_status.setText("Computing …")
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
        dev_tag = f"GPU ({_DEVICE})" if _DEVICE == "cuda" else "CPU"
        self._lbl_status.setText(
            f"Done — κ = {ar:.2f}  |  {elapsed:.1f} s  |  Grid {n}×{n}  |  {dev_tag}"
        )
        self._btn_run.setEnabled(True)

    # ── PDF export ────────────────────────────────────────────────────

    def _on_export_pdf(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save PDF", "benchmark_sdf.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if path:
            self._fig.savefig(path, bbox_inches="tight", facecolor=self._fig.get_facecolor())
            self._lbl_status.setText(f"Exported: {path}")

    def _on_interior_toggled(self, checked):
        if hasattr(self, "_last_results"):
            self._update_figure(self._last_results)
            self._update_table(self._last_results)

    def _on_toggle_theme(self):
        if self._theme_name == "dark":
            self._theme_name = "light"
            self._btn_theme.setText("🌙 Dark")
        else:
            self._theme_name = "dark"
            self._btn_theme.setText("☀ Light")
        self._apply_qt_palette()
        if hasattr(self, "_last_results"):
            self._update_figure(self._last_results)
        self._canvas.draw()

    # ── Update figure ─────────────────────────────────────────────────

    def _update_figure(self, res):
        self._fig.clear()
        t = self._theme
        self._fig.set_facecolor(t["fig_bg"])

        n_methods = len(METHODS)
        n_cols = 1 + n_methods  # GT + methods
        interior_only = self._chk_interior.isChecked()

        gs = gridspec.GridSpec(
            3, n_cols, figure=self._fig,
            height_ratios=[1, 1, 0.85],
            hspace=0.40, wspace=0.25,
        )

        tc = t["text"]
        radii = res["radii"]
        extent = res["extent"]
        ar = res["aspect_ratio"]

        sdf_vmin = -extent * 0.3
        sdf_vmax =  extent * 0.3

        sl = res["slices"]["XY"]
        coords = sl["coords"]
        gt = sl["gt"]
        ext_kw = dict(extent=[-extent, extent, -extent, extent],
                      origin="lower", aspect="equal")

        interior_mask = gt < 0.0

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

        # ── Row 0: SDF — Ground truth ─────────────────────────────────
        ax_gt = self._fig.add_subplot(gs[0, 0])
        ax_gt.imshow(gt, cmap="RdYlBu_r",
                     vmin=sdf_vmin, vmax=sdf_vmax, **ext_kw)
        ax_gt.contour(coords, coords, gt, levels=[0],
                      colors=t["contour"], linewidths=0.8)
        ax_gt.set_title("Ground Truth\nXY slice (z = 0)",
                        fontsize=9, color=tc)
        ax_gt.set_xlabel("x", fontsize=8, color=tc)
        ax_gt.set_ylabel("y", fontsize=8, color=tc)
        ax_gt.tick_params(colors=tc, labelsize=7)
        ax_gt.set_facecolor(t["ax_bg"])

        # ── Row 0: SDF — each method ─────────────────────────────────
        for col_idx, (name, _, short) in enumerate(METHODS):
            md = sl["methods"][name]
            ax = self._fig.add_subplot(gs[0, col_idx + 1])
            ax.imshow(md["sdf"], cmap="RdYlBu_r",
                      vmin=sdf_vmin, vmax=sdf_vmax, **ext_kw)
            ax.contour(coords, coords, md["sdf"], levels=[0],
                       colors=t["contour"], linewidths=0.6)
            ax.set_title(f"{name} — SDF",
                         fontsize=9, color=tc)
            ax.set_xlabel("x", fontsize=8, color=tc)
            ax.tick_params(colors=tc, labelsize=7)
            ax.set_facecolor(t["ax_bg"])

        # ── Row 1: Error — label in col 0 ────────────────────────────
        ax_empty = self._fig.add_subplot(gs[1, 0])
        ax_empty.set_facecolor(t["ax_bg"])
        label_extra = "\n(interior only)" if interior_only else ""
        ax_empty.text(0.5, 0.5, f"|Error|\nXY slice (z = 0){label_extra}",
                      ha="center", va="center", fontsize=10,
                      color=tc, transform=ax_empty.transAxes)
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])
        for spine in ax_empty.spines.values():
            spine.set_color(t["spine"])

        # ── Row 1: Error — each method heatmap ───────────────────────
        for col_idx, (name, _, short) in enumerate(METHODS):
            ax = self._fig.add_subplot(gs[1, col_idx + 1])
            abs_err = np.abs(error_maps[name])
            ax.imshow(abs_err, cmap="Reds", vmin=0, vmax=max_err, **ext_kw)
            ax.contour(coords, coords, gt, levels=[0],
                       colors=t["contour"], linewidths=0.6)
            mae = mae_vals[name]
            ax.set_title(f"{name} — |Error|\nMAE = {mae:.4f}",
                         fontsize=9, color=tc)
            ax.set_xlabel("x", fontsize=8, color=tc)
            ax.tick_params(colors=tc, labelsize=7)
            ax.set_facecolor(t["ax_bg"])

        # ── Row 2: bar charts + angular profile + timing ──────────────

        ax_bar = self._fig.add_subplot(gs[2, 0:2])
        self._draw_metric_bars(ax_bar, res, tc)

        ax_rad = self._fig.add_subplot(gs[2, 2:4])
        self._draw_radial_profile(ax_rad, res, tc)

        if n_cols > 4:
            ax_time = self._fig.add_subplot(gs[2, 4])
        else:
            ax_time = self._fig.add_subplot(gs[2, n_cols - 1])
        self._draw_timing_bars(ax_time, res, tc)

        # ── Suptitle ──────────────────────────────────────────────────
        self._fig.suptitle(
            f"Ellipsoid SDF Benchmark   —   r = ({radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f})"
            f"   κ = {ar:.2f}",
            fontsize=13, color=tc, y=0.99,
        )

        self._canvas.draw()

    # ── Sub-plots ─────────────────────────────────────────────────────

    def _draw_metric_bars(self, ax, res, tc):
        """Grouped bar chart: MAE, RMSE, L∞ for XY slice."""
        t = self._theme
        interior_only = self._chk_interior.isChecked()
        region = "interior" if interior_only else "total"
        region_label = "Interior" if interior_only else "Total"

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
        ax.set_ylabel("Error", fontsize=9, color=tc)
        ax.set_title(f"Error Metrics (XY, {region_label})", fontsize=9, color=tc)
        ax.legend(fontsize=8, loc="upper left", facecolor=t["legend_bg"],
                  edgecolor=t["legend_edge"], labelcolor=tc)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor(t["ax_bg"])
        ax.spines[:].set_color(t["spine"])

    def _draw_radial_profile(self, ax, res, tc):
        """Error vs. normalized radial distance along rays from center."""
        t = self._theme
        rad = res["radial"]
        t_norm = rad["t_norm"]
        method_colors = ["#f2e641", "#4962f2", "#50c878", "#c878ff"]
        line_styles = ["-", "--", ":"]

        for i, (name, _, short) in enumerate(METHODS):
            for j, (ray_label, ray_data) in enumerate(rad["rays"].items()):
                label = f"{short} ({ray_label})" if j == 0 or i == 0 else None
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

        from matplotlib.lines import Line2D
        handles = []
        for i, (name, _, short) in enumerate(METHODS):
            handles.append(Line2D([0], [0], color=method_colors[i % len(method_colors)],
                                  linewidth=1.5, label=short))
        for j, ray_label in enumerate(rad["rays"].keys()):
            handles.append(Line2D([0], [0], color=t["ray_legend"],
                                  linestyle=line_styles[j % len(line_styles)],
                                  linewidth=1.2, label=ray_label))

        ax.axvline(1.0, color=t["vline"], linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate("Surface", xy=(1.0, 0.95), xycoords=("data", "axes fraction"),
                    fontsize=7, color=t["annotate"], ha="left", va="top",
                    xytext=(4, 0), textcoords="offset points")

        ax.set_xlabel("Normalized distance (0=center, 1=surface)", fontsize=8, color=tc)
        ax.set_ylabel("|Error|", fontsize=9, color=tc)
        ax.set_title("Radial Error Profile", fontsize=9, color=tc)
        ax.legend(handles=handles, fontsize=6, loc="upper right",
                  facecolor=t["legend_bg"], edgecolor=t["legend_edge"],
                  labelcolor=tc, ncol=2)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor(t["ax_bg"])
        ax.spines[:].set_color(t["spine"])

    def _draw_timing_bars(self, ax, res, tc):
        """Timing bar chart."""
        t = self._theme
        timing = res["timing"]
        names = list(timing.keys())
        values = list(timing.values())
        short_names = ["Exact"] + [s for _, _, s in METHODS]

        colors = ["#888"] + ["#f2e641", "#4962f2", "#50c878", "#c878ff"]
        bars = ax.barh(range(len(names)), values, color=colors[:len(names)],
                       edgecolor="none", height=0.6)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=8, color=tc)
        ax.set_xlabel("ms (48³ Grid)", fontsize=9, color=tc)
        ax.set_title("Runtime", fontsize=9, color=tc)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor(t["ax_bg"])
        ax.spines[:].set_color(t["spine"])
        ax.invert_yaxis()

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=7, color=tc)

    # ── Update table ──────────────────────────────────────────────────

    def _update_table(self, res):
        sl = res["slices"]["XY"]
        regions = ["total", "interior", "exterior", "near_surface"]
        region_labels = ["Total", "Interior", "Exterior", "Surface (±15%)"]

        headers = ["Method"]
        for rl in region_labels:
            headers += [f"MAE ({rl})", f"RMSE ({rl})", f"L∞ ({rl})"]
        headers.append("Time (ms)")

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

    win = BenchmarkWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()