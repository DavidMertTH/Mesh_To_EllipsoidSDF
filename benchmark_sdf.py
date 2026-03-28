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
  - Gradient-Projektion       →  Newton-Schritt auf Closest-Point entlang ∇f
  - k₁/k₂                    →  (k0−1) · k1/k2  (hierarchische Normen)
  - Krümmungskorrektur        →  φ_Q / (1 + κ_n · φ_Q)
  - Newton-Schritt            →  Ray-Ellipsoid-Schnitt entlang Normale bei q₀
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


# ── Idee 1: Gradientenbasierte Projektion ────────────────────────────────
def sdf_gradient_proj(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Gradient-based projection (one Newton step on closest-point).

    Instead of projecting radially (in normalized space) onto the surface,
    project along the gradient direction ∇f = 2p/r².  This gives a better
    estimate q₁ of the closest surface point, then compute the tangent-plane
    distance at q₁.
    """
    r = radii.astype(np.float64)
    r2 = r * r
    p = points.astype(np.float64)
    eps = 1e-15

    # Gradient direction at p (unnormalized): g_i = p_i / r_i²
    g = p / r2                                        # (N, 3)
    g_norm = np.linalg.norm(g, axis=1, keepdims=True)
    g_hat = g / np.maximum(g_norm, eps)               # unit gradient

    # Find t such that q = p - t * g_hat lies on the ellipsoid:
    #   Σ (p_i - t * g_hat_i)² / r_i² = 1
    # This is a quadratic in t:  A t² - 2B t + C = 0
    #   A = Σ g_hat_i² / r_i²
    #   B = Σ p_i g_hat_i / r_i²
    #   C = Σ p_i² / r_i² = k₀²
    A = np.sum(g_hat ** 2 / r2, axis=1)               # (N,)
    B = np.sum(p * g_hat / r2, axis=1)                 # (N,)
    C = np.sum(p ** 2 / r2, axis=1)                    # = k₀²

    disc = B ** 2 - A * (C - 1.0)
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)

    # Two solutions: t = (B ± √disc) / A
    # For exterior points (C > 1) take the smaller root (closer intersection)
    # For interior points (C < 1) take the negative root
    inside = C < 1.0
    t = np.where(inside,
                 (B - sqrt_disc) / np.maximum(A, eps),
                 (B - sqrt_disc) / np.maximum(A, eps))

    # Projected surface point
    q1 = p - t[:, None] * g_hat                       # (N, 3)

    # Normal at q1: n = q1/r² (unnormalized)
    n = q1 / r2
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_hat = n / np.maximum(n_norm, eps)

    # Signed distance = (p - q1) · n_hat
    dist = np.sum((p - q1) * n_hat, axis=1)
    return dist


# ── Idee 2: Hierarchische skalierte Normen (k₁/k₂) ─────────────────────
def sdf_k1k2(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Hierarchical scaled norms:  (k₀ − 1) · k₁ / k₂.

    Uses k₂ = ||p/r³|| as the back-scaling factor evaluated at the
    approximate surface point rather than at p.  For a sphere k₁/k₂ = r,
    recovering the exact SDF.  For ellipsoids the shorter axes are weighted
    more strongly, potentially giving better directional adaptation.
    """
    r = radii.astype(np.float64)
    p = points.astype(np.float64)
    k0 = np.linalg.norm(p / r, axis=1)
    k1 = np.linalg.norm(p / (r ** 2), axis=1)
    k2 = np.linalg.norm(p / (r ** 3), axis=1)
    k2 = np.maximum(k2, 1e-15)
    return (k0 - 1.0) * k1 / k2


# ── Idee 3: Krümmungskorrektur ──────────────────────────────────────────
def sdf_curvature_corr(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Curvature-corrected Quílez:  φ_Q / (1 + κ_n · φ_Q).

    The Quílez formula is a tangent-plane distance and overestimates the
    true SDF for convex surfaces.  This correction uses the normal curvature
    κ_n at the radially projected surface point q₀ = p/k₀ to account for
    the surface curving away from p.
    """
    r = radii.astype(np.float64)
    r2 = r * r
    p = points.astype(np.float64)
    eps = 1e-15

    # Quílez SDF
    scaled = p / r
    scaled2 = p / r2
    k0 = np.linalg.norm(scaled, axis=1)
    k1 = np.maximum(np.linalg.norm(scaled2, axis=1), eps)
    phi_q = k0 * (k0 - 1.0) / k1

    # Projected surface point: q0 = p / k0
    k0_safe = np.maximum(k0, eps)
    q0 = p / k0_safe[:, None]                          # (N, 3)

    # Normal curvature at q0 in direction of (p - q0)
    # For an ellipsoid x²/a² + y²/b² + z²/c² = 1:
    #   Gaussian curvature K = 1/(a²b²c²) · 1/(Σ q_i²/r_i⁴)²
    #   Mean curvature H = ... (complex)
    # Simpler: normal curvature κ_n = II(v,v) / I(v,v)
    # where v is the projection direction.
    #
    # The second fundamental form coefficient in direction v
    # for implicit surface F(x) = Σ x_i²/r_i² - 1:
    #   κ_n = (v^T H_F v) / |∇F| where H_F is the Hessian of F
    # H_F = diag(2/r_i²), ∇F = 2q/r² at q0
    #
    # κ_n = Σ v_i² · (2/r_i²) / |∇F|   for unit v

    # Gradient of implicit function at q0
    grad_F = 2.0 * q0 / r2                            # (N, 3)
    grad_F_norm = np.maximum(np.linalg.norm(grad_F, axis=1), eps)

    # Direction from q0 to p (= radial direction in original space)
    v = p - q0                                         # (N, 3)
    v_norm = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), eps)
    v_hat = v / v_norm                                 # unit direction

    # Hessian of F is diagonal: H_ii = 2/r_i²
    # v^T H v = Σ 2 v_hat_i² / r_i²
    vHv = np.sum(2.0 * v_hat ** 2 / r2, axis=1)

    # Normal curvature (magnitude; always positive for convex surface
    # when measured from outside)
    kappa_n = vHv / grad_F_norm

    # Correction: φ_corr = φ_Q / (1 + κ_n · |φ_Q|)
    # Use |φ_Q| to ensure correction is always a reduction in magnitude
    # but preserve sign for interior points
    sign = np.sign(phi_q)
    abs_phi = np.abs(phi_q)
    phi_corr = abs_phi / (1.0 + kappa_n * abs_phi)
    return sign * phi_corr


# ── Idee 4: Iterative Verfeinerung (1 Newton-Schritt) ───────────────────
def sdf_newton_step(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """One Newton-like refinement step.

    1. Estimate surface point q₀ = p / k₀  (radial projection)
    2. Compute outward normal n₀ at q₀
    3. Intersect ray from p in direction −n₀ with ellipsoid → q₁
    4. Return ||p − q₁|| as the improved distance estimate
    """
    r = radii.astype(np.float64)
    r2 = r * r
    p = points.astype(np.float64)
    eps = 1e-15

    inside = np.sum((p / r) ** 2, axis=1) < 1.0

    # Step 1: radial projection q₀ = p / k₀
    k0 = np.maximum(np.linalg.norm(p / r, axis=1), eps)
    q0 = p / k0[:, None]

    # Step 2: outward normal at q₀ (gradient of implicit F = Σ x²/r² - 1)
    n0 = q0 / r2                                       # (N, 3)
    n0_norm = np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), eps)
    n0_hat = n0 / n0_norm                               # unit normal

    # Step 3: intersect ray  x(t) = p - t·n₀_hat  with ellipsoid
    #   Σ (p_i - t·n_i)² / r_i² = 1
    #   A t² - 2B t + C = 0
    A = np.sum(n0_hat ** 2 / r2, axis=1)
    B = np.sum(p * n0_hat / r2, axis=1)
    C = np.sum(p ** 2 / r2, axis=1)  # = k₀² before clamping

    disc = B ** 2 - A * (C - 1.0)
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)
    A_safe = np.maximum(A, eps)

    # Pick the root closest to the surface along the ray x(t) = p - t·n₀_hat:
    # For exterior (C > 1): both roots positive, want smaller → (B - √disc) / A
    # For interior (C < 1): disc > B², so t1 = (B - √disc)/A < 0.
    #   Negative t means q1 = p + |t|·n_hat → goes outward to nearest surface point.
    # In both cases, the (B - √disc) root gives the nearest intersection.
    t = (B - sqrt_disc) / A_safe

    q1 = p - t[:, None] * n0_hat

    # Step 4: distance
    dist = np.linalg.norm(p - q1, axis=1)
    return np.where(inside, -dist, dist)


# Method registry: (display_name, function, short_name_for_plots)
METHODS = [
    ("Quílez",                sdf_quilez,        "Quílez"),
    ("Scaled (min r)",        sdf_scaled_min,    "Sc-min"),
    ("Scaled (mean r)",       sdf_scaled_mean,   "Sc-mean"),
    ("Scaled (geom. mean r)", sdf_scaled_gmean,  "Sc-gmean"),
    # ("Gradient-Proj.",        sdf_gradient_proj,  "Grad-P"),
    # ("k₁/k₂",                sdf_k1k2,          "k₁/k₂"),
    # ("Krümmungskorr.",        sdf_curvature_corr, "Krümm"),
    # ("Newton-Schritt",        sdf_newton_step,    "Newton"),
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
        self.resize(2200, 1200)
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
        self._fig = Figure(figsize=(24, 13), dpi=100, facecolor="#0d1117")
        self._canvas = FigureCanvas(self._fig)
        root.addWidget(self._canvas, stretch=4)

        # ── Metrics table ─────────────────────────────────────────────
        self._table = QtWidgets.QTableWidget()
        self._table.setMaximumHeight(280)
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

        ax_bar = self._fig.add_subplot(gs[2, 0:3])
        self._draw_metric_bars(ax_bar, res, text_color)

        ax_rad = self._fig.add_subplot(gs[2, 3:6])
        self._draw_radial_profile(ax_rad, res, text_color)

        ax_time = self._fig.add_subplot(gs[2, 6:n_cols])
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
        w = 0.22
        colors = ["#f2e641", "#4962f2", "#f26450"]
        ax.bar(x - w, mae, w, label="MAE", color=colors[0], edgecolor="none")
        ax.bar(x, rmse, w, label="RMSE", color=colors[1], edgecolor="none")
        ax.bar(x + w, linf, w, label="L∞", color=colors[2], edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, color=tc, rotation=35, ha="right")
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
        method_colors = ["#f2e641", "#4962f2", "#50c878", "#c878ff",
                         "#ff9f43", "#1abc9c", "#e74c3c", "#3498db"]
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
        ax.legend(handles=handles, fontsize=5, loc="upper right",
                  facecolor="#1a2030", edgecolor="#333", labelcolor=tc, ncol=3)
        ax.tick_params(colors=tc, labelsize=7)
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#333")

    def _draw_timing_bars(self, ax, res, tc):
        """Timing bar chart."""
        timing = res["timing"]
        names = list(timing.keys())
        values = list(timing.values())
        short_names = ["Exakt"] + [s for _, _, s in METHODS]

        colors = ["#888"] + ["#f2e641", "#4962f2", "#50c878", "#c878ff",
                              "#ff9f43", "#1abc9c", "#e74c3c", "#3498db"]
        bars = ax.barh(range(len(names)), values, color=colors[:len(names)],
                       edgecolor="none", height=0.55)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=7, color=tc)
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