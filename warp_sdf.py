"""
warp_sdf.py -- Warp-GPU-Implementierungen der Ellipsoid-SDF-Methoden.

Parallele GPU-Varianten der analytischen Naeherungsformeln aus benchmark_sdf.py.
Alle Kernels arbeiten auf einem flachen Punkt-Array (beliebige 3-D-Punkte,
kein Voxelgitter) und werten einen einzelnen achsen-ausgerichteten Ellipsoid
am Ursprung aus.

Schnittstelle:
    evaluator = WarpSdfEvaluator(device="cuda:0")
    sdf = evaluator.quilez(points, radii)      # (N,3) float64 -> (N,) float64
"""

import numpy as np
import warp as wp


# ── Kernels ───────────────────────────────────────────────────────────────────
# Jeder Kernel: ein Thread pro Punkt.
# p in:  points[tid]       (wp.vec3, bereits im lokalen Ellipsoid-Rahmen)
# r:     wp.vec3           (Halbachsen)
# out:   wp.float32-Array  (SDF-Wert, float32)

@wp.kernel
def _quilez_kernel(
    points: wp.array(dtype=wp.vec3),
    r: wp.vec3,
    out: wp.array(dtype=wp.float32),
):
    """k0·(k0−1) / k1"""
    tid = wp.tid()
    p = points[tid]
    scaled  = wp.vec3(p[0] / r[0],          p[1] / r[1],          p[2] / r[2])
    scaled2 = wp.vec3(p[0] / (r[0] * r[0]), p[1] / (r[1] * r[1]), p[2] / (r[2] * r[2]))
    k0 = wp.length(scaled)
    k1 = wp.max(wp.length(scaled2), float(1.0e-15))
    out[tid] = k0 * (k0 - float(1.0)) / k1


@wp.kernel
def _scaled_min_kernel(
    points: wp.array(dtype=wp.vec3),
    r: wp.vec3,
    out: wp.array(dtype=wp.float32),
):
    """(|p/r| − 1) · min(r)"""
    tid = wp.tid()
    p = points[tid]
    scaled = wp.vec3(p[0] / r[0], p[1] / r[1], p[2] / r[2])
    k0 = wp.length(scaled)
    r_min = wp.min(wp.min(r[0], r[1]), r[2])
    out[tid] = (k0 - float(1.0)) * r_min


@wp.kernel
def _scaled_mean_kernel(
    points: wp.array(dtype=wp.vec3),
    r: wp.vec3,
    out: wp.array(dtype=wp.float32),
):
    """(|p/r| − 1) · mean(r)"""
    tid = wp.tid()
    p = points[tid]
    scaled = wp.vec3(p[0] / r[0], p[1] / r[1], p[2] / r[2])
    k0 = wp.length(scaled)
    r_mean = (r[0] + r[1] + r[2]) / float(3.0)
    out[tid] = (k0 - float(1.0)) * r_mean


@wp.kernel
def _scaled_gmean_kernel(
    points: wp.array(dtype=wp.vec3),
    r: wp.vec3,
    out: wp.array(dtype=wp.float32),
):
    """(|p/r| − 1) · ∛(r₀r₁r₂)"""
    tid = wp.tid()
    p = points[tid]
    scaled = wp.vec3(p[0] / r[0], p[1] / r[1], p[2] / r[2])
    k0 = wp.length(scaled)
    # Kubikwurzel via exp(log(x)/3) — sicher fuer positive Halbachsen
    r_gmean = wp.exp(wp.log(r[0] * r[1] * r[2]) / float(3.0))
    out[tid] = (k0 - float(1.0)) * r_gmean


# Kernel-Registry – Schluessel entsprechen benchmark_sdf.METHODS-Namen
KERNELS: dict = {
    "Quílez":                _quilez_kernel,
    "Scaled (min r)":        _scaled_min_kernel,
    "Scaled (mean r)":       _scaled_mean_kernel,
    "Scaled (geom. mean r)": _scaled_gmean_kernel,
}


# ── Evaluator ─────────────────────────────────────────────────────────────────

class WarpSdfEvaluator:
    """
    Verwaltet Warp-Device-Arrays und bietet bequeme Methoden fuer
    die GPU-parallele Auswertung der Ellipsoid-SDF-Naeherungen.

    Punkte werden einmalig hochgeladen (upload_points) und koennen
    danach fuer mehrere Kernel wiederverwendet werden — nuetzlich
    fuer das Timing, wo keine Datentransfer-Kosten gemessen werden.

    Benutzung::

        ev = WarpSdfEvaluator("cuda:0")
        sdf = ev.quilez(points, radii)           # alles in einem Schritt
        # --- oder fuer Timing: ---
        ev.upload_points(pts3d)
        ev.run_kernel(KERNELS["Quílez"], radii)  # nur Kernel, kein Upload
        wp.synchronize()
    """

    def __init__(self, device: str = "cuda:0"):
        self.device    = device
        self._wp_pts   = None   # wp.array(vec3) auf dem Device
        self._wp_out   = None   # wp.array(float32) Ausgabe
        self._n        = 0

    def upload_points(self, points: np.ndarray) -> None:
        """Lade Punkte auf das GPU-Device (float64 -> float32)."""
        pts32 = points.astype(np.float32)
        n = len(pts32)
        self._wp_pts = wp.array(pts32, dtype=wp.vec3, device=self.device)
        if self._wp_out is None or self._n != n:
            self._wp_out = wp.empty(n, dtype=wp.float32, device=self.device)
        self._n = n

    def run_kernel(self, kernel, radii: np.ndarray) -> np.ndarray:
        """
        Starte einen Kernel auf bereits hochgeladenen Punkten.
        Gibt (N,) float64 zurueck (inkl. GPU->CPU-Transfer).
        Kein Upload in dieser Methode.
        """
        r = wp.vec3(float(radii[0]), float(radii[1]), float(radii[2]))
        wp.launch(
            kernel=kernel,
            dim=self._n,
            inputs=[self._wp_pts, r, self._wp_out],
            device=self.device,
        )
        return self._wp_out.numpy().astype(np.float64)

    def _eval(self, kernel, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Upload + Kernel + Download in einem Schritt."""
        self.upload_points(points)
        return self.run_kernel(kernel, radii)

    # ── Bequeme Einzel-Methoden ───────────────────────────────────────────

    def quilez(self, points, radii):
        return self._eval(_quilez_kernel, points, radii)

    def scaled_min(self, points, radii):
        return self._eval(_scaled_min_kernel, points, radii)

    def scaled_mean(self, points, radii):
        return self._eval(_scaled_mean_kernel, points, radii)

    def scaled_gmean(self, points, radii):
        return self._eval(_scaled_gmean_kernel, points, radii)

    def warmup(self, radii: np.ndarray) -> None:
        """
        Einmalige Warmlauf-Auswertung aller Kernels (triggert Warp-JIT).
        Sollte vor dem Timing aufgerufen werden.
        """
        dummy = np.zeros((1, 3), dtype=np.float32)
        self.upload_points(dummy)
        for kernel in KERNELS.values():
            self.run_kernel(kernel, radii)
        wp.synchronize()
