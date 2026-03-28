"""
sdf_methods.py — Central registry of ellipsoid SDF approximation methods.

Single source of truth for both NumPy (CPU, benchmarking) and Warp (GPU,
optimization & display) implementations.

Adding a new method
───────────────────
  1. Write the NumPy function:  sdf_yourmethod(points, radii) -> ndarray
  2. Write the Warp @wp.func:  _sdf_yourmethod_wp(local_p, r) -> float
  3. Add a branch in ``sdf_ellipsoid_wp`` with the next METHOD_* constant.
  4. Append a ``SdfMethodInfo`` to ``SDF_METHODS``.
  Done — benchmark, optimisation and UI will pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import warp as wp


# ══════════════════════════════════════════════════════════════════════════════
# METHOD IDs  (used as int constants inside Warp kernels)
# ══════════════════════════════════════════════════════════════════════════════

METHOD_QUILEZ      = 0
METHOD_SCALED_MIN  = 1
METHOD_SCALED_MEAN = 2
METHOD_SCALED_GMEAN = 3
METHOD_MERTSTEIN   = 4


# ══════════════════════════════════════════════════════════════════════════════
# WARP DEVICE FUNCTIONS  (@wp.func — callable from any @wp.kernel)
# ══════════════════════════════════════════════════════════════════════════════

@wp.func
def _sdf_quilez_wp(local_p: wp.vec3, r: wp.vec3) -> float:
    """Ínigo Quílez approximation:  k0·(k0 − 1) / k1."""
    scaled = wp.vec3(
        local_p[0] / r[0],
        local_p[1] / r[1],
        local_p[2] / r[2],
    )
    scaled2 = wp.vec3(
        local_p[0] / (r[0] * r[0]),
        local_p[1] / (r[1] * r[1]),
        local_p[2] / (r[2] * r[2]),
    )
    k0 = wp.length(scaled)
    k1 = wp.max(wp.length(scaled2), 1.0e-8)
    return k0 * (k0 - 1.0) / k1


@wp.func
def _sdf_scaled_min_wp(local_p: wp.vec3, r: wp.vec3) -> float:
    """Scaled-sphere with min(r)."""
    scaled = wp.vec3(
        local_p[0] / r[0],
        local_p[1] / r[1],
        local_p[2] / r[2],
    )
    k = wp.length(scaled)
    min_r = wp.min(wp.min(r[0], r[1]), r[2])
    return (k - 1.0) * min_r


@wp.func
def _sdf_scaled_mean_wp(local_p: wp.vec3, r: wp.vec3) -> float:
    """Scaled-sphere with mean(r)."""
    scaled = wp.vec3(
        local_p[0] / r[0],
        local_p[1] / r[1],
        local_p[2] / r[2],
    )
    k = wp.length(scaled)
    mean_r = (r[0] + r[1] + r[2]) / 3.0
    return (k - 1.0) * mean_r


@wp.func
def _sdf_scaled_gmean_wp(local_p: wp.vec3, r: wp.vec3) -> float:
    """Scaled-sphere with geometric mean ∛(r₁r₂r₃)."""
    scaled = wp.vec3(
        local_p[0] / r[0],
        local_p[1] / r[1],
        local_p[2] / r[2],
    )
    k = wp.length(scaled)
    gmean_r = wp.pow(r[0] * r[1] * r[2], 1.0 / 3.0)
    return (k - 1.0) * gmean_r


@wp.func
def _sdf_mertstein_wp(local_p: wp.vec3, r: wp.vec3) -> float:
    """MertStein hybrid: Quílez outside, Scaled-min inside."""
    scaled = wp.vec3(
        local_p[0] / r[0],
        local_p[1] / r[1],
        local_p[2] / r[2],
    )
    k0 = wp.length(scaled)
    d = float(0.0)
    if k0 < 1.0:
        # Interior → Scaled-Sphere (min r)
        min_r = wp.min(wp.min(r[0], r[1]), r[2])
        d = (k0 - 1.0) * min_r
    else:
        # Exterior → Quílez
        scaled2 = wp.vec3(
            local_p[0] / (r[0] * r[0]),
            local_p[1] / (r[1] * r[1]),
            local_p[2] / (r[2] * r[2]),
        )
        k1 = wp.max(wp.length(scaled2), 1.0e-8)
        d = k0 * (k0 - 1.0) / k1
    return d


@wp.func
def sdf_ellipsoid_wp(local_p: wp.vec3, r: wp.vec3, method_id: int) -> float:
    """Dispatch to the correct SDF approximation on the device.

    This is the **single entry point** called from all Warp kernels.
    """
    d = float(1.0e6)
    if method_id == 0:
        d = _sdf_quilez_wp(local_p, r)
    elif method_id == 1:
        d = _sdf_scaled_min_wp(local_p, r)
    elif method_id == 2:
        d = _sdf_scaled_mean_wp(local_p, r)
    elif method_id == 3:
        d = _sdf_scaled_gmean_wp(local_p, r)
    elif method_id == 4:
        d = _sdf_mertstein_wp(local_p, r)
    else:
        d = _sdf_quilez_wp(local_p, r)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# NUMPY IMPLEMENTATIONS  (CPU, float64 — for benchmarking / analysis)
# ══════════════════════════════════════════════════════════════════════════════

def sdf_exact_np(points: np.ndarray, radii: np.ndarray,
                 n_bisect: int = 80) -> np.ndarray:
    """Ground-truth signed distance via bisection on the Lagrange multiplier.

    Negative inside, positive outside.
    """
    r = radii.astype(np.float64)
    r2 = r ** 2
    pts = points.astype(np.float64)

    inside = np.sum((pts / r) ** 2, axis=1) < 1.0

    eps = 1e-10 * np.min(r)
    p = np.maximum(np.abs(pts), eps)

    N = len(p)
    t_lo = np.empty(N, dtype=np.float64)
    t_hi = np.empty(N, dtype=np.float64)

    T_max = np.max(r) * np.linalg.norm(p, axis=1) + np.sum(r2)
    t_lo[~inside] = 0.0
    t_hi[~inside] = T_max[~inside]

    T_min = -np.min(r2) * (1.0 - 1e-15)
    t_lo[inside] = T_min
    t_hi[inside] = 0.0

    for _ in range(n_bisect):
        t_mid = 0.5 * (t_lo + t_hi)
        denom = r2 + t_mid[:, None]
        F = np.sum((r * p / denom) ** 2, axis=1) - 1.0
        move_lo = F > 0
        t_lo = np.where(move_lo, t_mid, t_lo)
        t_hi = np.where(~move_lo, t_mid, t_hi)

    t = 0.5 * (t_lo + t_hi)
    q = r2 * p / (r2 + t[:, None])
    dist = np.linalg.norm(p - q, axis=1)
    return np.where(inside, -dist, dist)


def sdf_quilez_np(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Ínigo Quílez approximation:  k0·(k0 − 1) / k1."""
    r = radii.astype(np.float64)
    p = points.astype(np.float64)
    scaled = p / r
    scaled2 = p / (r * r)
    k0 = np.linalg.norm(scaled, axis=1)
    k1 = np.maximum(np.linalg.norm(scaled2, axis=1), 1e-15)
    return k0 * (k0 - 1.0) / k1


def sdf_scaled_min_np(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · min(r)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.min(r)


def sdf_scaled_mean_np(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · mean(r)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.mean(r)


def sdf_scaled_gmean_np(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Scaled-sphere:  (|p/r| − 1) · ∛(r₁r₂r₃)."""
    r = radii.astype(np.float64)
    k = np.linalg.norm(points.astype(np.float64) / r, axis=1)
    return (k - 1.0) * np.cbrt(np.prod(r))


def sdf_mertstein_np(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """MertStein hybrid: Quílez outside, Scaled-Sphere (min r) inside."""
    r = radii.astype(np.float64)
    p = points.astype(np.float64)

    scaled = p / r
    k0 = np.linalg.norm(scaled, axis=1)
    inside = k0 < 1.0

    result = np.empty(len(p), dtype=np.float64)

    if np.any(~inside):
        scaled2 = p[~inside] / (r * r)
        k0_ext = k0[~inside]
        k1_ext = np.maximum(np.linalg.norm(scaled2, axis=1), 1e-15)
        result[~inside] = k0_ext * (k0_ext - 1.0) / k1_ext

    if np.any(inside):
        result[inside] = (k0[inside] - 1.0) * np.min(r)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SdfMethodInfo:
    """Describes one SDF approximation method."""
    name: str                   # Display name (e.g. "Quílez")
    short: str                  # Short label for plots (e.g. "Quílez")
    warp_id: int                # Integer constant passed to Warp kernels
    numpy_fn: Callable          # (points (N,3), radii (3,)) -> (N,) float64
    description: str = ""       # Optional human-readable formula/notes


SDF_METHODS: List[SdfMethodInfo] = [
    SdfMethodInfo(
        name="Quílez",
        short="Quílez",
        warp_id=METHOD_QUILEZ,
        numpy_fn=sdf_quilez_np,
        description="k0·(k0−1) / k1",
    ),
    SdfMethodInfo(
        name="Scaled (min r)",
        short="Sc-min",
        warp_id=METHOD_SCALED_MIN,
        numpy_fn=sdf_scaled_min_np,
        description="(|p/r|−1) · min(r)",
    ),
    SdfMethodInfo(
        name="Scaled (mean r)",
        short="Sc-mean",
        warp_id=METHOD_SCALED_MEAN,
        numpy_fn=sdf_scaled_mean_np,
        description="(|p/r|−1) · mean(r)",
    ),
    SdfMethodInfo(
        name="Scaled (geom. mean r)",
        short="Sc-gmean",
        warp_id=METHOD_SCALED_GMEAN,
        numpy_fn=sdf_scaled_gmean_np,
        description="(|p/r|−1) · ∛(r₁r₂r₃)",
    ),
    SdfMethodInfo(
        name="MertStein",
        short="MertSt",
        warp_id=METHOD_MERTSTEIN,
        numpy_fn=sdf_mertstein_np,
        description="Quílez (außen) + Sc-min (innen)",
    ),
]


# ── Convenience lookups ───────────────────────────────────────────────────────

def get_method_names() -> List[str]:
    """Return list of all registered method display names."""
    return [m.name for m in SDF_METHODS]


def get_method_by_name(name: str) -> SdfMethodInfo:
    """Look up a method by its display name.  Raises KeyError if not found."""
    for m in SDF_METHODS:
        if m.name == name:
            return m
    raise KeyError(f"Unknown SDF method: '{name}'.  "
                   f"Available: {get_method_names()}")


def get_method_by_warp_id(warp_id: int) -> SdfMethodInfo:
    """Look up a method by its Warp integer ID."""
    for m in SDF_METHODS:
        if m.warp_id == warp_id:
            return m
    raise KeyError(f"Unknown Warp method ID: {warp_id}")


def get_default_method() -> SdfMethodInfo:
    """Return the default method (Quílez)."""
    return SDF_METHODS[0]
