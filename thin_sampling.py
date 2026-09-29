"""Shared helpers for continuous thickness-weighted surface sampling."""

from __future__ import annotations

import numpy as np


_MIN_RELATIVE_THICKNESS = 1.0e-6


def positive_weighted_median(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> float | None:
    """Return the median of finite positive values, optionally area-weighted."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(flat) & (flat > 0.0)
    if weights is None:
        positive = flat[valid]
        return float(np.median(positive)) if positive.size else None

    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if mass.size != flat.size:
        raise ValueError("thickness/weight size mismatch")
    valid &= np.isfinite(mass) & (mass > 0.0)
    if not np.any(valid):
        return None
    ordered = np.argsort(flat[valid], kind="stable")
    ordered_values = flat[valid][ordered]
    ordered_mass = mass[valid][ordered]
    midpoint = 0.5 * float(np.sum(ordered_mass, dtype=np.float64))
    position = int(np.searchsorted(
        np.cumsum(ordered_mass, dtype=np.float64), midpoint, side="left"))
    return float(ordered_values[min(position, ordered_values.size - 1)])


def inverse_thickness_factors(
    thickness: np.ndarray,
    power: float,
    *,
    reference: float | None = None,
) -> np.ndarray:
    """Return continuous, scale-free ``(reference / thickness) ** power``.

    Positive finite thickness values are weighted continuously.  Unresolved
    values (zero, negative, NaN, or infinity) receive the neutral factor 1 so
    they never explode into ``1 / 0`` and are not silently removed.  A tiny
    relative floor only protects against numerical noise; it is far below the
    resolution of a meaningful thickness field.
    """
    values = np.asarray(thickness, dtype=np.float64)
    exponent = float(power)
    if not np.isfinite(exponent) or exponent < 0.0:
        raise ValueError(
            "thickness sampling power must be finite and non-negative")
    factors = np.ones(values.shape, dtype=np.float64)
    if exponent == 0.0:
        return factors

    valid = np.isfinite(values) & (values > 0.0)
    if not np.any(valid):
        return factors
    if reference is None:
        reference = float(np.median(values[valid]))
    reference = float(reference)
    if not np.isfinite(reference) or reference <= 0.0:
        return factors

    floor = max(
        reference * _MIN_RELATIVE_THICKNESS,
        float(np.finfo(np.float64).tiny),
    )
    # Evaluate in log space and clip only at the floating-point safety limit.
    # The common reference factor makes the result invariant to mesh units.
    log_factor = exponent * (
        np.log(reference) - np.log(np.maximum(values[valid], floor)))
    factors[valid] = np.exp(np.clip(log_factor, -700.0, 700.0))
    return factors


def thickness_sampling_probabilities(
    thickness: np.ndarray,
    power: float,
    *,
    base_weights: np.ndarray | None = None,
    reference: float | None = None,
) -> np.ndarray:
    """Normalize continuous inverse-thickness sampling probabilities.

    ``base_weights`` is uniform for an already enumerated surface band and is
    triangle area for mesh-surface generation.  Consequently ``power == 1``
    produces ``1 / thickness`` and ``area / thickness`` respectively.
    """
    values = np.asarray(thickness, dtype=np.float64).reshape(-1)
    if base_weights is None:
        base = np.ones(values.size, dtype=np.float64)
    else:
        base = np.asarray(base_weights, dtype=np.float64).reshape(-1)
        if base.size != values.size:
            raise ValueError("thickness/base-weight size mismatch")
        base = np.where(np.isfinite(base) & (base > 0.0), base, 0.0)
    base_sum = float(base.sum(dtype=np.float64))
    if not np.isfinite(base_sum) or base_sum <= 0.0:
        if values.size == 0:
            return np.empty(0, dtype=np.float64)
        base = np.ones(values.size, dtype=np.float64)
        base_sum = float(values.size)

    exponent = float(power)
    if not np.isfinite(exponent) or exponent < 0.0:
        raise ValueError(
            "thickness sampling power must be finite and non-negative")
    valid = np.isfinite(values) & (values > 0.0)
    if exponent == 0.0 or not np.any(valid):
        return base / base_sum

    if reference is None:
        reference = positive_weighted_median(values, base)
    factors = inverse_thickness_factors(
        values, exponent, reference=reference)
    mass = base * factors
    mass_sum = float(mass.sum(dtype=np.float64))
    if not np.isfinite(mass_sum) or mass_sum <= 0.0:
        return base / base_sum
    return mass / mass_sum
