"""Deterministic validation support for long-running SDF fits.

Training batches are deliberately stochastic.  This module provides the
opposite: a small, reproducible target set, a NumPy implementation of the fit
loss, and loss-based checkpoint/patience tracking.  It has no Warp or Qt
dependency, so it can also be used by offline tools and regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from sdf_samples import SdfSampleSet


__all__ = [
    "STRATUM_SURFACE",
    "STRATUM_INSIDE",
    "STRATUM_OUTSIDE",
    "STRATUM_COARSE_FAR",
    "ValidationSample",
    "ValidationLoss",
    "stratified_validation_from_grid",
    "stratified_validation_from_samples",
    "evaluate_validation_loss",
    "Patience",
    "BestCheckpoint",
]


# Stable public stratum codes.  Sparse coarse/far samples get their own quota so
# that the weak far-field loss remains observable in every validation run.
STRATUM_SURFACE = np.uint8(0)
STRATUM_INSIDE = np.uint8(1)
STRATUM_OUTSIDE = np.uint8(2)
STRATUM_COARSE_FAR = np.uint8(3)


@dataclass(frozen=True)
class ValidationSample:
    """A deterministic subset of an SDF target.

    ``source_indices`` always address the flattened source grid/sample set.
    ``thickness_reference`` is computed from the complete source rather than
    the subset, matching the production fitter's inverse-thickness weighting.
    """

    points: np.ndarray
    values: np.ndarray
    source_indices: np.ndarray
    strata: np.ndarray
    dx: float
    thickness: np.ndarray | None = None
    thickness_reference: float | None = None
    coarse_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.points, dtype=np.float32).reshape(-1, 3)
        values = np.ascontiguousarray(self.values, dtype=np.float32).reshape(-1)
        indices = np.ascontiguousarray(self.source_indices, dtype=np.int64).reshape(-1)
        strata = np.ascontiguousarray(self.strata, dtype=np.uint8).reshape(-1)
        size = values.size
        if points.shape[0] != size or indices.size != size or strata.size != size:
            raise ValueError("validation sample arrays must have the same length")
        if size == 0:
            raise ValueError("validation sample must not be empty")
        if not np.isfinite(points).all() or not np.isfinite(values).all():
            raise ValueError("validation points and values must be finite")
        if not np.isfinite(self.dx) or float(self.dx) <= 0.0:
            raise ValueError("dx must be finite and positive")

        thickness = self.thickness
        if thickness is not None:
            thickness = np.ascontiguousarray(thickness, dtype=np.float32).reshape(-1)
            if thickness.size != size:
                raise ValueError("validation thickness/value count mismatch")
            if not np.isfinite(thickness).all() or np.any(thickness < 0.0):
                raise ValueError("validation thickness must be finite and non-negative")

        coarse = self.coarse_mask
        if coarse is not None:
            coarse = np.ascontiguousarray(coarse, dtype=np.bool_).reshape(-1)
            if coarse.size != size:
                raise ValueError("validation coarse-mask/value count mismatch")

        reference = self.thickness_reference
        if reference is not None:
            reference = float(reference)
            if not np.isfinite(reference) or reference <= 0.0:
                raise ValueError("thickness_reference must be finite and positive")

        # The dataclass is frozen, but normalising caller-owned arrays here keeps
        # its public representation compact and independent from later mutations.
        object.__setattr__(self, "points", points.copy())
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "source_indices", indices.copy())
        object.__setattr__(self, "strata", strata.copy())
        object.__setattr__(self, "dx", float(self.dx))
        object.__setattr__(self, "thickness", None if thickness is None else thickness.copy())
        object.__setattr__(self, "coarse_mask", None if coarse is None else coarse.copy())
        object.__setattr__(self, "thickness_reference", reference)

    @property
    def size(self) -> int:
        return int(self.values.size)


def _positive_thickness_median(thickness: np.ndarray | None) -> float | None:
    if thickness is None:
        return None
    flat = np.asarray(thickness, dtype=np.float32).reshape(-1)
    if not np.isfinite(flat).all() or np.any(flat < 0.0):
        raise ValueError("thickness must be finite and non-negative")
    positive = flat[flat > 0.0]
    return float(np.median(positive)) if positive.size else None


def _validate_sampling_args(
    values: np.ndarray,
    sample_count: int,
    surface_band: float,
    surface_fraction: float,
    coarse_fraction: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("SDF target must not be empty")
    if not np.isfinite(flat).all():
        raise ValueError("SDF target must be finite")
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")
    if not np.isfinite(surface_band) or float(surface_band) < 0.0:
        raise ValueError("surface_band must be finite and non-negative")
    if not 0.0 <= float(surface_fraction) <= 1.0:
        raise ValueError("surface_fraction must lie in [0, 1]")
    if not 0.0 <= float(coarse_fraction) <= 1.0:
        raise ValueError("coarse_fraction must lie in [0, 1]")
    if float(surface_fraction) + float(coarse_fraction) > 1.0 + 1.0e-12:
        raise ValueError("surface_fraction + coarse_fraction must not exceed 1")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    return flat, min(count, flat.size)


def _bounded_apportion(capacities: np.ndarray, weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate ``total`` seats proportionally, respecting finite pool sizes."""
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    allocation = np.zeros_like(capacities)
    for _ in range(int(total)):
        active = allocation < capacities
        if not np.any(active):
            break
        scores = np.full(weights.shape, -np.inf, dtype=np.float64)
        scores[active] = weights[active] / (allocation[active] + 1.0)
        # Zero-weight pools are only used to absorb capacity left behind by an
        # exhausted requested stratum.
        if float(np.max(scores)) <= 0.0:
            spare = capacities - allocation
            choice = int(np.argmax(np.where(active, spare, -1)))
        else:
            choice = int(np.argmax(scores))
        allocation[choice] += 1
    return allocation


def _stratified_indices(
    values: np.ndarray,
    *,
    sample_count: int,
    surface_band: float,
    surface_fraction: float,
    seed: int,
    coarse_mask: np.ndarray | None,
    coarse_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    flat, count = _validate_sampling_args(
        values, sample_count, surface_band, surface_fraction, coarse_fraction, seed)
    surface = np.abs(flat) <= float(surface_band)

    if coarse_mask is None:
        coarse_far = np.zeros(flat.size, dtype=np.bool_)
    else:
        coarse = np.asarray(coarse_mask, dtype=np.bool_).reshape(-1)
        if coarse.size != flat.size:
            raise ValueError("coarse_mask/value count mismatch")
        coarse_far = coarse & ~surface

    pools = (
        np.flatnonzero(surface),
        np.flatnonzero((flat < -float(surface_band)) & ~coarse_far),
        np.flatnonzero((flat > float(surface_band)) & ~coarse_far),
        np.flatnonzero(coarse_far),
    )
    remainder = max(0.0, 1.0 - float(surface_fraction) -
                    (float(coarse_fraction) if pools[3].size else 0.0))
    weights = np.array([
        float(surface_fraction),
        0.5 * remainder,
        0.5 * remainder,
        float(coarse_fraction) if pools[3].size else 0.0,
    ], dtype=np.float64)
    # If requested fractions point only at empty strata, distribute the sample
    # over all populated pools instead of silently returning too few points.
    populated = np.array([pool.size > 0 for pool in pools], dtype=np.bool_)
    weights[~populated] = 0.0
    if not np.any(weights > 0.0):
        weights[populated] = 1.0

    allocation = _bounded_apportion(
        np.array([pool.size for pool in pools], dtype=np.int64), weights, count)
    selected_parts: list[np.ndarray] = []
    stratum_parts: list[np.ndarray] = []
    for code, (pool, take) in enumerate(zip(pools, allocation, strict=True)):
        if take <= 0:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), code, 0x51DF]))
        chosen = rng.choice(pool, size=int(take), replace=False).astype(np.int64)
        selected_parts.append(chosen)
        stratum_parts.append(np.full(int(take), code, dtype=np.uint8))

    selected = np.concatenate(selected_parts)
    strata = np.concatenate(stratum_parts)
    # Do not expose stratum-grouped order to downstream prediction code.
    order_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0xA11CE, len(selected)]))
    order = order_rng.permutation(len(selected))
    return np.ascontiguousarray(selected[order]), np.ascontiguousarray(strata[order])


def stratified_validation_from_grid(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    *,
    thickness: np.ndarray | None = None,
    sample_count: int = 4096,
    surface_band: float | None = None,
    surface_fraction: float = 0.5,
    seed: int = 0,
) -> ValidationSample:
    """Build a reproducible surface/inside/outside sample from a dense grid.

    Grid storage follows the project convention ``(z, y, x)`` and points are
    voxel centres.  The default validation band is three voxels wide.
    """
    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("grid must have shape (nz, ny, nx)")
    spacing = float(dx)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("dx must be finite and positive")
    base = np.asarray(origin, dtype=np.float32).reshape(-1)
    if base.size != 3 or not np.isfinite(base).all():
        raise ValueError("origin must contain three finite coordinates")
    thick = None if thickness is None else np.asarray(thickness, dtype=np.float32)
    if thick is not None and thick.shape != values.shape:
        raise ValueError("thickness must have the same shape as grid")

    band = 3.0 * spacing if surface_band is None else float(surface_band)
    indices, strata = _stratified_indices(
        values,
        sample_count=sample_count,
        surface_band=band,
        surface_fraction=surface_fraction,
        seed=seed,
        coarse_mask=None,
        coarse_fraction=0.0,
    )
    _, ny, nx = values.shape
    iz, rem = np.divmod(indices, ny * nx)
    iy, ix = np.divmod(rem, nx)
    points = np.column_stack((
        base[0] + (ix.astype(np.float32) + 0.5) * spacing,
        base[1] + (iy.astype(np.float32) + 0.5) * spacing,
        base[2] + (iz.astype(np.float32) + 0.5) * spacing,
    )).astype(np.float32)
    flat_thick = None if thick is None else thick.reshape(-1)
    return ValidationSample(
        points=points,
        values=values.reshape(-1)[indices],
        source_indices=indices,
        strata=strata,
        dx=spacing,
        thickness=None if flat_thick is None else flat_thick[indices],
        thickness_reference=_positive_thickness_median(flat_thick),
    )


def stratified_validation_from_samples(
    samples: SdfSampleSet,
    *,
    sample_count: int = 4096,
    surface_band: float | None = None,
    surface_fraction: float = 0.5,
    coarse_fraction: float = 0.2,
    seed: int = 0,
) -> ValidationSample:
    """Build a reproducible validation subset from ``SdfSampleSet``.

    If a coarse lattice mask is present, non-surface coarse samples receive a
    separate quota (20 % by default), mirroring sparse training's far-field
    coverage guarantee.
    """
    if not isinstance(samples, SdfSampleSet):
        raise TypeError("samples must be an SdfSampleSet")
    if not np.isfinite(samples.points).all():
        raise ValueError("sample points must be finite")
    spacing = float(samples.dx)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("samples.dx must be finite and positive")
    band = 3.0 * spacing if surface_band is None else float(surface_band)
    indices, strata = _stratified_indices(
        samples.values,
        sample_count=sample_count,
        surface_band=band,
        surface_fraction=surface_fraction,
        seed=seed,
        coarse_mask=samples.coarse_mask,
        coarse_fraction=coarse_fraction,
    )
    return ValidationSample(
        points=samples.points[indices],
        values=samples.values[indices],
        source_indices=indices,
        strata=strata,
        dx=spacing,
        thickness=None if samples.thickness is None else samples.thickness[indices],
        thickness_reference=_positive_thickness_median(samples.thickness),
        coarse_mask=(None if samples.coarse_mask is None
                     else samples.coarse_mask[indices]),
    )


@dataclass(frozen=True)
class ValidationLoss:
    """Mean loss and its additive, already-weighted components."""

    total: float
    reconstruction: float
    miss: float
    outside: float
    coarse_far_field: float
    mean_weight: float
    sample_count: int


def _huber_slope_one(error: np.ndarray, delta: float) -> np.ndarray:
    """Classic Huber loss, scaled to have unit slope outside the transition."""
    absolute = np.abs(error)
    return np.where(
        absolute < delta,
        0.5 * absolute * absolute / delta,
        absolute - 0.5 * delta,
    )


def evaluate_validation_loss(
    prediction: np.ndarray,
    sample: ValidationSample,
    *,
    huber_delta: float,
    clamp_limit: float = 0.1,
    miss_weight: float = 3.0,
    surface_weight: float = 4.0,
    surface_sigma: float | None = None,
    outside_weight: float = 14.0,
    thin_weight: float = 1.0,
    thin_max_factor: float = 6.0,
    thickness_reference: float | None = None,
    coarse_far_weight: float = 0.15,
    coarse_huber_delta: float | None = None,
) -> ValidationLoss:
    """Evaluate the production SDF loss using NumPy.

    The base term applies classic Huber to the difference of soft-clamped SDFs.
    ``huber_delta`` is intentionally mandatory; production normally passes
    ``0.5 * dx``.  Misses remain linear and protrusions quadratic for backwards
    compatibility.  Sparse coarse samples additionally use the production
    far-field Huber term when ``sample.coarse_mask`` is available.

    Non-finite predictions produce an infinite result, making them impossible
    to accept as a best checkpoint.
    """
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if pred.size != sample.size:
        raise ValueError("prediction/validation sample count mismatch")

    delta = float(huber_delta)
    limit = float(clamp_limit)
    sigma = 1.5 * sample.dx if surface_sigma is None else float(surface_sigma)
    coarse_delta = (max(4.0 * sample.dx, 0.02) if coarse_huber_delta is None
                    else float(coarse_huber_delta))
    scalar_values = {
        "huber_delta": delta,
        "clamp_limit": limit,
        "surface_sigma": sigma,
        "coarse_huber_delta": coarse_delta,
        "miss_weight": float(miss_weight),
        "surface_weight": float(surface_weight),
        "outside_weight": float(outside_weight),
        "thin_weight": float(thin_weight),
        "thin_max_factor": float(thin_max_factor),
        "coarse_far_weight": float(coarse_far_weight),
    }
    if not all(np.isfinite(value) for value in scalar_values.values()):
        raise ValueError("loss parameters must be finite")
    if delta <= 0.0 or limit <= 0.0 or sigma <= 0.0 or coarse_delta <= 0.0:
        raise ValueError("Huber deltas, clamp limit and surface sigma must be positive")
    if any(scalar_values[name] < 0.0 for name in (
            "miss_weight", "surface_weight", "outside_weight",
            "thin_weight", "coarse_far_weight")):
        raise ValueError("loss weights must be non-negative")
    if float(thin_max_factor) < 1.0:
        raise ValueError("thin_max_factor must be at least one")

    if not np.isfinite(pred).all():
        return ValidationLoss(
            total=float("inf"), reconstruction=float("inf"), miss=float("inf"),
            outside=float("inf"), coarse_far_field=float("inf"),
            mean_weight=float("inf"), sample_count=sample.size)

    target = sample.values.astype(np.float64, copy=False)
    pred_raw = np.clip(pred, -10.0, 10.0)
    weights = 1.0 + float(surface_weight) * np.exp(
        -(target * target) / (sigma * sigma))

    if sample.thickness is not None and float(thin_weight) > 0.0:
        reference = (sample.thickness_reference if thickness_reference is None
                     else float(thickness_reference))
        if reference is not None:
            if not np.isfinite(reference) or reference <= 0.0:
                raise ValueError("thickness_reference must be finite and positive")
            thickness = sample.thickness.astype(np.float64, copy=False)
            valid = thickness > 0.0
            boost = np.zeros_like(thickness)
            boost[valid] = np.maximum(reference / thickness[valid] - 1.0, 0.0)
            thin_factor = np.minimum(
                1.0 + float(thin_weight) * boost, float(thin_max_factor))
            weights *= thin_factor

    soft_pred = limit * np.tanh(pred_raw / limit)
    soft_target = limit * np.tanh(target / limit)
    reconstruction_values = weights * _huber_slope_one(
        soft_pred - soft_target, delta)

    miss_mask = (target < 0.0) & (pred_raw > 0.0)
    miss_values = np.zeros_like(target)
    miss_values[miss_mask] = (
        weights[miss_mask] * float(miss_weight)
        * (pred_raw[miss_mask] - target[miss_mask]))

    outside_mask = (target > 0.0) & (pred_raw < 0.0)
    outside_values = np.zeros_like(target)
    over = target[outside_mask] - pred_raw[outside_mask]
    outside_values[outside_mask] = (
        weights[outside_mask] * float(outside_weight) * over * over / sigma)

    coarse_values = np.zeros_like(target)
    if sample.coarse_mask is not None and float(coarse_far_weight) > 0.0:
        coarse = sample.coarse_mask
        coarse_error = (
            pred_raw[coarse] - np.clip(target[coarse], -10.0, 10.0))
        coarse_values[coarse] = float(coarse_far_weight) * _huber_slope_one(
            coarse_error, coarse_delta)

    reconstruction = float(np.mean(reconstruction_values))
    miss = float(np.mean(miss_values))
    outside = float(np.mean(outside_values))
    coarse_far = float(np.mean(coarse_values))
    return ValidationLoss(
        total=reconstruction + miss + outside + coarse_far,
        reconstruction=reconstruction,
        miss=miss,
        outside=outside,
        coarse_far_field=coarse_far,
        mean_weight=float(np.mean(weights)),
        sample_count=sample.size,
    )


class Patience:
    """Track strict loss improvements and consecutive failed checks."""

    def __init__(self, patience: int | None, min_delta: float = 0.0):
        if patience is not None and int(patience) <= 0:
            raise ValueError("patience must be positive or None")
        if not np.isfinite(min_delta) or float(min_delta) < 0.0:
            raise ValueError("min_delta must be finite and non-negative")
        self.limit = None if patience is None else int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.failed_checks = 0

    def update(self, loss: float) -> bool:
        """Return true only for a finite, strict improvement by ``min_delta``."""
        value = float(loss)
        improved = np.isfinite(value) and value < self.best - self.min_delta
        if improved:
            self.best = value
            self.failed_checks = 0
        else:
            self.failed_checks += 1
        return bool(improved)

    @property
    def should_stop(self) -> bool:
        return self.limit is not None and self.failed_checks >= self.limit

    def reset(self) -> None:
        self.best = float("inf")
        self.failed_checks = 0


class BestCheckpoint:
    """Deep-copy named NumPy arrays at the best validation loss.

    ``update`` is transactional: state is copied and validated before the saved
    checkpoint changes.  ``restore`` returns another deep copy so restoring and
    subsequently training cannot mutate the stored best state.
    """

    def __init__(self, patience: int | None, min_delta: float = 0.0):
        self.monitor = Patience(patience, min_delta)
        self._state: dict[str, np.ndarray] | None = None
        self.best_step: int | None = None

    def update(
        self,
        loss: float,
        state: Mapping[str, np.ndarray],
        *,
        step: int | None = None,
    ) -> bool:
        """Save ``state`` if ``loss`` is a genuine improvement."""
        value = float(loss)
        would_improve = np.isfinite(value) and (
            value < self.monitor.best - self.monitor.min_delta)
        copied: dict[str, np.ndarray] | None = None
        if would_improve:
            if not state:
                raise ValueError("checkpoint state must not be empty")
            copied = {}
            for name, array in state.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("checkpoint state names must be non-empty strings")
                if not isinstance(array, np.ndarray):
                    raise TypeError(f"checkpoint state '{name}' is not a NumPy array")
                copied[name] = np.array(array, copy=True, order="K")

        improved = self.monitor.update(value)
        if improved:
            assert copied is not None
            self._state = copied
            self.best_step = None if step is None else int(step)
        return improved

    @property
    def best_loss(self) -> float:
        return self.monitor.best

    @property
    def failed_checks(self) -> int:
        return self.monitor.failed_checks

    @property
    def should_stop(self) -> bool:
        return self.monitor.should_stop

    @property
    def has_checkpoint(self) -> bool:
        return self._state is not None

    def restore(self) -> dict[str, np.ndarray]:
        """Return a deep copy of the best state."""
        if self._state is None:
            raise RuntimeError("no best checkpoint has been recorded")
        return {name: np.array(array, copy=True, order="K")
                for name, array in self._state.items()}

    def reset(self) -> None:
        self.monitor.reset()
        self._state = None
        self.best_step = None
