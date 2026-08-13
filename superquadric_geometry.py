"""Numerically stable NumPy geometry for superquadric maintenance.

The differentiable Warp kernels used during fitting have a matching evaluator
in :mod:`optimization`.  Population management, validation and tests must use
the same primitive rather than silently falling back to ellipsoids, so this
module contains the CPU-side reference implementation.

Quaternions use the project-wide ``(x, y, z, w)`` convention.  Bend values are
physical quadratic curvatures: ``x += 0.5 * kx * z**2`` and likewise for ``y``.
"""

from __future__ import annotations

import math
import operator

import numpy as np


EPS_MIN = 0.1
EPS_MAX = 2.0


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return the local-to-world rotation matrix for an ``xyzw`` quaternion."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-15:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
             2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
             1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _logaddexp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    maximum = np.maximum(a, b)
    return maximum + np.log(np.exp(a - maximum) + np.exp(b - maximum))


def _log_beta_and_gradient_log_local(
    points_local: np.ndarray,
    radii: np.ndarray,
    eps: np.ndarray,
    bend: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``log(beta)`` and its bent-local gradient.

    This mirrors ``_sq_log_beta_grad`` and the bend-Jacobian transform in the
    Warp implementation.  In particular, the relative coordinate floor and its
    zero subgradient are deliberately identical on symmetry axes.
    """
    p = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    r = np.maximum(np.abs(np.asarray(radii, dtype=np.float64).reshape(3)), 1.0e-12)
    e1, e2 = np.clip(
        np.asarray(eps, dtype=np.float64).reshape(2), EPS_MIN, EPS_MAX)
    k = (np.zeros(2, dtype=np.float64) if bend is None
         else np.asarray(bend, dtype=np.float64).reshape(2))

    z = p[:, 2]
    unbent = p.copy()
    unbent[:, 0] -= 0.5 * k[0] * z * z
    unbent[:, 1] -= 0.5 * k[1] * z * z

    absolute = np.abs(unbent)
    safe_absolute = np.maximum(absolute, r[None, :] * 1.0e-7)
    log_safe = np.log(safe_absolute)
    log_normalized = log_safe - np.log(r)[None, :]

    log_x = (2.0 / e2) * log_normalized[:, 0]
    log_y = (2.0 / e2) * log_normalized[:, 1]
    log_a = _logaddexp(log_x, log_y)
    log_xy = (e2 / e1) * log_a
    log_z = (2.0 / e1) * log_normalized[:, 2]
    log_f = _logaddexp(log_xy, log_z)
    log_beta = 0.5 * e1 * log_f

    log_w_xy = log_xy - log_f
    log_w_z = log_z - log_f
    log_w_x = log_x - log_a
    log_w_y = log_y - log_a

    w_x = np.exp(log_w_x)
    w_y = np.exp(log_w_y)
    w_xy = np.exp(log_w_xy)
    w_z = np.exp(log_w_z)
    gradient_unbent = np.column_stack(
        [
            w_xy * w_x * unbent[:, 0] / safe_absolute[:, 0] ** 2,
            w_xy * w_y * unbent[:, 1] / safe_absolute[:, 1] ** 2,
            w_z * unbent[:, 2] / safe_absolute[:, 2] ** 2,
        ]
    )

    # Inverse bend Jacobian: u=(x-.5*kx*z², y-.5*ky*z², z), hence
    # grad_p beta = J_u^T grad_u beta.
    gradient = gradient_unbent.copy()
    gradient[:, 2] = (
        gradient_unbent[:, 2]
        - k[0] * z * gradient_unbent[:, 0]
        - k[1] * z * gradient_unbent[:, 1]
    )
    return log_beta, gradient


def beta_and_gradient_local(
    points_local: np.ndarray,
    radii: np.ndarray,
    eps: np.ndarray,
    bend: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate radial coordinate ``beta`` and its local-space gradient.

    ``beta == 1`` is the superquadric surface.  The complete power chain is
    evaluated in log space, including the mixture weights used by the analytic
    derivative.  This avoids both float overflow and the ``0 ** fractional``
    derivative singularities that destabilise direct formulations.
    """
    log_beta, gradient_log = _log_beta_and_gradient_log_local(
        points_local, radii, eps, bend)
    beta = np.exp(np.clip(log_beta, -80.0, 700.0))
    gradient = beta[:, None] * gradient_log
    return (
        np.nan_to_num(beta, nan=0.0, posinf=np.finfo(np.float64).max),
        np.nan_to_num(
            gradient, nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=-np.finfo(np.float64).max,
        ),
    )


def signed_distance_local(
    points_local: np.ndarray,
    radii: np.ndarray,
    eps: np.ndarray,
    bend: np.ndarray | None = None,
) -> np.ndarray:
    """First-order Euclidean signed distance to a local superquadric.

    Near every regular surface point, ``(beta - 1) / |grad(beta)|`` has unit
    normal derivative.  A smooth centre fallback is used only where the
    implicit gradient vanishes; that region is far from the fitted zero set.
    """
    p = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    r = np.maximum(np.abs(np.asarray(radii, dtype=np.float64).reshape(3)), 1.0e-12)
    log_beta, gradient_log = _log_beta_and_gradient_log_local(
        p, r, eps, bend)
    r_min = float(np.min(r))
    metric_epsilon = 1.0e-8 / r_min
    distance = np.empty(len(p), dtype=np.float64)

    outside = log_beta >= 0.0
    if np.any(outside):
        inv_beta = np.exp(np.maximum(-log_beta[outside], -80.0))
        denominator = np.sqrt(
            np.einsum(
                "ij,ij->i", gradient_log[outside], gradient_log[outside])
            + metric_epsilon ** 2)
        distance[outside] = (1.0 - inv_beta) / denominator

    inside = ~outside
    if np.any(inside):
        beta = np.exp(np.maximum(log_beta[inside], -80.0))
        gradient_beta = beta[:, None] * gradient_log[inside]
        center_gate = np.maximum((1.0e-4 - beta) / 1.0e-4, 0.0) / r_min
        denominator = np.sqrt(
            np.einsum("ij,ij->i", gradient_beta, gradient_beta)
            + center_gate ** 2
            + metric_epsilon ** 2)
        distance[inside] = (beta - 1.0) / denominator
    return np.nan_to_num(distance, nan=0.0, posinf=1.0e30, neginf=-1.0e30).astype(
        np.float32)


def signed_distance(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    eps: np.ndarray,
    points: np.ndarray,
    bend: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate one world-space superquadric at ``points``."""
    world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = quaternion_matrix(rotation)
    local = (world - np.asarray(center, dtype=np.float64).reshape(3)) @ matrix
    return signed_distance_local(local, radii, eps, bend)


def signed_distance_batch(
    centers: np.ndarray,
    radii: np.ndarray,
    rotations: np.ndarray,
    eps: np.ndarray,
    points: np.ndarray,
    bend: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate ``E`` superquadrics at ``N`` points, returning ``(E, N)``."""
    c = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    r = np.asarray(radii, dtype=np.float32).reshape(-1, 3)
    q = np.asarray(rotations, dtype=np.float32).reshape(-1, 4)
    e = np.asarray(eps, dtype=np.float32).reshape(-1, 2)
    if not (len(c) == len(r) == len(q) == len(e)):
        raise ValueError("superquadric population arrays must have equal length")
    if bend is None:
        b = np.zeros((len(c), 2), dtype=np.float32)
    else:
        b = np.asarray(bend, dtype=np.float32).reshape(-1, 2)
        if len(b) != len(c):
            raise ValueError("bend population must match centers")
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return np.stack(
        [signed_distance(c[i], r[i], q[i], e[i], p, b[i])
         for i in range(len(c))],
        axis=0,
    ) if len(c) else np.empty((0, len(p)), dtype=np.float32)


def _signed_power(value: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(value) * np.abs(value) ** float(exponent)


def surface_points(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    eps: np.ndarray,
    directions: np.ndarray,
    bend: np.ndarray | None = None,
) -> np.ndarray:
    """Map unit directions to deterministic points on a superquadric surface."""
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    norm = np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1.0e-30)
    dirs = dirs / norm
    e1, e2 = np.clip(
        np.asarray(eps, dtype=np.float64).reshape(2), EPS_MIN, EPS_MAX)
    r = np.abs(np.asarray(radii, dtype=np.float64).reshape(3))

    eta = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
    omega = np.arctan2(dirs[:, 1], dirs[:, 0])
    cos_eta = np.cos(eta)
    local = np.column_stack(
        [
            r[0] * _signed_power(cos_eta, e1) * _signed_power(np.cos(omega), e2),
            r[1] * _signed_power(cos_eta, e1) * _signed_power(np.sin(omega), e2),
            r[2] * _signed_power(np.sin(eta), e1),
        ]
    )
    if bend is not None:
        k = np.asarray(bend, dtype=np.float64).reshape(2)
        z2 = local[:, 2] ** 2
        local[:, 0] += 0.5 * k[0] * z2
        local[:, 1] += 0.5 * k[1] * z2
    matrix = quaternion_matrix(rotation)
    return (
        np.asarray(center, dtype=np.float64).reshape(3) + local @ matrix.T
    ).astype(np.float32)


def interior_points_local(
    radii: np.ndarray,
    eps: np.ndarray,
    count: int,
    bend: np.ndarray | None = None,
    *,
    seed: int = 0,
    beta_limit: float = 1.0,
) -> np.ndarray:
    """Return deterministic, volume-uniform local interior probes.

    Candidates are drawn uniformly from the unbent local bounding box and
    accepted using the same stable ``log(beta)`` evaluator as the distance
    functions.  Rejection sampling is preferable to deforming a unit-ball cloud:
    it preserves the superquadric's actual volume measure for every supported
    exponent pair instead of over- or under-weighting rounded corners.

    ``beta_limit`` homothetically contracts the accepted unbent cloud before the
    bend is applied.  Values below one are useful when probes must stay a fixed
    fractional margin away from the surface.  The quadratic forward bend

    ``(x, y, z) -> (x + .5*kx*z^2, y + .5*ky*z^2, z)``

    is applied last.  Its Jacobian determinant is one, so volume uniformity is
    retained and the returned points exactly invert through the evaluator's bend.
    The same arguments and seed always produce byte-identical float32 results.
    """
    try:
        n = operator.index(count)
    except TypeError as exc:
        raise ValueError("count must be a non-negative integer") from exc
    if isinstance(count, (bool, np.bool_)) or n < 0:
        raise ValueError("count must be a non-negative integer")

    r_input = np.asarray(radii, dtype=np.float64).reshape(3)
    e_input = np.asarray(eps, dtype=np.float64).reshape(2)
    k = (np.zeros(2, dtype=np.float64) if bend is None
         else np.asarray(bend, dtype=np.float64).reshape(2))
    limit = float(beta_limit)
    if not np.all(np.isfinite(r_input)):
        raise ValueError("radii must be finite")
    if not np.all(np.isfinite(e_input)):
        raise ValueError("eps must be finite")
    if not np.all(np.isfinite(k)):
        raise ValueError("bend must be finite")
    if not np.isfinite(limit) or limit <= 0.0 or limit > 1.0:
        raise ValueError("beta_limit must be finite and in (0, 1]")
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    r = np.maximum(np.abs(r_input), 1.0e-12)
    e = np.clip(e_input, EPS_MIN, EPS_MAX)
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    remaining = n
    draws = 0
    # Over the supported convex exponent range the smallest volume fraction is
    # the octahedral limit (1/6), so eight candidates per requested point leave
    # comfortable headroom.  Chunking bounds peak memory for large probe clouds.
    max_draws = max(4096, 256 * n)
    while remaining > 0:
        draw_count = min(65536, max(64, 8 * remaining))
        candidates = rng.uniform(-1.0, 1.0, size=(draw_count, 3)) * r[None, :]
        log_beta, _gradient_log = _log_beta_and_gradient_log_local(
            candidates, r, e, None)
        inside = candidates[log_beta < 0.0]
        if len(inside):
            take = min(remaining, len(inside))
            accepted.append(inside[:take])
            remaining -= take
        draws += draw_count
        if draws >= max_draws and remaining > 0:
            raise RuntimeError(
                "failed to generate the requested superquadric interior probes")

    unbent = np.concatenate(accepted, axis=0)
    if limit != 1.0:
        unbent *= limit
    local = unbent.copy()
    z2 = unbent[:, 2] * unbent[:, 2]
    local[:, 0] += 0.5 * k[0] * z2
    local[:, 1] += 0.5 * k[1] * z2
    return np.ascontiguousarray(local, dtype=np.float32)


def interior_points(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    eps: np.ndarray,
    count: int,
    bend: np.ndarray | None = None,
    *,
    seed: int = 0,
    beta_limit: float = 1.0,
) -> np.ndarray:
    """Return deterministic world-space probes filling one SQ or bent SQ."""
    local = interior_points_local(
        radii, eps, count, bend, seed=seed, beta_limit=beta_limit)
    matrix = quaternion_matrix(rotation)
    return np.ascontiguousarray(
        np.asarray(center, dtype=np.float64).reshape(3)
        + local.astype(np.float64) @ matrix.T,
        dtype=np.float32,
    )


def volume(radii: np.ndarray, eps: np.ndarray) -> float:
    """Analytic superquadric volume (quadratic bend preserves this volume)."""
    a, b, c = np.maximum(
        np.abs(np.asarray(radii, dtype=np.float64).reshape(3)), 1.0e-30)
    e1, e2 = np.clip(
        np.asarray(eps, dtype=np.float64).reshape(2), EPS_MIN, EPS_MAX)
    # Superellipse cross-section area times the integral of its z-dependent
    # scale.  Written through lgamma for stable behaviour at the full ε range.
    log_base_area = (
        math.log(4.0 * a * b)
        + 2.0 * math.lgamma(1.0 + 0.5 * e2)
        - math.lgamma(1.0 + e2)
    )
    log_z_integral = (
        math.log(c * e1)
        + math.lgamma(0.5 * e1)
        + math.lgamma(1.0 + e1)
        - math.lgamma(1.0 + 1.5 * e1)
    )
    return float(math.exp(log_base_area + log_z_integral))
