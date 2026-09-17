"""Thickness-relative offsets for mesh SDF targets.

The interactive blowup is expressed as a signed fraction of the local feature
diameter.  Consequently a hand, arm, and torso each receive an offset that is
proportional to their own size, independent of voxel spacing or pose AABB.

The older ``*_thickness_limited_*`` helpers remain available for loading old
code/tests, but application paths use the relative helpers below.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp


DEFAULT_MAX_THICKNESS_FRACTION = 0.25
MAX_UI_THICKNESS_FRACTION = DEFAULT_MAX_THICKNESS_FRACTION
LEGACY_MAX_UI_BLOWUP_VOXELS = 10.0
# Backwards-compatible import for callers that still migrate an old slider.
MAX_UI_BLOWUP_VOXELS = LEGACY_MAX_UI_BLOWUP_VOXELS
BLOWUP_CARRIER_MARGIN_VOXELS = 4.0
SURFACE_THICKNESS_PROBE_VOXELS = 8.5


_CARRIER_VEC8F = wp.types.vector(8, wp.float32)
_CARRIER_VEC8I = wp.types.vector(8, wp.int32)


@wp.func
def _carrier_index(
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
) -> int:
    return z * ny * nx + y * nx + x


@wp.func
def _carrier_trilinear_sdf(
    sdf: wp.array(dtype=wp.float32),
    px: wp.float32,
    py: wp.float32,
    pz: wp.float32,
    nx: int,
    ny: int,
    nz: int,
) -> wp.float32:
    x0 = wp.clamp(int(wp.floor(px)), 0, nx - 1)
    y0 = wp.clamp(int(wp.floor(py)), 0, ny - 1)
    z0 = wp.clamp(int(wp.floor(pz)), 0, nz - 1)
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)
    fx = wp.clamp(px - wp.float32(x0), 0.0, 1.0)
    fy = wp.clamp(py - wp.float32(y0), 0.0, 1.0)
    fz = wp.clamp(pz - wp.float32(z0), 0.0, 1.0)
    i000 = _carrier_index(x0, y0, z0, nx, ny)
    i100 = _carrier_index(x1, y0, z0, nx, ny)
    i010 = _carrier_index(x0, y1, z0, nx, ny)
    i110 = _carrier_index(x1, y1, z0, nx, ny)
    i001 = _carrier_index(x0, y0, z1, nx, ny)
    i101 = _carrier_index(x1, y0, z1, nx, ny)
    i011 = _carrier_index(x0, y1, z1, nx, ny)
    i111 = _carrier_index(x1, y1, z1, nx, ny)
    c00 = sdf[i000] * (1.0 - fx) + sdf[i100] * fx
    c10 = sdf[i010] * (1.0 - fx) + sdf[i110] * fx
    c01 = sdf[i001] * (1.0 - fx) + sdf[i101] * fx
    c11 = sdf[i011] * (1.0 - fx) + sdf[i111] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _carrier_grid_gradient(
    sdf: wp.array(dtype=wp.float32),
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    """Finite-difference SDF gradient in voxel-index coordinates."""
    xm = wp.max(x - 1, 0)
    xp = wp.min(x + 1, nx - 1)
    ym = wp.max(y - 1, 0)
    yp = wp.min(y + 1, ny - 1)
    zm = wp.max(z - 1, 0)
    zp = wp.min(z + 1, nz - 1)
    gx = (
        sdf[_carrier_index(xp, y, z, nx, ny)]
        - sdf[_carrier_index(xm, y, z, nx, ny)]
    ) / wp.float32(wp.max(xp - xm, 1))
    gy = (
        sdf[_carrier_index(x, yp, z, nx, ny)]
        - sdf[_carrier_index(x, ym, z, nx, ny)]
    ) / wp.float32(wp.max(yp - ym, 1))
    gz = (
        sdf[_carrier_index(x, y, zp, nx, ny)]
        - sdf[_carrier_index(x, y, zm, nx, ny)]
    ) / wp.float32(wp.max(zp - zm, 1))
    return wp.vec3(gx, gy, gz)


@wp.func
def _carrier_lower_median(
    sdf: wp.array(dtype=wp.float32),
    thickness: wp.array(dtype=wp.float32),
    px: wp.float32,
    py: wp.float32,
    pz: wp.float32,
    nx: int,
    ny: int,
    nz: int,
) -> wp.float32:
    """Lower median of resolved interior corners around one probe."""
    x0 = wp.clamp(int(wp.floor(px)), 0, nx - 1)
    y0 = wp.clamp(int(wp.floor(py)), 0, ny - 1)
    z0 = wp.clamp(int(wp.floor(pz)), 0, nz - 1)
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)
    indices = _CARRIER_VEC8I(
        _carrier_index(x0, y0, z0, nx, ny),
        _carrier_index(x1, y0, z0, nx, ny),
        _carrier_index(x0, y1, z0, nx, ny),
        _carrier_index(x1, y1, z0, nx, ny),
        _carrier_index(x0, y0, z1, nx, ny),
        _carrier_index(x1, y0, z1, nx, ny),
        _carrier_index(x0, y1, z1, nx, ny),
        _carrier_index(x1, y1, z1, nx, ny),
    )
    inf = wp.float32(3.4028235e38)
    values = _CARRIER_VEC8F(inf, inf, inf, inf, inf, inf, inf, inf)
    count = int(0)
    for i in range(8):
        index = indices[i]
        value = thickness[index]
        if sdf[index] < 0.0 and value > 0.0:
            values[i] = value
            count += 1

    # Eight-value insertion sort stays entirely thread-local.  This mirrors
    # the NumPy lower-median rule without allocating an (N, 8) temporary for
    # every half-voxel probe.
    for i in range(1, 8):
        key = values[i]
        j = i - 1
        while j >= 0 and values[j] > key:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = key
    if count > 0:
        return values[(count - 1) // 2]
    return 0.0


@wp.kernel
def _surface_carried_thickness_kernel(
    sdf: wp.array(dtype=wp.float32),
    thickness: wp.array(dtype=wp.float32),
    spacing: wp.float32,
    band_limit: wp.float32,
    interior_repair_limit: wp.float32,
    thickness_floor_world: wp.float32,
    probe_steps: int,
    nx: int,
    ny: int,
    nz: int,
    out: wp.array(dtype=wp.float32),
):
    """Carry thickness along SDF normals, one independent voxel per thread."""
    tid = wp.tid()
    x = tid % nx
    y = (tid // nx) % ny
    z = tid // (nx * ny)
    value = sdf[tid]
    carried = wp.float32(0.0)
    if value < 0.0:
        carried = thickness[tid]

    exterior_candidate = value >= 0.0 and value <= band_limit
    interior_candidate = value < 0.0 and value >= -interior_repair_limit
    if not exterior_candidate and not interior_candidate:
        out[tid] = carried
        return

    gradient = _carrier_grid_gradient(sdf, x, y, z, nx, ny, nz)
    gx = gradient[0]
    gy = gradient[1]
    gz = gradient[2]
    norm = wp.sqrt(gx * gx + gy * gy + gz * gz)
    # In float32 an analytically zero ridge gradient can retain cancellation
    # noise around 1e-7 * spacing (especially for oblique one-voxel sheets).
    # Normalizing that noise would launch the chord probe in an arbitrary
    # direction and can overestimate thickness by an order of magnitude.
    # 1e-3 is still three orders below a regular SDF gradient (~spacing), so it
    # distinguishes medial cancellation without swallowing valid normals.
    gradient_epsilon = 1.0e-3 * spacing
    if norm <= gradient_epsilon:
        # A one-voxel or odd-width thin part has a genuine medial voxel where
        # the centred gradient cancels.  Borrow the strongest adjacent gradient
        # only to choose one of its two surface normals; the subsequent
        # first-exit march still measures this voxel's own uninterrupted chord.
        neighbour_gradient = _carrier_grid_gradient(
            sdf, wp.max(x - 1, 0), y, z, nx, ny, nz)
        neighbour_norm = wp.length(neighbour_gradient)
        candidate_gradient = _carrier_grid_gradient(
            sdf, wp.min(x + 1, nx - 1), y, z, nx, ny, nz)
        candidate_norm = wp.length(candidate_gradient)
        if candidate_norm > neighbour_norm:
            neighbour_gradient = candidate_gradient
            neighbour_norm = candidate_norm
        candidate_gradient = _carrier_grid_gradient(
            sdf, x, wp.max(y - 1, 0), z, nx, ny, nz)
        candidate_norm = wp.length(candidate_gradient)
        if candidate_norm > neighbour_norm:
            neighbour_gradient = candidate_gradient
            neighbour_norm = candidate_norm
        candidate_gradient = _carrier_grid_gradient(
            sdf, x, wp.min(y + 1, ny - 1), z, nx, ny, nz)
        candidate_norm = wp.length(candidate_gradient)
        if candidate_norm > neighbour_norm:
            neighbour_gradient = candidate_gradient
            neighbour_norm = candidate_norm
        candidate_gradient = _carrier_grid_gradient(
            sdf, x, y, wp.max(z - 1, 0), nx, ny, nz)
        candidate_norm = wp.length(candidate_gradient)
        if candidate_norm > neighbour_norm:
            neighbour_gradient = candidate_gradient
            neighbour_norm = candidate_norm
        candidate_gradient = _carrier_grid_gradient(
            sdf, x, y, wp.min(z + 1, nz - 1), nx, ny, nz)
        candidate_norm = wp.length(candidate_gradient)
        if candidate_norm > neighbour_norm:
            neighbour_gradient = candidate_gradient
            neighbour_norm = candidate_norm
        gx = neighbour_gradient[0]
        gy = neighbour_gradient[1]
        gz = neighbour_gradient[2]
        norm = neighbour_norm
    if norm <= gradient_epsilon:
        out[tid] = carried
        return

    nx_dir = gx / norm
    ny_dir = gy / norm
    nz_dir = gz / norm
    surface_distance_vox = value / spacing
    resolved = wp.float32(0.0)
    previous_probe_sdf = wp.float32(0.0)
    previous_probe_vox = wp.float32(0.0)
    chord_vox = wp.float32(0.0)
    entered_interior = int(0)
    for probe_step in range(probe_steps):
        inward_probe_vox = 0.25 + 0.5 * wp.float32(probe_step)
        probe_x = wp.float32(x) - nx_dir * (
            surface_distance_vox + inward_probe_vox)
        probe_y = wp.float32(y) - ny_dir * (
            surface_distance_vox + inward_probe_vox)
        probe_z = wp.float32(z) - nz_dir * (
            surface_distance_vox + inward_probe_vox)
        probe_sdf = _carrier_trilinear_sdf(
            sdf, probe_x, probe_y, probe_z, nx, ny, nz)
        if probe_sdf >= 0.0:
            if entered_interior == 1:
                denom = probe_sdf - previous_probe_sdf
                alpha = wp.float32(0.0)
                if denom > 1.0e-12 * spacing:
                    alpha = wp.clamp(
                        -previous_probe_sdf / denom, 0.0, 1.0)
                chord_vox = previous_probe_vox + alpha * (
                    inward_probe_vox - previous_probe_vox)
            break
        entered_interior = 1
        previous_probe_sdf = probe_sdf
        previous_probe_vox = inward_probe_vox
        probe_resolved = _carrier_lower_median(
            sdf, thickness, probe_x, probe_y, probe_z,
            nx, ny, nz,
        )
        resolved = wp.max(resolved, probe_resolved)
    # Coarse thickness is deliberately cheap, but its one-coarse-voxel floor
    # can turn a one-voxel finger/webbing into four voxels at a 512→128
    # stride.  Where the raw donor never rises above that floor, the first
    # opposite zero crossing along the surface normal supplies the full-res
    # local wall thickness instead.  The same first-exit rule prevents crossing
    # an air gap into a neighbouring body.
    candidate = wp.max(carried, resolved)
    if chord_vox > 0.0 and candidate <= thickness_floor_world * 1.001:
        # Replace the coarse floor instead of merely max-combining with it.
        # This matters on the interior side of the zero crossing: keeping the
        # repeated coarse value there would still over-erode a two-voxel limb
        # even though the exterior side had already recovered its true chord.
        candidate = chord_vox * spacing
    out[tid] = candidate


def _build_surface_carried_thickness_warp(
    sdf: np.ndarray,
    thickness: np.ndarray,
    spacing: float,
    band_limit: float,
    interior_repair_limit: float,
    thickness_floor_world: float,
    probe_steps: int,
    device: str | None,
) -> np.ndarray:
    """Run the normal-extension pass without Python-sized probe temporaries."""
    dev = device
    if dev is None:
        try:
            dev = "cuda:0" if wp.is_cuda_available() else "cpu"
        except Exception:
            dev = "cpu"
    source_sdf = np.ascontiguousarray(sdf, dtype=np.float32)
    source_thickness = np.ascontiguousarray(thickness, dtype=np.float32)
    sdf_wp = wp.array(source_sdf.ravel(), dtype=wp.float32, device=dev)
    thickness_wp = wp.array(
        source_thickness.ravel(), dtype=wp.float32, device=dev)
    out_wp = wp.empty(source_sdf.size, dtype=wp.float32, device=dev)
    nz, ny, nx = (int(size) for size in source_sdf.shape)
    wp.launch(
        _surface_carried_thickness_kernel,
        dim=source_sdf.size,
        inputs=[
            sdf_wp,
            thickness_wp,
            float(spacing),
            float(band_limit),
            float(interior_repair_limit),
            float(thickness_floor_world),
            int(probe_steps),
            nx,
            ny,
            nz,
            out_wp,
        ],
        device=dev,
    )
    return np.ascontiguousarray(
        out_wp.numpy().reshape(source_sdf.shape), dtype=np.float32)


def _validated_thickness_fraction(thickness_fraction: float) -> float:
    """Return a safe signed fraction of local feature diameter."""
    fraction = float(thickness_fraction)
    if not math.isfinite(fraction):
        raise ValueError("thickness_fraction must be finite")
    if not -0.5 < fraction < 0.5:
        raise ValueError(
            "thickness_fraction magnitude must be smaller than 0.5")
    return fraction


def legacy_voxel_blowup_to_thickness_fraction(voxels: float) -> float:
    """Map the former slider position to the new local-thickness strength.

    The old slider covered ``+-10 vox`` and reached the 25%-thickness safety
    cap only at its end.  Preserving that relative slider position turns, for
    example, ``-2 vox`` into ``-5%`` local thickness without retaining any
    dependence on the current grid spacing.
    """
    value = float(voxels)
    if not math.isfinite(value):
        raise ValueError("legacy voxel blowup must be finite")
    normalized = np.clip(
        value / LEGACY_MAX_UI_BLOWUP_VOXELS, -1.0, 1.0)
    return float(normalized * MAX_UI_THICKNESS_FRACTION)


def thickness_relative_offsets(
    values: np.ndarray,
    thickness_fraction: float,
    thickness: np.ndarray | None,
) -> np.ndarray:
    """Return ``fraction * local feature diameter`` at every resolved sample.

    The fraction is dimensionless and signed: positive values erode the mesh,
    negative values dilate it.  Missing local thickness fails closed with zero
    offset; a thickness-relative operation must never silently fall back to a
    uniform voxel/world-space border.
    """
    sdf = np.asarray(values, dtype=np.float32)
    if not np.isfinite(sdf).all():
        raise ValueError("SDF values must be finite")
    fraction = _validated_thickness_fraction(thickness_fraction)
    if fraction == 0.0 or thickness is None:
        return np.zeros_like(sdf, dtype=np.float32)

    local_thickness = np.asarray(thickness, dtype=np.float32)
    if local_thickness.shape != sdf.shape:
        raise ValueError("thickness must have the same shape as SDF values")
    if not np.isfinite(local_thickness).all() or np.any(local_thickness < 0.0):
        raise ValueError("thickness must be finite and non-negative")

    offsets = np.zeros_like(sdf, dtype=np.float32)
    known = local_thickness > 0.0
    offsets[known] = np.float32(fraction) * local_thickness[known]
    return np.ascontiguousarray(offsets, dtype=np.float32)


def apply_thickness_relative_blowup(
    values: np.ndarray,
    thickness_fraction: float,
    thickness: np.ndarray | None,
) -> np.ndarray:
    """Apply a purely local-thickness-relative offset to an SDF array."""
    sdf = np.asarray(values, dtype=np.float32)
    local_thickness = (
        None if thickness is None
        else np.asarray(thickness, dtype=np.float32)
    )
    if local_thickness is not None and local_thickness.shape != sdf.shape:
        raise ValueError("thickness must have the same shape as SDF values")

    source = np.ascontiguousarray(sdf, dtype=np.float32)
    result = np.empty(source.shape, dtype=np.float32)
    source_flat = source.ravel()
    result_flat = result.ravel()
    thickness_flat = (
        None if local_thickness is None else local_thickness.ravel())
    chunk_size = 1_048_576
    if source_flat.size == 0:
        thickness_relative_offsets(
            source_flat, thickness_fraction, thickness_flat)
        return result
    for start in range(0, source_flat.size, chunk_size):
        stop = min(start + chunk_size, source_flat.size)
        thickness_chunk = (
            None if thickness_flat is None
            else thickness_flat[start:stop]
        )
        offsets = thickness_relative_offsets(
            source_flat[start:stop],
            thickness_fraction,
            thickness_chunk,
        )
        np.add(
            source_flat[start:stop],
            offsets,
            out=result_flat[start:stop],
            casting="unsafe",
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def relative_blowup_extent_voxels(
    thickness_fraction: float,
    thickness: np.ndarray | None,
    dx: float,
) -> float:
    """Maximum possible magnitude of a relative offset, in voxels."""
    fraction = _validated_thickness_fraction(thickness_fraction)
    spacing = float(dx)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("dx must be finite and positive")
    if fraction == 0.0 or thickness is None:
        return 0.0
    field = np.asarray(thickness, dtype=np.float32)
    if not np.isfinite(field).all() or np.any(field < 0.0):
        raise ValueError("thickness must be finite and non-negative")
    if field.size == 0:
        return 0.0
    return abs(fraction) * float(np.max(field)) / spacing


def required_relative_sdf_margin(
    requested_margin: float,
    thickness_fraction: float,
    resolution: int,
) -> float:
    """Return an AABB margin that cannot clip the relative zero surface.

    ``SdfComputer`` applies half of its fractional margin on each side.  A
    local feature diameter cannot exceed the mesh extent on that axis, so a
    total margin of ``2*abs(fraction)`` contains the requested displacement.
    The fixed interpolation/optimizer guard is added separately as explicit
    cells on every axis; expressing it as a fraction would fail on short axes.
    """
    margin = float(requested_margin)
    fraction = _validated_thickness_fraction(thickness_fraction)
    n = int(resolution)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("requested_margin must be finite and non-negative")
    if n <= 0:
        raise ValueError("resolution must be positive")
    needed = 2.0 * abs(fraction)
    return max(margin, needed if fraction != 0.0 else margin)


def conservative_mirror_min(
    values: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Make a non-negative field exactly symmetric without raising known caps.

    Zero denotes an unresolved sample rather than a measured zero thickness.
    When only one mirror partner is resolved, copy that value to close
    downsampling-phase holes.  When both are known, keep the smaller cap.
    """
    field = np.asarray(values, dtype=np.float32)
    if field.ndim == 0:
        raise ValueError("values must have at least one dimension")
    mirror_axis = int(axis)
    if not -field.ndim <= mirror_axis < field.ndim:
        raise ValueError("axis is out of bounds")
    if not np.isfinite(field).all() or np.any(field < 0.0):
        raise ValueError("values must be finite and non-negative")
    mirrored = np.flip(field, axis=mirror_axis)
    both_known = (field > 0.0) & (mirrored > 0.0)
    symmetric = np.where(
        both_known,
        np.minimum(field, mirrored),
        np.maximum(field, mirrored),
    )
    return np.ascontiguousarray(symmetric, dtype=np.float32)


def thickness_limited_offsets(
    values: np.ndarray,
    requested_offset: float,
    thickness: np.ndarray | None,
    dx: float,
    max_thickness_fraction: float = DEFAULT_MAX_THICKNESS_FRACTION,
) -> np.ndarray:
    """Return a local offset field with thin-feature protection.

    ``thickness`` is the local feature *diameter* in world units.  Known values
    cap the requested offset to ``max_thickness_fraction * thickness``.
    Whenever a thickness field is supplied, missing values are handled
    conservatively with zero offset everywhere.  This also prevents a large
    negative request from pulling a distant unresolved sample into the
    optimizer's surface band.
    """
    sdf = np.asarray(values, dtype=np.float32)
    if not np.isfinite(sdf).all():
        raise ValueError("SDF values must be finite")
    requested = float(requested_offset)
    spacing = float(dx)
    fraction = float(max_thickness_fraction)
    if not math.isfinite(requested):
        raise ValueError("requested_offset must be finite")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("dx must be finite and positive")
    if not math.isfinite(fraction) or not 0.0 < fraction < 0.5:
        raise ValueError("max_thickness_fraction must be between 0 and 0.5")
    if requested == 0.0:
        return np.zeros_like(sdf, dtype=np.float32)
    if thickness is None:
        return np.full_like(sdf, np.float32(requested), dtype=np.float32)

    local_thickness = np.asarray(thickness, dtype=np.float32)
    if local_thickness.shape != sdf.shape:
        raise ValueError("thickness must have the same shape as SDF values")
    if not np.isfinite(local_thickness).all() or np.any(local_thickness < 0.0):
        raise ValueError("thickness must be finite and non-negative")

    magnitude = abs(requested)
    direction = 1.0 if requested > 0.0 else -1.0
    offsets = np.zeros_like(sdf, dtype=np.float32)
    known = local_thickness > 0.0
    local_cap = fraction * local_thickness[known]
    offsets[known] = np.float32(direction) * np.minimum(
        np.float32(magnitude), local_cap)
    return np.ascontiguousarray(offsets, dtype=np.float32)


def apply_thickness_limited_blowup(
    values: np.ndarray,
    requested_offset: float,
    thickness: np.ndarray | None,
    dx: float,
    max_thickness_fraction: float = DEFAULT_MAX_THICKNESS_FRACTION,
) -> np.ndarray:
    """Add :func:`thickness_limited_offsets` to an SDF array.

    The result is built in bounded chunks so a large 512³ target does not also
    need one full-volume temporary offset array.
    """
    sdf = np.asarray(values, dtype=np.float32)
    local_thickness = (
        None if thickness is None
        else np.asarray(thickness, dtype=np.float32)
    )
    if local_thickness is not None and local_thickness.shape != sdf.shape:
        raise ValueError("thickness must have the same shape as SDF values")

    source = np.ascontiguousarray(sdf, dtype=np.float32)
    result = np.empty(source.shape, dtype=np.float32)
    source_flat = source.ravel()
    result_flat = result.ravel()
    thickness_flat = (
        None if local_thickness is None else local_thickness.ravel())
    chunk_size = 1_048_576
    if source_flat.size == 0:
        # Preserve all scalar/shape validation for empty inputs too.
        thickness_limited_offsets(
            source_flat,
            requested_offset,
            thickness_flat,
            dx,
            max_thickness_fraction=max_thickness_fraction,
        )
        return result
    for start in range(0, source_flat.size, chunk_size):
        stop = min(start + chunk_size, source_flat.size)
        source_chunk = source_flat[start:stop]
        thickness_chunk = (
            None if thickness_flat is None
            else thickness_flat[start:stop]
        )
        offsets = thickness_limited_offsets(
            source_chunk,
            requested_offset,
            thickness_chunk,
            dx,
            max_thickness_fraction=max_thickness_fraction,
        )
        np.add(
            source_chunk,
            offsets,
            out=result_flat[start:stop],
            casting="unsafe",
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def build_surface_carried_thickness(
    grid: np.ndarray,
    thickness: np.ndarray,
    dx: float,
    max_exterior_vox: float | None = None,
    *,
    thickness_stride_vox: float = 1.0,
    device: str | None = None,
) -> np.ndarray:
    """Carry reliable feature thickness across the surface and exterior band.

    Raw local thickness is intentionally zero outside the mesh and can contain
    a floor-valued surface layer when a maximal inscribed sphere misses that
    layer by a sub-voxel amount.  Both effects are repaired by projecting
    near-surface voxels to the surface and probing inward along the SDF normal.
    Each probe remains conservative across its cell, while the strongest probe
    along that uninterrupted interior ray is retained.  This removes voxel-
    phase shrinkage without max-dilating thickness sideways from a torso into
    fingers or across an air gap.  ``thickness_stride_vox`` records the exact
    coarse-thickness sampling stride; ``device`` selects the compiled Warp
    backend used for the single parallel volume pass.
    """
    sdf = np.asarray(grid, dtype=np.float32)
    local_thickness = np.asarray(thickness, dtype=np.float32)
    spacing = float(dx)
    thickness_stride = float(thickness_stride_vox)
    if max_exterior_vox is None:
        exterior_vox = (
            relative_blowup_extent_voxels(
                MAX_UI_THICKNESS_FRACTION, local_thickness, spacing)
            + BLOWUP_CARRIER_MARGIN_VOXELS
        )
    else:
        exterior_vox = float(max_exterior_vox)
    if sdf.ndim != 3:
        raise ValueError("grid must have shape (nz, ny, nx)")
    if local_thickness.shape != sdf.shape:
        raise ValueError("thickness must have the same shape as grid")
    if not np.isfinite(sdf).all():
        raise ValueError("SDF grid must be finite")
    if not np.isfinite(local_thickness).all() or np.any(local_thickness < 0.0):
        raise ValueError("thickness must be finite and non-negative")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("dx must be finite and positive")
    if not math.isfinite(thickness_stride) or thickness_stride < 1.0:
        raise ValueError(
            "thickness_stride_vox must be finite and at least one")
    if not math.isfinite(exterior_vox) or exterior_vox < 0.0:
        raise ValueError("max_exterior_vox must be finite and non-negative")
    if exterior_vox == 0.0:
        # Preserve the explicit no-carrier contract used by callers that only
        # need the raw interior field.
        return np.ascontiguousarray(
            np.where(sdf < 0.0, local_thickness, 0.0),
            dtype=np.float32,
        )

    band_limit = np.float32(exterior_vox * spacing)
    # The coarse thickness pass has a known stride.  Inferring it from the
    # smallest positive thickness was both incorrect (a solid sphere may have
    # no floor-valued voxel) and could turn 33 probes into hundreds.  The known
    # stride gives a fixed, geometry-independent repair depth.
    floor_vox = thickness_stride
    # A downsampled thickness pass repeats blocks ``factor`` voxels wide.  Four
    # such blocks cover the worst observed surface phase while the half-voxel
    # stepping below still detects a one-voxel air gap.
    probe_limit_vox = max(
        SURFACE_THICKNESS_PROBE_VOXELS,
        4.0 * floor_vox + 0.5,
    )
    interior_repair_limit = np.float32(probe_limit_vox * spacing)
    probe_steps = int(np.ceil((probe_limit_vox - 0.25) / 0.5)) + 1
    # The former NumPy implementation allocated and sorted an (N, 8) corner
    # matrix once per half-voxel probe.  At production resolution that turned a
    # linear extension pass into several minutes of Python-side memory churn.
    # Warp executes the identical ray/probe rule independently per voxel in one
    # compiled launch (CUDA when available, native CPU otherwise).
    return _build_surface_carried_thickness_warp(
        sdf,
        local_thickness,
        spacing,
        float(band_limit),
        float(interior_repair_limit),
        thickness_stride * spacing,
        probe_steps,
        device,
    )


def sparse_band_offsets(
    blowup_vox: float,
    base_offsets: tuple[float, ...] = (
        -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0
    ),
) -> tuple[float, ...]:
    """Extend sparse normal bands far enough to bracket the moved surface."""
    requested = abs(float(blowup_vox))
    if not math.isfinite(requested):
        raise ValueError("blowup_vox must be finite")
    offsets = {float(value) for value in base_offsets}
    if requested > max((abs(value) for value in offsets), default=0.0):
        offsets.update({
            -requested,
            requested,
            -(requested + 1.0),
            requested + 1.0,
        })
    return tuple(sorted(offsets))
