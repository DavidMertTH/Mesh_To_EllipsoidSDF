"""Thickness-aware offsets for mesh SDF targets.

The UI expresses SDF blowup as one requested distance, but applying that
distance uniformly can erase or disproportionately enlarge thin features.
This module treats the requested value as a maximum and caps its magnitude at
a fraction of the local feature diameter.
"""

from __future__ import annotations

import math

import numpy as np


DEFAULT_MAX_THICKNESS_FRACTION = 0.25
MAX_UI_BLOWUP_VOXELS = 10.0
BLOWUP_CARRIER_MARGIN_VOXELS = 4.0


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
    max_exterior_vox: float = (
        MAX_UI_BLOWUP_VOXELS + BLOWUP_CARRIER_MARGIN_VOXELS
    ),
    *,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Carry interior thickness along SDF normals into the exterior band.

    Raw local thickness is intentionally zero outside the mesh.  A negative
    SDF offset moves the target surface outside, so those voxels need the
    thickness of their nearest surface feature.  For each exterior band voxel
    we project back to the surface using the SDF gradient and conservatively
    inspect the enclosing surface cell.  This avoids the thick-to-thin leakage
    caused by max-dilating the field near fingers, cloth edges, or close limbs,
    and cannot jump through an unresolved sheet into a thicker body behind it.
    """
    sdf = np.asarray(grid, dtype=np.float32)
    local_thickness = np.asarray(thickness, dtype=np.float32)
    spacing = float(dx)
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
    if not math.isfinite(exterior_vox) or exterior_vox < 0.0:
        raise ValueError("max_exterior_vox must be finite and non-negative")

    carried = np.ascontiguousarray(local_thickness, dtype=np.float32).copy()
    if exterior_vox == 0.0:
        return carried

    band_limit = np.float32(exterior_vox * spacing)
    candidates = np.flatnonzero(
        (sdf >= 0.0) & (sdf <= band_limit) & (carried <= 0.0))
    if candidates.size == 0:
        return carried

    nz, ny, nx = (int(size) for size in sdf.shape)
    plane = ny * nx
    chunk_size = max(1, int(chunk_size))
    source = local_thickness
    for start in range(0, candidates.size, chunk_size):
        flat = candidates[start:start + chunk_size]
        z = flat // plane
        remainder = flat - z * plane
        y = remainder // nx
        x = remainder - y * nx

        xm = np.maximum(x - 1, 0)
        xp = np.minimum(x + 1, nx - 1)
        ym = np.maximum(y - 1, 0)
        yp = np.minimum(y + 1, ny - 1)
        zm = np.maximum(z - 1, 0)
        zp = np.minimum(z + 1, nz - 1)
        gx = (
            sdf[z, y, xp] - sdf[z, y, xm]
        ) / np.maximum(xp - xm, 1)
        gy = (
            sdf[z, yp, x] - sdf[z, ym, x]
        ) / np.maximum(yp - ym, 1)
        gz = (
            sdf[zp, y, x] - sdf[zm, y, x]
        ) / np.maximum(zp - zm, 1)
        norm = np.sqrt(gx * gx + gy * gy + gz * gz)
        valid_normal = norm > 1.0e-7
        safe_norm = np.where(valid_normal, norm, 1.0)
        nx_dir = gx / safe_norm
        ny_dir = gy / safe_norm
        nz_dir = gz / safe_norm
        surface_distance_vox = sdf[z, y, x] / np.float32(spacing)

        # Resolve positive interior samples among the eight corners enclosing
        # the projected nearest surface.  A reduced-resolution thickness pass
        # can leave a short run of zero-valued interior cells.  In that case,
        # advance at most four voxels along the inward normal.  Half-voxel
        # probes must remain inside according to the trilinearly interpolated
        # SDF; the first exterior probe terminates the search.  This prevents a
        # jump through even a one-voxel air gap from an unresolved sheet to a
        # thicker component.
        resolved = np.full(flat.size, np.inf, dtype=np.float32)
        has_resolved_corner = np.zeros(flat.size, dtype=np.bool_)
        active = valid_normal.copy()
        for probe_step in range(9):
            if not np.any(active):
                break
            inward_probe_vox = np.float32(0.25 + 0.5 * probe_step)
            probe_x = (
                x - nx_dir * (surface_distance_vox + inward_probe_vox))
            probe_y = (
                y - ny_dir * (surface_distance_vox + inward_probe_vox))
            probe_z = (
                z - nz_dir * (surface_distance_vox + inward_probe_vox))
            x0 = np.clip(np.floor(probe_x).astype(np.int64), 0, nx - 1)
            y0 = np.clip(np.floor(probe_y).astype(np.int64), 0, ny - 1)
            z0 = np.clip(np.floor(probe_z).astype(np.int64), 0, nz - 1)
            x1 = np.minimum(x0 + 1, nx - 1)
            y1 = np.minimum(y0 + 1, ny - 1)
            z1 = np.minimum(z0 + 1, nz - 1)
            fx = np.clip(probe_x - x0, 0.0, 1.0).astype(np.float32)
            fy = np.clip(probe_y - y0, 0.0, 1.0).astype(np.float32)
            fz = np.clip(probe_z - z0, 0.0, 1.0).astype(np.float32)
            probe_sdf = np.zeros(flat.size, dtype=np.float32)
            corners = (
                (z0, y0, x0, (1.0 - fz) * (1.0 - fy) * (1.0 - fx)),
                (z0, y0, x1, (1.0 - fz) * (1.0 - fy) * fx),
                (z0, y1, x0, (1.0 - fz) * fy * (1.0 - fx)),
                (z0, y1, x1, (1.0 - fz) * fy * fx),
                (z1, y0, x0, fz * (1.0 - fy) * (1.0 - fx)),
                (z1, y0, x1, fz * (1.0 - fy) * fx),
                (z1, y1, x0, fz * fy * (1.0 - fx)),
                (z1, y1, x1, fz * fy * fx),
            )
            for corner_z, corner_y, corner_x, weight in corners:
                probe_sdf += (
                    weight * sdf[corner_z, corner_y, corner_x])
            probe_inside = active & (probe_sdf < 0.0)
            for corner_z, corner_y, corner_x, _weight in corners:
                corner_thickness = source[
                    corner_z, corner_y, corner_x]
                interior_corner = (
                    probe_inside &
                    (sdf[corner_z, corner_y, corner_x] < 0.0)
                )
                resolved_corner = (
                    interior_corner & (corner_thickness > 0.0))
                has_resolved_corner |= resolved_corner
                resolved = np.where(
                    resolved_corner,
                    np.minimum(resolved, corner_thickness),
                    resolved,
                )
            active = probe_inside & ~has_resolved_corner
        resolved = np.where(
            has_resolved_corner & np.isfinite(resolved),
            resolved,
            0.0,
        ).astype(np.float32)
        carried.ravel()[flat] = resolved

    return carried


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
