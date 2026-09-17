"""Verify block rejection preserves the serial thickness rasterization."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thickness import (  # noqa: E402
    _refine_peak_depth,
    _sphere_offset_cache,
    local_thickness,
)


def _reference_thickness(grid: np.ndarray) -> np.ndarray:
    depth = np.maximum(-grid, 0.0).astype(np.float32)
    interior = depth > 0
    coords = np.argwhere(interior)
    radii = _refine_peak_depth(depth)[interior]
    order = np.argsort(-radii)
    coords = coords[order]
    radii = radii[order]
    out = np.zeros_like(depth)
    offsets = _sphere_offset_cache()
    for (z, y, x), radius in zip(coords, radii):
        diameter = 2.0 * float(radius)
        if out[z, y, x] >= diameter:
            continue
        points = offsets(float(radius)) + (z, y, x)
        valid = np.all((points >= 0) & (points < depth.shape), axis=1)
        points = points[valid]
        points = points[interior[points[:, 0], points[:, 1], points[:, 2]]]
        zz, yy, xx = points.T
        update = out[zz, yy, xx] < diameter
        out[zz[update], yy[update], xx[update]] = diameter
    out[interior] = np.maximum(out[interior], 1.0)
    return out


class ThicknessFastPathTests(unittest.TestCase):
    def test_block_rejection_matches_serial_splats(self) -> None:
        z, y, x = np.ogrid[-19:20, -16:17, -14:15]
        grid = (np.sqrt(x * x + y * y + z * z) - 12.0).astype(np.float32)
        np.testing.assert_array_equal(
            local_thickness(grid, 1.0), _reference_thickness(grid))


if __name__ == "__main__":
    unittest.main()
