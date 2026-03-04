"""
sdf_colormap.py — SDF-specific color mapping utilities for pyqtgraph.
"""

import numpy as np
import pyqtgraph as pg


def make_sdf_colormap() -> pg.ColorMap:

    pos = np.array([0.0, 0.45, 0.50, 0.55, 1.0], dtype=float)
    colors = np.array([
        [242, 230,  65, 255],
        [242, 213,  65, 255],
        [255, 255, 255, 255],
        [ 24,  40,  89, 255],
        [  2,  11,  13, 255],
    ], dtype=np.ubyte)
    return pg.ColorMap(pos, colors)


def make_sdf_lut(npts: int = 256) -> np.ndarray:
    """Pre-baked 256-entry RGBA lookup table for the SDF colormap."""
    cmap = make_sdf_colormap()
    return cmap.getLookupTable(0.0, 1.0, npts, alpha=True)