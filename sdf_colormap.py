"""
sdf_colormap.py — SDF-specific color mapping utilities for pyqtgraph.
"""

import numpy as np
import pyqtgraph as pg

import theme


def make_sdf_colormap() -> pg.ColorMap:
    # Diverging ramp from the brand colours (theme.py): brand secondary on the
    # negative side → a "surface" stop → brand primary → a "far" stop.
    #
    # Light/dark adaptive: in dark mode the surface pops bright (white) and the
    # far end blends into the dark background (near-black); in light mode this
    # inverts — dark surface stop, near-white far end — so the heatmap stays
    # readable on a white viewport.
    if theme.is_dark_mode():
        surface = [255, 255, 255, 255]
        far = [2, 11, 13, 255]
    else:
        surface = [12, 14, 18, 255]
        far = [240, 242, 246, 255]
    pos = np.array([0.0, 0.45, 0.50, 0.55, 1.0], dtype=float)
    colors = np.array([
        [*theme.YELLOW, 255],
        [*theme.scaled(theme.YELLOW, 0.9), 255],
        surface,
        [*theme.scaled(theme.BLUE, 0.37), 255],
        far,
    ], dtype=np.ubyte)
    return pg.ColorMap(pos, colors)


def make_sdf_lut(npts: int = 256) -> np.ndarray:
    """Pre-baked 256-entry RGBA lookup table for the SDF colormap."""
    cmap = make_sdf_colormap()
    return cmap.getLookupTable(0.0, 1.0, npts, alpha=True)


# Slice presentation constants shared by the 2-D panel and the 3-D viewport.
# A sub-linear interior curve makes thickness differences readable even when the
# global SDF depth is set by a much thicker body part elsewhere in the mesh.
SLICE_INTERIOR_GAMMA = 0.70
SLICE_INTERIOR_MIN_ALPHA = 225.0
SLICE_EXTERIOR_ALPHA_GAMMA = 1.35
SLICE_EXTERIOR_BAND_VOXELS = 8.0


def colorize_sdf_slice(slice2d: np.ndarray, lut: np.ndarray | None = None,
                       depth: float = 1.0, out_band: float = 1.0,
                       gamma: float = SLICE_INTERIOR_GAMMA) -> np.ndarray:
    """Map a 2-D SDF slice to an ``(W, H, 4)`` uint8 RGBA image.

    Interior and exterior are scaled INDEPENDENTLY:
      * exterior (SDF > 0): LUT mapping surface -> far, with alpha fading to
        zero across ``out_band``.  This provides distance context near the
        cross-section without drawing an opaque rectangular slice plane;
      * interior (SDF < 0): the LUT's SURFACE colour is blended toward the
        DEEPEST colour across the WHOLE interior.  Its alpha stays high so the
        cross-section reads as a filled area instead of only a surface rim.
    """
    if lut is None:
        lut = make_sdf_lut()
    n = int(lut.shape[0])
    sdf = np.asarray(slice2d, dtype=np.float32)

    pos = np.clip(sdf / max(float(out_band), 1e-9), 0.0, 1.0)    # exterior 0..1
    t = 0.5 + 0.5 * pos                              # exterior LUT position
    idx = np.clip((t * (n - 1)).astype(np.int32), 0, n - 1)
    rgba = lut[idx].astype(np.float32)
    exterior_alpha = 255.0 * np.power(
        np.clip(1.0 - pos, 0.0, 1.0),
        SLICE_EXTERIOR_ALPHA_GAMMA,
    )
    rgba[..., 3] = exterior_alpha

    surf = lut[(n - 1) // 2].astype(np.float32)      # surface colour (t=0.5)
    deep = lut[0].astype(np.float32)                 # deepest interior colour
    mag = np.clip(-sdf / max(float(depth), 1e-9), 0.0, 1.0) ** float(gamma)
    interior = surf * (1.0 - mag[..., None]) + deep * mag[..., None]
    interior[..., 3] = (
        SLICE_INTERIOR_MIN_ALPHA
        + (255.0 - SLICE_INTERIOR_MIN_ALPHA) * (1.0 - mag)
    )
    rgba = np.where((sdf < 0.0)[..., None], interior, rgba)
    return np.ascontiguousarray(rgba.astype(np.ubyte))
