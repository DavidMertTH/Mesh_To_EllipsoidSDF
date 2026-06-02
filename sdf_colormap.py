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


# Interior colour easing.  The interior magnitude (0 at the surface, 1 at the
# deepest point) is raised to this power before colouring.  >1 keeps the colour
# close to the surface tone for most of the interior and only ramps to the full
# colour near the genuinely deepest point — a longer, more gradual blend instead
# of an early saturation.  1.0 == plain linear.
SLICE_INTERIOR_GAMMA = 2.0


def colorize_sdf_slice(slice2d: np.ndarray, lut: np.ndarray | None = None,
                       depth: float = 1.0, out_band: float = 1.0,
                       gamma: float = SLICE_INTERIOR_GAMMA) -> np.ndarray:
    """Map a 2-D SDF slice to an ``(W, H, 4)`` uint8 RGBA image.

    Interior and exterior are scaled INDEPENDENTLY:
      * exterior (SDF > 0): LUT mapping surface -> far, normalised by a SMALL
        ``out_band`` (a few voxels) so the colour fades to background within a
        few voxels of the surface — a crisp boundary, not a wide halo;
      * interior (SDF < 0): the LUT's SURFACE colour is blended toward the
        DEEPEST colour across the WHOLE interior, eased by ``gamma`` (>1), so
        the full colour is reached only at the genuinely deepest point with a
        long, even gradient (the shared LUT itself plateaus just past the
        surface, which is why we lerp the two endpoints instead of indexing it).
    """
    if lut is None:
        lut = make_sdf_lut()
    n = int(lut.shape[0])
    sdf = np.asarray(slice2d, dtype=np.float32)

    pos = np.clip(sdf / max(float(out_band), 1e-9), 0.0, 1.0)    # exterior 0..1
    t = 0.5 + 0.5 * pos                              # exterior LUT position
    idx = np.clip((t * (n - 1)).astype(np.int32), 0, n - 1)
    rgba = lut[idx].astype(np.float32)

    surf = lut[(n - 1) // 2].astype(np.float32)      # surface colour (t=0.5)
    deep = lut[0].astype(np.float32)                 # deepest interior colour
    mag = np.clip(-sdf / max(float(depth), 1e-9), 0.0, 1.0) ** float(gamma)
    interior = surf * (1.0 - mag[..., None]) + deep * mag[..., None]
    rgba = np.where((sdf < 0.0)[..., None], interior, rgba)
    return np.ascontiguousarray(rgba.astype(np.ubyte))