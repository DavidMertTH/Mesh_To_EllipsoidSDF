"""
theme.py — central definition of the app's two brand colours.

The whole UI is styled around a **blue** and a **yellow**.  Change the two
constants below and everything that follows re-themes with them:

  * the mesh surface (blue) and the ellipsoids (yellow) in the 3-D viewport
    (``viewer3d.py``),
  * the SDF slice colormap (``sdf_colormap.py``),
  * the loss-curve colour cycle (``run_tracker.py``).

Colours are stored as 0–255 ``(r, g, b)`` int triples; helpers convert to the
0–1 float tuples / RGBA arrays / hex strings the various call sites need.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── The two brand colours — edit these to re-theme the app ────────────────────
# Primary (mesh) and secondary (ellipsoids).  These are the permanent defaults;
# a user's live picks are persisted to theme_colors.json and override them.
BLUE = (170, 170, 255)
YELLOW = (85, 0, 255)

# Where the user's chosen colours are persisted between runs.
_COLORS_FILE = Path(__file__).with_name("theme_colors.json")


# ── Conversions ───────────────────────────────────────────────────────────────

def rgb01(rgb: tuple) -> tuple:
    """(r, g, b) 0–255 → (r, g, b) floats in 0–1."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def rgba01(rgb: tuple, alpha: float = 1.0) -> tuple:
    """(r, g, b) 0–255 → (r, g, b, a) floats in 0–1."""
    return (*rgb01(rgb), float(alpha))


def rgba_array(rgb: tuple, alpha: float = 1.0) -> np.ndarray:
    """(r, g, b) 0–255 → float32 RGBA numpy array in 0–1."""
    return np.array(rgba01(rgb, alpha), dtype=np.float32)


def scaled(rgb: tuple, factor: float) -> tuple:
    """Brightness-scaled copy of a colour (clamped to 0–255 ints)."""
    return tuple(int(max(0, min(255, round(c * factor)))) for c in rgb)


def hex_str(rgb: tuple) -> str:
    """(r, g, b) 0–255 → ``'#rrggbb'`` for stylesheets."""
    return "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])


# ── Convenience forms (refreshed by the setters below) ────────────────────────
BLUE01 = rgb01(BLUE)
YELLOW01 = rgb01(YELLOW)
BLUE_HEX = hex_str(BLUE)
YELLOW_HEX = hex_str(YELLOW)


def set_primary(rgb: tuple) -> None:
    """Set the primary (mesh / blue) colour and refresh derived forms.

    Read ``theme.BLUE`` / ``theme.BLUE01`` at call time (not captured at import)
    so callers pick up live changes from the colour pickers.
    """
    global BLUE, BLUE01, BLUE_HEX
    BLUE = tuple(int(max(0, min(255, c))) for c in rgb)
    BLUE01 = rgb01(BLUE)
    BLUE_HEX = hex_str(BLUE)


def set_secondary(rgb: tuple) -> None:
    """Set the secondary (ellipsoid / yellow) colour and refresh derived forms."""
    global YELLOW, YELLOW01, YELLOW_HEX
    YELLOW = tuple(int(max(0, min(255, c))) for c in rgb)
    YELLOW01 = rgb01(YELLOW)
    YELLOW_HEX = hex_str(YELLOW)


# ── Light / dark mode ─────────────────────────────────────────────────────────

def is_dark_mode() -> bool:
    """True if the app palette is dark.  Defaults to dark if no app exists yet."""
    from PySide6 import QtGui, QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        return True
    return app.palette().color(QtGui.QPalette.Window).lightness() < 128


def bg(dark: tuple = (0, 0, 0)) -> tuple:
    """Background colour: white in light mode, the given dark colour otherwise.

    Used for the 3-D viewport and the loss-curve plot so both turn white when
    the OS / Qt is in light mode instead of staying black.
    """
    return (255, 255, 255) if not is_dark_mode() else tuple(dark)


def pg_fg() -> str:
    """pyqtgraph foreground (axis lines / text) spec for the current mode."""
    return "d" if is_dark_mode() else "k"


# ── Persistence (remember the user's colour choice across restarts) ───────────

def save_colors() -> None:
    """Persist the current primary/secondary colours to disk."""
    try:
        _COLORS_FILE.write_text(
            json.dumps({"primary": list(BLUE), "secondary": list(YELLOW)}),
            encoding="utf-8",
        )
    except Exception:
        pass


def load_colors() -> None:
    """Restore saved primary/secondary colours, if any (called at import)."""
    try:
        data = json.loads(_COLORS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(data.get("primary"), list) and len(data["primary"]) == 3:
        set_primary(tuple(data["primary"]))
    if isinstance(data.get("secondary"), list) and len(data["secondary"]) == 3:
        set_secondary(tuple(data["secondary"]))


# Apply any persisted colours as soon as the module is imported, so every widget
# is built with the user's choice from the start.
load_colors()
