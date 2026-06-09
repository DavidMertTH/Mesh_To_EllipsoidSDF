from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui

import theme


_FONT_FILE = Path(__file__).with_name("assets") / "fonts" / "Syne-Bold.ttf"
_FONT_FAMILY: str | None = None


def display_font_family() -> str:
    global _FONT_FAMILY
    if _FONT_FAMILY:
        return _FONT_FAMILY
    fid = QtGui.QFontDatabase.addApplicationFont(str(_FONT_FILE))
    if fid != -1:
        families = QtGui.QFontDatabase.applicationFontFamilies(fid)
        if families:
            _FONT_FAMILY = families[0]
            return _FONT_FAMILY
    _FONT_FAMILY = "Segoe UI"
    return _FONT_FAMILY


def _contrast_outline(rgb: tuple[int, int, int]) -> QtGui.QColor:
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return QtGui.QColor(12, 12, 14) if lum > 120 else QtGui.QColor(245, 245, 240)


def _paint_trashy_word(
    painter: QtGui.QPainter,
    text: str,
    rect: QtCore.QRectF,
    fill: QtGui.QColor,
    outline: QtGui.QColor,
    *,
    max_font_px: int,
    stretch_to_rect: bool = False,
) -> None:
    family = display_font_family()
    font = QtGui.QFont()
    font.setFamily(family)
    font.setPixelSize(max(1, int(max_font_px)))
    font.setWeight(QtGui.QFont.Black)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    path = QtGui.QPainterPath()
    path.addText(0, 0, font, text)
    br = path.boundingRect()
    sx = rect.width() / max(1.0, br.width())
    sy = rect.height() / max(1.0, br.height())
    if not stretch_to_rect:
        sx = sy = min(sx, sy)

    tr = QtGui.QTransform()
    tr.translate(
        rect.x() + (rect.width() - br.width() * sx) * 0.5 - br.x() * sx,
        rect.y() + (rect.height() - br.height() * sy) * 0.5 - br.y() * sy,
    )
    tr.scale(sx, sy)
    path = tr.map(path)

    # Rough black comic outline: several tiny offset strokes around the path.
    painter.setBrush(QtCore.Qt.NoBrush)
    for dx, dy in ((0, 0), (1.6, -1.0), (-1.5, 0.9), (0.8, 1.4)):
        shifted = QtGui.QPainterPath(path)
        shifted.translate(dx, dy)
        painter.setPen(QtGui.QPen(outline, max(7.0, rect.height() * 0.085),
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                  QtCore.Qt.RoundJoin))
        painter.drawPath(shifted)

    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(fill)
    painter.drawPath(path)


def render_splash_pixmap(width: int, height: int, supersample: int = 2) -> QtGui.QPixmap:
    w, h = int(width * supersample), int(height * supersample)
    pix = QtGui.QPixmap(w, h)
    bg = QtGui.QColor(*theme.BLUE)
    fg = QtGui.QColor(*theme.YELLOW)
    outline = _contrast_outline(theme.BLUE)
    pix.fill(bg)

    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
    margin_x = w * 0.06
    word_rect = QtCore.QRectF(margin_x, h * 0.20, w - 2 * margin_x, h * 0.54)
    _paint_trashy_word(
        painter, "EllipSDF", word_rect, fg, outline,
        max_font_px=int(118 * supersample),
    )
    painter.end()
    return pix


def make_sdf_icon() -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(render_sdf_icon_pixmap(size))
    return icon


def render_sdf_icon_pixmap(size: int) -> QtGui.QPixmap:
    ss = 4
    S = int(size * ss)
    pix = QtGui.QPixmap(S, S)
    pix.fill(QtGui.QColor(*theme.BLUE))

    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
    rect = QtCore.QRectF(S * 0.015, S * 0.035, S * 0.97, S * 0.91)
    _paint_trashy_word(
        painter, "SDF", rect, QtGui.QColor(*theme.YELLOW),
        _contrast_outline(theme.BLUE), max_font_px=int(150 * ss),
        stretch_to_rect=True,
    )
    painter.end()
    return pix.scaled(size, size, QtCore.Qt.KeepAspectRatio,
                      QtCore.Qt.SmoothTransformation)
