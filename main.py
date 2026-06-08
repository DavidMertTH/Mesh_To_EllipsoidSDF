

import os
import sys
import time
from pathlib import Path

# When launched via Start.bat (pythonw.exe, double-click) there is no console,
# so sys.stdout / sys.stderr are None.  Libraries imported below (e.g. Warp)
# print startup banners, and writing to a None stream raises AttributeError and
# crashes the app silently.  Redirect those streams to a sink so the writes are
# harmless.  No-op when a real console is attached (running via python.exe).
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from PySide6 import QtCore, QtGui, QtWidgets

import theme

# NB: pyqtgraph is *not* imported here.  It pulls in numpy + its own modules
# (~0.5 s on top of Qt) and is only needed once we build MainWindow — so we
# defer it until after the splash is on screen, letting the loading screen
# pop up about half a second sooner.

# 'Syne' is the display/heading font used on davidmertth.github.io.  Bundled as
# a TTF so we don't depend on it being installed system-wide.
_FONT_FILE = Path(__file__).with_name("assets") / "fonts" / "Syne-Bold.ttf"


def _load_display_font() -> str:
    """Register the bundled Syne font and return its family name.

    Falls back to a generic bold sans family name if the file is missing or
    Qt refuses to load it.
    """
    fid = QtGui.QFontDatabase.addApplicationFont(str(_FONT_FILE))
    if fid != -1:
        families = QtGui.QFontDatabase.applicationFontFamilies(fid)
        if families:
            return families[0]
    return "Segoe UI"


# Logical splash size (device-independent pixels); rendered at SS× for crispness.
_SPLASH_W, _SPLASH_H = 460, 240
_SS = 2          # supersampling factor for the rendered pixmaps
_BLUR_RADIUS = 14  # blur strength (in supersampled px) of the not-yet-loaded part


def _render_splash_pixmap() -> QtGui.QPixmap:
    """Render the sharp 'EllipSDF' splash artwork to a supersampled pixmap."""
    w, h = _SPLASH_W * _SS, _SPLASH_H * _SS
    pix = QtGui.QPixmap(w, h)
    dark = theme.is_dark_mode()
    pix.fill(QtGui.QColor(20, 22, 30) if dark else QtGui.QColor(245, 246, 250))

    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

    # "Ellip" in blue, "SDF" in yellow — Syne (the site's heading font).
    blue = QtGui.QColor(*theme.BLUE)
    yellow = QtGui.QColor(*theme.YELLOW)

    family = _load_display_font()
    font = QtGui.QFont(family, 38 * _SS, QtGui.QFont.Bold)
    painter.setFont(font)
    fm = QtGui.QFontMetrics(font)
    w1 = fm.horizontalAdvance("Ellip")
    w2 = fm.horizontalAdvance("SDF")
    x = (w - (w1 + w2)) // 2
    y = h // 2

    painter.setPen(blue)
    painter.drawText(x, y, "Ellip")
    painter.setPen(yellow)
    painter.drawText(x + w1, y, "SDF")
    # NB: the loading status line is *not* baked in here — it changes per step
    # and is drawn live in SplashScreen.paintEvent.
    painter.end()
    return pix


def _blur_pixmap(src: QtGui.QPixmap, radius: float) -> QtGui.QPixmap:
    """Return a Gaussian-blurred copy of *src* (via QGraphicsBlurEffect)."""
    scene = QtWidgets.QGraphicsScene()
    item = QtWidgets.QGraphicsPixmapItem(src)
    effect = QtWidgets.QGraphicsBlurEffect()
    effect.setBlurRadius(radius)
    item.setGraphicsEffect(effect)
    scene.addItem(item)

    out = QtGui.QImage(src.size(), QtGui.QImage.Format_ARGB32_Premultiplied)
    out.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(out)
    rect = QtCore.QRectF(0, 0, src.width(), src.height())
    scene.render(painter, rect, rect)
    painter.end()
    return QtGui.QPixmap.fromImage(out)


class SplashScreen(QtWidgets.QWidget):
    """Frameless loading screen whose progress bar doubles as a sharpness mask.

    The artwork is blurry to the right of the bar and crisp to its left.  As
    ``set_progress`` advances the bar from left → right, more of the image comes
    into focus, until at 100 % the whole splash is sharp.
    """

    def __init__(self, sharp: QtGui.QPixmap, blurry: QtGui.QPixmap):
        super().__init__(
            None,
            QtCore.Qt.SplashScreen
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint,
        )
        self._sharp = sharp
        self._blurry = blurry
        self._progress = 0.0
        self._status = "Loading …"
        self.setFixedSize(_SPLASH_W, _SPLASH_H)
        # Centre on the primary screen.
        scr = QtWidgets.QApplication.primaryScreen()
        if scr is not None:
            geo = scr.availableGeometry()
            self.move(geo.center().x() - _SPLASH_W // 2,
                      geo.center().y() - _SPLASH_H // 2)

    def set_progress(self, frac: float, message: str | None = None,
                     animate: bool = True, duration: float = 0.35) -> None:
        """Advance the bar to *frac* (0..1), optionally updating the status line.

        When *animate* is set, the bar glides from its current position to the
        target with an ease-out curve instead of snapping there — so the jumps
        between loading milestones look smooth.  The main thread is blocked
        during the heavy work between milestones, so each call plays as a short
        "catch-up" glide at that milestone.
        """
        if message is not None:
            self._status = message
        target = max(0.0, min(1.0, float(frac)))
        start = self._progress
        # Snap for backwards moves or when animation is disabled / pointless.
        if not animate or duration <= 0 or target <= start:
            self._progress = target
            self.repaint()
            QtWidgets.QApplication.processEvents()
            return

        t0 = time.perf_counter()
        while True:
            t = (time.perf_counter() - t0) / duration
            if t >= 1.0:
                break
            eased = 1.0 - (1.0 - t) ** 3        # ease-out cubic
            self._progress = start + (target - start) * eased
            self.repaint()
            QtWidgets.QApplication.processEvents()
            time.sleep(1.0 / 120.0)             # ~120 fps cap, spares the CPU
        self._progress = target
        self.repaint()
        QtWidgets.QApplication.processEvents()

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # 1) blurry everywhere
        p.drawPixmap(self.rect(), self._blurry)

        # 2) sharp on the already-loaded (left) portion
        reveal = int(round(w * self._progress))
        if reveal > 0:
            tgt = QtCore.QRect(0, 0, reveal, h)
            src = QtCore.QRect(0, 0, reveal * _SS, h * _SS)
            p.drawPixmap(tgt, self._sharp, src)

        # 3) the bar's leading edge — a bright vertical line (the brand yellow)
        if 0.0 < self._progress < 1.0:
            edge = QtGui.QColor(*theme.YELLOW)
            p.fillRect(QtCore.QRect(reveal - 2, 0, 3, h), edge)

        # 4) live status line at the bottom — what's loading right now
        if self._status:
            dark = theme.is_dark_mode()
            p.setPen(QtGui.QColor(170, 172, 182) if dark
                     else QtGui.QColor(110, 112, 122))
            sf = QtGui.QFont("Segoe UI", 9)
            p.setFont(sf)
            p.drawText(QtCore.QRect(0, h - 30, w, 22),
                       QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                       self._status)
        p.end()


def _make_splash() -> SplashScreen:
    sharp = _render_splash_pixmap()
    blurry = _blur_pixmap(sharp, _BLUR_RADIUS * _SS)
    return SplashScreen(sharp, blurry)


def main():
    # Create the QApplication with bare PySide6 (cheap) and get the splash on
    # screen *before* touching the heavy imports.  Everything below the
    # splash.show() runs while the loading screen is already visible.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    splash = _make_splash()
    splash.show()
    splash.set_progress(0.03, "Loading modules …")

    # Now the slow imports — pyqtgraph (+numpy) first, then warp/torch via
    # MainWindow.  The bar advances around them; the splash is already up.
    import pyqtgraph as pg
    pg.mkQApp()  # registers the existing QApplication with pyqtgraph
    # Theme pyqtgraph (axis text, plot/image backgrounds) to match light/dark
    # mode.  Must run before any pg widget is created (i.e. before MainWindow).
    if theme.is_dark_mode():
        pg.setConfigOptions(foreground="d", background="k")
    else:
        pg.setConfigOptions(foreground="k", background="w")

    from main_window import MainWindow
    splash.set_progress(0.35, "Modules loaded")

    # Stop the mouse wheel from accidentally editing spin boxes / combo boxes /
    # sliders while scrolling the settings panels.  Kept as an attribute so the
    # filter object outlives this function.  (Imported here, after the heavy
    # modules, so it doesn't delay the splash.)
    from widgets import WheelGuard
    app._wheel_guard = WheelGuard(app)
    app.installEventFilter(app._wheel_guard)

    win = MainWindow(
        progress=lambda f, m="": splash.set_progress(0.35 + 0.62 * f, m or None))
    win.resize(1400, 1000)
    # Windowed fullscreen (maximised, not exclusive): fills the screen but keeps
    # the title bar/taskbar and — importantly — lets combo-box popups show on top.
    win.showMaximized()
    splash.set_progress(1.0, "Done")   # glide to fully sharp; bar covers the screen
    time.sleep(0.20)           # brief hold so the crisp logo is visible
    QtWidgets.QApplication.processEvents()
    splash.close()
    sys.exit(pg.exec())


if __name__ == "__main__":
    main()