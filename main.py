

import argparse
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from PySide6 import QtCore, QtGui, QtWidgets

import branding
import theme


# Logical splash size in device-independent pixels.  Everything is painted as
# vector geometry, so the wordmark stays sharp on high-DPI displays.
_SPLASH_W, _SPLASH_H = 600, 280


class SplashScreen(QtWidgets.QWidget):

    def __init__(self):
        super().__init__(
            None,
            QtCore.Qt.SplashScreen
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint,
        )
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self._display_progress = 0.0
        self._target_progress = 0.0
        self._status = "Loading …"
        self._finish_animation = None
        self.setFixedSize(_SPLASH_W, _SPLASH_H)

        self._word_path, self._word_bounds = self._build_word_path()

        # Use the screen under the pointer when possible, which is less
        # surprising on multi-monitor workstations than always using screen 1.
        scr = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if scr is None:
            scr = QtWidgets.QApplication.primaryScreen()
        if scr is not None:
            geo = scr.availableGeometry()
            self.move(geo.center().x() - _SPLASH_W // 2,
                      geo.center().y() - _SPLASH_H // 2)

    @staticmethod
    def _visible_accent() -> QtGui.QColor:
        accent = QtGui.QColor(*theme.YELLOW)
        # User-selectable colours may be almost black.  Keep the trace visible
        # without replacing the chosen hue.
        if accent.lightness() < 105:
            accent = accent.lighter(180)
        return accent

    def _build_word_path(self) -> tuple[QtGui.QPainterPath, QtCore.QRectF]:
        font = QtGui.QFont(branding.display_font_family())
        font.setPixelSize(86)
        font.setWeight(QtGui.QFont.Bold)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        raw = QtGui.QPainterPath()
        raw.addText(0.0, 0.0, font, "EllipSDF")
        bounds = raw.boundingRect()
        max_width = 450.0
        scale = min(1.0, max_width / max(1.0, bounds.width()))
        transform = QtGui.QTransform()
        transform.translate(
            (self.width() - bounds.width() * scale) * 0.5
            - bounds.x() * scale,
            145.0 - bounds.center().y() * scale,
        )
        transform.scale(scale, scale)
        path = transform.map(raw)
        return path, path.boundingRect()

    def set_progress(self, frac: float, message: str | None = None,
                     animate: bool = True, duration: float = 0.35) -> None:
        """Reveal progress smoothly with a bounded total animation budget.

        ``duration`` is the time for the complete 0-to-1 journey, not for each
        individual loading milestone.  Splitting startup into more callbacks
        therefore does not make startup progressively slower.
        """
        if message is not None:
            self._status = message
        target = max(0.0, min(1.0, float(frac)))
        self._target_progress = max(self._target_progress, target)
        start = self._display_progress
        distance = self._target_progress - start
        animation_ms = round(max(0.0, duration) * distance * 1000.0)
        if not animate or animation_ms < 12 or distance <= 0.0:
            self._display_progress = self._target_progress
            self.repaint()
            QtWidgets.QApplication.processEvents()
            return

        loop = QtCore.QEventLoop(self)
        animation = QtCore.QVariantAnimation(self)
        animation.setStartValue(start)
        animation.setEndValue(self._target_progress)
        animation.setDuration(animation_ms)
        animation.setEasingCurve(QtCore.QEasingCurve.InOutCubic)

        def update_progress(value) -> None:
            self._display_progress = float(value)
            self.update()

        animation.valueChanged.connect(update_progress)
        animation.finished.connect(loop.quit)
        animation.start()
        loop.exec()
        self._display_progress = self._target_progress
        self.repaint()
        QtWidgets.QApplication.processEvents()

    def finish(self) -> None:
        """Reveal the completed wordmark and fade over the ready main window."""
        self.set_progress(1.0, "Ready")
        self.raise_()
        fade = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        fade.setDuration(160)
        fade.setStartValue(1.0)
        fade.setEndValue(0.0)
        fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        fade.finished.connect(self.close)
        self._finish_animation = fade
        fade.start()

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        p.fillRect(self.rect(), QtGui.QColor(39, 40, 44))
        p.setPen(QtGui.QPen(QtGui.QColor(66, 67, 72), 1.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(0.5, 0.5,
                                 self.width() - 1.0, self.height() - 1.0))

        progress = max(0.0, min(1.0, self._display_progress))
        accent = self._visible_accent()

        # A quiet preview of the complete lettering remains in the background.
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(225, 226, 230, 31), 1.1,
                            QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                            QtCore.Qt.RoundJoin))
        p.drawPath(self._word_path)

        # Loading reveals the actual wordmark from left to right.  A soft edge
        # makes it feel drawn rather than abruptly cropped.
        reveal_x = (self._word_bounds.left()
                    + self._word_bounds.width() * progress)
        if progress > 0.0:
            p.save()
            p.setClipRect(QtCore.QRectF(
                self._word_bounds.left() - 8.0,
                self._word_bounds.top() - 10.0,
                max(0.0, reveal_x - self._word_bounds.left() + 8.0),
                self._word_bounds.height() + 20.0,
            ))
            word_gradient = QtGui.QLinearGradient(
                self._word_bounds.left(), 0.0,
                self._word_bounds.right(), 0.0)
            word_gradient.setColorAt(0.0, QtGui.QColor(239, 240, 243))
            word_gradient.setColorAt(0.66, QtGui.QColor(239, 240, 243))
            word_gradient.setColorAt(1.0, accent)
            p.setBrush(QtGui.QBrush(word_gradient))
            p.setPen(QtGui.QPen(QtGui.QColor(250, 250, 252, 185), 0.8,
                                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                QtCore.Qt.RoundJoin))
            p.drawPath(self._word_path)
            p.restore()

            if progress < 0.995:
                edge = QtGui.QLinearGradient(reveal_x - 18.0, 0.0,
                                             reveal_x + 5.0, 0.0)
                edge.setColorAt(0.0, QtGui.QColor(
                    accent.red(), accent.green(), accent.blue(), 0))
                edge.setColorAt(0.72, QtGui.QColor(
                    accent.red(), accent.green(), accent.blue(), 55))
                edge.setColorAt(1.0, QtGui.QColor(
                    accent.red(), accent.green(), accent.blue(), 0))
                p.fillRect(QtCore.QRectF(reveal_x - 18.0,
                                         self._word_bounds.top() - 12.0,
                                         23.0,
                                         self._word_bounds.height() + 24.0),
                           edge)

        # Restrained status text and a two-pixel progress line.
        if self._status:
            p.setPen(QtGui.QColor(158, 159, 165))
            sf = QtGui.QFont("Segoe UI", 9)
            p.setFont(sf)
            p.drawText(QtCore.QRect(30, self.height() - 62,
                                    self.width() - 60, 20),
                       QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                       self._status)

        p.setFont(QtGui.QFont("Segoe UI", 8))
        p.setPen(QtGui.QColor(128, 129, 135))
        p.drawText(QtCore.QRect(30, self.height() - 62,
                                self.width() - 60, 20),
                   QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                   f"{round(progress * 100):d}%")

        bar = QtCore.QRectF(30.0, self.height() - 30.0,
                            self.width() - 60.0, 2.0)
        p.fillRect(bar, QtGui.QColor(76, 77, 82))
        if progress > 0.0:
            fill = QtCore.QRectF(
                bar.left(), bar.top(), bar.width() * progress, bar.height())
            bar_gradient = QtGui.QLinearGradient(
                bar.left(), 0.0, bar.right(), 0.0)
            muted_accent = QtGui.QColor(accent)
            muted_accent.setAlpha(205)
            bar_gradient.setColorAt(0.0, QtGui.QColor(195, 196, 201))
            bar_gradient.setColorAt(1.0, muted_accent)
            p.fillRect(fill, bar_gradient)
        p.end()


def _make_splash() -> SplashScreen:
    return SplashScreen()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EllipSDF — Mesh → Ellipsoid SDF")
    parser.add_argument(
        "--server", action="store_true", default=True,
        help="Start the embedded HTTP API / Unity bridge. This is now the default.")
    parser.add_argument(
        "--no-server", dest="server", action="store_false",
        help="Do not start the embedded HTTP API / Unity bridge.")
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port for the HTTP API (default: 8765).")
    # Ignore unknown args so launching via tools that append extras won't crash.
    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()

    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "EllipSDF.MeshToEllipsoidSDF")
        except Exception:
            pass

    # Create the QApplication with bare PySide6 (cheap) and get the splash on
    # screen *before* touching the heavy imports.  Everything below the
    # splash.show() runs while the loading screen is already visible.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(branding.make_sdf_icon())

    # Apply the saved appearance mode (light / dark / sync-with-OS) before the
    # splash and any widgets are built, so everything starts in the right scheme.
    theme.apply_mode()

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
    win.setWindowIcon(branding.make_sdf_icon())
    win.resize(1400, 1000)

    # Start the Unity/HTTP API if requested, now that the window exists.
    if args.server:
        win.start_api_server(port=args.port)

    # Windowed fullscreen (maximised, not exclusive): fills the screen but keeps
    # the title bar/taskbar and — importantly — lets combo-box popups show on top.
    win.showMaximized()
    splash.finish()
    sys.exit(pg.exec())


if __name__ == "__main__":
    main()
