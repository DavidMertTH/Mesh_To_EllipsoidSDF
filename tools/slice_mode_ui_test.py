"""Focused UI contracts for the generalized Slice controls."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

import widgets  # noqa: E402
from viewer3d import ViewportOverlay  # noqa: E402


class SliceModeUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_viewport_slice_has_sdf_and_color_coded_error_modes(self) -> None:
        overlay = ViewportOverlay()
        self.assertEqual(overlay._chk_slice.text(), "Slice")
        self.assertEqual(
            [
                overlay._combo_slice_mode.itemData(i)
                for i in range(overlay._combo_slice_mode.count())
            ],
            ["sdf", "error"],
        )
        self.assertEqual(
            overlay._combo_slice_mode.itemText(1),
            "Error (Color Coded)",
        )
        self.assertFalse(overlay._combo_slice_sdf_source.isHidden())

        overlay._combo_slice_mode.setCurrentIndex(1)
        overlay._on_slice_mode_activated(1)
        self.assertEqual(overlay.slice_source(), "difference")
        self.assertTrue(overlay._combo_slice_sdf_source.isHidden())

    def test_slice_panel_playback_advances_and_wraps(self) -> None:
        panel = widgets.SdfSlicePanel(default_n=4)
        panel.set_sdf(np.ones((4, 4, 4), dtype=np.float32), dx=0.25)
        self.assertTrue(panel._slice_play_button.isEnabled())

        panel.slider_z.setValue(panel.slider_z.maximum())
        panel._advance_slice_animation()
        self.assertEqual(panel.slider_z.value(), panel.slider_z.minimum())

        panel._toggle_slice_animation()
        self.assertTrue(panel._slice_animation_timer.isActive())
        panel._toggle_slice_animation()
        self.assertFalse(panel._slice_animation_timer.isActive())

    def test_slice_panel_error_mode_uses_live_fit_and_rgba_ramp(self) -> None:
        panel = widgets.SdfSlicePanel(default_n=4)
        mesh_sdf = np.zeros((4, 4, 4), dtype=np.float32)
        mesh_sdf[2] = np.linspace(
            1.0, -1.0, 16, dtype=np.float32).reshape(4, 4).T
        panel.set_sdf(mesh_sdf, dx=0.25)
        params = (
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        )
        fake_ellipsoid_sdf = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        with mock.patch.object(
            widgets.slice_module,
            "ellipsoid_slice_sdf",
            return_value=fake_ellipsoid_sdf,
        ):
            panel.set_error_source_provider(lambda: params)
            panel._combo_slice_mode.setCurrentIndex(1)
            panel._update_slice()

        image = panel.img_xy.getImageItem().image
        self.assertEqual(image.shape, (4, 4, 4))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[0, 0, :3], widgets.theme.YELLOW)
        np.testing.assert_array_equal(image[-1, -1, :3], widgets.theme.BLUE)
        self.assertGreater(int(image[0, 0, 3]), 0)
        self.assertGreater(int(image[-1, -1, 3]), 0)
        self.assertFalse(panel._slice_mode_hint.isHidden())
        self.assertIn(
            "under-coverage", panel._slice_mode_hint.text().lower())

    def test_error_colors_only_the_symmetric_difference(self) -> None:
        mesh = np.array([[-1.0, -1.0, 1.0, 1.0]], dtype=np.float32)
        ellipsoid = np.array([[-1.0, 1.0, -1.0, 1.0]], dtype=np.float32)
        rgba = widgets.slice_module.slice_rgba_error(
            ellipsoid, mesh, widgets.theme.BLUE, widgets.theme.YELLOW)

        self.assertEqual(int(rgba[0, 0, 3]), 0)  # both inside
        np.testing.assert_array_equal(rgba[0, 1, :3], widgets.theme.BLUE)
        self.assertGreater(int(rgba[0, 1, 3]), 0)  # under-coverage
        np.testing.assert_array_equal(rgba[0, 2, :3], widgets.theme.YELLOW)
        self.assertGreater(int(rgba[0, 2, 3]), 0)  # over-coverage
        self.assertEqual(int(rgba[0, 3, 3]), 0)  # both outside


if __name__ == "__main__":
    unittest.main()
