"""End-to-end test of detail zones in the real main window (headless).

A sphere with two thin mirrored rods is loaded as a static mesh.  The rods are
thinner than the zone threshold, so a full-object fit must
* plan two zones (one mirrored pair), carve them out of the main fit,
* fit the source zone in a parallel process (or in-process when disabled),
* merge the zone primitives (mirrored) into the result,
* and stop cleanly when the user presses Stop.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

import app_settings  # noqa: E402
import detail_zone_stage  # noqa: E402
import detail_zones as dz  # noqa: E402
from main_window import MainWindow  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _shape_mesh_file(directory: Path) -> Path:
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    parts = [sphere]
    for sign in (1.0, -1.0):
        rod = trimesh.creation.capsule(height=0.4, radius=0.02, count=[16, 16])
        rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        rod.apply_translation([sign * 0.65, 0.0, 0.0])
        parts.append(rod)
    path = directory / "rods.obj"
    trimesh.util.concatenate(parts).export(path)
    return path


def _wait(predicate, timeout=240.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        APP.processEvents()
        time.sleep(0.01)


class DetailZoneAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._patches = [
            mock.patch.object(app_settings, "save"),
            mock.patch.object(app_settings, "save_panel"),
            mock.patch.object(MainWindow, "_restore_workspace_layout", new=lambda self: False),
            mock.patch.object(MainWindow, "_save_workspace_layout", new=lambda self: None),
        ]
        for patch in cls._patches:
            patch.start()
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        (tmp / "meshes").mkdir()
        cls.mesh_path = _shape_mesh_file(tmp)
        cls.window = MainWindow(mesh_dir=tmp / "meshes")
        w = cls.window
        w._settings = dict(w._settings)
        w._settings.update(detail_zones=True, detail_zone_threshold_vox=4.0,
                           detail_zone_max_grid=160, use_sparse_sdf=True)
        w._spin_max_steps.setValue(300)
        w._spin_num_ellipsoids.setValue(12)
        w._spin_max_ellipsoids.setValue(30)
        w._mesh_sdf_panel.spin_n.setValue(64)
        w._chk_symmetry.setChecked(True)
        w._load_mesh(str(cls.mesh_path))
        _wait(lambda: w._sdf_worker is None and not w._sdf_finalize_active)

    @classmethod
    def tearDownClass(cls):
        cls.window._panel_save_timer.stop()
        cls.window.close()
        for patch in cls._patches:
            patch.stop()
        cls._tmp.cleanup()

    def _fit(self, **settings):
        w = self.window
        w._settings.update(settings)
        plans = []
        original = detail_zone_stage.plan_full_fit

        def recording(window, num, max_count):
            out = original(window, num, max_count)
            plans.append(out[2])
            return out
        with mock.patch.object(detail_zone_stage, "plan_full_fit", side_effect=recording):
            w._start_full_object_fit()
        return plans[0]

    def _wait_done(self):
        w = self.window
        _wait(lambda: w._opt_worker is None and w._detail_zone_stage is None)

    def _check_merged(self, plan):
        w = self.window
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.zones), 2)
        self.assertEqual(len(plan.jobs), 1, "mirrored pair must be fitted once")
        centers = np.asarray(w._ellipsoids.centers)
        x_axis = np.argmax(np.abs(plan.zones[0].center))
        rod_left = centers[:, x_axis] < -0.7
        rod_right = centers[:, x_axis] > 0.7
        self.assertGreater(int(rod_left.sum()), 0)
        self.assertEqual(int(rod_left.sum()), int(rod_right.sum()), "zone mirrored to both rods")
        self.assertLessEqual(len(centers), plan.total_max)
        shape = w._last_visual_shape
        self.assertIsNotNone(shape)
        self.assertEqual(len(shape), len(centers))
        self.assertTrue(w._btn_fit.isEnabled())

    def test_parallel_zones_merge(self):
        plan = self._fit(detail_zone_processes=True)
        self.assertIsNotNone(self.window._detail_zone_stage, "zones start with the main fit")
        self._wait_done()
        self._check_merged(plan)

    def test_in_process_zones_merge(self):
        plan = self._fit(detail_zone_processes=False)
        self._wait_done()
        self._check_merged(plan)

    def test_stop_keeps_main_fit(self):
        w = self.window
        self._fit(detail_zone_processes=True)
        stage = w._detail_zone_stage
        self.assertIsNotNone(stage)
        _wait(lambda: any(p.latest is not None for p in stage._processes), timeout=120)
        w._on_stop_clicked()
        self.assertIsNone(w._detail_zone_stage)
        self.assertFalse(stage.active)
        for process in stage._processes:
            self.assertEqual(process._process.state(), QtCore.QProcess.ProcessState.NotRunning)
        self.assertIsNone(w._opt_worker)

    def test_zones_disabled_is_plain_fit(self):
        plan = self._fit(detail_zones=False)
        self.assertIsNone(plan)
        self.assertIsNone(self.window._detail_zone_stage)
        self._wait_done()


if __name__ == "__main__":
    unittest.main()
