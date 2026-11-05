"""Regressions for live pose snapshots and configurable base-size colors."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import app_settings
from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from main_window import MainWindow
from pose_correctives import PoseCorrectiveKey, PoseCorrectiveLibrary, PoseCorrectiveWorker
from native_morph import MorphSettings, NativeMorphDriver
import raymarch
from settings_dialog import SettingsDialog
from skeleton import Bone, Pose, Skeleton, mat4_compose
from viewer3d import (SceneViewer3D, RENDER_LIGHTING, RENDER_SOLID, RENDER_WIREFRAME,
                      build_concatenated_mesh, pose_fit_size_colors)
import primitive_gpu


def _rgba(hex_color):
    color = QtGui.QColor(hex_color)
    return np.array([color.redF(), color.greenF(), color.blueF()], np.float32)


class _FrameCollector(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.frames = []

    @QtCore.Slot(int, int, float, object, object, object, object)
    def collect(self, pose_index, step, loss, centers, radii, rotations, eps):
        self.frames.append((pose_index, step, loss, radii.copy()))


class PoseFitLivePreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_volume_classification_and_tolerance_preserve_ids(self):
        settings = app_settings.defaults()
        base = np.array([[0.2, 0.3, 0.4]] * 6, np.float32)
        radii = base.copy()
        radii[0] *= 1.1
        radii[1] *= 0.9
        radii[2] *= 1.002  # less than 1% volume growth
        radii[3] *= [2.0, 0.5, 1.0]  # axis changes, same volume
        radii[4, 0] *= 1.01  # exactly on the tolerance boundary
        radii[5, 0] *= 0.99
        colors = pose_fit_size_colors(radii, base, settings)
        np.testing.assert_allclose(colors[0, :3], _rgba(settings["pose_fit_growth_color"]))
        np.testing.assert_allclose(colors[1, :3], _rgba(settings["pose_fit_shrink_color"]))
        for index in (2, 3, 4, 5):
            np.testing.assert_allclose(
                colors[index, :3], _rgba(settings["pose_fit_neutral_color"]))
        # Model normalization changes the absolute radii, not the classification.
        np.testing.assert_allclose(
            pose_fit_size_colors(radii * 7.0, base * 7.0, settings), colors)
        settings["pose_fit_size_tolerance_pct"] = 0.0
        colors = pose_fit_size_colors(radii, base, settings)
        # A small visible morph starts tinting; it does not jump to full orange.
        neutral, growth = _rgba(settings["pose_fit_neutral_color"]), _rgba(settings["pose_fit_growth_color"])
        self.assertGreater(np.linalg.norm(colors[2, :3] - neutral), 0)
        self.assertLess(np.linalg.norm(colors[2, :3] - neutral), np.linalg.norm(growth - neutral) * 0.05)

    def test_colors_interpolate_between_neutral_and_both_endpoints(self):
        settings = dict(pose_fit_neutral_color="#808080", pose_fit_growth_color="#ff0000",
                        pose_fit_shrink_color="#0000ff", pose_fit_color_span_pct=25)
        base = np.ones((7, 3))
        radii = base.copy()
        radii[:, 0] = [1, 1.01, 1.135, 1.26, 0.99, 0.865, 0.74]
        neutral, growth, shrink = _rgba("#808080"), _rgba("#ff0000"), _rgba("#0000ff")
        expected = [neutral, neutral, (neutral + growth) / 2, growth,
                    neutral, (neutral + shrink) / 2, shrink]
        colors = pose_fit_size_colors(radii, base, settings)
        np.testing.assert_allclose(colors[:, :3], expected, atol=1e-6)
        np.testing.assert_allclose(colors[:, 3], colors[0, 3])
        settings["pose_fit_color_span_pct"] = 50
        slower = pose_fit_size_colors(radii, base, settings)
        np.testing.assert_allclose(slower[2, :3], neutral * 0.75 + growth * 0.25, atol=1e-6)

    def test_color_ramp_has_no_jump_and_handles_extreme_radii(self):
        base = np.ones((2, 3))
        radii = base.copy()
        radii[:, 0] = [1.01 - 1e-5, 1.01 + 1e-5]
        colors = pose_fit_size_colors(radii, base, {})
        self.assertLess(np.linalg.norm(colors[1, :3] - colors[0, :3]), 1e-4)
        np.testing.assert_allclose(pose_fit_size_colors(radii, base,
            {"pose_fit_color_span_pct": float("nan")}), colors)
        extreme_base = np.array([[1e-200] * 3, [1e200] * 3])
        extreme_radii = extreme_base[::-1].copy()
        colors = pose_fit_size_colors(extreme_radii, extreme_base, {})
        self.assertTrue(np.isfinite(colors).all())
        defaults = app_settings.defaults()
        np.testing.assert_allclose(colors[0, :3], _rgba(defaults["pose_fit_growth_color"]))
        np.testing.assert_allclose(colors[1, :3], _rgba(defaults["pose_fit_shrink_color"]))

    def test_morph_color_tracks_the_smoothed_displayed_size(self):
        skeleton = Skeleton([Bone("Root", 0)])
        base = BoneLocalEllipsoids(
            local_centers=np.zeros((1, 3), np.float32),
            local_radii=np.ones((1, 3), np.float32),
            local_rotations=np.array([[0, 0, 0, 1]], np.float32),
            bone_assignments=np.array([0], np.int32))
        target_pose = Pose("target", {0: mat4_compose([0, 0, 0],
            [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], [1, 1, 1])})
        key = PoseCorrectiveKey("target", np.zeros((1, 3), np.float32),
            np.array([[0, 0, 0, 1]], np.float32),
            np.array([[np.log(1.26), 0, 0]], np.float32), pose=target_pose)
        driver = NativeMorphDriver(skeleton, PoseCorrectiveLibrary(base, [key], Pose.t_pose()),
                                   MorphSettings(smoothing_time=0.5))
        initial = driver.evaluate(None, dt=0)
        halfway = driver.evaluate(target_pose, dt=0.5 * np.log(2) / np.log(10))
        final = driver.evaluate(target_pose, dt=2.5)
        before = pose_fit_size_colors(initial.local_radii, base.local_radii, {})[0, :3]
        middle = pose_fit_size_colors(halfway.local_radii, base.local_radii, {})[0, :3]
        after = pose_fit_size_colors(final.local_radii, base.local_radii, {})[0, :3]
        self.assertGreater(np.linalg.norm(middle - before), 0)
        self.assertLess(np.linalg.norm(middle - before), np.linalg.norm(after - before))
        np.testing.assert_allclose(after, _rgba(app_settings.defaults()["pose_fit_growth_color"]), atol=1e-6)

    def test_primitives_use_the_gloss_material_with_colored_instances(self):
        viewer = SceneViewer3D()
        try:
            viewer._ell_render_mode = RENDER_LIGHTING
            viewer.set_pose_fit_reference(np.ones((1, 3)))
            viewer.show_ellipsoids_fast(np.zeros((1, 3)), np.array([[1.135, 1, 1]]),
                                        np.array([[0, 0, 0, 1]]))
            item = viewer._ell_item
            self.assertIsInstance(item, primitive_gpu.PrimitiveInstancesItem)
            self.assertEqual(item.style, primitive_gpu.STYLE_LIT)
            expected = pose_fit_size_colors(viewer._ell_radii, np.ones((1, 3)), {})[0, :3]
            np.testing.assert_allclose(item.instances[0, primitive_gpu.COLOR][:3], expected)
        finally:
            viewer.widget.deleteLater()

    def test_smooth_render_buffers_keep_color_attached_to_each_primitive(self):
        viewer = SceneViewer3D()
        try:
            viewer._ell_render_mode = RENDER_LIGHTING
            centers = np.array([[-4, 0, 0], [0, 0, 0], [4, 0, 0]])
            base = np.ones((3, 3))
            radii = base.copy()
            radii[:, 0] = [.74, 1, 1.26]
            viewer.set_pose_fit_reference(base)
            rotations = np.tile([0, 0, 0, 1], (3, 1))
            item = None
            for displayed in (radii, radii[::-1].copy()):
                viewer.show_ellipsoids_fast(centers, displayed, rotations)
                if item is not None:
                    self.assertIs(viewer._ell_item, item)
                item = viewer._ell_item
                expected = pose_fit_size_colors(displayed, base, {})
                # The shader builds each instance's surface around its own
                # centre and colours it with its own row.
                points, _normals = primitive_gpu.surface_reference(
                    item.directions, item.instances)
                for i, center in enumerate(centers):
                    self.assertTrue(np.all(np.linalg.norm(points[i] - center, axis=1) < 1.5))
                    np.testing.assert_allclose(item.instances[i, primitive_gpu.COLOR][:3],
                                               expected[i, :3])
        finally:
            viewer.widget.deleteLater()

    def test_missing_disabled_and_invalid_reference(self):
        radii = np.ones((2, 3))
        self.assertIsNone(pose_fit_size_colors(radii, None, {}))
        self.assertIsNone(pose_fit_size_colors(radii, radii[:1], {}))
        self.assertIsNone(pose_fit_size_colors(
            radii, radii, {"pose_fit_color_changes": False}))
        invalid = np.array([[0.0, 1.0, 1.0], [np.nan, 1.0, 1.0]])
        colors = pose_fit_size_colors(radii, invalid, {})
        np.testing.assert_allclose(colors[:, :3], np.tile(_rgba("#a0a7b4"), (2, 1)))

    def test_color_picker_persistence_cancel_and_restore_defaults(self):
        settings = app_settings.defaults()
        dialog = SettingsDialog(settings)
        try:
            dialog._widgets["pose_fit_growth_color"].setValue("#00ff00")
            dialog._widgets["pose_fit_shrink_color"].setValue("#ff00ff")
            dialog._widgets["pose_fit_live_preview"].setChecked(False)
            dialog._widgets["pose_fit_size_tolerance_pct"].setValue(5.0)
            dialog._widgets["pose_fit_color_span_pct"].setValue(40.0)
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(app_settings, "_FILE", Path(tmp) / "settings.json"):
                    app_settings.save(dialog.values())
                    loaded = app_settings.load()
            self.assertEqual(loaded["pose_fit_growth_color"], "#00ff00")
            self.assertFalse(loaded["pose_fit_live_preview"])
            self.assertEqual(loaded["pose_fit_size_tolerance_pct"], 5.0)
            self.assertEqual(loaded["pose_fit_color_span_pct"], 40.0)
            # Reopening after Cancel reloads the saved working values.
            dialog._widgets["pose_fit_growth_color"].setValue("#ff0000")
            dialog.load_values(loaded)
            self.assertEqual(dialog.values()["pose_fit_growth_color"], "#00ff00")
            dialog._restore_defaults()
            self.assertEqual(dialog.values(), settings)
            self.assertEqual(MainWindow._optimizer_settings({
                key: value for key, value in loaded.items()
                if key.startswith("pose_fit_")}), {})
        finally:
            dialog.close()

    def test_reference_is_copied_and_settings_recolor_current_geometry(self):
        base = np.ones((2, 3), np.float32)
        owner = SimpleNamespace(
            _ell_centers=np.zeros((2, 3)), _ell_radii=base * [[1.1], [0.9]],
            _ell_base_colors=None, _ell_metric_values={}, _ell_metric_mode="default",
            _pose_fit_color_settings={}, _rebuild_ellipsoids=mock.Mock(),
        )
        owner._apply_ellipsoid_metric_colors = lambda: (
            SceneViewer3D._apply_ellipsoid_metric_colors(owner))
        SceneViewer3D.set_pose_fit_reference(owner, base)
        base *= 10
        np.testing.assert_allclose(owner._pose_fit_base_radii, np.ones((2, 3)))
        SceneViewer3D.configure_pose_fit_colors(owner, {"pose_fit_growth_color": "#00ff00"})
        np.testing.assert_allclose(owner._ell_colors[0, :3], [0, 1, 0])
        owner._rebuild_ellipsoids.assert_called_once()
        _verts, faces, face_colors = build_concatenated_mesh(
            owner._ell_centers, owner._ell_radii,
            np.tile([0, 0, 0, 1], (2, 1)), owner._ell_colors)
        self.assertEqual(len(faces), len(face_colors))
        np.testing.assert_allclose(face_colors[0], owner._ell_colors[0])
        np.testing.assert_allclose(face_colors[-1], owner._ell_colors[1])
        SceneViewer3D.set_pose_fit_reference(owner, None)
        self.assertIsNone(owner._ell_colors)

    def test_unity_batch_snapshots_reach_render_timer_and_final_frame(self):
        centers = np.zeros((2, 3), np.float32)
        radii = np.ones((2, 3), np.float32)
        rotations = np.tile([0, 0, 0, 1], (2, 1))
        for enabled in (True, False):
            owner = SimpleNamespace(
                _api_job_id="test", _api_stage="fit", _api_batch_pipeline=True,
                _api_fit_existing=True, _settings={"pose_fit_live_preview": enabled},
                _api_primitive_type="ellipsoid", _pending_visual=None,
            )
            self.assertEqual(MainWindow._api_pose_live_preview_enabled(owner), enabled)
            MainWindow._on_opt_step_visual(owner, 2, 0.2, centers, radii, rotations)
            self.assertIs(owner._api_last[1], radii)
            self.assertIs(owner._pending_visual[3], radii)

    def test_wireframe_preserves_per_primitive_colors_and_reuses_item(self):
        colors = np.array([[1, 0, 0, 1], [0, 0, 1, 1]], np.float32)
        owner = SimpleNamespace(
            _ell_centers=np.array([[0, 0, 0], [3, 0, 0]], np.float32),
            _ell_radii=np.ones((2, 3), np.float32),
            _ell_rotations=np.tile([0, 0, 0, 1], (2, 1)).astype(np.float32),
            _ell_colors=colors, _ell_item=None, _ell_item_mode=None,
            _ell_render_mode=RENDER_WIREFRAME, _show_ellipsoids=True,
            _align_stored_ellipsoid_population=lambda: None, _view=mock.Mock(),
        )
        SceneViewer3D._rebuild_ellipsoids(owner)
        wire = owner._ell_item
        self.assertIsInstance(wire, primitive_gpu.PrimitiveInstancesItem)
        self.assertEqual(wire.style, primitive_gpu.STYLE_WIRE)
        np.testing.assert_allclose(wire.instances[:, primitive_gpu.COLOR][:, :3], colors[:, :3])
        np.testing.assert_allclose(wire.instances[:, 17], 0.4)
        owner._ell_colors = colors[::-1]
        SceneViewer3D._rebuild_ellipsoids(owner)
        self.assertIs(owner._ell_item, wire)
        np.testing.assert_allclose(wire.instances[0, primitive_gpu.COLOR][:3], colors[1, :3])
        # One item serves every mode; only its style and alpha change.
        owner._ell_render_mode = RENDER_SOLID
        SceneViewer3D._rebuild_ellipsoids(owner)
        self.assertIs(owner._ell_item, wire)
        self.assertEqual(wire.style, primitive_gpu.STYLE_UNLIT)
        np.testing.assert_allclose(wire.instances[:, 17], 1.0)
        owner._ell_render_mode = RENDER_WIREFRAME
        SceneViewer3D._rebuild_ellipsoids(owner)
        self.assertEqual(wire.style, primitive_gpu.STYLE_WIRE)
        owner._view.addItem.assert_called_once_with(wire)

    def test_fit_pose_preview_setting_controls_optimizer_emissions(self):
        centers = np.zeros((1, 3), np.float32)
        radii = np.ones((1, 3), np.float32)
        rotations = np.array([[0, 0, 0, 1]], np.float32)
        eps = np.ones((1, 2), np.float32)
        for enabled in (True, False):
            calls = []
            owner = SimpleNamespace(
                _api_job_id=None, _api_initial_ellipsoids=(centers, radii, rotations, eps),
                _api_existing_local_parameterization=lambda: None,
                _api_batch_pipeline=True, _api_fit_existing=True,
                _api_primitive_type="ellipsoid", _api_stage="sdf", _api_options={},
                _settings={"pose_fit_live_preview": enabled}, _viewer=mock.Mock(),
                _shapes_by_id={}, _shape=mock.Mock(), _gather_fit_kwargs=lambda: {},
                start_optimization=lambda **kwargs: calls.append(kwargs),
            )
            MainWindow._api_start_existing_ellipsoid_fit(owner)
            self.assertEqual(calls[0]["parameter_options"]["emit_intermediate_progress"], enabled)
            owner._viewer.set_pose_fit_reference.assert_called_once_with(radii)
            owner._shape.render.assert_called_once()

    def test_corrective_gui_coalesces_frames_and_flushes_when_preview_disabled(self):
        shape = mock.Mock()
        owner = SimpleNamespace(
            _pending_pose_visual=None, _settings={"pose_fit_live_preview": True},
            _rig_panel=SimpleNamespace(bone_local=SimpleNamespace(primitive_type="ellipsoid"),
                                       active_poses=[object()]),
            _shapes_by_id={}, _shape=shape, _viewer=mock.Mock(), _opt_total_steps=10,
            _api_job_id=None, _status=mock.Mock(), _run_tracker=mock.Mock(),
            _progress_set=mock.Mock(),
        )
        centers = np.zeros((1, 3))
        radii = np.ones((1, 3))
        rotations = np.array([[0, 0, 0, 1]])
        eps = np.ones((1, 2))
        for step in range(5):
            MainWindow._on_pose_corrective_step_visual(
                owner, 0, step, 0.1, centers, radii * (1 + step), rotations, eps)
        shape.render.assert_not_called()
        MainWindow._flush_pose_corrective_visual(owner)
        shape.render.assert_called_once()
        np.testing.assert_allclose(shape.render.call_args.args[2], radii * 5)
        self.assertIsNone(owner._pending_pose_visual)
        shape.reset_mock()
        owner._settings["pose_fit_live_preview"] = False
        MainWindow._on_pose_corrective_step_visual(
            owner, 0, 7, 0.1, centers, radii, rotations, eps)
        MainWindow._flush_pose_corrective_visual(owner, force=True)
        shape.render.assert_called_once()

    def test_background_corrective_emits_before_optimizer_finishes(self):
        release = threading.Event()

        class PausingOptimizer(QtCore.QObject):
            step_visual = QtCore.Signal(int, float, object, object, object, object)

            def __init__(self, **kwargs):
                super().__init__()
                self._c = kwargs["initial_centers"].copy()
                self._r = kwargs["initial_radii"].copy()
                self._q = kwargs["initial_rotations"].copy()
                self.optimized_parameter_result = (self._c, self._r, self._q)

            def request_stop(self):
                release.set()

            def run(self):
                self.step_visual.emit(0, 1.0, self._c, self._r, self._q, None)
                release.wait(5.0)
                self.step_visual.emit(1, 0.5, self._c, self._r, self._q, None)

        skeleton = Skeleton([Bone(
            name="Root", index=0, parent_index=-1,
            local_bind_transform=np.eye(4), inverse_bind_matrix=np.eye(4))])
        base = BoneLocalEllipsoids(
            local_centers=np.zeros((1, 3), np.float32),
            local_radii=np.full((1, 3), 0.25, np.float32),
            local_rotations=np.array([[0, 0, 0, 1]], np.float32),
            bone_assignments=np.zeros(1, np.int32),
        )
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        rig = SimpleNamespace(skeleton=skeleton, vertices=vertices,
                              faces=np.array([[0, 1, 2]], np.int32))
        sdf = SimpleNamespace(grid=np.zeros((2, 2, 2), np.float32),
                              origin=np.zeros(3), dx=0.1, n=2, thickness=None)
        collector = _FrameCollector()
        with mock.patch("pose_correctives.SdfComputer") as comp, mock.patch(
                "pose_correctives.OptimizationWorker", PausingOptimizer):
            comp.return_value.compute_voxel_grid.return_value = sdf
            worker = PoseCorrectiveWorker(
                rigged_mesh=rig, mapper=BoneEllipsoidMapper(skeleton), base=base,
                poses=[Pose.t_pose()], target_vertices=[vertices], grid_n=2,
                margin=0.1, fit_kwargs={}, device="cpu")
            worker.pose_fit_progress.connect(collector.collect, QtCore.Qt.QueuedConnection)
            worker.start()
            try:
                deadline = time.monotonic() + 2.0
                while not collector.frames and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.005)
                self.assertTrue(worker.isRunning())
                self.assertEqual(len(collector.frames), 1)
                self.assertEqual(collector.frames[0][1], 0)
            finally:
                release.set()
                self.assertTrue(worker.wait(5000))
                self.app.processEvents()
            self.assertIsNotNone(worker.result)
            self.assertEqual(len(collector.frames), 2)

    def test_raymarch_uses_each_primitive_color_and_resets_to_uniform(self):
        centers = np.array([[-0.45, 0, 0], [0.45, 0, 0]], np.float32)
        radii = np.full((2, 3), 0.28, np.float32)
        rotations = np.tile([0, 0, 0, 1], (2, 1)).astype(np.float32)
        colors = np.array([[1, 0, 0, 1], [0, 0, 1, 1]], np.float32)
        rm = raymarch.Raymarcher()
        rm.update(centers, radii, rotations, colors=colors)
        inv_mvp = np.diag([1, 1, 3, 1]).astype(np.float32)
        img = rm.render(inv_mvp, 96, 48, base_rgb=(0, 1, 0))
        hit = img[:, :, 3] > 0
        self.assertTrue(hit[:, :48].any() and hit[:, 48:].any())
        left = img[:, :48, :3][hit[:, :48]].mean(axis=0)
        right = img[:, 48:, :3][hit[:, 48:]].mean(axis=0)
        self.assertGreater(left[0], left[2] + 80)
        self.assertGreater(right[2], right[0] + 80)
        rm.update(centers, radii, rotations)
        img = rm.render(inv_mvp, 96, 48, base_rgb=(0, 1, 0))
        surface = img[:, :, :3][img[:, :, 3] > 0].mean(axis=0)
        self.assertGreater(surface[1], surface[0] + 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
