"""Settings, neighbour seeds, and real multi-pose fits obey all channel locks."""
from __future__ import annotations

import copy
import itertools
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtWidgets

import app_settings
from bone_ellipsoid_mapper import BoneEllipsoidMapper
from ellipsoid import best_device
from joint_pose import JointPoseEdits
from main_window import MainWindow
from pose_correctives import PoseCorrectiveWorker, _gather_local_seed
from settings_dialog import SettingsDialog
from skeleton import Pose
from tools.native_pose_ui_test import make_rig, make_library


SETTINGS_TO_FLAGS = (
    ("pose_fit_position", "optimize_centers", "local_centers"),
    ("pose_fit_rotation", "optimize_rotations", "local_rotations"),
    ("pose_fit_scale", "optimize_radii", "local_radii"),
    ("pose_fit_exponents", "optimize_shape_exponents", "shape_exponents"),
)


class NativePoseRefitSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_native_settings_override_generic_shape_fit_options(self):
        rig = make_rig()
        base = make_library(rig).base
        for flags in itertools.product((False, True), repeat=4):
            settings = dict(zip((entry[0] for entry in SETTINGS_TO_FLAGS), flags))
            owner = SimpleNamespace(
                _rig_panel=SimpleNamespace(bone_local=base, morph_steps=7), _settings=settings,
                _gather_fit_kwargs=lambda: {"num_steps": 5, "sq_eps_mode": "per_primitive",
                    **{flag: True for _, flag, _ in SETTINGS_TO_FLAGS}},
                _api_canonical_primitive_type=MainWindow._api_canonical_primitive_type,
                _optimizer_settings=MainWindow._optimizer_settings,
                _api_job_id=None, _api_options={key: True for key, _, _ in SETTINGS_TO_FLAGS},
                _report_every=1, _spin_lr_init=SimpleNamespace(value=lambda: .01),
                _spin_lr_final=SimpleNamespace(value=lambda: .001))
            resolved = MainWindow._pose_corrective_fit_kwargs(owner)
            self.assertEqual(tuple(resolved[flag] for _, flag, _ in SETTINGS_TO_FLAGS), flags)
            # Native morph-pose fits use their own step budget.
            self.assertEqual(resolved["num_steps"], 7)

    def test_api_correctives_keep_the_request_step_budget(self):
        base = make_library(make_rig()).base
        owner = SimpleNamespace(
            _rig_panel=SimpleNamespace(bone_local=base, morph_steps=7), _settings={},
            _gather_fit_kwargs=lambda: {"num_steps": 5, "sq_eps_mode": "per_primitive"},
            _api_canonical_primitive_type=MainWindow._api_canonical_primitive_type,
            _optimizer_settings=MainWindow._optimizer_settings,
            _api_job_id="unity job", _api_options={},
            _report_every=1, _spin_lr_init=SimpleNamespace(value=lambda: .01),
            _spin_lr_final=SimpleNamespace(value=lambda: .001))
        self.assertEqual(MainWindow._pose_corrective_fit_kwargs(owner)["num_steps"], 5)

    def test_exponent_setting_persists_and_accepts_api_override(self):
        dialog = SettingsDialog(app_settings.defaults())
        try:
            self.assertTrue(dialog._widgets["pose_fit_exponents"].isChecked())
            dialog._widgets["pose_fit_exponents"].setChecked(False)
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(app_settings, "_FILE", Path(tmp) / "settings.json"):
                app_settings.save(dialog.values())
                self.assertFalse(app_settings.load()["pose_fit_exponents"])
            result = MainWindow._api_pose_fit_transform_flags(dialog.values(), {})
            self.assertFalse(result["optimize_shape_exponents"])
            result = MainWindow._api_pose_fit_transform_flags(dialog.values(), {"pose_fit_exponents": True})
            self.assertTrue(result["optimize_shape_exponents"])
            with self.assertRaisesRegex(ValueError, "options.pose_fit_exponents"):
                MainWindow._api_pose_fit_transform_flags({}, {"pose_fit_exponents": "false"})
        finally:
            dialog.deleteLater()
            self.app.processEvents()

    def test_neighbour_seed_cannot_change_disabled_channels(self):
        rig = make_rig()
        library = make_library(rig)
        neighbour = library.corrected_bone_local(library.keys[0])
        for flags in itertools.product((False, True), repeat=4):
            seed = _gather_local_seed(library.base, [library.base, neighbour],
                                      np.ones(library.base.num_ellipsoids, dtype=int))
            worker = SimpleNamespace(_base=library.base,
                _fit_kwargs=dict(zip((entry[1] for entry in SETTINGS_TO_FLAGS), flags)))
            NativePoseRefitSettingsTest._apply_locks(worker, seed)
            for enabled, (_, _flag, attribute) in zip(flags, SETTINGS_TO_FLAGS):
                if enabled:
                    self.assertFalse(np.array_equal(getattr(seed, attribute), getattr(library.base, attribute)))
                else:
                    np.testing.assert_array_equal(getattr(seed, attribute), getattr(library.base, attribute))

    @staticmethod
    def _apply_locks(worker, local):
        PoseCorrectiveWorker._apply_refit_locks(worker, local)

    def test_real_multi_pose_fits_keep_disabled_parameters_exactly_at_base(self):
        rig = make_rig()
        base = make_library(rig).base
        edits = [JointPoseEdits(rig.skeleton, Pose.t_pose()) for _ in range(3)]
        for edit, angle in zip(edits, (-30, -40, -50)):
            edit.set_axis(1, 2, angle)
        poses = [edit.apply() for edit in edits]
        # All frozen, each parameter individually enabled, and all enabled.
        # Multiple nearby poses exercise the neighbour seed, not only pose 0.
        cases = [(False,) * 4, *[tuple(i == j for i in range(4)) for j in range(4)], (True,) * 4]
        for flags in cases:
            with self.subTest(flags=flags):
                failures, targets, frames = [], [], []
                worker = PoseCorrectiveWorker(rigged_mesh=rig,
                    mapper=BoneEllipsoidMapper(rig.skeleton), base=base, poses=poses,
                    grid_n=8, margin=.12, device=best_device(),
                    fit_kwargs={"num_steps": 5, "report_every": 1, "sample_budget": 128,
                        "maintenance_every": 0, "superfit": False, "local_fit": False,
                        "lr_init": .01, "lr_final": .01, "primitive_shape": "superquadric",
                        "sq_eps_mode": "per_primitive", "sq_unlock_frac": 0,
                        **dict(zip((entry[1] for entry in SETTINGS_TO_FLAGS), flags))})
                worker.failed.connect(failures.append)
                worker.pose_target_visual.connect(lambda *args: targets.append(args))
                worker.pose_fit_progress.connect(lambda *args: frames.append(args))
                expected = copy.deepcopy(base)
                worker.run()
                self.assertEqual(failures, [])
                self.assertIsNotNone(worker.result)
                self.assertEqual(len(targets), 3)
                self.assertEqual(len(worker.result.keys), 3)
                self.assertTrue(frames)
                deltas = ("delta_centers", "delta_rotations", "delta_log_radii", "delta_log_shape_exponents")
                for key in worker.result.keys:
                    for enabled, attribute in zip(flags, deltas):
                        if not enabled:
                            np.testing.assert_array_equal(getattr(key, attribute),
                                np.array([[0, 0, 0, 1]]) if attribute == "delta_rotations" else 0)
                for _, _flag, attribute in SETTINGS_TO_FLAGS:
                    np.testing.assert_array_equal(getattr(base, attribute), getattr(expected, attribute))
                if not flags[3]:
                    for frame in frames:
                        np.testing.assert_array_equal(frame[-1], base.shape_exponents)
                else:
                    self.assertTrue(any(np.max(np.abs(key.delta_log_shape_exponents)) > 1e-6
                        for key in worker.result.keys), "Enabled exponents must actually train")
                if not flags[0]:
                    for target in targets:
                        index, pose = target[0], target[-1]
                        expected_center = worker._mapper.local_to_world_np(base, pose)[0]
                        np.testing.assert_allclose(target[4], expected_center, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
