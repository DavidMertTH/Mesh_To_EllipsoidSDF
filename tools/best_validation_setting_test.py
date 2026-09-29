"""Regression tests for the selectable best-validation final result."""

from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_settings  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402
from main_window import MainWindow  # noqa: E402
from settings_dialog import SettingsDialog  # noqa: E402


class BestValidationSettingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_default_and_checkbox_are_enabled(self) -> None:
        defaults = app_settings.defaults()
        self.assertIs(defaults["use_best_validation_result"], True)

        dialog = SettingsDialog(defaults)
        try:
            checkbox = dialog._widgets["use_best_validation_result"]
            self.assertIsInstance(checkbox, QtWidgets.QCheckBox)
            self.assertTrue(checkbox.isChecked())
        finally:
            dialog.close()

    def test_legacy_settings_migrate_to_historical_true_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app_settings.json"
            path.write_text(json.dumps({"sample_budget": 2048}), encoding="utf-8")
            with mock.patch.object(app_settings, "_FILE", path):
                loaded = app_settings.load()
                persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertIs(loaded["use_best_validation_result"], True)
        self.assertIs(persisted["use_best_validation_result"], True)

    def test_disabled_value_round_trips_and_reaches_worker_filter(self) -> None:
        values = app_settings.defaults()
        values["use_best_validation_result"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app_settings.json"
            with mock.patch.object(app_settings, "_FILE", path):
                app_settings.save(values)
                loaded = app_settings.load()

        self.assertIs(loaded["use_best_validation_result"], False)
        self.assertEqual(
            MainWindow._optimizer_settings({
                "use_best_validation_result": False,
            }),
            {"use_best_validation_result": False},
        )
        start_parameter = inspect.signature(
            MainWindow.start_optimization).parameters[
                "use_best_validation_result"]
        self.assertIs(start_parameter.default, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
