from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_settings  # noqa: E402


class PanelSettingsPersistenceTest(unittest.TestCase):
    def test_legacy_superquadric_defaults_migrate_once(self) -> None:
        legacy = {
            "shape": "superquadric",
            "shared": {"max_steps": 6000},
            "shapes": {
                "superquadric": {
                    "eps1": 0.6,
                    "eps2": 0.6,
                    "eps_warmup": 20,
                    "local_fit": True,
                },
                "bent_superquadric": {
                    "eps1": 0.6,
                    "eps2": 0.6,
                    "eps_warmup": 20,
                    "bend_warmup": 40,
                },
                "ellipsoid": {"local_fit": False},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel_settings.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with mock.patch.object(app_settings, "_PANEL_FILE", path):
                migrated = app_settings.load_panel()
                second_load = app_settings.load_panel()

            self.assertEqual(
                migrated["schema_version"],
                app_settings.PANEL_SETTINGS_SCHEMA_VERSION,
            )
            for shape_id in ("superquadric", "bent_superquadric"):
                state = migrated["shapes"][shape_id]
                self.assertEqual(
                    (state["eps1"], state["eps2"], state["eps_warmup"]),
                    (1.0, 1.0, 5),
                )
            self.assertTrue(migrated["shapes"]["superquadric"]["local_fit"])
            self.assertEqual(
                migrated["shapes"]["bent_superquadric"]["bend_warmup"], 40)
            self.assertEqual(migrated["shapes"]["ellipsoid"], {"local_fit": False})
            self.assertEqual(second_load, migrated)

    def test_custom_or_already_versioned_values_are_not_rewritten(self) -> None:
        cases = (
            {
                "shapes": {"superquadric": {
                    "eps1": 0.7, "eps2": 0.6, "eps_warmup": 20,
                }},
            },
            {
                "schema_version": app_settings.PANEL_SETTINGS_SCHEMA_VERSION,
                "shapes": {"superquadric": {
                    "eps1": 0.6, "eps2": 0.6, "eps_warmup": 20,
                }},
            },
        )
        for index, payload in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "panel_settings.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with mock.patch.object(app_settings, "_PANEL_FILE", path):
                    loaded = app_settings.load_panel()
                state = loaded["shapes"]["superquadric"]
                expected = payload["shapes"]["superquadric"]
                self.assertEqual(state["eps1"], expected["eps1"])
                self.assertEqual(state["eps2"], expected["eps2"])
                self.assertEqual(state["eps_warmup"], expected["eps_warmup"])

    def test_legacy_voxel_blowup_migrates_to_local_thickness_fraction(
            self) -> None:
        payload = {
            "schema_version": 2,
            "shared": {"blowup": -2.0, "max_steps": 6000},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel_settings.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(app_settings, "_PANEL_FILE", path):
                loaded = app_settings.load_panel()
                persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            loaded["schema_version"],
            app_settings.PANEL_SETTINGS_SCHEMA_VERSION)
        self.assertNotIn("blowup", loaded["shared"])
        self.assertAlmostEqual(
            loaded["shared"]["blowup_fraction"], -0.05)
        self.assertEqual(persisted, loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
