"""Offscreen regression tests for superquadric option plumbing."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6 import QtWidgets  # noqa: E402

from main_window import MainWindow  # noqa: E402
from optimization import OptimizationWorker  # noqa: E402
from shape_plugins import BentSuperquadricShape, SuperquadricShape  # noqa: E402


class SuperquadricUiPlumbingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_each_mode_and_warmup_reaches_start_and_worker_interfaces(self) -> None:
        start_signature = inspect.signature(MainWindow.start_optimization)
        worker_parameters = inspect.signature(
            OptimizationWorker.__init__).parameters

        for shape_type in (SuperquadricShape, BentSuperquadricShape):
            shape = shape_type()
            shape.options_widget()
            for mode in ("per_primitive", "shared", "fixed"):
                with self.subTest(shape=shape_type.__name__, mode=mode):
                    index = shape._combo_eps_mode.findData(mode)
                    self.assertGreaterEqual(index, 0)
                    shape._combo_eps_mode.setCurrentIndex(index)
                    shape._spin_eps_warmup.setValue(35)
                    shape._spin_bend_warmup.setValue(55)

                    kwargs = shape.fit_kwargs()
                    # This is the exact expansion used by _gather_fit_kwargs;
                    # binding catches a missing MainWindow parameter before a
                    # real fit tries to start.
                    start_signature.bind(object(), **kwargs)
                    self.assertEqual(kwargs["sq_eps_mode"], mode)
                    self.assertEqual(kwargs["sq_unlock_frac"], 0.35)
                    self.assertEqual(kwargs["sq_bend_unlock_frac"], 0.55)
                    for name in (
                        "sq_eps_mode", "sq_unlock_frac",
                        "sq_bend_unlock_frac",
                    ):
                        self.assertIn(name, worker_parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
