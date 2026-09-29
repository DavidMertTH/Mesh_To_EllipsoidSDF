"""Headless contracts for the Unity-style dockable workspace views.

The real ``MainWindow`` constructor initializes Warp, loads a mesh and starts
background work.  These tests intentionally exercise its dock/menu helpers on
a tiny ``QMainWindow`` harness instead.  That keeps the test deterministic
while still running the production methods and real Qt dock widgets.
"""

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

from PySide6 import QtCore, QtWidgets  # noqa: E402

from main_window import MainWindow  # noqa: E402


VIEWS = (
    ("scene", "Scene"),
    ("sdf", "Slice"),
    ("mesh", "Mesh"),
    ("dashboard", "Dashboard"),
    ("runs", "Runs"),
    ("options", "Options"),
)
VIEW_TITLES = tuple(title for _key, title in VIEWS)


class _WorkspaceHarness(QtWidgets.QMainWindow):
    """Run the production dock helpers without constructing the full app."""

    _make_workspace_dock = MainWindow._make_workspace_dock
    _sync_workspace_dock_title_bar = (
        MainWindow._sync_workspace_dock_title_bar)
    _style_workspace_dock_tabs = MainWindow._style_workspace_dock_tabs
    _close_workspace_dock_tab = MainWindow._close_workspace_dock_tab
    _build_view_menu = MainWindow._build_view_menu

    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 0

    def _schedule_workspace_dock_tab_style(self) -> None:
        # Styling is independent of the docking behavior tested here.
        pass

    def _reset_workspace_layout(self) -> None:
        self.reset_count += 1

    def add_test_views(self) -> None:
        self._dock_widgets = {}
        for index, (key, title) in enumerate(VIEWS):
            dock = self._make_workspace_dock(
                title,
                f"TestDock{title}",
                QtWidgets.QLabel(title),
            )
            self._dock_widgets[key] = dock
            area = (
                QtCore.Qt.LeftDockWidgetArea
                if index == 0 else QtCore.Qt.RightDockWidgetArea
            )
            self.addDockWidget(area, dock)


class DockableViewsUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication([])
        )

    def setUp(self) -> None:
        self.window = _WorkspaceHarness()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_every_view_can_move_float_and_close(self) -> None:
        self.window.add_test_views()

        for dock in self.window._dock_widgets.values():
            with self.subTest(view=dock.windowTitle()):
                self.assertEqual(
                    dock.allowedAreas(), QtCore.Qt.AllDockWidgetAreas)
                features = dock.features()
                self.assertTrue(
                    features & QtWidgets.QDockWidget.DockWidgetMovable)
                self.assertTrue(
                    features & QtWidgets.QDockWidget.DockWidgetFloatable)
                self.assertTrue(
                    features & QtWidgets.QDockWidget.DockWidgetClosable)

        scene = self.window._dock_widgets["scene"]
        self.window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, scene)
        self.assertEqual(
            self.window.dockWidgetArea(scene),
            QtCore.Qt.BottomDockWidgetArea,
        )

        slice_dock = self.window._dock_widgets["sdf"]
        mesh_dock = self.window._dock_widgets["mesh"]
        self.window.tabifyDockWidget(slice_dock, mesh_dock)
        self.assertIn(mesh_dock, self.window.tabifiedDockWidgets(slice_dock))

    def test_mesh_can_be_dragged_from_slice_tabs_to_runs_tabs(self) -> None:
        """Mesh must expose a draggable tab and be accepted by the lower group.

        ``tabifyDockWidget`` models the layout transition that Qt performs
        after a successful mouse drop.  Keeping the visible-tab assertion
        separate from that transition makes this deterministic on the
        offscreen Qt platform, where native drag-and-drop is not available.
        """
        self.window.setDockOptions(
            self.window.dockOptions()
            | QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.GroupedDragging
        )
        self.window.add_test_views()
        slice_dock = self.window._dock_widgets["sdf"]
        mesh_dock = self.window._dock_widgets["mesh"]
        dashboard_dock = self.window._dock_widgets["dashboard"]
        runs_dock = self.window._dock_widgets["runs"]

        self.window.addDockWidget(
            QtCore.Qt.BottomDockWidgetArea, dashboard_dock)
        self.window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, runs_dock)
        self.window.tabifyDockWidget(slice_dock, mesh_dock)
        self.window.tabifyDockWidget(dashboard_dock, runs_dock)
        mesh_dock.raise_()
        self.window.show()
        self.app.processEvents()
        self.window._style_workspace_dock_tabs()
        self.app.processEvents()

        self.assertIn(mesh_dock, self.window.tabifiedDockWidgets(slice_dock))
        self.assertTrue(
            mesh_dock.features()
            & QtWidgets.QDockWidget.DockWidgetMovable)
        self.assertEqual(
            mesh_dock.allowedAreas(), QtCore.Qt.AllDockWidgetAreas)
        self.assertTrue(
            self.window.dockOptions()
            & QtWidgets.QMainWindow.GroupedDragging)

        source_tab_bar = next(
            bar
            for bar in self.window.findChildren(QtWidgets.QTabBar)
            if {bar.tabText(i) for i in range(bar.count())}
            >= {"Slice", "Mesh"}
        )
        mesh_index = next(
            i for i in range(source_tab_bar.count())
            if source_tab_bar.tabText(i) == "Mesh"
        )
        self.assertTrue(source_tab_bar.isVisible())
        self.assertTrue(source_tab_bar.isTabEnabled(mesh_index))
        self.assertFalse(source_tab_bar.tabRect(mesh_index).isEmpty())

        # This is the expected result of dropping Mesh onto the Runs group.
        self.window.tabifyDockWidget(runs_dock, mesh_dock)
        mesh_dock.raise_()
        self.app.processEvents()

        self.assertEqual(
            self.window.dockWidgetArea(mesh_dock),
            QtCore.Qt.BottomDockWidgetArea,
        )
        self.assertIn(mesh_dock, self.window.tabifiedDockWidgets(runs_dock))
        self.assertNotIn(
            mesh_dock, self.window.tabifiedDockWidgets(slice_dock))

    def test_views_menu_reopens_a_closed_view(self) -> None:
        self.window.add_test_views()
        self.window.menuBar().addAction("Settings", lambda: None)
        self.window._build_view_menu()

        top_level = {
            action.text(): action for action in self.window.menuBar().actions()
        }
        self.assertIn("Settings", top_level)
        self.assertIn("Views", top_level)
        top_level_order = [
            action.text() for action in self.window.menuBar().actions()
        ]
        self.assertEqual(
            top_level_order.index("Views"),
            top_level_order.index("Settings") + 1,
        )

        views_menu = top_level["Views"].menu()
        self.assertIsNotNone(views_menu)
        view_actions = {
            action.text(): action
            for action in views_menu.actions()
            if not action.isSeparator()
        }
        self.assertTrue(set(VIEW_TITLES).issubset(view_actions))

        scene = self.window._dock_widgets["scene"]
        scene.close()
        self.app.processEvents()
        self.assertTrue(scene.isHidden())
        self.assertFalse(view_actions["Scene"].isChecked())

        view_actions["Scene"].trigger()
        self.app.processEvents()
        self.assertFalse(scene.isHidden())
        self.assertTrue(view_actions["Scene"].isChecked())

    def test_tab_close_button_closes_only_the_requested_view(self) -> None:
        self.window.add_test_views()
        slice_dock = self.window._dock_widgets["sdf"]
        mesh_dock = self.window._dock_widgets["mesh"]
        self.window.tabifyDockWidget(slice_dock, mesh_dock)
        slice_dock.raise_()
        self.window.show()
        self.app.processEvents()

        self.window._style_workspace_dock_tabs()
        self.app.processEvents()
        tab_bar = next(
            bar
            for bar in self.window.findChildren(QtWidgets.QTabBar)
            if {bar.tabText(i) for i in range(bar.count())}
            >= {"Slice", "Mesh"}
        )
        self.assertTrue(tab_bar.tabsClosable())
        self.assertTrue(bool(tab_bar.property("workspaceCloseConnected")))

        mesh_index = next(
            i for i in range(tab_bar.count())
            if tab_bar.tabText(i) == "Mesh"
        )
        tab_bar.tabCloseRequested.emit(mesh_index)
        self.app.processEvents()

        self.assertTrue(mesh_dock.isHidden())
        self.assertFalse(slice_dock.isHidden())
        self.assertTrue(slice_dock.isVisible())

    def test_views_menu_offers_default_layout_reset(self) -> None:
        self.window.add_test_views()
        self.window._build_view_menu()
        reset = next(
            action
            for action in self.window._view_menu.actions()
            if action.text() == "Reset Workspace Layout"
        )

        reset.trigger()

        self.assertEqual(self.window.reset_count, 1)

    def test_main_window_wires_docking_and_views_menu_into_layout(self) -> None:
        source = inspect.getsource(MainWindow._build_layout)

        self.assertIn("AllowNestedDocks", source)
        self.assertIn("AllowTabbedDocks", source)
        self.assertIn("GroupedDragging", source)
        self.assertIn("self._build_view_menu()", source)
        self.assertIn('"Settings", self._open_settings_dialog', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
