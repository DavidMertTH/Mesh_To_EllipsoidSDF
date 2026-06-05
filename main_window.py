"""
main_window.py — Application main window (fullscreen, three columns).

  ┌──────────────────────────┬──────────────────┬──────────────────┐
  │  3-D Viewport            │  SDF Slice / Mesh │  Options column  │
  │  (mesh + skeleton +      │  (tabbed)        │  (scrollable):   │
  │   ellipsoids, overlay)   │  ┌ Mesh tab ────┐│   Mesh, Training,│
  │                          │  │ rotation/blow││   Maintenance    │
  │                          │  │ + FBX/rig    ││   (merge/spawn/  │
  │                          │  └ panel ───────┘│    split), LR,   │
  │                          ├──────────────────┤   colours, Fit   │
  │                          │  Convergence +   │                  │
  │                          │  statistics      │                  │
  └──────────────────────────┴──────────────────┴──────────────────┘

Starts fullscreen (F11 / Esc to toggle).
Left:    mesh, skeleton bones and fitted ellipsoids share one GL scene with an
         in-viewport overlay (visibility toggles + render mode).
Middle:  a tabbed top panel (SDF slice / Mesh) over the convergence curve + run
         statistics.  The "Mesh" tab holds per-mesh rotation/blowup controls and
         the FBX/rig panel (shown only when a rigged mesh is loaded).
Right:   a scrollable options column — every selectable control, including
         on/off switches for the merge / spawn / split maintenance moves and
         the learning-rate parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import warp as wp

import theme
import app_settings
from settings_dialog import SettingsDialog
import shape_plugins
from shape_plugins import (
    widget_value, set_widget_value, widget_change_signal, available_shapes,
)
from mesh_io import load_and_prepare
from sdf_compute import SdfComputer, SdfResult
from ellipsoid import EllipsoidSet, SDF_QUILEZ, SDF_METHOD_NAMES, best_device
from viewer3d import SceneViewer3D
from widgets import SdfSlicePanel
from optimization import OptimizationWorker
from run_tracker import RunTrackerPanel
from dashboard import DashboardPanel
from mesh_settings import MeshSettingsPanel, rotate_mesh
from rig_panel import RigModePanel, try_load_rigged
from skinning import deform_mesh
from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from skeleton import Pose

# Supported mesh file extensions (trimesh + glTF for rigged)
MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".dae", ".fbx"}

# Default mesh directory relative to this file
DEFAULT_MESH_DIR = Path(__file__).parent / "meshes"


class SdfWorker(QtCore.QThread):
    """Runs ``SdfComputer.compute_voxel_grid`` off the GUI thread.

    The heavy Warp grid + thickness pass otherwise stalls the main thread; here
    it runs in a QThread and streams progress via the ``progress`` signal so the
    UI stays responsive and can show a progress bar.
    """

    progress = QtCore.Signal(float, str)   # fraction 0..1, status message
    done = QtCore.Signal(object)           # SdfResult
    failed = QtCore.Signal(str)

    def __init__(self, computer: SdfComputer, n: int, margin: float,
                 parent: QtCore.QObject | None = None, symmetry: bool = False):
        super().__init__(parent)
        self._computer = computer
        self._n = n
        self._margin = margin
        self._symmetry = symmetry

    def run(self):
        try:
            result = self._computer.compute_voxel_grid(
                n=self._n, margin=self._margin,
                progress_cb=lambda f, m: self.progress.emit(float(f), str(m)),
                symmetry=self._symmetry,
            )
            self.done.emit(result)
        except Exception as e:                       # surfaced on the GUI thread
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mesh_dir: Path | str | None = None, progress=None):
        super().__init__()
        self.setWindowTitle("Mesh → Ellipsoid SDF Approximation")

        # Optional 0..1 progress callback (used by the startup splash screen).
        # Accepts a fraction plus a short status message for the splash's bottom
        # line; degrades gracefully if the callback takes only a fraction.
        def _tick(frac: float, msg: str = "") -> None:
            if progress is None:
                return
            try:
                progress(frac, msg)
            except TypeError:
                progress(frac)

        self._mesh_dir = Path(mesh_dir) if mesh_dir else DEFAULT_MESH_DIR
        self._mesh_dir.mkdir(parents=True, exist_ok=True)

        wp.init()
        self._device = best_device()
        _tick(0.20, "Initializing GPU …")

        self._sdf = SdfComputer(device=self._device)
        self._ellipsoids: EllipsoidSet | None = None

        self._last_mesh_result: SdfResult | None = None
        # Single unified viewport: mesh + skeleton + ellipsoids together.
        self._viewer = SceneViewer3D()
        # SDF slice is shown for the mesh only.  Without a CUDA GPU the n³ SDF
        # runs on the CPU, so start at a much smaller default grid (64 vs 512).
        self._cpu_only = not str(self._device).startswith("cuda")
        self._mesh_sdf_panel = SdfSlicePanel(
            default_n=128 if self._cpu_only else 512)
        # Visual-refresh cadence (steps between viewport/dashboard updates).  On
        # GPU each emit forces a device sync + readback, so a wider stride (20)
        # keeps throughput up; on CPU steps are slow, so refresh more often (5)
        # for responsiveness — the sync is essentially free there.
        self._report_every = 5 if self._cpu_only else 20

        self._run_tracker = RunTrackerPanel()
        # Compact live status board (cards + mini graphs); complements the tracker.
        self._dashboard = DashboardPanel()
        # Per-mesh adjustments: rotation (async SDF recompute) + live SDF blowup.
        self._mesh_settings = MeshSettingsPanel()
        self._base_verts: np.ndarray | None = None       # un-rotated loaded mesh
        self._base_faces: np.ndarray | None = None
        self._pending_rot_mesh: tuple | None = None       # (verts, faces) for recompute
        self._mesh_rotation = (0.0, 0.0, 0.0)             # global deg, applied to
        self._mesh_rot_center: np.ndarray | None = None   # all poses + skeleton

        # ── Rig mode (modular) ──
        self._rig_panel = RigModePanel()

        self._status = self.statusBar()
        # Bottom-right progress area: a label saying what's happening (left) +
        # a wide, theme-coloured progress bar (right).  Added in this order so
        # the label sits immediately to the LEFT of the bar.
        self._progress_label = QtWidgets.QLabel("")
        self._progress_label.setVisible(False)
        self._status.addPermanentWidget(self._progress_label)

        self._sdf_progress = QtWidgets.QProgressBar()
        # High internal resolution (0..1000, not 0..100) so the animated fill
        # moves smoothly instead of in 1 % integer jumps.
        self._sdf_progress.setRange(0, 1000)
        self._sdf_progress.setMinimumWidth(320)
        self._sdf_progress.setMaximumWidth(480)
        # The built-in percentage text (dark-on-purple) is hard to read; show
        # it in the left label instead (always on the neutral status-bar bg).
        self._sdf_progress.setTextVisible(False)
        self._sdf_progress.setVisible(False)
        self._status.addPermanentWidget(self._sdf_progress)

        # Smoothly interpolate (lerp) the bar between successive values instead
        # of snapping — animates the int ``value`` property in the background.
        # The left label's percentage rides the same animation (see
        # _update_progress_label) so the number counts up smoothly too.
        self._progress_msg = ""
        self._progress_anim = QtCore.QPropertyAnimation(
            self._sdf_progress, b"value", self)
        self._progress_anim.setDuration(320)
        self._progress_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._progress_anim.valueChanged.connect(self._update_progress_label)
        self._style_progress_bar()
        self._sdf_worker: SdfWorker | None = None

        # ── Shape plugins (multi-shape fitting infrastructure) ──
        # Each plugin owns its shape-specific options + how they map to the
        # optimiser.  Only EllipsoidShape is implemented; others are greyed-out
        # placeholders.  Must exist before _build_layout (the options column
        # builds the shape selector + mounts the active shape's widgets).
        self._shapes = available_shapes()
        self._shapes_by_id = {s.id: s for s in self._shapes}
        self._shape = next((s for s in self._shapes if s.available), self._shapes[0])
        _tick(0.45, "Loading 3-D viewport & panels …")

        self._build_layout()
        _tick(0.80, "Building interface …")
        self._build_toolbar()
        _tick(0.92, "Building toolbar …")
        self._connect_signals()
        _tick(1.0, "Done")

        # Advanced fitting / maintenance settings (edited via the Settings
        # dialog, persisted to app_settings.json, forwarded to the worker).
        self._settings = app_settings.load()
        self._settings_dialog: SettingsDialog | None = None

        self._opt_worker: OptimizationWorker | None = None
        self._multipose_worker = None  # MultiPoseOptimizationWorker
        self._current_mesh_name: str = ""
        self._current_sdf_mode: int = SDF_QUILEZ
        # Bind-pose bone centres for the bone-awareness penalty (rigged meshes).
        self._bone_centers: np.ndarray | None = None

        # Rig pose scrubbing: switching poses updates the view instantly while
        # the (heavy n³) SDF recompute is debounced + run async, so dragging
        # through poses stays real-time instead of blocking on every step.
        self._pending_pose_mesh: tuple | None = None
        self._pose_sdf_timer = QtCore.QTimer(self)
        self._pose_sdf_timer.setSingleShot(True)
        self._pose_sdf_timer.setInterval(200)   # ms after the user settles
        self._pose_sdf_timer.timeout.connect(self._recompute_pose_sdf)

        # Mesh rotation: debounced async SDF recompute (the rotated mesh's SDF is
        # n³ work, so it runs on the SdfWorker once the user stops sliding).
        self._rot_sdf_timer = QtCore.QTimer(self)
        self._rot_sdf_timer.setSingleShot(True)
        self._rot_sdf_timer.setInterval(250)
        self._rot_sdf_timer.timeout.connect(self._recompute_rotated_sdf)
        self._mesh_settings.rotationChanged.connect(self._on_mesh_rotation_changed)
        self._mesh_settings.blowupChanged.connect(
            lambda vox: self._viewer.set_sdf_blowup(vox))

        # Restore the right-hand options panel from the last session and keep it
        # persisted on every change (panel_settings.json).
        self._init_panel_persistence()

    def _build_layout(self):
        # Three columns:
        #   left   — 3-D viewport (with the FBX/rig panel below it when rigged)
        #   middle — SDF analysis (top) + convergence curve & statistics (bottom)
        #   right  — a scrollable column with every option
        self._build_option_widgets()

        # ── Left: viewport only (the FBX/rig panel now lives in the Mesh tab) ──
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(self._viewer.widget, 1)

        # ── Middle: tabbed (SDF Slice / Mesh) on top, tabbed Dashboard / Runs
        # below ──.  The Mesh tab bundles per-mesh settings + the FBX/rig panel.
        self._top_tabs = QtWidgets.QTabWidget()
        self._top_tabs.addTab(self._mesh_sdf_panel, "SDF Slice")
        self._top_tabs.addTab(self._build_mesh_tab(), "Mesh")
        self._analysis_tabs = QtWidgets.QTabWidget()
        self._analysis_tabs.addTab(self._dashboard, "Dashboard")
        self._analysis_tabs.addTab(self._run_tracker, "Runs")
        middle_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        middle_splitter.addWidget(self._top_tabs)
        middle_splitter.addWidget(self._analysis_tabs)
        middle_splitter.setSizes([420, 580])
        self._middle_panel = middle_splitter

        # ── Right: options column (scrollable) ──
        self._options_panel = self._build_options_column()

        main_hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_hsplitter.addWidget(left_panel)
        main_hsplitter.addWidget(middle_splitter)
        main_hsplitter.addWidget(self._options_panel)
        main_hsplitter.setSizes([900, 460, 320])
        main_hsplitter.setCollapsible(0, False)
        main_hsplitter.setStretchFactor(0, 1)
        self._build_view_menu()
        # 'Settings' is a clickable menu-bar entry (not a submenu) that opens a
        # small dialog with the detailed fitting knobs + theme colours.
        self.menuBar().addAction("Settings", self._open_settings_dialog)

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.addWidget(main_hsplitter, 1)
        self.setCentralWidget(central)
        self._scan_mesh_dir()
        self._load_default_mesh()

    def _build_mesh_tab(self) -> QtWidgets.QScrollArea:
        """Compose the 'Mesh' tab: per-mesh settings + the FBX/rig panel.

        The rig panel (pose scrubbing, multi-pose fit, bone assignment, Unity
        export) used to sit below the viewport; it now lives here, scrollable
        and hidden until a rigged FBX is loaded (``_rig_panel.setVisible``).
        """
        container = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        v.addWidget(self._mesh_settings)
        v.addWidget(self._rig_panel)
        self._rig_panel.setVisible(False)   # only when a rigged mesh is loaded
        v.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def _load_default_mesh(self):
        """Load T-Pose.fbx (if present) as the startup model."""
        default = self._mesh_dir / "T-Pose.fbx"
        if not default.is_file():
            return
        idx = self._mesh_combo.findData(str(default))
        if idx >= 1:
            self._mesh_combo.blockSignals(True)
            self._mesh_combo.setCurrentIndex(idx)
            self._mesh_combo.blockSignals(False)
        self._load_mesh(str(default))

    # ── option widgets + right-hand options column ────────────────────────

    def _build_option_widgets(self):
        """Create every option control (kept as attributes used elsewhere)."""
        # Mesh selection
        self._mesh_combo = QtWidgets.QComboBox()
        self._mesh_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._btn_refresh = QtWidgets.QPushButton("↻ Rescan")
        self._btn_refresh.setToolTip("Rescan meshes/ folder")
        self._btn_open_dir = QtWidgets.QPushButton("📂 Folder")
        self._btn_open_dir.setToolTip(f"Open {self._mesh_dir}")

        # SDF margin
        self._slider_margin = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_margin.setRange(0, 100)
        self._slider_margin.setValue(50)
        self._slider_margin.setToolTip("Fractional margin around the mesh bounding box (0.0–1.0)")
        self._lbl_margin = QtWidgets.QLabel("0.50")
        self._slider_margin.valueChanged.connect(
            lambda v: self._lbl_margin.setText(f"{v / 100:.2f}"))

        # Training
        self._spin_num_ellipsoids = QtWidgets.QSpinBox()
        self._spin_num_ellipsoids.setRange(1, 200)
        self._spin_num_ellipsoids.setValue(60)
        self._spin_num_ellipsoids.setToolTip("Number of ellipsoids to fit")

        self._spin_max_ellipsoids = QtWidgets.QSpinBox()
        self._spin_max_ellipsoids.setRange(1, 500)
        self._spin_max_ellipsoids.setValue(180)
        self._spin_max_ellipsoids.setToolTip("SuperFit ellipsoid budget (population grows up to this)")

        self._spin_max_steps = QtWidgets.QSpinBox()
        self._spin_max_steps.setRange(100, 100000)
        self._spin_max_steps.setValue(7000)
        self._spin_max_steps.setSingleStep(1000)
        self._spin_max_steps.setToolTip("Maximum training steps")

        # NOTE: the ellipsoid-specific controls (SDF method, SuperFit, local
        # fit, soft-min, maintenance moves) now live in EllipsoidShape
        # (shape_plugins.py) and are mounted dynamically per selected shape.
        # Symmetry stays here — it is shape-agnostic and also drives the SDF
        # grid computation (see _on_compute_all).
        self._chk_symmetry = QtWidgets.QCheckBox("Symmetry (auto)")
        self._chk_symmetry.setChecked(True)
        self._chk_symmetry.setToolTip(
            "Auto-detect a mirror plane and mirror-project during training —\n"
            "only applied if the mesh is actually symmetric.")

        # Rig / bone awareness (only shown when a rigged FBX is loaded)
        self._chk_bone_aware = QtWidgets.QCheckBox("Bone-aware")
        self._chk_bone_aware.setChecked(True)
        self._chk_bone_aware.setToolTip(
            "Penalizes ellipsoids that span more than one bone — a little\n"
            "overlap is fine, but no ellipsoid should span multiple bones.")

        # Learning rate
        self._spin_lr_init = QtWidgets.QDoubleSpinBox()
        self._spin_lr_init.setRange(0.0001, 0.5)
        self._spin_lr_init.setDecimals(4)
        self._spin_lr_init.setSingleStep(0.001)
        self._spin_lr_init.setValue(0.01)
        self._spin_lr_init.setToolTip("Initial learning rate")
        self._spin_lr_final = QtWidgets.QDoubleSpinBox()
        self._spin_lr_final.setRange(0.00001, 0.1)
        self._spin_lr_final.setDecimals(5)
        self._spin_lr_final.setSingleStep(0.0001)
        self._spin_lr_final.setValue(0.0002)
        self._spin_lr_final.setToolTip("Final learning rate (floor)")
        self._spin_lr_decay = QtWidgets.QDoubleSpinBox()
        self._spin_lr_decay.setRange(0.0, 20.0)
        self._spin_lr_decay.setDecimals(1)
        self._spin_lr_decay.setSingleStep(0.5)
        self._spin_lr_decay.setValue(7.0)
        self._spin_lr_decay.setToolTip("Decay steepness — higher drops the LR faster")

        # Actions + status
        self._btn_fit = QtWidgets.QPushButton("▶ Fit Ellipsoids")
        self._btn_fit.setToolTip("Start fitting ellipsoids to the loaded mesh SDF")
        self._btn_fit.setEnabled(False)
        self._btn_stop = QtWidgets.QPushButton("■ Stop")
        self._btn_stop.setToolTip("Stop the running optimisation")
        self._btn_stop.setEnabled(False)
        self._lbl_ell_count = QtWidgets.QLabel("Count: 0")
        self._lbl_ell_count.setToolTip("Current number of ellipsoids")

        # Theme colour pickers
        self._btn_primary = QtWidgets.QPushButton()
        self._btn_primary.setFixedSize(30, 22)
        self._btn_primary.setToolTip("Primary color (mesh) — live")
        self._btn_primary.clicked.connect(
            lambda: self._pick_color(self._btn_primary,
                                     lambda: theme.BLUE, theme.set_primary))
        self._btn_secondary = QtWidgets.QPushButton()
        self._btn_secondary.setFixedSize(30, 22)
        self._btn_secondary.setToolTip("Secondary color (ellipsoids) — live")
        self._btn_secondary.clicked.connect(
            lambda: self._pick_color(self._btn_secondary,
                                     lambda: theme.YELLOW, theme.set_secondary))
        self._refresh_color_swatches()

    def _build_options_column(self) -> QtWidgets.QScrollArea:
        col = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(col)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)

        def _group(title: str) -> tuple:
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignLeft)
            v.addWidget(box)
            return box, form

        # Shape selector (topmost) — picks which primitive is fitted.  The
        # shape-specific options are mounted below in a dynamic container.
        _, shape_form = _group("Shape")
        self._combo_shape = QtWidgets.QComboBox()
        for s in self._shapes:
            self._combo_shape.addItem(s.display_name, s.id)
            if not s.available:
                item = self._combo_shape.model().item(self._combo_shape.count() - 1)
                item.setEnabled(False)            # greyed-out placeholder
                self._combo_shape.setItemData(
                    self._combo_shape.count() - 1,
                    f"{s.display_name} — coming soon", QtCore.Qt.ToolTipRole)
        self._combo_shape.setToolTip("Primitive type to fit to the mesh SDF")
        self._select_shape_in_combo(self._shape)
        self._combo_shape.currentIndexChanged.connect(self._on_shape_changed)
        shape_form.addRow(self._combo_shape)

        # Mesh
        _, mesh_form = _group("Mesh")
        mesh_form.addRow(self._mesh_combo)
        mesh_btns = QtWidgets.QHBoxLayout()
        mesh_btns.addWidget(self._btn_refresh)
        mesh_btns.addWidget(self._btn_open_dir)
        mesh_form.addRow(mesh_btns)
        margin_row = QtWidgets.QHBoxLayout()
        margin_row.addWidget(self._slider_margin)
        margin_row.addWidget(self._lbl_margin)
        mesh_form.addRow("SDF Margin:", margin_row)

        # Mesh rotation + SDF blowup live in the "Mesh" tab (middle column).

        # Training — shared controls only (count / budget / steps / symmetry).
        _, tr_form = _group("Training")
        self._lbl_count_row = QtWidgets.QLabel("Primitives:")
        tr_form.addRow(self._lbl_count_row, self._spin_num_ellipsoids)
        tr_form.addRow("Max:", self._spin_max_ellipsoids)
        tr_form.addRow("Steps:", self._spin_max_steps)
        tr_form.addRow(self._chk_symmetry)

        # Shape-specific options — mounted/swapped by the shape selector.
        self._shape_options_container = QtWidgets.QWidget()
        self._shape_options_layout = QtWidgets.QVBoxLayout(self._shape_options_container)
        self._shape_options_layout.setContentsMargins(0, 0, 0, 0)
        self._shape_options_layout.setSpacing(8)
        v.addWidget(self._shape_options_container)
        self._mount_shape_options(self._shape)

        # Rig options — only visible once a rigged FBX is loaded.
        self._grp_rig = QtWidgets.QGroupBox("Rig (FBX)")
        rig_form = QtWidgets.QFormLayout(self._grp_rig)
        rig_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        rig_form.addRow(self._chk_bone_aware)
        v.addWidget(self._grp_rig)
        self._grp_rig.setVisible(False)

        # Learning rate
        _, lr_form = _group("Learning Rate")
        lr_form.addRow("Start:", self._spin_lr_init)
        lr_form.addRow("End:", self._spin_lr_final)
        lr_form.addRow("Decay k:", self._spin_lr_decay)

        # Theme colours now live in the "Settings" menu (see _build_settings_menu).

        # Actions
        actions = QtWidgets.QVBoxLayout()
        actions.addWidget(self._btn_fit)
        actions.addWidget(self._btn_stop)
        actions.addWidget(self._lbl_ell_count)
        v.addLayout(actions)
        v.addStretch(1)

        self._update_primitive_labels()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(col)
        scroll.setMinimumWidth(260)
        return scroll

    # ── shape selection ───────────────────────────────────────────────────

    def _select_shape_in_combo(self, shape) -> None:
        idx = self._combo_shape.findData(shape.id)
        if idx >= 0:
            self._combo_shape.blockSignals(True)
            self._combo_shape.setCurrentIndex(idx)
            self._combo_shape.blockSignals(False)

    def _mount_shape_options(self, shape) -> None:
        """Show *shape*'s option widget in the dynamic container."""
        while self._shape_options_layout.count():
            item = self._shape_options_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)               # detach (plugin keeps it alive)
        self._shape_options_layout.addWidget(shape.options_widget())

    def _update_primitive_labels(self) -> None:
        """Re-word the shared count label + fit button for the active shape."""
        noun = self._shape.primitive_noun
        plural = noun + "s"
        self._lbl_count_row.setText(f"{plural.capitalize()}:")
        self._btn_fit.setText(f"▶ Fit {plural.capitalize()}")

    def _on_shape_changed(self, index: int) -> None:
        sid = self._combo_shape.itemData(index)
        shape = self._shapes_by_id.get(sid)
        if shape is None or shape is self._shape:
            return
        if not shape.available:
            # Shouldn't happen (item disabled) — revert defensively.
            self._select_shape_in_combo(self._shape)
            self._status.showMessage(f"{shape.display_name} fitting is coming soon.")
            return
        self._save_panel_settings()             # persist the outgoing shape
        self._shape = shape
        self._mount_shape_options(shape)
        self._update_primitive_labels()
        self._save_panel_settings()             # remember the new selection
        self._status.showMessage(f"Shape: {shape.display_name}")

    def _build_view_menu(self):
        """An 'Ansicht' menu to show/hide each panel individually."""
        self._view_menu = self.menuBar().addMenu("View")
        self._view_actions = {}
        for label, widget in (
            ("3D viewport", self._viewer.widget),
            ("SDF analysis", self._mesh_sdf_panel),
            ("Convergence / statistics", self._run_tracker),
            ("Options", self._options_panel),
        ):
            act = QtGui.QAction(label, self, checkable=True)
            act.setChecked(True)
            act.toggled.connect(widget.setVisible)
            self._view_menu.addAction(act)
            self._view_actions[label] = act

    def _make_color_row(self) -> QtWidgets.QWidget:
        """A small row holding the (existing) primary/secondary swatch buttons.

        Built once and embedded into the Settings dialog's Appearance tab; reuses
        the same button objects so the live preview / persistence keep working.
        """
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(QtWidgets.QLabel("Primary"))
        h.addWidget(self._btn_primary)
        h.addSpacing(12)
        h.addWidget(QtWidgets.QLabel("Secondary"))
        h.addWidget(self._btn_secondary)
        h.addStretch()
        return w

    def _open_settings_dialog(self) -> None:
        """Open (or reopen) the Settings window and commit/persist on OK."""
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(
                self._settings, self._make_color_row(), self)
        else:
            self._settings_dialog.load_values(self._settings)
        if self._settings_dialog.exec() == QtWidgets.QDialog.Accepted:
            self._settings.update(self._settings_dialog.values())
            app_settings.save(self._settings)

    # ── options-panel persistence (right-hand column) ─────────────────────
    # Layout of panel_settings.json:
    #   { "shape": "<id>",
    #     "shared": { ...shared controls... },
    #     "shapes": { "<id>": { ...that shape's options... }, ... } }
    # Shared controls live in MainWindow; each shape persists its own options.

    def _shared_setting_specs(self) -> list:
        """(key, widget, kind) for the SHARED controls (shape-agnostic).

        The mesh selector is intentionally excluded (depends on the folder).
        Shape-specific controls are persisted by each ShapePlugin instead.
        """
        return [
            ("margin",          self._slider_margin,       "int"),
            ("num_ellipsoids",  self._spin_num_ellipsoids, "int"),
            ("max_ellipsoids",  self._spin_max_ellipsoids, "int"),
            ("max_steps",       self._spin_max_steps,      "int"),
            ("symmetry",        self._chk_symmetry,        "bool"),
            ("bone_aware",      self._chk_bone_aware,      "bool"),
            ("lr_init",         self._spin_lr_init,        "float"),
            ("lr_final",        self._spin_lr_final,       "float"),
            ("lr_decay",        self._spin_lr_decay,       "float"),
        ]

    def _init_panel_persistence(self) -> None:
        """Restore saved panel values (shared + per-shape), then auto-persist."""
        # Debounced save so dragging a slider doesn't hammer the disk.
        self._panel_save_timer = QtCore.QTimer(self)
        self._panel_save_timer.setSingleShot(True)
        self._panel_save_timer.setInterval(400)
        self._panel_save_timer.timeout.connect(self._save_panel_settings)

        saved = app_settings.load_panel()
        shared = saved.get("shared", {}) if isinstance(saved, dict) else {}
        shapes_state = saved.get("shapes", {}) if isinstance(saved, dict) else {}

        # Apply persisted values BEFORE wiring autosave, so restoring them
        # doesn't immediately trigger a (redundant) save.
        if isinstance(shared, dict):
            for key, w, kind in self._shared_setting_specs():
                if key in shared:
                    set_widget_value(w, kind, shared[key])
        if isinstance(shapes_state, dict):
            for shape in self._shapes:
                st = shapes_state.get(shape.id)
                if isinstance(st, dict):
                    shape.options_widget()          # ensure widgets exist
                    shape.apply_panel_state(st)

        # Restore the last selected shape (only if it's available).
        last = saved.get("shape") if isinstance(saved, dict) else None
        sel = self._shapes_by_id.get(last)
        if sel is not None and sel.available and sel is not self._shape:
            self._shape = sel
            self._mount_shape_options(sel)
            self._update_primitive_labels()
            self._select_shape_in_combo(sel)

        # Wire autosave on every change (shared widgets + each shape's widgets).
        kick = lambda *_: self._panel_save_timer.start()
        for _key, w, kind in self._shared_setting_specs():
            widget_change_signal(w, kind).connect(kick)
        for shape in self._shapes:
            shape.options_widget()          # ensure widgets exist before wiring
            shape.connect_changed(lambda: self._panel_save_timer.start())

    def _save_panel_settings(self) -> None:
        shared = {key: widget_value(w, kind)
                  for key, w, kind in self._shared_setting_specs()}
        shapes_state = {s.id: s.panel_state() for s in self._shapes if s.available}
        app_settings.save_panel({
            "shape": self._shape.id,
            "shared": shared,
            "shapes": shapes_state,
        })

    def _build_toolbar(self):
        act_compute = QtGui.QAction("Compute SDF Grid (G)", self)
        act_compute.setShortcut(QtGui.QKeySequence("G"))
        act_compute.triggered.connect(self._on_compute_all)

        act_maximize = QtGui.QAction("Toggle Maximise (F11)", self)
        act_maximize.setShortcut(QtGui.QKeySequence("F11"))
        act_maximize.triggered.connect(self._toggle_maximize)

        # No top toolbar — these actions are kept only for their keyboard
        # shortcuts (G = compute SDF grid, F11 = toggle maximise).
        self.addAction(act_compute)
        self.addAction(act_maximize)

    def _toggle_maximize(self) -> None:
        # Windowed fullscreen ↔ normal window (never exclusive fullscreen, which
        # makes combo-box popups appear behind the window on Windows).
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape and self.isMaximized():
            self.showNormal()
            return
        super().keyPressEvent(event)

    def _connect_signals(self):
        self._viewer.widget.fileDropped.connect(self._on_file_dropped)
        self._mesh_sdf_panel.computeRequested.connect(self._on_compute_all)
        self._mesh_combo.activated.connect(self._on_combo_selected)
        self._btn_refresh.clicked.connect(self._scan_mesh_dir)
        self._btn_open_dir.clicked.connect(self._open_mesh_dir)
        self._btn_fit.clicked.connect(self._on_fit_clicked)
        self._btn_stop.clicked.connect(self._on_stop_clicked)

        # ── Rig mode signals ──
        self._rig_panel.poseChanged.connect(self._on_rig_pose_changed)
        self._rig_panel.multiPoseRequested.connect(self._on_multipose_fit_clicked)
        self._rig_panel._btn_assign.clicked.connect(self._on_rig_assign_clicked)
        self._rig_panel.autoPipelineRequested.connect(self._on_auto_pipeline_clicked)
        self._rig_panel.exportUnityRequested.connect(self._on_export_unity_clicked)

    # ── runtime light/dark switching ──────────────────────────────────────

    def changeEvent(self, event):
        # The OS / Qt switching colour scheme delivers an ApplicationPaletteChange
        # to every widget; re-colour our custom-styled ones (standard widgets
        # follow the palette automatically).
        if event.type() in (
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.PaletteChange,
        ):
            self._apply_theme()
        super().changeEvent(event)

    def _apply_theme(self) -> None:
        if not hasattr(self, "_btn_primary"):
            return                      # window not fully built yet
        # pyqtgraph defaults for any widgets created afterwards.
        if theme.is_dark_mode():
            pg.setConfigOptions(foreground="d", background="k")
        else:
            pg.setConfigOptions(foreground="k", background="w")
        self._viewer.apply_theme()
        self._mesh_sdf_panel.apply_theme()
        self._run_tracker.apply_theme()
        self._dashboard.apply_theme()
        self._refresh_color_swatches()
        self._style_progress_bar()

    # ── status-bar progress bar (themed + animated) ───────────────────────

    def _style_progress_bar(self) -> None:
        """Theme the status-bar progress bar with the brand colours.

        The filled chunk is a primary→secondary gradient so it tracks the
        user's chosen colours live (re-applied from ``_apply_theme``).
        """
        if not hasattr(self, "_sdf_progress"):
            return
        primary = theme.BLUE_HEX
        secondary = theme.YELLOW_HEX
        fg = "#e6e6e6" if theme.is_dark_mode() else "#1e1e1e"
        self._sdf_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid rgba(128, 128, 128, 0.5);
                border-radius: 7px;
                background: rgba(128, 128, 128, 0.22);
                text-align: center;
                color: {fg};
                min-height: 16px;
                padding: 0px;
            }}
            QProgressBar::chunk {{
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 {primary}, stop:1 {secondary});
            }}
        """)

    def _update_progress_label(self, value: int) -> None:
        """Sync the left label's percentage to *value* (in bar units)."""
        span = max(1, self._sdf_progress.maximum())
        pct = int(round(value / span * 100))
        self._progress_label.setText(
            f"{self._progress_msg} · {pct}%" if self._progress_msg else f"{pct}%")

    def _progress_begin(self, msg: str) -> None:
        """Show the progress area, reset to 0, and set the activity label."""
        self._progress_msg = msg
        self._progress_anim.stop()
        self._sdf_progress.setValue(0)
        self._sdf_progress.setVisible(True)
        self._update_progress_label(0)
        self._progress_label.setVisible(True)

    def _progress_set(self, value: float, msg: str | None = None) -> None:
        """Lerp the bar toward *value* (a 0–100 percentage); update the label."""
        if msg is not None:
            self._progress_msg = msg
        self._sdf_progress.setVisible(True)
        self._progress_label.setVisible(True)
        span = self._sdf_progress.maximum()
        target = int(max(0, min(span, round(value / 100.0 * span))))
        self._progress_anim.stop()
        self._progress_anim.setStartValue(self._sdf_progress.value())
        self._progress_anim.setEndValue(target)
        self._progress_anim.start()
        # Sync immediately too, so a no-op animation (start == target) or a
        # message-only change still refreshes the label.
        self._update_progress_label(self._sdf_progress.value())

    def _progress_end(self) -> None:
        """Hide the progress bar and its activity label."""
        self._progress_anim.stop()
        self._progress_msg = ""
        self._sdf_progress.setVisible(False)
        self._progress_label.clear()
        self._progress_label.setVisible(False)

    # ── provisional colour pickers ───────────────────────────────────────

    def _refresh_color_swatches(self) -> None:
        for btn, rgb in ((self._btn_primary, theme.BLUE),
                         (self._btn_secondary, theme.YELLOW)):
            btn.setStyleSheet(
                f"background-color: {theme.hex_str(rgb)};"
                " border: 1px solid #888; border-radius: 3px;")

    def _pick_color(self, swatch_btn, getter, setter) -> None:
        """Open a live colour dialog; previews update the whole UI immediately.

        Uses the non-native Qt dialog so ``currentColorChanged`` fires live while
        dragging.  Cancelling reverts to the colour in use when it opened.
        """
        original = getter()
        dlg = QtWidgets.QColorDialog(QtGui.QColor(*original), self)
        dlg.setOption(
            QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog, True)

        def _live(c: QtGui.QColor) -> None:
            if not c.isValid():
                return
            setter((c.red(), c.green(), c.blue()))
            self._apply_theme()

        def _revert() -> None:
            setter(original)
            self._apply_theme()

        dlg.currentColorChanged.connect(_live)
        dlg.rejected.connect(_revert)
        # Remember the final choice across restarts (fires after accept/revert).
        dlg.finished.connect(lambda _r: theme.save_colors())
        dlg.open()

    # ── mesh directory scanning ───────────────────────────────────────────

    def _scan_mesh_dir(self):
        self._mesh_combo.blockSignals(True)
        prev_text = self._mesh_combo.currentText()
        self._mesh_combo.clear()
        self._mesh_combo.addItem("— select mesh —")

        if self._mesh_dir.is_dir():
            files = sorted(
                f for f in self._mesh_dir.iterdir()
                if f.is_file() and f.suffix.lower() in MESH_EXTENSIONS
            )
            for f in files:
                self._mesh_combo.addItem(f.name, str(f))

        idx = self._mesh_combo.findText(prev_text)
        if idx >= 1:
            self._mesh_combo.setCurrentIndex(idx)
        self._mesh_combo.blockSignals(False)

    def _on_combo_selected(self, idx: int):
        if idx < 1:
            return
        path = self._mesh_combo.itemData(idx)
        if path:
            self._load_mesh(path)

    def _open_mesh_dir(self):
        path = str(self._mesh_dir.resolve())
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_file_dropped(self, path: str):
        self._load_mesh(path)

    def _load_mesh(self, path: str):
        # ── Try as rigged mesh first (if rig mode possible) ──
        rigged = try_load_rigged(path, target_scale=1.0)
        if rigged is not None:
            self._load_rigged_mesh(path, rigged)
            return

        # ── Static mesh (original behaviour) ──
        self._rig_panel.setChecked(False)  # Deactivate rig mode
        self._rig_panel.setVisible(False)  # Hide all rig/FBX options
        self._viewer.remove_skeleton()     # No skeleton → hide its overlay toggle
        self._grp_rig.setVisible(False)    # Hide bone-awareness option
        self._bone_centers = None
        try:
            mesh = load_and_prepare(path, target_scale=1.0)
            verts = mesh.vertices.view(np.ndarray)
            faces = mesh.faces.view(np.ndarray)

            self._set_base_mesh(verts, faces)
            self._viewer.show_mesh(verts, faces)
            self._ensure_sdf_idle()
            self._sdf.set_mesh(verts, faces)
            self._current_mesh_name = Path(path).name

            name = Path(path).name
            idx = self._mesh_combo.findText(name)
            self._mesh_combo.blockSignals(True)
            if idx >= 1:
                self._mesh_combo.setCurrentIndex(idx)
            else:
                self._mesh_combo.setCurrentIndex(0)
            self._mesh_combo.blockSignals(False)

            self._status.showMessage(
                f"Loaded: {path} | verts={len(verts)} faces={len(faces)} | device={self._device}"
            )
            self._on_compute_all()

        except Exception as e:
            self._status.showMessage(f"Failed to load: {path} ({e})")

    def _load_rigged_mesh(self, path: str, rigged):
        """Load a rigged mesh and activate rig mode."""
        self._rig_panel.setVisible(True)  # Reveal rig/FBX options
        self._rig_panel.setChecked(True)
        self._rig_panel.set_rigged_mesh(rigged)
        self._grp_rig.setVisible(True)    # Reveal bone-awareness option
        self._bone_centers = self._compute_bone_centers(rigged)

        verts = rigged.vertices
        faces = rigged.faces

        self._set_base_mesh(verts, faces)
        self._viewer.show_mesh(verts, faces)
        self._ensure_sdf_idle()
        self._sdf.set_mesh(verts, faces)
        self._current_mesh_name = Path(path).name

        # Show skeleton bones in bind pose
        self._show_skeleton_for_pose(rigged, None)

        name = Path(path).name
        idx = self._mesh_combo.findText(name)
        self._mesh_combo.blockSignals(True)
        if idx >= 1:
            self._mesh_combo.setCurrentIndex(idx)
        else:
            self._mesh_combo.setCurrentIndex(0)
        self._mesh_combo.blockSignals(False)

        self._status.showMessage(
            f"Rigged mesh: {name} | verts={len(verts)} faces={len(faces)} | "
            f"{rigged.skeleton.num_bones} bones | {len(rigged.poses)} poses"
        )
        self._on_compute_all()

    def _show_skeleton_for_pose(self, rigged, pose):
        """Render skeleton bones for a given pose (None = bind pose)."""
        positions, _ = rigged.skeleton.compute_bone_positions_rotations(pose)
        # The global mesh rotation applies to the skeleton too.
        positions = self._apply_rotation(np.asarray(positions, dtype=np.float32))
        parent_indices = np.array(
            [b.parent_index for b in rigged.skeleton.bones], dtype=np.int32,
        )
        self._viewer.show_bones(positions, parent_indices)

    @staticmethod
    def _compute_bone_centers(rigged) -> np.ndarray | None:
        """Bind-pose centre of each bone segment (parent-joint → joint midpoint).

        One representative point per actual bone (root joints, which have no
        parent segment, are skipped) for the bone-awareness penalty.
        """
        positions, _ = rigged.skeleton.compute_bone_positions_rotations(None)
        positions = np.asarray(positions, dtype=np.float32)
        centers = []
        for i, b in enumerate(rigged.skeleton.bones):
            pi = b.parent_index
            if pi is not None and pi >= 0:
                centers.append(0.5 * (positions[i] + positions[pi]))
        if not centers:
            return None
        return np.asarray(centers, dtype=np.float32)

    # ── Rig-mode: pose changed ───────────────────────────────────────

    def _on_rig_pose_changed(self, pose_index: int):
        """User selected a pose — switch the view *instantly*; SDF updates async.

        Only the cheap, visible parts (deform → mesh + skeleton + ellipsoids)
        run here so scrubbing through poses stays real-time.  The expensive n³
        SDF recompute is debounced and run on the background ``SdfWorker`` (see
        ``_recompute_pose_sdf``) once the user settles on a pose.
        """
        if not self._rig_panel.is_active:
            return

        rm = self._rig_panel.rigged_mesh
        if rm is None:
            return

        # ── Instant, non-blocking visible update ──
        deformed = self._rig_panel.get_deformed_mesh(pose_index)
        if deformed is not None:
            # The global mesh rotation rides on top of every pose.
            deformed = self._apply_rotation(deformed)
            self._viewer.show_mesh(deformed, rm.faces)

            pose = rm.poses[pose_index] if pose_index < len(rm.poses) else Pose.t_pose()
            self._show_skeleton_for_pose(rm, pose)

            # Defer the heavy SDF recompute; keep only the latest pose pending.
            self._pending_pose_mesh = (deformed, rm.faces)
            self._pose_sdf_timer.start()

        # Update ellipsoids if bone-local params exist
        world_ell = self._rig_panel.get_world_ellipsoids(pose_index)
        if world_ell is not None:
            wc, wr, wq = world_ell
            self._viewer.show_ellipsoids_fast(wc, wr, wq)

        self._status.showMessage(
            f"Pose {pose_index}: {rm.poses[pose_index].name if pose_index < len(rm.poses) else 'T-Pose'}"
        )

    def _recompute_pose_sdf(self) -> None:
        """Debounced async SDF recompute for the current rig pose.

        Fires a short moment after the user stops changing poses.  Never blocks
        the GUI: if an SDF worker is still busy it just retries a little later,
        and the actual n³ work runs on the background ``SdfWorker``.
        """
        pending = self._pending_pose_mesh
        if pending is None or not self._rig_panel.is_active:
            return
        # Don't swap the mesh out from under a running worker — retry shortly.
        if self._sdf_worker is not None and self._sdf_worker.isRunning():
            self._pose_sdf_timer.start()
            return
        deformed, faces = pending
        self._pending_pose_mesh = None
        self._sdf.set_mesh(deformed, faces)
        self._on_compute_all()

    # ── Mesh settings: rotation + SDF blowup ───────────────────────────
    def _set_base_mesh(self, verts: np.ndarray, faces: np.ndarray) -> None:
        """Remember the un-rotated mesh + a fixed rotation pivot; reset controls."""
        self._base_verts = np.ascontiguousarray(verts, dtype=np.float32)
        self._base_faces = np.ascontiguousarray(faces)
        # Fixed pivot (bind-pose bbox centre) so the rotation is stable across
        # poses and the skeleton.
        self._mesh_rot_center = 0.5 * (self._base_verts.min(axis=0)
                                       + self._base_verts.max(axis=0))
        self._mesh_rotation = (0.0, 0.0, 0.0)
        self._pending_rot_mesh = None
        self._mesh_settings.reset()
        self._viewer.set_sdf_blowup(0.0)

    def _apply_rotation(self, points: np.ndarray) -> np.ndarray:
        """Apply the current global mesh rotation about the fixed pivot."""
        rx, ry, rz = self._mesh_rotation
        if rx == 0.0 and ry == 0.0 and rz == 0.0:
            return np.asarray(points, dtype=np.float32)
        return rotate_mesh(points, rx, ry, rz, center=self._mesh_rot_center)

    def _on_mesh_rotation_changed(self, rx: float, ry: float, rz: float) -> None:
        """Store the rotation and re-apply it to the current mesh (and pose)."""
        self._mesh_rotation = (rx, ry, rz)
        # Rigged: re-render the active pose so the rotation rides on top of it
        # (mesh + skeleton) and the SDF recompute uses the rotated posed mesh.
        if self._rig_panel.is_active and self._rig_panel.rigged_mesh is not None:
            self._on_rig_pose_changed(self._rig_panel.current_pose_index)
            return
        if self._base_verts is None or self._base_faces is None:
            return
        rotated = self._apply_rotation(self._base_verts)
        self._viewer.show_mesh(rotated, self._base_faces)     # immediate visual
        self._pending_rot_mesh = (rotated, self._base_faces)  # SDF recompute soon
        self._rot_sdf_timer.start()

    def _recompute_rotated_sdf(self) -> None:
        """Debounced async SDF recompute for the rotated mesh (never blocks)."""
        pending = self._pending_rot_mesh
        if pending is None:
            return
        if self._sdf_worker is not None and self._sdf_worker.isRunning():
            self._rot_sdf_timer.start()                       # retry shortly
            return
        verts, faces = pending
        self._pending_rot_mesh = None
        self._sdf.set_mesh(verts, faces)
        self._on_compute_all()

    # ── Rig-mode: one-click auto pipeline ───────────────────────────

    def _on_auto_pipeline_clicked(self):
        """One-click: initialize ellipsoids from bone structure → multi-pose train."""
        if not self._rig_panel.is_active:
            return
        rm = self._rig_panel.rigged_mesh
        if rm is None:
            self._status.showMessage("No rigged mesh loaded.")
            return

        self.stop_optimization()
        self._stop_multipose()

        from bone_ellipsoid_mapper import initialize_ellipsoids_from_bones

        n_ellipsoids = self._spin_num_ellipsoids.value()
        self._status.showMessage("Auto pipeline: initializing ellipsoids from bone structure…")
        QtWidgets.QApplication.processEvents()

        bone_local = initialize_ellipsoids_from_bones(
            skeleton=rm.skeleton,
            vertices=rm.vertices,
            skin_joints=rm.skin_joints,
            skin_weights=rm.skin_weights,
            n_ellipsoids=n_ellipsoids,
        )

        self._rig_panel.set_bone_local(bone_local)
        self._rig_panel.set_auto_pipeline_running(True)
        self._status.showMessage(
            f"Auto pipeline: training {bone_local.num_ellipsoids} ellipsoids "
            f"across {len(rm.poses)} poses…"
        )
        self._on_multipose_fit_clicked()

    # ── Rig-mode: assign to bones ────────────────────────────────────

    def _on_rig_assign_clicked(self):
        """Assign current ellipsoids to bones based on T-pose fit."""
        if not self._rig_panel.is_active:
            return
        rm = self._rig_panel.rigged_mesh
        mapper = self._rig_panel.mapper
        if rm is None or mapper is None:
            self._status.showMessage("Load a rigged mesh first.")
            return

        if self._ellipsoids is None:
            self._status.showMessage("Fit ellipsoids in T-pose first (press ▶ Fit).")
            return

        self._status.showMessage("Assigning ellipsoids to bones…")
        QtWidgets.QApplication.processEvents()

        bone_local = mapper.assign_to_bones(
            world_centers=self._ellipsoids.centers,
            world_radii=self._ellipsoids.radii,
            world_rotations=self._ellipsoids.rotations,
            mesh_vertices=rm.vertices,
            skin_joints=rm.skin_joints,
            skin_weights=rm.skin_weights,
            pose=Pose.t_pose(),
        )

        self._rig_panel.set_bone_local(bone_local)
        self._status.showMessage(
            f"Assigned {bone_local.num_ellipsoids} ellipsoids to bones. "
            "Ready for multi-pose training."
        )

    # ── Rig-mode: multi-pose training ────────────────────────────────

    def _on_multipose_fit_clicked(self):
        """Start multi-pose training."""
        if not self._rig_panel.is_active:
            return
        rm = self._rig_panel.rigged_mesh
        bl = self._rig_panel.bone_local
        mapper = self._rig_panel.mapper
        if rm is None or bl is None or mapper is None:
            self._status.showMessage("Assign ellipsoids to bones first.")
            return

        # Stop any running optimisation
        self.stop_optimization()
        self._stop_multipose()

        from pose_optimizer import MultiPoseOptimizationWorker

        grid_n = self._mesh_sdf_panel.requested_n
        margin = self._slider_margin.value() / 100.0

        # Drive multi-pose training from the SAME settings as the normal fit:
        # steps + LR from the right panel, all loss/sampling knobs from the
        # Settings dialog.  There are no separate rig settings anymore.
        self._multipose_worker = MultiPoseOptimizationWorker(
            rest_vertices=rm.vertices,
            faces=rm.faces,
            skeleton=rm.skeleton,
            skin_joints=rm.skin_joints,
            skin_weights=rm.skin_weights,
            poses=rm.poses,
            bone_local=bl,
            mapper=mapper,
            num_steps=self._spin_max_steps.value(),
            report_every=self._report_every,
            grid_n=grid_n,
            margin=margin,
            lr=self._spin_lr_init.value(),
            advanced=self._settings,
            parent=self,
        )

        self._multipose_worker.precompute_progress.connect(
            self._on_multipose_precompute)
        self._multipose_worker.precompute_done.connect(
            self._on_multipose_precompute_done)
        self._multipose_worker.step_visual.connect(
            self._on_multipose_step_visual)
        self._multipose_worker.step_sdf.connect(
            self._on_opt_step_sdf)
        self._multipose_worker.pose_loss.connect(
            self._on_multipose_pose_loss)
        self._multipose_worker.pose_switched.connect(
            self._on_multipose_pose_switched)
        self._multipose_worker.finished.connect(
            self._on_multipose_finished)

        self._multipose_worker.start()
        self._btn_fit.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._opt_total_steps = max(1, int(self._spin_max_steps.value()))
        self._status.showMessage("Multi-pose training: pre-computing SDFs…")

    def _stop_multipose(self):
        if self._multipose_worker is not None and self._multipose_worker.isRunning():
            self._multipose_worker.request_stop()
            self._multipose_worker.wait()
            self._multipose_worker = None

    def _on_multipose_precompute(self, current: int, total: int):
        self._rig_panel.set_progress(current, total)
        self._status.showMessage(
            f"Pre-computing SDF for pose {current + 1}/{total}…"
        )

    def _on_multipose_step_visual(
        self, step: int, loss: float,
        centers: np.ndarray, radii: np.ndarray, rotations: np.ndarray,
        pose_name: str,
    ):
        print(f"[MultiPose] Step {step}: loss={loss:.6f} pose={pose_name}")
        self._viewer.show_ellipsoids_fast(centers, radii, rotations)
        total = getattr(self, "_opt_total_steps", 0)
        if total > 0:
            self._progress_set(min(1.0, step / total) * 100,
                               f"Multi-pose · step {step}/{total}")
        self._status.showMessage(
            f"Multi-pose training — step {step}/{total}  loss={loss:.6f}  pose={pose_name}"
        )
        self._run_tracker.record_step(step, loss)

    def _on_multipose_precompute_done(self):
        """All poses pre-computed — cache deformed meshes for instant switching."""
        self._precomputed_meshes = {}
        if self._multipose_worker is None:
            return
        rm = self._rig_panel.rigged_mesh
        if rm is None:
            return
        for pi, pd in enumerate(self._multipose_worker.pose_data):
            if pd.deformed_verts is not None:
                self._precomputed_meshes[pi] = pd.deformed_verts
        self._status.showMessage(
            f"Pre-computed {len(self._precomputed_meshes)} poses. Training…"
        )

    def _on_multipose_pose_switched(self, pose_index: int):
        """Training switched to a new pose — show pre-computed mesh + bones."""
        rm = self._rig_panel.rigged_mesh
        if rm is None:
            return
        pose = rm.poses[pose_index] if pose_index < len(rm.poses) else Pose.t_pose()

        # Use pre-computed deformed mesh (instant)
        cached = getattr(self, '_precomputed_meshes', {})
        if pose_index in cached:
            self._viewer.show_mesh(cached[pose_index], rm.faces)
        self._show_skeleton_for_pose(rm, pose)

        # Update pose slider to match
        self._rig_panel._slider_pose.blockSignals(True)
        self._rig_panel._slider_pose.setValue(pose_index)
        self._rig_panel._slider_pose.blockSignals(False)
        self._rig_panel._lbl_pose.setText(
            f"{pose_index} / {len(rm.poses) - 1}")
        self._rig_panel._lbl_pose_name.setText(f"Pose: {pose.name}")

    def _on_multipose_pose_loss(self, step: int, pose_name: str, loss: float):
        pass  # Could add per-pose loss tracking in the future

    def _on_multipose_finished(self):
        self._rig_panel.set_auto_pipeline_running(False)
        self._status.showMessage("Multi-pose training finished.")
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._btn_stop.setEnabled(False)
        self._progress_end()
        self._rig_panel.set_progress(100, 100)

        # Update bone-local params from worker result
        if self._multipose_worker is not None:
            self._rig_panel.set_bone_local(self._multipose_worker.result)

        self._multipose_worker = None

        # Update ellipsoid view for current pose
        self._on_rig_pose_changed(self._rig_panel.current_pose_index)

    # ── Unity export ─────────────────────────────────────────────────────────

    def _on_export_unity_clicked(self):
        """Export bone-local ellipsoids to JSON for Unity."""
        if not self._rig_panel.is_active:
            return
        bl = self._rig_panel.bone_local
        rm = self._rig_panel.rigged_mesh
        if bl is None or rm is None:
            self._status.showMessage("No fitted ellipsoids to export. Run the pipeline first.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Ellipsoids for Unity",
            str(Path(self._current_mesh_name).stem) + "_ellipsoids.json"
            if self._current_mesh_name else "ellipsoids.json",
            "JSON files (*.json)",
        )
        if not path:
            return

        from ellipsoid_exporter import export_ellipsoids
        n = export_ellipsoids(bl, rm.skeleton, path)
        self._status.showMessage(f"Exported {n} ellipsoids → {path}")

    # ── Original methods (unchanged below) ────────────────────────────────

    def _on_compute_all(self, n: int | None = None):
        if not self._sdf.is_ready:
            self._status.showMessage("Load a mesh first.")
            return
        if self._sdf_worker is not None and self._sdf_worker.isRunning():
            self._status.showMessage("SDF computation already running …")
            return

        if n is None:
            n = self._mesh_sdf_panel.requested_n

        margin = self._slider_margin.value() / 100.0
        self._status.showMessage(
            f"Computing mesh SDF (n={n}, margin={margin:.2f}) on {self._device} …")

        # Don't let a fit start against a stale/absent grid while computing.
        self._btn_fit.setEnabled(False)
        self._progress_begin("Computing mesh SDF …")

        # When symmetry fitting is on, let the SDF computer exploit it too:
        # detect the mirror plane and evaluate only half the grid at full res.
        self._sdf_worker = SdfWorker(
            self._sdf, n, margin, parent=self,
            symmetry=self._chk_symmetry.isChecked(),
        )
        self._sdf_worker.progress.connect(self._on_sdf_progress)
        self._sdf_worker.done.connect(self._on_sdf_done)
        self._sdf_worker.failed.connect(self._on_sdf_failed)
        self._sdf_worker.start()

    def _on_sdf_progress(self, frac: float, msg: str) -> None:
        self._progress_set(frac * 100, f"Mesh SDF · {msg}")
        self._status.showMessage(f"Mesh SDF: {msg}  ({int(frac * 100)} %)")

    def _on_sdf_done(self, mesh_result) -> None:
        self._sdf_worker = None
        self._progress_end()

        self._last_mesh_result = mesh_result
        self._mesh_sdf_panel.set_sdf(mesh_result.grid, mesh_result.dx)
        self._viewer.set_sdf_volume(
            mesh_result.grid, mesh_result.origin, mesh_result.dx)
        self._show_thickness(mesh_result)
        self._btn_fit.setEnabled(True)

        self._status.showMessage(
            f"Mesh SDF done — min={float(np.min(mesh_result.grid)):.4f} "
            f"max={float(np.max(mesh_result.grid)):.4f}  |  "
            f"Ready to fit ellipsoids."
        )

    def _on_sdf_failed(self, msg: str) -> None:
        self._sdf_worker = None
        self._progress_end()
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._status.showMessage(f"Mesh SDF failed: {msg}")

    def _ensure_sdf_idle(self) -> None:
        """Block until any in-flight SDF worker finishes.

        Call before mutating the shared ``SdfComputer`` (e.g. ``set_mesh``) so the
        worker never reads a mesh that's being swapped out from under it.
        """
        w = self._sdf_worker
        if w is not None and w.isRunning():
            w.wait()

    def closeEvent(self, event) -> None:
        # Persist the latest options-panel state before shutting down.
        self._save_panel_settings()
        # Don't let a running SDF worker be destroyed mid-flight (crash risk).
        self._ensure_sdf_idle()
        super().closeEvent(event)

    def update_ellipsoids(self, ellipsoid_set: EllipsoidSet) -> None:
        # SDF slices are shown for the mesh only; this just refreshes the
        # ellipsoid geometry in the unified viewport.
        self._ellipsoids = ellipsoid_set
        self._viewer.show_ellipsoids(self._ellipsoids)

    # ── fit / stop ────────────────────────────────────────────────────────

    def _on_fit_clicked(self):
        if self._last_mesh_result is None:
            self._status.showMessage("Compute mesh SDF first (press G or Compute).")
            return
        # Shared controls + the active shape's shape-specific kwargs.
        shape_kwargs = self._shape.fit_kwargs()
        self.start_optimization(
            num_ellipsoids=self._spin_num_ellipsoids.value(),
            method="adam",
            num_steps=self._spin_max_steps.value(),
            report_every=self._report_every,
            maintenance_every=0,
            max_ellipsoids=self._spin_max_ellipsoids.value(),
            symmetry=self._chk_symmetry.isChecked(),
            lr_init=self._spin_lr_init.value(),
            lr_final=self._spin_lr_final.value(),
            lr_decay_k=self._spin_lr_decay.value(),
            bone_aware=(self._grp_rig.isVisible()
                        and self._chk_bone_aware.isChecked()
                        and self._bone_centers is not None),
            bone_centers=self._bone_centers,
            advanced=self._settings,
            **shape_kwargs,   # sdf_mode, superfit*, local_fit*, soft_union, merge/spawn/split
        )

    def _on_stop_clicked(self):
        self._rig_panel.set_auto_pipeline_running(False)
        self.stop_optimization()
        self._stop_multipose()

    # ── thickness heatmap on the mesh view ──────────────────────────────

    def _show_thickness(self, mesh_result) -> None:
        """Colour the mesh view (top-left) as a local-thickness heatmap."""
        if getattr(mesh_result, "thickness", None) is None:
            return
        rng = self._viewer.show_thickness(
            mesh_result.thickness, mesh_result.origin,
            mesh_result.dx, mesh_result.n,
        )
        if rng is not None:
            self._status.showMessage(
                f"Mesh thickness: blue={rng[0]:.3f} → red={rng[1]:.3f} (world units)"
            )

    # ── async optimization ────────────────────────────────────────────────

    def start_optimization(
        self,
        num_ellipsoids: int = 10,
        method: str = "adam",
        num_steps: int = 2000,
        report_every: int = 20,
        sdf_mode: int = SDF_QUILEZ,
        maintenance_every: int = 200,
        superfit: bool = False,
        superfit_every: int = 150,
        densify_until_frac: float = 0.75,
        soft_union: bool = False,
        max_ellipsoids: int = 60,
        local_fit: bool = True,
        local_fit_start_frac: float = 0.25,
        symmetry: bool = False,
        merge_enabled: bool = True,
        spawn_enabled: bool = True,
        split_enabled: bool = True,
        lr_init: float = 0.01,
        lr_final: float = 0.0002,
        lr_decay_k: float = 7.0,
        containment_weight: float = 6.0,
        bone_aware: bool = False,
        bone_centers: np.ndarray | None = None,
        advanced: dict | None = None,
        primitive_shape: str = "ellipsoid",
        sq_eps1: float = 1.0,
        sq_eps2: float = 1.0,
    ) -> None:
        if self._last_mesh_result is None:
            self._status.showMessage("No mesh SDF available. Load a mesh and compute SDF first.")
            return

        self.stop_optimization()
        self._current_sdf_mode = sdf_mode

        r = self._last_mesh_result
        # SDF blowup: bake the uniform voxel offset into the fit target so the
        # ellipsoids fit the eroded/dilated surface (matches the slice preview).
        blowup = self._mesh_settings.blowup_voxels() * float(r.dx)
        target_grid = (r.grid + np.float32(blowup)).astype(np.float32) \
            if blowup != 0.0 else r.grid
        worker_kwargs = dict(
            sdf_target_np=target_grid,
            origin=r.origin,
            dx=r.dx,
            n=r.n,
            num_ellipsoids=num_ellipsoids,
            method=method,
            num_steps=num_steps,
            report_every=report_every,
            sdf_mode=sdf_mode,
            maintenance_every=maintenance_every,
            superfit=superfit,
            superfit_every=superfit_every,
            densify_until_frac=densify_until_frac,
            soft_union=soft_union,
            max_ellipsoids=max_ellipsoids,
            thickness_np=r.thickness,
            sdf_computer=self._sdf,
            local_fit=local_fit,
            local_fit_start_frac=local_fit_start_frac,
            symmetry_enabled=symmetry,
            merge_enabled=merge_enabled,
            spawn_underrep=spawn_enabled,
            split_enabled=split_enabled,
            lr_init=lr_init,
            lr_final=lr_final,
            lr_decay_k=lr_decay_k,
            containment_weight=containment_weight,
            bone_aware=bone_aware,
            bone_centers_np=bone_centers,
            primitive_shape=primitive_shape,
            sq_eps1=sq_eps1,
            sq_eps2=sq_eps2,
            parent=self,
        )
        # Advanced settings from the Settings dialog override the defaults above.
        # Their keys match OptimizationWorker kwargs exactly (see app_settings).
        if advanced:
            worker_kwargs.update(advanced)
        self._opt_worker = OptimizationWorker(**worker_kwargs)
        self._opt_phase = "global"
        self._opt_worker.step_visual.connect(self._on_opt_step_visual)
        self._opt_worker.step_sdf.connect(self._on_opt_step_sdf)
        self._opt_worker.phase_changed.connect(self._on_opt_phase_changed)
        self._opt_worker.local_progress.connect(self._on_opt_local_progress)
        self._opt_worker.region_changed.connect(self._on_opt_region_changed)
        self._opt_worker.prep_progress.connect(self._on_opt_prep_progress)
        self._opt_worker.op_events.connect(self._on_opt_op_events)
        self._opt_worker.analysis_regions.connect(self._on_opt_analysis_regions)
        self._opt_worker.finished.connect(self._on_opt_finished)
        self._viewer.clear_op_gizmos()      # drop markers from a previous run
        self._viewer.clear_analysis_regions()
        self._opt_worker.start()

        sdf_name = SDF_METHOD_NAMES.get(sdf_mode, "?")
        self._run_tracker.begin_run(
            mesh_name=self._current_mesh_name,
            method=method,
            num_ellipsoids=num_ellipsoids,
            grid_n=r.n,
        )
        self._dashboard.begin(num_steps, self._current_mesh_name, num_ellipsoids)
        self._analysis_tabs.setCurrentWidget(self._dashboard)

        self._btn_fit.setEnabled(False)
        self._btn_stop.setEnabled(True)
        # Reuse the status-bar progress bar to show how far training has got.
        self._opt_total_steps = max(1, int(num_steps))
        self._progress_begin("Preparing …")
        self._status.showMessage(
            f"Optimization started ({method}, {num_ellipsoids} ellipsoids, SDF: {sdf_name}) …"
        )

    def stop_optimization(self) -> None:
        if self._opt_worker is not None and self._opt_worker.isRunning():
            self._opt_worker.request_stop()
            self._opt_worker.wait()
            self._opt_worker = None
            self._run_tracker.finish_run()
            self._progress_end()
            self._btn_fit.setEnabled(self._last_mesh_result is not None)
            self._btn_stop.setEnabled(False)

    def _on_opt_step_visual(
            self,
            step: int,
            loss: float,
            centers: np.ndarray,
            radii: np.ndarray,
            rotations: np.ndarray,
            eps: np.ndarray | None = None,
    ) -> None:
        # Route rendering through the active shape plugin (superquadrics use the
        # per-primitive eps array for the deformed mesh).
        self._shape.render(self._viewer, centers, radii, rotations, eps)
        self._lbl_ell_count.setText(f"Count: {len(centers)}")

        # Age out / fade the SuperFit operation gizmos as training advances.
        self._viewer.tick_op_gizmos(step)

        # Training progress in the status-bar progress bar (step / total).
        phase = getattr(self, "_opt_phase", "global")
        total = getattr(self, "_opt_total_steps", 0)
        if total > 0:
            self._progress_set(min(1.0, step / total) * 100,
                               f"Optimizing [{phase.upper()}] · step {step}/{total}")

        if phase != "local":
            print(f"Step {step}: loss = {loss:.6f}")
            self._status.showMessage(
                f"Optimizing [{phase.upper()}] … step {step}/{total}  loss={loss:.6f}")

        # Keep ellipsoids reference for rig assignment
        self._ellipsoids = EllipsoidSet(device=self._device)
        self._ellipsoids.set_parameters(centers, radii, rotations)

        # Feed loss to the run tracker + dashboard only during the global phase:
        # local-fit emits all carry the same global step number and would pollute
        # the curve.
        if phase != "local":
            self._run_tracker.record_step(step, loss)
            self._dashboard.record(step, loss, len(centers), radii)

    def _on_opt_local_progress(self, current: int, total: int) -> None:
        """Live feedback during an isolated local fit (SuperFit child fitting)."""
        if self._progress_label.isVisible():
            self._progress_msg = f"Optimizing [LOCAL] · fit {current}/{total}"
            self._update_progress_label(self._sdf_progress.value())
        self._status.showMessage(
            f"Optimizing [LOCAL] … local fit {current}/{total}")

    def _on_opt_prep_progress(self, frac: float, label: str) -> None:
        """Show pre-training setup (symmetry/thickness/buffers/sampler) on the bar.

        For large SDFs this host-side preprocessing runs for a while before the
        first optimisation step; map it onto the bar so it is not dead time.
        """
        self._progress_set(float(frac) * 100.0, f"Preparing · {label}")

    def _on_opt_phase_changed(self, phase: str) -> None:
        """Track whether a global (Adam) or local (SuperFit) fit is running."""
        self._opt_phase = phase
        self._dashboard.set_phase(phase)
        self._status.showMessage(
            f"Optimizing [{phase.upper()}] … "
            + ("isolated local fit of new ellipsoids"
               if phase == "local" else "global optimisation"))

    def _on_opt_op_events(self, step: int, events) -> None:
        """Mark where SuperFit just acted (merge/split/spawn/fuse/delete).

        Each marker is a colour-coded box in the 3-D view that fades over the
        next 50 steps; toggle the whole layer with the "Operations" checkbox.
        """
        self._viewer.add_op_gizmos(step, events)

    def _on_opt_analysis_regions(self, step: int, regions) -> None:
        """Show the current densify analysis (over/under/bridging) as
        transparent spheres; toggle with the "Analysis" checkbox."""
        self._viewer.set_analysis_regions(regions)

    def _on_opt_region_changed(self, box) -> None:
        """Mark (or clear) the high-res region boxes currently being optimised.

        ``box`` is None (clear), a list of ``(aabb_min, aabb_max)`` boxes (the
        combined local fit emits one small box per optimised region), or a single
        ``(aabb_min, aabb_max)`` pair (legacy single-region path).
        """
        if box is None:
            self._viewer.clear_region_box()
        elif isinstance(box, list):
            self._viewer.show_region_boxes(box)
        else:
            aabb_min, aabb_max = box
            self._viewer.show_region_box(aabb_min, aabb_max)

    def _on_opt_step_sdf(
            self,
            step: int,
            loss: float,
            ell_grid: np.ndarray,
            ur_points: np.ndarray,
            ur_values: np.ndarray,
    ) -> None:
        # The ellipsoid SDF slice view was removed (SDF slices are shown for the
        # mesh only), so the per-step ellipsoid grid is no longer displayed.
        return

    def _on_opt_finished(self) -> None:
        self._status.showMessage("Optimization finished.")
        self._run_tracker.finish_run()
        self._dashboard.finish()
        self._opt_worker = None
        self._viewer.clear_region_box()
        self._viewer.clear_op_gizmos()
        self._viewer.clear_analysis_regions()
        self._progress_end()
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._btn_stop.setEnabled(False)

        # Enable bone assignment if rig mode is active
        if self._rig_panel.is_active and self._ellipsoids is not None:
            self._rig_panel._btn_assign.setEnabled(True)
