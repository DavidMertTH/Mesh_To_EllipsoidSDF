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

import os
import time
from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import warp as wp

import branding
import theme
import app_settings
from settings_dialog import SettingsDialog
import shape_plugins
from shape_plugins import (
    widget_value, set_widget_value, widget_change_signal, available_shapes,
)
from mesh_io import load_and_prepare, load_and_prepare_arrays
from sdf_compute import SdfComputer, SdfResult
from api_server import ApiServer
from rig_ingest import (
    assign_ellipsoids_to_bones,
    build_skeleton_from_bones,
    world_to_bone_local_entries,
)
from bone_separation import partition_mesh_by_bone
from bonesep_controller import BoneSeparationController
from batched_fit import BatchedFitWorker
from ellipsoid import EllipsoidSet, SDF_QUILEZ, SDF_METHOD_NAMES, best_device
from viewer3d import SceneViewer3D
from widgets import SdfSlicePanel
from optimization import OptimizationWorker
from run_tracker import RunTrackerPanel
from dashboard import DashboardPanel
from mesh_settings import MeshSettingsPanel, rotate_mesh
from rig_panel import RigModePanel, try_load_rigged
from rig_loader import RiggedMesh
from skinning import deform_mesh
from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from skeleton import Bone, Pose, Skeleton, mat4_compose

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

        # Meshes loaded this session from *outside* the mesh dir (drag-and-drop
        # of an external file, or a mesh pushed in over the Unity/HTTP API).
        # They are added to the selector so they stay selectable; this list lets
        # a folder rescan re-add them instead of dropping them.
        #   _extra_mesh_paths : {normalised-key → original path} of dropped
        #     external files (the original path preserves the display casing)
        #   _special_mesh_loaders : data-key → callable that re-displays a
        #     non-file mesh (e.g. the in-memory Unity mesh)
        self._extra_mesh_paths: dict = {}
        self._special_mesh_loaders: dict = {}
        self._special_mesh_labels: dict = {}
        self._unity_mesh_cache: dict = {}

        wp.init()
        self._device = best_device()
        _tick(0.20, "Initializing GPU …")

        self._sdf = SdfComputer(device=self._device)
        self._ellipsoids: EllipsoidSet | None = None

        self._last_mesh_result: SdfResult | None = None

        # ── Headless/Unity API state ────────────────────────────────────────
        # When a fit is driven over the HTTP API instead of the GUI, these track
        # the in-flight job so the SDF-done / step / finished slots can chain
        # compute → fit → result and report progress back to the API client.
        self._api_server: ApiServer | None = None
        self._api_job_id: str | None = None
        self._api_stage: str | None = None        # None | "sdf" | "fit"
        self._api_norm = None                     # NormalizationTransform
        self._api_options: dict | None = None
        self._api_last = None                     # (centers, radii, rotations)
        self._api_verts = None                    # original (un-normalized) verts
        self._api_rig: dict | None = None         # Unity skinning payload

        # Bone Separation: parallel (batched) controller.  ``_bonesep_ctl`` is the
        # orchestration state machine (None when idle); ``_batched_worker`` is the
        # single Adam loop fitting all bones at once; ``_region_sdf_active`` routes
        # the shared SDF worker's result to the controller during precompute.
        self._bonesep_ctl: BoneSeparationController | None = None
        self._batched_worker: BatchedFitWorker | None = None
        self._region_sdf_active: bool = False
        self._bonesep_on_complete = None
        self._bonesep_is_api: bool = False
        self._bonesep_fit_kwargs: dict = {}
        # Dedicated SDF computer/worker for the region-SDF precompute phase, kept
        # separate from ``self._sdf`` so per-bone grids never clobber the loaded
        # mesh's grid, slice view or viewport volume.
        self._region_sdf: SdfComputer | None = None
        self._region_sdf_worker: SdfWorker | None = None
        # Sequential per-bone fit (reverted from the batched all-bones loop): one
        # bone is fitted fully on its own region SDF with the single-bone
        # ``OptimizationWorker`` before the next bone's SDF is computed.
        self._region_fit_worker: OptimizationWorker | None = None
        self._region_fit_active: bool = False
        self._region_last_params = None            # (centers, radii, rotations)
        # Ellipsoids of the bones already fitted this run, kept so the viewport
        # shows the GROWING union (finished bones + current bone in progress)
        # instead of resetting to just the current bone every frame.
        self._bonesep_done_c: list = []
        self._bonesep_done_r: list = []
        self._bonesep_done_q: list = []

        # Mesh-Blowup preview: cached per-bone submeshes + their centroids for
        # the exploded region view (None when no preview is active).
        self._region_parts: list | None = None
        self._region_centroids: np.ndarray | None = None
        self._region_global_center: np.ndarray | None = None
        self._region_colors: np.ndarray | None = None

        # Single unified viewport: mesh + skeleton + ellipsoids together.
        self._viewer = SceneViewer3D()
        self._viewer.connect_ellipsoid_metric_changed(self._on_view_ellipsoid_metric_changed)
        self._active_ellipsoid_metric = self._viewer.ellipsoid_metric_mode()
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

        # Second bar: the overall Bone-Separation progress (how many bone regions
        # are done), shown to the RIGHT of the per-bone bar above.  Only visible
        # during a bone-separation run; the per-bone bar tracks the current bone.
        self._overall_label = QtWidgets.QLabel("")
        self._overall_label.setVisible(False)
        self._status.addPermanentWidget(self._overall_label)
        self._overall_progress = QtWidgets.QProgressBar()
        self._overall_progress.setRange(0, 1000)
        self._overall_progress.setMinimumWidth(220)
        self._overall_progress.setMaximumWidth(320)
        self._overall_progress.setTextVisible(False)
        self._overall_progress.setVisible(False)
        self._status.addPermanentWidget(self._overall_progress)
        self._overall_msg = ""
        self._overall_anim = QtCore.QPropertyAnimation(
            self._overall_progress, b"value", self)
        self._overall_anim.setDuration(320)
        self._overall_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._overall_anim.valueChanged.connect(self._update_overall_label)

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
        # Live-render decoupling: the optimizer can emit ``step_visual`` far faster
        # than the GUI can render (tiny bones in Bone-Separation hit ~800 steps/s).
        # Doing the heavy work (3D rebuild, dashboard re-plot, animated progress
        # bars) in the signal slot floods the GUI event queue and freezes the UI
        # (Stop needs dozens of clicks).  Instead the slot only *stashes* the
        # latest frame; a GUI-thread timer renders it at a fixed rate, so the
        # update cost is bounded no matter how fast the optimizer runs.
        self._pending_visual: tuple | None = None
        self._visual_timer = QtCore.QTimer(self)
        self._visual_timer.setInterval(45)               # ~22 FPS GUI refresh
        self._visual_timer.timeout.connect(self._flush_visual)
        self._brand_icon_timer = QtCore.QTimer(self)
        self._brand_icon_timer.setSingleShot(True)
        self._brand_icon_timer.setInterval(250)
        self._brand_icon_timer.timeout.connect(self._update_brand_icon)
        self._color_dialog_active = False
        self._applying_theme = False
        self._last_theme_signature = None
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
        self._mesh_settings.regionPreviewToggled.connect(
            self._on_region_preview_toggled)
        self._mesh_settings.regionBlowupChanged.connect(
            self._on_region_blowup_changed)

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
        # _load_mesh selects it in the combo (via _select_loaded_mesh) itself.
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

        # Fit scope: whole mesh at once, or one independent SDF+fit per bone.
        self._cmb_fit_scope = QtWidgets.QComboBox()
        self._cmb_fit_scope.addItem("Full Object", "full")
        self._cmb_fit_scope.addItem("Bone Separation", "bone")
        self._cmb_fit_scope.setToolTip(
            "Full Object: one SDF for the whole mesh (default).\n"
            "Bone Separation: carve the mesh into per-bone regions, then build a\n"
            "separate SDF for each and fit it independently, one bone at a time.\n"
            "Needs skinning data (rigged FBX or a Unity rig payload).")

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
        rig_form.addRow("Fit scope:", self._cmb_fit_scope)
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

    def _make_appearance_mode_row(self) -> QtWidgets.QWidget:
        """Light/dark selector + a 'Sync with OS' checkbox for the Settings tab.

        Changes apply live (and persist) — toggling re-themes the whole UI via
        Qt's colour-scheme hint, which fires the palette-change handler.
        """
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        self._chk_sync_os = QtWidgets.QCheckBox("Sync with OS")
        self._chk_sync_os.setToolTip(
            "Follow the operating system's light/dark setting.")
        v.addWidget(self._chk_sync_os)

        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(18, 0, 0, 0)
        h.setSpacing(12)
        self._rb_light = QtWidgets.QRadioButton("Light")
        self._rb_dark = QtWidgets.QRadioButton("Dark")
        self._mode_btn_group = QtWidgets.QButtonGroup(w)
        self._mode_btn_group.addButton(self._rb_light)
        self._mode_btn_group.addButton(self._rb_dark)
        h.addWidget(self._rb_light)
        h.addWidget(self._rb_dark)
        h.addStretch()
        v.addWidget(row)

        # Initialise from the persisted mode.
        mode = theme.MODE
        self._chk_sync_os.setChecked(mode == "system")
        # When syncing, reflect the *current* effective scheme in the radios.
        effective_light = (mode == "light") or (
            mode == "system" and not theme.is_dark_mode())
        self._rb_light.setChecked(effective_light)
        self._rb_dark.setChecked(not effective_light)
        self._update_mode_controls_enabled()

        self._chk_sync_os.toggled.connect(self._on_appearance_mode_changed)
        self._rb_light.toggled.connect(self._on_appearance_mode_changed)
        return w

    def _update_mode_controls_enabled(self) -> None:
        """Grey out the Light/Dark radios while 'Sync with OS' is active."""
        sync = self._chk_sync_os.isChecked()
        self._rb_light.setEnabled(not sync)
        self._rb_dark.setEnabled(not sync)

    def _on_appearance_mode_changed(self, *_) -> None:
        """Apply + persist the chosen appearance mode (live re-theme)."""
        self._update_mode_controls_enabled()
        if self._chk_sync_os.isChecked():
            mode = "system"
        else:
            mode = "light" if self._rb_light.isChecked() else "dark"
        theme.apply_mode(mode)
        theme.save_mode()
        # _apply_theme() is normally triggered by the palette-change event, but
        # call it directly too so the refresh is immediate and reliable.
        self._apply_theme()

    def _open_settings_dialog(self) -> None:
        """Open (or reopen) the Settings window and commit/persist on OK."""
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(
                self._settings, self._make_color_row(),
                self._make_appearance_mode_row(), self)
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
            ("fit_scope",       self._cmb_fit_scope,       "combo_data"),
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
        # SDF blowup lives on the Mesh-settings panel (not a shared widget) and is
        # reset to 0 on every mesh load.  The startup mesh has already loaded by
        # now (see __init__ order), so restoring the persisted value here makes it
        # survive into the session.
        if isinstance(shared, dict) and "blowup" in shared:
            try:
                vox = float(shared["blowup"])
                self._mesh_settings.set_blowup_voxels(vox)
                self._viewer.set_sdf_blowup(vox)
            except Exception:
                pass
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
        self._mesh_settings.blowupChanged.connect(kick)   # persist SDF blowup too
        for shape in self._shapes:
            shape.options_widget()          # ensure widgets exist before wiring
            shape.connect_changed(lambda: self._panel_save_timer.start())

    def _save_panel_settings(self) -> None:
        shared = {key: widget_value(w, kind)
                  for key, w, kind in self._shared_setting_specs()}
        shared["blowup"] = self._mesh_settings.blowup_voxels()
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
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.PaletteChange,
        ) and not getattr(self, "_applying_theme", False):
            self._apply_theme()

    def _apply_theme(self) -> None:
        if not hasattr(self, "_btn_primary"):
            return                      # window not fully built yet
        is_dark = theme.is_dark_mode()
        signature = (theme.BLUE, theme.YELLOW, is_dark)
        if getattr(self, "_last_theme_signature", None) == signature:
            return
        self._last_theme_signature = signature
        if getattr(self, "_applying_theme", False):
            return
        self._applying_theme = True
        try:
            # pyqtgraph defaults for any widgets created afterwards.
            if is_dark:
                pg.setConfigOptions(foreground="d", background="k")
            else:
                pg.setConfigOptions(foreground="k", background="w")
            self._viewer.apply_theme()
            self._mesh_sdf_panel.apply_theme()
            self._run_tracker.apply_theme()
            self._dashboard.apply_theme()
            if not getattr(self, "_color_dialog_active", False):
                self._schedule_brand_icon_update()
                self._clear_custom_ui_style()
            self._refresh_color_swatches()
            self._style_progress_bar()
        finally:
            self._applying_theme = False

    def _schedule_brand_icon_update(self) -> None:
        if hasattr(self, "_brand_icon_timer"):
            self._brand_icon_timer.start()

    def _update_brand_icon(self) -> None:
        # Do not set a new QIcon while the native/non-native QColorDialog is
        # actively emitting live colour changes; on Windows that can destabilise
        # the dialog/window-icon plumbing.  Reschedule until the dialog closes.
        if getattr(self, "_color_dialog_active", False):
            self._schedule_brand_icon_update()
            return
        icon = branding.make_sdf_icon()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)
        self.setWindowIcon(icon)

    def _clear_custom_ui_style(self) -> None:
        """Use Qt's native widget styling; branding stays in logo/progress/view."""
        if self.styleSheet():
            self.setStyleSheet("")

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
        css = f"""
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
        """
        self._sdf_progress.setStyleSheet(css)
        if hasattr(self, "_overall_progress"):
            self._overall_progress.setStyleSheet(css)

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

    # ── overall (bone-separation) progress bar ────────────────────────────

    def _update_overall_label(self, value: int) -> None:
        """Sync the overall bar's label percentage to *value* (in bar units)."""
        span = max(1, self._overall_progress.maximum())
        pct = int(round(value / span * 100))
        self._overall_label.setText(
            f"{self._overall_msg} · {pct}%" if self._overall_msg else f"{pct}%")

    def _overall_begin(self, msg: str) -> None:
        """Show the overall progress bar, reset to 0, and set the label."""
        self._overall_msg = msg
        self._overall_anim.stop()
        self._overall_progress.setValue(0)
        self._overall_progress.setVisible(True)
        self._update_overall_label(0)
        self._overall_label.setVisible(True)

    def _overall_set(self, value: float, msg: str | None = None) -> None:
        """Lerp the overall bar toward *value* (0–100 %); update the label."""
        if msg is not None:
            self._overall_msg = msg
        self._overall_progress.setVisible(True)
        self._overall_label.setVisible(True)
        span = self._overall_progress.maximum()
        target = int(max(0, min(span, round(value / 100.0 * span))))
        self._overall_anim.stop()
        self._overall_anim.setStartValue(self._overall_progress.value())
        self._overall_anim.setEndValue(target)
        self._overall_anim.start()
        self._update_overall_label(self._overall_progress.value())

    def _overall_end(self) -> None:
        """Hide the overall progress bar and its label."""
        self._overall_anim.stop()
        self._overall_msg = ""
        self._overall_progress.setVisible(False)
        self._overall_label.clear()
        self._overall_label.setVisible(False)

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
        self._color_dialog_active = True

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
        # Remember the final choice across restarts (fires after accept/revert),
        # then update the dynamic icon once the colour dialog is gone.
        def _finished(_result: int) -> None:
            self._color_dialog_active = False
            theme.save_colors()
            self._last_theme_signature = None
            self._apply_theme()
            self._update_brand_icon()

        dlg.finished.connect(_finished)
        dlg.open()

    # ── mesh directory scanning ───────────────────────────────────────────

    @staticmethod
    def _norm_path(p) -> str:
        """Normalised absolute path used as the combo's per-file data key."""
        return os.path.normcase(os.path.abspath(str(p)))

    def _scan_mesh_dir(self):
        self._mesh_combo.blockSignals(True)
        prev_data = self._mesh_combo.currentData()
        self._mesh_combo.clear()
        self._mesh_combo.addItem("— select mesh —")

        if self._mesh_dir.is_dir():
            files = sorted(
                f for f in self._mesh_dir.iterdir()
                if f.is_file() and f.suffix.lower() in MESH_EXTENSIONS
            )
            for f in files:
                self._mesh_combo.addItem(f.name, self._norm_path(f))

        # Re-add session meshes loaded from outside the dir so a rescan doesn't
        # drop them from the selector.  External files that no longer exist are
        # forgotten; the special (in-memory) loaders are always re-added.
        self._extra_mesh_paths = {
            k: orig for k, orig in self._extra_mesh_paths.items()
            if os.path.isfile(orig)}
        for key, orig in self._extra_mesh_paths.items():
            if self._mesh_combo.findData(key) < 0:
                self._mesh_combo.addItem(Path(orig).name, key)
        for key, label in self._special_mesh_labels.items():
            if self._mesh_combo.findData(key) < 0:
                self._mesh_combo.addItem(label, key)

        if prev_data is not None:
            idx = self._mesh_combo.findData(prev_data)
            if idx >= 1:
                self._mesh_combo.setCurrentIndex(idx)
        self._mesh_combo.blockSignals(False)

    def _register_file_in_combo(self, path: str) -> int:
        """Ensure *path* has a selector entry (adding external files); return idx."""
        key = self._norm_path(path)
        idx = self._mesh_combo.findData(key)
        if idx < 0:
            # A file from outside the mesh dir — add it and remember it (with its
            # original path, to preserve display casing) so a later folder rescan
            # keeps it selectable.
            self._extra_mesh_paths[key] = str(path)
            self._mesh_combo.addItem(Path(path).name, key)
            idx = self._mesh_combo.findData(key)
        return idx

    def _select_loaded_mesh(self, path: str) -> None:
        """Register *path* if needed and show it as the current selection."""
        idx = self._register_file_in_combo(path)
        self._mesh_combo.blockSignals(True)
        self._mesh_combo.setCurrentIndex(idx if idx >= 1 else 0)
        self._mesh_combo.blockSignals(False)

    def _on_combo_selected(self, idx: int):
        if idx < 1:
            return
        data = self._mesh_combo.itemData(idx)
        if data in self._special_mesh_loaders:
            self._special_mesh_loaders[data]()
            return
        if data:
            self._load_mesh(data)

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
        self._mesh_settings.set_region_available(False)  # no rig → no regions
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

            self._select_loaded_mesh(path)

            self._status.showMessage(
                f"Loaded: {path} | verts={len(verts)} faces={len(faces)} | device={self._device}"
            )
            self._on_compute_all()

        except Exception as e:
            self._status.showMessage(f"Failed to load: {path} ({e})")

    def _load_rigged_mesh(self, path: str, rigged):
        """Load a rigged mesh and activate rig mode."""
        self._precomputed_meshes = {}
        self._pending_pose_mesh = None
        self._rig_panel.setVisible(True)  # Reveal rig/FBX options
        self._rig_panel.setChecked(True)
        self._rig_panel.set_rigged_mesh(rigged)
        self._grp_rig.setVisible(True)    # Reveal bone-awareness option
        self._mesh_settings.set_region_available(True)  # enable Mesh Blowup
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
        self._select_loaded_mesh(path)

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

            pose = self._rig_panel.pose_at(pose_index)
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
            f"Pose {pose_index}: {self._rig_panel.pose_at(pose_index).name}"
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
        # Drop any exploded region preview from the previous mesh (reset() below
        # unchecks the toggle silently, so clear the cached geometry here).
        self._clear_region_preview_state()
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
        # Keep the exploded region preview (if any) aligned with the rotation.
        if self._region_parts is not None:
            self._refresh_region_preview()
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
            poses=self._rig_panel.active_poses,
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
        pose = self._rig_panel.pose_at(pose_index)

        # Use pre-computed deformed mesh (instant)
        cached = getattr(self, '_precomputed_meshes', {})
        if pose_index in cached:
            self._viewer.show_mesh(cached[pose_index], rm.faces)
        self._show_skeleton_for_pose(rm, pose)

        # Update pose slider to match
        n_active = len(self._rig_panel.active_poses)
        self._rig_panel._slider_pose.blockSignals(True)
        self._rig_panel._slider_pose.setValue(pose_index)
        self._rig_panel._slider_pose.blockSignals(False)
        self._rig_panel._lbl_pose.setText(
            f"{pose_index} / {max(0, n_active - 1)}")
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

        # Chain into the fit when an API job is waiting for its SDF grid.
        if self._api_job_id is not None and self._api_stage == "sdf":
            self._api_start_fit()

    def _on_sdf_failed(self, msg: str) -> None:
        self._sdf_worker = None
        self._progress_end()
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._status.showMessage(f"Mesh SDF failed: {msg}")
        if self._api_job_id is not None and self._api_stage == "sdf":
            self._api_fail(self._api_job_id, f"SDF compute failed: {msg}")

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

    def _gather_fit_kwargs(self) -> dict:
        """Build the ``start_optimization`` kwargs from the live option widgets.

        Excludes ``num_ellipsoids`` / ``max_ellipsoids`` (the caller sets those,
        which lets Bone Separation override the count per bone) and the
        bone-awareness penalty (caller-specific).  Used by both the full-object
        fit and the per-bone fits so they share one source of settings.
        """
        shape_kwargs = self._shape.fit_kwargs()
        return dict(
            method="adam",
            num_steps=self._spin_max_steps.value(),
            report_every=self._report_every,
            maintenance_every=0,
            symmetry=self._chk_symmetry.isChecked(),
            lr_init=self._spin_lr_init.value(),
            lr_final=self._spin_lr_final.value(),
            lr_decay_k=self._spin_lr_decay.value(),
            advanced=self._settings,
            **shape_kwargs,   # sdf_mode, superfit*, local_fit*, soft_union, merge/spawn/split
        )

    def _on_fit_clicked(self):
        if self._last_mesh_result is None:
            self._status.showMessage("Compute mesh SDF first (press G or Compute).")
            return
        # Bone Separation: train each bone on its own SDF, one at a time.
        if (self._cmb_fit_scope.currentData() == "bone"
                and self._has_skin_for_bonesep()):
            self._start_bone_separation_manual()
            return
        self.start_optimization(
            num_ellipsoids=self._spin_num_ellipsoids.value(),
            max_ellipsoids=self._spin_max_ellipsoids.value(),
            bone_aware=(self._grp_rig.isVisible()
                        and self._chk_bone_aware.isChecked()
                        and self._bone_centers is not None),
            bone_centers=self._bone_centers,
            **self._gather_fit_kwargs(),
        )

    def _on_stop_clicked(self):
        self._rig_panel.set_auto_pipeline_running(False)
        # Cancel the bone-separation pipeline (controller + workers).
        if self._bonesep_ctl is not None:
            self._bonesep_ctl.cancel()
        if (self._batched_worker is not None
                and self._batched_worker.isRunning()):
            self._visual_timer.stop()
            self._batched_worker.request_stop()
            self._batched_worker.wait()
        # Sequential per-bone fit: stop the current bone's OptimizationWorker.
        if (self._region_fit_worker is not None
                and self._region_fit_worker.isRunning()):
            self._visual_timer.stop()
            self._region_fit_worker.request_stop()
            self._region_fit_worker.wait()
        self._region_fit_worker = None
        self._region_fit_active = False
        self._batched_worker = None
        self._bonesep_ctl = None
        self._overall_end()             # hide the overall bone-separation bar
        self.stop_optimization()
        self._stop_multipose()

    # ── Bone Separation: sequential per-bone fitting ─────────────────────
    #
    # The skinned mesh is carved into one compact submesh per bone
    # (``partition_mesh_by_bone``).  Each bone is then handled on its own,
    # one after another, orchestrated by :class:`BoneSeparationController`:
    #
    #   for each fit-bone:
    #     1. compute its isolated region SDF (``compute_region_sdf``), then
    #     2. fit it with the single-bone ``OptimizationWorker`` against THAT
    #        SDF (``run_region_fit``) — so every bone gets the full maintenance /
    #        densify / local-fit / symmetry treatment, never seeing any other
    #        bone's SDF.
    #
    # (This replaces an earlier GPU-batched all-bones-in-one-Adam-loop engine,
    # ``batched_fit.BatchedFitWorker``, which converged poorly; that module is
    # left in the tree but is no longer used by this pipeline.)
    #
    # Symmetry is handled by the controller: paired (left/right) bones are
    # mirrored from their fitted source; on-plane (centre) bones are fitted with
    # the single-bone optimizer's own symmetry enforcement.
    #
    # This MainWindow is the controller's *host* (see ``BoneSepHost``): it owns
    # the async SDF + per-bone fit workers and calls back into the controller's
    # ``on_sdf_ready`` / ``on_sdf_failed`` / ``fit_progress`` /
    # ``on_region_fit_finished`` on the GUI thread.  ``accum`` handed to the
    # completion callback is
    # ``(world_centers, world_radii, world_rotations, bone_indices)``.

    def _has_skin_for_bonesep(self) -> bool:
        """True when a rigged mesh with usable skinning + mapper is loaded."""
        rm = self._rig_panel.rigged_mesh
        return bool(
            self._rig_panel.is_active and rm is not None
            and getattr(rm, "skin_joints", None) is not None
            and getattr(rm, "skin_weights", None) is not None
            and self._rig_panel.mapper is not None)

    def _start_bone_separation_manual(self) -> None:
        """GUI entry point: bone-separate the loaded rigged mesh, then fit."""
        rm = self._rig_panel.rigged_mesh
        if rm is None:
            self._status.showMessage("Load a rigged mesh first.")
            return
        self.stop_optimization()
        self._stop_multipose()
        self._begin_bone_separation(
            rm.vertices, rm.faces, rm.skin_joints, rm.skin_weights,
            on_complete=lambda accum: self._bonesep_complete_gui(accum, rm),
            base_kwargs=self._gather_fit_kwargs(),
            is_api=False,
        )

    def _begin_bone_separation(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        skin_joints: np.ndarray,
        skin_weights: np.ndarray,
        on_complete,
        base_kwargs: dict,
        *,
        is_api: bool = False,
    ) -> None:
        """Partition the mesh and launch the parallel (batched) fit pipeline."""
        try:
            parts = partition_mesh_by_bone(
                verts, faces, skin_joints, skin_weights,
                total_budget=self._spin_num_ellipsoids.value(),
                total_max=self._spin_max_ellipsoids.value(),
            )
        except Exception as e:
            if is_api and self._api_job_id is not None:
                self._api_fail(self._api_job_id, f"bone partition failed: {e}")
            else:
                self._status.showMessage(f"Bone Separation failed: {e}")
            return

        if not parts:
            if is_api and self._api_job_id is not None:
                self._api_fail(self._api_job_id,
                               "no bone produced a usable region")
            else:
                self._status.showMessage(
                    "Bone Separation: no bone produced a usable region.")
            return

        # Stash the completion routing for ``bonesep_complete`` / ``bonesep_failed``.
        self._bonesep_on_complete = on_complete
        self._bonesep_is_api = bool(is_api)
        self._bonesep_fit_kwargs = dict(base_kwargs)
        # Reset the growing-union display accumulator for this run.
        self._bonesep_done_c = []
        self._bonesep_done_r = []
        self._bonesep_done_q = []

        self._btn_fit.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._overall_begin("Gesamt · Bone Separation")
        self._progress_begin("Bone Separation …")

        # The controller is GUI-free; this window is its host (see BoneSepHost).
        self._bonesep_ctl = BoneSeparationController(
            self, parts, base_kwargs,
            symmetry=bool(base_kwargs.get("symmetry")),
            mesh_vertices=verts,
        )
        self._bonesep_ctl.begin()

    # ── BoneSepHost interface (called by the controller on the GUI thread) ──

    def compute_region_sdf(self, vertices: np.ndarray, faces: np.ndarray,
                           symmetry: bool) -> None:
        """Start one isolated region-SDF compute on a dedicated worker.

        Uses a private ``SdfComputer`` so the per-bone grid never clobbers the
        loaded mesh's grid / slice view / viewport volume.  The result is routed
        back to the controller via ``on_sdf_ready`` / ``on_sdf_failed``.
        """
        if self._region_sdf is None:
            self._region_sdf = SdfComputer(device=self._device)
        if (self._region_sdf_worker is not None
                and self._region_sdf_worker.isRunning()):
            self._region_sdf_worker.wait()
        self._region_sdf.set_mesh(vertices, faces)
        n = self._mesh_sdf_panel.requested_n
        margin = self._slider_margin.value() / 100.0
        self._region_sdf_active = True
        w = SdfWorker(self._region_sdf, n, margin, parent=self,
                      symmetry=bool(symmetry))
        self._region_sdf_worker = w
        w.progress.connect(self._on_region_sdf_progress)
        w.done.connect(self._on_region_sdf_done)
        w.failed.connect(self._on_region_sdf_failed)
        w.start()

    def _on_region_sdf_progress(self, frac: float, msg: str) -> None:
        self._progress_set(float(frac) * 100.0, f"Region-SDF · {msg}")

    def _on_region_sdf_done(self, result) -> None:
        self._region_sdf_worker = None
        self._region_sdf_active = False
        if self._bonesep_ctl is not None:
            self._bonesep_ctl.on_sdf_ready(result)

    def _on_region_sdf_failed(self, msg: str) -> None:
        self._region_sdf_worker = None
        self._region_sdf_active = False
        if self._bonesep_ctl is not None:
            self._bonesep_ctl.on_sdf_failed(msg)

    def run_region_fit(self, result, part, symmetry) -> None:
        """Fit ONE bone on its own region SDF with the single-bone optimizer.

        This is the reverted *sequential* path: each bone gets the full
        single-bone ``OptimizationWorker`` treatment (maintenance / densify /
        local fit / symmetry) against its isolated region SDF, one bone at a
        time.  ``self._region_sdf`` (the computer that produced ``result``) is
        reused as the worker's ``sdf_computer`` so the local fit can decode the
        region grid.  On completion the fitted ellipsoids are handed back to the
        controller via ``on_region_fit_finished``.
        """
        # Hard guarantee: never run two optimizer threads at once.  Warp's
        # autodiff tape is process-global, so an overlap raises "entering a tape
        # while one is already active".  If the previous bone's worker has not
        # been joined yet, join it here before starting this one.
        prev = self._region_fit_worker
        if prev is not None:
            if prev.isRunning():
                prev.wait()
            self._region_fit_worker = None

        fk = dict(self._bonesep_fit_kwargs)
        # Map the gathered fit kwargs to OptimizationWorker kwargs.  All keys
        # already match except the two renames handled below; ``symmetry`` and
        # ``advanced`` are popped (replaced / merged explicitly).
        advanced = fk.pop("advanced", None)
        fk.pop("symmetry", None)
        spawn = fk.pop("spawn_enabled", True)

        worker_kwargs = dict(fk)
        worker_kwargs.update(
            sdf_target_np=np.asarray(result.grid, dtype=np.float32),
            origin=result.origin,
            dx=result.dx,
            n=result.n,
            thickness_np=getattr(result, "thickness", None),
            sdf_computer=self._region_sdf,
            num_ellipsoids=int(part.budget),
            max_ellipsoids=int(part.max_budget),
            spawn_underrep=bool(spawn),
            symmetry_enabled=bool(symmetry is not None),
            parent=self,
        )
        if advanced:
            worker_kwargs.update(advanced)

        worker = OptimizationWorker(**worker_kwargs)
        worker.set_live_metric(getattr(self, "_active_ellipsoid_metric", "default"))
        self._region_fit_worker = worker
        self._region_fit_active = True
        self._region_last_params = None
        self._pending_visual = None
        self._visual_timer.start()          # decouple GUI refresh from step rate
        worker.step_visual.connect(self._on_region_fit_step)
        worker.ellipsoid_metrics.connect(self._on_opt_ellipsoid_metrics)
        worker.prep_progress.connect(self._on_region_fit_prep)
        worker.finished.connect(self._on_region_fit_finished)
        worker.start()

    def report_overall(self, frac: float, msg: str) -> None:
        self._overall_set(float(np.clip(frac, 0.0, 1.0)) * 100.0, msg)

    def report_current(self, frac: float, msg: str) -> None:
        self._progress_set(float(np.clip(frac, 0.0, 1.0)) * 100.0, msg)

    def set_status(self, msg: str) -> None:
        self._status.showMessage(msg)

    def bonesep_complete(self, accum: tuple) -> None:
        """Controller finished: tear down state, then run the completion cb."""
        on_complete = self._bonesep_on_complete
        self._bonesep_ctl = None
        self._batched_worker = None
        self._region_fit_worker = None
        self._region_fit_active = False
        self._bonesep_on_complete = None
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._btn_stop.setEnabled(False)
        self._overall_end()
        self._progress_end()
        if on_complete is not None:
            on_complete(accum)

    def bonesep_failed(self, msg: str) -> None:
        """Controller aborted: tear down state and report the failure."""
        is_api = self._bonesep_is_api
        self._bonesep_ctl = None
        self._batched_worker = None
        self._region_fit_worker = None
        self._region_fit_active = False
        self._bonesep_on_complete = None
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._btn_stop.setEnabled(False)
        self._overall_end()
        self._progress_end()
        if is_api and self._api_job_id is not None:
            self._api_fail(self._api_job_id, msg)
        else:
            self._status.showMessage(f"Bone Separation: {msg}")

    # ── sequential per-bone fit worker handlers ──────────────────────────

    def _on_region_fit_step(self, step: int, loss: float,
                            centers: np.ndarray, radii: np.ndarray,
                            rotations: np.ndarray,
                            eps: np.ndarray | None = None) -> None:
        """Hot path: stash the latest frame + record it as this bone's result.

        The CURRENT bone's params alone are kept in ``_region_last_params`` (that
        is what gets handed back to the controller).  The DISPLAYED frame, though,
        is the union of all already-fitted bones plus this bone in progress, so
        the viewport shows the skeleton filling in rather than resetting to a
        single bone each time.
        """
        self._region_last_params = (centers, radii, rotations)
        disp_c, disp_r, disp_q = centers, radii, rotations
        # Prepend finished bones (ellipsoid mode only — superquadric eps arrays
        # can't be unioned without their own eps, which we don't retain).
        if eps is None and self._bonesep_done_c:
            disp_c = np.vstack([*self._bonesep_done_c, centers])
            disp_r = np.vstack([*self._bonesep_done_r, radii])
            disp_q = np.vstack([*self._bonesep_done_q, rotations])
        self._pending_visual = (int(step), float(loss),
                                disp_c, disp_r, disp_q, eps)
        if self._bonesep_ctl is not None:
            total = max(1, int(self._bonesep_fit_kwargs.get("num_steps", 1)))
            self._bonesep_ctl.fit_progress(step / total)

    def _on_region_fit_prep(self, frac: float, label: str) -> None:
        self._progress_set(float(frac) * 100.0, f"Bone-Fit · {label}")

    def _on_region_fit_finished(self) -> None:
        """One bone done: flush its final frame, hand its ellipsoids back."""
        self._visual_timer.stop()
        # Fully join the finished worker BEFORE advancing.  Warp's autodiff tape
        # is process-global, so the next bone's worker must not enter its tape
        # while this one's thread is still tearing down — otherwise Warp raises
        # "entering a tape while one is already active".
        worker = self._region_fit_worker
        self._region_fit_worker = None
        if worker is not None:
            worker.wait()
        self._flush_visual()
        self._pending_visual = None     # drop any stray frame a late timer tick
        self._region_fit_active = False
        params = self._region_last_params
        if params is None:
            c = np.zeros((0, 3), np.float32)
            r = np.zeros((0, 3), np.float32)
            q = np.zeros((0, 4), np.float32)
        else:
            c, r, q = params
        self._region_last_params = None
        # Keep this bone on screen while later bones fit (growing-union display).
        if len(c):
            self._bonesep_done_c.append(np.asarray(c, np.float32).reshape(-1, 3))
            self._bonesep_done_r.append(np.asarray(r, np.float32).reshape(-1, 3))
            self._bonesep_done_q.append(np.asarray(q, np.float32).reshape(-1, 4))
        if self._bonesep_ctl is not None:
            self._bonesep_ctl.on_region_fit_finished(c, r, q)

    def _bonesep_complete_gui(self, accum, rm) -> None:
        """GUI completion: show the assembled ellipsoids + bone-local mapping."""
        centers, radii, rotations, bone_idx = accum
        self._ellipsoids = EllipsoidSet(device=self._device)
        self._ellipsoids.set_parameters(centers, radii, rotations)
        self._viewer.show_ellipsoids(self._ellipsoids)
        self._lbl_ell_count.setText(f"Count: {len(centers)}")

        mapper = self._rig_panel.mapper
        if mapper is not None:
            try:
                bl = mapper.world_to_local(
                    centers, radii, rotations, bone_idx, pose=None)
                self._rig_panel.set_bone_local(bl)
                self._rig_panel._btn_assign.setEnabled(True)
            except Exception as e:
                self._status.showMessage(
                    f"Bone Separation: local mapping failed: {e}")
                return
        self._status.showMessage(
            f"Bone Separation done — {len(centers)} ellipsoids across "
            f"{len(np.unique(bone_idx))} bone(s).")

    def _api_bonesep_complete(self, accum) -> None:
        """API completion: denormalize, map to bone-local, publish v2 result."""
        job_id = self._api_job_id
        if job_id is None or self._api_server is None:
            self._api_reset()
            return
        try:
            centers, radii, rotations, bone_idx = accum
            t = self._api_norm
            centers_o = np.array([t.to_original_point(c) for c in centers],
                                 dtype=np.float64)
            radii_o = np.array([t.to_original_length(r) for r in radii],
                               dtype=np.float64)
            entries = world_to_bone_local_entries(
                centers_o, radii_o, rotations, bone_idx, self._api_rig)
            result = {
                "version": 2,
                "coordinate_system": "right_hand_y_up",
                "quaternion_convention": "xyzw",
                "rigged": True,
                "count": len(entries),
                "ellipsoids": entries,
            }
            self._api_server.registry.update(
                job_id, state="done", result=result, count=len(entries))
            self._status.showMessage(
                f"API fit {job_id[:8]} done — {len(entries)} ellipsoids "
                f"(bone-separation)")
        except Exception as e:
            self._api_server.registry.update(
                job_id, state="error", error=str(e))
            self._status.showMessage(f"API fit {job_id[:8]} failed: {e}")
        finally:
            self._api_reset()

    # ── Mesh Blowup: exploded per-bone region preview ────────────────────
    #
    # A verification view for the Bone-Separation carving: partition the loaded
    # rigged mesh into the same per-bone submeshes that Bone-Separation fits, and
    # push each region radially outward from the shared centroid so the regions
    # (and their seam overlap) can be inspected.  The spread is a pure
    # translation, so dragging the slider is cheap.

    @staticmethod
    def _region_palette(n: int) -> np.ndarray:
        """Distinct, stable RGBA colour per bone (golden-ratio hue spacing)."""
        import colorsys
        cols = np.ones((max(1, n), 4), dtype=np.float32)
        for i in range(n):
            h = (i * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.62, 1.0)
            cols[i] = (r, g, b, 1.0)
        return cols

    def _clear_region_preview_state(self) -> None:
        """Drop the cached preview geometry and remove the viewport item."""
        self._region_parts = None
        self._region_centroids = None
        self._region_global_center = None
        self._region_colors = None
        self._viewer.clear_region_preview()

    def _on_region_preview_toggled(self, on: bool) -> None:
        """Build (or tear down) the exploded per-bone region preview."""
        if not on:
            self._clear_region_preview_state()
            self._status.showMessage("Mesh Blowup preview off.")
            return
        if not self._has_skin_for_bonesep():
            self._mesh_settings.set_region_available(False)
            self._status.showMessage(
                "Mesh Blowup needs a rigged mesh with skin weights.")
            return
        rm = self._rig_panel.rigged_mesh
        try:
            parts = partition_mesh_by_bone(
                rm.vertices, rm.faces, rm.skin_joints, rm.skin_weights,
                total_budget=self._spin_num_ellipsoids.value(),
                total_max=self._spin_max_ellipsoids.value(),
            )
        except Exception as e:
            self._status.showMessage(f"Mesh Blowup: partition failed: {e}")
            return
        if not parts:
            self._status.showMessage(
                "Mesh Blowup: no bone produced a usable region.")
            return
        self._region_parts = parts
        self._region_centroids = np.array(
            [p.vertices.mean(axis=0) for p in parts], dtype=np.float32)
        self._region_global_center = self._region_centroids.mean(axis=0)
        self._region_colors = self._region_palette(len(parts))
        self._refresh_region_preview()
        self._status.showMessage(
            f"Mesh Blowup preview: {len(parts)} bone region(s).")

    def _on_region_blowup_changed(self, factor: float) -> None:
        """Live-update the explosion distance (translation only)."""
        if self._region_parts is not None:
            self._refresh_region_preview()

    def _refresh_region_preview(self) -> None:
        """Rebuild the exploded, per-bone-coloured combined mesh and show it."""
        parts = self._region_parts
        if not parts:
            return
        factor = self._mesh_settings.region_blowup()
        all_v: list[np.ndarray] = []
        all_f: list[np.ndarray] = []
        all_c: list[np.ndarray] = []
        voff = 0
        for i, p in enumerate(parts):
            disp = (self._region_centroids[i]
                    - self._region_global_center) * factor
            v = (p.vertices + disp).astype(np.float32)
            all_v.append(v)
            all_f.append(p.faces.astype(np.int64) + voff)
            voff += len(v)
            col = self._region_colors[i % len(self._region_colors)]
            all_c.append(np.tile(col, (len(v), 1)))
        verts = np.vstack(all_v)
        faces = np.vstack(all_f)
        colors = np.vstack(all_c).astype(np.float32)
        # Keep the preview aligned with the rest of the scene (skeleton etc.),
        # which is drawn under the current global mesh rotation.
        verts = self._apply_rotation(verts)
        self._viewer.show_region_preview(verts, faces, colors)

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
        densify_start_frac: float = 0.0,
        densify_until_frac: float = 0.75,
        soft_union: bool = False,
        max_ellipsoids: int = 60,
        local_fit: bool = True,
        local_fit_start_frac: float = 0.25,
        local_fit_end_frac: float = 1.0,
        local_fit_every: int = 150,
        symmetry: bool = False,
        merge_enabled: bool = True,
        spawn_enabled: bool = True,
        split_enabled: bool = True,
        prune_enabled: bool = True,
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
            densify_start_frac=densify_start_frac,
            densify_until_frac=densify_until_frac,
            soft_union=soft_union,
            max_ellipsoids=max_ellipsoids,
            thickness_np=r.thickness,
            sdf_computer=self._sdf,
            local_fit=local_fit,
            local_fit_start_frac=local_fit_start_frac,
            local_fit_end_frac=local_fit_end_frac,
            local_fit_every=local_fit_every,
            symmetry_enabled=symmetry,
            merge_enabled=merge_enabled,
            spawn_underrep=spawn_enabled,
            split_enabled=split_enabled,
            prune_enabled=prune_enabled,
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
        self._opt_worker.set_live_metric(
            getattr(self, "_active_ellipsoid_metric", "default"))
        self._opt_phase = "global"
        self._pending_visual = None
        self._visual_timer.start()         # decouple GUI refresh from step rate
        self._opt_worker.step_visual.connect(self._on_opt_step_visual)
        self._opt_worker.step_sdf.connect(self._on_opt_step_sdf)
        self._opt_worker.phase_changed.connect(self._on_opt_phase_changed)
        self._opt_worker.local_progress.connect(self._on_opt_local_progress)
        self._opt_worker.region_changed.connect(self._on_opt_region_changed)
        self._opt_worker.prep_progress.connect(self._on_opt_prep_progress)
        self._opt_worker.op_events.connect(self._on_opt_op_events)
        self._opt_worker.analysis_regions.connect(self._on_opt_analysis_regions)
        self._opt_worker.ellipsoid_metrics.connect(self._on_opt_ellipsoid_metrics)
        self._opt_worker.finished.connect(self._on_opt_finished)
        self._viewer.clear_op_gizmos()      # drop markers from a previous run
        self._viewer.clear_analysis_regions()
        self._viewer.clear_ellipsoid_metrics()
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
            self._visual_timer.stop()
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
        # Hot path — runs once per emitted step (can be ~40×/s at 800 steps/s).
        # Keep it cheap: only record state needed for correctness if the run ends
        # before the next timer tick, then stash the frame.  All heavy GUI work
        # happens in ``_flush_visual`` on the render timer.  (Bone Separation no
        # longer routes through here — it uses ``_on_batched_step``.)
        if self._api_job_id is not None and self._api_stage == "fit":
            self._api_last = (centers, radii, rotations)
        self._pending_visual = (int(step), float(loss),
                                centers, radii, rotations, eps)

    def _flush_visual(self) -> None:
        """Render the latest stashed optimizer frame (driven by a GUI timer).

        Decouples the GUI refresh rate from the optimizer's step rate so a fast
        fit can't flood the event queue.  Also called once on finish to guarantee
        the final frame is shown.
        """
        pv = self._pending_visual
        if pv is None:
            return
        self._pending_visual = None
        step, loss, centers, radii, rotations, eps = pv

        # Route rendering through the active shape plugin (superquadrics use the
        # per-primitive eps array for the deformed mesh).
        self._shape.render(self._viewer, centers, radii, rotations, eps)
        self._viewer.tick_op_gizmos(step)
        self._lbl_ell_count.setText(f"Count: {len(centers)}")

        # Keep ellipsoids reference for rig assignment.
        self._ellipsoids = EllipsoidSet(device=self._device)
        self._ellipsoids.set_parameters(centers, radii, rotations)

        # Bone Separation (batched): the worker emits already-combined world
        # ellipsoids for ALL bones, and the controller drives both progress bars
        # + the status line.  So here we only render — skip the optimizer-specific
        # progress/status/API/dashboard bookkeeping below.
        if self._batched_worker is not None:
            if (self._bonesep_is_api and self._api_server is not None
                    and self._api_job_id is not None):
                self._api_server.registry.update(
                    self._api_job_id, step=step,
                    total=int(getattr(self, "_batched_total_steps", 0)),
                    loss=loss, count=int(len(centers)))
            return

        # Sequential Bone Separation: the per-bone OptimizationWorker drives the
        # render above; the controller owns the progress bars + status line, so
        # skip the global-fit progress/dashboard bookkeeping below (the dashboard
        # is never begin()-ed for bone separation → would divide by zero).  Guard
        # on the controller being live so the gaps BETWEEN bones (when no region
        # worker is active) are covered too, not just ``_region_fit_active``.
        if self._region_fit_active or self._bonesep_ctl is not None:
            if (self._bonesep_is_api and self._api_server is not None
                    and self._api_job_id is not None):
                self._api_server.registry.update(
                    self._api_job_id, step=step, loss=loss,
                    count=int(len(centers)))
            return

        # Training progress in the status-bar progress bar (step / total).
        phase = getattr(self, "_opt_phase", "global")
        total = getattr(self, "_opt_total_steps", 0)
        if total > 0:
            self._progress_set(min(1.0, step / total) * 100,
                               f"Optimizing [{phase.upper()}] · step {step}/{total}")

        if phase != "local":
            self._status.showMessage(
                f"Optimizing [{phase.upper()}] … step {step}/{total}  loss={loss:.6f}")

        # Mirror live progress into the active API job (Debug-mode polling).
        if (self._api_job_id is not None and self._api_stage == "fit"
                and self._api_server is not None):
            self._api_server.registry.update(
                self._api_job_id, step=step, total=total,
                loss=loss, count=int(len(centers)))

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

    def _on_opt_ellipsoid_metrics(self, step: int, metrics) -> None:
        """Color source for the viewport's per-ellipsoid metric heatmaps."""
        self._viewer.set_ellipsoid_metrics(metrics)

    def _on_view_ellipsoid_metric_changed(self, metric: str) -> None:
        """Tell the active optimizer which heatmap metric needs fresh values."""
        self._active_ellipsoid_metric = metric or "default"
        for worker in (self._opt_worker, self._region_fit_worker):
            if worker is not None:
                try:
                    worker.set_live_metric(self._active_ellipsoid_metric)
                except Exception:
                    pass
        # Switching to Default should immediately restore the normal material.
        if self._active_ellipsoid_metric == "default":
            self._viewer.clear_ellipsoid_metrics()

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
        # Stop the GUI refresh timer and flush the last pending frame so the
        # finished result is always shown at full fidelity (the timer may have
        # skipped the final step).
        self._visual_timer.stop()
        self._flush_visual()
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

        # (Bone Separation no longer runs through the OptimizationWorker, so there
        # is nothing to bank here — it finishes via ``_on_batched_finished``.)

        # If this fit was driven over the API, assemble and publish the result.
        if self._api_job_id is not None and self._api_stage == "fit":
            self._api_finish_fit()

    # ── Unity / HTTP API ───────────────────────────────────────────────────

    def start_api_server(self, host: str = "127.0.0.1",
                         port: int = 8765) -> None:
        """Start the embedded HTTP API and route fit requests to this window.

        Fit requests arrive on a background HTTP thread and are delivered here
        via a queued signal, so the orchestration below runs on the GUI thread
        and the fit renders in the normal viewport.
        """
        if self._api_server is not None:
            return
        self._api_server = ApiServer(host=host, port=port)
        self._api_server.bridge.fit_requested.connect(
            self._api_on_fit_requested, QtCore.Qt.QueuedConnection)
        self._api_server.start()

    def _api_reset(self) -> None:
        """Clear all per-job API state so the next request starts clean."""
        self._api_job_id = None
        self._api_stage = None
        self._api_last = None
        self._api_verts = None
        self._api_rig = None

    def _api_fail(self, job_id: str, msg: str) -> None:
        if self._api_server is not None:
            self._api_server.registry.update(job_id, state="error", error=msg)
        self._status.showMessage(f"API job failed: {msg}")
        if self._bonesep_ctl is not None:
            self._bonesep_ctl.cancel()
        self._bonesep_ctl = None
        self._batched_worker = None
        self._api_reset()

    def _api_on_fit_requested(self, job_id: str) -> None:
        """GUI-thread slot: ingest the posted mesh and kick off SDF compute.

        The fit itself is started in ``_on_sdf_done`` once the grid is ready;
        the result is assembled in ``_on_opt_finished``.
        """
        if self._api_server is None:
            return
        job = self._api_server.registry.get(job_id)
        if job is None:
            return
        if self._api_job_id is not None:
            self._api_server.registry.update(
                job_id, state="error", error="another fit is in progress")
            return
        try:
            payload = job.payload
            verts = np.asarray(payload["vertices"], dtype=np.float32).reshape(-1, 3)
            faces = np.asarray(payload["faces"], dtype=np.int64).reshape(-1, 3)
            mesh, transform = load_and_prepare_arrays(verts, faces, target_scale=1.0)
            nverts = mesh.vertices.view(np.ndarray)
            nfaces = mesh.faces.view(np.ndarray)
        except Exception as e:                       # malformed payload
            self._api_fail(job_id, f"bad mesh payload: {e}")
            return

        self._api_job_id = job_id
        self._api_norm = transform
        self._api_options = dict(payload.get("options") or {})
        self._api_rig = payload.get("rig")
        self._api_verts = verts
        self._api_last = None
        self._bone_centers = None
        self._current_mesh_name = "unity-api"

        # ``_set_base_mesh`` resets per-mesh controls (incl. SDF blowup) the way
        # a fresh file load does.  Over the Unity bridge the user dials the
        # blowup in once and expects it to persist across fits, so carry the
        # current value across the reset.
        blowup_vox = self._mesh_settings.blowup_voxels()
        self._set_base_mesh(nverts, nfaces)
        # API jobs fit flat in the Python UI (the rig is only overlaid), so the
        # rigged-mesh-only Mesh-Blowup preview stays disabled here.
        self._mesh_settings.set_region_available(False)
        if blowup_vox != 0.0:
            self._mesh_settings.set_blowup_voxels(blowup_vox)
            self._viewer.set_sdf_blowup(blowup_vox)
        self._viewer.show_mesh(nverts, nfaces)
        self._ensure_sdf_idle()
        self._sdf.set_mesh(nverts, nfaces)

        # If Unity supplied rig + skinning, expose it through the normal Rig Mode
        # panel so saved pose-library clips deform the live Unity mesh itself.
        # Without skinning data we fall back to a skeleton-only overlay.
        self._activate_unity_rig_panel(nverts, nfaces, self._api_rig, transform)

        # Make the Unity-pushed mesh appear (and become selected) in the mesh
        # selector, so the user can switch away and come back to it like a file.
        self._register_unity_mesh_in_combo(nverts, nfaces, self._api_rig, transform)

        self._api_server.registry.update(job_id, state="running")
        self._api_stage = "sdf"
        self._status.showMessage(
            f"API fit {job_id[:8]} … computing SDF "
            f"(verts={len(nverts)} faces={len(nfaces)})")
        self._on_compute_all()

    def _api_show_rig_bones(self, rig: dict | None, transform) -> None:
        """Overlay the posted rig's bind-pose skeleton on the displayed mesh.

        The rig payload's bone positions are in the posted mesh's *original*
        space; the viewport shows the *normalized* mesh, so the joints are mapped
        through the same forward normalization (``(p - center) * scale``) before
        being drawn.  No rig → drop any previous skeleton.
        """
        if not (rig and rig.get("bones")):
            self._viewer.remove_skeleton()
            return
        try:
            skeleton = build_skeleton_from_bones(rig["bones"])
            positions, _ = skeleton.compute_bone_positions_rotations(None)
            positions = (np.asarray(positions, dtype=np.float64) - transform.center) \
                * transform.scale
            positions = self._apply_rotation(positions.astype(np.float32))
            # The skeleton itself is flat (each bone a root, by design — see
            # rig_ingest); use the payload's parent indices only to draw the
            # connecting bone segments.
            parent_indices = np.array(
                [int(b.get("parent", -1)) for b in rig["bones"]], dtype=np.int32)
            self._viewer.show_bones(positions, parent_indices)
        except Exception as e:                       # malformed rig → skip overlay
            self._viewer.remove_skeleton()
            self._status.showMessage(f"API rig overlay skipped: {e}")

    def _api_build_rigged_mesh(self, verts, faces, rig: dict | None, transform):
        """Build a RiggedMesh from the live Unity payload in normalized space.

        Unity posts bind-pose vertices in its original mesh space and
        ``load_and_prepare_arrays`` normalizes them for the viewer/optimizer.
        The rig panel can only apply saved poses correctly if its skeleton and
        rest vertices live in that same normalized coordinate frame.  Convert
        each posted bind-world bone transform into normalized world space, then
        derive the local bind transform from the posted parent hierarchy.
        """
        if not (rig and rig.get("bones")):
            return None
        try:
            joints = np.asarray(rig["boneIndices"]).reshape(len(verts), -1).astype(np.int32)
            weights = np.asarray(rig["boneWeights"], dtype=np.float32).reshape(len(verts), -1)
        except Exception as e:
            raise ValueError(f"bad Unity skinning arrays: {e}") from e
        if joints.shape != weights.shape:
            raise ValueError(
                f"boneIndices {joints.shape} and boneWeights {weights.shape} differ")

        src_bones = list(rig.get("bones") or [])
        if not src_bones:
            return None
        if int(joints.max(initial=0)) >= len(src_bones):
            raise ValueError("boneIndices reference a missing bone")

        world_norm: list[np.ndarray] = []
        parents: list[int] = []
        for i, b in enumerate(src_bones):
            parent = int(b.get("parent", -1))
            if parent < 0 or parent >= len(src_bones):
                parent = -1
            parents.append(parent)

            pos = np.asarray(b["position"], dtype=np.float64).reshape(3)
            rot = np.asarray(b["rotation"], dtype=np.float64).reshape(4)
            pos_norm = (pos - transform.center) * float(transform.scale)
            world_norm.append(mat4_compose(
                pos_norm, rot, np.ones(3, dtype=np.float64)))

        bones: list[Bone] = []
        for i, b in enumerate(src_bones):
            parent = parents[i]
            world = world_norm[i]
            if parent >= 0:
                local = np.linalg.inv(world_norm[parent]) @ world
            else:
                local = world
            bones.append(Bone(
                name=str(b.get("name", f"Bone_{i}")),
                index=i,
                parent_index=parent,
                local_bind_transform=local.astype(np.float64),
                inverse_bind_matrix=np.linalg.inv(world).astype(np.float64),
            ))

        return RiggedMesh(
            vertices=np.asarray(verts, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
            skeleton=Skeleton(bones),
            skin_weights=weights,
            skin_joints=joints,
            poses=[Pose.t_pose()],
            mesh_name="Unity (live)",
        )

    def _activate_unity_rig_panel(self, verts, faces, rig, transform) -> bool:
        """Expose Unity live rig to the rig panel so saved poses skin this mesh."""
        try:
            rigged = self._api_build_rigged_mesh(verts, faces, rig, transform)
        except Exception as e:
            self._status.showMessage(f"Unity rig panel skipped: {e}")
            rigged = None
        if rigged is None:
            self._rig_panel.setChecked(False)
            self._rig_panel.setVisible(False)
            self._grp_rig.setVisible(False)
            self._mesh_settings.set_region_available(False)
            self._bone_centers = None
            self._api_show_rig_bones(rig, transform)
            return False

        self._precomputed_meshes = {}
        self._pending_pose_mesh = None
        self._pose_sdf_timer.stop()
        self._rig_panel.setVisible(True)
        self._rig_panel.setChecked(True)
        self._rig_panel.blockSignals(True)
        self._rig_panel.set_rigged_mesh(rigged)
        self._rig_panel.blockSignals(False)
        self._grp_rig.setVisible(True)
        self._mesh_settings.set_region_available(True)
        self._bone_centers = self._compute_bone_centers(rigged)
        self._show_skeleton_for_pose(rigged, None)
        return True

    def _register_unity_mesh_in_combo(self, verts, faces, rig, transform) -> None:
        """Add (and select) the in-memory Unity mesh in the mesh selector.

        The mesh isn't a file, so it gets a fixed data key and a loader that
        re-displays the cached geometry/rig when the user picks it again.  Only
        the most recent Unity push is kept under this single entry.
        """
        key = "::unity-live::"
        label = "Unity (live)"
        self._unity_mesh_cache = {
            "verts": np.asarray(verts), "faces": np.asarray(faces),
            "rig": rig, "transform": transform,
        }
        self._special_mesh_loaders[key] = self._redisplay_unity_mesh
        self._special_mesh_labels[key] = label
        idx = self._mesh_combo.findData(key)
        self._mesh_combo.blockSignals(True)
        if idx < 0:
            self._mesh_combo.addItem(label, key)
            idx = self._mesh_combo.findData(key)
        self._mesh_combo.setCurrentIndex(idx if idx >= 1 else 0)
        self._mesh_combo.blockSignals(False)

    def _redisplay_unity_mesh(self) -> None:
        """Re-show the cached Unity mesh (no re-fit) when re-selected."""
        c = self._unity_mesh_cache
        if not c:
            return
        verts, faces = c["verts"], c["faces"]
        self._set_base_mesh(verts, faces)
        self._viewer.show_mesh(verts, faces)
        self._ensure_sdf_idle()
        self._sdf.set_mesh(verts, faces)
        self._current_mesh_name = "unity-api"
        self._activate_unity_rig_panel(verts, faces, c.get("rig"), c.get("transform"))
        self._on_compute_all()

    def _api_start_fit(self) -> None:
        """Begin the optimization for the active API job (post SDF compute).

        The fit uses whatever settings are *currently selected* in the EllipSDF
        window — exactly the same controls the manual "Fit" button reads — so a
        user can dial in the options live and the Unity request honours them.
        The options carried in the request payload are ignored on purpose.
        """
        # Bone Separation over the API: train each bone of the posted rig on its
        # own region SDF, then assemble bone-local entries.  Driven directly here
        # (not via _on_fit_clicked, which keys off the rig *panel* that an API
        # job never activates).  The "bonesep" stage keeps the plain "fit"/"sdf"
        # API hooks dormant so only the bone-separation controller runs.
        rig = self._api_rig
        if (self._cmb_fit_scope.currentData() == "bone"
                and rig and rig.get("bones")
                and rig.get("boneIndices") is not None
                and rig.get("boneWeights") is not None
                and self._base_verts is not None
                and self._base_faces is not None):
            self._api_stage = "bonesep"
            try:
                verts = np.asarray(self._base_verts, dtype=np.float32).reshape(-1, 3)
                faces = np.asarray(self._base_faces).reshape(-1, 3)
                joints = np.asarray(rig["boneIndices"]).reshape(
                    len(verts), -1).astype(np.int64)
                weights = np.asarray(rig["boneWeights"], dtype=np.float64).reshape(
                    len(verts), -1)
            except Exception as e:
                self._api_fail(self._api_job_id, f"bad rig skin data: {e}")
                return
            self._begin_bone_separation(
                verts, faces, joints, weights,
                on_complete=self._api_bonesep_complete,
                base_kwargs=self._gather_fit_kwargs(),
                is_api=True,
            )
            return

        self._api_stage = "fit"
        self._on_fit_clicked()

    def _api_finish_fit(self) -> None:
        """Denormalize the fitted ellipsoids and publish the job result."""
        job_id = self._api_job_id
        if job_id is None or self._api_server is None:
            return
        try:
            if self._api_last is None:
                raise RuntimeError("no ellipsoids were produced")
            centers, radii, rotations = self._api_last
            t = self._api_norm
            # Map fitted ellipsoids back to the posted mesh's original space.
            centers_o = np.array([t.to_original_point(c) for c in centers],
                                 dtype=np.float64)
            radii_o = np.array([t.to_original_length(r) for r in radii],
                               dtype=np.float64)

            rig = self._api_rig
            rigged = bool(rig and rig.get("bones"))
            if rigged:
                # Bone-local entries (Sphere_<i> parented under its bone).
                entries = assign_ellipsoids_to_bones(
                    centers_o, radii_o, rotations, self._api_verts, rig)
            else:
                # Flat entries in original mesh space.
                entries = [{
                    "name": f"Sphere_{i}",
                    "bone": None,
                    "center": [round(float(v), 7) for v in centers_o[i]],
                    "radii": [round(float(v), 7) for v in radii_o[i]],
                    "rotation": [round(float(v), 7) for v in rotations[i]],
                } for i in range(len(centers_o))]

            result = {
                "version": 2,
                "coordinate_system": "right_hand_y_up",
                "quaternion_convention": "xyzw",
                "rigged": rigged,
                "count": len(entries),
                "ellipsoids": entries,
            }
            self._api_server.registry.update(
                job_id, state="done", result=result, count=len(entries))
            self._status.showMessage(
                f"API fit {job_id[:8]} done — {len(entries)} ellipsoids"
                f" ({'rigged' if rigged else 'flat'})")
        except Exception as e:
            self._api_server.registry.update(
                job_id, state="error", error=str(e))
            self._status.showMessage(f"API fit {job_id[:8]} failed: {e}")
        finally:
            self._api_reset()
