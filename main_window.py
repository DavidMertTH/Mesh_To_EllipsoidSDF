"""
main_window.py — Application main window with 2×2 layout + evaluation panel.

  ┌─────────────────────┬─────────────────────┬──────────────────┐
  │  Mesh 3-D Viewer    │  Mesh SDF Slice     │                  │
  │  (top-left)         │  (top-right)        │   Auswertung     │
  ├─────────────────────┼─────────────────────┤   (Loss-Kurve,   │
  │  Ellipsoid 3-D      │  Ellipsoid SDF      │    Run-          │
  │  Viewer (bot-left)  │  Slice (bot-right)  │    Verwaltung)   │
  └─────────────────────┴─────────────────────┴──────────────────┘

Top row:    loaded mesh → SDF from Warp mesh queries
Bottom row: ellipsoid set → analytical SDF (Ínigo Quílez approx.)
Right:      live loss plot, run selection, naming & persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import warp as wp

from mesh_io import load_and_prepare
from sdf_compute import SdfComputer, SdfResult
from ellipsoid import EllipsoidSet, SDF_QUILEZ, SDF_METHOD_NAMES, best_device
from viewer3d import MeshViewer3D, EllipsoidViewer3D
from widgets import SdfSlicePanel
from optimization import OptimizationWorker
from run_tracker import RunTrackerPanel

# Supported mesh file extensions (trimesh)
MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".dae"}

# Default mesh directory relative to this file
DEFAULT_MESH_DIR = Path(__file__).parent / "meshes"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mesh_dir: Path | str | None = None):
        super().__init__()
        self.setWindowTitle("Mesh → Ellipsoid SDF Approximation")

        self._mesh_dir = Path(mesh_dir) if mesh_dir else DEFAULT_MESH_DIR
        self._mesh_dir.mkdir(parents=True, exist_ok=True)

        wp.init()
        self._device = best_device()

        self._sdf = SdfComputer(device=self._device)
        self._ellipsoids: EllipsoidSet | None = None

        self._last_mesh_result: SdfResult | None = None
        self._mesh_viewer = MeshViewer3D()
        self._mesh_sdf_panel = SdfSlicePanel()

        self._ell_viewer = EllipsoidViewer3D()
        self._ell_sdf_panel = SdfSlicePanel()

        self._run_tracker = RunTrackerPanel()

        self._status = self.statusBar()

        self._build_layout()
        self._build_toolbar()
        self._connect_signals()

        self._opt_worker: OptimizationWorker | None = None
        self._current_mesh_name: str = ""
        self._current_sdf_mode: int = SDF_QUILEZ

    def _build_layout(self):
        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # ══════════════════════════════════════════════════════════════
        # LEFT: Settings panel (vertical column)
        # ══════════════════════════════════════════════════════════════
        settings_panel = QtWidgets.QWidget()
        settings_panel.setFixedWidth(220)
        settings_layout = QtWidgets.QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(4, 4, 4, 4)
        settings_layout.setSpacing(6)

        # ── Mesh selection ────────────────────────────────────────────
        grp_mesh = QtWidgets.QGroupBox("Mesh")
        mesh_lay = QtWidgets.QVBoxLayout(grp_mesh)
        mesh_lay.setSpacing(4)

        self._mesh_combo = QtWidgets.QComboBox()
        self._mesh_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed,
        )
        mesh_lay.addWidget(self._mesh_combo)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_refresh = QtWidgets.QPushButton("Rescan")
        self._btn_refresh.setToolTip("Rescan meshes/ folder")
        btn_row.addWidget(self._btn_refresh)

        self._btn_open_dir = QtWidgets.QPushButton("Open folder")
        self._btn_open_dir.setToolTip(f"Open {self._mesh_dir}")
        btn_row.addWidget(self._btn_open_dir)
        mesh_lay.addLayout(btn_row)

        settings_layout.addWidget(grp_mesh)

        # ── SDF settings ─────────────────────────────────────────────
        grp_sdf = QtWidgets.QGroupBox("SDF")
        sdf_lay = QtWidgets.QFormLayout(grp_sdf)
        sdf_lay.setSpacing(4)

        self._slider_margin = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_margin.setRange(0, 100)
        self._slider_margin.setValue(50)
        self._slider_margin.setToolTip("Fractional margin around the mesh bounding box (0.0–1.0)")
        self._lbl_margin = QtWidgets.QLabel("0.50")
        self._lbl_margin.setFixedWidth(32)
        self._slider_margin.valueChanged.connect(
            lambda v: self._lbl_margin.setText(f"{v / 100:.2f}")
        )
        margin_row = QtWidgets.QHBoxLayout()
        margin_row.addWidget(self._slider_margin, 1)
        margin_row.addWidget(self._lbl_margin)
        sdf_lay.addRow("Margin:", margin_row)

        self._combo_sdf_method = QtWidgets.QComboBox()
        for mode_id in sorted(SDF_METHOD_NAMES.keys()):
            self._combo_sdf_method.addItem(SDF_METHOD_NAMES[mode_id], mode_id)
        self._combo_sdf_method.setCurrentIndex(0)
        self._combo_sdf_method.setToolTip("SDF approximation method for ellipsoid fitting")
        sdf_lay.addRow("Method:", self._combo_sdf_method)

        settings_layout.addWidget(grp_sdf)

        # ── Training parameters ──────────────────────────────────────
        grp_training = QtWidgets.QGroupBox("Training")
        train_lay = QtWidgets.QFormLayout(grp_training)
        train_lay.setSpacing(4)

        self._spin_num_ellipsoids = QtWidgets.QSpinBox()
        self._spin_num_ellipsoids.setRange(1, 200)
        self._spin_num_ellipsoids.setValue(10)
        self._spin_num_ellipsoids.setToolTip("Number of ellipsoids to fit")
        train_lay.addRow("Ellipsoids:", self._spin_num_ellipsoids)

        self._spin_miss_penalty = QtWidgets.QDoubleSpinBox()
        self._spin_miss_penalty.setRange(0.0, 50.0)
        self._spin_miss_penalty.setValue(3.0)
        self._spin_miss_penalty.setSingleStep(0.5)
        self._spin_miss_penalty.setDecimals(1)
        self._spin_miss_penalty.setToolTip(
            "Extra loss weight for interior regions missed by all ellipsoids.\n"
            "Higher = stronger pressure to cover thin structures (arms, fingers)."
        )
        train_lay.addRow("Miss penalty:", self._spin_miss_penalty)

        self._spin_maintenance = QtWidgets.QSpinBox()
        self._spin_maintenance.setRange(0, 5000)
        self._spin_maintenance.setValue(200)
        self._spin_maintenance.setSingleStep(50)
        self._spin_maintenance.setToolTip(
            "Maintenance interval (prune + spawn) in training steps.\n"
            "0 = disabled. Lower = more frequent population management."
        )
        self._spin_maintenance.setSpecialValueText("off")
        train_lay.addRow("Maintenance:", self._spin_maintenance)

        settings_layout.addWidget(grp_training)

        # ── Fit / Stop buttons ───────────────────────────────────────
        self._btn_fit = QtWidgets.QPushButton("Fit Ellipsoids")
        self._btn_fit.setToolTip("Start fitting ellipsoids to the loaded mesh SDF")
        self._btn_fit.setEnabled(False)
        settings_layout.addWidget(self._btn_fit)

        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.setToolTip("Stop the running optimisation")
        self._btn_stop.setEnabled(False)
        settings_layout.addWidget(self._btn_stop)

        settings_layout.addStretch(1)

        # ══════════════════════════════════════════════════════════════
        # CENTER: 2x2 viewer grid
        # ══════════════════════════════════════════════════════════════
        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top_splitter.addWidget(self._mesh_viewer.widget)
        top_splitter.addWidget(self._mesh_sdf_panel)
        top_splitter.setSizes([650, 650])

        bot_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bot_splitter.addWidget(self._ell_viewer.widget)
        bot_splitter.addWidget(self._ell_sdf_panel)
        bot_splitter.setSizes([650, 650])

        grid_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        grid_splitter.addWidget(top_splitter)
        grid_splitter.addWidget(bot_splitter)
        grid_splitter.setSizes([450, 450])

        # ══════════════════════════════════════════════════════════════
        # RIGHT: Evaluation panel (run tracker)
        # ══════════════════════════════════════════════════════════════
        center_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        center_right.addWidget(grid_splitter)
        center_right.addWidget(self._run_tracker)
        center_right.setSizes([900, 400])
        center_right.setCollapsible(1, False)

        root_layout.addWidget(settings_panel)
        root_layout.addWidget(center_right, 1)

        self.setCentralWidget(central)
        self._scan_mesh_dir()

    def _build_toolbar(self):
        act_compute = QtGui.QAction("Compute SDF Grid (G)", self)
        act_compute.setShortcut(QtGui.QKeySequence("G"))
        act_compute.triggered.connect(self._on_compute_all)

        tb = self.addToolBar("Tools")
        tb.addAction(act_compute)
        self.addAction(act_compute)

    def _connect_signals(self):
        self._mesh_viewer.widget.fileDropped.connect(self._on_file_dropped)
        self._mesh_sdf_panel.computeRequested.connect(self._on_compute_all)
        self._ell_sdf_panel.computeRequested.connect(self._on_compute_all)
        self._mesh_combo.activated.connect(self._on_combo_selected)
        self._btn_refresh.clicked.connect(self._scan_mesh_dir)
        self._btn_open_dir.clicked.connect(self._open_mesh_dir)
        self._btn_fit.clicked.connect(self._on_fit_clicked)
        self._btn_stop.clicked.connect(self._on_stop_clicked)

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

        count = self._mesh_combo.count() - 1
        self._status.showMessage(
            f"Found {count} mesh(es) in {self._mesh_dir}. "
            f"Select from dropdown or drag & drop."
        )

    def _on_combo_selected(self, index: int):
        if index < 1:
            return
        path = self._mesh_combo.itemData(index)
        if path:
            self._load_mesh(path)

    def _open_mesh_dir(self):
        path = str(self._mesh_dir.resolve())
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_file_dropped(self, path: str):
        self._load_mesh(path)

    def _load_mesh(self, path: str):
        try:
            mesh = load_and_prepare(path, target_scale=1.0)
            verts = mesh.vertices.view(np.ndarray)
            faces = mesh.faces.view(np.ndarray)

            self._mesh_viewer.show_mesh(verts, faces)
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

    def _on_compute_all(self, n: int | None = None):
        if not self._sdf.is_ready:
            self._status.showMessage("Load a mesh first.")
            return

        if n is None:
            n = self._mesh_sdf_panel.requested_n

        margin = self._slider_margin.value() / 100.0
        self._status.showMessage(f"Computing mesh SDF (n={n}, margin={margin:.2f}) on {self._device} …")
        try:
            mesh_result = self._sdf.compute_voxel_grid(n=n, margin=margin)
        except Exception as e:
            self._status.showMessage(f"Mesh SDF failed: {e}")
            return

        self._last_mesh_result = mesh_result
        self._mesh_sdf_panel.set_sdf(mesh_result.grid)
        self._btn_fit.setEnabled(True)

        self._status.showMessage(
            f"Mesh SDF done — min={float(np.min(mesh_result.grid)):.4f} "
            f"max={float(np.max(mesh_result.grid)):.4f}  |  "
            f"Ready to fit ellipsoids."
        )

    def update_ellipsoids(
            self,
            ellipsoid_set: EllipsoidSet,
            use_last_mesh_grid: bool = True,
            origin: np.ndarray = None,
            dx: float = None,
            n: int = None,
    ) -> None:
        self._ellipsoids = ellipsoid_set
        sdf_mode = self._current_sdf_mode

        self._ell_viewer.show_ellipsoids(self._ellipsoids)

        if use_last_mesh_grid and self._last_mesh_result is not None:
            r = self._last_mesh_result
            ell_grid = self._ellipsoids.compute_sdf_grid(
                origin=r.origin, dx=r.dx, n=r.n,
                sdf_mode=sdf_mode,
            )
            self._ell_sdf_panel.set_sdf(ell_grid)
        elif origin is not None and dx is not None and n is not None:
            ell_grid = self._ellipsoids.compute_sdf_grid(
                origin=origin, dx=dx, n=n,
                sdf_mode=sdf_mode,
            )
            self._ell_sdf_panel.set_sdf(ell_grid)

    # ── fit / stop ────────────────────────────────────────────────────────

    def _on_fit_clicked(self):
        if self._last_mesh_result is None:
            self._status.showMessage("Compute mesh SDF first (press G or Compute).")
            return
        num_e = self._spin_num_ellipsoids.value()
        sdf_mode = self._combo_sdf_method.currentData()
        miss_w = self._spin_miss_penalty.value()
        maint = self._spin_maintenance.value()
        self.start_optimization(
            num_ellipsoids=num_e,
            method="adam",
            num_steps=7000,
            report_every=20,
            sdf_mode=sdf_mode,
            miss_penalty_weight=miss_w,
            maintenance_every=maint,
        )

    def _on_stop_clicked(self):
        self.stop_optimization()

    # ── async optimization ────────────────────────────────────────────────

    def start_optimization(
        self,
        num_ellipsoids: int = 10,
        method: str = "adam",
        num_steps: int = 2000,
        report_every: int = 20,
        sdf_mode: int = SDF_QUILEZ,
        miss_penalty_weight: float = 3.0,
        maintenance_every: int = 200,
    ) -> None:
        if self._last_mesh_result is None:
            self._status.showMessage("No mesh SDF available. Load a mesh and compute SDF first.")
            return

        self.stop_optimization()
        self._current_sdf_mode = sdf_mode

        r = self._last_mesh_result
        self._opt_worker = OptimizationWorker(
            sdf_target_np=r.grid,
            origin=r.origin,
            dx=r.dx,
            n=r.n,
            num_ellipsoids=num_ellipsoids,
            method=method,
            num_steps=num_steps,
            report_every=report_every,
            sdf_mode=sdf_mode,
            miss_penalty_weight=miss_penalty_weight,
            maintenance_every=maintenance_every,
            parent=self,
        )
        self._opt_worker.step_visual.connect(self._on_opt_step_visual)
        self._opt_worker.step_sdf.connect(self._on_opt_step_sdf)
        self._opt_worker.finished.connect(self._on_opt_finished)
        self._opt_worker.start()

        sdf_name = SDF_METHOD_NAMES.get(sdf_mode, "?")
        self._run_tracker.begin_run(
            mesh_name=self._current_mesh_name,
            method=method,
            num_ellipsoids=num_ellipsoids,
            grid_n=r.n,
        )

        self._btn_fit.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status.showMessage(
            f"Optimization started ({method}, {num_ellipsoids} ellipsoids, SDF: {sdf_name}) …"
        )

    def stop_optimization(self) -> None:
        if self._opt_worker is not None and self._opt_worker.isRunning():
            self._opt_worker.request_stop()
            self._opt_worker.wait()
            self._opt_worker = None
            self._run_tracker.finish_run()
            self._btn_fit.setEnabled(self._last_mesh_result is not None)
            self._btn_stop.setEnabled(False)

    def _on_opt_step_visual(
            self,
            step: int,
            loss: float,
            centers: np.ndarray,
            radii: np.ndarray,
            rotations: np.ndarray,
    ) -> None:
        print(f"Step {step}: loss = {loss:.6f}")
        self._ell_viewer.show_ellipsoids_fast(centers, radii, rotations)
        self._status.showMessage(f"Optimizing … step {step}  loss={loss:.6f}")

        # Feed loss to the run tracker
        self._run_tracker.record_step(step, loss)

    def _on_opt_step_sdf(
            self,
            step: int,
            loss: float,
            ell_set: EllipsoidSet,
            use_last_mesh_grid: bool,
            origin: np.ndarray,
            dx: float,
            n: int,
    ) -> None:
        self._ellipsoids = ell_set
        sdf_mode = self._current_sdf_mode
        if use_last_mesh_grid and self._last_mesh_result is not None:
            r = self._last_mesh_result
            ell_grid = self._ellipsoids.compute_sdf_grid(
                origin=r.origin, dx=r.dx, n=r.n,
                sdf_mode=sdf_mode,
            )
            self._ell_sdf_panel.set_sdf(ell_grid)

    def _on_opt_finished(self) -> None:
        self._status.showMessage("Optimization finished.")
        self._run_tracker.finish_run()
        self._opt_worker = None
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._btn_stop.setEnabled(False)