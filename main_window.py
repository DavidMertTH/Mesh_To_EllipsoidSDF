"""
main_window.py — Application main window with 2×2 layout.

  ┌─────────────────────┬─────────────────────┐
  │  Mesh 3-D Viewer    │  Mesh SDF Slice     │
  │  (top-left)         │  (top-right)        │
  ├─────────────────────┼─────────────────────┤
  │  Ellipsoid 3-D      │  Ellipsoid SDF      │
  │  Viewer (bot-left)  │  Slice (bot-right)  │
  └─────────────────────┴─────────────────────┘

Top row:    loaded mesh → SDF from Warp mesh queries
Bottom row: ellipsoid set → analytical SDF (Ínigo Quílez approx.)
Both SDF panels share the same AABB / grid so slices are directly comparable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import warp as wp

from mesh_io import load_and_prepare
from sdf_compute import SdfComputer, SdfResult
from ellipsoid import EllipsoidSet
from viewer3d import MeshViewer3D, EllipsoidViewer3D
from widgets import SdfSlicePanel
from optimization import OptimizationWorker

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
        self._device = "cuda" if wp.is_cuda_available() else "cpu"

        self._sdf = SdfComputer(device=self._device)
        self._ellipsoids: EllipsoidSet | None = None

        self._last_mesh_result: SdfResult | None = None
        self._mesh_viewer = MeshViewer3D()
        self._mesh_sdf_panel = SdfSlicePanel()

        self._ell_viewer = EllipsoidViewer3D()
        self._ell_sdf_panel = SdfSlicePanel()
        self._status = self.statusBar()

        self._build_layout()
        self._build_toolbar()
        self._connect_signals()


        self._opt_worker: OptimizationWorker | None = None

    def _build_layout(self):
        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # ── Mesh selector bar ─────────────────────────────────────────────
        selector_bar = QtWidgets.QHBoxLayout()
        selector_bar.setSpacing(6)

        selector_bar.addWidget(QtWidgets.QLabel("Mesh:"))

        self._mesh_combo = QtWidgets.QComboBox()
        self._mesh_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed,
        )
        selector_bar.addWidget(self._mesh_combo)

        self._btn_refresh = QtWidgets.QPushButton("↻")
        self._btn_refresh.setFixedWidth(32)
        self._btn_refresh.setToolTip("Rescan meshes/ folder")
        selector_bar.addWidget(self._btn_refresh)

        self._btn_open_dir = QtWidgets.QPushButton("📂 Open folder")
        self._btn_open_dir.setToolTip(f"Open {self._mesh_dir}")
        selector_bar.addWidget(self._btn_open_dir)

        # ── Training controls ──────────────────────────────────────────────
        selector_bar.addSpacing(16)
        selector_bar.addWidget(QtWidgets.QLabel("Ellipsoids:"))

        self._spin_num_ellipsoids = QtWidgets.QSpinBox()
        self._spin_num_ellipsoids.setRange(1, 200)
        self._spin_num_ellipsoids.setValue(10)
        self._spin_num_ellipsoids.setToolTip("Number of ellipsoids to fit")
        selector_bar.addWidget(self._spin_num_ellipsoids)

        self._btn_fit = QtWidgets.QPushButton("▶ Fit Ellipsoids")
        self._btn_fit.setToolTip("Start fitting ellipsoids to the loaded mesh SDF")
        self._btn_fit.setEnabled(False)
        selector_bar.addWidget(self._btn_fit)

        self._btn_stop = QtWidgets.QPushButton("■ Stop")
        self._btn_stop.setToolTip("Stop the running optimisation")
        self._btn_stop.setEnabled(False)
        selector_bar.addWidget(self._btn_stop)

        root_layout.addLayout(selector_bar)

        # ── 2×2 splitter grid ────────────────────────────────────────────
        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top_splitter.addWidget(self._mesh_viewer.widget)
        top_splitter.addWidget(self._mesh_sdf_panel)
        top_splitter.setSizes([650, 650])

        bot_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bot_splitter.addWidget(self._ell_viewer.widget)
        bot_splitter.addWidget(self._ell_sdf_panel)
        bot_splitter.setSizes([650, 650])

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bot_splitter)
        main_splitter.setSizes([450, 450])

        root_layout.addWidget(main_splitter, 1)

        self.setCentralWidget(central)

        # Populate combo on startup
        self._scan_mesh_dir()

    def _build_toolbar(self):
        act_compute = QtGui.QAction("Compute SDF Grid (G)", self)
        act_compute.setShortcut(QtGui.QKeySequence("G"))
        act_compute.triggered.connect(self._on_compute_all)

        tb = self.addToolBar("Tools")
        tb.addAction(act_compute)
        self.addAction(act_compute)

    def _connect_signals(self):
        # Drag-and-drop on mesh viewer
        self._mesh_viewer.widget.fileDropped.connect(self._on_file_dropped)

        # Compute buttons on both panels
        self._mesh_sdf_panel.computeRequested.connect(self._on_compute_all)
        self._ell_sdf_panel.computeRequested.connect(self._on_compute_all)

        # Mesh selector
        self._mesh_combo.activated.connect(self._on_combo_selected)
        self._btn_refresh.clicked.connect(self._scan_mesh_dir)
        self._btn_open_dir.clicked.connect(self._open_mesh_dir)

        # Training
        self._btn_fit.clicked.connect(self._on_fit_clicked)
        self._btn_stop.clicked.connect(self._on_stop_clicked)

    # ── mesh directory scanning ───────────────────────────────────────────

    def _scan_mesh_dir(self):
        """Scan the meshes/ folder and populate the combo box."""
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

        # Try to restore previous selection
        idx = self._mesh_combo.findText(prev_text)
        if idx >= 1:
            self._mesh_combo.setCurrentIndex(idx)

        self._mesh_combo.blockSignals(False)

        count = self._mesh_combo.count() - 1  # minus placeholder
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
        """Open the meshes/ folder in the OS file explorer."""
        path = str(self._mesh_dir.resolve())
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_file_dropped(self, path: str):
        self._load_mesh(path)

    def _load_mesh(self, path: str):
        try:
            mesh = load_and_prepare(path, target_scale=1.0)

            verts: np.ndarray = mesh.vertices.view(np.ndarray)
            faces: np.ndarray = mesh.faces.view(np.ndarray)

            self._mesh_viewer.show_mesh(verts, faces)
            self._sdf.set_mesh(verts, faces)

            # Sync combo box with loaded file
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

            # Auto-compute both SDFs
            self._on_compute_all()

        except Exception as e:
            self._status.showMessage(f"Failed to load: {path} ({e})")

    def _on_compute_all(self, n: int | None = None):
        """Compute mesh SDF grid."""
        if not self._sdf.is_ready:
            self._status.showMessage("Load a mesh first.")
            return

        if n is None:
            n = self._mesh_sdf_panel.requested_n

        self._status.showMessage(f"Computing mesh SDF (n={n}) on {self._device} …")
        try:
            mesh_result = self._sdf.compute_voxel_grid(n=n)
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

        self._ell_viewer.show_ellipsoids(self._ellipsoids)

        if use_last_mesh_grid and self._last_mesh_result is not None:
            r = self._last_mesh_result
            ell_grid = self._ellipsoids.compute_sdf_grid(
                origin=r.origin,
                dx=r.dx,
                n=r.n,
            )
            self._ell_sdf_panel.set_sdf(ell_grid)
        elif origin is not None and dx is not None and n is not None:
            ell_grid = self._ellipsoids.compute_sdf_grid(
                origin=origin,
                dx=dx,
                n=n,
            )
            self._ell_sdf_panel.set_sdf(ell_grid)

    # ── fit / stop button handlers ──────────────────────────────────────

    def _on_fit_clicked(self):
        """Start fitting ellipsoids to the computed mesh SDF."""
        if self._last_mesh_result is None:
            self._status.showMessage("Compute mesh SDF first (press G or Compute).")
            return
        num_e = self._spin_num_ellipsoids.value()
        self.start_optimization(
            num_ellipsoids=num_e,
            method="adam",
            num_steps=7000,
            report_every=20,
        )

    def _on_stop_clicked(self):
        self.stop_optimization()

    # ── async optimization ────────────────────────────────────────────

    def start_optimization(
        self,
        num_ellipsoids: int = 10,
        method: str = "adam",
        num_steps: int = 2000,
        report_every: int = 20,
    ) -> None:
        """Launch the optimisation loop in a background thread."""
        if self._last_mesh_result is None:
            self._status.showMessage("No mesh SDF available. Load a mesh and compute SDF first.")
            return

        self.stop_optimization()  # stop any running worker first

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
            parent=self,
        )
        self._opt_worker.step_done.connect(self._on_opt_step)
        self._opt_worker.finished.connect(self._on_opt_finished)
        self._opt_worker.start()

        self._btn_fit.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status.showMessage(
            f"Optimization started ({method}, {num_ellipsoids} ellipsoids) …"
        )

    def stop_optimization(self) -> None:
        """Gracefully stop a running optimisation worker."""
        if self._opt_worker is not None and self._opt_worker.isRunning():
            self._opt_worker.request_stop()
            self._opt_worker.wait()
            self._opt_worker = None
            self._btn_fit.setEnabled(self._last_mesh_result is not None)
            self._btn_stop.setEnabled(False)

    def _on_opt_step(
            self,
            step: int,
            loss: float,
            ell_set: EllipsoidSet,
            use_last_mesh_grid: bool,
            origin: np.ndarray,
            dx: float,
            n: int,
    ) -> None:
        """Slot called on the main thread whenever the worker reports progress."""
        print(f"Step {step}: loss = {loss:.6f}")
        if ell_set is not None:
            self.update_ellipsoids(ell_set, use_last_mesh_grid, origin, dx, n)
        self._status.showMessage(f"Optimizing … step {step}  loss={loss:.6f}")

    def _on_opt_finished(self) -> None:
        self._status.showMessage("Optimization finished.")
        self._opt_worker = None
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._btn_stop.setEnabled(False)
