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

Rig mode (optional):
  When a glTF/GLB with skeleton is loaded, the Rig Mode panel activates.
  Poses can be scrubbed, and multi-pose training trains bone-local params.
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
from rig_panel import RigModePanel, try_load_rigged
from skinning import deform_mesh
from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids
from skeleton import Pose

# Supported mesh file extensions (trimesh + glTF for rigged)
MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".dae", ".fbx"}

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

        # ── Rig mode (modular) ──
        self._rig_panel = RigModePanel()

        self._status = self.statusBar()

        self._build_layout()
        self._build_toolbar()
        self._connect_signals()

        self._opt_worker: OptimizationWorker | None = None
        self._multipose_worker = None  # MultiPoseOptimizationWorker
        self._current_mesh_name: str = ""
        self._current_sdf_mode: int = SDF_QUILEZ

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

        # ── SDF margin slider ──────────────────────────────────────────────
        selector_bar.addSpacing(16)
        selector_bar.addWidget(QtWidgets.QLabel("SDF Margin:"))

        self._slider_margin = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_margin.setRange(0, 100)
        self._slider_margin.setValue(50)
        self._slider_margin.setFixedWidth(120)
        self._slider_margin.setToolTip("Fractional margin around the mesh bounding box (0.0–1.0)")
        selector_bar.addWidget(self._slider_margin)

        self._lbl_margin = QtWidgets.QLabel("0.50")
        self._lbl_margin.setFixedWidth(32)
        selector_bar.addWidget(self._lbl_margin)

        self._slider_margin.valueChanged.connect(
            lambda v: self._lbl_margin.setText(f"{v / 100:.2f}")
        )

        # ── Training controls ──────────────────────────────────────────────
        selector_bar.addSpacing(16)
        selector_bar.addWidget(QtWidgets.QLabel("Ellipsoids:"))

        self._spin_num_ellipsoids = QtWidgets.QSpinBox()
        self._spin_num_ellipsoids.setRange(1, 200)
        self._spin_num_ellipsoids.setValue(10)
        self._spin_num_ellipsoids.setToolTip("Number of ellipsoids to fit")
        selector_bar.addWidget(self._spin_num_ellipsoids)

        selector_bar.addWidget(QtWidgets.QLabel("SDF:"))
        self._combo_sdf_method = QtWidgets.QComboBox()
        for mode_id in sorted(SDF_METHOD_NAMES.keys()):
            self._combo_sdf_method.addItem(SDF_METHOD_NAMES[mode_id], mode_id)
        self._combo_sdf_method.setCurrentIndex(0)  # Quílez default
        self._combo_sdf_method.setToolTip("SDF approximation method for ellipsoid fitting")
        selector_bar.addWidget(self._combo_sdf_method)

        selector_bar.addWidget(QtWidgets.QLabel("Steps:"))
        self._spin_max_steps = QtWidgets.QSpinBox()
        self._spin_max_steps.setRange(100, 100000)
        self._spin_max_steps.setValue(7000)
        self._spin_max_steps.setSingleStep(1000)
        self._spin_max_steps.setToolTip("Maximum training steps for single-pose fitting")
        selector_bar.addWidget(self._spin_max_steps)

        self._chk_pruning = QtWidgets.QCheckBox("Pruning")
        self._chk_pruning.setChecked(False)
        self._chk_pruning.setToolTip("Enable containment-based pruning & spawning during fitting")
        selector_bar.addWidget(self._chk_pruning)

        self._btn_fit = QtWidgets.QPushButton("▶ Fit Ellipsoids")
        self._btn_fit.setToolTip("Start fitting ellipsoids to the loaded mesh SDF")
        self._btn_fit.setEnabled(False)
        selector_bar.addWidget(self._btn_fit)

        self._btn_stop = QtWidgets.QPushButton("■ Stop")
        self._btn_stop.setToolTip("Stop the running optimisation")
        self._btn_stop.setEnabled(False)
        selector_bar.addWidget(self._btn_stop)

        root_layout.addLayout(selector_bar)

        # ── 2×2 splitter grid + evaluation panel + rig panel ─────────────
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

        # Right panel: run tracker + rig panel stacked
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self._run_tracker, 1)
        right_layout.addWidget(self._rig_panel, 0)

        main_hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_hsplitter.addWidget(grid_splitter)
        main_hsplitter.addWidget(right_panel)
        main_hsplitter.setSizes([900, 400])
        main_hsplitter.setCollapsible(1, False)

        root_layout.addWidget(main_hsplitter, 1)

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

        # ── Rig mode signals ──
        self._rig_panel.poseChanged.connect(self._on_rig_pose_changed)
        self._rig_panel.multiPoseRequested.connect(self._on_multipose_fit_clicked)
        self._rig_panel._btn_assign.clicked.connect(self._on_rig_assign_clicked)

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

    def _load_rigged_mesh(self, path: str, rigged):
        """Load a rigged mesh and activate rig mode."""
        self._rig_panel.setChecked(True)
        self._rig_panel.set_rigged_mesh(rigged)

        verts = rigged.vertices
        faces = rigged.faces

        self._mesh_viewer.show_mesh(verts, faces)
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
        parent_indices = np.array(
            [b.parent_index for b in rigged.skeleton.bones], dtype=np.int32,
        )
        self._mesh_viewer.show_bones(positions, parent_indices)

    # ── Rig-mode: pose changed ───────────────────────────────────────

    def _on_rig_pose_changed(self, pose_index: int):
        """User moved the pose slider — update mesh and ellipsoids."""
        if not self._rig_panel.is_active:
            return

        rm = self._rig_panel.rigged_mesh
        if rm is None:
            return

        # Deform mesh to new pose
        deformed = self._rig_panel.get_deformed_mesh(pose_index)
        if deformed is not None:
            self._mesh_viewer.show_mesh(deformed, rm.faces)
            self._sdf.set_mesh(deformed, rm.faces)

            # Update skeleton overlay
            pose = rm.poses[pose_index] if pose_index < len(rm.poses) else Pose.t_pose()
            self._show_skeleton_for_pose(rm, pose)

            # Recompute mesh SDF
            margin = self._slider_margin.value() / 100.0
            n = self._mesh_sdf_panel.requested_n
            try:
                mesh_result = self._sdf.compute_voxel_grid(n=n, margin=margin)
                self._last_mesh_result = mesh_result
                self._mesh_sdf_panel.set_sdf(mesh_result.grid)
            except Exception:
                pass

        # Update ellipsoids if bone-local params exist
        world_ell = self._rig_panel.get_world_ellipsoids(pose_index)
        if world_ell is not None:
            wc, wr, wq = world_ell
            self._ell_viewer.show_ellipsoids_fast(wc, wr, wq)

            if self._last_mesh_result is not None:
                r = self._last_mesh_result
                ell_set = EllipsoidSet(device=self._device)
                ell_set.set_parameters(wc, wr, wq)
                ell_grid = ell_set.compute_sdf_grid(
                    origin=r.origin, dx=r.dx, n=r.n,
                )
                self._ell_sdf_panel.set_sdf(ell_grid)

        self._status.showMessage(
            f"Pose {pose_index}: {rm.poses[pose_index].name if pose_index < len(rm.poses) else 'T-Pose'}"
        )

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

        self._multipose_worker = MultiPoseOptimizationWorker(
            rest_vertices=rm.vertices,
            faces=rm.faces,
            skeleton=rm.skeleton,
            skin_joints=rm.skin_joints,
            skin_weights=rm.skin_weights,
            poses=rm.poses,
            bone_local=bl,
            mapper=mapper,
            num_steps=self._rig_panel.multi_pose_steps,
            steps_per_pose=self._rig_panel.multi_pose_steps_per_pose,
            report_every=20,
            grid_n=grid_n,
            margin=margin,
            lr=self._rig_panel.multi_pose_lr,
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
        self._ell_viewer.show_ellipsoids_fast(centers, radii, rotations)
        self._status.showMessage(
            f"Multi-pose training — step {step}  loss={loss:.6f}  pose={pose_name}"
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
            self._mesh_viewer.show_mesh(cached[pose_index], rm.faces)
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
        self._status.showMessage("Multi-pose training finished.")
        self._btn_fit.setEnabled(self._last_mesh_result is not None)
        self._btn_stop.setEnabled(False)
        self._rig_panel.set_progress(100, 100)

        # Update bone-local params from worker result
        if self._multipose_worker is not None:
            self._rig_panel.set_bone_local(self._multipose_worker.result)

        self._multipose_worker = None

        # Update ellipsoid view for current pose
        self._on_rig_pose_changed(self._rig_panel.current_pose_index)

    # ── Original methods (unchanged below) ────────────────────────────────

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
        maintenance = 200 if self._chk_pruning.isChecked() else 0
        self.start_optimization(
            num_ellipsoids=num_e,
            method="adam",
            num_steps=self._spin_max_steps.value(),
            report_every=20,
            sdf_mode=sdf_mode,
            maintenance_every=maintenance,
        )

    def _on_stop_clicked(self):
        self.stop_optimization()
        self._stop_multipose()

    # ── async optimization ────────────────────────────────────────────────

    def start_optimization(
        self,
        num_ellipsoids: int = 10,
        method: str = "adam",
        num_steps: int = 2000,
        report_every: int = 20,
        sdf_mode: int = SDF_QUILEZ,
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

        # Keep ellipsoids reference for rig assignment
        self._ellipsoids = EllipsoidSet(device=self._device)
        self._ellipsoids.set_parameters(centers, radii, rotations)

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

        # Enable bone assignment if rig mode is active
        if self._rig_panel.is_active and self._ellipsoids is not None:
            self._rig_panel._btn_assign.setEnabled(True)
