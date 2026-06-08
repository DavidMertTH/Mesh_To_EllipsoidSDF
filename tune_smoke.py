"""Headless smoke test: load T-Pose.fbx, compute SDF, run a tiny fit.

Validates that the full fit pipeline runs without a GUI so a tuning harness can
drive it.  Run with the project venv:  .venv/Scripts/python.exe tune_smoke.py
"""
import time
import numpy as np
from PySide6 import QtCore
import warp as wp

wp.init()

from ellipsoid import best_device          # noqa: E402
from sdf_compute import SdfComputer        # noqa: E402
from optimization import OptimizationWorker  # noqa: E402
from rig_panel import try_load_rigged      # noqa: E402

app = QtCore.QCoreApplication([])
dev = best_device()
print("device:", dev)

rm = try_load_rigged("meshes/T-Pose.fbx")
verts = np.asarray(rm.vertices, np.float32).reshape(-1, 3)
faces = np.asarray(rm.faces, np.int64).reshape(-1, 3)
print("mesh verts/faces:", verts.shape, faces.shape)

sdf = SdfComputer(device=dev)
sdf.set_mesh(verts, faces)
t0 = time.time()
res = sdf.compute_voxel_grid(n=64, margin=0.1)
t_sdf = time.time() - t0
print(f"SDF grid {res.grid.shape} dx={res.dx:.5f}  t_sdf={t_sdf:.2f}s")

losses = []
w = OptimizationWorker(
    sdf_target_np=res.grid, origin=res.origin, dx=res.dx, n=res.n,
    num_ellipsoids=20, method="adam", num_steps=200, report_every=20,
    sdf_mode=4, maintenance_every=0, superfit=False, max_ellipsoids=20,
    thickness_np=res.thickness, sdf_computer=sdf, local_fit=False,
    symmetry_enabled=False,
)
w.step_visual.connect(lambda s, l, c, r, q, e: losses.append((s, l)))
t0 = time.time()
w.run()                       # blocking, runs on this thread (no event loop needed)
t_fit = time.time() - t0
print(f"fit done  t_fit={t_fit:.2f}s  steps_emitted={len(losses)}  "
      f"final_loss={losses[-1][1] if losses else None}")
