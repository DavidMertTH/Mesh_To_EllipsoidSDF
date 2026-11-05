"""Fit one detail zone in its own process, concurrently with the main fit.

Warp's autodiff tape is process-global, so two optimizers cannot train in
parallel threads of one process.  Each zone therefore runs here, in a separate
Python process on the same GPU, while the main window trains the rest of the
mesh.

usage: python detail_zone_worker.py <job.npz>

The job file holds the zone's normalized mesh (``vertices``, ``faces``) and a
JSON ``job`` string: grid ``n``, ``margin``, SDF options, ``device``, ``seed``
and the ``OptimizationWorker`` keyword arguments.  Progress, preview frames
and the result are written to stdout as JSON lines prefixed with
``detail_zones.MESSAGE_PREFIX``; any other output is log text.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import detail_zones as dz  # noqa: E402

FRAME_INTERVAL = 0.25          # seconds between preview frames


def emit(message: dict) -> None:
    sys.stdout.write(dz.MESSAGE_PREFIX + json.dumps(message) + "\n")
    sys.stdout.flush()


def run(job_path: str) -> int:
    archive = np.load(job_path, allow_pickle=False)
    job = json.loads(str(archive["job"]))
    vertices = np.ascontiguousarray(archive["vertices"], dtype=np.float32)
    faces = np.ascontiguousarray(archive["faces"], dtype=np.int32)
    if job.get("seed") is not None:
        np.random.seed(int(job["seed"]))

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore
    _app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])  # noqa: F841

    from optimization import OptimizationWorker
    from sdf_blowup import sparse_band_offsets
    from sdf_compute import SdfComputer

    emit({"type": "stage", "stage": "sdf"})
    computer = SdfComputer(device=job.get("device"))
    computer.set_mesh(vertices, faces)
    margin = float(job["margin"])
    result = computer.compute_voxel_grid(
        n=int(job["n"]), margin=margin, compute_thickness=True,
        thickness_max_resolution=int(job.get("thickness_max_resolution", 128)))
    samples = None
    if job.get("use_sparse_sdf", True):
        samples = computer.compute_sparse_samples(
            n=int(result.n), margin=margin, thickness_result=result,
            offsets_vox=sparse_band_offsets(0.0),
            thickness_sampling_power=float(job.get("thickness_sampling_power", 1.0)))

    base_kwargs = dict(job["kwargs"])
    base_kwargs.update(
        sdf_target_np=np.asarray(result.grid, dtype=np.float32),
        sdf_samples=samples,
        origin=result.origin,
        dx=result.dx,
        n=result.n,
        thickness_np=getattr(result, "thickness", None),
        sdf_blowup_fraction=0.0,
        sdf_computer=computer,
    )
    total = int(base_kwargs.get("num_steps", 0) or 0)
    state = {"sent": 0.0}

    def fit(num: int, max_count: int):
        kwargs = dict(base_kwargs, num_ellipsoids=int(num), max_ellipsoids=int(max_count))
        worker = OptimizationWorker(**kwargs)
        frame = {"last": None}

        def on_step(step, loss, centers, radii, rotations, shape):
            frame["last"] = (np.asarray(centers, np.float32).copy(), np.asarray(radii, np.float32).copy(),
                             np.asarray(rotations, np.float32).copy(),
                             None if shape is None else np.asarray(shape, np.float32).copy())
            now = time.perf_counter()
            if now - state["sent"] >= FRAME_INTERVAL:
                state["sent"] = now
                emit({"type": "frame", "step": int(step), "total": total, "loss": float(loss),
                      **dz.pack_params(*frame["last"])})

        worker.step_visual.connect(on_step, QtCore.Qt.DirectConnection)
        worker.run()                   # synchronous: this process is the worker thread
        return frame["last"], getattr(worker, "error_message", None)

    emit({"type": "stage", "stage": "fit", "total": total})
    num, max_count = int(base_kwargs.get("num_ellipsoids", 8)), int(base_kwargs.get("max_ellipsoids", 8))
    params, error = fit(num, max_count)
    if error:
        emit({"type": "error", "message": str(error)})
        return 2
    if params is None:
        emit({"type": "error", "message": "zone fit produced no primitives"})
        return 3

    # Quality check: a whole finger or toe left uncovered is a few percent of
    # the zone surface.  Refit once with more primitives and keep the better.
    capsule = base_kwargs.get("primitive_shape") == "capsule"
    inset = 0.25 * float(job.get("median_diameter", 0.0) or 0.0)
    query = lambda pts: computer.query_points(np.ascontiguousarray(pts, np.float32))  # noqa: E731
    limit = float(job.get("max_uncovered", 0.04))
    if inset > 0.0:
        missing = dz.uncovered_fraction(vertices, faces, query, params, inset, capsule=capsule)
        emit({"type": "check", "uncovered": missing, "primitives": int(len(params[0]))})
        if missing > limit and job.get("retry", True):
            grow = float(job.get("retry_growth", 1.5))
            retry, error = fit(int(round(num * grow)), int(round(max_count * grow)))
            if retry is not None and not error:
                retry_missing = dz.uncovered_fraction(vertices, faces, query, retry, inset,
                                                      capsule=capsule)
                emit({"type": "check", "uncovered": retry_missing,
                      "primitives": int(len(retry[0])), "retry": True})
                if retry_missing < missing:
                    params = retry
    emit({"type": "result", **dz.pack_params(*params)})
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 64
    try:
        return run(sys.argv[1])
    except Exception as exc:          # report instead of a silent crash
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
