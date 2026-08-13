"""Headless regression test for Local Fit region picking on a real character.

Loads the rigged Mixamo T-pose mesh, computes the real mesh SDF + local
thickness field, then injects two synthetic under-representation misses:

  * one on an actual finger vertex region (thin),
  * one around the torso/spine (thick).

The torso miss is made stronger in raw error, but Local Fit's thin-preferred
ranking should still pick the finger first and keep the local box centred on the
finger seed.  This catches regressions where the picker snaps back to the
nearest large belly/torso ellipsoid.

Run:  .venv/Scripts/python.exe tools/local_fit_fingers_test.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PySide6 import QtCore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ellipsoid import best_device  # noqa: E402
from optimization import OptimizationWorker  # noqa: E402
from rig_panel import try_load_rigged  # noqa: E402
from sdf_compute import SdfComputer  # noqa: E402


PROJECT = Path(__file__).resolve().parent.parent
MESH_PATH = PROJECT / "meshes" / "T-Pose.fbx"


def _bone_indices(skeleton, tokens: tuple[str, ...]) -> set[int]:
    out = set()
    for i, b in enumerate(skeleton.bones):
        name = b.name.lower()
        if any(tok.lower() in name for tok in tokens):
            out.add(i)
    return out


def _weighted_vertices(rm, bone_ids: set[int], min_weight: float = 0.45) -> np.ndarray:
    joints = np.asarray(rm.skin_joints)
    weights = np.asarray(rm.skin_weights)
    mask = np.zeros(joints.shape[0], dtype=bool)
    for col in range(joints.shape[1]):
        mask |= np.isin(joints[:, col], list(bone_ids)) & (weights[:, col] >= min_weight)
    return np.asarray(rm.vertices, np.float32)[mask]


def _grid_points_for(grid_shape, origin, dx):
    nz, ny, nx = grid_shape
    iz, iy, ix = np.indices((nz, ny, nx), dtype=np.float32)
    return np.stack([
        origin[0] + (ix + 0.5) * dx,
        origin[1] + (iy + 0.5) * dx,
        origin[2] + (iz + 0.5) * dx,
    ], axis=-1).reshape(-1, 3)


def _nearest_good_voxel(points, target_flat, thickness_flat, center, *, want_thin: bool) -> int:
    interior = target_flat < 0.0
    finite_thick = thickness_flat > 0.0
    candidates = np.where(interior & finite_thick)[0]
    if candidates.size == 0:
        raise RuntimeError("No interior voxels with valid thickness found")

    th = thickness_flat[candidates]
    if want_thin:
        keep = th <= np.percentile(th, 25)
    else:
        keep = th >= np.percentile(th, 75)
    candidates = candidates[keep]
    if candidates.size == 0:
        raise RuntimeError("Thickness percentile filter removed all candidates")

    d2 = np.sum((points[candidates] - center[None, :]) ** 2, axis=1)
    return int(candidates[int(np.argmin(d2))])


def main() -> int:
    QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv)

    rm = try_load_rigged(str(MESH_PATH))
    if rm is None:
        print(f"Could not load rigged mesh: {MESH_PATH}")
        return 1

    finger_bones = _bone_indices(rm.skeleton, ("HandThumb", "HandIndex", "HandMiddle",
                                               "HandRing", "HandPinky"))
    torso_bones = _bone_indices(rm.skeleton, ("Hips", "Spine", "Spine1", "Spine2"))
    finger_verts = _weighted_vertices(rm, finger_bones, min_weight=0.35)
    torso_verts = _weighted_vertices(rm, torso_bones, min_weight=0.45)
    if len(finger_verts) == 0 or len(torso_verts) == 0:
        print("Could not find weighted finger/torso vertices")
        return 1

    # Use the right-hand fingers; this avoids averaging left+right hands back
    # toward the character centre.
    finger_center = finger_verts[finger_verts[:, 0] > np.median(finger_verts[:, 0])].mean(axis=0)
    torso_center = torso_verts.mean(axis=0)

    dev = best_device()
    sdf = SdfComputer(device=dev)
    sdf.set_mesh(np.asarray(rm.vertices, np.float32), np.asarray(rm.faces, np.int32))
    res = sdf.compute_voxel_grid(n=64, margin=0.1)
    if res.thickness is None:
        print("SDF result did not contain thickness")
        return 1

    target = np.asarray(res.grid, np.float32)
    thickness = np.asarray(res.thickness, np.float32)
    points = _grid_points_for(target.shape, res.origin.astype(np.float32), float(res.dx))
    flat_t = target.ravel()
    flat_th = thickness.ravel()

    finger_flat = _nearest_good_voxel(points, flat_t, flat_th, finger_center, want_thin=True)
    torso_flat = _nearest_good_voxel(points, flat_t, flat_th, torso_center, want_thin=False)

    pred = target.copy().ravel()
    pred[finger_flat] = flat_t[finger_flat] + 0.7 * float(res.dx)
    pred[torso_flat] = flat_t[torso_flat] + 2.0 * float(res.dx)
    pred = pred.reshape(target.shape).astype(np.float32)

    worker = OptimizationWorker(
        sdf_target_np=target,
        origin=res.origin,
        dx=float(res.dx),
        n=int(res.n),
        num_ellipsoids=1,
        num_steps=1,
        method="adam",
        thickness_np=thickness,
        sdf_computer=sdf,
        local_fit=True,
        superfit=False,
    )
    def _sample_pred(self, sample_points, _c, _r, _q):
        ijk = np.floor(
            (np.asarray(sample_points, np.float32) - res.origin[None, :])
            / float(res.dx)
        ).astype(np.int64)
        ijk = np.clip(
            ijk,
            0,
            np.array([target.shape[2] - 1,
                      target.shape[1] - 1,
                      target.shape[0] - 1]),
        )
        return pred[ijk[:, 2], ijk[:, 1], ijk[:, 0]]

    worker._pred_points_from_params = types.MethodType(_sample_pred, worker)

    regions = worker._detect_worst_regions(
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.float32),
        np.empty((0, 4), np.float32),
        k=2,
        min_severity=0.0,
        thin_preference=worker._local_fit_thin_preference,
    )
    if not regions:
        print("No Local Fit regions detected")
        return 1

    first = np.asarray(regions[0]["seed_world"], np.float32)
    finger_point = points[finger_flat]
    torso_point = points[torso_flat]
    d_finger = float(np.linalg.norm(first - finger_point))
    d_torso = float(np.linalg.norm(first - torso_point))
    half = float(worker._region_radius_vox) * float(res.dx)
    box_min, box_max = worker._region_box(first, half)

    ok = (
        d_finger <= 1.5 * float(res.dx)
        and d_finger < d_torso
        and np.all(first >= box_min)
        and np.all(first <= box_max)
    )

    print("finger bones:", len(finger_bones), "finger verts:", len(finger_verts))
    print("torso bones:", len(torso_bones), "torso verts:", len(torso_verts))
    print("finger voxel:", np.round(finger_point, 4), "thickness:", float(flat_th[finger_flat]))
    print("torso voxel: ", np.round(torso_point, 4), "thickness:", float(flat_th[torso_flat]))
    print("picked:      ", np.round(first, 4))
    print("distances: finger=", f"{d_finger:.6f}", "torso=", f"{d_torso:.6f}")
    print("box:        ", np.round(box_min, 4), np.round(box_max, 4))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
