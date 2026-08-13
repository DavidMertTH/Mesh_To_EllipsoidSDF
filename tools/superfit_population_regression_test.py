"""Real-mesh regression test for SuperFit population buffer replacement.

The fit must cross several population-changing SuperFit cycles.  This catches
stale center/rotation aliases that otherwise combine a new radii population
with the previous population and only fail later in the 3-D viewer.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ellipsoid import best_device  # noqa: E402
from mesh_io import load_and_prepare  # noqa: E402
from optimization import OptimizationWorker  # noqa: E402
from sdf_compute import SdfComputer  # noqa: E402
from viewer3d import SceneViewer3D, build_concatenated_mesh  # noqa: E402


def _union_sdf(
    centers: np.ndarray,
    radii: np.ndarray,
    rotations: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(points), 12_000):
        per_ellipsoid = OptimizationWorker._ellipsoid_sdf_np_batch(
            centers,
            radii,
            rotations,
            points[start:start + 12_000],
        )
        chunks.append(np.min(per_ellipsoid, axis=0))
    return np.concatenate(chunks)


def test_real_humanoid_superfit_population() -> None:
    mesh_path = ROOT / "meshes" / "T-Pose.fbx"
    mesh = load_and_prepare(mesh_path)
    sdf = SdfComputer(device=best_device())
    sdf.set_mesh(mesh.vertices, mesh.faces)
    dense = sdf.compute_voxel_grid(
        n=40,
        margin=0.12,
        compute_thickness=False,
    )
    samples = sdf.compute_sparse_samples(
        n=40,
        margin=0.12,
        surface_samples=8192,
        offsets_vox=(-2.0, -1.0, 0.0, 1.0, 2.0),
        coarse_n=10,
        seed=20260714,
    )

    num_steps = 300
    worker = OptimizationWorker(
        sdf_target_np=dense.grid,
        sdf_samples=samples,
        origin=dense.origin,
        dx=dense.dx,
        n=dense.n,
        num_ellipsoids=60,
        max_ellipsoids=72,
        num_steps=num_steps,
        report_every=10,
        sample_budget=4096,
        maintenance_every=0,
        superfit=True,
        superfit_every=50,
        densify_start_frac=0.0,
        densify_until_frac=0.65,
        spawn_underrep=True,
        split_enabled=True,
        merge_enabled=False,
        prune_enabled=False,
        local_fit=False,
        symmetry_enabled=False,
        lr_init=0.01,
        lr_final=0.0002,
    )
    worker._rng = np.random.default_rng(20260714)

    frames: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]] = []
    cycles: list[tuple[int, int, int, int]] = []

    def on_frame(step, loss, centers, radii, rotations, _extra) -> None:
        frames.append((
            int(step),
            float(loss),
            np.asarray(centers, dtype=np.float32).copy(),
            np.asarray(radii, dtype=np.float32).copy(),
            np.asarray(rotations, dtype=np.float32).copy(),
        ))

    worker.step_visual.connect(on_frame)
    worker.maintenance_done.connect(
        lambda step, before, removed, appended: cycles.append((
            int(step), int(before), int(removed), int(appended))))
    worker.run()

    assert frames, "optimizer emitted no visual frames"
    assert frames[-1][0] >= num_steps - 1, "optimizer stopped before final emit"
    assert cycles, "test did not exercise the SuperFit cadence"
    counts = [len(frame[2]) for frame in frames]
    assert any(count != counts[0] for count in counts[1:]), (
        f"SuperFit never changed the population: {counts}")
    assert max(counts) <= worker._max_ellipsoids, counts
    for step, _loss, centers, radii, rotations in frames:
        assert len(centers) == len(radii) == len(rotations), (
            f"population mismatch at step {step}: "
            f"{len(centers)}/{len(radii)}/{len(rotations)}")
        assert (
            np.isfinite(centers).all()
            and np.isfinite(radii).all()
            and np.isfinite(rotations).all()
        ), f"non-finite population values at step {step}"

    first = frames[0]
    final = frames[-1]
    near_surface = np.abs(samples.values) <= 2.0 * float(dense.dx)
    first_pred = _union_sdf(first[2], first[3], first[4], samples.points)
    final_pred = _union_sdf(final[2], final[3], final[4], samples.points)
    first_mae = float(np.mean(
        np.abs(first_pred[near_surface] - samples.values[near_surface])))
    final_mae = float(np.mean(
        np.abs(final_pred[near_surface] - samples.values[near_surface])))

    assert final[1] < first[1] * 0.5, (first[1], final[1])
    assert final_mae < first_mae, (first_mae, final_mae)
    center_sdf = sdf.query_points(final[2])
    assert float(np.mean(center_sdf <= dense.dx)) >= 0.95

    print(
        f"real humanoid SuperFit: counts={sorted(set(counts))}, "
        f"loss={first[1]:.6f}->{final[1]:.6f}, "
        f"near-surface MAE={first_mae:.6f}->{final_mae:.6f}"
    )


def test_viewer_legacy_population_mismatch() -> None:
    count = 66
    centers = np.zeros((count, 3), dtype=np.float32)
    radii = np.full((count + 1, 3), 0.1, dtype=np.float32)
    rotations = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (count, 1),
    )
    verts, faces, _colors = build_concatenated_mesh(
        centers, radii, rotations)

    viewer = object.__new__(SceneViewer3D)
    viewer._ell_centers = centers
    viewer._ell_radii = radii
    viewer._ell_rotations = rotations
    viewer._ell_sq_eps = None
    viewer._ell_sq_bend = None
    viewer._ell_primitive = None
    viewer._align_stored_ellipsoid_population()
    assert len(viewer._ell_centers) == count
    assert len(viewer._ell_radii) == count
    assert len(viewer._ell_rotations) == count
    normals = viewer._smooth_vertex_normals(verts, faces)

    assert normals.shape == verts.shape
    assert np.isfinite(normals).all()
    print("legacy viewer population mismatch: recovered")


def test_local_region_and_symmetry_budgets() -> None:
    worker = OptimizationWorker(
        sdf_target_np=-np.ones((4, 4, 4), dtype=np.float32),
        origin=np.array([-0.2, -0.2, -0.2], dtype=np.float32),
        dx=0.1,
        n=4,
        num_ellipsoids=2,
        max_ellipsoids=10,
        num_steps=1,
        maintenance_every=0,
        superfit=False,
        local_fit=False,
    )

    def fake_region_dc(
        self,
        _contrib_c,
        _contrib_r,
        _contrib_q,
        train_c,
        train_r,
        train_q,
        _res,
        _n_fixed,
        population_cap=None,
    ):
        del self, population_cap
        extra = np.repeat(np.arange(1, 5, dtype=np.float32)[:, None], 3, axis=1)
        extra_r = np.repeat(train_r[-1:], len(extra), axis=0)
        extra_q = np.repeat(train_q[-1:], len(extra), axis=0)
        return (
            np.concatenate([train_c, extra], axis=0),
            np.concatenate([train_r, extra_r], axis=0),
            np.concatenate([train_q, extra_q], axis=0),
        )

    worker._region_divide_conquer = MethodType(fake_region_dc, worker)
    fixed_c = np.zeros((2, 3), dtype=np.float32)
    fixed_r = np.full((2, 3), 0.1, dtype=np.float32)
    fixed_q = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1))
    train_c = np.arange(18, dtype=np.float32).reshape(6, 3)
    train_r = np.full((6, 3), 0.1, dtype=np.float32)
    train_q = np.tile(fixed_q[:1], (6, 1))
    train_box = np.repeat(np.arange(3, dtype=np.int32), 2)
    boxes = [{"res": object()} for _ in range(3)]
    out_c, out_r, out_q, out_box, changed = worker._region_dc_all_boxes(
        fixed_c,
        fixed_r,
        fixed_q,
        train_c,
        train_r,
        train_q,
        train_box,
        boxes,
        population_cap=10,
    )
    assert changed
    assert len(fixed_c) + len(out_c) == 10
    assert len(out_c) == len(out_r) == len(out_q) == len(out_box)
    np.testing.assert_array_equal(
        np.bincount(out_box, minlength=3), np.array([4, 2, 2]))

    worker._sym_axis = 0
    worker._sym_plane = 0.0
    positive = np.column_stack([
        np.linspace(0.16, 0.23, 8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
    ])
    centers = np.concatenate([positive, -positive], axis=0)
    radii = np.full((len(centers), 3), 0.05, dtype=np.float32)
    rotations = np.tile(fixed_q[:1], (len(centers), 1))
    sym_c, sym_r, sym_q, _sym_e = worker._build_symmetric_layout(
        centers, radii, rotations)
    assert len(sym_c) == len(sym_r) == len(sym_q) == worker._max_ellipsoids
    np.testing.assert_allclose(sym_c[:5, 0], -sym_c[5:, 0])
    print("local region and symmetry budgets: capped")


def main() -> None:
    test_real_humanoid_superfit_population()
    test_viewer_legacy_population_mismatch()
    test_local_region_and_symmetry_budgets()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
