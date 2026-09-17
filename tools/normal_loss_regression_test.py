"""CPU regressions for sparse target normals and the normal-alignment loss.

Run: .venv/Scripts/python.exe tools/normal_loss_regression_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimization as optimization_module  # noqa: E402
import fit_validation as fit_validation_module  # noqa: E402
from optimization import (  # noqa: E402
    OptimizationWorker,
    _ellipsoid_sdf_kernel_points,
    _normal_loss_kernel_batch,
    _primitive_normal_kernel_batch,
    _primitive_normal_kernel_points,
    _primitive_surface_normal,
)
from fit_validation import ValidationSample, evaluate_validation_loss  # noqa: E402
from sdf_samples import (  # noqa: E402
    SdfSampleSet,
    UploadedSdfSamples,
    sdf_grid_normals,
)


DEVICE = "cpu"


@wp.kernel
def _legacy_normal_kernel_points(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    eps: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    normal_scan: wp.array2d(dtype=wp.vec3),
    num_e: int,
    shape_kind: int,
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Pre-optimization running-min scan, retained as a parity oracle."""
    bid = wp.tid()
    p = points[indices[bid]]
    normal_scan[bid, 0] = wp.vec3(0.0, 0.0, 0.0)
    for i in range(num_e):
        next_normal = normal_scan[bid, i]
        if min_d[bid, i + 1] < min_d[bid, i]:
            base = i * 4
            q = wp.normalize(wp.quat(
                rot_flat[base + 0], rot_flat[base + 1],
                rot_flat[base + 2], rot_flat[base + 3]))
            be = i * 2
            next_normal = _primitive_surface_normal(
                p, centers[i], radii[i], q,
                eps[be], eps[be + 1], bend[be], bend[be + 1],
                shape_kind,
            )
        normal_scan[bid, i + 1] = next_normal
    out_normals[bid] = normal_scan[bid, num_e]


@wp.kernel
def _legacy_normal_kernel_batch(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rot_flat: wp.array(dtype=wp.float32),
    eps: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    min_d: wp.array2d(dtype=wp.float32),
    normal_scan: wp.array2d(dtype=wp.vec3),
    num_e: int,
    shape_kind: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    indices: wp.array(dtype=wp.int32),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Dense-sample variant of the pre-optimization normal scan."""
    bid = wp.tid()
    tid = indices[bid]
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)
    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )
    normal_scan[bid, 0] = wp.vec3(0.0, 0.0, 0.0)
    for i in range(num_e):
        next_normal = normal_scan[bid, i]
        if min_d[bid, i + 1] < min_d[bid, i]:
            base = i * 4
            q = wp.normalize(wp.quat(
                rot_flat[base + 0], rot_flat[base + 1],
                rot_flat[base + 2], rot_flat[base + 3]))
            be = i * 2
            next_normal = _primitive_surface_normal(
                p, centers[i], radii[i], q,
                eps[be], eps[be + 1], bend[be], bend[be + 1],
                shape_kind,
            )
        normal_scan[bid, i + 1] = next_normal
    out_normals[bid] = normal_scan[bid, num_e]


def _normal_loss(
    predicted: np.ndarray,
    target: np.ndarray,
    target_sdf: np.ndarray,
    *,
    weight: float = 2.5,
    band: float = 0.1,
) -> float:
    """Evaluate the production kernel on one complete, deterministic batch."""
    predicted = np.ascontiguousarray(predicted, dtype=np.float32).reshape(-1, 3)
    target = np.ascontiguousarray(target, dtype=np.float32).reshape(-1, 3)
    target_sdf = np.ascontiguousarray(target_sdf, dtype=np.float32).reshape(-1)
    if not (len(predicted) == len(target) == len(target_sdf)):
        raise ValueError("normal-loss test arrays must have the same length")

    batch_size = len(target_sdf)
    indices = np.arange(batch_size, dtype=np.int32)
    loss = wp.zeros(1, dtype=wp.float32, device=DEVICE)
    wp.launch(
        _normal_loss_kernel_batch,
        dim=batch_size,
        inputs=[
            wp.array(predicted, dtype=wp.vec3, device=DEVICE),
            wp.array(target, dtype=wp.vec3, device=DEVICE),
            wp.array(target_sdf, dtype=wp.float32, device=DEVICE),
            wp.array(indices, dtype=wp.int32, device=DEVICE),
            loss,
            batch_size,
            float(weight),
            float(band),
        ],
        device=DEVICE,
    )
    return float(loss.numpy()[0])


class NormalLossRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_loss_prefers_matching_oriented_normals(self) -> None:
        target = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        on_surface = np.array([0.0], dtype=np.float32)

        matching = _normal_loss(
            np.array([[7.0, 0.0, 0.0]], dtype=np.float32),
            target,
            on_surface,
        )
        orthogonal = _normal_loss(
            np.array([[0.0, 3.0, 0.0]], dtype=np.float32),
            target,
            on_surface,
        )
        opposite = _normal_loss(
            np.array([[-4.0, 0.0, 0.0]], dtype=np.float32),
            target,
            on_surface,
        )

        self.assertAlmostEqual(matching, 0.0, places=6)
        self.assertAlmostEqual(orthogonal, 2.5, places=6)
        self.assertAlmostEqual(opposite, 5.0, places=6)
        self.assertLess(matching, orthogonal)
        self.assertLess(orthogonal, opposite)

    def test_loss_is_narrow_band_and_skips_missing_normals(self) -> None:
        # Only row zero is both in-band and equipped with two valid normals.
        value = _normal_loss(
            np.array([
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ], dtype=np.float32),
            np.array([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32),
            np.array([0.0, 0.11, 0.0, 0.0], dtype=np.float32),
            weight=4.0,
            band=0.1,
        )
        # The kernel averages over the full batch, including deliberately
        # skipped rows: 4 * (1 - dot(orthogonal)) / 4 == 1.
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_predicted_ellipsoid_normal_matches_implicit_gradient(self) -> None:
        root_two = np.float32(np.sqrt(2.0))
        points_np = np.array([
            [2.0, 0.0, 0.0],
            [root_two, 1.0 / root_two, 0.0],
        ], dtype=np.float32)
        count = len(points_np)
        points = wp.array(points_np, dtype=wp.vec3, device=DEVICE)
        indices = wp.array(
            np.arange(count, dtype=np.int32), dtype=wp.int32, device=DEVICE)
        centers = wp.array(
            np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=DEVICE)
        radii = wp.array(
            np.array([[2.0, 1.0, 0.5]], dtype=np.float32),
            dtype=wp.vec3,
            device=DEVICE,
        )
        rotations = wp.array(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            dtype=wp.float32,
            device=DEVICE,
        )
        min_distance = wp.zeros(
            (count, 2), dtype=wp.float32, device=DEVICE)
        predicted_sdf = wp.empty(count, dtype=wp.float32, device=DEVICE)
        wp.launch(
            _ellipsoid_sdf_kernel_points,
            dim=count,
            inputs=[
                centers, radii, rotations, min_distance, 1,
                points, indices, predicted_sdf,
            ],
            device=DEVICE,
        )

        predicted_normals = wp.empty(count, dtype=wp.vec3, device=DEVICE)
        eps = wp.array(
            np.ones(2, dtype=np.float32), dtype=wp.float32, device=DEVICE)
        bend = wp.array(
            np.zeros(2, dtype=np.float32), dtype=wp.float32, device=DEVICE)
        wp.launch(
            _primitive_normal_kernel_points,
            dim=count,
            inputs=[
                centers, radii, rotations, eps, bend,
                min_distance, 1, 0,
                points, indices, predicted_normals,
            ],
            device=DEVICE,
        )

        expected = points_np / np.square(
            np.array([2.0, 1.0, 0.5], dtype=np.float32))
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(
            predicted_sdf.numpy(), np.zeros(count), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            predicted_normals.numpy(), expected, rtol=0.0, atol=1.0e-6)

    def test_normal_loss_backpropagates_to_ellipsoid_shape(self) -> None:
        point_np = np.array(
            [[np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]], dtype=np.float32)
        points = wp.array(point_np, dtype=wp.vec3, device=DEVICE)
        indices = wp.array(
            np.array([0], dtype=np.int32), dtype=wp.int32, device=DEVICE)
        centers = wp.array(
            np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3,
            device=DEVICE, requires_grad=True)
        radii = wp.array(
            np.array([[2.0, 1.0, 1.0]], dtype=np.float32), dtype=wp.vec3,
            device=DEVICE, requires_grad=True)
        rotations = wp.array(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            dtype=wp.float32, device=DEVICE, requires_grad=True)
        eps = wp.array(
            np.ones(2, dtype=np.float32), dtype=wp.float32,
            device=DEVICE, requires_grad=True)
        bend = wp.array(
            np.zeros(2, dtype=np.float32), dtype=wp.float32,
            device=DEVICE, requires_grad=True)
        min_distance = wp.zeros(
            (1, 2), dtype=wp.float32, device=DEVICE, requires_grad=True)
        predicted_sdf = wp.empty(
            1, dtype=wp.float32, device=DEVICE, requires_grad=True)
        predicted_normals = wp.empty(
            1, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        target_normals = wp.array(
            np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
            dtype=wp.vec3, device=DEVICE)
        target_sdf = wp.array(
            np.array([0.0], dtype=np.float32), dtype=wp.float32, device=DEVICE)
        loss = wp.zeros(
            1, dtype=wp.float32, device=DEVICE, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                _ellipsoid_sdf_kernel_points,
                dim=1,
                inputs=[
                    centers, radii, rotations, min_distance, 1,
                    points, indices, predicted_sdf,
                ],
                device=DEVICE,
            )
            wp.launch(
                _primitive_normal_kernel_points,
                dim=1,
                inputs=[
                    centers, radii, rotations, eps, bend,
                    min_distance, 1, 0,
                    points, indices, predicted_normals,
                ],
                device=DEVICE,
            )
            wp.launch(
                _normal_loss_kernel_batch,
                dim=1,
                inputs=[
                    predicted_normals, target_normals, target_sdf,
                    indices, loss, 1, 1.0, 0.1,
                ],
                device=DEVICE,
            )
        tape.backward(loss)

        radius_gradient = radii.grad.numpy()[0]
        self.assertTrue(np.isfinite(radius_gradient).all())
        self.assertGreater(float(np.linalg.norm(radius_gradient)), 1.0e-6)

    def test_winner_only_normals_match_legacy_values_and_gradients(self) -> None:
        """Later winners, ties and no-winner rows retain the old AD result."""
        shape = (6, 6, 6)
        origin_np = np.array([-0.75, -0.75, -0.75], dtype=np.float32)
        dx = 0.37
        voxels = [(1, 2, 3), (3, 1, 2), (2, 4, 1), (4, 3, 3), (3, 4, 2)]
        grid_ids = np.array(
            [x + shape[0] * (y + shape[1] * z) for x, y, z in voxels],
            dtype=np.int32,
        )
        points_np = np.array([
            origin_np + (np.array(xyz, dtype=np.float32) + 0.5) * dx
            for xyz in voxels
        ], dtype=np.float32)
        min_np = np.array([
            [1.0e6, 0.4, 0.4, -0.2],  # final winner 2
            [1.0e6, 0.2, -0.4, -0.4],  # final winner 1
            [1.0e6, 0.1, 0.1, 0.1],  # tie keeps winner 0
            [1.0e6, 1.0e6, 1.0e6, 1.0e6],  # no winner
            [1.0e6, 0.5, 0.2, -0.1],  # multiple improvements
        ], dtype=np.float32)
        centers_np = np.array([
            [0.05, -0.1, 0.15],
            [-0.2, 0.1, -0.1],
            [0.15, 0.05, -0.2],
        ], dtype=np.float32)
        radii_np = np.array([
            [0.7, 0.9, 0.6],
            [0.8, 0.5, 0.75],
            [0.6, 0.65, 0.9],
        ], dtype=np.float32)
        rotations_np = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), 3)
        eps_np = np.array([1.2, 0.8, 0.9, 1.1, 1.3, 0.7], dtype=np.float32)
        targets_np = np.tile(
            np.array([0.4, -0.8, 0.2], dtype=np.float32),
            (len(voxels), 1),
        )

        def run(device: str, *, legacy: bool, dense: bool):
            count = len(voxels)
            centers = wp.array(
                centers_np, dtype=wp.vec3, device=device, requires_grad=True)
            radii = wp.array(
                radii_np, dtype=wp.vec3, device=device, requires_grad=True)
            rotations = wp.array(
                rotations_np, dtype=wp.float32, device=device,
                requires_grad=True)
            eps = wp.array(
                eps_np, dtype=wp.float32, device=device, requires_grad=True)
            bend = wp.array(
                np.zeros(6, dtype=np.float32), dtype=wp.float32,
                device=device, requires_grad=True)
            min_d = wp.array(
                min_np, dtype=wp.float32, device=device, requires_grad=True)
            points = wp.array(points_np, dtype=wp.vec3, device=device)
            indices = wp.array(
                grid_ids if dense else np.arange(count, dtype=np.int32),
                dtype=wp.int32, device=device)
            scan = wp.zeros(
                (count, 4), dtype=wp.vec3, device=device,
                requires_grad=True) if legacy else None
            predicted = wp.empty(
                count, dtype=wp.vec3, device=device, requires_grad=True)
            target = wp.array(targets_np, dtype=wp.vec3, device=device)
            target_sdf = wp.array(
                np.zeros(count, dtype=np.float32), dtype=wp.float32,
                device=device)
            loss = wp.zeros(
                1, dtype=wp.float32, device=device, requires_grad=True)
            inputs = [centers, radii, rotations, eps, bend, min_d]
            if legacy:
                inputs.append(scan)
            inputs.extend([3, 4])
            if dense:
                inputs.extend([
                    wp.vec3(*origin_np), dx, *shape, indices, predicted,
                ])
                kernel = (_legacy_normal_kernel_batch if legacy
                          else _primitive_normal_kernel_batch)
            else:
                inputs.extend([points, indices, predicted])
                kernel = (_legacy_normal_kernel_points if legacy
                          else _primitive_normal_kernel_points)
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=count, inputs=inputs, device=device)
                wp.launch(
                    _normal_loss_kernel_batch,
                    dim=count,
                    inputs=[
                        predicted, target, target_sdf, indices if not dense
                        else wp.array(np.arange(count, dtype=np.int32),
                                      dtype=wp.int32, device=device),
                        loss, count, 1.0, 0.1,
                    ],
                    device=device,
                )
            tape.backward(loss)
            return (
                predicted.numpy(), float(loss.numpy()[0]),
                centers.grad.numpy(), radii.grad.numpy(), eps.grad.numpy(),
            )

        devices = ["cpu"] + [str(dev) for dev in wp.get_cuda_devices()]
        for dev in devices:
            for dense in (False, True):
                with self.subTest(device=dev, dense=dense):
                    old = run(dev, legacy=True, dense=dense)
                    new = run(dev, legacy=False, dense=dense)
                    for previous, current in zip(old, new):
                        np.testing.assert_allclose(
                            current, previous, rtol=2.0e-5, atol=2.0e-6)
                    np.testing.assert_array_equal(new[0][3], np.zeros(3))
                    self.assertGreater(float(np.linalg.norm(new[2])), 1.0e-6)
                    self.assertGreater(float(np.linalg.norm(new[3])), 1.0e-6)
                    self.assertGreater(float(np.linalg.norm(new[4])), 1.0e-6)

    def test_sample_transforms_and_upload_preserve_normals(self) -> None:
        normals = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        samples = SdfSampleSet(
            points=np.array([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ], dtype=np.float32),
            values=np.array([-0.05, 0.0, 0.05], dtype=np.float32),
            thickness=np.full(3, 0.5, dtype=np.float32),
            dx=0.1,
            source="normal-metadata-test",
            coarse_mask=np.array([False, False, True]),
            normals=normals,
        )

        shifted = samples.with_offset(0.02)
        limited = samples.with_thickness_limited_offset(-0.02)
        np.testing.assert_array_equal(shifted.normals, normals)
        np.testing.assert_array_equal(limited.normals, normals)

        uploaded = UploadedSdfSamples(samples, DEVICE)
        np.testing.assert_array_equal(uploaded.normals.numpy(), normals)

    def test_thickness_limited_offset_masks_mathematically_stale_normals(self) -> None:
        normals = np.eye(3, dtype=np.float32)
        samples = SdfSampleSet(
            points=np.array([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ], dtype=np.float32),
            values=np.array([-0.1, 0.0, 0.1], dtype=np.float32),
            # Row zero is unresolved, row one is thickness-capped, and row two
            # can accept the complete requested offset.
            thickness=np.array([0.0, 0.2, 4.0], dtype=np.float32),
            dx=0.1,
            normals=normals,
        )

        adjusted = samples.with_thickness_limited_offset(
            -0.2, max_thickness_fraction=0.25)

        np.testing.assert_allclose(
            adjusted.values - samples.values,
            np.array([0.0, -0.05, -0.2], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-7,
        )
        # A local thickness cap is a spatially varying offset.  Its gradient is
        # unavailable for unstructured sparse samples, so retaining the source
        # normal would be mathematically stale.  Zero is the loss's skip marker.
        np.testing.assert_array_equal(
            adjusted.normals,
            np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),
        )

    def test_dense_grid_gradient_uses_world_xyz_order(self) -> None:
        dx = 0.2
        nz, ny, nx = 4, 5, 6
        iz, iy, ix = np.meshgrid(
            np.arange(nz, dtype=np.float32),
            np.arange(ny, dtype=np.float32),
            np.arange(nx, dtype=np.float32),
            indexing="ij",
        )
        grid = 2.0 * ix * dx + 3.0 * iy * dx + 4.0 * iz * dx
        normals = sdf_grid_normals(grid, dx)

        expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        self.assertEqual(normals.shape, (nz, ny, nx, 3))
        np.testing.assert_allclose(
            normals,
            np.broadcast_to(expected, normals.shape),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_validation_metric_includes_normal_alignment(self) -> None:
        sample = ValidationSample(
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            values=np.array([0.0], dtype=np.float32),
            source_indices=np.array([0], dtype=np.int64),
            strata=np.array([0], dtype=np.uint8),
            dx=0.25,
            normals=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )
        common = dict(
            huber_delta=0.1,
            miss_weight=0.0,
            surface_weight=0.0,
            outside_weight=0.0,
            thin_weight=0.0,
            coarse_far_weight=0.0,
            normal_weight=2.0,
            normal_band=0.5,
        )
        matching = evaluate_validation_loss(
            np.array([0.0], dtype=np.float32), sample,
            prediction_normals=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            **common,
        )
        opposite = evaluate_validation_loss(
            np.array([0.0], dtype=np.float32), sample,
            prediction_normals=np.array([[-1.0, 0.0, 0.0]], dtype=np.float32),
            **common,
        )
        self.assertAlmostEqual(matching.normal, 0.0, places=7)
        self.assertAlmostEqual(opposite.normal, 1.0, places=7)
        self.assertLess(matching.total, opposite.total)

    def test_symmetry_reflects_normal_axis_component(self) -> None:
        samples = SdfSampleSet(
            points=np.array([
                [-2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ], dtype=np.float32),
            values=np.array([2.0, 1.0, 0.0, 1.0, 2.0], dtype=np.float32),
            normals=np.array([
                [-1.0, 0.0, 0.0],
                [-0.6, 0.8, 0.0],
                [0.0, 1.0, 0.0],
                [0.6, 0.8, 0.0],
                [1.0, 0.0, 0.0],
            ], dtype=np.float32),
            dx=0.25,
            source="normal-pair-test",
        )

        paired = OptimizationWorker._paired_symmetric_samples(
            samples, axis=0, plane=0.0, tolerance=1.0e-6)

        np.testing.assert_array_equal(
            paired.points,
            np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
            ], dtype=np.float32),
        )
        np.testing.assert_allclose(
            paired.normals,
            np.array([
                [0.0, 1.0, 0.0],
                [0.6, 0.8, 0.0],
                [1.0, 0.0, 0.0],
                [-0.6, 0.8, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertEqual(paired.source, "normal-pair-test-symmetric")

    def test_sparse_adam_path_runs_with_active_normal_loss(self) -> None:
        n = 4
        dx = 0.5
        origin = np.full(3, -1.0, dtype=np.float32)
        coords = origin[0] + (
            np.arange(n, dtype=np.float32) + np.float32(0.5)) * dx
        z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
        dense = (np.sqrt(x * x + y * y + z * z) - 0.5).astype(np.float32)
        points = np.array([
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
        ], dtype=np.float32)
        samples = SdfSampleSet(
            points=points,
            values=np.zeros(len(points), dtype=np.float32),
            normals=points / 0.5,
            thickness=np.full(len(points), 1.0, dtype=np.float32),
            coarse_mask=np.zeros(len(points), dtype=np.bool_),
            dx=dx,
            source="normal-adam-smoke",
        )
        worker = OptimizationWorker(
            sdf_target_np=dense,
            sdf_samples=samples,
            origin=origin,
            dx=dx,
            n=n,
            num_ellipsoids=1,
            num_steps=2,
            report_every=1,
            sample_budget=len(points),
            maintenance_every=0,
            containment_weight=0.0,
            flat_weight=0.0,
            superfit=False,
            local_fit=False,
            validation_patience=None,
            normal_loss_weight=1.0,
            normal_warmup_frac=0.0,
            normal_ramp_frac=0.0,
            initial_centers=np.zeros((1, 3), dtype=np.float32),
            initial_radii=np.array([[0.6, 0.4, 0.5]], dtype=np.float32),
            initial_rotations=np.array(
                [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        )
        reported: list[tuple[int, float]] = []
        worker.step_visual.connect(
            lambda step, loss, *_rest: reported.append((int(step), float(loss))))

        previous_device = optimization_module.device
        optimization_module.device = DEVICE
        try:
            worker._reset_stale_tape()
            worker._run_adam()
        finally:
            optimization_module.device = previous_device

        # The worker emits both training reports and one restored-checkpoint
        # frame.  Require both actual optimization steps without depending on
        # that final reporting detail.
        self.assertTrue({0, 1}.issubset({step for step, _loss in reported}))
        self.assertTrue(np.isfinite([loss for _step, loss in reported]).all())
        self.assertTrue(np.isfinite(worker.best_validation_loss))

    def test_disabled_normal_loss_skips_dense_normals_and_work_buffers(self) -> None:
        n = 3
        dx = 0.5
        origin = np.full(3, -0.75, dtype=np.float32)
        coords = origin[0] + (
            np.arange(n, dtype=np.float32) + np.float32(0.5)) * dx
        z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
        dense = (np.sqrt(x * x + y * y + z * z) - 0.5).astype(np.float32)
        worker = OptimizationWorker(
            sdf_target_np=dense,
            origin=origin,
            dx=dx,
            n=n,
            num_ellipsoids=1,
            num_steps=1,
            report_every=1,
            sample_budget=8,
            maintenance_every=0,
            containment_weight=0.0,
            flat_weight=0.0,
            superfit=False,
            local_fit=False,
            validation_sample_size=8,
            validation_patience=None,
            normal_loss_weight=0.0,
            initial_centers=np.zeros((1, 3), dtype=np.float32),
            initial_radii=np.full((1, 3), 0.4, dtype=np.float32),
            initial_rotations=np.array(
                [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        )

        allocated_normal_buffers: list[tuple[object, object]] = []
        original_alloc = worker._alloc_buffers

        def capture_allocations(*args, **kwargs):
            buffers = original_alloc(*args, **kwargs)
            allocated_normal_buffers.append((
                buffers.get("normal_scan"), buffers.get("pred_normals")))
            return buffers

        worker._alloc_buffers = capture_allocations

        def forbidden_dense_normals(*_args, **_kwargs):
            raise AssertionError(
                "normal_loss_weight=0 must not compute dense SDF normals")

        previous_device = optimization_module.device
        optimization_module.device = DEVICE
        original_optimizer_normals = optimization_module.sdf_grid_normals
        original_validation_normals = fit_validation_module.sdf_grid_normals
        optimization_module.sdf_grid_normals = forbidden_dense_normals
        fit_validation_module.sdf_grid_normals = forbidden_dense_normals
        try:
            worker._reset_stale_tape()
            worker._run_adam()
        finally:
            optimization_module.sdf_grid_normals = original_optimizer_normals
            fit_validation_module.sdf_grid_normals = original_validation_normals
            optimization_module.device = previous_device

        self.assertTrue(allocated_normal_buffers)
        self.assertTrue(all(
            scan is None and predicted is None
            for scan, predicted in allocated_normal_buffers
        ))


if __name__ == "__main__":
    unittest.main()
