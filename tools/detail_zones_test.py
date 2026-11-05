"""Unit tests for detail_zones (thin-feature zones fitted at their own resolution).

The synthetic shape is a sphere (radius 0.5) with two thin rods mirrored across
x = 0; distances are analytic, so no GPU is needed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import detail_zones as dz  # noqa: E402
from mesh_io import normalize_mesh_capture  # noqa: E402

ROD_RADIUS = 0.02
ROD_START, ROD_END = 0.45, 0.85


def _rod_sdf(p, sign):
    a = np.array([sign * ROD_START, 0.0, 0.0])
    b = np.array([sign * ROD_END, 0.0, 0.0])
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1) - ROD_RADIUS


def shape_sdf(points):
    p = np.asarray(points, np.float64).reshape(-1, 3)
    sphere = np.linalg.norm(p, axis=1) - 0.5
    return np.minimum(sphere, np.minimum(_rod_sdf(p, 1.0), _rod_sdf(p, -1.0))).astype(np.float32)


def shape_mesh():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    parts = [sphere]
    for sign in (1.0, -1.0):
        rod = trimesh.creation.capsule(height=ROD_END - ROD_START, radius=ROD_RADIUS, count=[16, 16])
        rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        rod.apply_translation([sign * 0.5 * (ROD_START + ROD_END), 0.0, 0.0])
        parts.append(rod)
    mesh = trimesh.util.concatenate(parts)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64)


def quat_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class DiameterTest(unittest.TestCase):
    def test_sphere_diameter_is_exact(self):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
        d = dz.exact_vertex_diameters(sphere.vertices, sphere.faces,
                                      lambda p: (np.linalg.norm(p, axis=1) - 0.3).astype(np.float32))
        np.testing.assert_allclose(d, 0.6, rtol=0.08)

    def test_rod_is_thinner_than_body(self):
        v, f = shape_mesh()
        d = dz.exact_vertex_diameters(v, f, shape_sdf)
        on_rod = np.abs(v[:, 0]) > 0.6
        self.assertLess(float(np.median(d[on_rod])), 3 * ROD_RADIUS)
        self.assertGreater(float(np.median(d[np.abs(v[:, 0]) < 0.3])), 0.8)


class ZoneDetectionTest(unittest.TestCase):
    def setUp(self):
        self.v, self.f = shape_mesh()
        self.d = dz.exact_vertex_diameters(self.v, self.f, shape_sdf)
        self.dx = 0.05     # rods (diameter 0.04) are thinner than one voxel

    def test_two_rod_zones_and_no_body_zone(self):
        zones = dz.detect_detail_zones(self.v, self.f, self.d, self.dx, min_area_vox=2.0)
        self.assertEqual(len(zones), 2)
        for zone in zones:
            self.assertGreater(abs(float(zone.center[0])), 0.5)
            self.assertLess(zone.median_diameter, 3 * ROD_RADIUS)
            self.assertGreater(zone.complexity, 0.0)
            inside = np.all((self.v >= zone.lo) & (self.v <= zone.hi), axis=1)
            self.assertFalse(np.any(inside & (np.linalg.norm(self.v, axis=1) < 0.3)))

    def test_fine_grid_needs_no_zone(self):
        zones = dz.detect_detail_zones(self.v, self.f, self.d, dx=0.005, min_area_vox=2.0)
        self.assertEqual(zones, [])

    def test_mirror_pairing(self):
        zones = dz.detect_detail_zones(self.v, self.f, self.d, self.dx, min_area_vox=2.0)
        dz.pair_mirrored_zones(zones, self.v, self.f, shape_sdf, axis=0, plane=0.0, dx=self.dx)
        self.assertEqual(sum(z.mirror_of is None for z in zones), 1)
        mirror = next(z for z in zones if z.mirror_of is not None)
        self.assertEqual(mirror.mirror_axis, 0)
        self.assertNotEqual(mirror.mirror_of, mirror.index)

    def test_asymmetric_geometry_is_not_paired(self):
        zones = dz.detect_detail_zones(self.v, self.f, self.d, self.dx, min_area_vox=2.0)
        shifted = lambda p: shape_sdf(np.asarray(p) + np.array([0.0, 0.08, 0.0]))  # noqa: E731
        dz.pair_mirrored_zones(zones, self.v, self.f, shifted, axis=0, plane=0.0, dx=self.dx)
        self.assertTrue(all(z.mirror_of is None for z in zones))

    def test_mirror_pairs_found_without_axis_hint(self):
        zones = dz.detect_detail_zones(self.v, self.f, self.d, self.dx, min_area_vox=2.0)
        found = dz.find_mirror_pairs(zones, self.v, self.f, shape_sdf, self.dx)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], 0)
        self.assertAlmostEqual(found[1], 0.0, places=5)

    def test_budget_allocation(self):
        zones = dz.detect_detail_zones(self.v, self.f, self.d, self.dx, min_area_vox=2.0)
        plan = dz.allocate_detail_zones(zones, self.v, self.f, self.d, self.dx, 120, 200)
        budgets = [z.reserve for z in zones]
        self.assertTrue(all(b >= 8 for b in budgets))
        self.assertEqual(plan.global_max, 200 - sum(budgets))
        self.assertLessEqual(plan.global_num, plan.global_max)
        heavier = dz.allocate_detail_zones(zones, self.v, self.f, self.d, self.dx, 120, 200, weight=2.0)
        self.assertGreater(heavier.total_max - heavier.global_max, plan.total_max - plan.global_max)
        capped = dz.allocate_detail_zones(zones, self.v, self.f, self.d, self.dx, 10, 40,
                                          weight=100.0, max_share=0.5)
        self.assertLessEqual(sum(z.reserve for z in zones), 20)
        self.assertGreaterEqual(capped.global_max, 10)


class CarveTest(unittest.TestCase):
    def setUp(self):
        self.zone = DummyZone()

    def test_core_becomes_empty_space(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.9, 0.0, 0.0], [3.0, 0.0, 0.0]], np.float32)
        values = np.full(4, -0.2, np.float32)
        carved, changed = dz.carve_values(pts, values, [self.zone])
        np.testing.assert_allclose(carved[:2], [0.5, 0.1], atol=1e-6)   # distance to the core face
        np.testing.assert_allclose(carved[2:], values[2:])
        self.assertEqual(changed.tolist(), [True, True, False, False])

    def test_grid_matches_point_carve(self):
        rng = np.random.default_rng(3)
        grid = rng.normal(size=(6, 7, 8)).astype(np.float32)
        origin, dx = np.array([-1.0, -1.1, -0.9], np.float32), 0.3
        carved = dz.carve_grid(grid, origin, dx, [self.zone])
        iz, iy, ix = np.meshgrid(np.arange(6), np.arange(7), np.arange(8), indexing="ij")
        pts = origin + (np.stack([ix, iy, iz], -1).reshape(-1, 3) + 0.5) * dx
        expected, _ = dz.carve_values(pts, grid.reshape(-1), [self.zone])
        np.testing.assert_allclose(carved.reshape(-1), expected, atol=1e-6)
        self.assertFalse(np.shares_memory(carved, grid))

    def test_seam_band_frees_only_the_rim_inside_the_mesh(self):
        # core box is [-0.5, 0.5]^3; band 0.2 inside each face
        pts = np.array([[0.45, 0, 0], [0.45, 0, 0], [0.0, 0, 0], [0.6, 0, 0]], np.float32)
        values = np.array([-0.1, 0.01, -0.1, -0.1], np.float32)  # inside, just outside, deep, beyond
        mask = dz.seam_band_mask(pts, values, [self.zone], 0.2)
        self.assertEqual(mask.tolist(), [True, False, False, False])
        self.assertFalse(dz.seam_band_mask(pts, values, [self.zone], 0.0).any())

    def test_inside_any_core(self):
        mask = dz.inside_any_core(np.array([[0, 0, 0], [0.6, 0, 0]]), [self.zone])
        self.assertEqual(mask.tolist(), [True, False])

    def test_pack_round_trip(self):
        rng = np.random.default_rng(4)
        for k in (0, 2, 4):
            c, r, q = rng.normal(size=(5, 3)), rng.normal(size=(5, 3)), rng.normal(size=(5, 4))
            shape = rng.normal(size=(5, k)) if k else None
            out = dz.unpack_params(dz.pack_params(c, r, q, shape))
            np.testing.assert_allclose(out[0], c, rtol=1e-6)
            np.testing.assert_allclose(out[2], q, rtol=1e-6)
            if k:
                np.testing.assert_allclose(out[3], shape, rtol=1e-6)
            else:
                self.assertIsNone(out[3])


class TransformTest(unittest.TestCase):
    def test_submesh_and_normalization_match_mesh_io(self):
        v, f = shape_mesh()
        d = dz.exact_vertex_diameters(v, f, shape_sdf)
        zone = dz.detect_detail_zones(v, f, d, 0.05, min_area_vox=2.0)[0]
        sv, sf = dz.zone_submesh(v, f, zone)
        self.assertGreater(len(sf), 0)
        self.assertTrue(np.all((sv >= zone.lo) & (sv <= zone.hi)))
        self.assertLess(int(sf.max()), len(sv))
        center, scale = dz.normalization(sv)
        _, tf = normalize_mesh_capture(trimesh.Trimesh(sv.copy(), sf.copy(), process=False))
        np.testing.assert_allclose(center, tf.center, atol=1e-6)
        self.assertAlmostEqual(scale, tf.scale, places=5)
        n = dz.zone_grid_resolution(sv, zone.median_diameter, voxels_per_feature=8, n_min=64, n_max=512)
        self.assertGreaterEqual(n, 64)
        self.assertEqual(n % 16, 0)

    def test_from_normalized_round_trip_and_bend_units(self):
        rng = np.random.default_rng(1)
        c = rng.normal(size=(5, 3)).astype(np.float32)
        r = rng.uniform(0.1, 0.5, (5, 3)).astype(np.float32)
        q = rng.normal(size=(5, 4)).astype(np.float32)
        shape = np.column_stack([np.ones((5, 2)), rng.normal(size=(5, 2))]).astype(np.float32)
        center, scale = np.array([1.0, 2.0, 3.0], np.float32), 4.0
        wc, wr, wq, ws = dz.from_normalized((c, r, q, shape), center, scale)
        np.testing.assert_allclose(wc, c / 4 + center, rtol=1e-6)
        np.testing.assert_allclose(wr, r / 4, rtol=1e-6)
        np.testing.assert_allclose(ws[:, 2:], shape[:, 2:] * 4, rtol=1e-6)
        np.testing.assert_allclose(ws[:, :2], 1.0)

    def test_mirror_reflects_surface_points(self):
        rng = np.random.default_rng(2)
        for axis in range(3):
            c = rng.normal(size=(3, 3)).astype(np.float32)
            r = rng.uniform(0.1, 0.5, (3, 3)).astype(np.float32)
            q = rng.normal(size=(3, 4)).astype(np.float32)
            shape = np.column_stack([np.ones((3, 2)), rng.normal(size=(3, 2))]).astype(np.float32)
            mc, mr, mq, ms = dz.mirror_params((c, r, q, shape), axis, 0.25)
            reflect = np.eye(3)
            reflect[axis, axis] = -1
            for i in range(3):
                local = rng.normal(size=(20, 3)) * r[i]
                # bend: x += .5 kx z^2, y += .5 ky z^2
                bent = local.copy()
                bent[:, 0] += 0.5 * shape[i, 2] * local[:, 2] ** 2
                bent[:, 1] += 0.5 * shape[i, 3] * local[:, 2] ** 2
                world = bent @ quat_matrix(q[i]).T + c[i]
                expected = world @ reflect.T
                expected[:, axis] += 0.5
                m_local = local @ reflect.T                 # local frame flips on the same axis
                m_bent = m_local.copy()
                m_bent[:, 0] += 0.5 * ms[i, 2] * m_local[:, 2] ** 2
                m_bent[:, 1] += 0.5 * ms[i, 3] * m_local[:, 2] ** 2
                got = m_bent @ quat_matrix(mq[i]).T + mc[i]
                np.testing.assert_allclose(got, expected, atol=1e-5)
            again = dz.mirror_params((mc, mr, mq, ms), axis, 0.25)
            np.testing.assert_allclose(again[0], c, atol=1e-6)
            np.testing.assert_allclose(again[3], shape, atol=1e-6)

    def test_merge_keeps_unowned_and_pads_shapes(self):
        g = (np.zeros((4, 3)), np.ones((4, 3)), np.tile([0, 0, 0, 1.0], (4, 1)), None)
        keep = ~dz.inside_any_core(np.array([[0, 0, 0], [5, 5, 5], [0.1, 0, 0], [9, 9, 9]]), [DummyZone()])
        z = (np.full((2, 3), 7.0), np.ones((2, 3)), np.tile([0, 0, 0, 1.0], (2, 1)),
             np.array([[0.5, 0.6], [0.7, 0.8]], np.float32))
        c, r, q, s = dz.merge_zone_results(g, keep, [z])
        self.assertEqual(len(c), int(keep.sum()) + 2)
        self.assertEqual(s.shape, (len(c), 2))
        np.testing.assert_allclose(s[:int(keep.sum())], 1.0)
        np.testing.assert_allclose(s[-2:], z[3])


class CoverageTest(unittest.TestCase):
    def test_union_inside_shapes(self):
        identity = np.array([[0, 0, 0, 1.0]])
        sphere = (np.zeros((1, 3)), np.ones((1, 3)), identity, np.ones((1, 2)))
        inside = dz.union_inside(np.array([[0.5, 0.5, 0.5], [0.9, 0.9, 0.0]]), sphere)
        self.assertEqual(inside.tolist(), [True, False])
        boxy = (np.zeros((1, 3)), np.ones((1, 3)), identity, np.full((1, 2), 0.1))
        self.assertTrue(dz.union_inside(np.array([[0.9, 0.9, 0.0]]), boxy)[0])
        bent = (np.zeros((1, 3)), np.array([[0.2, 0.2, 1.0]]), identity,
                np.array([[1.0, 1.0, 1.0, 0.0]]))
        # bend x += 0.5 * kx * z^2 moves the tip at z = 0.8 to x = 0.32
        tip = np.array([[0.32, 0.0, 0.8], [0.0, 0.0, 0.8]])
        self.assertEqual(dz.union_inside(tip, bent).tolist(), [True, False])
        capsule = (np.zeros((1, 3)), np.array([[0.2, 0.2, 1.0]]), identity, None)
        got = dz.union_inside(np.array([[0.1, 0, 1.05], [0.3, 0, 0]]), capsule, capsule=True)
        self.assertEqual(got.tolist(), [True, False])

    def test_missing_rod_is_detected(self):
        v, f = shape_mesh()
        identity = np.array([0, 0, 0, 1.0])
        body = ([0, 0, 0], [0.52, 0.52, 0.52])
        rods = [([s * 0.65, 0, 0], [0.24, 0.03, 0.03]) for s in (1.0, -1.0)]

        def params(parts):
            c = np.array([p[0] for p in parts], np.float32)
            r = np.array([p[1] for p in parts], np.float32)
            return c, r, np.tile(identity, (len(parts), 1)), np.ones((len(parts), 2))
        full = dz.uncovered_fraction(v, f, shape_sdf, params([body] + rods), inset=0.01)
        missing = dz.uncovered_fraction(v, f, shape_sdf, params([body, rods[0]]), inset=0.01)
        self.assertLess(full, 0.02)
        # one rod is ~1.5 % of this test surface (a finger is ~8 % of a hand zone)
        self.assertGreater(missing, full + 0.008)


class DummyZone(dz.DetailZone):
    def __init__(self):
        super().__init__(index=0, lo=np.full(3, -1.0), hi=np.full(3, 1.0),
                         core_lo=np.full(3, -0.5), core_hi=np.full(3, 0.5),
                         vertex_indices=np.arange(1), area=1.0, median_diameter=0.1)


class ConnectivityTest(unittest.TestCase):
    def test_components(self):
        labels = dz.connected_components(6, np.array([[0, 1], [1, 2], [4, 5]]))
        self.assertEqual(len(set(labels[[0, 1, 2]])), 1)
        self.assertEqual(len(set(labels[[4, 5]])), 1)
        self.assertNotEqual(labels[0], labels[4])
        self.assertNotIn(labels[3], (labels[0], labels[4]))


if __name__ == "__main__":
    unittest.main()
