"""Shader-built primitives: geometry, instance data and GL rendering.

The geometry tests compare the NumPy replica of the vertex shader with the
CPU mesh builder.  The GL tests need a real OpenGL context; they are skipped on
the offscreen Qt platform (run with QT_QPA_PLATFORM=windows to include them).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtGui, QtWidgets
import pyqtgraph.opengl as gl

import primitive_gpu
from viewer3d import (_UNIT_EDGES, _UNIT_FACES, _UNIT_N_VERTS, _UNIT_VERTS, SceneViewer3D,
                      RENDER_LIGHTING, RENDER_SOLID, RENDER_TRANSPARENT, RENDER_WIREFRAME,
                      _vertex_normals, build_concatenated_mesh, ellipsoid_color)


def _quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]


def _population():
    centers = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 1], [-2, -1, 0.5]], np.float32)
    radii = np.array([[.8, .5, .7], [.6, .6, .9], [.7, .4, .5], [.5, .8, .6]], np.float32)
    rotations = np.array([_quat([0, 0, 1], 0), _quat([1, 1, 0], .7),
                          _quat([0, 1, 0], 1.1), _quat([1, 0, 1], -.4)], np.float32)
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]], np.float32)
    return centers, radii, rotations, colors


def _instances(eps=None, bend=None, radii=None, colors=None):
    centers, base_radii, rotations, base_colors = _population()
    return primitive_gpu.instance_data(
        centers, base_radii if radii is None else radii, rotations,
        base_colors if colors is None else colors,
        default_color=ellipsoid_color(), eps=eps, bend=bend)


class ShaderGeometryTest(unittest.TestCase):
    """The vertex-shader math reproduces the CPU-built meshes."""

    def _check(self, eps=None, bend=None, primitive=None, radii=None):
        centers, base_radii, rotations, colors = _population()
        radii = base_radii if radii is None else radii
        verts, faces, _ = build_concatenated_mesh(
            centers, radii, rotations, colors, sq_eps=eps, sq_bend=bend, primitive=primitive)
        inst = _instances(eps, bend, radii)
        points, normals = primitive_gpu.surface_reference(
            _UNIT_VERTS, inst, capsule=primitive == "capsule")
        # At the two poles the CPU path rounds cos(asin(±1)) to -4e-8, which
        # small exponents blow up to a visible offset; the shader keeps the
        # exact pole.  Everywhere else both agree.
        poles = np.linalg.norm(_UNIT_VERTS[:, :2], axis=1) == 0
        cpu = verts.reshape(len(centers), -1, 3)
        np.testing.assert_allclose(points[:, ~poles], cpu[:, ~poles], atol=2e-4)
        if bend is None:
            from viewer3d import _quat_to_rotation_matrix
            for i in range(len(centers)):
                pole = np.c_[np.zeros((2, 2)), _UNIT_VERTS[poles, 2:] * radii[i, 2]]
                expected = pole @ _quat_to_rotation_matrix(rotations[i]).T + centers[i]
                np.testing.assert_allclose(points[i, poles], expected, atol=1e-5)
        else:
            np.testing.assert_allclose(points[:, poles], cpu[:, poles], atol=5e-3)
        # Analytic normals agree with the face-averaged normals of the CPU mesh.
        reference = _vertex_normals(verts, faces)
        agreement = np.sum(normals.reshape(-1, 3) * reference, axis=1)
        self.assertGreater(np.median(agreement), 0.995)
        self.assertGreater(np.percentile(agreement, 2), 0.9)
        return normals

    def test_ellipsoids(self):
        normals = self._check()
        _, radii, rotations, _ = _population()
        # Exact closed form for ellipsoids: rotate(u / radii).
        from viewer3d import _quat_to_rotation_matrix
        for i in range(len(radii)):
            expected = (_UNIT_VERTS / radii[i]) @ _quat_to_rotation_matrix(rotations[i]).T
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            np.testing.assert_allclose(normals[i], expected, atol=1e-5)

    def test_superquadrics(self):
        self._check(eps=np.array([[1, 1], [.3, .5], [1.6, .8], [.5, 1.8]], np.float32))

    def test_bent_superquadrics(self):
        self._check(eps=np.array([[1, 1], [.3, .5], [1.6, .8], [.5, 1.8]], np.float32),
                    bend=np.array([[0, 0], [.6, 0], [0, -.8], [.4, .4]], np.float32))

    def test_capsules(self):
        _, radii, _, _ = _population()
        capsule_radii = np.c_[radii[:, :1], radii[:, :1], radii[:, 2:]]
        centers, _, rotations, colors = _population()
        verts, _faces, _ = build_concatenated_mesh(
            centers, capsule_radii, rotations, colors, primitive="capsule")
        points, normals = primitive_gpu.surface_reference(
            _UNIT_VERTS, _instances(radii=capsule_radii), capsule=True)
        np.testing.assert_allclose(points.reshape(-1, 3), verts, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-5)

    def test_degenerate_rotation_and_tiny_radii_stay_finite(self):
        inst = _instances()
        inst[0, primitive_gpu.ROTATION] = 0
        inst[1, primitive_gpu.RADII] = 0
        points, normals = primitive_gpu.surface_reference(_UNIT_VERTS, inst)
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(np.isfinite(normals).all())
        np.testing.assert_allclose(points[0], _UNIT_VERTS * inst[0, primitive_gpu.RADII], atol=1e-6)


class InstanceDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_layout_defaults_and_colour_rules(self):
        centers, radii, rotations, colors = _population()
        inst = primitive_gpu.instance_data(centers, radii, rotations, colors[:2],
                                           default_color=np.zeros(4), alpha=.45)
        self.assertEqual(inst.shape, (4, primitive_gpu.INSTANCE_FLOATS))
        self.assertEqual(inst.dtype, np.float32)
        np.testing.assert_allclose(inst[:, primitive_gpu.CENTER], centers)
        np.testing.assert_allclose(inst[:, primitive_gpu.RADII], radii)
        np.testing.assert_allclose(inst[:, primitive_gpu.ROTATION], rotations)
        np.testing.assert_allclose(inst[:, primitive_gpu.SHAPE], np.tile([1, 1, 0, 0], (4, 1)))
        # Missing colour rows repeat the last one, as the CPU builder did.
        np.testing.assert_allclose(inst[:, 14:17], colors[[0, 1, 1, 1], :3])
        np.testing.assert_allclose(inst[:, 17], .45)
        default = primitive_gpu.instance_data(centers, radii, rotations, None,
                                              default_color=[.1, .2, .3, .4])
        np.testing.assert_allclose(default[:, primitive_gpu.COLOR], np.tile([.1, .2, .3, .4], (4, 1)))

    def test_viewer_population_flows_into_one_item(self):
        viewer = SceneViewer3D()
        try:
            centers, radii, rotations, colors = _population()
            eps = np.full((4, 2), .5, np.float32)
            bend = np.full((4, 2), .2, np.float32)
            viewer._ell_render_mode = RENDER_TRANSPARENT
            viewer.show_ellipsoids_fast(centers, radii, rotations, colors, sq_eps=eps,
                                        sq_bend=bend, primitive="capsule")
            item = viewer._ell_item
            self.assertIsInstance(item, primitive_gpu.PrimitiveInstancesItem)
            self.assertTrue(item.capsule)
            self.assertEqual(item.style, primitive_gpu.STYLE_UNLIT)
            np.testing.assert_allclose(item.instances[:, 10:12], eps)
            np.testing.assert_allclose(item.instances[:, 12:14], bend)
            np.testing.assert_allclose(item.instances[:, 17], .45)
            viewer.show_ellipsoids_fast(centers[:2], radii[:2], rotations[:2])
            self.assertIs(viewer._ell_item, item)
            self.assertFalse(item.capsule)
            self.assertEqual(len(item.instances), 2)
            viewer.clear_ellipsoids()
            self.assertIsNone(viewer._ell_item)
            viewer.set_analysis_regions({"under": [((0, 0, 0), .5)], "bridge": [((1, 0, 0), .2)]})
            analysis = viewer._analysis_item
            self.assertIsInstance(analysis, primitive_gpu.PrimitiveInstancesItem)
            np.testing.assert_allclose(analysis.instances[:, primitive_gpu.RADII],
                                       [[.5] * 3, [.2] * 3])
            np.testing.assert_allclose(analysis.instances[:, 17], .22)
        finally:
            viewer.widget.deleteLater()


def _image_array(image: QtGui.QImage) -> np.ndarray:
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine())
    return rows[:, :width * 3].reshape(height, width, 3).copy()


class _GLCase(unittest.TestCase):
    """Shares one real OpenGL view; skipped without a GL context."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.view = gl.GLViewWidget()
        cls.view.resize(320, 240)
        cls.view.setBackgroundColor((0, 0, 0))
        cls.view.setCameraPosition(distance=9.0, elevation=25, azimuth=40)
        cls.view.show()
        for _ in range(5):
            cls.app.processEvents()
        context = cls.view.context()
        if context is None or not context.isValid():
            cls.view.close()
            raise unittest.SkipTest("no OpenGL context on this Qt platform")

    @classmethod
    def tearDownClass(cls):
        cls.view.close()


class GLRenderTest(_GLCase):
    """Draw with real OpenGL: instanced, per-primitive fallback and CPU mesh agree."""

    def _grab(self, *items):
        for item in items:
            self.view.addItem(item)
        try:
            return _image_array(self.view.grabFramebuffer())
        finally:
            for item in items:
                self.view.removeItem(item)

    def _gpu_item(self, inst, style=primitive_gpu.STYLE_UNLIT, fallback=False, capsule=False):
        item = primitive_gpu.PrimitiveInstancesItem(_UNIT_VERTS, _UNIT_FACES, _UNIT_EDGES,
                                                    style=style)
        item.force_fallback = fallback
        item.setGLOptions("opaque")
        item.set_instances(inst, capsule=capsule)
        return item

    def test_instanced_fallback_and_cpu_mesh_render_the_same_pixels(self):
        eps = np.array([[1, 1], [.3, .5], [1.6, .8], [.5, 1.8]], np.float32)
        bend = np.array([[0, 0], [.6, 0], [0, -.8], [.4, .4]], np.float32)
        inst = _instances(eps, bend)
        instanced = self._gpu_item(inst)
        image = self._grab(instanced)
        self.assertEqual(instanced.last_draw, "instanced")
        fallback = self._gpu_item(inst, fallback=True)
        self.assertTrue(np.array_equal(image, self._grab(fallback)))
        self.assertEqual(fallback.last_draw, "per-primitive")
        centers, radii, rotations, colors = _population()
        verts, faces, face_colors = build_concatenated_mesh(
            centers, radii, rotations, colors, sq_eps=eps, sq_bend=bend)
        cpu = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=face_colors,
                            smooth=False, drawEdges=False)
        cpu.setGLOptions("opaque")
        reference = self._grab(cpu)
        self.assertGreater((reference.sum(axis=-1) > 0).mean(), .02)
        differing = np.abs(image.astype(int) - reference).max(axis=-1) > 16
        self.assertLess(differing.mean(), .003)

    def test_every_style_draws_and_leaves_attribute_state_clean(self):
        inst = _instances()
        for style in (primitive_gpu.STYLE_LIT, primitive_gpu.STYLE_UNLIT, primitive_gpu.STYLE_WIRE):
            item = self._gpu_item(inst, style=style)
            image = self._grab(item)
            self.assertGreater((image.sum(axis=-1) > 0).mean(), .005, style)
        # A pyqtgraph mesh with a constant colour drawn after the instances must
        # not pick up instance arrays or divisors left enabled.
        item = self._gpu_item(inst, style=primitive_gpu.STYLE_LIT)
        item.setDepthValue(-1)
        mesh = gl.GLMeshItem(vertexes=_UNIT_VERTS * .3 + [0, 0, 3], faces=_UNIT_FACES,
                             color=(1, 0, 1, 1), smooth=False)
        mesh.setGLOptions("opaque")
        alone = self._grab(mesh)
        instances_only = self._grab(item)
        together = self._grab(item, mesh)
        magenta = ((alone[..., 0] > 200) & (alone[..., 1] < 50) & (alone[..., 2] > 200)
                   & (instances_only.sum(axis=-1) == 0))
        self.assertGreater(magenta.sum(), 50)
        np.testing.assert_array_equal(together[magenta], alone[magenta])

    def test_capsules_render_with_both_draw_paths(self):
        _, radii, _, _ = _population()
        inst = _instances(radii=np.c_[radii[:, :1], radii[:, :1], radii[:, 2:]])
        a = self._grab(self._gpu_item(inst, capsule=True))
        b = self._grab(self._gpu_item(inst, capsule=True, fallback=True))
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(_UNIT_N_VERTS, len(_UNIT_VERTS))


class ViewerModesGLTest(_GLCase):
    """The scene viewer draws every primitive mode through the shader item."""

    def test_viewer_modes(self):
        viewer = SceneViewer3D()
        view = viewer.widget
        view.resize(320, 240)
        view.show()
        try:
            centers, radii, rotations, colors = _population()
            for mode in (RENDER_LIGHTING, RENDER_SOLID, RENDER_TRANSPARENT, RENDER_WIREFRAME):
                viewer._ell_render_mode = mode
                viewer.show_ellipsoids_fast(centers, radii, rotations, colors)
                self.app.processEvents()
                view.grabFramebuffer()
                self.assertEqual(viewer._ell_item.last_draw, "instanced", mode)
        finally:
            view.close()


if __name__ == "__main__":
    unittest.main()
