"""GPU-built primitive populations for the 3-D viewer.

Every displayed primitive (ellipsoid, superquadric, bent superquadric, capsule
or analysis sphere) is one *instance*: 18 floats holding its centre, radii,
rotation quaternion, superquadric exponents, bend and colour.  A unit icosphere
is uploaded once per GL context; the vertex shader turns each icosphere
direction into the instance's surface point and its analytic normal, then
rotates and moves it.  An update therefore only replaces the small instance
buffer (N × 72 bytes) instead of rebuilding and re-uploading N full meshes on
the CPU, and all primitives are drawn in one instanced call.

``surface_reference`` replicates the vertex shader in NumPy for tests.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np
from OpenGL import GL
from PySide6 import QtGui
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

import primitive_material

# Instance layout (float32 columns).
CENTER = slice(0, 3)
RADII = slice(3, 6)
ROTATION = slice(6, 10)      # quaternion x, y, z, w
SHAPE = slice(10, 14)        # exponent e1, exponent e2, bend x, bend y
COLOR = slice(14, 18)        # RGBA
INSTANCE_FLOATS = 18

STYLE_LIT = "lit"            # glossy primitive material
STYLE_UNLIT = "unlit"        # flat colour (solid, transparent, analysis)
STYLE_WIRE = "wire"          # icosphere edges

# Attribute locations, bound before linking.  0 must be the per-vertex
# direction: some drivers only draw when generic attribute 0 is an array.
_ATTRIBUTES = (("a_position", 0, 3), ("i_center", 1, 3), ("i_radii", 2, 3),
               ("i_rotation", 3, 4), ("i_shape", 4, 4), ("i_color", 5, 4))
_INSTANCE_ATTRIBUTES = _ATTRIBUTES[1:]

VERTEX_SHADER = """
uniform mat4 u_mvp;
uniform mat3 u_normal;
uniform float u_capsule;
attribute vec3 a_position;      // unit icosphere direction
attribute vec3 i_center;
attribute vec3 i_radii;
attribute vec4 i_rotation;      // quaternion x, y, z, w
attribute vec4 i_shape;         // exponents e1, e2; bend x, y
attribute vec4 i_color;
varying vec4 v_color;
varying vec3 v_normal;

float spow(float x, float p) {
    float a = abs(x);
    return a > 0.0 ? sign(x) * pow(a, p) : 0.0;
}

vec3 rotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    vec3 d = a_position;
    vec3 p;
    vec3 n;
    if (u_capsule > 0.5) {
        // Radius r.x; the hemispheres move apart by the half-length r.z.
        p = d * i_radii.x;
        p.z += d.z >= 0.0 ? i_radii.z : -i_radii.z;
        n = d;
    } else {
        // Superquadric spherical product of latitude/longitude cos and sin,
        // taken straight from the direction (no trig round trip).
        float e1 = i_shape.x;
        float e2 = i_shape.y;
        float ce = length(d.xy);
        float co = ce > 0.0 ? d.x / ce : 1.0;
        float so = ce > 0.0 ? d.y / ce : 0.0;
        float a = spow(ce, e1);
        p = vec3(a * spow(co, e2), a * spow(so, e2), spow(d.z, e1)) * i_radii;
        // Analytic normal: exponents 2 - e, scaled by the inverse radii.
        float m1 = max(2.0 - e1, 0.0);
        float m2 = max(2.0 - e2, 0.0);
        float b = spow(ce, m1);
        n = vec3(b * spow(co, m2), b * spow(so, m2), spow(d.z, m1))
            / max(abs(i_radii), vec3(1e-9));
        // Bend x, y by k z^2 / 2; the normal follows the inverse transpose.
        float z = p.z;
        p.xy += 0.5 * i_shape.zw * z * z;
        n.z -= z * dot(i_shape.zw, n.xy);
    }
    float ql = length(i_rotation);
    vec4 q = ql > 1e-12 ? i_rotation / ql : vec4(0.0, 0.0, 0.0, 1.0);
    v_normal = normalize(u_normal * rotate(q, n));
    v_color = i_color;
    gl_Position = u_mvp * vec4(rotate(q, p) + i_center, 1.0);
}
"""

UNLIT_FRAGMENT_SHADER = """
#ifdef GL_ES
precision mediump float;
#endif
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"""


# ── instance data ────────────────────────────────────────────────────────────

def instance_data(
    centers: np.ndarray,
    radii: np.ndarray,
    rotations: np.ndarray,
    colors: Optional[np.ndarray],
    *,
    default_color: np.ndarray,
    alpha: Optional[float] = None,
    eps: Optional[np.ndarray] = None,
    bend: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pack N primitives into the (N, 18) float32 instance buffer.

    ``colors`` may be shorter than the population: as before, missing rows
    repeat the last colour.  ``alpha`` overrides every colour's alpha.
    """
    centers = np.asarray(centers, np.float32).reshape(-1, 3)
    n = len(centers)
    out = np.zeros((n, INSTANCE_FLOATS), np.float32)
    out[:, CENTER] = centers
    out[:, RADII] = np.asarray(radii, np.float32).reshape(-1, 3)[:n]
    out[:, ROTATION] = np.asarray(rotations, np.float32).reshape(-1, 4)[:n]
    out[:, 10:12] = 1.0
    if eps is not None:
        out[:, 10:12] = np.asarray(eps, np.float32).reshape(-1, 2)[:n]
    if bend is not None:
        out[:, 12:14] = np.asarray(bend, np.float32).reshape(-1, 2)[:n]
    if colors is not None and len(colors) > 0:
        colors = np.asarray(colors, np.float32).reshape(-1, 4)
        out[:, COLOR] = colors[np.minimum(np.arange(n), len(colors) - 1)]
    else:
        out[:, COLOR] = np.asarray(default_color, np.float32).reshape(4)
    if alpha is not None:
        out[:, 17] = alpha
    return out


def surface_reference(directions: np.ndarray, instances: np.ndarray,
                      capsule: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """NumPy replica of the vertex shader → world points and unit normals (N, V, 3)."""
    d = np.asarray(directions, np.float32)[None]
    inst = np.asarray(instances, np.float32)
    radii = inst[:, None, RADII]
    if capsule:
        p = d * radii[..., :1]
        p[..., 2] += np.where(d[..., 2] >= 0, radii[..., 2], -radii[..., 2])
        n = np.broadcast_to(d, p.shape).copy()
    else:
        def spow(x, power):
            a = np.abs(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(a > 0, np.sign(x) * a ** power, 0.0)

        e1, e2 = inst[:, None, 10], inst[:, None, 11]
        ce = np.linalg.norm(d[..., :2], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            co = np.where(ce > 0, d[..., 0] / ce, 1.0)
            so = np.where(ce > 0, d[..., 1] / ce, 0.0)
        a = spow(ce, e1)
        p = np.stack([a * spow(co, e2), a * spow(so, e2), spow(d[..., 2], e1)], -1) * radii
        m1, m2 = np.maximum(2 - e1, 0), np.maximum(2 - e2, 0)
        b = spow(ce, m1)
        n = (np.stack([b * spow(co, m2), b * spow(so, m2), spow(d[..., 2], m1)], -1)
             / np.maximum(np.abs(radii), 1e-9))
        bend = inst[:, None, 12:14]
        z = p[..., 2:3]
        p[..., :2] += 0.5 * bend * z * z
        n[..., 2] -= z[..., 0] * np.sum(bend * n[..., :2], axis=-1)
    q = inst[:, ROTATION].astype(np.float64)
    length = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(length > 1e-12, q / np.maximum(length, 1e-30), [0, 0, 0, 1])[:, None]

    def rotate(v):
        u, w = q[..., :3], q[..., 3:]
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    normals = rotate(n)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-30)
    points = rotate(p) + inst[:, None, CENTER]
    return points.astype(np.float32), normals.astype(np.float32)


# ── GL item ──────────────────────────────────────────────────────────────────

class _GLResources:
    """Programs and buffers of one GL context."""

    def __init__(self):
        self.programs: dict[str, int] = {}
        self.template = self.faces = self.edges = self.instances = None
        self.instance_bytes = 0


def _compile(stage: int, source: str, es2: bool) -> int:
    shader = GL.glCreateShader(stage)
    sources = [source] if not es2 else ["#version 100\n", source]
    GL.glShaderSource(shader, sources)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        GL.glDeleteShader(shader)
        raise RuntimeError(f"primitive shader compile failed: {log!r}")
    return shader


def _link(fragment: str, es2: bool) -> int:
    program = GL.glCreateProgram()
    shaders = (_compile(GL.GL_VERTEX_SHADER, VERTEX_SHADER, es2),
               _compile(GL.GL_FRAGMENT_SHADER, fragment, es2))
    for shader in shaders:
        GL.glAttachShader(program, shader)
    for name, location, _size in _ATTRIBUTES:
        GL.glBindAttribLocation(program, location, name)
    GL.glLinkProgram(program)
    for shader in shaders:
        GL.glDetachShader(program, shader)
        GL.glDeleteShader(shader)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program)
        GL.glDeleteProgram(program)
        raise RuntimeError(f"primitive shader link failed: {log!r}")
    return program


def _upload(target: int, handle: Optional[int], array: np.ndarray, usage: int) -> int:
    if handle is None:
        handle = int(GL.glGenBuffers(1))
    GL.glBindBuffer(target, handle)
    GL.glBufferData(target, array.nbytes, array, usage)
    return handle


class PrimitiveInstancesItem(GLGraphicsItem):
    """Draws a primitive population whose surfaces are built in the vertex shader.

    ``directions`` (V, 3) unit icosphere vertices, ``faces`` (F, 3) and
    ``edges`` (E, 2) are uploaded once per GL context.  ``set_instances``
    replaces the population; ``set_style`` switches lit, unlit and wireframe.
    """

    def __init__(self, directions: np.ndarray, faces: np.ndarray, edges: np.ndarray,
                 *, style: str = STYLE_LIT, parentItem=None):
        super().__init__(parentItem=parentItem)
        self.directions = np.ascontiguousarray(directions, np.float32)
        self.faces = np.ascontiguousarray(faces, np.uint32).reshape(-1)
        self.edges = np.ascontiguousarray(edges, np.uint32).reshape(-1)
        self.instances = np.zeros((0, INSTANCE_FLOATS), np.float32)
        self.capsule = False
        self.style = style
        self.force_fallback = False      # tests: draw without instancing
        self.last_draw: Optional[str] = None
        self._resources: Optional[_GLResources] = None
        self._context: Optional[QtGui.QOpenGLContext] = None
        self._dirty = True

    # ── data ──
    def set_instances(self, instances: np.ndarray, *, capsule: bool = False) -> None:
        self.instances = np.ascontiguousarray(instances, np.float32).reshape(-1, INSTANCE_FLOATS)
        self.capsule = bool(capsule)
        self._dirty = True
        self.update()

    def set_style(self, style: str) -> None:
        if style not in (STYLE_LIT, STYLE_UNLIT, STYLE_WIRE):
            raise ValueError(f"unknown primitive style {style!r}")
        if style != self.style:
            self.style = style
            self.update()

    # ── GL ──
    def _drop_resources(self) -> None:
        # The context (and with it every handle) is going away.
        self._resources = None
        self._context = None
        self._dirty = True

    def _ensure_resources(self) -> _GLResources:
        context = QtGui.QOpenGLContext.currentContext()
        if self._resources is not None and context is self._context:
            return self._resources
        self._resources = res = _GLResources()
        self._context = context
        self._dirty = True
        context.aboutToBeDestroyed.connect(self._drop_resources)
        es2 = context.hasExtension(b"GL_ARB_ES2_compatibility")
        res.programs = {
            STYLE_LIT: _link(primitive_material.FRAGMENT_SHADER, es2),
            STYLE_UNLIT: _link(UNLIT_FRAGMENT_SHADER, es2),
        }
        res.programs[STYLE_WIRE] = res.programs[STYLE_UNLIT]
        res.template = _upload(GL.GL_ARRAY_BUFFER, None, self.directions, GL.GL_STATIC_DRAW)
        res.faces = _upload(GL.GL_ELEMENT_ARRAY_BUFFER, None, self.faces, GL.GL_STATIC_DRAW)
        res.edges = _upload(GL.GL_ELEMENT_ARRAY_BUFFER, None, self.edges, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        return res

    def paint(self):
        count = len(self.instances)
        if count == 0:
            return
        self.setupGLState()
        res = self._ensure_resources()
        if self._dirty:
            res.instances = _upload(GL.GL_ARRAY_BUFFER, res.instances, self.instances,
                                    GL.GL_DYNAMIC_DRAW)
            self._dirty = False
        program = res.programs[self.style]
        wire = self.style == STYLE_WIRE
        index_buffer, index_count = ((res.edges, self.edges.size) if wire
                                     else (res.faces, self.faces.size))
        primitive = GL.GL_LINES if wire else GL.GL_TRIANGLES
        instanced = (not self.force_fallback and bool(GL.glVertexAttribDivisor)
                     and bool(GL.glDrawElementsInstanced))

        GL.glUseProgram(program)
        try:
            mvp = np.array(self.mvpMatrix().data(), np.float32)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, "u_mvp"), 1, False, mvp)
            if (location := GL.glGetUniformLocation(program, "u_normal")) != -1:
                normal = np.array(self.modelViewMatrix().normalMatrix().data(), np.float32)
                GL.glUniformMatrix3fv(location, 1, False, normal)
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_capsule"),
                           1.0 if self.capsule else 0.0)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, res.template)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            if wire:
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
                GL.glLineWidth(1.0)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer)
            if instanced:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, res.instances)
                stride = INSTANCE_FLOATS * 4
                offset = 0
                for _name, location, size in _INSTANCE_ATTRIBUTES:
                    GL.glEnableVertexAttribArray(location)
                    GL.glVertexAttribPointer(location, size, GL.GL_FLOAT, GL.GL_FALSE,
                                             stride, ctypes.c_void_p(offset))
                    GL.glVertexAttribDivisor(location, 1)
                    offset += size * 4
                GL.glDrawElementsInstanced(primitive, index_count, GL.GL_UNSIGNED_INT,
                                           None, count)
                for _name, location, _size in _INSTANCE_ATTRIBUTES:
                    GL.glVertexAttribDivisor(location, 0)
                    GL.glDisableVertexAttribArray(location)
                self.last_draw = "instanced"
            else:
                # Without instancing: one draw per primitive with constant
                # attributes.  The surface is still built by the shader.
                for row in self.instances:
                    GL.glVertexAttrib3f(1, *row[CENTER])
                    GL.glVertexAttrib3f(2, *row[RADII])
                    GL.glVertexAttrib4f(3, *row[ROTATION])
                    GL.glVertexAttrib4f(4, *row[SHAPE])
                    GL.glVertexAttrib4f(5, *row[COLOR])
                    GL.glDrawElements(primitive, index_count, GL.GL_UNSIGNED_INT, None)
                self.last_draw = "per-primitive"
        finally:
            GL.glDisableVertexAttribArray(0)
            if wire:
                GL.glDisable(GL.GL_LINE_SMOOTH)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)
