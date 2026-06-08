"""
viewer3d.py — unified 3-D scene viewport.

Hierarchy:
  _BaseViewer       – shared setup (GL widget, axis)
  ViewportOverlay   – floating in-viewport menu: per-element display toggles
                      + render-mode selector (wireframe / transparent / lighting)
  SceneViewer3D     – one GL scene holding the mesh, skeleton bones and the
                      fitted ellipsoids together.  Ellipsoids are drawn as a
                      single concatenated GLMeshItem built from a precomputed
                      unit icosphere (scale → rotate → translate on CPU).

Render modes are applied uniformly to the mesh and ellipsoids; visibility of
each element is toggled independently via the overlay.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

import theme
import raymarch
import sdf_slice as slice_module
from widgets import DropGLView


# ── Ellipsoid colour ──────────────────────────────────────────────────────────

# Single uniform colour for every ellipsoid (the brand yellow, to contrast with
# the blue mesh).  Read from ``theme`` at call time so the colour pickers update
# it live — see ``ellipsoid_color()``.
_ELLIPSOID_ALPHA = 179 / 255.0


def ellipsoid_color() -> np.ndarray:
    """Current ellipsoid RGBA (brand secondary colour from ``theme``)."""
    return theme.rgba_array(theme.YELLOW, _ELLIPSOID_ALPHA)


# ── Precomputed unit icosphere ────────────────────────────────────────────────

def _make_unit_icosphere(subdivisions: int = 3):
    """Create a unit icosphere once.  Returns (verts, faces) as float32/int32."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    verts = np.ascontiguousarray(sphere.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(sphere.faces, dtype=np.int32)
    return verts, faces

_UNIT_VERTS, _UNIT_FACES = _make_unit_icosphere(3)
_UNIT_N_VERTS = _UNIT_VERTS.shape[0]
_UNIT_N_FACES = _UNIT_FACES.shape[0]


def superquadric_unit_verts(e1: float, e2: float) -> np.ndarray:
    """Deform the unit icosphere into a unit superquadric (radii = 1).

    Maps each sphere-direction (η, ω) through the superquadric spherical product
    with the signed-power function.  Same vertex/face topology as the icosphere,
    so it slots straight into ``build_concatenated_mesh`` (scale → rotate → move).
    """
    v = _UNIT_VERTS
    eta = np.arcsin(np.clip(v[:, 2], -1.0, 1.0))      # latitude
    omega = np.arctan2(v[:, 1], v[:, 0])              # longitude

    def _sp(x, p):                                    # signed power
        return np.sign(x) * (np.abs(x) ** p)

    ce, se = np.cos(eta), np.sin(eta)
    co, so = np.cos(omega), np.sin(omega)
    x = _sp(ce, e1) * _sp(co, e2)
    y = _sp(ce, e1) * _sp(so, e2)
    z = _sp(se, e1)
    return np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)


def capsule_local_verts(radius: float, half_length: float) -> np.ndarray:
    """Capsule surface (radius + cylinder half-length) in the primitive's local
    frame, axis = z.  Splits the unit icosphere into two hemispheres and pulls
    them ±half_length apart, with the cylinder spanned between.  Radius is baked
    in here — the caller must NOT scale by the radii again.
    """
    v = (_UNIT_VERTS * float(radius)).astype(np.float32)
    h = float(half_length)
    top = v[:, 2] >= 0.0
    v[top, 2] += h
    v[~top, 2] -= h
    return np.ascontiguousarray(v, dtype=np.float32)


def _quat_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert (x,y,z,w) quaternion to a 3×3 rotation matrix (float32)."""
    x, y, z, w = quat_xyzw.astype(np.float64)
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    x /= n; y /= n; z /= n; w /= n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
        [    2*(xy+wz), 1 - 2*(xx+zz),     2*(yz-wx)],
        [    2*(xz-wy),     2*(yz+wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)


def build_concatenated_mesh(
    centers: np.ndarray,
    radii: np.ndarray,
    rotations: np.ndarray,
    colors: Optional[np.ndarray] = None,
    sq_eps: Optional[np.ndarray] = None,
    sq_bend: Optional[np.ndarray] = None,
    primitive: Optional[str] = None,
) -> tuple:
    """Build a single concatenated mesh from N primitive parameters.

    ``sq_eps`` is an optional (N,2) array of per-primitive superquadric
    roundness exponents; when given, each primitive's unit sphere is deformed
    into its own superquadric (cached per distinct eps pair).
    ``primitive == "capsule"`` instead builds a capsule per primitive
    (radius = r[0], half-length = r[2]).  Returns
    (all_verts, all_faces, all_face_colors).
    """
    n_ell = centers.shape[0]
    if n_ell == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.int32),
                np.empty((0, 4), dtype=np.float32))

    V = _UNIT_N_VERTS
    F = _UNIT_N_FACES

    # Per-primitive base unit mesh (sphere, or a superquadric warp cached by eps).
    _sq_cache: dict = {}

    def _base_verts(i):
        if sq_eps is None:
            return _UNIT_VERTS
        key = (round(float(sq_eps[i][0]), 3), round(float(sq_eps[i][1]), 3))
        bv = _sq_cache.get(key)
        if bv is None:
            bv = superquadric_unit_verts(key[0], key[1])
            _sq_cache[key] = bv
        return bv

    all_verts = np.empty((n_ell * V, 3), dtype=np.float32)
    all_faces = np.empty((n_ell * F, 3), dtype=np.int32)
    all_face_colors = np.empty((n_ell * F, 4), dtype=np.float32)
    ell_color = ellipsoid_color()        # current brand secondary (live)

    is_capsule = (primitive == "capsule")

    for i in range(n_ell):
        if is_capsule:
            # radius baked into the capsule mesh → do NOT scale by radii again.
            v = capsule_local_verts(radii[i][0], radii[i][2])
        else:
            v = _base_verts(i) * radii[i]
            if sq_bend is not None:
                kx = float(sq_bend[i][0]); ky = float(sq_bend[i][1])
                if kx != 0.0 or ky != 0.0:
                    z2 = v[:, 2] * v[:, 2]
                    v = v.copy()
                    v[:, 0] += 0.5 * kx * z2       # forward bend (x, y ∝ z²)
                    v[:, 1] += 0.5 * ky * z2
        R = _quat_to_rotation_matrix(rotations[i])
        v = v @ R.T
        v += centers[i]

        vs = i * V
        fs = i * F
        all_verts[vs:vs + V] = v
        all_faces[fs:fs + F] = _UNIT_FACES + vs

        # All ellipsoids share one uniform colour.  An explicitly supplied
        # colour (e.g. rig/bone tinting) still wins; otherwise everything uses
        # the current brand secondary colour rather than cycling a palette.
        if colors is not None and len(colors) > 0:
            c = colors[0]
        else:
            c = ell_color
        all_face_colors[fs:fs + F] = c

    return all_verts, all_faces, all_face_colors


def ellipsoid_vertex_normals(radii: np.ndarray,
                             rotations: np.ndarray) -> np.ndarray:
    """Analytic per-vertex surface normals for N ellipsoids → ``(N·V, 3)``.

    For a scaled+rotated unit sphere the surface normal at icosphere vertex ``u``
    (which is also the unit-sphere normal) is ``normalize(R · (u / radii))`` — the
    inverse-transpose of the scaling.  This is far cheaper than pyqtgraph's
    per-vertex Python loop (it recomputes face + vertex normals on every update),
    which is what makes the lighting render mode crawl during live fitting.
    """
    n_ell = len(radii)
    V = _UNIT_N_VERTS
    out = np.empty((n_ell * V, 3), dtype=np.float32)
    u = _UNIT_VERTS                                   # (V,3) unit normals
    for i in range(n_ell):
        R = _quat_to_rotation_matrix(rotations[i])
        inv_r = 1.0 / np.maximum(np.abs(radii[i]), 1e-9)
        n = (u * inv_r) @ R.T                         # (V,3)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        out[i * V:(i + 1) * V] = n
    return out


# ── Thickness colour mapping ────────────────────────────────────────────────--

def _colormap_jet(t: np.ndarray) -> np.ndarray:
    """Map t∈[0,1] → (N,3) RGB via a blue→cyan→green→yellow→red ramp."""
    t = np.clip(t, 0.0, 1.0)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([
        [0.23, 0.30, 0.75],   # thin  → blue
        [0.00, 0.75, 0.85],   # cyan
        [0.25, 0.78, 0.30],   # green
        [0.95, 0.85, 0.15],   # yellow
        [0.85, 0.15, 0.10],   # thick → red
    ])
    r = np.interp(t, stops, cols[:, 0])
    g = np.interp(t, stops, cols[:, 1])
    b = np.interp(t, stops, cols[:, 2])
    return np.stack([r, g, b], axis=1).astype(np.float32)


def _dilate_zeros(field: np.ndarray, iters: int = 3) -> np.ndarray:
    """Grow non-zero values into zero voxels (6-neighbour max), ``iters`` times.

    Surface vertices map to voxels just outside the mesh where the interior
    thickness field is 0; dilating fills those with the adjacent interior
    thickness so the surface gets coloured instead of reading 0.
    """
    f = field.astype(np.float32, copy=True)
    for _ in range(iters):
        zero = f == 0.0
        if not zero.any():
            break
        m = f.copy()
        m[:-1] = np.maximum(m[:-1], f[1:])
        m[1:] = np.maximum(m[1:], f[:-1])
        m[:, :-1] = np.maximum(m[:, :-1], f[:, 1:])
        m[:, 1:] = np.maximum(m[:, 1:], f[:, :-1])
        m[:, :, :-1] = np.maximum(m[:, :, :-1], f[:, :, 1:])
        m[:, :, 1:] = np.maximum(m[:, :, 1:], f[:, :, :-1])
        f = np.where(zero, m, f)
    return f


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (orientation as given by ``faces``)."""
    v = verts.astype(np.float32)
    f = faces.astype(np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.maximum(norm, 1e-12)).astype(np.float32)


def thickness_vertex_colors(
    verts: np.ndarray,
    faces: np.ndarray,
    thickness_grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    n: int,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Per-vertex RGBA colours from the local-thickness field.

    Returns (colors (V,4) float32, vmin, vmax) where vmin/vmax are the colour
    range actually used (5th/95th percentile of the sampled thickness).

    ``local_thickness`` fills the *interior* of each feature with the full
    inscribed-sphere diameter, but surface vertices sit on the zero level set
    where the voxel value is ~0.  Sampling the voxel directly therefore reads
    the surface as thin everywhere.  Instead we march a few voxels *along the
    inward vertex normal* and take the max thickness encountered, which pulls
    each feature's true diameter onto its surface (torso → warm, fingers → cool).
    """
    tg = _dilate_zeros(thickness_grid, iters=2)
    verts = verts.astype(np.float32)
    normals = _vertex_normals(verts, faces)

    # Per-axis bounds from the (possibly anisotropic) grid — ``n`` is kept only
    # for signature compatibility and may not describe every axis.
    g_nz, g_ny, g_nx = tg.shape
    hi = np.array([g_nx - 1, g_ny - 1, g_nz - 1])

    def _sample(points: np.ndarray) -> np.ndarray:
        q = (points - origin.astype(np.float32)) / float(dx)
        idx = np.clip(np.floor(q).astype(np.int64), 0, hi)
        return tg[idx[:, 2], idx[:, 1], idx[:, 0]]   # grid is (nz, ny, nx)

    # March inward (and a little outward) along the normal, keep the max.
    th = _sample(verts)
    steps = 8
    for j in range(1, steps + 1):
        off = normals * (float(j) * float(dx))
        th = np.maximum(th, _sample(verts - off))    # inward (true interior side)
        if j <= 2:
            th = np.maximum(th, _sample(verts + off))  # cover flipped-normal faces

    pos = th[th > 0.0]
    if vmin is None:
        vmin = float(np.percentile(pos, 5)) if pos.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(pos, 95)) if pos.size else 1.0
    t = (th - vmin) / max(vmax - vmin, 1e-6)
    rgb = _colormap_jet(t)
    alpha = np.ones((len(verts), 1), dtype=np.float32)
    colors = np.concatenate([rgb, alpha], axis=1).astype(np.float32)
    return colors, float(vmin), float(vmax)


# ── Base viewer ───────────────────────────────────────────────────────────────

class _BaseViewer:
    """Common setup: GL widget with grid and axis."""

    def __init__(self):
        self._view = DropGLView()
        # White viewport in light mode, black in dark mode (see theme.bg).
        self._view.setBackgroundColor(theme.bg((0, 0, 0)))
        self._view.setCameraPosition(distance=3.0, elevation=15, azimuth=45)

        # grid = gl.GLGridItem()
        # grid.scale(1, 1, 1)
        # self._view.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self._view.addItem(axis)

    @property
    def widget(self) -> DropGLView:
        return self._view


# ── Unified scene viewer (mesh + skeleton + ellipsoids in one viewport) ────────

# Render modes
RENDER_WIREFRAME = "wireframe"
RENDER_TRANSPARENT = "transparent"
RENDER_SOLID = "solid"             # opaque flat-shaded faces (no blend, no light)
RENDER_LIGHTING = "lighting"
RENDER_THICKNESS = "thickness"     # mesh-only: local-thickness heatmap
RENDER_RAYMARCH = "raymarch"       # ellipsoids-only: GPU sphere-traced SDF

# Base solid colour for the mesh (brand primary).  Read from ``theme`` at call
# time so the colour pickers update it live.

# Name of our custom multi-light shader (registered lazily on first use).
_BRIGHT_SHADER = "brightMultiLight"


def _ensure_bright_shader() -> str:
    """Register (once) and return a brighter, multi-light shaded program.

    pyqtgraph's built-in ``shaded`` shader uses a single dim directional light
    (ambient 0.2, one light × 0.8).  This adds four directional lights (key /
    fill / rim / back) with a high ambient term and two-sided lighting (``abs``
    of the normal dot) so the lit model is noticeably brighter and never has
    fully black back-faces.
    """
    from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader
    if _BRIGHT_SHADER in ShaderProgram.names:
        return _BRIGHT_SHADER
    # pyqtgraph 0.14 uses a modern GL pipeline: u_mvp / u_normal uniforms and
    # a_position / a_normal / a_color attributes (camera-relative normals via
    # u_normal), mirroring the built-in 'shaded' program.
    ShaderProgram(_BRIGHT_SHADER, [
        VertexShader("""
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            attribute vec4 a_position;
            attribute vec3 a_normal;
            attribute vec4 a_color;
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                v_normal = normalize(u_normal * a_normal);
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
        """),
        FragmentShader("""
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                vec3 n = normalize(v_normal);
                vec3 V = vec3(0.0, 0.0, 1.0);                  // view dir (view space)
                // Orient the normal toward the viewer so highlights are stable
                // on both sides of the two-sided surface.
                vec3 nf = (dot(n, V) < 0.0) ? -n : n;
                vec3 l0 = normalize(vec3( 1.0, -1.0, -1.0));   // key
                vec3 l1 = normalize(vec3(-1.0, -0.3,  0.6));   // fill
                vec3 l2 = normalize(vec3( 0.2,  1.0,  0.5));   // rim
                vec3 l3 = normalize(vec3( 0.0,  0.0,  1.0));   // back

                // Diffuse (two-sided).  Dialled down a notch from the previous
                // (brighter) values for a softer, less blown-out look.
                float diff = 0.0;
                diff += abs(dot(n, l0)) * 0.55;
                diff += abs(dot(n, l1)) * 0.38;
                diff += abs(dot(n, l2)) * 0.34;
                diff += abs(dot(n, l3)) * 0.22;
                float ambient = 0.32;
                vec3 rgb = v_color.rgb * (ambient + diff);

                // Glossy white specular (Blinn-Phong) — gentler highlights.
                float specStr = 0.55;
                float spec = 0.0;
                spec += pow(max(dot(nf, normalize(l0 + V)), 0.0), 64.0);        // key
                spec += pow(max(dot(nf, normalize(l2 + V)), 0.0), 64.0) * 0.5;  // rim
                spec += pow(max(dot(nf, normalize(l1 + V)), 0.0), 18.0) * 0.25; // soft sheen
                rgb += specStr * spec * vec3(1.0);

                gl_FragColor = vec4(min(rgb, vec3(1.0)), v_color.a);
            }
        """),
    ])
    return _BRIGHT_SHADER


def _ell_alpha_for_mode(mode: str) -> float:
    """Face-colour alpha used for ellipsoids in each render mode (matches
    :func:`_build_mesh_item`).  Wireframe draws no faces, so the value is moot."""
    if mode == RENDER_TRANSPARENT:
        return 0.45
    return 1.0


# Edge alpha for the wireframe mode — slightly translucent so the wireframe is
# less visually dense than fully opaque lines.
_WIRE_ALPHA = 0.4


def _gl_opts(mode: str, cull: bool = False):
    """GL state dict for a given render mode."""
    from OpenGL import GL as _GL
    if mode in (RENDER_TRANSPARENT, RENDER_WIREFRAME):
        # Both need alpha blending: transparent faces, and translucent wire lines.
        opts = {
            _GL.GL_DEPTH_TEST: True,
            _GL.GL_BLEND: True,
            'glBlendFuncSeparate': (
                _GL.GL_SRC_ALPHA, _GL.GL_ONE_MINUS_SRC_ALPHA,
                _GL.GL_ONE, _GL.GL_ONE_MINUS_SRC_ALPHA,
            ),
        }
        # Cull only matters for solid faces (transparent mode).
        if cull and mode == RENDER_TRANSPARENT:
            opts[_GL.GL_CULL_FACE] = True
        return opts
    # Solid / lighting → opaque
    opts = {
        _GL.GL_DEPTH_TEST: True,
        _GL.GL_BLEND: False,
    }
    if cull:
        opts[_GL.GL_CULL_FACE] = True
    return opts


def _build_mesh_item(
    verts: np.ndarray,
    faces: np.ndarray,
    mode: str,
    *,
    vertex_colors: Optional[np.ndarray] = None,
    face_colors: Optional[np.ndarray] = None,
    base_rgb: Optional[tuple] = None,
    cull: bool = False,
) -> gl.GLMeshItem:
    """Create a GLMeshItem configured for the requested render mode.

    ``vertex_colors`` / ``face_colors`` (RGBA float arrays) win over ``base_rgb``
    when supplied; their alpha is rescaled per mode (translucent for transparent,
    opaque otherwise).  ``base_rgb`` defaults to the current brand primary.
    The wireframe edge colour is derived from the object's
    own colour so each object keeps the *same* colour across all render modes.
    """
    if base_rgb is None:
        base_rgb = theme.BLUE01
    r, g, b = base_rgb

    def _with_alpha(cols: np.ndarray, alpha: float) -> np.ndarray:
        out = cols.astype(np.float32, copy=True)
        out[:, 3] = alpha
        return out

    if mode == RENDER_WIREFRAME:
        # Edge colour = object colour (translucent), so wireframe matches the
        # transparent/lit colour instead of a separate accent hue.
        if face_colors is not None and len(face_colors) > 0:
            ec = (float(face_colors[0][0]), float(face_colors[0][1]),
                  float(face_colors[0][2]), _WIRE_ALPHA)
        else:
            ec = (r, g, b, _WIRE_ALPHA)
        item = gl.GLMeshItem(
            vertexes=verts, faces=faces,
            drawFaces=False, drawEdges=True,
            edgeColor=ec, smooth=False,
        )
    elif mode == RENDER_LIGHTING:
        kw = dict(vertexes=verts, faces=faces,
                  drawFaces=True, drawEdges=False, smooth=True,
                  shader=_ensure_bright_shader())
        if vertex_colors is not None:
            kw['vertexColors'] = _with_alpha(vertex_colors, 1.0)
        elif face_colors is not None:
            kw['faceColors'] = _with_alpha(face_colors, 1.0)
        else:
            kw['color'] = (r, g, b, 1.0)
        item = gl.GLMeshItem(**kw)
    elif mode == RENDER_SOLID:
        # Opaque flat-shaded surface: full alpha, no blending, no lighting shader
        # (the object's true colour, fully opaque).
        kw = dict(vertexes=verts, faces=faces,
                  drawFaces=True, drawEdges=False, smooth=False)
        if vertex_colors is not None:
            kw['vertexColors'] = _with_alpha(vertex_colors, 1.0)
        elif face_colors is not None:
            kw['faceColors'] = _with_alpha(face_colors, 1.0)
        else:
            kw['color'] = (r, g, b, 1.0)
        item = gl.GLMeshItem(**kw)
    elif mode == RENDER_THICKNESS:
        # Local-thickness heatmap: flat opaque surface, vertex colours shown
        # as-is (no lighting modulation so the colour ramp reads true).  Falls
        # back to a solid surface if the thickness field isn't computed yet.
        kw = dict(vertexes=verts, faces=faces,
                  drawFaces=True, drawEdges=False, smooth=True)
        if vertex_colors is not None:
            kw['vertexColors'] = _with_alpha(vertex_colors, 1.0)
        else:
            kw['color'] = (r, g, b, 1.0)
        item = gl.GLMeshItem(**kw)
    else:  # RENDER_TRANSPARENT
        kw = dict(vertexes=verts, faces=faces,
                  drawFaces=True, drawEdges=False, smooth=False)
        if vertex_colors is not None:
            kw['vertexColors'] = _with_alpha(vertex_colors, 0.5)
        elif face_colors is not None:
            kw['faceColors'] = _with_alpha(face_colors, 0.45)
        else:
            kw['color'] = (r, g, b, 0.30)
        item = gl.GLMeshItem(**kw)

    item.setGLOptions(_gl_opts(mode, cull=cull))
    return item


class ViewportOverlay(QtWidgets.QFrame):
    """Floating in-viewport menu.

    Mesh and ellipsoids each get a visibility checkbox and their own render-mode
    dropdown.  The mesh dropdown additionally offers a "Dicke" (thickness
    heatmap) mode.  The skeleton has a checkbox only, shown only when a skeleton
    is present (see :meth:`set_skeleton_available`).

    Signals
    -------
    visibilityChanged(str, bool)   — (key, on) for keys: mesh, skeleton, ellipsoids
    renderModeChanged(str, str)    — (target, mode); target ∈ {mesh, ellipsoids},
                                     mode is one of the RENDER_* constants
    """

    visibilityChanged = QtCore.Signal(str, bool)
    renderModeChanged = QtCore.Signal(str, str)
    # SDF slice plane (movable texture pushed through the volume)
    sliceToggled = QtCore.Signal(bool)         # on / off
    slicePlaneChanged = QtCore.Signal(str)     # "XY" | "XZ" | "YZ"
    sliceSourceChanged = QtCore.Signal(str)    # "mesh" | "ellipsoids" | "difference"
    # Raymarch smooth-union amount (fraction of ellipsoid size, 0 = hard union).
    raymarchBlendChanged = QtCore.Signal(float)

    # Render-mode dropdown items per target.  The mesh additionally offers the
    # thickness heatmap as a mode.
    _MODE_ITEMS = {
        "mesh": (
            ("Wireframe", RENDER_WIREFRAME),
            ("Transparent", RENDER_TRANSPARENT),
            ("Solid", RENDER_SOLID),
            ("Lighting", RENDER_LIGHTING),
            ("Thickness", RENDER_THICKNESS),
        ),
        "ellipsoids": (
            ("Wireframe", RENDER_WIREFRAME),
            ("Transparent", RENDER_TRANSPARENT),
            ("Solid", RENDER_SOLID),
            ("Lighting", RENDER_LIGHTING),
            ("Raymarch", RENDER_RAYMARCH),
        ),
    }

    # Default render mode per object — the character (mesh) starts as wireframe
    # so the fitted ellipsoids are visible through it.
    _DEFAULT_MODE = {
        "mesh": RENDER_WIREFRAME,
        "ellipsoids": RENDER_TRANSPARENT,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.apply_theme()
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(10, 8, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)

        grid.addWidget(QtWidgets.QLabel("Show"), 0, 0, 1, 2)

        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        self._combos: dict[str, QtWidgets.QComboBox] = {}

        # (key, label, has_render_combo)
        rows = (
            ("mesh", "Mesh", True),
            ("skeleton", "Skeleton", False),
            ("ellipsoids", "Ellipsoids", True),
            ("operations", "Operations", False),
            ("analysis", "Analysis", False),
        )
        for i, (key, label, has_combo) in enumerate(rows, start=1):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, k=key: self.visibilityChanged.emit(k, on))
            grid.addWidget(cb, i, 0)
            self._checks[key] = cb
            if has_combo:
                grid.addWidget(self._make_combo(key), i, 1)

        # SuperFit operation gizmos: colour-coded boxes marking where each
        # merge / split / spawn / fuse / delete happened (fade out over ~50 steps).
        self._checks["operations"].setToolTip(
            "Mark where SuperFit acted (fades over 50 steps):\n"
            "  green = spawn   orange = split   magenta = merge\n"
            "  blue = fuse     red = delete")
        # Transparent spheres for the live densify analysis (current snapshot).
        self._checks["analysis"].setToolTip(
            "Show the current densify analysis as transparent spheres:\n"
            "  cyan = under-represented region   yellow = over-represented\n"
            "  pink = bridging (spans a gap between structures)")

        # No skeleton until one is loaded.
        self.set_skeleton_available(False)

        # ── raymarch blend (only shown in the Raymarch ellipsoid mode) ──
        next_row = len(rows) + 1
        self._build_raymarch_controls(grid, next_row)

        # ── SDF slice plane ───────────────────────────────────────────
        self._build_slice_controls(grid, next_row + 1)

    def _build_raymarch_controls(self, grid: QtWidgets.QGridLayout,
                                 row: int) -> None:
        """A Blend slider that softly merges (smooth-union) the raymarched
        ellipsoids.  Hidden unless the ellipsoid render mode is Raymarch."""
        self._rm_blend_label = QtWidgets.QLabel("Blend")
        self._rm_blend_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._rm_blend_slider.setRange(0, 200)        # 0..2.0 of ellipsoid size
        self._rm_blend_slider.setValue(0)
        self._rm_blend_slider.setToolTip(
            "Smoothly merge neighbouring ellipsoids (smooth union).\n"
            "0 = hard union; higher = softer, more organic blending.")
        self._rm_blend_slider.valueChanged.connect(
            lambda v: self.raymarchBlendChanged.emit(v / 100.0))
        grid.addWidget(self._rm_blend_label, row, 0)
        grid.addWidget(self._rm_blend_slider, row, 1)
        self.set_raymarch_controls_visible(False)

    def set_raymarch_controls_visible(self, on: bool) -> None:
        self._rm_blend_label.setVisible(on)
        self._rm_blend_slider.setVisible(on)

    def raymarch_blend(self) -> float:
        return self._rm_blend_slider.value() / 100.0

    def _build_slice_controls(self, grid: QtWidgets.QGridLayout, row: int) -> None:
        """Toggle + plane/source dropdowns + position slider for the SDF slice."""
        # Thin separator so the slice block reads as its own group.
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(120,130,160,90); border: none;")
        grid.addWidget(sep, row, 0, 1, 2)
        row += 1

        self._chk_slice = QtWidgets.QCheckBox("SDF Slice")
        self._chk_slice.setChecked(False)
        self._chk_slice.setToolTip(
            "Push the SDF onto a texture plane and slide it through the volume.")
        grid.addWidget(self._chk_slice, row, 0)

        self._combo_slice_plane = QtWidgets.QComboBox()
        for lbl in slice_module.PLANE_LABELS:           # XY | XZ | YZ
            self._combo_slice_plane.addItem(lbl, lbl)
        self._combo_slice_plane.setToolTip("Slice plane (XY, XZ or YZ).")
        grid.addWidget(self._combo_slice_plane, row, 1)

        src_label = QtWidgets.QLabel("Source")
        self._combo_slice_source = QtWidgets.QComboBox()
        self._combo_slice_source.addItem("Mesh", "mesh")
        self._combo_slice_source.addItem("Ellipsoids", "ellipsoids")
        self._combo_slice_source.addItem("Difference", "difference")
        self._combo_slice_source.setToolTip(
            "Slice source: the mesh SDF, the ellipsoid-union SDF, or their\n"
            "difference (ellipsoid − mesh: where the fit over/under-covers).")
        grid.addWidget(src_label, row + 1, 0)
        grid.addWidget(self._combo_slice_source, row + 1, 1)

        # Plane + source are ALWAYS visible; only the checkbox toggles the slice
        # on/off.  The position slider lives in the 3-D viewport (not here).
        self._chk_slice.toggled.connect(self.sliceToggled)
        self._combo_slice_plane.activated.connect(
            lambda _i: self.slicePlaneChanged.emit(self._combo_slice_plane.currentData()))
        self._combo_slice_source.activated.connect(
            lambda _i: self.sliceSourceChanged.emit(self._combo_slice_source.currentData()))

    # ── slice accessors (single source of truth = the widgets) ─────────

    def slice_enabled(self) -> bool:
        return self._chk_slice.isChecked()

    def slice_plane(self) -> str:
        return self._combo_slice_plane.currentData()

    def slice_source(self) -> str:
        return self._combo_slice_source.currentData()

    def _make_combo(self, key: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        for label, mode in self._MODE_ITEMS[key]:
            combo.addItem(label, mode)
        default = self._DEFAULT_MODE.get(key, RENDER_TRANSPARENT)
        idx = combo.findData(default)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.activated.connect(
            lambda _i, k=key, c=combo: self.renderModeChanged.emit(k, c.currentData())
        )
        self._combos[key] = combo
        return combo

    def apply_theme(self) -> None:
        """(Re)apply the overlay style for the current light/dark mode.

        A translucent glass panel: dark glass + light text in dark mode, light
        glass + dark text in light mode (so it stays readable over a white
        viewport).
        """
        if theme.is_dark_mode():
            panel, border = "rgba(20, 22, 30, 200)", "rgba(120, 130, 160, 140)"
            label, text = "#c8d0e0", "#dde2ee"
            combo_bg, combo_hover, combo_border = "#2a2e3a", "#343a48", "rgba(120,130,160,140)"
            sel_bg = "#4a5a8a"
        else:
            panel, border = "rgba(245, 247, 250, 220)", "rgba(120, 130, 160, 160)"
            label, text = "#2a2e3a", "#1a1e28"
            combo_bg, combo_hover, combo_border = "#ffffff", "#e8ecf4", "rgba(120,130,160,160)"
            sel_bg = "#bcd0ff"
        self.setStyleSheet(
            "ViewportOverlay {"
            f"  background-color: {panel};"
            f"  border: 1px solid {border};"
            "  border-radius: 6px;"
            "}"
            f"QLabel {{ color: {label}; font-size: 11px; font-weight: bold; }}"
            f"QCheckBox {{ color: {text}; font-size: 11px; }}"
            f"QComboBox {{ color: {text}; background: {combo_bg};"
            f"            border: 1px solid {combo_border};"
            "            border-radius: 3px; font-size: 11px; padding: 1px 4px; }"
            f"QComboBox:hover {{ background: {combo_hover}; }}"
            "QComboBox QAbstractItemView {"
            f"  color: {text}; background: {combo_bg};"
            "  selection-color: #ffffff;"
            f"  selection-background-color: {sel_bg};"
            f"  border: 1px solid {combo_border};"
            "  outline: 0; }"
        )

    def set_skeleton_available(self, available: bool) -> None:
        """Show the skeleton checkbox only when a skeleton actually exists."""
        cb = self._checks.get("skeleton")
        if cb is not None:
            cb.setVisible(available)

    def is_checked(self, key: str) -> bool:
        cb = self._checks.get(key)
        return cb.isChecked() if cb else False

    def render_mode(self, target: str) -> Optional[str]:
        combo = self._combos.get(target)
        return combo.currentData() if combo else None


class SceneViewer3D(_BaseViewer):
    """One viewport showing mesh, skeleton bones and ellipsoids together.

    Per-element visibility and a *per-object* render mode (wireframe /
    transparent / lighting, chosen independently for the mesh and the
    ellipsoids) are driven by an in-viewport :class:`ViewportOverlay` menu.
    """

    # Skeleton colours.  These are deliberately mid-saturation tones that read
    # against *both* the black (dark) and white (light) viewport backgrounds —
    # the GL items render with translucent (not additive) blending so the
    # colours stay true instead of washing out to white on a light background.
    BONE_COLOR = (0.90, 0.20, 0.20, 1.0)    # warm red bones
    JOINT_COLOR = (1.0, 0.45, 0.0, 1.0)     # vivid orange joints

    def __init__(self):
        super().__init__()

        # ── element state ──
        self._mesh_verts: Optional[np.ndarray] = None
        self._mesh_faces: Optional[np.ndarray] = None
        self._thickness_colors: Optional[np.ndarray] = None

        self._ell_centers: Optional[np.ndarray] = None
        self._ell_radii: Optional[np.ndarray] = None
        self._ell_rotations: Optional[np.ndarray] = None
        self._ell_colors: Optional[np.ndarray] = None

        self._bone_positions: Optional[np.ndarray] = None
        self._bone_parents: Optional[np.ndarray] = None

        # ── SDF slice plane ──
        self._sdf_grid: Optional[np.ndarray] = None      # mesh SDF (nz, ny, nx)
        self._sdf_origin: Optional[np.ndarray] = None
        self._sdf_dx: Optional[float] = None
        self._mesh_depth: float = 1e-4
        self._grid_wp = None                             # mesh grid uploaded once
        self._sdf_blowup_vox: float = 0.0                # uniform SDF offset (voxels)
        self._slice_lut = slice_module.make_sdf_lut()
        self._slice_lut_wp, self._slice_lut_n = slice_module.make_lut_wp(self._slice_lut)

        # ── GL items ──
        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._ell_item: Optional[gl.GLMeshItem] = None
        self._slice_item: Optional[gl.GLImageItem] = None
        # Render mode the current ellipsoid item was built with, so live fit
        # updates can reuse it (setMeshData) instead of recreating it.
        self._ell_item_mode: Optional[str] = None
        # Superquadric rendering: per-primitive (N,2) roundness exponents that
        # warp each unit mesh, or None for plain ellipsoids/spheres.
        self._ell_sq_eps: Optional[np.ndarray] = None
        self._ell_sq_bend: Optional[np.ndarray] = None
        # Render primitive kind ("capsule" builds capsule meshes), or None.
        self._ell_primitive: Optional[str] = None
        self._bone_line_item: Optional[gl.GLLinePlotItem] = None
        self._joint_item: Optional[gl.GLScatterPlotItem] = None
        self._region_item: Optional[gl.GLLinePlotItem] = None
        # Exploded per-bone region preview (Bone-Separation verification).
        self._region_preview_item: Optional[gl.GLMeshItem] = None
        self._underrep_item: Optional[gl.GLScatterPlotItem] = None

        # ── SuperFit operation gizmos ──
        # Each event is dict(op, c=center(3,), r=radius, birth=step); drawn as a
        # colour-coded wireframe box that fades out over ``_op_gizmo_life`` steps.
        self._op_gizmo_item: Optional[gl.GLLinePlotItem] = None
        self._op_events: list[dict] = []
        self._op_current_step = 0
        self._op_gizmo_life = 50

        # ── densify analysis overlay (transparent spheres, current snapshot) ──
        self._analysis_item: Optional[gl.GLMeshItem] = None
        self._analysis_regions: dict = {}

        # ── display flags ──
        self._show_mesh = True
        self._show_skeleton = True
        self._show_ellipsoids = True
        self._show_op_gizmos = True
        self._show_analysis = True
        # Independent render mode per surface object — defaults come from the
        # overlay combos (the single source of truth); the mesh starts wireframe.
        self._mesh_render_mode = ViewportOverlay._DEFAULT_MODE["mesh"]
        self._ell_render_mode = ViewportOverlay._DEFAULT_MODE["ellipsoids"]

        # ── overlay menu ──
        self._overlay = ViewportOverlay()
        self._overlay.visibilityChanged.connect(self._on_visibility_changed)
        self._overlay.renderModeChanged.connect(self._on_render_mode_changed)
        self._overlay.sliceToggled.connect(self._on_slice_toggled)
        self._overlay.slicePlaneChanged.connect(self._on_slice_plane_changed)
        self._overlay.sliceSourceChanged.connect(self._on_slice_source_changed)
        self._overlay.raymarchBlendChanged.connect(self._on_raymarch_blend_changed)
        self._view.add_corner_widget(self._overlay, corner="tl", margin=10)

        # ── slice position slider (integrated along the viewport bottom) ──
        self._build_slice_slider()

        # ── GPU raymarch overlay ──
        # A full-viewport image label that shows the sphere-traced ellipsoid SDF
        # (ellipsoid render mode "Raymarch").  Transparent where rays miss, so
        # the rest of the GL scene composites through.  A poll timer re-renders
        # whenever the camera or viewport size changes.
        self._build_raymarch_overlay()

    def _build_raymarch_overlay(self) -> None:
        self._rm_label = QtWidgets.QLabel()
        self._rm_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._rm_label.setStyleSheet("background: transparent;")
        self._rm_label.setScaledContents(False)
        self._view.add_corner_widget(self._rm_label, corner="fill", margin=0)
        self._rm_label.lower()                # below the overlay menu / slider
        self._rm_label.setVisible(False)
        # Stateful GPU tracer (caches the ellipsoid population on the device).
        self._rm = raymarch.Raymarcher()
        self._rm_dirty = True                 # population needs (re)upload
        # Adaptive resolution: render at a reduced longest-edge while the camera
        # moves (smooth orbit), then a crisp full-resolution pass once it stops.
        self._rm_move_px = 460
        self._rm_pending_full = False
        self._rm_cam_sig = None               # last camera/size signature
        self._rm_blend = 0.0                  # smooth-union fraction (0 = hard)
        self._rm_timer = QtCore.QTimer(self._view)
        self._rm_timer.setInterval(33)        # ~30 fps camera-change polling
        self._rm_timer.timeout.connect(self._on_raymarch_poll)

    def _build_slice_slider(self) -> None:
        """Horizontal slice-position slider docked along the viewport bottom."""
        self._slice_slider_frame = QtWidgets.QFrame()
        lay = QtWidgets.QHBoxLayout(self._slice_slider_frame)
        lay.setContentsMargins(12, 5, 12, 5)
        self._slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slice_slider.setRange(0, 0)
        self._slice_slider.setToolTip("Slide the SDF slice plane through the volume.")
        self._slice_slider.valueChanged.connect(self._on_slice_position_changed)
        lay.addWidget(self._slice_slider)
        self._style_slice_slider()
        self._view.add_corner_widget(self._slice_slider_frame, corner="bottom",
                                     margin=12)
        self._slice_slider_frame.setVisible(False)

    def _style_slice_slider(self) -> None:
        """Translucent glass background so the slider reads over the viewport."""
        if theme.is_dark_mode():
            panel, border = "rgba(20, 22, 30, 200)", "rgba(120, 130, 160, 140)"
        else:
            panel, border = "rgba(245, 247, 250, 220)", "rgba(120, 130, 160, 160)"
        self._slice_slider_frame.setStyleSheet(
            f"QFrame {{ background-color: {panel};"
            f"          border: 1px solid {border}; border-radius: 6px; }}")

    def _set_slice_range(self, n: int, keep_value: bool = False) -> None:
        """Resize the slider to ``[0, n-1]`` (centre unless ``keep_value``)."""
        n = max(int(n), 1)
        self._slice_slider.blockSignals(True)
        cur = self._slice_slider.value()
        self._slice_slider.setRange(0, n - 1)
        self._slice_slider.setValue(min(cur, n - 1) if keep_value else n // 2)
        self._slice_slider.blockSignals(False)

    def apply_theme(self) -> None:
        """Re-colour the viewport, overlay and objects for the current theme.

        Rebuilds the mesh and ellipsoids so live primary/secondary colour
        changes (from the colour pickers) take effect immediately.
        """
        self._view.setBackgroundColor(theme.bg((0, 0, 0)))
        self._overlay.apply_theme()
        self._style_slice_slider()
        self._rebuild_mesh()
        self._rebuild_ellipsoids()
        # The SDF colormap is theme-dependent — rebuild the LUT (host + device).
        self._slice_lut = slice_module.make_sdf_lut()
        self._slice_lut_wp, self._slice_lut_n = slice_module.make_lut_wp(self._slice_lut)
        if self._overlay.slice_enabled():
            self._update_slice()

    # ── overlay callbacks ──────────────────────────────────────────────

    def _on_visibility_changed(self, key: str, on: bool) -> None:
        if key == "mesh":
            self._show_mesh = on
            if self._mesh_item is not None:
                self._mesh_item.setVisible(on)
        elif key == "skeleton":
            self._show_skeleton = on
            for it in (self._bone_line_item, self._joint_item):
                if it is not None:
                    it.setVisible(on)
        elif key == "ellipsoids":
            self._show_ellipsoids = on
            if self._ell_item is not None:
                self._ell_item.setVisible(on)
            if self._ell_render_mode == RENDER_RAYMARCH:
                self._rm_label.setVisible(on)
                if on:
                    self._render_raymarch(full=True)
        elif key == "operations":
            self._show_op_gizmos = on
            if self._op_gizmo_item is not None:
                self._op_gizmo_item.setVisible(on and bool(self._op_events))
        elif key == "analysis":
            self._show_analysis = on
            if self._analysis_item is not None:
                self._analysis_item.setVisible(on and bool(self._analysis_regions))

    def _on_render_mode_changed(self, target: str, mode: str) -> None:
        if target == "mesh":
            self._mesh_render_mode = mode
            self._rebuild_mesh()
        elif target == "ellipsoids":
            was_rm = self._ell_render_mode == RENDER_RAYMARCH
            self._ell_render_mode = mode
            self._overlay.set_raymarch_controls_visible(mode == RENDER_RAYMARCH)
            if mode == RENDER_RAYMARCH:
                self._enter_raymarch()
            else:
                if was_rm:
                    self._exit_raymarch()
                self._rebuild_ellipsoids()

    def _on_raymarch_blend_changed(self, frac: float) -> None:
        self._rm_blend = max(0.0, float(frac))
        if self._ell_render_mode == RENDER_RAYMARCH:
            # Low-res while dragging the slider; the poll refines to full res
            # once the value settles (camera signature unchanged + pending flag).
            self._render_raymarch(full=False)
            self._rm_pending_full = True

    # ── GPU raymarch overlay ───────────────────────────────────────────────

    def _enter_raymarch(self) -> None:
        """Switch ellipsoids to the sphere-traced overlay: drop the GL mesh
        item, show the image label and start camera-change polling."""
        if self._ell_item is not None:
            self._view.removeItem(self._ell_item)
            self._ell_item = None
            self._ell_item_mode = None
        self._rm_label.setVisible(self._show_ellipsoids)
        self._rm_cam_sig = None               # force a render on the next tick
        self._render_raymarch(full=True)      # camera is static -> crisp
        self._rm_timer.start()

    def _exit_raymarch(self) -> None:
        self._rm_timer.stop()
        self._rm_label.clear()
        self._rm_label.setVisible(False)

    def _camera_signature(self):
        """A hashable snapshot of everything that changes the marched image:
        camera pose and the viewport pixel size."""
        opts = self._view.opts
        c = opts.get("center")
        cx, cy, cz = (float(c.x()), float(c.y()), float(c.z())) if c is not None \
            else (0.0, 0.0, 0.0)
        return (
            round(cx, 5), round(cy, 5), round(cz, 5),
            round(float(opts.get("distance", 10.0)), 5),
            round(float(opts.get("elevation", 0.0)), 4),
            round(float(opts.get("azimuth", 0.0)), 4),
            round(float(opts.get("fov", 60.0)), 4),
            int(self._view.width()), int(self._view.height()),
        )

    def _on_raymarch_poll(self) -> None:
        if self._ell_render_mode != RENDER_RAYMARCH:
            self._rm_timer.stop()
            return
        sig = self._camera_signature()
        if sig != self._rm_cam_sig:
            # Camera is moving → cheap low-res pass; remember to refine later.
            self._render_raymarch(full=False)
            self._rm_pending_full = True
        elif self._rm_pending_full:
            # Camera has settled → one crisp full-resolution pass.
            self._render_raymarch(full=True)
            self._rm_pending_full = False

    def _inv_mvp(self) -> Optional[np.ndarray]:
        """Inverse model-view-projection of the live viewport (NDC → world),
        as a (4,4) row-major float32 array — or None if non-invertible."""
        try:
            # pyqtgraph 0.14 requires (region, viewport); the full viewport for
            # both gives the standard full-frame frustum.
            vp = self._view.getViewport()
            proj = self._view.projectionMatrix(vp, vp)
            view = self._view.viewMatrix()
        except Exception:
            return None
        inv, ok = (proj * view).inverted()
        if not ok:
            return None
        rows = [inv.row(i) for i in range(4)]
        return np.array([[r.x(), r.y(), r.z(), r.w()] for r in rows],
                        dtype=np.float32)

    def _render_raymarch(self, full: bool = False) -> None:
        """Sphere-trace the ellipsoid union for the current camera and push the
        result into the overlay label.

        ``full`` renders at native (device-pixel) resolution for a crisp still
        frame; otherwise at a reduced longest-edge (``_rm_move_px``) and the
        pixmap is stretched to fill — fast enough to orbit interactively.
        """
        if self._ell_render_mode != RENDER_RAYMARCH:
            return
        self._rm_cam_sig = self._camera_signature()
        vw, vh = int(self._view.width()), int(self._view.height())
        if vw <= 0 or vh <= 0:
            return
        if self._ell_centers is None or len(self._ell_centers) == 0 \
                or not self._show_ellipsoids:
            self._rm_label.clear()
            return
        inv_mvp = self._inv_mvp()
        if inv_mvp is None:
            return
        # Upload the population to the device only when it actually changed.
        if self._rm_dirty:
            try:
                self._rm.update(self._ell_centers, self._ell_radii,
                                self._ell_rotations)
            except Exception:
                return
            self._rm_dirty = False

        dpr = float(self._view.devicePixelRatioF())
        if full:
            rw = max(1, int(round(vw * dpr)))      # native device pixels
            rh = max(1, int(round(vh * dpr)))
        else:
            scale = min(1.0, float(self._rm_move_px) / float(max(vw, vh)))
            rw = max(1, int(round(vw * scale)))
            rh = max(1, int(round(vh * scale)))
        base = tuple(float(x) for x in ellipsoid_color()[:3])
        try:
            img = self._rm.render(inv_mvp, rw, rh, base_rgb=base,
                                  bg_alpha=0.0, blend=self._rm_blend)
        except Exception:
            return
        buf = np.ascontiguousarray(img)       # keep alive while QImage references it
        qimg = QtGui.QImage(buf.data, rw, rh, rw * 4,
                            QtGui.QImage.Format_RGBA8888)
        pix = QtGui.QPixmap.fromImage(qimg.copy())
        if full and dpr != 1.0:
            # A device-pixel pixmap shown crisply in the logical-sized label.
            pix.setDevicePixelRatio(dpr)
        elif (rw, rh) != (vw, vh):
            pix = pix.scaled(vw, vh, QtCore.Qt.IgnoreAspectRatio,
                             QtCore.Qt.SmoothTransformation)
        self._rm_label.setPixmap(pix)

    # ── mesh ────────────────────────────────────────────────────────────

    def show_mesh(self, verts: np.ndarray, faces: np.ndarray) -> None:
        self._mesh_verts = np.ascontiguousarray(verts, dtype=np.float32)
        self._mesh_faces = np.ascontiguousarray(faces)
        self._thickness_colors = None   # invalidated until recomputed
        self._rebuild_mesh()

    def show_thickness(
        self,
        thickness_grid: np.ndarray,
        origin: np.ndarray,
        dx: float,
        n: int,
    ) -> tuple[float, float] | None:
        """Compute the per-vertex thickness heatmap and cache it.

        The heatmap is only *displayed* when the mesh render mode is "Dicke"
        (RENDER_THICKNESS); this always recomputes and caches it so switching to
        that mode is instant.
        """
        if self._mesh_verts is None or self._mesh_faces is None \
                or thickness_grid is None:
            return None
        colors, vmin, vmax = thickness_vertex_colors(
            self._mesh_verts, self._mesh_faces, thickness_grid, origin, dx, n,
        )
        self._thickness_colors = colors
        if self._mesh_render_mode == RENDER_THICKNESS:
            self._rebuild_mesh()
        return vmin, vmax

    def _rebuild_mesh(self) -> None:
        if self._mesh_item is not None:
            self._view.removeItem(self._mesh_item)
            self._mesh_item = None
        if self._mesh_verts is None or self._mesh_faces is None:
            return
        # In thickness mode, colour vertices by the cached heatmap (if computed).
        vcols = (self._thickness_colors
                 if self._mesh_render_mode == RENDER_THICKNESS else None)
        self._mesh_item = _build_mesh_item(
            self._mesh_verts, self._mesh_faces, self._mesh_render_mode,
            vertex_colors=vcols,
        )
        self._mesh_item.setVisible(self._show_mesh)
        self._view.addItem(self._mesh_item)

    # ── exploded region preview (Bone-Separation verification) ──────────

    def show_region_preview(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        vertex_colors: np.ndarray,
    ) -> None:
        """Show an exploded, per-bone-coloured preview of the region submeshes.

        Used to verify the Bone-Separation carving: each bone's submesh is
        pushed radially outward (by the host's Mesh-Blowup control) and tinted a
        distinct colour.  The normal mesh is hidden while the preview is active
        so the coloured regions read cleanly; ``clear_region_preview`` restores
        it.  Always drawn opaque (RENDER_SOLID) so the colours stay legible
        regardless of the mesh's own render mode.
        """
        self._clear_region_preview_item()
        item = _build_mesh_item(
            np.ascontiguousarray(verts, dtype=np.float32),
            np.ascontiguousarray(faces),
            RENDER_SOLID,
            vertex_colors=np.ascontiguousarray(vertex_colors, dtype=np.float32),
        )
        self._region_preview_item = item
        self._view.addItem(item)
        if self._mesh_item is not None:
            self._mesh_item.setVisible(False)

    def clear_region_preview(self) -> None:
        """Remove the exploded region preview and restore the normal mesh."""
        self._clear_region_preview_item()
        if self._mesh_item is not None:
            self._mesh_item.setVisible(self._show_mesh)

    def _clear_region_preview_item(self) -> None:
        if self._region_preview_item is not None:
            self._view.removeItem(self._region_preview_item)
            self._region_preview_item = None

    # ── ellipsoids ──────────────────────────────────────────────────────

    def show_ellipsoids_fast(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        colors: Optional[np.ndarray] = None,
        sq_eps: Optional[np.ndarray] = None,
        sq_bend: Optional[np.ndarray] = None,
        primitive: Optional[str] = None,
    ) -> None:
        self._ell_centers = centers
        self._ell_radii = radii
        self._ell_rotations = rotations
        self._ell_colors = colors
        self._ell_sq_eps = sq_eps      # per-primitive (N,2) roundness, or None
        self._ell_sq_bend = sq_bend    # per-primitive (N,2) bend, or None
        self._ell_primitive = primitive  # e.g. "capsule", or None for ellipsoid
        self._rm_dirty = True          # population changed → re-upload for raymarch
        self._rebuild_ellipsoids()
        # Live slice refresh while the ellipsoid-union SDF is the slice source.
        if self._overlay.slice_enabled() and self._overlay.slice_source() == "ellipsoids":
            self._update_slice()

    def show_ellipsoids(self, ellipsoid_set) -> None:
        colors = None
        if ellipsoid_set.colors is not None:
            colors = np.array(ellipsoid_set.colors, dtype=np.float32)
        self.show_ellipsoids_fast(
            ellipsoid_set.centers, ellipsoid_set.radii,
            ellipsoid_set.rotations, colors,
        )

    def clear_ellipsoids(self) -> None:
        self._ell_centers = None
        self._rm_dirty = True
        if self._ell_item is not None:
            self._view.removeItem(self._ell_item)
            self._ell_item = None
            self._ell_item_mode = None
        if self._ell_render_mode == RENDER_RAYMARCH:
            self._rm_label.clear()

    def _rebuild_ellipsoids(self) -> None:
        # Raymarch mode draws the SDF directly into the overlay image instead of
        # building a GL mesh — re-render for the latest population and bail.
        # Low-res now (cheap during a live fit); the poll refines to full res
        # once updates stop.
        if self._ell_render_mode == RENDER_RAYMARCH:
            self._rm_dirty = True
            self._render_raymarch(full=False)
            self._rm_pending_full = True
            return

        if self._ell_centers is None or len(self._ell_centers) == 0:
            if self._ell_item is not None:
                self._view.removeItem(self._ell_item)
                self._ell_item = None
                self._ell_item_mode = None
            return

        verts, faces, face_colors = build_concatenated_mesh(
            self._ell_centers, self._ell_radii, self._ell_rotations,
            self._ell_colors,
            sq_eps=getattr(self, "_ell_sq_eps", None),
            sq_bend=getattr(self, "_ell_sq_bend", None),
            primitive=getattr(self, "_ell_primitive", None),
        )
        if verts.shape[0] == 0:
            if self._ell_item is not None:
                self._view.removeItem(self._ell_item)
                self._ell_item = None
                self._ell_item_mode = None
            return

        # Fast path: same render mode → update the existing item's mesh data in
        # place.  Recreating the GLMeshItem (new GL buffers) every fit step is
        # what made live fitting crawl.
        if self._ell_item is not None and self._ell_item_mode == self._ell_render_mode:
            fc = face_colors.astype(np.float32, copy=True)
            fc[:, 3] = _ell_alpha_for_mode(self._ell_render_mode)
            md = gl.MeshData(vertexes=verts, faces=faces, faceColors=fc)
            if self._ell_render_mode in (RENDER_LIGHTING, RENDER_THICKNESS):
                # Inject precomputed normals so pyqtgraph skips its per-vertex
                # Python normal loop (the reason lighting crawled while fitting).
                md._vertexNormals = self._smooth_vertex_normals(verts, faces)
            self._ell_item.setMeshData(meshdata=md)
            self._ell_item.setVisible(self._show_ellipsoids)
            return

        # Slow path: (re)create the item for a new render mode.
        if self._ell_item is not None:
            self._view.removeItem(self._ell_item)
            self._ell_item = None
        self._ell_item = _build_mesh_item(
            verts, faces, self._ell_render_mode,
            face_colors=face_colors, cull=True,
        )
        if self._ell_render_mode in (RENDER_LIGHTING, RENDER_THICKNESS):
            md = self._ell_item.opts.get("meshdata")
            if md is not None:
                md._vertexNormals = self._smooth_vertex_normals(verts, faces)
                self._ell_item.meshDataChanged()
        self._ell_item_mode = self._ell_render_mode
        self._ell_item.setVisible(self._show_ellipsoids)
        self._view.addItem(self._ell_item)

    def _smooth_vertex_normals(self, verts: np.ndarray,
                               faces: np.ndarray) -> np.ndarray:
        """Per-vertex normals for the smooth (lighting/thickness) modes.

        Analytic for plain ellipsoids (cheapest); vectorised face-average
        fallback for superquadrics / capsules.  Either way, far faster than
        pyqtgraph recomputing them in a Python loop on each update.
        """
        bend = getattr(self, "_ell_sq_bend", None)
        if (getattr(self, "_ell_sq_eps", None) is None
                and getattr(self, "_ell_primitive", None) not in ("capsule",)
                and (bend is None or not np.any(bend))
                and self._ell_radii is not None
                and self._ell_rotations is not None):
            return ellipsoid_vertex_normals(self._ell_radii, self._ell_rotations)
        return _vertex_normals(verts, faces)

    # ── skeleton ────────────────────────────────────────────────────────

    def show_bones(self, positions: np.ndarray, parent_indices: np.ndarray) -> None:
        self._bone_positions = np.ascontiguousarray(positions, dtype=np.float32)
        self._bone_parents = parent_indices
        self.clear_bones()

        line_pts = []
        for i, pi in enumerate(parent_indices):
            if pi >= 0:
                line_pts.append(positions[pi])
                line_pts.append(positions[i])
        if line_pts:
            pts = np.array(line_pts, dtype=np.float32)
            self._bone_line_item = gl.GLLinePlotItem(
                pos=pts, color=self.BONE_COLOR, width=3.0, mode='lines',
                antialias=True, glOptions='translucent',
            )
            self._bone_line_item.setVisible(self._show_skeleton)
            self._view.addItem(self._bone_line_item)

        self._joint_item = gl.GLScatterPlotItem(
            pos=positions, color=self.JOINT_COLOR, size=8.0, pxMode=True,
            glOptions='translucent',
        )
        self._joint_item.setVisible(self._show_skeleton)
        self._view.addItem(self._joint_item)

        # A skeleton now exists → reveal its checkbox in the overlay.
        self._overlay.set_skeleton_available(True)

    def clear_bones(self) -> None:
        for attr in ("_bone_line_item", "_joint_item"):
            it = getattr(self, attr)
            if it is not None:
                self._view.removeItem(it)
                setattr(self, attr, None)

    def remove_skeleton(self) -> None:
        """Drop the skeleton entirely and hide its overlay control.

        Called when a mesh without a skeleton (a static mesh) is loaded.
        """
        self.clear_bones()
        self._bone_positions = None
        self._bone_parents = None
        self._overlay.set_skeleton_available(False)

    # ── region box (high-res local fit) ────────────────────────────────

    _REGION_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    @staticmethod
    def _box_segments(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
        lo = np.asarray(aabb_min, dtype=np.float32)
        hi = np.asarray(aabb_max, dtype=np.float32)
        corners = np.array([
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ], dtype=np.float32)
        return np.concatenate(
            [corners[[a, b]] for a, b in SceneViewer3D._REGION_EDGES], axis=0
        ).astype(np.float32)

    def show_region_box(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> None:
        self.show_region_boxes([(aabb_min, aabb_max)])

    def show_region_boxes(self, boxes) -> None:
        """Draw one or more high-res local-fit boxes as wireframe outlines.

        ``boxes`` is an iterable of ``(aabb_min, aabb_max)`` world boxes — each
        small box marks an area currently being optimised.  All boxes share a
        single line item (one draw call).
        """
        segs = [self._box_segments(lo, hi) for lo, hi in boxes]
        if not segs:
            self.clear_region_box()
            return
        pts = np.concatenate(segs, axis=0).astype(np.float32)
        color = (0.2, 0.9, 1.0, 1.0)
        if self._region_item is None:
            self._region_item = gl.GLLinePlotItem(
                pos=pts, color=color, width=2.0, mode='lines', antialias=True,
            )
            self._region_item.setGLOptions('translucent')
            self._view.addItem(self._region_item)
        else:
            self._region_item.setData(pos=pts, color=color, width=2.0, mode='lines')

    def clear_region_box(self) -> None:
        if self._region_item is not None:
            self._view.removeItem(self._region_item)
            self._region_item = None

    # ── SuperFit operation gizmos ───────────────────────────────────────
    # Colour per operation type (RGB); alpha is added per event from its age.
    _OP_COLORS = {
        "spawn":  (0.25, 1.00, 0.35),   # green  — new ellipsoid
        "split":  (1.00, 0.65, 0.10),   # orange — divided ellipsoid
        "merge":  (0.90, 0.30, 1.00),   # magenta — fused pair → one
        "fuse":   (0.30, 0.60, 1.00),   # blue   — redundant, dropped
        "delete": (1.00, 0.25, 0.25),   # red    — outside / degenerate, dropped
    }

    def add_op_gizmos(self, step: int, events) -> None:
        """Record SuperFit operation markers emitted at maintenance ``step``.

        ``events`` is an iterable of ``(op, center_world, radius_world)``.  Each
        becomes a colour-coded wireframe box that lingers for ``_op_gizmo_life``
        steps (see :meth:`tick_op_gizmos`) and fades as it ages.
        """
        self._op_current_step = int(step)
        for op, center, radius in events:
            self._op_events.append({
                "op": str(op),
                "c": np.asarray(center, dtype=np.float32),
                "r": max(float(radius), 1e-4),
                "birth": int(step),
            })
        self._rebuild_op_gizmos()

    def tick_op_gizmos(self, step: int) -> None:
        """Advance the gizmo clock to ``step``: drop expired markers + refade."""
        self._op_current_step = int(step)
        if not self._op_events:
            return
        alive = [e for e in self._op_events
                 if 0 <= step - e["birth"] < self._op_gizmo_life]
        self._op_events = alive
        self._rebuild_op_gizmos()

    def clear_op_gizmos(self) -> None:
        self._op_events = []
        if self._op_gizmo_item is not None:
            self._view.removeItem(self._op_gizmo_item)
            self._op_gizmo_item = None

    def _rebuild_op_gizmos(self) -> None:
        """Rebuild the single line item holding every live operation box."""
        seg_list, col_list = [], []
        for e in self._op_events:
            age = self._op_current_step - e["birth"]
            frac = max(0.0, 1.0 - age / float(self._op_gizmo_life))
            half = e["r"]
            segs = self._box_segments(e["c"] - half, e["c"] + half)   # (E, 3)
            r, g, b = self._OP_COLORS.get(e["op"], (1.0, 1.0, 1.0))
            alpha = 0.20 + 0.80 * frac      # newest = opaque, oldest = faint
            col = np.tile(np.array([r, g, b, alpha], dtype=np.float32),
                          (len(segs), 1))
            seg_list.append(segs)
            col_list.append(col)

        if not seg_list:
            if self._op_gizmo_item is not None:
                self._op_gizmo_item.setVisible(False)
            return

        pts = np.concatenate(seg_list, axis=0).astype(np.float32)
        cols = np.concatenate(col_list, axis=0).astype(np.float32)
        if self._op_gizmo_item is None:
            self._op_gizmo_item = gl.GLLinePlotItem(
                pos=pts, color=cols, width=2.0, mode="lines", antialias=True)
            self._op_gizmo_item.setGLOptions("translucent")
            self._view.addItem(self._op_gizmo_item)
        else:
            self._op_gizmo_item.setData(pos=pts, color=cols, width=2.0,
                                        mode="lines")
        self._op_gizmo_item.setVisible(self._show_op_gizmos)

    # ── densify analysis overlay (transparent spheres) ──────────────────
    # Fill colour per analysis class (RGB); alpha is applied in _rebuild.
    _ANALYSIS_COLORS = {
        "under":  (0.15, 0.85, 0.95),   # cyan   — under-represented region
        "over":   (1.00, 0.85, 0.15),   # yellow — over-represented (protruding)
        "bridge": (1.00, 0.35, 0.65),   # pink   — bridging a gap
    }

    def set_analysis_regions(self, regions: dict) -> None:
        """Show the current densify analysis as transparent world-space spheres.

        ``regions`` maps each class ("under" / "over" / "bridge") to a list of
        ``(center_world, radius_world)`` pairs.  Replaces the previous snapshot.
        """
        self._analysis_regions = regions or {}
        self._rebuild_analysis()

    def clear_analysis_regions(self) -> None:
        self._analysis_regions = {}
        if self._analysis_item is not None:
            self._view.removeItem(self._analysis_item)
            self._analysis_item = None

    def _rebuild_analysis(self) -> None:
        """Concatenate one transparent icosphere per analysed region into a
        single GLMeshItem (one draw call), colour-coded per class."""
        verts_list, faces_list, cols_list = [], [], []
        offset = 0
        for cls, items in self._analysis_regions.items():
            rgb = self._ANALYSIS_COLORS.get(cls)
            if rgb is None:
                continue
            for center, radius in items:
                r = max(float(radius), 1e-4)
                v = (_UNIT_VERTS * r
                     + np.asarray(center, dtype=np.float32)[None, :])
                verts_list.append(v.astype(np.float32))
                faces_list.append(_UNIT_FACES + offset)
                cols_list.append(np.tile(
                    np.array([*rgb, 0.22], dtype=np.float32),
                    (_UNIT_N_VERTS, 1)))
                offset += _UNIT_N_VERTS

        if not verts_list:
            if self._analysis_item is not None:
                self._analysis_item.setVisible(False)
            return

        verts = np.concatenate(verts_list, axis=0).astype(np.float32)
        faces = np.concatenate(faces_list, axis=0).astype(np.int32)
        cols = np.concatenate(cols_list, axis=0).astype(np.float32)
        if self._analysis_item is None:
            self._analysis_item = gl.GLMeshItem(
                vertexes=verts, faces=faces, vertexColors=cols,
                drawFaces=True, drawEdges=False, smooth=True)
            self._analysis_item.setGLOptions("translucent")
            self._view.addItem(self._analysis_item)
        else:
            self._analysis_item.setMeshData(
                vertexes=verts, faces=faces, vertexColors=cols)
        self._analysis_item.setVisible(self._show_analysis)

    # ── SDF slice plane ─────────────────────────────────────────────────

    def set_sdf_volume(self, grid: np.ndarray, origin: np.ndarray,
                       dx: float) -> None:
        """Provide the mesh SDF volume the slice plane samples (and slides in).

        Stores the grid + world geometry, resizes the position slider to the
        current plane, and refreshes the slice if it is on.
        """
        self._sdf_grid = np.ascontiguousarray(grid, dtype=np.float32)
        self._sdf_origin = np.asarray(origin, dtype=np.float32)
        self._sdf_dx = float(dx)
        self._mesh_depth = max(-float(self._sdf_grid.min()), 1e-4)
        self._grid_wp = slice_module.upload_grid(self._sdf_grid)   # device, once
        normal = slice_module.PLANE_NORMAL[self._overlay.slice_plane()]
        self._set_slice_range(
            slice_module.n_slices(self._sdf_grid.shape, normal))
        if self._overlay.slice_enabled():
            self._update_slice()

    def set_sdf_blowup(self, voxels: float) -> None:
        """SDF blowup: add a uniform offset of ``voxels`` (× dx) to the mesh SDF.

        Positive erodes (surface inward), negative dilates.  Applied live in the
        slice render (GPU, no grid recompute); refreshes the slice if it is on.
        Stored in voxels and scaled by the current ``dx`` at render time.
        """
        self._sdf_blowup_vox = float(voxels)
        if self._overlay.slice_enabled() and self._sdf_grid is not None:
            self._update_slice()

    # ── overlay callbacks ──

    def _on_slice_toggled(self, on: bool) -> None:
        self._slice_slider_frame.setVisible(on)
        if on:
            self._update_slice()
        else:
            self._clear_slice()

    def _on_slice_plane_changed(self, _plane: str) -> None:
        # Plane changed → the normal axis (and its voxel count) changed; resize.
        if self._sdf_grid is not None:
            normal = slice_module.PLANE_NORMAL[self._overlay.slice_plane()]
            self._set_slice_range(
                slice_module.n_slices(self._sdf_grid.shape, normal))
        self._update_slice()

    def _on_slice_source_changed(self, _source: str) -> None:
        self._update_slice()

    def _on_slice_position_changed(self, _idx: int) -> None:
        self._update_slice()

    # ── slice computation + rendering ──

    def _ellipsoid_depth(self) -> float:
        """Deepest interior |SDF| of the ellipsoid union.

        The MertStein interior of one ellipsoid bottoms out at ``-min(radius)``
        at its centre, so the union's most-negative value over all space is
        ``-max_i(min(radii_i))``.  Returns that depth magnitude (cheap, exact).
        """
        if self._ell_radii is None or len(self._ell_radii) == 0:
            return 1e-4
        min_axis = np.min(np.abs(self._ell_radii), axis=1)
        return max(float(min_axis.max()), 1e-4)

    def _update_slice(self) -> None:
        if not self._overlay.slice_enabled() or self._sdf_grid is None:
            self._clear_slice()
            return
        normal = slice_module.PLANE_NORMAL[self._overlay.slice_plane()]
        n = slice_module.n_slices(self._sdf_grid.shape, normal)
        k = max(0, min(n - 1, int(self._slice_slider.value())))
        source = self._overlay.slice_source()
        # Always render the slice at ~2K (longest in-plane axis).  Sampling +
        # colouring run entirely on the GPU (see sdf_slice.render_*), so the GUI
        # thread is not blocked — only the final RGBA is read back.
        W, H, px = slice_module.slice_resolution(
            self._sdf_grid.shape, self._sdf_dx, normal)
        nz, ny, nx = self._sdf_grid.shape
        out_band = 3.0 * float(self._sdf_dx)

        blow = self._sdf_blowup_vox * float(self._sdf_dx)
        if source == "difference":
            if self._ell_centers is None or len(self._ell_centers) == 0:
                self._clear_slice()
                return
            rgba = slice_module.render_diff(
                self._ell_centers, self._ell_radii, self._ell_rotations,
                self._grid_wp, self._sdf_origin, self._sdf_dx, nx, ny, nz,
                self._sdf_origin, normal, k, W, H, px, theme.BLUE, theme.YELLOW,
                offset=blow)
        elif source == "ellipsoids":
            if self._ell_centers is None or len(self._ell_centers) == 0:
                self._clear_slice()
                return
            rgba = slice_module.render_ellipsoid(
                self._ell_centers, self._ell_radii, self._ell_rotations,
                self._sdf_origin, normal, k, W, H, px, self._sdf_dx,
                self._slice_lut_wp, self._slice_lut_n,
                self._ellipsoid_depth(), out_band)
        else:  # mesh
            depth = max(self._mesh_depth - blow, 1e-4)   # blowup shifts the surface
            rgba = slice_module.render_mesh(
                self._grid_wp, self._sdf_origin, self._sdf_dx, nx, ny, nz,
                self._sdf_origin, normal, k, W, H, px,
                self._slice_lut_wp, self._slice_lut_n, depth, out_band,
                offset=blow)

        mat = slice_module.slice_transform(
            self._sdf_origin, normal, k, px, self._sdf_dx)
        transform = pg.Transform3D(mat)

        if self._slice_item is None:
            # smooth=True → bilinear texture filtering between the 2K pixels.
            self._slice_item = gl.GLImageItem(
                rgba, smooth=True, glOptions="translucent")
            self._view.addItem(self._slice_item)
        else:
            self._slice_item.setData(rgba)
        self._slice_item.setTransform(transform)
        self._slice_item.setVisible(True)

    def _clear_slice(self) -> None:
        if self._slice_item is not None:
            self._view.removeItem(self._slice_item)
            self._slice_item = None

    # ── under-representation overlay ────────────────────────────────────

    def show_underrepresented(
        self,
        points: np.ndarray,
        values: np.ndarray,
        vmax: float = 3.0,
        size: float = 6.0,
    ) -> None:
        if points is None or len(points) == 0:
            self.clear_underrepresented()
            return
        v = np.clip(values.astype(np.float32), 0.0, vmax)
        t = (v / max(vmax, 1e-6))[:, None]
        yellow = np.array([1.0, 0.95, 0.2], dtype=np.float32)
        red = np.array([1.0, 0.1, 0.05], dtype=np.float32)
        rgb = (1.0 - t) * yellow + t * red
        alpha = (0.45 + 0.55 * t).astype(np.float32)
        colors = np.concatenate([rgb, alpha], axis=1).astype(np.float32)
        if self._underrep_item is None:
            self._underrep_item = gl.GLScatterPlotItem(
                pos=points.astype(np.float32), color=colors, size=size, pxMode=True,
            )
            self._underrep_item.setGLOptions('translucent')
            self._view.addItem(self._underrep_item)
        else:
            self._underrep_item.setData(
                pos=points.astype(np.float32), color=colors, size=size,
            )

    def clear_underrepresented(self) -> None:
        if self._underrep_item is not None:
            self._view.removeItem(self._underrep_item)
            self._underrep_item = None

