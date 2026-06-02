"""shape_plugins.py — modular per-shape fitting plugins.

The app is moving from *ellipsoid-only* to a general **multi-shape SDF fitter**.
A :class:`ShapePlugin` encapsulates everything that differs between primitive
shapes, while the rest of the app (mesh loading, the SDF grid, the count / steps
/ learning-rate controls, fit/stop, run tracking) is shared in ``MainWindow``.

A plugin owns:

* ``id`` / ``display_name`` / ``primitive_noun`` — identity + UI wording,
* ``available`` — whether it is implemented yet (placeholders are greyed out),
* its **shape-specific option widgets** (``options_widget``),
* how those widgets map to ``OptimizationWorker`` kwargs (``fit_kwargs``),
* how live primitives are drawn in the viewport (``render``),
* per-shape persistence of its option widgets (handled generically by the base
  class via ``_setting_specs``).

Adding a new primitive = a new ``ShapePlugin`` subclass + one entry in
:func:`available_shapes`.  No existing shape needs to change — that is the whole
point of this layer.

The small ``widget_value`` / ``set_widget_value`` / ``widget_change_signal``
helpers are deliberately module-level so both the plugins *and* ``MainWindow``
(for its shared controls) reuse the exact same get/set/observe logic.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets


# ── Shared widget get/set/observe helpers (used by plugins AND MainWindow) ─────
# ``kind`` ∈ {"bool", "int", "float", "combo_data"}.

def widget_value(w, kind):
    if kind == "bool":
        return w.isChecked()
    if kind == "combo_data":
        return w.currentData()
    return w.value()


def set_widget_value(w, kind, v) -> None:
    try:
        if kind == "bool":
            w.setChecked(bool(v))
        elif kind == "int":
            w.setValue(int(v))
        elif kind == "float":
            w.setValue(float(v))
        elif kind == "combo_data":
            idx = w.findData(v)
            if idx >= 0:
                w.setCurrentIndex(idx)
    except Exception:
        pass            # ignore a stale / out-of-range persisted value


def widget_change_signal(w, kind):
    if kind == "bool":
        return w.toggled
    if kind == "combo_data":
        return w.currentIndexChanged
    return w.valueChanged


# ── Base class ────────────────────────────────────────────────────────────────

class ShapePlugin:
    """Base class for a fittable primitive shape.

    Subclasses set the identity attributes and implement ``_build_options``,
    ``fit_kwargs`` and ``render``.  Persistence + change-observation are provided
    here, driven by ``_setting_specs`` so every shape shares the same logic.
    """

    id: str = ""
    display_name: str = ""
    primitive_noun: str = "primitive"
    available: bool = True
    render_kind: str = "ellipsoid"   # how the viewer instances the primitive

    def __init__(self) -> None:
        self._widget: QtWidgets.QWidget | None = None

    # ── UI ──────────────────────────────────────────────────────────────────
    def options_widget(self) -> QtWidgets.QWidget:
        """Lazily build + cache the shape-specific options widget."""
        if self._widget is None:
            self._widget = self._build_options()
        return self._widget

    def _build_options(self) -> QtWidgets.QWidget:
        raise NotImplementedError

    # ── fitting ─────────────────────────────────────────────────────────────
    def fit_kwargs(self) -> dict:
        """Shape-specific ``OptimizationWorker`` kwargs from the widgets."""
        return {}

    # ── rendering ───────────────────────────────────────────────────────────
    def render(self, viewer, centers, radii, rotations) -> None:
        """Draw the live primitives emitted by the worker into the viewport."""
        raise NotImplementedError

    # ── persistence / observation (generic over ``_setting_specs``) ─────────
    def _setting_specs(self) -> list:
        """``(key, widget, kind)`` for every persisted option widget."""
        return []

    def panel_state(self) -> dict:
        return {key: widget_value(w, kind) for key, w, kind in self._setting_specs()}

    def apply_panel_state(self, state: dict) -> None:
        for key, w, kind in self._setting_specs():
            if key in state:
                set_widget_value(w, kind, state[key])

    def connect_changed(self, callback) -> None:
        for _key, w, kind in self._setting_specs():
            widget_change_signal(w, kind).connect(lambda *_: callback())


# ── Shared base for optimizer-fitted shapes ───────────────────────────────────

class _OptimizedPrimitiveShape(ShapePlugin):
    """Base for shapes fitted by the differentiable SDF optimiser.

    Builds the option groups common to every such shape — SuperFit density
    control, local fit, soft-min and the maintenance moves — and provides the
    matching persistence specs + fit kwargs.  Subclasses add only their
    shape-specific extras (e.g. the ellipsoid's SDF-method picker) and their
    ``fit_kwargs``.  This is where the "shared code between shapes" lives.

    Spheres are rendered as ellipsoids with equal radii, so ``render`` defaults
    to the ellipsoid fast-path and is inherited unchanged.
    """

    # ── small build helpers ──
    @staticmethod
    def _new_root() -> tuple:
        root = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        return root, v

    @staticmethod
    def _group(v: QtWidgets.QVBoxLayout, title: str) -> QtWidgets.QFormLayout:
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        v.addWidget(box)
        return form

    def _build_common_groups(self, v: QtWidgets.QVBoxLayout) -> None:
        """Build the SuperFit + Maintenance groups shared by all fitted shapes."""
        # ── SuperFit (adaptive density) ──
        sf = self._group(v, "SuperFit")
        self._chk_superfit = QtWidgets.QCheckBox("SuperFit (adaptive density)")
        self._chk_superfit.setChecked(True)
        self._chk_superfit.setToolTip(
            "Residual-driven growth: periodically maintain the population\n"
            "(merge / spawn / split) and grow it up to Max.")
        self._spin_superfit_every = QtWidgets.QSpinBox()
        self._spin_superfit_every.setRange(10, 10000)
        self._spin_superfit_every.setValue(150)
        self._spin_superfit_every.setSingleStep(10)
        self._spin_superfit_every.setToolTip(
            "Adaptive density-control rate: every N steps a maintenance round\n"
            "(merge / spawn / split) runs.  Smaller = more frequent.")
        self._spin_densify_until = QtWidgets.QSpinBox()
        self._spin_densify_until.setRange(0, 100)
        self._spin_densify_until.setValue(75)
        self._spin_densify_until.setSingleStep(5)
        self._spin_densify_until.setSuffix(" %")
        self._spin_densify_until.setToolTip(
            "Densify stop: density control runs only up to this fraction of\n"
            "training; afterwards pure Adam refinement of the fixed population\n"
            "(cleaner snapping, à la Gaussian Splatting).")
        self._chk_superfit.toggled.connect(self._spin_superfit_every.setEnabled)
        self._chk_superfit.toggled.connect(self._spin_densify_until.setEnabled)
        sf.addRow(self._chk_superfit)
        sf.addRow("Density rate:", self._spin_superfit_every)
        sf.addRow("Densify until:", self._spin_densify_until)

        self._chk_local_fit = QtWidgets.QCheckBox("Local Fit")
        self._chk_local_fit.setChecked(False)
        self._chk_local_fit.setToolTip(
            "Re-fit each maintained region against a fresh high-res SDF box.")
        self._spin_local_fit_start = QtWidgets.QSpinBox()
        self._spin_local_fit_start.setRange(0, 100)
        self._spin_local_fit_start.setValue(25)
        self._spin_local_fit_start.setSingleStep(5)
        self._spin_local_fit_start.setSuffix(" %")
        self._spin_local_fit_start.setToolTip(
            "Local-fit start: the high-res per-region fit only kicks in after\n"
            "this fraction of training has elapsed (default 25 %).  Earlier steps\n"
            "settle the global layout first.")
        self._spin_local_fit_start.setEnabled(self._chk_local_fit.isChecked())
        self._chk_local_fit.toggled.connect(self._spin_local_fit_start.setEnabled)
        sf.addRow(self._chk_local_fit)
        sf.addRow("Local fit start:", self._spin_local_fit_start)

        self._chk_soft_union = QtWidgets.QCheckBox("Soft min (experimental)")
        self._chk_soft_union.setChecked(False)
        self._chk_soft_union.setToolTip(
            "⚠ Experimental — OFF by default.\n"
            "Soft (LogSumExp) union of the primitive SDFs during the densify phase\n"
            "(gradient spread across several nearby primitives → denser gradients).\n"
            "In tests the result tended to look WORSE than the hard min;\n"
            "only enable it and compare on your own mesh.")
        sf.addRow(self._chk_soft_union)

        # ── Maintenance moves ──
        mv = self._group(v, "Maintenance moves")
        self._chk_merge = QtWidgets.QCheckBox("Merge")
        self._chk_merge.setChecked(True)
        self._chk_merge.setToolTip(
            "Fuse two overlapping primitives into one when it does not raise the loss.")
        self._chk_spawn = QtWidgets.QCheckBox("Spawn")
        self._chk_spawn.setChecked(True)
        self._chk_spawn.setToolTip(
            "Spawn a new, fully-inside primitive in isolated under-represented gaps.")
        self._chk_split = QtWidgets.QCheckBox("Split")
        self._chk_split.setChecked(True)
        self._chk_split.setToolTip(
            "Split oversized / bridging primitives and the nearest one to a gap.")
        mv.addRow(self._chk_merge)
        mv.addRow(self._chk_spawn)
        mv.addRow(self._chk_split)

    def _common_specs(self) -> list:
        return [
            ("superfit",       self._chk_superfit,        "bool"),
            ("superfit_every", self._spin_superfit_every, "int"),
            ("densify_until",  self._spin_densify_until,  "int"),
            ("local_fit",      self._chk_local_fit,       "bool"),
            ("local_fit_start", self._spin_local_fit_start, "int"),
            ("soft_union",     self._chk_soft_union,      "bool"),
            ("merge",          self._chk_merge,           "bool"),
            ("spawn",          self._chk_spawn,           "bool"),
            ("split",          self._chk_split,           "bool"),
        ]

    def _common_fit_kwargs(self) -> dict:
        return {
            "superfit":             self._chk_superfit.isChecked(),
            "superfit_every":       self._spin_superfit_every.value(),
            "densify_until_frac":   self._spin_densify_until.value() / 100.0,
            "local_fit":            self._chk_local_fit.isChecked(),
            "local_fit_start_frac": self._spin_local_fit_start.value() / 100.0,
            "soft_union":           self._chk_soft_union.isChecked(),
            "merge_enabled":        self._chk_merge.isChecked(),
            "spawn_enabled":        self._chk_spawn.isChecked(),
            "split_enabled":        self._chk_split.isChecked(),
        }

    def viewer_eps(self):
        """Default superquadric roundness (e1, e2), or None for no warp."""
        return None

    def _resolve_eps(self, eps, n):
        """Per-primitive (N,2) eps array to render with, or None for ellipsoids."""
        return None

    def _resolve_bend(self, eps, n):
        """Per-primitive (N,2) bend array to render with, or None."""
        return None

    def render(self, viewer, centers, radii, rotations, eps=None) -> None:
        # Ellipsoids/spheres draw as plain ellipsoids; superquadrics pass their
        # per-primitive roundness; bent superquadrics also pass bend; capsules
        # tell the viewer to build capsule meshes (radius = r0, half-length = r2).
        n = len(centers)
        viewer.show_ellipsoids_fast(
            centers, radii, rotations,
            sq_eps=self._resolve_eps(eps, n),
            sq_bend=self._resolve_bend(eps, n),
            primitive=self.render_kind)


# ── Ellipsoid ─────────────────────────────────────────────────────────────────

class EllipsoidShape(_OptimizedPrimitiveShape):
    id = "ellipsoid"
    display_name = "Ellipsoid"
    primitive_noun = "ellipsoid"
    available = True

    def _build_options(self) -> QtWidgets.QWidget:
        from ellipsoid import SDF_METHOD_NAMES   # lazy: keeps the module light
        root, v = self._new_root()

        # ── Approximation (ellipsoid-only — spheres have an exact SDF) ──
        appx = self._group(v, "Approximation")
        self._combo_sdf_method = QtWidgets.QComboBox()
        for mode_id in sorted(SDF_METHOD_NAMES.keys()):
            self._combo_sdf_method.addItem(SDF_METHOD_NAMES[mode_id], mode_id)
        self._combo_sdf_method.setCurrentIndex(0)
        self._combo_sdf_method.setToolTip("SDF approximation method for ellipsoid fitting")
        appx.addRow("SDF:", self._combo_sdf_method)

        self._build_common_groups(v)
        return root

    def _setting_specs(self) -> list:
        return [("sdf_method", self._combo_sdf_method, "combo_data")] + self._common_specs()

    def fit_kwargs(self) -> dict:
        self.options_widget()   # ensure widgets exist
        return {"sdf_mode": self._combo_sdf_method.currentData(),
                **self._common_fit_kwargs()}


# ── Sphere ────────────────────────────────────────────────────────────────────

class SphereShape(_OptimizedPrimitiveShape):
    id = "sphere"
    display_name = "Sphere"
    primitive_noun = "sphere"
    available = True

    def _build_options(self) -> QtWidgets.QWidget:
        root, v = self._new_root()
        # A sphere's SDF is exact, so there is no "SDF method" to pick.  Spheres
        # are fitted as isotropic, rotation-free ellipsoids (a post-step
        # projection in the worker enforces equal radii + identity rotation).
        info = QtWidgets.QLabel(
            "Spheres are isotropic with no rotation — exact SDF, no method to pick.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")
        v.addWidget(info)
        self._build_common_groups(v)
        return root

    def _setting_specs(self) -> list:
        return self._common_specs()

    def fit_kwargs(self) -> dict:
        self.options_widget()   # ensure widgets exist
        # ``primitive_shape`` tells the worker to project to isotropic radii +
        # identity rotation each step.  No ``sdf_mode`` (the default analytic
        # kernel is exact for spheres).
        return {"primitive_shape": "sphere", **self._common_fit_kwargs()}


# ── Superquadric ──────────────────────────────────────────────────────────────

class SuperquadricShape(_OptimizedPrimitiveShape):
    id = "superquadric"
    display_name = "Superquadric"
    primitive_noun = "superquadric"
    available = True

    def _build_options(self) -> QtWidgets.QWidget:
        root, v = self._new_root()

        # ── Roundness (shared across all primitives, fixed during a fit) ──
        rg = self._group(v, "Roundness  (ε)")
        self._spin_eps1 = QtWidgets.QDoubleSpinBox()
        self._spin_eps1.setRange(0.2, 1.5)
        self._spin_eps1.setSingleStep(0.05)
        self._spin_eps1.setDecimals(2)
        self._spin_eps1.setValue(0.6)
        self._spin_eps1.setToolTip(
            "ε₁ — north-south roundness.\n"
            "1.0 = ellipsoid, < 1 = boxier (sharper edges), > 1 = pinched.")
        self._spin_eps2 = QtWidgets.QDoubleSpinBox()
        self._spin_eps2.setRange(0.2, 1.5)
        self._spin_eps2.setSingleStep(0.05)
        self._spin_eps2.setDecimals(2)
        self._spin_eps2.setValue(0.6)
        self._spin_eps2.setToolTip(
            "ε₂ — east-west roundness (cross-section).\n"
            "1.0 = ellipsoid, < 1 = boxier, > 1 = pinched.")
        info = QtWidgets.QLabel(
            "Shared roundness for all primitives (fixed during the fit). "
            "1.0 = ellipsoid; smaller = rounded-box.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")
        rg.addRow("ε₁ (lengthwise):", self._spin_eps1)
        rg.addRow("ε₂ (cross):", self._spin_eps2)
        rg.addRow(info)

        self._build_common_groups(v)
        return root

    def _setting_specs(self) -> list:
        return [
            ("eps1", self._spin_eps1, "float"),
            ("eps2", self._spin_eps2, "float"),
        ] + self._common_specs()

    def viewer_eps(self):
        self.options_widget()
        return (self._spin_eps1.value(), self._spin_eps2.value())

    def _resolve_eps(self, eps, n):
        # Use the worker's per-primitive eps when provided; otherwise fall back
        # to the UI defaults (e.g. local-fit emits that don't carry eps).
        if eps is not None and len(eps) == n:
            return np.asarray(eps, dtype=np.float32)
        e1, e2 = self.viewer_eps()
        return np.tile(np.array([e1, e2], dtype=np.float32), (int(n), 1))

    def fit_kwargs(self) -> dict:
        self.options_widget()
        return {
            "primitive_shape": "superquadric",
            "sq_eps1": self._spin_eps1.value(),
            "sq_eps2": self._spin_eps2.value(),
            **self._common_fit_kwargs(),
        }


# ── Bent superquadric ─────────────────────────────────────────────────────────

class BentSuperquadricShape(SuperquadricShape):
    id = "bent_superquadric"
    display_name = "Bent superquadric"
    primitive_noun = "bent superquadric"
    available = True

    def _build_options(self) -> QtWidgets.QWidget:
        root = super()._build_options()
        # Add a note: bend is trained per primitive (no slider).
        lbl = QtWidgets.QLabel(
            "Plus a per-primitive bend (curvature), trained automatically — "
            "lets a single primitive curve like a banana / limb.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: gray;")
        root.layout().addWidget(lbl)
        return root

    def _resolve_eps(self, eps, n):
        # The worker packs (N,4) = [e1, e2, kx, ky]; eps is the first two cols.
        if eps is not None and len(eps) == n:
            return np.asarray(eps, dtype=np.float32)[:, :2]
        e1, e2 = self.viewer_eps()
        return np.tile(np.array([e1, e2], dtype=np.float32), (int(n), 1))

    def _resolve_bend(self, eps, n):
        a = np.asarray(eps, dtype=np.float32) if eps is not None else None
        if a is not None and a.shape == (n, 4):
            return a[:, 2:4]
        return np.zeros((int(n), 2), dtype=np.float32)

    def fit_kwargs(self) -> dict:
        self.options_widget()
        return {
            "primitive_shape": "bent_superquadric",
            "sq_eps1": self._spin_eps1.value(),
            "sq_eps2": self._spin_eps2.value(),
            **self._common_fit_kwargs(),
        }


# ── Capsule ───────────────────────────────────────────────────────────────────

class CapsuleShape(_OptimizedPrimitiveShape):
    id = "capsule"
    display_name = "Capsule"
    primitive_noun = "capsule"
    available = True
    render_kind = "capsule"

    def _build_options(self) -> QtWidgets.QWidget:
        root, v = self._new_root()
        info = QtWidgets.QLabel(
            "Capsules: a line segment swept by a sphere (radius + length, "
            "oriented).  Exact SDF — ideal for physics / ragdoll colliders.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")
        v.addWidget(info)
        self._build_common_groups(v)
        return root

    def _setting_specs(self) -> list:
        return self._common_specs()

    def fit_kwargs(self) -> dict:
        self.options_widget()
        return {"primitive_shape": "capsule", **self._common_fit_kwargs()}


# ── Placeholders (not implemented yet — greyed out in the dropdown) ───────────

class _PlaceholderShape(ShapePlugin):
    available = False

    def _build_options(self) -> QtWidgets.QWidget:
        lbl = QtWidgets.QLabel(
            f"{self.display_name} fitting is not available yet.\n"
            f"The infrastructure is ready — this shape can be added as a "
            f"ShapePlugin subclass.")
        lbl.setWordWrap(True)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("color: gray; padding: 18px;")
        return lbl

    def render(self, viewer, centers, radii, rotations) -> None:
        pass


class BoxShape(_PlaceholderShape):
    id = "box"
    display_name = "Rounded box"
    primitive_noun = "box"


# ── Registry ──────────────────────────────────────────────────────────────────

def available_shapes() -> list:
    """All shape plugins, in display order.

    Only :class:`EllipsoidShape` is implemented; the placeholders advertise the
    planned shapes (disabled in the dropdown) so the multi-shape architecture is
    visible.  To add a real shape, implement a ``ShapePlugin`` subclass and list
    it here.
    """
    return [EllipsoidShape(), SphereShape(), SuperquadricShape(),
            BentSuperquadricShape(), CapsuleShape(), BoxShape()]
    # (BoxShape remains a placeholder.)
