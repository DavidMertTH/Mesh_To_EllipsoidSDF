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

from widgets import RangeSlider


# ── Shared widget get/set/observe helpers (used by plugins AND MainWindow) ─────
# ``kind`` ∈ {"bool", "int", "float", "combo_data", "range"}.

def widget_value(w, kind):
    if kind == "bool":
        return w.isChecked()
    if kind == "combo_data":
        return w.currentData()
    if kind == "range":
        return list(w.values())          # [low, high] — JSON-friendly
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
        elif kind == "range":
            lo, hi = int(v[0]), int(v[1])
            w.setValues(lo, hi, emit=True)   # emit → connected labels refresh
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

    @staticmethod
    def _build_window_group(
        v: QtWidgets.QVBoxLayout,
        title: str,
        chk_text: str,
        chk_tip: str,
        window_default: tuple[int, int],
        every_default: int,
        every_tip: str,
        *,
        checked: bool,
    ) -> tuple:
        """Build one uniform *window* control block.

        Layout (shared by Densification and Local fit so they look identical):

            [x] <enable checkbox>
            Window:  ⟨◖────────◗⟩   lo % – hi %
            Every:   [ N steps ]

        The two-handle :class:`RangeSlider` sets *from when* (low %) and *until
        when* (high %) of training the phase runs; the spinbox sets how often
        (every N steps).  Enabling the checkbox enables the slider + spinbox.

        Returns ``(checkbox, range_slider, range_label, every_spin)``.
        """
        form = _OptimizedPrimitiveShape._group(v, title)

        chk = QtWidgets.QCheckBox(chk_text)
        chk.setChecked(checked)
        chk.setToolTip(chk_tip)
        form.addRow(chk)

        lo0, hi0 = window_default
        rng = RangeSlider(0, 100)
        rng.setValues(lo0, hi0)
        rng.setToolTip(
            "Training window: the two handles set from when (left) until when\n"
            "(right) of the run this phase is active, as a fraction of total steps.")
        lbl = QtWidgets.QLabel(f"{lo0} % – {hi0} %")
        lbl.setMinimumWidth(72)
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        win_row = QtWidgets.QWidget()
        win_h = QtWidgets.QHBoxLayout(win_row)
        win_h.setContentsMargins(0, 0, 0, 0)
        win_h.setSpacing(6)
        win_h.addWidget(rng, 1)
        win_h.addWidget(lbl)
        rng.valueChanged.connect(
            lambda lo, hi: lbl.setText(f"{lo} % – {hi} %"))
        form.addRow("Window:", win_row)

        every = QtWidgets.QSpinBox()
        every.setRange(10, 10000)
        every.setValue(every_default)
        every.setSingleStep(10)
        every.setSuffix(" steps")
        every.setToolTip(every_tip)
        form.addRow("Every:", every)

        def _set_enabled(on: bool) -> None:
            rng.setEnabled(on)
            lbl.setEnabled(on)
            every.setEnabled(on)
        _set_enabled(checked)
        chk.toggled.connect(_set_enabled)

        return chk, rng, lbl, every

    @staticmethod
    def _build_window_toggle_group(
        v: QtWidgets.QVBoxLayout,
        title: str,
        chk_text: str,
        chk_tip: str,
        window_default: tuple[int, int],
        *,
        checked: bool,
    ) -> tuple:
        """Build an enable checkbox plus a two-handle training window slider."""
        form = _OptimizedPrimitiveShape._group(v, title)

        chk = QtWidgets.QCheckBox(chk_text)
        chk.setChecked(checked)
        chk.setToolTip(chk_tip)
        form.addRow(chk)

        lo0, hi0 = window_default
        rng = RangeSlider(0, 100)
        rng.setValues(lo0, hi0)
        rng.setToolTip(
            "Training window: the two handles set from when (left) until when\n"
            "(right) of the run this mode is active, as a fraction of total steps.")
        lbl = QtWidgets.QLabel(f"{lo0} % – {hi0} %")
        lbl.setMinimumWidth(72)
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        win_row = QtWidgets.QWidget()
        win_h = QtWidgets.QHBoxLayout(win_row)
        win_h.setContentsMargins(0, 0, 0, 0)
        win_h.setSpacing(6)
        win_h.addWidget(rng, 1)
        win_h.addWidget(lbl)
        rng.valueChanged.connect(
            lambda lo, hi: lbl.setText(f"{lo} % – {hi} %"))
        form.addRow("Window:", win_row)

        def _set_enabled(on: bool) -> None:
            rng.setEnabled(on)
            lbl.setEnabled(on)
        _set_enabled(checked)
        chk.toggled.connect(_set_enabled)

        return chk, rng, lbl

    def _build_common_groups(self, v: QtWidgets.QVBoxLayout) -> None:
        """Build the Densification + Local-fit + Maintenance groups shared by
        all fitted shapes.  Densification and Local fit use the *same* uniform
        window-block layout (enable + range-slider window + frequency)."""
        # ── Densification (adaptive density / SuperFit) ──
        (self._chk_superfit, self._rng_densify, self._lbl_densify,
         self._spin_superfit_every) = self._build_window_group(
            v, "Densification",
            "SuperFit (adaptive density)",
            "Residual-driven growth: periodically maintain the population\n"
            "(merge / spawn / split) and grow it up to Max.  The window sets\n"
            "from when until when of training density control runs; afterwards\n"
            "pure Adam refinement of the fixed population.",
            (0, 75), 150,
            "Density-control rate: every N steps a maintenance round\n"
            "(merge / spawn / split) runs.  Smaller = more frequent.",
            checked=True,
        )

        # ── Local fit (high-res per-region refit) ──
        (self._chk_local_fit, self._rng_local_fit, self._lbl_local_fit,
         self._spin_local_fit_every) = self._build_window_group(
            v, "Local fit",
            "Local Fit",
            "Re-fit the worst regions against a fresh high-res SDF box.\n"
            "The window sets from when until when of training the per-region\n"
            "fit is active.",
            (25, 100), 150,
            "Local-fit rate: every N steps one high-res per-region fit runs\n"
            "inside the window.  Smaller = more frequent.",
            checked=False,
        )

        # ── Soft union (experimental) ──
        su = self._group(v, "Union")
        self._chk_soft_union = QtWidgets.QCheckBox("Soft min (experimental)")
        self._chk_soft_union.setChecked(False)
        self._chk_soft_union.setToolTip(
            "⚠ Experimental — OFF by default.\n"
            "Soft (LogSumExp) union of the primitive SDFs during the densify phase\n"
            "(gradient spread across several nearby primitives → denser gradients).\n"
            "In tests the result tended to look WORSE than the hard min;\n"
            "only enable it and compare on your own mesh.")
        su.addRow(self._chk_soft_union)

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
        self._chk_prune = QtWidgets.QCheckBox("Prune")
        self._chk_prune.setChecked(True)
        self._chk_prune.setToolTip(
            "Remove redundant primitives: fuse fully-overlapping ones and drop\n"
            "those whose coverage is already provided by their neighbours.\n"
            "Safety deletes (degenerate / far-outside primitives) stay on.")
        mv.addRow(self._chk_merge)
        mv.addRow(self._chk_spawn)
        mv.addRow(self._chk_split)
        mv.addRow(self._chk_prune)

    def _common_specs(self) -> list:
        return [
            ("superfit",        self._chk_superfit,         "bool"),
            ("superfit_every",  self._spin_superfit_every,  "int"),
            ("densify_window",  self._rng_densify,          "range"),
            ("local_fit",       self._chk_local_fit,        "bool"),
            ("local_fit_every", self._spin_local_fit_every, "int"),
            ("local_fit_window", self._rng_local_fit,       "range"),
            ("soft_union",      self._chk_soft_union,       "bool"),
            ("merge",           self._chk_merge,            "bool"),
            ("spawn",           self._chk_spawn,            "bool"),
            ("split",           self._chk_split,            "bool"),
            ("prune",           self._chk_prune,            "bool"),
        ]

    def _common_fit_kwargs(self) -> dict:
        d_lo, d_hi = self._rng_densify.values()
        l_lo, l_hi = self._rng_local_fit.values()
        return {
            "superfit":             self._chk_superfit.isChecked(),
            "superfit_every":       self._spin_superfit_every.value(),
            "densify_start_frac":   d_lo / 100.0,
            "densify_until_frac":   d_hi / 100.0,
            "local_fit":            self._chk_local_fit.isChecked(),
            "local_fit_every":      self._spin_local_fit_every.value(),
            "local_fit_start_frac": l_lo / 100.0,
            "local_fit_end_frac":   l_hi / 100.0,
            "soft_union":           self._chk_soft_union.isChecked(),
            "merge_enabled":        self._chk_merge.isChecked(),
            "spawn_enabled":        self._chk_spawn.isChecked(),
            "split_enabled":        self._chk_split.isChecked(),
            "prune_enabled":        self._chk_prune.isChecked(),
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

        # ── Roundness ──
        rg = self._group(v, "Roundness  (ε)")
        self._combo_eps_mode = QtWidgets.QComboBox()
        self._combo_eps_mode.addItem("Per primitive (trainable)", "per_primitive")
        self._combo_eps_mode.addItem("Shared (trainable)", "shared")
        self._combo_eps_mode.addItem("Fixed", "fixed")
        self._combo_eps_mode.setToolTip(
            "Controls how the two roundness exponents are fitted.\n"
            "Per primitive gives every primitive its own ε₁/ε₂ pair; shared "
            "learns one pair for the whole population; fixed keeps the initial "
            "values unchanged.")
        self._spin_eps1 = QtWidgets.QDoubleSpinBox()
        self._spin_eps1.setRange(0.1, 2.0)
        self._spin_eps1.setSingleStep(0.05)
        self._spin_eps1.setDecimals(2)
        self._spin_eps1.setValue(0.6)
        self._spin_eps1.setToolTip(
            "ε₁ — north-south roundness.\n"
            "1.0 = ellipsoid, < 1 = boxier (sharper edges), > 1 = pinched.")
        self._spin_eps2 = QtWidgets.QDoubleSpinBox()
        self._spin_eps2.setRange(0.1, 2.0)
        self._spin_eps2.setSingleStep(0.05)
        self._spin_eps2.setDecimals(2)
        self._spin_eps2.setValue(0.6)
        self._spin_eps2.setToolTip(
            "ε₂ — east-west roundness (cross-section).\n"
            "1.0 = ellipsoid, < 1 = boxier, > 1 = pinched.")
        self._spin_eps_warmup = QtWidgets.QSpinBox()
        self._spin_eps_warmup.setRange(0, 80)
        self._spin_eps_warmup.setSingleStep(5)
        self._spin_eps_warmup.setSuffix(" %")
        self._spin_eps_warmup.setValue(20)
        self._spin_eps_warmup.setToolTip(
            "Fraction of training spent fitting centres, radii and rotations "
            "before trainable ε values are unlocked.")
        self._spin_bend_warmup = QtWidgets.QSpinBox()
        self._spin_bend_warmup.setRange(0, 90)
        self._spin_bend_warmup.setSingleStep(5)
        self._spin_bend_warmup.setSuffix(" %")
        self._spin_bend_warmup.setValue(40)
        self._spin_bend_warmup.setToolTip(
            "Bent superquadrics unlock curvature after this fraction of the run.")
        self._bend_warmup_row_label = QtWidgets.QLabel("Bend warm-up:")
        self._bend_warmup_row_label.setVisible(False)
        self._spin_bend_warmup.setVisible(False)

        info = QtWidgets.QLabel(
            "The ε values above are the initial values.  1.0 is an ellipsoid; "
            "smaller values produce rounded boxes and larger values produce "
            "pinched shapes.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")
        rg.addRow("Mode:", self._combo_eps_mode)
        rg.addRow("ε₁ (lengthwise):", self._spin_eps1)
        rg.addRow("ε₂ (cross):", self._spin_eps2)
        rg.addRow("Shape warm-up:", self._spin_eps_warmup)
        rg.addRow(self._bend_warmup_row_label, self._spin_bend_warmup)
        rg.addRow(info)

        self._build_common_groups(v)
        return root

    def _setting_specs(self) -> list:
        return [
            ("eps_mode", self._combo_eps_mode, "combo_data"),
            ("eps1", self._spin_eps1, "float"),
            ("eps2", self._spin_eps2, "float"),
            ("eps_warmup", self._spin_eps_warmup, "int"),
            ("bend_warmup", self._spin_bend_warmup, "int"),
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
            "sq_eps_mode": self._combo_eps_mode.currentData(),
            "sq_unlock_frac": self._spin_eps_warmup.value() / 100.0,
            "sq_bend_unlock_frac": self._spin_bend_warmup.value() / 100.0,
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
        self._bend_warmup_row_label.setVisible(True)
        self._spin_bend_warmup.setVisible(True)
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
            "sq_eps_mode": self._combo_eps_mode.currentData(),
            "sq_unlock_frac": self._spin_eps_warmup.value() / 100.0,
            "sq_bend_unlock_frac": self._spin_bend_warmup.value() / 100.0,
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
