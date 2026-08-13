"""app_settings.py — persistent *advanced* fitting / maintenance settings.

These are the detailed knobs that would clutter the main window: SuperFit
maintenance limits (how many ellipsoids merge / spawn / split per cycle), the
high-res local-fit resolution, the loss weights, the sampling budget, etc.

The values are stored as one flat ``{key: value}`` dict and persisted to
``app_settings.json`` next to this file.  Most keys are forwarded into
``OptimizationWorker``; global SDF/UI keys (for example sparse SDF and
thickness resolution) are consumed by the window before worker creation.

``SETTINGS_SPEC`` is a 3-level structure that drives the settings dialog UI:

    tab title → [ (group-box title, [field, ...]), ... ]

Each *field* is a tuple::

    (key, label, kind, minimum, maximum, step, decimals, default, tooltip)

with ``kind`` either ``"int"`` (→ QSpinBox) or ``"float"`` (→ QDoubleSpinBox).
"""

from __future__ import annotations

import json
from pathlib import Path

_FILE = Path(__file__).with_name("app_settings.json")


# ── The spec — single source of truth for the dialog AND the defaults ─────────
SETTINGS_SPEC = [
    ("Maintenance", [
        ("Merge", [
            ("merge_per_round", "Merges per cycle", "int", 0, 50, 1, 0, 3,
             "How many overlapping ellipsoid pairs may be fused into one per\n"
             "SuperFit maintenance cycle."),
            ("merge_tol", "Merge tolerance", "float", 0.0, 1.0, 0.01, 2, 0.12,
             "Allowed change of the union surface when fusing two ellipsoids.\n"
             "Higher = more aggressive merging."),
        ]),
        ("Spawn", [
            ("spawn_per_round", "Spawns per cycle", "int", 0, 50, 1, 0, 3,
             "How many new ellipsoids may be spawned into under-represented\n"
             "gaps per maintenance cycle."),
        ]),
        ("Split", [
            ("split_per_round", "Splits per cycle", "int", 0, 50, 1, 0, 7,
             "How many oversized / bridging ellipsoids may be split per cycle."),
            ("split_size_factor", "Oversize factor", "float", 1.0, 3.0, 0.05, 2, 1.2,
             "An ellipsoid is considered oversized (split candidate) when it is\n"
             "this many times larger than the local target scale."),
            ("split_margin_vox", "Split margin (vox)", "float", 0.0, 5.0, 0.1, 1, 0.5,
             "Voxel margin used when separating the two halves of a split."),
            ("min_split_radius_vox", "Min split radius (vox)", "float", 0.5, 10.0, 0.5, 1, 2.0,
             "Ellipsoids smaller than this (in voxels) are never split."),
            ("bridge_min_outside", "Bridge min outside", "float", 0.0, 1.0, 0.05, 2, 0.1,
             "Minimum fraction of an ellipsoid that must stick outside the mesh\n"
             "for it to count as a 'bridge' worth splitting."),
        ]),
        ("Fuse (divide & conquer)", [
            ("fuse_per_round", "Fuses per cycle", "int", 0, 50, 1, 0, 2,
             "How many fuse operations the divide-and-conquer pass may do per cycle."),
            ("fuse_overlap_frac", "Fuse overlap fraction", "float", 0.0, 1.0, 0.05, 2, 0.9,
             "Required overlap fraction before two ellipsoids are fused."),
            ("fuse_samples", "Fuse samples", "int", 8, 512, 8, 0, 96,
             "Number of unit-ball sample points used to estimate fuse overlap."),
        ]),
        ("Pruning", [
            ("max_prune_fraction", "Max prune fraction / round", "float", 0.0, 0.5, 0.01, 2, 0.15,
             "At most this fraction of the population may be pruned per round\n"
             "(keeps training stable)."),
            ("degenerate_flat_ratio", "Degenerate flat ratio", "float", 0.0, 1.0, 0.01, 2, 0.12,
             "Hard-delete an ellipsoid that collapsed into a flat disk:\n"
             "min-axis / median-axis below this ratio."),
            ("degenerate_spike_ratio", "Degenerate spike ratio", "float", 1.0, 20.0, 0.5, 1, 8.0,
             "Hard-delete an over-pointy spike: max-axis / median-axis above this."),
        ]),
    ]),

    ("Local fit", [
        ("High-res region (SuperFit)", [
            ("region_radius_vox", "Region radius (vox)", "float", 1.0, 30.0, 0.5, 1, 6.0,
             "Half-extent (in voxels) of the local box re-fitted around each\n"
             "maintained region."),
            ("region_res", "Region resolution", "int", 32, 256, 16, 0, 128,
             "Voxel resolution of the fresh high-res SDF box used for local fit."),
            ("region_steps", "Region steps", "int", 100, 20000, 100, 0, 400,
              "Total Adam steps spent per region local fit."),
            ("region_dc_cycles", "Divide & conquer cycles", "int", 1, 10, 1, 0, 3,
             "Number of divide-and-conquer passes within one region fit."),
            ("local_steps", "Local steps", "int", 100, 10000, 100, 0, 400,
              "Minimum Adam steps for an isolated local fit."),
            ("local_lr", "Local learning rate", "float", 0.0001, 0.5, 0.0001, 4, 0.001,
              "Learning rate used while fitting a newly spawned region in isolation."),
            ("local_center_trust_radius_factor", "Center trust radius", "float",
             0.0, 4.0, 0.05, 2, 0.75,
             "Maximum cumulative centre movement during one local fit, as a\n"
             "fraction of the ellipsoid's starting mean radius. 0 disables it."),
            ("local_radii_trust_factor", "Radii trust factor", "float",
             1.0, 4.0, 0.05, 2, 1.5,
             "Maximum factor by which each radius may grow or shrink during\n"
             "one local fit (symmetric in log space)."),
        ]),
    ]),

    ("Loss", [
        ("Surface & penalties", [
            ("surface_weight", "Surface weight", "float", 0.0, 20.0, 0.5, 1, 4.0,
             "Extra weight on samples near the zero level set (the surface)."),
            ("surface_sigma_vox", "Surface sigma (vox)", "float", 0.2, 10.0, 0.1, 1, 1.5,
             "Width (in voxels) of the surface-emphasis Gaussian."),
            ("miss_penalty_weight", "Miss penalty", "float", 0.0, 30.0, 0.5, 1, 3.0,
             "Penalty when the target is inside the mesh but the ellipsoids miss it."),
            ("outside_penalty_weight", "Protrusion penalty", "float", 0.0, 50.0, 0.5, 1, 14.0,
             "Quadratic per-location penalty when an ellipsoid bulges out past the true surface."),
            ("containment_weight", "Containment weight", "float", 0.0, 30.0, 0.5, 1, 6.0,
             "Pull an ellipsoid whose centre drifts outside the mesh back inside."),
            ("flat_weight", "Flatness penalty", "float", 0.0, 10.0, 0.1, 2, 0.5,
             "Penalise ellipsoids collapsing into flat disks (needles stay free)."),
            ("flat_min_ratio", "Flatness min ratio", "float", 0.0, 1.0, 0.05, 2, 0.35,
             "Min allowed min-axis / median-axis ratio before the flatness penalty bites."),
        ]),
        ("Thin features", [
            ("thin_loss_weight", "Thin loss weight", "float", 0.0, 10.0, 0.1, 1, 1.0,
             "Boost the loss on thin structures (inverse local thickness)."),
            ("thin_max_factor", "Thin max factor", "float", 1.0, 20.0, 0.5, 1, 6.0,
             "Cap on the thin-feature loss boost."),
            ("thin_sample_bias", "Thin sample bias", "float", 0.0, 3.0, 0.1, 1, 1.0,
             "Reserve more of the surface-band samples for thin features\n"
             "(1.0 = default 30 %, 0 = off)."),
        ]),
        ("Under-representation", [
            ("underrep_rel_threshold", "Detection threshold", "float", 0.05, 2.0, 0.05, 2, 0.6,
             "Relative miss (gap / local thickness) a region must reach to count\n"
             "as under-represented (drives spawn/split + the overlay).\n"
             "Lower = more sensitive (flags smaller gaps), higher = only gross misses."),
            ("underrep_min_gap_vox", "Min gap (vox)", "float", 0.0, 3.0, 0.1, 1, 0.5,
             "Absolute miss floor in voxels — kills sub-voxel surface noise.\n"
             "Lower = more sensitive."),
            ("underrep_min_thickness_vox", "Min feature thickness (vox)", "float", 0.0, 12.0, 0.5, 1, 4.0,
             "Floor (in voxels) on the local feature thickness used as the\n"
             "relative-miss scale.  Thin structures (fingers) below the grid\n"
             "resolution otherwise inflate the relative error without bound, so\n"
             "spawn fires endlessly and carpets them with tiny spheres.  Higher =\n"
             "fewer tiny ellipsoids on thin features; 0 = old (unbounded) behaviour."),
        ]),
    ]),

    ("Sampling", [
        ("SDF target", [
            ("use_sparse_sdf", "Use sparse SDF samples", "bool", 0, 1, 1, 0, True,
             "Train the optimiser on sparse surface-focused SDF samples.\n"
             "Disable to use the full dense SDF grid for the loss."),
            ("thickness_max_resolution", "Thickness max resolution", "int", 0, 512, 16, 0, 128,
             "Compute the expensive local-thickness field at at most this\n"
             "longest-axis resolution and upsample it to the SDF grid.\n"
             "0 = full-resolution thickness."),
        ]),
        ("Batch sampling", [
            ("sample_budget", "Sample budget (voxels)", "int", 64, 262144, 64, 0, 4096,
             "Voxels sampled per optimisation step — resolution-independent cost."),
            ("surface_band_vox", "Surface band (vox)", "float", 0.5, 10.0, 0.5, 1, 3.0,
             "Half-width (in voxels) of the surface band for importance sampling."),
            ("surface_fraction", "Surface fraction", "float", 0.0, 1.0, 0.05, 2, 0.75,
             "Fraction of each batch drawn from the surface band."),
            ("coverage_sample_size", "Coverage sample size", "int", 1000, 100000, 1000, 0, 20000,
             "Number of samples used to estimate coverage / under-representation."),
        ]),
    ]),

    ("Advanced", [
        ("Learning-rate multipliers", [
            ("lr_mult_radii", "LR × radii", "float", 0.0, 10.0, 0.5, 1, 2.0,
             "Per-group learning-rate multiplier for the (log-space) radii."),
            ("lr_mult_rot", "LR × rotation", "float", 0.0, 10.0, 0.5, 1, 1.0,
             "Per-group learning-rate multiplier for the rotations."),
            ("center_step_radius_frac", "Center step radius frac", "float", 0.0, 2.0, 0.05, 2, 0.5,
             "Limit one centre update to this fraction of the ellipsoid's\n"
             "mean radius. 0 disables the limiter."),
            ("center_step_min_vox", "Center step min (vox)", "float", 0.0, 5.0, 0.05, 2, 0.25,
             "Minimum allowed centre movement per Adam step, in voxels."),
            ("center_step_max_vox", "Center step max (vox)", "float", 0.1, 20.0, 0.1, 1, 4.0,
             "Maximum allowed centre movement per Adam step, in voxels."),
        ]),
        ("Symmetry", [
            ("symmetry_every", "Symmetry every (steps)", "int", 1, 1000, 10, 0, 100,
             "Re-project onto the mirror plane every N steps (when symmetry is on)."),
        ]),
        ("Bone-awareness", [
            ("bone_span_weight", "Bone span weight", "float", 0.0, 5.0, 0.1, 2, 0.4,
             "Strength of the penalty for ellipsoids spanning multiple bones."),
            ("bone_span_tol", "Bone span tolerance", "float", 0.0, 2.0, 0.05, 2, 0.35,
             "Slack before the multi-bone penalty starts (1 bone + tol is free)."),
            ("bone_span_soft", "Bone span softness", "float", 0.01, 1.0, 0.01, 2, 0.15,
             "Softness of the smooth bone-membership transition."),
        ]),
    ]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iter_fields():
    for _tab, groups in SETTINGS_SPEC:
        for _grp, fields in groups:
            for f in fields:
                yield f


def defaults() -> dict:
    """The default value for every setting key."""
    return {f[0]: f[7] for f in _iter_fields()}


def load() -> dict:
    """Current settings — defaults overlaid with any persisted values."""
    values = defaults()
    try:
        data = json.loads(_FILE.read_text(encoding="utf-8"))
    except Exception:
        return values
    if isinstance(data, dict):
        for key in values:
            if key in data:
                values[key] = data[key]
    return values


def save(values: dict) -> None:
    """Persist the given settings dict to disk (best effort)."""
    try:
        _FILE.write_text(json.dumps(values, indent=2, sort_keys=True),
                         encoding="utf-8")
    except Exception:
        pass


# ── Right-hand options panel (the basic controls) ─────────────────────────────
# Persisted separately from the advanced dialog settings: a free-form
# {key: value} dict of the main window's option widgets (margin, ellipsoid
# counts, SuperFit toggles, learning rate, …) so they survive across sessions.
_PANEL_FILE = Path(__file__).with_name("panel_settings.json")


def load_panel() -> dict:
    """The persisted options-panel values (empty dict if none saved yet)."""
    try:
        data = json.loads(_PANEL_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_panel(values: dict) -> None:
    """Persist the options-panel values to disk (best effort)."""
    try:
        _PANEL_FILE.write_text(json.dumps(values, indent=2, sort_keys=True),
                               encoding="utf-8")
    except Exception:
        pass
