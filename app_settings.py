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

with ``kind`` ``"bool"`` (→ QCheckBox), ``"int"`` (→ QSpinBox), or
``"float"`` (→ QDoubleSpinBox).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_FILE = Path(__file__).with_name("app_settings.json")


# ── The spec — single source of truth for the dialog AND the defaults ─────────
SETTINGS_SPEC = [
    ("Maintenance", [
        ("Regional population budget", [
            ("size_region_budget_enabled", "Limit ellipsoids per mesh region", "bool",
             0, 1, 1, 0, True,
             "Partition the mesh into coloured surface regions and enforce a\n"
             "size-dependent hard cap on spawn and split operations."),
            ("size_region_target_capacity", "Target ellipsoids per region", "int",
             2, 24, 1, 0, 6,
             "Controls the partition granularity. Lower values create more,\n"
             "smaller regions with tighter local population caps."),
            ("size_region_min_capacity", "Minimum per region", "int",
             1, 8, 1, 0, 2,
             "Minimum capacity reserved for every visible mesh region."),
            ("size_region_area_power", "Size weighting", "float",
             0.1, 1.0, 0.05, 2, 0.65,
             "How strongly region surface area controls its cap. Values below\n"
             "1 give small regions proportionally more capacity; 1 is linear."),
        ]),
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
             "How many 60%-of-local-thickness growth seeds may be spawned\n"
             "directly at under-represented regions per maintenance cycle."),
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
             "Maximum long-axis / median-axis ratio. Superquadrics above this\n"
             "are prioritised for splitting; other primitive types are deleted."),
        ]),
    ]),

    ("Local fit", [
        ("High-res region (SuperFit)", [
            ("region_radius_vox", "Region radius (vox)", "float", 1.0, 30.0, 0.5, 1, 6.0,
             "Base/minimum half-extent of a local-fit box, measured in global\n"
             "SDF voxels. The actual box may be larger for a split primitive\n"
             "and always includes a safety/blowup margin. All boxes use the\n"
             "same Region resolution, so a larger box has less spatial detail."),
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
        ("Final result", [
            ("use_best_validation_result", "Use best validation result", "bool",
             0, 1, 1, 0, True,
             "Restore the checkpoint with the lowest validation loss when the\n"
             "fit finishes. Disable this to keep the final optimiser state,\n"
             "including changes made by the last local fit."),
        ]),
    ]),

    ("Pose refit", [
        ("Allowed changes after the base fit", [
            ("pose_fit_position", "Refit position", "bool", 0, 1, 1, 0, True,
             "Allow primitive positions to be refined during /fit-pose.\n"
             "With bone-local fitting, locked primitives still follow their\n"
             "bones. The initial base fit is unaffected."),
            ("pose_fit_rotation", "Refit rotation", "bool", 0, 1, 1, 0, True,
             "Allow primitive rotations to be refined during /fit-pose.\n"
             "With bone-local fitting, locked primitives still rotate with\n"
             "their bones. The initial base fit is unaffected."),
            ("pose_fit_scale", "Refit scale", "bool", 0, 1, 1, 0, True,
             "Allow primitive radii (scale) to change during subsequent\n"
             "/fit-pose operations. Superquadric shape exponents and bend\n"
             "remain controlled separately; the base fit is unaffected."),
        ]),
    ]),

    ("Loss", [
        ("Surface & penalties", [
            ("surface_weight", "Surface weight", "float", 0.0, 20.0, 0.5, 1, 4.0,
             "Extra weight on samples near the zero level set (the surface)."),
            ("surface_sigma_vox", "Surface sigma (vox)", "float", 0.2, 10.0, 0.1, 1, 1.5,
             "Width (in voxels) of the surface-emphasis Gaussian."),
            ("normal_loss_weight", "Normal loss weight", "float", 0.0, 10.0, 0.1, 1, 1.0,
             "Match approximation normals to mesh normals near the surface.\n"
             "0 disables the normal loss."),
            ("normal_band_vox", "Normal band (vox)", "float", 0.5, 6.0, 0.25, 2, 2.0,
             "Half-width of the target-SDF band used for normal matching."),
            ("normal_warmup_frac", "Normal warm-up", "float", 0.0, 1.0, 0.05, 2, 0.20,
             "Fraction of training used for SDF placement before normal matching starts."),
            ("normal_ramp_frac", "Normal ramp", "float", 0.0, 1.0, 0.05, 2, 0.20,
             "Fraction of training over which the normal-loss weight reaches full strength."),
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
             "Additional inverse-thickness loss boost, applied on top of the\n"
             "sampling distribution. Set to 0 for sampling-only emphasis."),
            ("thin_max_factor", "Thin max factor", "float", 1.0, 20.0, 0.5, 1, 6.0,
             "Cap on the thin-feature loss boost."),
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
            ("thickness_sampling_power", "Thickness sampling power", "float",
             0.0, 2.0, 0.10, 2, 1.00,
             "Continuously weight surface sampling by local feature thickness.\n"
             "0 = area-uniform, 1 = probability proportional to 1 / thickness,\n"
             "2 = stronger 1 / thickness² weighting. There is no thin/thick split.\n"
             "Thin loss weight can independently add further emphasis."),
            ("coverage_sample_size", "Coverage sample size", "int", 1000, 100000, 1000, 0, 20000,
             "Number of samples used to estimate coverage / under-representation."),
        ]),
    ]),

    ("Advanced", [
        ("Learning-rate multipliers", [
            ("lr_mult_centers", "LR × position", "float", 0.0, 10.0, 0.5, 1, 1.0,
             "Per-group learning-rate multiplier for primitive centres."),
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
        migrated = False
        # Older settings files predate selectable final-checkpoint restore.
        # Persist the historical behaviour explicitly so reopening the dialog
        # exposes the enabled checkbox without changing existing fits.
        if "use_best_validation_result" not in data:
            data["use_best_validation_result"] = values[
                "use_best_validation_result"]
            migrated = True
        # The short-lived ``thin_surface_fraction`` setting used a binary
        # thin/regular quota.  Its numeric value now migrates to the continuous
        # inverse-thickness power so an explicit 1.0 becomes exactly 1/t.  The
        # older bias mapped 1.0 to a 30 % quota, so retain that approximate
        # strength as power 0.3 for legacy configurations.
        if "thickness_sampling_power" not in data:
            try:
                if "thin_surface_fraction" in data:
                    legacy = float(data["thin_surface_fraction"])
                elif "thin_sample_bias" in data:
                    legacy = 0.3 * float(data["thin_sample_bias"])
                else:
                    legacy = values["thickness_sampling_power"]
                if not math.isfinite(legacy):
                    raise ValueError("non-finite thickness sampling setting")
                data["thickness_sampling_power"] = max(
                    0.0, min(2.0, legacy))
            except (TypeError, ValueError):
                data["thickness_sampling_power"] = values[
                    "thickness_sampling_power"]
            migrated = migrated or bool(
                "thin_surface_fraction" in data or "thin_sample_bias" in data)
        for legacy_key in ("thin_surface_fraction", "thin_sample_bias"):
            if legacy_key in data:
                del data[legacy_key]
                migrated = True
        try:
            sampling_power = float(data["thickness_sampling_power"])
            if not math.isfinite(sampling_power):
                raise ValueError("non-finite thickness_sampling_power")
            sampling_power = max(0.0, min(2.0, sampling_power))
        except (KeyError, TypeError, ValueError):
            sampling_power = values["thickness_sampling_power"]
        if data.get("thickness_sampling_power") != sampling_power:
            data["thickness_sampling_power"] = sampling_power
            migrated = True
        for key in values:
            if key in data:
                values[key] = data[key]
        if migrated:
            try:
                _FILE.write_text(json.dumps(data, indent=2, sort_keys=True),
                                 encoding="utf-8")
            except Exception:
                pass
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
PANEL_SETTINGS_SCHEMA_VERSION = 3


def _migrate_panel(data: dict) -> tuple[dict, bool]:
    """Apply conservative one-time migrations to persisted panel values."""
    try:
        version = int(data.get("schema_version", 0))
    except (TypeError, ValueError):
        version = 0
    changed = False

    if version < 2:
        shapes = data.get("shapes")
        if isinstance(shapes, dict):
            for shape_id in ("superquadric", "bent_superquadric"):
                state = shapes.get(shape_id)
                if not isinstance(state, dict):
                    continue
                try:
                    legacy_defaults = (
                        float(state.get("eps1")) == 0.6
                        and float(state.get("eps2")) == 0.6
                        and int(state.get("eps_warmup")) == 20
                    )
                except (TypeError, ValueError):
                    legacy_defaults = False
                if legacy_defaults:
                    state["eps1"] = 1.0
                    state["eps2"] = 1.0
                    state["eps_warmup"] = 5
                    changed = True
        data["schema_version"] = 2
        changed = True

    if version < 3:
        shared = data.get("shared")
        if isinstance(shared, dict):
            if "blowup_fraction" not in shared and "blowup" in shared:
                try:
                    # Preserve the former slider's relative position:
                    # +/-10 vox (old full range) becomes +/-25% local diameter.
                    old = float(shared["blowup"])
                    shared["blowup_fraction"] = (
                        max(-1.0, min(1.0, old / 10.0)) * 0.25)
                except (TypeError, ValueError):
                    shared["blowup_fraction"] = 0.0
                changed = True
            if "blowup" in shared:
                del shared["blowup"]
                changed = True
        data["schema_version"] = 3
        changed = True

    return data, changed


def load_panel() -> dict:
    """The persisted options-panel values (empty dict if none saved yet)."""
    try:
        data = json.loads(_PANEL_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    data, changed = _migrate_panel(data)
    if changed:
        save_panel(data)
    return data


def save_panel(values: dict) -> None:
    """Persist the options-panel values to disk (best effort)."""
    try:
        payload = dict(values)
        payload["schema_version"] = PANEL_SETTINGS_SCHEMA_VERSION
        _PANEL_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True),
                               encoding="utf-8")
    except Exception:
        pass
