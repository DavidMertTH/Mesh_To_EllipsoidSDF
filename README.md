# Mesh_To_EllipsoidSDF

## Detail zones for thin features

A fit measures everything in voxels of one SDF grid over the whole mesh. A
finger thinner than a voxel is fitted poorly in a whole-body fit, although the
same hand fits well when loaded on its own. **Detail zones** do this
automatically and in parallel:

1. Before the fit, the local feature diameter is measured at every vertex with
   exact mesh-distance queries: the largest ball that fits inside the surface
   at that vertex, independent of the grid. Connected regions thinner than
   **Zone threshold** voxels (default 1.5) become zones, for a humanoid the
   hands and the toes. Small specks are ignored, and mirror-symmetric zones are
   paired when **Symmetry** is on.
2. The primitive budget is split up front. Each region gets primitives in
   proportion to area / diameter^1.25, scaled for the zones by **Zone budget
   weight**. The main fit gets the maximum minus the zone budgets. It starts
   with the initial count, but 10 % below that maximum, so SuperFit can still
   add primitives where they are needed.
3. Each zone's core is removed from the main fit's target, as the CSG
   difference "mesh minus core box". The main fit places no primitives there.
   Where main-fit surface continues into a core (wrist, ankle), the interior
   within three voxels of that surface is left free. The main fit's primitives
   can then run past the seam and overlap the zone's primitives, so no gap
   remains. Sticking out of the skin is still penalized there.
4. At the same time as the main fit, every zone is fitted in its own process on
   the same GPU. It uses the zone's cropped mesh in its own normalized frame,
   on a grid sized so the zone's median feature spans **Zone voxels per
   feature** voxels (default 8, up to **Zone max grid n**), with the normal fit
   settings. A mirrored pair is fitted once and reflected. The viewport shows
   the main fit and the zones together while they train. After its fit, each
   zone checks how much of its surface has no primitive just below the skin.
   Above 4 %, for example a whole finger left out, the zone is refitted once
   with 50 % more primitives, and the better result is kept.
5. The zone primitives are added to the main fit's primitives. Bone assignment
   and morph fitting then continue as usual.

Separate processes are needed because Warp's autodiff tape is process-wide.
The zones are smaller than the main fit, so they normally finish first and the
total time is about that of the main fit alone. If a zone process fails, that
zone is refitted in the main process before merging, so no region is left
without primitives. **Stop** ends the zones and keeps the main fit's result.

The options are under **Settings → Local fit → Detail zones (thin features)**.
Switching off **Fit zones in parallel processes** fits the zones one after
another after the main fit. Detail zones are skipped for Unity/API jobs,
fixed-population pose fits and an active SDF blowup.

## Live pose-fitting preview

**Settings → Pose refit → Live preview and size changes** enables intermediate
ellipsoid previews, including Unity pose batches. The preview is enabled by
default; disable it for faster batch processing.

Size colors compare each ellipsoid's volume (`rx * ry * rz`) to the base
configuration, preserving primitive IDs. By default, gray means unchanged
within ±1% of base volume. Outside that tolerance, the displayed size change
continuously interpolates from gray toward orange (growing) or blue (shrinking),
reaching full colour over an additional 25% volume change. Morph transitions
use the currently displayed size, so intermediate shapes retain intermediate
colours. **Color transition span (%)** controls this range in Settings.
The colors, transition span, neutral tolerance, and color toggle are configurable
and saved across sessions. Colors also apply when viewing trained corrective poses and
in raymarch mode. Selecting a quality heatmap in the viewer takes precedence.

The standard lit primitive material uses a coloured diffuse body with compact
white specular highlights for a glossy plastic appearance. Raymarch uses the
same material parameters.

The 3-D viewer builds primitive surfaces on the GPU. A unit icosphere is
uploaded once. Each primitive is one instance holding its centre, radii,
rotation, superquadric exponents, bend and colour. The vertex shader turns the
icosphere into the ellipsoid, superquadric, bent superquadric or capsule
surface and computes its analytic normal. An update during fitting or animation
therefore only replaces 72 bytes per primitive. All primitives are drawn in one
instanced call in the lighting, solid, transparent and wireframe modes; the
densify-analysis spheres use the same path. Raymarch keeps its separate
sphere-traced image.

## Joint controls in the 3-D viewer

For a loaded rig, **Show → Joint** toggles joint markers independently of the
skeleton lines. Hover a marker to see its bone-local X/Y/Z rotation axes and
allowed rotation arcs (red/green/blue). Click it to pin the gizmo and open a
compact editor. In **Rotation**, use the X/Y/Z sliders or angle fields to rotate
the joint live. The skinned mesh, child joints and bone-assigned ellipsoids
follow immediately. **Limits** enables/locks each axis and sets its minimum
and maximum angular offset in degrees. Semi-transparent pie sectors show the allowed angular
ranges; dashed rings mark the locked or excluded ranges.

You can also select any bone via **Mesh → Rig · Morph Poses → Joint →
Rotate…**, including small finger joints. Angles are measured from the local
T-pose/bind pose and manual changes respect the configured limits; locked axes
cannot be rotated. **Reset joint** restores that joint's source pose;
**Reset pose edits** restores all edited joints at the current frame. Edits stay
with their timeline frame while the model is loaded and are only a preview; they
are not saved. Merely selecting a joint does not clamp an imported animation
that exceeds the authoring limits.

On a pinned joint, drag a coloured limit handle to change the minimum or
maximum angle directly in its local rotation plane. The inner grip with a
white centre marks the minimum; the outer grip marks the maximum. Handles
also remain available at a full 360° or a zero-width range. Numeric fields
update during the drag. When looking along a rotation plane, drag vertically
to change the angle. Esc cancels the current drag and restores its start value.

Click another joint to switch selection. Click empty space, press Esc, or use
the editor's close button to release it. Camera dragging still works when a
drag starts on a joint. Joint orientation follows the displayed pose.

The bundled **T-Pose**, **Dancing** and **Fight Idle** Mixamo rigs now load the
authored [T-Pose joint profile](assets/joint_limits/t_pose.json). Equivalent
namespaces and bind-axis permutations are supported. Limits are measured from
the bind pose, with mirrored left/right axes accounted for; they are practical
viewer/animation-authoring allowances rather than limits extracted from FBX.

| Joint | Rotational DOF | Main range |
| --- | ---: | --- |
| Shoulder / upper arm | 3 | elevation ±95°, twist ±85° |
| Elbow / forearm | 2 | flexion 0–145°, forearm rotation ±80° |
| Wrist | 2 | flexion 75°, extension 60°, deviation 20° / 35° |
| Hip | 3 | flexion 125°, extension 20° |
| Knee | 1 | flexion 0–140°; sideways and axial rotation locked |
| Ankle | 2 | dorsiflexion 20°, plantarflexion 55° |
| Finger base | 2 | flexion up to 90°, spread ±20° |
| Finger middle / distal | 1 | flexion up to 105° / 80° |

Spine and neck motion is distributed across their joints. Thumb joints have
separate ranges, and the 13 finger/toe/head end markers are locked. The hips
root retains free orientation. The limb flexion ranges are rounded practical
choices informed by [CDC joint-ROM reference values](https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index_1715172647.html).

These defaults are restored on model load. Manual edits stay on the loaded
skeleton across pose changes and take precedence over preset reapplication;
they are not saved to disk. Unrecognized rigs or uncertain axis matches retain
their existing allowances. Limits apply to manual rotations and native joint-range
pose generation; imported animations are not automatically clamped.

## Native systematic pose fitting and morph preview

For a rigged model, the **Mesh** tab shows **Rig · Morph Poses**. Its poses
always come from the joint ranges: frame 0 is the bind/base pose, followed by
nearby poses generated from the current joint limits. It works with arbitrary
skeletons and local X/Y/Z ranges, without a humanoid Avatar, FBX animation or
Unity connection. After the base pose come parallel sweeps: each joint samples
one enabled axis at a time and steps outward through its angles in order
(shorter direction first), so consecutive poses change each joint by one angle
step. The first pass advances all joints together; further phase-shifted passes
combine joints differently. Duplicate poses are removed. Use the **Pose** slider
to preview them.

**Joint-range poses** configures **Angle step**, **Range coverage**,
**Recombined passes** and **Include root rotations**. Defaults are 30°, 90%,
four passes and no root rotation. With the bundled T-Pose profile this produces
63 poses, covering 51 joints and 93 axes. Editing limits or these settings
updates the poses immediately (a running fit keeps its poses until it ends).
The settings persist across sessions.

**Workflow**:

1. **1 · Fit Base Pose** fits the bind pose from scratch with the main
   **Steps / LR** controls and assigns the primitives to bones. An already
   computed bind-pose SDF is reused.
2. **2 · Fit Morph Poses** fits every joint-range pose as a fixed-ID morph key
   from the current base, using its own **Steps per morph pose** (default 1000).
   A base from the main **Fit** button is assigned to bones automatically.
   When it finishes, the morph fit is saved automatically to
   `morph_fits/<model>_morph_fit.json`.

**▶ Fit Base + Morph Poses** runs both steps in one click. **Stop** cancels
either step. Each generated pose is skinned natively and fitted with the
Settings dialog's options. Both ellipsoids and superquadrics retain their
primitive IDs and attachments throughout training. The fitted bind pose becomes
the neutral key without a second fit, so the morph stage starts directly with
the first moving pose.
Each pose fit starts from the most similar configuration fitted so far, chosen
per primitive by the morph driver's joint distance (its bone plus parent/child,
following **Joint context**); the base fit counts as the first one. Each fit may
move one trust-region step from that start, and its regularization pulls equally
toward that start and toward the base. Keys therefore continue one another
without the per-fit noise adding up along a chain, and no key leaves the base's
usual trust region. Saved keys remain deltas from the base.
Live fitting previews and the growth/shrink colors remain available.
The timeline and pose label follow the target currently being trained. The
**Settings → Pose refit → Refit position / rotation / scale / exponents** switches also
apply to native correctives; a locked parameter still follows its bones.
Disabled channels retain their fitted base values in seeds from similar poses
and in all saved pose keys. **Refit exponents** independently freezes each primitive's
base superquadric exponents; **Refit scale** controls its radii.
These switches do not restrict the initial base fit. Reusing a fit made in an
edited pose preserves that fit's source pose during bone assignment.

The **Morph Curves** workspace tab plots captured fit samples against a
joint's rotation relative to the saved base pose, like Unity's Morph Exponent
Plot. Choose a joint, primitive ID, angle axis (Auto / X / Y / Z / total) and
parameter: exponents, radius scale relative to base, local radii, local position,
position delta or rotation delta. Rotations use shortest signed rotation vectors
in degrees. The sample table and selectable points retain distinct poses at
the same angle; **Connect samples** adds a visual guide. **All primitives**
allows comparison with a different joint. The tab refreshes after training or
loading a morph fit and can be reopened through **Views → Morph Curves**.

After training, **Morph fit → Pose-driven primitive morphing** blends the
learned center, rotation, radius and superquadric exponent deltas from the
actual local joint rotations. This also works while rotating individual joints. It uses the Unity Morph Driver's
primary-bone/per-bone pose weighting: quaternion distances, neighbouring-joint
context, neutral-base support and up to six nearby keys. **Joint context**,
**Pose falloff** and **Transition** control the preview; a transition of zero
updates immediately. Disabling morphing displays the bone-driven base.

**Morph fit → Model** switches the preview between **Morph keys (pose
blend)** and **Joint functions (polynomial)**. Joint functions replace the key
database with one small polynomial per primitive. Its inputs are the rotations
of the primitive's bone, that bone's parent and children, and the primitive's
attachment joints. Per joint, the bind-relative quaternion gives 13 features
(x, y, z, w−1, their squares, pairwise products and cubes), which are all zero
at the bind pose. The coefficients are fitted to the stored keys by ridge
regression, with the regularization chosen by leave-one-out error. This takes
under a second whenever the keys change. The panel reports the coefficient
count. Outputs are clamped to slightly beyond the trained range. Functions
generalize best when the keys come from random multi-axis poses. Fitted on
one-axis sweeps alone, they interpolate poorly between joints.

**Animation preview** plays a clip without fitting anything:
- **Random motion (joint limits)**: a looping motion through random poses in
  which several joints rotate on several axes at once. **New** draws another
  one.
- Multi-frame clips from the pose library.
- The mesh's own animation.

**Play / Pause / Stop**, **Speed** and the scrub slider control playback.
**Stop**, moving the pose slider or rotating a joint returns to the timeline
pose. Starting a fit stops playback.

The **Pose Error** workspace tab shows the current total error: the mean
distance from 2,500 fixed mesh-surface vertices to the primitive union, in
percent of the model height, plus the 95th percentile. It compares no morph,
morph keys and joint functions in the same pose. During playback it plots all
three over time and keeps their running averages (**Reset averages** clears
them). Each vertex is checked against the ten primitives nearest to it in the
base pose, which keeps a measurement at a few milliseconds.

**Morph fit → Save as… / Load…** stores and restores the base primitives,
attachments and pose descriptors in Correctives V4 JSON. Native files also
record mesh, skinning and bind information to reject mismatched models before
replacing the current fit. The pose-generation method is model-independent;
the fitted morph geometry belongs to the model on which it was trained.

The Unity HTTP API and Unity batch fitter remain available alongside this
native workflow. Native runs and API jobs share the fitting queue; one finishes
before the other may replace its data. Unity V4 exports retain their existing
format, including pose descriptors for runtime morphing.

## Unity primitive pipeline

The Unity bridge supports two fitted primitive types:

- `ellipsoid`
- `superquadric` with per-primitive shape exponents `[epsilon1, epsilon2]`

Protocol version 4 carries `primitive_type` and `shape_exponents` through base
fitting, fixed-population pose fitting, synthetic pose batches, bone-local
storage, morph targets, previews, and JSON export. Version 2/3 data without the
new fields remains valid and is interpreted as an ellipsoid with exponents
`[1, 1]`.

> **NOT IMPLEMENTED:** Superquadric collision detection/response in Unity.
> Superquadrics can be fitted, previewed, stored, and morphed, but they are
> deliberately not submitted to the existing ellipsoid collision pipeline.
