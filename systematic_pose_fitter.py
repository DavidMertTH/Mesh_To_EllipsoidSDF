"""Model-independent, deterministic pose sweeps from editable joint ranges.

One channel at a time is sampled within each joint, stepping each direction
outward from the bind angle. Independent joint lanes are packed into whole-rig
poses as in the Unity systematic fitter, but ordered so consecutive fits change
as little as possible: the first pass advances every lane one angle step per
pose, and later passes recombine the lanes with cyclic phase shifts. (Unity
still interleaves directions and phase-shifts its first pass.) Only the current
rig's bind transforms and JointDof ranges are used; no FBX animation or
humanoid-name table is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import re

import numpy as np

from joint_dof import joint_rotation_frames
from joint_pose import _rotation
from skeleton import Pose, Skeleton


@dataclass(frozen=True)
class SystematicPoseSettings:
    angle_step: float = 30.0
    coverage: float = 0.9
    passes: int = 4
    include_roots: bool = False

    def __post_init__(self):
        _validate_sampling(self.angle_step, self.coverage)
        if isinstance(self.passes, bool) or not isinstance(self.passes, Integral) or not 1 <= self.passes <= 12:
            raise ValueError("Use between 1 and 12 passes")


@dataclass(frozen=True)
class JointAngleTarget:
    joint: int
    axis: int
    angle: float


@dataclass
class SystematicPosePlan:
    poses: list[Pose]
    targets: list[tuple[JointAngleTarget, ...]]
    settings: SystematicPoseSettings
    joint_count: int
    channel_count: int


def _validate_sampling(step: float, coverage: float) -> None:
    if not np.isfinite(step) or not 5 <= step <= 90:
        raise ValueError("Angle step must be between 5 and 90 degrees")
    if not np.isfinite(coverage) or not 0.05 <= coverage <= 1:
        raise ValueError("Range coverage must be between 5% and 100%")


def angle_samples(minimum: float, maximum: float, step: float, coverage: float):
    """Step outward through one direction, then the other; include each covered endpoint.

    Consecutive samples on one side differ by at most one step, and every
    sample is at most one step from an earlier sample or the neutral angle,
    so each pose fit can start from an already fitted, adjacent angle. The
    shorter direction goes first (negative on ties), which keeps the single
    jump across the neutral angle small.

    Positive-only or negative-only ranges need not contain the bind angle 0.
    Compress their interval toward its nearest-to-zero endpoint, keeping all
    sampled angles within the configured range.
    """
    _validate_sampling(step, coverage)
    if not np.isfinite([minimum, maximum]).all() or not -180 <= minimum <= maximum <= 180:
        raise ValueError("Joint angles must satisfy -180 <= min <= max <= 180")
    if maximum - minimum <= 1e-6:
        return []
    neutral = float(np.clip(0, minimum, maximum))
    lo = neutral + (minimum - neutral) * coverage
    hi = neutral + (maximum - neutral) * coverage

    def magnitudes(low, high):
        if high <= 1e-6:
            return []
        values = [float(v) for v in np.arange(step, high, step) if v >= low]
        if low > 1e-6:
            values.append(float(low))
        values.append(float(high))
        # Keep exact endpoints: rounding a limit just below a step would push a
        # sample outside the user's range. Sorting exact values is deterministic.
        return sorted(set(values))

    negative = [-value for value in magnitudes(max(0.0, -hi), max(0.0, -lo))]
    positive = magnitudes(max(0.0, lo), max(0.0, hi))
    if positive and negative and positive[-1] < -negative[-1]:
        return positive + negative
    return negative + positive


def _canonical_lane_name(name: str) -> str:
    name = name.rsplit(":", 1)[-1].casefold()
    name = re.sub(r"[\W_]", "", name)
    return re.sub(r"^(left|right)", "", name)


def _lane_phase(name, count, pass_index):
    # The first pass sweeps every lane from its first sample, one step per
    # pose. Later passes shift whole lanes cyclically: this recombines
    # independent joints while each lane still advances one step per pose.
    if not pass_index:
        return 0
    canonical = _canonical_lane_name(name)
    mask = 0xFFFFFFFF
    phase = 2166136261
    offset = 2246822519
    for char in canonical:
        phase = ((phase ^ ord(char)) * 16777619) & mask
        offset = ((offset ^ ord(char)) * 3266489917) & mask
    offset ^= (pass_index * 2654435761) & mask
    offset ^= offset >> 16
    offset = (offset * 2246822519) & mask
    offset ^= offset >> 13
    return (phase + offset) % count


def systematic_pose_signature(skeleton: Skeleton, settings: SystematicPoseSettings):
    """Cache key for the bind transforms, edited DOFs and authoring options."""
    return (settings, tuple((bone.name, bone.parent_index,
        tuple(float(v) for v in np.asarray(bone.local_bind_transform).ravel()),
        tuple(bool(v) for v in bone.joint_dof.enabled),
        tuple(float(v) for v in bone.joint_dof.minimum),
        tuple(float(v) for v in bone.joint_dof.maximum)) for bone in skeleton.bones))


def generate_systematic_poses(skeleton: Skeleton, settings=None) -> SystematicPosePlan:
    settings = settings or SystematicPoseSettings()
    lanes = []
    channel_count = 0
    for index, bone in enumerate(skeleton.bones):
        if bone.parent_index < 0 and not settings.include_roots:
            continue
        targets = []
        for axis in range(3):
            if not bone.joint_dof.enabled[axis]:
                continue
            samples = angle_samples(float(bone.joint_dof.minimum[axis]),
                float(bone.joint_dof.maximum[axis]), settings.angle_step, settings.coverage)
            if samples:
                channel_count += 1
            targets.extend(JointAngleTarget(index, axis, angle) for angle in samples)
        if targets:
            lanes.append((index, bone.name, targets))
    poses = [Pose.t_pose()]
    specifications = [()]
    if not lanes:
        return SystematicPosePlan(poses, specifications, settings, 0, 0)
    reference = joint_rotation_frames(np.asarray([b.local_bind_transform for b in skeleton.bones])).astype(np.float64)
    stretches = [reference[i].T @ bone.local_bind_transform[:3, :3]
                 for i, bone in enumerate(skeleton.bones)]
    packed_count = max(len(targets) for _i, _name, targets in lanes)
    seen = set()
    for pass_index in range(settings.passes):
        for sample_index in range(packed_count):
            selected, locals_ = [], {}
            for index, name, targets in lanes:
                if sample_index >= len(targets):
                    continue
                phase = _lane_phase(name, len(targets), pass_index)
                target = targets[(sample_index + phase) % len(targets)]
                angles = np.zeros(3)
                angles[target.axis] = target.angle
                local = np.asarray(skeleton.bones[index].local_bind_transform, dtype=np.float64).copy()
                local[:3, :3] = reference[index] @ _rotation(angles) @ stretches[index]
                locals_[index] = local
                selected.append(target)
            if selected:
                # Small rigs can exhaust all combinations before all requested
                # passes. Fitting duplicate supports would waste work and make
                # exact-pose morph matching ambiguous.
                signature = tuple((t.joint, t.axis, round((t.angle + 180) % 360 - 180, 12)) for t in selected)
                if signature in seen:
                    continue
                seen.add(signature)
                poses.append(Pose(name=f"Joint sweep · pass {pass_index + 1} · {sample_index + 1}/{packed_count}",
                                  bone_locals=locals_))
                specifications.append(tuple(selected))
    return SystematicPosePlan(poses, specifications, settings, len(lanes), channel_count)
