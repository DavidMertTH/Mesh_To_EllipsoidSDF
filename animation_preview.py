"""Animation tracks for live morph previews.

A track is a list of key poses played at a fixed key rate, interpolated in
local bone space.  Besides pose-library clips and the mesh's own animation,
``random_motion`` builds a looping motion through random poses inside the
joint limits.  Several joints rotate on several axes at once, which is the
case that systematic one-axis sweeps never cover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from joint_dof import joint_rotation_frames
from joint_pose import _rotation
from skeleton import Pose, Skeleton, interpolate_pose


def random_multi_axis_poses(skeleton: Skeleton, count: int, *, move_probability: float = 0.4,
                            coverage: float = 0.8, seed: int = 0,
                            include_roots: bool = False) -> list[Pose]:
    """Random poses inside the joint limits; a moving joint uses all its free axes."""
    rng = np.random.default_rng(seed)
    bones = skeleton.bones
    frames = joint_rotation_frames(np.asarray([b.local_bind_transform for b in bones])).astype(np.float64)
    stretch = [frames[i].T @ np.asarray(b.local_bind_transform)[:3, :3] for i, b in enumerate(bones)]
    poses = []
    for p in range(int(count)):
        locals_ = {}
        for i, bone in enumerate(bones):
            if bone.parent_index < 0 and not include_roots:
                continue
            dof = bone.joint_dof
            if not any(dof.enabled) or rng.random() > move_probability:
                continue
            angles = np.zeros(3)
            for axis in range(3):
                if dof.enabled[axis] and dof.maximum[axis] - dof.minimum[axis] > 1e-6:
                    neutral = float(np.clip(0, dof.minimum[axis], dof.maximum[axis]))
                    low = neutral + (dof.minimum[axis] - neutral) * coverage
                    high = neutral + (dof.maximum[axis] - neutral) * coverage
                    angles[axis] = rng.uniform(low, high)
            if not np.any(angles):
                continue
            local = np.asarray(bone.local_bind_transform, dtype=np.float64).copy()
            local[:3, :3] = frames[i] @ _rotation(angles) @ stretch[i]
            locals_[i] = local
        poses.append(Pose(name=f"random multi-axis {p + 1}", bone_locals=locals_))
    return poses


@dataclass
class AnimationTrack:
    name: str
    poses: list[Pose]
    keys_per_second: float = 8.0
    ease: bool = False                 # smoothstep between keys (calm random motion)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def duration(self) -> float:
        return max(0, len(self.poses) - 1) / max(self.keys_per_second, 1e-6)

    def frame_at(self, seconds: float) -> float:
        """Fractional key index at *seconds*, looping over the track."""
        if len(self.poses) <= 1:
            return 0.0
        duration = self.duration
        return float((seconds % duration) / duration * (len(self.poses) - 1))

    def pose_at(self, skeleton: Skeleton, seconds: float) -> Pose:
        frame = self.frame_at(seconds)
        i0 = int(np.floor(frame))
        i1 = min(i0 + 1, len(self.poses) - 1)
        w = frame - i0
        if self.ease:
            w = w * w * (3 - 2 * w)
        if i0 == i1 or w <= 1e-6:
            return self.poses[i0]
        if w >= 1 - 1e-6:
            return self.poses[i1]
        return interpolate_pose(skeleton, self.poses[i0], self.poses[i1], w,
                                name=f"{self.name} · {frame:.2f}")


def random_motion(skeleton: Skeleton, *, keys: int = 24, seed: int = 0,
                  keys_per_second: float = 0.8, include_roots: bool = False) -> AnimationTrack:
    """Looping motion bind → random multi-axis poses → bind."""
    poses = [Pose.t_pose()] + random_multi_axis_poses(
        skeleton, keys, seed=seed, include_roots=include_roots) + [Pose.t_pose()]
    return AnimationTrack(f"Random motion (seed {seed})", poses, keys_per_second, ease=True)
