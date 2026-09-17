"""Regression contracts for the authored Generic right-hand pose library.

The active Unity scene is intentionally not part of this contract: the hand
currently lives in Unity's unsaved scene state.  These tests protect the
durable pose asset and the rotation-delta capture API without requiring Unity
to be available in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = ROOT / "clothSimulation"
POSE_ASSET_PATH = (
    UNITY_ROOT / "Character" / "Poses" / "RightHandPoses.asset"
)
POSE_ASSET_META_PATH = POSE_ASSET_PATH.with_suffix(".asset.meta")
LIBRARY_SOURCE_PATH = (
    UNITY_ROOT / "EllipsoidLoading" / "EllipSDFGenericPoseLibrary.cs"
)
LIBRARY_SOURCE_META_PATH = LIBRARY_SOURCE_PATH.with_suffix(".cs.meta")

EXPECTED_LIBRARY_SCRIPT_GUID = "b2f4d569d0e64a4f9f8cb6d3ef430261"
EXPECTED_POSE_NAMES = [
    "Hand_Cup",
    "Hand_Fist",
    "Hand_Point",
    "Hand_Hook",
    "Hand_Splay",
    "Hand_Flat",
    "Hand_Hyperextend",
    "Hand_Claw",
    "Hand_VSign",
    "Hand_ThumbUp",
    "Hand_TripodPrep",
    "Hand_TightFist",
]

# Stable SkinnedMeshRenderer.bones mapping of Character/RightHand.fbx.  Pose
# entries carry both a hierarchy path and an index, so test the pair instead
# of merely accepting any non-negative index.
RIGHT_HAND_RENDERER_PATH_BY_INDEX = {
    0: ".",
    1: "R_IndexMetacarpal",
    2: (
        "R_LittleMetacarpal/R_LittleProximal/"
        "R_LittleIntermediate/R_LittleDistal/R_LittleTip"
    ),
    3: "R_MiddleMetacarpal",
    4: "R_MiddleMetacarpal/R_MiddleProximal",
    5: (
        "R_MiddleMetacarpal/R_MiddleProximal/"
        "R_MiddleIntermediate"
    ),
    6: (
        "R_MiddleMetacarpal/R_MiddleProximal/"
        "R_MiddleIntermediate/R_MiddleDistal"
    ),
    7: (
        "R_MiddleMetacarpal/R_MiddleProximal/"
        "R_MiddleIntermediate/R_MiddleDistal/R_MiddleTip"
    ),
    8: "R_RingMetacarpal",
    9: "R_RingMetacarpal/R_RingProximal",
    10: "R_RingMetacarpal/R_RingProximal/R_RingIntermediate",
    11: (
        "R_RingMetacarpal/R_RingProximal/"
        "R_RingIntermediate/R_RingDistal"
    ),
    12: "R_IndexMetacarpal/R_IndexProximal",
    13: (
        "R_RingMetacarpal/R_RingProximal/"
        "R_RingIntermediate/R_RingDistal/R_RingTip"
    ),
    14: "R_Palm",
    15: "R_ThumbMetacarpal",
    16: "R_ThumbMetacarpal/R_ThumbProximal",
    17: "R_ThumbMetacarpal/R_ThumbProximal/R_ThumbDistal",
    18: (
        "R_ThumbMetacarpal/R_ThumbProximal/"
        "R_ThumbDistal/R_ThumbTip"
    ),
    19: (
        "R_IndexMetacarpal/R_IndexProximal/"
        "R_IndexIntermediate"
    ),
    20: (
        "R_IndexMetacarpal/R_IndexProximal/"
        "R_IndexIntermediate/R_IndexDistal"
    ),
    21: (
        "R_IndexMetacarpal/R_IndexProximal/"
        "R_IndexIntermediate/R_IndexDistal/R_IndexTip"
    ),
    22: "R_LittleMetacarpal",
    23: "R_LittleMetacarpal/R_LittleProximal",
    24: (
        "R_LittleMetacarpal/R_LittleProximal/"
        "R_LittleIntermediate"
    ),
    25: (
        "R_LittleMetacarpal/R_LittleProximal/"
        "R_LittleIntermediate/R_LittleDistal"
    ),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _parse_inline_vector(value: str, dimensions: str) -> tuple[float, ...]:
    fields = {
        key: float(number)
        for key, number in re.findall(
            r"([xyzw]):\s*([^,}]+)", value, flags=re.IGNORECASE
        )
    }
    missing = [dimension for dimension in dimensions if dimension not in fields]
    if missing:
        raise AssertionError(
            f"Missing vector fields {missing!r} in serialized value {value!r}"
        )
    return tuple(fields[dimension] for dimension in dimensions)


@dataclass(frozen=True)
class BoneRecord:
    relative_path: str
    renderer_bone_index: int
    apply_position: int
    local_position: tuple[float, float, float]
    apply_rotation: int
    local_rotation: tuple[float, float, float, float]
    apply_scale: int
    local_scale: tuple[float, float, float]


@dataclass
class PoseRecord:
    name: str
    encoding: int | None = None
    bones: list[BoneRecord] = field(default_factory=list)


def _parse_pose_asset(source: str) -> list[PoseRecord]:
    """Parse the small, stable subset of Unity YAML used by this asset."""

    raw_poses: list[dict[str, object]] = []
    current_pose: dict[str, object] | None = None
    current_bone: dict[str, str] | None = None

    for line in source.splitlines():
        pose_match = re.fullmatch(r"  - name:\s*(.+)", line)
        if pose_match:
            current_pose = {
                "name": pose_match.group(1).strip(),
                "encoding": None,
                "bones": [],
            }
            raw_poses.append(current_pose)
            current_bone = None
            continue

        if current_pose is None:
            continue

        encoding_match = re.fullmatch(r"    encoding:\s*(-?\d+)", line)
        if encoding_match:
            current_pose["encoding"] = int(encoding_match.group(1))
            continue

        bone_match = re.fullmatch(r"    - relativePath:\s*(.+)", line)
        if bone_match:
            current_bone = {"relativePath": bone_match.group(1).strip()}
            bones = current_pose["bones"]
            assert isinstance(bones, list)
            bones.append(current_bone)
            continue

        if current_bone is None:
            continue
        field_match = re.fullmatch(r"      ([A-Za-z]+):\s*(.+)", line)
        if field_match:
            current_bone[field_match.group(1)] = field_match.group(2).strip()

    parsed: list[PoseRecord] = []
    required_bone_fields = {
        "relativePath",
        "rendererBoneIndex",
        "applyPosition",
        "localPosition",
        "applyRotation",
        "localRotation",
        "applyScale",
        "localScale",
    }
    for raw_pose in raw_poses:
        pose = PoseRecord(
            name=str(raw_pose["name"]),
            encoding=raw_pose["encoding"],  # type: ignore[arg-type]
        )
        raw_bones = raw_pose["bones"]
        assert isinstance(raw_bones, list)
        for raw_bone in raw_bones:
            assert isinstance(raw_bone, dict)
            missing = required_bone_fields.difference(raw_bone)
            if missing:
                raise AssertionError(
                    f"Pose {pose.name!r} has an incomplete bone record: "
                    f"missing {sorted(missing)!r}"
                )
            pose.bones.append(
                BoneRecord(
                    relative_path=raw_bone["relativePath"],
                    renderer_bone_index=int(raw_bone["rendererBoneIndex"]),
                    apply_position=int(raw_bone["applyPosition"]),
                    local_position=_parse_inline_vector(
                        raw_bone["localPosition"], "xyz"
                    ),
                    apply_rotation=int(raw_bone["applyRotation"]),
                    local_rotation=_parse_inline_vector(
                        raw_bone["localRotation"], "xyzw"
                    ),
                    apply_scale=int(raw_bone["applyScale"]),
                    local_scale=_parse_inline_vector(
                        raw_bone["localScale"], "xyz"
                    ),
                )
            )
        parsed.append(pose)
    return parsed


def _guid_from_meta(source: str) -> str:
    match = re.search(r"(?m)^guid:\s*([0-9a-f]{32})\s*$", source)
    if match is None:
        raise AssertionError("Unity meta file contains no 32-character GUID")
    return match.group(1)


def _quaternion_norm(rotation: tuple[float, float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in rotation))


def _rotation_angle_degrees(
    rotation: tuple[float, float, float, float],
) -> float:
    norm = _quaternion_norm(rotation)
    if not math.isfinite(norm) or norm <= 0.0:
        return math.nan
    normalized_w = max(-1.0, min(1.0, rotation[3] / norm))
    # q and -q encode the same rotation.
    return math.degrees(2.0 * math.acos(abs(normalized_w)))


def _canonical_rotation(
    rotation: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = _quaternion_norm(rotation)
    normalized = tuple(component / norm for component in rotation)
    if normalized[3] < 0.0:
        normalized = tuple(-component for component in normalized)
    return tuple(round(component, 6) for component in normalized)


def _pose_by_name(poses: list[PoseRecord], name: str) -> PoseRecord:
    for pose in poses:
        if pose.name == name:
            return pose
    raise AssertionError(f"Required hand pose {name!r} is missing")


def _bone_map(pose: PoseRecord) -> dict[str, BoneRecord]:
    return {bone.relative_path: bone for bone in pose.bones}


IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


def _rotation_at(
    bones: dict[str, BoneRecord], path: str
) -> tuple[float, float, float, float]:
    """Return the implicit identity delta of a pruned sparse entry."""

    bone = bones.get(path)
    return bone.local_rotation if bone is not None else IDENTITY_ROTATION


def _finger_chain(finger: str) -> tuple[str, str, str]:
    proximal = f"R_{finger}Metacarpal/R_{finger}Proximal"
    intermediate = f"{proximal}/R_{finger}Intermediate"
    distal = f"{intermediate}/R_{finger}Distal"
    return proximal, intermediate, distal


def _block_after(source: str, marker: str) -> str:
    try:
        start = source.index(marker)
        brace = source.index("{", start)
    except ValueError as exc:
        raise AssertionError(f"Missing C# block after {marker!r}") from exc

    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise AssertionError(f"Unclosed C# block after {marker!r}")


POSE_ASSET = _read(POSE_ASSET_PATH)
POSE_ASSET_META = _read(POSE_ASSET_META_PATH)
LIBRARY_SOURCE = _read(LIBRARY_SOURCE_PATH)
LIBRARY_SOURCE_META = _read(LIBRARY_SOURCE_META_PATH)
POSES = _parse_pose_asset(POSE_ASSET)


class RightHandPoseAssetTest(unittest.TestCase):
    def test_asset_uses_the_generic_pose_library_script(self) -> None:
        self.assertTrue(POSE_ASSET_PATH.exists())
        self.assertTrue(POSE_ASSET_META_PATH.exists())
        self.assertEqual(
            _guid_from_meta(LIBRARY_SOURCE_META),
            EXPECTED_LIBRARY_SCRIPT_GUID,
        )
        self.assertRegex(
            POSE_ASSET,
            rf"m_Script:\s*\{{fileID:\s*11500000,\s*"
            rf"guid:\s*{EXPECTED_LIBRARY_SCRIPT_GUID},\s*type:\s*3\}}",
        )
        self.assertEqual(
            _guid_from_meta(POSE_ASSET_META),
            "1c2b86bd496552d498f2b59961511ea3",
        )

    def test_catalog_has_exactly_twelve_stable_unique_names(self) -> None:
        names = [pose.name for pose in POSES]
        self.assertEqual(names, EXPECTED_POSE_NAMES)
        self.assertEqual(len(names), 12)
        self.assertEqual(len({name.casefold() for name in names}), len(names))

    def test_poses_are_sparse_rotation_only_deltas(self) -> None:
        for pose in POSES:
            with self.subTest(pose=pose.name):
                self.assertEqual(pose.encoding, 1, "Expected DeltaFromBase")
                self.assertGreater(len(pose.bones), 0)
                self.assertLess(
                    len(pose.bones),
                    26,
                    "A hand pose should not redundantly serialize every renderer bone",
                )

                paths = [bone.relative_path for bone in pose.bones]
                indices = [bone.renderer_bone_index for bone in pose.bones]
                self.assertEqual(len(paths), len(set(paths)))
                self.assertEqual(len(indices), len(set(indices)))
                self.assertNotIn(".", paths)
                self.assertNotIn("R_Wrist", paths)

                for bone in pose.bones:
                    self.assertTrue(bone.relative_path.startswith("R_"))
                    self.assertGreaterEqual(bone.renderer_bone_index, 0)
                    self.assertNotIn(
                        bone.relative_path.rsplit("/", 1)[-1],
                        {
                            "R_IndexTip",
                            "R_MiddleTip",
                            "R_RingTip",
                            "R_LittleTip",
                            "R_ThumbTip",
                        },
                    )
                    self.assertEqual(
                        RIGHT_HAND_RENDERER_PATH_BY_INDEX.get(
                            bone.renderer_bone_index
                        ),
                        bone.relative_path,
                        "Path/index pair must identify a real RightHand "
                        "renderer bone",
                    )
                    self.assertEqual(bone.apply_position, 0)
                    self.assertEqual(bone.local_position, (0.0, 0.0, 0.0))
                    self.assertEqual(bone.apply_rotation, 1)
                    self.assertEqual(bone.apply_scale, 0)
                    self.assertEqual(bone.local_scale, (1.0, 1.0, 1.0))

    def test_every_serialized_quaternion_is_finite_normalized_and_effective(
        self,
    ) -> None:
        for pose in POSES:
            effective_angles: list[float] = []
            for bone in pose.bones:
                with self.subTest(pose=pose.name, bone=bone.relative_path):
                    self.assertTrue(
                        all(math.isfinite(value) for value in bone.local_rotation)
                    )
                    self.assertAlmostEqual(
                        _quaternion_norm(bone.local_rotation), 1.0, delta=2.0e-5
                    )
                    angle = _rotation_angle_degrees(bone.local_rotation)
                    self.assertGreater(angle, 0.001)
                    effective_angles.append(angle)

            self.assertGreaterEqual(
                sum(angle > 5.0 for angle in effective_angles),
                3,
                f"{pose.name} must meaningfully move several joints",
            )
            self.assertGreater(
                max(effective_angles),
                8.0,
                f"{pose.name} needs at least one clearly visible rotation delta",
            )

    def test_pose_rotation_signatures_are_all_distinct(self) -> None:
        signatures: dict[
            tuple[tuple[str, tuple[float, float, float, float]], ...], str
        ] = {}
        for pose in POSES:
            signature = tuple(
                sorted(
                    (
                        bone.relative_path,
                        _canonical_rotation(bone.local_rotation),
                    )
                    for bone in pose.bones
                )
            )
            self.assertNotIn(
                signature,
                signatures,
                f"{pose.name} duplicates {signatures.get(signature)!r}",
            )
            signatures[signature] = pose.name

    def test_cup_and_fist_cover_all_finger_chains_at_distinct_strengths(
        self,
    ) -> None:
        cup = _bone_map(_pose_by_name(POSES, "Hand_Cup"))
        fist = _bone_map(_pose_by_name(POSES, "Hand_Fist"))
        required = {
            path
            for finger in ("Index", "Middle", "Ring", "Little")
            for path in _finger_chain(finger)
        }
        self.assertTrue(required.issubset(cup))
        self.assertTrue(required.issubset(fist))
        self.assertEqual(set(cup), set(fist))

        stronger_joint_count = 0
        cup_angles: list[float] = []
        fist_angles: list[float] = []
        for path in required:
            cup_angle = _rotation_angle_degrees(cup[path].local_rotation)
            fist_angle = _rotation_angle_degrees(fist[path].local_rotation)
            cup_angles.append(cup_angle)
            fist_angles.append(fist_angle)
            if fist_angle > cup_angle + 3.0:
                stronger_joint_count += 1
        self.assertGreaterEqual(stronger_joint_count, 9)
        self.assertGreater(
            sum(fist_angles) / len(fist_angles),
            sum(cup_angles) / len(cup_angles) + 10.0,
        )

    def test_point_keeps_index_open_while_other_fingers_curl(self) -> None:
        point = _bone_map(_pose_by_name(POSES, "Hand_Point"))
        index_angles: list[float] = []
        negative_index_deltas = 0
        present_index_deltas = 0
        for path in _finger_chain("Index"):
            rotation = _rotation_at(point, path)
            if path in point:
                present_index_deltas += 1
            index_angles.append(
                _rotation_angle_degrees(rotation)
            )
            if _canonical_rotation(rotation)[0] < 0.0:
                negative_index_deltas += 1
        self.assertGreaterEqual(present_index_deltas, 2)
        self.assertGreaterEqual(negative_index_deltas, 2)

        curled_angles: list[float] = []
        positive_curled_deltas = 0
        for finger in ("Middle", "Ring", "Little"):
            for path in _finger_chain(finger):
                self.assertIn(path, point)
                curled_angles.append(
                    _rotation_angle_degrees(point[path].local_rotation)
                )
                if _canonical_rotation(point[path].local_rotation)[0] > 0.0:
                    positive_curled_deltas += 1
        self.assertGreaterEqual(positive_curled_deltas, 8)
        self.assertGreater(
            sum(curled_angles) / len(curled_angles),
            sum(index_angles) / len(index_angles) + 15.0,
        )

    def test_hook_bends_distal_joints_without_closing_the_knuckles(self) -> None:
        hook = _bone_map(_pose_by_name(POSES, "Hand_Hook"))
        proximal_angles: list[float] = []
        intermediate_angles: list[float] = []
        distal_angles: list[float] = []
        for finger in ("Index", "Middle", "Ring", "Little"):
            proximal, intermediate, distal = _finger_chain(finger)
            for path in (proximal, intermediate, distal):
                self.assertIn(path, hook)
            proximal_angles.append(
                _rotation_angle_degrees(hook[proximal].local_rotation)
            )
            intermediate_angles.append(
                _rotation_angle_degrees(hook[intermediate].local_rotation)
            )
            distal_angles.append(
                _rotation_angle_degrees(hook[distal].local_rotation)
            )
            self.assertGreater(
                _canonical_rotation(hook[intermediate].local_rotation)[0],
                0.0,
            )
            self.assertGreater(
                _canonical_rotation(hook[distal].local_rotation)[0], 0.0
            )
            self.assertGreater(
                (
                    _rotation_angle_degrees(
                        hook[intermediate].local_rotation
                    )
                    + _rotation_angle_degrees(hook[distal].local_rotation)
                )
                / 2.0,
                _rotation_angle_degrees(hook[proximal].local_rotation)
                + 15.0,
            )
        self.assertGreater(
            sum(intermediate_angles) / len(intermediate_angles),
            sum(proximal_angles) / len(proximal_angles) + 20.0,
        )
        self.assertGreater(
            sum(distal_angles) / len(distal_angles),
            sum(proximal_angles) / len(proximal_angles) + 15.0,
        )

    def test_splay_contains_opposed_metacarpal_spread_deltas(self) -> None:
        splay = _bone_map(_pose_by_name(POSES, "Hand_Splay"))
        metacarpals = {
            "R_IndexMetacarpal": 1,
            "R_MiddleMetacarpal": 1,
            "R_RingMetacarpal": -1,
            "R_LittleMetacarpal": -1,
        }
        for path, expected_y_sign in metacarpals.items():
            self.assertIn(path, splay)
            rotation = _canonical_rotation(splay[path].local_rotation)
            self.assertGreater(abs(rotation[1]), 0.005)
            self.assertEqual(1 if rotation[1] > 0.0 else -1, expected_y_sign)

    def test_flat_opens_all_fingers_without_using_root_or_tip_bones(self) -> None:
        flat = _bone_map(_pose_by_name(POSES, "Hand_Flat"))
        finger_paths = [
            path
            for finger in ("Index", "Middle", "Ring", "Little")
            for path in _finger_chain(finger)
        ]
        for finger in ("Index", "Middle", "Ring", "Little"):
            self.assertGreaterEqual(
                len(set(_finger_chain(finger)).intersection(flat)), 2
            )
        negative_x_count = sum(
            _canonical_rotation(_rotation_at(flat, path))[0] < 0.0
            for path in finger_paths
        )
        self.assertGreaterEqual(negative_x_count, 8)

        for path in (
            "R_ThumbMetacarpal/R_ThumbProximal",
            "R_ThumbMetacarpal/R_ThumbProximal/R_ThumbDistal",
        ):
            self.assertIn(path, flat)
            self.assertLess(_canonical_rotation(flat[path].local_rotation)[0], 0.0)

    def test_hyperextend_uses_stronger_negative_x_than_flat(self) -> None:
        flat = _bone_map(_pose_by_name(POSES, "Hand_Flat"))
        hyperextend = _bone_map(_pose_by_name(POSES, "Hand_Hyperextend"))
        finger_paths = [
            path
            for finger in ("Index", "Middle", "Ring", "Little")
            for path in _finger_chain(finger)
        ]
        self.assertTrue(set(finger_paths).issubset(hyperextend))

        for finger in ("Index", "Middle", "Ring"):
            proximal = _finger_chain(finger)[0]
            self.assertLess(
                _canonical_rotation(hyperextend[proximal].local_rotation)[0],
                0.0,
            )
            self.assertGreater(
                _rotation_angle_degrees(hyperextend[proximal].local_rotation),
                _rotation_angle_degrees(flat[proximal].local_rotation) + 5.0,
            )

        little_proximal = _finger_chain("Little")[0]
        self.assertGreater(
            _canonical_rotation(flat[little_proximal].local_rotation)[0], 0.0
        )
        self.assertLess(
            _canonical_rotation(hyperextend[little_proximal].local_rotation)[0],
            0.0,
        )
        self.assertLess(
            max(
                _rotation_angle_degrees(hyperextend[path].local_rotation)
                for path in finger_paths
            ),
            50.0,
        )

    def test_claw_extends_knuckles_and_flexes_intermediate_distal_joints(
        self,
    ) -> None:
        claw = _bone_map(_pose_by_name(POSES, "Hand_Claw"))
        hook = _bone_map(_pose_by_name(POSES, "Hand_Hook"))

        for finger in ("Index", "Middle", "Ring"):
            proximal = _finger_chain(finger)[0]
            self.assertIn(proximal, claw)
            self.assertLess(
                _canonical_rotation(claw[proximal].local_rotation)[0], 0.0
            )
            self.assertGreater(
                _rotation_angle_degrees(claw[proximal].local_rotation),
                _rotation_angle_degrees(hook[proximal].local_rotation) + 5.0,
            )

        for finger in ("Index", "Middle", "Ring", "Little"):
            _, intermediate, distal = _finger_chain(finger)
            for path in (intermediate, distal):
                self.assertIn(path, claw)
                self.assertGreater(
                    _canonical_rotation(claw[path].local_rotation)[0], 0.0
                )
                self.assertGreater(
                    _rotation_angle_degrees(claw[path].local_rotation), 20.0
                )

    def test_v_sign_opens_two_fingers_and_curls_the_other_two(self) -> None:
        v_sign = _bone_map(_pose_by_name(POSES, "Hand_VSign"))

        open_paths = [
            path
            for finger in ("Index", "Middle")
            for path in _finger_chain(finger)
        ]
        self.assertGreaterEqual(len(set(open_paths).intersection(v_sign)), 5)
        self.assertGreaterEqual(
            sum(
                _canonical_rotation(_rotation_at(v_sign, path))[0] < 0.0
                for path in open_paths
            ),
            4,
        )

        curled_paths = [
            path
            for finger in ("Ring", "Little")
            for path in _finger_chain(finger)
        ]
        self.assertTrue(set(curled_paths).issubset(v_sign))
        self.assertTrue(
            all(
                _canonical_rotation(v_sign[path].local_rotation)[0] > 0.0
                for path in curled_paths
            )
        )
        self.assertGreaterEqual(
            sum(
                _rotation_angle_degrees(v_sign[path].local_rotation) > 25.0
                for path in curled_paths
            ),
            4,
        )

        index_meta = _canonical_rotation(
            v_sign["R_IndexMetacarpal"].local_rotation
        )
        middle_meta = _canonical_rotation(
            v_sign["R_MiddleMetacarpal"].local_rotation
        )
        self.assertGreater(index_meta[1], 0.0)
        self.assertLess(middle_meta[1], 0.0)

    def test_thumb_up_closes_fingers_and_extends_the_thumb(self) -> None:
        thumb_up = _bone_map(_pose_by_name(POSES, "Hand_ThumbUp"))
        finger_paths = [
            path
            for finger in ("Index", "Middle", "Ring", "Little")
            for path in _finger_chain(finger)
        ]
        self.assertTrue(set(finger_paths).issubset(thumb_up))
        self.assertGreaterEqual(
            sum(
                _canonical_rotation(thumb_up[path].local_rotation)[0] > 0.0
                for path in finger_paths
            ),
            11,
        )

        thumb_paths = (
            "R_ThumbMetacarpal",
            "R_ThumbMetacarpal/R_ThumbProximal",
            "R_ThumbMetacarpal/R_ThumbProximal/R_ThumbDistal",
        )
        for path in thumb_paths:
            self.assertIn(path, thumb_up)
            rotation = _canonical_rotation(thumb_up[path].local_rotation)
            self.assertLess(rotation[0], 0.0)
            self.assertAlmostEqual(rotation[1], 0.0, delta=1.0e-5)
            self.assertAlmostEqual(rotation[2], 0.0, delta=1.0e-5)

    def test_tripod_prep_uses_thumb_index_middle_without_fake_opposition(
        self,
    ) -> None:
        tripod = _bone_map(_pose_by_name(POSES, "Hand_TripodPrep"))
        for finger in ("Index", "Middle"):
            for path in _finger_chain(finger):
                self.assertIn(path, tripod)
                self.assertGreater(
                    _canonical_rotation(tripod[path].local_rotation)[0], 0.0
                )

        thumb_paths = (
            "R_ThumbMetacarpal",
            "R_ThumbMetacarpal/R_ThumbProximal",
            "R_ThumbMetacarpal/R_ThumbProximal/R_ThumbDistal",
        )
        for path in thumb_paths:
            self.assertIn(path, tripod)
            rotation = _canonical_rotation(tripod[path].local_rotation)
            self.assertGreater(rotation[0], 0.0)
            self.assertAlmostEqual(rotation[1], 0.0, delta=1.0e-5)
            self.assertAlmostEqual(rotation[2], 0.0, delta=1.0e-5)

        self.assertLessEqual(
            max(
                _rotation_angle_degrees(bone.local_rotation)
                for bone in tripod.values()
            ),
            30.5,
        )

    def test_tight_fist_is_stronger_than_fist_across_every_finger_chain(
        self,
    ) -> None:
        fist = _bone_map(_pose_by_name(POSES, "Hand_Fist"))
        tight_fist = _bone_map(_pose_by_name(POSES, "Hand_TightFist"))

        for finger in ("Index", "Middle", "Ring", "Little"):
            proximal, intermediate, distal = _finger_chain(finger)
            chain = (proximal, intermediate, distal)
            self.assertTrue(set(chain).issubset(tight_fist))
            self.assertTrue(
                all(
                    _canonical_rotation(tight_fist[path].local_rotation)[0]
                    > 0.0
                    for path in chain
                )
            )

            fist_angles = [
                _rotation_angle_degrees(fist[path].local_rotation)
                for path in chain
            ]
            tight_angles = [
                _rotation_angle_degrees(tight_fist[path].local_rotation)
                for path in chain
            ]
            for tight_angle, fist_angle in zip(tight_angles, fist_angles):
                self.assertGreater(tight_angle, fist_angle)
            self.assertGreater(
                sum(tight_angles) / len(tight_angles),
                sum(fist_angles) / len(fist_angles) + 5.0,
            )

        thumb_paths = (
            "R_ThumbMetacarpal",
            "R_ThumbMetacarpal/R_ThumbProximal",
            "R_ThumbMetacarpal/R_ThumbProximal/R_ThumbDistal",
        )
        for path in thumb_paths:
            self.assertIn(path, tight_fist)
            rotation = _canonical_rotation(tight_fist[path].local_rotation)
            self.assertGreater(rotation[0], 0.0)
            self.assertAlmostEqual(rotation[1], 0.0, delta=1.0e-5)
            self.assertAlmostEqual(rotation[2], 0.0, delta=1.0e-5)

        for path in thumb_paths[1:]:
            self.assertGreater(
                _rotation_angle_degrees(tight_fist[path].local_rotation),
                _rotation_angle_degrees(fist[path].local_rotation),
            )

        for finger in ("Index", "Middle", "Ring", "Little"):
            self.assertNotIn(f"R_{finger}Metacarpal", tight_fist)
        self.assertLessEqual(
            max(
                _rotation_angle_degrees(bone.local_rotation)
                for bone in tight_fist.values()
            ),
            72.5,
        )


class RotationDeltaCaptureSourceContractTest(unittest.TestCase):
    def test_capture_api_stores_only_effective_renderer_bone_deltas(self) -> None:
        self.assertTrue(LIBRARY_SOURCE_PATH.exists())
        self.assertRegex(
            LIBRARY_SOURCE,
            r"public\s+bool\s+CaptureRotationDeltaPose\s*\(\s*"
            r"string\s+poseName,[\s\S]{0,300}"
            r"IReadOnlyList<Transform>\s+poseBones,[\s\S]{0,120}"
            r"IReadOnlyList<Quaternion>\s+localRotationDeltas",
        )
        capture = _block_after(LIBRARY_SOURCE, "CaptureRotationDeltaPose(")

        for contract in (
            "poseBones.Count == 0",
            "poseBones.Count != localRotationDeltas.Count",
            "Dictionary<Transform, int> rendererIndices",
            "HashSet<Transform> seenBones",
            "!seenBones.Add(bone)",
            "rendererIndices.TryGetValue(bone, out rendererBoneIndex)",
            "BuildRelativePath(skeletonRoot, bone)",
            "!IsFinite(delta)",
            "QuaternionLengthSquared(delta) < 1e-12f",
            "delta = NormalizeQuaternion(delta);",
            "Quaternion.Angle(Quaternion.identity, delta) < 0.001f",
            "captured.encoding = TransformEncoding.DeltaFromBase;",
            "capturedBone.applyPosition = false;",
            "capturedBone.localPosition = Vector3.zero;",
            "capturedBone.applyRotation = true;",
            "capturedBone.localRotation = delta;",
            "capturedBone.applyScale = false;",
            "capturedBone.localScale = Vector3.one;",
            "captured.bones.Count == 0",
        ):
            self.assertIn(contract, capture)

        self.assertLess(
            capture.index(
                "Quaternion.Angle(Quaternion.identity, delta) < 0.001f"
            ),
            capture.index("captured.bones.Add(capturedBone);"),
            "Identity deltas must be pruned before serialization",
        )


if __name__ == "__main__":
    unittest.main()
