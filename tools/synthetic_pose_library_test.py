"""Static regression tests for the Unity synthetic Humanoid pose library."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "clothSimulation"
    / "EllipsoidLoading"
    / "EllipSDFSyntheticPoseBatchFitter.cs"
)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
EDITOR_SOURCE_PATH = (
    ROOT
    / "clothSimulation"
    / "EllipsoidLoading"
    / "Editor"
    / "EllipSDFSyntheticPoseBatchFitterEditor.cs"
)
EDITOR_SOURCE = EDITOR_SOURCE_PATH.read_text(encoding="utf-8")
DRIVER_SOURCE_PATH = (
    ROOT
    / "clothSimulation"
    / "EllipsoidLoading"
    / "EllipSDFMorphDriver.cs"
)
DRIVER_SOURCE = DRIVER_SOURCE_PATH.read_text(encoding="utf-8")

ORIGINAL_PRESET_PREFIX = [
    "TPose",
    "APose",
    "BentKneesAndElbows",
    "ArmsForward",
    "OverheadReach",
    "WideStance",
    "ShallowSquat",
    "LeftLunge",
    "RightLunge",
    "LeftStride",
    "RightStride",
    "TorsoTwistLeft",
    "TorsoTwistRight",
    "RelaxedStanding",
    "ElbowsBentWide",
    "LeftReachUp",
    "RightReachUp",
    "ForwardLean",
    "DeepSquat",
    "DeepLeftLunge",
    "DeepRightLunge",
    "LeftHighKnee",
    "RightHighKnee",
    "CrouchedElbowGuard",
]

NEW_PRESETS = [
    "BackArchLookUp",
    "TorsoSideBendLeft",
    "TorsoSideBendRight",
    "ArmTwistInWristsDown",
    "ArmTwistOutWristsUp",
    "LegsTurnInAnklesFlexed",
    "LegsTurnOutAnklesPointed",
]

EXPECTED_FINGER_SHAPES = {
    "Open",
    "Relaxed",
    "Fist",
    "Splayed",
    "Cup",
    "Claw",
    "Hook",
    "Point",
    "Pinch",
    "VSign",
    "ThumbUp",
    "Blade",
}

APPROVED_EXISTING_POSE_HASH = (
    "e146f84d2519513ceb9a5cdb2b52b4f1cf4cd2ddbc0b6fc119cf9bb267685bcb"
)


def _block_bounds_after_in(source: str, marker: str) -> tuple[int, int]:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return brace + 1, index
    raise AssertionError(f"Unclosed C# block after {marker!r}")


def _block_after_in(source: str, marker: str) -> str:
    start, end = _block_bounds_after_in(source, marker)
    return source[start:end]


def _block_after(marker: str) -> str:
    return _block_after_in(SOURCE, marker)


def _preset_order() -> list[str]:
    body = _block_after(
        "static readonly PosePreset[] PresetOrder")
    return re.findall(r"PosePreset\.(\w+)", body)


def _enum_bits() -> dict[str, int]:
    body = _block_after("public enum PosePreset")
    return {
        name: int(bit)
        for name, bit in re.findall(
            r"^\s*(\w+)\s*=\s*1\s*<<\s*(\d+)\s*,",
            body,
            flags=re.MULTILINE,
        )
    }


def _enum_members(marker: str) -> list[str]:
    body = re.sub(r"//.*", "", _block_after(marker))
    return re.findall(
        r"^\s*([A-Za-z_]\w*)\s*(?:=\s*[^,\n]+)?\s*,?\s*$",
        body,
        flags=re.MULTILINE,
    )


def _switch_cases(body: str, enum_name: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(
        rf"^\s*case\s+{re.escape(enum_name)}\.(\w+)\s*:",
        body,
        flags=re.MULTILINE,
    ))
    cases: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        default = re.search(r"^\s*default\s*:", body[match.end():end],
                            flags=re.MULTILINE)
        if default is not None:
            end = match.end() + default.start()
        cases.append((match.group(1), body[match.end():end]))
    return cases


def _finger_shape_assignment_cases(
) -> list[tuple[str, list[str], list[str], str]]:
    method = _block_after("void ApplyFingerVariation(")
    switch = _block_after_in(method, "switch (preset)")
    result: list[tuple[str, list[str], list[str], str]] = []
    for preset, body in _switch_cases(switch, "PosePreset"):
        left = re.findall(
            r"\bleftShape\s*=\s*FingerShape\.(\w+)\s*;", body)
        right = re.findall(
            r"\brightShape\s*=\s*FingerShape\.(\w+)\s*;", body)
        result.append((preset, left, right, body))
    return result


def _named_calls(body: str, names: tuple[str, ...]) -> list[str]:
    call_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, names)) + r")\s*\(")
    calls: list[str] = []
    for match in call_pattern.finditer(body):
        open_paren = body.index("(", match.start())
        depth = 0
        quote: str | None = None
        escaped = False
        for index in range(open_paren, len(body)):
            char = body[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in ('"', "'"):
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    arguments = re.sub(
                        r"\s+", "", body[open_paren + 1:index])
                    calls.append(f"{match.group(1)}({arguments})")
                    break
        else:
            raise AssertionError(
                f"Unclosed {match.group(1)} call in FingerShape definition")
    return calls


def _numeric_literals(text: str) -> tuple[float, ...]:
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', "", text)
    values = re.findall(
        r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)"
        r"(?:[eE][-+]?\d+)?f?\b",
        without_strings,
    )
    return tuple(float(value.rstrip("fF")) for value in values)


class SyntheticPoseLibraryTest(unittest.TestCase):
    def test_original_indices_are_stable_and_new_presets_are_appended(self) -> None:
        order = _preset_order()
        self.assertEqual(order[:len(ORIGINAL_PRESET_PREFIX)],
                         ORIGINAL_PRESET_PREFIX)
        self.assertEqual(order[len(ORIGINAL_PRESET_PREFIX):], NEW_PRESETS)

    def test_existing_pose_definitions_match_approved_baseline(self) -> None:
        original_cases = SOURCE[
            SOURCE.index("            case PosePreset.TPose:"):
            SOURCE.index("            case PosePreset.BackArchLookUp:")
        ]
        normalized = re.sub(r"\s+", " ", original_cases).strip()
        self.assertEqual(
            hashlib.sha256(normalized.encode()).hexdigest(),
            APPROVED_EXISTING_POSE_HASH,
        )

    def test_bent_knees_pose_uses_stronger_leg_flexion(self) -> None:
        pose = SOURCE[
            SOURCE.index("case PosePreset.BentKneesAndElbows:"):
            SOURCE.index(
                "break;",
                SOURCE.index("case PosePreset.BentKneesAndElbows:"),
            )
        ]
        self.assertRegex(
            pose,
            r"SetLegPair\(\s*muscles,\s*-0\.38f\s*\*\s*reach,\s*"
            r"0\.16f\s*\*\s*reach,\s*-0\.34f\s*\*\s*knee\s*\)",
        )
        self.assertRegex(
            pose,
            r"ApplyClosedHands\(\s*muscles,\s*s\s*\)",
        )

    def test_bent_knees_pose_curls_both_hands(self) -> None:
        both_hands = _block_after("void ApplyClosedHands")
        self.assertIn('ApplyClosedHand(muscles, "Left", strength);',
                      both_hands)
        self.assertIn('ApplyClosedHand(muscles, "Right", strength);',
                      both_hands)

        hand = _block_after("void ApplyClosedHand(")
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            self.assertIn(f'"{finger}"', hand)
        self.assertNotRegex(hand, r"(?<!-)\b0\.(?:55|65|68|70|75|78|80|82)f")

        curl = _block_after("void SetFingerCurl")
        for segment in ("1 Stretched", "2 Stretched", "3 Stretched"):
            self.assertIn(f'"{segment}"', curl)

    def test_finger_shape_assignments_cover_every_pose(self) -> None:
        order = _preset_order()
        cases = _finger_shape_assignment_cases()
        self.assertEqual(len(order), 31)
        self.assertEqual([preset for preset, _, _, _ in cases], order)

        enum_shapes = set(_enum_members("enum FingerShape"))
        self.assertTrue(EXPECTED_FINGER_SHAPES.issubset(enum_shapes))

        used_shapes: set[str] = set()
        for preset, left, right, body in cases:
            self.assertEqual(
                len(left), 1,
                f"{preset} must assign exactly one left FingerShape")
            self.assertEqual(
                len(right), 1,
                f"{preset} must assign exactly one right FingerShape")
            self.assertRegex(body, r"\bbreak\s*;")
            used_shapes.update((left[0], right[0]))

        self.assertTrue(used_shapes.issubset(enum_shapes))
        self.assertGreaterEqual(
            len(used_shapes), 10,
            "The pose library must exercise at least ten hand shapes")

    def test_finger_shape_assignments_respect_pose_symmetry(self) -> None:
        assignments = {
            preset: (left[0], right[0])
            for preset, left, right, _ in _finger_shape_assignment_cases()
            if len(left) == 1 and len(right) == 1
        }
        self.assertEqual(
            assignments.get("BentKneesAndElbows"),
            ("Fist", "Fist"),
        )

        self_symmetric = set(re.findall(
            r"PosePreset\.(\w+)",
            _block_after("static bool IsSelfSymmetricPreset"),
        ))
        for preset in self_symmetric:
            self.assertIn(preset, assignments)
            self.assertEqual(
                assignments[preset][0],
                assignments[preset][1],
                f"Self-symmetric {preset} must use the same shape per hand",
            )

        mirror_switch = _block_after_in(
            _block_after("static bool TryGetMirrorPreset"),
            "switch (preset)",
        )
        mirrors: dict[str, str] = {}
        for preset, body in _switch_cases(mirror_switch, "PosePreset"):
            opposite = re.findall(
                r"\bopposite\s*=\s*PosePreset\.(\w+)\s*;", body)
            self.assertEqual(
                len(opposite), 1,
                f"{preset} must declare exactly one mirror preset")
            mirrors[preset] = opposite[0]

        for preset, opposite in mirrors.items():
            self.assertEqual(mirrors.get(opposite), preset)
            self.assertEqual(
                assignments[opposite],
                assignments[preset][::-1],
                f"{preset}/{opposite} FingerShapes must be exact L/R mirrors",
            )

    def test_finger_shape_catalog_has_broad_numeric_variation(self) -> None:
        shape_enum = _enum_members("enum FingerShape")
        shape_switch = _block_after_in(
            _block_after("void ApplyFingerShape("),
            "switch (shape)",
        )
        shape_cases = _switch_cases(shape_switch, "FingerShape")
        self.assertEqual([name for name, _ in shape_cases], shape_enum)

        authored_calls = (
            "ApplyClosedHand",
            "SetAllFingerCurls",
            "SetFourFingerCurls",
            "SetFingerCurl",
            "SetFingerSpread",
        )
        signatures: dict[str, tuple[str, ...]] = {}
        numeric_values: set[float] = set()
        for shape, body in shape_cases:
            calls = tuple(_named_calls(body, authored_calls))
            self.assertTrue(calls, f"{shape} must author finger channels")
            signatures[shape] = calls
            for call in calls:
                numeric_values.update(_numeric_literals(call))

        self.assertGreaterEqual(
            len(set(signatures.values())), 10,
            "FingerShape cases must define at least ten distinct gestures",
        )
        self.assertGreaterEqual(
            len(numeric_values), 10,
            "FingerShape definitions need broad numeric curl/spread values",
        )
        self.assertTrue(any(value < 0 for value in numeric_values))
        self.assertTrue(any(value > 0 for value in numeric_values))

        open_values = tuple(
            value
            for call in signatures["Open"]
            for value in _numeric_literals(call)
        )
        self.assertGreaterEqual(len(open_values), 3)
        self.assertTrue(
            all(value >= 0 for value in open_values),
            "Open must explicitly straighten rather than curl the fingers",
        )

    def test_finger_shapes_apply_after_body_switch(self) -> None:
        body = _block_after("void ApplyPresetMuscles(")
        _, switch_end = _block_bounds_after_in(body, "switch (preset)")
        calls = list(re.finditer(
            r"\bApplyFingerVariation\s*\(\s*preset\s*,\s*muscles\s*,\s*s\s*\)"
            r"\s*;",
            body,
        ))
        self.assertEqual(len(calls), 1)
        self.assertGreater(
            calls[0].start(), switch_end,
            "Finger variation must override inherited fingers after body posing",
        )

        variation = _block_after("void ApplyFingerVariation(")
        _, mapping_end = _block_bounds_after_in(
            variation, "switch (preset)")
        left_call = variation.index(
            'ApplyFingerShape(muscles, "Left", leftShape, strength);')
        right_call = variation.index(
            'ApplyFingerShape(muscles, "Right", rightShape, strength);')
        self.assertGreater(left_call, mapping_end)
        self.assertGreater(right_call, left_call)

    def test_finger_gestures_are_optional_deterministic_and_mirror_safe(
        self,
    ) -> None:
        self.assertRegex(
            SOURCE,
            r"\[Range\(0f,\s*1f\)\]\s*"
            r"public float fingerGestureFrequency\s*=\s*0\.35f\s*;",
        )
        self.assertRegex(
            SOURCE,
            r"public int fingerGestureSeed\s*=\s*5\s*;",
        )

        variation = _block_after("void ApplyFingerVariation(")
        gate = variation.index("if (!ShouldUseFingerGesture(preset))")
        mapping = variation.index("switch (preset)")
        self.assertLess(gate, mapping)
        inactive = _block_after_in(
            variation, "if (!ShouldUseFingerGesture(preset))")
        self.assertIn("ResetBothHandsToNeutral(muscles);", inactive)
        self.assertIn("return;", inactive)

        preset_gate = _block_after(
            "bool ShouldUseFingerGesture(PosePreset preset)")
        self.assertIn("TryGetMirrorPreset(preset, out opposite)", preset_gate)
        self.assertRegex(
            preset_gate,
            r"Mathf\.Min\(\s*key\s*,\s*\(int\)opposite\s*\)",
        )
        deterministic_gate = _block_after(
            "bool ShouldUseFingerGesture(int stablePoseKey)")
        self.assertIn("Mathf.Clamp01(fingerGestureFrequency)", deterministic_gate)
        self.assertIn("fingerGestureSeed", deterministic_gate)
        self.assertIn("return sample < frequency;", deterministic_gate)
        self.assertNotRegex(
            deterministic_gate, r"\b(?:UnityEngine\.)?Random\s*\.")

        neutral = _block_after("void ResetBothHandsToNeutral")
        self.assertIn('ResetFingerChannels(muscles, "Left");', neutral)
        self.assertIn('ResetFingerChannels(muscles, "Right");', neutral)

        animation = _block_after("bool ApplyAnimationFrame")
        self.assertIn(
            "if (!ShouldUseFingerGesture(animationSample.StablePoseKey))",
            animation,
        )
        self.assertIn("ResetBothHandsToNeutral(pose.muscles);", animation)
        animation_sample = _block_after("bool TryGetAnimationSample")
        self.assertIn(
            "int.MinValue + frameIndex",
            animation_sample,
            "Legacy single-clip samples must keep their established gesture key",
        )

        # Mirror the fixed integer mixer to ensure the shipped defaults really
        # produce a mix rather than accidentally selecting all or no presets.
        bits = _enum_bits()
        order = _preset_order()
        mirror_switch = _block_after_in(
            _block_after("static bool TryGetMirrorPreset"),
            "switch (preset)",
        )
        mirrors: dict[str, str] = {}
        for preset, body in _switch_cases(mirror_switch, "PosePreset"):
            opposite = re.findall(
                r"\bopposite\s*=\s*PosePreset\.(\w+)\s*;", body)
            if opposite:
                mirrors[preset] = opposite[0]

        def sample(stable_key: int) -> float:
            value = stable_key & 0xFFFFFFFF
            value ^= (5 + 0x9E3779B9) & 0xFFFFFFFF
            value ^= value >> 16
            value = (value * 0x7FEB352D) & 0xFFFFFFFF
            value ^= value >> 15
            value = (value * 0x846CA68B) & 0xFFFFFFFF
            value ^= value >> 16
            return (value & 0x00FFFFFF) / 16777216.0

        selected: dict[str, bool] = {}
        for preset in order:
            key = 1 << bits[preset]
            opposite = mirrors.get(preset)
            if opposite is not None:
                key = min(key, 1 << bits[opposite])
            selected[preset] = sample(key) < 0.35

        active_count = sum(selected.values())
        self.assertGreater(active_count, 0)
        self.assertLess(active_count, len(order) // 2)
        self.assertFalse(selected["TPose"])
        self.assertFalse(selected["RelaxedStanding"])
        for preset, opposite in mirrors.items():
            self.assertEqual(selected[preset], selected[opposite])

    def test_finger_channels_reset_and_use_absolute_values(self) -> None:
        apply_shape = _block_after("void ApplyFingerShape(")
        reset_call = apply_shape.index(
            "ResetFingerChannels(muscles, side);")
        shape_switch = apply_shape.index("switch (shape)")
        self.assertLess(reset_call, shape_switch)

        reset = _block_after("void ResetFingerChannels")
        reset_calls = _named_calls(
            reset, ("SetFingerCurl", "SetFingerSpread"))
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            self.assertIn(
                f'SetFingerCurl(muscles,side,"{finger}",0f,0f,0f)',
                reset_calls,
            )
            self.assertIn(
                f'SetFingerSpread(muscles,side,"{finger}",0f)',
                reset_calls,
            )
        self.assertNotIn("SetMuscleOffset", reset)

        curl = _block_after("void SetFingerCurl")
        self.assertEqual(curl.count("SetMuscleAbsolute("), 3)
        self.assertNotIn("SetMuscleOffset", curl)
        for segment in ("1 Stretched", "2 Stretched", "3 Stretched"):
            self.assertIn(f'"{segment}"', curl)

        spread = _block_after("void SetFingerSpread")
        self.assertEqual(spread.count("SetMuscleAbsolute("), 1)
        self.assertNotIn("SetMuscleOffset", spread)
        self.assertIn('" Spread"', spread)

        absolute = _block_after("void SetMuscleAbsolute")
        self.assertNotIn("BaseMuscleValue", absolute)
        self.assertRegex(
            absolute,
            r"muscles\s*\[\s*index\s*\]\s*=\s*"
            r"Mathf\.Clamp\(\s*value\s*,\s*-1f\s*,\s*1f\s*\)\s*;",
        )

    def test_self_intersection_guard_preserves_authored_fingers(self) -> None:
        guard = _block_after(
            "void ApplyHumanPoseWithSelfIntersectionGuard(")
        restore_pattern = (
            r"RestoreFingerChannels\s*\(\s*pose\.muscles\s*,\s*"
            r"desired\s*\)\s*;"
        )
        restores = list(re.finditer(restore_pattern, guard))
        self.assertEqual(
            len(restores), 2,
            "Body attenuation and its fallback must restore target fingers",
        )

        lerp = guard.index("Mathf.Lerp(")
        loop_set_pose = guard.index(
            "_humanPoseHandler.SetHumanPose(ref pose);", lerp)
        self.assertGreater(restores[0].start(), lerp)
        self.assertLess(restores[0].end(), loop_set_pose)

        fallback_start = min(
            guard.index("Array.Copy("),
            guard.index("Array.Clear("),
        )
        fallback_set_pose = guard.index(
            "_humanPoseHandler.SetHumanPose(ref pose);", loop_set_pose + 1)
        self.assertGreater(restores[1].start(), fallback_start)
        self.assertLess(restores[1].end(), fallback_set_pose)

        desired_snapshot = guard.index(
            "float[] desired = (float[])pose.muscles.Clone();")
        self.assertLess(desired_snapshot, lerp)
        self.assertNotIn("ApplyFingerVariation(", guard)

        restore_start = SOURCE.index(
            "void RestoreFingerChannels(float[] muscles, float[] desired)")
        restore_end = SOURCE.index(
            "void SetAllFingerCurls(", restore_start)
        restore = SOURCE[restore_start:restore_end]
        for side in ("Left", "Right"):
            self.assertIn(f'"{side}"', restore)
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            self.assertIn(f'"{finger}"', restore)
        for channel in (
            "1 Stretched",
            "2 Stretched",
            "3 Stretched",
            "Spread",
        ):
            self.assertIn(f'"{channel}"', restore)
        self.assertIn("muscles[index] = desired[index];", restore)
        self.assertNotIn("SetMuscleAbsolute", restore)
        self.assertNotIn("SetMuscleOffset", restore)

    def test_flags_are_unique_and_all_presets_are_selected(self) -> None:
        bits = _enum_bits()
        order = _preset_order()
        self.assertEqual(list(bits), order)
        self.assertEqual(list(bits.values()), list(range(len(order))))
        self.assertLessEqual(max(bits.values()), 30)

        all_expression = SOURCE[
            SOURCE.index("public const PosePreset AllPosePresets ="):
            SOURCE.index(
                ";",
                SOURCE.index("public const PosePreset AllPosePresets ="),
            )
        ]
        selected = re.findall(r"PosePreset\.(\w+)", all_expression)
        self.assertIn("CorePosePresets", all_expression)
        core_expression = SOURCE[
            SOURCE.index("public const PosePreset CorePosePresets ="):
            SOURCE.index(
                ";",
                SOURCE.index("public const PosePreset CorePosePresets ="),
            )
        ]
        core = re.findall(r"PosePreset\.(\w+)", core_expression)
        self.assertEqual(core, order[:len(core)])
        self.assertEqual(core + selected, order)

    def test_previous_full_selection_is_migrated(self) -> None:
        for name, width in (
            ("LegacyAllPosePresets", 13),
            ("PreviousAllPosePresets", 18),
            ("PreviousFullPosePresets", 24),
        ):
            self.assertRegex(
                SOURCE,
                rf"{name}\s*=\s*"
                rf"\(PosePreset\)\(\(1\s*<<\s*{width}\)\s*-\s*1\)",
            )
        migration = _block_after("static bool IsLegacyAllPresetMask")
        for name in (
            "LegacyAllPosePresets",
            "PreviousAllPosePresets",
            "PreviousFullPosePresets",
        ):
            self.assertIn(f"presets == {name}", migration)
        effective_selection = SOURCE[
            SOURCE.index("PosePreset EffectiveSelectedPresets"):
            SOURCE.index(
                ";",
                SOURCE.index("PosePreset EffectiveSelectedPresets"),
            )
        ]
        self.assertIn("IsLegacyAllPresetMask(selectedPresets)",
                      effective_selection)
        self.assertIn("? AllPosePresets", effective_selection)
        self.assertIn(": selectedPresets", effective_selection)
        selected_count = _block_after("int CountSelectedPresets")
        self.assertIn(
            "PosePreset presets = EffectiveSelectedPresets;",
            selected_count,
        )

    def test_new_presets_have_switch_and_symmetry_classification(self) -> None:
        apply_body = _block_after("void ApplyPresetMuscles")
        for preset in NEW_PRESETS:
            self.assertIn(f"case PosePreset.{preset}:", apply_body)
        self.assertRegex(
            apply_body,
            r"ApplyTorsoSideBend\(\s*muscles,\s*true,\s*s,\s*reach\s*\)")
        self.assertRegex(
            apply_body,
            r"ApplyTorsoSideBend\(\s*muscles,\s*false,\s*s,\s*reach\s*\)")
        self.assertRegex(
            apply_body,
            r"ApplyArmTwistCoverage\(\s*muscles,\s*-1f,\s*reach,\s*elbow\)")
        self.assertRegex(
            apply_body,
            r"ApplyArmTwistCoverage\(\s*muscles,\s*1f,\s*reach,\s*elbow\)")
        self.assertRegex(
            apply_body,
            r"ApplyLegTwistAndFootCoverage\(\s*muscles,\s*-1f,\s*reach\)")
        self.assertRegex(
            apply_body,
            r"ApplyLegTwistAndFootCoverage\(\s*muscles,\s*1f,\s*reach\)")

        right_mirrors = _block_after("static bool IsRightMirrorPreset")
        mirror_pairs = _block_after("static bool TryGetMirrorPreset")
        self.assertIn("PosePreset.TorsoSideBendRight", right_mirrors)
        self.assertIn(
            "case PosePreset.TorsoSideBendLeft:", mirror_pairs)
        self.assertIn(
            "opposite = PosePreset.TorsoSideBendRight;", mirror_pairs)
        self.assertIn(
            "case PosePreset.TorsoSideBendRight:", mirror_pairs)
        self.assertIn(
            "opposite = PosePreset.TorsoSideBendLeft;", mirror_pairs)

        self_symmetric = _block_after(
            "static bool IsSelfSymmetricPreset")
        for preset in (
            "BackArchLookUp",
            "ArmTwistInWristsDown",
            "ArmTwistOutWristsUp",
            "LegsTurnInAnklesFlexed",
            "LegsTurnOutAnklesPointed",
        ):
            self.assertIn(f"PosePreset.{preset}", self_symmetric)
        self.assertNotIn("PosePreset.TorsoSideBendLeft", self_symmetric)
        self.assertNotIn("PosePreset.TorsoSideBendRight", self_symmetric)

    def test_previously_empty_muscle_channels_are_now_authored(self) -> None:
        back_arch = _block_after("void ApplyBackArchLookUp")
        for channel in (
            "Spine Front-Back",
            "Chest Front-Back",
            "UpperChest Front-Back",
            "Neck Nod Down-Up",
            "Head Nod Down-Up",
        ):
            self.assertIn(f'"{channel}"', back_arch)
        self.assertIn("SetArmPair(", back_arch)

        side_bend = _block_after("void ApplyTorsoSideBend")
        for channel in (
            "Spine Left-Right",
            "Chest Left-Right",
            "UpperChest Left-Right",
            "Neck Tilt Left-Right",
            "Head Tilt Left-Right",
        ):
            self.assertIn(f'"{channel}"', side_bend)

        arm_twist = _block_after("void ApplyArmTwistCoverage")
        for channel in (
            "Arm Twist In-Out",
            "Forearm Twist In-Out",
            "Hand Down-Up",
            "Hand In-Out",
            "Neck Nod Down-Up",
            "Head Nod Down-Up",
        ):
            self.assertIn(f'"{channel}"', arm_twist)

        leg_twist = _block_after("void ApplyLegTwistAndFootCoverage")
        for channel in (
            "Upper Leg Twist In-Out",
            "Lower Leg Twist In-Out",
            "Foot Twist In-Out",
            "Foot Up-Down",
            "Toes Up-Down",
        ):
            self.assertIn(f'"{channel}"', leg_twist)
        self.assertIn("SetUpperLegPair(", leg_twist)

        forward_lean = SOURCE[
            SOURCE.index("case PosePreset.ForwardLean:"):
            SOURCE.index(
                "break;",
                SOURCE.index("case PosePreset.ForwardLean:"),
            )
        ]
        self.assertNotIn('"Neck Nod Down-Up"', forward_lean)
        self.assertNotIn('"Head Nod Down-Up"', forward_lean)

    def test_every_body_humanoid_channel_has_an_authoring_path(self) -> None:
        global_channels = (
            "Spine Front-Back",
            "Spine Left-Right",
            "Spine Twist Left-Right",
            "Chest Front-Back",
            "Chest Left-Right",
            "Chest Twist Left-Right",
            "UpperChest Front-Back",
            "UpperChest Left-Right",
            "UpperChest Twist Left-Right",
            "Neck Nod Down-Up",
            "Neck Tilt Left-Right",
            "Neck Turn Left-Right",
            "Head Nod Down-Up",
            "Head Tilt Left-Right",
            "Head Turn Left-Right",
        )
        bilateral_channel_suffixes = (
            "Upper Leg Front-Back",
            "Upper Leg In-Out",
            "Upper Leg Twist In-Out",
            "Lower Leg Stretch",
            "Lower Leg Twist In-Out",
            "Foot Up-Down",
            "Foot Twist In-Out",
            "Toes Up-Down",
            "Shoulder Down-Up",
            "Shoulder Front-Back",
            "Arm Down-Up",
            "Arm Front-Back",
            "Arm Twist In-Out",
            "Forearm Stretch",
            "Forearm Twist In-Out",
            "Hand Down-Up",
            "Hand In-Out",
        )
        for channel in global_channels + bilateral_channel_suffixes:
            self.assertTrue(
                f'"{channel}"' in SOURCE or f'" {channel}"' in SOURCE,
                channel,
            )

    def test_inspector_reports_detection_and_limb_propagation_separately(
        self,
    ) -> None:
        start = EDITOR_SOURCE.index("static void DrawSymmetryPipeline")
        end = EDITOR_SOURCE.index("static void SetPresets", start)
        panel = EDITOR_SOURCE[start:end]

        for label in (
            "Base Symmetry Detected",
            "Next Batch Pose Pairing",
            "Pose -> Paired Limbs",
            "Stored Target Modes",
        ):
            self.assertIn(f'"{label}"', panel)
        for source in (
            "fitter.SymmetryBatchActive",
            "fitter.SymmetryEvaluationCompleted",
            "fitter.SymmetryEvaluationPending",
            "fitter.DebugResolvedConnector",
            "fitter.DebugResolvedMorphDriver",
            "driver.BaseSymmetryActive",
            "driver.BaseSymmetryPairCount",
            "driver.BaseSymmetryAxisName",
            "driver.StoredMorphTargetSlotCount",
            "driver.GetMorphPropagationStatus",
        ):
            self.assertIn(source, panel)
        for state in ("Checking base fit", "YES (", "NO (", "ACTIVE (",
                      "PARTIAL (", "WARNING (",
                      "Pending (run the base fit)",
                      "Unavailable (saved morph driver missing)",
                      "Unavailable (connector/driver missing)",
                      "Unavailable (connector required to evaluate)",
                      "Enabled (waiting for fitted targets)"):
            self.assertIn(state, panel)
        self.assertIn(
            "symmetry status are created when the base fit is saved",
            panel,
        )
        self.assertRegex(
            panel,
            r"symmetricTargets\s*==\s*0\s*&&\s*independentTargets\s*==\s*0",
        )
        self.assertIn("else if (invalidTargetSlots > 0)", panel)
        self.assertIn("else if (!baseRuntimeDataValid)", panel)
        self.assertLess(
            panel.index("else if (!baseRuntimeDataValid)"),
            panel.index("else if (invalidTargetSlots > 0)"),
        )
        self.assertLess(
            panel.index("else if (invalidTargetSlots > 0)"),
            panel.index(
                "else if (symmetricTargets == 0 && "
                "independentTargets == 0)",
            ),
        )
        detection = panel[
            panel.index("string detectionStatus"):
            panel.index("string pairingStatus")
        ]
        self.assertLess(
            detection.index("if (checking)"),
            detection.index("else if (driver == null"),
        )
        self.assertLess(
            detection.index("else if (evaluationPending)"),
            detection.index("else if (driver == null"),
        )
        self.assertLess(
            detection.index("else if (detected)"),
            detection.index(
                "else if (driver == null && "
                "fitter.SymmetryEvaluationCompleted)",
            ),
        )
        self.assertLess(
            detection.index(
                "else if (driver == null && "
                "fitter.SymmetryEvaluationCompleted)",
            ),
            detection.index(
                "else if (driver == null && resolvedConnector == null)",
            ),
        )
        pairing = panel[
            panel.index("string pairingStatus"):
            panel.index("string propagationStatus")
        ]
        self.assertLess(
            pairing.index("if (checking)"),
            pairing.index("else if (resolvedConnector == null)"),
        )
        self.assertLess(
            pairing.index("else if (evaluationPending)"),
            pairing.index("else if (resolvedConnector == null)"),
        )
        summary = panel[panel.index("MessageType messageType"):]
        self.assertLess(
            summary.index("if (checking)"),
            summary.index("else if (driver == null"),
        )
        self.assertLess(
            summary.index("else if (evaluationPending)"),
            summary.index("else if (driver == null"),
        )
        self.assertLess(
            summary.index("else if (propagationActive)"),
            summary.index("else if (!requested)"),
        )
        self.assertLess(
            summary.index(
                "else if (detected && invalidTargetSlots > 0)",
            ),
            summary.index(
                "else if (symmetricTargets == 0 && "
                "independentTargets == 0)",
            ),
        )

    def test_propagation_indicator_is_backed_by_runtime_symmetry_pipeline(
        self,
    ) -> None:
        self.assertIn("public int MirroredBonePairCount", DRIVER_SOURCE)
        self.assertIn(
            "public bool SymmetricMorphPropagationActive",
            DRIVER_SOURCE,
        )
        self.assertIn(
            "public bool SymmetricBaseRuntimeDataValid",
            DRIVER_SOURCE,
        )
        self.assertIn(
            "public void GetMorphPropagationStatus",
            DRIVER_SOURCE,
        )
        self.assertIn(
            "public void GetMorphTargetSymmetryCounts",
            DRIVER_SOURCE,
        )
        self.assertIn(
            "HasValidSymmetricRuntimeData(",
            DRIVER_SOURCE,
        )
        for mode in (
            "EllipSDFMorphTargetSymmetry.SelfSymmetric",
            "EllipSDFMorphTargetSymmetry.GenerateMirroredSupport",
        ):
            self.assertIn(mode, DRIVER_SOURCE)

        propagation_status = _block_after_in(
            DRIVER_SOURCE,
            "public void GetMorphPropagationStatus",
        )
        for requirement in (
            "baseSymmetryActive",
            "_cachedSelfSymmetricTargetCount +",
            "_cachedMirroredSupportTargetCount > 0",
            "_cachedIndependentTargetCount == 0",
            "_cachedSymmetricRuntimeDataValid",
            "_cachedMirroredSupportTargetCount == 0",
            "_cachedMirroredBonePairCount > 0",
        ):
            self.assertIn(requirement, propagation_status)
        self.assertIn(
            "_morphPropagationStatusVersion != _burstDataVersion",
            propagation_status,
        )
        self.assertIn(
            "_morphPropagationHierarchySignature != hierarchySignature",
            propagation_status,
        )
        self.assertIn(
            "_morphPropagationDataSignature != dataSignature",
            propagation_status,
        )
        for invalidation in (
            "_runtimeMorphTargetsVersion = -1;",
            "_burstCacheDirty = true;",
            "_weightCacheDirty = true;",
            "_transformAccessDirty = true;",
        ):
            self.assertIn(invalidation, propagation_status)
        self.assertRegex(
            propagation_status,
            r"(?s)_cachedSymmetricMorphPropagationActive\s*=\s*"
            r"baseSymmetryActive\s*&&\s*"
            r"_cachedSelfSymmetricTargetCount\s*\+\s*"
            r"_cachedMirroredSupportTargetCount\s*>\s*0\s*&&\s*"
            r"_cachedIndependentTargetCount\s*==\s*0\s*&&\s*"
            r"_cachedSymmetricRuntimeDataValid\s*&&\s*"
            r"\(\s*_cachedMirroredSupportTargetCount\s*==\s*0\s*"
            r"\|\|\s*_cachedMirroredBonePairCount\s*>\s*0\s*\)\s*;",
        )
        self.assertIn(
            "_morphPropagationStatusVersion = _burstDataVersion;",
            propagation_status,
        )
        self.assertIn(
            "propagationActive = "
            "_cachedSymmetricMorphPropagationActive;",
            propagation_status,
        )
        self.assertIn(
            "baseRuntimeDataValid = "
            "_cachedSymmetricBaseRuntimeDataValid;",
            propagation_status,
        )
        self.assertRegex(
            propagation_status,
            r"_cachedMirroredBonePairCount\s*=\s*"
            r"\n?\s*baseSymmetryActive\s*"
            r"\n?\s*\?\s*CountMirroredBonePairs\(mirroredBoneIndices\)",
        )

        runtime_start = DRIVER_SOURCE.index(
            "List<EllipSDFMorphTarget> RuntimeMorphTargets()")
        runtime_end = DRIVER_SOURCE.index(
            "EllipSDFMorphTarget BuildSelfSymmetricRuntimeTarget",
            runtime_start,
        )
        runtime = DRIVER_SOURCE[runtime_start:runtime_end]
        self.assertIn("BuildSelfSymmetricRuntimeTarget(target)", runtime)
        self.assertIn(
            "BuildMirroredRuntimeTarget(target, mirroredBoneIndices)",
            runtime,
        )

        mirrored_start = DRIVER_SOURCE.index(
            "EllipSDFMorphTarget BuildMirroredRuntimeTarget")
        mirrored_end = DRIVER_SOURCE.index(
            "EllipSDFMorphTarget CloneTargetHeader",
            mirrored_start,
        )
        mirrored = DRIVER_SOURCE[mirrored_start:mirrored_end]
        self.assertIn(
            "BuildMirroredPose(source.pose, mirroredBoneIndices)",
            mirrored,
        )
        self.assertNotIn(
            "reflected ?? FindDeltaById(source, destination.id)",
            mirrored,
        )

        start_batch = _block_after("public void StartBatch")
        self.assertIn(
            "_symmetryEvaluationCompleted = false;",
            start_batch,
        )
        self.assertRegex(
            start_batch,
            r"_symmetryEvaluationPending\s*=\s*"
            r"\n?\s*symmetryMode\s*==\s*SymmetryMode\.Auto;",
        )

        refresh = _block_after("void RefreshSymmetryBatchState")
        self.assertIn("_symmetryEvaluationCompleted = true;", refresh)
        self.assertIn("_symmetryEvaluationPending = false;", refresh)
        self.assertEqual(
            SOURCE.count("_symmetryEvaluationCompleted = true;"),
            1,
        )

        on_validate = _block_after("void OnValidate")
        mode_off = _block_after_in(on_validate, "if (symmetryMode ==")
        self.assertIn("_symmetryEvaluationCompleted = false;", mode_off)
        self.assertIn("_symmetryEvaluationPending = false;", mode_off)

        self.assertIn(
            "public bool SymmetryEvaluationCompleted",
            SOURCE,
        )
        self.assertIn(
            "public bool SymmetryEvaluationPending",
            SOURCE,
        )
        inspector_start = EDITOR_SOURCE.index(
            "public override void OnInspectorGUI()")
        inspector_end = EDITOR_SOURCE.index(
            "static void DrawSymmetryPipeline",
            inspector_start,
        )
        self.assertIn(
            "DrawSymmetryPipeline(fitter);",
            EDITOR_SOURCE[inspector_start:inspector_end],
        )

        driver_resolution = _block_after(
            "public EllipSDFMorphDriver DebugResolvedMorphDriver",
        )
        self.assertIn(
            "return resolvedConnector.DebugMorphDriver;",
            driver_resolution,
        )
        self.assertNotIn("GetComponentInParent", driver_resolution)
        self.assertNotIn("GetComponentInChildren", driver_resolution)

        invalidate = _block_after_in(
            DRIVER_SOURCE,
            "void InvalidateBurstCache",
        )
        self.assertIn("_burstDataVersion++;", invalidate)

        capture_base_pose = _block_after_in(
            DRIVER_SOURCE,
            "public void CaptureBasePose",
        )
        self.assertIn("InvalidateBurstCache();", capture_base_pose)

    def test_active_indicator_requires_complete_runtime_coverage(self) -> None:
        runtime_validation = _block_after_in(
            DRIVER_SOURCE,
            "bool HasValidSymmetricRuntimeData",
        )
        for requirement in (
            "if (!baseDataValid || morphTargets == null)",
            "HasCompleteRendererPose(target.pose)",
            "HasExactEllipsoidDeltaCoverage(target)",
            "CanBuildCompleteMirroredDeltaSet(target)",
            "CanBuildCompleteSelfSymmetricDeltaSet(target)",
            "HasCompleteMirroredBoneMapping(mirroredBoneIndices)",
            "HasDirectRendererBoneParenting()",
            "IsSelfSymmetricPose(",
            "RuntimeMorphTargets()",
            "runtimeTargets.Count != expectedRuntimeTargetCount",
            "HasCompleteRendererPose(runtimeTarget.pose)",
            "HasExactEllipsoidDeltaCoverage(runtimeTarget)",
        ):
            self.assertIn(requirement, runtime_validation)

        base_validation = _block_after_in(
            DRIVER_SOURCE,
            "bool HasValidSymmetricBaseData",
        )
        self.assertIn("HasCompleteRendererPose(basePose)", base_validation)
        self.assertIn(
            "HasCompleteBaseSymmetryClassification()",
            base_validation,
        )
        self.assertIn(
            "HasCompleteMirroredBoneMapping(mirroredBoneIndices)",
            base_validation,
        )
        self.assertIn(
            "HasDirectRendererBoneParenting()",
            base_validation,
        )
        self.assertIn(
            "IsSelfSymmetricPose(basePose, mirroredBoneIndices)",
            base_validation,
        )

        pose_validation = _block_after_in(
            DRIVER_SOURCE,
            "bool HasCompleteRendererPose",
        )
        for requirement in (
            "if (rendererBones[i] == null)",
            "pose.bones.Count != rendererBones.Length",
            "covered[bone.index]",
            "bone.transform != rendererBones[bone.index]",
            "StringComparison.Ordinal",
            "!bone.hasRendererLocalTransform",
            "bone.parentIndex == bone.index",
            "FindNearestRendererBoneParentIndex(",
            "bone.parentIndex != actualParentIndex",
            "rendererBones[i] != null && !covered[i]",
        ):
            self.assertIn(requirement, pose_validation)

        symmetry_classification = _block_after_in(
            DRIVER_SOURCE,
            "bool HasCompleteBaseSymmetryClassification",
        )
        for requirement in (
            "HashSet<int> baseIds",
            "!baseIds.Add(item.id)",
            "HashSet<int> classified",
            "!classified.Add(id)",
            "!classified.Add(pair.sourceId)",
            "!classified.Add(pair.mirrorId)",
            "classified.SetEquals(baseIds)",
        ):
            self.assertIn(requirement, symmetry_classification)

        delta_validation = _block_after_in(
            DRIVER_SOURCE,
            "bool HasExactEllipsoidDeltaCoverage",
        )
        for requirement in (
            "target.ellipsoids.Count != baseEllipsoids.Count",
            "HashSet<int> expectedIds",
            "HashSet<int> actualIds",
            "!expectedIds.Contains(delta.id)",
            "!actualIds.Add(delta.id)",
            "actualIds.SetEquals(expectedIds)",
        ):
            self.assertIn(requirement, delta_validation)

        bone_mapping = _block_after_in(
            DRIVER_SOURCE,
            "bool HasCompleteMirroredBoneMapping",
        )
        for requirement in (
            "RequiredLeftLimbBones",
            "!avatar.avatar.isHuman",
            "HashSet<int> expectedSidedBones",
            "hasLeft != hasRight",
            "opposite == index",
            "mirroredBoneIndices[opposite] != index",
            "expectedOppositeParent",
            "oppositeParentIndex != expectedOppositeParent",
        ):
            self.assertIn(requirement, bone_mapping)

        hierarchy_signature = _block_after_in(
            DRIVER_SOURCE,
            "int ComputeMorphPropagationHierarchySignature",
        )
        for requirement in (
            "skinnedMesh.GetInstanceID()",
            "bones.Length",
            "bone.GetInstanceID()",
            "bone.parent.GetInstanceID()",
            "bone.name.GetHashCode()",
            "avatar.avatar.GetInstanceID()",
        ):
            self.assertIn(requirement, hierarchy_signature)

        data_signature = _block_after_in(
            DRIVER_SOURCE,
            "int ComputeMorphPropagationDataSignature",
        )
        for requirement in (
            "baseSymmetryActive",
            "baseSymmetryReflectionRendererLocal",
            "AppendPoseSignature(hash, basePose)",
            "baseEllipsoids.Count",
            "baseSymmetryOnPlaneIds.Count",
            "baseSymmetryPairs.Count",
            "morphTargets.Count",
            "(int)target.symmetry",
            "target.ellipsoids.Count",
            "delta.id",
            "delta.deltaLocalCenter",
            "delta.deltaLocalRotation",
            "delta.deltaLogRadii",
        ):
            self.assertIn(requirement, data_signature)

        self_symmetric_pose = _block_after_in(
            DRIVER_SOURCE,
            "bool IsSelfSymmetricPose",
        )
        for requirement in (
            "BuildMirroredPose(pose, mirroredBoneIndices)",
            "HasCompleteRendererPose(mirrored)",
            "mirroredBase",
            "sourceBaseBone",
            "mirroredBaseBone",
            "sourceBone.localPosition",
            "sourceBone.localRotation",
            "mirroredBoneIndices[sourceBone.index] == sourceBone.index",
            "comparedSidedBones > 0",
            "Quaternion.Angle(",
            "SelfSymmetricPosePositionTolerance",
            "SelfSymmetricPoseRotationToleranceDegrees",
        ):
            self.assertIn(requirement, self_symmetric_pose)

        direct_parenting = _block_after_in(
            DRIVER_SOURCE,
            "bool HasDirectRendererBoneParenting",
        )
        self.assertIn(
            "bone.parent != rendererBones[parentIndex]",
            direct_parenting,
        )


if __name__ == "__main__":
    unittest.main()
