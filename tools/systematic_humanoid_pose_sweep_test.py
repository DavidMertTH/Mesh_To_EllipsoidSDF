"""Static contracts for the systematic Unity Humanoid joint-angle sweep."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
FITTER_PATH = (
    ROOT / "clothSimulation" / "EllipsoidLoading"
    / "EllipSDFSyntheticPoseBatchFitter.cs"
)
EDITOR_PATH = (
    ROOT / "clothSimulation" / "EllipsoidLoading" / "Editor"
    / "EllipSDFSyntheticPoseBatchFitterEditor.cs"
)
FITTER = FITTER_PATH.read_text(encoding="utf-8")
EDITOR = EDITOR_PATH.read_text(encoding="utf-8")


def _block_after(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1:index]
    raise AssertionError(f"Unclosed C# block after {marker!r}")


class SystematicHumanoidPoseSweepTest(unittest.TestCase):
    def test_systematic_joint_angles_are_the_default_humanoid_source(self) -> None:
        self.assertRegex(
            FITTER,
            r"HumanoidPoseSource\s+humanoidPoseSource\s*=\s*"
            r"HumanoidPoseSource\.SystematicJointAngles",
        )
        groups = _block_after(FITTER, "public enum HumanoidJointGroup")
        for name in (
            "Torso", "HeadAndNeck", "ArmsAndWrists",
            "LegsAndFeet", "Fingers", "Body", "All",
        ):
            self.assertRegex(groups, rf"\b{name}\b")
        self.assertRegex(
            FITTER,
            r"HumanoidJointGroup\s+systematicJointGroups\s*=\s*"
            r"HumanoidJointGroup\.All",
        )
        validation = _block_after(FITTER, "void OnValidate()")
        self.assertIn("systematicPoseSettingsVersion < 1", validation)
        self.assertIn(
            "systematicJointGroups = HumanoidJointGroup.All;", validation)
        self.assertIn("systematicPoseSettingsVersion < 2", validation)
        self.assertIn("systematicParallelPassCount = 4;", validation)

    def test_plan_packs_independent_joints_into_parallel_targets(self) -> None:
        build = _block_after(FITTER, "bool BuildPosePlan(")
        self.assertIn("AddSystematicHumanoidAngleTargets(destination);", build)
        add = _block_after(FITTER, "void AddSystematicHumanoidAngleTargets(")
        self.assertIn("HumanTrait.MuscleCount", add)
        self.assertIn("string[] muscleNames = HumanTrait.MuscleName;", add)
        self.assertIn("muscleNames[muscleIndex]", add)
        self.assertIn("HumanTrait.BoneFromMuscle(muscleIndex)", add)
        self.assertIn("Dictionary<int, SystematicJointLane>", add)
        self.assertIn("lane.targets.Add(target);", add)
        self.assertIn(
            "passIndex < passCount; passIndex++", add)
        self.assertIn("SystematicLanePassOffset(", add)
        self.assertIn("poseIndex + phase + passOffset", add)
        self.assertIn("item.humanoidMuscleTargets = packedTargets.ToArray();", add)
        self.assertIn("item.kind = BatchPoseKind.HumanoidJointAngle;", add)

        count = _block_after(
            FITTER, "int CountSystematicHumanoidAngleTargets()")
        self.assertIn("systematicParallelPassCount", count)

        apply_pose = _block_after(FITTER, "bool ApplyHumanoidJointAngle(")
        self.assertIn("BuildConfiguredBaseMuscles();", apply_pose)
        self.assertIn("HumanoidMuscleTarget[] targets", apply_pose)
        self.assertIn("for (int i = 0; i < targets.Length; i++)", apply_pose)
        self.assertIn("muscles[muscleIndex] = Mathf.Clamp(", apply_pose)

    def test_each_direction_uses_the_avatar_joint_limits(self) -> None:
        angles = _block_after(FITTER, "void BuildSystematicAngles(")
        self.assertIn("HumanTrait.GetMuscleDefaultMin(muscleIndex)", angles)
        self.assertIn("HumanTrait.GetMuscleDefaultMax(muscleIndex)", angles)
        self.assertIn("negativeLimit", angles)
        self.assertIn("positiveLimit", angles)
        self.assertIn("systematicAngleStepDegrees", angles)
        self.assertIn("AddAngleMagnitudes", angles)

    def test_joint_types_are_selected_by_humanoid_channels(self) -> None:
        classify = _block_after(FITTER, "SystematicGroupForMuscle(")
        for channel in (
            "Spine ", "Head ", "Shoulder", "Arm", "Forearm",
            " Hand ", "Leg", "Foot", "Toes",
        ):
            self.assertIn(f'"{channel}"', classify)
        fingers = _block_after(FITTER, "static bool ContainsFingerName(")
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            self.assertIn(f'"{finger}"', fingers)

    def test_symmetry_handles_legacy_and_packed_sweeps(self) -> None:
        prune = _block_after(
            FITTER, "void PruneRedundantMirroredPresetsFromBatchPlan()")
        self.assertIn("BatchPoseKind.HumanoidJointAngle", prune)
        self.assertIn('StartsWith("Right "', prune)
        self.assertIn('"Left " + item.sourceName.Substring(6)', prune)
        configure = _block_after(FITTER, "void ResolveSavedTargetSymmetry(")
        self.assertIn("IsSystematicPoseSelfSymmetric", configure)
        self.assertIn("SystematicPoseHasSidedTargets", configure)
        self.assertIn("SystematicPoseHasAsymmetricCentralTarget", configure)
        self.assertIn("EllipSDFMorphTargetSymmetry.Independent", configure)

    def test_inspector_explains_anatomical_channel_handling(self) -> None:
        self.assertIn("Systematic Angle Targets", EDITOR)
        self.assertIn("many independent joints in", EDITOR)
        self.assertIn("phase-shifted passes", EDITOR)
        self.assertIn("Every pass recombines", EDITOR)
        self.assertIn("ball-joint axes are scheduled", EDITOR)
        self.assertIn("hinge,", EDITOR)
        self.assertIn("twist, wrist, ankle, and finger", EDITOR)

    def test_preview_plan_is_cached_instead_of_rebuilt_per_name(self) -> None:
        self.assertIn("bool _previewPosePlanValid;", FITTER)
        ensure = _block_after(FITTER, "bool EnsurePreviewPosePlan()")
        self.assertIn("_previewPosePlanConfigurationHash", ensure)
        self.assertIn("BuildPosePlan(_previewPosePlan", ensure)
        current = _block_after(FITTER, "int CurrentPosePlanCount()")
        self.assertNotIn("BuildPosePlan", current)
        names = _block_after(FITTER, "public void GetGeneratedPoseNames(")
        self.assertIn("ResolveCurrentPosePlanForRead()", names)
        self.assertNotIn("GetGeneratedPoseName(", names)

    def test_inspector_reuses_pose_dropdown_labels(self) -> None:
        inspector = _block_after(EDITOR, "public override void OnInspectorGUI()")
        self.assertEqual(inspector.count("fitter.TotalPoseCount"), 1)
        self.assertEqual(
            inspector.count("fitter.SystematicHumanoidTargetCount"), 1)
        labels = _block_after(EDITOR, "string[] GetPoseLabels(")
        self.assertIn("fitter.GetGeneratedPoseNames(_poseNameScratch);", labels)
        self.assertNotIn("fitter.TotalPoseCount", labels)
        self.assertIn("_poseLabelsDirty = false;", labels)
        symmetry = _block_after(EDITOR, "static void DrawSymmetryPipeline(")
        self.assertIn("statusCache.morphDataVersion != morphDataVersion", symmetry)
        self.assertIn("if (refreshStatus)", symmetry)


if __name__ == "__main__":
    unittest.main()
