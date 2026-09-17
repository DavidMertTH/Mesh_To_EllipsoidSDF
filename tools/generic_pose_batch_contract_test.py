"""Static contracts for generic-skeleton synthetic pose batches.

These tests deliberately complement, rather than modify, the Humanoid pose
library tests.  They protect the generic data model and the lifecycle rules
that are easy to regress without requiring Unity to be available in CI.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = ROOT / "clothSimulation" / "EllipsoidLoading"
FITTER_PATH = UNITY_ROOT / "EllipSDFSyntheticPoseBatchFitter.cs"
LIBRARY_PATH = UNITY_ROOT / "EllipSDFGenericPoseLibrary.cs"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


FITTER = _read(FITTER_PATH)
LIBRARY = _read(LIBRARY_PATH)


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
                return source[brace + 1:index]
    raise AssertionError(f"Unclosed C# block after {marker!r}")


def _assert_in_order(
    case: unittest.TestCase,
    source: str,
    *markers: str,
) -> None:
    cursor = -1
    for marker in markers:
        position = source.find(marker, cursor + 1)
        case.assertGreater(
            position,
            cursor,
            f"Expected {marker!r} after the preceding lifecycle step",
        )
        cursor = position


class GenericPoseLibraryContractTest(unittest.TestCase):
    def test_generic_pose_library_is_a_serializable_asset(self) -> None:
        self.assertTrue(
            LIBRARY_PATH.exists(),
            "The generic pose library must live beside the fitter.",
        )
        self.assertIn("ScriptableObject", LIBRARY)
        self.assertIn("CreateAssetMenu", LIBRARY)
        self.assertRegex(LIBRARY, r"\[Serializable\][\s\S]{0,120}class\s+Pose\b")
        self.assertRegex(LIBRARY, r"\[Serializable\][\s\S]{0,120}class\s+BonePose\b")
        self.assertRegex(
            LIBRARY,
            r"enum\s+TransformEncoding[\s\S]{0,180}AbsoluteLocal"
            r"[\s\S]{0,180}DeltaFromBase",
        )

    def test_library_exposes_a_fixed_stable_pose_catalog(self) -> None:
        for contract in (
            "IReadOnlyList<Pose>",
            "public int Count",
            "TryGetPose(int",
            "TryGetPose(string",
            "Validate(List<string>",
        ):
            self.assertIn(contract, LIBRARY)

        pose = _block_after(LIBRARY, "class Pose")
        self.assertRegex(pose, r"\bstring\s+name\b")
        self.assertRegex(pose, r"\bTransformEncoding\s+encoding\b")
        self.assertRegex(pose, r"\b(List|IReadOnlyList)<BonePose>\s+bones\b")

        validation = _block_after(LIBRARY, "Validate(List<string>")
        self.assertIn("StringComparer.Ordinal", validation)
        self.assertRegex(validation, r"HashSet\s*<\s*string\s*>")
        self.assertRegex(
            validation,
            r"IsNullOrWhiteSpace|IsNullOrEmpty|\.Trim\s*\(",
        )

    def test_bones_resolve_by_index_then_relative_path_not_ambiguous_name(self) -> None:
        bone_pose = _block_after(LIBRARY, "class BonePose")
        for field in (
            "relativePath",
            "rendererBoneIndex",
            "localPosition",
            "localRotation",
            "localScale",
        ):
            self.assertIn(field, bone_pose)
        self.assertRegex(bone_pose, r"bool\s+apply(Position|LocalPosition)")
        self.assertRegex(bone_pose, r"bool\s+apply(Rotation|LocalRotation)")
        self.assertRegex(bone_pose, r"bool\s+apply(Scale|LocalScale)")

        resolve = _block_after(LIBRARY, "TryResolve(")
        _assert_in_order(
            self,
            resolve,
            "rendererBoneIndex",
            "rendererBones",
            "relativePath",
        )
        self.assertRegex(resolve, r"\.Find\s*\(\s*relativePath\s*\)")
        self.assertNotRegex(
            resolve,
            r"rendererBones\s*\[[^]]+\]\.name\s*==|\.name\s*==\s*relativePath",
        )


class GenericPoseBatchContractTest(unittest.TestCase):
    def test_rig_mode_separates_generic_from_humanoid_requirements(self) -> None:
        self.assertRegex(
            FITTER,
            r"enum\s+RigMode[\s\S]{0,200}\bAuto\b"
            r"[\s\S]{0,200}\bHumanoid\b[\s\S]{0,200}\bGeneric\b",
        )
        self.assertRegex(FITTER, r"public\s+RigMode\s+rigMode\b")
        self.assertRegex(
            FITTER,
            r"ResolvedRigMode|Resolve(?:Effective)?RigMode",
        )

        start = _block_after(FITTER, "public void StartBatch()")
        generic_guard = re.search(
            r"(ResolvedRigMode|resolvedRigMode|_resolvedRigMode|"
            r"effectiveRigMode|effectiveMode|_batchRigMode)\s*"
            r"(?:==|!=)\s*RigMode\.(?:Generic|Humanoid)",
            start,
        )
        self.assertIsNotNone(
            generic_guard,
            "Human Avatar/T-pose validation must be conditional on rig mode.",
        )
        self.assertIn("HasStoredTPose", start)
        self.assertLess(
            start.index(generic_guard.group(0)),
            start.index("HasStoredTPose"),
        )

    def test_fitter_freezes_one_flat_pose_plan_before_destructive_work(self) -> None:
        self.assertRegex(
            FITTER,
            r"enum\s+BatchPoseKind[\s\S]{0,320}\bHumanoidPreset\b"
            r"[\s\S]{0,320}\bGenericLibrary\b"
            r"[\s\S]{0,320}\bAnimationSample\b"
            r"[\s\S]{0,320}\bGenericRandom\b",
        )
        self.assertRegex(
            FITTER,
            r"(?:BatchPosePlanItem\s*\[\s*\]|"
            r"List\s*<\s*BatchPosePlanItem\s*>)\s+"
            r"_batchPosePlanSnapshot",
        )
        self.assertIn("_batchPosePlanSnapshotActive", FITTER)
        self.assertIn("TryPrepareBatchPosePlan(", FITTER)

        start = _block_after(FITTER, "public void StartBatch()")
        _assert_in_order(
            self,
            start,
            "TryPrepareBatchPosePlan(",
            "ClearMorphTargetsForNewBatch()",
        )

        self.assertRegex(
            FITTER,
            r"public\s+int\s+TotalPoseCount\s*=>\s*"
            r"CurrentPosePlanCount\s*\(\s*\)",
        )
        generated_name = _block_after(
            FITTER, "public string GetGeneratedPoseName("
        )
        self.assertIn("BuildPoseName(", generated_name)
        build_name = _block_after(FITTER, "string BuildPoseName(")
        self.assertIn("TryGetBatchPosePlanItem", build_name)
        apply_pose = _block_after(FITTER, "bool ApplyBatchPose(")
        self.assertIn("TryGetBatchPosePlanItem", apply_pose)
        symmetry = _block_after(
            FITTER, "void ResolveSavedTargetSymmetry("
        )
        self.assertIn("TryGetBatchPosePlanItem", symmetry)
        preview = _block_after(FITTER, "public bool PreviewGeneratedPose(")
        self.assertIn("BuildPoseName(", preview)
        self.assertIn("ApplyBatchPose(", preview)

    def test_pruned_plan_is_retained_after_runtime_snapshot_cleanup(
        self,
    ) -> None:
        # The finalized plan is also the durable post-batch lookup source.  In
        # particular, domain reload / cleanup must not make generated target
        # names drift back to mutable inspector configuration.
        for field_pattern in (
            r"\[\s*SerializeField\s*,\s*HideInInspector\s*\]\s*"
            r"List\s*<\s*BatchPosePlanItem\s*>\s+_retainedPosePlan\b",
            r"\[\s*SerializeField\s*,\s*HideInInspector\s*\]\s*"
            r"bool\s+_retainedPosePlanActive\b",
            r"\[\s*SerializeField\s*,\s*HideInInspector\s*\]\s*"
            r"RigMode\s+_retainedPosePlanRigMode\b",
        ):
            self.assertRegex(FITTER, field_pattern)

        retain = _block_after(FITTER, "void RetainCurrentBatchPosePlan(")
        _assert_in_order(
            self,
            retain,
            "_retainedPosePlan.Clear();",
            "_retainedPosePlan.AddRange(_batchPosePlanSnapshot);",
            "_retainedPosePlanActive = _retainedPosePlan.Count > 0;",
            "_retainedPosePlanRigMode = _batchRigMode;",
        )

        finalize = _block_after(
            FITTER, "bool TryFinalizeGenericPosePlan("
        )
        self.assertIn("RetainCurrentBatchPosePlan();", finalize)
        self.assertGreater(
            finalize.index("RetainCurrentBatchPosePlan();"),
            finalize.rfind("_batchPosePlanSnapshot.RemoveAt(i);"),
            "The retained plan must be copied only after neutral items are pruned.",
        )

        clear = _block_after(FITTER, "void ClearBatchPosePlanSnapshot()")
        self.assertIn("_batchPosePlanSnapshot.Clear();", clear)
        self.assertNotIn(
            "_retainedPosePlan",
            clear,
            "Runtime snapshot cleanup must preserve the finalized plan.",
        )
        cleanup = _block_after(FITTER, "void ReleaseBatchResources()")
        self.assertIn("ClearBatchPosePlanSnapshot();", cleanup)

        compatible = _block_after(FITTER, "bool CanUseRetainedPosePlan()")
        for contract in (
            "_retainedPosePlanActive",
            "_retainedPosePlan != null",
            "_retainedPosePlan.Count > 0",
            "_retainedPosePlanRigMode == ResolveEffectiveRigMode()",
        ):
            self.assertIn(contract, compatible)

        count = _block_after(FITTER, "int CurrentPosePlanCount()")
        _assert_in_order(
            self,
            count,
            "if (_batchPosePlanSnapshotActive)",
            "if (CanUseRetainedPosePlan())",
            "return _retainedPosePlan.Count;",
        )
        lookup = _block_after(FITTER, "bool TryGetBatchPosePlanItem(")
        self.assertRegex(
            lookup,
            r"useRetained\s*=\s*!_batchPosePlanSnapshotActive\s*&&\s*"
            r"CanUseRetainedPosePlan\s*\(\s*\)",
        )
        self.assertRegex(
            lookup,
            r"useRetained\s*\?\s*_retainedPosePlan\s*:\s*"
            r"_previewPosePlan",
        )

        build_name = _block_after(FITTER, "string BuildPoseName(")
        apply_pose = _block_after(FITTER, "bool ApplyBatchPose(")
        preview = _block_after(FITTER, "public bool PreviewGeneratedPose(")
        self.assertIn("TryGetBatchPosePlanItem", build_name)
        self.assertIn("TryGetBatchPosePlanItem", apply_pose)
        self.assertIn("BuildPoseName(", preview)
        self.assertIn("ApplyBatchPose(", preview)

    def test_retained_animation_items_are_the_primary_name_and_apply_source(
        self,
    ) -> None:
        apply_pose = _block_after(FITTER, "bool ApplyBatchPose(")
        self.assertRegex(
            apply_pose,
            r"BatchPoseKind\.AnimationSample[\s\S]{0,180}"
            r"ApplyAnimationFrame\s*\(\s*item\.sourceIndex\s*,\s*"
            r"item\.animationSample\s*\)",
        )

        apply_animation = _block_after(FITTER, "bool ApplyAnimationFrame(")
        self.assertRegex(
            apply_animation,
            r"if\s*\(\s*animationSample\.Clip\s*==\s*null\s*&&\s*"
            r"\(\s*!TryGetAnimationSample\s*\(",
        )
        _assert_in_order(
            self,
            apply_animation,
            "animationSample.Clip == null",
            "TryGetAnimationSample(",
            "AnimationClip sampledClip = animationSample.Clip;",
        )

        build_name = _block_after(FITTER, "string BuildPoseName(")
        _assert_in_order(
            self,
            build_name,
            "item.animationSample",
            "if (animationSample.Clip == null)",
            "TryGetAnimationSample(",
        )

    def test_generic_library_and_random_fallback_have_explicit_budgets(self) -> None:
        self.assertRegex(
            FITTER,
            r"public\s+EllipSDFGenericPoseLibrary\s+genericPoseLibrary\b",
        )
        self.assertRegex(FITTER, r"public\s+int\s+genericRandomPoseCount\b")
        self.assertRegex(FITTER, r"public\s+int\s+genericRandomSeed\b")
        self.assertRegex(
            FITTER,
            r"(?:Build|Apply|TryApply)GenericRandomPose\s*\(",
        )
        prepare = _block_after(FITTER, "bool TryPrepareBatchPosePlan(")
        self.assertIn("BuildPosePlan(_batchPosePlanSnapshot", prepare)
        self.assertIn("_batchPosePlanSnapshotActive = true;", prepare)
        build_plan = _block_after(FITTER, "bool BuildPosePlan(")
        self.assertIn("genericPoseLibrary", build_plan)
        self.assertIn("RigMode.Generic", build_plan)
        self.assertIn("genericRandomPoseCount", build_plan)
        self.assertRegex(
            build_plan,
            r"GenericPoseMode\.LibraryWithRandomFallback[\s\S]{0,300}"
            r"destination\.Count\s*==\s*0",
        )

    def test_random_fallback_is_local_deterministic_and_index_addressable(self) -> None:
        seed = _block_after(FITTER, "int BuildGenericRandomPoseSeed(")
        self.assertIn("genericRandomSeed", seed)
        self.assertRegex(seed, r"pose(Index|Key)|stable(Pose)?Key")
        self.assertRegex(seed, r"StableRandom|StableHash|HashInt|unchecked")
        random_pose = _block_after(FITTER, "bool ApplyGenericRandomPose(")
        self.assertIn("StableRandom", random_pose)
        self.assertNotIn("UnityEngine.Random", random_pose)
        self.assertNotIn("System.Random", random_pose)

    def test_random_rotations_are_bounded_local_deltas(self) -> None:
        start = FITTER.index("bool ApplyGenericRandomPose(")
        end = FITTER.index("bool ApplyAnimationFrame(", start)
        random_pose = FITTER[start:end]
        self.assertRegex(FITTER, r"genericRandomMax(imum)?(Bone)?Angle")
        self.assertIn("Mathf.Clamp", random_pose)
        self.assertIn("Quaternion.AngleAxis", random_pose)
        self.assertRegex(
            random_pose,
            r"(base|baseline)[A-Za-z]*Rotation\s*\*|"
            r"localRotation\s*=\s*[^;]*(base|baseline)",
        )
        self.assertRegex(
            random_pose,
            r"NormalizeQuaternion|\.normalized|Quaternion\.Normalize",
        )

    def test_generic_pose_failures_and_batch_cleanup_restore_hierarchy(self) -> None:
        apply_generic = _block_after(FITTER, "bool ApplyGenericLibraryPose(")
        self.assertIn("EnsureGenericPoseBaseline()", apply_generic)
        self.assertGreaterEqual(
            apply_generic.count("RestoreTransformHierarchy("), 2
        )
        apply_random = _block_after(FITTER, "bool ApplyGenericRandomPose(")
        self.assertIn("EnsureGenericPoseBaseline()", apply_random)
        self.assertGreaterEqual(
            apply_random.count("RestoreTransformHierarchy("), 2
        )

        restore = _block_after(FITTER, "void RestoreBasePose()")
        self.assertIn("RestoreTransformHierarchy(_restoreTransformStates);", restore)
        cleanup = _block_after(FITTER, "void ReleaseBatchResources()")
        self.assertIn("RestoreBasePose();", cleanup)

    def test_generic_targets_are_independent_of_humanoid_symmetry(self) -> None:
        symmetry = _block_after(FITTER, "void ResolveSavedTargetSymmetry(")
        self.assertRegex(
            symmetry,
            r"!UsesHumanoidRig|"
            r"(?:EffectiveRigMode|_batchRigMode|effectiveRigMode)\s*==\s*"
            r"RigMode\.Generic",
        )
        generic_marker = (
            "!UsesHumanoidRig"
            if "!UsesHumanoidRig" in symmetry
            else "RigMode.Generic"
        )
        generic_branch = symmetry[symmetry.index(generic_marker):]
        self.assertIn("EllipSDFMorphTargetSymmetry.Independent", generic_branch)
        self.assertRegex(generic_branch, r"mirroredName\s*=\s*\"\"")

    def test_library_pose_is_deep_snapshotted_into_the_flat_plan(self) -> None:
        plan_item = _block_after(FITTER, "struct BatchPosePlanItem")
        self.assertIn(
            "EllipSDFGenericPoseLibrary.Pose genericPoseSnapshot",
            plan_item,
        )

        snapshot = _block_after(
            LIBRARY, "public bool TryCreatePoseSnapshot("
        )
        for deep_copy_step in (
            "Pose copy = new Pose();",
            "new List<BonePose>(source.bones.Count)",
            "BonePose boneCopy = new BonePose();",
            "boneCopy.relativePath = sourceBone.relativePath;",
            "boneCopy.rendererBoneIndex = sourceBone.rendererBoneIndex;",
            "boneCopy.applyPosition = sourceBone.applyPosition;",
            "boneCopy.localPosition = sourceBone.localPosition;",
            "boneCopy.applyRotation = sourceBone.applyRotation;",
            "boneCopy.localRotation = sourceBone.localRotation;",
            "boneCopy.applyScale = sourceBone.applyScale;",
            "boneCopy.localScale = sourceBone.localScale;",
            "copy.bones.Add(boneCopy);",
            "snapshot = copy;",
        ):
            self.assertIn(deep_copy_step, snapshot)
        self.assertNotIn("snapshot = source;", snapshot)

        build = _block_after(FITTER, "bool BuildPosePlan(")
        _assert_in_order(
            self,
            build,
            "TryCreatePoseSnapshot(",
            "item.genericPoseSnapshot = poseSnapshot;",
            "destination.Add(item);",
        )
        apply = _block_after(FITTER, "bool ApplyBatchPose(")
        self.assertRegex(
            apply,
            r"ApplyGenericLibraryPose\s*\(\s*"
            r"item\.genericPoseSnapshot\s*\)",
        )
        apply_library = _block_after(
            FITTER, "bool ApplyGenericLibraryPose("
        )
        self.assertIn("EllipSDFGenericPoseLibrary.Pose pose", FITTER)
        self.assertNotIn("genericPoseLibrary", apply_library)

    def test_random_configuration_is_snapshotted_and_used_during_apply(
        self,
    ) -> None:
        plan_item = _block_after(FITTER, "struct BatchPosePlanItem")
        self.assertIn(
            "GenericRandomSettingsSnapshot randomSettings", plan_item
        )
        settings = _block_after(
            FITTER, "struct GenericRandomSettingsSnapshot"
        )
        for field in (
            "maximumAngleDegrees",
            "boneParticipation",
            "includeLeafBones",
            "hasConfiguredRules",
            "GenericRandomBoneRuleSnapshot[] rules",
        ):
            self.assertIn(field, settings)

        capture = _block_after(
            FITTER,
            "GenericRandomSettingsSnapshot CaptureGenericRandomSettings(",
        )
        for setting in (
            "genericRandomMaxAngleDegrees",
            "genericRandomBoneParticipation",
            "genericRandomIncludeLeafBones",
            "genericRandomBoneRules",
            "new GenericRandomBoneRuleSnapshot()",
            "rules.ToArray()",
        ):
            self.assertIn(setting, capture)

        build = _block_after(FITTER, "bool BuildPosePlan(")
        _assert_in_order(
            self,
            build,
            "CaptureGenericRandomSettings()",
            "item.randomSettings = randomSettings;",
            "destination.Add(item);",
        )
        apply = _block_after(FITTER, "bool ApplyBatchPose(")
        self.assertRegex(
            apply,
            r"ApplyGenericRandomPose\s*\([\s\S]{0,180}"
            r"item\.randomSettings\s*\)",
        )

        random_start = FITTER.index("bool ApplyGenericRandomPose(")
        random_end = FITTER.index("static bool HasParentBone(", random_start)
        random_pipeline = FITTER[random_start:random_end]
        self.assertIn("GenericRandomSettingsSnapshot settings", random_pipeline)
        for live_setting in (
            "genericRandomMaxAngleDegrees",
            "genericRandomBoneParticipation",
            "genericRandomIncludeLeafBones",
            "genericRandomBoneRules",
        ):
            self.assertNotIn(
                live_setting,
                random_pipeline,
                "A running/frozen pose must not read mutable inspector state.",
            )

    def test_generic_descriptors_only_accept_renderer_bones_and_rotation(
        self,
    ) -> None:
        library_validation = _block_after(
            FITTER, "bool TryValidateGenericPoseBindings("
        )
        for requirement in (
            "HashSet<Transform> rendererBoneSet",
            "rendererBoneSet.Add(bones[i])",
            "!rendererBoneSet.Contains(bone)",
            "hasRotationChannel |= bonePose.ApplyRotation;",
            "if (!hasRotationChannel)",
            "Translation- or ",
            "pose descriptor",
        ):
            self.assertIn(requirement, library_validation)

        library_apply = _block_after(
            FITTER, "bool ApplyGenericLibraryPose("
        )
        self.assertIn("!IsRendererBone(bone, rendererBones)", library_apply)
        rotation_branch = _block_after(
            library_apply, "if (bonePose.ApplyRotation)"
        )
        self.assertIn("descriptorChanges++;", rotation_branch)
        self.assertIn("if (descriptorChanges == 0)", library_apply)

        rule_filter = _block_after(
            FITTER,
            "List<GenericRandomBoneRuleSnapshot> "
            "CollectUsableGenericRandomRules(",
        )
        self.assertIn("!IsRendererBone(rule.bone, rendererBones)", rule_filter)

        clip_apply = _block_after(
            FITTER, "bool TryApplyGenericAnimationSample("
        )
        self.assertIn(
            "!AffectsGenericRendererBones(live, rendererBones)", clip_apply
        )
        self.assertIn("if (rotationChanged)", clip_apply)
        self.assertIn("descriptorChanges++;", clip_apply)
        self.assertIn("if (descriptorChanges == 0)", clip_apply)
        self.assertIn("did not rotate any bones", clip_apply)

        renderer_filter = _block_after(
            FITTER, "static bool AffectsGenericRendererBones("
        )
        self.assertIn("if (bone == candidate)", renderer_filter)
        self.assertNotIn("IsChildOf", renderer_filter)
        self.assertNotIn("candidate.parent", renderer_filter)

    def test_library_finalizer_validates_full_trs_for_every_bone_entry(
        self,
    ) -> None:
        evaluate = _block_after(
            FITTER, "bool TryEvaluateGenericLibraryPoseDescriptor("
        )
        self.assertRegex(
            evaluate,
            r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*"
            r"i\s*<\s*pose\.BoneCount\s*;\s*i\+\+\s*\)",
        )
        self.assertNotRegex(
            evaluate,
            r"!bonePose\.ApplyRotation[\s\S]{0,80}\bcontinue\s*;",
            "Position/scale entries still need binding and TRS validation.",
        )
        for contract in (
            "bonePose.TryResolve(root, rendererBones, out bone)",
            "!IsRendererBone(bone, rendererBones)",
            "_genericBaseStateByTransform.TryGetValue(",
            "bonePose.EvaluateLocalTransform(",
            "out localPosition",
            "out localRotation",
            "out localScale",
            "!IsFinite(localPosition)",
            "!IsFinite(localRotation)",
            "!IsFinite(localScale)",
            "Mathf.Abs(localScale.x) < 1e-5f",
            "Mathf.Abs(localScale.y) < 1e-5f",
            "Mathf.Abs(localScale.z) < 1e-5f",
        ):
            self.assertIn(contract, evaluate)
        self.assertRegex(
            evaluate,
            r"bonePose\.ApplyRotation\s*&&[\s\S]{0,180}"
            r"Quaternion\.Angle\s*\(",
            "Only a valid rotation delta counts as a morph descriptor.",
        )

    def test_generic_renderer_must_exactly_match_connector_even_if_null(
        self,
    ) -> None:
        validate = _block_after(FITTER, "bool TryValidateGenericRig(")
        connector = _block_after(validate, "if (connector != null)")
        _assert_in_order(
            self,
            connector,
            "connector.DebugResolvedSkinnedMesh",
            "if (connectorRenderer != genericSkinnedMesh)",
            "return false;",
        )
        self.assertNotRegex(
            connector,
            r"connectorRenderer\s*!=\s*null\s*&&[\s\S]{0,100}"
            r"connectorRenderer\s*!=\s*genericSkinnedMesh",
            "A null connector renderer is a mismatch, not permission to "
            "accept another SMR.",
        )

    def test_generic_preview_captures_reference_before_trained_early_return(
        self,
    ) -> None:
        baseline = _block_after(
            FITTER, "bool PrepareAnimationSamplingBaselineForPreview("
        )
        self.assertIn(
            "RestoreTransformHierarchy(_animationSamplingBaseline);",
            baseline,
        )
        _assert_in_order(
            self,
            baseline,
            "CaptureBasePose();",
            "ApplyConfiguredBasePose()",
            "CaptureAnimationSamplingBaseline();",
        )

        preview = _block_after(FITTER, "public bool PreviewGeneratedPose(")
        self.assertLess(
            preview.index("PrepareAnimationSamplingBaselineForPreview()"),
            preview.index("PreviewMorphTargetWithPose(targetIndex)"),
        )
        trained = _block_after(
            preview,
            "if (targetIndex >= 0 && driver.PreviewMorphTargetWithPose",
        )
        self.assertIn("_genericPreviewPoseActive = true;", trained)
        self.assertLess(
            trained.index("_genericPreviewPoseActive = true;"),
            trained.index("return true;"),
        )
        untrained_tail = preview[preview.index("if (!ApplyBatchPose("):]
        self.assertIn("_genericPreviewPoseActive = true;", untrained_tail)

    def test_batch_start_restores_generic_preview_before_reference_capture(
        self,
    ) -> None:
        restore = _block_after(
            FITTER, "void RestoreGenericPreviewPoseIfNeeded("
        )
        self.assertIn("if (!_genericPreviewPoseActive)", restore)
        self.assertIn(
            "RestoreTransformHierarchy(_animationSamplingBaseline);",
            restore,
        )
        _assert_in_order(
            self,
            restore,
            "RestoreTransformHierarchy(_animationSamplingBaseline);",
            "_genericPreviewPoseActive = false;",
            "ReleaseAnimatorPreviewHold();",
        )

        start = _block_after(FITTER, "public void StartBatch()")
        _assert_in_order(
            self,
            start,
            "RestoreGenericPreviewPoseIfNeeded();",
            "CaptureBasePose();",
            "ClearMorphTargetsForNewBatch()",
        )

    def test_disable_and_destroy_restore_generic_preview_before_dispose(
        self,
    ) -> None:
        for lifecycle_method in ("void OnDisable()", "void OnDestroy()"):
            lifecycle = _block_after(FITTER, lifecycle_method)
            self.assertIn("RestoreGenericPreviewPoseIfNeeded();", lifecycle)
            self.assertLess(
                lifecycle.index("RestoreGenericPreviewPoseIfNeeded();"),
                lifecycle.index("DisposeAnimationSamplingRig();"),
                lifecycle_method,
            )
            self.assertLess(
                lifecycle.index("RestoreGenericPreviewPoseIfNeeded();"),
                lifecycle.index("DisposeHumanPoseHandler();"),
                lifecycle_method,
            )

    def test_generic_plan_finalization_checks_each_frozen_descriptor(
        self,
    ) -> None:
        finalize = _block_after(
            FITTER, "bool TryFinalizeGenericPosePlan("
        )
        self.assertRegex(
            finalize,
            r"for\s*\(\s*int\s+i\s*=\s*"
            r"_batchPosePlanSnapshot\.Count\s*-\s*1\s*;\s*"
            r"i\s*>=\s*0\s*;\s*i--\s*\)",
        )

        library = _block_after(
            finalize,
            "if (item.kind == BatchPoseKind.GenericLibrary)",
        )
        self.assertIn("TryEvaluateGenericLibraryPoseDescriptor(", library)
        self.assertIn("item.genericPoseSnapshot", library)
        self.assertIn("if (!hasDescriptorDelta)", library)
        self.assertIn("_batchPosePlanSnapshot.RemoveAt(i);", library)

        self.assertIn(
            "if (item.kind != BatchPoseKind.AnimationSample)", finalize
        )
        self.assertIn("TryApplyGenericAnimationSample(", finalize)
        self.assertIn("item.animationSample", finalize)
        self.assertIn("out noDescriptorChange", finalize)
        self.assertGreaterEqual(
            finalize.count("_batchPosePlanSnapshot.RemoveAt(i);"), 2
        )

        # Finalization must validate the exact frozen sample descriptors. A
        # fixed-time probe can approve one frame while a different planned
        # frame remains neutral or invalid.
        for forbidden_probe in (
            "TryGetAnimationSample(",
            "GetAnimationSampleNormalizedTime(",
            "new EllipSDFAnimationPoseSampler.PoseSample(",
            "0.5f",
            "fixedTime",
            "probeTime",
        ):
            self.assertNotIn(forbidden_probe, finalize)

        start = _block_after(FITTER, "public void StartBatch()")
        _assert_in_order(
            self,
            start,
            "CaptureAnimationSamplingBaseline();",
            "TryFinalizeGenericPosePlan(",
            "ClearMorphTargetsForNewBatch()",
        )

    def test_finalization_adds_snapshotted_random_fallback_after_pruning(
        self,
    ) -> None:
        finalize = _block_after(
            FITTER, "bool TryFinalizeGenericPosePlan("
        )
        for contract in (
            "hasNonRandomTarget",
            "hasRandomTarget",
            "BatchPoseKind.GenericLibrary",
            "BatchPoseKind.AnimationSample",
            "BatchPoseKind.GenericRandom",
            "GenericPoseMode.LibraryWithRandomFallback",
            "CaptureGenericRandomSettings()",
            "CanGenerateGenericRandomPoses(settings, out error)",
            "Mathf.Max(1, genericRandomPoseCount)",
            "randomItem.stableSeed = BuildGenericRandomPoseSeed(i);",
            "randomItem.randomSettings = settings;",
            "_batchPosePlanSnapshot.Add(randomItem);",
        ):
            self.assertIn(contract, finalize)
        fallback_start = finalize.index("if (!hasNonRandomTarget")
        fallback = finalize[fallback_start:]
        _assert_in_order(
            self,
            fallback,
            "GenericPoseMode.LibraryWithRandomFallback",
            "CaptureGenericRandomSettings()",
            "CanGenerateGenericRandomPoses(settings, out error)",
            "randomItem.kind = BatchPoseKind.GenericRandom;",
            "randomItem.randomSettings = settings;",
            "_batchPosePlanSnapshot.Add(randomItem);",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
