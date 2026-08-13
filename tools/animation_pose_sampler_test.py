"""Static regression tests for the Unity multi-animation pose sampler."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = ROOT / "clothSimulation" / "EllipsoidLoading"
SAMPLER_PATH = UNITY_ROOT / "EllipSDFAnimationPoseSampler.cs"
SAMPLER_EDITOR_PATH = (
    UNITY_ROOT / "Editor" / "EllipSDFAnimationPoseSamplerEditor.cs"
)
FITTER_PATH = UNITY_ROOT / "EllipSDFSyntheticPoseBatchFitter.cs"

SAMPLER = SAMPLER_PATH.read_text(encoding="utf-8")
SAMPLER_EDITOR = SAMPLER_EDITOR_PATH.read_text(encoding="utf-8")
FITTER = FITTER_PATH.read_text(encoding="utf-8")


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


class AnimationPoseSamplerTest(unittest.TestCase):
    def test_sampler_is_a_first_class_unity_component(self) -> None:
        self.assertIn("[ExecuteAlways]", SAMPLER)
        self.assertIn("[DisallowMultipleComponent]", SAMPLER)
        self.assertIn(
            "public sealed class EllipSDFAnimationPoseSampler : MonoBehaviour",
            SAMPLER,
        )
        self.assertIn("public List<AnimationSource> animations", SAMPLER)
        self.assertIn("public int totalSampleCount", SAMPLER)
        self.assertIn("public int randomSeed", SAMPLER)

    def test_plan_is_local_deterministic_and_snapshot_capable(self) -> None:
        self.assertIn("struct StableRandom", SAMPLER)
        self.assertNotIn("UnityEngine.Random", SAMPLER)
        self.assertNotIn("System.Random", SAMPLER)
        self.assertIn("public bool TryCreateSnapshot(", SAMPLER)
        snapshot = _block_after(SAMPLER, "public bool TryCreateSnapshot(")
        self.assertIn("samples = _samplePlan.ToArray();", snapshot)

    def test_distribution_is_exactly_balanced(self) -> None:
        build = _block_after(SAMPLER, "void BuildBalancedSamplePlan(")
        self.assertIn(
            "int baseCount = totalSampleCount / sourceIndices.Count;",
            build,
        )
        self.assertIn(
            "int remainder = totalSampleCount % sourceIndices.Count;",
            build,
        )
        self.assertRegex(
            build,
            re.compile(r"baseCount\s*\+\s*\(i < remainder \? 1 : 0\)"),
        )
        self.assertIn("for (int round = 0; round < maximumCount; round++)", build)

    def test_each_clip_uses_stratified_random_time(self) -> None:
        build = _block_after(SAMPLER, "void BuildBalancedSamplePlan(")
        self.assertIn("(float)slot / count", build)
        self.assertIn("(float)(slot + 1) / count", build)
        self.assertIn(
            "stratumStart, stratumEnd, random.NextFloat01()",
            build,
        )
        self.assertIn(
            "normalizedTime, stratumStart, stratumEnd",
            build,
        )
        self.assertIn("half-open", build)

    def test_invalid_or_ambiguous_sources_fail_explicitly(self) -> None:
        validation = _block_after(SAMPLER, "bool TryCollectSources(")
        for requirement in (
            "has no clip",
            "not imported as a Humanoid animation",
            "has no usable duration",
            "has a non-finite normalized time range",
            "needs a non-zero normalized time range",
            "is listed more than once",
            "is not unique",
            "must be at least the enabled source count",
        ):
            self.assertIn(requirement, validation)
        self.assertIn("MaximumSampleCount", validation)

    def test_sample_budget_is_hard_bounded(self) -> None:
        self.assertIn("public const int MaximumSampleCount = 4096;", SAMPLER)
        self.assertIn(
            "[Range(1, MaximumSampleCount)]",
            SAMPLER,
        )
        on_validate = _block_after(SAMPLER, "void OnValidate()")
        self.assertIn(
            "totalSampleCount, 1, MaximumSampleCount",
            on_validate,
        )

    def test_fitter_freezes_plan_before_deleting_old_targets(self) -> None:
        start_batch = _block_after(FITTER, "public void StartBatch()")
        freeze = start_batch.index("TryPrepareAnimationSampleSnapshot(")
        destructive = start_batch.index("ClearMorphTargetsForNewBatch()")
        self.assertLess(freeze, destructive)
        prepare = _block_after(
            FITTER, "bool TryPrepareAnimationSampleSnapshot(")
        self.assertIn("animationPoseSampler.TryCreateSnapshot(", prepare)
        self.assertIn("_animationSampleSnapshotActive = true;", prepare)

    def test_count_name_apply_and_preview_share_the_same_descriptor(self) -> None:
        count = _block_after(FITTER, "public int AnimationTargetCount")
        self.assertIn("_animationSampleSnapshot.Length", count)
        apply_frame = _block_after(FITTER, "bool ApplyAnimationFrame(")
        build_name = _block_after(FITTER, "string BuildPoseName(")
        self.assertIn("TryGetAnimationSample(", apply_frame)
        self.assertIn("TryGetAnimationSample(", build_name)
        self.assertIn("animationSample.StablePoseKey", apply_frame)
        self.assertIn("BuildAnimationTargetIdentityKey(", build_name)

    def test_animation_sampling_uses_an_isolated_humanoid_rig(self) -> None:
        apply_frame = _block_after(FITTER, "bool ApplyAnimationFrame(")
        sample = _block_after(FITTER, "bool TrySampleAnimationHumanPose(")
        ensure_rig = _block_after(FITTER, "bool EnsureAnimationSamplingRig(")
        self.assertNotIn("AnimationPlayableOutput.Create(", apply_frame)
        self.assertIn("CloneTransformHierarchy(", ensure_rig)
        self.assertIn("new HumanPoseHandler(", ensure_rig)
        self.assertIn("AnimationPlayableOutput.Create(", sample)
        self.assertIn("_animationSamplingAnimator", sample)
        self.assertNotIn(
            '"EllipSDF Isolated Animation Pose",\n                animator)',
            sample,
        )
        self.assertIn("graph.Destroy();", sample)
        self.assertIn(
            "RestoreTransformHierarchy(_animationSamplingRigBaseline);",
            sample,
        )
        self.assertIn(
            "CaptureTransformHierarchy(",
            apply_frame,
        )
        self.assertIn(
            "RestoreTransformHierarchy(previousTransformStates);",
            apply_frame,
        )
        self.assertIn(
            "RestoreTransformHierarchy(_animationSamplingBaseline);",
            apply_frame,
        )
        self.assertIn("DisposeAnimationSamplingRig();", apply_frame)

    def test_base_sampling_baseline_is_captured_after_settling(self) -> None:
        run_batch = _block_after(FITTER, "IEnumerator RunBatch()")
        play_settle = run_batch.index(
            "for (int f = 0; f < settleFrames; f++)"
        )
        play_capture = run_batch.index(
            "CaptureAnimationSamplingBaseline();"
        )
        self.assertLess(play_settle, play_capture)

        editor_update = _block_after(FITTER, "void UpdateEditorBatch()")
        settle_stage = editor_update.index(
            "case EditorBatchStage.SettleBase:"
        )
        editor_capture = editor_update.index(
            "CaptureAnimationSamplingBaseline();",
            settle_stage,
        )
        fit_stage = editor_update.index(
            "_editorStage = EditorBatchStage.FitBaseStart;",
            settle_stage,
        )
        self.assertLess(settle_stage, editor_capture)
        self.assertLess(editor_capture, fit_stage)

    def test_seed_cannot_change_while_a_linked_batch_runs(self) -> None:
        advance = _block_after(SAMPLER, "public void AdvanceRandomSeed()")
        self.assertIn("IsUsedByRunningBatch()", advance)
        self.assertIn("return;", advance)
        self.assertNotIn(
            '[ContextMenu("Advance Animation Pose Random Seed")]',
            SAMPLER,
        )
        guard = _block_after(SAMPLER, "bool IsUsedByRunningBatch()")
        self.assertIn("fitter.isRunning", guard)
        self.assertIn("fitter.animationPoseSampler == this", guard)

    def test_untrained_preview_reuses_the_configured_base_baseline(self) -> None:
        preview = _block_after(FITTER, "public bool PreviewGeneratedPose(")
        self.assertIn(
            "PrepareAnimationSamplingBaselineForPreview()",
            preview,
        )
        self.assertIn("HoldAnimatorForPreview();", preview)
        base_preview = _block_after(
            FITTER, "public void PreviewBaseState()"
        )
        self.assertIn("ReleaseAnimatorPreviewHold();", base_preview)

    def test_batch_resources_are_released_when_the_component_disables(self) -> None:
        on_disable = _block_after(FITTER, "void OnDisable()")
        self.assertIn("StopCoroutine(runner);", on_disable)
        self.assertIn("ReleaseBatchResources();", on_disable)
        cleanup = _block_after(FITTER, "void ReleaseBatchResources()")
        for cleanup_step in (
            "RestoreBasePose();",
            "DisposeAnimationSamplingRig();",
            "animator.enabled = animatorEnabled;",
            "DisposeHumanPoseHandler();",
            "isRunning = false;",
            "connector.ClearSyntheticBatchProgressSource(this);",
            "ClearAnimationSampleSnapshot();",
        ):
            self.assertIn(cleanup_step, cleanup)

    def test_human_pose_handlers_and_preview_animator_have_clear_ownership(
        self,
    ) -> None:
        capture_base = _block_after(FITTER, "void CaptureBasePose()")
        self.assertLess(
            capture_base.index("DisposeHumanPoseHandler();"),
            capture_base.index(
                "_humanPoseHandler = new HumanPoseHandler("
            ),
        )
        capture_t_pose = _block_after(
            FITTER, "public bool CaptureCurrentPoseAsTPose()"
        )
        self.assertIn("DisposeHumanPoseHandler();", capture_t_pose)
        hold = _block_after(FITTER, "void HoldAnimatorForPreview()")
        release = _block_after(
            FITTER, "void ReleaseAnimatorPreviewHold()"
        )
        self.assertIn("_previewHeldAnimator = animator;", hold)
        self.assertIn(
            "heldAnimator.enabled = _previewAnimatorEnabled;",
            release,
        )

    def test_animation_sources_have_explicit_symmetry_policy(self) -> None:
        self.assertIn("public enum AnimationSymmetry", SAMPLER)
        self.assertIn(
            "AnimationSymmetry.Independent",
            SAMPLER,
        )
        symmetry = _block_after(
            FITTER, "bool ConfigureSavedTargetSymmetry(")
        self.assertIn(
            "!animationSample.GenerateMirroredSupport",
            symmetry,
        )
        self.assertIn(
            "EllipSDFMorphTargetSymmetry.Independent",
            symmetry,
        )

    def test_sampler_is_explicit_and_snapshot_beats_live_flags(self) -> None:
        resolve = _block_after(FITTER, "void ResolveReferences()")
        self.assertNotIn("EllipSDFAnimationPoseSampler", resolve)
        count = _block_after(FITTER, "public int AnimationTargetCount")
        self.assertLess(
            count.index("_animationSampleSnapshotActive"),
            count.index("!fitAnimationFrames"),
        )
        get_sample = _block_after(FITTER, "bool TryGetAnimationSample(")
        self.assertLess(
            get_sample.index("_animationSampleSnapshotActive"),
            get_sample.index("!fitAnimationFrames"),
        )

    def test_bound_sampler_inspector_locks_during_batch(self) -> None:
        self.assertIn("FindRunningFitter(sampler)", SAMPLER_EDITOR)
        self.assertIn(
            "new EditorGUI.DisabledScope(configurationLocked)",
            SAMPLER_EDITOR,
        )
        self.assertIn("is locked while", SAMPLER_EDITOR)

    def test_legacy_single_clip_configuration_remains_supported(self) -> None:
        for legacy_field in (
            "public AnimationClip animationClip;",
            "public int animationFrameCount = 16;",
            "public float animationStartNormalizedTime = 0f;",
            "public float animationEndNormalizedTime = 1f;",
        ):
            self.assertIn(legacy_field, FITTER)
        prepare = _block_after(
            FITTER, "bool TryPrepareAnimationSampleSnapshot(")
        self.assertIn("if (animationClip == null)", prepare)
        self.assertIn("GetAnimationSampleNormalizedTime(i)", prepare)

    def test_editor_can_assign_sampler_to_batch_fitter(self) -> None:
        self.assertIn(
            "[CustomEditor(typeof(EllipSDFAnimationPoseSampler))]",
            SAMPLER_EDITOR,
        )
        self.assertIn("Use In Pose Batch Fitter", SAMPLER_EDITOR)
        self.assertIn("fitter.animationPoseSampler = sampler;", SAMPLER_EDITOR)
        self.assertIn("fitter.fitAnimationFrames = true;", SAMPLER_EDITOR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
