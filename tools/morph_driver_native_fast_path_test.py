"""Static regression checks for Morph Driver native fast-path startup."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "clothSimulation"
    / "EllipsoidLoading"
    / "EllipSDFMorphDriver.cs"
)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")


def method_body(start_marker: str, end_marker: str) -> str:
    start = SOURCE.index(start_marker)
    end = SOURCE.index(end_marker, start)
    return SOURCE[start:end]


class MorphDriverNativeFastPathTests(unittest.TestCase):
    def test_optional_spatial_buffers_exist_before_first_morph_job(self) -> None:
        body = method_body(
            "bool EnsureBurstCache(",
            "void EnsureBurstWeightBuffers(",
        )
        self.assertIn(
            "EnsureNativeArray(ref _jobSpatialBoneIndices, 1);",
            body,
        )
        self.assertIn(
            "EnsureNativeArray(ref _jobSpatialBoneWeights, 1);",
            body,
        )

    def test_native_failure_is_retried_without_inspector_toggle(self) -> None:
        schedule = method_body(
            "bool TryScheduleCurrentPoseBurstFast(",
            "float RuntimeSmoothingAlpha(",
        )
        failure = method_body(
            "void HandleNativeFastPathFailure(",
            "void MarkNativeFastPathHealthy(",
        )
        self.assertIn(
            "Time.unscaledTime < _nextNativeFastPathRetryTime",
            schedule,
        )
        self.assertIn("_burstCacheDirty = true;", schedule)
        self.assertIn("_weightCacheDirty = true;", schedule)
        self.assertIn("_nativeFastPathFailureCount++;", failure)

    def test_hidden_scene_objects_do_not_enter_useless_managed_fallback(self) -> None:
        queue = method_body(
            "static void ProcessRuntimeDriverQueue(",
            "void OnGUI(",
        )
        self.assertGreaterEqual(
            len(re.findall(
                r"if \(driver\.showTransformedGameObjects\)\s+"
                r"FallbackRuntimeDrivers\.Add\(driver\);",
                queue,
            )),
            2,
        )


if __name__ == "__main__":
    unittest.main()
