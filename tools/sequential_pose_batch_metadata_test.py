"""Static contracts for sequential Unity synthetic-pose batch metadata."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = ROOT / "clothSimulation" / "EllipsoidLoading"
FITTER = (UNITY_ROOT / "EllipSDFSyntheticPoseBatchFitter.cs").read_text(
    encoding="utf-8"
)
CONNECTOR = (UNITY_ROOT / "EllipSDFConnector.cs").read_text(encoding="utf-8")


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


class SequentialPoseBatchMetadataTest(unittest.TestCase):
    def test_one_identity_is_created_and_cleared_per_sequential_run(self) -> None:
        start = _block_after(FITTER, "public void StartBatch()")
        release = _block_after(FITTER, "void ReleaseBatchResources()")
        begin_metadata = _block_after(
            FITTER, "void BeginSequentialPoseBatchMetadata()"
        )
        self.assertIn("BeginSequentialPoseBatchMetadata();", start)
        self.assertIn('Guid.NewGuid().ToString("N")', begin_metadata)
        self.assertIn("ClearSequentialPoseBatchMetadata();", release)

    def test_edit_and_play_paths_send_the_real_zero_based_pose_index(self) -> None:
        editor_update = _block_after(FITTER, "void UpdateEditorBatch()")
        play_pose = _block_after(FITTER, "IEnumerator FitAndSavePose(int poseIndex)")
        for block, index_name in (
            (editor_update, "currentPoseIndex"),
            (play_pose, "poseIndex"),
        ):
            self.assertRegex(
                block,
                re.compile(
                    r"BeginFitCurrentEllipsoidsToPoseOperation\s*\(\s*"
                    r"currentPoseName\s*,\s*followingRunIterations\s*,\s*"
                    r"_sequentialPoseBatchId\s*,\s*"
                    + index_name
                    + r"\s*\)",
                    re.MULTILINE,
                ),
            )

        self.assertIn("BeginFitPoseOperation(\n                    firstRunIterations)",
                      editor_update)
        base_fit = _block_after(FITTER, "IEnumerator FitAndSaveBase()")
        self.assertIn("BeginFitPoseOperation(firstRunIterations)", base_fit)

    def test_old_buffered_pipeline_is_not_started(self) -> None:
        self.assertEqual(FITTER.count("StartBufferedPosePipeline("), 1)

    def test_single_fit_overloads_explicitly_omit_batch_metadata(self) -> None:
        self.assertRegex(
            CONNECTOR,
            re.compile(
                r"BeginFitPoseOperation\(int iterations\)\s*\{\s*"
                r"return BeginFitPoseOperation\(iterations, null, -1\);",
                re.MULTILINE,
            ),
        )
        self.assertRegex(
            CONNECTOR,
            re.compile(
                r"BeginFitCurrentEllipsoidsToPoseOperation\(\s*"
                r"string targetName, int iterations\)\s*\{\s*"
                r"return BeginFitCurrentEllipsoidsToPoseOperation\(\s*"
                r"targetName, iterations, null, -1\);",
                re.MULTILINE,
            ),
        )

    def test_json_requires_both_batch_id_and_nonnegative_pose_index(self) -> None:
        build = _block_after(CONNECTOR, "string BuildRequestJsonWithIterations(")
        self.assertRegex(
            build,
            re.compile(
                r"bool batchPipeline\s*=\s*"
                r"!string\.IsNullOrEmpty\(batchPipelineId\)\s*&&\s*"
                r"batchPoseIndex\s*>=\s*0\s*;",
                re.MULTILINE,
            ),
        )
        self.assertIn('\\"batch_pipeline\\":true', build)
        self.assertIn('\\"batch_id\\":\\"', build)
        self.assertIn('\\"pose_index\\":', build)


if __name__ == "__main__":
    unittest.main()
