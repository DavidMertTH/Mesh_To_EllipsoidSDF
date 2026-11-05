"""GPU skinning must respect the supplied number of influences per vertex."""

from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinning import _deform_numpy, _deform_warp


class SkinningInfluenceCountTest(unittest.TestCase):
    def test_gpu_matches_cpu_with_one_to_four_influences(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA device unavailable")
        vertices = np.array([[.2, .3, .4], [-.4, .8, -.2],
                             [.1, -.7, .5], [.9, .2, -.1]], np.float32)
        matrices = np.tile(np.eye(4), (3, 1, 1))
        matrices[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        matrices[2, :3, 3] = [.3, -.2, .1]
        rng = np.random.default_rng(12)
        for count in range(1, 5):
            with self.subTest(influences=count):
                joints = rng.integers(0, 3, (len(vertices), count), dtype=np.int32)
                weights = rng.random((len(vertices), count)).astype(np.float32)
                weights /= weights.sum(axis=1, keepdims=True)
                expected = _deform_numpy(vertices, joints, weights, matrices)
                # Call the kernel path directly: a CPU fallback must not mask
                # an out-of-bounds read or a poisoned CUDA context.
                actual = _deform_warp(vertices, joints, weights, matrices)
                np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
