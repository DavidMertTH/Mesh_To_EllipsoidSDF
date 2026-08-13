"""Focused regression tests for optimizer/API symmetry metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_window import MainWindow  # noqa: E402
from mesh_io import NormalizationTransform  # noqa: E402
from optimization import OptimizationWorker  # noqa: E402


def _worker() -> OptimizationWorker:
    return OptimizationWorker(
        sdf_target_np=np.zeros((3, 3, 3), dtype=np.float32),
        origin=np.zeros(3, dtype=np.float32),
        dx=0.1,
        n=3,
        num_ellipsoids=1,
        max_ellipsoids=8,
        num_steps=1,
        report_every=1,
        maintenance_every=0,
        local_fit=False,
    )


class SymmetryMetadataTest(unittest.TestCase):
    def test_worker_reports_hard_mirror_partition(self) -> None:
        worker = _worker()
        worker._sym_axis = 1
        worker._sym_plane = np.float32(0.25)
        worker._sym_n_op = 2
        worker._sym_n_so = 3

        self.assertEqual(
            worker.symmetry_metadata(),
            {
                "axis": 1,
                "plane": 0.25,
                "on_plane_count": 2,
                "pair_count": 3,
            },
        )

    def test_worker_omits_unresolved_or_empty_symmetry(self) -> None:
        worker = _worker()
        self.assertIsNone(worker.symmetry_metadata())

        worker._sym_axis = 0
        worker._sym_plane = 0.0
        worker._sym_n_op = 0
        worker._sym_n_so = 0
        self.assertIsNone(worker.symmetry_metadata())

    def test_api_payload_maps_partition_to_ids_and_original_plane(self) -> None:
        host = SimpleNamespace(
            _api_symmetry={
                "axis": 1,
                "plane": 0.5,
                "on_plane_count": 1,
                "pair_count": 2,
            },
            _api_norm=NormalizationTransform(
                center=np.array([10.0, 20.0, 30.0], dtype=np.float64),
                scale=2.0,
            ),
        )
        entries = [
            {"id": 7},
            {"id": 11},
            {"id": 23},
            {"id": 31},
            {"id": 47},
        ]

        payload = MainWindow._api_build_symmetry_payload(host, entries)

        self.assertEqual(
            payload,
            {
                "active": True,
                "axis": 1,
                "axis_name": "y",
                "plane": 20.25,
                "on_plane_ids": [7],
                "pairs": [
                    {"source_id": 11, "mirror_id": 31},
                    {"source_id": 23, "mirror_id": 47},
                ],
            },
        )

    def test_api_payload_rejects_invalid_partition(self) -> None:
        host = SimpleNamespace(
            _api_symmetry={
                "axis": 3,
                "plane": 0.0,
                "on_plane_count": 0,
                "pair_count": 1,
            },
            _api_norm=NormalizationTransform(
                center=np.zeros(3, dtype=np.float64),
                scale=1.0,
            ),
        )
        self.assertIsNone(
            MainWindow._api_build_symmetry_payload(
                host, [{"id": 1}, {"id": 2}]))

        host._api_symmetry["axis"] = 0
        host._api_symmetry["pair_count"] = 2
        self.assertIsNone(
            MainWindow._api_build_symmetry_payload(
                host, [{"id": 1}, {"id": 2}]))

    def test_api_base_fit_can_override_live_symmetry_setting(self) -> None:
        host = SimpleNamespace(
            _api_job_id="base-fit",
            _api_fit_existing=False,
            _api_options={"symmetry": False},
            _chk_symmetry=SimpleNamespace(isChecked=lambda: True),
        )
        self.assertFalse(MainWindow._effective_symmetry_enabled(host))

        host._api_options["symmetry"] = True
        self.assertTrue(MainWindow._effective_symmetry_enabled(host))

        host._api_fit_existing = True
        self.assertFalse(MainWindow._effective_symmetry_enabled(host))


if __name__ == "__main__":
    unittest.main()
