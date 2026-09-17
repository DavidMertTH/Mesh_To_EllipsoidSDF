"""Protocol-v4 regression tests for ellipsoid/SQ API pose fitting."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_window import MainWindow  # noqa: E402


class _Transform:
    scale = 2.0
    center = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def to_original_point(self, point):
        return np.asarray(point, dtype=np.float64) / self.scale + self.center

    def to_original_length(self, length):
        return np.asarray(length, dtype=np.float64) / self.scale


def _owner(**fields):
    defaults = dict(
        _api_canonical_primitive_type=MainWindow._api_canonical_primitive_type,
        _api_shape_exponents_array=MainWindow._api_shape_exponents_array,
        _normalize_quat_np=MainWindow._normalize_quat_np,
    )
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def _rig():
    identity = np.eye(4, dtype=np.float64).tolist()
    return {
        "bones": [{
            "name": "Root",
            "parent": -1,
            "matrix": identity,
            "currentMatrix": identity,
        }]
    }


def _entry(index: int, *, primitive_type=None, eps=None, alias=False):
    entry = {
        "id": index,
        "name": f"P{index}",
        "bone": "Root",
        "bone_index": 0,
        "center": [1.0 + index, 2.0, 3.0],
        "radii": [0.5, 0.4, 0.3],
        "rotation": [0.0, 0.0, 0.0, 1.0],
    }
    if primitive_type is not None:
        entry["primitive_type"] = primitive_type
    if eps is not None:
        entry["eps" if alias else "shape_exponents"] = eps
    return entry


class ApiSuperquadricPoseFitTest(unittest.TestCase):
    def test_pose_refit_settings_and_request_overrides(self) -> None:
        defaults = MainWindow._api_pose_fit_transform_flags({}, {})
        self.assertEqual(defaults, {
            "optimize_centers": True,
            "optimize_rotations": True,
            "optimize_radii": True,
        })
        resolved = MainWindow._api_pose_fit_transform_flags(
            {
                "pose_fit_position": False,
                "pose_fit_rotation": False,
                "pose_fit_scale": True,
            },
            {"pose_fit_rotation": True, "pose_fit_scale": False},
        )
        self.assertEqual(resolved, {
            "optimize_centers": False,
            "optimize_rotations": True,
            "optimize_radii": False,
        })
        self.assertEqual(MainWindow._optimizer_settings({
            "pose_fit_position": False,
            "pose_fit_rotation": False,
            "pose_fit_scale": False,
        }), {})
        with self.assertRaisesRegex(ValueError, "options.pose_fit_scale"):
            MainWindow._api_pose_fit_transform_flags(
                {}, {"pose_fit_scale": "false"})

    def test_api_iteration_budget_is_validated_and_overrides_gui_per_job(self) -> None:
        self.assertIsNone(MainWindow._api_fit_steps_from_options({}))
        self.assertEqual(
            MainWindow._api_fit_steps_from_options({"num_steps": 350}), 350)
        self.assertEqual(
            MainWindow._api_fit_steps_from_options({"numSteps": 75}), 75)
        for value in (None, True, 0, -1, 1.5, "350", 1_000_001):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "options.num_steps"):
                MainWindow._api_fit_steps_from_options({"num_steps": value})
        with self.assertRaisesRegex(ValueError, "disagree"):
            MainWindow._api_fit_steps_from_options(
                {"num_steps": 100, "numSteps": 200})

        value = SimpleNamespace(value=lambda: 7000)
        owner = SimpleNamespace(
            _shape=SimpleNamespace(fit_kwargs=lambda: {}),
            _spin_max_steps=value,
            _report_every=20,
            _effective_symmetry_enabled=lambda: False,
            _spin_lr_init=SimpleNamespace(value=lambda: 0.01),
            _spin_lr_final=SimpleNamespace(value=lambda: 0.001),
            _spin_lr_decay=SimpleNamespace(value=lambda: 7.0),
            _settings={},
            _api_job_id="first-run",
            _api_options={"num_steps": 1200},
            _api_shape_fit_kwargs={},
        )
        self.assertEqual(MainWindow._gather_fit_kwargs(owner)["num_steps"], 1200)
        owner._api_options = {"num_steps": 120}
        self.assertEqual(MainWindow._gather_fit_kwargs(owner)["num_steps"], 120)
        owner._api_job_id = None
        self.assertEqual(MainWindow._gather_fit_kwargs(owner)["num_steps"], 7000)

    def test_legacy_request_is_canonical_ellipsoid(self) -> None:
        owner = _owner(_api_primitive_type="ellipsoid")
        parsed = MainWindow._api_parse_initial_ellipsoids(
            owner,
            {"ellipsoids": [_entry(3)], "rig": _rig()},
            _Transform(),
        )
        self.assertEqual(len(parsed), 5)
        np.testing.assert_allclose(parsed[3], [[1.0, 1.0]])
        self.assertEqual(parsed[4][0]["primitive_type"], "ellipsoid")
        self.assertEqual(parsed[4][0]["id"], 3)

    def test_superquadric_shape_and_eps_alias_round_trip_parse(self) -> None:
        owner = _owner(_api_primitive_type="superquadric")
        parsed = MainWindow._api_parse_initial_ellipsoids(
            owner,
            {
                "primitive_type": "superquadric",
                "ellipsoids": [
                    _entry(7, primitive_type="superquadric", eps=[0.45, 1.35]),
                    _entry(
                        9, primitive_type="superquadric", eps=[1.8, 0.65],
                        alias=True),
                ],
                "rig": _rig(),
            },
            _Transform(),
        )
        np.testing.assert_allclose(parsed[3], [[0.45, 1.35], [1.8, 0.65]])
        self.assertEqual([m["id"] for m in parsed[4]], [7, 9])
        self.assertTrue(all(
            m["primitive_type"] == "superquadric" for m in parsed[4]))

    def test_request_shape_options_override_gui_and_are_validated(self) -> None:
        kind, kwargs = MainWindow._api_resolve_primitive_request(
            {"primitive_type": "superquadric"},
            {
                "primitive_shape": "superquadric",
                "sq_eps1": 0.4,
                "sq_eps2": 1.6,
                "sq_eps_mode": "fixed",
                "sq_unlock_frac": 0.25,
            },
        )
        self.assertEqual(kind, "superquadric")
        self.assertEqual(kwargs["primitive_shape"], "superquadric")
        self.assertEqual(kwargs["sq_eps_mode"], "fixed")

        value = SimpleNamespace(value=lambda: 3)
        shape = SimpleNamespace(fit_kwargs=lambda: {
            "primitive_shape": "ellipsoid", "sdf_mode": 0})
        gather_owner = SimpleNamespace(
            _shape=shape,
            _spin_max_steps=value,
            _report_every=2,
            _effective_symmetry_enabled=lambda: False,
            _spin_lr_init=SimpleNamespace(value=lambda: 0.01),
            _spin_lr_final=SimpleNamespace(value=lambda: 0.001),
            _spin_lr_decay=SimpleNamespace(value=lambda: 7.0),
            _settings={},
            _api_job_id="job",
            _api_shape_fit_kwargs=kwargs,
        )
        gathered = MainWindow._gather_fit_kwargs(gather_owner)
        self.assertEqual(gathered["primitive_shape"], "superquadric")
        self.assertEqual(gathered["sq_eps1"], 0.4)

        with self.assertRaisesRegex(ValueError, "disagree"):
            MainWindow._api_resolve_primitive_request(
                {"primitive_type": "ellipsoid"},
                {"primitive_shape": "superquadric"})
        with self.assertRaisesRegex(ValueError, "not supported"):
            MainWindow._api_resolve_primitive_request(
                {"primitive_type": "bent_superquadric"}, {})

    def test_ids_entry_types_and_exponents_are_strict(self) -> None:
        owner = _owner(_api_primitive_type="superquadric")
        cases = [
            ([_entry(1, primitive_type="superquadric", eps=[0.5, 0.8]),
              _entry(1, primitive_type="superquadric", eps=[0.6, 0.9])],
             "duplicate primitive id"),
            ([_entry(1, eps=[0.5, 0.8])], "does not match"),
            ([_entry(1, primitive_type="superquadric", eps=[0.05, 0.8])],
             r"inside \[0.1, 2.0\]"),
        ]
        for entries, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message):
                MainWindow._api_parse_initial_ellipsoids(
                    owner, {"ellipsoids": entries, "rig": _rig()}, _Transform())

    def test_preview_and_result_publish_v4_shape_metadata(self) -> None:
        eps = np.array([[0.4, 0.7], [1.2, 1.8]], dtype=np.float32)
        centers = np.zeros((2, 3), dtype=np.float32)
        radii = np.ones((2, 3), dtype=np.float32)
        rotations = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1))
        owner = _owner(
            _api_norm=_Transform(),
            _api_primitive_type="superquadric",
            _api_rig=None,
            _api_fit_existing=False,
            _api_symmetry=None,
            _api_build_symmetry_payload=lambda _entries: None,
        )
        preview = MainWindow._api_build_world_preview_payload(
            owner, centers, radii, rotations, eps)
        result = MainWindow._api_build_result_payload(
            owner, centers, radii, rotations, eps)
        for payload in (preview, result):
            self.assertEqual(payload["version"], 4)
            self.assertEqual(payload["primitive_type"], "superquadric")
            self.assertEqual(payload["count"], 2)
            self.assertTrue(all(
                e["primitive_type"] == "superquadric"
                and len(e["shape_exponents"]) == 2
                for e in payload["ellipsoids"]))
        np.testing.assert_allclose(
            [e["shape_exponents"] for e in result["ellipsoids"]], eps)
        with self.assertRaisesRegex(ValueError, "required for superquadric"):
            MainWindow._api_build_result_payload(
                owner, centers, radii, rotations, None)

    def test_optimizer_progress_keeps_sq_exponents_in_api_last(self) -> None:
        eps = np.array([[0.35, 1.4]], dtype=np.float32)
        owner = SimpleNamespace(
            _api_job_id="job",
            _api_stage="fit",
            _api_primitive_type="superquadric",
            _api_batch_pipeline=True,
            _api_server=None,
            _api_progress_last_time=0.0,
            _pending_visual=None,
        )
        centers = np.zeros((1, 3), dtype=np.float32)
        radii = np.ones((1, 3), dtype=np.float32)
        rotations = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        MainWindow._on_opt_step_visual(
            owner, 1, 0.5, centers, radii, rotations, eps)
        np.testing.assert_allclose(owner._api_last[3], eps)
        self.assertIsNone(owner._pending_visual)

    def test_api_superquadrics_bypass_ui_bone_separation(self) -> None:
        calls: list[str] = []
        owner = SimpleNamespace(
            _api_server=SimpleNamespace(registry=SimpleNamespace(
                is_cancel_requested=lambda _job_id: False)),
            _api_fit_existing=False,
            _api_rig={"bones": [{}], "boneIndices": [[0]], "boneWeights": [[1.0]]},
            _api_train_correctives=False,
            _rig_panel=SimpleNamespace(shape_fitting_enabled=False),
            _api_options={},
            _api_primitive_type="superquadric",
            _cmb_fit_scope=SimpleNamespace(currentData=lambda: "bone"),
            _base_verts=np.zeros((1, 3), dtype=np.float32),
            _base_faces=np.zeros((1, 3), dtype=np.int32),
            _start_full_object_fit=lambda: calls.append("full"),
            _on_fit_clicked=lambda: calls.append("ui"),
        )
        MainWindow._api_start_fit_impl(owner, "job")
        self.assertEqual(calls, ["full"])
        self.assertEqual(owner._api_stage, "fit")

    def test_fit_pose_forwards_each_initial_epsilon_pair(self) -> None:
        calls: list[dict] = []
        centers = np.zeros((2, 3), dtype=np.float32)
        radii = np.ones((2, 3), dtype=np.float32)
        rotations = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1))
        eps = np.array([[0.35, 0.55], [1.4, 1.75]], dtype=np.float32)
        owner = SimpleNamespace(
            _api_initial_ellipsoids=(centers, radii, rotations, eps),
            _api_existing_local_parameterization=lambda: None,
            _api_batch_pipeline=False,
            _api_primitive_type="superquadric",
            _settings={
                "pose_fit_position": False,
                "pose_fit_rotation": True,
                "pose_fit_scale": False,
            },
            _api_options={},
            _gather_fit_kwargs=lambda: {"primitive_shape": "superquadric"},
            start_optimization=lambda **kwargs: calls.append(kwargs),
            _api_fail=lambda *_args: self.fail("fit-pose unexpectedly failed"),
            _api_last=None,
            _api_stage="sdf",
        )
        MainWindow._api_start_existing_ellipsoid_fit(owner)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["fixed_population"])
        self.assertEqual(calls[0]["primitive_shape"], "superquadric")
        self.assertEqual(calls[0]["parameter_options"], {
            "optimize_centers": False,
            "optimize_rotations": True,
            "optimize_radii": False,
        })
        np.testing.assert_allclose(calls[0]["initial_eps"], eps)
        np.testing.assert_allclose(owner._api_last[3], eps)

    def test_fixed_population_does_not_force_ellipsoid(self) -> None:
        source = inspect.getsource(MainWindow.start_optimization)
        fixed_block = source.split("if fixed_population:", 1)[1].split(
            "if parameter_options:", 1)[0]
        self.assertNotIn('"primitive_shape"', fixed_block)

    def test_corrective_object_center_deltas_are_unscaled_for_unity(self) -> None:
        payload = {
            "base": [{
                "local_center": [2.0, 4.0, 6.0],
                "radii": [1.0, 2.0, 3.0],
            }],
            "poses": [{
                "delta_centers": [[0.8, -0.4, 0.2]],
                "ellipsoids": [{
                    "id": 0,
                    "delta_local_center": [0.8, -0.4, 0.2],
                }],
            }],
        }
        owner = SimpleNamespace(
            _pose_correctives=SimpleNamespace(
                to_json=lambda _skeleton: payload),
            _rig_panel=SimpleNamespace(
                rigged_mesh=SimpleNamespace(skeleton=object())),
            _api_norm=SimpleNamespace(scale=4.0),
        )

        result = MainWindow._api_pose_correctives_payload(owner)

        np.testing.assert_allclose(
            result["poses"][0]["delta_centers"][0],
            [0.2, -0.1, 0.05],
        )
        np.testing.assert_allclose(
            result["poses"][0]["ellipsoids"][0]
            ["delta_local_center"],
            [0.2, -0.1, 0.05],
        )


if __name__ == "__main__":
    unittest.main()
