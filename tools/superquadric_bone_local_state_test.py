"""Regression checks for shape metadata in bone-local primitive state."""

from pathlib import Path
import json
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bone_ellipsoid_mapper import BoneEllipsoidMapper, BoneLocalEllipsoids  # noqa: E402
from ellipsoid_exporter import export_ellipsoids  # noqa: E402
from skeleton import Bone, Skeleton  # noqa: E402


def _identity_skeleton() -> Skeleton:
    matrix = np.eye(4, dtype=np.float64)
    return Skeleton([Bone(
        name="Root",
        index=0,
        parent_index=-1,
        local_bind_transform=matrix,
        inverse_bind_matrix=matrix,
    )])


def main() -> int:
    centers = np.array([[0.2, -0.1, 0.3]], dtype=np.float32)
    radii = np.array([[0.4, 0.25, 0.15]], dtype=np.float32)
    rotations = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    assignments = np.array([0], dtype=np.int32)

    legacy = BoneLocalEllipsoids(
        local_centers=centers,
        local_radii=radii,
        local_rotations=rotations,
        bone_assignments=assignments,
    )
    assert legacy.primitive_type == "ellipsoid"
    np.testing.assert_array_equal(
        legacy.shape_exponents, np.ones((1, 2), dtype=np.float32))

    eps = np.array([[0.45, 1.35]], dtype=np.float32)
    mapper = BoneEllipsoidMapper(_identity_skeleton())
    local = mapper.world_to_local(
        centers,
        radii,
        rotations,
        assignments,
        primitive_type="superquadric",
        shape_exponents=eps,
    )
    assert local.primitive_type == "superquadric"
    np.testing.assert_allclose(local.shape_exponents, eps)

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "superquadrics.json"
        assert export_ellipsoids(local, mapper.skeleton, output) == 1
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["version"] == 4
        assert payload["primitive_type"] == "superquadric"
        assert payload["ellipsoids"][0]["primitive_type"] == "superquadric"
        np.testing.assert_allclose(
            payload["ellipsoids"][0]["shape_exponents"], eps[0])
    world_centers, world_radii, world_rotations = mapper.local_to_world_np(local)
    np.testing.assert_allclose(world_centers, centers, atol=1.0e-6)
    np.testing.assert_allclose(world_radii, radii, atol=1.0e-6)
    np.testing.assert_allclose(
        np.abs(np.sum(world_rotations * rotations, axis=1)), [1.0], atol=1.0e-6)
    np.testing.assert_allclose(local.shape_exponents, eps)

    try:
        BoneLocalEllipsoids(
            local_centers=centers,
            local_radii=radii,
            local_rotations=rotations,
            bone_assignments=assignments,
            primitive_type="superquadric",
            shape_exponents=np.array([[0.09, 1.0]], dtype=np.float32),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range superquadric exponent was accepted")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
