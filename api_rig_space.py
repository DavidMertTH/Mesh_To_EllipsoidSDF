"""Unity API rig-space contract helpers.

Unity sends its baked mesh vertices, bind transforms, current transforms and
pose frames in one snapshot coordinate space. That space is part of the API
contract and must not be inferred from geometry: a mesh bounding-box centre is
not a skeleton origin (hands and other asymmetric rigs are common examples).

Older versions of EllipSDF tried to repair a suspected local/world mismatch by
aligning the mesh and joint bounding boxes. Besides moving valid rigs, that made
the returned bone-local ellipsoids incompatible with the unchanged Unity bones.
The compatibility entry point below therefore deliberately performs no implicit
correction. A future client that uses another space must provide an explicit
transform and apply it consistently to the complete payload.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _affine_matrices(raw: Any, count: int, label: str) -> np.ndarray:
    matrices = np.asarray(raw, dtype=np.float64)
    if matrices.shape != (count, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError(f"{label} must contain {count} finite 4x4 matrices")
    if not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], atol=1.0e-6):
        raise ValueError(f"{label} must contain affine matrices")
    return matrices


def compare_rig_snapshot(
    vertices: np.ndarray,
    rest_vertices: np.ndarray,
    bind_matrices: np.ndarray,
    current_matrices: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    *,
    name: str = "Unity Current",
) -> dict[str, Any]:
    """Compare baked vertices with LBS in the original snapshot space.

    Offset and uniform scale are diagnostic measurements only. They never
    change the mesh, bones, poses, or the bone-local coordinates returned to Unity.
    """
    actual = np.asarray(vertices, dtype=np.float64)
    rest = np.asarray(rest_vertices, dtype=np.float64)
    if (actual.ndim != 2 or actual.shape[1] != 3 or len(actual) == 0
            or rest.shape != actual.shape
            or not np.isfinite(actual).all() or not np.isfinite(rest).all()):
        raise ValueError(f"{name}: mesh and restVertices must be finite matching Nx3 arrays")
    bone_count = len(bind_matrices)
    bind = _affine_matrices(bind_matrices, bone_count, f"{name}: bind matrices")
    current = _affine_matrices(current_matrices, bone_count, f"{name}: current matrices")
    js = np.asarray(joints)
    ws = np.asarray(weights, dtype=np.float64)
    if (js.ndim != 2 or js.shape != ws.shape or len(js) != len(actual)
            or not np.isfinite(ws).all() or np.any(ws < 0)
            or not np.isfinite(js).all() or np.any(js != np.floor(js))):
        raise ValueError(f"{name}: invalid boneIndices/boneWeights")
    active = ws > 0
    if np.any((js[active] < 0) | (js[active] >= bone_count)):
        raise ValueError(f"{name}: boneIndices reference a missing bone")
    if not np.allclose(ws.sum(axis=1), 1.0, atol=1.0e-4, rtol=0):
        raise ValueError(f"{name}: boneWeights must sum to one per vertex")
    try:
        skin = current @ np.linalg.inv(bind)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name}: singular bind matrix") from exc
    expected = np.zeros_like(actual)
    for slot in range(js.shape[1]):
        used = active[:, slot]
        matrices = skin[js[used, slot].astype(np.int64)]
        predicted = np.einsum("vij,vj->vi", matrices[:, :3, :3], rest[used])
        predicted += matrices[:, :3, 3]
        expected[used] += predicted * ws[used, slot, None]
    if not np.isfinite(expected).all():
        raise ValueError(f"{name}: reconstructing the mesh produced non-finite positions")
    residual = actual - expected
    errors = np.linalg.norm(residual, axis=1)
    extent = float(np.ptp(actual, axis=0).max())
    # Account for Unity's float32 world positions, including far-away objects.
    magnitude = float(max(np.abs(actual).max(), np.abs(expected).max()))
    tolerance = max(1.0e-6, extent * 1.0e-4,
                    abs(float(np.spacing(np.float32(magnitude)))) * 4)
    offset = residual.mean(axis=0)
    expected_centered = expected - expected.mean(axis=0)
    actual_centered = actual - actual.mean(axis=0)
    denominator = float(np.sum(expected_centered ** 2))
    ratio = (float(np.sum(expected_centered * actual_centered)) / denominator
             if denominator > 1.0e-30 else 1.0)
    max_error = float(errors.max())
    classification = "aligned"
    if max_error > tolerance:
        if float(np.linalg.norm(residual - offset, axis=1).max()) <= tolerance:
            classification = "translation"
        elif (ratio > 0 and abs(ratio - 1) > 1.0e-4
              and float(np.linalg.norm(
                  actual_centered - ratio * expected_centered, axis=1).max()) <= tolerance):
            classification = "scale"
        else:
            classification = "pose_or_transform"
    return {
        "name": name,
        "status": "ok" if classification == "aligned" else "mismatch",
        "classification": classification,
        "rmsError": float(np.sqrt(np.mean(errors ** 2))),
        "maxError": max_error,
        "relativeMaxError": max_error / max(extent, 1.0e-30),
        "tolerance": tolerance,
        "meanOffset": offset.tolist(),
        "scaleRatio": ratio,
        "worstVertex": int(errors.argmax()),
    }


_UNSUPPORTED_COMPARISONS = {
    "active_blend_shapes", "more_than_four_influences", "reduced_skin_quality",
    "missing_bind_poses",
}


def validate_unity_rig_snapshot(
    rig: dict[str, Any] | None,
    vertices: np.ndarray,
) -> dict[str, Any]:
    """Independently validate a Unity snapshot and all supplied pose frames.

    Older clients without rest/bind data remain usable. Explicitly declared
    deformations that the four-weight LBS contract cannot reproduce are reported
    as skipped, rather than mistaken for a coordinate-space error.
    """
    if not rig or not rig.get("bones"):
        return {"status": "skipped", "reason": "no_rig", "snapshots": []}
    bones = list(rig["bones"])
    required = ("restVertices", "boneIndices", "boneWeights")
    if any(rig.get(key) is None for key in required) or any(
            b.get("matrix") is None for b in bones):
        return {"status": "skipped", "reason": "missing_rest_or_bind_data", "snapshots": []}
    declared_space = rig.get("coordinateSpace")
    if declared_space not in (None, "unity_world"):
        raise ValueError(f"Unsupported rig coordinateSpace: {declared_space!r}")
    bind = _affine_matrices([b["matrix"] for b in bones], len(bones), "rig.bones.matrix")
    current = [b.get("currentMatrix", b["matrix"]) for b in bones]
    rest = np.asarray(rig["restVertices"], dtype=np.float64)
    joints = np.asarray(rig["boneIndices"])
    weights = np.asarray(rig["boneWeights"], dtype=np.float64)

    def _compare(v, matrices, name, declaration):
        declaration = declaration if isinstance(declaration, dict) else {}
        if (declaration.get("status") == "skipped"
                and declaration.get("reason") in _UNSUPPORTED_COMPARISONS):
            return {"name": name, "status": "skipped", "reason": declaration["reason"]}
        return compare_rig_snapshot(v, rest, bind, matrices, joints, weights, name=name)

    reports = [_compare(vertices, current, "Unity Current", rig.get("snapshotValidation"))]
    for index, frame in enumerate(rig.get("poseFrames") or []):
        if not isinstance(frame, dict):
            raise ValueError(f"rig.poseFrames[{index}] must be an object")
        matrices = frame.get("boneMatrices", frame.get("currentMatrices", frame.get("matrices")))
        if frame.get("vertices") is None or matrices is None:
            raise ValueError(f"rig.poseFrames[{index}] needs vertices and boneMatrices")
        reports.append(_compare(
            frame["vertices"], matrices, str(frame.get("name") or f"Unity Pose {index:02d}"),
            frame.get("snapshotValidation")))
    status = ("mismatch" if any(r["status"] == "mismatch" for r in reports)
              else "ok" if all(r["status"] == "ok" for r in reports)
              else "skipped" if all(r["status"] == "skipped" for r in reports) else "partial")
    return {"status": status, "snapshots": reports}


def rig_snapshot_error(validation: dict[str, Any]) -> str:
    """Explain the first mismatch in caller units, before a fit is started."""
    report = next(r for r in validation["snapshots"] if r["status"] == "mismatch")
    offset = ", ".join(f"{v:.6g}" for v in report["meanOffset"])
    return (
        f"Mesh/skeleton mismatch in {report['name']} ({report['classification']}): "
        f"max error {report['maxError']:.6g}, tolerance {report['tolerance']:.6g}, "
        f"mesh offset [{offset}], scale ratio {report['scaleRatio']:.6g}. "
        "Capture baked vertices and current bone matrices in the same Unity world-space "
        "pose; check renderer/root transforms and apply scale exactly once."
    )


def correct_unity_rig_space(
    rig: dict[str, Any] | None,
    vertices: np.ndarray,
) -> tuple[dict[str, Any] | None, np.ndarray, str | None]:
    """Return the rig unchanged and report that no correction was applied.

    ``vertices`` remains in the signature for callers using the former helper.
    It is intentionally not inspected: no reliable rigid transform can be
    recovered from mesh and bone bounding boxes.
    """
    del vertices
    return rig, np.zeros(3, dtype=np.float64), None
