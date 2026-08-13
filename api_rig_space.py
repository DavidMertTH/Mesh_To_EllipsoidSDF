"""Helpers for keeping Unity API mesh and rig payloads in one space.

The API contract is simple: vertices, bone matrices, ellipsoids and returned
results must all share one snapshot coordinate space.  Some Unity exporters can
accidentally send mesh vertices in renderer-local space while bone matrices are
still in Unity world space.  That shows up as a skeleton that is uniformly
shifted away from the mesh in EllipSDF and also corrupts bone-local ellipsoid
output.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def _matrix_from_payload(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        return np.asarray(raw, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None


def _bone_positions(rig: dict[str, Any] | None) -> np.ndarray | None:
    if not (rig and rig.get("bones")):
        return None
    pts: list[np.ndarray] = []
    for b in list(rig.get("bones") or []):
        if not isinstance(b, dict):
            continue
        m = _matrix_from_payload(b.get("currentMatrix"))
        if m is None:
            m = _matrix_from_payload(b.get("matrix"))
        if m is not None:
            pts.append(m[:3, 3].astype(np.float64))
            continue
        raw_pos = b.get("position")
        if raw_pos is None:
            continue
        try:
            pts.append(np.asarray(raw_pos, dtype=np.float64).reshape(3))
        except Exception:
            continue
    if not pts:
        return None
    return np.vstack(pts)


def _axis_overlap_ratio(a_min: float, a_max: float,
                        b_min: float, b_max: float) -> float:
    denom = min(float(a_max - a_min), float(b_max - b_min))
    if denom <= 1.0e-12:
        return 1.0
    overlap = min(float(a_max), float(b_max)) - max(float(a_min), float(b_min))
    return float(np.clip(overlap / denom, 0.0, 1.0))


def _space_correction(vertices: np.ndarray,
                      bone_positions: np.ndarray) -> tuple[np.ndarray, str | None]:
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    bones = np.asarray(bone_positions, dtype=np.float64).reshape(-1, 3)
    if len(verts) == 0 or len(bones) == 0:
        return np.zeros(3, dtype=np.float64), None

    mesh_min = verts.min(axis=0)
    mesh_max = verts.max(axis=0)
    bone_min = bones.min(axis=0)
    bone_max = bones.max(axis=0)
    mesh_size = mesh_max - mesh_min
    extent = float(np.max(mesh_size))
    if not np.isfinite(extent) or extent <= 1.0e-12:
        return np.zeros(3, dtype=np.float64), None

    mesh_center = (mesh_min + mesh_max) * 0.5
    bone_center = (bone_min + bone_max) * 0.5
    delta = mesh_center - bone_center

    overlaps = np.array([
        _axis_overlap_ratio(mesh_min[i], mesh_max[i], bone_min[i], bone_max[i])
        for i in range(3)
    ], dtype=np.float64)
    rel = np.abs(delta) / extent

    # Conservative auto-fix: only translate axes whose centres are noticeably
    # apart.  Y gets an extra path because the common failure is a vertical
    # renderer/root offset while the skeleton still partly overlaps the mesh.
    correction = np.zeros(3, dtype=np.float64)
    axis_bad = (rel > 0.08) & (overlaps < 0.85)
    correction[axis_bad] = delta[axis_bad]
    if rel[1] > 0.12:
        correction[1] = delta[1]

    if not np.any(np.abs(correction) > extent * 1.0e-5):
        return np.zeros(3, dtype=np.float64), None

    reason = (
        "bone bbox was offset from mesh bbox; applying translation "
        f"[{correction[0]:.6g}, {correction[1]:.6g}, {correction[2]:.6g}]"
    )
    return correction, reason


def _translated_matrix(raw: Any, delta: np.ndarray) -> Any:
    m = _matrix_from_payload(raw)
    if m is None:
        return raw
    out = m.copy()
    out[:3, 3] += delta
    return out.tolist()


def _translated_position(raw: Any, delta: np.ndarray) -> Any:
    if raw is None:
        return raw
    try:
        p = np.asarray(raw, dtype=np.float64).reshape(3) + delta
        return p.tolist()
    except Exception:
        return raw


def _translate_rig_payload(rig: dict[str, Any],
                           delta: np.ndarray,
                           reason: str) -> dict[str, Any]:
    fixed = deepcopy(rig)
    for b in list(fixed.get("bones") or []):
        if not isinstance(b, dict):
            continue
        for key in ("matrix", "currentMatrix"):
            if key in b:
                b[key] = _translated_matrix(b.get(key), delta)
        if "position" in b:
            b["position"] = _translated_position(b.get("position"), delta)

    for frame in list(fixed.get("poseFrames") or []):
        if not isinstance(frame, dict):
            continue
        for key in ("boneMatrices", "currentMatrices", "matrices"):
            mats = frame.get(key)
            if mats is None:
                continue
            frame[key] = [_translated_matrix(m, delta) for m in list(mats)]

    fixed["_ellipsdf_space_correction"] = {
        "translation": [float(v) for v in delta],
        "reason": reason,
    }
    return fixed


def correct_unity_rig_space(
    rig: dict[str, Any] | None,
    vertices: np.ndarray,
) -> tuple[dict[str, Any] | None, np.ndarray, str | None]:
    """Return a rig payload whose bones share the posted vertex space.

    The function is intentionally conservative.  If the rig already overlaps the
    mesh in the same coordinate frame, it returns the original object unchanged.
    If it detects the typical Unity world-vs-renderer-local translation mismatch,
    it returns a deep-copied rig with all bone matrices/positions translated into
    the mesh snapshot space.
    """
    if not (rig and rig.get("bones")):
        return rig, np.zeros(3, dtype=np.float64), None
    bones = _bone_positions(rig)
    if bones is None:
        return rig, np.zeros(3, dtype=np.float64), None
    delta, reason = _space_correction(vertices, bones)
    if reason is None:
        return rig, delta, None
    return _translate_rig_payload(rig, delta, reason), delta, reason
