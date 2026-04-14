"""
skinning.py — Linear Blend Skinning (LBS) for mesh deformation.

Deforms mesh vertices according to bone transforms and skin weights.
Provides both a NumPy CPU path and an optional Warp GPU kernel.

Usage:
    deformed = deform_mesh(
        rest_vertices,       # (V, 3)
        skin_joints,         # (V, 4) bone indices
        skin_weights,        # (V, 4) weights
        skin_matrices,       # (B, 4, 4) = world_transform @ inverse_bind
    )
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False


# ── Warp GPU kernel ──────────────────────────────────────────────────────────

if _HAS_WARP:
    @wp.kernel
    def _lbs_kernel(
        rest_verts: wp.array(dtype=wp.vec3),
        joints: wp.array2d(dtype=wp.int32),          # (V, 4)
        weights: wp.array2d(dtype=wp.float32),        # (V, 4)
        skin_matrices: wp.array2d(dtype=wp.float32),  # (B*4, 4) flattened rows
        out_verts: wp.array(dtype=wp.vec3),
    ):
        vid = wp.tid()
        v = rest_verts[vid]

        # Accumulate weighted transform
        rx = float(0.0)
        ry = float(0.0)
        rz = float(0.0)

        for k in range(4):
            w = weights[vid, k]
            if w < 1.0e-8:
                continue

            bone = joints[vid, k]
            row0 = bone * 4 + 0
            row1 = bone * 4 + 1
            row2 = bone * 4 + 2

            # Affine transform: M @ [v, 1]
            px = skin_matrices[row0, 0] * v[0] + skin_matrices[row0, 1] * v[1] + skin_matrices[row0, 2] * v[2] + skin_matrices[row0, 3]
            py = skin_matrices[row1, 0] * v[0] + skin_matrices[row1, 1] * v[1] + skin_matrices[row1, 2] * v[2] + skin_matrices[row1, 3]
            pz = skin_matrices[row2, 0] * v[0] + skin_matrices[row2, 1] * v[1] + skin_matrices[row2, 2] * v[2] + skin_matrices[row2, 3]

            rx = rx + w * px
            ry = ry + w * py
            rz = rz + w * pz

        out_verts[vid] = wp.vec3(rx, ry, rz)


# ── NumPy CPU implementation ────────────────────────────────────────────────

def _deform_numpy(
    rest_vertices: np.ndarray,
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    skin_matrices: np.ndarray,
) -> np.ndarray:
    """CPU Linear Blend Skinning using vectorised NumPy.

    Parameters
    ----------
    rest_vertices : (V, 3) float32
    skin_joints : (V, 4) int32 — bone indices
    skin_weights : (V, 4) float32 — blend weights (sum to 1)
    skin_matrices : (B, 4, 4) float64 — skinning matrices per bone

    Returns
    -------
    (V, 3) float32 — deformed vertices
    """
    V = rest_vertices.shape[0]
    # Homogeneous coords: (V, 4)
    ones = np.ones((V, 1), dtype=np.float64)
    v_homo = np.concatenate([rest_vertices.astype(np.float64), ones], axis=1)

    result = np.zeros((V, 3), dtype=np.float64)

    for k in range(skin_joints.shape[1]):  # max 4 influences
        w = skin_weights[:, k].astype(np.float64)  # (V,)
        bone_idx = skin_joints[:, k]                # (V,)

        # Gather the skin matrix for each vertex's bone
        M = skin_matrices[bone_idx]  # (V, 4, 4)

        # Apply transform: (V, 4, 4) @ (V, 4, 1) → (V, 4, 1)
        transformed = np.einsum('vij,vj->vi', M, v_homo)[:, :3]

        result += w[:, np.newaxis] * transformed

    return result.astype(np.float32)


def _deform_warp(
    rest_vertices: np.ndarray,
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    skin_matrices: np.ndarray,
    device: str = "cuda:0",
) -> np.ndarray:
    """GPU Linear Blend Skinning using Warp."""
    V = rest_vertices.shape[0]
    B = skin_matrices.shape[0]

    # Flatten skin matrices to (B*4, 4) for Warp 2D array
    flat_mats = skin_matrices.reshape(B * 4, 4).astype(np.float32)

    wp_verts = wp.array(rest_vertices.astype(np.float32), dtype=wp.vec3, device=device)
    wp_joints = wp.array(skin_joints.astype(np.int32), dtype=wp.int32, device=device, ndim=2)
    wp_weights = wp.array(skin_weights.astype(np.float32), dtype=wp.float32, device=device, ndim=2)
    wp_mats = wp.array(flat_mats, dtype=wp.float32, device=device, ndim=2)
    wp_out = wp.empty(V, dtype=wp.vec3, device=device)

    wp.launch(
        _lbs_kernel,
        dim=V,
        inputs=[wp_verts, wp_joints, wp_weights, wp_mats, wp_out],
        device=device,
    )

    return wp_out.numpy().astype(np.float32)


# ── Public API ───────────────────────────────────────────────────────────────

def deform_mesh(
    rest_vertices: np.ndarray,
    skin_joints: np.ndarray,
    skin_weights: np.ndarray,
    skin_matrices: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """Deform a mesh using Linear Blend Skinning.

    Parameters
    ----------
    rest_vertices : (V, 3) float32 — vertices in bind pose
    skin_joints : (V, K) int32 — bone indices per vertex (K ≤ 4)
    skin_weights : (V, K) float32 — blend weights
    skin_matrices : (B, 4, 4) float64 — world @ inverse_bind per bone
    device : str, optional
        If "cuda:0" or similar, use Warp GPU kernel.
        If None, auto-detect. Falls back to NumPy.

    Returns
    -------
    (V, 3) float32 — deformed vertex positions
    """
    if device is None:
        if _HAS_WARP:
            try:
                if wp.is_cuda_available():
                    device = "cuda:0"
            except Exception:
                pass

    if device and _HAS_WARP and device.startswith("cuda"):
        try:
            return _deform_warp(rest_vertices, skin_joints, skin_weights,
                                skin_matrices, device)
        except Exception:
            pass  # Fall through to CPU

    return _deform_numpy(rest_vertices, skin_joints, skin_weights, skin_matrices)
