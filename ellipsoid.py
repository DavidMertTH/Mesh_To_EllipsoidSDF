from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import trimesh
import warp as wp


# ── SDF method constants ──────────────────────────────────────────────────────
SDF_QUILEZ  = 0
SDF_SC_MIN  = 1
SDF_SC_MEAN = 2
SDF_SC_GMEAN = 3
SDF_MERTSTEIN = 4

SDF_METHOD_NAMES = {
    SDF_QUILEZ:    "Quílez",
    SDF_SC_MIN:    "Scaled (min r)",
    SDF_SC_MEAN:   "Scaled (mean r)",
    SDF_SC_GMEAN:  "Scaled (geom. mean r)",
    SDF_MERTSTEIN: "MertStein",
}


def best_device() -> str:
    """Return 'cuda:0' if a CUDA device is available, otherwise 'cpu'."""
    try:
        if wp.is_cuda_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


@wp.kernel
def _ellipsoid_union_sdf_kernel(
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quat),
    num_ellipsoids: int,
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    out_sdf: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    ix = tid % nx
    iy = (tid // nx) % ny
    iz = tid // (nx * ny)

    p = origin + wp.vec3(
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dx,
        (float(iz) + 0.5) * dx,
    )

    min_d = float(1.0e6)

    for i in range(num_ellipsoids):
        # Transform to ellipsoid-local frame
        local_p = wp.quat_rotate_inv(rotations[i], p - centers[i])
        r = radii[i]

        #  MertStein hybrid SDF:
        #   k0 = |p / r|
        #   inside  (k0 < 1):  sdf ≈ (k0 − 1) · min(r)
        #   outside (k0 ≥ 1):  sdf ≈ k0 · (k0 − 1) / k1   (Quílez)
        scaled = wp.vec3(
            local_p[0] / r[0],
            local_p[1] / r[1],
            local_p[2] / r[2],
        )

        k0 = wp.length(scaled)

        d = float(1.0e6)
        if k0 < 1.0:
            # Interior → Scaled-Sphere (min r)
            r_min = wp.min(wp.min(r[0], r[1]), r[2])
            d = (k0 - 1.0) * r_min
        else:
            # Exterior → Quílez
            scaled2 = wp.vec3(
                local_p[0] / (r[0] * r[0]),
                local_p[1] / (r[1] * r[1]),
                local_p[2] / (r[2] * r[2]),
            )
            k1 = wp.length(scaled2)
            if k1 > 1.0e-12:
                d = k0 * (k0 - 1.0) / k1

        if d < min_d:
            min_d = d

    out_sdf[tid] = min_d



@dataclass
class Ellipsoid:
    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray

    @staticmethod
    def identity_quat() -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)



class EllipsoidSet:

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.centers: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.radii: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.rotations: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self.colors: list = None


    @classmethod
    def from_list(cls, ellipsoids: List[Ellipsoid], device: str = "cpu") -> "EllipsoidSet":
        es = cls(device=device)
        if not ellipsoids:
            return es
        es.centers = np.stack([e.center for e in ellipsoids]).astype(np.float32)
        es.radii = np.stack([e.radii for e in ellipsoids]).astype(np.float32)
        es.rotations = np.stack([e.rotation for e in ellipsoids]).astype(np.float32)
        return es

    @property
    def count(self) -> int:
        return self.centers.shape[0]


    def set_parameters(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        rotations: np.ndarray,
        colors: Optional[list] = None,
    ) -> None:

        assert centers.shape[0] == radii.shape[0] == rotations.shape[0]
        self.centers = centers.astype(np.float32, copy=False)
        self.radii = radii.astype(np.float32, copy=False)
        self.rotations = rotations.astype(np.float32, copy=False)
        self.colors = colors

    def compute_sdf_grid(
        self,
        origin: np.ndarray,
        dx: float,
        n: int,
        sdf_mode: int = SDF_MERTSTEIN,
    ) -> np.ndarray:

        if self.count == 0:
            return np.full((n, n, n), 1e6, dtype=np.float32)

        wp_centers = wp.array(self.centers, dtype=wp.vec3, device=self.device)
        wp_radii = wp.array(self.radii, dtype=wp.vec3, device=self.device)
        wp_rotations = wp.array(self.rotations, dtype=wp.quat, device=self.device)

        wp_origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))

        nx = ny = nz = int(n)
        total = nx * ny * nz
        out = wp.empty(total, dtype=wp.float32, device=self.device)

        wp.launch(
            kernel=_ellipsoid_union_sdf_kernel,
            dim=total,
            inputs=[
                wp_centers, wp_radii, wp_rotations,
                self.count,
                wp_origin, float(dx),
                nx, ny, nz,
                out,
            ],
            device=self.device,
        )

        return out.numpy().reshape((nz, ny, nx)).astype(np.float32, copy=False)


    def generate_meshes(self, subdivisions: int = 3) -> List[trimesh.Trimesh]:

        meshes = []
        for i in range(self.count):
            sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
            verts = sphere.vertices.astype(np.float64)

            verts *= self.radii[i].astype(np.float64)

            quat_xyzw = self.rotations[i]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
            R = trimesh.transformations.quaternion_matrix(quat_wxyz)[:3, :3]
            verts = verts @ R.T

            # Translate
            verts += self.centers[i].astype(np.float64)

            mesh = trimesh.Trimesh(
                vertices=verts.astype(np.float32),
                faces=sphere.faces.copy(),
                process=False,
            )
            meshes.append(mesh)

        return meshes



def create_demo_ellipsoids(device: str = "cpu") -> EllipsoidSet:

    q_id = Ellipsoid.identity_quat()

    angle = np.radians(45.0)
    half = angle * 0.5
    q_tilt_z = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)

    angle_x = np.radians(30.0)
    half_x = angle_x * 0.5
    q_tilt_x = np.array([np.sin(half_x), 0.0, 0.0, np.cos(half_x)], dtype=np.float32)

    ellipsoids = [
        Ellipsoid(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radii=np.array([0.5, 0.3, 0.3], dtype=np.float32),
            rotation=q_id,
        ),
        Ellipsoid(
            center=np.array([0.4, 0.4, 0.0], dtype=np.float32),
            radii=np.array([0.25, 0.15, 0.2], dtype=np.float32),
            rotation=q_tilt_z,
        ),
        Ellipsoid(
            center=np.array([-0.3, -0.3, 0.2], dtype=np.float32),
            radii=np.array([0.3, 0.2, 0.15], dtype=np.float32),
            rotation=q_tilt_x,
        ),
        Ellipsoid(
            center=np.array([0.0, 0.5, -0.3], dtype=np.float32),
            radii=np.array([0.15, 0.35, 0.15], dtype=np.float32),
            rotation=q_id,
        ),
        Ellipsoid(
            center=np.array([-0.5, 0.1, 0.1], dtype=np.float32),
            radii=np.array([0.2, 0.2, 0.35], dtype=np.float32),
            rotation=q_tilt_z,
        ),
    ]

    return EllipsoidSet.from_list(ellipsoids, device=device)