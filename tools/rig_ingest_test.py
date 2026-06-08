"""Unit test for rig_ingest.assign_ellipsoids_to_bones (no GUI, no server).

Builds a 2-bone skeleton, places vertices clearly around each bone, fits two
trivial ellipsoids near each bone, and checks:
  * each ellipsoid is assigned to the nearer bone,
  * the bone-local center reproduces the world center when transformed back.

Run:  .venv/Scripts/python tools/rig_ingest_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rig_ingest import assign_ellipsoids_to_bones, build_skeleton_from_bones  # noqa: E402
from skeleton import mat4_decompose, quat_rotate, quat_multiply  # noqa: E402


def main() -> int:
    # Two bones along X: bone A at origin, bone B at x=10, both unrotated.
    rig = {
        "bones": [
            {"name": "BoneA", "position": [0.0, 0.0, 0.0], "rotation": [0, 0, 0, 1]},
            {"name": "BoneB", "position": [10.0, 0.0, 0.0], "rotation": [0, 0, 0, 1]},
        ],
        # 6 verts around A (x≈0), 6 around B (x≈10)
        "boneIndices": [[0, 0, 0, 0]] * 6 + [[1, 1, 1, 1]] * 6,
        "boneWeights": [[1.0, 0, 0, 0]] * 12,
    }
    verts = np.array(
        [[0, 0, 0], [0.5, 0.3, 0], [-0.4, 0.2, 0.1],
         [0.2, -0.3, 0], [0.1, 0.1, 0.4], [-0.2, 0, -0.3]]
        + [[10, 0, 0], [10.5, 0.3, 0], [9.6, 0.2, 0.1],
           [10.2, -0.3, 0], [10.1, 0.1, 0.4], [9.8, 0, -0.3]],
        dtype=np.float64,
    )

    # One ellipsoid near each bone.
    world_centers = np.array([[0.3, 0.1, 0.0], [10.2, -0.1, 0.0]], dtype=np.float64)
    world_radii = np.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]], dtype=np.float64)
    world_rot = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float64)

    entries = assign_ellipsoids_to_bones(
        world_centers, world_radii, world_rot, verts, rig)

    ok = True
    print("assignments:")
    for e in entries:
        print(" ", e["name"], "->", e["bone"], "local_center=", e["local_center"])
    ok &= entries[0]["bone"] == "BoneA"
    ok &= entries[1]["bone"] == "BoneB"

    # Roundtrip: bone-local center → world should match the input world center.
    sk = build_skeleton_from_bones(rig["bones"])
    wts = sk.compute_world_transforms(None)
    for i, e in enumerate(entries):
        bi = sk.bone_index(e["bone"])
        t_bone, q_bone, _ = mat4_decompose(wts[bi])
        wc = quat_rotate(q_bone, np.asarray(e["local_center"])) + t_bone
        err = float(np.linalg.norm(wc - world_centers[i]))
        print(f"  roundtrip {e['name']}: world={wc}  err={err:.2e}")
        ok &= err < 1e-5
        # radii pass through unchanged
        ok &= np.allclose(e["radii"], world_radii[i], atol=1e-5)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
