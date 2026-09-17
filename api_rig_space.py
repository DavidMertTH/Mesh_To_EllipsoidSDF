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
