"""
bonesep_controller.py — orchestration for sequential per-bone Bone-Separation.

Each bone is fitted on its own, one after another:

    for each fit-bone:
        1. compute its isolated region SDF (the cheap phase), then
        2. fit it with the single-bone ``OptimizationWorker`` against THAT SDF —
           full maintenance / densify / local-fit / symmetry, never seeing any
           other bone's SDF.

(An earlier revision fitted ALL bones in a single GPU-batched Adam loop via
``batched_fit.BatchedFitWorker``; it converged poorly and was reverted to this
sequential flow.  ``batched_fit`` remains in the tree only for ``reflect_ellipsoids``.)

Symmetry:
    * paired (left/right) bones → only the *source* is fitted; the partner is
      derived by reflecting the source's fitted ellipsoids (no SDF, no fit);
    * on-plane (centre) bones → fitted with the single-bone optimizer's own
      symmetry enforcement enabled.

The controller is deliberately Qt-free: it talks to the application through a
small *host* object (duck-typed, see :class:`BoneSepHost`) so it can be unit
tested without a GUI.  The host drives the actual async SDF worker and the
per-bone fit worker and calls back into ``on_sdf_ready`` / ``on_sdf_failed`` /
``fit_progress`` / ``on_region_fit_finished``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from bone_symmetry import detect_mirror_plane, pair_bone_parts
from batched_fit import reflect_ellipsoids


class BoneSepHost(Protocol):
    """What the controller needs from the application (implemented by MainWindow).

    All methods are called on the GUI thread.  The async workers the host starts
    must call back into the controller (``on_sdf_ready`` / ``on_sdf_failed`` /
    ``fit_progress`` / ``on_region_fit_finished``) on the GUI thread too.
    """

    def compute_region_sdf(self, vertices: np.ndarray, faces: np.ndarray,
                           symmetry: bool) -> None: ...
    def run_region_fit(self, result, part, symmetry) -> None: ...
    def report_overall(self, frac: float, msg: str) -> None: ...
    def report_current(self, frac: float, msg: str) -> None: ...
    def set_status(self, msg: str) -> None: ...
    def bonesep_complete(self, accum: tuple) -> None: ...
    def bonesep_failed(self, msg: str) -> None: ...


class BoneSeparationController:
    """State machine for the two-phase parallel Bone-Separation pipeline."""

    def __init__(self, host: BoneSepHost, parts, base_kwargs: dict,
                 symmetry: bool, *, mesh_vertices: np.ndarray | None = None,
                 rng: np.random.Generator | None = None):
        self._host = host
        self._parts = list(parts)
        self._base = dict(base_kwargs)
        self._want_symmetry = bool(symmetry)
        # Full original mesh vertices for mirror-plane detection.  Detecting on the
        # whole mesh (not the stacked, overlap-duplicated per-bone submeshes) keeps
        # the plane identical to the old sequential pipeline.
        self._mesh_vertices = (
            None if mesh_vertices is None
            else np.asarray(mesh_vertices, np.float64).reshape(-1, 3))
        self._rng = rng or np.random.default_rng()

        # Symmetry classification (filled in ``begin``).
        self._plane: tuple[int, float] | None = None
        self._mirror_map: dict[int, int] = {}   # derived_part → source_part
        self._center: set[int] = set()          # on-plane part indices

        # Which part indices actually get an SDF + fit (sources + centre + lone).
        self._fit_indices: list[int] = []
        self._cursor = 0                         # progress through _fit_indices
        self._fitted: dict[int, tuple] = {}      # part_index → (c, r, q)
        self._done = False

    # ── public state ────────────────────────────────────────────────────
    @property
    def num_parts(self) -> int:
        return len(self._parts)

    @property
    def plane(self):
        return self._plane

    # ── phase 1: classify + start SDF precompute ────────────────────────
    def begin(self) -> None:
        """Classify bones by symmetry and kick off the first region SDF."""
        if not self._parts:
            self._host.bonesep_failed("no bone produced a usable region")
            return

        if self._want_symmetry:
            verts = (self._mesh_vertices if self._mesh_vertices is not None
                     else np.vstack([p.vertices for p in self._parts]))
            plane = detect_mirror_plane(verts)
            if plane is not None:
                mm, ctr = pair_bone_parts(self._parts, plane[0], plane[1])
                if mm or ctr:
                    self._plane = plane
                    self._mirror_map = mm
                    self._center = ctr

        # Fit-bones = everything that is NOT a derived mirror partner.
        derived = set(self._mirror_map.keys())
        self._fit_indices = [i for i in range(len(self._parts)) if i not in derived]

        if self._plane is not None:
            self._host.set_status(
                f"Bone Separation (parallel): {len(self._parts)} region(s), "
                f"{len(self._mirror_map)} mirrored / {len(self._center)} symmetric "
                f"across axis {'XYZ'[self._plane[0]]} — computing SDFs …")
        else:
            self._host.set_status(
                f"Bone Separation (parallel): {len(self._parts)} region(s) "
                f"— computing SDFs …")

        self._cursor = 0
        self._start_next_sdf()

    def _start_next_sdf(self) -> None:
        n = len(self._fit_indices)
        if self._cursor >= n:
            self._finalize()
            return
        # Sequential: each bone owns an equal slice of the overall bar.  The SDF
        # is the cheap head of the slice; the fit (driven by ``fit_progress``)
        # fills the rest.  One bone is fully fitted before the next bone's SDF is
        # computed (the GPU-batched all-bones loop was reverted).
        pidx = self._fit_indices[self._cursor]
        part = self._parts[pidx]
        self._host.report_overall(self._cursor / max(1, n),
                                  f"Gesamt · Bone {self._cursor + 1}/{n}")
        self._host.report_current(0.0,
                                 f"Region-SDF · Bone {part.bone_index}")
        sym = self._want_symmetry and (pidx in self._center)
        self._host.compute_region_sdf(part.vertices, part.faces, sym)

    # called by the host when the async SDF worker finishes
    def on_sdf_ready(self, result) -> None:
        """SDF for the current bone is ready → fit THIS bone on its own SDF."""
        if self._done:
            return
        pidx = self._fit_indices[self._cursor]
        part = self._parts[pidx]
        sym = (self._plane if (self._want_symmetry and pidx in self._center)
               else None)
        self._host.set_status(
            f"Bone Separation: fitting bone {part.bone_index} "
            f"({self._cursor + 1}/{len(self._fit_indices)}) …")
        self._host.run_region_fit(result, part, sym)

    def on_sdf_failed(self, msg: str) -> None:
        if self._done:
            return
        self._done = True
        self._host.bonesep_failed(f"region SDF failed: {msg}")

    # ── phase 2: sequential per-bone fit ────────────────────────────────
    def fit_progress(self, frac: float) -> None:
        """Host forwards the CURRENT bone's fit step fraction (0..1)."""
        n = len(self._fit_indices)
        frac = float(np.clip(frac, 0.0, 1.0))
        self._host.report_overall((self._cursor + frac) / max(1, n),
                                  f"Gesamt · Bone {self._cursor + 1}/{n}")
        self._host.report_current(frac, "Bone-Fit")

    # called by the host when the current bone's OptimizationWorker finishes
    def on_region_fit_finished(self, centers, radii, rots) -> None:
        """Store the current bone's fitted ellipsoids, then fit the next bone."""
        if self._done:
            return
        pidx = self._fit_indices[self._cursor]
        self._fitted[pidx] = (np.asarray(centers, np.float32).reshape(-1, 3),
                              np.asarray(radii, np.float32).reshape(-1, 3),
                              np.asarray(rots, np.float32).reshape(-1, 4))
        self._cursor += 1
        self._start_next_sdf()

    def _finalize(self) -> None:
        """All fit-bones done: reflect mirror partners and hand back the union."""
        if self._done:
            return
        self._done = True

        acc_c, acc_r, acc_q, acc_b = [], [], [], []
        for pidx, part in enumerate(self._parts):
            if pidx in self._fitted:
                c, r, q = self._fitted[pidx]
            elif pidx in self._mirror_map and self._plane is not None:
                src = self._mirror_map[pidx]
                if src not in self._fitted:
                    continue
                ax, co = self._plane
                c, r, q = reflect_ellipsoids(*self._fitted[src], ax, co)
            else:
                continue
            if not len(c):
                continue
            acc_c.append(c)
            acc_r.append(r)
            acc_q.append(q)
            acc_b.append(np.full(len(c), part.bone_index, dtype=np.int32))

        if not acc_c:
            self._host.bonesep_failed("Bone Separation produced no ellipsoids")
            return

        self._host.report_overall(1.0, "Gesamt · fertig")
        accum = (np.vstack(acc_c).astype(np.float32),
                 np.vstack(acc_r).astype(np.float32),
                 np.vstack(acc_q).astype(np.float32),
                 np.concatenate(acc_b).astype(np.int32))
        self._host.bonesep_complete(accum)

    # ── cancellation ────────────────────────────────────────────────────
    def cancel(self) -> None:
        self._done = True
