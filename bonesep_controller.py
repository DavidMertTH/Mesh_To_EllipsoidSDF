"""
bonesep_controller.py — orchestration for parallel (batched) Bone-Separation.

The old flow trained bones strictly one-after-another (SDF → fit, SDF → fit …),
which left the GPU idle most of the time because each small bone underfills it.
This controller restructures the flow into two clean phases:

    1. PRECOMPUTE  — compute every fit-bone's isolated region SDF (still one at a
       time, but SDF is the cheap phase) and turn each into a :class:`BoneFitJob`.
    2. BATCHED FIT — hand ALL jobs to :class:`batched_fit.BatchedFitWorker`, which
       fits every bone in a single Adam loop (each against its own SDF), so the
       GPU runs full and the slow phase happens once instead of N times.

Symmetry is exploited exactly as before, but cleanly separated here:
    * paired (left/right) bones → only the *source* is fitted; the partner is
      derived by reflecting the source's fitted ellipsoids (no SDF, no fit);
    * on-plane (centre) bones → fitted in the batch, then their result is
      symmetrised (handled inside the batched worker via ``BoneFitJob.symmetry``).

The controller is deliberately Qt-free: it talks to the application through a
small *host* object (duck-typed, see :class:`BoneSepHost`) so it can be unit
tested without a GUI.  The host drives the actual async SDF worker and the
batched-fit worker and calls back into ``on_sdf_ready`` / ``on_sdf_failed`` /
``on_fit_finished``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from bone_symmetry import detect_mirror_plane, pair_bone_parts
from batched_fit import BoneFitJob, make_bone_fit_job, reflect_ellipsoids


class BoneSepHost(Protocol):
    """What the controller needs from the application (implemented by MainWindow).

    All methods are called on the GUI thread.  The async workers the host starts
    must call back into the controller (``on_sdf_ready`` / ``on_sdf_failed`` /
    ``on_fit_finished``) on the GUI thread too.
    """

    def compute_region_sdf(self, vertices: np.ndarray, faces: np.ndarray,
                           symmetry: bool) -> None: ...
    def run_batched_fit(self, jobs: list[BoneFitJob]) -> None: ...
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
        self._jobs: list[BoneFitJob] = []        # parallel to _fit_indices
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
            self._launch_batched_fit()
            return
        # Precompute spans the first ~40 % of the overall bar; the fit the rest.
        self._host.report_overall(0.4 * self._cursor / max(1, n),
                                  f"Gesamt · SDF {self._cursor + 1}/{n}")
        pidx = self._fit_indices[self._cursor]
        part = self._parts[pidx]
        self._host.report_current(0.0,
                                 f"Region-SDF · Bone {part.bone_index}")
        sym = self._want_symmetry and (pidx in self._center)
        self._host.compute_region_sdf(part.vertices, part.faces, sym)

    # called by the host when the async SDF worker finishes
    def on_sdf_ready(self, result) -> None:
        if self._done:
            return
        pidx = self._fit_indices[self._cursor]
        part = self._parts[pidx]
        sym = (self._plane if (self._want_symmetry and pidx in self._center)
               else None)
        job = make_bone_fit_job(
            bone_index=part.bone_index,
            grid=result.grid, origin=result.origin, dx=result.dx,
            nx=result.nx, ny=result.ny, nz=result.nz,
            budget=int(part.max_budget),
            thickness=getattr(result, "thickness", None),
            symmetry=sym, rng=self._rng,
        )
        self._jobs.append(job)
        self._cursor += 1
        self._start_next_sdf()

    def on_sdf_failed(self, msg: str) -> None:
        if self._done:
            return
        self._done = True
        self._host.bonesep_failed(f"region SDF failed: {msg}")

    # ── phase 2: batched fit ────────────────────────────────────────────
    def _launch_batched_fit(self) -> None:
        if not self._jobs:
            self._done = True
            self._host.bonesep_failed("no fittable bone regions")
            return
        self._host.report_overall(0.4, "Gesamt · Fit (alle Bones)")
        self._host.report_current(0.0, "Batched-Fit · alle Bones")
        self._host.set_status(
            f"Bone Separation: fitting {len(self._jobs)} bone(s) in parallel …")
        self._host.run_batched_fit(self._jobs)

    def fit_progress(self, frac: float) -> None:
        """Host forwards the batched worker's step fraction (0..1)."""
        frac = float(np.clip(frac, 0.0, 1.0))
        self._host.report_overall(0.4 + 0.6 * frac, "Gesamt · Fit (alle Bones)")
        self._host.report_current(frac, "Batched-Fit · alle Bones")

    # called by the host when the batched worker finishes
    def on_fit_finished(self, results: list[tuple]) -> None:
        """``results`` = ordered ``[(bone_index, c, r, q), ...]`` from the worker.

        Worker order matches ``self._jobs`` order, which matches
        ``self._fit_indices`` order, so we can map each result back to its part.
        """
        if self._done:
            return
        self._done = True

        for k, (_, c, r, q) in enumerate(results):
            pidx = self._fit_indices[k]
            self._fitted[pidx] = (np.asarray(c, np.float32).reshape(-1, 3),
                                  np.asarray(r, np.float32).reshape(-1, 3),
                                  np.asarray(q, np.float32).reshape(-1, 4))

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
