"""Main-window orchestration of detail zones (see ``detail_zones``).

A full-object fit with detail zones runs like this:

1. ``plan_full_fit`` (before the fit) measures exact feature diameters,
   detects zones thinner than the global grid can resolve, pairs
   mirror-symmetric zones and splits the primitive budget between the main fit
   and the zones (area / diameter^1.25 each).
2. ``carve_global_target`` removes every zone core from the main fit's target
   (CSG "mesh minus core box").  The main fit therefore spends no primitives
   inside a zone and keeps its primitives out of it.
3. ``DetailZoneRun`` starts one process per zone (``detail_zone_worker.py``)
   at the same time as the main fit.  Each fits the zone's normalized crop on
   its own grid, sized to the zone's feature diameter, with the user's fit
   settings.  Separate processes are needed because Warp's autodiff tape is
   process-global; the GPU is shared.  Previews of all zones are merged into
   the live viewport.
4. When the main fit and all zones are done, the zone primitives (mirrored
   where paired) are added to the main fit's primitives, and the usual
   follow-ups (bone assignment, morph fitting) continue.  A zone whose process
   failed is refitted in this process before merging, so no region is ever
   left without primitives.

Stopping cancels the zones and keeps the main fit's result.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore

import detail_zones as dz
from ellipsoid import EllipsoidSet
from optimization import OptimizationWorker
from sdf_blowup import required_relative_sdf_margin, sparse_band_offsets
from sdf_compute import SdfComputer
from sdf_samples import SdfSampleSet

WORKER_SCRIPT = Path(__file__).resolve().with_name("detail_zone_worker.py")
SEAM_BAND_VOX = 3.0          # free band inside each core face (main-fit voxels)

SETTING_KEYS = (
    "detail_zones", "detail_zone_threshold_vox", "detail_zone_budget",
    "detail_zone_voxels_per_feature", "detail_zone_max_grid", "detail_zone_processes",
    "detail_zone_growth", "detail_zone_seam_vox",
)


# ── planning ─────────────────────────────────────────────────────────────────

def _json_value(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def zone_worker_kwargs(window) -> dict:
    """OptimizationWorker kwargs for a zone fit, from the live fit settings.

    Same mapping as the per-bone fits: the gathered GUI kwargs plus the
    advanced settings, without symmetry (each zone is fitted on its own and
    mirrored explicitly) and without the SDF arrays (added by the zone worker).
    """
    fit_kwargs = dict(window._gather_fit_kwargs())
    advanced = dict(fit_kwargs.pop("advanced", None) or {})
    fit_kwargs.pop("symmetry", None)
    spawn = fit_kwargs.pop("spawn_enabled", True)
    kwargs = dict(fit_kwargs)
    kwargs.update(spawn_underrep=bool(spawn), symmetry_enabled=False)
    if advanced:
        kwargs.update(window._optimizer_settings(advanced))
    kwargs = window._optimizer_settings(kwargs)
    return {k: _json_value(v) for k, v in kwargs.items()
            if isinstance(_json_value(v), (int, float, bool, str, type(None), list, dict))}


def plan_full_fit(window, num: int, max_count: int) -> tuple[int, int, object]:
    """Detect zones for the loaded mesh and split the budget.

    Returns ``(main_num, main_max, plan)``; ``plan`` is None when detail zones
    are disabled, unsupported for this fit, or not needed.
    """
    settings = getattr(window, "_settings", {}) or {}
    if not bool(settings.get("detail_zones", True)):
        return num, max_count, None
    if window._api_job_id is not None or getattr(window, "_api_batch_pipeline", False):
        return num, max_count, None
    result = window._last_mesh_result
    geometry = window._viewer.mesh_geometry()
    sdf = window._sdf
    if result is None or geometry is None or sdf is None or not sdf.is_ready:
        return num, max_count, None
    if abs(float(window._mesh_settings.blowup_fraction())) > 0.0:
        print("[DetailZones] skipped: SDF blowup is active")
        return num, max_count, None
    vertices = np.asarray(geometry[0], np.float32)
    faces = np.asarray(geometry[1], np.int64)
    sdf_vertices = getattr(sdf, "_verts", None)
    if (sdf_vertices is None or len(sdf_vertices) != len(vertices)
            or not np.allclose(np.asarray(sdf_vertices[:64]), vertices[:64], atol=1e-5)):
        print("[DetailZones] skipped: displayed mesh differs from the SDF mesh")
        return num, max_count, None
    query = lambda points: sdf.query_points(np.ascontiguousarray(points, np.float32))  # noqa: E731
    t0 = time.perf_counter()
    try:
        diameters = dz.exact_vertex_diameters(vertices, faces, query)
        zones = dz.detect_detail_zones(
            vertices, faces, diameters, float(result.dx),
            threshold_vox=float(settings.get("detail_zone_threshold_vox", 1.5)))
        if not zones:
            return num, max_count, None
        if window._effective_symmetry_enabled():
            dz.find_mirror_pairs(zones, vertices, faces, query, float(result.dx))
        plan = dz.allocate_detail_zones(
            zones, vertices, faces, diameters, float(result.dx), num, max_count,
            weight=float(settings.get("detail_zone_budget", 1.0)),
            growth=float(settings.get("detail_zone_growth") if settings.get("detail_zone_growth") is not None else 0.1),
            settings={key: settings.get(key) for key in SETTING_KEYS})
        plan.vertices = vertices
        plan.faces = faces
        plan.jobs = _zone_jobs(window, plan)
        plan.worker_kwargs = zone_worker_kwargs(window)
    except Exception as exc:          # never block a normal fit
        print(f"[DetailZones] planning failed, fitting without zones: {exc}")
        return num, max_count, None
    if not plan.jobs:
        return num, max_count, None
    print(f"[DetailZones] {len(zones)} zone(s), {len(plan.jobs)} to fit "
          f"({time.perf_counter() - t0:.2f} s): budgets {[z.reserve for z in zones]}, "
          f"main fit {plan.global_num}/{plan.global_max}")
    return plan.global_num, plan.global_max, plan


def _zone_jobs(window, plan) -> list:
    settings = plan.settings
    jobs = []
    for zone in plan.sources:
        sub_v, sub_f = dz.zone_submesh(plan.vertices, plan.faces, zone)
        if len(sub_f) < 8:
            continue
        center, scale = dz.normalization(sub_v)
        norm_v = ((sub_v - center) * np.float32(scale)).astype(np.float32)
        n = dz.zone_grid_resolution(
            norm_v, zone.median_diameter * scale,
            voxels_per_feature=float(settings.get("detail_zone_voxels_per_feature") or 8.0),
            n_min=int(window._mesh_sdf_panel.requested_n),
            n_max=int(settings.get("detail_zone_max_grid") or 320))
        partners = [z for z in plan.zones if z.mirror_of == zone.index]
        budget = max(8, int(zone.reserve))
        jobs.append(dict(zone=zone, partners=partners, center=center, scale=scale, n=n,
                         num=max(4, int(round(0.67 * budget))), max=budget,
                         vertices=norm_v, faces=sub_f.astype(np.int32)))
    # Zones whose crop is empty fall back to the main fit: do not carve them.
    fitted = {job["zone"].index for job in jobs}
    for job in jobs:
        fitted.update(p.index for p in job["partners"])
    plan.zones = [z for z in plan.zones if z.index in fitted]
    return jobs


def carve_global_target(plan, grid: np.ndarray, origin, dx: float, samples):
    """Main-fit target with every zone core removed (copies; caches untouched)."""
    zones = plan.zones
    carved_grid = dz.carve_grid(grid, origin, float(dx), zones)
    if samples is None:
        return carved_grid, None
    # Seam band: samples just inside a core face are dropped, so the main fit's
    # primitives can run past the seam and overlap the zone's primitives.
    band = float(plan.settings.get("detail_zone_seam_vox") or SEAM_BAND_VOX) * float(dx)
    drop = dz.seam_band_mask(samples.points, samples.values, zones, band,
                             vertices=getattr(plan, "vertices", None))
    keep = ~drop
    values, changed = dz.carve_values(samples.points[keep], samples.values[keep], zones)
    normals = None
    if samples.normals is not None:
        normals = np.array(samples.normals[keep], dtype=np.float32, copy=True)
        normals[changed] = 0.0        # carved points have no mesh normal

    def subset(array):
        return None if array is None else np.asarray(array)[keep]
    carved_samples = SdfSampleSet(
        samples.points[keep], values, subset(samples.thickness), dx=samples.dx,
        source=f"{samples.source}+zones-carved", coarse_mask=subset(samples.coarse_mask),
        normals=normals, thickness_sampling_power=samples.thickness_sampling_power)
    print(f"[DetailZones] main fit ignores {len(zones)} zone core(s): "
          f"{int(changed.sum()):,} training samples re-targeted, "
          f"{int(drop.sum()):,} left free at the seams")
    return carved_grid, carved_samples


# ── zone process ─────────────────────────────────────────────────────────────

class ZoneProcess(QtCore.QObject):
    """One zone fitted by ``detail_zone_worker.py`` in a separate process."""

    frame = QtCore.Signal()
    done = QtCore.Signal()

    def __init__(self, job: dict, window, plan, parent=None):
        super().__init__(parent)
        self.job = job
        self.latest = None            # normalized params of the newest frame
        self.result = None
        self.error: str | None = None
        self.step = 0
        self.total = 0
        self.finished = False
        self._buffer = ""
        self._dir = tempfile.mkdtemp(prefix="detail_zone_")
        settings = window._settings
        margin = required_relative_sdf_margin(window._slider_margin.value() / 100.0, 0.0, int(job["n"]))
        kwargs = dict(plan.worker_kwargs)
        kwargs.update(num_ellipsoids=int(job["num"]), max_ellipsoids=int(job["max"]))
        payload = dict(
            n=int(job["n"]), margin=float(margin), device=str(window._device),
            thickness_max_resolution=int(settings.get("thickness_max_resolution", 128)),
            use_sparse_sdf=bool(settings.get("use_sparse_sdf", True)),
            thickness_sampling_power=float(settings.get("thickness_sampling_power", 1.0)),
            seed=os.environ.get("DETAIL_ZONE_SEED"), kwargs=kwargs,
            median_diameter=float(job["zone"].median_diameter * job["scale"]))
        self._job_path = os.path.join(self._dir, "job.npz")
        np.savez(self._job_path, vertices=job["vertices"], faces=job["faces"],
                 job=np.array(json.dumps(payload)))
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._process.setWorkingDirectory(str(WORKER_SCRIPT.parent))
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

    def start(self) -> None:
        self._process.start(sys.executable, ["-u", str(WORKER_SCRIPT), self._job_path])

    def kill(self) -> None:
        if self._process.state() != QtCore.QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)
        self._cleanup()

    def _cleanup(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    @QtCore.Slot()
    def _on_output(self) -> None:
        self._buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buffer = self._buffer.split("\n")
        updated = False
        for line in lines:
            line = line.strip()
            if not line.startswith(dz.MESSAGE_PREFIX):
                if "Error" in line or "Traceback" in line:
                    print(f"[DetailZone {self.job['zone'].index}] {line}")
                continue
            try:
                message = json.loads(line[len(dz.MESSAGE_PREFIX):])
            except ValueError:
                continue
            kind = message.get("type")
            if kind == "stage" and message.get("stage") == "fit":
                self.total = int(message.get("total", 0))
            elif kind == "frame":
                self.latest = dz.unpack_params(message)
                self.step = int(message.get("step", 0))
                self.total = int(message.get("total", self.total))
                updated = True
            elif kind == "result":
                self.result = dz.unpack_params(message)
                self.latest = self.result
                updated = True
            elif kind == "check":
                retry = " after refit with more primitives" if message.get("retry") else ""
                print(f"[DetailZones] zone {self.job['zone'].index}: "
                      f"{100.0 * float(message.get('uncovered', 0.0)):.1f} % of the surface "
                      f"uncovered with {message.get('primitives')} primitives{retry}")
            elif kind == "error":
                self.error = str(message.get("message"))
        if updated:
            self.frame.emit()

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _on_finished(self, code, status) -> None:
        self._on_output()
        if self.result is None and self.error is None:
            self.error = f"zone process exited with code {code}"
        self.finished = True
        self._cleanup()
        self.done.emit()

    @QtCore.Slot(QtCore.QProcess.ProcessError)
    def _on_error(self, error) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart and not self.finished:
            self.error = "zone process failed to start"
            self.finished = True
            self._cleanup()
            self.done.emit()


# ── in-process fallback (one zone at a time, after the main fit) ─────────────

class InProcessZoneFit(QtCore.QObject):
    """Fits one zone job in this process (used when its process failed)."""

    done = QtCore.Signal()

    def __init__(self, job: dict, window, plan, parent=None):
        super().__init__(parent)
        self.job = job
        self.result = None
        self.error = None
        self._w = window
        self._plan = plan
        self._computer = SdfComputer(device=window._device)
        self._worker = None
        self._last = None
        self._stopped = False

    def start(self) -> None:
        job, w = self.job, self._w
        self._computer.set_mesh(job["vertices"], job["faces"])
        margin = required_relative_sdf_margin(w._slider_margin.value() / 100.0, 0.0, int(job["n"]))
        result = self._computer.compute_voxel_grid(
            n=int(job["n"]), margin=margin, compute_thickness=True,
            thickness_max_resolution=int(w._settings.get("thickness_max_resolution", 128)))
        samples = None
        if bool(w._settings.get("use_sparse_sdf", True)):
            samples = self._computer.compute_sparse_samples(
                n=int(result.n), margin=margin, thickness_result=result,
                offsets_vox=sparse_band_offsets(0.0),
                thickness_sampling_power=float(w._settings.get("thickness_sampling_power", 1.0)))
        kwargs = dict(self._plan.worker_kwargs)
        kwargs.update(num_ellipsoids=int(job["num"]), max_ellipsoids=int(job["max"]),
                      sdf_target_np=np.asarray(result.grid, dtype=np.float32), sdf_samples=samples,
                      origin=result.origin, dx=result.dx, n=result.n,
                      thickness_np=getattr(result, "thickness", None), sdf_blowup_fraction=0.0,
                      sdf_computer=self._computer, parent=self)
        self._worker = OptimizationWorker(**kwargs)
        self._worker.step_visual.connect(self._on_step)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def stop(self) -> None:
        self._stopped = True
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait()

    @QtCore.Slot(int, float, object, object, object, object)
    def _on_step(self, step, loss, c, r, q, shape) -> None:
        self._last = (np.asarray(c, np.float32).copy(), np.asarray(r, np.float32).copy(),
                      np.asarray(q, np.float32).copy(),
                      None if shape is None else np.asarray(shape, np.float32).copy())

    @QtCore.Slot()
    def _on_finished(self) -> None:
        if self._worker is not None:
            self._worker.wait()
        if self._stopped:
            return
        error = getattr(self._worker, "error_message", None)
        if error or self._last is None:
            self.error = str(error or "no primitives")
        else:
            self.result = self._last
        self.done.emit()


# ── controller ───────────────────────────────────────────────────────────────

class DetailZoneRun(QtCore.QObject):
    """Runs all zone fits next to the main fit and merges the result."""

    finished = QtCore.Signal(bool)      # True when zone primitives were merged

    def __init__(self, window, plan):
        super().__init__(window)
        self._w = window
        self._plan = plan
        self._processes: list[ZoneProcess] = []
        self._fallback: InProcessZoneFit | None = None
        self._fallback_queue: list = []
        self._global = None
        self._global_done = False
        self._active = False
        self._last_render = 0.0
        self._started = 0.0
        self._fallback_results: list = []      # (job, normalized params)
        self._failed_zones: list = []
        self._fallback_scheduled: set = set()  # zone indices

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._active = True
        self._started = time.perf_counter()
        allow = self._plan.settings.get("detail_zone_processes")
        allow_processes = True if allow is None else bool(allow)
        for job in self._plan.jobs:
            if allow_processes:
                process = ZoneProcess(job, self._w, self._plan, parent=self)
                process.frame.connect(self._on_zone_frame)
                process.done.connect(self._on_zone_done)
                self._processes.append(process)
                process.start()
            else:
                self._fallback_queue.append(job)
        self._w._viewer.show_region_boxes([(z.core_lo, z.core_hi) for z in self._plan.zones])

    def cancel(self) -> None:
        self._active = False
        for process in self._processes:
            process.kill()
        if self._fallback is not None:
            self._fallback.stop()
        self._fallback = None
        self._fallback_queue = []
        self._w._viewer.clear_region_box()

    # ── live preview ────────────────────────────────────────────────────
    def _entries(self, final: bool):
        entries = []
        sources = [(p.job, p.result if final else p.latest) for p in self._processes]
        sources += list(self._fallback_results)
        for job, params in sources:
            if params is None:
                continue
            world = dz.from_normalized(params, job["center"], job["scale"])
            entries.append(world)
            for partner in job["partners"]:
                entries.append(dz.mirror_params(world, partner.mirror_axis, partner.mirror_plane))
        return entries

    def preview(self, global_params):
        """Main-fit frame plus the newest frame of every zone (for display)."""
        if not self._active:
            return global_params
        keep = ~dz.inside_any_core(global_params[0], self._plan.zones)
        return dz.merge_zone_results(global_params, keep, self._entries(final=False))

    @QtCore.Slot()
    def _on_zone_frame(self) -> None:
        if not self._active:
            return
        steps = [p.step / max(1, p.total) for p in self._processes if p.total]
        if self._global_done:
            now = time.perf_counter()
            if now - self._last_render >= 0.1 and self._global is not None:
                self._last_render = now
                self._render(self.preview(self._global))
            if steps:
                self._w._progress_set(100.0 * float(np.mean(steps)),
                                      f"Detail zones · {100 * float(np.mean(steps)):.0f} %")

    # ── completion ──────────────────────────────────────────────────────
    def on_global_finished(self) -> None:
        """The main fit is done; merge as soon as every zone is."""
        w = self._w
        ellipsoids = w._ellipsoids
        if ellipsoids is None or not self._active:
            self.finished.emit(False)
            return
        shape = getattr(w, "_last_visual_shape", None)
        c = np.asarray(ellipsoids.centers, np.float32)
        if shape is not None and len(shape) != len(c):
            shape = None
        self._global = (c, np.asarray(ellipsoids.radii, np.float32),
                        np.asarray(ellipsoids.rotations, np.float32),
                        None if shape is None else np.asarray(shape, np.float32))
        self._global_done = True
        w._btn_fit.setEnabled(False)
        w._btn_stop.setEnabled(True)
        if not self._try_finish():
            running = sum(not p.finished for p in self._processes)
            w._status.showMessage(f"Main fit done · waiting for {running} detail zone(s) …")
            self._render(self.preview(self._global))

    @QtCore.Slot()
    def _on_zone_done(self) -> None:
        if not self._active:
            return
        for process in self._processes:
            index = process.job["zone"].index
            if (process.finished and process.result is None
                    and index not in self._fallback_scheduled):
                print(f"[DetailZones] zone {index} process failed "
                      f"({process.error}); refitting it in the main process")
                self._fallback_scheduled.add(index)
                self._fallback_queue.append(process.job)
        if self._global_done:
            self._try_finish()

    def _try_finish(self) -> bool:
        if not self._global_done or any(not p.finished for p in self._processes):
            return False
        if self._fallback is not None:
            return False
        if self._fallback_queue:
            self._start_fallback()
            return False
        self._finish()
        return True

    def _start_fallback(self) -> None:
        job = self._fallback_queue.pop(0)
        self._w._status.showMessage(f"Detail zone {job['zone'].index}: fitting in the main process …")
        fit = InProcessZoneFit(job, self._w, self._plan, parent=self)
        fit.done.connect(self._on_fallback_done)
        self._fallback = fit
        try:
            fit.start()
        except Exception as exc:
            print(f"[DetailZones] zone {job['zone'].index} could not be fitted: {exc}")
            self._fallback = None
            self._failed_zones.append(job)
            self._try_finish()

    @QtCore.Slot()
    def _on_fallback_done(self) -> None:
        fit = self._fallback
        self._fallback = None
        if not self._active or fit is None:
            return
        if fit.result is not None:
            self._fallback_results.append((fit.job, fit.result))
        else:
            print(f"[DetailZones] zone {fit.job['zone'].index} failed again: {fit.error}")
            self._failed_zones.append(fit.job)
        self._try_finish()

    def _finish(self) -> None:
        w = self._w
        failed = {job["zone"].index for job in self._failed_zones}
        for job in self._failed_zones:
            failed.update(p.index for p in job["partners"])
        if failed:
            print(f"[DetailZones] zones {sorted(failed)} have no zone fit; the main fit "
                  "did not cover them — refit without detail zones if this persists")
        entries = self._entries(final=True)
        keep = ~dz.inside_any_core(self._global[0], self._plan.zones)
        merged = dz.merge_zone_results(self._global, keep, entries)
        self._apply(merged)
        self._active = False
        w._viewer.clear_region_box()
        w._progress_end()
        zones = len(self._plan.zones) - len(failed)
        w._status.showMessage(
            f"Fit finished with {zones} detail zone(s) in parallel "
            f"({time.perf_counter() - self._started:.0f} s) · {len(merged[0])} primitives")
        self.finished.emit(bool(entries))

    # ── output ──────────────────────────────────────────────────────────
    def _render(self, params) -> None:
        w = self._w
        c, r, q, shape = params
        w._shape.render(w._viewer, c, r, q, shape)
        w._lbl_ell_count.setText(f"Count: {len(c)}")

    def _apply(self, params) -> None:
        w = self._w
        c, r, q, shape = params
        self._render(params)
        ellipsoids = EllipsoidSet(device=w._device)
        ellipsoids.set_parameters(c, r, q)
        w._ellipsoids = ellipsoids
        w._last_visual_shape = None if shape is None else np.asarray(shape, np.float32).copy()
        kind = getattr(w._shape, "id", "ellipsoid")
        if kind in ("ellipsoid", "superquadric"):
            w._fitted_primitive_type = kind
            try:
                w._fitted_shape_exponents = w._api_shape_exponents_array(
                    None if shape is None else shape[:, :2], len(c),
                    primitive_type=kind, allow_missing_ellipsoid=True)
            except ValueError:
                w._fitted_shape_exponents = None
