"""
api_server.py — Embedded HTTP API for driving EllipSDF from another process.

Purpose
-------
Lets an external client (e.g. a Unity ``MonoBehaviour``) push a triangle mesh
into a *running* EllipSDF instance, watch the fit happen live in the normal
viewport, and pull the fitted primitives back out.  The same server runs whether
EllipSDF was started manually or launched on demand by the client, so "attach to
running app" and "start it for me" share one code path.

Architecture
------------
The HTTP server runs in a daemon thread (``http.server``), so request handlers do
*not* run on the Qt GUI thread.  Starting a fit must happen on the GUI thread
(the optimizer is a ``QThread`` owned by ``MainWindow`` and renders into the
viewport), so a request handler never touches Qt directly: it stores the job in a
thread-safe bounded :class:`JobRegistry`.  The registry dispatches exactly one
ready job at a time through :class:`ApiBridge.fit_requested`; terminal completion
atomically releases that slot and dispatches the next queued job.  The signal is
delivered via a *queued* connection to ``MainWindow`` on the GUI thread, which
then drives the existing load → compute-SDF → fit pipeline and writes
progress/result back into the job.  Pollers read job state under a lock.

Endpoints
---------
``GET  /ping``                 → ``{status, app, version, busy}``
``POST /fit``                  → ``{job_id}``  (202; 429 if the queue is full)
``POST /fit-pose``             → ``{job_id}``  (fit posted primitive IDs to mesh)
``POST /fit/<id>/cancel``      → request cooperative cancellation
``GET  /fit/<id>/status``      → ``{state, step, total, loss, count, error,
                                      preview?}``
``GET  /fit/<id>/result``      → result JSON (200) / not-ready or canceled
                                      (409) / error (500)

Request body for ``POST /fit`` (JSON)::

    {
      "vertices": [[x,y,z], ...],          # required
      "faces":    [[a,b,c], ...],          # required
      "primitive_type": "ellipsoid",       # ellipsoid | superquadric
      "options":  { "num_ellipsoids": 20, "max_ellipsoids": 60,
                    "num_steps": 2000, "symmetry": false,
                    "primitive_shape": "ellipsoid",
                    "sq_eps1": 1.0, "sq_eps2": 1.0,
                    "sq_eps_mode": "per_primitive",
                    "sq_unlock_frac": 0.05, ... },                  # optional
      "rig":      { "bones": [...], "boneWeights": [...], ... }     # optional
    }

``options.num_steps`` is an optional positive integer for this one fit request
(``numSteps`` is also accepted); if omitted, the running app's Steps control is
used.  The value is independent for each queued ``/fit`` or ``/fit-pose`` job.
For ``/fit-pose``, optional boolean ``options.pose_fit_position``,
``options.pose_fit_rotation``, and ``options.pose_fit_scale`` override the
corresponding persistent EllipSDF Settings for that request.  A false value
freezes that primitive transform channel; ``/fit`` base fitting is unaffected.

Protocol v4 results include ``primitive_type`` both at the top level and on
every entry, plus per-entry ``shape_exponents: [e1, e2]``.  Missing legacy
fields mean an ellipsoid with exponents ``[1, 1]``.  ``eps`` is accepted as an
internal input alias for ``shape_exponents``.  Bent superquadrics are outside
the Unity bridge contract.

Clients that need to cancel while the request body is still uploading may send
``X-EllipSDF-Job-Id`` with a lowercase 32-hex UUID (Unity
``Guid.NewGuid().ToString("N")``).  The API reserves that ID atomically before
reading the body, so ``POST /fit/<id>/cancel`` can already address the job.

Coordinate convention is caller-defined: vertices, rig transforms, and returned
primitives stay in the same snapshot space.  The Unity bridge uses Unity world
space so transformed and skinned meshes line up with the scene view.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from PySide6 import QtCore

API_VERSION = 4
DEFAULT_PORT = 8765
DEFAULT_QUEUE_LIMIT = 8
DEFAULT_TERMINAL_JOB_TTL_SECONDS = 15.0 * 60.0
DEFAULT_TERMINAL_JOB_LIMIT = 256
DEFAULT_RESERVATION_TTL_SECONDS = 60.0
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 30.0
CLIENT_JOB_ID_HEADER = "X-EllipSDF-Job-Id"

_ACTIVE_STATES = frozenset(("queued", "running", "canceling"))
_TERMINAL_STATES = frozenset(("canceled", "done", "error"))


# ── Job state ───────────────────────────────────────────────────────────────

@dataclass
class FitJob:
    """One fit request and its evolving state.

    Inputs are filled at creation (HTTP thread); ``state``/progress/``result``
    are mutated by ``MainWindow`` on the GUI thread.  All access goes through
    :class:`JobRegistry`, which serialises it with a lock.
    """

    id: str
    payload: dict[str, Any] | None          # None while a client ID is reserved/uploading
    state: str = "queued"                   # queued | running | canceling | canceled | done | error
    step: int = 0
    total: int = 0
    loss: float = 0.0
    count: int = 0
    stage: str = "queued"
    detail: str = "Waiting in fit queue"
    stage_progress: float = 0.0
    stage_started_at: float = field(default_factory=lambda: time.monotonic())
    result: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None
    error: str | None = None
    suppress_preview: bool = False
    terminal_at: float | None = None
    created_at: float = field(default_factory=lambda: time.monotonic())

    def status_dict(self) -> dict[str, Any]:
        status = {
            "job_id": self.id,
            "state": self.state,
            "step": self.step,
            "total": self.total,
            "loss": self.loss,
            "count": self.count,
            "stage": self.stage,
            "detail": self.detail,
            "stage_progress": self.stage_progress,
            "stage_seconds": max(0.0, time.monotonic() - self.stage_started_at),
            "error": self.error,
        }
        if self.preview is not None and not self.suppress_preview:
            status["preview"] = self.preview
        return status


class JobRegistry:
    """Thread-safe store for :class:`FitJob` objects.

    Mutations from the GUI thread and reads from HTTP handler threads are
    serialised with a single lock.  Callers must go through ``update`` rather
    than mutating a job in place so every change is published under the lock.
    """

    def __init__(
        self,
        queue_limit: int = DEFAULT_QUEUE_LIMIT,
        dispatch_callback: Callable[[str], None] | None = None,
        terminal_ttl_seconds: float = DEFAULT_TERMINAL_JOB_TTL_SECONDS,
        terminal_job_limit: int = DEFAULT_TERMINAL_JOB_LIMIT,
        reservation_ttl_seconds: float = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> None:
        queue_limit = int(queue_limit)
        if queue_limit < 1:
            raise ValueError("queue_limit must be at least 1")
        terminal_ttl_seconds = float(terminal_ttl_seconds)
        terminal_job_limit = int(terminal_job_limit)
        reservation_ttl_seconds = float(reservation_ttl_seconds)
        if terminal_ttl_seconds < 0.0:
            raise ValueError("terminal_ttl_seconds must be non-negative")
        if terminal_job_limit < 1:
            raise ValueError("terminal_job_limit must be at least 1")
        if reservation_ttl_seconds <= 0.0:
            raise ValueError("reservation_ttl_seconds must be positive")
        self._lock = threading.Lock()
        self._jobs: dict[str, FitJob] = {}
        # ``queue_limit`` bounds all admitted non-terminal jobs, including the
        # one currently owned by MainWindow and reservations still uploading.
        self._queue_limit = queue_limit
        self._ready_job_ids: deque[str] = deque()
        self._dispatched_job_id: str | None = None
        self._dispatch_callback = dispatch_callback
        # Terminal records remain queryable for result/status polling, but no
        # longer accumulate forever.  Access refreshes their LRU position; all
        # expiry/eviction happens under the same lock as status/result reads.
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._terminal_job_limit = terminal_job_limit
        self._terminal_lru: OrderedDict[str, float] = OrderedDict()
        self._reservation_ttl_seconds = reservation_ttl_seconds

    @property
    def queue_limit(self) -> int:
        return self._queue_limit

    def set_dispatch_callback(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        """Install the single-consumer callback and dispatch pending work."""
        with self._lock:
            self._dispatch_callback = callback
            dispatch_id = self._claim_next_dispatch_locked()
        self._emit_dispatch(dispatch_id)

    def _active_count_locked(self) -> int:
        return sum(job.state in _ACTIVE_STATES for job in self._jobs.values())

    def _purge_terminal_locked(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        stale_reservations = [
            job_id for job_id, job in self._jobs.items()
            if (job.state == "queued" and job.payload is None
                and now - job.created_at >= self._reservation_ttl_seconds)
        ]
        for job_id in stale_reservations:
            self._remove_ready_locked(job_id)
            del self._jobs[job_id]
        if self._terminal_ttl_seconds == 0.0:
            expired = list(self._terminal_lru)
        else:
            expired = [
                job_id for job_id, terminal_at in self._terminal_lru.items()
                if now - terminal_at >= self._terminal_ttl_seconds
            ]
        for job_id in expired:
            self._terminal_lru.pop(job_id, None)
            job = self._jobs.get(job_id)
            if job is not None and job.state in _TERMINAL_STATES:
                del self._jobs[job_id]
        while len(self._terminal_lru) > self._terminal_job_limit:
            job_id, _ = self._terminal_lru.popitem(last=False)
            job = self._jobs.get(job_id)
            if job is not None and job.state in _TERMINAL_STATES:
                del self._jobs[job_id]

    def _touch_terminal_locked(self, job: FitJob) -> None:
        if job.state not in _TERMINAL_STATES:
            return
        terminal_at = (
            time.monotonic() if job.terminal_at is None else job.terminal_at)
        self._terminal_lru.pop(job.id, None)
        self._terminal_lru[job.id] = terminal_at

    def _queue_position_locked(self, job_id: str) -> int | None:
        if self._dispatched_job_id == job_id:
            return 0
        position = 1
        for queued_id in self._ready_job_ids:
            queued = self._jobs.get(queued_id)
            if (queued is None or queued.state != "queued"
                    or queued.payload is None):
                continue
            if queued_id == job_id:
                return position
            position += 1
        return None

    def _status_dict_locked(self, job: FitJob) -> dict[str, Any]:
        status = job.status_dict()
        if job.state == "queued":
            position = self._queue_position_locked(job.id)
            if position is not None:
                status["queue_position"] = position
        return status

    def queue_status(self) -> dict[str, int]:
        """Return a compact, lock-consistent capacity snapshot."""
        with self._lock:
            self._purge_terminal_locked()
            active = self._active_count_locked()
            waiting = sum(
                1 for job_id in self._ready_job_ids
                if (job_id in self._jobs
                    and self._jobs[job_id].state == "queued"
                    and self._jobs[job_id].payload is not None)
            )
            return {
                "queue_limit": self._queue_limit,
                "active_jobs": active,
                "waiting_jobs": waiting,
            }

    def _claim_next_dispatch_locked(self) -> str | None:
        if (self._dispatch_callback is None
                or self._dispatched_job_id is not None):
            return None
        while self._ready_job_ids:
            job_id = self._ready_job_ids.popleft()
            job = self._jobs.get(job_id)
            if (job is None or job.state != "queued" or job.payload is None):
                continue
            self._dispatched_job_id = job_id
            return job_id
        return None

    def _emit_dispatch(self, job_id: str | None) -> None:
        if job_id is None:
            return
        callback = self._dispatch_callback
        if callback is not None:
            callback(job_id)

    def _remove_ready_locked(self, job_id: str) -> None:
        try:
            self._ready_job_ids.remove(job_id)
        except ValueError:
            pass

    def _finish_job_locked(self, job: FitJob) -> str | None:
        # Request payloads can contain millions of JSON number objects.  Once a
        # job is terminal only status/result are needed, so release the input.
        job.payload = None
        self._remove_ready_locked(job.id)
        if self._dispatched_job_id == job.id:
            self._dispatched_job_id = None
        job.terminal_at = time.monotonic()
        self._touch_terminal_locked(job)
        self._purge_terminal_locked(job.terminal_at)
        return self._claim_next_dispatch_locked()

    def reserve(
        self,
        job_id: str | None = None,
    ) -> tuple[str, FitJob | None]:
        """Reserve one bounded queue slot before reading an HTTP body.

        Returns ``("reserved", job)``, ``("duplicate", None)`` or
        ``("full", None)``.  Reservations count against the bound so slow or
        malicious uploads cannot grow memory/state without limit.
        """
        with self._lock:
            self._purge_terminal_locked()
            if job_id is not None and job_id in self._jobs:
                return "duplicate", None
            if self._active_count_locked() >= self._queue_limit:
                return "full", None
            reserved_id = job_id or uuid.uuid4().hex
            job = FitJob(id=reserved_id, payload=None)
            self._jobs[job.id] = job
            return "reserved", job

    def enqueue(
        self,
        payload: dict[str, Any],
        job_id: str | None = None,
    ) -> tuple[str, FitJob | None]:
        """Reserve and attach an in-memory job using the normal FIFO path."""
        outcome, job = self.reserve(job_id)
        if outcome != "reserved" or job is None:
            return outcome, None
        attach_outcome, _ = self.attach_payload(job.id, payload)
        if attach_outcome != "ready":
            return attach_outcome, None
        return "queued", job

    def add(self, payload: dict[str, Any]) -> FitJob:
        """Compatibility helper; prefer :meth:`enqueue` for new callers."""
        outcome, job = self.enqueue(payload)
        if job is None:
            raise RuntimeError(f"could not add fit job: {outcome}")
        return job

    def add_if_idle(self, payload: dict[str, Any]) -> FitJob | None:
        """Legacy strict-idle helper retained for external callers/tests."""
        with self._lock:
            self._purge_terminal_locked()
            if self._active_count_locked() != 0:
                return None
            job = FitJob(id=uuid.uuid4().hex, payload=payload)
            self._jobs[job.id] = job
            self._ready_job_ids.append(job.id)
            dispatch_id = self._claim_next_dispatch_locked()
        self._emit_dispatch(dispatch_id)
        return job

    def reserve_if_idle(self, job_id: str | None = None) -> FitJob | None:
        """Legacy strict-idle reservation helper.

        Explicit client IDs are not reused while their terminal record remains
        inside the configured TTL/LRU retention window.
        """
        with self._lock:
            self._purge_terminal_locked()
            if job_id is not None and job_id in self._jobs:
                return None
            if self._active_count_locked() != 0:
                return None
            reserved_id = job_id or uuid.uuid4().hex
            job = FitJob(id=reserved_id, payload=None)
            self._jobs[job.id] = job
            return job

    def attach_payload(
        self,
        job_id: str,
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        """Attach a parsed body unless cancellation won the upload race."""
        invalid_message = None
        options: dict[str, Any] = {}
        if not isinstance(payload, dict):
            invalid_message = "request body must be a JSON object"
        else:
            raw_options = payload.get("options")
            if raw_options is not None and not isinstance(raw_options, dict):
                invalid_message = "'options' must be a JSON object"
            elif raw_options is not None:
                options = raw_options
        dispatch_id = None
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return "missing", None
            if job.state == "queued" and job.payload is None:
                if invalid_message is not None:
                    # Validation precedes mutation.  Remove the still-empty
                    # reservation atomically so malformed options cannot strand
                    # a capacity-consuming job outside the ready FIFO.
                    self._remove_ready_locked(job_id)
                    del self._jobs[job_id]
                    return "invalid", {"error": invalid_message}
                job.payload = payload
                job.suppress_preview = bool(
                    options.get("batch_pipeline", False)
                    or options.get("batchPipeline", False)
                    or payload.get("batch_pipeline", False)
                    or payload.get("batchPipeline", False)
                )
                self._ready_job_ids.append(job.id)
                dispatch_id = self._claim_next_dispatch_locked()
                status = self._status_dict_locked(job)
            elif job.state in ("canceling", "canceled"):
                return job.state, self._status_dict_locked(job)
            else:
                return "unavailable", self._status_dict_locked(job)
        self._emit_dispatch(dispatch_id)
        return "ready", status

    def discard_reservation(self, job_id: str) -> bool:
        """Remove a still-empty queued reservation after a bad HTTP body."""
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if (job is None or job.state != "queued"
                    or job.payload is not None):
                return False
            self._remove_ready_locked(job_id)
            del self._jobs[job_id]
            return True

    def get(self, job_id: str) -> FitJob | None:
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if job is not None:
                self._touch_terminal_locked(job)
            return job

    def update(self, job_id: str, **fields: Any) -> bool:
        """Publish a normal job update.

        Cancellation owns the job once it reaches ``canceling``.  Late worker
        progress or completion callbacks must not be able to resurrect it as a
        running/done job; only :meth:`complete_cancel` may leave that state.
        Terminal states are immutable for the same reason.
        """
        dispatch_id = None
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state in ("canceling", "canceled", "done", "error"):
                return False
            requested_state = fields.get("state")
            if (requested_state == "running"
                    and self._dispatch_callback is not None
                    and self._dispatched_job_id != job_id):
                return False
            if "stage" in fields and fields["stage"] != job.stage:
                job.stage_started_at = time.monotonic()
            for k, v in fields.items():
                if k == "preview" and job.suppress_preview:
                    continue
                setattr(job, k, v)
            if job.state in _TERMINAL_STATES:
                dispatch_id = self._finish_job_locked(job)
        self._emit_dispatch(dispatch_id)
        return True

    def request_cancel(self, job_id: str) -> tuple[str, dict[str, Any] | None]:
        """Atomically request cancellation.

        Returns ``(outcome, status)``.  ``accepted`` is the only outcome that
        should emit the GUI bridge signal.  Repeated requests are no-ops and
        report ``canceling``/``canceled``; completed jobs report ``finished``.
        """
        dispatch_id = None
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return "missing", None
            if (job.state == "queued"
                    and self._dispatched_job_id != job_id):
                # A waiting/reserved job owns no GUI or GPU resources and can
                # terminate synchronously without involving MainWindow.
                job.state = "canceled"
                job.error = "canceled by client"
                job.result = None
                dispatch_id = self._finish_job_locked(job)
                outcome = "canceled"
                status = self._status_dict_locked(job)
            elif job.state in ("queued", "running"):
                job.state = "canceling"
                job.error = None
                outcome = "accepted"
                status = self._status_dict_locked(job)
            elif job.state == "canceling":
                return "canceling", self._status_dict_locked(job)
            elif job.state == "canceled":
                return "canceled", self._status_dict_locked(job)
            else:
                return "finished", self._status_dict_locked(job)
        self._emit_dispatch(dispatch_id)
        return outcome, status

    def complete_cancel(
        self,
        job_id: str,
        reason: str = "canceled by client",
    ) -> bool:
        """Mark a requested cancellation complete after workers have stopped."""
        dispatch_id = None
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state == "canceled":
                return True
            if job.state not in ("queued", "running", "canceling"):
                return False
            job.state = "canceled"
            job.error = str(reason)
            job.result = None
            dispatch_id = self._finish_job_locked(job)
        self._emit_dispatch(dispatch_id)
        return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            return job is not None and job.state == "canceling"

    def status_dict(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if job is not None:
                self._touch_terminal_locked(job)
            return self._status_dict_locked(job) if job is not None else None

    def result_of(self, job_id: str) -> tuple[str, Any]:
        """Return ``(state, payload)`` where payload is the result dict (done),
        the error string (error), or the status dict (not ready/canceled)."""
        with self._lock:
            self._purge_terminal_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return "missing", None
            self._touch_terminal_locked(job)
            if job.state == "done":
                return "done", job.result
            if job.state == "error":
                return "error", job.error
            return job.state, self._status_dict_locked(job)

    def has_active(self) -> bool:
        """True until an accepted cancellation has actually finished."""
        with self._lock:
            self._purge_terminal_locked()
            return self._active_count_locked() > 0


# ── GUI-thread bridge ───────────────────────────────────────────────────────

class ApiBridge(QtCore.QObject):
    """Marshals fit requests from HTTP threads onto the Qt GUI thread.

    Created on (and thus owned by) the GUI thread.  HTTP handlers emit
    ``fit_requested``; because the receiver lives on the GUI thread the slot
    runs there via a queued connection — making it safe to start the optimizer
    and touch widgets from the slot.
    """

    fit_requested = QtCore.Signal(str)      # job_id
    cancel_requested = QtCore.Signal(str)   # job_id


# ── HTTP server ─────────────────────────────────────────────────────────────

def _make_handler(registry: JobRegistry, bridge: ApiBridge):
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # Quieter logs: route through nothing by default (the GUI owns stdout).
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
            return

        # ── helpers ──
        def _send_json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> Any:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b""
            return json.loads(raw.decode("utf-8")) if raw else {}

        def _discard_body(self) -> None:
            """Drain an optional body so keep-alive connections stay aligned."""
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                self.rfile.read(length)

        def _reserved_job_response(self, job_id: str) -> dict[str, Any]:
            status = registry.status_dict(job_id) or {}
            response: dict[str, Any] = {"job_id": job_id}
            state = status.get("state")
            if state is not None:
                response["state"] = state
            if "queue_position" in status:
                response["queue_position"] = status["queue_position"]
            return response

        def _send_bad_reserved_request(
            self,
            job_id: str,
            message: str,
        ) -> None:
            # If cancel won while the upload was being parsed, preserve its
            # terminal identity and answer the fit POST with the same job ID.
            if registry.discard_reservation(job_id):
                self._send_json(400, {"error": message})
                return
            status = registry.status_dict(job_id)
            if status is not None and status.get("state") in (
                    "canceling", "canceled"):
                self._send_json(202, self._reserved_job_response(job_id))
                return
            self._send_json(400, {"error": message})

        # ── routing ──
        def do_GET(self) -> None:  # noqa: N802
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if parts == ["ping"]:
                response = {
                    "status": "ok", "app": "EllipSDF",
                    "version": API_VERSION, "busy": registry.has_active(),
                }
                response.update(registry.queue_status())
                self._send_json(200, response)
                return
            if len(parts) == 3 and parts[0] == "fit" and parts[2] == "status":
                status = registry.status_dict(parts[1])
                if status is None:
                    self._send_json(404, {"error": "unknown job_id"})
                else:
                    self._send_json(200, status)
                return
            if len(parts) == 3 and parts[0] == "fit" and parts[2] == "result":
                state, payload = registry.result_of(parts[1])
                if state == "missing":
                    self._send_json(404, {"error": "unknown job_id"})
                elif state == "done":
                    self._send_json(200, payload)
                elif state == "error":
                    self._send_json(500, {"error": payload})
                else:                                   # queued / running
                    self._send_json(409, payload)
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if len(parts) == 3 and parts[0] == "fit" and parts[2] == "cancel":
                self._discard_body()
                job_id = parts[1]
                outcome, status = registry.request_cancel(job_id)
                if outcome == "missing":
                    self._send_json(404, {"error": "unknown job_id"})
                elif outcome == "accepted":
                    bridge.cancel_requested.emit(job_id)
                    self._send_json(202, status)
                elif outcome == "canceling":
                    self._send_json(202, status)
                elif outcome == "canceled":
                    self._send_json(200, status)
                else:  # done / error cannot be canceled retroactively
                    response = dict(status or {})
                    response["message"] = (
                        f"job is already {response.get('state', 'finished')}")
                    self._send_json(409, response)
                return

            is_fit = parts == ["fit"]
            is_pose_fit = parts in (["fit-pose"], ["fit_pose"])
            if not (is_fit or is_pose_fit):
                self._send_json(404, {"error": "not found"})
                return

            client_job_id = self.headers.get(CLIENT_JOB_ID_HEADER)
            if client_job_id is not None and (
                    len(client_job_id) != 32
                    or any(c not in "0123456789abcdef"
                           for c in client_job_id)):
                self._discard_body()
                self._send_json(400, {
                    "error": (
                        f"{CLIENT_JOB_ID_HEADER} must be exactly 32 lowercase "
                        "hex characters")
                })
                return

            # Reserve before reading the potentially large body.  A second HTTP
            # connection can now cancel this exact client-known ID while this
            # handler is blocked in ``_read_body``.
            reserve_outcome, job = registry.reserve(client_job_id)
            if job is None:
                self._discard_body()
                if reserve_outcome == "duplicate":
                    self._send_json(409, {
                        "error": "job_id already exists",
                        "job_id": client_job_id,
                    })
                else:
                    capacity = registry.queue_status()
                    self._send_json(429, {
                        "error": "fit queue is full",
                        **capacity,
                    })
                return

            try:
                previous_timeout = self.connection.gettimeout()
                self.connection.settimeout(DEFAULT_UPLOAD_TIMEOUT_SECONDS)
                try:
                    payload = self._read_body()
                finally:
                    self.connection.settimeout(previous_timeout)
            except (ValueError, json.JSONDecodeError, socket.timeout, OSError) as e:
                self._send_bad_reserved_request(
                    job.id, f"invalid JSON: {e}")
                return
            if (not isinstance(payload, dict)
                    or "vertices" not in payload or "faces" not in payload):
                self._send_bad_reserved_request(
                    job.id, "body must contain 'vertices' and 'faces'")
                return
            if is_pose_fit:
                if "ellipsoids" not in payload:
                    self._send_bad_reserved_request(
                        job.id, "body must contain 'ellipsoids'")
                    return
                payload["_api_mode"] = "fit_pose"

            outcome, attach_status = registry.attach_payload(job.id, payload)
            if outcome == "ready":
                self._send_json(202, self._reserved_job_response(job.id))
                return
            if outcome == "invalid":
                self._send_json(400, attach_status or {
                    "error": "invalid fit options",
                })
                return
            if outcome in ("canceling", "canceled"):
                self._send_json(202, self._reserved_job_response(job.id))
                return
            self._send_json(409, {
                "error": "reserved job is no longer available",
                "job_id": job.id,
            })

    return _Handler


class ApiServer:
    """Owns the registry, the GUI bridge, and the background HTTP server.

    Construct on the GUI thread (so ``ApiBridge`` gets GUI affinity), connect
    ``bridge.fit_requested`` to your ``MainWindow`` slot, then call
    :meth:`start`.  ``MainWindow`` writes progress/results back through
    :attr:`registry`.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        queue_limit: int = DEFAULT_QUEUE_LIMIT,
    ) -> None:
        self.host = host
        self.port = port
        self.bridge = ApiBridge()
        self.registry = JobRegistry(
            queue_limit=queue_limit,
            dispatch_callback=self.bridge.fit_requested.emit,
        )
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = _make_handler(self.registry, self.bridge)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        # Port 0 is useful for isolated tests and embedding; expose the actual
        # ephemeral port selected by the OS to callers.
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="EllipSDF-API", daemon=True)
        self._thread.start()
        print(f"[API] EllipSDF API listening on http://{self.host}:{self.port}")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
