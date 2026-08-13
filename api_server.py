"""
api_server.py — Embedded HTTP API for driving EllipSDF from another process.

Purpose
-------
Lets an external client (e.g. a Unity ``MonoBehaviour``) push a triangle mesh
into a *running* EllipSDF instance, watch the fit happen live in the normal
viewport, and pull the fitted ellipsoids back out.  The same server runs whether
EllipSDF was started manually or launched on demand by the client, so "attach to
running app" and "start it for me" share one code path.

Architecture
------------
The HTTP server runs in a daemon thread (``http.server``), so request handlers do
*not* run on the Qt GUI thread.  Starting a fit must happen on the GUI thread
(the optimizer is a ``QThread`` owned by ``MainWindow`` and renders into the
viewport), so a request handler never touches Qt directly: it stores the job in a
thread-safe :class:`JobRegistry` and emits :class:`ApiBridge.fit_requested`.  That
signal is delivered via a *queued* connection to ``MainWindow`` on the GUI thread,
which then drives the existing load → compute-SDF → fit pipeline and writes
progress/result back into the job.  Pollers read job state under a lock.

Endpoints
---------
``GET  /ping``                 → ``{status, app, version, busy}``
``POST /fit``                  → ``{job_id}``  (202; 409 if a fit is in flight)
``POST /fit-pose``             → ``{job_id}``  (fit posted ellipsoid IDs to mesh)
``POST /fit/<id>/cancel``      → request cooperative cancellation
``GET  /fit/<id>/status``      → ``{state, step, total, loss, count, error,
                                      preview?}``
``GET  /fit/<id>/result``      → result JSON (200) / not-ready or canceled
                                      (409) / error (500)

Request body for ``POST /fit`` (JSON)::

    {
      "vertices": [[x,y,z], ...],          # required
      "faces":    [[a,b,c], ...],          # required
      "options":  { "num_ellipsoids": 20, "max_ellipsoids": 60,
                    "num_steps": 2000, "symmetry": false, ... },   # optional
      "rig":      { "bones": [...], "boneWeights": [...], ... }     # optional
    }

Clients that need to cancel while the request body is still uploading may send
``X-EllipSDF-Job-Id`` with a lowercase 32-hex UUID (Unity
``Guid.NewGuid().ToString("N")``).  The API reserves that ID atomically before
reading the body, so ``POST /fit/<id>/cancel`` can already address the job.

Coordinate convention is caller-defined: vertices, rig transforms, and returned
ellipsoids stay in the same snapshot space.  The Unity bridge uses Unity world
space so transformed and skinned meshes line up with the scene view.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PySide6 import QtCore

API_VERSION = 2
DEFAULT_PORT = 8765
CLIENT_JOB_ID_HEADER = "X-EllipSDF-Job-Id"


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
    result: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None
    error: str | None = None

    def status_dict(self) -> dict[str, Any]:
        status = {
            "job_id": self.id,
            "state": self.state,
            "step": self.step,
            "total": self.total,
            "loss": self.loss,
            "count": self.count,
            "error": self.error,
        }
        if self.preview is not None:
            status["preview"] = self.preview
        return status


class JobRegistry:
    """Thread-safe store for :class:`FitJob` objects.

    Mutations from the GUI thread and reads from HTTP handler threads are
    serialised with a single lock.  Callers must go through ``update`` rather
    than mutating a job in place so every change is published under the lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, FitJob] = {}

    def add(self, payload: dict[str, Any]) -> FitJob:
        job = FitJob(id=uuid.uuid4().hex, payload=payload)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def add_if_idle(self, payload: dict[str, Any]) -> FitJob | None:
        """Atomically add a job if no fit (including cancellation) is active."""
        with self._lock:
            if any(j.state in ("queued", "running", "canceling")
                   for j in self._jobs.values()):
                return None
            job = FitJob(id=uuid.uuid4().hex, payload=payload)
            self._jobs[job.id] = job
            return job

    def reserve_if_idle(self, job_id: str | None = None) -> FitJob | None:
        """Reserve an ID before its HTTP request body has been read.

        Explicit client IDs are never reused, including after a terminal job.
        This keeps retries and late cancel requests unambiguous.
        """
        with self._lock:
            if job_id is not None and job_id in self._jobs:
                return None
            if any(j.state in ("queued", "running", "canceling")
                   for j in self._jobs.values()):
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
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "missing", None
            if job.state == "queued" and job.payload is None:
                job.payload = payload
                return "ready", job.status_dict()
            if job.state in ("canceling", "canceled"):
                return job.state, job.status_dict()
            return "unavailable", job.status_dict()

    def discard_reservation(self, job_id: str) -> bool:
        """Remove a still-empty queued reservation after a bad HTTP body."""
        with self._lock:
            job = self._jobs.get(job_id)
            if (job is None or job.state != "queued"
                    or job.payload is not None):
                return False
            del self._jobs[job_id]
            return True

    def get(self, job_id: str) -> FitJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> bool:
        """Publish a normal job update.

        Cancellation owns the job once it reaches ``canceling``.  Late worker
        progress or completion callbacks must not be able to resurrect it as a
        running/done job; only :meth:`complete_cancel` may leave that state.
        Terminal states are immutable for the same reason.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state in ("canceling", "canceled", "done", "error"):
                return False
            for k, v in fields.items():
                setattr(job, k, v)
            return True

    def request_cancel(self, job_id: str) -> tuple[str, dict[str, Any] | None]:
        """Atomically request cancellation.

        Returns ``(outcome, status)``.  ``accepted`` is the only outcome that
        should emit the GUI bridge signal.  Repeated requests are no-ops and
        report ``canceling``/``canceled``; completed jobs report ``finished``.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "missing", None
            if job.state in ("queued", "running"):
                job.state = "canceling"
                job.error = None
                return "accepted", job.status_dict()
            if job.state == "canceling":
                return "canceling", job.status_dict()
            if job.state == "canceled":
                return "canceled", job.status_dict()
            return "finished", job.status_dict()

    def complete_cancel(
        self,
        job_id: str,
        reason: str = "canceled by client",
    ) -> bool:
        """Mark a requested cancellation complete after workers have stopped."""
        with self._lock:
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
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.state == "canceling"

    def status_dict(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.status_dict() if job is not None else None

    def result_of(self, job_id: str) -> tuple[str, Any]:
        """Return ``(state, payload)`` where payload is the result dict (done),
        the error string (error), or the status dict (not ready/canceled)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "missing", None
            if job.state == "done":
                return "done", job.result
            if job.state == "error":
                return "error", job.error
            return job.state, job.status_dict()

    def has_active(self) -> bool:
        """True until an accepted cancellation has actually finished."""
        with self._lock:
            return any(j.state in ("queued", "running", "canceling")
                       for j in self._jobs.values())


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
            if state in ("canceling", "canceled"):
                response["state"] = state
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
                self._send_json(200, {
                    "status": "ok", "app": "EllipSDF",
                    "version": API_VERSION, "busy": registry.has_active(),
                })
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
            job = registry.reserve_if_idle(client_job_id)
            if job is None:
                self._discard_body()
                if (client_job_id is not None
                        and registry.status_dict(client_job_id) is not None):
                    self._send_json(409, {
                        "error": "job_id already exists",
                        "job_id": client_job_id,
                    })
                else:
                    self._send_json(
                        409, {"error": "a fit is already in progress"})
                return

            try:
                payload = self._read_body()
            except (ValueError, json.JSONDecodeError, OSError) as e:
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

            outcome, _ = registry.attach_payload(job.id, payload)
            if outcome == "ready":
                bridge.fit_requested.emit(job.id)
                self._send_json(202, {"job_id": job.id})
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

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self.registry = JobRegistry()
        self.bridge = ApiBridge()
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
