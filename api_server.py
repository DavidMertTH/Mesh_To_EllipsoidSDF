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
``GET  /fit/<id>/status``      → ``{state, step, total, loss, count, error}``
``GET  /fit/<id>/result``      → result JSON (200) / not-ready (409) / error (500)

Request body for ``POST /fit`` (JSON)::

    {
      "vertices": [[x,y,z], ...],          # required
      "faces":    [[a,b,c], ...],          # required
      "options":  { "num_ellipsoids": 20, "max_ellipsoids": 60,
                    "num_steps": 2000, "symmetry": false, ... },   # optional
      "rig":      { "bones": [...], "boneWeights": [...], ... }     # optional
    }

Coordinate convention is the project's internal right-hand Y-up; callers are
responsible for any handedness conversion (see ``ellipsoid_exporter.py`` /
``unity/Editor/EllipsoidImporter.cs``).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PySide6 import QtCore

API_VERSION = 1
DEFAULT_PORT = 8765


# ── Job state ───────────────────────────────────────────────────────────────

@dataclass
class FitJob:
    """One fit request and its evolving state.

    Inputs are filled at creation (HTTP thread); ``state``/progress/``result``
    are mutated by ``MainWindow`` on the GUI thread.  All access goes through
    :class:`JobRegistry`, which serialises it with a lock.
    """

    id: str
    payload: dict[str, Any]                 # raw parsed request body
    state: str = "queued"                   # queued | running | done | error
    step: int = 0
    total: int = 0
    loss: float = 0.0
    count: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None

    def status_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "state": self.state,
            "step": self.step,
            "total": self.total,
            "loss": self.loss,
            "count": self.count,
            "error": self.error,
        }


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

    def get(self, job_id: str) -> FitJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def status_dict(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.status_dict() if job is not None else None

    def result_of(self, job_id: str) -> tuple[str, Any]:
        """Return ``(state, payload)`` where payload is the result dict (done),
        the error string (error), or the status dict (still running)."""
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
        """True while any job is queued or running (single-fit constraint)."""
        with self._lock:
            return any(j.state in ("queued", "running")
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
            if parts != ["fit"]:
                self._send_json(404, {"error": "not found"})
                return
            try:
                payload = self._read_body()
            except (ValueError, json.JSONDecodeError) as e:
                self._send_json(400, {"error": f"invalid JSON: {e}"})
                return
            if "vertices" not in payload or "faces" not in payload:
                self._send_json(
                    400, {"error": "body must contain 'vertices' and 'faces'"})
                return
            if registry.has_active():
                self._send_json(
                    409, {"error": "a fit is already in progress"})
                return
            job = registry.add(payload)
            bridge.fit_requested.emit(job.id)
            self._send_json(202, {"job_id": job.id})

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
