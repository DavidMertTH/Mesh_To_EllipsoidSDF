"""Smoke test for the EllipSDF HTTP API layer (no GUI).

Exercises the server + registry round-trip without launching the Qt window:
a stub slot stands in for MainWindow and writes a fake result, so this verifies
routing, JSON, the busy guard, and status/result transitions — not the fit.

Run:  .venv/Scripts/python tools/api_smoke_test.py
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6 import QtCore  # noqa: E402

from api_server import ApiServer  # noqa: E402


def _get(url: str):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _post(url: str, payload: dict):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def main() -> int:
    app = QtCore.QCoreApplication(sys.argv)
    server = ApiServer(port=8799)

    # Stub MainWindow: on a fit request, immediately publish a fake 2-ellipsoid
    # result. Direct connection → runs synchronously on the HTTP thread.
    def on_fit(job_id: str):
        server.registry.update(job_id, state="running")
        result = {
            "version": 2, "rigged": False, "count": 2,
            "ellipsoids": [
                {"name": "Sphere_0", "bone": None, "center": [0, 0, 0],
                 "radii": [1, 1, 1], "rotation": [0, 0, 0, 1]},
                {"name": "Sphere_1", "bone": None, "center": [1, 0, 0],
                 "radii": [0.5, 0.5, 0.5], "rotation": [0, 0, 0, 1]},
            ],
        }
        server.registry.update(job_id, state="done", result=result, count=2)

    server.bridge.fit_requested.connect(on_fit, QtCore.Qt.DirectConnection)
    server.start()

    base = "http://127.0.0.1:8799"
    ok = True

    code, body = _get(f"{base}/ping")
    print("ping:", code, body)
    ok &= code == 200 and body["busy"] is False

    cube = {
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        "options": {"num_ellipsoids": 2, "num_steps": 10},
    }
    code, body = _post(f"{base}/fit", cube)
    print("fit:", code, body)
    ok &= code == 202 and "job_id" in body
    job_id = body["job_id"]

    code, body = _get(f"{base}/fit/{job_id}/status")
    print("status:", code, body)
    ok &= code == 200 and body["state"] == "done"

    code, body = _get(f"{base}/fit/{job_id}/result")
    print("result:", code, body)
    ok &= code == 200 and body["count"] == 2
    ok &= body["ellipsoids"][0]["name"] == "Sphere_0"

    # Missing-fields guard.
    code, body = _post(f"{base}/fit", {"vertices": []})
    print("bad-body:", code, body)
    ok &= code == 400

    # Unknown job.
    code, body = _get(f"{base}/fit/deadbeef/status")
    print("unknown:", code, body)
    ok &= code == 404

    server.stop()
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
