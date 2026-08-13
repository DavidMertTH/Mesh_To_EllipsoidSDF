"""Cancellation tests for the embedded EllipSDF HTTP API.

Run:  .venv/Scripts/python tools/api_cancel_test.py
"""

from __future__ import annotations

import json
import http.client
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6 import QtCore  # noqa: E402

from api_server import (  # noqa: E402
    ApiServer,
    CLIENT_JOB_ID_HEADER,
    JobRegistry,
)


CUBE = {
    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
}


def _request(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _raw_request(
    url: str,
    raw: bytes,
    headers: dict[str, str] | None = None,
):
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


class JobRegistryCancelTests(unittest.TestCase):
    def test_client_id_reservation_attach_and_discard(self):
        registry = JobRegistry()
        job_id = "1" * 32
        job = registry.reserve_if_idle(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.id, job_id)
        self.assertIsNone(job.payload)
        self.assertTrue(registry.has_active())
        self.assertIsNone(registry.reserve_if_idle("2" * 32))

        outcome, status = registry.attach_payload(job_id, CUBE)
        self.assertEqual(outcome, "ready")
        self.assertEqual(status["state"], "queued")
        self.assertEqual(registry.get(job_id).payload, CUBE)
        self.assertFalse(registry.discard_reservation(job_id))

        other = JobRegistry()
        reserved = other.reserve_if_idle(job_id)
        self.assertTrue(other.discard_reservation(reserved.id))
        self.assertFalse(other.has_active())
        self.assertIsNone(other.status_dict(job_id))

    def test_cancel_wins_before_reserved_payload_is_attached(self):
        registry = JobRegistry()
        job_id = "3" * 32
        registry.reserve_if_idle(job_id)
        outcome, _ = registry.request_cancel(job_id)
        self.assertEqual(outcome, "accepted")
        outcome, status = registry.attach_payload(job_id, CUBE)
        self.assertEqual(outcome, "canceling")
        self.assertEqual(status["state"], "canceling")
        self.assertIsNone(registry.get(job_id).payload)
        self.assertFalse(registry.discard_reservation(job_id))

    def test_atomic_add_respects_canceling_job(self):
        registry = JobRegistry()
        job = registry.add_if_idle(CUBE)
        self.assertIsNotNone(job)
        self.assertIsNone(registry.add_if_idle(CUBE))
        registry.request_cancel(job.id)
        self.assertIsNone(registry.add_if_idle(CUBE))
        registry.complete_cancel(job.id)
        self.assertIsNotNone(registry.add_if_idle(CUBE))

    def test_canceling_remains_busy_and_rejects_late_completion(self):
        registry = JobRegistry()
        job = registry.add(CUBE)
        self.assertTrue(registry.update(job.id, state="running", step=3))

        outcome, status = registry.request_cancel(job.id)
        self.assertEqual(outcome, "accepted")
        self.assertEqual(status["state"], "canceling")
        self.assertTrue(registry.has_active())

        # A worker completion racing the GUI cancel slot cannot resurrect the
        # job, and an idempotent second request emits no second cancellation.
        self.assertFalse(registry.update(job.id, state="done", result={}))
        outcome, status = registry.request_cancel(job.id)
        self.assertEqual(outcome, "canceling")
        self.assertEqual(status["state"], "canceling")
        self.assertTrue(registry.has_active())

        self.assertTrue(registry.complete_cancel(job.id))
        self.assertFalse(registry.has_active())
        self.assertEqual(registry.status_dict(job.id)["state"], "canceled")
        self.assertTrue(registry.complete_cancel(job.id))

    def test_finished_jobs_are_not_rewritten(self):
        registry = JobRegistry()
        job = registry.add(CUBE)
        self.assertTrue(registry.update(job.id, state="done", result={"count": 0}))
        outcome, status = registry.request_cancel(job.id)
        self.assertEqual(outcome, "finished")
        self.assertEqual(status["state"], "done")
        self.assertFalse(registry.complete_cancel(job.id))


class ApiCancelRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv)

    def setUp(self):
        self.server = ApiServer(port=0)
        self.fit_events: list[str] = []
        self.cancel_events: list[str] = []

        def on_fit(job_id: str):
            self.fit_events.append(job_id)
            self.server.registry.update(job_id, state="running")

        def on_cancel(job_id: str):
            self.cancel_events.append(job_id)
            self.server.registry.complete_cancel(job_id, "test cancellation")

        # Direct connections keep this HTTP-layer test independent of a GUI
        # event loop; MainWindow integration uses queued GUI-thread connections.
        self.server.bridge.fit_requested.connect(on_fit, QtCore.Qt.DirectConnection)
        self.server.bridge.cancel_requested.connect(
            on_cancel, QtCore.Qt.DirectConnection)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self):
        self.server.stop()

    def test_cancel_route_is_idempotent_and_releases_busy_guard(self):
        code, body = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 202)
        job_id = body["job_id"]

        code, body = _request(f"{self.base}/ping")
        self.assertEqual(code, 200)
        self.assertTrue(body["busy"])

        code, body = _request(f"{self.base}/fit/{job_id}/cancel", "POST")
        self.assertEqual(code, 202)
        self.assertEqual(body["state"], "canceling")
        self.assertEqual(self.cancel_events, [job_id])

        code, body = _request(f"{self.base}/fit/{job_id}/cancel", "POST")
        self.assertEqual(code, 200)
        self.assertEqual(body["state"], "canceled")
        self.assertEqual(self.cancel_events, [job_id])

        code, body = _request(f"{self.base}/fit/{job_id}/status")
        self.assertEqual(code, 200)
        self.assertEqual(body["state"], "canceled")
        self.assertEqual(body["error"], "test cancellation")

        code, body = _request(f"{self.base}/fit/{job_id}/result")
        self.assertEqual(code, 409)
        self.assertEqual(body["state"], "canceled")

        code, body = _request(f"{self.base}/ping")
        self.assertEqual(code, 200)
        self.assertFalse(body["busy"])

        # A new fit is accepted only after cancellation has really completed.
        code, body = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 202)
        self.assertNotEqual(body["job_id"], job_id)

    def test_valid_client_job_id_is_echoed(self):
        job_id = "4" * 32
        code, body = _request(
            f"{self.base}/fit",
            "POST",
            CUBE,
            {CLIENT_JOB_ID_HEADER: job_id},
        )
        self.assertEqual(code, 202)
        self.assertEqual(body["job_id"], job_id)
        self.assertEqual(self.fit_events, [job_id])
        self.assertEqual(
            self.server.registry.status_dict(job_id)["state"], "running")

    def test_invalid_client_job_id_and_bad_json_leave_no_reservation(self):
        bad_id = "A" * 32
        code, body = _request(
            f"{self.base}/fit",
            "POST",
            CUBE,
            {CLIENT_JOB_ID_HEADER: bad_id},
        )
        self.assertEqual(code, 400)
        self.assertIn(CLIENT_JOB_ID_HEADER, body["error"])
        self.assertFalse(self.server.registry.has_active())

        job_id = "5" * 32
        code, body = _raw_request(
            f"{self.base}/fit",
            b"{",
            {CLIENT_JOB_ID_HEADER: job_id},
        )
        self.assertEqual(code, 400)
        self.assertIn("invalid JSON", body["error"])
        self.assertIsNone(self.server.registry.status_dict(job_id))
        self.assertFalse(self.server.registry.has_active())

        missing_id = "6" * 32
        code, body = _request(
            f"{self.base}/fit",
            "POST",
            {"vertices": []},
            {CLIENT_JOB_ID_HEADER: missing_id},
        )
        self.assertEqual(code, 400)
        self.assertIsNone(self.server.registry.status_dict(missing_id))
        self.assertFalse(self.server.registry.has_active())

    def test_client_id_can_cancel_while_fit_body_is_still_uploading(self):
        job_id = "7" * 32
        raw = json.dumps(CUBE).encode("utf-8")
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.port, timeout=2.0)
        try:
            connection.putrequest("POST", "/fit")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(raw)))
            connection.putheader(CLIENT_JOB_ID_HEADER, job_id)
            connection.endheaders()

            deadline = time.monotonic() + 2.0
            while (self.server.registry.status_dict(job_id) is None
                   and time.monotonic() < deadline):
                time.sleep(0.005)
            self.assertEqual(
                self.server.registry.status_dict(job_id)["state"], "queued")
            self.assertIsNone(self.server.registry.get(job_id).payload)

            # Reservation itself enforces the single-fit constraint, before
            # the first request has uploaded or parsed its body.
            code, body = _request(f"{self.base}/fit", "POST", CUBE)
            self.assertEqual(code, 409)

            code, body = _request(
                f"{self.base}/fit/{job_id}/cancel", "POST")
            self.assertEqual(code, 202)
            self.assertEqual(body["job_id"], job_id)
            self.assertEqual(self.cancel_events, [job_id])

            connection.send(raw)
            response = connection.getresponse()
            post_body = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 202)
            self.assertEqual(post_body["job_id"], job_id)
            self.assertEqual(post_body["state"], "canceled")
        finally:
            connection.close()

        self.assertEqual(self.fit_events, [])
        self.assertIsNone(self.server.registry.get(job_id).payload)
        self.assertEqual(
            self.server.registry.status_dict(job_id)["state"], "canceled")
        self.assertFalse(self.server.registry.has_active())

    def test_unknown_and_finished_cancel_responses(self):
        code, body = _request(f"{self.base}/fit/deadbeef/cancel", "POST")
        self.assertEqual(code, 404)
        self.assertEqual(body["error"], "unknown job_id")

        code, body = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 202)
        job_id = body["job_id"]
        self.assertTrue(self.server.registry.update(
            job_id, state="done", result={"count": 0}))

        code, body = _request(f"{self.base}/fit/{job_id}/cancel", "POST")
        self.assertEqual(code, 409)
        self.assertEqual(body["state"], "done")
        self.assertEqual(self.cancel_events, [])

    def test_canceling_http_job_stays_busy_until_gui_acknowledges_stop(self):
        self.server.bridge.cancel_requested.disconnect()
        code, body = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 202)
        job_id = body["job_id"]

        code, body = _request(
            f"{self.base}/fit/{job_id}/cancel", "POST", {"reason": "test"})
        self.assertEqual(code, 202)
        self.assertEqual(body["state"], "canceling")

        code, body = _request(f"{self.base}/ping")
        self.assertEqual(code, 200)
        self.assertTrue(body["busy"])
        code, body = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 409)

        self.server.registry.complete_cancel(job_id)
        code, body = _request(f"{self.base}/ping")
        self.assertEqual(code, 200)
        self.assertFalse(body["busy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
