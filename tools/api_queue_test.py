"""Focused tests for the bounded FIFO in the embedded EllipSDF API.

Run:  .venv/Scripts/python tools/api_queue_test.py
"""

from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6 import QtCore  # noqa: E402

import api_server  # noqa: E402
from api_server import ApiServer, JobRegistry  # noqa: E402


CUBE = {
    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
}


def _request(url: str, method: str = "GET", payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


class JobRegistryQueueTests(unittest.TestCase):
    def test_fifo_dispatches_exactly_one_job_until_terminal_transition(self):
        dispatched: list[str] = []
        registry = JobRegistry(
            queue_limit=3,
            dispatch_callback=dispatched.append,
        )

        jobs = []
        for marker in ("1", "2", "3"):
            outcome, job = registry.enqueue(
                {**CUBE, "marker": marker}, marker * 32)
            self.assertEqual(outcome, "queued")
            self.assertIsNotNone(job)
            jobs.append(job)

        self.assertEqual(dispatched, [jobs[0].id])
        self.assertEqual(registry.status_dict(jobs[0].id)["queue_position"], 0)
        self.assertEqual(registry.status_dict(jobs[1].id)["queue_position"], 1)
        self.assertEqual(registry.status_dict(jobs[2].id)["queue_position"], 2)
        self.assertEqual(registry.status_dict(jobs[1].id)["stage"], "queued")
        self.assertEqual(
            registry.queue_status(),
            {"queue_limit": 3, "active_jobs": 3, "waiting_jobs": 2},
        )

        outcome, job = registry.reserve("4" * 32)
        self.assertEqual(outcome, "full")
        self.assertIsNone(job)

        self.assertTrue(registry.update(jobs[0].id, state="running"))
        self.assertTrue(registry.update(
            jobs[0].id, stage="sdf", detail="Thickness spheres 5/10",
            stage_progress=0.5))
        self.assertEqual(registry.status_dict(jobs[0].id)["detail"],
                         "Thickness spheres 5/10")
        self.assertEqual(registry.status_dict(jobs[0].id)["stage_progress"], 0.5)
        self.assertGreaterEqual(registry.status_dict(jobs[0].id)["stage_seconds"], 0.0)
        self.assertEqual(dispatched, [jobs[0].id])
        self.assertTrue(registry.update(
            jobs[0].id, state="done", result={"count": 1}))
        self.assertEqual(dispatched, [jobs[0].id, jobs[1].id])

        self.assertTrue(registry.update(jobs[1].id, state="running"))
        self.assertTrue(registry.update(
            jobs[1].id, state="error", error="expected failure"))
        self.assertEqual(
            dispatched, [jobs[0].id, jobs[1].id, jobs[2].id])

        self.assertTrue(registry.update(jobs[2].id, state="running"))
        self.assertTrue(registry.update(
            jobs[2].id, state="done", result={"count": 1}))
        self.assertFalse(registry.has_active())

    def test_waiting_cancel_is_immediate_and_releases_capacity(self):
        dispatched: list[str] = []
        registry = JobRegistry(
            queue_limit=2,
            dispatch_callback=dispatched.append,
        )
        _, active = registry.enqueue(CUBE, "a" * 32)
        _, waiting = registry.enqueue(CUBE, "b" * 32)
        self.assertEqual(dispatched, [active.id])

        outcome, status = registry.request_cancel(waiting.id)
        self.assertEqual(outcome, "canceled")
        self.assertEqual(status["state"], "canceled")
        self.assertEqual(dispatched, [active.id])

        outcome, replacement = registry.enqueue(CUBE, "c" * 32)
        self.assertEqual(outcome, "queued")
        self.assertIsNotNone(replacement)
        self.assertEqual(dispatched, [active.id])

        registry.update(active.id, state="running")
        registry.update(active.id, state="done", result={})
        self.assertEqual(dispatched, [active.id, replacement.id])

    def test_non_dispatched_job_cannot_claim_running_state(self):
        registry = JobRegistry(queue_limit=2, dispatch_callback=lambda _job: None)
        _, active = registry.enqueue(CUBE, "d" * 32)
        _, waiting = registry.enqueue(CUBE, "e" * 32)
        self.assertTrue(registry.update(active.id, state="running"))
        self.assertFalse(registry.update(waiting.id, state="running"))
        self.assertEqual(registry.status_dict(waiting.id)["state"], "queued")

    def test_invalid_options_atomically_release_reservation(self):
        registry = JobRegistry(queue_limit=1)
        outcome, job = registry.reserve("f" * 32)
        self.assertEqual(outcome, "reserved")

        outcome, detail = registry.attach_payload(
            job.id, {**CUBE, "options": "invalid"})

        self.assertEqual(outcome, "invalid")
        self.assertIn("options", detail["error"])
        self.assertIsNone(registry.status_dict(job.id))
        self.assertFalse(registry.has_active())
        outcome, replacement = registry.enqueue(CUBE, "0" * 32)
        self.assertEqual(outcome, "queued")
        self.assertIsNotNone(replacement)

    def test_batch_status_omits_preview_but_result_remains_available(self):
        registry = JobRegistry(dispatch_callback=lambda _job_id: None)
        _, job = registry.enqueue({
            **CUBE,
            "options": {"batch_pipeline": True},
        }, "1" * 32)
        self.assertTrue(registry.update(job.id, state="running"))
        result = {"count": 2, "ellipsoids": [{"id": 0}, {"id": 1}]}
        self.assertTrue(registry.update(
            job.id, state="done", result=result, preview=result))

        self.assertNotIn("preview", registry.status_dict(job.id))
        self.assertEqual(registry.result_of(job.id), ("done", result))

    def test_terminal_ttl_and_lru_bound_only_terminal_records(self):
        registry = JobRegistry(
            dispatch_callback=lambda _job_id: None,
            terminal_ttl_seconds=1000.0,
            terminal_job_limit=2,
        )

        def finish(job_id: str) -> None:
            _, job = registry.enqueue(CUBE, job_id)
            self.assertTrue(registry.update(job.id, state="running"))
            self.assertTrue(registry.update(
                job.id, state="done", result={"job": job_id}))

        first, second, third = ("2" * 32, "3" * 32, "4" * 32)
        finish(first)
        finish(second)
        self.assertIsNotNone(registry.status_dict(first))  # make first MRU
        finish(third)
        self.assertIsNotNone(registry.status_dict(first))
        self.assertIsNone(registry.status_dict(second))
        self.assertEqual(registry.result_of(third)[0], "done")

        with mock.patch.object(api_server.time, "monotonic", return_value=10.0):
            expiring = JobRegistry(
                dispatch_callback=lambda _job_id: None,
                terminal_ttl_seconds=5.0,
            )
            _, job = expiring.enqueue(CUBE, "5" * 32)
            expiring.update(job.id, state="running")
            expiring.update(job.id, state="done", result={})
        with mock.patch.object(api_server.time, "monotonic", return_value=16.0):
            self.assertIsNone(expiring.status_dict(job.id))

    def test_stale_upload_reservation_is_reaped_on_admission(self):
        with mock.patch.object(api_server.time, "monotonic", return_value=20.0):
            registry = JobRegistry(
                queue_limit=1, reservation_ttl_seconds=5.0)
            _, stale = registry.reserve("6" * 32)
        with mock.patch.object(api_server.time, "monotonic", return_value=26.0):
            outcome, replacement = registry.reserve("7" * 32)
        self.assertEqual(outcome, "reserved")
        self.assertIsNotNone(replacement)
        self.assertIsNone(registry.status_dict(stale.id))


class ApiQueueRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(
            sys.argv)

    def setUp(self):
        self.server = ApiServer(port=0, queue_limit=3)
        self.fit_events: list[str] = []
        self.server.bridge.fit_requested.connect(
            self.fit_events.append, QtCore.Qt.DirectConnection)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self):
        self.server.stop()

    def test_http_queue_limit_and_automatic_fifo_dispatch(self):
        job_ids = []
        for expected_position in (0, 1, 2):
            code, body = _request(f"{self.base}/fit", "POST", CUBE)
            self.assertEqual(code, 202)
            self.assertEqual(body["state"], "queued")
            self.assertEqual(body["queue_position"], expected_position)
            job_ids.append(body["job_id"])

        self.assertEqual(self.fit_events, [job_ids[0]])

        code, body = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 429)
        self.assertEqual(body["error"], "fit queue is full")
        self.assertEqual(body["queue_limit"], 3)
        self.assertEqual(body["active_jobs"], 3)

        code, body = _request(f"{self.base}/ping")
        self.assertEqual(code, 200)
        self.assertEqual(body["queue_limit"], 3)
        self.assertEqual(body["active_jobs"], 3)
        self.assertEqual(body["waiting_jobs"], 2)

        self.assertTrue(self.server.registry.update(job_ids[0], state="running"))
        self.assertTrue(self.server.registry.update(
            job_ids[0], state="done", result={"count": 0}))
        self.assertEqual(self.fit_events, job_ids[:2])

        self.assertTrue(self.server.registry.update(job_ids[1], state="running"))
        self.assertTrue(self.server.registry.update(
            job_ids[1], state="error", error="expected failure"))
        self.assertEqual(self.fit_events, job_ids)

    def test_fit_pose_payload_remains_tagged_while_waiting(self):
        code, active = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 202)
        pose_payload = {
            **CUBE,
            "ellipsoids": [{"id": 0, "radii": [1, 1, 1]}],
        }
        code, waiting = _request(
            f"{self.base}/fit-pose", "POST", pose_payload)
        self.assertEqual(code, 202)
        self.assertEqual(waiting["queue_position"], 1)
        queued_job = self.server.registry.get(waiting["job_id"])
        self.assertEqual(queued_job.payload["_api_mode"], "fit_pose")

        self.server.registry.update(active["job_id"], state="running")
        self.server.registry.update(
            active["job_id"], state="done", result={"count": 0})
        self.assertEqual(self.fit_events[-1], waiting["job_id"])

    def test_invalid_options_returns_400_without_leaking_capacity(self):
        code, body = _request(
            f"{self.base}/fit", "POST", {**CUBE, "options": "invalid"})
        self.assertEqual(code, 400)
        self.assertIn("options", body["error"])

        code, status = _request(f"{self.base}/ping")
        self.assertEqual(code, 200)
        self.assertEqual(status["active_jobs"], 0)

        code, accepted = _request(f"{self.base}/fit", "POST", CUBE)
        self.assertEqual(code, 202)
        self.server.registry.complete_cancel(accepted["job_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
