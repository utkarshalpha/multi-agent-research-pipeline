"""End-to-end tests for ``POST /research`` in MOCK_MODE via the TestClient.

The whole LangGraph pipeline (planner -> researcher -> critic -> writer) runs
in-process against the deterministic offline mocks: no API keys, no network.
Auth and rate limiting are exercised by patching the ``config`` settings
attributes at runtime (main.py reads them per-request via ``config.settings``),
so no module reloading is needed.
"""

from __future__ import annotations

import os

# config.py reads the environment at import time — these must be set before any
# project module is imported (load_dotenv() never overrides existing vars).
os.environ["MOCK_MODE"] = "true"
os.environ["QDRANT_PATH"] = ":memory:"

import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import config
import main
from config import MAX_RETRIES


class ResearchEndpointTest(unittest.TestCase):
    """Full-pipeline HTTP tests against the FastAPI app in MOCK_MODE."""

    @classmethod
    def setUpClass(cls) -> None:
        # No lifespan context: Qdrant/Redis stay disconnected and every cache
        # access degrades gracefully to a miss — the run stays fully offline.
        cls.client = TestClient(main.app)

    def setUp(self) -> None:
        # Isolate the in-memory rate limiter between tests.
        main._rate_limit_windows.clear()

    def _post(self, query: str, **kwargs) -> httpx.Response:
        return self.client.post("/research", json={"query": query}, **kwargs)

    # ------------------------------------------------------------------ #
    # Happy path + response contract
    # ------------------------------------------------------------------ #
    def test_research_returns_full_contract_in_mock_mode(self) -> None:
        response = self._post("What are the benefits of solar energy for cities?")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in ("run_id", "report", "citations", "metadata"):
            self.assertIn(key, payload)

        self.assertTrue(payload["run_id"])
        self.assertTrue(payload["report"].startswith("# "))
        self.assertIn("## References", payload["report"])

        self.assertIsInstance(payload["citations"], list)
        self.assertGreater(len(payload["citations"]), 0)
        for url in payload["citations"]:
            self.assertIsInstance(url, str)

        metadata = payload["metadata"]
        self.assertTrue(metadata["model"].startswith("mock ("))
        self.assertEqual(metadata["retry_count"], 0)
        self.assertGreaterEqual(metadata["num_sub_questions"], 3)
        self.assertGreater(metadata["num_sources"], 0)

        token_usage = metadata["token_usage"]
        self.assertIsInstance(token_usage, dict)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.assertIn(key, token_usage)

    def test_health_labels_model_as_mock(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["model"].startswith("mock ("))

    # ------------------------------------------------------------------ #
    # Critic retry loop (force-retry test hook)
    # ------------------------------------------------------------------ #
    def test_force_retry_marker_exercises_retry_loop(self) -> None:
        response = self._post("Explain force-retry semantics in distributed job queues")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        # The mock critic scores force-retry content ~0.1, so the loop must
        # retry at least once (and never beyond the configured budget).
        self.assertGreaterEqual(payload["metadata"]["retry_count"], 1)
        self.assertLessEqual(payload["metadata"]["retry_count"], MAX_RETRIES)
        # A report is still produced after the budget is exhausted.
        self.assertTrue(payload["report"].strip())
        self.assertTrue(payload["report"].startswith("# "))

    # ------------------------------------------------------------------ #
    # Auth (X-API-Key)
    # ------------------------------------------------------------------ #
    def test_auth_rejects_missing_and_wrong_key_and_accepts_correct_key(self) -> None:
        with patch.object(config, "PIPELINE_API_KEY", "test-secret"):
            missing = self._post("What is photosynthesis?")
            self.assertEqual(missing.status_code, 401)

            wrong = self._post(
                "What is photosynthesis?", headers={"X-API-Key": "wrong-key"}
            )
            self.assertEqual(wrong.status_code, 401)

            correct = self._post(
                "What is photosynthesis?", headers={"X-API-Key": "test-secret"}
            )
            self.assertEqual(correct.status_code, 200)
            self.assertIn("report", correct.json())

    def test_auth_disabled_when_key_is_empty(self) -> None:
        with patch.object(config, "PIPELINE_API_KEY", ""):
            response = self._post("What is the speed of light in a vacuum?")
        self.assertEqual(response.status_code, 200)

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #
    def test_rate_limit_returns_429_on_excess_requests(self) -> None:
        with patch.object(config, "RATE_LIMIT_PER_MINUTE", 2):
            main._rate_limit_windows.clear()
            first = self._post("How do tides work?")
            second = self._post("How do tides work near coastlines?")
            third = self._post("How do tides affect shipping?")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)
        self.assertIn("Rate limit exceeded", third.json()["detail"])

    # ------------------------------------------------------------------ #
    # Request validation
    # ------------------------------------------------------------------ #
    def test_query_below_min_length_is_rejected(self) -> None:
        response = self._post("ab")
        self.assertEqual(response.status_code, 422)

    def test_missing_query_is_rejected(self) -> None:
        response = self.client.post("/research", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
