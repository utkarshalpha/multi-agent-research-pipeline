"""Smoke tests for the demo-facing routes (console, sample response, health)."""

import os

# config.py reads the environment at import time — these must be set before any
# project module is imported (load_dotenv() never overrides existing vars), so
# this module stays offline even when run standalone.
os.environ["MOCK_MODE"] = "true"
os.environ["QDRANT_PATH"] = ":memory:"

import unittest

from fastapi.testclient import TestClient

from main import app


class DemoRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_homepage_serves_demo_console(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Research Pipeline Console", response.text)
        self.assertIn("/static/app.js", response.text)

    def test_sample_response_matches_public_contract(self) -> None:
        response = self.client.get("/sample-response")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run_id"], "demo-portfolio-sample")
        self.assertTrue(payload["report"].startswith("# RAG vs Fine-Tuning"))
        self.assertGreaterEqual(len(payload["citations"]), 3)
        self.assertIn("token_usage", payload["metadata"])

    def test_health_response_has_status_and_model(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["model"])


if __name__ == "__main__":
    unittest.main()
