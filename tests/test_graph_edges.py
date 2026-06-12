"""Unit tests for the critic-retry decision edge (``graph.edges.should_retry``).

``should_retry`` is a pure function of state, so these tests construct minimal
state dicts and assert the routing decision for every branch: critique passed,
critique failed with retry budget remaining, budget exhausted, total search
failure (zero results), and missing ``critique``/``retry_count`` fields.
"""

from __future__ import annotations

import os

# config.py reads the environment at import time — these must be set before any
# project module is imported (load_dotenv() never overrides existing vars).
os.environ["MOCK_MODE"] = "true"
os.environ["QDRANT_PATH"] = ":memory:"

import unittest

from config import MAX_RETRIES
from graph.edges import should_retry
from schemas.models import CritiqueResult


def _critique(
    overall_pass: bool,
    scores: list[float] | None = None,
    low_confidence_indices: list[int] | None = None,
    feedback: str = "Results were too weak; reformulate the searches.",
) -> CritiqueResult:
    """Build a CritiqueResult with sensible defaults for edge testing."""
    return CritiqueResult(
        scores=scores if scores is not None else [0.9],
        low_confidence_indices=low_confidence_indices or [],
        overall_pass=overall_pass,
        feedback=feedback,
    )


class ShouldRetryTest(unittest.TestCase):
    """Branch coverage for the researcher/writer routing decision."""

    def test_retry_budget_is_configured(self) -> None:
        # The retry branch tests below assume at least one retry is allowed.
        self.assertGreaterEqual(MAX_RETRIES, 1)

    def test_passing_critique_routes_to_writer(self) -> None:
        state = {"critique": _critique(overall_pass=True), "retry_count": 0}
        self.assertEqual(should_retry(state), "writer")

    def test_passing_critique_routes_to_writer_even_with_budget_spent(self) -> None:
        state = {"critique": _critique(overall_pass=True), "retry_count": MAX_RETRIES}
        self.assertEqual(should_retry(state), "writer")

    def test_failed_critique_with_full_budget_routes_to_researcher(self) -> None:
        state = {
            "critique": _critique(overall_pass=False, scores=[0.3], low_confidence_indices=[0]),
            "retry_count": 0,
        }
        self.assertEqual(should_retry(state), "researcher")

    def test_failed_critique_on_last_allowed_retry_routes_to_researcher(self) -> None:
        state = {
            "critique": _critique(overall_pass=False, scores=[0.3], low_confidence_indices=[0]),
            "retry_count": MAX_RETRIES - 1,
        }
        self.assertEqual(should_retry(state), "researcher")

    def test_failed_critique_with_exhausted_budget_routes_to_writer(self) -> None:
        state = {
            "critique": _critique(overall_pass=False, scores=[0.3], low_confidence_indices=[0]),
            "retry_count": MAX_RETRIES,
        }
        self.assertEqual(should_retry(state), "writer")

    def test_failed_critique_beyond_budget_routes_to_writer(self) -> None:
        state = {
            "critique": _critique(overall_pass=False, scores=[0.3], low_confidence_indices=[0]),
            "retry_count": MAX_RETRIES + 1,
        }
        self.assertEqual(should_retry(state), "writer")

    def test_zero_result_failure_retries_while_budget_remains(self) -> None:
        # Total search failure: the critic fails with an empty score list; the
        # edge must still loop back to the researcher while budget remains.
        state = {
            "critique": _critique(overall_pass=False, scores=[], low_confidence_indices=[]),
            "retry_count": 0,
        }
        self.assertEqual(should_retry(state), "researcher")

    def test_zero_result_failure_with_exhausted_budget_routes_to_writer(self) -> None:
        state = {
            "critique": _critique(overall_pass=False, scores=[], low_confidence_indices=[]),
            "retry_count": MAX_RETRIES,
        }
        self.assertEqual(should_retry(state), "writer")

    def test_missing_critique_routes_to_writer(self) -> None:
        self.assertEqual(should_retry({"retry_count": 0}), "writer")

    def test_none_critique_routes_to_writer(self) -> None:
        self.assertEqual(should_retry({"critique": None, "retry_count": 0}), "writer")

    def test_missing_retry_count_defaults_to_zero_and_retries(self) -> None:
        state = {"critique": _critique(overall_pass=False, scores=[0.2], low_confidence_indices=[0])}
        self.assertEqual(should_retry(state), "researcher")


if __name__ == "__main__":
    unittest.main()
