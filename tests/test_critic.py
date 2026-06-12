"""Unit tests for the Critic's deterministic scoring and failure semantics.

Covers the credibility tiers (arXiv=1.0, .gov/.edu and known outlets=0.8,
unknown=0.5), the recency decay curve, the weighted score combination with
relevance clamping, zero-result sub-question flagging, and the rule that total
search failure fails the critique while the retry budget remains.

The LLM relevance call is patched with a deterministic stub so the combination
math can be asserted exactly; ``redis_store`` is unconnected in tests, so its
``save`` calls are no-ops.
"""

from __future__ import annotations

import os

# config.py reads the environment at import time — these must be set before any
# project module is imported (load_dotenv() never overrides existing vars).
os.environ["MOCK_MODE"] = "true"
os.environ["QDRANT_PATH"] = ":memory:"

import asyncio
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from agents.critic import (
    _W_CREDIBILITY,
    _W_RECENCY,
    _W_RELEVANCE,
    _RelevanceAssessment,
    _credibility,
    _recency,
    critic_node,
)
from config import CONFIDENCE_PASS_THRESHOLD, MAX_RETRIES
from graph.edges import should_retry
from schemas.models import ResearchResult


def _result(
    question: str = "What is X?",
    source_type: str = "web",
    source_url: str = "https://blog.unknown-site.io/post",
    published_date: str | None = None,
    content: str = "Some evidence about X.",
) -> ResearchResult:
    """Build a ResearchResult with overridable fields for scoring tests."""
    return ResearchResult(
        question=question,
        content=content,
        source_url=source_url,
        source_type=source_type,
        published_date=published_date,
    )


def _days_ago(days: int) -> str:
    """ISO date string ``days`` days before today."""
    return (date.today() - timedelta(days=days)).isoformat()


def _run_critic(state: dict, scores: list[float], feedback: str = "Tighten the searches.") -> dict:
    """Run critic_node with the LLM relevance assessment stubbed out."""

    async def _fake_assess(results: list[ResearchResult]) -> _RelevanceAssessment:
        return _RelevanceAssessment(relevance_scores=scores, feedback=feedback)

    with patch("agents.critic._assess_relevance", new=_fake_assess):
        return asyncio.run(critic_node(state))


def _combined(relevance: float, credibility: float, recency: float) -> float:
    """The Critic's weighted combination (relevance pre-clamped by caller)."""
    return round(
        _W_RELEVANCE * relevance + _W_CREDIBILITY * credibility + _W_RECENCY * recency, 3
    )


class CredibilityTierTest(unittest.TestCase):
    """arXiv=1.0, .gov/.edu and known outlets=0.8, unknown=0.5."""

    def test_arxiv_source_scores_top_tier(self) -> None:
        result = _result(source_type="arxiv", source_url="https://arxiv.org/pdf/2601.00001v1")
        self.assertEqual(_credibility(result), 1.0)

    def test_arxiv_tier_ignores_url(self) -> None:
        result = _result(source_type="arxiv", source_url="https://mirror.example.com/paper.pdf")
        self.assertEqual(_credibility(result), 1.0)

    def test_gov_domain_scores_reputable_tier(self) -> None:
        result = _result(source_url="https://www.nasa.gov/mission/artemis")
        self.assertEqual(_credibility(result), 0.8)

    def test_edu_domain_scores_reputable_tier(self) -> None:
        result = _result(source_url="https://www.berkeley.edu/research/ai-safety")
        self.assertEqual(_credibility(result), 0.8)

    def test_known_outlet_scores_reputable_tier(self) -> None:
        result = _result(source_url="https://www.reuters.com/technology/some-story")
        self.assertEqual(_credibility(result), 0.8)

    def test_unknown_domain_scores_default_tier(self) -> None:
        result = _result(source_url="https://blog.unknown-site.io/post")
        self.assertEqual(_credibility(result), 0.5)

    def test_empty_url_scores_default_tier(self) -> None:
        result = _result(source_url="")
        self.assertEqual(_credibility(result), 0.5)


class RecencyDecayTest(unittest.TestCase):
    """<=1yr -> 1.0, <=3yr -> 0.7, older -> 0.4, unknown/malformed -> 0.6."""

    def test_recent_date_scores_full(self) -> None:
        self.assertEqual(_recency(_result(published_date=_days_ago(30))), 1.0)

    def test_one_year_boundary_scores_full(self) -> None:
        self.assertEqual(_recency(_result(published_date=_days_ago(365))), 1.0)

    def test_between_one_and_three_years_decays(self) -> None:
        self.assertEqual(_recency(_result(published_date=_days_ago(730))), 0.7)

    def test_older_than_three_years_decays_further(self) -> None:
        self.assertEqual(_recency(_result(published_date=_days_ago(365 * 4))), 0.4)

    def test_missing_date_is_neutral(self) -> None:
        self.assertEqual(_recency(_result(published_date=None)), 0.6)

    def test_malformed_date_is_neutral(self) -> None:
        self.assertEqual(_recency(_result(published_date="not-a-date")), 0.6)

    def test_full_iso_datetime_is_parsed(self) -> None:
        published = f"{_days_ago(10)}T08:30:00+00:00"
        self.assertEqual(_recency(_result(published_date=published)), 1.0)


class ScoreCombinationTest(unittest.TestCase):
    """Weighted relevance/credibility/recency combination, with clamping."""

    def test_weighted_combination_overwrites_confidence(self) -> None:
        result = _result(published_date=_days_ago(30))  # unknown web, recent
        state = {"research_results": [result], "run_id": "test-combo"}

        update = _run_critic(state, scores=[0.8])

        expected = _combined(0.8, 0.5, 1.0)
        critique = update["critique"]
        self.assertEqual(critique.scores, [expected])
        self.assertEqual(update["research_results"][0].confidence, expected)
        self.assertGreaterEqual(expected, CONFIDENCE_PASS_THRESHOLD)
        self.assertTrue(critique.overall_pass)
        self.assertEqual(critique.low_confidence_indices, [])

    def test_relevance_above_one_is_clamped(self) -> None:
        result = _result(
            source_type="arxiv",
            source_url="https://arxiv.org/pdf/2601.00042v1",
            published_date=_days_ago(30),
        )
        state = {"research_results": [result], "run_id": "test-clamp-high"}

        update = _run_critic(state, scores=[2.5])

        # Clamped relevance 1.0 + arXiv credibility 1.0 + recent 1.0 -> exactly 1.0.
        self.assertEqual(update["critique"].scores, [_combined(1.0, 1.0, 1.0)])
        self.assertEqual(update["critique"].scores, [1.0])

    def test_relevance_below_zero_is_clamped_and_flagged(self) -> None:
        result = _result(published_date=_days_ago(30))  # unknown web, recent
        state = {"research_results": [result], "run_id": "test-clamp-low"}

        update = _run_critic(state, scores=[-3.0])

        expected = _combined(0.0, 0.5, 1.0)
        critique = update["critique"]
        self.assertEqual(critique.scores, [expected])
        self.assertLess(expected, CONFIDENCE_PASS_THRESHOLD)
        self.assertEqual(critique.low_confidence_indices, [0])
        self.assertFalse(critique.overall_pass)

    def test_only_below_threshold_results_are_flagged(self) -> None:
        strong = _result(
            question="Strong Q",
            source_type="arxiv",
            source_url="https://arxiv.org/pdf/2601.00007v1",
            published_date=_days_ago(30),
        )
        weak = _result(question="Weak Q", published_date=_days_ago(30))
        state = {"research_results": [strong, weak], "run_id": "test-mixed"}

        update = _run_critic(state, scores=[0.9, 0.2])

        critique = update["critique"]
        self.assertEqual(critique.scores[0], _combined(0.9, 1.0, 1.0))
        self.assertEqual(critique.scores[1], _combined(0.2, 0.5, 1.0))
        self.assertEqual(critique.low_confidence_indices, [1])
        self.assertFalse(critique.overall_pass)


class ZeroResultFlaggingTest(unittest.TestCase):
    """Sub-questions with zero evidence must fail the critique, not vanish."""

    def test_unanswered_questions_fail_an_otherwise_passing_critique(self) -> None:
        strong = _result(
            source_type="arxiv",
            source_url="https://arxiv.org/pdf/2601.00099v1",
            published_date=_days_ago(30),
        )
        missing = "What sub-question got zero results?"
        state = {
            "research_results": [strong],
            "unanswered_questions": [missing],
            "run_id": "test-unanswered",
        }

        update = _run_critic(state, scores=[0.9])

        critique = update["critique"]
        # Every retrieved result is strong, yet the critique must fail because
        # one sub-question has no evidence at all.
        self.assertEqual(critique.low_confidence_indices, [])
        self.assertFalse(critique.overall_pass)
        self.assertIn(missing, critique.feedback)


class TotalSearchFailureTest(unittest.TestCase):
    """An empty result set fails the critique and retries while budget remains."""

    def test_empty_results_fail_critique_with_actionable_feedback(self) -> None:
        state = {"research_results": [], "run_id": "test-empty"}

        update = asyncio.run(critic_node(state))

        critique = update["critique"]
        self.assertFalse(critique.overall_pass)
        self.assertEqual(critique.scores, [])
        self.assertEqual(critique.low_confidence_indices, [])
        self.assertTrue(critique.feedback.strip())

    def test_empty_results_retry_while_budget_remains_then_proceed(self) -> None:
        update = asyncio.run(critic_node({"research_results": [], "run_id": "test-empty-edge"}))
        critique = update["critique"]

        # Budget remains -> loop back to the researcher.
        self.assertEqual(should_retry({"critique": critique, "retry_count": 0}), "researcher")
        # Budget exhausted -> hand off to the writer (stub report path).
        self.assertEqual(
            should_retry({"critique": critique, "retry_count": MAX_RETRIES}), "writer"
        )


if __name__ == "__main__":
    unittest.main()
