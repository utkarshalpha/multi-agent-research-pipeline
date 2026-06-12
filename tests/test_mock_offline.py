"""Determinism tests for the MOCK_MODE fixtures (offline by construction).

Proving "no network calls" directly is impractical, so these tests assert the
contract that makes mock mode trustworthy instead: every mock is a pure,
hash-derived function of its inputs. The same query must yield byte-identical
Tavily/arXiv results and an identical embedding vector on every call, the tool
wrappers must route to the mocks when MOCK_MODE is on, and the mock critic LLM
must honour the ``force-retry`` scoring hook.
"""

from __future__ import annotations

import os

# config.py reads the environment at import time — these must be set before any
# project module is imported (load_dotenv() never overrides existing vars).
os.environ["MOCK_MODE"] = "true"
os.environ["QDRANT_PATH"] = ":memory:"

import asyncio
import math
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from config import get_llm
from schemas.models import SubQuestions
from tools.arxiv import arxiv_search
from tools.mocks import (
    MockChatModel,
    mock_arxiv_results,
    mock_embedding,
    mock_tavily_results,
)
from tools.search import tavily_search

_QUERY = "What are the latest advances in solid-state batteries?"


class _Assessment(BaseModel):
    """Critic-shaped schema used to exercise the mock relevance hook."""

    relevance_scores: list[float]
    feedback: str


class MockDeterminismTest(unittest.TestCase):
    """Same input twice -> identical output, with the documented shapes."""

    def test_tavily_mock_is_deterministic(self) -> None:
        first = mock_tavily_results(_QUERY)
        second = mock_tavily_results(_QUERY)

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 2)
        for hit in first:
            self.assertEqual(
                set(hit), {"url", "title", "content", "published_date"}
            )
            self.assertTrue(hit["url"].startswith("https://example.org/research/"))
            self.assertIn(_QUERY, hit["content"])

    def test_arxiv_mock_is_deterministic(self) -> None:
        first = mock_arxiv_results(_QUERY)
        second = mock_arxiv_results(_QUERY)

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 1)
        for paper in first:
            self.assertEqual(
                set(paper), {"title", "summary", "pdf_url", "published_date"}
            )
            self.assertTrue(paper["pdf_url"].startswith("https://arxiv.org/pdf/"))

    def test_embedding_is_deterministic_unit_vector(self) -> None:
        first = mock_embedding(_QUERY)
        second = mock_embedding(_QUERY)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 384)
        norm = math.sqrt(sum(v * v for v in first))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_distinct_inputs_produce_distinct_outputs(self) -> None:
        other = "How do coral reefs recover from bleaching events?"
        self.assertNotEqual(mock_embedding(_QUERY), mock_embedding(other))
        self.assertNotEqual(mock_tavily_results(_QUERY), mock_tavily_results(other))
        self.assertNotEqual(mock_arxiv_results(_QUERY), mock_arxiv_results(other))


class MockRoutingTest(unittest.TestCase):
    """The real tool wrappers must serve the canned results in MOCK_MODE."""

    def test_tavily_search_returns_mock_results(self) -> None:
        results = asyncio.run(tavily_search(_QUERY))
        self.assertEqual(results, mock_tavily_results(_QUERY)[: len(results)])
        self.assertGreater(len(results), 0)

    def test_arxiv_search_returns_mock_results(self) -> None:
        results = asyncio.run(arxiv_search(_QUERY))
        self.assertEqual(results, mock_arxiv_results(_QUERY)[: len(results)])
        self.assertGreater(len(results), 0)

    def test_get_llm_returns_mock_chat_model(self) -> None:
        self.assertIsInstance(get_llm(), MockChatModel)


class MockChatModelTest(unittest.TestCase):
    """Structured + plain mock LLM responses are deterministic and labelled."""

    def _messages(self) -> list:
        return [
            SystemMessage(content="You are a research planning expert."),
            HumanMessage(content=f"Main query: {_QUERY}\n\nDecompose this."),
        ]

    def test_structured_output_is_deterministic(self) -> None:
        runnable = get_llm().with_structured_output(SubQuestions)

        first = asyncio.run(runnable.ainvoke(self._messages()))
        second = asyncio.run(runnable.ainvoke(self._messages()))

        self.assertIsInstance(first, SubQuestions)
        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertGreaterEqual(len(first.questions), 3)
        for question in first.questions:
            self.assertIn(_QUERY, question)

    def test_plain_ainvoke_returns_mock_labelled_aimessage(self) -> None:
        message = asyncio.run(get_llm().ainvoke(self._messages()))
        repeat = asyncio.run(get_llm().ainvoke(self._messages()))

        self.assertIsInstance(message, AIMessage)
        self.assertIn("mock (", message.content)
        self.assertEqual(message.content, repeat.content)

    def test_force_retry_marker_drives_low_relevance(self) -> None:
        runnable = get_llm().with_structured_output(_Assessment)
        prompt = (
            "Rate the relevance of these 2 results. Return exactly 2 scores in "
            "order.\n\n"
            "[0] Sub-question: What is force-retry behaviour?\n"
            "    Source (web): Example\n"
            "    Content: force-retry evidence snippet\n\n"
            "[1] Sub-question: What is force-retry behaviour?\n"
            "    Source (arxiv): Example paper\n"
            "    Content: more force-retry evidence"
        )

        low = asyncio.run(runnable.ainvoke([HumanMessage(content=prompt)]))
        self.assertEqual(len(low.relevance_scores), 2)
        for score in low.relevance_scores:
            self.assertLess(score, 0.3)

        clean = asyncio.run(
            runnable.ainvoke([HumanMessage(content=prompt.replace("force-retry", "ordinary"))])
        )
        self.assertEqual(len(clean.relevance_scores), 2)
        for score in clean.relevance_scores:
            self.assertGreater(score, 0.6)


if __name__ == "__main__":
    unittest.main()
