"""Critic agent: scores every research result and decides whether to retry.

Each result is scored on three dimensions:

* **Relevance** to its sub-question — judged by the LLM (the subjective part).
* **Source credibility** — computed deterministically (arXiv=1.0, known
  outlets=0.8, unknown=0.5).
* **Recency** — computed deterministically from the publication date.

The three are combined into a single confidence per result. ``overall_pass`` is
True only when *every* result clears the pass threshold AND every sub-question
has at least one result. An empty result set always fails the critique so that
total search failure triggers the retry loop; the budget gate lives in
``graph.edges.should_retry``, which routes to the writer once retries run out.
"""

from __future__ import annotations

from datetime import date, datetime
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from loguru import logger
from pydantic import BaseModel, Field

from config import CONFIDENCE_PASS_THRESHOLD, ainvoke_with_retry, get_llm
from graph.state import AgentState
from memory.redis_store import redis_store
from schemas.models import CritiqueResult, ResearchResult

# Dimension weights for the combined confidence score.
_W_RELEVANCE = 0.5
_W_CREDIBILITY = 0.25
_W_RECENCY = 0.25

# Reputable domains earn the 0.8 credibility tier; everything else gets 0.5.
_KNOWN_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "wsj.com", "bloomberg.com", "ft.com", "theguardian.com", "economist.com",
    "nature.com", "science.org", "sciencedirect.com", "ieee.org", "acm.org",
    "nih.gov", "who.int", "cdc.gov", "europa.eu", "oecd.org", "imf.org",
    "worldbank.org", "federalreserve.gov", "sec.gov", "techcrunch.com",
    "arstechnica.com", "wired.com", "mit.edu", "stanford.edu", "harvard.edu",
}

SYSTEM_PROMPT = (
    "You are a meticulous research critic. For each numbered research result, "
    "rate how RELEVANT its content is to the stated sub-question on a 0.0-1.0 "
    "scale (1.0 = directly and completely answers it, 0.0 = irrelevant). "
    "Return one score per result, in order, plus concise feedback the "
    "researcher can act on if any results are weak."
)


class _RelevanceAssessment(BaseModel):
    """Structured relevance judgement returned by the LLM."""

    relevance_scores: list[float] = Field(
        ..., description="One relevance score in [0,1] per result, in order."
    )
    feedback: str = Field(..., description="Actionable feedback for the researcher.")


def _credibility(result: ResearchResult) -> float:
    """Score source credibility: arXiv=1.0, known outlet=0.8, unknown=0.5."""
    if result.source_type == "arxiv":
        return 1.0
    domain = urlparse(result.source_url).netloc.lower().removeprefix("www.")
    if not domain:
        return 0.5
    if domain in _KNOWN_DOMAINS or domain.endswith((".gov", ".edu")):
        return 0.8
    return 0.5


def _recency(result: ResearchResult) -> float:
    """Score recency from the publication date: <=1yr=1.0, <=3yr=0.7, else=0.4."""
    if not result.published_date:
        return 0.6  # neutral default when the date is unknown
    try:
        published = datetime.fromisoformat(result.published_date).date()
    except (ValueError, TypeError):
        try:
            published = date.fromisoformat(result.published_date[:10])
        except (ValueError, TypeError):
            return 0.6
    age_days = (date.today() - published).days
    if age_days <= 365:
        return 1.0
    if age_days <= 365 * 3:
        return 0.7
    return 0.4


async def _assess_relevance(results: list[ResearchResult]) -> _RelevanceAssessment:
    """Ask the LLM to rate the relevance of each result to its sub-question."""
    lines = []
    for i, r in enumerate(results):
        snippet = r.content[:800].replace("\n", " ")
        lines.append(
            f"[{i}] Sub-question: {r.question}\n"
            f"    Source ({r.source_type}): {r.title or r.source_url}\n"
            f"    Content: {snippet}"
        )
    payload = "\n\n".join(lines)

    structured_llm = get_llm().with_structured_output(_RelevanceAssessment)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Rate the relevance of these {len(results)} results. Return exactly "
                f"{len(results)} scores in order.\n\n{payload}"
            )
        ),
    ]
    return await ainvoke_with_retry(structured_llm, messages)


@traceable(name="critic", run_type="chain")
async def critic_node(state: AgentState) -> dict:
    """Score the research results and produce a critique.

    Args:
        state: Current graph state; must contain ``research_results``.

    Returns:
        A partial state update with ``critique`` populated and each result's
        ``confidence`` overwritten with the final combined score.
    """
    results: list[ResearchResult] = state.get("research_results", [])
    unanswered: list[str] = state.get("unanswered_questions", [])
    run_id = state.get("run_id", "")

    if not results:
        # Total search failure must FAIL so the loop retries with reformulated
        # queries; should_retry hands off to the writer once the budget is spent.
        logger.warning("[critic] no research results at all — failing critique to trigger a retry")
        critique = CritiqueResult(
            scores=[],
            low_confidence_indices=[],
            overall_pass=False,
            feedback=(
                "No evidence was retrieved for any sub-question. Reformulate "
                "each search with more specific terminology, synonyms, or "
                "broader phrasing so at least one usable source is found."
            ),
        )
        await redis_store.save(run_id, "critique", critique.model_dump())
        return {"critique": critique}

    logger.info("[critic] scoring {} research results", len(results))

    # Subjective dimension via the LLM.
    try:
        assessment = await _assess_relevance(results)
        relevance_scores = list(assessment.relevance_scores)
        feedback = assessment.feedback
    except Exception as exc:  # noqa: BLE001 - never let scoring crash the run
        logger.error("[critic] relevance assessment failed, defaulting to 0.6: {}", exc)
        relevance_scores = []
        feedback = "Relevance assessment unavailable; scored on credibility and recency only."

    # Defend against length mismatch from the LLM.
    if len(relevance_scores) < len(results):
        relevance_scores += [0.6] * (len(results) - len(relevance_scores))
    relevance_scores = relevance_scores[: len(results)]

    scores: list[float] = []
    low_confidence_indices: list[int] = []
    for i, result in enumerate(results):
        relevance = max(0.0, min(1.0, float(relevance_scores[i])))
        credibility = _credibility(result)
        recency = _recency(result)
        combined = round(
            _W_RELEVANCE * relevance + _W_CREDIBILITY * credibility + _W_RECENCY * recency, 3
        )
        result.confidence = combined  # finalise the result's confidence in place
        scores.append(combined)
        if combined < CONFIDENCE_PASS_THRESHOLD:
            low_confidence_indices.append(i)
        logger.debug(
            "[critic] [{}] rel={:.2f} cred={:.2f} rec={:.2f} -> {:.3f}",
            i, relevance, credibility, recency, combined,
        )

    # Zero-coverage sub-questions also fail the critique so they re-enter the
    # researcher's retry set instead of being silently dropped.
    overall_pass = not low_confidence_indices and not unanswered
    if unanswered:
        feedback = (
            f"{feedback} Additionally, {len(unanswered)} sub-question(s) "
            f"returned no results and need fresh evidence: {'; '.join(unanswered)}"
        ).strip()
    critique = CritiqueResult(
        scores=scores,
        low_confidence_indices=low_confidence_indices,
        overall_pass=overall_pass,
        feedback=feedback,
    )
    logger.info(
        "[critic] overall_pass={} | {}/{} results below threshold | {} unanswered sub-question(s)",
        overall_pass, len(low_confidence_indices), len(results), len(unanswered),
    )

    await redis_store.save(run_id, "critique", critique.model_dump())
    return {"critique": critique, "research_results": results}
