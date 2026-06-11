"""Researcher agent: gathers evidence for each sub-question.

For every sub-question it first consults the Qdrant semantic cache. On a cache
miss it runs Tavily and arXiv searches in parallel, normalises the hits into
``ResearchResult`` objects, and writes them back to the cache.

On a retry (triggered by the Critic), it re-researches only the low-confidence
sub-questions, bypassing the cache so it actually fetches fresh evidence, and
keeps the previously-approved results untouched.
"""

from __future__ import annotations

import asyncio

from langsmith import traceable
from loguru import logger

from config import ARXIV_MAX_RESULTS
from graph.state import AgentState
from memory.redis_store import redis_store
from schemas.models import CritiqueResult, ResearchResult
from tools.arxiv import arxiv_search
from tools.search import tavily_search
from tools.vector_store import embed_text, search as cache_search, upsert as cache_upsert

# How many sources of each type to keep per sub-question (keeps prompts bounded).
_MAX_WEB_PER_QUESTION = 2
_MAX_ARXIV_PER_QUESTION = 2
# Provisional confidence before the Critic scores; arXiv is treated as stronger.
_WEB_PROVISIONAL_CONFIDENCE = 0.5
_ARXIV_PROVISIONAL_CONFIDENCE = 0.7


def _build_web_results(question: str, hits: list[dict]) -> list[ResearchResult]:
    """Convert Tavily hits into ``ResearchResult`` objects."""
    results: list[ResearchResult] = []
    for hit in hits[:_MAX_WEB_PER_QUESTION]:
        if not hit.get("content"):
            continue
        results.append(
            ResearchResult(
                question=question,
                content=hit["content"],
                source_url=hit.get("url", ""),
                source_type="web",
                confidence=_WEB_PROVISIONAL_CONFIDENCE,
                published_date=hit.get("published_date"),
                title=hit.get("title"),
            )
        )
    return results


def _build_arxiv_results(question: str, papers: list[dict]) -> list[ResearchResult]:
    """Convert arXiv papers into ``ResearchResult`` objects."""
    results: list[ResearchResult] = []
    for paper in papers[:_MAX_ARXIV_PER_QUESTION]:
        results.append(
            ResearchResult(
                question=question,
                content=paper.get("summary", ""),
                source_url=paper.get("pdf_url", ""),
                source_type="arxiv",
                confidence=_ARXIV_PROVISIONAL_CONFIDENCE,
                published_date=paper.get("published_date"),
                title=paper.get("title"),
            )
        )
    return results


async def _research_question(question: str, use_cache: bool = True) -> list[ResearchResult]:
    """Research a single sub-question, consulting the cache first.

    Args:
        question: The sub-question to research.
        use_cache: If True, return cached results on a hit and skip fetching.

    Returns:
        A list of ``ResearchResult`` for the question (possibly empty).
    """
    # 1. Semantic-cache lookup.
    if use_cache:
        try:
            embedding = await embed_text(question)
            cached = await cache_search(embedding)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[researcher] cache lookup failed for {!r}: {}", question, exc)
            cached = []
        if cached:
            logger.info("[researcher] cache HIT for {!r} ({} results)", question, len(cached))
            # Re-bind to the current sub-question so downstream scoring aligns.
            for result in cached:
                result.question = question
            return cached

    # 2. Cache miss (or bypass): fetch from both sources in parallel.
    logger.info("[researcher] cache MISS for {!r} — querying Tavily + arXiv", question)
    web_hits, arxiv_papers = await asyncio.gather(
        tavily_search(question),
        arxiv_search(question, max_results=ARXIV_MAX_RESULTS),
        return_exceptions=True,
    )
    if isinstance(web_hits, Exception):
        logger.error("[researcher] Tavily error for {!r}: {}", question, web_hits)
        web_hits = []
    if isinstance(arxiv_papers, Exception):
        logger.error("[researcher] arXiv error for {!r}: {}", question, arxiv_papers)
        arxiv_papers = []

    results = _build_web_results(question, web_hits) + _build_arxiv_results(question, arxiv_papers)

    # 3. Write fresh results back to the cache (best-effort).
    if results:
        question_embedding = await embed_text(question)
        await asyncio.gather(
            *(cache_upsert(r, question_embedding) for r in results),
            return_exceptions=True,
        )
    else:
        logger.warning("[researcher] no results found for {!r}", question)
    return results


@traceable(name="researcher", run_type="chain")
async def researcher_node(state: AgentState) -> dict:
    """Gather (or refresh) evidence for the planned sub-questions.

    Args:
        state: Current graph state; must contain ``sub_questions``.

    Returns:
        A partial state update with the merged ``research_results`` and the
        (possibly incremented) ``retry_count``.
    """
    sub_questions: list[str] = state["sub_questions"]
    run_id = state.get("run_id", "")
    critique: CritiqueResult | None = state.get("critique")
    previous_results: list[ResearchResult] = state.get("research_results", [])

    is_retry = critique is not None and not critique.overall_pass
    retry_count = state.get("retry_count", 0) + (1 if is_retry else 0)

    if is_retry:
        # Identify the sub-questions whose results scored too low.
        low_questions = {
            previous_results[i].question
            for i in critique.low_confidence_indices
            if 0 <= i < len(previous_results)
        }
        logger.info(
            "[researcher] retry #{} — re-researching {} low-confidence question(s)",
            retry_count,
            len(low_questions),
        )
        # Keep the results we were happy with.
        kept = [r for r in previous_results if r.question not in low_questions]
        questions_to_research = [q for q in sub_questions if q in low_questions]
        use_cache = False  # bypass cache so we fetch genuinely fresh evidence
    else:
        logger.info("[researcher] initial research over {} sub-questions", len(sub_questions))
        kept = []
        questions_to_research = sub_questions
        use_cache = True

    fetched = await asyncio.gather(
        *(_research_question(q, use_cache=use_cache) for q in questions_to_research)
    )
    new_results = [r for batch in fetched for r in batch]
    merged = kept + new_results

    logger.info("[researcher] total results after merge: {}", len(merged))
    await redis_store.save(run_id, "research_results", [r.model_dump() for r in merged])
    await redis_store.save(run_id, "retry_count", retry_count)

    return {"research_results": merged, "retry_count": retry_count}
