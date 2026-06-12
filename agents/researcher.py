"""Researcher agent: gathers evidence for each sub-question.

For every sub-question it first consults the Qdrant semantic cache. On a cache
miss it runs Tavily and arXiv searches in parallel, normalises the hits into
``ResearchResult`` objects, and writes them back to the cache.

On a retry (triggered by the Critic), it re-researches only the low-confidence
and zero-result sub-questions, bypassing the cache so it actually fetches fresh
evidence, and keeps the previously-approved results untouched. Each retried
query is *reformulated* using the Critic's feedback (deterministic keyword
augmentation — works identically in MOCK_MODE), so the search tools see a
different query than on the previous attempt. Sub-questions that still have no
evidence are reported via the ``unanswered_questions`` state field rather than
being silently dropped.
"""

from __future__ import annotations

import asyncio
import re

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

# Words ignored when distilling the Critic's feedback into query-refinement
# terms: short function words plus generic critique vocabulary.
_FEEDBACK_STOPWORDS: frozenset[str] = frozenset({
    "about", "additionally", "address", "addresses", "adequately", "all",
    "and", "answer", "answered", "answers", "any", "are", "but", "can",
    "confidence", "could", "did", "does", "each", "evidence", "feedback",
    "for", "found", "fresh", "from", "gather", "had", "has", "have", "into",
    "irrelevant", "low", "may", "might", "more", "most", "must", "need",
    "needs", "not", "onto", "please", "provide", "quality", "relevance",
    "relevant", "research", "researcher", "result", "results", "retrieve",
    "retrieved", "returned", "score", "scores", "search", "searches", "should",
    "some", "source", "sources", "sub-question", "sub-questions", "such",
    "than", "that", "the", "their", "them", "then", "these", "they", "this",
    "those", "try", "weak", "were", "will", "with", "would",
})
# Used when the feedback yields no usable refinement terms.
_FALLBACK_REFINEMENT_TERMS: tuple[str, ...] = ("detailed", "authoritative", "primary")


def _refinement_terms(feedback: str, max_terms: int = 4) -> list[str]:
    """Distil the Critic's feedback into a few content-bearing keywords.

    Args:
        feedback: The Critic's natural-language feedback.
        max_terms: Maximum number of keywords to keep.

    Returns:
        Deduplicated, order-preserving lowercase keywords (possibly empty).
    """
    terms: list[str] = []
    for word in re.findall(r"[a-z][a-z\-]+", feedback.lower()):
        if len(word) < 4 or word in _FEEDBACK_STOPWORDS or word in terms:
            continue
        terms.append(word)
        if len(terms) == max_terms:
            break
    return terms


def _reformulate_query(question: str, feedback: str, attempt: int) -> str:
    """Reformulate a retried search query using the Critic's feedback.

    Deterministic (no LLM call, no randomness — identical in MOCK_MODE):
    feedback-derived refinement terms are appended to the sub-question, rotated
    by attempt number so every retry sends a query to Tavily/arXiv that differs
    from the first attempt and from earlier retries.

    Args:
        question: The canonical sub-question (left unchanged in the results).
        feedback: The Critic's feedback driving the refinement.
        attempt: 1-based retry attempt number.

    Returns:
        The refined query string to send to the search tools.
    """
    terms = _refinement_terms(feedback) or list(_FALLBACK_REFINEMENT_TERMS)
    rotation = max(attempt - 1, 0) % len(terms)
    rotated = terms[rotation:] + terms[:rotation]
    return f"{question} {' '.join(rotated)}"


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


async def _research_question(
    question: str, search_query: str | None = None, use_cache: bool = True
) -> list[ResearchResult]:
    """Research a single sub-question, consulting the cache first.

    Args:
        question: The canonical sub-question (bound to every result it yields).
        search_query: The query actually sent to Tavily/arXiv — on a retry this
            is the Critic-refined reformulation. Defaults to ``question``.
        use_cache: If True, return cached results on a hit and skip fetching.

    Returns:
        A list of ``ResearchResult`` for the question (possibly empty).
    """
    search_query = search_query or question
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
    if search_query != question:
        logger.info("[researcher] refined query for {!r} -> {!r}", question, search_query)
    logger.info("[researcher] cache MISS for {!r} — querying Tavily + arXiv", question)
    web_hits, arxiv_papers = await asyncio.gather(
        tavily_search(search_query),
        arxiv_search(search_query, max_results=ARXIV_MAX_RESULTS),
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
        A partial state update with the merged ``research_results``, the
        (possibly incremented) ``retry_count``, and the list of
        ``unanswered_questions`` that still have zero evidence.
    """
    sub_questions: list[str] = state["sub_questions"]
    run_id = state.get("run_id", "")
    critique: CritiqueResult | None = state.get("critique")
    previous_results: list[ResearchResult] = state.get("research_results", [])
    previously_unanswered: list[str] = state.get("unanswered_questions", [])

    is_retry = critique is not None and not critique.overall_pass
    retry_count = state.get("retry_count", 0) + (1 if is_retry else 0)

    if is_retry:
        # Retry set = sub-questions whose results scored too low, plus the ones
        # whose searches previously returned nothing at all.
        low_questions = {
            previous_results[i].question
            for i in critique.low_confidence_indices
            if 0 <= i < len(previous_results)
        }
        retry_questions = low_questions | set(previously_unanswered)
        # Keep the results we were happy with.
        kept = [r for r in previous_results if r.question not in retry_questions]
        questions_to_research = [q for q in sub_questions if q in retry_questions]
        if not questions_to_research:
            # Defensive: the critique failed but nothing mapped back to a
            # sub-question — re-research everything rather than spinning idle.
            kept = []
            questions_to_research = list(sub_questions)
        # Critic feedback drives the retry: each query is reformulated so the
        # search tools receive something different from the previous attempt.
        queries = {
            q: _reformulate_query(q, critique.feedback, retry_count)
            for q in questions_to_research
        }
        use_cache = False  # bypass cache so we fetch genuinely fresh evidence
        logger.info(
            "[researcher] retry #{} — re-researching {} question(s) "
            "({} low-confidence, {} unanswered) with critic-refined queries",
            retry_count,
            len(questions_to_research),
            len(low_questions),
            len(previously_unanswered),
        )
    else:
        logger.info("[researcher] initial research over {} sub-questions", len(sub_questions))
        kept = []
        questions_to_research = list(sub_questions)
        queries = {q: q for q in questions_to_research}
        use_cache = True

    fetched = await asyncio.gather(
        *(
            _research_question(q, search_query=queries[q], use_cache=use_cache)
            for q in questions_to_research
        )
    )
    new_results = [r for batch in fetched for r in batch]
    merged = kept + new_results

    # Sub-questions with zero evidence are tracked (not silently dropped): they
    # join the next retry set, and if retries run out the state still surfaces
    # them to the writer/API layer.
    answered = {r.question for r in merged}
    unanswered = [q for q in sub_questions if q not in answered]
    if unanswered:
        logger.warning(
            "[researcher] {} sub-question(s) still have no evidence: {}",
            len(unanswered),
            unanswered,
        )

    logger.info("[researcher] total results after merge: {}", len(merged))
    await redis_store.save(run_id, "research_results", [r.model_dump() for r in merged])
    await redis_store.save(run_id, "retry_count", retry_count)
    await redis_store.save(run_id, "unanswered_questions", unanswered)

    return {
        "research_results": merged,
        "retry_count": retry_count,
        "unanswered_questions": unanswered,
    }
