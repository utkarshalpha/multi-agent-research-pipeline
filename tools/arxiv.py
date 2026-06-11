"""arXiv search tool wrapper.

The ``arxiv`` library is synchronous and does blocking network I/O, so we run it
in a thread via ``asyncio.to_thread`` to preserve the async interface the
Researcher expects.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from config import ARXIV_MAX_RESULTS

try:
    import arxiv
except ImportError:  # pragma: no cover
    arxiv = None  # type: ignore[assignment]


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Blocking arXiv search executed inside a worker thread."""
    if arxiv is None:
        raise RuntimeError("arxiv is not installed. Run: pip install arxiv")

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results: list[dict[str, Any]] = []
    for paper in client.results(search):
        published = paper.published.date().isoformat() if paper.published else None
        results.append(
            {
                "title": paper.title.strip(),
                "summary": paper.summary.strip(),
                "pdf_url": paper.pdf_url or paper.entry_id,
                "published_date": published,
            }
        )
    return results


async def arxiv_search(query: str, max_results: int = ARXIV_MAX_RESULTS) -> list[dict[str, Any]]:
    """Search arXiv and return normalised paper metadata.

    Args:
        query: The search query (typically a sub-question).
        max_results: Maximum number of papers to return (default 3).

    Returns:
        A list of dicts with keys ``title``, ``summary``, ``pdf_url`` and
        ``published_date``. Returns an empty list on failure.
    """
    try:
        results = await asyncio.to_thread(_search_sync, query, max_results)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any API error
        logger.error("arXiv search failed for query {!r}: {}", query, exc)
        return []

    logger.debug("arXiv returned {} papers for {!r}", len(results), query)
    return results
