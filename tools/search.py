"""Tavily web-search tool wrapper.

Returns a normalised list of dicts so the Researcher never has to care about
Tavily's raw response shape. In ``MOCK_MODE`` deterministic canned results are
returned instead — no network call, no ``TAVILY_API_KEY`` needed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import MOCK_MODE, TAVILY_MAX_RESULTS, TAVILY_TIMEOUT_SECONDS

try:  # Imported lazily-friendly: the package may be absent in some envs.
    from tavily import AsyncTavilyClient
except ImportError:  # pragma: no cover
    AsyncTavilyClient = None  # type: ignore[assignment]


_client: "AsyncTavilyClient | None" = None


def _get_client() -> "AsyncTavilyClient":
    """Return a lazily-instantiated singleton ``AsyncTavilyClient``.

    The client reads ``TAVILY_API_KEY`` from the environment.
    """
    global _client
    if AsyncTavilyClient is None:
        raise RuntimeError("tavily-python is not installed. Run: pip install tavily-python")
    if _client is None:
        _client = AsyncTavilyClient()  # picks up TAVILY_API_KEY from env
    return _client


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
)
async def _search_with_retry(
    client: "AsyncTavilyClient", query: str, max_results: int
) -> dict[str, Any]:
    """Issue one Tavily API call, retrying transient failures up to 3 attempts.

    This inner function deliberately *raises* on failure (timeouts included) so
    the tenacity decorator actually fires; the public wrapper converts
    exhausted retries into an empty result list.
    """
    return await asyncio.wait_for(
        client.search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
            search_depth="advanced",
        ),
        timeout=TAVILY_TIMEOUT_SECONDS,
    )


async def tavily_search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict[str, Any]]:
    """Run a Tavily web search and return normalised results.

    Args:
        query: The search query (typically a sub-question).
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of dicts with keys ``url``, ``title``, ``content`` and
        ``published_date``. Returns an empty list only after retries are
        exhausted, so the pipeline degrades gracefully rather than crashing.
    """
    if MOCK_MODE:
        from tools.mocks import mock_tavily_results  # local import: mock-only dep

        results = mock_tavily_results(query)[:max_results]
        logger.debug("MOCK_MODE: returning {} canned Tavily results for {!r}", len(results), query)
        return results

    try:
        client = _get_client()
    except RuntimeError as exc:  # missing package — not transient, do not retry
        logger.error("Tavily client unavailable: {}", exc)
        return []

    try:
        response = await _search_with_retry(client, query, max_results)
    except asyncio.TimeoutError:
        logger.warning(
            "Tavily search timed out after {}s (retries exhausted) for query: {!r}",
            TAVILY_TIMEOUT_SECONDS,
            query,
        )
        return []
    except Exception as exc:  # noqa: BLE001 - degrade gracefully after retries
        logger.error("Tavily search failed after retries for query {!r}: {}", query, exc)
        return []

    results: list[dict[str, Any]] = []
    for item in response.get("results", []):
        # Prefer the fuller raw content when available, falling back to the snippet.
        content = item.get("raw_content") or item.get("content") or ""
        results.append(
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": content[:4000],  # keep prompts bounded
                "published_date": item.get("published_date"),
            }
        )
    logger.debug("Tavily returned {} results for {!r}", len(results), query)
    return results
