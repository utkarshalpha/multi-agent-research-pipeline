"""Conditional edge logic for the self-healing critic loop."""

from __future__ import annotations

from loguru import logger

from config import MAX_RETRIES
from graph.state import AgentState


def should_retry(state: AgentState) -> str:
    """Decide whether to loop back to the Researcher or proceed to the Writer.

    Pure function of state: retry iff the critique failed AND retry budget
    remains. The Critic fails the critique when any result scores below the
    threshold, when any sub-question has zero results, or when nothing was
    retrieved at all — so total search failure also loops back here until the
    budget is exhausted, after which the Writer produces its (possibly stub)
    report.

    Args:
        state: Current graph state; reads ``critique`` and ``retry_count``.

    Returns:
        ``"researcher"`` to retry, or ``"writer"`` to finish.
    """
    critique = state.get("critique")
    retry_count = state.get("retry_count", 0)

    if critique is not None and not critique.overall_pass and retry_count < MAX_RETRIES:
        logger.info(
            "[edge] critique failed (retry {}/{}) -> researcher", retry_count, MAX_RETRIES
        )
        return "researcher"

    logger.info("[edge] proceeding -> writer (retry_count={})", retry_count)
    return "writer"
