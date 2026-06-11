"""Conditional edge logic for the self-healing critic loop."""

from __future__ import annotations

from loguru import logger

from config import MAX_RETRIES
from graph.state import AgentState


def should_retry(state: AgentState) -> str:
    """Decide whether to loop back to the Researcher or proceed to the Writer.

    The loop is self-healing: if the Critic failed any result and we still have
    retries left, we send control back to the Researcher to re-gather evidence
    for the weak sub-questions. Otherwise we hand off to the Writer.

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
