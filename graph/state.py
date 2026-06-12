"""The shared graph state passed between every LangGraph node.

Each node returns a partial dict; LangGraph merges it into this ``AgentState``.
No custom reducers are needed because every field is fully replaced by the node
that owns it (the Researcher, for example, returns the complete merged results
list rather than a delta).
"""

from __future__ import annotations

from typing import Optional, TypedDict

from schemas.models import CritiqueResult, ResearchResult


class AgentState(TypedDict, total=False):
    """State threaded through Planner -> Researcher -> Critic -> Writer."""

    query: str
    sub_questions: list[str]
    research_results: list[ResearchResult]
    critique: Optional[CritiqueResult]
    final_report: str
    retry_count: int
    run_id: str
    # Sub-questions whose searches returned zero results (owned by the
    # Researcher). They join the retry set, and if they still have nothing
    # once retries are exhausted they are surfaced here instead of being
    # silently dropped.
    unanswered_questions: list[str]
