"""Pydantic v2 models for every structured payload in the pipeline.

These are the contracts shared between agents, the graph state, the tools, and
the FastAPI layer. Keeping them in one place means a schema change is a
single-file edit.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SubQuestions(BaseModel):
    """The Planner's decomposition of a query into focused sub-questions."""

    questions: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Between 3 and 5 focused sub-questions that together fully "
        "answer the main query.",
    )


class ResearchResult(BaseModel):
    """A single piece of evidence gathered for one sub-question.

    One sub-question typically yields several ``ResearchResult`` objects (e.g.
    one per web hit and one per arXiv paper).
    """

    question: str = Field(..., description="The sub-question this result answers.")
    content: str = Field(..., description="The retrieved content / summary.")
    source_url: str = Field(..., description="Canonical URL of the source.")
    source_type: Literal["web", "arxiv"] = Field(
        ..., description="Where the result came from."
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score. Set provisionally by the Researcher and "
        "finalised by the Critic.",
    )
    # Not in the original spec, but required for the Critic's recency scoring.
    published_date: Optional[str] = Field(
        None, description="ISO-8601 publication date of the source, if known."
    )
    title: Optional[str] = Field(None, description="Human-readable source title.")


class CritiqueResult(BaseModel):
    """The Critic's verdict over the current set of research results."""

    scores: list[float] = Field(
        default_factory=list,
        description="Combined confidence score per ResearchResult, index-aligned.",
    )
    low_confidence_indices: list[int] = Field(
        default_factory=list,
        description="Indices into the research-results list whose score < threshold.",
    )
    overall_pass: bool = Field(
        False,
        description="True only if at least one result exists, every result "
        "scores >= the pass threshold, and every sub-question has evidence. "
        "An empty result set fails so total search failure triggers a retry.",
    )
    feedback: str = Field(
        "",
        description="Natural-language guidance consumed by the Researcher on "
        "a retry to reformulate the search queries for the weak sub-questions.",
    )


class ReportSection(BaseModel):
    """One section of the final Markdown report."""

    heading: str = Field(..., description="Section heading.")
    content: str = Field(
        ...,
        description="Section body in Markdown, with inline [n] citations "
        "referencing the sources list.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Source URLs cited in this section.",
    )


class FinalReport(BaseModel):
    """The Writer's synthesised, fully-structured report."""

    title: str = Field(..., description="Report title.")
    summary: str = Field(..., description="Executive summary / abstract.")
    sections: list[ReportSection] = Field(
        ..., description="Ordered body sections."
    )
    all_citations: list[str] = Field(
        default_factory=list,
        description="De-duplicated list of every source URL cited, in [n] order.",
    )


# --------------------------------------------------------------------------- #
# HTTP API contracts (FastAPI layer)
# --------------------------------------------------------------------------- #
class ResearchRequest(BaseModel):
    """Body for ``POST /research``."""

    query: str = Field(..., min_length=3, description="The research question.")


class ResearchResponse(BaseModel):
    """Response for ``POST /research`` — the main portfolio-facing contract."""

    run_id: str
    report: str
    citations: list[str]
    metadata: dict[str, Any]
