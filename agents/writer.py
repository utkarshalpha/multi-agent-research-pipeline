"""Writer agent: synthesises approved research into a cited Markdown report."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from loguru import logger

from config import CONFIDENCE_PASS_THRESHOLD, LLM_MAX_TOKENS, ainvoke_with_retry, get_llm
from graph.state import AgentState
from memory.redis_store import redis_store
from schemas.models import FinalReport, ReportSection, ResearchResult

# Keep the synthesis prompt bounded.
_MAX_SOURCES = 12
_MAX_CONTENT_CHARS = 1200

SYSTEM_PROMPT = (
    "You are an expert research writer. Synthesise the provided research into a "
    "clear, well-structured report. Every factual claim MUST carry an inline "
    "citation like [1], [2] that matches the numbered sources list you are "
    "given. Do not invent sources or citation numbers. Populate `all_citations` "
    "with the source URLs you actually cited, ordered to match your [n] markers."
)


def to_markdown(report: FinalReport) -> str:
    """Render a ``FinalReport`` as clean Markdown with a References section.

    Args:
        report: The structured report.

    Returns:
        A Markdown string ready for storage or display.
    """
    parts: list[str] = [f"# {report.title}", "", report.summary, ""]
    for section in report.sections:
        parts.append(f"## {section.heading}")
        parts.append("")
        parts.append(section.content)
        parts.append("")
    if report.all_citations:
        parts.append("## References")
        parts.append("")
        for i, url in enumerate(report.all_citations, start=1):
            parts.append(f"{i}. {url}")
    return "\n".join(parts).strip() + "\n"


def _select_sources(results: list[ResearchResult]) -> list[ResearchResult]:
    """Pick the strongest results to feed the writer.

    Prefers results that cleared the pass threshold; falls back to the best
    available (highest confidence) so the report is never empty after a
    retry-exhausted run.
    """
    approved = [r for r in results if r.confidence >= CONFIDENCE_PASS_THRESHOLD]
    pool = approved or results
    pool = sorted(pool, key=lambda r: r.confidence, reverse=True)
    return pool[:_MAX_SOURCES]


def _format_sources(sources: list[ResearchResult]) -> str:
    """Build the numbered source block passed to the writer."""
    lines = []
    for i, r in enumerate(sources, start=1):
        date = f", {r.published_date}" if r.published_date else ""
        snippet = r.content[:_MAX_CONTENT_CHARS].replace("\n", " ")
        lines.append(
            f"[{i}] {r.title or r.source_url} ({r.source_type}{date})\n"
            f"    URL: {r.source_url}\n"
            f"    Sub-question: {r.question}\n"
            f"    Content: {snippet}"
        )
    return "\n\n".join(lines)


def _empty_report(query: str) -> FinalReport:
    """Produce a graceful stub when no evidence was gathered."""
    return FinalReport(
        title=f"Research Report: {query}",
        summary="No sufficient evidence could be retrieved to answer this query. "
        "Please try rephrasing the question or check the search API credentials.",
        sections=[
            ReportSection(
                heading="Status",
                content="The research pipeline returned no usable sources.",
                citations=[],
            )
        ],
        all_citations=[],
    )


@traceable(name="writer", run_type="chain")
async def writer_node(state: AgentState) -> dict:
    """Synthesise the final report from approved research results.

    Args:
        state: Current graph state; uses ``research_results`` and ``query``.

    Returns:
        A partial state update with ``final_report`` set to Markdown.
    """
    query = state["query"]
    run_id = state.get("run_id", "")
    results: list[ResearchResult] = state.get("research_results", [])

    if not results:
        logger.warning("[writer] no results — emitting stub report")
        report = _empty_report(query)
        markdown = to_markdown(report)
        await redis_store.save(run_id, "final_report", markdown)
        return {"final_report": markdown}

    sources = _select_sources(results)
    logger.info("[writer] synthesising report from {} sources", len(sources))
    source_block = _format_sources(sources)

    # Writer often needs a generous output budget; bump max_tokens for this call.
    structured_llm = get_llm(max_tokens=max(LLM_MAX_TOKENS, 8000)).with_structured_output(FinalReport)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original query: {query}\n\n"
                f"Numbered sources:\n{source_block}\n\n"
                "Write a structured report that answers the original query, citing "
                "the sources inline with [n] markers."
            )
        ),
    ]

    report: FinalReport = await ainvoke_with_retry(structured_llm, messages)

    # Fall back to the source URLs we provided if the model under-populated citations.
    if not report.all_citations:
        report.all_citations = [r.source_url for r in sources if r.source_url]

    markdown = to_markdown(report)
    logger.info("[writer] report complete ({} chars, {} citations)", len(markdown), len(report.all_citations))

    await redis_store.save(run_id, "final_report", markdown)
    await redis_store.save(run_id, "final_report_structured", report.model_dump())
    return {"final_report": markdown}
