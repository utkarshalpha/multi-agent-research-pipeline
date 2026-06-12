"""Evaluation harness: run every question in ``eval_set.json`` through the graph.

For each run it measures latency, retry count, citation count, and estimated
token cost **and** scores the *structural quality* of the produced report so the
suite can detect a wrong, empty, or hallucinated answer — not just a slow or
expensive one. Quality dimensions per run:

* report present and above a minimum length;
* at least ``_MIN_SECTIONS`` sections;
* citations present;
* **groundedness** — every cited URL must appear in the sources actually
  gathered during the run (catches hallucinated citations);
* **sub-question coverage** — how many of the Planner's sub-questions ended up
  with supporting evidence, and how many were left unanswered.

These collapse into a per-run ``quality_score`` in ``[0, 1]`` and a suite-level
quality pass rate.

The harness is ``MOCK_MODE``-aware: ``results.json`` records ``mock: true/false``
and a clearly-labelled model name so offline/mock results are never mistaken for
live ones.

Run from the project root::

    python -m evals.run_evals            # full suite
    python -m evals.run_evals --limit 3  # only the first 3 questions

Results are written to ``evals/results.json`` (gitignored) and a readable
summary table is printed to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config import ANTHROPIC_MODEL, MOCK_MODE, MODEL_LABEL
from graph.graph import get_compiled_graph
from memory.redis_store import redis_store
from schemas.models import ResearchResult
from tools.vector_store import close as close_qdrant, init_collection

try:
    from langchain_core.callbacks import UsageMetadataCallbackHandler
except ImportError:  # pragma: no cover
    UsageMetadataCallbackHandler = None  # type: ignore[assignment]

_HERE = Path(__file__).resolve().parent
_EVAL_SET = _HERE / "eval_set.json"
_RESULTS = _HERE / "results.json"

# --------------------------------------------------------------------------- #
# Structural-quality thresholds
# --------------------------------------------------------------------------- #
# A report shorter than this (after stripping) is almost certainly a stub/error.
_MIN_REPORT_CHARS = 200
# A real report has at least this many body sections.
_MIN_SECTIONS = 2
# A run "passes" on quality when its aggregate score clears this bar.
_QUALITY_PASS_THRESHOLD = 0.7

# Approximate Anthropic pricing in USD per 1M tokens (input, output).
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a run from token counts and the pricing table."""
    in_price, out_price = _PRICING.get(model, (3.0, 15.0))
    return round(input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price, 4)


def _count_markdown_sections(markdown: str) -> int:
    """Count ``##`` body headings in the report, ignoring the References block."""
    count = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.lower().startswith("## references"):
            count += 1
    return count


def _assess_quality(
    *,
    markdown: str,
    structured: Optional[dict[str, Any]],
    citations: list[str],
    results: list[ResearchResult],
    sub_questions: list[str],
) -> dict[str, Any]:
    """Score the structural quality of a single run.

    Args:
        markdown: The final Markdown report (``""`` on failure).
        structured: The structured ``FinalReport`` dump if persisted, else None.
        citations: The citation URLs the run returned.
        results: The ``ResearchResult`` evidence gathered during the run.
        sub_questions: The Planner's sub-questions.

    Returns:
        A dict of per-dimension findings plus an aggregate ``quality_score`` in
        ``[0, 1]`` and a boolean ``quality_pass``.
    """
    report_chars = len(markdown.strip())
    report_present = report_chars > 0
    report_length_ok = report_chars >= _MIN_REPORT_CHARS

    # Prefer the structured section count; fall back to parsing the Markdown.
    if structured and structured.get("sections"):
        num_sections = len(structured["sections"])
    else:
        num_sections = _count_markdown_sections(markdown)
    sections_ok = num_sections >= _MIN_SECTIONS

    num_citations = len(citations)
    citations_present = num_citations > 0

    # Groundedness: every cited URL must be among the gathered source URLs.
    gathered_urls = {r.source_url for r in results if r.source_url}
    ungrounded = sorted({c for c in citations if c not in gathered_urls})
    grounded_count = num_citations - sum(1 for c in citations if c not in gathered_urls)
    groundedness = round(grounded_count / num_citations, 3) if num_citations else 0.0

    # Sub-question coverage: how many planned questions have any evidence.
    planner_count = len(sub_questions)
    answered_questions = {r.question for r in results} & set(sub_questions)
    unanswered_questions = [q for q in sub_questions if q not in answered_questions]
    coverage = round(len(answered_questions) / planner_count, 3) if planner_count else 0.0

    # Aggregate: equal-weighted mean of every dimension (booleans as 1.0/0.0).
    components = {
        "report_present": 1.0 if report_present else 0.0,
        "report_length_ok": 1.0 if report_length_ok else 0.0,
        "sections_ok": 1.0 if sections_ok else 0.0,
        "citations_present": 1.0 if citations_present else 0.0,
        "groundedness": groundedness,
        "coverage": coverage,
    }
    quality_score = round(sum(components.values()) / len(components), 3)

    return {
        "report_chars": report_chars,
        "report_present": report_present,
        "report_length_ok": report_length_ok,
        "num_sections": num_sections,
        "sections_ok": sections_ok,
        "num_citations": num_citations,
        "citations_present": citations_present,
        "groundedness": groundedness,
        "ungrounded_citations": ungrounded,
        "planner_question_count": planner_count,
        "answered_question_count": len(answered_questions),
        "unanswered_question_count": len(unanswered_questions),
        "unanswered_questions": unanswered_questions,
        "coverage": coverage,
        "quality_score": quality_score,
        "quality_pass": quality_score >= _QUALITY_PASS_THRESHOLD,
    }


async def _run_one(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Run a single eval question and return its measured metrics + quality."""
    run_id = str(uuid.uuid4())
    usage_cb = UsageMetadataCallbackHandler() if UsageMetadataCallbackHandler else None
    config: dict[str, Any] = {"configurable": {"thread_id": run_id}}
    if usage_cb is not None:
        config["callbacks"] = [usage_cb]

    initial_state = {
        "query": item["query"],
        "run_id": run_id,
        "retry_count": 0,
        "research_results": [],
        "critique": None,
    }

    start = time.perf_counter()
    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        error = None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Eval {} failed", item["id"])
        final_state = {}
        error = str(exc)
    elapsed = round(time.perf_counter() - start, 2)

    results: list[ResearchResult] = final_state.get("research_results", [])
    sub_questions: list[str] = final_state.get("sub_questions", [])
    markdown: str = final_state.get("final_report", "") or ""
    structured = await redis_store.load(run_id, "final_report_structured")
    citations = structured["all_citations"] if structured and structured.get("all_citations") else [
        r.source_url for r in results if r.source_url
    ]

    quality = _assess_quality(
        markdown=markdown,
        structured=structured,
        citations=citations,
        results=results,
        sub_questions=sub_questions,
    )

    input_tokens = output_tokens = 0
    if usage_cb is not None:
        for usage in usage_cb.usage_metadata.values():
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    return {
        "id": item["id"],
        "category": item["category"],
        "query": item["query"],
        "mock": MOCK_MODE,
        "latency_seconds": elapsed,
        "retry_count": final_state.get("retry_count", 0),
        "num_sources": len(results),
        "num_citations": len(citations),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": _estimate_cost(ANTHROPIC_MODEL, input_tokens, output_tokens),
        "quality_score": quality["quality_score"],
        "quality_pass": quality["quality_pass"],
        "quality": quality,
        "error": error,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print a fixed-width summary table of the eval results."""
    header = (
        f"{'ID':<8}{'CATEGORY':<12}{'LAT(s)':>8}{'RETRY':>7}{'SRC':>5}{'CITE':>6}"
        f"{'GRND':>6}{'COV':>6}{'QUAL':>6}{'PASS':>6}{'TOKENS':>9}{'COST$':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        q = r["quality"]
        tokens = r["input_tokens"] + r["output_tokens"]
        flag = "  ERROR" if r["error"] else ""
        print(
            f"{r['id']:<8}{r['category']:<12}{r['latency_seconds']:>8.2f}"
            f"{r['retry_count']:>7}{r['num_sources']:>5}{r['num_citations']:>6}"
            f"{q['groundedness']:>6.2f}{q['coverage']:>6.2f}{r['quality_score']:>6.2f}"
            f"{('Y' if r['quality_pass'] else 'N'):>6}"
            f"{tokens:>9}{r['estimated_cost_usd']:>9.4f}{flag}"
        )


def _print_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute, print, and return aggregate statistics."""
    n = len(rows)
    ok = [r for r in rows if not r["error"]]
    passed = [r for r in rows if r["quality_pass"]]
    total_cost = round(sum(r["estimated_cost_usd"] for r in rows), 4)
    agg = {
        "total_runs": n,
        "successful_runs": len(ok),
        "failed_runs": n - len(ok),
        "quality_pass_count": len(passed),
        "quality_pass_rate": round(len(passed) / n, 3) if n else 0,
        "avg_quality_score": round(sum(r["quality_score"] for r in rows) / n, 3) if n else 0,
        "avg_groundedness": round(sum(r["quality"]["groundedness"] for r in rows) / n, 3) if n else 0,
        "avg_coverage": round(sum(r["quality"]["coverage"] for r in rows) / n, 3) if n else 0,
        "total_unanswered_questions": sum(r["quality"]["unanswered_question_count"] for r in rows),
        "avg_latency_seconds": round(sum(r["latency_seconds"] for r in rows) / n, 2) if n else 0,
        "avg_retry_count": round(sum(r["retry_count"] for r in rows) / n, 2) if n else 0,
        "avg_citations": round(sum(r["num_citations"] for r in rows) / n, 2) if n else 0,
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
        "total_estimated_cost_usd": total_cost,
        "mock": MOCK_MODE,
        "model": MODEL_LABEL,
    }
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    return agg


async def main(limit: Optional[int] = None) -> None:
    """Run the eval questions (optionally a subset) and persist the results.

    Args:
        limit: If set, run only the first ``limit`` questions.
    """
    eval_data = json.loads(_EVAL_SET.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    if limit is not None:
        questions = questions[:limit]
    logger.info(
        "Loaded {} eval questions (mock={}, model={})",
        len(questions),
        MOCK_MODE,
        MODEL_LABEL,
    )

    await init_collection()
    await redis_store.connect()
    graph = get_compiled_graph()

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(questions, start=1):
        logger.info("Running eval {}/{}: {}", i, len(questions), item["id"])
        rows.append(await _run_one(graph, item))

    _print_table(rows)
    aggregate = _print_aggregate(rows)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mock": MOCK_MODE,
        "model": MODEL_LABEL,
        "limit": limit,
        "num_questions": len(rows),
        "min_report_chars": _MIN_REPORT_CHARS,
        "min_sections": _MIN_SECTIONS,
        "quality_pass_threshold": _QUALITY_PASS_THRESHOLD,
    }

    _RESULTS.write_text(
        json.dumps({"metadata": metadata, "aggregate": aggregate, "runs": rows}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved results to {}", _RESULTS)

    await redis_store.close()
    await close_qdrant()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (``--help`` exits here, before any work)."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_evals",
        description="Run the research eval suite and score structural quality.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N eval questions (default: run all).",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


if __name__ == "__main__":
    _args = _parse_args()
    asyncio.run(main(limit=_args.limit))
