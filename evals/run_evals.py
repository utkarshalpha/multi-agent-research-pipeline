"""Evaluation harness: run every question in ``eval_set.json`` through the graph.

Measures per-run latency, retry count, citation count, and estimated token
cost, prints a summary table, and writes ``evals/results.json``.

Run from the project root:  ``python -m evals.run_evals``
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from config import ANTHROPIC_MODEL
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


async def _run_one(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Run a single eval question and return its measured metrics."""
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
    structured = await redis_store.load(run_id, "final_report_structured")
    citations = structured["all_citations"] if structured and structured.get("all_citations") else [
        r.source_url for r in results if r.source_url
    ]

    input_tokens = output_tokens = 0
    if usage_cb is not None:
        for usage in usage_cb.usage_metadata.values():
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    return {
        "id": item["id"],
        "category": item["category"],
        "query": item["query"],
        "latency_seconds": elapsed,
        "retry_count": final_state.get("retry_count", 0),
        "num_sources": len(results),
        "num_citations": len(citations),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": _estimate_cost(ANTHROPIC_MODEL, input_tokens, output_tokens),
        "error": error,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print a fixed-width summary table of the eval results."""
    header = f"{'ID':<8}{'CATEGORY':<12}{'LAT(s)':>8}{'RETRY':>7}{'SRC':>5}{'CITE':>6}{'TOKENS':>9}{'COST$':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        tokens = r["input_tokens"] + r["output_tokens"]
        flag = "  ERROR" if r["error"] else ""
        print(
            f"{r['id']:<8}{r['category']:<12}{r['latency_seconds']:>8.2f}"
            f"{r['retry_count']:>7}{r['num_sources']:>5}{r['num_citations']:>6}"
            f"{tokens:>9}{r['estimated_cost_usd']:>9.4f}{flag}"
        )


def _print_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute, print, and return aggregate statistics."""
    n = len(rows)
    ok = [r for r in rows if not r["error"]]
    total_cost = round(sum(r["estimated_cost_usd"] for r in rows), 4)
    agg = {
        "total_runs": n,
        "successful_runs": len(ok),
        "failed_runs": n - len(ok),
        "avg_latency_seconds": round(sum(r["latency_seconds"] for r in rows) / n, 2) if n else 0,
        "avg_retry_count": round(sum(r["retry_count"] for r in rows) / n, 2) if n else 0,
        "avg_citations": round(sum(r["num_citations"] for r in rows) / n, 2) if n else 0,
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
        "total_estimated_cost_usd": total_cost,
        "model": ANTHROPIC_MODEL,
    }
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    return agg


async def main() -> None:
    """Run all eval questions and persist the results."""
    eval_data = json.loads(_EVAL_SET.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    logger.info("Loaded {} eval questions", len(questions))

    await init_collection()
    await redis_store.connect()
    graph = get_compiled_graph()

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(questions, start=1):
        logger.info("Running eval {}/{}: {}", i, len(questions), item["id"])
        rows.append(await _run_one(graph, item))

    _print_table(rows)
    aggregate = _print_aggregate(rows)

    _RESULTS.write_text(
        json.dumps({"aggregate": aggregate, "runs": rows}, indent=2), encoding="utf-8"
    )
    logger.info("Saved results to {}", _RESULTS)

    await redis_store.close()
    await close_qdrant()


if __name__ == "__main__":
    asyncio.run(main())
