"""FastAPI HTTP interface and application entry point.

Exposes:
* ``POST /research`` — run the full multi-agent pipeline for a query.
* ``GET  /health``   — liveness probe.

Run locally with:  ``uvicorn main:app --reload``
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from config import ANTHROPIC_MODEL
from graph.graph import get_compiled_graph
from memory.redis_store import redis_store
from schemas.models import ResearchResult
from tools.vector_store import close as close_qdrant, init_collection

try:  # Available in recent langchain-core; degrade gracefully if absent.
    from langchain_core.callbacks import UsageMetadataCallbackHandler
except ImportError:  # pragma: no cover
    UsageMetadataCallbackHandler = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_RESPONSE_PATH = BASE_DIR / "examples" / "sample_response.json"


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ResearchRequest(BaseModel):
    """Body for ``POST /research``."""

    query: str = Field(..., min_length=3, description="The research question.")


class ResearchResponse(BaseModel):
    """Response for ``POST /research``."""

    run_id: str
    report: str
    citations: list[str]
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Application lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise external services on startup; tear them down on shutdown."""
    logger.info("Starting research pipeline (model={})", ANTHROPIC_MODEL)
    await init_collection()
    await redis_store.connect()
    get_compiled_graph()  # warm the compiled graph
    yield
    logger.info("Shutting down research pipeline")
    await redis_store.close()
    await close_qdrant()


app = FastAPI(
    title="Multi-Agent Research Pipeline",
    description="Planner -> Researcher -> Critic -> Writer, orchestrated with LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the portfolio demo console."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "model": ANTHROPIC_MODEL}


@app.get("/sample-response", response_model=ResearchResponse, tags=["demo"])
async def sample_response() -> ResearchResponse:
    """Return a saved successful run for UI demos without spending tokens."""
    try:
        payload = json.loads(SAMPLE_RESPONSE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - packaging guard
        raise HTTPException(status_code=404, detail="Sample response not found") from exc
    return ResearchResponse.model_validate(payload)


def _extract_citations(final_state: dict, structured: dict | None) -> list[str]:
    """Resolve the list of cited URLs, preferring the writer's structured output."""
    if structured and structured.get("all_citations"):
        return structured["all_citations"]
    results: list[ResearchResult] = final_state.get("research_results", [])
    seen: list[str] = []
    for r in results:
        if r.source_url and r.source_url not in seen:
            seen.append(r.source_url)
    return seen


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run the multi-agent research pipeline for a query.

    Args:
        request: The research request containing the query.

    Returns:
        The run id, the Markdown report, the citation list, and run metadata
        (latency, token usage, retry count).
    """
    run_id = str(uuid.uuid4())
    logger.info("[/research] run_id={} query={!r}", run_id, request.query)

    graph = get_compiled_graph()
    usage_cb = UsageMetadataCallbackHandler() if UsageMetadataCallbackHandler else None

    config: dict[str, Any] = {"configurable": {"thread_id": run_id}}
    if usage_cb is not None:
        config["callbacks"] = [usage_cb]

    initial_state = {
        "query": request.query,
        "run_id": run_id,
        "retry_count": 0,
        "research_results": [],
        "critique": None,
    }

    start = time.perf_counter()
    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[/research] pipeline failed for run_id={}", run_id)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc
    elapsed = time.perf_counter() - start

    report = final_state.get("final_report", "")
    retry_count = final_state.get("retry_count", 0)
    results: list[ResearchResult] = final_state.get("research_results", [])
    sub_questions = final_state.get("sub_questions", [])

    structured = await redis_store.load(run_id, "final_report_structured")
    citations = _extract_citations(final_state, structured)

    # Aggregate token usage across every model call in the run.
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if usage_cb is not None:
        for model_usage in usage_cb.usage_metadata.values():
            token_usage["input_tokens"] += model_usage.get("input_tokens", 0)
            token_usage["output_tokens"] += model_usage.get("output_tokens", 0)
            token_usage["total_tokens"] += model_usage.get("total_tokens", 0)

    metadata = {
        "model": ANTHROPIC_MODEL,
        "latency_seconds": round(elapsed, 2),
        "retry_count": retry_count,
        "num_sub_questions": len(sub_questions),
        "num_sources": len(results),
        "num_citations": len(citations),
        "token_usage": token_usage,
    }
    logger.info("[/research] run_id={} done in {:.2f}s ({} sources)", run_id, elapsed, len(results))

    return ResearchResponse(
        run_id=run_id, report=report, citations=citations, metadata=metadata
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
