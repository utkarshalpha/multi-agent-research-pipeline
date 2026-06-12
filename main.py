"""FastAPI HTTP interface and application entry point.

Exposes:
* ``POST /research`` — run the full multi-agent pipeline for a query.
* ``GET  /health``   — liveness probe.

Run locally with:  ``uvicorn main:app --reload``
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config import settings
from graph.graph import get_compiled_graph
from memory.redis_store import redis_store
from schemas.models import ResearchRequest, ResearchResponse, ResearchResult
from tools.vector_store import close as close_qdrant, init_collection

try:  # Available in recent langchain-core; degrade gracefully if absent.
    from langchain_core.callbacks import UsageMetadataCallbackHandler
except ImportError:  # pragma: no cover
    UsageMetadataCallbackHandler = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_RESPONSE_PATH = BASE_DIR / "examples" / "sample_response.json"


# --------------------------------------------------------------------------- #
# Request hardening: optional API-key auth and in-memory rate limiting
# --------------------------------------------------------------------------- #
# Sliding-window request timestamps (monotonic seconds) per client IP.
_rate_limit_windows: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_WINDOW_SECONDS = 60.0


async def _enforce_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject the request with 401 unless the X-API-Key header matches.

    No-op when ``PIPELINE_API_KEY`` is empty (the local-demo default). The
    setting is read per-request so tests can patch ``config.PIPELINE_API_KEY``.
    """
    expected = settings.PIPELINE_API_KEY
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _prune_rate_limit_windows(now: float) -> None:
    """Drop expired timestamps and evict per-IP entries whose deque empties.

    Without the eviction, every distinct client IP that ever hits the endpoint
    would leave an (eventually empty) deque behind for the life of the process
    — an unbounded memory leak under client-IP rotation. Sweeping the whole
    dict keeps it bounded to IPs seen within the last window.
    """
    stale_ips: list[str] = []
    for ip, window in _rate_limit_windows.items():
        while window and now - window[0] >= _RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if not window:
            stale_ips.append(ip)
    for ip in stale_ips:
        del _rate_limit_windows[ip]


async def _enforce_rate_limit(request: Request) -> None:
    """Apply a per-client-IP sliding-window rate limit; 429 on excess.

    No-op when ``RATE_LIMIT_PER_MINUTE`` is 0 (disabled). State is in-memory
    and per-process — adequate for the single-instance demo deployment.
    """
    limit = settings.RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    _prune_rate_limit_windows(now)
    window = _rate_limit_windows[client_ip]
    if len(window) >= limit:
        logger.warning("[/research] rate limit exceeded for client={}", client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {limit} requests per minute. Try again shortly.",
        )
    window.append(now)


# --------------------------------------------------------------------------- #
# Application lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise external services on startup; tear them down on shutdown."""
    logger.info("Starting research pipeline (model={})", settings.MODEL_LABEL)
    try:
        await init_collection()
    except Exception:  # noqa: BLE001 - degrade gracefully, mirroring Redis below
        logger.exception("Qdrant init failed — continuing degraded (cache misses only)")
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Return an empty 204 so browser favicon probes stop polluting the logs."""
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. In MOCK_MODE the model is labelled e.g. ``mock (...)``."""
    return {"status": "ok", "model": settings.MODEL_LABEL}


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


@app.post(
    "/research",
    response_model=ResearchResponse,
    dependencies=[Depends(_enforce_api_key), Depends(_enforce_rate_limit)],
)
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
        "sub_questions": [],
        "research_results": [],
        "critique": None,
    }

    timeout_seconds = settings.RESEARCH_TIMEOUT_SECONDS
    start = time.perf_counter()
    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config), timeout=timeout_seconds
        )
    except asyncio.TimeoutError as exc:  # alias of builtin TimeoutError on 3.11+
        logger.error(
            "[/research] run_id={} timed out after {}s", run_id, timeout_seconds
        )
        raise HTTPException(
            status_code=504,
            detail=f"Research run timed out after {timeout_seconds}s (run_id={run_id}).",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[/research] pipeline failed for run_id={}", run_id)
        raise HTTPException(
            status_code=500,
            detail=f"Research pipeline failed due to an internal error (run_id={run_id}).",
        ) from exc
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
        "model": settings.MODEL_LABEL,
        "latency_seconds": round(elapsed, 2),
        "retry_count": retry_count,
        "num_sub_questions": len(sub_questions),
        "num_sources": len(results),
        "num_citations": len(citations),
        "token_usage": token_usage,
        "unanswered_questions": final_state.get("unanswered_questions", []),
    }
    logger.info("[/research] run_id={} done in {:.2f}s ({} sources)", run_id, elapsed, len(results))

    return ResearchResponse(
        run_id=run_id, report=report, citations=citations, metadata=metadata
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
