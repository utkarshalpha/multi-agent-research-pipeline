"""Central configuration, the shared LLM factory, and a retry helper.

Everything that needs an environment variable, a model name, or a tunable
threshold reads it from here so the rest of the codebase stays DRY.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load .env once at import time so every module sees the same configuration.
load_dotenv()

if TYPE_CHECKING:  # imported lazily at runtime inside get_llm() to avoid cycles
    from tools.mocks import MockChatModel


def _env_flag(name: str, default: str = "") -> bool:
    """Parse a boolean environment flag (1/true/yes, case-insensitive)."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------- #
# Mock / offline mode
# --------------------------------------------------------------------------- #
# When on, the ENTIRE pipeline runs offline: deterministic mocks replace the
# LLM, Tavily, arXiv, and the embedder — zero network calls, zero API keys.
MOCK_MODE: bool = _env_flag("MOCK_MODE")

# --------------------------------------------------------------------------- #
# Model / API configuration
# --------------------------------------------------------------------------- #
# NOTE: the original spec named ``claude-3-5-sonnet``, which has since been
# retired by Anthropic and now returns a 404. We default to the current Sonnet
# tier (``claude-sonnet-4-6``). Set ANTHROPIC_MODEL to ``claude-opus-4-8`` for
# the most capable tier.
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Human-readable model label for /health and response metadata. In mock mode it
# is clearly marked, e.g. "mock (claude-sonnet-4-6)".
MODEL_LABEL: str = f"mock ({ANTHROPIC_MODEL})" if MOCK_MODE else ANTHROPIC_MODEL

# --------------------------------------------------------------------------- #
# Infrastructure endpoints
# --------------------------------------------------------------------------- #
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
# Optional embedded mode: set QDRANT_PATH to a local folder (e.g. ./qdrant_local)
# to run Qdrant fully in-process with NO Docker/server required. Leave empty to
# use QDRANT_URL (the docker-compose setup).
QDRANT_PATH: str = os.getenv("QDRANT_PATH", "")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

# --------------------------------------------------------------------------- #
# Vector store / embedding configuration
# --------------------------------------------------------------------------- #
# The spec asked for a 1536-dim vector (OpenAI's ``text-embedding-3-small``).
# To keep the project runnable with **no extra API keys**, we default to a local
# fastembed model (``BAAI/bge-small-en-v1.5``, 384 dims). Override both values
# together if you swap in a different embedder.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "384"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "research_cache")
# Cosine similarity above which a cached answer is considered a hit.
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))

# --------------------------------------------------------------------------- #
# Pipeline tuning
# --------------------------------------------------------------------------- #
CONFIDENCE_PASS_THRESHOLD: float = float(os.getenv("CONFIDENCE_PASS_THRESHOLD", "0.7"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
TAVILY_TIMEOUT_SECONDS: float = float(os.getenv("TAVILY_TIMEOUT_SECONDS", "10"))
ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "3"))

# --------------------------------------------------------------------------- #
# API hardening
# --------------------------------------------------------------------------- #
# Hard wall-clock limit for a single /research run, in seconds.
RESEARCH_TIMEOUT_SECONDS: int = int(os.getenv("RESEARCH_TIMEOUT_SECONDS", "300"))
# Static API key required by the FastAPI layer; empty string disables auth.
PIPELINE_API_KEY: str = os.getenv("PIPELINE_API_KEY", "")
# Max requests per minute per client; 0 disables rate limiting.
RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))

# Convenience namespace so callers can use ``config.settings.MOCK_MODE`` (or
# ``from config import settings``) interchangeably with the bare constants.
settings = sys.modules[__name__]


def _configure_langsmith() -> None:
    """Enable LangSmith tracing if an API key is present.

    LangChain/LangGraph runnables and the ``@traceable`` decorator emit traces
    automatically once these environment variables are set.
    """
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault(
            "LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "research-pipeline")
        )
        logger.info("LangSmith tracing enabled (project={})", os.environ["LANGCHAIN_PROJECT"])
    else:
        logger.info("LANGSMITH_API_KEY not set — tracing disabled")


_configure_langsmith()


# API constraint: Claude Opus 4.7+, Fable, and Mythos models reject sampling
# params (temperature/top_p/top_k) with HTTP 400 — they must be omitted there.
_NO_SAMPLING_PARAM_MARKERS: tuple[str, ...] = ("opus-4-7", "opus-4-8", "fable", "mythos")


def _accepts_temperature(model: str) -> bool:
    """Return True if ``model`` accepts sampling params such as ``temperature``."""
    lowered = model.lower()
    return not any(marker in lowered for marker in _NO_SAMPLING_PARAM_MARKERS)


@lru_cache(maxsize=4)
def get_llm(
    temperature: float | None = None, max_tokens: int | None = None
) -> ChatAnthropic | MockChatModel:
    """Return a (cached) chat model: ``ChatAnthropic``, or a mock in MOCK_MODE.

    Args:
        temperature: Sampling temperature; defaults to ``LLM_TEMPERATURE``.
            Ignored (omitted) for model families that reject sampling params.
        max_tokens: Max output tokens; defaults to ``LLM_MAX_TOKENS``.

    Returns:
        A configured ``ChatAnthropic`` instance shared across agents, or a
        duck-type-compatible ``MockChatModel`` when ``MOCK_MODE`` is on.
    """
    if MOCK_MODE:
        from tools.mocks import MockChatModel  # local import avoids import cycles

        logger.info("MOCK_MODE on — using offline {}", MODEL_LABEL)
        return MockChatModel(
            model=ANTHROPIC_MODEL,
            max_tokens=LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        )

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY is not set — LLM calls will fail until you set it.")

    kwargs: dict[str, Any] = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        "timeout": 60,
        "max_retries": 2,
    }
    # API constraint: Claude Opus 4.7/4.8, Fable, and Mythos reject sampling
    # params (temperature/top_p/top_k) with HTTP 400, so only pass temperature
    # to model families that accept it.
    if _accepts_temperature(ANTHROPIC_MODEL):
        kwargs["temperature"] = LLM_TEMPERATURE if temperature is None else temperature
    return ChatAnthropic(**kwargs)


# Exponential-backoff retry decorator for *transient* Anthropic API failures.
# The Anthropic SDK already retries 429/5xx, but the spec explicitly asks for
# tenacity-based backoff, so we layer it on top for resilience.
#
# Only genuinely retryable errors are retried: connection drops, timeouts,
# 429 rate limits, and 5xx server errors (including 529 overloaded, which the
# SDK maps to InternalServerError). Non-retryable client errors (400/401/403/
# 404/...) fail fast — retrying them would burn ~60s of backoff on a request
# that can never succeed.
_RETRYABLE_ANTHROPIC_ERRORS: tuple[type[Exception], ...] = (
    anthropic.APIConnectionError,  # network failures (APITimeoutError subclass too)
    anthropic.APITimeoutError,
    anthropic.RateLimitError,  # HTTP 429
    anthropic.InternalServerError,  # HTTP >= 500
)

llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE_ANTHROPIC_ERRORS),
)


@llm_retry
async def ainvoke_with_retry(runnable: Runnable, messages: Any) -> Any:
    """Invoke a runnable asynchronously with exponential-backoff retries.

    Args:
        runnable: Any LangChain runnable (e.g. an LLM bound to structured output).
        messages: The input passed to ``runnable.ainvoke``.

    Returns:
        Whatever the runnable returns (typically a Pydantic model instance).
    """
    return await runnable.ainvoke(messages)
