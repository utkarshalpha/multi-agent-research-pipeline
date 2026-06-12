"""Deterministic offline mocks used when ``MOCK_MODE`` is enabled.

Everything in this module is hash-derived from its inputs: no network calls, no
API keys, and no randomness, so repeated runs produce identical output. The
mocks are duck-type compatible with the real implementations, which lets the
rest of the pipeline run completely unchanged:

* :class:`MockChatModel` mimics how the agents use ``config.get_llm()`` —
  ``.with_structured_output(Schema)`` returns a runnable whose ``ainvoke``
  yields a valid ``Schema`` instance, and a plain ``.ainvoke`` returns a
  ``langchain_core`` ``AIMessage``.
* :func:`mock_tavily_results` / :func:`mock_arxiv_results` return canned search
  hits shaped exactly like the real tools' normalised output.
* :func:`mock_embedding` returns a deterministic 384-dim unit vector.

Retry-loop test hook: when the text the mock critic LLM is scoring contains the
marker ``force-retry``, the relevance assessment returns a very low score
(~0.1); otherwise it scores high (~0.85). Because the mock search results echo
the query verbatim, putting ``force-retry`` in a query exercises the critic
retry loop end to end in mock mode.
"""

from __future__ import annotations

import hashlib
import math
import re
import types
from typing import Any, Literal, Union, get_args, get_origin

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from config import ANTHROPIC_MODEL

# Marker honoured by the mock critic relevance assessment (see module docstring).
FORCE_RETRY_MARKER = "force-retry"

_LOW_RELEVANCE = 0.1
_HIGH_RELEVANCE = 0.85
# Fixed recent ISO date so the Critic's recency scoring stays high and stable.
_MOCK_PUBLISHED_DATE = "2026-01-15"
_EMBEDDING_DIM = 384

_LOW_FEEDBACK = (
    "The retrieved results do not adequately address their sub-questions. "
    "Gather more specific, higher-quality sources that directly answer each "
    "sub-question."
)
_HIGH_FEEDBACK = (
    "The results are relevant, credible, and recent; no follow-up research is "
    "required."
)


# --------------------------------------------------------------------------- #
# Deterministic primitives
# --------------------------------------------------------------------------- #
def _digest(key: str) -> int:
    """Return a deterministic 64-bit integer derived from ``key``."""
    raw = hashlib.sha256(key.encode("utf-8", errors="replace")).digest()
    return int.from_bytes(raw[:8], "big")


def _slugify(text: str, max_len: int = 48) -> str:
    """Build a URL-safe slug from arbitrary text."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len].strip("-")
    return slug or "query"


def mock_embedding(text: str) -> list[float]:
    """Return a deterministic 384-dim unit vector derived from ``text``.

    Identical inputs always produce identical vectors (so semantic-cache hits
    work for repeated questions), while distinct inputs produce essentially
    uncorrelated vectors. No model download, no randomness.

    Args:
        text: The text to "embed".

    Returns:
        A 384-dimensional unit-norm list of floats.
    """
    seed = text.encode("utf-8", errors="replace")
    raw = bytearray()
    counter = 0
    while len(raw) < _EMBEDDING_DIM * 4:
        raw.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1

    values: list[float] = []
    for i in range(_EMBEDDING_DIM):
        chunk = bytes(raw[i * 4 : (i + 1) * 4])
        n = int.from_bytes(chunk, "big")
        values.append((n / 2**32) * 2.0 - 1.0)  # uniform-ish in [-1, 1)

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


# --------------------------------------------------------------------------- #
# Canned search results
# --------------------------------------------------------------------------- #
def mock_tavily_results(query: str) -> list[dict[str, Any]]:
    """Return 2-3 deterministic canned web results for ``query``.

    Shaped exactly like the real normalised Tavily output:
    ``{url, title, content, published_date}``. The content echoes the query
    verbatim so test markers (e.g. ``force-retry``) propagate to the Critic.

    Args:
        query: The search query (typically a sub-question).

    Returns:
        A list of normalised result dicts.
    """
    slug = _slugify(query)
    count = 2 + _digest(f"tavily:{query}") % 2  # deterministic 2 or 3
    angles = ("overview", "analysis", "evidence")

    results: list[dict[str, Any]] = []
    for i in range(count):
        angle = angles[i]
        results.append(
            {
                "url": f"https://example.org/research/{slug}-{angle}",
                "title": f"{angle.title()}: {query}",
                "content": (
                    f"This mock article ({angle}) covers the topic: {query}. "
                    f"It summarises the current state of knowledge about {query}, "
                    "including definitions, recent developments, and practical "
                    "implications, drawn from deterministic offline fixtures. "
                    f"Key takeaway #{i + 1}: the evidence base for {query} is "
                    "restated here verbatim so downstream agents can score it."
                ),
                "published_date": _MOCK_PUBLISHED_DATE,
            }
        )
    return results


def mock_arxiv_results(query: str) -> list[dict[str, Any]]:
    """Return deterministic canned arXiv-shaped results for ``query``.

    Shaped exactly like the real normalised arXiv output:
    ``{title, summary, pdf_url, published_date}``.

    Args:
        query: The search query (typically a sub-question).

    Returns:
        A list of normalised paper dicts.
    """
    results: list[dict[str, Any]] = []
    for i in range(2):
        number = _digest(f"arxiv:{query}:{i}") % 100_000
        results.append(
            {
                "title": f"A Mock Study of {query} (Part {i + 1})",
                "summary": (
                    f"We present a deterministic mock paper examining {query}. "
                    f"Our offline analysis reproduces the key findings relevant "
                    f"to {query} and discusses methodology, results, and "
                    "limitations of the approach."
                ),
                "pdf_url": f"https://arxiv.org/pdf/2601.{number:05d}v{i + 1}",
                "published_date": _MOCK_PUBLISHED_DATE,
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Prompt-text extraction helpers
# --------------------------------------------------------------------------- #
def _messages_to_text(messages: Any) -> str:
    """Flatten LangChain messages / dicts / strings into one prompt string."""
    if isinstance(messages, str):
        return messages
    items = messages if isinstance(messages, (list, tuple)) else [messages]
    parts: list[str] = []
    for msg in items:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        if isinstance(content, list):  # content-block style messages
            content = " ".join(
                str(block.get("text", block)) if isinstance(block, dict) else str(block)
                for block in content
            )
        parts.append(str(content) if content is not None else str(msg))
    return "\n\n".join(parts)


def _extract_query(text: str) -> str:
    """Pull the user's query out of an agent prompt (with graceful fallback)."""
    for pattern in (r"Main query:\s*(.+)", r"Original query:\s*(.+)"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:200] if lines else "the requested topic"


def _count_results(text: str) -> int:
    """Count how many numbered results the critic prompt asks us to score."""
    numbered = re.findall(r"^\[\d+\]\s", text, flags=re.MULTILINE)
    if numbered:
        return len(numbered)
    match = re.search(r"these\s+(\d+)\s+results", text)
    if match:
        return int(match.group(1))
    return 1


def _extract_sources(text: str) -> list[dict[str, Any]]:
    """Parse the writer's numbered source block into (index, title, url) dicts."""
    sources: list[dict[str, Any]] = []
    pattern = re.compile(r"^\[(\d+)\]\s*(.+?)\s*\n\s*URL:\s*(\S+)", flags=re.MULTILINE)
    for match in pattern.finditer(text):
        sources.append(
            {"index": int(match.group(1)), "title": match.group(2), "url": match.group(3)}
        )
    if not sources:  # fallback: any URLs present anywhere in the prompt
        for i, url in enumerate(re.findall(r"https?://[^\s)\]]+", text), start=1):
            sources.append({"index": i, "title": url, "url": url})
    return sources


# --------------------------------------------------------------------------- #
# Schema-specific builders
# --------------------------------------------------------------------------- #
def _mock_sub_questions(query: str) -> list[str]:
    """Three focused sub-questions derived from (and echoing) the query."""
    return [
        f"What are the key concepts and background needed to understand {query}?",
        f"What do current sources and recent research say about {query}?",
        f"What are the main challenges, limitations, and open questions around {query}?",
    ]


def _section_class(annotation: Any) -> type[BaseModel] | None:
    """Find the nested pydantic model inside e.g. ``list[ReportSection]``."""
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _mock_sections(
    section_cls: type[BaseModel], query: str, sources: list[dict[str, Any]], text: str
) -> list[BaseModel]:
    """Build coherent report sections that cite the provided sources."""
    headings = ("Overview", "Key Findings", "Outlook and Limitations")
    buckets: list[list[dict[str, Any]]] = [[] for _ in headings]
    for pos, src in enumerate(sources):
        buckets[pos % len(headings)].append(src)

    sections: list[BaseModel] = []
    for heading, bucket in zip(headings, buckets):
        if bucket:
            markers = " ".join(f"[{s['index']}]" for s in bucket)
            refs = "; ".join(f"{s['title']} [{s['index']}]" for s in bucket)
            content = (
                f"{heading} of {query}: this section synthesises the cited "
                f"evidence {markers}. According to {refs}, the available "
                f"sources consistently characterise {query} and support the "
                "report's conclusions."
            )
            citations = [s["url"] for s in bucket]
        else:
            content = (
                f"{heading}: no additional sources were assigned to this "
                "section; see the cited evidence in the other sections."
            )
            citations = []
        try:
            sections.append(
                section_cls(heading=heading, content=content, citations=citations)
            )
        except Exception:  # noqa: BLE001 - unknown section shape: go generic
            sections.append(_generic_instance(section_cls, text))
    return sections


# --------------------------------------------------------------------------- #
# Generic pydantic-schema instance generation
# --------------------------------------------------------------------------- #
def _min_len(metadata: Any) -> int | None:
    """Extract a ``min_length`` constraint from pydantic field metadata."""
    for meta in metadata or ():
        value = getattr(meta, "min_length", None)
        if isinstance(value, int):
            return value
    return None


def _generic_value(
    name: str, annotation: Any, metadata: Any, text: str, depth: int = 0
) -> Any:
    """Generate a deterministic, type-appropriate value for one field."""
    if depth > 4 or annotation is None:
        return None

    origin = get_origin(annotation)

    if origin is Literal:
        options = get_args(annotation)
        return options[_digest(f"{name}:literal") % len(options)]

    if origin is Union or origin is getattr(types, "UnionType", None):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if not non_none:
            return None
        return _generic_value(name, non_none[0], (), text, depth + 1)

    if origin in (list, set, tuple, frozenset):
        args = get_args(annotation)
        inner = args[0] if args else str
        count = _min_len(metadata) or 2
        return [
            _generic_value(f"{name}[{i}]", inner, (), text, depth + 1)
            for i in range(count)
        ]

    if origin is dict:
        return {}

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _generic_instance(annotation, text, depth + 1)

    if annotation is bool:
        return _digest(f"{name}:bool") % 2 == 0
    if annotation is int:
        return _digest(f"{name}:int") % 10
    if annotation is float:
        return round((_digest(f"{name}:float") % 1000) / 1000, 3)
    if annotation is str:
        return f"Mock {name.replace('_', ' ')} for: {_extract_query(text)[:160]}"

    # Last resort for exotic annotations: a deterministic string.
    return f"mock-{name}-{_digest(name) % 10**6:06d}"


def _generic_instance(schema: type[BaseModel], text: str, depth: int = 0) -> BaseModel:
    """Build a valid instance of an arbitrary pydantic schema by introspection."""
    data: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue  # let declared defaults stand
        data[name] = _generic_value(name, field.annotation, field.metadata, text, depth)
    return schema(**data)


def _build_structured(schema: type[BaseModel], text: str) -> BaseModel:
    """Build a sensible, deterministic instance of ``schema`` from prompt text.

    Known field names get quality special-cases (planner sub-questions, critic
    relevance assessment with the ``force-retry`` hook, writer report); every
    other field falls back to generic type-driven generation.
    """
    fields = schema.model_fields
    names = set(fields)
    query = _extract_query(text)
    force_retry = FORCE_RETRY_MARKER in text
    sources = (
        _extract_sources(text) if names & {"sections", "all_citations"} else []
    )

    data: dict[str, Any] = {}
    for name, field in fields.items():
        if name == "questions":
            data[name] = _mock_sub_questions(query)
        elif name == "relevance_scores":
            score = _LOW_RELEVANCE if force_retry else _HIGH_RELEVANCE
            data[name] = [score] * _count_results(text)
        elif name == "feedback":
            data[name] = _LOW_FEEDBACK if force_retry else _HIGH_FEEDBACK
        elif name == "title":
            data[name] = f"Research Report: {query}"
        elif name == "summary":
            tail = " The cited evidence is listed in the references [1]." if sources else ""
            data[name] = (
                f"This report synthesises {len(sources) or 'the available'} "
                f"sources to answer: {query}. It was generated "
                f"deterministically in MOCK_MODE for offline testing; its "
                f"structure, citations, and section flow mirror a real "
                f"pipeline run.{tail}"
            )
        elif name == "sections":
            section_cls = _section_class(field.annotation)
            if section_cls is not None:
                data[name] = _mock_sections(section_cls, query, sources, text)
            else:
                data[name] = _generic_value(name, field.annotation, field.metadata, text)
        elif name == "all_citations":
            data[name] = [s["url"] for s in sources]
        elif field.is_required():
            data[name] = _generic_value(name, field.annotation, field.metadata, text)

    try:
        return schema(**data)
    except Exception:  # noqa: BLE001 - validation mismatch: fall back to generic
        return _generic_instance(schema, text)


# --------------------------------------------------------------------------- #
# The mock chat model
# --------------------------------------------------------------------------- #
class _MockStructuredRunnable:
    """What ``MockChatModel.with_structured_output(schema)`` returns."""

    def __init__(self, schema: type[BaseModel]) -> None:
        self._schema = schema

    async def ainvoke(self, messages: Any, config: Any = None, **_: Any) -> BaseModel:
        """Return a deterministic, valid instance of the bound schema."""
        return _build_structured(self._schema, _messages_to_text(messages))

    def invoke(self, messages: Any, config: Any = None, **_: Any) -> BaseModel:
        """Synchronous mirror of :meth:`ainvoke` (handy in tests)."""
        return _build_structured(self._schema, _messages_to_text(messages))


class MockChatModel:
    """Offline, deterministic stand-in for ``ChatAnthropic``.

    Duck-type compatible with how the agents use ``config.get_llm()``:
    ``with_structured_output(Schema)`` plus async ``ainvoke``. No network, no
    API key, no randomness.
    """

    def __init__(
        self, model: str | None = None, max_tokens: int | None = None, **_: Any
    ) -> None:
        self.model = model or ANTHROPIC_MODEL
        self.max_tokens = max_tokens
        self.model_label = f"mock ({self.model})"

    def with_structured_output(
        self, schema: type[BaseModel], **_: Any
    ) -> _MockStructuredRunnable:
        """Bind a pydantic schema; mirrors ``ChatAnthropic.with_structured_output``."""
        return _MockStructuredRunnable(schema)

    def _respond(self, messages: Any) -> AIMessage:
        """Build the deterministic plain-text response."""
        text = _messages_to_text(messages)
        query = _extract_query(text)
        content = (
            f"[{self.model_label}] Deterministic mock response about: {query}. "
            f"Prompt digest: {_digest(text) % 10**8:08d}."
        )
        return AIMessage(
            content=content,
            response_metadata={"model_name": self.model_label, "mock": True},
        )

    async def ainvoke(self, messages: Any, config: Any = None, **_: Any) -> AIMessage:
        """Return a deterministic ``AIMessage`` (async, like the real client)."""
        return self._respond(messages)

    def invoke(self, messages: Any, config: Any = None, **_: Any) -> AIMessage:
        """Synchronous mirror of :meth:`ainvoke`."""
        return self._respond(messages)
