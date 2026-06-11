"""Qdrant vector-store helpers used as a semantic cache for research results.

Embeddings are produced locally with fastembed (no external API key required).
All public functions degrade gracefully: if Qdrant is unreachable the pipeline
simply treats every lookup as a cache miss instead of crashing.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Optional

from loguru import logger

from config import (
    CACHE_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    VECTOR_SIZE,
)
from schemas.models import ResearchResult

try:
    from qdrant_client import AsyncQdrantClient, models
except ImportError:  # pragma: no cover
    AsyncQdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]

try:
    from fastembed import TextEmbedding
except ImportError:  # pragma: no cover
    TextEmbedding = None  # type: ignore[assignment]


_client: "AsyncQdrantClient | None" = None
_embedder: "TextEmbedding | None" = None


def _get_embedder() -> "TextEmbedding":
    """Return a lazily-loaded fastembed model (singleton)."""
    global _embedder
    if TextEmbedding is None:
        raise RuntimeError("fastembed is not installed. Run: pip install fastembed")
    if _embedder is None:
        logger.info("Loading embedding model {} ...", EMBEDDING_MODEL)
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def _get_client() -> "AsyncQdrantClient":
    """Return a lazily-instantiated singleton ``AsyncQdrantClient``."""
    global _client
    if AsyncQdrantClient is None:
        raise RuntimeError("qdrant-client is not installed. Run: pip install qdrant-client")
    if _client is None:
        if QDRANT_PATH:
            logger.info("Using embedded Qdrant (no server) at path={}", QDRANT_PATH)
            _client = AsyncQdrantClient(path=QDRANT_PATH)
        elif QDRANT_URL.strip().lower() in (":memory:", "memory"):
            logger.info("Using in-memory Qdrant (no server)")
            _client = AsyncQdrantClient(location=":memory:")
        else:
            _client = AsyncQdrantClient(url=QDRANT_URL)
    return _client


def _embed_sync(text: str) -> list[float]:
    """Blocking embedding call (fastembed runs ONNX on CPU)."""
    embedder = _get_embedder()
    vector = next(iter(embedder.embed([text])))
    return vector.tolist()


async def embed_text(text: str) -> list[float]:
    """Embed a string into a dense vector, off the event loop.

    Args:
        text: The text to embed.

    Returns:
        A ``VECTOR_SIZE``-dimensional embedding as a list of floats.
    """
    return await asyncio.to_thread(_embed_sync, text)


async def init_collection() -> None:
    """Create the ``research_cache`` collection if it does not already exist.

    Uses cosine distance and the configured vector size. Safe to call on every
    startup — existing collections are left untouched.
    """
    try:
        client = _get_client()
        existing = await client.get_collections()
        names = {c.name for c in existing.collections}
        if QDRANT_COLLECTION in names:
            logger.info("Qdrant collection {!r} already exists", QDRANT_COLLECTION)
            return
        await client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
        logger.info("Created Qdrant collection {!r} (size={}, cosine)", QDRANT_COLLECTION, VECTOR_SIZE)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not initialise Qdrant collection: {}", exc)


async def search(
    query_embedding: list[float],
    threshold: float = CACHE_SIMILARITY_THRESHOLD,
    limit: int = 3,
) -> list[ResearchResult]:
    """Return cached research results similar to the query embedding.

    Args:
        query_embedding: The query vector.
        threshold: Minimum cosine similarity for a hit (default 0.85).
        limit: Maximum number of cached results to return.

    Returns:
        A list of ``ResearchResult`` reconstructed from cache payloads. Empty on
        a cache miss or any error.
    """
    try:
        client = _get_client()
        response = await client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant search failed (treating as cache miss): {}", exc)
        return []

    hits: list[ResearchResult] = []
    for point in response.points:
        payload: dict[str, Any] = point.payload or {}
        try:
            hits.append(ResearchResult(**payload))
        except Exception as exc:  # noqa: BLE001 - skip malformed cache entries
            logger.debug("Skipping malformed cache payload: {}", exc)
    if hits:
        logger.debug("Cache hit: {} cached results above threshold {}", len(hits), threshold)
    return hits


async def upsert(result: ResearchResult, embedding: Optional[list[float]] = None) -> None:
    """Store a research result in the cache with its embedding.

    Args:
        result: The result to cache.
        embedding: Optional precomputed embedding; computed from the question if
            omitted.
    """
    try:
        if embedding is None:
            embedding = await embed_text(result.question)
        client = _get_client()
        await client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=result.model_dump(),
                )
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant upsert failed (result not cached): {}", exc)


async def close() -> None:
    """Close the Qdrant client connection (called on app shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.close()
        except Exception:  # noqa: BLE001
            pass
        _client = None
