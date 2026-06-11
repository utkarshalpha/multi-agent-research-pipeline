"""Redis-backed short-term agent memory.

Each pipeline run writes its intermediate artefacts (sub-questions, critique,
etc.) under a single Redis hash keyed by ``run:<run_id>``. This gives you an
out-of-process audit trail of what each agent produced, with a TTL so the store
self-cleans. Every method degrades gracefully if Redis is unavailable.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

from config import REDIS_URL

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]

# Run artefacts expire after this many seconds (24h) to keep Redis tidy.
_DEFAULT_TTL_SECONDS = 60 * 60 * 24


class RedisStore:
    """Thin async wrapper over a Redis hash per run."""

    def __init__(self, url: str = REDIS_URL, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        """Create a store bound to a Redis URL.

        Args:
            url: Redis connection URL (e.g. ``redis://localhost:6379``).
            ttl_seconds: Expiry applied to each run's hash.
        """
        self._url = url
        self._ttl = ttl_seconds
        self._client: Optional["aioredis.Redis"] = None

    async def connect(self) -> None:
        """Open the connection pool and verify connectivity with a PING."""
        if aioredis is None:
            logger.warning("redis package not installed — short-term memory disabled.")
            return
        try:
            self._client = aioredis.from_url(self._url, decode_responses=True)
            await self._client.ping()
            logger.info("Connected to Redis at {}", self._url)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not connect to Redis ({}): short-term memory disabled.", exc)
            self._client = None

    @staticmethod
    def _key(run_id: str) -> str:
        return f"run:{run_id}"

    async def save(self, run_id: str, field: str, value: Any) -> None:
        """Persist a single field for a run.

        Args:
            run_id: The run identifier.
            field: Field name within the run's hash (e.g. ``"sub_questions"``).
            value: JSON-serialisable value to store.
        """
        if self._client is None:
            return
        try:
            await self._client.hset(self._key(run_id), field, json.dumps(value, default=str))
            await self._client.expire(self._key(run_id), self._ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis save failed for {}::{}: {}", run_id, field, exc)

    async def load(self, run_id: str, field: str) -> Optional[Any]:
        """Read a single field for a run.

        Args:
            run_id: The run identifier.
            field: Field name to read.

        Returns:
            The deserialised value, or ``None`` if absent / on error.
        """
        if self._client is None:
            return None
        try:
            raw = await self._client.hget(self._key(run_id), field)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis load failed for {}::{}: {}", run_id, field, exc)
            return None

    async def load_run(self, run_id: str) -> dict[str, Any]:
        """Return every stored field for a run as a dict.

        Args:
            run_id: The run identifier.

        Returns:
            A dict of all stored fields (deserialised). Empty if none / on error.
        """
        if self._client is None:
            return {}
        try:
            raw = await self._client.hgetall(self._key(run_id))
            return {k: json.loads(v) for k, v in raw.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis load_run failed for {}: {}", run_id, exc)
            return {}

    async def close(self) -> None:
        """Close the Redis connection (called on app shutdown)."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


# Module-level singleton shared by the agents and the FastAPI app.
redis_store = RedisStore()
