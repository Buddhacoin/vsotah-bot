"""Performance / Scale helpers for VSotahBot.

This module is intentionally optional-safe:
- if REDIS_URL is not configured, everything falls back to small in-memory helpers;
- if Redis is temporarily down, the bot keeps working;
- event logging can use a background queue so user-facing handlers do not wait for DB writes.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import redis.asyncio as redis


REDIS_URL = (
    os.getenv("REDIS_URL")
    or os.getenv("REDIS_PRIVATE_URL")
    or os.getenv("REDISCLOUD_URL")
)

CACHE_PREFIX = os.getenv("CACHE_PREFIX", "vsotah")
EVENT_QUEUE_MAXSIZE = int(os.getenv("EVENT_QUEUE_MAXSIZE", "2000"))
EVENT_BATCH_SIZE = int(os.getenv("EVENT_BATCH_SIZE", "50"))
EVENT_FLUSH_SECONDS = float(os.getenv("EVENT_FLUSH_SECONDS", "1.0"))

redis_client: redis.Redis | None = None
redis_available = False
_local_cache: dict[str, tuple[float, Any]] = {}
_event_queue: asyncio.Queue[tuple[int | None, str, str]] | None = None
_event_worker_task: asyncio.Task | None = None


def _key(key: str) -> str:
    return f"{CACHE_PREFIX}:{key}"


async def init_performance_layer() -> bool:
    """Initialise optional Redis cache.

    Returns True when Redis is configured and ping succeeds.
    """
    global redis_client, redis_available

    if not REDIS_URL:
        redis_client = None
        redis_available = False
        return False

    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        redis_available = True
        return True
    except Exception as e:
        print(f"REDIS DISABLED: {e}")
        redis_client = None
        redis_available = False
        return False


async def close_performance_layer() -> None:
    global redis_client, redis_available, _event_worker_task

    if _event_worker_task:
        _event_worker_task.cancel()
        try:
            await _event_worker_task
        except Exception:
            pass
        _event_worker_task = None

    if redis_client:
        try:
            await redis_client.aclose()
        except Exception:
            pass

    redis_client = None
    redis_available = False


async def cache_get(key: str) -> Any | None:
    now = time.time()
    local_item = _local_cache.get(key)
    if local_item:
        expires_at, value = local_item
        if expires_at > now:
            return value
        _local_cache.pop(key, None)

    if redis_client and redis_available:
        try:
            raw = await redis_client.get(_key(key))
            if raw is not None:
                return json.loads(raw)
        except Exception:
            pass

    return None


async def cache_set(key: str, value: Any, ttl_seconds: int = 30) -> None:
    expires_at = time.time() + ttl_seconds
    _local_cache[key] = (expires_at, value)

    # keep the local fallback small
    if len(_local_cache) > 1000:
        stale_keys = [k for k, (expires, _) in _local_cache.items() if expires <= time.time()]
        for stale_key in stale_keys[:300]:
            _local_cache.pop(stale_key, None)

    if redis_client and redis_available:
        try:
            await redis_client.setex(_key(key), ttl_seconds, json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            pass


async def cache_delete(key: str) -> None:
    _local_cache.pop(key, None)
    if redis_client and redis_available:
        try:
            await redis_client.delete(_key(key))
        except Exception:
            pass


async def redis_health() -> tuple[bool, int | None, str]:
    if not REDIS_URL:
        return False, None, "not configured"
    if not redis_client:
        return False, None, "client not initialised"
    try:
        start = time.perf_counter()
        await redis_client.ping()
        latency = round((time.perf_counter() - start) * 1000)
        return True, latency, "OK"
    except Exception as e:
        return False, None, str(e)[:120]


async def start_event_worker(db_pool: Any) -> None:
    """Start background DB writer for events.

    This keeps Telegram handlers fast because log_event can enqueue and return.
    """
    global _event_queue, _event_worker_task
    if _event_worker_task:
        return

    _event_queue = asyncio.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
    _event_worker_task = asyncio.create_task(_event_worker(db_pool))


async def enqueue_event(telegram_id: int | None, event_type: str, details: str = "") -> bool:
    if not _event_queue:
        return False

    try:
        _event_queue.put_nowait((telegram_id, event_type, details[:1000]))
        return True
    except asyncio.QueueFull:
        return False


async def event_queue_size() -> int:
    if not _event_queue:
        return 0
    return _event_queue.qsize()


async def _event_worker(db_pool: Any) -> None:
    assert _event_queue is not None
    buffer: list[tuple[int | None, str, str]] = []

    async def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        rows = buffer
        buffer = []
        try:
            async with db_pool.acquire() as conn:
                await conn.executemany(
                    "INSERT INTO events (telegram_id, event_type, details) VALUES ($1, $2, $3)",
                    rows,
                )
        except Exception as e:
            print(f"EVENT WORKER DB ERROR: {e}")

    while True:
        try:
            item = await asyncio.wait_for(_event_queue.get(), timeout=EVENT_FLUSH_SECONDS)
            buffer.append(item)
            if len(buffer) >= EVENT_BATCH_SIZE:
                await flush()
        except asyncio.TimeoutError:
            await flush()
        except asyncio.CancelledError:
            await flush()
            raise
        except Exception as e:
            print(f"EVENT WORKER ERROR: {e}")
            await asyncio.sleep(1)
