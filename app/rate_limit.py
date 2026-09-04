import time
from collections import deque

from fastapi import Request
import redis.asyncio as redis

from app.config import settings

_memory_events: dict[str, deque[float]] = {}
_redis: redis.Redis | None = None


async def get_rate_limit_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def close_rate_limit_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def check_rate_limit(bucket: str, identifier: str, limit: int, window_seconds: int) -> bool:
    if limit <= 0:
        return True

    key = f"ratelimit:{bucket}:{identifier}"
    try:
        rate_redis = await get_rate_limit_redis()
        current = await rate_redis.incr(key)
        if current == 1:
            await rate_redis.expire(key, window_seconds)
        return current <= limit
    except Exception:
        now = time.monotonic()
        events = _memory_events.setdefault(key, deque())
        while events and events[0] <= now - window_seconds:
            events.popleft()
        if len(events) >= limit:
            return False
        events.append(now)
        return True


def reset_rate_limits() -> None:
    _memory_events.clear()
