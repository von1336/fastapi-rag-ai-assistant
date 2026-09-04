import asyncio
import json
import logging
import math
import random
from hashlib import sha256

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

REQUEST_RETRY_DELAYS = (0.25, 0.5, 1.0)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingService:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = (base_url or settings.OPENAI_BASE_URL).rstrip("/")
        self.model = model or settings.EMBEDDING_MODEL

    @staticmethod
    def _seed_from_text(text: str) -> int:
        digest = sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def _build_fallback_embedding(self, text: str) -> list[float]:
        generator = random.Random(self._seed_from_text(text))
        return [generator.uniform(-1, 1) for _ in range(1536)]

    async def get_embedding(self, text: str) -> list[float]:
        if not self.api_key:
            return self._build_fallback_embedding(text)

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt, delay in enumerate((0.0, *REQUEST_RETRY_DELAYS), start=1):
                if delay:
                    await asyncio.sleep(delay)

                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "input": text[:8192],
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    embedding = data.get("data", [{}])[0].get("embedding", [])
                    return embedding
                except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                    logger.warning("Embedding API transient error on attempt %s: %s", attempt, exc)
                    if attempt == len(REQUEST_RETRY_DELAYS) + 1:
                        break
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    if status_code >= 500 and attempt < len(REQUEST_RETRY_DELAYS) + 1:
                        logger.warning(
                            "Embedding API server error on attempt %s: %s",
                            attempt,
                            status_code,
                        )
                        continue
                    logger.exception("Embedding API HTTP error: %s", status_code)
                    break
                except Exception as exc:
                    logger.exception("Embedding API error: %s", exc)
                    break

        return self._build_fallback_embedding(text)
