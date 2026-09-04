import asyncio
import json
import logging
from collections.abc import AsyncGenerator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

STUB_RESPONSE = "API key is not configured. Set OPENAI_API_KEY in .env"
REQUEST_RETRY_DELAYS = (0.25, 0.5, 1.0)


def _approx_token_count(text: str) -> int:
    return len(text) // 4


class LLMService:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = (base_url or settings.OPENAI_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL

    async def generate(self, messages: list[dict], max_tokens: int | None = None) -> str:
        if not self.api_key:
            return STUB_RESPONSE

        max_tokens = max_tokens or settings.MAX_TOKENS

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt, delay in enumerate((0.0, *REQUEST_RETRY_DELAYS), start=1):
                if delay:
                    await asyncio.sleep(delay)

                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content or ""
                except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                    logger.warning("LLM API transient error on attempt %s: %s", attempt, exc)
                    if attempt == len(REQUEST_RETRY_DELAYS) + 1:
                        return f"LLM request failed: {exc}"
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    if status_code >= 500 and attempt < len(REQUEST_RETRY_DELAYS) + 1:
                        logger.warning("LLM API server error on attempt %s: %s", attempt, status_code)
                        continue
                    logger.error("LLM API HTTP error: %s %s", status_code, exc.response.text)
                    return f"API error: {status_code}"
                except Exception as exc:
                    logger.exception("LLM API error: %s", exc)
                    return f"LLM request failed: {exc}"

        return "LLM request failed: unknown error"

    async def generate_stream(
        self, messages: list[dict], max_tokens: int | None = None
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield STUB_RESPONSE
            return

        max_tokens = max_tokens or settings.MAX_TOKENS

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt, delay in enumerate((0.0, *REQUEST_RETRY_DELAYS), start=1):
                if delay:
                    await asyncio.sleep(delay)

                try:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "stream": True,
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue

                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                return

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        return
                except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                    logger.warning("LLM stream transient error on attempt %s: %s", attempt, exc)
                    if attempt == len(REQUEST_RETRY_DELAYS) + 1:
                        yield f"LLM request failed: {exc}"
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    if status_code >= 500 and attempt < len(REQUEST_RETRY_DELAYS) + 1:
                        logger.warning("LLM stream server error on attempt %s: %s", attempt, status_code)
                        continue
                    logger.error("LLM stream HTTP error: %s %s", status_code, exc.response.text)
                    yield f"API error: {status_code}"
                    return
                except Exception as exc:
                    logger.exception("LLM stream error: %s", exc)
                    yield f"LLM request failed: {exc}"
                    return
