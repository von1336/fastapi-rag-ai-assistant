from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.services.llm_service import LLMService


@pytest.mark.asyncio
@patch.object(LLMService, "generate", new_callable=AsyncMock, return_value="Mocked response")
async def test_chat_create_conversation(mock_generate, client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/chat",
        headers=auth_headers,
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert "message" in data
    assert data["message"]["content"] == "Mocked response"
    assert data["message"]["role"] == "assistant"


@pytest.mark.asyncio
@patch.object(LLMService, "generate", new_callable=AsyncMock, return_value="RAG answer")
async def test_chat_with_documents(mock_generate, client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/chat",
        headers=auth_headers,
        json={"message": "What is in my documents?", "use_documents": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"] == "RAG answer"


@pytest.mark.asyncio
async def test_chat_unauthorized(client: AsyncClient):
    response = await client.post(
        "/api/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 401
