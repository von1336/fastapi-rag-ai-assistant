from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.services.llm_service import LLMService


@pytest.mark.asyncio
@patch.object(LLMService, "generate", new_callable=AsyncMock, return_value="Hi")
async def test_list_conversations(mock_generate, client: AsyncClient, auth_headers):
    await client.post("/api/chat", headers=auth_headers, json={"message": "Hello"})

    response = await client.get("/api/conversations", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "title" in data[0]
    assert "id" in data[0]


@pytest.mark.asyncio
@patch.object(LLMService, "generate", new_callable=AsyncMock, return_value="Reply")
async def test_get_conversation_with_messages(mock_generate, client: AsyncClient, auth_headers):
    chat_resp = await client.post(
        "/api/chat",
        headers=auth_headers,
        json={"message": "Test message"},
    )
    conv_id = chat_resp.json()["conversation_id"]

    response = await client.get(
        f"/api/conversations/{conv_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == conv_id


@pytest.mark.asyncio
@patch.object(LLMService, "generate", new_callable=AsyncMock, return_value="Ok")
async def test_update_conversation(mock_generate, client: AsyncClient, auth_headers):
    chat_resp = await client.post(
        "/api/chat",
        headers=auth_headers,
        json={"message": "Hi"},
    )
    conv_id = chat_resp.json()["conversation_id"]

    response = await client.put(
        f"/api/conversations/{conv_id}",
        headers=auth_headers,
        json={"title": "Updated Title"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Title"


@pytest.mark.asyncio
@patch.object(LLMService, "generate", new_callable=AsyncMock, return_value="Bye")
async def test_delete_conversation(mock_generate, client: AsyncClient, auth_headers):
    chat_resp = await client.post(
        "/api/chat",
        headers=auth_headers,
        json={"message": "Create"},
    )
    conv_id = chat_resp.json()["conversation_id"]

    response = await client.delete(
        f"/api/conversations/{conv_id}",
        headers=auth_headers,
    )
    assert response.status_code == 204

    get_resp = await client.get(
        f"/api/conversations/{conv_id}",
        headers=auth_headers,
    )
    assert get_resp.status_code == 404
