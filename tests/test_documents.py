from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.api import documents
from app.services.embedding_service import EmbeddingService


@pytest.mark.asyncio
@patch.object(EmbeddingService, "get_embedding", new_callable=AsyncMock)
async def test_upload_document(mock_embedding, client: AsyncClient, auth_headers, tmp_path):
    mock_embedding.return_value = [0.1] * 1536
    tmp_path.mkdir(parents=True, exist_ok=True)

    with patch.object(documents.document_service, "upload_dir", tmp_path):
        files = {"file": ("test.txt", b"Hello world content", "text/plain")}
        response = await client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files=files,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["original_name"] == "test.txt"
    assert data["status"] == "ready"
    assert "id" in data


@pytest.mark.asyncio
async def test_upload_invalid_type(client: AsyncClient, auth_headers):
    files = {"file": ("test.exe", b"binary", "application/octet-stream")}
    response = await client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files=files,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
@patch.object(EmbeddingService, "get_embedding", new_callable=AsyncMock)
async def test_list_documents(mock_embedding, client: AsyncClient, auth_headers, tmp_path):
    mock_embedding.return_value = [0.1] * 1536
    tmp_path.mkdir(parents=True, exist_ok=True)

    with patch.object(documents.document_service, "upload_dir", tmp_path):
        files = {"file": ("doc.txt", b"Content", "text/plain")}
        await client.post("/api/documents/upload", headers=auth_headers, files=files)

    response = await client.get("/api/documents", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["original_name"] == "doc.txt"


@pytest.mark.asyncio
@patch.object(EmbeddingService, "get_embedding", new_callable=AsyncMock)
async def test_search_documents(mock_embedding, client: AsyncClient, auth_headers, tmp_path):
    mock_embedding.side_effect = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    tmp_path.mkdir(parents=True, exist_ok=True)

    with patch.object(documents.document_service, "upload_dir", tmp_path):
        files = {"file": ("doc.txt", b"Important alpha information", "text/plain")}
        upload_response = await client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files=files,
        )

    assert upload_response.status_code == 200

    response = await client.get(
        "/api/documents/search",
        headers=auth_headers,
        params={"query": "alpha", "limit": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["document_name"] == "doc.txt"
    assert data[0]["score"] == 1.0
    assert "alpha" in data[0]["content"].lower()


@pytest.mark.asyncio
async def test_search_documents_empty_query(client: AsyncClient, auth_headers):
    response = await client.get(
        "/api/documents/search",
        headers=auth_headers,
        params={"query": "   "},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_documents_unauthorized(client: AsyncClient):
    response = await client.get("/api/documents")
    assert response.status_code == 401
