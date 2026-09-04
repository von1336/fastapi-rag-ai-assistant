import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Document, DocumentChunk, DocumentStatus
from app.services.embedding_service import EmbeddingService, cosine_similarity

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()

    async def get_relevant_chunks(
        self,
        db: AsyncSession,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        query_embedding = await self.embedding_service.get_embedding(query)

        result = await db.execute(
            select(DocumentChunk)
            .join(Document)
            .where(Document.user_id == user_id, Document.status == DocumentStatus.ready)
        )
        chunks = result.scalars().all()

        scored: list[tuple[DocumentChunk, float]] = []
        for chunk in chunks:
            try:
                emb = json.loads(chunk.embedding)
                score = cosine_similarity(query_embedding, emb)
                scored.append((chunk, score))
            except (json.JSONDecodeError, TypeError):
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    async def search_chunks(
        self,
        db: AsyncSession,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        query_embedding = await self.embedding_service.get_embedding(query)

        result = await db.execute(
            select(DocumentChunk)
            .options(selectinload(DocumentChunk.document))
            .join(Document)
            .where(Document.user_id == user_id, Document.status == DocumentStatus.ready)
        )
        chunks = result.scalars().all()

        scored: list[tuple[DocumentChunk, float]] = []
        for chunk in chunks:
            try:
                emb = json.loads(chunk.embedding)
                score = cosine_similarity(query_embedding, emb)
                scored.append((chunk, score))
            except (json.JSONDecodeError, TypeError):
                continue

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def build_context(self, chunks: list[DocumentChunk]) -> str:
        if not chunks:
            return ""
        parts = [f"[Документ {i + 1}]\n{c.content}" for i, c in enumerate(chunks)]
        return "\n\n---\n\n".join(parts)
