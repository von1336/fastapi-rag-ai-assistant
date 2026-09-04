import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from app.config import settings
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.embedding_service import EmbeddingService
import json

from app.utils.file_parser import extract_text
from app.utils.text_splitter import split_text

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, original_name: str) -> str:
        ext = Path(original_name).suffix
        return f"{uuid.uuid4()}{ext}"

    def validate_upload(self, original_name: str, file_content: bytes, content_type: str) -> None:
        suffix = Path(original_name).suffix.lower()
        if suffix in {".txt", ".md", ".csv"}:
            try:
                file_content.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError("Text files must be valid UTF-8") from exc
            return

        if suffix == ".pdf":
            if not file_content.startswith(b"%PDF"):
                raise ValueError("Invalid PDF file")
            return

        raise ValueError("Unsupported file type")

    async def save_upload(
        self,
        db: AsyncSession,
        file_content: bytes,
        original_name: str,
        content_type: str,
        user_id: str,
    ) -> Document:
        filename = self._generate_filename(original_name)
        filepath = self.upload_dir / filename

        filepath.write_bytes(file_content)
        size_bytes = len(file_content)

        doc = Document(
            user_id=user_id,
            filename=filename,
            original_name=original_name,
            content_type=content_type,
            size_bytes=size_bytes,
            status=DocumentStatus.uploaded,
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        return doc

    async def process_document(self, db: AsyncSession, document_id: str) -> None:
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        doc.status = DocumentStatus.processing
        await db.flush()

        try:
            filepath = self.upload_dir / doc.filename
            await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
            text = extract_text(
                str(filepath),
                doc.content_type,
                max_pdf_pages=settings.MAX_PDF_PAGES,
                max_extracted_chars=settings.MAX_EXTRACTED_TEXT_CHARS,
            )
            chunks = split_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

            for i, chunk_text in enumerate(chunks):
                embedding = await self.embedding_service.get_embedding(chunk_text)
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=chunk_text,
                    chunk_index=i,
                    embedding=json.dumps(embedding),
                )
                db.add(chunk)

            doc.status = DocumentStatus.ready
        except Exception as e:
            logger.exception("Document processing failed for %s: %s", document_id, e)
            doc.status = DocumentStatus.error
            raise
        finally:
            await db.flush()

    async def delete_document(self, db: AsyncSession, document_id: str) -> None:
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        filepath = self.upload_dir / doc.filename
        if filepath.exists():
            try:
                filepath.unlink()
            except OSError as e:
                logger.warning("Failed to delete file %s: %s", filepath, e)

        await db.delete(doc)
