import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Document, DocumentChunk, User
from app.rate_limit import check_rate_limit
from app.schemas import (
    DocumentListItem,
    DocumentResponse,
    DocumentSearchResult,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])
document_service = DocumentService()
rag_service = RAGService()

ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}


def _check_file_type(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not await check_rate_limit(
        "document-upload-user",
        user.id,
        settings.DOCUMENT_UPLOAD_RATE_LIMIT_PER_MINUTE,
        60,
    ):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")

    if not file.filename or not _check_file_type(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File exceeds maximum allowed size",
        )

    try:
        document_service.validate_upload(
            file.filename,
            content,
            file.content_type or "application/octet-stream",
        )
        doc = await document_service.save_upload(
            db,
            content,
            file.filename,
            file.content_type or "application/octet-stream",
            user.id,
        )
        await document_service.process_document(db, doc.id)
        await db.refresh(doc)
        return doc
    except ValueError as e:
        logger.warning("Upload validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed",
        )


@router.get("", response_model=list[DocumentListItem])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.user_id == user.id).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [DocumentListItem.model_validate(d) for d in docs]


@router.get("/search", response_model=list[DocumentSearchResult])
async def search_documents(
    query: str,
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    normalized_query = query.strip()
    if not normalized_query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    safe_limit = max(1, min(limit, 20))
    matches = await rag_service.search_chunks(db, normalized_query, user.id, safe_limit)

    return [
        DocumentSearchResult(
            document_id=chunk.document_id,
            document_name=chunk.document.original_name,
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=round(score, 6),
        )
        for chunk, score in matches
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_count_result = await db.execute(
        select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    chunk_count = chunk_count_result.scalar() or 0

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        original_name=doc.original_name,
        content_type=doc.content_type,
        size_bytes=doc.size_bytes,
        status=doc.status.value,
        created_at=doc.created_at,
        chunk_count=chunk_count,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        await document_service.delete_document(db, document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Document not found")
