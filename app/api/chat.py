import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.database import async_session_maker, get_db
from app.models import Conversation, Message, MessageRole, User
from app.rate_limit import check_rate_limit
from app.schemas import ChatRequest, ChatResponse, ChatStreamEvent, MessageResponse
from app.services.llm_service import LLMService
from app.services.llm_service import _approx_token_count as approx_token_count
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
llm_service = LLMService()
rag_service = RAGService()


def _format_sse(event: ChatStreamEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def _get_or_create_conversation(
    db: AsyncSession, user_id: str, conversation_id: str | None
) -> Conversation:
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        conv = result.scalar_one_or_none()
        if conv:
            return conv
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = Conversation(
        user_id=user_id,
        title="New Conversation",
        model_name=settings.LLM_MODEL,
    )
    db.add(conv)
    await db.flush()
    await db.refresh(conv)
    return conv


async def _build_messages(
    db: AsyncSession,
    conversation: Conversation,
    user_message: str,
    use_documents: bool,
    user_id: str,
) -> list[dict]:
    messages: list[dict] = []

    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})

    if use_documents:
        chunks = await rag_service.get_relevant_chunks(
            db, user_message, user_id, settings.TOP_K_RESULTS
        )
        context = rag_service.build_context(chunks)
        if context:
            rag_prompt = (
                "Используй следующий контекст из загруженных документов для ответа.\n\n"
                "Контекст:\n"
                f"{context}\n\n"
                "Если контекст не содержит релевантной информации, ответь на основе своих знаний."
            )
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n{rag_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": rag_prompt})

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    history = result.scalars().all()
    for msg in history:
        messages.append({"role": msg.role.value, "content": msg.content})

    # Only append if not already present in the history
    if not history or history[-1].content != user_message or history[-1].role != MessageRole.user:
        messages.append({"role": "user", "content": user_message})
    return messages


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not await check_rate_limit(
        "chat-user",
        user.id,
        settings.CHAT_RATE_LIMIT_PER_MINUTE,
        60,
    ):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")

    conversation = await _get_or_create_conversation(
        db, user.id, request.conversation_id
    )

    user_msg = Message(
        conversation_id=conversation.id,
        role=MessageRole.user,
        content=request.message,
    )
    db.add(user_msg)
    await db.flush()

    messages = await _build_messages(
        db, conversation, request.message, request.use_documents, user.id
    )

    response_text = await llm_service.generate(messages)
    token_count = approx_token_count(response_text)

    assistant_msg = Message(
        conversation_id=conversation.id,
        role=MessageRole.assistant,
        content=response_text,
        token_count=token_count,
    )
    db.add(assistant_msg)
    await db.flush()
    await db.refresh(assistant_msg)

    return ChatResponse(
        conversation_id=conversation.id,
        message=MessageResponse.model_validate(assistant_msg),
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not await check_rate_limit(
        "chat-user",
        user.id,
        settings.CHAT_RATE_LIMIT_PER_MINUTE,
        60,
    ):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")

    conversation = await _get_or_create_conversation(
        db, user.id, request.conversation_id
    )

    user_msg = Message(
        conversation_id=conversation.id,
        role=MessageRole.user,
        content=request.message,
    )
    db.add(user_msg)
    await db.flush()

    messages = await _build_messages(
        db, conversation, request.message, request.use_documents, user.id
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        full_text = ""
        try:
            async for token in llm_service.generate_stream(messages):
                full_text += token
                yield _format_sse(ChatStreamEvent(type="token", content=token))
            yield _format_sse(ChatStreamEvent(type="done", content=full_text))

            async with async_session_maker() as save_session:
                assistant_msg = Message(
                    conversation_id=conversation.id,
                    role=MessageRole.assistant,
                    content=full_text,
                    token_count=approx_token_count(full_text),
                )
                save_session.add(assistant_msg)
                await save_session.commit()
        except Exception as e:
            logger.exception("Stream error: %s", e)
            yield _format_sse(ChatStreamEvent(type="error", content=str(e)))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
