from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    username: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ConversationCreate(BaseModel):
    title: str = Field(default="New Conversation", max_length=255)
    system_prompt: str | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    user_id: str
    model_name: str
    system_prompt: str | None
    created_at: datetime
    updated_at: datetime


class ConversationListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationUpdate(BaseModel):
    title: str | None = Field(None, max_length=255)
    system_prompt: str | None = None


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    token_count: int | None
    created_at: datetime

    @field_validator("role", mode="before")
    @classmethod
    def role_to_str(cls, v):
        return v.value if hasattr(v, "value") else v


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    use_documents: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    message: MessageResponse


class ChatStreamEvent(BaseModel):
    type: str = Field(..., pattern="^(token|done|error)$")
    content: str = ""


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_name: str
    content_type: str
    size_bytes: int
    status: str
    created_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def status_to_str(cls, v):
        return v.value if hasattr(v, "value") else v


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_name: str
    content_type: str
    size_bytes: int
    status: str
    created_at: datetime
    chunk_count: int = 0


class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_name: str
    status: str
    created_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def status_to_str(cls, v):
        return v.value if hasattr(v, "value") else v


class DocumentSearchResult(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    content: str
    score: float
