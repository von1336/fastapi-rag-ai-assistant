from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = "development"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_assistant"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "development-secret-key-not-for-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    ALLOW_PUBLIC_REGISTRATION: bool = False
    ALLOWED_ORIGINS: list[str] = ["http://localhost", "http://127.0.0.1", "http://test"]
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1", "test"]
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    MAX_TOKENS: int = 2000
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_PDF_PAGES: int = 100
    MAX_EXTRACTED_TEXT_CHARS: int = 1_000_000
    LOGIN_RATE_LIMIT_PER_MINUTE: int = 5
    LOGIN_RATE_LIMIT_PER_HOUR: int = 20
    REGISTER_RATE_LIMIT_PER_HOUR: int = 3
    CHAT_RATE_LIMIT_PER_MINUTE: int = 30
    DOCUMENT_UPLOAD_RATE_LIMIT_PER_MINUTE: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    @field_validator("ALLOWED_ORIGINS", "ALLOWED_HOSTS", mode="before")
    @classmethod
    def split_csv_list(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def validate_security(self) -> None:
        if self.APP_ENV == "public-production":
            if len(self.SECRET_KEY) < 32 or "secret" in self.SECRET_KEY.lower():
                raise RuntimeError("SECRET_KEY must be set to a strong value in public-production mode")
            if not self.ALLOWED_ORIGINS:
                raise RuntimeError("ALLOWED_ORIGINS must be configured in public-production mode")
            if not self.ALLOWED_HOSTS:
                raise RuntimeError("ALLOWED_HOSTS must be configured in public-production mode")


settings = Settings()
