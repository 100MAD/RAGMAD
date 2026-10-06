from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://ragmad:ragmad@localhost:5433/ragmad"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ragmad_chunks"

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "ragmad"
    s3_region: str = "us-east-1"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "not-set"
    llm_model: str = "gpt-4o-mini"
    llm_context_window: int = 128000

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )
    chunk_max_tokens: int = 512
    similarity_top_k: int = 5

    cors_origins: str = "http://localhost:5173"

    eval_llm_base_url: str | None = None
    eval_llm_api_key: str | None = None
    eval_llm_model: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
