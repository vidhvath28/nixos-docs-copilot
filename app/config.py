"""Runtime configuration, read from environment variables (or a local .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLM: "groq" (free tier) or "openai".
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # Judge model used by the eval harness. Defaults to the answer model.
    judge_model: str | None = None

    # Embeddings run locally (ONNX), so no API key is needed to index or retrieve.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Source corpus: the NixOS manual, pinned to one release.
    manual_url: str = "https://nixos.org/manual/nixos/stable/"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection: str = "nixos_manual"

    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 6
    # How many times the graph may rewrite the query when retrieval comes back irrelevant.
    max_rewrites: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
