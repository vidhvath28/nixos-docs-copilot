"""Chat model + vector store factories."""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from app.config import get_settings
from app.embeddings import FastEmbedEmbeddings


def make_chat_model(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    s = get_settings()
    name = model or s.llm_model
    if s.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=name, temperature=temperature)
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=temperature)
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r} (expected 'groq' or 'openai')")


@lru_cache
def get_vectorstore() -> Chroma:
    s = get_settings()
    if not (s.chroma_dir / "chroma.sqlite3").exists():
        raise RuntimeError(f"no index at {s.chroma_dir} - run `python -m app.ingest` first")
    return Chroma(
        collection_name=s.collection,
        embedding_function=FastEmbedEmbeddings(s.embedding_model),
        persist_directory=str(s.chroma_dir),
    )
