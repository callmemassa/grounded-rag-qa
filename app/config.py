from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"{name} is not set")
    return v


class Settings:
    # --- Auth ---
    OPENAI_API_KEY: str = _req("OPENAI_API_KEY")

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.40"))

    # --- Chunking ---
    CHUNK_SIZE_CHARS: int = int(os.getenv("CHUNK_SIZE_CHARS", "1000"))
    OVERLAP_CHARS: int = int(os.getenv("OVERLAP_CHARS", "160"))

    # --- Embeddings ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    EMBED_TIMEOUT_S: float = float(os.getenv("EMBED_TIMEOUT_S", "60"))
    EMBED_RETRIES: int = int(os.getenv("EMBED_RETRIES", "2"))
    PRICE_EMBED_INPUT: float = float(os.getenv("PRICE_EMBED_INPUT", "0"))

    # --- LLM generation (RAG answer) ---
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "400"))
    LLM_TIMEOUT_S: float = float(os.getenv("LLM_TIMEOUT_S", "30"))
    LLM_RETRIES: int = int(os.getenv("LLM_RETRIES", "2"))

    # --- Pricing ---
    PRICE_LLM_INPUT: float = float(os.getenv("PRICE_LLM_INPUT", "0"))
    PRICE_LLM_OUTPUT: float = float(os.getenv("PRICE_LLM_OUTPUT", "0"))
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1")

    SYSTEM_PROMPT: str = os.getenv(
        "SYSTEM_PROMPT",
        "You answer using ONLY the provided context. If context is insufficient, say you don't know."
    )


settings = Settings()