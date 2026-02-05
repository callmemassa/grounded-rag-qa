from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from app.config import settings
from app.rag.ingest import iter_documents
from app.rag.chunking import chunk_text
from app.rag.embedder import Embedder
from app.rag.store_faiss import build_index, save_index
from app.rag.types import EmbeddedChunk


DOCS_DIR = Path("data/docs")
OUT_DIR = Path("data/artifacts/index")

FAISS_PATH = OUT_DIR / "faiss.index"
CHUNKS_JSONL = OUT_DIR / "chunks.jsonl"
STATS_JSON = OUT_DIR / "stats.json"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DOCS_DIR.exists():
        raise RuntimeError(f"Missing folder: {DOCS_DIR.resolve()}")

    # --- OpenAI client + embedder (ONE TIME) ---
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    embedder = Embedder(
        client=client,
        model=settings.EMBEDDING_MODEL,
        batch_size=settings.EMBED_BATCH_SIZE,
        timeout_s=settings.EMBED_TIMEOUT_S,
        retries=settings.EMBED_RETRIES,
        logger=None,
        price_input_per_1m=settings.PRICE_EMBED_INPUT,
    )

    embedded_chunks: list[EmbeddedChunk] = []
    seen_files: set[str] = set()

    # --- 1) ingest -> chunk ---
    for path, doc_id, page, text in iter_documents(DOCS_DIR):
        seen_files.add(str(path))

        if not text or not text.strip():
            continue

        source_ext = path.suffix.lower().lstrip(".")  # "pdf"/"txt"/"md"

        pairs = chunk_text(
            text,
            doc_id=doc_id,
            source=source_ext,
            path=str(path),
            page=page,
            chunk_size_chars=settings.CHUNK_SIZE_CHARS,
            overlap_chars=settings.OVERLAP_CHARS,
        )

        for chunk, meta in pairs:
            embedded_chunks.append(EmbeddedChunk(vector=[], chunk=chunk, meta=meta))

    if not embedded_chunks:
        raise RuntimeError("No chunks produced (all docs empty?)")

    # --- 2) embeddings ---
    texts = [ec.chunk.text for ec in embedded_chunks]
    emb_res = embedder.embed_texts(texts)
    vectors = emb_res.vectors

    if len(vectors) != len(embedded_chunks):
        raise RuntimeError(f"Vectors mismatch: {len(vectors)} != {len(embedded_chunks)}")

    for ec, v in zip(embedded_chunks, vectors):
        ec.vector = v

    # --- 3) build + save FAISS ---
    vectors_only = [ec.vector for ec in embedded_chunks]
    index = build_index(vectors_only)
    save_index(index, FAISS_PATH)

    # --- 4) write chunks.jsonl ---
    write_jsonl(
        CHUNKS_JSONL,
        ({"text": ec.chunk.text, "meta": ec.meta.model_dump()} for ec in embedded_chunks),
    )

    # --- 5) stats.json ---
    dt = time.perf_counter() - t0

    embed_cost_usd = 0.0
    if emb_res.usage and settings.PRICE_EMBED_INPUT:
        input_tokens = int(emb_res.usage.get("input_tokens", 0) or 0)
        embed_cost_usd = round((input_tokens / 1_000_000) * float(settings.PRICE_EMBED_INPUT), 6)

    stats = {
        "documents": len(seen_files),
        "chunks": len(embedded_chunks),
        "build_time_sec": round(dt, 3),
        "chunk_size_chars": settings.CHUNK_SIZE_CHARS,
        "overlap_chars": settings.OVERLAP_CHARS,
        "embed_model": emb_res.model,
        "embed_batches": emb_res.batches,
        "embed_latency_ms": emb_res.latency_ms,
        "embed_usage": emb_res.usage,
        "embed_cost_usd": embed_cost_usd,
        "faiss_index_path": str(FAISS_PATH),
        "chunks_path": str(CHUNKS_JSONL),
    }

    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()