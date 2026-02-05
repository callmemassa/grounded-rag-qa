from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import settings
from app.rag.embedder import Embedder
from app.rag.store_faiss import load_index, query as faiss_query
from app.rag.types import SearchHit, ChunkMeta

ART_DIR = Path("data/artifacts/index")
FAISS_PATH = ART_DIR / "faiss.index"
CHUNKS_JSONL = ART_DIR / "chunks.jsonl"


def _load_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _snippet(text: str, n: int = 360) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[:n].rstrip() + "…"


@dataclass
class RetrieveMetrics:
    embed_latency_ms: int
    search_latency_ms: int
    total_latency_ms: int
    candidates: int
    returned: int
    top_score: float | None


class Retriever:
    def __init__(
        self,
        *,
        client: OpenAI,
        embedder: Embedder,
        index_path: Path = FAISS_PATH,
        chunks_path: Path = CHUNKS_JSONL,
        min_score: float | None = None,
        top_k: int | None = None,
    ):
        self.client = client
        self.embedder = embedder
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.min_score = float(min_score) if min_score is not None else float(settings.MIN_SCORE)
        self.top_k = int(top_k) if top_k is not None else int(settings.TOP_K)

        if not self.index_path.exists():
            raise RuntimeError(f"Missing FAISS index: {self.index_path}")
        if not self.chunks_path.exists():
            raise RuntimeError(f"Missing chunks.jsonl: {self.chunks_path}")

        self._index = load_index(self.index_path)
        self._rows = _load_chunks_jsonl(self.chunks_path)
        if not self._rows:
            raise RuntimeError("chunks.jsonl is empty")

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[list[SearchHit], RetrieveMetrics]:
        q = (query or "").strip()
        if not q:
            raise ValueError("Empty query")
        if len(q) < 3:
            raise ValueError("Query too short (< 3 chars)")

        k = int(top_k) if top_k is not None else self.top_k
        if k <= 0:
            raise ValueError("top_k must be > 0")

        t0 = time.perf_counter()

        # 1) embed query
        emb_t0 = time.perf_counter()
        emb_res = self.embedder.embed_texts([q])
        embed_latency_ms = int((time.perf_counter() - emb_t0) * 1000)

        if not emb_res.vectors:
            raise RuntimeError("Failed to embed query")

        q_vec = emb_res.vectors[0]

        # 2) faiss query -> scores, ids
        srch_t0 = time.perf_counter()
        scores, ids = faiss_query(self._index, q_vec, top_k=k)
        search_latency_ms = int((time.perf_counter() - srch_t0) * 1000)

        # 3) zip + filter by min_score
        candidates = 0
        pairs: list[tuple[int, float]] = []
        for s, i in zip(scores, ids):
            if not isinstance(i, int) or i < 0:
                continue
            candidates += 1
            sf = float(s)
            if sf >= self.min_score:
                pairs.append((i, sf))

        # sort by score desc
        pairs.sort(key=lambda x: x[1], reverse=True)

        # 4) build SearchHit objects
        hits: list[SearchHit] = []
        top_score: float | None = None

        for idx, score in pairs:
            if idx >= len(self._rows):
                continue

            row = self._rows[idx]
            text = row.get("text", "") or ""
            meta_dict = row.get("meta", {}) or {}

            # validate/normalize meta using Pydantic model
            meta = ChunkMeta(**meta_dict)

            if top_score is None:
                top_score = score

            hits.append(
                SearchHit(
                    score=score,
                    text=(text),
                    meta=meta,
                )
            )

        total_latency_ms = int((time.perf_counter() - t0) * 1000)
        metrics = RetrieveMetrics(
            embed_latency_ms=embed_latency_ms,
            search_latency_ms=search_latency_ms,
            total_latency_ms=total_latency_ms,
            candidates=candidates,
            returned=len(hits),
            top_score=top_score,
        )
        return hits, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retriever: query -> top-k chunks (FAISS)")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument("--top_k", type=int, default=None, help="Override TOP_K from env")
    parser.add_argument("--min_score", type=float, default=None, help="Override MIN_SCORE from env")
    args = parser.parse_args(argv)

    # init OpenAI + embedder
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    embedder = Embedder(
        client=client,
        model=settings.EMBEDDING_MODEL,
        batch_size=1,
        timeout_s=settings.EMBED_TIMEOUT_S,
        retries=settings.EMBED_RETRIES,
        logger=None,
        price_input_per_1m=settings.PRICE_EMBED_INPUT,
    )

    retriever = Retriever(
        client=client,
        embedder=embedder,
        min_score=args.min_score,
        top_k=args.top_k,
    )

    try:
        hits, m = retriever.retrieve(args.query, top_k=args.top_k)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 2

    q = (args.query or "").strip()
    print(f"\nQUERY: {q}")
    print(f"TOP_K: {args.top_k if args.top_k is not None else settings.TOP_K}")
    print(f"MIN_SCORE: {args.min_score if args.min_score is not None else settings.MIN_SCORE}\n")

    if not hits:
        print("NOT FOUND: no relevant chunks above threshold.\n")
    else:
        for rank, h in enumerate(hits, start=1):
            src = h.meta.path or h.meta.doc_id
            print(f"[{rank}] score={h.score:.4f}")
            print(f"    source: {src}")
            print(f"    doc_id: {h.meta.doc_id}")
            print(f"    chunk_id: {h.meta.chunk_id}   page: {h.meta.page}")
            print(f"    snippet: {_snippet(h.text)}")
            print()

    print(
        "METRICS:"
        f" embed_latency_ms={m.embed_latency_ms}"
        f" search_latency_ms={m.search_latency_ms}"
        f" total_latency_ms={m.total_latency_ms}"
        f" candidates={m.candidates}"
        f" returned={m.returned}"
        f" top_score={(f'{m.top_score:.4f}' if m.top_score is not None else 'None')}"
    )
    print("\nOK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())