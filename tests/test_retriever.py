from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.retriever import Retriever
from app.rag.store_faiss import build_index, save_index


class DummyEmbedder:
    def __init__(self, vec: list[float]):
        self._vec = vec

    def embed_texts(self, texts: list[str]):
        return SimpleNamespace(vectors=[self._vec])


def _write_chunks_jsonl(path: Path) -> None:
    rows = [
        {
            "text": "Fonts: use Arial for notes. Font sizes and styles are specified.",
            "meta": {"doc_id": "DOC-001", "source": "pdf", "chunk_id": 0, "page": 10, "path": "data/docs/DOC-001.pdf",
                     "start_char": 0, "end_char": 100},
        },
        {
            "text": "Title block requirements: include project name and sheet number.",
            "meta": {"doc_id": "DOC-001", "source": "pdf", "chunk_id": 1, "page": 18, "path": "data/docs/DOC-001.pdf",
                     "start_char": 101, "end_char": 220},
        },
        {
            "text": "Unrelated: concrete mix design table.",
            "meta": {"doc_id": "DOC-001", "source": "pdf", "chunk_id": 2, "page": 55, "path": "data/docs/DOC-001.pdf",
                     "start_char": 221, "end_char": 320},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _prepare_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "faiss.index"

    _write_chunks_jsonl(chunks_path)

    vectors = [
        [1.0, 0.0, 0.0],  # chunk 0
        [0.9, 0.1, 0.0],  # chunk 1
        [0.0, 1.0, 0.0],  # chunk 2
    ]
    index = build_index(vectors)
    save_index(index, index_path)

    return index_path, chunks_path


def test_retriever_fail_fast_empty_query(tmp_path: Path):
    index_path, chunks_path = _prepare_artifacts(tmp_path)

    r = Retriever(
        client=None,
        embedder=DummyEmbedder([1.0, 0.0, 0.0]),
        index_path=index_path,
        chunks_path=chunks_path,
        min_score=0.0,
        top_k=2,
    )

    with pytest.raises(ValueError):
        r.retrieve("")


def test_retriever_reject_short_query(tmp_path: Path):
    index_path, chunks_path = _prepare_artifacts(tmp_path)

    r = Retriever(
        client=None,
        embedder=DummyEmbedder([1.0, 0.0, 0.0]),
        index_path=index_path,
        chunks_path=chunks_path,
        min_score=0.0,
        top_k=2,
    )

    with pytest.raises(ValueError):
        r.retrieve("hi")  # < 3 chars


def test_retriever_returns_sorted_hits_and_threshold(tmp_path: Path):
    index_path, chunks_path = _prepare_artifacts(tmp_path)

    r = Retriever(
        client=None,
        embedder=DummyEmbedder([1.0, 0.0, 0.0]),
        index_path=index_path,
        chunks_path=chunks_path,
        min_score=0.0,
        top_k=3,
    )

    hits, metrics = r.retrieve("fonts requirements")

    assert metrics.returned >= 1

    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))

    assert hits[0].meta.chunk_id == 0

    r2 = Retriever(
        client=None,
        embedder=DummyEmbedder([1.0, 0.0, 0.0]),
        index_path=index_path,
        chunks_path=chunks_path,
        min_score=0.9999,  # почти 1.0
        top_k=3,
    )
    hits2, _ = r2.retrieve("fonts requirements")

    assert len(hits2) <= 1