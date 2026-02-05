from __future__ import annotations

from typing import Optional, Literal, List
from pydantic import BaseModel, Field


SourceType = Literal["pdf", "txt", "md", "docx", "html", "unknown"]


class Chunk(BaseModel):
    text: str = Field(min_length=1)


class ChunkMeta(BaseModel):
    doc_id: str = Field(min_length=1)          # e.g. "DOC-015" / "ISO-9001"
    source: SourceType = "unknown"
    chunk_id: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    page: Optional[int] = Field(default=None, ge=1)

    path: Optional[str] = None


class SearchHit(BaseModel):
    score: float
    text: str
    meta: ChunkMeta


class EmbeddedChunk(BaseModel):
    vector: List[float]
    chunk: Chunk
    meta: ChunkMeta