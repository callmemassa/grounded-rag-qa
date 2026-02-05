from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class SourceItem(BaseModel):
    doc_id: str
    source: str
    chunk_id: int
    score: float
    page: Optional[int] = None
    snippet: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AskResponse(BaseModel):
    ok: bool = True
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    latency_ms: int = 0
    usage: Optional[Usage] = None
    cost_usd: float = 0.0
    request_id: Optional[str] = None