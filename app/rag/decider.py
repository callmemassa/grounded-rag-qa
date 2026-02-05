from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from app.config import settings
from app.rag.types import SearchHit


@dataclass(frozen=True)
class Decision:
    ok: bool
    reason: str
    top_score: Optional[float]
    hits_used: List[SearchHit]


def decide(hits: List[SearchHit]) -> Decision:
    if not hits:
        return Decision(ok=False, reason="no_hits", top_score=None, hits_used=[])

    top_score = float(max(h.score for h in hits))

    if top_score < float(settings.MIN_SCORE):
        return Decision(ok=False, reason="low_score", top_score=top_score, hits_used=[])

    min_hits = int(getattr(settings, "MIN_HITS", 1))
    if len(hits) < min_hits:
        return Decision(ok=False, reason="too_few_hits", top_score=top_score, hits_used=[])

    max_ctx = int(getattr(settings, "MAX_CONTEXT_HITS", settings.TOP_K))
    return Decision(ok=True, reason="enough_context", top_score=top_score, hits_used=hits[:max_ctx])