from __future__ import annotations

from typing import List, Dict, Any

from app.rag.types import SearchHit

SYSTEM_RULES = """You are a strict RAG assistant.

You MUST return ONLY valid JSON (no markdown, no prose, no code fences).
The JSON must have exactly two keys: answer, citations.

Rules:
1) Use ONLY the provided context snippets.
2) If the context does not contain the answer, return:
   {"answer":"I don't know based on the provided documents.","citations":[]}
3) Do NOT use general knowledge. Do NOT guess. Do NOT invent.
4) If you answer, you MUST include at least 1 citation that directly supports the answer.
5) citations may reference ONLY the provided snippets by (doc_id, chunk_id, page).
6) For questions about order/which comes first, answer explicitly using the snippet wording (e.g. "Justification for Change appears before Description of Change").
"""

def _format_context(hits: List[SearchHit]) -> str:
    if not hits:
        return ""

    blocks: List[str] = []
    for i, h in enumerate(hits, start=1):
        m = h.meta
        page_str = f", page={m.page}" if m.page is not None else ""
        src = m.path or m.doc_id
        blocks.append(
            f"[{i}] doc_id={m.doc_id}, chunk_id={m.chunk_id}{page_str}, source={src}\n"
            f"snippet:\n{h.text}\n"
        )
    return "\n---\n".join(blocks)

def build(question: str, hits: List[SearchHit]) -> List[Dict[str, Any]]:
    question = (question or "").strip()
    context = _format_context(hits)

    user_prompt = f"""QUESTION:
{question}

CONTEXT SNIPPETS:
{context if context else "(no context snippets)"}

Return ONLY valid JSON in exactly this shape:
{{
  "answer": "string",
  "citations": [{{"doc_id": "string", "chunk_id": 0, "page": 1}}]
}}

Rules for citations:
- cite ONLY snippets that support the answer
- if unsure / not found in context -> answer must be:
  "I don't know based on the provided documents."
  and citations must be []
"""

    return [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": user_prompt},
    ]