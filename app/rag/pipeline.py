from __future__ import annotations

import time
import uuid
from typing import Dict, Tuple

from openai import OpenAI

from app.config import settings
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.decider import decide
from app.rag.prompt import build as build_prompt
from app.rag.generator import Generator
from app.schemas import AskResponse, SourceItem, Usage


def _snippet(text: str, n: int = 320) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[:n].rstrip() + "…"


def answer(question: str) -> AskResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    q = (question or "").strip()
    if len(q) < 3:
        return AskResponse(
            ok=True,
            answer="I don't know based on the provided documents.",
            sources=[],
            latency_ms=0,
            usage=None,
            request_id=request_id,
        )

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # --- retriever deps ---
    embedder = Embedder(
        client=client,
        model=settings.EMBEDDING_MODEL,
        batch_size=1,
        timeout_s=settings.EMBED_TIMEOUT_S,
        retries=settings.EMBED_RETRIES,
        logger=None,
        price_input_per_1m=settings.PRICE_EMBED_INPUT,
    )

    retriever = Retriever(client=client, embedder=embedder)
    hits, _metrics = retriever.retrieve(q)

    # --- decider ---
    d = decide(hits)
    if not d.ok:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return AskResponse(
            ok=True,
            answer="I don't know based on the provided documents.",
            sources=[],
            latency_ms=latency_ms,
            usage=None,
            request_id=request_id,
        )

    # --- prompt ---
    messages = build_prompt(q, d.hits_used)

    gen = Generator(
        client=client,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        timeout_s=settings.LLM_TIMEOUT_S,
        retries=settings.LLM_RETRIES,
    )

    try:
        gen_res = gen.generate(messages)
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return AskResponse(
            ok=True,
            answer="I don't know based on the provided documents.",
            sources=[],
            latency_ms=latency_ms,
            usage=None,
            request_id=request_id,
        )

    # --- map hits by (doc_id, chunk_id, page) so citations can resolve ---
    hit_map: Dict[Tuple[str, int, int | None], SourceItem] = {}
    for h in d.hits_used:
        m = h.meta
        key = (m.doc_id, int(m.chunk_id), (int(m.page) if m.page is not None else None))
        hit_map[key] = SourceItem(
            doc_id=m.doc_id,
            source=m.path or m.doc_id,
            chunk_id=m.chunk_id,
            page=m.page,
            score=float(h.score),
            snippet=_snippet(h.text),
        )

    sources: list[SourceItem] = []

    # 1) Prefer citations from LLM (if any)
    for c in gen_res.data.citations:
        key = (c.doc_id, int(c.chunk_id), (int(c.page) if c.page is not None else None))
        if key in hit_map:
            sources.append(hit_map[key])
        else:
            # fallback: try match without page
            key2 = (c.doc_id, int(c.chunk_id), None)
            if key2 in hit_map:
                sources.append(hit_map[key2])

    # 2) If no citations, fallback to top N hits
    if not sources:
        top_n = int(getattr(settings, "TOP_N", min(3, int(settings.TOP_K))))
        for h in d.hits_used[:top_n]:
            m = h.meta
            sources.append(
                SourceItem(
                    doc_id=m.doc_id,
                    source=m.path or m.doc_id,
                    chunk_id=m.chunk_id,
                    page=m.page,
                    score=float(h.score),
                    snippet=_snippet(h.text),
                )
            )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    usage = Usage(**gen_res.usage) if gen_res.usage else None
    cost_usd = _calc_cost_usd(gen_res.usage)

    return AskResponse(
        ok=True,
        answer=gen_res.data.answer,
        sources=sources,
        latency_ms=latency_ms,
        usage=usage,
        cost_usd=cost_usd,
        request_id=request_id,
    )

def _calc_cost_usd(usage: dict | None) -> float:
    if not usage:
        return 0.0
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)

    cost = (inp / 1_000_000) * float(settings.PRICE_LLM_INPUT) + (out / 1_000_000) * float(settings.PRICE_LLM_OUTPUT)
    return round(cost, 6)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("question", type=str)
    args = p.parse_args()

    res = answer(args.question)
    print(res.model_dump_json(ensure_ascii=False, indent=2))