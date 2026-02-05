from __future__ import annotations

import time
import uuid
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.schemas import AskRequest, AskResponse
from app.utils.logging import setup_logging
from app.rag.pipeline import answer as rag_answer


log = setup_logging()
app = FastAPI(title="RAG QA Service", version="1.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, __: RequestValidationError):

    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "request_id": str(uuid.uuid4()),
            "answer": "",
            "sources": [],
            "latency_ms": 0,
            "usage": None,
            "cost_usd": None,
        },
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    q = (req.question or "").strip()

    if not q:
        return JSONResponse(
            status_code=422,
            content=AskResponse(
                ok=False,
                request_id=request_id,
                answer="",
                sources=[],
                latency_ms=0,
                usage=None,
                cost_usd=None,
            ).model_dump(),
        )

    log.info("REQ /ask | request_id=%s | q_len=%s", request_id, len(q))

    try:
        res = rag_answer(q)
        res.request_id = request_id

        latency_ms = int((time.perf_counter() - t0) * 1000)
        res.latency_ms = latency_ms

        tokens = (res.usage.total_tokens if res.usage else None)
        log.info(
            "RES /ask | request_id=%s | status=%s | latency_ms=%s | sources=%s | tokens=%s | cost_usd=%.6f",
            request_id, "ok", res.latency_ms, len(res.sources), tokens, float(res.cost_usd or 0.0)
        )

        return JSONResponse(status_code=200, content=res.model_dump())

    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.exception(
            "RES /ask | request_id=%s | status=error | latency_ms=%s | err=%s",
            request_id,
            latency_ms,
            f"{type(e).__name__}: {e}",
        )


        return JSONResponse(
            status_code=500,
            content=AskResponse(
                ok=False,
                request_id=request_id,
                answer="",
                sources=[],
                latency_ms=latency_ms,
                usage=None,
                cost_usd=None,
            ).model_dump(),
        )


@app.get("/")
def root():
    return {"ok": True}