# RAG QA — Grounded Question Answering over Documents

Production-minded Retrieval-Augmented Generation (RAG) system for asking questions over internal documents without hallucinations.

The system retrieves relevant document fragments, decides whether there is enough context, and only then generates an answer.
If the answer is not grounded in documents, it explicitly replies “I don’t know based on the provided documents.”

⸻

## What does this solve?
- 🔍 Search and QA over company documents (PDF / TXT / MD)
- 🚫 No hallucinations — strict grounding in sources
- 🧠 Clear separation of responsibilities:
  - retrieval
  - decision logic
  - generation
- 📊 Built-in evaluation and metrics
- 🧪 Tests included

**Typical use cases:**
- Internal knowledge base
- Engineering / standards documentation
- Compliance & procedures
- Corporate policies

Typical use cases:
- Internal knowledge base
- Engineering / standards documentation
- Compliance & procedures
- Corporate policies

⸻

## High-level pipeline

- documents
- ingest → chunking → embeddings → FAISS index
- retriever → decider → prompt builder → LLM
- answer + sources (or refusal)

Key principle:

LLM does NOT search.
FAISS searches.
LLM only writes answers from retrieved context.

⸻

## Project structure

```text
app/
  rag/
    ingest.py         # load documents
    chunking.py       # text -> chunks
    embedder.py       # chunks -> vectors
    store_faiss.py    # FAISS index
    retriever.py      # vector search + threshold
    decider.py        # should answer or refuse
    prompt.py         # strict prompt builder
    generator.py      # LLM call (JSON output)
    pipeline.py       # full RAG orchestration
    run_pipeline.py   # CLI entry

  utils/              # logging, helpers
  config.py           # env-based config
  main.py             # FastAPI app
  schemas.py          # API contracts

data/
  docs/               # your documents
  artifacts/
    index/            # FAISS index + chunks.jsonl + stats.json

eval/
  cases.jsonl         # evaluation cases
  run_eval.py         # eval runner

tests/
  test_retriever.py
  test_api.py
```

⸻

## Installation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Create .env from example:

cp .env.example .env

Set your API key and models inside .env.

⸻

**Add documents**

Put your files into:

data/docs/

Supported formats:
- .pdf
- .txt
- .md

⸻

**Build index**

python -m app.rag.ingest

Artifacts created:
- faiss.index
- chunks.jsonl
- stats.json

⸻

**Ask questions (CLI)**

python -m app.rag.run_pipeline "drawing title block requirements"

Example output:

{
  "answer": "...",
  "sources": [
    {
      "doc_id": "DOC-001",
      "chunk_id": 0,
      "page": 18,
      "score": 0.57
    }
  ]
}


⸻

**API usage**

Run server:

uvicorn app.main:app --port 8001

POST /ask

{
  "question": "What are title block requirements?"
}

Responses:
- 200 OK — answer or explicit refusal
- 422 — invalid input
- 500 — infrastructure failure only

Swagger available at /docs.

⸻

### Evaluation

Run RAG evaluation:

python -m eval.run_eval

Metrics:
- hit@k — correct document retrieved
- grounded_rate — sources present when expected
- refusal_quality — correct “I don’t know”
- latency & cost

Example summary:

hit@k: 0.93
grounded_rate: 0.93
refusal_quality: 1.00


⸻

### Design decisions
- FAISS + cosine similarity
- Threshold-based retrieval
- Explicit decider before generation
- Strict JSON-only LLM output
- No re-ranking / no hybrid search (yet)

⸻

### Limitations & future work
- No BM25 / hybrid retrieval
- No cross-encoder re-ranking
- No layout-aware PDF understanding
- No OCR

Planned:
- re-ranking
- hybrid search
- RAG v2 with multi-step reasoning

⸻

## Why this repo matters

**This is not a demo toy.**

This is a clean, inspectable, production-grade RAG baseline that:
- refuses when uncertain
- exposes metrics
- is testable
- is extendable
