# app/rag/run_pipeline.py
from __future__ import annotations

import argparse
import json

from app.rag.pipeline import answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    parser.add_argument("question", type=str, help="Question to ask")
    args = parser.parse_args()

    res = answer(args.question)

    print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())