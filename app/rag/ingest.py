# app/rag/ingest.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def iter_pdf_pages(path: Path) -> Iterator[Tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = (
            text.replace("\u00A0", " ")
                .replace("\u202F", " ")
                .replace("\x00", "")
        ).strip()
        yield i, text

def iter_documents(root: Path) -> Iterator[Tuple[Path, str, int | None, str]]:

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue

        doc_id = p.stem
        ext = p.suffix.lower()

        if ext == ".pdf":
            for page, text in iter_pdf_pages(p):
                yield p, doc_id, page, text

        elif ext in (".txt", ".md"):
            yield p, doc_id, None, load_text_file(p)

        else:
            continue