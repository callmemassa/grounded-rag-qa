from __future__ import annotations

import re
from typing import List, Tuple, Optional

from app.rag.types import Chunk, ChunkMeta, SourceType


def clean_text(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)          # пробелы перед \n
    t = re.sub(r"\n{3,}", "\n\n", t)          # слишком много пустых строк
    t = re.sub(r"[ \t]{2,}", " ", t)          # многопробельность в строках
    return t.strip()


def chunk_text(
    text: str,
    *,
    doc_id: str,
    source: SourceType = "unknown",
    chunk_size_chars: int = 1000,
    overlap_chars: int = 160,
    path: Optional[str] = None,
    page: Optional[int] = None,
) -> List[Tuple[Chunk, ChunkMeta]]:

    if chunk_size_chars < 50:
        raise ValueError("chunk_size_chars too small")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be < chunk_size_chars")

    cleaned = clean_text(text)
    if not cleaned:
        return []

    out: List[Tuple[Chunk, ChunkMeta]] = []

    step = chunk_size_chars - overlap_chars
    n = len(cleaned)

    start = 0
    chunk_id = 0

    while start < n:
        end = min(n, start + chunk_size_chars)
        chunk_str = cleaned[start:end]


        chunk_str = chunk_str.strip()
        if chunk_str:
            left_trim = len(cleaned[start:end]) - len(cleaned[start:end].lstrip())
            right_trim = len(cleaned[start:end]) - len(cleaned[start:end].rstrip())

            real_start = start + left_trim
            real_end = end - right_trim

            c = Chunk(text=chunk_str)
            m = ChunkMeta(
                doc_id=doc_id,
                source=source,
                chunk_id=chunk_id,
                start_char=real_start,
                end_char=real_end,
                page=page,
                path=path,
            )
            out.append((c, m))
            chunk_id += 1

        if end >= n:
            break
        start += step

    return out