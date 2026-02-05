from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

# pip install faiss-cpu
import faiss


def l2_normalize(v: np.ndarray) -> np.ndarray:
    # v: (n, d)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def build_index(vectors: list[list[float]]) -> faiss.Index:
    arr = np.array(vectors, dtype="float32")
    arr = l2_normalize(arr)

    d = arr.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine == inner product when normalized
    index.add(arr)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def query(index: faiss.Index, query_vector: list[float], top_k: int = 5) -> tuple[list[float], list[int]]:
    q = np.array([query_vector], dtype="float32")
    q = l2_normalize(q)
    scores, ids = index.search(q, top_k)
    return scores[0].tolist(), ids[0].tolist()