from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from openai import OpenAI


@dataclass(frozen=True)
class EmbedResult:
    vectors: List[List[float]]
    model: str
    usage: Optional[Dict[str, int]]
    latency_ms: int
    batches: int


class Embedder:

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        batch_size: int = 64,
        timeout_s: float = 30.0,
        retries: int = 2,
        logger=None,
        price_input_per_1m: float = 0.0,
        sleep_base_s: float = 0.5,
    ):
        self.client = client
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self.log = logger
        self.price_input_per_1m = float(price_input_per_1m or 0.0)
        self.sleep_base_s = float(sleep_base_s)

    def _cost_usd(self, input_tokens: int) -> float:
        if self.price_input_per_1m <= 0:
            return 0.0
        return round((input_tokens / 1_000_000) * self.price_input_per_1m, 6)

    @staticmethod
    def _normalize_texts(texts: List[str]) -> List[str]:
        out: List[str] = []
        for t in texts:
            s = (t or "").strip()
            out.append(s if s else " ")
        return out

    def embed_texts(self, texts: List[str]) -> EmbedResult:
        if not texts:
            return EmbedResult(
                vectors=[],
                model=self.model,
                usage={"input_tokens": 0},
                latency_ms=0,
                batches=0,
            )

        normalized = self._normalize_texts(texts)

        t0 = time.perf_counter()
        all_vecs: List[List[float]] = []
        total_input_tokens = 0
        batches = 0

        total_batches = (len(normalized) + self.batch_size - 1) // self.batch_size

        for b in range(total_batches):
            i = b * self.batch_size
            batch = normalized[i : i + self.batch_size]
            batches += 1

            last_err: Optional[Exception] = None

            for attempt in range(1, self.retries + 2):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        timeout=self.timeout_s,
                    )

                    vecs = [item.embedding for item in resp.data]

                    if len(vecs) != len(batch):
                        raise RuntimeError(
                            f"Embeddings length mismatch: got {len(vecs)} vectors for {len(batch)} texts"
                        )

                    all_vecs.extend(vecs)

                    usage_obj = getattr(resp, "usage", None)
                    if usage_obj is not None:
                        total_input_tokens += int(getattr(usage_obj, "prompt_tokens", 0) or 0)

                    if self.log:
                        self.log.info(
                            "EMBED batch ok | model=%s | batch=%s/%s | size=%s | attempt=%s",
                            self.model, b + 1, total_batches, len(batch), attempt
                        )
                    last_err = None
                    break

                except Exception as e:
                    last_err = e
                    if self.log:
                        self.log.info(
                            "EMBED batch err | model=%s | batch=%s/%s | attempt=%s | err=%s",
                            self.model, b + 1, total_batches, attempt, f"{type(e).__name__}: {e}"
                        )

                    if attempt <= self.retries:
                        time.sleep(self.sleep_base_s * attempt)
                        continue

            if last_err is not None:
                raise last_err

        # final sanity
        if len(all_vecs) != len(normalized):
            raise RuntimeError(
                f"Total vectors mismatch: {len(all_vecs)} != {len(normalized)}"
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = {"input_tokens": total_input_tokens}

        if self.log:
            self.log.info(
                "EMBED done | texts=%s | batches=%s | model=%s | latency_ms=%s | input_tokens=%s | cost_usd=%.6f",
                len(normalized),
                batches,
                self.model,
                latency_ms,
                total_input_tokens,
                self._cost_usd(total_input_tokens),
            )

        return EmbedResult(
            vectors=all_vecs,
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
            batches=batches,
        )